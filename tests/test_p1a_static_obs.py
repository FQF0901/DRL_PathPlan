"""P1-A（schema v4）静态障碍可观测性单测。

覆盖：
1. ``env.obs.static`` 几何口径（走廊 / 净距 / 相对车道 one-hot / range / 身后 / 回退）；
2. ``others`` 通道 static 段落位与 road_class 平移；
3. schema 版本/hash/manifest 记录（当前 v6）；
4. net 旧数据回退（28 维 others → 33 维重排 + 一次性告警）与新特征消费（梯度/形状/无 NaN）。

纯 NumPy + 假 env；用 ``FakeBuilding`` 顶替 ``env.obs.static.BaseBuilding`` 的 isinstance 检查，
不建 MetaDrive env（真实几何在 mini 冒烟里验证）。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

import env.obs.static as static_module
from env.obs.others import OTHERS_HEAD_DIM, STATIC_OFFSET
from env.obs.schema import STATIC_LAYOUT, schema_manifest
from env.obs.static import STATIC_DIM, STATIC_REL_BUCKETS, scan_static_blocker, static_features


# --------------------------------------------------------------------------- #
# 假 env / 车道 / 建筑
# --------------------------------------------------------------------------- #

class FakeLane:
    """直道（沿 +x，中心线 y=0）：local_coordinates 符号与 MetaDrive StraightLane 一致（lat 右正）。"""

    def __init__(self, *, width: float = 3.5, length: float = 300.0, index=("test", "road", 0)):
        self.width = float(width)
        self.length = float(length)
        self.index = tuple(index)
        self.speed_limit = 1000.0  # 未设置（speed_limit_value 回退 0）

    def local_coordinates(self, position):
        return float(position[0]), -float(position[1])

    def position(self, longitudinal, lateral):
        return np.array([float(longitudinal), -float(lateral), 0.0])

    def heading_theta_at(self, longitudinal):
        return 0.0


class FakeBuilding:
    """最小建筑接口：static 扫描只需要 position/LENGTH。"""

    def __init__(self, x: float, y: float, length: float = 10.0):
        self.position = np.array([x, y, 0.0], dtype=np.float64)
        self.LENGTH = float(length)


class FakeEgo:
    def __init__(self, x: float = 0.0, y: float = 0.0, lane=None):
        self.id = "ego"
        self.name = "ego"
        self.position = np.array([x, y, 0.0], dtype=np.float64)
        self.heading_theta = 0.0
        self.speed = 0.0
        self.LENGTH = 4.5
        self.lane = lane
        self.navigation = None


class FakeEngine:
    def __init__(self, objects=()):
        self.objects = {getattr(obj, "id", str(index)): obj for index, obj in enumerate(objects)}

    def get_objects(self):
        return self.objects


class FakeEnv:
    def __init__(self, ego: FakeEgo, objects=(), step: int = 0):
        self.engine = FakeEngine([ego, *objects])
        self.episode_step = int(step)
        self.config = {}
        self.prev_policy_action = None

    @property
    def agent(self):
        return self.engine.objects["ego"]


@pytest.fixture(autouse=True)
def _patch_base_building(monkeypatch):
    monkeypatch.setattr(static_module, "BaseBuilding", FakeBuilding)


# --------------------------------------------------------------------------- #
# 1) 几何口径
# --------------------------------------------------------------------------- #

def test_scan_same_lane_gap_and_bucket():
    """同车道岗亭：gap = s_obj - s_ego - 0.5·L（IDM 口径），相对车道 = 0。"""
    lane = FakeLane()
    ego = FakeEgo(x=5.0, lane=lane)
    booth = FakeBuilding(49.5, 0.0)  # s_obj=49.5, half=5 -> gap=39.5
    env = FakeEnv(ego, [booth])

    found = scan_static_blocker(env, ego)
    assert found is not None
    gap, bucket = found
    assert gap == pytest.approx(39.5)
    assert bucket == 0

    feats = static_features(env, ego)
    assert feats.shape == (STATIC_DIM, )
    assert feats[0] == 1.0
    assert feats[1] == pytest.approx(39.5 / 50.0)
    np.testing.assert_allclose(feats[2:], [0.0, 1.0, 0.0])  # buckets (-1,0,+1) -> same


def test_scan_adjacent_lane_one_hot_left_right():
    """相邻车道岗亭：+1 = 左侧（y 左正）、-1 = 右侧。"""
    lane = FakeLane()
    ego = FakeEgo(x=0.0, lane=lane)

    left = FakeBuilding(45.0, +3.5)
    found_left = scan_static_blocker(FakeEnv(ego, [left]), ego)
    assert found_left is not None and found_left[1] == 1
    np.testing.assert_allclose(static_features(FakeEnv(ego, [left]), ego)[2:], [0.0, 0.0, 1.0])

    right = FakeBuilding(45.0, -3.5)
    found_right = scan_static_blocker(FakeEnv(ego, [right]), ego)
    assert found_right is not None and found_right[1] == -1
    np.testing.assert_allclose(static_features(FakeEnv(ego, [right]), ego)[2:], [1.0, 0.0, 0.0])


def test_scan_picks_nearest_ahead_and_ignores_behind_or_far():
    """取走廊内前方最近者；身后/超 range/超走廊（±2 车道）不参与。"""
    lane = FakeLane()
    ego = FakeEgo(x=100.0, lane=lane)
    near = FakeBuilding(130.0, 3.5)  # gap=130-100-5=25
    far_same = FakeBuilding(200.0, 0.0)  # gap=95 > range
    behind = FakeBuilding(80.0, 0.0)  # gap=-25 < -half
    off_corridor = FakeBuilding(110.0, 7.0)  # |lat|=7.0 > 6.25（lane_span=1 走廊）
    env = FakeEnv(ego, [near, far_same, behind, off_corridor])

    found = scan_static_blocker(env, ego)
    assert found is not None
    gap, bucket = found
    assert gap == pytest.approx(25.0)
    assert bucket == 1  # 最近者在左邻车道

    # 无任何有效建筑 -> present=0 全 0
    empty = static_features(FakeEnv(ego, [behind, off_corridor]), ego)
    np.testing.assert_allclose(empty, np.zeros(STATIC_DIM, dtype=np.float32))


def test_scan_lane_span_widens_corridor_and_clips_bucket():
    """lane_span=2 时走廊覆盖 ±2 车道，超出 one-hot 桶范围归入最近桶。"""
    lane = FakeLane()
    ego = FakeEgo(x=0.0, lane=lane)
    two_left = FakeBuilding(40.0, 7.0)  # 2 条车道左侧
    env = FakeEnv(ego, [two_left])
    assert scan_static_blocker(env, ego, lane_span=1) is None
    found = scan_static_blocker(env, ego, lane_span=2)
    assert found is not None and found[1] == 2
    feats = static_features(env, ego, lane_span=2)
    assert feats[0] == 1.0
    np.testing.assert_allclose(feats[2:], [0.0, 0.0, 1.0])  # clip 到 +1


def test_scan_fallbacks_without_lane_or_engine():
    """无 lane / 无 engine → None / 全 0（观测通道不得崩）。"""
    ego = FakeEgo(x=0.0, lane=None)
    assert scan_static_blocker(FakeEnv(ego, []), ego) is None
    np.testing.assert_allclose(static_features(FakeEnv(ego, []), ego), 0.0)

    lane = FakeLane()
    ego2 = FakeEgo(x=0.0, lane=lane)
    env = FakeEnv(ego2, [])

    class BrokenEngine:
        def get_objects(self):
            raise KeyError("boom")

    env.engine = BrokenEngine()
    assert scan_static_blocker(env, ego2) is None


# --------------------------------------------------------------------------- #
# 2) others 通道落位
# --------------------------------------------------------------------------- #

def test_others_channel_static_segment_and_road_class_offset():
    """static 段 = others[16:21]；road_class 段整体平移到 [21,21+K)。"""
    from env.obs.others import OthersChannel

    lane = FakeLane()
    ego = FakeEgo(x=0.0, lane=lane)
    booth = FakeBuilding(39.5 + 5.0, 0.0)  # gap = 39.5
    env = FakeEnv(ego, [booth])

    channel = OthersChannel()
    feats, mask = channel.build(env, None)
    assert feats.shape == (1, channel.feature_dim)
    assert channel.feature_dim == 33
    assert mask[0] == 1.0
    np.testing.assert_allclose(feats[0, STATIC_OFFSET:OTHERS_HEAD_DIM], [1.0, 39.5 / 50.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(feats[0, OTHERS_HEAD_DIM:], 0.0)  # road_class 回退


# --------------------------------------------------------------------------- #
# 3) schema v4 / manifest / 指纹
# --------------------------------------------------------------------------- #

def test_schema_v6_manifest_records_static_layout():
    from env.obs import OBS_SCHEMA_VERSION, obs_fingerprint

    manifest = schema_manifest()
    assert OBS_SCHEMA_VERSION == 6
    assert manifest["schema_version"] == 6
    assert manifest["frame"]["others"]["shape"] == [1, 33]
    assert manifest["history"]["keys"]["others_hist"]["shape"] == [6, 1, 33]
    assert manifest["static_layout"]["feature_names"] == ["present", "gap_norm", "rel_left", "rel_same", "rel_right"]
    assert manifest["static_layout"]["rel_buckets"] == list(STATIC_REL_BUCKETS)
    segments = manifest["others_layout"]["segments"]
    assert segments["static"]["dims"] == [STATIC_OFFSET, OTHERS_HEAD_DIM]
    assert segments["road_class"]["dims"][0] == OTHERS_HEAD_DIM
    assert obs_fingerprint().startswith("v6-")
    assert STATIC_LAYOUT["scan_range_m"] == 50.0


# --------------------------------------------------------------------------- #
# 4) net：旧数据回退 + 新特征消费
# --------------------------------------------------------------------------- #

def _legacy_others_from_v4(tensor: torch.Tensor) -> torch.Tensor:
    """33 维 v4 others → 28 维旧布局（去掉 static 段、road_class 平移回 [16,28)）。"""
    return torch.cat([tensor[..., :16], tensor[..., 21:]], dim=-1)


def test_mem_from_obs_remaps_legacy_others_with_one_time_warning():
    from net import mem as mem_module
    from net.encoders import DEFAULT_OTHERS_DIM, OTHERS_STATIC_DIM
    from net.mem import mem_from_obs
    from net.param_probe import make_dummy_obs

    mem_module._FALLBACK_WARNED.discard("others_legacy_layout")
    obs = make_dummy_obs(batch=2)
    assert int(obs["others_hist"].shape[-1]) == DEFAULT_OTHERS_DIM == 33
    legacy_hist = _legacy_others_from_v4(obs["others_hist"])
    legacy_now = _legacy_others_from_v4(obs["others"])
    assert int(legacy_hist.shape[-1]) == 28

    with pytest.warns(RuntimeWarning, match="旧布局"):
        mem = mem_from_obs(
            {**obs, "others_hist": legacy_hist, "others": legacy_now},
            others_dim=DEFAULT_OTHERS_DIM,
            history_frames=6,
        )
    assert int(mem.others.shape[-1]) == 33
    # static 段补 0；其余两段逐位保留
    assert torch.equal(mem.others[..., 16:16 + OTHERS_STATIC_DIM], torch.zeros(2, 6, OTHERS_STATIC_DIM))
    assert torch.equal(mem.others[..., :16], obs["others_hist"].squeeze(2)[..., :16])
    assert torch.equal(mem.others[..., 16 + OTHERS_STATIC_DIM:], obs["others_hist"].squeeze(2)[..., 21:])

    # 第二次调用不再告警（一次性；key 已由首次调用写入）
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mem_from_obs(
            {**obs, "others_hist": legacy_hist, "others": legacy_now},
            others_dim=DEFAULT_OTHERS_DIM,
            history_frames=6,
        )
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


def test_mem_from_obs_new_layout_has_no_warning_and_rejects_bad_dim():
    from net import mem as mem_module
    from net.mem import mem_from_obs
    from net.param_probe import make_dummy_obs

    mem_module._FALLBACK_WARNED.discard("others_legacy_layout")
    obs = make_dummy_obs(batch=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mem = mem_from_obs(obs, others_dim=33, history_frames=6)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert int(mem.others.shape[-1]) == 33

    with pytest.raises(ValueError, match="others_dim"):
        mem_from_obs(obs, others_dim=8, history_frames=6)


def test_model_consumes_static_dims_and_backprop_reaches_them():
    """新特征进入令牌集合/融合路径：前向无 NaN、static 维可收到梯度。"""
    from net.model import DrivingModel
    from net.param_probe import make_dummy_obs

    torch.manual_seed(0)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=2).eval()
    obs = make_dummy_obs(batch=2)
    obs["others_hist"] = obs["others_hist"].clone().requires_grad_(True)

    out = model(obs, rollout=True, world_model=True)
    for key in ("action_mu", "action_logstd", "value", "traj_xy", "plan", "od_pred"):
        assert torch.isfinite(out[key]).all(), f"{key} 出现 NaN/Inf"

    loss = out["action_mu"].sum() + out["value"].sum() + out["traj_xy"].sum()
    loss.backward()
    grad = obs["others_hist"].grad
    assert grad is not None
    static_grad = grad[..., 16:21].abs().sum()
    assert float(static_grad) > 0.0, "static 段必须参与策略/plan 计算并可反向"


# --------------------------------------------------------------------------- #
# 5) 数据管线：BC 采集 / buffer 透传自动携带 static 段（others 维内）
# --------------------------------------------------------------------------- #

def test_data_pipeline_channels_carry_static_segment():
    from pipeline.buffer import DEFAULT_CHANNELS, _others_dim
    from pipeline.frames import HISTORY_CHANNELS
    from tools.collect_expert import CURRENT_CHANNELS, _channel_dim, _npz_schema_manifest

    assert "others" in CURRENT_CHANNELS
    assert _channel_dim("others") == 33
    assert _others_dim() == 33
    assert DEFAULT_CHANNELS["others"] == (1, 33)
    assert "others" in HISTORY_CHANNELS  # 6 帧历史重建包含 others

    manifest = _npz_schema_manifest(num_slots=16, frames=6, others_dim=33, label_count=8)
    assert manifest["others"]["shape"] == ["<N>", 1, 33]
    assert manifest["others_mask"]["shape"] == ["<N>", 1]
