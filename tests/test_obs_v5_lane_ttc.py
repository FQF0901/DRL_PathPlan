"""obs schema v5（结构迭代 A）单测。

覆盖：
1. LD 远场采样（offset {20,40,60,80}、当前车道 primary + 其余车道环填充、短车道降级）；
2. 当前车道块（``lane``：d_lat/航向误差/曲率/近场+远场点，已知场景数值校验）；
3. TTC 上下文 token（``ttc``：ttc_x/ttc_path 数值、责任槽位、cap、presence 门控、只读 OD）；
4. schema v5 manifest/指纹/回退、``tools/collect_expert.py`` 透传；
5. net：新 token 消费（梯度可达）与旧数据缺键回退（全 0 + mask=0 + 一次性告警）。

纯 NumPy + 假 env（不建 MetaDrive env；真实几何在 GPU 冒烟里验证）。
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch

import env.obs.od as od_module
from env.obs.builder import DEFAULT_CHANNELS, ObservationBuilder
from env.obs.lane import LANE_DIM, LANE_FEATURE_NAMES, LaneChannel
from env.obs.ld import LD_OFFSETS_M, LDChannel
from env.obs.od import ODChannel
from env.obs.schema import schema_manifest
from env.obs.ttc import TTC_DIM, TTC_FEATURE_NAMES, TTCChannel, slot_ttc

# --------------------------------------------------------------------------- #
# 假 env / 车道 / 车辆
# --------------------------------------------------------------------------- #


class FakeStraightLane:
    """直道（车道系：lat 正 = 车道方向右侧，与 MetaDrive StraightLane 一致）。"""

    def __init__(
        self,
        *,
        x0: float = 0.0,
        y0: float = 0.0,
        heading: float = 0.0,
        length: float = 300.0,
        width: float = 3.5,
        index=("road", "node", 0),
        speed_limit: float = 1000.0,
    ):
        self.x0, self.y0, self.heading = float(x0), float(y0), float(heading)
        self.length, self.width = float(length), float(width)
        self.index = tuple(index)
        self.speed_limit = float(speed_limit)
        self.line_types = ["solid", "broken"]

    @property
    def _c(self) -> float:
        return math.cos(self.heading)

    @property
    def _s(self) -> float:
        return math.sin(self.heading)

    def position(self, longitudinal, lateral=0.0):
        # direction = (c, s)；direction_lateral = (s, -c)
        dx = float(longitudinal) * self._c + float(lateral) * self._s
        dy = float(longitudinal) * self._s - float(lateral) * self._c
        return np.array([self.x0 + dx, self.y0 + dy, 0.0], dtype=np.float64)

    def heading_theta_at(self, longitudinal):
        return self.heading

    def local_coordinates(self, position):
        px = float(position[0]) - self.x0
        py = float(position[1]) - self.y0
        return float(px * self._c + py * self._s), float(px * self._s - py * self._c)


class FakeArcLane:
    """左转圆弧（起点在原点、初始航向 0，曲率 k = 1/R > 0）。"""

    def __init__(self, *, radius: float = 50.0, length: float = 100.0, width: float = 3.5,
                 index=("road", "node", 0), speed_limit: float = 1000.0):
        self.radius, self.length, self.width = float(radius), float(length), float(width)
        self.index = tuple(index)
        self.speed_limit = float(speed_limit)
        self.line_types = ["solid", "solid"]

    def _angle(self, s: float) -> float:
        return float(s) / self.radius

    def position(self, longitudinal, lateral=0.0):
        phi = self._angle(longitudinal)
        cx = self.radius * math.sin(phi)
        cy = self.radius * (1.0 - math.cos(phi))
        # 车道系 lateral basis = (sin φ, -cos φ)
        return np.array(
            [cx + float(lateral) * math.sin(phi), cy - float(lateral) * math.cos(phi), 0.0],
            dtype=np.float64,
        )

    def heading_theta_at(self, longitudinal):
        return self._angle(longitudinal)

    def local_coordinates(self, position):
        x, y = float(position[0]), float(position[1])
        phi = math.atan2(x, self.radius - y)
        s = phi * self.radius
        # 到中心线点的横向距离（正 = 车道方向右侧）
        cx = self.radius * math.sin(phi)
        cy = self.radius * (1.0 - math.cos(phi))
        lateral = (x - cx) * math.sin(phi) - (y - cy) * math.cos(phi)
        return float(s), float(lateral)


class FakeEgo:
    def __init__(self, *, x: float = 0.0, y: float = 0.0, heading: float = 0.0,
                 vx: float = 0.0, vy: float = 0.0, lane=None, ref_lanes=None):
        self.id, self.name = "ego", "ego"
        self.position = np.array([x, y, 0.0], dtype=np.float64)
        self.velocity = np.array([vx, vy, 0.0], dtype=np.float64)
        self.heading_theta = float(heading)
        self.speed = float(math.hypot(vx, vy))
        self.last_speed = self.speed
        self.steering = 0.0
        self.last_heading_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
        self.lane = lane
        if ref_lanes is not None:
            nav = type("_Nav", (), {"current_ref_lanes": list(ref_lanes), "next_ref_lanes": []})()
            self.navigation = nav
        else:
            self.navigation = None

    def convert_to_local_coordinates(self, point, origin):
        p = np.asarray(point, dtype=np.float64)[:2] - np.asarray(origin, dtype=np.float64)[:2]
        c, s = math.cos(self.heading_theta), math.sin(self.heading_theta)
        return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1], 0.0], dtype=np.float64)


class FakeVehicle:
    """OD/TTC 所需最小车辆接口。"""

    def __init__(self, name: str, x: float = 0.0, y: float = 0.0, vx: float = 0.0,
                 vy: float = 0.0, heading: float = 0.0):
        self.id, self.name = str(name), str(name)
        self.position = np.array([x, y, 0.0], dtype=np.float64)
        self.velocity = np.array([vx, vy, 0.0], dtype=np.float64)
        self.heading_theta = float(heading)
        self.LENGTH, self.WIDTH = 4.5, 1.8


class FakeEngine:
    def __init__(self, objects):
        self.objects = {obj.id: obj for obj in objects}

    def get_objects(self):
        return self.objects


class FakeRoadNetwork:
    def __init__(self, graph):
        self.graph = graph


class FakeMap:
    def __init__(self, graph):
        self.road_network = FakeRoadNetwork(graph)


class FakeEnv:
    def __init__(self, ego: FakeEgo, others=(), step: int = 0, map_graph=None):
        self._objects = [ego, *others]
        self.engine = FakeEngine(self._objects)
        self.episode_step = int(step)
        self.config = {}
        self.prev_policy_action = None
        self.current_map = FakeMap(map_graph) if map_graph is not None else None

    @property
    def agent(self):
        return self._objects[0]

    def add(self, obj) -> None:
        self._objects.append(obj)
        self.engine.objects[obj.id] = obj


@pytest.fixture(autouse=True)
def _patch_base_vehicle(monkeypatch):
    monkeypatch.setattr(od_module, "BaseVehicle", FakeVehicle)


# --------------------------------------------------------------------------- #
# 1) LD 远场
# --------------------------------------------------------------------------- #


def test_ld_offsets_are_far_field_and_current_lane_primary() -> None:
    """默认 offset {20,40,60,80}；当前车道占 4 primary 槽，其余车道按环填充。"""
    lane = FakeStraightLane(length=300.0, index=("road", "node", 0))
    sibling = FakeStraightLane(y0=-3.5, length=300.0, index=("road", "node", 1))
    env = FakeEnv(FakeEgo(lane=lane), map_graph={"road": {"node": [lane, sibling]}})

    channel = LDChannel()
    assert channel.offsets == tuple(LD_OFFSETS_M) == (20.0, 40.0, 60.0, 80.0)
    feats, mask = channel.build(env, None)

    for k, offset in enumerate(LD_OFFSETS_M):  # primary：当前车道 20/40/60/80
        assert mask[k] == 1.0
        assert feats[k, 0] == pytest.approx(offset)
        assert feats[k, 1] == pytest.approx(0.0)
    for k, offset in enumerate(LD_OFFSETS_M):  # secondary：相邻车道 20/40/60/80
        assert mask[4 + k] == 1.0
        assert feats[4 + k, 0] == pytest.approx(offset)
        assert feats[4 + k, 1] == pytest.approx(-3.5)
    assert mask[8:].sum() == 0.0
    # v5：当前车道不再有 <20 m 的近场采样（由 lane 通道补偿）
    assert float(feats[:4, 0].min()) >= 20.0 - 1e-6


def test_ld_short_lane_drops_far_offsets_and_ring_fills_others() -> None:
    """当前车道 <60 m：60/80 槽丢弃（mask=0）；相邻车道继续按环填充。"""
    lane = FakeStraightLane(length=50.0, index=("road", "node", 0))
    sibling = FakeStraightLane(y0=-3.5, length=300.0, index=("road", "node", 1))
    env = FakeEnv(FakeEgo(lane=lane), map_graph={"road": {"node": [lane, sibling]}})
    feats, mask = LDChannel().build(env, None)

    assert mask[0] == 1.0 and feats[0, 0] == pytest.approx(20.0)  # 当前 20
    assert mask[1] == 1.0 and feats[1, 0] == pytest.approx(40.0)  # 当前 40
    # 后续槽：相邻车道 20/40/60/80（环优先；当前车道 60/80 已丢弃）
    assert [float(mask[k]) for k in (2, 3, 4, 5)] == [1.0] * 4
    assert [float(feats[k, 0]) for k in (2, 3, 4, 5)] == pytest.approx([20.0, 40.0, 60.0, 80.0])
    assert [float(feats[k, 1]) for k in (2, 3, 4, 5)] == pytest.approx([-3.5] * 4)
    assert mask[6:].sum() == 0.0


# --------------------------------------------------------------------------- #
# 2) 当前车道块
# --------------------------------------------------------------------------- #


def test_lane_channel_known_straight_scenario() -> None:
    """直道已知场景：d_lat/宽度/近场点数值校验（d_lat = -near_dy 符号锚定）。"""
    lane = FakeStraightLane(length=300.0, width=3.5)
    ego = FakeEgo(x=10.0, y=0.7, heading=0.0, lane=lane)  # ego 在中心线左侧 0.7 m
    feats, mask = LaneChannel().build(FakeEnv(ego), None)
    assert feats.shape == (1, LANE_DIM) and mask[0] == 1.0
    f = feats[0]

    assert f[0] == pytest.approx(0.7)  # d_lat 正 = ego 在中心线左侧
    assert f[1] == pytest.approx(0.0)
    assert f[2] == pytest.approx(3.5)
    assert f[3] == pytest.approx(0.0)  # 直道曲率 0
    # 中心线点在 ego 系：x = s0+offset - ego.x；y = -0.7
    assert f[5] == pytest.approx(5.0) and f[6] == pytest.approx(-0.7)
    assert f[7] == pytest.approx(0.0)
    assert f[8] == pytest.approx(15.0) and f[9] == pytest.approx(-0.7)
    assert f[11] == pytest.approx(60.0) and f[12] == pytest.approx(-0.7)
    assert list(f[14:17]) == [1.0, 1.0, 1.0]
    assert f[0] == pytest.approx(-f[6])  # 符号锚定：d_lat = -near_dy（直道/对齐航向）


def test_lane_channel_heading_error_sign_and_rotation() -> None:
    """航向误差 = 车道 - ego（ego 朝左偏 → 负）；近场点在 ego 系旋转。"""
    lane = FakeStraightLane()
    ego = FakeEgo(x=0.0, y=0.0, heading=0.1, lane=lane)
    f = LaneChannel().build(FakeEnv(ego), None)[0][0]

    assert f[1] == pytest.approx(-0.1)  # ego 航向左偏 0.1 rad
    c, s = math.cos(0.1), math.sin(0.1)
    assert f[5] == pytest.approx(5.0 * c) and f[6] == pytest.approx(-5.0 * s)
    assert f[7] == pytest.approx(-0.1)


def test_lane_channel_arc_curvature_and_short_lane_validity() -> None:
    """圆弧车道曲率（左转正）；短车道 far_valid=0 且远场字段清零。"""
    arc = FakeArcLane(radius=50.0, length=100.0)
    f = LaneChannel().build(FakeEnv(FakeEgo(lane=arc)), None)[0][0]
    assert f[3] == pytest.approx(1.0 / 50.0, abs=1e-3)  # 左转（逆时针）为正
    assert list(f[14:17]) == [1.0, 1.0, 1.0]

    short = FakeStraightLane(length=20.0)
    f2 = LaneChannel().build(FakeEnv(FakeEgo(lane=short)), None)[0][0]
    assert list(f2[14:17]) == [1.0, 1.0, 0.0]
    assert list(f2[11:14]) == [0.0, 0.0, 0.0]  # far 点超车道末端 → 清零


def test_lane_channel_fallbacks() -> None:
    """无 ego / 无 lane / 无导航参考 → 全 0 + mask=0（不崩）。"""
    ch = LaneChannel()
    feats, mask = ch.build(FakeEnv(FakeEgo(lane=None)), None)
    assert mask[0] == 0.0 and np.allclose(feats, 0.0)

    ref = FakeStraightLane()
    ego = FakeEgo(lane=None, ref_lanes=[ref])  # 退化为 navigation.current_ref_lanes[0]
    feats2, mask2 = ch.build(FakeEnv(ego), None)
    assert mask2[0] == 1.0 and feats2[0, 2] == pytest.approx(3.5)

    class _NoEgo:
        engine = None
        episode_step = 0

    feats3, mask3 = ch.build(_NoEgo(), None)
    assert mask3[0] == 0.0 and np.allclose(feats3, 0.0)


# --------------------------------------------------------------------------- #
# 3) TTC
# --------------------------------------------------------------------------- #


def test_slot_ttc_formula() -> None:
    """ttc_x 与 ttc_path（进入碰撞圆）解析值。"""
    tx, tp = slot_ttc(10.0, 0.0, -2.0, 0.0)  # 对头接近：x 口径 5 s；path 口径先进入 2.5 m 圆
    assert tx == pytest.approx(5.0) and tp == pytest.approx(3.75)
    tx, tp = slot_ttc(0.0, 6.0, 0.0, -5.0)  # 纯横向接近：x 口径漏检，path 口径 0.7 s
    assert tx == math.inf and tp == pytest.approx(0.7)
    assert slot_ttc(1.0, 0.0, -2.0, 0.0)[1] == 0.0  # 已在圆内且接近
    assert slot_ttc(1.0, 0.0, 2.0, 0.0)[1] == math.inf  # 已在圆内但远离
    assert slot_ttc(10.0, 0.0, 1.0, 0.0)[0] == math.inf  # 不接近


def _ttc_stack(objects, *, step: int = 0):
    ego = FakeEgo(vx=10.0)  # 自车世界系 +x 10 m/s
    env = FakeEnv(ego, objects, step=step)
    od = ODChannel()
    ttc = TTCChannel(od_channel=od)
    return env, od, ttc


def test_ttc_numeric_min_counts_and_responsible_slot() -> None:
    """构造相遇场景：min-TTC/<3 s 计数/责任槽位位置速度数值校验。"""
    ahead = FakeVehicle("ahead", 10.0, 0.0, vx=5.0)  # rel vx = -5 → ttc_x = 2.0 s
    cross = FakeVehicle("cross", 0.0, 6.0, vx=10.0, vy=-5.0)  # 横向接近 → ttc_path = 0.7 s
    env, od, ttc = _ttc_stack([ahead, cross])

    feats, mask = ttc.build(env, None)
    assert feats.shape == (1, TTC_DIM) and mask[0] == 1.0
    f = feats[0]
    # OD 紧迫度排序：ahead（ttc_x=2）先占槽 0，cross（ttc_x=inf→cap）槽 1
    ids = od.companions()["od_id"]
    assert ids[0] != -1 and ids[1] != -1
    assert f[0] == pytest.approx(2.0)  # min_ttc_x（ahead）
    assert f[1] == pytest.approx(0.7)  # min_ttc_path（cross 横向进入 2.5 m 圆 = 0.7 s）
    assert f[2] == pytest.approx(1.0) and f[3] == pytest.approx(2.0)  # x 口径 1 个、path 口径 2 个 <3 s
    assert f[4] == pytest.approx(1.0)  # 责任槽位 = cross（0.7 < 2.0）
    assert f[5] == pytest.approx(0.0) and f[6] == pytest.approx(6.0)
    assert f[7] == pytest.approx(0.0) and f[8] == pytest.approx(-5.0)
    assert f[9] == pytest.approx(0.7)
    assert f[10] == pytest.approx(1.0)  # path 口径获胜
    assert f[11] == pytest.approx(1.0)


def test_ttc_cap_and_no_risk_semantics() -> None:
    """远距离接近 → min 截断 cap；无对象 → cap + resp=-1 + valid=0。"""
    far = FakeVehicle("far", 100.0, 0.0, vx=9.0)  # rel vx=-1 → ttc_x=100 s
    env, _, ttc = _ttc_stack([far])
    f = ttc.build(env, None)[0][0]
    assert f[0] == pytest.approx(5.0)  # clamp 到 cap
    assert f[2] == pytest.approx(0.0)  # 100 s 不算 <3 s
    assert f[11] == pytest.approx(1.0)  # 有有限 TTC（只是远）

    env2, _, ttc2 = _ttc_stack([])
    f2 = ttc2.build(env2, None)[0][0]
    assert f2[0] == pytest.approx(5.0) and f2[1] == pytest.approx(5.0)
    assert f2[4] == pytest.approx(-1.0) and f2[11] == pytest.approx(0.0)


def test_ttc_ignores_stale_slots_and_does_not_mutate_od() -> None:
    """presence=0 的陈旧槽不参与；ttc 只读 OD（同 step 槽位/特征不变）。"""
    target = FakeVehicle("t", 10.0, 0.0, vx=5.0)
    env, od, ttc = _ttc_stack([target], step=0)
    od_before, mask_before = od.build(env, None)
    ids_before = od.companions()["od_id"].copy()
    ttc.build(env, None)
    od_after, mask_after = od.build(env, None)  # 同 step：OD 幂等渲染
    np.testing.assert_allclose(od_before, od_after)
    np.testing.assert_array_equal(mask_before, mask_after)
    np.testing.assert_array_equal(ids_before, od.companions()["od_id"])

    target.position = np.array([500.0, 0.0, 0.0])  # 出盒未释放（presence=0）
    env.episode_step = 5
    f = ttc.build(env, None)[0][0]
    assert f[11] == pytest.approx(0.0) and f[0] == pytest.approx(5.0)

    env.episode_step = 15  # 超过 1.0 s → 释放（mask=0）
    f2 = ttc.build(env, None)[0][0]
    assert f2[11] == pytest.approx(0.0)


def test_ttc_mask_zero_without_od_reference() -> None:
    """未接 OD 通道 → mask=0（不静默给"无风险"）。"""
    env, _, _ = _ttc_stack([])
    feats, mask = TTCChannel(od_channel=None).build(env, None)
    assert mask[0] == 0.0 and np.allclose(feats, 0.0)


# --------------------------------------------------------------------------- #
# 4) schema v5 / builder / collect_expert
# --------------------------------------------------------------------------- #


def test_schema_v5_manifest_and_fingerprint() -> None:
    from env.obs import OBS_SCHEMA_VERSION, obs_fingerprint

    manifest = schema_manifest()
    assert OBS_SCHEMA_VERSION == 5 and manifest["schema_version"] == 5
    assert manifest["frame"]["lane"]["shape"] == [1, LANE_DIM]
    assert manifest["frame"]["lane"]["feature_names"] == list(LANE_FEATURE_NAMES)
    assert manifest["frame"]["ttc"]["shape"] == [1, TTC_DIM]
    assert manifest["frame"]["ttc"]["feature_names"] == list(TTC_FEATURE_NAMES)
    assert manifest["ld_layout"]["offsets_m"] == list(LD_OFFSETS_M)
    assert manifest["lane_layout"]["sample_offsets_m"] == [5.0, 15.0, 60.0]
    assert manifest["ttc_layout"]["collision_radius_m"] == 2.5
    assert "lane" in DEFAULT_CHANNELS and "ttc" in DEFAULT_CHANNELS
    assert obs_fingerprint().startswith("v5-")


def test_builder_emits_lane_and_ttc_channels() -> None:
    """默认 builder 输出 lane/ttc（形状/掩码）且不影响既有通道形状。"""
    lane = FakeStraightLane()
    sibling = FakeStraightLane(y0=-3.5, index=("road", "node", 1))
    obj = FakeVehicle("obj", 20.0, 0.0, vx=5.0)
    ego = FakeEgo(vx=10.0, lane=lane)
    env = FakeEnv(ego, [obj], map_graph={"road": {"node": [lane, sibling]}})
    builder = ObservationBuilder(
        {"channels": ["ego", "od", "ld", "lane", "nav", "signal", "others", "ttc"]}
    )
    obs = builder.build(env, None)

    assert obs["lane"].shape == (1, LANE_DIM) and obs["lane_mask"][0] == 1.0
    assert obs["ttc"].shape == (1, TTC_DIM) and obs["ttc_mask"][0] == 1.0
    assert obs["ld"].shape == (16, 7) and obs["ld_mask"].shape == (16, )
    assert obs["ttc"][0, 0] == pytest.approx(4.0)  # dx=20, rel vx=5-10=-5 → ttc_x=4 s
    assert builder.feature_spec["lane"] == (1, LANE_DIM)
    assert builder.feature_spec["ttc"] == (1, TTC_DIM)


def test_collect_expert_passthrough_channels() -> None:
    from tools.collect_expert import CURRENT_CHANNELS, _channel_dim, _channel_slots, _npz_schema_manifest

    assert {"lane", "ttc"} <= set(CURRENT_CHANNELS)
    assert _channel_dim("lane") == LANE_DIM and _channel_dim("ttc") == TTC_DIM
    assert _channel_slots("lane", 16) == 1 and _channel_slots("ttc", 16) == 1
    manifest = _npz_schema_manifest(num_slots=16, frames=6, others_dim=33, label_count=8)
    assert manifest["lane"]["shape"] == ["<N>", 1, LANE_DIM]
    assert manifest["lane_mask"]["shape"] == ["<N>", 1]
    assert manifest["ttc"]["shape"] == ["<N>", 1, TTC_DIM]
    assert manifest["ttc_mask"]["shape"] == ["<N>", 1]


# --------------------------------------------------------------------------- #
# 5) net：消费 + 旧数据回退
# --------------------------------------------------------------------------- #


def test_model_consumes_lane_ttc_tokens_with_gradients() -> None:
    from net.model import DrivingModel
    from tests.test_net_shapes import make_obs

    torch.manual_seed(0)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=2).eval()
    obs = make_obs(batch=2, with_struct_context=True)
    obs["lane"] = obs["lane"].clone().requires_grad_(True)
    obs["ttc"] = obs["ttc"].clone().requires_grad_(True)

    out = model(obs, rollout=True, world_model=True)
    for key in ("action_mu", "action_logstd", "value", "traj_xy", "plan", "od_pred"):
        assert torch.isfinite(out[key]).all(), f"{key} 出现 NaN/Inf"
    (out["action_mu"].sum() + out["value"].sum()).backward()
    assert obs["lane"].grad is not None and float(obs["lane"].grad.abs().sum()) > 0.0
    assert obs["ttc"].grad is not None and float(obs["ttc"].grad.abs().sum()) > 0.0

    encoded = model.encode(make_obs(batch=2, with_struct_context=True))
    assert tuple(encoded["tokens"].shape) == (2, 39, 32)
    # 令牌序：[OD16, LD16, lane, others, ego, nav, ttc, signal, latent]
    assert encoded["key_mask"][:, 32].all() and encoded["key_mask"][:, 36].all()


def test_model_falls_back_on_missing_lane_ttc_once() -> None:
    from net import mem as mem_module
    from net.model import DrivingModel
    from tests.test_net_shapes import make_obs

    for key in ("context_missing_lane", "context_missing_ttc"):
        mem_module._FALLBACK_WARNED.discard(key)
    torch.manual_seed(0)
    model = DrivingModel(hidden=32, num_experts=2, expert_hidden=32, wm_steps=2).eval()
    obs = make_obs(batch=2, with_struct_context=False)

    with pytest.warns(RuntimeWarning, match="lane"):
        out = model(obs, rollout=False, world_model=False)
    assert torch.isfinite(out["action_mu"]).all()
    encoded = model.encode(obs)
    assert encoded["lane_mask"].abs().sum() == 0.0 and encoded["ttc_mask"].abs().sum() == 0.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model(obs, rollout=False, world_model=False)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]
