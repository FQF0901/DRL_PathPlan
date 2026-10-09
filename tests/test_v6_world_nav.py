"""v6 A4（env lane）测试：obs 世界系键 / nav 逐步重建 helper / teacher forcing 同步 / buffer 透传。

不建 MetaDrive env（纯 NumPy/torch + 假导航/假 lane）；live env 的逐点一致性由
``tools/measure`` 类脚本与 P1 验收脚本另行覆盖。覆盖：

1. ``env.obs.world``：``ego_world``/``route_world`` 键存在、形状、mask、世界系值正确；
2. ``net.mem.nav_features_from_world``：checkpoint/命令/route_completion 与路线几何一致，
   **位移后 nav 不再陈旧**（对照探针：t0 vs 位移后必须不同且按几何前进）；
3. ``rebuild_nav_from_world``：**纯函数**——返回 nav 特征、**不原地改写 mem**；others nav
   子向量的同步由独立 helper ``sync_others_nav_dims`` 完成（返回新 MemBank，赋值式、可反向）；
   旧键缺失 → 回退 t0 + 一次性 RuntimeWarning；
4. ``wm_teacher_forcing_predictions``：逐步重建 nav（与 rollout 同一步进语义）+ others 同步
   （经 ``sync_others_nav_dims`` 赋值式替换）；
5. ``RolloutBuffer`` / ``_assemble_obs_batch``：collect→update 双路径携带新键。
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

import net.mem as mem_module
from env.obs.builder import DEFAULT_CHANNELS, ObservationBuilder
from env.obs.world import ROUTE_WORLD_MAX_POINTS, route_world_polyline
from net.mem import (
    MemBank,
    advance_pose_world,
    mem_from_obs,
    nav_features_from_world,
    rebuild_nav_from_world,
    sync_others_nav_dims,
)
from pipeline.trainer import PPOConfig, PPOTrainer, _SmokeModel, wm_teacher_forcing_predictions

# --------------------------------------------------------------------------- #
# 假导航：A->B->C->D（BC 直行 40 m；CD 左转 30°），单车道（later_middle=0）
# --------------------------------------------------------------------------- #

LANE_AB = SimpleNamespace(start=np.array([0.0, 0.0]), heading=0.0, length=10.0, width=3.5)
LANE_BC = SimpleNamespace(start=np.array([10.0, 0.0]), heading=0.0, length=40.0, width=3.5)
LANE_CD = SimpleNamespace(
    start=np.array([50.0, 0.0]), heading=math.radians(30.0), length=20.0, width=3.5
)


def _lane_position(lane: SimpleNamespace, long: float, lat: float) -> np.ndarray:
    c, s = math.cos(lane.heading), math.sin(lane.heading)
    return np.array(
        [lane.start[0] + long * c - lat * s, lane.start[1] + long * s + lat * c, 0.0],
        dtype=np.float32,
    )


for _lane in (LANE_AB, LANE_BC, LANE_CD):
    _lane.position = lambda long, lat, _lane=_lane: _lane_position(_lane, long, lat)  # type: ignore[attr-defined]

FAKE_GRAPH = {"A": {"B": [LANE_AB]}, "B": {"C": [LANE_BC]}, "C": {"D": [LANE_CD]}}
FAKE_CHECKPOINTS = ["A", "B", "C", "D"]
ROUTE = np.array(
    [
        [0.0, 0.0],
        [10.0, 0.0],
        [50.0, 0.0],
        [50.0 + 20.0 * math.cos(math.radians(30.0)), 20.0 * math.sin(math.radians(30.0))],
    ],
    dtype=np.float32,
)


class _FakeEgo:
    def __init__(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0):
        self.position = np.array([x, y, 0.0], dtype=np.float64)
        self.heading_theta = float(theta)
        self.navigation = SimpleNamespace(
            checkpoints=list(FAKE_CHECKPOINTS),
            map=SimpleNamespace(road_network=SimpleNamespace(graph=FAKE_GRAPH)),
        )

    def convert_to_local_coordinates(self, point, origin):
        p = np.asarray(point, dtype=np.float64)[:2] - np.asarray(origin, dtype=np.float64)[:2]
        c, s = math.cos(self.heading_theta), math.sin(self.heading_theta)
        return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1], 0.0])


class _FakeEnv:
    def __init__(self, ego: _FakeEgo, step: int = 0):
        self.agent = ego
        self.episode_step = int(step)


def _pose(x: float, y: float, theta: float) -> torch.Tensor:
    return torch.tensor([[x, y, theta]], dtype=torch.float32)


def _route_t(batch: int = 1) -> torch.Tensor:
    return torch.as_tensor(ROUTE, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1).contiguous()


# --------------------------------------------------------------------------- #
# 1. env 侧：obs 世界系键
# --------------------------------------------------------------------------- #

def test_route_world_polyline_from_fake_nav():
    """折线 = 首段起点 + 各路段终点（road 中心），与 checkpoint 同口径。"""
    nav = SimpleNamespace(
        checkpoints=list(FAKE_CHECKPOINTS),
        map=SimpleNamespace(road_network=SimpleNamespace(graph=FAKE_GRAPH)),
    )
    polyline = route_world_polyline(nav)
    assert polyline.shape == (4, 2)
    np.testing.assert_allclose(polyline, ROUTE, atol=1e-5)


def test_builder_emits_ego_world_and_route_world_with_mask():
    """builder 默认通道含新键；形状/掩码/世界系值正确；补位点 = 末点重复 + mask 0。"""
    assert "ego_world" in DEFAULT_CHANNELS and "route_world" in DEFAULT_CHANNELS
    builder = ObservationBuilder({"channels": ["ego_world", "route_world"]})
    env = _FakeEnv(_FakeEgo(6.43, 0.0, 0.0), step=0)
    obs = builder.build(env, None)

    assert obs["ego_world"].shape == (1, 3)
    np.testing.assert_allclose(obs["ego_world"][0], [6.43, 0.0, 0.0], atol=1e-6)
    assert obs["ego_world_mask"][0] == 1.0

    route = obs["route_world"]
    assert route.shape == (ROUTE_WORLD_MAX_POINTS, 2)
    mask = obs["route_world_mask"]
    assert mask.sum() == 4.0
    np.testing.assert_allclose(route[:4], ROUTE, atol=1e-5)
    # 补位：末点重复、mask=0（下游必须按 mask 过滤）
    np.testing.assert_allclose(route[4:], np.broadcast_to(ROUTE[-1], route[4:].shape), atol=1e-5)
    assert mask[4:].sum() == 0.0


def test_route_world_unavailable_returns_zero_and_mask():
    """无 navigation（或非 NodeNetworkNavigation）→ 全 0 + mask 0，不抛异常。"""
    ego = _FakeEgo(0.0, 0.0, 0.0)
    ego.navigation = None
    obs = ObservationBuilder({"channels": ["ego_world", "route_world"]}).build(_FakeEnv(ego), None)
    assert obs["route_world_mask"].sum() == 0.0
    assert obs["route_world"].sum() == 0.0
    assert obs["ego_world_mask"][0] == 1.0


# --------------------------------------------------------------------------- #
# 2. nav_features_from_world：几何正确 + 位移后不陈旧（对照探针）
# --------------------------------------------------------------------------- #

def test_nav_features_at_t0_matches_route_geometry():
    feats, mask = nav_features_from_world(_pose(6.43, 0.0, 0.0), _route_t())
    assert mask.item() == 1.0
    # checkpoint 自车系：下一顶点 (10,0) / 下下顶点 (50,0)
    np.testing.assert_allclose(feats[0, :4].numpy(), [3.57, 0.0, 43.57, 0.0], atol=1e-4)
    # 命令 forward（BC 与 CD 的航向差 30° > 10°，但当前段是 AB->BC 直行）
    np.testing.assert_allclose(feats[0, 4:10].numpy(), [1, 0, 0, 0, 0, 0], atol=1e-6)
    # route_completion = 6.43 / (10+40+20)
    assert float(feats[0, 10]) == pytest.approx(6.43 / 70.0, abs=1e-5)


def test_nav_not_stale_after_displacement_control_probe():
    """位移 30 m 后：checkpoint 前进、route_completion 增加、命令在分叉前切换（对照探针）。"""
    t0_feats, _ = nav_features_from_world(_pose(6.43, 0.0, 0.0), _route_t())
    moved_feats, _ = nav_features_from_world(_pose(36.43, 0.0, 0.0), _route_t())
    assert not torch.allclose(t0_feats, moved_feats), "位移后 nav 不得等于 t0（陈旧）"
    # 下一 checkpoint 从 (10,0) 前进到 (50,0)
    np.testing.assert_allclose(t0_feats[0, :2].numpy(), [3.57, 0.0], atol=1e-4)
    np.testing.assert_allclose(moved_feats[0, :2].numpy(), [13.57, 0.0], atol=1e-4)
    assert float(moved_feats[0, 10]) == pytest.approx(36.43 / 70.0, abs=1e-5)
    # 命令：进入分叉段后 AB->BC 直行 → BC->CD 左转
    np.testing.assert_allclose(t0_feats[0, 4:7].numpy(), [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(moved_feats[0, 4:7].numpy(), [0, 1, 0], atol=1e-6)


def test_route_completion_monotone_and_final_segment_forward():
    poses = [(0.0, 0.0), (10.0, 0.0), (30.0, 0.0), (55.0, 0.0)]
    rc = []
    for x, y in poses:
        feats, _ = nav_features_from_world(_pose(x, y, 0.0), _route_t())
        rc.append(float(feats[0, 10]))
    assert rc == sorted(rc), f"route_completion 必须单调不减：{rc}"
    # 末段（CD 上）：下一/下下 checkpoint 同段 → 命令 forward（上游 next_ref_lanes=None 口径）
    last, _ = nav_features_from_world(_pose(60.0, 4.0, math.radians(30.0)), _route_t())
    np.testing.assert_allclose(last[0, 4:7].numpy(), [1, 0, 0], atol=1e-6)


def test_advance_pose_world_matches_model_geometry():
    """与 net.model.arc_step/compose_pose 逐位同口径（教师强制/rollout 共用步进语义）。"""
    from net.model import arc_step, compose_pose

    pose = _pose(3.0, -2.0, 0.4)
    ds = torch.tensor([5.0])
    dtheta = torch.tensor([0.3])
    dx, dy = arc_step(ds, dtheta)
    expected = compose_pose(pose, dx, dy, dtheta)
    actual = advance_pose_world(pose, ds, dtheta)
    assert torch.allclose(actual, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# 3. rebuild_nav_from_world：mem 同步 + 回退
# --------------------------------------------------------------------------- #

def _mem_with_others(nav_features: torch.Tensor, batch: int = 1) -> MemBank:
    others = torch.zeros(batch, 6, 28)
    others[:, -1, :11] = nav_features
    zeros = torch.zeros(batch, 6, 1)
    return MemBank(
        ego=torch.zeros(batch, 6, 8),
        ego_mask=torch.ones(batch, 6),
        ego_valid=torch.ones(batch, 6),
        od=torch.zeros(batch, 6, 16, 9),
        od_mask=torch.zeros(batch, 6, 16),
        od_valid=torch.ones(batch, 6),
        od_id=torch.full((batch, 6, 16), -1, dtype=torch.long),
        od_presence=torch.zeros(batch, 6, 16),
        ld=torch.zeros(batch, 6, 16, 7),
        ld_mask=torch.zeros(batch, 6, 16),
        ld_valid=torch.ones(batch, 6),
        others=others,
        others_mask=torch.ones(batch, 6),
        others_valid=torch.ones(batch, 6),
        frames=6,
    )


def test_rebuild_nav_is_pure_and_sync_returns_new_mem():
    """rebuild 不原地写 mem；sync_others_nav_dims 返回新 MemBank（赋值式替换，可反向）。"""
    t0, _ = nav_features_from_world(_pose(6.43, 0.0, 0.0), _route_t())
    mem = _mem_with_others(t0.clone())
    feats, mask = rebuild_nav_from_world(mem, _pose(36.43, 0.0, 0.0), _route_t())
    assert mask.item() == 1.0
    assert torch.allclose(mem.others[:, -1, :11], t0), "rebuild 必须纯函数、不改 mem.others"
    synced = sync_others_nav_dims(mem, feats)
    assert torch.allclose(synced.others[:, -1, :11], feats), "sync 后新 mem 的 nav 维 = 重建值"
    assert not torch.allclose(synced.others[:, -1, :11], t0), "others nav 不得停留在 t0"
    assert torch.allclose(mem.others[:, -1, :11], t0), "原 mem 仍不被修改（无 in-place）"
    assert synced.others is not mem.others
    # 历史帧与其他字段保持共享/不变
    assert torch.equal(synced.others[:, :-1], mem.others[:, :-1])
    assert synced.ego is mem.ego


def test_rebuild_nav_falls_back_when_world_keys_missing(monkeypatch):
    monkeypatch.setattr(mem_module, "_FALLBACK_WARNED", set())
    t0, _ = nav_features_from_world(_pose(6.43, 0.0, 0.0), _route_t())
    mem = _mem_with_others(t0.clone())
    with pytest.warns(RuntimeWarning, match="ego_world/route_world"):
        feats, mask = rebuild_nav_from_world(mem, None, None)
    assert torch.allclose(feats, t0), "旧数据回退：返回 t0 冻结的 nav 子向量"
    assert mask.item() == 1.0
    # 一次性告警：第二次不再重复
    import warnings as _warnings

    with _warnings.catch_warnings(record=True) as recorded:
        _warnings.simplefilter("always")
        feats2, _ = rebuild_nav_from_world(mem, None, None)
    assert not [item for item in recorded if issubclass(item.category, RuntimeWarning)]
    assert torch.allclose(feats2, feats)


def test_rebuild_nav_degenerate_route_falls_back(monkeypatch):
    """全 0 路线（无有效线段）→ 回退 + 告警（不产生 NaN/错误 checkpoint）。"""
    monkeypatch.setattr(mem_module, "_FALLBACK_WARNED", set())
    t0, _ = nav_features_from_world(_pose(6.43, 0.0, 0.0), _route_t())
    mem = _mem_with_others(t0.clone())
    with pytest.warns(RuntimeWarning, match="route_world 不可用"):
        feats, _ = rebuild_nav_from_world(mem, _pose(0.0, 0.0, 0.0), torch.zeros(1, 8, 2))
    assert torch.allclose(feats, t0)


# --------------------------------------------------------------------------- #
# 4. teacher forcing：逐步重建 nav（trainer 版，与 rollout 同一步进语义）
# --------------------------------------------------------------------------- #

H_STUB = 8
_B = 2


def _teacher_obs(with_world: bool = True) -> Dict[str, torch.Tensor]:
    obs = {
        "od_hist": torch.zeros(_B, 6, 16, 9),
        "od_hist_mask": torch.zeros(_B, 6, 16),
        "ld_hist": torch.zeros(_B, 6, 16, 7),
        "ld_hist_mask": torch.zeros(_B, 6, 16),
        "others_hist": torch.zeros(_B, 6, 28),
        "others_hist_mask": torch.ones(_B, 6),
        "hist_valid": torch.ones(_B, 6),
        "ego": torch.zeros(_B, 8),
        "nav": torch.zeros(_B, 11),
        "nav_mask": torch.ones(_B, 1),
    }
    obs["nav"][:, 0] = 1.0  # 非零 t0 nav，便于区分
    if with_world:
        obs["ego_world"] = torch.tensor([[6.43, 0.0, 0.0], [6.43, 0.0, 0.0]])
        obs["route_world"] = _route_t(_B)
        obs["route_world_mask"] = torch.ones(_B, 4)
    return obs


class _StubFrame:
    def __init__(self, batch: int):
        self.node_mask = torch.zeros(batch, 33)
        self.pose = torch.zeros(batch, 33, 3)


class _StubEncoded:
    def __init__(self, batch: int):
        self.ego_ctx = torch.zeros(batch, H_STUB)
        self.od_ctx = torch.zeros(batch, 16, H_STUB)
        self.ld_ctx = torch.zeros(batch, 16, H_STUB)
        self.others_ctx = torch.zeros(batch, H_STUB)
        self.od_now = torch.zeros(batch, 16, 9)
        self.od_live = torch.zeros(batch, 16)
        self.ld_now = torch.zeros(batch, 16, 7)
        self.ld_live = torch.zeros(batch, 16)
        self.frame = _StubFrame(batch)


class _StubEncoders:
    @staticmethod
    def embed_nav(feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return feats[:, 0, :H_STUB] * mask


class _StubMemEncoder:
    def __init__(self):
        self.synced_nav: List[torch.Tensor] = []

    def encode(self, encoders, mem: MemBank) -> _StubEncoded:
        self.synced_nav.append(mem.others[:, -1, :11].detach().clone())
        return _StubEncoded(mem.batch)


class _StubStGNN:
    def od_state_from_features(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def ld_state_from_features(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    def __call__(self, **kwargs):  # noqa: ANN003
        z_ego = kwargs["z_ego"]
        z_od = kwargs["z_od"]
        z_ld = kwargs["z_ld"]
        batch = int(z_ego.shape[0])
        return (
            z_ego,  # z_ego_next
            z_od,   # z_od_next
            z_ld,   # z_ld_next
            torch.zeros(batch, 16, 5),   # od_pred
            torch.zeros(batch, 16, 4),   # ld_pred
            torch.zeros(batch, 16),      # presence
            torch.zeros(batch, 16),      # entry
        )


class _StubModel:
    dt = 0.5

    def __init__(self):
        self.encoders = _StubEncoders()
        self.mem_encoder = _StubMemEncoder()
        self.st_gnn = _StubStGNN()
        self.seen_nav: List[torch.Tensor] = []

    def encode(self, obs):  # noqa: ANN001
        mem = mem_from_obs(obs, others_dim=28, history_frames=6)
        nav = obs["nav"]
        if nav.ndim == 3:
            nav = nav[:, 0]
        encoded = _StubEncoded(mem.batch)
        return {
            "mem": mem,
            "encoded": encoded,
            "frame": encoded.frame,
            "z_ego": torch.zeros(mem.batch, H_STUB),
            "z_od": torch.zeros(mem.batch, 16, H_STUB),
            "z_ld": torch.zeros(mem.batch, 16, H_STUB),
            "od_pool": torch.zeros(mem.batch, H_STUB),
            "ld_pool": torch.zeros(mem.batch, H_STUB),
            "others_ctx": torch.zeros(mem.batch, H_STUB),
            "nav_token": nav[:, :H_STUB],
            "signal_token": torch.zeros(mem.batch, H_STUB),
        }

    def plan_step(self, z_ego, z_od, z_ld, od_live, ld_live, others_ctx, nav_token, signal_token):  # noqa: ANN001
        self.seen_nav.append(nav_token.detach().clone())
        return None, torch.zeros(nav_token.shape[0], 6), {}


def test_teacher_forcing_rebuilds_nav_step_by_step():
    """v8：教师强制逐步重建 nav（6 次）；不再回写 others（t0 静态上下文）。"""
    model = _StubModel()
    obs = _teacher_obs(with_world=True)
    actions = torch.tensor(
        [[[5.0, 0.05], [5.0, 0.05], [5.0, 0.05], [5.0, 0.05], [5.0, 0.05], [5.0, 0.05]]] * _B
    )
    mem_before = mem_from_obs(obs, others_dim=28, history_frames=6).others.clone()
    wm_teacher_forcing_predictions(model, obs, {}, actions)

    assert len(model.seen_nav) == 6, "教师强制 6 步各跑一次 plan head"
    # 第 1 步 plan head 用 t0 nav（来自 obs）；第 2..6 步用逐步重建结果
    t0_nav = obs["nav"][:, :H_STUB]
    assert torch.allclose(model.seen_nav[0], t0_nav)
    assert not torch.allclose(model.seen_nav[0], model.seen_nav[1]), "nav 必须逐步重建"

    pose = obs["ego_world"].clone()
    for k in range(6):
        pose = advance_pose_world(pose, actions[:, k, 0], actions[:, k, 1])
        expected, _ = nav_features_from_world(pose, obs["route_world"], obs["route_world_mask"])
        if k + 1 < 6:
            assert torch.allclose(model.seen_nav[k + 1], expected[:, :H_STUB]), f"第 {k + 2} 步 nav 不同步"
    # v8：mem.others 不得被逐步改写（others 上下文 = t0 静态；nav 由独立 token 承载）
    after = mem_from_obs(obs, others_dim=28, history_frames=6).others
    assert torch.equal(after, mem_before), "教师强制不得改写 raw mem 的 others"


def test_teacher_forcing_without_world_keys_keeps_t0_nav():
    """旧数据（无 ego_world/route_world）→ 回退 t0 冻结（逐位兼容）。"""
    model = _StubModel()
    obs = _teacher_obs(with_world=False)
    actions = torch.zeros(_B, 6, 2)
    wm_teacher_forcing_predictions(model, obs, {}, actions)
    assert len(model.seen_nav) == 6
    for token in model.seen_nav:
        assert torch.allclose(token, obs["nav"][:, :H_STUB]), "缺世界系键时必须保持 t0 nav"


class _GradStubModel(_StubModel):
    """带可微 others/nav 通路的 teacher forcing stub（用于 backward 回归）。"""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(H_STUB, 6)

    def plan_step(self, z_ego, z_od, z_ld, od_live, ld_live, others_ctx, nav_token, signal_token):  # noqa: ANN001
        self.seen_nav.append(nav_token.detach().clone())
        ego_next = self.head(others_ctx + nav_token)
        return None, ego_next, {}


def test_teacher_forcing_backward_survives_nav_rebuild():
    """回归：nav 重建不得破坏 teacher forcing 反向（v8：不写 mem；nav/others 通路可微）。"""
    model = _GradStubModel()
    obs = _teacher_obs(with_world=True)
    actions = torch.full((_B, 6, 2), 0.1)
    predictions = wm_teacher_forcing_predictions(model, obs, {}, actions)
    loss = sum(tensor.float().mean() for tensor in predictions.values())
    loss.backward()
    grads = [parameter.grad for parameter in model.head.parameters()]
    assert all(grad is not None for grad in grads), "plan head 参数必须收到梯度"
    assert any(float(grad.abs().sum()) > 0.0 for grad in grads), "梯度不得为全 0"
    assert len(model.seen_nav) == 6


def test_teacher_forcing_does_not_write_mem_others(monkeypatch):
    """v8 对照：教师强制不得调用 ``sync_others_nav_dims``（不写 raw mem 的 others）。"""
    import net.mem as mem_module

    called: List[int] = []

    def _detect(*args, **kwargs):  # noqa: ANN002, ANN003
        called.append(1)
        raise AssertionError("v8：教师强制不得写回 mem.others")

    monkeypatch.setattr(mem_module, "sync_others_nav_dims", _detect)
    model = _GradStubModel()
    obs = _teacher_obs(with_world=True)
    actions = torch.full((_B, 6, 2), 0.1)
    wm_teacher_forcing_predictions(model, obs, {}, actions)
    assert not called, "v8：教师强制不得写回 mem.others"
    assert len(model.seen_nav) == 6


# --------------------------------------------------------------------------- #
# 5. buffer / _assemble_obs_batch 透传
# --------------------------------------------------------------------------- #

OBS_WORLD = {
    "ego": np.full((1, 8), 0.5, dtype=np.float32),
    "ego_world": np.array([[1.0, 2.0, 0.1]], dtype=np.float32),
    "route_world": np.arange(8, dtype=np.float32).reshape(4, 2),
    "route_world_mask": np.ones(4, dtype=np.float32),
}


class _ScriptedPool:
    num_envs = 1
    accepts_pre_step_labels = False

    def reset(self):  # noqa: ANN201
        return [self._record()]

    def step(self, actions, references=None):  # noqa: ANN001
        return [self._record()]

    @staticmethod
    def _record() -> Dict[str, Any]:
        return {
            "obs": OBS_WORLD,
            "info": {},
            "reward": 0.25,
            "terminated": False,
            "truncated": False,
            "spec_id": 0,
        }


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:
        pass

    def step(self, env_index, info, obs, done, pool_reward, step_index=0):  # noqa: ANN001
        return float(pool_reward), {}


class _MultiKeySmokeModel(_SmokeModel):
    """多键 obs 的 smoke stub：``_SmokeModel`` 的 trunk 只接受单键拼接，这里取各键编码均值。"""

    def forward(self, obs, *, rollout: bool = True, world_model: bool = True, wm_detach: bool = False):  # noqa: ANN001
        encoded = [self.encoders[key](torch.flatten(obs[key], start_dim=1)) for key in self.keys]
        hidden = self.trunk(torch.stack(encoded, dim=0).mean(dim=0))
        span = self.action_high - self.action_low
        action = self.action_low + span * torch.sigmoid(self.head_mu(hidden))
        return {
            "action_mu": action,
            "action_logstd": self.head_logstd.clamp(-5.0, 0.0).unsqueeze(0).expand(hidden.shape[0], -1),
            "value": self.head_value(hidden),
            "traj_xy": self.head_traj(hidden).reshape(-1, 6, 2),
            "plan": action.unsqueeze(1).expand(-1, 6, -1).contiguous(),
            "router_logits": self.head_router(hidden),
            "latent": hidden,
        }


def test_buffer_and_assemble_carry_world_keys_collect_to_update():
    torch.manual_seed(0)
    model = _MultiKeySmokeModel(OBS_WORLD)
    trainer = PPOTrainer(
        model,
        _ScriptedPool(),
        PPOConfig(epochs=1, minibatch_size=4, lr=1e-3, device="cpu", router_expert_norm_probe=False),
        reward_adapter=_RewardStub(),
        probe_batch=None,
        logger=lambda _line: None,
    )
    trainer.adopt_obs([OBS_WORLD])
    trainer.collect_rollout(4)
    assert "ego_world" in trainer.buffer.channels
    assert "route_world" in trainer.buffer.channels
    assert tuple(trainer.buffer.channels["route_world"]) == (4, 2)

    batch = trainer._assemble_obs_batch(np.arange(3))
    assert batch["route_world"].shape == (3, 4, 2)
    assert batch["route_world_mask"].shape == (3, 4)
    assert batch["ego_world"].shape == (3, 3), "单槽键必须被挤压为 (B,3)"
    np.testing.assert_allclose(batch["route_world"][0], OBS_WORLD["route_world"])
    np.testing.assert_allclose(batch["ego_world"][0], OBS_WORLD["ego_world"][0])
    # update 路径（cheap path）必须能消费含新键的 obs（旧模型忽略未知键）
    metrics = trainer.update()
    assert metrics["batches"] >= 1
