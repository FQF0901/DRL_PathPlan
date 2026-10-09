"""P0-B forensics 扩展（D1/D2/repeat_action + footprint/时间戳）CPU 单测。

不建 env、不加载 ckpt；metadrive 仅在 oracle (N,3) plan 分支被惰性 import。
覆盖：
- ``expand_plan_poses``：arc_step 同口径展开、plan 节点保留、0.5–1.0 m 重采样间距；
- ``footprint_points``：车辆矩形 9 点（四角+四边中点+中心）与旋转；
- ``check_footprint_path``：退化三元组（inside_ratio / first_invalid_pose /
  invalid_footprint_point_count；``min_signed_margin`` 恒 None，不伪造距离）；
- ``finalize_episode``：T_plan/T_track/T_cross/T_term 与层级分类（plan/tracker/
  recovery/anomaly）；
- D1 ``expert_chain_plan``/``relocate_cursor``（复用 ``collect_expert._window_actions`` 口径）；
- ``InstrumentedCkpt`` 记录扩展与 ``--reference repeat_action``（stub 模型/跟踪器）；
- CLI 新增 ``d1``/``d2``/``--reference`` 选项。
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pytest
import torch

from tools.diagnostics.forensics_closed_loop import (
    FOOTPRINT_MAX_GAP_M,
    STEPS_PER_POLICY,
    TRACK_RESIDUAL_THRESHOLD_M,
    _policy_fields,
    attach_footprints,
    attach_tracking_residual,
    build_parser,
    check_footprint_path,
    expert_chain_plan,
    expand_plan_poses,
    finalize_episode,
    footprint_points,
    relocate_cursor,
    resample_poses,
)


# --------------------------------------------------------------------------- #
# 独立参考实现：逐段圆弧闭式推进（与 pipeline.trainer._arc_endpoint 同式）
# --------------------------------------------------------------------------- #

def _arc_advance(pose, ds: float, dtheta: float) -> np.ndarray:
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    if abs(dtheta) < 1e-12:
        return np.array([x + ds * math.cos(theta), y + ds * math.sin(theta), theta], dtype=np.float64)
    radius = ds / dtheta
    theta_new = theta + dtheta
    return np.array(
        [
            x + radius * (math.sin(theta_new) - math.sin(theta)),
            y - radius * (math.cos(theta_new) - math.cos(theta)),
            theta_new,
        ],
        dtype=np.float64,
    )


def _plan_nodes(plan, base_pose) -> np.ndarray:
    pose = np.asarray(base_pose, dtype=np.float64)
    out = [pose.copy()]
    for ds, dtheta in plan:
        pose = _arc_advance(pose, float(ds), float(dtheta))
        out.append(pose.copy())
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# 插值
# --------------------------------------------------------------------------- #

def test_expand_plan_poses_preserves_nodes_and_matches_arc() -> None:
    plan = np.array(
        [[2.0, 0.2], [1.5, -0.1], [0.8, 0.0], [2.4, 0.3], [1.2, 0.05], [1.0, -0.2]]
    )
    base = (1.0, 2.0, 0.3)
    poses = expand_plan_poses(plan, base)
    nodes = _plan_nodes(plan, base)
    assert poses.shape[1] == 3
    assert np.allclose(poses[0], base)
    # 每个 plan 节点都在采样序列中（arc_step 组合 = 整段闭式解）
    for node in nodes:
        assert np.min(np.linalg.norm(poses[:, :2] - node[:2], axis=1)) < 1e-9
        assert np.min(np.abs(poses[:, 2] - node[2])) < 1e-9
    # 相邻采样弦长 ≤ 1.0 m；|ds| > 1.0 的段内间距 ≥ 0.5 m
    gaps = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    assert float(gaps.max()) <= FOOTPRINT_MAX_GAP_M + 1e-9
    n_prev = 1
    for ds, _ in plan:
        n_sub = max(1, int(math.ceil(abs(float(ds)) / FOOTPRINT_MAX_GAP_M - 1e-9)))
        seg = gaps[n_prev - 1: n_prev - 1 + n_sub]
        if abs(float(ds)) > FOOTPRINT_MAX_GAP_M:
            assert float(seg.min()) >= 0.5 - 1e-9
        n_prev += n_sub
    assert len(poses) == n_prev


def test_expand_plan_poses_empty_and_gap_override() -> None:
    assert expand_plan_poses(np.zeros((0, 2)), (0.0, 0.0, 0.0)).shape == (0, 3)
    plan = np.array([[3.0, 0.0]])
    poses = expand_plan_poses(plan, (0.0, 0.0, 0.0), max_gap_m=0.5)
    gaps = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    assert len(poses) == 7  # 3.0 / 0.5
    assert float(gaps.max()) <= 0.5 + 1e-9


def test_resample_poses_gap_and_endpoints() -> None:
    poses = np.column_stack([np.linspace(0.0, 10.0, 3), np.zeros(3), np.zeros(3)])
    out = resample_poses(poses, max_gap_m=1.0)
    gaps = np.linalg.norm(np.diff(out[:, :2], axis=0), axis=1)
    assert float(gaps.max()) <= 1.0 + 1e-9
    assert np.allclose(out[0], poses[0])
    assert np.allclose(out[-1], poses[-1])


# --------------------------------------------------------------------------- #
# 矩形采样
# --------------------------------------------------------------------------- #

def test_footprint_points_nine_points_identity() -> None:
    points = footprint_points((0.0, 0.0, 0.0), 4.0, 2.0)
    expected = np.array(
        [[2, 1], [2, -1], [-2, -1], [-2, 1], [2, 0], [-2, 0], [0, 1], [0, -1], [0, 0]],
        dtype=np.float64,
    )
    assert points.shape == (9, 2)
    assert np.allclose(points, expected)


def test_footprint_points_rotation_and_translation() -> None:
    pose = (3.0, -1.0, math.pi / 2.0)
    points = footprint_points(pose, 4.0, 2.0)
    body = np.array(
        [[2, 1], [2, -1], [-2, -1], [-2, 1], [2, 0], [-2, 0], [0, 1], [0, -1], [0, 0]],
        dtype=np.float64,
    )
    cos_t, sin_t = math.cos(pose[2]), math.sin(pose[2])
    expected = np.column_stack(
        [
            pose[0] + body[:, 0] * cos_t - body[:, 1] * sin_t,
            pose[1] + body[:, 0] * sin_t + body[:, 1] * cos_t,
        ]
    )
    assert np.allclose(points, expected)
    assert np.allclose(points[-1], [pose[0], pose[1]])  # 中心 = 位姿原点


# --------------------------------------------------------------------------- #
# footprint 判定 + 退化三元组
# --------------------------------------------------------------------------- #

def _path_along_x(xs) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.float64)
    return np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)])


def test_check_footprint_path_degenerate_triple() -> None:
    poses = _path_along_x(np.arange(0.0, 11.0, 1.0))

    def on_lane(point_xy) -> bool:
        return float(point_xy[0]) <= 7.0

    info = check_footprint_path(poses, 4.0, 2.0, on_lane)
    # 独立复算：车长 4 m → 车头角点 x+2；x+2 > 7 的首个位姿 = x=6（index 6）
    expected_bad = 0
    expected_first: Optional[int] = None
    for index, pose in enumerate(poses):
        bad = int(np.sum(footprint_points(pose, 4.0, 2.0)[:, 0] > 7.0))
        expected_bad += bad
        if bad and expected_first is None:
            expected_first = index
    assert info["valid"] is False
    assert info["n_poses"] == 11
    assert info["inside_ratio"] == pytest.approx(6.0 / 11.0, abs=1e-4)
    assert info["first_invalid_pose"]["index"] == expected_first == 6
    assert info["first_invalid_pose"]["s_m"] == pytest.approx(6.0)
    assert info["invalid_footprint_point_count"] == expected_bad == 27
    assert info["min_signed_margin"] is None  # 引擎只给布尔 → 不伪造连续距离


def test_check_footprint_path_all_valid_and_query_failure() -> None:
    poses = _path_along_x([0.0, 1.0, 2.0])
    info = check_footprint_path(poses, 4.0, 2.0, lambda point: True)
    assert info["valid"] is True
    assert info["inside_ratio"] == 1.0
    assert info["first_invalid_pose"] is None
    assert info["invalid_footprint_point_count"] == 0
    assert info["min_signed_margin"] is None

    def boom(point_xy):
        raise RuntimeError("query down")

    failed = check_footprint_path(poses, 4.0, 2.0, boom)
    assert failed["valid"] is False
    assert failed["invalid_footprint_point_count"] == 3 * 9  # 查询失败按不可行计（保守）


def test_attach_footprints_fills_plan_and_exec_fields() -> None:
    plan = np.array([[2.0, 0.0]] * 6)
    records = [
        {"step": 0, "decision": True, "x": 0.0, "y": 0.0, "theta": 0.0, "plan": plan.tolist()},
        {"step": 1, "decision": False, "x": 0.1, "y": 0.0, "theta": 0.0},
    ]
    steps = [
        {"x": 0.1, "y": 0.0, "theta": 0.0},
        {"x": 0.2, "y": 0.0, "theta": 0.0},
    ]
    attach_footprints(records, steps, lambda point: float(point[0]) <= 20.0, 4.0, 2.0)
    assert records[0]["plan_footprint_valid"] is True  # 12 m plan（含车头 +2 m）全在 lane 内
    assert records[0]["exec_footprint_valid"] is True
    assert records[1]["plan_footprint_valid"] is None  # 非决策步无 plan
    assert records[1]["exec_footprint_valid"] is True
    assert records[0]["min_signed_margin"] is None
    assert records[0]["plan_footprint"]["first_invalid_pose"] is None


def test_attach_footprints_oracle_pose_plan() -> None:
    pytest.importorskip("metadrive")
    # oracle 分支：plan 是自车系 (N,3) 位姿（前向直线），world 后重采样
    local = np.column_stack([np.linspace(0.0, 3.0, 31), np.zeros(31), np.zeros(31)])
    records = [{"step": 0, "decision": True, "x": 0.0, "y": 0.0, "theta": 0.0, "plan": local.tolist()}]
    steps = [{"x": 0.1, "y": 0.0, "theta": 0.0}]
    attach_footprints(records, steps, lambda point: True, 4.0, 2.0)
    assert records[0]["plan_footprint_valid"] is True
    assert records[0]["plan_footprint"]["n_poses"] >= 4


def test_attach_tracking_residual_window_alignment() -> None:
    records = [
        {"step": 0, "decision": True, "tracking_residual": None},
        {"step": 1, "decision": False, "tracking_residual": None},
        {"step": 2, "decision": False, "tracking_residual": None},
    ]
    steps = [{"x": 0.0, "y": 0.0}, {"x": 0.11, "y": 0.0}, {"x": 0.22, "y": 0.0}]
    windows = [np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]), None, None]
    attach_tracking_residual(records, steps, windows)
    # 同时刻最大 xy 偏差 = max(|0.0-0.1|, |0.11-0.2|, |0.22-0.3|) = 0.1
    assert records[0]["tracking_residual"] == pytest.approx(0.1)
    assert records[1]["tracking_residual"] is None


# --------------------------------------------------------------------------- #
# 时间戳 / 层级分类
# --------------------------------------------------------------------------- #

def _rec(step: int, decision: bool, *, plan_valid=None, exec_valid=None, residual=None) -> dict:
    return {
        "step": step,
        "decision": decision,
        "plan_footprint_valid": plan_valid,
        "exec_footprint_valid": exec_valid,
        "tracking_residual": residual,
    }


def _steps(n: int) -> List[dict]:
    return [{"x": 0.1 * i, "y": 0.0, "theta": 0.0} for i in range(n)]


def test_finalize_plan_originated() -> None:
    records = [_rec(0, True, plan_valid=True, exec_valid=True)]
    records += [_rec(i, i % 5 == 0, exec_valid=True) for i in range(1, 5)]
    records += [_rec(5, True, plan_valid=False, exec_valid=True)]
    records += [_rec(i, False, exec_valid=True) for i in range(6, 7)]
    records += [_rec(7, False, exec_valid=False)]
    out = finalize_episode(records, _steps(8), "out_of_road")
    assert out["timestamps"]["T_plan"]["step"] == 5
    assert out["timestamps"]["T_cross"]["step"] == 7
    assert out["timestamps"]["T_term"]["step"] == 8
    assert out["timestamps"]["T_term"]["reason"] == "out_of_road"
    assert out["classification"]["primary"] == "plan_originated"
    assert out["classification"]["factors"]["plan_originated"] is True
    assert out["classification"]["factors"]["anomaly"] is False
    # 每行记录冗余 episode 级时间戳（字段冻结）
    assert records[0]["T_plan"]["step"] == 5
    assert records[0]["T_cross"]["step"] == 7


def test_finalize_tracker_originated() -> None:
    records = [_rec(0, True, plan_valid=True, exec_valid=True, residual=0.1)]
    records += [_rec(i, False, exec_valid=True) for i in range(1, 5)]
    records += [_rec(5, True, plan_valid=True, exec_valid=True, residual=TRACK_RESIDUAL_THRESHOLD_M + 0.3)]
    records += [_rec(i, False, exec_valid=True) for i in range(6, 8)]
    records += [_rec(8, False, exec_valid=False)]
    out = finalize_episode(records, _steps(9), "out_of_road")
    assert out["timestamps"]["T_track"]["step"] == 5
    assert out["timestamps"]["T_plan"] is None
    assert out["classification"]["primary"] == "tracker_originated"
    assert out["classification"]["factors"]["tracker_originated"] is True


def test_finalize_recovery_failure_and_none() -> None:
    records = [_rec(0, True, plan_valid=True, exec_valid=True)]
    records += [_rec(1, False, exec_valid=False)]
    records += [_rec(i, False, exec_valid=False) for i in range(2, 5)]
    records += [_rec(5, True, plan_valid=True, exec_valid=False)]  # 危险后仍存在可行 recovery
    out = finalize_episode(records, _steps(6), "collision")
    assert out["classification"]["factors"]["recovery_failure"] is True
    assert out["classification"]["primary"] == "recovery_failure"

    clean = [_rec(0, True, plan_valid=True, exec_valid=True)]
    clean += [_rec(i, False, exec_valid=True) for i in range(1, 4)]
    out_clean = finalize_episode(clean, _steps(4), "arrive_dest")
    assert out_clean["classification"]["primary"] == "none"
    assert all(value is False for value in out_clean["classification"]["factors"].values())


def test_finalize_anomaly_geometry_vs_termination() -> None:
    records = [_rec(0, True, plan_valid=True, exec_valid=True)]
    records += [_rec(i, False, exec_valid=True) for i in range(1, 4)]
    out = finalize_episode(records, _steps(4), "out_of_road")
    assert out["classification"]["factors"]["anomaly"] is True
    assert out["classification"]["primary"] == "anomaly"


# --------------------------------------------------------------------------- #
# D1：expert 动作链（复用 collect_expert._window_actions）
# --------------------------------------------------------------------------- #

def test_expert_chain_plan_matches_collect_expert_caliber() -> None:
    from tools.collect_expert import _window_actions

    steps = 120
    speed = 4.0
    poses = np.column_stack(
        [speed * 0.1 * np.arange(steps), np.zeros(steps), 0.02 * np.arange(steps)]
    )
    plan = expert_chain_plan(poses, 0)
    expected = _window_actions(poses, 0, n_policies=6)
    assert plan is not None and plan.shape == (6, 2)
    assert np.allclose(plan, expected)
    assert plan[0, 0] == pytest.approx(2.0)  # 5 × 0.4 m
    assert plan[0, 1] == pytest.approx(0.1)  # 5 × 0.02 rad


def test_expert_chain_plan_clamps_and_short_trajectory() -> None:
    poses = np.column_stack([0.4 * np.arange(40), np.zeros(40), np.zeros(40)])
    plan = expert_chain_plan(poses, 999)  # clamp 到 last - 30
    assert plan is not None
    assert np.allclose(plan[:, 0], 2.0)
    assert expert_chain_plan(poses[:20], 0) is None  # 不足 3 s 窗口
    assert expert_chain_plan(np.zeros((0, 3)), 0) is None


def test_relocate_cursor_nearest_point() -> None:
    poses = np.column_stack([np.arange(20.0), np.zeros(20), np.zeros(20)])
    assert relocate_cursor(poses, 0, (7.3, 0.0)) == 7
    assert relocate_cursor(poses, 15, (2.0, 0.0)) == 2  # 窗口 [0, 75) 内全局最近
    assert relocate_cursor(poses, 19, (30.0, 0.0)) == 19  # clamp 到 len-2


# --------------------------------------------------------------------------- #
# 记录扩展：stub 模型/跟踪器（不建 env）
# --------------------------------------------------------------------------- #

class _StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rollout_flags: List[bool] = []

    def forward(self, obs, *, rollout: bool = True, world_model: bool = True, wm_detach: bool = False):
        self.rollout_flags.append(bool(rollout))
        mu = torch.tensor([[3.0, 0.2]], dtype=torch.float64)
        plan = mu.unsqueeze(1) + torch.arange(6, dtype=torch.float64).reshape(1, 6, 1) * 10.0
        logstd = torch.tensor([[-1.0, -2.0]], dtype=torch.float64)
        weights = torch.tensor([[0.0, 0.6, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        return {
            "action_mu": mu,
            "action_logstd": logstd,
            "plan": plan,
            "expert_weights": weights,
        }


class _StubTracker:
    def __init__(self) -> None:
        self.references: Optional[np.ndarray] = None
        self._ref_world = None

    def set_reference(self, reference) -> None:
        self.references = np.asarray(reference, dtype=np.float64).copy()


class _StubBuilder:
    def build(self, env, spec):  # noqa: D102 - stub
        return {"ego": np.zeros((1, 8), dtype=np.float32)}


class _StubAgent:
    position = (0.0, 0.0)
    heading_theta = 0.0
    speed = 4.0


class _StubEnv:
    def __init__(self) -> None:
        self.agent = _StubAgent()
        self.prev_policy_action = np.zeros(2)


class _StubSpec:
    id = 7
    seed = 11
    blocks = "SC"


def _stub_controller(eval_reference: str, dtheta_gain: float = 1.0):
    from pipeline import eval_runner as ev

    from tools.diagnostics.forensics_closed_loop import InstrumentedCkpt

    controller = object.__new__(InstrumentedCkpt)
    controller.spec = _StubSpec()
    controller.device = torch.device("cpu")
    controller.records = []
    controller.model = _StubModel()
    controller.builder = _StubBuilder()
    controller.tracker_kind = "lqr"
    controller.tracker = _StubTracker()
    controller.decision_interval = 5
    controller._steps = 0
    controller._action = [0.0, 0.0]
    controller._pose_history = []
    controller._ref_windows = []
    controller.eval_reference = eval_reference
    controller.dtheta_gain = dtheta_gain
    controller._ev = ev
    return controller


def test_instrumented_ckpt_plan_mode_records_new_fields() -> None:
    controller = _stub_controller("plan", dtheta_gain=2.0)
    controller.action(_StubEnv())
    rec = controller.records[-1]
    assert controller.model.rollout_flags == [True]
    assert rec["plan"][0] == pytest.approx([3.0, 0.4])  # plan[0]=mu，dθ 放大 2×
    assert rec["mu"] == pytest.approx([3.0, 0.2])
    assert rec["reference"] == "plan"
    assert rec["policy_action_vs_plan_first_action"] == pytest.approx([0.0, 0.2])
    assert rec["policy_std"] == pytest.approx([math.exp(-1.0), math.exp(-2.0)], abs=1e-6)
    assert rec["router_topk"] == [[1, 0.6], [3, 0.4]]
    assert rec["min_signed_margin"] is None
    assert rec["T_plan"] is None
    assert rec["tracking_residual"] is None


def test_instrumented_ckpt_repeat_action_reference() -> None:
    controller = _stub_controller("repeat_action")
    controller.action(_StubEnv())
    rec = controller.records[-1]
    assert controller.model.rollout_flags == [False]  # cheap path（与 eval_runner 同口径）
    expected = np.repeat(np.asarray([3.0, 0.2])[None, :], 6, axis=0)
    assert np.allclose(controller.tracker.references, expected)
    assert np.allclose(np.asarray(rec["plan"]), expected)
    assert rec["reference"] == "repeat_action"
    assert rec["policy_action_vs_plan_first_action"] == pytest.approx([0.0, 0.0])
    # 非决策步不再前向，但记录骨架字段齐全
    controller.action(_StubEnv())
    rec2 = controller.records[-1]
    assert rec2["decision"] is False
    assert "plan" not in rec2
    assert rec2["plan_footprint_valid"] is None


def test_policy_fields_robust_to_missing_keys() -> None:
    assert _policy_fields({}) == {"policy_std": None, "router_topk": None}
    fields = _policy_fields(
        {
            "action_logstd": torch.zeros(1, 2),
            "expert_weights": torch.zeros(1, 8),  # MoE 关闭：无非零权重
        }
    )
    assert fields["policy_std"] == [1.0, 1.0]
    assert fields["router_topk"] == []


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_modes_and_reference() -> None:
    parser = build_parser()
    assert parser.parse_args(["--out", "x"]).mode == "lqr"
    assert parser.parse_args(["--out", "x"]).reference == "plan"
    assert parser.parse_args(["--mode", "d1", "--out", "x"]).mode == "d1"
    assert parser.parse_args(["--mode", "d2", "--out", "x"]).mode == "d2"
    assert parser.parse_args(["--reference", "repeat_action", "--out", "x"]).reference == "repeat_action"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "nope", "--out", "x"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--reference", "nope", "--out", "x"])


def test_constants_match_collect_expert_caliber() -> None:
    from tools.collect_expert import STEPS_PER_POLICY as CE_STEPS
    from tools.collect_expert import WINDOW_POLICIES as CE_WINDOW

    from tools.diagnostics.forensics_closed_loop import WINDOW_POLICIES

    assert STEPS_PER_POLICY == CE_STEPS == 5
    assert WINDOW_POLICIES == CE_WINDOW == 6
