"""P1-0C 探针 12 单测：forensics 三轨中轨 ``ref_world_head`` / ``ref_arc_len``（CPU，不建 env）。

覆盖：
- (a) 决策步列存在、6×2 有限值、坐标 round 1e-4、``ref_arc_len`` 与折线弧长一致；
- (b) 首点 ≈ 决策时自车位姿（本单测用真实 ``interpolate`` + ``ego_to_world`` 造参考，
      首点与 ego 位姿逐位一致；round 1e-4 后容差 ≤5e-5）；
- (c) 取不到（tracker=None / 非决策步 / ref_window=None）→ None 且不报错；
- ExactTracker ``_world`` 回退；参考点数 <6 时取全部。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest
import torch

from tools.diagnostics.forensics_closed_loop import (
    REF_WORLD_HEAD_POINTS,
    InstrumentedCkpt,
    _base_record,
)


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


class _StubTracker:
    """最小 tracker：``set_reference`` 用真实几何造 ``_ref_world``/``_ref_arc``（LqrTracker 同口径）。"""

    def __init__(self, *, world: Optional[np.ndarray] = None, use_world_attr: bool = False) -> None:
        self._ref_world = None if use_world_attr else world
        self._world = world if use_world_attr else None
        self._ref_arc = None
        self.references = None

    def set_reference(self, reference) -> None:
        from env.tracking import ego_to_world, interpolate

        self.references = np.asarray(reference, dtype=np.float64).copy()
        self._ref_world = ego_to_world(interpolate(self.references, dt=0.5, hz=10), (0.0, 0.0, 0.0))
        prev = np.vstack([np.zeros(2), self._ref_world[:-1, :2]])
        chords = np.linalg.norm(self._ref_world[:, :2] - prev, axis=1)
        self._ref_arc = np.cumsum(chords)


class _StubModel(torch.nn.Module):
    def forward(self, obs, *, forward_mode: str = "world_model_teacher_forcing", wm_detach: bool = False):
        mu = torch.tensor([[3.0, 0.2]], dtype=torch.float64)
        plan = mu.unsqueeze(1) + torch.arange(6, dtype=torch.float64).reshape(1, 6, 1) * 1.0
        return {"action_mu": mu, "action_logstd": torch.zeros(1, 2), "plan": plan}


class _StubBuilder:
    def build(self, env, spec):  # noqa: D102 - stub
        return {"ego": np.zeros((1, 8), dtype=np.float32)}


def _stub_controller(*, tracker=None, tracker_kind: str = "lqr"):
    from pipeline import eval_runner as ev

    controller = object.__new__(InstrumentedCkpt)
    controller.spec = _StubSpec()
    controller.device = torch.device("cpu")
    controller.records = []
    controller.model = _StubModel()
    controller.builder = _StubBuilder()
    controller.tracker_kind = tracker_kind
    controller.tracker = tracker
    controller.decision_interval = 5
    controller._steps = 0
    controller._action = [0.0, 0.0]
    controller._pose_history = []
    controller._ref_windows = []
    controller.eval_reference = "plan"
    controller.dtheta_gain = 1.0
    controller._ev = ev
    return controller


# --------------------------------------------------------------------------- #
# (a) 列存在 / 形状 / 取值语义
# --------------------------------------------------------------------------- #

def test_base_record_has_ref_columns_default_none() -> None:
    rec = _base_record(0, False, _StubEnv(), _StubSpec())
    assert rec["ref_world_head"] is None
    assert rec["ref_arc_len"] is None


def test_capture_ref_head_six_points_rounded_and_arc_len() -> None:
    world = np.column_stack(
        [np.arange(30, dtype=np.float64) * 0.5, np.zeros(30), np.zeros(30)]
    )
    world[0, 0] = 0.123456789  # round 1e-4 检查
    tracker = _StubTracker(world=world)
    controller = _stub_controller(tracker=tracker)
    points, arc_len = controller._capture_ref_head()
    assert len(points) == REF_WORLD_HEAD_POINTS == 6
    assert all(len(point) == 2 for point in points)
    assert all(math.isfinite(v) for point in points for v in point)
    assert points[0][0] == round(0.123456789, 4) == 0.1235
    # 折线弧长 = (0.5 − 0.123456789) + 28 × 0.5（首点位移参与弧长）
    assert arc_len == pytest.approx(0.5 - 0.123456789 + 14.0, abs=1e-4)


def test_capture_ref_head_exact_tracker_world_fallback() -> None:
    world = np.column_stack([np.arange(30, dtype=np.float64) * 0.5, np.zeros(30), np.zeros(30)])
    tracker = _StubTracker(world=world, use_world_attr=True)  # ExactTracker：只有 _world
    controller = _stub_controller(tracker=tracker, tracker_kind="exact")
    points, arc_len = controller._capture_ref_head()
    assert len(points) == 6
    assert arc_len == pytest.approx(14.5, abs=1e-4)


def test_capture_ref_head_arc_includes_base_when_ref_arc_missing() -> None:
    world = np.column_stack([np.arange(30, dtype=np.float64) * 0.5, np.zeros(30), np.zeros(30)])
    tracker = _StubTracker(world=world, use_world_attr=True)
    tracker.base_pose = np.array([1.0, 0.0, 0.0])  # ExactTracker 风格：无 _ref_arc，用 base_pose
    controller = _stub_controller(tracker=tracker, tracker_kind="exact")
    _, arc_len = controller._capture_ref_head()
    assert arc_len == pytest.approx(1.0 + 14.5, abs=1e-4)  # base→首点弦 + 折线


def test_capture_ref_head_short_reference_takes_available() -> None:
    world = np.column_stack([np.arange(5, dtype=np.float64) * 0.1, np.zeros(5), np.zeros(5)])
    controller = _stub_controller(tracker=_StubTracker(world=world))
    points, arc_len = controller._capture_ref_head()
    assert len(points) == 5  # 参考不足 6 点 → 取全部，不报错
    assert arc_len == pytest.approx(0.4, abs=1e-4)


# --------------------------------------------------------------------------- #
# (c) 取不到 → None 不报错
# --------------------------------------------------------------------------- #

def test_capture_ref_head_missing_tracker_returns_none() -> None:
    controller = _stub_controller(tracker=None)
    assert controller._capture_ref_head() == (None, None)


def test_emit_without_ref_window_keeps_none_columns() -> None:
    world = np.column_stack([np.arange(30, dtype=np.float64) * 0.5, np.zeros(30), np.zeros(30)])
    controller = _stub_controller(tracker=_StubTracker(world=world))
    rec = _base_record(0, False, _StubEnv(), _StubSpec())
    controller._emit(rec, None)  # 非决策步 / 无参考窗口
    assert rec["ref_world_head"] is None and rec["ref_arc_len"] is None
    assert controller.records[-1] is rec and controller._ref_windows[-1] is None


# --------------------------------------------------------------------------- #
# 集成：decision 步经 action() 走通（真实 interpolate/ego_to_world 参考）
# --------------------------------------------------------------------------- #

def test_action_decision_step_head_content_matches_tracker_reference() -> None:
    from env.tracking import ego_to_world, interpolate

    tracker = _StubTracker()
    controller = _stub_controller(tracker=tracker)
    controller.action(_StubEnv())
    rec = controller.records[-1]
    assert rec["decision"] is True
    head = rec["ref_world_head"]
    assert isinstance(head, list) and len(head) == 6 and all(len(p) == 2 for p in head)
    # 内容逐点 = tracker 世界系参考前 6 点（独立复算；round 1e-4 → 容差 5e-5）
    expected = ego_to_world(interpolate(tracker.references, dt=0.5, hz=10), (rec["x"], rec["y"], rec["theta"]))[
        :6, :2
    ]
    assert np.allclose(np.asarray(head), expected, atol=5e-5)
    # (b) 首点与决策时 ego 距离 ≤ 首子步弦长（interpolate 不含起点；上界 max(ds)/5 = 2.0 m）
    first_step_chord = float(np.linalg.norm(expected[0]))
    assert abs(head[0][0] - rec["x"]) <= first_step_chord + 5e-5
    assert abs(head[0][1] - rec["y"]) <= first_step_chord + 5e-5
    # ref_arc_len = 全参考折线弧长（从决策时 ego 起算）
    assert rec["ref_arc_len"] == pytest.approx(float(tracker._ref_arc[-1]), abs=1e-4)
    assert rec["ref_arc_len"] > 0.0

    # 非决策步：不刷新参考，列缺省 None
    controller.action(_StubEnv())
    rec2 = controller.records[-1]
    assert rec2["decision"] is False
    assert rec2["ref_world_head"] is None and rec2["ref_arc_len"] is None
