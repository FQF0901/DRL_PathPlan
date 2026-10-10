"""LqrTracker ``ref_speed_scale``（P1-0B §7.10-A3 V3 配速）CPU 单测。

不建 env：以 stub control_object 直接实例化（MetaDrive ``BasePolicy.show_policy_mark``
需要 engine，测试内 monkeypatch 为空操作）。覆盖：
- 默认 1.0：``_ref_speed`` 与旧公式 ``chords / dt_sub`` 逐位一致（两个默认实例对比）；
- 1.15：``_ref_speed`` = 1.15×旧值（逐位），且 ``_ref_world`` 几何逐位不变。
"""

from __future__ import annotations

import numpy as np
import pytest

from metadrive.policy.base_policy import BasePolicy

from env.tracking import LqrTracker, ego_to_world, interpolate


class _StubControlObject:
    """仅提供 LqrTracker 参数推导所需字段（轴距 / max_steering）。"""

    FRONT_WHEELBASE = 1.05234
    REAR_WHEELBASE = 1.4166
    config = {"max_steering": 40.0}


@pytest.fixture(autouse=True)
def _no_policy_mark(monkeypatch):
    """无 MetaDrive engine 时 ``BasePolicy.__init__`` 的调试标记为空操作。"""
    monkeypatch.setattr(BasePolicy, "show_policy_mark", lambda self: None)


def _old_ref_speed(actions: np.ndarray, base_pose) -> tuple[np.ndarray, np.ndarray]:
    """独立复算旧公式：``(ds,dθ)`` → 10 Hz 位姿 → 世界系弦长 / 0.1 s。"""
    points = interpolate(actions, dt=0.5, hz=10)
    world = ego_to_world(points, base_pose)
    prev = np.vstack([np.asarray(base_pose, dtype=np.float64)[:2], world[:-1, :2]])
    chords = np.linalg.norm(world[:, :2] - prev, axis=1)
    return chords / 0.1, world


def _make_tracker(**kwargs) -> LqrTracker:
    return LqrTracker(_StubControlObject(), 0, **kwargs)


def test_default_scale_matches_old_formula_bitwise() -> None:
    actions = np.array(
        [[2.0, 0.10], [2.1, 0.12], [1.9, 0.08], [2.0, -0.05], [1.8, -0.10], [2.2, 0.0]]
    )
    base_pose = (3.0, -1.0, 0.25)

    old_speed, old_world = _old_ref_speed(actions, base_pose)
    default = _make_tracker()  # 不传参数 = 默认 1.0
    explicit = _make_tracker(ref_speed_scale=1.0)

    assert default.ref_speed_scale == 1.0
    assert explicit.ref_speed_scale == 1.0
    default.set_reference(actions, base_pose=base_pose)
    explicit.set_reference(actions, base_pose=base_pose)

    # 默认 1.0 与旧公式逐位一致
    np.testing.assert_array_equal(default._ref_speed, old_speed)
    np.testing.assert_array_equal(default._ref_world, old_world)
    # 两个实例（默认 vs 显式 1.0）逐位一致
    np.testing.assert_array_equal(default._ref_speed, explicit._ref_speed)
    np.testing.assert_array_equal(default._ref_world, explicit._ref_world)


def test_scale_115_scales_speed_only_and_keeps_geometry() -> None:
    actions = np.array(
        [[2.0, 0.10], [2.1, 0.12], [1.9, 0.08], [2.0, -0.05], [1.8, -0.10], [2.2, 0.0]]
    )
    base_pose = (3.0, -1.0, 0.25)

    old_speed, old_world = _old_ref_speed(actions, base_pose)
    baseline = _make_tracker()
    scaled = _make_tracker(ref_speed_scale=1.15)
    baseline.set_reference(actions, base_pose=base_pose)
    scaled.set_reference(actions, base_pose=base_pose)

    assert scaled.ref_speed_scale == 1.15
    # _ref_speed = 1.15 × 旧值（逐位）
    np.testing.assert_array_equal(scaled._ref_speed, old_speed * 1.15)
    np.testing.assert_array_equal(scaled._ref_speed, baseline._ref_speed * 1.15)
    # 几何参考逐位不变
    np.testing.assert_array_equal(scaled._ref_world, old_world)
    np.testing.assert_array_equal(scaled._ref_world, baseline._ref_world)
    # 弧长/游标也不受影响
    np.testing.assert_array_equal(scaled._ref_arc, baseline._ref_arc)
    assert scaled._cursor == baseline._cursor == 0
