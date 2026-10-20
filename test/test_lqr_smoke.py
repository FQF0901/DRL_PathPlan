from unittest import mock

import numpy as np
import pytest

from env.scenario.spec import load_specs
from env.metadrive_env import build_env
from env.tracking import LqrTracker


class _FakeControl:
    """Minimal control object for unit-level LQR reference tests."""

    config = {}


def test_lqr_set_reference_4col_uses_provided_speeds():
    with mock.patch.object(LqrTracker, "show_policy_mark", lambda self: None):
        tracker = LqrTracker(_FakeControl())

    # non-uniform spacing: chord speed would be constant, provided speeds are not
    points = np.zeros((5, 3))
    points[:, 0] = [0.0, 2.0, 4.0, 6.0, 8.0]
    speeds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ref = np.column_stack([points, speeds])
    tracker.set_reference(ref, base_pose=np.zeros(3))

    assert np.allclose(tracker._ref_speed, speeds)
    assert not np.allclose(tracker._ref_speed, np.full(5, 20.0))
    assert np.allclose(tracker._ref_arc, [0.0, 2.0, 4.0, 6.0, 8.0])

    # (N,3) keeps the old chord-speed behavior: first chord is origin -> x=0
    # (zero), then 2 m / 0.1 s = 20 m/s.
    tracker.set_reference(points, base_pose=np.zeros(3))
    assert np.allclose(tracker._ref_speed, [0.0, 20.0, 20.0, 20.0, 20.0])

    # the 4-column speed column is still scaled by ref_speed_scale
    with mock.patch.object(LqrTracker, "show_policy_mark", lambda self: None):
        scaled = LqrTracker(_FakeControl(), ref_speed_scale=0.5)
    scaled.set_reference(ref, base_pose=np.zeros(3))
    assert np.allclose(scaled._ref_speed, speeds * 0.5)

    with pytest.raises(ValueError):
        tracker.set_reference(np.column_stack([points, -np.ones(5)]), base_pose=np.zeros(3))


def test_lqr_straight_reference_smoke():
    spec = load_specs("env/specs/scenarios_train_slice200.json")[0]
    env = build_env(spec, use_render=False)
    try:
        env.reset(); ego = env.agent
        tr = env.engine.add_policy(ego.id, LqrTracker, ego, int(spec.seed))
        ref = np.zeros((30, 3)); ref[:, 0] = 0.6 * np.arange(1, 31)
        tr.set_reference(ref, base_pose=np.array([ego.position[0], ego.position[1], ego.heading_theta]))
        off = False
        for _ in range(30):
            _, _, term, trunc, info = env.step([0.0, 0.0])
            off = off or bool(info.get("out_of_road", False))
            if term or trunc: break
        assert not off
    finally:
        env.close()
