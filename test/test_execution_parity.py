"""Parity tests for the shared executed-segment bookkeeping."""

from __future__ import annotations

import numpy as np

from pipeline.execution import DecisionState, relative_action, segment_dt, wrap_angle


def test_relative_action_translation_and_rotation():
    a = relative_action([0.0, 0.0, 0.0], 0.0, 6.0, [1.0, 2.0, 0.0], 0.0, 5.0)
    np.testing.assert_allclose(a, [1.0, 2.0, 0.0, 5.0], atol=1e-12)

    # A world +y displacement is ego +x when the pre-step heading is +90 deg.
    a = relative_action([0.0, 0.0, 0.0], np.pi / 2.0, 6.0, [0.0, 1.0, 0.0], np.pi / 2.0, 6.0)
    np.testing.assert_allclose(a, [1.0, 0.0, 0.0, 6.0], atol=1e-12)


def test_relative_action_wraps_heading():
    a = relative_action([0.0, 0.0, 0.0], 3.0, 1.0, [0.0, 0.0, 0.0], -3.0, 1.0)
    assert abs(float(a[2]) - wrap_angle(-6.0)) < 1e-12
    assert -np.pi <= float(a[2]) < np.pi


def test_decision_state_uses_segment_start_state_and_realized_action():
    state = DecisionState.initial()
    assert state.prev_speed is None
    nxt = state.next(
        start_speed=4.0,
        start_heading=0.2,
        realized_action=[0.5, -0.1, 0.05, 3.5],
    )
    assert nxt.prev_speed == 4.0
    assert nxt.prev_heading == 0.2
    np.testing.assert_allclose(nxt.prev_action, [0.5, -0.1, 0.05, 3.5])
    kwargs = nxt.observation_kwargs(0.3)
    assert kwargs["prev_speed"] == 4.0
    assert kwargs["dt"] == 0.3


def test_segment_dt_for_truncated_segment():
    assert abs(segment_dt(5) - 0.5) < 1e-12
    assert abs(segment_dt(3) - 0.3) < 1e-12
    assert segment_dt(0) == 0.0


def test_limit_history_masks_older_frames():
    import numpy as np
    from pipeline.execution import limit_history
    hist = {"hist_valid": np.ones((1, 6), dtype=np.float32)}
    out = limit_history(hist, 2)
    assert out["hist_valid"][0, -2:].tolist() == [1.0, 1.0]
    assert out["hist_valid"][0, :-2].sum() == 0.0
