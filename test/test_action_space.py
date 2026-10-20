"""PSTL 4D action -> trajectory/speed execution layer tests."""

import numpy as np

from net.action_space import (
    COMFORT_LIMITS,
    DEFAULT_BOUNDS,
    HARD_LIMITS,
    CLEAN_BOUNDS,
    action_to_trajectory,
    clip_components,
    denormalize,
    feasible_v1_interval,
    gate,
    normalize,
    project,
)


def test_bounds_constants():
    assert DEFAULT_BOUNDS is CLEAN_BOUNDS
    assert np.allclose(CLEAN_BOUNDS.low, [0.0, -1.5, -0.8, 0.0])
    assert np.allclose(CLEAN_BOUNDS.high, [7.5, 1.5, 0.8, 15.0])
    assert np.all(CLEAN_BOUNDS.width > 0.0)
    # provisional limits are exposed and ordered
    assert HARD_LIMITS.a_long_min < HARD_LIMITS.a_long_max
    assert COMFORT_LIMITS.a_long_max < HARD_LIMITS.a_long_max
    assert COMFORT_LIMITS.kappa_max < HARD_LIMITS.kappa_max


def test_normalize_denormalize_roundtrip_and_clip():
    actions = np.array([[3.0, 0.2, 0.1, 6.0], [7.5, -1.5, -0.8, 15.0]])
    n = normalize(actions, CLEAN_BOUNDS)
    assert n.shape == actions.shape
    back = denormalize(n, CLEAN_BOUNDS)
    assert np.allclose(back, actions)

    # batch clipping is static and per-component
    clipped = clip_components(np.array([[-1.0, 9.0, -9.0, 99.0]]), CLEAN_BOUNDS)
    assert np.allclose(clipped, [[0.0, 1.5, -0.8, 15.0]])

    # denormalize clamps normalized values to [0,1] like legacy denorm_action
    assert np.allclose(
        denormalize(np.array([[-1.0, 2.0, 0.5, 0.5]]), CLEAN_BOUNDS),
        [[0.0, 1.5, 0.0, 7.5]],
    )


def test_endpoint_position_heading_speed_exact():
    cases = [
        (np.array([3.0, 0.0, 0.0, 6.0]), 6.0),
        (np.array([3.0, 1.0, 0.1, 6.0]), 6.0),
        (np.array([2.0, -1.0, -0.2, 5.0]), 5.0),
        (np.array([0.5, 0.0, 0.0, 2.0]), 2.0),
        (np.array([1.0, 0.2, 0.3, 4.0]), 3.0),
    ]
    for action, v0 in cases:
        tr = action_to_trajectory(action, v0, bounds=CLEAN_BOUNDS)
        assert tr.positions.shape == (5, 3)
        assert tr.speeds.shape == (5,)
        assert np.isfinite(tr.positions).all()
        assert np.isfinite(tr.speeds).all()
        assert np.allclose(tr.positions[-1, :2], action[:2], atol=1e-12)
        assert abs(float(tr.positions[-1, 2]) - float(action[2])) < 1e-12
        assert abs(float(tr.speeds[-1]) - float(action[3])) < 1e-12
        assert tr.metrics["start_speed_mps"] == v0
        assert tr.metrics["end_speed_mps"] == float(action[3])
        assert max(tr.endpoint_errors[k] for k in ("position_m", "heading_rad", "speed_mps")) < 1e-12
        # dict-style interface
        assert tr["positions"] is tr.positions
        assert tr.as_dict()["speeds"].shape == (5,)


def test_constant_accel_and_decel_time_law():
    # v0=0 -> v1=2 over T=0.5 with dx=0.5: a=4, jerk=0 exactly
    acc = action_to_trajectory(np.array([0.5, 0.0, 0.0, 2.0]), 0.0, bounds=CLEAN_BOUNDS)
    assert np.allclose(acc.a_long, 4.0)
    assert np.allclose(acc.jerk_long, 0.0)
    assert np.allclose(acc.speeds, [0.4, 0.8, 1.2, 1.6, 2.0])
    assert acc.metrics["start_accel_mps2"] == 4.0
    assert acc.metrics["end_accel_mps2"] == 4.0

    # v0=2 -> v1=1 over T=0.5 with dx=0.75: a=-2, jerk=0 exactly
    dec = action_to_trajectory(np.array([0.75, 0.0, 0.0, 1.0]), 2.0, bounds=CLEAN_BOUNDS)
    assert np.allclose(dec.a_long, -2.0)
    assert np.allclose(dec.jerk_long, 0.0)
    assert np.allclose(dec.speeds, [1.8, 1.6, 1.4, 1.2, 1.0])
    assert dec.metrics["start_accel_mps2"] == -2.0
    assert dec.metrics["end_accel_mps2"] == -2.0


def test_full_stop_hold():
    tr = action_to_trajectory(np.array([0.0, 0.0, 0.0, 0.0]), 0.0, bounds=CLEAN_BOUNDS)
    assert tr.valid and tr.feasible
    assert "FULL_STOP_HOLD" in tr.flags
    assert np.allclose(tr.positions, 0.0)
    assert np.allclose(tr.speeds, 0.0)
    assert np.allclose(tr.a_long, 0.0)
    assert np.allclose(tr.kappa, 0.0)
    assert np.allclose(tr.jerk_long, 0.0)


def test_stop_to_go_endpoint_and_flags():
    tr = action_to_trajectory(np.array([0.5, 0.0, 0.0, 2.0]), 0.0, bounds=CLEAN_BOUNDS)
    assert tr.valid
    assert "STOP_TO_GO" in tr.flags
    assert np.isfinite(tr.positions).all()
    assert np.isfinite(tr.speeds).all()
    assert np.allclose(tr.positions[-1, :2], [0.5, 0.0], atol=1e-12)
    assert abs(float(tr.speeds[-1]) - 2.0) < 1e-12
    assert tr.metrics["speed_nonnegative"]


def test_zero_length_with_speed_rejected():
    tr = action_to_trajectory(np.array([0.0, 0.0, 0.0, 1.0]), 0.0, bounds=CLEAN_BOUNDS)
    assert not tr.valid
    assert not tr.feasible
    assert "ZERO_LENGTH_WITH_SPEED" in tr.codes
    assert np.allclose(tr.positions, 0.0)


def test_rotation_in_place_rejected():
    tr = action_to_trajectory(np.array([0.0, 0.0, 0.5, 0.0]), 0.0, bounds=CLEAN_BOUNDS)
    assert not tr.valid
    assert not tr.feasible
    assert "ROTATION_IN_PLACE" in tr.codes
    assert abs(float(tr.positions[-1, 2]) - 0.5) < 1e-12


def test_negative_dx_is_reported_not_rewritten():
    tr = action_to_trajectory(np.array([-0.5, 0.0, 0.0, 1.0]), 1.0, bounds=CLEAN_BOUNDS)
    assert tr.metrics["dx_negative"]
    assert "DX_NEGATIVE" in tr.flags
    assert np.allclose(tr.positions[-1, :2], [-0.5, 0.0], atol=1e-12)
    assert np.allclose(tr.action, [-0.5, 0.0, 0.0, 1.0])


def test_short_path_skips_curvature_checks():
    # arc length < 0.25 m: huge geometric curvature must not produce a hard code
    tr = action_to_trajectory(np.array([0.2, 0.1, 0.8, 1.0]), 1.0, bounds=CLEAN_BOUNDS)
    assert tr.metrics["path_length_m"] < 0.25
    assert not tr.metrics["curvature_checked"]
    assert "CURVATURE_SKIPPED" in tr.flags
    assert not any(code.startswith("HARD_KAPPA") for code in tr.codes)


def test_project_clips_components_and_reports():
    res = project(np.array([-0.05, 3.0, 2.0, 40.0]), 1.0, bounds=CLEAN_BOUNDS)
    assert np.allclose(res.requested_action, [-0.05, 3.0, 2.0, 40.0])
    assert np.allclose(res.executed_action, [0.0, 1.5, 0.8, 15.0])
    assert res.component_clipped.all()
    assert set(res.clipped_components) == {"dx", "dy", "dpsi", "v_next"}
    assert any("negative dx drift" in reason for reason in res.reasons)
    assert np.allclose(res.trajectory.action, res.executed_action)
    assert not res.feasible


def test_project_feasible_action_is_passthrough():
    action = np.array([3.0, 0.0, 0.0, 6.0])
    res = project(action, 6.0, bounds=CLEAN_BOUNDS)
    assert np.array_equal(res.executed_action, action)
    assert not res.component_clipped.any()
    assert res.feasible and res.comfortable
    assert res.v1_interval.contains(6.0)
    assert res.codes == ()


def test_project_detects_longitudinal_violation_without_rewrite():
    action = np.array([0.5, 0.0, 0.0, 2.0])
    res = project(action, 0.0, bounds=CLEAN_BOUNDS)
    assert np.array_equal(res.executed_action, action)
    assert not res.component_clipped.any()
    assert not res.feasible
    assert "HARD_A_LONG_MAX" in res.codes
    assert not res.v1_interval.nonempty
    # no hidden state-dependent v_next rewrite
    assert res.executed_action[3] == 2.0


def test_project_detects_lateral_and_curvature_violations():
    res = project(np.array([0.3, 0.3, 0.5, 1.0]), 1.0, bounds=CLEAN_BOUNDS)
    assert not res.component_clipped.any()
    assert not res.feasible
    assert any(code in res.codes for code in ("HARD_A_LAT", "HARD_KAPPA", "HARD_KAPPA_RATE"))
    assert res.metrics["max_abs_kappa"] > HARD_LIMITS.kappa_max


def test_feasible_v1_interval_and_gate():
    interval = feasible_v1_interval(np.array([3.0, 0.0, 0.0, 6.0]), 6.0, bounds=CLEAN_BOUNDS)
    assert interval.nonempty
    assert interval.width > 0.0
    assert interval.contains(6.0)
    assert CLEAN_BOUNDS.low[3] <= interval.low <= interval.high <= CLEAN_BOUNDS.high[3]

    bad = feasible_v1_interval(np.array([0.5, 0.0, 0.0, 2.0]), 0.0, bounds=CLEAN_BOUNDS)
    assert not bad.nonempty
    assert "V1_INTERVAL_EMPTY" in bad.codes

    ok_gate = gate(np.array([3.0, 0.0, 0.0, 6.0]), 6.0, bounds=CLEAN_BOUNDS)
    assert ok_gate.accepted
    assert ok_gate.codes == ()
    assert np.allclose(ok_gate.trajectory.action, [3.0, 0.0, 0.0, 6.0])

    bad_gate = gate(np.array([0.5, 0.0, 0.0, 2.0]), 0.0, bounds=CLEAN_BOUNDS)
    assert not bad_gate.accepted
    assert "HARD_A_LONG_MAX" in bad_gate.codes


def test_gate_rejects_out_of_bounds_but_does_not_clip():
    action = np.array([9.0, 0.0, 0.0, 20.0])
    res = gate(action, 6.0, bounds=CLEAN_BOUNDS)
    assert not res.accepted
    assert "OUT_OF_BOUNDS_DX" in res.codes
    assert "OUT_OF_BOUNDS_V_NEXT" in res.codes
    assert np.allclose(res.trajectory.action, action)  # gate never rewrites


def test_trajectory_arrays_finite_for_infeasible_actions():
    rng = np.random.default_rng(0)
    for _ in range(20):
        action = rng.uniform(CLEAN_BOUNDS.low, CLEAN_BOUNDS.high)
        v0 = float(rng.uniform(0.0, 15.0))
        tr = action_to_trajectory(action, v0, bounds=CLEAN_BOUNDS)
        for key in (
            "positions", "speeds", "a_long", "a_lat", "kappa",
            "kappa_rate", "jerk_long", "jerk_lat",
        ):
            assert np.isfinite(getattr(tr, key)).all(), (key, action, v0)
        if tr.feasible:
            assert tr.metrics["speed_nonnegative"]
