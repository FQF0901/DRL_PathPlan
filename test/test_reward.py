"""Unit tests for the rule-based RL reward model (arch_lock §12.4)."""

from __future__ import annotations

import math

import pytest

from pipeline.train.rl_reward import RewardConfig, RewardTracker, compute_step_reward


def _step(**overrides):
    params = {
        "rc0": 0.0,
        "rc1": 0.0,
        "v_end": 5.0,
        "metrics": {},
    }
    params.update(overrides)
    return compute_step_reward(**params)


# ---------------------------------------------------------------------------
# Dense reward
# ---------------------------------------------------------------------------
def test_progress_reward_is_weight_times_route_completion_delta():
    out = _step(rc0=0.2, rc1=0.35)
    assert out["components"]["progress"] == pytest.approx(1.5)
    assert out["dense_reward"] == pytest.approx(1.5)
    assert out["terminal_reward"] == 0.0
    assert out["reward"] == pytest.approx(1.5)
    assert out["terminal_event"] == "none"
    assert out["done"] is False


def test_low_speed_penalty_sign_and_value():
    assert _step(v_end=0.0)["components"]["low_speed"] == pytest.approx(-0.2)
    assert _step(v_end=1.0)["components"]["low_speed"] == pytest.approx(-0.1)
    assert _step(v_end=2.0)["components"]["low_speed"] == 0.0
    assert _step(v_end=5.0)["components"]["low_speed"] == 0.0
    assert _step(v_end=0.5)["dense_reward"] < 0.0


def test_comfort_penalty_uses_segment_max_excess_over_thresholds():
    out = _step(
        metrics={
            "max_abs_a_long": 4.0,
            "max_abs_a_lat": 5.0,
            "max_abs_jerk_long": 9.0,
        }
    )
    # excess: 2 + 2 + 4 = 8; penalty = 0.05 * 8
    assert out["components"]["comfort"] == pytest.approx(-0.4)
    assert out["dense_reward"] == pytest.approx(-0.4)


def test_comfort_is_zero_at_or_below_thresholds():
    out = _step(
        metrics={
            "max_abs_a_long": 2.0,
            "max_abs_a_lat": 3.0,
            "max_abs_jerk_long": 5.0,
        }
    )
    assert out["components"]["comfort"] == 0.0


def test_missing_metrics_are_treated_as_zero():
    out = _step(metrics={"max_abs_a_lat": 4.0})
    assert out["components"]["comfort"] == pytest.approx(-0.05)


# ---------------------------------------------------------------------------
# Terminal priority and reward
# ---------------------------------------------------------------------------
def test_invalid_action_beats_all_other_terminal_events():
    out = _step(
        rc1=1.0,
        invalid_action=True,
        collision=True,
        off_road=True,
        success=True,
        timeout=True,
    )
    assert out["terminal_event"] == "invalid_action"
    assert out["terminal_reward"] == pytest.approx(-20.0)
    assert out["done"] is True


def test_collision_beats_success():
    out = _step(collision=True, success=True)
    assert out["terminal_event"] == "collision"
    assert out["terminal_reward"] == pytest.approx(-20.0)
    assert out["done"] is True


def test_off_road_beats_success():
    out = _step(off_road=True, success=True)
    assert out["terminal_event"] == "off_road"
    assert out["terminal_reward"] == pytest.approx(-15.0)
    assert out["done"] is True


def test_success_beats_timeout():
    out = _step(rc1=0.8, success=True, timeout=True)
    assert out["terminal_event"] == "success"
    assert out["terminal_reward"] == pytest.approx(20.0)
    assert out["done"] is True


def test_success_requires_no_collision_or_off_road():
    assert _step(success=True)["terminal_event"] == "success"
    assert _step(success=True, collision=True)["terminal_event"] == "collision"
    assert _step(success=True, off_road=True)["terminal_event"] == "off_road"


def test_collision_and_off_road_use_adapt_coeff():
    cfg = RewardConfig(adapt_coeff=0.5)
    assert _step(collision=True, config=cfg)["terminal_reward"] == pytest.approx(-10.0)
    assert _step(off_road=True, config=cfg)["terminal_reward"] == pytest.approx(-7.5)


def test_timeout_shaping_uses_remaining_route_completion():
    assert _step(timeout=True)["terminal_reward"] == pytest.approx(-10.0)
    assert _step(rc1=0.25, timeout=True)["terminal_reward"] == pytest.approx(-7.5)
    assert _step(rc1=0.9, timeout=True)["terminal_reward"] == pytest.approx(-1.0)
    assert _step(rc1=1.0, timeout=True)["terminal_reward"] == pytest.approx(0.0)


def test_non_terminal_step_has_no_terminal_reward_and_not_done():
    out = _step()
    assert out["terminal_event"] == "none"
    assert out["terminal_reward"] == 0.0
    assert out["done"] is False


@pytest.mark.parametrize(
    "terminal",
    [
        {"invalid_action": True},
        {"collision": True},
        {"off_road": True},
        {"success": True},
        {"timeout": True},
    ],
)
def test_reward_equals_dense_plus_terminal(terminal):
    out = _step(rc0=0.1, rc1=0.2, v_end=1.0, metrics={"max_abs_a_long": 4.0}, **terminal)
    assert out["reward"] == pytest.approx(
        out["dense_reward"] + out["terminal_reward"]
    )
    assert out["components"]["terminal"] == pytest.approx(out["terminal_reward"])


def test_dense_components_sum_to_dense_reward():
    out = _step(
        rc0=0.1,
        rc1=0.2,
        v_end=1.0,
        metrics={"max_abs_a_long": 4.0},
        wrong_route_event=True,
        stuck_event=True,
    )
    dense_components = {
        key: value
        for key, value in out["components"].items()
        if key != "terminal"
    }
    assert sum(dense_components.values()) == pytest.approx(out["dense_reward"])
    assert out["components"]["wrong_route"] == pytest.approx(-2.0)
    assert out["components"]["stuck"] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# RewardTracker: wrong_route
# ---------------------------------------------------------------------------
def test_wrong_route_is_one_shot_across_tracker_updates():
    tracker = RewardTracker()
    assert tracker.update(0.50, 0.50, 5.0)["wrong_route_event"] is False
    assert tracker.update(0.50, 0.40, 5.0)["wrong_route_event"] is True
    assert tracker.update(0.40, 0.30, 5.0)["wrong_route_event"] is False
    assert tracker.update(0.30, 0.10, 5.0)["wrong_route_event"] is False
    tracker.reset()
    assert tracker.update(0.50, 0.40, 5.0)["wrong_route_event"] is True


def test_wrong_route_threshold_is_strictly_greater_than_drop():
    tracker = RewardTracker()
    assert tracker.update(0.50, 0.45, 5.0)["wrong_route_event"] is False
    tracker.reset()
    assert tracker.update(0.50, 0.449, 5.0)["wrong_route_event"] is True


# ---------------------------------------------------------------------------
# RewardTracker: stuck
# ---------------------------------------------------------------------------
def test_stuck_triggers_at_window_and_only_once():
    tracker = RewardTracker()
    for _ in range(19):
        assert tracker.update(0.10, 0.10, 0.0)["stuck_event"] is False
    assert tracker.update(0.10, 0.10, 0.0)["stuck_event"] is True
    for _ in range(21):
        assert tracker.update(0.10, 0.10, 0.0)["stuck_event"] is False

    # Progress after the penalty does not re-arm the one-shot event.
    assert tracker.update(0.10, 0.20, 0.0)["stuck_event"] is False
    for _ in range(40):
        assert tracker.update(0.20, 0.20, 0.0)["stuck_event"] is False


def test_stuck_run_resets_on_route_progress():
    tracker = RewardTracker()
    for _ in range(19):
        assert tracker.update(0.10, 0.10, 0.0)["stuck_event"] is False
    # A single progress step resets the consecutive run.
    assert tracker.update(0.10, 0.11, 0.0)["stuck_event"] is False
    for _ in range(19):
        assert tracker.update(0.11, 0.11, 0.0)["stuck_event"] is False
    assert tracker.update(0.11, 0.11, 0.0)["stuck_event"] is True


def test_stuck_run_resets_on_speed_above_threshold():
    tracker = RewardTracker()
    for _ in range(19):
        assert tracker.update(0.10, 0.10, 0.0)["stuck_event"] is False
    assert tracker.update(0.10, 0.10, 1.0)["stuck_event"] is False
    for _ in range(19):
        assert tracker.update(0.10, 0.10, 0.0)["stuck_event"] is False
    assert tracker.update(0.10, 0.10, 0.0)["stuck_event"] is True


def test_speed_at_threshold_is_not_stuck():
    tracker = RewardTracker()
    for _ in range(40):
        assert tracker.update(0.10, 0.10, 0.1)["stuck_event"] is False


def test_stuck_counts_rc_decreases_as_no_progress():
    tracker = RewardTracker()
    for step in range(20):
        out = tracker.update(0.50 - 0.001 * step, 0.499 - 0.001 * step, 0.0)
        assert out["stuck_event"] is (step == 19)


def test_tracker_config_is_used():
    tracker = RewardTracker(RewardConfig(stuck_window_steps=3, stuck_speed=0.5))
    assert tracker.update(0.0, 0.0, 0.4)["stuck_event"] is False
    assert tracker.update(0.0, 0.0, 0.4)["stuck_event"] is False
    assert tracker.update(0.0, 0.0, 0.4)["stuck_event"] is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_metric_raises_value_error(bad):
    with pytest.raises(ValueError):
        _step(metrics={"max_abs_a_long": bad})


def test_non_finite_inputs_raise_value_error():
    with pytest.raises(ValueError):
        _step(rc1=math.nan)
    with pytest.raises(ValueError):
        RewardTracker().update(0.0, math.inf, 1.0)


def test_invalid_action_zeroes_dense_reward():
    out = compute_step_reward(
        0.0, 1.0, 0.0, invalid_action=True, wrong_route_event=True,
        stuck_event=True,
    )
    assert out["dense_reward"] == 0.0
    assert out["terminal_reward"] == -20.0
    assert out["reward"] == -20.0


def test_real_metrics_non_scalar_keys_are_ignored():
    metrics = {
        "max_abs_a_long": 2.5,
        "max_abs_a_lat": 3.5,
        "max_abs_jerk_long": 6.0,
        "out_of_bounds_components": [],
        "codes": ("X",),
    }
    out = compute_step_reward(0.0, 0.0, 5.0, metrics)
    assert out["dense_reward"] < 0.0


def test_info_flag_union_keys():
    from pipeline.train.rl_reward import info_flag
    assert info_flag({"crash_vehicle": True}, "crash", "crash_vehicle")
    assert info_flag({"crash": 1}, "crash")
    assert not info_flag({"crash": False}, "crash")


def test_aggregate_substep_events_ors_substeps():
    from pipeline.train.rl_reward import aggregate_substep_events
    infos = [
        {"crash": False},
        {"crash_vehicle": True},
        {"out_of_road": True},
        {"arrive_dest": True},
    ]
    ev = aggregate_substep_events(infos, rc0=0.1, rc1=0.2, v_end=3.0, timeout=False)
    assert ev.collision and ev.off_road and ev.arrive_dest
    assert not ev.timeout
