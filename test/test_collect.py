"""Contract tests for the schema-v2 expert BC collector.

These tests do not build MetaDrive or run a collection.  They exercise the
pure packing/schema helpers with synthetic v2 history tensors produced by
``stack_history`` (no env required) and check the provisional clean limits.
"""

from __future__ import annotations

import numpy as np

from dataset import collect
from env.obs import (
    HISTORY_FRAMES,
    NAV_DIM,
    _empty_history,
    stack_history,
)
from net.action_space import HARD_LIMITS

# Independent list of per-row fields (kept explicit so a dropped field fails).
EXPECTED_ROW_KEYS = {
    "action",
    "action_lift",
    "residual_xy",
    "residual_xy_norm",
    "residual_dpsi",
    "residual_v",
    "expert_steer",
    "expert_throttle",
    "expert_lookahead_m",
    "expert_static_gap_m",
    "expert_target_lane_id",
    "expert_anchor_lane_id",
    "expert_ref_lanes_id",
    "expert_next_ref_lanes_id",
    "expert_used_centerline",
    "expert_used_static_scan",
    "expert_used_future_state",
    "expert_lane_change_decision",
    "privileged_expert_sample",
    "source",
    "feasible",
    "comfortable",
    "v_interval_low",
    "v_interval_high",
    "v_interval_nonempty",
    "v1_in_interval",
    "label_pass",
    "endpoint_error_position_m",
    "endpoint_error_heading_rad",
    "endpoint_error_speed_mps",
    "endpoint_error_raw_position_m",
    "endpoint_error_raw_heading_rad",
    "endpoint_error_raw_speed_mps",
    "path_length_m",
    "chord_length_m",
    "start_accel_mps2",
    "end_accel_mps2",
    "max_abs_a_long",
    "max_abs_a_lat",
    "max_abs_kappa",
    "max_abs_kappa_rate",
    "max_abs_jerk_long",
    "max_abs_jerk_lat",
    "stop_class",
    "action_in_bounds",
    "out_of_bounds_components",
    "codes_summary",
    "n_codes",
    "label_source",
    "label_valid",
    "n_substeps",
    "step",
    "episode_id",
    "scene_seed",
    "done",
    "success",
    "collision",
    "off_road",
    "wrong_route",
    "stuck",
    "timeout",
    "termination_id",
    "expert_plan_valid",
    "expert_failed",
}

FLOAT32_KEYS = {
    "residual_xy_norm",
    "residual_dpsi",
    "residual_v",
    "expert_steer",
    "expert_throttle",
    "expert_lookahead_m",
    "expert_static_gap_m",
    "v_interval_low",
    "v_interval_high",
    "endpoint_error_position_m",
    "endpoint_error_heading_rad",
    "endpoint_error_speed_mps",
    "endpoint_error_raw_position_m",
    "endpoint_error_raw_heading_rad",
    "endpoint_error_raw_speed_mps",
    "path_length_m",
    "chord_length_m",
    "start_accel_mps2",
    "end_accel_mps2",
    "max_abs_a_long",
    "max_abs_a_lat",
    "max_abs_kappa",
    "max_abs_kappa_rate",
    "max_abs_jerk_long",
    "max_abs_jerk_lat",
}
INT8_KEYS = {
    "expert_used_centerline",
    "expert_used_static_scan",
    "expert_used_future_state",
    "expert_lane_change_decision",
    "privileged_expert_sample",
    "source",
    "feasible",
    "comfortable",
    "v_interval_nonempty",
    "v1_in_interval",
    "label_pass",
    "stop_class",
    "action_in_bounds",
    "label_source",
    "label_valid",
    "done",
    "success",
    "collision",
    "off_road",
    "wrong_route",
    "stuck",
    "timeout",
    "expert_plan_valid",
    "expert_failed",
}
INT16_KEYS = {"n_codes", "n_substeps", "termination_id"}
INT64_KEYS = {
    "expert_target_lane_id",
    "expert_anchor_lane_id",
    "expert_ref_lanes_id",
    "expert_next_ref_lanes_id",
    "step",
    "episode_id",
    "scene_seed",
}


def _synthetic_hist():
    """One-frame v2 history built without MetaDrive (all slots empty)."""
    template = _empty_history()
    frame = {}
    for key, value in template.items():
        if key == "hist_valid":
            continue
        frame[key] = np.array(np.asarray(value)[0], copy=True)
    frame["meta"] = {
        "scene_seed": 7,
        "dt": 0.5,
        "timestamp": 0.0,
        "source": "REAL",
    }
    return stack_history([frame], [np.zeros(3, dtype=np.float64)])


def _synthetic_rows(hist, n=2):
    rows = []
    for i in range(n):
        row = {key: np.array(hist[key], copy=True) for key in collect.HISTORY_KEYS}
        for key, dtype in collect.ROW_DTYPES.items():
            np_dtype = np.dtype(dtype)
            if np_dtype.kind == "U":
                row[key] = "TEST"
            elif key in ("action", "action_lift"):
                row[key] = np.zeros(4, dtype=dtype)
            elif key == "residual_xy":
                row[key] = np.zeros(2, dtype=dtype)
            elif np_dtype.kind in "iu":
                row[key] = dtype(i)
            else:
                row[key] = dtype(0.0)
        rows.append(row)
    return rows


def test_packed_output_keys_shapes_and_dtypes():
    hist = _synthetic_hist()
    assert set(hist) == set(collect.HISTORY_KEYS) | {"meta"}
    assert set(collect.ROW_DTYPES) == EXPECTED_ROW_KEYS

    packed = collect._pack_rows(_synthetic_rows(hist, n=2))
    assert set(packed) == set(collect.HISTORY_KEYS) | EXPECTED_ROW_KEYS

    for key in collect.HISTORY_KEYS:
        assert packed[key].shape == (2,) + hist[key].shape, key
        assert packed[key].dtype == hist[key].dtype, key

    assert packed["action"].shape == (2, 4)
    assert packed["action"].dtype == np.float32
    assert packed["action_lift"].shape == (2, 4)
    assert packed["residual_xy"].shape == (2, 2)
    assert packed["residual_xy"].dtype == np.float32

    for key in FLOAT32_KEYS:
        assert packed[key].dtype == np.float32, key
        assert packed[key].shape == (2,), key
    for key in INT8_KEYS:
        assert packed[key].dtype == np.int8, key
        assert packed[key].shape == (2,), key
    for key in INT16_KEYS:
        assert packed[key].dtype == np.int16, key
        assert packed[key].shape == (2,), key
    for key in INT64_KEYS:
        assert packed[key].dtype == np.int64, key
        assert packed[key].shape == (2,), key

    for key in ("codes_summary", "out_of_bounds_components"):
        assert packed[key].dtype.kind == "U", key
        assert packed[key].shape == (2,), key


def test_history_features_exclude_privileged_metadata():
    # Privileged expert audit / outcome fields are side arrays only.
    forbidden_prefixes = ("expert_", "privileged_")
    for key in collect.HISTORY_KEYS:
        assert not key.startswith(forbidden_prefixes), key
        assert "target_lane" not in key, key
        assert "lane_level" not in key, key

    # Road-level nav tensor has no ID slot; IDs live in separate metadata arrays.
    assert "nav" in collect.HISTORY_KEYS
    assert "nav_road_ids" in collect.HISTORY_KEYS
    hist = _synthetic_hist()
    assert hist["nav"].shape == (HISTORY_FRAMES, NAV_DIM)
    assert hist["nav"].dtype == np.float32
    assert hist["nav_road_ids"].dtype == np.int64
    assert hist["nav_route_id"].shape == (HISTORY_FRAMES,)

    # Route / target IDs must not appear inside the feature tensors.
    for feature in ("od", "ld", "nav", "ego"):
        assert hist[feature].dtype == np.float32, feature


def test_backfill_outcomes_applies_to_all_rows():
    rows = [{}, {}, {}]
    collect._backfill_outcomes(
        rows,
        termination="collision",
        done=True,
        success=0,
        collision=1,
        off_road=0,
    )
    for row in rows:
        assert row["collision"] == 1
        assert row["success"] == 0
        assert row["expert_plan_valid"] == 0
        assert row["expert_failed"] == 1
        assert row["timeout"] == 0
        assert row["termination_id"] == collect.TERMINATION_IDS["collision"]
    assert [int(row["done"]) for row in rows] == [0, 0, 1]

    rows = [{}, {}]
    collect._backfill_outcomes(
        rows,
        termination="arrive",
        done=True,
        success=1,
        collision=0,
        off_road=0,
    )
    for row in rows:
        assert row["success"] == 1
        assert row["expert_plan_valid"] == 1
        assert row["expert_failed"] == 0
        assert row["termination_id"] == collect.TERMINATION_IDS["arrive"]
    assert [int(row["done"]) for row in rows] == [0, 1]

    rows = [{}]
    collect._backfill_outcomes(
        rows,
        termination="max_step",
        done=False,
        success=0,
        collision=0,
        off_road=0,
    )
    assert rows[0]["timeout"] == 1
    assert rows[0]["termination_id"] == 0
    assert rows[0]["done"] == 0


def test_clean_limits_are_provisional_calibration():
    hard = collect.HARD_LIMITS
    assert (
        hard.a_long_min,
        hard.a_long_max,
        hard.a_lat_max,
        hard.kappa_max,
        hard.kappa_rate_max,
        hard.jerk_long_max,
        hard.jerk_lat_max,
    ) == (-10.0, 3.5, 8.0, 0.345, 1.0, 20.0, 20.0)

    comfort = collect.COMFORT_LIMITS
    assert (
        comfort.a_lat_max,
        comfort.kappa_max,
        comfort.kappa_rate_max,
        comfort.jerk_long_max,
        comfort.jerk_lat_max,
    ) == (3.0, 0.20, 0.05, 5.0, 5.0)



def _reference_lift(
    steer,
    throttle,
    *,
    speed,
    max_steer_angle_rad,
    wheelbase,
    accel_action_scale,
    brake_action_scale,
):
    """Copy of dataset/collect_dagger.py:34-49 for regression comparison."""
    delta = float(steer) * float(max_steer_angle_rad)
    a = float(throttle) * (
        accel_action_scale if throttle >= 0 else brake_action_scale
    )
    v = float(speed)
    x = y = th = 0.0
    dt = 0.1
    for _ in range(5):
        v1 = max(0.0, v + a * dt)
        vm = 0.5 * (v + v1)
        dth = vm / max(wheelbase, 1e-3) * np.tan(delta) * dt
        x += vm * np.cos(th + 0.5 * dth) * dt
        y += vm * np.sin(th + 0.5 * dth) * dt
        th += dth
        v = v1
    return np.array([x, y, th, v], dtype=np.float32)


def test_lift_action_matches_collect_dagger_logic():
    kwargs = dict(
        speed=8.0,
        max_steer_angle_rad=0.6981317,
        wheelbase=2.46894,
        accel_action_scale=3.0,
        brake_action_scale=3.0,
    )
    for steer, throttle in ((0.0, 1.0), (0.25, -0.4), (-0.6, 0.2)):
        lift = collect._lift_action(steer, throttle, **kwargs)
        ref = _reference_lift(steer, throttle, **kwargs)
        np.testing.assert_allclose(lift, ref, rtol=0.0, atol=1e-6)
