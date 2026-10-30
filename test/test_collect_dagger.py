"""Contract tests for the schema-v2 DAgger collector.

The synthetic tests do not build MetaDrive.  They exercise the pure schema,
mode-selection, label-composition, audit and fallback helpers; the end-to-end
1-scene smoke is run from the shell with ``tools/venv-python``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dataset import collect_dagger as dagger
from dataset import collect
from env.obs import HISTORY_FRAMES, NAV_DIM, _empty_history, stack_history
from env.action.config import load_action_contract

CLEAN_BOUNDS = load_action_contract().bounds

#: Fields added by this collector on top of the collect row schema.
DAGGER_ROW_KEYS = {
    "action_replay",
    "student_action",
    "executed_action",
    "expert_action",
    "intervened",
    "shield",
    "student_feasible",
    "fallback",
    "round",
    "student_success",
    "student_failure",
}

#: Row fields that are vectors rather than scalars.
VECTOR_KEYS = {
    "action": 4,
    "action_lift": 4,
    "action_replay": 4,
    "student_action": 4,
    "executed_action": 4,
    "expert_action": 2,
    "residual_xy": 2,
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
        row = {key: np.array(hist[key], copy=True) for key in dagger.HISTORY_KEYS}
        for key, dtype in dagger.ROW_DTYPES.items():
            np_dtype = np.dtype(dtype)
            if np_dtype.kind == "U":
                row[key] = "TEST"
            elif key in VECTOR_KEYS:
                row[key] = np.full(VECTOR_KEYS[key], i, dtype=np_dtype)
            elif np_dtype.kind in "iu":
                row[key] = np_dtype.type(i)
            else:
                row[key] = np_dtype.type(0.0)
        rows.append(row)
    return rows


def test_schema_extends_collect_without_dropping_fields():
    assert dagger.HISTORY_KEYS == collect.HISTORY_KEYS
    assert set(dagger.ROW_DTYPES) == set(collect.ROW_DTYPES) | DAGGER_ROW_KEYS
    # Every historical v2 field keeps its dtype.
    for key, dtype in collect.ROW_DTYPES.items():
        assert dagger.ROW_DTYPES[key] is dtype, key


def test_packed_output_keys_shapes_and_dtypes():
    hist = _synthetic_hist()
    assert set(hist) == set(dagger.HISTORY_KEYS) | {"meta"}

    packed = dagger._pack_rows(_synthetic_rows(hist, n=2))
    assert set(packed) == set(dagger.HISTORY_KEYS) | set(dagger.ROW_DTYPES)

    for key in dagger.HISTORY_KEYS:
        assert packed[key].shape == (2,) + hist[key].shape, key
        assert packed[key].dtype == hist[key].dtype, key

    assert packed["action"].shape == (2, 4)
    assert packed["action"].dtype == np.float32
    assert packed["action_replay"].shape == (2, 4)
    assert packed["student_action"].shape == (2, 4)
    assert packed["executed_action"].shape == (2, 4)
    assert packed["expert_action"].shape == (2, 2)
    assert packed["residual_xy"].shape == (2, 2)
    assert packed["residual_xy"].dtype == np.float32
    assert packed["round"].dtype == np.int32
    assert packed["intervened"].dtype == np.int8

    # All dtype declarations are honoured by the packed arrays.
    for key, dtype in dagger.ROW_DTYPES.items():
        expected = np.dtype(dtype)
        if expected.kind == "U":
            assert packed[key].dtype.kind == "U", key
        else:
            assert packed[key].dtype == expected, key


def test_history_features_exclude_dagger_and_privileged_fields():
    forbidden = ("expert_", "privileged_", "student_", "action_replay")
    for key in dagger.HISTORY_KEYS:
        assert not key.startswith(forbidden), key
        assert "target_lane" not in key, key
        assert "ref_lanes" not in key, key

    hist = _synthetic_hist()
    assert hist["nav"].shape == (HISTORY_FRAMES, NAV_DIM)
    assert hist["nav_road_ids"].dtype == np.int64
    for feature in ("od", "ld", "nav", "ego"):
        assert hist[feature].dtype == np.float32, feature


class _FakeRng:
    def __init__(self, value: float):
        self.value = float(value)

    def random(self) -> float:
        return self.value


def test_mode_selection_probability_and_shield():
    # Boundary ratios are deterministic independent of the draw.
    assert dagger._select_mode(
        shield=False, expert_ratio=0.0, rng=_FakeRng(0.0)
    ) == (False, False)
    assert dagger._select_mode(
        shield=False, expert_ratio=1.0, rng=_FakeRng(0.999)
    ) == (True, False)
    assert dagger._select_mode(
        shield=False, expert_ratio=0.5, rng=_FakeRng(0.49)
    ) == (True, False)
    assert dagger._select_mode(
        shield=False, expert_ratio=0.5, rng=_FakeRng(0.50)
    ) == (False, False)
    # Shield forces expert replay even when the random draw would say student.
    assert dagger._select_mode(
        shield=True, expert_ratio=0.0, rng=_FakeRng(0.99)
    ) == (True, True)


def test_episode_rng_seed_is_deterministic_per_spec():
    class _Spec:
        def __init__(self, seed):
            self.seed = seed

    a = dagger._episode_rng_seed(3, _Spec(1001), 2)
    b = dagger._episode_rng_seed(3, _Spec(1001), 2)
    c = dagger._episode_rng_seed(3, _Spec(1002), 2)
    assert a == b
    assert a != c


def test_compose_label_measured_replay():
    lift = np.array([1.0, 2.0, 0.10, 3.0])
    replay = np.array([1.2, 1.8, 0.05, 3.5])
    out = dagger._compose_label(
        replay=True,
        action_lift=lift,
        action_replay=replay,
        n_substeps=5,
        v0=3.0,
    )
    np.testing.assert_allclose(out["action"], replay)
    np.testing.assert_allclose(out["action_replay"], replay)
    assert int(out["label_source"]) == 1
    assert int(out["label_valid"]) == 1
    assert int(out["n_substeps"]) == 5
    np.testing.assert_allclose(out["residual_xy"], [-0.2, 0.2], atol=1e-6)
    assert out["residual_xy_norm"] == pytest.approx(np.hypot(0.2, 0.2))
    assert out["residual_dpsi"] == pytest.approx(0.05)
    assert out["residual_v"] == pytest.approx(-0.5)


def test_compose_label_partial_replay_is_invalid_but_keeps_partial_measurement():
    lift = np.array([1.0, 2.0, 0.10, 3.0])
    replay = np.array([0.6, 1.0, 0.03, 2.0])
    out = dagger._compose_label(
        replay=True,
        action_lift=lift,
        action_replay=replay,
        n_substeps=3,
        v0=3.0,
    )
    # Literal label rule: measured replay is still the action, but it is not a
    # full 0.5 s label.
    np.testing.assert_allclose(out["action"], replay)
    assert int(out["label_source"]) == 1
    assert int(out["label_valid"]) == 0
    assert int(out["n_substeps"]) == 3
    assert np.isnan(out["residual_xy"]).all()
    assert np.isnan(out["residual_xy_norm"])
    assert np.isnan(out["residual_dpsi"])
    assert np.isnan(out["residual_v"])


def test_compose_label_student_uses_lift_and_has_no_measured_residual():
    lift = np.array([4.0, 0.1, 0.02, 6.0])
    out = dagger._compose_label(
        replay=False,
        action_lift=lift,
        action_replay=None,
        n_substeps=5,
        v0=5.0,
    )
    np.testing.assert_allclose(out["action"], lift)
    assert np.isnan(out["action_replay"]).all()
    assert int(out["label_source"]) == 0
    assert int(out["label_valid"]) == 1
    assert np.isnan(out["residual_xy"]).all()
    assert np.isnan(out["residual_xy_norm"])


def test_compose_label_partial_student_is_invalid():
    out = dagger._compose_label(
        replay=False,
        action_lift=np.array([4.0, 0.1, 0.02, 6.0]),
        action_replay=None,
        n_substeps=4,
        v0=5.0,
    )
    assert int(out["label_source"]) == 0
    assert int(out["label_valid"]) == 0


def test_compose_label_nonfinite_replay_falls_back_to_lift():
    lift = np.array([4.0, 0.1, 0.02, 6.0])
    out = dagger._compose_label(
        replay=True,
        action_lift=lift,
        action_replay=np.array([np.nan, 0.0, 0.0, 6.0]),
        n_substeps=5,
        v0=5.0,
    )
    np.testing.assert_allclose(out["action"], lift)
    assert np.isnan(out["action_replay"]).all()
    assert int(out["label_source"]) == 0
    assert int(out["label_valid"]) == 0


def test_audit_fields_mark_privileged_flags_and_source():
    info = {
        "pp_target_lane": "('0S0', '0S1', 0)",
        "pp_anchor_lane": "('0S0', '0S1', 1)",
        "pp_ref_lanes": ["('0S0', '0S1', 0)"],
        "pp_next_ref_lanes": ["('0S1', '0S2', 0)"],
    }
    audit = dagger._audit_fields(11, info, source=1)
    assert int(audit["privileged_expert_sample"]) == 1
    assert int(audit["expert_used_centerline"]) == 1
    assert int(audit["expert_used_static_scan"]) == 1
    assert int(audit["expert_used_future_state"]) == 0
    assert int(audit["source"]) == 1
    assert int(audit["expert_lane_change_decision"]) == 1
    assert int(audit["expert_target_lane_id"]) != 0
    assert int(audit["expert_anchor_lane_id"]) != 0
    assert int(audit["expert_ref_lanes_id"]) != 0
    assert int(audit["expert_next_ref_lanes_id"]) != 0

    same = dagger._audit_fields(
        11, {"pp_target_lane": "lane", "pp_anchor_lane": "lane"}, source=1
    )
    assert int(same["expert_lane_change_decision"]) == 0


def test_fallback_projects_v1_or_brakes_but_never_executes_infeasible():
    # Requested action is within CLEAN_BOUNDS but PSTL-infeasible (v_next outside
    # the hard-feasible interval); projection keeps the geometry.
    v0 = 5.0
    requested = np.array([2.0, 0.0, 0.0, 12.0])
    assert not dagger._trajectory(requested, v0).feasible
    executed, traj, name = dagger._resolve_student_action(requested, v0)
    assert name == "v1_projection"
    assert traj is not None and traj.feasible
    assert np.all(np.isfinite(executed))
    assert executed[3] < requested[3]
    np.testing.assert_allclose(executed[:3], requested[:3])

    # Zero-length path with speed cannot be repaired by v_next projection.
    requested = np.array([0.0, 0.0, 0.0, 8.0])
    v0 = 8.0
    assert not dagger._trajectory(requested, v0).feasible
    executed, traj, name = dagger._resolve_student_action(requested, v0)
    assert name == "safe_brake"
    assert traj is not None and traj.feasible
    assert executed[3] == pytest.approx(v0 - 6.0 * dagger.DECISION_DT)

    # A feasible student action is executed unchanged.
    requested = np.array([3.0, 0.0, 0.0, 6.0])
    v0 = 6.0
    assert dagger._trajectory(requested, v0).feasible
    executed, _traj, name = dagger._resolve_student_action(requested, v0)
    assert name == "none"
    np.testing.assert_allclose(executed, requested)


def test_backfill_outcomes_keeps_label_validity_separate():
    rows = [{}, {}]
    for row in rows:
        row["expert_plan_valid"] = np.int8(1)
        row["label_valid"] = np.int8(1)
    dagger._backfill_outcomes(
        rows,
        termination="collision",
        done=True,
        success=0,
        collision=1,
        off_road=0,
        wrong_route=0,
        stuck=0,
    )
    for row in rows:
        assert int(row["success"]) == 0
        assert int(row["collision"]) == 1
        assert int(row["student_success"]) == 0
        assert int(row["student_failure"]) == 1
        assert int(row["timeout"]) == 0
        assert int(row["termination_id"]) == dagger.TERMINATION_IDS["collision"]
        # A failed student episode does not invalidate a valid expert label.
        assert int(row["expert_plan_valid"]) == 1
        assert int(row["label_valid"]) == 1
    assert [int(row["done"]) for row in rows] == [0, 1]

    rows = [{}]
    dagger._backfill_outcomes(
        rows,
        termination="max_step",
        done=False,
        success=0,
        collision=0,
        off_road=0,
    )
    assert int(rows[0]["timeout"]) == 1
    assert int(rows[0]["student_failure"]) == 1
    assert int(rows[0]["done"]) == 0


def test_new_bounds_are_used_for_student_actions():
    # The collector must denormalize with the canonical action box.
    assert dagger.CLEAN_BOUNDS == CLEAN_BOUNDS
    assert dagger.CLEAN_BOUNDS.name == "clean"
