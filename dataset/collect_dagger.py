"""Schema-v2 DAgger collection with measured expert replay.

This collector keeps the
v2 observation/audit/row contract of :mod:`dataset.collect` and adds the
DAgger roll-in fields:

* the **student** (``net.model.StructuredPlanner``) is rolled in through the
  same 0.5 s decision / 5 x 0.1 s PSTL execution layer as ``pipeline.eval``
  (``CLEAN_BOUNDS.denormalize(action_mu[0])`` + :class:`env.tracking.ExactTracker`);
* at every decision state the expert (``PurePursuitIDMPolicy``) is queried and
  its ``(steer, throttle)`` is also audited with a 5 x 0.1 s kinematic
  bicycle (``action_lift``);
* with probability ``--expert-ratio`` (or when the TTC shield triggers) the
  expert instead drives the segment online and the **measured** endpoint
  ``(dx, dy, dpsi, v_next)`` becomes the label (``action_replay``);
* ``action = action_replay`` on replay rows, ``action = action_lift`` on student
  rows.  ``label_source`` is 1 for a measured replay label and 0 for a lifted
  label; ``label_valid`` is 0 when the segment did not complete 5 substeps
  (mid-segment termination / budget cutoff) or the endpoint is not finite.

Privileged expert audit fields are copied from :mod:`dataset.collect` and
live only in side arrays.  They never enter the student OD/LD/Nav/Ego features.
The expert is the only consumer of ``current_ref_lanes`` / target-lane state;
the student observation is built exclusively by
``env.obs.build_observation``, which does not read ``get_checkpoints``,
``navi_arrow_dir`` or ``current_ref_lanes``.

Rows carry the student roll-in episode outcome (``success`` / ``collision`` /
``off_road`` / ``wrong_route`` / ``stuck`` / ``timeout``) backfilled to every
row, while ``expert_plan_valid`` follows the *label* validity, so recovery rows
still teach the expert plan even when the student episode fails.

Artifacts (same names as :mod:`dataset.collect`)::

    expert_bc.npz
    expert_bc.meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from env.metadrive_env import build_env
from env.obs import (
    HISTORY_FRAMES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    _stable_hash,
    build_observation,
    stack_history,
)
from env.scenario.spec import load_specs
from env.tracking import ExactTracker
from net.action_space import (
    CLEAN_BOUNDS,
    COMFORT_LIMITS,
    HARD_LIMITS,
    action_to_trajectory,
    feasible_v1_interval,
)
from net.model import H, HEADS, SPATIAL_LAYERS, StructuredPlanner
from pipeline.checkpointing import load_model_from_checkpoint

DECISION_DT = 0.5
SIM_DT = 0.1
STEPS_PER_POLICY = 5
ACTION_DIM = 4
HZ = 10

#: Stuck: >20 consecutive 0.1 s sub-steps below this speed (same as eval).
STUCK_SPEED_MPS = 0.1
STUCK_STEPS = 20
#: Conservative wrong-route signal: route_completion drops by more than this
#: within one 0.5 s decision segment (same as eval).
WRONG_ROUTE_DROP = 0.05

# ---------------------------------------------------------------------------
# Canonical PSTL limits
# ---------------------------------------------------------------------------
HARD_LIMITS_SOURCE = "net.action_space.HARD_LIMITS"

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
#: History tensors produced by ``stack_history`` (``meta`` is not an array).
HISTORY_KEYS: Tuple[str, ...] = (
    "od",
    "od_mask",
    "od_ids",
    "od_anchor",
    "od_taxonomy",
    "od_age",
    "ld",
    "ld_mask",
    "ld_ids",
    "ld_road_ids",
    "ld_anchor",
    "ld_chunk_s",
    "nav",
    "nav_mask",
    "nav_road_mask",
    "nav_road_ids",
    "nav_route_id",
    "ego",
    "ego_mask",
    "pose",
    "hist_valid",
)

#: Per-row arrays/scalars and their on-disk dtypes.
ROW_DTYPES: Dict[str, Any] = {
    # labels + lift/replay audit
    "action": np.float32,
    "action_lift": np.float32,
    "action_replay": np.float32,
    "student_action": np.float32,
    "executed_action": np.float32,
    "expert_action": np.float32,
    "residual_xy": np.float32,
    "residual_xy_norm": np.float32,
    "residual_dpsi": np.float32,
    "residual_v": np.float32,
    "expert_steer": np.float32,
    "expert_throttle": np.float32,
    "expert_lookahead_m": np.float32,
    "expert_static_gap_m": np.float32,
    # privileged expert audit (side channel; not student input)
    "expert_target_lane_id": np.int64,
    "expert_anchor_lane_id": np.int64,
    "expert_ref_lanes_id": np.int64,
    "expert_next_ref_lanes_id": np.int64,
    "expert_used_centerline": np.int8,
    "expert_used_static_scan": np.int8,
    "expert_used_future_state": np.int8,
    "expert_lane_change_decision": np.int8,
    "privileged_expert_sample": np.int8,
    "source": np.int8,
    # PSTL label diagnostics
    "feasible": np.int8,
    "comfortable": np.int8,
    "v_interval_low": np.float32,
    "v_interval_high": np.float32,
    "v_interval_nonempty": np.int8,
    "v1_in_interval": np.int8,
    "label_pass": np.int8,
    "endpoint_error_position_m": np.float32,
    "endpoint_error_heading_rad": np.float32,
    "endpoint_error_speed_mps": np.float32,
    "endpoint_error_raw_position_m": np.float32,
    "endpoint_error_raw_heading_rad": np.float32,
    "endpoint_error_raw_speed_mps": np.float32,
    "path_length_m": np.float32,
    "chord_length_m": np.float32,
    "start_accel_mps2": np.float32,
    "end_accel_mps2": np.float32,
    "max_abs_a_long": np.float32,
    "max_abs_a_lat": np.float32,
    "max_abs_kappa": np.float32,
    "max_abs_kappa_rate": np.float32,
    "max_abs_jerk_long": np.float32,
    "max_abs_jerk_lat": np.float32,
    "stop_class": np.int8,
    "action_in_bounds": np.int8,
    "out_of_bounds_components": str,
    "codes_summary": str,
    "n_codes": np.int16,
    # DAgger / label / episode metadata
    "label_source": np.int8,
    "label_valid": np.int8,
    "n_substeps": np.int16,
    "intervened": np.int8,
    "shield": np.int8,
    "student_feasible": np.int8,
    "fallback": np.int8,
    "round": np.int32,
    "step": np.int64,
    "episode_id": np.int64,
    "scene_seed": np.int64,
    "done": np.int8,
    "terminal_failed": np.int8,
    "success": np.int8,
    "collision": np.int8,
    "off_road": np.int8,
    "wrong_route": np.int8,
    "stuck": np.int8,
    "timeout": np.int8,
    "student_success": np.int8,
    "student_failure": np.int8,
    "termination_id": np.int16,
    "expert_plan_valid": np.int8,
    "expert_failed": np.int8,
}

TERMINATION_IDS = {
    "max_step": 0,
    "timeout": 0,
    "arrive": 1,
    "collision": 2,
    "out_of_road": 3,
    "terminal": 4,
    "invalid_action": 5,
}

#: ``fallback`` column codes (the student action was not executed as-is).
FALLBACK_IDS = {
    "none": 0,
    "v1_projection": 1,
    "safe_brake": 2,
    "nonfinite_safe_brake": 3,
    "safe_brake_infeasible": 4,
    "not_evaluated": -1,
}


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _expert_class():
    from env.expert.pure_pursuit_idm import PurePursuitIDMPolicy

    return PurePursuitIDMPolicy


def _safe_float(value: Any, fallback: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _episode_rng_seed(seed: int, spec: Any, round_id: int) -> int:
    """Deterministic per-episode DAgger RNG seed (independent of worker order)."""
    scene_seed = int(getattr(spec, "seed", 0))
    mixed = (
        int(seed) * 1_000_003
        + scene_seed * 97
        + int(round_id) * 7919
    )
    return int(abs(mixed) % (2**32))


# ---------------------------------------------------------------------------
# Expert 4D lift (same as collect / collect_dagger)
# ---------------------------------------------------------------------------
def _lift_action(
    steer: float,
    throttle: float,
    *,
    speed: float,
    max_steer_angle_rad: float,
    wheelbase: float,
    accel_action_scale: float,
    brake_action_scale: float,
    substeps: int = STEPS_PER_POLICY,
    dt: float = SIM_DT,
) -> np.ndarray:
    """Hold ``(steer, throttle)`` through ``substeps`` kinematic-bicycle steps."""
    delta = float(steer) * float(max_steer_angle_rad)
    a = float(throttle) * (accel_action_scale if throttle >= 0 else brake_action_scale)
    v = float(speed)
    x = y = th = 0.0
    for _ in range(int(substeps)):
        v1 = max(0.0, v + a * dt)
        vm = 0.5 * (v + v1)
        dth = vm / max(wheelbase, 1e-3) * math.tan(delta) * dt
        x += vm * math.cos(th + 0.5 * dth) * dt
        y += vm * math.sin(th + 0.5 * dth) * dt
        th += dth
        v = v1
    return np.array([x, y, _wrap(th), v], dtype=np.float32)


# ---------------------------------------------------------------------------
# Action / label helpers
# ---------------------------------------------------------------------------
def safe_brake_action(v0: float) -> np.ndarray:
    """Explicit safe-brake action: decelerate at 6 m/s^2, no lateral motion."""
    v0 = max(float(v0), 0.0)
    v1 = max(0.0, v0 - 6.0 * DECISION_DT)
    dx = (v0 + v1) * DECISION_DT / 2.0
    return np.array([dx, 0.0, 0.0, v1], dtype=np.float64)


def _trajectory(action: Sequence[float], v0: float):
    return action_to_trajectory(
        action,
        v0,
        dt=DECISION_DT,
        hz=HZ,
        bounds=CLEAN_BOUNDS,
        limits=HARD_LIMITS,
        comfort_limits=COMFORT_LIMITS,
    )


def _trajectory_or_none(action: Sequence[float], v0: float):
    try:
        return _trajectory(action, v0)
    except (TypeError, ValueError):
        return None


def _resolve_student_action(
    action: Sequence[float], v0: float
) -> Tuple[np.ndarray, Any, str]:
    """Return ``(executed_action, trajectory, fallback_name)``.

    The student action is never silently executed when its PSTL trajectory is
    infeasible.  The fallback order is: keep geometry and project ``v_next`` to
    the hard-feasible interval, then an explicit safe brake.
    """
    requested = np.asarray(action, dtype=np.float64).reshape(-1)
    if requested.shape != (ACTION_DIM,) or not np.all(np.isfinite(requested)):
        brake = safe_brake_action(v0)
        return brake, _trajectory_or_none(brake, v0), "nonfinite_safe_brake"

    traj = _trajectory_or_none(requested, v0)
    if traj is not None and traj.feasible:
        return requested, traj, "none"

    try:
        interval = feasible_v1_interval(
            requested,
            v0,
            dt=DECISION_DT,
            bounds=CLEAN_BOUNDS,
            limits=HARD_LIMITS,
        )
    except (TypeError, ValueError):
        interval = None
    if interval is not None and interval.nonempty:
        candidate = requested.copy()
        candidate[3] = float(np.clip(candidate[3], interval.low, interval.high))
        candidate = np.asarray(CLEAN_BOUNDS.clip(candidate), dtype=np.float64)
        candidate_traj = _trajectory_or_none(candidate, v0)
        if candidate_traj is not None and candidate_traj.feasible:
            return candidate, candidate_traj, "v1_projection"

    brake = safe_brake_action(v0)
    brake_traj = _trajectory_or_none(brake, v0)
    if brake_traj is not None and brake_traj.feasible:
        return brake, brake_traj, "safe_brake"
    return brake, brake_traj, "safe_brake_infeasible"


def _select_mode(
    *, shield: bool, expert_ratio: float, rng: Any
) -> Tuple[bool, bool]:
    """Decide ``(expert_replay, shield)`` for this decision state.

    The random draw is always consumed so the mode sequence is reproducible
    regardless of how often the shield fires.
    """
    shield = bool(shield)
    replay = shield or bool(float(rng.random()) < float(expert_ratio))
    return replay, shield


def _compose_label(
    *,
    replay: bool,
    action_lift: Sequence[float],
    action_replay: Optional[Sequence[float]],
    n_substeps: int,
    v0: float,
) -> Dict[str, Any]:
    """Compose the training label and DAgger audit fields for one row.

    * ``action`` is the measured replay endpoint on replay rows, otherwise the
      lifted expert action (the literal DAgger label rule);
    * ``label_source`` is 1 only for a finite measured replay label;
    * ``label_valid`` requires a full 5-substep segment, a finite label and a
      finite non-negative ``v0``;
    * residuals compare the lift with the measured replay and are NaN when no
      fully measured replay exists (student rows / partial replay).
    """
    lift = np.asarray(action_lift, dtype=np.float64).reshape(-1)
    replay_arr: Optional[np.ndarray] = None
    if action_replay is not None:
        candidate = np.asarray(action_replay, dtype=np.float64).reshape(-1)
        if candidate.shape == (ACTION_DIM,) and np.all(np.isfinite(candidate)):
            replay_arr = candidate

    measured = bool(replay and replay_arr is not None)
    action = replay_arr if measured else lift
    label_source = 1 if measured else 0

    v0_ok = bool(np.isfinite(v0) and float(v0) >= 0.0)
    label_valid = bool(
        n_substeps == STEPS_PER_POLICY
        and np.all(np.isfinite(action))
        and v0_ok
        and (not replay or measured)
    )

    nan4 = np.full(ACTION_DIM, np.nan, dtype=np.float64)
    action_replay_out = replay_arr.copy() if measured else nan4.copy()
    if measured and label_valid:
        residual = lift - action
        residual_xy = residual[:2].copy()
        residual_dpsi = _wrap(float(residual[2]))
        residual_v = float(residual[3])
    else:
        residual_xy = np.full(2, np.nan, dtype=np.float64)
        residual_dpsi = float("nan")
        residual_v = float("nan")

    return {
        "action": np.asarray(action, dtype=np.float64),
        "action_replay": action_replay_out,
        "label_source": np.int8(label_source),
        "label_valid": np.int8(1 if label_valid else 0),
        "n_substeps": np.int16(int(n_substeps)),
        "residual_xy": np.asarray(residual_xy, dtype=np.float32),
        "residual_xy_norm": np.float32(
            float(np.linalg.norm(residual_xy))
            if np.all(np.isfinite(residual_xy))
            else float("nan")
        ),
        "residual_dpsi": np.float32(residual_dpsi),
        "residual_v": np.float32(residual_v),
    }


# ---------------------------------------------------------------------------
# Label diagnostics / audit
# ---------------------------------------------------------------------------
def _label_diagnostics(action: np.ndarray, v0: float) -> Dict[str, Any]:
    traj = _trajectory(action, v0)
    interval = feasible_v1_interval(
        action,
        v0,
        dt=DECISION_DT,
        bounds=CLEAN_BOUNDS,
        limits=HARD_LIMITS,
    )
    v1_in_interval = bool(interval.contains(float(action[3])))
    metrics = traj.metrics
    interval_low = float(interval.low) if interval.nonempty else float("nan")
    interval_high = float(interval.high) if interval.nonempty else float("nan")
    return {
        "feasible": np.int8(bool(traj.feasible)),
        "comfortable": np.int8(bool(traj.comfortable)),
        "v_interval_low": np.float32(interval_low),
        "v_interval_high": np.float32(interval_high),
        "v_interval_nonempty": np.int8(bool(interval.nonempty)),
        "v1_in_interval": np.int8(v1_in_interval),
        "label_pass": np.int8(
            bool(traj.valid and traj.feasible and interval.nonempty and v1_in_interval)
        ),
        "endpoint_error_position_m": np.float32(traj.endpoint_errors["position_m"]),
        "endpoint_error_heading_rad": np.float32(traj.endpoint_errors["heading_rad"]),
        "endpoint_error_speed_mps": np.float32(traj.endpoint_errors["speed_mps"]),
        "endpoint_error_raw_position_m": np.float32(
            traj.endpoint_errors["raw_position_m"]
        ),
        "endpoint_error_raw_heading_rad": np.float32(
            traj.endpoint_errors["raw_heading_rad"]
        ),
        "endpoint_error_raw_speed_mps": np.float32(
            traj.endpoint_errors["raw_speed_mps"]
        ),
        "path_length_m": np.float32(metrics["path_length_m"]),
        "chord_length_m": np.float32(metrics["chord_length_m"]),
        "start_accel_mps2": np.float32(metrics["start_accel_mps2"]),
        "end_accel_mps2": np.float32(metrics["end_accel_mps2"]),
        "max_abs_a_long": np.float32(metrics["max_abs_a_long"]),
        "max_abs_a_lat": np.float32(metrics["max_abs_a_lat"]),
        "max_abs_kappa": np.float32(metrics["max_abs_kappa"]),
        "max_abs_kappa_rate": np.float32(metrics["max_abs_kappa_rate"]),
        "max_abs_jerk_long": np.float32(metrics["max_abs_jerk_long"]),
        "max_abs_jerk_lat": np.float32(metrics["max_abs_jerk_lat"]),
        "stop_class": np.int8(bool(metrics["stop_class"])),
        "action_in_bounds": np.int8(bool(metrics["action_in_bounds"])),
        "out_of_bounds_components": ";".join(
            str(name) for name in metrics.get("out_of_bounds_components", [])
        ),
        "codes_summary": ";".join(str(code) for code in traj.codes),
        "n_codes": np.int16(len(traj.codes)),
    }


def _diagnostics_or_empty(action: np.ndarray, v0: float) -> Dict[str, Any]:
    try:
        return _label_diagnostics(action, v0)
    except (TypeError, ValueError):
        return {
            "feasible": np.int8(0),
            "comfortable": np.int8(0),
            "v_interval_low": np.float32("nan"),
            "v_interval_high": np.float32("nan"),
            "v_interval_nonempty": np.int8(0),
            "v1_in_interval": np.int8(0),
            "label_pass": np.int8(0),
            "endpoint_error_position_m": np.float32("nan"),
            "endpoint_error_heading_rad": np.float32("nan"),
            "endpoint_error_speed_mps": np.float32("nan"),
            "endpoint_error_raw_position_m": np.float32("nan"),
            "endpoint_error_raw_heading_rad": np.float32("nan"),
            "endpoint_error_raw_speed_mps": np.float32("nan"),
            "path_length_m": np.float32("nan"),
            "chord_length_m": np.float32("nan"),
            "start_accel_mps2": np.float32("nan"),
            "end_accel_mps2": np.float32("nan"),
            "max_abs_a_long": np.float32("nan"),
            "max_abs_a_lat": np.float32("nan"),
            "max_abs_kappa": np.float32("nan"),
            "max_abs_kappa_rate": np.float32("nan"),
            "max_abs_jerk_long": np.float32("nan"),
            "max_abs_jerk_lat": np.float32("nan"),
            "stop_class": np.int8(0),
            "action_in_bounds": np.int8(0),
            "out_of_bounds_components": "",
            "codes_summary": "LABEL_DIAGNOSTICS_UNAVAILABLE",
            "n_codes": np.int16(1),
        }


def _audit_fields(
    scene_seed: int, info: Dict[str, Any], *, source: int = 1
) -> Dict[str, Any]:
    target = info.get("pp_target_lane")
    anchor = info.get("pp_anchor_lane")
    refs = tuple(info.get("pp_ref_lanes") or ())
    next_refs = tuple(info.get("pp_next_ref_lanes") or ())

    # Same namespace for target/anchor so equal lane keys produce equal IDs;
    # the ref-lane tuples use separate namespaces because they are sets.
    target_id = (
        np.int64(_stable_hash(scene_seed, "expert_lane", target))
        if target is not None
        else np.int64(0)
    )
    anchor_id = (
        np.int64(_stable_hash(scene_seed, "expert_lane", anchor))
        if anchor is not None
        else np.int64(0)
    )
    refs_id = (
        np.int64(_stable_hash(scene_seed, "expert_ref_lanes", refs))
        if refs
        else np.int64(0)
    )
    next_refs_id = (
        np.int64(_stable_hash(scene_seed, "expert_next_ref_lanes", next_refs))
        if next_refs
        else np.int64(0)
    )
    return {
        "expert_target_lane_id": target_id,
        "expert_anchor_lane_id": anchor_id,
        "expert_ref_lanes_id": refs_id,
        "expert_next_ref_lanes_id": next_refs_id,
        "expert_used_centerline": np.int8(1),
        "expert_used_static_scan": np.int8(1),
        "expert_used_future_state": np.int8(0),
        # Literal requirement: 1 when target lane differs from the anchor lane.
        "expert_lane_change_decision": np.int8(1 if str(target) != str(anchor) else 0),
        "privileged_expert_sample": np.int8(1),
        "source": np.int8(source),
    }


# ---------------------------------------------------------------------------
# Shield (copied from dataset/collect_dagger.py; v2 OD columns are identical)
# ---------------------------------------------------------------------------
def _needs_shield(obs: Dict[str, np.ndarray], ttc_thresh: float) -> bool:
    if ttc_thresh <= 0:
        return False
    od = obs["od"]
    mask = obs["od_mask"] > 0.5
    if not mask.any():
        return False
    dx, dy, vx = od[:, 0], od[:, 1], od[:, 4]
    ttc = np.where((dx > 0) & (vx < -0.1), dx / np.maximum(-vx, 0.1), 1e6)
    lateral = np.abs(dy) < 2.5
    return bool(np.any(mask & lateral & (ttc < ttc_thresh)))


# ---------------------------------------------------------------------------
# Packing / outcomes
# ---------------------------------------------------------------------------
def _pack_rows(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Stack per-row dicts into the final array mapping (deterministic order)."""
    if not rows:
        raise ValueError("_pack_rows: no rows")
    expected = set(HISTORY_KEYS) | set(ROW_DTYPES)
    first = set(rows[0])
    if first != expected:
        raise ValueError(
            f"row schema mismatch: missing={sorted(expected - first)} "
            f"extra={sorted(first - expected)}"
        )
    for i, row in enumerate(rows[1:], start=1):
        if set(row) != expected:
            raise ValueError(f"row {i} schema mismatch")

    out: Dict[str, np.ndarray] = {}
    for key in HISTORY_KEYS:
        out[key] = np.stack([row[key] for row in rows], axis=0)
    for key, dtype in ROW_DTYPES.items():
        out[key] = np.asarray([row[key] for row in rows], dtype=dtype)
    return out


def _backfill_outcomes(
    rows: List[Dict[str, Any]],
    *,
    termination: str,
    done: bool,
    success: int,
    collision: int,
    off_road: int,
    wrong_route: int = 0,
    stuck: int = 0,
) -> None:
    """Backfill the student roll-in outcome to every row.

    This deliberately does **not** touch ``label_valid`` / ``expert_plan_valid``:
    a valid expert label remains usable for plan learning even when the student
    episode failed.
    """
    term_id = np.int16(TERMINATION_IDS.get(termination, 5))
    timeout = np.int8(1 if termination in ("max_step", "timeout") else 0)
    success_i8 = np.int8(success)
    failure_i8 = np.int8(0 if success else 1)
    for i, row in enumerate(rows):
        row["done"] = np.int8(1 if (done and i == len(rows) - 1) else 0)
        row["success"] = success_i8
        row["collision"] = np.int8(collision)
        row["off_road"] = np.int8(off_road)
        row["wrong_route"] = np.int8(wrong_route)
        row["stuck"] = np.int8(stuck)
        row["timeout"] = timeout
        row["terminal_failed"] = np.int8(1 if (failure_i8 and i == len(rows) - 1) else 0)
        row["student_success"] = success_i8
        row["student_failure"] = failure_i8
        row["termination_id"] = term_id


def _relative_action(
    p0: np.ndarray, h0: float, v0: float, p1: np.ndarray, h1: float, v1: float
) -> np.ndarray:
    """Measured 0.5 s endpoint in the pre-step ego frame."""
    dxw, dyw = float(p1[0] - p0[0]), float(p1[1] - p0[1])
    c0, s0 = math.cos(h0), math.sin(h0)
    rel = np.array([c0 * dxw + s0 * dyw, -s0 * dxw + c0 * dyw], dtype=np.float64)
    return np.array([rel[0], rel[1], _wrap(h1 - h0), float(v1)], dtype=np.float64)


# ---------------------------------------------------------------------------
# Model loading / batching
# ---------------------------------------------------------------------------
def _config_int(config: Any, key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (AttributeError, TypeError, ValueError):
        return int(default)


def load_model(ckpt: str, device: str = "cpu") -> StructuredPlanner:
    """Strict checkpoint loader shared with eval/training."""
    return load_model_from_checkpoint(ckpt, device, strict=True)

def _read_ckpt_config(ckpt: str) -> Dict[str, Any]:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    config = dict(config or {})
    return {
        "hidden": _config_int(config, "hidden", H),
        "heads": _config_int(config, "heads", HEADS),
        "spatial_layers": _config_int(config, "spatial_layers", SPATIAL_LAYERS),
    }


def _batch_from_history(
    hist: Dict[str, Any], device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    keys = (
        "od",
        "od_mask",
        "ld",
        "ld_mask",
        "nav",
        "nav_mask",
        "ego",
        "ego_mask",
        "hist_valid",
    )
    batch: Dict[str, torch.Tensor] = {}
    for key in keys:
        arr = np.asarray(hist[key], dtype=np.float32)
        batch[key] = torch.from_numpy(arr).unsqueeze(0).to(device)
    return batch


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------
def run_episode(
    env: Any,
    model: StructuredPlanner,
    spec: Any,
    *,
    max_steps: int,
    device: str,
    expert_ratio: float,
    shield_ttc: float,
    round_id: int,
    seed: int,
) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    """Run one DAgger student roll-in; return ``(arrays, episode_info)`` or None."""
    env.reset()
    ego = env.agent
    expert = _expert_class()(ego, int(getattr(spec, "seed", 0)))
    if hasattr(expert, "reset"):
        expert.reset()
    tracker = ExactTracker(dt=DECISION_DT, hz=HZ)
    rng = np.random.default_rng(_episode_rng_seed(seed, spec, round_id))

    frames: List[Dict[str, Any]] = []
    poses: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    prev_action = np.zeros(ACTION_DIM, dtype=np.float64)
    prev_speed: Optional[float] = None
    prev_heading: Optional[float] = None
    ep_step = 0
    done = False
    termination = "max_step"
    success = collision = off_road = wrong_route = 0
    rc = 0.0
    low_speed_run = 0
    stuck_run = False
    replay_rows = lift_rows = shield_rows = fallback_rows = 0

    while ep_step < max_steps:
        obs = build_observation(
            env,
            prev_action=prev_action,
            prev_speed=prev_speed,
            prev_heading=prev_heading,
            dt=DECISION_DT,
        )
        frames.append(obs)
        poses.append(obs["pose"].copy())
        if len(frames) > HISTORY_FRAMES:
            frames = frames[-HISTORY_FRAMES:]
            poses = poses[-HISTORY_FRAMES:]
        hist = stack_history(frames, poses)
        scene_seed = int(obs["meta"]["scene_seed"])

        p0 = poses[-1].copy()
        v0 = float(ego.speed)
        h0 = float(ego.heading_theta)

        batch = _batch_from_history(hist, device=device)
        with torch.no_grad():
            out = model(batch)
        a_norm = out["action_mu"][0].detach().cpu().numpy().astype(np.float64)
        student_action = np.asarray(CLEAN_BOUNDS.denormalize(a_norm, clip=True), dtype=np.float64)

        # Expert is queried at the decision state for the lift and the audit.
        expert_raw = expert.act()
        steer = _safe_float(expert_raw[0] if len(expert_raw) > 0 else 0.0, 0.0)
        throttle = _safe_float(expert_raw[1] if len(expert_raw) > 1 else 0.0, 0.0)
        action_info = dict(getattr(expert, "action_info", {}) or {})
        lift = _lift_action(
            steer,
            throttle,
            speed=v0,
            max_steer_angle_rad=float(expert.max_steer_angle_rad),
            wheelbase=float(expert.wheelbase),
            accel_action_scale=float(expert.accel_action_scale),
            brake_action_scale=float(expert.brake_action_scale),
        )

        shield = _needs_shield(obs, shield_ttc)
        replay, shield = _select_mode(
            shield=shield, expert_ratio=expert_ratio, rng=rng
        )

        n_substeps = 0
        segment_rc_start = rc
        fallback_name = "not_evaluated"
        executed_action = lift.astype(np.float64)
        replay_action: Optional[np.ndarray] = None

        if replay:
            for _ in range(STEPS_PER_POLICY):
                if ep_step >= max_steps:
                    break
                _, _, term, trunc, info = env.step(expert.act())
                ep_step += 1
                n_substeps += 1
                info = info or {}
                rc = _safe_float(info.get("route_completion"), rc)
                success = int(success or bool(info.get("arrive_dest", False)))
                collision = int(collision or bool(info.get("crash", False)))
                off_road = int(off_road or bool(info.get("out_of_road", False)))
                wrong_route = int(
                    wrong_route
                    or bool(info.get("wrong_route", False))
                    or bool(info.get("wrong_way", False))
                )
                low_speed_run, stuck_run = _speed_run(
                    float(ego.speed), low_speed_run, stuck_run
                )
                if term or trunc:
                    done = True
                    termination = _termination_name(
                        success=success,
                        collision=collision,
                        off_road=off_road,
                        term=term,
                        trunc=trunc,
                    )
                    break

            p1 = np.asarray(ego.position, dtype=np.float64)[:2]
            h1 = float(ego.heading_theta)
            v1 = float(ego.speed)
            replay_action = _relative_action(p0, h0, v0, p1, h1, v1)
            executed_action = (
                replay_action.copy()
                if np.all(np.isfinite(replay_action))
                else lift.astype(np.float64)
            )
            replay_rows += 1
        else:
            resolved_action, traj_exec, fallback_name = _resolve_student_action(
                student_action, v0
            )
            executed_action = np.asarray(resolved_action, dtype=np.float64)
            if fallback_name != "none":
                fallback_rows += 1
            if traj_exec is None:
                # The safe brake can only be unavailable for a non-finite state.
                done = True
                termination = "invalid_action"
            else:
                tracker.arm(env, trajectory=traj_exec)
                for _ in range(STEPS_PER_POLICY):
                    if ep_step >= max_steps:
                        break
                    _, _, term, trunc, info = env.step([0.0, 0.0])
                    ep_step += 1
                    n_substeps += 1
                    tracker.apply(env)
                    info = info or {}
                    rc = _safe_float(info.get("route_completion"), rc)
                    success = int(success or bool(info.get("arrive_dest", False)))
                    collision = int(collision or bool(info.get("crash", False)))
                    off_road = int(off_road or bool(info.get("out_of_road", False)))
                    wrong_route = int(
                        wrong_route
                        or bool(info.get("wrong_route", False))
                        or bool(info.get("wrong_way", False))
                    )
                    low_speed_run, stuck_run = _speed_run(
                        float(ego.speed), low_speed_run, stuck_run
                    )
                    if term or trunc:
                        done = True
                        termination = _termination_name(
                            success=success,
                            collision=collision,
                            off_road=off_road,
                            term=term,
                            trunc=trunc,
                        )
                        break
            lift_rows += 1

        if rc < segment_rc_start - WRONG_ROUTE_DROP:
            wrong_route = 1
        if shield:
            shield_rows += 1

        label = _compose_label(
            replay=replay,
            action_lift=lift,
            action_replay=replay_action,
            n_substeps=n_substeps,
            v0=v0,
        )
        action = np.asarray(label["action"], dtype=np.float64)
        diagnostics = _diagnostics_or_empty(action, v0)
        audit = _audit_fields(scene_seed, action_info, source=1)
        student_feasible = FALLBACK_IDS["not_evaluated"]
        if not replay:
            student_feasible = 1 if fallback_name == "none" else 0

        row: Dict[str, Any] = {key: hist[key] for key in HISTORY_KEYS}
        row.update(
            {
                "action": np.asarray(action, dtype=np.float32),
                "action_lift": lift.astype(np.float32),
                "action_replay": np.asarray(label["action_replay"], dtype=np.float32),
                "student_action": student_action.astype(np.float32),
                "executed_action": executed_action.astype(np.float32),
                "expert_action": np.array([steer, throttle], dtype=np.float32),
                "residual_xy": np.asarray(label["residual_xy"], dtype=np.float32),
                "residual_xy_norm": np.float32(label["residual_xy_norm"]),
                "residual_dpsi": np.float32(label["residual_dpsi"]),
                "residual_v": np.float32(label["residual_v"]),
                "expert_steer": np.float32(steer),
                "expert_throttle": np.float32(throttle),
                "expert_lookahead_m": np.float32(
                    _safe_float(action_info.get("pp_lookahead_m"), -1.0)
                ),
                "expert_static_gap_m": np.float32(
                    _safe_float(action_info.get("pp_static_gap_m"), -1.0)
                ),
                **audit,
                **diagnostics,
                "label_source": np.int8(label["label_source"]),
                "label_valid": np.int8(label["label_valid"]),
                "n_substeps": np.int16(label["n_substeps"]),
                "intervened": np.int8(1 if replay else 0),
                "shield": np.int8(1 if shield else 0),
                "student_feasible": np.int8(student_feasible),
                "fallback": np.int8(FALLBACK_IDS[fallback_name]),
                "round": np.int32(round_id),
                "step": np.int64(len(rows)),
                "episode_id": np.int64(0),  # filled by the caller
                "scene_seed": np.int64(scene_seed),
                "done": np.int8(0),
                "terminal_failed": np.int8(0),
                "success": np.int8(0),
                "collision": np.int8(0),
                "off_road": np.int8(0),
                "wrong_route": np.int8(0),
                "stuck": np.int8(0),
                "timeout": np.int8(0),
                "student_success": np.int8(0),
                "student_failure": np.int8(0),
                "termination_id": np.int16(0),
                "expert_plan_valid": np.int8(label["label_valid"]),
                "expert_failed": np.int8(0 if int(label["label_valid"]) else 1),
            }
        )
        rows.append(row)

        prev_action = np.asarray(executed_action, dtype=np.float64)
        prev_speed = v0
        prev_heading = h0
        if done or termination == "invalid_action":
            break

    if not rows:
        return None

    # A budget cutoff without a terminal env signal keeps termination="max_step"
    # (timeout=1 after backfill), matching collect's TERMINATION_IDS mapping.
    stuck = int(bool(stuck_run and not success))
    _backfill_outcomes(
        rows,
        termination=termination,
        done=done,
        success=success,
        collision=collision,
        off_road=off_road,
        wrong_route=wrong_route,
        stuck=stuck,
    )
    data = _pack_rows(rows)
    info = {
        "termination": termination,
        "done": bool(done),
        "success": int(success),
        "collision": int(collision),
        "off_road": int(off_road),
        "wrong_route": int(wrong_route),
        "stuck": int(stuck),
        "rows": len(rows),
        "env_steps": ep_step,
        "scene_seed": int(getattr(spec, "seed", 0)),
        "spec_id": getattr(spec, "id", None),
        "expert_class": type(expert).__name__,
        "expert_params": expert.params() if hasattr(expert, "params") else {},
        "replay_rows": int(replay_rows),
        "lift_rows": int(lift_rows),
        "shield_rows": int(shield_rows),
        "fallback_rows": int(fallback_rows),
        "label_valid_rows": int(np.sum(data["label_valid"])),
    }
    return data, info


def _speed_run(
    speed: float, low_speed_run: int, stuck_run: bool
) -> Tuple[int, bool]:
    if speed < STUCK_SPEED_MPS:
        low_speed_run += 1
        if low_speed_run > STUCK_STEPS:
            stuck_run = True
    else:
        low_speed_run = 0
    return low_speed_run, stuck_run


def _termination_name(
    *, success: int, collision: int, off_road: int, term: bool, trunc: bool
) -> str:
    if success:
        return "arrive"
    if collision:
        return "collision"
    if off_road:
        return "out_of_road"
    if trunc and not term:
        return "timeout"
    return "terminal"


_WORKER_MODEL: Optional[StructuredPlanner] = None
_WORKER_DEVICE = "cpu"
_WORKER_CKPT = ""


def _init_worker(ckpt: str, device: str) -> None:
    global _WORKER_MODEL, _WORKER_DEVICE, _WORKER_CKPT
    _WORKER_MODEL = load_model(ckpt, device)
    _WORKER_DEVICE = device
    _WORKER_CKPT = ckpt


def _worker(
    args: Tuple[int, Any, int, float, float, int, int]
) -> Tuple[int, Optional[Dict[str, np.ndarray]], float, str, Optional[Dict[str, Any]]]:
    idx, spec, max_steps, expert_ratio, shield_ttc, round_id, seed = args
    t0 = time.time()
    model = _WORKER_MODEL
    try:
        if model is None:  # defensive: pool initializer did not run
            model = load_model(_WORKER_CKPT, _WORKER_DEVICE)
        env = build_env(spec, use_render=False, lru_size=0)
        try:
            result = run_episode(
                env,
                model,
                spec,
                max_steps=int(max_steps),
                device=_WORKER_DEVICE,
                expert_ratio=float(expert_ratio),
                shield_ttc=float(shield_ttc),
                round_id=int(round_id),
                seed=int(seed),
            )
        finally:
            env.close()
        if result is None:
            return idx, None, time.time() - t0, "no_rows", None
        data, info = result
        return idx, data, time.time() - t0, "ok", info
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return idx, None, time.time() - t0, f"{type(exc).__name__}: {exc}", None


# ---------------------------------------------------------------------------
# Meta summaries
# ---------------------------------------------------------------------------
def _rate(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values.astype(np.float64)))


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _residual_summary(merged: Dict[str, np.ndarray]) -> Dict[str, Any]:
    xy = _finite(merged["residual_xy_norm"])
    dpsi = np.abs(_finite(merged["residual_dpsi"]))
    dv = np.abs(_finite(merged["residual_v"]))
    if xy.size == 0:
        return {
            "measured_rows": 0,
            "residual_xy_mean_m": None,
            "residual_xy_p95_m": None,
            "residual_xy_max_m": None,
            "residual_dpsi_abs_mean_rad": None,
            "residual_v_abs_mean_mps": None,
            "note": "residuals are only defined on fully measured replay rows",
        }
    return {
        "measured_rows": int(xy.size),
        "residual_xy_mean_m": float(np.mean(xy)),
        "residual_xy_p95_m": float(np.percentile(xy, 95)),
        "residual_xy_max_m": float(np.max(xy)),
        "residual_dpsi_abs_mean_rad": (
            float(np.mean(dpsi)) if dpsi.size else None
        ),
        "residual_v_abs_mean_mps": float(np.mean(dv)) if dv.size else None,
        "note": "residual = action_lift - action_replay on measured replay rows",
    }


def _label_summary(merged: Dict[str, np.ndarray], dagger: Dict[str, Any]) -> Dict[str, Any]:
    n = int(merged["action"].shape[0])
    if n == 0:
        return {}
    substeps, sub_counts = np.unique(merged["n_substeps"], return_counts=True)
    finite_res = _finite(merged["residual_xy_norm"])
    return {
        "rows": n,
        "label_valid_rate": _rate(merged["label_valid"]),
        "feasible_rate": _rate(merged["feasible"]),
        "comfortable_rate": _rate(merged["comfortable"]),
        "pass_rate": _rate(merged["label_pass"]),
        "bounds_ok_rate": _rate(merged["action_in_bounds"]),
        "v1_interval_empty_rows": int(
            np.count_nonzero(merged["v_interval_nonempty"] == 0)
        ),
        "v1_in_interval_rate": _rate(merged["v1_in_interval"]),
        "residual_xy_mean_m": (
            float(np.mean(finite_res)) if finite_res.size else None
        ),
        "residual_xy_p95_m": (
            float(np.percentile(finite_res, 95)) if finite_res.size else None
        ),
        "residual_xy_max_m": (
            float(np.max(finite_res)) if finite_res.size else None
        ),
        "residual_dpsi_abs_mean_rad": (
            float(np.mean(np.abs(_finite(merged["residual_dpsi"]))))
            if _finite(merged["residual_dpsi"]).size
            else None
        ),
        "residual_v_abs_mean_mps": (
            float(np.mean(np.abs(_finite(merged["residual_v"]))))
            if _finite(merged["residual_v"]).size
            else None
        ),
        "n_substeps_histogram": {
            int(k): int(v) for k, v in zip(substeps, sub_counts)
        },
        "replay_fraction": dagger["replay_fraction"],
        "lift_fraction": dagger["lift_fraction"],
    }


def _dagger_summary(merged: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = int(merged["action"].shape[0])
    source = merged["source"]
    replay_mask = merged["label_source"] > 0.5
    shield_mask = merged["shield"] > 0.5
    intervened_mask = merged["intervened"] > 0.5
    fallback = merged["fallback"]
    replay_rows = int(np.count_nonzero(replay_mask))
    shield_rows = int(np.count_nonzero(shield_mask))
    intervened_rows = int(np.count_nonzero(intervened_mask))
    return {
        "rows": n,
        "replay_rows": replay_rows,
        "lift_rows": int(n - replay_rows),
        "replay_fraction": float(replay_rows / n) if n else 0.0,
        "lift_fraction": float((n - replay_rows) / n) if n else 0.0,
        "shield_rows": shield_rows,
        "random_replay_rows": int(
            np.count_nonzero(intervened_mask & ~shield_mask)
        ),
        "intervened_rows": intervened_rows,
        "intervention_rate": float(intervened_rows / n) if n else 0.0,
        "fallback_rows": int(
            np.count_nonzero(np.isin(fallback, (1, 2, 3, 4)))
        ),
        "v1_projection_rows": int(
            np.count_nonzero(fallback == FALLBACK_IDS["v1_projection"])
        ),
        "safe_brake_rows": int(
            np.count_nonzero(fallback == FALLBACK_IDS["safe_brake"])
        ),
        "nonfinite_safe_brake_rows": int(
            np.count_nonzero(fallback == FALLBACK_IDS["nonfinite_safe_brake"])
        ),
        "student_feasible_rows": int(
            np.count_nonzero(merged["student_feasible"] == 1)
        ),
        "student_infeasible_rows": int(
            np.count_nonzero(merged["student_feasible"] == 0)
        ),
        "student_feasible_not_evaluated_rows": int(
            np.count_nonzero(merged["student_feasible"] < 0)
        ),
        "label_valid_rows": int(np.count_nonzero(merged["label_valid"] > 0.5)),
        "partial_label_rows": int(np.count_nonzero(merged["label_valid"] == 0)),
        "source_counts": {
            str(int(value)): int(count)
            for value, count in zip(*np.unique(source, return_counts=True))
        },
        "source_names": {"1": "dagger", "0": "expert_bc"},
        "residual_stats": _residual_summary(merged),
    }


def _audit_summary(merged: Dict[str, np.ndarray]) -> Dict[str, Any]:
    target_ids = merged["expert_target_lane_id"]
    anchor_ids = merged["expert_anchor_lane_id"]
    ref_ids = merged["expert_ref_lanes_id"]
    return {
        "privileged_expert_rows": int(
            np.count_nonzero(merged["privileged_expert_sample"])
        ),
        "used_centerline_rows": int(np.count_nonzero(merged["expert_used_centerline"])),
        "used_static_scan_rows": int(
            np.count_nonzero(merged["expert_used_static_scan"])
        ),
        "used_future_state_rows": int(
            np.count_nonzero(merged["expert_used_future_state"])
        ),
        "lane_change_rows": int(
            np.count_nonzero(merged["expert_lane_change_decision"])
        ),
        "lane_change_ratio": _rate(merged["expert_lane_change_decision"]),
        "rows_with_target_lane": int(np.count_nonzero(target_ids)),
        "unique_target_lane_ids": int(len(np.unique(target_ids[target_ids != 0]))),
        "unique_anchor_lane_ids": int(len(np.unique(anchor_ids[anchor_ids != 0]))),
        "unique_ref_lane_sets": int(len(np.unique(ref_ids[ref_ids != 0]))),
        "note": (
            "Target lane and lane-level route IDs are audit metadata only; "
            "they never enter the student OD/LD/Nav/Ego features."
        ),
    }


def _episode_audit(
    episode_id: int, shard: Dict[str, np.ndarray], info: Dict[str, Any]
) -> Dict[str, Any]:
    target_ids = shard["expert_target_lane_id"]
    residual = shard["residual_xy_norm"]
    finite_residual = residual[np.isfinite(residual)]
    return {
        "episode_id": int(episode_id),
        "spec_id": info.get("spec_id"),
        "scene_seed": int(info["scene_seed"]),
        "termination": info["termination"],
        "rows": int(info["rows"]),
        "env_steps": int(info["env_steps"]),
        "success": int(info["success"]),
        "collision": int(info["collision"]),
        "off_road": int(info["off_road"]),
        "wrong_route": int(info["wrong_route"]),
        "stuck": int(info["stuck"]),
        "replay_rows": int(info.get("replay_rows", 0)),
        "lift_rows": int(info.get("lift_rows", 0)),
        "shield_rows": int(info.get("shield_rows", 0)),
        "fallback_rows": int(info.get("fallback_rows", 0)),
        "label_valid_rows": int(info.get("label_valid_rows", 0)),
        "lane_change_rows": int(
            np.count_nonzero(shard["expert_lane_change_decision"])
        ),
        "unique_target_lane_ids": int(
            len(np.unique(target_ids[target_ids != 0]))
        ),
        "mean_residual_xy_m": (
            float(np.mean(finite_residual)) if finite_residual.size else None
        ),
        "max_residual_xy_m": (
            float(np.max(finite_residual)) if finite_residual.size else None
        ),
        "feasible_rate": float(np.mean(shard["feasible"])),
    }


def _git_commit() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = result.stdout.strip()
        return commit or None
    except Exception:  # noqa: BLE001
        return None


def _limits_dict(limits: Any) -> Dict[str, Any]:
    return {key: float(value) for key, value in asdict(limits).items()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Schema-v2 DAgger collection with measured expert replay"
    )
    parser.add_argument("--specs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default=None,
        help="torch device; defaults to cuda when available, else cpu",
    )
    parser.add_argument(
        "--expert-ratio",
        type=float,
        default=0.5,
        help="probability of expert replay at each decision state (0..1)",
    )
    parser.add_argument(
        "--shield-ttc",
        type=float,
        default=0.0,
        help="force expert replay when closest in-path TTC is below this (0=off)",
    )
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not 0.0 <= float(args.expert_ratio) <= 1.0:
        parser.error("--expert-ratio must be in [0, 1]")
    if float(args.shield_ttc) < 0.0:
        parser.error("--shield-ttc must be >= 0")
    if int(args.workers) < 1:
        parser.error("--workers must be >= 1")
    if int(args.round) < 0:
        parser.error("--round must be >= 0")
    # Resolve lazily so an explicit --device cpu never initializes CUDA in the
    # parent process (fork-safety), and so the default matches the other tools.
    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    specs = load_specs(args.specs)
    offset = max(0, int(args.offset))
    specs = specs[offset:]
    if args.limit is not None:
        specs = specs[: max(0, int(args.limit))]
    if not specs:
        print("[collect_dagger] no specs")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_config = _read_ckpt_config(args.ckpt)

    tasks = [
        (
            i,
            spec,
            int(args.max_steps),
            float(args.expert_ratio),
            float(args.shield_ttc),
            int(args.round),
            int(args.seed),
        )
        for i, spec in enumerate(specs)
    ]
    results: List[
        Tuple[int, Optional[Dict[str, np.ndarray]], float, str, Optional[Dict[str, Any]]]
    ] = []
    mp_method = "none"
    if args.workers <= 1:
        _init_worker(args.ckpt, args.device)
        for task in tasks:
            results.append(_worker(task))
    else:
        import multiprocessing as mp

        # Use spawn for every multi-worker run: the StructuredPlanner may live on
        # CUDA, and CUDA cannot be re-initialized in a forked child.  Spawn also
        # guarantees each worker loads its own model instead of inheriting the
        # parent's CUDA context.
        mp_method = "spawn"
        ctx = mp.get_context(mp_method)
        with ctx.Pool(
            processes=int(args.workers),
            initializer=_init_worker,
            initargs=(args.ckpt, args.device),
        ) as pool:
            for result in pool.imap_unordered(_worker, tasks):
                results.append(result)
                if len(results) % 10 == 0:
                    print(
                        f"[collect_dagger] {len(results)}/{len(tasks)}",
                        flush=True,
                    )

    results.sort(key=lambda item: item[0])

    shards: List[Tuple[int, Dict[str, np.ndarray], Dict[str, Any]]] = []
    failed_indices: List[int] = []
    failed_reasons: Dict[int, str] = {}
    for idx, data, _elapsed, status, info in results:
        if data is None or info is None:
            failed_indices.append(int(idx))
            failed_reasons[int(idx)] = status
            continue
        data["episode_id"] = np.full(
            (int(data["action"].shape[0]),), idx, dtype=np.int64
        )
        shards.append((int(idx), data, info))

    if not shards:
        print(f"[collect_dagger] all failed: {failed_indices}")
        return 1

    keys = list(shards[0][1].keys())
    merged = {
        key: np.concatenate([shard[1][key] for shard in shards], axis=0)
        for key in keys
    }

    npz_path = out_dir / "expert_bc.npz"
    np.savez_compressed(npz_path, **merged)

    episodes = [
        _episode_audit(idx, shard, info) for idx, shard, info in shards
    ]
    termination_histogram = Counter(info["termination"] for _, _, info in shards)
    rows = int(merged["action"].shape[0])
    dagger = _dagger_summary(merged)
    expert_class = shards[0][2].get("expert_class")
    expert_params = shards[0][2].get("expert_params", {})

    meta: Dict[str, Any] = {
        "schema": "pnp_v2",
        "schema_version": 2,
        "dataset": "expert_bc",
        "variant": "dagger",
        "collector": "dataset.collect_dagger",
        "obs_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "ckpt": str(args.ckpt),
        "specs": str(args.specs),
        "out": str(out_dir),
        "limit": args.limit,
        "offset": args.offset,
        "workers": args.workers,
        "mp_method": mp_method,
        "max_steps": args.max_steps,
        "device": str(args.device),
        "expert_ratio": float(args.expert_ratio),
        "shield_ttc": float(args.shield_ttc),
        "round": int(args.round),
        "seed": int(args.seed),
        "decision_dt": DECISION_DT,
        "sim_dt": SIM_DT,
        "steps_per_policy": STEPS_PER_POLICY,
        "history_frames": HISTORY_FRAMES,
        "seeds": [int(getattr(spec, "seed", 0)) for spec in specs],
        "spec_ids": [getattr(spec, "id", None) for spec in specs],
        "git_commit": _git_commit(),
        "model": {
            "class": "StructuredPlanner",
            "ckpt": str(args.ckpt),
            "config": model_config,
            "student_action": "CLEAN_BOUNDS.denormalize(action_mu[0], clip=True)",
        },
        "expert": {
            "class": expert_class,
            "params": expert_params,
            "privileged": True,
            "audit": (
                "lane centerline / target lane / lane-level ref lanes / "
                "static obstacle scan are used; future traffic state is not"
            ),
        },
        "action_bounds": CLEAN_BOUNDS.as_dict(),
        "action_limits": {
            "bounds": CLEAN_BOUNDS.as_dict(),
            "hard": {**_limits_dict(HARD_LIMITS), "provisional": True},
            "comfort": {
                **_limits_dict(COMFORT_LIMITS),
                "provisional": True,
            },
            "hard_source": HARD_LIMITS_SOURCE,
            "note": (
                "Provisional clean limits calibrated on expert BC rows. "
                "PSTL continuous derivatives are stricter than label "
                "differences; net.action_space.HARD_LIMITS is intentionally "
                "not used here."
            ),
        },
        "label": {
            "source_measured": 1,
            "source_lift": 0,
            "definition": (
                "action = measured 0.5 s expert replay endpoint on replay rows; "
                "action = 5 x 0.1 s kinematic-bicycle lift of the expert "
                "(steer, throttle) at the decision state on student rows"
            ),
            "replay_definition": (
                "expert physically drives 5 x 0.1 s with env.step(expert.act()); "
                "endpoint measured from pre/post ego pose and speed"
            ),
            "lift_definition": (
                "5 x 0.1 s kinematic bicycle holding the expert (steer, throttle)"
            ),
            "residual_definition": (
                "residual = action_lift - action_replay only on fully measured "
                "replay rows; NaN on lift-only rows and partial labels"
            ),
            "expert_plan_valid_definition": (
                "1 when label_valid=1 (a full, finite expert 0.5 s label), "
                "independent of the student episode outcome"
            ),
            "expert_failed_definition": "1 - expert_plan_valid",
            **_label_summary(merged, dagger),
        },
        "dagger": dagger,
        # Top-level aliases for the DAgger audit fields required by the spec.
        "source_counts": dagger["source_counts"],
        "replay_rows": dagger["replay_rows"],
        "lift_rows": dagger["lift_rows"],
        "replay_fraction": dagger["replay_fraction"],
        "lift_fraction": dagger["lift_fraction"],
        "intervention_rate": dagger["intervention_rate"],
        "residual_stats": dagger["residual_stats"],
        "audit": _audit_summary(merged),
        "counts": {
            "specs": len(specs),
            "episodes_ok": len(shards),
            "episodes_fail": len(failed_indices),
            "rows": rows,
            "policy_steps": rows,
            "label_substeps": int(np.sum(merged["n_substeps"])),
            "env_steps": int(sum(info["env_steps"] for _, _, info in shards)),
            "success_episodes": int(sum(info["success"] for _, _, info in shards)),
            "collision_episodes": int(
                sum(info["collision"] for _, _, info in shards)
            ),
            "off_road_episodes": int(
                sum(info["off_road"] for _, _, info in shards)
            ),
            "wrong_route_episodes": int(
                sum(info["wrong_route"] for _, _, info in shards)
            ),
            "stuck_episodes": int(sum(info["stuck"] for _, _, info in shards)),
            "timeout_episodes": int(
                sum(
                    1
                    for _, _, info in shards
                    if info["termination"] in ("max_step", "timeout")
                )
            ),
            "student_failed_episodes": int(
                sum(1 for _, _, info in shards if not info["success"])
            ),
        },
        "termination_histogram": dict(termination_histogram),
        "episodes": episodes,
        "failed_indices": failed_indices,
        "failed_reasons": failed_reasons,
        "shapes": {key: list(value.shape) for key, value in merged.items()},
        "dtypes": {key: str(value.dtype) for key, value in merged.items()},
    }
    meta_path = out_dir / "expert_bc.meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[collect_dagger] wrote {npz_path} rows={rows} "
        f"episodes={len(shards)} failed={len(failed_indices)} "
        f"replay={dagger['replay_rows']} lift={dagger['lift_rows']} "
        f"intervention={dagger['intervention_rate']:.4f} "
        f"fallback={dagger['fallback_rows']} "
        f"valid={dagger['label_valid_rows']}"
    )
    print(f"[collect_dagger] meta {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
