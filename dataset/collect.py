"""Collect schema-v2 expert BC data for P1.

Each row is one 0.5 s policy decision:

* OD / LD / road-level Nav / Ego history from :mod:`env.obs`
  (``HISTORY_FRAMES = 6`` frames, SE(2)-compensated by ``stack_history``);
* a **measured** expert endpoint label ``(dx, dy, dpsi, v_next)`` obtained from
  the ego pose/speed after the expert physically drives 5 x 0.1 s;
* a kinematic-bicycle 4D lift of the expert's current ``(steer, throttle)``
  only as an audit/residual signal, never as the label;
* privileged expert audit fields (target lane / lane-level route references)
  kept in side arrays and meta only, never inside student features;
* episode outcome backfilled to every row.

The expert is ``PurePursuitIDMPolicy`` and acts through the engine exactly as
in ``dataset/collect.py``: it is registered once, then each decision calls
``env.step([0, 0])`` five times.

Notes on limits
---------------
``net.action_space.HARD_LIMITS`` is too strict for *continuous PSTL*
derivatives of expert BC labels (about 24% hard failures on 2k expert rows).
This collector therefore uses a local provisional clean hard set. If
``net.action_space`` later exposes ``HARD_LIMITS`` (or
``COMFORT_LIMITS``), that constant is used instead and recorded in meta.
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
from net.action_space import (
    CLEAN_BOUNDS,
    COMFORT_LIMITS,
    HARD_LIMITS,
    action_to_trajectory,
    feasible_v1_interval,
)

DECISION_DT = 0.5
SIM_DT = 0.1
STEPS_PER_POLICY = 5
ACTION_DIM = 4

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
    # measured label + lift audit
    "action": np.float32,
    "action_lift": np.float32,
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
    # label / episode metadata
    "label_source": np.int8,
    "label_valid": np.int8,
    "n_substeps": np.int16,
    "step": np.int64,
    "episode_id": np.int64,
    "scene_seed": np.int64,
    "done": np.int8,
    "success": np.int8,
    "collision": np.int8,
    "off_road": np.int8,
    "wrong_route": np.int8,
    "stuck": np.int8,
    "timeout": np.int8,
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


# ---------------------------------------------------------------------------
# Expert 4D lift (copied from dataset/collect_dagger.py:34-49)
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
# Label diagnostics / audit
# ---------------------------------------------------------------------------
def _label_diagnostics(action: np.ndarray, v0: float) -> Dict[str, Any]:
    traj = action_to_trajectory(
        action,
        v0,
        dt=DECISION_DT,
        bounds=CLEAN_BOUNDS,
        limits=HARD_LIMITS,
        comfort_limits=COMFORT_LIMITS,
    )
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


def _audit_fields(scene_seed: int, info: Dict[str, Any]) -> Dict[str, Any]:
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
        "source": np.int8(0),
    }


# ---------------------------------------------------------------------------
# Episode rollout
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
    term_id = np.int16(TERMINATION_IDS.get(termination, 5))
    timeout = np.int8(1 if termination in ("max_step", "timeout") else 0)
    success_i8 = np.int8(success)
    for i, row in enumerate(rows):
        row["done"] = np.int8(1 if (done and i == len(rows) - 1) else 0)
        row["success"] = success_i8
        row["collision"] = np.int8(collision)
        row["off_road"] = np.int8(off_road)
        row["wrong_route"] = np.int8(wrong_route)
        row["stuck"] = np.int8(stuck)
        row["timeout"] = timeout
        row["termination_id"] = term_id
        row["expert_plan_valid"] = success_i8
        row["expert_failed"] = np.int8(0 if success else 1)


def run_episode(
    spec: Any, *, max_steps: int
) -> Optional[Tuple[Dict[str, np.ndarray], Dict[str, Any]]]:
    """Run one expert episode; return ``(arrays, episode_info)`` or ``None``."""
    env = build_env(spec, use_render=False, lru_size=0)
    try:
        env.reset()
        ego = env.agent
        policy = env.engine.add_policy(
            ego.id, _expert_class(), ego, int(getattr(spec, "seed", 0))
        )
        if hasattr(policy, "reset"):
            policy.reset()

        frames: List[Dict[str, Any]] = []
        poses: List[np.ndarray] = []
        rows: List[Dict[str, Any]] = []
        prev_action = np.zeros(ACTION_DIM, dtype=np.float64)
        prev_speed: Optional[float] = None
        prev_heading: Optional[float] = None
        ep_step = 0
        done = False
        termination = "max_step"
        success = collision = off_road = 0

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
            hist = stack_history(frames, poses)
            scene_seed = int(obs["meta"]["scene_seed"])

            p0 = poses[-1].copy()
            v0 = float(ego.speed)
            h0 = float(ego.heading_theta)

            # Query the expert at the decision state: same stateless call the
            # engine will make for the first 0.1 s substep.
            expert_action = policy.act()
            steer = float(expert_action[0])
            throttle = float(expert_action[1])
            action_info = dict(getattr(policy, "action_info", {}) or {})
            lift = _lift_action(
                steer,
                throttle,
                speed=v0,
                max_steer_angle_rad=float(policy.max_steer_angle_rad),
                wheelbase=float(policy.wheelbase),
                accel_action_scale=float(policy.accel_action_scale),
                brake_action_scale=float(policy.brake_action_scale),
            )

            # The expert physically drives for one 0.5 s segment.
            term_any = False
            info_last: Dict[str, Any] = {}
            n_substeps = 0
            for _ in range(STEPS_PER_POLICY):
                _, _, term, trunc, info = env.step([0.0, 0.0])
                ep_step += 1
                n_substeps += 1
                info_last = info or {}
                if term or trunc:
                    term_any = True
                    done = True
                    if info_last.get("arrive_dest"):
                        termination = "arrive"
                        success = 1
                    elif info_last.get("crash"):
                        termination = "collision"
                        collision = 1
                    elif info_last.get("out_of_road"):
                        termination = "out_of_road"
                        off_road = 1
                    elif trunc and not term:
                        termination = "timeout"
                    else:
                        termination = "terminal"
                    break

            if term_any:
                # Do not save a partial 0.5 s label (same policy as collect.py).
                break

            p1 = np.asarray(ego.position, dtype=np.float64)[:2]
            h1 = float(ego.heading_theta)
            v1 = float(ego.speed)
            dxw, dyw = float(p1[0] - p0[0]), float(p1[1] - p0[1])
            c0, s0 = math.cos(h0), math.sin(h0)
            rel = np.array(
                [c0 * dxw + s0 * dyw, -s0 * dxw + c0 * dyw], dtype=np.float64
            )
            action = np.array(
                [rel[0], rel[1], _wrap(h1 - h0), v1], dtype=np.float32
            )
            if not (np.all(np.isfinite(action)) and math.isfinite(v0) and v0 >= 0.0):
                termination = "invalid_action"
                break

            diagnostics = _label_diagnostics(action, v0)
            audit = _audit_fields(scene_seed, action_info)
            residual = lift.astype(np.float64) - action.astype(np.float64)

            row: Dict[str, Any] = {key: hist[key] for key in HISTORY_KEYS}
            row.update(
                {
                    "action": action,
                    "action_lift": lift,
                    "residual_xy": np.array(
                        [residual[0], residual[1]], dtype=np.float32
                    ),
                    "residual_xy_norm": np.float32(
                        float(np.linalg.norm(residual[:2]))
                    ),
                    "residual_dpsi": np.float32(_wrap(float(residual[2]))),
                    "residual_v": np.float32(float(residual[3])),
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
                    "label_source": np.int8(1),
                    "label_valid": np.int8(1),
                    "n_substeps": np.int16(n_substeps),
                    "step": np.int64(len(rows)),
                    "episode_id": np.int64(0),  # filled by the caller
                    "scene_seed": np.int64(scene_seed),
                    "done": np.int8(0),
                    "success": np.int8(0),
                    "collision": np.int8(0),
                    "off_road": np.int8(0),
                    "wrong_route": np.int8(0),
                    "stuck": np.int8(0),
                    "timeout": np.int8(0),
                    "termination_id": np.int16(0),
                    "expert_plan_valid": np.int8(0),
                    "expert_failed": np.int8(0),
                }
            )
            rows.append(row)

            prev_action = action.astype(np.float64)
            prev_speed = v0
            prev_heading = h0
            if len(frames) > HISTORY_FRAMES:
                frames = frames[-HISTORY_FRAMES:]
                poses = poses[-HISTORY_FRAMES:]

        if not rows:
            return None

        _backfill_outcomes(
            rows,
            termination=termination,
            done=done,
            success=success,
            collision=collision,
            off_road=off_road,
        )
        data = _pack_rows(rows)
        info = {
            "termination": termination,
            "done": bool(done),
            "success": int(success),
            "collision": int(collision),
            "off_road": int(off_road),
            "wrong_route": 0,
            "stuck": 0,
            "rows": len(rows),
            "env_steps": ep_step,
            "scene_seed": int(getattr(spec, "seed", 0)),
            "spec_id": getattr(spec, "id", None),
            "expert_class": type(policy).__name__,
            "expert_params": policy.params() if hasattr(policy, "params") else {},
        }
        return data, info
    finally:
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass


def _worker(
    args: Tuple[int, Any, int]
) -> Tuple[int, Optional[Dict[str, np.ndarray]], float, str, Optional[Dict[str, Any]]]:
    idx, spec, max_steps = args
    t0 = time.time()
    try:
        result = run_episode(spec, max_steps=max_steps)
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


def _label_summary(merged: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = int(merged["action"].shape[0])
    if n == 0:
        return {}
    residual = merged["residual_xy_norm"]
    substeps, sub_counts = np.unique(merged["n_substeps"], return_counts=True)
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
        "residual_xy_mean_m": float(np.mean(residual)),
        "residual_xy_p95_m": float(np.percentile(residual, 95)),
        "residual_xy_max_m": float(np.max(residual)),
        "residual_dpsi_abs_mean_rad": float(np.mean(np.abs(merged["residual_dpsi"]))),
        "residual_v_abs_mean_mps": float(np.mean(np.abs(merged["residual_v"]))),
        "n_substeps_histogram": {
            int(k): int(v) for k, v in zip(substeps, sub_counts)
        },
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
        "lane_change_rows": int(
            np.count_nonzero(shard["expert_lane_change_decision"])
        ),
        "unique_target_lane_ids": int(
            len(np.unique(target_ids[target_ids != 0]))
        ),
        "mean_residual_xy_m": float(np.mean(residual)),
        "max_residual_xy_m": float(np.max(residual)),
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
    parser = argparse.ArgumentParser(description="Collect schema-v2 expert BC data")
    parser.add_argument("--specs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args(argv)

    specs = load_specs(args.specs)
    offset = max(0, int(args.offset))
    specs = specs[offset:]
    if args.limit is not None:
        specs = specs[: max(0, int(args.limit))]
    if not specs:
        print("[collect] no specs")
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(i, spec, int(args.max_steps)) for i, spec in enumerate(specs)]
    results: List[
        Tuple[int, Optional[Dict[str, np.ndarray]], float, str, Optional[Dict[str, Any]]]
    ] = []
    if args.workers <= 1:
        for task in tasks:
            results.append(_worker(task))
    else:
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=int(args.workers)) as pool:
            for result in pool.imap_unordered(_worker, tasks):
                results.append(result)
                if len(results) % 10 == 0:
                    print(f"[collect] {len(results)}/{len(tasks)}", flush=True)

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
        print(f"[collect] all failed: {failed_indices}")
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
    expert_class = shards[0][2].get("expert_class")
    expert_params = shards[0][2].get("expert_params", {})

    meta: Dict[str, Any] = {
        "schema": "pnp_v2",
        "schema_version": 2,
        "dataset": "expert_bc",
        "collector": "dataset.collect",
        "obs_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "specs": str(args.specs),
        "out": str(out_dir),
        "limit": args.limit,
        "offset": args.offset,
        "workers": args.workers,
        "max_steps": args.max_steps,
        "decision_dt": DECISION_DT,
        "sim_dt": SIM_DT,
        "steps_per_policy": STEPS_PER_POLICY,
        "history_frames": HISTORY_FRAMES,
        "seeds": [int(getattr(spec, "seed", 0)) for spec in specs],
        "spec_ids": [getattr(spec, "id", None) for spec in specs],
        "git_commit": _git_commit(),
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
            "source": 1,
            "source_name": "measured_replay",
            "definition": (
                "endpoint measured from ego pose/speed after 5 x 0.1 s expert "
                "replay relative to the pre-step pose"
            ),
            "lift": "5 x 0.1 s kinematic bicycle holding expert (steer, throttle)",
            **_label_summary(merged),
        },
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
            "expert_failed_episodes": int(
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

    label = meta["label"]
    print(
        f"[collect] wrote {npz_path} rows={rows} episodes={len(shards)} "
        f"failed={len(failed_indices)} feasible={label.get('feasible_rate', 0.0):.4f} "
        f"pass={label.get('pass_rate', 0.0):.4f}"
    )
    print(f"[collect] meta {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
