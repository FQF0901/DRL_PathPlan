"""Closed-loop evaluation for the canonical planner.

The previous exact/LQR split is available only from git history/tag
``pre-consolidation``:

* observation history comes from :mod:`env.obs` (``build_observation`` +
  ``stack_history``);
* the policy is :class:`net.model.StructuredPlanner` with normalized
  ``action_mu`` in ``[0, 1]``;
* actions are denormalized with :data:`net.action_space.CLEAN_BOUNDS` and turned
  into one PSTL 0.5 s segment by ``action_to_trajectory``.

Only the **first 0.5 s segment** of a plan is executed (5 x 0.1 s substeps).
The 6-step / world-model preview is deliberately **not implemented** here.

Feasibility is diagnosed, never silently repaired: if the requested action
fails ``ActionTrajectory.feasible``, a safe-brake action is built explicitly,
the executed (fallback) action is what the tracker receives, and both the
requested codes and the fallback flag are written to the episode row.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from metadrive.utils.math import wrap_to_pi

from env.metadrive_env import build_env
from env.obs import (
    HISTORY_FRAMES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_observation,
    stack_history,
)
from env.scenario.spec import load_specs
from env.tracking import ExactTracker, LqrTracker
from net.action_space import (
    CLEAN_BOUNDS,
    COMFORT_LIMITS,
    HARD_LIMITS,
    DynamicsLimits,
    action_to_trajectory,
    feasible_v1_interval,
)
from net.model import H, HEADS, SPATIAL_LAYERS, StructuredPlanner
from dataset.splits import verify_split_manifest
from pipeline.checkpointing import load_model_from_checkpoint
from pipeline.execution import limit_history, relative_action
from pipeline.eval_metrics import summarize_rows

DECISION_DT = 0.5
SIM_DT = 0.1
STEPS_PER_POLICY = 5
ACTION_DIM = 4
HZ = 10

#: A stuck episode is >20 consecutive 0.1 s env steps below this speed.
STUCK_SPEED_MPS = 0.1
STUCK_STEPS = 20
#: Conservative wrong-route signal: route_completion drops by more than this
#: within one 0.5 s decision segment.
WRONG_ROUTE_DROP = 0.05

# ---------------------------------------------------------------------------
# Canonical PSTL limits
# ---------------------------------------------------------------------------
LIMITS_SOURCE = "net.action_space.HARD_LIMITS"

#: Keys StructuredPlanner.forward consumes (metadata IDs are not needed for acting).
BATCH_KEYS = (
    "od", "od_mask", "ld", "ld_mask", "nav", "nav_mask",
    "ego", "ego_mask", "hist_valid",
)


# ---------------------------------------------------------------------------
# Model / observation helpers
# ---------------------------------------------------------------------------
def _config_int(config: Any, key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (AttributeError, TypeError, ValueError):
        return int(default)


def load_model(ckpt: Optional[str], device: str = "cpu", *, strict: bool = False) -> StructuredPlanner:
    """Load a checkpoint, validating schema/bounds when metadata is present.

    ``strict=True`` rejects legacy/incompatible metadata; the default keeps a
    warning-only path for old smoke tests. Gate runs must use strict mode.
    """
    if ckpt is None:
        print(
            "[eval] WARNING: no --ckpt given; using a randomly initialized "
            "StructuredPlanner for smoke only",
            flush=True,
        )
        model = StructuredPlanner()
        model.eval_config = {
            "hidden": H,
            "heads": HEADS,
            "spatial_layers": SPATIAL_LAYERS,
        }
    else:
        model = load_model_from_checkpoint(ckpt, device, strict=strict)
        for warning in getattr(model, "checkpoint_warnings", []):
            print(f"[eval] checkpoint warning: {warning}", flush=True)
    model.to(device).eval()
    return model

def _batch_from_history(hist: Dict[str, Any], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Batch-first float32 tensors for one history window."""
    batch: Dict[str, torch.Tensor] = {}
    for key in BATCH_KEYS:
        arr = np.asarray(hist[key], dtype=np.float32)
        batch[key] = torch.from_numpy(arr).unsqueeze(0).to(device)
    return batch


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


def _wrap_angle(value: Any) -> float:
    return float(wrap_to_pi(float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _info_flag(info: Dict[str, Any], key: str) -> bool:
    value = info.get(key, False)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return False


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(finite)) if finite else None


def _add_codes(counts: Dict[str, int], codes: Sequence[str]) -> None:
    for code in codes:
        key = str(code)
        counts[key] = counts.get(key, 0) + 1


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
def run_episode(
    env: Any,
    model: StructuredPlanner,
    spec: Any,
    *,
    controller: str = "exact",
    max_steps: int = 600,
    device: str = "cpu",
    lqr_kwargs: Optional[Dict[str, Any]] = None,
    history_frames: int = HISTORY_FRAMES,
) -> Dict[str, Any]:
    """Run one closed-loop episode with 0.5 s segment execution.

    ``max_steps`` counts 0.1 s ``env.step`` calls, matching the canonical
    evaluators.  Only the first segment of each planner query is executed;
    the 6-step / WM preview is not implemented.
    """
    if controller not in ("exact", "lqr"):
        raise ValueError(f"controller must be 'exact' or 'lqr', got {controller!r}")
    max_steps = max(int(max_steps), 0)
    model.to(device).eval()

    env.reset()
    ego = env.agent
    if controller == "exact":
        tracker: Any = ExactTracker(dt=DECISION_DT, hz=HZ)
    else:
        tracker = env.engine.add_policy(
            ego.id,
            LqrTracker,
            ego,
            int(getattr(spec, "seed", 0)),
            **(lqr_kwargs or {}),
        )

    frames: List[Dict[str, Any]] = []
    poses: List[np.ndarray] = []
    prev_action = np.zeros(ACTION_DIM, dtype=np.float64)
    prev_speed: Optional[float] = None
    prev_heading: Optional[float] = None

    n = 0
    policy_steps = 0
    success = collision = off_road = wrong_route = False
    wrong_route_drops = 0
    done = False
    timeout = False
    termination = "max_step"
    rc = 0.0
    low_speed_run = 0
    stuck_run = False

    infeasible_count = 0
    fallback_count = 0
    comfort_violation = False
    max_curvature = max_lateral_accel = max_jerk = 0.0
    terminal_speed_error: Optional[float] = None
    track_xy: List[float] = []
    track_heading: List[float] = []
    track_speed: List[float] = []
    steer_saturation_count = 0
    throttle_saturation_count = 0
    control_steps = 0
    requested_code_counts: Dict[str, int] = {}
    executed_code_counts: Dict[str, int] = {}
    requested_action = np.zeros(ACTION_DIM, dtype=np.float64)
    first_requested_action: Optional[np.ndarray] = None
    postprocess_count = 0
    executed_action = np.zeros(ACTION_DIM, dtype=np.float64)

    while n < max_steps:
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
        hist = limit_history(hist, int(history_frames))
        batch = _batch_from_history(hist, device=device)

        with torch.no_grad():
            out = model(batch)
        a_norm = out["action_mu"][0].detach().cpu().numpy().astype(np.float64)

        v0 = float(ego.speed)
        h0 = float(ego.heading_theta)
        p0 = np.asarray(ego.position[:2], dtype=np.float64).copy()
        requested_action = CLEAN_BOUNDS.denormalize(a_norm, clip=True)
        if first_requested_action is None:
            first_requested_action = requested_action.copy()
        requested_codes: Sequence[str] = ()
        requested_tr = None
        if np.all(np.isfinite(requested_action)):
            requested_tr = _trajectory(requested_action, v0)
            requested_codes = requested_tr.codes
            _add_codes(requested_code_counts, requested_codes)

        requested_feasible = bool(requested_tr is not None and requested_tr.feasible)
        if requested_feasible:
            tr_exec = requested_tr
            executed_action = requested_action.copy()
        else:
            infeasible_count += 1
            if requested_tr is None:
                requested_codes = ("NONFINITE_ACTION",)
                _add_codes(requested_code_counts, requested_codes)
            # First try the least-rewrite fallback: keep geometry and project only
            # v_next into the PSTL hard-feasible interval.  This is an explicit
            # evaluation post-process, never a PPO log-prob rewrite.
            projected = False
            if np.all(np.isfinite(requested_action)):
                interval = feasible_v1_interval(
                    requested_action,
                    v0,
                    dt=DECISION_DT,
                    bounds=CLEAN_BOUNDS,
                    limits=HARD_LIMITS,
                )
                if interval.nonempty:
                    candidate = requested_action.copy()
                    candidate[3] = float(np.clip(candidate[3], interval.low, interval.high))
                    candidate_tr = _trajectory(candidate, v0)
                    if candidate_tr.feasible:
                        executed_action = candidate
                        tr_exec = candidate_tr
                        projected = True
                        postprocess_count += 1
            if not projected:
                fallback_count += 1
                executed_action = safe_brake_action(v0)
                tr_exec = _trajectory(executed_action, v0)
                fallback = True

        # A velocity-profile trajectory must never reach LQR with a negative
        # speed sample; old action gates could accept such a numerical corner.
        speeds = np.asarray(getattr(tr_exec, "speeds", []), dtype=np.float64).reshape(-1)
        if (
            tr_exec is None
            or speeds.size == 0
            or not np.all(np.isfinite(speeds))
            or float(speeds.min()) < -1e-6
        ):
            fallback_count += 1
            fallback = True
            executed_action = safe_brake_action(v0)
            tr_exec = _trajectory(executed_action, v0)
            speeds = np.asarray(tr_exec.speeds, dtype=np.float64)
        _add_codes(executed_code_counts, tr_exec.codes)
        policy_steps += 1

        comfort_violation = comfort_violation or (not bool(tr_exec.comfortable))
        metrics = tr_exec.metrics
        max_curvature = max(max_curvature, _safe_float(metrics.get("max_abs_kappa")))
        max_lateral_accel = max(
            max_lateral_accel, _safe_float(metrics.get("max_abs_a_lat"))
        )
        max_jerk = max(max_jerk, _safe_float(metrics.get("max_abs_jerk_long")))

        if controller == "exact":
            tracker.arm(env, trajectory=tr_exec)
        else:
            ref_positions = np.asarray(tr_exec.positions, dtype=np.float64)
            if ref_positions.ndim != 2 or ref_positions.shape[1] < 3:
                raise ValueError(f"invalid LQR positions shape {ref_positions.shape}")
            ref_speeds = np.asarray(tr_exec.speeds, dtype=np.float64).reshape(-1)
            if ref_speeds.shape != (ref_positions.shape[0],):
                raise ValueError(
                    f"LQR speed/position length mismatch {ref_speeds.shape} vs {ref_positions.shape}"
                )
            # Tiny negative values (e.g. -1e-9) are numerical dust from the
            # cubic time law; clamp them before the LQR non-negative check.
            ref_speeds = np.maximum(ref_speeds, 0.0)
            tracker.set_reference(
                np.column_stack([ref_positions[:, :3], ref_speeds]),
                base_pose=obs["pose"],
            )

        segment_rc_start = rc
        segment_steps = 0
        for _ in range(STEPS_PER_POLICY):
            if n >= max_steps:
                break
            speed_before = float(ego.speed)
            _, _, term, trunc, info = env.step([0.0, 0.0])
            n += 1
            segment_steps += 1
            info = info or {}

            if controller == "exact":
                realized = tracker.apply(env)
                index = int(tracker.index) - 1
                reference = tracker.reference[index]
                track_xy.append(float(np.linalg.norm(realized[:2] - reference[:2])))
                track_heading.append(
                    abs(_wrap_angle(realized[2] - reference[2]))
                )
            else:
                action_info = dict(getattr(tracker, "action_info", {}) or {})
                if "lqr_error_y" in action_info:
                    track_xy.append(abs(_safe_float(action_info["lqr_error_y"])))
                if "lqr_error_psi" in action_info:
                    track_heading.append(
                        abs(_wrap_angle(action_info["lqr_error_psi"]))
                    )
                if "lqr_speed_error" in action_info:
                    track_speed.append(abs(_safe_float(action_info["lqr_speed_error"])))
                elif "lqr_ref_speed_mps" in action_info:
                    track_speed.append(
                        abs(speed_before - _safe_float(action_info["lqr_ref_speed_mps"]))
                    )
                if "action" in action_info:
                    act = action_info["action"]
                    steer = _safe_float(act[0]) if len(act) > 0 else 0.0
                    throttle = _safe_float(act[1]) if len(act) > 1 else 0.0
                    steer_raw = abs(_safe_float(action_info.get("steer_raw", steer)))
                    throttle_raw = abs(
                        _safe_float(action_info.get("throttle_raw", throttle))
                    )
                    control_steps += 1
                    if max(abs(steer), steer_raw) >= 1.0 - 1e-6:
                        steer_saturation_count += 1
                    if max(abs(throttle), throttle_raw) >= 1.0 - 1e-6:
                        throttle_saturation_count += 1

            if "route_completion" in info:
                rc = _safe_float(info["route_completion"], rc)
            success = success or _info_flag(info, "arrive_dest")
            collision = collision or _info_flag(info, "crash")
            off_road = off_road or _info_flag(info, "out_of_road")
            wrong_route = (
                wrong_route
                or _info_flag(info, "wrong_route")
                or _info_flag(info, "wrong_way")
            )

            speed_now = float(ego.speed)
            if speed_now < STUCK_SPEED_MPS:
                low_speed_run += 1
                if low_speed_run > STUCK_STEPS:
                    stuck_run = True
            else:
                low_speed_run = 0

            if term or trunc:
                done = True
                if success:
                    termination = "arrive"
                elif collision:
                    termination = "collision"
                elif off_road:
                    termination = "out_of_road"
                elif trunc and not term:
                    termination = "timeout"
                else:
                    termination = "terminal"
                break

        if rc < segment_rc_start - WRONG_ROUTE_DROP:
            wrong_route = True
            wrong_route_drops += 1

        if segment_steps == STEPS_PER_POLICY:
            terminal_speed_error = abs(
                float(ego.speed) - float(tr_exec.speeds[-1])
            )
            if controller == "exact":
                # Exact tracking realizes the planned terminal speed exactly;
                # keep the same definition in tracking_error_speed so the
                # metric matches.
                track_speed.append(terminal_speed_error)

        # Match the collector: prev decision state is the segment START, and
        # prev_action is the measured realized endpoint, not merely the command.
        realized_action = relative_action(
            p0,
            h0,
            v0,
            np.asarray(ego.position[:2], dtype=np.float64),
            float(ego.heading_theta),
            float(ego.speed),
        )
        prev_action = realized_action
        prev_speed = v0
        prev_heading = h0
        if done:
            break

    if not done:
        timeout = True
        termination = "timeout"
    stuck = bool(stuck_run and not success)

    return {
        "scene_seed": int(getattr(spec, "seed", 0)),
        "spec_id": int(spec.id) if getattr(spec, "id", None) is not None else None,
        "geometry": (getattr(spec, "geometry", None) or ["unknown"])[0],
        "difficulty": getattr(spec, "difficulty", None),
        "controller": controller,
        "steps": int(n),
        "policy_steps": int(policy_steps),
        "success": bool(success),
        "collision": bool(collision),
        "off_road": bool(off_road),
        "wrong_route": bool(wrong_route),
        "stuck": bool(stuck),
        "timeout": bool(timeout),
        "termination": str(termination),
        "route_completion": float(rc),
        "route_completion_drop_events": int(wrong_route_drops),
        "comfort_violation": bool(comfort_violation),
        "feasible": bool(infeasible_count == 0),
        "fallback": bool(fallback_count > 0),
        "infeasible_count": int(infeasible_count),
        "fallback_count": int(fallback_count),
        "postprocess_count": int(postprocess_count),
        "v1_projection_count": int(postprocess_count),
        "first_requested_action": (
            None if first_requested_action is None else first_requested_action.tolist()
        ),
        "max_curvature": float(max_curvature),
        "max_lateral_accel": float(max_lateral_accel),
        "max_jerk": float(max_jerk),
        "tracking_error_xy": _mean_or_none(track_xy),
        "tracking_error_heading": _mean_or_none(track_heading),
        "tracking_error_speed": _mean_or_none(track_speed),
        "terminal_speed_error": terminal_speed_error,
        "steer_saturation": (
            float(steer_saturation_count / control_steps) if control_steps else 0.0
        ),
        "throttle_saturation": (
            float(throttle_saturation_count / control_steps) if control_steps else 0.0
        ),
        "control_steps": int(control_steps),
        "requested_action": requested_action.tolist(),
        "executed_action": executed_action.tolist(),
        "realized_action": realized_action.tolist(),
        "requested_action_codes": requested_code_counts,
        "executed_action_codes": executed_code_counts,
        "preview": "NOT_IMPLEMENTED: 6-step / world-model preview",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _limits_dict(limits: DynamicsLimits) -> Dict[str, float]:
    return {key: float(value) for key, value in asdict(limits).items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", default=None, help="optional; random init for smoke")
    parser.add_argument("--strict-checkpoint", action="store_true")
    parser.add_argument("--history-frames", type=int, default=HISTORY_FRAMES)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--controller", choices=("exact", "lqr"), default="exact")
    parser.add_argument("--lqr-speed-gain", type=float, default=None)
    parser.add_argument("--lqr-accel-scale", type=float, default=None)
    parser.add_argument("--lqr-brake-scale", type=float, default=None)
    parser.add_argument("--lqr-lookahead-time", type=float, default=None)
    parser.add_argument("--lqr-lookahead-base", type=float, default=None)
    parser.add_argument("--lqr-ref-speed-scale", type=float, default=None)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args(argv)

    model = load_model(args.ckpt, args.device, strict=bool(args.strict_checkpoint))
    lqr_kwargs: Dict[str, Any] = {}
    if args.lqr_speed_gain is not None:
        lqr_kwargs["speed_gain"] = float(args.lqr_speed_gain)
    if args.lqr_accel_scale is not None:
        lqr_kwargs["accel_action_scale"] = float(args.lqr_accel_scale)
    if args.lqr_brake_scale is not None:
        lqr_kwargs["brake_action_scale"] = float(args.lqr_brake_scale)
    if args.lqr_lookahead_time is not None:
        lqr_kwargs["lookahead_time"] = float(args.lqr_lookahead_time)
    if args.lqr_lookahead_base is not None:
        lqr_kwargs["lookahead_base"] = float(args.lqr_lookahead_base)
    if args.lqr_ref_speed_scale is not None:
        lqr_kwargs["ref_speed_scale"] = float(args.lqr_ref_speed_scale)
    all_specs = load_specs(args.specs)
    if args.split_manifest:
        verify_split_manifest(
            args.split_manifest,
            all_specs,
            split=args.split,
            spec_path=args.specs,
        )
    specs = all_specs[: int(args.limit)]

    rows: List[Dict[str, Any]] = []
    for i, spec in enumerate(specs):
        env = build_env(spec, use_render=False, lru_size=0)
        try:
            row = run_episode(
                env,
                model,
                spec,
                controller=args.controller,
                max_steps=int(args.max_steps),
                device=args.device,
                lqr_kwargs=lqr_kwargs,
                history_frames=int(args.history_frames),
            )
        finally:
            env.close()
        rows.append(row)
        print(f"[eval/{args.controller}] {i + 1}/{len(specs)} {row}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        **summarize_rows(rows),
        "schema": "drl_pathplan_eval",
        "schema_version": 1,
        "obs_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "controller": args.controller,
        "config": {
            "specs": str(args.specs),
            "split": args.split,
            "split_manifest": args.split_manifest,
            "ckpt": None if args.ckpt is None else str(args.ckpt),
            "limit": int(args.limit),
            "max_steps": int(args.max_steps),
            "controller": args.controller,
            "device": str(args.device),
            "lqr_kwargs": dict(lqr_kwargs),
            "decision_dt": DECISION_DT,
            "sim_dt": SIM_DT,
            "steps_per_policy": STEPS_PER_POLICY,
            "history_frames": int(args.history_frames),
            "action_bounds": CLEAN_BOUNDS.as_dict(),
            "limits": {
                "hard": _limits_dict(HARD_LIMITS),
                "comfort": _limits_dict(COMFORT_LIMITS),
                "source": LIMITS_SOURCE,
                "provisional": True,
            },
            "model": dict(getattr(model, "eval_config", {})),
            "execution": "first 0.5 s segment only (5 x 0.1 s)",
            "preview": "NOT_IMPLEMENTED: 6-step / world-model preview",
        },
        "episodes": rows,
    }
    (out / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({k: v for k, v in payload.items() if k != "episodes"},
                   ensure_ascii=False, indent=2),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
