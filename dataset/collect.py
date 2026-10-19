"""Collect structured expert episodes for P1 BC.

Each row is one 0.5 s policy decision:
  history tensors (6 frames) + expert action (dx,dy,dpsi,v_next).

The expert is the deterministic PurePursuitIDMPolicy in env/expert.
The environment is stepped physically by the expert for 5 x 0.1 s per label.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from env.metadrive_env import build_env
from env.obs import (
    EGO_DIM, HISTORY_FRAMES, LD_DIM, LD_POINTS, LD_SLOTS, NAV_DIM,
    OD_DIM, OD_SLOTS, build_observation, stack_history,
)
from env.scenario.spec import load_specs

STEPS_PER_POLICY = 5
ACTION_DIM = 4


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def _expert_class():
    from env.expert.pure_pursuit_idm import PurePursuitIDMPolicy
    return PurePursuitIDMPolicy


def _to_local_xy(ego: Any, xy: Sequence[float]) -> np.ndarray:
    p = np.asarray(ego.convert_to_local_coordinates(np.asarray(xy, dtype=np.float64), ego.position), dtype=np.float64)
    return p[:2]


def run_episode(spec: Any, *, max_steps: int) -> Dict[str, np.ndarray] | None:
    env = build_env(spec, use_render=False, lru_size=0)
    try:
        env.reset()
        ego = env.agent
        policy = env.engine.add_policy(ego.id, _expert_class(), ego, int(getattr(spec, "seed", 0)))
        if hasattr(policy, "reset"):
            policy.reset()

        frames: List[Dict[str, np.ndarray]] = []
        poses: List[np.ndarray] = []
        rows: List[Dict[str, np.ndarray]] = []
        prev_action = np.zeros(ACTION_DIM, dtype=np.float64)
        prev_speed = None
        prev_heading = None
        ep_step = 0
        done = False
        termination = "max_step"
        success = collision = off_road = 0.0

        while ep_step < max_steps:
            # current policy observation
            obs = build_observation(
                env, prev_action=prev_action,
                prev_speed=prev_speed,
                prev_heading=prev_heading,
                dt=0.5,
            )
            frames.append(obs)
            poses.append(obs["pose"].copy())
            hist = stack_history(frames, poses)
            p0 = poses[-1].copy()
            v0 = float(ego.speed)
            h0 = float(ego.heading_theta)

            # execute the expert for one 0.5 s segment
            term_any = False
            info_last: Dict[str, Any] = {}
            for _ in range(STEPS_PER_POLICY):
                _, _, term, trunc, info = env.step([0.0, 0.0])
                ep_step += 1
                info_last = info or {}
                if term or trunc:
                    term_any = True
                    done = True
                    if info_last.get("arrive_dest"):
                        termination = "arrive"
                        success = 1.0
                    elif info_last.get("crash"):
                        termination = "collision"
                        collision = 1.0
                    elif info_last.get("out_of_road"):
                        termination = "out_of_road"
                        off_road = 1.0
                    else:
                        termination = "terminal"
                    break

            if term_any:
                # no valid 0.5 s label after termination
                break

            p1 = np.asarray(ego.position, dtype=np.float64)[:2]
            h1 = float(ego.heading_theta)
            v1 = float(ego.speed)
            # p1 expressed in the policy frame p0 (x forward, y left)
            dxw, dyw = float(p1[0] - p0[0]), float(p1[1] - p0[1])
            c0, s0 = math.cos(h0), math.sin(h0)
            rel = np.array([c0 * dxw + s0 * dyw, -s0 * dxw + c0 * dyw], dtype=np.float64)
            action = np.array([rel[0], rel[1], _wrap(h1 - h0), v1], dtype=np.float32)
            # reject pathological labels (expert should not produce them, but fail loud in stats)
            if not np.all(np.isfinite(action)):
                break
            rows.append({
                "od": hist["od"], "od_mask": hist["od_mask"],
                "ld": hist["ld"], "ld_mask": hist["ld_mask"],
                "nav": hist["nav"], "nav_mask": hist["nav_mask"],
                "ego": hist["ego"], "ego_mask": hist["ego_mask"],
                "hist_valid": hist["hist_valid"],
                "action": action,
                "episode_id": np.int64(0),  # filled by caller
                "step": np.int64(ep_step - STEPS_PER_POLICY),
                "done": np.float32(0.0),
                "success": np.float32(0.0),
                "collision": np.float32(0.0),
                "off_road": np.float32(0.0),
                "scene_seed": np.int64(int(getattr(spec, "seed", 0))),
            })
            prev_action = action.astype(np.float64)
            prev_speed = v0
            prev_heading = h0
            # p1 is the next policy frame; keep it for the next obs
            if len(frames) > HISTORY_FRAMES:
                frames = frames[-HISTORY_FRAMES:]
                poses = poses[-HISTORY_FRAMES:]

        if not rows:
            return None
        out = {k: np.stack([r[k] for r in rows], axis=0) for k in rows[0]}
        # mark final row done/termination summary
        out["done"][-1] = 1.0 if done else 0.0
        out["success"][-1] = np.float32(success)
        out["collision"][-1] = np.float32(collision)
        out["off_road"][-1] = np.float32(off_road)
        out["termination_id"] = np.full((len(rows),), {"max_step": 0, "arrive": 1, "collision": 2, "out_of_road": 3, "terminal": 4}.get(termination, 5), dtype=np.int16)
        return out
    finally:
        try:
            env.close()
        except Exception:
            pass


def _worker(args: tuple[int, Any, int]) -> tuple[int, Dict[str, np.ndarray] | None, float, str]:
    idx, spec, max_steps = args
    t0 = time.time()
    try:
        data = run_episode(spec, max_steps=max_steps)
        return idx, data, time.time() - t0, "ok"
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return idx, None, time.time() - t0, f"{type(exc).__name__}: {exc}"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--specs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args(argv)

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

    tasks = [(i, s, int(args.max_steps)) for i, s in enumerate(specs)]
    results: List[tuple[int, Dict[str, np.ndarray] | None, float, str]] = []
    if args.workers <= 1:
        for t in tasks:
            results.append(_worker(t))
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=int(args.workers)) as pool:
            for r in pool.imap_unordered(_worker, tasks):
                results.append(r)
                if len(results) % 10 == 0:
                    print(f"[collect] {len(results)}/{len(tasks)}", flush=True)

    results.sort(key=lambda x: x[0])
    shards = []
    ok = 0
    fail = 0
    for idx, data, dt, status in results:
        if data is None:
            fail += 1
            continue
        data["episode_id"] = np.full((len(data["action"]),), idx, dtype=np.int64)
        shards.append(data)
        ok += 1
    if not shards:
        print(f"[collect] all failed: {fail}")
        return 1

    keys = shards[0].keys()
    merged = {k: np.concatenate([s[k] for s in shards], axis=0) for k in keys}
    npz_path = out_dir / "expert_bc.npz"
    np.savez_compressed(npz_path, **merged)
    meta = {
        "schema": "pnp_v1",
        "specs": str(args.specs),
        "limit": args.limit,
        "workers": args.workers,
        "max_steps": args.max_steps,
        "episodes_ok": ok,
        "episodes_fail": fail,
        "rows": int(len(merged["action"])),
        "steps_per_policy": STEPS_PER_POLICY,
        "history_frames": HISTORY_FRAMES,
        "action_dim": ACTION_DIM,
        "shapes": {k: list(v.shape) for k, v in merged.items()},
        "features": {
            "od": ["dx","dy","cos_theta","sin_theta","vx_rel","vy_rel","length","width","type_id","static_flag","presence"],
            "ld": ["dx","dy","heading_rel","curvature","speed_limit","left_type","right_type","lane_role","lat_offset"],
            "nav": ["cp0_dx","cp0_dy","cp1_dx","cp1_dy","forward","left","right","route_completion","r0","r1","r2"],
            "ego": ["speed","a_long","a_lat","yaw_rate","steer","curvature","prev_dx","prev_dpsi"],
            "action": ["dx","dy","dpsi","v_next"],
        },
        "failed_indices": [int(i) for i, d, _, _ in results if d is None],
    }
    (out_dir / "expert_bc.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[collect] wrote {npz_path} rows={len(merged['action'])} episodes={ok} failed={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
