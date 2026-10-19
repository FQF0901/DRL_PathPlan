"""Closed-loop P1 evaluation with LQR tracking (3 s preview, 5 x 0.1 s steps)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from env.metadrive_env import build_env
from env.obs import build_observation, stack_history
from env.scenario.spec import load_specs
from env.tracking import LqrTracker
from net.model import StructuredPlanner
from pipeline.rollout import batch_from_obs, first_segment_points, plan_actions

STEPS_PER_POLICY = 5


def run_episode(env: Any, model: StructuredPlanner, spec: Any, *, max_steps: int, device: str) -> Dict[str, Any]:
    env.reset()
    ego = env.agent
    tracker = env.engine.add_policy(ego.id, LqrTracker, ego, int(getattr(spec, "seed", 0)))
    frames: List[Dict[str, np.ndarray]] = []
    poses: List[np.ndarray] = []
    prev_action = np.zeros(4, dtype=np.float64)
    prev_speed = prev_heading = None
    success = collision = off_road = False
    rc = 0.0
    n = 0
    fail_reason = "max_step"
    while n < max_steps:
        obs = build_observation(env, prev_action=prev_action, prev_speed=prev_speed, prev_heading=prev_heading, dt=0.5)
        frames.append(obs)
        poses.append(obs["pose"].copy())
        if len(frames) > 6:
            frames = frames[-6:]
            poses = poses[-6:]
        hist = stack_history(frames, poses)
        actions = plan_actions(model, batch_from_obs(hist, device=device), steps=6)
        # P1 canary: only the current 0.5 s segment, not the unvalidated 3 s preview.
        ref = first_segment_points(actions[0])
        tracker.set_reference(ref, base_pose=obs["pose"])
        done = False
        for _ in range(STEPS_PER_POLICY):
            _, _, term, trunc, info = env.step([0.0, 0.0])
            n += 1
            if info:
                rc = float(info.get("route_completion", rc))
                success = success or bool(info.get("arrive_dest", False))
                collision = collision or bool(info.get("crash", False))
                off_road = off_road or bool(info.get("out_of_road", False))
            if term or trunc:
                done = True
                fail_reason = "arrive" if success else ("collision" if collision else ("out_of_road" if off_road else "terminal"))
                break
        prev_action = actions[0].astype(np.float64)
        prev_speed = float(ego.speed)
        prev_heading = float(ego.heading_theta)
        if done:
            break
    return {
        "scene_seed": int(getattr(spec, "seed", 0)),
        "steps": n,
        "success": bool(success),
        "collision": bool(collision),
        "off_road": bool(off_road),
        "route_completion": float(rc),
        "termination": fail_reason,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--specs", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = StructuredPlanner(hidden=int(ck.get("config", {}).get("hidden", 128)))
    model.load_state_dict(ck["model"])
    model.to(args.device).eval()
    specs = load_specs(args.specs)[: int(args.limit)]
    rows = []
    for i, spec in enumerate(specs):
        env = build_env(spec, use_render=False)
        try:
            r = run_episode(env, model, spec, max_steps=int(args.max_steps), device=args.device)
        finally:
            env.close()
        rows.append(r)
        print(f"[lqr] {i+1}/{len(specs)} {r}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = {
        "n": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
        "collision_rate": float(np.mean([r["collision"] for r in rows])) if rows else 0.0,
        "off_road_rate": float(np.mean([r["off_road"] for r in rows])) if rows else 0.0,
        "route_completion_mean": float(np.mean([r["route_completion"] for r in rows])) if rows else 0.0,
        "episodes": rows,
    }
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
