#!/usr/bin/env python3
"""P0 spike ④: can we harvest BC expert data from MetaDrive's built-in IDMPolicy?

Checks:
  1. env with `agent_policy=IDMPolicy` drives the ego along the navigation route
  2. navigation interface (checkpoints + navigation_command) available per step
  3. expert future targets can be built at 0.5 s spacing (6 points / 3 s)
  4. episode outcome: route_completion / crash / timeout

Usage:
    .venv/bin/python tools/measure/idm_expert_spike.py --episodes 3 --map 3
"""
import argparse
import json
import os
import time


def run_episode(seed: int, map_cfg, max_steps: int, density: float):
    from metadrive.envs import MetaDriveEnv
    from metadrive.policy.idm_policy import IDMPolicy

    env = MetaDriveEnv(
        dict(
            use_render=False,
            log_level=50,
            num_scenarios=1,
            start_seed=seed,
            map=map_cfg,
            traffic_density=density,
            preload_models=False,
            store_map=False,
            agent_policy=IDMPolicy,
        )
    )
    trace = []
    try:
        env.reset()
        for step in range(max_steps):
            _, _, terminated, truncated, info = env.step([0.0, 0.0])  # IDM ignores external action
            v = env.vehicle
            nav = getattr(v, "navigation", None)
            checkpoints = nav.get_checkpoints() if nav is not None else []
            trace.append(
                {
                    "t": round(step * 0.1, 2),
                    "x": round(float(v.position[0]), 3),
                    "y": round(float(v.position[1]), 3),
                    "heading": round(float(v.heading_theta), 4),
                    "speed": round(float(v.speed_km_h), 2),
                    "route_completion": round(float(nav.route_completion), 4) if nav is not None else None,
                    "nav_command": info.get("navigation_command"),
                    "n_checkpoints": len(checkpoints),
                    "crash": bool(info.get("crash")),
                    "out_of_road": bool(info.get("out_of_road")),
                    "arrive_dest": bool(info.get("arrive_dest")),
                }
            )
            if terminated or truncated:
                break
        if trace:
            trace[-1]["end"] = "terminated" if terminated else ("truncated" if truncated else "max_steps")
    finally:
        env.close()

    # build 0.5 s-spaced 6-point targets (env step = 0.1 s)
    stride = 5
    horizon = 6
    valid_targets = 0
    for i in range(len(trace)):
        if i + stride * horizon < len(trace):
            valid_targets += 1
    return trace, valid_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--map", default="3")
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--max-steps", type=int, default=1500)  # 150 s at 10 Hz
    parser.add_argument("--out", default="runs/p0_idm_spike_sample.json")
    args = parser.parse_args()
    map_cfg = int(args.map) if args.map.isdigit() else args.map

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    summary = []
    for k in range(args.episodes):
        seed = args.start_seed + k
        t0 = time.time()
        trace, valid = run_episode(seed, map_cfg, args.max_steps, args.density)
        dt = time.time() - t0
        if not trace:
            print(f"seed={seed} EMPTY TRACE")
            continue
        last = trace[-1]
        n_crash = sum(1 for r in trace if r["crash"])
        summary.append(
            {
                "seed": seed,
                "steps": len(trace),
                "valid_6pt_targets": valid,
                "route_completion": last["route_completion"],
                "crashed": n_crash > 0,
                "out_of_road_any": any(r["out_of_road"] for r in trace),
                "end": last.get("end"),
                "nav_cmds": sorted({r["nav_command"] for r in trace}),
                "wall_s": round(dt, 2),
            }
        )
        print(
            f"seed={seed} steps={len(trace)} valid_targets={valid} "
            f"route_completion={last['route_completion']} crashed={n_crash > 0} "
            f"out_of_road={any(r['out_of_road'] for r in trace)} end={last.get('end')} "
            f"nav_cmds={sorted({r['nav_command'] for r in trace})} wall={dt:.1f}s"
        )
        if k == 0:
            with open(args.out, "w") as fh:
                json.dump(trace[:200], fh, indent=1)
            print(f"sample trace (first 200 steps) -> {args.out}")

    print("SUMMARY:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
