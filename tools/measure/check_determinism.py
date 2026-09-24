#!/usr/bin/env python3
"""P0: seed determinism — same seed must produce the same map + rollout trace,
both across repeated builds in one process and across processes (spawn).

Usage:
    .venv/bin/python tools/measure/check_determinism.py --workers 3 --steps 50
"""
import argparse
import hashlib
import multiprocessing as mp


def _fingerprint(seed: int, map_cfg, steps: int) -> str:
    """Build env for `seed`, run fixed actions, hash rounded state trace."""
    import numpy as np
    from metadrive.envs import MetaDriveEnv

    env = MetaDriveEnv(
        dict(
            use_render=False,
            log_level=50,
            num_scenarios=1,
            start_seed=seed,
            map=map_cfg,
            traffic_density=0.1,
            preload_models=False,
            store_map=False,
        )
    )
    try:
        env.reset(seed=seed)
        trace = []
        actions = [[0.1, 0.3], [-0.1, 0.4], [0.0, 0.2]]
        for i in range(steps):
            _, _, terminated, truncated, _ = env.step(actions[i % len(actions)])
            v = env.vehicle
            pos = np.asarray(v.position, dtype=np.float64)
            vel = np.asarray(v.velocity, dtype=np.float64)
            trace.append([round(float(x), 4) for x in (pos[0], pos[1], float(v.heading_theta), *vel[:2])])
            if terminated or truncated:
                env.reset(seed=seed + i + 1)
        payload = np.asarray(trace, dtype=np.float64).tobytes()
        return hashlib.sha256(payload).hexdigest()[:16]
    finally:
        env.close()


def _worker(args):
    seed, map_cfg, steps = args
    return _fingerprint(seed, map_cfg, steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--map", default="3")
    args = parser.parse_args()
    map_cfg = int(args.map) if args.map.isdigit() else args.map

    print("=== same-process repeats ===")
    fps = [_fingerprint(args.seed, map_cfg, args.steps) for _ in range(2)]
    print("repeat fingerprints:", fps, "MATCH" if fps[0] == fps[1] else "MISMATCH")

    print("=== cross-process (spawn) ===")
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        results = pool.map(_worker, [(args.seed, map_cfg, args.steps)] * args.workers)
    print("cross-process fingerprints:", results, "MATCH" if len(set(results)) == 1 else "MISMATCH")


if __name__ == "__main__":
    main()
