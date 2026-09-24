#!/usr/bin/env python3
"""P0 smoke test: headless MetaDrive env, measure reset time / FPS / RSS.

Usage:
    .venv/bin/python tools/measure/smoke_env.py --seed 1000 --steps 200
"""
import argparse
import os
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument("--map", default="3", help="int-like string = block num; else explicit sequence e.g. SCX")
    args = parser.parse_args()

    import psutil
    import numpy
    import gymnasium
    import torch

    print(
        f"numpy={numpy.__version__} gymnasium={gymnasium.__version__} "
        f"torch={torch.__version__} cuda={torch.cuda.is_available()}"
    )
    from metadrive.envs import MetaDriveEnv

    map_cfg = int(args.map) if args.map.isdigit() else args.map
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss / 1e6

    t0 = time.time()
    env = MetaDriveEnv(
        dict(
            use_render=False,
            log_level=50,
            num_scenarios=1,
            start_seed=args.seed,
            map=map_cfg,
            traffic_density=args.traffic_density,
            preload_models=False,
            store_map=False,
        )
    )
    env.reset()
    reset_s = time.time() - t0
    rss_after_reset = proc.memory_info().rss / 1e6

    n = 0
    t = time.time()
    for _ in range(args.steps):
        _, _, terminated, truncated, _ = env.step([0.1, 0.3])
        n += 1
        if terminated or truncated:
            env.reset()
    dt = time.time() - t
    rss_end = proc.memory_info().rss / 1e6
    env.close()

    print(
        f"reset_s={reset_s:.2f} fps={n / dt:.1f} "
        f"rss_MB start={rss_start:.0f} after_reset={rss_after_reset:.0f} end={rss_end:.0f}"
    )


if __name__ == "__main__":
    main()
