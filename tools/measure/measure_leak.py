#!/usr/bin/env python3
"""P0: RSS growth per reset (leak curve). Two modes:
  store_map=False -> rebuild map every reset (the mode we plan to use with a bounded LRU cache)
  store_map=True  -> map retained (shows per-map RAM cost; NOT usable for 10k maps)

Usage:
    .venv/bin/python tools/measure/measure_leak.py --resets 1000 --mode False
"""
import argparse
import os


def run(mode: bool, resets: int, sample_every: int, start_seed: int) -> None:
    import psutil
    from metadrive.envs import MetaDriveEnv

    proc = psutil.Process(os.getpid())
    env = MetaDriveEnv(
        dict(
            use_render=False,
            log_level=50,
            num_scenarios=max(resets, 2),
            start_seed=start_seed,
            map=3,
            traffic_density=0.1,
            preload_models=False,
            store_map=mode,
        )
    )
    rss = []
    try:
        env.reset()
        base = proc.memory_info().rss / 1e6
        for i in range(resets):
            env.reset()
            if (i + 1) % sample_every == 0:
                rss.append(proc.memory_info().rss / 1e6)
    finally:
        env.close()

    if len(rss) >= 2:
        growth = rss[-1] - rss[0]
        per_1k = growth / (resets / 1000.0)
        print(f"store_map={mode} resets={resets} base={base:.0f}MB samples={['%.0f' % x for x in rss]}")
        print(f"  growth={growth:+.0f}MB over {resets} resets -> {per_1k:+.0f}MB/1000 resets")
    else:
        print(f"store_map={mode}: not enough samples ({len(rss)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resets", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=30000)
    parser.add_argument("--mode", default="False", help="False (rebuild) or True (keep maps)")
    args = parser.parse_args()
    run(args.mode.lower() == "true", args.resets, args.sample_every, args.start_seed)


if __name__ == "__main__":
    main()
