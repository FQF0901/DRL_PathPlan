#!/usr/bin/env python3
"""P0: PG map build time — random generation (block_num) vs explicit BLOCK_SEQUENCE.

MetaDrive: `map=<int>` -> BIG_BLOCK_NUM (random search + backtracking);
           `map="SCXRO"` -> BIG_BLOCK_SEQUENCE (explicit, should skip the search).

Usage:
    .venv/bin/python tools/measure/measure_mapgen.py --seeds 20
"""
import argparse
import statistics
import time


def time_build(map_cfg, seeds):
    """Return list of build seconds, or ('ERR', msg) on failure."""
    from metadrive.envs import MetaDriveEnv

    times = []
    for seed in seeds:
        t0 = time.time()
        try:
            env = MetaDriveEnv(
                dict(
                    use_render=False,
                    log_level=50,
                    num_scenarios=1,
                    start_seed=int(seed),
                    map=map_cfg,
                    traffic_density=0.0,
                    preload_models=False,
                    store_map=False,
                )
            )
            env.reset()
        except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
            return ("ERR", f"{type(exc).__name__}: {exc}")
        finally:
            try:
                env.close()  # type: ignore[possibly-undefined]
            except Exception:  # noqa: BLE001
                pass
        times.append(time.time() - t0)
    return times


def summarize(label, result):
    if isinstance(result, tuple):
        print(f"{label:<28} FAILED: {result[1][:120]}")
        return
    print(
        f"{label:<28} n={len(result):<3} mean={statistics.mean(result):6.2f}s "
        f"median={statistics.median(result):6.2f}s max={max(result):6.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20, help="number of seeds per config")
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--seq", nargs="*", default=["S", "SC", "SCXRO"], help="explicit block sequences to try")
    args = parser.parse_args()

    seeds = [args.start_seed + i for i in range(args.seeds)]

    print("=== random generation (block_num) ===")
    for n_blocks in (3, 5, 8):
        summarize(f"random block_num={n_blocks}", time_build(n_blocks, seeds))

    print("=== explicit BLOCK_SEQUENCE (skips random search) ===")
    for seq in args.seq:
        summarize(f"sequence={seq}", time_build(seq, seeds))


if __name__ == "__main__":
    main()
