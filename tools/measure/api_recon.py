#!/usr/bin/env python3
"""P1a API reconnaissance: dump the MetaDrive APIs needed to build the custom observation
channels (OD / LD / nav / speed limit / ego) and the scenario post-processing (limits, line types).

Defensive: prints what exists, never assumes. Read-only w.r.t. the project.

Usage:
    tools/venv-python tools/measure/api_recon.py --map SCX --steps 20
"""
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="SCX")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--traffic-density", type=float, default=0.3)
    args = parser.parse_args()
    map_cfg = int(args.map) if args.map.isdigit() else args.map

    from metadrive.envs import MetaDriveEnv

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
    try:
        env.reset()
        info = {}
        for _ in range(args.steps):
            _, _, terminated, truncated, info = env.step([0.0, 0.3])
            if terminated or truncated:
                break

        print("=== info keys ===")
        print(sorted(info.keys()))

        v = env.vehicle
        print("=== ego ===")
        for name in ("position", "heading_theta", "velocity", "speed_km_h", "on_lane", "lane", "lane_index"):
            obj = getattr(v, name, None)
            shape = getattr(obj, "shape", "")
            print(f"  {name}: {type(obj).__name__} {shape}")

        nav = getattr(v, "navigation", None)
        print("=== nav ===")
        if nav is not None:
            cps = nav.get_checkpoints()
            print("  checkpoints type/len:", type(cps).__name__, len(cps) if hasattr(cps, "__len__") else "?")
            if hasattr(cps, "__len__") and len(cps) > 0:
                print("  first checkpoint:", cps[0])
            print("  route_completion:", nav.route_completion)
            cur = nav.current_ref_lanes
            print("  current_ref_lanes:", type(cur).__name__, len(cur) if cur else 0)
            nxt = nav.next_ref_lanes
            print("  next_ref_lanes:", type(nxt).__name__, len(nxt) if nxt else 0)
            lane = cur[0] if cur else None
            if lane is not None:
                print("  lane public attrs:", [a for a in dir(lane) if not a.startswith("_")][:45])
                for a in ("speed_limit", "length", "width", "line_types", "line_colors", "index", "heading_theta"):
                    print(f"    {a}:", getattr(lane, a, "N/A"))
                try:
                    print("    local_coordinates(ego):", lane.local_coordinates(v.position))
                except Exception as exc:  # noqa: BLE001
                    print("    local_coordinates failed:", exc)
                try:
                    print("    distance(ego):", lane.distance(v.position))
                except Exception as exc:  # noqa: BLE001
                    print("    distance failed:", exc)
        else:
            print("  no navigation on ego")

        print("=== objects ===")
        try:
            objs = env.engine.get_objects()
            print("  n_objects:", len(objs))
            for k, o in list(objs.items())[:10]:
                print(f"  - {k}: {type(o).__name__} class_name={getattr(o, 'class_name', None)}")
            first = next(iter(objs.values()), None)
            if first is not None:
                print("  sample object attrs:", [a for a in dir(first) if not a.startswith("_")][:45])
        except Exception as exc:  # noqa: BLE001
            print("  get_objects failed:", exc)

        print("=== map / road network ===")
        m = env.current_map
        print("  map type:", type(m).__name__)
        rn = m.road_network
        print("  road_network type:", type(rn).__name__)
        try:
            lanes = rn.get_all_lanes() if hasattr(rn, "get_all_lanes") else None
            print("  get_all_lanes:", (type(lanes).__name__, len(lanes)) if lanes is not None else "N/A")
        except Exception as exc:  # noqa: BLE001
            print("  get_all_lanes failed:", exc)
        print("  graph nodes:", len(getattr(rn, "graph", {})) if hasattr(rn, "graph") else "N/A")
    finally:
        env.close()


if __name__ == "__main__":
    main()
