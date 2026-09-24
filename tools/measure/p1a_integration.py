#!/usr/bin/env python3
"""P1a integration check (orchestrator-owned): spec -> build_env -> obs -> per-step labels -> short rollout.

Verifies the cross-lane contract end-to-end and reports:
  - env reset OK + speed limits actually applied (lane.speed_limit < 1000)
  - obs channel shapes / dtypes / mask counts (flattened, incl. history stack)
  - per-step observable labels (keys + values)
  - short rollout: crash / out-of-road counts, route_completion start
  - step-time breakdown (obs build vs env.step)

Usage:
    tools/venv-python tools/measure/p1a_integration.py --spec env/specs/scenarios_train_slice200.json --limit 5
"""
import argparse
import json
import os
import sys
import time

# Allow running this file by path (sys.path[0] would otherwise be tools/measure).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _flatten(prefix, obj, shapes, masks):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, shapes, masks)
        return
    if hasattr(obj, "shape"):
        shapes[prefix] = [list(obj.shape), str(obj.dtype)]
        if "mask" in prefix or "valid" in prefix:
            try:
                import numpy as np

                masks[prefix] = int(np.asarray(obj).sum())
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="env/specs/scenarios_train_slice200.json")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--config", default="config/env.yaml")
    parser.add_argument("--out", default="runs/p1a_integration.json")
    args = parser.parse_args()

    import numpy as np
    import yaml

    from env.scenario.spec import load_specs
    from env.scenario import labels as label_mod
    from env.metadrive_env import build_env
    from env.obs.builder import ObservationBuilder

    cfg = yaml.safe_load(open(args.config))
    builder = ObservationBuilder(cfg.get("obs", {}))

    specs = load_specs(args.spec)[: args.limit]
    rows = []
    for spec in specs:
        env = build_env(spec)
        try:
            env.reset()
            lanes = env.current_map.road_network.get_all_lanes()
            limits = sorted({round(float(getattr(lane, "speed_limit", 1000.0)), 1) for lane in lanes})
            shapes, masks = {}, {}
            _flatten("", builder.build(env, spec), shapes, masks)
            labels = label_mod.compute_step_labels(env, spec)

            t_obs = t_step = 0.0
            rc0 = None
            crashes = oor = 0
            steps_done = 0
            for _ in range(args.steps):
                t0 = time.perf_counter()
                builder.build(env, spec)
                t_obs += time.perf_counter() - t0
                t0 = time.perf_counter()
                _, _, terminated, truncated, info = env.step([0.0, 0.3])
                t_step += time.perf_counter() - t0
                steps_done += 1
                if rc0 is None:
                    rc0 = info.get("route_completion")
                crashes += int(bool(info.get("crash")))
                oor += int(bool(info.get("out_of_road")))
                if terminated or truncated:
                    break

            rows.append(
                {
                    "id": spec.id,
                    "seed": spec.seed,
                    "geometry": spec.geometry,
                    "speed_limits_applied": limits[:5],
                    "speed_limit_ok": bool(limits and limits[0] < 1000),
                    "obs_shapes": shapes,
                    "mask_counts": masks,
                    "labels": {k: round(float(v), 3) for k, v in labels.items()},
                    "rc_start": rc0,
                    "crashes": crashes,
                    "out_of_road": oor,
                    "steps": steps_done,
                    "obs_ms": round(1000 * t_obs / max(steps_done, 1), 2),
                    "env_step_ms": round(1000 * t_step / max(steps_done, 1), 2),
                }
            )
            print(
                f"spec {spec.id} seed={spec.seed} geom={spec.geometry} "
                f"limit_ok={rows[-1]['speed_limit_ok']} limits={limits[:3]} "
                f"obs_ms={rows[-1]['obs_ms']} step_ms={rows[-1]['env_step_ms']} "
                f"crash={crashes} oor={oor} labels={sorted(labels)[:4]}..."
            )
        finally:
            env.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}  ({len(rows)} specs)")


if __name__ == "__main__":
    main()
