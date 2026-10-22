"""Closed-loop checkpoint selection (no open-loop best.pt promotion)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from env.metadrive_env import build_env
from env.scenario.spec import load_specs
from pipeline.eval import load_model, run_episode
from pipeline.eval_metrics import summarize_rows
from dataset.splits import verify_split_manifest


def _eligible(row: Dict[str, Any], collision_max: float, offroad_max: float) -> bool:
    return (
        float(row.get("collision_rate", 1.0)) <= float(collision_max)
        and float(row.get("off_road_rate", 1.0)) <= float(offroad_max)
    )


def _score(row: Dict[str, Any]) -> tuple:
    # Eligible candidates: maximize success, then completion; safety is a hard
    # threshold above, not a continuous lexicographic tie-breaker.
    return (float(row.get("success_rate", 0.0)), float(row.get("route_completion_mean", 0.0)))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Closed-loop checkpoint selector")
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--specs", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="cuda")
    p.add_argument("--collision-max", type=float, default=0.20)
    p.add_argument("--offroad-max", type=float, default=0.20)
    args = p.parse_args(argv)
    if args.split == "test":
        parser.error("Test500 is read-only; checkpoint selection must use val")

    all_specs = load_specs(args.specs)
    verify_split_manifest(args.split_manifest, all_specs, split=args.split, spec_path=args.specs)
    specs = all_specs[: int(args.limit)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for ckpt in args.candidates:
        model = load_model(ckpt, args.device, strict=True)
        rows = []
        for spec in specs:
            env = build_env(spec, use_render=False, lru_size=0)
            try:
                row = run_episode(env, model, spec, controller="exact", max_steps=int(args.max_steps), device=args.device)
            finally:
                env.close()
            rows.append(row)
        summary = summarize_rows(rows)
        record = {"checkpoint": str(ckpt), **summary}
        record["eligible"] = _eligible(summary, args.collision_max, args.offroad_max)
        record["score"] = list(_score(summary))
        records.append(record)
        print(f"[select] {ckpt} eligible={record['eligible']} score={record['score']}", flush=True)

    eligible = [r for r in records if r["eligible"]]
    if not eligible:
        payload = {
            "selected": None,
            "candidates": records,
            "policy": "no candidate met hard collision/off-road thresholds",
            "split_manifest": args.split_manifest,
            "limit": int(args.limit),
            "controller": "exact",
        }
        (out / "selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("[select] no eligible candidate; selected=null")
        return 2
    best = max(eligible, key=_score)
    shutil.copyfile(best["checkpoint"], out / "selected.pt")
    payload = {
        "selected": best,
        "candidates": records,
        "policy": "hard collision/off-road thresholds -> success -> RC",
        "split_manifest": args.split_manifest,
        "limit": int(args.limit),
        "controller": "exact",
    }
    (out / "selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"selected": best["checkpoint"], "score": best["score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
