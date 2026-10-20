#!/usr/bin/env python3
"""Deployable parameter ledger and shape smoke."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from net.model import StructuredPlanner, count_parameters
from net.clean_model import CleanPlanner

BUDGET_MIN = 800_000
BUDGET_MAX = 1_200_000


def _entry(model) -> dict:
    total = count_parameters(model)
    per_module = {name: sum(p.numel() for p in mod.parameters()) for name, mod in model.named_children()}
    # top-level parameters (e.g. learned queries) are not named children
    for name, param in model.named_parameters():
        if "." not in name:
            per_module[name] = param.numel()
    return {
        "deployable_total": total,
        "per_module": per_module,
        "within_budget": BUDGET_MIN <= total <= BUDGET_MAX,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    legacy = _entry(StructuredPlanner())
    clean = _entry(CleanPlanner())
    report = {
        # legacy top-level fields kept for backwards compatibility
        "deployable_total": legacy["deployable_total"],
        "per_module": legacy["per_module"],
        "budget_min": BUDGET_MIN,
        "budget_max": BUDGET_MAX,
        "within_budget": legacy["within_budget"],
        # both models, explicit
        "legacy": legacy,
        "clean": clean,
        "both_within_budget": legacy["within_budget"] and clean["within_budget"],
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        open(args.json_out, "w").write(json.dumps(report, indent=2))
    assert report["within_budget"], report
    assert clean["within_budget"], report
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
