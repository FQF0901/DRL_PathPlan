#!/usr/bin/env python3
"""Deployable parameter ledger for the canonical planner."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net.model import StructuredPlanner, count_parameters

BUDGET_MIN = 800_000
BUDGET_MAX = 1_200_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    model = StructuredPlanner()
    total = count_parameters(model)
    per_module = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.named_children()
    }
    # Top-level parameters (learned query tokens) are not named children.
    for name, param in model.named_parameters():
        if "." not in name:
            per_module[name] = param.numel()

    report = {
        "deployable_total": total,
        "per_module": per_module,
        "budget_min": BUDGET_MIN,
        "budget_max": BUDGET_MAX,
        "within_budget": BUDGET_MIN <= total <= BUDGET_MAX,
    }
    print(json.dumps(report, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert report["within_budget"], report
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
