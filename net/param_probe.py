#!/usr/bin/env python3
"""Deployable parameter ledger and shape smoke."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from net.model import StructuredPlanner, count_parameters

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", default=None)
    args = p.parse_args()
    m = StructuredPlanner()
    total = count_parameters(m)
    per_module = {name: sum(p.numel() for p in mod.parameters()) for name, mod in m.named_children()}
    report = {"deployable_total": total, "per_module": per_module,
              "budget_min": 800_000, "budget_max": 1_200_000,
              "within_budget": 800_000 <= total <= 1_200_000}
    print(json.dumps(report, indent=2))
    if args.json_out:
        open(args.json_out, "w").write(json.dumps(report, indent=2))
    assert report["within_budget"], report
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
