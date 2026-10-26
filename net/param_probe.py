#!/usr/bin/env python3
"""Parameter ledger for the bank-native network built from config/model.yaml."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow both ``python -m net.param_probe`` and ``python net/param_probe.py``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net.build_net import build_net, load_config


def _budget(cfg: dict) -> tuple[int, int]:
    # Budget keys live at the top level in config/model.yaml; support the
    # legacy nested-model form as a fallback.
    model = cfg.get("model", cfg)
    if not isinstance(model, dict):
        model = cfg
    return (
        int(cfg.get("deployable_param_min", model.get("deployable_param_min", 800_000))),
        int(cfg.get("deployable_param_max", model.get("deployable_param_max", 1_200_000))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/model.yaml")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = build_net(cfg)
    total = sum(p.numel() for p in model.parameters())
    per_module = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.named_children()
    }
    per_submodule = {
        f"{top}.{name}": sum(p.numel() for p in child.parameters())
        for top, module in model.named_children()
        for name, child in module.named_children()
    }
    lo, hi = _budget(cfg)
    report = {
        "config": args.config,
        "deployable_total": total,
        "per_module": per_module,
        "per_submodule": per_submodule,
        "module_share": {
            name: (float(value) / float(total) if total else 0.0)
            for name, value in per_module.items()
        },
        "budget_min": lo,
        "budget_max": hi,
        "param_target": int(cfg.get("deployable_param_target", 0) or 0),
        "target_delta": (total - int(cfg.get("deployable_param_target", 0) or 0)),
        "within_budget": lo <= total <= hi,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    return 0 if report["within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
