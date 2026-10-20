#!/usr/bin/env python3
"""Create the frozen Val500/Test500 specs and a deterministic manifest.

Usage::

    tools/venv-python tools/freeze_splits.py

The source is ``env/specs/scenarios_val.json`` (1000 scenes, disjoint from
train seeds).  The first 500 seeds are validation, the last 500 are test.
The spec files are gitignored; ``docs/eval_split_manifest.json`` records the
seed ranges and SHA-256 so a rerun can be checked.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.scenario.spec import load_specs, save_specs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = ROOT
    source = root / "env/specs/scenarios_val.json"
    specs = load_specs(source)
    if len(specs) < 1000:
        raise SystemExit(f"need at least 1000 val specs, got {len(specs)}")
    val, test = specs[:500], specs[500:1000]
    val_path = root / "env/specs/scenarios_val500.json"
    test_path = root / "env/specs/scenarios_test500.json"
    save_specs(val, val_path)
    save_specs(test, test_path)
    manifest = {
        "schema": "pnp_eval_split_v1",
        "source": str(source.relative_to(root)),
        "val": {
            "path": str(val_path.relative_to(root)),
            "n": len(val),
            "seeds": [s.seed for s in val],
            "seed_range": [val[0].seed, val[-1].seed],
            "sha256": _sha256(val_path),
        },
        "test": {
            "path": str(test_path.relative_to(root)),
            "n": len(test),
            "seeds": [s.seed for s in test],
            "seed_range": [test[0].seed, test[-1].seed],
            "sha256": _sha256(test_path),
        },
        "split_rule": "first 500 -> val, last 500 -> test; seeds disjoint; test is read-only",
    }
    out = root / "docs/eval_split_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {val_path} ({len(val)}) and {test_path} ({len(test)})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
