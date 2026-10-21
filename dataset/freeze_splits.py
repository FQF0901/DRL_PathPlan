#!/usr/bin/env python3
"""Generate the frozen Val500/Test500 specs and machine-readable manifests.

Usage::

    tools/venv-python -m dataset.freeze_splits

The source is ``env/specs/scenarios_val.json`` (1000 scenes, disjoint from
train seeds).  The first 500 seeds are validation, the last 500 are test.
Raw spec files stay gitignored; ``config/splits/*.json`` are tracked and
verified by the evaluator before a gate run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from dataset.splits import file_sha256
from env.scenario.spec import load_specs, save_specs

ROOT = Path(__file__).resolve().parents[1]


def _write_manifest(
    *,
    split: str,
    specs,
    spec_path: Path,
    source: Path,
) -> Path:
    out = ROOT / "config/splits" / f"{split}500.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "pnp_eval_split_v2",
        "split": split,
        "source": str(source.relative_to(ROOT)),
        "spec_path": str(spec_path.relative_to(ROOT)),
        "n": len(specs),
        "seed_range": [int(specs[0].seed), int(specs[-1].seed)],
        "seeds": [int(s.seed) for s in specs],
        "sha256": file_sha256(spec_path),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    source = ROOT / "env/specs/scenarios_val.json"
    specs = load_specs(source)
    if len(specs) < 1000:
        raise SystemExit(f"need at least 1000 val specs, got {len(specs)}")
    val = specs[:500]
    test = [dataclasses.replace(s, split="test") for s in specs[500:1000]]
    val_path = ROOT / "env/specs/scenarios_val500.json"
    test_path = ROOT / "env/specs/scenarios_test500.json"
    save_specs(val, val_path)
    save_specs(test, test_path)
    val_manifest = _write_manifest(
        split="val", specs=val, spec_path=val_path, source=source
    )
    test_manifest = _write_manifest(
        split="test", specs=test, spec_path=test_path, source=source
    )
    print(f"wrote {val_path} ({len(val)}) -> {val_manifest}")
    print(f"wrote {test_path} ({len(test)}) -> {test_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
