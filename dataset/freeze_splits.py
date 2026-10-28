#!/usr/bin/env python3
"""Freeze the canonical scene specs and seed manifests.

Canonical scene sets:
  train_2000 : first 2000 scenes from source/train_10000.json (1000..2999)
  val_500    : first 500 scenes from source/val_1000.json (5000000..5000499)
  test_500   : last 500 scenes from source/val_1000.json (5000500..5000999)
  mini_*     : deterministic smoke-test prefixes of train/val

Raw spec JSON stays gitignored under ``data/specs``; the seed manifests under
``data/splits`` are tracked and verified by the evaluator.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from dataset.splits import file_sha256
from env.scenario.spec import load_specs, save_specs

ROOT = Path(__file__).resolve().parents[1]
SPEC_ROOT = ROOT / "data/specs"
SPLIT_ROOT = ROOT / "data/splits"


def _write_manifest(
    *,
    name: str,
    split: str,
    specs,
    spec_path: Path,
    source: Path,
    schema: str = "pnp_data_split_v1",
) -> Path:
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": schema,
        "name": name,
        "split": split,
        "source": str(source.relative_to(ROOT)),
        "spec_path": str(spec_path.relative_to(ROOT)),
        "n": len(specs),
        "seed_range": [int(specs[0].seed), int(specs[-1].seed)],
        "seeds": [int(s.seed) for s in specs],
        "sha256": file_sha256(spec_path),
    }
    out = SPLIT_ROOT / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def _save(name: str, specs, *, split: str | None = None):
    if split is not None:
        specs = [dataclasses.replace(s, split=split) for s in specs]
    path = SPEC_ROOT / f"{name}.json"
    save_specs(specs, path)
    return path, specs


def main() -> int:
    train_source = SPEC_ROOT / "source/train_10000.json"
    val_source = SPEC_ROOT / "source/val_1000.json"
    train = load_specs(train_source)
    val = load_specs(val_source)
    if len(train) < 2000:
        raise SystemExit(f"need >=2000 train specs, got {len(train)}")
    if len(val) < 1000:
        raise SystemExit(f"need >=1000 val specs, got {len(val)}")

    train2000 = train[:2000]
    val500 = val[:500]
    test500 = val[500:1000]
    mini_train = train2000[:100]
    mini_val = val500[:20]

    outputs = []
    for name, specs, split, schema in (
        ("train_2000", train2000, "train", "pnp_data_split_v1"),
        ("val_500", val500, "val", "pnp_eval_split_v2"),
        ("test_500", test500, "test", "pnp_eval_split_v2"),
        ("mini_train_100", mini_train, "train", "pnp_data_split_v1"),
        ("mini_val_20", mini_val, "val", "pnp_data_split_v1"),
    ):
        path, saved = _save(name, specs, split=split)
        source = train_source if name.startswith("train") or name.startswith("mini_train") else val_source
        manifest = _write_manifest(
            name=name, split=split, specs=saved, spec_path=path, source=source, schema=schema
        )
        outputs.append((path, manifest))

    for path, manifest in outputs:
        print(f"wrote {path.relative_to(ROOT)} -> {manifest.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
