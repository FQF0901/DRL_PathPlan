#!/usr/bin/env python3
"""Freeze the canonical scene specs and the non-derivable split views.

Canonical scene sets:
  train_2000 : first 2000 scenes from source/train_10000.json (1000..2999)
  val_500    : first 500 scenes from source/val_1000.json (5000000..5000499)
  test_500   : last 500 scenes from source/val_1000.json (5000500..5000999)
  mini_*     : deterministic smoke-test prefixes of train/val

The spec's own ``split`` field is the source of truth for train/val/test/mini.
Only the gate seals (val/test) and the two RL subset views are written under
``data/splits``.  ``--check`` verifies those files without writing anything.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from pipeline.dataio.splits import file_sha256
from env.scenario.spec import load_specs, save_specs

ROOT = Path(__file__).resolve().parents[2]
SPEC_ROOT = ROOT / "data/specs"
SPLIT_ROOT = ROOT / "data/splits"

RL_TRAIN_COUNT = 1500
RL_HOLDOUT_COUNT = 500
RL_DIFFICULTY_WEIGHTS = {"easy": 1.0, "medium": 1.5, "hard": 2.0}


def _with_split(specs: Sequence[Any], split: str) -> List[Any]:
    return [dataclasses.replace(spec, split=split) for spec in specs]


def build_specs() -> Dict[str, List[Any]]:
    """Return the five canonical spec sets, without writing them."""
    train = load_specs(SPEC_ROOT / "source/train_10000.json")
    val = load_specs(SPEC_ROOT / "source/val_1000.json")
    if len(train) < 2000:
        raise SystemExit(f"need >=2000 train specs, got {len(train)}")
    if len(val) < 1000:
        raise SystemExit(f"need >=1000 val specs, got {len(val)}")
    train2000 = train[:2000]
    val500 = val[:500]
    test500 = val[500:1000]
    return {
        "train_2000": _with_split(train2000, "train"),
        "val_500": _with_split(val500, "val"),
        "test_500": _with_split(test500, "test"),
        "mini_train_100": _with_split(train2000[:100], "train"),
        "mini_val_20": _with_split(val500[:20], "val"),
    }


def write_specs(specs_by_name: Mapping[str, Sequence[Any]]) -> None:
    for name, specs in specs_by_name.items():
        path = SPEC_ROOT / f"{name}.json"
        save_specs(specs, path)
        print(f"wrote {path.relative_to(ROOT)}")


def _manifest_payload(
    *,
    name: str,
    split: str,
    specs: Sequence[Any],
    spec_path: Path,
    source: Path,
    schema: str,
    role: str | None = None,
    difficulty_weights: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema": schema,
        "name": name,
        "split": split,
    }
    if role is not None:
        payload["role"] = role
    payload.update(
        {
            "source": str(source.relative_to(ROOT)),
            "spec_path": str(spec_path.relative_to(ROOT)),
            "n": len(specs),
            "seed_range": [int(specs[0].seed), int(specs[-1].seed)],
            "seeds": [int(spec.seed) for spec in specs],
            "sha256": file_sha256(spec_path),
        }
    )
    if difficulty_weights is not None:
        payload["difficulty_weights"] = dict(difficulty_weights)
    return payload


def build_manifests(specs_by_name: Mapping[str, Sequence[Any]]) -> Dict[str, Dict[str, Any]]:
    """Return the gate seals + RL views; does not write files."""
    train_source = SPEC_ROOT / "source/train_10000.json"
    val_source = SPEC_ROOT / "source/val_1000.json"
    train_spec_path = SPEC_ROOT / "train_2000.json"
    train_specs = specs_by_name["train_2000"]
    manifests: Dict[str, Dict[str, Any]] = {}
    for name, source, schema in (
        ("val_500", val_source, "pnp_eval_split_v2"),
        ("test_500", val_source, "pnp_eval_split_v2"),
    ):
        manifests[name] = _manifest_payload(
            name=name,
            split=name.split("_")[0],
            specs=specs_by_name[name],
            spec_path=SPEC_ROOT / f"{name}.json",
            source=source,
            schema=schema,
        )
    manifests["rl_train_1500"] = _manifest_payload(
        name="rl_train_1500",
        split="train",
        specs=train_specs[:RL_TRAIN_COUNT],
        spec_path=train_spec_path,
        source=train_spec_path,
        schema="pnp_data_split_v1",
        role="rl_train",
        difficulty_weights=RL_DIFFICULTY_WEIGHTS,
    )
    manifests["rl_holdout_500"] = _manifest_payload(
        name="rl_holdout_500",
        split="train",
        specs=train_specs[RL_TRAIN_COUNT : RL_TRAIN_COUNT + RL_HOLDOUT_COUNT],
        spec_path=train_spec_path,
        source=train_spec_path,
        schema="pnp_data_split_v1",
        role="rl_holdout",
    )
    return manifests


def write_manifests(manifests: Mapping[str, Mapping[str, Any]]) -> None:
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, payload in manifests.items():
        path = SPLIT_ROOT / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


def check_manifests(manifests: Mapping[str, Mapping[str, Any]]) -> int:
    failures = 0
    for name, expected in manifests.items():
        path = SPLIT_ROOT / f"{name}.json"
        if not path.is_file():
            print(f"MISSING {path.relative_to(ROOT)}")
            failures += 1
            continue
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            print(f"STALE {path.relative_to(ROOT)}")
            failures += 1
    # A stale hand-written manifest is just as bad as a missing one.
    expected_names = set(manifests)
    actual_names = {p.stem for p in SPLIT_ROOT.glob("*.json")}
    for name in sorted(actual_names - expected_names):
        print(f"UNEXPECTED {(SPLIT_ROOT / f'{name}.json').relative_to(ROOT)}")
        failures += 1
    return 0 if failures == 0 else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze specs and split views")
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args(argv)

    specs = build_specs()
    if args.check:
        return check_manifests(build_manifests(specs))
    # Write specs first so the manifest SHA reflects the bytes on disk.
    write_specs(specs)
    write_manifests(build_manifests(specs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
