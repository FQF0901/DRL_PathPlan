"""Data layout invariants.

These tests keep the on-disk layout from drifting back into per-role
subdirectories or filename/manifest-name mismatches.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/splits"


def _manifests() -> list[Path]:
    return sorted(SPLITS.glob("*.json"))


def test_splits_are_flat_and_named_by_basename():
    assert not any(p.is_dir() for p in SPLITS.iterdir())
    for path in _manifests():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["name"] == path.stem, f"{path}: name != basename"
        assert payload["n"] == len(payload["seeds"])
        assert payload["seeds"] == sorted(payload["seeds"])
        assert len(payload["seeds"]) == len(set(payload["seeds"]))


def test_manifest_spec_path_and_sha_match():
    for path in _manifests():
        payload = json.loads(path.read_text(encoding="utf-8"))
        spec_path = ROOT / payload["spec_path"]
        assert spec_path.is_file(), f"{path}: missing {spec_path}"
        digest = hashlib.sha256(spec_path.read_bytes()).hexdigest()
        assert payload["sha256"] == digest, f"{path}: stale spec sha256"
        assert payload["seed_range"] == [payload["seeds"][0], payload["seeds"][-1]]


def test_only_gate_seals_and_rl_views_live_in_splits():
    names = {path.stem for path in _manifests()}
    assert names == {"val_500", "test_500", "rl_train_1500", "rl_holdout_500"}
