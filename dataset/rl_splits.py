"""RL scene-split manifests for real-env group-relative PPO.

The RL split is scene-level and intentionally separate from ``val_500`` and
``test_500``.  Manifests reference the full ``train_2000`` spec file for SHA
verification, but their ``seeds`` define the actual subset.  A trainer must use
:func:`load_rl_spec_splits` (never load the full spec file and train on it).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RLSpecSplit:
    train: List[Any]
    holdout: List[Any]
    calibration: List[Any]
    difficulty_weights: Dict[str, float]
    train_manifest: Mapping[str, Any]
    holdout_manifest: Mapping[str, Any]
    calibration_manifest: Mapping[str, Any]


def load_manifest(path: str | Path, *, expected_role: str | None = None) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    role = str(payload.get("role", ""))
    if expected_role is not None and role != expected_role:
        raise ValueError(f"{path}: role={role!r} != expected {expected_role!r}")
    seeds = [int(x) for x in payload.get("seeds", [])]
    if int(payload.get("n", -1)) != len(seeds):
        raise ValueError(f"{path}: n != len(seeds)")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{path}: duplicate seeds")
    if seeds != sorted(seeds):
        raise ValueError(f"{path}: seeds must be sorted")
    return payload


def filter_specs_by_manifest(specs: Sequence[Any], manifest: Mapping[str, Any]) -> List[Any]:
    by_seed = {int(getattr(s, "seed")): s for s in specs}
    seeds = [int(x) for x in manifest["seeds"]]
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        raise ValueError(f"manifest seeds missing from specs: {missing[:5]}")
    return [by_seed[s] for s in seeds]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rl_spec_splits(
    specs: Sequence[Any],
    *,
    train_manifest: str | Path,
    holdout_manifest: str | Path,
    calibration_manifest: str | Path,
    specs_path: str | Path | None = None,
) -> RLSpecSplit:
    """Load and validate the canonical RL split manifests.

    ``specs`` must be the full ``train_2000`` spec sequence.  The train and
    holdout seed sets must be disjoint and their union must cover all supplied
    scene seeds.  Calibration is a subset of train.
    """
    train_m = load_manifest(train_manifest, expected_role="rl_train")
    holdout_m = load_manifest(holdout_manifest, expected_role="rl_holdout")
    cal_m = load_manifest(calibration_manifest, expected_role="rl_calibration")
    for m in (train_m, holdout_m, cal_m):
        spec_path = str(m.get("spec_path", ""))
        if spec_path and _sha256(spec_path) != str(m.get("sha256", "")):
            raise ValueError(f"{m.get('name')}: spec sha256 mismatch")
        if specs_path is not None:
            if Path(spec_path).resolve() != Path(specs_path).resolve():
                raise ValueError(
                    f"{m.get('name')}: manifest spec_path {spec_path!r} "
                    f"does not match supplied specs {str(specs_path)!r}"
                )
    train_seeds = [int(x) for x in train_m["seeds"]]
    holdout_seeds = [int(x) for x in holdout_m["seeds"]]
    cal_seeds = [int(x) for x in cal_m["seeds"]]
    if set(train_seeds) & set(holdout_seeds):
        raise ValueError("RL train/holdout overlap")
    if set(cal_seeds) - set(train_seeds):
        raise ValueError("RL calibration must be a subset of train")
    all_spec_seeds = {int(getattr(s, "seed")) for s in specs}
    if set(train_seeds) | set(holdout_seeds) != all_spec_seeds:
        raise ValueError("RL train+holdout must cover the full train_2000 spec set")
    weights = {str(k): float(v) for k, v in train_m.get("difficulty_weights", {}).items()}
    for key, value in weights.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"difficulty weight {key!r} must be finite and > 0")
    return RLSpecSplit(
        train=filter_specs_by_manifest(specs, train_m),
        holdout=filter_specs_by_manifest(specs, holdout_m),
        calibration=filter_specs_by_manifest(specs, cal_m),
        difficulty_weights=weights,
        train_manifest=train_m,
        holdout_manifest=holdout_m,
        calibration_manifest=cal_m,
    )
