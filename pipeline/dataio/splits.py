"""Scene-level split helpers.

The contract forbids train/validation/test leakage by map/scene seed
(``docs/arch_lock.md`` §14.1).  Episode ids are *not* sufficient: the same
scene can be re-collected as several episodes (BC + DAgger), so splitting by
episode can put two copies of one scene on both sides.

This module is deliberately tiny and dependency-free so both the dataset
loader and tests use the same rule.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def split_by_scene_seed(
    scene_seed: Sequence[int] | np.ndarray,
    val_frac: float,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(train_mask, val_mask)`` grouped by unique scene seed.

    All rows with the same seed land in exactly one split.  The split is
    deterministic for a given ``(unique seeds, val_frac, seed)``.
    """
    seeds = np.asarray(scene_seed, dtype=np.int64)
    if seeds.ndim != 1:
        raise ValueError(f"scene_seed must be 1-D, got shape={seeds.shape}")
    if not (0.0 <= float(val_frac) < 1.0):
        raise ValueError(f"val_frac must be in [0,1), got {val_frac!r}")
    unique = np.unique(seeds)
    if len(unique) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    if len(unique) == 1 or val_frac <= 0.0:
        return np.ones(len(seeds), dtype=bool), np.zeros(len(seeds), dtype=bool)

    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(unique) * float(val_frac))))
    n_val = min(n_val, len(unique) - 1)
    val_seeds = set(int(x) for x in shuffled[-n_val:])
    val_mask = np.array([int(x) in val_seeds for x in seeds], dtype=bool)
    return ~val_mask, val_mask


def assert_disjoint(*masks_and_names: tuple[np.ndarray, str]) -> None:
    """Raise if any two boolean masks overlap; useful for P0 split tests."""
    for i in range(len(masks_and_names)):
        for j in range(i + 1, len(masks_and_names)):
            a, an = masks_and_names[i]
            b, bn = masks_and_names[j]
            overlap = int(np.logical_and(np.asarray(a, bool), np.asarray(b, bool)).sum())
            if overlap:
                raise AssertionError(f"{an} and {bn} overlap on {overlap} rows")


def unique_seed_count(scene_seed: Iterable[int]) -> int:
    return int(len(np.unique(np.asarray(list(scene_seed), dtype=np.int64))))


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a generated spec file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_split_manifest(
    manifest_path: str | Path,
    specs: Sequence[Any],
    *,
    split: str,
    spec_path: str | Path,
) -> dict:
    """Validate that ``specs`` is exactly the frozen seed set for ``split``.

    The manifest is tracked under ``data/splits``; the raw spec file is
    generated/ignored.  Evaluation must call this before running a gate.
    """
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    # Support both the compact per-split file (data/splits/val_500.json) and
    # the older combined {val:..., test:...} manifest shape.
    block = raw.get(split) if isinstance(raw.get(split), dict) else raw
    if not isinstance(block, dict) or str(block.get("split", split)) != split:
        raise ValueError(f"manifest {manifest_path} has no valid {split!r} block")
    seeds = [int(s.seed) for s in specs]
    expected = [int(x) for x in block.get("seeds", [])]
    if int(block.get("n", -1)) != len(specs):
        raise ValueError(
            f"split {split!r}: manifest n={block.get('n')} != specs n={len(specs)}"
        )
    if seeds != expected:
        raise ValueError(f"split {split!r}: spec seed order/set differs from manifest")
    if any(str(getattr(s, "split", "")) != split for s in specs):
        bad = sorted({str(getattr(s, "split", "")) for s in specs})[:5]
        raise ValueError(f"split {split!r}: specs contain split fields {bad}")
    digest = file_sha256(spec_path)
    if digest != str(block.get("sha256", "")):
        raise ValueError(
            f"split {split!r}: spec SHA-256 {digest} != manifest {block.get('sha256')}"
        )
    return raw


# ---------------------------------------------------------------------------
# RL subset manifests
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RLSpecSplit:
    """Frozen RL train/holdout subsets of one spec file."""

    train: List[Any]
    holdout: List[Any]
    difficulty_weights: Dict[str, float]
    train_manifest: Mapping[str, Any]
    holdout_manifest: Mapping[str, Any]


def load_manifest(path: str | Path, *, expected_role: str | None = None) -> Dict[str, Any]:
    """Load one split manifest and validate its basic seed invariants."""
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
    """Return specs in manifest seed order; reject missing seeds."""
    by_seed = {int(getattr(spec, "seed")): spec for spec in specs}
    seeds = [int(x) for x in manifest["seeds"]]
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"manifest seeds missing from specs: {missing[:5]}")
    return [by_seed[seed] for seed in seeds]


def load_rl_spec_splits(
    specs: Sequence[Any],
    *,
    train_manifest: str | Path,
    holdout_manifest: str | Path,
    specs_path: str | Path | None = None,
) -> RLSpecSplit:
    """Load the canonical RL train/holdout subset manifests.

    ``specs`` must be the full spec sequence (``train_2000``).  Train and
    holdout must be disjoint and cover every supplied scene seed.
    """
    train_m = load_manifest(train_manifest, expected_role="rl_train")
    holdout_m = load_manifest(holdout_manifest, expected_role="rl_holdout")
    for manifest in (train_m, holdout_m):
        spec_path = str(manifest.get("spec_path", ""))
        if spec_path and file_sha256(spec_path) != str(manifest.get("sha256", "")):
            raise ValueError(f"{manifest.get('name')}: spec sha256 mismatch")
        if specs_path is not None and Path(spec_path).resolve() != Path(specs_path).resolve():
            raise ValueError(
                f"{manifest.get('name')}: manifest spec_path {spec_path!r} "
                f"does not match supplied specs {str(specs_path)!r}"
            )
    train_seeds = [int(x) for x in train_m["seeds"]]
    holdout_seeds = [int(x) for x in holdout_m["seeds"]]
    if set(train_seeds) & set(holdout_seeds):
        raise ValueError("RL train/holdout overlap")
    all_spec_seeds = {int(getattr(spec, "seed")) for spec in specs}
    if set(train_seeds) | set(holdout_seeds) != all_spec_seeds:
        raise ValueError("RL train+holdout must cover the full spec set")
    weights = {str(k): float(v) for k, v in train_m.get("difficulty_weights", {}).items()}
    for key, value in weights.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"difficulty weight {key!r} must be finite and > 0")
    return RLSpecSplit(
        train=filter_specs_by_manifest(specs, train_m),
        holdout=filter_specs_by_manifest(specs, holdout_m),
        difficulty_weights=weights,
        train_manifest=train_m,
        holdout_manifest=holdout_m,
    )
