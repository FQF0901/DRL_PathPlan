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
from pathlib import Path
from typing import Any, Iterable, Sequence, Tuple

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

    The manifest is tracked under ``config/splits``; the raw spec file is
    generated/ignored.  Evaluation must call this before running a gate.
    """
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    # Support both the compact per-split file (config/splits/val500.json) and
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
