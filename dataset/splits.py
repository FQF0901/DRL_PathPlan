"""Scene-level split helpers.

The contract forbids train/validation/test leakage by map/scene seed
(``docs/arch_lock.md`` §14.1).  Episode ids are *not* sufficient: the same
scene can be re-collected as several episodes (BC + DAgger), so splitting by
episode can put two copies of one scene on both sides.

This module is deliberately tiny and dependency-free so both the dataset
loader and tests use the same rule.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

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
