import numpy as np
from dataset.splits import split_by_scene_seed, assert_disjoint


def test_split_by_scene_seed_groups_duplicates():
    seeds = np.array([10, 10, 11, 11, 12, 12, 13, 13], dtype=np.int64)
    train, val = split_by_scene_seed(seeds, val_frac=0.25, seed=3)
    assert train.any() and val.any()
    assert not np.logical_and(train, val).any()
    for seed in np.unique(seeds):
        assert np.all(train[seeds == seed]) or np.all(val[seeds == seed]), f"seed {seed} leaked across split"


def test_split_is_deterministic_and_disjoint():
    seeds = np.arange(1000, 1100).repeat(2)
    a = split_by_scene_seed(seeds, 0.2, seed=7)
    b = split_by_scene_seed(seeds, 0.2, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert_disjoint((a[0], "train"), (a[1], "val"))
