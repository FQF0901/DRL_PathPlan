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


def test_frozen_val_and_test_manifests_are_verified():
    from dataset.splits import verify_split_manifest
    from env.scenario.spec import load_specs

    val = load_specs("data/specs/val_500.json")
    test = load_specs("data/specs/test_500.json")
    raw_v = verify_split_manifest(
        "data/splits/val_500.json", val, split="val", spec_path="data/specs/val_500.json"
    )
    raw_t = verify_split_manifest(
        "data/splits/test_500.json", test, split="test", spec_path="data/specs/test_500.json"
    )
    assert raw_v["split"] == "val" and raw_t["split"] == "test"
    assert {s.seed for s in val}.isdisjoint({s.seed for s in test})
    assert all(s.split == "test" for s in test)


def test_canonical_train_val_test_sizes_and_disjoint():
    from env.scenario.spec import load_specs

    train = load_specs("data/specs/train_2000.json")
    val = load_specs("data/specs/val_500.json")
    test = load_specs("data/specs/test_500.json")
    assert len(train) == 2000 and len(val) == 500 and len(test) == 500
    assert min(s.seed for s in train) == 1000 and max(s.seed for s in train) == 2999
    tr = {s.seed for s in train}
    va = {s.seed for s in val}
    te = {s.seed for s in test}
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
