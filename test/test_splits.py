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


# ---------------------------------------------------------------------------
# RL subset manifests
# ---------------------------------------------------------------------------
def test_rl_manifest_roles_and_counts():
    from dataset.splits import load_manifest

    train = load_manifest(
        "data/splits/train_2000_rl_train_1500.json", expected_role="rl_train"
    )
    holdout = load_manifest(
        "data/splits/train_2000_rl_holdout_500.json", expected_role="rl_holdout"
    )
    assert train["n"] == 1500
    assert holdout["n"] == 500


def test_load_rl_spec_splits():
    from dataset.splits import load_rl_spec_splits
    from env.scenario.spec import load_specs

    specs = load_specs("data/specs/train_2000.json")
    split = load_rl_spec_splits(
        specs,
        train_manifest="data/splits/train_2000_rl_train_1500.json",
        holdout_manifest="data/splits/train_2000_rl_holdout_500.json",
        specs_path="data/specs/train_2000.json",
    )
    assert len(split.train) == 1500
    assert len(split.holdout) == 500
    train_seeds = {int(spec.seed) for spec in split.train}
    holdout_seeds = {int(spec.seed) for spec in split.holdout}
    assert train_seeds.isdisjoint(holdout_seeds)
    assert train_seeds | holdout_seeds == {int(spec.seed) for spec in specs}
    assert split.difficulty_weights["hard"] > split.difficulty_weights["easy"]


def test_rl_manifest_rejects_wrong_role():
    import pytest
    from dataset.splits import load_manifest

    with pytest.raises(ValueError, match="role"):
        load_manifest(
            "data/splits/train_2000_rl_holdout_500.json",
            expected_role="rl_train",
        )


def test_rl_filter_rejects_missing_seed(tmp_path):
    import json
    from pathlib import Path

    import pytest
    from dataset.splits import filter_specs_by_manifest, load_manifest
    from env.scenario.spec import load_specs

    payload = json.loads(
        Path("data/splits/train_2000_rl_train_1500.json").read_text(encoding="utf-8")
    )
    payload["seeds"] = payload["seeds"] + [999999]
    payload["n"] = len(payload["seeds"])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_manifest(path)
    with pytest.raises(ValueError, match="missing"):
        filter_specs_by_manifest(load_specs("data/specs/train_2000.json"), manifest)
