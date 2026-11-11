import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "data/specs/train_2000.json"
TRAIN_MANIFEST = ROOT / "data/splits/train_2000_rl_train_1500.json"
HOLDOUT_MANIFEST = ROOT / "data/splits/train_2000_rl_holdout_500.json"


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def spec():
    return _load_json(SPEC_PATH)


@pytest.fixture(scope="module")
def manifests():
    return _load_json(TRAIN_MANIFEST), _load_json(HOLDOUT_MANIFEST)


def test_manifest_schema_and_metadata(manifests):
    train, holdout = manifests
    expected_sha = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    expected = (
        (train, "train_2000_rl_train_1500", 1500, [1000, 2499]),
        (holdout, "train_2000_rl_holdout_500", 500, [2500, 2999]),
    )
    for manifest, name, n, seed_range in expected:
        assert manifest["schema"] == "pnp_data_split_v1"
        assert manifest["name"] == name
        assert manifest["split"] == "train"
        assert manifest["source"]
        assert manifest["spec_path"] == "data/specs/train_2000.json"
        assert manifest["n"] == n == len(manifest["seeds"])
        assert manifest["seed_range"] == seed_range
        assert manifest["sha256"] == expected_sha


def test_manifest_seeds_are_strictly_ascending_unique_and_in_range(manifests):
    for manifest in manifests:
        seeds = manifest["seeds"]
        assert all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
        assert seeds == sorted(seeds)
        assert len(seeds) == len(set(seeds)), f"{manifest['name']} has duplicate seeds"
        assert all(1000 <= seed <= 2999 for seed in seeds)


def test_rl_train_and_holdout_partition_full_train_spec(manifests, spec):
    train, holdout = manifests
    train_seeds = train["seeds"]
    holdout_seeds = holdout["seeds"]
    full_seeds = [int(item["seed"]) for item in spec["specs"]]

    assert train_seeds == list(range(1000, 2500))
    assert holdout_seeds == list(range(2500, 3000))
    assert set(train_seeds).isdisjoint(holdout_seeds)
    assert sorted(train_seeds + holdout_seeds) == sorted(full_seeds)
    assert len(train_seeds) + len(holdout_seeds) == len(full_seeds) == 2000


def test_rl_subsets_cover_all_difficulties(manifests, spec):
    difficulty_by_seed = {
        int(item["seed"]): item["difficulty"] for item in spec["specs"]
    }
    for manifest in manifests:
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for seed in manifest["seeds"]:
            counts[difficulty_by_seed[seed]] += 1
        assert all(count > 0 for count in counts.values()), (
            f"{manifest['name']} difficulty distribution is {counts}"
        )
        assert sum(counts.values()) == manifest["n"]
