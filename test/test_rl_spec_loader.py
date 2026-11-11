from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset.rl_splits import filter_specs_by_manifest, load_manifest, load_rl_spec_splits
from env.scenario.spec import load_specs

SPECS = load_specs("data/specs/train_2000.json")
TRAIN = "data/splits/train_2000_rl_train_1500.json"
HOLDOUT = "data/splits/train_2000_rl_holdout_500.json"
CAL = "data/splits/train_2000_rl_calibration_200.json"


def test_roles_and_counts():
    assert load_manifest(TRAIN, expected_role="rl_train")["n"] == 1500
    assert load_manifest(HOLDOUT, expected_role="rl_holdout")["n"] == 500
    assert load_manifest(CAL, expected_role="rl_calibration")["n"] == 200


def test_load_rl_spec_splits():
    split = load_rl_spec_splits(
        SPECS, train_manifest=TRAIN, holdout_manifest=HOLDOUT,
        calibration_manifest=CAL, specs_path="data/specs/train_2000.json",
    )
    assert len(split.train) == 1500
    assert len(split.holdout) == 500
    assert len(split.calibration) == 200
    assert set(s.seed for s in split.train).isdisjoint({s.seed for s in split.holdout})
    assert {s.seed for s in split.calibration} <= {s.seed for s in split.train}
    assert split.difficulty_weights["hard"] > split.difficulty_weights["easy"]


def test_manifest_rejects_wrong_role():
    with pytest.raises(ValueError, match="role"):
        load_manifest(HOLDOUT, expected_role="rl_train")


def test_filter_rejects_missing_seed(tmp_path: Path):
    payload = json.loads(Path(TRAIN).read_text())
    payload["seeds"] = payload["seeds"] + [999999]
    payload["n"] = len(payload["seeds"])
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload))
    m = load_manifest(p)
    with pytest.raises(ValueError, match="missing"):
        filter_specs_by_manifest(SPECS, m)
