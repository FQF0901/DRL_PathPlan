import hashlib
import json
from pathlib import Path

import pytest

from env.scenario.spec import load_specs
from pipeline.train.scene_manifest import load_scene_manifest

SPECS = "data/specs/train_2000.json"
STD_MIN = [0.005, 0.0, 0.0, 0.005]


def _payload(selected):
    return {
        "schema": "drl_pathplan_rl_scene_manifest_v1",
        "selected": selected,
        "members": 8,
        "std_min": STD_MIN,
        "action_mode": "4d_exact",
        "spec_sha256": hashlib.sha256(Path(SPECS).read_bytes()).hexdigest(),
    }


def _write(tmp_path: Path, payload) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_scene_manifest_filters_in_order(tmp_path):
    specs = load_specs(SPECS)
    selected = [int(specs[5].seed), int(specs[2].seed)]
    path = _write(tmp_path, _payload(selected))
    filtered, meta = load_scene_manifest(
        path,
        specs,
        specs_path=SPECS,
        group_size=8,
        std_min=STD_MIN,
        action_mode="4d_exact",
    )
    assert [int(spec.seed) for spec in filtered] == selected
    assert meta["selected"] == 2
    assert meta["action_mode"] == "4d_exact"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema="bad"),
        lambda p: p.update(selected=[]),
        lambda p: p.update(selected=[1000, 1000]),
        lambda p: p.update(selected=[999999]),
        lambda p: p.update(members=4),
        lambda p: p.update(std_min=[0.01, 0.0, 0.0, 0.01]),
        lambda p: p.update(action_mode="3d_derived_dx"),
        lambda p: p.update(spec_sha256="bad"),
    ],
)
def test_load_scene_manifest_rejects_invalid(tmp_path, mutate):
    specs = load_specs(SPECS)
    payload = _payload([int(specs[0].seed)])
    mutate(payload)
    path = _write(tmp_path, payload)
    with pytest.raises(ValueError):
        load_scene_manifest(
            path,
            specs,
            specs_path=SPECS,
            group_size=8,
            std_min=STD_MIN,
            action_mode="4d_exact",
        )
