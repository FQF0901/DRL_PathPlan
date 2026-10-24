"""Checkpoint schema/action-bound validation tests."""

from __future__ import annotations

import pytest
import torch

from env.action.action_space import CLEAN_BOUNDS
from net.model import StructuredPlanner
from pipeline.checkpointing import load_model_from_checkpoint, validate_checkpoint


def _payload(*, bounds=None):
    model = StructuredPlanner(hidden=32, heads=2, spatial_layers=1)
    low = CLEAN_BOUNDS.low if bounds is None else bounds[0]
    high = CLEAN_BOUNDS.high if bounds is None else bounds[1]
    return {
        "model": model.state_dict(),
        "config": {
            "architecture": "StructuredPlanner",
            "schema_name": "drl_pathplan_obs",
            "schema_version": 2,
            "hidden": 32,
            "heads": 2,
            "spatial_layers": 1,
            "action_low": list(map(float, low)),
            "action_high": list(map(float, high)),
            "decision": {"dt": 0.5, "steps_per_policy": 5, "history_frames": 6},
        },
    }


def test_valid_checkpoint_passes_strict(tmp_path):
    path = tmp_path / "clean.pt"
    torch.save(_payload(), path)
    model = load_model_from_checkpoint(str(path), "cpu", strict=True)
    assert model.hidden == 32
    assert model.eval_config["hidden"] == 32


def test_incompatible_action_bounds_are_rejected(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(_payload(bounds=([0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])), path)
    with pytest.raises(ValueError, match="action_"):
        load_model_from_checkpoint(str(path), "cpu", strict=True)


def test_legacy_checkpoint_is_explicitly_warned():
    payload = _payload()
    payload["config"] = {"hidden": 32, "heads": 2, "spatial_layers": 1}
    _state, _config, info = validate_checkpoint(payload, strict=False)
    assert info.warnings
    assert any("schema_name" in w for w in info.warnings)
