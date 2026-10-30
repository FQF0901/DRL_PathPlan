"""Checkpoint schema/action-bound validation tests for DriveModel."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from env.action.config import load_action_contract
from net.model import DriveModel
from pipeline.checkpointing import (
    load_drive_model_from_checkpoint,
    load_model_from_checkpoint,
    save_drive_model_checkpoint,
)


def _model(hidden: int = 32, *, low=None, high=None) -> DriveModel:
    contract = load_action_contract()
    return DriveModel(
        hidden=hidden,
        heads=2,
        spatial_layers=1,
        action_low=contract.bounds.low if low is None else low,
        action_high=contract.bounds.high if high is None else high,
        ego_dt=0.5,
        ego_wheelbase=2.46894,
        ego_max_steer_rad=0.6981317007977318,
    )


def test_drive_checkpoint_strict_roundtrip(tmp_path):
    model = _model()
    path = tmp_path / "drive.pt"
    save_drive_model_checkpoint(model, str(path))
    loaded = load_model_from_checkpoint(str(path), "cpu", strict=True)
    assert isinstance(loaded, DriveModel)
    assert loaded.hidden == 32
    assert loaded.eval_config["hidden"] == 32


def test_custom_action_bounds_roundtrip(tmp_path):
    low = [0.5, -1.25, -0.7, 0.0]
    high = [8.0, 1.25, 0.7, 16.0]
    model = _model(low=low, high=high)
    path = tmp_path / "custom.pt"
    save_drive_model_checkpoint(model, str(path))
    loaded = load_drive_model_from_checkpoint(str(path), "cpu", strict=True)
    assert np.allclose(loaded.action_low.tolist(), low)
    assert np.allclose(loaded.action_high.tolist(), high)


def test_legacy_checkpoint_is_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model": {"value.weight": torch.zeros(1, 1)}, "config": {}}, path)
    with pytest.raises(ValueError, match="unsupported|must have|validation failed"):
        load_model_from_checkpoint(str(path), "cpu", strict=True)


def test_merge_wm_into_drive_roundtrip(tmp_path):
    from net.world_model import EgoConditionedLatentWM
    from pipeline.checkpointing import merge_wm_into_drive

    drive_path = tmp_path / "drive.pt"
    save_drive_model_checkpoint(_model(), str(drive_path))
    wm = EgoConditionedLatentWM(hidden=32, action_dim=4)
    wm_path = tmp_path / "wm.pt"
    torch.save({"model": wm.state_dict(), "config": {"hidden": 32, "action_dim": 4}}, wm_path)
    merged = tmp_path / "merged.pt"
    merge_wm_into_drive(str(drive_path), str(wm_path), str(merged), device="cpu")
    loaded = load_drive_model_from_checkpoint(str(merged), "cpu", strict=True)
    assert loaded.world_model is not None


def test_best_drive_loads_as_drive_model(tmp_path):
    path = tmp_path / "best_drive.pt"
    save_drive_model_checkpoint(_model(), str(path))
    loaded = load_model_from_checkpoint(str(path), "cpu", strict=True)
    assert isinstance(loaded, DriveModel)

def test_bad_schema_is_rejected_in_strict_mode(tmp_path):
    from pipeline.checkpointing import load_model_from_checkpoint, save_drive_model_checkpoint

    path = tmp_path / "bad.pt"
    save_drive_model_checkpoint(_model(), str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["schema_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="validation failed"):
        load_model_from_checkpoint(str(path), "cpu", strict=True)


def test_mutated_top_level_bounds_are_rejected(tmp_path):
    path = tmp_path / "bounds.pt"
    save_drive_model_checkpoint(_model(), str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["config"]["action_low"] = [0.0, 0.0, 0.0, 0.0]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="validation failed"):
        load_model_from_checkpoint(str(path), "cpu", strict=True)
