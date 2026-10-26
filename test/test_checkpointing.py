"""Checkpoint schema/action-bound validation tests."""

from __future__ import annotations

import pytest
import torch

from pipeline.action_compat import CLEAN_BOUNDS
from pipeline.legacy_model import StructuredPlanner
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


def test_load_drive_model_from_adapter_checkpoint(tmp_path):
    from pipeline.checkpointing import load_drive_model_from_checkpoint
    from pipeline.legacy_model import StructuredPlanner
    from env.action.config import load_action_contract

    model = StructuredPlanner(hidden=32, heads=4, spatial_layers=1)
    contract = load_action_contract()
    path = tmp_path / "drive.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "architecture": "DriveModelAdapter",
            "hidden": 32,
            "heads": 4,
            "spatial_layers": 1,
            "action": {
                "bounds": {
                    "low": contract.bounds.low.tolist(),
                    "high": contract.bounds.high.tolist(),
                }
            },
            "ego": {
                "dt": 0.5,
                "wheelbase": 2.46894,
                "max_steer_rad": 0.6981317007977318,
            },
        },
    }, path)
    loaded = load_drive_model_from_checkpoint(str(path), "cpu", strict=True)
    assert loaded.hidden == 32
    assert loaded.training is False


def test_load_drive_model_from_top_level_action_checkpoint(tmp_path):
    from pipeline.checkpointing import load_drive_model_from_checkpoint
    from pipeline.legacy_model import StructuredPlanner

    model = StructuredPlanner(hidden=32, heads=4, spatial_layers=1)
    path = tmp_path / "adapter-top.pt"
    payload = _payload()
    payload["model"] = model.state_dict()
    payload["config"]["architecture"] = "DriveModelAdapter"
    payload["config"]["hidden"] = 32
    payload["config"]["heads"] = 4
    payload["config"]["spatial_layers"] = 1
    torch.save(payload, path)
    loaded = load_drive_model_from_checkpoint(str(path), "cpu", strict=True)
    assert loaded.hidden == 32


def test_merge_wm_into_drive_checkpoint_roundtrip(tmp_path):
    from pipeline.checkpointing import load_drive_model_from_checkpoint, merge_wm_into_drive
    from pipeline.legacy_model import StructuredPlanner
    from net.world_model import EgoConditionedLatentWM

    adapter = StructuredPlanner(hidden=32, heads=4, spatial_layers=1)
    drive_path = tmp_path / "drive.pt"
    torch.save({
        "model": adapter.state_dict(),
        "config": {
            "architecture": "DriveModelAdapter",
            "hidden": 32,
            "heads": 4,
            "spatial_layers": 1,
            "action_low": [0.0, -1.5, -0.8, 0.0],
            "action_high": [7.5, 1.5, 0.8, 15.0],
        },
    }, drive_path)
    wm = EgoConditionedLatentWM(hidden=32, action_dim=4)
    wm_path = tmp_path / "wm.pt"
    torch.save({"model": wm.state_dict(), "config": {"hidden": 32, "action_dim": 4}}, wm_path)
    merged = tmp_path / "merged.pt"
    merge_wm_into_drive(str(drive_path), str(wm_path), str(merged), device="cpu")
    loaded = load_drive_model_from_checkpoint(str(merged), "cpu", strict=True)
    assert loaded.world_model is not None


def test_drive_checkpoint_preserves_custom_action_bounds(tmp_path):
    from env.action.config import load_action_contract
    from pipeline.checkpointing import load_drive_model_from_checkpoint, save_drive_model_checkpoint
    from pipeline.legacy_model import StructuredPlanner

    low = [0.5, -1.25, -0.7, 0.0]
    high = [8.0, 1.25, 0.7, 16.0]
    adapter = StructuredPlanner(hidden=32, heads=4, spatial_layers=1, action_low=low, action_high=high)
    path = tmp_path / "custom.pt"
    save_drive_model_checkpoint(adapter.drive, str(path))
    loaded = load_drive_model_from_checkpoint(str(path), "cpu", strict=True)
    import numpy as np
    assert np.allclose(loaded.action_low.tolist(), low)
    assert np.allclose(loaded.action_high.tolist(), high)


def test_best_drive_loads_into_eval_adapter(tmp_path):
    from pipeline.checkpointing import load_model_from_checkpoint, save_drive_model_checkpoint
    from pipeline.legacy_model import StructuredPlanner

    adapter = StructuredPlanner(hidden=32, heads=4, spatial_layers=1)
    path = tmp_path / "best_drive.pt"
    save_drive_model_checkpoint(adapter.drive, str(path))
    loaded = load_model_from_checkpoint(str(path), "cpu", strict=True)
    assert isinstance(loaded, StructuredPlanner)
    assert loaded.hidden == 32


def test_legacy_action_limits_are_preserved(tmp_path):
    from pipeline.checkpointing import load_model_from_checkpoint
    from pipeline.legacy_model import StructuredPlanner

    model = StructuredPlanner(hidden=32, heads=4, spatial_layers=1)
    payload = {
        "model": model.state_dict(),
        "config": {
            "architecture": "DriveModelAdapter",
            "schema_name": "drl_pathplan_obs",
            "schema_version": 2,
            "hidden": 32,
            "heads": 4,
            "spatial_layers": 1,
            "action_bounds_name": "custom",
            "action_low": [0.0, -1.5, -0.8, 0.0],
            "action_high": [7.5, 1.5, 0.8, 15.0],
            "limits": {
                "hard": {"a_long_min": -9, "a_long_max": 3.5, "a_lat_max": 8, "kappa_max": 0.5, "kappa_rate_max": 1, "jerk_long_max": 20, "jerk_lat_max": 20},
                "comfort": {"a_long_min": -4, "a_long_max": 2, "a_lat_max": 3, "kappa_max": 0.25, "kappa_rate_max": 0.05, "jerk_long_max": 5, "jerk_lat_max": 5},
            },
            "decision": {"dt": 0.5, "steps_per_policy": 5, "history_frames": 6},
        },
    }
    path = tmp_path / "custom-limits.pt"
    torch.save(payload, path)
    loaded = load_model_from_checkpoint(str(path), "cpu", strict=True)
    assert loaded.action_config["limits"]["hard"]["kappa_max"] == 0.5
    assert loaded.action_config["limits"]["comfort"]["kappa_max"] == 0.25
