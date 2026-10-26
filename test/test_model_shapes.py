"""Shape/contract tests for the bank-native DriveModel."""

from __future__ import annotations

import torch

from net.build_net import build_net, load_config
from net.model import DriveModel, count_parameters


B = 2


def _batch(b: int = B) -> dict:
    return {
        "od": torch.randn(b, 16, 18),
        "od_mask": torch.ones(b, 16),
        "ld": torch.randn(b, 16, 5, 16),
        "ld_mask": torch.ones(b, 16, 5),
        "nav": torch.randn(b, 108),
        "ego": torch.randn(b, 12),
        "od_ids": torch.arange(16).repeat(b, 1),
        "ld_ids": torch.arange(16).repeat(b, 1),
        "pose": torch.zeros(b, 3),
        "episode_id": "ep",
        "frame_id": 0,
        "timestamp": 0.0,
    }


def _model():
    ego = load_config()["ego"]
    return DriveModel(
        hidden=32, heads=4, spatial_layers=1,
        ego_dt=ego["dt"], ego_wheelbase=ego["wheelbase"],
        ego_max_steer_rad=ego["max_steer_rad"],
    )


def test_forward_shapes_and_finiteness():
    model = _model()
    out = model.forward_current(_batch())
    assert out["action_mu"].shape == (B, 4)
    assert out["action_logstd"].shape == (B, 4)
    assert set(out["aux"]) == {"collision", "offroad", "timeout", "urgency", "progress"}
    assert torch.all(out["action_mu"] >= 0.0) and torch.all(out["action_mu"] <= 1.0)
    assert torch.all(out["action_logstd"] >= -5.0) and torch.all(out["action_logstd"] <= 0.0)
    for value in out.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()


def test_all_invalid_entities_are_safe():
    model = _model()
    batch = _batch()
    batch["od_mask"][0] = 0.0
    batch["ld_mask"][0] = 0.0
    out = model.forward_current(batch)
    assert torch.isfinite(out["action_mu"]).all()


def test_build_net_from_config():
    model = build_net()
    assert isinstance(model, DriveModel)
    out = model.forward_current(_batch())
    assert out["action_mu"].shape == (B, 4)
    assert count_parameters(model) > 0
