"""Smoke tests for the bank-native network modules."""

from __future__ import annotations

import torch

from net.build_net import build_net
from net.encoder import CurrentFrameEncoder, TargetEncoder
from net.mem_bank import (
    BankWindowBuilder,
    EntityRecord,
    FrameRecord,
    NodeType,
    RealMemoryBank,
    Source,
)
from net.model import DriveModel
from net.planner import Planner
from net.world_model import TemporalSpatialBlock, FactorizedEgoConditionedLatentWM


H = 16
B = 2


def _batch():
    return {
        "od": torch.randn(B, 16, 18),
        "od_mask": torch.ones(B, 16),
        "ld": torch.randn(B, 16, 5, 16),
        "ld_mask": torch.ones(B, 16, 5),
        "nav": torch.randn(B, 108),
        "ego": torch.randn(B, 12),
        "od_ids": torch.arange(16).repeat(B, 1),
        "ld_ids": torch.arange(16).repeat(B, 1),
    }


def _window(enc):
    od = tuple(
        EntityRecord(
            s,
            int(enc["od_stable_id"][0, s]),
            NodeType.OD,
            enc["od_content"][0, s],
            enc["od_geometry"][0, s],
            int(enc["od_type"][0, s]),
            bool(enc["od_valid"][0, s]),
        )
        for s in range(16)
    )
    ld = tuple(
        EntityRecord(
            s,
            int(enc["ld_stable_id"][0, s]),
            NodeType.LD,
            enc["ld_content"][0, s],
            enc["ld_geometry"][0, s],
            int(enc["ld_type"][0, s]),
            bool(enc["ld_valid"][0, s]),
        )
        for s in range(16)
    )
    frame = FrameRecord("ep", 0, 0.0, torch.zeros(3), od, ld, source=Source.REAL)
    return BankWindowBuilder(hidden=H).build((frame,))


def test_bank_window_and_geometry_transform():
    bank = RealMemoryBank(6)
    for t in range(2):
        od = (EntityRecord(0, 7, NodeType.OD, torch.ones(H), torch.tensor([10.0 - 2 * t, 0.0, 0.0, 1.0, 0.0])),)
        ld = (EntityRecord(0, 8, NodeType.LD, torch.ones(H), torch.tensor([10.0 - 2 * t, 0.0, 11.0 - 2 * t, 0.0, 12.0 - 2 * t, 0.0, 13.0 - 2 * t, 0.0, 14.0 - 2 * t, 0.0])),)
        bank.append(
            FrameRecord("ep", t, float(t) * 0.5, torch.tensor([2.0 * t, 0.0, 0.0]), od, ld)
        )
    window = bank.window(builder=BankWindowBuilder(hidden=H))
    assert window.od_content.shape == (1, 6, 16, H)
    assert window.od_valid[0, -1, 0]
    assert torch.allclose(window.od_geometry[0, -1, 0, 0], torch.tensor(8.0))
    assert window.ld_valid[0, -1, 0]
    pose = BankWindowBuilder.relative_pose(window.od_geometry, NodeType.OD)
    assert pose.shape == (1, 6, 16, 3)


def test_encoder_and_target_clone():
    enc = CurrentFrameEncoder(hidden=H)
    out = enc.encode_current(_batch())
    assert out["od_content"].shape == (B, 16, H)
    assert out["ld_geometry"].shape == (B, 16, 10)
    target = TargetEncoder(enc).eval()
    assert all(not p.requires_grad for p in target.parameters())
    with torch.no_grad():
        target.update_ema(enc, decay=0.5)


def test_temporal_spatial_and_planner():
    enc = CurrentFrameEncoder(hidden=H)
    out = enc.encode_current(_batch())
    window = _window(out)
    block = TemporalSpatialBlock(hidden=H, heads=4, spatial_layers=1)
    z = block(window, out["nav"][:1], out["ego"][:1])
    assert z["od"].shape == (1, 16, H)
    assert z["ld"].shape == (1, 16, H)
    planner = Planner(hidden=H)
    p = planner(z["od"], z["ld"], out["nav"][:1], out["ego"][:1], window.od_valid[:, -1], window.ld_valid[:, -1])
    assert p["action_mu"].shape == (1, 4)
    assert set(p["aux"]) == {"collision", "offroad", "timeout", "progress"}


def test_build_and_imagine():
    model = build_net()
    assert isinstance(model, DriveModel)
    out = model.forward_current(_batch())
    assert out["action_mu"].shape == (B, 4)
    # one-window imagine path
    batch_one = {k: v[:1] for k, v in _batch().items()}
    enc = model.encoder.encode_current(batch_one)
    window = _window(enc)
    traj = model.imagine(window, enc["nav"], enc["ego"], actions=torch.rand(1, 6, 4), horizon=6)
    assert len(traj) == 6
    assert all(p["action_mu"].shape == (1, 4) for p in traj)
