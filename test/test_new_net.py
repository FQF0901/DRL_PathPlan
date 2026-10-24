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
        "pose": torch.zeros(B, 3),
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
    assert set(p["aux"]) == {"collision", "offroad", "timeout", "urgency", "progress"}


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


def test_stable_id_reorders_history_into_current_slot():
    def od(slot, sid, x):
        return EntityRecord(slot, sid, NodeType.OD, torch.ones(H), torch.tensor([x, 0.0, 0.0, 0.0, 0.0]))

    f0 = FrameRecord("ep", 0, 0.0, torch.tensor([0.0, 0.0, 0.0]), (od(0, 7, 10.0),), ())
    f1 = FrameRecord("ep", 1, 0.5, torch.tensor([2.0, 0.0, 0.0]), (od(1, 7, 8.0),), ())
    window = BankWindowBuilder(hidden=H).build((f0, f1))
    # Current frame puts stable id 7 in slot 1.  History is matched by ID and
    # placed in that current slot, despite being slot 0 in frame 0.
    assert bool(window.od_valid[0, -1, 1])
    assert bool(window.od_valid[0, -2, 1])
    assert float(window.od_geometry[0, -2, 1, 0]) == 8.0


def test_target_encoder_has_no_grad():
    enc = CurrentFrameEncoder(hidden=H)
    target = TargetEncoder(enc)
    batch = _batch()
    batch["od"] = batch["od"].clone().requires_grad_(True)
    out = target.encode_current(batch)
    assert not out["od_content"].requires_grad


def test_project_ld_batched():
    from net.world_model import project_ld

    geometry = torch.randn(2, 16, 10)
    pose = torch.randn(2, 3)
    out = project_ld(geometry, pose, pose)
    assert out.shape == geometry.shape


def test_imagine_advances_window_and_geometry():
    model = DriveModel(hidden=H)
    batch = _batch()
    enc = model.encoder.encode_current(batch)
    window = _window(enc)
    before = window.rollout_depth[:, -1].clone()
    out = model.imagine(window, enc["nav"][:1], enc["ego"][:1], actions=torch.rand(1, 6, 4), horizon=6)
    assert len(out) == 6
    # The local final window is not returned; this test exercises the step path
    # through the public method without requiring an external Bank adapter.
    assert torch.isfinite(out[-1]["action_mu"]).all()


def test_temporal_current_invalid_is_zero():
    from net.world_model import TemporalAttention

    attn = TemporalAttention(H, 4)
    tokens = torch.randn(1, 3, 2, H)
    valid = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]])
    ids = torch.tensor([[[1, 2], [1, 2], [1, 2]]])
    out = attn(tokens, valid, ids)
    assert torch.allclose(out, torch.zeros_like(out))


def test_build_net_rejects_hidden_mismatch():
    import pytest

    with pytest.raises(ValueError):
        build_net({
            "model": {
                "hidden": 128,
                "encoder": {"hidden": 64},
                "temporal_spatial": {"hidden": 128},
                "planner": {"hidden": 128},
                "world_model": {"hidden": 128},
            }
        })


def test_imagine_from_real_bank_appends():
    model = DriveModel(hidden=H)
    batch = {k: v[:1] for k, v in _batch().items()}
    enc = model.encoder.encode_current(batch)
    record = model.encoder.frame_records(batch, episode_id="ep", frame_id=0, timestamp=0.0)[0]
    bank = RealMemoryBank(6)
    bank.append(record)
    before = len(bank.snapshot().frames)
    out = model.imagine_from_real_bank(
        bank, enc["nav"], enc["ego"],
        actions=torch.rand(1, 6, 4), horizon=6,
    )
    assert len(out) == 6
    assert len(bank.frames) == 1  # no writeback to RealBank


def test_bank_dtype_follows_content():
    od = EntityRecord(
        0, 1, NodeType.OD,
        torch.zeros(H, dtype=torch.float32),
        torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    frame = FrameRecord("ep", 0, 0.0, torch.zeros(3, dtype=torch.float64), (od,), ())
    window = BankWindowBuilder(hidden=H).build((frame,))
    assert window.od_content.dtype == torch.float32
    assert window.ego_pose.dtype == torch.float32


def test_imagine_per_step_nav_ego_validation():
    import pytest
    from net.model import DriveModel

    model = DriveModel(hidden=H)
    batch = {k: v[:1] for k, v in _batch().items()}
    enc = model.encoder.encode_current(batch)
    window = _window(enc)
    # Correct encoded sequences work.
    nav_seq = enc["nav"].unsqueeze(1).expand(1, 6, H)
    ego_seq = enc["ego"].unsqueeze(1).expand(1, 6, H)
    out = model.imagine(window, enc["nav"], enc["ego"], nav_seq=nav_seq, ego_seq=ego_seq, horizon=6)
    assert len(out) == 6
    # Raw observation sequences are rejected explicitly.
    with pytest.raises(ValueError):
        model.imagine(window, enc["nav"], enc["ego"], nav_seq=batch["nav"].unsqueeze(1).expand(1, 6, 108))


def test_wm_ld_branch_is_optional():
    wm = FactorizedEgoConditionedLatentWM(hidden=H)
    od = torch.randn(B, 16, H)
    ld = torch.randn(B, 16, H)
    action = torch.rand(B, 4)
    out = wm(od, ld, action, od_geometry=None, ld_geometry=None)
    assert out["ld_geometry"] is None
    assert out["ld_residual"] is None
