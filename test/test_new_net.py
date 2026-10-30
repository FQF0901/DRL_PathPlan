"""Smoke tests for the new bank-native network contract."""

from __future__ import annotations

import pytest
import torch

from env.action.config import load_action_contract

CLEAN_BOUNDS = load_action_contract().bounds
from net.build_net import build_net, load_config
from net.encoder import CurrentFrameEncoder, TargetEncoder
from net.mem_bank import (
    BankWindow,
    BankWindowBuilder,
    EntityRecord,
    FrameRecord,
    ImagineMemoryBank,
    NodeType,
    RealMemoryBank,
    Source,
)
from net.model import DriveModel
from net.planner import Planner
from net.world_model import (
    EgoConditionedLatentWM,
    TemporalAttention,
    TemporalSpatialBlock,
    frame_delta,
)

H = 16
B = 2


def _model(hidden=H):
    ego = load_config()["ego"]
    return DriveModel(
        hidden=hidden,
        action_low=CLEAN_BOUNDS.low,
        action_high=CLEAN_BOUNDS.high,
        ego_dt=ego["dt"],
        ego_wheelbase=ego["wheelbase"],
        ego_max_steer_rad=ego["max_steer_rad"],
    )


def _batch(b: int = B):
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


def _window(enc):
    def make(content, geometry, typ, valid, sid, node):
        return tuple(
            EntityRecord(
                s,
                int(sid[0, s]),
                node,
                content[0, s],
                geometry[0, s],
                int(typ[0, s]),
                bool(valid[0, s]),
            )
            for s in range(16)
        )

    od = make(enc["od_content"], enc["od_geometry"], enc["od_type"], enc["od_valid"], enc["od_stable_id"], NodeType.OD)
    ld = make(enc["ld_content"], enc["ld_geometry"], enc["ld_type"], enc["ld_valid"], enc["ld_stable_id"], NodeType.LD)
    frame = FrameRecord("ep", 0, 0.0, torch.zeros(3), od, ld, source=Source.REAL)
    return BankWindowBuilder(hidden=H).build((frame,))


def test_bank_window_keeps_local_geometry_without_projection():
    bank = RealMemoryBank(6)
    for t in range(2):
        od = (EntityRecord(0, 7, NodeType.OD, torch.ones(H), torch.tensor([10.0 - 2 * t, 0.0, 0.0, 1.0, 0.0])),)
        ld = (EntityRecord(0, 8, NodeType.LD, torch.ones(H), torch.tensor([10.0 - 2 * t, 0.0, 11.0 - 2 * t, 0.0, 12.0 - 2 * t, 0.0, 13.0 - 2 * t, 0.0, 14.0 - 2 * t, 0.0])),)
        bank.append(FrameRecord("ep", t, float(t) * 0.5, torch.tensor([2.0 * t, 0.0, 0.0]), od, ld))
    window = bank.window(builder=BankWindowBuilder(hidden=H))
    assert window.od_content.shape == (1, 6, 16, H)
    assert window.od_valid[0, -1, 0]
    # No projection: geometry stays in each frame's local coordinates.
    assert torch.allclose(window.od_geometry[0, -1, 0, 0], torch.tensor(8.0))
    assert torch.allclose(window.od_geometry[0, -2, 0, 0], torch.tensor(10.0))
    assert window.ld_valid[0, -1, 0]


def test_frame_delta_moves_history_to_query():
    pose = torch.tensor([[[1.0, 2.0, 0.0], [3.0, 2.0, 0.0]]])
    delta = frame_delta(pose)
    assert delta.shape == (1, 2, 3)
    assert torch.allclose(delta[0, -1], torch.zeros(3), atol=1e-6)
    assert torch.allclose(delta[0, 0, 0], torch.tensor(2.0), atol=1e-6)


def test_encoder_target_and_z_geometry_coupling():
    enc = CurrentFrameEncoder(hidden=H)
    out = enc.encode_current(_batch())
    assert out["od_content"].shape == (B, 16, H)
    assert out["ld_content"].shape == (B, 16, H)
    changed = _batch()
    changed["od"][..., :2] += 5.0
    out2 = enc.encode_current(changed)
    assert not torch.allclose(out["od_content"], out2["od_content"])
    target = TargetEncoder(enc).eval()
    assert all(not p.requires_grad for p in target.parameters())
    with torch.no_grad():
        target.update_ema(enc, decay=0.5)


def test_tsb_temporal_spatial_and_policy():
    enc = CurrentFrameEncoder(hidden=H)
    out = enc.encode_current(_batch())
    window = _window(out)
    block = TemporalSpatialBlock(hidden=H, heads=4, spatial_layers=1)
    z = block(window, out["ego"][:1])
    assert z["od"].shape == (1, 16, H)
    assert z["ld"].shape == (1, 16, H)
    planner = Planner(hidden=H, heads=4)
    p = planner(z["od"], z["ld"], out["nav"][:1], out["ego"][:1], window.od_valid[:, -1], window.ld_valid[:, -1])
    assert p["action_mu"].shape == (1, 4)
    assert set(p["aux"]) == {"collision", "offroad", "timeout", "urgency", "progress"}


def test_policy_all_invalid_is_finite():
    planner = Planner(hidden=H, heads=4)
    od = torch.randn(1, 16, H)
    ld = torch.randn(1, 16, H)
    valid = torch.zeros(1, 16)
    out = planner(od, ld, torch.randn(1, H), torch.randn(1, H), valid, valid)
    assert torch.isfinite(out["action_mu"]).all()


def test_temporal_current_invalid_is_zero():
    attn = TemporalAttention(H, 4)
    tokens = torch.randn(1, 3, 2, H)
    valid = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]])
    ids = torch.tensor([[[1, 2], [1, 2], [1, 2]]])
    out = attn(tokens, valid, ids)
    assert torch.allclose(out, torch.zeros_like(out))


def test_wm_residual_and_direct_geometry():
    wm = EgoConditionedLatentWM(hidden=H)
    od = torch.randn(B, 16, H)
    ld = torch.randn(B, 16, H)
    action = torch.rand(B, 4)
    out = wm(od, ld, action, od_base=od, ld_base=ld)
    assert out["od_residual"].shape == (B, 16, H)
    assert out["ld_residual"].shape == (B, 16, H)
    assert out["od_geometry"].shape == (B, 16, 5)
    assert out["ld_geometry"].shape == (B, 16, 10)
    # zero-init residual -> persistence at initialization
    assert torch.allclose(out["od_residual"], torch.zeros_like(od))
    assert torch.allclose(out["od"], od)
    loss = out["od_geometry"].pow(2).mean() + out["ld_geometry"].pow(2).mean()
    loss.backward()


def test_build_and_forward_current():
    model = build_net()
    out = model.forward_current(_batch())
    assert out["action_mu"].shape == (B, 4)
    assert out["action_logstd"].shape == (B, 4)
    assert set(out["aux"]) == {"collision", "offroad", "timeout", "urgency", "progress"}


def test_imagine_six_transitions_and_no_real_writeback():
    model = _model()
    batch = _batch(b=1)
    enc = model.encoder.encode_current(batch)
    window = _window(enc)
    real_bank = RealMemoryBank(6)
    real_bank.append(model.encoder.frame_records(batch, episode_id="ep", frame_id=0, timestamp=0.0)[0])
    imagine_bank = real_bank.snapshot()
    out = model.imagine(
        window,
        enc["nav"],
        enc["ego"],
        actions=torch.rand(1, 6, 4),
        ego_features=batch["ego"],
        imagine_bank=imagine_bank,
        horizon=6,
    )
    assert len(out) == 6
    assert all(p["action_mu"].shape == (1, 4) for p in out)
    # six imagined frames, depth 1..6
    imagined = [f for f in imagine_bank.frames if f.source == Source.IMAGINED]
    assert len(imagined) == 6
    assert [f.rollout_depth for f in imagined] == [1, 2, 3, 4, 5, 6]
    assert len(real_bank.frames) == 1


def test_action_and_ego_step_helpers():
    model = _model()
    action = torch.rand(1, 4)
    requested = model._to_requested(action)
    assert requested.shape == (1, 4)
    ego = torch.randn(1, 12)
    pose = torch.zeros(1, 3)
    next_pose, next_ego = model._ego_step(ego, requested, pose)
    assert next_pose.shape == (1, 3)
    assert next_ego.shape == (1, 12)
    assert torch.isfinite(next_pose).all() and torch.isfinite(next_ego).all()


def test_stable_id_reorders_history_into_current_slot():
    def od(slot, sid, x):
        return EntityRecord(slot, sid, NodeType.OD, torch.ones(H), torch.tensor([x, 0.0, 0.0, 0.0, 0.0]))

    f0 = FrameRecord("ep", 0, 0.0, torch.tensor([0.0, 0.0, 0.0]), (od(0, 7, 10.0),), ())
    f1 = FrameRecord("ep", 1, 0.5, torch.tensor([2.0, 0.0, 0.0]), (od(1, 7, 8.0),), ())
    window = BankWindowBuilder(hidden=H).build((f0, f1))
    assert bool(window.od_valid[0, -1, 1])
    assert bool(window.od_valid[0, -2, 1])
    assert float(window.od_geometry[0, -2, 1, 0]) == 10.0


def test_invalid_nan_slots_are_sanitized():
    attn = TemporalAttention(H, 4)
    tokens = torch.randn(1, 3, 2, H)
    tokens[0, 0, 0] = float("nan")
    tokens[0, -1, 0] = float("nan")
    valid = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]])
    ids = torch.tensor([[[1, 2], [1, 2], [1, 2]]])
    out = attn(tokens, valid, ids)
    assert torch.isfinite(out).all()


def test_imagine_requires_ego_features_and_normalized_actions():
    model = _model()
    batch = _batch(b=1)
    enc = model.encoder.encode_current(batch)
    window = _window(enc)
    with pytest.raises(ValueError):
        model.imagine(window, enc["nav"], enc["ego"], actions=torch.rand(1, 6, 4))
    with pytest.raises(ValueError):
        model.imagine(
            window, enc["nav"], enc["ego"],
            actions=torch.full((1, 6, 4), 2.0),
            ego_features=batch["ego"],
        )


def test_build_net_requires_action_bounds():
    with pytest.raises(ValueError, match="action.bounds"):
        build_net({"model": {"hidden": H, "attention_heads": 4}})


def test_builder_rejects_hidden_mismatch():
    entity = EntityRecord(0, 1, NodeType.OD, torch.ones(H), torch.zeros(5))
    frame = FrameRecord("ep", 0, 0.0, torch.zeros(3), (entity,), tuple())
    with pytest.raises(ValueError, match="hidden"):
        BankWindowBuilder(hidden=H + 1).build((frame,))


def test_frame_records_support_per_sample_metadata():
    model = _model()
    batch = _batch(b=2)
    batch["episode_id"] = torch.tensor([11, 12])
    batch["frame_id"] = torch.tensor([3, 4])
    batch["timestamp"] = torch.tensor([1.0, 1.5])
    enc = model.encoder.encode_current(batch)
    records = model.encoder.frame_records(batch, episode_id=batch["episode_id"], frame_id=batch["frame_id"], timestamp=batch["timestamp"], encoded=enc)
    assert [r.episode_id for r in records] == ["11", "12"]
    assert [r.frame_id for r in records] == [3, 4]
    assert [r.timestamp for r in records] == [1.0, 1.5]


def test_imagine_bank_rejects_mismatched_window():
    model = _model()
    batch = _batch(b=1)
    enc = model.encoder.encode_current(batch)
    window = _window(enc)
    real_bank = RealMemoryBank(6)
    real_bank.append(model.encoder.frame_records(batch, episode_id="ep", frame_id=0, timestamp=0.0)[0])
    bank = real_bank.snapshot()
    bad = window
    bad.ego_pose = bad.ego_pose.clone()
    bad.ego_pose += 1.0
    with pytest.raises(ValueError, match="ego_pose"):
        model.imagine(window=bad, nav=enc["nav"], ego=enc["ego"], imagine_bank=bank, ego_features=batch["ego"])


def test_imagine_batch_gt_one_depth():
    model = _model()
    batch = _batch(b=2)
    enc = model.encoder.encode_current(batch)
    records = model.encoder.frame_records(batch, episode_id="ep", frame_id=0, timestamp=0.0, encoded=enc)
    window = BankWindow.stack([BankWindowBuilder(hidden=H).build((r,)) for r in records])
    out = model.imagine(
        window, enc["nav"], enc["ego"],
        actions=torch.rand(2, 6, 4),
        ego_features=batch["ego"],
        horizon=6,
    )
    assert len(out) == 6
    assert all(p["action_mu"].shape == (2, 4) for p in out)


def test_forward_history_respects_hist_valid():
    model = _model()
    base = _batch(b=1)
    history = {}
    for key, value in base.items():
        if torch.is_tensor(value) and value.dim() >= 2:
            history[key] = value[:, None].expand(-1, 6, *value.shape[1:]).clone()
        else:
            history[key] = value
    history["od"] = history["od"] + torch.arange(6).view(1, 6, 1, 1)
    history["hist_valid"] = torch.ones(1, 6)
    full = model.forward_history(history)["action_mu"]
    history["hist_valid"] = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    current_only = model.forward_history(history)["action_mu"]
    assert not torch.allclose(full, current_only)
