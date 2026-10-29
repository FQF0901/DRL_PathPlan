"""Tests for the direct-decoder latent WM and temporal-spatial encoder."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from net.mem_bank import BankWindowBuilder, EntityRecord, FrameRecord, NodeType, Source
from net.world_model import EgoConditionedLatentWM, TemporalSpatialBlock
from pipeline.losses import masked_mse


def _wm_batch(b=2, n=4, h=16):
    return {
        "od": torch.randn(b, n, h),
        "ld": torch.randn(b, n, h),
        "action": torch.rand(b, 4),
        "od_valid": torch.ones(b, n),
        "ld_valid": torch.ones(b, n),
    }


def _window(h=16, t=2):
    od = (EntityRecord(0, 7, NodeType.OD, torch.ones(h), torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0])),)
    ld = (EntityRecord(0, 8, NodeType.LD, torch.ones(h), torch.zeros(10)),)
    frames = [
        FrameRecord("ep", i, float(i) * 0.5, torch.tensor([float(i), 0.0, 0.0]), od, ld)
        for i in range(t)
    ]
    return BankWindowBuilder(hidden=h).build(frames)


def test_wm_shapes_residual_and_gradient():
    wm = EgoConditionedLatentWM(hidden=16)
    b = _wm_batch()
    out = wm(b["od"], b["ld"], b["action"], od_base=b["od"], ld_base=b["ld"], od_valid=b["od_valid"], ld_valid=b["ld_valid"])
    assert out["od"].shape == (2, 4, 16)
    assert out["ld"].shape == (2, 4, 16)
    assert out["od_residual"].shape == (2, 4, 16)
    assert out["od_geometry"].shape == (2, 4, 5)
    assert out["ld_geometry"].shape == (2, 4, 10)
    # Residuals are zero-initialized.
    assert torch.allclose(out["od_residual"], torch.zeros_like(out["od"]))
    loss = out["od_geometry"].pow(2).mean() + out["ld_geometry"].pow(2).mean()
    loss.backward()
    assert wm.od_geometry.net[0].weight.grad is not None


def test_tsb_uses_frame_delta_without_geometry_projection():
    block = TemporalSpatialBlock(hidden=16, heads=4, spatial_layers=1)
    win = _window(t=2)
    ego = torch.randn(1, 16)
    out = block(win, ego)
    assert out["od"].shape == (1, 16, 16)
    assert out["ld"].shape == (1, 16, 16)
    # No nav token exists in the spatial input.
    assert out["tokens"].shape == (1, 1 + 16 + 16, 16)


def test_masked_mse_is_feature_mean_then_entity_mean():
    p = torch.zeros(2, 4, 8)
    t = torch.ones(2, 4, 8) * 10
    m = torch.ones(2, 4)
    assert abs(float(masked_mse(p, t, m)) - 100.0) < 1e-6
    m3 = torch.ones(2, 4, 1)
    assert abs(float(masked_mse(p, t, m3)) - 100.0) < 1e-6


def test_wm_sanitizes_nan_in_invalid_slots():
    wm = EgoConditionedLatentWM(hidden=8)
    od = torch.randn(2, 4, 8)
    ld = torch.randn(2, 4, 8)
    od[0, 1] = float("nan")
    ld[1, 2] = float("nan")
    valid = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 1]], dtype=torch.bool)
    out = wm(od, ld, torch.rand(2, 4), od_base=od, ld_base=ld, od_valid=valid, ld_valid=valid)
    assert torch.isfinite(out["od"]).all()
    assert torch.isfinite(out["ld"]).all()
    assert torch.isfinite(out["od_geometry"]).all()
    assert torch.isfinite(out["ld_geometry"]).all()


def test_masked_mse_ignores_nan_in_masked_slots():
    pred = torch.zeros(1, 2, 4)
    target = torch.ones(1, 2, 4)
    target[0, 1] = float("nan")
    mask = torch.tensor([[1.0, 0.0]])
    assert float(masked_mse(pred, target, mask)) == pytest.approx(1.0)


def test_align_by_id_zero_ids_falls_back_to_slots():
    from pipeline.train_wm import _align_by_id, _align_geometry_by_id
    cur = np.arange(8, dtype=np.float32).reshape(2, 4)
    tgt = cur + 10
    ids = np.zeros(2, dtype=np.int64)
    mask = np.ones(2, dtype=np.float32)
    out, om = _align_by_id(cur, ids, mask, tgt, ids, mask)
    assert np.allclose(out, tgt)
    assert om.tolist() == [1.0, 1.0]
    geom = np.arange(8, dtype=np.float32).reshape(2, 4)
    gout, gm = _align_geometry_by_id(ids, mask, geom, ids, mask)
    assert np.allclose(gout, geom)
    assert gm.tolist() == [1.0, 1.0]


def test_align_by_id_single_nonzero_id_uses_id_not_slot():
    from pipeline.train_wm import _align_by_id
    cur = np.array([[1.0], [2.0]], dtype=np.float32)
    tgt = np.array([[999.0], [42.0]], dtype=np.float32)
    out, mask = _align_by_id(
        cur, np.array([42, 0], dtype=np.int64), np.array([1.0, 0.0], dtype=np.float32),
        tgt, np.array([7, 42], dtype=np.int64), np.array([1.0, 1.0], dtype=np.float32),
    )
    assert out[0, 0] == 42.0 and mask[0] == 1.0
