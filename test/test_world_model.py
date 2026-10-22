"""Stage D1 latent world-model shape/gradient tests."""

from __future__ import annotations

import torch

from net.world_model import EgoConditionedLatentWM, masked_mse, rollout_latents


def _batch(b=2, n=4, h=16):
    return {
        "od": torch.randn(b, n, h),
        "ld": torch.randn(b, n, h),
        "nav": torch.randn(b, h),
        "ego": torch.randn(b, h),
        "od_mask": torch.ones(b, n),
        "ld_mask": torch.ones(b, n),
        "action": torch.randn(b, 4),
    }


def test_wm_shapes_and_gradient():
    wm = EgoConditionedLatentWM(hidden=16, latent_dim=16, od_slots=4, ld_slots=4, layers=1, heads=2)
    b = _batch()
    out = wm(**b)
    assert out["od"].shape == (2, 4, 16)
    assert out["ld"].shape == (2, 4, 16)
    assert out["nav"].shape == (2, 16)
    assert masked_mse(out["od"], b["od"], b["od_mask"]).item() >= 0.0
    loss = sum(v.pow(2).mean() for v in out.values())
    loss.backward()
    assert wm.od_head.weight.grad is not None


def test_wm_rollout_history():
    wm = EgoConditionedLatentWM(hidden=16, latent_dim=16, od_slots=4, ld_slots=4, layers=1, heads=2)
    b = _batch()
    initial = {k: b[k] for k in ("od", "ld", "nav", "ego")}
    state, history = rollout_latents(
        wm,
        initial,
        torch.randn(2, 3, 4),
        {"od": b["od_mask"], "ld": b["ld_mask"]},
    )
    assert len(history) == 3
    assert state["od"].shape == (2, 4, 16)


def test_persistence_uses_true_per_element_mse():
    import numpy as np
    from pipeline.train_wm import _persistence
    a = np.zeros((2, 4), dtype=np.float32)
    b = np.ones((2, 4), dtype=np.float32)
    arrays = {
        "od": np.zeros((2, 1, 4), dtype=np.float32),
        "od_target": np.ones((2, 1, 4), dtype=np.float32),
        "od_tmask": np.ones((2, 1), dtype=np.float32),
        "ld": np.zeros((2, 1, 4), dtype=np.float32),
        "ld_target": np.ones((2, 1, 4), dtype=np.float32),
        "ld_tmask": np.ones((2, 1), dtype=np.float32),
        "nav": a, "nav_target": b,
        "ego": a, "ego_target": b,
    }
    out = _persistence(arrays, np.arange(2))
    assert abs(out["nav"] - 1.0) < 1e-6
    assert abs(out["ego"] - 1.0) < 1e-6
