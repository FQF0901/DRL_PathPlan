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
    wm = EgoConditionedLatentWM(hidden=16, od_slots=4, ld_slots=4, layers=1, heads=2)
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
    wm = EgoConditionedLatentWM(hidden=16, od_slots=4, ld_slots=4, layers=1, heads=2)
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
