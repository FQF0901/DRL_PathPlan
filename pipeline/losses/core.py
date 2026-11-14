"""Pure loss functions for the bank-native policy / world model.

This module deliberately has no model, env, Bank, or training-loop imports.
Invalid slots may contain NaN; every masked reduction must remove NaN *before*
multiplying by a zero mask.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import Tensor


def _normalize_mask(mask: Tensor, target: Tensor) -> Tensor:
    """Broadcast/normalize an entity mask to ``target``'s leading shape."""
    mask = mask.to(dtype=torch.bool)
    if mask.shape == target.shape:
        return mask
    if mask.shape == target.shape[:-1] + (1,):
        return mask[..., 0]
    if mask.shape == target.shape[:-1]:
        return mask
    # Allow [B,N] masks for [B,N,G] geometry targets.
    if mask.dim() == target.dim() - 1:
        return mask
    raise ValueError(f"mask shape {tuple(mask.shape)} incompatible with {tuple(target.shape)}")


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """NaN-safe MSE.

    ``pred``/``target`` are ``[..., N, H]`` (or any trailing feature shape);
    ``mask`` is the corresponding ``[..., N]`` / ``[..., N, 1]`` mask.
    The feature dimensions are reduced with mean, then valid entities are
    averaged.  Masked-out values are replaced with zero before the reduction.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target {tuple(target.shape)}")
    valid = _normalize_mask(mask, target).to(device=target.device)
    sq = (pred - target).pow(2)
    valid = valid.unsqueeze(-1) if valid.dim() == sq.dim() - 1 else valid
    valid = valid.expand_as(sq)
    sq = torch.where(valid, sq, torch.zeros_like(sq))
    per_entity = sq.mean(dim=-1)
    m = valid[..., 0].to(per_entity.dtype)
    return (per_entity * m).sum() / m.sum().clamp_min(1.0)


def masked_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """NaN-safe L1 analogue of :func:`masked_mse`."""
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != target {tuple(target.shape)}")
    valid = _normalize_mask(mask, target).to(device=target.device)
    err = (pred - target).abs()
    valid = valid.unsqueeze(-1) if valid.dim() == err.dim() - 1 else valid
    valid = valid.expand_as(err)
    err = torch.where(valid, err, torch.zeros_like(err))
    per_entity = err.mean(dim=-1)
    m = valid[..., 0].to(per_entity.dtype)
    return (per_entity * m).sum() / m.sum().clamp_min(1.0)


def wm_losses(
    pred: Mapping[str, Tensor],
    *,
    od_target: Tensor,
    ld_target: Tensor,
    od_valid: Tensor,
    ld_valid: Tensor,
    od_geometry_target: Optional[Tensor] = None,
    ld_geometry_target: Optional[Tensor] = None,
    od_geometry_mask: Optional[Tensor] = None,
    ld_geometry_mask: Optional[Tensor] = None,
    content_weight: float = 1.0,
    od_geometry_weight: float = 1.0,
    ld_geometry_weight: float = 1.0,
) -> Dict[str, Tensor]:
    """One-step latent + direct-geometry losses for the bank-native WM.

    ``pred`` is the WM output dict (``od``, ``ld``, ``od_geometry``,
    ``ld_geometry``).  Targets/valid masks are current-frame, slot-aligned.
    """
    out = {
        "od_content": float(content_weight) * masked_mse(pred["od"], od_target, od_valid),
        "ld_content": float(content_weight) * masked_mse(pred["ld"], ld_target, ld_valid),
    }
    if od_geometry_target is not None:
        out["od_geometry"] = float(od_geometry_weight) * masked_l1(
            pred["od_geometry"], od_geometry_target,
            od_valid if od_geometry_mask is None else od_geometry_mask,
        )
    if ld_geometry_target is not None:
        out["ld_geometry"] = float(ld_geometry_weight) * masked_l1(
            pred["ld_geometry"], ld_geometry_target,
            ld_valid if ld_geometry_mask is None else ld_geometry_mask,
        )
    out["total"] = sum(out.values())
    return out


__all__ = ["masked_mse", "masked_l1", "wm_losses"]
