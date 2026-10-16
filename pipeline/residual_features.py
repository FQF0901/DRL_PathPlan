#!/usr/bin/env python3
"""残差训练链（R41 第一批-B 入库）：**特征/目标构建（train ↔ eval 同口径）**。

规格出处
--------
- R41 复审（``docs/deepseekv4p1_argue.md`` §2.3 / §三 第一批②）：`tools/train_residual.py`
  的残差训练链完全入库；
- 本模块 = R40-V5 runner 内环的**逐位迁移**（语义不变；对拍证据 = ``round41/b_bitwise.json``，
  max_abs_diff=0）：obs → ``model.encode`` → ``WorldState.t0`` → ``readout``（prev）→
  ``analytic_update(ds, dθ, step=1)`` → ``readout``（base）→ ``readout_residual_features``
  （= 评测侧 ``net.model`` 想象链同一函数，**单一出处**）；
- 特征键序/宽度 = ``net.world_residual.readout_residual_features`` 返回键（合计 26 =
  8+8+2+4+4）；目标列 = ``READOUT_RESIDUAL_INDICES`` = (0,3,4,5)（speed/static_distance/
  dynamic_distance/dynamic_ttc）；目标 ``Δ* = real_next − base``，掩码 = ``base_present ∧
  real_present``。

口径约束
--------
- **禁自产状态回灌**：特征/目标只来自真实行 encode 与数据集真实列；不闭环、不迭代自产链；
- ΔA_θ 的调用口径 = 5 键关键字调用（``head_delta``）；训练与评测消费同一 26 列切分
  （:func:`tensor_to_features`），逐 chunk 逐位（TF32 GEMM tiling ⇒ 跨 batch 拼接非同源口径）。

本模块**零研究工作区依赖**：只 import torch 与仓库内 ``net``。
"""

from __future__ import annotations

import torch

from net.world_residual import (
    READOUT_RESIDUAL_FIELDS,
    READOUT_RESIDUAL_INDICES,
    readout_residual_features,
)
from net.world_state import analytic_update, readout, world_state_from_encode

#: 特征键序（= ``readout_residual_features`` 返回键；列序敏感：贯穿特征/损失/账本）
FEATURE_KEYS: tuple[str, ...] = ("prev_exog", "prev_present", "action", "base_exog", "base_present")
#: 各特征键宽度（26 列拼接口径）
FEATURE_WIDTHS: tuple[int, ...] = (8, 8, 2, 4, 4)
#: 目标列（readout 列序中的列号；= ``READOUT_RESIDUAL_INDICES``）
TARGET_COLUMNS: tuple[int, ...] = tuple(int(index) for index in READOUT_RESIDUAL_INDICES)
#: 目标列名（= ``READOUT_RESIDUAL_FIELDS``）
COLUMN_NAMES: tuple[str, ...] = tuple(str(name) for name in READOUT_RESIDUAL_FIELDS)


def transition_features(model, obs) -> dict:
    """obs → one-step transition 特征（5 键）+ base 目标列（``no_grad`` 由调用方控制）。

    逐位口径 = 评测侧想象链的 k=1 调用：``encode`` → ``world_state_from_encode``（t0 真实
    encode）→ ``readout`` → ``analytic_update``（生产结构化头在行 t 的 ``(ds,dθ)``）→
    ``readout`` → ``readout_residual_features``。

    Returns: ``{"features": {5 键}, "feature_tensor": (B,26), "base_cols": (B,4),
    "base_present": (B,4), "action": (B,2)}``。
    """
    encoded = model.encode(obs)
    state = world_state_from_encode(
        model.st_gnn, encoded, dt=model.dt, speed_alpha=1.0, speed_max=20.0
    )
    prev_view = readout(state)
    action = encoded["path"]["action"]
    analytic_update(
        state, ds=action[:, 0], dtheta=action[:, 1], pose=None,
        z_od=encoded["z_od"], z_ld=encoded["z_ld"], step=1,
    )
    base_view = readout(state)
    features = readout_residual_features(prev_view=prev_view, action=action, base_view=base_view)
    index = list(READOUT_RESIDUAL_INDICES)
    return {
        "features": features,
        "feature_tensor": features_to_tensor(features),
        "base_cols": base_view["long_exog"][:, index],
        "base_present": base_view["long_present"][:, index],
        "action": action,
    }


def features_to_tensor(features) -> torch.Tensor:
    """5 键字典 → ``(B,26)``（列序 = :data:`FEATURE_KEYS`/:data:`FEATURE_WIDTHS`）。"""
    tensor = torch.cat([features[key] for key in FEATURE_KEYS], dim=-1)
    if int(tensor.shape[-1]) != sum(FEATURE_WIDTHS):
        raise SystemExit(
            f"[residual] fail-fast：特征宽度 {int(tensor.shape[-1])} != {sum(FEATURE_WIDTHS)}"
        )
    return tensor


def tensor_to_features(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """``(B,26)`` → 5 键字典（训练/评测消费同一 26 列切分）。"""
    if int(raw.shape[-1]) != sum(FEATURE_WIDTHS):
        raise SystemExit(
            f"[residual] fail-fast：特征宽度 {int(raw.shape[-1])} != {sum(FEATURE_WIDTHS)}"
        )
    out: dict[str, torch.Tensor] = {}
    offset = 0
    for key, width in zip(FEATURE_KEYS, FEATURE_WIDTHS):
        out[key] = raw[..., offset : offset + width]
        offset += width
    return out


def head_delta(head, raw: torch.Tensor) -> torch.Tensor:
    """``(B,26)`` → ΔA_θ（5 键关键字调用口径；训练/评测共享）。"""
    return head(**tensor_to_features(raw))


def residual_targets(
    *,
    base_cols: torch.Tensor,
    base_present: torch.Tensor,
    real_cols: torch.Tensor,
    real_present: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """one-step 目标 ``Δ* = (real_next − base) · mask``；``mask = (base_present>0.5) ∧ (real_present>0.5)``。"""
    mask = ((base_present > 0.5) & (real_present > 0.5)).to(torch.float32)
    target = (real_cols - base_cols) * mask
    return target, mask


__all__ = [
    "COLUMN_NAMES",
    "FEATURE_KEYS",
    "FEATURE_WIDTHS",
    "TARGET_COLUMNS",
    "features_to_tensor",
    "head_delta",
    "residual_targets",
    "tensor_to_features",
    "transition_features",
]
