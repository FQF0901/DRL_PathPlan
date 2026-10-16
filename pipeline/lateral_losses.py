#!/usr/bin/env python3
"""R41 第二批-F：**横向训练栈 / loss 侧（默认关；全项权重可配）**。

规格出处
--------
- R41 复审（``docs/deepseekv4p1_argue.md`` §2）：第二批 P0–P4 臂 + ``L_replan = w(ΔS)·
  ‖κ_t − Warp(κ_{t−1})‖`` **门控版**（安全边界收缩/障碍变化 ⇒ 降权）+ 批次重加权钩子；
- 本文实现四项**新**损失项 + 组合权重钩子（既有 ``L_plan`` = ``trainer.weighted_action_chain_loss``
  ``(ds,dθ)`` 空间链损失与既有曲率正则 ``trainer.plan_curvature_loss`` **保持不动**，由上层组装）：

  1. :func:`kappa_supervision_loss` —— **κ 监督**（站位空间 κ_near/mid/far vs 专家 κ 插值；
     掩码 = 专家站有效 ∧ 锚视界（见 :mod:`pipeline.lateral_labels`））；P1 起启用；
  2. :func:`plan_delta_kappa_loss` —— **L_Δκ（计划内平滑）**：相邻步 ``|Δκ|`` hinge，配对掩码；
     默认权重 0（机制备件；启用需新预注册）；
  3. :func:`replan_consistency_loss` —— **L_replan**：``w(ΔS)·g_safety·g_align·
     ‖κ_t − Warp(κ_{t−1})‖``（warp 索引/权重 = :mod:`pipeline.lateral_replan` 数据侧；
     门控在权重里，本函数只做加权差项，**无梯度进权重**）；P3 起启用；
  4. :func:`combine_sample_weights` —— **批次重加权钩子**：逐样本权重乘积 + clip + 有效行
     归一化（P2 速度桶 / P4 LQR 差距权重由其组合；权重张量必须 detach/非梯度）。

Arm 预置（P0–P4 = 增量堆叠；数值为**预注册拟值**，冻结前可调，见 ``f_prereg.md``）
-------------------------------------------------------------------------------------
======== ================================================================
arm      增量
======== ================================================================
p0       全 0（现训练基线；新项不触达）
p1       ``kappa_coef=0.5``（κ_near/mid/far 站监督，恒权）
p2       p1 + 速度分桶权重（``speed_weight_mode="bucket"``）
p3       p2 + ``replan_coef=0.5``（门控 L_replan）
p4       p3 + LQR 执行差距重加权（``lqr_reweight=True``，`w = w_base·(1+α·w_lqr)`）
======== ================================================================

默认全 0 = 逐位不变：系数 0 的项**不计算**（返回零张量、不建图）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from pipeline.lateral_labels import (
    SPEED_BUCKET_EDGES,
    SPEED_BUCKET_WEIGHTS_DEFAULT,
)

#: L_Δκ hinge 阈值（1/m；= ``trainer.curv_dk_thresh`` 默认，单一出处对齐）
DK_THRESH_DEFAULT = 0.1
#: smooth_l1 beta（与 ``column_residual_loss``/J1 ``_smooth_l1`` 口径一致）
BETA_DEFAULT = 1.0


# --------------------------------------------------------------------------- #
# 0) 基础：掩码加权 smooth_l1
# --------------------------------------------------------------------------- #
def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    beta: float = BETA_DEFAULT,
) -> dict:
    """掩码 + 逐样本/逐步权重 smooth_l1（单一基础项；所有 κ 项共用）。

    ``pred/target (B,S)``、``mask (B,S)``（>0.5 有效）；``sample_weights (B,)``、
    ``step_weights (S,)``（均须非梯度；缺省 = 全 1）。返回 ``{"loss", "n_valid"}``：
    ``loss = Σ w_b·w_s·m·smooth_l1 / Σ w_b·w_s·m``（分母 0 ⇒ loss 0）。
    """
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise SystemExit(
            f"[lateral] fail-fast：形状不一致 pred {tuple(pred.shape)} / target {tuple(target.shape)}"
            f" / mask {tuple(mask.shape)}"
        )
    valid = (mask > 0.5).to(pred.dtype)
    weight = valid
    if sample_weights is not None:
        weight = weight * sample_weights.to(pred.dtype).reshape(-1, 1)
    if step_weights is not None:
        weight = weight * step_weights.to(pred.dtype).reshape(1, -1)
    element = F.smooth_l1_loss(pred, target, beta=float(beta), reduction="none")
    denominator = weight.sum()
    if float(denominator.detach()) <= 0.0:
        return {"loss": torch.zeros((), device=pred.device, dtype=pred.dtype), "n_valid": 0.0}
    return {"loss": (element * weight).sum() / denominator, "n_valid": float(valid.sum().item())}


# --------------------------------------------------------------------------- #
# 1) κ 监督（站位空间；P1）
# --------------------------------------------------------------------------- #
def kappa_supervision_loss(
    kappa_pred: torch.Tensor,
    kappa_target: torch.Tensor,
    mask: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    step_weights: torch.Tensor | None = None,
    beta: float = BETA_DEFAULT,
) -> dict:
    """κ 站位监督（``kappa_pred/target/mask (B,S)``；S=3 = near/mid/far）。

    κ 空间理由/掩码口径见 :mod:`pipeline.lateral_labels` 模块 docstring。
    """
    return masked_smooth_l1(
        kappa_pred,
        kappa_target,
        mask,
        sample_weights=sample_weights,
        step_weights=step_weights,
        beta=beta,
    )


def plan_kappa_from_plan(
    plan: torch.Tensor,
    *,
    ds_min: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预测计划 ``(B,K,2)`` → 步段 κ ``(B,K)`` + 掩码 ``(B,K)``（``κ = dθ/ds``）。

    与数据侧 :func:`pipeline.lateral_labels.plan_targets_from_action` 同口径（``ds<ds_min``
    或非有限 ⇒ κ=0、mask=0）。
    """
    if plan.ndim != 3 or int(plan.shape[2]) != 2:
        raise SystemExit(f"[lateral] fail-fast：plan 形状应为 (B,K,2)，收到 {tuple(plan.shape)}")
    ds = plan[..., 0]
    dtheta = plan[..., 1]
    valid = torch.isfinite(ds) & torch.isfinite(dtheta) & (ds >= float(ds_min))
    kappa = torch.where(valid, dtheta / ds.clamp_min(float(ds_min)), torch.zeros_like(ds))
    return kappa, valid.to(torch.float32)


def plan_delta_kappa_loss(
    kappa: torch.Tensor,
    mask: torch.Tensor,
    *,
    dk_thresh: float = DK_THRESH_DEFAULT,
    sample_weights: torch.Tensor | None = None,
) -> dict:
    """L_Δκ：``relu(|κ_k − κ_{k−1}| − dk_thresh)`` 的掩码加权均值（配对掩码 = 双步有效）。

    口径与既有 ``pipeline.trainer.plan_curvature_loss`` 的 Δκ 项一致（本函数只含
    平滑项、不含大 κ 幅值项；幅值项继续由既有 ``plan_curvature_loss`` 承担）。
    """
    if kappa.shape != mask.shape or kappa.ndim != 2:
        raise SystemExit(
            f"[lateral] fail-fast：kappa/mask 形状应为同形 (B,K)，收到 {tuple(kappa.shape)}/{tuple(mask.shape)}"
        )
    valid = mask > 0.5
    pair_valid = valid[:, 1:] & valid[:, :-1]
    delta = (kappa[:, 1:] - kappa[:, :-1]).abs()
    hinge = torch.relu(delta - float(dk_thresh))
    weight = pair_valid.to(kappa.dtype)
    if sample_weights is not None:
        weight = weight * sample_weights.to(kappa.dtype).reshape(-1, 1)
    denominator = weight.sum()
    if float(denominator.detach()) <= 0.0:
        return {"loss": torch.zeros((), device=kappa.device, dtype=kappa.dtype), "n_valid": 0.0}
    return {"loss": (hinge * weight).sum() / denominator, "n_valid": float(pair_valid.sum().item())}


# --------------------------------------------------------------------------- #
# 2) L_replan（跨帧一致性；P3）
# --------------------------------------------------------------------------- #
def replan_consistency_loss(
    kappa_pred_t: torch.Tensor,
    kappa_pred_prev: torch.Tensor,
    warp_index: torch.Tensor,
    warp_mask: torch.Tensor,
    weight: torch.Tensor,
    *,
    mode: str = "l1",
    beta: float = BETA_DEFAULT,
) -> dict:
    """``w·‖κ_t − Warp(κ_{t−1})‖``（门控权重已含数据侧；权重须非梯度）。

    - ``kappa_pred_t (B,K)``：当前帧计划步段 κ（:func:`plan_kappa_from_plan`）；
    - ``kappa_pred_prev (B,K)``：上一帧计划步段 κ（同帧模型**同权重**前向产物；可微）；
    - ``warp_index (B,K) int64``（-1 = 无效）、``warp_mask (B,K) float``、``weight (B,) float``：
      :mod:`pipeline.lateral_replan` 数据侧产出；
    - ``Warp = gather(kappa_pred_prev, warp_index)``（可微；无效列由 mask 剔除）。
    """
    if mode not in ("l1", "smooth_l1"):
        raise SystemExit(f"[lateral] fail-fast：未知 mode {mode!r}（l1 | smooth_l1）")
    index = warp_index.to(kappa_pred_t.device).long().clamp_min(0)
    if index.shape != kappa_pred_t.shape:
        raise SystemExit(
            f"[lateral] fail-fast：warp_index 形状 {tuple(index.shape)} != κ_t 形状 {tuple(kappa_pred_t.shape)}"
        )
    warped = torch.gather(kappa_pred_prev.to(kappa_pred_t.device), 1, index)
    if mode == "l1":
        element = (kappa_pred_t - warped).abs()
    else:
        element = F.smooth_l1_loss(kappa_pred_t, warped, beta=float(beta), reduction="none")
    valid = warp_mask.to(kappa_pred_t.dtype) > 0.5
    weight_matrix = valid * weight.to(kappa_pred_t.dtype).reshape(-1, 1)
    denominator = weight_matrix.sum()
    if float(denominator.detach()) <= 0.0:
        return {"loss": torch.zeros((), device=kappa_pred_t.device, dtype=kappa_pred_t.dtype), "n_valid": 0.0}
    return {
        "loss": (element * weight_matrix).sum() / denominator,
        "n_valid": float(valid.sum().item()),
        "mean_abs_unweighted": float(((kappa_pred_t - warped).abs() * valid).sum().detach())
        / max(1.0, float(valid.sum().item())),
    }


# --------------------------------------------------------------------------- #
# 3) 批次重加权钩子（P2 速度桶 / P4 LQR 差距）
# --------------------------------------------------------------------------- #
def combine_sample_weights(
    parts: Sequence[torch.Tensor],
    *,
    cap: float | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """逐样本权重乘积 + （可选）caps + 有效行归一化到均值 1。

    - 输入各 ``(B,)``（**非梯度**；调用方保证 detach；本函数 `.detach()` 兜底）；
    - ``cap``：逐元素 clip 上界（None = 不 clip）；
    - ``normalize``：对 ``w>0`` 的行归一化到均值 1（保持有效学习率尺度）。
    """
    if not parts:
        raise SystemExit("[lateral] fail-fast：combine_sample_weights 需要至少一个权重分量")
    combined: torch.Tensor | None = None
    for part in parts:
        value = part.detach().to(torch.float32).reshape(-1)
        combined = value if combined is None else combined * value
    assert combined is not None
    if cap is not None:
        combined = combined.clamp(max=float(cap))
    if normalize:
        positive = combined > 0.0
        if bool(positive.any()):
            mean = combined[positive].mean()
            if float(mean) > 0.0:
                combined = torch.where(positive, combined / mean, combined)
    return combined


# --------------------------------------------------------------------------- #
# 4) 组装（arm 预置 + 总项；系数 0 ⇒ 不计算）
# --------------------------------------------------------------------------- #
@dataclass
class LateralLossConfig:
    """横向损失旋钮（默认全关；arm 预置见 :data:`ARM_PRESETS`）。"""

    kappa_coef: float = 0.0
    delta_kappa_coef: float = 0.0
    replan_coef: float = 0.0
    beta: float = BETA_DEFAULT
    dk_thresh: float = DK_THRESH_DEFAULT
    #: P2：速度分桶权重模式（``off`` | ``bucket``）
    speed_weight_mode: str = "off"
    speed_bucket_edges: tuple[float, ...] = SPEED_BUCKET_EDGES
    speed_bucket_weights: tuple[float, ...] = SPEED_BUCKET_WEIGHTS_DEFAULT
    #: L_replan 数据侧尺度（m；w_ds = exp(−ΔS/s_ref)）
    replan_ds_ref_m: float = 5.0
    #: P4：LQR 执行差距重加权（w = w_base·(1 + alpha·w_lqr)）
    lqr_reweight: bool = False
    lqr_alpha: float = 1.0
    lqr_clip_max: float = 1.0
    #: 权重乘积 cap（None = 不 cap）
    weight_cap: float | None = 10.0
    extra: Mapping[str, float] = field(default_factory=dict)

    def active(self) -> tuple[str, ...]:
        """启用项清单（诊断/预注册核对）。"""
        names = []
        if self.kappa_coef:
            names.append("kappa")
        if self.delta_kappa_coef:
            names.append("delta_kappa")
        if self.replan_coef:
            names.append("replan")
        if self.speed_weight_mode != "off":
            names.append("speed_bucket")
        if self.lqr_reweight:
            names.append("lqr_reweight")
        return tuple(names)


#: P0–P4 arm 预置（增量堆叠；数值 = 预注册拟值）
ARM_PRESETS: dict[str, dict] = {
    "p0": {},
    "p1": {"kappa_coef": 0.5},
    "p2": {"kappa_coef": 0.5, "speed_weight_mode": "bucket"},
    "p3": {"kappa_coef": 0.5, "speed_weight_mode": "bucket", "replan_coef": 0.5},
    "p4": {
        "kappa_coef": 0.5,
        "speed_weight_mode": "bucket",
        "replan_coef": 0.5,
        "lqr_reweight": True,
    },
}
ARM_ORDER: tuple[str, ...] = ("p0", "p1", "p2", "p3", "p4")


def config_for_arm(arm: str, **overrides) -> LateralLossConfig:
    """arm 名 → :class:`LateralLossConfig`（未知 arm fail-fast；overrides 覆盖预置）。"""
    name = str(arm).strip().lower()
    if name not in ARM_PRESETS:
        raise SystemExit(f"[lateral] fail-fast：未知 arm {arm!r}（可选 {' | '.join(ARM_ORDER)}）")
    return LateralLossConfig(**{**ARM_PRESETS[name], **overrides})


def lateral_loss_terms(
    config: LateralLossConfig,
    *,
    kappa_pred: torch.Tensor | None = None,
    kappa_target: torch.Tensor | None = None,
    kappa_mask: torch.Tensor | None = None,
    kappa_step_weights: torch.Tensor | None = None,
    plan_pred: torch.Tensor | None = None,
    plan_target_mask: torch.Tensor | None = None,
    kappa_prev_pred: torch.Tensor | None = None,
    warp_index: torch.Tensor | None = None,
    warp_mask: torch.Tensor | None = None,
    warp_weight: torch.Tensor | None = None,
    base_sample_weights: torch.Tensor | None = None,
    lqr_weights: torch.Tensor | None = None,
) -> dict:
    """按 :class:`LateralLossConfig` 组装横向损失项（各项系数 0 ⇒ 跳项不建图）。

    输入口径：
    - κ 监督：``kappa_pred/target/mask (B,S)``（站位三列；数据侧 = ``lateral_labels.station_targets``）、
      ``kappa_step_weights (S,)`` 可选；
    - L_Δκ / L_replan：``plan_pred (B,K,2)``（预测计划，κ= dθ/ds）、``plan_target_mask (B,K)``
      （= 专家步掩码，重规划项的目标有效性；L_Δκ 用预测 plan 自身掩码）。
    - L_replan：``kappa_prev_pred (B,K)`` + 数据侧 ``warp_index/warp_mask/warp_weight``；
    - 批次权：``base_sample_weights (B,)``（P2 速度桶；缺省见 :func:`combine_sample_weights`）、
      ``lqr_weights (B,)``（P4 gap 权重，含入因子 ``1 + lqr_alpha·w``；仅作用于 κ 监督 +
      L_replan，不作用于 L_Δκ）。

    返回 ``{"terms": {name: Tensor}, "total": Tensor, "sample_weights": Tensor|None,
    "active": tuple}``。**全默认（arm=p0）⇒ ``terms={}``、``total`` 恒 0**（逐位不变）。
    """
    terms: dict[str, torch.Tensor] = {}
    device = None
    for tensor in (kappa_pred, plan_pred, kappa_prev_pred):
        if torch.is_tensor(tensor):
            device = tensor.device
            break
    if device is None:
        device = torch.device("cpu")
    total = torch.zeros((), device=device, dtype=torch.float32)

    # ---- 批权重（P2/P4 组合；非梯度） ----
    sample_weights: torch.Tensor | None = None
    weight_parts: list[torch.Tensor] = []
    if base_sample_weights is not None:
        weight_parts.append(base_sample_weights)
    elif config.speed_weight_mode != "off" and kappa_pred is not None:
        raise SystemExit(
            "[lateral] fail-fast：speed_weight_mode != off 需要 base_sample_weights（数据侧速度桶权重）"
        )
    if config.lqr_reweight:
        if lqr_weights is None:
            raise SystemExit("[lateral] fail-fast：lqr_reweight=True 需要 lqr_weights（数据侧 gap 权重）")
        factor = 1.0 + float(config.lqr_alpha) * lqr_weights.detach().reshape(-1).to(torch.float32)
        weight_parts.append(factor)
    if weight_parts:
        sample_weights = combine_sample_weights(weight_parts, cap=config.weight_cap, normalize=True)

    # ---- 1) κ 监督（P1+） ----
    if config.kappa_coef:
        if kappa_pred is None or kappa_target is None or kappa_mask is None:
            raise SystemExit("[lateral] fail-fast：kappa_coef>0 需要 kappa_pred/target/mask")
        terms["kappa"] = float(config.kappa_coef) * kappa_supervision_loss(
            kappa_pred, kappa_target, kappa_mask,
            sample_weights=sample_weights, step_weights=kappa_step_weights, beta=config.beta,
        )["loss"]

    # ---- 2) L_Δκ（计划内平滑；默认 0） ----
    if config.delta_kappa_coef:
        if plan_pred is None:
            raise SystemExit("[lateral] fail-fast：delta_kappa_coef>0 需要 plan_pred")
        plan_kappa, plan_valid = plan_kappa_from_plan(plan_pred)
        terms["delta_kappa"] = float(config.delta_kappa_coef) * plan_delta_kappa_loss(
            plan_kappa, plan_valid, dk_thresh=config.dk_thresh,
            sample_weights=base_sample_weights,
        )["loss"]

    # ---- 3) L_replan（P3+；门控权重在数据侧） ----
    if config.replan_coef:
        if plan_pred is None or kappa_prev_pred is None or warp_index is None or warp_mask is None or warp_weight is None:
            raise SystemExit(
                "[lateral] fail-fast：replan_coef>0 需要 plan_pred/kappa_prev_pred/warp_index/warp_mask/warp_weight"
            )
        plan_kappa, plan_valid = plan_kappa_from_plan(plan_pred)
        if plan_target_mask is not None:
            plan_valid = plan_valid * (plan_target_mask.to(plan_valid.dtype) > 0.5).to(plan_valid.dtype)
        weight = warp_weight if sample_weights is None else warp_weight * sample_weights
        terms["replan"] = float(config.replan_coef) * replan_consistency_loss(
            plan_kappa, kappa_prev_pred, warp_index, warp_mask * plan_valid, weight,
        )["loss"]

    for value in terms.values():
        total = total + value
    return {"terms": terms, "total": total, "sample_weights": sample_weights, "active": config.active()}


__all__ = [
    "ARM_ORDER",
    "ARM_PRESETS",
    "BETA_DEFAULT",
    "DK_THRESH_DEFAULT",
    "LateralLossConfig",
    "combine_sample_weights",
    "config_for_arm",
    "kappa_supervision_loss",
    "masked_smooth_l1",
    "plan_delta_kappa_loss",
    "plan_kappa_from_plan",
    "replan_consistency_loss",
]
