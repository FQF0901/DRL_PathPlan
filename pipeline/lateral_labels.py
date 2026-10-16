#!/usr/bin/env python3
"""R41 第二批-F：**横向训练栈 / 数据侧（κ·计划监督目标 + 掩码 + 速度分桶）**。

规格出处
--------
- R41 复审（``docs/chatgpt5p6sol_review.md`` §三 第二批 P0–P4；``docs/deepseekv4p1_argue.md``
  §1.3②③④⑤ / §2 第 4–6 点）：横向训练前必须修正输入结构、κ_near/mid/far 监督带有效性掩码、
  E 证 ``horizon/present 掩码必做``；
- 数据契约 = ``datasets/path_exec_v1``：``action (N,6,2)`` = 专家 6 步 ``(ds,dθ)``
  （决策步 0.5 s；出处 I1 ``_extract_dataset``，H2 §1.1）；锚折线 = ego t0 系（x 前 / y 左）。

监督空间选择（**κ 空间**；理由）
--------------------------------
目标 ``κ_k = dθ_k / ds_k``（rad/m），k=0..5（h1..h6），而不是直接回归 ``(ds,dθ)``：

1. 横向头（``net/path_head.StructuredPolicyHead``）的横向分支**本就在 κ 空间**参数化
   （``κ = κ_ref + 残差``，残差按速度条件化限幅 ``κ_p95/p99(v)``；执行 ``dθ = κ·ds`` 是
   ``compose`` 确定性层）。κ 空间监督与输出参数化、限幅口径、以及横向门指标（κ 幅值/
   帧间翻转率/转向饱和 = ``atan(κ·L)/δ_max``）同轴；
2. ``(ds,dθ)`` 空间的 dθ 误差含纵向头误差的乘积耦合（``dθ = κ·ds``）——κ 空间把横向
   梯度与 ``ds`` 幅值解耦（现有 ``(ds,dθ)`` 链损失 **L_plan 保留**，本模块是再参数化补充项）；
3. 病态域显式掩码：``ds_k < ds_min``（默认 0.1 m，与 ``trainer.curv_min_ds_m`` 同口径）
   或非有限 ⇒ 该步 ``mask=0``（不给 0 目标；0/0 增益不注入）。

掩码口径（E §5.2 + C §6-5，单一出处 = :func:`plan_step_masks` / :func:`station_targets`）
------------------------------------------------------------------------------------------
- ``step_valid``：``finite(action) ∧ ds ≥ ds_min``；
- **锚视界掩码**：步 k 的**段末端**弧长 ``arc_k = Σ_{j≤k} ds_j`` 必须 ≤ 锚折线总弧长
  （``anchor_horizon_m``）——超出锚视界处 ``anchor_reference`` 是端点常数延伸（κ_ref 恒 0），
  监督在第 k 步与其输入结构不自洽；C/E 已证评测帧 99.1% 锚视界 < 29.75 m（κ_far≈0）；
- ``plan_mask = step_valid ∧ anchor_horizon_valid``（float32；统一掩码）；
- 站位插值（``station_targets``）：κ 按**步段分段常数**取值（``arc_{k-1} < s ≤ arc_k ⇒ κ_k``），
  站位超过专家计划末弧 ⇒ ``mask=0``（不能外推）。

本模块零研究工作区依赖：只 import numpy。
"""

from __future__ import annotations

import numpy as np

#: 计划步数（= ``net.path_head.NUM_STEPS``；守卫测试断言一致）与单步时长
PLAN_STEPS = 6
DT = 0.5
#: 步有效的最小弧长（m；= ``pipeline.trainer`` ``curv_min_ds_m`` 默认口径）
DS_MIN = 0.1
#: 横向控制站位（m；= ``net.path_head.KAPPA_CTRL_S`` = 0.15/0.50/0.85 × 35 m）
KAPPA_STATIONS: tuple[float, float, float] = (5.25, 17.5, 29.75)
#: 速度桶边界（m/s；= ``net.path_head.KAPPA_SPEED_BUCKETS``）
SPEED_BUCKET_EDGES: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0)
#: 速度分桶权重默认表（len = len(edges)；[0,3)/[3,6)/[6,9)/[9,12)/[12,∞)）：
#: 风险（``a_y = κv²``、饱和耗损）随速度单调 ⇒ 高速桶温和上调；低/中速桶保持 1.0
#: （数据支撑：训练帧 |κ| 均值低/中速桶 0.013/0.015 vs [6,9) 0.005 ⇒ 曲率多样性主要在
#: 低/中速，权重不得反向压制；见 ``f_report.md`` 表 1）。
SPEED_BUCKET_WEIGHTS_DEFAULT: tuple[float, ...] = (1.0, 1.0, 1.25, 1.5, 1.5)


def plan_targets_from_action(action: np.ndarray, *, ds_min: float = DS_MIN) -> dict:
    """``action (N,6,2)``（专家 ``(ds,dθ)``）→ h1..h6 目标（κ 空间 + 弧长 + 步有效性）。

    Returns dict（全部 float64 numpy；``kappa`` 无效步置 0）:
    - ``ds (N,6)`` / ``dtheta (N,6)``：原样透传（供 ``(ds,dθ)`` 口径消费者）；
    - ``kappa (N,6)`` = ``dtheta / ds``（``ds < ds_min`` ⇒ 0）；
    - ``arc (N,6)``：逐步累计弧长（``cumsum(clip(ds, 0, None))``；步 k 的**段末端**弧长）；
    - ``step_valid (N,6)`` bool：``finite ∧ ds ≥ ds_min``。
    """
    arr = np.asarray(action, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1] != PLAN_STEPS or arr.shape[2] != 2:
        raise SystemExit(
            f"[lateral] fail-fast：action 形状应为 (N,{PLAN_STEPS},2)，收到 {tuple(arr.shape)}"
        )
    ds = arr[:, :, 0].copy()
    dtheta = arr[:, :, 1].copy()
    finite = np.isfinite(ds) & np.isfinite(dtheta)
    step_valid = finite & (ds >= float(ds_min))
    ds_pos = np.where(finite, np.clip(ds, 0.0, None), 0.0)
    arc = np.cumsum(ds_pos, axis=1)
    kappa = np.zeros_like(ds)
    np.divide(dtheta, ds, out=kappa, where=step_valid)
    return {
        "ds": ds,
        "dtheta": dtheta,
        "kappa": kappa,
        "arc": arc,
        "step_valid": step_valid,
    }


def anchor_horizon_lengths(anchor_list) -> np.ndarray:
    """锚折线列表（ragged）→ 每行总弧长 ``(N,)``（m；<2 点 ⇒ 0）。"""
    out = np.zeros(len(anchor_list), dtype=np.float64)
    for index, anchor in enumerate(anchor_list):
        xy = np.asarray(anchor, dtype=np.float64).reshape(-1, 2)
        if len(xy) >= 2:
            out[index] = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    return out


def plan_step_masks(
    step_valid: np.ndarray,
    arc: np.ndarray,
    anchor_horizon_m: np.ndarray,
    *,
    horizon_margin_m: float = 0.0,
) -> np.ndarray:
    """步掩码 = ``step_valid ∧ (arc_k ≤ anchor_horizon_m − margin)``（float32 (N,6)）。

    ``arc_k`` = 步 k 的段末端弧长（锚视界必须覆盖整段）；``horizon_margin_m`` =
    预注册的安全边际（默认 0）。
    """
    valid = np.asarray(step_valid, dtype=bool)
    arcs = np.asarray(arc, dtype=np.float64)
    horizon = np.asarray(anchor_horizon_m, dtype=np.float64).reshape(-1, 1)
    if arcs.shape != valid.shape or horizon.shape[0] != arcs.shape[0]:
        raise SystemExit(
            f"[lateral] fail-fast：掩码形状不一致（valid {valid.shape} / arc {arcs.shape} / "
            f"horizon {horizon.shape}）"
        )
    inside = arcs <= (horizon - float(horizon_margin_m))
    return (valid & inside).astype(np.float32)


def station_targets(
    kappa: np.ndarray,
    arc: np.ndarray,
    step_mask: np.ndarray,
    *,
    stations: tuple[float, ...] = KAPPA_STATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """步段分段常数 κ 在固定站位上的取值 → ``(target (N,S), mask (N,S))``。

    口径：站位 ``s`` 落在步 k 段（``arc_{k-1} < s ≤ arc_k``，``arc_{-1}=0``）⇒ 取 ``κ_k``；
    该步 ``mask_k=0`` 或 ``s`` 超出最后一个有效步的段末端 ⇒ 掩码 0（不外推）。
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    arcs = np.asarray(arc, dtype=np.float64)
    mask = np.asarray(step_mask, dtype=np.float64) > 0.5
    n, steps = kappa.shape
    targets = np.zeros((n, len(stations)), dtype=np.float64)
    out_mask = np.zeros((n, len(stations)), dtype=np.float64)
    lower = np.concatenate([np.zeros((n, 1), dtype=np.float64), arcs[:, :-1]], axis=1)
    for column, station in enumerate(stations):
        for k in range(steps):
            hit = mask[:, k] & (float(station) > lower[:, k]) & (float(station) <= arcs[:, k])
            targets[hit, column] = kappa[hit, k]
            out_mask[hit, column] = 1.0
    return targets, out_mask


def speed_bucket_ids(speed: np.ndarray, *, edges: tuple[float, ...] = SPEED_BUCKET_EDGES) -> np.ndarray:
    """速度 → 桶号（``np.digitize``，右开区间；``edges`` 升序）。非有限值 → -1。"""
    values = np.asarray(speed, dtype=np.float64).reshape(-1)
    ids = np.digitize(values, np.asarray(edges, dtype=np.float64), right=False) - 1
    ids = np.clip(ids, 0, len(edges) - 1)
    ids[~np.isfinite(values)] = -1
    return ids.astype(np.int64)


def speed_bucket_weights(
    speed: np.ndarray,
    present: np.ndarray,
    *,
    edges: tuple[float, ...] = SPEED_BUCKET_EDGES,
    weights: tuple[float, ...] = SPEED_BUCKET_WEIGHTS_DEFAULT,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """P2 速度分桶逐样本权重：``w_i = weights[bucket(v_i)]``（缺速度 ⇒ 1.0 中性）。

    - ``present`` = 速度列 presence（``>0.5`` 才算有效）；速度无效 ⇒ ``w=1.0``（中性，
      不参与归一化统计）；
    - ``normalize=True``（默认）：对**有效样本**归一化到均值 = 1（保持有效学习率不因
      权重表整体尺度漂移；预注册冻结口径）。
    Returns ``(w (N,), bucket_id (N,))``。
    """
    values = np.asarray(speed, dtype=np.float64).reshape(-1)
    present = np.asarray(present, dtype=np.float64).reshape(-1) > 0.5
    table = np.asarray(weights, dtype=np.float64).reshape(-1)
    if table.size != len(edges):
        raise SystemExit(
            f"[lateral] fail-fast：weights 长度 {table.size} != edges 长度 {len(edges)}"
        )
    ids = speed_bucket_ids(values, edges=edges)
    valid = present & (ids >= 0) & np.isfinite(values)
    out = np.ones(values.shape[0], dtype=np.float64)
    out[valid] = table[ids[valid]]
    if normalize and valid.any():
        mean = float(out[valid].mean())
        if mean > 0.0:
            out[valid] = out[valid] / mean
    return out.astype(np.float32), ids


__all__ = [
    "DT",
    "DS_MIN",
    "KAPPA_STATIONS",
    "PLAN_STEPS",
    "SPEED_BUCKET_EDGES",
    "SPEED_BUCKET_WEIGHTS_DEFAULT",
    "anchor_horizon_lengths",
    "plan_step_masks",
    "plan_targets_from_action",
    "speed_bucket_ids",
    "speed_bucket_weights",
    "station_targets",
]
