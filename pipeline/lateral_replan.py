#!/usr/bin/env python3
"""R41 第二批-F：**横向训练栈 / 跨帧重规划对 + Warp + 门控（P3 数据侧）**。

规格出处
--------
- R41 复审（``docs/chatgpt5p6sol_review.md`` §三：P3 = P2 + 跨帧重规划一致性
  ``w(ΔS)·‖κ_t − Warp(κ_{t−1})‖``；**门控版：安全边界收缩/障碍变化 ⇒ 降权**）；
- 数据契约 = ``datasets/path_exec_v1``：决策步 stride 5（0.5 s）；``ego_world (N,3)``
  世界位姿；``long_exog/long_present`` 8 列（margin/static/dyn/ttc 见 manifest）。

口径（单一出处）
----------------
- **对** = 同一 episode 内**连续两个决策窗** ``t → t+1``（``step+5``），且两帧都在
  可训练集合（``train_weight>0 ∧ horizon_ok ∧ …``，= ``build_pairs`` 的 ``main_ids``）内；
- **Warp**：t−1 的计划 κ 是**按步段分段常数**的 κ(s)（与 :mod:`pipeline.lateral_labels` 的
  step 语义一致）。当前帧第 k 步的段末端弧长为 ``arc_k^curr``（专家弧长，数据侧固定），
  经**执行位移** ``ΔS`` 映射回上一帧弧轴：``s_prev = arc_k^curr + ΔS`` ⇒ 取包含
  ``s_prev`` 的上一帧步段；超出上一帧计划末弧 ⇒ 该列无效（不外推）。
  **为什么用执行位移（ego pose 对齐字段）**：``ΔS = ‖xy_{t+1} − xy_t‖`` 是**实际**运动
  位移（``ego_world`` 差分），不含"计划-执行落差"累积误差；同时给 ``(Δx,Δy,Δψ)``
  供位姿对齐门（航向失配 ⇒ 降权，见 :func:`psi_alignment_gate`）。
- **门控权重** ``w_replan = pair_valid · exp(−ΔS/s_ref) · g_safety · g_align``：
  - ``exp(−ΔS/s_ref)``：位移越大 warping 外推误差越大 ⇒ 降权（``s_ref`` 默认 5 m）；
  - ``g_safety = exp(−λ·Λ)``：Λ = {"corridor 半宽和收缩", "static 间距下降", "dyn 间距下降",
    "TTC 下降"} 的归一化不利变化最大值（各通道 ∈[0,1]、双侧 present 才计）——
    **安全边界收缩/障碍变化 ⇒ 降权**（R41 字面口径）；
  - ``g_align = exp(−|wrap(Δψ − κ_prev(ego位置)·ΔS)| / τ)``：实际航向变化与上一帧计划
    在当前位置的预测航向变化失配 ⇒ 计划/执行发散 ⇒ 降权（τ 默认 0.2 rad）。

本模块零研究工作区依赖：只 import numpy。
"""

from __future__ import annotations

import numpy as np

from pipeline.lateral_labels import PLAN_STEPS

#: 位移降权尺度（m；w_ds = exp(−ΔS/s_ref)）
DS_REF_M = 5.0
#: 安全门衰减系数（g = exp(−λ·Λ)）
SAFETY_GATE_LAMBDA = 1.0
#: 安全通道归一化尺度：corridor 半宽和 / static 间距 / dyn 间距 / TTC
SAFETY_WIDTH_SCALE_M = 3.5
SAFETY_DIST_SCALE_M = 50.0
SAFETY_TTC_SCALE_S = 30.0
#: 位姿对齐门尺度（rad）
PSI_ALIGN_TAU_RAD = 0.2


def _wrap_pi(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def build_replan_pairs(
    rows: dict,
    present: np.ndarray,
    exog: np.ndarray,
    main_ids: np.ndarray,
    succ_ids: np.ndarray,
    *,
    kappa: np.ndarray,
    arc: np.ndarray,
    step_mask: np.ndarray,
) -> dict:
    """连续决策窗对（t → t+1）→ warp 索引 + 位姿/安全字段 + 逐对权重。

    输入 = ``build_pairs`` 输出（``rows``/``present``/``exog``/``main_ids``/``succ_ids``）
    + :mod:`pipeline.lateral_labels` 的 ``kappa/arc/step_mask``。返回 dict（长度 P）:
    ``prev_ids/curr_ids``（行号）、``pose_prev/pose_curr (P,3)``、``delta_xy (P,2)``、
    ``delta_psi (P,)``、``ds_exec (P,)``、``warp_index (P,6)``（int64；-1 = 无效列）、
    ``warp_cost (P,6)``、``lam (P,)``、``safe_terms (P,4)``、``gate_safety/gate_align/
    w_replan (P,)``、``pair_valid (P,)``。
    """
    ego_world = rows.get("ego_world")
    if ego_world is None:
        raise SystemExit("[lateral] fail-fast：P3 跨帧重规划对需要 rows['ego_world']（世界位姿）")
    ego_world = np.asarray(ego_world, dtype=np.float64).reshape(-1, 3)
    present = np.asarray(present, dtype=np.float64)
    exog = np.asarray(exog, dtype=np.float64)
    main_ids = np.asarray(main_ids, dtype=np.int64).reshape(-1)
    succ_ids = np.asarray(succ_ids, dtype=np.int64).reshape(-1)
    kappa = np.asarray(kappa, dtype=np.float64)
    arc = np.asarray(arc, dtype=np.float64)
    step_mask = np.asarray(step_mask, dtype=np.float64) > 0.5

    position = {int(row): pos for pos, row in enumerate(main_ids.tolist())}
    prev_list: list[int] = []
    curr_list: list[int] = []
    for pos, row in enumerate(main_ids.tolist()):
        succ = int(succ_ids[pos])
        if succ >= 0 and succ in position:
            prev_list.append(position[int(row)])
            curr_list.append(position[succ])
    prev = np.asarray(prev_list, dtype=np.int64)
    curr = np.asarray(curr_list, dtype=np.int64)
    count = int(prev.size)

    prev_rows = main_ids[prev] if count else np.zeros(0, dtype=np.int64)
    curr_rows = main_ids[curr] if count else np.zeros(0, dtype=np.int64)
    pose_prev = ego_world[prev_rows] if count else np.zeros((0, 3))
    pose_curr = ego_world[curr_rows] if count else np.zeros((0, 3))
    delta_xy = pose_curr[:, :2] - pose_prev[:, :2] if count else np.zeros((0, 2))
    delta_psi = _wrap_pi(pose_curr[:, 2] - pose_prev[:, 2]) if count else np.zeros(0)
    ds_exec = np.linalg.norm(delta_xy, axis=1) if count else np.zeros(0)

    # ---- warp：当前帧专家弧 → 上一帧弧轴（执行位移平移；分段常数取值）
    warp_index = np.full((count, PLAN_STEPS), -1, dtype=np.int64)
    warp_cost = np.zeros((count, PLAN_STEPS), dtype=np.float64)
    if count:
        lower = np.concatenate([np.zeros((count, 1)), arc[prev_rows][:, :-1]], axis=1)
        for k in range(PLAN_STEPS):
            query = arc[curr_rows][:, k] + ds_exec
            for j in range(PLAN_STEPS):
                hit = (
                    step_mask[prev_rows, j]
                    & step_mask[curr_rows, k]
                    & (query > lower[:, j])
                    & (query <= arc[prev_rows, j])
                )
                warp_index[hit, k] = j
                warp_cost[hit, k] = np.abs(query[hit] - 0.5 * (lower[hit, j] + arc[prev_rows][hit, j]))

    # ---- 安全通道（双侧 present 才计；归一化 + clip [0,1]）
    terms = np.zeros((count, 4), dtype=np.float64)
    if count:
        both_w = (present[prev_rows, 1] > 0.5) & (present[prev_rows, 2] > 0.5) & (
            present[curr_rows, 1] > 0.5
        ) & (present[curr_rows, 2] > 0.5)
        width_prev = exog[prev_rows, 1] + exog[prev_rows, 2]
        width_curr = exog[curr_rows, 1] + exog[curr_rows, 2]
        terms[:, 0] = np.where(
            both_w, np.clip((width_prev - width_curr) / SAFETY_WIDTH_SCALE_M, 0.0, 1.0), 0.0
        )
        for column, (scale, index) in enumerate(((SAFETY_DIST_SCALE_M, 3), (SAFETY_DIST_SCALE_M, 4), (SAFETY_TTC_SCALE_S, 5))):
            both = (present[prev_rows, index] > 0.5) & (present[curr_rows, index] > 0.5)
            drop = exog[prev_rows, index] - exog[curr_rows, index]
            terms[:, column + 1] = np.where(both, np.clip(drop / scale, 0.0, 1.0), 0.0)
    lam = terms.max(axis=1) if count else np.zeros(0)
    gate_safety = np.exp(-float(SAFETY_GATE_LAMBDA) * lam) if count else np.zeros(0)

    # ---- 位姿对齐门：Δψ vs 上一帧计划在 ego 当前位置的预测航向变化（κ_prev(ΔS)·ΔS）
    gate_align = np.ones(count, dtype=np.float64)
    if count:
        kappa_at_ego = np.zeros(count, dtype=np.float64)
        lower = np.concatenate([np.zeros((count, 1)), arc[prev_rows][:, :-1]], axis=1)
        for j in range(PLAN_STEPS):
            hit = step_mask[prev_rows, j] & (ds_exec > lower[:, j]) & (ds_exec <= arc[prev_rows, j])
            kappa_at_ego[hit] = kappa[prev_rows][hit, j]
        mismatch = _wrap_pi(delta_psi - kappa_at_ego * ds_exec)
        gate_align = np.exp(-np.abs(mismatch) / float(PSI_ALIGN_TAU_RAD))

    pair_valid = (
        np.isfinite(delta_xy).all(axis=1)
        & np.isfinite(delta_psi)
        & (ds_exec > 1e-3)
    ) if count else np.zeros(0, dtype=bool)
    w_replan = np.where(
        pair_valid,
        np.exp(-ds_exec / float(DS_REF_M)) * gate_safety * gate_align,
        0.0,
    ) if count else np.zeros(0)
    return {
        "prev_ids": prev,
        "curr_ids": curr,
        "prev_row_ids": prev_rows,
        "curr_row_ids": curr_rows,
        "pose_prev": pose_prev,
        "pose_curr": pose_curr,
        "delta_xy": delta_xy,
        "delta_psi": delta_psi,
        "ds_exec": ds_exec,
        "warp_index": warp_index,
        "warp_cost": warp_cost,
        "lam": lam,
        "safe_terms": terms,
        "gate_safety": gate_safety,
        "gate_align": gate_align,
        "pair_valid": pair_valid,
        "w_replan": w_replan,
    }


def apply_warp(warp_index: np.ndarray, kappa_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``Warp(κ_{t−1})``：按 ``warp_index`` 收集上一帧 κ（-1 ⇒ 0 + 掩码 0）。

    ``kappa_prev (P,6)``（上一帧计划的步段 κ）→ ``(warped (P,6), mask (P,6))``。
    """
    index = np.asarray(warp_index, dtype=np.int64)
    kappa_prev = np.asarray(kappa_prev, dtype=np.float64)
    if index.shape != kappa_prev.shape:
        raise SystemExit(
            f"[lateral] fail-fast：warp_index 形状 {index.shape} != kappa_prev 形状 {kappa_prev.shape}"
        )
    valid = index >= 0
    rows = np.arange(index.shape[0])[:, None]
    cols = np.clip(index, 0, kappa_prev.shape[1] - 1)
    warped = np.where(valid, kappa_prev[rows, cols], 0.0)
    return warped, valid.astype(np.float32)


__all__ = [
    "DS_REF_M",
    "PSI_ALIGN_TAU_RAD",
    "SAFETY_DIST_SCALE_M",
    "SAFETY_GATE_LAMBDA",
    "SAFETY_TTC_SCALE_S",
    "SAFETY_WIDTH_SCALE_M",
    "apply_warp",
    "build_replan_pairs",
]
