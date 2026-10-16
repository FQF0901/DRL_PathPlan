#!/usr/bin/env python3
"""R41 第二批-F：**横向训练栈 / 锚输入结构修正 + 噪声增强（默认关；逐位）**。

规格出处
--------
- C 消费者图（``.slim/deepwork/p1_v2/round41/c_report.md`` §6-1/§6-6）：lane_anchor 是
  κ 通道**最高梯度弹性输入**（eval G=160.3 / F=241.5，≈次高 ego_phys 6.25 的 2 个数量级），
  机制 = ``κ_ref`` 由锚**顶点差分**解析提取（``Δθ/Δs``，锚点距 0.5 m）⇒ 点级噪声被
  **~1/Δs² 增益**放大（锚点级 5% 扰动即 >κ 尺度）；⇒ 修正 = **先平滑/低阶/≥2 m 重采样
  再差分**；另建议注入锚加 2–5 cm iid 噪声做输入增强（§6-6）。
- R41-F 任务书：修正作为**可开关选项，默认关**，并给**逐位证据**。

实现定位（写域说明）
--------------------
本模块作用于送入横向头的**锚折线输入**（obs ``lane_anchor_xy``/``lane_anchor_mask``），
不修改 ``net/path_head.anchor_reference``（net/ 为写域外）：默认 ``off`` ⇒ 返回输入本身
（逐位不变；守卫测试 ``torch.equal``）；开启后把平滑/重采样后的折线送入既有
``anchor_reference``（其顶点差分对 ≥2 m 间距的增益下降 ≈ (0.5/2.0)² = 1/16）。
**训练/评测一致性**：开启该开关属于输入结构变更，训练与评测必须同开（预注册 AC 轴）；
评测侧注入器接线为遗留项（见 ``f_report.md``）。

本模块零研究工作区依赖：只 import numpy/torch。
"""

from __future__ import annotations

import numpy as np

#: 修正模式（默认 off）
ANCHOR_CORRECTION_MODES: tuple[str, ...] = ("off", "resample", "smooth_resample")
#: ≥2 m 重采样最小值（C §6-1 建议）
MIN_RESAMPLE_DS_M = 2.0
#: 平滑窗口（点；仅内部点 → 均匀滤波）
SMOOTH_WINDOW_DEFAULT = 3
#: 锚噪声增强默认 sigma 区间（m；C §6-6：2–5 cm）
NOISE_SIGMA_RANGE_M: tuple[float, float] = (0.02, 0.05)


def resample_polyline(xy: np.ndarray, min_ds_m: float = MIN_RESAMPLE_DS_M) -> np.ndarray:
    """折线按**弧长 ≥ min_ds_m** 重采样（首点保留；≥ min_ds 才落点；末点保留）。

    返回新折线（float64）；点距 < 2 点的输入原样返回副本。
    """
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return points.copy()
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    total = float(segments.sum())
    if total <= float(min_ds_m):
        return np.vstack([points[:1], points[-1:]])
    out = [points[0]]
    travelled = 0.0
    for index, length in enumerate(segments):
        if length <= 0.0:
            continue
        # 段内按等弧长间距落点
        reached = travelled + length
        next_at = np.ceil((travelled + 1e-12) / float(min_ds_m)) * float(min_ds_m)
        while next_at < reached - 1e-9:
            ratio = (next_at - travelled) / length
            out.append(points[index] + ratio * (points[index + 1] - points[index]))
            next_at += float(min_ds_m)
        travelled = reached
    # 末端吸附：末段余量 < min_ds 时用原末点**替换**最后一个落点（保持"点距 ≥ min_ds"；
    # 首末点仍与原折线端点一致）；余量 ≥ min_ds（不可能出现）则直接追加。
    if np.linalg.norm(out[-1] - points[-1]) > 1e-9:
        if len(out) > 1 and np.linalg.norm(out[-1] - points[-1]) < float(min_ds_m):
            out[-1] = points[-1]
        else:
            out.append(points[-1])
    return np.asarray(out, dtype=np.float64)


def smooth_polyline(xy: np.ndarray, window: int = SMOOTH_WINDOW_DEFAULT) -> np.ndarray:
    """均匀滑动平均（仅内部点；首末点保持）——低阶平滑替代（不改变端点）。"""
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    window = int(window)
    if window <= 1 or len(points) <= 2:
        return points.copy()
    half = window // 2
    out = points.copy()
    for index in range(1, len(points) - 1):
        lo = max(0, index - half)
        hi = min(len(points), index + half + 1)
        out[index] = points[lo:hi].mean(axis=0)
    return out


def correct_polyline(
    xy: np.ndarray,
    *,
    mode: str = "off",
    min_ds_m: float = MIN_RESAMPLE_DS_M,
    smooth_window: int = SMOOTH_WINDOW_DEFAULT,
) -> np.ndarray:
    """单条折线修正（off ⇒ 原样副本；resample ⇒ 重采样；smooth_resample ⇒ 先平滑再重采样）。"""
    name = str(mode).strip().lower()
    if name not in ANCHOR_CORRECTION_MODES:
        raise SystemExit(f"[lateral] fail-fast：未知锚修正模式 {mode!r}（{' | '.join(ANCHOR_CORRECTION_MODES)}）")
    points = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if name == "off":
        return points.copy()
    if name == "smooth_resample":
        points = smooth_polyline(points, smooth_window)
    return resample_polyline(points, min_ds_m)


def correct_anchor_tensor(
    anchor_xy,
    anchor_mask=None,
    *,
    mode: str = "off",
    min_ds_m: float = MIN_RESAMPLE_DS_M,
    smooth_window: int = SMOOTH_WINDOW_DEFAULT,
):
    """（B,P,2）+（B,P）mask → 修正后的锚张量 ``(anchor_xy', anchor_mask')``。

    - ``off``：**原对象逐位返回**（零拷贝；默认逐位不变的最强形式）；
    - 修正：逐行抽取有效点（``mask>0.5``）→ :func:`correct_polyline` → 重打包回 padding 张量
      （长度变化；``anchor_mask'`` 标出有效点；新长度 ≤ 原 padding 宽度以外时扩宽）。
    """
    import torch

    if str(mode).strip().lower() == "off":
        return anchor_xy, anchor_mask
    if not torch.is_tensor(anchor_xy):
        anchor_xy = torch.as_tensor(anchor_xy, dtype=torch.float32)
    if anchor_mask is None:
        anchor_mask = torch.ones(anchor_xy.shape[:2], dtype=torch.float32, device=anchor_xy.device)
    batch = int(anchor_xy.shape[0])
    rows: list[np.ndarray] = []
    for index in range(batch):
        valid = anchor_mask[index].detach().cpu().numpy() > 0.5
        points = anchor_xy[index].detach().cpu().numpy()[valid]
        rows.append(correct_polyline(points, mode=mode, min_ds_m=min_ds_m, smooth_window=smooth_window))
    width = max(1, max(len(row) for row in rows))
    out_xy = torch.zeros((batch, width, 2), dtype=anchor_xy.dtype, device=anchor_xy.device)
    out_mask = torch.zeros((batch, width), dtype=torch.float32, device=anchor_xy.device)
    for index, row in enumerate(rows):
        count = len(row)
        if count:
            out_xy[index, :count] = torch.as_tensor(row, dtype=anchor_xy.dtype, device=anchor_xy.device)
            out_mask[index, :count] = 1.0
    return out_xy, out_mask


def perturb_anchor_tensor(
    anchor_xy,
    anchor_mask=None,
    *,
    sigma_m: float,
    generator=None,
):
    """锚噪声增强：对 ``mask>0.5`` 的点加 iid 高斯噪声 ``N(0, σ²)``（C §6-6；σ∈2–5 cm）。

    只扰动有效点（padding 恒 0）；返回新张量（不原地改）。
    """
    import torch

    if float(sigma_m) <= 0.0:
        raise SystemExit(f"[lateral] fail-fast：sigma_m 必须 >0，收到 {sigma_m}")
    if not torch.is_tensor(anchor_xy):
        anchor_xy = torch.as_tensor(anchor_xy, dtype=torch.float32)
    if anchor_mask is None:
        anchor_mask = torch.ones(anchor_xy.shape[:2], dtype=torch.float32, device=anchor_xy.device)
    noise = torch.randn(
        anchor_xy.shape, dtype=anchor_xy.dtype, device=anchor_xy.device, generator=generator
    ) * float(sigma_m)
    return anchor_xy + noise * (anchor_mask > 0.5).unsqueeze(-1).to(anchor_xy.dtype), anchor_mask


def apply_anchor_options(obs: dict, *, correction: str = "off", noise_sigma_m: float = 0.0, generator=None) -> dict:
    """obs 锚输入 → （可选）修正 + 噪声增强的**就地替换**（默认 off 且 σ=0 ⇒ obs 逐位不变）。

    返回同一 obs 字典（便于链式调用）；未提供 ``lane_anchor_xy`` 时 fail-fast。
    """
    if "lane_anchor_xy" not in obs:
        raise SystemExit("[lateral] fail-fast：obs 缺 lane_anchor_xy")
    anchor_mask = obs.get("lane_anchor_mask")
    corrected, new_mask = correct_anchor_tensor(
        obs["lane_anchor_xy"], anchor_mask, mode=correction
    )
    if float(noise_sigma_m) > 0.0:
        corrected, new_mask = perturb_anchor_tensor(
            corrected, new_mask, sigma_m=float(noise_sigma_m), generator=generator
        )
    obs["lane_anchor_xy"] = corrected
    obs["lane_anchor_mask"] = new_mask
    return obs


__all__ = [
    "ANCHOR_CORRECTION_MODES",
    "MIN_RESAMPLE_DS_M",
    "NOISE_SIGMA_RANGE_M",
    "SMOOTH_WINDOW_DEFAULT",
    "apply_anchor_options",
    "correct_anchor_tensor",
    "correct_polyline",
    "perturb_anchor_tensor",
    "resample_polyline",
    "smooth_polyline",
]
