"""R21-27B：正常状态（expert-state）residual 三类分解 —— **定义锁定模块**（trainer/dataset 层）。

分类（round-27 §27B；不进 model forward）：

- **N0**（legacy 已足够正确）：``|expert−legacy|`` 小（≤ 教师行残差 P50）**且** 未来 H 步无失败前兆
  （横偏/航向/速度均在安全界内）→ semantic residual 目标 ≈ 0（preservation 仅作用于本类）；
- **N1**（稳定小偏差）：残差中等（≤ P90）且**同 episode 内残差符号一致**（系统性 bias，如跨路型
  一致的 ds 低估）且未来安全 → 允许小 residual（不做 preservation）；
- **C**（perturbation/learner 修正）：``row_type ∈ {1,2}``，或教师行但残差大/未来不可验证/不安全
  → 学习完整 expert residual。

阈值全部来自**数据分位**（无手调）：``tau_ds/tau_dth`` = 教师行残差 P50；``tau_ds_n1/tau_dth_n1`` = P90。
安全界（物理定义，非调参）：``LAT_SAFE=1.0 m``、``HEAD_SAFE=0.10 rad``、``SPEED_FLOOR=1.0 m/s``
（未来 H 步均需满足）；N1 的 episode 一致性 = 同 episode 教师行中与 episode 残差中位数同号比例 ≥ 0.7。

纯函数（无 torch/IO）：``classify_rows`` / ``class_stats``；比例统计供报告。
"""

from __future__ import annotations

import numpy as np

CLASS_N0 = "N0"
CLASS_N1 = "N1"
CLASS_C = "C"
CLASSES = (CLASS_N0, CLASS_N1, CLASS_C)

LAT_SAFE = 1.0
HEAD_SAFE = 0.10
SPEED_FLOOR = 1.0
CONSISTENCY_MIN = 0.7


def compute_thresholds(res_ds: np.ndarray, res_dth: np.ndarray, teacher_mask: np.ndarray) -> dict:
    """阈值 = 教师行残差分位（P50/P90）；无教师行 → fail-fast。"""
    ds = np.abs(np.asarray(res_ds, dtype=np.float64))[teacher_mask]
    dth = np.abs(np.asarray(res_dth, dtype=np.float64))[teacher_mask]
    if ds.size == 0:
        raise ValueError("compute_thresholds：无教师行（teacher_mask 全 False）")
    return {
        "tau_ds": float(np.quantile(ds, 0.50)),
        "tau_dth": float(np.quantile(dth, 0.50)),
        "tau_ds_n1": float(np.quantile(ds, 0.90)),
        "tau_dth_n1": float(np.quantile(dth, 0.90)),
        "n_teacher": int(ds.size),
    }


def episode_consistency(episode_id: np.ndarray, res_ds: np.ndarray, teacher_mask: np.ndarray) -> dict:
    """逐 episode：教师行 Δds 与 episode 中位数同号比例（≥ ``CONSISTENCY_MIN`` = 系统性 bias）。"""
    ep = np.asarray(episode_id)
    res = np.asarray(res_ds, dtype=np.float64)
    out: dict[int, float] = {}
    for key in np.unique(ep[teacher_mask]):
        mask = teacher_mask & (ep == key)
        values = res[mask]
        if values.size == 0:
            continue
        # 与"主导符号"一致的比例（全同号 → 1.0；来回振荡 → ~0.5；中位数≈0 不做特殊化）
        frac_pos = float(np.mean(values > 0.0))
        out[int(key)] = max(frac_pos, 1.0 - frac_pos)
    return out


def classify_rows(features: dict, thresholds: dict) -> np.ndarray:
    """逐行分类（返回 ``np.ndarray[object]`` ∈ {N0, N1, C}）。

    ``features`` 需含：``row_type``（0/1/2）、``res_ds``、``res_dth``、``future_valid``（bool）、
    ``future_lat_max``、``future_head_max``、``future_speed_min``、``episode_id``。
    """
    row_type = np.asarray(features["row_type"], dtype=np.int64)
    res_ds_signed = np.asarray(features["res_ds"], dtype=np.float64)
    res_ds = np.abs(res_ds_signed)          # 幅度口径（定义：|expert−legacy| 小）
    res_dth = np.abs(np.asarray(features["res_dth"], dtype=np.float64))
    valid = np.asarray(features["future_valid"], dtype=bool)
    lat_max = np.asarray(features["future_lat_max"], dtype=np.float64)
    head_max = np.asarray(features["future_head_max"], dtype=np.float64)
    speed_min = np.asarray(features["future_speed_min"], dtype=np.float64)
    n = row_type.shape[0]
    labels = np.array([CLASS_C] * n, dtype=object)
    teacher = row_type == 0
    safe_future = valid & (lat_max <= LAT_SAFE) & (head_max <= HEAD_SAFE) & (speed_min >= SPEED_FLOOR)
    n0 = (
        teacher & safe_future
        & (res_ds <= thresholds["tau_ds"]) & (res_dth <= thresholds["tau_dth"])
    )
    labels[n0] = CLASS_N0
    # episode 一致性用**带符号**残差（系统性 bias 判定）
    consistency = episode_consistency(features["episode_id"], res_ds_signed, teacher & safe_future)
    ep = np.asarray(features["episode_id"])
    consistent = np.array([consistency.get(int(key), 0.0) >= CONSISTENCY_MIN for key in ep], dtype=bool)
    n1 = (
        teacher & safe_future & (~n0)
        & (res_ds <= thresholds["tau_ds_n1"]) & (res_dth <= thresholds["tau_dth_n1"])
        & consistent
    )
    labels[n1] = CLASS_N1
    return labels


def class_stats(labels: np.ndarray, features: dict) -> dict:
    """比例统计：总体 + 按 row_type/geometry/速度带/|lat| 带分层（供报告）。"""
    labels = np.asarray(labels)
    n = labels.shape[0]
    stats: dict = {"n": int(n), "overall": {c: int((labels == c).sum()) for c in CLASSES},
                   "share": {c: float(np.mean(labels == c)) for c in CLASSES}}
    row_type = np.asarray(features["row_type"], dtype=np.int64)
    by_type = {}
    for rt, name in ((0, "teacher"), (1, "pert"), (2, "learner")):
        mask = row_type == rt
        if int(mask.sum()) == 0:
            continue
        by_type[name] = {c: {"n": int((labels[mask] == c).sum()), "share": float(np.mean(labels[mask] == c))}
                         for c in CLASSES}
    stats["by_row_type"] = by_type
    geometry = np.asarray(features.get("geometry_code", np.zeros(n)), dtype=np.int64)
    stats["by_geometry"] = {}
    for code in np.unique(geometry):
        mask = geometry == code
        stats["by_geometry"][int(code)] = {c: int((labels[mask] == c).sum()) for c in CLASSES}
    speed = np.asarray(features.get("speed", np.zeros(n)), dtype=np.float64)
    stats["by_speed_band"] = {}
    for band, mask in (("<5", speed < 5.0), ("5-10", (speed >= 5.0) & (speed < 10.0)), (">=10", speed >= 10.0)):
        if int(mask.sum()) == 0:
            continue
        stats["by_speed_band"][band] = {c: int((labels[mask] == c).sum()) for c in CLASSES}
    return stats
