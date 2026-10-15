"""``path_dynamic_distance`` / ``path_dynamic_ttc`` / ``path_static_distance`` 的**公式单一出处**。

出处与口径（R38-Q4A：A 类落地 = 生产者补全 + 禁静默 0；分类/裁定见
``docs/repo_handbook.md`` 附录 D）
--------------------------------------------------------------------------------
1. OD 槽布局（``env/obs/od.py`` 的 ``_candidate_rows`` 输出，自车 t0 系 x 前 / y 左）::

       row = [dx, dy, vx, vy, cosθ_rel, sinθ_rel, LENGTH, WIDTH, vehicle_type_id]

   ``od_presence=1`` = 该槽对象**本帧在盒内被观测**（``env/obs/od.py``）。

2. TTC（= DP-6 ``DYN_TTC_CAP_S`` 口径，round-36 ``i1_extract_dataset.load_source`` 逐字同公式）：
   接近判据 ``dx > 0 ∧ vx < −TTC_EPS``（``TTC_EPS = 1e-3`` = ``env.obs.od.ODChannel._EPS``）；
   取 presence=1 槽 ``dx / (−vx)`` 的 min 并 **cap 60 s**；无接近 ⇒ 60 = 无冲突（present 恒 1）。

3. 动态距离（= DP-6 口径）：ego 中心 → 目标矩形 (L,W,朝向 cos/sin) 的最近距离取 min
   （矩形内 ⇒ 0；含后向物体；scope 前 150 / 后 50 / 左右 25 m）；无有效槽 ⇒ ``present=0``。

4. 静态距离（= ``env.obs.static.scan_static_blocker`` 同函数）：自车当前车道中心线走廊内
   前向最近静态障碍净距，范围 50 m、clamp ≥ 0；无 ⇒ ``present=0``。

消费方（**同一函数，不复制**）：
- 生产 env 侧：``env/obs/path_field.py``（``ObservationBuilder`` 的 t0 段 8 列生产者）；
- 评测注入器：``.slim/deepwork/p1_v2/round37/m7_fill.py``（re-export 本模块，R37-M7 口径）;
- 训练侧数据集抽取：``.slim/deepwork/p1_v2/round36/i1_extract_dataset.py::load_source``。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

__all__ = [
    "OD_FEATURE_DIM",
    "OD_LAYOUT",
    "OD_NUM_SLOTS",
    "TTC_CAP_S",
    "TTC_EPS",
    "TRUTH_FILL_KEYS",
    "TRUTH_TELEMETRY_KEYS",
    "od_slots",
    "od_truth_slots",
    "od_truth",
    "static_gap",
    "truth_fill",
]

#: OD 槽数 / 特征维（``env/obs/builder.py`` od 段；(16,9)）
OD_NUM_SLOTS = 16
OD_FEATURE_DIM = 9
#: 槽特征列序（``env/obs/od.py``）
OD_LAYOUT = ("dx", "dy", "vx", "vy", "cos", "sin", "L", "W", "type")
#: 接近判据（= ``env.obs.od.ODChannel._EPS``）
TTC_EPS = 1e-3
#: TTC 上限（= DP-6 ``DYN_TTC_CAP_S``）
TTC_CAP_S = 60.0

#: 真值回填键（= ``net/path_head.LONG_EXOG_OBS_KEYS`` 的 static/dynamic 三列）
TRUTH_FILL_KEYS = ("path_static_distance", "path_dynamic_distance", "path_dynamic_ttc")
#: true 档专属 trace 键（head 实际输入 4 键由评测侧与 M6 同口径记录）
TRUTH_TELEMETRY_KEYS = (
    "long_exog_od_live", "long_exog_od_appr",
    "long_exog_true_static", "long_exog_true_dyn", "long_exog_true_ttc",
)


# --------------------------------------------------------------------------- #
# OD 槽 → 公式
# --------------------------------------------------------------------------- #
def od_slots(obs: Mapping[str, Any]) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``obs`` → ``(od (S,9) float64, presence (S,) bool)``；缺键 → ``(None, None)``。

    ``obs['od']`` / ``obs['od_presence']`` 由 ``ObservationBuilder.build`` 输出
    （``env/obs/builder.py``；(16,9)/(16,)，ego t0 系）。
    """
    raw = obs.get("od")
    if raw is None:
        return None, None
    od = np.asarray(raw, dtype=np.float64)
    if od.ndim > 2:  # (1,S,9) 容忍
        od = od.reshape(-1, od.shape[-1])
    if od.ndim != 2 or od.shape[-1] < OD_FEATURE_DIM:
        return None, None
    od = od[:, :OD_FEATURE_DIM]
    presence_raw = obs.get("od_presence")
    if presence_raw is None:
        presence = np.ones((od.shape[0], ), dtype=bool)
    else:
        presence = np.asarray(presence_raw, dtype=np.float64).reshape(-1) > 0.5
        presence = presence[:od.shape[0]]
    return od, presence


def od_truth_slots(
    od: np.ndarray,
    presence: np.ndarray,
    *,
    ttc_cap_s: float = TTC_CAP_S,
    eps: float = TTC_EPS,
) -> dict[str, np.ndarray]:
    """**向量化** OD 槽 → 动态真值（末二 = 槽、末一 = 特征；前导维任意）。

    ``od`` ``(..., S, 9)`` / ``presence`` ``(..., S)`` → 各输出 ``(...)``：
    ``dyn_distance``（无有效槽 = ``inf``）/ ``dyn_present`` / ``ttc``（cap 后恒有限）/
    ``ttc_present`` / ``n_live`` / ``n_approach``。公式见模块 docstring（1–3 条）。
    """
    od = np.asarray(od, dtype=np.float64)
    if od.ndim < 2 or od.shape[-1] < OD_FEATURE_DIM:
        raise ValueError(f"od 形状应为 (...,S,{OD_FEATURE_DIM})，收到 {tuple(od.shape)}")
    slots = int(od.shape[-2])
    od = od[..., :OD_FEATURE_DIM]
    # presence 与 od 前导维同形、末维 = 槽（只在末轴对齐：容忍槽数不足/超出的装配变体）
    presence = np.asarray(presence, dtype=np.float64)
    if presence.ndim == 0:
        presence = np.full(od.shape[:-1], float(presence))
    if int(presence.shape[-1]) < slots:  # 缺的槽视为不 present
        presence = np.concatenate(
            [presence, np.zeros(presence.shape[:-1] + (slots - int(presence.shape[-1]), ))], axis=-1
        )
    presence = presence[..., :slots]
    if slots == 0:  # 空槽表（不会发生在生产装配）：无观测
        lead = od.shape[:-2]
        inf = np.full(lead, np.inf)
        return {
            "dyn_distance": inf, "dyn_present": np.zeros(lead, dtype=bool),
            "ttc": np.full(lead, float(ttc_cap_s)), "ttc_present": np.ones(lead, dtype=bool),
            "n_live": np.zeros(lead, dtype=np.int64),
            "n_approach": np.zeros(lead, dtype=np.int64),
        }

    dx, dy, vx, vy = od[..., 0], od[..., 1], od[..., 2], od[..., 3]
    cth, sth, ln, wd = od[..., 4], od[..., 5], od[..., 6], od[..., 7]
    valid = (
        (presence > 0.5) & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(vx)
        & np.isfinite(vy) & np.isfinite(cth) & np.isfinite(sth) & (ln > 0.0) & (wd > 0.0)
    )
    # 距离：ego 中心 → 目标矩形最近距离（矩形内 = 0）
    u = -(dx * cth + dy * sth)
    v = dx * sth - dy * cth
    ex = np.maximum(np.abs(u) - 0.5 * ln, 0.0)
    ey = np.maximum(np.abs(v) - 0.5 * wd, 0.0)
    dist = np.where(valid, np.hypot(ex, ey), np.inf)
    dyn_distance = np.min(dist, axis=-1)
    # TTC：dx>0 ∧ vx<−eps 的接近槽取 min，cap（无接近 ⇒ cap = 无冲突，present 恒 1）
    approaching = valid & (dx > 0.0) & (vx < -float(eps))
    ttc_raw = np.where(approaching, dx / np.maximum(-vx, 1e-9), np.inf)
    ttc = np.minimum(np.min(ttc_raw, axis=-1), float(ttc_cap_s))
    return {
        "dyn_distance": dyn_distance,
        "dyn_present": np.isfinite(dyn_distance),
        "ttc": ttc,
        "ttc_present": np.isfinite(ttc),
        "n_live": valid.sum(axis=-1).astype(np.int64),
        "n_approach": approaching.sum(axis=-1).astype(np.int64),
    }


def od_truth(
    od: np.ndarray,
    presence: np.ndarray,
    *,
    ttc_cap_s: float = TTC_CAP_S,
    eps: float = TTC_EPS,
) -> dict:
    """单帧 OD 槽 ``(S,9)/(S,)`` → 真值标量（R37-M7 ``od_truth`` 同签名/同返回）。

    实现 = :func:`od_truth_slots` 取第 0 行（单帧），保证与向量化路径**同一公式**。
    """
    od_arr = np.asarray(od, dtype=np.float64)
    if od_arr.ndim == 2 and od_arr.shape[-1] >= OD_FEATURE_DIM:
        out = od_truth_slots(od_arr[None, ...], np.asarray(presence, dtype=np.float64).reshape(1, -1),
                             ttc_cap_s=ttc_cap_s, eps=eps)
        dyn_distance = float(out["dyn_distance"][0])
        return {
            "dyn_distance": (dyn_distance if bool(out["dyn_present"][0]) else float("nan")),
            "dyn_present": bool(out["dyn_present"][0]),
            "ttc": float(out["ttc"][0]),
            "ttc_present": bool(out["ttc_present"][0]),
            "n_live": int(out["n_live"][0]),
            "n_approach": int(out["n_approach"][0]),
        }
    # 退化输入（空槽表/畸形）：与 od_truth_slots 的空表分支同口径
    return {
        "dyn_distance": float("nan"), "dyn_present": False,
        "ttc": float(ttc_cap_s), "ttc_present": True,
        "n_live": 0, "n_approach": 0,
    }


def static_gap(env: Any, ego: Any) -> Optional[float]:
    """走廊内前向最近静态障碍净距（``env.obs.static.scan_static_blocker`` 同函数）；无 → None。"""
    from env.obs.static import scan_static_blocker

    found = scan_static_blocker(env, ego)
    if found is None:
        return None
    return float(max(float(found[0]), 0.0))


# --------------------------------------------------------------------------- #
# obs 回填（仅缺键；R37-M7 口径）
# --------------------------------------------------------------------------- #
def truth_fill(obs: dict, env: Any, ego: Any, *,
               ttc_cap_s: float = TTC_CAP_S, eps: float = TTC_EPS) -> dict:
    """决策步 ``obs`` **缺键** → 真值三键；返回记录 + OD 统计（已有键不动）。

    Returns:
        dict：``filled``（键 → 写入值；仅本次真正写入的键）、``truth``（三列本步真值；
        static/dyn 无观测 = NaN）、``od``（``n_live`` / ``n_approach`` / ``ok``）、
        ``present``（写入键的 present 位，用于对照训练侧 present 口径）。
    """
    od, presence = od_slots(obs)
    record: dict = {
        "filled": {}, "present": {}, "static_hit": False, "dyn_hit": False,
        "ttc": float("nan"), "truth": {"static": float("nan"), "dyn": float("nan"),
                                       "ttc": float("nan")},
        "od": {"n_live": 0, "n_approach": 0, "ok": bool(od is not None)},
    }
    if od is None:
        # 无 OD 通道（本 eval 装配不会发生）：不猜值，保持缺键
        return record
    truth = od_truth(od, presence if presence is not None else np.ones(od.shape[0], bool),
                     ttc_cap_s=ttc_cap_s, eps=eps)
    record["od"] = {"n_live": truth["n_live"], "n_approach": truth["n_approach"], "ok": True}
    record["truth"]["dyn"] = float(truth["dyn_distance"])
    record["truth"]["ttc"] = float(truth["ttc"])
    record["ttc"] = float(truth["ttc"])

    if obs.get("path_dynamic_ttc") is None:
        obs["path_dynamic_ttc"] = np.asarray(np.float32(truth["ttc"]))
        record["filled"]["path_dynamic_ttc"] = float(truth["ttc"])
        record["present"]["path_dynamic_ttc"] = bool(truth["ttc_present"])
    if truth["dyn_present"] and obs.get("path_dynamic_distance") is None:
        obs["path_dynamic_distance"] = np.asarray(np.float32(truth["dyn_distance"]))
        record["filled"]["path_dynamic_distance"] = float(truth["dyn_distance"])
        record["present"]["path_dynamic_distance"] = True
        record["dyn_hit"] = True

    gap = static_gap(env, ego)
    record["truth"]["static"] = float("nan") if gap is None else float(gap)
    if gap is not None and obs.get("path_static_distance") is None:
        obs["path_static_distance"] = np.asarray(np.float32(gap))
        record["filled"]["path_static_distance"] = float(gap)
        record["present"]["path_static_distance"] = True
        record["static_hit"] = True
    return record
