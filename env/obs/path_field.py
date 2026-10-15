"""t0 段 ``path_*`` 纵向外源几何的**生产者**（8 列 + 8 present；值/present 同源）。

背景（R38-Q4A，A 类落地；分类 = ``docs/repo_handbook.md`` 附录 D）
--------------------------------------------------------------------------------
结构化头的 8 列外源几何（列序 = ``net/path_head.LONG_EXOG_FEATURE_NAMES``）此前**没有生产
者**：eval env 的 obs 里压根没有这些键 ⇒ ``net/path_head.py::long_exog_from_obs`` 缺键回退 0
（其中 ``dynamic_ttc`` = 分布外「立即碰撞」，R37-M3/M6/M7 已证为最大边际项）。本模块把
**t0 段 8 列**改由 env 侧显式产出（并同时产出与值同源的 ``<key>__present`` 掩码列）：

===============================  ==================================================
列（obs 键）                       口径
===============================  ==================================================
``path_speed``                    ``ego[:, 0]``（head 的 speed 首选源；同源 present）
``path_left_margin``              ``lane_semantic[:, 4]``（既有回退口径；同值）
``path_right_margin``             ``lane_semantic[:, 5]``
``path_static_distance``          诊断列：``scan_static_blocker`` 真值 / 无 ⇒ 0+present=0
``path_dynamic_distance``         OD 槽矩形最近距（``env.obs.long_exog`` 同公式）
``path_dynamic_ttc``              OD 槽 ``dx/(−vx)`` min cap 60 s（同公式）
``path_nav_progress``             ``nav[:, -1]``（route_completion）
``path_distance_to_target_lane_end``  ``lane_semantic[:, 11]``
===============================  ==================================================

**源通道不在场 ⇒ 该列不产出**（不写 0）：头侧对 margin/nav/d2end 保留既有回退口径
（``lane_semantic`` / ``nav``），对 dynamic 两列**缺键即 fail-fast**（禁静默 0）。
``static`` 列恒产出（诊断；无真值 ⇒ 值 0 + present 0，显式主态）。

公式出处：``env/obs/long_exog.py``（动态距离 / TTC / 静态距；M7 注入器与训练侧抽取引用
同一函数，不复制）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from env.obs.base import safe_ego
from env.obs.long_exog import od_slots, od_truth_slots, static_gap

__all__ = [
    "PATH_LONG_EXOG_KEYS",
    "PATH_PRESENT_SUFFIX",
    "path_present_key",
    "path_long_exog_fields",
]

#: t0 段 8 列（顺序 = 头输入列序 = ``net/path_head.LONG_EXOG_FEATURE_NAMES``）
PATH_LONG_EXOG_KEYS: tuple[str, ...] = (
    "path_speed",
    "path_left_margin",
    "path_right_margin",
    "path_static_distance",
    "path_dynamic_distance",
    "path_dynamic_ttc",
    "path_nav_progress",
    "path_distance_to_target_lane_end",
)

#: present 掩码键后缀（与训练侧 ``j1/j2`` 注入口径一致：``<key>__present``）
PATH_PRESENT_SUFFIX = "__present"

#: ``lane_semantic`` 通道里 margin/d2end 的列号（见 ``env/obs/lane_semantic.py``）
_SEM_LEFT_MARGIN, _SEM_RIGHT_MARGIN, _SEM_TARGET_VALID, _SEM_D2END = 4, 5, 6, 11
#: ``nav`` 通道的 route_completion 列（末维）
_NAV_PROGRESS_DIM = 11


def path_present_key(key: str) -> str:
    """值键 → present 掩码键（``path_x`` → ``path_x__present``）。"""
    return f"{key}{PATH_PRESENT_SUFFIX}"


def _frame(value: Any, index: int) -> np.ndarray:
    """``obs`` 单帧数组 → ``(F,)``（去 batch/槽位单维）；形状不足 → 空。"""
    arr = np.asarray(value, dtype=np.float32)
    while arr.ndim > 1 and int(arr.shape[0]) == 1:
        arr = arr[0]
    flat = arr.reshape(-1)
    return flat if 0 <= index < flat.size else flat[:0]


def _scalar(value: Any, index: int = 0, *, default: float = 0.0) -> float:
    flat = _frame(value, index)
    if flat.size <= index:
        return float(default)
    return float(flat[index])


def _mask_ok(obs: Mapping[str, Any], name: str) -> bool:
    """通道 mask（``<name>_mask``）首元素 > 0.5；mask 缺键 ⇒ 视为在场。"""
    raw = obs.get(f"{name}_mask")
    if raw is None:
        return True
    flat = np.asarray(raw, dtype=np.float32).reshape(-1)
    return bool(flat.size and float(flat[0]) > 0.5)


def _semantic_field(obs: Mapping[str, Any], column: int) -> tuple[float, float] | None:
    """``lane_semantic`` 单帧列 → ``(value, present)``；通道不在场/维度不足 → ``None``。

    present = 通道 mask ∧ ``target_lane_valid``（列 6，与值同源；target lane 不可用时该帧
    特征全 0，present=0 才不是"静默 0"）。值本身 = 既有回退口径（``lane_semantic`` 列）逐位。
    """
    sem = obs.get("lane_semantic")
    if sem is None:
        return None
    feat = _frame(sem, column)
    if feat.size < 12:
        return None
    value = float(feat[column])
    ready = _mask_ok(obs, "lane_semantic") and float(feat[_SEM_TARGET_VALID]) > 0.5
    return value, (1.0 if ready else 0.0)


def _nav_progress(obs: Mapping[str, Any]) -> tuple[float, float] | None:
    """``nav[:, -1]``（route_completion）→ ``(value, present)``；通道不在场 → ``None``。"""
    nav = obs.get("nav")
    if nav is None:
        return None
    feat = _frame(nav, _NAV_PROGRESS_DIM - 1)
    if feat.size < _NAV_PROGRESS_DIM:
        return None
    return float(feat[_NAV_PROGRESS_DIM - 1]), (1.0 if _mask_ok(obs, "nav") else 0.0)


def path_long_exog_fields(env: Any, obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """组装好的 t0 ``obs`` → 8 列 ``path_*`` + 8 ``__present``（float32 ``(1,)``）。

    需在 ``ObservationBuilder`` 通道全部 build 完之后调用（读 ``od``/``od_presence``/
    ``lane_semantic``/``nav``/``ego`` 的**同帧输出**，保证值 = 下游回退口径逐位）。只产出
    源通道在场的列；``static/dynamic`` 三列恒产出（禁静默 0，无真值 ⇒ present=0）。
    """
    out: dict[str, np.ndarray] = {}

    def _put(key: str, value: float, present: float) -> None:
        out[key] = np.asarray([value], dtype=np.float32)
        out[path_present_key(key)] = np.asarray([present], dtype=np.float32)

    # 0) speed：ego[:,0]（head 同源）
    ego_feats = obs.get("ego")
    if ego_feats is not None:
        _put("path_speed", _scalar(ego_feats, 0), 1.0 if _mask_ok(obs, "ego") else 0.0)

    # 1/2/7) margin / d2end：lane_semantic 既有回退口径（同值同源）
    left = _semantic_field(obs, _SEM_LEFT_MARGIN)
    right = _semantic_field(obs, _SEM_RIGHT_MARGIN)
    if left is not None and right is not None:
        _put("path_left_margin", left[0], left[1])
        _put("path_right_margin", right[0], right[1])
    d2end = _semantic_field(obs, _SEM_D2END)
    if d2end is not None:
        _put("path_distance_to_target_lane_end", d2end[0], d2end[1])

    # 6) nav_progress：nav[:,-1]
    progress = _nav_progress(obs)
    if progress is not None:
        _put("path_nav_progress", progress[0], progress[1])

    # 3/4/5) static / dynamic / ttc：真值公式单一出处（env.obs.long_exog）
    od, presence = od_slots(obs)
    if od is not None and od.shape[0] > 0:
        truth = od_truth_slots(od, presence)
        dyn_distance = float(np.atleast_1d(truth["dyn_distance"])[0])
        dyn_present = bool(np.atleast_1d(truth["dyn_present"])[0])
        _put(
            "path_dynamic_distance",
            dyn_distance if dyn_present else 0.0,
            1.0 if dyn_present else 0.0,
        )
        _put(
            "path_dynamic_ttc",
            float(np.atleast_1d(truth["ttc"])[0]),
            1.0 if bool(np.atleast_1d(truth["ttc_present"])[0]) else 0.0,
        )
    # static 恒产出（诊断列；无真值 ⇒ 0 + present=0，显式主态）
    try:
        gap = static_gap(env, safe_ego(env))
    except Exception:  # noqa: BLE001 - 静态走廊扫描任何异常 = 无观测（scan_static_blocker 自身已捕获）
        gap = None
    _put("path_static_distance", 0.0 if gap is None else float(gap), 0.0 if gap is None else 1.0)
    return out
