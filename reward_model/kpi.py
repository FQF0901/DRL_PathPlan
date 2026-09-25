"""评估 KPI（与 ``config/eval.yaml`` 同口径；纯 Python/NumPy，可脱离 env 单测）。

``KPI_NAMES`` 与 ``config/eval.yaml::kpis`` 顺序一致：
collision / offroad / min_ttc / a_lon / a_lat / jerk / speed_ratio /
solid_line_crossing / speed_limit_violation / route_completion /
per_category_success / overall_success

口径
----
- 输入是普通 dict 的 episode 序列；每个 episode 可给顶层布尔/标量，也可给 ``steps``
  （逐步 info 字典序列）。缺失的导数按 eval 协议从相邻步差分（``dt`` 默认 0.1 s）：
  ``a_lon=(v_t−v_{t−1})/dt``、``a_lat=0.5(v_t+v_{t−1})·Δθ/dt``（Δθ 回绕 (−π,π]）、
  ``jerk=(a_{t}−a_{t−1})/dt``、``speed_ratio=v/限速(m/s)``。
- ``min_ttc``：episode 级或逐步 ``ttc`` 的最小有限值（无前车记 NaN）。
- 分组：主标签 = ``spec.labels.geometry``；带**附带标签**（几何并集 >1、非直行
  maneuver、``control != none``、``traffic`` 非空）的 episode 另计入 ``compound`` 桶；
  ``compound`` 仅报告 + Wilson CI，不参与判定。
- 判定（契约 §8.6）：``n >= per_category_n_min``（默认 30）才判定；弱类
  （baseline < 0.7）用绝对 floor ``max(baseline + 0.15, 0.75)``，否则 ``>= baseline − 0.10``；
  ``n < n_min`` 只报告 + Wilson CI。``overall_success`` 条款 = ``max(baseline − 0.05, 0.70)``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "KPI_NAMES",
    "SOLID_LINE_TYPE_IDS",
    "Thresholds",
    "wilson_ci",
    "episode_kpi",
    "primary_label",
    "incidental_labels",
    "compute_kpis",
    "compute_kpis_by_primary",
    "per_category_target",
    "overall_success_target",
]

KPI_NAMES: tuple[str, ...] = (
    "collision",
    "offroad",
    "min_ttc",
    "a_lon",
    "a_lat",
    "jerk",
    "speed_ratio",
    "solid_line_crossing",
    "speed_limit_violation",
    "route_completion",
    "per_category_success",
    "overall_success",
)

#: 连续实线线型 id（来源 ``env/obs/ld.py::LINE_TYPE_IDS``：2/3 白实线，6/7/8 黄实线）
SOLID_LINE_TYPE_IDS: frozenset[int] = frozenset({2, 3, 6, 7, 8})

#: 无标签/错误终止的兜底键
UNLABELED = "unlabeled"
ERROR_TERMINATION = "error"
NOMINAL_MANEUVERS: frozenset[str] = frozenset({"straight", "none", ""})


@dataclass(frozen=True)
class Thresholds:
    """判定阈值（默认值逐项对应 ``config/eval.yaml::thresholds``）。"""

    per_category_n_min: int = 30
    weak_baseline: float = 0.7
    weak_margin: float = 0.15
    weak_floor: float = 0.75
    normal_margin: float = 0.10
    overall_margin: float = 0.05
    overall_floor: float = 0.70
    speed_limit_tolerance: float = 0.05
    z: float = 1.96


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 二项比例置信区间（闭区间，裁剪到 [0,1]）；``n=0`` 返回 ``(0.0, 0.0)``。"""
    n = int(n)
    successes = int(successes)
    if n <= 0:
        return (0.0, 0.0)
    successes = max(0, min(n, successes))
    p = successes / n
    z2 = float(z) * float(z)
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    half = float(z) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denominator
    return (max(0.0, center - half), min(1.0, center + half))


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _numeric_list(value: Any) -> list[float]:
    """把标量/序列折叠成有限 float 列表（忽略 None/非数/非有限）。"""
    if value is None or isinstance(value, (str, bytes)):
        return []
    if np.isscalar(value):
        number = _optional_float(value)
        return [] if number is None else [number]
    out: list[float] = []
    for item in value:
        number = _optional_float(item)
        if number is not None:
            out.append(number)
    return out


def _step_series(steps: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[float | None]:
    """逐步取首个命中的数值字段（缺省 None，保持与步对齐）。"""
    series: list[float | None] = []
    for step in steps:
        value: float | None = None
        if isinstance(step, Mapping):
            for key in keys:
                if key in step and step[key] is not None:
                    value = _optional_float(step[key])
                    if value is not None:
                        break
        series.append(value)
    return series


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _is_step_solid(step: Mapping[str, Any], solid_type_ids: frozenset[int]) -> bool:
    if _truthy(step.get("solid_line_crossing")) or _truthy(step.get("crossed_solid_line")):
        return True
    if _truthy(step.get("on_white_continuous_line")) or _truthy(
        step.get("on_yellow_continuous_line")
    ):
        return True
    for key in ("left_line_type_id", "right_line_type_id", "line_type_id"):
        value = _optional_float(step.get(key))
        if value is not None and int(value) in solid_type_ids:
            return True
    return False


# --------------------------------------------------------------------------- #
# episode 归一化
# --------------------------------------------------------------------------- #

def _normalize_steps(episode: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = episode.get("steps")
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        return [raw]
    return [step for step in raw if isinstance(step, Mapping)]


def _derive_samples(
    steps: Sequence[Mapping[str, Any]],
    episode: Mapping[str, Any],
    *,
    dt: float,
    solid_type_ids: frozenset[int],
    speed_limit_tolerance: float,
) -> dict[str, Any]:
    """从 step 序列推导/读取连续量与违规事件（返回值均为纯 Python 标量/列表）。"""
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError(f"dt 必须 > 0，实际 {dt}")

    speeds = _step_series(steps, ("speed", "velocity", "v"))
    headings = _step_series(steps, ("heading", "theta", "yaw"))
    limits = _step_series(steps, ("speed_limit_mps", "speed_limit", "limit_mps"))

    # ---- a_lon ----
    a_lon = _numeric_list(_first(episode, ("a_lon_samples", "_a_lon")))
    if not a_lon:
        explicit = [v for v in _step_series(steps, ("a_lon", "longitudinal_accel", "accel_lon")) if v is not None]
        if explicit:
            a_lon = explicit
        else:
            for index in range(1, len(steps)):
                current, previous = speeds[index], speeds[index - 1]
                if current is not None and previous is not None:
                    a_lon.append((current - previous) / dt)

    # ---- a_lat ----
    a_lat = _numeric_list(_first(episode, ("a_lat_samples", "_a_lat")))
    if not a_lat:
        explicit = [v for v in _step_series(steps, ("a_lat", "lateral_accel", "accel_lat")) if v is not None]
        if explicit:
            a_lat = explicit
        else:
            for index in range(1, len(steps)):
                current, previous = speeds[index], speeds[index - 1]
                heading_now, heading_prev = headings[index], headings[index - 1]
                if current is None or previous is None:
                    continue
                if heading_now is None or heading_prev is None:
                    continue
                delta_theta = _wrap_to_pi(heading_now - heading_prev)
                a_lat.append(0.5 * (current + previous) * delta_theta / dt)

    # ---- jerk ----
    jerk = _numeric_list(_first(episode, ("jerk_samples", "_jerk")))
    if not jerk:
        explicit = [v for v in _step_series(steps, ("jerk",)) if v is not None]
        if explicit:
            jerk = explicit
        else:
            for index in range(1, len(a_lon)):
                jerk.append((a_lon[index] - a_lon[index - 1]) / dt)

    # ---- speed_ratio ----
    speed_ratio = _numeric_list(_first(episode, ("speed_ratio_samples", "_speed_ratio")))
    ratio_flags: list[bool] = []
    if not speed_ratio:
        explicit = [
            v for v in _step_series(steps, ("speed_ratio", "ratio")) if v is not None
        ]
        if explicit:
            speed_ratio = explicit
        else:
            for index, speed in enumerate(speeds):
                limit = limits[index] if index < len(limits) else None
                if speed is not None and limit is not None and limit > 0.0:
                    speed_ratio.append(speed / limit)
    for step in steps:
        if _truthy(step.get("speed_limit_violation")):
            ratio_flags.append(True)

    # ---- min_ttc ----
    min_ttc = _optional_float(_first(episode, ("min_ttc", "ttc_min")))
    if min_ttc is None:
        ttc_values: list[float] = []
        for value in _step_series(steps, ("ttc", "min_ttc", "ttc_min")):
            if value is not None:
                ttc_values.append(value)
        min_ttc = min(ttc_values) if ttc_values else float("nan")

    # ---- 连续实线 ----
    solid_steps = sum(1 for step in steps if _is_step_solid(step, solid_type_ids))
    explicit_solid = _first(episode, ("solid_line_crossing", "crossed_solid_line"))
    count_field = _first(episode, ("solid_line_crossing_count",))
    if count_field is not None:
        solid_count = int(count_field)
        solid_crossing = solid_count > 0
    else:
        solid_crossing = bool(_truthy(explicit_solid)) or solid_steps > 0
        solid_count = max(solid_steps, 1 if solid_crossing else 0)

    # ---- 超速 ----
    tolerance = float(speed_limit_tolerance)
    violating_ratios = sum(1 for ratio in speed_ratio if ratio > 1.0 + tolerance)
    explicit_violation = _first(episode, ("speed_limit_violation",))
    explicit_count = _first(episode, ("speed_limit_violation_count",))
    if explicit_count is not None:
        violation_count = int(explicit_count)
    else:
        violation_count = violating_ratios + sum(ratio_flags)
    speed_violation = bool(_truthy(explicit_violation)) or violation_count > 0
    if explicit_count is None and not speed_violation:
        violation_count = 0
    ratio_denominator = len(speed_ratio) + len(ratio_flags)
    violation_rate = (
        float(violation_count) / float(ratio_denominator) if ratio_denominator > 0 else float("nan")
    )

    return {
        "a_lon": a_lon,
        "a_lat": a_lat,
        "jerk": jerk,
        "speed_ratio": speed_ratio,
        "min_ttc": min_ttc,
        "solid_line_crossing": solid_crossing,
        "solid_line_crossing_count": solid_count,
        "speed_limit_violation": speed_violation,
        "speed_limit_violation_count": violation_count,
        "speed_limit_violation_rate": violation_rate,
    }


def episode_kpi(
    episode: Mapping[str, Any],
    *,
    dt: float = 0.1,
    solid_line_type_ids: Iterable[int] = SOLID_LINE_TYPE_IDS,
    speed_limit_tolerance: float = 0.05,
) -> dict[str, Any]:
    """把单个 episode（普通 dict）归一化成 KPI 记录（不修改输入）。"""
    steps = _normalize_steps(episode)
    solid_ids = frozenset(int(item) for item in solid_line_type_ids)
    samples = _derive_samples(
        steps,
        episode,
        dt=dt,
        solid_type_ids=solid_ids,
        speed_limit_tolerance=speed_limit_tolerance,
    )

    termination = _first(episode, ("termination", "reason"))
    termination = str(termination) if termination is not None else ""

    success_raw = _first(episode, ("success", "arrive_dest"))
    if success_raw is None:
        success = termination == "arrive_dest"
    else:
        success = _truthy(success_raw)

    collision_raw = _first(episode, ("collision",))
    collision = _truthy(collision_raw) or termination == "collision"
    off_road_raw = _first(episode, ("off_road", "offroad"))
    off_road = _truthy(off_road_raw) or termination == "out_of_road"

    route_completion = _optional_float(_first(episode, ("route_completion", "rc")))
    if route_completion is None:
        max_rc = float("nan")
        for step in steps:
            value = _optional_float(_first(step, ("route_completion", "rc")))
            if value is not None:
                max_rc = value if math.isnan(max_rc) else max(max_rc, value)
        route_completion = 0.0 if math.isnan(max_rc) else max_rc

    return {
        "success": bool(success),
        "collision": bool(collision),
        "off_road": bool(off_road),
        "termination": termination or "unknown",
        "route_completion": float(route_completion),
        "n_steps": len(steps),
        "min_ttc": float(samples["min_ttc"]),
        "a_lon": samples["a_lon"],
        "a_lat": samples["a_lat"],
        "jerk": samples["jerk"],
        "speed_ratio": samples["speed_ratio"],
        "solid_line_crossing": bool(samples["solid_line_crossing"]),
        "solid_line_crossing_count": int(samples["solid_line_crossing_count"]),
        "speed_limit_violation": bool(samples["speed_limit_violation"]),
        "speed_limit_violation_count": int(samples["speed_limit_violation_count"]),
        "speed_limit_violation_rate": float(samples["speed_limit_violation_rate"]),
        "primary_label": primary_label(episode),
        "incidental_labels": incidental_labels(episode),
    }


# --------------------------------------------------------------------------- #
# 分组标签（primary = spec.labels.geometry；附带标签 -> compound）
# --------------------------------------------------------------------------- #

def primary_label(episode: Mapping[str, Any]) -> str:
    """主标签 = ``labels.geometry``（兼容顶层 ``primary_label``），缺失记 ``unlabeled``。"""
    labels = episode.get("labels")
    if isinstance(labels, Mapping) and labels.get("geometry") is not None:
        return str(labels["geometry"])
    explicit = episode.get("primary_label")
    if explicit is not None:
        return str(explicit)
    return UNLABELED


def incidental_labels(episode: Mapping[str, Any]) -> list[str]:
    """附带标签：非主标签的几何标签、非直行 maneuver、control 事件、traffic pattern。

    这些标签不进主标签分组（避免与冻结基线口径混淆），统一落到 ``compound`` 桶报告。
    """
    labels = episode.get("labels")
    labels = labels if isinstance(labels, Mapping) else {}
    primary = primary_label(episode)
    collected: set[str] = set()

    geometry = episode.get("geometry")
    if isinstance(geometry, (list, tuple, set, frozenset)):
        collected.update(str(item) for item in geometry if str(item) != primary)
    maneuver = labels.get("maneuver", episode.get("maneuver"))
    if isinstance(maneuver, (list, tuple, set, frozenset)):
        collected.update(str(item) for item in maneuver if str(item) not in NOMINAL_MANEUVERS)
    elif maneuver is not None and str(maneuver) not in NOMINAL_MANEUVERS:
        collected.add(str(maneuver))
    control = labels.get("control", episode.get("control"))
    if control is not None and str(control) != "none":
        collected.add(str(control))
    traffic = labels.get("traffic", episode.get("traffic"))
    if isinstance(traffic, (list, tuple, set, frozenset)):
        collected.update(str(item) for item in traffic if str(item))
    elif traffic is not None and str(traffic):
        collected.add(str(traffic))
    collected.discard("")
    return sorted(collected)


# --------------------------------------------------------------------------- #
# 汇总与判定
# --------------------------------------------------------------------------- #

def _mean(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _percentile_abs(values: Sequence[float], q: float) -> float:
    finite = [abs(float(v)) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.percentile(finite, q)) if finite else float("nan")


def _percentile(values: Sequence[float], q: float) -> float:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.percentile(finite, q)) if finite else float("nan")


def _pool(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    pooled: list[float] = []
    for record in records:
        pooled.extend(record.get(key) or [])
    return pooled


def _summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把 KPI 记录池化成指标（成功率 + Wilson CI、池化样本均值/p95 等）。"""
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "n_error": 0,
            "success_rate": float("nan"),
            "success_wilson": {"lo": 0.0, "hi": 0.0, "successes": 0, "n": 0},
            "collision_rate": float("nan"),
            "offroad_rate": float("nan"),
            "min_ttc_mean": float("nan"),
            "min_ttc_p05": float("nan"),
            "a_lon_mean": float("nan"),
            "a_lon_abs_p95": float("nan"),
            "a_lat_mean": float("nan"),
            "a_lat_abs_p95": float("nan"),
            "jerk_mean": float("nan"),
            "jerk_abs_p95": float("nan"),
            "speed_ratio_mean": float("nan"),
            "speed_ratio_p95": float("nan"),
            "solid_line_crossing_rate": float("nan"),
            "speed_limit_violation_rate": float("nan"),
            "speed_limit_violation_episode_rate": float("nan"),
            "route_completion_mean": float("nan"),
            "route_completion_p10": float("nan"),
            "terminations": {},
        }

    successes = sum(1 for record in records if record["success"])
    collisions = sum(1 for record in records if record["collision"])
    off_roads = sum(1 for record in records if record["off_road"])
    min_ttcs = [float(record["min_ttc"]) for record in records]
    route_completions = [float(record["route_completion"]) for record in records]
    violation_rates = [float(record["speed_limit_violation_rate"]) for record in records]

    terminations: dict[str, int] = {}
    for record in records:
        key = str(record["termination"])
        terminations[key] = terminations.get(key, 0) + 1

    return {
        "n": n,
        "n_error": int(terminations.get(ERROR_TERMINATION, 0)),
        "success_rate": successes / n,
        "success_wilson": {
            "lo": wilson_ci(successes, n)[0],
            "hi": wilson_ci(successes, n)[1],
            "successes": successes,
            "n": n,
        },
        "collision_rate": collisions / n,
        "offroad_rate": off_roads / n,
        "min_ttc_mean": _mean(min_ttcs),
        "min_ttc_p05": _percentile(min_ttcs, 5.0),
        "a_lon_mean": _mean(_pool(records, "a_lon")),
        "a_lon_abs_p95": _percentile_abs(_pool(records, "a_lon"), 95.0),
        "a_lat_mean": _mean(_pool(records, "a_lat")),
        "a_lat_abs_p95": _percentile_abs(_pool(records, "a_lat"), 95.0),
        "jerk_mean": _mean(_pool(records, "jerk")),
        "jerk_abs_p95": _percentile_abs(_pool(records, "jerk"), 95.0),
        "speed_ratio_mean": _mean(_pool(records, "speed_ratio")),
        "speed_ratio_p95": _percentile(_pool(records, "speed_ratio"), 95.0),
        "solid_line_crossing_rate": sum(
            1 for record in records if record["solid_line_crossing"]
        )
        / n,
        "speed_limit_violation_rate": _mean(violation_rates),
        "speed_limit_violation_episode_rate": sum(
            1 for record in records if record["speed_limit_violation"]
        )
        / n,
        "route_completion_mean": float(np.mean(route_completions)),
        "route_completion_p10": _percentile(route_completions, 10.0),
        "terminations": dict(sorted(terminations.items())),
    }


def per_category_target(
    baseline: float | None, thresholds: Thresholds | None = None
) -> tuple[float | None, str]:
    """分组成功目标：弱类（baseline < 0.7）用绝对 floor，否则相对 ``baseline − 0.10``。"""
    th = thresholds or Thresholds()
    if baseline is None:
        return None, "no_baseline"
    value = float(baseline)
    if value < th.weak_baseline:
        return max(value + th.weak_margin, th.weak_floor), "weak_floor"
    return value - th.normal_margin, "relative"


def overall_success_target(
    baseline: float | None, thresholds: Thresholds | None = None
) -> float | None:
    """``overall_success`` 条款：``max(baseline − 0.05, 0.70)``。"""
    th = thresholds or Thresholds()
    if baseline is None:
        return None
    return max(float(baseline) - th.overall_margin, th.overall_floor)


def _baseline_success(value: Any) -> float | None:
    """baseline 允许是标量，或 ``{"success": x}`` / ``{"success_rate": x}`` 映射。"""
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("success", "success_rate"):
            if value.get(key) is not None:
                number = _optional_float(value.get(key))
                if number is not None:
                    return number
        return None
    return _optional_float(value)


def _judge_category(summary: Mapping[str, Any], baseline: float | None, th: Thresholds) -> dict:
    ci = dict(summary["success_wilson"])
    n = int(summary["n"])
    if n < int(th.per_category_n_min):
        return {
            "judged": False,
            "rule": "n_below_min",
            "n_min": int(th.per_category_n_min),
            "target": None,
            "passed": None,
            "wilson_ci": ci,
        }
    target, rule = per_category_target(baseline, th)
    if target is None:
        return {
            "judged": False,
            "rule": "no_baseline",
            "n_min": int(th.per_category_n_min),
            "target": None,
            "passed": None,
            "wilson_ci": ci,
        }
    return {
        "judged": True,
        "rule": rule,
        "baseline": float(baseline),
        "target": float(target),
        "passed": bool(summary["success_rate"] >= target),
        "wilson_ci": ci,
    }


def compute_kpis(
    episodes: Iterable[Mapping[str, Any]],
    *,
    baseline: float | Mapping[str, Any] | None = None,
    thresholds: Thresholds | None = None,
    dt: float = 0.1,
    solid_line_type_ids: Iterable[int] = SOLID_LINE_TYPE_IDS,
    speed_limit_tolerance: float | None = None,
) -> dict[str, Any]:
    """整体 KPI（池化）+ ``overall_success`` 判定。"""
    th = thresholds or Thresholds()
    tolerance = th.speed_limit_tolerance if speed_limit_tolerance is None else speed_limit_tolerance
    records = [
        episode_kpi(
            episode,
            dt=dt,
            solid_line_type_ids=solid_line_type_ids,
            speed_limit_tolerance=tolerance,
        )
        for episode in episodes
    ]
    summary = _summarize(records)
    baseline_value = _baseline_success(baseline)
    target = overall_success_target(baseline_value, th)
    summary["judgment"] = {
        "clause": "overall_success",
        "baseline": baseline_value,
        "target": target,
        "passed": None if target is None else bool(summary["success_rate"] >= target),
    }
    summary["kpi_names"] = list(KPI_NAMES)
    return summary


def compute_kpis_by_primary(
    episodes: Iterable[Mapping[str, Any]],
    *,
    baselines: Mapping[str, Any] | None = None,
    thresholds: Thresholds | None = None,
    dt: float = 0.1,
    solid_line_type_ids: Iterable[int] = SOLID_LINE_TYPE_IDS,
    speed_limit_tolerance: float | None = None,
) -> dict[str, Any]:
    """按主标签（``labels.geometry``）分组 KPI + ``compound`` 附带标签桶 + 总体条款。

    - 每个主标签：指标 + 判定（``n >= n_min``；弱类绝对 floor）。
    - ``compound``：携带附带标签的 episode（几何并集 >1 / 非直行 maneuver / control 事件 /
      traffic pattern），仅报告 + Wilson CI，不判定。
    - ``overall``：全量指标 + ``overall_success`` 条款。
    """
    th = thresholds or Thresholds()
    tolerance = th.speed_limit_tolerance if speed_limit_tolerance is None else speed_limit_tolerance
    records = [
        episode_kpi(
            episode,
            dt=dt,
            solid_line_type_ids=solid_line_type_ids,
            speed_limit_tolerance=tolerance,
        )
        for episode in episodes
    ]

    buckets: dict[str, list[dict[str, Any]]] = {}
    compound_records: list[dict[str, Any]] = []
    for record in records:
        buckets.setdefault(str(record["primary_label"]), []).append(record)
        if record["incidental_labels"]:
            compound_records.append(record)

    baselines = baselines or {}
    by_primary: dict[str, Any] = {}
    for label in sorted(buckets):
        summary = _summarize(buckets[label])
        summary["judgment"] = _judge_category(summary, _baseline_success(baselines.get(label)), th)
        by_primary[label] = summary

    compound = _summarize(compound_records)
    compound["judgment"] = {
        "judged": False,
        "rule": "compound_report_only",
        "target": None,
        "passed": None,
        "wilson_ci": dict(compound["success_wilson"]),
    }

    overall = _summarize(records)
    baseline_overall = _baseline_success(baselines.get("overall", baselines.get("_overall")))
    target = overall_success_target(baseline_overall, th)
    overall["judgment"] = {
        "clause": "overall_success",
        "baseline": baseline_overall,
        "target": target,
        "passed": None if target is None else bool(overall["success_rate"] >= target),
    }

    return {
        "overall": overall,
        "by_primary": by_primary,
        "compound": compound,
        "kpi_names": list(KPI_NAMES),
    }
