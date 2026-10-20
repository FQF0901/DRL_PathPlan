"""Evaluation summaries and uncertainty for closed-loop rows.

The P1/P2 gates use paired per-scene comparisons and bootstrap intervals rather
than a single aggregate over 50 scenes.  This module is intentionally independent
of the simulator and of the action representation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

BOOL_KEYS = (
    "success", "collision", "off_road", "wrong_route", "stuck",
    "timeout", "comfort_violation", "feasible", "fallback",
)
FLOAT_KEYS = (
    "route_completion", "tracking_error_xy", "tracking_error_heading",
    "tracking_error_speed", "steer_saturation", "throttle_saturation",
    "max_curvature", "max_lateral_accel", "max_jerk",
)


def bootstrap_ci(values: Sequence[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """Mean and percentile bootstrap CI for one scalar sample."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    means = x[idx].mean(axis=1)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "lo": float(np.quantile(means, alpha / 2.0)),
        "hi": float(np.quantile(means, 1.0 - alpha / 2.0)),
    }


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Paired difference ``mean(a - b)`` with bootstrap CI.

    Rows must be aligned by scene/seed.  Positive means ``a`` is larger.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired arrays must have same shape, got {x.shape} vs {y.shape}")
    ok = np.isfinite(x) & np.isfinite(y)
    d = (x[ok] - y[ok])
    if len(d) == 0:
        return {"n": 0, "mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, len(d), size=(int(n_boot), len(d)))
    means = d[idx].mean(axis=1)
    return {
        "n": int(len(d)),
        "mean_diff": float(d.mean()),
        "lo": float(np.quantile(means, alpha / 2.0)),
        "hi": float(np.quantile(means, 1.0 - alpha / 2.0)),
    }


def _mean(values: Iterable[Any]) -> float:
    x = np.asarray([float(v) for v in values], dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate closed-loop rows; missing optional keys are ignored."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    out: Dict[str, Any] = {"n": n}
    for key in BOOL_KEYS:
        vals = [bool(r.get(key, False)) for r in rows]
        out[f"{key}_rate"] = float(np.mean(vals))
        out[f"{key}_count"] = int(np.sum(vals))
    for key in FLOAT_KEYS:
        vals = [r[key] for r in rows if key in r and r[key] is not None]
        if vals:
            out[f"{key}_mean"] = _mean(vals)

    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        geo = r.get("geometry", "unknown")
        by[str(geo)].append(r)
        difficulty = r.get("difficulty")
        if difficulty is not None:
            by[f"difficulty:{difficulty}"].append(r)
    out["by_group"] = {
        name: {
            "n": len(group),
            "success_rate": _mean([x.get("success", False) for x in group]),
            "collision_rate": _mean([x.get("collision", False) for x in group]),
            "off_road_rate": _mean([x.get("off_road", False) for x in group]),
            "route_completion_mean": _mean([x.get("route_completion", float("nan")) for x in group]),
        }
        for name, group in sorted(by.items())
    }
    return out


def paired_scene_table(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    *,
    key: str = "success",
) -> Tuple[List[float], List[float], List[int]]:
    """Align two row sets by ``scene_seed`` and return paired values."""
    a = {int(r["scene_seed"]): r for r in rows_a if "scene_seed" in r}
    b = {int(r["scene_seed"]): r for r in rows_b if "scene_seed" in r}
    scenes = sorted(set(a) & set(b))
    va = [float(bool(a[s].get(key, False))) for s in scenes]
    vb = [float(bool(b[s].get(key, False))) for s in scenes]
    return va, vb, scenes


def gate_report(
    rows_a: Sequence[Dict[str, Any]],
    rows_b: Sequence[Dict[str, Any]],
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    """P1/P2 gate statistics: paired success and safety rates."""
    out: Dict[str, Any] = {}
    for key in ("success", "collision", "off_road", "wrong_route", "stuck"):
        va, vb, scenes = paired_scene_table(rows_a, rows_b, key=key)
        out[key] = {
            "n_paired": len(scenes),
            "a_mean": float(np.mean(va)) if va else float("nan"),
            "b_mean": float(np.mean(vb)) if vb else float("nan"),
            "diff": paired_bootstrap(va, vb, seed=seed),
        }
    return out
