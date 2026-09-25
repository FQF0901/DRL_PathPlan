#!/usr/bin/env python3
"""独立进程、确定性的冻结验证集评测（P2/N5，契约 §5 + §8.1/§8.6）。

职责
----
1. 在**冻结验证集**（默认 ``env/specs/scenarios_val.json``，1000 条）上评测策略；
   ``--policy baseline`` = 规则基线 ``PurePursuitIDMPolicy``（同 harness/同种子，复现
   ``runs/baseline_eval/val_reference.json``）；``--policy ckpt`` = 加载 N1 的
   ``DrivingModel`` 权重并用 N3 的 ``env.tracking`` 执行动作：
   ``--tracker exact``（默认，阶段 A/B：每 0.1 s 子步置于插值位姿，衡量规划轨迹本身）
   或 ``--tracker lqr``（阶段 C 闭环，含控制器跟踪误差）；
2. **内存纪律（§8.1）**：运行前程序化检查 ``MemAvailable >= eval.mem_available_floor_mb``，
   按 ``eval.train_pool_policy`` 调可选的训练池暂停钩子；spawn 进程池按
   ``eval.recycle_every_specs``（每 worker 每代处理的 spec 数）重建池 —— 照抄
   ``env/scenario/validator.py::validate_specs`` 的 ``specs_per_worker_per_pool`` 实现
   （MetaDrive 每 ``build_env``+``close`` 残留 ~3.5MB/spec）；
3. **KPI 口径**（config/eval.yaml 冻结）：success / collision / off_road / min_ttc /
   a_lon / a_lat / jerk / speed_ratio / solid_line_crossing / speed_limit_violation /
   route_completion；按 **primary 标签**（``spec.labels.geometry``）分组，其余附带标签单列
   ``compound``（报告口径，不判定）；每组 n >= ``per_category_n_min`` 才判定，并给 Wilson 95% CI；
4. 与冻结基线（``runs/baseline_eval/val_reference.json`` + ``..._by_primary.json``）比较，
   生成阈值判定：overall_success、collision/offroad ε 保护、route_completion、speed_ratio
   效率守卫、a_lat_mean/p95，以及弱类 floor（``target = max(baseline+0.15, 0.75)``，baseline<0.7）；
5. 输出 ``<out>/<name>/metrics.json``（overall + by_primary + compound + by_difficulty +
   by_geometry + verdict）与 ``episodes.csv``（逐 episode 明细）。

关键实现事实（先核对源码再写码）
------------------------------
- 策略注册走官方 agent 调用路径：``engine.add_policy(ego.id, PurePursuitIDMPolicy, ...)``
  （``engine/base_engine.py:98-104``），注册后 ``env.step`` 传入的占位动作被忽略；
- ``env.step`` = 0.1 s（``physics_world_step_size=0.02 × decision_repeat=5``，envs/base_env.py:188-190）；
- 加速度用相邻步状态差分（**不用** ``info["acceleration"]``，它是油门动作值）；
- ``solid_line_crossing``：MetaDrive 的 ``ego.on_white_continuous_line`` /
  ``on_yellow_continuous_line`` 只在 ``_state_check`` 置 True、从不复位
  （``component/vehicle/base_vehicle.py:56-59, 741-760``），因此仅对"episode 内是否压过实线"
  这一 **episode 级** 语义有效（每次评测新建 env，不会跨 episode 污染）；
- ``min_ttc``：与 ``env/obs/od.py`` 同口径（自车系相对位置/速度，仅前方 |lat|<=2m 且接近的
  车辆），cap = 10 s；无风险目标记 cap；
- 单条 spec 失败记 ``termination="error"``（计为失败样本、单列 ``n_error``），不中断整批。

用法
----
    tools/venv-python pipeline/eval_runner.py --policy baseline --limit 10 --workers 1
    tools/venv-python -m pipeline.eval_runner --policy ckpt --ckpt runs/train/stageA/best.pt
    # 独立进程由 tools/test.py 调用（评测不占训练进程；worker 由 spawn 池承载）

import 时只有 stdlib + numpy（metadrive/torch 全部延迟到 worker 内），保证纯测试可导入。
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# 允许 `python pipeline/eval_runner.py` 直接运行（此时 sys.path[0] 是 pipeline/）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:  # PyYAML 在 venv 内必装；纯测试环境缺失时仅 load_config 报错
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

__all__ = [
    "DEFAULT_BASELINE_REF",
    "DEFAULT_MAX_STEPS",
    "KPI_DEFINITIONS",
    "build_report",
    "category_target",
    "evaluate_verdicts",
    "group_episodes",
    "is_compound_spec",
    "load_config",
    "main",
    "primary_label",
    "read_mem_available_mb",
    "set_train_pool_hook",
    "summarize",
    "wilson_ci",
]

# --------------------------------------------------------------------------- #
# 常量与 KPI 定义
# --------------------------------------------------------------------------- #

DEFAULT_MAX_STEPS = 1000          # 100 s @ 0.1 s/step（与冻结基线口径一致）
DEFAULT_WORKERS = 2               # 保守默认：eval 与训练池互斥（§8.1），长跑显式传 --workers
DEFAULT_MEM_FLOOR_MB = 3000.0     # config/eval.yaml::eval.mem_available_floor_mb
DEFAULT_RECYCLE_EVERY_SPECS = 150  # config/eval.yaml::eval.recycle_every_specs（每 worker 每代）
MEM_HARD_FLOOR_RATIO = 0.5        # MemAvailable < floor*0.5 时拒绝开跑（连 1 个 env 都不安全）
PAUSE_WAIT_S = 30.0               # 暂停训练池后等待 MemAvailable 回升的最长时间
DEFAULT_TTC_CAP_S = 10.0          # min_ttc cap（无风险目标记 cap）
DEFAULT_TTC_LATERAL_M = 2.0       # min_ttc 只统计自车前方 |lat| <= 2m 的车辆
DEFAULT_SPEED_LIMIT_FALLBACK = 13.9
WILSON_Z = 1.96                   # 95% Wilson CI
PRIMARY_GROUP_MIN_N = 30          # config/eval.yaml::thresholds.per_category_n_min
WEAK_BASELINE_SUCCESS = 0.7       # 弱类判定阈值
WEAK_GAIN = 0.15                  # 弱类 target = max(baseline + WEAK_GAIN, WEAK_FLOOR)
WEAK_FLOOR = 0.75
NON_WEAK_MARGIN = 0.10            # 非弱类 target = baseline - NON_WEAK_MARGIN
OVERALL_MARGIN = 0.05             # overall target = max(baseline - 0.05, OVERALL_ABS_FLOOR)
OVERALL_ABS_FLOOR = 0.70
ZERO_BASELINE_EPSILON = 0.01      # 零基线保护 ε（collision/off_road）
DEFAULT_BASELINE_REF = "runs/baseline_eval/val_reference.json"
DEFAULT_BASELINE_PRIMARY_REF = "runs/baseline_eval/val_reference_by_primary.json"

KPI_DEFINITIONS: Dict[str, str] = {
    "success": "episode 内 info['arrive_dest'] 曾为 True（envs/metadrive_env.py:218-231）",
    "collision": "episode 内 info['crash']（= crash_vehicle/object/building/human/sidewalk 的或）曾为 True",
    "off_road": "episode 内 info['out_of_road'] 曾为 True",
    "min_ttc": "逐步最小 TTC（s，自车系：仅前方 |lat|<=2m 且接近的车辆，cap=10s）；episode 取 min、"
               "组内取均值",
    "a_lon": "相邻步差分 a_lon=(v_t-v_{t-1})/dt，dt=0.1s（**不用** info['acceleration']）",
    "a_lat": "相邻步差分 a_lat=0.5(v_t+v_{t-1})*Δθ/dt，Δθ 回绕到 (-π,π]",
    "jerk": "相邻步 a_lon 差分 jerk=(a_t-a_{t-1})/dt",
    "speed_ratio": "逐步 v/车道限速(m/s)；限速未设置(>=1000)时用 13.9 m/s 兜底",
    "solid_line_crossing": "episode 内 ego.on_white/yellow_continuous_line 曾为 True（MetaDrive 该标志"
                           "只置位不复位，故仅 episode 级有效）",
    "speed_limit_violation": "episode 内任一步 v > 车道限速(m/s) 记为 True；CSV 另给 step_rate",
    "route_completion": "episode 最后一步 info['route_completion']",
    "grouping": "by_primary 按 spec.labels.geometry（判定口径）；compound = 附带（非 primary/straight）"
                "几何标签的 spec 汇总（报告口径，不判定）；by_geometry 为逐标签视图（可重复计数，报告口径）",
    "error": "该 spec 实例化/推进抛异常：计为失败样本，不进 KPI 样本池，n_error 单列",
}

# --------------------------------------------------------------------------- #
# 训练池暂停钩子（optional，§8.1）
# --------------------------------------------------------------------------- #

#: 训练协调器（N4）可在同进程内注册：hook() 暂停训练 env 池并返回 True；返回 False 表示
#: 无法暂停（例如没有训练池）。未注册时视为"本机只有评测任务"，仅依赖 MemAvailable 检查。
_TRAIN_POOL_HOOK: Optional[Callable[[], Any]] = None


def set_train_pool_hook(hook: Optional[Callable[[], Any]]) -> None:
    """注册/清除训练池暂停钩子（optional；见模块 docstring §8.1）。"""
    global _TRAIN_POOL_HOOK
    _TRAIN_POOL_HOOK = hook


def read_mem_available_mb(path: str = "/proc/meminfo") -> Optional[float]:
    """读 ``MemAvailable``（MB）；非 Linux / 读取失败返回 None（调用方跳过检查）。"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------- #
# KPI 统计（纯函数，供 tests/test_kpi_grouping.py 直接覆盖）
# --------------------------------------------------------------------------- #

def wilson_ci(k: int, n: int, z: float = WILSON_Z) -> Tuple[float, float]:
    """成功 k/n 的 Wilson 95% 置信区间（score interval）。

    n == 0 或 k 非法时返回 ``(nan, nan)``。Wilson 区间在小样本/极端比例下仍落在 [0,1]，
    优于 Wald 区间（per-category n>=30 的协议需要它）。
    """
    if n <= 0 or k < 0 or k > n:
        return (float("nan"), float("nan"))
    p = k / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def category_target(baseline_success: float) -> float:
    """分组 success 目标（config/eval.yaml::thresholds.per_category_success）。

    弱类（baseline < 0.7）用绝对 floor：``max(baseline + 0.15, 0.75)``；否则 ``baseline - 0.10``。
    """
    if baseline_success < WEAK_BASELINE_SUCCESS:
        return max(baseline_success + WEAK_GAIN, WEAK_FLOOR)
    return baseline_success - NON_WEAK_MARGIN


def primary_label(spec: Any) -> str:
    """primary 分组键 = ``spec.labels["geometry"]``（缺失时退回 geometry[0]/unlabeled）。"""
    labels = getattr(spec, "labels", None)
    if isinstance(labels, Mapping):
        value = labels.get("geometry")
        if value:
            return str(value)
    geometry = getattr(spec, "geometry", None) or []
    return str(geometry[0]) if geometry else "unlabeled"


def extra_geometry_labels(spec: Any) -> List[str]:
    """除 primary 与 ubiquitous 的 straight 之外的几何标签（sorted，去重）。"""
    primary = primary_label(spec)
    geometry = getattr(spec, "geometry", None) or []
    return sorted({str(label) for label in geometry} - {primary, "straight"})


def is_compound_spec(spec: Any) -> bool:
    """是否为复合/附带标签 spec（compound 报告口径，见 KPI_DEFINITIONS['grouping']）。"""
    return bool(extra_geometry_labels(spec))


def group_episodes(episodes: Sequence[Mapping[str, Any]], key: str) -> Dict[str, List[Mapping[str, Any]]]:
    """按 episode 字段分组（保持首次出现顺序）；key 缺失记 ``unlabeled``。"""
    groups: Dict[str, List[Mapping[str, Any]]] = {}
    for episode in episodes:
        name = str(episode.get(key) or "unlabeled")
        groups.setdefault(name, []).append(episode)
    return groups


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _p95_abs(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(array), 95.0))


def _pool_samples(episodes: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    arrays = [np.asarray(ep[key], dtype=np.float64) for ep in episodes if ep.get(key)]
    if not arrays:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(arrays)


def _rate(success_flags: Sequence[bool]) -> float:
    if not success_flags:
        return float("nan")
    return float(sum(1 for flag in success_flags if flag) / len(success_flags))


def summarize(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """把一组 episode 聚合成 KPI（率 + Wilson CI + 池化样本统计）。空组返回 ``{"n": 0}``。"""
    if not episodes:
        return {"n": 0}

    def _wilson(key: str) -> Tuple[float, float]:
        count = sum(1 for ep in episodes if ep.get(key))
        return wilson_ci(count, len(episodes))

    a_lon = _pool_samples(episodes, "_a_lon")
    a_lat = _pool_samples(episodes, "_a_lat")
    jerk = _pool_samples(episodes, "_jerk")
    speed_ratio = _pool_samples(episodes, "_speed_ratio")

    terminations: Dict[str, int] = {}
    for episode in episodes:
        name = str(episode.get("termination", "other"))
        terminations[name] = terminations.get(name, 0) + 1

    def _per_episode_mean(key: str) -> float:
        return _finite_mean([ep[key] for ep in episodes if isinstance(ep.get(key), (int, float))])

    speed_ratio_mean = _finite_mean(speed_ratio) if speed_ratio.size else _per_episode_mean("speed_ratio_mean")
    route_completion_mean = _finite_mean(
        [ep["route_completion"] for ep in episodes if isinstance(ep.get("route_completion"), (int, float))]
    )
    min_ttc_values = [ep["min_ttc"] for ep in episodes if isinstance(ep.get("min_ttc"), (int, float))]
    return {
        "n": len(episodes),
        "n_error": int(terminations.get("error", 0)),
        "success_rate": _rate([bool(ep.get("success")) for ep in episodes]),
        "success_wilson": list(_wilson("success")),
        "collision_rate": _rate([bool(ep.get("collision")) for ep in episodes]),
        "collision_wilson": list(_wilson("collision")),
        "off_road_rate": _rate([bool(ep.get("off_road")) for ep in episodes]),
        "off_road_wilson": list(_wilson("off_road")),
        "solid_line_crossing_rate": _rate([bool(ep.get("solid_line_crossing")) for ep in episodes]),
        "solid_line_crossing_wilson": list(_wilson("solid_line_crossing")),
        "speed_limit_violation_rate": _rate([bool(ep.get("speed_limit_violation")) for ep in episodes]),
        "speed_limit_violation_wilson": list(_wilson("speed_limit_violation")),
        "route_completion_mean": route_completion_mean,
        "min_ttc_mean": _finite_mean(min_ttc_values),
        "min_ttc_min": float(min(min_ttc_values)) if min_ttc_values else float("nan"),
        "a_lon_mean": _finite_mean(a_lon) if a_lon.size else _per_episode_mean("a_lon_mean"),
        "a_lon_abs_p95": _p95_abs(a_lon) if a_lon.size else _per_episode_mean("a_lon_abs_p95"),
        "a_lat_mean": _finite_mean(a_lat) if a_lat.size else _per_episode_mean("a_lat_mean"),
        "a_lat_abs_p95": _p95_abs(a_lat) if a_lat.size else _per_episode_mean("a_lat_abs_p95"),
        "jerk_mean": _finite_mean(jerk) if jerk.size else _per_episode_mean("jerk_mean"),
        "jerk_abs_p95": _p95_abs(jerk) if jerk.size else _per_episode_mean("jerk_abs_p95"),
        "speed_ratio_mean": speed_ratio_mean,
        "mean_speed_mps": _per_episode_mean("mean_speed_mps"),
        "mean_steps": _per_episode_mean("steps"),
        "mean_duration_s": _per_episode_mean("duration_s"),
        "terminations": dict(sorted(terminations.items())),
    }


# --------------------------------------------------------------------------- #
# 判定（与冻结基线比较；config/eval.yaml::thresholds）
# --------------------------------------------------------------------------- #

def _opt_num(container: Optional[Mapping[str, Any]], *keys: str) -> Optional[float]:
    """安全取嵌套数值；缺失/None/非数值返回 None。"""
    current: Any = container
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    value = float(current)
    return value if math.isfinite(value) else None


def _check_ge(value: Optional[float], target: Optional[float], rule: str) -> Dict[str, Any]:
    if value is None or target is None:
        return {"value": value, "target": target, "passed": None, "rule": rule, "reason": "缺少可比数据"}
    return {"value": value, "target": target, "passed": bool(value >= target), "rule": rule}


def _check_le(value: Optional[float], target: Optional[float], rule: str) -> Dict[str, Any]:
    if value is None or target is None:
        return {"value": value, "target": target, "passed": None, "rule": rule, "reason": "缺少可比数据"}
    return {"value": value, "target": target, "passed": bool(value <= target), "rule": rule}


def judge_group(view: Mapping[str, Any], baseline_success: Optional[float]) -> Dict[str, Any]:
    """按 primary 组判定 success（n < 30 或缺少基线 → 仅报告 + Wilson CI，不判定）。"""
    n = int(view.get("n") or 0)
    success = view.get("success_rate")
    base = baseline_success
    if base is None:
        return {"judged": False, "passed": None, "n": n, "baseline_success": None,
                "reason": "缺少该主标签的冻结基线，仅报告"}
    if n < PRIMARY_GROUP_MIN_N:
        return {"judged": False, "passed": None, "n": n, "baseline_success": float(base),
                "reason": f"n={n} < per_category_n_min={PRIMARY_GROUP_MIN_N}，仅报告 + Wilson CI"}
    target = category_target(float(base))
    weak = bool(float(base) < WEAK_BASELINE_SUCCESS)
    return {
        "judged": True,
        "passed": bool(isinstance(success, (int, float)) and success >= target),
        "n": n,
        "success_rate": success,
        "baseline_success": float(base),
        "weak": weak,
        "target": target,
        "rule": ("weak(baseline<0.7): target=max(baseline+0.15, 0.75)" if weak
                 else "baseline-0.10"),
        "wilson_ci": view.get("success_wilson"),
    }


def evaluate_verdicts(
    overall: Mapping[str, Any],
    primary_views: Mapping[str, Mapping[str, Any]],
    baseline_doc: Optional[Mapping[str, Any]] = None,
    baseline_primary_doc: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """生成对冻结基线的判定字典（见 config/eval.yaml::thresholds 的公式）。

    ``baseline_doc`` 可以是完整的 ``val_reference.json``（含 ``overall``），也可以直接是
    overall 子字典（便于调用方/单测复用）。
    """
    base_overall: Optional[Mapping[str, Any]] = baseline_doc
    if isinstance(baseline_doc, Mapping) and isinstance(baseline_doc.get("overall"), Mapping):
        base_overall = baseline_doc["overall"]
    base_success = _opt_num(base_overall, "success_rate")
    base_collision = _opt_num(base_overall, "collision_rate")
    base_offroad = _opt_num(base_overall, "off_road_rate")
    base_rc = _opt_num(base_overall, "route_completion_mean")
    base_speed_ratio = _opt_num(base_overall, "speed_ratio_mean")
    base_a_lat_mean = _opt_num(base_overall, "a_lat_mean")
    base_a_lat_p95 = _opt_num(base_overall, "a_lat_abs_p95")

    checks: Dict[str, Dict[str, Any]] = {}
    checks["overall_success"] = _check_ge(
        _opt_num(overall, "success_rate"),
        max(base_success - OVERALL_MARGIN, OVERALL_ABS_FLOOR) if base_success is not None else None,
        "success_rate >= max(baseline - 0.05, 0.70)",
    )
    checks["collision_rate"] = _check_le(
        _opt_num(overall, "collision_rate"),
        max(base_collision, ZERO_BASELINE_EPSILON) if base_collision is not None else None,
        "collision_rate <= max(baseline, 0.01)",
    )
    checks["off_road_rate"] = _check_le(
        _opt_num(overall, "off_road_rate"),
        max(base_offroad, ZERO_BASELINE_EPSILON) if base_offroad is not None else None,
        "off_road_rate <= max(baseline, 0.01)",
    )
    checks["route_completion"] = _check_ge(
        _opt_num(overall, "route_completion_mean"),
        base_rc - OVERALL_MARGIN if base_rc is not None else None,
        "route_completion_mean >= baseline - 0.05",
    )
    checks["speed_ratio"] = _check_ge(
        _opt_num(overall, "speed_ratio_mean"),
        base_speed_ratio * 0.90 if base_speed_ratio is not None else None,
        "speed_ratio_mean >= baseline * 0.90",
    )
    checks["a_lat_mean"] = _check_le(
        _opt_num(overall, "a_lat_mean"),
        base_a_lat_mean * 1.10 if base_a_lat_mean is not None else None,
        "a_lat_mean <= baseline * 1.10",
    )
    checks["a_lat_p95"] = _check_le(
        _opt_num(overall, "a_lat_abs_p95"),
        base_a_lat_p95 * 1.20 if base_a_lat_p95 is not None else None,
        "a_lat_abs_p95 <= baseline * 1.20",
    )

    per_primary: Dict[str, Dict[str, Any]] = {}
    for label, view in primary_views.items():
        baseline_success = _opt_num(baseline_primary_doc, label, "success")
        verdict = judge_group(view, baseline_success)
        verdict["success_rate"] = view.get("success_rate")
        verdict["wilson_ci"] = view.get("success_wilson")
        per_primary[label] = verdict

    judged = [check for check in checks.values() if check.get("judged", True) and check.get("passed") is not None]
    primary_judged = [item for item in per_primary.values() if item["judged"]]
    all_passed = (
        baseline_doc is not None
        and bool(judged or primary_judged)
        and all(bool(check["passed"]) for check in judged)
        and all(bool(item["passed"]) for item in primary_judged)
    )
    return {
        "reference_available": baseline_doc is not None,
        "checks": checks,
        "per_primary": per_primary,
        "n_checks_judged": len(judged) + len(primary_judged),
        "all_passed": bool(all_passed),
    }


# --------------------------------------------------------------------------- #
# episode 控制（worker 内运行；metadrive/torch 延迟导入）
# --------------------------------------------------------------------------- #

def _wrap_to_pi(angle: float) -> float:
    """回绕到 (-π, π]（与 metadrive.utils.math.wrap_to_pi 同实现，避免模块级重依赖）。"""
    value = angle % (2.0 * math.pi)
    if value > math.pi:
        value -= 2.0 * math.pi
    return value


def _step_dt(env: Any) -> float:
    """环境单步物理时长（s）：默认 0.02 s × 5 = 0.1 s（envs/base_env.py:188-190）。"""
    config = getattr(env, "config", None)
    if config is None:
        return 0.1
    try:
        return float(config["physics_world_step_size"]) * float(config["decision_repeat"])
    except (KeyError, TypeError, ValueError):
        return 0.1


def _unwrap_env(env: Any) -> Any:
    """兼容可选 MetaDriveWrapper：外层无 engine 而内层有则取内层。"""
    if hasattr(env, "engine"):
        return env
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "engine"):
        return inner
    return env


def _is_crash(info: Mapping[str, Any]) -> bool:
    """episode 内碰撞：crash 或任一分量标志（与 baseline_eval 同口径）。"""
    return bool(
        info.get("crash")
        or info.get("crash_vehicle")
        or info.get("crash_object")
        or info.get("crash_building")
        or info.get("crash_human")
        or info.get("crash_sidewalk")
    )


def _lane_limit_mps(ego: Any, controller: Any) -> float:
    """车道限速（m/s）；未设置(>=1000)时用 controller 的兜底值。"""
    from env.expert.pure_pursuit_idm import lane_speed_limit_mps

    return lane_speed_limit_mps(
        getattr(ego, "lane", None),
        units=str(getattr(controller, "speed_limit_units", "mps")),
        fallback_mps=float(getattr(controller, "fallback_speed_limit_mps", DEFAULT_SPEED_LIMIT_FALLBACK)),
    )


def _min_ttc_step(env: Any, ego: Any) -> float:
    """本步最小 TTC（s）：与 env/obs/od.py 同口径（自车系相对位置/速度，前方走廊）。"""
    engine = getattr(env, "engine", None)
    if engine is None:
        return DEFAULT_TTC_CAP_S
    best = DEFAULT_TTC_CAP_S
    try:
        objects = list(engine.get_objects().values())
    except Exception:  # noqa: BLE001 - 车辆检索失败不应中断评测
        return best
    for obj in objects:
        if obj is ego:
            continue
        try:
            rel_pos = np.asarray(
                ego.convert_to_local_coordinates(obj.position, ego.position), dtype=np.float64
            )
            rel_vel = np.asarray(
                ego.convert_to_local_coordinates(obj.velocity, ego.velocity), dtype=np.float64
            )
        except Exception:  # noqa: BLE001 - 静态物体/无速度对象跳过
            continue
        if rel_pos[0] <= 0.0 or abs(float(rel_pos[1])) > DEFAULT_TTC_LATERAL_M:
            continue
        closing = -float(rel_vel[0])
        if closing <= 1e-3:
            continue
        best = min(best, float(rel_pos[0]) / closing)
    return min(best, DEFAULT_TTC_CAP_S)


class _BaselineController:
    """规则基线：把 ``PurePursuitIDMPolicy`` 注册进 engine（官方 agent 调用路径）。

    注册后 ``env.step`` 传入的占位动作被忽略（``engine/base_engine.py:98-104``），
    其 ``action_info`` 在自动驾驶路径（before_step）中被填充。
    """

    kind = "baseline"
    speed_limit_units = "mps"
    fallback_speed_limit_mps = DEFAULT_SPEED_LIMIT_FALLBACK

    def __init__(self, env: Any, spec: Any, params: Optional[Mapping[str, Any]] = None):
        self._spec = spec
        self._params = dict(params or {})
        self.policy = None  # bind() 里注册（env.engine 在 reset 前是 None）

    def bind(self, env: Any) -> None:
        """episode 开始（此时已 reset、engine 已建）：注册规则基线策略。

        为什么不在 ``__init__``：``build_env`` 只建 env 对象，engine 在首次 ``reset()``
        才惰性创建（``engine_utils.py``），提前访问 ``env.engine`` 会是 None。
        """
        from env.expert.pure_pursuit_idm import PurePursuitIDMPolicy

        self.policy = env.engine.add_policy(
            env.agent.id,
            PurePursuitIDMPolicy,
            env.agent,
            int(getattr(self._spec, "seed", 0)),
            **self._params,
        )
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def action(self, env: Any) -> List[float]:
        return [0.0, 0.0]  # 注册后由 engine 策略接管

    def action_info(self, env: Any) -> Dict[str, float]:
        info = getattr(self.policy, "action_info", None) or {}
        action = info.get("action", [0.0, 0.0])
        return {
            "steer": float(action[0]),
            "throttle": float(action[1]),
            "lead_gap_m": float(info.get("pp_lead_gap_m", -1.0)),
        }

    def params(self) -> Dict[str, Any]:
        params = getattr(self.policy, "params", None)
        return dict(params()) if callable(params) else {}


def _build_ckpt_model(model_cls: Any, config: Mapping[str, Any]) -> Any:
    """构造 ``DrivingModel``：单参数（整份 config）优先，否则按签名过滤 kwargs。"""
    try:
        return model_cls(dict(config))
    except TypeError as first_error:
        try:
            signature = inspect.signature(model_cls.__init__)
        except (TypeError, ValueError):
            raise first_error
        accepted = {
            name for name, parameter in signature.parameters.items()
            if name != "self" and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        kwargs = {key: value for key, value in dict(config).items() if key in accepted}
        if not kwargs:
            raise first_error
        return model_cls(**kwargs)


_CKPT_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}


def _filter_checkpoint_state(model: Any, state: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str], List[str]]:
    """按当前模型形状过滤 ckpt state_dict（架构变更后 strict=False 也会因 shape mismatch 报错）。

    返回 ``(filtered, missing, unexpected, shape_mismatch)``：
    - ``missing``：模型需要但 ckpt 没有的键（随机初始化保留）；
    - ``shape_mismatch``：ckpt 有但形状不符的键（跳过加载、随机初始化保留）——H=96→128
      的架构放大后属预期；
    - ``unexpected``：ckpt 多余（模型没有）的键。
    """
    model_state = model.state_dict()
    filtered: Dict[str, Any] = {}
    unexpected: List[str] = []
    shape_mismatch: List[str] = []
    loaded_like: List[str] = []
    for key, value in state.items():
        target = model_state.get(key)
        if target is None:
            unexpected.append(str(key))
            continue
        loaded_like.append(str(key))
        try:
            same_shape = tuple(target.shape) == tuple(value.shape)
        except AttributeError:
            same_shape = False
        if not same_shape:
            shape_mismatch.append(str(key))
            continue
        filtered[key] = value
    missing = [key for key in model_state if key not in loaded_like]
    return filtered, missing, unexpected, shape_mismatch


def _load_ckpt_model(ckpt: str, config: Mapping[str, Any], device: str = "cpu") -> Any:
    """worker 内按 ``(ckpt, device)`` 缓存模型（回收前后各加载一次，不会每 spec 重载）。

    架构变更（如 H 96→128）后用 :func:`_filter_checkpoint_state` 只加载形状一致的键，
    其余键打印上报（不修改 ckpt 文件）。
    """
    cached = _CKPT_MODEL_CACHE.get((ckpt, str(device)))
    if cached is not None:
        return cached
    import torch

    model = _build_model_from_config(config)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in payload and isinstance(payload[key], Mapping):
                state = payload[key]
                break
    if not isinstance(state, Mapping):
        raise ValueError(f"ckpt {ckpt!r} 不含可识别的 state_dict（顶层键 {list(payload)[:5]}）")
    filtered, missing, unexpected, shape_mismatch = _filter_checkpoint_state(model, state)
    model.load_state_dict(filtered, strict=False)
    print(
        f"[eval_runner] ckpt 载入 {ckpt}: loaded={len(filtered)}/{len(model.state_dict())} "
        f"ckpt_keys={len(state)} missing={len(missing)} shape_mismatch={len(shape_mismatch)} "
        f"unexpected={len(unexpected)} device={device} torch_threads={torch.get_num_threads()}",
        flush=True,
    )
    for label, keys in (("missing", missing), ("shape_mismatch", shape_mismatch), ("unexpected", unexpected)):
        if keys:
            print(f"[eval_runner]   {label}（{len(keys)}，示例 {list(keys)[:5]}）", flush=True)
    model.to(device).eval()
    _CKPT_MODEL_CACHE[(ckpt, str(device))] = model
    return model


def _build_model_from_config(config: Mapping[str, Any]) -> Any:
    """优先复用训练侧 ``pipeline.stages.build_model``（保证与 ckpt 架构一致）。

    训练侧从 ``config/model.yaml`` 的嵌套结构（``hidden_dim`` / ``moe.experts.count`` …）
    构造 ``DrivingModel``；评测必须走同一条路，否则键名不匹配（历史 bug：把整份 config
    当第一个位置参数传进去 → ``int(dict)`` 报错）。
    """
    try:
        from pipeline.stages import build_model  # 训练同源（延迟导入避免环）
    except ImportError:
        build_model = None
    if build_model is not None:
        return build_model(dict(config))
    from net.model import DrivingModel  # 回退：旧的最小兼容构造

    return _build_ckpt_model(DrivingModel, config)


# NOTE(_make_tracker): 已删除。历史版本用构造器直接 new LqrTracker（签名不强）导致
# `int(dict)`/None-engine 报错；现在 ckpt 路径与训练侧一致：`engine.add_policy(...)` 注册
# + 每策略步 `set_reference((N,2))`（见 `_CkptController.bind/action`）。


class _CkptController:
    """ckpt 策略：每 0.5 s 用 ``DrivingModel`` 出 ``(ds, dθ)``，交给 N3 跟踪器逐步执行。

    观测每 env step 都过 ``ObservationBuilder``（历史窗口依赖逐 step push），动作只
    在 5 步（0.5 s）边界重算；跟踪器参考由 ``env.tracking.interpolate`` 从单个动作
    插值成 30 点（契约 §1）。
    """

    kind = "ckpt"
    speed_limit_units = "mps"
    fallback_speed_limit_mps = DEFAULT_SPEED_LIMIT_FALLBACK
    decision_interval = 5  # 0.5 s / 0.1 s

    def __init__(self, env: Any, spec: Any, task: Mapping[str, Any]):
        import torch

        self.spec = spec
        self.device = torch.device(str(task.get("device") or "cpu"))
        self.model = _load_ckpt_model(str(task["ckpt"]), task.get("model_config") or {}, str(self.device))
        self.obs_config = dict(task.get("obs_config") or {})
        self.tracker_config = dict(task.get("tracker_config") or {})
        #: "exact"（阶段 A/B 精确/运动学执行，默认）| "lqr"（阶段 C 闭环）
        self.tracker_kind = str(task.get("tracker") or "exact").lower()
        self.builder = None
        self.tracker = None
        self._steps = 0
        self._action = [0.0, 0.0]
        #: 逐 env step 的自车实测位姿（用于重建"上一策略步实测动作"，见 _measured_prev_action）
        self._pose_history: List[Tuple[float, float, float]] = []

    def bind(self, env: Any) -> None:
        from env.obs.builder import ObservationBuilder

        self.builder = ObservationBuilder(self.obs_config)
        self.builder.reset()
        self._steps = 0
        self._action = [0.0, 0.0]
        self._pose_history = []
        # §8.4：episode 起点没有上一动作（与 collect_expert/training pools 同口径）
        env.prev_policy_action = np.zeros(2, dtype=np.float64)
        if self.tracker_kind == "exact":
            from env.tracking import ExactTracker  # 阶段 A/B 运动学执行器（step 后 apply）

            self.tracker = ExactTracker(dt=0.5, hz=10)
            return
        from env.tracking import LqrTracker  # N3 契约

        # 与训练侧 ``LocalEnvPool(_setup_episode)`` 同款：engine 注册 + 每策略步 set_reference
        self.tracker = env.engine.add_policy(
            env.agent.id,
            LqrTracker,
            env.agent,
            int(getattr(self.spec, "seed", 0)),
        )
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()

    def _tensors(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        """与训练侧 ``PPOTrainer._to_tensor_obs`` 同口径：加 batch 维 + 单槽通道去槽位维。

        （历史 bug：只加 ``[None]`` 会让 ``ego`` 变成 (1,1,8)，模型校验直接报错。）
        """
        import torch

        from pipeline.trainer import NON_OBS_KEYS, squeeze_single_slot

        batch = {
            key: np.asarray(value, dtype=np.float32)[None]
            for key, value in obs.items()
            if key not in NON_OBS_KEYS
        }
        squeeze_single_slot(batch)
        return {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}

    def _measured_prev_action(self, env: Any) -> Optional[np.ndarray]:
        """最近 ``decision_interval`` 个 env step 的**实测**动作 ``(ds,dθ)``。

        与 ``tools/collect_expert.py::_window_actions`` 同口径（折线弧长 + 航向差 wrap）：
        BC 数据集的 ego reserved 6:8 正是该量，eval 必须同口径注入，否则策略看到
        训练中不存在的 ``prev_action=0``（分布外输入）。
        """
        from env.obs.base import wrap_to_pi

        ego = env.agent
        self._pose_history.append(
            (float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta))
        )
        if len(self._pose_history) <= self.decision_interval:
            return None
        window = self._pose_history[-(self.decision_interval + 1) :]
        length = 0.0
        for (ax, ay, _), (bx, by, _) in zip(window[:-1], window[1:]):
            length += math.hypot(bx - ax, by - ay)
        dtheta = float(wrap_to_pi(window[-1][2] - window[0][2]))
        return np.asarray([length, dtheta], dtype=np.float64)

    def action(self, env: Any) -> List[float]:
        import torch

        measured = self._measured_prev_action(env)
        if measured is not None:
            env.prev_policy_action = measured  # 供本次决策的 obs（ego reserved 6:8）
        obs = self.builder.build(env, self.spec)
        if self._steps % self.decision_interval == 0:
            with torch.no_grad():
                # 需要 6 步规划预览（跟踪器参考）：走 rollout 路径（跳过 WM 直接多步）。
                # 单动作参考会退化为"瞄准参考终点"的 4–5 m 前视 → 车道保持/速度都会退化。
                output = self.model(self._tensors(obs), rollout=True, world_model=False)
            mu = np.asarray(output["action_mu"].detach().cpu(), dtype=np.float64).reshape(-1)
            plan = np.asarray(output["plan"].detach().cpu(), dtype=np.float64).reshape(-1, 2)
            plan[0] = mu[:2]  # 执行动作与策略均值一致（确定性评测）
            if self.tracker_kind == "exact":
                # 阶段 A/B：把 6 步预览插值成 30 点参考，逐步 apply（env.step 之后）
                self.tracker.arm(env, actions=plan)
            else:
                # 阶段 C：set_reference((N,2) 动作序列)，跟踪器内部插值成 30 点
                self.tracker.set_reference(plan)
        self._steps += 1
        # engine 已注册策略，传回的占位动作会被忽略（engine/base_engine.py:98-104）
        return self._action

    def post_step(self, env: Any) -> None:
        """env.step 之后调用（exact 执行器专用）：把 ego 置于插值参考位姿。"""
        if self.tracker_kind == "exact" and self.tracker is not None:
            self.tracker.apply(env)

    def action_info(self, env: Any) -> Dict[str, float]:
        return {"steer": self._action[0], "throttle": self._action[1], "lead_gap_m": -1.0}


def _run_episode(env: Any, spec: Any, *, max_steps: int, controller: Any) -> Dict[str, Any]:
    """跑完单个 episode，返回逐 episode KPI（含 ``_`` 前缀的池化样本）。"""
    reset_out = env.reset()
    reset_info: Dict[str, Any] = {}
    if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        reset_info = dict(reset_out[1])
    controller.bind(env)
    ego = env.agent

    dt = _step_dt(env)
    a_lon_samples: List[float] = []
    a_lat_samples: List[float] = []
    jerk_samples: List[float] = []
    speed_ratio_samples: List[float] = []
    steer_abs_samples: List[float] = []
    throttle_samples: List[float] = []
    lead_gap_samples: List[float] = []

    prev_speed = float(ego.speed)
    prev_theta = float(ego.heading_theta)
    prev_a_lon: Optional[float] = None
    info: Dict[str, Any] = dict(reset_info)
    route_completion = float(info.get("route_completion", 0.0))
    max_route_completion = route_completion
    min_ttc = DEFAULT_TTC_CAP_S
    solid_line_crossing = False
    speed_limit_violation = False
    violation_steps = 0
    speed_limit_steps = 0
    steps = 0
    success = collision = off_road = False
    terminated = truncated = False
    started_at = time.time()

    for step_index in range(max_steps):
        action = controller.action(env)
        _, _, terminated, truncated, info = env.step(action)
        if hasattr(controller, "post_step"):
            # 阶段 A/B 语义（exact/kinematic）：env.step 之后把 ego 置于插值参考位姿
            # （ExactTracker 的标准用法；见 env/tracking.py::ExactTracker docstring）。
            controller.post_step(env)
        info = info if isinstance(info, dict) else {}
        steps = step_index + 1

        speed = float(info.get("velocity", float(ego.speed)))
        theta = float(ego.heading_theta)
        delta_theta = _wrap_to_pi(theta - prev_theta)
        a_lon = (speed - prev_speed) / dt
        a_lon_samples.append(a_lon)
        a_lat_samples.append(0.5 * (speed + prev_speed) * delta_theta / dt)
        if prev_a_lon is not None:
            jerk_samples.append((a_lon - prev_a_lon) / dt)
        prev_a_lon = a_lon
        prev_speed, prev_theta = speed, theta

        route_completion = float(info.get("route_completion", route_completion))
        max_route_completion = max(max_route_completion, route_completion)
        success = success or bool(info.get("arrive_dest", False))
        collision = collision or _is_crash(info)
        off_road = off_road or bool(info.get("out_of_road", False))

        limit_mps = _lane_limit_mps(ego, controller)
        if limit_mps > 0.0:
            ratio = speed / limit_mps
            speed_ratio_samples.append(ratio)
            speed_limit_steps += 1
            if ratio > 1.0:
                speed_limit_violation = True
                violation_steps += 1

        min_ttc = min(min_ttc, _min_ttc_step(env, ego))
        try:
            if bool(getattr(ego, "on_white_continuous_line", False)) or bool(
                getattr(ego, "on_yellow_continuous_line", False)
            ):
                solid_line_crossing = True
        except Exception:  # noqa: BLE001 - 标志读取失败不影响其它 KPI
            pass

        action_info = controller.action_info(env)
        steer_abs_samples.append(abs(float(action_info["steer"])))
        throttle_samples.append(float(action_info["throttle"]))
        lead_gap = float(action_info["lead_gap_m"])
        if lead_gap >= 0.0:
            lead_gap_samples.append(lead_gap)

        if terminated or truncated:
            break

    if success:
        termination = "arrive_dest"
    elif collision:
        termination = "collision"
    elif off_road:
        termination = "out_of_road"
    elif truncated or steps >= max_steps or bool(info.get("max_step", False)):
        termination = "max_step"
    else:
        termination = "other"

    a_lon_array = np.asarray(a_lon_samples, dtype=np.float64)
    a_lat_array = np.asarray(a_lat_samples, dtype=np.float64)
    jerk_array = np.asarray(jerk_samples, dtype=np.float64)
    speed_ratio_array = np.asarray(speed_ratio_samples, dtype=np.float64)
    return {
        "id": int(getattr(spec, "id", -1)),
        "seed": int(getattr(spec, "seed", -1)),
        "split": str(getattr(spec, "split", "unknown")),
        "difficulty": str(getattr(spec, "difficulty", "unknown")),
        "primary": primary_label(spec),
        "geometry": [str(label) for label in (getattr(spec, "geometry", None) or [])],
        "extra_geometry": extra_geometry_labels(spec),
        "compound": is_compound_spec(spec),
        "success": bool(success),
        "collision": bool(collision),
        "off_road": bool(off_road),
        "solid_line_crossing": bool(solid_line_crossing),
        "speed_limit_violation": bool(speed_limit_violation),
        "speed_limit_violation_step_rate": (
            float(violation_steps / speed_limit_steps) if speed_limit_steps else float("nan")
        ),
        "termination": termination,
        "route_completion": float(route_completion),
        "max_route_completion": float(max_route_completion),
        "steps": int(steps),
        "duration_s": float(time.time() - started_at),
        "min_ttc": float(min_ttc),
        "mean_speed_mps": _finite_mean([float(info.get("velocity", 0.0))]),
        "a_lon_mean": _finite_mean(a_lon_array),
        "a_lon_abs_p95": _p95_abs(a_lon_array),
        "a_lat_mean": _finite_mean(a_lat_array),
        "a_lat_abs_p95": _p95_abs(a_lat_array),
        "jerk_mean": _finite_mean(jerk_array),
        "jerk_abs_p95": _p95_abs(jerk_array),
        "speed_ratio_mean": _finite_mean(speed_ratio_array),
        "speed_ratio_p95": (
            float(np.nanpercentile(speed_ratio_array, 95.0)) if speed_ratio_array.size else float("nan")
        ),
        "steer_abs_mean": _finite_mean(steer_abs_samples),
        "throttle_mean": _finite_mean(throttle_samples),
        "lead_gap_mean_m": float(np.mean(lead_gap_samples)) if lead_gap_samples else -1.0,
        # 池化样本（聚合后由 _public_episode 剥离，不进 JSON/CSV）
        "_a_lon": a_lon_samples,
        "_a_lat": a_lat_samples,
        "_jerk": jerk_samples,
        "_speed_ratio": speed_ratio_samples,
    }


def _error_episode(spec: Any, task: Mapping[str, Any], exc: BaseException) -> Dict[str, Any]:
    """单条 spec 失败时的占位结果：计为失败但单列 n_error，不终止整批评测。"""
    return {
        "id": int(getattr(spec, "id", -1)),
        "seed": int(getattr(spec, "seed", -1)),
        "split": str(getattr(spec, "split", "unknown")),
        "difficulty": str(getattr(spec, "difficulty", "unknown")),
        "primary": primary_label(spec),
        "geometry": [str(label) for label in (getattr(spec, "geometry", None) or [])],
        "extra_geometry": extra_geometry_labels(spec),
        "compound": is_compound_spec(spec),
        "success": False,
        "collision": False,
        "off_road": False,
        "solid_line_crossing": False,
        "speed_limit_violation": False,
        "speed_limit_violation_step_rate": float("nan"),
        "termination": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "error_traceback": traceback.format_exc(),
        "policy": str(task.get("policy", "?")),
        "route_completion": 0.0,
        "max_route_completion": 0.0,
        "steps": 0,
        "duration_s": 0.0,
        "min_ttc": float("nan"),
        "mean_speed_mps": float("nan"),
        "a_lon_mean": float("nan"),
        "a_lon_abs_p95": float("nan"),
        "a_lat_mean": float("nan"),
        "a_lat_abs_p95": float("nan"),
        "jerk_mean": float("nan"),
        "jerk_abs_p95": float("nan"),
        "speed_ratio_mean": float("nan"),
        "speed_ratio_p95": float("nan"),
        "steer_abs_mean": float("nan"),
        "throttle_mean": float("nan"),
        "lead_gap_mean_m": -1.0,
        "_a_lon": [],
        "_a_lat": [],
        "_jerk": [],
        "_speed_ratio": [],
    }


def _apply_worker_thread_limits(task: Mapping[str, Any]) -> None:
    """ckpt worker 入口：设 OMP/MKL 默认值并钳制 torch 线程（防每进程 14 线程的风暴）。

    ``task['torch_threads']`` 由主进程按 ``min(cap, cpu//workers)`` 计算（见
    ``pipeline.trainer.resolve_torch_threads``）；缺失时按同一公式回退。
    """
    omp = max(1, int(task.get("omp_num_threads") or 1))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, str(omp))
    threads = task.get("torch_threads")
    if threads is None:
        workers = max(1, int(task.get("workers") or 1))
        threads = max(1, min(4, (os.cpu_count() or 1) // workers))
    try:
        import torch

        torch.set_num_threads(max(1, int(threads)))
    except Exception:  # noqa: BLE001 - 钳制失败不影响评测
        pass


def _evaluate_spec(task: Mapping[str, Any]) -> Dict[str, Any]:
    """构建环境并评测单个 spec（顶层函数，供 spawn 池 pickle）。"""
    spec = task["spec"]
    env = None
    try:
        if task.get("policy") == "ckpt":
            _apply_worker_thread_limits(task)
        from env.metadrive_env import build_env  # L2 契约；延迟到 worker 内导入

        env = _unwrap_env(
            build_env(spec, traffic_density=task.get("traffic_density"), use_render=False)
        )
        if task["policy"] == "baseline":
            controller = _BaselineController(env, spec, task.get("policy_params"))
        elif task["policy"] == "ckpt":
            controller = _CkptController(env, spec, task)
        else:
            raise ValueError(f"未知 policy={task['policy']!r}（应为 baseline|ckpt）")
        episode = _run_episode(env, spec, max_steps=int(task["max_steps"]), controller=controller)
        episode["policy"] = str(task["policy"])
        if isinstance(controller, _BaselineController):
            episode["_policy_params"] = controller.params()
        return episode
    except BaseException as exc:  # noqa: BLE001 - 单条场景失败不应终止整批评测
        return _error_episode(spec, task, exc)
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass


# --------------------------------------------------------------------------- #
# 报告与输出
# --------------------------------------------------------------------------- #

def _public_episode(episode: Mapping[str, Any]) -> Dict[str, Any]:
    """剥离下划线开头的私有池化样本字段。"""
    return {key: value for key, value in episode.items() if not key.startswith("_")}


def _sanitize(obj: Any) -> Any:
    """NaN/Inf → null，numpy 标量 → Python 标量，保证 JSON 合法。"""
    if isinstance(obj, Mapping):
        return {key: _sanitize(value) for key, value in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_sanitize(value) for value in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_sanitize(value) for value in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


def build_report(
    episodes: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    baseline_doc: Optional[Mapping[str, Any]] = None,
    baseline_primary_doc: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """生成最终报告：overall + by_primary（判定）+ compound + by_difficulty + by_geometry。"""
    by_primary_raw = group_episodes(episodes, "primary")
    primary_views = {name: summarize(group) for name, group in sorted(by_primary_raw.items())}
    compound = [ep for ep in episodes if ep.get("compound")]
    by_difficulty = group_episodes(episodes, "difficulty")
    by_geometry: Dict[str, List[Mapping[str, Any]]] = {}
    for episode in episodes:
        for label in (episode.get("geometry") or ["unlabeled"]):
            by_geometry.setdefault(str(label), []).append(episode)

    overall = summarize(episodes)
    verdict = evaluate_verdicts(overall, primary_views, baseline_doc, baseline_primary_doc)
    return {
        "meta": dict(meta),
        "verdict": verdict,
        "overall": overall,
        "by_primary": primary_views,
        "compound": summarize(compound),
        "by_difficulty": {name: summarize(group) for name, group in sorted(by_difficulty.items())},
        # 逐标签视图（一条 spec 可计入多个标签，报告口径，不判定）
        "by_geometry": {name: summarize(group) for name, group in sorted(by_geometry.items())},
        "episodes": [_public_episode(episode) for episode in sorted(episodes, key=lambda ep: ep["id"])],
    }


CSV_FIELDS = (
    "id", "seed", "split", "difficulty", "primary", "geometry", "extra_geometry", "compound",
    "success", "collision", "off_road", "solid_line_crossing", "speed_limit_violation",
    "speed_limit_violation_step_rate", "termination", "route_completion", "max_route_completion",
    "steps", "duration_s", "min_ttc", "mean_speed_mps",
    "a_lon_mean", "a_lon_abs_p95", "a_lat_mean", "a_lat_abs_p95", "jerk_mean", "jerk_abs_p95",
    "speed_ratio_mean", "speed_ratio_p95", "steer_abs_mean", "throttle_mean", "lead_gap_mean_m",
    "policy", "error",
)


def write_episodes_csv(episodes: Sequence[Mapping[str, Any]], path: Path) -> None:
    """逐 episode 明细写 CSV（几何标签用 ';' 连接；NaN 写空）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for episode in sorted(episodes, key=lambda ep: ep["id"]):
            row = dict(_public_episode(episode))
            row["geometry"] = ";".join(str(item) for item in row.get("geometry") or [])
            row["extra_geometry"] = ";".join(str(item) for item in row.get("extra_geometry") or [])
            row["compound"] = bool(row.get("compound"))
            for key, value in list(row.items()):
                if isinstance(value, float) and not math.isfinite(value):
                    row[key] = ""
            writer.writerow(row)


def _print_summary(report: Mapping[str, Any]) -> None:
    """打印 overall / by_primary / by_difficulty 简表 + 判定结论。"""
    header = (
        f"{'group':<22}{'n':>5}{'succ':>8}{'95% CI':>18}{'coll':>7}{'offrd':>7}"
        f"{'ttc':>7}{'a_lat95':>9}{'spd':>7}{'pass':>7}"
    )
    print("[eval_runner] " + header, flush=True)

    def _row(name: str, group: Mapping[str, Any], verdict: Optional[Mapping[str, Any]] = None) -> None:
        if not group.get("n"):
            return
        interval = group.get("success_wilson")
        ci_text = (
            f"[{interval[0]:.3f},{interval[1]:.3f}]"
            if isinstance(interval, (list, tuple)) and len(interval) == 2
            and all(isinstance(item, (int, float)) and math.isfinite(item) for item in interval)
            else "n/a"
        )
        passed = verdict.get("passed") if isinstance(verdict, Mapping) else None
        pass_text = "pass" if passed else ("FAIL" if passed is False else "-")
        print(
            f"[eval_runner] {name:<22}{group['n']:>5}{group.get('success_rate', float('nan')):>8.3f}"
            f"{ci_text:>18}{group.get('collision_rate', float('nan')):>7.3f}"
            f"{group.get('off_road_rate', float('nan')):>7.3f}"
            f"{group.get('min_ttc_mean', float('nan')):>7.2f}"
            f"{group.get('a_lat_abs_p95', float('nan')):>9.3f}"
            f"{group.get('speed_ratio_mean', float('nan')):>7.3f}{pass_text:>7}",
            flush=True,
        )

    _row("overall", report["overall"], report["verdict"]["checks"]["overall_success"])
    for label, group in report["by_primary"].items():
        _row(f"primary/{label}", group, report["verdict"]["per_primary"].get(label))
    _row("compound(report)", report["compound"])
    for name, group in report["by_difficulty"].items():
        _row(f"difficulty/{name}", group)


# --------------------------------------------------------------------------- #
# 配置加载 / 内存检查 / 池执行
# --------------------------------------------------------------------------- #

def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def load_config(path: Any, _seen: Optional[set] = None) -> Dict[str, Any]:
    """加载主配置并递归合并 ``includes``（子配置先合并，主文件最后覆盖）。"""
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML 不可用（需要 tools/venv-python）")
    path = Path(path).resolve()
    seen = _seen if _seen is not None else set()
    if str(path) in seen:
        raise ValueError(f"配置 includes 存在环：{path}")
    seen.add(str(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"配置 {path} 必须是 YAML 映射")
    data = dict(payload)
    includes = data.pop("includes", []) or []
    merged: Dict[str, Any] = {}
    for include in includes:
        include_path = Path(str(include))
        if not include_path.is_absolute():
            include_path = Path(_PROJECT_ROOT) / include_path
        merged = _deep_merge(merged, load_config(include_path, seen))
    return _deep_merge(merged, data)


def _pause_train_pool(policy: str) -> Dict[str, Any]:
    """按 ``eval.train_pool_policy`` 处理训练池（pause / workers_le_2 / ignore）。"""
    report: Dict[str, Any] = {"policy": policy, "hook_registered": _TRAIN_POOL_HOOK is not None}
    if policy in ("ignore", "none", "off"):
        report["action"] = "skip"
        return report
    if policy in ("workers_le_2", "reduce"):
        report["action"] = "workers_le_2"
        return report
    # 默认 pause：调用钩子并等待 MemAvailable 回升
    report["action"] = "pause"
    if _TRAIN_POOL_HOOK is None:
        report["paused"] = None
        report["note"] = "未注册训练池钩子（视为本机无训练池）"
        return report
    try:
        report["paused"] = bool(_TRAIN_POOL_HOOK())
    except Exception as exc:  # noqa: BLE001 - 钩子失败不致命，但如实记录
        report["paused"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _require_memory(floor_mb: float, policy: str, workers: int, wait_s: float = PAUSE_WAIT_S) -> Tuple[int, Dict[str, Any]]:
    """程序化内存检查（§8.1）：暂停训练池 → 等待/缩减 worker → 必要时拒绝开跑。

    Returns:
        (effective_workers, memory_report)
    """
    report: Dict[str, Any] = {
        "floor_mb": float(floor_mb),
        "initial_mb": read_mem_available_mb(),
        "train_pool": _pause_train_pool(policy),
        "workers_requested": int(workers),
    }
    available = report["initial_mb"]
    if available is None:
        report["note"] = "无法读取 /proc/meminfo（非 Linux？），跳过内存检查"
        report["workers_effective"] = int(workers)
        return int(workers), report
    deadline = time.time() + max(0.0, float(wait_s))
    while available < floor_mb and time.time() < deadline:
        time.sleep(0.5)
        available = read_mem_available_mb()
        if available is None:
            break
    if available < floor_mb and policy not in ("ignore", "none", "off"):
        workers = min(int(workers), 2)
        report["note"] = f"MemAvailable={available:.0f}MB < floor={floor_mb:.0f}MB，worker 缩减到 {workers}"
    if available is not None and available < floor_mb * MEM_HARD_FLOOR_RATIO:
        raise SystemExit(
            f"[eval_runner] 拒绝开跑：MemAvailable={available:.0f}MB < 硬下限 "
            f"{floor_mb * MEM_HARD_FLOOR_RATIO:.0f}MB（floor={floor_mb:.0f}MB）；"
            "请先暂停/缩减训练池后重试"
        )
    report["after_wait_mb"] = available
    report["workers_effective"] = int(workers)
    return int(workers), report


def _run_tasks(tasks: Sequence[Mapping[str, Any]], workers: int, recycle_every_specs: int) -> List[Dict[str, Any]]:
    """spawn 池执行；按 ``recycle_every_specs × workers`` 为一代重建池回收内存。

    与 ``env/scenario/validator.py::validate_specs`` 同实现：MetaDrive 每次
    ``build_env``+``close`` 残留 ~3.5MB/spec，长跑必须按 spec 实例数回收 worker。
    """
    results: List[Dict[str, Any]] = []
    total = len(tasks)
    if total == 0:
        return results
    # GL 修复：父进程（可能未经 tools/venv-python 启动）预载 venv glvnd，并保证 spawn worker
    # 继承 LD_LIBRARY_PATH（否则 Panda3D n_pipes=0 → MetaDrive "Known Pipes" IndexError）。
    from pipeline.gl_runtime import ensure_gl_library_path

    ensure_gl_library_path(preload=True)
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            result = _evaluate_spec(task)
            results.append(result)
            _print_progress(index, total, result)
        return results

    chunk = max(1, int(recycle_every_specs) * workers)
    n_chunks = (total + chunk - 1) // chunk
    started = time.time()
    for chunk_index, start in enumerate(range(0, total, chunk), start=1):
        stop = min(start + chunk, total)
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, stop - start), mp_context=context) as executor:
            futures = {executor.submit(_evaluate_spec, task): task for task in tasks[start:stop]}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _print_progress(len(results), total, result)
        print(
            f"[eval_runner] chunk {chunk_index}/{n_chunks}: {stop}/{total} specs, "
            f"elapsed {time.time() - started:.0f}s",
            flush=True,
        )
    return results


def _print_progress(index: int, total: int, result: Mapping[str, Any]) -> None:
    route_completion = result.get("route_completion")
    rc_text = f"{route_completion:.3f}" if isinstance(route_completion, (int, float)) else "nan"
    print(
        f"[eval_runner] {index}/{total} spec={result.get('id')} "
        f"primary={result.get('primary')} success={result.get('success')} "
        f"rc={rc_text} reason={result.get('termination')}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval_runner.py",
        description="P2 冻结验证集评测（primary 标签分组 + Wilson CI + 弱类 floor）",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                        help="主配置（默认 config/default.yaml；含 eval.yaml includes）")
    parser.add_argument("--spec", type=Path, default=None,
                        help="验证场景 spec（默认取 eval.yaml::eval.spec = 冻结 1000 条）")
    parser.add_argument("--policy", choices=("baseline", "ckpt"), default="baseline",
                        help="baseline=PurePursuitIDMPolicy（参考对比）；ckpt=加载 N1 权重（需 --ckpt）")
    parser.add_argument("--ckpt", type=Path, default=None, help="策略权重路径（--policy ckpt 必填）")
    parser.add_argument("--out", type=Path, default=Path("runs/eval"),
                        help="评测输出根目录（实际写到 <out>/<name>/，默认 runs/eval）")
    parser.add_argument("--name", default=None,
                        help="运行名（默认 '<policy>_<时间戳>'；落地 <out>/<name>/metrics.json）")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条 spec（按文件顺序，冒烟用）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"spawn worker 数（默认 {DEFAULT_WORKERS}，评测与训练池互斥）")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"单 episode 最大步数（默认 {DEFAULT_MAX_STEPS} = 100 s @ 0.1 s）")
    parser.add_argument("--baseline-ref", type=Path, default=Path(DEFAULT_BASELINE_REF),
                        help="冻结基线 JSON（默认 runs/baseline_eval/val_reference.json；"
                             "同目录 *_by_primary.json 用于分组判定）")
    parser.add_argument("--seed", type=int, default=0, help="评测种子（确定性；默认 0）")
    parser.add_argument("--tracker", choices=("lqr", "exact"), default="exact",
                        help="ckpt 动作执行器：exact=阶段 A/B 精确/运动学执行（默认，衡量规划轨迹本身）；"
                             "lqr=阶段 C 闭环跟踪（含控制器跟踪误差）")
    parser.add_argument("--device", default=None,
                        help="ckpt 策略运行设备；默认取 config train.device（auto=cuda 可用则 cuda）")
    return parser


def _baseline_primary_path(baseline_ref: Path) -> Path:
    """从 ``val_reference.json`` 推出 ``val_reference_by_primary.json``。"""
    return baseline_ref.with_name(f"{baseline_ref.stem}_by_primary{baseline_ref.suffix}")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.policy == "ckpt" and args.ckpt is None:
        print("[eval_runner] --policy ckpt 需要 --ckpt", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"[eval_runner] 配置加载失败：{exc}", file=sys.stderr)
        return 2

    eval_cfg = dict(config.get("eval") or {})
    spec_path = Path(args.spec or eval_cfg.get("spec") or "env/specs/scenarios_val.json")
    if not spec_path.is_file():
        print(f"[eval_runner] spec 文件不存在：{spec_path}", file=sys.stderr)
        return 2
    if args.policy == "ckpt" and not Path(args.ckpt).is_file():
        print(f"[eval_runner] ckpt 文件不存在：{args.ckpt}", file=sys.stderr)
        return 2

    mem_floor = float(eval_cfg.get("mem_available_floor_mb", DEFAULT_MEM_FLOOR_MB))
    train_policy = str(eval_cfg.get("train_pool_policy", "pause"))
    recycle_every = int(eval_cfg.get("recycle_every_specs", DEFAULT_RECYCLE_EVERY_SPECS))
    workers, mem_report = _require_memory(mem_floor, train_policy, max(1, int(args.workers)))

    # ---- 设备 / 线程（ckpt 路径需要 torch；baseline 不导入 torch）----
    threads_cfg = dict((config.get("train") or {}).get("threads") or {})
    omp_num_threads = max(1, int(threads_cfg.get("omp_num_threads") or 1))
    for env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(env_key, str(omp_num_threads))  # spawn 子进程继承
    device: Optional[str] = None
    torch_threads: Optional[int] = None
    if args.policy == "ckpt":
        from pipeline.trainer import resolve_device, resolve_torch_threads  # 延迟导入（需 torch）

        device = resolve_device(args.device, config)
        torch_threads = resolve_torch_threads(workers, config=config)
        print(
            f"[eval_runner] device={device} torch_threads/worker={torch_threads} omp={omp_num_threads} "
            f"(workers={workers})",
            flush=True,
        )

    from env.scenario.spec import load_specs  # 轻量模块（仅 taxonomy 依赖）

    all_specs = list(load_specs(spec_path))
    specs = all_specs[: max(0, int(args.limit))] if args.limit is not None else all_specs
    if not specs:
        print(f"[eval_runner] 过滤后无可评测 spec（total={len(all_specs)}, limit={args.limit}）",
              file=sys.stderr)
        return 1

    env_cfg = dict(config.get("env") or {})
    run_name = str(args.name or f"{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}")
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_ref = Path(args.baseline_ref)
    baseline_doc = _load_json(baseline_ref) if baseline_ref.is_file() else None
    baseline_primary_doc = _load_json(_baseline_primary_path(baseline_ref))
    if baseline_doc is None:
        print(f"[eval_runner] 警告：基线参照不可用（{baseline_ref}），判定将标记为缺少可比数据",
              flush=True)

    tasks: List[Dict[str, Any]] = [
        {
            "spec": spec,
            "policy": args.policy,
            "ckpt": str(args.ckpt) if args.ckpt is not None else None,
            "max_steps": int(args.max_steps),
            "traffic_density": None,
            "policy_params": {},
            "model_config": {
                "hidden_dim": config.get("hidden_dim", 128),
                "moe": dict(config.get("moe") or {}),
                "world_model": dict(config.get("world_model") or {}),
            },
            "obs_config": dict(env_cfg.get("obs") or {}),
            "tracker_config": dict(env_cfg.get("tracking") or {}),
            "device": device,
            "workers": int(workers),
            "torch_threads": torch_threads,
            "omp_num_threads": int(omp_num_threads),
            "tracker": str(args.tracker),
        }
        for spec in specs
    ]

    print(
        f"[eval_runner] policy={args.policy} specs={len(specs)}/{len(all_specs)} "
        f"workers={workers} recycle_every_specs={recycle_every} max_steps={args.max_steps} "
        f"mem={mem_report.get('after_wait_mb') or mem_report.get('initial_mb')}MB "
        f"out={out_dir}",
        flush=True,
    )

    started_at = time.time()
    episodes = _run_tasks(tasks, workers, recycle_every)
    wall_time = time.time() - started_at

    effective_params: Dict[str, Any] = {}
    for episode in episodes:
        if episode.get("_policy_params"):
            effective_params = dict(episode["_policy_params"])
            break

    eval_section = config.get("eval")
    eval_section = eval_section if isinstance(eval_section, Mapping) else {}
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, Mapping):
        thresholds = eval_section.get("thresholds")
    thresholds = dict(thresholds) if isinstance(thresholds, Mapping) else {}
    meta: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "specs_path": str(spec_path.resolve()),
        "n_specs_total": len(all_specs),
        "n_evaluated": len(episodes),
        "n_error": sum(1 for ep in episodes if ep["termination"] == "error"),
        "limit": args.limit,
        "policy": args.policy,
        "ckpt": str(args.ckpt) if args.ckpt is not None else None,
        "policy_class": ("env.expert.pure_pursuit_idm.PurePursuitIDMPolicy"
                         if args.policy == "baseline" else "net.model.DrivingModel"),
        "tracker": str(args.tracker) if args.policy == "ckpt" else None,
        "policy_params_effective": effective_params,
        "workers": workers,
        "recycle_every_specs": recycle_every,
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "device": device,
        "torch_threads_per_worker": torch_threads,
        "omp_num_threads": int(omp_num_threads),
        "deterministic": bool(eval_cfg.get("deterministic", True)),
        "wall_time_s": wall_time,
        "memory": mem_report,
        "thresholds": thresholds,
        "baseline_ref": str(baseline_ref.resolve()),
        "baseline_primary_ref": str(_baseline_primary_path(baseline_ref).resolve()),
        "components": {"process": str(eval_cfg.get("process", "separate"))},
        "compound_definition": (
            "compound = spec.geometry 含 primary/straight 之外的几何标签（报告口径，不判定）；"
            "primary = spec.labels.geometry"
        ),
        "kpi_definitions": KPI_DEFINITIONS,
    }
    report = build_report(episodes, meta, baseline_doc, baseline_primary_doc)

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(_sanitize(report), handle, ensure_ascii=False, indent=2)
    csv_path = out_dir / "episodes.csv"
    write_episodes_csv(episodes, csv_path)

    _print_summary(report)
    verdict = report["verdict"]
    print(
        f"[eval_runner] verdict: all_passed={verdict['all_passed']} "
        f"(judged checks={verdict['n_checks_judged']})"
    )
    print(f"[eval_runner] metrics -> {metrics_path}")
    print(f"[eval_runner] episodes -> {csv_path} (n={len(episodes)}, errors={meta['n_error']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
