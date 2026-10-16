#!/usr/bin/env python3
"""独立进程、确定性的冻结验证集评测（P2/N5，契约 §5 + §8.1/§8.6）。

职责
----
1. 在**冻结验证集**（默认 ``env/specs/scenarios_val.json``，1000 条）上评测策略；
   ``--policy baseline`` = 规则基线 ``PurePursuitIDMPolicy``（同 harness/同种子，复现
   ``runs/baseline_eval/val_reference.json``）；``--policy ckpt`` = 加载 N1 的
   ``DrivingModel`` 权重并用 N3 的 ``env.tracking`` 执行动作：
   ``--tracker exact``（默认，阶段 A/B：每 0.1 s 子步置于插值位姿，衡量规划轨迹本身）
   或 ``--tracker lqr``（阶段 C 闭环，含控制器跟踪误差）；跟踪器参考口径由
   ``--eval-reference`` 选择：``plan``（默认，6 步规划预览，历史口径）/
   ``repeat_action``（``repeat(action_mu, 6)``，与训练 A-hold 同构的语义定价口径）；
2. **内存纪律（§8.1）**：运行前程序化检查 ``MemAvailable >= eval.mem_available_floor_mb``，
   按 ``eval.train_pool_policy`` 调可选的训练池暂停钩子；spawn 进程池按
   ``eval.recycle_every_specs``（每 worker 每代处理的 spec 数）重建池 —— 照抄
   ``env/scenario/validator.py::validate_specs`` 的 ``specs_per_worker_per_pool`` 实现
   （MetaDrive 每 ``build_env``+``close`` 残留 ~3.5MB/spec）；
3. **KPI 口径**（config/eval.yaml 冻结）：success / collision / off_road / min_ttc /
   a_lon / a_lat / jerk / speed_ratio / solid_line_crossing / speed_limit_violation /
   route_completion；**速度取逐 env step 口径**（2026-09-26 修正）：``mean_speed_mps`` =
   episode 内逐 step 速度**真均值**，``final_speed_mps`` = 末步速度（旧 ``mean_speed_mps``
   的实际口径，保留对照），``low_speed_step_ratio`` / ``crawl_seconds`` = v < 2 m/s
   （``CRAWL_SPEED_MPS``，与 trainer ``probe/low_speed_alert`` 同阈值）的步占比 / 累计秒数；
   ckpt 路径另记实际下发动作 ``action_ds_mean_m`` / ``action_dtheta_abs_mean_rad``（m / rad，
   每 0.5 s 决策的 ``plan[0] == action_mu``），``steer_abs_mean`` / ``throttle_mean`` 仅在
   存在 [steer, throttle] 概念时有效（ckpt+exact 无此概念 → NaN/N-A，不静默填 0）；
   按 **primary 标签**（``spec.labels.geometry``）分组，其余附带标签单列
   ``compound``（报告口径，不判定）；每组 n >= ``per_category_n_min`` 才判定，并给 Wilson 95% CI；
4. 与冻结基线（``runs/baseline_eval/val_reference.json`` + ``..._by_primary.json``）比较，
   生成阈值判定：overall_success、collision/offroad ε 保护、route_completion、speed_ratio
   效率守卫、a_lat_mean/p95，以及弱类 floor（``target = max(baseline+0.15, 0.75)``，baseline<0.7）；
5. 输出 ``<out>/<name>/metrics.json``（overall + by_primary + compound + by_difficulty +
   by_geometry + verdict）与 ``episodes.csv``（逐 episode 明细）；
6. **P0-A 评测安全前置**：ckpt 加载默认 **fail-fast**（``missing/unexpected/shape_mismatch``
   任一非零 → 打印三计数与明细后 ``raise SystemExit(2)``，拒绝用部分加载的模型产出数字）；
   仅 ``--allow-partial-load``（显式）放行，放行时强制在日志与 ``metrics.json::meta.ckpt_load``
   记录三计数与 missing 列表；``metrics.json::meta`` 另落版本戳（evaluator / git commit /
   argv / ckpt sha256；config 快照由入口层 manifest 落盘）。干净加载行为逐位不变。
7. **R39-A1 工程闭合 + A1b 装配全开**（GPT §二-A 任务 1/2/4；遗留 2 关闭，零训练）：
   - **装配开关（全部默认 = 生产口径，逐位不变）**：``--path-head {off,structured}``、
     ``--ld-mode {baseline,…,semantic,semantic_no_chain}``、``--rollout-semantic
     {full,t0_only,static_role_only}``、``--lane-anchor {off,inject}``、``--fill-from <ckpt>``
     —— 口径对齐 ``.slim/deepwork/p1_v2/a3/eval_a3.py``（C1_HEAD = structured + semantic +
     ``--moe-off`` + fill K1e2 + inject 可经 ``tools/test.py`` 复现）；
   - ``--path-head structured`` 在装载前惰性构造 ``policy.structured_action`` 结构化头
     （``set_path_head(True)`` + ``set_path_head_ablation("structured")``，对齐 eval_a3）；``--ld-mode``/
     ``--rollout-semantic`` 非默认时先 ``set_ld_mode``/``set_rollout_semantic`` 再装载
     （惰性模块必须先建，否则 ckpt 键成 unexpected）；``--fill-from`` 只回填 migrate 后
     ``missing`` 中存在的键（口径 = a3 canary ``fill_from``）；
   - **正式加载路径接入 ckpt migration**（``tools.legacy.legacy_key_mapper`` +
     ``checkpoint_migration.migrate_state_dict``；干净无回填路径经统一原语
     ``load_checkpoint_migrated(strict=True)``）：已登记族（K-anchor 丢弃族 /
     ``path_head.*→policy.structured_action.*`` 改名族）逐族计数，未登记 unexpected 仍拒绝；
     **严格加载语义**：迁移 + 回填后 missing/unexpected/shape_mismatch 全零 ⇒ ``strict=True``
     语义装载；任一非零 ⇒ 默认 fail-fast（``--allow-partial-load`` 显式放行时才 ``strict=False``）；
   - **LoadReport**（missing/unexpected/filled/rejected/renamed/dropped 计数 + 全量键列表 +
     逐族计数 + ckpt 路径/sha256 + mapper 版本 + 装配轴/回填报告 + strict 判定）随加载摘要落
     ``metrics.json::meta`` 与 ``run_meta.json``；
   - **四 hash 原生**：``run_meta.json``（+ 既有 ``manifest.txt`` 追加 ``r39a1_hashes:`` 行）
     记录 ``evaluation_protocol_hash``（口径复用 ``.slim/deepwork/p1_v2/round38/c6_protocol.py``；
     正式入口轴映射：tracker/tail(=eval-reference)/moe/path_head + **ld_mode/rollout_semantic/
     lane_anchor/fill_from_sha16 入 hash**）、``model_tree_hash``（``net/**/*.py`` 树指纹，m2 口径）、
     ``schema_hash``（``env.obs.obs_fingerprint()``）、``checkpoint_hash``（ckpt 内容 sha256）；
   - **``lane_anchor_xy`` 口径（选项 a：缺键明确报错 + 显式注入开关）**：结构化头前向要求
     ``obs['lane_anchor_xy']``（目标车道中心线折线，ego t0 系），本入口 obs builder 不生产该键 ⇒
     ``--lane-anchor inject`` 显式开启（R36 C lane 注入器同源：``select_target_lane`` + E3 采样器；
     决策步 builder.build 后写入，与 a3 ``--lane-anchor inject`` 同点同口径）；不开注入而缺键 ⇒
     ``net/path_head.py::anchor_from_obs`` **fail-fast**（不静默回退），报错文案指路本开关与 a3。

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
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import subprocess
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
    "EVAL_REFERENCES",
    "KPI_DEFINITIONS",
    "build_eval_references",
    "task_model_config",
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
CRAWL_SPEED_MPS = 2.0             # crawl（低速蠕动）阈值：v < 2 m/s，与 trainer probe 同阈值
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
    "mean_speed_mps": "episode 内逐 env step 速度的真均值（m/s；v = info['velocity']，缺省回退 ego.speed）；"
                      "组内取逐 episode 均值",
    "final_speed_mps": "episode 末步速度（m/s；2026-09-26 前 mean_speed_mps 的实际口径，保留对照）",
    "low_speed_step_ratio": f"episode 内 v < {CRAWL_SPEED_MPS:g} m/s 的步占比（0–1）；组内取逐 episode 均值",
    "crawl_seconds": f"episode 内 v < {CRAWL_SPEED_MPS:g} m/s 的累计秒数（= 低速步数 × dt，dt=0.1 s）；"
                     "组内取逐 episode 均值",
    "steer_abs_mean": "|转向动作| 均值：baseline=PurePursuitIDMPolicy 归一化 steer∈[-1,1]；"
                      "ckpt+lqr=LqrTracker 实测归一化 steer（同单位）；ckpt+exact=运动学执行、"
                      "无 [steer,throttle] 动作概念 → NaN（N/A，不静默填 0）",
    "throttle_mean": "油门动作均值（可用性口径同 steer_abs_mean）",
    "action_ds_mean_m": "ckpt 路径实际下发的 |ds| 均值（m）：每 0.5 s 决策的 plan 首动作（= action_mu），"
                        "exact/lqr 同口径；baseline 无 (ds,dθ) 概念 → NaN（N/A）",
    "action_dtheta_abs_mean_rad": "ckpt 路径实际下发的 |dθ| 均值（rad；来源同 action_ds_mean_m）",
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


def _speed_metrics(speeds: Sequence[float], dt: float) -> Dict[str, float]:
    """逐 env step 速度 → 真均速 / 末步速度 / 低速步占比 / 蠕动秒数（纯函数，供单测覆盖）。

    - 均值用 :func:`_finite_mean`（过滤非有限值）；低速判据 ``v < CRAWL_SPEED_MPS``
      （与 trainer ``probe/low_speed_alert`` 的 speed<2 m/s 档同阈值）；
    - 步数分母为全部 step（非有限速度不判低速，但仍计一步）；无 step → 全 NaN。
    """
    if not speeds:
        return {
            "mean_speed_mps": float("nan"),
            "final_speed_mps": float("nan"),
            "low_speed_step_ratio": float("nan"),
            "crawl_seconds": float("nan"),
        }
    low_steps = sum(
        1 for value in speeds if math.isfinite(float(value)) and float(value) < CRAWL_SPEED_MPS
    )
    return {
        "mean_speed_mps": _finite_mean(speeds),
        "final_speed_mps": float(speeds[-1]),
        "low_speed_step_ratio": float(low_steps / len(speeds)),
        "crawl_seconds": float(low_steps * dt),
    }


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
        "final_speed_mps": _per_episode_mean("final_speed_mps"),
        "low_speed_step_ratio": _per_episode_mean("low_speed_step_ratio"),
        "crawl_seconds": _per_episode_mean("crawl_seconds"),
        "steer_abs_mean": _per_episode_mean("steer_abs_mean"),
        "throttle_mean": _per_episode_mean("throttle_mean"),
        "action_ds_mean_m": _per_episode_mean("action_ds_mean_m"),
        "action_dtheta_abs_mean_rad": _per_episode_mean("action_dtheta_abs_mean_rad"),
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
            # 规则基线的动作 = (steer, throttle)：无 (ds,dθ) 概念 → N/A（不静默填 0）
            "action_ds_m": float("nan"),
            "action_dtheta_rad": float("nan"),
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


_CKPT_MODEL_CACHE: Dict[Tuple, Any] = {}
#: P0-A 版本戳/审计：每个装配键（见 :func:`_model_cache_key`）最近一次成功加载的摘要
#: （含三计数、明细列表与 R39-A1 migration LoadReport）。
_CKPT_LOAD_SUMMARY: Dict[Tuple, Dict[str, Any]] = {}


#: R39-A1（GPT §二-A 任务 1）/ A1b（遗留 2 关闭）：正式入口**模型装配开关**——口径对齐
#: ``.slim/deepwork/p1_v2/a3/eval_a3.py`` 的 canary 装载顺序；全默认 = 生产口径（逐位不变）。
#: - ``--path-head``：结构化头 ``policy.structured_action``（off=默认旧 PolicyHead）；
#: - ``--ld-mode``：LD 消融（ckpt 必须与训练同模式装载；semantic = R21-3A 语义接线）；
#: - ``--rollout-semantic``：rollout 阶段 semantic 注入口径；
#: - ``--lane-anchor``：``lane_anchor_xy`` 注入器（选项 a 的显式开关；off=默认不触达）；
#: - ``--fill-from``：缺键回填源 ckpt（K1e2 血统语义键；显式给出才触达）；
#: - ``--world-state-api`` / ``--world-residual-head``：R39-B2c 统一 WorldState API
#:   （off=默认不触达，与 A1b 口径逐字节同 hash；非默认需 ``--path-head structured``，
#:   learned 档由独立头 ckpt 提供 ΔA_θ；装配轴非默认时入 hash）。
LOAD_PATH_HEAD_CHOICES: tuple[str, ...] = ("off", "structured")
LOAD_LD_MODES: tuple[str, ...] = (
    "baseline", "no_chain", "set_attn", "attn", "attn_no_chain", "semantic", "semantic_no_chain",
)
LOAD_ROLLOUT_SEMANTIC_MODES: tuple[str, ...] = ("full", "t0_only", "static_role_only")
LOAD_LANE_ANCHOR_MODES: tuple[str, ...] = ("off", "inject")
#: R39-B2c：WorldState API 档位（net 侧 ``net.model.WORLD_STATE_API_MODES`` 的闭环子集；
#: ``analytic_residual`` 属离线诊断面不开放）。``off`` = 生产口径（逐位不变）。
LOAD_WORLD_STATE_API_MODES: tuple[str, ...] = ("off", "analytic", "analytic_learned")
#: R41-A1：teacher-forced oracle 注入档（net 侧 ``net.model.TEACHER_FORCE_MODES`` 全集的闭环面）。
#: ``off`` = 默认零触达（不装 hook、不新增键、前向逐位不变）。
LOAD_TEACHER_FORCE_MODES: tuple[str, ...] = (
    "off", "ego", "od", "ld", "ego_od_ld", "lateral", "all",
)
#: R41-A1：速度反事实档（``off`` = 零触达；``expert`` = 专家空问 ds 缩放，需 pure_pursuit labeler）。
LOAD_SPEED_ORACLE_MODES: tuple[str, ...] = ("off", "expert")
#: R41-D：几何口径实例配置档（正式入口 ``--readout-lane-geometry``；``default`` = 未配置哨兵
#: ⇒ 沿用 config/ckpt metadata/历史默认）。``fixed``/``production`` = 历史内部值别名
#: （映射 ``fixed``→``corrected``、``production``→``legacy_aligned``）。
LOAD_READOUT_LANE_GEOMETRY_MODES: tuple[str, ...] = (
    "default", "corrected", "legacy_aligned", "corrected_t0margins", "fixed", "production",
)

# --------------------------------------------------------------------------- #
# R41-A1：oracle/teacher-force/speed 反事实轴（五要件：CLI ∧ 模型执行路径 ∧ protocol hash
# ∧ 模型缓存键 ∧ metrics/manifest；默认关 ⇒ 逐位一致）
# --------------------------------------------------------------------------- #
#: truth npz 文件名后缀（两遍法：``<truth_dir>/<spec_id:04d>.npz``）。
TRUTH_NPZ_SUFFIX = ".npz"
#: 注入视野（想象步 k=1..6；= net rollout_steps = V4B 记载口径）。
TEACHER_FORCE_HORIZON = 6
#: 决策帧记录键（V4B 兼容集 + R41-A1 扩展：动作条件/漂移证据）。
TRUTH_FRAME_KEYS: tuple[str, ...] = (
    "frame_env_step", "ego_pose_world", "ego_speed",
    "od", "od_mask", "od_presence",
    "ld", "ld_mask",
    "kappa_expert", "ds_expert", "expert_valid",
    "action_exec", "plan_exec", "rng_digest",
)
#: 逐 env step 轨迹键（五类漂移中的"逐步状态误差"口径）。
TRUTH_STEP_KEYS: tuple[str, ...] = ("step_index", "step_pose", "step_speed", "step_rng_digest")
#: episode 级键（termination 一致性口径）。
TRUTH_EPISODE_KEYS: tuple[str, ...] = ("termination_code", "n_steps", "meta_json")
#: termination 名称 → 码（npz 只存整型；名称映射 = 本表 + ``_run_episode`` 的判定串）。
TERMINATION_CODES: dict[str, int] = {
    "arrive_dest": 0, "collision": 1, "out_of_road": 2, "max_step": 3, "other": 4, "error": 5,
}
#: 决策帧"同状态门"（预检硬门：超出 ⇒ 该帧所有注入自动拒绝——非法组合不得进入模型路径）。
DEFAULT_FRAME_GATE: Dict[str, float] = {
    "pose_pos_m": 0.5,        # 位置误差（m；帧级 max）
    "heading_rad": 0.15,      # 航向误差（rad；帧级 max，wrap 后）
    "od_mean_abs": 0.5,       # OD 特征均绝对差（掩码内）
    "rng_required": 1.0,      # 1 = 要求 RNG 摘要一致（无数据时按"不可用"放行并计账）
}
#: 两遍重放五类漂移门（预注册默认；config ``eval.counterfactual.drift_thresholds`` 可覆盖）。
#: 超标 episode ⇒ ``oracle_eligible=False``（剔除出 oracle 裁决视图）。
DEFAULT_DRIFT_THRESHOLDS: Dict[str, float] = {
    "pose_p95_m": 0.5,             # ① replay pose divergence：决策帧位置误差 p95（m）
    "pose_max_m": 2.0,             # ① 决策帧位置误差 max（m）
    "heading_p95_rad": 0.15,       # ① 航向误差 p95（rad，wrap 后）
    "od_mean_abs": 0.25,           # ② OD state divergence：掩码内均绝对差
    "od_max_abs": 2.0,             # ② 掩码内最大绝对差
    "rng_match_frac_min": 0.98,    # ③ traffic RNG 一致性：决策帧 RNG 摘要一致率下限
    "stepwise_pos_p95_m": 0.5,     # ⑤ 逐步状态误差：逐 env step 位置误差 p95（m）
    "stepwise_speed_mean_abs": 0.5,  # ⑤ 逐 env step 速度误差均值（m/s）
    "termination_step_tol": 2.0,   # ④ termination 一致性：步数容差（解）
    "illegal_frac_max": 0.05,      # 动作条件非法注入步占比上限（超出 ⇒ 剔除）
    "unproven_frac_max": 0.05,     # 动作条件不可判定步占比上限（超出 ⇒ 剔除）
    "action_tol_ds_m": 5e-3,       # 命令动作条件容差（ds，m）
    "action_tol_dtheta_rad": 5e-3,  # 命令动作条件容差（dθ，rad）
    "realized_tol_ds_m": 5e-3,     # 实测（位姿派生）动作条件容差（ds，m）
    "realized_tol_dtheta_rad": 5e-3,  # 实测动作条件容差（dθ，rad）
}


# --------------------------------------------------------------------------- #
# R41-D：几何口径实例配置轴（五要件：CLI ∧ 模型构造参数/config ∧ protocol hash ∧
# 模型缓存键 ∧ metrics/manifest；默认哨兵 ⇒ 逐位一致）
# --------------------------------------------------------------------------- #
#: ckpt ``meta`` 的几何口径键（= ``net.model.DrivingModel.readout_lane_geometry``；
#: 保存侧 = ``pipeline.stages.save_checkpoint`` 注入，字面量同源）。
CKPT_GEOMETRY_META_KEY = "readout_lane_geometry"


def _normalize_readout_lane_geometry(value: Any) -> Optional[str]:
    """取值 → canonical（``corrected``/``legacy_aligned``）；哨兵/None ⇒ ``None``；未知 ⇒ ValueError。"""
    if value is None or str(value).strip().lower() in ("", "default"):
        return None
    from net.path_head import normalize_readout_lane_geometry  # lazy（baseline 路径不 import torch 面）

    return normalize_readout_lane_geometry(str(value))


def _resolve_readout_lane_geometry(
    *,
    requested: Optional[str] = None,
    requested_source: str = "cli",
    ckpt_meta: Optional[str] = None,
    config_value: Optional[str] = None,
    ckpt_meta_raw: Optional[str] = None,
) -> Dict[str, Any]:
    """几何口径解析：**CLI/config（requested） > ckpt metadata > model config > 默认哨兵**。

    - ``requested`` = CLI ``--readout-lane-geometry`` 或 config 键的 canonical 值（``None`` = 未声明）；
    - ``requested_source`` = ``"cli"`` / ``"model_config"``（值来源标注）；
    - ``ckpt_meta`` = ckpt ``meta["readout_lane_geometry"]`` 的 canonical 值（``None`` = 缺失/哨兵）；
    - ``config_value``：单独记录 config 侧声明（= requested 当它来自 config 时同值）。

    返回记录：``{value: canonical|"default", source, requested, ckpt_meta, config,
    conflict(声明不一致)}``。纯函数（无 IO/torch）。
    """
    source = "default"
    value = "default"
    if requested is not None:
        value, source = requested, str(requested_source or "cli")
    elif ckpt_meta is not None:
        value, source = ckpt_meta, "checkpoint_metadata"
    elif config_value is not None:
        value, source = config_value, "model_config"
    conflict = bool(requested is not None and ckpt_meta is not None and requested != ckpt_meta)
    return {
        "value": value,
        "source": source,
        "requested": requested,
        "ckpt_meta": (ckpt_meta if ckpt_meta is not None else ckpt_meta_raw),
        "config": config_value,
        "conflict": conflict,
    }


def _readout_geometry_config_value(config: Mapping[str, Any] | None) -> Optional[str]:
    """``task_model_config``（或 config/model.yaml 根层）的几何键 → canonical；未声明 ⇒ ``None``。"""
    if not isinstance(config, Mapping):
        return None
    return _normalize_readout_lane_geometry(config.get("readout_lane_geometry"))


def _geometry_artifact_record(resolution: Mapping[str, Any]) -> Dict[str, Any]:
    """解析记录 → 产物口径（run_meta/metrics/load_report 同键名 ``readout_lane_geometry``）。"""
    return {
        "readout_lane_geometry": resolution.get("value"),
        "source": resolution.get("source"),
        "requested": resolution.get("requested"),
        "ckpt_meta": resolution.get("ckpt_meta"),
        "config": resolution.get("config"),
        "conflict": bool(resolution.get("conflict")),
    }


def _model_cache_key(
    ckpt: str,
    device: str,
    *,
    path_head: str = "off",
    ld_mode: str = "baseline",
    rollout_semantic: str = "full",
    fill_from: Optional[str] = None,
    world_state_api: str = "off",
    world_residual_head: Optional[str] = None,
    world_residual_scale: Optional[float] = None,
    teacher_force: str = "off",
    record_truth: bool = False,
    inject_truth_sha16: Optional[str] = None,
    speed_scale: Optional[float] = None,
    speed_oracle: str = "off",
    readout_lane_geometry: Optional[str] = None,
) -> Tuple:
    """缓存/加载摘要键：**全默认装配 = 旧键 ``(ckpt, device)``**（历史调用点逐位不变；
    a3 controller 的 canary 预填缓存按该旧键命中）；任何非默认轴追加一段避免互撞。

    R41-A1（五要件之"模型缓存键"）：teacher_force/record_truth/inject_truth_sha16/speed_scale/
    speed_oracle 五轴**任一非默认**时整体追加 ``("counterfactual", …)`` 段（默认全关 ⇒ 逐位
    不改旧键）。五轴同段记录：同进程内不同反事实语义绝不共享同一模型装配。

    R41-D（几何五要件之"模型缓存键"）：``readout_lane_geometry`` 非默认（canonical/别名）时
    追加 ``("readout_lane_geometry", <canonical>)`` 段；哨兵/None ⇒ 不追加（旧键逐位不变）。
    """
    key: Tuple = (str(ckpt), str(device))
    if str(path_head or "off") != "off":
        key += (("path_head", str(path_head)),)
    if str(ld_mode or "baseline") != "baseline":
        key += (("ld_mode", str(ld_mode)),)
    if str(rollout_semantic or "full") != "full":
        key += (("rollout_semantic", str(rollout_semantic)),)
    if fill_from:
        key += (("fill_from", str(fill_from)),)
    if str(world_state_api or "off") != "off":
        # R39-B2c：非默认档单独缓存（默认 off 时不改旧键 ⇒ 历史调用点逐位不变）
        key += (("world_state_api", str(world_state_api)),
                ("world_residual_head", (str(world_residual_head) if world_residual_head else None)))
        if world_residual_scale is not None and float(world_residual_scale) != 1.0:
            # R39-B2d：非默认残差缩放单独缓存（默认 None / 1.0 时不改旧键）
            key += (("world_residual_scale", float(world_residual_scale)),)
    # R41-D：几何实例配置轴（哨兵/None ⇒ 不追加段）
    geometry_value = _normalize_readout_lane_geometry(readout_lane_geometry)
    if geometry_value is not None:
        key += (("readout_lane_geometry", geometry_value),)
    # R41-A1：反事实五轴（默认全关 ⇒ 不追加段，旧键逐位不变）
    speed_scale_norm = None if (speed_scale is None or float(speed_scale) == 1.0) else float(speed_scale)
    counterfactual_active = (
        str(teacher_force or "off") != "off"
        or bool(record_truth)
        or bool(inject_truth_sha16)
        or speed_scale_norm is not None
        or str(speed_oracle or "off") != "off"
    )
    if counterfactual_active:
        key += ((
            "counterfactual",
            str(teacher_force or "off"),
            bool(record_truth),
            (str(inject_truth_sha16) if inject_truth_sha16 else None),
            speed_scale_norm,
            str(speed_oracle or "off"),
        ),)
    return key


def _sha256_file(path: str) -> str:
    """ckpt 内容 sha256（版本戳；读取失败返回 ``"unknown"``，不阻断评测）。"""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return "unknown"


def _git_commit() -> str:
    """HEAD commit（版本戳；取证尽力而为，失败返回 ``"unknown"``）。"""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
            capture_output=True, text=True, timeout=5, check=False,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - 取证字段不阻断评测
        return "unknown"


# --------------------------------------------------------------------------- #
# R39-A1：四 hash 原生记录（evaluation_protocol_hash / model tree / schema / ckpt）
# --------------------------------------------------------------------------- #
#: 口径来源（只读复用；路径以仓内实际为准）：
#: - ``evaluation_protocol_hash`` → ``.slim/deepwork/p1_v2/round38/c6_protocol.py``
#:   （R38 C6 统一闭环基准；canonical JSON + sha256 hex16；正式入口按本文件映射调用）；
#: - ``model tree hash`` → ``.slim/deepwork/p1_v2/round37/m2_net_sha.py``（``net/**/*.py`` 树指纹）；
#: - ``schema hash`` → ``env.obs.obs_fingerprint()``（obs 实现 + schema 版本短指纹）；
#: - ``checkpoint hash`` → 本文件 ``_sha256_file``（ckpt 内容 sha256）。
R39_PROTOCOL_HASH_SCRIPT = ".slim/deepwork/p1_v2/round38/c6_protocol.py"
R39_NET_SHA_SCRIPT = ".slim/deepwork/p1_v2/round37/m2_net_sha.py"


def _load_readonly_script(name: str, rel_path: str) -> Any:
    """只读装载仓内工具脚本（importlib；口径单一出处，避免重复实现）。"""
    import importlib.util as importlib_util

    path = Path(_PROJECT_ROOT) / rel_path
    if not path.is_file():
        raise FileNotFoundError(
            f"[eval_runner] R39-A1 缺 hash 口径工具 {rel_path}（四 hash 原生记录依赖；不静默跳过）"
        )
    spec = importlib_util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"[eval_runner] 无法装载 {rel_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _net_tree_digest() -> Dict[str, Any]:
    """``net/**/*.py`` 树指纹（m2 口径：逐文件 sha16 → 排序连接哨兵 sha256 前 16 hex）。"""
    module = _load_readonly_script("r39a1_m2_net_sha", R39_NET_SHA_SCRIPT)
    return dict(module.net_tree_digest())


def _formal_entry_protocol(
    *,
    spec_path: Any,
    policy: str,
    tracker: str,
    eval_reference: str,
    moe_off: bool,
    path_head: str,
    ld_mode: str = "baseline",
    rollout_semantic: str = "full",
    lane_anchor: str = "off",
    fill_from_sha16: Optional[str] = None,
    workers: int,
    max_steps: int,
    device: Optional[str],
    net_tree_sha16: str,
    world_state_api: str = "off",
    world_residual_head_sha16: Optional[str] = None,
    world_residual_scale: Optional[float] = None,
    teacher_force: str = "off",
    record_truth: bool = False,
    inject_truth_sha16: Optional[str] = None,
    speed_scale: Optional[float] = None,
    speed_oracle: str = "off",
    readout_lane_geometry: Optional[str] = None,
) -> Dict[str, Any]:
    """正式入口 evaluation protocol（口径复用 c6：``build_protocol`` + canonical JSON sha256 hex16）。

    轴映射（正式入口 → c6 协议字段；记录在 ``notes``）：
    ``control.tracker ← --tracker``；``control.tail ← --eval-reference``（正式入口参考口径）；
    ``model.ld_mode ← --ld-mode`` / ``model.rollout_semantic ← --rollout-semantic``（R39-A1b 起为
    正式入口开关，**入 hash**）；``model.moe ← --moe-off``；``model.path_head ← --path-head``；
    R39-A1b 追加 ``model.lane_anchor``（``--lane-anchor`` 注入开关）与 ``model.fill_from_sha16``
    （``--fill-from`` 源 ckpt 内容指纹；none）—— 均**入 hash**（模型输入/权重装配的可辨识性）。
    R39-B2c：``model.world_state_api`` / ``model.world_residual_head_sha16`` **条件入 hash**——
    默认口径（``off``/none）时 protocol dict 与上面口径**逐字节相同**（历史 hash 可比性保持），
    非默认档才追加两键（装配轴敏感，避免不同档共享同一 hash）。
    R41-D：``model.readout_lane_geometry``（canonical ``corrected``/``legacy_aligned``；解析结果
    含 ckpt metadata 采用值）**条件入 hash**——哨兵/未声明 ⇒ 不追加键（逐字节相同）。
    """
    module = _load_readonly_script("r39a1_c6_protocol", R39_PROTOCOL_HASH_SCRIPT)
    protocol = module.build_protocol(
        spec_path=Path(str(spec_path)),
        set_name=Path(str(spec_path)).stem,
        policy_kind=str(policy),
        entry="pipeline/eval_runner.py",
        tracker=str(tracker),
        tail=str(eval_reference),
        ld_mode=str(ld_mode),
        rollout_semantic=str(rollout_semantic),
        moe=("off" if moe_off else "on"),
        workers=int(workers),
        max_steps=int(max_steps),
        device=str(device or ""),
        net_tree_sha16=str(net_tree_sha16),
        notes=(
            "R39-A1/A1b 正式入口协议（tools/test.py / pipeline.eval_runner.py）："
            "ld_mode/rollout_semantic/path_head/lane_anchor/fill_from_sha16 均为正式入口装配开关"
            "（入 hash；缺省 = 生产口径 baseline/full/off/off/none）"
        ),
    )
    protocol["model"]["path_head"] = str(path_head)
    protocol["model"]["lane_anchor"] = str(lane_anchor)
    protocol["model"]["fill_from_sha16"] = (str(fill_from_sha16) if fill_from_sha16 else "none")
    # R39-B2c：WorldState API 轴条件入 hash（默认 off/none ⇒ 不追加键 ⇒ 与 A1b 口径逐字节同）
    if str(world_state_api or "off") != "off":
        protocol["model"]["world_state_api"] = str(world_state_api)
        protocol["model"]["world_residual_head_sha16"] = (
            str(world_residual_head_sha16) if world_residual_head_sha16 else "none"
        )
        if world_residual_scale is not None and float(world_residual_scale) != 1.0:
            # R39-B2d：残差缩放入 hash（默认 None / 1.0 ⇒ 不追加键，保持逐字节可比）
            protocol["model"]["world_residual_scale"] = float(world_residual_scale)
    # R40-V4B：teacher-forced oracle / 速度反事实轴**条件入 hash**（默认 off/None/1.0 ⇒
    # protocol dict 与上面口径逐字节相同；非默认档才追加键 ⇒ 不同档不共享同一 hash）。
    if str(teacher_force or "off") != "off":
        protocol["model"]["teacher_force"] = str(teacher_force)
        protocol["model"]["inject_truth_sha16"] = (
            str(inject_truth_sha16) if inject_truth_sha16 else "none"
        )
    if record_truth:
        protocol["model"]["record_truth"] = "on"
    if speed_scale is not None and float(speed_scale) != 1.0:
        protocol["model"]["speed_scale"] = float(speed_scale)
    if str(speed_oracle or "off") != "off":
        protocol["model"]["speed_oracle"] = str(speed_oracle)
    # R41-D：几何实例配置轴条件入 hash（哨兵/未声明 ⇒ 不追加键 ⇒ 历史 hash 逐字节可比）
    geometry_value = _normalize_readout_lane_geometry(readout_lane_geometry)
    if geometry_value is not None:
        protocol["model"]["readout_lane_geometry"] = geometry_value
    protocol["hashes"] = {
        "protocol_hash": module.hash_protocol(protocol),
        "protocol_hash_full": module.hash_protocol(protocol, full=True),
        "protocol_common_hash": module.common_hash(protocol),
    }
    return protocol


def _evaluation_hashes(
    *,
    spec_path: Any,
    policy: str,
    ckpt: Optional[str],
    tracker: str,
    eval_reference: str,
    moe_off: bool,
    path_head: str,
    ld_mode: str = "baseline",
    rollout_semantic: str = "full",
    lane_anchor: str = "off",
    fill_from: Optional[str] = None,
    workers: int,
    max_steps: int,
    device: Optional[str],
    world_state_api: str = "off",
    world_residual_head: Optional[str] = None,
    world_residual_scale: Optional[float] = None,
    teacher_force: str = "off",
    record_truth: bool = False,
    inject_truth_sha16: Optional[str] = None,
    speed_scale: Optional[float] = None,
    speed_oracle: str = "off",
    readout_lane_geometry: Optional[str] = None,
    readout_lane_geometry_source: Optional[str] = None,
) -> Dict[str, Any]:
    """R39-A1 四 hash（+ 完整 protocol dict；供 run_meta/manifest 与 metrics 记录）。

    R39-B2c：``world_state_api`` / ``world_residual_head`` 可选装配轴（默认 off/none ⇒ 与
    A1b 口径逐字节相同；非默认档入 protocol hash，见 :func:`_formal_entry_protocol`）。
    R39-B2d：``world_residual_scale``（默认 none ⇒ 不触达；非默认 ≠1.0 入 hash 轴）。
    R40-V4B：``teacher_force``/``record_truth``/``inject_truth_sha16``/``speed_scale``/
    ``speed_oracle``（默认 off/False/none/none/off ⇒ **不追加任何键**，保持逐字节可比；
    非默认档入 hash 轴 —— 闭环 teacher-forced oracle 臂与速度反事实的可辨识性）。
    R41-D：``readout_lane_geometry``（解析结果；``None``/哨兵 ⇒ 不追加键）+ ``source`` 记录。
    """
    net_tree = _net_tree_digest()
    from env.obs import OBS_SCHEMA_VERSION, obs_fingerprint  # 轻量（stdlib only）

    fill_sha = _sha256_file(str(fill_from)) if fill_from else None
    head_sha = _sha256_file(str(world_residual_head)) if world_residual_head else None
    protocol = _formal_entry_protocol(
        spec_path=spec_path, policy=policy, tracker=tracker, eval_reference=eval_reference,
        moe_off=moe_off, path_head=path_head, ld_mode=ld_mode, rollout_semantic=rollout_semantic,
        lane_anchor=lane_anchor, fill_from_sha16=(fill_sha[:16] if fill_sha else None),
        workers=workers, max_steps=max_steps, device=device,
        net_tree_sha16=str(net_tree["tree_sha16"]),
        world_state_api=str(world_state_api or "off"),
        world_residual_head_sha16=(head_sha[:16] if head_sha else None),
        world_residual_scale=world_residual_scale,
        teacher_force=str(teacher_force or "off"),
        record_truth=bool(record_truth),
        inject_truth_sha16=(str(inject_truth_sha16) if inject_truth_sha16 else None),
        speed_scale=(None if speed_scale is None else float(speed_scale)),
        speed_oracle=str(speed_oracle or "off"),
        readout_lane_geometry=_normalize_readout_lane_geometry(readout_lane_geometry),
    )
    ckpt_sha = _sha256_file(str(ckpt)) if ckpt is not None else None
    geometry_value = _normalize_readout_lane_geometry(readout_lane_geometry)
    return {
        # 四 hash 主口径（字符串）
        "evaluation_protocol_hash": protocol["hashes"]["protocol_hash"],
        "model_tree_hash": str(net_tree["tree_sha16"]),
        "schema_hash": str(obs_fingerprint()),
        "checkpoint_hash": ckpt_sha,
        # 辅助记录（不改主口径）
        "evaluation_protocol_hash_full": protocol["hashes"]["protocol_hash_full"],
        "evaluation_protocol_common_hash": protocol["hashes"]["protocol_common_hash"],
        "checkpoint_hash16": (ckpt_sha[:16] if isinstance(ckpt_sha, str) and len(ckpt_sha) == 64 else None),
        "checkpoint_path": str(ckpt) if ckpt is not None else None,
        "fill_from": ({"path": str(fill_from), "sha256": fill_sha,
                       "sha16": (fill_sha[:16] if fill_sha else None)} if fill_from else None),
        "model_tree": {"root": "net", "n_files": int(net_tree["n_files"]),
                       "tree_sha16": str(net_tree["tree_sha16"]),
                       "tool": R39_NET_SHA_SCRIPT},
        "schema": {"tool": "env.obs.obs_fingerprint", "obs_schema_version": int(OBS_SCHEMA_VERSION)},
        # R39-B2c：WorldState API 轴留痕（默认 off/none ⇒ {"api": "off", "residual_head": None}）
        # R39-B2d：residual_scale 仅非默认（≠None/1.0）时追加（默认 payload 逐字节不变）
        "world_state": {
            "api": str(world_state_api or "off"),
            "residual_head": ({"path": str(world_residual_head), "sha256": head_sha,
                               "sha16": (head_sha[:16] if head_sha else None)}
                              if world_residual_head else None),
            **({"residual_scale": float(world_residual_scale)}
               if world_residual_scale is not None and float(world_residual_scale) != 1.0 else {}),
        },
        # R41-D：几何口径解析留痕（非默认才追加键 ⇒ 默认 payload 逐字节不变）
        **({"geometry": {
            "readout_lane_geometry": geometry_value,
            "source": (str(readout_lane_geometry_source) if readout_lane_geometry_source else "requested"),
        }} if geometry_value is not None else {}),
        "protocol": protocol,
    }


def _record_run_meta(out_dir: Path, payload: Dict[str, Any]) -> Path:
    """写 ``<out>/run_meta.json``（R39-A1 四 hash + LoadReport + protocol）并登记 manifest。

    manifest（入口层 ``run_paths.write_manifest`` 产物，若存在）追加单行
    ``r39a1_hashes: {…}``（幂等：已有标记则不重复追加）；无 manifest 时只写 run_meta.json。
    """
    meta_path = out_dir / "run_meta.json"
    meta_path.write_text(
        json.dumps(_sanitize(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = out_dir / "manifest.txt"
    marker = "r39a1_hashes:"
    if manifest.is_file():
        text = manifest.read_text(encoding="utf-8")
        if marker not in text:
            line = json.dumps(
                {
                    "evaluation_protocol_hash": payload["hashes"]["evaluation_protocol_hash"],
                    "model_tree_hash": payload["hashes"]["model_tree_hash"],
                    "schema_hash": payload["hashes"]["schema_hash"],
                    "checkpoint_hash": payload["hashes"]["checkpoint_hash"],
                },
                ensure_ascii=False, sort_keys=True,
            )
            manifest.write_text(text.rstrip("\n") + "\n" + marker + " " + line + "\n", encoding="utf-8")
        # R41-A1：反事实轴非默认时追加 manifest 行（默认关 ⇒ 不改 manifest，逐位一致）
        cf_marker = "r41a1_counterfactual:"
        counterfactual = payload.get("counterfactual")
        if isinstance(counterfactual, Mapping) and counterfactual.get("active") and cf_marker not in text:
            cf_line = json.dumps(
                {
                    "teacher_force": counterfactual.get("teacher_force"),
                    "record_truth": counterfactual.get("record_truth"),
                    "inject_truth": counterfactual.get("inject_truth"),
                    "inject_truth_sha16": counterfactual.get("inject_truth_sha16"),
                    "speed_scale": counterfactual.get("speed_scale"),
                    "speed_oracle": counterfactual.get("speed_oracle"),
                    "evaluation_protocol_hash": payload["hashes"]["evaluation_protocol_hash"],
                    "cache_key_axes": "counterfactual(=五轴同段)",
                },
                ensure_ascii=False, sort_keys=True,
            )
            manifest.write_text(
                manifest.read_text(encoding="utf-8").rstrip("\n") + "\n" + cf_marker + " " + cf_line + "\n",
                encoding="utf-8",
            )
        # R41-D：几何实例配置非默认时追加 manifest 行（哨兵 ⇒ 不改 manifest，逐位一致）
        geometry_marker = "r41d_geometry:"
        geometry = payload.get("geometry")
        if isinstance(geometry, Mapping) and geometry_marker not in text:
            geometry_line = json.dumps(
                {
                    "readout_lane_geometry": geometry.get("readout_lane_geometry"),
                    "source": geometry.get("source"),
                    "evaluation_protocol_hash": payload["hashes"]["evaluation_protocol_hash"],
                    "cache_key_axes": "readout_lane_geometry",
                },
                ensure_ascii=False, sort_keys=True,
            )
            manifest.write_text(
                manifest.read_text(encoding="utf-8").rstrip("\n") + "\n" + geometry_marker + " "
                + geometry_line + "\n",
                encoding="utf-8",
            )
    return meta_path


def _ckpt_load_summary(
    ckpt: str,
    device: str = "cpu",
    path_head: str = "off",
    ld_mode: str = "baseline",
    rollout_semantic: str = "full",
    fill_from: Optional[str] = None,
    *,
    world_state_api: str = "off",
    world_residual_head: Optional[str] = None,
    world_residual_scale: Optional[float] = None,
    teacher_force: str = "off",
    record_truth: bool = False,
    inject_truth_sha16: Optional[str] = None,
    speed_scale: Optional[float] = None,
    speed_oracle: str = "off",
    readout_lane_geometry: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """取 ``_load_ckpt_model`` 最近一次成功加载的摘要（未加载过 → None）。"""
    return _CKPT_LOAD_SUMMARY.get(_model_cache_key(
        ckpt, device, path_head=path_head, ld_mode=ld_mode,
        rollout_semantic=rollout_semantic, fill_from=fill_from,
        world_state_api=world_state_api, world_residual_head=world_residual_head,
        world_residual_scale=world_residual_scale,
        teacher_force=teacher_force, record_truth=record_truth,
        inject_truth_sha16=inject_truth_sha16, speed_scale=speed_scale,
        speed_oracle=speed_oracle, readout_lane_geometry=readout_lane_geometry,
    ))


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


def _load_ckpt_model(
    ckpt: str,
    config: Mapping[str, Any],
    device: str = "cpu",
    *,
    allow_partial: bool = False,
    path_head: str = "off",
    ld_mode: str = "baseline",
    rollout_semantic: str = "full",
    fill_from: Optional[str] = None,
    world_state_api: str = "off",
    world_residual_head: Optional[str] = None,
    world_residual_scale: Optional[float] = None,
    teacher_force: str = "off",
    record_truth: bool = False,
    inject_truth_sha16: Optional[str] = None,
    speed_scale: Optional[float] = None,
    speed_oracle: str = "off",
    readout_lane_geometry: Optional[str] = None,
    readout_lane_geometry_source: Optional[str] = None,
) -> Any:
    """worker 内按装配键（``_model_cache_key``；全默认 = ``(ckpt, device)``）缓存模型。

    R39-A1（GPT §二-A 任务 1/2）+ A1b（遗留 2 关闭）：正式加载路径接入 ckpt migration
    （``tools.legacy.legacy_key_mapper`` + ``checkpoint_migration.migrate_state_dict``）：

    - **装配（装载前，顺序对齐 a3 canary ``load_model_ce``）**：
      ``set_ld_mode``（非 baseline）→ ``set_rollout_semantic``（非 full）→
      ``--path-head structured`` 惰性构造 ``policy.structured_action``；
      ``--lane-anchor inject`` 属 controller 侧（每决策步注入 ``lane_anchor_xy``，见
      :class:`_CkptController`）；
    - **已登记族**（K-anchor 丢弃族 / ``path_head.*→policy.structured_action.*`` 改名族）从
      unexpected 里移走并逐族计数（LoadReport）；未登记 unexpected 键仍 = 拒绝（reject）；
    - **缺键回填（``--fill-from``，显式）**：migrate 后 ``missing`` 中的键若存在于回填 ckpt
      且形状一致 ⇒ 装载并计入 LoadReport ``fill_from``（口径 = a3 canary ``fill_from``；
      不给则零触达）；
    - **严格加载语义（明确）**：迁移 + 回填后 ``missing/unexpected/shape_mismatch`` 任一非零 ⇒
      **默认 fail-fast**（``raise SystemExit(2)``，打印三计数与明细）；三者全零时以
      ``strict=True`` 语义装载（无回填 ⇒ 统一原语 ``load_checkpoint_migrated(strict=True)``；
      有回填 ⇒ 合并子集 ``strict=True``）；仅 ``--allow-partial-load``（显式）放行并以
      ``strict=False`` 装载（形状过滤子集）。干净加载行为逐位不变；
    - **LoadReport**（``summary["load_report"]``）：missing/unexpected/filled/rejected/
      renamed/dropped 计数 + 全量键列表 + 逐族计数 + ckpt 路径/sha256 + mapper 版本 +
      装配轴/回填报告 + strict 判定，供 run_meta/manifest 落盘审计。
    - **R41-D 几何口径**（``readout_lane_geometry``；默认 None/哨兵 ⇒ 零触达）：解析顺序 =
      ``requested``（CLI/config 显式）> ckpt ``meta["readout_lane_geometry"]`` > config 键 >
      默认哨兵；非默认 ⇒ ``model.set_readout_lane_geometry(...)``（实例配置落地），并把
      解析记录写入 ``load_report/summary["readout_lane_geometry"]``（有轴/有 meta 才写 ⇒
      纯 legacy 默认逐字节不变）。冲突（requested ≠ meta）⇒ 警告 + 按 requested 执行（记录）。
    """
    path_head = str(path_head or "off")
    ld_mode = str(ld_mode or "baseline")
    rollout_semantic = str(rollout_semantic or "full")
    world_state_api = str(world_state_api or "off")
    if path_head not in LOAD_PATH_HEAD_CHOICES:
        raise ValueError(f"未知 path_head={path_head!r}（可选 {'/'.join(LOAD_PATH_HEAD_CHOICES)}）")
    if ld_mode not in LOAD_LD_MODES:
        raise ValueError(f"未知 ld_mode={ld_mode!r}（可选 {'/'.join(LOAD_LD_MODES)}）")
    if rollout_semantic not in LOAD_ROLLOUT_SEMANTIC_MODES:
        raise ValueError(
            f"未知 rollout_semantic={rollout_semantic!r}（可选 {'/'.join(LOAD_ROLLOUT_SEMANTIC_MODES)}）"
        )
    # R39-B2c：WorldState API 轴（默认 off ⇒ 零触达；非默认需结构化头；learned 需头 ckpt）
    if world_state_api not in LOAD_WORLD_STATE_API_MODES:
        raise ValueError(
            f"未知 world_state_api={world_state_api!r}（可选 {'/'.join(LOAD_WORLD_STATE_API_MODES)}）"
        )
    if world_state_api != "off":
        if path_head != "structured":
            raise ValueError(
                f"[eval_runner] --world-state-api {world_state_api} 需要 --path-head structured"
                "（net 契约：readout = 结构化头想象步几何输入）"
            )
        if world_state_api == "analytic_learned" and not world_residual_head:
            raise ValueError(
                "[eval_runner] --world-state-api analytic_learned 需要 --world-residual-head <ckpt>"
            )
    elif world_residual_head:
        raise ValueError(
            "[eval_runner] --world-residual-head 只对 --world-state-api analytic_learned 有意义"
        )
    # R41-A1：teacher-forced oracle / 速度反事实轴（默认 off/None ⇒ 零触达、零行为变化）
    teacher_force = str(teacher_force or "off")
    if teacher_force not in LOAD_TEACHER_FORCE_MODES:
        raise ValueError(
            f"[eval_runner] 未知 teacher_force={teacher_force!r}"
            f"（可选 {'/'.join(LOAD_TEACHER_FORCE_MODES)}）"
        )
    if teacher_force != "off" and world_state_api == "off":
        raise ValueError(
            "[eval_runner] --teacher-force 非 off 需要 --world-state-api 非 off"
            "（状态注入落点 = analytic_update 后 / readout 前；口径 = R40-V4B）"
        )
    speed_oracle = str(speed_oracle or "off")
    if speed_oracle not in LOAD_SPEED_ORACLE_MODES:
        raise ValueError(
            f"[eval_runner] 未知 speed_oracle={speed_oracle!r}（可选 {'/'.join(LOAD_SPEED_ORACLE_MODES)}）"
        )
    if speed_scale is not None:
        speed_scale = float(speed_scale)
        if not (np.isfinite(speed_scale) and speed_scale > 0.0):
            raise ValueError(f"[eval_runner] --speed-scale 必须 >0 有限，收到 {speed_scale}")
        if speed_scale == 1.0:
            speed_scale = None  # 显式 1.0 = 默认档（不触达、不入 hash/缓存键）
    # R39-B2d：残差缩放轴（默认 None ⇒ 零触达；仅 learned 档；∈[0,4] = net 契约）
    if world_residual_scale is not None:
        world_residual_scale = float(world_residual_scale)
        if not 0.0 <= world_residual_scale <= 4.0:
            raise ValueError(
                f"[eval_runner] --world-residual-scale 必须 ∈ [0,4]，收到 {world_residual_scale}"
            )
        if world_state_api != "analytic_learned":
            raise ValueError(
                "[eval_runner] --world-residual-scale 只对 --world-state-api analytic_learned 有意义"
            )
    # R41-D：几何实例配置轴（默认哨兵/None ⇒ 零触达、逐位不变；canonical/别名 ⇒ 显式装配）
    geometry_requested = _normalize_readout_lane_geometry(readout_lane_geometry)
    geometry_requested_source = str(
        readout_lane_geometry_source
        or ("cli" if readout_lane_geometry is not None else "model_config")
    )
    geometry_config_value = _readout_geometry_config_value(config)
    cache_key = _model_cache_key(
        ckpt, device, path_head=path_head, ld_mode=ld_mode,
        rollout_semantic=rollout_semantic, fill_from=fill_from,
        world_state_api=world_state_api, world_residual_head=world_residual_head,
        world_residual_scale=world_residual_scale,
        teacher_force=teacher_force, record_truth=record_truth,
        inject_truth_sha16=inject_truth_sha16, speed_scale=speed_scale,
        speed_oracle=speed_oracle, readout_lane_geometry=geometry_requested,
    )
    cached = _CKPT_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached
    import torch

    from tools.legacy import legacy_key_mapper as _key_mapper
    from tools.legacy.checkpoint_migration import migrate_state_dict

    model = _build_model_from_config(config)
    if ld_mode != "baseline":
        model.set_ld_mode(ld_mode)  # M2/R21 惰性构造（默认关 = 不新增参数/键）
    if rollout_semantic != "full":
        model.set_rollout_semantic(rollout_semantic)
    if path_head == "structured":
        model.set_path_head(True)  # 惰性构造（默认关 = 不新增参数/键）
        model.set_path_head_ablation("structured")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    # R41-D：「读取」链——ckpt metadata 几何口径（缺失 ⇒ 按默认并记录；非法 ⇒ 警告 + 按缺失）
    raw_meta = payload.get("meta") if isinstance(payload, Mapping) else None
    geometry_meta_raw = raw_meta.get("readout_lane_geometry") if isinstance(raw_meta, Mapping) else None
    geometry_meta_value: Optional[str] = None
    if geometry_meta_raw is not None:
        try:
            geometry_meta_value = _normalize_readout_lane_geometry(geometry_meta_raw)
        except ValueError:
            print(
                f"[eval_runner] 警告：ckpt {ckpt} 的 readout_lane_geometry={geometry_meta_raw!r} 非法"
                " ⇒ 按缺失处理（默认）",
                flush=True,
            )
            geometry_meta_value = None
    geometry_resolution = _resolve_readout_lane_geometry(
        requested=geometry_requested,
        requested_source=geometry_requested_source,
        ckpt_meta=geometry_meta_value,
        config_value=geometry_config_value,
        ckpt_meta_raw=(str(geometry_meta_raw) if geometry_meta_raw is not None else None),
    )
    if geometry_resolution["conflict"]:
        print(
            f"[eval_runner] 警告：几何口径声明不一致（requested={geometry_requested} vs "
            f"ckpt metadata={geometry_meta_value}）⇒ 按 requested 执行（交叉/显式臂）；"
            "已记录 run_meta/metrics.geometry",
            flush=True,
        )
    if geometry_resolution["value"] != "default":
        model.set_readout_lane_geometry(geometry_resolution["value"])
        print(
            f"[eval_runner] 几何实例配置 readout_lane_geometry={geometry_resolution['value']}"
            f"（source={geometry_resolution['source']}；ws readout + ws-off 想象步统一轴）",
            flush=True,
        )
    state = payload
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in payload and isinstance(payload[key], Mapping):
                state = payload[key]
                break
    if not isinstance(state, Mapping):
        raise ValueError(f"ckpt {ckpt!r} 不含可识别的 state_dict（顶层键 {list(payload)[:5]}）")
    # R38-P2a/P2b → R39-A1：已登记丢弃族/改名族在分类层应用（逐族计数；未登记键仍 reject）。
    migrated, migration = migrate_state_dict(state, model.state_dict().keys())
    filtered, missing, unexpected, shape_mismatch = _filter_checkpoint_state(model, migrated)
    # R39-A1b：缺键回填（--fill-from；口径 = a3 canary fill_from —— 只回填 missing 中存在的键）
    fill_report: Optional[Dict[str, Any]] = None
    if fill_from:
        fill_path = str(fill_from)
        if not Path(fill_path).is_file():
            raise FileNotFoundError(f"[eval_runner] --fill-from 缺文件 {fill_path}")
        fill_payload = torch.load(fill_path, map_location="cpu", weights_only=False)
        fill_state = fill_payload
        if isinstance(fill_payload, Mapping):
            for key in ("model", "state_dict", "model_state_dict", "net"):
                if key in fill_payload and isinstance(fill_payload[key], Mapping):
                    fill_state = fill_payload[key]
                    break
        if not isinstance(fill_state, Mapping):
            raise ValueError(
                f"[eval_runner] --fill-from {fill_path!r} 不含可识别的 state_dict"
                f"（顶层键 {list(fill_payload)[:5]}）"
            )
        missing_set = set(missing)
        model_state = model.state_dict()
        fill_loaded: Dict[str, Any] = {}
        fill_shape_mismatch: List[str] = []
        for key, value in fill_state.items():
            if key not in missing_set:
                continue
            target = model_state.get(key)
            if target is None:
                continue
            try:
                same_shape = tuple(target.shape) == tuple(value.shape)
            except AttributeError:
                same_shape = False
            if same_shape:
                fill_loaded[key] = value
            else:
                fill_shape_mismatch.append(str(key))
        filtered.update(fill_loaded)
        missing = [key for key in missing if key not in fill_loaded]
        fill_report = {
            "path": fill_path,
            "sha256": _sha256_file(fill_path),
            "n_filled": len(fill_loaded),
            "filled_keys": sorted(str(key) for key in fill_loaded),
            "candidate_keys": sorted(str(key) for key in missing_set),
            "shape_mismatch": fill_shape_mismatch,
        }
    counts = {
        "missing": len(missing),
        "unexpected": len(unexpected),
        "shape_mismatch": len(shape_mismatch),
    }
    blocked = any(counts.values())
    ckpt_sha256 = _sha256_file(ckpt)
    # R39-A1 LoadReport（口径：missing/unexpected/filled/rejected/renamed + ckpt hash + mapper 版本）。
    load_report: Dict[str, Any] = {
        "ckpt": {"path": str(ckpt), "sha256": ckpt_sha256, "ckpt_keys": len(state)},
        "model_keys": len(model.state_dict()),
        "counts": {
            "missing": len(missing),
            "unexpected": len(unexpected),
            "filled": len(filtered),
            "rejected": len(migration.reject_keys),
            "renamed": int(migration.n_rename),
            "dropped": int(migration.n_drop),
        },
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "filled_keys": sorted(str(key) for key in filtered),
        "rejected_keys": list(migration.reject_keys),
        "renamed_keys": dict(migration.rename_map),
        "renamed_by_family": dict(migration.renamed_by_family),
        "dropped_by_family": dict(migration.dropped_by_family),
        "reasons": dict(migration.reasons),
        "mapper": {
            "module": "tools.legacy.legacy_key_mapper",
            "version": str(getattr(_key_mapper, "MAPPER_VERSION", "unknown")),
        },
        "path_head": path_head,
        "ld_mode": ld_mode,
        "rollout_semantic": rollout_semantic,
        "fill_from": fill_report,
        "strict": bool(not blocked),
        "loader": None,  # 实际装载原语（干净 ⇒ load_checkpoint_migrated(strict=True)；逃生门 ⇒ 本地子集）
        "strict_semantics": (
            "已登记迁移（+显式回填）后 missing/unexpected/shape_mismatch 全零 ⇒ strict=True 语义装载；"
            "任一非零 ⇒ 默认 fail-fast（--allow-partial-load 显式放行时 strict=False）"
        ),
        "allow_partial_load": bool(allow_partial),
        "explicit_escape": bool(blocked and allow_partial),
    }
    # R41-D：几何口径解析记录（有轴/有 meta/有声明才写 ⇒ 纯 legacy 默认路径逐字节不变）
    if geometry_resolution["value"] != "default" or geometry_meta_raw is not None \
            or geometry_requested is not None:
        load_report["geometry"] = _geometry_artifact_record(geometry_resolution)
    summary: Dict[str, Any] = {
        "ckpt": str(ckpt),
        "sha256": ckpt_sha256,
        "device": str(device),
        "allow_partial_load": bool(allow_partial),
        "path_head": path_head,
        "ld_mode": ld_mode,
        "rollout_semantic": rollout_semantic,
        "fill_from": (fill_report or {}).get("path"),
        "loaded_keys": len(filtered),
        "model_keys": len(model.state_dict()),
        "ckpt_keys": len(state),
        "counts": counts,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "shape_mismatch": list(shape_mismatch),
        # R39-A1：迁移 + LoadReport（run_meta/manifest 审计口径）
        "load_report": load_report,
    }
    if "geometry" in load_report:
        summary["geometry"] = dict(load_report["geometry"])
    # lane U1：MoE 输出口径 = primary + Σ_{i∈top2} g_i·expert_i（推理与训练一致、全场景生效；
    # 无硬切/二值门），评测侧无需任何开关（模型默认 moe_enabled=True）。
    print(
        f"[eval_runner] ckpt 载入 {ckpt}: loaded={len(filtered)}/{len(model.state_dict())} "
        f"ckpt_keys={len(state)} missing={counts['missing']} shape_mismatch={counts['shape_mismatch']} "
        f"unexpected={counts['unexpected']} path_head={path_head} ld_mode={ld_mode} "
        f"rollout_semantic={rollout_semantic} fill_from={(fill_report or {}).get('n_filled', 0)} "
        f"renamed={migration.n_rename} dropped={migration.n_drop} device={device} "
        f"torch_threads={torch.get_num_threads()}",
        flush=True,
    )
    for label, keys in (("missing", missing), ("shape_mismatch", shape_mismatch), ("unexpected", unexpected)):
        if keys:
            print(f"[eval_runner]   {label}（{len(keys)}，示例 {list(keys)[:5]}）", flush=True)
    if fill_report:
        print(
            f"[eval_runner]   回填族（--fill-from；missing 中命中 {fill_report['n_filled']} 键）："
            f"{fill_report['path']}",
            flush=True,
        )
    if migration.dropped_by_family:
        print(f"[eval_runner]   已登记丢弃族（逐族计数）：{dict(migration.dropped_by_family)}", flush=True)
    if migration.renamed_by_family:
        print(f"[eval_runner]   已登记改名族（逐族计数）：{dict(migration.renamed_by_family)}", flush=True)
    if blocked:
        if allow_partial:
            print(
                f"[eval_runner] 警告：--allow-partial-load 放行部分加载 missing={counts['missing']} "
                f"unexpected={counts['unexpected']} shape_mismatch={counts['shape_mismatch']}"
                "（强制记录到日志与 metrics.json::meta.ckpt_load）",
                flush=True,
            )
            print(f"[eval_runner]   missing 全量（{len(missing)}）：{list(missing)}", flush=True)
        else:
            print(
                f"[eval_runner] ckpt 加载 FAIL-FAST：missing={counts['missing']} "
                f"unexpected={counts['unexpected']} shape_mismatch={counts['shape_mismatch']}"
                "（默认拒绝部分加载；如确需放行请显式传 --allow-partial-load）",
                flush=True,
            )
            raise SystemExit(2)
        # 逃生门（显式）：按调用点形状过滤后的子集 strict=False（形状门由本 loader 承担；
        # tools.legacy 原语不做形状过滤 —— 见其 docstring）
        model.load_state_dict(filtered, strict=False)
    elif fill_report is None:
        # R39-A1：**干净迁移（无回填）⇒ 统一迁移装载原语**（tools.legacy.checkpoint_migration.
        # load_checkpoint_migrated，strict=True）——正式装载路径的严格语义钉住口径。
        from tools.legacy.checkpoint_migration import load_checkpoint_migrated

        info = load_checkpoint_migrated(str(ckpt), model, strict=True, state=state)
        if info["missing_keys"] or info["unexpected_keys"]:  # pragma: no cover - 防御（counts 已零）
            raise SystemExit(
                "[eval_runner] ckpt 加载 FAIL-FAST：迁移原语报告非零差异"
                f" missing={info['missing_keys'][:5]} unexpected={info['unexpected_keys'][:5]}"
            )
        load_report["loader"] = "tools.legacy.checkpoint_migration.load_checkpoint_migrated(strict=True)"
    else:
        # 有回填（显式）：migrate + fill 合并子集，缺失已清零 ⇒ strict=True 语义装载
        # （统一原语不支持 fill；见 checkpoint_migration.load_checkpoint_migrated docstring）
        model.load_state_dict(filtered, strict=True)
        load_report["loader"] = (
            "pipeline.eval_runner（migrate + --fill-from 合并子集，strict=True；"
            "无回填时走 tools.legacy.checkpoint_migration.load_checkpoint_migrated）"
        )
    load_report["loader"] = load_report.get("loader", "pipeline.eval_runner（形状过滤子集，strict=False）")
    model.to(device).eval()
    # R39-B2c：WorldState API（默认 off ⇒ 零触达，不调用任何 net API）。时序刻意放在
    # state_dict 装载 + ``model.state_dict()`` 计数（model_keys）之后：残差头挂载不影响
    # 键面/LoadReport（头 = 独立 ckpt、装配面非参数开关）。
    if world_state_api != "off":
        model.set_world_state_api(world_state_api)
        ws_record: Dict[str, Any] = {"mode": world_state_api, "residual_head": None}
        if world_state_api == "analytic_learned":
            from net.world_residual import load_residual_head  # 仅 learned 档触达

            head, head_meta = load_residual_head(str(world_residual_head), device=str(device))
            model.set_world_residual_head(head)
            if world_residual_scale is not None:
                # R39-B2d：残差缩放（默认 None ⇒ 不调用；1.0 = net 默认逐位，不记新键）
                model.set_world_residual_scale(world_residual_scale)
                if float(world_residual_scale) != 1.0:
                    ws_record["residual_scale"] = float(world_residual_scale)
            ws_record["residual_head"] = {
                "path": str(world_residual_head),
                "sha256": _sha256_file(str(world_residual_head)),
                "head_params": int(sum(p.numel() for p in head.parameters())),
                "meta": dict(head_meta or {}),
            }
        load_report["world_state"] = ws_record
        summary["world_state"] = ws_record
        print(
            f"[eval_runner] WorldState API={world_state_api}"
            + (f"（residual head {ws_record['residual_head']['head_params']} params ← "
               f"{world_residual_head}）" if ws_record["residual_head"] else "")
            + (f"（residual scale ×{ws_record['residual_scale']}）"
               if ws_record.get("residual_scale") is not None else "")
            + "：k≥1 想象步几何 = 统一 readout",
            flush=True,
        )
    # R41-A1：teacher-forced oracle 档（模型内原生路径；默认 off ⇒ 不调用任何 net API）。
    # 时序与 ws API 一致：状态装载/键面计数之后设置（非参数行为开关，不影响 LoadReport 键面）。
    if teacher_force != "off":
        model.set_teacher_force(teacher_force)
    if teacher_force != "off" or record_truth or inject_truth_sha16 or speed_scale is not None \
            or speed_oracle != "off":
        summary["counterfactual"] = {
            "teacher_force": teacher_force,
            "record_truth": bool(record_truth),
            "inject_truth_sha16": str(inject_truth_sha16) if inject_truth_sha16 else None,
            "speed_scale": (None if speed_scale is None else float(speed_scale)),
            "speed_oracle": speed_oracle,
        }
    _CKPT_LOAD_SUMMARY[cache_key] = summary
    _CKPT_MODEL_CACHE[cache_key] = model
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


#: 评测跟踪器参考口径（``--eval-reference``）：
#: ``plan``（默认，历史口径）= 6 步 plan 预览（``plan[0] = action_mu``，rollout 前向）；
#: ``repeat_action`` = ``repeat(action_mu, 6)``（与训练 P0-1 A-hold 同构；走 cheap path
#: （``policy_single_step``），不需要 plan）。
EVAL_REFERENCES: tuple[str, ...] = ("plan", "repeat_action")


def task_model_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """评测侧模型配置：key 清单必须与 ``stages.build_model`` 实际读取的键对齐。

    2026-10-10 修复（fail-fast 事故记录）：此前漏传 ``spatial``，stg3 臂（``spatial.layers=3``）
    的 ckpt 在评测侧被按默认 2 层建模型 → ``unexpected=10`` 被 fail-fast 拦下（rc=2）；
    若在 fail-fast 引入前，会静默按 2 层评测出错误数字。build_model 读取键：
    hidden_dim / moe / policy / value / world_model / spatial。
    """
    return {
        "hidden_dim": config.get("hidden_dim", 128),
        # v8 修复：policy/value 隐藏维必须透传，否则评测模型按代码默认
        # （trunk 160 / net 256）构造 → 与 ckpt 形状不符 → 随机初始化评测（假阴性）。
        "moe": dict(config.get("moe") or {}),
        "policy": dict(config.get("policy") or {}),
        "value": dict(config.get("value") or {}),
        "world_model": dict(config.get("world_model") or {}),
        # 2026-10-10：spatial 层数（stg3 排摸臂）必须透传——见上方事故记录。
        "spatial": dict(config.get("spatial") or {}),
        # R41-D：几何口径实例配置（config/model.yaml 根键；缺席 ⇒ 不携带 ⇒ 默认哨兵）。
        **({"readout_lane_geometry": config.get("readout_lane_geometry")}
           if config.get("readout_lane_geometry") is not None else {}),
    }


def build_eval_references(
    mu_action: np.ndarray, plan: Optional[np.ndarray], reference: str
) -> np.ndarray:
    """构造跟踪器 6 步参考 ``(6,2)``（确定性评测：动作 = ``action_mu``）。

    - ``repeat_action``：``np.repeat(mu_action[None, :], 6, axis=0)``——与训练
      ``PPOTrainer.collect_rollout`` 的 A-hold 口径同构（训练用采样动作 ``a_t``，评测确定性
      用 ``mu``）；不需要 plan 输出（可走 ``policy_single_step`` cheap path，R39-A1 枚举口径）；
    - ``plan``（默认，现状）：plan 预览 ``(6,2)``，首步强制 = ``mu_action``（旧行为逐位不变）。
    """
    mu = np.asarray(mu_action, dtype=np.float64).reshape(2)
    if reference == "repeat_action":
        return np.repeat(mu[None, :], 6, axis=0)
    if reference != "plan":
        raise ValueError(f"未知 eval_reference={reference!r}（可选 {'/'.join(EVAL_REFERENCES)}）")
    if plan is None:
        raise ValueError("eval_reference='plan' 需要模型 plan 输出（forward_mode='policy_wm_rollout'）")
    tail = np.asarray(plan, dtype=np.float64).reshape(-1, 2).copy()
    if tail.shape[0] != 6:
        raise ValueError(f"plan 参考期望 (6,2)，收到 {tail.shape}")
    tail[0] = mu
    return tail


#: R41-J：``env.tracking`` 可覆盖的 ``LqrTracker`` 构造键（白名单；空配置 = 逐位不变）。
LQR_TRACKER_KEYS: Tuple[str, ...] = (
    "wheelbase",
    "max_steer_angle_rad",
    "lookahead_time",
    "lookahead_base",
    "lookahead_min",
    "lookahead_max",
    "q_lateral",
    "q_heading",
    "r_steer",
    "steer_gain",
    "design_speed_mps",
    "reschedule_dv",
    "riccati_iters",
    "riccati_tol",
    "speed_gain",
    "accel_action_scale",
    "brake_action_scale",
    "reference_dt",
    "reference_hz",
    "ref_speed_scale",
    "telemetry_coupling",
    "curvature_feedforward",
    "ff_gain",
)


def lqr_tracker_kwargs(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``env.tracking`` 配置 → ``LqrTracker`` 构造 kwargs（R41-J 预瞄口径轴）。

    默认路径（``config`` 为空/None）返回 ``{}`` ⇒ ``engine.add_policy`` 调用**逐位不变**。
    非空时按 :data:`LQR_TRACKER_KEYS` 白名单透传（键名 = ``LqrTracker`` 构造参数；
    ``env/tracking.py::LqrTracker.params`` 为单一出处）；未知键 **fail-fast**（禁静默忽略）。
    """
    if not config:
        return {}
    unknown = sorted(set(str(key) for key in config) - set(LQR_TRACKER_KEYS))
    if unknown:
        raise ValueError(
            f"env.tracking 含未知 LqrTracker 键 {unknown}（可选 {' | '.join(LQR_TRACKER_KEYS)}）"
        )
    return {str(key): value for key, value in config.items()}


# --------------------------------------------------------------------------- #
# R41-A1：两遍法 truth IO（记录 pass → npz；注入 pass 读取 + 载荷构造）
# --------------------------------------------------------------------------- #
def _truth_path(truth_dir: Any, spec_id: int) -> Path:
    """``<truth_dir>/<spec_id:04d>.npz``（两遍法文件名口径，与 V4B 相同）。"""
    return Path(str(truth_dir)) / f"{int(spec_id):04d}{TRUTH_NPZ_SUFFIX}"


def _truth_dir_digest(truth_dir: Any) -> Dict[str, Any]:
    """truth 目录内容指纹（逐文件 sha256 → 排序连接 → 聚合 sha256；protocol hash 轴用）。

    口径 = R40-V4B ``a3/v4b_teacher_force.py::truth_dir_digest``（逐位一致：目录名排除在
    聚合之外，只依赖文件内容与文件名）。
    """
    root = Path(str(truth_dir))
    entries: List[Dict[str, str]] = []
    aggregate = hashlib.sha256()
    for path in sorted(root.glob(f"*{TRUTH_NPZ_SUFFIX}")):
        digest = _sha256_file(str(path))
        entries.append({"name": path.name, "sha256": digest})
        aggregate.update(f"{path.name}:{digest}\n".encode("utf-8"))
    return {
        "dir": str(root),
        "n_files": len(entries),
        "sha256": aggregate.hexdigest(),
        "sha16": aggregate.hexdigest()[:16],
        "files": entries,
    }


def _env_rng_of(env: Any) -> Any:
    """取 env.engine 的全局 RNG（MetaDrive ``Randomizable``；缺失 ⇒ None）。"""
    engine = getattr(env, "engine", None)
    return getattr(engine, "np_random", None)


def _rng_digest(rng: Any) -> Optional[int]:
    """``np.random.RandomState.get_state()`` 内容摘要（uint64；失败 ⇒ None）。

    口径 = ``(str, uint32[624], pos, has_gauss, cached_gaussian)`` 的 str + 状态数组 + pos
    的 sha256 前 8 字节（不依赖浮点 cached 值本身的表示）。
    """
    if rng is None:
        return None
    try:
        state = rng.get_state()
        array = np.ascontiguousarray(np.asarray(state[1]))
        digest = hashlib.sha256()
        digest.update(str(state[0]).encode("utf-8"))
        digest.update(array.tobytes())
        digest.update(f"pos={int(state[2])}".encode("utf-8"))
        return int.from_bytes(digest.digest()[:8], "big")
    except Exception:  # noqa: BLE001 - 取证字段不阻断评测
        return None


def _tensor_bundle_digest(bundle: Any) -> Optional[str]:
    """张量/数组字典内容摘要（sha16；键排序 + dtype + 原始字节；空/None ⇒ None）。

    Oracle Injection Hook 的"替换前后 state hash"与"替换前后 Policy 输出"统一口径
    （float 字节按存储 dtype 原样），保证同输入同 hash。
    """
    if not isinstance(bundle, Mapping) or not bundle:
        return None
    try:
        import torch

        digest = hashlib.sha256()
        for key in sorted(str(k) for k in bundle):
            value = bundle[key]
            if value is None:
                continue
            if torch.is_tensor(value):
                array = value.detach().to("cpu").numpy()
            else:
                array = np.asarray(value)
            digest.update(key.encode("utf-8"))
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()[:16]
    except Exception:  # noqa: BLE001 - 取证字段不阻断评测
        return None


def _termination_code(name: str) -> int:
    return int(TERMINATION_CODES.get(str(name), TERMINATION_CODES["other"]))


class _TruthRecorder:
    """记录 pass：逐决策帧真值 + 逐 env step 轨迹 + episode 元数据（只观察、零执行副作用）。

    帧记录（决策帧，0.5 s 间隔）：ego pose/speed、OD/LD 特征与掩码、专家空问（κ/ds/valid）、
    实际下发动作（``action_exec`` = references[0]，``plan_exec`` = 6 步参考 —— 动作条件证据）、
    env RNG 摘要（traffic RNG 一致性证据）。
    逐 step 记录：step index / pose / speed / RNG 摘要（逐步状态误差证据）。
    """

    def __init__(self) -> None:
        self.frames: List[Dict[str, Any]] = []
        self.steps: List[Dict[str, Any]] = []
        self.n_label_invalid = 0

    # ---------------------------------------------------------------- 采集
    def capture_step(self, *, env: Any, step_index: int, rng_digest: Optional[int]) -> None:
        """逐 env step 轨迹（在 ``action()`` 每步调用；只读 pose/speed）。

        ``step_rng_digest`` 以 **uint64** 存储（float64 无法精确表示 64 位摘要 ⇒ 低比特
        失配假象；0 = 不可用）。
        """
        ego = env.agent
        self.steps.append({
            "step_index": float(int(step_index)),
            "step_pose": np.asarray(
                [float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)],
                dtype=np.float64,
            ),
            "step_speed": float(ego.speed),
            "step_rng_digest": np.uint64(rng_digest if rng_digest is not None else 0),
        })

    def capture_frame(
        self,
        *,
        env: Any,
        obs: Mapping[str, Any],
        label: Optional[Mapping[str, Any]],
        action_exec: np.ndarray,
        plan_exec: np.ndarray,
        rng_digest: Optional[int],
    ) -> None:
        """逐决策帧真值（专家空问 ``label`` 由 ``tools.dagger_collect._make_labeler`` 产出）。"""
        ego = env.agent
        kappa = ds_e = np.nan
        valid = 0.0
        if label is not None:
            action = np.asarray(label.get("action"), dtype=np.float64).reshape(-1)[:2]
            ds_e = float(action[0])
            dtheta_e = float(action[1])
            if np.isfinite(ds_e) and np.isfinite(dtheta_e) and ds_e > 1e-3:
                kappa = dtheta_e / ds_e
                valid = 1.0
        if valid <= 0.5:
            self.n_label_invalid += 1
        self.frames.append({
            "frame_env_step": float(int(getattr(env, "episode_step", 0))),
            "ego_pose_world": np.asarray(
                [float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)],
                dtype=np.float64,
            ),
            "ego_speed": float(np.asarray(obs["ego"], dtype=np.float64).reshape(-1)[0]),
            "od": np.asarray(obs["od"], dtype=np.float32).reshape(-1, 9),
            "od_mask": np.asarray(obs["od_mask"], dtype=np.float32).reshape(-1),
            "od_presence": np.asarray(obs["od_presence"], dtype=np.float32).reshape(-1),
            "ld": np.asarray(obs["ld"], dtype=np.float32).reshape(-1, 7),
            "ld_mask": np.asarray(obs["ld_mask"], dtype=np.float32).reshape(-1),
            "kappa_expert": float(kappa),
            "ds_expert": float(ds_e),
            "expert_valid": float(valid),
            "action_exec": np.asarray(action_exec, dtype=np.float64).reshape(2),
            "plan_exec": np.asarray(plan_exec, dtype=np.float64).reshape(-1, 2),
            # uint64（精确；float64 会丢低比特 ⇒ 假失配；0 = 不可用）
            "rng_digest": np.uint64(rng_digest if rng_digest is not None else 0),
        })

    # ---------------------------------------------------------------- 落盘
    def as_arrays(self, *, termination: str, meta: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        arrays: Dict[str, np.ndarray] = {}
        if self.frames:
            for key in TRUTH_FRAME_KEYS:
                arrays[key] = np.stack(
                    [np.asarray(frame[key]) for frame in self.frames], axis=0
                )
        else:
            arrays.update({key: np.zeros((0,), dtype=np.float64) for key in TRUTH_FRAME_KEYS})
        if self.steps:
            for key in TRUTH_STEP_KEYS:
                arrays[key] = np.stack([np.asarray(step[key]) for step in self.steps], axis=0)
        else:
            arrays.update({key: np.zeros((0,), dtype=np.float64) for key in TRUTH_STEP_KEYS})
        arrays["termination_code"] = np.asarray([_termination_code(termination)], dtype=np.int64)
        arrays["n_steps"] = np.asarray([len(self.steps)], dtype=np.int64)
        arrays["meta_json"] = np.asarray(json.dumps(_sanitize(dict(meta)), ensure_ascii=False))
        return arrays

    def save(
        self, truth_dir: Any, spec_id: int, *, termination: str, meta: Mapping[str, Any]
    ) -> Path:
        path = _truth_path(truth_dir, spec_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **self.as_arrays(termination=termination, meta=meta))
        return path

    def summary(self) -> Dict[str, Any]:
        return {
            "n_capture": int(len(self.frames)),
            "n_steps": int(len(self.steps)),
            "n_label_invalid": int(self.n_label_invalid),
        }


class _TruthRecord:
    """注入 pass：读取记录 pass 的 npz（只读）；兼容 R40-V4B legacy 键面（缺扩展键 = 不可用）。

    扩展键（``action_exec/plan_exec/rng_digest/step_*``）缺失时：动作条件按"实测（位姿派生）"
    口径降级（legacy 文件也有 ``ego_pose_world`` 可与重放位姿对照），RNG/逐步误差报告显式
    标注 n/a。
    """

    def __init__(self, path: Any):
        self.path = Path(str(path))
        with np.load(str(self.path)) as data:
            self.data = {key: np.asarray(data[key]) for key in data.files}
        for key in ("ego_pose_world", "ego_speed", "od", "od_mask", "od_presence", "ld", "ld_mask"):
            if key not in self.data:
                raise ValueError(f"[eval_runner] truth 文件缺必需键 {key!r}：{self.path}")
        self.n_frames = int(self.data["ego_pose_world"].shape[0])
        self.n_steps = int(np.asarray(self.data.get("n_steps", [0])).reshape(-1)[0])
        self.termination_code = int(
            np.asarray(self.data.get("termination_code", [_termination_code("other")])).reshape(-1)[0]
        )
        self.meta: Dict[str, Any] = {}
        raw_meta = self.data.get("meta_json")
        if raw_meta is not None:
            try:
                payload = json.loads(str(np.asarray(raw_meta).reshape(-1)[0]))
                self.meta = dict(payload) if isinstance(payload, Mapping) else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                self.meta = {}

    # ---------------------------------------------------------------- 只读访问
    @property
    def poses(self) -> np.ndarray:
        return np.asarray(self.data["ego_pose_world"], dtype=np.float64).reshape(self.n_frames, 3)

    @property
    def speeds(self) -> np.ndarray:
        return np.asarray(self.data["ego_speed"], dtype=np.float64).reshape(self.n_frames)

    def has(self, key: str) -> bool:
        return key in self.data

    def array(self, key: str) -> Optional[np.ndarray]:
        value = self.data.get(key)
        return None if value is None else np.asarray(value)

    def realized_action(self, frame: int) -> Optional[np.ndarray]:
        """决策帧 ``frame → frame+1`` 的**实测动作** ``(ds, dθ)``（位姿派生；越界 ⇒ None）。"""
        if int(frame) < 0 or int(frame) + 1 >= self.n_frames:
            return None
        return _pose_realized_action(self.poses[int(frame)], self.poses[int(frame) + 1])

    def commanded_action(self, frame: int, step: int = 1) -> Optional[np.ndarray]:
        """决策帧 ``frame`` 的第 ``step`` 个**命令动作**（``plan_exec``；legacy 缺键 ⇒ None）。"""
        plan = self.array("plan_exec")
        if plan is None or plan.ndim != 3 or int(frame) < 0 or int(frame) >= int(plan.shape[0]):
            return None
        k = int(step) - 1
        if k < 0 or k >= int(plan.shape[1]):
            return None
        return np.asarray(plan[int(frame), k], dtype=np.float64).reshape(2)

    def rng_digest(self, frame: int) -> Optional[int]:
        """决策帧 RNG 摘要（精确 int；0/缺失 ⇒ None = 不可用；legacy 文件无该键）。"""
        arr = self.array("rng_digest")
        if arr is None or int(frame) < 0 or int(frame) >= arr.shape[0]:
            return None
        value = int(np.asarray(arr[int(frame)]).reshape(-1)[0])
        return None if value == 0 else value

    # ---------------------------------------------------------------- 载荷构造
    @staticmethod
    def _relative(pose_f: np.ndarray, pose_g: np.ndarray) -> np.ndarray:
        """帧 f 系的相对位姿（t0 帧）：``[R(−θ_f)(p_g−p_f), wrap(θ_g−θ_f)]``（V4B §2.3 同式）。"""
        theta_f = float(pose_f[2])
        dx = float(pose_g[0]) - float(pose_f[0])
        dy = float(pose_g[1]) - float(pose_f[1])
        cos_t = math.cos(theta_f)
        sin_t = math.sin(theta_f)
        return np.asarray(
            [cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy,
             _wrap_to_pi(float(pose_g[2]) - theta_f)],
            dtype=np.float64,
        )

    def build_payload(self, frame: int, device: str = "cpu", max_step: int = 6) -> Dict[int, Dict[str, Any]]:
        """重放帧 ``frame`` 的注入载荷 ``{k: entry}``（k=1..；``g = frame + k`` 越界 ⇒ 缺键）。

        逐位对齐 R40-V4B ``TruthRecord.build_payload``（§2.3 重表达公式）；entry 张量 +
        ``meta``（真实来源：truth 文件/记录帧/env step/来源 protocol hash）。
        """
        import torch

        payload: Dict[int, Dict[str, Any]] = {}
        if int(frame) < 0 or int(frame) >= self.n_frames:
            return payload
        poses = self.poses
        speeds = self.speeds
        od_all = np.asarray(self.data["od"], dtype=np.float64).reshape(self.n_frames, -1, 9)
        od_mask = np.asarray(self.data["od_mask"], dtype=np.float64).reshape(self.n_frames, -1)
        od_presence = np.asarray(self.data["od_presence"], dtype=np.float64).reshape(self.n_frames, -1)
        ld_all = np.asarray(self.data["ld"], dtype=np.float64).reshape(self.n_frames, -1, 7)
        ld_mask = np.asarray(self.data["ld_mask"], dtype=np.float64).reshape(self.n_frames, -1)
        kappa_expert = np.asarray(self.data["kappa_expert"], dtype=np.float64).reshape(self.n_frames)
        expert_valid = np.asarray(self.data["expert_valid"], dtype=np.float64).reshape(self.n_frames)
        frame_env_step = self.array("frame_env_step")
        pose_f = poses[int(frame)]
        for k in range(1, int(max_step) + 1):
            g = int(frame) + k
            if g >= self.n_frames:
                break  # 越界：该步无真值（显式跳过，计数见 runtime）
            pose_rel = self._relative(pose_f, poses[g])
            theta_rel = float(pose_rel[2])
            cos_t = math.cos(theta_rel)
            sin_t = math.sin(theta_rel)
            speed_g = float(speeds[g])

            od = od_all[g]                       # (S,9) 帧 g 的 ego 系特征
            od_live = (od_mask[g] > 0.5) & (od_presence[g] > 0.5)
            dx, dy = od[:, 0], od[:, 1]
            vx_f, vy_f = od[:, 2], od[:, 3]
            px = cos_t * dx - sin_t * dy + pose_rel[0]
            py = sin_t * dx + cos_t * dy + pose_rel[1]
            vx_world = vx_f + speed_g               # 沿 ego_g 纵轴（无侧向速度假设，同 V4A）
            vwx = cos_t * vx_world - sin_t * vy_f
            vwy = sin_t * vx_world + cos_t * vy_f
            cos_f, sin_f = od[:, 4], od[:, 5]
            cos_t0 = cos_f * cos_t - sin_f * sin_t
            sin_t0 = sin_f * cos_t + cos_f * sin_t
            heading_t0 = np.arctan2(sin_t0, cos_t0)
            od_valid = od_live.astype(np.float64)

            ld = ld_all[g]                          # (L,7) 帧 g 的 ego 系特征
            ld_live = ld_mask[g] > 0.5
            ldx, ldy = ld[:, 0], ld[:, 1]
            lpx = cos_t * ldx - sin_t * ldy + pose_rel[0]
            lpy = sin_t * ldx + cos_t * ldy + pose_rel[1]
            ld_h = ld[:, 2] + theta_rel
            ld_curv = ld[:, 3]
            ld_valid = ld_live.astype(np.float64)

            kappa_valid = bool(expert_valid[g] > 0.5 and np.isfinite(kappa_expert[g]))
            meta = {
                "truth_file": str(self.path),
                "recorded_frame": int(g),
                "replay_frame": int(frame),
                "recorded_env_step": (
                    int(frame_env_step[g]) if frame_env_step is not None and g < frame_env_step.shape[0]
                    else None
                ),
                "source_protocol_hash": self.meta.get("protocol_hash"),
                "source_teacher_force": self.meta.get("teacher_force"),
            }
            payload[k] = {
                "pose": torch.as_tensor(pose_rel, dtype=torch.float32, device=device),
                "speed": torch.as_tensor([speed_g], dtype=torch.float32, device=device),
                "od_pos": torch.as_tensor(np.stack([px, py], axis=-1), dtype=torch.float32, device=device),
                "od_vw": torch.as_tensor(np.stack([vwx, vwy], axis=-1), dtype=torch.float32, device=device),
                "od_cs": torch.as_tensor(np.stack([cos_t0, sin_t0], axis=-1), dtype=torch.float32, device=device),
                "od_h": torch.as_tensor(heading_t0, dtype=torch.float32, device=device),
                "od_static": torch.as_tensor(od[:, 6:9], dtype=torch.float32, device=device),
                "od_valid": torch.as_tensor(od_valid, dtype=torch.float32, device=device),
                "ld_pos": torch.as_tensor(np.stack([lpx, lpy], axis=-1), dtype=torch.float32, device=device),
                "ld_h": torch.as_tensor(ld_h, dtype=torch.float32, device=device),
                "ld_curv": torch.as_tensor(ld_curv, dtype=torch.float32, device=device),
                "ld_valid": torch.as_tensor(ld_valid, dtype=torch.float32, device=device),
                "kappa": torch.as_tensor(
                    [float(kappa_expert[g]) if kappa_valid else 0.0],
                    dtype=torch.float32, device=device,
                ),
                "kappa_valid": torch.as_tensor([1.0 if kappa_valid else 0.0],
                                               dtype=torch.float32, device=device),
                "meta": meta,
            }
        return payload

    def summary(self) -> Dict[str, Any]:
        expert_valid = self.array("expert_valid")
        return {
            "path": str(self.path),
            "n_frames": int(self.n_frames),
            "n_steps": int(self.n_steps),
            "termination_code": int(self.termination_code),
            "extended_schema": bool(self.has("action_exec") and self.has("plan_exec")),
            "has_rng": bool(self.has("rng_digest")),
            "has_step_trace": bool(self.has("step_pose")),
            "expert_valid_frac": (
                float(np.mean(np.asarray(expert_valid, dtype=np.float64) > 0.5))
                if expert_valid is not None and expert_valid.size else None
            ),
            "source_meta": dict(self.meta),
        }


def _pose_realized_action(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """相邻决策帧位姿 → 实测动作 ``(ds, dθ)``（位移模 + wrap 航向差）。"""
    a = np.asarray(pose_a, dtype=np.float64).reshape(3)
    b = np.asarray(pose_b, dtype=np.float64).reshape(3)
    ds = float(math.hypot(b[0] - a[0], b[1] - a[1]))
    dtheta = float(_wrap_to_pi(b[2] - a[2]))
    return np.asarray([ds, dtheta], dtype=np.float64)


def _od_divergence(
    replay_od: np.ndarray,
    replay_mask: np.ndarray,
    rec_od: np.ndarray,
    rec_mask: np.ndarray,
    rec_presence: np.ndarray,
) -> Dict[str, Any]:
    """② OD state divergence：掩码内（记录 live ∩ 重放 mask）逐槽特征绝对差。"""
    replay = np.asarray(replay_od, dtype=np.float64).reshape(-1, 9)
    rec = np.asarray(rec_od, dtype=np.float64).reshape(-1, 9)
    if replay.shape != rec.shape:
        return {"available": False, "reason": f"slot shape {replay.shape} vs {rec.shape}"}
    live = (
        (np.asarray(rec_mask, dtype=np.float64).reshape(-1) > 0.5)
        & (np.asarray(rec_presence, dtype=np.float64).reshape(-1) > 0.5)
        & (np.asarray(replay_mask, dtype=np.float64).reshape(-1) > 0.5)
    )
    if not bool(live.any()):
        return {"available": False, "reason": "no live slot", "n_live": 0}
    diff = np.abs(replay[live] - rec[live])
    return {
        "available": True,
        "n_live": int(live.sum()),
        "mean_abs": float(np.mean(diff)),
        "max_abs": float(np.max(diff)),
    }


def _p95(values: Sequence[float]) -> Optional[float]:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(np.percentile(array, 95.0)) if array.size else None


def _mean_abs(values: Sequence[float]) -> Optional[float]:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(np.mean(np.abs(array))) if array.size else None


def _max_abs(values: Sequence[float]) -> Optional[float]:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    return float(np.max(np.abs(array))) if array.size else None


def _plan_scale_factor(speed_scale: float, speed_oracle: str, plan0_ds: float,
                       expert_ds: Optional[float]) -> float:
    """速度反事实缩放因子（``--speed-scale`` × ``--speed-oracle``；纯函数，口径 = V4B）。

    ``expert``：``scale *= clip(ds_expert / plan0_ds, 0.05, 2.0)``（专家空问纵向速度替换）；
    ``ds`` 极端比值计数并钳位，不静默逸出。
    """
    scale = float(speed_scale)
    if str(speed_oracle or "off") == "expert":
        if expert_ds is None or not np.isfinite(float(expert_ds)):
            return scale
        denom = max(float(plan0_ds), 1e-3)
        scale *= float(np.clip(float(expert_ds) / denom, 0.05, 2.0))
    return scale


def _apply_plan_speed_scale(plan: np.ndarray, scale: float) -> np.ndarray:
    """``plan (N,2) ← scale·plan``（ds/dθ 同缩 ⇒ κ = dθ/ds 不变；纯函数）。"""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"[eval_runner] plan 缩放因子必须 >0 有限，收到 {scale}")
    return np.asarray(plan, dtype=np.float64) * float(scale)


def probe_env_snapshot_support(env: Any) -> Dict[str, Any]:
    """R41-A1 §三.①：核实 MetaDrive engine 是否支持 snapshot/clone（一步合法 transition 路线）。

    候选 API（engine 及当前 manager 对象上逐名探测可调用性）：``dump_state/load_state/
    snapshot/clone/save_state/restore_state``。任一可用 ⇒ 路线 ①（clone_env 分支）；
    全不可用 ⇒ 路线 ②（两遍重放 + 五类漂移报告 + 按阈剔除）。

    事实（2026-10 环境，MetaDrive 2.3.0a0）：engine 无全状态 dump/load —— 仅有
    ``dump_episode``（record_manager 元数据）与逐对象 ``get_state/set_state``（不含 RNG/
    manager/physics 全局态）⇒ 本仓取路线 ②（见 a1 报告）。
    """
    candidates = ("dump_state", "load_state", "snapshot", "clone", "save_state", "restore_state")
    engine = getattr(env, "engine", env)
    present: List[str] = []
    checked: List[Dict[str, Any]] = []
    for scope_name, scope in (("engine", engine),):
        for name in candidates:
            callable_ = callable(getattr(scope, name, None))
            checked.append({"scope": scope_name, "name": name, "callable": bool(callable_)})
            if callable_:
                present.append(f"{scope_name}.{name}")
    managers = getattr(engine, "managers", None)
    if isinstance(managers, Mapping):
        for manager_name, manager in managers.items():
            for name in candidates:
                if callable(getattr(manager, name, None)):
                    present.append(f"manager[{manager_name}].{name}")
    supported = bool(present)
    return {
        "candidates": list(candidates),
        "present": sorted(set(present)),
        "supported": supported,
        "route": ("clone_env_branch" if supported else "two_pass_replay"),
        "checked": checked,
        "evidence": (
            "MetaDrive engine 无 dump_state/load_state/snapshot/clone（全状态分支不可用）⇒ "
            "采用两遍重放 + 五类漂移报告 + 按阈剔除（报告显式声明阈值）"
            if not supported else "engine 级 snapshot/clone 可用"
        ),
        "engine_class": type(engine).__name__,
    }


class _OracleInjectionHook:
    """R41-A1 §七.1：Oracle Injection Hook —— 注入事件五要件记录 + 非法组合自动拒绝 + 审计落盘。

    每注入事件记录：注入模块（surface 级）/注入步数/真实来源（truth 文件 + 记录帧 + 来源
    protocol hash）/动作条件一致性（由 :meth:`note_action_condition` 后置判定）/替换前后
    state hash/替换前后 Policy 输出（含 hash）。

    自动拒绝语义（两级）：

    - **预检硬拒**（不进入模型路径）：帧级"同状态门"失败/载荷结构性缺失 ⇒ runtime 直接不注入
      （``n_rejected_frame_gate``/``n_missing`` 计账）；
    - **后置裁定拒**（不得进入裁决）：注入后按动作条件判定 ``illegal``（学生动作条件下产生的
      future state 与记录不一致）⇒ 该注入步不计入 ``legal_steps``，``adjudication_excluded``；
      episode 级 ``illegal_frac`` 超阈 ⇒ 整条 episode 剔除出 oracle 裁决视图。
    """

    def __init__(self, *, spec_id: int, truth_summary: Mapping[str, Any], detail_cap: int = 500) -> None:
        self.spec_id = int(spec_id)
        self.truth_summary = dict(truth_summary)
        self.detail_cap = int(detail_cap)
        self.events: List[Dict[str, Any]] = []
        self.pending: Dict[int, List[Dict[str, Any]]] = {}
        self.verdicts: Dict[int, Dict[str, Any]] = {}
        self.n_events = 0
        self.n_detail_dropped = 0
        self.counts: Dict[str, int] = {
            "n_events": 0,
            "n_state_events": 0,
            "n_lateral_events": 0,
            "n_legal": 0,
            "n_illegal": 0,
            "n_unproven": 0,
        }
        self.by_surface: Dict[str, int] = {}

    # ---------------------------------------------------------------- 模型侧回调
    def on_injection(self, event: Mapping[str, Any]) -> None:
        """模型注入事件（替换前后 state 张量 / policy 输出 → hash + 明细；受 detail_cap 限制）。"""
        record = {
            "spec_id": int(self.spec_id),
            "frame": int(event.get("frame", -1)),
            "step": int(event.get("step", -1)),
            "module": str(event.get("module", "?")),
            "surface": str(event.get("surface", "?")),
            "status": "injected",
            "source": _sanitize(dict(event.get("meta") or {})),
            "probe_mode": str(event.get("probe_mode", "")),
            "state_hash_before": _tensor_bundle_digest(event.get("before_state")),
            "state_hash_after": _tensor_bundle_digest(event.get("after_state")),
            "policy_hash_before": _tensor_bundle_digest(event.get("before_policy")),
            "policy_hash_after": _tensor_bundle_digest(event.get("after_policy")),
            "policy_output_before": _sanitize(event.get("before_policy")),
            "policy_output_after": _sanitize(event.get("after_policy")),
        }
        self.n_events += 1
        self.counts["n_events"] += 1
        surface = record["surface"]
        self.by_surface[surface] = int(self.by_surface.get(surface, 0)) + 1
        if surface == "lateral":
            self.counts["n_lateral_events"] += 1
        else:
            self.counts["n_state_events"] += 1
        self.pending.setdefault(record["frame"], []).append(record)
        if len(self.events) < self.detail_cap:
            self.events.append(record)
        else:
            self.n_detail_dropped += 1

    # ---------------------------------------------------------------- 裁定
    def _classify(self, record: Dict[str, Any], verdict: Optional[Mapping[str, Any]]) -> None:
        """按裁定（``legal``/``unproven``/其余=不一致）落事件状态并计账。

        - ``legal`` ⇒ ``n_legal``（**进入** oracle 裁决）；
        - 不一致（``auto_rejected``）⇒ ``n_illegal``（**裁定剔除**：非法组合不得进入裁决）；
        - 无证据（空裁定/``unproven``）⇒ ``n_unproven``（同样剔除，不默认合法）。
        """
        payload = dict(verdict) if isinstance(verdict, Mapping) else {}
        record["action_condition"] = _sanitize(payload)
        if not payload or bool(payload.get("unproven")):
            record["status"] = "unproven"
            self.counts["n_unproven"] += 1
        elif bool(payload.get("legal")):
            record["status"] = "legal"
            self.counts["n_legal"] += 1
        else:
            record["status"] = "auto_rejected"
            self.counts["n_illegal"] += 1

    def note_action_condition(self, frame: int, verdict: Mapping[str, Any]) -> None:
        """帧级动作条件裁定（逐 **步** 归因：事件按 ``(frame, step)`` 取 ``verdict["steps"][k]``）。"""
        frame = int(frame)
        verdict = dict(verdict)
        self.verdicts[frame] = verdict
        steps = dict(verdict.get("steps") or {})
        for record in self.pending.pop(frame, []):
            self._classify(record, steps.get(int(record.get("step", -1))))

    def attribute(self, verdicts: Mapping[Tuple[int, int], Mapping[str, Any]]) -> Dict[str, Any]:
        """收尾裁定：按 ``(frame, step)`` 键逐事件落状态（``verdicts`` 缺键 ⇒ unproven）；返回摘要。"""
        for frame in sorted(list(self.pending)):
            for record in self.pending.pop(frame):
                key = (int(record["frame"]), int(record["step"]))
                self._classify(record, verdicts.get(key))
        return self.summary()

    def finalize(self, extra_verdict: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """收尾兜底：未裁定事件按 ``extra_verdict``（缺省 unproven）落定；返回审计摘要。"""
        fallback = dict(extra_verdict or {})
        if not fallback:
            fallback = {"legal": False, "unproven": True, "reason": "unattributed"}
        for frame in sorted(self.pending):
            for record in self.pending.pop(frame):
                self._classify(record, fallback)
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        return {
            "spec_id": int(self.spec_id),
            "truth": dict(self.truth_summary),
            "counts": dict(self.counts),
            "by_surface": dict(self.by_surface),
            "n_frames_with_events": int(len(self.verdicts)),
            "n_detail_recorded": int(len(self.events)),
            "n_detail_dropped": int(self.n_detail_dropped),
        }

    def write_detail(self, path: Any) -> Path:
        """逐事件明细 JSONL（含 hash/动作条件/policy 输出；受 detail_cap 截断，计数如实）。"""
        path = Path(str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.events:
                handle.write(json.dumps(_sanitize(record), ensure_ascii=False, sort_keys=True) + "\n")
        return path


# --------------------------------------------------------------------------- #
# R39-A1b：lane_anchor_xy 注入器（选项 a 的显式开关；口径单一出处 = R36 C lane）
# --------------------------------------------------------------------------- #
#: R36 C lane 注入器（``select_target_lane`` + E3 车道链采样；与 a3 ``--lane-anchor inject`` 同源）。
LANE_ANCHOR_SCRIPT = ".slim/deepwork/p1_v2/round36/c_lane_anchor.py"
_LANE_ANCHOR_MODULE: Any = None


def _lane_anchor_module() -> Any:
    """只读装载 ``round36/c_lane_anchor.py``（注入口径单一出处；仅 ``--lane-anchor inject`` 触达）。"""
    global _LANE_ANCHOR_MODULE
    if _LANE_ANCHOR_MODULE is None:
        _LANE_ANCHOR_MODULE = _load_readonly_script("r39a1b_lane_anchor", LANE_ANCHOR_SCRIPT)
    return _LANE_ANCHOR_MODULE


class _CkptController:
    """ckpt 策略：每 0.5 s 用 ``DrivingModel`` 出 ``(ds, dθ)``，交给 N3 跟踪器逐步执行。

    观测每 env step 都过 ``ObservationBuilder``（历史窗口依赖逐 step push），动作只
    在 5 步（0.5 s）边界重算；跟踪器参考由 ``env.tracking.interpolate`` 从 6 步动作序列
    插值成 30 点（契约 §1）。

    ``task["eval_reference"]``（``--eval-reference``，默认 ``plan``）：

    - ``plan``（历史口径，默认）：6 步 plan 预览（``plan[0] = action_mu``），模型前向走
      ``policy_wm_rollout``（R39-A1 枚举口径）；
    - ``repeat_action``：参考 = ``repeat(action_mu, 6)``（训练 A-hold 同构）→ ``policy_single_step``
      cheap path（R39-A1 枚举口径，不消费 plan）。

    ``task["model_assembly"]``（R39-A1b；正式入口 ``main`` 装配）：``{path_head, ld_mode,
    rollout_semantic, fill_from, lane_anchor, world_state_api, world_residual_head,
    world_residual_scale, readout_lane_geometry}``（R41-D：几何实例配置；缺省/哨兵 ⇒
    沿用 config/ckpt metadata/历史默认）—— 透传 :func:`_load_ckpt_model` 与注入器；
    **缺省（无该键，如 a3 controller 自己的 canary 预填路径）= 全默认装配**，缓存键仍为
    ``(ckpt, device)``（旧行为逐位不变）。

    ``task["counterfactual"]``（R41-A1；正式入口 ``main`` 装配）：``{teacher_force,
    record_truth, inject_truth, speed_scale, speed_oracle, drift_thresholds, protocol_hash,
    out_dir}`` —— teacher-forced oracle 注入（模型内原生路径）+ 两遍法 truth 记录/注入 +
    速度反事实 + Oracle Injection Hook；**缺省 = 全关**（不装 hook、不读 npz、不新增 episode
    键，逐位不变）。
    """

    kind = "ckpt"
    speed_limit_units = "mps"
    fallback_speed_limit_mps = DEFAULT_SPEED_LIMIT_FALLBACK
    decision_interval = 5  # 0.5 s / 0.1 s
    #: R39-A1b：注入器/模式的类级默认（``object.__new__`` 构造的测试实例亦有默认值）
    anchor_injector: Any = None
    lane_anchor_mode: str = "off"
    #: R41-A1：反事实轴类级默认（``object.__new__`` 测试实例亦可用；全默认 = 全关）
    teacher_force_mode: str = "off"
    record_truth_dir: Optional[str] = None
    inject_truth_dir: Optional[str] = None
    speed_scale: Optional[float] = None
    speed_oracle: str = "off"
    cf_active: bool = False
    #: R41-A1：两遍法运行时类级默认（None = 未挂载；仅 cf_active/对应开关非默认时被触达）
    labeler: Any = None
    record: Any = None
    recorder: Any = None
    hook: Any = None

    def __init__(self, env: Any, spec: Any, task: Mapping[str, Any]):
        import torch

        self.spec = spec
        self.device = torch.device(str(task.get("device") or "cpu"))
        assembly = dict(task.get("model_assembly") or {})
        cf = dict(task.get("counterfactual") or {})
        self.teacher_force_mode = str(cf.get("teacher_force") or "off")
        if self.teacher_force_mode not in LOAD_TEACHER_FORCE_MODES:
            raise ValueError(
                f"未知 teacher_force={self.teacher_force_mode!r}"
                f"（可选 {'/'.join(LOAD_TEACHER_FORCE_MODES)}）"
            )
        self.record_truth_dir = str(cf.get("record_truth") or "").strip() or None
        self.inject_truth_dir = str(cf.get("inject_truth") or "").strip() or None
        self.speed_scale = None if cf.get("speed_scale") is None else float(cf["speed_scale"])
        if self.speed_scale is not None and self.speed_scale == 1.0:
            self.speed_scale = None
        if self.speed_scale is not None and not (np.isfinite(self.speed_scale) and self.speed_scale > 0.0):
            raise ValueError(f"speed_scale 必须 >0 有限，收到 {self.speed_scale}")
        self.speed_oracle = str(cf.get("speed_oracle") or "off")
        if self.speed_oracle not in LOAD_SPEED_ORACLE_MODES:
            raise ValueError(f"未知 speed_oracle={self.speed_oracle!r}")
        if self.record_truth_dir and self.inject_truth_dir:
            raise ValueError("[eval_runner] record_truth 与 inject_truth 互斥（两遍法 = 两次独立 run）")
        if self.inject_truth_dir and self.teacher_force_mode == "off":
            raise ValueError("[eval_runner] --inject-truth 只对 --teacher-force 非 off 有意义")
        if self.teacher_force_mode != "off" and str(assembly.get("world_state_api") or "off") == "off":
            raise ValueError(
                "[eval_runner] --teacher-force 非 off 需要 --world-state-api 非 off"
                "（状态注入落点 = analytic_update 后 / readout 前）"
            )
        self.cf_thresholds: Dict[str, float] = dict(DEFAULT_DRIFT_THRESHOLDS)
        self.cf_thresholds.update(
            {str(k): float(v) for k, v in dict(cf.get("drift_thresholds") or {}).items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
        )
        self.cf_frame_gate: Dict[str, float] = dict(DEFAULT_FRAME_GATE)
        self.cf_frame_gate.update(
            {str(k): float(v) for k, v in dict(cf.get("frame_gate") or {}).items()
             if isinstance(v, (int, float)) and not isinstance(v, bool)}
        )
        self.cf_protocol_hash = str(cf.get("protocol_hash") or "")
        self.cf_out_dir = str(cf.get("out_dir") or "").strip() or None
        self.cf_active = bool(
            self.teacher_force_mode != "off" or self.record_truth_dir or self.inject_truth_dir
            or self.speed_scale is not None or self.speed_oracle != "off"
        )
        self.model = _load_ckpt_model(
            str(task["ckpt"]),
            task.get("model_config") or {},
            str(self.device),
            allow_partial=bool(task.get("allow_partial_load")),
            path_head=str(assembly.get("path_head") or "off"),
            ld_mode=str(assembly.get("ld_mode") or "baseline"),
            rollout_semantic=str(assembly.get("rollout_semantic") or "full"),
            fill_from=(str(assembly["fill_from"]) if assembly.get("fill_from") else None),
            world_state_api=str(assembly.get("world_state_api") or "off"),
            world_residual_head=(str(assembly["world_residual_head"])
                                 if assembly.get("world_residual_head") else None),
            world_residual_scale=(float(assembly["world_residual_scale"])
                                  if assembly.get("world_residual_scale") is not None else None),
            teacher_force=self.teacher_force_mode,
            record_truth=bool(self.record_truth_dir),
            inject_truth_sha16=(str(cf.get("inject_truth_sha16")) if cf.get("inject_truth_sha16") else None),
            speed_scale=self.speed_scale,
            speed_oracle=self.speed_oracle,
            readout_lane_geometry=(str(assembly["readout_lane_geometry"])
                                   if assembly.get("readout_lane_geometry") else None),
            readout_lane_geometry_source=(str(assembly["readout_lane_geometry_source"])
                                          if assembly.get("readout_lane_geometry_source") else None),
        )
        # R39-A1b：lane_anchor_xy 注入器（选项 a 的显式开关；默认 off ⇒ None，不触达任何新代码路径）
        self.lane_anchor_mode = str(assembly.get("lane_anchor") or "off")
        if self.lane_anchor_mode not in LOAD_LANE_ANCHOR_MODES:
            raise ValueError(
                f"未知 lane_anchor={self.lane_anchor_mode!r}（可选 {'/'.join(LOAD_LANE_ANCHOR_MODES)}）"
            )
        self.anchor_injector = (
            _lane_anchor_module().LaneAnchorInjector()
            if self.lane_anchor_mode == "inject" else None
        )
        if bool(task.get("moe_off")):
            # A/B：primary-only 臂 —— 专家分支不参与（评测侧关闭 MoE）
            self.model.set_moe(enabled=False)
            print("[eval_runner] MoE 已关闭（primary-only 臂）", flush=True)
        self.obs_config = dict(task.get("obs_config") or {})
        self.tracker_config = dict(task.get("tracker_config") or {})
        #: "exact"（阶段 A/B 精确/运动学执行，默认）| "lqr"（阶段 C 闭环）
        self.tracker_kind = str(task.get("tracker") or "exact").lower()
        #: 跟踪器参考口径（``--eval-reference``；默认 plan = 历史口径，行为不变）
        self.eval_reference = str(task.get("eval_reference") or "plan")
        if self.eval_reference not in EVAL_REFERENCES:
            raise ValueError(
                f"未知 eval_reference={self.eval_reference!r}（可选 {'/'.join(EVAL_REFERENCES)}）"
            )
        self.builder = None
        self.tracker = None
        self._steps = 0
        self._action = [0.0, 0.0]
        #: 最近一次策略决策实际下发的 ``(ds[m], dθ[rad])``（``references[0] == action_mu``）
        self._plan_action = np.zeros(2, dtype=np.float64)
        #: 逐 env step 的自车实测位姿（用于重建"上一策略步实测动作"，见 _measured_prev_action）
        self._pose_history: List[Tuple[float, float, float]] = []
        # ---- R41-A1：两遍法运行时（默认全 None/全空 ⇒ 零触达）----
        self.labeler = None                    # 专家空问 labeler（record pass / speed_oracle）
        self.record: Optional[_TruthRecord] = None   # 注入 pass：记录 pass 的 npz
        self.recorder: Optional[_TruthRecorder] = None  # 记录 pass：本 episode 真值
        self.hook: Optional[_OracleInjectionHook] = None
        self._oracle_route: Optional[Dict[str, Any]] = None
        self._cf_counts: Dict[str, int] = {}
        self._drift_frames: List[Dict[str, Any]] = []
        self._pending_verdicts: Dict[int, Dict[str, Any]] = {}
        self._final_verdicts: Dict[int, Dict[str, Any]] = {}
        self._replay_step_pose: List[Tuple[float, float, float]] = []
        self._replay_step_speed: List[float] = []
        self._replay_step_index: List[int] = []
        self._replay_step_rng: List[Optional[int]] = []

    def _cf_reset_episode(self) -> None:
        """R41-A1：episode 级运行时重置（bind/finish 对称；默认关路径零触达）。"""
        self._cf_counts = {
            "n_frames": 0, "n_frames_rejected_gate": 0, "n_speed_scaled": 0,
            "n_payload_steps": 0, "n_payload_steps_structural_skip": 0,
        }
        self._drift_frames = []
        self._executed_first: Dict[int, np.ndarray] = {}  # frame -> 本帧下发首动作（动作条件链）
        self._realized_intervals: Dict[int, Dict[str, Any]] = {}  # 区间起点 -> 实测证据（lag-1）
        self._final_verdicts: Dict[Tuple[int, int], Dict[str, Any]] = {}  # (frame, step) -> 裁定
        self._cf_frame_poses: List[Tuple[float, float, float]] = []
        self._replay_step_pose = []
        self._replay_step_speed = []
        self._replay_step_index = []
        self._replay_step_rng = []

    def bind(self, env: Any) -> None:
        from env.obs.builder import ObservationBuilder

        self.builder = ObservationBuilder(self.obs_config)
        self.builder.reset()
        self._steps = 0
        self._action = [0.0, 0.0]
        self._plan_action = np.zeros(2, dtype=np.float64)
        self._pose_history = []
        self.recorder = None
        self.record = None
        self.hook = None
        self.labeler = None
        self._cf_reset_episode()
        # §8.4：episode 起点没有上一动作（与 collect_expert/training pools 同口径）
        env.prev_policy_action = np.zeros(2, dtype=np.float64)
        if self.tracker_kind == "exact":
            from env.tracking import ExactTracker  # 阶段 A/B 运动学执行器（step 后 apply）

            self.tracker = ExactTracker(dt=0.5, hz=10)
        else:
            from env.tracking import LqrTracker  # N3 契约

            # R41-J：预瞄口径轴（``env.tracking``；默认空 = 逐位不变的不变调用）
            tracker_kwargs = lqr_tracker_kwargs(self.tracker_config)
            if tracker_kwargs:
                print(f"[eval_runner] LqrTracker 覆盖参数（R41-J env.tracking）：{tracker_kwargs}", flush=True)
            # 与训练侧 ``LocalEnvPool(_setup_episode)`` 同款：engine 注册 + 每策略步 set_reference
            self.tracker = env.engine.add_policy(
                env.agent.id,
                LqrTracker,
                env.agent,
                int(getattr(self.spec, "seed", 0)),
                **tracker_kwargs,
            )
            if hasattr(self.tracker, "reset"):
                self.tracker.reset()
        # ---- R41-A1：两遍法运行时（默认关 ⇒ 以下全不触达；hook 逐 episode 挂/卸）----
        if self.recorder is None and self.record_truth_dir is not None:
            self.recorder = _TruthRecorder()
        if self.inject_truth_dir is not None:
            truth_path = _truth_path(self.inject_truth_dir, int(getattr(self.spec, "id", -1)))
            if not truth_path.is_file():
                raise SystemExit(
                    f"[eval_runner] fail-fast：缺 truth 文件 {truth_path}"
                    "（--inject-truth 需第一遍记录 pass 产物）"
                )
            self.record = _TruthRecord(truth_path)
            self.hook = _OracleInjectionHook(
                spec_id=int(getattr(self.spec, "id", -1)), truth_summary=self.record.summary()
            )
            self.model.set_teacher_force_hooks(self.hook)
            self._oracle_route = probe_env_snapshot_support(env)
        if self.record_truth_dir is not None or self.speed_oracle == "expert":
            # 专家空问（无执行副作用；构造在 env.reset 之后 —— 规则专家依赖全局 engine）
            from tools.dagger_collect import _make_labeler

            self.labeler = _make_labeler(env, self.spec, "pure_pursuit")

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

    # ------------------------------------------------------------------ R41-A1
    def _cf_frame_gate(self, env: Any, obs: Mapping[str, Any], frame: int) -> Dict[str, Any]:
        """决策帧"同状态门"（预检硬门；超标 ⇒ 该帧注入自动拒绝，不进入模型路径）。

        条件（阈值见 ``DEFAULT_FRAME_GATE`` / config 覆盖）：位置误差 ≤ ``pose_pos_m``、
        航向误差 ≤ ``heading_rad``、OD 掩码内均绝对差 ≤ ``od_mean_abs``、RNG 摘要一致
        （``rng_required``=1；无数据 ⇒ n/a 放行并计账）。
        """
        record = self.record
        gate: Dict[str, Any] = {"frame": int(frame), "available": True, "ok": True, "checks": {}}
        if record is None:
            return {"frame": int(frame), "available": False, "ok": False, "reason": "no_record"}
        if int(frame) >= record.n_frames:
            return {"frame": int(frame), "available": False, "ok": False,
                    "reason": "beyond_record", "n_record_frames": int(record.n_frames)}
        ego = env.agent
        replay_pose = np.asarray(
            [float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)],
            dtype=np.float64,
        )
        rec_pose = record.poses[int(frame)]
        pose_err = float(math.hypot(replay_pose[0] - rec_pose[0], replay_pose[1] - rec_pose[1]))
        head_err = float(abs(_wrap_to_pi(replay_pose[2] - rec_pose[2])))
        od = _od_divergence(
            obs["od"], obs["od_mask"],
            record.array("od")[int(frame)], record.array("od_mask")[int(frame)],
            record.array("od_presence")[int(frame)],
        )
        replay_rng = _rng_digest(_env_rng_of(env))
        rec_rng = record.rng_digest(int(frame))
        rng_match = None if (replay_rng is None or rec_rng is None) else bool(replay_rng == rec_rng)
        checks = {
            "pose_pos_m": {"value": pose_err, "limit": float(self.cf_frame_gate["pose_pos_m"])},
            "heading_rad": {"value": head_err, "limit": float(self.cf_frame_gate["heading_rad"])},
        }
        if bool(od.get("available")):
            checks["od_mean_abs"] = {
                "value": float(od["mean_abs"]), "limit": float(self.cf_frame_gate["od_mean_abs"]),
            }
        ok = pose_err <= float(self.cf_frame_gate["pose_pos_m"]) \
            and head_err <= float(self.cf_frame_gate["heading_rad"]) \
            and (not bool(od.get("available")) or float(od["mean_abs"]) <= float(self.cf_frame_gate["od_mean_abs"]))
        if float(self.cf_frame_gate.get("rng_required", 1.0)) >= 0.5 and rng_match is False:
            ok = False
        gate.update({
            "ok": bool(ok), "checks": checks, "pose_pos_m": pose_err, "heading_rad": head_err,
            "od": od, "rng_match": rng_match, "replay_rng": replay_rng, "record_rng": rec_rng,
        })
        return gate

    def _cf_frame_drift(self, env: Any, obs: Mapping[str, Any], frame: int) -> Dict[str, Any]:
        """五类漂移中的逐帧证据（① pose / ② OD / ③ RNG）。"""
        record = self.record
        if record is None or int(frame) >= record.n_frames:
            return {"frame": int(frame), "available": False}
        ego = env.agent
        replay_pose = np.asarray(
            [float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)],
            dtype=np.float64,
        )
        rec_pose = record.poses[int(frame)]
        od = _od_divergence(
            obs["od"], obs["od_mask"],
            record.array("od")[int(frame)], record.array("od_mask")[int(frame)],
            record.array("od_presence")[int(frame)],
        )
        replay_rng = _rng_digest(_env_rng_of(env))
        rec_rng = record.rng_digest(int(frame))
        return {
            "frame": int(frame),
            "available": True,
            "pose_pos_m": float(math.hypot(replay_pose[0] - rec_pose[0], replay_pose[1] - rec_pose[1])),
            "heading_rad": float(abs(_wrap_to_pi(replay_pose[2] - rec_pose[2]))),
            "speed_abs": float(abs(float(ego.speed) - float(record.speeds[int(frame)]))),
            "od_mean_abs": (float(od["mean_abs"]) if od.get("available") else None),
            "od_max_abs": (float(od["max_abs"]) if od.get("available") else None),
            "rng_match": (None if (replay_rng is None or rec_rng is None) else bool(replay_rng == rec_rng)),
            "recorded_env_step": (
                int(record.array("frame_env_step")[int(frame)])
                if record.has("frame_env_step") else None
            ),
            "replay_env_step": int(self._steps),
        }

    def _cf_verdict(self, frame: int, step: int) -> Dict[str, Any]:
        """(frame, step) 动作条件裁定（legality：可用判据全一致 ⇒ legal；无证据 ⇒ unproven）。

        判据 = **执行命令链**（非循环口径）：注入状态 S_{f+k}（记录）由记录 run 的**实际下发动作**
        ``plan_exec[g][0]``（g ∈ [f, f+k-1]）产生；重放侧同一区间的实际下发动作 =
        ``_executed_first[g]``（首动作，含速度缩放；想象步注入不改它）。逐步比较，容差
        ``action_tol_ds_m`` / ``action_tol_dtheta_rad``：

        - k=1（一步合法 transition）：比较 f 帧；记录缺 ``plan_exec``（legacy）/重放帧缺失 ⇒
          unproven（不默认合法）；
        - k≥2（链步）：比较 f..f+k-1 逐帧；任一帧缺失 ⇒ unproven。

        实测（位姿派生）区间一致性不进裁定（属**漂移证据**，见 ``drift.realized_*``）——
        注入会经 LQR 前视改变同区间实测（反事实效应），把它当动作条件判据 = 自证循环。
        """
        if self.record is None:
            return {"legal": False, "unproven": True, "criteria": {}}
        th = self.cf_thresholds
        per_frame: List[Dict[str, Any]] = []
        for g in range(int(frame), int(frame) + int(step)):
            rec_action = self.record.commanded_action(int(g), 1)
            replay_action = self._executed_first.get(int(g))
            if rec_action is None or replay_action is None:
                return {"legal": False, "unproven": True, "criteria": {}}
            diff = np.abs(np.asarray(replay_action, dtype=np.float64).reshape(2) - rec_action)
            per_frame.append({
                "frame": int(g),
                "recorded": [float(rec_action[0]), float(rec_action[1])],
                "replay": [float(replay_action[0]), float(replay_action[1])],
                "abs_diff_ds_m": float(diff[0]), "abs_diff_dtheta_rad": float(diff[1]),
                "consistent": bool(
                    float(diff[0]) <= float(th["action_tol_ds_m"])
                    and float(diff[1]) <= float(th["action_tol_dtheta_rad"])
                ),
            })
        chain = {
            "per_frame": per_frame,
            "consistent": bool(all(bool(row["consistent"]) for row in per_frame)),
            "max_abs_diff_ds_m": max(float(row["abs_diff_ds_m"]) for row in per_frame),
            "max_abs_diff_dtheta_rad": max(float(row["abs_diff_dtheta_rad"]) for row in per_frame),
        }
        return {
            "legal": bool(chain["consistent"]),
            "unproven": False,
            "criteria": _sanitize({"executed_chain": chain}),
        }

    def _cf_finalize_injection_verdicts(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """收尾：为全部 (frame, step) 生成裁定并交 hook 归因（非法组合 ⇒ 裁决剔除）。"""
        frames = sorted(set(self._executed_first) | set(self._realized_intervals))
        verdicts: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for frame in frames:
            for step in range(1, int(TEACHER_FORCE_HORIZON) + 1):
                verdicts[(int(frame), int(step))] = self._cf_verdict(int(frame), int(step))
        return verdicts

    def _cf_realized_interval(self, frame: int) -> Dict[str, Any]:
        """区间 [frame-1, frame] 实测动作一致性（lag-1；**漂移证据**，不进裁定）。

        口径：位姿派生 ``(ds, dθ)`` 的 记录 vs 重放 差；注入经 LQR 前视会改变同区间实测
        （反事实效应），故它只作五类漂移的补充读数（``drift.realized_*``），不作动作条件判据。
        """
        record = self.record
        if record is None or int(frame) <= 0:
            return {"available": False, "reason": "no_previous_interval"}
        prev = int(frame) - 1
        if prev + 1 >= len(self._cf_frame_poses):  # 重放决策帧位姿（本帧已在 action() 内捕获）
            return {"available": False, "reason": "replay_pose_missing"}
        rec_action = record.realized_action(prev)
        if rec_action is None:
            return {"available": False, "reason": "record_pose_missing"}
        replay_action = _pose_realized_action(self._frame_pose(prev), self._frame_pose(int(frame)))
        diff = np.abs(replay_action - rec_action)
        th = self.cf_thresholds
        consistent = bool(
            float(diff[0]) <= float(th["realized_tol_ds_m"])
            and float(diff[1]) <= float(th["realized_tol_dtheta_rad"])
        )
        return {
            "available": True, "interval": [prev, int(frame)],
            "recorded": [float(rec_action[0]), float(rec_action[1])],
            "replay": [float(replay_action[0]), float(replay_action[1])],
            "abs_diff_ds_m": float(diff[0]), "abs_diff_dtheta_rad": float(diff[1]),
            "consistent": consistent,
        }

    def _frame_pose(self, frame: int) -> np.ndarray:
        """重放决策帧位姿（``_cf_frame_poses``；缺失 ⇒ 零位姿（调用方以 available 拦下））。"""
        poses = getattr(self, "_cf_frame_poses", None)
        if poses is not None and 0 <= int(frame) < len(poses):
            return np.asarray(poses[int(frame)], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def action(self, env: Any) -> List[float]:
        import torch

        measured = self._measured_prev_action(env)
        if measured is not None:
            env.prev_policy_action = measured  # 供本次决策的 obs（ego reserved 6:8）
        obs = self.builder.build(env, self.spec)
        if self.anchor_injector is not None and self._steps % self.decision_interval == 0:
            # R39-A1b：lane_anchor_xy 注入（默认 off ⇒ 不触达；只在决策步写 obs，与 a3 同点
            # 同口径 —— builder.build 之后、_tensors 之前；口径见 round36/c_lane_anchor.py）
            self.anchor_injector.apply(env, obs)
        if self.cf_active:
            # R41-A1：逐 env step 轨迹（两遍法漂移证据；默认关 ⇒ 零触达）
            self._cf_capture_step(env)
        if self._steps % self.decision_interval == 0:
            frame = self._steps // self.decision_interval
            # 参考口径（--eval-reference，默认 plan=现状）：
            #   plan：需要 6 步规划预览（跟踪器参考：首步执行动作 + 后 5 步 WM/plan 预览）；
            #   repeat_action：参考 = repeat(mu, 6)（训练 A-hold 同构）→ cheap path
            #   （forward_mode="policy_single_step"，不消费 plan）。
            label = self.labeler.label(env.agent) if self.labeler is not None else None
            # R41-A1：注入载荷（预检硬门 ⇒ 超标帧整帧自动拒绝：不进入模型路径）
            tf_payload: Dict[int, Dict[str, Any]] = {}
            tf_gate: Optional[Dict[str, Any]] = None
            if self.teacher_force_mode != "off":
                if self.record is not None:
                    tf_gate = self._cf_frame_gate(env, obs, frame)
                    self._cf_counts["n_frames"] += 1
                    if tf_gate.get("ok"):
                        tf_payload = self.record.build_payload(frame, device=str(self.device))
                        self._cf_counts["n_payload_steps"] += int(len(tf_payload))
                    else:
                        self._cf_counts["n_frames_rejected_gate"] += 1
                self.model.begin_teacher_force_frame(tf_payload or None, frame=frame)
            need_plan = self.eval_reference == "plan"
            with torch.no_grad():
                output = self.model(
                    self._tensors(obs),
                    forward_mode=("policy_wm_rollout" if need_plan else "policy_single_step"),
                )
            tf_stats = None
            if self.teacher_force_mode != "off":
                tf_stats = self.model.end_teacher_force_frame()
            mu = np.asarray(output["action_mu"].detach().cpu(), dtype=np.float64).reshape(-1)
            plan = None
            if need_plan:
                plan = np.asarray(output["plan"].detach().cpu(), dtype=np.float64).reshape(-1, 2)
            references = build_eval_references(mu[:2], plan, self.eval_reference)
            # R41-A1：速度反事实（plan 后处理同缩：ds/dθ 同缩 ⇒ κ 不变；默认关 ⇒ 零触达）
            if self.speed_scale is not None or self.speed_oracle != "off":
                expert_ds = None
                if self.speed_oracle == "expert" and label is not None:
                    expert_ds = float(np.asarray(label["action"], dtype=np.float64).reshape(-1)[0])
                ps_scale = _plan_scale_factor(
                    (1.0 if self.speed_scale is None else float(self.speed_scale)),
                    self.speed_oracle, float(references[0, 0]), expert_ds,
                )
                if ps_scale != 1.0:
                    references = _apply_plan_speed_scale(references, ps_scale)
                    self._cf_counts["n_speed_scaled"] += 1
            # 实际下发动作（供 action_ds/action_dtheta 指标；见 action_info）
            self._plan_action = np.array(references[0], dtype=np.float64, copy=True)
            if self.tracker_kind == "exact":
                # 阶段 A/B：把 6 步参考插值成 30 点参考，逐步 apply（env.step 之后）
                self.tracker.arm(env, actions=references)
            else:
                # 阶段 C：set_reference((N,2) 动作序列)，跟踪器内部插值成 30 点
                self.tracker.set_reference(references)
            # R41-A1：两遍法记录 pass（只观察；默认关 ⇒ 零触达）
            if self.recorder is not None:
                self.recorder.capture_frame(
                    env=env, obs=obs, label=label, action_exec=references[0],
                    plan_exec=references, rng_digest=_rng_digest(_env_rng_of(env)),
                )
            # R41-A1：注入 pass 漂移 + 动作条件证据（非法组合 ⇒ 收尾裁定剔除）
            if self.record is not None:
                ego = env.agent
                self._cf_frame_poses.append(
                    (float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta))
                )
                self._drift_frames.append(self._cf_frame_drift(env, obs, frame))
                self._executed_first[int(frame)] = np.asarray(references[0], dtype=np.float64).reshape(2)
                realized_prev = self._cf_realized_interval(frame)  # 区间 [frame-1, frame]（lag-1 证据）
                if frame > 0 and bool(realized_prev.get("available")):
                    self._realized_intervals[int(frame) - 1] = realized_prev
        self._steps += 1
        # engine 已注册策略，传回的占位动作会被忽略（engine/base_engine.py:98-104）
        return self._action

    def _cf_capture_step(self, env: Any) -> None:
        """逐 env step 轨迹采集（记录 pass 落 npz；注入 pass 留内存供逐步漂移对照）。"""
        rng = _rng_digest(_env_rng_of(env))
        if self.recorder is not None:
            self.recorder.capture_step(env=env, step_index=self._steps, rng_digest=rng)
        if self.record is not None:
            ego = env.agent
            self._replay_step_pose.append(
                (float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta))
            )
            self._replay_step_speed.append(float(ego.speed))
            self._replay_step_index.append(int(self._steps))
            self._replay_step_rng.append(rng)

    def post_step(self, env: Any) -> None:
        """env.step 之后调用（exact 执行器专用）：把 ego 置于插值参考位姿。"""
        if self.tracker_kind == "exact" and self.tracker is not None:
            self.tracker.apply(env)

    def action_info(self, env: Any) -> Dict[str, float]:
        """本 env step 实际下发的动作（2026-09-26 修正：不再恒为 0）。

        - ``action_ds_m`` / ``action_dtheta_rad``：每 0.5 s 决策下发的 ``(ds[m], dθ[rad])``
          首动作（``plan[0] == action_mu``），exact/lqr 两个 tracker 同口径；
        - ``steer`` / ``throttle``：仅 ``lqr`` tracker 有该概念（LqrTracker 每 env step 实测
          ``[steer, throttle]``，归一化 [-1,1]，与 baseline 同单位）；``exact`` 是运动学执行，
          无转向/油门动作 → NaN（N/A，不静默填 0）。
        """
        info: Dict[str, float] = {
            "action_ds_m": float(self._plan_action[0]),
            "action_dtheta_rad": float(self._plan_action[1]),
            "lead_gap_m": -1.0,
        }
        if self.tracker_kind == "lqr":
            tracker_info = getattr(self.tracker, "action_info", None) or {}
            action = tracker_info.get("action") or [float("nan"), float("nan")]
            info["steer"] = float(action[0])
            info["throttle"] = float(action[1])
        else:
            info["steer"] = float("nan")
            info["throttle"] = float("nan")
        return info

    # ------------------------------------------------------------------ R41-A1
    def finish_episode(self, env: Any, *, steps: int, termination: str,
                       success: bool, collision: bool, off_road: bool) -> Dict[str, Any]:
        """episode 收尾（R41-A1；默认全关 ⇒ 返回 {}，episode 键集逐位不变）。

        - 记录 pass：落 truth npz（含 episode 元数据/protocol hash）；
        - 注入 pass：五类漂移聚合 + 按阈剔除（``oracle_eligible``）+ hook 收尾落盘 + 卸 hook。
        """
        if not self.cf_active:
            return {}
        if self.recorder is not None:
            meta = {
                "producer": "pipeline.eval_runner._TruthRecorder",
                "schema": "r41a1-truth/v1",
                "protocol_hash": self.cf_protocol_hash,
                "spec_id": int(getattr(self.spec, "id", -1)),
                "termination": str(termination),
                "n_frames": int(len(self.recorder.frames)),
                "n_steps": int(len(self.recorder.steps)),
                "teacher_force": str(self.teacher_force_mode),
                "speed_scale": (None if self.speed_scale is None else float(self.speed_scale)),
                "speed_oracle": str(self.speed_oracle),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            path = self.recorder.save(
                self.record_truth_dir, int(getattr(self.spec, "id", -1)),
                termination=str(termination), meta=meta,
            )
            summary = self.recorder.summary()
            self.recorder = None
            return {"oracle": {
                "mode": "record",
                "record": {**summary, "path": str(path)},
                "hook": None,
            }}
        if self.record is None:
            # 仅速度反事实（无 truth 读写）⇒ 轻量计数（无裁决面）
            return {"oracle": {
                "mode": "speed_only",
                "injection": {"n_speed_scaled": int(self._cf_counts.get("n_speed_scaled", 0))},
                "hook": None,
            }}
        return self._cf_finish_inject(env, steps=steps, termination=termination)

    def _cf_finish_inject(self, env: Any, *, steps: int, termination: str) -> Dict[str, Any]:
        """注入 pass 收尾：五类漂移聚合 + 阈剔除 + hook 摘要/明细落盘 + hook 卸载。"""
        record = self.record
        if record is None:
            return {}
        # 动作条件裁定（全 lag：k=1 = 首动作 ∧ 实测区间；k≥2 = 实测链）→ hook 归因
        self._final_verdicts = self._cf_finalize_injection_verdicts()
        hook_summary = None
        if self.hook is not None:
            hook_summary = self.hook.attribute(self._final_verdicts)
            if self.cf_out_dir:
                detail_path = Path(self.cf_out_dir) / "oracle_hook" / \
                    f"spec_{int(getattr(self.spec, 'id', -1)):04d}.jsonl"
                self.hook.write_detail(detail_path)
            self.model.set_teacher_force_hooks(None)
        # ---- ①/②/③ 逐帧漂移聚合 ----
        frames = [f for f in self._drift_frames if f.get("available")]
        pose_errs = [f["pose_pos_m"] for f in frames]
        head_errs = [f["heading_rad"] for f in frames]
        od_means = [f["od_mean_abs"] for f in frames if f.get("od_mean_abs") is not None]
        od_maxs = [f["od_max_abs"] for f in frames if f.get("od_max_abs") is not None]
        rng_flags = [f["rng_match"] for f in frames if f.get("rng_match") is not None]
        realized_available = [
            iv for iv in self._realized_intervals.values() if bool(iv.get("available"))
        ]
        drift: Dict[str, Any] = {
            "n_frames_compared": int(len(frames)),
            "pose_pos_p95_m": _p95(pose_errs), "pose_pos_max_m": _max_abs(pose_errs),
            "heading_p95_rad": _p95(head_errs),
            "speed_abs_mean": _mean_abs([f.get("speed_abs") for f in frames]),
            "od_mean_abs": (float(np.mean(od_means)) if od_means else None),
            "od_max_abs": (float(np.max(od_maxs)) if od_maxs else None),
            "rng_match_frac": (float(np.mean([bool(x) for x in rng_flags])) if rng_flags else None),
            "rng_available": bool(rng_flags),
            # 实测（位姿派生）区间动作差（漂移证据；不进动作条件裁定 —— 见 _cf_verdict）
            "n_realized_intervals": int(len(realized_available)),
            "realized_first_ds_p95_m": _p95([iv["abs_diff_ds_m"] for iv in realized_available]),
            "realized_first_dtheta_p95_rad": _p95(
                [iv["abs_diff_dtheta_rad"] for iv in realized_available]
            ),
        }
        # 逐帧漂移样本（前 5 + 后 2；裁决审计证据：帧 0 = 注入前反馈窗口，应 ~0）
        frame_ids = [int(f["frame"]) for f in frames]
        sample_ids = set(frame_ids[:5] + frame_ids[-2:])
        drift["samples"] = [
            {
                "frame": int(f["frame"]),
                "pose_pos_m": float(f["pose_pos_m"]),
                "heading_rad": float(f["heading_rad"]),
                "od_mean_abs": f.get("od_mean_abs"),
                "od_max_abs": f.get("od_max_abs"),
                "rng_match": f.get("rng_match"),
                "replay_env_step": f.get("replay_env_step"),
                "recorded_env_step": f.get("recorded_env_step"),
            }
            for f in frames if int(f["frame"]) in sample_ids
        ]
        # ---- ⑤ 逐步状态误差（replay 逐步 vs 记录逐步） ----
        step_errs: List[float] = []
        speed_errs: List[float] = []
        step_rng_flags: List[bool] = []
        rec_step_pose = record.array("step_pose")
        rec_step_speed = record.array("step_speed")
        rec_step_rng = record.array("step_rng_digest")
        if rec_step_pose is not None:
            n_common = min(int(rec_step_pose.shape[0]), len(self._replay_step_pose))
            for index in range(n_common):
                replay_pose = np.asarray(self._replay_step_pose[index], dtype=np.float64).reshape(3)
                record_pose = np.asarray(rec_step_pose[index], dtype=np.float64).reshape(3)
                step_errs.append(float(math.hypot(
                    replay_pose[0] - record_pose[0], replay_pose[1] - record_pose[1]
                )))
                if rec_step_speed is not None and index < int(rec_step_speed.shape[0]) \
                        and index < len(self._replay_step_speed):
                    speed_errs.append(float(abs(
                        self._replay_step_speed[index] - float(np.asarray(rec_step_speed[index]).reshape(-1)[0])
                    )))
                if rec_step_rng is not None and index < int(rec_step_rng.shape[0]) \
                        and index < len(self._replay_step_rng):
                    rec_value = int(np.asarray(rec_step_rng[index]).reshape(-1)[0])
                    replay_value = self._replay_step_rng[index]
                    if rec_value != 0 and replay_value is not None:
                        step_rng_flags.append(bool(int(rec_value) == int(replay_value)))
            drift.update({
                "stepwise_n_common": int(n_common),
                "stepwise_pos_p95_m": _p95(step_errs),
                "stepwise_pos_max_m": _max_abs(step_errs),
                "stepwise_speed_mean_abs": _mean_abs(speed_errs),
                "stepwise_rng_match_frac": (
                    float(np.mean([bool(x) for x in step_rng_flags])) if step_rng_flags else None
                ),
            })
        else:
            drift.update({
                "stepwise_n_common": 0, "stepwise_pos_p95_m": None, "stepwise_pos_max_m": None,
                "stepwise_speed_mean_abs": None, "stepwise_rng_match_frac": None,
            })
        # ---- ④ termination 一致性 ----
        rec_steps = int(record.n_steps)
        step_tol = float(self.cf_thresholds.get("termination_step_tol", 2.0))
        termination_consistency: Dict[str, Any] = {
            "replay_steps": int(steps), "recorded_steps": rec_steps,
            "replay_termination": str(termination),
            "recorded_termination_code": int(record.termination_code),
            "step_delta": (None if rec_steps <= 0 else int(abs(int(steps) - rec_steps))),
        }
        if rec_steps > 0:
            termination_consistency["kind_match"] = bool(
                _termination_code(termination) == int(record.termination_code)
            )
            termination_consistency["consistent"] = bool(
                termination_consistency["kind_match"]
                and termination_consistency["step_delta"] <= step_tol
            )
        else:
            termination_consistency["kind_match"] = None
            termination_consistency["consistent"] = None
        # ---- 阈剔除（oracle_eligible） ----
        th = self.cf_thresholds
        reasons: List[str] = []
        if drift["pose_pos_p95_m"] is not None and drift["pose_pos_p95_m"] > float(th["pose_p95_m"]):
            reasons.append(f"pose_p95>{th['pose_p95_m']}")
        if drift["pose_pos_max_m"] is not None and drift["pose_pos_max_m"] > float(th["pose_max_m"]):
            reasons.append(f"pose_max>{th['pose_max_m']}")
        if drift["heading_p95_rad"] is not None and drift["heading_p95_rad"] > float(th["heading_p95_rad"]):
            reasons.append(f"heading_p95>{th['heading_p95_rad']}")
        if drift["od_mean_abs"] is not None and drift["od_mean_abs"] > float(th["od_mean_abs"]):
            reasons.append(f"od_mean>{th['od_mean_abs']}")
        if drift["od_max_abs"] is not None and drift["od_max_abs"] > float(th["od_max_abs"]):
            reasons.append(f"od_max>{th['od_max_abs']}")
        if drift["rng_match_frac"] is not None \
                and drift["rng_match_frac"] < float(th["rng_match_frac_min"]):
            reasons.append(f"rng_match<{th['rng_match_frac_min']}")
        if drift["stepwise_pos_p95_m"] is not None \
                and drift["stepwise_pos_p95_m"] > float(th["stepwise_pos_p95_m"]):
            reasons.append(f"stepwise_pos_p95>{th['stepwise_pos_p95_m']}")
        if drift["stepwise_speed_mean_abs"] is not None \
                and drift["stepwise_speed_mean_abs"] > float(th["stepwise_speed_mean_abs"]):
            reasons.append(f"stepwise_speed>{th['stepwise_speed_mean_abs']}")
        if termination_consistency["consistent"] is False:
            reasons.append("termination_mismatch")
        injection_counts = dict((hook_summary or {}).get("counts") or {})
        n_injected = int(injection_counts.get("n_events", 0))
        n_illegal = int(injection_counts.get("n_illegal", 0))
        n_unproven = int(injection_counts.get("n_unproven", 0))
        illegal_frac = (float(n_illegal) / n_injected) if n_injected else 0.0
        unproven_frac = (float(n_unproven) / n_injected) if n_injected else 0.0
        if illegal_frac > float(th["illegal_frac_max"]):
            reasons.append(f"illegal_frac>{th['illegal_frac_max']}")
        if unproven_frac > float(th["unproven_frac_max"]):
            reasons.append(f"unproven_frac>{th['unproven_frac_max']}")
        eligible = bool(not reasons)
        self.record = None
        self.hook = None
        return {"oracle": {
            "mode": "inject",
            "route": self._oracle_route,
            "eligible": eligible,
            "excluded_reasons": reasons,
            "thresholds": {key: float(value) for key, value in self.cf_thresholds.items()},
            "drift": drift,
            "termination": termination_consistency,
            "action_condition": {
                "n_injected_steps": n_injected,
                "n_legal_steps": int(injection_counts.get("n_legal", 0)),
                "n_illegal_steps": n_illegal,
                "n_unproven_steps": n_unproven,
                "illegal_frac": illegal_frac,
                "unproven_frac": unproven_frac,
            },
            "injection": {**dict(self._cf_counts), "hook": hook_summary},
        }}


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
    speed_samples: List[float] = []
    steer_abs_samples: List[float] = []
    throttle_samples: List[float] = []
    action_ds_samples: List[float] = []
    action_dtheta_samples: List[float] = []
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
        speed_samples.append(speed)
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
        action_ds_samples.append(abs(float(action_info.get("action_ds_m", float("nan")))))
        action_dtheta_samples.append(abs(float(action_info.get("action_dtheta_rad", float("nan")))))
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
    speed_metrics = _speed_metrics(speed_samples, dt)
    result = {
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
        "mean_speed_mps": speed_metrics["mean_speed_mps"],
        "final_speed_mps": speed_metrics["final_speed_mps"],
        "low_speed_step_ratio": speed_metrics["low_speed_step_ratio"],
        "crawl_seconds": speed_metrics["crawl_seconds"],
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
        "action_ds_mean_m": _finite_mean(action_ds_samples),
        "action_dtheta_abs_mean_rad": _finite_mean(action_dtheta_samples),
        "lead_gap_mean_m": float(np.mean(lead_gap_samples)) if lead_gap_samples else -1.0,
        # 池化样本（聚合后由 _public_episode 剥离，不进 JSON/CSV）
        "_a_lon": a_lon_samples,
        "_a_lat": a_lat_samples,
        "_jerk": jerk_samples,
        "_speed_ratio": speed_ratio_samples,
    }
    # R41-A1：两遍法收尾（记录落 npz / 注入裁决聚合；默认关 ⇒ finish_episode 返回 {}，
    # episode 键集与数值逐位不变）
    finish_episode = getattr(controller, "finish_episode", None)
    if callable(finish_episode):
        extra = finish_episode(
            env, steps=int(steps), termination=str(termination),
            success=bool(success), collision=bool(collision), off_road=bool(off_road),
        )
        if isinstance(extra, Mapping):
            result.update(extra)
    return result


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
        "final_speed_mps": float("nan"),
        "low_speed_step_ratio": float("nan"),
        "crawl_seconds": float("nan"),
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
        "action_ds_mean_m": float("nan"),
        "action_dtheta_abs_mean_rad": float("nan"),
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
    except SystemExit:
        # P0-A fail-fast：ckpt 部分加载被拒（_load_ckpt_model）时必须硬失败，
        # 不得降级为 termination="error" 的普通样本继续产出指标。
        raise
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
    """NaN/Inf → null，numpy 标量 → Python 标量，保证 JSON 合法。

    R41-A1：追加 torch.Tensor 分支（Oracle Injection Hook 的替换前后 state/policy 输出；
    评测导入期仍不 import torch —— 鸭子类型判定 module 前缀 + ``detach/tolist``）。
    """
    if isinstance(obj, Mapping):
        return {key: _sanitize(value) for key, value in obj.items()}
    module = type(obj).__module__
    if module.startswith("torch") and hasattr(obj, "detach") and hasattr(obj, "tolist"):
        try:
            return _sanitize(obj.detach().to("cpu").numpy())
        except Exception:  # noqa: BLE001 - 取证兜底（不阻断落盘）
            return str(obj)
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


def _oracle_audit_report(episodes: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """R41-A1：oracle 审计汇总（五要件证据 + **裁决视图**；无 oracle 数据 ⇒ None）。

    - ``route``：①/② 路线声明与 worker 内实测探针（``probe_env_snapshot_support``）；
    - ``action_condition``：注入步 legal/illegal/unproven 聚合（**非法组合不得进入裁决**）；
    - ``eligible_view``：只对 ``oracle_eligible=True`` 的 episode 求 KPI 汇总（剔除后的裁决口径）；
      被剔除 episode 明细（id + 原因）如实列出。
    """
    oracle_rows = [(ep, ep.get("oracle")) for ep in episodes if isinstance(ep.get("oracle"), Mapping)]
    if not oracle_rows:
        return None
    record_rows = [(ep, o) for ep, o in oracle_rows if str(o.get("mode")) == "record"]
    inject_rows = [(ep, o) for ep, o in oracle_rows if str(o.get("mode")) == "inject"]
    speed_rows = [(ep, o) for ep, o in oracle_rows if str(o.get("mode")) == "speed_only"]
    audit: Dict[str, Any] = {
        "n_record_episodes": len(record_rows),
        "n_inject_episodes": len(inject_rows),
        "n_speed_only_episodes": len(speed_rows),
        "route": None,
        "action_condition": {},
        "thresholds": None,
    }
    if inject_rows:
        first = inject_rows[0][1]
        audit["route"] = first.get("route")
        audit["thresholds"] = first.get("thresholds")
        totals = {key: 0 for key in (
            "n_injected_steps", "n_legal_steps", "n_illegal_steps", "n_unproven_steps",
        )}
        for _, oracle in inject_rows:
            block = dict(oracle.get("action_condition") or {})
            for key in totals:
                totals[key] += int(block.get(key, 0))
        totals["illegal_frac"] = (
            float(totals["n_illegal_steps"]) / totals["n_injected_steps"]
            if totals["n_injected_steps"] else 0.0
        )
        totals["unproven_frac"] = (
            float(totals["n_unproven_steps"]) / totals["n_injected_steps"]
            if totals["n_injected_steps"] else 0.0
        )
        audit["action_condition"] = totals
        eligible = [ep for ep, oracle in inject_rows if bool(oracle.get("eligible"))]
        excluded = [
            {"id": int(ep.get("id", -1)), "reasons": list(oracle.get("excluded_reasons") or [])}
            for ep, oracle in inject_rows if not bool(oracle.get("eligible"))
        ]
        audit["n_eligible"] = len(eligible)
        audit["n_excluded"] = len(excluded)
        audit["excluded"] = excluded
        audit["eligible_view"] = summarize(eligible) if eligible else None
    return audit


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
    report: Dict[str, Any] = {
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
    # R41-A1：oracle 审计（含裁决视图）；默认全关 ⇒ 不追加键（metrics 逐字节不变）
    oracle_audit = _oracle_audit_report(episodes)
    if oracle_audit is not None:
        report["oracle_audit"] = oracle_audit
    return report


CSV_FIELDS = (
    "id", "seed", "split", "difficulty", "primary", "geometry", "extra_geometry", "compound",
    "success", "collision", "off_road", "solid_line_crossing", "speed_limit_violation",
    "speed_limit_violation_step_rate", "termination", "route_completion", "max_route_completion",
    "steps", "duration_s", "min_ttc", "mean_speed_mps", "final_speed_mps",
    "low_speed_step_ratio", "crawl_seconds",
    "a_lon_mean", "a_lon_abs_p95", "a_lat_mean", "a_lat_abs_p95", "jerk_mean", "jerk_abs_p95",
    "speed_ratio_mean", "speed_ratio_p95", "steer_abs_mean", "throttle_mean",
    "action_ds_mean_m", "action_dtheta_abs_mean_rad", "lead_gap_mean_m",
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


def write_eval_tensorboard(report: Mapping[str, Any], monitor_dir: Path, *, step: int = 1) -> bool:
    """把 ``metrics.json`` 的 overall / by_primary / by_difficulty **全量 KPI** 写 TB（lane B B5）。

    tag：``eval/<metric>``；分组 ``eval/by_primary/<metric>/<group>``、
    ``eval/by_difficulty/<metric>/<group>``（同 run 内并图）；单点 ``step=1``；
    **不写逐 spec**（明细在 episodes.csv）。tensorboard 不可用 → 返回 False（不 raise）。
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001 - 可选依赖
        print(f"[eval_runner] tensorboard 不可用（{type(exc).__name__}: {exc}）→ 跳过 eval TB", flush=True)
        return False

    def _numeric(group: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in (group or {}).items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                out[str(key)] = float(value)
        return out

    monitor_dir.mkdir(parents=True, exist_ok=True)
    try:
        writer = SummaryWriter(log_dir=str(monitor_dir))
        for metric, value in _numeric(report.get("overall")).items():
            writer.add_scalar(f"eval/{metric}", value, global_step=step)
        for group_name, group in (report.get("by_primary") or {}).items():
            for metric, value in _numeric(group).items():
                writer.add_scalar(f"eval/by_primary/{metric}/{group_name}", value, global_step=step)
        for group_name, group in (report.get("by_difficulty") or {}).items():
            for metric, value in _numeric(group).items():
                writer.add_scalar(f"eval/by_difficulty/{metric}/{group_name}", value, global_step=step)
        writer.flush()
        writer.close()
    except Exception as exc:  # noqa: BLE001 - 评测结果不应因 TB 失败而丢失
        print(f"[eval_runner] eval TB 写入失败（{type(exc).__name__}: {exc}）", flush=True)
        return False
    return True


def _print_summary(report: Mapping[str, Any]) -> None:
    """打印 overall / by_primary / by_difficulty 简表 + 判定结论。"""
    header = (
        f"{'group':<22}{'n':>5}{'succ':>8}{'95% CI':>18}{'coll':>7}{'offrd':>7}"
        f"{'ttc':>7}{'a_lat95':>9}{'spd':>7}{'crwls':>8}{'pass':>7}"
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
            f"{group.get('speed_ratio_mean', float('nan')):>7.3f}"
            f"{group.get('crawl_seconds', float('nan')):>8.1f}{pass_text:>7}",
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
    parser.add_argument("--allow-partial-load", action="store_true",
                        help="显式放行 ckpt 部分加载（missing/unexpected/shape_mismatch 非零）；"
                             "默认 fail-fast 打印三计数与明细后退出码 2；放行时强制在日志与 "
                             "metrics.json::meta.ckpt_load 记录三计数与 missing 列表")
    parser.add_argument("--out", type=Path, default=Path("runs/eval"),
                        help="评测输出根目录（实际写到 <out>/<name>/，默认 runs/eval）")
    parser.add_argument("--name", default=None,
                        help="运行名（默认 'BTC<秒级北京戳>_<spec kind>_<policy>'；非规范名自动规范化，"
                             "路径型 --name 原样；落地 <out>/<name>/metrics.json）")
    parser.add_argument("--limit", type=int, default=None, help="只评测前 N 条 spec（按文件顺序，冒烟用）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"spawn worker 数（默认 {DEFAULT_WORKERS}，评测与训练池互斥）")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help=f"单 episode 最大步数（默认 {DEFAULT_MAX_STEPS} = 100 s @ 0.1 s）")
    parser.add_argument("--baseline-ref", type=Path, default=Path(DEFAULT_BASELINE_REF),
                        help="冻结基线 JSON（默认 runs/baseline_eval/val_reference.json；"
                             "同目录 *_by_primary.json 用于分组判定）")
    parser.add_argument("--seed", type=int, default=0, help="评测种子（确定性；默认 0）")
    parser.add_argument("--moe-off", action="store_true",
                        help="评测时关闭 MoE（专家分支不参与；用于 A/B 的 primary-only 臂）")
    parser.add_argument("--tracker", choices=("lqr", "exact"), default="exact",
                        help="ckpt 动作执行器：exact=阶段 A/B 精确/运动学执行（默认，衡量规划轨迹本身）；"
                             "lqr=阶段 C 闭环跟踪（含控制器跟踪误差）")
    parser.add_argument("--eval-reference", choices=EVAL_REFERENCES, default="plan",
                        help="ckpt 跟踪器参考口径（默认 plan = 现状：6 步规划预览，plan[0]=action_mu）；"
                             "repeat_action = repeat(action_mu,6)（训练 A-hold 同构，cheap path，不消费 plan）")
    parser.add_argument("--path-head", choices=LOAD_PATH_HEAD_CHOICES, default="off",
                        help="动作头装配（R39-A1）：off=默认旧 PolicyHead（逐位不变）；"
                             "structured=惰性构造结构化头 policy.structured_action 并启用结构化动作"
                             "出口（口径对齐 .slim/deepwork/p1_v2/a3/eval_a3.py 的结构化装载顺序）。"
                             "结构化头前向需要 obs['lane_anchor_xy']（目标车道中心线折线，ego t0 系）："
                             "本入口用 --lane-anchor inject 显式注入（缺键否则在 forward fail-fast；"
                             "net/path_head.py::anchor_from_obs）。"
                             "装载失败面（missing/unexpected/shape_mismatch）仍由 P0-A fail-fast 把关")
    parser.add_argument("--ld-mode", choices=LOAD_LD_MODES, default="baseline",
                        help="LD 消融装配（R39-A1b；口径对齐 eval_a3/a3 canary；默认 baseline=生产口径）。"
                             "ckpt 必须与训练同模式装载：semantic=惰性构造 semantic_wiring"
                             "（K1e2 血统缺键用 --fill-from 回填）；其余档位见 net/model.py::set_ld_mode")
    parser.add_argument("--rollout-semantic", choices=LOAD_ROLLOUT_SEMANTIC_MODES, default="full",
                        help="rollout 阶段 semantic 注入口径（R39-A1b；默认 full=生产口径；"
                             "static_role_only = R37 §1.3 协议口径）")
    parser.add_argument("--lane-anchor", choices=LOAD_LANE_ANCHOR_MODES, default="off",
                        help="lane_anchor_xy 注入器（R39-A1b 选项 a 的显式开关；默认 off=不触达）："
                             "inject=每决策步从活 env 采目标车道中心线（R36 C lane 注入器同源，"
                             ".slim/deepwork/p1_v2/round36/c_lane_anchor.py）写入 obs")
    parser.add_argument("--fill-from", type=Path, default=None,
                        help="缺键回填源 ckpt（R39-A1b；显式给出才触达）：migrate 后 missing 中存在于"
                             "回填源且形状一致的键装载并计入 LoadReport.fill_from；"
                             "K1e2 血统语义键例 = runs/p1_v2/round28/K1/ckpt_epoch002.pt")
    parser.add_argument("--world-state-api", choices=LOAD_WORLD_STATE_API_MODES, default="off",
                        help="R39-B2c：统一 imagined WorldState API（默认 off=不触达、与 A1b 口径"
                             "逐字节同 hash）；analytic=k≥1 想象步几何改由 WorldState 统一 readout"
                             "（需 --path-head structured + --lane-anchor inject；与 a3 "
                             "--world-state-api 同口径）；analytic_learned=再加学习残差头 ΔA_θ"
                             "（需 --world-residual-head）。非默认档追加 protocol hash 轴 "
                             "model.world_state_api")
    parser.add_argument("--world-residual-head", type=Path, default=None,
                        help="R39-B2c：学习残差头 ckpt（net.world_residual.load_residual_head 口径；"
                             "如 runs/round39/b2/residual_head.pt）。仅 --world-state-api "
                             "analytic_learned 生效（其余组合 fail-fast）；state_dict 装载后挂载"
                             "（不影响 base ckpt 键面/LoadReport）")
    parser.add_argument("--world-residual-scale", type=float, default=None,
                        help="R39-B2d：学习残差缩放（net.model.set_world_residual_scale；默认 None="
                             "零触达、与 B2c 口径逐字节同 hash）。仅 --world-state-api "
                             "analytic_learned 生效（其余组合 fail-fast）；∈[0,4]；非默认（≠1.0）"
                             "追加 protocol hash 轴 model.world_residual_scale")
    parser.add_argument("--teacher-force", choices=LOAD_TEACHER_FORCE_MODES, default="off",
                        help="R41-A1：teacher-forced oracle 注入（默认 off=零触达、前向逐位不变）。"
                             "注入下沉为模型内原生路径（net/model.py::set_teacher_force；不再依赖 "
                             ".slim 运行时补丁）；非 off 需要 --world-state-api 非 off 与 "
                             "--inject-truth（状态注入落点 = analytic_update 后/readout 前）。"
                             "非默认档进入 protocol hash（model.teacher_force/inject_truth_sha16）"
                             "+ 模型缓存键 + metrics/manifest")
    parser.add_argument("--record-truth", type=Path, default=None,
                        help="R41-A1：两遍法第一遍（记录 pass）truth 输出目录（<dir>/<spec_id:04d>.npz；"
                             "含逐决策帧真值/命令动作/RNG 摘要 + 逐 env step 轨迹）。只观察、零执行"
                             "副作用；与 --inject-truth 互斥；非默认档入 protocol hash（model.record_truth）")
    parser.add_argument("--inject-truth", type=Path, default=None,
                        help="R41-A1：两遍法第二遍（注入 pass）truth 源目录（第一遍产物）。"
                             "只对 --teacher-force 非 off 有意义；目录内容 sha16 入 protocol hash"
                             "（model.inject_truth_sha16）；配合 Oracle Injection Hook 记录五要件并按"
                             "动作条件/五类漂移剔除非法组合（oracle 裁决视图）")
    parser.add_argument("--speed-scale", type=float, default=None,
                        help="R41-A1：速度反事实（plan 后处理同缩：ds/dθ ×s ⇒ κ 不变；默认 None=关、"
                             "1.0 视同关；须 >0 有限）。非默认档入 protocol hash + 缓存键")
    parser.add_argument("--speed-oracle", choices=LOAD_SPEED_ORACLE_MODES, default="off",
                        help="R41-A1：速度 oracle（默认 off）。expert=按专家空问 ds 缩放 plan"
                             "（clip∈[0.05,2.0]，需 pure_pursuit labeler）；非默认档入 protocol hash")
    parser.add_argument("--readout-lane-geometry", choices=LOAD_READOUT_LANE_GEOMETRY_MODES,
                        default="default",
                        help="R41-D：几何口径**实例配置**（默认 default=哨兵：按 ckpt metadata ∧ "
                             "config/model.yaml ∧ 历史默认解析）。corrected=R38-A4 两条几何缺陷修复"
                             "（= 历史内部值 fixed）；legacy_aligned=legacy 产线几何"
                             "（= 历史内部值 production）。非默认档 ⇒ 模型实例统一轴（ws readout + "
                             "ws-off 想象步）+ protocol hash（model.readout_lane_geometry）+ 模型缓存键 "
                             "+ metrics/run_meta/manifest 记录")
    parser.add_argument("--device", default=None,
                        help="ckpt 策略运行设备；默认取 config train.device（auto=cuda 可用则 cuda）")
    return parser


def _counterfactual_from_args(args: Any, eval_cfg: Mapping[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """R41-A1：从 CLI 构造反事实轴装配（校验 + 阈值覆盖 + truth 目录指纹）；非法 ⇒ ``(payload, 错误串)``。

    默认全关（``teacher_force=off``、无 record/inject、``speed_scale=None``、``speed_oracle=off``）
    ⇒ ``active=False``：控制器零触达、协议 hash/缓存键不追加任何键（逐位一致）。
    """
    teacher_force = str(getattr(args, "teacher_force", "off") or "off")
    record_dir = str(args.record_truth) if getattr(args, "record_truth", None) is not None else None
    inject_dir = str(args.inject_truth) if getattr(args, "inject_truth", None) is not None else None
    speed_scale = None if getattr(args, "speed_scale", None) is None else float(args.speed_scale)
    if speed_scale is not None and speed_scale == 1.0:
        speed_scale = None  # 显式 1.0 = 默认档（不触达、不入 hash/缓存键）
    speed_oracle = str(getattr(args, "speed_oracle", "off") or "off")
    thresholds = dict(DEFAULT_DRIFT_THRESHOLDS)
    frame_gate = dict(DEFAULT_FRAME_GATE)
    cf_cfg = dict(eval_cfg.get("counterfactual") or {}) if isinstance(eval_cfg, Mapping) else {}
    for key, value in dict(cf_cfg.get("drift_thresholds") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            thresholds[str(key)] = float(value)
    for key, value in dict(cf_cfg.get("frame_gate") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            frame_gate[str(key)] = float(value)
    active = bool(
        teacher_force != "off" or record_dir or inject_dir or speed_scale is not None
        or speed_oracle != "off"
    )
    payload: Dict[str, Any] = {
        "active": active,
        "teacher_force": teacher_force,
        "record_truth": record_dir,
        "inject_truth": inject_dir,
        "inject_truth_sha16": None,
        "inject_truth_digest": None,
        "speed_scale": speed_scale,
        "speed_oracle": speed_oracle,
        "drift_thresholds": {key: float(value) for key, value in thresholds.items()},
        "frame_gate": {key: float(value) for key, value in frame_gate.items()},
        "route": {
            "declared": "two_pass_replay",
            "env_probe": "per-episode（worker 内 probe_env_snapshot_support：engine 无 snapshot/clone）",
        },
        "protocol_hash": None,
        "out_dir": None,
    }
    if not active:
        return payload, None
    if teacher_force not in LOAD_TEACHER_FORCE_MODES:
        return payload, f"未知 --teacher-force {teacher_force!r}"
    if speed_oracle not in LOAD_SPEED_ORACLE_MODES:
        return payload, f"未知 --speed-oracle {speed_oracle!r}"
    if str(getattr(args, "policy", "")) != "ckpt":
        return payload, "反事实轴（--teacher-force/--record-truth/--inject-truth/--speed-*）只在 --policy ckpt 生效"
    if record_dir and inject_dir:
        return payload, "--record-truth 与 --inject-truth 互斥（两遍法 = 两次独立 run）"
    if inject_dir and teacher_force == "off":
        return payload, "--inject-truth 只对 --teacher-force 非 off 有意义"
    if teacher_force != "off":
        if str(getattr(args, "world_state_api", "off")) == "off":
            return payload, ("--teacher-force 非 off 需要 --world-state-api 非 off"
                             "（状态注入落点 = analytic_update 后 / readout 前）")
        if inject_dir is None:
            return payload, "--teacher-force 非 off 需要 --inject-truth（第一遍记录产物）"
    if speed_scale is not None and not (np.isfinite(speed_scale) and speed_scale > 0.0):
        return payload, f"--speed-scale 必须 >0 有限，收到 {speed_scale}"
    if inject_dir is not None:
        truth_root = Path(inject_dir)
        if not truth_root.is_dir():
            return payload, f"--inject-truth 目录不存在：{inject_dir}"
        digest = _truth_dir_digest(truth_root)
        if int(digest["n_files"]) <= 0:
            return payload, f"--inject-truth 目录无 npz 文件：{inject_dir}"
        payload["inject_truth_sha16"] = str(digest["sha16"])
        payload["inject_truth_digest"] = {
            "dir": str(truth_root), "n_files": int(digest["n_files"]), "sha16": str(digest["sha16"]),
        }
    if record_dir is not None:
        Path(record_dir).mkdir(parents=True, exist_ok=True)
    return payload, None


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
    from pipeline import run_paths  # run 名规范（--out 是路径根；调用方不手拼名字）

    run_name = run_paths.eval_run_name(args.name, spec=spec_path, policy=args.policy)
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_ref = Path(args.baseline_ref)
    baseline_doc = _load_json(baseline_ref) if baseline_ref.is_file() else None
    baseline_primary_doc = _load_json(_baseline_primary_path(baseline_ref))
    if baseline_doc is None:
        print(f"[eval_runner] 警告：基线参照不可用（{baseline_ref}），判定将标记为缺少可比数据",
              flush=True)

    model_config: Dict[str, Any] = task_model_config(config)

    # R39-A1b：模型装配开关（口径对齐 a3 canary；缺省 = 生产口径逐位不变）
    # R39-B2c：追加 world_state_api / world_residual_head（默认 off/none ⇒ 不触达任何 net API）
    model_assembly: Dict[str, Any] = {
        "path_head": str(args.path_head),
        "ld_mode": str(args.ld_mode),
        "rollout_semantic": str(args.rollout_semantic),
        "lane_anchor": str(args.lane_anchor),
        "fill_from": (str(args.fill_from) if args.fill_from is not None else None),
        "world_state_api": str(args.world_state_api),
        "world_residual_head": (str(args.world_residual_head)
                                if args.world_residual_head is not None else None),
        "world_residual_scale": (float(args.world_residual_scale)
                                 if args.world_residual_scale is not None else None),
    }
    # R41-D：几何口径实例配置（CLI > config/model.yaml；哨兵 ⇒ 不写装配键 ⇒ 默认逐位不变）
    geometry_cli = _normalize_readout_lane_geometry(getattr(args, "readout_lane_geometry", None))
    geometry_config = _readout_geometry_config_value(model_config)
    geometry_requested = geometry_cli if geometry_cli is not None else geometry_config
    geometry_requested_source = "cli" if geometry_cli is not None else (
        "model_config" if geometry_config is not None else "default"
    )
    if geometry_requested is not None:
        model_assembly["readout_lane_geometry"] = geometry_requested
        model_assembly["readout_lane_geometry_source"] = geometry_requested_source
        if args.policy != "ckpt":
            print("[eval_runner] 警告：--readout-lane-geometry 非默认只在 --policy ckpt 生效"
                  "（基线臂忽略）", flush=True)
        elif str(args.path_head) != "structured":
            print("[eval_runner] 警告：--readout-lane-geometry 非默认在 --path-head off（旧头）下"
                  "不消费想象步几何（无触达；hash/缓存键仍按显式轴区分）", flush=True)
    if str(args.world_state_api) != "off" and args.policy != "ckpt":
        print("[eval_runner] 警告：--world-state-api 非默认只在 --policy ckpt 生效（基线臂忽略）",
              flush=True)
    if str(args.world_state_api) != "off" and str(args.path_head) != "structured":
        print("[eval_runner] WorldState API FAIL-FAST：--world-state-api "
              f"{args.world_state_api} 需要 --path-head structured（readout = 结构化头想象步几何输入）",
              file=sys.stderr)
        return 2
    if str(args.world_state_api) == "analytic_learned" and args.world_residual_head is None:
        print("[eval_runner] WorldState API FAIL-FAST：--world-state-api analytic_learned 需要 "
              "--world-residual-head <ckpt>", file=sys.stderr)
        return 2
    if str(args.world_state_api) == "off" and args.world_residual_head is not None:
        print("[eval_runner] WorldState API FAIL-FAST：--world-residual-head 只对 "
              "--world-state-api analytic_learned 有意义", file=sys.stderr)
        return 2
    if str(args.world_state_api) != "off" and args.world_residual_head is not None:
        if not Path(str(args.world_residual_head)).is_file():
            print(f"[eval_runner] WorldState API FAIL-FAST：缺 residual head ckpt "
                  f"{args.world_residual_head}", file=sys.stderr)
            return 2
    # R39-B2d：残差缩放轴 fail-fast（默认 None ⇒ 零触达）
    if args.world_residual_scale is not None:
        if str(args.world_state_api) != "analytic_learned":
            print("[eval_runner] WorldState API FAIL-FAST：--world-residual-scale 只对 "
                  "--world-state-api analytic_learned 有意义", file=sys.stderr)
            return 2
        if not 0.0 <= float(args.world_residual_scale) <= 4.0:
            print(f"[eval_runner] WorldState API FAIL-FAST：--world-residual-scale 必须 ∈ [0,4]，"
                  f"收到 {args.world_residual_scale}", file=sys.stderr)
            return 2

    # ---- R41-A1：oracle/teacher-force/speed 反事实轴（默认全关 ⇒ 零触达、逐位不变）----
    counterfactual, cf_error = _counterfactual_from_args(args, eval_cfg)
    if cf_error is not None:
        print(f"[eval_runner] counterfactual FAIL-FAST：{cf_error}", file=sys.stderr)
        return 2
    counterfactual["out_dir"] = str(out_dir)
    if counterfactual["active"]:
        print(
            f"[eval_runner] counterfactual 激活：teacher_force={counterfactual['teacher_force']} "
            f"record_truth={counterfactual['record_truth']} inject_truth={counterfactual['inject_truth']} "
            f"speed_scale={counterfactual['speed_scale']} speed_oracle={counterfactual['speed_oracle']} "
            f"(truth_sha16={counterfactual['inject_truth_sha16']})",
            flush=True,
        )

    # ---- P0-A 评测安全前置：ckpt 加载 fail-fast（默认开；父进程 CPU 预检，干净加载零行为变化）----
    ckpt_load_summary: Optional[Dict[str, Any]] = None
    if args.policy == "ckpt":
        print(
            f"[eval_runner] ckpt 加载预检（fail-fast={'off' if args.allow_partial_load else 'on'}, "
            f"assembly={model_assembly}）: {args.ckpt}",
            flush=True,
        )
        _load_ckpt_model(
            str(args.ckpt), model_config, "cpu",
            allow_partial=bool(args.allow_partial_load),
            path_head=model_assembly["path_head"],
            ld_mode=model_assembly["ld_mode"],
            rollout_semantic=model_assembly["rollout_semantic"],
            fill_from=model_assembly["fill_from"],
            world_state_api=model_assembly["world_state_api"],
            world_residual_head=model_assembly["world_residual_head"],
            world_residual_scale=model_assembly["world_residual_scale"],
            teacher_force=counterfactual["teacher_force"],
            record_truth=bool(counterfactual["record_truth"]),
            inject_truth_sha16=counterfactual["inject_truth_sha16"],
            speed_scale=counterfactual["speed_scale"],
            speed_oracle=counterfactual["speed_oracle"],
            readout_lane_geometry=model_assembly.get("readout_lane_geometry"),
            readout_lane_geometry_source=model_assembly.get("readout_lane_geometry_source"),
        )
        ckpt_load_summary = _ckpt_load_summary(
            str(args.ckpt), "cpu",
            path_head=model_assembly["path_head"],
            ld_mode=model_assembly["ld_mode"],
            rollout_semantic=model_assembly["rollout_semantic"],
            fill_from=model_assembly["fill_from"],
            world_state_api=model_assembly["world_state_api"],
            world_residual_head=model_assembly["world_residual_head"],
            world_residual_scale=model_assembly["world_residual_scale"],
            teacher_force=counterfactual["teacher_force"],
            record_truth=bool(counterfactual["record_truth"]),
            inject_truth_sha16=counterfactual["inject_truth_sha16"],
            speed_scale=counterfactual["speed_scale"],
            speed_oracle=counterfactual["speed_oracle"],
            readout_lane_geometry=model_assembly.get("readout_lane_geometry"),
        )

    # R41-D：几何口径**解析结果**（含 ckpt metadata 采用值；hash/metrics 以实际生效值为准）。
    # 非 ckpt 臂（baseline）不消费几何轴 ⇒ 解析记录 None。
    geometry_resolved: Optional[Dict[str, Any]] = None
    if args.policy == "ckpt":
        geometry_resolved = dict((ckpt_load_summary or {}).get("geometry") or {})
        if not geometry_resolved:
            # 纯 legacy 默认加载（无 meta/无声明）⇒ 显式补一条（内部记录用；不落默认产物）
            geometry_resolved = _geometry_artifact_record(_resolve_readout_lane_geometry(
                requested=geometry_requested, requested_source=geometry_requested_source,
                ckpt_meta=None, config_value=geometry_config,
            ))
        geometry_value = geometry_resolved.get("readout_lane_geometry")
    else:
        geometry_value = None

    # ---- R39-A1：四 hash 原生记录（protocol / model tree / schema / checkpoint）→ run_meta ----
    eval_hashes = _evaluation_hashes(
        spec_path=spec_path, policy=str(args.policy),
        ckpt=str(args.ckpt) if args.ckpt is not None else None,
        tracker=str(args.tracker), eval_reference=str(args.eval_reference),
        moe_off=bool(getattr(args, "moe_off", False)), path_head=model_assembly["path_head"],
        ld_mode=model_assembly["ld_mode"], rollout_semantic=model_assembly["rollout_semantic"],
        lane_anchor=model_assembly["lane_anchor"], fill_from=model_assembly["fill_from"],
        workers=int(workers), max_steps=int(args.max_steps), device=device,
        world_state_api=model_assembly["world_state_api"],
        world_residual_head=model_assembly["world_residual_head"],
        world_residual_scale=model_assembly["world_residual_scale"],
        teacher_force=counterfactual["teacher_force"],
        record_truth=bool(counterfactual["record_truth"]),
        inject_truth_sha16=counterfactual["inject_truth_sha16"],
        speed_scale=counterfactual["speed_scale"],
        speed_oracle=counterfactual["speed_oracle"],
        readout_lane_geometry=geometry_value,
        readout_lane_geometry_source=(
            str(geometry_resolved.get("source")) if geometry_resolved else None
        ),
    )
    counterfactual["protocol_hash"] = str(eval_hashes["evaluation_protocol_hash"])
    run_meta: Dict[str, Any] = {
        "schema": "r39a1-run-meta/v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "evaluator": "pipeline.eval_runner",
        "git_commit": _git_commit(),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "policy": str(args.policy),
        "ckpt": str(args.ckpt) if args.ckpt is not None else None,
        "run_name": run_name,
        "out_dir": str(out_dir),
        "spec": str(spec_path),
        "tracker": str(args.tracker) if args.policy == "ckpt" else None,
        "eval_reference": str(args.eval_reference) if args.policy == "ckpt" else None,
        "path_head": model_assembly["path_head"],
        "model_assembly": model_assembly,
        "fill_from": eval_hashes["fill_from"],
        "world_state": {**eval_hashes["world_state"],
                        "protocol_model": {
                            k: v for k, v in eval_hashes["protocol"]["model"].items()
                            if str(k).startswith("world_")}},
        # R41-D：几何实例配置解析留痕（非默认才追加键 ⇒ 默认 run_meta 结构不变）
        **({"geometry": {
            **dict(geometry_resolved),
            "protocol_model": eval_hashes["protocol"]["model"].get(CKPT_GEOMETRY_META_KEY),
        }} if geometry_value not in (None, "default") else {}),
        "hashes": {
            "evaluation_protocol_hash": eval_hashes["evaluation_protocol_hash"],
            "evaluation_protocol_hash_full": eval_hashes["evaluation_protocol_hash_full"],
            "evaluation_protocol_common_hash": eval_hashes["evaluation_protocol_common_hash"],
            "model_tree_hash": eval_hashes["model_tree_hash"],
            "schema_hash": eval_hashes["schema_hash"],
            "checkpoint_hash": eval_hashes["checkpoint_hash"],
            "checkpoint_hash16": eval_hashes["checkpoint_hash16"],
        },
        "protocol": eval_hashes["protocol"],
        "model_tree": eval_hashes["model_tree"],
        "schema": eval_hashes["schema"],
        "ckpt_load": ckpt_load_summary,
        # R41-A1：反事实轴留痕（默认全关也记录声明值；非默认档书面化到 manifest）
        "counterfactual": {
            **{key: counterfactual[key] for key in (
                "active", "teacher_force", "record_truth", "inject_truth", "inject_truth_sha16",
                "speed_scale", "speed_oracle", "route",
            )},
            "inject_truth_digest": counterfactual["inject_truth_digest"],
            "drift_thresholds": counterfactual["drift_thresholds"],
            "frame_gate": counterfactual["frame_gate"],
        } if args.policy == "ckpt" else None,
    }
    run_meta_path = _record_run_meta(out_dir, run_meta)
    print(
        f"[eval_runner] R39-A1 hashes: protocol={eval_hashes['evaluation_protocol_hash']} "
        f"model_tree={eval_hashes['model_tree_hash']} schema={eval_hashes['schema_hash']} "
        f"ckpt={(eval_hashes['checkpoint_hash'] or 'n/a')[:16]} -> run_meta.json",
        flush=True,
    )

    tasks: List[Dict[str, Any]] = [
        {
            "spec": spec,
            "policy": args.policy,
            "ckpt": str(args.ckpt) if args.ckpt is not None else None,
            "max_steps": int(args.max_steps),
            "traffic_density": None,
            "policy_params": {},
            "model_config": model_config,
            "obs_config": dict(env_cfg.get("obs") or {}),
            "tracker_config": dict(env_cfg.get("tracking") or {}),
            "device": device,
            "workers": int(workers),
            "torch_threads": torch_threads,
            "omp_num_threads": int(omp_num_threads),
            "tracker": str(args.tracker),
            "eval_reference": str(args.eval_reference),
            "moe_off": bool(getattr(args, "moe_off", False)),
            "allow_partial_load": bool(args.allow_partial_load),
            # R39-A1b：模型装配（controller 装载 + lane_anchor 注入的唯一透传口）
            "model_assembly": dict(model_assembly),
            # R41-A1：反事实轴（controller 侧唯一透传口；默认全关 ⇒ 零触达）
            "counterfactual": dict(counterfactual),
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
        # P0-A 版本戳（§2.2）：evaluator 标识 / git commit / argv / ckpt sha256；
        # config 快照已由入口层 manifest（run_paths.write_manifest）落盘。
        "evaluator": "pipeline.eval_runner",
        "git_commit": _git_commit(),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "ckpt_sha256": (ckpt_load_summary or {}).get("sha256"),
        # P0-A 加载审计：三计数 + missing 列表（--allow-partial-load 放行时强制记录；干净加载同样落盘）
        "ckpt_load": ckpt_load_summary,
        # R39-A1：四 hash（protocol/model tree/schema/ckpt）+ 迁移 LoadReport 同步记录于
        # run_meta.json（本处落指标文件，保证 metrics 与 run_meta 可互相审计）
        "hashes": run_meta["hashes"],
        "load_report": (ckpt_load_summary or {}).get("load_report"),
        "policy_class": ("env.expert.pure_pursuit_idm.PurePursuitIDMPolicy"
                         if args.policy == "baseline" else "net.model.DrivingModel"),
        "tracker": str(args.tracker) if args.policy == "ckpt" else None,
        "eval_reference": str(args.eval_reference) if args.policy == "ckpt" else None,
        # R39-A1b：模型装配开关（ld_mode/rollout_semantic/path_head/lane_anchor/fill_from）
        "model_assembly": dict(model_assembly) if args.policy == "ckpt" else None,
        # R41-D：几何实例配置解析（非默认才追加键 ⇒ 默认 metrics 结构不变）
        **({"geometry": dict(geometry_resolved)}
           if args.policy == "ckpt" and geometry_value not in (None, "default") else {}),
        # R41-A1：反事实轴（teacher_force/record_truth/inject_truth(+sha16)/speed_* + 阈值 + 路线）
        "counterfactual": {
            **{key: counterfactual[key] for key in (
                "active", "teacher_force", "record_truth", "inject_truth", "inject_truth_sha16",
                "speed_scale", "speed_oracle", "route", "protocol_hash",
            )},
            "inject_truth_digest": counterfactual["inject_truth_digest"],
            "drift_thresholds": counterfactual["drift_thresholds"],
            "frame_gate": counterfactual["frame_gate"],
        } if args.policy == "ckpt" else None,
        "fill_from": eval_hashes["fill_from"],
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
    if write_eval_tensorboard(report, out_dir / "monitor"):
        print(f"[eval_runner] TB -> {out_dir / 'monitor'}（eval/* + by_primary/by_difficulty）")

    _print_summary(report)
    verdict = report["verdict"]
    print(
        f"[eval_runner] verdict: all_passed={verdict['all_passed']} "
        f"(judged checks={verdict['n_checks_judged']})"
    )
    print(f"[eval_runner] metrics -> {metrics_path}")
    print(f"[eval_runner] episodes -> {csv_path} (n={len(episodes)}, errors={meta['n_error']})")
    # R39-A1：run_meta 终稿（补 wall_time / 判定；hashes/LoadReport 已在预检后写入）
    run_meta.update({
        "wall_time_s": wall_time,
        "n_evaluated": len(episodes),
        "n_error": meta["n_error"],
        "verdict": {"all_passed": bool(verdict["all_passed"]),
                    "n_checks_judged": int(verdict["n_checks_judged"])},
        "metrics_path": str(metrics_path),
    })
    oracle_audit = report.get("oracle_audit")
    if oracle_audit is not None:
        # R41-A1：反事实 run 的裁决视图随 run_meta 落盘（默认全关 ⇒ 不新增键）
        run_meta["oracle_audit"] = oracle_audit
    _record_run_meta(out_dir, run_meta)  # manifest 已含标记 ⇒ 幂等只更新 run_meta.json
    if oracle_audit is not None:
        print(
            f"[eval_runner] oracle_audit: inject_episodes={oracle_audit.get('n_inject_episodes')} "
            f"eligible={oracle_audit.get('n_eligible')} excluded={oracle_audit.get('n_excluded')} "
            f"steps={dict(oracle_audit.get('action_condition') or {}).get('n_injected_steps')} "
            f"illegal={dict(oracle_audit.get('action_condition') or {}).get('n_illegal_steps')}",
            flush=True,
        )
    print(f"[eval_runner] run_meta -> {run_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
