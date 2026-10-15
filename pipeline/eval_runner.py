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
#: - ``--fill-from``：缺键回填源 ckpt（K1e2 血统语义键；显式给出才触达）。
LOAD_PATH_HEAD_CHOICES: tuple[str, ...] = ("off", "structured")
LOAD_LD_MODES: tuple[str, ...] = (
    "baseline", "no_chain", "set_attn", "attn", "attn_no_chain", "semantic", "semantic_no_chain",
)
LOAD_ROLLOUT_SEMANTIC_MODES: tuple[str, ...] = ("full", "t0_only", "static_role_only")
LOAD_LANE_ANCHOR_MODES: tuple[str, ...] = ("off", "inject")


def _model_cache_key(
    ckpt: str,
    device: str,
    *,
    path_head: str = "off",
    ld_mode: str = "baseline",
    rollout_semantic: str = "full",
    fill_from: Optional[str] = None,
) -> Tuple:
    """缓存/加载摘要键：**全默认装配 = 旧键 ``(ckpt, device)``**（历史调用点逐位不变；
    a3 controller 的 canary 预填缓存按该旧键命中）；任何非默认轴追加一段避免互撞。"""
    key: Tuple = (str(ckpt), str(device))
    if str(path_head or "off") != "off":
        key += (("path_head", str(path_head)),)
    if str(ld_mode or "baseline") != "baseline":
        key += (("ld_mode", str(ld_mode)),)
    if str(rollout_semantic or "full") != "full":
        key += (("rollout_semantic", str(rollout_semantic)),)
    if fill_from:
        key += (("fill_from", str(fill_from)),)
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
) -> Dict[str, Any]:
    """正式入口 evaluation protocol（口径复用 c6：``build_protocol`` + canonical JSON sha256 hex16）。

    轴映射（正式入口 → c6 协议字段；记录在 ``notes``）：
    ``control.tracker ← --tracker``；``control.tail ← --eval-reference``（正式入口参考口径）；
    ``model.ld_mode ← --ld-mode`` / ``model.rollout_semantic ← --rollout-semantic``（R39-A1b 起为
    正式入口开关，**入 hash**）；``model.moe ← --moe-off``；``model.path_head ← --path-head``；
    R39-A1b 追加 ``model.lane_anchor``（``--lane-anchor`` 注入开关）与 ``model.fill_from_sha16``
    （``--fill-from`` 源 ckpt 内容指纹；none）—— 均**入 hash**（模型输入/权重装配的可辨识性）。
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
) -> Dict[str, Any]:
    """R39-A1 四 hash（+ 完整 protocol dict；供 run_meta/manifest 与 metrics 记录）。"""
    net_tree = _net_tree_digest()
    from env.obs import OBS_SCHEMA_VERSION, obs_fingerprint  # 轻量（stdlib only）

    fill_sha = _sha256_file(str(fill_from)) if fill_from else None
    protocol = _formal_entry_protocol(
        spec_path=spec_path, policy=policy, tracker=tracker, eval_reference=eval_reference,
        moe_off=moe_off, path_head=path_head, ld_mode=ld_mode, rollout_semantic=rollout_semantic,
        lane_anchor=lane_anchor, fill_from_sha16=(fill_sha[:16] if fill_sha else None),
        workers=workers, max_steps=max_steps, device=device,
        net_tree_sha16=str(net_tree["tree_sha16"]),
    )
    ckpt_sha = _sha256_file(str(ckpt)) if ckpt is not None else None
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
    return meta_path


def _ckpt_load_summary(
    ckpt: str,
    device: str = "cpu",
    path_head: str = "off",
    ld_mode: str = "baseline",
    rollout_semantic: str = "full",
    fill_from: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """取 ``_load_ckpt_model`` 最近一次成功加载的摘要（未加载过 → None）。"""
    return _CKPT_LOAD_SUMMARY.get(_model_cache_key(
        ckpt, device, path_head=path_head, ld_mode=ld_mode,
        rollout_semantic=rollout_semantic, fill_from=fill_from,
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
    """
    path_head = str(path_head or "off")
    ld_mode = str(ld_mode or "baseline")
    rollout_semantic = str(rollout_semantic or "full")
    if path_head not in LOAD_PATH_HEAD_CHOICES:
        raise ValueError(f"未知 path_head={path_head!r}（可选 {'/'.join(LOAD_PATH_HEAD_CHOICES)}）")
    if ld_mode not in LOAD_LD_MODES:
        raise ValueError(f"未知 ld_mode={ld_mode!r}（可选 {'/'.join(LOAD_LD_MODES)}）")
    if rollout_semantic not in LOAD_ROLLOUT_SEMANTIC_MODES:
        raise ValueError(
            f"未知 rollout_semantic={rollout_semantic!r}（可选 {'/'.join(LOAD_ROLLOUT_SEMANTIC_MODES)}）"
        )
    cache_key = _model_cache_key(
        ckpt, device, path_head=path_head, ld_mode=ld_mode,
        rollout_semantic=rollout_semantic, fill_from=fill_from,
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
    rollout_semantic, fill_from, lane_anchor}`` —— 透传 :func:`_load_ckpt_model` 与注入器；
    **缺省（无该键，如 a3 controller 自己的 canary 预填路径）= 全默认装配**，缓存键仍为
    ``(ckpt, device)``（旧行为逐位不变）。
    """

    kind = "ckpt"
    speed_limit_units = "mps"
    fallback_speed_limit_mps = DEFAULT_SPEED_LIMIT_FALLBACK
    decision_interval = 5  # 0.5 s / 0.1 s
    #: R39-A1b：注入器/模式的类级默认（``object.__new__`` 构造的测试实例亦有默认值）
    anchor_injector: Any = None
    lane_anchor_mode: str = "off"

    def __init__(self, env: Any, spec: Any, task: Mapping[str, Any]):
        import torch

        self.spec = spec
        self.device = torch.device(str(task.get("device") or "cpu"))
        assembly = dict(task.get("model_assembly") or {})
        self.model = _load_ckpt_model(
            str(task["ckpt"]),
            task.get("model_config") or {},
            str(self.device),
            allow_partial=bool(task.get("allow_partial_load")),
            path_head=str(assembly.get("path_head") or "off"),
            ld_mode=str(assembly.get("ld_mode") or "baseline"),
            rollout_semantic=str(assembly.get("rollout_semantic") or "full"),
            fill_from=(str(assembly["fill_from"]) if assembly.get("fill_from") else None),
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

    def bind(self, env: Any) -> None:
        from env.obs.builder import ObservationBuilder

        self.builder = ObservationBuilder(self.obs_config)
        self.builder.reset()
        self._steps = 0
        self._action = [0.0, 0.0]
        self._plan_action = np.zeros(2, dtype=np.float64)
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
        if self.anchor_injector is not None and self._steps % self.decision_interval == 0:
            # R39-A1b：lane_anchor_xy 注入（默认 off ⇒ 不触达；只在决策步写 obs，与 a3 同点
            # 同口径 —— builder.build 之后、_tensors 之前；口径见 round36/c_lane_anchor.py）
            self.anchor_injector.apply(env, obs)
        if self._steps % self.decision_interval == 0:
            # 参考口径（--eval-reference，默认 plan=现状）：
            #   plan：需要 6 步规划预览（跟踪器参考：首步执行动作 + 后 5 步 WM/plan 预览）；
            #   repeat_action：参考 = repeat(mu, 6)（训练 A-hold 同构）→ cheap path
            #   （forward_mode="policy_single_step"，不消费 plan）。
            need_plan = self.eval_reference == "plan"
            with torch.no_grad():
                output = self.model(
                    self._tensors(obs),
                    forward_mode=("policy_wm_rollout" if need_plan else "policy_single_step"),
                )
            mu = np.asarray(output["action_mu"].detach().cpu(), dtype=np.float64).reshape(-1)
            plan = None
            if need_plan:
                plan = np.asarray(output["plan"].detach().cpu(), dtype=np.float64).reshape(-1, 2)
            references = build_eval_references(mu[:2], plan, self.eval_reference)
            # 实际下发动作（供 action_ds/action_dtheta 指标；见 action_info）
            self._plan_action = np.array(references[0], dtype=np.float64, copy=True)
            if self.tracker_kind == "exact":
                # 阶段 A/B：把 6 步参考插值成 30 点参考，逐步 apply（env.step 之后）
                self.tracker.arm(env, actions=references)
            else:
                # 阶段 C：set_reference((N,2) 动作序列)，跟踪器内部插值成 30 点
                self.tracker.set_reference(references)
        self._steps += 1
        # engine 已注册策略，传回的占位动作会被忽略（engine/base_engine.py:98-104）
        return self._action

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
    model_assembly: Dict[str, Any] = {
        "path_head": str(args.path_head),
        "ld_mode": str(args.ld_mode),
        "rollout_semantic": str(args.rollout_semantic),
        "lane_anchor": str(args.lane_anchor),
        "fill_from": (str(args.fill_from) if args.fill_from is not None else None),
    }

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
        )
        ckpt_load_summary = _ckpt_load_summary(
            str(args.ckpt), "cpu",
            path_head=model_assembly["path_head"],
            ld_mode=model_assembly["ld_mode"],
            rollout_semantic=model_assembly["rollout_semantic"],
            fill_from=model_assembly["fill_from"],
        )

    # ---- R39-A1：四 hash 原生记录（protocol / model tree / schema / checkpoint）→ run_meta ----
    eval_hashes = _evaluation_hashes(
        spec_path=spec_path, policy=str(args.policy),
        ckpt=str(args.ckpt) if args.ckpt is not None else None,
        tracker=str(args.tracker), eval_reference=str(args.eval_reference),
        moe_off=bool(getattr(args, "moe_off", False)), path_head=model_assembly["path_head"],
        ld_mode=model_assembly["ld_mode"], rollout_semantic=model_assembly["rollout_semantic"],
        lane_anchor=model_assembly["lane_anchor"], fill_from=model_assembly["fill_from"],
        workers=int(workers), max_steps=int(args.max_steps), device=device,
    )
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
    _record_run_meta(out_dir, run_meta)  # manifest 已含标记 ⇒ 幂等只更新 run_meta.json
    print(f"[eval_runner] run_meta -> {run_meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
