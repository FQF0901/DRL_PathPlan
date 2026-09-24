#!/usr/bin/env python3
"""规则基线（``PurePursuitIDMPolicy``）批量评测 CLI。

用途
----
在给定 scenario-spec 文件上批量运行规则基线专家（``env.expert.pure_pursuit_idm``），
按 P1a eval 协议口径统计 route_completion / 碰撞率 / 出界率 / 舒适性（a_lon / a_lat
均值与 p95）/ 速度比，输出“逐类别（difficulty、geometry）+ 总体 KPI + 逐 episode
明细”的 JSON，供验收（easy ≥95%、整体 ≥90%）与后续对比使用。

用法
----
    tools/venv-python tools/baseline_eval.py --specs env/specs/specs.json --workers 8 \\
        --out runs/baseline_eval/specs.json
    # 单进程调试：--workers 1 --render

契约（P1a §2，其他实现线提供，本脚本只按接口调用）
------------------------------------------------
- ``env.scenario.spec.load_specs(path) -> list[ScenarioSpec]``
- ``env.metadrive_env.build_env(spec, traffic_density=None, use_render=False) -> MetaDriveEnv``
- 动作口径：``policy.act() -> [steer, throttle]`` ∈ Box(-1, 1, (2,))。

为什么把策略注册进 engine 而不是在脚本里自己调 ``act()``：官方 agent 调用路径
（``manager/agent_manager.py:170-193``）在 ``before_step`` 阶段调用
``policy.act(agent_id)`` 并把返回值当作自车动作，与训练/评测时的控制时序一致；
因此这里用 ``engine.add_policy(ego.id, ...)``（``engine/base_engine.py:94-99``）注册，
随后给 ``env.step`` 传占位动作（注册后该动作会被忽略）。

KPI 口径（同时写入输出 JSON 的 ``meta.kpi_definitions``）
--------------------------------------------------------
- success：episode 结束时 ``info["arrive_dest"]``（MetaDrive 到达判定，
  ``envs/metadrive_env.py:218-231``）。
- collision / off_road：episode 内 ``info["crash"]`` / ``info["out_of_road"]`` 曾为 True。
- route_completion：最后一步 ``info["route_completion"]``（导航
  ``travelled_length / total_length``，``node_network_navigation.py:366-368``）。
- a_lon / a_lat：相邻两步状态差分，``dt = physics_world_step_size * decision_repeat``
  （默认 0.02 s × 5 = 0.1 s，``envs/base_env.py:188-190``）：
  ``a_lon = (v_t - v_{t-1}) / dt``，``a_lat = 0.5 (v_t + v_{t-1}) Δθ / dt``
  （Δθ 回绕到 (-π, π]）。**不**用 ``info["acceleration"]``——它是油门动作值
  （``component/vehicle/base_vehicle.py:247``）而非物理加速度。
- speed_ratio：逐步 ``v / 车道限速(m/s)``；限速由 ``lane_speed_limit_mps`` 折算
  （项目口径为 m/s，见 ``env/metadrive_env.py``；MetaDrive 上游单位不统一，故该函数
  支持 ``--speed-limit-units`` 显式切换），未设置（默认 1000，``abs_lane.py:22``）
  时用策略的 ``fallback_speed_limit_mps``。
- mean 为池化样本均值，p95 为池化样本 ``|x|`` 的 95 分位（np.percentile, linear）。
- 分组：``by_difficulty`` 按 ``spec.difficulty``；``by_geometry`` 按 ``spec.geometry`` 的
  每个标签（一条 spec 可计入多组，故各组成员数之和可大于总体 n）。
- 评测失败的 spec 记 ``termination="error"``：计入分母（视为失败）但不贡献任何
  成功/碰撞/出界分子，并在 ``n_error`` 中单列（避免基础设施问题被默默吞掉）。

import 时只做 ``sys.path`` 修正（支持直接以脚本路径运行）与函数定义，不建 env、不做 IO。
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 允许 `python tools/baseline_eval.py` 直接运行（此时 sys.path[0] 是 tools/）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.expert.pure_pursuit_idm import PurePursuitIDMPolicy, lane_speed_limit_mps  # noqa: E402
from env.metadrive_env import build_env  # noqa: E402
from env.scenario.spec import load_specs  # noqa: E402
from metadrive.utils.math import wrap_to_pi  # noqa: E402

__all__ = ["main"]

DEFAULT_MAX_STEPS = 1000  # 100 s @ 0.1 s/step

KPI_DEFINITIONS: Dict[str, str] = {
    "success":
        "episode 结束时 info['arrive_dest'] 为 True（到达判定见 metadrive_env.py:218-231）",
    "collision":
        "episode 内 info['crash'] 曾为 True（= crash_vehicle/object/building/human/sidewalk 的或）",
    "off_road":
        "episode 内 info['out_of_road'] 曾为 True",
    "route_completion":
        "episode 最后一步 info['route_completion']；成功但未到达/失败均按实际值计入",
    "a_lon/a_lat":
        "相邻步差分：a_lon=(v_t-v_{t-1})/dt，a_lat=0.5(v_t+v_{t-1})*Δθ/dt，"
        "dt=physics_world_step_size*decision_repeat（默认 0.1 s），Δθ 回绕到 (-π, π]",
    "speed_ratio":
        "逐步 v/车道限速(m/s)；限速由 lane_speed_limit_mps(ego.lane, units=策略同口径) "
        "折算（项目口径 m/s；MetaDrive 上游单位不统一，见该函数 docstring），"
        "未设置(>=1000)时用策略 fallback_speed_limit_mps",
    "mean/p95":
        "mean 为该组全部 episode 的池化样本均值；p95 为池化样本 |x| 的 95 分位",
    "grouping":
        "by_difficulty 按 spec.difficulty；by_geometry 按 spec.geometry 的每个标签"
        "（一条 spec 可计入多组，故各组成员数之和可大于总体 n）",
    "error":
        "该 spec 实例化/推进抛异常：success/collision/off_road 均记 False，样本不参与 "
        "a_lon/a_lat/speed_ratio 统计，n_error 单列",
}


# --------------------------------------------------------------------------- #
# 单 episode 评测（worker 进程内执行）
# --------------------------------------------------------------------------- #

def _step_dt(env: Any) -> float:
    """环境单步物理时长（s）：MetaDrive 默认 0.02 s × 5 步 = 0.1 s（base_env.py:188-190）。"""
    config = getattr(env, "config", None)
    if config is None:
        return 0.1
    try:
        return float(config["physics_world_step_size"]) * float(config["decision_repeat"])
    except (KeyError, TypeError, ValueError):
        return 0.1


def _unwrap_env(env: Any) -> Any:
    """兼容可选的 MetaDriveWrapper（P1a §2）：外层无 engine 而内层有则取内层。"""
    if hasattr(env, "engine"):
        return env
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "engine"):
        return inner
    return env


def _p95_abs(values: Sequence[float]) -> float:
    """样本人群的 95 分位（对 |x| 取分位；NaN/Inf 先剔除）。"""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    return float(np.percentile(np.abs(array), 95.0))


def _finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def _lane_limit_mps(ego: Any, policy: PurePursuitIDMPolicy) -> float:
    """车道限速（m/s），口径与策略一致（含 MetaDrive 未设置时的兜底）。"""
    return lane_speed_limit_mps(
        getattr(ego, "lane", None),
        units=str(getattr(policy, "speed_limit_units", "mps")),
        fallback_mps=float(getattr(policy, "fallback_speed_limit_mps", 13.9)),
    )


def _run_episode(
    env: Any,
    spec: Any,
    *,
    max_steps: int,
    policy_params: Dict[str, Any],
    policy_seed: int,
    debug: bool = False,
) -> Dict[str, Any]:
    """跑完一个 spec 的单个 episode，返回逐 episode KPI（含池化所需的原始样本）。

    ``debug=True`` 时逐步打印 ``[baseline_debug]`` 轨迹（车道链/前视点/转向/碰撞），
    仅用于定位失败场景（调用方应强制单进程，避免输出交错）。
    """
    reset_out = env.reset()
    reset_info: Dict[str, Any] = {}
    if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        reset_info = reset_out[1]
    ego = env.agent

    # 注册基线策略为 ego 的动作来源（官方 agent 调用路径，见模块 docstring）
    policy = env.engine.add_policy(ego.id, PurePursuitIDMPolicy, ego, policy_seed, **policy_params)
    policy.reset()

    dt = _step_dt(env)
    a_lon_samples: List[float] = []
    a_lat_samples: List[float] = []
    speed_samples: List[float] = []
    speed_ratio_samples: List[float] = []
    steer_abs_samples: List[float] = []
    throttle_samples: List[float] = []
    lead_gap_samples: List[float] = []

    prev_speed = float(ego.speed)
    prev_theta = float(ego.heading_theta)
    info: Dict[str, Any] = dict(reset_info)
    route_completion = float(info.get("route_completion", 0.0))
    max_route_completion = route_completion
    steps = 0
    success = collision = off_road = False
    terminated = truncated = False
    started_at = time.time()

    for step_index in range(max_steps):
        _, _, terminated, truncated, info = env.step([0.0, 0.0])
        steps = step_index + 1

        speed = float(info.get("velocity", float(ego.speed)))
        theta = float(ego.heading_theta)
        delta_theta = float(wrap_to_pi(theta - prev_theta))
        a_lon_samples.append((speed - prev_speed) / dt)
        a_lat_samples.append(0.5 * (speed + prev_speed) * delta_theta / dt)
        speed_samples.append(speed)
        prev_speed, prev_theta = speed, theta

        route_completion = float(info.get("route_completion", route_completion))
        max_route_completion = max(max_route_completion, route_completion)
        success = success or bool(info.get("arrive_dest", False))
        collision = collision or bool(info.get("crash", False))
        off_road = off_road or bool(info.get("out_of_road", False))

        limit_mps = _lane_limit_mps(ego, policy)
        if limit_mps > 0.0:
            speed_ratio_samples.append(speed / limit_mps)

        action = policy.action_info.get("action", [0.0, 0.0])
        steer_abs_samples.append(abs(float(action[0])))
        throttle_samples.append(float(action[1]))
        lead_gap = float(policy.action_info.get("pp_lead_gap_m", -1.0))
        if lead_gap >= 0.0:
            lead_gap_samples.append(lead_gap)

        if debug:
            _print_debug_step(env, ego, policy, spec, step_index + 1, speed, info, lead_gap)

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

    a_lon = np.asarray(a_lon_samples, dtype=np.float64)
    a_lat = np.asarray(a_lat_samples, dtype=np.float64)
    speed_ratio = np.asarray(speed_ratio_samples, dtype=np.float64)
    return {
        "id": int(getattr(spec, "id", -1)),
        "seed": int(getattr(spec, "seed", -1)),
        "split": str(getattr(spec, "split", "unknown")),
        "difficulty": str(getattr(spec, "difficulty", "unknown")),
        "geometry": [str(label) for label in (getattr(spec, "geometry", None) or [])],
        "success": bool(success),
        "collision": bool(collision),
        "off_road": bool(off_road),
        "termination": termination,
        "route_completion": float(route_completion),
        "max_route_completion": float(max_route_completion),
        "steps": int(steps),
        "duration_s": float(time.time() - started_at),
        "mean_speed_mps": _finite_mean(speed_samples),
        "a_lon_mean": _finite_mean(a_lon),
        "a_lon_abs_p95": _p95_abs(a_lon),
        "a_lat_mean": _finite_mean(a_lat),
        "a_lat_abs_p95": _p95_abs(a_lat),
        "speed_ratio_mean": _finite_mean(speed_ratio),
        "speed_ratio_p95": float(np.nanpercentile(speed_ratio, 95.0)) if speed_ratio.size else float("nan"),
        "steer_abs_mean": _finite_mean(steer_abs_samples),
        "throttle_mean": _finite_mean(throttle_samples),
        "lead_gap_mean_m": float(np.mean(lead_gap_samples)) if lead_gap_samples else -1.0,
        # 池化样本（聚合后由 _public_episode 剥离，不进 JSON）
        "_a_lon": a_lon_samples,
        "_a_lat": a_lat_samples,
        "_speed_ratio": speed_ratio_samples,
        "_policy_params": policy.params(),
    }


def _debug_colliders(env: Any) -> List[str]:
    """碰撞时的物体快照（带 ``crash_*`` 标志的对象：类型 / id / 位置 / 速度）。"""
    try:
        objects = env.engine.get_objects()
    except Exception:  # noqa: BLE001 - 调试输出不应影响评测
        return []
    snapshots: List[str] = []
    for obj in objects.values():
        if not any(
            bool(getattr(obj, flag, False))
            for flag in ("crash_vehicle", "crash_object", "crash_building", "crash_human")
        ):
            continue
        try:
            position = "[{:.1f}, {:.1f}]".format(float(obj.position[0]), float(obj.position[1]))
        except Exception:  # noqa: BLE001
            position = "?"
        try:
            velocity = "[{:.1f}, {:.1f}]".format(float(obj.velocity[0]), float(obj.velocity[1]))
        except Exception:  # noqa: BLE001
            velocity = "?"
        snapshots.append(
            f"{type(obj).__name__}(id={getattr(obj, 'id', '?')}, pos={position}, vel={velocity})"
        )
    return snapshots


def _debug_value(value: Any, fmt: str = "{}") -> str:
    """调试打印的 None 安全格式化。"""
    if value is None:
        return "None"
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


def _print_debug_step(
    env: Any, ego: Any, policy: PurePursuitIDMPolicy, spec: Any, step: int, speed: float,
    info: Dict[str, Any], lead_gap: float,
) -> None:
    """逐步打印 ``[baseline_debug]`` 轨迹：车道链 / 前视点 / 转向 / 限速 / 碰撞标志。"""
    ai = policy.action_info
    action = ai.get("action", [0.0, 0.0])
    flags = ",".join(
        name for name in (
            "crash_vehicle", "crash_object", "crash_building", "crash_human", "out_of_road", "arrive_dest"
        ) if info.get(name)
    ) or "-"
    print(
        f"[baseline_debug] spec={getattr(spec, 'id', '?')} step={step} v={speed:.2f} "
        f"pos=({float(ego.position[0]):.1f},{float(ego.position[1]):.1f}) "
        f"ego_lane={ai.get('pp_ego_lane')} anchor={ai.get('pp_anchor_lane')} "
        f"lat={_debug_value(ai.get('pp_anchor_lat'), '{:+.2f}')} "
        f"latL={_debug_value(ai.get('pp_lat_left'), '{:+.2f}')} "
        f"target={ai.get('pp_target_lane')} merge={ai.get('pp_merge_lane')} alt={ai.get('pp_alternate_lane')} "
        f"rem={ai.get('pp_remaining')} plan={ai.get('pp_plan_dist')} gap={ai.get('pp_connect_gap')} "
        f"refs={ai.get('pp_ref_lanes')} nxt={ai.get('pp_next_ref_lanes')} "
        f"chain={ai.get('pp_chain')} succ={ai.get('pp_succ_source')}{ai.get('pp_succ_candidates')} "
        f"Ld={_debug_value(ai.get('pp_lookahead_m'), '{:.2f}')} pt={ai.get('pp_lookahead_xy')} "
        f"steer={float(action[0]):+.3f} thr={float(action[1]):+.3f} "
        f"vtgt_kmh={_debug_value(ai.get('pp_target_speed_kmh'), '{:.1f}')} "
        f"gap_lead={lead_gap:.1f} blocker={ai.get('pp_blocker_gap')} "
        f"rc={_debug_value(info.get('route_completion'), '{:.3f}')} flags={flags}",
        flush=True,
    )
    if flags != "-":
        for snapshot in _debug_colliders(env):
            print(f"[baseline_debug]   collider: {snapshot}", flush=True)


def _error_episode(spec: Any, exc: BaseException) -> Dict[str, Any]:
    """单条 spec 失败时的占位结果：计为失败但单列 n_error，不终止整批评测。"""
    return {
        "id": int(getattr(spec, "id", -1)),
        "seed": int(getattr(spec, "seed", -1)),
        "split": str(getattr(spec, "split", "unknown")),
        "difficulty": str(getattr(spec, "difficulty", "unknown")),
        "geometry": [str(label) for label in (getattr(spec, "geometry", None) or [])],
        "success": False,
        "collision": False,
        "off_road": False,
        "termination": "error",
        "error": f"{type(exc).__name__}: {exc}",
        "error_traceback": traceback.format_exc(),
        "route_completion": 0.0,
        "max_route_completion": 0.0,
        "steps": 0,
        "duration_s": 0.0,
        "mean_speed_mps": float("nan"),
        "a_lon_mean": float("nan"),
        "a_lon_abs_p95": float("nan"),
        "a_lat_mean": float("nan"),
        "a_lat_abs_p95": float("nan"),
        "speed_ratio_mean": float("nan"),
        "speed_ratio_p95": float("nan"),
        "steer_abs_mean": float("nan"),
        "throttle_mean": float("nan"),
        "lead_gap_mean_m": -1.0,
        "_a_lon": [],
        "_a_lat": [],
        "_speed_ratio": [],
    }


def _evaluate_spec(task: Dict[str, Any]) -> Dict[str, Any]:
    """构建环境并评测单个 spec（顶层函数，供 spawn 多进程 pickle）。"""
    spec = task["spec"]
    env = None
    try:
        env = _unwrap_env(
            build_env(spec, traffic_density=task["traffic_density"], use_render=task["use_render"])
        )
        return _run_episode(
            env,
            spec,
            max_steps=int(task["max_steps"]),
            policy_params=dict(task["policy_params"] or {}),
            policy_seed=int(task["policy_seed"]),
            debug=bool(task.get("debug", False)),
        )
    except BaseException as exc:  # noqa: BLE001 - 单条场景失败不应终止整批评测
        return _error_episode(spec, exc)
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass


# --------------------------------------------------------------------------- #
# 聚合与报告
# --------------------------------------------------------------------------- #

def _pool_samples(episodes: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    arrays = [np.asarray(ep[key], dtype=np.float64) for ep in episodes if ep.get(key)]
    if not arrays:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(arrays)


def _summarize(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """把一组 episode 聚合成 KPI（成功/碰撞/出界率、rc、舒适性、速度比、终止原因）。"""
    if not episodes:
        return {"n": 0}
    success = np.asarray([1.0 if ep["success"] else 0.0 for ep in episodes], dtype=np.float64)
    collision = np.asarray([1.0 if ep["collision"] else 0.0 for ep in episodes], dtype=np.float64)
    off_road = np.asarray([1.0 if ep["off_road"] else 0.0 for ep in episodes], dtype=np.float64)
    route_completion = np.asarray([ep["route_completion"] for ep in episodes], dtype=np.float64)
    a_lon = _pool_samples(episodes, "_a_lon")
    a_lat = _pool_samples(episodes, "_a_lat")
    speed_ratio = _pool_samples(episodes, "_speed_ratio")

    terminations: Dict[str, int] = {}
    for ep in episodes:
        key = str(ep["termination"])
        terminations[key] = terminations.get(key, 0) + 1

    return {
        "n": len(episodes),
        "n_error": int(terminations.get("error", 0)),
        "success_rate": float(success.mean()),
        "collision_rate": float(collision.mean()),
        "off_road_rate": float(off_road.mean()),
        "route_completion_mean": float(route_completion.mean()),
        "route_completion_p10": float(np.percentile(route_completion, 10.0)),
        "a_lon_mean": _finite_mean(a_lon),
        "a_lon_abs_p95": _p95_abs(a_lon),
        "a_lat_mean": _finite_mean(a_lat),
        "a_lat_abs_p95": _p95_abs(a_lat),
        "speed_ratio_mean": _finite_mean(speed_ratio),
        "mean_speed_mps": _finite_mean([ep["mean_speed_mps"] for ep in episodes]),
        "mean_steps": _finite_mean([ep["steps"] for ep in episodes]),
        "mean_duration_s": _finite_mean([ep["duration_s"] for ep in episodes]),
        "terminations": dict(sorted(terminations.items())),
    }


def _public_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    """剥离下划线开头的私有池化样本字段。"""
    return {key: value for key, value in episode.items() if not key.startswith("_")}


def _build_report(episodes: Sequence[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    """生成最终报告：总体 + 按 difficulty / geometry 分组 + 逐 episode 明细。"""
    by_difficulty: Dict[str, List[Dict[str, Any]]] = {}
    by_geometry: Dict[str, List[Dict[str, Any]]] = {}
    for episode in episodes:
        by_difficulty.setdefault(str(episode.get("difficulty") or "unknown"), []).append(episode)
        for label in (episode.get("geometry") or ["unlabeled"]):
            by_geometry.setdefault(str(label), []).append(episode)

    # 注意：先算完所有分组（需要私有样本字段），再生成剥离后的 episode 明细
    report: Dict[str, Any] = {
        "meta": meta,
        "overall": _summarize(episodes),
        "by_difficulty": {key: _summarize(group) for key, group in sorted(by_difficulty.items())},
        "by_geometry": {key: _summarize(group) for key, group in sorted(by_geometry.items())},
    }
    report["episodes"] = [_public_episode(ep) for ep in sorted(episodes, key=lambda ep: ep["id"])]
    return report


def _sanitize(obj: Any) -> Any:
    """把 NaN/Inf 转为 null，numpy 标量转为 Python 标量，保证 JSON 合法。"""
    if isinstance(obj, dict):
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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="baseline_eval.py",
        description="PurePursuitIDMPolicy 规则基线批量评测（P1a KPI 基线）",
    )
    parser.add_argument("--specs", required=True, help="scenario-spec JSON 路径（load_specs 可读）")
    parser.add_argument(
        "--out", default=None, help="输出 JSON 路径（默认 runs/baseline_eval/<specs 文件名>.json）"
    )
    parser.add_argument("--workers", type=int, default=4, help="并行进程数（默认 4；--render 时强制为 1）")
    parser.add_argument("--limit", type=int, default=None, help="按文件顺序只评测前 N 条（split 过滤后）")
    parser.add_argument(
        "--split", choices=("all", "train", "val"), default="all", help="按 spec.split 过滤（默认 all）"
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS, help=f"单 episode 最大步数（默认 {DEFAULT_MAX_STEPS}）"
    )
    parser.add_argument("--traffic-density", type=float, default=None, help="覆盖 build_env 的 traffic_density")
    parser.add_argument(
        "--policy-params", default="{}",
        help='策略参数覆盖（JSON 对象字符串），如 \'{"steer_gain": 1.2, "lookahead_time": 1.2}\'',
    )
    parser.add_argument(
        "--speed-limit-units", choices=("mps", "auto", "kmh"), default=None,
        help="车道 speed_limit 单位口径（默认随策略 mps=项目口径；MetaDrive 上游单位不统一，见 lane_speed_limit_mps）",
    )
    parser.add_argument("--render", action="store_true", help="打开渲染（调试用；强制单进程）")
    parser.add_argument(
        "--debug", action="store_true",
        help="逐步打印 [baseline_debug] 轨迹（车道链/前视点/碰撞；强制单进程，配合 --debug-ids）",
    )
    parser.add_argument(
        "--debug-ids", default="",
        help="仅对逗号分隔的 spec id 打印 --debug 轨迹（默认全部）；未给 --debug 时忽略",
    )
    return parser.parse_args(argv)


def _validate_policy_params(params: Dict[str, Any]) -> Optional[str]:
    """提前校验策略参数名，避免每条 episode 都到 worker 里才报 TypeError。"""
    signature = inspect.signature(PurePursuitIDMPolicy.__init__)
    valid = {
        name for name, parameter in signature.parameters.items()
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY
    }
    unknown = sorted(set(params) - valid)
    if unknown:
        return f"未知策略参数 {unknown}；可用：{sorted(valid)}"
    return None


def _make_task(spec: Any, args: argparse.Namespace, policy_params: Dict[str, Any], debug: bool) -> Dict[str, Any]:
    return {
        "spec": spec,
        "max_steps": int(args.max_steps),
        "traffic_density": args.traffic_density,
        "use_render": bool(args.render),
        "policy_params": dict(policy_params),
        "policy_seed": int(getattr(spec, "seed", 0)),
        "debug": bool(debug),
    }


def _run_tasks(tasks: Sequence[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    """串行或 spawn 多进程执行全部任务；逐条结果到达即打印进度。"""
    if workers <= 1:
        results: List[Dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            result = _evaluate_spec(task)
            results.append(result)
            _print_progress(index, len(tasks), result)
        return results

    context = mp.get_context("spawn")
    results = []
    with context.Pool(processes=workers) as pool:
        for index, result in enumerate(pool.imap_unordered(_evaluate_spec, tasks, chunksize=1), start=1):
            results.append(result)
            _print_progress(index, len(tasks), result)
    return results


def _print_progress(index: int, total: int, result: Dict[str, Any]) -> None:
    route_completion = result.get("route_completion")
    rc_text = f"{route_completion:.3f}" if isinstance(route_completion, (int, float)) else "nan"
    print(
        f"[baseline_eval] {index}/{total} spec={result.get('id')} "
        f"success={result.get('success')} rc={rc_text} reason={result.get('termination')}",
        flush=True,
    )


def _print_summary(report: Dict[str, Any]) -> None:
    header = (
        f"{'group':<26}{'n':>4}{'succ':>8}{'coll':>8}{'offrd':>8}"
        f"{'rc':>8}{'a_lon95':>9}{'a_lat95':>9}{'spd':>7}"
    )
    print("[baseline_eval] " + header)
    for name, group in [("overall", report["overall"])] + [
        (f"difficulty/{key}", value) for key, value in report["by_difficulty"].items()
    ] + [
        (f"geometry/{key}", value) for key, value in report["by_geometry"].items()
    ]:
        if not group.get("n"):
            continue
        print(
            f"[baseline_eval] {name:<26}{group['n']:>4}{group['success_rate']:>8.3f}"
            f"{group['collision_rate']:>8.3f}{group['off_road_rate']:>8.3f}"
            f"{group['route_completion_mean']:>8.3f}{group['a_lon_abs_p95']:>9.3f}"
            f"{group['a_lat_abs_p95']:>9.3f}{group['speed_ratio_mean']:>7.3f}"
        )


def _default_out_path(specs_path: str) -> str:
    stem = os.path.splitext(os.path.basename(specs_path))[0]
    return os.path.join("runs", "baseline_eval", f"{stem}.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if not os.path.isfile(args.specs):
        print(f"[baseline_eval] specs 文件不存在: {args.specs}", file=sys.stderr)
        return 2
    try:
        policy_params = json.loads(args.policy_params or "{}")
    except json.JSONDecodeError as exc:
        print(f"[baseline_eval] --policy-params 不是合法 JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(policy_params, dict):
        print("[baseline_eval] --policy-params 必须是 JSON 对象", file=sys.stderr)
        return 2
    if args.speed_limit_units is not None and "speed_limit_units" not in policy_params:
        policy_params["speed_limit_units"] = args.speed_limit_units
    param_error = _validate_policy_params(policy_params)
    if param_error:
        print(f"[baseline_eval] {param_error}", file=sys.stderr)
        return 2

    all_specs = list(load_specs(args.specs))
    total = len(all_specs)
    specs = [spec for spec in all_specs if args.split == "all" or str(getattr(spec, "split", "")) == args.split]
    if args.limit is not None:
        specs = specs[: max(0, int(args.limit))]
    if not specs:
        print(
            f"[baseline_eval] 过滤后无可评测 spec（total={total}, split={args.split}, limit={args.limit}）",
            file=sys.stderr,
        )
        return 1

    workers = 1 if args.render else max(1, int(args.workers))
    debug_ids = {
        int(text) for text in str(args.debug_ids).replace(" ", "").split(",") if text
    }
    debug_all = bool(args.debug) and not debug_ids
    if args.debug:
        workers = 1  # 逐 step 轨迹：单进程保证输出顺序
    tasks = [
        _make_task(
            spec, args, policy_params,
            debug=bool(args.debug) and (debug_all or int(getattr(spec, "id", -1)) in debug_ids),
        )
        for spec in specs
    ]
    if args.debug:
        print(
            f"[baseline_eval] debug 轨迹开启：ids={sorted(debug_ids) if debug_ids else 'all'}（workers=1）",
            flush=True,
        )
    print(
        f"[baseline_eval] specs={len(specs)}/{total} workers={workers} "
        f"max_steps={args.max_steps} traffic_density={args.traffic_density}",
        flush=True,
    )

    started_at = time.time()
    results = _run_tasks(tasks, workers)
    episodes = results

    effective_params: Dict[str, Any] = {}
    for episode in episodes:
        if episode.get("_policy_params"):
            effective_params = dict(episode["_policy_params"])
            break

    meta: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "specs_path": os.path.abspath(args.specs),
        "n_specs_total": total,
        "n_evaluated": len(episodes),
        "n_error": sum(1 for ep in episodes if ep["termination"] == "error"),
        "split": args.split,
        "limit": args.limit,
        "workers": workers,
        "max_steps": int(args.max_steps),
        "traffic_density": args.traffic_density,
        "use_render": bool(args.render),
        "wall_time_s": float(time.time() - started_at),
        "policy_class": "env.expert.pure_pursuit_idm.PurePursuitIDMPolicy",
        "policy_params_override": policy_params,
        "policy_params_effective": effective_params,
        "kpi_definitions": KPI_DEFINITIONS,
    }

    report = _build_report(episodes, meta)
    out_path = args.out or _default_out_path(args.specs)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(_sanitize(report), handle, ensure_ascii=False, indent=2)

    _print_summary(report)
    print(f"[baseline_eval] report -> {out_path} (n={len(episodes)}, errors={meta['n_error']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
