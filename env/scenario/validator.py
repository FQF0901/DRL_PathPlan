"""逐条实例化校验 spec + JSON 报告（覆盖直方图 / 失败分类 / 重采样建议）。

校验流程（每条 spec 一个 env；经 ``env.metadrive_env.build_env`` 构建——L2 契约接口）
1. ``env = build_env(spec)`` -> ``reset()`` 成功且 ``info`` 非空、ego 存在；L2 的
   ``env.behavior_install_error`` 为空（behaviors.install 未抛异常）。
2. 导航可用：``ego.navigation.get_checkpoints()`` 长度 2 且 ``current_ref_lanes`` 非空。
3. 限速已设置：ego 当前 lane 与所有 ``current_ref_lanes`` 的 ``speed_limit < 1000``；
   其余 lane 未设置只记为 warning（``speed_limit_partial``）。
4. 几何标签与 block 序列一致：
   - spec 内部：``geometry`` 经 ``taxonomy.BLOCK_CHARS`` 映射出的序列 == ``spec.blocks``；
   - 地图实际：``env.current_map.blocks[1:]`` 的 ID 序列 == ``spec.blocks``
     （BIG 回溯可能改变序列 -> ``block_sequence_mismatch``）；
   - 若标注 curve 但地图无曲率 > 0.01 的车道 -> ``geometry_label_mismatch``。
5. rollout：
   - 用轻量循迹+定速控制器驱动自车（**不是**零动作）：零动作会在弯道上于事件窗口
     （``trigger_step`` 可达 110）之前出界终止，从而看不到事件是否触发；
   - rollout 长度 = ``max(rollout_steps, 最晚事件窗口结束 + 1.5 s)``（上限 ``MAX_ROLLOUT_STEPS``）；
   - 前 ``IMMEDIATE_WINDOW_STEPS`` 步内碰撞/出界 = 硬失败（``immediate_*``），之后只记 warning；
   - ``spec.traffic["events"]`` 的事件必须被安装（``event_not_spawned``）、
     在 rollout 覆盖到窗口时确实触发（``cut_in_not_fired`` / ``cut_out_not_fired``）；
     若 episode 在窗口前终止，记 warning ``event_window_not_reached`` 而不是失败
     （随机交通导致的早终止不是 spec 缺陷）；
   - 脚本车生成点与 ego 过近 -> ``spawn_overlap``。

报告结构（``write_report`` 落 JSON）
- ``meta`` / ``summary`` / ``n_failed``（顶层，供 ``cli.py`` 直接读取）/ ``failure_categories`` /
  ``warning_categories``
- ``coverage``：split / difficulty / geometry / traffic_patterns 四个直方图，每项含 total/passed/failed
- ``resample_suggestions``：失败维度的重采样建议
- ``results``：逐条明细（checks / categories / warnings / event_active_steps / elapsed_s）

并发：默认 ``ProcessPoolExecutor``（MetaDrive 每进程一个 engine）；``workers<=1`` 时在当前进程内联执行
（调用方需保证当前进程没有已开启的 MetaDrive env）。任何单条异常都会转成 ``worker_crash`` 记录，
不中断整批判定。
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from env.scenario.behaviors import (
    ego_lane,
    ego_vehicle,
    event_state,
    lane_curvature,
    lane_projection,
    map_info,
    spec_field,
)

IMMEDIATE_WINDOW_STEPS = 5  # 0.5 s（env.step = 0.1 s）：真正的"生成期"即时失败窗口
EVENT_MARGIN_STEPS = 15  # 事件窗口结束后多看 1.5 s（机动收尾 / 交接 IDM）
MAX_ROLLOUT_STEPS = 400  # 40 s 上限，控制 10k 校验成本
UNSET_SPEED_LIMIT = 1000.0  # AbstractLane 默认值（abs_lane.py:22）
SPAWN_OVERLAP_FAIL_M = 3.0
SPAWN_CLOSE_WARN_M = 6.0


# --------------------------------------------------------------------------------------
# 通用小工具
# --------------------------------------------------------------------------------------


def _add_once(bucket: list, item: str) -> None:
    if item not in bucket:
        bucket.append(item)


def _fail(result: dict, category: str) -> None:
    _add_once(result["categories"], category)


def _warn(result: dict, category: str) -> None:
    _add_once(result["warnings"], category)


def _spec_summary(spec: Any) -> dict:
    geometry = spec_field(spec, "geometry", []) or []
    traffic = spec_field(spec, "traffic", {}) or {}
    patterns = traffic.get("patterns", []) if isinstance(traffic, dict) else []
    return {
        "id": spec_field(spec, "id"),
        "seed": spec_field(spec, "seed"),
        "split": spec_field(spec, "split"),
        "difficulty": spec_field(spec, "difficulty"),
        "blocks": str(spec_field(spec, "blocks", "") or ""),
        "geometry": [str(g) for g in geometry],
        "patterns": [str(p) for p in (patterns or [])],
    }


def _traffic_patterns(spec: Any) -> list[str]:
    traffic = spec_field(spec, "traffic", {}) or {}
    if not isinstance(traffic, dict):
        return []
    return [str(p).lower() for p in (traffic.get("patterns") or ())]


def _block_chars() -> Optional[dict]:
    try:
        from env.scenario.taxonomy import BLOCK_CHARS  # L1a 交付；缺失时仅跳过标签->字符检查

        return dict(BLOCK_CHARS)
    except Exception:  # noqa: BLE001
        return None


def _unset(limit: Any) -> bool:
    try:
        return float(limit) >= UNSET_SPEED_LIMIT
    except (TypeError, ValueError):
        return True


# --------------------------------------------------------------------------------------
# 自车控制（validator 专用：只求 episode 活到事件窗口，不评估驾驶质量）
# --------------------------------------------------------------------------------------


class _LaneKeepController:
    """轻量循迹 + 定速控制器。

    横向复刻 MetaDrive 自带策略的已验证做法（``LaneChangePolicy.steering_control``，
    lange_change_policy.py:64-72：heading PID + lateral PID，增益同源）；纵向用简单 P 控制
    维持 spec.ego.spawn_velocity 量级的速度。目的：让 validator 的 rollout 不因零动作漂出
    路面而提前终止（否则 trigger_step 可达 110 的事件窗口根本观察不到）。
    """

    def __init__(self, spec: Any):
        from metadrive.component.vehicle.PID_controller import PIDController
        from metadrive.utils.math import wrap_to_pi

        self._wrap_to_pi = wrap_to_pi
        self.heading_pid = PIDController(1.7, 0.01, 3.5)
        self.lateral_pid = PIDController(0.3, .002, 0.05)
        ego = spec_field(spec, "ego", {}) or {}
        try:
            target_speed = float(ego.get("spawn_velocity", 8.3))
        except (AttributeError, TypeError, ValueError):
            target_speed = 8.3
        self.target_speed = float(np.clip(target_speed, 3.0, 13.9))

    def action(self, env: Any) -> list:
        ego = ego_vehicle(env)
        if ego is None:
            return [0.0, 0.0]
        steering = 0.0
        try:
            lane = ego_lane(env.engine, ego)
            projection = lane_projection(lane, np.asarray(ego.position, dtype=float))
            if projection is not None and lane is not None:
                long, lat = projection
                lane_heading = lane.heading_theta_at(long + 1.0)
                steering = self.heading_pid.get_result(-self._wrap_to_pi(lane_heading - ego.heading_theta))
                steering += self.lateral_pid.get_result(-lat)
            speed = float(ego.speed)
        except Exception:  # noqa: BLE001 - 控制失败退化为滑行，不中断校验
            speed = self.target_speed
        throttle = float(np.clip(0.2 * (self.target_speed - speed), -1.0, 1.0))
        return [float(np.clip(steering, -1.0, 1.0)), throttle]


# --------------------------------------------------------------------------------------
# 单项检查
# --------------------------------------------------------------------------------------


def _check_navigation(ego: Any) -> tuple[bool, dict]:
    detail: dict = {}
    try:
        navigation = getattr(ego, "navigation", None)
        if navigation is None:
            return False, {"reason": "navigation is None"}
        checkpoints = navigation.get_checkpoints()
        ref_lanes = navigation.current_ref_lanes
        route_completion = float(getattr(navigation, "route_completion", float("nan")))
        detail = {
            "n_checkpoints": len(checkpoints) if hasattr(checkpoints, "__len__") else None,
            "n_ref_lanes": len(ref_lanes) if ref_lanes else 0,
            "route_completion": route_completion if math.isfinite(route_completion) else None,
        }
        ok = bool(detail["n_checkpoints"] == 2 and detail["n_ref_lanes"] > 0)
        if not ok:
            detail["reason"] = "checkpoints/ref_lanes 不可用"
        return ok, detail
    except Exception as exc:  # noqa: BLE001
        detail["reason"] = repr(exc)
        return False, detail


def _check_speed_limits(env: Any, ego: Any) -> tuple[bool, dict, int]:
    detail: dict = {"n_lanes": 0, "n_unset": 0, "ego_lane_limit": None, "ref_limits": []}
    roadmap = getattr(getattr(env, "current_map", None), "road_network", None)
    if roadmap is not None and hasattr(roadmap, "get_all_lanes"):
        try:
            lanes = list(roadmap.get_all_lanes())
        except Exception:  # noqa: BLE001
            lanes = []
        detail["n_lanes"] = len(lanes)
        detail["n_unset"] = int(sum(1 for lane in lanes if _unset(getattr(lane, "speed_limit", None))))
    try:
        detail["ego_lane_limit"] = float(ego.lane.speed_limit)
    except Exception:  # noqa: BLE001
        detail["ego_lane_limit"] = None
    try:
        navigation = getattr(ego, "navigation", None)
        detail["ref_limits"] = [float(lane.speed_limit) for lane in (navigation.current_ref_lanes or [])]
    except Exception:  # noqa: BLE001
        detail["ref_limits"] = []
    ego_ok = detail["ego_lane_limit"] is not None and not _unset(detail["ego_lane_limit"])
    refs_ok = bool(detail["ref_limits"]) and all(not _unset(limit) for limit in detail["ref_limits"])
    ok = bool(ego_ok and refs_ok)
    return ok, detail, detail["n_unset"]


def _check_geometry(spec: Any, env: Any) -> tuple[bool, dict]:
    detail: dict = {"categories": []}
    summary = _spec_summary(spec)
    blocks = summary["blocks"]
    expected_blocks = blocks[1:] if blocks.startswith("I") else blocks
    geometry = summary["geometry"]
    chars = _block_chars()
    if chars is not None and geometry:
        expected_from_geometry = "".join(str(chars.get(label, "?")) for label in geometry)
        detail["expected_from_geometry"] = expected_from_geometry
        if expected_from_geometry != expected_blocks:
            detail["categories"].append("geometry_sequence_mismatch")

    map_ = getattr(env, "current_map", None)
    block_objects = list(getattr(map_, "blocks", []) or [])
    actual_blocks = "".join(str(getattr(block, "ID", "?")) for block in block_objects[1:])
    detail["actual_blocks"] = actual_blocks
    if expected_blocks and actual_blocks != expected_blocks:
        detail["categories"].append("block_sequence_mismatch")

    if "curve" in geometry:  # 几何标签的物理核对：地图真的存在弯曲车道
        info = map_info(env)
        if info is not None:
            has_curve = False
            roadmap = getattr(map_, "road_network", None)
            lanes = []
            if roadmap is not None and hasattr(roadmap, "get_all_lanes"):
                try:
                    lanes = list(roadmap.get_all_lanes())
                except Exception:  # noqa: BLE001
                    lanes = []
            for lane in lanes:
                try:
                    if lane_curvature(lane, float(lane.length) * 0.5) > 0.01:
                        has_curve = True
                        break
                except Exception:  # noqa: BLE001
                    continue
            detail["has_curved_lane"] = has_curve
            if not has_curve:
                detail["categories"].append("geometry_label_mismatch")
    return not detail["categories"], detail


def _event_rollout_length(env: Any, rollout_steps: int) -> int:
    """rollout 至少跑到最晚事件窗口结束 + margin（上限 MAX_ROLLOUT_STEPS）。"""
    longest = int(rollout_steps)
    try:
        events = event_state(env).get("events", {}) or {}
    except Exception:  # noqa: BLE001
        events = {}
    for event in events.values():
        end = event.get("end_step")
        if isinstance(end, int) and not isinstance(end, bool):
            longest = max(longest, int(end) + EVENT_MARGIN_STEPS)
    return int(np.clip(longest, 1, MAX_ROLLOUT_STEPS))


def _rollout(env: Any, spec: Any, rollout_steps: int) -> dict:
    """循迹 rollout：事件是否触发 + 是否立即碰撞/出界。"""
    out = {
        "checks": {},
        "categories": [],
        "warnings": [],
        "event_active_steps": {},
        "error": None,
    }
    horizon = _event_rollout_length(env, rollout_steps)
    out["checks"]["rollout_horizon"] = horizon
    active_steps: Counter = Counter()
    fired: Counter = Counter()
    degraded: set = set()
    planned: set = set()
    start_steps: list = []
    spawn_clearance: Optional[float] = None
    immediate: Optional[dict] = None
    late: set = set()
    terminated_at: Optional[int] = None
    controller = _LaneKeepController(spec)
    n_steps = 0
    for step in range(horizon):
        try:
            _, _, terminated, truncated, info = env.step(controller.action(env))
        except Exception as exc:  # noqa: BLE001
            out["error"] = repr(exc)
            break
        n_steps = step + 1
        state = event_state(env)
        for key, event in (state.get("events", {}) or {}).items():
            kind = str(event.get("type", "")).lower()
            if not kind:
                continue
            planned.add(kind)
            if event.get("active"):
                active_steps[kind] += 1
            if event.get("fired"):
                fired[kind] += 1
            if event.get("degraded"):
                degraded.add(kind)
            start = event.get("start_step")
            if isinstance(start, int) and not isinstance(start, bool):
                start_steps.append((kind, int(start)))
        if spawn_clearance is None and state.get("min_distance_to_ego") is not None:
            spawn_clearance = float(state["min_distance_to_ego"])
        info = info if isinstance(info, dict) else {}
        crash = bool(info.get("crash") or info.get("crash_vehicle") or info.get("crash_object")
                     or info.get("crash_building") or info.get("crash_sidewalk") or info.get("crash_human"))
        out_of_road = bool(info.get("out_of_road"))
        if crash or out_of_road:
            record = {"step": step, "crash": crash, "out_of_road": out_of_road}
            if step < IMMEDIATE_WINDOW_STEPS:
                immediate = record
            else:
                late.add("crash_in_rollout" if crash else "out_of_road_in_rollout")
        if terminated or truncated:
            terminated_at = step
            break

    reached: dict = {}
    for kind, start in start_steps:
        reached[kind] = reached.get(kind, False) or (n_steps >= start)
    out["checks"]["n_steps"] = n_steps
    out["checks"]["terminated_at"] = terminated_at
    out["checks"]["spawn_clearance_m"] = spawn_clearance
    out["checks"]["immediate_ok"] = immediate is None
    out["checks"]["immediate_event"] = immediate
    out["checks"]["event_planned"] = sorted(planned)
    out["checks"]["event_fired"] = {kind: bool(fired.get(kind)) for kind in sorted(planned)}
    out["checks"]["event_active_steps"] = dict(active_steps)
    out["checks"]["event_window_reached"] = reached
    out["checks"]["degraded_events"] = sorted(degraded)
    out["event_active_steps"] = dict(active_steps)

    if spawn_clearance is not None:
        if spawn_clearance < SPAWN_OVERLAP_FAIL_M:
            out["categories"].append("spawn_overlap")
        elif spawn_clearance < SPAWN_CLOSE_WARN_M:
            out["warnings"].append("spawn_too_close")
    if immediate is not None:
        out["categories"].append("immediate_crash" if immediate["crash"] else "immediate_out_of_road")
    for category in sorted(late):
        out["warnings"].append(category)
    if degraded:
        out["warnings"].append("scripted_event_degraded")
    return out


# --------------------------------------------------------------------------------------
# 单条 spec 校验
# --------------------------------------------------------------------------------------


def _validate_one(spec: Any, rollout_steps: int) -> dict:
    start_time = time.perf_counter()
    result: dict = {
        "spec": _spec_summary(spec),
        "ok": False,
        "categories": [],
        "warnings": [],
        "checks": {},
        "event_active_steps": {},
        "error": None,
        "elapsed_s": 0.0,
    }
    env = None
    try:
        try:
            from env.metadrive_env import build_env  # L2 交付；延迟导入避免循环依赖
        except Exception as exc:  # noqa: BLE001
            _fail(result, "build_env_unavailable")
            result["error"] = repr(exc)
            return result

        try:
            env = build_env(spec, use_render=False)
        except Exception as exc:  # noqa: BLE001
            _fail(result, "build_env_failed")
            result["error"] = repr(exc)
            return result

        try:
            _, info = env.reset()
        except Exception as exc:  # noqa: BLE001
            _fail(result, "reset_failed")
            result["error"] = repr(exc)
            return result

        checks = result["checks"]
        checks["reset_ok"] = isinstance(info, dict) and bool(info)
        ego = ego_vehicle(env)
        checks["ego_exists"] = ego is not None
        if not (checks["reset_ok"] and checks["ego_exists"]):
            checks["info_keys"] = sorted(str(k) for k in (info or {}).keys())[:40]
            _fail(result, "reset_incomplete")
            return result

        install_error = getattr(env, "behavior_install_error", None)
        checks["behavior_install_error"] = install_error
        if install_error:
            _fail(result, "behaviors_install_failed")

        nav_ok, nav_detail = _check_navigation(ego)
        checks["navigation_ok"] = nav_ok
        checks["navigation"] = nav_detail
        if not nav_ok:
            _fail(result, "navigation_unavailable")

        speed_ok, speed_detail, n_unset = _check_speed_limits(env, ego)
        checks["speed_limit_ok"] = speed_ok
        checks["speed_limits"] = speed_detail
        if not speed_ok:
            _fail(result, "speed_limit_not_applied")
        elif n_unset > 0:
            _warn(result, "speed_limit_partial")

        geometry_ok, geometry_detail = _check_geometry(spec, env)
        checks["geometry_ok"] = geometry_ok
        checks["geometry"] = geometry_detail
        for category in geometry_detail.get("categories", []):
            _fail(result, category)

        rollout = _rollout(env, spec, rollout_steps)
        checks.update(rollout["checks"])
        for category in rollout["categories"]:
            _fail(result, category)
        for category in rollout["warnings"]:
            _warn(result, category)
        result["event_active_steps"] = rollout["event_active_steps"]
        if rollout["error"]:
            result["error"] = rollout["error"]
            _fail(result, "rollout_failed")

        planned = set(rollout["checks"].get("event_planned") or [])
        reached = rollout["checks"].get("event_window_reached") or {}
        for kind in ("cut_in", "cut_out"):
            if kind not in _traffic_patterns(spec):
                continue
            if kind not in planned:
                _fail(result, f"{kind}_not_spawned")
            elif rollout["event_active_steps"].get(kind, 0) <= 0:
                if reached.get(kind):
                    _fail(result, f"{kind}_not_fired")
                else:
                    _warn(result, "event_window_not_reached")
        return result
    except Exception as exc:  # noqa: BLE001 - 任何未预期异常都归类，不中断整批
        _fail(result, "validator_exception")
        result["error"] = repr(exc)
        return result
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        result["ok"] = bool(not result["categories"] and result["error"] is None)
        result["elapsed_s"] = round(time.perf_counter() - start_time, 4)


# --------------------------------------------------------------------------------------
# 批量校验
# --------------------------------------------------------------------------------------


def validate_specs(
    specs: Iterable[Any],
    workers: int = 8,
    rollout_steps: int = 50,
    specs_per_worker_per_pool: int = 100,
) -> dict:
    """逐条实例化校验；返回报告 dict（结构见模块 docstring）。

    ``rollout_steps`` 是 rollout 的**下界**：含脚本事件时自动延长到覆盖事件窗口
    （``_event_rollout_length``，上限 ``MAX_ROLLOUT_STEPS``）。

    **worker 回收（内存关键）**：实测 MetaDrive 每次 ``build_env`` + ``close`` 会在进程内留下
    ~3.5MB 残留（P0 的 reset 循环是干净的，泄漏只发生在"每条 spec 新建 env"的路径上），
    长跑会让 worker RSS 从 1.2GB 涨到 4GB+ 并显著拖慢吞吐。因此按"每个 worker 处理
    ``specs_per_worker_per_pool`` 条 spec"为一批，**每批重建进程池**把 RSS 拉回基线；
    每批结束打印进度（便于外部监控）。
    """
    spec_list = list(specs)
    total_start = time.perf_counter()
    n_specs = len(spec_list)
    max_workers = max(1, int(workers or 1))
    if n_specs:
        max_workers = min(max_workers, n_specs)
    results: list = [None] * n_specs
    pool_errors: list[str] = []

    if n_specs == 1 or max_workers <= 1:
        for index, spec in enumerate(spec_list):
            results[index] = _validate_one(spec, rollout_steps)
    else:
        chunk = max(1, int(specs_per_worker_per_pool) * max_workers)
        n_chunks = (n_specs + chunk - 1) // chunk
        for chunk_index, start in enumerate(range(0, n_specs, chunk), start=1):
            stop = min(start + chunk, n_specs)
            indices = list(range(start, stop))
            pending = set(indices)
            try:
                with ProcessPoolExecutor(max_workers=min(max_workers, len(indices))) as executor:
                    futures = {
                        executor.submit(_validate_one, spec_list[index], rollout_steps): index
                        for index in indices
                    }
                    for future in as_completed(futures):
                        index = futures[future]
                        pending.discard(index)
                        try:
                            results[index] = future.result()
                        except Exception as exc:  # noqa: BLE001
                            results[index] = _crash_result(spec_list[index], repr(exc))
            except Exception as exc:  # noqa: BLE001 - 进程池崩溃（如 panda 段错误）后退化为内联
                pool_errors.append(repr(exc))
                for index in sorted(pending):
                    results[index] = _validate_one(spec_list[index], rollout_steps)
            print(
                f"[validator] chunk {chunk_index}/{n_chunks}: {stop}/{n_specs} specs, "
                f"elapsed {time.perf_counter() - total_start:.0f}s",
                flush=True,
            )
        for index in range(n_specs):
            if results[index] is None:
                results[index] = _crash_result(
                    spec_list[index], "; ".join(pool_errors) or "missing result"
                )

    elapsed = time.perf_counter() - total_start
    report = _build_report(results, max_workers, rollout_steps, elapsed)
    report["meta"]["pool_chunk_specs"] = (
        max(1, int(specs_per_worker_per_pool) * max_workers) if max_workers > 1 else n_specs
    )
    if pool_errors:
        report["meta"]["pool_error"] = "; ".join(pool_errors)
    return report


def _crash_result(spec: Any, error: str) -> dict:
    return {
        "spec": _spec_summary(spec),
        "ok": False,
        "categories": ["worker_crash"],
        "warnings": [],
        "checks": {},
        "event_active_steps": {},
        "error": error,
        "elapsed_s": 0.0,
    }


def _hist_add(hist: dict, value: Any, ok: bool) -> None:
    key = "None" if value is None else str(value)
    entry = hist.setdefault(key, {"total": 0, "passed": 0, "failed": 0})
    entry["total"] += 1
    entry["passed" if ok else "failed"] += 1


def _build_report(results: list, workers: int, rollout_steps: int, elapsed: float) -> dict:
    failure_categories: Counter = Counter()
    warning_categories: Counter = Counter()
    coverage: dict = {
        "split": {},
        "difficulty": {},
        "geometry": {},
        "traffic_patterns": {},
    }
    for result in results:
        ok = bool(result.get("ok"))
        for category in result.get("categories", []):
            failure_categories[category] += 1
        for category in result.get("warnings", []):
            warning_categories[category] += 1
        summary = result.get("spec", {})
        _hist_add(coverage["split"], summary.get("split"), ok)
        _hist_add(coverage["difficulty"], summary.get("difficulty"), ok)
        for label in summary.get("geometry", []):
            _hist_add(coverage["geometry"], label, ok)
        for pattern in summary.get("patterns", []):
            _hist_add(coverage["traffic_patterns"], pattern, ok)

    n_failed = sum(1 for result in results if not result.get("ok"))
    n_passed = len(results) - n_failed
    summary_out = {
        "n_specs": len(results),
        "passed": n_passed,
        "failed": n_failed,
        "pass_rate": round(n_passed / len(results), 6) if results else 0.0,
        "immediate_events": sum(
            1 for result in results if result.get("checks", {}).get("immediate_ok") is False
        ),
    }

    suggestions = []
    for dimension, histogram in coverage.items():
        for value, stats in histogram.items():
            if stats["failed"] > 0:
                suggestions.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "failed": stats["failed"],
                        "total": stats["total"],
                        "failure_rate": round(stats["failed"] / stats["total"], 4),
                        "suggested_resample": int(math.ceil(stats["failed"] * 1.5)),
                        "reason": "该类别存在失败条目，建议重采样/复核对应几何或事件参数",
                    }
                )
    suggestions.sort(key=lambda item: (-item["failure_rate"], item["dimension"], item["value"]))

    # 非失败但影响数据质量的告警 -> 可执行建议（不进 resample 计数）
    warning_notes = []
    if warning_categories.get("scripted_event_degraded"):
        warning_notes.append(
            {
                "warning": "scripted_event_degraded",
                "count": warning_categories["scripted_event_degraded"],
                "reason": "事件请求的侧向车道不存在（自车车道随机）→ 建议生成器显式设置 "
                          "ego.spawn_lane_index 使请求侧存在（脚本车已按对侧降级执行）",
            }
        )
    if warning_categories.get("event_window_not_reached"):
        warning_notes.append(
            {
                "warning": "event_window_not_reached",
                "count": warning_categories["event_window_not_reached"],
                "reason": "episode 在事件窗口前终止（随机交通/出界）→ 复核该难度档的密度与初始车距",
            }
        )

    return {
        "meta": {
            "n_specs": len(results),
            "workers": workers,
            "rollout_steps": int(rollout_steps),
            "rollout_policy": "max(rollout_steps, 最晚事件窗口结束 + 1.5s)，上限 %d 步" % MAX_ROLLOUT_STEPS,
            "elapsed_s": round(elapsed, 3),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary_out,
        # 顶层失败计数：tools/gene_env.sh 的 cli 直接读 n_failed
        "n_failed": n_failed,
        "failure_categories": dict(sorted(failure_categories.items(), key=lambda kv: (-kv[1], kv[0]))),
        "warning_categories": dict(sorted(warning_categories.items(), key=lambda kv: (-kv[1], kv[0]))),
        "coverage": coverage,
        "resample_suggestions": suggestions,
        "warning_notes": warning_notes,
        "results": results,
    }


def write_report(report: dict, path: str) -> None:
    """写 JSON 报告（含覆盖直方图 / 失败分类 / 重采样建议）；自动创建父目录。"""
    target = Path(path)
    if str(target.parent) not in ("", "."):
        target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=str, sort_keys=False)
