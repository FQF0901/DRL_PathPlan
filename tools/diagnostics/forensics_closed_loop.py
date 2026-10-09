#!/usr/bin/env python3
"""闭环取证（只读诊断脚本；不改任何行为代码）。

在同一冻结评测协议下重跑指定 spec，并记录逐 env-step 状态 + 逐策略步 plan，
以区分"预瞄本身不可跟" vs "tracker 增益不适配" vs "策略动作错误"。

模式
----
- ``lqr``           : 完整复现评测协议（ckpt + LqrTracker），带逐 step 记录；
- ``lqr_gain``      : 同上，但用 ``--tracker-json`` 覆盖 LqrTracker 参数（增益/预瞄）；
- ``exact``         : ckpt + ExactTracker（把 plan 精确置位）→ "若 plan 被完美执行会怎样"；
- ``baseline``      : PurePursuitIDMPolicy（规则专家参照）；
- ``arc``           : 合成圆弧参考实验：每 0.5 s 以当前位姿为原点设 κ 参考
                      （``--arc-radius`` / ``--arc-speed``），测 LQR 的曲率-速度可行域；
- ``laneplan``      : 车道中心线 3 s 参考（oracle plan）；
- ``oracle``        : 专家实测未来 3 s 位姿参考（oracle 路径）；
- ``d1``            : 未来 expert 动作链（privileged ceiling，P0-B/D1）：pass 1 对同
                      (spec, seed) 跑规则专家取实测轨迹 → 按 ``tools/collect_expert._window_actions``
                      同款 pose-delta 口径生成 6 步 (ds,dθ) → 交同一 LQR（首点不做 mu 覆盖）；
- ``d2``            : 当前 expert 动作 repeat 6 步（P0-B/D2）：每策略步对现状态做专家
                      **空问**（不注册 engine，只调 ``act()``；与 ``tools/dagger_collect``
                      采集侧同口径）→ ``(ds,dθ)`` → ``np.repeat(·, 6)`` → 交同一 LQR。

新增参数
--------
- ``--reference {plan,repeat_action}``：模型行（lqr/lqr_gain/exact）跟踪器参考口径；
  缺省 ``plan`` = 现状逐位不变；``repeat_action`` = ``np.repeat(action_mu, 6)``
  （``pipeline.eval_runner.build_eval_references`` 同口径，供 P0-B/F 格）。

逐策略步记录扩展（设计文档 ``docs/p0_eval_compat_design.md`` §4/§5，字段冻结）
----------------------------------------------------------------------------
- ``plan_footprint_valid`` / ``exec_footprint_valid``：计划/执行车辆矩形是否整体可行驶；
- ``plan_footprint`` / ``exec_footprint``：退化三元组 ``inside_ratio /
  first_invalid_pose / invalid_footprint_point_count``（见 :func:`check_footprint_path`）；
- ``min_signed_margin``：**恒 None** —— 引擎的 off-road 几何查询只给布尔（射线命中
  lane 面），拿不到可靠 signed distance，按设计不伪造连续距离，改记退化三元组；
- ``tracking_residual``：本策略步 0.5 s 窗口内实际位姿 vs 跟踪器参考的同时刻最大
  xy 偏差（m；baseline 无参考 → None）；
- ``road_class`` / ``map_id``：当前 lane 几何类别（``env.obs.others.road_class_labels``）
  / 地图标识 ``"<seed>:<blocks>"``；
- ``policy_action_vs_plan_first_action`` / ``router_topk`` / ``policy_std``：模型行专用
  （专家/baseline 行 None）；
- ``T_plan / T_track / T_cross / T_term``：episode 级首次事件时间戳（每行记录冗余一份，
  见 :func:`finalize_episode`）。

footprint 几何查询出处（代码注释要求记录所用 API）
-------------------------------------------------
- 终止判定：``metadrive/envs/metadrive_env.py::MetaDriveEnv._is_out_of_road`` ——
  ``not vehicle.on_lane``（+ 连续线/人行道等 flag，取决于 config）；
- ``on_lane`` 赋值：``metadrive/component/navigation_module/node_network_navigation.py::
  _update_current_lane`` → ``metadrive/utils/pg/utils.py::ray_localization(..., return_on_lane=True)``
  —— 对车辆中心点做竖直 ``rayTestAll``，命中节点名过 ``MetaDriveType.is_lane`` 即 on_lane；
- 本工具对 footprint 采样点逐点调用**同一函数** ``ray_localization``（点级），9 点全中
  lane 面才判该位姿可行。

用法::

    tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode lqr --ids 0,5,11,14,18,19,22,23,28,29,30,32,34,40,43,44,47 --out runs/forensics/closed_lqr.json
    tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode d1 --ids 0,5 --out runs/forensics/closed_d1.json
    tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode lqr --reference repeat_action --out runs/forensics/closed_f.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo 根（本文件在 tools/<sub>/ 下，上溯 3 层）
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CKPT = "runs/train/il_v2_10x10_b_fixed/stage_b/final.pt"
SPEC = "env/specs/scenarios_val_slice50.json"


def _wrap(o: object) -> object:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [round(float(x), 5) for x in np.asarray(o, dtype=np.float64).reshape(-1).tolist()]
    return o


def _lane_block(env) -> str:
    try:
        from env.scenario.behaviors import map_info

        info = map_info(env)
        idx = getattr(env.agent, "lane_index", None)
        if info is not None and idx is not None:
            return str(info.lane_block.get(tuple(idx), "?"))
    except Exception:  # noqa: BLE001
        return "?"
    return "?"


def _lane_lat(env) -> float:
    """ego 相对当前车道中心线的横向偏差（m；正值 = 车道中心左侧）。"""
    try:
        from env.scenario.behaviors import lane_projection

        lane = getattr(env.agent, "lane", None)
        if lane is None:
            return float("nan")
        proj = lane_projection(lane, env.agent.position)
        if proj is None:
            return float("nan")
        return float(proj[1]) if not isinstance(proj, float) else float(proj)
    except Exception:  # noqa: BLE001
        return float("nan")


def _lane_curvature(env) -> float:
    """自车当前车道、当前位置的曲率（1/m，finite difference）。"""
    try:
        from env.scenario.behaviors import lane_curvature, lane_projection

        lane = getattr(env.agent, "lane", None)
        if lane is None:
            return float("nan")
        proj = lane_projection(lane, env.agent.position)
        if proj is None:
            return float("nan")
        return float(lane_curvature(lane, proj[0]))
    except Exception:  # noqa: BLE001
        return float("nan")


def _nav_checkpoints(env):
    try:
        nav = getattr(env.agent, "navigation", None)
        if nav is None:
            return []
        return [[float(p[0]), float(p[1])] for p in nav.get_checkpoints()]
    except Exception:  # noqa: BLE001
        return []


def _road_class(env):
    """自车当前 lane 的几何类别名（``env.obs.others`` 口径）；不可用 → None。"""
    try:
        from env.obs.others import road_class_index, road_class_labels

        index = road_class_index(env)
        if index is None:
            return None
        labels = road_class_labels()
        if 0 <= int(index) < len(labels):
            return str(labels[int(index)])
    except Exception:  # noqa: BLE001
        return None
    return None


def _map_id(env, spec) -> str:
    """地图标识：MetaDrive 地图由 ``(blocks, seed)`` 决定 → ``"<seed>:<blocks>"``。"""
    seed = getattr(spec, "seed", None)
    if seed is None:
        seed = getattr(getattr(env, "engine", None), "global_seed", None)
    blocks = getattr(spec, "blocks", None)
    return f"{seed}:{blocks}" if blocks is not None else str(seed)


# --------------------------------------------------------------------------- #
# 运动学：plan → 位姿（复用 net/model.py::arc_step 同口径）
# --------------------------------------------------------------------------- #

#: 策略步长 / 每策略步 env step 数 / 3 s 窗口策略步数（与 collect_expert 同口径）
POLICY_DT = 0.5
STEPS_PER_POLICY = 5
WINDOW_POLICIES = 6
#: footprint 路径重采样上限间距（m）；plan 节点间按 ≤ 该值插值（设计文档 §5：0.5–1.0 m）
FOOTPRINT_MAX_GAP_M = 1.0
#: T_track 显式阈值：0.5 s 窗口内同时刻最大 xy 偏差（m）超过即"明显偏离仍可行的计划"
TRACK_RESIDUAL_THRESHOLD_M = 0.5


def expand_plan_poses(plan, base_pose, *, max_gap_m: float = FOOTPRINT_MAX_GAP_M) -> np.ndarray:
    """``(K,2)`` 动作 plan → 世界系 ``(N,3)`` 位姿，按 ≤ ``max_gap_m`` 空间重采样。

    运动学**复用** ``net/model.py::arc_step`` + ``compose_pose``（与模型 plan 展开/跟踪器
    插值同一约定：段内常曲率圆弧）。每个 plan 段按 ``n = ceil(|ds| / max_gap_m)`` 等弧长
    细分（``arc_step(ds/n, dθ/n)`` 组合 n 次与整段严格等价），因此：
    - 每个 plan 节点（段终点）必在输出中；
    - ``ds > max_gap_m`` 时相邻采样弧长间距 ∈ (max_gap_m/2, max_gap_m]（即 0.5–1.0 m）。
    """
    import torch

    from net.model import arc_step, compose_pose

    arr = np.asarray(plan, dtype=np.float64).reshape(-1, 2)
    if arr.shape[0] == 0 or not np.all(np.isfinite(arr)):
        return np.zeros((0, 3), dtype=np.float64)
    gap = float(max_gap_m)
    if gap <= 0.0:
        raise ValueError(f"max_gap_m 必须为正，收到 {max_gap_m}")
    pose = torch.tensor(
        [[float(base_pose[0]), float(base_pose[1]), float(base_pose[2])]], dtype=torch.float64
    )
    out = [pose[0].numpy().copy()]
    for ds, dtheta in arr:
        n = max(1, int(math.ceil(abs(float(ds)) / gap - 1e-9)))
        sub_ds = torch.tensor([float(ds) / n], dtype=torch.float64)
        sub_dth = torch.tensor([float(dtheta) / n], dtype=torch.float64)
        for _ in range(n):
            dx, dy = arc_step(sub_ds, sub_dth)
            pose = compose_pose(pose, dx, dy, sub_dth)
            out.append(pose[0].numpy().copy())
    return np.asarray(out, dtype=np.float64)


def resample_poses(poses, *, max_gap_m: float = FOOTPRINT_MAX_GAP_M) -> np.ndarray:
    """已有稠密世界系位姿 ``(N,3)`` → 按 ≤ ``max_gap_m`` 弦长线性重采样（航向先 unwrap）。"""
    arr = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    if arr.shape[0] <= 1:
        return arr.copy()
    gap = float(max_gap_m)
    if gap <= 0.0:
        raise ValueError(f"max_gap_m 必须为正，收到 {max_gap_m}")
    theta = np.unwrap(arr[:, 2])
    out = [arr[0].copy()]
    for i in range(len(arr) - 1):
        length = float(np.linalg.norm(arr[i + 1, :2] - arr[i, :2]))
        n = max(1, int(math.ceil(length / gap - 1e-9)))
        for k in range(1, n + 1):
            t = k / n
            out.append(
                np.array(
                    [
                        arr[i, 0] + (arr[i + 1, 0] - arr[i, 0]) * t,
                        arr[i, 1] + (arr[i + 1, 1] - arr[i, 1]) * t,
                        theta[i] + (theta[i + 1] - theta[i]) * t,
                    ],
                    dtype=np.float64,
                )
            )
    return np.asarray(out, dtype=np.float64)


def footprint_points(pose, length: float, width: float) -> np.ndarray:
    """车辆矩形 9 点采样：四角 + 四边中点 + 中心（车体系 → 世界系，``(9,2)``）。

    车体系 x 前向 / y 左向（与 ``env/tracking.py`` 顶部约定一致）；``pose = (x,y,θ)``。
    角点顺序：左前、右前、右后、左后；边中点：前、后、左、右；最后中心。
    """
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    half_l, half_w = 0.5 * float(length), 0.5 * float(width)
    body = np.array(
        [
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
            [half_l, 0.0],
            [-half_l, 0.0],
            [0.0, half_w],
            [0.0, -half_w],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.column_stack(
        [x + body[:, 0] * cos_t - body[:, 1] * sin_t, y + body[:, 0] * sin_t + body[:, 1] * cos_t]
    )


def _make_on_lane_fn(env):
    """返回逐点 drivable 查询 ``point_xy -> bool``；与 env off-road 终止同款几何查询。

    定位（代码注释要求）：``MetaDriveEnv._is_out_of_road``（``metadrive/envs/metadrive_env.py``）
    用 ``not vehicle.on_lane``；``on_lane`` 由 ``node_network_navigation.py::_update_current_lane``
    经 ``metadrive/utils/pg/utils.py::ray_localization(..., return_on_lane=True)`` 赋值 ——
    竖直 ``rayTestAll`` + ``MetaDriveType.is_lane``。这里对每个 footprint 采样点直接调用
    同一 ``ray_localization``（点级布尔；引擎不提供 signed distance）。
    """
    try:
        from metadrive.utils.pg.utils import ray_localization

        engine = env.engine
    except Exception:  # noqa: BLE001 - 无 metadrive/engine 时跳过 footprint
        return None

    def on_lane(point_xy) -> bool:
        _, flag = ray_localization(
            (1.0, 0.0),
            (float(point_xy[0]), float(point_xy[1])),
            engine,
            return_on_lane=True,
        )
        return bool(flag)

    return on_lane


def _vehicle_dims(env):
    """车长/车宽（env 参数：``BaseVehicle.LENGTH/WIDTH``）；不可用 → (None, None)。"""
    agent = getattr(env, "agent", None)
    try:
        length = float(getattr(agent, "LENGTH"))
        width = float(getattr(agent, "WIDTH"))
    except (TypeError, ValueError):
        return None, None
    if not (length > 0.0 and width > 0.0):
        return None, None
    return length, width


def check_footprint_path(poses, length: float, width: float, on_lane_fn) -> dict:
    """逐位姿 9 点 footprint 可行性 + 退化三元组（**不伪造** signed distance）。

    ``on_lane_fn(point_xy) -> bool`` 必须是 env off-road 终止同款查询（见
    :func:`_make_on_lane_fn`）。返回：

    - ``valid``：所有位姿的 9 点全部在 lane 面；
    - ``inside_ratio``：全点可行位姿数 / 位姿数；
    - ``first_invalid_pose``：首个不可行位姿 ``{index, s_m, x, y, theta}``（无 → None）；
    - ``invalid_footprint_point_count``：失败 (位姿, 采样点) 对总数；
    - ``n_poses``；``min_signed_margin`` 恒 None（引擎查询只给布尔）。
    """
    arr = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    n = len(arr)
    result = {
        "valid": True,
        "inside_ratio": 1.0,
        "first_invalid_pose": None,
        "invalid_footprint_point_count": 0,
        "n_poses": int(n),
        "min_signed_margin": None,
    }
    if n == 0:
        return result
    s_values = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1))]
    )
    valid_poses = 0
    invalid_points = 0
    first_invalid = None
    for index, pose in enumerate(arr):
        bad = 0
        for point in footprint_points(pose, length, width):
            try:
                ok = bool(on_lane_fn((float(point[0]), float(point[1]))))
            except Exception:  # noqa: BLE001 - 查询失败按不可行计（保守）
                ok = False
            if not ok:
                bad += 1
        if bad:
            invalid_points += bad
            if first_invalid is None:
                first_invalid = {
                    "index": int(index),
                    "s_m": round(float(s_values[index]), 3),
                    "x": round(float(pose[0]), 4),
                    "y": round(float(pose[1]), 4),
                    "theta": round(float(pose[2]), 5),
                }
        else:
            valid_poses += 1
    result.update(
        {
            "valid": bool(invalid_points == 0),
            "inside_ratio": round(float(valid_poses) / float(n), 4),
            "first_invalid_pose": first_invalid,
            "invalid_footprint_point_count": int(invalid_points),
        }
    )
    return result


def attach_footprints(
    records,
    steps,
    on_lane_fn,
    length: float,
    width: float,
    *,
    max_gap_m: float = FOOTPRINT_MAX_GAP_M,
) -> None:
    """后处理：给逐 env-step 记录补 plan/exec footprint 字段（原地写 records）。

    - plan（仅决策步，``rec["plan"]`` 存在时）：``(6,2)`` 动作 → :func:`expand_plan_poses`；
      ``(N,3)`` 自车系位姿（oracle）→ 世界系后 :func:`resample_poses`；
    - exec（每步）：该 env step 执行位姿 ``(x,y,θ)`` 的单矩形检查；
    - ``min_signed_margin`` 顶层恒 None（引擎无 signed distance；三元组在 footprint dict 内）。
    """
    for rec in records:
        rec.setdefault("min_signed_margin", None)
        rec.setdefault("plan_footprint_valid", None)
        rec.setdefault("exec_footprint_valid", None)
        if not rec.get("decision"):
            continue
        plan = rec.get("plan")
        if plan is not None:
            try:
                plan_arr = np.asarray(plan, dtype=np.float64)
                if plan_arr.ndim == 2 and plan_arr.shape[1] == 2:
                    poses = expand_plan_poses(plan_arr, (rec["x"], rec["y"], rec["theta"]), max_gap_m=max_gap_m)
                elif plan_arr.ndim == 2 and plan_arr.shape[1] == 3:
                    from env.tracking import ego_to_world

                    world = ego_to_world(plan_arr, (rec["x"], rec["y"], rec["theta"]))
                    poses = resample_poses(world, max_gap_m=max_gap_m)
                else:
                    poses = None
            except Exception:  # noqa: BLE001 - 展开失败不阻塞其它记录
                poses = None
            if poses is not None:
                info = check_footprint_path(poses, length, width, on_lane_fn)
                rec["plan_footprint"] = info
                rec["plan_footprint_valid"] = bool(info["valid"])
    for index, rec in enumerate(records):
        if index >= len(steps):
            break
        row = steps[index]
        try:
            pose = (float(row["x"]), float(row["y"]), float(row["theta"]))
            info = check_footprint_path([pose], length, width, on_lane_fn)
            rec["exec_footprint"] = info
            rec["exec_footprint_valid"] = bool(info["valid"])
        except Exception:  # noqa: BLE001 - 单步检查失败不阻塞
            continue


def attach_tracking_residual(records, steps, ref_windows) -> None:
    """后处理：``tracking_residual`` = 0.5 s 窗口内实际位姿 vs 参考的同时刻最大 xy 偏差（m）。

    ``ref_windows[j]``（决策步 j 的跟踪器世界系参考前 5 个 0.1 s 点）与
    ``steps[j..j+4]`` 同时刻对齐；无参考（baseline）→ 字段保持 None。
    """
    if not ref_windows:
        return
    n = min(len(records), len(steps))
    for j in range(n):
        window = ref_windows[j] if j < len(ref_windows) else None
        if window is None:
            records[j].setdefault("tracking_residual", None)
            continue
        residual = None
        for i in range(min(len(window), STEPS_PER_POLICY)):
            k = j + i
            if k >= n:
                break
            row = steps[k]
            distance = math.hypot(float(row["x"]) - float(window[i][0]), float(row["y"]) - float(window[i][1]))
            residual = distance if residual is None else max(residual, distance)
        records[j]["tracking_residual"] = None if residual is None else round(float(residual), 4)


def _timestamp(step, reason: str):
    if step is None:
        return None
    return {"step": int(step), "t_s": round(0.1 * int(step), 3), "reason": str(reason)}


def finalize_episode(records, steps, termination, *, residual_threshold_m: float = TRACK_RESIDUAL_THRESHOLD_M) -> dict:
    """计算 episode 级时间戳（§5 定义）与失败分类（层级 + 多因素），并回填每行记录。

    - ``T_plan``  = 首次计划 footprint 不可行（决策步）；
    - ``T_track`` = 首次实际轨迹明显偏离"仍可行的计划"（计划可行 ∧ residual > 阈值）；
    - ``T_cross`` = 首次实际 footprint 越界；
    - ``T_term``  = 环境正式终止。

    分类（**同时保留全部 contributing factors**）：
    - ``plan_originated``：``T_plan < T_cross``（计划先越界）；
    - ``tracker_originated``：计划仍可行时已 tracking 偏离，且随后越界
      （``T_track < T_plan``/``T_plan=None`` ∧ ``T_track < T_cross``）；
    - ``recovery_failure``：进入危险状态（``T_cross``）后仍出现过可行计划（存在 recovery）
      而 episode 最终未成功；
    - ``anomaly``：几何判定与 termination 不一致（``out_of_road`` 但无 ``T_cross``；
      或已有 ``T_cross`` 却 ``arrive_dest``）。注意 env 的 out_of_road 还含连续线压线/
      人行道碰撞等非"可行驶面"判据，故线压线终止会落在此类（记录而非解释）。
    """
    n_steps = len(steps)
    threshold = float(residual_threshold_m)

    def first_step(predicate):
        for rec in records:
            if predicate(rec):
                return int(rec.get("step", -1))
        return None

    t_plan = first_step(lambda r: bool(r.get("decision")) and r.get("plan_footprint_valid") is False)
    t_cross = first_step(lambda r: r.get("exec_footprint_valid") is False)

    def _track(r) -> bool:
        if not r.get("decision") or r.get("plan_footprint_valid") is not True:
            return False
        value = r.get("tracking_residual")
        return isinstance(value, (int, float)) and value == value and float(value) > threshold

    t_track = first_step(_track)
    t_term = n_steps if n_steps > 0 else None

    timestamps = {
        "T_plan": _timestamp(t_plan, "plan_footprint_invalid"),
        "T_track": _timestamp(t_track, f"tracking_residual>{threshold:g}m"),
        "T_cross": _timestamp(t_cross, "exec_footprint_invalid"),
        "T_term": _timestamp(t_term, str(termination)),
    }
    term = str(termination)
    plan_originated = t_plan is not None and (t_cross is None or t_plan < t_cross)
    tracker_originated = (
        t_track is not None
        and (t_plan is None or t_track < t_plan)
        and t_cross is not None
        and t_track < t_cross
    )
    recovery_failure = t_cross is not None and term != "arrive_dest" and any(
        bool(r.get("decision"))
        and r.get("plan_footprint_valid") is True
        and int(r.get("step", -1)) > t_cross
        for r in records
    )
    anomaly = (term == "out_of_road" and t_cross is None) or (t_cross is not None and term == "arrive_dest")
    factors = {
        "plan_originated": bool(plan_originated),
        "tracker_originated": bool(tracker_originated),
        "recovery_failure": bool(recovery_failure),
        "anomaly": bool(anomaly),
    }
    if anomaly:
        primary = "anomaly"
    elif plan_originated:
        primary = "plan_originated"
    elif tracker_originated:
        primary = "tracker_originated"
    elif recovery_failure:
        primary = "recovery_failure"
    else:
        primary = "none"
    for rec in records:
        rec["T_plan"] = timestamps["T_plan"]
        rec["T_track"] = timestamps["T_track"]
        rec["T_cross"] = timestamps["T_cross"]
        rec["T_term"] = timestamps["T_term"]
    return {
        "timestamps": timestamps,
        "classification": {"primary": primary, "factors": factors, "residual_threshold_m": threshold},
    }


def expert_chain_plan(expert_poses, start: int, *, n_policies: int = WINDOW_POLICIES):
    """D1：专家实测轨迹 → 6 步 ``(ds,dθ)``（**复用** ``tools/collect_expert._window_actions`` 口径）。

    首点不做 mu 覆盖（expert 链首点 = 其下一步动作）。轨迹不足一个 3 s 窗口 → None。
    """
    poses = np.asarray(expert_poses, dtype=np.float64).reshape(-1, 3)
    horizon = STEPS_PER_POLICY * int(n_policies)
    last = len(poses) - 1
    if last < horizon:
        return None
    start = int(min(max(0, int(start)), last - horizon))
    from tools.collect_expert import _window_actions

    return np.asarray(_window_actions(poses, start, n_policies=int(n_policies)), dtype=np.float64)


def relocate_cursor(expert_poses, cursor: int, position) -> int:
    """把 D1 cursor 重定位到 expert 轨迹上离当前位姿最近的点（对执行漂移鲁棒）。"""
    poses = np.asarray(expert_poses, dtype=np.float64).reshape(-1, 3)
    start = min(max(int(cursor), 0), max(0, len(poses) - 2))
    window = poses[max(0, start - 20): start + 60]
    if len(window):
        distances = np.linalg.norm(window[:, :2] - np.asarray(position, dtype=np.float64)[:2], axis=1)
        start = max(0, start - 20) + int(np.argmin(distances))
    return start


def _base_record(step: int, decision: bool, env, spec) -> dict:
    """所有控制器共用的逐 env-step 记录骨架（§4 冻结字段默认值；决策步覆盖模型/plan 字段）。"""
    ego = env.agent
    return {
        "step": int(step),
        "decision": bool(decision),
        "x": float(ego.position[0]),
        "y": float(ego.position[1]),
        "theta": float(ego.heading_theta),
        "speed": float(ego.speed),
        "lane_lat": _lane_lat(env),
        "lane_curv": _lane_curvature(env),
        "block": _lane_block(env),
        "nav_cps": _nav_checkpoints(env),
        "road_class": _road_class(env),
        "map_id": _map_id(env, spec),
        # 模型行专用（决策步覆盖；专家/baseline 行留空）：
        "policy_action_vs_plan_first_action": None,
        "router_topk": None,
        "policy_std": None,
        # footprint / 跟踪残差 / 时间戳：后处理（attach_* / finalize_episode）填充
        "min_signed_margin": None,
        "plan_footprint_valid": None,
        "exec_footprint_valid": None,
        "tracking_residual": None,
        "T_plan": None,
        "T_track": None,
        "T_cross": None,
        "T_term": None,
    }


def _policy_fields(output) -> dict:
    """模型行专用字段：``policy_std = exp(action_logstd)``；``router_topk`` = MoE 非零权重前 2。"""
    fields = {"policy_std": None, "router_topk": None}
    try:
        logstd = np.asarray(output["action_logstd"].detach().cpu(), dtype=np.float64).reshape(-1)
        fields["policy_std"] = [round(float(np.exp(logstd[0])), 6), round(float(np.exp(logstd[1])), 6)]
    except Exception:  # noqa: BLE001
        pass
    try:
        weights = np.asarray(output["expert_weights"].detach().cpu(), dtype=np.float64).reshape(-1)
        order = np.argsort(-weights)
        fields["router_topk"] = [
            [int(i), round(float(weights[i]), 6)] for i in order if float(weights[i]) > 1e-12
        ][:2]
    except Exception:  # noqa: BLE001
        pass
    return fields


class InstrumentedCkpt:
    """评测协议（pipeline.eval_runner._CkptController）的带记录副本。

    与 ``_CkptController`` 的差异**仅**为：记录 plan/动作/误差/位姿，并允许注册
    带自定义参数的 LqrTracker（``tracker_json``）；执行语义逐行照抄。
    """

    def __init__(
        self,
        env,
        spec,
        task,
        *,
        tracker_json: dict | None = None,
        seed: int = 0,
        dtheta_gain: float = 1.0,
        load_model: bool = True,
        eval_reference: str = "plan",
    ):
        import torch

        from pipeline import eval_runner as ev

        self._ev = ev
        self.spec = spec
        self.seed = int(seed)
        self.dtheta_gain = float(dtheta_gain)
        self.records: list = []
        self.steps: list = []
        self.device = torch.device(str(task.get("device") or "cpu"))
        #: False = 专家参考行（D1/D2）不加载 ckpt（无模型前向；policy/router 字段留空）
        self.model = (
            ev._load_ckpt_model(str(task["ckpt"]), task.get("model_config") or {}, str(self.device))
            if load_model else None
        )
        self.obs_config = dict(task.get("obs_config") or {})
        self.tracker_kind = str(task.get("tracker") or "lqr").lower()
        self.tracker_json = dict(tracker_json or {})
        #: 跟踪器参考口径（--reference）：plan（现状）| repeat_action（F 格）
        self.eval_reference = str(eval_reference or "plan")
        self.builder = None
        self.tracker = None
        self._steps = 0
        self._action = [0.0, 0.0]
        self._pose_history: list = []
        self._ref_windows: list = []  # 每决策步的跟踪器世界系参考前 5 点（0.5 s）
        self._on_lane = None
        self._vehicle_length = None
        self._vehicle_width = None
        self.decision_interval = 5

    # ---------------------------------------------------------------- bind
    def bind(self, env) -> None:
        from env.obs.builder import ObservationBuilder

        self.builder = ObservationBuilder(self.obs_config)
        self.builder.reset()
        self._steps = 0
        self._action = [0.0, 0.0]
        self._pose_history = []
        self._ref_windows = []
        self._on_lane = _make_on_lane_fn(env)
        self._vehicle_length, self._vehicle_width = _vehicle_dims(env)
        env.prev_policy_action = np.zeros(2, dtype=np.float64)
        if self.tracker_kind == "exact":
            from env.tracking import ExactTracker

            self.tracker = ExactTracker(dt=0.5, hz=10)
            return
        if self.tracker_kind == "arc":
            from env.tracking import LqrTracker

            self.tracker = env.engine.add_policy(
                env.agent.id, LqrTracker, env.agent, self.seed, **self.tracker_json
            )
            if hasattr(self.tracker, "reset"):
                self.tracker.reset()
            return
        from env.tracking import LqrTracker

        self.tracker = env.engine.add_policy(
            env.agent.id, LqrTracker, env.agent, self.seed, **self.tracker_json
        )
        if hasattr(self.tracker, "reset"):
            self.tracker.reset()

    # ---------------------------------------------------------------- 动作
    def _tensors(self, obs):
        import torch

        from pipeline.trainer import NON_OBS_KEYS, squeeze_single_slot

        batch = {k: np.asarray(v, dtype=np.float32)[None] for k, v in obs.items() if k not in NON_OBS_KEYS}
        squeeze_single_slot(batch)
        return {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}

    def _measured_prev_action(self, env):
        from env.obs.base import wrap_to_pi

        ego = env.agent
        self._pose_history.append((float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)))
        if len(self._pose_history) <= self.decision_interval:
            return None
        window = self._pose_history[-(self.decision_interval + 1):]
        length = 0.0
        for (ax, ay, _), (bx, by, _) in zip(window[:-1], window[1:]):
            length += math.hypot(bx - ax, by - ay)
        return np.asarray([length, float(wrap_to_pi(window[-1][2] - window[0][2]))], dtype=np.float64)

    def _capture_ref_window(self):
        """跟踪器世界系参考的前 5 个 0.1 s 点（决策后立即调用）；不可用 → None。"""
        if self.tracker is None:
            return None
        world = getattr(self.tracker, "_ref_world", None)
        if world is None:  # ExactTracker 用 _world
            world = getattr(self.tracker, "_world", None)
        if world is None:
            return None
        arr = np.asarray(world, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        idx = [min(i, arr.shape[0] - 1) for i in range(STEPS_PER_POLICY)]
        return arr[idx].copy()

    def _emit(self, rec, ref_window=None) -> None:
        """追加记录（与 ``_ref_windows`` 严格对齐；run_episode 另行追加 ``steps``）。"""
        self.records.append(rec)
        self._ref_windows.append(ref_window)

    def action(self, env):
        import torch

        ego = env.agent
        measured = self._measured_prev_action(env)
        if measured is not None:
            env.prev_policy_action = measured
        obs = self.builder.build(env, self.spec)
        rec = _base_record(self._steps, self._steps % self.decision_interval == 0, env, self.spec)
        ref_window = None
        if self._steps % self.decision_interval == 0:
            need_plan = self.eval_reference == "plan"
            with torch.no_grad():
                output = self.model(self._tensors(obs), rollout=need_plan, world_model=False)
            mu = np.asarray(output["action_mu"].detach().cpu(), dtype=np.float64).reshape(-1)
            plan = None
            if need_plan:
                plan = np.asarray(output["plan"].detach().cpu(), dtype=np.float64).reshape(-1, 2)
            # 参考口径与 eval_runner._CkptController 逐位一致（plan：首步 = mu，历史行为）
            references = np.array(
                self._ev.build_eval_references(mu[:2], plan, self.eval_reference),
                dtype=np.float64,
                copy=True,
            )
            if self.dtheta_gain != 1.0:
                # 纯诊断：只放大"参考"的转向通道，不改模型（判"幅度不足" vs "结构缺陷"）
                references[:, 1] *= self.dtheta_gain
            rec["mu"] = [float(mu[0]), float(mu[1])]
            rec["plan"] = [[float(a), float(b)] for a, b in references]
            rec["reference"] = self.eval_reference
            rec["policy_action_vs_plan_first_action"] = [
                float(references[0, 0] - mu[0]),
                float(references[0, 1] - mu[1]),
            ]
            rec.update(_policy_fields(output))
            if self.tracker_kind == "exact":
                self.tracker.arm(env, actions=references)
            else:
                self.tracker.set_reference(references)
            ref_window = self._capture_ref_window()
        self._action = [0.0, 0.0]
        self._steps += 1
        rec["_ego"] = ego
        self._emit(rec, ref_window)
        self.current = rec
        return self._action

    def post_step(self, env) -> None:
        if self.tracker_kind == "exact" and self.tracker is not None:
            self.tracker.apply(env)

    def action_info(self, env):
        return {"steer": float(self._action[0]), "throttle": float(self._action[1]), "lead_gap_m": -1.0}

    # ---------------------------------------------------------------- tracker 反馈
    def tracker_info(self):
        info = getattr(self.tracker, "action_info", None) or {}
        action = info.get("action") or [None, None]
        return {
            "lqr_error_y": info.get("lqr_error_y"),
            "lqr_error_psi": info.get("lqr_error_psi"),
            "lqr_preview_index": info.get("lqr_preview_index"),
            "lqr_lookahead_m": info.get("lqr_lookahead_m"),
            "lqr_ref_speed_mps": info.get("lqr_ref_speed_mps"),
            "lqr_status": info.get("lqr_status"),
            "steer_applied": action[0],
            "throttle_applied": action[1],
        }


class ArcController(InstrumentedCkpt):
    """合成圆弧参考：每策略步以当前位姿重置 κ 参考（不依赖策略网络）。

    参考 = 6 个 (ds=arc_speed*0.5, dtheta=ds*κ)；κ = 1/radius（左正）。
    """

    def __init__(self, env, spec, task, *, tracker_json=None, radius=8.0, arc_speed=5.0):
        super().__init__(env, spec, task, tracker_json=tracker_json)
        self.radius = float(radius)
        self.arc_speed = float(arc_speed)

    def action(self, env):
        ego = env.agent
        self._measured_prev_action(env)
        rec = _base_record(self._steps, self._steps % self.decision_interval == 0, env, self.spec)
        ref_window = None
        if self._steps % self.decision_interval == 0:
            ds = self.arc_speed * 0.5
            dth = ds / self.radius
            plan = np.asarray([[ds, dth]] * 6, dtype=np.float64)
            rec["mu"] = [float(ds), float(dth)]
            rec["plan"] = [[float(a), float(b)] for a, b in plan]
            rec["reference"] = "arc"
            self.tracker.set_reference(plan)
            ref_window = self._capture_ref_window()
        self._steps += 1
        rec["_ego"] = ego
        self._emit(rec, ref_window)
        self.current = rec
        return [0.0, 0.0]


class InstrumentedBaseline:
    """``_BaselineController``（PurePursuitIDM 规则专家）的带记录适配器（语义全部委托）。"""

    def __init__(self, env, spec, params=None):
        from pipeline import eval_runner as ev

        self._inner = ev._BaselineController(env, spec, params or {})
        self.spec = spec
        self.records: list = []
        self.steps: list = []
        self._steps = 0
        self._ref_windows: list = []
        self._on_lane = None
        self._vehicle_length = None
        self._vehicle_width = None

    def bind(self, env) -> None:
        self._inner.bind(env)
        self._ref_windows = []
        self._on_lane = _make_on_lane_fn(env)
        self._vehicle_length, self._vehicle_width = _vehicle_dims(env)

    def action(self, env):
        rec = _base_record(self._steps, True, env, self.spec)  # baseline 每 env step 都决策
        self._steps += 1
        self.records.append(rec)
        self._ref_windows.append(None)
        return self._inner.action(env)

    def action_info(self, env):
        return self._inner.action_info(env)

    def tracker_info(self):
        return {}

    def params(self):
        return self._inner.params()


class LanePlanController(InstrumentedCkpt):
    """Oracle plan：每 0.5 s 用**车道中心线**构造 3 s 参考（不依赖策略网络），交给同一个 LQR。

    参考构造：从自车当前车道位置沿路由（ego lane → nav.next_ref_lanes）按 0.1 s 间隔采样
    世界系中心线位姿 → 转到自车系 → 按 0.5 s 窗口折算 ``(ds, dtheta)``（与
    ``env.tracking.roundtrip_error`` 同口径），正好 6 段。
    速度：``speed_mode=hold`` 保持当前速度；``limit`` 用车道限速（clip 到 [1, speed_cap]）。
    """

    def __init__(self, env, spec, task, *, tracker_json=None, speed_mode="limit", speed_cap=8.0):
        super().__init__(env, spec, task, tracker_json=tracker_json)
        self.speed_mode = str(speed_mode)
        self.speed_cap = float(speed_cap)

    def _route_poses(self, env, horizon_s=3.0, dt=0.1):
        ego = env.agent
        nav = getattr(ego, "navigation", None)
        lane = getattr(ego, "lane", None)
        candidates = []
        try:
            candidates = [item for item in list(getattr(nav, "next_ref_lanes", []) or []) if item is not lane]
        except Exception:  # noqa: BLE001
            candidates = []
        lanes = [ln for ln in [lane] if ln is not None]
        # 贪心接续：优先 index 前缀匹配 + 车道号相同，其次起点最近
        while lanes:
            cur = lanes[-1]
            cur_idx = tuple(getattr(cur, "index", ()) or ())
            best, best_key = None, None
            try:
                cur_end = cur.position(float(cur.length), 0.0)
            except Exception:  # noqa: BLE001
                cur_end = None
            for cand in candidates:
                if cand in lanes:
                    continue
                cidx = tuple(getattr(cand, "index", ()) or ())
                if len(cur_idx) < 3 or len(cidx) < 3:
                    continue
                # index = (from_block, block, lane_no)：接续 = 本车道 block == 候选 from_block
                if cidx[0] != cur_idx[1]:
                    continue
                same_lane_no = bool(cidx[2] == cur_idx[2])
                dist = 0.0
                if cur_end is not None:
                    try:
                        dist = float(np.linalg.norm(np.asarray(cand.position(0.0, 0.0)) - np.asarray(cur_end)))
                    except Exception:  # noqa: BLE001
                        dist = 0.0
                key = (0 if same_lane_no else 1, round(dist, 2))
                if best_key is None or key < best_key:
                    best, best_key = cand, key
            if best is None:
                break
            lanes.append(best)
            if len(lanes) >= 3:
                break
        if not lanes:
            return None
        speed = float(ego.speed)
        if self.speed_mode == "limit":
            try:
                lim = float(lane.speed_limit) if lane is not None else float("nan")
                if not (0.0 < lim < 1000.0):
                    lim = 8.0
            except Exception:  # noqa: BLE001
                lim = 8.0
            speed = float(min(max(lim, 1.0), self.speed_cap))
        else:
            speed = float(max(min(speed, self.speed_cap), 1.0))
        # 自车在首条车道的纵向位置
        s = 0.0
        try:
            proj = lane.local_coordinates(ego.position)
            s = float(proj[0])
        except Exception:  # noqa: BLE001
            s = 0.0
        poses = []
        t = dt
        while t <= horizon_s + 1e-9:
            target = t * speed
            remain = target
            cur_s, k = s, 0
            while k < len(lanes):
                ln = lanes[k]
                ln_len = float(getattr(ln, "length", 0.0))
                if cur_s + remain <= ln_len:
                    sample_s = cur_s + remain
                    break
                remain -= max(ln_len - cur_s, 0.0)
                k += 1
                cur_s = 0.0
            else:
                k = len(lanes) - 1
                sample_s = float(getattr(lanes[k], "length", 0.0))
            ln = lanes[k]
            try:
                pos = ln.position(sample_s, 0.0)
                th = float(ln.heading_theta_at(sample_s))
                poses.append([float(pos[0]), float(pos[1]), th])
            except Exception:  # noqa: BLE001
                break
            t += dt
        return np.asarray(poses, dtype=np.float64) if poses else None

    def action(self, env):
        ego = env.agent
        self._measured_prev_action(env)
        rec = _base_record(self._steps, self._steps % self.decision_interval == 0, env, self.spec)
        ref_window = None
        if self._steps % self.decision_interval == 0:
            poses = self._route_poses(env, horizon_s=3.0, dt=0.1)
            if poses is not None and len(poses) >= 30:
                from env.tracking import world_to_ego
                from env.obs.base import wrap_to_pi

                base = np.array([float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)])
                local = world_to_ego(poses, base)
                plan = []
                for seg in range(6):
                    win = local[seg * 5: (seg + 1) * 5 + 1]
                    ds = float(np.linalg.norm(np.diff(win[:, :2], axis=0), axis=1).sum())
                    dth = float(wrap_to_pi(win[-1, 2] - win[0, 2]))
                    plan.append([ds, dth])
                plan = np.asarray(plan, dtype=np.float64)
                rec["mu"] = [float(plan[0, 0]), float(plan[0, 1])]
                rec["plan"] = [[float(a), float(b)] for a, b in plan]
                rec["reference"] = "laneplan"
                self.tracker.set_reference(plan)
                ref_window = self._capture_ref_window()
        self._steps += 1
        rec["_ego"] = ego
        self._emit(rec, ref_window)
        self.current = rec
        return [0.0, 0.0]


class OracleReplayController(InstrumentedCkpt):
    """Oracle 路径执行：参考 = 同一 spec 上规则专家（PurePursuitIDM）实测轨迹的**未来 3 s**。

    用于判定"执行栈（LqrTracker）能否跟上一条**好路径**"：把专家轨迹按当前位姿转到自车系，
    每 0.5 s 刷新一次参考。速度参考同样来自专家轨迹弦长。
    """

    def __init__(self, env, spec, task, *, tracker_json=None, oracle_poses=None):
        super().__init__(env, spec, task, tracker_json=tracker_json)
        self.oracle = np.asarray(oracle_poses, dtype=np.float64)  # (N,3) 世界系
        self.k = 0

    def action(self, env):
        ego = env.agent
        self._measured_prev_action(env)
        rec = _base_record(self._steps, self._steps % self.decision_interval == 0, env, self.spec)
        ref_window = None
        if self._steps % self.decision_interval == 0 and self.oracle is not None and len(self.oracle) > 1:
            from env.tracking import world_to_ego

            base = np.array([float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)])
            start = min(self.k, len(self.oracle) - 2)
            # 用最近点重定位 cursor（对位姿漂移鲁棒），再取未来 30 点
            window = self.oracle[max(0, start - 20): start + 60]
            if len(window):
                d = np.linalg.norm(window[:, :2] - base[:2], axis=1)
                start = max(0, start - 20) + int(np.argmin(d))
            chunk = self.oracle[start: start + 31]
            if len(chunk) >= 2:
                local = world_to_ego(chunk, base)
                rec["plan"] = [[float(p[0]), float(p[1]), float(p[2])] for p in local]
                rec["mu"] = [float(local[0, 0]), float(local[0, 2])]
                rec["reference"] = "oracle_poses"
                self.tracker.set_reference(local)  # (N,3) 自车系位姿参考
                ref_window = self._capture_ref_window()
            self.k = start + 5
        self._steps += 1
        rec["_ego"] = ego
        self._emit(rec, ref_window)
        self.current = rec
        return [0.0, 0.0]


class D1ReplayController(InstrumentedCkpt):
    """D1（privileged ceiling）：未来 expert 动作链。

    pass 1 对同 (spec, seed) 跑规则专家取实测 10 Hz 轨迹（见 main 的 ``d1`` 分支）；
    本控制器每 0.5 s 把 cursor 重定位到轨迹上离当前位姿最近点，取未来 3 s 按
    ``tools/collect_expert._window_actions`` 同款 pose-delta 口径生成 6 步 ``(ds,dθ)``
    （首点 = expert 下一步动作，**不做 mu 覆盖**），交给同一 LQR。
    """

    def __init__(self, env, spec, task, *, tracker_json=None, expert_poses=None):
        super().__init__(env, spec, task, tracker_json=tracker_json, load_model=False)
        self.expert = np.asarray(expert_poses, dtype=np.float64) if expert_poses is not None else None
        self.k = 0

    def action(self, env):
        ego = env.agent
        self._measured_prev_action(env)
        rec = _base_record(self._steps, self._steps % self.decision_interval == 0, env, self.spec)
        ref_window = None
        if self._steps % self.decision_interval == 0 and self.expert is not None and len(self.expert) > 1:
            start = relocate_cursor(self.expert, self.k, (float(ego.position[0]), float(ego.position[1])))
            plan = expert_chain_plan(self.expert, start)
            self.k = start + STEPS_PER_POLICY
            if plan is not None:
                rec["mu"] = [float(plan[0, 0]), float(plan[0, 1])]
                rec["plan"] = [[float(a), float(b)] for a, b in plan]
                rec["reference"] = "expert_chain"
                rec["privileged"] = True
                self.tracker.set_reference(plan)
                ref_window = self._capture_ref_window()
        self._steps += 1
        rec["_ego"] = ego
        self._emit(rec, ref_window)
        self.current = rec
        return [0.0, 0.0]


class D2RepeatExpertController(InstrumentedCkpt):
    """D2：当前 expert 动作 repeat 6 步（与 F 格 ``repeat_action`` 直接可比）。

    每 0.5 s 对**现状态**做专家空问：规则专家（未注册 engine，只调 ``act()``，无执行
    副作用）→ ``[steer, throttle]`` → ``expert_action_to_ds_dtheta``（与
    ``tools/dagger_collect`` 采集侧同口径，即 ``pipeline.trainer.expand_policy_action``
    的解析逆映射）→ ``np.repeat(·, 6)`` → 交同一 LQR。
    """

    def __init__(self, env, spec, task, *, tracker_json=None, expert_kind="pure_pursuit"):
        super().__init__(env, spec, task, tracker_json=tracker_json, load_model=False)
        self.expert_kind = str(expert_kind)
        self.labeler = None

    def bind(self, env) -> None:
        super().bind(env)
        from tools.dagger_collect import _make_labeler

        # 复用 DAgger 采集侧的专家空问实现（不注册 engine；口径唯一来源）
        self.labeler = _make_labeler(env, self.spec, self.expert_kind)

    def action(self, env):
        ego = env.agent
        self._measured_prev_action(env)
        rec = _base_record(self._steps, self._steps % self.decision_interval == 0, env, self.spec)
        ref_window = None
        if self._steps % self.decision_interval == 0 and self.labeler is not None:
            labeled = self.labeler.label(ego)
            action = np.asarray(labeled["action"], dtype=np.float64).reshape(-1)[:2]
            plan = np.repeat(action[None, :], WINDOW_POLICIES, axis=0)
            rec["mu"] = [float(action[0]), float(action[1])]
            rec["plan"] = [[float(a), float(b)] for a, b in plan]
            rec["reference"] = "expert_repeat"
            rec["privileged"] = True
            rec["expert_raw_action"] = [
                float(x) for x in np.asarray(labeled["raw_action"], dtype=np.float64).reshape(-1)[:2]
            ]
            self.tracker.set_reference(plan)
            ref_window = self._capture_ref_window()
        self._steps += 1
        rec["_ego"] = ego
        self._emit(rec, ref_window)
        self.current = rec
        return [0.0, 0.0]


def run_episode(env, spec, controller, *, max_steps: int = 1000):
    """按 _run_episode 的执行顺序跑一条 episode，并给 controller 的记录补 env 侧信息。"""
    from pipeline import eval_runner as ev

    reset_out = env.reset()
    controller.bind(env)
    ego = env.agent
    info: dict = {}
    if isinstance(reset_out, tuple) and len(reset_out) == 2 and isinstance(reset_out[1], dict):
        info = dict(reset_out[1])
    steps = 0
    for step_index in range(max_steps):
        action = controller.action(env)
        _, _, terminated, truncated, info = env.step(action)
        if hasattr(controller, "post_step"):
            controller.post_step(env)
        info = info if isinstance(info, dict) else {}
        steps = step_index + 1
        trk = controller.tracker_info()
        row = {
            "step": int(step_index),
            "t": round(0.1 * steps, 2),
            "x": float(ego.position[0]),
            "y": float(ego.position[1]),
            "theta": round(float(ego.heading_theta), 5),
            "speed": round(float(info.get("velocity", float(ego.speed))), 3),
            "rc": round(float(info.get("route_completion", 0.0)), 4),
            "out_of_road": bool(info.get("out_of_road", False)),
            "crash": bool(ev._is_crash(info)),
            "arrive": bool(info.get("arrive_dest", False)),
            "lane_lat": _lane_lat(env),
            "lane_curv": _lane_curvature(env),
            "block": _lane_block(env),
            "steer": float(action[0]),
            "throttle": float(action[1]),
        }
        row.update({k: _wrap(v) for k, v in trk.items()})
        if controller.records:
            prev = controller.records[-1]
            prev["env_after"] = row
            prev["steer"] = row.get("steer")
        controller.steps.append(row)
        if terminated or truncated:
            break
    last = controller.steps[-1] if controller.steps else {}
    return {
        "steps": steps,
        "termination": ("arrive_dest" if last.get("arrive") else "collision" if last.get("crash")
                        else "out_of_road" if last.get("out_of_road") else "max_step"),
        "rc_final": last.get("rc"),
        "rc_max": max([s.get("rc") or 0.0 for s in controller.steps] or [0.0]),
        "speed_mean": float(np.mean([s["speed"] for s in controller.steps])) if controller.steps else float("nan"),
    }


def _build_env(spec, config):
    from pipeline import eval_runner as ev
    from env.metadrive_env import build_env

    return ev._unwrap_env(build_env(spec, traffic_density=None, use_render=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forensics_closed_loop.py")
    parser.add_argument("--ids", default="", help="逗号分隔 spec id（缺省=全部焦点几何）")
    parser.add_argument(
        "--mode",
        default="lqr",
        choices=("lqr", "lqr_gain", "exact", "baseline", "arc", "laneplan", "oracle", "d1", "d2"),
    )
    parser.add_argument(
        "--reference",
        default="plan",
        choices=("plan", "repeat_action"),
        help="模型行跟踪器参考口径（缺省 plan=现状；repeat_action=eval_runner 同口径，供 F 格）",
    )
    parser.add_argument("--speed-mode", default="limit", choices=("hold", "limit"))
    parser.add_argument("--dtheta-gain", type=float, default=1.0, help="诊断：放大 plan 的转向通道")
    parser.add_argument("--speed-cap", type=float, default=8.0)
    parser.add_argument("--spec", default=SPEC)
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--tracker-json", default="{}")
    parser.add_argument("--arc-radius", type=float, default=8.0)
    parser.add_argument("--arc-speed", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True)
    return parser


def _analyze_episode(controller, outcome) -> dict:
    """后处理一条 episode：footprint + tracking_residual + 时间戳/分类（原地写 records）。"""
    records = getattr(controller, "records", None)
    steps = getattr(controller, "steps", None)
    if not records or steps is None:
        return {"timestamps": {}, "classification": {}}
    on_lane = getattr(controller, "_on_lane", None)
    length = getattr(controller, "_vehicle_length", None)
    width = getattr(controller, "_vehicle_width", None)
    if on_lane is not None and length and width:
        attach_footprints(records, steps, on_lane, length, width)
    attach_tracking_residual(records, steps, getattr(controller, "_ref_windows", None))
    return finalize_episode(records, steps, (outcome or {}).get("termination") or "error")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from pipeline import eval_runner as ev
    from pipeline.gl_runtime import ensure_gl_library_path

    ensure_gl_library_path(preload=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    config = ev.load_config(args.config)
    env_cfg = dict(config.get("env") or {})
    task = {
        "policy": "ckpt",
        "ckpt": args.ckpt,
        "model_config": {
            "hidden_dim": config.get("hidden_dim", 128),
            "moe": dict(config.get("moe") or {}),
            "world_model": dict(config.get("world_model") or {}),
        },
        "obs_config": dict(env_cfg.get("obs") or {}),
        "device": args.device,
        "tracker": "exact" if args.mode == "exact" else "lqr",
        "eval_reference": args.reference,
    }
    tracker_json = json.loads(args.tracker_json)
    if args.reference != "plan" and args.mode not in ("lqr", "lqr_gain", "exact"):
        print(f"[forensics] 警告：--reference={args.reference} 仅对 lqr/lqr_gain/exact 生效，{args.mode} 忽略", flush=True)

    from env.scenario.spec import load_specs

    specs = list(load_specs(args.spec))
    wanted = [int(x) for x in args.ids.split(",") if x.strip()] if args.ids.strip() else None
    if wanted is None:
        focus = {"curve", "roundabout", "uturn", "tollgate"}
        wanted = [int(s.id) for s in specs if str(getattr(s, "labels", {}).get("geometry", "")) in focus]
    chosen = [s for s in specs if int(s.id) in set(wanted)]

    report = {
        "mode": args.mode,
        "reference": args.reference,
        "privileged": args.mode in ("d1", "d2"),
        "tracker_json": tracker_json,
        "episodes": [],
    }
    started = time.time()
    for spec in chosen:
        t0 = time.time()
        controller = None
        env = None
        try:
            if args.mode in ("oracle", "d1"):
                # pass 1：规则专家跑一遍，取实测轨迹（oracle 路径 / D1 动作链来源）
                env1 = _build_env(spec, config)
                base_ctrl = InstrumentedBaseline(env1, spec, {})
                run_episode(env1, spec, base_ctrl, max_steps=args.max_steps)
                expert_poses = [[s["x"], s["y"], s["theta"]] for s in base_ctrl.steps]
                try:
                    env1.close()
                except BaseException:
                    pass
                env = _build_env(spec, config)
                if args.mode == "oracle":
                    controller = OracleReplayController(env, spec, task, tracker_json=tracker_json,
                                                        oracle_poses=expert_poses)
                else:
                    controller = D1ReplayController(env, spec, task, tracker_json=tracker_json,
                                                    expert_poses=expert_poses)
            else:
                env = _build_env(spec, config)
                if args.mode == "baseline":
                    controller = InstrumentedBaseline(env, spec, {})
                elif args.mode == "arc":
                    controller = ArcController(env, spec, task, tracker_json=tracker_json,
                                               radius=args.arc_radius, arc_speed=args.arc_speed)
                elif args.mode == "laneplan":
                    controller = LanePlanController(env, spec, task, tracker_json=tracker_json,
                                                    speed_mode=args.speed_mode, speed_cap=args.speed_cap)
                elif args.mode == "d2":
                    controller = D2RepeatExpertController(env, spec, task, tracker_json=tracker_json)
                else:
                    controller = InstrumentedCkpt(env, spec, task, tracker_json=tracker_json,
                                                   dtheta_gain=args.dtheta_gain,
                                                   eval_reference=args.reference)
            outcome = run_episode(env, spec, controller, max_steps=args.max_steps)
        except BaseException as exc:  # noqa: BLE001
            import traceback

            outcome = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        # footprint 的射线查询需要**存活**的 physics world → 分析必须在 env.close() 之前
        analysis = _analyze_episode(controller, outcome) if controller is not None else {
            "timestamps": {}, "classification": {}
        }
        try:
            if env is not None:
                env.close()
        except BaseException:
            pass
        entry = {
            "id": int(spec.id),
            "seed": int(getattr(spec, "seed", -1)),
            "geometry": str(getattr(spec, "labels", {}).get("geometry", "?")),
            "geometry_seq": list(getattr(spec, "geometry", []) or []),
            "difficulty": str(getattr(spec, "difficulty", "?")),
            "success": bool(outcome.get("termination") == "arrive_dest"),
            "off_road": bool(outcome.get("termination") == "out_of_road"),
            "collision": bool(outcome.get("termination") == "collision"),
            "rc": outcome.get("rc_final"),
            "outcome": outcome,
            "timestamps": analysis["timestamps"],
            "classification": analysis["classification"],
            "wall_s": round(time.time() - t0, 1),
        }
        records = getattr(controller, "records", None)
        if records is not None:
            entry["records"] = records
            entry["steps"] = getattr(controller, "steps", [])
        report["episodes"].append(entry)
        print(
            f"[forensics] id={spec.id} {entry['geometry']:<12} "
            f"term={outcome.get('termination')} rc={outcome.get('rc_final')} steps={outcome.get('steps')} "
            f"cls={entry['classification'].get('primary')} ({entry['wall_s']}s)",
            flush=True,
        )
    report["elapsed_s"] = round(time.time() - started, 1)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[forensics] → {out_path} ({report['elapsed_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
