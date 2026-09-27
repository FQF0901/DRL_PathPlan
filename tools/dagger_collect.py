#!/usr/bin/env python3
"""DAgger-lite 采集（lane D2，2026-09-27 重写）：**策略驱动 roll-in + 规则专家"空问"标签**。

协议（用户定稿）
----------------
1. **场景池**：``--from-eval runs/*_eval*/episodes.csv``（``success=False`` 的 spec；spec 源默认
   从该 run 的 ``metrics.json::meta.specs_path`` 解析，缺省回退 ``env/specs/scenarios_eval500.json``），
   或 ``--specs`` 直接给池；不给则取最新的 ``runs/*eval500*/episodes.csv``。
2. **roll-in = 学生策略驱动**（``--ckpt``，与评测同协议）：复用
   ``pipeline.eval_runner._CkptController``（``--tracker lqr`` 默认 = ``engine.add_policy(LqrTracker)``
   + 每 0.5 s ``set_reference(plan)``；``--tracker exact`` = 运动学精确执行），观测链 =
   ``env.obs.builder.ObservationBuilder``（与 collect_expert/eval 同源）。**访问状态 = 学生自己踩到的状态**，
   不是专家状态分布（旧实现的差异点）。
3. **标签 = 专家空问**：每个策略步对**当前状态**调用规则专家 ``act()``（默认
   ``env.expert.pure_pursuit_idm.PurePursuitIDMPolicy``；不注册进 engine —— 只计算不执行，
   策略是纯状态函数），首步动作换算成 BC 口径 ``(ds, dθ)``。
4. **动作换算链（已核对，不复制既有逻辑）**：
   - 仓库内没有现成的 ``[steer, throttle] → (ds, dθ)`` 反解函数（核对 ``env/tracking.py``、
     ``tools/collect_expert.py``、``tools/baseline_eval.py``）；本工具实现
     ``pipeline.trainer.expand_policy_action``（运动学执行模型）的解析逆映射：
     ``ω = tan(steer·max_steer_rad)·max(v, 0.5)/wheelbase`` → ``dθ = ω·dt``；
     ``v_ref = v + throttle·scale``（``scale`` 取专家自身 ``accel/brake_action_scale``）→
     ``ds = max(v_ref, 0)·dt``。单测用 ``expand_policy_action`` 正算做 round-trip 校验。
   - ``(ds,dθ) → 轨迹`` 复用 ``tools/collect_expert.resolve_interpolate``（``env.tracking.interpolate``，
     §8.5 单一真源）；6 点未来窗口无法从"空问"获得 → ``action[1:]`` = 首步**常量重复**、
     ``traj6/traj30`` 由其插值生成，meta 显式标注 ``label_scope=first_step_only`` /
     ``window_fill=constant_repeat``。
5. **过滤**：``extract_samples`` 的 on-lane / terminal_window / cut_label_unverified（round-trip
   过滤对"空问标签"不适用 → 阈值置 ∞ 并记录），外加 lane T 病态规则 ``stuck``（< 0.5 m/s）/
   ``yaw_outlier``（|dθ| > 0.35 rad/0.5 s）；逐行 ``train_weight`` 置 0（不删行），统计进
   report/meta。
6. **产物**（与 ``tools/collect_expert.py`` 同 schema 的**完整 episode** 数据）：
   ``expert_bc.npz`` + ``expert_bc.meta.json``（``dagger`` 块：driver ckpt sha256 / labeler /
   spec 清单 / 过滤统计 / ``roll_in: true`` / 标签口径）+ ``report.json`` + ``dagger.json`` +
   ``dagger_specs.json``；分片目录默认清理（``--keep-shards`` 保留）。``--workers N`` = spawn 多核。

复用（不复制实现）
------------------
``tools/collect_expert.py``：``extract_samples``（行装配 + 过滤 + wm_valid/labels/pre 动作）、
``resolve_interpolate``（运动学）、``_summarize_records``/``_spec_entries``/``_shard_segments``/
``_concat_shards``/``_merge_filter_counts``（分片合并）、``balance_from_specs``（配平）、
``_write_dataset``（schema 写出）、``write_shard``、``_spec_progress_line``、
``_canonical_key_order``、常量；
``pipeline.eval_runner._CkptController``（评测同款策略驱动）、``pipeline.hard_mining.ckpt_identity``
（driver ckpt sha256）。

用法::

    tools/venv-python tools/dagger_collect.py --ckpt <stage_b/final.pt> \\
        --from-eval runs/BTC20260927-1839_eval500_lqr/episodes.csv --limit 5 --max-steps 60 \\
        --workers 1 --device cpu --out /tmp/opencode/dagger_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.metadrive_env import build_env  # noqa: E402
from env.obs.builder import ObservationBuilder  # noqa: E402
from env.scenario.labels import LABEL_ORDER, compute_step_labels  # noqa: E402
from env.scenario.behaviors import event_state  # noqa: E402
from env.scenario.spec import load_specs, save_specs  # noqa: E402
from pipeline.gl_runtime import ensure_gl_library_path  # noqa: E402
from tools.collect_expert import (  # noqa: E402 - 复用共享实现（不改其行为）
    DEFAULT_MIN_SAMPLES_PER_CATEGORY,
    DEFAULT_MIN_YIELD,
    PHYSICS_DT,
    POLICY_DT,
    RECYCLE_EVERY_SPECS,
    SHARD_DIRNAME,
    STEPS_PER_POLICY,
    WINDOW_POLICIES,
    _concat_shards,
    _ego_pose,
    _events_summary,
    _flag_dict,
    _merge_filter_counts,
    _on_lane,
    _shard_segments,
    _spec_entries,
    _spec_progress_line,
    _summarize_records,
    balance_from_specs,
    extract_samples,
    load_supervised_labels,
    resolve_interpolate,
    _write_dataset,
)

__all__ = ["main", "expert_action_to_ds_dtheta"]

#: 病态过滤阈值（沿用 lane T；写死并记录进 report/meta）
STUCK_SPEED_MPS = 0.5
YAW_OUTLIER_RAD = 0.35
#: 默认 labeler / spec 源 / 每 worker 一个任务的 spec 数（分片粒度）
DEFAULT_LABELER = "pure_pursuit"
DEFAULT_SPEC_SOURCE = "env/specs/scenarios_eval500.json"
CHUNK_SPECS = 10
#: 换算链默认常数（DefaultVehicle 轴距 + 平台 max_steering=40°；见 pipeline.trainer.expand_policy_action）
DEFAULT_WHEELBASE_M = 2.46894
DEFAULT_MAX_STEER_RAD = math.radians(40.0)
DEFAULT_ACCEL_SCALE = 2.0


# --------------------------------------------------------------------------- #
# 空问标签：专家动作 [steer, throttle] → BC 口径 (ds, dθ)
# --------------------------------------------------------------------------- #

def expert_action_to_ds_dtheta(
    action: Sequence[float],
    *,
    speed: float,
    dt: float = POLICY_DT,
    wheelbase: float = DEFAULT_WHEELBASE_M,
    max_steer_rad: float = DEFAULT_MAX_STEER_RAD,
    accel_scale: float = DEFAULT_ACCEL_SCALE,
    brake_scale: Optional[float] = None,
) -> np.ndarray:
    """``[steer, throttle]`` → 一个策略步的 ``(ds[m], dθ[rad])``（解析反解，与执行模型互逆）。

    与 ``pipeline.trainer.expand_policy_action``（(ds,dθ) → 子步 [steer,throttle] 的运动学
    执行模型）同参数时互为逆映射（单测 round-trip）：
    ``ω = tan(steer·max_steer_rad)·max(v,0.5)/wheelbase``；``v_ref = v + throttle·scale``。
    """
    values = np.asarray(action, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError(f"专家动作需要 2 维 [steer, throttle]，收到 {values!r}")
    steer = float(np.clip(values[0], -1.0, 1.0))
    throttle = float(np.clip(values[1], -1.0, 1.0))
    scale = float(brake_scale) if (brake_scale is not None and throttle < 0.0) else float(accel_scale)
    omega = math.tan(steer * float(max_steer_rad)) * max(float(speed), 0.5) / float(wheelbase)
    v_ref = max(float(speed) + throttle * scale, 0.0)
    return np.asarray([v_ref * float(dt), omega * float(dt)], dtype=np.float64)


class _ExpertLabeler:
    """专家空问：持有一个**未注册**的规则策略，只对当前状态调 ``act()``（无执行副作用）。"""

    def __init__(self, policy: Any, kind: str):
        self.policy = policy
        self.kind = str(kind)
        params = policy.params() if callable(getattr(policy, "params", None)) else {}
        self.wheelbase = float(params.get("wheelbase", DEFAULT_WHEELBASE_M))
        self.max_steer_rad = float(params.get("max_steer_angle_rad", DEFAULT_MAX_STEER_RAD))
        self.accel_scale = float(params.get("accel_action_scale", DEFAULT_ACCEL_SCALE))
        self.brake_scale = float(params.get("brake_action_scale", DEFAULT_ACCEL_SCALE))

    def label(self, ego: Any) -> Dict[str, Any]:
        raw = np.asarray(self.policy.act(), dtype=np.float64).reshape(-1)[:2]
        action = expert_action_to_ds_dtheta(
            raw,
            speed=float(ego.speed),
            wheelbase=self.wheelbase,
            max_steer_rad=self.max_steer_rad,
            accel_scale=self.accel_scale,
            brake_scale=self.brake_scale,
        )
        return {"action": action, "raw_action": raw}

    def params(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "wheelbase": self.wheelbase,
            "max_steer_angle_rad": self.max_steer_rad,
            "accel_action_scale": self.accel_scale,
            "brake_action_scale": self.brake_scale,
        }


def _make_labeler(env: Any, spec: Any, kind: str) -> _ExpertLabeler:
    """构造规则专家（**不**注册进 engine：空问 = 只计算不执行）。"""
    seed = int(getattr(spec, "seed", 0))
    if str(kind).lower() == "idm":
        from metadrive.policy.idm_policy import IDMPolicy

        policy = IDMPolicy(env.agent, seed)
    else:
        from env.expert.pure_pursuit_idm import PurePursuitIDMPolicy

        policy = PurePursuitIDMPolicy(env.agent, seed)
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    return _ExpertLabeler(policy, kind)


# --------------------------------------------------------------------------- #
# 策略驱动 episode（学生闭环 + 每策略步专家空问）
# --------------------------------------------------------------------------- #

def _dagger_episode(
    env: Any,
    spec: Any,
    *,
    controller: Any,
    labeler_factory: Any,
    builder: ObservationBuilder,
    max_steps: int,
) -> Dict[str, Any]:
    """跑一个**策略驱动** episode，逐 env-step 记位姿/标记 + 逐策略步记学生 obs 与专家空问标签。

    记录结构与 ``tools.collect_expert.collect_episode`` 完全一致（``frames``/``poses``/``flags``/
    ``termination``/``steps``/``ended_by_env``），外加 ``expert_labels[step]``；
    ``extract_samples`` 因此可原样复用（过滤/权重/wm_valid/labels 全走同一实现）。

    ``labeler_factory`` 在 **env.reset() 之后**调用：规则专家构造依赖全局 engine
    （``BasePolicy.show_policy_mark`` → ``get_engine()``），reset 前建 env 会触发
    "Please initialize the environment first!"。
    """
    env.reset()
    controller.bind(env)
    labeler = labeler_factory()  # 空问专家（不注册 engine）
    expert_labels: Dict[int, Dict[str, Any]] = {}
    frames: Dict[int, Dict[str, Any]] = {}
    poses: List[np.ndarray] = [_ego_pose(env.agent)]
    flags: List[Optional[Dict[str, Any]]] = [None]
    termination = "max_step"  # 工具自身 max_steps 截断（未到 env 终局）
    ended_by_env = False
    step = 0
    env.prev_policy_action = np.zeros(2, dtype=np.float64)
    while step < int(max_steps):
        action = controller.action(env)  # 学生策略驱动（同时写 prev_policy_action + 建 obs）
        if step % STEPS_PER_POLICY == 0:
            obs = builder.build(env, spec)  # 同链二次 build（FrameMemory 同 step 幂等）
            labels_raw = compute_step_labels(env, spec)
            state = event_state(env)
            on_lane, lateral, width = _on_lane(env.agent)
            current = {key: np.array(obs[key], copy=True) for key in obs if "_hist" not in key}
            frames[step] = {
                "obs": current,
                "hist_valid": np.array(obs.get("hist_valid", np.zeros(6, dtype=np.float32)), dtype=np.float32),
                "od_id_hist": np.array(
                    obs.get("od_id_hist", np.full((6, 16), -1, dtype=np.int64)), dtype=np.int64, copy=True
                ),
                "od_presence_hist": np.array(
                    obs.get("od_presence_hist", np.zeros((6, 16), dtype=np.float32)), dtype=np.float32, copy=True
                ),
                "labels_raw": np.array([labels_raw.get(name, 0.0) for name in LABEL_ORDER], dtype=np.float32),
                "events": _events_summary(state),
                "pose": poses[-1].copy(),
                "on_lane": bool(on_lane),
                "lane_lat": float(lateral),
                "lane_width": float(width),
            }
            expert_labels[step] = labeler.label(env.agent)  # 空问：当前状态 → 专家首步标签
        _, _, terminated, truncated, info = env.step(action)
        post_step = getattr(controller, "post_step", None)
        if callable(post_step):
            post_step(env)  # exact 跟踪器专有（lqr 无此方法）
        step += 1
        poses.append(_ego_pose(env.agent))
        flag = _flag_dict(info)
        flags.append(flag)
        if terminated or truncated:
            ended_by_env = True
            if flag["crash"]:
                termination = "collision"
            elif flag["out_of_road"]:
                termination = "out_of_road"
            elif flag["arrive"]:
                termination = "arrive_dest"
            else:
                termination = "max_step" if flag["max_step"] else "terminal"
            break
    return {
        "frames": frames,
        "poses": poses,
        "flags": flags,
        "termination": termination,
        "steps": step,
        "ended_by_env": ended_by_env,
        "expert_labels": expert_labels,
        "labeler_params": labeler.params(),
    }


def _interpolate(interpolate_fn: Any, actions: np.ndarray) -> np.ndarray:
    """调用运动学插值（兼容不吃 kwargs 的实现；与 collect_expert 同口径）。"""
    try:
        return np.asarray(
            interpolate_fn(actions, dt=POLICY_DT, hz=int(round(1.0 / PHYSICS_DT))), dtype=np.float64
        )
    except TypeError:
        return np.asarray(interpolate_fn(actions), dtype=np.float64)


def _relabel_rows_with_expert(
    rows: Sequence[Dict[str, Any]], episode: Dict[str, Any], *, interpolate_fn: Any
) -> int:
    """把 ``extract_samples`` 的实测轨迹目标替换为**专家空问标签**（首步真标签 + 常量重复窗口）。

    - ``action[0]`` = 专家首步 ``(ds, dθ)``；``action[1:]`` = 常量重复（meta 标注）；
    - ``traj6/traj30`` = 上述窗口经 ``env.tracking.interpolate`` 的运动学轨迹；
    - ``roundtrip_*`` 置 NaN（对"空问标签"无意义；round-trip 过滤已按阈值 ∞ 关闭）；
    - ``traj30_measured``（学生实测未来位姿）保持不变，作为 roll-in 行为诊断。
    返回被重标注的行数。
    """
    labels = episode.get("expert_labels") or {}
    key_indices = [index * STEPS_PER_POLICY - 1 for index in range(1, WINDOW_POLICIES + 1)]
    relabeled = 0
    for row in rows:
        info = labels.get(int(row["step"]))
        if info is None:
            continue
        window = np.repeat(np.asarray(info["action"], dtype=np.float64)[None, :], WINDOW_POLICIES, axis=0)
        traj = _interpolate(interpolate_fn, window)
        row["action"] = window.astype(np.float32)
        row["traj6"] = traj[key_indices, :2].astype(np.float32)
        row["traj30"] = traj[:, :2].astype(np.float32)
        row["roundtrip_key_err"] = float("nan")
        row["roundtrip_dense_err"] = float("nan")
        relabeled += 1
    return relabeled


# --------------------------------------------------------------------------- #
# 病态过滤（沿用 lane T 规则；行级：只改 train_weight，不删行）
# --------------------------------------------------------------------------- #

def _apply_pathological_filters(
    samples: Sequence[Dict[str, Any]],
    *,
    stuck_speed: float = STUCK_SPEED_MPS,
    yaw_outlier: float = YAW_OUTLIER_RAD,
    filter_counter: Optional[Counter] = None,
) -> Dict[str, Any]:
    """病态行过滤（写死阈值，记录进 report/meta）：卡死 / 大转向离群 → ``train_weight=0``。

    速度 = ``traj30_measured``（缺失回退 ``traj30``）首末点距离 / 2.5 s（0.5 s × 5 段）；
    dθ = ``action[0, 1]``（专家首步标签）。只作用于 ``train_weight>0`` 的行；命中行保留
    （不删行），``filter_reason`` 首个原因 + ``filter_counter`` 同步累加。
    """
    stuck = yaw = 0
    for sample in samples:
        if float(sample.get("train_weight", 1.0)) <= 0.0:
            continue
        measured = np.asarray(
            sample.get("traj30_measured", sample.get("traj30")), dtype=np.float64
        )
        speed = float(np.linalg.norm(measured[-1, :2] - measured[0, :2]) / 2.5) if measured.size else 0.0
        dtheta = abs(float(np.asarray(sample["action"], dtype=np.float64)[0, 1]))
        reason = ""
        if speed < float(stuck_speed):
            reason, stuck = "stuck", stuck + 1
        elif dtheta > float(yaw_outlier):
            reason, yaw = "yaw_outlier", yaw + 1
        if not reason:
            continue
        sample["train_weight"] = 0.0
        if not str(sample.get("filter_reason", "")).strip():
            sample["filter_reason"] = reason
        if filter_counter is not None:
            filter_counter[reason] += 1
    active = sum(1 for sample in samples if float(sample.get("train_weight", 1.0)) > 0.0)
    return {
        "rows": int(len(samples)),
        "stuck": int(stuck),
        "yaw_outlier": int(yaw),
        "trainable_after": int(active),
        "thresholds": {"stuck_speed_mps": float(stuck_speed), "yaw_outlier_rad": float(yaw_outlier)},
    }


# --------------------------------------------------------------------------- #
# 单 spec / worker / 编排
# --------------------------------------------------------------------------- #

def _dagger_one_spec(
    spec: Any,
    *,
    spec_index: int,
    config: Dict[str, Any],
    builder: ObservationBuilder,
    interpolate_fn: Any,
) -> Dict[str, Any]:
    """策略驱动 + 专家空问采集单个 spec，返回与 ``collect_expert._collect_one_spec`` 同形状的记录。"""
    from pipeline.eval_runner import _CkptController  # 评测同款驱动（延迟导入：仅 worker 需要）

    env = None
    filter_counter: Counter = Counter()
    started = time.perf_counter()
    try:
        env = build_env(spec, traffic_density=config.get("traffic_density"), use_render=False)
        controller = _CkptController(env, spec, dict(config["task"]))
        episode = _dagger_episode(
            env,
            spec,
            controller=controller,
            labeler_factory=lambda: _make_labeler(env, spec, str(config["labeler"])),
            builder=builder,
            max_steps=int(config["max_steps"]),
        )
        rows = extract_samples(
            episode,
            spec,
            episode_id=int(spec_index),
            label_order=config["label_order"],
            interpolate_fn=interpolate_fn,
            on_lane_frac=float(config["on_lane_frac"]),
            on_lane_margin=float(config["on_lane_margin"]),
            # 空问标签的窗口不是实测轨迹 → round-trip 过滤不适用（阈值 ∞，诊断字段重标为 NaN）
            roundtrip_key_mean=float("inf"),
            roundtrip_key_max=float("inf"),
            filter_counter=filter_counter,
            require_dense=False,
            roundtrip_dense_mean=float("inf"),
        )
        _relabel_rows_with_expert(rows, episode, interpolate_fn=interpolate_fn)
        pathological = _apply_pathological_filters(
            rows,
            stuck_speed=float(config["stuck_speed"]),
            yaw_outlier=float(config["yaw_outlier"]),
            filter_counter=filter_counter,
        )
        candidates = len(episode["frames"])
        trainable = int(sum(1 for row in rows if row["train_weight"] > 0.0))
        return {
            "spec_index": int(spec_index),
            "kept": rows,
            "filter_counts": filter_counter,
            "candidates": candidates,
            "steps": int(episode["steps"]),
            "elapsed_s": time.perf_counter() - started,
            "pathological": pathological,
            "expert_params": episode.get("labeler_params") or {},
            "report": {
                "id": int(getattr(spec, "id", -1)),
                "seed": int(getattr(spec, "seed", -1)),
                "difficulty": str(getattr(spec, "difficulty", "unknown")),
                "geometry": str(getattr(spec, "labels", {}).get("geometry", "unknown")),
                "termination": episode["termination"],
                "env_steps": int(episode["steps"]),
                "candidate_policy_steps": candidates,
                "stored_rows": len(rows),
                "retained_steps": trainable,
                "step_yield": (trainable / candidates) if candidates else 0.0,
                "pathological": pathological,
                "expert_params": episode.get("labeler_params") or {},
            },
        }
    except Exception as exc:  # noqa: BLE001 - 单条失败不中断整批（与 collect_expert 同口径）
        return {
            "spec_index": int(spec_index),
            "kept": [],
            "filter_counts": filter_counter,
            "candidates": 0,
            "steps": 0,
            "elapsed_s": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "report": {
                "id": int(getattr(spec, "id", -1)),
                "termination": "error",
                "error": f"{type(exc).__name__}: {exc}",
            },
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


def _dagger_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """spawn 任务：顺序采集一组 spec，写一个分片，只回传标量摘要（见 ``_summarize_records``）。"""
    ensure_gl_library_path()
    builder = ObservationBuilder({})
    interpolate_fn, _ = resolve_interpolate()
    records = []
    for spec_index, spec in task["specs"]:
        record = _dagger_one_spec(
            spec,
            spec_index=int(spec_index),
            config=dict(task["config"]),
            builder=builder,
            interpolate_fn=interpolate_fn,
        )
        records.append(record)
        print(
            f"[dagger] [w{int(task['slot'])}] spec#{int(spec_index)} {_spec_progress_line(record)}",
            flush=True,
        )
    return _summarize_records(
        int(task["slot"]), int(task["slot"]), records, Path(task["shard_dir"])
    )


def _chunk_tasks(specs: Sequence[Any], chunk_size: int, *, workers: int = 1) -> List[List[Tuple[int, Any]]]:
    """按 ``chunk_size`` 切任务块（每块一个 spawn 任务：≤chunk_size 个 spec，落一个分片）。

    块大小还会按 ``ceil(specs/workers)`` 收紧，保证小批量（specs < chunk_size）也能铺满多核。
    """
    per_worker = max(1, math.ceil(len(specs) / max(1, int(workers))))
    size = max(1, min(int(chunk_size), per_worker))
    indexed = [(index, spec) for index, spec in enumerate(specs)]
    return [indexed[start : start + size] for start in range(0, len(indexed), size)]


def _run_dagger_workers(
    specs: Sequence[Any], *, workers: int, config: Dict[str, Any], shard_dir: Path
) -> List[Dict[str, Any]]:
    """spawn 多核采集：静态分块 + ``Pool``（``maxtasksperchild`` 定期回收进程控 RSS）。"""
    from multiprocessing import get_context

    ensure_gl_library_path()
    chunks = _chunk_tasks(specs, CHUNK_SPECS, workers=max(1, int(workers)))
    tasks = [
        {"slot": index, "specs": chunk, "config": config, "shard_dir": str(shard_dir)}
        for index, chunk in enumerate(chunks)
    ]
    if int(workers) <= 1:
        return [_dagger_task(task) for task in tasks]
    recycle = max(1, int(RECYCLE_EVERY_SPECS) // max(1, CHUNK_SPECS))
    with get_context("spawn").Pool(processes=int(workers), maxtasksperchild=recycle) as pool:
        return list(pool.map(_dagger_task, tasks, chunksize=1))


# --------------------------------------------------------------------------- #
# 场景池选择
# --------------------------------------------------------------------------- #

def _latest_eval_csv() -> Optional[Path]:
    """最新 ``runs/*eval500*/episodes.csv``（回退 ``runs/*eval*/episodes.csv``）。"""
    for pattern in ("*eval500*/episodes.csv", "*eval*/episodes.csv"):
        candidates = sorted(
            (Path("runs")).glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True
        )
        if candidates:
            return candidates[0]
    return None


def _spec_source_from_eval_run(csv_path: Path) -> str:
    """从 eval run 的 ``metrics.json::meta.specs_path`` 解析 spec 源（缺省回退 eval500 冻结集）。"""
    metrics = csv_path.parent / "metrics.json"
    if metrics.is_file():
        try:
            meta = dict(json.loads(metrics.read_text(encoding="utf-8")).get("meta") or {})
            recorded = str(meta.get("specs_path") or "")
            if recorded and Path(recorded).is_file():
                return recorded
        except Exception:  # noqa: BLE001 - 报告损坏不阻塞：退回默认源
            pass
    return DEFAULT_SPEC_SOURCE


def _load_failed_ids(csv_path: Path) -> Tuple[set, int]:
    """读 eval ``episodes.csv`` → ``(失败 spec id 集合, 总条数)``；兼容 ``id``/``spec_id`` 列。"""
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        id_key = "spec_id" if "spec_id" in (reader.fieldnames or []) else "id"
        failed, total = set(), 0
        for row in reader:
            total += 1
            if str(row.get("success", "")).strip().lower() in ("false", "0", "no"):
                failed.add(str(row.get(id_key, "")).strip())
    return failed, total


def _select_specs(args: argparse.Namespace) -> Tuple[List[Any], Dict[str, Any]]:
    """场景池：``--specs`` 直给，否则 ``--from-eval``（默认最新 eval500 失败清单）。"""
    provenance: Dict[str, Any] = {"specs": str(args.specs or ""), "from_eval": str(args.from_eval or "")}
    if args.specs:
        specs = list(load_specs(str(args.specs)))
        provenance.update({"source": f"specs:{args.specs}", "spec_source": str(args.specs)})
    else:
        csv_path = Path(args.from_eval) if args.from_eval else _latest_eval_csv()
        if csv_path is None or not Path(csv_path).is_file():
            raise SystemExit(
                "[dagger] 需要 --specs 或 --from-eval（默认最新 runs/*eval500*/episodes.csv）；"
                "未找到任何 eval 失败清单"
            )
        if not Path(csv_path).is_file():
            raise SystemExit(f"[dagger] 失败清单不存在：{csv_path}")
        spec_source = str(args.spec_source or _spec_source_from_eval_run(Path(csv_path)))
        failed, total = _load_failed_ids(Path(csv_path))
        specs = [spec for spec in load_specs(spec_source) if str(getattr(spec, "id", "")) in failed]
        provenance.update(
            {
                "source": f"from_eval:{csv_path}",
                "from_eval": str(csv_path),
                "eval_rows": int(total),
                "failed_ids": int(len(failed)),
                "spec_source": spec_source,
                "matched_specs": int(len(specs)),
            }
        )
    if int(args.limit or 0):
        specs = specs[: int(args.limit)]
    if not specs:
        raise SystemExit("[dagger] 场景池为空（检查 --specs/--from-eval/spec 源/--limit）")
    provenance["poll_specs"] = int(len(specs))
    print(
        f"[dagger] 场景池 {provenance.get('source')} → {len(specs)} spec"
        f"（spec 源 {provenance.get('spec_source')}）",
        flush=True,
    )
    return specs, provenance


# --------------------------------------------------------------------------- #
# 报告 / meta
# --------------------------------------------------------------------------- #

def _build_report(
    *,
    entries: Sequence[Dict[str, Any]],
    arrays: Dict[str, np.ndarray],
    balance: Dict[str, Any],
    filter_counter: Counter,
    config: Dict[str, Any],
    provenance: Dict[str, Any],
    label_order: Sequence[str],
    kinematics_source: str,
    elapsed_s: float,
) -> Dict[str, Any]:
    """与 collect_expert 同键的 report（计数/加权双口径 + dataset_gate），外加 DAgger 协议块。"""
    train_weight = np.asarray(arrays["train_weight"], dtype=np.float64)
    effective = train_weight * np.asarray(arrays["balance_weight"], dtype=np.float64)
    positive = np.asarray(arrays["labels"]) > 0.5
    gate = train_weight > 0.0
    per_label_count = {
        str(name): int(np.count_nonzero(positive[:, index] & gate))
        for index, name in enumerate(label_order)
    }
    per_label_weighted = {
        str(name): float((positive[:, index] * effective).sum())
        for index, name in enumerate(label_order)
    }
    total_candidates = int(sum(int(entry.get("candidates", 0)) for entry in entries))
    stored_rows = int(train_weight.shape[0])
    trainable_rows = int(gate.sum())
    row_yield = (trainable_rows / total_candidates) if total_candidates else 0.0
    weighted_yield = (float(effective.sum()) / total_candidates) if total_candidates else 0.0
    below_min = sorted(
        name for name, count in per_label_count.items() if count < DEFAULT_MIN_SAMPLES_PER_CATEGORY
    )
    balance_stats = dict(balance["stats"])
    return {
        "specs": str(config.get("specs_path", "")),
        "expert": str(config.get("labeler", "")),
        "expert_query": "act_empty（不注册 engine、只计算不执行）",
        "roll_in": True,
        "protocol": "dagger-lite: student(policy) roll-in + expert act() empty-query labels",
        "provenance": provenance,
        "kinematics_source": str(kinematics_source),
        "elapsed_s": round(float(elapsed_s), 3),
        "total_candidate_steps": total_candidates,
        "stored_rows": stored_rows,
        "retained_steps": trainable_rows,
        "bc_retained_step_yield": float(row_yield),
        "per_label_positive": per_label_count,
        "counts": {
            "convention": "行数（train_weight>0 的可训练行）",
            "candidate_steps": total_candidates,
            "stored_rows": stored_rows,
            "trainable_rows": trainable_rows,
            "zero_weight_rows": int(stored_rows - trainable_rows),
            "step_yield": float(row_yield),
            "per_label_positive": per_label_count,
        },
        "weighted": {
            "convention": "权重和（train_weight*sample_weight）",
            "trainable_weight_sum": float(effective.sum()),
            "step_yield_weighted": float(weighted_yield),
            "per_label_positive_weighted": per_label_weighted,
        },
        "filter_counts": dict(filter_counter),
        "min_samples_per_category": DEFAULT_MIN_SAMPLES_PER_CATEGORY,
        "labels_below_min": below_min,
        "balance": {
            "mode": balance_stats["mode"],
            "ratio": balance_stats["ratio"],
            "group_counts_before": balance_stats["group_counts_before"],
            "group_counts_after": balance_stats["group_counts_after"],
            "group_trainable_before": balance_stats["group_trainable_before"],
            "counts": balance_stats["counts"],
            **({"weight_range": balance_stats["weight_range"]} if "weight_range" in balance_stats else {}),
        },
        "per_spec": [entry["report"] for entry in entries],
        "config": dict(config),
        "dataset_gate": {
            "bc_retained_step_yield": {
                "value": float(row_yield),
                "min": DEFAULT_MIN_YIELD,
                "pass": bool(row_yield >= DEFAULT_MIN_YIELD),
                "convention": "行数口径：train_weight>0 行数 / 候选行数",
            },
            "min_samples_per_category": {
                "min": DEFAULT_MIN_SAMPLES_PER_CATEGORY,
                "labels_below_min": below_min,
                "pass": bool(not below_min),
                "convention": "行数口径（保守）；加权口径见 report.weighted",
            },
        },
    }


def _augment_meta(meta_path: Path, dagger: Dict[str, Any]) -> None:
    """把 ``dagger`` 块写入 ``expert_bc.meta.json``（保留 collect_expert 的 schema/meta 字段）。"""
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    meta["created_by"] = "tools/dagger_collect.py"
    meta["dagger"] = dagger
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/dagger_collect.py",
        description="DAgger-lite 采集（策略驱动 roll-in + 规则专家空问标签）",
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="学生策略权重（Stage B；驱动闭环）")
    parser.add_argument("--out", type=Path, required=True, help="输出数据集目录（datasets/BTC<ts>_daggerN）")
    parser.add_argument("--specs", type=str, default="", help="spec 池 json（与 --from-eval 二选一）")
    parser.add_argument("--from-eval", type=str, default="",
                        help="失败清单 episodes.csv（默认最新 runs/*eval500*/episodes.csv）")
    parser.add_argument("--spec-source", type=str, default="",
                        help=f"--from-eval 的 spec 源（默认读 eval run metrics.json::meta.specs_path；"
                             f"回退 {DEFAULT_SPEC_SOURCE}）")
    parser.add_argument("--labeler", choices=("pure_pursuit", "idm"), default=DEFAULT_LABELER,
                        help="空问专家（默认 pure_pursuit；idm = MetaDrive 原始 IDMPolicy，动作语义不同）")
    parser.add_argument("--tracker", choices=("lqr", "exact"), default="lqr",
                        help="学生动作执行器（默认 lqr = 评测闭环协议；exact = 运动学精确执行）")
    parser.add_argument("--workers", type=int, default=4, help="spawn worker 数（多核；默认 4）")
    parser.add_argument("--max-steps", type=int, default=600, help="单 episode 最大 env step（默认 600）")
    parser.add_argument("--limit", type=int, default=0, help="调试：spec 数上限（0=全部）")
    parser.add_argument("--device", type=str, default=None, help="策略运行设备（默认取 config train.device）")
    parser.add_argument("--model-config", type=str, default="config/model.yaml")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--balance", choices=("weights", "cap", "none"), default="weights",
                        help="(difficulty, geometry) 组配平模式（默认 weights）")
    parser.add_argument("--balance-ratio", type=float, default=3.0, help="cap 模式最大组/最小组样本比")
    parser.add_argument("--traffic-density", type=float, default=None, help="覆盖 build_env 的 traffic_density")
    parser.add_argument("--keep-shards", action="store_true", help="保留 _shards 分片目录（默认清理）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    started = time.perf_counter()
    from pipeline.eval_runner import load_config as load_eval_config
    from pipeline.hard_mining import ckpt_identity
    from pipeline.trainer import resolve_device

    config = load_eval_config(args.config)
    model_config = {
        "hidden_dim": config.get("hidden_dim", 128),
        "moe": dict(config.get("moe") or {}),
        "world_model": dict(config.get("world_model") or {}),
    }
    env_cfg = dict(config.get("env") or {})
    device = resolve_device(args.device, config)
    task = {
        "ckpt": str(args.ckpt),
        "model_config": model_config,
        "obs_config": dict(env_cfg.get("obs") or {}),
        "tracker_config": dict(env_cfg.get("tracking") or {}),
        "device": device,
        "tracker": str(args.tracker),
    }
    label_order = tuple(load_supervised_labels(str(args.model_config)))  # 8 维监督顺序（与训练/BC 一致）
    interpolate_fn, kinematics_source = resolve_interpolate()
    driver = ckpt_identity(str(args.ckpt))
    if not driver.get("sha256"):
        print(f"[dagger] 警告：ckpt 不存在/不可读：{args.ckpt}", file=sys.stderr)
    specs, provenance = _select_specs(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec_path = out / "dagger_specs.json"
    save_specs(specs, spec_path)
    print(f"[dagger] 学生={args.ckpt}（sha256={str(driver.get('sha256'))[:12]}，tracker={args.tracker}，"
          f"device={device}）· 专家={args.labeler}(empty-query) · spec 清单 → {spec_path}", flush=True)

    worker_config = {
        "labeler": str(args.labeler),
        "max_steps": int(args.max_steps),
        "on_lane_frac": 0.5,
        "on_lane_margin": 0.3,
        "traffic_density": args.traffic_density,
        "label_order": label_order,
        "task": task,
        "stuck_speed": STUCK_SPEED_MPS,
        "yaw_outlier": YAW_OUTLIER_RAD,
        "specs_path": str(provenance.get("spec_source", args.specs or "")),
    }
    shard_dir = out / SHARD_DIRNAME
    if shard_dir.exists():
        shutil.rmtree(shard_dir, ignore_errors=True)
    summaries = _run_dagger_workers(
        specs, workers=max(1, int(args.workers)), config=worker_config, shard_dir=shard_dir
    )
    entries, stored_rows, missing = _spec_entries(summaries, len(specs))
    if missing:
        print(f"[dagger] 警告：{len(missing)} 条 spec 无产出（worker 崩溃）：{missing[:8]}", flush=True)
    if stored_rows == 0:
        print("[dagger] 未产出任何行；检查策略/专家/过滤阈值", file=sys.stderr)
        return 1
    arrays = _concat_shards(shard_dir, _shard_segments(entries), stored_rows)
    expected_episode = np.concatenate(
        [np.full(int(entry["rows"]), int(entry["spec_index"]), dtype=np.int64) for entry in entries]
    )
    if "episode_id" not in arrays or not np.array_equal(arrays["episode_id"], expected_episode):
        raise ValueError("分片拼接行序与 spec 下标不一致（分片写入顺序被破坏）")
    del expected_episode
    balance = balance_from_specs(
        entries, arrays["train_weight"], mode=str(args.balance), ratio=float(args.balance_ratio)
    )
    arrays["sample_weight"] = np.asarray(balance["sample_weight"], dtype=np.float32)
    arrays["balance_weight"] = np.asarray(balance["sample_weight"], dtype=np.float32)
    arrays["balance_group"] = np.asarray(balance["balance_group"], dtype="U64")
    filter_counter = _merge_filter_counts(entries)
    report = _build_report(
        entries=entries,
        arrays=arrays,
        balance=balance,
        filter_counter=filter_counter,
        config=worker_config,
        provenance=provenance,
        label_order=label_order,
        kinematics_source=kinematics_source,
        elapsed_s=time.perf_counter() - started,
    )
    pathological_total = {
        key: int(sum(int((entry["report"].get("pathological") or {}).get(key, 0)) for entry in entries))
        for key in ("stuck", "yaw_outlier")
    }
    labeler_params = next(
        (entry["report"].get("expert_params") for entry in entries if entry["report"].get("expert_params")),
        None,
    )
    dagger = {
        "protocol": "dagger-lite: student(policy) roll-in + expert act() empty-query labels",
        "roll_in": True,
        "driver_ckpt": driver,
        "tracker": str(args.tracker),
        "labeler": str(args.labeler),
        "labeler_params": labeler_params,
        "labeler_action_space": (
            "pure_pursuit: 归一化 [steer, throttle] ∈ [-1,1]（本项目动作口径）"
            if str(args.labeler) == "pure_pursuit"
            else "idm: MetaDrive 原始 IDMPolicy 动作（IDM 加速度语义，换算前 clip 到 [-1,1]，仅冒烟对照）"
        ),
        "label_scope": "first_step_only",
        "window_fill": "constant_repeat(first_step)",
        "traj_source": f"{kinematics_source} over repeated first step",
        "action_conversion": (
            "inverse of pipeline.trainer.expand_policy_action: "
            "dtheta=tan(steer*max_steer_rad)*max(v,0.5)/wheelbase*dt; ds=max(v+throttle*scale,0)*dt"
        ),
        "spec_source": provenance,
        "spec_ids": [int(getattr(spec, "id", -1)) for spec in specs],
        "workers": int(args.workers),
        "max_steps": int(args.max_steps),
        "rows": int(stored_rows),
        "episodes": int(len([entry for entry in entries if int(entry["rows"]) > 0])),
        "filter_counts": dict(filter_counter),
        "pathological": pathological_total,
        "balance": report["balance"],
        "dataset_gate": report["dataset_gate"],
        "elapsed_s": round(time.perf_counter() - started, 3),
    }
    report["dagger"] = dagger
    paths = _write_dataset(
        out,
        arrays,
        label_order=label_order,
        builder=ObservationBuilder({}),
        config=worker_config,
        report=report,
        num_slots=int(arrays["od"].shape[1]) if "od" in arrays else 16,
    )
    _augment_meta(Path(paths["meta"]), dagger)
    (out / "dagger.json").write_text(json.dumps(dagger, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.keep_shards:
        print(f"[dagger] 保留分片目录 {shard_dir}（--keep-shards）", flush=True)
    else:
        shutil.rmtree(shard_dir, ignore_errors=True)
    print(
        f"[dagger] retained-step yield = {report['bc_retained_step_yield']:.3f} "
        f"({report['retained_steps']} trainable / {stored_rows} stored / {report['total_candidate_steps']} candidates) "
        f"gate>= {DEFAULT_MIN_YIELD}: {'PASS' if report['dataset_gate']['bc_retained_step_yield']['pass'] else 'FAIL'}",
        flush=True,
    )
    print(f"[dagger] filter counts (rows): {dict(filter_counter)}", flush=True)
    print(f"[dagger] pathological: {pathological_total}（阈值 {STUCK_SPEED_MPS} m/s / {YAW_OUTLIER_RAD} rad）", flush=True)
    print(f"[dagger] DONE → {out}（rows={stored_rows}，episodes={dagger['episodes']}，"
          f"elapsed={dagger['elapsed_s']:.1f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
