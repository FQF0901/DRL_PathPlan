#!/usr/bin/env python3
"""DAgger-lite 采集（lane U5 v2）：**策略驱动 roll-in + 规则专家"空问"标签 + 失败窗口**。

协议（用户定稿，2026-09-28）
----------------------------
0. **数据隔离（事故防复发）**：**只有 train spec 允许生成训练数据** —— ``--specs`` 必填，
   载入后硬校验池 ``(id, seed)`` 与 ``env/specs/scenarios_eval500.json`` / ``scenarios_val.json``
   的交集为空（命中 → ``SystemExit``；provenance 记录池路径 + sha256 + 校验结果 + 池大小）。
   **``--from-eval`` 路径已删除**（v1 曾用 eval500 失败清单当池 → train-on-test 泄漏，数据已删）。
1. **场景池**：``--specs <train spec json>``（必填）；``--limit`` 调试截断。
2. **roll-in = 学生策略驱动**（``--ckpt``，与评测同协议）：复用
   ``pipeline.eval_runner._CkptController``（``--tracker lqr`` 默认 = ``engine.add_policy(LqrTracker)``
   + 每 0.5 s ``set_reference(plan)``；``--tracker exact`` = 运动学精确执行），观测链 =
   ``env.obs.builder.ObservationBuilder``（与 collect_expert/eval 同源）。**访问状态 = 学生自己踩到的状态**，
   不是专家状态分布（旧实现的差异点）。
3. **窗口模式（stage B phase 2b 迭代）**：``--window-fail-before <秒>``（>0 生效）——
   **只保留失败 episode**（``--fail-terminations`` 默认 ``collision,out_of_road,terminal``；
   ``max_step``（工具截断/env timeout）**不算失败**）的"终止前 N 秒"窗口行
   （``ceil(秒/0.5)`` 个策略帧，含终止帧；不足则全取）；成功/``max_step`` episode **零行**（只记统计）。
   窗口模式**自动关闭 terminal_window 过滤**，且默认**关闭 on_lane 与 yaw_outlier 过滤**
   （恢复训练需要偏移/大修正样本；``--window-strict-filters`` 恢复 v1 行为）；
   report 记录窗口秒数/失败类型/窗口帧数分布（min/p50/max）/各过滤"本会被拦掉"的计数。
4. **标签 = 专家空问**：每个策略步对**当前状态**调用规则专家 ``act()``（默认
   ``env.expert.pure_pursuit_idm.PurePursuitIDMPolicy``；不注册进 engine —— 只计算不执行，
   策略是纯状态函数），首步动作换算成 BC 口径 ``(ds, dθ)``。
5. **动作换算链（已核对，不复制既有逻辑）**：
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
6. **过滤**：``extract_samples`` 的 on-lane / terminal_window / cut_label_unverified（round-trip
   过滤对"空问标签"不适用 → 阈值置 ∞ 并记录），外加 lane T 病态规则 ``stuck``（< 0.5 m/s）/
   ``yaw_outlier``（|dθ| > 0.35 rad/0.5 s）；逐行 ``train_weight`` 置 0（不删行），统计进
   report/meta；窗口模式按 §3 松弛（stuck 恒生效）。
7. **吞吐剖分**：worker 内累计 ``timing``（env 重建 / 学生步进 / 专家查询 / 行提取 / 分片写盘 / 其它），
   report 新增 ``timing`` 段（总计 + 每 spec 均值）。
8. **产物**（与 ``tools/collect_expert.py`` 同 schema 的**完整 episode** 数据）：
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
        --specs env/specs/scenarios_train_slice200.json --limit 5 --max-steps 60 \\
        --window-fail-before 10 --workers 1 --device cpu --out /tmp/opencode/dagger_smoke
"""

from __future__ import annotations

import argparse
import hashlib
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
    WINDOW_STEPS,
    _concat_shards,
    _ego_pose,
    _events_summary,
    _flag_dict,
    _merge_filter_counts,
    _on_lane,
    _on_lane_ok,
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
#: ``traj30`` 的 env-step 跨度（30 个 0.1 s 点）；episode 末帧附近该字段是补齐值（无观测）
#: → ``distance_to_end < TRAJ30_SPAN`` 的行跳过 ``stuck`` 判决（lane U6；实测边界 25，保守取 30）。
TRAJ30_SPAN = int(WINDOW_STEPS)
#: 默认 labeler / 每 worker 一个任务的 spec 数（分片粒度）
DEFAULT_LABELER = "pure_pursuit"
CHUNK_SPECS = 10
#: eval/val 冻结集（训练数据**禁止**取自这里；隔离守卫的唯一口径）
EVAL_SPEC_SOURCES: Tuple[str, ...] = ("env/specs/scenarios_eval500.json", "env/specs/scenarios_val.json")
#: 失败终止类型默认值（``max_step``（工具截断/env timeout）不算失败）
DEFAULT_FAIL_TERMINATIONS: Tuple[str, ...] = ("collision", "out_of_road", "terminal")
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
    timing: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """跑一个**策略驱动** episode，逐 env-step 记位姿/标记 + 逐策略步记学生 obs 与专家空问标签。

    记录结构与 ``tools.collect_expert.collect_episode`` 完全一致（``frames``/``poses``/``flags``/
    ``termination``/``steps``/``ended_by_env``），外加 ``expert_labels[step]``；
    ``extract_samples`` 因此可原样复用（过滤/权重/wm_valid/labels 全走同一实现）。

    ``labeler_factory`` 在 **env.reset() 之后**调用：规则专家构造依赖全局 engine
    （``BasePolicy.show_policy_mark`` → ``get_engine()``），reset 前建 env 会触发
    "Please initialize the environment first!"。
    """
    def _add(key: str, started_at: float) -> None:
        if timing is not None:
            timing[key] = float(timing.get(key, 0.0)) + (time.perf_counter() - started_at)

    _t = time.perf_counter()
    env.reset()
    controller.bind(env)
    _add("other_s", _t)
    _t = time.perf_counter()
    labeler = labeler_factory()  # 空问专家（不注册 engine）
    _add("other_s", _t)
    expert_labels: Dict[int, Dict[str, Any]] = {}
    frames: Dict[int, Dict[str, Any]] = {}
    poses: List[np.ndarray] = [_ego_pose(env.agent)]
    flags: List[Optional[Dict[str, Any]]] = [None]
    termination = "max_step"  # 工具自身 max_steps 截断（未到 env 终局）
    ended_by_env = False
    step = 0
    env.prev_policy_action = np.zeros(2, dtype=np.float64)
    while step < int(max_steps):
        _t = time.perf_counter()
        action = controller.action(env)  # 学生策略驱动（同时写 prev_policy_action + 建 obs）
        _add("student_step_s", _t)
        if step % STEPS_PER_POLICY == 0:
            _t = time.perf_counter()
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
            _add("other_s", _t)
            _t = time.perf_counter()
            expert_labels[step] = labeler.label(env.agent)  # 空问：当前状态 → 专家首步标签
            _add("expert_query_s", _t)
        _t = time.perf_counter()
        _, _, terminated, truncated, info = env.step(action)
        post_step = getattr(controller, "post_step", None)
        if callable(post_step):
            post_step(env)  # exact 跟踪器专有（lqr 无此方法）
        _add("student_step_s", _t)
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

def future_truncation_mask(
    rows: Sequence[Dict[str, Any]], episode: Mapping[str, Any], *, span: int = TRAJ30_SPAN
) -> np.ndarray:
    """逐行 ``future_truncated`` 掩码（lane U6）：``distance_to_end < span`` 为 True。

    - ``distance_to_end`` = episode **末帧 env step**（``len(poses)-1``，回退 ``episode["steps"]``）
      − 本行 ``step``（单位 env step）；
    - episode 末 ``span`` 个 env step 内的行，``traj30_measured`` 是补齐值（首末点距 0）→
      ``stuck`` 规则**无观测不判决**（由 :func:`_apply_pathological_filters` 跳过）；
    - 掩码**由 episode 侧显式计算并传入**（过滤器不猜）；返回与 ``rows`` 等长的 bool 数组。
    """
    poses = episode.get("poses") or []
    last_step = int(len(poses) - 1) if poses else int(episode.get("steps") or 0)
    return np.asarray(
        [int(row.get("step", 0)) > (last_step - int(span)) for row in rows], dtype=bool
    )


def _apply_pathological_filters(
    samples: Sequence[Dict[str, Any]],
    *,
    stuck_speed: float = STUCK_SPEED_MPS,
    yaw_outlier: float = YAW_OUTLIER_RAD,
    filter_counter: Optional[Counter] = None,
    future_truncated: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """病态行过滤（写死阈值，记录进 report/meta）：卡死 / 大转向离群 → ``train_weight=0``。

    速度 = ``traj30_measured``（缺失回退 ``traj30``）首末点距离 / 2.5 s（0.5 s × 5 段）；
    dθ = ``action[0, 1]``（专家首步标签）。只作用于 ``train_weight>0`` 的行；命中行保留
    （不删行），``filter_reason`` 首个原因 + ``filter_counter`` 同步累加。

    ``future_truncated``（lane U6；bool 数组，与 ``samples`` 等长；None = 无截断信息）：
    **为 True 的行跳过 ``stuck`` 判决**（末帧附近的 ``traj30_measured`` 是补齐值 → 无观测不判决；
    保持 ``train_weight``/``filter_reason`` 不变），单独计数 ``stuck_skipped_truncated``；
    ``yaw_outlier`` 规则与未来轨迹无关 → 不变。窗口/整段两种模式都生效。
    ``stuck_skipped_truncated`` = **本会命中 stuck 但被跳过**的截断行数（yaw 仍可能另行命中该行）。
    """
    flags = None
    if future_truncated is not None:
        flags = np.asarray(future_truncated, dtype=bool).reshape(-1)
        if flags.size != len(samples):
            raise ValueError(
                f"future_truncated 长度 {flags.size} != 行数 {len(samples)}（掩码必须逐行对齐）"
            )
    stuck = yaw = skipped = 0
    for index, sample in enumerate(samples):
        if float(sample.get("train_weight", 1.0)) <= 0.0:
            continue
        measured = np.asarray(
            sample.get("traj30_measured", sample.get("traj30")), dtype=np.float64
        )
        speed = float(np.linalg.norm(measured[-1, :2] - measured[0, :2]) / 2.5) if measured.size else 0.0
        dtheta = abs(float(np.asarray(sample["action"], dtype=np.float64)[0, 1]))
        truncated = bool(flags[index]) if flags is not None else False
        reason = ""
        if speed < float(stuck_speed):
            if truncated:
                skipped += 1  # 无观测不判决：末帧附近 traj30 是补齐值
            else:
                reason, stuck = "stuck", stuck + 1
        if not reason and dtheta > float(yaw_outlier):
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
        "stuck_skipped_truncated": int(skipped),
        "future_truncated_rows": int(np.count_nonzero(flags)) if flags is not None else 0,
        "trainable_after": int(active),
        "traj30_span_steps": int(TRAJ30_SPAN),
        "thresholds": {"stuck_speed_mps": float(stuck_speed), "yaw_outlier_rad": float(yaw_outlier)},
    }


def _final_filter_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    """窗口模式：``filter_counts`` 只统计**最终被置零**的行（lane U7）。

    被窗口丢弃的行（不在 ``rows`` 里）与"松弛保留"的 ``terminal_window`` 行都不计入；
    逐行取 ``filter_reason``（置零行必有原因）。``would_filter`` 仍是独立诊断字段
    （"本会被过滤"的计数，语义不变）。
    """
    counter: Counter = Counter()
    for row in rows:
        if float(row.get("train_weight", 1.0)) <= 0.0:
            reason = str(row.get("filter_reason", "")).strip()
            if reason:
                counter[reason] += 1
    return counter


# --------------------------------------------------------------------------- #
# 失败窗口选择（lane U5：stage B phase 2b 迭代协议）
# --------------------------------------------------------------------------- #

def _window_mode(config: Mapping[str, Any]) -> bool:
    """窗口模式是否开启（``--window-fail-before`` > 0）。"""
    return float(config.get("window_fail_before") or 0.0) > 0.0


def _effective_on_lane_frac(config: Mapping[str, Any]) -> float:
    """on_lane 阈值：窗口模式默认松弛（``inf``），``--window-strict-filters`` 恢复 v1 值。"""
    if _window_mode(config) and not bool(config.get("window_strict_filters")):
        return float("inf")
    return float(config.get("on_lane_frac", 0.5))


def _effective_yaw_threshold(config: Mapping[str, Any]) -> float:
    """yaw_outlier 阈值：窗口模式默认松弛（``inf``），``--window-strict-filters`` 恢复 v1 值。"""
    if _window_mode(config) and not bool(config.get("window_strict_filters")):
        return float("inf")
    return float(config.get("yaw_outlier", YAW_OUTLIER_RAD))


def _relax_episode_on_lane(episode: Mapping[str, Any]) -> int:
    """窗口模式：把 episode 帧的 on_lane 视为通过（原值存 ``_on_lane_raw`` 供"本会被拦掉"统计）。

    ``extract_samples`` 的 ``not frame["on_lane"]`` 分支不看阈值 → 必须从帧侧松弛；
    返回被松弛的帧数。**仅在窗口模式且非 strict 时调用。**
    """
    relaxed = 0
    for frame in (episode.get("frames") or {}).values():
        raw = bool(frame.get("on_lane", True))
        frame["_on_lane_raw"] = raw
        if not raw:
            frame["on_lane"] = True
            relaxed += 1
    return relaxed

def _window_select_rows(
    rows: Sequence[Dict[str, Any]],
    episode: Mapping[str, Any],
    *,
    window_s: float,
    fail_terminations: Sequence[str],
    strict_filters: bool,
    on_lane_frac: float = 0.5,
    on_lane_margin: float = 0.3,
    filter_counter: Optional[Counter] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """窗口模式：失败 episode 只保留"终止前 ``ceil(window_s/0.5)`` 个策略帧"的行；成功/``max_step`` → 零行。

    - 行序 = 策略帧升序（``extract_samples`` 按 step 排序）→ 尾部切片即窗口（含终止帧）；
    - **terminal_window 松弛**：窗口行必然"跨终止"→ 恢复 ``train_weight=1.0`` 并从
      ``filter_counter`` 扣回（窗口模式自动关闭该过滤）；
    - ``stats["would_filter"]`` 记录各过滤"本会被拦掉"的计数（on_lane 用 episode 帧按 v1 阈值复算；
      yaw_outlier 按 ``|action[0,1]| > 阈值`` 复算；terminal_window = 实际松弛数）。
    """
    termination = str(episode.get("termination") or "")
    fails = {str(item).strip().lower() for item in fail_terminations}
    frames_per = max(1, int(math.ceil(float(window_s) / POLICY_DT)))
    stats: Dict[str, Any] = {
        "applied": True,
        "window_s": float(window_s),
        "frames_per_window": int(frames_per),
        "fail_terminations": sorted(fails),
        "strict_filters": bool(strict_filters),
        "termination": termination,
        "is_failure": termination.lower() in fails,
    }
    if termination.lower() not in fails:
        stats.update({"kept_rows": 0, "dropped_rows": int(len(rows)), "window_frames": 0, "reason": "not_failure"})
        return [], stats
    kept = list(rows[-frames_per:]) if rows else []
    stats.update({
        "kept_rows": int(len(kept)),
        "dropped_rows": int(len(rows) - len(kept)),
        "window_frames": int(len(kept)),
        "reason": "failure_window",
    })
    relaxed_terminal = 0
    for row in kept:
        if str(row.get("filter_reason", "")) == "terminal_window":
            row["train_weight"] = 1.0
            row["filter_reason"] = ""
            relaxed_terminal += 1
    if filter_counter is not None and relaxed_terminal:
        remaining = int(filter_counter.get("terminal_window", 0)) - int(relaxed_terminal)
        if remaining > 0:
            filter_counter["terminal_window"] = remaining
        else:
            filter_counter.pop("terminal_window", None)
    frames = episode.get("frames") or {}
    would_on_lane = 0
    for row in kept:
        frame = frames.get(int(row.get("step", -1)))
        if frame is None:
            continue
        raw_on_lane = bool(frame.get("_on_lane_raw", frame.get("on_lane", True)))
        if (not raw_on_lane) or (
            not _on_lane_ok(
                float(frame.get("lane_lat", 0.0)),
                float(frame.get("lane_width", 0.0)),
                float(on_lane_frac),
                float(on_lane_margin),
            )
        ):
            would_on_lane += 1
    would_yaw = 0
    for row in kept:
        action = np.asarray(row.get("action", np.zeros((1, 2))), dtype=np.float64)
        if action.ndim == 2 and action.size and abs(float(action[0, 1])) > YAW_OUTLIER_RAD:
            would_yaw += 1
    stats["would_filter"] = {
        "terminal_window": int(relaxed_terminal),
        "on_lane": int(would_on_lane),
        "yaw_outlier": int(would_yaw),
    }
    return kept, stats


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
    timing: Dict[str, float] = {"env_build_s": 0.0, "student_step_s": 0.0, "expert_query_s": 0.0,
                                "extract_s": 0.0, "other_s": 0.0}
    window_s = float(config.get("window_fail_before") or 0.0)
    window_mode = window_s > 0.0
    strict = bool(config.get("window_strict_filters"))
    try:
        _t = time.perf_counter()
        env = build_env(spec, traffic_density=config.get("traffic_density"), use_render=False)
        timing["env_build_s"] += time.perf_counter() - _t
        controller = _CkptController(env, spec, dict(config["task"]))
        episode = _dagger_episode(
            env,
            spec,
            controller=controller,
            labeler_factory=lambda: _make_labeler(env, spec, str(config["labeler"])),
            builder=builder,
            max_steps=int(config["max_steps"]),
            timing=timing,
        )
        _t = time.perf_counter()
        if window_mode and not strict:
            _relax_episode_on_lane(episode)  # 窗口模式默认松弛 on_lane（帧侧）
        on_lane_frac = _effective_on_lane_frac(config)  # 窗口模式默认松弛 on_lane
        rows = extract_samples(
            episode,
            spec,
            episode_id=int(spec_index),
            label_order=config["label_order"],
            interpolate_fn=interpolate_fn,
            on_lane_frac=on_lane_frac,
            on_lane_margin=float(config["on_lane_margin"]),
            # 空问标签的窗口不是实测轨迹 → round-trip 过滤不适用（阈值 ∞，诊断字段重标为 NaN）
            roundtrip_key_mean=float("inf"),
            roundtrip_key_max=float("inf"),
            filter_counter=filter_counter,
            require_dense=False,
            roundtrip_dense_mean=float("inf"),
        )
        _relabel_rows_with_expert(rows, episode, interpolate_fn=interpolate_fn)
        window_stats: Optional[Dict[str, Any]] = None
        if window_mode:
            rows, window_stats = _window_select_rows(
                rows,
                episode,
                window_s=window_s,
                fail_terminations=config.get("fail_terminations") or DEFAULT_FAIL_TERMINATIONS,
                strict_filters=strict,
                on_lane_frac=float(config["on_lane_frac"]),
                on_lane_margin=float(config["on_lane_margin"]),
                filter_counter=filter_counter,
            )
        # lane U6：future_truncated 掩码（episode 侧显式计算；末帧附近 traj30 是补齐值）
        truncated_mask = future_truncation_mask(rows, episode)
        for row, truncated in zip(rows, truncated_mask.tolist()):
            row["future_truncated"] = bool(truncated)  # 逐行留痕（诊断；不进 schema）
        pathological = _apply_pathological_filters(
            rows,
            stuck_speed=float(config["stuck_speed"]),
            yaw_outlier=_effective_yaw_threshold(config),  # 窗口模式默认松弛 yaw_outlier
            filter_counter=filter_counter,
            future_truncated=truncated_mask,
        )
        if window_mode:
            # lane U7：filter_counts 只统计**最终被置零**的行 —— 被窗口丢弃的行与"松弛保留"的
            # terminal_window 行都不计入（would_filter 仍是独立诊断：本会被过滤的计数）。
            filter_counter = _final_filter_counts(rows)
        timing["extract_s"] += time.perf_counter() - _t
        candidates = len(episode["frames"])
        trainable = int(sum(1 for row in rows if row["train_weight"] > 0.0))
        timing["total_s"] = time.perf_counter() - started
        timing["other_s"] += max(
            0.0,
            timing["total_s"]
            - timing["env_build_s"] - timing["student_step_s"] - timing["expert_query_s"] - timing["extract_s"],
        )
        return {
            "spec_index": int(spec_index),
            "kept": rows,
            "filter_counts": filter_counter,
            "candidates": candidates,
            "steps": int(episode["steps"]),
            "elapsed_s": time.perf_counter() - started,
            "pathological": pathological,
            "timing": timing,
            "window": window_stats,
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
                "timing": dict(timing),
                "window": window_stats,
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
    _t = time.perf_counter()
    summary = _summarize_records(
        int(task["slot"]), int(task["slot"]), records, Path(task["shard_dir"])
    )
    summary["shard_write_s"] = float(time.perf_counter() - _t)
    return summary


def _chunk_tasks(specs: Sequence[Any], chunk_size: int, *, workers: int = 1) -> List[List[Tuple[int, Any]]]:
    """按 ``chunk_size`` 切任务块（每块一个 spawn 任务：≤chunk_size 个 spec，落一个分片）。

    块大小还会按 ``ceil(specs/workers)`` 收紧，保证小批量（specs < chunk_size）也能铺满多核。
    """
    per_worker = max(1, math.ceil(len(specs) / max(1, int(workers))))
    size = max(1, min(int(chunk_size), per_worker))
    indexed = [(index, spec) for index, spec in enumerate(specs)]
    return [indexed[start : start + size] for start in range(0, len(indexed), size)]


def _scan_progress(
    summaries: Sequence[Mapping[str, Any]],
    *,
    fail_terminations: Sequence[str],
    target_fails: int,
) -> Dict[str, Any]:
    """流式扫描进度（lane P3-A，纯函数）：已扫 spec 数 / 失败 episode 数 / 是否达目标。

    ``summaries`` = 已完成的 chunk 摘要（``_summarize_records`` 产物）；失败判定 = 该 spec 的
    ``report.termination`` ∈ ``fail_terminations``（``max_step``/``error`` 不计）。
    """
    fails = {str(item).strip().lower() for item in fail_terminations}
    scanned = 0
    fail_count = 0
    for summary in summaries:
        for entry in summary.get("specs", []) or []:
            scanned += 1
            termination = str((entry.get("report") or {}).get("termination", "")).strip().lower()
            if termination in fails:
                fail_count += 1
    target = int(target_fails or 0)
    return {
        "scanned_specs": int(scanned),
        "fail_count": int(fail_count),
        "target_fails": target,
        "target_reached": bool(target > 0 and fail_count >= target),
    }


def _run_dagger_workers(
    specs: Sequence[Any],
    *,
    workers: int,
    config: Dict[str, Any],
    shard_dir: Path,
    target_fails: int = 0,
    fail_terminations: Sequence[str] = (),
    logger: Any = print,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """spawn 多核采集：静态分块 + ``Pool``；**按 chunk 顺序流式扫描**，失败数达 ``target_fails``
    立即停止（不再跑后续 chunk；已完成 chunk 的分片保留）。返回 ``(summaries, scan)``。

    ``target_fails<=0`` = 扫完整个池（旧行为，``target_reached=False``）。
    """
    from multiprocessing import get_context

    ensure_gl_library_path()
    chunks = _chunk_tasks(specs, CHUNK_SPECS, workers=max(1, int(workers)))
    tasks = [
        {"slot": index, "specs": chunk, "config": config, "shard_dir": str(shard_dir)}
        for index, chunk in enumerate(chunks)
    ]
    summaries: List[Dict[str, Any]] = []
    if int(workers) <= 1:
        for task in tasks:
            summaries.append(_dagger_task(task))
            progress = _scan_progress(
                summaries, fail_terminations=fail_terminations, target_fails=target_fails
            )
            if progress["target_reached"]:
                break
    else:
        recycle = max(1, int(RECYCLE_EVERY_SPECS) // max(1, CHUNK_SPECS))
        pool = get_context("spawn").Pool(processes=int(workers), maxtasksperchild=recycle)
        try:
            for summary in pool.imap(_dagger_task, tasks, chunksize=1):  # 有序：与静态分块顺序一致
                summaries.append(summary)
                progress = _scan_progress(
                    summaries, fail_terminations=fail_terminations, target_fails=target_fails
                )
                if progress["target_reached"]:
                    break
        finally:
            pool.terminate()  # 达目标/异常：不再跑剩余 chunk（未完成 chunk 的分片不被合并）
            pool.join()
    scan = _scan_progress(
        summaries, fail_terminations=fail_terminations, target_fails=target_fails
    )
    logger(
        f"[dagger] 流式扫描：已扫 {scan['scanned_specs']} 条 spec · 失败 {scan['fail_count']} 个"
        f"（目标 {scan['target_fails']}，{'已达 → 提前收尾' if scan['target_reached'] else '未达/未启用'}）"
    )
    return summaries, scan


# --------------------------------------------------------------------------- #
# 场景池选择
# --------------------------------------------------------------------------- #

def spec_keys(specs: Sequence[Any]) -> set:
    """spec 列表 → ``{(id, seed)}`` 键集合（隔离守卫与 provenance 的唯一口径）。"""
    return {(int(getattr(spec, "id", -1)), int(getattr(spec, "seed", -1))) for spec in specs}


def eval_val_spec_keys(sources: Sequence[str] = EVAL_SPEC_SOURCES) -> Dict[str, Any]:
    """读 eval/val 冻结集的 ``(id, seed)`` 键（文件缺失 → 记 ``missing``，不报错）。"""
    out: Dict[str, Any] = {}
    for source in sources:
        path = Path(str(source))
        if not path.is_file():
            out[str(source)] = {"path": str(path), "specs": 0, "keys": set(), "missing": True}
            continue
        keys = spec_keys(list(load_specs(str(path))))
        out[str(source)] = {"path": str(path), "specs": len(keys), "keys": keys, "missing": False}
    return out


def assert_no_eval_val_overlap(
    specs: Sequence[Any], *, context: str, sources: Sequence[str] = EVAL_SPEC_SOURCES
) -> Dict[str, Any]:
    """隔离守卫（lane U5）：池 ``(id, seed)`` 与 eval500/val 交集必须为空，否则 ``SystemExit``。

    返回可进 provenance 的校验记录：``{context, pool_specs, checked: {源: {path, specs, overlap}},
    overlap}``（错误信息附交集样例 + 提示）。
    """
    pool = spec_keys(specs)
    record: Dict[str, Any] = {"context": str(context), "pool_specs": int(len(pool)), "checked": {}, "overlap": 0}
    overlaps: Dict[str, list] = {}
    for name, info in eval_val_spec_keys(sources).items():
        entry: Dict[str, Any] = {
            "path": info["path"], "specs": int(info["specs"]), "missing": bool(info["missing"]),
        }
        if not info["missing"]:
            inter = pool & info["keys"]
            entry["overlap"] = int(len(inter))
            if inter:
                overlaps[name] = sorted(inter)[:5]
        record["checked"][name] = entry
    record["overlap"] = int(sum(int(v.get("overlap", 0)) for v in record["checked"].values()))
    if overlaps:
        sample = "；".join(f"{name} → {keys}" for name, keys in overlaps.items())
        raise SystemExit(
            f"[dagger] 隔离守卫失败：场景池与 eval/val 冻结集有交集（{record['overlap']} 个 (id,seed)）：{sample}\n"
            "[dagger] 训练数据**只允许取自 train spec**（--specs env/specs/scenarios_train*.json）；"
            "eval500/val 仅用于评测（train-on-test 泄漏事故防复发）。"
        )
    return record


def dagger_spec_sources(dagger_dir: Any) -> List[str]:
    """从 dagger 目录的 ``report.json`` / ``expert_bc.meta.json`` 读 spec 来源（读不到 → 空列表）。

    用于训练侧防线（``pipeline.stages._merge_dagger_rows``）：能读到来源就必须过隔离守卫。
    """
    base = Path(str(dagger_dir))
    found: List[str] = []
    report_path = base / "report.json"
    if report_path.is_file():
        try:
            doc = json.loads(report_path.read_text(encoding="utf-8"))
            prov = dict(doc.get("provenance") or {})
            for value in (doc.get("specs"), prov.get("spec_source"), prov.get("specs")):
                text = str(value or "").strip()
                if text:
                    found.append(text)
        except Exception:  # noqa: BLE001 - 报告损坏 → 视为缺失（调用方警告）
            pass
    meta_path = base / "expert_bc.meta.json"
    if meta_path.is_file():
        try:
            doc = json.loads(meta_path.read_text(encoding="utf-8"))
            prov = dict((doc.get("dagger") or {}).get("spec_source") or {})
            for value in (prov.get("spec_source"), prov.get("specs")):
                text = str(value or "").strip()
                if text:
                    found.append(text)
        except Exception:  # noqa: BLE001
            pass
    seen, ordered = set(), []
    for item in found:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _looks_like_eval_or_val(path_text: str) -> bool:
    """路径名是否疑似 eval/val 集（文件缺失时的兜底判断；命中 → 拒绝）。"""
    name = Path(str(path_text)).name.lower()
    return ("eval" in name) or ("val" in name)


def _select_specs(args: argparse.Namespace) -> Tuple[List[Any], Dict[str, Any]]:
    """场景池：``--specs``（**必填**，只允许 train spec）→ 隔离守卫 → provenance。"""
    specs_path = str(getattr(args, "specs", "") or "").strip()
    if not specs_path:
        raise SystemExit(
            "[dagger] 需要 --specs <train spec json>（必填）——训练数据只允许取自 train spec；"
            "eval500/val 仅用于评测（v1 的 --from-eval 路径已删除）"
        )
    path = Path(specs_path)
    if not path.is_file():
        raise SystemExit(f"[dagger] spec 池不存在：{path}")
    specs = list(load_specs(str(path)))
    if not specs:
        raise SystemExit(f"[dagger] 场景池为空（检查 --specs {path}）")
    # 隔离守卫按**全池**校验（shuffle/limit 之前的加载集）
    guard = assert_no_eval_val_overlap(specs, context=f"pool:{path}")
    pool_specs = int(len(specs))
    # lane P3-A：先 shuffle 后截断（--limit 与 --target-fails 以先到者为准）
    shuffle_seed = int(getattr(args, "shuffle_seed", 0) or 0)
    if shuffle_seed > 0:
        order = np.random.default_rng(shuffle_seed).permutation(pool_specs)
        specs = [specs[int(index)] for index in order]
    limit = int(getattr(args, "limit", 0) or 0)
    if limit:
        specs = specs[:limit]
    provenance: Dict[str, Any] = {
        "source": f"specs:{path}",
        "spec_source": str(path),
        "specs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "pool_specs": pool_specs,
        "scan_pool_specs": int(len(specs)),
        "shuffle_seed": int(shuffle_seed),
        "limit": int(limit),
        "isolation_guard": guard,
    }
    print(
        f"[dagger] 场景池 {path} → {pool_specs} spec（sha256={provenance['specs_sha256'][:12]}；"
        f"隔离守卫：与 eval/val 交集 = {guard['overlap']} ✓；shuffle_seed={shuffle_seed} → "
        f"本次扫描上限 {len(specs)} 条）",
        flush=True,
    )
    return specs, provenance


# --------------------------------------------------------------------------- #
# 报告 / meta
# --------------------------------------------------------------------------- #

def _aggregate_timing(entries: Sequence[Dict[str, Any]], *, shard_write_s: float) -> Dict[str, Any]:
    """吞吐剖分汇总：逐 spec 计时 → 总计 + 每 spec 均值（env 重建/学生步进/专家查询/行提取/其它）。"""
    keys = ("env_build_s", "student_step_s", "expert_query_s", "extract_s", "other_s", "total_s")
    totals = {key: 0.0 for key in keys}
    specs = 0
    for entry in entries:
        timing = dict((entry.get("report") or {}).get("timing") or {})
        if not timing:
            continue
        specs += 1
        for key in keys:
            totals[key] += float(timing.get(key, 0.0))
    mean = {key: (value / specs if specs else 0.0) for key, value in totals.items()}
    return {
        "specs": int(specs),
        "shard_write_s": round(float(shard_write_s), 4),
        "total_s": {key: round(value, 4) for key, value in totals.items()},
        "mean_per_spec_s": {key: round(value, 4) for key, value in mean.items()},
    }


def _aggregate_window(entries: Sequence[Dict[str, Any]], *, window_s: float) -> Optional[Dict[str, Any]]:
    """窗口模式汇总：失败类型计数 / 窗口帧数分布（min/p50/max）/ 各过滤"本会被拦掉"的计数。"""
    if float(window_s or 0.0) <= 0.0:
        return None
    frames: List[int] = []
    terminations: Counter = Counter()
    would: Counter = Counter()
    kept_rows = dropped_rows = 0
    failures = 0
    for entry in entries:
        window = (entry.get("report") or {}).get("window")
        if not isinstance(window, dict):
            continue
        terminations[str(window.get("termination", "unknown"))] += 1
        if bool(window.get("is_failure")):
            failures += 1
            frames.append(int(window.get("window_frames", 0)))
        kept_rows += int(window.get("kept_rows", 0))
        dropped_rows += int(window.get("dropped_rows", 0))
        for key, value in dict(window.get("would_filter") or {}).items():
            would[str(key)] += int(value)
    frames_sorted = sorted(frames)
    def _pct(values: Sequence[int], q: float) -> int:
        if not values:
            return 0
        index = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
        return int(values[index])
    return {
        "window_s": float(window_s),
        "frames_per_window": int(math.ceil(float(window_s) / POLICY_DT)),
        "specs": int(len(terminations)),
        "failures": int(failures),
        "terminations": dict(terminations),
        "window_frames": {
            "min": int(frames_sorted[0]) if frames_sorted else 0,
            "p50": _pct(frames_sorted, 0.5),
            "max": int(frames_sorted[-1]) if frames_sorted else 0,
        },
        "kept_rows": int(kept_rows),
        "dropped_rows": int(dropped_rows),
        "would_filter": dict(would),
        "note": "窗口模式：失败 episode 保留终止前窗口行；成功/max_step 零行；terminal_window 恒松弛，"
                "on_lane/yaw_outlier 默认松弛（--window-strict-filters 恢复）",
    }


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
    shard_write_s: float = 0.0,
    scan: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """与 collect_expert 同键的 report（计数/加权双口径 + dataset_gate），外加 DAgger 协议块。

    ``scan``（lane P3-A）：流式扫描结果（``shuffle_seed``/``target_fails``/``scanned_specs``/
    ``fail_count``/``target_reached``）→ 顶层同名键 + ``scan`` 块（provenance 亦带 shuffle_seed）。
    """
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
    # lane U7：窗口模式的 yield 分母 = **窗口候选行数**（窗口内实际考虑的帧 = 存储行；含窗口内被
    # 真正置零的行）——分母若仍用整段 episode 候选帧（total_candidates）则比值失义、gate 误报。
    # 整段模式（非窗口）分母 = 整段候选帧（与修复前逐位一致）。
    window_mode = float(config.get("window_fail_before") or 0.0) > 0.0
    if window_mode:
        yield_denominator_name, yield_denominator = "window_rows", int(stored_rows)
    else:
        yield_denominator_name, yield_denominator = "candidates", int(total_candidates)
    row_yield = (trainable_rows / yield_denominator) if yield_denominator else 0.0
    weighted_yield = (float(effective.sum()) / yield_denominator) if yield_denominator else 0.0
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
        # lane P3-A：流式扫描（shuffle / target-fails）——顶层同名键便于消费
        "shuffle_seed": int((scan or {}).get("shuffle_seed", provenance.get("shuffle_seed", 0)) or 0),
        "target_fails": int((scan or {}).get("target_fails", 0) or 0),
        "scanned_specs": int((scan or {}).get("scanned_specs", 0) or 0),
        "fail_count": int((scan or {}).get("fail_count", 0) or 0),
        "target_reached": bool((scan or {}).get("target_reached", False)),
        "total_candidate_steps": total_candidates,
        "stored_rows": stored_rows,
        "retained_steps": trainable_rows,
        "bc_retained_step_yield": float(row_yield),
        "yield_denominator": yield_denominator_name,
        "yield_denominator_value": int(yield_denominator),
        "per_label_positive": per_label_count,
        "counts": {
            "convention": "行数（train_weight>0 的可训练行）",
            "candidate_steps": total_candidates,
            "stored_rows": stored_rows,
            "trainable_rows": trainable_rows,
            "zero_weight_rows": int(stored_rows - trainable_rows),
            "step_yield": float(row_yield),
            "yield_denominator": yield_denominator_name,
            "yield_denominator_value": int(yield_denominator),
            "per_label_positive": per_label_count,
        },
        "weighted": {
            "convention": "权重和（train_weight*sample_weight）",
            "trainable_weight_sum": float(effective.sum()),
            "step_yield_weighted": float(weighted_yield),
            "yield_denominator": yield_denominator_name,
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
        "window": _aggregate_window(entries, window_s=float(config.get("window_fail_before") or 0.0)),
        "timing": _aggregate_timing(entries, shard_write_s=shard_write_s),
        "scan": dict(scan or {}),
        "config": dict(config),
        "dataset_gate": {
            "bc_retained_step_yield": {
                "value": float(row_yield),
                "min": DEFAULT_MIN_YIELD,
                "pass": bool(row_yield >= DEFAULT_MIN_YIELD),
                "denominator": yield_denominator_name,
                "denominator_value": int(yield_denominator),
                "convention": (
                    "行数口径：train_weight>0 行数 / 窗口候选行数（窗口模式：窗口内实际考虑的行 = stored）"
                    if window_mode
                    else "行数口径：train_weight>0 行数 / 候选行数（整段 episode 候选帧）"
                ),
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
    parser.add_argument("--specs", type=str, required=True,
                        help="场景池 json（**必填**；只允许 train spec——隔离守卫拒绝 eval/val 交集）")
    parser.add_argument("--labeler", choices=("pure_pursuit", "idm"), default=DEFAULT_LABELER,
                        help="空问专家（默认 pure_pursuit；idm = MetaDrive 原始 IDMPolicy，动作语义不同）")
    parser.add_argument("--tracker", choices=("lqr", "exact"), default="lqr",
                        help="学生动作执行器（默认 lqr = 评测闭环协议；exact = 运动学精确执行）")
    parser.add_argument("--workers", type=int, default=4, help="spawn worker 数（多核；默认 4）")
    parser.add_argument("--max-steps", type=int, default=600, help="单 episode 最大 env step（默认 600）")
    parser.add_argument("--limit", type=int, default=0,
                        help="调试：spec 数上限（0=全部；**shuffle 之后**截断；与 --target-fails 先到者为准）")
    parser.add_argument("--target-fails", type=int, default=1000,
                        help="流式扫描目标失败数（默认 1000；>0 生效：按扫描顺序处理，失败 episode 数达 N "
                             "立即停止并优雅收尾；0 = 关闭，扫完整个池）")
    parser.add_argument("--shuffle-seed", type=int, default=0,
                        help="扫描顺序：0/缺省 = 池原始顺序；>0 = numpy.default_rng(k) 无放回打乱（每轮换 k）")
    parser.add_argument("--device", type=str, default=None, help="策略运行设备（默认取 config train.device）")
    parser.add_argument("--model-config", type=str, default="config/model.yaml")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--balance", choices=("weights", "cap", "none"), default="weights",
                        help="(difficulty, geometry) 组配平模式（默认 weights）")
    parser.add_argument("--balance-ratio", type=float, default=3.0, help="cap 模式最大组/最小组样本比")
    parser.add_argument("--traffic-density", type=float, default=None, help="覆盖 build_env 的 traffic_density")
    parser.add_argument("--window-fail-before", type=float, default=0.0,
                        help="窗口模式（>0 生效）：失败 episode 只保留终止前 N 秒（ceil(N/0.5) 个策略帧）的行；"
                             "成功/max_step 零行。自动松弛 terminal_window，默认松弛 on_lane/yaw_outlier")
    parser.add_argument("--fail-terminations", type=str, default=",".join(DEFAULT_FAIL_TERMINATIONS),
                        help=f"失败终止类型（逗号分隔；默认 {','.join(DEFAULT_FAIL_TERMINATIONS)}；"
                             "max_step（工具截断/env timeout）不算失败）")
    parser.add_argument("--window-strict-filters", action="store_true",
                        help="窗口模式下恢复 v1 过滤（on_lane + yaw_outlier 生效；默认关闭）")
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
    _, kinematics_source = resolve_interpolate()
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

    fail_terminations = [
        item.strip().lower() for item in str(args.fail_terminations or "").split(",") if item.strip()
    ] or list(DEFAULT_FAIL_TERMINATIONS)
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
        "window_fail_before": float(args.window_fail_before or 0.0),
        "fail_terminations": fail_terminations,
        "window_strict_filters": bool(args.window_strict_filters),
    }
    shard_dir = out / SHARD_DIRNAME
    if shard_dir.exists():
        shutil.rmtree(shard_dir, ignore_errors=True)
    summaries, scan = _run_dagger_workers(
        specs,
        workers=max(1, int(args.workers)),
        config=worker_config,
        shard_dir=shard_dir,
        target_fails=int(args.target_fails or 0),
        fail_terminations=fail_terminations,
        logger=print,
    )
    scan["shuffle_seed"] = int(provenance.get("shuffle_seed", 0) or 0)
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
        shard_write_s=float(sum(float(item.get("shard_write_s", 0.0)) for item in summaries)),
        scan=scan,
    )
    provenance.update(
        {key: scan[key] for key in ("shuffle_seed", "target_fails", "scanned_specs", "fail_count", "target_reached")}
    )
    pathological_total = {
        key: int(sum(int((entry["report"].get("pathological") or {}).get(key, 0)) for entry in entries))
        for key in ("stuck", "yaw_outlier", "stuck_skipped_truncated", "future_truncated_rows")
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
        "window": report["window"],
        "timing": report["timing"],
        "scan": dict(report["scan"]),
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
    print(
        f"[dagger] pathological: {pathological_total}"
        f"（stuck 阈值 {STUCK_SPEED_MPS} m/s / yaw {YAW_OUTLIER_RAD} rad；"
        f"末帧 {TRAJ30_SPAN} env steps 内 future_truncated → stuck 跳过 "
        f"{pathological_total.get('stuck_skipped_truncated', 0)} 行）",
        flush=True,
    )
    print(f"[dagger] DONE → {out}（rows={stored_rows}，episodes={dagger['episodes']}，"
          f"scanned={report['scanned_specs']} specs / fails={report['fail_count']}"
          f"（target={report['target_fails']}，reached={report['target_reached']}），"
          f"elapsed={dagger['elapsed_s']:.1f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
