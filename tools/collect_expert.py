"""BC 专家数据采集工具（阶段 A 冷启动数据源）。

用法::

    tools/venv-python tools/collect_expert.py \
        --specs env/specs/scenarios_train_slice200.json --limit 20 --out runs/bc_expert

专家
----
默认 MetaDrive 内置 ``IDMPolicy``（``config/train.yaml::stages.A.bc.expert=idmpolicy``，
沿导航路线行驶）；``--expert pure_pursuit`` 可切换到仓库内的
``env/expert/pure_pursuit_idm.py::PurePursuitIDMPolicy``（确定性规则基线）。

产出（``--out`` 目录）
----------------------
- ``expert_bc.npz``：按帧存（**不存 6 帧堆叠**）的 BC 样本；历史窗口由训练侧在线拼；
- ``expert_bc.meta.json``：schema / 通道形状 / 对齐元数据 / 过滤统计；
- ``report.json``：产出率 ``bc_retained_step_yield`` + 每标签正样本数 + dataset_gate 判定。

过滤规则（p2-contract §8.2 规则 1–5，逐条落码）
-----------------------------------------------
1. **终末截断**：episode 遇 terminated/truncated 后不再产生任何样本（终末帧及其后目标丢弃）；
2. **干净 3 s 窗口**：候选帧之后 6 个策略步（30 个 env step）内不得出现
   crash / out_of_road / arrive_dest / 截断；
3. **on_lane + round-trip**：专家必须在自己车道内；且 6 个 ``(ds,dθ)`` 动作经运动学插值
   得到的 6 个关键点与实测关键点误差在阈值内（阈值可配；dense 30 点误差一并报告）；
4. **事件标签可核验**：``cutin_active`` / ``cutout_active`` 为 1 的帧，仅当其事件
   ``fired ∧ actor_alive`` 时保留；保留后核对每标签正样本数
   （``config/eval.yaml::dataset_gate.min_samples_per_category=50``）；
5. **难度/几何配平**：默认给每组 (difficulty, 主几何标签) 等贡献权重（``--balance weights``），
   可选按组截断（``--balance cap``）或关闭（``--balance none``）。

运动学单一真源（§8.5）
----------------------
优先调用 ``env.tracking.interpolate``（若该模块已落地）；否则使用本模块的纯 NumPy
精确圆弧积分（constant v/ω，与 ``ExactTracker`` 的插值数学一致）作为 fallback，并把
``kinematics_source`` 记录进 meta。

import 时只做 ``sys.path`` 修正与函数定义，不建 env、不做 IO。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 允许 `python tools/collect_expert.py` 直接运行（此时 sys.path[0] 是 tools/）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.metadrive_env import build_env  # noqa: E402
from env.obs.builder import ObservationBuilder  # noqa: E402
from env.scenario.behaviors import event_state  # noqa: E402
from env.scenario.labels import LABEL_ORDER, compute_step_labels  # noqa: E402
from env.scenario.spec import load_specs  # noqa: E402

__all__ = ["main", "arc_interpolate", "SUPERVISED_LABELS"]

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

POLICY_DT = 0.5  # 策略步长（s）= 2 Hz（config/env.yaml::timing.policy_dt）
PHYSICS_DT = 0.1  # env.step 步长（s）
STEPS_PER_POLICY = 5  # POLICCY_DT / PHYSICS_DT
WINDOW_POLICIES = 6  # 3 s 目标窗口 = 6 个策略步
WINDOW_STEPS = WINDOW_POLICIES * STEPS_PER_POLICY  # 30 env steps
CURRENT_CHANNELS = ("ego", "od", "ld", "nav", "signal")
SUPERVISED_LABELS: Tuple[str, ...] = (
    "cutin_active",
    "cutout_active",
    "crowded",
    "car_following",
    "on_curve",
    "merging",
    "roundabout_near",
    "near_intersection",
)
#: 受监督标签的 9→8 映射来源：config/model.yaml::moe.router.supervised_labels（固定顺序）。
_MODEL_CONFIG_DEFAULT = "config/model.yaml"
#: dataset_gate 口径（config/eval.yaml）
DEFAULT_MIN_YIELD = 0.60
DEFAULT_MIN_SAMPLES_PER_CATEGORY = 50

_CRASH_KEYS = ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_sidewalk", "crash_human")


# --------------------------------------------------------------------------- #
# 运动学：6×0.5 s → 30×0.1 s（§8.5 单一真源）
# --------------------------------------------------------------------------- #

def arc_interpolate(seq: np.ndarray, dt: float = POLICY_DT, hz: int = 10) -> np.ndarray:
    """把 ``(K,2)`` 的 ``(ds,dθ)`` 序列按精确圆弧积分成 ``(K*dt*hz, 3)`` 的 ``(x,y,θ)``。

    每个动作段内假设 ``v=ds/dt``、``ω=dθ/dt`` 恒定，按 ``1/hz`` 子步用圆弧闭式解推进
    （``ω≈0`` 时退化为直线）。输出为**自车系**（x 前向 / y 左向），起点为原点。

    为什么用闭式解而不是中点积分：与 ``ExactTracker`` 的"每个子步置于插值位姿"语义一致，
    同一 ``(ds,dθ)`` 在任何调用点得到同一轨迹（§8.5 单一真源）。
    """
    arr = np.asarray(seq, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"seq 必须是 (K,2) 的 (ds,dθ)，实际 shape={arr.shape}")
    if dt <= 0.0 or hz <= 0:
        raise ValueError(f"dt/hz 非法：dt={dt}, hz={hz}")
    n_sub = max(1, int(round(dt * hz)))
    sub_dt = dt / float(n_sub)
    out = np.zeros((arr.shape[0] * n_sub, 3), dtype=np.float64)
    x = y = theta = 0.0
    cursor = 0
    for ds, dtheta in arr:
        v = float(ds) / dt
        omega = float(dtheta) / dt
        for _ in range(n_sub):
            if abs(omega) > 1e-9:
                radius = v / omega
                theta_new = theta + omega * sub_dt
                x += radius * (math.sin(theta_new) - math.sin(theta))
                y -= radius * (math.cos(theta_new) - math.cos(theta))
                theta = theta_new
            else:
                x += v * sub_dt * math.cos(theta)
                y += v * sub_dt * math.sin(theta)
            out[cursor] = (x, y, theta)
            cursor += 1
    return out


_KINEMATICS_CACHE: Dict[str, Any] = {}


def resolve_interpolate() -> Tuple[Any, str]:
    """返回 ``(interpolate 函数, 来源标识)``；优先 ``env.tracking.interpolate``（§8.5）。"""
    if "fn" in _KINEMATICS_CACHE:
        return _KINEMATICS_CACHE["fn"], _KINEMATICS_CACHE["source"]
    fn, source = None, "collect_expert.arc_interpolate"
    try:
        from env.tracking import interpolate as tracking_interpolate  # type: ignore

        if callable(tracking_interpolate):
            fn, source = tracking_interpolate, "env.tracking.interpolate"
    except Exception:  # noqa: BLE001 - 未落地时用本模块 fallback
        fn = None
    if fn is None:
        fn = arc_interpolate
    _KINEMATICS_CACHE["fn"] = fn
    _KINEMATICS_CACHE["source"] = source
    return fn, source


# --------------------------------------------------------------------------- #
# 工具：位姿 / 信息解析 / 标签
# --------------------------------------------------------------------------- #

def _ego_pose(ego: Any) -> np.ndarray:
    pos = np.asarray(ego.position, dtype=np.float64)
    return np.array([float(pos[0]), float(pos[1]), float(ego.heading_theta)], dtype=np.float64)


def _to_local(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """世界坐标 ``(N,2)`` → 位姿 ``pose=(x,y,θ)`` 自车系（x 前向 / y 左向）的 ``(N,2)``。"""
    pts = np.asarray(points_world, dtype=np.float64) - np.asarray(pose[:2], dtype=np.float64)
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    rot = np.array([[c, s], [-s, c]], dtype=np.float64)  # R(-θ) 的显式形式
    return pts @ rot.T


def _flag_dict(info: Dict[str, Any]) -> Dict[str, Any]:
    """把 MetaDrive ``info`` 解析成本步终止/违例标记。"""
    info = info if isinstance(info, dict) else {}
    crash = any(bool(info.get(key, False)) for key in _CRASH_KEYS)
    out_of_road = bool(info.get("out_of_road", False))
    arrive = bool(info.get("arrive_dest", False))
    return {
        "crash": crash,
        "out_of_road": out_of_road,
        "arrive": arrive,
        "bad": crash or out_of_road or arrive,
        "max_step": bool(info.get("max_step", False)),
    }


def load_supervised_labels(path: str = _MODEL_CONFIG_DEFAULT) -> Tuple[str, ...]:
    """读取 ``config/model.yaml::moe.router.supervised_labels``（固定顺序单一出处）。

    读取失败（PyYAML 缺失 / 文件缺失）时退回模块常量，并与 ``LABEL_ORDER`` 交叉校验。
    """
    labels: Optional[Sequence[str]] = None
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        labels = payload.get("moe", {}).get("router", {}).get("supervised_labels")
    except Exception:  # noqa: BLE001
        labels = None
    order = tuple(str(x) for x in labels) if labels else SUPERVISED_LABELS
    missing = [name for name in order if name not in LABEL_ORDER]
    if missing:
        raise ValueError(f"受监督标签 {missing} 不在 labels.LABEL_ORDER={LABEL_ORDER} 中")
    return order


def _supervised_vector(labels_raw: Dict[str, float], order: Sequence[str]) -> np.ndarray:
    return np.array([float(labels_raw.get(name, 0.0)) for name in order], dtype=np.float32)


def _events_summary(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 ``event_state`` 压缩成可核验的 ``[{type, fired, alive}, ...]``（规则 4 用）。"""
    out: List[Dict[str, Any]] = []
    events = state.get("events", {}) if isinstance(state, dict) else {}
    actors = state.get("actors", {}) if isinstance(state, dict) else {}
    for key, event in (events or {}).items():
        actor = (actors or {}).get(key) or {}
        out.append(
            {
                "type": str(event.get("type", "")),
                "fired": bool(event.get("fired", False)),
                "alive": bool(event.get("actor_alive", False)) and bool(actor.get("alive", True)),
            }
        )
    return out


def _on_lane(ego: Any) -> Tuple[bool, float, float]:
    """``(是否在车道内, 横向偏差, 车道宽)``；无车道对象时视为不在车道内。"""
    lane = getattr(ego, "lane", None)
    if lane is None:
        return False, float("nan"), float("nan")
    try:
        width = float(lane.width)
        _, lateral = lane.local_coordinates(ego.position)
        lateral = float(lateral)
    except Exception:  # noqa: BLE001
        return False, float("nan"), float("nan")
    return True, lateral, width


def _on_lane_ok(lateral: float, width: float, frac: float, margin: float) -> bool:
    if not math.isfinite(lateral) or not math.isfinite(width) or width <= 0.0:
        return False
    return abs(lateral) <= frac * width + margin


# --------------------------------------------------------------------------- #
# episode 采集与窗口筛选
# --------------------------------------------------------------------------- #

def collect_episode(
    env: Any,
    spec: Any,
    *,
    builder: ObservationBuilder,
    expert_kind: str,
    max_steps: int,
) -> Dict[str, Any]:
    """跑一个 episode，记录逐 env-step 位姿/标记与逐策略步观测帧。

    返回 ``{"frames", "poses", "flags", "termination", "steps", "ended_by_env"}``；
    不做任何过滤判断，过滤由 :func:`extract_samples` 用完整窗口信息统一执行
    （规则 1–4 需要 look-ahead）。
    """
    env.reset()
    ego = env.agent
    # add_policy 会把 *args 透传给策略构造函数；两种专家的前两个位置参数都是
    # (control_object, random_seed)（idm_policy.py:224 / pure_pursuit_idm.py:159）。
    policy = env.engine.add_policy(
        ego.id, _expert_class(expert_kind), ego, int(getattr(spec, "seed", 0))
    )
    if hasattr(policy, "reset"):
        policy.reset()

    frames: Dict[int, Dict[str, Any]] = {}
    poses: List[np.ndarray] = [_ego_pose(ego)]
    flags: List[Optional[Dict[str, Any]]] = [None]
    termination = "max_step"  # 我们自己截断（未到 env 终局）
    ended_by_env = False
    step = 0
    # §8.4：ego 通道 reserved0/1 = 上一策略步动作 (ds,dθ)；调用方负责在每次策略决策后注入。
    env.prev_policy_action = np.zeros(2, dtype=np.float64)
    while step < max_steps:
        if step % STEPS_PER_POLICY == 0:
            if step >= STEPS_PER_POLICY:
                # 上一策略步的实测动作（由位姿窗口反推，与 BC 目标同一口径）
                env.prev_policy_action = _window_actions(poses, step - STEPS_PER_POLICY, n_policies=1)[0]
            else:
                env.prev_policy_action = np.zeros(2, dtype=np.float64)
            obs = builder.build(env, spec)
            labels_raw = compute_step_labels(env, spec)
            state = event_state(env)
            on_lane, lateral, width = _on_lane(ego)
            current = {key: np.array(obs[key], dtype=np.float32, copy=True) for key in obs if "_hist" not in key}
            frames[step] = {
                "obs": current,
                "hist_valid": np.array(obs.get("hist_valid", np.zeros(6, dtype=np.float32)), dtype=np.float32),
                "labels_raw": np.array([labels_raw.get(name, 0.0) for name in LABEL_ORDER], dtype=np.float32),
                "events": _events_summary(state),
                "pose": poses[-1].copy(),
                "on_lane": bool(on_lane),
                "lane_lat": float(lateral),
                "lane_width": float(width),
            }
        _, _, terminated, truncated, info = env.step([0.0, 0.0])
        step += 1
        poses.append(_ego_pose(ego))
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
    }


def _expert_class(kind: str) -> Any:
    """专家策略类（供 ``engine.add_policy`` 构造，类签名 ``(control_object, random_seed)``）。"""
    if str(kind).lower() == "idm":
        from metadrive.policy.idm_policy import IDMPolicy

        return IDMPolicy
    from env.expert.pure_pursuit_idm import PurePursuitIDMPolicy

    return PurePursuitIDMPolicy


def _window_actions(poses: Sequence[np.ndarray], t: int, n_policies: int = WINDOW_POLICIES) -> np.ndarray:
    """由实测位姿窗口算 6 个 ``(ds,dθ)``：ds=折线弧长，dθ=航向差（wrap 到 (-π,π]）。"""
    actions = np.zeros((n_policies, 2), dtype=np.float64)
    for i in range(n_policies):
        a, b = t + i * STEPS_PER_POLICY, t + (i + 1) * STEPS_PER_POLICY
        length = 0.0
        for j in range(a, b):
            length += float(np.linalg.norm(poses[j + 1][:2] - poses[j][:2]))
        dtheta = _wrap_to_pi(float(poses[b][2]) - float(poses[a][2]))
        actions[i] = (length, dtheta)
    return actions


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def extract_samples(
    episode: Dict[str, Any],
    spec: Any,
    *,
    episode_id: int,
    label_order: Sequence[str],
    interpolate_fn: Any,
    on_lane_frac: float,
    on_lane_margin: float,
    roundtrip_key_mean: float,
    roundtrip_key_max: float,
    filter_counter: Counter,
    require_dense: bool = False,
    roundtrip_dense_mean: float = 0.5,
) -> List[Dict[str, Any]]:
    """按 §8.2 规则 1–4 从 episode 记录中抽取保留样本。"""
    poses, flags = episode["poses"], episode["flags"]
    frames: Dict[int, Dict[str, Any]] = episode["frames"]
    last_state = len(poses) - 1
    # 规则 1/2：窗口 [t+1, t+30] 内不得出现 crash/off-road/arrive，且不得含终末帧。
    # 最后一个状态要么是 env 终局（terminated/truncated），要么是工具自身 --max-steps 截断；
    # 前者按规则 1 丢弃该帧及其后目标，后者只是"看不到更远"，窗口末端最多到 last_state。
    first_bad = last_state + 1
    for index in range(1, last_state + 1):
        flag = flags[index]
        if flag and flag["bad"]:
            first_bad = index
            break
    if episode.get("ended_by_env", False):
        first_bad = min(first_bad, last_state)
    limit = first_bad - 1  # 窗口末端最大可到 limit
    out: List[Dict[str, Any]] = []
    for t in sorted(frames.keys()):
        if t + WINDOW_STEPS > limit:
            filter_counter["terminal_window"] += 1
            continue
        frame = frames[t]
        # 规则 3a：专家在车道内
        if not frame["on_lane"] or not _on_lane_ok(
            frame["lane_lat"], frame["lane_width"], on_lane_frac, on_lane_margin
        ):
            filter_counter["not_on_lane"] += 1
            continue
        # 规则 4：cut*_active 仅事件 fired ∧ actor_alive 时保留
        cut_ok = True
        for cut_name, event_type in (("cutin_active", "cut_in"), ("cutout_active", "cut_out")):
            if frame["labels_raw"][LABEL_ORDER.index(cut_name)] > 0.5:
                if not any(
                    ev["type"] == event_type and ev["fired"] and ev["alive"] for ev in frame["events"]
                ):
                    cut_ok = False
                    break
        if not cut_ok:
            filter_counter["cut_label_unverified"] += 1
            continue

        # 规则 3b：6 点目标 round-trip 可复现
        actions = _window_actions(poses, t)
        try:
            interp = np.asarray(interpolate_fn(actions, dt=POLICY_DT, hz=int(round(1.0 / PHYSICS_DT))), dtype=np.float64)
        except TypeError:  # env.tracking.interpolate 可能不接受 kwargs
            interp = np.asarray(interpolate_fn(actions), dtype=np.float64)
        if interp.shape[0] != WINDOW_STEPS:
            raise ValueError(f"interpolate 输出 {interp.shape}，期望 ({WINDOW_STEPS},3)")
        measured = np.stack([poses[t + j][:2] for j in range(1, WINDOW_STEPS + 1)], axis=0)
        measured_local = _to_local(measured, frame["pose"])
        key_indices = [i * STEPS_PER_POLICY - 1 for i in range(1, WINDOW_POLICIES + 1)]
        key_err = np.linalg.norm(interp[key_indices, :2] - measured_local[key_indices], axis=1)
        dense_err = np.linalg.norm(interp[:, :2] - measured_local, axis=1)
        if float(key_err.mean()) > roundtrip_key_mean or float(key_err.max()) > roundtrip_key_max:
            filter_counter["roundtrip_fail"] += 1
            continue
        if require_dense and float(dense_err.mean()) > roundtrip_dense_mean:
            filter_counter["roundtrip_dense_fail"] += 1
            continue

        if t >= STEPS_PER_POLICY:
            prev_action = _window_actions(poses, t - STEPS_PER_POLICY, n_policies=1)[0]
        else:
            prev_action = np.zeros(2, dtype=np.float64)
        labels_raw = frame["labels_raw"]
        out.append(
            {
                "spec_id": int(getattr(spec, "id", -1)),
                "seed": int(getattr(spec, "seed", -1)),
                "split": str(getattr(spec, "split", "unknown")),
                "episode_id": int(episode_id),
                "step": int(t),
                "difficulty": str(getattr(spec, "difficulty", "unknown")),
                "geometry": str(getattr(spec, "labels", {}).get("geometry", "unknown")),
                "obs": frame["obs"],
                "hist_valid": frame["hist_valid"],
                "pose": frame["pose"],
                "action": actions.astype(np.float32),
                "traj6": interp[key_indices, :2].astype(np.float32),
                "traj30": interp[:, :2].astype(np.float32),
                "traj30_measured": measured_local.astype(np.float32),
                "roundtrip_key_err": float(key_err.mean()),
                "roundtrip_dense_err": float(dense_err.mean()),
                "labels": _supervised_vector(
                    {name: float(labels_raw[i]) for i, name in enumerate(LABEL_ORDER)}, label_order
                ),
                "labels_raw": np.array(labels_raw, dtype=np.float32),
                "prev_action": prev_action.astype(np.float32),
                "lane_lat": float(frame["lane_lat"]),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 配平
# --------------------------------------------------------------------------- #

def apply_balance(
    samples: List[Dict[str, Any]],
    *,
    mode: str,
    ratio: float,
    seed: int,
) -> Dict[str, Any]:
    """按 (difficulty, 主几何) 组配平（规则 5）。

    - ``weights``（默认）：保留全部样本，权重 ``n_total / (n_groups * n_group)``，
      使每组对损失的贡献近似相等；
    - ``cap``：每组最多保留 ``ceil(ratio * 最小非空组样本数)``，组内按 (spec,step)
      确定性等距抽样（不引入 RNG，复现性最好）；
    - ``none``：权重恒 1。

    返回统计 dict（写入 meta/report）。
    """
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[(sample["difficulty"], sample["geometry"])].append(index)
    weights = np.ones(len(samples), dtype=np.float64)
    group_key = np.array(
        [f"{sample['difficulty']}/{sample['geometry']}" for sample in samples], dtype="U64"
    ) if samples else np.zeros(0, dtype="U64")
    stats: Dict[str, Any] = {
        "mode": mode,
        "ratio": float(ratio),
        "group_counts_before": {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(groups.items())},
    }
    if mode == "none" or not samples:
        stats["group_counts_after"] = dict(stats["group_counts_before"])
        return {"sample_weight": weights, "balance_group": group_key, "stats": stats}
    if mode == "weights":
        n_groups = max(1, len(groups))
        for key, indices in groups.items():
            weights[indices] = float(len(samples)) / (n_groups * len(indices))
        stats["weight_range"] = [float(weights.min()), float(weights.max())]
    elif mode == "cap":
        min_count = min(len(indices) for indices in groups.values())
        cap = max(1, int(math.ceil(float(ratio) * min_count)))
        keep: List[int] = []
        for key in sorted(groups):
            indices = sorted(groups[key], key=lambda i: (samples[i]["spec_id"], samples[i]["step"]))
            if len(indices) <= cap:
                keep.extend(indices)
            else:
                stride = len(indices) / float(cap)
                picks = [indices[min(len(indices) - 1, int(round(i * stride)))] for i in range(cap)]
                keep.extend(picks)
        dropped = sorted(set(range(len(samples))) - set(keep))
        stats["cap"] = cap
        stats["min_group_count"] = min_count
        stats["dropped_indices"] = dropped
        samples[:] = [samples[i] for i in sorted(keep)]
        weights = np.ones(len(samples), dtype=np.float64)
        group_key = np.array(
            [f"{s['difficulty']}/{s['geometry']}" for s in samples], dtype="U64"
        ) if samples else np.zeros(0, dtype="U64")
        groups = defaultdict(list)
        for index, sample in enumerate(samples):
            groups[(sample["difficulty"], sample["geometry"])].append(index)
    else:
        raise ValueError(f"未知配平模式 {mode!r}")
    stats["group_counts_after"] = {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(groups.items())}
    return {"sample_weight": weights, "balance_group": group_key, "stats": stats}


# --------------------------------------------------------------------------- #
# 保存
# --------------------------------------------------------------------------- #

def _obs_fingerprint() -> str:
    """观测实现指纹（``env/obs/*.py`` 内容哈希）；用于检测"数据与观测版本不一致"。"""
    try:
        from env.obs import obs_fingerprint

        return str(obs_fingerprint())
    except Exception:  # noqa: BLE001 - 指纹失败不阻断采集
        return ""


def _alignment_meta(builder: ObservationBuilder) -> Dict[str, Any]:
    """导出各通道的历史对齐规则（训练侧在线拼历史时复用，避免导入 env 实现）。"""
    out: Dict[str, Any] = {}
    for name, channel in builder.channels.items():
        alignment = getattr(channel, "alignment", None)
        if alignment is None:
            continue
        out[name] = {
            "point_pairs": [list(pair) for pair in getattr(alignment, "point_pairs", ())],
            "vector_pairs": [list(pair) for pair in getattr(alignment, "vector_pairs", ())],
            "angle_dims": list(getattr(alignment, "angle_dims", ())),
        }
    return out


def _stack_arrays(arrays: List[np.ndarray], key: str, shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.stack([np.asarray(item, dtype=np.float32) for item in arrays], axis=0)
    expected = (len(arrays),) + shape
    if arr.shape != expected:
        raise ValueError(f"样本字段 {key} 形状 {arr.shape}，期望 {expected}")
    return arr


def save_dataset(
    out_dir: Path,
    samples: List[Dict[str, Any]],
    *,
    label_order: Sequence[str],
    builder: ObservationBuilder,
    config: Dict[str, Any],
    report: Dict[str, Any],
    sample_weight: np.ndarray,
    balance_group: np.ndarray,
) -> Dict[str, str]:
    """保存 ``expert_bc.npz`` / ``expert_bc.meta.json`` / ``report.json``，返回路径表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "expert_bc.npz"
    meta_path = out_dir / "expert_bc.meta.json"
    report_path = out_dir / "report.json"

    arrays: Dict[str, np.ndarray] = {}
    for channel in CURRENT_CHANNELS:
        sample_shape = tuple(np.asarray(samples[0]["obs"][channel]).shape)
        arrays[channel] = _stack_arrays([sample["obs"][channel] for sample in samples], channel, sample_shape)
        mask_key = f"{channel}_mask"
        if mask_key in samples[0]["obs"]:
            mask_shape = tuple(np.asarray(samples[0]["obs"][mask_key]).shape)
            arrays[mask_key] = _stack_arrays(
                [sample["obs"][mask_key] for sample in samples], mask_key, mask_shape
            )
    arrays["hist_valid"] = _stack_arrays([sample["hist_valid"] for sample in samples], "hist_valid", (6,))
    arrays["pose"] = np.stack([np.asarray(sample["pose"], dtype=np.float32) for sample in samples], axis=0)
    arrays["action"] = np.stack([sample["action"] for sample in samples], axis=0)
    arrays["traj6"] = np.stack([sample["traj6"] for sample in samples], axis=0)
    arrays["traj30"] = np.stack([sample["traj30"] for sample in samples], axis=0)
    arrays["traj30_measured"] = np.stack([sample["traj30_measured"] for sample in samples], axis=0)
    arrays["roundtrip_key_err"] = np.array([sample["roundtrip_key_err"] for sample in samples], dtype=np.float32)
    arrays["roundtrip_dense_err"] = np.array([sample["roundtrip_dense_err"] for sample in samples], dtype=np.float32)
    arrays["labels"] = np.stack([sample["labels"] for sample in samples], axis=0)
    arrays["labels_raw"] = np.stack([sample["labels_raw"] for sample in samples], axis=0)
    arrays["prev_action"] = np.stack([sample["prev_action"] for sample in samples], axis=0)
    arrays["episode_id"] = np.array([sample["episode_id"] for sample in samples], dtype=np.int64)
    arrays["step"] = np.array([sample["step"] for sample in samples], dtype=np.int64)
    arrays["spec_id"] = np.array([sample["spec_id"] for sample in samples], dtype=np.int64)
    arrays["seed"] = np.array([sample["seed"] for sample in samples], dtype=np.int64)
    arrays["difficulty"] = np.array([sample["difficulty"] for sample in samples], dtype="U16")
    arrays["geometry"] = np.array([sample["geometry"] for sample in samples], dtype="U32")
    arrays["split"] = np.array([sample["split"] for sample in samples], dtype="U16")
    arrays["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)
    arrays["balance_group"] = np.asarray(balance_group, dtype="U64")
    arrays["lane_lat"] = np.array([sample["lane_lat"] for sample in samples], dtype=np.float32)
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "schema_version": 1,
        "kind": "bc_expert",
        "created_by": "tools/collect_expert.py",
        "obs_fingerprint": _obs_fingerprint(),
        "label_names": list(label_order),
        "raw_label_names": list(LABEL_ORDER),
        "channel_shapes": {
            name: list(np.asarray(samples[0]["obs"][name]).shape)
            for name in CURRENT_CHANNELS
            if name in samples[0]["obs"]
        },
        "channel_alignments": _alignment_meta(builder),
        "kinematics_source": report["kinematics_source"],
        "config": config,
        "count": len(samples),
        "history_storage": "per_frame",  # 不存 6 帧堆叠；训练侧按 episode_id/step 在线拼
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"npz": str(npz_path), "meta": str(meta_path), "report": str(report_path)}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="collect_expert.py",
        description="过滤 IDMPolicy 专家轨迹，产出阶段 A 的 BC 冷启动数据集（p2-contract §8.2）",
    )
    parser.add_argument("--specs", required=True, help="scenario-spec JSON 路径（load_specs 可读）")
    parser.add_argument("--out", required=True, help="输出目录（npz + meta + report）")
    parser.add_argument("--limit", type=int, default=None, help="按文件顺序只采集前 N 条（split 过滤后）")
    parser.add_argument("--split", choices=("all", "train", "val"), default="all", help="按 spec.split 过滤")
    parser.add_argument("--expert", choices=("idm", "pure_pursuit"), default="idm", help="专家策略（默认内置 IDM）")
    parser.add_argument("--max-steps", type=int, default=600, help="单 episode 最大 env step（默认 600 = 60 s）")
    parser.add_argument("--on-lane-frac", type=float, default=0.5, help="on_lane 横向阈值（× 车道宽，默认 0.5）")
    parser.add_argument("--on-lane-margin", type=float, default=0.3, help="on_lane 横向额外余量（m）")
    parser.add_argument("--roundtrip-key-mean", type=float, default=0.25, help="6 关键点平均误差阈值（m）")
    parser.add_argument("--roundtrip-key-max", type=float, default=0.5, help="6 关键点最大误差阈值（m）")
    parser.add_argument("--require-dense", action="store_true", help="额外要求 30 点平均误差 <= --roundtrip-dense-mean")
    parser.add_argument("--roundtrip-dense-mean", type=float, default=0.5, help="30 点平均误差阈值（m）")
    parser.add_argument("--balance", choices=("weights", "cap", "none"), default="weights", help="配平模式（规则 5）")
    parser.add_argument("--balance-ratio", type=float, default=3.0, help="cap 模式下最大组/最小组样本比")
    parser.add_argument("--seed", type=int, default=0, help="工具随机种子（仅用于诊断）")
    parser.add_argument("--traffic-density", type=float, default=None, help="覆盖 build_env 的 traffic_density")
    parser.add_argument(
        "--model-config", default=_MODEL_CONFIG_DEFAULT, help="读取 router.supervised_labels 的配置文件"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    specs = load_specs(args.specs)
    if args.split != "all":
        specs = [spec for spec in specs if getattr(spec, "split", None) == args.split]
    if args.limit is not None:
        specs = specs[: max(0, int(args.limit))]
    if not specs:
        print("[collect_expert] 没有可采集的 spec（检查 --specs/--split/--limit）", flush=True)
        return 2

    label_order = load_supervised_labels(args.model_config)
    interpolate_fn, kinematics_source = resolve_interpolate()
    builder = ObservationBuilder({})
    filter_counter: Counter = Counter()
    samples: List[Dict[str, Any]] = []
    spec_reports: List[Dict[str, Any]] = []
    total_candidates = 0
    episode_id = 0
    print(
        f"[collect_expert] specs={len(specs)} expert={args.expert} kinematics={kinematics_source} "
        f"labels={list(label_order)}",
        flush=True,
    )

    for index, spec in enumerate(specs, start=1):
        env = None
        spec_start = time.perf_counter()
        try:
            env = build_env(spec, traffic_density=args.traffic_density, use_render=False)
            episode = collect_episode(
                env,
                spec,
                builder=builder,
                expert_kind=args.expert,
                max_steps=int(args.max_steps),
            )
            kept = extract_samples(
                episode,
                spec,
                episode_id=episode_id,
                label_order=label_order,
                interpolate_fn=interpolate_fn,
                on_lane_frac=float(args.on_lane_frac),
                on_lane_margin=float(args.on_lane_margin),
                roundtrip_key_mean=float(args.roundtrip_key_mean),
                roundtrip_key_max=float(args.roundtrip_key_max),
                filter_counter=filter_counter,
                require_dense=bool(args.require_dense),
                roundtrip_dense_mean=float(args.roundtrip_dense_mean),
            )
            total_candidates += len(episode["frames"])
            samples.extend(kept)
            spec_reports.append(
                {
                    "id": int(getattr(spec, "id", -1)),
                    "seed": int(getattr(spec, "seed", -1)),
                    "difficulty": str(getattr(spec, "difficulty", "unknown")),
                    "geometry": str(getattr(spec, "labels", {}).get("geometry", "unknown")),
                    "termination": episode["termination"],
                    "env_steps": int(episode["steps"]),
                    "candidate_policy_steps": len(episode["frames"]),
                    "retained_steps": len(kept),
                    "step_yield": (len(kept) / len(episode["frames"])) if episode["frames"] else 0.0,
                }
            )
            print(
                f"[collect_expert] {index}/{len(specs)} id={getattr(spec, 'id', -1)} "
                f"term={episode['termination']} steps={episode['steps']} "
                f"cand={len(episode['frames'])} kept={len(kept)} "
                f"({time.perf_counter() - spec_start:.1f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - 单条失败不中断整批（与 validator 同口径）
            print(f"[collect_expert] id={getattr(spec, 'id', -1)} 失败：{type(exc).__name__}: {exc}", flush=True)
            spec_reports.append(
                {
                    "id": int(getattr(spec, "id", -1)),
                    "termination": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
        episode_id += 1

    if not samples:
        print("[collect_expert] 未保留任何样本；检查过滤阈值或专家质量", flush=True)

    balance = apply_balance(
        samples,
        mode=str(args.balance),
        ratio=float(args.balance_ratio),
        seed=int(args.seed),
    )
    per_label = {
        name: int(sum(1 for sample in samples if float(sample["labels"][i]) > 0.5))
        for i, name in enumerate(label_order)
    }
    yield_value = (len(samples) / total_candidates) if total_candidates else 0.0
    below_min = sorted(name for name, count in per_label.items() if count < DEFAULT_MIN_SAMPLES_PER_CATEGORY)
    report: Dict[str, Any] = {
        "specs": str(args.specs),
        "expert": str(args.expert),
        "kinematics_source": kinematics_source,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "total_candidate_steps": int(total_candidates),
        "retained_steps": int(len(samples)),
        "bc_retained_step_yield": float(yield_value),
        "filter_counts": dict(filter_counter),
        "per_label_positive": per_label,
        "min_samples_per_category": DEFAULT_MIN_SAMPLES_PER_CATEGORY,
        "labels_below_min": below_min,
        "balance": balance,
        "per_spec": spec_reports,
        "config": {
            "limit": args.limit,
            "split": args.split,
            "max_steps": int(args.max_steps),
            "on_lane_frac": float(args.on_lane_frac),
            "on_lane_margin": float(args.on_lane_margin),
            "roundtrip_key_mean": float(args.roundtrip_key_mean),
            "roundtrip_key_max": float(args.roundtrip_key_max),
            "require_dense": bool(args.require_dense),
            "roundtrip_dense_mean": float(args.roundtrip_dense_mean),
            "balance": str(args.balance),
            "balance_ratio": float(args.balance_ratio),
            "traffic_density": args.traffic_density,
        },
        "dataset_gate": {
            "bc_retained_step_yield": {
                "value": float(yield_value),
                "min": DEFAULT_MIN_YIELD,
                "pass": bool(yield_value >= DEFAULT_MIN_YIELD),
            },
            "min_samples_per_category": {
                "min": DEFAULT_MIN_SAMPLES_PER_CATEGORY,
                "labels_below_min": below_min,
                "pass": bool(not below_min),
            },
        },
    }
    # report 内的 balance 需要序列化 np 标量 → 转换
    report["balance"] = {
        "mode": balance["stats"]["mode"],
        "ratio": balance["stats"]["ratio"],
        "group_counts_before": balance["stats"]["group_counts_before"],
        "group_counts_after": balance["stats"]["group_counts_after"],
        **({"weight_range": balance["stats"]["weight_range"]} if "weight_range" in balance["stats"] else {}),
        **({"cap": balance["stats"]["cap"], "min_group_count": balance["stats"]["min_group_count"]}
           if "cap" in balance["stats"] else {}),
    }
    paths = save_dataset(
        Path(args.out),
        samples,
        label_order=label_order,
        builder=builder,
        config=report["config"],
        report=report,
        sample_weight=balance["sample_weight"],
        balance_group=balance["balance_group"],
    )

    print("", flush=True)
    print(f"[collect_expert] retained-step yield = {yield_value:.3f} "
          f"({len(samples)}/{total_candidates})  gate>= {DEFAULT_MIN_YIELD}: "
          f"{'PASS' if yield_value >= DEFAULT_MIN_YIELD else 'FAIL'}", flush=True)
    print(f"[collect_expert] filter counts: {dict(filter_counter)}", flush=True)
    print("[collect_expert] per-label positives "
          f"(min {DEFAULT_MIN_SAMPLES_PER_CATEGORY}):", flush=True)
    for name in label_order:
        flag = "" if per_label[name] >= DEFAULT_MIN_SAMPLES_PER_CATEGORY else "  <-- below min"
        print(f"  {name:>18}: {per_label[name]}{flag}", flush=True)
    print(f"[collect_expert] balance: {report['balance']}", flush=True)
    print(f"[collect_expert] wrote {paths['npz']} ({len(samples)} samples), "
          f"{paths['meta']}, {paths['report']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
