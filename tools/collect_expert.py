"""BC 专家数据采集工具（阶段 A 冷启动数据源，schema v2 / ``per_frame_v2``）。

用法::

    tools/venv-python tools/collect_expert.py \
        --specs env/specs/scenarios_train_slice200.json --limit 20 --out runs/bc_expert

专家
----
默认 MetaDrive 内置 ``IDMPolicy``（``config/train.yaml::stages.A.bc.expert=idmpolicy``，
沿导航路线行驶）；``--expert pure_pursuit`` 可切换到仓库内的
``env/expert/pure_pursuit_idm.py::PurePursuitIDMPolicy``（确定性规则基线）。

并行采集（``--workers``，默认 auto）
-----------------------------------
``--workers N`` 用 **spawn** 子进程并行跑 spec 列表（MetaDrive 每进程只能有一个 engine，
见 ``pipeline/vector_env.py``；每个 worker 同一时刻只建一个 env，跑完即 ``close``）：

- spec 按全局下标轮转分配（``index % N``，确定性、互斥、覆盖全部）；父进程按**原始
  spec 顺序**合并，``episode_id`` 直接用全局下标，因此 ``expert_bc.npz`` / meta /
  report 统计与 ``--workers 1`` 一致（worker 数不影响产出）；
- **分片落盘（v3）**：worker 在**自己进程内**把每块（``--recycle-every`` 条 spec，默认
  150）的逐帧记录写成 ``<out>/_shards/shard_w<worker>_c<chunk>.npz``（``np.savez``
  不压缩，快），只把**逐 spec 标量摘要**（行数 / 过滤计数 / 可训练行 / 分组）经队列
  回传父进程；父进程不再持有逐帧数据 → **父进程 RSS 不随数据量增长**。全部 worker
  退出后自动合并（见下），合并峰值 ≈ 最终数据集一份；
- 每个 worker 在建 env 前调用 ``pipeline.gl_runtime.ensure_gl_library_path()``，父进程
  spawn 前也先把 venv 本地 glvnd 写进 ``LD_LIBRARY_PATH``（spawn 子进程启动时继承），
  避免子进程的 "Known Pipes" 崩溃；
- **worker 内存**：单 worker 基线 ≈0.7 GB（``pipeline/vector_env.ENV_RSS_PER_WORKER_MB``
  =650 MB），另加 MetaDrive ``build_env``+``close`` 每 spec ≈3.5–4 MB 残留（本机实测
  线性上涨）与当前块待落盘的行（≈0.2 GB/150 spec）；worker 每 ``--recycle-every`` 条
  落盘并退出重启，峰值 ≈1.5 GB/worker；本机 15 GB 建议 ``--workers<=8``（6 更稳）；
- worker 异常退出时只丢失当前块，其余块继续；末尾打印 ``missing`` 统计。

分片合并（自动；``--keep-shards`` 保留分片）
------------------------------------------
收集结束后父进程按**原始 spec 下标顺序**逐分片、逐键拼接：每个键先分配最终数组，再按
``episode_id`` 序从分片读入对应行（峰值 ≈ 最终数组一份，不把全部数据复制多份），最后
``np.savez_compressed`` 写 ``expert_bc.npz``；``sample_weight`` / ``balance_weight`` /
``balance_group`` 由逐 spec 摘要生成（额外内存 O(specs)，不随行数增长）。默认合并成功后
删除 ``<out>/_shards/``；``--keep-shards`` 保留（调试 / 后处理）。

例（2000 条 spec、6 worker）::

    tools/venv-python tools/collect_expert.py \
        --specs env/specs/scenarios_train.json --limit 2000 --out runs/bc_expert_2k --workers 6

v2 相对 v1 的变更（2026-09-26）
-------------------------------
1. **存全部 policy 帧，不再逐帧删行**。过滤命中（terminal_window / 不在车道 / cut 标签未验证 /
   roundtrip 与密点失败）→ 该行 ``train_weight=0``，行仍写入 npz。这修复了"未来目标
   ``step+5k`` 缺失 25%（k=6）"——缺失的根因是过滤行被删除后查表落空；
2. 每行 ``wm_valid[6]``：horizon k 的目标帧**存在且仍可用**（未越过 episode 终止、自车仍在
   车道容差内），供 Stage A 逐 horizon 掩码（缺失/污染 horizon 显式置 0，不再按有效项
   归一造成隐性重加权）；
3. 每帧存 OD 槽位身份：``od_id``（int64，track id，episode 内稳定）、``od_presence``，
   以及 mem 所需的 ``od_id_hist``/``od_presence_hist``（6 帧，精确同槽位）；
4. ``od_fingerprint`` 升级为 ``v2-`` 前缀，``schema_version=2``，``history_storage=per_frame_v2``；
   ``expert_bc.meta.json::schema`` 写入完整 schema 清单（键/形状/dtype/语义/单位，单一出处
   ``env/obs/schema.py``），下游按契约开发；
5. report.json 权重感知：**计数口径 = 行数 / 加权口径 = 权重和**（见 ``report["counts"]`` 与
   ``report["weighted"]``）；per-label 统计同时给出行数与加权和，dataset_gate 用行数口径
   （保守下界），加权口径用于损失贡献分析。

产出（``--out`` 目录）
----------------------
- ``expert_bc.npz``：按帧存（**不存 6 帧堆叠**）的 BC 样本 + ``od_id``/``od_presence`` 及其历史；
  其余 6 帧历史（ego/others/od/ld）由训练侧用 ``pipeline.frames.FrameLookup`` 按
  ``(episode_id, step)`` 精确查表重建；
- ``expert_bc.meta.json``：schema / 通道形状 / 对齐元数据 / 过滤统计 / **完整 schema 清单**；
- ``report.json``：产出率 ``bc_retained_step_yield``（行数口径 = train_weight>0 行数/候选行数）
  + 计数/加权两套 per-label 统计 + dataset_gate 判定；
- ``_shards/``：采集期逐块中间分片（默认合并后删除；``--keep-shards`` 保留）。

过滤规则（p2-contract §8.2 规则 1–5，v2 语义：命中 → 权重 0，不删行）
------------------------------------------------------------------
1. **终末截断**：候选帧之后 6 个策略步（30 个 env step）内不得出现
   crash / out_of_road / arrive_dest / 截断；
2. **干净 3 s 窗口**：同上（合并入 1）；
3. **on_lane + round-trip**：专家必须在自己车道内；且 6 个 ``(ds,dθ)`` 动作经运动学插值
   得到的 6 个关键点与实测关键点误差在阈值内（阈值可配；dense 30 点误差一并报告）。
   **P1 iter2 变道放宽**：``|lane_lat| > --roundtrip-lc-lat``（默认 0.8 m）的执行帧在
   首步误差 ≤ ``--roundtrip-lc-first-max`` 且 6 点均值 ≤ ``--roundtrip-lc-mean`` 时保留
   （弧模型低速大转角误差 ~ |dθ|；原门丢 62% 变道帧）；
4. **事件标签可核验**：``cutin_active`` / ``cutout_active`` 为 1 的帧，仅当其事件
   ``fired ∧ actor_alive`` 时 ``train_weight=1``；保留后核对每标签正样本数
   （``config/eval.yaml::dataset_gate.min_samples_per_category=50``）；
5. **难度/几何配平**：默认给每组 (difficulty, 主几何标签) 等贡献权重（``--balance weights``），
   可选按组封顶（``--balance cap``，把超出的可训练行 ``sample_weight`` 置 0，仍不删行）
   或关闭（``--balance none``）。

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
import shutil
import sys
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# 允许 `python tools/collect_expert.py` 直接运行（此时 sys.path[0] 是 tools/）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from env.metadrive_env import build_env  # noqa: E402
from env.obs import OBS_SCHEMA_VERSION  # noqa: E402
from env.obs.builder import ObservationBuilder  # noqa: E402
from env.obs.others import OTHERS_HEAD_DIM, road_class_labels  # noqa: E402
from env.obs.schema import schema_manifest  # noqa: E402
from env.scenario.behaviors import event_state  # noqa: E402
from env.scenario.labels import LABEL_ORDER, compute_step_labels  # noqa: E402
from env.scenario.spec import load_specs  # noqa: E402
from pipeline.gl_runtime import ensure_gl_library_path  # noqa: E402

__all__ = ["main", "arc_interpolate", "SUPERVISED_LABELS"]

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

POLICY_DT = 0.5  # 策略步长（s）= 2 Hz（config/env.yaml::timing.policy_dt）
PHYSICS_DT = 0.1  # env.step 步长（s）
STEPS_PER_POLICY = 5  # POLICCY_DT / PHYSICS_DT
WINDOW_POLICIES = 6  # 3 s 目标窗口 = 6 个策略步
WINDOW_STEPS = WINDOW_POLICIES * STEPS_PER_POLICY  # 30 env steps
#: 逐帧存储的当前帧通道（v2 增加 others；v3 增加 ego_world/route_world 世界系键，
#: 供 Stage A/B 的 A4 nav 逐步重建；v4 的静态障碍段在 others 内（others 28→33 维），
#: 无需新键即可被采集/透传；nav/signal 兼容保留）
CURRENT_CHANNELS = ("ego", "od", "ld", "nav", "signal", "others", "ego_world", "route_world")
#: OD 槽位级伴随键（int64 / float32）
COMPANION_KEYS = ("od_id", "od_presence")
#: 每帧额外存的历史键（id/presence 历史不能靠行位置重建）
MEM_HISTORY_KEYS = ("od_id_hist", "od_presence_hist")
#: 需要保持 int64 的 npz 键（其余一律 float32）
INT64_KEYS = frozenset({"od_id", "od_id_hist"})
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
#: 变道执行帧 roundtrip 放宽（P1 iter2，2026-10-02）：|lane_lat| > 该值视为变道/绕行执行帧。
#: 依据：变道帧 62% 被原 6 点 roundtrip 门过滤（tollgate 829/1333），而弧模型误差与 |dθ|
#: 高度相关（corr 0.96，低速大转角下侧偏本质）；BC 动作目标只是首步 (ds,dθ)。
ROUNDTRIP_LC_LAT = 0.8
#: 放宽门的首步关键点最大误差（m）——首步对应动作监督口径，必须仍是可信弧。
ROUNDTRIP_LC_FIRST_MAX = 0.5
#: 放宽门的 6 点平均误差上限（m）——限制 3 s 窗口整体发散，保持轨迹目标质量。
ROUNDTRIP_LC_MEAN = 0.5
#: 并行采集默认/上限：单 worker env RSS ≈0.65 GB（pipeline/vector_env.ENV_RSS_PER_WORKER_MB），
#: 本机 15 GB → 建议 <=8 个 worker（超出只告警，不阻塞）。
DEFAULT_WORKERS = 0  # 0 = auto：按 CPU 核数取半、上限 MAX_RECOMMENDED_WORKERS（默认吃满多核；内存 ≈1.3GB/worker）
MAX_RECOMMENDED_WORKERS = 10  # 上限（用户要求数据集生成默认 8–10 并发）
#: worker 进程级回收间隔（spec 数）：MetaDrive ``build_env``+``close`` 实测残留
#: ≈3.5 MB/spec（与 pipeline/vector_env.DEFAULT_RECYCLE_EVERY_SPECS 同口径），
#: 长跑线性上涨；每块跑完落一个分片并重启进程把 RSS 拉回基线。0 = 不回收。
RECYCLE_EVERY_SPECS = 150
#: 分片目录/文件名（``<out>/_shards/shard_w<worker>_c<chunk>.npz``，未压缩）
SHARD_DIRNAME = "_shards"


def _resolve_workers(requested: int) -> int:
    """``--workers`` 解析：0/负值 = auto。

    auto = min(CPU 核数取半, 按**当前可用内存**折算（每 worker ≈1.3GB、留 10% 余量）, ``MAX_RECOMMENDED_WORKERS``)，至少 1。
    用户要求：数据集生成默认吃满 8–10 并发；内存口径只用于避免把机器压进 swap（2026-09-26 实测：8 workers +
    并发冒烟 → 可用内存 2GB、load 9.6、采集慢 6×，故保留内存上限但放宽余量）。
    """
    value = int(requested)
    if value > 0:
        return value
    # 用户要求：数据集生成默认吃满 8–10 并发（本机 cpu=20 → 10）。内存不足时由调用方打印警告，
    # 不做静默降级（2026-09-26 用户明确要求）。
    cpu_cap = max(1, (os.cpu_count() or 4) // 2)
    return max(1, min(cpu_cap, MAX_RECOMMENDED_WORKERS))

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
            # 保留原始 dtype（od_id 是 int64；v1 的 float32 强转会把 track id 变成浮点）
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
    roundtrip_lc_lat: float = ROUNDTRIP_LC_LAT,
    roundtrip_lc_first_max: float = ROUNDTRIP_LC_FIRST_MAX,
    roundtrip_lc_mean: float = ROUNDTRIP_LC_MEAN,
) -> List[Dict[str, Any]]:
    """按 §8.2 规则 1–4 抽取**全部 policy 帧**（v2：命中过滤不删行，记 ``train_weight=0``）。

    - ``train_weight``：1 = 可训练（BC 动作/轨迹/标签可用）；0 = 命中 terminal_window /
      not_on_lane / cut_label_unverified / roundtrip_fail / roundtrip_dense_fail 中首个原因；
    - ``wm_valid[6]``：horizon k 的目标帧存在且仍可用（未越终止、自车在车道容差内）；
    - ``frame_usable``：本帧自身是否可用（同 ``wm_valid`` 的目标判据），供
      ``pipeline.frames.lookup_from_arrays(usable_key="frame_usable")`` 复算未来掩码；
    - 未命中过滤的行也照常计算动作/轨迹字段；窗口不可算（超出记录范围）时置零，此时
      ``train_weight`` 必为 0（terminal_window），下游不得读取。
    - **变道放宽（P1 iter2）**：6 点 roundtrip 门对 |lane_lat|>``roundtrip_lc_lat`` 的
      变道/绕行执行帧放宽——首步误差 ≤ ``roundtrip_lc_first_max`` 且 6 点均值 ≤
      ``roundtrip_lc_mean`` 时不再判 roundtrip_fail（弧模型在低速大转角下有本质误差，
      原门丢弃了 ~62% 变道帧；动作目标只是首步，全窗发散由均值上限约束）。
    """
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
    key_indices = [i * STEPS_PER_POLICY - 1 for i in range(1, WINDOW_POLICIES + 1)]

    def _frame_usable(step: int) -> bool:
        """目标帧是否仍可用：存在 ∧ 未越过 episode 终止（step<=limit）∧ 自车仍在车道容差内。"""
        target = frames.get(step)
        if target is None or step > limit:
            return False
        return bool(target["on_lane"]) and _on_lane_ok(
            target["lane_lat"], target["lane_width"], on_lane_frac, on_lane_margin
        )

    out: List[Dict[str, Any]] = []
    for t in sorted(frames.keys()):
        frame = frames[t]
        # ---- 过滤判定（v2：命中不删行，只把 train_weight 置 0；按优先级取首个原因）----
        reason: Optional[str] = None
        if t + WINDOW_STEPS > limit:
            reason = "terminal_window"
        elif not frame["on_lane"] or not _on_lane_ok(
            frame["lane_lat"], frame["lane_width"], on_lane_frac, on_lane_margin
        ):
            reason = "not_on_lane"
        if reason is None:
            # 规则 4：cut*_active 仅事件 fired ∧ actor_alive 时保留
            for cut_name, event_type in (("cutin_active", "cut_in"), ("cutout_active", "cut_out")):
                if frame["labels_raw"][LABEL_ORDER.index(cut_name)] > 0.5:
                    if not any(
                        ev["type"] == event_type and ev["fired"] and ev["alive"] for ev in frame["events"]
                    ):
                        reason = "cut_label_unverified"
                        break

        # ---- 规则 3b：6 点目标 round-trip 可复现（窗口可算时才计算，供诊断与目标字段）----
        computable = (t + WINDOW_STEPS) <= last_state
        actions = np.zeros((WINDOW_POLICIES, 2), dtype=np.float64)
        interp = np.zeros((WINDOW_STEPS, 3), dtype=np.float64)
        measured_local = np.zeros((WINDOW_STEPS, 2), dtype=np.float64)
        key_err = np.full(WINDOW_POLICIES, np.nan)
        dense_err = np.full(WINDOW_STEPS, np.nan)
        if computable:
            actions = _window_actions(poses, t)
            try:
                interp = np.asarray(
                    interpolate_fn(actions, dt=POLICY_DT, hz=int(round(1.0 / PHYSICS_DT))), dtype=np.float64
                )
            except TypeError:  # env.tracking.interpolate 可能不接受 kwargs
                interp = np.asarray(interpolate_fn(actions), dtype=np.float64)
            if interp.shape[0] != WINDOW_STEPS:
                raise ValueError(f"interpolate 输出 {interp.shape}，期望 ({WINDOW_STEPS},3)")
            measured = np.stack([poses[t + j][:2] for j in range(1, WINDOW_STEPS + 1)], axis=0)
            measured_local = _to_local(measured, frame["pose"])
            key_err = np.linalg.norm(interp[key_indices, :2] - measured_local[key_indices], axis=1)
            dense_err = np.linalg.norm(interp[:, :2] - measured_local, axis=1)
            if reason is None and (
                float(key_err.mean()) > roundtrip_key_mean or float(key_err.max()) > roundtrip_key_max
            ):
                # P1 iter2：变道/绕行执行帧放宽（见 extract_samples docstring）。只改这一条
                # 判据；其余帧的 roundtrip 行为与旧版逐位一致。
                lane_change = abs(float(frame["lane_lat"])) > float(roundtrip_lc_lat)
                if not (
                    lane_change
                    and float(key_err[0]) <= float(roundtrip_lc_first_max)
                    and float(key_err.mean()) <= float(roundtrip_lc_mean)
                ):
                    reason = "roundtrip_fail"
            if reason is None and require_dense and float(dense_err.mean()) > roundtrip_dense_mean:
                reason = "roundtrip_dense_fail"

        if reason is not None:
            filter_counter[reason] += 1
        train_weight = 0.0 if reason else 1.0

        # ---- wm_valid[6]：horizon k 的目标帧存在且仍可用（Stage A 逐 horizon 掩码）----
        wm_valid = np.zeros(WINDOW_POLICIES, dtype=np.float32)
        for k in range(1, WINDOW_POLICIES + 1):
            if _frame_usable(t + k * STEPS_PER_POLICY):
                wm_valid[k - 1] = 1.0

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
                "od_id_hist": frame["od_id_hist"],
                "od_presence_hist": frame["od_presence_hist"],
                "pose": frame["pose"],
                "action": actions.astype(np.float32),
                "traj6": interp[key_indices, :2].astype(np.float32),
                "traj30": interp[:, :2].astype(np.float32),
                "traj30_measured": measured_local.astype(np.float32),
                "roundtrip_key_err": float(key_err.mean()) if computable else float("nan"),
                "roundtrip_dense_err": float(dense_err.mean()) if computable else float("nan"),
                "labels": _supervised_vector(
                    {name: float(labels_raw[i]) for i, name in enumerate(LABEL_ORDER)}, label_order
                ),
                "labels_raw": np.array(labels_raw, dtype=np.float32),
                "prev_action": prev_action.astype(np.float32),
                "lane_lat": float(frame["lane_lat"]),
                "train_weight": float(train_weight),
                "wm_valid": wm_valid,
                "frame_usable": 1.0 if _frame_usable(t) else 0.0,
                "filter_reason": reason or "",
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 单 spec 采集（单进程主循环与 spawn worker 的唯一共用实现）
# --------------------------------------------------------------------------- #

def _collect_one_spec(
    spec: Any,
    *,
    spec_index: int,
    builder: ObservationBuilder,
    label_order: Sequence[str],
    interpolate_fn: Any,
    expert_kind: str = "idm",
    max_steps: int = 600,
    on_lane_frac: float = 0.5,
    on_lane_margin: float = 0.3,
    roundtrip_key_mean: float = 0.25,
    roundtrip_key_max: float = 0.5,
    require_dense: bool = False,
    roundtrip_dense_mean: float = 0.5,
    roundtrip_lc_lat: float = ROUNDTRIP_LC_LAT,
    roundtrip_lc_first_max: float = ROUNDTRIP_LC_FIRST_MAX,
    roundtrip_lc_mean: float = ROUNDTRIP_LC_MEAN,
    traffic_density: Optional[float] = None,
) -> Dict[str, Any]:
    """采集 + 过滤单个 spec，返回可跨进程传递的逐 spec 记录。

    ``spec_index`` = 该 spec 在过滤后列表里的全局下标，直接作为 ``episode_id``：
    并行合并后 ``episode_id`` 仍连续、且与单进程逐条执行一致。异常不向上抛
    （与旧主循环同口径：单条失败不中断整批），错误记录 ``report.termination=="error"``。
    """
    env = None
    filter_counter: Counter = Counter()
    started = time.perf_counter()
    try:
        env = build_env(spec, traffic_density=traffic_density, use_render=False)
        episode = collect_episode(
            env,
            spec,
            builder=builder,
            expert_kind=str(expert_kind),
            max_steps=int(max_steps),
        )
        rows = extract_samples(
            episode,
            spec,
            episode_id=int(spec_index),
            label_order=label_order,
            interpolate_fn=interpolate_fn,
            on_lane_frac=float(on_lane_frac),
            on_lane_margin=float(on_lane_margin),
            roundtrip_key_mean=float(roundtrip_key_mean),
            roundtrip_key_max=float(roundtrip_key_max),
            filter_counter=filter_counter,
            require_dense=bool(require_dense),
            roundtrip_dense_mean=float(roundtrip_dense_mean),
            roundtrip_lc_lat=float(roundtrip_lc_lat),
            roundtrip_lc_first_max=float(roundtrip_lc_first_max),
            roundtrip_lc_mean=float(roundtrip_lc_mean),
        )
        candidates = len(episode["frames"])
        trainable = int(sum(1 for row in rows if row["train_weight"] > 0.0))
        return {
            "spec_index": int(spec_index),
            "kept": rows,  # v2：全部 policy 帧（train_weight 标过滤；键名保持以兼容 _merge_records）
            "filter_counts": filter_counter,
            "candidates": candidates,
            "steps": int(episode["steps"]),
            "elapsed_s": time.perf_counter() - started,
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
            },
        }
    except Exception as exc:  # noqa: BLE001 - 单条失败不中断整批（与 validator 同口径）
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


# --------------------------------------------------------------------------- #
# 并行采集（--workers N；spawn，每 worker 单 engine）
# --------------------------------------------------------------------------- #

def _worker_spec_indices(total: int, worker_index: int, num_workers: int) -> List[int]:
    """worker 的确定性 spec 下标（轮转 ``index % N``）：互斥且覆盖 ``range(total)``。"""
    return list(range(int(worker_index), int(total), int(num_workers)))


def _chunk_tasks(tasks: List[Tuple[int, Any]], chunk_size: int) -> List[List[Tuple[int, Any]]]:
    """把分区任务切成 ≤``chunk_size`` 的块（每块一个 worker 进程生命周期）。

    ``chunk_size<=0`` = 不分块（整段单进程）。进程级回收用于释放 MetaDrive
    ``build_env``+``close`` 每 spec ≈3.5 MB 的残留（见模块 docstring 内存说明）。
    """
    size = int(chunk_size or 0)
    if size <= 0:
        return [list(tasks)]
    return [list(tasks[start : start + size]) for start in range(0, len(tasks), size)]


def _spec_progress_line(record: Dict[str, Any]) -> str:
    """单 spec 进度行（worker 与单进程路径共用；不打印逐帧数据）。"""
    report = record.get("report", {})
    if "error" in record:
        return f"id={report.get('id', -1)} 失败：{record['error']}"
    kept = record.get("kept") or []
    return (
        f"id={report.get('id', -1)} term={report.get('termination')} steps={record.get('steps')} "
        f"cand={record.get('candidates')} stored={report.get('stored_rows', len(kept))} "
        f"trainable={report.get('retained_steps', -1)} ({record.get('elapsed_s', 0.0):.1f}s)"
    )


def _summarize_records(
    worker_index: int,
    chunk_id: int,
    records: Sequence[Dict[str, Any]],
    shard_dir: Path,
) -> Dict[str, Any]:
    """把一个块的逐 spec 记录落成 npz 分片，返回可跨进程传递的**标量摘要**。

    逐帧数据只写进 ``<shard_dir>/shard_w<worker>_c<chunk>.npz``（``np.savez`` 未压缩），
    摘要（每 spec 行数 / 候选数 / 过滤计数 / 可训练行 / report）走队列回父进程 →
    父进程 RSS 与数据量无关。整块无行（全部 error）时不写分片（``shard=None``）。
    """
    specs: List[Dict[str, Any]] = []
    rows_total = 0
    for record in records:
        kept = record.get("kept") or []
        rows_total += len(kept)
        specs.append(
            {
                "spec_index": int(record["spec_index"]),
                "rows": len(kept),
                "candidates": int(record.get("candidates", 0)),
                "trainable": int(sum(1 for row in kept if float(row["train_weight"]) > 0.0)),
                "filter_counts": dict(record.get("filter_counts", {})),
                "report": record.get("report", {}),
                "error": record.get("error"),
            }
        )
    shard: Optional[str] = None
    if rows_total > 0:
        shard = f"shard_w{int(worker_index)}_c{int(chunk_id)}.npz"
        write_shard(
            Path(shard_dir) / shard,
            [row for record in records for row in (record.get("kept") or [])],
        )
    return {
        "worker_index": int(worker_index),
        "chunk_id": int(chunk_id),
        "shard": shard,
        "specs": specs,
        "rows": rows_total,
    }


def _expert_worker(
    worker_index: int,
    chunk_id: int,
    tasks: List[Tuple[int, Any]],
    config: Dict[str, Any],
    shard_dir: str,
    out_queue: Any,
) -> None:
    """spawn 子进程入口：顺序采集本块 spec，落一个分片，只回传标量摘要。

    每个 worker 同一时刻只持有一个 env（MetaDrive 每进程单 engine）。``ensure_gl_library_path``
    必须在建 env 前调用：父进程 spawn 前已写入 ``LD_LIBRARY_PATH``（子进程启动即继承），
    这里再兜底一次，确保不触发 "Known Pipes" 崩溃。

    消息 ``(worker_index, chunk_id, summary | None)``：``summary`` = 本块分片与逐 spec
    摘要；``None`` = 本块完成（父进程据此重启进程回收内存）；``chunk_id`` 让父进程忽略
    崩溃块的迟到哨兵（不误续块）。
    """
    ensure_gl_library_path()
    builder = ObservationBuilder({})
    interpolate_fn, _ = resolve_interpolate()
    records: List[Dict[str, Any]] = []
    for spec_index, spec in tasks:
        record = _collect_one_spec(
            spec,
            spec_index=int(spec_index),
            builder=builder,
            interpolate_fn=interpolate_fn,
            **config,
        )
        records.append(record)
        print(
            f"[collect_expert] [w{int(worker_index)}] spec#{int(spec_index)} {_spec_progress_line(record)}",
            flush=True,
        )
    summary = _summarize_records(worker_index, chunk_id, records, Path(shard_dir))
    out_queue.put((int(worker_index), int(chunk_id), summary))
    out_queue.put((int(worker_index), int(chunk_id), None))


def _run_workers(
    specs: Sequence[Any],
    num_workers: int,
    config: Dict[str, Any],
    shard_dir: Path,
    recycle_every: int = RECYCLE_EVERY_SPECS,
) -> List[Dict[str, Any]]:
    """父进程侧并行编排：静态分区 + 分块 spawn，返回逐块标量摘要（乱序）。

    每个分区（``index % N``）的 spec 按 ``recycle_every`` 切块，依次由一个 spawn
    子进程处理；一块跑完落盘一个分片、进程退出、父进程重启下一块 → RSS 回落到基线。
    父进程只收摘要（O(specs) 标量），逐帧数据留在分片里，最终由 :func:`_spec_entries`
    / :func:`_concat_shards` 按 ``spec_index`` 归并，与 worker 数、分块方式无关。
    """
    total = len(specs)
    ctx = get_context("spawn")
    # spawn 子进程继承父进程 environ：先把 venv 本地 glvnd 写进 LD_LIBRARY_PATH，
    # 子进程启动时 ld.so 即可解析（worker 内部再调一次作兜底）。
    ensure_gl_library_path()
    out_queue = ctx.Queue()
    chunks: List[List[Tuple[int, Any]]] = []
    for worker_index in range(int(num_workers)):
        tasks = [(index, specs[index]) for index in _worker_spec_indices(total, worker_index, num_workers)]
        chunks.append(_chunk_tasks(tasks, recycle_every))
    procs: Dict[int, Any] = {}  # worker_index -> 当前块进程
    running_chunk: Dict[int, int] = {}  # worker_index -> 当前块号
    next_chunk: Dict[int, int] = {index: 0 for index in range(int(num_workers))}
    done: set = set()

    def _spawn(worker_index: int) -> None:
        running_chunk[worker_index] = next_chunk[worker_index]
        proc = ctx.Process(
            target=_expert_worker,
            args=(
                worker_index,
                running_chunk[worker_index],
                chunks[worker_index][running_chunk[worker_index]],
                config,
                str(shard_dir),
                out_queue,
            ),
            name=f"collect-expert-{worker_index}",
            daemon=True,
        )
        proc.start()
        procs[worker_index] = proc

    def _advance(worker_index: int) -> None:
        """当前块结束/崩溃：进入下一块（重启进程）或标记分区完成。"""
        next_chunk[worker_index] += 1
        if next_chunk[worker_index] < len(chunks[worker_index]):
            print(
                f"[collect_expert] [w{worker_index}] 进程回收重启（块 "
                f"{next_chunk[worker_index] + 1}/{len(chunks[worker_index])}）",
                flush=True,
            )
            _spawn(worker_index)
        else:
            done.add(worker_index)

    for worker_index in range(int(num_workers)):
        if chunks[worker_index] and chunks[worker_index][0]:
            _spawn(worker_index)
        else:
            done.add(worker_index)

    summaries: List[Dict[str, Any]] = []
    specs_done = 0
    dead: set = set()
    try:
        while len(done) < int(num_workers):
            try:
                worker_index, chunk_id, summary = out_queue.get(timeout=1.0)
            except Empty:
                for worker_index, proc in list(procs.items()):
                    if proc.is_alive():
                        continue
                    proc.join(timeout=0.5)
                    del procs[worker_index]
                    dead.add(worker_index)
                    print(
                        f"[collect_expert] worker {worker_index} 异常退出（exitcode={proc.exitcode}）；"
                        "崩溃块的 spec 不会产出（其余块继续，见末尾 missing 统计）",
                        flush=True,
                    )
                    _advance(worker_index)
                continue
            if summary is None:
                proc = procs.pop(worker_index, None)
                if proc is None or running_chunk.get(worker_index) != int(chunk_id):
                    continue  # 崩溃块的迟到哨兵：忽略，避免误续
                proc.join(timeout=30.0)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5.0)
                    print(f"[collect_expert] worker {worker_index} 退出超时，已强制回收", flush=True)
                _advance(worker_index)
                continue
            summaries.append(summary)
            specs_done += len(summary.get("specs", []))
            trainable = int(sum(int(spec.get("trainable", 0)) for spec in summary.get("specs", [])))
            print(
                f"[collect_expert] {specs_done}/{total} [w{worker_index}] 块 {int(chunk_id) + 1} 完成："
                f"{len(summary.get('specs', []))} specs, {int(summary.get('rows', 0))} rows, "
                f"{trainable} trainable → {summary.get('shard') or '空块（无行）'}",
                flush=True,
            )
        # worker 正常退出会 flush 队列；这里兜底回收崩溃 worker 已入队但未读的摘要
        while True:
            try:
                _, _, summary = out_queue.get(timeout=1.0)
            except Empty:
                break
            if summary is not None:
                summaries.append(summary)
    finally:
        out_queue.close()
        for proc in procs.values():
            if proc.is_alive():
                proc.terminate()
            proc.join(timeout=10.0)
    if dead:
        print(f"[collect_expert] 异常退出的 worker：{sorted(dead)}", flush=True)
    return summaries


def _collect_single(
    specs: Sequence[Any],
    config: Dict[str, Any],
    shard_dir: Path,
    builder: ObservationBuilder,
    interpolate_fn: Any,
    recycle_every: int = RECYCLE_EVERY_SPECS,
) -> List[Dict[str, Any]]:
    """``--workers 1``：与 worker **同一条分片落盘路径**（只是不 spawn 进程）。

    每 ``recycle_every`` 条 spec 落一个分片并释放行引用，父进程 RSS 不随 specs 增长。
    """
    chunks = _chunk_tasks([(index, spec) for index, spec in enumerate(specs)], int(recycle_every))
    summaries: List[Dict[str, Any]] = []
    done = 0
    for chunk_id, chunk in enumerate(chunks):
        records: List[Dict[str, Any]] = []
        for spec_index, spec in chunk:
            record = _collect_one_spec(
                spec, spec_index=int(spec_index), builder=builder, interpolate_fn=interpolate_fn, **config
            )
            records.append(record)
            done += 1
            print(f"[collect_expert] {done}/{len(specs)} {_spec_progress_line(record)}", flush=True)
        summary = _summarize_records(0, chunk_id, records, shard_dir)
        summaries.append(summary)
        trainable = int(sum(int(spec.get("trainable", 0)) for spec in summary["specs"]))
        print(
            f"[collect_expert] 块 {chunk_id + 1}/{len(chunks)} 完成：{len(chunk)} specs, "
            f"{summary['rows']} rows, {trainable} trainable → {summary['shard'] or '空块（无行）'}",
            flush=True,
        )
    return summaries


def _spec_entries(
    summaries: Sequence[Dict[str, Any]],
    total_specs: int,
) -> Tuple[List[Dict[str, Any]], int, List[int]]:
    """按全局 spec 下标展开摘要，返回 ``(entries, total_rows, missing_indices)``。

    ``entries`` 按原始 spec 顺序（``spec_index`` 升序）排列；每项带最终全局行偏移
    ``start`` 与分片内偏移 ``shard_offset``（分片内 spec 也是下标升序，见
    :func:`_summarize_records` 的写入顺序）。``missing_indices`` = 无摘要的 spec
    （单进程恒为空；并行下 worker 崩溃才会出现）。
    """
    entries: List[Dict[str, Any]] = []
    offsets: Dict[str, int] = {}
    for summary in summaries:
        shard = summary.get("shard")
        for spec in summary.get("specs", []):
            entry = dict(spec)
            entry["shard"] = shard
            entry["shard_offset"] = int(offsets.get(shard, 0)) if shard else 0
            entry["worker_index"] = int(summary.get("worker_index", -1))
            entry["chunk_id"] = int(summary.get("chunk_id", -1))
            if shard:
                offsets[shard] = entry["shard_offset"] + int(entry["rows"])
            entries.append(entry)
    entries.sort(key=lambda item: int(item["spec_index"]))
    covered = {int(entry["spec_index"]) for entry in entries}
    missing = [index for index in range(int(total_specs)) if index not in covered]
    start = 0
    for entry in entries:
        entry["start"] = start
        start += int(entry["rows"])
    return entries, start, missing


def _merge_filter_counts(entries: Sequence[Dict[str, Any]]) -> Counter:
    """按 spec 顺序合并过滤计数（键顺序 = 首次出现顺序，与单进程逐条累加一致）。"""
    counter: Counter = Counter()
    for entry in entries:
        counter.update(entry.get("filter_counts", {}))
    return counter


def _shard_segments(entries: Sequence[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int, int]]]:
    """分片名 → ``[(分片内行偏移, 行数, 最终全局行偏移), ...]``。"""
    segments: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    for entry in entries:
        if entry.get("shard") and int(entry["rows"]) > 0:
            segments[str(entry["shard"])].append(
                (int(entry["shard_offset"]), int(entry["rows"]), int(entry["start"]))
            )
    return dict(segments)


def _concat_shards(
    shard_dir: Path,
    segments: Dict[str, List[Tuple[int, int, int]]],
    total_rows: int,
) -> Dict[str, np.ndarray]:
    """逐分片、逐键拼接成最终逐行数组（峰值 ≈ 输出数组一份 + 单分片单键临时数组）。

    每个键先按 ``total_rows`` 分配最终数组，再按分段把各分片对应行拷入；分片与键都
    只读一份，不把全部数据复制多份。
    """
    names = sorted(segments)
    if not names:
        return {}
    arrays: Dict[str, np.ndarray] = {}
    with ExitStack() as stack:
        payloads = {name: stack.enter_context(np.load(Path(shard_dir) / name)) for name in names}
        keys = list(payloads[names[0]].files)
        for key in keys:
            out: Optional[np.ndarray] = None
            for name in names:
                payload = payloads[name]
                if key not in payload.files:
                    raise ValueError(f"分片 {name} 缺少键 {key}（schema 不一致）")
                chunk = payload[key]
                if out is None:
                    out = np.empty((int(total_rows),) + tuple(chunk.shape[1:]), dtype=chunk.dtype)
                for offset, count, start in segments[name]:
                    out[start:start + count] = chunk[offset:offset + count]
            if out is None:  # pragma: no cover - names 非空时必然分配
                raise RuntimeError(f"键 {key} 没有任何分片数据")
            arrays[key] = out
    return arrays


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
    """按 (difficulty, 主几何) 组配平（规则 5，v2 权重感知）。

    - ``weights``（默认）：对 ``train_weight>0`` 的行给权重 ``n_trainable / (n_groups * n_group)``，
      使每组对损失的贡献近似相等；``train_weight=0`` 的行 ``sample_weight=1``（下游先乘
      ``train_weight``，不参与损失）；
    - ``cap``：每组最多 ``ceil(ratio * 最小非空组可训练行数)`` 个可训练行参与损失——
      超出行的 ``sample_weight`` 置 0（**不删行**：行是历史/未来精确查表所需的帧；
      v1 的物理删除会和 Stage A 的目标缺失问题冲突）；组内按 (spec_id, step) 确定性等距抽样；
    - ``none``：权重恒 1。

    计数口径 = 行数；加权口径 = 权重和（见 report["counts"] / report["weighted"]）。
    返回统计 dict（写入 meta/report）。
    """
    groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[(sample["difficulty"], sample["geometry"])].append(index)
    trainable = [index for index, sample in enumerate(samples) if float(sample.get("train_weight", 1.0)) > 0.0]
    trainable_groups: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index in trainable:
        trainable_groups[(samples[index]["difficulty"], samples[index]["geometry"])].append(index)
    weights = np.ones(len(samples), dtype=np.float64)
    group_key = np.array(
        [f"{sample['difficulty']}/{sample['geometry']}" for sample in samples], dtype="U64"
    ) if samples else np.zeros(0, dtype="U64")
    stats: Dict[str, Any] = {
        "mode": mode,
        "ratio": float(ratio),
        "group_counts_before": {f"{k[0]}/{k[1]}": len(v) for k, v in sorted(groups.items())},
        "group_trainable_before": {
            f"{k[0]}/{k[1]}": len(v) for k, v in sorted(trainable_groups.items())
        },
        "counts": {"rows": len(samples), "trainable_rows": len(trainable)},
    }
    if mode == "none" or not samples or not trainable:
        stats["group_counts_after"] = dict(stats["group_counts_before"])
        stats["weight_range"] = [1.0, 1.0]
        return {"sample_weight": weights, "balance_group": group_key, "stats": stats}
    if mode == "weights":
        n_groups = max(1, len(trainable_groups))
        n_trainable = len(trainable)
        for key, indices in trainable_groups.items():
            weights[indices] = float(n_trainable) / (n_groups * len(indices))
        stats["weight_range"] = [float(weights[trainable].min()), float(weights[trainable].max())]
    elif mode == "cap":
        min_count = min(len(indices) for indices in trainable_groups.values())
        cap = max(1, int(math.ceil(float(ratio) * min_count)))
        dropped: List[int] = []
        for key in sorted(trainable_groups):
            indices = sorted(
                trainable_groups[key], key=lambda i: (samples[i]["spec_id"], samples[i]["step"])
            )
            if len(indices) <= cap:
                continue
            stride = len(indices) / float(cap)
            picks = {indices[min(len(indices) - 1, int(round(i * stride)))] for i in range(cap)}
            dropped.extend(index for index in indices if index not in picks)
        weights[dropped] = 0.0
        stats["cap"] = cap
        stats["min_group_trainable"] = min_count
        stats["dropped_indices"] = sorted(dropped)
        stats["counts"]["trainable_after_cap"] = int((weights > 0.0).sum())
    else:
        raise ValueError(f"未知配平模式 {mode!r}")
    # "配平后有效行" = train_weight>0 且 sample_weight>0（下游损失真正使用的行）
    stats["group_counts_after"] = {
        f"{k[0]}/{k[1]}": sum(
            1
            for index in v
            if weights[index] > 0.0 and float(samples[index].get("train_weight", 1.0)) > 0.0
        )
        for k, v in sorted(groups.items())
    }
    return {"sample_weight": weights, "balance_group": group_key, "stats": stats}


def _cap_picks(length: int, cap: int) -> set:
    """cap 模式的确定性等距抽样位置（与 :func:`apply_balance` 逐字一致）。"""
    stride = length / float(cap)
    return {min(length - 1, int(round(i * stride))) for i in range(cap)}


def _entry_group(entry: Dict[str, Any]) -> Tuple[str, str]:
    report = entry["report"]
    return str(report.get("difficulty", "unknown")), str(report.get("geometry", "unknown"))


def balance_from_specs(
    entries: Sequence[Dict[str, Any]],
    train_weight: np.ndarray,
    *,
    mode: str,
    ratio: float,
) -> Dict[str, Any]:
    """由逐 spec 摘要 + 最终 ``train_weight`` 数组生成配平权重（语义 = :func:`apply_balance`）。

    额外内存 O(specs + groups)：组计数按 spec 累加；``cap`` 模式的抽样位置也按
    ``(spec_id, spec_index)`` 序、逐 spec 的 ``train_weight`` 行内推进，不复制逐帧数据。
    ``entries`` 需按**全局行序**给出，且每项含 ``start`` / ``rows`` / ``trainable`` /
    ``report{difficulty,geometry,id}``；返回的 ``sample_weight`` / ``balance_group``
    即最终 npz 的两列。
    """
    total_rows = int(train_weight.shape[0])
    weights = np.ones(total_rows, dtype=np.float64)
    group_key = np.empty(total_rows, dtype="U64")
    group_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    trainable_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for entry in entries:
        rows = int(entry["rows"])
        if rows <= 0:  # 错误 spec：无行，不参与分组（与 apply_balance 只遍历样本一致）
            continue
        key = _entry_group(entry)
        name = f"{key[0]}/{key[1]}"
        group_key[entry["start"]:entry["start"] + rows] = name
        group_counts[key] += rows
        trainable_counts[key] += int(entry["trainable"])
    trainable_groups = {key: count for key, count in trainable_counts.items() if count > 0}
    stats: Dict[str, Any] = {
        "mode": mode,
        "ratio": float(ratio),
        "group_counts_before": {f"{k[0]}/{k[1]}": group_counts[k] for k in sorted(group_counts)},
        "group_trainable_before": {
            f"{k[0]}/{k[1]}": trainable_counts[k] for k in sorted(trainable_counts) if trainable_counts[k] > 0
        },
        "counts": {"rows": total_rows, "trainable_rows": int((train_weight > 0.0).sum())},
    }
    if mode == "none" or total_rows == 0 or not trainable_groups:
        stats["group_counts_after"] = dict(stats["group_counts_before"])
        stats["weight_range"] = [1.0, 1.0]
        return {"sample_weight": weights, "balance_group": group_key, "stats": stats}
    if mode == "weights":
        n_groups = max(1, len(trainable_groups))
        n_trainable = stats["counts"]["trainable_rows"]
        group_weight = {key: float(n_trainable) / (n_groups * count) for key, count in trainable_groups.items()}
        for entry in entries:
            rows = int(entry["rows"])
            if rows <= 0:
                continue
            weights[entry["start"]:entry["start"] + rows] = group_weight.get(_entry_group(entry), 1.0)  # 0 可训练行组：保持 1（下游由 train_weight 置零；与 apply_balance 一致）
        weights[train_weight <= 0.0] = 1.0  # 过滤行保持 1（下游先乘 train_weight）
        stats["weight_range"] = [min(group_weight.values()), max(group_weight.values())]
        after = dict(trainable_counts)
    elif mode == "cap":
        min_count = min(trainable_groups.values())
        cap = max(1, int(math.ceil(float(ratio) * min_count)))
        by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for entry in entries:
            if int(entry["rows"]) > 0 and int(entry["trainable"]) > 0:
                by_group[_entry_group(entry)].append(entry)
        after = dict(trainable_counts)
        dropped_total = 0
        for key in sorted(trainable_counts):
            length = int(trainable_counts[key])
            if length <= cap:
                continue
            picks = _cap_picks(length, cap)
            ordered = sorted(
                by_group[key],
                key=lambda item: (int(item["report"].get("id", -1)), int(item["spec_index"])),
            )
            position = 0
            for entry in ordered:
                start, rows = int(entry["start"]), int(entry["rows"])
                for offset in np.nonzero(train_weight[start:start + rows] > 0.0)[0]:
                    if position not in picks:
                        weights[start + int(offset)] = 0.0
                        dropped_total += 1
                    position += 1
            after[key] = len(picks)
        stats["cap"] = cap
        stats["min_group_trainable"] = min_count
        # 与 apply_balance 同口径（权重>0 的行数，含 train_weight=0 的行）
        stats["counts"]["trainable_after_cap"] = total_rows - dropped_total
        stats["dropped_indices_count"] = dropped_total
    else:
        raise ValueError(f"未知配平模式 {mode!r}")
    stats["group_counts_after"] = {
        f"{k[0]}/{k[1]}": after.get(k, 0) for k in sorted(group_counts)
    }
    return {"sample_weight": weights, "balance_group": group_key, "stats": stats}


# --------------------------------------------------------------------------- #
# 保存
# --------------------------------------------------------------------------- #

def label_statistics(
    samples: Sequence[Dict[str, Any]], label_order: Sequence[str]
) -> Tuple[Dict[str, int], Dict[str, float]]:
    """per-label 正样本的两套口径：``counts`` = 行数，``weighted`` = 有效权重和。

    - 只统计 ``train_weight>0`` 的行（可训练）；权重 = ``train_weight * sample_weight``；
    - 计数口径是保守下界（dataset_gate 用），加权口径反映实际损失贡献。
    """
    counts = {str(name): 0 for name in label_order}
    weighted = {str(name): 0.0 for name in label_order}
    for sample in samples:
        gate = float(sample.get("train_weight", 1.0))
        if gate <= 0.0:
            continue
        weight = gate * float(sample.get("sample_weight", 1.0))
        labels = sample["labels"]
        for index, name in enumerate(label_order):
            if float(labels[index]) > 0.5:
                counts[str(name)] += 1
                weighted[str(name)] += weight
    return counts, {name: float(value) for name, value in weighted.items()}


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


def _stack_arrays(
    arrays: List[np.ndarray], key: str, shape: Tuple[int, ...], dtype: Any = np.float32
) -> np.ndarray:
    """按 key 堆叠并校验形状；dtype 显式给定（``od_id``/``od_id_hist`` 必须 int64）。"""
    arr = np.stack([np.asarray(item) for item in arrays], axis=0)
    if arr.dtype != np.dtype(dtype):
        arr = arr.astype(dtype)
    expected = (len(arrays),) + shape
    if arr.shape != expected:
        raise ValueError(f"样本字段 {key} 形状 {arr.shape}，期望 {expected}")
    return arr


def _npz_schema_manifest(num_slots: int, frames: int, others_dim: int, label_count: int) -> Dict[str, Any]:
    """``expert_bc.npz`` 键清单（形状/dtype/语义/单位）——下游契约。"""
    n = "<N>"  # 行数
    manifest: Dict[str, Any] = {
        "episode_id": {"shape": [n], "dtype": "int64", "semantics": "spec 全局下标；同一 episode 的行共享它", "unit": "-"},
        "step": {"shape": [n], "dtype": "int64", "semantics": "env step（0.1 s；策略帧每 5 步一行）", "unit": "env step"},
        "spec_id": {"shape": [n], "dtype": "int64", "semantics": "场景 spec id", "unit": "-"},
        "seed": {"shape": [n], "dtype": "int64", "semantics": "MetaDrive start_seed", "unit": "-"},
        "split": {"shape": [n], "dtype": "U16", "semantics": "train/val", "unit": "-"},
        "difficulty": {"shape": [n], "dtype": "U16", "semantics": "easy/medium/hard", "unit": "-"},
        "geometry": {"shape": [n], "dtype": "U32", "semantics": "主几何标签（配平键）", "unit": "-"},
        "pose": {"shape": [n, 3], "dtype": "float32", "semantics": "该帧 ego 世界位姿 (x,y,theta)，历史/未来对齐用", "unit": "m, m, rad"},
        "train_weight": {
            "shape": [n],
            "dtype": "float32",
            "semantics": "过滤门（1=可训练；0=命中 terminal_window/not_on_lane/cut_label_unverified/roundtrip_fail/roundtrip_dense_fail 中首个原因）。BC 损失前先乘它",
            "unit": "0/1",
        },
        "frame_usable": {
            "shape": [n],
            "dtype": "float32",
            "semantics": "该帧是否可用（未越终止 ∧ 自车在车道容差内）；pipeline.frames 的 usable 输入",
            "unit": "0/1",
        },
        "wm_valid": {
            "shape": [n, frames],
            "dtype": "float32",
            "semantics": "Stage A：horizon k 的目标帧 (step+5(k+1)) 存在且仍可用（未越终止 ∧ 目标帧 on-lane）",
            "unit": "0/1",
        },
        "hist_valid": {
            "shape": [n, frames],
            "dtype": "float32",
            "semantics": "采集时真实缓冲长度算出的历史槽有效性（1=真实帧；0=预热补位）。v2 训练侧应以 pipeline.frames 精确查表为准",
            "unit": "0/1",
        },
        "sample_weight": {"shape": [n], "dtype": "float32", "semantics": "配平权重（legacy 名；v2 规范名 balance_weight，两者同值）", "unit": "-"},
        "balance_weight": {"shape": [n], "dtype": "float32", "semantics": "配平权重（v2 规范名；trainer row_balance_weights 消费）；有效权重 = train_weight*balance_weight", "unit": "-"},
        "filter_reason": {"shape": [n], "dtype": "U24", "semantics": "train_weight=0 的原因（terminal_window/not_on_lane/cut_label_unverified/roundtrip_fail/roundtrip_dense_fail）；可训练行为空串", "unit": "-"},
        "balance_group": {"shape": [n], "dtype": "U64", "semantics": "difficulty/geometry 组名", "unit": "-"},
        "action": {"shape": [n, frames, 2], "dtype": "float32", "semantics": "6 个专家策略动作 (ds,dθ)（窗口不可算的零权重行为 0）", "unit": "m, rad"},
        "traj6": {"shape": [n, frames, 2], "dtype": "float32", "semantics": "动作端点关键轨迹（自车系，t=0.5..3.0 s）", "unit": "m"},
        "traj30": {"shape": [n, 30, 2], "dtype": "float32", "semantics": "10 Hz 插值轨迹（自车系）", "unit": "m"},
        "traj30_measured": {"shape": [n, 30, 2], "dtype": "float32", "semantics": "实测位姿轨迹（自车系），round-trip 对照", "unit": "m"},
        "roundtrip_key_err": {"shape": [n], "dtype": "float32", "semantics": "6 关键点误差均值（窗口不可算=NaN）", "unit": "m"},
        "roundtrip_dense_err": {"shape": [n], "dtype": "float32", "semantics": "30 点误差均值（窗口不可算=NaN）", "unit": "m"},
        "prev_action": {"shape": [n, 2], "dtype": "float32", "semantics": "上一策略步专家动作（ego 通道 reserved 两维的来源）", "unit": "m, rad"},
        "lane_lat": {"shape": [n], "dtype": "float32", "semantics": "该帧 ego 相对车道中心横向偏差", "unit": "m"},
        "labels": {"shape": [n, label_count], "dtype": "float32", "semantics": f"8 个受监督 router 标签（顺序=meta.label_names）", "unit": "0/1"},
        "labels_raw": {"shape": [n, 9], "dtype": "float32", "semantics": "9 个原始标签（顺序=meta.raw_label_names）", "unit": "0/1"},
    }
    for channel in CURRENT_CHANNELS:
        slots = _channel_slots(channel, num_slots)
        manifest[channel] = {"shape": [n, slots, _channel_dim(channel)],
                             "dtype": "float32", "semantics": "当前帧通道（见 meta.schema.frame）", "unit": "-"}
        manifest[f"{channel}_mask"] = {
            "shape": [n, slots],
            "dtype": "float32", "semantics": "当前帧掩码", "unit": "0/1",
        }
    manifest["od_id"] = {"shape": [n, num_slots], "dtype": "int64", "semantics": "OD track id（-1=空槽），episode 内稳定", "unit": "-"}
    manifest["od_presence"] = {"shape": [n, num_slots], "dtype": "float32", "semantics": "OD 本帧观测标志（0=出盒未释放，特征陈旧）", "unit": "0/1"}
    manifest["od_id_hist"] = {"shape": [n, frames, num_slots], "dtype": "int64", "semantics": "6 帧 od_id（同槽位，不做 SE(2) 变换）", "unit": "-"}
    manifest["od_presence_hist"] = {"shape": [n, frames, num_slots], "dtype": "float32", "semantics": "6 帧 od_presence", "unit": "0/1"}
    return manifest


def _channel_slots(channel: str, num_slots: int) -> int:
    """当前帧通道的槽位数（与 builder 输出一致；OD/LD = 数据集槽位数，world 定长）。"""
    from env.obs.world import ROUTE_WORLD_MAX_POINTS

    if channel == "route_world":
        return int(ROUTE_WORLD_MAX_POINTS)
    if channel in ("ego", "nav", "signal", "others", "ego_world"):
        return 1
    return int(num_slots)


def _channel_dim(channel: str) -> int:
    """当前帧通道的特征维（与 builder 输出一致）。"""
    from env.obs.ego import EGO_DIM
    from env.obs.ld import LDChannel
    from env.obs.nav import NavChannel
    from env.obs.others import OTHERS_HEAD_DIM, road_class_labels
    from env.obs.signal import SIGNAL_DIM
    from env.obs.world import EGO_WORLD_DIM, ROUTE_WORLD_DIM

    dims = {
        "ego": EGO_DIM,
        "od": 9,
        "ld": LDChannel.feature_dim,
        "nav": NavChannel.feature_dim,
        "signal": SIGNAL_DIM,
        "others": OTHERS_HEAD_DIM + len(road_class_labels()),
        "ego_world": EGO_WORLD_DIM,
        "route_world": ROUTE_WORLD_DIM,
    }
    return int(dims[channel])


def _samples_to_arrays(samples: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """把逐行样本字典转成逐行 npz 数组（**不含**全局量 sample_weight/balance_weight/balance_group）。

    单一实现供两条路径复用：worker 写分片（``write_shard``）与最终 ``save_dataset``；
    分片键序 = 最终 npz 键序。
    """
    arrays: Dict[str, np.ndarray] = {}
    for channel in CURRENT_CHANNELS:
        if channel not in samples[0]["obs"]:
            continue
        sample_shape = tuple(np.asarray(samples[0]["obs"][channel]).shape)
        dtype = np.int64 if channel in INT64_KEYS else np.float32
        arrays[channel] = _stack_arrays([sample["obs"][channel] for sample in samples], channel, sample_shape, dtype)
        mask_key = f"{channel}_mask"
        if mask_key in samples[0]["obs"]:
            mask_shape = tuple(np.asarray(samples[0]["obs"][mask_key]).shape)
            arrays[mask_key] = _stack_arrays(
                [sample["obs"][mask_key] for sample in samples], mask_key, mask_shape
            )
    # 伴随数组（v2）：OD 槽位身份/存在性
    for key in COMPANION_KEYS:
        if key not in samples[0]["obs"]:
            continue
        shape = tuple(np.asarray(samples[0]["obs"][key]).shape)
        arrays[key] = _stack_arrays(
            [sample["obs"][key] for sample in samples], key, shape, np.int64 if key in INT64_KEYS else np.float32
        )
    for key in MEM_HISTORY_KEYS:
        shape = tuple(np.asarray(samples[0][key]).shape)
        arrays[key] = _stack_arrays(
            [sample[key] for sample in samples], key, shape, np.int64 if key in INT64_KEYS else np.float32
        )
    arrays["hist_valid"] = _stack_arrays([sample["hist_valid"] for sample in samples], "hist_valid", (6,))
    arrays["wm_valid"] = _stack_arrays([sample["wm_valid"] for sample in samples], "wm_valid", (WINDOW_POLICIES,))
    arrays["frame_usable"] = np.array([sample["frame_usable"] for sample in samples], dtype=np.float32)
    arrays["train_weight"] = np.array([sample["train_weight"] for sample in samples], dtype=np.float32)
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
    arrays["filter_reason"] = np.array(
        [str(sample.get("filter_reason", "")) for sample in samples], dtype="U24"
    )
    arrays["lane_lat"] = np.array([sample["lane_lat"] for sample in samples], dtype=np.float32)
    return arrays


def write_shard(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    """把一个块的逐行记录写成**未压缩** npz 分片（worker 进程内调用；快、可逐键读回）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **_samples_to_arrays(rows))


def _canonical_key_order(arrays: Dict[str, np.ndarray]) -> List[str]:
    """npz 成员顺序 = v2 ``save_dataset`` 的顺序（仅影响 zip 内排列，便于与旧实现逐字节对比）。"""
    order: List[str] = []
    for channel in CURRENT_CHANNELS:
        if channel in arrays:
            order.append(channel)
            if f"{channel}_mask" in arrays:
                order.append(f"{channel}_mask")
    for key in (
        *COMPANION_KEYS,
        *MEM_HISTORY_KEYS,
        "hist_valid",
        "wm_valid",
        "frame_usable",
        "train_weight",
        "pose",
        "action",
        "traj6",
        "traj30",
        "traj30_measured",
        "roundtrip_key_err",
        "roundtrip_dense_err",
        "labels",
        "labels_raw",
        "prev_action",
        "episode_id",
        "step",
        "spec_id",
        "seed",
        "difficulty",
        "geometry",
        "split",
        "sample_weight",
        "balance_weight",
        "balance_group",
        "filter_reason",
        "lane_lat",
    ):
        if key in arrays:
            order.append(key)
    known = set(order)
    order.extend(key for key in arrays if key not in known)
    return order


def _write_dataset(
    out_dir: Path,
    arrays: Dict[str, np.ndarray],
    *,
    label_order: Sequence[str],
    builder: ObservationBuilder,
    config: Dict[str, Any],
    report: Dict[str, Any],
    num_slots: int,
) -> Dict[str, str]:
    """写 ``expert_bc.npz``（压缩）/ ``expert_bc.meta.json`` / ``report.json``，返回路径表。

    v2：``od_id``/``od_id_hist`` 保持 int64；meta 写入完整 schema 清单（``schema`` +
    ``dataset_schema``）；``history_storage="per_frame_v2"``。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "expert_bc.npz"
    meta_path = out_dir / "expert_bc.meta.json"
    report_path = out_dir / "report.json"
    np.savez_compressed(npz_path, **{key: arrays[key] for key in _canonical_key_order(arrays)})

    others_dim = _channel_dim("others")
    meta = {
        "schema_version": 2,
        "kind": "bc_expert",
        "created_by": "tools/collect_expert.py",
        "obs_fingerprint": _obs_fingerprint(),
        "obs_schema_version": OBS_SCHEMA_VERSION,
        "label_names": list(label_order),
        "raw_label_names": list(LABEL_ORDER),
        "channel_shapes": {
            name: list(arrays[name].shape[1:]) for name in CURRENT_CHANNELS if name in arrays
        },
        "channel_alignments": _alignment_meta(builder),
        "kinematics_source": report["kinematics_source"],
        "config": config,
        "count": int(arrays["train_weight"].shape[0]),
        "history_storage": "per_frame_v2",  # 每帧存当前通道 + od_id/presence 历史；其余按 (episode_id, step) 精确查表
        "history_lookup": {
            "module": "pipeline.frames",
            "api": "FrameLookup.build_history(episode_id, step, stride=5, k=6) / build_future(...)",
            "window_anchor": "floor(step/stride)*stride（stride=5）；历史取 base-5j（旧→新），未来取 base+5(k+1)",
            "missing_frame": "hist_valid=0 / mask=0 / 特征 0 / od_id=-1",
        },
        "weight_semantics": {
            "train_weight": "过滤门（0/1）；terminal_window / not_on_lane / cut_label_unverified / roundtrip_fail / roundtrip_dense_fail 命中 → 0",
            "balance_weight": "配平权重（weights 模式：组均衡；cap 模式：超限行置 0；none：1）；npz 同时存 legacy 别名 sample_weight（同值）",
            "effective_weight": "train_weight * balance_weight（下游损失前必须相乘；pipeline.trainer.row_action_weights 的唯一入口）",
            "wm_valid": "Stage A 逐 horizon 掩码：目标帧存在且仍可用（未越终止 ∧ on-lane）",
        },
        "schema": schema_manifest(
            num_slots=int(num_slots),
            frames=6,
            others_dim=others_dim,
        ),
        "dataset_schema": _npz_schema_manifest(
            num_slots=int(num_slots),
            frames=6,
            others_dim=others_dim,
            label_count=len(label_order),
        ),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"npz": str(npz_path), "meta": str(meta_path), "report": str(report_path)}


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
    """由全量逐行样本直接保存（单进程旧路径 / 测试用；worker 路径见 ``write_shard``）。"""
    arrays = _samples_to_arrays(samples)
    arrays["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)
    arrays["balance_weight"] = np.asarray(sample_weight, dtype=np.float32)  # v2 规范名（trainer 消费）
    arrays["balance_group"] = np.asarray(balance_group, dtype="U64")
    num_slots = int(np.asarray(samples[0]["obs"]["od"]).shape[0]) if "od" in samples[0]["obs"] else 16
    return _write_dataset(
        out_dir, arrays, label_order=label_order, builder=builder, config=config, report=report,
        num_slots=num_slots,
    )


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
    parser.add_argument(
        "--roundtrip-lc-lat", type=float, default=ROUNDTRIP_LC_LAT,
        help="变道放宽：|lane_lat| 超过该值的执行帧走放宽门（默认 0.8 m）",
    )
    parser.add_argument(
        "--roundtrip-lc-first-max", type=float, default=ROUNDTRIP_LC_FIRST_MAX,
        help="变道放宽：首步关键点最大误差阈值（m，默认 0.5）",
    )
    parser.add_argument(
        "--roundtrip-lc-mean", type=float, default=ROUNDTRIP_LC_MEAN,
        help="变道放宽：6 点平均误差上限（m，默认 0.5）",
    )
    parser.add_argument("--balance", choices=("weights", "cap", "none"), default="weights", help="配平模式（规则 5）")
    parser.add_argument("--balance-ratio", type=float, default=3.0, help="cap 模式下最大组/最小组样本比")
    parser.add_argument("--seed", type=int, default=0, help="工具随机种子（仅用于诊断）")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="并行采集的 spawn worker 数（0=auto：cpu 核数取半、上限 8；默认 auto）；每 worker 峰值 ≈1.3GB 内存，本机建议 <=8",
    )
    parser.add_argument(
        "--recycle-every",
        type=int,
        default=RECYCLE_EVERY_SPECS,
        help="每跑满 N 条 spec 落一个分片（--workers>1 时同时重启进程回收内存；默认 150；0=不分块）",
    )
    parser.add_argument(
        "--keep-shards",
        action="store_true",
        help="合并后保留 <out>/_shards/ 分片（默认删除；调试 / 后处理用）",
    )
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
    # 每个 spec 的采集参数（单进程直调 / worker 经 **config 传入，保证两条路径同一实现）
    spec_config: Dict[str, Any] = {
        "label_order": tuple(label_order),
        "expert_kind": str(args.expert),
        "max_steps": int(args.max_steps),
        "on_lane_frac": float(args.on_lane_frac),
        "on_lane_margin": float(args.on_lane_margin),
        "roundtrip_key_mean": float(args.roundtrip_key_mean),
        "roundtrip_key_max": float(args.roundtrip_key_max),
        "require_dense": bool(args.require_dense),
        "roundtrip_dense_mean": float(args.roundtrip_dense_mean),
        "roundtrip_lc_lat": float(args.roundtrip_lc_lat),
        "roundtrip_lc_first_max": float(args.roundtrip_lc_first_max),
        "roundtrip_lc_mean": float(args.roundtrip_lc_mean),
        "traffic_density": args.traffic_density,
    }
    num_workers = _resolve_workers(args.workers)
    if int(args.workers) <= 0:  # 0 = auto：CPU 取半、上限 MAX_RECOMMENDED_WORKERS（用户要求默认 8–10）
        try:
            avail_gb = 0.0
            with open("/proc/meminfo") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        avail_gb = int(line.split()[1]) / 1024.0 / 1024.0
                        break
            if avail_gb and avail_gb < num_workers * 1.3:
                print(
                    f"[collect_expert] 警告：可用内存 {avail_gb:.1f}GB < workers×1.3GB={num_workers * 1.3:.1f}GB，"
                    "可能与并发任务抖动（本项不做静默降级；需要时显式 --workers 调低）",
                    flush=True,
                )
        except OSError:  # pragma: no cover
            pass
        print(
            f"[collect_expert] workers=auto → {num_workers}（cpu={os.cpu_count()}，上限 {MAX_RECOMMENDED_WORKERS}；"
            "每 worker 峰值 ≈1.3GB）",
            flush=True,
        )
    if num_workers > len(specs):
        print(f"[collect_expert] --workers {num_workers} > specs {len(specs)}，按 specs 数收敛", flush=True)
        num_workers = len(specs)
    if num_workers > MAX_RECOMMENDED_WORKERS:
        print(
            f"[collect_expert] 警告：--workers {num_workers} > 建议上限 {MAX_RECOMMENDED_WORKERS}"
            "（峰值 ≈1.3GB/worker，本机 15GB）",
            flush=True,
        )
    print(
        f"[collect_expert] specs={len(specs)} expert={args.expert} kinematics={kinematics_source} "
        f"labels={list(label_order)}"
        + (f" workers={num_workers} recycle_every={int(args.recycle_every)}" if num_workers > 1 else ""),
        flush=True,
    )

    out_dir = Path(args.out)
    shard_dir = out_dir / SHARD_DIRNAME
    if shard_dir.exists():
        shutil.rmtree(shard_dir, ignore_errors=True)
        print(f"[collect_expert] 清理旧分片目录 {shard_dir}", flush=True)

    summaries: List[Dict[str, Any]]
    if num_workers > 1:
        summaries = _run_workers(
            specs, num_workers, spec_config, shard_dir, recycle_every=int(args.recycle_every)
        )
    else:
        summaries = _collect_single(
            specs, spec_config, shard_dir, builder, interpolate_fn,
            recycle_every=int(args.recycle_every),
        )

    entries, stored_rows, missing = _spec_entries(summaries, len(specs))
    if missing:
        shown = missing[:8]
        print(
            f"[collect_expert] 警告：{len(missing)} 条 spec 无产出（worker 崩溃）"
            f"：{shown}{' ...' if len(missing) > len(shown) else ''}",
            flush=True,
        )
    if stored_rows == 0:
        print("[collect_expert] 未保留任何样本；检查过滤阈值或专家质量", flush=True)
        return 1

    total_candidates = int(sum(int(entry["candidates"]) for entry in entries))
    filter_counter = _merge_filter_counts(entries)
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
    arrays["balance_weight"] = np.asarray(balance["sample_weight"], dtype=np.float32)  # v2 规范名
    arrays["balance_group"] = np.asarray(balance["balance_group"], dtype="U64")

    # 统计口径与 v2 相同（计数=行数 / 加权=权重和），但直接在最终数组上向量化计算
    train_weight = np.asarray(arrays["train_weight"], dtype=np.float64)
    effective = train_weight * np.asarray(balance["sample_weight"], dtype=np.float64)
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
    trainable_rows = int(gate.sum())
    trainable_weight = float(effective.sum())
    row_yield = (trainable_rows / total_candidates) if total_candidates else 0.0
    weighted_yield = (trainable_weight / total_candidates) if total_candidates else 0.0
    below_min = sorted(
        name for name, count in per_label_count.items() if count < DEFAULT_MIN_SAMPLES_PER_CATEGORY
    )
    report: Dict[str, Any] = {
        "specs": str(args.specs),
        "expert": str(args.expert),
        "kinematics_source": kinematics_source,
        "elapsed_s": round(time.perf_counter() - started, 3),
        # ---- 行数口径（legacy 键保持：retained_steps / bc_retained_step_yield / per_label_positive）----
        "total_candidate_steps": int(total_candidates),
        "stored_rows": int(stored_rows),
        "retained_steps": int(trainable_rows),
        "bc_retained_step_yield": float(row_yield),
        "per_label_positive": per_label_count,
        "counts": {
            "convention": "行数（train_weight>0 的可训练行）",
            "candidate_steps": int(total_candidates),
            "stored_rows": int(stored_rows),
            "trainable_rows": int(trainable_rows),
            "zero_weight_rows": int(stored_rows - trainable_rows),
            "step_yield": float(row_yield),
            "per_label_positive": per_label_count,
        },
        # ---- 加权口径（权重和；有效权重 = train_weight * sample_weight）----
        "weighted": {
            "convention": "权重和（train_weight*sample_weight）",
            "trainable_weight_sum": float(trainable_weight),
            "step_yield_weighted": float(weighted_yield),
            "per_label_positive_weighted": per_label_weighted,
        },
        "filter_counts": dict(filter_counter),
        "min_samples_per_category": DEFAULT_MIN_SAMPLES_PER_CATEGORY,
        "labels_below_min": below_min,
        "balance": balance,
        "per_spec": [entry["report"] for entry in entries],
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
            "roundtrip_lc_lat": float(args.roundtrip_lc_lat),
            "roundtrip_lc_first_max": float(args.roundtrip_lc_first_max),
            "roundtrip_lc_mean": float(args.roundtrip_lc_mean),
            "balance": str(args.balance),
            "balance_ratio": float(args.balance_ratio),
            "traffic_density": args.traffic_density,
        },
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
                "convention": "行数口径（保守）；加权口径见 report.weighted.per_label_positive_weighted",
            },
        },
    }
    # report 内的 balance 需要序列化 np 标量 → 转换
    report["balance"] = {
        "mode": balance["stats"]["mode"],
        "ratio": balance["stats"]["ratio"],
        "group_counts_before": balance["stats"]["group_counts_before"],
        "group_counts_after": balance["stats"]["group_counts_after"],
        "group_trainable_before": balance["stats"]["group_trainable_before"],
        "counts": balance["stats"]["counts"],
        **({"weight_range": balance["stats"]["weight_range"]} if "weight_range" in balance["stats"] else {}),
        **({"cap": balance["stats"]["cap"], "min_group_trainable": balance["stats"]["min_group_trainable"],
            "dropped_indices_count": int(balance["stats"].get("dropped_indices_count", 0))}
           if "cap" in balance["stats"] else {}),
    }
    if missing:  # 仅并行 worker 崩溃时出现；单进程恒为空（报告可与单进程逐键对比）
        report["missing_spec_indices"] = [int(index) for index in missing]
    paths = _write_dataset(
        out_dir,
        arrays,
        label_order=label_order,
        builder=builder,
        config=report["config"],
        report=report,
        num_slots=int(arrays["od"].shape[1]) if "od" in arrays else 16,
    )
    if args.keep_shards:
        print(f"[collect_expert] 保留分片目录 {shard_dir}（--keep-shards）", flush=True)
    else:
        shutil.rmtree(shard_dir, ignore_errors=True)

    print("", flush=True)
    print(f"[collect_expert] retained-step yield = {row_yield:.3f} "
          f"({trainable_rows} trainable / {stored_rows} stored / {total_candidates} candidates)  "
          f"gate>= {DEFAULT_MIN_YIELD}: {'PASS' if row_yield >= DEFAULT_MIN_YIELD else 'FAIL'}  "
          f"(weighted {weighted_yield:.3f})", flush=True)
    print(f"[collect_expert] filter counts (rows): {dict(filter_counter)}", flush=True)
    print("[collect_expert] per-label positives (count / weighted; min "
          f"{DEFAULT_MIN_SAMPLES_PER_CATEGORY}):", flush=True)
    for name in label_order:
        flag = "" if per_label_count[name] >= DEFAULT_MIN_SAMPLES_PER_CATEGORY else "  <-- below min"
        print(f"  {name:>18}: {per_label_count[name]} / {per_label_weighted[name]:.1f}{flag}", flush=True)
    print(f"[collect_expert] balance: {report['balance']}", flush=True)
    print(f"[collect_expert] wrote {paths['npz']} ({stored_rows} rows, {trainable_rows} trainable), "
          f"{paths['meta']}, {paths['report']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
