"""PPO+GAE 训练器、BC 预训练循环与 MoE 路由监控（N4，阶段 A/B/C 共用）。

与已落地组件的对接（P2 契约 §2/§4）
-----------------------------------
- **网络** ``net/model.py::DrivingModel``：``forward(obs) -> dict``（B 维在前）返回
  ``action_mu (B,2)``（**已 sigmoid 压缩到动作界内**）、``action_logstd (B,2)``、``value (B,1)``、
  ``traj_xy (B,6,2)``、``od_pred``、``ld_pred``、``router_logits (B,8)``、``latent``。
  本训练器的动作分布约定 = 网络 ``PolicyHead`` 的约定：
  ``action = low + (high-low)·sigmoid(raw)``，``raw = logit((action-low)/(high-low))``；
  ``action_mode="sigmoid_squashed"``（默认）按该式反解均值后在 raw 空间采样/重算 logprob，
  与 ``PolicyHead.log_prob`` 完全一致（含 Jacobian）。
- **rollout 缓冲** ``pipeline.buffer.RolloutBuffer``（canonical，按帧存 + 在线拼历史 + GAE）。
  训练器按 **env-major** 追加帧，并在每个 env 的帧尾追加一行"bootstrap 行"
  （``value=下一观测的 V``），使 ``buffer.compute_gae`` 在每个 env 的 rollout 末端正确 bootstrap；
  bootstrap 行不参与 minibatch（``valid_mask``）。
- **环境池** ``pipeline/vector_env.py::VectorEnvPool``（spawn 子进程；每进程一个 engine）。
  本模块用 :class:`VectorPoolAdapter` 适配其 record 协议；单进程/阶段 A 精确执行用
  :class:`LocalEnvPool`（精确 ``ExactTracker`` / 阶段 C ``LqrTracker`` / 近似运动学）。
- **动作执行** ``env/tracking.py``：``(ds,dθ)`` → 30 点参考的单一真源。对子进程池，
  训练器把策略动作展开成 5 个 env-step ``[steer, throttle]``（一阶运动学近似，文档化）；
  ``LocalEnvPool(tracker="exact"/"lqr")`` 用真实 tracker（阶段 A/C）。

位姿与历史窗口（RAM 关键，§8.4/§4）
-----------------------------------
rollout 只存单帧当前通道；更新时用 ``buffer.build_history`` 在 SE(2) 下重拼 6 帧。
历史对齐需要各帧位姿：

- 池的 obs 带 ``pose``（``LocalEnvPool`` / smoke stub 提供）时直接使用；
- 缺失时（``VectorEnvPool`` 的 worker record 目前不带位姿）用**动作航位推算**：
  ``pose_{t+1} = arc_advance(pose_t, ds_t, dθ_t)``，起点任意（(0,0,0)）。
  对齐只需帧间相对位姿与航向差，任意常数偏移在 ``se2_align`` 中约掉，因此该推算与真实
  对齐等价（误差仅来自被跟踪动作与参考弧的偏差）；每 episode 起点复位。

奖励
----
``reward_model.aggregation.RewardAggregator``（每 env 一份）：``step_ctx = MetaDrive info +
派生量 a_lon/a_lat/jerk/speed_ratio/speed_limit_mps``（见 :class:`RewardAdapter`）；
``reward_model`` 不可用时回退到文档化 stub（打印告警）。

PPO 诊断（无行为改变，2026-09-25 崩塌复盘后加入）
--------------------------------------------------
每个 ``update()`` 的指标除 PPO 损失/KL/clip/entropy/grad_norm 外还包含：

- ``reward/<term>`` + ``reward/dense|terminating|carl_multiplier|carl_penalty|terminal|total``：
  rollout 内逐步奖励分解（:class:`RewardStatistics`）；
- ``advantage/{mean,std,min,max}``（归一化前）、``returns/mean``、
  ``value/{mean,std,explained_var}``（``1 − Var(ret−V)/Var(ret)``）；
- ``probe/*``：**固定探针批**（``train.probe_batch``，默认 :data:`DEFAULT_PROBE_BATCH`，
  加载一次）上的 ``action_mu[:,0]``/``logstd`` 均值-方差与分速度档 ds 均值；
  任一 speed<2 m/s 档 ds<1.5 m → ``probe/low_speed_alert=1`` + 打印告警（低速吸引子）。

``pipeline.monitoring`` 递归展平嵌套指标 → CSV + tensorboard（``train/...``）。

预算与纪律
----------
- γ=0.99、λ=0.95、clip=0.2、多 epoch × minibatch、value/entropy、梯度裁剪；
- 阶段 C：``kl_anchor_coef``（冻结参考模型）+ ``bc_anchor_coef``（专家动作，权重感知）+ primary lr ×0.1；
  critic warmup（``train.critic_warmup_updates`` / CLI ``--critic-warmup-updates``）：前 N 个
  update 只拟合 value 头（策略/主干冻结，指标 ``critic_warmup=true``），之后恢复常规 PPO；
- Stage B（lane U1，2026-09-27 去聚类）：router **无监督标签**（无簇 CE/acc）；
  MoE 负载均衡 aux = Switch 式 ``α·E·Σ_i f_i·P_i``（α=``load_balance_coef``），KPI =
  ``router/expert_load/*`` + ``router/load_cv`` + ``router/gate_entropy``
  多数类基线）；
  另输出 MoE 负载诊断（``bc_expert_load_*``/``bc_load_cv``/``bc_gate_entropy``）。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
import zipfile
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from pipeline.buffer import RolloutBuffer

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as _exc:  # noqa: BLE001 - 允许纯 NumPy 单测在无 torch 环境导入
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_IMPORT_ERROR: Optional[Exception] = _exc
else:
    _TORCH_IMPORT_ERROR = None

__all__ = [
    "PPOConfig",
    "BCConfig",
    "PPOTrainer",
    "BCDataset",
    "MaterializedBCDataset",
    "episode_prefix_row_count",
    "to_device_tensor",
    "to_device_tensors",
    "DEFAULT_STAGE_BATCH_SIZE",
    "RouterMonitor",
    "RewardAdapter",
    "RewardStatistics",
    "DEFAULT_PROBE_BATCH",
    "STAGE_C_DESIGN_PREFIXES",
    "STAGE_C_TRAINABLE_SCOPES",
    "apply_trainable_allowlist",
    "trainable_param_groups",
    "LocalEnvPool",
    "VectorPoolAdapter",
    "build_pool",
    "build_reward_adapter",
    "load_supervised_labels",
    "pretrain_bc",
    "evaluate_bc",
    "save_checkpoint",
    "load_checkpoint",
    "load_training_checkpoint",
    "load_optimizer_state",
    "capture_rng_state",
    "restore_rng_state",
    "config_snapshot_hash",
    "expand_policy_action",
    "sanitize_masked_od",
    "squeeze_single_slot",
    "SUPERVISED_LABELS",
    "resolve_device",
    "resolve_torch_threads",
    "apply_thread_limits",
    # schema v2 / 权重感知会计
    "V2_HISTORY_KEYS",
    "row_train_weights",
    "row_balance_weights",
    "row_action_weights",
    "weighted_stats",
    "binary_auc",
    "weighted_od_multi_step_loss",
    "presence_entry_loss",
    "worst_flags_for_rows",
    "row_scale_from_worst",
    "traj_valid_for_rows",
    "ego_kpi_arrays",
    "dataset_weight_report",
]

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #

#: 受监督 8 标签固定顺序（与 config/model.yaml::moe.router.supervised_labels 一致）。
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
_MODEL_CONFIG_DEFAULT = "config/model.yaml"
#: 动作界兜底（与 net/policy.py::ACTION_LOW/HIGH 一致）
DEFAULT_ACTION_LOW: Tuple[float, float] = (0.0, -0.6)
DEFAULT_ACTION_HIGH: Tuple[float, float] = (10.0, 0.6)
_POLICY_DT = 0.5
_PHYSICS_HZ = 10
NON_OBS_KEYS = ("pose",)
_HIST_SUFFIX = "_hist"

# --------------------------------------------------------------------------- #
# PPO 诊断（无行为改变；每 update 输出指标 → pipeline.monitoring）
# --------------------------------------------------------------------------- #

#: 固定探针批默认路径（BC 数据集目录/npz）：动作 ds/logstd 漂移 + 低速吸引子检测。
#: 默认为仓库内**实际存在**的数据集目录；配置覆盖：``config/train.yaml::train.probe_batch``
#: （null = 关闭探针；路径缺失时显式告警并关闭）。
DEFAULT_PROBE_BATCH = "datasets/BTC20260926-2343_expert5k"
#: 探针批大小（帧；固定不随 update 变化，保证序列可比）。
DEFAULT_PROBE_SIZE = 256
#: 探针分速度档（m/s）：0–1 / 1–2 / 2–4 / 4–8 / >8。
_PROBE_SPEED_BINS: Tuple[Tuple[float, float], ...] = (
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
    (8.0, float("inf")),
)
#: 低速吸引子阈值：speed < 2 m/s 档的 ds 均值低于该值 → 告警（打印 + 指标字段）。
_PROBE_LOW_SPEED_DS_THRESHOLD = 1.5


def load_supervised_labels(path: str = _MODEL_CONFIG_DEFAULT) -> Tuple[str, ...]:
    """读取 router 监督标签顺序（固定顺序单一出处 = ``config/model.yaml``）。"""
    labels: Optional[Sequence[str]] = None
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        labels = payload.get("moe", {}).get("router", {}).get("supervised_labels")
    except Exception:  # noqa: BLE001
        labels = None
    order = tuple(str(item) for item in labels) if labels else SUPERVISED_LABELS
    if len(order) != 8:
        raise ValueError(f"supervised_labels 必须是 8 个，实际 {order}")
    return order


def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError(f"PyTorch 不可用：{_TORCH_IMPORT_ERROR}")


# --------------------------------------------------------------------------- #
# 设备 / 线程（GPU 化与 worker 线程风暴防护）
# --------------------------------------------------------------------------- #

#: torch 线程上限默认值（``config/train.yaml::train.threads.max_per_process``）
DEFAULT_TORCH_THREAD_CAP = 4
#: OMP/MKL 线程数默认值（``config/train.yaml::train.threads.omp_num_threads``）
DEFAULT_OMP_NUM_THREADS = 1
#: 阶段 A/B 默认 batch（2026-09-26 GPU 利用率改造：256 → 1024；CLI ``--batch-size`` 覆盖）
DEFAULT_STAGE_BATCH_SIZE = 1024


def _thread_config(config: Optional[Mapping[str, Any]] = None) -> Tuple[int, int]:
    """读 ``config['train']['threads']`` → ``(max_per_process, omp_num_threads)``（缺失用默认）。"""
    threads = ((config or {}).get("train", {}) or {}).get("threads", {}) or {}
    cap = int(threads.get("max_per_process") or DEFAULT_TORCH_THREAD_CAP)
    omp = int(threads.get("omp_num_threads") or DEFAULT_OMP_NUM_THREADS)
    return max(1, cap), max(1, omp)


def resolve_torch_threads(workers: int = 1, *, config: Optional[Mapping[str, Any]] = None) -> int:
    """每进程 torch 线程数 = ``min(cap, max(1, cpu_count // workers))``（cap 默认 4，可配置）。"""
    cap, _ = _thread_config(config)
    return max(1, min(cap, (os.cpu_count() or 1) // max(1, int(workers))))


def apply_thread_limits(
    workers: int = 1,
    *,
    config: Optional[Mapping[str, Any]] = None,
    threads: Optional[int] = None,
    omp: Optional[int] = None,
) -> int:
    """进程入口调用：设 OMP/MKL 环境默认值 + 限制 torch 线程，返回设置的线程数。

    实测（2026-09-25）：eval worker 每进程 torch 默认 14 线程 → 20 核机 load average 40，
    50-spec eval >18 min。``os.environ.setdefault`` 在 spawn 子进程前调用会随环境传播；
    ``torch.set_num_threads`` 对当前进程即时生效。CPU 行为不变（只是线程数被钳制）。
    """
    _, omp_default = _thread_config(config)
    omp_value = max(1, int(omp if omp is not None else omp_default))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, str(omp_value))
    count = max(1, int(threads)) if threads is not None else resolve_torch_threads(workers, config=config)
    if torch is not None:
        try:
            torch.set_num_threads(count)
        except Exception:  # noqa: BLE001 - 钳制失败不应影响训练
            pass
    return count


def resolve_device(device: Optional[str] = None, config: Optional[Mapping[str, Any]] = None) -> str:
    """解析运行设备：显式 ``device`` > ``config['train']['device']`` > auto。

    ``auto``（或未配置）= CUDA 可用则 ``cuda``，否则 ``cpu``；显式 ``"cpu"`` 行为与旧版一致。
    """
    configured = ((config or {}).get("train", {}) or {}).get("device")
    for candidate in (device, configured):
        if candidate and str(candidate).lower() != "auto":
            return str(candidate)
    return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


@contextlib.contextmanager
def safe_od_pose():
    """临时修补 ``net.encoders.ObsEncoders.od_pose`` 的 ``atan2(0,0)`` 反向 NaN（N1 缺陷绕行）。

    N1 现状：``od_pose`` 先 ``atan2(sinθ, cosθ)`` 再乘掩码；mask=0 槽位特征全 0，反向传播时
    ``Atan2Backward0`` 在 (0,0) 处未定义 → 任何经过 B1 rollout 轨迹的损失（BC 的 ``traj_xy``）
    直接 NaN。补丁把 (0,0) 的 cos 暂换为 1：**前向值不变**（atan2(0,1)=atan2(0,0)=0，且该槽位
    输出仍被 mask 清零），只消除 NaN 梯度。退出时恢复原实现；N1 修复后本上下文等价于空操作。
    """
    try:
        import net.encoders as _enc
    except Exception:  # noqa: BLE001
        yield
        return
    original = _enc.ObsEncoders.od_pose

    @staticmethod
    def _patched(feat, mask):  # type: ignore[misc]
        cos = feat[..., 4]
        sin = feat[..., 5]
        safe_cos = torch.where((cos == 0.0) & (sin == 0.0), torch.ones_like(cos), cos)
        heading = torch.atan2(sin, safe_cos).unsqueeze(-1)
        return torch.cat([feat[..., 0:2], heading], dim=-1) * mask.unsqueeze(-1)

    _enc.ObsEncoders.od_pose = _patched
    try:
        yield
    finally:
        _enc.ObsEncoders.od_pose = staticmethod(original)


def _with_safe_od_pose(fn):
    """把函数体包进 :func:`safe_od_pose`（见其 docstring 的 N1 缺陷说明）。"""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with safe_od_pose():
            return fn(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------- #
# 策略分布（与 net/policy.py 同约定）
# --------------------------------------------------------------------------- #

def _bounds_from_model(model: "nn.Module", low: Sequence[float], high: Sequence[float]) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """动作界：优先取模型 ``policy`` 上的 buffer（net 契约），否则用配置。"""
    for attr in ("policy",):
        head = getattr(model, attr, None)
        if head is None:
            continue
        head_low = getattr(head, "action_low", None)
        head_high = getattr(head, "action_high", None)
        if head_low is not None and head_high is not None:
            return head_low.detach().clone(), head_high.detach().clone()
    return (
        torch.tensor(list(low), dtype=torch.float32),
        torch.tensor(list(high), dtype=torch.float32),
    )


def unit_from_action(action: "torch.Tensor", low: "torch.Tensor", high: "torch.Tensor") -> "torch.Tensor":
    """动作 → sigmoid 单元变量 ``(a-low)/(high-low)``，裁剪避免 logit 发散。"""
    span = (high - low).clamp(min=1e-6)
    return torch.clamp((action - low) / span, 1e-6, 1.0 - 1e-6)


def raw_mu_from_action(
    mu_squashed: "torch.Tensor", low: "torch.Tensor", high: "torch.Tensor", *, mode: str = "sigmoid_squashed"
) -> "torch.Tensor":
    """有界均值 → raw 空间均值（``sigmoid`` 的反解；``clip`` 模式下 raw=动作）。"""
    if mode == "clip":
        return mu_squashed
    unit = unit_from_action(mu_squashed, low, high)
    return torch.log(unit) - torch.log1p(-unit)  # logit(unit)


def squash_unit(unit: "torch.Tensor", low: "torch.Tensor", high: "torch.Tensor") -> "torch.Tensor":
    return low + span_of(low, high) * unit


def span_of(low: "torch.Tensor", high: "torch.Tensor") -> "torch.Tensor":
    return (high - low).clamp(min=1e-6)


def sample_action(
    mu_squashed: "torch.Tensor",
    logstd: "torch.Tensor",
    low: "torch.Tensor",
    high: "torch.Tensor",
    *,
    mode: str = "sigmoid_squashed",
    generator: Optional["torch.Generator"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """采样动作，返回 ``(action, logprob)``。

    - ``sigmoid_squashed``（默认，匹配 net ``PolicyHead``）：由有界均值反解 raw 均值后加噪、
      再 sigmoid 压缩；logprob 用 raw 样本 + Jacobian 修正（与 ``PolicyHead.log_prob`` 同式）。
    - ``clip``：直接对 ``mu + σ·ε`` 截断（CleanRL 风格；logprob 用未截断样本）。
    """
    if mode not in ("sigmoid_squashed", "clip"):
        raise ValueError(f"未知 action_mode={mode!r}（可选 sigmoid_squashed/clip）")
    if mode == "sigmoid_squashed":
        raw_mu = raw_mu_from_action(mu_squashed, low, high, mode=mode)
        noise = torch.randn(raw_mu.shape, generator=generator, dtype=raw_mu.dtype, device=raw_mu.device)
        raw = raw_mu + torch.exp(logstd) * noise
        action = squash_unit(torch.sigmoid(raw), low, high)
        return action, logprob_raw(raw, raw_mu, logstd, low, high, mode=mode)
    noise = torch.randn(mu_squashed.shape, generator=generator, dtype=mu_squashed.dtype, device=mu_squashed.device)
    raw = mu_squashed + torch.exp(logstd) * noise
    return torch.clamp(raw, min=low, max=high), logprob_raw(raw, mu_squashed, logstd, low, high, mode=mode)


def logprob_raw(
    raw: "torch.Tensor",
    raw_mu: "torch.Tensor",
    logstd: "torch.Tensor",
    low: "torch.Tensor",
    high: "torch.Tensor",
    *,
    mode: str = "sigmoid_squashed",
) -> "torch.Tensor":
    """raw 空间高斯 logprob + 压缩变换的 Jacobian（``Σ_dim``，返回 ``(B,)``）。

    ``sigmoid`` 压缩：``d(a)/d(raw) = span·σ(raw)·(1-σ(raw))``；
    ``clip`` 模式无变换 Jacobian（恒等 1）。
    """
    std = torch.exp(logstd)
    base = -0.5 * (((raw - raw_mu) / std) ** 2 + 2.0 * logstd + math.log(2.0 * math.pi))
    if mode == "clip":
        return base.sum(dim=-1)
    unit = torch.sigmoid(raw)
    jacobian = torch.log(span_of(low, high) * unit * (1.0 - unit) + 1e-6)
    return (base - jacobian).sum(dim=-1)


def logprob_from_action(
    mu_squashed: "torch.Tensor",
    logstd: "torch.Tensor",
    action: "torch.Tensor",
    low: "torch.Tensor",
    high: "torch.Tensor",
    *,
    mode: str = "sigmoid_squashed",
) -> "torch.Tensor":
    """在 PPO 更新时对**已执行动作**重算 logprob（与采样路径同式，ratio 才可比）。"""
    if mode == "clip":
        return logprob_raw(action, mu_squashed, logstd, low, high, mode=mode)
    raw_mu = raw_mu_from_action(mu_squashed, low, high, mode=mode)
    raw_action = raw_mu_from_action(action, low, high, mode=mode)
    return logprob_raw(raw_action, raw_mu, logstd, low, high, mode=mode)


def gaussian_entropy(logstd: "torch.Tensor") -> "torch.Tensor":
    """raw 空间对角高斯微分熵 ``Σ(logσ + 0.5·log(2πe))``（``(B,)``）。"""
    return (logstd + 0.5 * math.log(2.0 * math.pi * math.e)).sum(dim=-1)


def gaussian_kl(
    mu1: "torch.Tensor", logstd1: "torch.Tensor", mu2: "torch.Tensor", logstd2: "torch.Tensor"
) -> "torch.Tensor":
    """``KL(N1‖N2)``（对角高斯闭式，raw 参数；``(B,)``）。"""
    var1 = torch.exp(2.0 * logstd1)
    var2 = torch.exp(2.0 * logstd2)
    return (logstd2 - logstd1 + (var1 + (mu1 - mu2) ** 2) / (2.0 * var2) - 0.5).sum(dim=-1)


def build_optimizer(model: "nn.Module", lr: float, primary_lr_scale: float = 1.0) -> "torch.optim.Optimizer":
    """参数分组：名字含 ``primary`` 的参数用 ``lr × primary_lr_scale``（阶段 C 契约 ×0.1）。"""
    primary: List["torch.Tensor"] = []
    other: List["torch.Tensor"] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (primary if "primary" in name.lower() else other).append(parameter)
    groups: List[Dict[str, Any]] = [{"params": other, "lr": float(lr)}]
    if primary:
        groups.append({"params": primary, "lr": float(lr) * float(primary_lr_scale)})
    return torch.optim.Adam(groups, lr=float(lr))


#: phase 3（lane P3-B）specific 组参数名前缀：experts/gate(router)/residual_scale
#: （与 ``_SPECIFIC_PHASE_FREEZE`` 的可训练集合同一口径）；其余 = base 主干
#: （encoders/mem_encoder/plan_head 其余/WM(st_gnn)/policy/value）。
_PHASE3_SPECIFIC_PREFIXES: Tuple[str, ...] = (
    "plan_head.moe.experts.",
    "plan_head.moe.router.",
    "plan_head.moe.residual_scale",
)


def build_phase3_optimizer(
    model: "nn.Module",
    lr: float,
    *,
    base_scale: float = 0.25,
    specific_scale: float = 0.5,
    freeze_mode: str = "all",
) -> "torch.optim.Optimizer":
    """phase 3（lane P3-B/P3-I）LR 分组优化器：base 主干 × ``base_scale`` / specific × ``specific_scale``。

    组序固定 ``[base?, specific]``（与 :data:`_PHASE3_SPECIFIC_PREFIXES` 同一划分口径）：

    - **base**：encoders / mem_encoder / plan_head（不含 MoE specific）/ WM(st_gnn) / policy / value；
    - **specific**：``plan_head.moe.experts.*`` / ``plan_head.moe.router.*`` / ``residual_scale``。

    ``freeze_mode="all"``（旧行为）：要求两组齐全（模型无 MoE / 未解冻 → ``ValueError``，拒绝静默退化）；
    ``freeze_mode="specific_only"``（lane P3-I）：base 组为空是**预期**（主干已冻结），只返回 specific 组；
    此时 specific 组为空仍 ``ValueError``（experts/router/residual_scale 必须有可训参数）。
    """
    base: List["torch.Tensor"] = []
    specific: List["torch.Tensor"] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (specific if name.startswith(_PHASE3_SPECIFIC_PREFIXES) else base).append(parameter)
    if not specific or (not base and freeze_mode != "specific_only"):
        raise ValueError(
            f"phase 3 LR 分组为空：base={len(base)} · specific={len(specific)}"
            "（all 模式要求全参数解冻且两组齐全；specific_only 模式要求模型含 MoE "
            "experts/router/residual_scale）"
        )
    groups: List[Dict[str, Any]] = []
    if base:
        groups.append({"name": "base", "params": base, "lr": float(lr) * float(base_scale)})
    groups.append({"name": "specific", "params": specific, "lr": float(lr) * float(specific_scale)})
    return torch.optim.Adam(groups, lr=float(lr))


def _value_parameter_names(model: "nn.Module") -> Tuple[str, ...]:
    """critic warmup 的可训练参数名：价值头（真实模型 ``value.*``；smoke stub ``head_value.*``）。

    warmup 只解冻这些参数，其余（policy 头 + 共享主干/encoders/MoE/WM）全部冻结——
    因此 value 梯度不会流回共享主干，warmup 期间策略输出逐位不变（见
    :meth:`PPOTrainer.update`）。
    """
    named = list(model.named_parameters())
    names = tuple(name for name, _ in named if name.lower().startswith("value."))
    if not names:
        names = tuple(name for name, _ in named if "value" in name.lower())
    return names


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

@dataclass
class PPOConfig:
    """PPO/GAE 超参（默认即契约口径：clip=0.2、γ=0.99、λ=0.95）。"""

    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    action_mode: str = "sigmoid_squashed"  # sigmoid_squashed（net 约定）| clip
    action_low: Tuple[float, float] = DEFAULT_ACTION_LOW
    action_high: Tuple[float, float] = DEFAULT_ACTION_HIGH
    kl_anchor_coef: float = 0.0
    bc_anchor_coef: float = 0.0
    primary_lr_scale: float = 1.0
    target_kl: Optional[float] = None
    #: 前 N 个 update 只拟合 critic（value 头），策略/共享主干冻结（0 = 关闭，行为同旧版）。
    critic_warmup_updates: int = 0
    seed: int = 0
    device: str = "auto"  # auto = CUDA 可用则 cuda，否则 cpu（显式 "cpu" 行为不变）
    #: 收集侧跟踪器参考口径（P0-1 A-hold）：
    #: ``repeat_action``（默认）= 6 步参考全部 = 采样动作 ``a_t``（执行只依赖 PPO 记账的随机变量）；
    #: 该臂 collect 走 cheap path（rollout=False，V9）；``plan`` = 旧行为（首步 ``a_t`` + 后 5 步
    #: WM/plan head 规划预览，需 rollout 前向），仅供对照，不得用于 RL 结论。
    plan_reference: str = "repeat_action"

    def __post_init__(self) -> None:
        if self.plan_reference not in ("repeat_action", "plan"):
            raise ValueError(f"未知 plan_reference={self.plan_reference!r}（可选 repeat_action/plan）")


@dataclass
class BCConfig:
    """BC 预训练超参（轨迹 L1/L2 + 可选动作回归 + router BCE，带样本权重）。

    阶段 B（v1.1）扩展：
    - ``wm_detach``：B1 rollout 内 WM 输出 detach（WM 冻结 + 因果链有效）；
    - ``freeze_prefixes``：按参数名前缀冻结模块（primary→specific 分阶段训练用）；
    - ``phase``：仅写入指标，便于区分 primary/specific 两段。

    阶段 B（v2）扩展：
    - ``slice_*``：动作误差分切片阈值（急刹/急转；弯道用 ``on_curve`` 标签）；
    - ``history_stride``：BC harvest 帧间距（env step，默认 5 = 0.5 s）。

    留出集 / 逐 epoch 回调（2026-09-26）：
    - ``train_indices``：训练行索引子集（``None`` = 全量；Stage B 按 episode 留出后传
      ``train_idx``，保证 val 集完全不参与梯度）；
    - ``val_indices``：留出行索引；非空时每 epoch 末在留出集上做确定性前向
      （:func:`evaluate_bc`），val 指标经 ``epoch_callback`` 的第二个 dict 传出；
    - ``epoch_callback(epoch_index, train_metrics, val_metrics)``：0 基 epoch、
      epoch 内增量指标（**不含**全阶段累计汇总），供编排层逐 epoch 落盘
      （step 轴由调用方负责；见 ``pipeline.stages.run_stage_b``）。
    """

    epochs: int = 10
    batch_size: int = 256
    #: micro-batch（梯度累积）：>0 且 < batch_size 时按 micro 前向/反向、宏 batch 一次更新
    #: （损失按宏口径精确缩放，Σ_m s_m·L_m = L_macro；None = 不分片）。
    micro_batch_size: Optional[int] = None
    lr: float = 3e-4
    loss_type: str = "l2"  # l1 | l2
    traj_weight: float = 1.0
    action_weight: float = 0.5
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"  # auto = CUDA 可用则 cuda，否则 cpu（显式 "cpu" 行为不变）
    shuffle: bool = True
    max_batches: Optional[int] = None
    wm_detach: bool = False
    freeze_prefixes: Tuple[str, ...] = ()
    phase: str = "bc"
    #: 动作误差切片阈值：专家即时 ``ds``（m/0.5 s）低于该值算急刹。
    slice_brake_ds: float = 1.0
    #: 动作误差切片阈值：专家即时 ``|dθ|``（rad/0.5 s）高于该值算急转。
    slice_turn_dtheta: float = 0.10
    #: BC harvest 帧间距（env step）；v2 数据集可在 meta 里覆盖。
    history_stride: int = 5
    #: 训练行索引子集（None = ``dataset.sample_indices()`` 全量）。
    train_indices: Optional[np.ndarray] = None
    #: 留出行索引（非空时每 epoch 末跑确定性评估，见 :func:`evaluate_bc`）。
    val_indices: Optional[np.ndarray] = None
    #: 每 epoch 回调 ``(epoch_index, train_metrics, val_metrics)``（0 基；指标为 epoch 增量）。
    epoch_callback: Optional[Callable[[int, Mapping[str, Any], Mapping[str, Any]], None]] = None
    #: resume：起始 epoch（0 基；已完成 epoch 数）→ 循环从 ``start_epoch`` 继续。
    start_epoch: int = 0
    #: resume：优化器 state_dict（同相位恢复；参数组不匹配时忽略并用全新优化器）。
    optimizer_state: Optional[Mapping[str, Any]] = None
    #: resume：RNG 状态（见 :func:`capture_rng_state`）；缺省时按 seed 重放 shuffle。
    rng_state: Optional[Mapping[str, Any]] = None
    #: 周期 ckpt 回调 ``(epoch_index, model, optimizer, val_metrics, rng)``（0 基本地 epoch）。
    checkpoint_callback: Optional[Callable[..., None]] = None

    # ---- lane U1：MoE 负载均衡 + 权重化 specific 训练 ----
    #: MoE 开关（phase 1 = False：专家不参与、输出严格 = primary；phase 2 = True）。
    moe_enabled: bool = True
    #: Switch 式负载均衡 aux 权重 α（``α·E·Σ_i f_i·P_i``；0 = 关闭）。
    load_balance_coef: float = 0.0
    #: 数据集行空间的 worst-50% 标记（0/1；-1=未知=不缩放）。来自权重 sidecar
    #: （``pipeline.hard_mining``），行数必须 == ``dataset.count``。
    worst_flags: Optional[np.ndarray] = None
    #: worst 行权重倍率（默认 1.0）；其余行 = ``mild_weight``（默认 0.1）；
    #: 两者与 ``train_weight×balance_weight`` 相乘（有效质量 0.5·1.0 + 0.5·0.1 = 0.55）。
    hard_weight: float = 1.0
    mild_weight: float = 0.1
    #: 留出集的 worst 标记（None = 留出集不缩放；legacy episode 切分时与 ``worst_flags`` 同源）。
    val_worst_flags: Optional[np.ndarray] = None
    #: traj-aux 逐行掩码（lane U4；0/1，数据集行空间；None = 全 1 = 旧行为）。
    #: DAgger 行的 ``traj6`` 是"常量动作外推"合成值 → 掩 0（不吃 traj 损失）；
    #: **只作用于 traj 损失项**，action/load 项与全部监控统计仍为全量口径。
    traj_aux_valid: Optional[np.ndarray] = None
    #: 独立留出数据集（``--val-dir``；None = 用主数据集 + ``val_indices`` 的旧口径）。
    val_dataset: Optional[Any] = None


@dataclass
class Phase3Config:
    """stage B phase 3（迭代恢复训练；lane P3-B / P3-I）超参。

    与 :class:`BCConfig` 的差异（用户定案）：

    - **数据**：单一 dagger 目录（无 worst/mild 权重 / 无 mining）；lane P3-I 可选 5k 锚行
      由 stages 层合并（``anchor``），本类只消费合并后的行权重；
    - **冻结配方**（``freeze_mode``，lane P3-I）：``all`` = 全参数解冻（旧行为，含 WM）/
      ``specific_only`` = 只训 experts/router/residual_scale（冻结前缀由 stages 传入）；
    - **损失组合**（上游三层监督 + 监控）：首步动作 + 多步动作链 + plan head ``ego_next`` +
      WM OD/LD + presence/entry BCE + MoE 负载均衡；``traj`` **只做监控**（不进损失，
      ``traj_aux_weight`` 默认 0 —— dagger 的 ``traj6`` 是常量外推合成值）。
      ``freeze_mode="specific_only"`` 时上游监督项（:data:`PHASE3_WM_LOSS_KEYS`）无梯度 →
      **自动降级为 0**（权重置 0 + 跳过 WM 教师强制前向，见 :func:`phase3_effective_config`）。
    """

    epochs: int = 5
    batch_size: int = 1024
    micro_batch_size: Optional[int] = 512
    lr: float = 3e-4
    loss_type: str = "l2"
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    shuffle: bool = True
    max_batches: Optional[int] = None
    phase: str = "phase3"
    #: 冻结模式（lane P3-I）：``all`` = 全参数解冻（旧行为）/ ``specific_only`` = 只训
    #: experts/router/residual_scale（``freeze_prefixes`` 由 stages 层给 phase-2 冻结清单，
    #: 且 :data:`PHASE3_WM_LOSS_KEYS` 自动降级为 0）。
    freeze_mode: str = "all"
    #: 冻结参数前缀（``apply_freeze_prefixes``；``all`` 模式为空 = 全参数可训）。
    freeze_prefixes: Tuple[str, ...] = ()
    #: 训练行索引子集（None = 全部行）。
    train_indices: Optional[np.ndarray] = None
    #: 留出行索引（非空时每 epoch 末跑确定性评估，见 :func:`evaluate_bc_phase3`）。
    val_indices: Optional[np.ndarray] = None
    #: 留出数据集（None = 与训练同源）。
    val_dataset: Optional[Any] = None
    epoch_callback: Optional[Callable[[int, Mapping[str, Any], Mapping[str, Any]], None]] = None
    start_epoch: int = 0
    optimizer_state: Optional[Mapping[str, Any]] = None
    rng_state: Optional[Mapping[str, Any]] = None
    checkpoint_callback: Optional[Callable[..., None]] = None

    # ---- 损失权重（config stages.B.phase3.losses.*）----
    action_weight: float = 1.0
    #: 多步动作链（rollout plan 第 2..6 步 vs 未来专家首步动作）。
    action_chain_weight: float = 0.2
    #: plan head ``ego_next``（未来 ego 特征；``ego_fut`` + wm_valid 尾部 mask）。
    ego_next_weight: float = 0.1
    #: WM OD 直接多步（Huber + angle；与 stage A 同构）。
    od_weight: float = 1.0
    #: WM LD 恢复监督（smooth_l1 + angle，4 维预测空间）。
    ld_weight: float = 1.0
    #: presence/entry BCE（id 轴目标；stage A 同系数口径）。
    presence_weight: float = 0.1
    entry_weight: float = 0.1
    #: 轨迹辅助（监控口径保留；0 = 不进损失，见类 docstring）。
    traj_aux_weight: float = 0.0
    load_balance_coef: float = 0.01
    moe_enabled: bool = True

    # ---- LR 分组（config stages.B.phase3.lr_scale.*）----
    lr_base_scale: float = 0.25
    lr_specific_scale: float = 0.5

    def __post_init__(self) -> None:
        if self.freeze_mode not in ("all", "specific_only"):
            raise ValueError(f"freeze_mode 非法：{self.freeze_mode!r}（all | specific_only）")


def apply_freeze_prefixes(model: "nn.Module", prefixes: Sequence[str]) -> Tuple[str, ...]:
    """按参数名前缀设置 ``requires_grad``（不匹配的前缀列表 = 全部可训练）。

    返回实际被冻结的参数名（便于日志核对）。空 ``prefixes`` 表示全部解冻。
    """
    frozen: List[str] = []
    normalized = tuple(str(prefix) for prefix in prefixes)
    for name, parameter in model.named_parameters():
        trainable = not any(name.startswith(prefix) for prefix in normalized)
        parameter.requires_grad_(trainable)
        if not trainable:
            frozen.append(name)
    return tuple(frozen)


#: 阶段 C 可训练范围（R2/P0-6）：``design``（默认，docs/db44fefe-system-review.md P0-6/阶段C）
#: 的 allowlist = policy/value 头 + MoE specific（experts/residual_scale）；
#: 其余（encoders/mem_encoder/plan_head 主干（fusion/norm/ego_next/primary）/st_gnn）全部冻结。
#: 2026-09-30（V8r，G1 §2 依据 docs/db44fefe-system-review.md:127,240"shared/primary/router/WM
#: 冻结"）：router（``plan_head.moe.router.*``）移出 allowlist——阶段 C 无 router BCE/校准信号，
#: 行为 loss 反传 gate 会把标签语义塑造成策略附庸；router 只观测不训练。
STAGE_C_DESIGN_PREFIXES: Tuple[str, ...] = (
    "policy.",
    "value.",
    "plan_head.moe.experts.",
    "plan_head.moe.residual_scale",
)
#: ``--trainable-scope`` 取值：``design``（默认，P0-6 设计冻结）/ ``all``（旧行为：仅 st_gnn 冻结）。
STAGE_C_TRAINABLE_SCOPES: Tuple[str, ...] = ("all", "design")


def apply_trainable_allowlist(model: "nn.Module", prefixes: Sequence[str]) -> Tuple[str, ...]:
    """只保留 ``prefixes`` 前缀参数可训练，其余 ``requires_grad_(False)``；返回被冻结参数名。

    与 :func:`apply_freeze_prefixes` 互补（后者按前缀冻结，本函数按前缀**保留**），
    供阶段 C ``trainable_scope=design`` 的 allowlist 冻结使用。
    """
    frozen: List[str] = []
    normalized = tuple(str(prefix) for prefix in prefixes)
    for name, parameter in model.named_parameters():
        trainable = any(name.startswith(prefix) for prefix in normalized)
        parameter.requires_grad_(trainable)
        if not trainable:
            frozen.append(name)
    return tuple(frozen)


#: 可训练参数组表的前缀划分（最长匹配优先；未匹配 = 参数名首段）。
_TRAINABLE_GROUP_PREFIXES: Tuple[str, ...] = (
    "plan_head.moe.experts.",
    "plan_head.moe.router.",
    "plan_head.moe.residual_scale",
    "plan_head.moe.primary.",
    "plan_head.",
    "encoders.",
    "mem_encoder.",
    "st_gnn.",
    "policy.",
    "value.",
)


def trainable_param_groups(
    model: "nn.Module", lr: float, primary_lr_scale: float = 1.0
) -> Tuple[Dict[str, Any], ...]:
    """可训练参数组表 ``(prefix, params, lr)``：与 :func:`build_optimizer` 的 LR 分组同口径。

    ``primary`` 参数取 ``lr × primary_lr_scale``（阶段 C 契约 ×0.1），其余取 ``lr``；
    只统计 ``requires_grad=True`` 的参数（启动打印 + metrics 用）。
    """
    totals: Dict[Tuple[str, float], int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_lr = float(lr) * (float(primary_lr_scale) if "primary" in name.lower() else 1.0)
        prefix = next(
            (candidate for candidate in _TRAINABLE_GROUP_PREFIXES if name.startswith(candidate)),
            name.split(".")[0] + ".",
        )
        key = (prefix, group_lr)
        totals[key] = totals.get(key, 0) + int(parameter.numel())
    return tuple(
        {"prefix": prefix, "params": count, "lr": group_lr} for (prefix, group_lr), count in totals.items()
    )


# --------------------------------------------------------------------------- #
# 奖励：RewardAggregator 适配 + 文档化 stub 兜底
# --------------------------------------------------------------------------- #

_CRASH_KEYS = ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_sidewalk", "crash_human")


def make_stub_reward() -> Callable[..., float]:
    """**文档化 stub 奖励**（仅 ``reward_model`` 不可用时兜底，不用于正式训练）。

    ``r = 2.0·Δroute_completion - 5.0·crash - 5.0·out_of_road + 10.0·arrive - 0.05·|acceleration|``
    """
    state: Dict[int, float] = {}

    def reward(env_index: int, info: Dict[str, Any], done: bool = False) -> float:
        info = info if isinstance(info, dict) else {}
        progress = float(info.get("route_completion", 0.0) or 0.0)
        delta = progress - state.get(env_index, progress)
        state[env_index] = 0.0 if done else progress
        crash = any(bool(info.get(key, False)) for key in _CRASH_KEYS)
        value = 2.0 * delta
        value -= 5.0 if crash else 0.0
        value -= 5.0 if bool(info.get("out_of_road", False)) else 0.0
        value += 10.0 if bool(info.get("arrive_dest", False)) else 0.0
        value -= 0.05 * abs(float(info.get("acceleration", 0.0) or 0.0))
        return float(value)

    return reward


class RewardAdapter:
    """把 ``reward_model.RewardAggregator`` 适配成 trainer 的逐步奖励接口。

    - 每个 env 一份聚合器（内部持有 ``route_completion_prev`` 势能塑形状态）；
    - ``step_ctx = MetaDrive info + 派生量``（缺省补齐，不覆盖 info 已有键）：

      * ``a_lon``：优先按策略步速度差（与 eval 的 a_lon 口径一致），否则 obs ego 的 0.1s 加速度；
      * ``a_lat``：obs ego 通道 index 2（``v·yaw_rate``，m/s²）；
      * ``jerk``：相邻策略步 ``a_lon`` 差分 / ``dt``；
      * ``speed_limit_mps``：obs LD 通道 **slot 0**（ego 车道 5 m 采样点）第 4 维；
      * ``speed_ratio``：``velocity / speed_limit_mps``。
    - 终止步先算奖励再 ``reset()``（终局 outcome 不能丢）。
    """

    def __init__(
        self,
        factory: Optional[Callable[[], Any]] = None,
        fallback: Optional[Callable[..., float]] = None,
        *,
        dt: float = _POLICY_DT,
    ):
        self.factory = factory
        self.fallback = fallback
        self.dt = float(dt)
        self._aggregators: Dict[int, Any] = {}
        self._prev_a_lon: Dict[int, float] = {}
        self._prev_speed: Dict[int, float] = {}
        self.early_terminations = 0

    def reset_all(self) -> None:
        self._aggregators.clear()
        self._prev_a_lon.clear()
        self._prev_speed.clear()
        self.early_terminations = 0

    def _build_ctx(self, env_index: int, info: Dict[str, Any], obs: Mapping[str, np.ndarray]) -> Dict[str, Any]:
        ctx: Dict[str, Any] = dict(info) if isinstance(info, dict) else {}
        ego = obs.get("ego") if isinstance(obs, Mapping) else None
        ego_flat = np.asarray(ego, dtype=np.float32).reshape(-1) if ego is not None else None
        speed = ctx.get("velocity", ctx.get("speed"))
        if "a_lon" not in ctx:
            if speed is not None and env_index in self._prev_speed:
                try:
                    ctx["a_lon"] = (float(speed) - self._prev_speed[env_index]) / self.dt
                except (TypeError, ValueError):
                    pass
            elif ego_flat is not None and ego_flat.shape[0] >= 2:
                ctx["a_lon"] = float(ego_flat[1])
        if ego_flat is not None and ego_flat.shape[0] >= 3 and "a_lat" not in ctx:
            ctx["a_lat"] = float(ego_flat[2])
        if speed is not None:
            try:
                self._prev_speed[env_index] = float(speed)
            except (TypeError, ValueError):
                pass
        ld = obs.get("ld") if isinstance(obs, Mapping) else None
        if ld is not None and "speed_limit_mps" not in ctx:
            ld_array = np.asarray(ld, dtype=np.float32)
            mask = obs.get("ld_mask") if isinstance(obs, Mapping) else None
            mask_array = np.asarray(mask, dtype=np.float32).reshape(-1) if mask is not None else None
            if ld_array.ndim == 2 and ld_array.shape[0] >= 1 and ld_array.shape[1] >= 5:
                if mask_array is None or mask_array.shape[0] == 0 or mask_array[0] > 0.5:
                    limit = float(ld_array[0, 4])
                    if limit > 0.0:
                        ctx["speed_limit_mps"] = limit
        return ctx

    def step(
        self,
        env_index: int,
        info: Dict[str, Any],
        obs: Mapping[str, np.ndarray],
        done: bool,
        pool_reward: float,
        step_index: Optional[int] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """返回 ``(reward, meta)``；stub 路径 meta 为空。"""
        if self.factory is None:
            if self.fallback is None:
                return float(pool_reward), {}
            return float(self.fallback(env_index, info, done)), {}
        ctx = self._build_ctx(env_index, info, obs)
        ctx["done"] = bool(done)
        a_lon = ctx.get("a_lon")
        if a_lon is not None:
            previous = self._prev_a_lon.get(env_index)
            if "jerk" not in ctx and previous is not None:
                ctx["jerk"] = (float(a_lon) - previous) / self.dt
            self._prev_a_lon[env_index] = float(a_lon)
        speed = ctx.get("velocity", ctx.get("speed"))
        limit = ctx.get("speed_limit_mps")
        if "speed_ratio" not in ctx and speed is not None and limit:
            try:
                ctx["speed_ratio"] = float(speed) / float(limit)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        aggregator = self._aggregators.get(env_index)
        if aggregator is None:
            aggregator = self.factory()
            self._aggregators[env_index] = aggregator
        result = aggregator.step(ctx, step_index=step_index)
        meta = {
            "reason": result.reason,
            "terminal_key": result.terminal_key,
            "components": dict(result.components),
            "terminal_value": float(result.terminal_value),
            # 诊断用派生量（RewardStatistics 消费；不影响奖励数值/训练行为）
            "dense_sum": float(result.dense_sum),
            "terminating_sum": float(result.terminating_sum),
            "carl_multiplier": float(result.carl_multiplier),
            "carl_penalty": float(result.carl_penalty),
            "shaping_decay": float(result.shaping_decay),
        }
        if bool(result.done) and not done:
            self.early_terminations += 1
            meta["early_termination"] = True
        if done:
            aggregator.reset()
            self._prev_a_lon.pop(env_index, None)
            self._prev_speed.pop(env_index, None)
        return float(result.reward), meta


def build_reward_adapter(
    config: Optional[Dict[str, Any]] = None,
    *,
    dt: float = _POLICY_DT,
    logger: Callable[[str], None] = print,
) -> Tuple[RewardAdapter, str]:
    """构造奖励适配器：优先 N2 的 ``RewardAggregator``，不可用时回退文档化 stub。"""
    factory: Optional[Callable[[], Any]] = None
    source = "trainer.make_stub_reward"
    try:
        from reward_model import (  # type: ignore
            DEFAULT_TERM_CONFIGS,
            AggregationConfig,
            RewardAggregator,
            build_terms,
        )

        cfg = dict(config or {})
        term_configs = list(cfg.get("terms") or DEFAULT_TERM_CONFIGS)
        aggregation = dict(cfg.get("aggregation") or {})

        def factory() -> Any:  # noqa: F811 - 每 env 一份聚合器
            return RewardAggregator(build_terms(term_configs), AggregationConfig(**aggregation))

        source = "reward_model.aggregation.RewardAggregator"
    except Exception as exc:  # noqa: BLE001
        logger(f"[reward] reward_model 不可用（{type(exc).__name__}: {exc}）→ 使用文档化 stub 奖励")
    fallback = None if factory is not None else make_stub_reward()
    return RewardAdapter(factory=factory, fallback=fallback, dt=dt), source


class RewardStatistics:
    """一次 rollout 内逐步奖励分解的窗口均值（PPO 诊断指标，不影响训练）。

    ``RewardAdapter.step`` 的 meta（``components`` + dense/terminating/carl/terminal）+ 实际
    逐步奖励 → ``{"reward": {term: 均值, ..., "dense": 均值, "terminal": 均值, ..., "steps": N}}``。
    """

    _PARTS = (
        ("dense", "dense_sum"),
        ("terminating", "terminating_sum"),
        ("carl_multiplier", "carl_multiplier"),
        ("carl_penalty", "carl_penalty"),
        ("terminal", "terminal_value"),
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._steps = 0
        self._terms: Dict[str, float] = {}
        self._parts: Dict[str, float] = {}

    def update(self, meta: Mapping[str, Any], reward: float) -> None:
        """累计一步：逐项加权贡献 + 派生部分（stub 路径 meta 为空时只记 total）。"""
        self._steps += 1
        components = meta.get("components") if isinstance(meta, _MappingABC) else None
        if isinstance(components, _MappingABC):
            for name, value in components.items():
                try:
                    self._terms[str(name)] = self._terms.get(str(name), 0.0) + float(value)
                except (TypeError, ValueError):
                    continue
        for out_key, meta_key in self._PARTS:
            if not isinstance(meta, _MappingABC) or meta_key not in meta:
                continue  # stub 路径 meta 为空 → 只记 total，不虚构 0 值部分
            try:
                self._parts[out_key] = self._parts.get(out_key, 0.0) + float(meta[meta_key])
            except (TypeError, ValueError):
                continue
        self._parts["total"] = self._parts.get("total", 0.0) + float(reward)

    def summary(self) -> Dict[str, Dict[str, float]]:
        if self._steps <= 0:
            return {}
        steps = float(self._steps)
        payload = {name: value / steps for name, value in self._terms.items()}
        payload.update({name: value / steps for name, value in self._parts.items()})
        payload["steps"] = steps
        return {"reward": payload}


def _probe_speed_bin_key(low: float, high: float) -> str:
    """速度档标签：``0-1`` / ``1-2`` / ``2-4`` / ``4-8`` / ``8-inf``（标签用 ``_`` 分隔）。"""
    left = f"{low:g}"
    right = "inf" if math.isinf(high) else f"{high:g}"
    return f"{left}_{right}"


def _advantage_stats(
    advantages: np.ndarray, returns: np.ndarray, values: np.ndarray
) -> Dict[str, Any]:
    """原始（未归一化）优势/回报/价值统计 + 价值解释方差 ``1 - Var(ret-V)/Var(ret)``。"""
    if advantages.size <= 0:
        return {}
    returns_variance = float(np.var(returns))
    explained = (
        1.0 - float(np.var(returns - values)) / returns_variance
        if returns_variance > 1e-12
        else 0.0
    )
    return {
        "advantage": {
            "mean": float(advantages.mean()),
            "std": float(advantages.std()),
            "min": float(advantages.min()),
            "max": float(advantages.max()),
        },
        "returns": {"mean": float(returns.mean()), "std": float(returns.std())},
        "value": {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "explained_var": float(explained),
        },
    }


# --------------------------------------------------------------------------- #
# 历史窗口 / BC 数据集（schema 见 tools/collect_expert.py）
# --------------------------------------------------------------------------- #

def _default_alignments() -> Dict[str, Any]:
    """从 ``env.obs`` 通道类读取历史对齐规则（od/ld 需要 SE(2) 重表达）。"""
    out: Dict[str, Any] = {}
    try:
        from env.obs.ld import LDChannel
        from env.obs.od import ODChannel

        out["od"] = ODChannel.alignment
        out["ld"] = LDChannel.alignment
    except Exception:  # noqa: BLE001 - env.obs 不可用时退化为 identity
        pass
    return out


def _alignment_from_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """meta JSON 的对齐规则 → ``FrameAlignment``（不可用时退回默认表）。"""
    alignments = _default_alignments()
    stored = (meta or {}).get("channel_alignments") or {}
    if not stored:
        return alignments
    try:
        from env.obs.base import FrameAlignment

        for name, spec in stored.items():
            alignments[name] = FrameAlignment(
                point_pairs=tuple(tuple(pair) for pair in spec.get("point_pairs", ())),
                vector_pairs=tuple(tuple(pair) for pair in spec.get("vector_pairs", ())),
                angle_dims=tuple(int(i) for i in spec.get("angle_dims", ())),
            )
    except Exception:  # noqa: BLE001
        pass
    return alignments


def _se2_align(
    features: np.ndarray, alignment: Any, delta_theta: float, delta_xy: np.ndarray, current_theta: float
) -> np.ndarray:
    try:
        from env.obs.base import se2_align

        return se2_align(
            features,
            alignment=alignment,
            delta_theta=float(delta_theta),
            delta_xy=np.asarray(delta_xy, dtype=np.float32),
            current_theta=float(current_theta),
        )
    except Exception:  # noqa: BLE001
        return np.asarray(features, dtype=np.float32)


def stack_history(
    entries: Sequence[Dict[str, Any]],
    alignments: Dict[str, Any],
    *,
    current_pose: np.ndarray,
    valid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """6 个单帧 ``entries``（旧→新）→ ``*_hist``/``*_hist_mask``（对齐到 ``current_pose``）。

    ``valid`` 为 ``(6,)``：0 的槽位掩码清零（warmup 补位帧不得当真实历史，§8.4）。
    """
    frames = list(entries)
    if not frames:
        return {}
    if len(frames) < 6:
        frames = [frames[0]] * (6 - len(frames)) + frames
    frames = frames[-6:]
    out: Dict[str, np.ndarray] = {}
    current_pose = np.asarray(current_pose, dtype=np.float32).reshape(-1)
    for name, alignment in alignments.items():
        sample = None
        for entry in frames:
            candidate = entry.get("obs", {}).get(name)
            if candidate is not None:
                sample = np.asarray(candidate, dtype=np.float32)
                break
        if sample is None or sample.ndim != 2:
            continue
        num_slots, dim = int(sample.shape[0]), int(sample.shape[1])
        hist = np.zeros((6, num_slots, dim), dtype=np.float32)
        hist_mask = np.zeros((6, num_slots), dtype=np.float32)
        for slot, entry in enumerate(frames):
            feats = entry.get("obs", {}).get(name)
            mask = entry.get("obs", {}).get(f"{name}_mask")
            if feats is None or np.asarray(feats).shape != (num_slots, dim):
                continue
            pose = np.asarray(entry.get("pose", np.zeros(3)), dtype=np.float32).reshape(-1)
            aligned = _se2_align(
                feats,
                alignment,
                delta_theta=float(current_pose[2] - pose[2]),
                delta_xy=current_pose[:2] - pose[:2],
                current_theta=float(current_pose[2]),
            )
            hist[slot] = aligned
            if mask is not None:
                hist_mask[slot] = np.asarray(mask, dtype=np.float32).reshape(num_slots)
        if valid is not None:
            valid_array = np.asarray(valid, dtype=np.float32).reshape(-1)
            if valid_array.shape[0] == 6:
                hist_mask *= valid_array[:, None]
        out[f"{name}{_HIST_SUFFIX}"] = hist
        out[f"{name}{_HIST_SUFFIX}_mask"] = hist_mask
    return out


# --------------------------------------------------------------------------- #
# Schema v2：权重感知会计（count vs weighted 两种口径）
# --------------------------------------------------------------------------- #
#
# 数据集 v2（lane A）在 v1 基础上新增（全部可选；缺失时退化为 v1 语义）：
#
# - ``train_weight (N,)``：0 = 该帧不作为动作/轨迹目标（仍然可以进 WM 目标查表）；
# - ``balance_weight (N,)``：难度/几何配平权重（与 train_weight 相乘）；
# - ``wm_valid (N,6)``：WM 目标帧可用性（(episode, step+5k) 查表 + 帧可用）；
# - ``od_id (N,16) / od_presence (N,16)``：OD 槽位对象 id / 占据真值（presence/entry BCE）；
# - （lane U1）router 无监督标签；MoE 负载 aux 由 ``net.moe`` 在线计算；
#   软分布 ``router_soft_targets (N,8)`` 是历史数据集键，**训练不再读取**（lane B B3）；
# - mem 历史数组 ``ego_hist/others_hist/od_hist/ld_hist``（+ ``_mask``、``hist_valid``）：
#   采集时按 env memory 精确入库，训练侧直接复用（禁止按行位置重拼，见
#   ``pipeline.stages.build_history``）。
#
# 本节的"权重感知"约定（两套口径都记录，避免混淆）：
# - **count 口径**：样本数 / 槽位数的算术统计（诊断用，和旧日志可比）；
# - **weighted 口径**：``Σ w·x / Σ w``，训练损失与聚合指标的主口径。

V2_HISTORY_KEYS: Tuple[str, ...] = ("ego_hist", "others_hist", "od_hist", "ld_hist")
V2_TRAIN_WEIGHT = "train_weight"
V2_BALANCE_WEIGHT = "balance_weight"
V2_WM_VALID = "wm_valid"


def row_train_weights(dataset: Any, indices: np.ndarray) -> np.ndarray:
    """每行**训练权重**（动作/轨迹目标可用性）：``train_weight`` → 1。

    v2 采集侧保证 ``train_weight=0`` 的行不参与损失（配平权重在 ``sample_weight``，
    见 :func:`row_balance_weights`，两者相乘 = :func:`row_action_weights`）。
    """
    arrays = getattr(dataset, "arrays", dataset)
    index = np.asarray(indices, dtype=np.int64)
    if V2_TRAIN_WEIGHT in arrays:
        return np.asarray(arrays[V2_TRAIN_WEIGHT], dtype=np.float64)[index]
    return np.ones(index.shape[0], dtype=np.float64)


def row_balance_weights(dataset: Any, indices: np.ndarray) -> np.ndarray:
    """每行**配平权重**：``balance_weight`` → ``sample_weight`` → 1。

    v1 数据集只有 ``sample_weight``（难度/几何配平）；v2 采集侧继续用 ``sample_weight``
    记录配平（``train_weight=0`` 的行置 1，由 train_weight 负责排除）。
    """
    arrays = getattr(dataset, "arrays", dataset)
    index = np.asarray(indices, dtype=np.int64)
    for key in (V2_BALANCE_WEIGHT, "sample_weight"):
        if key in arrays:
            return np.asarray(arrays[key], dtype=np.float64)[index]
    return np.ones(index.shape[0], dtype=np.float64)


def row_action_weights(dataset: Any, indices: np.ndarray) -> np.ndarray:
    """动作/轨迹目标的**有效权重** = ``train_weight × 配平权重``（各缺失时取 1）。

    配平权重优先 ``balance_weight``，否则 ``sample_weight``（v1/v2 采集口径一致）；
    Stage A/B 的损失、``_action_mu_stats``、PPO BC 锚全部用本函数取权重
    （权重感知会计的唯一入口）。
    """
    return row_train_weights(dataset, indices) * row_balance_weights(dataset, indices)


def weighted_stats(values: Any, weights: Any = None, *, prefix: str = "") -> Dict[str, float]:
    """标量的**计数/加权双口径**统计。

    返回 ``{prefix}count``（样本数）、``{prefix}weight``（权重和）、``{prefix}mean``（计数均值）、
    ``{prefix}weighted_mean``、``{prefix}median``、``{prefix}p95``（后三者为权重分位数）。
    空集/全零权重 → 统计量为 ``nan``（预算 count/weight 仍保留，便于诊断）。
    """
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if weights is None:
        w = np.ones_like(v)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != v.shape:
            raise ValueError(f"weights 形状 {w.shape} 与 values {v.shape} 不一致")
    finite = np.isfinite(v) & np.isfinite(w) & (w >= 0.0)
    v, w = v[finite], w[finite]
    out: Dict[str, float] = {
        f"{prefix}count": float(v.size),
        f"{prefix}weight": float(w.sum()),
    }
    if v.size == 0 or float(w.sum()) <= 0.0:
        for key in ("mean", "weighted_mean", "median", "p95"):
            out[f"{prefix}{key}"] = float("nan")
        return out
    out[f"{prefix}mean"] = float(v.mean())
    out[f"{prefix}weighted_mean"] = float(np.dot(v, w) / w.sum())
    order = np.argsort(v, kind="mergesort")
    v_sorted, w_sorted = v[order], w[order]
    cumulative = np.cumsum(w_sorted)

    def _weighted_quantile(q: float) -> float:
        target = float(q) * float(cumulative[-1])
        idx = int(np.searchsorted(cumulative, target, side="left"))
        return float(v_sorted[min(idx, v_sorted.size - 1)])

    out[f"{prefix}median"] = _weighted_quantile(0.5)
    out[f"{prefix}p95"] = _weighted_quantile(0.95)
    return out


def binary_auc(scores: Any, labels: Any) -> float:
    """rank-based AUC（Mann-Whitney U，含并列秩平均）；单类/空 → ``nan``。"""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1) > 0.5
    finite = np.isfinite(s) & np.isfinite(np.asarray(labels, dtype=np.float64).reshape(-1))
    s, y = s[finite], y[finite]
    positives = int(y.sum())
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # 并列秩平均（否则 AUC 有偏）
    sorted_scores = s[order]
    start = 0
    for end in range(1, s.size + 1):
        if end == s.size or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                ranks[order[start:end]] = ranks[order[start:end]].mean()
            start = end
    return float((ranks[y].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def weighted_od_multi_step_loss(
    od_pred: "torch.Tensor",
    od_target: "torch.Tensor",
    od_mask: "torch.Tensor",
    *,
    frame_weight: Optional["torch.Tensor"] = None,
    valid: Optional["torch.Tensor"] = None,
    beta: float = 1.0,
) -> Tuple["torch.Tensor", List[Dict[str, float]]]:
    """v2 加权直接多步 **OD** 损失（阶段 A；LD 用同构的 ``weighted_ld_multi_step_loss``，lane P3-F）。

    与 ``net.world_model.direct_multi_step_loss`` 的差别是**显式加权**：

    - 权重 ``w_b``（``train_weight × balance_weight``）与 ``valid[b,k]``（``wm_valid × 查表存在``）
      逐帧生效；分子 ``Σ w_b·valid·mask·err``，分母 ``Σ w_b·valid·mask``；
    - ``w_b = 0`` 的帧既不贡献损失也不贡献分母（不再用"有效项总数"做隐性重加权——
      旧口径下权重大的帧的有效槽位越多，其对总损失的贡献被摊薄）；
    - 返回 ``(total, per_horizon)``：``per_horizon[k]`` 含 ``loss/weight/count/frames``
      四个标量（tensor/float），供逐 horizon 日志。

    Args:
        od_pred/od_target: ``(B,K,16,5)``；``od_mask``: ``(B,K,16)``。
        frame_weight: ``(B,)`` 权重，默认全 1。
        valid: ``(B,K)`` 未来步有效性，默认全 1。
    """
    import torch
    import torch.nn.functional as F

    if valid is None:
        valid = torch.ones(od_mask.shape[:2], dtype=od_mask.dtype, device=od_mask.device)
    if frame_weight is None:
        frame_weight = torch.ones(od_mask.shape[0], dtype=od_mask.dtype, device=od_mask.device)
    frame_weight = frame_weight.to(dtype=od_mask.dtype, device=od_mask.device).reshape(-1)
    valid = valid.to(dtype=od_mask.dtype, device=od_mask.device)

    smooth = F.smooth_l1_loss(od_pred[..., :4], od_target[..., :4], beta=beta, reduction="none").sum(dim=-1) / 4.0
    angle = 1.0 - torch.cos(od_pred[..., 4] - od_target[..., 4])
    err = smooth + angle  # (B,K,16)

    weight = od_mask * valid.unsqueeze(-1) * frame_weight.reshape(-1, 1, 1)  # (B,K,16)
    numerator = (err * weight).sum()
    denominator = weight.sum().clamp(min=1e-8)
    total = numerator / denominator

    per_horizon: List[Dict[str, float]] = []
    for k in range(int(od_mask.shape[1])):
        wk = weight[:, k, :]
        num_k = (err[:, k, :] * wk).sum()
        den_k = wk.sum().clamp(min=1e-8)
        frame_has = (wk > 0).any(dim=1)
        per_horizon.append(
            {
                "loss": num_k / den_k,
                "weight": float(den_k.detach()),
                "count": float(wk.gt(0).sum().detach()),
                "frames": float(frame_has.sum().detach()),
            }
        )
    return total, per_horizon


def weighted_ld_multi_step_loss(
    ld_pred: "torch.Tensor",
    ld_target: "torch.Tensor",
    ld_mask: "torch.Tensor",
    *,
    frame_weight: Optional["torch.Tensor"] = None,
    valid: Optional["torch.Tensor"] = None,
    beta: float = 1.0,
) -> Tuple["torch.Tensor", List[Dict[str, float]]]:
    """v2 加权直接多步 **LD** 损失（lane P3-B：LD 恢复监督）。

    与 :func:`weighted_od_multi_step_loss` 同构，维度按 LD 预测空间 4 维
    ``[dx, dy, heading, curvature]``（``net.st_gnn.ld_state_from_features``）：

    - ``smooth_l1`` 作用于非角度维 ``[0, 1, 3]``（位置 + 曲率）；
    - 角度维 ``2`` 用 ``1 - cos(Δheading)``；
    - 权重 = ``ld_mask × valid × frame_weight``（分子/分母同口径）；``ld_mask`` 由
      ``pipeline.frames.build_future`` 产出（目标帧 ld mask × wm_valid，**已在 t0 自车系**）。

    Args:
        ld_pred/ld_target: ``(B,K,16,4)``；``ld_mask``: ``(B,K,16)``。
        frame_weight: ``(B,)`` 权重，默认全 1。
        valid: ``(B,K)`` 未来步有效性，默认全 1。

    返回 ``(total, per_horizon)``：``per_horizon[k]`` 含 ``loss/weight/count/frames``。
    """
    import torch
    import torch.nn.functional as F

    if valid is None:
        valid = torch.ones(ld_mask.shape[:2], dtype=ld_mask.dtype, device=ld_mask.device)
    if frame_weight is None:
        frame_weight = torch.ones(ld_mask.shape[0], dtype=ld_mask.dtype, device=ld_mask.device)
    frame_weight = frame_weight.to(dtype=ld_mask.dtype, device=ld_mask.device).reshape(-1)
    valid = valid.to(dtype=ld_mask.dtype, device=ld_mask.device)

    # 非角度维 0/1/3（位置 + 曲率）走 smooth_l1；角度维 2 走 1-cos
    smooth = (
        F.smooth_l1_loss(
            ld_pred[..., [0, 1, 3]], ld_target[..., [0, 1, 3]], beta=beta, reduction="none"
        ).sum(dim=-1)
        / 3.0
    )
    angle = 1.0 - torch.cos(ld_pred[..., 2] - ld_target[..., 2])
    err = smooth + angle  # (B,K,16)

    weight = ld_mask * valid.unsqueeze(-1) * frame_weight.reshape(-1, 1, 1)  # (B,K,16)
    numerator = (err * weight).sum()
    denominator = weight.sum().clamp(min=1e-8)
    total = numerator / denominator

    per_horizon: List[Dict[str, float]] = []
    for k in range(int(ld_mask.shape[1])):
        wk = weight[:, k, :]
        num_k = (err[:, k, :] * wk).sum()
        den_k = wk.sum().clamp(min=1e-8)
        frame_has = (wk > 0).any(dim=1)
        per_horizon.append(
            {
                "loss": num_k / den_k,
                "weight": float(den_k.detach()),
                "count": float(wk.gt(0).sum().detach()),
                "frames": float(frame_has.sum().detach()),
            }
        )
    return total, per_horizon


def weighted_action_chain_loss(
    plan: "torch.Tensor",
    action_target: "torch.Tensor",
    chain_valid: "torch.Tensor",
    *,
    frame_weight: Optional["torch.Tensor"] = None,
    loss_type: str = "l2",
) -> Tuple["torch.Tensor", List[Dict[str, float]]]:
    """多步动作链监督（lane P3-B）：rollout ``plan`` 第 2..K 步 vs 未来专家首步动作。

    - ``plan/action_target``: ``(B,K,2)``（``plan[:, k]`` = 第 k+1 个计划动作，k=0 已由
      ``action_mu`` 首步损失监督 → 本项只监督 ``k=1..K-1``，即未来的 t+1..t+K-1 策略步）；
    - ``chain_valid``: ``(B,K)`` 逐帧 mask（同 episode 未来帧缺失 → 0；``[:,0]`` 不参与）；
    - ``frame_weight``: ``(B,)``（``train_weight×balance``），缺省全 1；
    - 损失 = ``Σ w·valid·err / Σ w·valid``（``loss_type``：l2→MSE / l1→MAE）。

    返回 ``(total, per_step)``：``per_step[k]``（k=1..K-1）含 ``loss/weight/frames``。
    """
    import torch

    if chain_valid.ndim != 2:
        raise ValueError(f"chain_valid 形状应为 (B,K)，收到 {tuple(chain_valid.shape)}")
    batch, steps, _ = plan.shape
    if tuple(action_target.shape) != tuple(plan.shape):
        raise ValueError(f"action_target 形状 {tuple(action_target.shape)} != plan {tuple(plan.shape)}")
    if tuple(chain_valid.shape) != (batch, steps):
        raise ValueError(f"chain_valid 形状 {tuple(chain_valid.shape)} != (B,K)={(batch, steps)}")
    device = plan.device
    if frame_weight is None:
        frame_weight_t = torch.ones(batch, dtype=plan.dtype, device=device)
    else:
        frame_weight_t = frame_weight.to(dtype=plan.dtype, device=device).reshape(-1)
    valid = chain_valid.to(dtype=plan.dtype, device=device)

    diff = plan[:, 1:] - action_target[:, 1:]  # (B,K-1,2)，k=0 不参与
    per_step = (diff ** 2).mean(dim=-1) if loss_type == "l2" else diff.abs().mean(dim=-1)  # (B,K-1)
    weight = frame_weight_t.reshape(-1, 1) * valid[:, 1:]  # (B,K-1)
    denominator = weight.sum().clamp(min=1e-8)
    total = (per_step * weight).sum() / denominator

    per_horizon: List[Dict[str, float]] = []
    for k in range(steps - 1):
        wk = weight[:, k]
        den_k = wk.sum().clamp(min=1e-8)
        per_horizon.append(
            {
                "loss": (per_step[:, k] * wk).sum() / den_k,
                "weight": float(den_k.detach()),
                "frames": float(wk.gt(0).sum().detach()),
            }
        )
    return total, per_horizon


def presence_entry_loss(
    pred_presence: "torch.Tensor",
    pred_entry: "torch.Tensor",
    presence_target: "torch.Tensor",
    entry_target: "torch.Tensor",
    *,
    frame_weight: Optional["torch.Tensor"] = None,
    collect: Optional[Dict[str, List[np.ndarray]]] = None,
) -> Dict[str, Any]:
    """presence/entry BCE + AUC（每槽位二分类；支持逐帧或逐 (帧,horizon) 权重）。

    Args:
        pred_presence/pred_entry: ``(B,S)`` 或 ``(B,K,S)`` logits。
        presence_target/entry_target: 同形 0/1 目标。
        frame_weight: ``(B,)`` 或 ``(B,K)``（v2：``train_weight × wm_valid``）；0 的样本
            既不计损失也不计分母与 AUC。
        collect: 梯度累积用；传入 dict 时把逐 micro 的 scores/labels 追加进池
            （键 ``presence_scores/presence_labels/entry_scores/entry_labels``），
            本函数返回的 AUC 为 nan，由调用方在宏 batch 上一次性计算。
    """
    import torch
    import torch.nn.functional as F

    num_slots = int(pred_presence.shape[-1])
    logits_p = pred_presence.reshape(-1, num_slots)
    logits_e = pred_entry.reshape(-1, num_slots)
    target_p = presence_target.reshape(-1, num_slots)
    target_e = entry_target.reshape(-1, num_slots)
    if frame_weight is None:
        weight = torch.ones((logits_p.shape[0], ), dtype=logits_p.dtype, device=logits_p.device)
    else:
        weight = frame_weight.to(dtype=logits_p.dtype, device=logits_p.device).reshape(-1)
        if weight.numel() != logits_p.shape[0]:
            num_frames = int(weight.numel())
            repeats = logits_p.shape[0] // max(num_frames, 1)
            weight = weight.repeat_interleave(max(repeats, 1))
        weight = weight.reshape(-1)

    def _bce(logits: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
        element = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        denominator = weight.sum().clamp(min=1e-8) * num_slots
        return (element * weight.unsqueeze(-1)).sum() / denominator

    presence_loss = _bce(logits_p, target_p)
    entry_loss = _bce(logits_e, target_e)
    mask = (weight > 0.0).cpu().numpy()
    presence_scores = logits_p.detach().float().cpu().numpy()[mask]
    presence_labels = target_p.detach().float().cpu().numpy()[mask]
    entry_scores = logits_e.detach().float().cpu().numpy()[mask]
    entry_labels = target_e.detach().float().cpu().numpy()[mask]
    if collect is not None:
        # 梯度累积：逐 micro 汇入 scores/labels，宏 batch 上一次性算 AUC（pos_rate 用池化值）
        collect.setdefault("presence_scores", []).append(presence_scores)
        collect.setdefault("presence_labels", []).append(presence_labels)
        collect.setdefault("entry_scores", []).append(entry_scores)
        collect.setdefault("entry_labels", []).append(entry_labels)
        presence_auc = float("nan")
        entry_auc = float("nan")
    else:
        presence_auc = binary_auc(presence_scores, presence_labels)
        entry_auc = binary_auc(entry_scores, entry_labels)
    return {
        "presence": presence_loss,
        "entry": entry_loss,
        "presence_auc": float(presence_auc),
        "entry_auc": float(entry_auc),
        "presence_pos_rate": float(presence_labels.mean()) if presence_labels.size else float("nan"),
        "entry_pos_rate": float(entry_labels.mean()) if entry_labels.size else float("nan"),
    }


# --------------------------------------------------------------------------- #
# lane U1：权重化 specific 训练（worst/mild 行权重）+ MoE 负载诊断
# --------------------------------------------------------------------------- #

def worst_flags_for_rows(
    worst_flags: Optional[np.ndarray], indices: np.ndarray
) -> Optional[np.ndarray]:
    """取一个 batch 的 worst-50% 标记（``(B,)`` float32；``-1`` = 未知/不缩放）。

    ``worst_flags`` 是**数据集行空间**的 0/1（-1=未知）数组（权重 sidecar 载入，见
    :func:`pipeline.hard_mining.load_weight_sidecar`）；``None`` → ``None``（不启用加权）。
    """
    if worst_flags is None:
        return None
    flags = np.asarray(worst_flags, dtype=np.float32).reshape(-1)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size and (idx.min() < 0 or idx.max() >= flags.size):
        raise ValueError(
            f"worst_flags 行数 {flags.size} 覆盖不了 batch 索引 [{int(idx.min())}, {int(idx.max())}]"
        )
    return flags[idx]


def traj_valid_for_rows(
    traj_valid: Optional[np.ndarray], indices: np.ndarray
) -> Optional[np.ndarray]:
    """取一个 batch 的 traj-aux 逐行掩码（``(B,)`` float32；``None`` → ``None``（全 1））。

    ``traj_valid`` 是**数据集行空间**的 0/1 数组（lane U4：DAgger 行 = 0、主集行 = 1，
    由 ``pipeline.stages._merge_dagger_rows`` 生成）；``None`` = 不启用掩码（旧行为）。
    行数覆盖不了 batch 索引 → ``ValueError``（与 :func:`worst_flags_for_rows` 同模式）。
    """
    if traj_valid is None:
        return None
    flags = np.asarray(traj_valid, dtype=np.float32).reshape(-1)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size and (idx.min() < 0 or idx.max() >= flags.size):
        raise ValueError(
            f"traj_aux_valid 行数 {flags.size} 覆盖不了 batch 索引 [{int(idx.min())}, {int(idx.max())}]"
        )
    return flags[idx]


def row_scale_from_worst(
    worst: Optional[np.ndarray], *, hard_weight: float = 1.0, mild_weight: float = 0.1
) -> np.ndarray:
    """worst-50% 标记 → 逐行权重倍率（worst=``hard_weight``；其余=``mild_weight``；未知=1.0）。"""
    if worst is None:
        return np.ones(0, dtype=np.float64)
    flags = np.asarray(worst, dtype=np.float64).reshape(-1)
    scale = np.where(flags > 0.5, float(hard_weight), float(mild_weight))
    return np.where(flags < 0.0, 1.0, scale)


def _load_summary(
    *,
    count: int,
    expert_load_sum: Optional[np.ndarray],
    load_cv_num: float,
    load_cv_den: float,
    entropy_num: float,
    entropy_den: float,
    num_experts: int,
) -> Dict[str, Any]:
    """MoE 负载汇总（训练累计 / 逐 epoch 增量 / val 评估共用同一公式）。

    ``expert_load_sum`` = 逐 expert 负载 × 行数（→ 行数加权均值，Σ=1）；
    ``load_cv`` / ``gate_entropy`` = 行数加权均值。``count<=0``（MoE 关闭/无样本）→ 只返回占位。
    """
    out: Dict[str, Any] = {"bc_moe_placeholder": 0.0 if int(count) > 0 else 1.0}
    if int(count) <= 0 or expert_load_sum is None:
        return out
    load = np.asarray(expert_load_sum, dtype=np.float64) / max(float(count), 1e-12)
    out["bc_load_count"] = int(count)
    for index in range(int(num_experts)):
        out[f"bc_expert_load_{index}"] = float(load[index]) if index < load.size else float("nan")
    out["bc_load_cv"] = float(load_cv_num) / max(float(load_cv_den), 1e-12)
    out["bc_gate_entropy"] = float(entropy_num) / max(float(entropy_den), 1e-12)
    return out


def ego_kpi_arrays(
    out: Mapping[str, Any],
    targets: Mapping[str, "torch.Tensor"],
) -> Dict[str, "torch.Tensor"]:
    """B2 统一 ego KPI 的逐样本数组：``action_err`` (B) / ``traj_mae_point`` (B,6) / ``traj_fde_end`` (B)。

    定义（Stage A/B 同口径，均由 policy/plan 头 + WM rollout 产出）：
    动作 = 首步 ``(ds,dθ)`` 的 L1 均值；轨迹 MAE = 6 点逐点 L1 均值；末点 FDE = 第 6 点 L2。
    加权由调用方按 ``train_weight×配平`` 合成（``Σw·x/Σw``）。
    """
    import torch

    diff = out["traj_xy"] - targets["traj6"]
    action_diff = out["action_mu"] - targets["action"][:, 0, :]
    return {
        "action_err": action_diff.abs().mean(dim=-1),
        "traj_mae_point": diff.abs().mean(dim=-1),
        "traj_fde_end": torch.linalg.norm(diff[:, -1, :], dim=-1),
    }


def dataset_weight_report(dataset: Any, *, prefix: str = "dataset") -> Dict[str, float]:
    """数据集统计的**计数/加权双口径**报告（配平前后权重量级、train_weight=0 占比）。

    输出（全部无偏标量，可直接进 JSON/CSV）：
    ``<prefix>/rows``、``<prefix>/rows_weighted``（= Σw）、``<prefix>/train_weight_zero``、
    ``<prefix>/weight_min|max|mean|p95``、``<prefix>/balance_weight_*``（若存在）。
    """
    arrays = getattr(dataset, "arrays", dataset)
    count = int(len(arrays.get("episode_id", arrays.get("train_weight", []))))
    report: Dict[str, float] = {}
    if count <= 0:
        return {f"{prefix}/rows": 0.0}
    indices = np.arange(count, dtype=np.int64)
    train_w = row_train_weights(dataset, indices)
    balance_w = row_balance_weights(dataset, indices)
    report[f"{prefix}/rows"] = float(count)
    report[f"{prefix}/rows_weighted"] = float((train_w * balance_w).sum())
    report[f"{prefix}/train_weight_zero"] = float((train_w <= 0.0).mean())
    for name, values in (("weight", train_w), ("balance_weight", balance_w)):
        if name == "balance_weight" and np.allclose(balance_w, 1.0):
            continue
        order = np.argsort(values, kind="mergesort")
        sorted_w = values[order]
        cumulative = np.cumsum(sorted_w)

        def _q(q: float) -> float:
            index = int(np.searchsorted(cumulative, q * float(cumulative[-1]), side="left"))
            return float(sorted_w[min(index, sorted_w.size - 1)])

        report[f"{prefix}/{name}_min"] = float(values.min())
        report[f"{prefix}/{name}_max"] = float(values.max())
        report[f"{prefix}/{name}_mean"] = float(values.mean())
        report[f"{prefix}/{name}_p95"] = _q(0.95)
    return report


# --------------------------------------------------------------------------- #
# ``--limit-dataset``：读取阶段按 **完整 episode 前缀** 截断（2026-09-27）
# --------------------------------------------------------------------------- #
#
# 背景（2026-09-27 冒烟事故）：``--limit-dataset`` 旧实现先 ``BCDataset.load`` 全量解压
# （5k 数据集实测峰值 RSS ≈ 2.0 GB）再切片，且训练侧默认 macro/micro batch（1024/256）
# 的 host 前向图 ≈ 10 GB → 冒烟与全量训练同样挤爆机器。这里在**解压前**只读前缀行
# （npz 成员是 C 连续 .npy → 逐成员流式解压前 N 行，不碰余下字节），并保持行内列值不变。

#: 前缀读取的分块字节数（DEFLATE 流式解压；避免一次性超大 read）。
_NPZ_PREFIX_READ_CHUNK = 8 << 20


def episode_prefix_row_count(episode_ids: np.ndarray, limit: int) -> int:
    """``--limit-dataset`` 的截断口径：覆盖前 ``limit`` 行的**完整 episode 前缀**行数。

    - 数据按 episode 连续追加（``tools/collect_expert.py`` 的顺序）→ 前缀 = 若干完整 episode；
    - 取累计行数 **≤ limit** 的最长完整 episode 前缀（避免半截 episode 破坏历史/未来查表）；
    - 首个 episode 自身就长于 ``limit`` 时保留该 episode（保证冒烟子集非空）；
    - ``limit`` ≥ 总行数时返回总行数。
    """
    ids = np.asarray(episode_ids)
    count = int(ids.shape[0])
    if count == 0:
        return 0
    limit = max(int(limit), 1)
    if limit >= count:
        return count
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1  # episode 结束行号（升序，不含末集）
    if boundaries.size == 0:
        return count  # 单个 episode 且比 limit 长 → 保留完整 episode
    first = int(boundaries[0])
    if first > limit:
        return first
    index = int(np.searchsorted(boundaries, limit, side="right")) - 1
    return int(boundaries[index])


def _read_exact(stream: Any, size: int) -> Optional[bytes]:
    """从（zip）流精确读 ``size`` 字节；提前 EOF → ``None``（``read`` 可能短读）。"""
    if size <= 0:
        return b""
    chunks: List[bytes] = []
    remaining = int(size)
    while remaining > 0:
        chunk = stream.read(min(remaining, _NPZ_PREFIX_READ_CHUNK))
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _npz_episode_prefix(
    path: Path, limit: int
) -> Optional[Tuple[Dict[str, np.ndarray], int, int, int]]:
    """只解压 ``path``（npz）各成员的**前 M 行**（M = :func:`episode_prefix_row_count`）。

    ``npz`` 成员是标准 C 连续 ``.npy``：前 M 行 = 前 ``M × row_bytes`` 字节，zipfile 流式
    解压只覆盖该前缀。返回 ``(arrays, M, total_rows, total_episodes)``；结构不适用
    （非 zip / Fortran order / 行数不一致 / 读截断）→ ``None``（调用方回退全量加载 + 同口径
    内存截断）。
    """
    if not zipfile.is_zipfile(path):
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if any(name.endswith(".npy") and "/" in name for name in names):
                return None  # 成员在子目录（key 含 "/"）→ 结构不符，回退全量加载
            members = {
                name[: -len(".npy")]: name for name in names if name.endswith(".npy")
            }
            episode_name = members.get("episode_id")
            if episode_name is None:
                return None
            with archive.open(episode_name) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
                if fortran_order or len(shape) != 1 or dtype.hasobject or dtype.kind not in ("i", "u"):
                    return None
                total = int(shape[0])
                buffer = _read_exact(stream, total * dtype.itemsize)
                if buffer is None:
                    return None
                episode_ids = np.frombuffer(buffer, dtype=dtype, count=total)
            total_episodes = int(np.unique(episode_ids).size)
            rows = episode_prefix_row_count(episode_ids, int(limit))
            if rows <= 0:
                return None
            arrays: Dict[str, np.ndarray] = {"episode_id": np.array(episode_ids[:rows], copy=True)}
            for key, name in members.items():
                if key == "episode_id":
                    continue
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    shape, fortran_order, dtype = np.lib.format._read_array_header(stream, version)
                    if fortran_order or not shape or dtype.hasobject or int(shape[0]) < rows:
                        return None
                    row_items = 1
                    for dim in shape[1:]:
                        row_items *= int(dim)
                    buffer = _read_exact(stream, rows * row_items * dtype.itemsize)
                    if buffer is None:
                        return None
                    arrays[key] = np.array(
                        np.frombuffer(buffer, dtype=dtype, count=rows * row_items).reshape(
                            (rows,) + tuple(int(dim) for dim in shape[1:])
                        ),
                        copy=True,  # 与 np.load 语义一致：连续可写
                    )
            return arrays, rows, total, total_episodes
    except (OSError, ValueError, EOFError, KeyError, TypeError, IndexError, zipfile.BadZipFile):
        return None


def _log_dataset_limit(
    path: Path, limit: int, rows: int, total: int, kept_episode_ids: np.ndarray, total_episodes: int
) -> None:
    """打印 ``limit=N → rows=M``（含 episode 数；物化/val 切分都基于该子集）。"""
    kept = np.asarray(kept_episode_ids)
    kept_episodes = int(np.unique(kept).size) if rows > 0 else 0
    print(
        f"[dataset] limit={int(limit)} → rows={int(rows)}/{int(total)}"
        f"（episodes={kept_episodes}/{int(total_episodes)}，按完整 episode 前缀截断；{path}）",
        flush=True,
    )


class BCDataset:
    """按帧存 + 在线拼 6 帧历史的 BC/回放数据集（schema = ``tools/collect_expert.py``）。

    Schema v2（lane A）兼容（全部字段可选，缺失时退化为 v1 行为）：

    - ``train_weight/balance_weight``：动作/轨迹目标权重（见 :func:`row_action_weights`）；
    - ``wm_valid (N,6)``：WM 目标帧可用性；
    - ``od_id/od_presence (N,16)``：OD 槽位对象 id / 占据真值；
    - （lane U1）去聚类：无 ``router_cluster`` 读取；MoE 负载 aux 在线计算；
    - **mem 历史数组**（``ego_hist/others_hist/od_hist/ld_hist`` + masks + ``hist_valid``）：
      采集时按 env memory 精确入库。存在时训练侧**直接复用**（``build_obs_batch``），
      否则按 ``(episode_id, step−5j)`` **精确查表**重拼（禁止按行位置取窗口）。
    """

    def __init__(self, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]):
        self.arrays = arrays
        self.meta = meta
        self.count = int(len(arrays["episode_id"]))
        self.label_names = tuple(meta.get("label_names") or SUPERVISED_LABELS)
        self.alignments = _alignment_from_meta(meta)
        self._obs_keys = [key for key in ("ego", "od", "ld", "nav", "signal", "others") if key in arrays]
        self.history_stride = max(
            1,
            int(
                meta.get("history_stride")
                or meta.get("steps_per_policy")
                or BCConfig.history_stride
            ),
        )
        self._episode_ranges: Dict[int, Tuple[int, int]] = {}
        for index in range(self.count):
            episode = int(arrays["episode_id"][index])
            start, stop = self._episode_ranges.get(episode, (index, index))
            self._episode_ranges[episode] = (min(start, index), max(stop, index))
        # 精确查表 (episode_id, step) → 行号；无 step 的旧 schema 退化为按行位置。
        self._step_lookup: Dict[Tuple[int, int], int] = {}
        if "step" in arrays:
            self._step_lookup = {
                (int(arrays["episode_id"][i]), int(arrays["step"][i])): i for i in range(self.count)
            }
        # v2 mem 历史：od/ld 必需，ego/others 可选（TODO(lane)：net v2 接入 others_hist 语义）。
        self._mem_history_keys = tuple(
            key
            for key in V2_HISTORY_KEYS
            if key in arrays and f"{key}_mask" in arrays
        )
        self.has_mem_history = bool("od_hist" in arrays and "hist_valid" in arrays)
        self.schema_version = int(
            meta.get("schema_version") or (2 if (V2_TRAIN_WEIGHT in arrays or V2_WM_VALID in arrays) else 1)
        )

    @classmethod
    def load(cls, path: str, *, limit: Optional[int] = None) -> "BCDataset":
        """``path`` 可以是 npz 文件或包含 ``expert_bc.npz`` 的目录。

        ``limit``（``--limit-dataset``，冒烟用）：**读取阶段**按完整 episode 前缀截断
        （口径见 :func:`episode_prefix_row_count`；只解压前 M 行 → 物化/val 切分/权重
        统计都基于该子集，列语义不变）。不可前缀读取时回退「全量加载 + 同口径内存截断」。
        """
        target = Path(path)
        if target.is_dir():
            target = target / "expert_bc.npz"
        meta_path = target.parent / (target.name.replace(".npz", ".meta.json"))
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        stored_fingerprint = str(meta.get("obs_fingerprint") or "")
        if stored_fingerprint:
            try:
                from env.obs import obs_fingerprint

                current = str(obs_fingerprint())
            except Exception:  # noqa: BLE001
                current = ""
            if current and current != stored_fingerprint:
                import warnings

                warnings.warn(
                    f"BC 数据集 obs 指纹不一致（数据 {stored_fingerprint} != 当前 {current}）："
                    "观测 scope/特征语义已变更，必须重新采集 BC 数据（tools/collect_expert.py）。",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if limit is not None:
            prefixed = _npz_episode_prefix(target, int(limit))
            if prefixed is not None:
                arrays, rows, total, total_episodes = prefixed
                _log_dataset_limit(
                    target, int(limit), rows, total, arrays["episode_id"], total_episodes
                )
                return cls(arrays, meta)
            print(
                f"[dataset] 警告：{target} 不支持前缀读取（非 npz/C 连续）→ 全量加载后按 "
                "episode 前缀截断（峰值 RSS 不随 limit 下降）",
                flush=True,
            )
        with np.load(target) as payload:
            arrays = {key: payload[key] for key in payload.files}
        dataset = cls(arrays, meta)
        if limit is not None:
            rows = episode_prefix_row_count(dataset.arrays["episode_id"], int(limit))
            if rows < dataset.count:
                _log_dataset_limit(
                    target,
                    int(limit),
                    rows,
                    dataset.count,
                    dataset.arrays["episode_id"][:rows],
                    int(np.unique(dataset.arrays["episode_id"]).size),
                )
                keep = np.arange(rows, dtype=np.int64)
                dataset = cls(
                    {key: value[keep] for key, value in dataset.arrays.items()}, dataset.meta
                )
        return dataset

    def _frame_obs(self, item: int) -> Dict[str, Any]:
        """单帧当前通道 + mask（不拼历史）。"""
        obs = {key: self.arrays[key][item] for key in self._obs_keys if key in self.arrays}
        for key in list(obs):
            mask_key = f"{key}_mask"
            if mask_key in self.arrays:
                obs[mask_key] = self.arrays[mask_key][item]
        return obs

    def _entries(self, index: int) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """6 帧历史 entries（旧→新）+ valid；优先精确查表，无 step 时按行位置（兼容）。"""
        if not self._step_lookup:
            episode = int(self.arrays["episode_id"][index])
            start, _ = self._episode_ranges[episode]
            first = max(start, index - 5)
            entries: List[Dict[str, Any]] = []
            for item in range(first, index + 1):
                entries.append({"obs": self._frame_obs(item), "pose": self.arrays["pose"][item]})
            valid = np.asarray(
                self.arrays.get("hist_valid", np.ones((self.count, 6), dtype=np.float32))[index],
                dtype=np.float32,
            )
            if len(entries) < 6:
                entries = [entries[0]] * (6 - len(entries)) + entries
            return entries[-6:], valid
        episode = int(self.arrays["episode_id"][index])
        step = int(self.arrays["step"][index])
        valid = np.zeros(6, dtype=np.float32)
        rows: List[Optional[int]] = []
        for j in range(5, -1, -1):
            row = self._step_lookup.get((episode, step - self.history_stride * j))
            rows.append(row)
            if row is not None:
                valid[5 - j] = 1.0
        # 缺失帧用最近的真实帧补位（mask 由 valid 清零；对齐结果不参与有效计算）
        fallback = next((row for row in reversed(rows) if row is not None), int(index))
        entries = [
            {"obs": self._frame_obs(row), "pose": self.arrays["pose"][row]}  # type: ignore[index]
            if row is not None
            else {"obs": self._frame_obs(fallback), "pose": self.arrays["pose"][fallback]}
            for row in rows
        ]
        return entries, valid

    def build_obs_batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """样本索引 → 网络输入（当前帧 + 6 帧历史，B 维在前）。

        v2 mem 历史数组存在时直接复用（采集时精确入库，避免按行位置重拼）；
        否则按 :meth:`_entries` 的精确查表/行位置回退重建。
        """
        idx = np.asarray(indices, dtype=np.int64)
        current: Dict[str, List[np.ndarray]] = {key: [] for key in self._obs_keys}
        masks: Dict[str, List[np.ndarray]] = {
            f"{key}_mask": [] for key in self._obs_keys if f"{key}_mask" in self.arrays
        }
        for index in idx:
            for key in self._obs_keys:
                current[key].append(self.arrays[key][index])
            for key in masks:
                masks[key].append(self.arrays[key][index])
        batch = {key: np.stack(values, axis=0).astype(np.float32) for key, values in current.items()}
        batch.update({key: np.stack(values, axis=0).astype(np.float32) for key, values in masks.items()})
        # v2：od_presence 是 OD 有效性的优先口径（聚类特征/软目标与 net 都用它）
        if "od_presence" in self.arrays:
            batch["od_presence"] = np.asarray(self.arrays["od_presence"], dtype=np.float32)[idx]
        if self.has_mem_history:
            history = {key: self.arrays[key][idx] for key in self._mem_history_keys}
            for key in self._mem_history_keys:
                history[f"{key}_mask"] = self.arrays[f"{key}_mask"][idx]
            history["hist_valid"] = self.arrays["hist_valid"][idx]
        else:
            history: Dict[str, List[np.ndarray]] = {}
            hist_valid: List[np.ndarray] = []
            for index in idx:
                entries, valid = self._entries(int(index))
                patch = stack_history(
                    entries, self.alignments, current_pose=self.arrays["pose"][index], valid=valid
                )
                for key, value in patch.items():
                    history.setdefault(key, []).append(value)
                hist_valid.append(valid)
            history = {key: np.stack(values, axis=0) for key, values in history.items()}
            history["hist_valid"] = np.stack(hist_valid, axis=0)
        batch.update({key: np.asarray(value, dtype=np.float32) for key, value in history.items()})
        return sanitize_masked_od(squeeze_single_slot(batch))

    def targets(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        idx = np.asarray(indices, dtype=np.int64)
        out = {
            "action": self.arrays["action"][idx].astype(np.float32),
            "traj6": self.arrays["traj6"][idx].astype(np.float32),
            "traj30": self.arrays["traj30"][idx].astype(np.float32),
            "labels": self.arrays["labels"][idx].astype(np.float32),
            "sample_weight": self.arrays.get("sample_weight", np.ones(self.count, dtype=np.float32))[idx].astype(
                np.float32
            ),
            # v2 权重（train_weight/balance_weight 的合成分；缺失时 = sample_weight/1）
            "train_weight": row_action_weights(self, idx).astype(np.float32),
        }
        if V2_WM_VALID in self.arrays:
            out["wm_valid"] = self.arrays[V2_WM_VALID][idx].astype(np.float32)
        return out

    def sample_indices(self) -> np.ndarray:
        return np.arange(self.count, dtype=np.int64)


class MaterializedBCDataset:
    """``BCDataset`` 的一次性物化（obs 历史 + 可选目标 → 连续数组）。

    背景（实测 2026-09-26，H=128 / batch=256）：逐样本 ``build_obs_batch``（6 帧历史精确
    查表 + SE(2) 对齐 + numpy 拼接）占单 batch 墙钟 >95%（Stage A ~0.44 s vs GPU
    前反向 5–15 ms）。本类把逐样本重建**移出训练循环**，循环内只做 ``arr[idx]`` 切片
    + ``torch.as_tensor(...)``（H2D 走 pin_memory + non_blocking，见
    :func:`to_device_tensors`）。

    数学等价：obs 由 ``BCDataset.build_obs_batch`` 分块生成后逐字段原样存储（bitwise
    相同，dtype 不变）；目标由 ``BCDataset.targets`` 的同一公式一次性预计算
    （``train_weight = row_action_weights``；``traj30`` 训练不用，不物化以省内存）。

    内存量级（144k 行 v2）：obs ≈ 1.35 GB；Stage B 目标 < 30 MB。Stage A 的未来目标由
    ``pipeline.stages`` 单独物化（同一 ``_future_targets`` 路径）。
    """

    #: 目标键（与 ``BCDataset.targets`` 对齐；缺失的源键跳过）
    TARGET_SOURCE_KEYS: Tuple[str, ...] = ("action", "traj6", "labels", "sample_weight")

    def __init__(
        self,
        dataset: BCDataset,
        *,
        chunk_size: int = 2048,
        include_targets: bool = True,
        logger: Callable[[str], None] = print,
    ):
        started = time.perf_counter()
        self.dataset = dataset
        self.count = int(dataset.count)
        chunk = max(1, int(chunk_size))
        self.arrays: Dict[str, np.ndarray] = {}
        for start in range(0, self.count, chunk):
            stop = min(start + chunk, self.count)
            batch = dataset.build_obs_batch(np.arange(start, stop, dtype=np.int64))
            if not self.arrays:
                self.arrays = {
                    key: np.empty(
                        (self.count, ) + tuple(np.asarray(value).shape[1:]), dtype=np.asarray(value).dtype
                    )
                    for key, value in batch.items()
                }
            for key, value in batch.items():
                self.arrays[key][start:stop] = value
        self.targets: Dict[str, np.ndarray] = {}
        if include_targets:
            source = dataset.arrays
            for key in self.TARGET_SOURCE_KEYS:
                if key in source:
                    self.targets[key] = np.asarray(source[key], dtype=np.float32)
            if V2_WM_VALID in source:
                self.targets["wm_valid"] = np.asarray(source[V2_WM_VALID], dtype=np.float32)
            self.targets["train_weight"] = row_action_weights(
                dataset, np.arange(self.count, dtype=np.int64)
            ).astype(np.float32)
        self.nbytes = int(
            sum(value.nbytes for value in self.arrays.values())
            + sum(value.nbytes for value in self.targets.values())
        )
        logger(
            f"[materialize] obs+targets {self.count} 行 → {self.nbytes / 1e6:.1f} MB "
            f"（{time.perf_counter() - started:.1f}s，chunk={chunk}）"
        )

    def obs_batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """训练 batch 的 obs（与 ``BCDataset.build_obs_batch`` 逐值相同；只做切片）。"""
        idx = np.asarray(indices, dtype=np.int64)
        return {key: value[idx] for key, value in self.arrays.items()}

    def targets_batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """训练 batch 的目标（与 ``BCDataset.targets`` 相同键值；只做切片）。"""
        idx = np.asarray(indices, dtype=np.int64)
        return {key: value[idx] for key, value in self.targets.items()}


def to_device_tensor(
    value: np.ndarray,
    device: Any,
    *,
    dtype: Optional["torch.dtype"] = None,
    pin: bool = False,
    non_blocking: bool = True,
) -> "torch.Tensor":
    """numpy → device 张量（可选 pin_memory + non_blocking H2D）。

    ``dtype=None`` 保持原 dtype（future 目标的 ``od_id_fut`` 是 int64，不能被强转 fp32）；
    ``pin`` 只对 CPU 张量生效，失败（不支持的内存）时退化为普通拷贝。
    """
    _require_torch()
    tensor = torch.as_tensor(value, dtype=dtype)
    if pin and tensor.device.type == "cpu":
        try:
            tensor = tensor.pin_memory()
        except Exception:  # noqa: BLE001 - pin 失败退化为普通 H2D
            pass
    return tensor.to(device, non_blocking=bool(non_blocking))


def to_device_tensors(
    batch: Mapping[str, np.ndarray],
    device: Any,
    *,
    dtype: Optional["torch.dtype"] = None,
    pin: bool = False,
    non_blocking: bool = True,
) -> Dict[str, "torch.Tensor"]:
    """:func:`to_device_tensor` 的映射批量版（键集/顺序保持）。"""
    return {
        key: to_device_tensor(value, device, dtype=dtype, pin=pin, non_blocking=non_blocking)
        for key, value in batch.items()
    }


def _to_device_obs(
    batch: Mapping[str, np.ndarray], device: "torch.device", *, pin: bool = False
) -> Dict[str, "torch.Tensor"]:
    return to_device_tensors(batch, device, dtype=torch.float32, pin=pin)


#: 单槽通道（net ``_validate_obs`` 期望 ``(B,F)`` 而非 ``(B,1,F)``）
SINGLE_SLOT_CHANNELS: Tuple[str, ...] = ("ego", "nav", "signal", "others")


def squeeze_single_slot(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """把单槽通道的槽位维去掉（``(B,1,F) → (B,F)``），匹配 ``net.model.DrivingModel`` 的输入校验。"""
    for name in SINGLE_SLOT_CHANNELS:
        value = batch.get(name)
        if value is not None and value.ndim == 3 and value.shape[1] == 1:
            batch[name] = value[:, 0]
    return batch


def sanitize_masked_od(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """把 OD 通道中 ``mask=0`` 槽位的 ``(cosθ, sinθ)`` 置为 ``(1, 0)``。

    为什么需要（N1 接口现状）：``net/encoders.py::od_pose`` 先用 ``atan2(sin,cos)`` 再乘掩码；
    全 0 槽位在反向传播时 ``atan2`` 的梯度未定义（实测 ``Atan2Backward0 returned nan``），
    会让**任何用到 ``traj_xy`` 的损失**（BC 轨迹损失）直接 NaN。置 ``(1,0)`` 不改变掩码
    语义（该槽位最终仍被 mask 清零），只提供合法梯度；正式修复应由 N1 在 ``od_pose`` 内完成。
    """
    for feat_key, mask_key in (("od", "od_mask"), ("od_hist", "od_hist_mask")):
        feats = batch.get(feat_key)
        mask = batch.get(mask_key)
        if feats is None or mask is None or feats.ndim < 3:
            continue
        invalid = np.asarray(mask) < 0.5
        if not bool(invalid.any()):
            continue
        if invalid.shape != feats.shape[:-1]:  # 形状不一致时跳过（由模型侧报错更清晰）
            continue
        feats = np.array(feats, copy=True)
        feats[..., 4][invalid] = 1.0  # cosθ = 1
        feats[..., 5][invalid] = 0.0  # sinθ = 0
        batch[feat_key] = feats
    return batch


def trajectory_target_indices(num_points: int, dense_points: int) -> "np.ndarray":
    """模型 ``num_points`` 个动作端点（t=0.5..K·0.5 s）在密集目标中的时间对齐下标。

    密集目标按 0.1 s 采样（``traj30``：t=0.1..3.0 s）：第 k 个模型点（t=(k+1)·0.5 s）
    对应下标 ``(k+1)·dense/num - 1``。K=6、D=30 → ``[4, 9, 14, 19, 24, 29]``
    （= 数据集 ``traj6`` 的确切来源；旧的 ``linspace(0,29,6)`` 给
    ``[0, 6, 12, 18, 24, 29]`` = t=0.1/0.7/1.3/1.9/2.5/3.0 s，时间错位）。
    """
    step = float(dense_points) / float(num_points)
    idx = np.rint((np.arange(num_points) + 1) * step - 1.0).astype(np.int64)
    return np.clip(idx, 0, dense_points - 1)


def bc_trajectory_loss(
    pred: "torch.Tensor",
    target_dense: "torch.Tensor",
    *,
    loss_type: str = "l2",
    weights: Optional["torch.Tensor"] = None,
    metric_weights: Optional["torch.Tensor"] = None,
) -> Tuple["torch.Tensor", Dict[str, float]]:
    """轨迹 L1/L2：模型 ``(B,K,2)`` vs 密集插值目标 ``(B,D,2)``（K≠D 时按时间对齐取点）。

    优先直接传数据集 ``traj6``（与模型 6 个端点时间对齐、无重采样误差）；
    传 ``traj30`` 时用 :func:`trajectory_target_indices` 取 t=k+1 的采样点。

    返回 ``(loss, metrics)``：

    - ``loss``：``loss_type`` 口径的**加权**训练损失（l2→加权 MSE，l1→加权 MAE；
      ``Σ w·err_i/Σ w``，权重全 0 时按 1e-8 兜底防除零）；
    - ``metrics``（纯诊断标量，与梯度无关，2026-09-26 新增）：
      ``traj_mse_weighted`` = 加权 MSE（m²，与 ``loss_type`` 无关）、
      ``traj_mae_weighted_m`` = 加权 MAE（m，与 ``loss_type`` 无关）、
      ``traj_mae_all_m`` = **未加权** MAE（m，**含 w=0 的被过滤帧**，仅诊断）。
    - ``metric_weights``（lane U4）：**诊断指标的权重**（缺省 = ``weights``）——
      traj 损失可被逐行掩码（DAgger 行不吃 traj-aux），但 ``traj_mse/mae`` 监控统计
      保持全量口径（报告照旧）。

    .. warning::
        旧接口返回的第二个元素是**未加权 MSE（m²）**却被命名为 ``mae``：因为
        ``train_weight=0`` 的近崩溃/离道帧被排除在加权值之外、而这些帧未来误差极大，
        两者相差近一个量级。新接口用显式字典区分口径，数值与本函数 ``loss`` 的
        加权口径一致。
    """
    if pred.shape[1] != target_dense.shape[1]:
        target_idx = torch.as_tensor(
            trajectory_target_indices(pred.shape[1], target_dense.shape[1]), device=pred.device, dtype=torch.long
        )
        target = target_dense[:, target_idx]
    else:
        target = target_dense
    diff = pred - target
    mse_sample = (diff ** 2).mean(dim=(-1, -2))
    mae_sample = diff.abs().mean(dim=(-1, -2))
    per_sample = mse_sample if loss_type == "l2" else mae_sample
    if weights is not None:
        # 显式加权：Σ w·err / Σ w（等价于旧式 w/mean(w) 的 .mean()，但避免权重全 0 时除零放大）
        weight = weights.reshape(-1)
        denom = weight.sum().clamp(min=1e-8)
        loss = (per_sample * weight).sum() / denom
    else:
        loss = per_sample.mean()
    metric_weight = weights if metric_weights is None else metric_weights
    if metric_weight is not None:
        weight_m = metric_weight.reshape(-1)
        denom_m = weight_m.sum().clamp(min=1e-8)
        mse_weighted = (mse_sample * weight_m).sum() / denom_m
        mae_weighted = (mae_sample * weight_m).sum() / denom_m
    else:
        mse_weighted = mse_sample.mean()
        mae_weighted = mae_sample.mean()
    metrics = {
        "traj_mse_weighted": float(mse_weighted.detach()),  # m²，加权（与 loss_type 无关）
        "traj_mae_weighted_m": float(mae_weighted.detach()),  # m，加权（与 loss_type 无关）
        "traj_mae_all_m": float(mae_sample.mean().detach()),  # m，未加权，含被过滤帧（诊断）
    }
    return loss, metrics


def wm_teacher_forcing_predictions(
    model: "nn.Module",
    obs: Mapping[str, "torch.Tensor"],
    future: Mapping[str, Any],
    actions: "torch.Tensor",
) -> Dict[str, "torch.Tensor"]:
    """WM 教师强制多步前向（stage A ``_wm_predictions`` 的 trainer 版；lane P3-B）。

    与 stage A 同一机制，但 action 条件来自调用方（phase 3 = rollout ``plan`` 的动作链，
    **已在调用方 detach**）：

    - 每步先用当前编码 mem 跑 plan head 得 ``ego_next_pred``（第 k 帧 ego 特征预测）；
    - 再把**目标帧真实 ego**（``future["ego_fut"][:, k-1]``，无则解析运动学兜底）挤入 mem
      副本（``reserved`` 两维写第 k 个动作），合成帧一律 ``detach``；
    - ST-GNN 单步推演得 t0 帧的 ``od_pred/ld_pred/presence/entry``。

    返回逐键堆叠 ``(B,K,...)``：``od_pred (B,K,16,5)`` / ``ld_pred (B,K,16,4)`` /
    ``presence_pred|entry_pred (B,K,16)`` / ``ego_next_pred (B,K,H6)``（ego 前 6 维）。
    """
    import torch

    encoded = model.encode(obs)
    mem = encoded["mem"].clone()
    enc = encoded["encoded"]
    nav_token = encoded["nav_token"]
    signal_token = encoded["signal_token"]
    anchor_od = model.st_gnn.od_state_from_features(enc.od_now, enc.od_live)
    anchor_ld = model.st_gnn.ld_state_from_features(enc.ld_now, enc.ld_live)
    ego_fut = future.get("ego_fut")
    ego_now = obs["ego"]
    if ego_now.ndim == 3:
        ego_now = ego_now[:, 0]
    horizon = int(actions.shape[1])
    predictions: Dict[str, List["torch.Tensor"]] = {
        "od_pred": [], "ld_pred": [], "presence_pred": [], "entry_pred": [], "ego_next_pred": []
    }
    analytic_ego = None
    if ego_fut is None:
        try:
            from net.model import ego_next_features  # type: ignore

            analytic_ego = ego_next_features
        except Exception:  # noqa: BLE001
            analytic_ego = None
    for k in range(1, horizon + 1):
        _, ego_next_pred, _ = model.plan_step(enc, nav_token, signal_token)
        predictions["ego_next_pred"].append(ego_next_pred)
        raw_action = actions[:, k - 1]
        if ego_fut is not None:
            ego_frame = ego_fut[:, k - 1]
            if not torch.is_tensor(ego_frame):
                ego_frame = torch.as_tensor(
                    np.asarray(ego_frame), dtype=torch.float32, device=raw_action.device
                )
        elif analytic_ego is not None:
            ego_frame = analytic_ego(
                ego_now, raw_action[:, 0], raw_action[:, 1], dt=float(model.dt), prev_speed=ego_now[:, 0]
            )
        else:  # 兜底：复制当前帧 + 动作条件
            ego_frame = torch.cat([ego_now[:, :6], raw_action], dim=-1)
        if ego_frame.ndim == 3 and ego_frame.shape[1] == 1:
            ego_frame = ego_frame[:, 0, :]
        ego_frame = torch.cat([ego_frame[..., :6], raw_action], dim=-1)  # reserved 维 = 该步动作
        mem.shift_ego(ego_frame.detach())
        enc_k = model.mem_encoder.encode(model.encoders, mem)
        od_pred, ld_pred, presence, entry = model.st_gnn(
            ego_ctx=enc_k.ego_ctx,
            od_ctx=enc_k.od_ctx,
            ld_ctx=enc_k.ld_ctx,
            node_mask=enc_k.frame.node_mask,
            pose=enc_k.frame.pose,
            step_index=k,
            od_anchor=anchor_od,
            ld_anchor=anchor_ld,
        )
        predictions["od_pred"].append(od_pred)
        predictions["ld_pred"].append(ld_pred)
        predictions["presence_pred"].append(presence)
        predictions["entry_pred"].append(entry)
        enc = enc_k  # 下一轮 plan head 看到 ≤ k 帧
    return {key: torch.stack(values, dim=1) for key, values in predictions.items()}


@_with_safe_od_pose
def pretrain_bc(
    model: "nn.Module",
    dataset: BCDataset,
    config: BCConfig,
    *,
    logger: Callable[[str], None] = print,
    batch_source: Optional[MaterializedBCDataset] = None,
) -> Dict[str, Any]:
    """BC 预热循环：轨迹 L1/L2 + **首步**动作回归 + router **硬标签 CE**（权重感知）。

    - 轨迹目标用 ``targets["traj6"]``（与模型 6 个 rollout 端点逐点时间对齐；``traj30``
      是 0.1 s 密集采样，直接对 6 点预测使用会时间错位）；
    - 动作目标 = ``targets["action"][:, 0, :]``（专家当前帧的**即时** (ds,dθ)），
      监督 ``out["action_mu"]``；v2 语义只监督**首步**动作，6 点轨迹辅助承担多步信息；
    - **权重感知**：动作/轨迹/router 损失与所有误差统计都用
      ``targets["train_weight"]``（= ``train_weight × 配平权重``）；显式 ``Σ w·x / Σ w``，
      权重 0 的帧不参与（不再按有效项均摊）；
    - **MoE（lane U1）**：无聚类监督标签；``moe_enabled=False``（phase 1）时专家不参与、
      输出 = primary；``moe_enabled=True``（phase 2）时输出 = ``primary + Σ_{i∈top2} g_i·expert_i``，
      并加 Switch 式负载均衡 aux（``load_balance_coef``）。
    - **traj-aux 逐行掩码（lane U4）**：``traj_aux_valid``（0/1，数据集行空间；None = 全 1）只作用于
      **traj 损失项**（``specific_weight × traj_valid``）——DAgger 行的 traj6/traj30 是常量外推
      合成值、不吃 traj 损失；action/负载项与 ``bc_traj_*`` 监控统计保持全量口径。

    阶段 B（v1.1）：``config.wm_detach`` 时 rollout 内 WM 输出 detach；
    ``config.freeze_prefixes`` 在优化器构建前生效（primary→specific 分段训练）。

    ``batch_source``（2026-09-26 GPU 路径）：传入 :class:`MaterializedBCDataset` 时训练
    循环只做切片 + pin/non-blocking H2D（逐样本历史重建移到训练前一次性完成）；
    语义/数值与 ``None``（旧路径）逐位一致（等价性测试见 ``tests/test_fast_data_path.py``）。

    ``config.micro_batch_size``（梯度累积）：< ``batch_size`` 时按 micro 前向/反向、
    宏 batch 一次更新；各损失项按宏口径精确缩放（``Σ_m s_m·L_m = L_macro``），
    12GB 卡上宏 batch 1024 靠它把显存峰值压到 micro 级别（net ST-GNN 6 步展开 ~32MB/样本）。

    留出集与逐 epoch 落盘（2026-09-26）：``config.train_indices`` 限定训练行（Stage B
    传按 episode 留出后的 ``train_idx``）；``config.val_indices`` 非空时每 epoch 末调用
    :func:`evaluate_bc` 在留出集上评估（``torch.no_grad()``，不参与梯度）；
    ``config.epoch_callback(epoch, epoch_metrics, val_metrics)`` 每 epoch 末回调，
    ``epoch_metrics`` 是**该 epoch 增量**（含逐 horizon/切片/标签/router 统计），
    ``val_metrics`` 为留出集指标（无留出集时 ``{}``）。step 轴由编排层负责。

    返回指标（既有键不退化，新增按 horizon/label/slice 分组；全部带 count/weighted 双口径；
    2026-09-26 轨迹度量单位修正；**末尾汇总仍为全阶段累计**，逐 epoch 增量走回调；
    有 ``val_indices`` 时另含 ``metrics["val"]`` = 最后一个 epoch 的留出集快照
    （:func:`evaluate_bc` 返回，供 metrics.json/il_report 读取））：

    - 轨迹全局：``bc_traj_loss``（加权损失项，含 ``traj_weight``）、``bc_traj_mse``（加权 MSE，m²）、
      ``bc_traj_mae_m``（**加权 MAE，m**；旧语义 = 未加权 MSE(m²)，已修正）、
      ``bc_traj_fde_m``（**末点 FDE，m，加权**；lane B B2）、
      ``bc_traj_mae_all_m``（未加权 MAE，m，含 ``train_weight=0`` 的被过滤帧，仅诊断）、
      ``bc_traj_mse_unweighted``（旧 ``bc_traj_mae_m`` 口径 alias：未加权 MSE，m²）；
    - 轨迹逐 horizon：``bc_traj_mse_h{k}``（加权 MSE，m²）、``bc_traj_mae_h{k}_m``（加权 MAE，m）、
      ``bc_traj_err_h{k}``（legacy alias：loss_type 口径的加权误差——l2→MSE(m²) / l1→MAE(m)）；
    - 动作侧统计是显式 L1（米/弧度），不受 ``loss_type`` 影响：``bc_action_err_{mean,weighted_mean,
      median,p95}``、``bc_action_err_slice_{brake,turn,curve}_*``、``bc_action_err_label_<name>``；
    - 另有 ``bc_loss``、``bc_action_loss``、``bc_load_balance_loss``、``bc_action_mu_ds_mean``、
      ``bc_expert_load_{0..7}``/``bc_load_cv``/``bc_gate_entropy``/``bc_load_count``
      （lane U1：MoE 负载诊断；phase 1 MoE 关闭 → 缺省）。
    """
    _require_torch()
    order_config = load_supervised_labels()
    if tuple(dataset.label_names) != order_config:
        raise ValueError(
            f"数据集标签顺序 {tuple(dataset.label_names)} 与 config/model.yaml 的 {order_config} 不一致（固定顺序契约）"
        )
    device = torch.device(resolve_device(config.device))
    model.to(device).train()
    frozen = apply_freeze_prefixes(model, config.freeze_prefixes)
    optimizer = build_optimizer(model, config.lr)
    if config.optimizer_state:
        load_optimizer_state(optimizer, config.optimizer_state, logger=logger)
        move_optimizer_state_to_device(optimizer)  # 跨设备：状态张量跟随参数设备（幂等加固）
    if config.train_indices is not None:
        indices = np.asarray(config.train_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("config.train_indices 为空 → 无训练样本")
    else:
        indices = dataset.sample_indices()
    val_indices = (
        None
        if config.val_indices is None
        else np.asarray(config.val_indices, dtype=np.int64).reshape(-1)
    )
    if val_indices is not None and val_indices.size == 0:
        val_indices = None
    start_epoch = max(0, min(int(config.start_epoch or 0), int(config.epochs)))
    if start_epoch:
        logger(f"[bc] resume：跳过已完成 {start_epoch}/{int(config.epochs)} 个 epoch（phase={config.phase}）")
    rng = np.random.default_rng(config.seed)
    if start_epoch:
        if config.rng_state:
            # 周期 ckpt 的 RNG 状态：数据顺序/前向随机性逐位续跑
            restore_rng_state(config.rng_state, numpy_generator=rng)
        elif config.shuffle:
            # 无 RNG 状态（旧 ckpt/未捕获）→ seed 重放：跳过已完成 epoch 的 shuffle
            for _ in range(start_epoch):
                probe = indices.copy()
                rng.shuffle(probe)
    metrics: Dict[str, Any] = {
        "epochs": int(config.epochs),
        "batches": 0,
        "phase": str(config.phase),
        "frozen_params": len(frozen),
        "trainable_params": sum(1 for parameter in model.parameters() if parameter.requires_grad),
    }
    on_curve_index = (
        dataset.label_names.index("on_curve") if "on_curve" in dataset.label_names else None
    )
    # 逐样本误差/权重缓冲（整个训练累计后统一做 count/weighted 统计，避免 epoch 均值混淆口径）
    action_err_values: List[np.ndarray] = []
    label_values: List[np.ndarray] = []
    slice_err_values: Dict[str, Dict[str, List[np.ndarray]]] = {
        "brake": {"err": [], "weight": []},
        "turn": {"err": [], "weight": []},
        "curve": {"err": [], "weight": []},
    }
    traj_mse_values: List[np.ndarray] = []
    traj_mae_values: List[np.ndarray] = []
    traj_fde_values: List[np.ndarray] = []
    weight_values: List[np.ndarray] = []
    mu_ds_sum, mu_ds_weight = 0.0, 0.0
    # lane U1：MoE 负载累计（行数加权；expert_load Σ=1）
    num_experts_cfg = int(getattr(model.plan_head.moe, "num_experts", 0) or 0)
    expert_load_sum: Optional[np.ndarray] = None
    load_cv_num, load_cv_den = 0.0, 0.0
    gate_entropy_num, gate_entropy_den = 0.0, 0.0
    load_count = 0

    def _load_snapshot() -> Dict[str, Any]:
        """MoE 负载累计状态的 epoch 起点快照（用于增量统计）。"""
        return {
            "load_sum": None if expert_load_sum is None else expert_load_sum.copy(),
            "cv_num": float(load_cv_num),
            "cv_den": float(load_cv_den),
            "entropy_num": float(gate_entropy_num),
            "entropy_den": float(gate_entropy_den),
            "count": int(load_count),
        }

    def _load_epoch_metrics(start: Mapping[str, Any]) -> Dict[str, Any]:
        """epoch 增量负载指标（与全阶段累计块同一 :func:`_load_summary` 公式）。"""
        load_sum = None
        if expert_load_sum is not None:
            load_sum = np.asarray(expert_load_sum)
            if start["load_sum"] is not None:
                load_sum = load_sum - np.asarray(start["load_sum"])
        return _load_summary(
            count=int(load_count) - int(start["count"]),
            expert_load_sum=load_sum,
            load_cv_num=float(load_cv_num) - float(start["cv_num"]),
            load_cv_den=float(load_cv_den) - float(start["cv_den"]),
            entropy_num=float(gate_entropy_num) - float(start["entropy_num"]),
            entropy_den=float(gate_entropy_den) - float(start["entropy_den"]),
            num_experts=num_experts_cfg,
        )

    def _epoch_stats(mark: Mapping[str, Any]) -> Dict[str, Any]:
        """epoch 增量统计：动作误差（count/weighted 双口径）、分切片/标签、逐 horizon 轨迹、MoE 负载。

        与末尾全阶段累计块同公式（同一 :func:`weighted_stats` / :func:`_load_summary`），
        只是对 epoch 起点快照后的缓冲切片计算；对齐异常时跳过并记
        ``bc_action_err_alignment_warning=1``（正常数据不应出现）。
        """
        out: Dict[str, Any] = {}
        epoch_errors = (
            np.concatenate(action_err_values[mark["action_err"]:])
            if len(action_err_values) > mark["action_err"]
            else None
        )
        epoch_weights = (
            np.concatenate(weight_values[mark["weights"]:])
            if len(weight_values) > mark["weights"]
            else None
        )
        if epoch_errors is not None:
            if epoch_weights is None or epoch_weights.shape[0] != epoch_errors.shape[0]:
                out["bc_action_err_alignment_warning"] = 1.0
            else:
                out.update(weighted_stats(epoch_errors, epoch_weights, prefix="bc_action_err_"))
                for name, parts in slice_err_values.items():
                    err_start, weight_start = mark["slices"][name]
                    if len(parts["err"]) <= err_start:
                        continue
                    err_slice = np.concatenate(parts["err"][err_start:])
                    weight_slice = np.concatenate(parts["weight"][weight_start:])
                    if err_slice.shape != weight_slice.shape:
                        continue
                    stats = weighted_stats(err_slice, weight_slice, prefix="")
                    out[f"bc_action_err_slice_{name}"] = stats
                    out[f"bc_action_err_slice_{name}_weighted_mean"] = stats["weighted_mean"]
                if len(label_values) > mark["labels"]:
                    labels_epoch = np.concatenate(label_values[mark["labels"]:])
                    if labels_epoch.ndim == 2 and labels_epoch.shape[0] == epoch_errors.shape[0]:
                        for label_index, label_name in enumerate(dataset.label_names):
                            if label_index >= labels_epoch.shape[1]:
                                break
                            mask = labels_epoch[:, label_index] > 0.5
                            if not bool(np.any(mask)):
                                continue
                            stats = weighted_stats(epoch_errors[mask], epoch_weights[mask], prefix="")
                            out[f"bc_action_err_label_{label_name}"] = stats["weighted_mean"]
                            out[f"bc_action_err_label_{label_name}_count"] = stats["count"]
        if len(traj_mse_values) > mark["traj_mse"] and len(traj_mae_values) > mark["traj_mae"]:
            epoch_mse = np.concatenate(traj_mse_values[mark["traj_mse"]:])
            epoch_mae = np.concatenate(traj_mae_values[mark["traj_mae"]:])
            if (
                epoch_weights is not None
                and epoch_weights.shape[0] == epoch_mse.shape[0]
                and epoch_mse.shape == epoch_mae.shape
            ):
                legacy = epoch_mse if config.loss_type == "l2" else epoch_mae
                for k in range(epoch_mse.shape[1]):
                    out[f"bc_traj_mse_h{k + 1}"] = weighted_stats(
                        epoch_mse[:, k], epoch_weights, prefix=""
                    )["weighted_mean"]
                    out[f"bc_traj_mae_h{k + 1}_m"] = weighted_stats(
                        epoch_mae[:, k], epoch_weights, prefix=""
                    )["weighted_mean"]
                    out[f"bc_traj_err_h{k + 1}"] = weighted_stats(
                        legacy[:, k], epoch_weights, prefix=""
                    )["weighted_mean"]
        if len(traj_fde_values) > mark["traj_fde"]:
            epoch_fde = np.concatenate(traj_fde_values[mark["traj_fde"]:])
            if epoch_weights is not None and epoch_weights.shape[0] == epoch_fde.shape[0]:
                out["bc_traj_fde_m"] = weighted_stats(epoch_fde, epoch_weights, prefix="")["weighted_mean"]
        out.update(_load_epoch_metrics(mark["load"]))
        return out

    # lane U1：worst-50% 行权重（权重化 specific；全量曝光，不做硬子集）
    worst_all = (
        None
        if config.worst_flags is None
        else np.asarray(config.worst_flags, dtype=np.float32).reshape(-1)
    )
    if worst_all is not None:
        if worst_all.size != int(dataset.count):
            raise ValueError(
                f"worst_flags 行数 {worst_all.size} != 数据集行数 {int(dataset.count)}"
                "（权重 sidecar 必须与训练集逐行对齐）"
            )
        worst_rows = int(np.count_nonzero(worst_all > 0.5))
        logger(
            f"[bc] 权重化 specific：worst={worst_rows}/{worst_all.size}"
            f"（{worst_rows / max(1, worst_all.size):.1%}；hard_weight={config.hard_weight} · "
            f"mild_weight={config.mild_weight} · moe_enabled={config.moe_enabled} · "
            f"load_balance_coef={config.load_balance_coef}）"
        )
    model.set_moe(enabled=bool(config.moe_enabled), load_balance_coef=float(config.load_balance_coef))
    # lane U4：traj-aux 逐行掩码（DAgger 行 = 0：其 traj6 是合成常量外推，不得进 traj 损失）
    traj_valid_all = (
        None
        if config.traj_aux_valid is None
        else np.asarray(config.traj_aux_valid, dtype=np.float32).reshape(-1)
    )
    if traj_valid_all is not None:
        if traj_valid_all.size != int(dataset.count):
            raise ValueError(
                f"traj_aux_valid 行数 {traj_valid_all.size} != 数据集行数 {int(dataset.count)}"
                "（掩码必须与训练集逐行对齐）"
            )
        masked_rows = int(np.count_nonzero(traj_valid_all <= 0.5))
        logger(
            f"[bc] traj-aux 掩码：masked={masked_rows}/{traj_valid_all.size}"
            f"（{masked_rows / max(1, traj_valid_all.size):.1%}；DAgger 行不吃 traj 损失，"
            "action/负载与监控统计仍全量口径）"
        )

    macro_size = max(1, int(config.batch_size))
    micro_size = (
        macro_size
        if config.micro_batch_size is None
        else max(1, min(macro_size, int(config.micro_batch_size)))
    )
    if micro_size < macro_size:
        logger(
            f"[bc] 梯度累积：宏 batch={macro_size} · micro batch={micro_size}"
            f"（phase={config.phase}；损失按宏口径精确缩放）"
        )
    for epoch in range(start_epoch, int(config.epochs)):
        order = indices.copy()
        if config.shuffle:
            rng.shuffle(order)
        # epoch 起点快照：用于该 epoch 增量的动作/轨迹/router 统计（监控逐 epoch 落盘）
        epoch_mark: Dict[str, Any] = {
            "action_err": len(action_err_values),
            "labels": len(label_values),
            "weights": len(weight_values),
            "traj_mse": len(traj_mse_values),
            "traj_mae": len(traj_mae_values),
            "traj_fde": len(traj_fde_values),
            "slices": {
                name: (len(parts["err"]), len(parts["weight"]))
                for name, parts in slice_err_values.items()
            },
            "load": _load_snapshot(),
        }
        totals = {
            "loss": 0.0,
            "traj": 0.0,
            "action": 0.0,
            "load_balance": 0.0,
            "mu_ds": 0.0,
            # 轨迹度量（2026-09-26 单位修正）：加权 MSE(m²)/MAE(m) + 未加权诊断
            "traj_mse": 0.0,
            "traj_mae": 0.0,
            "traj_mae_all": 0.0,
            "traj_mse_all": 0.0,
        }
        batches = 0
        data_seconds = forward_seconds = backward_seconds = 0.0
        for start in range(0, len(order), macro_size):
            if config.max_batches is not None and batches >= int(config.max_batches):
                break
            batch_indices = order[start : start + macro_size]
            n_macro = int(batch_indices.shape[0])
            data_started = time.perf_counter()
            if batch_source is not None:
                obs_macro = batch_source.obs_batch(batch_indices)
                targets_macro = batch_source.targets_batch(batch_indices)
            else:
                obs_macro = dataset.build_obs_batch(batch_indices)
                targets_macro = dataset.targets(batch_indices)
            weight_macro_np = np.asarray(targets_macro["train_weight"], dtype=np.float64).reshape(-1)
            # lane U1：worst/mild 行权重倍率（与 train_weight×balance_weight 相乘；全量曝光）
            worst_macro_np = worst_flags_for_rows(worst_all, batch_indices)
            scale_macro_np = (
                np.ones_like(weight_macro_np)
                if worst_macro_np is None
                else row_scale_from_worst(
                    worst_macro_np, hard_weight=config.hard_weight, mild_weight=config.mild_weight
                )
            )
            weighted_macro_np = weight_macro_np * scale_macro_np
            weight_macro_total = max(float(weighted_macro_np.sum()), 1e-12)
            # lane U4：traj 项专用宏权重（乘逐行掩码；None = 全 1 → 与 weight_macro 逐位相同）
            traj_valid_macro_np = traj_valid_for_rows(traj_valid_all, batch_indices)
            traj_macro_np = (
                weighted_macro_np
                if traj_valid_macro_np is None
                else weighted_macro_np * traj_valid_macro_np.astype(np.float64)
            )
            traj_macro_total = max(float(traj_macro_np.sum()), 1e-12)
            data_seconds += time.perf_counter() - data_started
            # 梯度累积（micro-batch，可选）：每 micro 前向/反向，按宏 batch 口径精确缩放
            # （Σ_m s_m·L_m = L_macro），optimizer 每宏 batch 一次；micro == macro 时退化旧行为。
            optimizer.zero_grad(set_to_none=True)
            totals_micro = {
                "loss": 0.0,
                "traj": 0.0,
                "action": 0.0,
                "load_balance": 0.0,
                "traj_mse": 0.0,
                "traj_mae": 0.0,
                "traj_mae_all": 0.0,
                "traj_mse_all": 0.0,
            }
            mu_num = mu_den = 0.0
            for m_start in range(0, n_macro, micro_size):
                micro_indices = batch_indices[m_start : m_start + micro_size]
                m_lo, m_hi = m_start, m_start + int(micro_indices.shape[0])
                data_started = time.perf_counter()
                if micro_size >= n_macro:
                    obs_np, targets_np = obs_macro, targets_macro
                else:
                    obs_np = {key: value[m_lo:m_hi] for key, value in obs_macro.items()}
                    targets_np = {key: value[m_lo:m_hi] for key, value in targets_macro.items()}
                obs = _to_device_obs(obs_np, device, pin=batch_source is not None)
                targets = to_device_tensors(targets_np, device, pin=batch_source is not None)
                frame_weight = targets["train_weight"].to(device=device, dtype=torch.float32)
                # lane U1：行权重倍率（worst=hard_weight / 其余=mild_weight；未知=1.0）
                worst_np = worst_flags_for_rows(worst_all, micro_indices)
                if worst_np is None:
                    row_scale_t = torch.ones_like(frame_weight)
                else:
                    row_scale_t = torch.as_tensor(
                        row_scale_from_worst(
                            worst_np, hard_weight=config.hard_weight, mild_weight=config.mild_weight
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                specific_weight = frame_weight * row_scale_t
                weight_sum = specific_weight.sum().clamp(min=1e-8)
                # lane U4：traj 项权重 = specific_weight × traj 掩码（仅 traj 损失；action/负载不变）
                traj_valid_np = traj_valid_for_rows(traj_valid_all, micro_indices)
                if traj_valid_np is None:
                    traj_weight_t = specific_weight
                else:
                    traj_weight_t = specific_weight * torch.as_tensor(
                        traj_valid_np, dtype=torch.float32, device=device
                    )
                # 精确缩放因子：加权项（traj/action/加权轨迹度量）= W_m/W_macro；
                # 逐行均值项（负载 aux / 未加权诊断口径）= n_m/n_macro
                scale_weight = float(weighted_macro_np[m_lo:m_hi].sum()) / weight_macro_total
                scale_traj = float(traj_macro_np[m_lo:m_hi].sum()) / traj_macro_total
                scale_rows = float(m_hi - m_lo) / float(max(1, n_macro))
                weight_values.append(specific_weight.detach().double().cpu().numpy())
                data_seconds += time.perf_counter() - data_started
                forward_started = time.perf_counter()
                # 阶段 B 只需要 traj_xy（B1 rollout）；WM 直接多步预测不参与损失（world_model=False），
                # 且 ``wm_detach`` 时 rollout 内 WM 预测不回传梯度（因果链：WM 冻结 + 输出 detach）。
                # lane U1：MoE 输出 = primary + Σ_{i∈top2} g_i·expert_i（全场景；phase 1 已 set_moe 关闭）。
                out = model(
                    obs,
                    rollout=True,
                    world_model=False,
                    wm_detach=config.wm_detach,
                )
                traj_pred = out.get("traj_xy")
                if traj_pred is None:
                    raise KeyError("模型 forward 缺少 'traj_xy'（契约见 p2-contract §2）")
                traj_loss, traj_metrics = bc_trajectory_loss(
                    traj_pred, targets["traj6"], loss_type=config.loss_type,
                    weights=traj_weight_t,          # lane U4：DAgger 行掩 0（不吃 traj 损失）
                    metric_weights=specific_weight,  # 监控统计保持全量口径（报告照旧）
                )
                traj_term = config.traj_weight * traj_loss
                action_term = torch.zeros((), device=device)
                traj_diff = traj_pred - targets["traj6"]
                # 逐 horizon 矩阵：MSE 与 L1 分开累计（口径与 loss_type 解耦）
                traj_mse_point = (traj_diff ** 2).mean(dim=-1)
                traj_mae_point = traj_diff.abs().mean(dim=-1)
                traj_mse_values.append(traj_mse_point.detach().double().cpu().numpy())
                traj_mae_values.append(traj_mae_point.detach().double().cpu().numpy())
                traj_fde_values.append(
                    torch.linalg.norm(traj_diff[:, -1, :], dim=-1).detach().double().cpu().numpy()
                )
                action_loss = torch.zeros((), device=device)
                action_pred = out.get("action_mu")
                target_action = targets["action"][:, 0, :]
                if action_pred is not None and tuple(action_pred.shape) == tuple(target_action.shape):
                    diff = action_pred - target_action
                    per_sample = (diff ** 2).mean(dim=-1) if config.loss_type == "l2" else diff.abs().mean(dim=-1)
                    action_loss = (per_sample * specific_weight).sum() / weight_sum * config.action_weight
                    action_term = action_loss
                    mu_num += float((action_pred[:, 0] * specific_weight).sum())
                    mu_den += float(weight_sum)
                    # 统计口径：逐样本 L1（不受 loss_type 影响，便于跨实验比较）+ 分切片/分标签误差
                    sample_error = diff.abs().mean(dim=-1).detach().double().cpu().numpy()
                    sample_weight = specific_weight.detach().double().cpu().numpy()
                    action_err_values.append(sample_error)
                    labels_np = targets["labels"].detach().double().cpu().numpy()
                    label_values.append(labels_np)
                    expert_action = target_action.detach().double().cpu().numpy()
                    slice_masks = {
                        "brake": expert_action[:, 0] < float(config.slice_brake_ds),
                        "turn": np.abs(expert_action[:, 1]) >= float(config.slice_turn_dtheta),
                    }
                    if on_curve_index is not None:
                        slice_masks["curve"] = labels_np[:, on_curve_index] > 0.5
                    for name, mask in slice_masks.items():
                        if bool(np.any(mask)):
                            slice_err_values[name]["err"].append(sample_error[mask])
                            slice_err_values[name]["weight"].append(sample_weight[mask])
                    mu_ds_sum += float((action_pred[:, 0] * frame_weight).sum())
                    mu_ds_weight += float(weight_sum)
                load_term = torch.zeros((), device=device)
                # lane U1：MoE 负载统计（行数加权）+ Switch 式 aux（α 已在 MoEBlock 内乘好）
                if out.get("expert_weights") is not None and config.moe_enabled:
                    logits = out.get("router_logits")
                    if logits is not None and hasattr(model, "plan_head"):
                        stats = model.plan_head.moe.load_stats(logits)
                        n_rows = int(logits.shape[0])
                        load = stats["expert_load"].detach().double().cpu().numpy()
                        if expert_load_sum is None or expert_load_sum.size != load.size:
                            expert_load_sum = np.zeros(load.size, dtype=np.float64)
                        expert_load_sum += load * n_rows
                        load_cv_num += float(stats["load_cv"]) * n_rows
                        load_cv_den += n_rows
                        gate_entropy_num += float(stats["gate_entropy"]) * n_rows
                        gate_entropy_den += n_rows
                        load_count += n_rows
                if out.get("load_balance_loss") is not None:
                    load_term = out["load_balance_loss"]
                forward_seconds += time.perf_counter() - forward_started
                # 组合缩放后的宏 batch 损失：Σ_m s_m·L_m = L_macro（micro == macro 时 s=1）
                # router CE 现为**加权均值**（Σw·ce/Σw）→ 与 traj/action 同用权重比缩放
                loss = (
                    traj_term * scale_traj      # lane U4：traj 项按掩码后的宏权重缩放
                    + action_term * scale_weight
                    + load_term * scale_rows
                )
                backward_started = time.perf_counter()
                loss.backward()
                backward_seconds += time.perf_counter() - backward_started
                totals_micro["loss"] += float(loss.detach())
                totals_micro["traj"] += float(traj_term.detach()) * scale_traj
                totals_micro["action"] += float(action_term.detach()) * scale_weight
                totals_micro["load_balance"] += float(load_term.detach()) * scale_rows
                # 轨迹度量：加权口径按 W_m/W_macro、未加权诊断口径按 n_m/n_macro 缩放
                totals_micro["traj_mse"] += traj_metrics["traj_mse_weighted"] * scale_weight
                totals_micro["traj_mae"] += traj_metrics["traj_mae_weighted_m"] * scale_weight
                totals_micro["traj_mae_all"] += traj_metrics["traj_mae_all_m"] * scale_rows
                totals_micro["traj_mse_all"] += float(traj_mse_point.mean()) * scale_rows
            for key, value in totals_micro.items():
                totals[key] += value
            if mu_den > 0.0:
                totals["mu_ds"] += mu_num / mu_den
            # micro 累积结束 → 一次梯度裁剪 + optimizer 步进（等价于宏 batch 一次更新）
            backward_started = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], float(config.grad_clip)
            )
            optimizer.step()
            backward_seconds += time.perf_counter() - backward_started
            batches += 1
        metrics["batches"] += batches
        divisor = max(1, batches)
        epoch_update: Dict[str, Any] = {
            "bc_loss": totals["loss"] / divisor,
            "bc_traj_loss": totals["traj"] / divisor,
            "bc_action_loss": totals["action"] / divisor,
            # lane U1：MoE 负载均衡 aux（Switch 式；phase 2）
            "bc_load_balance_loss": totals["load_balance"] / divisor,
            # 轨迹度量（单位见 docstring）：bc_traj_loss 是加权损失项（含 traj_weight）；
            # bc_traj_mse 加权 MSE(m²)；bc_traj_mae_m 加权 MAE(m)；
            # bc_traj_mae_all_m 未加权 MAE(m，含被过滤帧)；bc_traj_mse_unweighted 旧口径 alias(m²)。
            "bc_traj_mse": totals["traj_mse"] / divisor,
            "bc_traj_mae_m": totals["traj_mae"] / divisor,
            "bc_traj_mae_all_m": totals["traj_mae_all"] / divisor,
            "bc_traj_mse_unweighted": totals["traj_mse_all"] / divisor,
            "bc_action_mu_ds_mean": totals["mu_ds"] / divisor,
            "bc_batches_count": int(batches),
            "last_epoch": epoch + 1,
        }
        if epoch == start_epoch:
            # 阶段常量只记一次（监控降噪；完整值在 metrics.json phase 结果里）
            epoch_update.update(
                {
                    "epochs": int(config.epochs),
                    "frozen_params": int(metrics["frozen_params"]),
                    "trainable_params": int(metrics["trainable_params"]),
                }
            )
        epoch_update.update(_epoch_stats(epoch_mark))
        metrics.update(epoch_update)
        val_metrics: Dict[str, Any] = {}
        if val_indices is not None:
            # lane T：独立留出数据集（``--val-dir``）时在 val_dataset 上评估（物化快路径只服务主集）
            val_dataset = config.val_dataset if config.val_dataset is not None else dataset
            val_batch_source = batch_source if val_dataset is dataset else None
            model.eval()
            try:
                val_metrics = evaluate_bc(
                    model,
                    val_dataset,
                    config,
                    val_indices,
                    batch_source=val_batch_source,
                )
            finally:
                model.train()
            metrics["val"] = val_metrics
        if config.epoch_callback is not None:
            config.epoch_callback(epoch, dict(epoch_update), dict(val_metrics))
        if config.checkpoint_callback is not None:
            config.checkpoint_callback(epoch, model, optimizer, dict(val_metrics), rng)
        val_text = (
            f" val={val_metrics['bc_loss']:.4f}(traj={val_metrics['bc_traj_loss']:.4f}"
            f"/action={val_metrics['bc_action_loss']:.4f}"
            f"/load={val_metrics.get('bc_load_balance_loss', 0.0):.4f})"
            if val_metrics
            else ""
        )
        logger(
            f"[bc] phase={config.phase} epoch {epoch + 1}/{config.epochs} loss={metrics['bc_loss']:.4f} "
            f"traj={metrics['bc_traj_loss']:.4f} traj_mse_m2={metrics['bc_traj_mse']:.4f} "
            f"traj_mae_m={metrics['bc_traj_mae_m']:.3f}m "
            f"action={metrics['bc_action_loss']:.4f} "
            f"load_balance={metrics['bc_load_balance_loss']:.4f} "
            f"mu_ds={metrics['bc_action_mu_ds_mean']:.3f}m "
            f"data={data_seconds:.2f}s fwd={forward_seconds:.2f}s bwd={backward_seconds:.2f}s "
            f"it/s={batches / max(data_seconds + forward_seconds + backward_seconds, 1e-9):.2f}"
            f"{val_text}"
        )

    # ---------------------------------------------------------------- 汇总统计
    error_matrix = np.concatenate(action_err_values) if action_err_values else np.zeros(0)
    all_weights = np.concatenate(weight_values) if weight_values else np.zeros(0)
    metrics.update(weighted_stats(error_matrix, all_weights, prefix="bc_action_err_"))
    metrics["bc_action_err_slice_brake"] = {}
    metrics["bc_action_err_slice_turn"] = {}
    metrics["bc_action_err_slice_curve"] = {}
    for name, parts in slice_err_values.items():
        if not parts["err"]:
            continue
        metrics[f"bc_action_err_slice_{name}"] = weighted_stats(
            np.concatenate(parts["err"]), np.concatenate(parts["weight"]), prefix=""
        )
        metrics[f"bc_action_err_slice_{name}_weighted_mean"] = metrics[f"bc_action_err_slice_{name}"][
            "weighted_mean"
        ]
    # 逐 horizon 轨迹度量（B1 6 点端点；权重与动作误差同源，均为加权口径）：
    # - bc_traj_mse_h{k}   ：加权 MSE（m²）
    # - bc_traj_mae_h{k}_m ：加权 MAE（m）
    # - bc_traj_err_h{k}   ：legacy alias = loss_type 口径（l2→MSE(m²)，l1→MAE(m)）
    if traj_mse_values:
        traj_mse_matrix = np.concatenate(traj_mse_values)
        traj_mae_matrix = np.concatenate(traj_mae_values)
        traj_weights = np.concatenate(weight_values)
        legacy_matrix = traj_mse_matrix if config.loss_type == "l2" else traj_mae_matrix
        for k in range(traj_mse_matrix.shape[1]):
            metrics[f"bc_traj_mse_h{k + 1}"] = weighted_stats(
                traj_mse_matrix[:, k], traj_weights, prefix=""
            )["weighted_mean"]
            metrics[f"bc_traj_mae_h{k + 1}_m"] = weighted_stats(
                traj_mae_matrix[:, k], traj_weights, prefix=""
            )["weighted_mean"]
            metrics[f"bc_traj_err_h{k + 1}"] = weighted_stats(
                legacy_matrix[:, k], traj_weights, prefix=""
            )["weighted_mean"]
    # ego 末点 FDE（m，加权；lane B B2：`ego/traj/fde_m`）
    if traj_fde_values:
        metrics["bc_traj_fde_m"] = weighted_stats(
            np.concatenate(traj_fde_values), np.concatenate(weight_values), prefix=""
        )["weighted_mean"]
    # 逐标签动作误差（样本可属于多个标签；顺序与 batch 累积一致）
    if label_values and action_err_values:
        labels_used = np.concatenate(label_values)
        err_all = error_matrix
        w_all = all_weights
        for label_index, label_name in enumerate(dataset.label_names):
            if label_index >= labels_used.shape[1]:
                break
            mask = labels_used[:, label_index] > 0.5
            if not bool(np.any(mask)):
                continue
            stats = weighted_stats(err_all[mask], w_all[mask], prefix="")
            metrics[f"bc_action_err_label_{label_name}"] = stats["weighted_mean"]
            metrics[f"bc_action_err_label_{label_name}_count"] = stats["count"]
    metrics["bc_action_mu_ds_weighted_mean"] = (
        mu_ds_sum / mu_ds_weight if mu_ds_weight > 0 else float("nan")
    )
    # lane U1：MoE 负载全阶段汇总（expert_load Σ=1 / load_cv / gate_entropy；phase 1 MoE 关闭 → 占位）
    metrics.update(
        _load_summary(
            count=int(load_count),
            expert_load_sum=expert_load_sum,
            load_cv_num=float(load_cv_num),
            load_cv_den=float(load_cv_den),
            entropy_num=float(gate_entropy_num),
            entropy_den=float(gate_entropy_den),
            num_experts=num_experts_cfg,
        )
    )
    return metrics


@torch.no_grad()
@_with_safe_od_pose
def evaluate_bc(
    model: "nn.Module",
    dataset: BCDataset,
    config: BCConfig,
    indices: np.ndarray,
    *,
    batch_source: Optional[MaterializedBCDataset] = None,
) -> Dict[str, Any]:
    """**留出集**确定性前向（Stage B 每 epoch 末调用；不参与梯度/优化器）。

    指标口径与 :func:`pretrain_bc` 的训练统计同族（`loss/action/traj/load` + 关键切分）：
    权重 = ``train_weight×配平×worst/mild 行倍率``（``val_worst_flags``；None = 不缩放），
    动作误差为显式 L1（m/rad），轨迹为加权 MSE(m²)/MAE(m)；全局与逐 horizon 统计都在
    **整个留出子集**上按权重合成（非 batch 均值），``bc_traj_loss``/``bc_action_loss``/
    ``bc_load_balance_loss`` 仍然按 batch 求平均（与训练侧逐 epoch 口径一致）。
    lane U1：MoE 负载诊断（``bc_expert_load_*``/``bc_load_cv``/``bc_gate_entropy``）为
    val 孪生（行数加权；phase 1 MoE 关闭 → 缺省）。
    返回：

    - 扁平标量：``bc_loss``/``bc_traj_*``/``bc_action_*``/``bc_action_mu_*``/``bc_load_*``；
    - ``per_horizon``：``h{k} -> {traj_mse_m2, traj_mae_m, traj_err}``（加权口径）；
    - ``slices`` / ``labels``：``{name: weighted_stats(...)}``（含 ``count``/``weight``）。

    调用方负责 train/eval 模式切换（本函数把模型置为 ``eval`` 且不恢复）；
    空 ``indices`` 返回 ``{}``。
    """
    _require_torch()
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return {}
    device = torch.device(resolve_device(config.device))
    model.to(device).eval()
    macro_size = max(1, int(config.batch_size))
    on_curve_index = (
        dataset.label_names.index("on_curve") if "on_curve" in dataset.label_names else None
    )
    traj_mse_values: List[np.ndarray] = []
    traj_mae_values: List[np.ndarray] = []
    traj_fde_values: List[np.ndarray] = []
    weight_values: List[np.ndarray] = []
    action_err_values: List[np.ndarray] = []
    label_values: List[np.ndarray] = []
    slice_err_values: Dict[str, Dict[str, List[np.ndarray]]] = {
        "brake": {"err": [], "weight": []},
        "turn": {"err": [], "weight": []},
        "curve": {"err": [], "weight": []},
    }
    mu_ds_num, mu_ds_den = 0.0, 0.0
    mu_ds_batch_sum = 0.0
    batch_count = 0
    batch_traj_loss = batch_action_loss = batch_load_loss = 0.0
    # lane U1：MoE 负载 val 孪生累计（行数加权）
    worst_all = (
        None
        if config.val_worst_flags is None
        else np.asarray(config.val_worst_flags, dtype=np.float32).reshape(-1)
    )
    num_experts_cfg = int(getattr(model.plan_head.moe, "num_experts", 0) or 0)
    expert_load_sum: Optional[np.ndarray] = None
    load_cv_num, load_cv_den = 0.0, 0.0
    gate_entropy_num, gate_entropy_den = 0.0, 0.0
    load_count = 0

    for start in range(0, idx.size, macro_size):
        batch_indices = idx[start : start + macro_size]
        if batch_source is not None:
            obs_np = batch_source.obs_batch(batch_indices)
            targets_np = batch_source.targets_batch(batch_indices)
        else:
            obs_np = dataset.build_obs_batch(batch_indices)
            targets_np = dataset.targets(batch_indices)
        obs = _to_device_obs(obs_np, device, pin=batch_source is not None)
        targets = to_device_tensors(targets_np, device, pin=batch_source is not None)
        frame_weight = targets["train_weight"].to(device=device, dtype=torch.float32)
        # lane U1：worst/mild 行权重（留出集不缩放时 worst_all=None → 倍率 1.0）
        worst_np = worst_flags_for_rows(worst_all, batch_indices)
        if worst_np is None:
            row_scale_t = torch.ones_like(frame_weight)
        else:
            row_scale_t = torch.as_tensor(
                row_scale_from_worst(
                    worst_np, hard_weight=config.hard_weight, mild_weight=config.mild_weight
                ),
                dtype=torch.float32,
                device=device,
            )
        specific_weight = frame_weight * row_scale_t
        out = model(obs, rollout=True, world_model=False, wm_detach=config.wm_detach)
        traj_pred = out.get("traj_xy")
        if traj_pred is None:
            raise KeyError("模型 forward 缺少 'traj_xy'（契约见 p2-contract §2）")
        batch_count += 1
        diff = traj_pred - targets["traj6"]
        traj_mse_point = (diff ** 2).mean(dim=-1)
        traj_mae_point = diff.abs().mean(dim=-1)
        traj_mse_values.append(traj_mse_point.detach().double().cpu().numpy())
        traj_mae_values.append(traj_mae_point.detach().double().cpu().numpy())
        traj_fde_values.append(
            torch.linalg.norm(diff[:, -1, :], dim=-1).detach().double().cpu().numpy()
        )
        sample_weight = specific_weight.detach().double().cpu().numpy()
        weight_values.append(sample_weight)
        batch_weight_sum = float(specific_weight.sum().clamp(min=1e-8))
        batch_traj_loss += config.traj_weight * (
            float((traj_mse_point * specific_weight.reshape(-1, 1)).sum()) / batch_weight_sum
            if config.loss_type == "l2"
            else float((traj_mae_point * specific_weight.reshape(-1, 1)).sum()) / batch_weight_sum
        )
        action_pred = out.get("action_mu")
        target_action = targets["action"][:, 0, :]
        action_ok = action_pred is not None and tuple(action_pred.shape) == tuple(target_action.shape)
        action_diff = (action_pred - target_action) if action_ok else None
        # 统计口径：逐样本 L1（与训练侧 pretrain_bc 同族；不受 loss_type 影响）
        # NOTE(2026-09-28 修复)：此前误把轨迹误差（diff=traj_pred-traj6）写进 action_err_values，
        # 导致 val 的 action_err/slices 与 traj_mae 数值雷同（train 侧口径一直正确）。
        action_err_point = (
            action_diff.abs().mean(dim=-1).detach().double().cpu().numpy()
            if action_diff is not None
            else np.full(int(traj_pred.shape[0]), np.nan, dtype=np.float64)
        )
        action_err_values.append(action_err_point)
        if action_ok:
            per_sample = (
                (action_diff ** 2).mean(dim=-1)
                if config.loss_type == "l2"
                else action_diff.abs().mean(dim=-1)
            )
            batch_action_loss += config.action_weight * float(
                (per_sample * specific_weight).sum() / specific_weight.sum().clamp(min=1e-8)
            )
            mu_ds_num += float((action_pred[:, 0] * specific_weight).sum())
            mu_ds_den += batch_weight_sum
            mu_ds_batch_sum += float((action_pred[:, 0] * specific_weight).sum()) / batch_weight_sum
            labels_np = targets["labels"].detach().double().cpu().numpy()
            label_values.append(labels_np)
            expert_action = target_action.detach().double().cpu().numpy()
            slice_masks = {
                "brake": expert_action[:, 0] < float(config.slice_brake_ds),
                "turn": np.abs(expert_action[:, 1]) >= float(config.slice_turn_dtheta),
            }
            if on_curve_index is not None:
                slice_masks["curve"] = labels_np[:, on_curve_index] > 0.5
            for name, mask in slice_masks.items():
                if bool(np.any(mask)):
                    slice_err_values[name]["err"].append(action_err_point[mask])
                    slice_err_values[name]["weight"].append(sample_weight[mask])
        # lane U1：MoE 负载 val 孪生（行数加权；与训练侧同公式）
        if out.get("expert_weights") is not None and config.moe_enabled:
            logits = out.get("router_logits")
            if logits is not None:
                stats = model.plan_head.moe.load_stats(logits)
                n_rows = int(logits.shape[0])
                load = stats["expert_load"].detach().double().cpu().numpy()
                if expert_load_sum is None or expert_load_sum.size != load.size:
                    expert_load_sum = np.zeros(load.size, dtype=np.float64)
                expert_load_sum += load * n_rows
                load_cv_num += float(stats["load_cv"]) * n_rows
                load_cv_den += n_rows
                gate_entropy_num += float(stats["gate_entropy"]) * n_rows
                gate_entropy_den += n_rows
                load_count += n_rows
        if out.get("load_balance_loss") is not None:
            batch_load_loss += float(out["load_balance_loss"])

    # ---------------------------------------------------------------- 汇总（全留出子集）
    mse_matrix = np.concatenate(traj_mse_values) if traj_mse_values else np.zeros((0, 6))
    mae_matrix = np.concatenate(traj_mae_values) if traj_mae_values else np.zeros((0, 6))
    error_matrix = np.concatenate(action_err_values) if action_err_values else np.zeros(0)
    all_weights = np.concatenate(weight_values) if weight_values else np.zeros(0)
    divisor = max(1, batch_count)
    mse_sample = mse_matrix.mean(axis=1) if mse_matrix.size else np.zeros(0)
    mae_sample = mae_matrix.mean(axis=1) if mae_matrix.size else np.zeros(0)
    weighted = weighted_stats(mse_sample, all_weights, prefix="")
    weighted_mae = weighted_stats(mae_sample, all_weights, prefix="")
    fde_sample = np.concatenate(traj_fde_values) if traj_fde_values else np.zeros(0)
    weighted_fde = weighted_stats(fde_sample, all_weights, prefix="")
    result: Dict[str, Any] = {
        "bc_loss": (
            batch_traj_loss / divisor + batch_action_loss / divisor + batch_load_loss / divisor
        ),
        "bc_traj_loss": batch_traj_loss / divisor,
        "bc_action_loss": batch_action_loss / divisor,
        # lane U1：MoE 负载均衡 aux（val/ 孪生；phase 1 MoE 关闭 → 0）
        "bc_load_balance_loss": batch_load_loss / divisor,
        "bc_traj_mse": weighted["weighted_mean"],
        "bc_traj_mae_m": weighted_mae["weighted_mean"],
        "bc_traj_fde_m": weighted_fde["weighted_mean"],
        "bc_traj_mae_all_m": float(np.mean(mae_sample)) if mae_sample.size else float("nan"),
        "bc_traj_mse_unweighted": float(np.mean(mse_sample)) if mse_sample.size else float("nan"),
        "bc_action_mu_ds_mean": mu_ds_batch_sum / divisor,
        "bc_action_mu_ds_weighted_mean": mu_ds_num / mu_ds_den if mu_ds_den > 0 else float("nan"),
    }
    result.update(weighted_stats(error_matrix, all_weights, prefix="bc_action_err_"))
    per_horizon: Dict[str, Dict[str, float]] = {}
    legacy_matrix = mse_matrix if config.loss_type == "l2" else mae_matrix
    for k in range(int(mse_matrix.shape[1])):
        per_horizon[f"h{k + 1}"] = {
            "traj_mse_m2": weighted_stats(mse_matrix[:, k], all_weights, prefix="")["weighted_mean"],
            "traj_mae_m": weighted_stats(mae_matrix[:, k], all_weights, prefix="")["weighted_mean"],
            "traj_err": weighted_stats(legacy_matrix[:, k], all_weights, prefix="")["weighted_mean"],
        }
    slices: Dict[str, Dict[str, float]] = {}
    for name, parts in slice_err_values.items():
        if parts["err"]:
            slices[name] = weighted_stats(
                np.concatenate(parts["err"]), np.concatenate(parts["weight"]), prefix=""
            )
    labels: Dict[str, Dict[str, float]] = {}
    if label_values and action_err_values:
        labels_used = np.concatenate(label_values)
        if labels_used.ndim == 2 and labels_used.shape[0] == error_matrix.shape[0]:
            for label_index, label_name in enumerate(dataset.label_names):
                if label_index >= labels_used.shape[1]:
                    break
                mask = labels_used[:, label_index] > 0.5
                if not bool(np.any(mask)):
                    continue
                labels[label_name] = weighted_stats(error_matrix[mask], all_weights[mask], prefix="")
    result["per_horizon"] = per_horizon
    result["slices"] = slices
    result["labels"] = labels
    result.update(
        _load_summary(
            count=int(load_count),
            expert_load_sum=expert_load_sum,
            load_cv_num=float(load_cv_num),
            load_cv_den=float(load_cv_den),
            entropy_num=float(gate_entropy_num),
            entropy_den=float(gate_entropy_den),
            num_experts=num_experts_cfg,
        )
    )
    return result


# --------------------------------------------------------------------------- #
# stage B phase 3（迭代恢复训练；lane P3-B）
# --------------------------------------------------------------------------- #

#: phase 3 **上游监督项**（WM 教师强制 ``od/ld/presence/entry`` + plan head ``ego_next``）。
#: ``freeze_mode="specific_only"`` 下主干/WM 冻结 → 无梯度（规格：损失自动降级为 0，见
#: :func:`phase3_effective_config`）；权重全 0 时 :func:`_phase3_loss_terms` 跳过教师强制前向。
PHASE3_WM_LOSS_KEYS: Tuple[str, ...] = ("ego_next", "od", "ld", "presence", "entry")

#: phase 3 损失/监控所需的目标键（future_fn/物化源必须全部提供）。
PHASE3_FUTURE_KEYS: Tuple[str, ...] = (
    "od_fut",
    "ld_fut",
    "od_mask",
    "ld_mask",
    "wm_valid",
    "ego_fut",
    "presence_target",
    "entry_target",
    "action_chain",
    "action_chain_valid",
)


def _phase3_term_denominators(
    future: Mapping[str, np.ndarray], frame_weight_np: np.ndarray
) -> Dict[str, float]:
    """phase 3 各损失项的**宏/micro 权重分母**（纯数据计算，无前向）。

    与训练损失函数的分母口径一一对应：

    - ``action``：``Σ w``（首步动作）；
    - ``action_chain``：``Σ w·chain_valid[:,1:]``（第 2..6 步；第 1 步由 action 项监督）；
    - ``step``：``Σ w·wm_valid``（ego_next / 逐帧项）；
    - ``od``/``ld``：``Σ w·wm_valid·mask``（槽位级）；
    - ``presence``：``Σ w·wm_valid × slots``（BCE 分母 = 帧权重和 × 槽位数）。
    """
    valid = np.asarray(future["wm_valid"], dtype=np.float64)
    weight = np.asarray(frame_weight_np, dtype=np.float64).reshape(-1, 1)
    step_weight = weight * valid
    od_mask = np.asarray(future["od_mask"], dtype=np.float64)
    ld_mask = np.asarray(future["ld_mask"], dtype=np.float64)
    chain_valid = np.asarray(future["action_chain_valid"], dtype=np.float64)
    slots = int(od_mask.shape[-1]) if od_mask.ndim >= 2 else 0
    return {
        "action": float(weight.sum()),
        "action_chain": float((weight * chain_valid[:, 1:]).sum()),
        "step": float(step_weight.sum()),
        "od": float((step_weight[:, :, None] * od_mask).sum()),
        "ld": float((step_weight[:, :, None] * ld_mask).sum()),
        "presence": float(step_weight.sum()) * float(slots),
    }


def _phase3_scale_factors(
    micro: Mapping[str, float], macro: Mapping[str, float]
) -> Dict[str, float]:
    """micro → 宏口径精确缩放因子（``Σ_m s_m·L_m = L_macro``；宏分母 0 → 该项恒 0）。"""
    return {
        key: (float(micro[key]) / float(macro[key]) if float(macro[key]) > 0.0 else 0.0)
        for key in macro
    }


def _phase3_future_tensors(
    future_np: Mapping[str, np.ndarray], device: Any, *, pin: bool = False
) -> Dict[str, "torch.Tensor"]:
    """future 目标 numpy → device 张量（只转损失所需键；float32）。"""
    _require_torch()
    return {
        key: to_device_tensor(np.asarray(future_np[key]), device, dtype=torch.float32, pin=pin)
        for key in PHASE3_FUTURE_KEYS
        if key in future_np
    }


def phase3_effective_config(config: Phase3Config) -> "tuple[Phase3Config, Tuple[str, ...]]":
    """冻结降级的**有效损失配置**（lane P3-I）。

    ``freeze_mode="specific_only"`` 时上层监督项（:data:`PHASE3_WM_LOSS_KEYS`：WM 教师强制
    ``od/ld/presence/entry`` + ``ego_next``）在冻结主干下无梯度 → 权重强制置 0（规格：自动
    降级；保留 action/action_chain/load_balance）。返回 ``(effective_config, 被置零键)``；
    ``freeze_mode="all"`` 原样返回 ``(config, ())``。
    """
    if config.freeze_mode != "specific_only":
        return config, ()
    zeroed = PHASE3_WM_LOSS_KEYS
    return replace(config, **{f"{key}_weight": 0.0 for key in zeroed}), zeroed


def _phase3_loss_terms(
    model: "nn.Module",
    obs: Mapping[str, "torch.Tensor"],
    future_t: Mapping[str, "torch.Tensor"],
    targets: Mapping[str, "torch.Tensor"],
    frame_weight: "torch.Tensor",
    config: Phase3Config,
) -> Dict[str, Any]:
    """一次前向 + phase 3 全部损失项（train 侧按 micro 调用；eval 侧也可复用）。

    返回 dict：各损失项标量（含权重，**未做梯度累积缩放**）、``total``、``out``（rollout
    原始输出，供 traj 监控/负载统计）、``od_per_horizon``/``action_chain_per_step``。
    第三层 WM 监督走教师强制（``wm_teacher_forcing_predictions``），action 条件 =
    rollout ``plan`` 的 **detach** 动作链（policy 只由 ①/①b 直接监督）。

    :data:`PHASE3_WM_LOSS_KEYS` 权重全 0（如 ``specific_only`` 降级后）→ **跳过教师强制前向**
    （省一次 WM 编码/推演），对应项恒 0（不产生梯度也不占显存）。
    """
    import torch
    import torch.nn.functional as F

    out = model(obs, rollout=True, world_model=False)
    plan = out.get("plan")
    if plan is None:
        raise KeyError("phase 3 需要模型 forward(rollout=True) 输出 'plan'")
    action_pred = out.get("action_mu")
    target_action = targets["action"][:, 0, :]
    device = frame_weight.device
    if action_pred is not None and tuple(action_pred.shape) == tuple(target_action.shape):
        diff = action_pred - target_action
        per_sample = (diff ** 2).mean(dim=-1) if config.loss_type == "l2" else diff.abs().mean(dim=-1)
        action_loss = (per_sample * frame_weight).sum() / frame_weight.sum().clamp(min=1e-8)
        action_err = diff.abs().mean(dim=-1).detach()
    else:
        action_loss = torch.zeros((), device=device)
        action_err = torch.zeros_like(frame_weight)
    chain_loss, _ = weighted_action_chain_loss(
        plan,
        future_t["action_chain"],
        future_t["action_chain_valid"],
        frame_weight=frame_weight,
        loss_type=config.loss_type,
    )
    wm_active = any(
        float(getattr(config, f"{key}_weight")) > 0.0 for key in PHASE3_WM_LOSS_KEYS
    )
    if wm_active:
        predictions = wm_teacher_forcing_predictions(model, obs, future_t, plan.detach())
        od_target = model.st_gnn.od_state_from_features(future_t["od_fut"])
        od_loss, _ = weighted_od_multi_step_loss(
            predictions["od_pred"],
            od_target,
            future_t["od_mask"],
            frame_weight=frame_weight,
            valid=future_t["wm_valid"],
        )
        ld_loss, _ = weighted_ld_multi_step_loss(
            predictions["ld_pred"],
            future_t["ld_fut"][..., :4],
            future_t["ld_mask"],
            frame_weight=frame_weight,
            valid=future_t["wm_valid"],
        )
        # ② plan head ego_next：目标 = 未来 ego 特征前 H6 维；尾部按 wm_valid mask
        ego_next_pred = predictions["ego_next_pred"]
        ego_target = future_t["ego_fut"][..., : int(ego_next_pred.shape[-1])]
        horizon = min(int(ego_target.shape[1]), int(ego_next_pred.shape[1]))
        error = F.smooth_l1_loss(
            ego_next_pred[:, :horizon] - ego_target[:, :horizon],
            torch.zeros_like(ego_target[:, :horizon]),
            beta=1.0,
            reduction="none",
        ).mean(dim=-1)
        step_weight = frame_weight.reshape(-1, 1) * future_t["wm_valid"][:, :horizon]
        ego_loss = (error * step_weight).sum() / step_weight.sum().clamp(min=1e-8)
        # ③ presence/entry BCE（id 轴目标；帧权重 = w×wm_valid）；AUC 不进 phase 3 口径
        presence_terms = presence_entry_loss(
            predictions["presence_pred"],
            predictions["entry_pred"],
            future_t["presence_target"],
            future_t["entry_target"],
            frame_weight=step_weight,
            collect={},
        )
    else:
        zero = torch.zeros((), device=device)
        od_loss = ld_loss = ego_loss = zero
        presence_terms = {"presence": zero, "entry": zero}
    load_loss = out.get("load_balance_loss")
    if load_loss is None:
        load_loss = torch.zeros((), device=device)
    terms: Dict[str, Any] = {
        "action": config.action_weight * action_loss,
        "action_chain": config.action_chain_weight * chain_loss,
        "ego_next": config.ego_next_weight * ego_loss,
        "od": config.od_weight * od_loss,
        "ld": config.ld_weight * ld_loss,
        "presence": config.presence_weight * presence_terms["presence"],
        "entry": config.entry_weight * presence_terms["entry"],
        "load_balance": load_loss,
        # 监控口径（不进梯度）：逐样本首步动作 L1
        "action_err": action_err,
        "out": out,
    }
    return terms


def _phase3_traj_metrics(
    out: Mapping[str, Any], targets: Mapping[str, "torch.Tensor"], frame_weight: "torch.Tensor"
) -> Dict[str, "torch.Tensor"]:
    """traj 监控（gate 丢失：不进损失）：加权 MSE/MAE（m/m²）与末点 FDE（m）。"""
    import torch

    diff = out["traj_xy"] - targets["traj6"]
    weight = frame_weight.reshape(-1)
    denominator = weight.sum().clamp(min=1e-8)
    mse_point = (diff ** 2).mean(dim=-1)
    mae_point = diff.abs().mean(dim=-1)
    return {
        "traj_mse": (mse_point.mean(dim=-1) * weight).sum() / denominator,
        "traj_mae": (mae_point.mean(dim=-1) * weight).sum() / denominator,
        "traj_fde": (torch.linalg.norm(diff[:, -1, :], dim=-1) * weight).sum() / denominator,
    }


@_with_safe_od_pose
def pretrain_bc_phase3(
    model: "nn.Module",
    dataset: BCDataset,
    config: Phase3Config,
    *,
    logger: Callable[[str], None] = print,
    batch_source: Optional[MaterializedBCDataset] = None,
    future_source: Optional[Any] = None,
    future_fn: Optional[Callable[[np.ndarray, Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]] = None,
    val_future_fn: Optional[Callable[[np.ndarray, Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """stage B phase 3 训练循环（迭代恢复训练；lane P3-B / P3-I）。

    与 :func:`pretrain_bc` 的差异：

    - LR 分组（base/specific，见 :func:`build_phase3_optimizer`）：``freeze_mode="all"`` 时
      全参数可训（``freeze_prefixes`` 为空 = 旧行为）；``"specific_only"`` 时只留 specific 组；
    - 数据 = 单一 dagger 目录（或 stages 层合并 5k 锚后的行）；无 worst/mild 权重、无 mining
      （权重 = ``train_weight``）；
    - 冻结配方 ``freeze_mode``（lane P3-I）：``all`` = 全参数解冻（旧行为）/
      ``specific_only`` = 只训 experts/router/residual_scale，且 :func:`phase3_effective_config`
      把上游监督项（WM ``od``/``ld``/``presence``/``entry`` + ``ego_next``）权重置 0（自动降级）；
    - 损失 = 首步动作 + **动作链多步** + plan head ``ego_next`` + WM ``od``/``ld``
      （教师强制）+ ``presence``/``entry`` BCE + MoE 负载均衡；``traj`` 只监控；
    - ``future_source``（物化）/``future_fn``（逐 batch）必须提供
      :data:`PHASE3_FUTURE_KEYS` 全部键。

    ``batch_source``/``future_source`` 为物化快路径（只做切片 + H2D），语义与逐 batch
    路径一致；``micro_batch_size`` 时按宏 batch 口径精确缩放各损失项。
    ``epoch_callback``/``checkpoint_callback``/resume 语义同 :func:`pretrain_bc`。
    """
    _require_torch()
    order_config = load_supervised_labels()
    if tuple(dataset.label_names) != order_config:
        raise ValueError(
            f"数据集标签顺序 {tuple(dataset.label_names)} 与 config/model.yaml 的 {order_config} 不一致（固定顺序契约）"
        )
    if future_source is None and future_fn is None:
        raise ValueError("pretrain_bc_phase3 需要 future_source 或 future_fn（未来目标/动作链）")
    device = torch.device(resolve_device(config.device))
    model.to(device).train()
    # lane P3-I：冻结降级（effective_config 的损失权重已置 0；freeze_mode=all 时原样）
    effective_config, frozen_loss_keys = phase3_effective_config(config)
    frozen = apply_freeze_prefixes(model, config.freeze_prefixes)
    optimizer = build_phase3_optimizer(
        model,
        config.lr,
        base_scale=config.lr_base_scale,
        specific_scale=config.lr_specific_scale,
        freeze_mode=config.freeze_mode,
    )
    if config.optimizer_state:
        load_optimizer_state(optimizer, config.optimizer_state, logger=logger)
        move_optimizer_state_to_device(optimizer)
    if config.train_indices is not None:
        indices = np.asarray(config.train_indices, dtype=np.int64).reshape(-1)
        if indices.size == 0:
            raise ValueError("config.train_indices 为空 → 无训练样本")
    else:
        indices = dataset.sample_indices()
    val_indices = (
        None if config.val_indices is None else np.asarray(config.val_indices, dtype=np.int64).reshape(-1)
    )
    if val_indices is not None and val_indices.size == 0:
        val_indices = None
    start_epoch = max(0, min(int(config.start_epoch or 0), int(config.epochs)))
    if start_epoch:
        logger(f"[bc3] resume：跳过已完成 {start_epoch}/{int(config.epochs)} 个 epoch（phase={config.phase}）")
    rng = np.random.default_rng(config.seed)
    if start_epoch:
        if config.rng_state:
            restore_rng_state(config.rng_state, numpy_generator=rng)
        elif config.shuffle:
            for _ in range(start_epoch):
                rng.shuffle(indices.copy())
    model.set_moe(enabled=bool(config.moe_enabled), load_balance_coef=float(config.load_balance_coef))
    num_experts_cfg = int(getattr(model.plan_head.moe, "num_experts", 0) or 0)
    metrics: Dict[str, Any] = {
        "epochs": int(config.epochs),
        "batches": 0,
        "phase": str(config.phase),
        "freeze_mode": str(config.freeze_mode),
        "frozen_loss_keys": list(frozen_loss_keys),
        "frozen_params": len(frozen),
        "trainable_params": sum(1 for parameter in model.parameters() if parameter.requires_grad),
        "trainable_param_groups": [str(group.get("name")) for group in optimizer.param_groups],
    }
    logger(
        f"[bc3] phase={config.phase} freeze={config.freeze_mode}：frozen={len(frozen)} · "
        f"可训={metrics['trainable_params']} · "
        f"lr={float(config.lr):.2e}（base×{config.lr_base_scale} · specific×{config.lr_specific_scale}）· "
        f"moe={config.moe_enabled} · load_balance_coef={config.load_balance_coef}"
        + (
            f" · 损失降级置 0：{', '.join(frozen_loss_keys)}（冻结下无梯度）"
            if frozen_loss_keys
            else ""
        )
    )
    loss_keys = ("action", "action_chain", "ego_next", "od", "ld", "presence", "entry", "load_balance")
    macro_size = max(1, int(config.batch_size))
    micro_size = (
        macro_size
        if config.micro_batch_size is None
        else max(1, min(macro_size, int(config.micro_batch_size)))
    )
    if micro_size < macro_size:
        logger(f"[bc3] 梯度累积：宏 batch={macro_size} · micro batch={micro_size}（损失按宏口径精确缩放）")
    weight_values: List[np.ndarray] = []
    action_err_values: List[np.ndarray] = []
    traj_mse_values: List[np.ndarray] = []
    traj_mae_values: List[np.ndarray] = []
    traj_fde_values: List[np.ndarray] = []

    def _future_batch(batch_indices: np.ndarray, obs_np: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        if future_source is not None:
            return future_source.batch(batch_indices)
        return future_fn(batch_indices, obs_np)  # type: ignore[misc]

    for epoch in range(start_epoch, int(config.epochs)):
        order = indices.copy()
        if config.shuffle:
            rng.shuffle(order)
        weight_values.clear()
        action_err_values.clear()
        traj_mse_values.clear()
        traj_mae_values.clear()
        traj_fde_values.clear()
        totals: Dict[str, float] = {key: 0.0 for key in loss_keys}
        totals.update({"total": 0.0, "traj_mse": 0.0, "traj_mae": 0.0, "traj_fde": 0.0, "mu_ds_num": 0.0, "mu_ds_den": 0.0})
        expert_load_sum: Optional[np.ndarray] = None
        load_cv_num, load_cv_den = 0.0, 0.0
        gate_entropy_num, gate_entropy_den = 0.0, 0.0
        load_count = 0
        batches = 0
        data_seconds = forward_seconds = backward_seconds = 0.0
        for start in range(0, len(order), macro_size):
            if config.max_batches is not None and batches >= int(config.max_batches):
                break
            batch_indices = order[start : start + macro_size]
            n_macro = int(batch_indices.shape[0])
            data_started = time.perf_counter()
            if batch_source is not None:
                obs_np = batch_source.obs_batch(batch_indices)
                targets_np = batch_source.targets_batch(batch_indices)
            else:
                obs_np = dataset.build_obs_batch(batch_indices)
                targets_np = dataset.targets(batch_indices)
            future_np = _future_batch(batch_indices, obs_np)
            frame_weight_np = np.asarray(targets_np["train_weight"], dtype=np.float64).reshape(-1)
            macro_den = _phase3_term_denominators(future_np, frame_weight_np)
            macro_den["rows"] = float(n_macro)
            data_seconds += time.perf_counter() - data_started
            optimizer.zero_grad(set_to_none=True)
            macro_totals = {key: 0.0 for key in list(totals)}
            for m_start in range(0, n_macro, micro_size):
                m_lo, m_hi = m_start, min(m_start + micro_size, n_macro)
                data_started = time.perf_counter()
                if micro_size >= n_macro:
                    obs = _to_device_obs(obs_np, device, pin=batch_source is not None)
                    targets = to_device_tensors(
                        {key: targets_np[key] for key in ("action", "traj6", "train_weight")},
                        device,
                        dtype=torch.float32,
                        pin=batch_source is not None,
                    )
                    future_t = _phase3_future_tensors(
                        future_np, device, pin=batch_source is not None
                    )
                    den_micro = dict(macro_den)
                    den_macro = dict(macro_den)
                else:
                    obs = _to_device_obs({key: value[m_lo:m_hi] for key, value in obs_np.items()}, device)
                    targets = to_device_tensors(
                        {key: targets_np[key][m_lo:m_hi] for key in ("action", "traj6", "train_weight")},
                        device,
                        dtype=torch.float32,
                    )
                    future_t = _phase3_future_tensors(
                        {key: value[m_lo:m_hi] for key, value in future_np.items()}, device
                    )
                    den_micro = _phase3_term_denominators(
                        {key: value[m_lo:m_hi] for key, value in future_np.items()},
                        frame_weight_np[m_lo:m_hi],
                    )
                    den_micro["rows"] = float(m_hi - m_lo)
                    den_macro = dict(macro_den)
                frame_weight = targets["train_weight"].to(device=device, dtype=torch.float32)
                data_seconds += time.perf_counter() - data_started
                forward_started = time.perf_counter()
                terms = _phase3_loss_terms(model, obs, future_t, targets, frame_weight, effective_config)
                components = {key: terms[key] for key in loss_keys}
                if micro_size < n_macro:
                    scales = _phase3_scale_factors(den_micro, den_macro)
                    for key in loss_keys:
                        # 负载均衡 aux 是逐行均值项 → 按 n_micro/n_macro 缩放（BC 同口径）；
                        # ego_next/entry 分别用 step/presence 分母（见 _term_key）
                        scale_key = "rows" if key == "load_balance" else _term_key(key)
                        components[key] = components[key] * scales[scale_key]
                total = (
                    components["action"]
                    + components["action_chain"]
                    + components["ego_next"]
                    + components["od"]
                    + components["ld"]
                    + components["presence"]
                    + components["entry"]
                    + components["load_balance"]
                )
                traj_metrics = _phase3_traj_metrics(terms["out"], targets, frame_weight)
                forward_seconds += time.perf_counter() - forward_started
                backward_started = time.perf_counter()
                total.backward()
                backward_seconds += time.perf_counter() - backward_started
                macro_totals["total"] += float(total.detach())
                for key in loss_keys:
                    macro_totals[key] += float(components[key].detach())
                weight_micro = float(den_micro["action"])
                macro_totals["traj_mse"] += float(traj_metrics["traj_mse"]) * weight_micro
                macro_totals["traj_mae"] += float(traj_metrics["traj_mae"]) * weight_micro
                macro_totals["traj_fde"] += float(traj_metrics["traj_fde"]) * weight_micro
                macro_totals["mu_ds_num"] += float(
                    (terms["out"]["action_mu"][:, 0] * frame_weight).sum()
                )
                macro_totals["mu_ds_den"] += weight_micro
                with torch.no_grad():
                    diff = terms["out"]["traj_xy"] - targets["traj6"]
                    traj_mse_values.append((diff ** 2).mean(dim=-1).detach().double().cpu().numpy())
                    traj_mae_values.append(diff.abs().mean(dim=-1).detach().double().cpu().numpy())
                    traj_fde_values.append(
                        torch.linalg.norm(diff[:, -1, :], dim=-1).detach().double().cpu().numpy()
                    )
                    action_err_values.append(terms["action_err"].double().cpu().numpy())
                    weight_values.append(frame_weight.detach().double().cpu().numpy())
                    if terms["out"].get("expert_weights") is not None and config.moe_enabled:
                        logits = terms["out"].get("router_logits")
                        if logits is not None:
                            stats = model.plan_head.moe.load_stats(logits)
                            load = stats["expert_load"].detach().double().cpu().numpy()
                            if expert_load_sum is None or expert_load_sum.size != load.size:
                                expert_load_sum = np.zeros(load.size, dtype=np.float64)
                            n_rows = int(logits.shape[0])
                            expert_load_sum += load * n_rows
                            load_cv_num += float(stats["load_cv"]) * n_rows
                            load_cv_den += n_rows
                            gate_entropy_num += float(stats["gate_entropy"]) * n_rows
                            gate_entropy_den += n_rows
                            load_count += n_rows
            for key, value in macro_totals.items():
                totals[key] += value
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(config.grad_clip),
            )
            optimizer.step()
            batches += 1
        metrics["batches"] += batches
        divisor = max(1, batches)
        weight_total = sum(float(np.sum(item)) for item in weight_values)
        weight_denominator = max(weight_total, 1e-12)
        error_matrix = np.concatenate(action_err_values) if action_err_values else np.zeros(0)
        weight_matrix = np.concatenate(weight_values) if weight_values else np.zeros(0)
        epoch_update: Dict[str, Any] = {
            "bc_loss": totals["total"] / divisor,
            "bc_action_loss": totals["action"] / divisor,
            "bc_action_chain_loss": totals["action_chain"] / divisor,
            "bc_ego_next_loss": totals["ego_next"] / divisor,
            "bc_od_loss": totals["od"] / divisor,
            "bc_ld_loss": totals["ld"] / divisor,
            "bc_presence_loss": totals["presence"] / divisor,
            "bc_entry_loss": totals["entry"] / divisor,
            "bc_load_balance_loss": totals["load_balance"] / divisor,
            # traj 监控（不进损失；loss_type 口径的加权误差 + MSE/MAE/FDE）
            "bc_traj_loss": (
                totals["traj_mse" if config.loss_type == "l2" else "traj_mae"] / weight_denominator
            ),
            "bc_traj_mse": totals["traj_mse"] / weight_denominator,
            "bc_traj_mae_m": totals["traj_mae"] / weight_denominator,
            "bc_traj_fde_m": totals["traj_fde"] / weight_denominator,
            "bc_action_err_weighted_mean": (
                float((error_matrix * weight_matrix).sum() / max(weight_total, 1e-12))
                if error_matrix.size
                else float("nan")
            ),
            "bc_action_err_count": float(error_matrix.size),
            "bc_action_err_weight": float(weight_total),
            "bc_action_mu_ds_weighted_mean": (
                totals["mu_ds_num"] / max(totals["mu_ds_den"], 1e-12)
            ),
            "bc_batches_count": int(batches),
            "last_epoch": epoch + 1,
            "epoch_data_seconds": data_seconds,
            "epoch_forward_seconds": forward_seconds,
            "epoch_backward_seconds": backward_seconds,
        }
        if traj_mse_values and weight_values and len(traj_mse_values[0]) == len(weight_values[0]):
            mse_matrix = np.concatenate(traj_mse_values)
            mae_matrix = np.concatenate(traj_mae_values)
            weights_all = np.concatenate(weight_values)
            legacy_matrix = mse_matrix if config.loss_type == "l2" else mae_matrix
            for k in range(mse_matrix.shape[1]):
                epoch_update[f"bc_traj_mse_h{k + 1}"] = weighted_stats(
                    mse_matrix[:, k], weights_all, prefix=""
                )["weighted_mean"]
                epoch_update[f"bc_traj_mae_h{k + 1}_m"] = weighted_stats(
                    mae_matrix[:, k], weights_all, prefix=""
                )["weighted_mean"]
                epoch_update[f"bc_traj_err_h{k + 1}"] = weighted_stats(
                    legacy_matrix[:, k], weights_all, prefix=""
                )["weighted_mean"]
        epoch_update.update(
            _load_summary(
                count=int(load_count),
                expert_load_sum=expert_load_sum,
                load_cv_num=float(load_cv_num),
                load_cv_den=float(load_cv_den),
                entropy_num=float(gate_entropy_num),
                entropy_den=float(gate_entropy_den),
                num_experts=num_experts_cfg,
            )
        )
        if epoch == start_epoch:
            epoch_update.update(
                {
                    "epochs": int(config.epochs),
                    "frozen_params": int(metrics["frozen_params"]),
                    "trainable_params": int(metrics["trainable_params"]),
                }
            )
        metrics.update(epoch_update)
        val_metrics: Dict[str, Any] = {}
        if val_indices is not None:
            val_dataset = config.val_dataset if config.val_dataset is not None else dataset
            val_batch_source = batch_source if val_dataset is dataset else None
            val_future_source = future_source if val_dataset is dataset else None
            val_fn = future_fn if val_dataset is dataset else (val_future_fn or future_fn)
            model.eval()
            try:
                val_metrics = evaluate_bc_phase3(
                    model,
                    val_dataset,
                    config,
                    val_indices,
                    batch_source=val_batch_source,
                    future_source=val_future_source,
                    future_fn=val_fn,
                )
            finally:
                model.train()
            metrics["val"] = val_metrics
        if config.epoch_callback is not None:
            config.epoch_callback(epoch, dict(epoch_update), dict(val_metrics))
        if config.checkpoint_callback is not None:
            config.checkpoint_callback(epoch, model, optimizer, dict(val_metrics), rng)
        val_text = (
            f" val={val_metrics['bc_loss']:.4f}(od={val_metrics.get('bc_od_loss', float('nan')):.4f}"
            f"/ld={val_metrics.get('bc_ld_loss', float('nan')):.4f})"
            if val_metrics
            else ""
        )
        logger(
            f"[bc3] phase={config.phase} epoch {epoch + 1}/{config.epochs} "
            f"loss={metrics['bc_loss']:.4f} action={metrics['bc_action_loss']:.4f} "
            f"chain={metrics['bc_action_chain_loss']:.4f} ego_next={metrics['bc_ego_next_loss']:.4f} "
            f"od={metrics['bc_od_loss']:.4f} ld={metrics['bc_ld_loss']:.4f} "
            f"presence={metrics['bc_presence_loss']:.4f} entry={metrics['bc_entry_loss']:.4f} "
            f"traj(monitor)={metrics['bc_traj_mae_m']:.3f}m "
            f"data={data_seconds:.2f}s fwd={forward_seconds:.2f}s bwd={backward_seconds:.2f}s"
            f"{val_text}"
        )
    return metrics


@torch.no_grad()
@_with_safe_od_pose
def evaluate_bc_phase3(
    model: "nn.Module",
    dataset: BCDataset,
    config: Phase3Config,
    indices: np.ndarray,
    *,
    batch_source: Optional[MaterializedBCDataset] = None,
    future_source: Optional[Any] = None,
    future_fn: Optional[Callable[[np.ndarray, Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """phase 3 留出集确定性前向（不参与梯度/优化器；口径与训练侧同族）。

    损失项在**整个留出子集**上按权重合成（``Σnum/Σden``，非 batch 均值）；``traj`` 只做
    监控（``bc_traj_mse/mae_m/fde_m`` + 逐 horizon）；动作误差为显式 L1（m/rad）。
    空 ``indices`` 返回 ``{}``；调用方负责 train/eval 模式切换。
    """
    import torch

    _require_torch()
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return {}
    if future_source is None and future_fn is None:
        raise ValueError("evaluate_bc_phase3 需要 future_source 或 future_fn（未来目标/动作链）")
    device = torch.device(resolve_device(config.device))
    model.to(device).eval()
    # lane P3-I：留出口径与训练侧同族 —— specific_only 降级项同样置 0
    effective_config, _ = phase3_effective_config(config)
    macro_size = max(1, int(config.batch_size))
    loss_keys = ("action", "action_chain", "ego_next", "od", "ld", "presence", "entry")
    numerator = {key: 0.0 for key in loss_keys}
    denominator = {key: 0.0 for key in loss_keys}
    load_sum, load_count = 0.0, 0
    action_err_values: List[np.ndarray] = []
    weight_values: List[np.ndarray] = []
    traj_mse_values: List[np.ndarray] = []
    traj_mae_values: List[np.ndarray] = []
    traj_fde_values: List[np.ndarray] = []
    for start in range(0, idx.size, macro_size):
        batch_indices = idx[start : start + macro_size]
        if batch_source is not None:
            obs_np = batch_source.obs_batch(batch_indices)
            targets_np = batch_source.targets_batch(batch_indices)
        else:
            obs_np = dataset.build_obs_batch(batch_indices)
            targets_np = dataset.targets(batch_indices)
        if future_source is not None:
            future_np = future_source.batch(batch_indices)
        else:
            future_np = future_fn(batch_indices, obs_np)  # type: ignore[misc]
        obs = _to_device_obs(obs_np, device)
        targets = to_device_tensors(
            {key: targets_np[key] for key in ("action", "traj6", "train_weight")},
            device,
            dtype=torch.float32,
        )
        future_t = _phase3_future_tensors(future_np, device)
        frame_weight = targets["train_weight"].to(device=device, dtype=torch.float32)
        den = _phase3_term_denominators(future_np, np.asarray(targets_np["train_weight"], dtype=np.float64))
        terms = _phase3_loss_terms(model, obs, future_t, targets, frame_weight, effective_config)
        for key in loss_keys:
            denominator[key] += float(den[_term_key(key)])
            numerator[key] += float(terms[key].detach()) * float(den[_term_key(key)])
        if terms["out"].get("load_balance_loss") is not None:
            load_sum += float(terms["out"]["load_balance_loss"].detach())
            load_count += 1
        traj_mse_values.append(
            (terms["out"]["traj_xy"] - targets["traj6"]).pow(2).mean(dim=-1).double().cpu().numpy()
        )
        traj_mae_values.append(
            (terms["out"]["traj_xy"] - targets["traj6"]).abs().mean(dim=-1).double().cpu().numpy()
        )
        traj_fde_values.append(
            torch.linalg.norm(
                (terms["out"]["traj_xy"] - targets["traj6"])[:, -1, :], dim=-1
            ).double().cpu().numpy()
        )
        action_err_values.append(terms["action_err"].double().cpu().numpy())
        weight_values.append(frame_weight.detach().double().cpu().numpy())
    # 总损失 = 各加权损失项和（含权重系数）；逐项 = Σnum/Σden
    result: Dict[str, Any] = {
        "bc_load_balance_loss": load_sum / max(load_count, 1),
        "bc_loss": 0.0,
    }
    result["bc_loss"] = sum(
        numerator[key] / max(denominator[key], 1e-12) for key in loss_keys
    ) + result["bc_load_balance_loss"]
    result.update(
        {
            "bc_action_loss": numerator["action"] / max(denominator["action"], 1e-12),
            "bc_action_chain_loss": numerator["action_chain"] / max(denominator["action_chain"], 1e-12),
            "bc_ego_next_loss": numerator["ego_next"] / max(denominator["ego_next"], 1e-12),
            "bc_od_loss": numerator["od"] / max(denominator["od"], 1e-12),
            "bc_ld_loss": numerator["ld"] / max(denominator["ld"], 1e-12),
            "bc_presence_loss": numerator["presence"] / max(denominator["presence"], 1e-12),
            "bc_entry_loss": numerator["entry"] / max(denominator["entry"], 1e-12),
        }
    )
    error_matrix = np.concatenate(action_err_values) if action_err_values else np.zeros(0)
    all_weights = np.concatenate(weight_values) if weight_values else np.zeros(0)
    weight_total = max(float(all_weights.sum()) if all_weights.size else 0.0, 1e-12)
    mse_matrix = np.concatenate(traj_mse_values) if traj_mse_values else np.zeros((0, 6))
    mae_matrix = np.concatenate(traj_mae_values) if traj_mae_values else np.zeros((0, 6))
    fde_values = np.concatenate(traj_fde_values) if traj_fde_values else np.zeros(0)
    mse_sample = mse_matrix.mean(axis=1) if mse_matrix.size else np.zeros(0)
    mae_sample = mae_matrix.mean(axis=1) if mae_matrix.size else np.zeros(0)
    legacy_sample = mse_sample if config.loss_type == "l2" else mae_sample
    result.update(
        {
            "bc_traj_loss": (
                float((legacy_sample * all_weights).sum() / weight_total) if mse_sample.size else float("nan")
            ),
            "bc_traj_mse": float((mse_sample * all_weights).sum() / weight_total) if mse_sample.size else float("nan"),
            "bc_traj_mae_m": float((mae_sample * all_weights).sum() / weight_total) if mae_sample.size else float("nan"),
            "bc_traj_fde_m": float((fde_values * all_weights).sum() / weight_total) if fde_values.size else float("nan"),
            "bc_action_err_weighted_mean": (
                float((error_matrix * all_weights).sum() / weight_total) if error_matrix.size else float("nan")
            ),
            "bc_action_err_count": float(error_matrix.size),
            "bc_action_err_weight": float(all_weights.sum()) if all_weights.size else 0.0,
        }
    )
    per_horizon: Dict[str, Dict[str, float]] = {}
    if mse_matrix.size:
        for k in range(int(mse_matrix.shape[1])):
            per_horizon[f"h{k + 1}"] = {
                "traj_mse_m2": weighted_stats(mse_matrix[:, k], all_weights, prefix="")["weighted_mean"],
                "traj_mae_m": weighted_stats(mae_matrix[:, k], all_weights, prefix="")["weighted_mean"],
                "traj_err": weighted_stats(
                    (mse_matrix if config.loss_type == "l2" else mae_matrix)[:, k],
                    all_weights,
                    prefix="",
                )["weighted_mean"],
            }
    result["per_horizon"] = per_horizon
    return result


def _term_key(loss_key: str) -> str:
    """损失项键 → :func:`_phase3_term_denominators` 的分母键。

    ``entry`` 与 ``presence`` 共用 BCE 分母（帧权重和 × 槽位数）；``ego_next`` 的分母是
    ``step``（``Σ w·wm_valid``）。
    """
    return {"entry": "presence", "ego_next": "step"}.get(loss_key, loss_key)


# --------------------------------------------------------------------------- #
# MoE 路由监控
# --------------------------------------------------------------------------- #

class RouterMonitor:
    """MoE 路由监控（负载均衡 / 熵 / 每 expert 权重、激活率与输出范数）。"""

    def __init__(self, num_experts: int = 8):
        self.num_experts = int(num_experts)
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._weight_sum = np.zeros(self.num_experts, dtype=np.float64)
        self._active_sum = np.zeros(self.num_experts, dtype=np.float64)
        self._entropy_sum = 0.0
        self._effective_sum = 0.0
        self._norm_sum = np.zeros(self.num_experts, dtype=np.float64)
        self._norm_count = 0

    @torch.no_grad() if torch is not None else (lambda fn: fn)
    def update(self, outputs: Mapping[str, Any]) -> None:
        """累计一个 batch 的路由统计（只读输入，不参与梯度）。"""
        logits = outputs.get("router_logits")
        if logits is None or not hasattr(logits, "detach"):
            return
        probs = torch.sigmoid(logits.detach()).double().cpu().numpy().reshape(-1, self.num_experts)
        if probs.shape[0] == 0:
            return
        self._count += probs.shape[0]
        self._weight_sum += probs.sum(axis=0)
        self._active_sum += (probs > 0.5).sum(axis=0)
        clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        self._entropy_sum += float(
            (-(clipped * np.log(clipped) - (1 - clipped) * np.log(1 - clipped))).sum(axis=1).mean()
        )
        self._effective_sum += float(probs.sum(axis=1).mean())
        norms = outputs.get("expert_outputs_norm")
        if norms is not None and hasattr(norms, "detach"):
            norm_array = norms.detach().double().cpu().numpy().reshape(-1, self.num_experts)
            self._norm_sum += norm_array.mean(axis=0)
            self._norm_count += 1
        elif outputs.get("expert_outputs") is not None and hasattr(outputs["expert_outputs"], "detach"):
            norm_array = outputs["expert_outputs"].detach().double().norm(dim=-1).mean(dim=0).cpu().numpy().reshape(-1)
            if norm_array.shape[0] == self.num_experts:
                self._norm_sum += norm_array
                self._norm_count += 1

    def summary(self) -> Dict[str, Any]:
        if self._count == 0:
            return {"router_count": 0}
        mean_weight = self._weight_sum / self._count
        return {
            "router_count": int(self._count),
            "router_effective_n": self._effective_sum / self._count,
            "router_mean_entropy": self._entropy_sum / self._count,
            "router_mean_weight": [float(x) for x in mean_weight],
            "router_active_rate": [float(x) for x in (self._active_sum / self._count)],
            "router_mean_output_norm": (
                [float(x) for x in (self._norm_sum / self._norm_count)] if self._norm_count else None
            ),
            "router_load_imbalance": float((mean_weight.max() - mean_weight.min()) / (mean_weight.mean() + 1e-9)),
        }


# --------------------------------------------------------------------------- #
# 动作展开（策略步 (ds,dθ) → 逐 env-step [steer,throttle]；子进程池用）
# --------------------------------------------------------------------------- #

def _arc_endpoint(pose: np.ndarray, ds: float, dtheta: float) -> np.ndarray:
    """``(ds,dθ)`` 圆弧推进（与 net.arc_step / env.tracking.interpolate 同约定）。"""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    if abs(dtheta) < 1e-9:
        return np.array([x + ds * math.cos(theta), y + ds * math.sin(theta), theta], dtype=np.float64)
    radius = ds / dtheta
    theta_new = theta + dtheta
    return np.array(
        [x + radius * (math.sin(theta_new) - math.sin(theta)), y - radius * (math.cos(theta_new) - math.cos(theta)), theta_new],
        dtype=np.float64,
    )


def expand_policy_action(
    action: np.ndarray,
    *,
    speed: float,
    dt: float = _POLICY_DT,
    hz: int = _PHYSICS_HZ,
    wheelbase: float = 2.46894,
    max_steer_rad: float = math.radians(40.0),
    accel_scale: float = 2.0,
) -> np.ndarray:
    """把策略动作 ``(ds,dθ)`` 展开成 ``(dt·hz, 2)`` 的 env-step ``[steer, throttle]``。

    一阶运动学近似（**子进程池没有 in-worker tracker 时的文档化替代**）：
    以 ``env.tracking.interpolate`` 的 30 点参考（不可用时用本地圆弧）为参考，
    第 k 子步的参考速度/航向变化率 → 自行车模型转向 + 速度误差比例油门。
    """
    seq = np.asarray(action, dtype=np.float64).reshape(1, 2)
    n_sub = max(1, int(round(dt * hz)))
    try:
        from env.tracking import interpolate as tracking_interpolate

        reference = np.asarray(tracking_interpolate(seq, dt=dt, hz=hz), dtype=np.float64)
    except Exception:  # noqa: BLE001 - env.tracking 不可用时本地圆弧
        reference = np.zeros((n_sub, 3), dtype=np.float64)
        pose = np.zeros(3, dtype=np.float64)
        for k in range(n_sub):
            pose = _arc_endpoint(pose, float(seq[0, 0]) / n_sub, float(seq[0, 1]) / n_sub)
            reference[k] = pose
    sub_dt = 1.0 / float(hz)
    out = np.zeros((n_sub, 2), dtype=np.float32)
    v_ref = float(seq[0, 0]) / dt
    omega = float(seq[0, 1]) / dt
    steer = math.atan2(omega * wheelbase, max(float(speed), 0.5)) / max_steer_rad
    for k in range(n_sub):
        if k > 0:
            prev = reference[k - 1]
            v_ref = float(np.linalg.norm(reference[k][:2] - prev[:2])) / sub_dt
            omega = float(reference[k][2] - prev[2]) / sub_dt
            steer = math.atan2(omega * wheelbase, max(float(speed), 0.5)) / max_steer_rad
        throttle = (v_ref - float(speed)) / accel_scale
        out[k] = (float(np.clip(steer, -1.0, 1.0)), float(np.clip(throttle, -1.0, 1.0)))
    return out


# --------------------------------------------------------------------------- #
# 环境池
# --------------------------------------------------------------------------- #

def _router_labels_from_env(env: Any, spec: Any, order: Sequence[str]) -> Optional[np.ndarray]:
    try:
        from env.scenario.labels import compute_step_labels

        raw = compute_step_labels(env, spec)
        return np.array([float(raw.get(name, 0.0)) for name in order], dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None


def _labels_for_env(labels_pre: Any, env_index: int) -> Optional[Any]:
    """``labels_now()`` 返回值 → 第 ``env_index`` 个 env 的标签（``(8,)`` 数组或 per-env 列表）。"""
    if labels_pre is None:
        return None
    if isinstance(labels_pre, np.ndarray):
        if labels_pre.ndim == 1:
            return labels_pre if env_index == 0 else None
        return labels_pre[env_index] if env_index < labels_pre.shape[0] else None
    if isinstance(labels_pre, (list, tuple)):
        return labels_pre[env_index] if env_index < len(labels_pre) else None
    return None


class LocalEnvPool:
    """**单进程常驻池**（MetaDrive 每进程一个 engine → 该池恰好 1 个 env）。

    ``tracker``：

    - ``exact``（阶段 A）：每个策略步 ``arm`` 一次 ``ExactTracker``，逐 0.1 s
      ``env.step`` 后 ``apply``（§1 的精确定义；观测/奖励随后驱动）；
    - ``lqr``（阶段 C）：``LqrTracker`` 经 ``engine.add_policy`` 注册，逐策略步
      ``set_reference``，物理闭环执行；
    - ``kinematic``：展开成逐子步动作的近似执行（用于 smoke/对照）。

    记录 dict 含 ``obs``（含 ``pose``/``hist_valid``）、``info``（含 ``router_labels`` 与
    ``on_*_continuous_line``）、``reward/terminated/truncated``；env 终局后池内自动 reset。
    """

    def __init__(
        self,
        specs: Sequence[Any],
        *,
        tracker: str = "exact",
        max_episode_steps: int = 600,
        traffic_density: Optional[float] = None,
        obs_config: Optional[Dict[str, Any]] = None,
        logger: Callable[[str], None] = print,
    ):
        from env.obs.builder import ObservationBuilder

        self.specs = list(specs)
        if not self.specs:
            raise ValueError("LocalEnvPool 需要至少一条 spec")
        if tracker not in ("exact", "lqr", "kinematic"):
            raise ValueError(f"未知 tracker={tracker!r}（exact/lqr/kinematic）")
        self.tracker_kind = str(tracker)
        self.max_episode_steps = int(max_episode_steps)
        self.traffic_density = traffic_density
        self.obs_config = dict(obs_config or {})
        self.logger = logger
        self.num_envs = 1
        self._builder = ObservationBuilder(self.obs_config)
        self._env: Any = None
        self._spec: Any = None
        self._index = -1
        self._steps = 0
        self._order = load_supervised_labels()
        self._tracker: Any = None
        self._tracker_policy: Any = None
        #: 最近一次 env.step 的 info（P0-2：reset 时清空，终局 record 与 reset 分离）
        self._current_info: Dict[str, Any] = {}

    # ---------------------------------------------------------------- 生命周期
    def _build(self, spec: Any) -> None:
        from env.metadrive_env import build_env

        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
        self._env = build_env(spec, traffic_density=self.traffic_density, use_render=False)
        self._spec = spec

    def _setup_episode(self) -> None:
        """reset 后重建执行器（env.reset 会清掉 ego 上的 policy 注册）。"""
        env = self._env
        self._steps = 0
        if self.tracker_kind == "exact":
            from env.tracking import ExactTracker

            self._tracker = ExactTracker()
        elif self.tracker_kind == "lqr":
            from env.tracking import LqrTracker

            self._tracker = None
            self._tracker_policy = env.engine.add_policy(
                env.agent.id, LqrTracker, env.agent, int(getattr(self._spec, "seed", 0))
            )
            if hasattr(self._tracker_policy, "reset"):
                self._tracker_policy.reset()
        else:
            self._tracker = None

    def reset(self) -> List[Dict[str, Any]]:
        """切换到下一条 spec 并 reset；返回单元素记录列表。"""
        self._index = (self._index + 1) % len(self.specs)
        spec = self.specs[self._index]
        self._build(spec)
        self._env.reset()
        self._setup_episode()
        # §8.4：新 episode 起点没有上一动作 → reserved 维保持 0
        self._env.prev_policy_action = np.zeros(2, dtype=np.float64)
        # P0-2：episode 首帧不得复用上一 episode（尤其终局）残留的 step info
        self._current_info = {}
        return [self._record(reward=0.0, terminated=False, truncated=False)]

    def _obs_now(self) -> Dict[str, Any]:
        """当前 env 状态的 obs 快照（含 ``pose``）。"""
        import numpy as np  # noqa: F811 - 局部引用保持模块顶层无 MetaDrive 依赖

        env = self._env
        obs = self._builder.build(env, self._spec)
        ego = env.agent
        obs["pose"] = np.array(
            [float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)], dtype=np.float32
        )
        return obs

    def labels_now(self) -> Optional[np.ndarray]:
        """**步前** router 标签快照（与调用侧 ``obs_current`` 同帧；标签模块不可用 → None）。

        P0-2：标签必须在动作执行/状态推进之前取，否则会配给"动作后"的观测（``p(y_{t+1}|o_t)``）。
        """
        if self._env is None or not self._order:
            return None
        return _router_labels_from_env(self._env, self._spec, self._order)

    def _record(self, *, reward: float, terminated: bool, truncated: bool) -> Dict[str, Any]:
        env = self._env
        obs = self._obs_now()
        ego = env.agent
        info = dict(self._current_info)
        info["pose"] = obs["pose"].copy()
        labels = _router_labels_from_env(env, self._spec, self._order)
        if labels is not None:
            info["router_labels"] = labels
        info["on_white_continuous_line"] = bool(getattr(ego, "on_white_continuous_line", False))
        info["on_yellow_continuous_line"] = bool(getattr(ego, "on_yellow_continuous_line", False))
        return {
            "obs": obs,
            "info": info,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }

    def step(self, actions: np.ndarray, references: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """执行一个策略步（1 个动作 ``(2,)``），返回单元素记录；终局自动 reset。

        P0-2：终局记录的 obs/info/pose 取自**终局帧**；reset 后的新 episode 首帧只经
        ``record["next_obs"]`` 返回（不再混入终局记录）。

        ``references``：``(N,6,2)`` 策略规划预览（阶段 C LQR 用）。给定且 ``tracker_kind=="lqr"``
        时，跟踪器参考 = 该 6 动作序列（30 点 / 3 s），而不是单动作（单动作会退化为
        "瞄准参考终点"的 4–5 m 短前视 → 车道保持与速度都会退化）；第一个动作已被调用方
        替换为**实际执行的采样动作**，保证 PPO on-policy 口径一致。
        """
        if self._env is None:
            raise RuntimeError("LocalEnvPool.step 前必须先 reset()")
        action = np.asarray(actions, dtype=np.float64).reshape(-1)[:2]
        env = self._env
        info: Dict[str, Any] = {}
        terminated = truncated = False
        if self.tracker_kind == "exact":
            self._tracker.arm(env, actions=action.reshape(1, 2))
            for _ in range(self._tracker.substeps):
                _, _, terminated, truncated, info = env.step([0.0, 0.0])
                self._tracker.apply(env)
                if terminated or truncated:
                    break
        elif self.tracker_kind == "lqr":
            reference = action.reshape(1, 2)
            if references is not None:
                reference = np.asarray(references, dtype=np.float64).reshape(-1, 2)
            self._tracker_policy.set_reference(reference)
            for _ in range(max(1, int(round(_POLICY_DT * _PHYSICS_HZ)))):
                _, _, terminated, truncated, info = env.step([0.0, 0.0])
                if terminated or truncated:
                    break
        else:
            expanded = expand_policy_action(action, speed=float(env.agent.speed))
            for env_action in expanded:
                _, _, terminated, truncated, info = env.step([float(env_action[0]), float(env_action[1])])
                if terminated or truncated:
                    break
        self._steps += 1
        self._current_info = info if isinstance(info, dict) else {}
        if not (terminated or truncated) and self._steps >= self.max_episode_steps:
            truncated = True
        if terminated or truncated:
            # P0-2：终局 record 与 reset 分离。record 必须来自终局帧（obs/pose/info/labels）；
            # reset 后的新 episode 首帧只作为 next_obs 交回 trainer。
            record = self._record(reward=0.0, terminated=terminated, truncated=truncated)
            self._env.reset()
            self._setup_episode()
            self._current_info = {}
            # §8.4：终局 reset 后上一动作为 0
            self._env.prev_policy_action = np.zeros(2, dtype=np.float64)
            record["next_obs"] = self._obs_now()
            return [record]
        # §8.4：下一次 build 的 ego reserved0/1 = 本策略步动作 (ds,dθ)
        self._env.prev_policy_action = np.asarray(action, dtype=np.float64).reshape(2)
        return [self._record(reward=0.0, terminated=False, truncated=False)]

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
            self._env = None


class VectorPoolAdapter:
    """``pipeline.vector_env.VectorEnvPool`` 的适配器（策略动作 → 逐子步动作 + 终局整池 reset）。

    worker record 携带 ``pose``、``router_labels``/``has_router_labels``、``prev_action``；
    策略动作经 ``pool.step(..., policy_actions=...)`` 在 worker 内注入
    ``env.prev_policy_action``（步后 obs 的 ego reserved 6:8），本适配器不再事后改写 obs。
    动作执行由本适配器展开（近似运动学，非 in-worker tracker）。

    reset 语义：:meth:`reset` = 整池 reset（episode 起点，spec 按 per-env 轮转分配）；
    :meth:`step` 中**任一** env 终局即整池 reset（未终局 env 标记 ``cut`` 当 episode
    截断、不 bootstrap），池级 RSS 回收经整池 reset 内的 ``_maybe_recycle`` 顺带触发。

    A/B 结论（负结果，已采纳；harness ``/tmp/opencode/ab_harness.py``，日志
    ``ab_full_r1.log`` / ``ab_part_fixed.log``）：本机 per-worker reset **更慢**——
    整池 130 波/36.27 s（279 ms/波）vs per-worker 252 波/63.09 s（250 ms/波）：单波
    耗时相近（重活是 env/地图重建，同波内两次重建并行），但 per-worker 把同样次数的
    重建拆成约 2 倍波数 → 总重置耗时 ≈1.7×。真正的大头是 spec 切换：换 spec reset
    188 ms vs 同 spec 复用 67 ms（``/tmp/opencode/micro_reset_cost.py``），但同 spec ⇒
    重复 episode（数据多样性换速度），留待真实终局率下再评估。
    ``VectorEnvPool.reset_workers`` / ``recycle_due`` 能力保留备用（见 vector_env 模块文档）。
    """

    def __init__(
        self,
        specs: Sequence[Any],
        *,
        num_envs: int = 2,
        tracker: str = "kinematic",
        max_episode_steps: int = 600,
        mem_floor_mb: Optional[float] = None,
        seed_pool: Optional[Sequence[int]] = None,
        logger: Callable[[str], None] = print,
    ):
        from pipeline.vector_env import VectorEnvPool

        self.specs = list(specs)
        if not self.specs:
            raise ValueError("VectorPoolAdapter 需要至少一条 spec")
        self.num_envs = int(num_envs)
        if tracker != "kinematic":
            logger(f"[pool] VectorEnvPool 目前只支持 kinematic 展开（请求 {tracker!r}）→ 用近似执行器")
        self.pool = VectorEnvPool(
            self.num_envs,
            seed_pool=seed_pool,
            mem_floor_mb=mem_floor_mb,
        )
        self.logger = logger
        self._specs_for_env: List[Any] = []
        #: per-env spec 轮转游标：env e 依次取 ``specs[e::num_envs]``（与旧整批分配的
        #: 序列逐项一致；每次 :meth:`reset` 推进各自的游标）。
        self._spec_cursor: List[int] = list(range(self.num_envs))

    def _next_spec(self, env_index: int) -> Any:
        spec = self.specs[self._spec_cursor[env_index] % len(self.specs)]
        self._spec_cursor[env_index] += self.num_envs
        return spec

    def reset(self) -> List[Dict[str, Any]]:
        """整池 reset：每个 env 取下一条轮转 spec（episode 起点）。"""
        self._specs_for_env = [self._next_spec(e) for e in range(self.num_envs)]
        records = self.pool.reset(self._specs_for_env)
        # 新 episode 首帧没有"上一策略步动作"：worker 已把 env.prev_policy_action 置 0
        return records

    def step(self, actions: np.ndarray, references: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        # references（阶段 C 的 6 步规划预览）在 kinematic 展开路径不使用（只执行单动作）
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, 2)
        expanded = []
        for env_index in range(self.num_envs):
            speed = 5.0
            try:
                record_speed = getattr(self, "_last_speed", None)
                if record_speed is not None:
                    speed = float(record_speed[env_index])
            except Exception:  # noqa: BLE001
                pass
            expanded.append(expand_policy_action(actions[env_index], speed=speed))
        records = self.pool.step(expanded, policy_actions=actions)
        # record["obs"] 是**本步之后**的观测：worker 在构建它之前写入了
        # env.prev_policy_action = 本步动作 (ds,dθ)，故 ego reserved 6:8 已就位（§8.4）。
        # 记录速度估计（供下一策略步展开使用）：obs ego[0,0]
        speeds = []
        for record in records:
            obs = record.get("obs") or {}
            ego = np.asarray(obs.get("ego", np.zeros((1, 8))), dtype=np.float32).reshape(-1)
            speeds.append(float(ego[0]) if ego.shape[0] >= 1 else 5.0)
        self._last_speed = speeds
        terminated = [bool(record.get("terminated")) for record in records]
        truncated = [bool(record.get("truncated")) for record in records]
        if not (any(terminated) or any(truncated)):
            return records
        # 旧（更快）语义：任一 env 终局 → 整池 reset（A/B 见类 docstring）。未终局 env
        # 记为 cut（episode 在此截断、不 bootstrap）；所有 env 取下一条轮转 spec。
        for index, record in enumerate(records):
            if not (terminated[index] or truncated[index]):
                record["cut"] = True
        fresh = self.reset()
        for index, record in enumerate(records):
            record["next_obs"] = fresh[index]["obs"]
            record["next_info"] = fresh[index]["info"]
        return records

    def stats(self) -> Dict[str, Any]:
        try:
            return self.pool.recycle_stats()
        except Exception:  # noqa: BLE001
            return {}

    def close(self) -> None:
        try:
            self.pool.close()
        except Exception:  # noqa: BLE001
            pass


def build_pool(
    specs: Sequence[Any],
    *,
    kind: str = "auto",
    num_envs: int = 2,
    tracker: str = "kinematic",
    max_episode_steps: int = 600,
    traffic_density: Optional[float] = None,
    mem_floor_mb: Optional[float] = None,
    seed_pool: Optional[Sequence[int]] = None,
    logger: Callable[[str], None] = print,
) -> Any:
    """构造环境池。

    - ``auto``：``pipeline.vector_env`` 可用且 ``num_envs>1`` → ``VectorPoolAdapter``；
      否则 ``LocalEnvPool``（单进程，1 env，MetaDrive singleton 约束）；
    - ``vector``：强制子进程池；
    - ``local``：强制单进程池（``num_envs`` 强制 1）。
    """
    kind = str(kind).lower()
    if kind not in ("auto", "vector", "local"):
        raise ValueError(f"未知 pool kind={kind!r}（auto/vector/local）")
    if kind in ("auto", "vector"):
        try:
            pool = VectorPoolAdapter(
                specs,
                num_envs=int(num_envs),
                tracker=tracker,
                max_episode_steps=max_episode_steps,
                mem_floor_mb=mem_floor_mb,
                seed_pool=seed_pool,
                logger=logger,
            )
            return pool
        except Exception as exc:  # noqa: BLE001
            if kind == "vector":
                raise
            logger(f"[pool] VectorEnvPool 不可用（{type(exc).__name__}: {exc}）→ 回退 LocalEnvPool（单进程 1 env）")
    if int(num_envs) > 1:
        logger("[pool] MetaDrive 0.4.3 每进程仅一个 engine（base_env.py:543）；LocalEnvPool 只用 1 个 env")
    return LocalEnvPool(
        specs,
        tracker=tracker if tracker in ("exact", "lqr") else "kinematic",
        max_episode_steps=max_episode_steps,
        traffic_density=traffic_density,
        logger=logger,
    )


# --------------------------------------------------------------------------- #
# PPO 训练器
# --------------------------------------------------------------------------- #

class PPOTrainer:
    """CleanRL 风格 PPO：收集 rollout（环境池）→ GAE（``pipeline.buffer``）→ minibatch 更新。"""

    def __init__(
        self,
        model: "nn.Module",
        pool: Any,
        config: PPOConfig,
        *,
        reward_adapter: Optional[RewardAdapter] = None,
        ref_model: Optional["nn.Module"] = None,
        bc_dataset: Optional[BCDataset] = None,
        probe_batch: Optional[str] = None,
        probe_size: int = DEFAULT_PROBE_SIZE,
        logger: Callable[[str], None] = print,
        dt: float = _POLICY_DT,
    ):
        _require_torch()
        self.model = model
        self.pool = pool
        self.config = config
        self.device = torch.device(resolve_device(config.device))
        self.model.to(self.device)
        if reward_adapter is None:
            reward_adapter, self.reward_source = build_reward_adapter(dt=dt, logger=logger)
        else:
            self.reward_source = type(reward_adapter).__name__
        self.reward_adapter = reward_adapter
        self.ref_model = ref_model.to(self.device).eval() if ref_model is not None else None
        if self.ref_model is not None:
            for parameter in self.ref_model.parameters():
                parameter.requires_grad_(False)
        self.bc_dataset = bc_dataset
        self.logger = logger
        self.optimizer = build_optimizer(self.model, config.lr, config.primary_lr_scale)
        self.monitor = RouterMonitor()
        self.rng = np.random.default_rng(config.seed)
        self.torch_generator = torch.Generator(device=self.device).manual_seed(int(config.seed))
        self.low, self.high = _bounds_from_model(model, config.action_low, config.action_high)
        self.low = self.low.to(self.device)
        self.high = self.high.to(self.device)
        self._num_envs = int(getattr(pool, "num_envs", 1))
        self._obs_list: Optional[List[Dict[str, np.ndarray]]] = None
        self._episode_id = [0] * self._num_envs
        self._step_in_episode = [0] * self._num_envs
        self._pose_est: List[Optional[np.ndarray]] = [None] * self._num_envs
        self.buffer: Optional[RolloutBuffer] = None
        self._valid_mask: Optional[np.ndarray] = None
        self._router_labels: Optional[np.ndarray] = None
        self._has_router_labels: Optional[np.ndarray] = None
        self._total_steps = 0
        self._updates_done = 0
        self._value_param_names = _value_parameter_names(model)
        self._last_metrics: Dict[str, Any] = {}
        self._last_collect_timing: Dict[str, Any] = {}
        self._pose_warned = False
        # 诊断：逐步奖励分解窗口 + 固定探针批（动作漂移 / 低速吸引子）
        self._reward_stats = RewardStatistics()
        self._probe_obs: Optional[Dict[str, "torch.Tensor"]] = None
        self._probe_speed: Optional[np.ndarray] = None
        self._setup_probe(probe_batch, probe_size)

    # ---------------------------------------------------------------- 工具
    def adopt_obs(self, obs_list: Sequence[Dict[str, np.ndarray]]) -> None:
        """接管外部已 reset 的观测（避免重复 reset 建图）。"""
        self._obs_list = list(obs_list)
        self._num_envs = int(len(self._obs_list))
        self._episode_id = [0] * self._num_envs
        self._step_in_episode = [0] * self._num_envs
        self._pose_est = [None] * self._num_envs
        self.reward_adapter.reset_all()

    # ---------------------------------------------------------------- 探针批
    def _setup_probe(self, probe_batch: Optional[str], probe_size: int) -> None:
        """加载**固定**探针批（一次），供每个 update 检测动作分布漂移/低速吸引子。

        失败/路径缺失时静默关闭（只打日志），绝不影响训练；观测/ego 速度在加载时固化，
        保证跨 update 可比（唯一变量 = 模型权重）。
        """
        if not probe_batch:
            return
        target = Path(str(probe_batch))
        if not target.exists():
            self.logger(
                f"[probe] 警告：train.probe_batch 路径不存在：{target} → "
                "动作漂移/低速吸引子探针关闭（probe.available=0；置 null 可显式关闭告警）"
            )
            return
        try:
            dataset = BCDataset.load(str(target))
            size = max(1, min(int(probe_size), int(dataset.count)))
            if size >= int(dataset.count):
                indices = np.arange(int(dataset.count), dtype=np.int64)
            else:
                # 均匀跨全数据集抽样（固定）：覆盖各速度档/场景，避免前缀批在低速档为空
                indices = np.linspace(0, int(dataset.count) - 1, num=size, dtype=np.int64)
            obs = dataset.build_obs_batch(indices)
            ego = np.asarray(dataset.arrays["ego"], dtype=np.float64)
            speed = ego[indices, 0, 0] if ego.ndim == 3 else ego[indices, 0]
            self._probe_obs = _to_device_obs(obs, self.device)
            self._probe_speed = np.asarray(speed, dtype=np.float64).reshape(-1)
            self.logger(f"[probe] 固定探针批已加载：{target}（{size} 帧，device={self.device}）")
        except Exception as exc:  # noqa: BLE001 - 探针是诊断，绝不阻塞训练
            self._probe_obs = None
            self._probe_speed = None
            self.logger(f"[probe] 探针批加载失败（{type(exc).__name__}: {exc}）→ 探针关闭")

    @torch.no_grad() if torch is not None else (lambda fn: fn)
    def _probe_metrics(self) -> Dict[str, Any]:
        """固定批 cheap-path 前向：``action_mu[:,0]``/``logstd`` 均值-方差 + 分速度档 ds。

        ``probe/low_speed_alert=1`` ⇔ 任一 speed<2 m/s 档的 ds 均值 < 1.5 m（低速吸引子），
        同时打印一行告警。返回 ``{"probe": {...}}``（无探针时 ``available=0``）。
        """
        if self._probe_obs is None or self._probe_speed is None:
            return {"probe": {"available": 0.0}}
        was_training = bool(self.model.training)
        self.model.eval()
        try:
            out = self.model(self._probe_obs, rollout=False, world_model=False)
            mu = out["action_mu"].detach().double().cpu().numpy()
            logstd = out["action_logstd"].detach().double().cpu().numpy()
        finally:
            self.model.train(was_training)
        ds = np.asarray(mu[:, 0], dtype=np.float64)
        speed = self._probe_speed
        stats: Dict[str, float] = {
            "available": 1.0,
            "n": float(ds.shape[0]),
            "action_mu_ds_mean": float(ds.mean()),
            "action_mu_ds_std": float(ds.std()),
            "action_logstd_mean": float(logstd.mean()),
            "action_logstd_std": float(logstd.std()),
        }
        alert = 0.0
        low_speed_reports: List[str] = []
        for low, high in _PROBE_SPEED_BINS:
            key = _probe_speed_bin_key(low, high)
            mask = (speed >= low) & (speed < high)
            count = int(mask.sum())
            stats[f"n_speed_{key}"] = float(count)
            if count <= 0:
                continue
            mean_ds = float(ds[mask].mean())
            stats[f"ds_mean_speed_{key}"] = mean_ds
            if high <= 2.0:
                low_speed_reports.append(f"{key}m/s→ds={mean_ds:.2f}m")
                if mean_ds < _PROBE_LOW_SPEED_DS_THRESHOLD:
                    alert = 1.0
        stats["low_speed_alert"] = alert
        if alert:
            self.logger(
                f"[probe] 低速吸引子告警：ds 均值 < {_PROBE_LOW_SPEED_DS_THRESHOLD:.1f} m"
                f"（speed<2 m/s：{', '.join(low_speed_reports)}）"
            )
        return {"probe": stats}

    def _to_tensor_obs(self, obs_list: Sequence[Mapping[str, np.ndarray]]) -> Dict["torch.Tensor"]:
        keys = [key for key in obs_list[0] if key not in NON_OBS_KEYS]
        batch = {
            key: np.stack([np.asarray(obs[key], dtype=np.float32) for obs in obs_list], axis=0)
            for key in keys
        }
        squeeze_single_slot(batch)
        return {key: torch.as_tensor(value, device=self.device) for key, value in batch.items()}

    @staticmethod
    def _current_template(obs: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {
            key: np.asarray(value)
            for key, value in obs.items()
            if key not in NON_OBS_KEYS and not key.endswith(_HIST_SUFFIX) and key != "hist_valid"
        }

    def _frame_pose(self, env_index: int, obs: Mapping[str, np.ndarray]) -> np.ndarray:
        """帧位姿：优先 obs['pose']，否则用动作航位推算（相对量足够对齐，见模块 docstring）。"""
        pose = obs.get("pose") if isinstance(obs, Mapping) else None
        if pose is not None:
            return np.asarray(pose, dtype=np.float32).reshape(3)
        if self._pose_est[env_index] is None:
            self._pose_est[env_index] = np.zeros(3, dtype=np.float64)
        return self._pose_est[env_index].astype(np.float32)

    def _advance_pose(self, env_index: int, action: np.ndarray) -> None:
        if self._pose_est[env_index] is None:
            self._pose_est[env_index] = np.zeros(3, dtype=np.float64)
        pose = _arc_endpoint(self._pose_est[env_index], float(action[0]), float(action[1]))
        pose[2] = (pose[2] + math.pi) % (2.0 * math.pi) - math.pi
        self._pose_est[env_index] = pose

    def _reward(
        self, env_index: int, info: Dict[str, Any], obs: Mapping[str, np.ndarray], done: bool, pool_reward: float
    ) -> Tuple[float, Dict[str, Any]]:
        return self.reward_adapter.step(
            env_index, info, obs, done, pool_reward, step_index=self._step_in_episode[env_index]
        )

    # ---------------------------------------------------------------- 收集
    def reset_pool(self) -> List[Dict[str, np.ndarray]]:
        records = self.pool.reset()
        self.adopt_obs([record["obs"] for record in records])
        return self._obs_list  # type: ignore[return-value]

    def collect_rollout(self, horizon: int) -> int:
        """收集 ``horizon`` 个策略步（每 env）；返回总帧数。"""
        if self._obs_list is None:
            self.reset_pool()
        assert self._obs_list is not None
        self._num_envs = int(len(self._obs_list))
        horizon = int(horizon)
        self._reward_stats.reset()
        # 每个 env 的逐帧记录（env-major 写入 buffer，见模块 docstring）
        pending: List[List[Dict[str, Any]]] = [[{} for _ in range(horizon)] for _ in range(self._num_envs)]
        self.model.eval()
        t_policy = 0.0
        t_env = 0.0
        # V9（P2 性能，2026-09-30）：collect 默认走与 update 同款的 cheap path（rollout=False，
        # 不跑 B1 递归编排/WM 多步）。P0-1 后默认 plan_reference=repeat_action，plan 输出无人
        # 消费（LQR 参考 = repeat(a_t)，见下）；cheap 路径的 action_mu/logstd/value 与 rollout
        # 路径逐位一致（net/model.py:466-468；tests/test_net_shapes.py::test_cheap_path_matches_full_forward）。
        # plan_reference="plan"（旧行为对照）需 plan 输出 → 保留 rollout=True。
        need_plan = self.config.plan_reference == "plan"
        for step in range(horizon):
            t0 = time.perf_counter()
            batch = self._to_tensor_obs(self._obs_list)
            with torch.no_grad():
                out = self.model(batch, rollout=need_plan, world_model=False)
                mu = out["action_mu"]
                logstd = out["action_logstd"]
                value = out["value"].reshape(-1)
                if need_plan:
                    plan = out["plan"]
                action, logprob = sample_action(
                    mu, logstd, self.low, self.high, mode=self.config.action_mode, generator=self.torch_generator
                )
            actions_np = action.detach().cpu().numpy().astype(np.float32)
            logprob_np = logprob.detach().cpu().numpy()
            value_np = value.detach().cpu().numpy()
            # 跟踪器参考（P0-1 A-hold）：执行侧只依赖 PPO 记账的随机变量 a_t。
            #   repeat_action（默认）：6 步参考 = repeat(a_t)——与 buffer 的 (obs_t, a_t, r_t) 口径一致；
            #   plan（仅对照，旧行为）：首步 = a_t，后 5 步 = WM/plan head 规划预览（记账外变量）。
            if need_plan:
                references_np = plan.detach().cpu().numpy().astype(np.float32)
                references_np[:, 0, :] = actions_np
            else:
                references_np = np.repeat(actions_np[:, None, :], 6, axis=1)
            t_policy += time.perf_counter() - t0
            # P0-2：router 标签必须与 buffer 的 obs_current 同帧 → 在动作执行**前**取快照。
            # LocalEnvPool 提供 labels_now()（1 env）；Vector 池由 worker 在 step 开始时快照
            # （record["labels_at_step_start"]）。
            labels_pre: Optional[Any] = None
            labels_now = getattr(self.pool, "labels_now", None)
            if callable(labels_now):
                try:
                    labels_pre = labels_now()
                except Exception:  # noqa: BLE001 - 标签失败不阻塞训练
                    labels_pre = None
            t1 = time.perf_counter()
            records = self.pool.step(actions_np, references=references_np)
            t_env += time.perf_counter() - t1
            next_obs = []
            for env_index in range(self._num_envs):
                record = records[env_index] if env_index < len(records) else {}
                obs_current = self._obs_list[env_index]
                info = record.get("info") if isinstance(record.get("info"), dict) else {}
                terminated = bool(record.get("terminated", False))
                truncated = bool(record.get("truncated", False))
                cut = bool(record.get("cut", False))
                done = terminated or truncated or cut
                reward, reward_meta = self._reward(
                    env_index, info, obs_current, done, float(record.get("reward", 0.0) or 0.0)
                )
                self._reward_stats.update(reward_meta, reward)
                pose = self._frame_pose(env_index, obs_current)
                # router_labels（P0-2 对齐）：worker 步开始快照（Vector）> 池级步前快照（Local）
                #   > record/info 的步后标签（旧接口回退）。
                labels = record.get("labels_at_step_start") if isinstance(record, dict) else None
                if labels is None and labels_pre is not None:
                    labels = _labels_for_env(labels_pre, env_index)
                if labels is None:
                    labels = record.get("router_labels") if isinstance(record, dict) else None
                if labels is None:
                    labels = info.get("router_labels") if isinstance(info, dict) else None
                pending[env_index][step] = {
                    "obs": obs_current,
                    "pose": pose,
                    "action": actions_np[env_index],
                    "logprob": float(logprob_np[env_index]),
                    "value": float(value_np[env_index]),
                    "reward": float(reward),
                    "terminated": bool(terminated or cut),
                    "truncated": bool(truncated),
                    "router_labels": np.asarray(labels, dtype=np.float32) if labels is not None else None,
                    "episode": int(self._episode_id[env_index]),
                    "step": int(self._step_in_episode[env_index]),
                }
                self._advance_pose(env_index, actions_np[env_index])
                self._step_in_episode[env_index] += 1
                if done:
                    self._episode_id[env_index] += 1
                    self._step_in_episode[env_index] = 0
                    self._pose_est[env_index] = None
                next_obs.append(record.get("next_obs", record.get("obs")))
            self._obs_list = [obs if obs is not None else self._obs_list[i] for i, obs in enumerate(next_obs)]
        # 末端 bootstrap 值
        last_batch = self._to_tensor_obs(self._obs_list)
        with torch.no_grad():
            last_values = (
                self.model(last_batch, rollout=False, world_model=False)["value"]
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )
        # 组装 buffer（env-major + bootstrap 行）
        template = self._current_template(self._obs_list[0])
        channels = {key: tuple(np.asarray(value).shape) for key, value in template.items()}
        # R7（G1 复核）：与 RolloutBuffer 默认 / env builder / Stage A/B 物化一致的四通道。
        # 旧行为仅 ("od", "ld")：update 侧 ego/others 历史缺失 → net.mem 回退"当前帧复制 6 帧"，
        # 与 collect 真历史不一致 ⇒ no-op 优化器下 approx_kl 仍非零（系统性 collect≠update 偏置）。
        history_channels = tuple(
            name for name in ("ego", "others", "od", "ld") if f"{name}{_HIST_SUFFIX}" in self._obs_list[0]
        ) or ("ego", "others", "od", "ld")
        capacity = self._num_envs * (horizon + 1)
        buffer = RolloutBuffer(
            capacity,
            channels=channels,
            history_frames=6,
            history_interval=1,
            history_channels=history_channels,
        )
        valid = np.zeros(capacity, dtype=bool)
        router_labels = np.zeros((capacity, 8), dtype=np.float32)
        has_labels = np.zeros(capacity, dtype=bool)
        zero_obs = {key: np.zeros_like(np.asarray(template[key])) for key in template}
        for env_index in range(self._num_envs):
            last_episode = 0
            last_step = 0
            for step in range(horizon):
                frame = pending[env_index][step]
                index = buffer.add_step(
                    frame["obs"],
                    pose=frame["pose"],
                    action=frame["action"],
                    logprob=frame["logprob"],
                    value=frame["value"],
                    reward=frame["reward"],
                    terminated=frame["terminated"],
                    truncated=frame["truncated"],
                    episode=frame["episode"],
                    step=frame["step"],
                )
                valid[index] = True
                if frame["router_labels"] is not None:
                    router_labels[index] = frame["router_labels"]
                    has_labels[index] = True
                last_episode, last_step = frame["episode"], frame["step"]
            buffer.add_step(
                zero_obs,
                pose=np.zeros(3, dtype=np.float32),
                action=np.zeros(2, dtype=np.float32),
                logprob=0.0,
                value=float(last_values[env_index]),
                reward=0.0,
                terminated=False,
                truncated=False,
                episode=last_episode,
                step=last_step + 1,
            )
        self.buffer = buffer
        self._valid_mask = valid
        self._router_labels = router_labels
        self._has_router_labels = has_labels
        self._total_steps += self._num_envs * horizon
        self._last_collect_timing = {
            "policy_s": float(t_policy),
            "env_step_s": float(t_env),
            "horizon": int(horizon),
            "num_envs": int(self._num_envs),
            # V9：collect 前向路径口径（cheap = rollout=False，repeat_action 臂；rollout = plan 对照臂）
            "path": "rollout" if need_plan else "cheap",
        }
        if not has_labels.any() and not self._pose_warned:
            self.logger("[trainer] rollout 无 router_labels（标签模块不可用？）→ router BCE 本轮跳过")
            self._pose_warned = True
        return self._num_envs * horizon

    # ---------------------------------------------------------------- 更新
    def _assemble_obs_batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        """展平索引 → 模型输入：当前帧取缓冲，历史窗口用 ``buffer.build_history`` 重拼。"""
        assert self.buffer is not None
        indices = np.asarray(indices, dtype=np.int64)
        history = self.buffer.build_history(indices)
        batch: Dict[str, np.ndarray] = {}
        for name in self.buffer.channels:
            batch[name] = np.stack([self.buffer.obs[name][index] for index in indices], axis=0).astype(np.float32)
            mask = self.buffer.obs_mask.get(name)
            if mask is not None:
                batch[f"{name}_mask"] = np.stack([mask[index] for index in indices], axis=0).astype(np.float32)
        for key, value in history.items():
            batch[key] = np.asarray(value, dtype=np.float32)
        return squeeze_single_slot(batch)

    def update(self) -> Dict[str, Any]:
        """PPO 更新（clip/value/entropy/router/KL/BC 锚）；返回聚合指标。

        ``critic_warmup_updates``（默认 0）：前 N 个 update 只做 value-only 优化——
        仅 value 头可训练（其余参数 ``requires_grad`` 临时冻结并在 update 结束时恢复），
        损失 = ``value_loss``（MSE）+ 梯度裁剪，无 policy/entropy/router/KL/BC 项，
        指标不含 ``policy_loss``；指标 ``critic_warmup=true/false`` 标记当前 update 口径。
        """
        if self.buffer is None or self._valid_mask is None:
            raise RuntimeError("update() 之前必须先 collect_rollout()")
        buffer = self.buffer
        valid_indices = np.where(self._valid_mask)[0]
        advantages_all, returns_all = buffer.compute_gae(
            last_value=0.0, gamma=self.config.gamma, lam=self.config.lam
        )
        # 诊断统计：原始（归一化前）优势 + 回报/价值解释方差（进入 metrics → monitor）
        advantage_stats = _advantage_stats(
            np.asarray(advantages_all[valid_indices], dtype=np.float64),
            np.asarray(returns_all[valid_indices], dtype=np.float64),
            np.asarray(buffer.value[valid_indices], dtype=np.float64),
        )
        advantages = advantages_all[valid_indices].astype(np.float64)
        returns = returns_all[valid_indices].astype(np.float32)
        if self.config.normalize_advantage and advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        old_logprobs_t = torch.as_tensor(buffer.logprob[valid_indices], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(buffer.action[valid_indices], dtype=torch.float32, device=self.device)
        assert self._router_labels is not None and self._has_router_labels is not None
        # critic warmup：前 N 个 update 只拟合 value 头（策略/主干冻结，value 梯度不回流共享主干）
        warmup = self._updates_done < max(0, int(self.config.critic_warmup_updates))
        saved_requires: Optional[List[Tuple["torch.Tensor", bool]]] = None
        if warmup:
            value_names = set(self._value_param_names)
            if not value_names:
                raise RuntimeError("critic warmup 需要可识别的 value 头参数（参数名需含 'value'）")
            saved_requires = [(parameter, bool(parameter.requires_grad)) for parameter in self.model.parameters()]
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad_(name in value_names)
            self.logger(
                f"[trainer] critic warmup {self._updates_done + 1}/{int(self.config.critic_warmup_updates)}："
                f"仅 value 头可训练（{len(value_names)} 个参数），策略/主干冻结"
            )
        self.model.train()
        self.monitor.reset()
        aggregates: Dict[str, List[float]] = {}
        batches = 0
        total = int(valid_indices.shape[0])
        t_data = t_forward = t_backward = 0.0
        try:
            for epoch in range(max(1, int(self.config.epochs))):
                rng = np.random.default_rng(self.config.seed + 1000 * epoch)
                order = rng.permutation(total)
                for start in range(0, total, max(1, int(self.config.minibatch_size))):
                    selection = order[start : start + max(1, int(self.config.minibatch_size))]
                    t0 = time.perf_counter()
                    obs_batch = _to_device_obs(self._assemble_obs_batch(valid_indices[selection]), self.device)
                    t_data += time.perf_counter() - t0
                    t1 = time.perf_counter()
                    # PPO 只需要 heads/router/logstd/value：走 cheap path（不跑 B1 rollout 与 WM 多步）
                    out = self.model(obs_batch, rollout=False, world_model=False)
                    value = out["value"].reshape(-1)
                    if warmup:
                        # value-only：损失 = critic MSE（无 policy/entropy/router/KL/BC 项）
                        value_loss = F.mse_loss(value, returns_t[selection])
                        loss = value_loss
                    else:
                        mu = out["action_mu"]
                        logstd = out["action_logstd"]
                        new_logprob = logprob_from_action(
                            mu, logstd, actions_t[selection], self.low, self.high, mode=self.config.action_mode
                        )
                        old_logprob = old_logprobs_t[selection]
                        ratio = torch.exp(new_logprob - old_logprob)
                        adv = advantages_t[selection]
                        policy_loss = -torch.min(
                            ratio * adv, torch.clamp(ratio, 1.0 - self.config.clip, 1.0 + self.config.clip) * adv
                        ).mean()
                        value_loss = F.mse_loss(value, returns_t[selection])
                        entropy = gaussian_entropy(logstd).mean()
                        loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
                        kl_anchor = torch.zeros((), device=self.device)
                        if self.ref_model is not None and self.config.kl_anchor_coef > 0.0:
                            with torch.no_grad():
                                ref_out = self.ref_model(obs_batch, rollout=False, world_model=False)
                            with torch.no_grad():
                                ref_raw_mu = raw_mu_from_action(
                                    ref_out["action_mu"], self.low, self.high, mode=self.config.action_mode
                                )
                                raw_mu = raw_mu_from_action(mu, self.low, self.high, mode=self.config.action_mode)
                            kl_anchor = self.config.kl_anchor_coef * gaussian_kl(
                                raw_mu, logstd, ref_raw_mu, ref_out["action_logstd"]
                            ).mean()
                            loss = loss + kl_anchor
                        bc_anchor = torch.zeros((), device=self.device)
                        if self.bc_dataset is not None and self.config.bc_anchor_coef > 0.0:
                            # 权重感知 BC 锚：每个 rollout 样本抽一个专家动作（等长，避免广播错位），
                            # 用 row_action_weights（train_weight×balance_weight）显式加权
                            # Σ w·|μ−a| / Σ w（P0-4 的完整修复=按 BC 观测前向，另行处理）。
                            expert_idx = self.rng.integers(0, self.bc_dataset.count, size=len(selection))
                            expert_action = torch.as_tensor(
                                self.bc_dataset.arrays["action"][expert_idx, 0],
                                dtype=torch.float32,
                                device=self.device,
                            )
                            expert_weight = torch.as_tensor(
                                row_action_weights(self.bc_dataset, expert_idx),
                                dtype=torch.float32,
                                device=self.device,
                            )
                            per_sample = (mu - expert_action).abs().mean(dim=-1)
                            bc_anchor = self.config.bc_anchor_coef * (
                                (per_sample * expert_weight).sum() / expert_weight.sum().clamp(min=1e-8)
                            )
                            loss = loss + bc_anchor
                    t_forward += time.perf_counter() - t1
                    t2 = time.perf_counter()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm))
                    self.optimizer.step()
                    t_backward += time.perf_counter() - t2
                    self.monitor.update(out)
                    if warmup:
                        aggregates.setdefault("value_loss", []).append(float(value_loss.detach()))
                        aggregates.setdefault("total_loss", []).append(float(loss.detach()))
                        aggregates.setdefault("grad_norm", []).append(grad_norm)
                    else:
                        with torch.no_grad():
                            approx_kl = float(((ratio - 1.0) - (new_logprob - old_logprob)).mean())
                            clipfrac = float(((ratio - 1.0).abs() > self.config.clip).float().mean())
                        for key, val in {
                            "policy_loss": policy_loss,
                            "value_loss": value_loss,
                            "entropy": entropy,
                            "total_loss": loss,
                            "kl_anchor": kl_anchor,
                            "bc_anchor": bc_anchor,
                        }.items():
                            aggregates.setdefault(key, []).append(float(val.detach()))
                        aggregates.setdefault("approx_kl", []).append(approx_kl)
                        aggregates.setdefault("clipfrac", []).append(clipfrac)
                        aggregates.setdefault("grad_norm", []).append(grad_norm)
                    batches += 1
                if not warmup and self.config.target_kl is not None and aggregates.get("approx_kl"):
                    recent = aggregates["approx_kl"][-max(1, batches // max(1, self.config.epochs)) :]
                    if float(np.mean(recent)) > 1.5 * float(self.config.target_kl):
                        break
        finally:
            if saved_requires is not None:
                for parameter, flag in saved_requires:
                    parameter.requires_grad_(flag)
        metrics = {key: float(np.mean(values)) for key, values in aggregates.items()}
        metrics["critic_warmup"] = bool(warmup)
        metrics["batches"] = batches
        metrics["total_steps"] = self._total_steps
        metrics["update_timing_s"] = {
            "data_s": float(t_data),
            "forward_s": float(t_forward),
            "backward_s": float(t_backward),
        }
        metrics["reward_early_terminations"] = int(self.reward_adapter.early_terminations)
        metrics.update(self._reward_stats.summary())
        metrics.update(advantage_stats)
        metrics.update(self._probe_metrics())
        metrics.update(self.monitor.summary())
        self._last_metrics = metrics
        self._updates_done += 1
        return metrics

    def train(self, updates: int, horizon: int) -> List[Dict[str, Any]]:
        """便捷循环：``updates`` 次 ``collect_rollout + update``；返回每次指标。"""
        history = []
        for _ in range(int(updates)):
            self.collect_rollout(int(horizon))
            history.append(self.update())
        return history

    # ---------------------------------------------------------------- 回放
    def save_replay(self, path: str) -> str:
        """保存最近一次 rollout（``buffer.to_arrays`` schema）供阶段 B 离线 WM。"""
        if self.buffer is None:
            raise RuntimeError("save_replay() 之前必须先 collect_rollout()")
        arrays = self.buffer.to_arrays(copy=True)
        if self._valid_mask is not None:
            arrays["valid_mask"] = self._valid_mask.copy()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        target.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "kind": "ppo_replay",
                    "label_names": list(load_supervised_labels()),
                    "history_storage": "per_frame",
                    "lab": {
                        "gamma": self.config.gamma,
                        "lam": self.config.lam,
                        "clip": self.config.clip,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(target)


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #

def config_snapshot_hash(config: Mapping[str, Any]) -> str:
    """配置快照哈希（sha256 前 16 hex）：canonical JSON（键排序、非 JSON 值 ``str`` 化）。"""
    canonical = json.dumps(dict(config or {}), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def capture_rng_state(numpy_generator: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """捕获 RNG 状态（``torch``/``numpy``/``random``；可选训练用 ``default_rng`` 生成器）。

    周期 ckpt 恢复用：``numpy_generator`` 是训练数据顺序的来源（每个 epoch 一次 shuffle /
    permutation），不捕获它就只能靠 seed 重放。CUDA 未初始化/无 torch 时对应键省略。
    """
    state: Dict[str, Any] = {
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }
    if numpy_generator is not None:
        try:
            state["numpy_generator"] = copy.deepcopy(numpy_generator.bit_generator.state)
        except Exception:  # noqa: BLE001 - 非标准 bit_generator → 退化到重放
            pass
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            try:
                state["torch_cuda"] = torch.cuda.get_rng_state_all()
            except Exception:  # noqa: BLE001 - CUDA 未初始化
                pass
    except Exception:  # noqa: BLE001 - 无 torch 环境
        pass
    return state


def restore_rng_state(
    state: Optional[Mapping[str, Any]], numpy_generator: Optional[np.random.Generator] = None
) -> None:
    """恢复 :func:`capture_rng_state` 的状态（缺失键跳过；非法状态忽略并告警）。"""
    if not state:
        return
    try:
        import torch

        if state.get("torch") is not None:
            value = state["torch"]
            torch.set_rng_state(value.to("cpu") if hasattr(value, "to") else value)
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.to("cpu") if hasattr(item, "to") else item
                                          for item in state["torch_cuda"]])
    except Exception as exc:  # noqa: BLE001
        print(f"[ckpt] torch RNG 状态恢复失败（{type(exc).__name__}: {exc}）→ 保持当前状态", flush=True)
    if numpy_generator is not None and state.get("numpy_generator") is not None:
        try:
            numpy_generator.bit_generator.state = state["numpy_generator"]
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt] numpy 生成器状态恢复失败（{type(exc).__name__}: {exc}）→ 依赖 seed 重放", flush=True)
    if state.get("numpy") is not None:
        try:
            np.random.set_state(state["numpy"])
        except Exception:  # noqa: BLE001
            pass
    if state.get("random") is not None:
        try:
            random.setstate(state["random"])
        except Exception:  # noqa: BLE001
            pass


def move_optimizer_state_to_device(optimizer: "torch.optim.Optimizer") -> int:
    """把优化器状态张量迁到**各参数当前所在设备**（跨设备 resume 必需；幂等，返回迁移个数）。

    必须在 ``model.to(device)`` **之后**调用：优化器通常在 CPU 上创建/恢复状态，随后模型才搬到
    CUDA；若顺序颠倒，Adam 会在首个 step 抛
    ``RuntimeError: Expected all tensors to be on the same device, cuda:0 and cpu``（2026-09-27 事故）。
    """
    params_flat = [param for group in optimizer.param_groups for param in group.get("params", [])]
    fallback = params_flat[0].device if params_flat else None
    moved = 0
    for key, entry in list((optimizer.state or {}).items()):
        if not isinstance(entry, Mapping):
            continue
        device = None
        try:
            index = int(key)  # state_dict 载入后键是展开参数的下标
            if 0 <= index < len(params_flat):
                device = params_flat[index].device
        except (TypeError, ValueError):
            device = None
        if device is None:
            device = fallback
        if device is None:
            continue
        for name, value in list(entry.items()):
            if torch.is_tensor(value) and value.device != device:
                entry[name] = value.to(device)
                moved += 1
    return moved


def load_optimizer_state(
    optimizer: "torch.optim.Optimizer",
    state: Optional[Mapping[str, Any]],
    *,
    logger: Callable[[str], None] = print,
) -> bool:
    """把 ckpt 里的优化器状态载入 ``optimizer``；不可用/不匹配时告警并返回 False（用全新优化器）。

    ⚠️ **跨设备**：ckpt 可能来自别的设备（如 GPU 训练 → `torch.load(map_location="cpu")` 载入），
    载入前必须把状态张量搬到**对应参数所在设备**，否则第一个 step 会
    ``RuntimeError: Expected all tensors to be on the same device``（2026-09-27 事故）。
    """
    if not state:
        return False
    try:
        payload = dict(state)
        # torch 约定：state 的键是 param_groups 展开后的参数下标 → 与 params_flat 一一对应
        params_flat = [p for group in optimizer.param_groups for p in group.get("params", [])]
        fallback_device = params_flat[0].device if params_flat else None
        entries = payload.get("state")
        if isinstance(entries, Mapping):
            for key, entry in entries.items():
                if not isinstance(entry, Mapping):
                    continue
                device = fallback_device
                try:
                    index = int(key)
                    if 0 <= index < len(params_flat):
                        device = params_flat[index].device
                except (TypeError, ValueError):
                    pass
                if device is None:
                    continue
                for name, value in list(entry.items()):
                    if torch.is_tensor(value) and value.device != device:
                        entry[name] = value.to(device)
        optimizer.load_state_dict(payload)
        move_optimizer_state_to_device(optimizer)
        return True
    except Exception as exc:  # noqa: BLE001 - 参数组/形状不匹配（如阶段切换）→ 全新优化器
        logger(f"[ckpt] 优化器状态载入失败（{type(exc).__name__}: {exc}）→ 使用全新优化器")
        return False


def save_checkpoint(
    path: str,
    model: "nn.Module",
    *,
    meta: Optional[Dict[str, Any]] = None,
    optimizer: Optional["torch.optim.Optimizer"] = None,
    epoch: Optional[int] = None,
    val_metrics: Optional[Mapping[str, Any]] = None,
    rng_state: Optional[Mapping[str, Any]] = None,
    config_hash: Optional[str] = None,
) -> str:
    """保存 checkpoint（**固定键集**，周期 ckpt 与 ``final.pt`` 同格式/payload）。

    payload 键：``model``（state_dict）、``optimizer``（state_dict 或 None）、``epoch``
    （已完成的全局 epoch 数；resume 从 ``epoch+1`` 继续）、``val_metrics``、``rng_state``
    （见 :func:`capture_rng_state`）、``config_hash``（见 :func:`config_snapshot_hash`）、
    ``meta``。键集固定 → ``torch.load(...).keys()`` 可跨文件比较。
    """
    _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "meta": dict(meta or {}),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": None if epoch is None else int(epoch),
        "val_metrics": dict(val_metrics or {}),
        "rng_state": dict(rng_state or {}),
        "config_hash": config_hash,
    }
    torch.save(payload, target)
    return str(target)


def load_checkpoint(path: str, model: "nn.Module", *, strict: bool = True) -> Dict[str, Any]:
    """加载 checkpoint，返回 meta（含 missing/unexpected keys）。

    向后兼容（lane T）：旧 ckpt 缺**新增头**（如二值门控 ``plan_head.gate.*``）时
    ``strict=True`` 会直接抛错、整个 lane 无法加载历史权重；这里在严格加载失败后自动回退
    ``strict=False`` 并在 meta 记录 missing/unexpected（新增头保持初始化值），旧行为不变
    （严格加载成功时仍是严格路径）。
    """
    _require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    try:
        missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    except RuntimeError:
        if not strict:
            raise
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        print(
            f"[ckpt] 兼容加载 {path}：missing={len(missing)}（新增头保持初始化）· "
            f"unexpected={len(unexpected)}",
            flush=True,
        )
    meta = dict(payload.get("meta") or {})
    meta["missing_keys"], meta["unexpected_keys"] = list(missing), list(unexpected)
    return meta


def load_training_checkpoint(
    path: str, model: "nn.Module", *, strict: bool = True
) -> Dict[str, Any]:
    """加载训练恢复用 checkpoint（model + 可选优化器/epoch/val/rng/配置哈希）。

    返回 dict：``meta``/``epoch``/``val_metrics``/``rng_state``/``config_hash``/
    ``optimizer``（原始 state_dict，交给调用方在优化器建好后经
    :func:`load_optimizer_state` 载入；``None`` = 无）/``missing_keys``/``unexpected_keys``。
    旧格式（仅 ``model``+``meta``）兼容：缺失字段为 None/空。
    """
    _require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(payload["model"], strict=strict)
    return {
        "meta": dict(payload.get("meta") or {}),
        "epoch": None if payload.get("epoch") is None else int(payload["epoch"]),
        "val_metrics": dict(payload.get("val_metrics") or {}),
        "rng_state": dict(payload.get("rng_state") or {}),
        "config_hash": payload.get("config_hash"),
        "optimizer": payload.get("optimizer"),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


# --------------------------------------------------------------------------- #
# smoke：真实模型 + 真实池（缺依赖时文档化回退）
# --------------------------------------------------------------------------- #

class _SmokeModel(nn.Module):
    """**smoke 专用** stub：实现 ``net/model.py`` 契约的最小接口（真实模型不可用时）。"""

    def __init__(
        self,
        obs_template: Dict[str, np.ndarray],
        hidden: int = 64,
        num_experts: int = 8,
        action_low: Sequence[float] = DEFAULT_ACTION_LOW,
        action_high: Sequence[float] = DEFAULT_ACTION_HIGH,
    ):
        super().__init__()
        self.keys = sorted(obs_template.keys())
        self.encoders = nn.ModuleDict()
        for key in self.keys:
            dim = int(np.prod(np.asarray(obs_template[key]).shape))
            self.encoders[key] = nn.Linear(dim, hidden)
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.head_mu = nn.Linear(hidden, 2)
        self.head_logstd = nn.Parameter(torch.full((2,), -1.0))
        self.head_value = nn.Linear(hidden, 1)
        self.head_traj = nn.Linear(hidden, 2 * 6)
        self.head_router = nn.Linear(hidden, num_experts)
        self.register_buffer("action_low", torch.tensor(list(action_low), dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(list(action_high), dtype=torch.float32))

    def forward(
        self,
        obs: Dict[str, "torch.Tensor"],
        *,
        rollout: bool = True,
        world_model: bool = True,
        wm_detach: bool = False,
    ) -> Dict[str, "torch.Tensor"]:
        """stub 无多步分支：``rollout/world_model/wm_detach`` 标志只为兼容真实模型签名。"""
        hidden = self.trunk(torch.cat([self.encoders[key](torch.flatten(obs[key], start_dim=1)) for key in self.keys], dim=-1))
        span = (self.action_high - self.action_low)
        unit = torch.sigmoid(self.head_mu(hidden))
        action = self.action_low + span * unit
        return {
            "action_mu": action,
            "action_logstd": self.head_logstd.clamp(-5.0, 0.0).unsqueeze(0).expand(hidden.shape[0], -1),
            "value": self.head_value(hidden),
            "traj_xy": self.head_traj(hidden).reshape(-1, 6, 2),
            # 6 步规划预览（收集侧跟踪器参考用）：stub 用同一动作重复
            "plan": action.unsqueeze(1).expand(-1, 6, -1).contiguous(),
            "router_logits": self.head_router(hidden),
            "latent": hidden,
        }

    def rollout(self, obs: Dict[str, "torch.Tensor"]) -> Dict[str, "torch.Tensor"]:
        return self.forward(obs)


def _try_build_real_model(spec: str, model_cfg: Dict[str, Any]) -> Any:
    """构造真实 ``net.model.DrivingModel``；失败返回 None（调用方回退 stub）。"""
    module_name, _, class_name = spec.partition(":")
    if not module_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001
        return None
    builder = getattr(module, "build_model", None)
    if callable(builder):
        try:
            return builder(model_cfg)
        except Exception:  # noqa: BLE001
            pass
    cls = getattr(module, class_name or "DrivingModel", None)
    if cls is None:
        return None
    experts_cfg = (model_cfg.get("moe", {}) or {}).get("experts", {}) or {}
    hidden = int(model_cfg.get("hidden_dim", 128))
    experts = int(experts_cfg.get("count", 8))
    expert_hidden = int(experts_cfg.get("hidden_dim", 256))
    wm_steps = int((model_cfg.get("world_model", {}) or {}).get("rollout_steps", 6))
    for kwargs in (
        {"hidden": hidden, "num_experts": experts, "expert_hidden": expert_hidden, "wm_steps": wm_steps},
        {"hidden": hidden, "num_experts": experts, "expert_hidden": expert_hidden},
        {"hidden": hidden, "num_experts": experts, "wm_steps": wm_steps},
        {"hidden": hidden, "num_experts": experts},
        {},
    ):
        try:
            return cls(**kwargs)
        except Exception:  # noqa: BLE001
            continue
    return None


def _rss_mb() -> Tuple[float, float]:
    """返回 ``(当前 RSS, 峰值 RSS)`` MB。"""
    current = 0.0
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    current = float(line.split()[1]) / 1024.0
                    break
    except Exception:  # noqa: BLE001
        pass
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return current, peak


def trim_memory() -> None:
    """把 glibc 空闲 arena 归还操作系统（PyTorch CPU 自动求导后 RSS 回落的关键）。

    实证：同一 mini-batch 反复 forward/backward，不 trim 时 RSS 高水位可达 ~2.5GB（缓存
    分配器 + glibc arena 不归还），``gc.collect() + malloc_trim(0)`` 后稳定在 ~0.4GB。
    非 glibc 平台静默 no-op。训练循环每个 update 后调用一次。
    """
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _load_train_section(path: str = "config/train.yaml") -> Dict[str, Any]:
    """读 ``config/train.yaml`` 的 ``train`` 段（冒烟用；文件缺失/不可解析返回空 dict）。"""
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        section = payload.get("train")
        return dict(section) if isinstance(section, Mapping) else {}
    except Exception:  # noqa: BLE001
        return {}


def _parse_smoke_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="pipeline.trainer --smoke", description="PPO 冒烟（straight/curve 切片）")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--specs", default="env/specs/scenarios_train_slice200.json")
    parser.add_argument("--split", choices=("all", "train", "val"), default="train")
    parser.add_argument("--geometry", default="straight,curve", help="主几何标签过滤（逗号分隔；空=不过滤）")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--pool", choices=("auto", "vector", "local"), default="auto")
    parser.add_argument("--envs", type=int, default=2, help="resident env 数（>1 需 pipeline.vector_env 子进程池）")
    parser.add_argument("--tracker", choices=("kinematic", "exact", "lqr"), default="kinematic")
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--minutes", type=float, default=2.0, help="墙钟预算（验收 ≤2 分钟）")
    parser.add_argument("--minibatch-size", type=int, default=None, help="默认取 config/train.yaml train.ppo.minibatch_size")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="默认取 config/train.yaml train.device（auto=cuda 可用则 cuda）")
    parser.add_argument("--reward-config", default="{}", help="奖励配置 JSON（terms/aggregation 覆盖）")
    parser.add_argument("--model", default="net.model:DrivingModel")
    parser.add_argument("--mem-floor-mb", type=float, default=None, help="创建子进程池前要求的最小 MemAvailable")
    parser.add_argument("--malloc-trim", action="store_true", default=True, help="每 update 后 gc+malloc_trim 回落 RSS")
    parser.add_argument("--no-malloc-trim", dest="malloc_trim", action="store_false")
    parser.add_argument("--max-steps-per-episode", type=int, default=600)
    return parser.parse_args(argv)


def _load_specs_filtered(args: argparse.Namespace) -> List[Any]:
    from env.scenario.spec import load_specs

    specs = load_specs(args.specs)
    if args.split != "all":
        specs = [spec for spec in specs if getattr(spec, "split", None) == args.split]
    wanted = {item.strip() for item in str(args.geometry).split(",") if item.strip()}
    if wanted:
        specs = [spec for spec in specs if str(getattr(spec, "labels", {}).get("geometry", "")) in wanted]
    if args.limit is not None:
        specs = specs[: max(1, int(args.limit))]
    return specs


def smoke_main(argv: Optional[Sequence[str]] = None) -> int:
    """PPO 冒烟：真实模型/奖励/池（不可用时文档化回退），打印 steps/s、loss、RSS、VRAM。"""
    args = _parse_smoke_args(argv)
    started = time.perf_counter()
    specs = _load_specs_filtered(args)
    if not specs:
        print("[smoke] 没有匹配的 spec（检查 --specs/--geometry/--limit）", flush=True)
        return 2
    train_cfg = _load_train_section()
    device = resolve_device(args.device, {"train": train_cfg})
    threads = apply_thread_limits(workers=1, config={"train": train_cfg})
    train_ppo_cfg = dict((train_cfg.get("ppo", {}) or {}))
    minibatch_size = int(args.minibatch_size or train_ppo_cfg.get("minibatch_size") or 1024)
    print(
        f"[smoke] specs={len(specs)} pool={args.pool} envs={args.envs} tracker={args.tracker} "
        f"rollout={args.rollout_steps} updates<={args.updates} budget<={args.minutes}min "
        f"device={device} threads={threads} minibatch={minibatch_size}",
        flush=True,
    )
    pool = None
    try:
        pool = build_pool(
            specs,
            kind=str(args.pool),
            num_envs=int(args.envs),
            tracker=str(args.tracker),
            max_episode_steps=int(args.max_steps_per_episode),
            mem_floor_mb=args.mem_floor_mb,
            logger=print,
        )
        initial = pool.reset()
        obs_template = PPOTrainer._current_template(initial[0]["obs"])
        print(
            f"[smoke] obs channels: {[(k, tuple(np.asarray(v).shape)) for k, v in sorted(obs_template.items())]}",
            flush=True,
        )
        model_cfg: Dict[str, Any] = {}
        try:
            import yaml

            with open(_MODEL_CONFIG_DEFAULT, "r", encoding="utf-8") as handle:
                model_cfg = yaml.safe_load(handle) or {}
        except Exception:  # noqa: BLE001
            pass
        model = _try_build_real_model(args.model, model_cfg)
        if model is None:
            print(f"[smoke] NOTE: {args.model} 构造失败 → 使用文档化 stub 模型（接口同 p2-contract §2）", flush=True)
            model = _SmokeModel(obs_template)
        param_count = sum(p.numel() for p in model.parameters())
        try:
            reward_cfg = json.loads(str(args.reward_config) or "{}")
            if not isinstance(reward_cfg, dict):
                raise ValueError("必须是 JSON 对象")
        except Exception as exc:  # noqa: BLE001
            print(f"[smoke] --reward-config 解析失败（{exc}）→ 用默认奖励项", flush=True)
            reward_cfg = {}
        reward_adapter, reward_source = build_reward_adapter(reward_cfg, logger=print)
        print(f"[smoke] model={type(model).__name__} params={param_count} reward={reward_source}", flush=True)

        config = PPOConfig(
            lr=float(args.lr),
            epochs=int(args.epochs),
            minibatch_size=int(minibatch_size),
            load_balance_coef=0.0,
            seed=int(args.seed),
            device=str(device),
        )
        probe_cfg = train_cfg.get("probe_batch", DEFAULT_PROBE_BATCH)
        probe_batch = None if probe_cfg is None else str(probe_cfg)
        trainer = PPOTrainer(model, pool, config, reward_adapter=reward_adapter, probe_batch=probe_batch)
        trainer.adopt_obs([record["obs"] for record in initial])
        use_cuda = str(device).startswith("cuda") and torch is not None and torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        deadline = started + float(args.minutes) * 60.0
        updates = 0
        collected = 0
        last_metrics: Dict[str, Any] = {}
        while updates < int(args.updates) and time.perf_counter() < deadline:
            t0 = time.perf_counter()
            collected += trainer.collect_rollout(int(args.rollout_steps))
            t_collect = time.perf_counter() - t0
            t1 = time.perf_counter()
            metrics = trainer.update()
            t_update = time.perf_counter() - t1
            updates += 1
            last_metrics = metrics
            if bool(args.malloc_trim):
                trim_memory()
            current_rss, peak_rss = _rss_mb()
            workers = ""
            if hasattr(pool, "stats"):
                stats = pool.stats()
                worker_rss = stats.get("workers") or []
                if worker_rss:
                    workers = f" workers_rss={[round(float(w.get('rss_mb') or 0), 0) for w in worker_rss]}MB"
            collect_timing = trainer._last_collect_timing or {}
            update_timing = metrics.get("update_timing_s") or {}
            vram = ""
            if use_cuda:
                vram = f" vram_peak={torch.cuda.max_memory_allocated() / 2**20:.0f}MB"
            print(
                f"[smoke] update {updates:>3} steps={collected:>6} "
                f"steps/s={int(args.rollout_steps) * int(getattr(pool, 'num_envs', 1)) / max(t_collect, 1e-9):7.1f} "
                f"loss={metrics.get('total_loss', float('nan')):+.4f} pi={metrics.get('policy_loss', float('nan')):+.4f} "
                f"v={metrics.get('value_loss', float('nan')):.4f} ent={metrics.get('entropy', float('nan')):.3f} "
                f"kl={metrics.get('approx_kl', float('nan')):+.4f} clip={metrics.get('clipfrac', float('nan')):.3f} "
                f"rss={current_rss:.0f}MB peak={peak_rss:.0f}MB{workers}{vram} "
                f"collect={t_collect:.2f}s(policy={collect_timing.get('policy_s', 0.0):.2f} env={collect_timing.get('env_step_s', 0.0):.2f}) "
                f"update={t_update:.2f}s(data={update_timing.get('data_s', 0.0):.2f} fwd={update_timing.get('forward_s', 0.0):.2f} bwd={update_timing.get('backward_s', 0.0):.2f})",
                flush=True,
            )
        wall = time.perf_counter() - started
        current_rss, peak_rss = _rss_mb()
        vram = ""
        if use_cuda:
            vram = f" vram_peak={torch.cuda.max_memory_allocated() / 2**20:.0f}MB"
        print(
            f"[smoke] DONE updates={updates} steps={collected} wall={wall:.1f}s "
            f"steps/s={collected / max(wall, 1e-9):.1f} rss={current_rss:.0f}MB peak={peak_rss:.0f}MB{vram}",
            flush=True,
        )
        if last_metrics:
            printable = {
                key: (round(value, 5) if isinstance(value, float) else value)
                for key, value in last_metrics.items()
                if key not in ("router_mean_weight", "router_active_rate", "router_mean_output_norm")
            }
            print(f"[smoke] last metrics: {json.dumps(printable, ensure_ascii=False)}", flush=True)
        return 0
    finally:
        if pool is not None:
            try:
                pool.close()
            except Exception:  # noqa: BLE001
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    """入口：``--smoke`` 走 :func:`smoke_main`；阶段编排见 ``pipeline.stages``。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--smoke" in argv:
        return smoke_main(argv)
    print(
        "pipeline.trainer：`python -m pipeline.trainer --smoke ...` 运行 PPO 冒烟；阶段编排见 pipeline.stages。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
