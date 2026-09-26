#!/usr/bin/env python3
"""训练监控：tensorboard + CSV，hook 式、非侵入（P2/N5，契约 §5）。

Tier-1 瘦身（2026-09-27，默认口径；完整清单见 ``docs/metrics.md``）
--------------------------------------------------------------------
用户只关心 **OD/EGO 的 loss 与 KPI + router 的 loss 与 KPI**，其余不再记录：

- 训练侧照旧喂 canonical tag（``train/wm_loss`` / ``train/primary_bc_router_soft_ce`` ...），
  monitor 落盘前按 :func:`_slim_tag` **重命名**为保留清单（``wm/...`` / ``ego/...`` /
  ``router/...`` / ``stageB/...``）；**未列入的 tag 直接丢弃**（TB/CSV 都不写）：
  ``slice/*``、``label/*``、``*/n_updates``、计数/权重和、``cv_ade``/``cv_fde`` 独立 tag、
  ``traj_mse_m2`` 曲线、``expert_util_*``/``expert_mix_util_*``、``grad_norm_*``、
  计时/显存、动作误差 median/p95、``kpi/*`` / ``moe/*`` / ``scene_label/*`` 等；
- ``legacy_tags=True``（CLI ``--monitor-legacy-tags``，默认关）= **旧行为原样落盘**
  （全部 tag + 旧多线分组），供需要时回退/对比；
- tensorboard 写入只服务成图：多线族 ``add_scalars(main, {sub: value})``（``wm/od/loss``
  h1..h6、``wm/od/ade_m`` h1..h6+cv_h1..cv_h6、``wm/ego_next/loss`` h1..h6、
  ``ego/traj/mae_m`` h1..h6、``router/<phase>/expert_mix_weight`` e0..e7、
  ``stageB/<phase>/loss_terms`` loss/traj/action/router；``val_`` 前缀同族各一图），
  标量（``wm/loss`` / ``wm/presence_auc`` / ``wm/entry_auc`` / ``ego/action/err_weighted`` /
  ``router/soft_*`` / ``router/top1_cluster_acc|nmi|entropy``）写单线 ``add_scalar``；
  **canonical 单 tag 不再额外写一份**（旧行为是单点+分组双写）。注意 torch 的
  ``add_scalars`` 语义：每个 sub 写成独立 sub-run 事件文件 ``<log_dir>/<main>_<sub>/``，
  文件内 tag = ``main_tag``，因此 tensorboard 标量面板里同一 tag 的多 run 即多线同图；
  同族 <2 个 sub 不写（避免单点噪声）；
- CSV 长表 ``step,tag,value`` 保留**全部**瘦身后 tag（供 ``tools/il_report.py`` /
  ``tools/plot_curves.py`` 读取）。

职责（legacy 模式下全部生效）
-----------------------------
1. **KPI 序列**：``on_episode`` / ``on_train_step`` 把 episode KPI 与训练指标写成
   ``kpi/...`` / ``train/...`` 标量（tensorboard + ``metrics.csv``）；
2. **场景 label 统计**：``on_scene_step`` 逐 step 接收 ``env.scenario.labels.compute_step_labels``
   的输出（纯 dict，监控不碰 env），累计各类触发频率；
3. **MoE 路由统计**：``on_moe_step`` 接收 ``router_weights`` / 可选 ``expert_outputs`` /
   ``primary_output``，统计 **有效专家数 N=Σw**、每 expert 平均权重、输出 L2 范数、
   token 数（w > 阈值）、**primary 漂移**（``‖Σ w_e·expert_e‖ / (‖primary‖+ε)``，
   即 MoE 相对纯 primary 输出的残差占比）；
   ⚠️ ``router_weights`` 的口径是 **router logits 的 softmax 全专家概率分布**
   （v2 监督面；``Σw ∈ [1,E]``，反映 gate 分布/熵），**不是** net 输出的
   ``expert_weights``（top-2 混合权重：恰 2 个非零、每 token 和为 1，``Σw ≡ 1``）。
   传 ``expert_weights`` 只适合看 top-2 利用率，effective_n 恒为 1；两者都要时分开两次
   ``on_moe_step``（monitor 会累计，但语义不同，推荐默认传 softmax 分布）；
4. **分组指标**：``on_grouped_step`` 接收 per-horizon / per-label / 分切片三组窗口统计
   （``GroupedMetricStatistics``），flush 时输出 ``horizon/<h>/<m>``、
   ``label/<name>/<m>``、``slice/<name>/<m>`` 的 ``mean``/``n_updates``/``weighted_mean``。
   ``n_updates`` = 该均值由几次 ``update`` 贡献合成（**不是样本数**）。瘦身模式下上述
   tag 经 :func:`_slim_tag` 过滤：Stage A 的 ``horizon/h*/loss|ade|cv_ade|ego_next_loss`` →
   ``wm/...`` 保留族，Stage B 的 ``horizon/h*/traj_mae_m`` → ``ego/traj/mae_m`` 保留族；
   slice/label/计数/`fde`/`traj_mse_m2` 等全部丢弃（形参保留仅为 legacy 兼容）；
5. **train/val 命名（Stage B）**：``on_train_step`` 写 ``train/<name>``；留出集用
   ``on_val_step`` 写 ``val/<name>``（同族指标同后缀），分组指标用 ``on_val_grouped_step``
   写 ``val/horizon/...`` / ``val/label/...`` / ``val/slice/...``。瘦身模式下 ``val/`` 族
   重命名为 ``val_`` 前缀（如 ``val_ego/traj/mae_m``），train/val 各一图；
6. **PPO 诊断序列**（嵌套 dict，如 ``reward/...``、``advantage/...``、``probe/...``）：
   ``log_scalars`` 递归展平嵌套 Mapping → 标签 ``a/b/c``；瘦身模式下未列入清单的 tag 丢弃；
7. 输出：``<log_dir>/metrics.csv``（长表 ``step,tag,value``）与 tensorboard event 文件。

设计
----
- **hook 式**：训练循环只调 ``on_*`` 方法（或 ``log_scalar``），监控不修改 env/模型/优化器，
  也不持有它们的引用；tensorboard 不可用时自动退化为 CSV（打一条 warning，不 raise）；
- **非侵入**：输入张量只读（``torch.no_grad()``），支持 torch.Tensor / numpy / list；
  统计窗口在 ``flush(step)`` 后重置（窗口内均值），避免长跑时数值被早期样本稀释；
- 本模块 import 时**不导入 torch/tensorboard**（惰性），纯 CSV 用法无重依赖。

用法
----
    monitor = TrainingMonitor("runs/train/stageA/monitor")            # Tier-1 瘦身口径
    monitor = TrainingMonitor("runs/train/stageA/monitor", legacy_tags=True)  # 回退旧口径
    ...
    monitor.on_scene_step(compute_step_labels(env, spec))     # 每步（可降频）
    # 口径：router_weights = router_logits 的 softmax 全专家分布（监督面；Σw∈[1,E]）。
    # 不要传 net 的 expert_weights（top-2 混合，Σw≡1）；两者语义不同，见 MoERoutingStatistics。
    monitor.on_moe_step(router_weights=out["router_logits"].softmax(dim=-1),
                        expert_outputs=expert_outs, primary_output=primary_out)
    monitor.on_episode(episode_kpi, step=train_step)
    monitor.on_train_step({"loss": loss.item(), "lr": lr}, step=train_step)
    monitor.flush(train_step)                                 # 写窗口统计 + 多线图
    ...
    monitor.close()
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Mapping as _MappingABC
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["TrainingMonitor", "SceneLabelStatistics", "MoERoutingStatistics", "GroupedMetricStatistics"]

_SCENE_LABEL_PREFIX = "scene_label"
_MOE_PREFIX = "moe"
_KPI_PREFIX = "kpi"
_TRAIN_PREFIX = "train"
_VAL_PREFIX = "val"
_HORIZON_PREFIX = "horizon"
_LABEL_PREFIX = "label"
_SLICE_PREFIX = "slice"
#: 瘦身模式下 ``val/`` 前缀族的 canonical main 前缀（``val/ego/...`` → ``val_ego/...``）
_VAL_TAG_PREFIX = "val_"


def _as_array(value: Any) -> Optional[np.ndarray]:
    """把 torch.Tensor / numpy / list 只读转为 float64 numpy 数组；None → None。"""
    if value is None:
        return None
    if hasattr(value, "detach"):  # torch.Tensor
        try:
            import torch

            with torch.no_grad():
                value = value.detach().cpu().numpy()
        except Exception:  # noqa: BLE001 - 转换失败按普通序列处理
            pass
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def _finite(value: Any) -> Optional[float]:
    """转 float；NaN/Inf/非数值 → None。"""
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _flatten_scalars(values: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    """递归展平嵌套 Mapping：``{"a": {"b": 1}}`` → ``{"a/b": 1}``。

    非 Mapping 值（标量/列表/数组）原样返回，由 :func:`_finite` 过滤；这样训练侧可以
    用嵌套 dict 组织指标（奖励分解 / 优势统计 / 探针），无需手写扁平标签。
    """
    flat: Dict[str, Any] = {}
    for tag, value in values.items():
        name = f"{prefix}{tag}"
        if isinstance(value, _MappingABC):
            flat.update(_flatten_scalars(value, prefix=f"{name}/"))
        else:
            flat[name] = value
    return flat


# --------------------------------------------------------------------------- #
# Tier-1 瘦身：canonical tag 重命名 + tensorboard 多线族（2026-09-27）
# --------------------------------------------------------------------------- #
# 默认（``legacy_tags=False``）：训练侧照旧喂入 canonical tag，落盘前按 :func:`_slim_tag`
# 重命名；**未列入**的 tag 直接丢弃（TB/CSV 都不写）。legacy 模式 = 旧行为（原样落盘 +
# ``_group_of_tag`` 旧族多线图），仅由 ``TrainingMonitor(legacy_tags=True)`` 启用。
#
# 瘦身后的 canonical tag 形如 ``<main>/<sub>``（多线族）或独立标量名；tensorboard 只写
# 两类：多线族 ``add_scalars(main, {sub: value})`` + 标量 ``add_scalar(tag, value)``。

#: 旧 canonical tag → 瘦身后独立标量（逐 key 直查）
_SLIM_DIRECT: Dict[str, str] = {
    f"{_TRAIN_PREFIX}/wm_loss": "wm/loss",
    f"{_TRAIN_PREFIX}/presence_auc": "wm/presence_auc",
    f"{_TRAIN_PREFIX}/entry_auc": "wm/entry_auc",
}

#: 逐 horizon 分组窗口 tag（Stage A OD/ADE/ego_next + Stage B ego traj MAE）
#: ``horizon/h{k}/{metric}/mean`` → ``(main, sub)``；``val/`` 旧前缀 → ``val_`` 新 main 前缀。
_HORIZON_RE = re.compile(r"^(?P<val>%s/)?%s/h(?P<k>\d+)/(?P<metric>[^/]+)/mean$" % (_VAL_PREFIX, _HORIZON_PREFIX))
_HORIZON_RENAMES: Dict[str, Tuple[str, str]] = {
    "loss": ("wm/od/loss", "h{k}"),                    # Stage A：WM 对 OD 未来位置预测损失
    "ade": ("wm/od/ade_m", "h{k}"),                    # Stage A：OD ADE（米）
    "cv_ade": ("wm/od/ade_m", "cv_h{k}"),              # Stage A：匀速基线 ADE（同一族另一组线）
    "ego_next_loss": ("wm/ego_next/loss", "h{k}"),     # Stage A：plan head 下一时刻 ego 损失
    "traj_mae_m": ("ego/traj/mae_m", "h{k}"),          # Stage B：ego 6 点轨迹逐 horizon 加权 MAE
}

#: Stage B 相位标量（``train|val/<phase>_bc_*``）；``{phase}`` 插值，val 前缀另加 ``val_``
_BC_RE = re.compile(r"^(?P<phase>primary|specific)_(?P<key>.+)$")
_BC_SCALAR_RENAMES: Dict[str, str] = {
    "bc_loss": "stageB/{phase}/loss_terms/loss",
    "bc_traj_loss": "stageB/{phase}/loss_terms/traj",
    "bc_action_loss": "stageB/{phase}/loss_terms/action",
    "bc_router_loss": "stageB/{phase}/loss_terms/router",
    "bc_action_err_weighted_mean": "ego/action/err_weighted",
    "bc_router_soft_ce": "router/soft_ce",
    "bc_router_soft_kl": "router/soft_kl",
    "bc_router_top1_cluster_acc": "router/top1_cluster_acc",
    "bc_router_nmi": "router/nmi",
    "bc_router_entropy": "router/entropy",
}
_BC_EXPERT_RE = re.compile(r"^bc_router_expert_mix_weight_(?P<index>\d+)$")


def _slim_tag(tag: str) -> Optional[str]:
    """旧 canonical tag → 瘦身后 tag；不属于 Tier-1 保留清单 → None（TB/CSV 都不写）。"""
    direct = _SLIM_DIRECT.get(tag)
    if direct is not None:
        return direct
    match = _HORIZON_RE.match(tag)
    if match:
        rename = _HORIZON_RENAMES.get(match.group("metric"))
        if rename is None:
            return None
        main, sub = rename
        prefix = _VAL_TAG_PREFIX if match.group("val") else ""
        return f"{prefix}{main}/{sub.format(k=match.group('k'))}"
    if tag.startswith(f"{_TRAIN_PREFIX}/"):
        prefix, name = "", tag[len(_TRAIN_PREFIX) + 1:]
    elif tag.startswith(f"{_VAL_PREFIX}/"):
        prefix, name = _VAL_TAG_PREFIX, tag[len(_VAL_PREFIX) + 1:]
    else:
        return None
    match = _BC_RE.match(name)
    if match is None:
        return None
    phase, key = match.group("phase"), match.group("key")
    if key in _BC_SCALAR_RENAMES:
        return f"{prefix}{_BC_SCALAR_RENAMES[key].format(phase=phase)}"
    match = _BC_EXPERT_RE.match(key)
    if match is not None:
        return f"{prefix}router/{phase}/expert_mix_weight/e{match.group('index')}"
    return None


#: 瘦身模式 tensorboard 多线族：``main`` 前缀 → 合法 sub 正则（``val_`` 前缀自动同族成图）
_TB_FAMILIES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("wm/od/loss", re.compile(r"^h[1-6]$")),
    ("wm/od/ade_m", re.compile(r"^(cv_)?h[1-6]$")),
    ("wm/ego_next/loss", re.compile(r"^h[1-6]$")),
    ("ego/traj/mae_m", re.compile(r"^h[1-6]$")),
    ("router/primary/expert_mix_weight", re.compile(r"^e[0-7]$")),
    ("router/specific/expert_mix_weight", re.compile(r"^e[0-7]$")),
    ("stageB/primary/loss_terms", re.compile(r"^(loss|traj|action|router)$")),
    ("stageB/specific/loss_terms", re.compile(r"^(loss|traj|action|router)$")),
)


def _slim_group_of_tag(tag: str) -> Optional[Tuple[str, str]]:
    """瘦身后 tag → ``(main_tag, sub_tag)``；独立标量 → None（写单线 ``add_scalar``）。"""
    prefix, name = "", tag
    if tag.startswith(_VAL_TAG_PREFIX):
        prefix, name = _VAL_TAG_PREFIX, tag[len(_VAL_TAG_PREFIX):]
    for main, sub_re in _TB_FAMILIES:
        if name.startswith(f"{main}/"):
            sub = name[len(main) + 1:]
            if sub_re.match(sub):
                return f"{prefix}{main}", sub
    return None


# --------------------------------------------------------------------------- #
# legacy 模式：旧 canonical tag → 旧 (main_tag, sub_tag) 分组（2026-09-27 前的行为）
# --------------------------------------------------------------------------- #
_LEGACY_HORIZON_FAMILIES: Dict[str, str] = {
    "loss": "horizon_loss",
    "ade": "horizon_ade",
    "cv_ade": "horizon_cv_ade",
    "ego_next_loss": "horizon_ego_next",
    "traj_mse_m2": "horizon_traj_mse_m2",
    "traj_mae_m": "horizon_traj_mae_m",
}
_LEGACY_STAGE_A_WM_TERMS = ("wm_loss", "wm_loss_od", "wm_loss_ego_next", "presence_loss", "entry_loss")
_LEGACY_STAGE_A_HEALTH = ("presence_auc", "entry_auc")
_LEGACY_BC_TERM_SUBS = {"loss": "loss", "traj_loss": "traj", "action_loss": "action", "router_loss": "router"}

_LEGACY_HORIZON_GROUP_RE = re.compile(r"^horizon/(?P<group>h\d+)/(?P<metric>[^/]+)/mean$")
_LEGACY_SLICE_GROUP_RE = re.compile(r"^slice/(?P<group>[^/]+)/action_err/mean$")
_LEGACY_LABEL_GROUP_RE = re.compile(r"^label/(?P<group>[^/]+)/action_err/mean$")
_LEGACY_EXPERT_MIX_RE = re.compile(
    r"^(?P<phase>[A-Za-z][A-Za-z0-9_]*)_bc_router_expert_mix_weight_(?P<index>\d+)$"
)
_LEGACY_BC_TERM_RE = re.compile(
    r"^(?P<phase>[A-Za-z][A-Za-z0-9_]*)_bc_(?P<term>loss|traj_loss|action_loss|router_loss)$"
)
_LEGACY_GRAD_NORM_RE = re.compile(r"^grad_norm_(?P<name>.+)$")


def _group_of_tag(tag: str) -> Optional[Tuple[str, str]]:
    """legacy：单 tag → 旧 ``(main_tag, sub_tag)``；不属于任何分组族 → None。"""
    prefix = ""
    name = tag
    if tag.startswith(f"{_VAL_PREFIX}/"):
        prefix, name = f"{_VAL_PREFIX}_", tag[len(_VAL_PREFIX) + 1:]
    elif tag.startswith(f"{_TRAIN_PREFIX}/"):
        name = tag[len(_TRAIN_PREFIX) + 1:]

    match = _LEGACY_HORIZON_GROUP_RE.match(name)
    if match:
        family = _LEGACY_HORIZON_FAMILIES.get(match.group("metric"))
        return None if family is None else (f"{prefix}{family}", match.group("group"))
    match = _LEGACY_SLICE_GROUP_RE.match(name)
    if match:
        return f"{prefix}slice_action_err", match.group("group")
    match = _LEGACY_LABEL_GROUP_RE.match(name)
    if match:
        return f"{prefix}label_action_err", match.group("group")
    match = _LEGACY_EXPERT_MIX_RE.match(name)
    if match:
        return f"{prefix}expert_mix_weight_{match.group('phase')}", f"e{match.group('index')}"
    match = _LEGACY_BC_TERM_RE.match(name)
    if match:
        return f"{prefix}bc_terms_{match.group('phase')}", _LEGACY_BC_TERM_SUBS[match.group("term")]
    if not prefix:  # Stage A 规则只在 train/ 前缀下生效（val/ 无同族）
        if name in _LEGACY_STAGE_A_WM_TERMS:
            return "wm_terms", name
        if name in _LEGACY_STAGE_A_HEALTH:
            return "health", name
        match = _LEGACY_GRAD_NORM_RE.match(name)
        if match:
            return "grad_norm", match.group("name")
    return None


def _grouped_scalars(
    scalars: Mapping[str, Any],
    *,
    group_fn: Any = _slim_group_of_tag,
    min_subs: int = 2,
) -> Dict[str, Dict[str, float]]:
    """扁平 ``tag → value`` → ``{main_tag: {sub: value}}``（仅保留 ≥``min_subs`` 个 sub 的族）。

    供 :meth:`TrainingMonitor.flush` 用 ``add_scalars`` 多线成图；瘦身模式默认按
    :func:`_slim_group_of_tag` 归类（``val_`` 前缀同族各一图），legacy 模式传
    :func:`_group_of_tag`。
    """
    groups: Dict[str, Dict[str, float]] = {}
    for raw_tag, raw_value in scalars.items():
        number = _finite(raw_value)
        if number is None:
            continue
        identified = group_fn(str(raw_tag))
        if identified is None:
            continue
        main_tag, sub_tag = identified
        groups.setdefault(main_tag, {})[sub_tag] = number
    return {main: subs for main, subs in groups.items() if len(subs) >= min_subs}


class SceneLabelStatistics:
    """逐步场景标签的窗口累计统计（触发频率）。"""

    def __init__(self, label_names: Optional[Sequence[str]] = None, *, active_threshold: float = 0.5):
        self.label_names: List[str] = list(label_names) if label_names else []
        self.active_threshold = float(active_threshold)
        self.reset()

    def update(self, labels: Mapping[str, Any]) -> None:
        """累计一步标签（0/1 或概率，> 阈值记 positive）。"""
        for name, value in labels.items():
            number = _finite(value)
            if number is None:
                continue
            key = str(name)
            if key not in self.label_names:
                self.label_names.append(key)
            entry = self._counts.setdefault(key, {"steps": 0, "positive": 0, "sum": 0.0})
            entry["steps"] += 1
            entry["sum"] += number
            if number > self.active_threshold:
                entry["positive"] += 1

    def flush(self) -> Dict[str, float]:
        """返回 ``{tag: value}``（频率 + 均值），并清空窗口。"""
        out: Dict[str, float] = {}
        for name in self.label_names:
            entry = self._counts.get(name)
            if not entry or entry["steps"] <= 0:
                continue
            out[f"{_SCENE_LABEL_PREFIX}/{name}/freq"] = entry["positive"] / entry["steps"]
            out[f"{_SCENE_LABEL_PREFIX}/{name}/mean"] = entry["sum"] / entry["steps"]
            out[f"{_SCENE_LABEL_PREFIX}/{name}/steps"] = float(entry["steps"])
        self.reset()
        return out

    def reset(self) -> None:
        self._counts: Dict[str, Dict[str, float]] = {}


class MoERoutingStatistics:
    """MoE 路由窗口统计（有效 N / 每 expert 权重 / 输出范数 / token 数 / primary 漂移）。

    ``router_weights`` 的规范口径 = **router logits 的 softmax 全专家概率**（v2 监督面，
    ``Σw ∈ [1,E]``）；top-2 混合权重（``net.moe`` 的 ``expert_weights``，恰 2 个非零、
    每 token 和为 1）**不是**本口径 —— 传它只会得到 ``effective_n ≡ 1``（见模块 docstring）。
    """

    def __init__(self, *, token_threshold: float = 0.5, eps: float = 1e-6):
        self.token_threshold = float(token_threshold)
        self.eps = float(eps)
        self.reset()

    def update(
        self,
        router_weights: Any,
        *,
        expert_outputs: Any = None,
        primary_output: Any = None,
    ) -> None:
        """累计一次前向（可多次调用再统一 flush）。

        Args:
            router_weights: (B, E) 或 (E,) **router logits 的 softmax 全专家概率**
                （规范口径；E = expert 数）；top-2 混合权重（``expert_weights``）语义不同，
                不要混用（Σw≡1 会让 effective_n 失去意义）。
            expert_outputs: 可选 (B, E, H)：每 expert 输出（用于范数/漂移）。
            primary_output: 可选 (B, H)：primary 输出（用于 primary 漂移）。
        """
        weights = _as_array(router_weights)
        if weights is None:
            return
        if weights.ndim == 1:
            weights = weights[None, :]
        if weights.ndim != 2:
            return
        num_tokens, num_experts = int(weights.shape[0]), int(weights.shape[1])
        self._n_tokens += num_tokens
        self._n_calls += 1
        self._effective_n_sum += float(weights.sum(axis=1).mean())

        if self._expert_weight_sum is None:
            self._expert_weight_sum = np.zeros(num_experts, dtype=np.float64)
            self._expert_tokens = np.zeros(num_experts, dtype=np.int64)
        self._expert_weight_sum += weights.sum(axis=0)
        self._expert_tokens += (weights > self.token_threshold).sum(axis=0)

        outputs = _as_array(expert_outputs)
        primary = _as_array(primary_output)
        if outputs is not None and outputs.ndim == 3 and outputs.shape[0] == num_tokens:
            if self._expert_norm_sum is None:
                self._expert_norm_sum = np.zeros(int(outputs.shape[1]), dtype=np.float64)
                self._expert_norm_count = 0
            norms = np.linalg.norm(outputs, axis=-1)  # (B, E)
            self._expert_norm_sum += norms.sum(axis=0)
            self._expert_norm_count += num_tokens
            if primary is not None and primary.ndim == 2 and primary.shape[0] == num_tokens \
                    and outputs.shape[1] == num_experts:
                residual = np.einsum("be,beh->bh", weights, outputs)
                residual_norm = np.linalg.norm(residual, axis=-1)
                primary_norm = np.linalg.norm(primary, axis=-1)
                drift = residual_norm / (primary_norm + self.eps)
                self._drift_sum += float(np.mean(drift))
                self._drift_calls += 1
        elif primary is not None and primary.ndim == 2:
            self._primary_norm_sum += float(np.linalg.norm(primary, axis=-1).mean())
            self._primary_norm_calls += 1

    def flush(self) -> Dict[str, float]:
        """返回窗口统计 ``{tag: value}``，并清空窗口。"""
        if self._n_calls <= 0 or self._expert_weight_sum is None:
            return {}
        calls = float(self._n_calls)
        out: Dict[str, float] = {
            f"{_MOE_PREFIX}/effective_n": self._effective_n_sum / calls,
            f"{_MOE_PREFIX}/n_tokens": float(self._n_tokens),
        }
        for index in range(self._expert_weight_sum.shape[0]):
            out[f"{_MOE_PREFIX}/expert_{index}/weight"] = float(self._expert_weight_sum[index]) / max(1, self._n_tokens)
            out[f"{_MOE_PREFIX}/expert_{index}/tokens"] = float(self._expert_tokens[index])
            if self._expert_norm_sum is not None and self._expert_norm_count > 0:
                out[f"{_MOE_PREFIX}/expert_{index}/out_norm"] = (
                    float(self._expert_norm_sum[index]) / self._expert_norm_count
                )
        if self._drift_calls > 0:
            out[f"{_MOE_PREFIX}/primary_drift"] = self._drift_sum / self._drift_calls
        if self._primary_norm_calls > 0:
            out[f"{_MOE_PREFIX}/primary_out_norm"] = self._primary_norm_sum / self._primary_norm_calls
        self.reset()
        return out

    def reset(self) -> None:
        self._n_tokens = 0
        self._n_calls = 0
        self._effective_n_sum = 0.0
        self._expert_weight_sum: Optional[np.ndarray] = None
        self._expert_tokens: Optional[np.ndarray] = None
        self._expert_norm_sum: Optional[np.ndarray] = None
        self._expert_norm_count = 0
        self._drift_sum = 0.0
        self._drift_calls = 0
        self._primary_norm_sum = 0.0
        self._primary_norm_calls = 0


class GroupedMetricStatistics:
    """**分组指标窗口**（per-horizon / per-label / 分切片序列）。

    训练侧按 ``update({"h1": {"traj_mae_m": 2.3, "traj_mse_m2": 5.5}}`` 的形式喂入窗口统计，
    ``flush()`` 输出（CSV + tensorboard 同一套 tag；**瘦身模式下由
    :func:`_slim_tag` 重命名/过滤**，只有保留清单内的 ``/mean`` 会进 TB/CSV）：

    - ``<prefix>/<group>/<metric>/mean``：窗口内均值（算术）；
    - ``<prefix>/<group>/<metric>/n_updates``：**该均值由几次 ``update`` 贡献合成**
      （逐 epoch flush 时恒 1；旧名 ``/count`` 会被误读为样本数，2026-09-26 改名）；
    - 传入组权重时额外输出 ``/weighted_mean``（``Σ w·x/Σ w``）与 ``/weight``（权重和）。

    ``group`` 可以是任意字符串（``h1..h6`` / 标签名 / ``brake|turn|curve``）；
    非标量/NaN 值静默跳过（与 :func:`_finite` 同口径）。物理样本数（有效帧/槽位数）
    由调用方以独立标量记录（如 ``valid_samples``/``valid_weight_sum``），不复用本窗口口径。
    """

    def __init__(self, prefix: str):
        self.prefix = str(prefix)
        self.reset()

    def update(self, groups: Mapping[str, Any], *, weights: Optional[Mapping[str, Any]] = None) -> None:
        """累计一个窗口步：``groups = {group: {metric: scalar}}``（可空）。"""
        for group, metrics in (groups or {}).items():
            if not isinstance(metrics, _MappingABC):
                continue
            weight_raw = weights.get(group) if isinstance(weights, _MappingABC) else None
            weight = _finite(weight_raw)
            has_weight = weight is not None
            weight = 1.0 if weight is None else float(weight)
            for name, value in metrics.items():
                number = _finite(value)
                if number is None:
                    continue
                key = (str(group), str(name))
                entry = self._entries.setdefault(
                    key, {"n_updates": 0.0, "sum": 0.0, "wsum": 0.0, "weighted": 0.0, "weighted_flag": 0.0}
                )
                entry["n_updates"] += 1.0
                entry["sum"] += number
                entry["wsum"] += weight
                entry["weighted"] += weight * number
                entry["weighted_flag"] = max(entry["weighted_flag"], 1.0 if has_weight else 0.0)

    def flush(self) -> Dict[str, float]:
        """输出窗口统计 ``{tag: value}`` 并清空窗口。"""
        out: Dict[str, float] = {}
        for (group, name), entry in self._entries.items():
            base = f"{self.prefix}/{group}/{name}"
            out[f"{base}/mean"] = entry["sum"] / max(entry["n_updates"], 1.0)
            out[f"{base}/n_updates"] = entry["n_updates"]
            if entry["weighted_flag"] > 0.0 and entry["wsum"] > 0.0:
                out[f"{base}/weighted_mean"] = entry["weighted"] / entry["wsum"]
                out[f"{base}/weight"] = entry["wsum"]
        self.reset()
        return out

    def reset(self) -> None:
        self._entries: Dict[tuple, Dict[str, float]] = {}


class TrainingMonitor:
    """tensorboard + CSV 监控器（hook 式；见模块 docstring）。

    命名（**瘦身默认**）：训练侧 ``on_train_step`` → ``train/<name>``、留出 ``on_val_step``
    → ``val/<name>``、分组 ``on_grouped_step`` → ``horizon|label|slice/...``（**喂入侧命名
    不变**），落盘前统一按 :func:`_slim_tag` 重命名为 Tier-1 tag（``wm/...`` / ``ego/...`` /
    ``router/...`` / ``stageB/...``）；未列入清单的 tag 丢弃。``legacy_tags=True`` = 回退旧
    口径（全部 tag 原样落盘 + 旧多线分组）。
    """

    def __init__(
        self,
        log_dir: Any,
        *,
        tensorboard: bool = True,
        csv: bool = True,
        token_threshold: float = 0.5,
        scene_label_names: Optional[Sequence[str]] = None,
        legacy_tags: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_tags = bool(legacy_tags)
        self.scene_labels = SceneLabelStatistics(scene_label_names)
        self.moe = MoERoutingStatistics(token_threshold=token_threshold)
        # 分组窗口（per-horizon / per-label / 分切片）；瘦身模式下落盘前按 _slim_tag 过滤
        self.horizon = GroupedMetricStatistics(_HORIZON_PREFIX)
        self.labels = GroupedMetricStatistics(_LABEL_PREFIX)
        self.slices = GroupedMetricStatistics(_SLICE_PREFIX)
        # 留出集分组窗口（Stage B；前缀 val/ 避免与训练侧互相覆盖）
        self.val_horizon = GroupedMetricStatistics(f"{_VAL_PREFIX}/{_HORIZON_PREFIX}")
        self.val_labels = GroupedMetricStatistics(f"{_VAL_PREFIX}/{_LABEL_PREFIX}")
        self.val_slices = GroupedMetricStatistics(f"{_VAL_PREFIX}/{_SLICE_PREFIX}")
        self._csv_path: Optional[Path] = None
        self._csv_handle = None
        self._csv_writer = None
        self._writer: Any = None
        # 自上次 flush 以来记录的标量（tag → (value, step)），flush 末尾成图/写标量
        self._pending_group_tags: Dict[str, Tuple[float, Optional[int]]] = {}
        self._closed = False

        if csv:
            # 注意：``csv`` 形参遮蔽了模块名，这里显式取模块（历史 bug：``csv.writer`` 解析到 bool）
            import csv as _csv_module

            self._csv_path = self.log_dir / "metrics.csv"
            new_file = not self._csv_path.exists()
            self._csv_handle = self._csv_path.open("a", encoding="utf-8", newline="")
            self._csv_writer = _csv_module.writer(self._csv_handle)
            if new_file:
                self._csv_writer.writerow(["step", "tag", "value"])
                self._csv_handle.flush()

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.log_dir))
            except Exception as exc:  # noqa: BLE001 - tensorboard 缺失 → 仅 CSV
                print(f"[monitoring] tensorboard 不可用（{type(exc).__name__}: {exc}），仅写 CSV",
                      flush=True)
                self._writer = None

    # ------------------------------------------------------------------ 基础写
    def log_scalar(self, tag: str, value: Any, step: Optional[int] = None) -> None:
        """写单个标量并暂存到 flush 缓存（NaN/Inf/非数值静默跳过）。

        瘦身模式（默认）：非 Tier-1 tag 直接丢弃；Tier-1 tag 重命名后进 CSV + flush 缓存
        （TB 由 ``flush`` 统一写多线族/标量）。legacy 模式：原样落 CSV + 立即 ``add_scalar``。
        """
        number = _finite(value)
        if number is None:
            return
        key = str(tag)
        if not self.legacy_tags:
            key = _slim_tag(key)
            if key is None:
                return  # 非保留清单：TB/CSV 都不写（清单见 docs/metrics.md）
        self._pending_group_tags[key] = (number, step)
        if self._csv_writer is not None:
            self._csv_writer.writerow(["" if step is None else int(step), key, repr(number)])
            self._csv_handle.flush()
        if self._writer is not None and self.legacy_tags:
            self._writer.add_scalar(key, number, global_step=0 if step is None else int(step))

    def log_scalars(self, values: Mapping[str, Any], step: Optional[int] = None) -> None:
        """批量写标量（嵌套 Mapping 递归展平为 ``a/b/c`` 标签）。"""
        for tag, value in _flatten_scalars(values).items():
            self.log_scalar(str(tag), value, step=step)

    def flush(self, step: Optional[int] = None) -> None:
        """把场景标签 / MoE / 分组（训练 + 留出）窗口统计写入后端，并重置窗口。

        末尾按模式成图：瘦身模式写多线族 ``add_scalars`` + 标量 ``add_scalar``；
        legacy 模式写旧分组多线图。CSV 长表在 ``log_scalar`` 时已落盘。
        """
        self.log_scalars(self.scene_labels.flush(), step=step)
        self.log_scalars(self.moe.flush(), step=step)
        self.log_scalars(self.horizon.flush(), step=step)
        self.log_scalars(self.labels.flush(), step=step)
        self.log_scalars(self.slices.flush(), step=step)
        self.log_scalars(self.val_horizon.flush(), step=step)
        self.log_scalars(self.val_labels.flush(), step=step)
        self.log_scalars(self.val_slices.flush(), step=step)
        if self.legacy_tags:
            self._write_legacy_grouped_tensorboard(step)
        else:
            self._write_slim_tensorboard(step)
        if self._writer is not None:
            self._writer.flush()

    def _write_slim_tensorboard(self, step: Optional[int]) -> None:
        """瘦身模式：缓存里的 tag 按族写 ``add_scalars`` 多线图，其余（标量）写 ``add_scalar``。

        多线族 tag 不写 canonical 单点（族内 <2 个 sub 时整族不写，避免单点噪声；
        数值仍在 CSV）；无 tensorboard writer（或缓存为空）时只清缓存。
        """
        pending, self._pending_group_tags = self._pending_group_tags, {}
        if self._writer is None or not pending:
            return
        global_step = 0 if step is None else int(step)
        values = {tag: number for tag, (number, _) in pending.items()}
        grouped_tags = {tag for tag in pending if _slim_group_of_tag(tag) is not None}
        for main_tag, subs in _grouped_scalars(values).items():
            self._writer.add_scalars(main_tag, subs, global_step=global_step)
        for tag, (number, tag_step) in pending.items():
            if tag in grouped_tags:
                continue
            self._writer.add_scalar(
                tag, number, global_step=global_step if tag_step is None else int(tag_step)
            )

    def _write_legacy_grouped_tensorboard(self, step: Optional[int]) -> None:
        """legacy 模式：把自上次 flush 以来记录的标量按旧族写 ``add_scalars`` 多线图（仅 TB）。"""
        pending, self._pending_group_tags = self._pending_group_tags, {}
        if self._writer is None or not pending:
            return
        global_step = 0 if step is None else int(step)
        values = {tag: number for tag, (number, _) in pending.items()}
        for main_tag, subs in _grouped_scalars(values, group_fn=_group_of_tag).items():
            self._writer.add_scalars(main_tag, subs, global_step=global_step)

    # ------------------------------------------------------------------ hooks
    def on_scene_step(self, labels: Mapping[str, Any]) -> None:
        """hook：逐步场景标签（``labels.compute_labels`` 的输出；瘦身模式丢弃 ``scene_label/*``）。"""
        if labels:
            self.scene_labels.update(labels)

    def on_moe_step(
        self,
        router_weights: Any,
        *,
        expert_outputs: Any = None,
        primary_output: Any = None,
    ) -> None:
        """hook：一次 MoE 前向的路由统计（瘦身模式丢弃 ``moe/*``；参数含义见 ``MoERoutingStatistics``）。"""
        self.moe.update(router_weights, expert_outputs=expert_outputs, primary_output=primary_output)

    def on_episode(self, kpis: Mapping[str, Any], step: Optional[int] = None) -> None:
        """hook：episode KPI（写 ``kpi/<name>``；瘦身模式丢弃）。"""
        self.log_scalars({f"{_KPI_PREFIX}/{name}": value for name, value in kpis.items()}, step=step)

    def on_train_step(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """hook：训练指标（写 ``train/<name>``）。"""
        self.log_scalars({f"{_TRAIN_PREFIX}/{name}": value for name, value in metrics.items()}, step=step)

    def on_val_step(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """hook：留出集指标（写 ``val/<name>``；与 ``on_train_step`` 同族同后缀）。

        Stage B 每 epoch 末在**按 episode 留出**的 val 子集上评估后调用；训练集数字
        不写这里，避免 train/val 混淆（命名约定见模块 docstring 第 5 条）。
        """
        self.log_scalars({f"{_VAL_PREFIX}/{name}": value for name, value in metrics.items()}, step=step)

    def on_grouped_step(
        self,
        *,
        horizon: Optional[Mapping[str, Any]] = None,
        labels: Optional[Mapping[str, Any]] = None,
        slices: Optional[Mapping[str, Any]] = None,
        weights: Optional[Mapping[str, Any]] = None,
        step: Optional[int] = None,
    ) -> None:
        """hook：分组指标（per-horizon ``horizon/<h>/<m>``、per-label ``label/<name>/<m>``、
        分切片 ``slice/<name>/<m>``）；写后端并清空对应窗口。

        瘦身模式（默认）落盘前重命名：Stage A ``horizon/h*/loss|ade|cv_ade|ego_next_loss`` →
        ``wm/...``，Stage B ``horizon/h*/traj_mae_m`` → ``ego/traj/mae_m``；``labels``/``slices``
        形参仅为 legacy 兼容（瘦身模式丢弃）。``weights`` 用于 weighted_mean 口径（legacy）。
        """
        _ = step  # 窗口在 flush() 时统一落盘；保留形参以便调用方显式传步号
        if horizon:
            self.horizon.update(horizon, weights=weights)
        if labels:
            self.labels.update(labels, weights=weights)
        if slices:
            self.slices.update(slices, weights=weights)

    def on_val_grouped_step(
        self,
        *,
        horizon: Optional[Mapping[str, Any]] = None,
        labels: Optional[Mapping[str, Any]] = None,
        slices: Optional[Mapping[str, Any]] = None,
        weights: Optional[Mapping[str, Any]] = None,
        step: Optional[int] = None,
    ) -> None:
        """hook：留出集分组指标（旧 tag ``val/horizon|val/label|val/slice/...``）。

        与 :meth:`on_grouped_step` 同口径，仅前缀不同；瘦身模式下 ``val/`` 族重命名为
        ``val_`` 前缀（如 ``val_ego/traj/mae_m``），``labels``/``slices`` 丢弃（legacy 兼容）。
        """
        _ = step
        if horizon:
            self.val_horizon.update(horizon, weights=weights)
        if labels:
            self.val_labels.update(labels, weights=weights)
        if slices:
            self.val_slices.update(slices, weights=weights)

    # ------------------------------------------------------------------ 生命周期
    def close(self) -> None:
        """flush 窗口并关闭文件/SummaryWriter（幂等）。"""
        if self._closed:
            return
        try:
            self.flush()
        except Exception:  # noqa: BLE001 - 关闭阶段不应抛异常掩盖训练结果
            pass
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
            self._writer = None
        if self._csv_handle is not None:
            try:
                self._csv_handle.close()
            except Exception:  # noqa: BLE001
                pass
            self._csv_handle = None
            self._csv_writer = None
        self._closed = True

    def __enter__(self) -> "TrainingMonitor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
