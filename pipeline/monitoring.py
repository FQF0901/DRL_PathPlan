#!/usr/bin/env python3
"""训练监控：tensorboard + CSV，hook 式、非侵入（P2/N5，契约 §5）。

职责
----
1. **KPI 序列**：``on_episode`` / ``on_train_step`` 把 episode KPI 与训练指标写成
   ``kpi/...`` / ``train/...`` 标量（tensorboard + ``metrics.csv``）；
2. **场景 label 统计**：``on_scene_step`` 逐 step 接收 ``env.scenario.labels.compute_step_labels``
   的输出（纯 dict，监控不碰 env），累计各类触发频率；
3. **MoE 路由统计**：``on_moe_step`` 接收 ``router_weights`` / 可选 ``expert_outputs`` /
   ``primary_output``，统计 **有效专家数 N=Σw**、每 expert 平均权重、输出 L2 范数、
   token 数（w > 阈值）、**primary 漂移**（``‖Σ w_e·expert_e‖ / (‖primary‖+ε)``，
   即 MoE 相对纯 primary 输出的残差占比）；
4. 输出：``<log_dir>/metrics.csv``（长表 ``step,tag,value``）与 tensorboard event 文件。

设计
----
- **hook 式**：训练循环只调 ``on_*`` 方法（或 ``log_scalar``），监控不修改 env/模型/优化器，
  也不持有它们的引用；tensorboard 不可用时自动退化为 CSV（打一条 warning，不 raise）；
- **非侵入**：输入张量只读（``torch.no_grad()``），支持 torch.Tensor / numpy / list；
  统计窗口在 ``flush(step)`` 后重置（窗口内均值），避免长跑时数值被早期样本稀释；
- 本模块 import 时**不导入 torch/tensorboard**（惰性），纯 CSV 用法无重依赖。

用法
----
    monitor = TrainingMonitor("runs/train/stageA/monitor")
    ...
    monitor.on_scene_step(compute_step_labels(env, spec))     # 每步（可降频）
    monitor.on_moe_step(router_weights=out["router_logits"].sigmoid(),
                        expert_outputs=expert_outs, primary_output=primary_out)
    monitor.on_episode(episode_kpi, step=train_step)
    monitor.on_train_step({"loss": loss.item(), "lr": lr}, step=train_step)
    monitor.flush(train_step)                                 # 写窗口统计
    ...
    monitor.close()
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

__all__ = ["TrainingMonitor", "SceneLabelStatistics", "MoERoutingStatistics"]

_SCENE_LABEL_PREFIX = "scene_label"
_MOE_PREFIX = "moe"
_KPI_PREFIX = "kpi"
_TRAIN_PREFIX = "train"


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
    """MoE 路由窗口统计（有效 N / 每 expert 权重 / 输出范数 / token 数 / primary 漂移）。"""

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
            router_weights: (B, E) 或 (E,) 门控权重，E = expert 数。
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


class TrainingMonitor:
    """tensorboard + CSV 监控器（hook 式；见模块 docstring）。"""

    def __init__(
        self,
        log_dir: Any,
        *,
        tensorboard: bool = True,
        csv: bool = True,
        token_threshold: float = 0.5,
        scene_label_names: Optional[Sequence[str]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scene_labels = SceneLabelStatistics(scene_label_names)
        self.moe = MoERoutingStatistics(token_threshold=token_threshold)
        self._csv_path: Optional[Path] = None
        self._csv_handle = None
        self._csv_writer = None
        self._writer: Any = None
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
        """写单个标量（NaN/Inf/非数值静默跳过）。"""
        number = _finite(value)
        if number is None:
            return
        if self._csv_writer is not None:
            self._csv_writer.writerow(["" if step is None else int(step), str(tag), repr(number)])
            self._csv_handle.flush()
        if self._writer is not None:
            self._writer.add_scalar(str(tag), number, global_step=0 if step is None else int(step))

    def log_scalars(self, values: Mapping[str, Any], step: Optional[int] = None) -> None:
        for tag, value in values.items():
            self.log_scalar(str(tag), value, step=step)

    def flush(self, step: Optional[int] = None) -> None:
        """把场景标签 / MoE 窗口统计写入后端，并重置窗口。"""
        self.log_scalars(self.scene_labels.flush(), step=step)
        self.log_scalars(self.moe.flush(), step=step)
        if self._writer is not None:
            self._writer.flush()

    # ------------------------------------------------------------------ hooks
    def on_scene_step(self, labels: Mapping[str, Any]) -> None:
        """hook：逐步场景标签（``labels.compute_step_labels`` 的输出）。"""
        if labels:
            self.scene_labels.update(labels)

    def on_moe_step(
        self,
        router_weights: Any,
        *,
        expert_outputs: Any = None,
        primary_output: Any = None,
    ) -> None:
        """hook：一次 MoE 前向的路由统计（参数含义见 ``MoERoutingStatistics.update``）。"""
        self.moe.update(router_weights, expert_outputs=expert_outputs, primary_output=primary_output)

    def on_episode(self, kpis: Mapping[str, Any], step: Optional[int] = None) -> None:
        """hook：episode KPI（写 ``kpi/<name>``）。"""
        self.log_scalars({f"{_KPI_PREFIX}/{name}": value for name, value in kpis.items()}, step=step)

    def on_train_step(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        """hook：训练指标（写 ``train/<name>``）。"""
        self.log_scalars({f"{_TRAIN_PREFIX}/{name}": value for name, value in metrics.items()}, step=step)

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
