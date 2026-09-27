"""MoE 模块：primary（恒激活）+ 8 个场景专家（零初始化残差）+ **top-2 软混合**路由。

设计（v1.3 定稿：去聚类 + MoE 负载均衡）
----------------------------------------
- **输出口径**（推理与训练一致、**全场景生效**）：
  ``out = primary(x) + residual_scale · Σ_{i∈top2} g_i · expert_i(x)`` —— 无硬切、
  无二值门、无 hard_mask；
- **top-2 软混合**：router 输出 8 个 logits，取 top-2 后在**被选中的两个 logit 上做
  softmax**（和为 1，其余 6 个权重恒为 0）；
- **MoE 开关**（``moe_enabled``）：**phase 1（primary）关闭** —— 专家分支严格不参与
  （输出 = primary，专家/路由无梯度）；phase 2（specific）打开；
- **负载均衡 aux（Switch 式）**：``α · E · Σ_i f_i · P_i`` ——
  ``f_i`` = **门控权重质量占比**（``mean_t g_{t,i}``，detach，Σf=1）、``P_i`` = 全专家
  softmax 概率的 batch 均值（可微）；``α = load_balance_coef``（默认 0.01，可配；0 = 关闭）；
  值域 ``[1, E]``：均匀路由（``f_i=P_i=1/E``）→ 1（下界）；门控质量全压到同一专家 → ≈ E（上界）；
- 8 个 expert 结构 ``H→expert_hidden→H``，**输出层零初始化**：训练开始时专家分支严格为 0，
  不破坏 primary 的初始表示；
- 对外暴露（``aux``）：

  - ``router_logits (B,8)``：路由 logits；
  - ``expert_weights (B,8)``：top-2 混合权重（恰好 2 个非零、和为 1）；
  - ``effective_experts``：``Σw``（=1）；
  - ``load_balance_loss``：标量（已乘 α；MoE 关闭或 α=0 时缺省）；
  - ``expert_load (E,)``（门控质量，Σ=1）/ ``load_cv`` / ``gate_entropy``：TB 负载诊断（no_grad）。

- ``gates`` 可显式传入（测试/固定路由消融），此时跳过 top-2，仍走同一条前向。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from net.encoders import H


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, *, zero_init: bool = False) -> nn.Sequential:
    """``in→hidden→out`` 两层 MLP；``zero_init=True`` 时输出层权重与偏置清零。"""
    layer = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )
    if zero_init:
        nn.init.zeros_(layer[-1].weight)
        nn.init.zeros_(layer[-1].bias)
    return layer


def top_k_softmax(logits: Tensor, k: int = 2) -> Tensor:
    """``(B,E)`` logits → ``(B,E)`` 权重：top-k 上 softmax，其余严格为 0。"""
    if logits.ndim != 2:
        raise ValueError(f"top_k_softmax 期望 (B,E)，收到 {tuple(logits.shape)}")
    width = int(logits.shape[-1])
    k = max(1, min(int(k), width))
    values, indices = logits.topk(k, dim=-1)
    weights = torch.zeros_like(logits).scatter(-1, indices, torch.softmax(values, dim=-1))
    return weights


def switch_load_balance_aux(logits: Tensor, *, top_k: int = 2) -> Tensor:
    """Switch 式负载均衡 aux：``E · Σ_i f_i · P_i``（不含 α；可微部分 = ``P_i``）。

    - ``P_i`` = ``softmax(logits)`` 的 batch 均值（可微）；
    - ``f_i`` = top-k 门控权重质量的 batch 均值（detach；``Σf = 1``）；
    - 均匀路由（``f_i=P_i=1/E``）→ 1（下界）；门控质量全压到同一专家 → ≈ E（上界）。
    """
    if logits.ndim != 2:
        raise ValueError(f"switch_load_balance_aux 期望 (B,E)，收到 {tuple(logits.shape)}")
    num_experts = int(logits.shape[-1])
    prob_mean = torch.softmax(logits, dim=-1).mean(dim=0)  # (E,)
    with torch.no_grad():
        load = top_k_softmax(logits, int(top_k)).mean(dim=0)  # (E,)，Σ=1
        load = load / load.sum().clamp(min=1e-8)
    return float(num_experts) * (load * prob_mean).sum()


class MoEBlock(nn.Module):
    """plan-head 内的 MoE（作用于 fusion 后的全局 token）。"""

    def __init__(
        self,
        hidden: int = H,
        num_experts: int = 8,
        expert_hidden: int = 192,
        router_hidden: int = 64,
        top_k: int = 2,
        load_balance_coef: float = 0.0,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.primary = _mlp(hidden, expert_hidden, hidden)
        self.experts = nn.ModuleList(
            [_mlp(hidden, expert_hidden, hidden, zero_init=True) for _ in range(self.num_experts)]
        )
        self.router = nn.Sequential(
            nn.Linear(hidden, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, self.num_experts),
        )
        #: 残差尺度（可学）；expert 零初始化时不影响初始输出
        self.residual_scale = nn.Parameter(torch.ones(1))
        #: 运行时开关（非参数，不进 state_dict）：phase 1 关 / phase 2 开；推理默认开
        self.moe_enabled = True
        #: 负载均衡 aux 权重 α（非参数）；0 = 不加 aux
        self.load_balance_coef = float(load_balance_coef)

    # ---------------------------------------------------------------- 运行时开关
    def set_enabled(self, enabled: bool) -> "MoEBlock":
        """MoE 开关（phase 1 关：输出严格 = primary；phase 2 开）。返回 self 便于链式调用。"""
        self.moe_enabled = bool(enabled)
        return self

    def set_load_balance_coef(self, coef: float) -> "MoEBlock":
        """负载均衡 aux 权重 α（0 = 关闭）。返回 self 便于链式调用。"""
        self.load_balance_coef = float(coef)
        return self

    # ---------------------------------------------------------------- 负载诊断
    @torch.no_grad()
    def load_stats(self, logits: Tensor) -> dict[str, Tensor]:
        """负载诊断：``expert_load (E,)``（Σ=1）/ ``load_cv`` / ``gate_entropy``（均 no_grad）。

        - ``expert_load``：top-k **门控权重质量**的 batch 均值（Σ=1）；
        - ``load_cv``：``expert_load`` 的变异系数（0 = 完全均衡）；
        - ``gate_entropy``：逐 token ``softmax(logits)`` 熵的均值（归一化到 log E；1 = 均匀）。
        """
        prob = torch.softmax(logits, dim=-1)
        load = top_k_softmax(logits, self.top_k).mean(dim=0)  # 门控权重质量（Σ=1）
        load = load / load.sum().clamp(min=1e-8)
        mean = load.mean()
        cv = load.std(unbiased=False) / mean.clamp(min=1e-8)
        p = prob.clamp(min=1e-8)
        entropy = (-(p * p.log()).sum(dim=-1) / math.log(self.num_experts)).mean()
        return {"expert_load": load, "load_cv": cv, "gate_entropy": entropy}

    # ---------------------------------------------------------------- 前向
    def forward(
        self,
        x: Tensor,
        router_input: Tensor | None = None,
        gates: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """``x (B,H)`` -> ``(out (B,H), aux)``。

        Args:
            x: MoE 输入（也是默认的 router 输入）。
            router_input: 覆盖 router 输入（默认等于 ``x``）。
            gates: 显式门控 ``(B,E)``，覆盖 top-2 softmax（测试/消融用）。
        """
        logits = self.router(x if router_input is None else router_input)
        weights = top_k_softmax(logits, self.top_k) if gates is None else gates
        aux: dict[str, Tensor] = {
            "router_logits": logits,
            "expert_weights": weights,
            "effective_experts": weights.sum(dim=-1),
            "moe_enabled": torch.as_tensor(float(self.moe_enabled), device=logits.device),
        }
        primary_out = self.primary(x)
        if not self.moe_enabled:
            # phase 1：MoE 关闭 —— 专家分支严格不参与（输出 = primary；专家/路由无梯度）
            return primary_out, aux
        expert_stack = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (weights.unsqueeze(-1) * expert_stack).sum(dim=1)
        out = primary_out + self.residual_scale * mixed
        aux.update(self.load_stats(logits))
        if self.load_balance_coef > 0.0:
            aux["load_balance_loss"] = (
                float(self.load_balance_coef) * switch_load_balance_aux(logits, top_k=self.top_k)
            )
        return out, aux
