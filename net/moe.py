"""MoE 模块：primary（恒激活）+ 8 个场景专家（零初始化残差）+ **top-2 软混合**路由。

设计（v2 规格第 3 条：MoE 只在 plan head，不再共享）
---------------------------------------------------
- **primary 不受路由门控**（DeepSeekMoE shared-expert 模式）：
  ``out = primary(x) + residual_scale · Σ_i w_i · expert_i(x)``；
- **top-2 软混合**：router 输出 8 个 logits，取 top-2 后在**被选中的两个 logit 上做
  softmax**（和为 1，其余 6 个权重恒为 0）；比 sigmoid 全专家加权更稀疏、比 hard top-1
  更平滑。混合权重来自网络自身的路由（由训练侧软目标监督学习，见下）；
- 8 个 expert 结构 ``H→expert_hidden→H``，**输出层零初始化**：训练开始时 expert 分支
  严格为 0，不破坏 primary 的初始表示；
- 对外暴露：

  - ``router_logits (B,8)``：路由 logits，**监督面**——trainer 侧对聚类软目标
    （``pipeline.trainer.router_soft_target_loss``）做 **CE/KL（softmax + 温度）**；
    “权重由训练侧软目标提供”即指 router 通过这些软目标学习到最终用于混合的概率结构；
  - ``expert_weights (B,8)``：top-2 softmax 的混合权重（恰好 2 个非零、和为 1）；
  - ``effective_experts``：``Σw``（=1，供监控口径兼容；注意与 router softmax 分布的
    ``Σw ∈ [1,E]`` 语义不同，见 ``pipeline.monitoring``）。

- ``gates`` 可显式传入（测试/固定路由消融），此时跳过 top-2，仍走同一条前向，
  便于验证 primary 与门控严格解耦。
"""

from __future__ import annotations

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


class MoEBlock(nn.Module):
    """plan-head 内的 MoE（作用于 fusion 后的全局 token）。"""

    def __init__(
        self,
        hidden: int = H,
        num_experts: int = 8,
        expert_hidden: int = 192,
        router_hidden: int = 64,
        top_k: int = 2,
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

    def forward(
        self,
        x: Tensor,
        router_input: Tensor | None = None,
        gates: Tensor | None = None,
        hard_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """``x (B,H)`` -> ``(out (B,H), aux)``。

        Args:
            x: MoE 输入（也是默认的 router 输入）。
            router_input: 覆盖 router 输入（默认等于 ``x``）。
            gates: 显式门控 ``(B,E)``，覆盖 top-2 softmax（测试/消融用）。
            hard_mask: 可选 ``(B,)`` 二值硬切（lane T 双分支）：**1 = 该样本走 specific
                8 路专家混合，0 = 只走 primary（specific 分支输出严格置 0，不做软混合）**。
                ``None``（默认）= 旧行为（specific 分支恒激活）。
        """
        logits = self.router(x if router_input is None else router_input)
        weights = top_k_softmax(logits, self.top_k) if gates is None else gates
        primary_out = self.primary(x)
        expert_stack = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (weights.unsqueeze(-1) * expert_stack).sum(dim=1)
        if hard_mask is not None:
            mask = hard_mask.to(dtype=mixed.dtype).reshape(-1, 1)
            mixed = mixed * mask
        out = primary_out + self.residual_scale * mixed
        aux = {
            "router_logits": logits,
            "expert_weights": weights,
            "effective_experts": weights.sum(dim=-1),
        }
        return out, aux
