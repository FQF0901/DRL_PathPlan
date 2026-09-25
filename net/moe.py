"""MoE 模块：primary（恒激活）+ 8 个场景专家（零初始化残差）+ sigmoid 多标签路由。

设计（p2-contract §2 / 已确认决策 6）
-------------------------------------
- **primary 不受路由门控**（DeepSeekMoE shared-expert 模式）：
  ``out = primary(x) + residual_scale · Σ_i gate_i · expert_i(x)``；
- 8 个 expert 结构 ``H→192→H``，**输出层零初始化**：训练开始时 expert 分支严格为 0，
  不破坏 primary 的初始表示；
- router = ``sigmoid(W·x + b)``（多标签，非 softmax），与 `config/model.yaml` 的
  ``moe.router.supervised_labels`` 8 个标签一一对应（固定顺序，见
  :data:`net.model.SUPERVISED_LABELS`），训练时用 BCE 监督；
- 对外暴露 ``router_logits (B,8)``、``expert_weights (B,8)``（即门控值，供监控
  "有效专家数 N=Σw、每 expert 权重"）与 ``residual_scale``；
- ``gates`` 可显式传入（测试/监控/固定路由消融），此时仍走同一条前向，
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


class MoEBlock(nn.Module):
    """场景级 MoE（作用于全局 latent token）。"""

    def __init__(
        self,
        hidden: int = H,
        num_experts: int = 8,
        expert_hidden: int = 192,
        router_hidden: int = 64,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.num_experts = int(num_experts)
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
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """``x (B,H)`` -> ``(out (B,H), aux)``。

        Args:
            x: MoE 输入（也是默认的 router 输入）。
            router_input: 覆盖 router 输入（默认等于 ``x``）。
            gates: 显式门控 ``(B,E)``，覆盖 sigmoid(router)（测试/消融用）。
        """
        logits = self.router(x if router_input is None else router_input)
        if gates is None:
            gates = torch.sigmoid(logits)
        primary_out = self.primary(x)
        expert_stack = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (gates.unsqueeze(-1) * expert_stack).sum(dim=1)
        out = primary_out + self.residual_scale * mixed
        aux = {
            "router_logits": logits,
            "expert_weights": gates,
            "effective_experts": gates.sum(dim=-1),
        }
        return out, aux
