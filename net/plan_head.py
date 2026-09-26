"""Plan head：4 个 mem 聚合（+ nav/signal）→ 下一时刻 ego 特征；**MoE 只在这里**。

v2 规格（第 3 条）
-----------------
- 输入 = ``ego/od/ld/others`` 四个 mem 聚合 + ``nav`` + ``signal``（共 6H）；OD/LD 用
  掩码均值池化为全局 token，ego/others 已是注意力聚合后的单 token；
- 输出 = **下一时刻 ego 特征**（8 维中的前 6 维；最后 2 维 reserved 承载刚执行的动作，
  与 p2-contract §8.4 的"reserved = 上一策略步动作"约定一致，由 :class:`DrivingModel`
  在 rollout 里拼接），以及供策略/价值头使用的融合 latent；
- **MoE 只在本模块内**（primary 常开 + 8 个 specific 专家，top-2 软混合，见
  :mod:`net.moe`），不再共享给其它分支；``router_logits(8)``/``expert_weights(8)``
  经 :class:`DrivingModel` 原样输出，供训练侧软目标监督。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from net.encoders import EGO_MEM_DIM, H
from net.moe import MoEBlock

#: plan head 消费的全局 token 数（ego / OD / LD / others / nav / signal）
FUSION_TOKENS = 6
#: ego 特征里由动作占用（reserved）的维度数
EGO_ACTION_DIM = 2
#: plan head 直接预测的 ego 特征维数（其余 reserved 由 rollout 写入执行动作）
EGO_NEXT_DIM = EGO_MEM_DIM - EGO_ACTION_DIM


class PlanHead(nn.Module):
    """mem 聚合 → 融合 + MoE → 下一 ego 特征 / 策略-价值 latent。"""

    def __init__(
        self,
        hidden: int = H,
        num_experts: int = 8,
        expert_hidden: int = 192,
        router_hidden: int = 64,
        top_k: int = 2,
        ego_next_dim: int = EGO_NEXT_DIM,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.ego_next_dim = int(ego_next_dim)
        self.fusion = nn.Sequential(
            nn.Linear(FUSION_TOKENS * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.moe = MoEBlock(hidden, num_experts, expert_hidden, router_hidden, top_k=top_k)
        self.ego_next = nn.Sequential(
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(8, hidden // 2), self.ego_next_dim),
        )

    def forward(
        self,
        ego_ctx: Tensor,
        od_pool: Tensor,
        ld_pool: Tensor,
        others_ctx: Tensor,
        nav_token: Tensor,
        signal_token: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """返回 ``(latent (B,H), ego_next (B,EGO_NEXT_DIM), moe_aux)``。"""
        pooled = torch.cat([ego_ctx, od_pool, ld_pool, others_ctx, nav_token, signal_token], dim=-1)
        fused = self.fusion(pooled)
        moe_out, moe_aux = self.moe(fused)
        latent = self.norm(fused + moe_out)
        return latent, self.ego_next(latent), moe_aux
