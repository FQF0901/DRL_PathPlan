"""Plan head：4 个 mem 聚合（+ nav/signal）→ 下一时刻 ego 特征；**MoE 只在这里**。

v1.3 规格（第 3 条；去聚类 + MoE 负载均衡）
-------------------------------------------
- 输入 = ``ego/od/ld/others`` 四个 mem 聚合 + ``nav`` + ``signal``（共 6H）；OD/LD 用
  掩码均值池化为全局 token，ego/others 已是注意力聚合后的单 token；
- 输出 = **下一时刻 ego 特征**（8 维中的前 6 维；最后 2 维 reserved 承载刚执行的动作，
  与 p2-contract §8.4 的"reserved = 上一策略步动作"约定一致，由 :class:`DrivingModel`
  在 rollout 里拼接），以及供策略/价值头使用的融合 latent；
- **MoE 只在本模块内**（primary 常开 + 8 个 specific 专家，top-2 软混合，见 :mod:`net.moe`）：
  输出 = ``primary + Σ_{i∈top2} g_i · expert_i``，推理与训练一致、全场景生效；
  **无二值门控头、无硬切掩码**（lane U1 摘除）。
- ``router_logits(8)``/``expert_weights(8)``/负载诊断（``expert_load``/``load_cv``/
  ``gate_entropy``）与 ``load_balance_loss``（Switch 式 aux）经 :class:`DrivingModel`
  原样输出，供训练侧损失与 TB 使用。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from net.anchor import anchor_mixture
from net.encoders import EGO_MEM_DIM, H
from net.moe import MoEBlock

#: plan head 消费的全局 token 数（ego / OD / LD / others / nav / signal）
FUSION_TOKENS = 6
#: ego 特征里由动作占用（reserved）的维度数
EGO_ACTION_DIM = 2
#: plan head 直接预测的 ego 特征维数（其余 reserved 由 rollout 写入执行动作）
EGO_NEXT_DIM = EGO_MEM_DIM - EGO_ACTION_DIM
#: 连续速度头初始速度（m/0.5s；expert ds 中位 ≈3.5 m → 初始 raw = logit(3.5/10)）
_SPEED_INIT_DS = 3.5
#: 动作 ds 上界（与 net.policy.ACTION_HIGH[0] 一致；速度头 sigmoid 压缩到 [0,10]）
_SPEED_DS_HIGH = 10.0


class PlanHead(nn.Module):
    """mem 聚合 → 融合 + MoE → 下一 ego 特征 / 策略-价值 latent。"""

    def __init__(
        self,
        hidden: int = H,
        num_experts: int = 8,
        expert_hidden: int = 76,
        router_hidden: int = 384,
        primary_hidden: int = 768,
        top_k: int = 2,
        ego_next_dim: int = EGO_NEXT_DIM,
        load_balance_coef: float = 0.0,
        num_anchors: int = 0,
        anchor_embed_dim: int = 32,
        anchor_temperature: float = 1.0,
        anchor_hard: bool = False,
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
        self.moe = MoEBlock(
            hidden,
            num_experts,
            expert_hidden,
            router_hidden,
            primary_hidden=primary_hidden,
            top_k=top_k,
            load_balance_coef=load_balance_coef,
        )
        self.ego_next = nn.Sequential(
            nn.Linear(hidden, max(8, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(8, hidden // 2), self.ego_next_dim),
        )
        # ---- v7 结构迭代 B：K-anchor 计划头（num_anchors=0 = 完全关闭，旧行为逐位不变）----
        self.num_anchors = int(num_anchors)
        self.anchor_embed_dim = int(anchor_embed_dim)
        self.anchor_temperature = float(anchor_temperature)
        self.anchor_hard = bool(anchor_hard)
        if self.num_anchors > 0:
            if self.num_anchors < 2:
                raise ValueError(f"num_anchors 必须 >=2（或 0=关闭），收到 {self.num_anchors}")
            #: 选择头：latent → 锚 logits（零初始化 ⇒ 初始均匀 softmax）
            self.anchor_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.num_anchors),
            )
            nn.init.zeros_(self.anchor_head[-1].weight)
            nn.init.zeros_(self.anchor_head[-1].bias)
            #: 连续速度头：latent → 6 步 ds（sigmoid 压缩到 [0,10]；初始 ≈3.5 m/步）
            self.speed_head = nn.Sequential(
                nn.Linear(hidden, max(8, hidden // 2)),
                nn.GELU(),
                nn.Linear(max(8, hidden // 2), 6),
            )
            nn.init.zeros_(self.speed_head[-1].weight)
            nn.init.constant_(
                self.speed_head[-1].bias, math.log(_SPEED_INIT_DS / (_SPEED_DS_HIGH - _SPEED_INIT_DS))
            )
            #: 逐锚 6×2 残差头（anchor embedding + latent；末层零初始化 ⇒ 初始残差=0）
            self.anchor_embed = nn.Parameter(torch.zeros(self.num_anchors, self.anchor_embed_dim))
            nn.init.normal_(self.anchor_embed, std=0.02)
            self.residual_head = nn.Sequential(
                nn.Linear(hidden + self.anchor_embed_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, 12),
            )
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
            #: 锚字典（K,6,2；lane 帧 ds/dθ；由 :meth:`set_anchors` 写入）
            #: 注意：仅在 num_anchors>0 时注册 → 关闭时 state_dict 键集合与旧模型逐位一致
            #: （旧 ckpt 严格加载不受影响）。
            self.register_buffer("anchors", torch.zeros(self.num_anchors, 6, 2))
            self.register_buffer("anchor_ds", torch.zeros(self.num_anchors, 6))
            self.register_buffer("anchor_dth", torch.zeros(self.num_anchors, 6))

    def set_moe(self, *, enabled: bool = True, load_balance_coef: float = 0.0) -> "PlanHead":
        """MoE 运行时开关 + 负载均衡 α（phase 1 关 / phase 2 开；推理默认开）。"""
        self.moe.set_enabled(enabled).set_load_balance_coef(load_balance_coef)
        return self

    # ---------------------------------------------------------------- K-anchor
    def set_anchors(self, anchors: Tensor) -> "PlanHead":
        """写入锚字典 ``(K,6,2)``（lane 帧 ds/dθ）；K 必须与 ``num_anchors`` 一致。"""
        if self.num_anchors <= 0:
            raise RuntimeError("PlanHead 未启用 K-anchor（num_anchors=0）")
        if anchors.ndim != 3 or tuple(anchors.shape[1:]) != (6, 2):
            raise ValueError(f"anchors 形状应为 (K,6,2)，收到 {tuple(anchors.shape)}")
        if int(anchors.shape[0]) != self.num_anchors:
            raise ValueError(
                f"anchors K={int(anchors.shape[0])} != num_anchors={self.num_anchors}"
            )
        if not bool(torch.isfinite(anchors).all()):
            raise ValueError("anchors 含非有限值")
        with torch.no_grad():
            self.anchors.copy_(anchors.to(dtype=self.anchors.dtype, device=self.anchors.device))
            self.anchor_ds.copy_(self.anchors[..., 0])
            self.anchor_dth.copy_(self.anchors[..., 1])
        return self

    def set_anchor_mode(
        self, *, temperature: float | None = None, hard: bool | None = None
    ) -> "PlanHead":
        """运行时锚混合模式（τ 退火 / 推理 argmax）。"""
        if temperature is not None:
            if float(temperature) <= 0.0:
                raise ValueError(f"temperature 必须 > 0，收到 {temperature}")
            self.anchor_temperature = float(temperature)
        if hard is not None:
            self.anchor_hard = bool(hard)
        return self

    def plan_anchors(self, latent: Tensor, lane_ctx: Tensor) -> dict[str, Tensor]:
        """K-anchor 计划（方案 A 软混合）→ dict（全部为 ego 系 / lane 上下文口径）。

        ``lane_ctx (B,3)`` = ``[heading_err, curvature, valid]``（lane 帧；valid=0 → 恒等）。

        返回键：``plan (B,6,2)``（``Σ p_k(anchor_k+residual_k)`` 经 lane→ego 回投）、
        ``logits (B,K)``、``probs (B,K)``、``speed_ds (B,6)``（连续速度头）、
        ``residual (B,K,6,2)``（逐锚残差均值）、``anchor_dth (K,6)``、``ctx (B,3)``。
        """
        if self.num_anchors <= 0:
            raise RuntimeError("PlanHead 未启用 K-anchor（num_anchors=0）")
        if lane_ctx.ndim != 2 or int(lane_ctx.shape[-1]) != 3:
            raise ValueError(f"lane_ctx 应为 (B,3)，收到 {tuple(lane_ctx.shape)}")
        batch = int(latent.shape[0])
        logits = self.anchor_head(latent)
        speed_raw = self.speed_head(latent)
        speed_ds = _SPEED_DS_HIGH * torch.sigmoid(speed_raw)
        embed = self.anchor_embed.unsqueeze(0).expand(batch, -1, -1)
        fused = torch.cat([latent.unsqueeze(1).expand(-1, self.num_anchors, -1), embed], dim=-1)
        residual = self.residual_head(fused).reshape(batch, self.num_anchors, 6, 2)
        plan, probs = anchor_mixture(
            logits,
            residual,
            self.anchor_dth,
            speed_ds,
            lane_ctx[:, 0],
            lane_ctx[:, 1],
            lane_ctx[:, 2],
            temperature=self.anchor_temperature,
            hard=self.anchor_hard,
        )
        return {
            "plan": plan,
            "logits": logits,
            "probs": probs,
            "speed_ds": speed_ds,
            "residual": residual,
            "anchor_dth": self.anchor_dth,
            "ctx": lane_ctx,
        }

    def forward(
        self,
        ego_ctx: Tensor,
        od_pool: Tensor,
        ld_pool: Tensor,
        others_ctx: Tensor,
        nav_token: Tensor,
        signal_token: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """返回 ``(latent (B,H), ego_next (B,EGO_NEXT_DIM), moe_aux)``。

        ``moe_aux`` 含 ``router_logits``/``expert_weights``/``effective_experts``/
        ``moe_enabled``（+ MoE 开启时 ``expert_load``/``load_cv``/``gate_entropy``/
        ``load_balance_loss``）。
        """
        pooled = torch.cat([ego_ctx, od_pool, ld_pool, others_ctx, nav_token, signal_token], dim=-1)
        fused = self.fusion(pooled)
        moe_out, moe_aux = self.moe(fused)
        latent = self.norm(fused + moe_out)
        return latent, self.ego_next(latent), moe_aux
