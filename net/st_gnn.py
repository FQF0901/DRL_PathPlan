"""Spatio-temporal GNN：latent 自回归转移 + t0 锚定物理小解码（诊断）。

v8（B3）规格
-----------
- **输入**：latent 状态 ``z_ego (B,H)`` / ``z_od (B,S,H)`` / ``z_ld (B,L,H)``、``node_mask``、
  ``pose (B,1+S+L,3)``（调用方构造：ego = 累积/解析推进位姿、OD = t0 位姿 + k·dt·v_t0、
  LD = t0 静止）、``step_index``；
- **节点** = ``[z_ego + step_embed(k), z_od, z_ld]``；MP 2 层（复用 :class:`net.spatial.SpatialEncoder`）；
- **latent 转移**（自回归）：``z_*_next = z_* + head([h_*, z_*])``（3 个 ``2H→H→H`` 头，
  输出层零初始化 ⇒ 初始 = 恒等转移）；无效槽位输出保持 0（编码器空间不变量）；
- **presence/entry logits**（``h_od``，零权重 + 先验 bias）；
- **物理小解码头**（od 5 维 / ld 4 维，先验 + 残差，输出层零初始化）——**诊断/旧消费者**：
  先验 = t0 锚定状态 + ``k·dt·v_t0``（OD 匀速外推）/ t0 状态（LD 静止），在 **t0 帧**表达
  （与数据集 ``od_fut/ld_fut`` 的 t0 对齐口径一致）。

参数（H=128、MP 2 层）≈384k：MP 199k + latent 头 148k + 物理解码 35k + step/presence/entry ~1k。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from net.encoders import H, LD_SLOTS, OD_SLOTS
from net.spatial import SpatialEncoder

OD_PRED_DIM = 5
LD_PRED_DIM = 4


def _latent_head(hidden: int) -> nn.Sequential:
    """latent 转移头 ``2H → H → H``（输出层零初始化 ⇒ 初始为恒等转移）。"""
    head = nn.Sequential(
        nn.Linear(2 * hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
    )
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class SpatioTemporalGNN(nn.Module):
    """latent 状态的自回归转移器（+ t0 锚定物理小解码诊断）。"""

    def __init__(
        self,
        hidden: int = H,
        od_slots: int = OD_SLOTS,
        ld_slots: int = LD_SLOTS,
        steps: int = 6,
        dt: float = 0.5,
        spatial_layers: int = 2,
        od_knn: int = 4,
        presence_bias: float = 2.0,
        entry_bias: float = -2.0,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.steps = int(steps)
        self.dt = float(dt)
        self.spatial = SpatialEncoder(
            hidden, layers=spatial_layers, od_knn=od_knn, num_od=self.od_slots, num_ld=self.ld_slots
        )
        self.step_embed = nn.Embedding(self.steps, hidden)
        # ---- latent 转移头（v8 B3：自回归状态更新）----
        self.ego_transition = _latent_head(hidden)
        self.od_transition = _latent_head(hidden)
        self.ld_transition = _latent_head(hidden)
        # ---- 物理小解码头（诊断；先验 + 零初始化残差）----
        self.od_head = nn.Sequential(
            nn.Linear(hidden + OD_PRED_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, OD_PRED_DIM),
        )
        self.ld_head = nn.Sequential(
            nn.Linear(hidden + LD_PRED_DIM, hidden),
            nn.GELU(),
            nn.Linear(hidden, LD_PRED_DIM),
        )
        # presence：初始 bias=+2（sigmoid≈0.88，"目标大多持续存在"先验）；entry：bias=-2
        # （"新目标进入罕见"先验）。零权重 + 非零 bias ⇒ 初始与输入无关的稳定先验。
        self.presence = nn.Linear(hidden, 1)
        self.entry = nn.Linear(hidden, 1)
        self._reset_heads(presence_bias, entry_bias)

    def _reset_heads(self, presence_bias: float, entry_bias: float) -> None:
        for head in (self.od_head[-1], self.ld_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for head, bias in ((self.presence, presence_bias), (self.entry, entry_bias)):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, float(bias))

    # ---------------------------------------------------------------- 状态/目标
    @staticmethod
    def od_state_from_features(od_feat: Tensor, od_mask: Tensor | None = None) -> Tensor:
        """OD 原始 9 维 → 预测空间 5 维 ``[dx,dy,vx,vy,heading]``；掩码外清零。

        空槽位（``cos=sin=0``）会让 ``atan2(0,0)`` 的反向产生 NaN 并污染输入梯度
        （与 :meth:`net.encoders.ObsEncoders.od_pose` 同源问题）：退化槽位改用
        ``(cos=1, sin=0)``，前向仍被 ``mask`` 乘为 0，反向有限。
        """
        cos_t = od_feat[..., 4]
        sin_t = od_feat[..., 5]
        norm = torch.sqrt(cos_t * cos_t + sin_t * sin_t)
        safe = norm > 1e-6
        cos_safe = torch.where(safe, cos_t, torch.ones_like(cos_t))
        sin_safe = torch.where(safe, sin_t, torch.zeros_like(sin_t))
        heading = torch.atan2(sin_safe, cos_safe).unsqueeze(-1)
        state = torch.cat([od_feat[..., 0:4], heading], dim=-1)
        if od_mask is not None:
            state = state * od_mask.unsqueeze(-1)
        return state

    @staticmethod
    def ld_state_from_features(ld_feat: Tensor, ld_mask: Tensor | None = None) -> Tensor:
        """LD 原始 7 维 → 预测空间 4 维 ``[dx,dy,heading,curvature]``；掩码外清零。"""
        state = ld_feat[..., 0:4]
        if ld_mask is not None:
            state = state * ld_mask.unsqueeze(-1)
        return state

    # ---------------------------------------------------------------- 节点特征（公共委托）
    def _message_pass(
        self,
        z_ego: Tensor,
        z_od: Tensor,
        z_ld: Tensor,
        node_mask: Tensor,
        pose: Tensor,
        step_index: int,
    ) -> Tensor:
        """节点构造（ego 加步嵌入）+ 空间消息传递 → ``nodes (B,1+S+L,H)``。"""
        batch = int(z_ego.shape[0])
        index = torch.full((batch, ), int(step_index) - 1, dtype=torch.long, device=z_ego.device)
        nodes = torch.cat(
            [z_ego.unsqueeze(1) + self.step_embed(index).unsqueeze(1), z_od, z_ld],
            dim=1,
        )
        return self.spatial(nodes, node_mask, pose)

    def node_features(
        self,
        *,
        ego_ctx: Tensor,
        od_ctx: Tensor,
        ld_ctx: Tensor,
        node_mask: Tensor,
        pose: Tensor,
        step_index: int = 1,
    ) -> tuple[Tensor, Tensor]:
        """**公共委托**：节点构造 + 步嵌入 + ``spatial`` 消息传递 → ``(h_od (B,S,H), h_ld (B,L,H))``。

        与 :meth:`forward` 解码器前的节点级输出同一实现（forward 内部亦调用本方法）；
        供诊断/外部消费者取对象级节点特征。

        Args:
            ego_ctx: ``(B,H)`` ego 节点特征（v8：latent 状态 ``z_ego``）。
            od_ctx/ld_ctx: ``(B,S,H)/(B,L,H)`` OD/LD 节点特征。
            node_mask: ``(B,1+S+L)`` 图节点掩码（ego 恒 1）。
            pose: ``(B,1+S+L,3)`` 节点位姿（t0 帧；ego 在原点/累积位姿）。
            step_index: 1..steps（1-based，决定步嵌入；默认 1）。
        """
        nodes = self._message_pass(ego_ctx, od_ctx, ld_ctx, node_mask, pose, int(step_index))
        return nodes[:, 1 : 1 + self.od_slots], nodes[:, 1 + self.od_slots :]

    # ---------------------------------------------------------------- 单步推演
    def forward(
        self,
        *,
        z_ego: Tensor,
        z_od: Tensor,
        z_ld: Tensor,
        node_mask: Tensor,
        pose: Tensor,
        step_index: int,
        od_anchor: Tensor | None = None,
        ld_anchor: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """第 ``step_index``（1-based）步的 latent 自回归转移 + 物理小解码。

        Args:
            z_ego: ``(B,H)`` ego latent 状态（当前步）。
            z_od/z_ld: ``(B,S,H)/(B,L,H)`` OD/LD latent 状态（当前步；无效槽位为 0）。
            node_mask: ``(B,1+S+L)`` 图节点掩码（t0 帧口径，rollout 逐步复用）。
            pose: ``(B,1+S+L,3)`` 节点位姿（t0 帧；ego = 累积位姿、OD = t0 + k·dt·v_t0、
                LD 静止；由调用方构造）。
            step_index: 1..steps，决定步嵌入与 OD 先验外推视界。
            od_anchor/ld_anchor: ``(B,S,5)/(B,L,4)`` **t0 帧**锚定状态（物理小解码先验）；
                缺省（None）→ 先验全 0（仅诊断路径）。

        Returns:
            ``(z_ego_next (B,H), z_od_next (B,S,H), z_ld_next (B,L,H), od_pred (B,S,5),
            ld_pred (B,L,4), presence_logit (B,S), entry_logit (B,S))``。
        """
        if not 1 <= int(step_index) <= self.steps:
            raise ValueError(f"step_index 必须在 1..{self.steps}，收到 {step_index}")
        nodes = self._message_pass(z_ego, z_od, z_ld, node_mask, pose, int(step_index))
        h_ego = nodes[:, 0]
        h_od = nodes[:, 1 : 1 + self.od_slots]
        h_ld = nodes[:, 1 + self.od_slots :]

        # (a) latent 自回归转移（残差；零初始化 ⇒ 初始恒等）
        z_ego_next = z_ego + self.ego_transition(torch.cat([h_ego, z_ego], dim=-1))
        z_od_next = z_od + self.od_transition(torch.cat([h_od, z_od], dim=-1))
        z_ld_next = z_ld + self.ld_transition(torch.cat([h_ld, z_ld], dim=-1))
        # 无效槽位保持严格 0（编码器空间不变量：embed 掩码外清零）
        od_slot_mask = node_mask[:, 1 : 1 + self.od_slots]
        ld_slot_mask = node_mask[:, 1 + self.od_slots :]
        z_od_next = z_od_next * od_slot_mask.unsqueeze(-1)
        z_ld_next = z_ld_next * ld_slot_mask.unsqueeze(-1)

        # (b) 物理小解码（诊断）：t0 锚定先验 + 零初始化残差
        if od_anchor is None:
            prior_od = torch.zeros(
                (int(z_od.shape[0]), self.od_slots, OD_PRED_DIM),
                dtype=z_od.dtype,
                device=z_od.device,
            )
        else:
            offset = int(step_index) * self.dt
            prior_od = torch.stack(
                [
                    od_anchor[..., 0] + od_anchor[..., 2] * offset,
                    od_anchor[..., 1] + od_anchor[..., 3] * offset,
                    od_anchor[..., 2],
                    od_anchor[..., 3],
                    od_anchor[..., 4],
                ],
                dim=-1,
            )
        prior_ld = (
            torch.zeros(
                (int(z_ld.shape[0]), self.ld_slots, LD_PRED_DIM),
                dtype=z_ld.dtype,
                device=z_ld.device,
            )
            if ld_anchor is None
            else ld_anchor
        )
        od_pred = prior_od + self.od_head(torch.cat([h_od, prior_od], dim=-1))
        ld_pred = prior_ld + self.ld_head(torch.cat([h_ld, prior_ld], dim=-1))
        presence_logit = self.presence(h_od).squeeze(-1)
        entry_logit = self.entry(h_od).squeeze(-1)
        return z_ego_next, z_od_next, z_ld_next, od_pred, ld_pred, presence_logit, entry_logit
