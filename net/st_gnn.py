"""Spatio-temporal GNN：由"更新后的 ego mem + od/ld mem"推演下一时刻 OD/LD。

v2 规格（第 4、6、7 条）
-----------------------
- 输入：ego 节点 = 更新后 ego mem 的注意力聚合（含刚挤入的合成帧，reserved 维携带
  执行动作）；OD 节点 = OD mem 的逐槽注意力聚合；LD 节点 = **当前帧**编码；
  图结构复用 :class:`net.spatial.SpatialEncoder` 的 2 层消息传递（OD↔ego、LD↔ego、
  LD↔LD 相邻、OD↔OD 近邻），节点位姿取 mem 当前帧（自车系，ego 在原点）；
- 输出（**t0 帧预测空间**，与旧世界模型同约定，规格第 6 条）：
  ``od_pred (B,S,5) = [dx,dy,vx,vy,heading_rel]``、``ld_pred (B,L,4) = [dx,dy,heading_rel,curvature]``，
  以及 ``od_presence_logit (B,S)``、``od_entry_logit (B,S)``；
- 预测以 **t0 锚定状态 + 匀速/静止先验**为基准、解码器输出**残差**；解码器输出层零初始化
  ⇒ 未训练时严格退化为匀速（OD）/静止（LD）基线。确定性（无 dropout/噪声）。

坐标约定（规格第 6 条，保持既有正确实现）
-----------------------------------------
本模块的输出锚定在 **t0 帧**；调用方（:class:`net.model.DrivingModel`）用
``od_pred_to_features/ld_pred_to_features`` 做 SE(2) 逆变换到**当前累积位姿系**、
重新编码后才挤入 mem 副本——这就是"t0 预测 → 逆变换 → 重编码"的既有正确语义。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from net.encoders import H, LD_SLOTS, OD_SLOTS
from net.spatial import SpatialEncoder

OD_PRED_DIM = 5
LD_PRED_DIM = 4


class SpatioTemporalGNN(nn.Module):
    """消息传递 + t0 锚定残差解码的 OD/LD 单步推演器（rollout 内逐步调用）。"""

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
        ``net.model`` 的 t0 单次消息传递（A3）用它取对象级特征，不再直连 ``self.spatial``/
        ``self.step_embed``（消除对内部结构的耦合）。

        Args:
            ego_ctx: ``(B,H)`` ego 节点特征（forward 内为更新后 ego mem 聚合）。
            od_ctx/ld_ctx: ``(B,S,H)/(B,L,H)`` OD/LD 节点特征。
            node_mask: ``(B,1+S+L)`` 图节点掩码（ego 恒 1）。
            pose: ``(B,1+S+L,3)`` 节点位姿（ego 在原点）。
            step_index: 1..steps（1-based，决定步嵌入；默认 1 = t0 单次 pass）。
        """
        batch = int(ego_ctx.shape[0])
        index = torch.full((batch, ), int(step_index) - 1, dtype=torch.long, device=ego_ctx.device)
        nodes = torch.cat(
            [ego_ctx.unsqueeze(1) + self.step_embed(index).unsqueeze(1), od_ctx, ld_ctx],
            dim=1,
        )
        nodes = self.spatial(nodes, node_mask, pose)
        return nodes[:, 1 : 1 + self.od_slots], nodes[:, 1 + self.od_slots :]

    # ---------------------------------------------------------------- 单步推演
    def forward(
        self,
        *,
        ego_ctx: Tensor,
        od_ctx: Tensor,
        ld_ctx: Tensor,
        node_mask: Tensor,
        pose: Tensor,
        step_index: int,
        od_anchor: Tensor,
        ld_anchor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """推演第 ``step_index``（1-based）步。

        Args:
            ego_ctx: ``(B,H)`` 更新后 ego mem 的聚合（含执行动作的合成帧信息）。
            od_ctx/ld_ctx: ``(B,S,H)/(B,L,H)`` OD/LD 节点特征（LD 为当前帧编码）。
            node_mask: ``(B,1+S+L)`` 图节点掩码（ego 恒 1）。
            pose: ``(B,1+S+L,3)`` 节点位姿（ego 在原点）。
            step_index: 1..steps，决定先验外推视界与步嵌入。
            od_anchor/ld_anchor: ``(B,S,5)/(B,L,4)`` **t0 帧**锚定状态（预测空间）。

        Returns:
            ``(od_pred (B,S,5), ld_pred (B,L,4), presence_logit (B,S), entry_logit (B,S))``。
        """
        if not 1 <= int(step_index) <= self.steps:
            raise ValueError(f"step_index 必须在 1..{self.steps}，收到 {step_index}")
        h_od, h_ld = self.node_features(
            ego_ctx=ego_ctx,
            od_ctx=od_ctx,
            ld_ctx=ld_ctx,
            node_mask=node_mask,
            pose=pose,
            step_index=int(step_index),
        )

        # 先验：OD 匀速外推（t0 状态 + k·dt·v），LD 在 t0 帧静止
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
        prior_ld = ld_anchor
        od_pred = prior_od + self.od_head(torch.cat([h_od, prior_od], dim=-1))
        ld_pred = prior_ld + self.ld_head(torch.cat([h_ld, prior_ld], dim=-1))
        presence_logit = self.presence(h_od).squeeze(-1)
        entry_logit = self.entry(h_od).squeeze(-1)
        return od_pred, ld_pred, presence_logit, entry_logit
