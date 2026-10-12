"""空间编码：2 层消息传递，覆盖 {ego, 16 OD, 16 LD} 节点。

邻接规则（p2-contract §2）
-------------------------
- ``OD ↔ ego``：所有有效 OD 与 ego 双向相连；
- ``LD ↔ ego``：所有有效 LD 与 ego 双向相连；
- ``LD ↔ LD``：槽位相邻（``i`` 与 ``i+1``，均有效）——LD 槽位按"当前车道 offset 优先、
  其余车道环优先"排序（``env/obs/ld.py``），槽位相邻近似"沿线相邻采样点"；
- ``OD ↔ OD``：欧氏距离最近的 ``od_knn`` 个有效邻居（对称化）。

边特征 = 相对位姿 ``[Δx/scale, Δy/scale, cosΔθ, sinΔθ, ‖Δp‖/scale]``。
因为所有节点位姿都已表达到**同一自车系**，任意两节点的相对位姿可直接由位姿差得到
（相对量在坐标变换下不变），无需再做旋转。

消息函数为 ``MLP([h_i, h_j, edge_ij])``，按有效边做掩码均值聚合；
无效节点的输出每层后清零，保证填充槽位不参与任何计算。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from net.encoders import H, LD_SLOTS, OD_SLOTS


def wrap_angle(x: Tensor) -> Tensor:
    """把角度 wrap 到 (-π, π]（用 atan2 保持可导且数值稳定）。"""
    return torch.atan2(torch.sin(x), torch.cos(x))


class _MessagePassingLayer(nn.Module):
    """一层边条件消息传递 + 残差 + LayerNorm。"""

    def __init__(self, hidden: int = H, edge_dim: int = 5):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: Tensor, adj: Tensor, edge: Tensor, node_mask: Tensor) -> Tensor:
        """``h (B,N,H)``, ``adj (B,N,N) bool``, ``edge (B,N,N,E)``, ``node_mask (B,N)``。"""
        batch, num_nodes, _ = h.shape
        h_i = h.unsqueeze(2).expand(batch, num_nodes, num_nodes, -1)
        h_j = h.unsqueeze(1).expand(batch, num_nodes, num_nodes, -1)
        msg = self.edge_mlp(torch.cat([h_i, h_j, edge], dim=-1))
        weight = adj.to(msg.dtype).unsqueeze(-1)
        agg = (msg * weight).sum(dim=2) / weight.sum(dim=2).clamp(min=1.0)
        update = self.node_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + update) * node_mask.unsqueeze(-1)


class LdSetAttention(nn.Module):
    """Round-20 M2：LD tokens 的 masked set cross-attention（query=context，key/value=LD tokens）。

    - ``ld_tokens (B,L,H)``、``ld_mask (B,L)``（1=有效）、``context (B,C)``（ego/nav/others 拼接）；
    - ``key_padding_mask`` = ``ld_mask < 0.5``（无效槽不参与注意力；**全无效**时返回零向量，
      由 ``_attended_or_zero`` 保证，避免 softmax 全 -inf 的 NaN）；
    - 输出 ``(B,H)``：替换 PlanHead 融合路径的 ``ld_pool``（set-level 几何上下文）。
    """

    def __init__(self, hidden: int, *, heads: int = 4, ctx_dim: int | None = None):
        super().__init__()
        ctx = int(ctx_dim or hidden)
        self.query = nn.Linear(ctx, hidden)
        self.attn = nn.MultiheadAttention(hidden, int(heads), batch_first=True)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, ld_tokens: Tensor, ld_mask: Tensor, context: Tensor) -> Tensor:
        import torch

        valid_any = (ld_mask > 0.5).any(dim=1, keepdim=True)  # (B,1)
        query = self.query(context).unsqueeze(1)  # (B,1,H)
        padding = ld_mask < 0.5  # (B,L)
        attended, _ = self.attn(
            query, ld_tokens, ld_tokens, key_padding_mask=padding, need_weights=False
        )
        out = self.out(attended.squeeze(1))  # (B,H)
        return torch.where(valid_any, out, torch.zeros_like(out))


class SpatialEncoder(nn.Module):
    """2 层消息传递编码器（层数可配）。"""

    def __init__(
        self,
        hidden: int = H,
        layers: int = 2,
        od_knn: int = 4,
        num_od: int = OD_SLOTS,
        num_ld: int = LD_SLOTS,
        pos_scale: float = 100.0,
    ):
        super().__init__()
        if layers < 1:
            raise ValueError("layers 必须 >= 1")
        self.hidden = int(hidden)
        self.num_od = int(num_od)
        self.num_ld = int(num_ld)
        #: Round-20 消融开关（**默认 = 既有行为逐位不变**）：
        #: ``include_ld_chain=False``（M1）= 移除 LD↔LD 槽位链边（保留 ego↔LD / ego↔OD / OD-KNN）；
        #: ``exclude_ld_nodes=True``（M2）= LD 节点完全绕过消息传递（独立编码，输入特征直通）。
        self.include_ld_chain: bool = True
        self.exclude_ld_nodes: bool = False
        self.od_knn = int(od_knn)
        self.pos_scale = float(pos_scale)
        self.layers = nn.ModuleList([_MessagePassingLayer(hidden) for _ in range(int(layers))])

    # ---------------------------------------------------------------- 图结构
    def build_adjacency(self, node_mask: Tensor, pose: Tensor) -> Tensor:
        """构造 ``(B,N,N) bool`` 邻接矩阵；不依赖 batch（类型由固定节点顺序确定）。"""
        batch, num_nodes = node_mask.shape
        expected = 1 + self.num_od + self.num_ld
        if num_nodes != expected:
            raise ValueError(f"节点数应为 {expected}（1 ego + {self.num_od} OD + {self.num_ld} LD），收到 {num_nodes}")
        valid = node_mask > 0.5
        ego_ok = valid[:, 0]
        od_ok = valid[:, 1 : 1 + self.num_od]
        ld_ok = valid[:, 1 + self.num_od :]

        adj = torch.zeros((batch, num_nodes, num_nodes), dtype=torch.bool, device=node_mask.device)
        # ego ↔ OD / LD（双向）
        adj[:, 0, 1 : 1 + self.num_od] = ego_ok.unsqueeze(1) & od_ok
        adj[:, 1 : 1 + self.num_od, 0] = adj[:, 0, 1 : 1 + self.num_od]
        adj[:, 0, 1 + self.num_od :] = ego_ok.unsqueeze(1) & ld_ok
        adj[:, 1 + self.num_od :, 0] = adj[:, 0, 1 + self.num_od :]

        # LD ↔ LD 槽位相邻（Round-20 M1：include_ld_chain=False 时整段移除）
        for i in range(self.num_ld - 1) if self.include_ld_chain else ():
            both = ld_ok[:, i] & ld_ok[:, i + 1]
            adj[:, 1 + self.num_od + i, 1 + self.num_od + i + 1] = both
            adj[:, 1 + self.num_od + i + 1, 1 + self.num_od + i] = both

        # OD ↔ OD 最近邻（对称化）
        if self.num_od > 1:
            positions = pose[:, 1 : 1 + self.num_od, :2]
            dist = torch.cdist(positions, positions)
            pair_ok = od_ok.unsqueeze(2) & od_ok.unsqueeze(1)
            eye = torch.eye(self.num_od, dtype=torch.bool, device=node_mask.device).unsqueeze(0)
            dist = dist.masked_fill(~pair_ok | eye, torch.finfo(dist.dtype).max)
            k = min(self.od_knn, self.num_od - 1)
            indices = dist.topk(k, dim=2, largest=False).indices
            od_adj = torch.zeros(
                (batch, self.num_od, self.num_od), dtype=torch.bool, device=node_mask.device
            )
            od_adj.scatter_(2, indices, True)
            od_adj = (od_adj & pair_ok) | (od_adj.transpose(1, 2) & pair_ok)
            adj[:, 1 : 1 + self.num_od, 1 : 1 + self.num_od] = od_adj
        return adj

    def build_edge_features(self, pose: Tensor, node_mask: Tensor) -> Tensor:
        """``(B,N,N,5)``：``[Δx/scale, Δy/scale, cosΔθ, sinΔθ, ‖Δp‖/scale]``，无效边清零。"""
        pos = pose[..., :2]
        theta = pose[..., 2]
        rel_xy = (pos.unsqueeze(2) - pos.unsqueeze(1)) / self.pos_scale
        delta_theta = wrap_angle(theta.unsqueeze(2) - theta.unsqueeze(1))
        radius = rel_xy.norm(dim=-1, keepdim=True)
        edge = torch.cat(
            [
                rel_xy,
                torch.cos(delta_theta).unsqueeze(-1),
                torch.sin(delta_theta).unsqueeze(-1),
                radius,
            ],
            dim=-1,
        )
        valid = (node_mask > 0.5).unsqueeze(2) & (node_mask > 0.5).unsqueeze(1)
        return edge * valid.unsqueeze(-1)

    # ---------------------------------------------------------------- 前向
    def forward(self, nodes: Tensor, node_mask: Tensor, pose: Tensor) -> Tensor:
        """``nodes (B,N,H)`` -> ``(B,N,H)``（无效节点输出为 0）。

        Round-20 M2（``exclude_ld_nodes=True``）：LD 节点从消息传递中排除（MP 掩码置 0），
        其输出**直通输入特征**（= 独立编码），供上层 set-attention 消费。
        """
        import torch

        mp_mask = node_mask
        if self.exclude_ld_nodes:
            mp_mask = node_mask.clone()
            mp_mask[:, 1 + self.num_od:] = 0.0
        adj = self.build_adjacency(mp_mask, pose)
        edge = self.build_edge_features(pose, mp_mask)
        h = nodes
        for layer in self.layers:
            h = layer(h, adj, edge, mp_mask)
        if self.exclude_ld_nodes:
            keep = node_mask.unsqueeze(-1) * (1.0 - mp_mask.unsqueeze(-1))
            h = h * (1.0 - keep) + nodes * keep
        return h
