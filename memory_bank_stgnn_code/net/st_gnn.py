"""Small typed spatiotemporal GNN for BankView inputs (no PyG dependency)."""
from __future__ import annotations

from typing import Dict
import torch
from torch import Tensor, nn
from net.memory_bank import BankView, NodeType


def _scatter_sum(src: Tensor, index: Tensor, size: int) -> Tensor:
    out = src.new_zeros((size, src.shape[-1]))
    if index.numel():
        out.index_add_(0, index, src)
    return out


class TypedMessageLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.gate = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.update = nn.GRUCell(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        if edge_index.numel() == 0:
            return x
        src, dst = edge_index
        z = torch.cat((x[src], x[dst], edge_attr), dim=-1)
        msg = self.message(z) * torch.sigmoid(self.gate(z))
        agg = _scatter_sum(msg, dst, x.shape[0])
        degree = torch.bincount(dst, minlength=x.shape[0]).clamp_min(1).to(x.dtype).unsqueeze(-1)
        updated = self.update(agg / degree, x)
        return self.norm(x + updated)


class STGNN(nn.Module):
    """Temporal same-ID messages followed by current-frame spatial messages."""
    def __init__(self, hidden: int = 128, geometry_dim: int = 5, layers: int = 2, max_depth: int = 6):
        super().__init__()
        self.hidden = int(hidden)
        self.node_type = nn.Embedding(4, hidden)
        self.source = nn.Embedding(2, hidden)
        self.depth = nn.Embedding(max_depth + 1, hidden)
        self.birth = nn.Embedding(2, hidden)
        self.geometry = nn.Sequential(nn.Linear(geometry_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        # temporal attr: dt, dx, dy, distance, sin/cos dyaw
        self.temporal_layers = nn.ModuleList([TypedMessageLayer(hidden, 6) for _ in range(layers)])
        self.spatial_layers = nn.ModuleList([TypedMessageLayer(hidden, 6) for _ in range(layers)])
        self.out_norm = nn.LayerNorm(hidden)

    def _temporal_attr(self, view: BankView) -> Tensor:
        if view.temporal_edge_index.numel() == 0:
            return view.x.new_empty((0, 6))
        src, dst = view.temporal_edge_index
        dg = view.geometry[dst, :3] - view.geometry[src, :3]
        dt = (view.timestamp[dst] - view.timestamp[src]).unsqueeze(-1)
        dist = torch.linalg.vector_norm(dg[:, :2], dim=-1, keepdim=True)
        return torch.cat((dt, dg[:, :2], dist, torch.sin(dg[:, 2:3]), torch.cos(dg[:, 2:3])), dim=-1)

    def forward(self, view: BankView) -> Dict[str, Tensor]:
        g = view.geometry
        if g.shape[1] < self.geometry[0].in_features:
            g = torch.nn.functional.pad(g, (0, self.geometry[0].in_features - g.shape[1]))
        else:
            g = g[:, :self.geometry[0].in_features]
        x = (view.x + self.node_type(view.node_type) + self.source(view.source)
             + self.depth(view.rollout_depth.clamp(0, self.depth.num_embeddings - 1))
             + self.birth(view.birth.long()) + self.geometry(g))
        temporal_attr = self._temporal_attr(view)
        for layer in self.temporal_layers:
            x = layer(x, view.temporal_edge_index, temporal_attr)
        for layer in self.spatial_layers:
            x = layer(x, view.spatial_edge_index, view.spatial_edge_attr)
        x = self.out_norm(x)
        current = view.readout_mask
        result: Dict[str, Tensor] = {"all": x, "current": x[current], "current_type": view.node_type[current]}
        for kind, name in ((NodeType.EGO, "ego"), (NodeType.NAV, "nav"), (NodeType.OD, "od"), (NodeType.LD, "ld")):
            result[name] = x[current & (view.node_type == int(kind))]
        return result
