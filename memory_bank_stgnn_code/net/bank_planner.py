"""Bank-native planner core and minimal ego-conditioned imagined rollout."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, Tuple
import torch
from torch import Tensor, nn
from net.memory_bank import (BankViewBuilder, EntityRecord, FrameRecord, ImagineMemoryBank,
                             NodeType, Source)
from net.st_gnn import STGNN


class BankPlannerCore(nn.Module):
    """Consumes encoded entity records from a MemoryBank; no stacked observation API."""
    def __init__(self, hidden: int = 128, action_dim: int = 4):
        super().__init__()
        self.st_gnn = STGNN(hidden=hidden)
        self.ego_fallback = nn.Parameter(torch.zeros(hidden))
        self.nav_fallback = nn.Parameter(torch.zeros(hidden))
        self.od_query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.ld_query = nn.Parameter(torch.randn(hidden) * 0.02)
        self.plan = nn.Sequential(nn.Linear(6 * hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, action_dim))
        self.value = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.risk = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 5))

    @staticmethod
    def _one_or(tokens: Tensor, fallback: Tensor) -> Tensor:
        return tokens[-1] if tokens.numel() else fallback

    @staticmethod
    def _query_pool(tokens: Tensor, query: Tensor) -> Tensor:
        if tokens.numel() == 0:
            return torch.zeros_like(query)
        score = (tokens * query).sum(-1) / (tokens.shape[-1] ** 0.5)
        return (torch.softmax(score, dim=0).unsqueeze(-1) * tokens).sum(0)

    def forward(self, view) -> Dict[str, Tensor]:
        z = self.st_gnn(view)
        ego = self._one_or(z["ego"], self.ego_fallback)
        nav = self._one_or(z["nav"], self.nav_fallback)
        od_pool = self._query_pool(z["od"], self.od_query)
        ld_pool = self._query_pool(z["ld"], self.ld_query)
        od_mean = z["od"].mean(0) if z["od"].numel() else torch.zeros_like(ego)
        ld_mean = z["ld"].mean(0) if z["ld"].numel() else torch.zeros_like(ego)
        context = torch.cat((ego, nav, od_pool, ld_pool, od_mean, ld_mean), dim=-1)
        return {"action_mu": torch.sigmoid(self.plan(context)), "value": self.value(context),
                "risk": self.risk(context), "latent": ego, "entity_latent": z}


class DynamicODWorldModel(nn.Module):
    """Predict only dynamic-OD residual and survival; ego/LD/Nav are analytic/copied."""
    def __init__(self, hidden: int = 128, action_dim: int = 4):
        super().__init__()
        self.action = nn.Sequential(nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.delta = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.alive = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, od: Tensor, context: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        a = self.action(action).expand_as(od)
        c = context.expand_as(od)
        joint = torch.cat((od, c, a), dim=-1)
        return od + self.delta(joint), self.alive(joint).squeeze(-1)


def advance_ego_pose(pose: Tensor, action: Tensor, dt: float = 0.5) -> Tensor:
    """Simple deterministic placeholder; replace with the project's canonical action decoder.

    Assumes normalized action [d_longitudinal, d_lateral, d_yaw, speed-like].
    The explicit function prevents the latent WM from learning the ego coordinate transform.
    """
    out = pose.clone()
    ds = action[0] * dt
    dyaw = (action[2] * 2.0 - 1.0) * 0.35 * dt
    out[0] = pose[0] + torch.cos(pose[2]) * ds
    out[1] = pose[1] + torch.sin(pose[2]) * ds
    out[2] = torch.atan2(torch.sin(pose[2] + dyaw), torch.cos(pose[2] + dyaw))
    return out
