"""Policy and auxiliary heads for the factorized bank planner."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


def masked_mean(x: Tensor, valid: Tensor) -> Tensor:
    m = (valid > 0.5).to(x.dtype).unsqueeze(-1)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class PlanHead(nn.Module):
    """Deterministic policy-head scaffold with a diagonal Gaussian output."""

    def __init__(self, hidden: int = 128, action_dim: int = 4) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden), nn.GELU(),
            nn.Linear(2 * hidden, hidden), nn.GELU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.logstd = nn.Linear(hidden, action_dim)
        nn.init.constant_(self.logstd.bias, -1.0)

    def forward(
        self,
        od: Tensor,
        ld: Tensor,
        nav: Tensor,
        ego: Tensor,
        od_valid: Tensor,
        ld_valid: Tensor,
    ) -> Dict[str, Tensor]:
        od_pool = masked_mean(od, od_valid)
        ld_pool = masked_mean(ld, ld_valid)
        context = self.context(torch.cat([ego, nav, od_pool, ld_pool], dim=-1))
        return {
            "action_mu": torch.sigmoid(self.mu(context)),
            "action_logstd": self.logstd(context).clamp(-5.0, 0.0),
            "latent": context,
        }


class AuxHead(nn.Module):
    """Auxiliary safety/progress predictions; never a plan input."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.collision = nn.Linear(hidden, 1)
        self.offroad = nn.Linear(hidden, 1)
        self.timeout = nn.Linear(hidden, 1)
        self.urgency = nn.Linear(hidden, 1)
        self.progress = nn.Linear(hidden, 1)

    def forward(self, context: Tensor) -> Dict[str, Tensor]:
        return {
            "collision": self.collision(context),
            "offroad": self.offroad(context),
            "timeout": self.timeout(context),
            "urgency": self.urgency(context),
            "progress": self.progress(context),
        }


class Planner(nn.Module):
    def __init__(self, hidden: int = 128, action_dim: int = 4) -> None:
        super().__init__()
        self.plan = PlanHead(hidden, action_dim)
        self.aux = AuxHead(hidden)

    def forward(
        self,
        od: Tensor,
        ld: Tensor,
        nav: Tensor,
        ego: Tensor,
        od_valid: Tensor,
        ld_valid: Tensor,
    ) -> Dict[str, Tensor]:
        out = self.plan(od, ld, nav, ego, od_valid, ld_valid)
        out["aux"] = self.aux(out["latent"])
        return out
