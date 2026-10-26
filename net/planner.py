"""Policy and auxiliary heads for the bank-native planner.

The policy head is intentionally small: the shared TemporalSpatialBlock has
already encoded OD/LD history and current-frame interactions.  Here ego/nav
act as queries over the entity latents; no second temporal encoder is used.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


def masked_mean(x: Tensor, valid: Tensor) -> Tensor:
    m = (valid > 0.5).to(x.dtype).unsqueeze(-1)
    x = torch.where(m > 0.5, x, torch.zeros_like(x))
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class PlanHead(nn.Module):
    """Ego/nav-query cross-attention + action distribution."""

    def __init__(self, hidden: int = 128, action_dim: int = 4, heads: int = 4) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by policy heads")
        self.hidden = int(hidden)
        self.query = nn.Linear(2 * hidden, hidden)
        self.cross = nn.MultiheadAttention(hidden, int(heads), batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden)
        )
        self.context = nn.Sequential(
            nn.Linear(5 * hidden, 2 * hidden), nn.GELU(),
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
        b = od.shape[0]
        kv = torch.cat([od, ld], dim=1)
        valid = torch.cat([od_valid > 0.5, ld_valid > 0.5], dim=1)
        # MultiheadAttention does not accept a fully-masked row.  Keep a
        # numerical anchor; its output is not used semantically for invalid
        # entities.
        if not bool(valid.any(dim=1).all()):
            valid = valid.clone()
            valid[~valid.any(dim=1), 0] = True
        kv = torch.where(valid[..., None], kv, torch.zeros_like(kv))
        q = self.query(torch.cat([ego, nav], dim=-1)).unsqueeze(1)
        attn = self.cross(q, kv, kv, key_padding_mask=~valid, need_weights=False)[0][:, 0]
        attn = self.norm(attn + q[:, 0])
        attn = attn + self.ffn(attn)
        od_pool = masked_mean(od, od_valid)
        ld_pool = masked_mean(ld, ld_valid)
        context = self.context(torch.cat([attn, ego, nav, od_pool, ld_pool], dim=-1))
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
    def __init__(self, hidden: int = 128, action_dim: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.plan = PlanHead(hidden, action_dim, heads)
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
