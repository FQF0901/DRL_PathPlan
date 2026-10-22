"""Ego-action-conditioned latent world model (Stage D1/D2 interface).

The model operates on the frozen planner's per-entity latents, not raw sensor
tokens.  It predicts the next per-entity latent for each current slot, with a
strict ``ego action`` condition.  The target encoder is frozen by the caller;
this module contains only the transition model and a small autoregressive
rollout helper for D2 probing.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class EgoConditionedLatentWM(nn.Module):
    """One-step latent residual model with ego-action conditioning."""

    def __init__(
        self,
        *,
        hidden: int = 80,
        latent_dim: int = 128,
        od_slots: int = 16,
        ld_slots: int = 16,
        action_dim: int = 4,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.action_dim = int(action_dim)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.ego_proj = nn.Linear(self.latent_dim, hidden)
        self.nav_proj = nn.Linear(self.latent_dim, hidden)
        self.od_proj = nn.Linear(self.latent_dim, hidden)
        self.ld_proj = nn.Linear(self.latent_dim, hidden)
        self.modality = nn.Embedding(4, hidden)  # ego, nav, od, ld
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=4 * hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=int(layers))
        self.norm = nn.LayerNorm(hidden)
        self.od_head = nn.Linear(hidden, self.latent_dim)
        self.ld_head = nn.Linear(hidden, self.latent_dim)
        self.nav_head = nn.Linear(hidden, self.latent_dim)
        self.ego_head = nn.Linear(hidden, self.latent_dim)
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # Residual heads start at zero so D1 initializes exactly at persistence;
        # otherwise the WM can be worse than the baseline for many epochs.
        for head in (self.od_head, self.ld_head, self.nav_head, self.ego_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        *,
        od: Tensor,
        ld: Tensor,
        nav: Tensor,
        ego: Tensor,
        od_mask: Tensor,
        ld_mask: Tensor,
        action: Tensor,
    ) -> Dict[str, Tensor]:
        """Predict next latents in the current slot order.

        Shapes:
        - od [B,16,H], ld [B,16,H], nav/ego [B,H]
        - od_mask [B,16], ld_mask [B,16]
        - action [B,4] physical/normalized action
        """
        b = od.shape[0]
        action_ctx = self.action_mlp(action)
        ego_t = self.ego_proj(ego) + action_ctx + self.modality.weight[0]
        nav_t = self.nav_proj(nav) + action_ctx + self.modality.weight[1]
        od_t = self.od_proj(od) + action_ctx[:, None, :] + self.modality.weight[2]
        ld_t = self.ld_proj(ld) + action_ctx[:, None, :] + self.modality.weight[3]

        tokens = torch.cat(
            [ego_t[:, None, :], nav_t[:, None, :], od_t, ld_t], dim=1
        )
        valid = torch.cat(
            [
                torch.ones((b, 1), dtype=torch.bool, device=od.device),
                torch.ones((b, 1), dtype=torch.bool, device=od.device),
                od_mask > 0.5,
                ld_mask > 0.5,
            ],
            dim=1,
        )
        out = self.fusion(tokens, src_key_padding_mask=~valid)
        out = self.norm(out)
        ego_out = out[:, 0]
        nav_out = out[:, 1]
        od_out = out[:, 2 : 2 + self.od_slots]
        ld_out = out[:, 2 + self.od_slots :]
        return {
            "od": od + self.od_head(od_out),
            "ld": ld + self.ld_head(ld_out),
            "nav": nav + self.nav_head(nav_out),
            "ego": ego + self.ego_head(ego_out),
        }


def masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Mean squared error over valid entries, with zero when no entries."""
    m = (mask > 0.5).to(pred.dtype)
    while m.ndim < pred.ndim:
        m = m.unsqueeze(-1)
    # Count every valid latent element, not just valid slots.  Otherwise the
    # denominator is H× too small and all latent MSEs are inflated.
    m = m.expand_as(pred)
    denom = m.sum().clamp_min(1.0)
    return ((pred - target).pow(2) * m).sum() / denom


def rollout_latents(
    wm: EgoConditionedLatentWM,
    initial: Dict[str, Tensor],
    actions: Tensor,
    masks: Dict[str, Tensor],
) -> Tuple[Dict[str, Tensor], list[Dict[str, Tensor]]]:
    """Autoregressive 1..H rollout with externally supplied actions.

    ``actions`` is [B,H,4].  This is a diagnostic interface for D2; it does not
    add an imagined reward or change the planner.
    """
    state = {key: value for key, value in initial.items()}
    history: list[Dict[str, Tensor]] = []
    for t in range(int(actions.shape[1])):
        state = wm(
            od=state["od"],
            ld=state["ld"],
            nav=state["nav"],
            ego=state["ego"],
            od_mask=masks["od"],
            ld_mask=masks["ld"],
            action=actions[:, t],
        )
        history.append(state)
    return state, history
