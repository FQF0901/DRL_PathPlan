"""Legacy latent WM plus the factorized bank-native world model.

``EgoConditionedLatentWM`` below is retained for the legacy D1/D2 pipeline.
New code should use ``TemporalSpatialBlock`` and
``FactorizedEgoConditionedLatentWM``.

Legacy description follows.

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


# ---------------------------------------------------------------------------
# Factorized temporal-spatial encoder and the new OD/LD world model
# ---------------------------------------------------------------------------

from net.mem_bank import BankWindow, BankWindowBuilder, NodeType


class TemporalAttention(nn.Module):
    """Per-entity masked temporal attention with stable-ID filtering.

    Inputs are batch-first ``[B,T,N,H]``; output is the current entity token
    ``[B,N,H]``.  The implementation deliberately uses ``permute`` before the
    reshape so ``N`` and ``T`` are never conflated.
    """

    def __init__(self, hidden: int, heads: int, frames: int = 6) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.frames = int(frames)
        self.age = nn.Embedding(self.frames, self.hidden)
        self.attn = nn.MultiheadAttention(self.hidden, int(heads), batch_first=True)
        self.norm = nn.LayerNorm(self.hidden)

    def forward(self, tokens: Tensor, valid: Tensor, stable_id: Tensor) -> Tensor:
        b, t, n, h = tokens.shape
        if t > self.frames:
            raise ValueError(f"window has {t} frames, max {self.frames}")
        x = tokens.permute(0, 2, 1, 3).reshape(b * n, t, h)
        v = valid.permute(0, 2, 1).reshape(b * n, t)
        ids = stable_id.permute(0, 2, 1).reshape(b * n, t)
        q = x[:, -1:, :]
        k = x + self.age.weight[:t].view(1, t, h)
        same = ids == ids[:, -1:]
        mask = ~((v > 0.5) & same)
        all_bad = mask.all(dim=1)
        if bool(all_bad.any()):
            mask = mask.clone()
            mask[all_bad, -1] = False
        out = self.attn(q, k, k, key_padding_mask=mask, need_weights=False)[0][:, 0]
        out = self.norm(out + q[:, 0])
        out = out * (~all_bad).to(out.dtype).unsqueeze(-1)
        return out.reshape(b, n, h)


class TemporalSpatialBlock(nn.Module):
    """Factorized temporal then current-frame spatial fusion for OD/LD.

    Ego/Nav are current policy inputs, not Bank records.  They participate as
    conditioning tokens in the spatial transformer but are not returned as
    bank entities.
    """

    def __init__(
        self,
        hidden: int = 128,
        heads: int = 4,
        spatial_layers: int = 2,
        od_slots: int = 16,
        ld_slots: int = 16,
        type_count: int = 8,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.temporal_od = TemporalAttention(hidden, heads)
        self.temporal_ld = TemporalAttention(hidden, heads)
        self.pose = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.type_od = nn.Embedding(type_count, hidden)
        self.type_ld = nn.Embedding(type_count, hidden)
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
        self.spatial = nn.TransformerEncoder(layer, num_layers=int(spatial_layers))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, window: BankWindow, nav: Tensor, ego: Tensor) -> Dict[str, Tensor]:
        if nav.shape[-1] != self.hidden or ego.shape[-1] != self.hidden:
            raise ValueError("nav/ego must have hidden size")
        od = self.temporal_od(window.od_content, window.od_valid, window.od_stable_id)
        ld = self.temporal_ld(window.ld_content, window.ld_valid, window.ld_stable_id)

        od_pose = BankWindowBuilder.relative_pose(window.od_geometry, NodeType.OD)[:, -1]
        ld_pose = BankWindowBuilder.relative_pose(window.ld_geometry, NodeType.LD)[:, -1]
        od = od + self.pose(od_pose) + self.type_od(window.od_type[:, -1].clamp(0, self.type_od.num_embeddings - 1))
        ld = ld + self.pose(ld_pose) + self.type_ld(window.ld_type[:, -1].clamp(0, self.type_ld.num_embeddings - 1))

        b = od.shape[0]
        od_valid = window.od_valid[:, -1]
        ld_valid = window.ld_valid[:, -1]
        tokens = torch.cat(
            [
                ego[:, None] + self.modality.weight[0],
                nav[:, None] + self.modality.weight[1],
                od + self.modality.weight[2],
                ld + self.modality.weight[3],
            ],
            dim=1,
        )
        valid = torch.cat(
            [
                torch.ones((b, 1), dtype=torch.bool, device=od.device),
                torch.ones((b, 1), dtype=torch.bool, device=od.device),
                od_valid > 0.5,
                ld_valid > 0.5,
            ],
            dim=1,
        )
        out = self.spatial(tokens, src_key_padding_mask=~valid)
        out = self.norm(out)
        od_out = out[:, 2 : 2 + self.od_slots]
        ld_out = out[:, 2 + self.od_slots : 2 + self.od_slots + self.ld_slots]
        return {"od": od_out, "ld": ld_out, "tokens": out}


class ODGeometryDecoder(nn.Module):
    """Predict OD relative pose in the current query ego frame."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, content: Tensor) -> Tensor:
        return self.net(content)


class LDResidualDecoder(nn.Module):
    """Predict a small LD polyline residual; the explicit projection is the base."""

    def __init__(self, hidden: int = 128, action_dim: int = 4, points: int = 5) -> None:
        super().__init__()
        self.points = int(points)
        self.net = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * self.points)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, content: Tensor, action: Tensor) -> Tensor:
        b, n, _ = content.shape
        a = action[:, None, :].expand(b, n, -1)
        return self.net(torch.cat([content, a], dim=-1)).reshape(b, n, self.points, 2)


def project_ld(geometry: Tensor, source_pose: Tensor, target_pose: Tensor) -> Tensor:
    """Project LD polyline geometry into ``target_pose`` ego coordinates."""
    return BankWindowBuilder._transform_geometry(
        geometry, source_pose, target_pose, node_type=NodeType.LD
    )


def add_ld_residual(projected: Tensor, residual: Tensor) -> Tensor:
    """Add a ``[...,5,2]`` residual to flattened ``[...,10]`` LD geometry."""
    return projected + residual.reshape(*residual.shape[:-2], -1)


class FactorizedEgoConditionedLatentWM(nn.Module):
    """OD content transition + OD pose decoder + LD residual decoder.

    LD content is not a future target.  The LD branch only learns a residual
    on top of explicit polygon projection.
    """

    def __init__(self, hidden: int = 128, action_dim: int = 4, points: int = 5) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.action = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.od_delta = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.od_geometry = ODGeometryDecoder(hidden)
        self.ld_residual = LDResidualDecoder(hidden, action_dim=action_dim, points=points)

    def forward(self, od: Tensor, ld: Tensor, action: Tensor) -> Dict[str, Tensor]:
        b, n, _ = od.shape
        a = self.action(action)[:, None, :].expand(b, n, -1)
        od_next = od + self.od_delta(torch.cat([od, a], dim=-1))
        return {
            "od": od_next,
            "ld": ld,
            "od_geometry": self.od_geometry(od_next),
            "ld_residual": self.ld_residual(ld, action),
        }

    @torch.no_grad()
    def rollout(self, od: Tensor, ld: Tensor, actions: Tensor) -> Dict[str, Tensor]:
        """Pure tensor rollout helper; actions are ``[B,H,4]``."""
        state = {"od": od, "ld": ld}
        history = []
        for t in range(int(actions.shape[1])):
            state = self(state["od"], state["ld"], actions[:, t])
            history.append(state)
        return {"final": state, "history": history}


# The old :class:`EgoConditionedLatentWM` above is retained for the legacy
# D1/D2 pipeline.  New code should use the factorized class.
EgoConditionedLatentWMV2 = FactorizedEgoConditionedLatentWM
