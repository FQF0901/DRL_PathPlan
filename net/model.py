"""Canonical feature-governance planner (observation schema version 2).

This is the canonical planner: every environment semantic enters through an
encoder, the only OD/LD "selection" is a learned query attention, and the plan
head receives exactly six latent vectors::

    [ctx_ego, nav_ctx, od_pool, ld_pool, lane_ctx, od_ctx]

There is no role0 hard selection, no nearest-OD readout, no hand-computed TTC
and no raw lane/TTC/gap scalar on the plan path.  ``lane_role`` remains a
per-lane *encoder* feature because it is environment topology, not a policy
oracle.  Gap/TTC/lane-role predictions exist only as auxiliary supervision
heads and are never fed back into planning.

Input keys (batch-first, schema v2)::

    od        [B,T,16,18]
    od_mask   [B,T,16]
    ld        [B,T,16,5,16]
    ld_mask   [B,T,16,5]
    nav       [B,T,108]
    nav_mask  [B,T,1]
    ego       [B,T,12]
    ego_mask  [B,T,1]
    hist_valid[B,T]                 (optional, defaults to ones)

Optional metadata (only returned by ``encode_entities``)::

    od_ids [B,T,16], ld_ids [B,T,16]
    od_anchor [B,T,16,3], ld_anchor [B,T,16,3]

Outputs::

    action_mu     [B,4]   normalized action in [0,1] (sigmoid raw)
    action_logstd [B,4]   clamped to [-5, 0]
    value         [B,1]
    risk          [B,5]   collision/offroad/wrongroute/stuck/comfort logits
    cost_value    [B,1]
    latent        [B,H]   spatial ego context
    aux           dict:
        lane_role_logits [B,16,4]
        od_gap           [B,16]
        od_ttc           [B,16]
        od_urgent        [B,16]

``encode_entities`` returns the temporally pooled per-entity encoder latents
(``od``, ``ld``) plus nav/ego context and current-frame masks/IDs/anchors.  It
is the P4 world-model target interface: it deliberately exposes per-entity
latents, not the fused spatial context.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from env.obs import (
    EGO_DIM,
    HISTORY_FRAMES,
    LD_DIM,
    LD_POINTS,
    LD_SLOTS,
    NAV_DIM,
    OD_DIM,
    OD_SLOTS,
)

H = 128
HEADS = 4
SPATIAL_LAYERS = 2
ACTION_DIM = 4
RISK_DIM = 5
LANE_ROLE_CLASSES = 4


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Masked mean over the entity axis; all-masked rows yield zeros."""
    m = (mask > 0.5).to(x.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


class TemporalPool(nn.Module):
    """Masked temporal attention per entity token; query is the current frame."""

    def __init__(self, hidden: int = H, heads: int = HEADS, frames: int = HISTORY_FRAMES):
        super().__init__()
        self.hidden = hidden
        self.frames = frames
        self.age = nn.Embedding(frames, hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        # tokens [B,T,N,H], valid [B,T,N]
        B, T, N, Hd = tokens.shape
        x = tokens.reshape(B * N, T, Hd)
        q = x[:, -1:, :]
        k = x + self.age.weight[:T].view(1, T, Hd)
        mask = ~(valid.reshape(B * N, T) > 0.5)
        all_bad = mask.all(dim=1)
        if bool(all_bad.any()):
            mask = mask.clone()
            mask[all_bad, -1] = False
        out = self.attn(q, k, k, key_padding_mask=mask, need_weights=False)[0][:, 0]
        out = self.norm(out + q[:, 0])
        out = out * (~all_bad).to(out.dtype).unsqueeze(-1)
        return out.reshape(B, N, Hd)


class StructuredPlanner(nn.Module):
    """Feature-governed planner: encoders -> temporal pool -> spatial/cross -> plan."""

    def __init__(self, hidden: int = H, heads: int = HEADS, spatial_layers: int = SPATIAL_LAYERS):
        super().__init__()
        self.hidden = int(hidden)
        self.heads = int(heads)

        # ---- encoders: all environment semantics enter here ----
        self.od_encoder = _mlp(OD_DIM, hidden, hidden)
        self.od_type_embed = nn.Embedding(8, hidden)
        self.od_static_embed = nn.Linear(1, hidden)

        self.ld_point = _mlp(LD_DIM, hidden // 2, hidden // 2)
        self.ld_lane = nn.Sequential(
            nn.Linear(hidden // 2, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        # lane_role is an environment topology fact, kept as an encoder feature.
        self.ld_role_embed = nn.Embedding(LANE_ROLE_CLASSES, hidden)

        self.nav_encoder = _mlp(NAV_DIM, hidden, hidden)
        self.ego_encoder = _mlp(EGO_DIM, hidden, hidden)

        # ---- temporal fusion (same-ID alignment already done by env.obs) ----
        self.temporal_od = TemporalPool(hidden, heads)
        self.temporal_ld = TemporalPool(hidden, heads)
        self.temporal_nav = TemporalPool(hidden, heads)
        self.temporal_ego = TemporalPool(hidden, heads)

        # ---- current-frame spatial fusion over ego/nav/od/ld tokens ----
        self.modality = nn.Embedding(4, hidden)  # ego, nav, od, ld
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.spatial = nn.TransformerEncoder(layer, num_layers=int(spatial_layers))
        self.norm = nn.LayerNorm(hidden)

        # ---- learned entity queries (no hard role0 / nearest-OD selection) ----
        self.q_lane = nn.Parameter(torch.zeros(hidden))
        self.q_od = nn.Parameter(torch.zeros(hidden))
        self.lane_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.od_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.lane_norm = nn.LayerNorm(hidden)
        self.od_norm = nn.LayerNorm(hidden)

        # ---- plan head: exactly six latents, 6 * hidden ----
        self.plan = nn.Sequential(
            nn.Linear(6 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mu = nn.Linear(hidden, ACTION_DIM)
        self.logstd = nn.Linear(hidden, ACTION_DIM)
        self.value = nn.Linear(hidden, 1)
        self.risk = nn.Linear(hidden, RISK_DIM)
        self.cost_value = nn.Linear(hidden, 1)

        # ---- auxiliary supervision only; never plan inputs ----
        self.lane_role_head = nn.Linear(hidden, LANE_ROLE_CLASSES)
        self.od_gap_head = nn.Linear(hidden, 1)
        self.od_ttc_head = nn.Linear(hidden, 1)
        self.od_urgent_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.q_lane, std=0.02)
        nn.init.normal_(self.q_od, std=0.02)
        for head in (
            self.mu, self.logstd, self.value, self.risk, self.cost_value,
            self.lane_role_head, self.od_gap_head, self.od_ttc_head, self.od_urgent_head,
        ):
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)
        self.logstd.bias.data.fill_(-1.0)

    # ------------------------------------------------------------------
    # Input normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _history_valid(batch: Dict[str, Tensor], B: int, T: int) -> Tensor:
        if "hist_valid" in batch:
            return batch["hist_valid"].float().reshape(B, T)
        return torch.ones((B, T), dtype=torch.float32, device=batch["od"].device)

    @staticmethod
    def _frame_mask(batch: Dict[str, Tensor], key: str, B: int, T: int, dim: int) -> Tensor:
        if key in batch:
            return batch[key].float().reshape(B, T, dim)
        return torch.ones((B, T, dim), dtype=torch.float32, device=batch["od"].device)

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def _encode_frame(
        self,
        od: Tensor,
        ld: Tensor,
        ld_mask: Tensor,
        nav: Tensor,
        ego: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # od [B*T,16,18]
        od_h = self.od_encoder(od)
        od_h = od_h + self.od_type_embed(od[..., 8].long().clamp(0, 7))
        od_h = od_h + self.od_static_embed(od[..., 9:10])

        # ld [B*T,16,5,16] -> point MLP -> masked lane pooling
        M, N, P, D = ld.shape
        point = self.ld_point(ld.reshape(M * N * P, D)).reshape(M, N, P, -1)
        pm = (ld_mask.reshape(M, N, P) > 0.5).to(point.dtype).unsqueeze(-1)
        lane = (point * pm).sum(dim=2) / pm.sum(dim=2).clamp_min(1.0)
        ld_h = self.ld_lane(lane)
        ld_h = ld_h + self.ld_role_embed(ld[..., 0, 7].long().clamp(0, LANE_ROLE_CLASSES - 1))

        nav_h = self.nav_encoder(nav)
        ego_h = self.ego_encoder(ego)
        return od_h, ld_h, nav_h, ego_h

    def encode_entities(self, batch: Dict[str, Tensor] | None = None, **kwargs: Tensor) -> Dict[str, Tensor]:
        """Return per-entity latents + current masks/IDs/anchors (P4 WM target)."""
        if batch is None:
            batch = kwargs
        od = batch["od"].float()
        ld = batch["ld"].float()
        nav = batch["nav"].float()
        ego = batch["ego"].float()
        B, T = od.shape[0], od.shape[1]

        od_mask = self._frame_mask(batch, "od_mask", B, T, OD_SLOTS)
        ld_mask = self._frame_mask(batch, "ld_mask", B, T, LD_SLOTS * LD_POINTS).reshape(
            B, T, LD_SLOTS, LD_POINTS
        )
        nav_mask = self._frame_mask(batch, "nav_mask", B, T, 1)
        ego_mask = self._frame_mask(batch, "ego_mask", B, T, 1)
        hist_valid = self._history_valid(batch, B, T)

        od_h, ld_h, nav_h, ego_h = self._encode_frame(
            od.reshape(B * T, OD_SLOTS, OD_DIM),
            ld.reshape(B * T, LD_SLOTS, LD_POINTS, LD_DIM),
            ld_mask.reshape(B * T, LD_SLOTS, LD_POINTS),
            nav.reshape(B * T, NAV_DIM),
            ego.reshape(B * T, EGO_DIM),
        )
        od_h = od_h.reshape(B, T, OD_SLOTS, self.hidden)
        ld_h = ld_h.reshape(B, T, LD_SLOTS, self.hidden)
        nav_h = nav_h.reshape(B, T, 1, self.hidden)
        ego_h = ego_h.reshape(B, T, 1, self.hidden)

        hv = hist_valid[:, :, None]
        ld_lane_valid = ld_mask.any(dim=-1).float() * hv
        od_t = self.temporal_od(od_h, od_mask * hv)
        ld_t = self.temporal_ld(ld_h, ld_lane_valid)
        nav_t = self.temporal_nav(nav_h, nav_mask * hv)[:, 0]
        ego_t = self.temporal_ego(ego_h, ego_mask * hv)[:, 0]

        out: Dict[str, Tensor] = {
            "od": od_t,
            "ld": ld_t,
            "nav": nav_t,
            "ego": ego_t,
            # current-frame entity masks
            "od_mask": od_mask[:, -1],
            "ld_mask": ld_lane_valid[:, -1],
            "ld_point_mask": ld_mask[:, -1],
            "nav_mask": nav_mask[:, -1],
            "ego_mask": ego_mask[:, -1],
            "hist_valid": hist_valid,
        }
        for key in ("od_ids", "ld_ids"):
            if key in batch:
                out[key] = batch[key][:, -1]
        for key in ("od_anchor", "ld_anchor"):
            if key in batch:
                out[key] = batch[key][:, -1]
        return out

    # ------------------------------------------------------------------
    # Query attention / plan
    # ------------------------------------------------------------------

    def _query_attend(self, query: Tensor, tokens: Tensor, valid: Tensor,
                      attn: nn.MultiheadAttention, norm: nn.LayerNorm) -> Tuple[Tensor, Tensor]:
        """Return (query context, per-entity attention weights).

        The weights are exposed as *auxiliary* supervision only.  They are used
        to teach the learned lane/OD query which environment entity it should
        select; they are never fed back into the plan path.
        """
        B, N, _ = tokens.shape
        q = query.view(1, 1, -1).expand(B, 1, -1)
        mask = ~(valid > 0.5)
        all_bad = mask.all(dim=1)
        if bool(all_bad.any()):
            mask = mask.clone()
            mask[all_bad, -1] = False
        out, weights = attn(
            q, tokens, tokens,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=True,
        )
        out = norm(out[:, 0] + query)
        out = out * (~all_bad).to(out.dtype).unsqueeze(-1)
        # weights: [B, 1, N] -> [B, N]; zero invalid entities.
        weights = weights[:, 0] * (valid > 0.5).to(weights.dtype)
        return out, weights

    def forward(self, batch: Dict[str, Tensor] | None = None, **kwargs: Tensor) -> Dict[str, Tensor]:
        """Encode entities and plan; ``aux`` is supervision-only, never an input."""
        if batch is None:
            batch = kwargs
        ent = self.encode_entities(batch)
        od_t = ent["od"]
        ld_t = ent["ld"]
        nav_t = ent["nav"]
        ego_t = ent["ego"]
        od_valid = ent["od_mask"]
        ld_valid = ent["ld_mask"]
        B = od_t.shape[0]

        tokens = torch.cat([
            ego_t[:, None] + self.modality.weight[0],
            nav_t[:, None] + self.modality.weight[1],
            od_t + self.modality.weight[2],
            ld_t + self.modality.weight[3],
        ], dim=1)
        token_valid = torch.cat([
            torch.ones((B, 1), dtype=torch.bool, device=od_t.device),
            ent["nav_mask"] > 0.5,
            od_valid > 0.5,
            ld_valid > 0.5,
        ], dim=1)
        token_valid[:, 0] = True
        latent = self.spatial(tokens, src_key_padding_mask=~token_valid)
        ctx_ego = self.norm(latent[:, 0])

        lane_ctx, lane_query_weights = self._query_attend(
            self.q_lane, ld_t, ld_valid, self.lane_attn, self.lane_norm
        )
        od_ctx, od_query_weights = self._query_attend(
            self.q_od, od_t, od_valid, self.od_attn, self.od_norm
        )

        od_pool = _masked_mean(od_t, od_valid)
        ld_pool = _masked_mean(ld_t, ld_valid)

        fused = self.plan(torch.cat(
            [ctx_ego, nav_t, od_pool, ld_pool, lane_ctx, od_ctx], dim=-1
        ))

        aux = {
            "lane_role_logits": self.lane_role_head(ld_t),
            "od_gap": self.od_gap_head(od_t).squeeze(-1),
            "od_ttc": self.od_ttc_head(od_t).squeeze(-1),
            "od_urgent": self.od_urgent_head(od_t).squeeze(-1),
            # Query-selection supervision: current lane / urgent OD.  Aux only.
            "lane_query_weights": lane_query_weights,
            "od_query_weights": od_query_weights,
        }

        return {
            "action_mu": torch.sigmoid(self.mu(fused)),
            "action_logstd": self.logstd(fused).clamp(-5.0, 0.0),
            "value": self.value(fused),
            "risk": self.risk(fused),
            "cost_value": self.cost_value(fused),
            "aux": aux,
            "latent": ctx_ego,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
