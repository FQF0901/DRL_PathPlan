"""Small structured planner used by P1 BC.

Input keys (batch-first):
  od        [B,T,16,11]
  od_mask   [B,T,16]
  ld        [B,T,16,5,8]
  ld_mask   [B,T,16,5]
  nav       [B,T,11]
  nav_mask  [B,T,1]
  ego       [B,T,8]
  ego_mask  [B,T,1]
  hist_valid[B,T]

Output:
  action_mu     [B,4]  (dx,dy,dpsi,v_next)
  action_logstd [B,4]
  value         [B,1]
  risk          [B,1]
  latent        [B,H]
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from env.obs import (
    EGO_DIM, HISTORY_FRAMES, LD_DIM, LD_POINTS, LD_SLOTS, NAV_DIM,
    OD_DIM, OD_SLOTS,
)

H = 128
HEADS = 4
SPATIAL_LAYERS = 2
ACTION_DIM = 4
ACTION_LOW = torch.tensor([0.0, -2.0, -0.6, 0.0])
ACTION_HIGH = torch.tensor([10.0, 2.0, 0.6, 30.0])


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


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
        # every row has at least one valid frame in normal use; guard all-invalid rows
        all_bad = mask.all(dim=1)
        if bool(all_bad.any()):
            mask = mask.clone()
            mask[all_bad, -1] = False
        out = self.attn(q, k, k, key_padding_mask=mask, need_weights=False)[0][:, 0]
        out = self.norm(out + q[:, 0])
        out = out * (~all_bad).to(out.dtype).unsqueeze(-1)
        return out.reshape(B, N, Hd)


class StructuredPlanner(nn.Module):
    def __init__(self, hidden: int = H, heads: int = HEADS, spatial_layers: int = SPATIAL_LAYERS):
        super().__init__()
        self.hidden = int(hidden)
        self.od_encoder = _mlp(OD_DIM, hidden, hidden)
        self.ld_point = _mlp(LD_DIM, hidden // 2, hidden // 2)
        self.ld_lane = nn.Sequential(nn.Linear(hidden // 2, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.nav_encoder = _mlp(NAV_DIM, hidden, hidden)
        self.ego_encoder = _mlp(EGO_DIM, hidden, hidden)
        self.type_embed = nn.Embedding(8, hidden)
        self.static_embed = nn.Linear(1, hidden)
        self.role_embed = nn.Embedding(4, hidden)
        self.modality = nn.Embedding(4, hidden)  # ego, nav, od, ld

        self.temporal_od = TemporalPool(hidden, heads)
        self.temporal_ld = TemporalPool(hidden, heads)
        self.temporal_nav = TemporalPool(hidden, heads)
        self.temporal_ego = TemporalPool(hidden, heads)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.spatial = nn.TransformerEncoder(layer, num_layers=int(spatial_layers))
        self.norm = nn.LayerNorm(hidden)
        self.plan = nn.Sequential(
            nn.Linear(7 * hidden + 3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mu = nn.Linear(hidden, ACTION_DIM)
        self.logstd = nn.Linear(hidden, ACTION_DIM)
        self.value = nn.Linear(hidden, 1)
        self.risk = nn.Linear(hidden, 1)

        # bounded-action helpers (raw output is squashed in loss/inference)
        self.register_buffer("action_low", ACTION_LOW, persistent=False)
        self.register_buffer("action_high", ACTION_HIGH, persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # small final heads: not zero, to keep gradients alive
        for head in (self.mu, self.logstd, self.value, self.risk):
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)
        self.logstd.bias.data.fill_(-1.0)

    def _encode_frame(self, od: Tensor, ld: Tensor, nav: Tensor, ego: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # od [B*T,N,D]
        od_h = self.od_encoder(od)
        od_h = od_h + self.type_embed(od[..., 8].long().clamp(0, 7))
        od_h = od_h + self.static_embed(od[..., 9:10])
        # ld [B*T,N,P,D]
        B, N, P, D = ld.shape
        point = self.ld_point(ld.reshape(B * N * P, D)).reshape(B, N, P, -1)
        lane = point.mean(dim=2)
        ld_h = self.ld_lane(lane)
        ld_h = ld_h + self.role_embed(ld[..., 0, 7].long().clamp(0, 3))
        nav_h = self.nav_encoder(nav)
        ego_h = self.ego_encoder(ego)
        return od_h, ld_h, nav_h, ego_h

    def forward(self, batch: Dict[str, Tensor] | None = None, **kwargs: Tensor) -> Dict[str, Tensor]:
        if batch is None:
            batch = kwargs
        od = batch["od"].float()
        ld = batch["ld"].float()
        nav = batch["nav"].float()
        ego = batch["ego"].float()
        od_mask = batch["od_mask"].float()
        ld_mask = batch["ld_mask"].float()
        nav_mask = batch["nav_mask"].float().reshape(od.shape[0], od.shape[1], 1)
        ego_mask = batch["ego_mask"].float().reshape(od.shape[0], od.shape[1], 1)
        hist_valid = batch.get("hist_valid", torch.ones_like(ego_mask[..., 0])).float()
        B, T = od.shape[0], od.shape[1]

        od_h, ld_h, nav_h, ego_h = self._encode_frame(
            od.reshape(B * T, OD_SLOTS, OD_DIM),
            ld.reshape(B * T, LD_SLOTS, LD_POINTS, LD_DIM),
            nav.reshape(B * T, NAV_DIM),
            ego.reshape(B * T, EGO_DIM),
        )
        od_h = od_h.reshape(B, T, OD_SLOTS, self.hidden)
        ld_h = ld_h.reshape(B, T, LD_SLOTS, self.hidden)
        nav_h = nav_h.reshape(B, T, 1, self.hidden)
        ego_h = ego_h.reshape(B, T, 1, self.hidden)

        hv = hist_valid[:, :, None]
        od_valid = od_mask * hv
        ld_lane_valid = ld_mask.any(dim=-1).float() * hv
        nav_valid = nav_mask * hv
        ego_valid = ego_mask * hv

        od_t = self.temporal_od(od_h, od_valid)
        ld_t = self.temporal_ld(ld_h, ld_lane_valid)
        nav_t = self.temporal_nav(nav_h, nav_valid)[:, 0]
        ego_t = self.temporal_ego(ego_h, ego_valid)[:, 0]

        tokens = torch.cat([
            ego_t[:, None] + self.modality.weight[0],
            nav_t[:, None] + self.modality.weight[1],
            od_t + self.modality.weight[2],
            ld_t + self.modality.weight[3],
        ], dim=1)
        token_valid = torch.cat([
            torch.ones((B, 1), dtype=torch.bool, device=ego.device),
            torch.ones((B, 1), dtype=torch.bool, device=ego.device),
            od_mask[:, -1] > 0.5,
            ld_lane_valid[:, -1] > 0.5,
        ], dim=1)
        # ensure at least one valid token
        token_valid[:, 0] = True
        latent = self.spatial(tokens, src_key_padding_mask=~token_valid)
        ctx = self.norm(latent[:, 0])

        od_pool = _masked_mean(od_t, od_valid[:, -1])
        ld_pool = _masked_mean(ld_t, ld_lane_valid[:, -1])
        role_last = batch["ld"][:, -1, :, 0, 7].long()
        role0 = (role_last == 0).to(ld_t.dtype) * ld_lane_valid[:, -1]
        ld_role0 = _masked_mean(ld_t, role0)
        lat0 = _masked_mean(batch["ld"][:, -1, :, 0, 8:9], role0)
        # explicit nearest-OD readout: a masked mean over 16 ODs dilutes the urgent one.
        od_pos = batch["od"][:, -1, :, 0:2]
        od_dist = torch.linalg.norm(od_pos, dim=-1)
        od_valid_last = od_valid[:, -1] > 0.5
        od_dist = torch.where(od_valid_last, od_dist, torch.full_like(od_dist, 1e6))
        nearest_idx = od_dist.argmin(dim=1)
        od_near = od_t[torch.arange(B, device=od_t.device), nearest_idx]
        od_min = od_dist[torch.arange(B, device=od_t.device), nearest_idx].clamp_max(100.0)[:, None] / 100.0
        od_vel = batch["od"][:, -1, :, 4:6]
        gap = torch.where(od_pos[..., 0] > 0, od_pos[..., 0], torch.full_like(od_pos[..., 0], 1e6))
        closing = torch.clamp(-od_vel[..., 0], min=0.0)
        ttc = torch.where(closing > 0.1, gap / closing.clamp_min(0.1), torch.full_like(gap, 1e6))
        ttc = torch.where(od_valid_last, ttc, torch.full_like(ttc, 1e6))
        ttc_idx = ttc.argmin(dim=1)
        od_urgent = od_t[torch.arange(B, device=od_t.device), ttc_idx]
        ttc_min = ttc[torch.arange(B, device=od_t.device), ttc_idx].clamp_max(10.0)[:, None] / 10.0
        fused = self.plan(torch.cat([ctx, nav_t, od_pool, ld_pool, ld_role0, od_near, od_urgent, lat0, od_min, ttc_min], dim=-1))
        mu = self.mu(fused)
        logstd = self.logstd(fused).clamp(-5.0, 0.0)
        return {
            "action_mu": mu,
            "action_logstd": logstd,
            "value": self.value(fused),
            "risk": self.risk(fused),
            "latent": ctx,
        }


def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    m = (mask > 0.5).to(x.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
