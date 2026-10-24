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

from net.mem_bank import BankWindow, BankWindowBuilder, NodeType


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



class TemporalAttention(nn.Module):
    """Masked temporal attention over content + relative-pose tokens.

    Inputs are batch-first ``[B,T,N,H]``; output is the current entity token
    ``[B,N,H]``.  The current-frame validity is applied to the whole output.
    """

    def __init__(self, hidden: int, heads: int, frames: int = 6) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.frames = int(frames)
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
        same = ids == ids[:, -1:]
        mask = ~((v > 0.5) & same)
        all_bad = mask.all(dim=1)
        if bool(all_bad.any()):
            mask = mask.clone()
            mask[all_bad, -1] = False
        out = self.attn(q, x, x, key_padding_mask=mask, need_weights=False)[0][:, 0]
        out = self.norm(out + q[:, 0])
        current_valid = v[:, -1].to(out.dtype)
        out = out * current_valid.unsqueeze(-1) * (~all_bad).to(out.dtype).unsqueeze(-1)
        return out.reshape(b, n, h)


class TemporalSpatialBlock(nn.Module):
    """Factorized temporal then current-frame spatial fusion for OD/LD.

    Ego/Nav are current policy inputs, not Bank records.  Their optional masks
    are applied in the spatial transformer.
    """

    def __init__(
        self,
        hidden: int = 128,
        heads: int = 4,
        spatial_layers: int = 2,
        od_slots: int = 16,
        ld_slots: int = 16,
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

    def forward(
        self,
        window: BankWindow,
        nav: Tensor,
        ego: Tensor,
        *,
        nav_valid: Tensor | None = None,
        ego_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if nav.shape[-1] != self.hidden or ego.shape[-1] != self.hidden:
            raise ValueError("nav/ego must have hidden size")
        if window.od_content.shape[2] != self.od_slots or window.ld_content.shape[2] != self.ld_slots:
            raise ValueError("BankWindow slot count does not match TemporalSpatialBlock")

        od_pose = BankWindowBuilder.relative_pose(window.od_geometry, NodeType.OD)
        ld_pose = BankWindowBuilder.relative_pose(window.ld_geometry, NodeType.LD)
        od_tokens = window.od_content + self.pose(od_pose)
        ld_tokens = window.ld_content + self.pose(ld_pose)
        od = self.temporal_od(od_tokens, window.od_valid, window.od_stable_id)
        ld = self.temporal_ld(ld_tokens, window.ld_valid, window.ld_stable_id)

        b = od.shape[0]
        od_valid = window.od_valid[:, -1]
        ld_valid = window.ld_valid[:, -1]
        ego_ok = torch.ones((b, 1), dtype=torch.bool, device=od.device) if ego_valid is None else ego_valid.reshape(b, 1) > 0.5
        nav_ok = torch.ones((b, 1), dtype=torch.bool, device=od.device) if nav_valid is None else nav_valid.reshape(b, 1) > 0.5
        tokens = torch.cat(
            [
                ego[:, None] + self.modality.weight[0],
                nav[:, None] + self.modality.weight[1],
                od + self.modality.weight[2],
                ld + self.modality.weight[3],
            ],
            dim=1,
        )
        valid = torch.cat([ego_ok, nav_ok, od_valid > 0.5, ld_valid > 0.5], dim=1)
        # torch's TransformerEncoder can produce NaN if every key is masked.
        # Keep the ego anchor valid as a numerical fallback; its output is
        # ignored by the planner when all entity valid masks are false.
        if not bool(valid.any(dim=1).all()):
            valid = valid.clone()
            empty = ~valid.any(dim=1)
            valid[empty, 0] = True
        out = self.spatial(tokens, src_key_padding_mask=~valid)
        out = self.norm(out)
        od_out = out[:, 2 : 2 + self.od_slots]
        ld_out = out[:, 2 + self.od_slots : 2 + self.od_slots + self.ld_slots]
        return {"od": od_out, "ld": ld_out, "tokens": out}


class ODGeometryDecoder(nn.Module):
    """Predict an OD relative-pose residual in the query ego frame."""

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
    """Predict a small LD polyline residual on top of explicit projection."""

    def __init__(self, hidden: int = 128, action_dim: int = 4, points: int = 5) -> None:
        super().__init__()
        self.points = int(points)
        self.net = nn.Sequential(
            nn.Linear(hidden + action_dim + 2 * self.points, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * self.points),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, content: Tensor, action: Tensor, base_ld: Tensor) -> Tensor:
        b, n, _ = content.shape
        a = action[:, None, :].expand(b, n, -1)
        base = base_ld.reshape(b, n, 2 * self.points)
        return self.net(torch.cat([content, a, base], dim=-1))


def project_ld(geometry: Tensor, source_pose: Tensor, target_pose: Tensor) -> Tensor:
    """Project LD polyline geometry into ``target_pose`` ego coordinates.

    Accepts a single ``[10]`` geometry/``[3]`` pose or batched leading
    dimensions such as ``[B,16,10]`` with poses ``[B,3]``.
    """
    if geometry.dim() == 1:
        return BankWindowBuilder._transform_geometry(
            geometry, source_pose, target_pose, node_type=NodeType.LD
        )
    if geometry.dim() == 2:
        if source_pose.dim() == 1 and target_pose.dim() == 1:
            return torch.stack(
                [
                    BankWindowBuilder._transform_geometry(
                        geometry[i], source_pose, target_pose, node_type=NodeType.LD
                    )
                    for i in range(geometry.shape[0])
                ],
                dim=0,
            )
        if source_pose.dim() == 2 and target_pose.dim() == 2:
            return torch.stack(
                [
                    BankWindowBuilder._transform_geometry(
                        geometry[i], source_pose[i], target_pose[i], node_type=NodeType.LD
                    )
                    for i in range(geometry.shape[0])
                ],
                dim=0,
            )
        raise ValueError("batched LD projection needs [3] or [B,3] poses")
    if geometry.dim() == 3:
        if source_pose.dim() == 2 and target_pose.dim() == 2:
            # [B,N,10] geometry with [B,3] poses.
            return torch.stack(
                [
                    project_ld(geometry[i], source_pose[i], target_pose[i])
                    for i in range(geometry.shape[0])
                ],
                dim=0,
            )
        raise ValueError("LD projection pose dimensions do not match geometry")
    if geometry.dim() == 4:
        if source_pose.dim() != 3 or target_pose.dim() != 3:
            raise ValueError("temporal LD projection needs [B,T,3] poses")
        return torch.stack(
            [
                project_ld(geometry[i], source_pose[i], target_pose[i])
                for i in range(geometry.shape[0])
            ],
            dim=0,
        )
    raise ValueError(f"unsupported LD geometry shape {tuple(geometry.shape)}")


def add_ld_residual(projected: Tensor, residual: Tensor) -> Tensor:
    """Add a ``[...,5,2]`` residual to flattened ``[...,10]`` LD geometry."""
    if projected.shape[-1] != 10:
        raise ValueError("projected LD geometry must be flattened [...,10]")
    if residual.shape[-1] == 10:
        return projected + residual
    if residual.shape[-2:] == (5, 2):
        return projected + residual.reshape(*residual.shape[:-2], -1)
    raise ValueError("LD residual must be [...,5,2] or [...,10]")


class FactorizedEgoConditionedLatentWM(nn.Module):
    """OD content transition + OD pose decoder + LD residual decoder."""

    def __init__(self, hidden: int = 128, action_dim: int = 4, points: int = 5) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.action = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.od_delta = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        nn.init.zeros_(self.od_delta[-1].weight)
        nn.init.zeros_(self.od_delta[-1].bias)
        self.od_geometry = ODGeometryDecoder(hidden)
        self.ld_residual = LDResidualDecoder(hidden, action_dim=action_dim, points=points)

    def forward(
        self,
        od: Tensor,
        ld: Tensor,
        action: Tensor,
        *,
        od_geometry: Tensor | None = None,
        ld_geometry: Tensor | None = None,
        od_valid: Tensor | None = None,
        ld_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        b, n, _ = od.shape
        a = self.action(action)[:, None, :].expand(b, n, -1)
        od_mask = torch.ones((b, n, 1), dtype=od.dtype, device=od.device) if od_valid is None else od_valid.unsqueeze(-1).to(od.dtype)
        ld_mask = torch.ones((b, n, 1), dtype=ld.dtype, device=ld.device) if ld_valid is None else ld_valid.unsqueeze(-1).to(ld.dtype)
        od_next = od + self.od_delta(torch.cat([od, a], dim=-1)) * od_mask
        if ld_geometry is None:
            raise ValueError("LD residual requires base ld_geometry")
        base_ld = ld_geometry
        ld_res = self.ld_residual(ld, action, base_ld) * ld_mask
        ld_next = base_ld + ld_res
        if od_geometry is None:
            od_next_geometry = self.od_geometry(od_next)
        else:
            od_next_geometry = od_geometry.clone()
            od_next_geometry[..., :3] = od_geometry[..., :3] + self.od_geometry(od_next) * od_mask
        od_next_geometry[..., 2] = torch.atan2(
            torch.sin(od_next_geometry[..., 2]),
            torch.cos(od_next_geometry[..., 2]),
        )
        return {
            "od": od_next,
            "ld": ld,
            "od_geometry": od_next_geometry,
            "ld_residual": ld_res,
            "ld_geometry": ld_next,
            "od_valid": od_valid,
            "ld_valid": ld_valid,
        }

    def rollout(
        self,
        od: Tensor,
        ld: Tensor,
        actions: Tensor,
        *,
        od_geometry: Tensor | None = None,
        ld_geometry: Tensor | None = None,
        od_valid: Tensor | None = None,
        ld_valid: Tensor | None = None,
        detach: bool = True,
    ) -> Dict[str, Tensor]:
        """Multi-step tensor rollout; no environment and no no-grad wrapper."""
        state = {"od": od, "ld": ld, "od_geometry": od_geometry, "ld_geometry": ld_geometry}
        history = []
        for t in range(int(actions.shape[1])):
            state = self(
                state["od"], state["ld"], actions[:, t],
                od_geometry=state["od_geometry"], ld_geometry=state["ld_geometry"],
                od_valid=od_valid, ld_valid=ld_valid,
            )
            if detach:
                state = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in state.items()}
            history.append(state)
        return {"final": state, "history": history}


# The old :class:`EgoConditionedLatentWM` above is retained for the legacy
# D1/D2 pipeline.  New code should use the factorized class.
EgoConditionedLatentWMV2 = FactorizedEgoConditionedLatentWM
