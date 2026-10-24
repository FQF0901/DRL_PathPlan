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

from net.mem_bank import BankWindow, BankWindowBuilder, ImagineMemoryBank, NodeType


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
    """Predict the absolute OD relative pose in the query ego frame."""

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
    """Project LD polyline geometry into ``target_pose`` ego coordinates."""
    return BankWindowBuilder.project_geometry(
        geometry, source_pose, target_pose, NodeType.LD
    )


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
    ) -> Dict[str, Tensor | None]:
        b, n, _ = od.shape
        a = self.action(action)[:, None, :].expand(b, n, -1)
        od_mask = torch.ones((b, n, 1), dtype=od.dtype, device=od.device) if od_valid is None else od_valid.unsqueeze(-1).to(od.dtype)
        ld_mask = torch.ones((b, n, 1), dtype=ld.dtype, device=ld.device) if ld_valid is None else ld_valid.unsqueeze(-1).to(ld.dtype)
        od_next = od + self.od_delta(torch.cat([od, a], dim=-1)) * od_mask
        if ld_geometry is None:
            ld_res = None
            ld_next = None
        else:
            base_ld = ld_geometry
            ld_res = self.ld_residual(ld, action, base_ld) * ld_mask
            ld_next = base_ld + ld_res

        # OD decoder is absolute relative pose, not a residual.
        pose = self.od_geometry(od_next) * od_mask
        yaw = torch.atan2(torch.sin(pose[..., 2]), torch.cos(pose[..., 2]))
        if od_geometry is not None and od_geometry.shape[-1] >= 5:
            od_next_geometry = torch.cat(
                [pose[..., :2], yaw.unsqueeze(-1), od_geometry[..., 3:]], dim=-1
            )
        else:
            od_next_geometry = torch.cat([pose[..., :2], yaw.unsqueeze(-1)], dim=-1)
        return {
            "od": od_next,
            "ld": ld,
            "od_geometry": od_next_geometry,
            "ld_residual": ld_res,
            "ld_geometry": ld_next,
            "od_valid": od_valid,
            "ld_valid": ld_valid,
        }



# The old :class:`EgoConditionedLatentWM` above is retained for the legacy
# D1/D2 pipeline.  New code should use the factorized class.
EgoConditionedLatentWMV2 = FactorizedEgoConditionedLatentWM



def _sequence_step(x: Tensor | None, n: int, name: str, *, horizon: int, batch: int, hidden: int | None = None) -> Tensor | None:
    if x is None:
        return None
    if name.endswith("valid_seq") and x.dim() == 2:
        if x.shape[0] != batch or x.shape[1] < horizon:
            raise ValueError(f"{name} must be [B,H>=horizon] or [B,H>=horizon,1], got {tuple(x.shape)}")
        return x[:, n, None]
    if x.dim() != 3 or x.shape[0] != batch or x.shape[1] < horizon:
        raise ValueError(f"{name} must be [B,H,D] with H>={horizon}, got {tuple(x.shape)}")
    if hidden is not None and x.shape[2] != hidden:
        raise ValueError(f"{name} last dim must be hidden={hidden}, got {x.shape[2]}")
    return x[:, n]


def imagine_rollout(
    window: BankWindow,
    nav: Tensor,
    ego: Tensor,
    *,
    temporal_spatial: TemporalSpatialBlock,
    planner,
    world_model: FactorizedEgoConditionedLatentWM,
    actions: Tensor | None = None,
    action_provider=None,
    horizon: int = 6,
    ego_pose: Tensor | None = None,
    nav_valid: Tensor | None = None,
    ego_valid: Tensor | None = None,
    nav_seq: Tensor | None = None,
    ego_seq: Tensor | None = None,
    nav_valid_seq: Tensor | None = None,
    ego_valid_seq: Tensor | None = None,
    imagine_bank: ImagineMemoryBank | None = None,
    detach: bool = True,
) -> list[Dict[str, Tensor]]:
    """Model-level imagined rollout.

    This is the single rollout implementation.  ``DriveModel.imagine`` delegates
    here.  ``nav_seq``/``ego_seq`` are encoded hidden vectors ``[B,H,hidden]``;
    raw observations are not accepted.
    """
    if not 1 <= int(horizon) <= 6:
        raise ValueError("imagine horizon must be in 1..6")
    b = int(window.batch_size)
    if imagine_bank is not None and b != 1:
        raise ValueError("ImagineMemoryBank rollout supports B=1 only")
    hidden = int(nav.shape[-1])
    if ego.shape[-1] != hidden:
        raise ValueError("nav/ego hidden size mismatch")
    if actions is not None:
        if actions.dim() != 3 or actions.shape[0] != b or actions.shape[-1] != 4:
            raise ValueError("actions must be [B,H,4]")
        if actions.shape[1] < horizon:
            raise ValueError("actions sequence is shorter than horizon")
    if ego_pose is not None:
        if ego_pose.dim() != 3 or ego_pose.shape[0] != b or ego_pose.shape[-1] != 3:
            raise ValueError("ego_pose must be [B,H,3]")
        if ego_pose.shape[1] < horizon:
            raise ValueError("ego_pose sequence is shorter than horizon")

    current_window = imagine_bank.window() if imagine_bank is not None else window
    outputs: list[Dict[str, Tensor]] = []
    for n in range(int(horizon)):
        nav_n = _sequence_step(nav_seq, n, "nav_seq", horizon=horizon, batch=b, hidden=hidden)
        ego_n = _sequence_step(ego_seq, n, "ego_seq", horizon=horizon, batch=b, hidden=hidden)
        if nav_n is None:
            nav_n = nav
        if ego_n is None:
            ego_n = ego
        nav_v = nav_valid if nav_valid_seq is None else _sequence_step(nav_valid_seq, n, "nav_valid_seq", horizon=horizon, batch=b)
        ego_v = ego_valid if ego_valid_seq is None else _sequence_step(ego_valid_seq, n, "ego_valid_seq", horizon=horizon, batch=b)
        z = temporal_spatial(current_window, nav_n, ego_n, nav_valid=nav_v, ego_valid=ego_v)
        od_valid = current_window.od_valid[:, -1]
        ld_valid = current_window.ld_valid[:, -1]
        policy = planner(z["od"], z["ld"], nav_n, ego_n, od_valid, ld_valid)
        outputs.append(policy)
        if n + 1 >= int(horizon):
            break
        if actions is not None:
            action = actions[:, n]
        elif action_provider is not None:
            action = action_provider(policy, ego_n, n)
        else:
            action = policy["action_mu"]
        if action.shape[-1] != 4:
            raise ValueError("action must have 4 dims")

        source_pose = current_window.ego_pose[:, -1]
        if ego_pose is not None:
            next_idx = min(n + 1, int(ego_pose.shape[1]) - 1)
            next_pose = ego_pose[:, next_idx]
        else:
            next_pose = source_pose
        base_od = BankWindowBuilder.project_geometry(
            current_window.od_geometry[:, -1], source_pose, next_pose, NodeType.OD
        )
        base_ld = BankWindowBuilder.project_geometry(
            current_window.ld_geometry[:, -1], source_pose, next_pose, NodeType.LD
        )
        step = world_model(
            z["od"], z["ld"], action,
            od_geometry=base_od, ld_geometry=base_ld,
            od_valid=od_valid, ld_valid=ld_valid,
        )
        if detach:
            step = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in step.items()}
        depth = int(current_window.rollout_depth[0, -1].item()) + 1
        next_window = current_window.advance(
            od_content=step["od"],
            od_geometry=step["od_geometry"],
            od_type=current_window.od_type[:, -1],
            od_valid=step["od_valid"] if step["od_valid"] is not None else od_valid,
            od_stable_id=current_window.od_stable_id[:, -1],
            ld_content=step["ld"],
            ld_geometry=step["ld_geometry"],
            ld_type=current_window.ld_type[:, -1],
            ld_valid=step["ld_valid"] if step["ld_valid"] is not None else ld_valid,
            ld_stable_id=current_window.ld_stable_id[:, -1],
            ego_pose=next_pose,
            source=1,
            rollout_depth=depth,
            source_pose=source_pose,
            target_pose=next_pose,
            project_new_geometry=False,
        )
        if imagine_bank is not None:
            prev = imagine_bank.frames[-1]
            record = next_window.to_frame_record(
                episode_id=prev.episode_id,
                frame_id=prev.frame_id + 1,
                timestamp=prev.timestamp + 0.5,
                parent_frame_id=prev.frame_id,
                action_from_parent=action,
            )
            imagine_bank.append(record, detach=detach)
            current_window = imagine_bank.window()
        else:
            current_window = next_window
    return outputs


# The contract calls this the model-level rollout.
rollout = imagine_rollout
