"""Bank-native temporal-spatial encoder, latent WM, and imagined rollout.

Contract highlights (docs/arch_lock.md):
* ``z`` is a frame-dependent full-state latent; geometry is stored separately.
* TSB temporal attention uses the same stable ID plus a frame-level
  ``delta_pose(t -> query)``.  It does not project entity geometry.
* TSB spatial attention uses current OD/LD plus an ego alignment token; nav is
  policy-only.
* The WM predicts a latent residual ``r_z``; geometry decoders are direct and
  return absolute geometry in the current ego frame.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import torch
from torch import Tensor, nn

from net.mem_bank import BankWindow, ImagineMemoryBank


def _wrap_angle(x: Tensor) -> Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def frame_delta(ego_pose: Tensor) -> Tensor:
    """Return ``delta_pose(t -> query)`` for a [B,T,3] pose sequence.

    The result is the query pose expressed in each historical ego frame:
    ``(dx, dy, dyaw)``.  The last frame is zero; padded/invalid poses simply
    produce a finite delta and are masked by the caller.
    """
    if ego_pose.dim() != 3 or ego_pose.shape[-1] != 3:
        raise ValueError("ego_pose must be [B,T,3]")
    q = ego_pose[:, -1:]
    p = ego_pose
    c = torch.cos(p[..., 2])
    s = torch.sin(p[..., 2])
    dx_w = q[..., 0] - p[..., 0]
    dy_w = q[..., 1] - p[..., 1]
    dx = c * dx_w + s * dy_w
    dy = -s * dx_w + c * dy_w
    dyaw = _wrap_angle(q[..., 2] - p[..., 2])
    return torch.stack([dx, dy, dyaw], dim=-1)


class TemporalAttention(nn.Module):
    """Masked temporal attention over current entity + same-ID history."""

    def __init__(self, hidden: int, heads: int, frames: int = 6) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.frames = int(frames)
        self.attn = nn.MultiheadAttention(self.hidden, int(heads), batch_first=True)
        self.norm = nn.LayerNorm(self.hidden)

    def forward(
        self,
        tokens: Tensor,
        valid: Tensor,
        stable_id: Tensor,
        frame_valid: Tensor | None = None,
    ) -> Tensor:
        b, t, n, h = tokens.shape
        if t > self.frames:
            raise ValueError(f"window has {t} frames, max {self.frames}")
        if frame_valid is not None:
            if frame_valid.shape != (b, t):
                raise ValueError("frame_valid must be [B,T]")
            valid = valid & frame_valid[:, :, None]
        x = tokens.permute(0, 2, 1, 3).reshape(b * n, t, h)
        v = valid.permute(0, 2, 1).reshape(b * n, t)
        ids = stable_id.permute(0, 2, 1).reshape(b * n, t)
        same = ids == ids[:, -1:]
        mask = ~((v > 0.5) & same)
        all_bad = mask.all(dim=1)
        if bool(all_bad.any()):
            mask = mask.clone()
            mask[all_bad, -1] = False
        # Invalid slots may contain NaN; zero them before the matmul because
        # key_padding_mask only removes the weight, not the NaN value.
        x = torch.where(v[..., None] > 0.5, x, torch.zeros_like(x))
        q = x[:, -1:, :]
        out = self.attn(q, x, x, key_padding_mask=mask, need_weights=False)[0][:, 0]
        out = self.norm(out + q[:, 0])
        keep = ((v[:, -1] > 0.5) & (~all_bad)).unsqueeze(-1)
        out = torch.where(keep, out, torch.zeros_like(out))
        return out.reshape(b, n, h)


class TemporalSpatialBlock(nn.Module):
    """Factorized temporal + current-frame spatial fusion.

    Temporal tokens carry a frame-level ego delta; spatial tokens carry current
    OD/LD and one ego alignment token.  Nav is intentionally absent.
    """

    def __init__(
        self,
        hidden: int = 128,
        heads: int = 4,
        spatial_layers: int = 2,
        od_slots: int = 16,
        ld_slots: int = 16,
        frames: int = 6,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.temporal_od = TemporalAttention(hidden, heads, frames=frames)
        self.temporal_ld = TemporalAttention(hidden, heads, frames=frames)
        self.frame_embed = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.modality = nn.Embedding(3, hidden)  # ego, od, ld
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
        ego: Tensor,
        *,
        ego_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if ego.shape[-1] != self.hidden:
            raise ValueError("ego must have hidden size")
        if window.od_content.shape[2] != self.od_slots or window.ld_content.shape[2] != self.ld_slots:
            raise ValueError("BankWindow slot count does not match TemporalSpatialBlock")

        # z is frame-dependent; add only a frame-level ego delta.  No entity
        # geometry is projected or added here.
        delta = frame_delta(window.ego_pose)
        frame_token = self.frame_embed(delta)
        od_tokens = window.od_content + frame_token[:, :, None, :]
        ld_tokens = window.ld_content + frame_token[:, :, None, :]
        od = self.temporal_od(od_tokens, window.od_valid, window.od_stable_id, window.frame_valid)
        ld = self.temporal_ld(ld_tokens, window.ld_valid, window.ld_stable_id, window.frame_valid)

        b = od.shape[0]
        od_valid = window.od_valid[:, -1]
        ld_valid = window.ld_valid[:, -1]
        ego_ok = (
            torch.ones((b, 1), dtype=torch.bool, device=od.device)
            if ego_valid is None
            else ego_valid.reshape(b, 1) > 0.5
        )
        tokens = torch.cat(
            [
                ego[:, None] + self.modality.weight[0],
                od + self.modality.weight[1],
                ld + self.modality.weight[2],
            ],
            dim=1,
        )
        valid = torch.cat([ego_ok, od_valid > 0.5, ld_valid > 0.5], dim=1)
        if not bool(valid.any(dim=1).all()):
            valid = valid.clone()
            valid[~valid.any(dim=1), 0] = True
        tokens = torch.where(valid[..., None], tokens, torch.zeros_like(tokens))
        out = self.spatial(tokens, src_key_padding_mask=~valid)
        out = self.norm(out)
        out = torch.where(valid[..., None], out, torch.zeros_like(out))
        od_out = out[:, 1 : 1 + self.od_slots]
        ld_out = out[:, 1 + self.od_slots : 1 + self.od_slots + self.ld_slots]
        return {"od": od_out, "ld": ld_out, "tokens": out}


class ODGeometryDecoder(nn.Module):
    """Decode absolute OD geometry ``[x,y,yaw,vx,vy]`` from state latent."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 5)
        )

    def forward(self, content: Tensor) -> Tensor:
        return self.net(content)


class LDGeometryDecoder(nn.Module):
    """Decode absolute LD polyline ``[x0,y0,...,x4,y4]`` from state latent."""

    def __init__(self, hidden: int = 128, points: int = 5) -> None:
        super().__init__()
        self.points = int(points)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2 * self.points)
        )

    def forward(self, content: Tensor) -> Tensor:
        return self.net(content)


class EgoConditionedLatentWM(nn.Module):
    """Lightweight latent transition plus direct OD/LD geometry decoders."""

    def __init__(self, hidden: int = 128, action_dim: int = 4, points: int = 5) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.action = nn.Sequential(
            nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.od_delta = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.ld_delta = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        nn.init.zeros_(self.od_delta[-1].weight)
        nn.init.zeros_(self.od_delta[-1].bias)
        nn.init.zeros_(self.ld_delta[-1].weight)
        nn.init.zeros_(self.ld_delta[-1].bias)
        self.od_geometry = ODGeometryDecoder(hidden)
        self.ld_geometry = LDGeometryDecoder(hidden, points=points)
        # Slot-storage cue for currently invalid slots.  Target-only/new
        # entities have no content; without this cue every free slot would
        # decode from the identical zero vector.  Valid slots are unaffected.
        self.od_slot_embed = nn.Parameter(torch.zeros(16, hidden))
        self.ld_slot_embed = nn.Parameter(torch.zeros(16, hidden))
        nn.init.normal_(self.od_slot_embed, std=0.02)
        nn.init.normal_(self.ld_slot_embed, std=0.02)

    def forward(
        self,
        od_context: Tensor,
        ld_context: Tensor,
        action: Tensor,
        *,
        od_base: Tensor,
        ld_base: Tensor,
        od_valid: Tensor | None = None,
        ld_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        """Predict next-state residuals from TSB history context.

        ``od_context/ld_context`` are the per-entity TSB outputs ``h_t``.
        ``od_base/ld_base`` are the required raw Bank state ``z_hat_t`` to which
        the residuals are added.  There is no direct-context fallback.
        """
        b, n, _ = od_context.shape
        if od_base.shape != od_context.shape or ld_base.shape != ld_context.shape:
            raise ValueError("WM context and Bank base shapes must match")
        if not torch.isfinite(action).all():
            raise ValueError("action must be finite")
        od_bool = (
            torch.ones((b, n), dtype=torch.bool, device=od_context.device)
            if od_valid is None
            else od_valid > 0.5
        )
        ld_bool = (
            torch.ones((b, n), dtype=torch.bool, device=ld_context.device)
            if ld_valid is None
            else ld_valid > 0.5
        )
        # The encoder may emit NaN for invalid slots; sanitize before the
        # MLP because 0 * NaN would poison every entity in the batch.
        od_context = torch.where(od_bool[..., None], od_context, torch.zeros_like(od_context))
        ld_context = torch.where(ld_bool[..., None], ld_context, torch.zeros_like(ld_context))
        od_base = torch.where(od_bool[..., None], od_base, torch.zeros_like(od_base))
        ld_base = torch.where(ld_bool[..., None], ld_base, torch.zeros_like(ld_base))
        a = self.action(action)[:, None, :].expand(b, n, -1)
        od_mask = od_bool.unsqueeze(-1).to(od_context.dtype)
        ld_mask = ld_bool.unsqueeze(-1).to(ld_context.dtype)
        od_res = self.od_delta(torch.cat([od_context, a], dim=-1)) * od_mask
        ld_res = self.ld_delta(torch.cat([ld_context, a], dim=-1)) * ld_mask
        od_next = od_base + od_res
        ld_next = ld_base + ld_res
        return {
            "od": od_next,
            "ld": ld_next,
            "od_residual": od_res,
            "ld_residual": ld_res,
            # Geometry is decoded for every slot; validity is applied by the
            # loss mask.  Invalid slots receive a slot-storage cue so
            # target-only/new entities are distinguishable from each other.
            "od_geometry": self.od_geometry(
                od_next + (~od_bool)[..., None] * self.od_slot_embed[:n]
            ),
            "ld_geometry": self.ld_geometry(
                ld_next + (~ld_bool)[..., None] * self.ld_slot_embed[:n]
            ),
            "od_valid": od_valid,
            "ld_valid": ld_valid,
        }


def compose_pose(pose: Tensor, action: Tensor) -> Tensor:
    """SE(2) endpoint from current pose and physical ``(dx,dy,dpsi,...)``."""
    if pose.shape[-1] != 3 or action.shape[-1] < 3:
        raise ValueError("pose must be [...,3] and action [...,>=3]")
    c = torch.cos(pose[..., 2])
    s = torch.sin(pose[..., 2])
    x = pose[..., 0] + c * action[..., 0] - s * action[..., 1]
    y = pose[..., 1] + s * action[..., 0] + c * action[..., 1]
    yaw = _wrap_angle(pose[..., 2] + action[..., 2])
    return torch.stack([x, y, yaw], dim=-1)


def imagine_rollout(
    window: BankWindow,
    nav: Tensor,
    ego: Tensor,
    *,
    temporal_spatial: TemporalSpatialBlock,
    planner,
    world_model: EgoConditionedLatentWM,
    action_to_requested: Callable[[Tensor], Tensor],
    ego_features: Tensor,
    ego_step: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
    ego_encoder: Callable[[Tensor], Tensor],
    actions: Tensor | None = None,
    action_provider: Callable[[Dict[str, Tensor], Tensor, int], Tensor] | None = None,
    horizon: int = 6,
    ego_valid: Tensor | None = None,
    imagine_bank: ImagineMemoryBank | None = None,
    detach: bool = True,
) -> List[Dict[str, Tensor]]:
    """Full model-level 6-transition rollout.

    ``policy_0..policy_5`` are produced at t0..t5.  Each policy supplies one
    action; action ``a_t`` drives the transition to ``t+1``.  The final
    transition produces terminal state t6.  ``t6`` has no policy.
    """
    if not 1 <= int(horizon) <= 6:
        raise ValueError("imagine horizon must be in 1..6")
    b = int(window.batch_size)
    device = ego.device
    window = window.to(device)
    if imagine_bank is not None:
        if b != 1:
            raise ValueError("ImagineMemoryBank rollout supports B=1 only")
        bank_window = imagine_bank.window().to(device)
        all_fields = (
            "od_content", "ld_content", "od_geometry", "ld_geometry", "ego_pose",
            "od_type", "ld_type", "od_stable_id", "ld_stable_id", "od_valid",
            "ld_valid", "rollout_depth", "source", "frame_valid",
        )
        for field in all_fields:
            if getattr(window, field).shape != getattr(bank_window, field).shape:
                raise ValueError(f"window/{field} shape does not match imagine_bank")
        for field in all_fields:
            a, bank_value = getattr(window, field), getattr(bank_window, field)
            if a.is_floating_point():
                if not torch.allclose(a, bank_value):
                    raise ValueError(f"window/{field} does not match imagine_bank")
            elif not torch.equal(a, bank_value):
                raise ValueError(f"window/{field} does not match imagine_bank")
    hidden = int(nav.shape[-1])
    if ego.shape[-1] != hidden:
        raise ValueError("nav/ego hidden size mismatch")
    if actions is not None:
        if actions.dim() != 3 or actions.shape[0] != b or actions.shape[-1] != 4:
            raise ValueError("actions must be [B,H,4]")
        if actions.shape[1] < horizon:
            raise ValueError("actions sequence is shorter than horizon")
        if (
            not torch.isfinite(actions).all()
            or bool(((actions < 0.0) | (actions > 1.0)).any())
        ):
            raise ValueError("actions must be finite normalized policy actions in [0,1]")
    if ego_features.shape[0] != b:
        raise ValueError("ego_features batch mismatch")

    current_window = imagine_bank.window().to(device) if imagine_bank is not None else window
    current_ego = ego.detach() if detach else ego
    current_ego_features = ego_features.detach() if (detach and ego_features is not None) else ego_features
    if detach:
        nav = nav.detach()
    outputs: List[Dict[str, Tensor]] = []
    for n in range(int(horizon)):
        z = temporal_spatial(current_window, current_ego, ego_valid=ego_valid)
        od_valid = current_window.od_valid[:, -1]
        ld_valid = current_window.ld_valid[:, -1]
        policy = planner(z["od"], z["ld"], nav, current_ego, od_valid, ld_valid)
        outputs.append(policy)

        if actions is not None:
            action = actions[:, n]
        elif action_provider is not None:
            action = action_provider(policy, current_ego, n)
        else:
            action = policy["action_mu"]
        if action.shape[-1] != 4:
            raise ValueError("action must have 4 dims")
        if (
            not torch.isfinite(action).all()
            or bool(((action < 0.0) | (action > 1.0)).any())
        ):
            raise ValueError("actions must be finite normalized policy actions in [0,1]")
        if detach:
            action = action.detach()

        requested = action_to_requested(action)
        if detach:
            requested = requested.detach()
        source_pose = current_window.ego_pose[:, -1]
        next_pose, next_ego_features = ego_step(current_ego_features, requested, source_pose)
        if detach:
            next_pose = next_pose.detach()
            if next_ego_features is not None:
                next_ego_features = next_ego_features.detach()
        # TSB context h_t is shared by policy and WM.  WM predicts the
        # residual r_z; the residual base is the raw Bank state z_hat_t.
        wm_context_od = z["od"]
        wm_context_ld = z["ld"]
        wm_base_od = current_window.od_content[:, -1]
        wm_base_ld = current_window.ld_content[:, -1]
        if detach:
            wm_context_od = wm_context_od.detach()
            wm_context_ld = wm_context_ld.detach()
            wm_base_od = wm_base_od.detach()
            wm_base_ld = wm_base_ld.detach()
        step = world_model(
            wm_context_od,
            wm_context_ld,
            action,
            od_base=wm_base_od,
            ld_base=wm_base_ld,
            od_valid=od_valid,
            ld_valid=ld_valid,
        )
        if detach:
            step = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in step.items()}
        depth = current_window.rollout_depth[:, -1] + 1
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
        )
        if imagine_bank is not None:
            prev = imagine_bank.frames[-1]
            record = next_window.to_frame_record(
                episode_id=prev.episode_id,
                frame_id=prev.frame_id + 1,
                timestamp=prev.timestamp + 0.5,
                parent_frame_id=prev.frame_id,
                action_from_parent=requested,
            )
            imagine_bank.append(record, detach=detach)
            current_window = imagine_bank.window().to(device)
        else:
            current_window = next_window

        current_ego = ego_encoder(next_ego_features)
        if detach:
            current_ego = current_ego.detach()
        current_ego_features = next_ego_features
    return outputs
