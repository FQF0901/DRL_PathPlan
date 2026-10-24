"""Current-frame OD/LD/Nav/Ego encoder and a frozen/EMA target copy.

This module is the upstream of ``RealMemoryBank``.  It only encodes the
current frame; no history stacking and no temporal fusion live here.

Geometry layouts:
* OD: ``[x, y, yaw, vx, vy]``
* LD: flattened five ``(x, y)`` lane points
"""

from __future__ import annotations

import copy
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn

from env.obs import (
    EGO_DIM,
    LD_DIM,
    LD_POINTS,
    LD_SLOTS,
    NAV_DIM,
    OD_DIM,
    OD_SLOTS,
)
from net.mem_bank import EntityRecord, FrameRecord, NodeType, Source

LANE_ROLE_CLASSES = 4


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


def _take_last(x: Tensor, current_ndim: int) -> Tensor:
    """Take the last history frame when the tensor has one extra axis."""
    if x.dim() == current_ndim + 1:
        return x[:, -1]
    if x.dim() != current_ndim:
        raise ValueError(f"expected {current_ndim}D current or {current_ndim + 1}D history tensor, got {tuple(x.shape)}")
    return x


def _geometry_od(od: Tensor, anchor: Optional[Tensor] = None) -> Tensor:
    """Return [B,N,5] explicit OD geometry in the current ego frame."""
    if anchor is not None:
        x = anchor[..., 0]
        y = anchor[..., 1]
        yaw = anchor[..., 2]
    else:
        x = od[..., 0]
        y = od[..., 1]
        yaw = torch.atan2(od[..., 3], od[..., 2])
    vx = od[..., 4] if od.shape[-1] > 5 else torch.zeros_like(x)
    vy = od[..., 5] if od.shape[-1] > 5 else torch.zeros_like(x)
    return torch.stack((x, y, yaw, vx, vy), dim=-1)


def _geometry_ld(ld: Tensor) -> Tensor:
    """Return [B,N,10] flattened lane points in the current ego frame."""
    if ld.dim() != 4:
        raise ValueError(f"ld must be [B,N,P,D], got {tuple(ld.shape)}")
    if ld.shape[2] != LD_POINTS:
        raise ValueError(f"ld must have {LD_POINTS} points, got {ld.shape[2]}")
    return ld[..., :, :2].reshape(ld.shape[0], ld.shape[1], LD_POINTS * 2)


def _ids(batch: Mapping[str, Tensor], key: str, b: int, n: int, device: torch.device) -> Tensor:
    if key not in batch:
        raise ValueError(f"encoder requires {key!r}; stable IDs must not default to -1")
    value = _take_last(batch[key], 2)
    return value.reshape(b, n)


def _mask(batch: Mapping[str, Tensor], key: str, b: int, n: int, device: torch.device, *, ld: bool = False) -> Tensor:
    if key not in batch:
        raise ValueError(f"encoder requires {key!r}; masks must not default to all-valid")
    value = _take_last(batch[key], 3 if ld else 2)
    if ld:
        return value.reshape(b, n, -1).any(dim=-1)
    return value.reshape(b, n) > 0.5


def _type(batch: Mapping[str, Tensor], key: str, od: Tensor, ld: Tensor, b: int, n: int, device: torch.device, *, ld_node: bool = False) -> Tensor:
    taxonomy = batch.get(key)
    if taxonomy is not None:
        value = _take_last(taxonomy, 3)
        return value.reshape(b, n, -1)[..., 0].long()
    if ld_node:
        # env.obs has no ld_taxonomy; use the normalized road type, not lane_role.
        return (ld[:, :, 0, 13] * 15.0).round().long() if ld.shape[-1] > 14 else torch.zeros((b, n), dtype=torch.long, device=device)
    return od[..., 8].round().long() if od.shape[-1] > 8 else torch.zeros((b, n), dtype=torch.long, device=device)


class CurrentFrameEncoder(nn.Module):
    """Encode one current MetaDrive frame into OD/LD content + geometry."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.od_encoder = _mlp(OD_DIM, self.hidden, self.hidden)
        self.od_type_embed = nn.Embedding(8, self.hidden)
        self.od_static_embed = nn.Linear(1, self.hidden)

        self.ld_point = _mlp(LD_DIM, max(1, self.hidden // 2), max(1, self.hidden // 2))
        self.ld_lane = nn.Sequential(
            nn.Linear(max(1, self.hidden // 2), self.hidden),
            nn.GELU(),
            nn.LayerNorm(self.hidden),
        )
        self.ld_role_embed = nn.Embedding(LANE_ROLE_CLASSES, self.hidden)

        self.nav_encoder = _mlp(NAV_DIM, self.hidden, self.hidden)
        self.ego_encoder = _mlp(EGO_DIM, self.hidden, self.hidden)

    def _encode_od(self, od: Tensor) -> Tensor:
        h = self.od_encoder(od)
        if od.shape[-1] > 8:
            h = h + self.od_type_embed(od[..., 8].long().clamp(0, 7))
        if od.shape[-1] > 9:
            h = h + self.od_static_embed(od[..., 9:10])
        return h

    def _encode_ld(self, ld: Tensor, ld_mask: Tensor) -> Tensor:
        b, n, p, d = ld.shape
        points = self.ld_point(ld.reshape(b * n * p, d)).reshape(b, n, p, -1)
        mask = ld_mask.reshape(b, n, p).to(points.dtype).unsqueeze(-1)
        lane = (points * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        h = self.ld_lane(lane)
        if ld.shape[-1] > 7:
            h = h + self.ld_role_embed(ld[:, :, 0, 7].long().clamp(0, LANE_ROLE_CLASSES - 1))
        return h

    def encode_current(
        self,
        batch: Mapping[str, Tensor] | None = None,
        **kwargs: Tensor,
    ) -> Dict[str, Tensor]:
        """Encode the current frame.

        Accepts either a current-frame batch (``[B,...]``) or a legacy history
        batch (``[B,T,...]``); only the last frame is used.  The output is
        batch-first and never contains a temporal dimension.
        """
        if batch is None:
            batch = kwargs
        od = _take_last(batch["od"], 3).float()
        ld = _take_last(batch["ld"], 4).float()
        nav = _take_last(batch["nav"], 2).float()
        ego = _take_last(batch["ego"], 2).float()
        b = od.shape[0]
        device = od.device

        od_mask = _mask(batch, "od_mask", b, OD_SLOTS, device)
        ld_mask = _mask(batch, "ld_mask", b, LD_SLOTS, device, ld=True)
        od_anchor = batch.get("od_anchor")
        od_anchor = _take_last(od_anchor, 3) if od_anchor is not None else None
        if od_anchor is not None:
            od_anchor = od_anchor.reshape(b, OD_SLOTS, 3)
        od_geometry = _geometry_od(od, od_anchor)
        ld_geometry = _geometry_ld(ld)

        od_content = self._encode_od(od)
        ld_mask_points = _take_last(batch["ld_mask"], 3).reshape(b, LD_SLOTS, LD_POINTS)
        ld_content = self._encode_ld(ld, ld_mask_points)
        od_type = _type(batch, "od_taxonomy", od, ld, b, OD_SLOTS, device)
        ld_type = _type(batch, "ld_taxonomy", od, ld, b, LD_SLOTS, device, ld_node=True)
        od_ids = _ids(batch, "od_ids", b, OD_SLOTS, device)
        ld_ids = _ids(batch, "ld_ids", b, LD_SLOTS, device)
        nav_mask = _mask(batch, "nav_mask", b, 1, device) if "nav_mask" in batch else torch.ones((b, 1), dtype=torch.bool, device=device)
        ego_mask = _mask(batch, "ego_mask", b, 1, device) if "ego_mask" in batch else torch.ones((b, 1), dtype=torch.bool, device=device)

        return {
            "od_content": od_content,
            "ld_content": ld_content,
            "od_geometry": od_geometry,
            "ld_geometry": ld_geometry,
            "od_type": od_type,
            "ld_type": ld_type,
            "od_valid": od_mask,
            "ld_valid": ld_mask,
            "od_stable_id": od_ids,
            "ld_stable_id": ld_ids,
            "nav": self.nav_encoder(nav),
            "ego": self.ego_encoder(ego),
            "nav_valid": nav_mask,
            "ego_valid": ego_mask,
        }

    def frame_records(
        self,
        batch: Mapping[str, Tensor],
        *,
        episode_id: str,
        frame_id: int,
        timestamp: float,
        ego_pose: Optional[Tensor] = None,
        encoder_version: int = 0,
        encoded: Optional[Mapping[str, Tensor]] = None,
    ) -> Tuple[FrameRecord, ...]:
        """Build one ``FrameRecord`` per batch element from the current frame."""
        out = self.encode_current(batch) if encoded is None else encoded
        if ego_pose is None:
            ego_pose = batch.get("ego_pose", batch.get("pose"))
            if ego_pose is None:
                raise ValueError("frame_records requires batch['ego_pose'] or batch['pose']")
            ego_pose = _take_last(ego_pose, 2)
        records = []
        for i in range(out["od_content"].shape[0]):
            od = tuple(
                EntityRecord(
                    slot=s,
                    stable_id=int(out["od_stable_id"][i, s]),
                    node_type=NodeType.OD,
                    content=out["od_content"][i, s],
                    geometry=out["od_geometry"][i, s],
                    type_id=int(out["od_type"][i, s]),
                    valid=bool(out["od_valid"][i, s]),
                )
                for s in range(OD_SLOTS)
            )
            ld = tuple(
                EntityRecord(
                    slot=s,
                    stable_id=int(out["ld_stable_id"][i, s]),
                    node_type=NodeType.LD,
                    content=out["ld_content"][i, s],
                    geometry=out["ld_geometry"][i, s],
                    type_id=int(out["ld_type"][i, s]),
                    valid=bool(out["ld_valid"][i, s]),
                )
                for s in range(LD_SLOTS)
            )
            records.append(
                FrameRecord(
                    episode_id=episode_id,
                    frame_id=int(frame_id),
                    timestamp=float(timestamp),
                    ego_pose=ego_pose[i].detach(),
                    od=od,
                    ld=ld,
                    source=Source.REAL,
                    rollout_depth=0,
                    encoder_version=int(encoder_version),
                )
            )
        return tuple(records)


class TargetEncoder(nn.Module):
    """Frozen/EMA target copy of :class:`CurrentFrameEncoder`."""

    def __init__(self, online: CurrentFrameEncoder | None = None, hidden: int = 128) -> None:
        super().__init__()
        if online is None:
            online = CurrentFrameEncoder(hidden=hidden)
        self.online = copy.deepcopy(online)
        for p in self.online.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True) -> "TargetEncoder":
        super().train(False)
        self.online.eval()
        return self

    @torch.no_grad()
    def update_ema(self, online: CurrentFrameEncoder, decay: float = 0.999) -> None:
        decay = float(decay)
        for target, source in zip(self.online.parameters(), online.parameters()):
            target.mul_(decay).add_(source.detach(), alpha=1.0 - decay)
        for target, source in zip(self.online.buffers(), online.buffers()):
            target.copy_(source.detach())

    @torch.no_grad()
    def encode_current(self, batch: Mapping[str, Tensor] | None = None, **kwargs: Tensor) -> Dict[str, Tensor]:
        return self.online.encode_current(batch, **kwargs)


# Stable public aliases for callers that use domain names.
Encoder = CurrentFrameEncoder
EMATargetEncoder = TargetEncoder

__all__ = [
    "CurrentFrameEncoder",
    "TargetEncoder",
    "Encoder",
    "EMATargetEncoder",
]
