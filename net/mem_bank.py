"""Real/imagined OD/LD memory banks and batched BankWindow views.

The bank is deliberately a *data* layer, not an ``nn.Module``:

* a :class:`FrameRecord` stores one environment/imagined frame;
* :class:`RealMemoryBank` is a six-frame FIFO of real frames;
* :class:`ImagineMemoryBank` is a copy-on-write extension of a real prefix;
* :class:`BankWindow` is the batched tensor view consumed by the network.

Bank records contain only OD/LD entities.  Ego/Nav are policy inputs and are
not stored in the bank.  Geometry is stored in each frame's ego frame and is
transformed lazily into the query (last-frame) ego frame when a window is
built.  Content latents are never rotated.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from enum import IntEnum
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

OD_SLOTS = 16
LD_SLOTS = 16
OD_GEOM_DIM = 5       # x, y, yaw, vx, vy
LD_POINTS = 5
LD_GEOM_DIM = 2 * LD_POINTS  # flattened x,y points
HISTORY_FRAMES = 6


class Source(IntEnum):
    REAL = 0
    IMAGINED = 1


class NodeType(IntEnum):
    OD = 0
    LD = 1


@dataclass(frozen=True)
class EntityRecord:
    """One OD or LD record in one frame.

    ``geometry`` layouts (all in the frame's ego coordinates):

    * OD: ``[x, y, yaw, vx, vy]``
    * LD: ``[x0, y0, ..., x4, y4]``
    """

    slot: int
    stable_id: int
    node_type: NodeType
    content: Tensor
    geometry: Tensor
    type_id: int = 0
    valid: bool = True

    def detached(self) -> "EntityRecord":
        return EntityRecord(
            slot=int(self.slot),
            stable_id=int(self.stable_id),
            node_type=self.node_type,
            content=self.content.detach(),
            geometry=self.geometry.detach(),
            type_id=int(self.type_id),
            valid=bool(self.valid),
        )


@dataclass(frozen=True)
class FrameRecord:
    """One real or imagined frame."""

    episode_id: str
    frame_id: int
    timestamp: float
    ego_pose: Tensor                  # [3] = x, y, yaw in world frame
    od: Tuple[EntityRecord, ...]
    ld: Tuple[EntityRecord, ...]
    source: Source = Source.REAL
    rollout_depth: int = 0
    parent_frame_id: int = -1
    action_from_parent: Optional[Tensor] = None
    encoder_version: int = 0

    def detached(self) -> "FrameRecord":
        return FrameRecord(
            episode_id=str(self.episode_id),
            frame_id=int(self.frame_id),
            timestamp=float(self.timestamp),
            ego_pose=self.ego_pose.detach(),
            od=tuple(e.detached() for e in self.od),
            ld=tuple(e.detached() for e in self.ld),
            source=self.source,
            rollout_depth=int(self.rollout_depth),
            parent_frame_id=int(self.parent_frame_id),
            action_from_parent=None if self.action_from_parent is None else self.action_from_parent.detach(),
            encoder_version=int(self.encoder_version),
        )


@dataclass
class BankWindow:
    """Batched fixed-slot view of a bank window.

    Shapes are batch-first.  A single bank returns ``B=1``; training can stack
    multiple per-episode windows with :meth:`stack`.
    """

    od_content: Tensor       # [B,T,16,H]
    od_geometry: Tensor      # [B,T,16,5]
    od_type: Tensor          # [B,T,16]
    od_valid: Tensor         # [B,T,16] bool
    od_stable_id: Tensor     # [B,T,16]
    ld_content: Tensor       # [B,T,16,H]
    ld_geometry: Tensor      # [B,T,16,10]
    ld_type: Tensor          # [B,T,16]
    ld_valid: Tensor         # [B,T,16] bool
    ld_stable_id: Tensor     # [B,T,16]
    ego_pose: Tensor         # [B,T,3]
    source: Tensor           # [B,T] int64
    rollout_depth: Tensor    # [B,T] int64
    frame_valid: Tensor      # [B,T] bool

    def to(self, device: torch.device | str) -> "BankWindow":
        values = {}
        for name, value in self.__dict__.items():
            values[name] = value.to(device) if torch.is_tensor(value) else value
        return BankWindow(**values)

    @classmethod
    def stack(cls, windows: Sequence["BankWindow"]) -> "BankWindow":
        if not windows:
            raise ValueError("BankWindow.stack requires at least one window")
        first = windows[0]
        fields = first.__dataclass_fields__
        for window in windows[1:]:
            for name in fields:
                a = getattr(first, name)
                b = getattr(window, name)
                if a.shape[1:] != b.shape[1:]:
                    raise ValueError(f"BankWindow field {name!r} has incompatible shape")
        values = {
            name: torch.cat([getattr(w, name) for w in windows], dim=0)
            for name in fields
        }
        return cls(**values)

    def to_frame_record(
        self,
        *,
        episode_id: str,
        frame_id: int,
        timestamp: float,
        source: Source | None = None,
        parent_frame_id: int = -1,
        encoder_version: int = 0,
        action_from_parent: Tensor | None = None,
    ) -> FrameRecord:
        """Convert the current (last) slice of a B=1 window to a FrameRecord."""
        if self.batch_size != 1:
            raise ValueError("to_frame_record requires B=1")
        i = -1

        def make(content: Tensor, geometry: Tensor, typ: Tensor, valid: Tensor, sid: Tensor, node: NodeType):
            return tuple(
                EntityRecord(
                    slot=s,
                    stable_id=int(sid[0, i, s]),
                    node_type=node,
                    content=content[0, i, s],
                    geometry=geometry[0, i, s],
                    type_id=int(typ[0, i, s]),
                    valid=bool(valid[0, i, s]),
                )
                for s in range(content.shape[2])
            )

        if source is None:
            raw_source = int(self.source[0, i].item())
            source = Source(raw_source) if raw_source in (0, 1) else Source.IMAGINED
        return FrameRecord(
            episode_id=str(episode_id),
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            ego_pose=self.ego_pose[0, i].detach(),
            od=make(self.od_content, self.od_geometry, self.od_type, self.od_valid, self.od_stable_id, NodeType.OD),
            ld=make(self.ld_content, self.ld_geometry, self.ld_type, self.ld_valid, self.ld_stable_id, NodeType.LD),
            source=source,
            rollout_depth=int(self.rollout_depth[0, i]),
            parent_frame_id=int(parent_frame_id),
            action_from_parent=None if action_from_parent is None else action_from_parent.detach(),
            encoder_version=int(encoder_version),
        )

    def advance(
        self,
        *,
        od_content: Tensor,
        od_geometry: Tensor,
        od_type: Tensor,
        od_valid: Tensor,
        od_stable_id: Tensor,
        ld_content: Tensor,
        ld_geometry: Tensor,
        ld_type: Tensor,
        ld_valid: Tensor,
        ld_stable_id: Tensor,
        ego_pose: Tensor,
        source: int = 1,
        rollout_depth: int = 1,
        source_pose: Tensor | None = None,
        target_pose: Tensor | None = None,
        project_new_geometry: bool = True,
    ) -> "BankWindow":
        """Shift the window one step and append current entities.

        This is the tensor-level state transition used by ``DriveModel.imagine``.
        It does not bypass the Bank schema: the caller remains responsible for
        writing the corresponding FrameRecord to an ImagineMemoryBank if a
        record-level audit is needed.
        """
        def shift_time(x: Tensor, y: Tensor) -> Tensor:
            return torch.cat([x[:, 1:], y[:, None]], dim=1)

        b = self.batch_size
        n_od, n_ld = self.od_content.shape[2], self.ld_content.shape[2]
        expected = {
            "od_content": (b, n_od, self.od_content.shape[-1]),
            "od_geometry": (b, n_od, self.od_geometry.shape[-1]),
            "od_type": (b, n_od),
            "od_valid": (b, n_od),
            "od_stable_id": (b, n_od),
            "ld_content": (b, n_ld, self.ld_content.shape[-1]),
            "ld_geometry": (b, n_ld, self.ld_geometry.shape[-1]),
            "ld_type": (b, n_ld),
            "ld_valid": (b, n_ld),
            "ld_stable_id": (b, n_ld),
            "ego_pose": (b, 3),
        }
        for name, want in expected.items():
            got = tuple(locals()[name].shape)
            if got != want:
                raise ValueError(f"advance {name} shape {got} != {want}")
        if int(rollout_depth) > 6:
            raise ValueError("rollout_depth must be <= 6")
        old_od = self.od_geometry[:, 1:]
        old_ld = self.ld_geometry[:, 1:]
        if source_pose is not None and target_pose is not None:
            old_od = BankWindowBuilder.project_geometry(old_od, source_pose, target_pose, NodeType.OD)
            old_ld = BankWindowBuilder.project_geometry(old_ld, source_pose, target_pose, NodeType.LD)
            if project_new_geometry:
                new_od = BankWindowBuilder.project_geometry(od_geometry, source_pose, target_pose, NodeType.OD)
                new_ld = BankWindowBuilder.project_geometry(ld_geometry, source_pose, target_pose, NodeType.LD)
            else:
                new_od, new_ld = od_geometry, ld_geometry
        else:
            new_od, new_ld = od_geometry, ld_geometry
        od_geometry = torch.cat([old_od, new_od[:, None]], dim=1)
        ld_geometry = torch.cat([old_ld, new_ld[:, None]], dim=1)
        new_source = torch.full((b,), int(source), dtype=self.source.dtype, device=self.source.device)
        new_depth = torch.full((b,), int(rollout_depth), dtype=self.rollout_depth.dtype, device=self.rollout_depth.device)
        new_frame = torch.ones((b,), dtype=torch.bool, device=self.frame_valid.device)
        return BankWindow(
            od_content=shift_time(self.od_content, od_content),
            od_geometry=od_geometry,
            od_type=shift_time(self.od_type, od_type),
            od_valid=shift_time(self.od_valid, od_valid),
            od_stable_id=shift_time(self.od_stable_id, od_stable_id),
            ld_content=shift_time(self.ld_content, ld_content),
            ld_geometry=ld_geometry,
            ld_type=shift_time(self.ld_type, ld_type),
            ld_valid=shift_time(self.ld_valid, ld_valid),
            ld_stable_id=shift_time(self.ld_stable_id, ld_stable_id),
            ego_pose=shift_time(self.ego_pose, ego_pose),
            source=shift_time(self.source, new_source),
            rollout_depth=shift_time(self.rollout_depth, new_depth),
            frame_valid=shift_time(self.frame_valid, new_frame),
        )

    @property
    def batch_size(self) -> int:
        return int(self.od_content.shape[0])


def _wrap_angle(x: Tensor) -> Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


class BankWindowBuilder:
    """Build a batched window from per-frame records."""

    def __init__(
        self,
        *,
        capacity: int = HISTORY_FRAMES,
        od_slots: int = OD_SLOTS,
        ld_slots: int = LD_SLOTS,
        od_geom_dim: int = OD_GEOM_DIM,
        ld_geom_dim: int = LD_GEOM_DIM,
        hidden: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.od_slots = int(od_slots)
        self.ld_slots = int(ld_slots)
        self.od_geom_dim = int(od_geom_dim)
        self.ld_geom_dim = int(ld_geom_dim)
        self.hidden = None if hidden is None else int(hidden)

    @staticmethod
    def _transform_geometry(
        geometry: Tensor,
        source_pose: Tensor,
        query_pose: Tensor,
        *,
        node_type: NodeType,
    ) -> Tensor:
        dtype, device = geometry.dtype, geometry.device
        sx, sy, syaw = source_pose.to(device=device, dtype=dtype)
        qx, qy, qyaw = query_pose.to(device=device, dtype=dtype)
        cs, ss = torch.cos(syaw), torch.sin(syaw)
        cq, sq = torch.cos(qyaw), torch.sin(qyaw)

        def to_query(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
            wx = sx + cs * x - ss * y
            wy = sy + ss * x + cs * y
            dx, dy = wx - qx, wy - qy
            return cq * dx + sq * dy, -sq * dx + cq * dy

        out = geometry.clone()
        if node_type == NodeType.OD:
            dx, dy = to_query(geometry[0], geometry[1])
            out[0], out[1] = dx, dy
            if geometry.numel() >= 3:
                out[2] = _wrap_angle(geometry[2] + syaw - qyaw)
            if geometry.numel() >= 5:
                vx, vy = geometry[3], geometry[4]
                wvx = cs * vx - ss * vy
                wvy = ss * vx + cs * vy
                out[3] = cq * wvx + sq * wvy
                out[4] = -sq * wvx + cq * wvy
            return out

        pts = geometry.reshape(-1, 2)
        x, y = to_query(pts[:, 0], pts[:, 1])
        return torch.stack((x, y), dim=-1).reshape(-1)

    @staticmethod
    def relative_pose(geometry: Tensor, node_type: NodeType) -> Tensor:
        """Return ``[...,3]`` relative (x,y,yaw) in the query ego frame."""
        if node_type == NodeType.OD:
            if geometry.shape[-1] < 3:
                raise ValueError("OD geometry needs at least x,y,yaw")
            return geometry[..., :3]
        pts = geometry.reshape(*geometry.shape[:-1], -1, 2)
        center = pts.mean(dim=-2)
        delta = pts[..., -1, :] - pts[..., 0, :]
        norm = torch.linalg.vector_norm(delta, dim=-1)
        safe_x = torch.where(norm < 1e-8, torch.ones_like(delta[..., 0]), delta[..., 0])
        safe_y = torch.where(norm < 1e-8, torch.zeros_like(delta[..., 1]), delta[..., 1])
        yaw = torch.atan2(safe_y, safe_x)
        yaw = torch.where(norm < 1e-8, torch.zeros_like(yaw), yaw)
        return torch.stack([center[..., 0], center[..., 1], yaw], dim=-1)

    @staticmethod
    def project_geometry(
        geometry: Tensor,
        source_pose: Tensor,
        target_pose: Tensor,
        node_type: NodeType,
    ) -> Tensor:
        """Project geometry to a target ego frame, preserving leading dims.

        Supports ``[G]``, ``[N,G]``, ``[B,N,G]`` and ``[B,T,N,G]``.  Pose
        tensors may be ``[3]``, ``[B,3]`` or ``[B,T,3]``.
        """
        if geometry.dim() == 1:
            return BankWindowBuilder._transform_geometry(
                geometry, source_pose, target_pose, node_type=node_type
            )
        if geometry.dim() == 2:
            if source_pose.dim() == 1 and target_pose.dim() == 1:
                return torch.stack(
                    [
                        BankWindowBuilder._transform_geometry(
                            geometry[i], source_pose, target_pose, node_type=node_type
                        )
                        for i in range(geometry.shape[0])
                    ],
                    dim=0,
                )
            raise ValueError("2D geometry needs 1D poses")
        if geometry.dim() == 3:
            if source_pose.dim() == 1 and target_pose.dim() == 1:
                return torch.stack(
                    [
                        BankWindowBuilder.project_geometry(
                            geometry[i], source_pose, target_pose, node_type
                        )
                        for i in range(geometry.shape[0])
                    ],
                    dim=0,
                )
            if source_pose.dim() == 2 and target_pose.dim() == 2:
                return torch.stack(
                    [
                        BankWindowBuilder.project_geometry(
                            geometry[i], source_pose[i], target_pose[i], node_type
                        )
                        for i in range(geometry.shape[0])
                    ],
                    dim=0,
                )
            raise ValueError("3D geometry needs 1D or [B,3] poses")
        if geometry.dim() == 4:
            if source_pose.dim() == 3 and target_pose.dim() == 3:
                return torch.stack(
                    [
                        BankWindowBuilder.project_geometry(
                            geometry[i], source_pose[i], target_pose[i], node_type
                        )
                        for i in range(geometry.shape[0])
                    ],
                    dim=0,
                )
            if source_pose.dim() == 2 and target_pose.dim() == 2:
                return torch.stack(
                    [
                        BankWindowBuilder.project_geometry(
                            geometry[i], source_pose[i], target_pose[i], node_type
                        )
                        for i in range(geometry.shape[0])
                    ],
                    dim=0,
                )
            raise ValueError("[B,T,N,G] geometry needs [B,T,3] or [B,3] poses")
        raise ValueError(f"unsupported geometry shape {tuple(geometry.shape)}")

    def _infer_dims(self, frames: Sequence[FrameRecord]) -> Tuple[int, int, int]:
        contents = [e.content.numel() for f in frames for e in (*f.od, *f.ld)]
        if not contents:
            if self.hidden is None:
                raise ValueError("cannot infer hidden size from empty frames")
            hidden = self.hidden
        else:
            hidden = int(contents[0])
            if any(int(x) != hidden for x in contents):
                raise ValueError("all entity contents must have the same hidden size")
        return hidden, self.od_geom_dim, self.ld_geom_dim

    def build(self, frames: Sequence[FrameRecord]) -> BankWindow:
        if not frames:
            raise ValueError("cannot build BankWindow from an empty frame list")
        frames = tuple(frames[-self.capacity:])
        hidden, od_g, ld_g = self._infer_dims(frames)
        T = self.capacity
        pad = T - len(frames)
        device = frames[-1].ego_pose.device
        content_dtype = next(
            (e.content.dtype for f in frames for e in (*f.od, *f.ld)),
            torch.float32,
        )

        def zeros(*shape: int) -> Tensor:
            return torch.zeros(shape, dtype=content_dtype, device=device)

        od_content = zeros(1, T, self.od_slots, hidden)
        od_geometry = zeros(1, T, self.od_slots, od_g)
        od_type = torch.zeros((1, T, self.od_slots), dtype=torch.long, device=device)
        od_valid = torch.zeros((1, T, self.od_slots), dtype=torch.bool, device=device)
        od_stable = torch.full((1, T, self.od_slots), -1, dtype=torch.long, device=device)
        ld_content = zeros(1, T, self.ld_slots, hidden)
        ld_geometry = zeros(1, T, self.ld_slots, ld_g)
        ld_type = torch.zeros((1, T, self.ld_slots), dtype=torch.long, device=device)
        ld_valid = torch.zeros((1, T, self.ld_slots), dtype=torch.bool, device=device)
        ld_stable = torch.full((1, T, self.ld_slots), -1, dtype=torch.long, device=device)
        ego_pose = zeros(1, T, 3)
        source = torch.full((1, T), -1, dtype=torch.long, device=device)
        depth = torch.full((1, T), -1, dtype=torch.long, device=device)
        frame_valid = torch.zeros((1, T), dtype=torch.bool, device=device)

        # History is indexed by the *current* slot order.  An entity is matched
        # by stable ID, not by its rank in the historical frame.  This is the
        # critical difference between a storage slot and an identity.
        def current_map(field: str) -> dict[int, EntityRecord]:
            out: dict[int, EntityRecord] = {}
            for rec in getattr(frames[-1], field):
                if not rec.valid:
                    continue
                if not (0 <= rec.slot < (self.od_slots if field == "od" else self.ld_slots)):
                    raise ValueError(f"{field} slot {rec.slot} out of range")
                if rec.stable_id in out:
                    raise ValueError(f"duplicate stable_id {rec.stable_id} in current {field}")
                out[int(rec.stable_id)] = rec
            slots = [rec.slot for rec in out.values()]
            if len(slots) != len(set(slots)):
                raise ValueError(f"duplicate {field} slot in current frame")
            return out

        current_od = current_map("od")
        current_ld = current_map("ld")
        query_pose = frames[-1].ego_pose.to(device=device, dtype=content_dtype)

        def frame_map(frame: FrameRecord, field: str, slots: int) -> dict[int, EntityRecord]:
            out: dict[int, EntityRecord] = {}
            used_slots: set[int] = set()
            for rec in getattr(frame, field):
                expected = NodeType.OD if field == "od" else NodeType.LD
                if rec.node_type != expected:
                    raise ValueError(f"{field} record has node_type={rec.node_type}")
                if not (0 <= rec.slot < slots):
                    raise ValueError(f"{field} slot {rec.slot} out of range")
                if rec.valid and rec.slot in used_slots:
                    raise ValueError(f"duplicate {field} slot {rec.slot} in frame")
                if rec.valid:
                    used_slots.add(rec.slot)
                    if rec.stable_id in out:
                        raise ValueError(f"duplicate stable_id {rec.stable_id} in {field} frame")
                    out[int(rec.stable_id)] = rec
            return out

        for t, frame in enumerate(frames, start=pad):
            frame_valid[0, t] = True
            source[0, t] = int(frame.source)
            depth[0, t] = int(frame.rollout_depth)
            pose = frame.ego_pose.to(device=device, dtype=content_dtype)
            ego_pose[0, t] = pose
            hist_od = frame_map(frame, "od", self.od_slots)
            hist_ld = frame_map(frame, "ld", self.ld_slots)
            for sid, cur in current_od.items():
                rec = hist_od.get(sid)
                if rec is None:
                    od_stable[0, t, cur.slot] = int(sid)
                    continue
                if rec.content.numel() != hidden:
                    raise ValueError("OD content hidden mismatch")
                if rec.geometry.numel() != od_g:
                    raise ValueError(f"OD geometry must have {od_g} values")
                g = self._transform_geometry(rec.geometry, pose, query_pose, node_type=NodeType.OD)
                od_content[0, t, cur.slot] = rec.content.to(dtype=content_dtype, device=device)
                od_geometry[0, t, cur.slot] = g.to(dtype=content_dtype, device=device)
                od_type[0, t, cur.slot] = int(cur.type_id)
                od_valid[0, t, cur.slot] = bool(rec.valid)
                od_stable[0, t, cur.slot] = int(sid)
            for sid, cur in current_ld.items():
                rec = hist_ld.get(sid)
                if rec is None:
                    ld_stable[0, t, cur.slot] = int(sid)
                    continue
                if rec.content.numel() != hidden:
                    raise ValueError("LD content hidden mismatch")
                if rec.geometry.numel() != ld_g:
                    raise ValueError(f"LD geometry must have {ld_g} values")
                g = self._transform_geometry(rec.geometry, pose, query_pose, node_type=NodeType.LD)
                ld_content[0, t, cur.slot] = rec.content.to(dtype=content_dtype, device=device)
                ld_geometry[0, t, cur.slot] = g.to(dtype=content_dtype, device=device)
                ld_type[0, t, cur.slot] = int(cur.type_id)
                ld_valid[0, t, cur.slot] = bool(rec.valid)
                ld_stable[0, t, cur.slot] = int(sid)

        return BankWindow(
            od_content=od_content,
            od_geometry=od_geometry,
            od_type=od_type,
            od_valid=od_valid,
            od_stable_id=od_stable,
            ld_content=ld_content,
            ld_geometry=ld_geometry,
            ld_type=ld_type,
            ld_valid=ld_valid,
            ld_stable_id=ld_stable,
            ego_pose=ego_pose,
            source=source,
            rollout_depth=depth,
            frame_valid=frame_valid,
        )


class RealMemoryBank:
    """Six-frame real FIFO."""

    def __init__(self, capacity: int = HISTORY_FRAMES) -> None:
        self.capacity = int(capacity)
        self._frames: list[FrameRecord] = []
        self._episode_id: Optional[str] = None

    def reset(self, episode_id: Optional[str] = None) -> None:
        self._frames.clear()
        self._episode_id = episode_id

    def append(self, frame: FrameRecord, *, detach: bool = True) -> None:
        if frame.source != Source.REAL:
            raise ValueError("RealMemoryBank accepts REAL frames only")
        if self._episode_id is None:
            self._episode_id = frame.episode_id
        elif frame.episode_id != self._episode_id:
            self.reset(frame.episode_id)
        if self._frames:
            prev = self._frames[-1]
            if frame.frame_id <= prev.frame_id:
                raise ValueError("frame_id must strictly increase")
            if frame.timestamp <= prev.timestamp:
                raise ValueError("timestamp must strictly increase")
        self._frames.append(frame.detached() if detach else frame)
        if len(self._frames) > self.capacity:
            self._frames = self._frames[-self.capacity:]

    def snapshot(self) -> "ImagineMemoryBank":
        if not self._frames:
            raise RuntimeError("cannot snapshot an empty RealMemoryBank")
        return ImagineMemoryBank(copy.deepcopy(self._frames), capacity=self.capacity)

    def window(self, *, builder: Optional[BankWindowBuilder] = None) -> BankWindow:
        if not self._frames:
            raise RuntimeError("cannot build a window from an empty RealMemoryBank")
        return (builder or BankWindowBuilder(capacity=self.capacity)).build(self._frames)

    @property
    def frames(self) -> Tuple[FrameRecord, ...]:
        return tuple(self._frames)

    def __len__(self) -> int:
        return len(self._frames)


class ImagineMemoryBank:
    """Copy-on-write imagined extension of a real prefix."""

    def __init__(self, real_prefix: Sequence[FrameRecord], capacity: int = HISTORY_FRAMES) -> None:
        if not real_prefix:
            raise ValueError("real_prefix must not be empty")
        self.capacity = int(capacity)
        self._real_prefix = tuple(real_prefix[-self.capacity :])
        self._imagined: list[FrameRecord] = []
        self.episode_id = self._real_prefix[-1].episode_id

    def append(self, frame: FrameRecord, *, detach: bool = True) -> None:
        if frame.source != Source.IMAGINED:
            raise ValueError("ImagineMemoryBank accepts IMAGINED frames only")
        if frame.episode_id != self.episode_id:
            raise ValueError("imagined episode mismatch")
        prev = self.frames[-1]
        if frame.parent_frame_id != prev.frame_id:
            raise ValueError("parent_frame_id must reference the current tail")
        if frame.rollout_depth != prev.rollout_depth + 1:
            raise ValueError("rollout_depth must increase by one")
        if frame.rollout_depth > self.capacity:
            raise ValueError(f"rollout_depth must be <= {self.capacity}")
        if frame.timestamp <= prev.timestamp:
            raise ValueError("imagined timestamp must increase")
        self._imagined.append(frame.detached() if detach else frame)
        if len(self._imagined) > self.capacity:
            self._imagined.pop(0)

    def clear(self) -> None:
        self._imagined.clear()

    @property
    def frames(self) -> Tuple[FrameRecord, ...]:
        return tuple((list(self._real_prefix) + self._imagined)[-self.capacity :])

    def window(self, *, builder: Optional[BankWindowBuilder] = None) -> BankWindow:
        return (builder or BankWindowBuilder(capacity=self.capacity)).build(self.frames)

    def __len__(self) -> int:
        return len(self.frames)
