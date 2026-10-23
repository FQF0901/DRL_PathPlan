"""Real/imagined entity memory banks and deterministic SE(2) view building.

The bank is the only history API. Environment observations append one FrameRecord
at a time; no env-side stacked history is required. Geometry is stored in the
frame ego coordinate system and transformed lazily into the query frame.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


class Source(IntEnum):
    REAL = 0
    IMAGINED = 1


class NodeType(IntEnum):
    EGO = 0
    NAV = 1
    OD = 2
    LD = 3


@dataclass(frozen=True)
class EntityRecord:
    stable_id: int
    node_type: NodeType
    latent: Tensor                 # [H], content latent only
    geometry: Tensor               # [G>=4]: x,y,heading plus optional geometry
    present: bool = True
    taxonomy: int = 0
    metadata: Mapping[str, float] = field(default_factory=dict)

    def detached(self) -> "EntityRecord":
        return EntityRecord(
            stable_id=int(self.stable_id), node_type=self.node_type,
            latent=self.latent.detach(), geometry=self.geometry.detach(),
            present=bool(self.present), taxonomy=int(self.taxonomy),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class FrameRecord:
    episode_id: str
    frame_id: int
    timestamp: float
    ego_pose_world: Tensor         # [3] = x,y,yaw
    entities: Tuple[EntityRecord, ...]
    source: Source = Source.REAL
    rollout_depth: int = 0
    parent_frame_id: int = -1
    action_from_parent: Optional[Tensor] = None
    encoder_version: int = 0

    def detached(self) -> "FrameRecord":
        return FrameRecord(
            episode_id=self.episode_id, frame_id=int(self.frame_id),
            timestamp=float(self.timestamp), ego_pose_world=self.ego_pose_world.detach(),
            entities=tuple(e.detached() for e in self.entities), source=self.source,
            rollout_depth=int(self.rollout_depth), parent_frame_id=int(self.parent_frame_id),
            action_from_parent=None if self.action_from_parent is None else self.action_from_parent.detach(),
            encoder_version=int(self.encoder_version),
        )


@dataclass
class BankView:
    """Packed graph in one query ego frame."""
    x: Tensor                      # [N,H]
    geometry: Tensor               # [N,G], x/y/heading transformed to query frame
    stable_id: Tensor              # [N]
    node_type: Tensor              # [N]
    frame_index: Tensor            # [N], chronological 0..T-1
    timestamp: Tensor              # [N]
    source: Tensor                 # [N]
    rollout_depth: Tensor          # [N]
    present: Tensor                # [N] bool
    birth: Tensor                  # [N] bool, no temporal parent in previous frame
    temporal_parent: Tensor        # [N], -1 or node index
    temporal_edge_index: Tensor    # [2,E_t], parent -> child
    spatial_edge_index: Tensor     # [2,E_s], current-query-frame edges
    spatial_edge_attr: Tensor      # [E_s,6]: dx,dy,dist,sin(dyaw),cos(dyaw),type_pair
    readout_mask: Tensor           # [N] current frame and active
    query_pose_world: Tensor       # [3]
    frame_valid: Tensor            # [T] bool

    def to(self, device: torch.device | str) -> "BankView":
        values = {}
        for name, value in self.__dict__.items():
            values[name] = value.to(device) if torch.is_tensor(value) else value
        return BankView(**values)


class RealMemoryBank:
    def __init__(self, capacity: int = 6, expected_dt: float = 0.5):
        self.capacity = int(capacity)
        self.expected_dt = float(expected_dt)
        self._frames: Deque[FrameRecord] = deque(maxlen=self.capacity)
        self._episode_id: Optional[str] = None

    def reset(self, episode_id: Optional[str] = None) -> None:
        self._frames.clear()
        self._episode_id = episode_id

    def append(self, frame: FrameRecord, detach: bool = True) -> None:
        if frame.source != Source.REAL:
            raise ValueError("RealMemoryBank accepts REAL frames only")
        if self._episode_id is None:
            self._episode_id = frame.episode_id
        if frame.episode_id != self._episode_id:
            self.reset(frame.episode_id)
        if self._frames:
            prev = self._frames[-1]
            if frame.frame_id <= prev.frame_id or frame.timestamp <= prev.timestamp:
                raise ValueError("frames must be appended in strictly increasing order")
        self._frames.append(frame.detached() if detach else frame)

    @property
    def frames(self) -> Tuple[FrameRecord, ...]:
        return tuple(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def snapshot(self) -> "ImagineMemoryBank":
        if not self._frames:
            raise RuntimeError("cannot snapshot an empty RealMemoryBank")
        return ImagineMemoryBank(tuple(self._frames), capacity=self.capacity)


class ImagineMemoryBank:
    """Copy-on-write rollout bank. Real parent frames are never mutated."""
    def __init__(self, real_prefix: Sequence[FrameRecord], capacity: int = 6):
        if not real_prefix:
            raise ValueError("real_prefix must not be empty")
        self.capacity = int(capacity)
        self._real_prefix = tuple(real_prefix[-self.capacity:])
        self._imagined: List[FrameRecord] = []
        self.episode_id = self._real_prefix[-1].episode_id

    @property
    def frames(self) -> Tuple[FrameRecord, ...]:
        return tuple((list(self._real_prefix) + self._imagined)[-self.capacity:])

    def append(self, frame: FrameRecord, detach: bool = True) -> None:
        if frame.source != Source.IMAGINED:
            raise ValueError("ImagineMemoryBank accepts IMAGINED frames only")
        if frame.episode_id != self.episode_id:
            raise ValueError("imagined frame episode mismatch")
        previous = self.frames[-1]
        if frame.parent_frame_id != previous.frame_id:
            raise ValueError("parent_frame_id must reference current rollout tail")
        if frame.rollout_depth != previous.rollout_depth + 1:
            raise ValueError("rollout_depth must increase by one")
        if frame.timestamp <= previous.timestamp:
            raise ValueError("imagined timestamps must increase")
        self._imagined.append(frame.detached() if detach else frame)

    def clear(self) -> None:
        self._imagined.clear()


class BankViewBuilder:
    def __init__(self, spatial_radius: float = 60.0, spatial_topk: int = 16):
        self.spatial_radius = float(spatial_radius)
        self.spatial_topk = int(spatial_topk)

    @staticmethod
    def _transform_geometry(geometry: Tensor, source_pose: Tensor, query_pose: Tensor) -> Tensor:
        """Transform source-ego geometry into query-ego coordinates."""
        if geometry.numel() < 3:
            raise ValueError("geometry requires at least x,y,heading")
        out = geometry.clone()
        dtype, device = geometry.dtype, geometry.device
        sx, sy, syaw = source_pose.to(device=device, dtype=dtype)
        qx, qy, qyaw = query_pose.to(device=device, dtype=dtype)
        cs, ss = torch.cos(syaw), torch.sin(syaw)
        wx = sx + cs * geometry[0] - ss * geometry[1]
        wy = sy + ss * geometry[0] + cs * geometry[1]
        dx, dy = wx - qx, wy - qy
        cq, sq = torch.cos(qyaw), torch.sin(qyaw)
        out[0] = cq * dx + sq * dy
        out[1] = -sq * dx + cq * dy
        out[2] = torch.atan2(torch.sin(geometry[2] + syaw - qyaw),
                             torch.cos(geometry[2] + syaw - qyaw))
        # Optional vx,vy at indices 3,4: rotate only. Other scalars stay unchanged.
        if geometry.numel() >= 5:
            vwx = cs * geometry[3] - ss * geometry[4]
            vwy = ss * geometry[3] + cs * geometry[4]
            out[3] = cq * vwx + sq * vwy
            out[4] = -sq * vwx + cq * vwy
        return out

    def build(self, frames: Sequence[FrameRecord]) -> BankView:
        if not frames:
            raise ValueError("frames must not be empty")
        episode = frames[-1].episode_id
        if any(f.episode_id != episode for f in frames):
            raise ValueError("a BankView cannot cross episode boundaries")
        latent_dim = frames[-1].entities[0].latent.numel()
        geom_dim = max(e.geometry.numel() for f in frames for e in f.entities)
        query_pose = frames[-1].ego_pose_world

        xs: List[Tensor] = []
        geoms: List[Tensor] = []
        ids: List[int] = []
        types: List[int] = []
        frame_idx: List[int] = []
        times: List[float] = []
        sources: List[int] = []
        depths: List[int] = []
        presents: List[bool] = []
        births: List[bool] = []
        parents: List[int] = []
        temporal_edges: List[Tuple[int, int]] = []
        previous_by_key: Dict[Tuple[int, int], int] = {}

        for t, frame in enumerate(frames):
            current_by_key: Dict[Tuple[int, int], int] = {}
            for entity in frame.entities:
                if entity.latent.numel() != latent_dim:
                    raise ValueError("all entity latents must have the same hidden size")
                idx = len(xs)
                key = (int(entity.node_type), int(entity.stable_id))
                parent = previous_by_key.get(key, -1)
                if parent >= 0:
                    temporal_edges.append((parent, idx))
                g = torch.zeros(geom_dim, dtype=entity.geometry.dtype, device=entity.geometry.device)
                transformed = self._transform_geometry(entity.geometry, frame.ego_pose_world, query_pose)
                g[:transformed.numel()] = transformed
                xs.append(entity.latent)
                geoms.append(g)
                ids.append(entity.stable_id); types.append(int(entity.node_type)); frame_idx.append(t)
                times.append(frame.timestamp); sources.append(int(frame.source)); depths.append(frame.rollout_depth)
                presents.append(entity.present); births.append(parent < 0); parents.append(parent)
                current_by_key[key] = idx
            previous_by_key = current_by_key

        x = torch.stack(xs)
        geometry = torch.stack(geoms)
        device = x.device
        node_type = torch.tensor(types, dtype=torch.long, device=device)
        frame_index = torch.tensor(frame_idx, dtype=torch.long, device=device)
        present = torch.tensor(presents, dtype=torch.bool, device=device)
        readout = (frame_index == len(frames) - 1) & present
        temporal_edge_index = self._edge_tensor(temporal_edges, device)
        spatial_edge_index, spatial_edge_attr = self._spatial_edges(geometry, node_type, readout)
        return BankView(
            x=x, geometry=geometry,
            stable_id=torch.tensor(ids, dtype=torch.long, device=device), node_type=node_type,
            frame_index=frame_index, timestamp=torch.tensor(times, dtype=x.dtype, device=device),
            source=torch.tensor(sources, dtype=torch.long, device=device),
            rollout_depth=torch.tensor(depths, dtype=torch.long, device=device), present=present,
            birth=torch.tensor(births, dtype=torch.bool, device=device),
            temporal_parent=torch.tensor(parents, dtype=torch.long, device=device),
            temporal_edge_index=temporal_edge_index,
            spatial_edge_index=spatial_edge_index, spatial_edge_attr=spatial_edge_attr,
            readout_mask=readout, query_pose_world=query_pose.to(device),
            frame_valid=torch.ones(len(frames), dtype=torch.bool, device=device),
        )

    @staticmethod
    def _edge_tensor(edges: Sequence[Tuple[int, int]], device: torch.device) -> Tensor:
        if not edges:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()

    def _spatial_edges(self, geometry: Tensor, node_type: Tensor, readout: Tensor) -> Tuple[Tensor, Tensor]:
        current = torch.nonzero(readout, as_tuple=False).flatten()
        edges: List[Tuple[int, int]] = []
        attrs: List[Tensor] = []
        for dst in current.tolist():
            delta = geometry[current, :2] - geometry[dst, :2]
            dist = torch.linalg.vector_norm(delta, dim=-1)
            keep = (dist <= self.spatial_radius) & (current != dst)
            candidates = current[keep]
            dists = dist[keep]
            if candidates.numel() > self.spatial_topk:
                order = torch.topk(dists, self.spatial_topk, largest=False).indices
                candidates, dists = candidates[order], dists[order]
            for src, distance in zip(candidates.tolist(), dists):
                dxdy = geometry[src, :2] - geometry[dst, :2]
                dyaw = geometry[src, 2] - geometry[dst, 2]
                pair = node_type[src].to(geometry.dtype) * 4.0 + node_type[dst].to(geometry.dtype)
                attrs.append(torch.stack((dxdy[0], dxdy[1], distance,
                                          torch.sin(dyaw), torch.cos(dyaw), pair)))
                edges.append((src, dst))
        edge_index = self._edge_tensor(edges, geometry.device)
        edge_attr = torch.stack(attrs) if attrs else geometry.new_empty((0, 6))
        return edge_index, edge_attr
