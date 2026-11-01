"""Bank-native top-level DriveModel.

The old legacy wrapper path has been removed from the
network package.  Policy, world model and rollout are all composed here from
the bank-native components.
"""

from __future__ import annotations

from typing import Dict, Mapping

import torch
from torch import Tensor, nn

from env.obs import EGO_DIM, EGO_INDEX
from net.encoder import CurrentFrameEncoder
from net.mem_bank import (
    BankWindow,
    BankWindowBuilder,
    ImagineMemoryBank,
    RealMemoryBank,
)
from net.planner import Planner
from net.world_model import (
    EgoConditionedLatentWM,
    TemporalSpatialBlock,
    compose_pose,
    imagine_rollout,
)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DriveModel(nn.Module):
    """Current-frame policy + temporal-spatial bank encoder + latent WM."""

    def __init__(
        self,
        encoder: CurrentFrameEncoder | None = None,
        temporal_spatial: TemporalSpatialBlock | None = None,
        planner: Planner | None = None,
        world_model: EgoConditionedLatentWM | None = None,
        *,
        hidden: int = 128,
        heads: int = 4,
        spatial_layers: int = 2,
        action_dim: int = 4,
        rollout_horizon: int = 6,
        action_low=None,
        action_high=None,
        ego_dt: float | None = None,
        ego_wheelbase: float | None = None,
        ego_max_steer_rad: float | None = None,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.spatial_layers = int(spatial_layers)
        self.action_dim = int(action_dim)
        self.rollout_horizon = int(rollout_horizon)
        if ego_dt is None or ego_wheelbase is None or ego_max_steer_rad is None:
            raise ValueError("ego dt/wheelbase/max_steer_rad are required; use build_net or pass config values")
        self.ego_dt = float(ego_dt)
        self.ego_wheelbase = float(ego_wheelbase)
        self.ego_max_steer_rad = float(ego_max_steer_rad)
        if self.ego_dt <= 0.0 or self.ego_wheelbase <= 0.0 or self.ego_max_steer_rad <= 0.0:
            raise ValueError("ego dt/wheelbase/max_steer_rad must be positive")
        self.encoder = encoder or CurrentFrameEncoder(hidden=self.hidden)
        self.temporal_spatial = temporal_spatial or TemporalSpatialBlock(
            hidden=self.hidden,
            heads=heads,
            spatial_layers=spatial_layers,
        )
        self.planner = planner or Planner(
            hidden=self.hidden, action_dim=self.action_dim, heads=heads
        )
        self.world_model = world_model or EgoConditionedLatentWM(
            hidden=self.hidden,
            action_dim=self.action_dim,
        )
        if action_low is None or action_high is None:
            self.action_low = None
            self.action_high = None
            self.action_config = {"name": "unset", "low": [], "high": [], "limits": {}}
        else:
            self.register_buffer("action_low", torch.tensor(list(action_low), dtype=torch.float32))
            self.register_buffer("action_high", torch.tensor(list(action_high), dtype=torch.float32))
            self.action_config = {
                "name": "custom",
                "low": [float(x) for x in action_low],
                "high": [float(x) for x in action_high],
                "limits": {},
            }
        self.model_config = {
            "hidden": self.hidden,
            "attention_heads": int(heads),
            "spatial_layers": int(spatial_layers),
            "history_frames": 6,
            "action_dim": self.action_dim,
        }
        self.eval_config = {
            "hidden": self.hidden,
            "heads": int(heads),
            "spatial_layers": int(spatial_layers),
        }
        self.ego_config = {
            "dt": self.ego_dt,
            "wheelbase": self.ego_wheelbase,
            "max_steer_rad": self.ego_max_steer_rad,
        }

    # ------------------------------------------------------------------
    # Action / ego helpers
    # ------------------------------------------------------------------
    def _to_requested(self, action: Tensor) -> Tensor:
        """Affine normalized policy action -> physical requested action."""
        if action.shape[-1] != 4:
            raise ValueError("action must have 4 dims")
        if self.action_low is None or self.action_high is None:
            raise RuntimeError("action bounds are not configured; use build_net or pass action_low/action_high")
        low = self.action_low.to(dtype=action.dtype, device=action.device)
        high = self.action_high.to(dtype=action.dtype, device=action.device)
        return low + action * (high - low)

    def _ego_step(
        self,
        ego_features: Tensor,
        requested: Tensor,
        source_pose: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Bicycle-ish ego feature update; nav is fixed at t0.

        ``dt``/wheelbase/max-steer come from ``config/model.yaml`` and are
        stored on the model at build time.
        """
        if ego_features.shape[-1] != EGO_DIM:
            raise ValueError(f"ego_features must have {EGO_DIM} values")
        speed = requested[..., 3]
        yaw_rate = requested[..., 2] / self.ego_dt
        a_long = (speed - ego_features[..., EGO_INDEX["speed"]]) / self.ego_dt
        curvature = yaw_rate / speed.clamp_min(0.5)
        a_lat = speed * yaw_rate
        next_ego = ego_features.clone()
        next_ego[..., EGO_INDEX["speed"]] = speed
        next_ego[..., EGO_INDEX["a_long"]] = a_long
        next_ego[..., EGO_INDEX["a_lat"]] = a_lat
        next_ego[..., EGO_INDEX["yaw_rate"]] = yaw_rate
        # Match env.tracking's normalized steering convention.
        next_ego[..., EGO_INDEX["steer"]] = torch.clamp(
            torch.atan(self.ego_wheelbase * curvature) / self.ego_max_steer_rad,
            -1.0,
            1.0,
        )
        next_ego[..., EGO_INDEX["curvature"]] = curvature
        next_ego[..., EGO_INDEX["prev_dx"]] = requested[..., 0]
        next_ego[..., EGO_INDEX["prev_dpsi"]] = requested[..., 2]
        next_ego[..., EGO_INDEX["v_long"]] = speed
        next_ego[..., EGO_INDEX["v_lat"]] = 0.0
        next_pose = compose_pose(source_pose, requested)
        return next_pose, next_ego

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def encode_current(self, batch):
        return self.encoder.encode_current(batch)

    def forward(
        self,
        window: BankWindow | Dict[str, Tensor],
        nav: Tensor | None = None,
        ego: Tensor | None = None,
        *,
        ego_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if isinstance(window, BankWindow):
            if nav is None or ego is None:
                raise ValueError("forward_window requires nav and ego")
            return self.forward_window(window, nav, ego, ego_valid=ego_valid)
        if nav is not None or ego is not None:
            raise ValueError("forward takes only the observation batch")
        if torch.is_tensor(window.get("od")) and window["od"].dim() == 4:
            return self.forward_history(window)
        return self.forward_current(window)

    def forward_window(
        self,
        window: BankWindow,
        nav: Tensor,
        ego: Tensor,
        *,
        ego_valid: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        z = self.temporal_spatial(window, ego, ego_valid=ego_valid)
        return self.planner(
            z["od"], z["ld"], nav, ego,
            window.od_valid[:, -1], window.ld_valid[:, -1],
        )

    def forward_current(self, batch) -> Dict[str, Tensor]:
        """Build one frame-record window from current observation and act."""
        enc = self.encoder.encode_current(batch)
        records = self.encoder.frame_records(
            batch,
            episode_id=batch.get("episode_id", "ep"),
            frame_id=batch.get("frame_id", 0),
            timestamp=batch.get("timestamp", 0.0),
            encoded=enc,
        )
        windows = [BankWindowBuilder(hidden=self.hidden).build((frame,)) for frame in records]
        window = BankWindow.stack(windows)
        return self.forward_window(
            window, enc["nav"], enc["ego"], ego_valid=enc.get("ego_valid")
        )

    def build_history_window(self, batch):
        """Encode a raw [B,T,...] history batch into a BankWindow.

        This is the offline bridge used by BC/WM training.  The live path uses
        ``LiveBankRunner`` and calls ``forward_window`` directly.
        """
        mode = batch.get("history_mode")
        if mode is None or not torch.is_tensor(mode) or bool((mode.reshape(-1) != 1.0).any()):
            raise ValueError("build_history_window requires raw history_mode=1")
        od = batch["od"]
        if od.dim() != 4:
            raise ValueError("history expects od [B,T,16,18]")
        b, t = int(od.shape[0]), int(od.shape[1])
        batch = dict(batch)
        # Legacy synthetic/history fixtures often use 0 for every ID.  The
        # Bank requires stable identity uniqueness within a frame; fall back to
        # slot IDs only for rows that have duplicate IDs.
        for key, mask_key, n in (
            ("od_ids", "od_mask", int(od.shape[2])),
            ("ld_ids", "ld_mask", int(batch["ld"].shape[2])),
        ):
            ids = batch.get(key)
            mask = batch.get(mask_key)
            if not (torch.is_tensor(ids) and torch.is_tensor(mask)):
                continue
            if ids.dim() != 3 or ids.shape[0] != b or ids.shape[-1] != n:
                continue
            cleaned = ids.clone()
            for bi in range(b):
                for ti in range(t):
                    m = mask[bi, ti]
                    if m.dim() > 1:
                        m = m.any(dim=-1)
                    valid = m > 0.5
                    row = cleaned[bi, ti]
                    vals = row[valid]
                    if int(valid.sum()) > 1 and len(set(int(x) for x in vals.tolist())) < int(valid.sum()):
                        cleaned[bi, ti] = torch.arange(n, dtype=ids.dtype, device=ids.device)
            batch[key] = cleaned
        pose = batch.get("ego_pose", batch.get("pose"))
        if pose is None:
            raise ValueError("raw history requires ego_pose/pose")
        if pose.dim() != 3 or pose.shape[0] != b or pose.shape[1] != t:
            raise ValueError("history pose must be [B,T,3]")
        metadata = {}
        defaults = {"episode_id": "ep", "frame_id": 0, "timestamp": 0.0}
        for key in ("episode_id", "frame_id", "timestamp"):
            value = batch.get(key)
            if torch.is_tensor(value) and value.dim() == 1 and value.numel() == b:
                metadata[key] = value
            elif isinstance(value, (list, tuple)) and len(value) == b:
                metadata[key] = value
            else:
                metadata[key] = batch.get(key, defaults[key])
        history_keys = {
            "od", "od_mask", "ld", "ld_mask", "nav", "nav_mask", "ego", "ego_mask",
            "hist_valid", "od_ids", "ld_ids", "od_anchor", "od_taxonomy", "od_age",
            "ld_road_ids", "ld_anchor", "ld_chunk_s", "nav_road_mask", "nav_road_ids",
            "nav_route_id", "pose", "ego_pose",
        }
        records_by_t = []
        last_enc = None
        for i in range(t):
            item = {}
            for key, value in batch.items():
                if (
                    key in history_keys
                    and torch.is_tensor(value)
                    and value.dim() > 1
                    and value.shape[0] == b
                ):
                    item[key] = value[:, i]
                else:
                    item[key] = value
            enc = self.encoder.encode_current(item)
            recs = self.encoder.frame_records(
                item,
                episode_id=metadata["episode_id"],
                frame_id=metadata["frame_id"],
                timestamp=metadata["timestamp"],
                ego_pose=pose[:, i],
                encoded=enc,
            )
            records_by_t.append(recs)
            last_enc = enc
        windows = [
            BankWindowBuilder(hidden=self.hidden).build(tuple(records[i] for records in records_by_t))
            for i in range(b)
        ]
        window = BankWindow.stack(windows)
        if "hist_valid" in batch:
            hist_valid = batch["hist_valid"]
            if torch.is_tensor(hist_valid):
                hist_valid = hist_valid.detach().to(dtype=torch.bool)
                if hist_valid.dim() != 2 or hist_valid.shape[0] != b:
                    raise ValueError("hist_valid must be [B,T]")
                n_hist = min(int(hist_valid.shape[1]), window.frame_valid.shape[1])
                if n_hist > 0:
                    fv = window.frame_valid.clone()
                    fv[:, -n_hist:] = fv[:, -n_hist:] & hist_valid[:, -n_hist:]
                    window.frame_valid = fv
        assert last_enc is not None
        return window, last_enc

    def _can_use_fast_history(self, batch) -> bool:
        """Whether the batch can use the tensor-only window builder.

        The fast builder is raw-history only: it aligns every historical frame
        to the current frame's stable-ID slot order, matching
        ``BankWindowBuilder``.  It requires a 6-frame history, explicit IDs and
        a history_mode=1 marker; malformed/legacy batches fall back.
        """
        mode = batch.get("history_mode")
        if mode is None or not torch.is_tensor(mode):
            return False
        mode = mode.reshape(-1)
        if mode.numel() != int(batch["od"].shape[0]):
            return False
        if bool((mode != 1.0).any()):
            return False
        if int(batch["od"].shape[1]) != 6:
            return False
        if int(batch["ld"].shape[0]) != int(batch["od"].shape[0]) or int(batch["ld"].shape[1]) != 6:
            return False
        pose = batch.get("ego_pose", batch.get("pose"))
        if pose is not None and (pose.dim() != 3 or pose.shape[0] != int(batch["od"].shape[0]) or pose.shape[1] != 6):
            return False
        for ids_key, mask_key, n in (
            ("od_ids", "od_mask", int(batch["od"].shape[2])),
            ("ld_ids", "ld_mask", int(batch["ld"].shape[2])),
        ):
            ids = batch.get(ids_key)
            mask = batch.get(mask_key)
            if not (torch.is_tensor(ids) and torch.is_tensor(mask)):
                return False
            if ids.dim() != 3 or ids.shape[0] != int(batch["od"].shape[0]):
                return False
            if ids.shape[1] != 6 or ids.shape[2] != n:
                return False
        return True

    @staticmethod
    def _sanitize_ids(ids: Tensor, mask: Tensor) -> Tensor:
        """Vectorized equivalent of the duplicate-ID fallback.

        A frame with duplicate valid stable IDs is replaced by slot IDs, just
        like the record-path loop in ``build_history_window``.
        """
        valid = mask
        if valid.dim() == 4:
            valid = valid.any(dim=-1)
        valid = valid.bool()
        ids = torch.where(valid, ids.long(), torch.full_like(ids.long(), -1))
        sentinel = torch.full_like(ids, -(2**62))
        ordered, order = torch.where(valid, ids, sentinel).sort(dim=-1)
        ordered_valid = valid.gather(-1, order)
        duplicate = (
            (ordered[..., 1:] == ordered[..., :-1])
            & ordered_valid[..., 1:]
            & ordered_valid[..., :-1]
        )
        bad = duplicate.any(dim=-1)[..., None]
        if ids.dim() == 2:
            slots = torch.arange(ids.shape[-1], dtype=ids.dtype, device=ids.device).view(1, -1)
        else:
            slots = torch.arange(ids.shape[-1], dtype=ids.dtype, device=ids.device).view(1, 1, -1)
        return torch.where(bad, slots, ids)

    @staticmethod
    def _align_history(
        content: Tensor,
        geometry: Tensor,
        type_id: Tensor,
        valid: Tensor,
        ids: Tensor,
        current_ids: Tensor,
        current_valid: Tensor,
        current_type: Tensor,
    ):
        """Align [B,T,S,...] history to current-frame slot order by stable ID."""
        b, t, old_slots = ids.shape
        new_slots = current_ids.shape[1]
        out_content = content.new_zeros((b, t, new_slots, *content.shape[3:]))
        out_geometry = geometry.new_zeros((b, t, new_slots, *geometry.shape[3:]))
        out_type = current_type.new_zeros((b, t, new_slots))
        out_valid = torch.zeros((b, t, new_slots), dtype=torch.bool, device=ids.device)
        out_ids = torch.full((b, t, new_slots), -1, dtype=torch.long, device=ids.device)
        for ti in range(t):
            eq = current_ids[:, :, None] == ids[:, ti, None, :]
            match = eq & valid[:, ti, None, :] & current_valid[:, :, None]
            has = match.any(dim=-1)
            idx = match.to(torch.int64).argmax(dim=-1)
            gather_dim = 1
            c = content[:, ti].gather(gather_dim, idx[..., None].expand(-1, -1, content.shape[-1]))
            g = geometry[:, ti].gather(gather_dim, idx[..., None].expand(-1, -1, geometry.shape[-1]))
            out_content[:, ti] = torch.where(has[..., None], c, torch.zeros_like(c))
            out_geometry[:, ti] = torch.where(has[..., None], g, torch.zeros_like(g))
            out_type[:, ti] = torch.where(has, current_type, torch.zeros_like(current_type))
            out_valid[:, ti] = has
            # The record builder keeps the current stable ID even when the
            # history frame has no matching entity; ``valid`` carries the
            # missing-history signal.
            out_ids[:, ti] = torch.where(current_valid, current_ids, torch.full_like(current_ids, -1))
        return out_content, out_geometry, out_type, out_valid, out_ids

    def build_history_window_fast(self, batch):
        """Build a 6-frame ``BankWindow`` with one batched encoder call.

        Historical entity slots are reindexed to the current frame's stable-ID
        order.  This path is raw-history only; the record path remains for
        malformed/legacy batches.
        """
        if not self._can_use_fast_history(batch):
            mode = batch.get("history_mode")
            if mode is None or not torch.is_tensor(mode) or bool((mode.reshape(-1) != 1.0).any()):
                raise ValueError("fast history requires raw history_mode=1")
            return self.build_history_window(batch)
        od = batch["od"]
        b, t = int(od.shape[0]), int(od.shape[1])
        od_ids = self._sanitize_ids(batch["od_ids"], batch["od_mask"])
        ld_ids = self._sanitize_ids(batch["ld_ids"], batch["ld_mask"])
        frame_keys = (
            "od", "od_mask", "ld", "ld_mask", "nav", "nav_mask", "ego", "ego_mask",
            "od_anchor", "od_taxonomy", "ld_taxonomy",
        )
        flat = {}
        for key in frame_keys:
            value = batch.get(key)
            if torch.is_tensor(value) and value.dim() >= 2 and value.shape[0] == b and value.shape[1] == t:
                flat[key] = value.reshape(b * t, *value.shape[2:])
        flat["od_ids"] = od_ids.reshape(b * t, -1)
        flat["ld_ids"] = ld_ids.reshape(b * t, -1)
        enc = self.encoder.encode_current(flat)
        def unflat(value: Tensor) -> Tensor:
            return value.reshape(b, t, *value.shape[1:])
        od_content = unflat(enc["od_content"])
        ld_content = unflat(enc["ld_content"])
        od_geometry = unflat(enc["od_geometry"])
        ld_geometry = unflat(enc["ld_geometry"])
        od_type = unflat(enc["od_type"])
        ld_type = unflat(enc["ld_type"])
        od_valid = unflat(enc["od_valid"]).bool()
        ld_valid = unflat(enc["ld_valid"]).bool()
        od_stable = unflat(enc["od_stable_id"])
        ld_stable = unflat(enc["ld_stable_id"])
        cur_od_ids = od_stable[:, -1]
        cur_ld_ids = ld_stable[:, -1]
        cur_od_valid = od_valid[:, -1]
        cur_ld_valid = ld_valid[:, -1]
        cur_od_type = od_type[:, -1]
        cur_ld_type = ld_type[:, -1]
        (
            od_content, od_geometry, od_type, od_valid, od_stable,
        ) = self._align_history(
            od_content, od_geometry, od_type, od_valid, od_stable,
            cur_od_ids, cur_od_valid, cur_od_type,
        )
        (
            ld_content, ld_geometry, ld_type, ld_valid, ld_stable,
        ) = self._align_history(
            ld_content, ld_geometry, ld_type, ld_valid, ld_stable,
            cur_ld_ids, cur_ld_valid, cur_ld_type,
        )
        pose = batch.get("ego_pose", batch.get("pose"))
        if pose is None:
            raise ValueError("raw history requires ego_pose/pose")
        pose = pose.to(device=od.device, dtype=enc["od_content"].dtype)
        hist_valid = batch.get("hist_valid")
        if hist_valid is None:
            frame_valid = torch.ones((b, t), dtype=torch.bool, device=od.device)
        else:
            if hist_valid.dim() != 2 or tuple(hist_valid.shape) != (b, t):
                raise ValueError("hist_valid must be [B,T]")
            frame_valid = hist_valid.to(device=od.device).bool()
        source = torch.zeros((b, t), dtype=torch.long, device=od.device)
        depth = torch.zeros((b, t), dtype=torch.long, device=od.device)
        window = BankWindow(
            od_content=od_content, od_geometry=od_geometry, od_type=od_type,
            od_valid=od_valid, od_stable_id=od_stable,
            ld_content=ld_content, ld_geometry=ld_geometry, ld_type=ld_type,
            ld_valid=ld_valid, ld_stable_id=ld_stable,
            ego_pose=pose, source=source, rollout_depth=depth, frame_valid=frame_valid,
        )
        last_enc = {
            key: enc[key].reshape(b, t, *enc[key].shape[1:])[:, -1]
            for key in enc
        }
        return window, last_enc

    def prepare_current_encoding(self, enc: Mapping[str, Tensor]) -> Dict[str, Tensor]:
        """Mask invalid entities and sanitize IDs before FIFO append."""
        out = dict(enc)
        od_valid = out["od_valid"].bool()
        ld_valid = out["ld_valid"].bool()
        out["od_stable_id"] = torch.where(
            od_valid,
            self._sanitize_ids(out["od_stable_id"], od_valid),
            torch.full_like(out["od_stable_id"], -1),
        )
        out["ld_stable_id"] = torch.where(
            ld_valid,
            self._sanitize_ids(out["ld_stable_id"], ld_valid),
            torch.full_like(out["ld_stable_id"], -1),
        )
        out["od_content"] = torch.where(od_valid[..., None], out["od_content"], torch.zeros_like(out["od_content"]))
        out["ld_content"] = torch.where(ld_valid[..., None], out["ld_content"], torch.zeros_like(out["ld_content"]))
        out["od_geometry"] = torch.where(od_valid[..., None], out["od_geometry"], torch.zeros_like(out["od_geometry"]))
        out["ld_geometry"] = torch.where(ld_valid[..., None], out["ld_geometry"], torch.zeros_like(out["ld_geometry"]))
        out["od_type"] = torch.where(od_valid, out["od_type"], torch.zeros_like(out["od_type"]))
        out["ld_type"] = torch.where(ld_valid, out["ld_type"], torch.zeros_like(out["ld_type"]))
        return out

    def fifo_append_safe(self, window: BankWindow, enc: Mapping[str, Tensor]) -> bool:
        """Check whether every current entity exists in the previous frame.

        ``BankWindow.advance`` shifts an already entity-filtered view.  If a
        current entity was absent in the previous frame but exists in older
        history, the filtered FIFO would lose that older history; the trainer
        must rebuild instead.  A disappearing entity is safe to shift.
        """
        checks = (
            (window.od_stable_id[:, -1], window.od_valid[:, -1], enc["od_stable_id"], enc["od_valid"]),
            (window.ld_stable_id[:, -1], window.ld_valid[:, -1], enc["ld_stable_id"], enc["ld_valid"]),
        )
        for prev_ids, prev_valid, cur_ids, cur_valid in checks:
            cur_ids = self._sanitize_ids(cur_ids, cur_valid)
            cur_valid = cur_valid.bool()
            hit = (cur_ids[:, :, None] == prev_ids[:, None, :]) & prev_valid[:, None, :]
            missing = cur_valid & ~hit.any(dim=-1)
            if bool(missing.any()):
                return False
        return True

    def align_window_to_current(self, window: BankWindow, enc: Mapping[str, Tensor]) -> BankWindow:
        """Reindex a BankWindow's history to ``enc``'s current slot order."""
        od_ids = self._sanitize_ids(enc["od_stable_id"], enc["od_valid"])
        ld_ids = self._sanitize_ids(enc["ld_stable_id"], enc["ld_valid"])
        od = self._align_history(
            window.od_content, window.od_geometry, window.od_type,
            window.od_valid, window.od_stable_id,
            od_ids, enc["od_valid"].bool(), enc["od_type"],
        )
        ld = self._align_history(
            window.ld_content, window.ld_geometry, window.ld_type,
            window.ld_valid, window.ld_stable_id,
            ld_ids, enc["ld_valid"].bool(), enc["ld_type"],
        )
        return BankWindow(
            od_content=od[0], od_geometry=od[1], od_type=od[2],
            od_valid=od[3], od_stable_id=od[4],
            ld_content=ld[0], ld_geometry=ld[1], ld_type=ld[2],
            ld_valid=ld[3], ld_stable_id=ld[4],
            ego_pose=window.ego_pose, source=window.source,
            rollout_depth=window.rollout_depth, frame_valid=window.frame_valid,
        )

    def forward_history(self, batch) -> Dict[str, Tensor]:
        """Policy forward from a raw [B,T,...] history batch."""
        if self._can_use_fast_history(batch):
            window, last_enc = self.build_history_window_fast(batch)
        else:
            mode = batch.get("history_mode")
            if mode is None or not torch.is_tensor(mode) or bool((mode.reshape(-1) != 1.0).any()):
                raise ValueError("forward_history requires raw history_mode=1")
            window, last_enc = self.build_history_window(batch)
        return self.forward_window(
            window, last_enc["nav"], last_enc["ego"], ego_valid=last_enc.get("ego_valid")
        )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------
    def imagine(
        self,
        window: BankWindow,
        nav: Tensor,
        ego: Tensor,
        *,
        actions: Tensor | None = None,
        action_provider=None,
        horizon: int | None = None,
        ego_features: Tensor | None = None,
        ego_valid: Tensor | None = None,
        imagine_bank: ImagineMemoryBank | None = None,
        detach: bool = True,
    ) -> list[Dict[str, Tensor]]:
        """Policy-conditioned 6-transition imagination.

        ``ego`` is the current ego hidden state; ``ego_features`` is the raw
        12D ego vector needed to update the shared ego encoder.  It is required
        so that policy ego state and pose remain on the same trajectory.
        """
        if horizon is None:
            horizon = self.rollout_horizon
        if ego_features is None:
            raise ValueError("ego_features is required for ego recurrence; pass the raw 12D ego vector")
        return imagine_rollout(
            window,
            nav,
            ego,
            temporal_spatial=self.temporal_spatial,
            planner=self.planner,
            world_model=self.world_model,
            actions=actions,
            action_provider=action_provider,
            horizon=int(horizon),
            action_to_requested=self._to_requested,
            ego_features=ego_features,
            ego_step=self._ego_step,
            ego_encoder=self.encoder.encode_ego,
            ego_valid=ego_valid,
            imagine_bank=imagine_bank,
            detach=detach,
        )

    def rollout(self, *args, **kwargs):
        """Contract alias for the model-level imagine rollout."""
        return self.imagine(*args, **kwargs)

    def imagine_from_real_bank(
        self,
        real_bank: RealMemoryBank,
        nav: Tensor,
        ego: Tensor,
        **kwargs,
    ) -> list[Dict[str, Tensor]]:
        """Run imagine from a RealMemoryBank snapshot (copy-on-write)."""
        imagine_bank = real_bank.snapshot()
        window = real_bank.window()
        return self.imagine(window, nav, ego, imagine_bank=imagine_bank, **kwargs)


# Public domain alias for the bank-native top-level model.
Model = DriveModel

__all__ = ["DriveModel", "Model", "count_parameters"]
