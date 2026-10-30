"""Bank-native top-level DriveModel.

The old legacy wrapper path has been removed from the
network package.  Policy, world model and rollout are all composed here from
the bank-native components.
"""

from __future__ import annotations

from typing import Dict

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

    def build_history_window(self, batch, *, zero_pose: bool | Tensor = False):
        """Encode a [B,T,...] history batch into a BankWindow.

        This is the offline bridge used by BC/WM training and legacy eval.  The
        live path uses ``LiveBankRunner`` and calls ``forward_window`` directly.
        """
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
            pose = torch.zeros((b, t, 3), dtype=od.dtype, device=od.device)
        if pose.dim() != 3 or pose.shape[0] != b or pose.shape[1] != t:
            raise ValueError("history pose must be [B,T,3]")
        if torch.is_tensor(zero_pose):
            if zero_pose.dim() != 1 or zero_pose.shape[0] != b:
                raise ValueError("zero_pose tensor must be [B]")
            pose = pose.clone()
            pose[zero_pose.to(device=pose.device, dtype=torch.bool)] = 0.0
        elif zero_pose:
            pose = torch.zeros((b, t, 3), dtype=od.dtype, device=od.device)
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

    def forward_history(self, batch, *, zero_pose: bool = False) -> Dict[str, Tensor]:
        """Policy forward from a [B,T,...] history batch."""
        if not zero_pose:
            mode = batch.get("history_mode")
            if torch.is_tensor(mode):
                mode = mode.reshape(-1)
                if mode.numel() == int(batch["od"].shape[0]):
                    zero_pose = mode < 0.5
        window, last_enc = self.build_history_window(batch, zero_pose=zero_pose)
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
