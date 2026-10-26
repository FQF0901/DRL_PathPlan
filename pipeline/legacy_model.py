"""Compatibility adapter for legacy BC/PPO/eval call sites.

The canonical network is :class:`net.model.DriveModel`.  This adapter exposes
the old ``StructuredPlanner`` output envelope to the legacy training/eval code
while routing the actual policy through the bank-native model.  The legacy
value/risk/cost and query-supervision heads are adapter-only and are not part
of the new network contract.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn

from env.action.config import load_action_contract
from env.obs import EGO_DIM, HISTORY_FRAMES, LD_SLOTS, OD_SLOTS
from net.model import DriveModel

H = 128
HEADS = 4
SPATIAL_LAYERS = 2
ACTION_DIM = 4
RISK_DIM = 5
LANE_ROLE_CLASSES = 4


class StructuredPlanner(nn.Module):
    """Legacy API wrapper around :class:`net.model.DriveModel`."""

    def __init__(
        self,
        hidden: int = H,
        heads: int = HEADS,
        spatial_layers: int = SPATIAL_LAYERS,
        action_dim: int = ACTION_DIM,
        *,
        action_low=None,
        action_high=None,
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.spatial_layers = int(spatial_layers)
        contract = load_action_contract()
        low = contract.bounds.low if action_low is None else action_low
        high = contract.bounds.high if action_high is None else action_high
        from net.build_net import load_config
        ego = load_config().get("ego", {})
        self.drive = DriveModel(
            hidden=self.hidden,
            heads=self.heads,
            spatial_layers=self.spatial_layers,
            action_dim=int(action_dim),
            action_low=low,
            action_high=high,
            ego_dt=float(ego.get("dt", 0.5)),
            ego_wheelbase=float(ego.get("wheelbase", 2.46894)),
            ego_max_steer_rad=float(ego.get("max_steer_rad", 0.6981317007977318)),
        )
        if action_low is None and action_high is None:
            from dataclasses import asdict
            self.drive.action_config = {
                "name": contract.bounds.name,
                "low": contract.bounds.low.tolist(),
                "high": contract.bounds.high.tolist(),
                "limits": {"hard": asdict(contract.hard), "comfort": asdict(contract.comfort)},
            }

    @property
    def action_low(self):
        return self.drive.action_low

    @property
    def action_high(self):
        return self.drive.action_high

    @property
    def action_config(self):
        return self.drive.action_config

    @property
    def ego_config(self):
        return self.drive.ego_config

    def encode_entities(self, batch=None, **kwargs) -> Dict[str, Tensor]:
        batch = kwargs if batch is None else batch
        out = self.drive.encoder.encode_current(batch)
        return {
            "od": out["od_content"],
            "ld": out["ld_content"],
            "nav": out["nav"],
            "ego": out["ego"],
            "od_mask": out["od_valid"],
            "ld_mask": out["ld_valid"],
            "od_ids": out.get("od_stable_id"),
            "ld_ids": out.get("ld_stable_id"),
            "od_geometry": out["od_geometry"],
            "ld_geometry": out["ld_geometry"],
        }

    def _augment(self, out: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # No value/risk/cost heads on the bank-native contract.  Legacy
        # callers may keep the keys as zero-valued compatibility outputs; they
        # have no parameters and cannot be optimized.
        out = dict(out)
        ctx = out["latent"]
        out["value"] = torch.zeros((ctx.shape[0], 1), dtype=ctx.dtype, device=ctx.device)
        out["risk"] = torch.zeros((ctx.shape[0], RISK_DIM), dtype=ctx.dtype, device=ctx.device)
        out["cost_value"] = torch.zeros((ctx.shape[0], 1), dtype=ctx.dtype, device=ctx.device)
        return out

    def forward_window(self, window, nav, ego, **kwargs) -> Dict[str, Tensor]:
        return self._augment(self.drive.forward_window(window, nav, ego, **kwargs))

    def forward(self, batch=None, **kwargs) -> Dict[str, Tensor]:
        batch = dict(kwargs if batch is None else batch)
        # The legacy history contract did not carry IDs; the new encoder
        # requires them.  Slot IDs are a deterministic compatibility fallback.
        b = int(batch["od"].shape[0])
        t = int(batch["od"].shape[1]) if batch["od"].dim() == 4 else 1
        if "od_ids" not in batch:
            batch["od_ids"] = torch.arange(OD_SLOTS, device=batch["od"].device).view(1, 1, OD_SLOTS).expand(b, t, -1)
        if "ld_ids" not in batch:
            batch["ld_ids"] = torch.arange(LD_SLOTS, device=batch["ld"].device).view(1, 1, LD_SLOTS).expand(b, t, -1)
        if "pose" not in batch and "ego_pose" not in batch:
            batch["pose"] = torch.zeros(
                (b, t, 3), dtype=batch["ego"].dtype, device=batch["ego"].device
            )
        if batch["od"].dim() == 4:
            mode = batch.get("history_mode")
            if torch.is_tensor(mode):
                mode = mode.reshape(-1)
                if mode.numel() == batch["od"].shape[0]:
                    pose = batch.get("pose", batch.get("ego_pose"))
                    if pose is not None:
                        pose = pose.clone()
                        pose[mode < 0.5] = 0.0
                        batch["pose"] = pose
                        if "ego_pose" in batch:
                            batch["ego_pose"] = pose
            out = self.drive.forward_history(batch)
        else:
            out = self.drive.forward_current(batch)
        return self._augment(out)

    def imagine(self, *args, **kwargs):
        return self.drive.imagine(*args, **kwargs)

    def rollout(self, *args, **kwargs):
        return self.drive.rollout(*args, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "H", "HEADS", "SPATIAL_LAYERS", "StructuredPlanner", "count_parameters",
]
