"""Live Bank runner for eval/PPO/collector environments.

The environment supplies one current observation per step.  This helper encodes
that frame, appends a real FrameRecord to a 6-frame RealMemoryBank, and returns
the latest BankWindow plus the current nav/ego latents.  It intentionally does
not use ``env.obs.stack_history``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from env.obs import (
    EGO_DIM,
    LD_DIM,
    LD_POINTS,
    LD_SLOTS,
    NAV_DIM,
    OD_DIM,
    OD_SLOTS,
)
from net.mem_bank import RealMemoryBank


def _drive(model):
    return model


def _tensor(value, dtype=torch.float32):
    return torch.from_numpy(np.asarray(value, dtype=np.float32 if dtype is torch.float32 else np.int64)).unsqueeze(0)


class LiveBankRunner:
    """Append current observations to a real Bank and expose the latest window."""

    def __init__(self, model, *, device: str = "cpu", capacity: int = 6):
        self.model = model
        self.drive = _drive(model)
        self.encoder = self.drive.encoder
        self.device = torch.device(device)
        self.capacity = int(capacity)
        if not 1 <= self.capacity <= 6:
            raise ValueError("LiveBankRunner capacity must be in 1..6")
        self.reset()

    def reset(self) -> None:
        self.bank = RealMemoryBank(self.capacity)
        self.frame_id = 0

    def _batch(self, obs: Mapping[str, Any], episode_id: str) -> Dict[str, torch.Tensor]:
        keys = {
            "od": (OD_SLOTS, OD_DIM),
            "od_mask": (OD_SLOTS,),
            "ld": (LD_SLOTS, LD_POINTS, LD_DIM),
            "ld_mask": (LD_SLOTS, LD_POINTS),
            "nav": (NAV_DIM,),
            "ego": (EGO_DIM,),
            "od_ids": (OD_SLOTS,),
            "ld_ids": (LD_SLOTS,),
            "pose": (3,),
        }
        batch: Dict[str, torch.Tensor] = {}
        for key, shape in keys.items():
            arr = np.asarray(obs[key], dtype=np.int64 if key in ("od_ids", "ld_ids") else np.float32)
            batch[key] = torch.from_numpy(arr.reshape(1, *shape)).to(self.device)
        if "nav_mask" in obs:
            batch["nav_mask"] = _tensor(obs["nav_mask"]).to(self.device)
        if "ego_mask" in obs:
            batch["ego_mask"] = _tensor(obs["ego_mask"]).to(self.device)
        return batch

    def push(self, obs: Mapping[str, Any], *, episode_id: str = "live") -> Tuple[Any, torch.Tensor, torch.Tensor]:
        batch = self._batch(obs, episode_id)
        with torch.no_grad():
            enc = self.encoder.encode_current(batch)
        record = self.encoder.frame_records(
            batch,
            episode_id=episode_id,
            frame_id=self.frame_id,
            timestamp=float(obs.get("meta", {}).get("timestamp", self.frame_id * 0.5)),
            encoded=enc,
        )[0]
        self.bank.append(record, detach=True)
        self.frame_id += 1
        window = self.bank.window().to(self.device)
        return window, enc["nav"], enc["ego"]
