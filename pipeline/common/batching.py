"""Batch helpers shared by the training entry points."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
from torch import Tensor

INPUT_KEYS: Tuple[str, ...] = (
    "od",
    "od_mask",
    "ld",
    "ld_mask",
    "nav",
    "nav_mask",
    "ego",
    "ego_mask",
    "hist_valid",
    "od_ids",
    "ld_ids",
    "pose",
)


def batch_from_history(hist: Mapping[str, Any], device: str = "cpu") -> Dict[str, Tensor]:
    """Batch-first float32 tensors for one schema-v2 history window."""
    batch: Dict[str, Tensor] = {}
    for key in INPUT_KEYS:
        dtype = np.int64 if key in ("od_ids", "ld_ids") else np.float32
        arr = np.asarray(hist[key], dtype=dtype)
        batch[key] = torch.from_numpy(arr).unsqueeze(0).to(device)
    batch["history_mode"] = torch.ones((1,), dtype=torch.float32, device=device)
    return batch
