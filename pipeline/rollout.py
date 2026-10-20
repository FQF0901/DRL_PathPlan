"""Plan-action helpers for the canonical PSTL planner.

The planner's 4D action is ``(dx, dy, dpsi, v_next)`` and is executed by
``net.action_space.action_to_trajectory``.  This module only contains the
current API; the previous 3D Hermite implementation is available from git
history (tag ``pre-consolidation``), not as a second live path.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from net.action_space import (
    CLEAN_BOUNDS,
    ActionBounds,
    ActionTrajectory,
    action_to_trajectory,
)
from net.model import StructuredPlanner

#: Public alias for the PSTL result object (also supports dict-style access).
Trajectory = ActionTrajectory


def denorm_action(a: torch.Tensor) -> torch.Tensor:
    """Map a normalized action tensor to the canonical physical action box."""
    return denorm_action_with_bounds(a, CLEAN_BOUNDS)


def denorm_action_with_bounds(a: torch.Tensor | np.ndarray, bounds: ActionBounds) -> torch.Tensor | np.ndarray:
    """Denormalize with an explicit bounds object (clamped to the box)."""
    if isinstance(a, torch.Tensor):
        low = torch.as_tensor(np.asarray(bounds.low, dtype=np.float32), device=a.device)
        high = torch.as_tensor(np.asarray(bounds.high, dtype=np.float32), device=a.device)
        return low + a.clamp(0.0, 1.0) * (high - low)
    return bounds.denormalize(np.asarray(a, dtype=np.float32), clip=True)


def segment_trajectory(
    action: np.ndarray,
    v0: float,
    *,
    dt: float = 0.5,
    hz: int = 10,
    bounds: ActionBounds = CLEAN_BOUNDS,
) -> ActionTrajectory:
    """PSTL trajectory for one 0.5 s action; ``v0`` is the current speed."""
    return action_to_trajectory(action, v0, dt=dt, hz=hz, bounds=bounds)


def first_segment_trajectory(
    action: np.ndarray,
    v0: float,
    *,
    dt: float = 0.5,
    hz: int = 10,
    bounds: ActionBounds = CLEAN_BOUNDS,
) -> ActionTrajectory:
    """Alias for the first (and only executed) 0.5 s segment."""
    return segment_trajectory(action, v0, dt=dt, hz=hz, bounds=bounds)


def batch_from_obs(hist: Dict[str, np.ndarray], device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Convert one observation history into a batch-first model input."""
    keys = (
        "od", "od_mask", "ld", "ld_mask", "nav", "nav_mask",
        "ego", "ego_mask", "hist_valid",
    )
    return {
        key: torch.from_numpy(np.asarray(hist[key], dtype=np.float32)).unsqueeze(0).to(device)
        for key in keys
    }


@torch.no_grad()
def plan_actions(
    model: StructuredPlanner,
    batch: Dict[str, torch.Tensor],
    steps: int = 6,
    *,
    bounds: ActionBounds = CLEAN_BOUNDS,
) -> np.ndarray:
    """Return ``[steps,4]`` physical actions.

    P1 has no world model: all queries use the same context, so this is a
    diagnostic preview only.  Only ``actions[0]`` is executed by the evaluator.
    """
    model.eval()
    acts: List[np.ndarray] = []
    for _ in range(int(steps)):
        out = model(batch)
        acts.append(denorm_action_with_bounds(out["action_mu"], bounds)[0].detach().cpu().numpy())
    return np.stack(acts, axis=0)
