"""Plan rollout and 4D-action -> trajectory helpers for P1 evaluation."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from env.tracking import ego_to_world, interpolate
from net.action_space import (
    DEFAULT_BOUNDS,
    ActionBounds,
    ActionTrajectory,
    action_to_trajectory,
)
from net.model import ACTION_HIGH, ACTION_LOW, StructuredPlanner

#: Public alias for the PSTL result object (also supports dict-style access).
Trajectory = ActionTrajectory


def denorm_action(a: torch.Tensor) -> torch.Tensor:
    low = ACTION_LOW.to(a.device)
    high = ACTION_HIGH.to(a.device)
    return low + a.clamp(0.0, 1.0) * (high - low)


def denorm_action_with_bounds(a: torch.Tensor, bounds: ActionBounds) -> torch.Tensor:
    """Denormalize with an explicit static box (legacy tensors stay unchanged).

    ``torch`` inputs follow :func:`denorm_action` exactly (clamp to ``[0,1]``);
    numpy inputs use the same static clamp.  Callers evaluating old checkpoints
    must pass :data:`net.action_space.LEGACY_BOUNDS`.
    """
    if not isinstance(a, torch.Tensor):
        return bounds.denormalize(np.asarray(a, dtype=np.float64), clip=True)
    low = torch.as_tensor(np.array(bounds.low, dtype=np.float32), device=a.device)
    high = torch.as_tensor(np.array(bounds.high, dtype=np.float32), device=a.device)
    return low + a.clamp(0.0, 1.0) * (high - low)


def segment_trajectory(
    action: np.ndarray,
    v0: float,
    *,
    dt: float = 0.5,
    hz: int = 10,
    bounds: ActionBounds = DEFAULT_BOUNDS,
) -> ActionTrajectory:
    """PSTL trajectory for one 0.5 s action; ``v0`` is the current speed."""
    return action_to_trajectory(action, v0, dt=dt, hz=hz, bounds=bounds)


def first_segment_trajectory(
    action: np.ndarray,
    v0: float,
    *,
    dt: float = 0.5,
    hz: int = 10,
    bounds: ActionBounds = DEFAULT_BOUNDS,
) -> ActionTrajectory:
    """Alias for :func:`segment_trajectory` (one action == one first segment)."""
    return segment_trajectory(action, v0, dt=dt, hz=hz, bounds=bounds)


def batch_from_obs(hist: Dict[str, np.ndarray], device: str = "cpu") -> Dict[str, torch.Tensor]:
    out = {}
    for k in ("od", "od_mask", "ld", "ld_mask", "nav", "nav_mask", "ego", "ego_mask", "hist_valid"):
        out[k] = torch.from_numpy(np.asarray(hist[k], dtype=np.float32)).unsqueeze(0).to(device)
    return out


@torch.no_grad()
def plan_actions(model: StructuredPlanner, batch: Dict[str, torch.Tensor], steps: int = 6) -> np.ndarray:
    """Return [steps,4] denormalized actions. No WM: same context is queried repeatedly."""
    model.eval()
    acts: List[np.ndarray] = []
    for _ in range(int(steps)):
        out = model(batch)
        a = denorm_action(out["action_mu"])[0].detach().cpu().numpy()
        acts.append(a)
    return np.stack(acts, axis=0)


def action_to_segment(a: np.ndarray) -> Tuple[float, float]:
    dx, dy, dpsi, _v = [float(x) for x in a[:4]]
    ds = float(math.hypot(dx, dy))
    return ds, float(dpsi)


def segment_points(action: np.ndarray, n: int = 5) -> np.ndarray:
    """Cubic Hermite through (0,0,0) -> (dx,dy,dpsi), n samples in (0,1]."""
    dx, dy, dpsi = [float(x) for x in action[:3]]
    ds = float(math.hypot(dx, dy))
    # start/end tangent directions: 0 and dpsi; magnitudes keep the path well-conditioned
    m = max(0.5, ds)
    m0 = np.array([m, 0.0], dtype=np.float64)
    m1 = np.array([m * math.cos(dpsi), m * math.sin(dpsi)], dtype=np.float64)
    p0 = np.zeros(2, dtype=np.float64)
    p1 = np.array([dx, dy], dtype=np.float64)
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(1, n + 1):
        t = i / float(n)
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        p = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
        # derivative
        dh00 = 6 * t**2 - 6 * t
        dh10 = 3 * t**2 - 4 * t + 1
        dh01 = -6 * t**2 + 6 * t
        dh11 = 3 * t**2 - 2 * t
        d = dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1
        th = math.atan2(float(d[1]), float(d[0])) if float(np.linalg.norm(d)) > 1e-9 else dpsi * t
        out[i - 1] = (float(p[0]), float(p[1]), float(_wrap(th)))
    # force exact endpoint
    out[-1, 0], out[-1, 1], out[-1, 2] = dx, dy, float(_wrap(dpsi))
    return out


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def compose_plan_poses(actions: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """Compose 6 x (dx,dy,dpsi) into 30 0.1 s poses in the t0 ego frame."""
    poses: List[np.ndarray] = []
    cur = np.asarray(base_pose, dtype=np.float64).copy()
    for a in actions:
        seg = segment_points(a, n=5)
        seg_w = ego_to_world(seg, cur)
        poses.append(seg_w)
        if len(seg_w):
            cur = seg_w[-1].copy()
    if not poses:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(poses, axis=0)


def first_segment_points(action: np.ndarray) -> np.ndarray:
    return segment_points(action, n=5)
