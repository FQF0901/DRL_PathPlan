"""Single source of truth for executed-segment bookkeeping.

The collector, evaluator and PPO rollout must agree on three things:

* the observation at decision time uses the **pre-step** speed/heading;
* the next observation's previous action is the **realized** 4D endpoint in
  the pre-step ego frame, not the planner's requested/commanded action;
* a segment that is cut short carries its **actual** elapsed time.

Keeping these rules in one small dependency-free module prevents the
collector/eval/PPO mirror implementations from drifting again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    value = float(angle)
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def relative_action(
    p0: Any,
    h0: float,
    v0: float,
    p1: Any,
    h1: float,
    v1: float,
) -> np.ndarray:
    """Return the measured 4D action from segment start to segment end.

    ``(dx, dy)`` are expressed in the pre-step ego frame (heading ``h0``);
    ``dpsi`` is the measured heading change and ``v_next`` is the measured
    end speed.  This is the definition used by the collector for replay rows.
    """
    start = np.asarray(p0, dtype=np.float64).reshape(-1)
    end = np.asarray(p1, dtype=np.float64).reshape(-1)
    if start.size < 2 or end.size < 2:
        raise ValueError("relative_action needs 2-D or 3-D positions")
    dxw = float(end[0] - start[0])
    dyw = float(end[1] - start[1])
    c0 = math.cos(float(h0))
    s0 = math.sin(float(h0))
    dx = c0 * dxw + s0 * dyw
    dy = -s0 * dxw + c0 * dyw
    return np.array([dx, dy, wrap_angle(float(h1) - float(h0)), float(v1)], dtype=np.float64)


def segment_dt(substeps: int, substep_dt: float = 0.1) -> float:
    """Actual duration of a possibly truncated 5-substep segment."""
    n = int(substeps)
    if n < 0:
        raise ValueError("substeps must be non-negative")
    return float(n) * float(substep_dt)


@dataclass(frozen=True)
class DecisionState:
    """Previous-observation state shared by collect/eval/PPO."""

    prev_action: np.ndarray
    prev_speed: Optional[float]
    prev_heading: Optional[float]

    @classmethod
    def initial(cls, action_dim: int = 4) -> "DecisionState":
        return cls(
            prev_action=np.zeros(int(action_dim), dtype=np.float64),
            prev_speed=None,
            prev_heading=None,
        )

    def next(
        self,
        *,
        start_speed: float,
        start_heading: float,
        realized_action: Any,
    ) -> "DecisionState":
        action = np.asarray(realized_action, dtype=np.float64).reshape(-1).copy()
        if action.shape != self.prev_action.shape:
            raise ValueError(
                f"realized_action shape {action.shape} != {self.prev_action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("realized_action contains NaN/Inf")
        return DecisionState(
            prev_action=action,
            prev_speed=float(start_speed),
            prev_heading=float(start_heading),
        )

    def observation_kwargs(self, dt: float) -> dict[str, Any]:
        return {
            "prev_action": self.prev_action,
            "prev_speed": self.prev_speed,
            "prev_heading": self.prev_heading,
            "dt": float(dt),
        }
