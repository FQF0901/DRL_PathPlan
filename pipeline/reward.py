"""Rule-based RL reward model for real-env group-relative PPO.

Contract reference: ``docs/arch_lock.md`` §12.4.

The model is pure Python (no torch / env imports).  Each call aggregates the
events of one 0.5 s policy decision (5 env substeps):

* dense: route progress, comfort, low speed, one-shot wrong-route / stuck;
* terminal: exactly one of invalid > collision > off-road > success > timeout.

``wrong_route`` and ``stuck`` are per-step one-shot penalties and never enter
the terminal term; otherwise the terminal event would mask them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class RewardConfig:
    """Weights and thresholds of the rule-based reward model."""

    w_progress: float = 10.0
    w_success: float = 20.0
    w_collision: float = 20.0
    w_offroad: float = 15.0
    w_timeout: float = 10.0
    w_invalid: float = 20.0
    w_low_speed: float = 0.2
    w_comfort: float = 0.05
    w_wrong_route: float = 2.0
    w_stuck: float = 2.0
    adapt_coeff: float = 1.0
    low_speed_threshold: float = 2.0
    comfort_a_long: float = 2.0
    comfort_a_lat: float = 3.0
    comfort_jerk_long: float = 5.0
    wrong_route_drop: float = 0.05
    stuck_speed: float = 0.1
    stuck_window_steps: int = 20

    def __post_init__(self) -> None:
        for name in (
            "w_progress", "w_success", "w_collision", "w_offroad", "w_timeout",
            "w_invalid", "w_low_speed", "w_comfort", "w_wrong_route", "w_stuck",
            "adapt_coeff", "low_speed_threshold", "comfort_a_long",
            "comfort_a_lat", "comfort_jerk_long", "wrong_route_drop",
            "stuck_speed",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)):
                raise ValueError(f"RewardConfig.{name} must be finite, got {value!r}")
        if float(self.low_speed_threshold) <= 0.0:
            raise ValueError("RewardConfig.low_speed_threshold must be > 0")
        if int(self.stuck_window_steps) < 1:
            raise ValueError("RewardConfig.stuck_window_steps must be >= 1")
        if float(self.adapt_coeff) < 0.0:
            raise ValueError("RewardConfig.adapt_coeff must be >= 0")


def _finite(value: Any, name: str) -> float:
    """Convert ``value`` to float and fail fast on non-finite input."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


@dataclass
class RewardTracker:
    """Per-episode state for the one-shot wrong-route and stuck events."""

    config: RewardConfig = field(default_factory=RewardConfig)
    _wrong_route_seen: bool = field(default=False, init=False, repr=False)
    _stuck_run: int = field(default=0, init=False, repr=False)
    _stuck_seen: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all episode state."""
        self._wrong_route_seen = False
        self._stuck_run = 0
        self._stuck_seen = False

    def update(self, rc0: float, rc1: float, v_end: float) -> Dict[str, bool]:
        """Update one policy step and return the one-shot event flags.

        ``wrong_route`` triggers on the first segment whose route completion
        drops by more than ``config.wrong_route_drop``.

        ``stuck`` triggers once when ``config.stuck_window_steps`` consecutive
        policy steps have ``v_end < config.stuck_speed`` and no route progress.
        The consecutive run resets on any progress or on speed at/above the
        threshold.
        """
        rc0 = _finite(rc0, "rc0")
        rc1 = _finite(rc1, "rc1")
        v_end = _finite(v_end, "v_end")

        delta = rc1 - rc0

        wrong_route_event = False
        if (
            not self._wrong_route_seen
            and delta < -self.config.wrong_route_drop
        ):
            self._wrong_route_seen = True
            wrong_route_event = True

        stuck_event = False
        if v_end >= self.config.stuck_speed or delta > 0.0:
            self._stuck_run = 0
        else:
            self._stuck_run += 1
            if (
                not self._stuck_seen
                and self._stuck_run >= self.config.stuck_window_steps
            ):
                self._stuck_seen = True
                stuck_event = True

        return {
            "wrong_route_event": wrong_route_event,
            "stuck_event": stuck_event,
        }


def compute_step_reward(
    rc0: float,
    rc1: float,
    v_end: float,
    metrics: Optional[Mapping[str, float]] = None,
    *,
    collision: bool = False,
    off_road: bool = False,
    success: bool = False,
    timeout: bool = False,
    invalid_action: bool = False,
    wrong_route_event: bool = False,
    stuck_event: bool = False,
    config: RewardConfig = RewardConfig(),
) -> Dict[str, Any]:
    """Compute dense + terminal reward for one policy step.

    ``metrics`` is the segment-level ``ActionTrajectory.metrics`` mapping.  A
    missing key means zero; a present non-finite value raises ``ValueError``.

    Terminal priority is ``invalid_action > collision > off_road > success >
    timeout``.  ``success`` cannot override collision or off-road because those
    branches are evaluated first.  ``done`` is true for every terminal event.
    """
    rc0 = _finite(rc0, "rc0")
    rc1 = _finite(rc1, "rc1")
    v_end = _finite(v_end, "v_end")

    metric_map = metrics or {}
    max_abs_a_long = _finite(metric_map.get("max_abs_a_long", 0.0), "metrics['max_abs_a_long']")
    max_abs_a_lat = _finite(metric_map.get("max_abs_a_lat", 0.0), "metrics['max_abs_a_lat']")
    max_abs_jerk_long = _finite(metric_map.get("max_abs_jerk_long", 0.0), "metrics['max_abs_jerk_long']")

    if invalid_action:
        # No segment was executed: the contract defines r_t = 0 so the
        # terminal invalid penalty is the only reward on this decision.
        progress = comfort = low_speed = wrong_route = stuck = 0.0
    else:
        progress = config.w_progress * (rc1 - rc0)
        comfort = -config.w_comfort * (
            max(0.0, max_abs_a_long - config.comfort_a_long)
            + max(0.0, max_abs_a_lat - config.comfort_a_lat)
            + max(0.0, max_abs_jerk_long - config.comfort_jerk_long)
        )
        low_speed = -config.w_low_speed * max(
            0.0, 1.0 - v_end / config.low_speed_threshold
        )
        wrong_route = -config.w_wrong_route if wrong_route_event else 0.0
        stuck = -config.w_stuck if stuck_event else 0.0

    dense_reward = progress + comfort + low_speed + wrong_route + stuck

    if invalid_action:
        terminal_event = "invalid_action"
        terminal_reward = -config.w_invalid
    elif collision:
        terminal_event = "collision"
        terminal_reward = -config.adapt_coeff * config.w_collision
    elif off_road:
        terminal_event = "off_road"
        terminal_reward = -config.adapt_coeff * config.w_offroad
    elif success:
        terminal_event = "success"
        terminal_reward = config.w_success
    elif timeout:
        terminal_event = "timeout"
        rc = min(max(rc1, 0.0), 1.0)
        terminal_reward = -config.w_timeout * (1.0 - rc)
    else:
        terminal_event = "none"
        terminal_reward = 0.0

    components = {
        "progress": float(progress),
        "comfort": float(comfort),
        "low_speed": float(low_speed),
        "wrong_route": float(wrong_route),
        "stuck": float(stuck),
        "terminal": float(terminal_reward),
    }

    return {
        "dense_reward": float(dense_reward),
        "terminal_reward": float(terminal_reward),
        "reward": float(dense_reward + terminal_reward),
        "terminal_event": terminal_event,
        "done": terminal_event != "none",
        "components": components,
    }


#: Event keys used by MetaDrive ``info`` and the project evaluator.
COLLISION_KEYS = (
    "crash", "crash_vehicle", "crash_object", "crash_building",
    "crash_human", "crash_sidewalk",
)
OFFROAD_KEYS = ("out_of_road", "off_road")
SUCCESS_KEYS = ("arrive_dest",)
TIMEOUT_KEYS = ("max_step", "timeout")


def info_flag(info: Mapping[str, Any], *keys: str) -> bool:
    """Return True if any of ``keys`` is a truthy scalar in ``info``.

    This is the shared event predicate for eval, RL rollout and joint rollback.
    It intentionally ignores non-scalar values.
    """
    for key in keys:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, bool):
            if value:
                return True
            continue
        try:
            if float(value) > 0.5:
                return True
        except (TypeError, ValueError):
            continue
    return False


@dataclass(frozen=True)
class StepEvents:
    """Aggregated events of one 0.5 s policy decision (5 env substeps)."""

    collision: bool
    off_road: bool
    arrive_dest: bool
    timeout: bool
    invalid_action: bool
    rc0: float
    rc1: float
    v_end: float
    metrics: Mapping[str, float]


def aggregate_substep_events(
    infos: Any,
    *,
    rc0: float,
    rc1: float,
    v_end: float,
    timeout: bool = False,
    invalid_action: bool = False,
    metrics: Optional[Mapping[str, float]] = None,
) -> StepEvents:
    """Aggregate 5 env-substep info dicts into one policy-step event set.

    Collision/off-road/arrive are OR over substeps; timeout is supplied by the
    caller or read from info.  ``metrics`` is the segment-level
    ``ActionTrajectory.metrics`` mapping (only the three comfort keys are used
    by :func:`compute_step_reward`).
    """
    infos = list(infos or ())
    collision = any(info_flag(info, *COLLISION_KEYS) for info in infos)
    off_road = any(info_flag(info, *OFFROAD_KEYS) for info in infos)
    arrive_dest = any(info_flag(info, *SUCCESS_KEYS) for info in infos)
    timeout = bool(timeout) or any(info_flag(info, *TIMEOUT_KEYS) for info in infos)
    return StepEvents(
        collision=bool(collision),
        off_road=bool(off_road),
        arrive_dest=bool(arrive_dest),
        timeout=bool(timeout),
        invalid_action=bool(invalid_action),
        rc0=float(rc0),
        rc1=float(rc1),
        v_end=float(v_end),
        metrics=dict(metrics or {}),
    )
