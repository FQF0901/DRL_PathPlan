"""Load the single action contract from ``config/model.yaml``.

Importing ``ActionBounds``/``DynamicsLimits`` is deferred so this module can be
used by ``action_space.py`` without a circular top-level import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "model.yaml"


@dataclass(frozen=True)
class ActionContract:
    bounds: Any
    hard: Any
    comfort: Any


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _limits(mapping: Mapping[str, Any], name: str):
    from env.action.action_space import DynamicsLimits

    required = (
        "a_long_min", "a_long_max", "a_lat_max", "kappa_max",
        "kappa_rate_max", "jerk_long_max", "jerk_lat_max",
    )
    missing = [key for key in required if key not in mapping]
    if missing:
        raise ValueError(f"action.limits.{name} missing {missing}")
    return DynamicsLimits(**{key: float(mapping[key]) for key in required})


def load_action_contract(path: str | Path = DEFAULT_CONFIG_PATH) -> ActionContract:
    from env.action.action_space import ActionBounds

    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    action = _mapping(raw.get("action"), "action")
    bounds = _mapping(action.get("bounds"), "action.bounds")
    if "low" not in bounds or "high" not in bounds:
        raise ValueError("action.bounds must provide low and high")
    low = [float(x) for x in bounds["low"]]
    high = [float(x) for x in bounds["high"]]
    if len(low) != 4 or len(high) != 4:
        raise ValueError("action.bounds must have 4 low and 4 high values")
    limits = _mapping(action.get("limits"), "action.limits")
    return ActionContract(
        bounds=ActionBounds(low=low, high=high, name=str(bounds.get("name", "clean"))),
        hard=_limits(_mapping(limits.get("hard"), "action.limits.hard"), "hard"),
        comfort=_limits(_mapping(limits.get("comfort"), "action.limits.comfort"), "comfort"),
    )


__all__ = ["ActionContract", "load_action_contract", "DEFAULT_CONFIG_PATH"]
