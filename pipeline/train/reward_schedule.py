from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Mapping, Sequence

from pipeline.train.rl_reward import RewardConfig

_FIELDS = frozenset(
    field.name
    for field in fields(RewardConfig)
    if field.name.startswith("w_") or field.name == "adapt_coeff"
)
_TOP_KEYS = frozenset({"phases", "interpolate"})


def _phase(phase: Mapping[str, Any], previous: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(phase, Mapping):
        raise ValueError("reward schedule phase must be a mapping")
    values = dict(previous)
    for key, value in phase.items():
        if key == "until_update":
            continue
        if key not in _FIELDS:
            raise ValueError(f"unknown reward schedule field: {key}")
        if isinstance(value, bool):
            raise ValueError(f"reward schedule {key} must be numeric")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"reward schedule {key} must be numeric") from exc
        if number != number or number in (float("inf"), float("-inf")) or number < 0.0:
            raise ValueError(f"reward schedule {key} must be finite and >= 0")
        values[key] = number
    return values


def validate_reward_schedule(schedule: Mapping[str, Any], updates: int) -> None:
    if not isinstance(schedule, Mapping):
        raise ValueError("reward_schedule must be a mapping")
    unknown = set(schedule) - _TOP_KEYS
    if unknown:
        raise ValueError(f"unknown reward_schedule keys: {sorted(unknown)}")
    if "interpolate" in schedule and not isinstance(schedule["interpolate"], bool):
        raise ValueError("reward_schedule.interpolate must be bool")
    phases = schedule.get("phases")
    if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)) or not phases:
        raise ValueError("reward_schedule.phases must be a non-empty list")
    previous: dict[str, float] = {}
    last = 0
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ValueError("reward schedule phase must be a mapping")
        until = phase.get("until_update")
        if isinstance(until, bool) or not isinstance(until, int):
            raise ValueError("reward schedule until_update must be int")
        if until <= last:
            raise ValueError("reward schedule until_update must be strictly increasing")
        previous = _phase(phase, previous)
        last = until
    if last < int(updates):
        raise ValueError("reward schedule must cover all updates")


def resolve_reward_config(
    base: RewardConfig,
    schedule: Mapping[str, Any] | None,
    update: int,
) -> RewardConfig:
    if schedule is None or update <= 0:
        return base
    phases = list(schedule["phases"])
    interpolate = bool(schedule.get("interpolate", True))
    previous = {field.name: float(getattr(base, field.name)) for field in fields(RewardConfig) if field.name in _FIELDS}
    phase_values: list[dict[str, float]] = []
    for phase in phases:
        previous = _phase(phase, previous)
        phase_values.append(previous.copy())
    previous_until = 0
    previous_values = {
        field.name: float(getattr(base, field.name))
        for field in fields(RewardConfig)
        if field.name in _FIELDS
    }
    for index, phase in enumerate(phases):
        until = int(phase["until_update"])
        target = phase_values[index]
        if update <= until:
            if not interpolate or index == 0 or until == previous_until:
                values = target
            else:
                span = until - previous_until
                fraction = (update - previous_until - 1) / float(span - 1) if span > 1 else 1.0
                fraction = min(max(fraction, 0.0), 1.0)
                values = {
                    key: previous_values[key] + fraction * (target[key] - previous_values[key])
                    for key in _FIELDS
                }
            return replace(base, **values)
        previous_until = until
        previous_values = target
    return replace(base, **phase_values[-1])
