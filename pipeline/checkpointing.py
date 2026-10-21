"""Checkpoint schema / action-contract validation.

A checkpoint is a contract boundary: hidden/model width and action bounds must
match the evaluator before a rollout is meaningful.  This module validates the
metadata that ``train_bc`` writes and keeps a legacy fallback explicit rather
than silently loading an incompatible checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from net.action_space import CLEAN_BOUNDS
from net.model import StructuredPlanner


@dataclass(frozen=True)
class CheckpointInfo:
    hidden: int
    heads: int
    spatial_layers: int
    schema_name: str | None
    schema_version: int | None
    warnings: tuple[str, ...]


def _int(config: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (AttributeError, TypeError, ValueError):
        return int(default)


def _flat(value: Any) -> list[float]:
    return [float(x) for x in np.asarray(value, dtype=np.float64).reshape(-1)]


def validate_checkpoint(
    payload: Any,
    *,
    expected_low: Any = CLEAN_BOUNDS.low,
    expected_high: Any = CLEAN_BOUNDS.high,
    expected_dt: float = 0.5,
    expected_steps: int = 5,
    expected_history: int = 6,
    strict: bool = False,
) -> tuple[Mapping[str, Any], Mapping[str, Any], CheckpointInfo]:
    """Return ``(state_dict, config, info)`` or raise on a strict mismatch."""
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("checkpoint must be a dict with a 'model' state_dict")
    state = payload["model"]
    config = payload.get("config", {}) or {}
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    warnings: list[str] = []
    schema_name = config.get("schema_name")
    schema_version = config.get("schema_version")
    if schema_name is None or schema_version is None:
        warnings.append("legacy checkpoint has no schema_name/schema_version")
    else:
        if str(schema_name) != "drl_pathplan_obs":
            warnings.append(f"unexpected schema_name={schema_name!r}")
        if int(schema_version) != 2:
            warnings.append(f"unexpected schema_version={schema_version!r}")

    for key in ("action_low", "action_high"):
        if key not in config:
            warnings.append(f"missing {key}")
            continue
        expected = expected_low if key == "action_low" else expected_high
        if not np.allclose(_flat(config[key]), _flat(expected), atol=1e-9, rtol=0.0):
            warnings.append(f"{key} mismatch: {config[key]!r} vs {_flat(expected)!r}")

    decision = config.get("decision")
    if not isinstance(decision, Mapping):
        warnings.append("missing decision block")
    else:
        for key, expected in (
            ("dt", expected_dt),
            ("steps_per_policy", expected_steps),
            ("history_frames", expected_history),
        ):
            if key not in decision:
                warnings.append(f"decision.{key} missing")
            elif not np.isclose(float(decision[key]), float(expected), atol=1e-9):
                warnings.append(
                    f"decision.{key}={decision[key]!r} expected {expected!r}"
                )

    hidden = _int(config, "hidden", 128)
    heads = _int(config, "heads", 4)
    spatial_layers = _int(config, "spatial_layers", 2)
    info = CheckpointInfo(
        hidden=hidden,
        heads=heads,
        spatial_layers=spatial_layers,
        schema_name=None if schema_name is None else str(schema_name),
        schema_version=None if schema_version is None else int(schema_version),
        warnings=tuple(warnings),
    )
    if strict and warnings:
        raise ValueError("checkpoint validation failed: " + "; ".join(warnings))
    return state, config, info


def load_model_from_checkpoint(
    ckpt: str,
    device: str = "cpu",
    *,
    strict: bool = False,
) -> StructuredPlanner:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state, config, info = validate_checkpoint(payload, strict=strict)
    model = StructuredPlanner(
        hidden=info.hidden,
        heads=info.heads,
        spatial_layers=info.spatial_layers,
    )
    model.load_state_dict(state, strict=True)
    model.eval_config = {
        "hidden": info.hidden,
        "heads": info.heads,
        "spatial_layers": info.spatial_layers,
    }
    model.checkpoint_warnings = list(info.warnings)
    model.to(device).eval()
    return model
