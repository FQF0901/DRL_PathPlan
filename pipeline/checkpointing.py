"""Checkpoint schema / action-contract validation.

A checkpoint is a contract boundary: hidden/model width and action bounds must
match the evaluator before a rollout is meaningful.  This module validates the
metadata that ``train_bc`` writes and keeps a legacy fallback explicit rather
than silently loading an incompatible checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch

from env.action.config import load_action_contract

_CONTRACT = load_action_contract()
CLEAN_BOUNDS = _CONTRACT.bounds


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
    nested = config.get("obs_schema")
    if isinstance(nested, Mapping):
        schema_name = config.get("schema_name") or nested.get("name")
        schema_version = config.get("schema_version", nested.get("version"))
    else:
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
        if not np.allclose(_flat(config[key]), _flat(expected), atol=1e-6, rtol=1e-6):
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
):
    """Load the current bank-native ``DriveModel`` policy.

    Legacy adapters are no longer accepted.  This is intentionally a thin
    alias over :func:`load_drive_model_from_checkpoint` so BC/eval/PPO all use
    one checkpoint format.
    """
    return load_drive_model_from_checkpoint(ckpt, device, strict=strict)


def load_drive_model_from_checkpoint(
    ckpt: str,
    device: str = "cpu",
    *,
    strict: bool = False,
):
    """Load the bank-native ``DriveModel`` from a new-schema checkpoint.

    New checkpoints store the config under ``config`` with a ``model`` mapping
    (and action/ego metadata).  Legacy checkpoints are
    explicitly rejected; retrain/save a bank-native ``DriveModel`` checkpoint.
    """
    from net.build_net import build_net
    from net.model import DriveModel

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("checkpoint must be a dict with a 'model' state_dict")
    config = payload.get("config", {}) or {}
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
    # Validate schema/decision metadata as a contract boundary.  Use the
    # checkpoint's own serialized action bounds as the expected bounds here;
    # the caller's env/action contract is validated by eval/PPO separately.
    raw_action = config.get("action", {})
    raw_bounds = raw_action.get("bounds", {}) if isinstance(raw_action, Mapping) else {}
    action_name = str(raw_bounds.get("name", config.get("action_bounds_name", "clean")))
    if action_name in ("clean", "None"):
        contract = load_action_contract()
        expected_low = contract.bounds.low
        expected_high = contract.bounds.high
    else:
        expected_low = raw_bounds.get("low", config.get("action_low", CLEAN_BOUNDS.low))
        expected_high = raw_bounds.get("high", config.get("action_high", CLEAN_BOUNDS.high))
    _state, _cfg, info = validate_checkpoint(
        payload,
        expected_low=expected_low,
        expected_high=expected_high,
        strict=strict,
    )
    model_cfg = config.get("model")
    if isinstance(model_cfg, Mapping):
        raw = dict(config)
        model = build_net(raw)
    else:
        hidden = int(config.get("hidden", 128))
        action_dim = int(config.get("action_dim", 4))
        action = config.get("action", {})
        bounds = action.get("bounds", {}) if isinstance(action, Mapping) else {}
        low = bounds.get("low", config.get("action_low"))
        high = bounds.get("high", config.get("action_high"))
        ego = config.get("ego", {})
        model = DriveModel(
            hidden=hidden,
            heads=int(config.get("heads", 4)),
            spatial_layers=int(config.get("spatial_layers", 2)),
            action_dim=action_dim,
            action_low=low,
            action_high=high,
            ego_dt=float(ego.get("dt", 0.5)),
            ego_wheelbase=float(ego.get("wheelbase", 2.46894)),
            ego_max_steer_rad=float(ego.get("max_steer_rad", 0.6981317007977318)),
        )
    state = payload["model"]
    if any(str(key).startswith("drive.") for key in state):
        raise ValueError(
            "legacy drive.* checkpoint is unsupported; retrain/save a "
            "bank-native DriveModel checkpoint"
        )
    if not any(str(key).startswith(("encoder.", "temporal_spatial.")) for key in state):
        raise ValueError(
            "legacy checkpoint is unsupported by DriveModel; "
            "retrain/save a bank-native DriveModel checkpoint"
        )
    model.load_state_dict(state, strict=bool(strict))
    model.model_config = dict(config.get("model", {})) if isinstance(config.get("model"), Mapping) else {
        "hidden": hidden,
        "attention_heads": int(config.get("heads", 4)),
        "spatial_layers": int(config.get("spatial_layers", 2)),
    }
    resolved_low = locals().get("low")
    resolved_high = locals().get("high")
    if resolved_low is None and model.action_low is not None:
        resolved_low = model.action_low.detach().cpu().tolist()
    if resolved_high is None and model.action_high is not None:
        resolved_high = model.action_high.detach().cpu().tolist()
    action_cfg = config.get("action", {})
    action_bounds = action_cfg.get("bounds", {}) if isinstance(action_cfg, Mapping) else {}
    model.action_config = {
        "name": str(action_bounds.get("name", config.get("action_bounds_name", "clean"))),
        "low": list(map(float, resolved_low or [])),
        "high": list(map(float, resolved_high or [])),
        "limits": dict(action_cfg.get("limits", config.get("limits", {}))),
    }
    ego_cfg = config.get("ego", {})
    model.ego_config = {
        "dt": float(ego_cfg.get("dt", 0.5)),
        "wheelbase": float(ego_cfg.get("wheelbase", 2.46894)),
        "max_steer_rad": float(ego_cfg.get("max_steer_rad", 0.6981317007977318)),
    }
    model.eval_config = {
        "hidden": int(getattr(model, "hidden", 128)),
        "heads": int(getattr(model, "heads", 4)),
        "spatial_layers": int(getattr(model, "spatial_layers", 2)),
    }
    model.checkpoint_warnings = list(info.warnings)
    model.to(device).eval()
    return model


def merge_wm_into_drive(
    drive_ckpt: str,
    wm_ckpt: str,
    out_ckpt: str,
    *,
    device: str = "cpu",
) -> str:
    """Merge a ``train_wm`` checkpoint into a bank-native DriveModel checkpoint."""
    drive_payload = torch.load(drive_ckpt, map_location="cpu", weights_only=False)
    wm_payload = torch.load(wm_ckpt, map_location="cpu", weights_only=False)
    drive = load_drive_model_from_checkpoint(drive_ckpt, device, strict=True)
    drive.world_model.load_state_dict(wm_payload["model"], strict=True)
    config = dict(drive_payload.get("config", {}) or {})
    config["model"] = dict(drive.model_config)
    from dataclasses import asdict
    from env.action.config import load_action_contract
    hard = drive.action_config.get("limits", {}).get("hard")
    comfort = drive.action_config.get("limits", {}).get("comfort")
    if hard is None or comfort is None:
        contract = load_action_contract()
        hard, comfort = asdict(contract.hard), asdict(contract.comfort)
    if drive.action_low is None or drive.action_high is None:
        raise ValueError("DriveModel checkpoint has no action bounds")
    config["action"] = {
        "bounds": {
            "name": drive.action_config.get("name", "clean"),
            "low": drive.action_low.detach().cpu().tolist(),
            "high": drive.action_high.detach().cpu().tolist(),
        },
        "limits": {"hard": hard, "comfort": comfort},
    }
    config["ego"] = dict(drive.ego_config)
    torch.save({"model": drive.state_dict(), "config": config}, out_ckpt)
    return out_ckpt


def save_drive_model_checkpoint(model, path: str, *, extra: Mapping[str, Any] | None = None) -> str:
    from dataclasses import asdict
    from env.action.config import load_action_contract
    if model.action_low is None or model.action_high is None:
        raise ValueError("DriveModel has no action bounds; refusing to save a lying checkpoint")
    hard = model.action_config.get("limits", {}).get("hard")
    comfort = model.action_config.get("limits", {}).get("comfort")
    if hard is None or comfort is None:
        contract = load_action_contract()
        hard, comfort = asdict(contract.hard), asdict(contract.comfort)
    config = {
        "schema_name": "drl_pathplan_obs",
        "schema_version": 2,
        "model": dict(model.model_config),
        "decision": {
            "dt": float(model.ego_config.get("dt", 0.5)),
            "steps_per_policy": 5,
            "history_frames": 6,
        },
        "rollout": {"horizon": int(getattr(model, "rollout_horizon", 6))},
        "action_low": model.action_low.detach().cpu().tolist(),
        "action_high": model.action_high.detach().cpu().tolist(),
        "action": {
            "bounds": {
                "name": model.action_config.get("name", "clean"),
                "low": model.action_low.detach().cpu().tolist(),
                "high": model.action_high.detach().cpu().tolist(),
            },
            "limits": {"hard": hard, "comfort": comfort},
        },
        "ego": dict(model.ego_config),
    }
    if extra:
        config.update(dict(extra))
    torch.save({"model": model.state_dict(), "config": config}, path)
    return path
