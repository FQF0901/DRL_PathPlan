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

from pipeline.action_compat import CLEAN_BOUNDS
from pipeline.legacy_model import StructuredPlanner


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
) -> StructuredPlanner:
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw_state = payload.get("model", {}) if isinstance(payload, dict) else {}
    if isinstance(raw_state, Mapping) and raw_state and any(
        str(key).startswith("encoder.") for key in raw_state
    ) and not any(str(key).startswith("drive.") for key in raw_state):
        # Pure DriveModel checkpoint (e.g. train_wm best_drive.pt): load it
        # into a compatibility adapter for the legacy eval/BC/PPO entrypoints.
        drive = load_drive_model_from_checkpoint(ckpt, device, strict=True)
        cfg = payload.get("config", {}) or {}
        model_cfg = cfg.get("model", {}) if isinstance(cfg, Mapping) else {}
        adapter = StructuredPlanner(
            hidden=drive.hidden,
            heads=int(model_cfg.get("attention_heads", 4)),
            spatial_layers=int(model_cfg.get("spatial_layers", 2)),
            action_low=drive.action_low,
            action_high=drive.action_high,
        )
        adapter.drive.load_state_dict(drive.state_dict(), strict=True)
        adapter.drive.action_config = dict(drive.action_config)
        adapter.drive.ego_config = dict(drive.ego_config)
        adapter.eval_config = {
            "hidden": drive.hidden,
            "heads": adapter.heads,
            "spatial_layers": adapter.spatial_layers,
        }
        adapter.checkpoint_warnings = []
        adapter.to(device).eval()
        return adapter
    raw_cfg = payload.get("config", {}) or {}
    custom_name = str(raw_cfg.get("action_bounds_name", "clean")) if isinstance(raw_cfg, Mapping) else "clean"
    if (
        isinstance(raw_cfg, Mapping)
        and custom_name != "clean"
        and "action_low" in raw_cfg
        and "action_high" in raw_cfg
    ):
        state, config, info = validate_checkpoint(
            payload,
            expected_low=raw_cfg["action_low"],
            expected_high=raw_cfg["action_high"],
            strict=strict,
        )
    else:
        state, config, info = validate_checkpoint(payload, strict=strict)
    if not any(str(key).startswith("drive.") for key in state):
        raise ValueError(
            "legacy StructuredPlanner checkpoint is unsupported by the bank-native "
            "adapter; retrain/save a DriveModelAdapter checkpoint"
        )
    model = StructuredPlanner(
        hidden=info.hidden,
        heads=info.heads,
        spatial_layers=info.spatial_layers,
    )
    model.load_state_dict(state, strict=True)
    if isinstance(config.get("action"), Mapping):
        model.drive.action_config = {
            "name": str(config["action"].get("bounds", {}).get("name", "clean")),
            "low": list(map(float, config["action"].get("bounds", {}).get("low", []))),
            "high": list(map(float, config["action"].get("bounds", {}).get("high", []))),
            "limits": dict(config["action"].get("limits", {})),
        }
    else:
        from env.action.config import load_action_contract
        fallback = load_action_contract()
        limits = dict(config.get("limits", {}) or {})
        hard = limits.get("hard", asdict(fallback.hard))
        comfort = limits.get("comfort", asdict(fallback.comfort))
        model.drive.action_config = {
            "name": str(config.get("action_bounds_name", "clean")),
            "low": list(map(float, config.get("action_low", fallback.bounds.low.tolist()))),
            "high": list(map(float, config.get("action_high", fallback.bounds.high.tolist()))),
            "limits": {"hard": hard, "comfort": comfort},
        }
    if isinstance(config.get("ego"), Mapping):
        model.drive.ego_config = dict(config["ego"])
    model.eval_config = {
        "hidden": info.hidden,
        "heads": info.heads,
        "spatial_layers": info.spatial_layers,
    }
    model.checkpoint_warnings = list(info.warnings)
    model.to(device).eval()
    return model


def load_drive_model_from_checkpoint(
    ckpt: str,
    device: str = "cpu",
    *,
    strict: bool = False,
):
    """Load the bank-native ``DriveModel`` from a new-schema checkpoint.

    New checkpoints store the config under ``config`` with a ``model`` mapping
    (and action/ego metadata).  Legacy ``StructuredPlanner`` checkpoints are
    explicitly rejected; retrain/save a ``DriveModelAdapter`` checkpoint.
    """
    from net.build_net import build_net
    from net.model import DriveModel

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("checkpoint must be a dict with a 'model' state_dict")
    config = payload.get("config", {}) or {}
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint config must be a mapping")
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
    if not any(str(key).startswith(("drive.", "encoder.", "temporal_spatial.")) for key in state):
        raise ValueError(
            "legacy StructuredPlanner checkpoint is unsupported by DriveModel; "
            "retrain/save a DriveModelAdapter checkpoint"
        )
    if any(str(key).startswith("drive.") for key in state):
        state = {
            str(key)[6:]: value
            for key, value in state.items()
            if str(key).startswith("drive.")
        }
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
    model.checkpoint_warnings = []
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
