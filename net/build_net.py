"""Build the bank-native network from ``config/model.yaml``.

The YAML is configuration, not an executable forward graph.  This module
instantiates the component classes; ``net.model.DriveModel`` owns the forward
and imagine orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from net.encoder import CurrentFrameEncoder
from net.model import DriveModel
from net.planner import Planner
from net.world_model import EgoConditionedLatentWM, TemporalSpatialBlock
from env.obs import LD_POINTS


def load_config(path: str | Path = "config/model.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config/model.yaml must contain a mapping")
    return data


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def build_net(config: Mapping[str, Any] | str | Path | None = None) -> DriveModel:
    if config is None:
        raw = load_config()
    elif isinstance(config, (str, Path)):
        raw = load_config(config)
    else:
        raw = dict(config)

    model_cfg = dict(raw.get("model", raw))
    hidden = int(model_cfg.get("hidden", 128))
    heads = int(model_cfg.get("attention_heads", model_cfg.get("heads", 4)))
    spatial_layers = int(model_cfg.get("spatial_layers", 2))
    action_dim = int(model_cfg.get("action_dim", 4))
    if action_dim != 4:
        raise ValueError("this version fixes action_dim at 4")

    enc_cfg = _section(model_cfg, "encoder")
    temporal_cfg = _section(model_cfg, "temporal_spatial")
    planner_cfg = _section(model_cfg, "planner")
    wm_cfg = _section(model_cfg, "world_model")
    bank_cfg = _section(raw, "bank")
    rollout_cfg = _section(raw, "rollout")
    action_cfg = _section(raw, "action")
    action_bounds = _section(action_cfg, "bounds") if action_cfg else {}
    action_low = action_bounds.get("low")
    action_high = action_bounds.get("high")
    if (action_low is None) != (action_high is None):
        raise ValueError("action bounds must provide both low and high")
    if action_low is None:
        raise ValueError("config/model.yaml must provide action.bounds low/high")
    if len(action_low) != action_dim or len(action_high) != action_dim:
        raise ValueError("action bounds length must match action_dim")
    if any(float(lo) >= float(hi) for lo, hi in zip(action_low, action_high)):
        raise ValueError("action bounds low must be <= high")
    limits_cfg = _section(action_cfg, "limits")
    hard_cfg = _section(limits_cfg, "hard")
    comfort_cfg = _section(limits_cfg, "comfort")
    limit_keys = (
        "a_long_min", "a_long_max", "a_lat_max", "kappa_max",
        "kappa_rate_max", "jerk_long_max", "jerk_lat_max",
    )
    for name, section in (("hard", hard_cfg), ("comfort", comfort_cfg)):
        missing = [key for key in limit_keys if key not in section]
        if missing:
            raise ValueError(f"action.limits.{name} missing {missing}")
        if float(section["a_long_min"]) > float(section["a_long_max"]):
            raise ValueError(f"action.limits.{name}.a_long_min > a_long_max")

    enc_hidden = int(enc_cfg.get("hidden", hidden))
    temporal_hidden = int(temporal_cfg.get("hidden", hidden))
    planner_hidden = int(planner_cfg.get("hidden", hidden))
    wm_hidden = int(wm_cfg.get("hidden", hidden))
    if len({hidden, enc_hidden, temporal_hidden, planner_hidden, wm_hidden}) != 1:
        raise ValueError(
            "hidden mismatch: model=%d encoder=%d temporal=%d planner=%d wm=%d"
            % (hidden, enc_hidden, temporal_hidden, planner_hidden, wm_hidden)
        )
    if hidden % heads != 0:
        raise ValueError("hidden must be divisible by attention heads")
    if int(temporal_cfg.get("heads", heads)) != heads:
        raise ValueError("temporal heads must match model heads")
    if int(temporal_cfg.get("spatial_layers", spatial_layers)) != spatial_layers:
        raise ValueError("temporal spatial_layers must match model spatial_layers")
    if int(model_cfg.get("history_frames", 6)) != 6:
        raise ValueError("history_frames is fixed at 6 for this version")
    if int(bank_cfg.get("capacity", 6)) != 6:
        raise ValueError("bank.capacity is fixed at 6 for this version")
    od_slots = int(bank_cfg.get("od_slots", 16))
    ld_slots = int(bank_cfg.get("ld_slots", 16))
    if (od_slots, ld_slots) != (16, 16):
        raise ValueError("bank od_slots/ld_slots are fixed at 16 in this version")
    ld_points = int(wm_cfg.get("ld_points", LD_POINTS))
    if ld_points != LD_POINTS:
        raise ValueError(f"world_model.ld_points must equal env.obs.LD_POINTS={LD_POINTS}")
    horizon = int(rollout_cfg.get("horizon", 6))
    if not 1 <= horizon <= 6:
        raise ValueError("rollout.horizon must be in 1..6")
    ego_cfg = _section(raw, "ego")
    ego_dt = float(ego_cfg.get("dt", 0.5))
    ego_wheelbase = float(ego_cfg.get("wheelbase", 2.46894))
    ego_max_steer_rad = float(ego_cfg.get("max_steer_rad", 0.6981317007977318))
    if min(ego_dt, ego_wheelbase, ego_max_steer_rad) <= 0.0:
        raise ValueError("ego.dt/wheelbase/max_steer_rad must be positive")
    if int(planner_cfg.get("action_dim", action_dim)) != action_dim:
        raise ValueError("planner action_dim must match model action_dim")
    if int(wm_cfg.get("action_dim", action_dim)) != action_dim:
        raise ValueError("world_model action_dim must match model action_dim")

    encoder = CurrentFrameEncoder(hidden=enc_hidden)
    temporal = TemporalSpatialBlock(
        hidden=temporal_hidden,
        heads=heads,
        spatial_layers=spatial_layers,
        od_slots=od_slots,
        ld_slots=ld_slots,
    )
    planner = Planner(hidden=planner_hidden, action_dim=action_dim, heads=heads)
    wm = EgoConditionedLatentWM(
        hidden=wm_hidden,
        action_dim=action_dim,
        points=ld_points,
    )
    model = DriveModel(
        encoder=encoder,
        temporal_spatial=temporal,
        planner=planner,
        world_model=wm,
        hidden=hidden,
        heads=heads,
        spatial_layers=spatial_layers,
        action_dim=action_dim,
        rollout_horizon=horizon,
        action_low=action_low,
        action_high=action_high,
        ego_dt=ego_dt,
        ego_wheelbase=ego_wheelbase,
        ego_max_steer_rad=ego_max_steer_rad,
    )
    # Runtime configuration; not nn.Module singletons.
    model.memory_config = dict(bank_cfg)
    model.rollout_config = dict(rollout_cfg)
    model.model_config = dict(model_cfg)
    model.action_config = {
        "name": str(action_bounds.get("name", "clean")),
        "low": [float(x) for x in action_low],
        "high": [float(x) for x in action_high],
        "limits": {
            "hard": {key: float(hard_cfg[key]) for key in limit_keys},
            "comfort": {key: float(comfort_cfg[key]) for key in limit_keys},
        },
    }
    model.ego_config = {
        "dt": ego_dt,
        "wheelbase": ego_wheelbase,
        "max_steer_rad": ego_max_steer_rad,
    }
    return model


__all__ = ["build_net", "load_config"]
