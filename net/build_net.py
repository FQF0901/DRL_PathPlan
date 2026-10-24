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
from net.world_model import FactorizedEgoConditionedLatentWM, TemporalSpatialBlock
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

    enc_cfg = _section(model_cfg, "encoder")
    temporal_cfg = _section(model_cfg, "temporal_spatial")
    planner_cfg = _section(model_cfg, "planner")
    wm_cfg = _section(model_cfg, "world_model")
    bank_cfg = _section(raw, "bank")
    rollout_cfg = _section(raw, "rollout")

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
    topk = int(bank_cfg.get("topk", 16))
    if not 1 <= topk <= 16:
        raise ValueError("bank.topk must be in 1..16")
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
    planner = Planner(hidden=planner_hidden, action_dim=action_dim)
    wm = FactorizedEgoConditionedLatentWM(
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
    )
    # Runtime configuration; not nn.Module singletons.
    model.memory_config = dict(bank_cfg)
    model.rollout_config = dict(rollout_cfg)
    model.model_config = dict(model_cfg)
    return model


__all__ = ["build_net", "load_config"]
