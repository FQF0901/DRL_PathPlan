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
        cfg = load_config()
    elif isinstance(config, (str, Path)):
        cfg = load_config(config)
    else:
        cfg = dict(config)

    if isinstance(cfg.get("model"), Mapping):
        cfg = dict(cfg["model"])

    hidden = int(cfg.get("hidden", 128))
    heads = int(cfg.get("attention_heads", cfg.get("heads", 4)))
    spatial_layers = int(cfg.get("spatial_layers", 2))
    action_dim = int(cfg.get("action_dim", 4))

    enc_cfg = _section(cfg, "encoder")
    temporal_cfg = _section(cfg, "temporal_spatial")
    planner_cfg = _section(cfg, "planner")
    wm_cfg = _section(cfg, "world_model")

    encoder = CurrentFrameEncoder(hidden=int(enc_cfg.get("hidden", hidden)))
    temporal = TemporalSpatialBlock(
        hidden=int(temporal_cfg.get("hidden", hidden)),
        heads=int(temporal_cfg.get("heads", heads)),
        spatial_layers=int(temporal_cfg.get("spatial_layers", spatial_layers)),
    )
    planner = Planner(
        hidden=int(planner_cfg.get("hidden", hidden)),
        action_dim=int(planner_cfg.get("action_dim", action_dim)),
    )
    wm = FactorizedEgoConditionedLatentWM(
        hidden=int(wm_cfg.get("hidden", hidden)),
        action_dim=int(wm_cfg.get("action_dim", action_dim)),
        points=int(wm_cfg.get("ld_points", 5)),
    )
    return DriveModel(
        encoder=encoder,
        temporal_spatial=temporal,
        planner=planner,
        world_model=wm,
        hidden=hidden,
        heads=heads,
        spatial_layers=spatial_layers,
        action_dim=action_dim,
    )


__all__ = ["build_net", "load_config"]
