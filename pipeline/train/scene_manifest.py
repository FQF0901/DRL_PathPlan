from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "drl_pathplan_rl_scene_manifest_v1"


def _as_list(value: Any) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(item) for item in value]


def load_scene_manifest(
    path: str | Path,
    specs: Sequence[Any],
    *,
    specs_path: str | Path,
    group_size: int,
    std_min: float | Sequence[float],
    action_mode: str = "4d_exact",
) -> tuple[list[Any], dict[str, Any]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("invalid scene manifest schema")
    selected = payload.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("scene manifest selected must be a non-empty list")
    selected = [int(seed) for seed in selected]
    if len(selected) != len(set(selected)):
        raise ValueError("scene manifest selected must be unique")
    if int(payload.get("members", -1)) != int(group_size):
        raise ValueError("scene manifest members != group_size")
    if _as_list(payload.get("std_min", [])) != _as_list(std_min):
        raise ValueError("scene manifest std_min != training std_min")
    if str(payload.get("action_mode", "")) != str(action_mode):
        raise ValueError("scene manifest action_mode != training action_mode")
    digest = hashlib.sha256(Path(specs_path).read_bytes()).hexdigest()
    if str(payload.get("spec_sha256", "")) != digest:
        raise ValueError("scene manifest spec_sha256 != specs file")
    by_seed = {int(spec.seed): spec for spec in specs}
    missing = sorted(set(selected) - set(by_seed))
    if missing:
        raise ValueError(f"scene manifest seeds not in specs: {missing[:5]}")
    train_specs = [by_seed[seed] for seed in selected]
    meta = {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selected": len(selected),
        "ckpt": payload.get("ckpt"),
        "spec_sha256": digest,
        "action_mode": action_mode,
    }
    return train_specs, meta
