"""BC/DAgger training for the canonical planner.

This is the canonical trainer.  The previous implementation is available only
from git history/tag ``pre-consolidation``.

Data contract
-------------
Each input directory (or ``.npz`` file) must contain the arrays written by
``dataset.collect``: all ``HISTORY_KEYS`` plus the measured ``action`` row,
``scene_seed``, ``episode_id`` and outcome columns.  Only the model feature
keys below enter the adapter/DriveModel; IDs are metadata needed for stable
identity alignment, while audit fields and outcomes remain side arrays::

    od, od_mask, ld, ld_mask, nav, nav_mask, ego, ego_mask, hist_valid

The train/validation split is by **scene seed**, never by episode id.  The
same scene can be re-collected as multiple episodes (BC + DAgger), so an
episode split would leak scenes across the split boundary.

Loss
----
* truncated-Normal NLL on the normalized expert action (``[0,1]`` under
  ``CLEAN_BOUNDS``), with out-of-bounds labels clipped for the NLL only;
* point / heading / velocity losses on the *physical* denormalized action;
* curvature / longitudinal-acceleration / jerk-proxy hinge smoothness on the
  predicted action, active only on moving actions;

Per-row plan weights implement the failure/privilege policy:

* failed expert episode -> terminal segment gets ``--w-failed-expert`` (0.2),
  earlier segments keep their label unless another gate fails;
* ``label_valid == 0`` -> 0;
* ``action_in_bounds == 0`` -> 0;
* PSTL-infeasible rows -> ``--w-infeasible`` (0.2), not an unconditional drop;
* source expert (``source == 0``) -> ``--source-weight-expert`` (default 1);
* other sources -> ``--source-weight-other`` (default 1, DAgger hook);
* measured replay / lift -> ``--source-weight-replay/lift``;
* ``shield`` rows -> ``--w-shield`` (default 3); legacy ``intervened`` is
  recorded but is not a shield weight;
* ``privileged_expert_sample`` -> ``--w-privileged`` only when the batch is
  mixed; a constant multiplier cancels in ``_weighted_mean``;
* measured/finite residual disagreement -> ``--w-residual``.

"""

from __future__ import annotations


if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from dataset.collect import (
    COMFORT_LIMITS,
    HARD_LIMITS,
    HISTORY_KEYS,
)
from dataset.splits import assert_disjoint, split_by_scene_seed
from env.obs import (
    EGO_INDEX,
    HISTORY_FRAMES,
    LD_INDEX,
    OD_INDEX,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    validate_stable_ids,
)
from env.action.config import load_action_contract
from net.build_net import build_net
from net.model import DriveModel, count_parameters

_CONTRACT = load_action_contract()
CLEAN_BOUNDS = _CONTRACT.bounds
COMFORT_LIMITS = _CONTRACT.comfort
HARD_LIMITS = _CONTRACT.hard
from pipeline.common.checkpointing import load_model_from_checkpoint
from pipeline.common.distributions import truncated_normal_nll
from pipeline.train.scene_batches import SceneBatch, iter_scene_chunks
from pipeline.common.train_config import parse_train_args, timestamped_run_dir, latest_run_alias

#: Feature keys fed to the adapter.  IDs are metadata, not policy features.
INPUT_KEYS: Tuple[str, ...] = (
    "od",
    "od_mask",
    "ld",
    "ld_mask",
    "nav",
    "nav_mask",
    "ego",
    "ego_mask",
    "hist_valid",
    "od_ids",
    "ld_ids",
    "pose",
    "od_anchor",
    "od_taxonomy",
)

#: Per-row arrays kept in memory (features + labels/weights only).
KEEP_KEYS = set(INPUT_KEYS) | {
    "history_mode",
    "action",
    "scene_seed",
    "episode_id",
    "step",
    "success",
    "collision",
    "off_road",
    "wrong_route",
    "stuck",
    "timeout",
    "label_valid",
    "expert_plan_valid",
    "source",
    "privileged_expert_sample",
    "residual_xy",
    "residual_xy_norm",
    "residual_dpsi",
    "comfortable",
    "intervened",
    "shield",
    "done",
    "terminal_failed",
    "student_failure",
    "executed_action",
    "n_substeps",
    "feasible",
    "label_pass",
    "action_in_bounds",
    "label_source",
    "pose",
}

REQUIRED_ROW_KEYS = (
    "action",
    "scene_seed",
    "episode_id",
    "step",
    "success",
    "collision",
    "off_road",
    "timeout",
)

PLAN_KEYS = {"total", "nll", "point", "heading", "velocity", "curvature", "accel"}

#: Columns that must be available (hence required above) or can be defaulted.
_DEFAULT_ONE_SCALARS = (
    "label_valid", "expert_plan_valid", "feasible", "label_pass",
    "action_in_bounds", "label_source",
)
_DEFAULT_MINUS_ONE_SCALARS = ("comfortable",)
_DEFAULT_PRIVILEGED = "privileged_expert_sample"
_ALL_OPTIONAL_KEYS = (
    "wrong_route",
    "stuck",
    "label_valid",
    "expert_plan_valid",
    "source",
    "privileged_expert_sample",
    "residual_xy",
    "residual_xy_norm",
    "residual_dpsi",
    "comfortable",
    "intervened",
    "shield",
    "done",
    "terminal_failed",
    "student_failure",
    "executed_action",
    "n_substeps",
    "feasible",
    "label_pass",
    "action_in_bounds",
    "label_source",
)


@dataclass
class LossConfig:
    """Loss and plan-weight configuration (serialized into checkpoints)."""

    nll_coef: float = 1.0

    # physical-action plan terms
    w_point: float = 1.0
    w_heading: float = 1.0
    w_velocity: float = 0.2
    w_curvature: float = 0.1
    w_accel: float = 0.1

    # targets / units
    dt: float = 0.5

    # per-row plan weights
    w_privileged: float = 1.0
    w_residual: float = 0.2
    w_shield: float = 3.0
    w_student_failure: float = 2.0
    w_failed_expert: float = 0.2
    w_infeasible: float = 0.2
    brake_weight: float = 0.0
    near_weight: float = 0.0
    source_weight_expert: float = 1.0
    source_weight_other: float = 1.0
    source_weight_replay: float = 1.0
    source_weight_lift: float = 0.5
    residual_xy_thresh: float = 0.5
    residual_dpsi_thresh: float = 0.1


# ---------------------------------------------------------------------------
# Data loading / targets
# ---------------------------------------------------------------------------
def _resolve_npz_paths(path: Path) -> list[Path]:
    """Resolve a dataset file or a canonical shard directory.

    Canonical logical datasets hold ``shard_*.npz`` (or ``expert_bc.npz``)
    directly; historical directories still contain one ``expert_bc.npz``.
    """
    if not path.is_dir():
        return [path]
    preferred = path / "expert_bc.npz"
    if preferred.exists():
        return [preferred]
    files = sorted(path.glob("*.npz"))
    if files:
        return files
    nested = sorted(path.rglob("*.npz"))
    if nested:
        return nested
    return [preferred]


def _default_value(key: str, n: int) -> np.ndarray:
    if key == "executed_action":
        return np.full((n, 4), np.nan, dtype=np.float32)
    if key == "residual_xy":
        return np.zeros((n, 2), dtype=np.float32)
    if key in _DEFAULT_ONE_SCALARS:
        return np.ones(n, dtype=np.int8)
    if key in _DEFAULT_MINUS_ONE_SCALARS:
        return np.full(n, -1, dtype=np.int8)
    if key == _DEFAULT_PRIVILEGED:
        return np.zeros(n, dtype=np.int8)
    return np.zeros(n, dtype=np.float32)


def _sample_episode_rows(
    z: np.lib.npyio.NpzFile, max_rows: int, rng: np.random.Generator
) -> Optional[np.ndarray]:
    """Return row indices covering whole episodes, or ``None`` for no cap.

    Sampling whole episodes keeps the 6-frame history sequences intact.  The
    cap is a budget, not an exact target: the first selected episode may be
    larger than what remains.
    """
    n = int(len(np.asarray(z["episode_id"])))
    if max_rows <= 0 or n <= max_rows:
        return None
    episode_id = np.asarray(z["episode_id"], dtype=np.int64)
    unique, counts = np.unique(episode_id, return_counts=True)
    chosen: list[int] = []
    total = 0
    for index in rng.permutation(len(unique)):
        count = int(counts[index])
        if chosen and total + count > max_rows:
            continue
        chosen.append(int(unique[index]))
        total += count
        if total >= max_rows:
            break
    keep = np.isin(episode_id, np.asarray(chosen, dtype=np.int64))
    return np.flatnonzero(keep)


def _split_cap(max_rows: int, shards: int) -> list[int]:
    """Split one input-path row cap evenly across its npz shards."""
    if max_rows <= 0 or shards <= 1:
        return [max_rows] * max(1, shards)
    base, extra = divmod(int(max_rows), int(shards))
    return [base + (1 if i < extra else 0) for i in range(shards)]


def load_data(
    paths: Sequence[str | Path],
    max_rows: Optional[Sequence[int] | int] = None,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Load and concatenate one or more ``expert_bc.npz`` datasets.

    ``max_rows`` optionally caps each input path; ``0`` disables the cap.  If
    one integer is given it applies to every path, otherwise one value per
    path is required.  Complement episodes are sampled whole, deterministically
    from ``seed``.
    """
    if not paths:
        raise ValueError("load_data: no input paths")
    if max_rows is None:
        caps = [0] * len(paths)
    elif isinstance(max_rows, (int, np.integer)):
        caps = [int(max_rows)] * len(paths)
    else:
        caps = [int(value) for value in max_rows]
        if len(caps) == 1:
            caps *= len(paths)
        elif len(caps) != len(paths):
            raise ValueError(
                f"load_data: got {len(caps)} row caps for {len(paths)} paths"
            )
    parts = []
    for path_index, (raw, cap) in enumerate(zip(paths, caps)):
        resolved = _resolve_npz_paths(Path(raw))
        shard_caps = _split_cap(cap, len(resolved))
        for shard_index, npz_path in enumerate(resolved):
            if not npz_path.exists():
                raise FileNotFoundError(f"schema-v2 dataset not found: {npz_path}")
            with np.load(npz_path) as z:
                files = set(z.files)
                missing_hist = [key for key in HISTORY_KEYS if key not in files]
                if missing_hist:
                    raise ValueError(
                        f"{npz_path}: missing schema-v2 history keys {missing_hist}"
                    )
                missing_rows = [key for key in REQUIRED_ROW_KEYS if key not in files]
                if missing_rows:
                    raise ValueError(
                        f"{npz_path}: missing required row keys {missing_rows}"
                    )
                rng = np.random.default_rng(int(seed) + path_index)
                row_idx = _sample_episode_rows(z, shard_caps[shard_index], rng)
                part = {
                    key: np.asarray(
                        z[key] if row_idx is None else z[key][row_idx]
                    )
                    for key in files
                    if key in KEEP_KEYS
                }
                if row_idx is not None:
                    print(
                        f"[bc] sampled {len(row_idx)}/{len(np.asarray(z['episode_id']))} "
                        f"rows from {npz_path} (cap={shard_caps[shard_index]})"
                    )
            action = np.asarray(part["action"], dtype=np.float32)
            if action.ndim != 2 or action.shape[1] != 4:
                raise ValueError(f"{npz_path}: action must be [N,4], got {action.shape}")
            if not np.all(np.isfinite(action)):
                raise ValueError(f"{npz_path}: action contains NaN/Inf")
            n = int(action.shape[0])
            if n == 0:
                raise ValueError(f"{npz_path}: zero rows")
            validate_stable_ids(
                part["od_ids"], part["od_mask"] > 0.5,
                field="od", context=str(npz_path),
            )
            validate_stable_ids(
                part["ld_ids"], part["ld_mask"].any(axis=-1) > 0.5,
                field="ld", context=str(npz_path),
            )
            for key in REQUIRED_ROW_KEYS:
                if key in part and len(part[key]) != n:
                    raise ValueError(f"{npz_path}: {key} length {len(part[key])} != {n}")
            # Never silently merge episode 0 from two collection shards.
            episode_id = np.asarray(part["episode_id"], dtype=np.int64).copy()
            if parts:
                episode_id = episode_id + (int(parts[-1]["_episode_offset"]) + 1)
            part["_episode_offset"] = np.int64(
                int(episode_id.max()) if len(episode_id) else -1
            )
            part["episode_id"] = episode_id
            parts.append(part)

    merged: Dict[str, np.ndarray] = {}
    keys = set().union(*(set(part) for part in parts)) - {"_episode_offset"}
    for key in keys:
        arrays = []
        for part in parts:
            n = len(part["action"])
            arrays.append(part[key] if key in part else _default_value(key, n))
        try:
            merged[key] = np.concatenate(arrays, axis=0)
        except ValueError as exc:  # e.g. incompatible residual_xy shapes
            raise ValueError(f"cannot concatenate {key!r}: {exc}") from exc
    del parts, arrays

    merged["action"] = np.asarray(merged["action"], dtype=np.float32)
    merged["scene_seed"] = np.asarray(merged["scene_seed"], dtype=np.int64)
    merged["episode_id"] = np.asarray(merged["episode_id"], dtype=np.int64)
    merged["step"] = np.asarray(merged["step"], dtype=np.int64)
    if "history_mode" not in merged:
        raise ValueError(
            "raw history requires history_mode=1; regenerate shards with the "
            "current dataset.collect"
        )
    mode = np.asarray(merged["history_mode"], dtype=np.float32)
    if not np.all(np.isfinite(mode)) or not np.all(mode == 1.0):
        raise ValueError(
            "projected history is no longer supported; regenerate raw shards "
            "with the current dataset.collect (history_mode=1)"
        )
    for key in _ALL_OPTIONAL_KEYS:
        if key not in merged:
            merged[key] = _default_value(key, len(merged["action"]))
    n_total = len(merged["action"])
    for key, value in merged.items():
        if value.shape[0] != n_total:
            raise ValueError(
                f"merged {key!r} has {value.shape[0]} rows, expected {n_total}"
            )
    if not np.all(np.isfinite(merged["action"])):
        raise ValueError("merged action contains NaN/Inf")
    return merged


def concat_data(parts: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate data shards/pools by row; all keys must be present."""
    if not parts:
        raise ValueError("concat_data: no parts")
    keys = set(parts[0])
    for part in parts[1:]:
        keys &= set(part)
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        values = [np.asarray(part[key]) for part in parts]
        if len({tuple(v.shape[1:]) for v in values}) != 1:
            raise ValueError(f"concat_data: incompatible trailing shape for {key!r}")
        out[key] = np.concatenate(values, axis=0)
    return out


def split_masks(
    data: Mapping[str, np.ndarray], val_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Scene-seed split with explicit overlap checks."""
    train_mask, val_mask = split_by_scene_seed(data["scene_seed"], val_frac, seed)
    train_mask = np.asarray(train_mask, dtype=bool)
    val_mask = np.asarray(val_mask, dtype=bool)
    if int(np.logical_and(train_mask, val_mask).sum()):
        raise AssertionError("scene-seed split produced overlapping rows")
    if not bool(np.all(train_mask | val_mask)):
        raise AssertionError("scene-seed split left rows unassigned")
    train_seeds = set(int(x) for x in np.unique(data["scene_seed"][train_mask]))
    val_seeds = set(int(x) for x in np.unique(data["scene_seed"][val_mask]))
    overlap = train_seeds & val_seeds
    if overlap:
        raise AssertionError(f"scene seeds leak across split: {sorted(overlap)}")
    assert_disjoint((train_mask, "train"), (val_mask, "val"))
    return train_mask, val_mask


def _plan_weights(data: Mapping[str, np.ndarray], cfg: LossConfig) -> np.ndarray:
    """Per-row plan weight implementing failure / privilege / DAgger policy."""
    n = len(data["action"])
    w = np.ones(n, dtype=np.float32)

    expert_plan_valid = np.asarray(data["expert_plan_valid"], dtype=np.float32)
    label_valid = np.asarray(data["label_valid"], dtype=np.float32)
    feasible = np.asarray(data["feasible"], dtype=np.float32)
    label_pass = np.asarray(data["label_pass"], dtype=np.float32)
    action_in_bounds = np.asarray(data["action_in_bounds"], dtype=np.float32)
    # New collection marks the terminal failed segment explicitly.  The
    # `done` fallback keeps old data readable, but old data cannot recover a
    # terminal segment that was never saved.
    if "terminal_failed" in data and bool(
        np.any(np.asarray(data["terminal_failed"], dtype=np.float32) > 0.5)
    ):
        terminal_failed = np.asarray(data["terminal_failed"], dtype=np.float32) > 0.5
        # In DAgger, `terminal_failed` describes a failed *student* episode,
        # not a failed expert label.  Only expert-source rows may receive the
        # failed-expert plan downweight; recovery rows stay failure-prioritized.
        if "source" in data:
            terminal_failed = terminal_failed & (np.asarray(data["source"], dtype=np.float32) == 0.0)
    else:
        # Missing/old data: best-effort fallback.  A saved terminal segment is
        # required for a real fix; this branch only keeps old artifacts readable.
        failed_expert = expert_plan_valid <= 0.5
        done = np.asarray(data.get("done", np.zeros(n)), dtype=np.float32) > 0.5
        terminal_failed = failed_expert & done
    w *= np.where(terminal_failed, float(cfg.w_failed_expert), 1.0)
    w *= label_valid > 0.5
    # Keep a downweighted signal from PSTL-infeasible expert segments instead
    # of deleting the hard/curve states entirely.  The evaluator still never
    # executes an infeasible action; this is a training-data weight.
    w *= np.where((feasible > 0.5) & (label_pass > 0.5), 1.0, float(cfg.w_infeasible))

    source = np.asarray(data["source"], dtype=np.float32)
    w *= np.where(
        source == 0.0, float(cfg.source_weight_expert), float(cfg.source_weight_other)
    )
    label_source = np.asarray(data["label_source"], dtype=np.float32)
    w *= np.where(
        label_source >= 0.5,
        float(cfg.source_weight_replay),
        float(cfg.source_weight_lift),
    )

    privileged = np.asarray(data["privileged_expert_sample"], dtype=np.float32) > 0.5
    # A constant multiplier over all rows cancels in `_weighted_mean`; the knob
    # only has an effect when a mixed batch contains non-privileged rows.
    if not bool(privileged.all()):
        w *= np.where(privileged, float(cfg.w_privileged), 1.0)

    if "residual_xy_norm" in data:
        residual_xy = np.asarray(data["residual_xy_norm"], dtype=np.float32)
    else:
        residual_xy = np.linalg.norm(
            np.asarray(data["residual_xy"], dtype=np.float32), axis=1
        )
    residual_dpsi = np.abs(
        np.asarray(data["residual_dpsi"], dtype=np.float32)
    )
    measured = label_source >= 0.5
    finite_residual = np.isfinite(residual_xy) & np.isfinite(residual_dpsi)
    residual = measured & finite_residual & (
        (residual_xy > cfg.residual_xy_thresh) | (residual_dpsi > cfg.residual_dpsi_thresh)
    )
    w *= np.where(residual, float(cfg.w_residual), 1.0)

    # `shield` is the safety intervention; legacy `intervened` meant expert
    # replay and must not silently become a shield weight.
    if "shield" in data:
        shield = np.asarray(data["shield"], dtype=np.float32) > 0.5
    else:
        shield = np.zeros(n, dtype=bool)
    if bool(shield.any()):
        w *= np.where(shield, float(cfg.w_shield), 1.0)

    # Stage B failure-prioritized aggregation: rows from a failed student
    # episode are the recovery buffer and get extra plan weight.
    if "student_failure" in data and float(cfg.w_student_failure) != 1.0:
        student_failure = np.asarray(data["student_failure"], dtype=np.float32) > 0.5
        if bool(student_failure.any()):
            w *= np.where(student_failure, float(cfg.w_student_failure), 1.0)

    if cfg.brake_weight > 0.0 or cfg.near_weight > 0.0:
        action = np.asarray(data["action"], dtype=np.float32)
        ego = np.asarray(data["ego"], dtype=np.float32)
        v0 = ego[:, -1, EGO_INDEX["speed"]]
        brake = np.clip((v0 - action[:, 3]) / 1.0, 0.0, 2.0)
        od = np.asarray(data["od"], dtype=np.float32)[:, -1]
        od_valid = np.asarray(data["od_mask"], dtype=np.float32)[:, -1] > 0.5
        dx = od[..., OD_INDEX["dx"]]
        vx = od[..., OD_INDEX["vx_rel"]]
        ttc = np.where(
            od_valid & (dx > 0.0) & (vx < -0.1),
            dx / np.maximum(-vx, 0.1),
            1e6,
        )
        near = (ttc.min(axis=1) < 2.0).astype(np.float32)
        w *= 1.0 + float(cfg.brake_weight) * brake + float(cfg.near_weight) * near

    return np.clip(w, 0.0, None).astype(np.float32)


def build_targets(
    data: Mapping[str, np.ndarray], cfg: LossConfig, *, action_low=None, action_high=None
) -> Dict[str, np.ndarray]:
    """Precompute normalized/physical actions and plan weights."""
    action = np.asarray(data["action"], dtype=np.float32)
    low = CLEAN_BOUNDS.low.astype(np.float32) if action_low is None else np.asarray(action_low, dtype=np.float32)
    high = CLEAN_BOUNDS.high.astype(np.float32) if action_high is None else np.asarray(action_high, dtype=np.float32)
    width = np.maximum(high - low, 1e-6)
    action_norm = (action - low) / width
    label_oob = (action_norm < 0.0) | (action_norm > 1.0)
    action_norm_clipped = np.clip(action_norm, 0.0, 1.0)

    plan_weight = _plan_weights(data, cfg)
    # Bounds are model/checkpoint-specific; do not trust the collector's
    # clean-box action_in_bounds column.
    plan_weight = plan_weight * (~label_oob.any(axis=1))

    return {
        "action_phys": action.copy(),
        "action_norm": action_norm.astype(np.float32),
        "action_norm_clipped": action_norm_clipped.astype(np.float32),
        "label_oob": label_oob.astype(bool),
        "plan_weight": plan_weight,
    }


def _batch_from_indices(
    data: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    indices: np.ndarray,
    device: torch.device | str,
) -> Dict[str, Tensor]:
    idx = np.asarray(indices, dtype=np.int64)
    batch: Dict[str, Tensor] = {}
    for key in INPUT_KEYS:
        dtype = np.int64 if key in ("od_ids", "ld_ids") else np.float32
        arr = np.ascontiguousarray(data[key][idx], dtype=dtype)
        batch[key] = torch.from_numpy(arr).to(device)
    for key in (
        "action_phys",
        "action_norm",
        "action_norm_clipped",
        "plan_weight",
    ):
        arr = np.ascontiguousarray(targets[key][idx], dtype=np.float32)
        batch[key] = torch.from_numpy(arr).to(device)
    if "history_mode" in data:
        batch["history_mode"] = torch.from_numpy(
            np.asarray(data["history_mode"][idx], dtype=np.float32)
        ).to(device)
    return batch


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.float()
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return _weighted_mean(values, mask.float())


def _denorm(mu: Tensor, low=None, high=None) -> Tensor:
    low = torch.tensor(CLEAN_BOUNDS.low if low is None else low, dtype=mu.dtype, device=mu.device)
    high = torch.tensor(CLEAN_BOUNDS.high if high is None else high, dtype=mu.dtype, device=mu.device)
    return low + mu * (high - low)


def compute_losses(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    cfg: LossConfig,
    *,
    action_low=None,
    action_high=None,
    hard_limits=None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return ``(total, components)`` for one batch.

    Plan components use ``batch['plan_weight']`` as their row weight.
    """
    mu = outputs["action_mu"]
    logstd = outputs["action_logstd"]
    weight = batch["plan_weight"]
    target_phys = batch["action_phys"]
    pred_phys = _denorm(mu, action_low, action_high)

    # ---- truncated Normal NLL on clipped normalized targets -----------------
    nll_row, _outside = truncated_normal_nll(
        batch["action_norm_clipped"], mu, logstd, low=0.0, high=1.0
    )
    nll = _weighted_mean(nll_row, weight)

    # ---- physical point / heading / velocity --------------------------------
    point = _weighted_mean(
        F.mse_loss(pred_phys[:, :2], target_phys[:, :2], reduction="none").mean(-1),
        weight,
    )
    heading = _weighted_mean(
        (pred_phys[:, 2] - target_phys[:, 2]).pow(2), weight
    )
    velocity = _weighted_mean(
        (pred_phys[:, 3] - target_phys[:, 3]).pow(2), weight
    )

    # ---- curvature / accel / jerk-proxy hinges on the predicted action ------
    chord = pred_phys[:, :2].pow(2).sum(-1).sqrt()
    kappa = pred_phys[:, 2] / chord.clamp_min(1e-3)
    v0 = batch["ego"][:, -1, EGO_INDEX["speed"]]
    a_long = (pred_phys[:, 3] - v0) / max(float(cfg.dt), 1e-6)
    ego_a_long = batch["ego"][:, -1, EGO_INDEX["a_long"]]
    ego_curvature = batch["ego"][:, -1, EGO_INDEX["curvature"]]
    jerk_proxy = (a_long - ego_a_long) / max(float(cfg.dt), 1e-6)
    kappa_rate_proxy = (kappa - ego_curvature) / max(float(cfg.dt), 1e-6)

    target_chord = target_phys[:, :2].pow(2).sum(-1).sqrt()
    moving = ((target_chord > 0.1) | (target_phys[:, 3] > 0.1)).float()
    smooth_weight = moving * weight

    hard = HARD_LIMITS if hard_limits is None else hard_limits
    curvature_hinge = (
        F.relu(kappa.abs() - float(hard.kappa_max)).pow(2)
        + F.relu(kappa_rate_proxy.abs() - float(hard.kappa_rate_max)).pow(2)
    )
    accel_hinge = (
        F.relu(float(hard.a_long_min) - a_long).pow(2)
        + F.relu(a_long - float(hard.a_long_max)).pow(2)
        + F.relu(jerk_proxy.abs() - float(hard.jerk_long_max)).pow(2)
    )
    curvature = _weighted_mean(curvature_hinge, smooth_weight)
    accel = _weighted_mean(accel_hinge, smooth_weight)

    total = (
        float(cfg.nll_coef) * nll
        + float(cfg.w_point) * point
        + float(cfg.w_heading) * heading
        + float(cfg.w_velocity) * velocity
        + float(cfg.w_curvature) * curvature
        + float(cfg.w_accel) * accel
    )

    components = {
        "total": total,
        "nll": nll,
        "point": point,
        "heading": heading,
        "velocity": velocity,
        "curvature": curvature,
        "accel": accel,
    }
    return total, components


def physical_mae(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    *,
    action_low=None,
    action_high=None,
) -> Tuple[Tensor, Tensor]:
    """Plan-weighted per-dimension physical MAE and its weight sum."""
    pred_phys = _denorm(outputs["action_mu"], action_low, action_high)
    weight = batch["plan_weight"].unsqueeze(-1)
    err = (pred_phys - batch["action_phys"]).abs()
    return (err * weight).sum(dim=0), weight.sum()


# ---------------------------------------------------------------------------
# Epoch loop
# ---------------------------------------------------------------------------
def _accumulate(
    sums: Dict[str, float],
    denoms: Dict[str, float],
    components: Mapping[str, Tensor],
    plan_weight_sum: float,
    n_rows: int,
) -> None:
    for key, value in components.items():
        denom = plan_weight_sum if key in PLAN_KEYS else float(n_rows)
        sums[key] = sums.get(key, 0.0) + float(value.detach()) * denom
        denoms[key] = denoms.get(key, 0.0) + denom


def _finalize(sums: Mapping[str, float], denoms: Mapping[str, float]) -> Dict[str, float]:
    return {
        key: (sums[key] / denoms[key] if denoms.get(key, 0.0) > 0.0 else 0.0)
        for key in sums
    }


def run_epoch(
    model: DriveModel,
    data: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    indices: np.ndarray,
    cfg: LossConfig,
    device: torch.device | str,
    batch_size: int,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    rng: Optional[np.random.Generator] = None,
    max_grad_norm: float = 1.0,
) -> Tuple[Dict[str, float], np.ndarray, int]:
    """Run one train/eval pass; return component means, MAE, plan row count."""
    training = optimizer is not None
    model.train(training)
    order = np.asarray(indices, dtype=np.int64)
    if training and rng is not None:
        order = order[rng.permutation(len(order))]
    if len(order) == 0:
        return {}, np.full(4, np.nan, dtype=np.float64), 0

    sums: Dict[str, float] = {}
    denoms: Dict[str, float] = {}
    mae_sum = torch.zeros(4, dtype=torch.float64)
    mae_weight = torch.zeros(1, dtype=torch.float64)
    n_plan_rows = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for start in range(0, len(order), int(batch_size)):
            idx = order[start : start + int(batch_size)]
            batch = _batch_from_indices(data, targets, idx, device)
            outputs = model(batch)
            _loss, components = compute_losses(
                outputs, batch, cfg,
                action_low=getattr(model, "action_low", None).detach().cpu().numpy()
                if getattr(model, "action_low", None) is not None else None,
                action_high=getattr(model, "action_high", None).detach().cpu().numpy()
                if getattr(model, "action_high", None) is not None else None,
                hard_limits=(
                    __import__("env.action.action_space", fromlist=["DynamicsLimits"]).DynamicsLimits(
                        **model.action_config["limits"]["hard"]
                    )
                    if getattr(model, "action_config", {}).get("limits", {}).get("hard")
                    else None
                ),
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                _loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            plan_sum = float(batch["plan_weight"].sum().detach())
            _accumulate(sums, denoms, components, plan_sum, len(idx))

            mae, wsum = physical_mae(
                outputs, batch,
                action_low=getattr(model, "action_low", None).detach().cpu().numpy()
                if getattr(model, "action_low", None) is not None else None,
                action_high=getattr(model, "action_high", None).detach().cpu().numpy()
                if getattr(model, "action_high", None) is not None else None,
            )
            mae_sum += mae.detach().double().cpu()
            mae_weight += wsum.detach().double().cpu()
            n_plan_rows += int((batch["plan_weight"] > 0).sum())

    mae = (mae_sum / mae_weight.clamp_min(1e-12)).numpy()
    if float(mae_weight) <= 0.0:
        mae = np.full(4, np.nan, dtype=np.float64)
    return _finalize(sums, denoms), mae, n_plan_rows


def run_scene_epoch(
    model: DriveModel,
    data: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    scene_batches: Sequence[SceneBatch],
    cfg: LossConfig,
    device: torch.device | str,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_grad_norm: float = 1.0,
    tbptt_chunk: int = 16,
) -> Tuple[Dict[str, float], np.ndarray, int]:
    """TBPTT over scene-ordered frames with chunk-boundary FIFO rebuilds.

    Frames are never shuffled inside a scene.  At every chunk boundary the
    initial 6-frame window is rebuilt from stored history with the *current*
    weights, so a post-step FIFO never carries stale encoder features.
    """
    training = optimizer is not None
    model.train(training)
    sums: Dict[str, float] = {}
    denoms: Dict[str, float] = {}
    mae_sum = torch.zeros(4, dtype=torch.float64)
    mae_weight = torch.zeros(1, dtype=torch.float64)
    n_plan_rows = 0
    hard_limits = (
        __import__("env.action.action_space", fromlist=["DynamicsLimits"]).DynamicsLimits(
            **model.action_config["limits"]["hard"]
        )
        if getattr(model, "action_config", {}).get("limits", {}).get("hard")
        else None
    )
    action_low = (
        model.action_low.detach().cpu().numpy()
        if getattr(model, "action_low", None) is not None else None
    )
    action_high = (
        model.action_high.detach().cpu().numpy()
        if getattr(model, "action_high", None) is not None else None
    )

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for scene_batch in scene_batches:
            if scene_batch.batch_size == 0 or scene_batch.max_length == 0:
                continue
            chunks = list(iter_scene_chunks(scene_batch, tbptt_chunk))
            # Fold a short tail into the previous chunk so AdamW does not give
            # the final few frames a disproportionately large per-frame step.
            if len(chunks) > 1 and (chunks[-1][1] - chunks[-1][0]) < int(tbptt_chunk):
                chunks[-2] = (chunks[-2][0], chunks[-1][1])
                chunks.pop()
            for chunk_start, chunk_stop in chunks:
                if not bool((scene_batch.lengths > chunk_start).any()):
                    break
                chunk_loss = torch.zeros((), dtype=torch.float32, device=device)
                chunk_plan_sum = 0.0
                chunk_has_rows = False
                for t in range(chunk_start, chunk_stop):
                    rows = scene_batch.rows[:, t]
                    batch = dict(_batch_from_indices(data, targets, rows, device))
                    active = torch.from_numpy(
                        (scene_batch.lengths > t).astype(np.float32)
                    ).to(device)
                    if not bool((active > 0.5).any()):
                        break
                    plan_weight = batch["plan_weight"] * active
                    batch["plan_weight"] = plan_weight
                    enc = None
                    if t > chunk_start:
                        enc = model.prepare_current_encoding(
                            model.encoder.encode_current(batch)
                        )
                    use_fifo = (
                        t > chunk_start
                        and enc is not None
                        and model.fifo_append_safe(window, enc)
                    )
                    if use_fifo:
                        prev_rows = scene_batch.rows[:, t - 1]
                        active_np = scene_batch.lengths > t
                        step_ok = data["step"][rows[active_np]] == data["step"][prev_rows[active_np]] + 1
                        hist_ok = batch["hist_valid"][:, -1].detach().cpu().numpy() > 0.5
                        use_fifo = bool(np.all(step_ok) and np.all(hist_ok[active_np]))
                    if not use_fifo:
                        # Raw data can have a re-entering entity that the
                        # filtered BankWindow FIFO cannot recover.
                        window, last_enc = model.build_history_window_fast(batch)
                    else:
                        window = model.align_window_to_current(window, enc)
                        pose = batch.get("ego_pose", batch.get("pose"))
                        if pose is None:
                            raise ValueError("raw sequence requires ego_pose/pose")
                        pose = pose[:, -1]
                        window = window.advance(
                            od_content=enc["od_content"],
                            od_geometry=enc["od_geometry"],
                            od_type=enc["od_type"],
                            od_valid=enc["od_valid"],
                            od_stable_id=enc["od_stable_id"],
                            ld_content=enc["ld_content"],
                            ld_geometry=enc["ld_geometry"],
                            ld_type=enc["ld_type"],
                            ld_valid=enc["ld_valid"],
                            ld_stable_id=enc["ld_stable_id"],
                            ego_pose=pose,
                            source=0,
                            rollout_depth=0,
                        )
                        last_enc = enc
                    outputs = model.forward_window(
                        window, last_enc["nav"], last_enc["ego"],
                        ego_valid=last_enc.get("ego_valid"),
                    )
                    loss, components = compute_losses(
                        outputs, batch, cfg,
                        action_low=action_low,
                        action_high=action_high,
                        hard_limits=hard_limits,
                    )
                    plan_sum = float(plan_weight.sum().detach())
                    chunk_loss = chunk_loss + loss * plan_sum
                    chunk_plan_sum += plan_sum
                    chunk_has_rows = True
                    active_rows = int((plan_weight > 0).sum())
                    _accumulate(sums, denoms, components, plan_sum, active_rows)
                    mae, wsum = physical_mae(
                        outputs, batch,
                        action_low=action_low,
                        action_high=action_high,
                    )
                    mae_sum += mae.detach().double().cpu()
                    mae_weight += wsum.detach().double().cpu()
                    n_plan_rows += int((plan_weight > 0).sum())
                if training and chunk_has_rows:
                    chunk_loss = chunk_loss / max(chunk_plan_sum, 1e-12)
                    optimizer.zero_grad(set_to_none=True)
                    chunk_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

    mae = (mae_sum / mae_weight.clamp_min(1e-12)).numpy()
    if float(mae_weight) <= 0.0:
        mae = np.full(4, np.nan, dtype=np.float64)
    return _finalize(sums, denoms), mae, n_plan_rows


# ---------------------------------------------------------------------------
# Checkpoints / serialization
# ---------------------------------------------------------------------------
def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _save_checkpoint(
    path: Path,
    model: DriveModel,
    config: Mapping,
    epoch: int,
    metrics: Mapping,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "epoch": int(epoch),
            "metrics": _jsonable(metrics),
        },
        path,
    )


def _make_config(
    model: DriveModel,
    cfg: LossConfig,
    args: argparse.Namespace,
    data: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> Dict:
    train_seeds = sorted(int(x) for x in np.unique(data["scene_seed"][train_mask]))
    val_seeds = sorted(int(x) for x in np.unique(data["scene_seed"][val_mask]))
    from net.build_net import load_config
    ego = load_config().get("ego", {})
    return {
        "model_type": "clean",
        "architecture": "DriveModel",
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "hidden": int(model.hidden),
        "heads": int(model.heads),
        "spatial_layers": int(getattr(model, "spatial_layers", 2)),
        "input_keys": list(INPUT_KEYS),
        "excluded_inputs": [
            "anchors",
            "expert audit fields",
            "episode outcomes",
        ],
        "action_bounds_name": str(getattr(model, "action_config", {}).get("name", "clean")),
        "action_low": model.action_low.detach().cpu().tolist() if getattr(model, "action_low", None) is not None else CLEAN_BOUNDS.low.tolist(),
        "action_high": model.action_high.detach().cpu().tolist() if getattr(model, "action_high", None) is not None else CLEAN_BOUNDS.high.tolist(),
        "decision": {
            "dt": float(cfg.dt),
            "steps_per_policy": 5,
            "history_frames": HISTORY_FRAMES,
        },
        "ego": dict(getattr(model, "ego_config", {
            "dt": float(ego.get("dt", cfg.dt)),
            "wheelbase": float(ego.get("wheelbase", 2.46894)),
            "max_steer_rad": float(ego.get("max_steer_rad", 0.6981317007977318)),
        })),
        "limits": {
            "hard": dict(getattr(model, "action_config", {}).get("limits", {}).get("hard", asdict(HARD_LIMITS))),
            "comfort": dict(getattr(model, "action_config", {}).get("limits", {}).get("comfort", asdict(COMFORT_LIMITS))),
            "smoothness_hinge": "hard",
        },
        "loss": asdict(cfg),
        "split": {
            "by": "scene_seed",
            "mode": "fixed_val_data" if args.val_data else "scene_seed_fraction",
            "val_frac": float(args.val_frac),
            "seed": int(args.seed),
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
            "n_train_rows": int(train_mask.sum()),
            "n_val_rows": int(val_mask.sum()),
            "val_split_empty": bool(not val_mask.any()),
            "val_data": None if not args.val_data else [str(p) for p in args.val_data],
        },
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "params": count_parameters(model),
        "data": [str(p) for p in args.data],
        "data_max_rows": (
            None
            if not getattr(args, "data_max_rows", None)
            else [int(value) for value in args.data_max_rows]
        ),
        "val_data": None if not args.val_data else [str(p) for p in args.val_data],
        "config_file": getattr(args, "config", None),
        "run_dir": str(getattr(args, "out", "")),
        "sequence": {
            "enabled": bool(args.sequence_batch),
            "scenes_per_batch": int(args.scenes_per_batch),
            "clip_length": int(args.clip_length),
            "tbptt_chunk": int(args.tbptt_chunk),
        },
        "closed_loop": {
            "enabled": bool(args.closed_loop),
            "specs": args.closed_loop_specs,
            "split_manifest": args.closed_loop_split_manifest,
            "limit": int(args.closed_loop_limit),
            "final_limit": int(args.closed_loop_final_limit),
            "every": int(args.closed_loop_every),
            "patience": int(args.closed_loop_patience),
            "max_steps": int(args.closed_loop_max_steps),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "max_grad_norm": float(args.max_grad_norm),
            "seed": int(args.seed),
            "device": str(args.device),
        },
        "selection": (
            "best.pt minimizes the total validation objective (sum of all "
            "logged loss components) on the scene-seed validation split. "
            "Open-loop diagnostic only; no single-MSE or closed-loop claim."
        ),
    }


def _fmt(values: Mapping[str, float], key: str) -> str:
    return f"{float(values.get(key, 0.0)):.5f}"


def run_closed_loop_validation(
    model: DriveModel,
    *,
    specs_path: str,
    split_manifest: str | None,
    limit: int,
    max_steps: int,
    device: str,
) -> Dict:
    """Run fixed val scenes through the real env and LiveBankRunner."""
    from env.metadrive_env import build_env
    from env.scenario.spec import load_specs
    from pipeline.eval import run_episode
    from pipeline.eval.metrics import summarize_rows
    from dataset.splits import verify_split_manifest

    all_specs = load_specs(specs_path)
    if split_manifest:
        verify_split_manifest(
            split_manifest, all_specs, split="val", spec_path=specs_path
        )
    specs = all_specs[: max(0, int(limit))]
    rows = []
    for spec in specs:
        env = build_env(spec, use_render=False, lru_size=0)
        try:
            rows.append(
                run_episode(
                    env, model, spec,
                    controller="exact",
                    max_steps=int(max_steps),
                    device=device,
                    history_frames=HISTORY_FRAMES,
                )
            )
        finally:
            env.close()
    summary = summarize_rows(rows)
    summary["scenes_requested"] = int(limit)
    summary["specs"] = str(specs_path)
    summary["split_manifest"] = None if split_manifest is None else str(split_manifest)
    return {"summary": summary, "rows": rows}


def _closed_loop_score(summary: Mapping, *, max_steps: int = 600) -> tuple:
    """Lexicographic success-first closed-loop ordering.

    A high success rate is never compensated by a lower infeasible count.
    Collision, off-road, route completion and safety diagnostics are
    tie-breakers in that order.
    """
    infeasible = float(summary.get("infeasible_count_mean", 0.0)) / max(1.0, float(max_steps))
    return (
        float(summary.get("success_rate", 0.0)),
        -float(summary.get("collision_rate", 0.0)),
        -float(summary.get("off_road_rate", 0.0)),
        float(summary.get("route_completion_mean", 0.0)),
        -infeasible,
    )


def _score_is_finite(score: tuple) -> bool:
    return all(math.isfinite(float(x)) for x in score)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Schema-v2 BC training for DriveModel"
    )
    p.add_argument(
        "--config",
        default="config/train_bc.yaml",
        help="YAML training config (CLI flags override it)",
    )
    p.add_argument("--data", nargs="+", default=None,
                   help="dirs containing expert_bc.npz (or .npz files)")
    p.add_argument(
        "--data-max-rows",
        nargs="+",
        type=int,
        default=None,
        help="optional per-input row caps; whole episodes are sampled; 0=off",
    )
    p.add_argument(
        "--val-data",
        nargs="+",
        default=None,
        help="optional fixed validation .npz file(s)/directory(ies)",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--sequence-batch", action=argparse.BooleanOptionalAction, default=False,
        help="scene-level ordered training with tensor FIFO (default from config)",
    )
    p.add_argument("--scenes-per-batch", type=int, default=1024)
    p.add_argument("--clip-length", type=int, default=100)
    p.add_argument("--tbptt-chunk", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--init", default=None)
    p.add_argument("--nll-coef", type=float, default=1.0)
    p.add_argument("--w-point", type=float, default=1.0)
    p.add_argument("--w-heading", type=float, default=1.0)
    p.add_argument("--w-velocity", type=float, default=0.2)
    p.add_argument("--w-curvature", type=float, default=0.1)
    p.add_argument("--w-accel", type=float, default=0.1)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--w-privileged", type=float, default=1.0)
    p.add_argument("--w-residual", type=float, default=0.2)
    p.add_argument("--w-shield", type=float, default=3.0)
    p.add_argument("--w-student-failure", type=float, default=2.0)
    p.add_argument("--w-failed-expert", type=float, default=0.2)
    p.add_argument("--w-infeasible", type=float, default=0.2)
    p.add_argument("--brake-weight", type=float, default=0.0)
    p.add_argument("--near-weight", type=float, default=0.0)
    p.add_argument("--source-weight-expert", type=float, default=1.0)
    p.add_argument("--source-weight-other", type=float, default=1.0)
    p.add_argument("--source-weight-replay", type=float, default=1.0)
    p.add_argument("--source-weight-lift", type=float, default=0.5)
    p.add_argument("--residual-xy-thresh", type=float, default=0.5)
    p.add_argument("--residual-dpsi-thresh", type=float, default=0.1)
    p.add_argument(
        "--closed-loop", action=argparse.BooleanOptionalAction, default=False,
        help="run fixed val500 closed-loop validation each epoch",
    )
    p.add_argument("--closed-loop-specs", default="data/specs/val_500.json")
    p.add_argument("--closed-loop-split-manifest", default="data/splits/val_500.json")
    p.add_argument("--closed-loop-limit", type=int, default=50)
    p.add_argument("--closed-loop-final-limit", type=int, default=500)
    p.add_argument("--closed-loop-every", type=int, default=1)
    p.add_argument("--closed-loop-patience", type=int, default=5)
    p.add_argument("--closed-loop-max-steps", type=int, default=600)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_train_args(build_arg_parser(), argv, "config/train_bc.yaml")
    if not args.data:
        raise SystemExit("--data is required (CLI or config)")
    if not args.out:
        raise SystemExit("--out is required (CLI or config)")
    args.out = str(timestamped_run_dir(args.out))
    latest_run_alias(args.out)
    print(f"[bc] run_dir={args.out}")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.scenes_per_batch < 1:
        raise SystemExit("--scenes-per-batch must be >= 1")
    if args.tbptt_chunk < 1:
        raise SystemExit("--tbptt-chunk must be >= 1")
    if args.clip_length < 1:
        raise SystemExit("--clip-length must be >= 1")
    if args.closed_loop_every < 1:
        raise SystemExit("--closed-loop-every must be >= 1")
    if args.closed_loop_limit < 0 or args.closed_loop_final_limit < 0:
        raise SystemExit("--closed-loop limits must be >= 0")
    if args.closed_loop_patience < 1:
        raise SystemExit("--closed-loop-patience must be >= 1")
    if not (0.0 <= args.val_frac < 1.0):
        raise SystemExit("--val-frac must be in [0, 1)")
    if args.data_max_rows and any(int(value) < 0 for value in args.data_max_rows):
        raise SystemExit("--data-max-rows values must be >= 0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    print(f"[bc] config={args.config}")

    cfg = LossConfig(
        nll_coef=args.nll_coef,
        w_point=args.w_point,
        w_heading=args.w_heading,
        w_velocity=args.w_velocity,
        w_curvature=args.w_curvature,
        w_accel=args.w_accel,
        dt=args.dt,
        w_privileged=args.w_privileged,
        w_residual=args.w_residual,
        w_shield=args.w_shield,
        w_student_failure=args.w_student_failure,
        w_failed_expert=args.w_failed_expert,
        w_infeasible=args.w_infeasible,
        brake_weight=args.brake_weight,
        near_weight=args.near_weight,
        source_weight_expert=args.source_weight_expert,
        source_weight_other=args.source_weight_other,
        source_weight_replay=args.source_weight_replay,
        source_weight_lift=args.source_weight_lift,
        residual_xy_thresh=args.residual_xy_thresh,
        residual_dpsi_thresh=args.residual_dpsi_thresh,
    )

    train_data = load_data(
        args.data,
        max_rows=args.data_max_rows,
        seed=args.seed,
    )
    if args.val_data:
        val_data = load_data(args.val_data)
        train_seeds = set(int(x) for x in np.unique(train_data["scene_seed"]))
        val_seeds = set(int(x) for x in np.unique(val_data["scene_seed"]))
        overlap = train_seeds & val_seeds
        if overlap:
            raise SystemExit(f"--data/--val-data scene_seed overlap: {sorted(overlap)[:10]}")
        n_train = len(train_data["action"])
        data = concat_data([train_data, val_data])
        del train_data, val_data
        train_mask = np.zeros(len(data["action"]), dtype=bool)
        train_mask[:n_train] = True
        val_mask = ~train_mask
        print(f"[bc] fixed val_data={args.val_data} scenes={len(val_seeds)}")
    else:
        data = train_data
        train_mask = val_mask = None
    if args.init:
        model = load_model_from_checkpoint(args.init, device, strict=True)
        print(f"[bc] initialized from {args.init}")
    else:
        model = build_net().to(device)
    action_low = model.action_low.detach().cpu().numpy() if getattr(model, "action_low", None) is not None else None
    action_high = model.action_high.detach().cpu().numpy() if getattr(model, "action_high", None) is not None else None
    targets = build_targets(data, cfg, action_low=action_low, action_high=action_high)
    if train_mask is None or val_mask is None:
        train_mask, val_mask = split_masks(data, args.val_frac, args.seed)
    if not val_mask.any():
        print(
            "[bc] WARNING: validation split is empty (not enough scene "
            "seeds); monitoring train split instead"
        )

    config = _make_config(model, cfg, args, data, train_mask, val_mask)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss-only TensorBoard monitoring; no weight histograms or model graph.
    tb_writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))
        print(f"[bc] tensorboard: {out_dir / 'tensorboard'}")
    except Exception as exc:  # pragma: no cover - environment fallback
        print(f"[bc] WARNING: tensorboard disabled: {exc}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    rng = np.random.default_rng(args.seed)
    label_oob = targets["label_oob"]
    oob_stats = {
        "all": float(label_oob.any(axis=1).mean()),
        "train": float(label_oob[train_mask].any(axis=1).mean())
        if train_mask.any()
        else 0.0,
        "val": float(label_oob[val_mask].any(axis=1).mean())
        if val_mask.any()
        else 0.0,
        "per_dim": {
            name: float(label_oob[:, i].mean())
            for i, name in enumerate(("dx", "dy", "dpsi", "v_next"))
        },
    }
    plan_weight = targets["plan_weight"]
    weight_stats = {
        "plan_rows": int((plan_weight > 0).sum()),
        "zero_plan_rows": int((plan_weight <= 0).sum()),
        "plan_weight_sum": float(plan_weight.sum()),
        "expert_plan_valid_rate": float(np.mean(data["expert_plan_valid"] > 0.5)),
        "label_valid_rate": float(np.mean(data["label_valid"] > 0.5)),
        "residual_downweight_rows": int(
            np.sum(
                (np.asarray(data["label_source"], dtype=np.float32) >= 0.5)
                & np.isfinite(np.asarray(data["residual_xy_norm"], dtype=np.float32))
                & np.isfinite(np.asarray(data["residual_dpsi"], dtype=np.float32))
                & (
                    (np.asarray(data["residual_xy_norm"], dtype=np.float32)
                     > cfg.residual_xy_thresh)
                    | (
                        np.abs(np.asarray(data["residual_dpsi"], dtype=np.float32))
                        > cfg.residual_dpsi_thresh
                    )
                )
            )
        ),
        "shield_rows": int(np.sum(np.asarray(data.get("shield", np.zeros(len(data["action"])))) > 0.5)),
        "student_failure_rows": int(
            np.sum(np.asarray(data.get("student_failure", np.zeros(len(data["action"])))) > 0.5)
        ),
        "intervened_rows_legacy": int(
            np.sum(np.asarray(data.get("intervened", np.zeros(len(data["action"])))) > 0.5)
        ),
        "failed_expert_rows": int(np.sum(np.asarray(data["expert_plan_valid"]) <= 0.5)),
        "failed_expert_terminal_rows": int(
            np.sum(np.asarray(data.get("terminal_failed", np.zeros(len(data["action"])))) > 0.5)
            if "terminal_failed" in data
            else np.sum((np.asarray(data["expert_plan_valid"]) <= 0.5)
                        & (np.asarray(data.get("done", np.zeros(len(data["action"])))) > 0.5))
        ),
    }

    print(
        f"[bc] rows={len(data['action'])} "
        f"train={int(train_mask.sum())} val={int(val_mask.sum())} "
        f"train_scenes={len(config['split']['train_seeds'])} "
        f"val_scenes={len(config['split']['val_seeds'])} "
        f"params={config['params']} device={device}"
    )
    print(
        f"[bc] label_oob_rate all={oob_stats['all']:.5f} "
        f"train={oob_stats['train']:.5f} val={oob_stats['val']:.5f} "
        f"plan_rows={weight_stats['plan_rows']}"
    )

    history = []
    best_score = math.inf
    best_epoch = -1
    best_closed_score = None
    best_closed_epoch = -1
    closed_loop_wait = 0
    stop_reason = None
    closed_dir = out_dir / "closed_loop"
    if args.closed_loop:
        closed_dir.mkdir(parents=True, exist_ok=True)
    eval_mask = val_mask if val_mask.any() else train_mask
    selection_split = "val" if val_mask.any() else "train"
    closed_loop_selectable = bool(args.closed_loop)
    if args.sequence_batch:
        from pipeline.train.scene_batches import make_scene_batches
        eval_scene_batches = make_scene_batches(
            data, np.flatnonzero(eval_mask), args.scenes_per_batch,
            clip_length=args.clip_length,
        )
        print(
            f"[bc] sequence_batch scenes_per_batch={args.scenes_per_batch} "
            f"clip_length={args.clip_length} tbptt_chunk={args.tbptt_chunk} "
            f"eval_scene_batches={len(eval_scene_batches)}"
        )
    else:
        eval_scene_batches = None

    for epoch in range(1, args.epochs + 1):
        if args.sequence_batch:
            from pipeline.train.scene_batches import make_scene_batches
            train_scene_batches = make_scene_batches(
                data, np.flatnonzero(train_mask), args.scenes_per_batch,
                rng=rng, clip_length=args.clip_length,
            )
            train_metrics, train_mae, train_plan_rows = run_scene_epoch(
                model, data, targets, train_scene_batches, cfg, device,
                optimizer=optimizer, max_grad_norm=args.max_grad_norm,
                tbptt_chunk=args.tbptt_chunk,
            )
            eval_metrics, eval_mae, eval_plan_rows = run_scene_epoch(
                model, data, targets, eval_scene_batches or [], cfg, device,
                optimizer=None, tbptt_chunk=args.tbptt_chunk,
            )
        else:
            train_metrics, train_mae, train_plan_rows = run_epoch(
                model,
                data,
                targets,
                np.flatnonzero(train_mask),
                cfg,
                device,
                args.batch_size,
                optimizer=optimizer,
                rng=rng,
                max_grad_norm=args.max_grad_norm,
            )
            eval_metrics, eval_mae, eval_plan_rows = run_epoch(
                model,
                data,
                targets,
                np.flatnonzero(eval_mask),
                cfg,
                device,
                args.batch_size,
                optimizer=None,
            )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "eval": eval_metrics,
            "train_loss": float(train_metrics.get("total", math.nan)),
            "val_loss": (
                float(eval_metrics.get("total", math.nan))
                if val_mask.any()
                else None
            ),
            "train_mae_physical": [float(x) for x in train_mae],
            "eval_mae_physical": [float(x) for x in eval_mae],
            "train_plan_rows": int(train_plan_rows),
            "eval_plan_rows": int(eval_plan_rows),
            "eval_split": selection_split,
        }
        history.append(record)

        if (
            args.closed_loop
            and int(args.closed_loop_limit) > 0
            and epoch % int(args.closed_loop_every) == 0
        ):
            try:
                closed = run_closed_loop_validation(
                    model,
                    specs_path=args.closed_loop_specs,
                    split_manifest=args.closed_loop_split_manifest,
                    limit=args.closed_loop_limit,
                    max_steps=args.closed_loop_max_steps,
                    device=str(device),
                )
            except Exception as exc:
                # Keep the offline checkpoint/metrics path alive when the
                # simulator/headless stack is unavailable; do not select a
                # closed-loop checkpoint from an empty/failed evaluation.
                record["closed_loop_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[bc] WARNING closed-loop unavailable: {record['closed_loop_error']}", flush=True)
                closed = None
            if closed is not None:
                closed_summary = closed["summary"]
                score = _closed_loop_score(closed_summary, max_steps=args.closed_loop_max_steps)
                if int(closed_summary.get("n", 0)) <= 0 or not _score_is_finite(score):
                    record["closed_loop_error"] = "empty or non-finite closed-loop summary"
                    print(f"[bc] WARNING {record['closed_loop_error']}", flush=True)
                else:
                    record["closed_loop"] = closed_summary
                    record["closed_loop_score"] = list(score)
                    (closed_dir / f"epoch_{epoch:04d}.json").write_text(
                        json.dumps(_jsonable({"epoch": epoch, "score": score, **closed}), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    if closed_loop_selectable:
                        if best_closed_score is None or score > best_closed_score:
                            best_closed_score = score
                            best_closed_epoch = epoch
                            closed_loop_wait = 0
                            _save_checkpoint(
                                out_dir / "best_closed_loop.pt", model, config, epoch, record
                            )
                        else:
                            closed_loop_wait += 1
                    print(
                        f"[bc] closed-loop epoch {epoch:03d} score={score} "
                        f"success={closed_summary.get('success_rate', float('nan')):.4f} "
                        f"collision={closed_summary.get('collision_rate', float('nan')):.4f} "
                        f"route={closed_summary.get('route_completion_mean', float('nan')):.4f} "
                        f"best_epoch={best_closed_epoch} wait={closed_loop_wait}"
                    )

        if tb_writer is not None:
            for split_name, split_metrics in (("train", train_metrics), ("eval", eval_metrics)):
                for key, value in split_metrics.items():
                    try:
                        scalar = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(scalar):
                        tb_writer.add_scalar(f"{split_name}_loss/{key}", scalar, epoch)
            if "closed_loop" in record:
                for key, value in record["closed_loop"].items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        tb_writer.add_scalar(f"closed_loop/{key}", float(value), epoch)
            tb_writer.flush()

        print(
            f"[bc] epoch {epoch:03d} "
            f"train total={_fmt(train_metrics, 'total')} "
            f"nll={_fmt(train_metrics, 'nll')} "
            f"point={_fmt(train_metrics, 'point')} "
            f"heading={_fmt(train_metrics, 'heading')} "
            f"velocity={_fmt(train_metrics, 'velocity')} "
            f"curv={_fmt(train_metrics, 'curvature')} "
            f"accel={_fmt(train_metrics, 'accel')} "
            f"mae={np.round(train_mae, 4).tolist()} | "
            f"eval/{selection_split} total={_fmt(eval_metrics, 'total')} "
            f"nll={_fmt(eval_metrics, 'nll')} "
            f"point={_fmt(eval_metrics, 'point')} "
            f"heading={_fmt(eval_metrics, 'heading')} "
            f"velocity={_fmt(eval_metrics, 'velocity')} "
            f"mae={np.round(eval_mae, 4).tolist()}"
        )

        _save_checkpoint(
            out_dir / "last.pt", model, config, epoch, record
        )
        score = float(eval_metrics.get("total", math.inf))
        if score < best_score:
            best_score = score
            best_epoch = epoch
            _save_checkpoint(
                out_dir / "best.pt", model, config, epoch, record
            )
        if closed_loop_selectable and closed_loop_wait >= int(args.closed_loop_patience):
            stop_reason = f"closed_loop_patience_{int(args.closed_loop_patience)}"
            print(f"[bc] early stop: {stop_reason}", flush=True)
            break

    closed_selected = (out_dir / "best_closed_loop.pt").exists()
    metrics = {
        "config": config,
        "history": history,
        "best": {
            "epoch": best_epoch,
            "objective": best_score,
            "split": selection_split,
            "note": (
                "Open-loop selection by the total weighted validation objective; "
                "all components and physical MAE are logged alongside. "
                "Not a closed-loop claim."
            ),
        },
        "closed_loop": {
            "enabled": bool(args.closed_loop),
            "specs": args.closed_loop_specs,
            "split_manifest": args.closed_loop_split_manifest,
            "limit_per_epoch": int(args.closed_loop_limit),
            "every": int(args.closed_loop_every),
            "patience": int(args.closed_loop_patience),
            "best_epoch": best_closed_epoch,
            "selectable": bool(args.closed_loop and closed_selected),
            "selected_checkpoint": str(out_dir / "best_closed_loop.pt") if closed_selected else None,
            "best_score": None if best_closed_score is None else list(best_closed_score),
            "stopped_by": stop_reason,
            "history": [h.get("closed_loop", {}).get("success_rate") for h in history],
        },
        "label_oob_rate": oob_stats["all"],
        "label_oob_rate_by_split": {
            "train": oob_stats["train"],
            "val": oob_stats["val"],
        },
        "label_oob_rate_per_dim": oob_stats["per_dim"],
        "plan_weights": weight_stats,
        "command": {
            "argv": list(argv) if argv is not None else None,
        },
        "assumptions": [
            "Train/val split is by scene_seed; episode ids never define the split.",
            "Only od/ld/nav/ego/mask/hist_valid are model inputs; IDs and "
            "expert audit fields stay side arrays.",
            "Normalized targets outside [0,1] are clipped for the truncated "
            "Normal NLL and counted by label_oob_rate.",
            "Failed expert episodes downweight only the terminal segment; "
            "earlier segments keep their label unless label_valid/feasibility "
            "or bounds fail.  Shield is separate from expert replay.",
            "Per-dimension physical MAE is plan-weight-weighted over rows with "
            "plan_weight > 0.",
            "Smoothness hinges use the collector's provisional HARD_LIMITS "
            "and are gated to moving expert labels.",
        ],
    }
    closed_loop_final = None
    if (
        args.closed_loop
        and int(args.closed_loop_final_limit) > 0
        and (out_dir / "best_closed_loop.pt").exists()
    ):
        try:
            best_payload = torch.load(out_dir / "best_closed_loop.pt", map_location="cpu", weights_only=False)
            model.load_state_dict(best_payload["model"], strict=True)
            final_closed = run_closed_loop_validation(
                model,
                specs_path=args.closed_loop_specs,
                split_manifest=args.closed_loop_split_manifest,
                limit=int(args.closed_loop_final_limit),
                max_steps=args.closed_loop_max_steps,
                device=str(device),
            )
            closed_loop_final = final_closed["summary"]
            final_score = _closed_loop_score(closed_loop_final, max_steps=args.closed_loop_max_steps)
            if int(closed_loop_final.get("n", 0)) <= 0 or not _score_is_finite(final_score):
                raise RuntimeError("final closed-loop summary is empty or non-finite")
            (closed_dir / "best_final.json").write_text(
                json.dumps(_jsonable({"score": final_score, **final_closed}), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[bc] closed-loop final best_epoch={best_closed_epoch} "
                f"n={closed_loop_final.get('n')} success={closed_loop_final.get('success_rate')} "
                f"collision={closed_loop_final.get('collision_rate')}",
                flush=True,
            )
        except Exception as exc:
            closed_loop_final = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[bc] WARNING final closed-loop unavailable: {closed_loop_final['error']}", flush=True)
    metrics["closed_loop"]["final"] = closed_loop_final
    if tb_writer is not None:
        tb_writer.close()
    (out_dir / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[bc] wrote {out_dir / 'best.pt'} (epoch {best_epoch}, "
        f"{selection_split} objective {best_score:.6f}) and {out_dir / 'last.pt'}"
    )
    if args.closed_loop and not closed_selected:
        print(
            "[bc] ERROR: closed-loop selection was enabled but no "
            "best_closed_loop.pt was produced; downstream must not consume "
            "this run as a closed-loop-selected checkpoint.",
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
