"""Schema-v2 BC training for the clean planner.

This is the *new* clean-path trainer.  It deliberately does not touch the
legacy ``pipeline/train_bc.py`` or ``net/clean_model.py`` contracts.

Data contract
-------------
Each input directory (or ``.npz`` file) must contain the arrays written by
``dataset.collect_v2``: all ``HISTORY_KEYS`` plus the measured ``action`` row,
``scene_seed``, ``episode_id`` and outcome columns.  Only the model feature
keys below enter ``CleanPlanner``; IDs, audit fields and outcomes are side
arrays and never model inputs::

    od, od_mask, ld, ld_mask, nav, nav_mask, ego, ego_mask, hist_valid

The train/validation split is by **scene seed**, never by episode id.  The
same scene can be re-collected as multiple episodes (BC + DAgger), so an
episode split would leak scenes across the split boundary.

Loss
----
* truncated-Normal NLL on the normalized expert action (``[0,1]`` under
  ``NEW_BOUNDS``), with out-of-bounds labels clipped for the NLL only;
* point / heading / velocity losses on the *physical* denormalized action;
* curvature / longitudinal-acceleration / jerk-proxy hinge smoothness on the
  predicted action, active only on moving actions;
* value / risk / cost-value supervision from episode outcome and row comfort;
* auxiliary lane-role CE and OD gap/TTC/urgent heads.

Per-row plan weights implement the failure/privilege policy:

* ``expert_plan_valid == 0`` -> 0 (failed expert episodes never teach the plan);
* ``label_valid == 0`` -> 0;
* source expert (``source == 0``) -> ``--source-weight-expert`` (default 1);
* other sources -> ``--source-weight-other`` (default 1, DAgger hook);
* ``privileged_expert_sample`` -> ``--w-privileged`` (default 1, no change);
* ``residual_xy_norm > 0.5`` or ``|residual_dpsi| > 0.1`` -> ``--w-residual``;
* ``intervened`` rows -> ``--w-intervened`` (default 3).

Value and risk losses are *not* gated by the plan weight: failed expert
episodes still provide useful safety labels.
"""

from __future__ import annotations

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

from dataset.collect_v2 import (
    CLEAN_COMFORT_LIMITS,
    CLEAN_HARD_LIMITS,
    HISTORY_KEYS,
)
from dataset.splits import assert_disjoint, split_by_scene_seed
from env.obs_v2 import (
    EGO_INDEX,
    HISTORY_FRAMES,
    LD_INDEX,
    OD_INDEX,
    SCHEMA_NAME,
    SCHEMA_VERSION,
)
from net.action_space import COMFORT_LIMITS, HARD_LIMITS, NEW_BOUNDS
from net.clean_model import CleanPlanner, count_parameters
from pipeline.distributions import truncated_normal_nll

#: Feature keys fed to CleanPlanner.  No IDs / audit / outcome columns here.
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
)

#: Per-row arrays kept in memory (features + labels/weights only).
KEEP_KEYS = set(INPUT_KEYS) | {
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

RISK_NAMES = ("collision", "off_road", "wrong_route", "stuck", "comfort_violation")
PLAN_KEYS = {"nll", "point", "heading", "velocity", "curvature", "accel"}

#: Columns that must be available (hence required above) or can be defaulted.
_DEFAULT_ONE_SCALARS = ("label_valid", "expert_plan_valid")
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
)


@dataclass
class LossConfig:
    """Loss and plan-weight configuration (serialized into checkpoints)."""

    nll_coef: float = 1.0
    value_coef: float = 0.5
    risk_coef: float = 0.5
    cost_coef: float = 0.1
    aux_coef: float = 0.2

    # physical-action plan terms
    w_point: float = 1.0
    w_heading: float = 1.0
    w_velocity: float = 0.2
    w_curvature: float = 0.1
    w_accel: float = 0.1

    # auxiliary heads (all inside aux_coef)
    w_lane_role: float = 1.0
    w_gap: float = 1.0
    w_ttc: float = 1.0
    w_urgent: float = 1.0

    # targets / units
    gamma: float = 0.95
    dt: float = 0.5
    ttc_cap: float = 10.0
    urgent_ttc: float = 2.0

    # per-row plan weights
    w_privileged: float = 1.0
    w_residual: float = 0.2
    w_intervened: float = 3.0
    source_weight_expert: float = 1.0
    source_weight_other: float = 1.0
    residual_xy_thresh: float = 0.5
    residual_dpsi_thresh: float = 0.1


# ---------------------------------------------------------------------------
# Data loading / targets
# ---------------------------------------------------------------------------
def _resolve_npz(path: Path) -> Path:
    if path.is_dir():
        return path / "expert_bc_v2.npz"
    return path


def _default_value(key: str, n: int) -> np.ndarray:
    if key == "residual_xy":
        return np.zeros((n, 2), dtype=np.float32)
    if key in _DEFAULT_ONE_SCALARS:
        return np.ones(n, dtype=np.int8)
    if key in _DEFAULT_MINUS_ONE_SCALARS:
        return np.full(n, -1, dtype=np.int8)
    if key == _DEFAULT_PRIVILEGED:
        return np.zeros(n, dtype=np.int8)
    return np.zeros(n, dtype=np.float32)


def load_data(paths: Sequence[str | Path]) -> Dict[str, np.ndarray]:
    """Load and concatenate one or more ``expert_bc_v2.npz`` datasets."""
    if not paths:
        raise ValueError("load_data: no input paths")
    parts = []
    for raw in paths:
        npz_path = _resolve_npz(Path(raw))
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
                raise ValueError(f"{npz_path}: missing required row keys {missing_rows}")
            part = {
                key: np.asarray(z[key])
                for key in files
                if key in KEEP_KEYS
            }
        action = np.asarray(part["action"], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 4:
            raise ValueError(f"{npz_path}: action must be [N,4], got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{npz_path}: action contains NaN/Inf")
        n = int(action.shape[0])
        if n == 0:
            raise ValueError(f"{npz_path}: zero rows")
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

    merged["action"] = np.asarray(merged["action"], dtype=np.float32)
    merged["scene_seed"] = np.asarray(merged["scene_seed"], dtype=np.int64)
    merged["episode_id"] = np.asarray(merged["episode_id"], dtype=np.int64)
    merged["step"] = np.asarray(merged["step"], dtype=np.int64)
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


def _outcome_targets(
    data: Mapping[str, np.ndarray], gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value / 5 risk labels / scalar cost targets from episode outcomes."""
    n = len(data["action"])
    episode = np.asarray(data["episode_id"], dtype=np.int64)
    step = np.asarray(data["step"], dtype=np.int64)

    risk = np.zeros((n, 5), dtype=np.float32)
    for i, name in enumerate(RISK_NAMES[:4]):
        risk[:, i] = np.asarray(data[name], dtype=np.float32) > 0.5
    comfortable = np.asarray(data["comfortable"], dtype=np.float32)
    comfort_violation = np.where(comfortable >= 0.0, 1.0 - comfortable, 0.0)
    risk[:, 4] = np.clip(comfort_violation, 0.0, 1.0).astype(np.float32)

    value = np.zeros(n, dtype=np.float32)
    for ep in np.unique(episode):
        idx = np.flatnonzero(episode == ep)
        if len(idx) == 0:
            continue
        success = bool(np.max(np.asarray(data["success"])[idx]) > 0.5)
        collision = bool(np.max(np.asarray(data["collision"])[idx]) > 0.5)
        off_road = bool(np.max(np.asarray(data["off_road"])[idx]) > 0.5)
        if success:
            outcome = 1.0
        elif collision or off_road:
            outcome = -1.0
        else:
            outcome = 0.0
        last_step = int(np.max(step[idx]))
        remaining = np.maximum(last_step - step[idx], 0)
        value[idx] = outcome * np.power(float(gamma), remaining.astype(np.float64))
        # Episode-level terminal risk labels are broadcast to every row.
        for i, name in enumerate(RISK_NAMES[:4]):
            risk[idx, i] = float(np.max(np.asarray(data[name])[idx]) > 0.5)
    cost = risk[:, :4].sum(axis=1) + risk[:, 4]
    return value.astype(np.float32), risk, cost.astype(np.float32)


def _plan_weights(data: Mapping[str, np.ndarray], cfg: LossConfig) -> np.ndarray:
    """Per-row plan weight implementing failure / privilege / DAgger policy."""
    n = len(data["action"])
    w = np.ones(n, dtype=np.float32)

    expert_plan_valid = np.asarray(data["expert_plan_valid"], dtype=np.float32)
    label_valid = np.asarray(data["label_valid"], dtype=np.float32)
    w *= expert_plan_valid > 0.5
    w *= label_valid > 0.5

    source = np.asarray(data["source"], dtype=np.float32)
    w *= np.where(
        source == 0.0, float(cfg.source_weight_expert), float(cfg.source_weight_other)
    )

    privileged = np.asarray(data["privileged_expert_sample"], dtype=np.float32) > 0.5
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
    residual = (residual_xy > cfg.residual_xy_thresh) | (
        residual_dpsi > cfg.residual_dpsi_thresh
    )
    w *= np.where(residual, float(cfg.w_residual), 1.0)

    if "intervened" in data:
        intervened = np.asarray(data["intervened"], dtype=np.float32) > 0.5
        w *= np.where(intervened, float(cfg.w_intervened), 1.0)

    return np.clip(w, 0.0, None).astype(np.float32)


def build_targets(
    data: Mapping[str, np.ndarray], cfg: LossConfig
) -> Dict[str, np.ndarray]:
    """Precompute normalized/physical actions, value/risk/cost and weights."""
    action = np.asarray(data["action"], dtype=np.float32)
    low = NEW_BOUNDS.low.astype(np.float32)
    high = NEW_BOUNDS.high.astype(np.float32)
    width = np.maximum(high - low, 1e-6)
    action_norm = (action - low) / width
    label_oob = (action_norm < 0.0) | (action_norm > 1.0)
    action_norm_clipped = np.clip(action_norm, 0.0, 1.0)

    value, risk, cost = _outcome_targets(data, cfg.gamma)
    plan_weight = _plan_weights(data, cfg)

    return {
        "action_phys": action.copy(),
        "action_norm": action_norm.astype(np.float32),
        "action_norm_clipped": action_norm_clipped.astype(np.float32),
        "label_oob": label_oob.astype(bool),
        "value": value,
        "risk": risk,
        "cost": cost,
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
        arr = np.ascontiguousarray(data[key][idx], dtype=np.float32)
        batch[key] = torch.from_numpy(arr).to(device)
    for key in (
        "action_phys",
        "action_norm",
        "action_norm_clipped",
        "value",
        "risk",
        "cost",
        "plan_weight",
    ):
        arr = np.ascontiguousarray(targets[key][idx], dtype=np.float32)
        batch[key] = torch.from_numpy(arr).to(device)
    return batch


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    weights = weights.float()
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return _weighted_mean(values, mask.float())


def _denorm(mu: Tensor) -> Tensor:
    low = torch.tensor(NEW_BOUNDS.low, dtype=mu.dtype, device=mu.device)
    high = torch.tensor(NEW_BOUNDS.high, dtype=mu.dtype, device=mu.device)
    return low + mu * (high - low)


def compute_losses(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    cfg: LossConfig,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Return ``(total, components)`` for one batch.

    Plan components use ``batch['plan_weight']`` as their row weight; value,
    risk, cost and auxiliary components use all rows.
    """
    mu = outputs["action_mu"]
    logstd = outputs["action_logstd"]
    weight = batch["plan_weight"]
    target_phys = batch["action_phys"]
    pred_phys = _denorm(mu)

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

    curvature_hinge = (
        F.relu(kappa.abs() - float(CLEAN_HARD_LIMITS.kappa_max)).pow(2)
        + F.relu(
            kappa_rate_proxy.abs() - float(CLEAN_HARD_LIMITS.kappa_rate_max)
        ).pow(2)
    )
    accel_hinge = (
        F.relu(float(CLEAN_HARD_LIMITS.a_long_min) - a_long).pow(2)
        + F.relu(a_long - float(CLEAN_HARD_LIMITS.a_long_max)).pow(2)
        + F.relu(
            jerk_proxy.abs() - float(CLEAN_HARD_LIMITS.jerk_long_max)
        ).pow(2)
    )
    curvature = _weighted_mean(curvature_hinge, smooth_weight)
    accel = _weighted_mean(accel_hinge, smooth_weight)

    # ---- value / risk / cost ------------------------------------------------
    value = F.mse_loss(outputs["value"].squeeze(-1), batch["value"])
    risk = F.binary_cross_entropy_with_logits(outputs["risk"], batch["risk"])
    cost = F.mse_loss(outputs["cost_value"].squeeze(-1), batch["cost"])

    # ---- auxiliary: lane role + OD gap/TTC/urgent ---------------------------
    aux = outputs["aux"]
    ld = batch["ld"]
    lane_role = ld[:, -1, :, 0, LD_INDEX["lane_role"]].long().clamp(0, 3)
    lane_valid = (batch["ld_mask"][:, -1] > 0.5).any(dim=-1)
    lane_ce = F.cross_entropy(
        aux["lane_role_logits"].reshape(-1, 4),
        lane_role.reshape(-1),
        reduction="none",
    ).reshape(lane_role.shape)
    lane_role_loss = _masked_mean(lane_ce, lane_valid)

    od = batch["od"][:, -1]
    od_valid = batch["od_mask"][:, -1] > 0.5
    dx = od[..., OD_INDEX["dx"]]
    vx = od[..., OD_INDEX["vx_rel"]]
    length = od[..., OD_INDEX["length"]]
    ahead = od_valid & (dx > 0.0)

    # Gap is bumper-to-bumper longitudinal distance (ego front to object rear
    # approximated by the object center minus half its length).
    gap_target = (dx - 0.5 * length).clamp_min(0.0)
    gap_loss = _masked_mean(
        F.smooth_l1_loss(aux["od_gap"], gap_target, reduction="none", beta=1.0),
        ahead,
    )

    closing = ahead & (vx < -0.1)
    ttc_target = (gap_target / (-vx).clamp_min(0.1)).clamp_max(float(cfg.ttc_cap))
    ttc_loss = _masked_mean(
        F.smooth_l1_loss(aux["od_ttc"], ttc_target, reduction="none", beta=0.5),
        closing,
    )

    ttc_all = torch.where(
        vx < -0.1,
        gap_target / (-vx).clamp_min(0.1),
        torch.full_like(gap_target, float(cfg.ttc_cap)),
    )
    urgent_target = (ttc_all < float(cfg.urgent_ttc)).float()
    urgent_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(
            aux["od_urgent"], urgent_target, reduction="none"
        ),
        ahead,
    )

    aux_loss = (
        float(cfg.w_lane_role) * lane_role_loss
        + float(cfg.w_gap) * gap_loss
        + float(cfg.w_ttc) * ttc_loss
        + float(cfg.w_urgent) * urgent_loss
    )

    total = (
        float(cfg.nll_coef) * nll
        + float(cfg.w_point) * point
        + float(cfg.w_heading) * heading
        + float(cfg.w_velocity) * velocity
        + float(cfg.w_curvature) * curvature
        + float(cfg.w_accel) * accel
        + float(cfg.value_coef) * value
        + float(cfg.risk_coef) * risk
        + float(cfg.cost_coef) * cost
        + float(cfg.aux_coef) * aux_loss
    )

    components = {
        "total": total,
        "nll": nll,
        "point": point,
        "heading": heading,
        "velocity": velocity,
        "curvature": curvature,
        "accel": accel,
        "value": value,
        "risk": risk,
        "cost": cost,
        "aux": aux_loss,
        "lane_role": lane_role_loss,
        "od_gap": gap_loss,
        "od_ttc": ttc_loss,
        "od_urgent": urgent_loss,
    }
    return total, components


def physical_mae(outputs: Mapping[str, Tensor], batch: Mapping[str, Tensor]) -> Tuple[Tensor, Tensor]:
    """Plan-weighted per-dimension physical MAE and its weight sum."""
    pred_phys = _denorm(outputs["action_mu"])
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
    model: CleanPlanner,
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
            _loss, components = compute_losses(outputs, batch, cfg)

            if training:
                optimizer.zero_grad(set_to_none=True)
                _loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            plan_sum = float(batch["plan_weight"].sum().detach())
            _accumulate(sums, denoms, components, plan_sum, len(idx))

            mae, wsum = physical_mae(outputs, batch)
            mae_sum += mae.detach().double().cpu()
            mae_weight += wsum.detach().double().cpu()
            n_plan_rows += int((batch["plan_weight"] > 0).sum())

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
    model: CleanPlanner,
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
    model: CleanPlanner,
    cfg: LossConfig,
    args: argparse.Namespace,
    data: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> Dict:
    train_seeds = sorted(int(x) for x in np.unique(data["scene_seed"][train_mask]))
    val_seeds = sorted(int(x) for x in np.unique(data["scene_seed"][val_mask]))
    return {
        "model_type": "clean",
        "architecture": "CleanPlanner",
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "hidden": int(model.hidden),
        "heads": int(model.heads),
        "input_keys": list(INPUT_KEYS),
        "excluded_inputs": [
            "entity ids",
            "anchors",
            "expert audit fields",
            "episode outcomes",
        ],
        "action_bounds_name": NEW_BOUNDS.name,
        "action_low": NEW_BOUNDS.low.tolist(),
        "action_high": NEW_BOUNDS.high.tolist(),
        "decision": {
            "dt": float(cfg.dt),
            "steps_per_policy": 5,
            "history_frames": HISTORY_FRAMES,
        },
        "limits": {
            "hard": asdict(HARD_LIMITS),
            "comfort": asdict(COMFORT_LIMITS),
            "clean_hard": asdict(CLEAN_HARD_LIMITS),
            "clean_comfort": asdict(CLEAN_COMFORT_LIMITS),
            "smoothness_hinge": "clean_hard",
        },
        "loss": asdict(cfg),
        "split": {
            "by": "scene_seed",
            "val_frac": float(args.val_frac),
            "seed": int(args.seed),
            "train_seeds": train_seeds,
            "val_seeds": val_seeds,
            "n_train_rows": int(train_mask.sum()),
            "n_val_rows": int(val_mask.sum()),
            "val_split_empty": bool(not val_mask.any()),
        },
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "params": count_parameters(model),
        "data": [str(p) for p in args.data],
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Schema-v2 BC training for CleanPlanner"
    )
    p.add_argument("--data", nargs="+", required=True,
                   help="dirs containing expert_bc_v2.npz (or .npz files)")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
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
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--risk-coef", type=float, default=0.5)
    p.add_argument("--cost-coef", type=float, default=0.1)
    p.add_argument("--aux-coef", type=float, default=0.2)
    p.add_argument("--w-point", type=float, default=1.0)
    p.add_argument("--w-heading", type=float, default=1.0)
    p.add_argument("--w-velocity", type=float, default=0.2)
    p.add_argument("--w-curvature", type=float, default=0.1)
    p.add_argument("--w-accel", type=float, default=0.1)
    p.add_argument("--w-lane-role", type=float, default=1.0)
    p.add_argument("--w-gap", type=float, default=1.0)
    p.add_argument("--w-ttc", type=float, default=1.0)
    p.add_argument("--w-urgent", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--ttc-cap", type=float, default=10.0)
    p.add_argument("--urgent-ttc", type=float, default=2.0)
    p.add_argument("--w-privileged", type=float, default=1.0)
    p.add_argument("--w-residual", type=float, default=0.2)
    p.add_argument("--w-intervened", type=float, default=3.0)
    p.add_argument("--source-weight-expert", type=float, default=1.0)
    p.add_argument("--source-weight-other", type=float, default=1.0)
    p.add_argument("--residual-xy-thresh", type=float, default=0.5)
    p.add_argument("--residual-dpsi-thresh", type=float, default=0.1)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if not (0.0 <= args.val_frac < 1.0):
        raise SystemExit("--val-frac must be in [0, 1)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    cfg = LossConfig(
        nll_coef=args.nll_coef,
        value_coef=args.value_coef,
        risk_coef=args.risk_coef,
        cost_coef=args.cost_coef,
        aux_coef=args.aux_coef,
        w_point=args.w_point,
        w_heading=args.w_heading,
        w_velocity=args.w_velocity,
        w_curvature=args.w_curvature,
        w_accel=args.w_accel,
        w_lane_role=args.w_lane_role,
        w_gap=args.w_gap,
        w_ttc=args.w_ttc,
        w_urgent=args.w_urgent,
        gamma=args.gamma,
        dt=args.dt,
        ttc_cap=args.ttc_cap,
        urgent_ttc=args.urgent_ttc,
        w_privileged=args.w_privileged,
        w_residual=args.w_residual,
        w_intervened=args.w_intervened,
        source_weight_expert=args.source_weight_expert,
        source_weight_other=args.source_weight_other,
        residual_xy_thresh=args.residual_xy_thresh,
        residual_dpsi_thresh=args.residual_dpsi_thresh,
    )

    data = load_data(args.data)
    targets = build_targets(data, cfg)
    train_mask, val_mask = split_masks(data, args.val_frac, args.seed)
    if not val_mask.any():
        print(
            "[bc_v2] WARNING: validation split is empty (not enough scene "
            "seeds); monitoring train split instead"
        )

    model = CleanPlanner().to(device)
    if args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state)
        print(f"[bc_v2] initialized from {args.init}")

    config = _make_config(model, cfg, args, data, train_mask, val_mask)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
                (np.asarray(data["residual_xy_norm"]) > cfg.residual_xy_thresh)
                | (
                    np.abs(np.asarray(data["residual_dpsi"]))
                    > cfg.residual_dpsi_thresh
                )
            )
        ),
        "intervened_rows": int(
            np.sum(np.asarray(data["intervened"]) > 0.5)
        ),
    }

    print(
        f"[bc_v2] rows={len(data['action'])} "
        f"train={int(train_mask.sum())} val={int(val_mask.sum())} "
        f"train_scenes={len(config['split']['train_seeds'])} "
        f"val_scenes={len(config['split']['val_seeds'])} "
        f"params={config['params']} device={device}"
    )
    print(
        f"[bc_v2] label_oob_rate all={oob_stats['all']:.5f} "
        f"train={oob_stats['train']:.5f} val={oob_stats['val']:.5f} "
        f"plan_rows={weight_stats['plan_rows']}"
    )

    history = []
    best_score = math.inf
    best_epoch = -1
    eval_mask = val_mask if val_mask.any() else train_mask
    selection_split = "val" if val_mask.any() else "train"

    for epoch in range(1, args.epochs + 1):
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

        print(
            f"[bc_v2] epoch {epoch:03d} "
            f"train total={_fmt(train_metrics, 'total')} "
            f"nll={_fmt(train_metrics, 'nll')} "
            f"point={_fmt(train_metrics, 'point')} "
            f"heading={_fmt(train_metrics, 'heading')} "
            f"velocity={_fmt(train_metrics, 'velocity')} "
            f"curv={_fmt(train_metrics, 'curvature')} "
            f"accel={_fmt(train_metrics, 'accel')} "
            f"value={_fmt(train_metrics, 'value')} "
            f"risk={_fmt(train_metrics, 'risk')} "
            f"cost={_fmt(train_metrics, 'cost')} "
            f"aux={_fmt(train_metrics, 'aux')} "
            f"mae={np.round(train_mae, 4).tolist()} | "
            f"eval/{selection_split} total={_fmt(eval_metrics, 'total')} "
            f"nll={_fmt(eval_metrics, 'nll')} "
            f"point={_fmt(eval_metrics, 'point')} "
            f"heading={_fmt(eval_metrics, 'heading')} "
            f"velocity={_fmt(eval_metrics, 'velocity')} "
            f"value={_fmt(eval_metrics, 'value')} "
            f"risk={_fmt(eval_metrics, 'risk')} "
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

    metrics = {
        "config": config,
        "history": history,
        "best": {
            "epoch": best_epoch,
            "objective": best_score,
            "split": selection_split,
            "note": (
                "Selection by the total weighted validation objective; all "
                "components and physical MAE are logged alongside. Open-loop "
                "diagnostic only, not a closed-loop claim."
            ),
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
            "Failed expert episodes (expert_plan_valid=0) have zero plan "
            "weight but still supervise value/risk/cost/aux.",
            "Per-dimension physical MAE is plan-weight-weighted over rows with "
            "plan_weight > 0.",
            "Smoothness hinges use the collector's provisional CLEAN_HARD_LIMITS "
            "and are gated to moving expert labels.",
        ],
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[bc_v2] wrote {out_dir / 'best.pt'} (epoch {best_epoch}, "
        f"{selection_split} objective {best_score:.6f}) and {out_dir / 'last.pt'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
