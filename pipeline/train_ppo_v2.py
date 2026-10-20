"""Clean-schema PPO canary (schema v2, probability-action contract).

This module is the clean-path counterpart of :mod:`pipeline.train_ppo`.  It
does **not** import or modify the legacy trainer.  It is a *contract canary*,
not a performance claim: the goal is to prove that the sampled action, the
stored log-prob, the executed PSTL action, the reward/cost split, GAE episode
boundaries and the update path are all wired correctly.

Contract references: ``docs/arch_lock.md`` §6.5 (probability action),
§12.2 (reward/cost separation), §13 Stage C (short real-env PPO canary) and
§22 (hard constraints).

Action contract
---------------
``CleanPlanner`` emits ``action_mu`` in ``[0, 1]`` and a diagonal
``TruncatedNormal(mu, exp(logstd), low=0, high=1)`` is used for exploration.
Every sample is therefore executable under :data:`net.action_space.NEW_BOUNDS`
without a post-hoc clamp.  The stored log-prob always belongs to the sampled
normalized action.  If the state-dependent PSTL feasibility check rejects that
action, the same v1-projection / safe-brake fallback as :mod:`pipeline.eval_clean`
is executed, the transition is marked ``masked`` and its policy term is
excluded from the PPO loss.  The projected/fallback action is *never* assigned
the sampled action's log-prob.

Execution / preview
-------------------
Only the first 0.5 s PSTL segment (``a0``, 5 x 0.1 s) is executed and stored.
``a1..a5`` are **not** computed, stored or credited; the 6-step / world-model
preview is a labelled placeholder.  No 6-step WM claim is made.

Reward and cost
---------------
Reward is route progress + success + comfort only.  Collision, off-road,
wrong-route, stuck, comfort violation and overspeed live in a separate
non-negative cost vector and scalar.  Crash / off-road are never folded into
the reward.

GAE
---
GAE is computed per episode.  Episode boundaries never leak: ``done``
(termination) uses zero next value, while truncation / max-step budget uses
``V(s_T)`` as the bootstrap value.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from env.metadrive_env import build_env
from env.obs_v2 import (
    HISTORY_FRAMES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    build_observation_v2,
    stack_history_v2,
)
from env.scenario.spec import load_specs
from env.tracking import ExactTracker
from net.action_space import (
    CLEAN_COMFORT_LIMITS,
    CLEAN_LIMITS,
    NEW_BOUNDS,
    action_to_trajectory,
    feasible_v1_interval,
)
from net.clean_model import SPATIAL_LAYERS, CleanPlanner
from pipeline.distributions import TruncatedNormal
from pipeline.eval_clean import load_clean_model

# ---------------------------------------------------------------------------
# Timing / action contract
# ---------------------------------------------------------------------------
DECISION_DT = 0.5
SIM_DT = 0.1
STEPS_PER_POLICY = 5
HZ = 10
ACTION_DIM = 4
SUPPORT_LOW = 0.0
SUPPORT_HIGH = 1.0
BOUNDARY_EPS = 1e-9

#: Inner PPO passes over one collected batch, and the global grad-norm clip.
PPO_EPOCHS = 4
PPO_MAX_GRAD_NORM = 0.5

# ---------------------------------------------------------------------------
# Reward / cost scales (provisional canary values, documented in contract.json)
# ---------------------------------------------------------------------------
ROUTE_PROGRESS_SCALE = 10.0
SUCCESS_REWARD = 20.0
COMFORT_PENALTY = 0.1
OVERSPEED_COST_SCALE = 0.2

#: Wrong-route event: route completion drops by more than this in one segment.
WRONG_ROUTE_DROP = 0.05
#: Stuck detector: >20 consecutive 0.1 s steps below this speed.
STUCK_SPEED_MPS = 0.1
STUCK_STEPS = 20

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
REWARD_COMPONENTS: Tuple[str, ...] = ("route_progress", "success", "comfort")
COST_NAMES: Tuple[str, ...] = (
    "collision",
    "off_road",
    "wrong_route",
    "stuck",
    "comfort",
    "overspeed",
)

FALLBACK_IDS: Dict[str, int] = {
    "none": 0,
    "v1_projection": 1,
    "safe_brake": 2,
    "nonfinite_safe_brake": 3,
    "safe_brake_infeasible": 4,
}

PREVIEW_LABEL = (
    "PLACEHOLDER_NOT_IMPLEMENTED: only a0 (first 0.5 s segment) is executed and "
    "stored; a1..a5 are not computed, not stored and not credited; no 6-step "
    "world-model claim"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _info_flag(info: Mapping[str, Any], *keys: str) -> bool:
    for key in keys:
        value = info.get(key, False)
        if isinstance(value, (bool, np.bool_)):
            if bool(value):
                return True
        elif isinstance(value, (int, np.integer)):
            if int(value) != 0:
                return True
        elif isinstance(value, (float, np.floating)):
            if math.isfinite(float(value)) and float(value) > 0.5:
                return True
    return False


def _info_cost(info: Mapping[str, Any]) -> float:
    """Optional simulator cost, used only as an extra scalar safety penalty."""
    for key in ("cost", "safety_cost", "env_cost"):
        if key in info:
            value = _safe_float(info.get(key), 0.0)
            if value > 0.0:
                return value
    return 0.0


def _route_completion(ego: Any, info: Mapping[str, Any], fallback: float) -> float:
    for value in (
        info.get("route_completion", None),
        getattr(getattr(ego, "navigation", None), "route_completion", None),
    ):
        if value is None:
            continue
        out = _safe_float(value, float("nan"))
        if math.isfinite(out):
            return out
    return float(fallback)


def _speed_limit(ego: Any) -> Optional[float]:
    lane = getattr(ego, "lane", None)
    if lane is None:
        return None
    value = _safe_float(getattr(lane, "speed_limit", None), float("nan"))
    # MetaDrive uses 1000.0 for "unset".
    if not math.isfinite(value) or value <= 0.0 or value >= 1000.0:
        return None
    return value


def _overspeed_excess(ego: Any, info: Mapping[str, Any]) -> float:
    speed = _safe_float(getattr(ego, "speed", 0.0), 0.0)
    limit = _speed_limit(ego)
    if _info_flag(info, "overspeed", "over_speed"):
        if limit is not None:
            return max(0.0, speed - limit)
        return 1.0
    if limit is None:
        return 0.0
    return max(0.0, speed - limit)


def _advance_speed_run(
    speed: float, low_speed_run: int, stuck_run: bool
) -> Tuple[int, bool]:
    if float(speed) < STUCK_SPEED_MPS:
        low_speed_run += 1
        if low_speed_run > STUCK_STEPS:
            stuck_run = True
    else:
        low_speed_run = 0
    return low_speed_run, stuck_run


def _batch_from_history(hist: Mapping[str, Any], device: str = "cpu") -> Dict[str, Tensor]:
    """Batch-first float32 tensors for one schema-v2 history window."""
    batch: Dict[str, Tensor] = {}
    for key in INPUT_KEYS:
        arr = np.asarray(hist[key], dtype=np.float32)
        batch[key] = torch.from_numpy(arr).unsqueeze(0).to(device)
    return batch


def _trajectory(action: Sequence[float], v0: float):
    return action_to_trajectory(
        action,
        v0,
        dt=DECISION_DT,
        hz=HZ,
        bounds=NEW_BOUNDS,
        limits=CLEAN_LIMITS,
        comfort_limits=CLEAN_COMFORT_LIMITS,
    )


def _trajectory_or_none(action: Sequence[float], v0: float):
    try:
        return _trajectory(action, v0)
    except (TypeError, ValueError):
        return None


def safe_brake_action(v0: float) -> np.ndarray:
    """Explicit safe brake: 6 m/s^2 deceleration, no lateral motion."""
    v0 = max(float(v0), 0.0)
    v1 = max(0.0, v0 - 6.0 * DECISION_DT)
    dx = (v0 + v1) * DECISION_DT / 2.0
    return np.array([dx, 0.0, 0.0, v1], dtype=np.float64)


# ---------------------------------------------------------------------------
# Feasibility fallback (mirrors pipeline.eval_clean.run_episode)
# ---------------------------------------------------------------------------
def resolve_action(sampled_norm: Sequence[float], v0: float) -> Dict[str, Any]:
    """Denormalize then resolve the state-dependent PSTL fallback.

    The sampled normalized action is never mutated.  The returned
    ``requested_action`` is the exact denormalized sample; ``executed_action``
    may differ only through the explicit v1-projection / safe-brake fallback.
    """
    sampled = np.asarray(sampled_norm, dtype=np.float64).reshape(-1)
    if sampled.shape != (ACTION_DIM,):
        raise ValueError(f"sampled_norm must have shape (4,), got {sampled.shape}")
    requested = np.asarray(NEW_BOUNDS.denormalize(sampled, clip=True), dtype=np.float64)

    if not np.all(np.isfinite(requested)):
        brake = safe_brake_action(v0)
        traj = _trajectory_or_none(brake, v0)
        return {
            "requested_action": requested,
            "executed_action": brake,
            "trajectory": traj,
            "fallback": FALLBACK_IDS["nonfinite_safe_brake"],
            "fallback_name": "nonfinite_safe_brake",
            "masked": True,
            "requested_codes": ("NONFINITE_ACTION",),
            "executed_codes": tuple(traj.codes) if traj is not None else (),
            "codes": tuple(traj.codes) if traj is not None else ("NONFINITE_ACTION",),
        }

    requested_traj = _trajectory_or_none(requested, v0)
    requested_codes = (
        tuple(requested_traj.codes) if requested_traj is not None else ("INVALID_ACTION",)
    )
    if requested_traj is not None and requested_traj.feasible:
        return {
            "requested_action": requested,
            "executed_action": requested.copy(),
            "trajectory": requested_traj,
            "fallback": FALLBACK_IDS["none"],
            "fallback_name": "none",
            "masked": False,
            "requested_codes": requested_codes,
            "executed_codes": requested_codes,
            "codes": requested_codes,
        }

    # Least-rewrite fallback: keep geometry, project only v_next into the
    # hard-feasible interval (identical to pipeline.eval_clean).
    if requested_traj is not None:
        try:
            interval = feasible_v1_interval(
                requested,
                v0,
                dt=DECISION_DT,
                bounds=NEW_BOUNDS,
                limits=CLEAN_LIMITS,
            )
        except (TypeError, ValueError):
            interval = None
        if interval is not None and interval.nonempty:
            candidate = requested.copy()
            candidate[3] = float(np.clip(candidate[3], interval.low, interval.high))
            candidate_traj = _trajectory_or_none(candidate, v0)
            if candidate_traj is not None and candidate_traj.feasible:
                executed_codes = tuple(candidate_traj.codes)
                return {
                    "requested_action": requested,
                    "executed_action": candidate,
                    "trajectory": candidate_traj,
                    "fallback": FALLBACK_IDS["v1_projection"],
                    "fallback_name": "v1_projection",
                    "masked": True,
                    "requested_codes": requested_codes,
                    "executed_codes": executed_codes,
                    "codes": tuple(dict.fromkeys(requested_codes + executed_codes)),
                }

    brake = safe_brake_action(v0)
    brake_traj = _trajectory_or_none(brake, v0)
    executed_codes = tuple(brake_traj.codes) if brake_traj is not None else ()
    name = "safe_brake" if brake_traj is not None and brake_traj.feasible else "safe_brake_infeasible"
    return {
        "requested_action": requested,
        "executed_action": brake,
        "trajectory": brake_traj,
        "fallback": FALLBACK_IDS[name],
        "fallback_name": name,
        "masked": True,
        "requested_codes": requested_codes,
        "executed_codes": executed_codes,
        "codes": tuple(dict.fromkeys(requested_codes + executed_codes)),
    }


# ---------------------------------------------------------------------------
# Reward / cost separation
# ---------------------------------------------------------------------------
def compute_reward_cost(
    *,
    route_completion_delta: float,
    success: bool,
    comfortable: bool,
    collision: bool = False,
    off_road: bool = False,
    wrong_route: bool = False,
    stuck: bool = False,
    overspeed_excess: float = 0.0,
    info_cost: float = 0.0,
) -> Dict[str, Any]:
    """Return reward, reward components, cost vector, scalar cost, env cost.

    Reward is intentionally *only* route progress + success + comfort.  All
    safety events are in the cost vector.  ``info_cost`` (if the simulator
    exposes one) is added to the scalar cost only; it is recorded separately.
    """
    route_progress = ROUTE_PROGRESS_SCALE * _safe_float(route_completion_delta, 0.0)
    success_reward = SUCCESS_REWARD if bool(success) else 0.0
    comfort_reward = -COMFORT_PENALTY if not bool(comfortable) else 0.0
    reward_components = {
        "route_progress": float(route_progress),
        "success": float(success_reward),
        "comfort": float(comfort_reward),
    }
    reward = float(sum(reward_components.values()))

    excess = max(_safe_float(overspeed_excess, 0.0), 0.0)
    cost_vector = np.array(
        [
            1.0 if bool(collision) else 0.0,
            1.0 if bool(off_road) else 0.0,
            1.0 if bool(wrong_route) else 0.0,
            1.0 if bool(stuck) else 0.0,
            1.0 if not bool(comfortable) else 0.0,
            OVERSPEED_COST_SCALE * excess,
        ],
        dtype=np.float32,
    )
    env_cost = max(_safe_float(info_cost, 0.0), 0.0)
    cost = float(cost_vector.sum()) + env_cost
    return {
        "reward": reward,
        "reward_components": reward_components,
        "cost_vector": cost_vector,
        "cost": cost,
        "env_cost": float(env_cost),
    }


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------
def _policy_distribution(out: Mapping[str, Tensor]) -> TruncatedNormal:
    return TruncatedNormal(
        out["action_mu"],
        out["action_logstd"].clamp(-5.0, 0.0).exp(),
        low=SUPPORT_LOW,
        high=SUPPORT_HIGH,
    )


def policy_logprob(
    model: CleanPlanner,
    states: Mapping[str, Tensor],
    sampled_norm: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Recompute the stored sampled-action log-prob under current parameters."""
    out = model(states)
    dist = _policy_distribution(out)
    return dist.log_prob(sampled_norm).sum(-1), out


def _policy_logprob_per_transition(
    model: CleanPlanner,
    states: Mapping[str, Tensor],
    sampled_norm: Tensor,
) -> Tensor:
    """Audit log-probs one transition at a time (same batch size as rollout)."""
    logps = []
    for i in range(int(sampled_norm.shape[0])):
        sub = {key: value[i : i + 1] for key, value in states.items()}
        logp, _out = policy_logprob(model, sub, sampled_norm[i : i + 1])
        logps.append(logp.reshape(()))
    return torch.stack(logps) if logps else torch.empty(0, device=sampled_norm.device)


def boundary_rate(sampled_norm: Tensor) -> float:
    """Fraction of transitions with any component at the exact support edge."""
    x = sampled_norm.detach()
    if x.numel() == 0:
        return 0.0
    if x.ndim == 1:
        x = x.reshape(-1, ACTION_DIM)
    at_edge = (x <= SUPPORT_LOW + BOUNDARY_EPS) | (x >= SUPPORT_HIGH - BOUNDARY_EPS)
    return float(at_edge.any(dim=-1).float().mean().cpu())


# ---------------------------------------------------------------------------
# Rollout storage / GAE
# ---------------------------------------------------------------------------
REQUIRED_TRANSITION_KEYS: Tuple[str, ...] = (
    "state",
    "sampled_norm",
    "executed_action",
    "old_logp",
    "old_value",
    "old_cost_value",
    "reward",
    "reward_components",
    "cost",
    "cost_vector",
    "env_cost",
    "terminated",
    "truncated",
    "episode_id",
    "t",
    "fallback",
    "fallback_name",
    "masked",
    "codes",
    "requested_codes",
    "executed_codes",
    "bootstrap_value",
    "bootstrap_cost_value",
)


def compute_gae(
    values: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    bootstrap_value: Any,
    gamma: float,
    lam: float,
) -> Tuple[Tensor, Tensor]:
    """Per-episode GAE.

    ``terminated`` zeroes the next-value term; a truncated / max-step last
    transition passes its ``V(s_T)`` through ``bootstrap_value``.  The caller
    must never mix transitions from different episodes in one call.
    """
    if values.ndim != 1 or rewards.shape != values.shape or terminated.shape != values.shape:
        raise ValueError("GAE inputs must be 1-D tensors of equal length")
    n = int(values.shape[0])
    adv = torch.zeros_like(values)
    boot = torch.as_tensor(bootstrap_value, dtype=values.dtype, device=values.device)
    last = torch.zeros((), dtype=values.dtype, device=values.device)
    for i in range(n - 1, -1, -1):
        nonterminal = 1.0 - terminated[i].to(values.dtype)
        next_value = boot if i == n - 1 else values[i + 1]
        delta = rewards[i] + float(gamma) * next_value * nonterminal - values[i]
        last = delta + float(gamma) * float(lam) * nonterminal * last
        adv[i] = last
    return adv, adv + values


class RolloutBuffer:
    """Per-transition storage with per-episode GAE and tensor collation."""

    def __init__(self, *, gamma: float = 0.95, lam: float = 0.95) -> None:
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.transitions: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.transitions)

    def add_transition(self, transition: Mapping[str, Any]) -> None:
        missing = [key for key in REQUIRED_TRANSITION_KEYS if key not in transition]
        if missing:
            raise KeyError(f"transition missing fields: {missing}")
        state = transition["state"]
        missing_state = [key for key in INPUT_KEYS if key not in state]
        if missing_state:
            raise KeyError(f"transition state missing fields: {missing_state}")
        sampled = torch.as_tensor(transition["sampled_norm"])
        executed = torch.as_tensor(transition["executed_action"])
        cost_vector = torch.as_tensor(transition["cost_vector"])
        if tuple(sampled.shape) != (ACTION_DIM,):
            raise ValueError(f"sampled_norm must be (4,), got {tuple(sampled.shape)}")
        if tuple(executed.shape) != (ACTION_DIM,):
            raise ValueError(f"executed_action must be (4,), got {tuple(executed.shape)}")
        if tuple(cost_vector.shape) != (len(COST_NAMES),):
            raise ValueError(
                f"cost_vector must be ({len(COST_NAMES)},), got {tuple(cost_vector.shape)}"
            )
        fallback = int(transition["fallback"])
        if fallback not in FALLBACK_IDS.values():
            raise ValueError(f"unknown fallback id {fallback}")
        self.transitions.append(dict(transition))

    def add_episode(self, transitions: Sequence[Mapping[str, Any]]) -> None:
        for transition in transitions:
            self.add_transition(transition)

    def _stack_scalar(self, key: str, device: str) -> Tensor:
        return torch.stack(
            [torch.as_tensor(tr[key], dtype=torch.float32).reshape(()) for tr in self.transitions]
        ).to(device)

    def to_training_batch(
        self,
        device: str = "cpu",
        *,
        gamma: Optional[float] = None,
        lam: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self.transitions:
            raise ValueError("cannot collate an empty RolloutBuffer")
        gamma = self.gamma if gamma is None else float(gamma)
        lam = self.lam if lam is None else float(lam)

        states: Dict[str, Tensor] = {}
        for key in INPUT_KEYS:
            states[key] = torch.cat(
                [
                    torch.as_tensor(tr["state"][key], dtype=torch.float32)
                    for tr in self.transitions
                ],
                dim=0,
            ).to(device)

        sampled_norm = torch.stack(
            [torch.as_tensor(tr["sampled_norm"], dtype=torch.float32) for tr in self.transitions]
        ).to(device)
        executed_action = torch.stack(
            [torch.as_tensor(tr["executed_action"], dtype=torch.float32) for tr in self.transitions]
        ).to(device)
        cost_vector = torch.stack(
            [torch.as_tensor(tr["cost_vector"], dtype=torch.float32) for tr in self.transitions]
        ).to(device)

        old_logp = self._stack_scalar("old_logp", device)
        old_value = self._stack_scalar("old_value", device)
        old_cost_value = self._stack_scalar("old_cost_value", device)
        reward = self._stack_scalar("reward", device)
        cost = self._stack_scalar("cost", device)
        env_cost = self._stack_scalar("env_cost", device)
        bootstrap_value = self._stack_scalar("bootstrap_value", device)
        bootstrap_cost_value = self._stack_scalar("bootstrap_cost_value", device)

        terminated = torch.tensor(
            [bool(tr["terminated"]) for tr in self.transitions],
            dtype=torch.bool,
            device=device,
        )
        truncated = torch.tensor(
            [bool(tr["truncated"]) for tr in self.transitions],
            dtype=torch.bool,
            device=device,
        )
        episode_id = torch.tensor(
            [int(tr["episode_id"]) for tr in self.transitions],
            dtype=torch.long,
            device=device,
        )
        t = torch.tensor(
            [int(tr["t"]) for tr in self.transitions], dtype=torch.long, device=device
        )
        fallback = torch.tensor(
            [int(tr["fallback"]) for tr in self.transitions],
            dtype=torch.long,
            device=device,
        )
        masked = torch.tensor(
            [bool(tr["masked"]) for tr in self.transitions],
            dtype=torch.bool,
            device=device,
        )
        fallback_name = [str(tr["fallback_name"]) for tr in self.transitions]
        codes = [tuple(tr["codes"]) for tr in self.transitions]
        requested_codes = [tuple(tr["requested_codes"]) for tr in self.transitions]
        executed_codes = [tuple(tr["executed_codes"]) for tr in self.transitions]

        reward_components = {
            name: torch.tensor(
                [
                    float(tr["reward_components"].get(name, 0.0))
                    for tr in self.transitions
                ],
                dtype=torch.float32,
                device=device,
            )
            for name in REWARD_COMPONENTS
        }

        valid = (
            torch.isfinite(sampled_norm).all(dim=-1)
            & torch.isfinite(executed_action).all(dim=-1)
            & torch.isfinite(old_logp)
            & torch.isfinite(old_value)
            & torch.isfinite(old_cost_value)
            & torch.isfinite(reward)
            & torch.isfinite(cost)
            & torch.isfinite(cost_vector).all(dim=-1)
        )
        policy_valid = valid & (~masked)

        adv, ret = self._gae_grouped(
            old_value, reward, terminated, episode_id, bootstrap_value, gamma, lam
        )
        cost_adv, cost_ret = self._gae_grouped(
            old_cost_value, cost, terminated, episode_id, bootstrap_cost_value, gamma, lam
        )
        if not torch.isfinite(cost_adv).all():
            raise RuntimeError("non-finite cost advantage")

        return {
            "state": states,
            "sampled_norm": sampled_norm,
            "executed_action": executed_action,
            "old_logp": old_logp,
            "old_value": old_value,
            "old_cost_value": old_cost_value,
            "reward": reward,
            "reward_components": reward_components,
            "cost": cost,
            "cost_vector": cost_vector,
            "env_cost": env_cost,
            "terminated": terminated,
            "truncated": truncated,
            "episode_id": episode_id,
            "t": t,
            "fallback": fallback,
            "fallback_name": fallback_name,
            "masked": masked,
            "codes": codes,
            "requested_codes": requested_codes,
            "executed_codes": executed_codes,
            "bootstrap_value": bootstrap_value,
            "bootstrap_cost_value": bootstrap_cost_value,
            "valid": valid,
            "policy_valid": policy_valid,
            "adv": adv,
            "ret": ret,
            "cost_adv": cost_adv,
            "cost_ret": cost_ret,
            "truncation_bootstrap": truncated & (~terminated),
        }

    def _gae_grouped(
        self,
        values: Tensor,
        rewards: Tensor,
        terminated: Tensor,
        episode_id: Tensor,
        bootstrap_value: Tensor,
        gamma: float,
        lam: float,
    ) -> Tuple[Tensor, Tensor]:
        adv = torch.zeros_like(values)
        ret = torch.zeros_like(values)
        for ep in torch.unique(episode_id, sorted=True):
            idx = torch.nonzero(episode_id == ep, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            order = torch.argsort(self._tensor_t(idx))
            idx = idx[order]
            ep_adv, ep_ret = compute_gae(
                values[idx],
                rewards[idx],
                terminated[idx],
                bootstrap_value[idx[-1]],
                gamma,
                lam,
            )
            adv[idx] = ep_adv
            ret[idx] = ep_ret
        return adv, ret

    def _tensor_t(self, idx: Tensor) -> Tensor:
        return torch.tensor(
            [int(self.transitions[int(i)]["t"]) for i in idx],
            dtype=torch.long,
            device=idx.device,
        )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _termination_name(
    *,
    success: bool,
    collision: bool,
    off_road: bool,
    terminated: bool,
    truncated: bool,
) -> str:
    if success:
        return "arrive"
    if collision:
        return "collision"
    if off_road:
        return "out_of_road"
    if truncated and not terminated:
        return "timeout"
    return "terminal" if terminated else "max_step"


def collect_episode(
    env: Any,
    model: CleanPlanner,
    spec: Any = None,
    *,
    max_steps: int = 200,
    device: str = "cpu",
    episode_id: int = 0,
    sampling_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one real-env on-policy episode and return transitions + summary.

    ``max_steps`` counts 0.1 s ``env.step`` calls.  Only the first 0.5 s PSTL
    segment of each decision is executed.  A max-step cutoff marks the last
    transition ``truncated`` so GAE bootstraps ``V(s_T)``.
    """
    max_steps = int(max_steps)
    if max_steps < 0:
        raise ValueError("max_steps must be >= 0")
    model.to(device).eval()
    if sampling_seed is not None:
        torch.manual_seed(int(sampling_seed))

    env.reset()
    ego = env.agent
    tracker = ExactTracker(dt=DECISION_DT, hz=HZ)

    frames: List[Dict[str, Any]] = []
    poses: List[np.ndarray] = []
    transitions: List[Dict[str, Any]] = []
    prev_action = np.zeros(ACTION_DIM, dtype=np.float64)
    prev_speed: Optional[float] = None
    prev_heading: Optional[float] = None

    n = 0
    policy_steps = 0
    termination = "max_step"
    success = collision = off_road = wrong_route = False
    low_speed_run = 0
    stuck_run = False
    rc = _route_completion(ego, {}, 0.0)
    truncation_bootstrap_count = 0

    while n < max_steps:
        obs = build_observation_v2(
            env,
            prev_action=prev_action,
            prev_speed=prev_speed,
            prev_heading=prev_heading,
            dt=DECISION_DT,
        )
        frames.append(obs)
        poses.append(obs["pose"].copy())
        if len(frames) > HISTORY_FRAMES:
            frames = frames[-HISTORY_FRAMES:]
            poses = poses[-HISTORY_FRAMES:]
        hist = stack_history_v2(frames, poses)
        batch = _batch_from_history(hist, device=device)

        with torch.no_grad():
            out = model(batch)
            dist = _policy_distribution(out)
            sampled = dist.sample()
            old_logp = dist.log_prob(sampled).sum(-1)[0]
            old_value = out["value"][0, 0]
            old_cost_value = out["cost_value"][0, 0]

        sampled_np = sampled[0].detach().cpu().numpy().astype(np.float64)
        v0 = float(ego.speed)
        resolved = resolve_action(sampled_np, v0)
        executed = np.asarray(resolved["executed_action"], dtype=np.float64)
        tr_exec = resolved["trajectory"]
        fallback_name = str(resolved["fallback_name"])
        masked = bool(resolved["masked"])

        rc0 = _route_completion(ego, {}, rc)
        seg_success = False
        seg_collision = False
        seg_off_road = False
        seg_wrong_route = False
        seg_stuck = False
        seg_comfort_violation = False
        seg_overspeed = 0.0
        seg_info_cost = 0.0
        terminated = False
        truncated = False

        if tr_exec is None:
            # Defensive: no executable geometry (should not happen for a finite
            # sample under NEW_BOUNDS).  End the episode without inventing steps.
            termination = "invalid_action"
            terminated = True
            resolved["codes"] = tuple(dict.fromkeys(tuple(resolved["codes"]) + ("EXECUTION_UNAVAILABLE",)))
        else:
            tracker.arm(env, trajectory=tr_exec)
            for _ in range(STEPS_PER_POLICY):
                if n >= max_steps:
                    break
                _, _, term, trunc, info = env.step([0.0, 0.0])
                n += 1
                tracker.apply(env)
                info = info or {}

                rc = _route_completion(ego, info, rc)
                seg_success = seg_success or _info_flag(info, "arrive_dest")
                seg_collision = seg_collision or _info_flag(
                    info,
                    "crash",
                    "crash_vehicle",
                    "crash_object",
                    "crash_building",
                    "crash_human",
                    "crash_sidewalk",
                )
                seg_off_road = seg_off_road or _info_flag(info, "out_of_road", "off_road")
                seg_wrong_route = seg_wrong_route or _info_flag(
                    info, "wrong_route", "wrong_way"
                )
                seg_stuck = seg_stuck or _info_flag(info, "stuck")
                seg_comfort_violation = seg_comfort_violation or _info_flag(
                    info, "comfort_violation", "uncomfortable"
                )
                seg_overspeed = max(seg_overspeed, _overspeed_excess(ego, info))
                seg_info_cost = max(seg_info_cost, _info_cost(info))
                low_speed_run, stuck_run = _advance_speed_run(
                    float(ego.speed), low_speed_run, stuck_run
                )
                if term or trunc:
                    terminated = bool(term)
                    truncated = bool(trunc and not term)
                    break

            if not terminated and not truncated and n >= max_steps:
                truncated = True

        if rc < rc0 - WRONG_ROUTE_DROP:
            seg_wrong_route = True
        seg_stuck = bool(seg_stuck or stuck_run)
        comfortable = bool(getattr(tr_exec, "comfortable", False)) if tr_exec is not None else False
        seg_comfort_violation = bool(seg_comfort_violation or (not comfortable))

        success = success or seg_success
        collision = collision or seg_collision
        off_road = off_road or seg_off_road
        wrong_route = wrong_route or seg_wrong_route

        rc_info = compute_reward_cost(
            route_completion_delta=rc - rc0,
            success=seg_success,
            comfortable=not seg_comfort_violation,
            collision=seg_collision,
            off_road=seg_off_road,
            wrong_route=seg_wrong_route,
            stuck=seg_stuck,
            overspeed_excess=seg_overspeed,
            info_cost=seg_info_cost,
        )

        transitions.append(
            {
                "state": {key: batch[key].detach().cpu() for key in INPUT_KEYS},
                "sampled_norm": sampled[0].detach().cpu(),
                "executed_action": torch.as_tensor(executed, dtype=torch.float32),
                "old_logp": old_logp.detach().cpu(),
                "old_value": old_value.detach().cpu(),
                "old_cost_value": old_cost_value.detach().cpu(),
                "reward": torch.tensor(rc_info["reward"], dtype=torch.float32),
                "reward_components": dict(rc_info["reward_components"]),
                "cost": torch.tensor(rc_info["cost"], dtype=torch.float32),
                "cost_vector": torch.as_tensor(rc_info["cost_vector"], dtype=torch.float32),
                "env_cost": torch.tensor(rc_info["env_cost"], dtype=torch.float32),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "episode_id": int(episode_id),
                "t": int(policy_steps),
                "fallback": int(resolved["fallback"]),
                "fallback_name": fallback_name,
                "masked": masked,
                "codes": tuple(resolved["codes"]),
                "requested_codes": tuple(resolved["requested_codes"]),
                "executed_codes": tuple(resolved["executed_codes"]),
                "bootstrap_value": torch.tensor(0.0, dtype=torch.float32),
                "bootstrap_cost_value": torch.tensor(0.0, dtype=torch.float32),
            }
        )
        policy_steps += 1

        prev_action = executed.astype(np.float64)
        prev_speed = float(ego.speed)
        prev_heading = float(ego.heading_theta)

        if terminated or truncated:
            termination = _termination_name(
                success=seg_success,
                collision=seg_collision,
                off_road=seg_off_road,
                terminated=terminated,
                truncated=truncated,
            )
            break

    if transitions:
        last = transitions[-1]
        if bool(last["truncated"]) and not bool(last["terminated"]):
            final_obs = build_observation_v2(
                env,
                prev_action=prev_action,
                prev_speed=prev_speed,
                prev_heading=prev_heading,
                dt=DECISION_DT,
            )
            final_frames = frames + [final_obs]
            final_poses = poses + [final_obs["pose"].copy()]
            if len(final_frames) > HISTORY_FRAMES:
                final_frames = final_frames[-HISTORY_FRAMES:]
                final_poses = final_poses[-HISTORY_FRAMES:]
            final_hist = stack_history_v2(final_frames, final_poses)
            final_batch = _batch_from_history(final_hist, device=device)
            with torch.no_grad():
                final_out = model(final_batch)
                last["bootstrap_value"] = final_out["value"][0, 0].detach().cpu()
                last["bootstrap_cost_value"] = (
                    final_out["cost_value"][0, 0].detach().cpu()
                )
            truncation_bootstrap_count = 1

    fallback_counts = Counter(str(tr["fallback_name"]) for tr in transitions)
    v1_projection_count = int(fallback_counts.get("v1_projection", 0))
    safe_brake_count = int(
        sum(
            count
            for name, count in fallback_counts.items()
            if name in ("safe_brake", "nonfinite_safe_brake", "safe_brake_infeasible")
        )
    )
    masked_count = int(sum(1 for tr in transitions if tr["masked"]))
    summary = {
        "episode_id": int(episode_id),
        "spec_id": getattr(spec, "id", None),
        "scene_seed": int(getattr(spec, "seed", 0)) if spec is not None else None,
        "steps": int(n),
        "policy_steps": int(policy_steps),
        "transitions": int(len(transitions)),
        "success": bool(success),
        "collision": bool(collision),
        "off_road": bool(off_road),
        "wrong_route": bool(wrong_route),
        "stuck": bool(stuck_run and not success),
        "timeout": bool(termination in ("max_step", "timeout")),
        "termination": str(termination),
        "route_completion": float(rc),
        "reward_sum": float(sum(float(tr["reward"]) for tr in transitions)),
        "cost_sum": float(sum(float(tr["cost"]) for tr in transitions)),
        "infeasible_count": int(masked_count),
        "fallback_count": int(safe_brake_count),
        "postprocess_count": int(v1_projection_count),
        "v1_projection_count": int(v1_projection_count),
        "masked_count": int(masked_count),
        "fallback_counts": dict(fallback_counts),
        "truncation_bootstrap_count": int(truncation_bootstrap_count),
        "preview": PREVIEW_LABEL,
    }
    return {"transitions": transitions, "summary": summary}


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def ppo_update(
    model: CleanPlanner,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, Any],
    *,
    clip: float = 0.2,
    value_coef: float = 0.5,
    cost_coef: float = 0.1,
    epochs: int = PPO_EPOCHS,
    max_grad_norm: float = PPO_MAX_GRAD_NORM,
    device: str = "cpu",
    update_index: int = 0,
) -> Dict[str, Any]:
    """One clipped-PPO update on a collated on-policy batch."""
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1")
    model.to(device).train()
    states = batch["state"]
    sampled = batch["sampled_norm"]
    old_logp = batch["old_logp"]
    old_value = batch["old_value"]
    adv = batch["adv"]
    ret = batch["ret"]
    cost_ret = batch["cost_ret"]
    policy_mask = batch["policy_valid"]
    value_mask = batch["valid"]

    if not bool(value_mask.any()):
        raise ValueError("ppo_update: no finite transitions")

    # Audit the stored log-prob at the old parameters.  The rollout sampled one
    # state at a time (batch size 1), so the contract audit mirrors that shape;
    # the batched recompute is kept as a separate diagnostic.
    model.eval()
    with torch.no_grad():
        newlogp0 = _policy_logprob_per_transition(model, states, sampled)
        newlogp0_batch, _out0 = policy_logprob(model, states, sampled)
        log_ratio0 = newlogp0 - old_logp
        log_ratio0_batch = newlogp0_batch - old_logp
        ratio0 = log_ratio0.exp()
        ratio0_batch = log_ratio0_batch.exp()
        diff0 = log_ratio0.abs()
        max_diff = float(diff0.max().item()) if diff0.numel() else 0.0
        batch_max_diff = (
            float(log_ratio0_batch.abs().max().item())
            if log_ratio0_batch.numel()
            else 0.0
        )
        ratio_at_old_mean = (
            float(ratio0[value_mask].mean().item()) if bool(value_mask.any()) else math.nan
        )
        ratio_at_old_max_abs_dev = (
            float((ratio0[value_mask] - 1.0).abs().max().item())
            if bool(value_mask.any())
            else math.nan
        )
        ratio_at_old_batch_mean = (
            float(ratio0_batch[value_mask].mean().item())
            if bool(value_mask.any())
            else math.nan
        )
        ratio_at_old_batch_max_abs_dev = (
            float((ratio0_batch[value_mask] - 1.0).abs().max().item())
            if bool(value_mask.any())
            else math.nan
        )
    model.train()

    if max_diff > 1e-4:
        raise RuntimeError(
            "stored log-prob does not match the policy at old parameters: "
            f"max |new-old|={max_diff:.3e}"
        )

    epoch_history: List[Dict[str, float]] = []
    for _epoch in range(int(epochs)):
        out = model(states)
        dist = _policy_distribution(out)
        newlogp = dist.log_prob(sampled).sum(-1)
        log_ratio = newlogp - old_logp
        ratio = log_ratio.clamp(-20.0, 20.0).exp()

        if bool(policy_mask.any()):
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - float(clip), 1.0 + float(clip)) * adv
            policy_loss = -torch.min(surr1, surr2)[policy_mask].mean()
            approx_kl = ((ratio[policy_mask] - 1.0) - log_ratio[policy_mask]).mean()
            clip_frac = (
                (ratio[policy_mask] - 1.0).abs() > float(clip)
            ).float().mean()
        else:
            # Keep the graph connected but contribute exactly zero.
            policy_loss = out["action_mu"].sum() * 0.0
            approx_kl = torch.zeros((), device=out["action_mu"].device)
            clip_frac = torch.zeros((), device=out["action_mu"].device)

        value_pred = out["value"].squeeze(-1)
        cost_pred = out["cost_value"].squeeze(-1)
        value_loss = F.mse_loss(value_pred[value_mask], ret[value_mask])
        cost_loss = F.mse_loss(cost_pred[value_mask], cost_ret[value_mask])
        total_loss = policy_loss + float(value_coef) * value_loss + float(cost_coef) * cost_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        optimizer.step()

        epoch_history.append(
            {
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "cost_loss": float(cost_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "approx_kl": float(approx_kl.detach()),
                "clip_frac": float(clip_frac.detach()),
                "grad_norm": float(grad_norm),
                "ratio_mean": float(ratio[value_mask].mean().detach()),
            }
        )

    first_epoch = epoch_history[0]
    last = epoch_history[-1]
    boundary = boundary_rate(sampled)
    fallback = batch["fallback"]
    fallback_names = batch["fallback_name"]
    transitions = int(sampled.shape[0])
    postprocess_count = sum(1 for name in fallback_names if name == "v1_projection")
    record = {
        "update": int(update_index) + 1,
        "transitions": transitions,
        "policy_valid_transitions": int(policy_mask.sum()),
        "value_valid_transitions": int(value_mask.sum()),
        "masked_transitions": int(batch["masked"].sum()),
        "fallback_rate": float((fallback != 0).float().mean()),
        "mask_rate": float(batch["masked"].float().mean()),
        "postprocess_rate": float(postprocess_count / max(transitions, 1)),
        "boundary_rate": float(boundary),
        "boundary_free_rate": float(1.0 - boundary),
        "logprob_recompute_max_abs_diff": max_diff,
        "logprob_recompute_batch_max_abs_diff": batch_max_diff,
        "ratio_at_old_mean": ratio_at_old_mean,
        "ratio_at_old_max_abs_dev": ratio_at_old_max_abs_dev,
        "ratio_at_old_batch_mean": ratio_at_old_batch_mean,
        "ratio_at_old_batch_max_abs_dev": ratio_at_old_batch_max_abs_dev,
        "truncation_bootstrap_count": int(batch["truncation_bootstrap"].sum()),
        "policy_loss": last["policy_loss"],
        "value_loss": last["value_loss"],
        "cost_loss": last["cost_loss"],
        "total_loss": last["total_loss"],
        # Primary KL/clip metrics are the first inner epoch, before the first
        # optimizer step; the last-epoch values are kept separately.
        "approx_kl": first_epoch["approx_kl"],
        "clip_frac": first_epoch["clip_frac"],
        "approx_kl_last": last["approx_kl"],
        "clip_frac_last": last["clip_frac"],
        "grad_norm": last["grad_norm"],
        "ratio_mean": last["ratio_mean"],
        "epochs": epoch_history,
        "old_value_mean": float(old_value[value_mask].mean()),
        "reward_mean": float(batch["reward"][value_mask].mean()),
        "cost_mean": float(batch["cost"][value_mask].mean()),
        "return_mean": float(ret[value_mask].mean()),
        "cost_return_mean": float(cost_ret[value_mask].mean()),
    }
    return record


# ---------------------------------------------------------------------------
# Contract summary / CLI
# ---------------------------------------------------------------------------
def summarize_rollout(
    episodes: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate reward/cost, fallback/mask, boundary and episode summaries."""
    n_episodes = int(len(episodes))
    n_transitions = int(len(transitions))
    fallback_counts = Counter(str(tr["fallback_name"]) for tr in transitions)
    masked_count = int(sum(1 for tr in transitions if bool(tr["masked"])))
    postprocess_count = int(fallback_counts.get("v1_projection", 0))
    truncation_bootstrap_count = int(
        sum(
            1
            for tr in transitions
            if bool(tr["truncated"]) and not bool(tr["terminated"])
        )
    )
    if transitions:
        sampled = torch.stack(
            [torch.as_tensor(tr["sampled_norm"], dtype=torch.float32) for tr in transitions]
        )
        boundary = boundary_rate(sampled)
    else:
        boundary = 0.0

    reward_sum = float(sum(float(tr["reward"]) for tr in transitions))
    reward_components = {
        name: float(sum(float(tr["reward_components"].get(name, 0.0)) for tr in transitions))
        for name in REWARD_COMPONENTS
    }
    cost_sum = float(sum(float(tr["cost"]) for tr in transitions))
    env_cost_sum = float(sum(float(tr.get("env_cost", 0.0)) for tr in transitions))
    cost_components: Dict[str, float] = {}
    for i, name in enumerate(COST_NAMES):
        cost_components[name] = float(
            sum(float(torch.as_tensor(tr["cost_vector"])[i]) for tr in transitions)
        )
    denom = max(n_transitions, 1)

    def _rate(key: str) -> float:
        if not episodes:
            return 0.0
        return float(np.mean([bool(ep.get(key, False)) for ep in episodes]))

    return {
        "n_episodes": n_episodes,
        "transitions": n_transitions,
        "boundary_rate": float(boundary),
        "boundary_free_rate": float(1.0 - boundary),
        "postprocess_rate": float(postprocess_count / denom),
        "postprocess_count": int(postprocess_count),
        "v1_projection_count": int(postprocess_count),
        "safe_brake_count": int(
            sum(
                count
                for name, count in fallback_counts.items()
                if name in ("safe_brake", "nonfinite_safe_brake", "safe_brake_infeasible")
            )
        ),
        "mask_rate": float(masked_count / denom),
        "masked_count": int(masked_count),
        "fallback_rate": float(masked_count / denom),
        "fallback_counts": dict(fallback_counts),
        "truncation_bootstrap_count": int(truncation_bootstrap_count),
        "reward": {
            "sum": reward_sum,
            "mean": float(reward_sum / denom),
            "components_sum": reward_components,
            "components_mean": {
                name: float(value / denom) for name, value in reward_components.items()
            },
        },
        "cost": {
            "sum": cost_sum,
            "mean": float(cost_sum / denom),
            "env_cost_sum": env_cost_sum,
            "components_sum": cost_components,
            "components_mean": {
                name: float(value / denom) for name, value in cost_components.items()
            },
            "names": list(COST_NAMES),
        },
        "outcomes": {
            "success_rate": _rate("success"),
            "collision_rate": _rate("collision"),
            "off_road_rate": _rate("off_road"),
            "wrong_route_rate": _rate("wrong_route"),
            "stuck_rate": _rate("stuck"),
            "timeout_rate": _rate("timeout"),
            "route_completion_mean": (
                float(np.mean([float(ep.get("route_completion", 0.0)) for ep in episodes]))
                if episodes
                else 0.0
            ),
        },
        "episodes": list(episodes),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean-schema PPO contract canary (schema v2, no WM)"
    )
    parser.add_argument("--specs", required=True)
    parser.add_argument(
        "--ckpt",
        default=None,
        help="clean BC/DAgger checkpoint; random init only for a smoke run",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--cost-coef", type=float, default=0.1)
    return parser


def _limits_dict() -> Dict[str, float]:
    return {
        "a_long_min": float(CLEAN_LIMITS.a_long_min),
        "a_long_max": float(CLEAN_LIMITS.a_long_max),
        "a_lat_max": float(CLEAN_LIMITS.a_lat_max),
        "kappa_max": float(CLEAN_LIMITS.kappa_max),
        "kappa_rate_max": float(CLEAN_LIMITS.kappa_rate_max),
        "jerk_long_max": float(CLEAN_LIMITS.jerk_long_max),
        "jerk_lat_max": float(CLEAN_LIMITS.jerk_lat_max),
    }


def _checkpoint_config(
    model: CleanPlanner, args: argparse.Namespace
) -> Dict[str, Any]:
    eval_config = dict(getattr(model, "eval_config", {}) or {})
    return {
        "model_type": "clean",
        "architecture": "CleanPlanner",
        "hidden": int(model.hidden),
        "heads": int(model.heads),
        "spatial_layers": int(eval_config.get("spatial_layers", SPATIAL_LAYERS)),
        "obs_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "action_bounds_name": NEW_BOUNDS.name,
        "action_low": NEW_BOUNDS.low.tolist(),
        "action_high": NEW_BOUNDS.high.tolist(),
        "action_support": [SUPPORT_LOW, SUPPORT_HIGH],
        "gamma": float(args.gamma),
        "lam": float(args.lam),
        "clip": float(args.clip),
        "value_coef": float(args.value_coef),
        "cost_coef": float(args.cost_coef),
        "decision": {
            "dt": DECISION_DT,
            "sim_dt": SIM_DT,
            "hz": HZ,
            "steps_per_policy": STEPS_PER_POLICY,
            "history_frames": HISTORY_FRAMES,
            "execution": "first 0.5 s segment only (a0); a1..a5 preview not stored",
        },
        "limits": {"clean_hard": _limits_dict()},
        "reward": {
            "route_progress_scale": ROUTE_PROGRESS_SCALE,
            "success": SUCCESS_REWARD,
            "comfort_penalty": COMFORT_PENALTY,
            "components": list(REWARD_COMPONENTS),
        },
        "cost": {
            "names": list(COST_NAMES),
            "overspeed_scale": OVERSPEED_COST_SCALE,
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.updates < 1:
        raise SystemExit("--updates must be >= 1")
    if args.episodes < 1:
        raise SystemExit("--episodes must be >= 1")
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be >= 0")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    model = load_clean_model(args.ckpt, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    specs = load_specs(args.specs)
    if not specs:
        raise SystemExit(f"no specs in {args.specs}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_episodes: List[Dict[str, Any]] = []
    all_transitions: List[Dict[str, Any]] = []
    update_records: List[Dict[str, Any]] = []

    for update in range(int(args.updates)):
        buffer = RolloutBuffer(gamma=float(args.gamma), lam=float(args.lam))
        for episode in range(int(args.episodes)):
            index = update * int(args.episodes) + episode
            spec = specs[index % len(specs)]
            env = build_env(spec, use_render=False, lru_size=0)
            try:
                result = collect_episode(
                    env,
                    model,
                    spec,
                    max_steps=int(args.max_steps),
                    device=str(args.device),
                    episode_id=index,
                    sampling_seed=int(args.seed) * 1_000_003 + index,
                )
            finally:
                env.close()
            buffer.add_episode(result["transitions"])
            all_episodes.append(result["summary"])
            all_transitions.extend(result["transitions"])
            print(
                f"[ppo_v2/update {update + 1}] episode {episode + 1}/"
                f"{args.episodes} {result['summary']}",
                flush=True,
            )

        if len(buffer) == 0:
            print("[ppo_v2] WARNING: no transitions; skipping update", flush=True)
            continue

        batch = buffer.to_training_batch(
            device=str(args.device), gamma=float(args.gamma), lam=float(args.lam)
        )
        record = ppo_update(
            model,
            optimizer,
            batch,
            clip=float(args.clip),
            value_coef=float(args.value_coef),
            cost_coef=float(args.cost_coef),
            device=str(args.device),
            update_index=update,
        )
        update_records.append(record)
        print(
            "[ppo_v2] "
            + json.dumps(
                {
                    key: record[key]
                    for key in (
                        "update",
                        "transitions",
                        "policy_valid_transitions",
                        "approx_kl",
                        "clip_frac",
                        "value_loss",
                        "cost_loss",
                        "boundary_rate",
                        "fallback_rate",
                        "logprob_recompute_max_abs_diff",
                        "ratio_at_old_mean",
                    )
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = summarize_rollout(all_episodes, all_transitions)
    first_update = update_records[0] if update_records else {}
    checkpoint_path = out_dir / "ppo_canary.pt"
    config = _checkpoint_config(model, args)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config,
            "action_bounds": NEW_BOUNDS.as_dict(),
            "gamma": float(args.gamma),
            "lam": float(args.lam),
            "obs_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
            "training": {
                "updates": int(args.updates),
                "episodes_per_update": int(args.episodes),
                "max_steps": int(args.max_steps),
                "lr": float(args.lr),
                "seed": int(args.seed),
                "device": str(args.device),
            },
        },
        checkpoint_path,
    )

    contract = {
        **summary,
        "schema": "drl_pathplan_ppo_contract_v2",
        "schema_version": 1,
        "obs_schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "action_bounds": NEW_BOUNDS.as_dict(),
        "support": [SUPPORT_LOW, SUPPORT_HIGH],
        "decision": {
            "dt": DECISION_DT,
            "sim_dt": SIM_DT,
            "hz": HZ,
            "steps_per_policy": STEPS_PER_POLICY,
            "history_frames": HISTORY_FRAMES,
        },
        "preview": PREVIEW_LABEL,
        "logprob_recompute_max_abs_diff": first_update.get(
            "logprob_recompute_max_abs_diff"
        ),
        "logprob_recompute_scope": "first update, before any optimizer step",
        "logprob_recompute_tolerance": 1e-5,
        "ratio_at_old": first_update.get("ratio_at_old_mean"),
        "ratio_at_old_max_abs_dev": first_update.get("ratio_at_old_max_abs_dev"),
        "ratio_at_old_tolerance": 1e-5,
        "ratio_at_old_scope": "first update, before any optimizer step",
        "updates": update_records,
        "config": {
            "specs": str(args.specs),
            "ckpt": None if args.ckpt is None else str(args.ckpt),
            "out": str(out_dir),
            "updates": int(args.updates),
            "episodes": int(args.episodes),
            "episodes_per_update": int(args.episodes),
            "max_steps": int(args.max_steps),
            "lr": float(args.lr),
            "device": str(args.device),
            "seed": int(args.seed),
            "gamma": float(args.gamma),
            "lam": float(args.lam),
            "clip": float(args.clip),
            "value_coef": float(args.value_coef),
            "cost_coef": float(args.cost_coef),
            "episode_spec_order": "cycled",
        },
        "checkpoint": str(checkpoint_path),
        "assumptions": [
            "Only a0 (the first 0.5 s PSTL segment) is executed and stored; "
            "a1..a5 preview is a labelled placeholder and is not stored or credited.",
            "PPO ratios use the stored log-prob of the sampled normalized action; "
            "fallback/projected executed actions never receive that log-prob.",
            "Masked/fallback transitions are excluded from the policy loss; value "
            "and cost critics use all finite real transitions.",
            "GAE is per episode: termination uses 0 next value, truncation/max-step "
            "bootstraps V(s_T).",
            "Reward is route progress + success + comfort; collision/off-road/"
            "wrong-route/stuck/comfort/overspeed are a separate cost vector "
            "(scalar cost = vector sum + optional simulator info cost).",
            "boundary_rate counts sampled normalized components exactly at 0/1; "
            "the truncated Normal should keep it at 0.",
            "The stored-log-prob audit recomputes one transition at a time, "
            "matching the batch-size-1 rollout; the batched update recompute is "
            "logged separately as logprob_recompute_batch_max_abs_diff.",
            "TruncatedNormal support [0,1] equals the normalized NEW_BOUNDS box, so "
            "samples are executable without a post-hoc clamp.",
            "This is a contract canary; it makes no PPO performance claim.",
        ],
        "warnings": (
            []
            if update_records
            else ["no update ran because all episodes produced zero transitions"]
        ),
    }
    (out_dir / "contract.json").write_text(
        json.dumps(_jsonable(contract), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[ppo_v2] "
        + json.dumps(
            {
                key: contract[key]
                for key in (
                    "n_episodes",
                    "transitions",
                    "boundary_rate",
                    "postprocess_rate",
                    "mask_rate",
                    "fallback_rate",
                    "truncation_bootstrap_count",
                    "logprob_recompute_max_abs_diff",
                )
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"[ppo_v2] wrote {checkpoint_path} and {out_dir / 'contract.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
