"""Real-env group-relative PPO for the bank-native planner.

Contract reference: ``docs/arch_lock.md`` §12.4.

This trainer is the RL mainline. The old PPO contract canary has been removed.
It implements the RL contract:

* scene-level RL splits from ``dataset.splits`` (train 1500 / holdout 500);
* real-env rollouts in spawned worker processes, 8 scene seeds x 8 policy
  samples per update, identical env seed per scene, independent policy
  sampling seed per member;
* one 0.5 s policy decision = 5 x 0.1 s env steps, only the first segment is
  executed, exact execution (no projection, no safe brake);
* rule-based dense + terminal reward from ``pipeline.train.rl_reward``;
* per-step active-set leave-one-out advantages with global normalization;
* clipped PPO on planner + TSB only (encoder and aux head are frozen);
* holdout-500 selection using strict success, never test500 / val500.

There is no value head, no GAE, no cost vector and no reward-shaping module in
this trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import math
import multiprocessing as mp
import shutil
import sysconfig
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from dataset.splits import RLSpecSplit, load_rl_spec_splits
from env.action.action_space import ActionBounds, DynamicsLimits, action_to_trajectory
from env.action.config import load_action_contract
from env.metadrive_env import build_env
from env.obs import HISTORY_FRAMES, build_observation
from env.scenario.spec import load_specs
from env.tracking import ExactTracker
from net.mem_bank import BankWindow
from net.model import DriveModel
from pipeline.common.checkpointing import save_drive_model_checkpoint
from pipeline.common.distributions import TruncatedNormal
from pipeline.eval.closed_loop import load_model, run_episode
from pipeline.common.execution import relative_action
from pipeline.common.live_bank import LiveBankRunner
from pipeline.train.rl_reward import (
    RewardConfig,
    RewardTracker,
    aggregate_substep_events,
    compute_step_reward,
)
from pipeline.common.train_config import latest_run_alias, parse_train_args, timestamped_run_dir

# The stored-log-prob audit compares a batch recompute with the rollout forward.
# TF32 would silently change the apples-to-apples comparison on CUDA.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DECISION_DT = 0.5
STEPS_PER_POLICY = 5
HZ = 10
ACTION_DIM = 4
SUPPORT_LOW = 0.0
SUPPORT_HIGH = 1.0
DEFAULT_MAX_DECISIONS = 120
LOGPROB_AUDIT_TOL = 1e-3

_CONTRACT = load_action_contract()
CLEAN_BOUNDS = _CONTRACT.bounds
HARD_LIMITS = _CONTRACT.hard
COMFORT_LIMITS = _CONTRACT.comfort


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return float(out)


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


def _model_action_contract(model: DriveModel):
    """Checkpoint action bounds/limits, falling back to the canonical contract."""
    if getattr(model, "action_low", None) is None or getattr(model, "action_high", None) is None:
        return CLEAN_BOUNDS, HARD_LIMITS, COMFORT_LIMITS
    bounds = ActionBounds(
        low=model.action_low.detach().cpu().numpy(),
        high=model.action_high.detach().cpu().numpy(),
        name=getattr(model, "action_config", {}).get("name", "checkpoint"),
    )
    limits = getattr(model, "action_config", {}).get("limits", {}) or {}
    hard = DynamicsLimits(**limits["hard"]) if "hard" in limits else HARD_LIMITS
    comfort = DynamicsLimits(**limits["comfort"]) if "comfort" in limits else COMFORT_LIMITS
    return bounds, hard, comfort


def _resolve_exact_execution(
    action: Sequence[float],
    v0: float,
    *,
    bounds: ActionBounds,
    limits: DynamicsLimits,
    comfort_limits: DynamicsLimits,
) -> Tuple[np.ndarray, Any]:
    """Return the requested action and its exact PSTL trajectory.

    The action is never projected, clamped or replaced.  ``None`` means the
    action is non-finite/unconstructible and the episode must terminate.
    """
    requested = np.asarray(action, dtype=np.float64).reshape(-1)
    if requested.shape != (ACTION_DIM,):
        return requested, None
    try:
        trajectory = action_to_trajectory(
            requested,
            float(v0),
            dt=DECISION_DT,
            hz=HZ,
            bounds=bounds,
            limits=limits,
            comfort_limits=comfort_limits,
        )
    except (TypeError, ValueError):
        return requested, None
    return requested, trajectory


def _executable(trajectory: Any) -> bool:
    """Mirror the evaluator's faithful-execution validity checks."""
    if trajectory is None:
        return False
    speeds = np.asarray(getattr(trajectory, "speeds", []), dtype=np.float64).reshape(-1)
    if speeds.size == 0 or not np.all(np.isfinite(speeds)) or float(speeds.min()) < -1e-6:
        return False
    return True


def _policy_distribution(out: Mapping[str, Tensor]) -> TruncatedNormal:
    return TruncatedNormal(
        out["action_mu"],
        out["action_logstd"].clamp(-5.0, 0.0).exp(),
        low=SUPPORT_LOW,
        high=SUPPORT_HIGH,
    )


def _policy_log_prob(out: Mapping[str, Tensor], sampled_norm: Tensor) -> Tensor:
    return _policy_distribution(out).log_prob(sampled_norm).sum(-1)


# ---------------------------------------------------------------------------
# Trainable parameter groups
# ---------------------------------------------------------------------------
def configure_trainable(model: DriveModel) -> Dict[str, int]:
    """Freeze everything except the planner head and the shared TSB.

    The encoder, aux safety head and world model are excluded from the RL
    optimizer.  ``LiveBankRunner`` already keeps encoder encoding no-grad.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    plan = getattr(getattr(model, "planner", None), "plan", None)
    if plan is None:
        plan = model.planner
    for parameter in plan.parameters():
        parameter.requires_grad_(True)
    for parameter in model.temporal_spatial.parameters():
        parameter.requires_grad_(True)

    aux = getattr(getattr(model, "planner", None), "aux", None)
    if aux is not None:
        for parameter in aux.parameters():
            parameter.requires_grad_(False)

    def count(module) -> int:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

    counts = {
        "planner_trainable": count(plan),
        "tsb_trainable": count(model.temporal_spatial),
        "encoder_trainable": count(model.encoder),
        "aux_trainable": count(aux) if aux is not None else 0,
        "wm_trainable": count(model.world_model),
    }
    if counts["planner_trainable"] <= 0 or counts["tsb_trainable"] <= 0:
        raise RuntimeError(f"no trainable planner/TSB parameters: {counts}")
    if counts["encoder_trainable"] or counts["aux_trainable"] or counts["wm_trainable"]:
        raise RuntimeError(f"frozen RL modules received gradients: {counts}")
    return counts


def build_optimizer(
    model: DriveModel,
    *,
    lr: float,
    tsb_lr: float,
    weight_decay: float = 0.0,
) -> Tuple[torch.optim.Optimizer, Dict[str, int]]:
    counts = configure_trainable(model)
    plan = getattr(getattr(model, "planner", None), "plan", None)
    if plan is None:
        plan = model.planner
    planner_params = [p for p in plan.parameters() if p.requires_grad]
    tsb_params = [p for p in model.temporal_spatial.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": planner_params, "lr": float(lr), "name": "planner"},
            {"params": tsb_params, "lr": float(tsb_lr), "name": "tsb"},
        ],
        weight_decay=float(weight_decay),
    )
    return optimizer, counts


# ---------------------------------------------------------------------------
# Active-set group-relative advantages (pure functions, CPU-testable)
# ---------------------------------------------------------------------------
def compute_active_set_advantages(
    episodes: Sequence[Mapping[str, Any]],
    gamma: float,
) -> Tuple[List[List[Optional[float]]], Dict[str, Any]]:
    """Per-step active-set leave-one-out advantages.

    Each episode is ``{"transitions": [...]}`` where every transition has
    ``reward``, ``valid_for_ppo``, ``t`` and ``scene_seed``.  The terminal
    transition (including an invalid-action terminal-only transition) is part
    of the episode, so ``tau_i = len(transitions) - 1``.

    ``G_{i,t} = sum_{k=t}^{tau_i} gamma^(k-t) * reward_k``.  For every group
    step ``t`` the active set is ``{i : t <= tau_i}``; the baseline is the
    leave-one-out mean of the other active members.  Steps with fewer than
    three active members produce no advantage.

    Returns ``(advantages, stats)`` where ``advantages[i][t]`` is ``None`` for
    invalid-terminal transitions, skipped small-active-set steps and steps
    outside the episode.  Global normalization is intentionally *not* applied
    here; use :func:`normalize_advantages`.
    """
    gamma = float(gamma)
    if gamma < 0.0 or not math.isfinite(gamma):
        raise ValueError(f"gamma must be finite and >= 0, got {gamma!r}")

    episode_transitions: List[List[Mapping[str, Any]]] = []
    episode_scene: List[int] = []
    episode_tau: List[int] = []
    advantages: List[List[Optional[float]]] = []
    returns: List[List[float]] = []

    for episode in episodes:
        transitions = list(episode.get("transitions", []))
        episode_transitions.append(transitions)
        advantages.append([None] * len(transitions))
        if not transitions:
            returns.append([])
            episode_tau.append(-1)
            scene = int(episode.get("scene_seed", 0))
            episode_scene.append(scene)
            continue
        curve = [0.0] * len(transitions)
        running = 0.0
        for t in range(len(transitions) - 1, -1, -1):
            running = float(transitions[t].get("reward", 0.0)) + gamma * running
            curve[t] = running
        returns.append(curve)
        episode_tau.append(len(transitions) - 1)
        scene = episode.get("scene_seed", transitions[0].get("scene_seed", 0))
        episode_scene.append(int(scene))

    groups: Dict[int, List[int]] = {}
    for index, scene in enumerate(episode_scene):
        groups.setdefault(scene, []).append(index)

    stats: Dict[str, Any] = {
        "groups": int(len(groups)),
        "episodes": int(len(episodes)),
        "transitions": int(sum(len(t) for t in episode_transitions)),
        "active_steps": 0,
        "skipped_steps": 0,
        "active_members_sum": 0,
        "transitions_with_adv": 0,
    }

    for members in groups.values():
        taus = [episode_tau[i] for i in members]
        max_tau = max(taus) if taus else -1
        for t in range(max_tau + 1):
            active = [i for i, tau in zip(members, taus) if t <= tau]
            if len(active) < 3:
                stats["skipped_steps"] += 1
                continue
            stats["active_steps"] += 1
            stats["active_members_sum"] += len(active)
            for i in active:
                others = [returns[j][t] for j in active if j != i]
                baseline = sum(others) / float(len(others))
                advantages[i][t] = float(returns[i][t] - baseline)

    # Invalid-action terminal transitions are part of G and of the active set,
    # but never receive a PPO advantage.
    for i, transitions in enumerate(episode_transitions):
        for t, transition in enumerate(transitions):
            if not bool(transition.get("valid_for_ppo", True)):
                advantages[i][t] = None

    stats["transitions_with_adv"] = int(
        sum(1 for row in advantages for value in row if value is not None)
    )
    stats["mean_active_set"] = (
        float(stats["active_members_sum"]) / float(stats["active_steps"])
        if stats["active_steps"]
        else 0.0
    )
    return advantages, stats


def normalize_advantages(
    advantages: Sequence[Sequence[Optional[float]]],
    *,
    floor: float = 1e-6,
) -> Tuple[List[List[Optional[float]]], Dict[str, Any]]:
    """Global mean-0 / std-1 normalization with a std floor.

    Leave-one-out means were already applied per group; there is deliberately
    no per-group std division here (that would double-normalize).  If the
    global std is below ``floor`` the signal is degenerate: every non-None
    value becomes 0.0 and ``applied`` is False.
    """
    values = [float(v) for row in advantages for v in row if v is not None]
    if not values:
        return [list(row) for row in advantages], {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "applied": False,
            "floor": float(floor),
        }
    mean = float(np.mean(values))
    std = float(np.std(values))
    applied = bool(math.isfinite(std) and std >= float(floor))
    if not applied:
        normalized = [
            [None if v is None else 0.0 for v in row] for row in advantages
        ]
    else:
        normalized = [
            [None if v is None else float((float(v) - mean) / std) for v in row]
            for row in advantages
        ]
    return normalized, {
        "count": int(len(values)),
        "mean": mean,
        "std": std,
        "applied": applied,
        "floor": float(floor),
    }


def flatten_ppo_samples(
    episodes: Sequence[Mapping[str, Any]],
    advantages: Sequence[Sequence[Optional[float]]],
) -> List[Dict[str, Any]]:
    """Flatten valid transitions that have an advantage.

    Invalid-action terminal transitions stay in the advantage return curve but
    never enter the PPO batch.
    """
    samples: List[Dict[str, Any]] = []
    for episode, row in zip(episodes, advantages):
        transitions = list(episode.get("transitions", []))
        if len(transitions) != len(row):
            raise ValueError("advantage row length does not match episode transitions")
        for transition, advantage in zip(transitions, row):
            if not bool(transition.get("valid_for_ppo", True)):
                continue
            if advantage is None:
                continue
            value = float(advantage)
            if not math.isfinite(value):
                continue
            item = dict(transition)
            item["_advantage"] = value
            samples.append(item)
    return samples


# ---------------------------------------------------------------------------
# Rollout (worker process)
# ---------------------------------------------------------------------------
_WORKER: Dict[str, Any] = {}


def _worker_init(ckpt: str, device: str) -> None:
    model = load_model(ckpt, device, strict=True)
    model.eval()
    _WORKER["model"] = model
    _WORKER["device"] = str(device)


def _worker_rollout(task: Mapping[str, Any]) -> Dict[str, Any]:
    return run_rollout_episode(
        _WORKER["model"],
        task["spec"],
        device=_WORKER["device"],
        scene_seed=int(task["scene_seed"]),
        member=int(task["member"]),
        sampling_seed=int(task["sampling_seed"]),
        max_steps=int(task["max_steps"]),
        max_decisions=int(task["max_decisions"]),
        reward_config=task["reward_config"],
    )


def _invalid_terminal_transition(
    *,
    state: Mapping[str, Tensor],
    sampled: Tensor,
    old_logp: Tensor,
    policy_step: int,
    scene_seed: int,
    member: int,
    v_end: float,
    rc: float,
    reward_config: RewardConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Terminal-only transition for an invalid/non-executable action.

    No env step is taken.  The transition is kept for the return curve and the
    active set, but ``valid_for_ppo=False`` excludes it from the PPO batch.
    """
    res = compute_step_reward(
        rc0=float(rc),
        rc1=float(rc),
        v_end=float(v_end),
        metrics={},
        invalid_action=True,
        config=reward_config,
    )
    transition = {
        "state": state,
        "sampled_norm": sampled[0].detach().cpu(),
        "old_logp": old_logp.detach().cpu(),
        "reward": torch.tensor(float(res["reward"]), dtype=torch.float32),
        "done": True,
        "terminal_event": "invalid_action",
        "components": dict(res["components"]),
        "valid_for_ppo": False,
        "t": int(policy_step),
        "scene_seed": int(scene_seed),
        "member": int(member),
    }
    return transition, res


def _timeout_terminal_transition(
    *,
    state: Mapping[str, Tensor],
    policy_step: int,
    scene_seed: int,
    member: int,
    v_end: float,
    rc: float,
    reward_config: RewardConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Terminal-only timeout transition when the decision cap is reached.

    The last executable action already has its dense reward; this extra
    terminal-only entry carries the timeout penalty so the return curve and
    active set still credit the terminal event.  It is excluded from PPO.
    """
    res = compute_step_reward(
        rc0=float(rc),
        rc1=float(rc),
        v_end=float(v_end),
        metrics={},
        timeout=True,
        config=reward_config,
    )
    transition = {
        "state": state,
        "sampled_norm": torch.zeros(ACTION_DIM),
        "old_logp": torch.tensor(0.0),
        "reward": torch.tensor(float(res["reward"]), dtype=torch.float32),
        "done": True,
        "terminal_event": "timeout",
        "components": dict(res["components"]),
        "valid_for_ppo": False,
        "t": int(policy_step),
        "scene_seed": int(scene_seed),
        "member": int(member),
    }
    return transition, res


def run_rollout_episode(
    model: DriveModel,
    spec: Any,
    *,
    device: str = "cpu",
    scene_seed: Optional[int] = None,
    member: int = 0,
    sampling_seed: int = 0,
    max_steps: int = 600,
    max_decisions: int = DEFAULT_MAX_DECISIONS,
    reward_config: RewardConfig = RewardConfig(),
) -> Dict[str, Any]:
    """Run one real-env on-policy episode and return transitions + summary.

    ``max_steps`` counts 0.1 s env steps; ``max_decisions`` caps policy
    decisions independently.  Only the first 0.5 s segment is executed.  An
    invalid action creates a terminal-only transition with
    ``valid_for_ppo=False`` (it stays in the return curve), never a rewrite of
    the previous transition.
    """
    max_steps = int(max_steps)
    max_decisions = int(max_decisions)
    if max_steps < 0 or max_decisions < 0:
        raise ValueError("max_steps and max_decisions must be >= 0")
    scene_seed = int(getattr(spec, "seed", 0) if scene_seed is None else scene_seed)

    model.to(device).eval()
    bounds, hard, comfort = _model_action_contract(model)

    env = build_env(spec, use_render=False, lru_size=0)
    try:
        env.reset(seed=scene_seed)
        # Policy sampling must be independent per member while the environment
        # seed stays identical across the group.  Seed *after* env.reset so
        # simulator-side RNG use cannot perturb the policy draw stream.
        torch.manual_seed(int(sampling_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sampling_seed))
        ego = env.agent
        runner = LiveBankRunner(model, device=device, capacity=HISTORY_FRAMES)
        tracker = ExactTracker(dt=DECISION_DT, hz=HZ)
        reward_tracker = RewardTracker(reward_config)

        transitions: List[Dict[str, Any]] = []
        prev_action = np.zeros(ACTION_DIM, dtype=np.float64)
        prev_speed: Optional[float] = None
        prev_heading: Optional[float] = None
        rc = _route_completion(ego, {}, 0.0)

        n = 0
        success = collision = off_road = False
        stuck = False
        invalid_action = False
        terminal_event = "none"
        reward_sum = 0.0
        done = False

        while not done and n < max_steps and len(transitions) < max_decisions:
            policy_step = len(transitions)
            obs = build_observation(
                env,
                prev_action=prev_action,
                prev_speed=prev_speed,
                prev_heading=prev_heading,
                dt=DECISION_DT,
            )
            window, nav, ego_hidden = runner.push(obs)
            with torch.no_grad():
                out = model.forward_window(window, nav, ego_hidden)
                dist = _policy_distribution(out)
                sampled = dist.sample()
                old_logp = dist.log_prob(sampled).sum(-1)[0]

            sampled_np = sampled[0].detach().cpu().numpy().astype(np.float64)
            v0 = float(ego.speed)
            h0 = float(ego.heading_theta)
            p0 = np.asarray(ego.position[:2], dtype=np.float64).copy()
            state = {
                "window": window.to("cpu"),
                "nav": nav.detach().cpu(),
                "ego": ego_hidden.detach().cpu(),
            }
            rc0 = _route_completion(ego, {}, rc)
            requested = np.asarray(bounds.denormalize(sampled_np, clip=False), dtype=np.float64)
            _requested, tr_exec = _resolve_exact_execution(
                requested,
                v0,
                bounds=bounds,
                limits=hard,
                comfort_limits=comfort,
            )

            if not _executable(tr_exec):
                invalid_action = True
                transition, res = _invalid_terminal_transition(
                    state=state,
                    sampled=sampled,
                    old_logp=old_logp,
                    policy_step=policy_step,
                    scene_seed=int(scene_seed),
                    member=int(member),
                    v_end=float(ego.speed),
                    rc=rc0,
                    reward_config=reward_config,
                )
                transitions.append(transition)
                reward_sum += float(res["reward"])
                terminal_event = "invalid_action"
                done = True
                break

            try:
                tracker.arm(env, trajectory=tr_exec)
            except (TypeError, ValueError):
                invalid_action = True
                transition, res = _invalid_terminal_transition(
                    state=state,
                    sampled=sampled,
                    old_logp=old_logp,
                    policy_step=policy_step,
                    scene_seed=int(scene_seed),
                    member=int(member),
                    v_end=float(ego.speed),
                    rc=rc0,
                    reward_config=reward_config,
                )
                transitions.append(transition)
                reward_sum += float(res["reward"])
                terminal_event = "invalid_action"
                done = True
                break

            infos: List[Dict[str, Any]] = []
            env_done = False
            term = trunc = False
            for _ in range(STEPS_PER_POLICY):
                if n >= max_steps:
                    break
                _, _, term, trunc, info = env.step([0.0, 0.0])
                n += 1
                tracker.apply(env)
                infos.append(info or {})
                if term or trunc:
                    env_done = True
                    break

            rc1 = _route_completion(ego, infos[-1], rc0) if infos else rc0
            v_end = float(ego.speed)
            metrics = dict(getattr(tr_exec, "metrics", {}) or {})
            events = aggregate_substep_events(
                infos,
                rc0=rc0,
                rc1=rc1,
                v_end=v_end,
                timeout=False,
                invalid_action=False,
                metrics=metrics,
            )
            known_terminal = bool(
                events.collision or events.off_road or events.arrive_dest
            )
            timed_out = bool(
                (env_done or n >= max_steps or len(transitions) + 1 >= max_decisions)
                and not known_terminal
            )
            if timed_out and not events.timeout:
                events = aggregate_substep_events(
                    infos,
                    rc0=rc0,
                    rc1=rc1,
                    v_end=v_end,
                    timeout=True,
                    invalid_action=False,
                    metrics=metrics,
                )

            one_shot = reward_tracker.update(rc0, rc1, v_end)
            res = compute_step_reward(
                rc0=rc0,
                rc1=rc1,
                v_end=v_end,
                metrics=metrics,
                success=events.arrive_dest,
                collision=events.collision,
                off_road=events.off_road,
                timeout=events.timeout,
                invalid_action=False,
                wrong_route_event=bool(one_shot["wrong_route_event"]),
                stuck_event=bool(one_shot["stuck_event"]),
                config=reward_config,
            )
            transitions.append(
                {
                    "state": state,
                    "sampled_norm": sampled[0].detach().cpu(),
                    "old_logp": old_logp.detach().cpu(),
                    "reward": torch.tensor(float(res["reward"]), dtype=torch.float32),
                    "done": bool(res["done"]),
                    "terminal_event": str(res["terminal_event"]),
                    "components": dict(res["components"]),
                    "valid_for_ppo": True,
                    "t": int(policy_step),
                    "scene_seed": int(scene_seed),
                    "member": int(member),
                }
            )
            reward_sum += float(res["reward"])
            rc = rc1
            stuck = bool(stuck or one_shot["stuck_event"])

            if bool(res["done"]):
                done = True
                terminal_event = str(res["terminal_event"])
                success = terminal_event == "success"
                collision = terminal_event == "collision"
                off_road = terminal_event == "off_road"
                break

            realized_action = relative_action(
                p0,
                h0,
                v0,
                np.asarray(ego.position[:2], dtype=np.float64),
                float(ego.heading_theta),
                float(ego.speed),
            )
            prev_action = realized_action
            prev_speed = v0
            prev_heading = h0

        if not done and len(transitions) >= max_decisions:
            # max_decisions is a cap on the number of decision transitions;
            # if the episode is still alive, charge an explicit timeout at the
            # next (action-less) decision so G_t still sees the terminal cost.
            last_state = transitions[-1]["state"] if transitions else None
            if last_state is None:
                raise RuntimeError("max_decisions reached without a transition state")
            terminal, _res = _timeout_terminal_transition(
                state=last_state,
                policy_step=len(transitions),
                scene_seed=int(scene_seed),
                member=int(member),
                v_end=float(ego.speed),
                rc=float(rc),
                reward_config=reward_config,
            )
            transitions.append(terminal)
            terminal_event = "timeout"
            done = True

        decision_count = len(transitions)
        summary = {
            "scene_seed": int(scene_seed),
            "spec_id": getattr(spec, "id", None),
            "member": int(member),
            "sampling_seed": int(sampling_seed),
            "steps": int(n),
            "policy_steps": int(decision_count),
            "decision_count": int(decision_count),
            "success": bool(success),
            "collision": bool(collision),
            "off_road": bool(off_road),
            "timeout": bool(terminal_event == "timeout"),
            "stuck": bool(stuck),
            "invalid_action": bool(invalid_action),
            "terminal_event": str(terminal_event),
            "route_completion": float(rc),
            "reward_sum": float(reward_sum),
            "transitions": int(decision_count),
        }
        return {
            "scene_seed": int(scene_seed),
            "member": int(member),
            "sampling_seed": int(sampling_seed),
            "transitions": transitions,
            "summary": summary,
        }
    finally:
        env.close()


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def _collate_batch(
    samples: Sequence[Mapping[str, Any]],
    device: str,
) -> Tuple[BankWindow, Tensor, Tensor, Tensor, Tensor]:
    """Move one minibatch of CPU bank windows to the device.

    Rollout storage keeps every ``BankWindow`` on CPU; only the current
    minibatch is stacked/moved.  This avoids a full-batch GPU stack.
    """
    windows = BankWindow.stack([sample["state"]["window"] for sample in samples]).to(device)
    nav = torch.cat([sample["state"]["nav"] for sample in samples], dim=0).to(device)
    ego = torch.cat([sample["state"]["ego"] for sample in samples], dim=0).to(device)
    sampled = torch.stack([sample["sampled_norm"] for sample in samples], dim=0).to(device).float()
    old_logp = torch.stack([sample["old_logp"] for sample in samples], dim=0).to(device).float()
    return windows, nav, ego, sampled, old_logp


def _advantages_tensor(samples: Sequence[Mapping[str, Any]], device: str) -> Tensor:
    return torch.tensor(
        [float(sample["_advantage"]) for sample in samples],
        dtype=torch.float32,
        device=device,
    )


def ppo_update(
    model: DriveModel,
    optimizer: torch.optim.Optimizer,
    samples: Sequence[Mapping[str, Any]],
    *,
    clip: float = 0.2,
    epochs: int = 4,
    minibatch: int = 256,
    device: str = "cpu",
    max_grad_norm: float = 0.5,
    seed: int = 0,
    kl_stop: float = 0.02,
    clip_frac_stop: float = 0.3,
    logprob_audit_size: int = 32,
) -> Dict[str, Any]:
    """Clipped PPO update on pre-flattened valid transitions."""
    samples = list(samples)
    if not samples:
        raise ValueError("ppo_update: no valid samples")
    if int(epochs) < 1 or int(minibatch) < 1:
        raise ValueError("epochs and minibatch must be >= 1")

    model.to(device)
    audit_count = min(len(samples), max(int(logprob_audit_size), 1))
    audit = samples[:audit_count]

    # Audit: a batched recompute at the old parameters must reproduce the
    # stored rollout log-prob.  A mismatch means the update would optimize a
    # different distribution than the one that acted.
    model.eval()
    with torch.no_grad():
        windows, nav, ego, sampled, old_logp = _collate_batch(audit, device)
        out = model.forward_window(windows, nav, ego)
        recomputed = _policy_log_prob(out, sampled)
        audit_max_diff = float((recomputed - old_logp).abs().max().item())
    model.train()
    if audit_max_diff > LOGPROB_AUDIT_TOL:
        raise RuntimeError(
            "stored log-prob does not match the old policy: "
            f"max |new-old|={audit_max_diff:.3e} > {LOGPROB_AUDIT_TOL:.1e}"
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    epoch_history: List[Dict[str, float]] = []

    for epoch in range(int(epochs)):
        order = torch.randperm(len(samples), generator=generator).tolist()
        totals = {
            "policy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "grad_norm": 0.0,
            "ratio_mean": 0.0,
            "entropy_proxy": 0.0,
            "batches": 0,
        }
        for start in range(0, len(samples), int(minibatch)):
            index = order[start : start + int(minibatch)]
            batch = [samples[i] for i in index]
            windows, nav, ego, sampled, old_logp = _collate_batch(batch, device)
            out = model.forward_window(windows, nav, ego)
            log_prob = _policy_log_prob(out, sampled)
            advantage = _advantages_tensor(batch, device)

            log_ratio = (log_prob - old_logp).clamp(-20.0, 20.0)
            ratio = log_ratio.exp()
            surrogate = torch.min(
                ratio * advantage,
                torch.clamp(ratio, 1.0 - float(clip), 1.0 + float(clip)) * advantage,
            )
            loss = -surrogate.mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite PPO loss at epoch {epoch}: {loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grads = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            grad_norm = torch.nn.utils.clip_grad_norm_(grads, float(max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_frac = ((ratio - 1.0).abs() > float(clip)).float().mean()
                totals["policy_loss"] += float(loss.detach())
                totals["approx_kl"] += float(approx_kl)
                totals["clip_frac"] += float(clip_frac)
                totals["grad_norm"] += float(grad_norm)
                totals["ratio_mean"] += float(ratio.mean())
                totals["entropy_proxy"] += float(
                    out["action_logstd"].clamp(-5.0, 0.0).mean()
                )
                totals["batches"] += 1

        batches = max(int(totals["batches"]), 1)
        history = {
            "policy_loss": totals["policy_loss"] / batches,
            "approx_kl": totals["approx_kl"] / batches,
            "clip_frac": totals["clip_frac"] / batches,
            "grad_norm": totals["grad_norm"] / batches,
            "ratio_mean": totals["ratio_mean"] / batches,
            "entropy_proxy": totals["entropy_proxy"] / batches,
        }
        epoch_history.append(history)
        if history["approx_kl"] > float(kl_stop) or history["clip_frac"] > float(clip_frac_stop):
            break

    last = epoch_history[-1]
    return {
        "samples": int(len(samples)),
        "minibatches": int(math.ceil(len(samples) / float(minibatch))),
        "epochs_run": int(len(epoch_history)),
        "policy_loss": last["policy_loss"],
        "approx_kl": last["approx_kl"],
        "clip_frac": last["clip_frac"],
        "grad_norm": last["grad_norm"],
        "ratio_mean": last["ratio_mean"],
        "entropy_proxy": last["entropy_proxy"],
        "logprob_audit_count": int(audit_count),
        "logprob_audit_max_abs_diff": audit_max_diff,
        "epochs": epoch_history,
    }


# ---------------------------------------------------------------------------
# Holdout evaluation / selection
# ---------------------------------------------------------------------------
def evaluate_holdout(
    model: DriveModel,
    specs: Sequence[Any],
    *,
    device: str,
    max_steps: int = 600,
    limit: int = 100,
) -> Dict[str, Any]:
    """Sequential exact-execution evaluation on the RL holdout subset."""
    selected = list(specs)[: int(limit)]
    if not selected:
        raise ValueError("evaluate_holdout: empty holdout selection")
    rows: List[Dict[str, Any]] = []
    for spec in selected:
        env = build_env(spec, use_render=False, lru_size=0)
        try:
            row = run_episode(
                env,
                model,
                spec,
                controller="exact",
                max_steps=int(max_steps),
                device=device,
            )
        finally:
            env.close()
        rows.append(
            {
                "scene_seed": int(row["scene_seed"]),
                "success": bool(row["success"]),
                "collision": bool(row["collision"]),
                "off_road": bool(row["off_road"]),
                "timeout": bool(row["timeout"]),
                "stuck": bool(row["stuck"]),
                "route_completion": float(row["route_completion"]),
                "strict_success": bool(
                    row["success"] and not row["collision"] and not row["off_road"]
                ),
            }
        )

    count = float(len(rows))

    def rate(key: str) -> float:
        return float(sum(1.0 for row in rows if row[key]) / count)

    route = float(sum(row["route_completion"] for row in rows) / count)
    return {
        "episodes": int(len(rows)),
        "limit": int(limit),
        "strict_success": float(
            sum(1.0 for row in rows if row["strict_success"]) / count
        ),
        "legacy_success": rate("success"),
        "collision": rate("collision"),
        "off_road": rate("off_road"),
        "timeout": rate("timeout"),
        "stuck": rate("stuck"),
        "route_completion": route,
        "rows": rows,
    }


def _selection_key(metrics: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    return (
        float(metrics["strict_success"]),
        -float(metrics["collision"]),
        -float(metrics["off_road"]),
        float(metrics["route_completion"]),
    )


# ---------------------------------------------------------------------------
# Official Python / site-packages read-only audit
# ---------------------------------------------------------------------------
def _audit_roots() -> List[Path]:
    roots: List[Path] = []
    paths = sysconfig.get_paths()
    for key in ("stdlib", "purelib", "platlib"):
        value = paths.get(key)
        if value:
            roots.append(Path(value))
    for entry in sys.path:
        path = Path(entry) if entry else None
        if path is not None and path.name in ("site-packages", "dist-packages"):
            roots.append(path)
    resolved: List[Path] = []
    seen = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root.is_dir() and str(root) not in seen:
            seen.add(str(root))
            resolved.append(root)
    return resolved


def snapshot_official_python() -> Dict[str, Tuple[int, int]]:
    """Snapshot ``.py`` files under stdlib / site-packages.

    The project policy is that the official Python installation and every
    third-party package are read-only.  Comparing size+mtime before and after
    a run catches accidental edits or package installs without hashing the
    whole environment.
    """
    snapshot: Dict[str, Tuple[int, int]] = {}
    for root in _audit_roots():
        try:
            iterator = root.rglob("*.py")
            for path in iterator:
                try:
                    stat = path.stat()
                except OSError:
                    continue
                snapshot[str(path)] = (int(stat.st_size), int(stat.st_mtime_ns))
        except OSError:
            continue
    return snapshot


def assert_official_python_untouched(
    before: Mapping[str, Tuple[int, int]],
) -> None:
    after = snapshot_official_python()
    if after == before:
        return
    added = sorted(set(after) - set(before))[:3]
    removed = sorted(set(before) - set(after))[:3]
    changed = sorted(
        path for path in set(before) & set(after) if before[path] != after[path]
    )[:3]
    raise RuntimeError(
        "official Python/site-packages was modified during the run: "
        f"added={added} removed={removed} changed={changed}"
    )


# ---------------------------------------------------------------------------
# Scene sampling / summaries
# ---------------------------------------------------------------------------
def sample_scenes(
    specs: Sequence[Any],
    count: int,
    *,
    weights: Optional[Mapping[str, float]] = None,
    rng: np.random.Generator,
) -> List[Any]:
    """Weighted sampling without replacement; difficulty oversampling."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be >= 1")
    if count > len(specs):
        raise ValueError(f"cannot sample {count} distinct scenes from {len(specs)}")
    remaining = list(specs)
    chosen: List[Any] = []
    difficulty_weights = dict(weights or {})
    for _ in range(count):
        probabilities = np.asarray(
            [
                max(float(difficulty_weights.get(str(getattr(spec, "difficulty", "")), 1.0)), 1e-9)
                for spec in remaining
            ],
            dtype=np.float64,
        )
        probabilities = probabilities / probabilities.sum()
        index = int(rng.choice(len(remaining), p=probabilities))
        chosen.append(remaining.pop(index))
    return chosen


def summarize_episodes(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not summaries:
        return {
            "episodes": 0,
            "success": 0.0,
            "collision": 0.0,
            "off_road": 0.0,
            "timeout": 0.0,
            "stuck": 0.0,
            "invalid_action": 0.0,
            "route_completion": 0.0,
            "decisions_mean": 0.0,
            "reward_mean": 0.0,
        }
    count = float(len(summaries))

    def rate(key: str) -> float:
        return float(sum(1.0 for item in summaries if item[key]) / count)

    return {
        "episodes": int(len(summaries)),
        "success": rate("success"),
        "collision": rate("collision"),
        "off_road": rate("off_road"),
        "timeout": rate("timeout"),
        "stuck": rate("stuck"),
        "invalid_action": rate("invalid_action"),
        "route_completion": float(
            sum(float(item["route_completion"]) for item in summaries) / count
        ),
        "decisions_mean": float(
            sum(float(item["decision_count"]) for item in summaries) / count
        ),
        "reward_mean": float(
            sum(float(item["reward_sum"]) for item in summaries) / count
        ),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_policy_checkpoint(
    model: DriveModel,
    path: Path,
    *,
    extra: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    return save_drive_model_checkpoint(model, str(path), extra=dict(extra))


def _write_latest_alias(out_dir: Path) -> None:
    target = out_dir / "last.pt"
    link = out_dir / "latest.pt"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.name)
    except OSError:
        shutil.copyfile(target, link)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-env group-relative PPO (arch_lock §12.4)"
    )
    parser.add_argument("--config", default="config/train_rl.yaml")
    parser.add_argument("--specs", default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--holdout-manifest", default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--out", default=None)

    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--scenes-per-update", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-decisions", type=int, default=DEFAULT_MAX_DECISIONS)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--tsb-lr", type=float, default=3e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--kl-stop", type=float, default=0.02)
    parser.add_argument("--clip-frac-stop", type=float, default=0.3)
    parser.add_argument("--adv-std-floor", type=float, default=1e-6)
    parser.add_argument("--logprob-audit-size", type=int, default=32)

    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-limit", type=int, default=100)
    parser.add_argument("--eval-max-steps", type=int, default=600)

    # Optional reward-weight overrides; RewardConfig defaults are the contract.
    parser.add_argument("--w-progress", type=float, default=None)
    parser.add_argument("--w-success", type=float, default=None)
    parser.add_argument("--w-collision", type=float, default=None)
    parser.add_argument("--w-offroad", type=float, default=None)
    parser.add_argument("--w-timeout", type=float, default=None)
    parser.add_argument("--w-invalid", type=float, default=None)
    parser.add_argument("--adapt-coeff", type=float, default=None)
    return parser


def reward_config_from_args(args: argparse.Namespace) -> RewardConfig:
    overrides = {
        name: getattr(args, name)
        for name in (
            "w_progress", "w_success", "w_collision", "w_offroad",
            "w_timeout", "w_invalid", "adapt_coeff",
        )
        if getattr(args, name, None) is not None
    }
    return RewardConfig(**overrides)


def _sampling_seed(seed: int, update: int, scene_index: int, member: int) -> int:
    value = (
        int(seed) * 1_000_003
        + int(update) * 10_007
        + int(scene_index) * 211
        + int(member) * 17
        + 1
    )
    return int(value % (2**31 - 1))


def _validate_args(args: argparse.Namespace) -> None:
    required = {
        "specs": args.specs,
        "train_manifest": args.train_manifest,
        "holdout_manifest": args.holdout_manifest,
        "ckpt": args.ckpt,
        "out": args.out,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise SystemExit(f"missing required config values: {missing}")
    if args.updates < 1:
        raise SystemExit("--updates must be >= 1")
    if args.scenes_per_update < 1 or args.group_size < 1:
        raise SystemExit("--scenes-per-update and --group-size must be >= 1")
    if args.max_steps < 0 or args.max_decisions < 0:
        raise SystemExit("--max-steps and --max-decisions must be >= 0")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.group_size < 3:
        raise SystemExit("--group-size must be >= 3 for active-set LOO")
    if args.max_decisions < 1:
        raise SystemExit("--max-decisions must be >= 1")
    if args.epochs < 1 or args.minibatch < 1:
        raise SystemExit("--epochs and --minibatch must be >= 1")
    if args.eval_every < 1:
        raise SystemExit("--eval-every must be >= 1")
    if not Path(args.ckpt).is_file():
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    if not Path(args.specs).is_file():
        raise SystemExit(f"spec file not found: {args.specs}")


def _checkpoint_extra(
    args: argparse.Namespace,
    reward_config: RewardConfig,
    *,
    update: int,
    selection: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "rl_group": {
            "schema": "drl_pathplan_rl_group_v1",
            "contract": "arch_lock §12.4",
            "update": int(update),
            "updates": int(args.updates),
            "scenes_per_update": int(args.scenes_per_update),
            "group_size": int(args.group_size),
            "gamma": float(args.gamma),
            "clip": float(args.clip),
            "epochs": int(args.epochs),
            "minibatch": int(args.minibatch),
            "max_steps": int(args.max_steps),
            "max_decisions": int(args.max_decisions),
            "lr": float(args.lr),
            "tsb_lr": float(args.tsb_lr),
            "max_grad_norm": float(args.max_grad_norm),
            "split": {
                "specs": str(args.specs),
                "train_manifest": str(args.train_manifest),
                "holdout_manifest": str(args.holdout_manifest),
            },
            "optimizer": {
                "name": "AdamW",
                "weight_decay": 0.0,
                "param_groups": ["planner", "tsb"],
                "encoder_frozen": True,
                "aux_frozen": True,
            },
        },
        "reward_config": asdict(reward_config),
        "holdout_selection": dict(selection or {}),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_train_args(build_arg_parser(), argv, "config/train_rl.yaml")
    _validate_args(args)
    reward_config = reward_config_from_args(args)

    if not (float(0.0) < float(args.gamma) <= 1.0):
        raise SystemExit("--gamma must be in (0, 1]")
    if not (0.0 < float(args.clip) < 1.0):
        raise SystemExit("--clip must be in (0, 1)")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    args.out = str(timestamped_run_dir(args.out))
    latest_run_alias(args.out)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_file = log_path.open("w", encoding="utf-8")

    def log(record: Mapping[str, Any]) -> None:
        line = json.dumps(_jsonable(record), ensure_ascii=False)
        log_file.write(line + "\n")
        log_file.flush()
        print(line, flush=True)

    print(f"[rl_group] config={args.config} out={out_dir}", flush=True)

    official_before = snapshot_official_python()

    specs = load_specs(args.specs)
    if not specs:
        raise SystemExit(f"no specs in {args.specs}")
    split: RLSpecSplit = load_rl_spec_splits(
        specs,
        train_manifest=args.train_manifest,
        holdout_manifest=args.holdout_manifest,
        specs_path=args.specs,
    )
    train_seeds = {int(spec.seed) for spec in split.train}
    holdout_seeds = {int(spec.seed) for spec in split.holdout}
    if train_seeds & holdout_seeds:
        raise SystemExit("RL train/holdout seed overlap")
    if len(split.train) < int(args.scenes_per_update):
        raise SystemExit(
            f"only {len(split.train)} train scenes for scenes_per_update={args.scenes_per_update}"
        )

    model = load_model(args.ckpt, args.device, strict=True)
    optimizer, trainable = build_optimizer(
        model,
        lr=float(args.lr),
        tsb_lr=float(args.tsb_lr),
        weight_decay=0.0,
    )

    run_meta = {
        "schema": "drl_pathplan_rl_group_run_v1",
        "config": str(args.config),
        "args": vars(args),
        "reward_config": asdict(reward_config),
        "split": {
            "train": len(split.train),
            "holdout": len(split.holdout),
            "difficulty_weights": dict(split.difficulty_weights),
            "train_seed_range": [min(train_seeds), max(train_seeds)],
            "holdout_seed_range": [min(holdout_seeds), max(holdout_seeds)],
        },
        "trainable": trainable,
        "official_python_files": int(len(official_before)),
        "execution": "exact; only the first 0.5 s segment; no projection/safe-brake",
        "reward": "pipeline.train.rl_reward rule model; no cost vector / Lagrangian",
        "advantage": "per-step active-set leave-one-out + global normalization",
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(_jsonable(run_meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(
        {
            "event": "start",
            "out": str(out_dir),
            "train": len(split.train),
            "holdout": len(split.holdout),
            "trainable": trainable,
            "reward_config": asdict(reward_config),
        }
    )

    rng = np.random.default_rng(int(args.seed))
    best_path = out_dir / "best_holdout.pt"
    previous_ckpt = str(Path(args.ckpt).resolve())

    # The pre-RL checkpoint is the rollback baseline.  Evaluate it on exactly
    # the same holdout subset before any update; never promote a checkpoint
    # that is worse than r3.
    baseline_eval = evaluate_holdout(
        model, split.holdout,
        device=str(args.device), max_steps=int(args.eval_max_steps),
        limit=int(args.eval_limit),
    )
    best_key: Optional[Tuple[float, float, float, float]] = _selection_key(baseline_eval)
    save_policy_checkpoint(
        model, best_path,
        extra=_checkpoint_extra(
            args, reward_config, update=0,
            selection={
                **{name: baseline_eval[name] for name in (
                    "strict_success", "legacy_success", "collision", "off_road",
                    "timeout", "stuck", "route_completion",
                )},
                "selection_key": list(best_key), "evaluated_update": 0,
                "baseline": True,
            },
        ),
    )
    log({
        "event": "baseline_holdout",
        "episodes": int(baseline_eval["episodes"]),
        "limit": int(baseline_eval["limit"]),
        "strict_success": float(baseline_eval["strict_success"]),
        "legacy_success": float(baseline_eval["legacy_success"]),
        "collision": float(baseline_eval["collision"]),
        "off_road": float(baseline_eval["off_road"]),
        "route_completion": float(baseline_eval["route_completion"]),
        "selection_key": list(best_key),
    })

    try:
        for update in range(int(args.updates)):
            scenes = sample_scenes(
                split.train,
                int(args.scenes_per_update),
                weights=split.difficulty_weights,
                rng=rng,
            )
            scene_seeds = [int(spec.seed) for spec in scenes]

            tasks: List[Dict[str, Any]] = []
            for scene_index, spec in enumerate(scenes):
                for member in range(int(args.group_size)):
                    tasks.append(
                        {
                            "spec": spec,
                            "scene_seed": int(spec.seed),
                            "member": int(member),
                            "sampling_seed": _sampling_seed(
                                int(args.seed), update, scene_index, member
                            ),
                            "max_steps": int(args.max_steps),
                            "max_decisions": int(args.max_decisions),
                            "reward_config": reward_config,
                        }
                    )

            context = mp.get_context("spawn")
            episodes: List[Dict[str, Any]] = []
            with context.Pool(
                processes=int(args.workers),
                initializer=_worker_init,
                initargs=(previous_ckpt, str(args.device)),
            ) as pool:
                for result in pool.imap_unordered(_worker_rollout, tasks, chunksize=1):
                    episodes.append(result)
            episodes.sort(key=lambda item: (item["scene_seed"], item["member"]))

            summaries = [episode["summary"] for episode in episodes]
            rollout_summary = summarize_episodes(summaries)
            advantages, advantage_stats = compute_active_set_advantages(
                episodes, float(args.gamma)
            )
            advantages, normalization_stats = normalize_advantages(
                advantages, floor=float(args.adv_std_floor)
            )
            samples = flatten_ppo_samples(episodes, advantages)
            valid_for_ppo = sum(
                1
                for episode in episodes
                for transition in episode["transitions"]
                if bool(transition.get("valid_for_ppo", True))
            )
            total_transitions = sum(
                len(episode["transitions"]) for episode in episodes
            )

            component_sums: Dict[str, float] = {}
            terminal_counts: Dict[str, int] = {}
            for episode in episodes:
                for transition in episode["transitions"]:
                    for key, value in dict(transition.get("components", {})).items():
                        component_sums[str(key)] = component_sums.get(str(key), 0.0) + float(value)
                    event = str(transition.get("terminal_event", "none"))
                    terminal_counts[event] = terminal_counts.get(event, 0) + 1
            denom = max(int(total_transitions), 1)
            update_record: Dict[str, Any] = {
                "event": "update",
                "update": int(update + 1),
                "scene_seeds": scene_seeds,
                "task_count": int(len(tasks)),
                "rollout": rollout_summary,
                "advantage": advantage_stats,
                "normalization": normalization_stats,
                "transitions": int(total_transitions),
                "valid_for_ppo": int(valid_for_ppo),
                "invalid_for_ppo": int(total_transitions - valid_for_ppo),
                "ppo_samples": int(len(samples)),
                "reward_components_mean": {
                    key: float(value / denom) for key, value in sorted(component_sums.items())
                },
                "terminal_event_counts": dict(sorted(terminal_counts.items())),
            }

            if samples and bool(normalization_stats["applied"]):
                ppo_stats = ppo_update(
                    model,
                    optimizer,
                    samples,
                    clip=float(args.clip),
                    epochs=int(args.epochs),
                    minibatch=int(args.minibatch),
                    device=str(args.device),
                    max_grad_norm=float(args.max_grad_norm),
                    seed=int(args.seed) + int(update),
                    kl_stop=float(args.kl_stop),
                    clip_frac_stop=float(args.clip_frac_stop),
                    logprob_audit_size=int(args.logprob_audit_size),
                )
                update_record["ppo"] = ppo_stats
                update_record["status"] = "updated"
            else:
                update_record["status"] = "skipped_no_advantage"
                update_record["reason"] = (
                    "normalization std below floor"
                    if samples
                    else "no valid advantages (group/active-set too small)"
                )

            checkpoint_path = out_dir / f"policy_update_{update + 1}.pt"
            save_policy_checkpoint(
                model,
                checkpoint_path,
                extra=_checkpoint_extra(
                    args, reward_config, update=int(update + 1)
                ),
            )
            save_policy_checkpoint(
                model,
                out_dir / "last.pt",
                extra=_checkpoint_extra(
                    args, reward_config, update=int(update + 1)
                ),
            )
            _write_latest_alias(out_dir)
            previous_ckpt = str(checkpoint_path.resolve())
            update_record["checkpoint"] = str(checkpoint_path)

            if int(update + 1) % int(args.eval_every) == 0 and int(args.eval_limit) > 0:
                eval_metrics = evaluate_holdout(
                    model,
                    split.holdout,
                    device=str(args.device),
                    max_steps=int(args.eval_max_steps),
                    limit=int(args.eval_limit),
                )
                key = _selection_key(eval_metrics)
                improved = key > best_key
                update_record["holdout"] = {
                    key_name: eval_metrics[key_name]
                    for key_name in (
                        "episodes",
                        "limit",
                        "strict_success",
                        "legacy_success",
                        "collision",
                        "off_road",
                        "timeout",
                        "stuck",
                        "route_completion",
                    )
                }
                update_record["holdout"]["selection_key"] = list(key)
                update_record["holdout"]["improved"] = bool(improved)
                (out_dir / f"holdout_update_{update + 1}.json").write_text(
                    json.dumps(_jsonable(eval_metrics["rows"]), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if improved:
                    best_key = key
                    save_policy_checkpoint(
                        model,
                        best_path,
                        extra=_checkpoint_extra(
                            args,
                            reward_config,
                            update=int(update + 1),
                            selection={
                                **{
                                    name: eval_metrics[name]
                                    for name in (
                                        "strict_success",
                                        "legacy_success",
                                        "collision",
                                        "off_road",
                                        "timeout",
                                        "stuck",
                                        "route_completion",
                                    )
                                },
                                "selection_key": list(key),
                                "evaluated_update": int(update + 1),
                            },
                        ),
                    )
                    update_record["holdout"]["best_checkpoint"] = str(best_path)

            log(update_record)

        split_summary = {
            "event": "done",
            "updates": int(args.updates),
            "best_selection_key": None if best_key is None else list(best_key),
            "best_checkpoint": str(best_path) if best_key is not None else None,
            "last_checkpoint": str(out_dir / "last.pt"),
            "latest_checkpoint": str(out_dir / "latest.pt"),
        }
        log(split_summary)
    finally:
        log_file.close()

    assert_official_python_untouched(official_before)
    print(f"[rl_group] wrote {out_dir / 'last.pt'} and {log_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
