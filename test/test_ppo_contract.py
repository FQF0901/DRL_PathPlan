"""Contract tests for the clean-schema PPO canary (schema v2).

These tests are CPU-only, simulator-free and use tiny synthetic transitions.
They check the probability-action contract, stored-log-prob recomputation,
per-episode GAE / max-step bootstrapping, buffer fields, reward/cost
separation, fallback masking and a one-update smoke.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from env.obs import _empty_history
from env.action.config import load_action_contract
from net.build_net import build_net
from net.model import DriveModel

_C = load_action_contract()
CLEAN_BOUNDS = _C.bounds
from pipeline import train_ppo as ppo
from pipeline.distributions import TruncatedNormal


def _tiny_model() -> DriveModel:
    torch.manual_seed(0)
    return build_net()


def _state() -> dict:
    """One valid synthetic schema-v2 history window, batch-first."""
    hist = _empty_history()
    hist["hist_valid"][:] = 1.0
    hist["od_mask"][:, 0] = 1.0
    hist["ld_mask"][:, 0, 0] = 1.0
    hist["nav_mask"][:] = 1.0
    hist["ego_mask"][:] = 1.0
    return ppo._batch_from_history(hist, device="cpu")


def _make_transition(
    model: DriveModel,
    *,
    state: dict | None = None,
    sampled_norm: torch.Tensor | None = None,
    old_value: float | None = None,
    old_cost_value: float | None = None,
    reward: float = 0.0,
    cost: float = 0.0,
    cost_vector: list[float] | None = None,
    env_cost: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    episode_id: int = 0,
    t: int = 0,
    fallback: int = 0,
    fallback_name: str = "none",
    masked: bool = False,
    bootstrap_value: float = 0.0,
    bootstrap_cost_value: float = 0.0,
) -> dict:
    state = _state() if state is None else state
    if sampled_norm is None:
        sampled_norm = torch.tensor([[0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    sampled_norm = torch.as_tensor(sampled_norm, dtype=torch.float32).reshape(1, 4)

    model.eval()
    with torch.no_grad():
        out = model(state)
        dist = TruncatedNormal(
            out["action_mu"],
            out["action_logstd"].clamp(-5.0, 0.0).exp(),
            low=0.0,
            high=1.0,
        )
        old_logp = dist.log_prob(sampled_norm).sum(-1)[0]
        if old_value is None:
            old_value = 0.0
        if old_cost_value is None:
            old_cost_value = 0.0

    executed = CLEAN_BOUNDS.denormalize(sampled_norm[0].numpy(), clip=True)
    if cost_vector is None:
        cost_vector = [float(cost), 0.0, 0.0, 0.0, 0.0, 0.0]
    return {
        "state": {key: value.detach().cpu() for key, value in state.items()},
        "sampled_norm": sampled_norm[0].detach().cpu(),
        "executed_action": torch.as_tensor(executed, dtype=torch.float32),
        "old_logp": old_logp.detach().cpu(),
        "old_value": torch.tensor(float(old_value), dtype=torch.float32),
        "old_cost_value": torch.tensor(float(old_cost_value), dtype=torch.float32),
        "reward": torch.tensor(float(reward), dtype=torch.float32),
        "reward_components": {
            "route_progress": float(reward),
            "success": 0.0,
            "comfort": 0.0,
        },
        "cost": torch.tensor(float(cost), dtype=torch.float32),
        "cost_vector": torch.tensor(cost_vector, dtype=torch.float32),
        "env_cost": torch.tensor(float(env_cost), dtype=torch.float32),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode_id": int(episode_id),
        "t": int(t),
        "fallback": int(fallback),
        "fallback_name": str(fallback_name),
        "masked": bool(masked),
        "codes": (),
        "requested_codes": (),
        "executed_codes": (),
        "bootstrap_value": torch.tensor(float(bootstrap_value), dtype=torch.float32),
        "bootstrap_cost_value": torch.tensor(
            float(bootstrap_cost_value), dtype=torch.float32
        ),
    }


# ---------------------------------------------------------------------------
# Probability / action contract
# ---------------------------------------------------------------------------
def test_truncated_normal_samples_stay_inside_support():
    loc = torch.tensor([[0.2, 0.5, 0.8, 0.3]])
    scale = torch.tensor([[0.2, 0.1, 0.2, 0.05]])
    dist = TruncatedNormal(loc, scale, low=0.0, high=1.0)

    samples = dist.sample((1024,))
    assert samples.shape == (1024, 1, 4)
    assert (samples >= 0.0).all()
    assert (samples <= 1.0).all()
    assert torch.isfinite(dist.log_prob(samples)).all()

    # A truncated distribution does not place stored samples on the support
    # edges; the helper must report 0 for interior values.
    assert ppo.boundary_rate(torch.full((3, 4), 0.5)) == 0.0
    assert ppo.boundary_rate(torch.tensor([[1.0, 0.5, 0.5, 0.5]])) == 1.0


def test_stored_logp_recomputes_at_old_params():
    model = _tiny_model().eval()
    state = _state()
    sampled = torch.tensor([[0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    transition = _make_transition(model, state=state, sampled_norm=sampled)

    buffer = ppo.RolloutBuffer(gamma=0.95, lam=0.95)
    buffer.add_transition(transition)
    batch = buffer.to_training_batch("cpu", gamma=0.95, lam=0.95)

    newlogp, _out = ppo.policy_logprob(model, batch["state"], batch["sampled_norm"])
    assert torch.allclose(newlogp, batch["old_logp"], atol=1e-6, rtol=0.0)
    assert float((newlogp - batch["old_logp"]).abs().max()) < 1e-5


def test_infeasible_sample_is_masked_and_not_rewritten():
    # dx=0 with v0=14 and v_next=15 is a zero-length path with speed: the
    # requested action is hard-infeasible, so the explicit safe brake executes.
    sampled = np.array([0.0, 0.5, 0.5, 1.0], dtype=np.float64)
    resolved = ppo.resolve_action(sampled, v0=14.0)

    assert resolved["masked"] is True
    assert resolved["fallback_name"] != "none"
    assert resolved["requested_action"][3] == pytest.approx(15.0)
    assert "ZERO_LENGTH_WITH_SPEED" in resolved["requested_codes"]
    assert not np.allclose(
        resolved["requested_action"], resolved["executed_action"]
    )


# ---------------------------------------------------------------------------
# Buffer / GAE
# ---------------------------------------------------------------------------
def test_buffer_has_required_transition_fields():
    model = _tiny_model().eval()
    transition = _make_transition(model)
    for key in ppo.REQUIRED_TRANSITION_KEYS:
        assert key in transition, key

    buffer = ppo.RolloutBuffer(gamma=0.95, lam=0.95)
    buffer.add_transition(transition)
    assert len(buffer) == 1
    batch = buffer.to_training_batch("cpu", gamma=0.95, lam=0.95)

    for key in ppo.INPUT_KEYS:
        assert key in batch["state"]
        assert batch["state"][key].shape[0] == 1
    for key in (
        "sampled_norm",
        "executed_action",
        "old_logp",
        "old_value",
        "old_cost_value",
        "reward",
        "cost",
        "cost_vector",
        "terminated",
        "truncated",
        "episode_id",
        "t",
        "fallback",
        "masked",
        "codes",
        "valid",
        "policy_valid",
        "adv",
        "ret",
        "cost_ret",
        "truncation_bootstrap",
    ):
        assert key in batch, key
    assert batch["sampled_norm"].shape == (1, 4)
    assert batch["executed_action"].shape == (1, 4)
    assert batch["cost_vector"].shape == (1, len(ppo.COST_NAMES))


def test_gae_two_episodes_and_max_step_bootstrap():
    model = _tiny_model().eval()
    state = _state()
    buffer = ppo.RolloutBuffer(gamma=0.5, lam=0.5)

    # Episode 0: 2 transitions; the last one is truncated at the max-step
    # boundary and must bootstrap V(s_T)=10 instead of using 0.
    # Episode 1: 1 terminal transition; the supplied bootstrap 999 must be
    # ignored and must not leak into episode 0.
    rows = [
        dict(episode_id=0, t=0, old_value=2.0, reward=1.0, terminated=False,
             truncated=False, bootstrap_value=0.0),
        dict(episode_id=0, t=1, old_value=3.0, reward=1.0, terminated=False,
             truncated=True, bootstrap_value=10.0),
        dict(episode_id=1, t=0, old_value=0.5, reward=2.0, terminated=True,
             truncated=False, bootstrap_value=999.0),
    ]
    for row in rows:
        buffer.add_transition(_make_transition(model, state=state, **row))

    batch = buffer.to_training_batch("cpu", gamma=0.5, lam=0.5)
    torch.testing.assert_close(
        batch["adv"], torch.tensor([1.25, 3.0, 1.5]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        batch["ret"], torch.tensor([3.25, 6.0, 2.0]), atol=1e-6, rtol=0.0
    )
    assert int(batch["truncation_bootstrap"].sum()) == 1
    assert torch.isfinite(batch["cost_ret"]).all()


# ---------------------------------------------------------------------------
# Reward / cost separation
# ---------------------------------------------------------------------------
def test_reward_and_cost_are_separate():
    clean = ppo.compute_reward_cost(
        route_completion_delta=0.1,
        success=False,
        comfortable=True,
    )
    crashed = ppo.compute_reward_cost(
        route_completion_delta=0.1,
        success=False,
        comfortable=True,
        collision=True,
        off_road=True,
        wrong_route=True,
        stuck=True,
    )
    # Safety events must not change the reward.
    assert crashed["reward"] == pytest.approx(clean["reward"])
    assert crashed["cost"] > clean["cost"]
    assert crashed["cost_vector"][0] == 1.0  # collision
    assert crashed["cost_vector"][1] == 1.0  # off-road
    assert crashed["cost_vector"][2] == 1.0  # wrong-route
    assert crashed["cost_vector"][3] == 1.0  # stuck

    success = ppo.compute_reward_cost(
        route_completion_delta=0.1,
        success=True,
        comfortable=True,
    )
    assert success["reward"] == pytest.approx(clean["reward"] + ppo.SUCCESS_REWARD)
    assert success["cost"] == pytest.approx(clean["cost"])

    uncomfortable = ppo.compute_reward_cost(
        route_completion_delta=0.1,
        success=False,
        comfortable=False,
    )
    assert uncomfortable["reward"] < clean["reward"]
    assert uncomfortable["cost_vector"][4] == 1.0  # comfort cost

    fast = ppo.compute_reward_cost(
        route_completion_delta=0.1,
        success=False,
        comfortable=True,
        overspeed_excess=2.0,
    )
    assert fast["cost_vector"][5] == pytest.approx(
        ppo.OVERSPEED_COST_SCALE * 2.0
    )
    assert fast["reward"] == pytest.approx(clean["reward"])


def test_contract_summary_exposes_required_metrics():
    model = _tiny_model().eval()
    transition = _make_transition(model, reward=0.5, cost=1.0, cost_vector=[1.0, 0, 0, 0, 0, 0])
    episodes = [
        {
            "episode_id": 0,
            "success": False,
            "collision": True,
            "off_road": False,
            "wrong_route": False,
            "stuck": False,
            "timeout": False,
            "route_completion": 0.25,
        }
    ]
    summary = ppo.summarize_rollout(episodes, [transition])

    for key in (
        "n_episodes",
        "transitions",
        "boundary_rate",
        "postprocess_rate",
        "mask_rate",
        "fallback_rate",
        "truncation_bootstrap_count",
        "reward",
        "cost",
        "outcomes",
    ):
        assert key in summary, key
    assert summary["n_episodes"] == 1
    assert summary["transitions"] == 1
    assert summary["boundary_rate"] == 0.0
    assert summary["cost"]["sum"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# One-update CPU smoke
# ---------------------------------------------------------------------------
def test_one_update_cpu_smoke():
    torch.manual_seed(0)
    model = _tiny_model()
    state = _state()
    buffer = ppo.RolloutBuffer(gamma=0.95, lam=0.95)

    samples = [
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
        [0.4, 0.5, 0.6, 0.7],
        [0.5, 0.6, 0.7, 0.8],
    ]
    for i, values in enumerate(samples):
        episode = 0 if i < 2 else 1
        t = i if i < 2 else i - 2
        transition = _make_transition(
            model,
            state=state,
            sampled_norm=torch.tensor([values], dtype=torch.float32),
            reward=0.1 * (i + 1),
            cost=0.05 * i,
            cost_vector=[0.05 * i, 0.0, 0.0, 0.0, 0.0, 0.0],
            terminated=(i == 3),
            truncated=(i == 1),
            episode_id=episode,
            t=t,
            bootstrap_value=0.2 if i == 1 else 0.0,
            bootstrap_cost_value=0.1 if i == 1 else 0.0,
        )
        buffer.add_transition(transition)

    batch = buffer.to_training_batch("cpu", gamma=0.95, lam=0.95)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    record = ppo.ppo_update(
        model,
        optimizer,
        batch,
        clip=0.2,
        value_coef=0.5,
        cost_coef=0.1,
        epochs=1,
        device="cpu",
        update_index=0,
    )

    assert record["transitions"] == 4
    assert record["policy_valid_transitions"] == 4
    assert record["masked_transitions"] == 0
    assert record["truncation_bootstrap_count"] == 1
    assert record["logprob_recompute_max_abs_diff"] < 1e-5
    assert abs(record["ratio_at_old_mean"] - 1.0) < 1e-5
    assert record["ratio_at_old_max_abs_dev"] < 1e-5
    assert record["boundary_rate"] == 0.0
    assert 0.0 <= record["clip_frac"] <= 1.0
    assert math.isfinite(record["approx_kl"])
    assert record["approx_kl"] >= -1e-6
    assert math.isfinite(record["value_loss"])
    assert math.isfinite(record["cost_loss"])
    assert all(torch.isfinite(p).all() for p in model.parameters())
