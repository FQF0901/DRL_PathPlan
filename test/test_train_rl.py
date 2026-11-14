"""Unit tests for real-env group-relative PPO (arch_lock §12.4).

CPU-only and simulator-free unless ``RUN_PPO_GROUP_SMOKE=1`` explicitly opts
into the one-update spawned-worker smoke test.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from env.obs import _empty_history
from net.build_net import build_net
from pipeline.common.distributions import TruncatedNormal
from pipeline.train import rl

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tiny_model():
    torch.manual_seed(0)
    model = build_net()
    rl.configure_trainable(model)
    model.eval()
    return model


def _state_batch(model):
    hist = _empty_history()
    hist["hist_valid"][:] = 1.0
    hist["od_mask"][:, 0] = 1.0
    hist["od_ids"][:, 0] = 1
    hist["ld_mask"][:, 0, 0] = 1.0
    hist["ld_ids"][:, 0] = 2
    hist["nav_mask"][:] = 1.0
    hist["ego_mask"][:] = 1.0

    batch = {}
    for key in (
        "od", "od_mask", "ld", "ld_mask", "nav", "nav_mask",
        "ego", "ego_mask", "hist_valid", "od_ids", "ld_ids", "pose",
    ):
        dtype = np.int64 if key in ("od_ids", "ld_ids") else np.float32
        batch[key] = torch.from_numpy(np.asarray(hist[key], dtype=dtype)).unsqueeze(0).to("cpu")
    batch["history_mode"] = torch.ones((1,), dtype=torch.float32, device="cpu")
    window, encoded = model.build_history_window(batch)
    return {
        "window": window.to("cpu"),
        "nav": encoded["nav"].detach().cpu(),
        "ego": encoded["ego"].detach().cpu(),
    }


def _transition(model, state, *, reward, t, valid=True, scene_seed=1000, member=0):
    sampled = torch.tensor([[0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        out = model.forward_window(
            state["window"], state["nav"], state["ego"]
        )
        dist = TruncatedNormal(
            out["action_mu"],
            out["action_logstd"].clamp(-5.0, 0.0).exp(),
            low=0.0,
            high=1.0,
        )
        old_logp = dist.log_prob(sampled).sum(-1)[0].detach().cpu()
    return {
        "state": state,
        "sampled_norm": sampled[0],
        "old_logp": old_logp,
        "reward": torch.tensor(float(reward), dtype=torch.float32),
        "done": False,
        "terminal_event": "none",
        "components": {},
        "valid_for_ppo": bool(valid),
        "t": int(t),
        "scene_seed": int(scene_seed),
        "member": int(member),
    }


def _episode(rewards, *, scene_seed=1000, member=0, invalid_last=False):
    transitions = []
    for t, reward in enumerate(rewards):
        transitions.append(
            {
                "reward": torch.tensor(float(reward), dtype=torch.float32),
                "valid_for_ppo": bool(not (invalid_last and t == len(rewards) - 1)),
                "t": t,
                "scene_seed": scene_seed,
                "member": member,
            }
        )
    return {
        "scene_seed": scene_seed,
        "member": member,
        "transitions": transitions,
    }


def _values(advantages):
    return [v for row in advantages for v in row if v is not None]


# ---------------------------------------------------------------------------
# Active-set advantages
# ---------------------------------------------------------------------------
def test_terminal_reward_enters_last_step_advantage():
    episodes = [
        _episode([0.0, 10.0], member=0),
        _episode([0.0, 10.0], member=1),
        _episode([0.0, 20.0], member=2),
    ]
    advantages, stats = rl.compute_active_set_advantages(episodes, gamma=0.99)
    assert advantages[0][1] == pytest.approx(-5.0)
    assert advantages[1][1] == pytest.approx(-5.0)
    assert advantages[2][1] == pytest.approx(10.0)
    assert advantages[0][1] != 0.0
    assert stats["active_steps"] == 2
    assert stats["transitions_with_adv"] == 6


def test_identical_group_advantages_are_zero():
    episodes = [
        _episode([0.0, 0.0, 10.0], member=member) for member in range(4)
    ]
    advantages, _stats = rl.compute_active_set_advantages(episodes, gamma=0.99)
    for row in advantages:
        for value in row:
            assert value is not None
            assert value == pytest.approx(0.0, abs=1e-6)


def test_leave_one_out_mean_is_exact():
    episodes = [
        _episode([1.0], member=0),
        _episode([2.0], member=1),
        _episode([3.0], member=2),
    ]
    advantages, _stats = rl.compute_active_set_advantages(episodes, gamma=0.99)
    assert [row[0] for row in advantages] == pytest.approx([-1.5, 0.0, 1.5])


def test_active_set_shrinks_after_early_termination():
    episodes = [
        _episode([0.0, 10.0], member=0),  # tau = 1
        _episode([0.0, 0.0, 0.0, 10.0], member=1),
        _episode([0.0, 0.0, 0.0, 10.0], member=2),
    ]
    advantages, _stats = rl.compute_active_set_advantages(episodes, gamma=0.99)
    assert advantages[0][1] is not None  # 3 active at t=1
    assert advantages[1][1] is not None
    assert advantages[1][2] is None  # only 2 active at t=2
    assert advantages[2][3] is None


def test_invalid_terminal_excluded_from_ppo_but_included_in_return():
    # Three normal episodes with a +1 terminal reward, one invalid terminal
    # with -20.  The invalid terminal is excluded from the PPO batch, but its
    # return must still be part of the leave-one-out baseline (otherwise the
    # surviving episodes' advantages would be exactly zero).
    episodes = [
        _episode([0.0, 1.0], member=0),
        _episode([0.0, 1.0], member=1),
        _episode([0.0, 1.0], member=2),
        _episode([0.0, -20.0], member=3, invalid_last=True),
    ]
    gamma = 0.99
    advantages, stats = rl.compute_active_set_advantages(episodes, gamma=gamma)
    assert advantages[3][1] is None
    assert advantages[0][1] is not None
    expected = 1.0 - (1.0 + 1.0 - 20.0) / 3.0
    assert advantages[0][1] == pytest.approx(expected)
    # At t=0 the invalid episode contributes gamma * (-20) to the active set.
    g_invalid_t0 = gamma * -20.0
    expected_t0 = gamma * 1.0 - (gamma * 1.0 + gamma * 1.0 + g_invalid_t0) / 3.0
    assert advantages[0][0] == pytest.approx(expected_t0)
    assert abs(advantages[0][1]) > 1.0
    assert stats["transitions_with_adv"] == 7

    samples = rl.flatten_ppo_samples(episodes, advantages)
    assert all(sample["valid_for_ppo"] for sample in samples)
    assert {sample["t"] for sample in samples} == {0, 1}
    assert not any(sample["member"] == 3 and sample["t"] == 1 for sample in samples)


def test_normalize_advantages_global_mean_std_and_floor():
    normalized, stats = rl.normalize_advantages(
        [[1.0, None], [2.0], [3.0]], floor=1e-6
    )
    values = _values(normalized)
    assert stats["applied"] is True
    assert float(np.mean(values)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.std(values)) == pytest.approx(1.0, abs=1e-6)
    assert normalized[0][1] is None

    degenerate, stats2 = rl.normalize_advantages(
        [[2.0], [2.0], [2.0]], floor=1e-6
    )
    assert stats2["applied"] is False
    assert _values(degenerate) == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def _samples(model, count=6):
    state = _state_batch(model)
    samples = []
    for i in range(count):
        transition = _transition(
            model,
            state,
            reward=float(i - count / 2.0),
            t=i,
            member=i,
        )
        transition["_advantage"] = float((-1.0) ** i) * (0.5 + 0.1 * i)
        samples.append(transition)
    return samples


def test_rl_update_audits_ratio_and_updates_planner():
    model = _tiny_model()
    optimizer, _counts = rl.build_optimizer(model, lr=1e-3, tsb_lr=1e-4)
    samples = _samples(model, count=6)
    before = [p.detach().clone() for p in model.planner.plan.parameters()]

    record = rl.ppo_update(
        model,
        optimizer,
        samples,
        clip=0.2,
        epochs=1,
        minibatch=3,
        device="cpu",
        seed=0,
    )

    assert record["logprob_audit_max_abs_diff"] < 1e-4
    assert math.isfinite(record["policy_loss"])
    assert math.isfinite(record["approx_kl"])
    assert 0.5 < record["ratio_mean"] < 1.5
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.planner.plan.parameters())
    )


def test_rl_zero_advantage_has_zero_loss_and_no_update():
    model = _tiny_model()
    optimizer, _counts = rl.build_optimizer(model, lr=1e-3, tsb_lr=1e-4)
    samples = _samples(model, count=4)
    for sample in samples:
        sample["_advantage"] = 0.0
    before = [p.detach().clone() for p in model.planner.plan.parameters()]

    record = rl.ppo_update(
        model,
        optimizer,
        samples,
        clip=0.2,
        epochs=1,
        minibatch=2,
        device="cpu",
        seed=0,
    )

    assert record["policy_loss"] == pytest.approx(0.0, abs=1e-6)
    assert record["grad_norm"] == pytest.approx(0.0, abs=1e-6)
    for old, new in zip(before, model.planner.plan.parameters()):
        assert torch.allclose(old, new, atol=1e-9)


def test_rl_clipped_loss_at_ratio_one_matches_mean_advantage():
    # With the update frozen at the old parameters (lr=0, one batch) the
    # clipped surrogate is exactly A, so loss == -mean(A).  This is the
    # direct check of the PPO loss formula, separate from the ratio audit.
    model = _tiny_model()
    optimizer, _counts = rl.build_optimizer(model, lr=0.0, tsb_lr=0.0)
    samples = _samples(model, count=4)
    advantages = [1.0, 0.0, 0.0, 0.0]
    for sample, advantage in zip(samples, advantages):
        sample["_advantage"] = advantage
    before = [p.detach().clone() for p in model.planner.plan.parameters()]

    record = rl.ppo_update(
        model,
        optimizer,
        samples,
        clip=0.2,
        epochs=1,
        minibatch=4,
        device="cpu",
        seed=0,
    )

    assert record["ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    assert record["policy_loss"] == pytest.approx(-0.25, abs=1e-6)
    for old, new in zip(before, model.planner.plan.parameters()):
        assert torch.allclose(old, new, atol=1e-9)


# ---------------------------------------------------------------------------
# Config / split loading
# ---------------------------------------------------------------------------
def test_config_and_rl_split_loader():
    args = rl.parse_train_args(
        rl.build_arg_parser(), [], "config/train_rl.yaml"
    )
    assert args.specs.endswith("data/specs/train_2000.json")
    assert args.scenes_per_update == 8
    assert args.group_size == 8
    assert args.max_steps == 600
    assert args.gamma == pytest.approx(0.99)
    assert args.workers == 4
    assert args.device == "cuda"

    from env.scenario.spec import load_specs
    from pipeline.dataio.splits import load_rl_spec_splits

    specs = load_specs(args.specs)
    split = load_rl_spec_splits(
        specs,
        train_manifest=args.train_manifest,
        holdout_manifest=args.holdout_manifest,
    )
    assert len(split.train) == 1500
    assert len(split.holdout) == 500
    train_seeds = {int(spec.seed) for spec in split.train}
    holdout_seeds = {int(spec.seed) for spec in split.holdout}
    assert train_seeds.isdisjoint(holdout_seeds)
    assert train_seeds | holdout_seeds == {int(spec.seed) for spec in specs}

    rng = np.random.default_rng(0)
    scenes = rl.sample_scenes(
        split.train, 8, weights=split.difficulty_weights, rng=rng
    )
    assert len(scenes) == 8
    assert len({int(spec.seed) for spec in scenes}) == 8


# ---------------------------------------------------------------------------
# Opt-in real-env smoke
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("RUN_PPO_GROUP_SMOKE") != "1",
    reason="set RUN_PPO_GROUP_SMOKE=1 to run the spawned-worker real-env smoke",
)
def test_train_rl_real_env_smoke(tmp_path):
    out = tmp_path / "train_rl_smoke"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    gl = ROOT / ".venv" / "gl" / "usr" / "lib" / "x86_64-linux-gnu"
    if gl.is_dir():
        env["LD_LIBRARY_PATH"] = str(gl) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
        )
    command = [
        sys.executable,
        "-m",
        "pipeline.train.rl",
        "--config",
        "config/train_rl.yaml",
        "--out",
        str(out),
        "--device",
        "cpu",
        "--workers",
        "1",
        "--updates",
        "1",
        "--scenes-per-update",
        "1",
        "--group-size",
        "3",
        "--max-steps",
        "5",
        "--eval-limit",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-4000:]
    assert (out / "last.pt").is_file()
    assert (out / "latest.pt").exists()
    assert (out / "train_log.jsonl").is_file()
    lines = (out / "train_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert lines
