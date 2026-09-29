"""P0-2 回归：PPO 采集的 router 标签与 buffer 的 ``obs_current`` 同帧（步前取）。

一步翻转 stub：``labels_now()`` / ``record["labels_at_step_start"]`` = 步前值，
``record["router_labels"]`` = 步后值。优先级契约：
worker 步开始快照（Vector）> 池级 ``labels_now()``（Local）> record/info 步后标签（旧接口回退）。
"""

from __future__ import annotations

import numpy as np
import torch

from pipeline.trainer import PPOConfig, PPOTrainer

HORIZON = 3


class _AlignModel(torch.nn.Module):
    """最小 PPO 模型（ego-only obs；仅 collect 用，不跑 update）。"""

    def __init__(self) -> None:
        super().__init__()
        self.head_mu = torch.nn.Linear(8, 2)
        self.head_logstd = torch.nn.Parameter(torch.full((2,), -1.0))
        self.head_value = torch.nn.Linear(8, 1)

    def forward(  # noqa: D102 - stub 契约
        self, obs: dict, *, rollout: bool = True, world_model: bool = True, wm_detach: bool = False
    ) -> dict:
        x = obs["ego"].flatten(start_dim=1)
        mu = torch.sigmoid(self.head_mu(x))
        return {
            "action_mu": mu,
            "action_logstd": self.head_logstd.clamp(-5.0, 0.0).unsqueeze(0).expand(x.shape[0], -1),
            "value": self.head_value(x),
            "plan": mu.unsqueeze(1).expand(-1, 6, -1).contiguous(),
        }


class _FlipLabelPool:
    """步前/步后标签值不同的 stub 池（不触环境）。"""

    num_envs = 1

    PRE = 1.0
    POST = 2.0
    START = 3.0

    def __init__(self, *, expose_now: bool = True, expose_step_start: bool = False) -> None:
        self.expose_now = bool(expose_now)
        self.expose_step_start = bool(expose_step_start)

    def labels_now(self):
        if not self.expose_now:
            return None
        return np.full((8,), self.PRE, dtype=np.float32)

    def _record(self) -> dict:
        record = {
            "obs": {"ego": np.zeros((1, 8), dtype=np.float32)},
            "info": {},
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
        if self.expose_step_start:
            record["labels_at_step_start"] = np.full((8,), self.START, dtype=np.float32)
        record["router_labels"] = np.full((8,), self.POST, dtype=np.float32)
        return record

    def reset(self):
        return [self._record()]

    def step(self, actions: np.ndarray, references=None):
        return [self._record()]


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index: int, info: dict, obs: dict, done: bool, pool_reward: float, step_index: int = 0):
        return float(pool_reward), {}


def _collect(pool: _FlipLabelPool) -> PPOTrainer:
    torch.manual_seed(0)
    trainer = PPOTrainer(
        _AlignModel(),
        pool,
        PPOConfig(seed=0, device="cpu", epochs=1, minibatch_size=8),
        reward_adapter=_RewardStub(),
        probe_batch=None,
    )
    trainer.adopt_obs([{"ego": np.zeros((1, 8), dtype=np.float32)}])
    trainer.collect_rollout(HORIZON)
    return trainer


def _recorded_labels(trainer: PPOTrainer) -> np.ndarray:
    assert trainer._router_labels is not None and trainer._has_router_labels is not None
    valid = np.where(trainer._valid_mask)[0]
    assert valid.shape[0] == HORIZON
    assert trainer._has_router_labels[valid].all()
    return trainer._router_labels[valid]


def test_pre_step_labels_are_recorded_with_obs_current() -> None:
    labels = _recorded_labels(_collect(_FlipLabelPool(expose_now=True)))
    assert np.allclose(labels, _FlipLabelPool.PRE), "必须用工步前标签（与 obs_current 同帧）"


def test_step_start_snapshot_takes_precedence() -> None:
    pool = _FlipLabelPool(expose_now=True, expose_step_start=True)
    labels = _recorded_labels(_collect(pool))
    assert np.allclose(labels, _FlipLabelPool.START), "worker 步开始快照必须优先于池级 labels_now()"


def test_legacy_post_step_labels_still_supported() -> None:
    """无任何步前快照时的旧接口回退（步后标签）——兼容路径不被破坏。"""
    pool = _FlipLabelPool(expose_now=False, expose_step_start=False)
    labels = _recorded_labels(_collect(pool))
    assert np.allclose(labels, _FlipLabelPool.POST)
