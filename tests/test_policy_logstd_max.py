"""S2（v3 探索降噪）：策略分布 logstd 上界钳制（采样与 logprob 同一钳制后分布）。

锁定：

- ``clamp_logstd``：``None`` 原样返回（同一对象，位级不变）；给值 → ``min(x, max)`` 且不原地改；
- ``collect_rollout``：``sample_action`` 收到的 logstd == clamp(模型输出)；
- 一致性：buffer 存的 logprob == 用**同一钳制后 logstd** 重算的 ``logprob_from_action``
  （且显著区别于未钳制口径）→ 采样与 update 的 ratio 同分布；
- ``update``：``logprob_from_action`` / ``gaussian_entropy`` 消费同一钳制后 logstd；
- 默认 ``None``：两条路径逐位同旧行为（重算与 buffer 存值一致）；
- ``PPOConfig`` 校验（非有限值 fail-fast）；阶段 C 解析（CLI > config > 默认 null）。

不建 env、不 import metadrive（stub 模型 + capture 池）。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest
import torch

from pipeline import trainer as trainer_mod
from pipeline.buffer import RolloutBuffer
from pipeline.stages import _parse_args, _resolve_stage_c_policy_logstd_max
from pipeline.trainer import (
    PPOConfig,
    PPOTrainer,
    _SmokeModel,
    clamp_logstd,
    logprob_from_action,
)


# --------------------------------------------------------------------------- #
# stub 池 / 奖励（不触环境）
# --------------------------------------------------------------------------- #

class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index, info, obs, done, pool_reward, step_index=0):  # noqa: ANN001
        return float(pool_reward), {}


class _FixedObsPool:
    """每次 reset/step 都返回同一观测（模型确定性 → 可离线重算 logprob）。"""

    num_envs = 1
    accepts_pre_step_labels = False

    def __init__(self, obs: dict) -> None:
        self.obs = obs

    def reset(self):
        return [self._record()]

    def step(self, actions, references=None):  # noqa: ANN001
        return [self._record()]

    def _record(self) -> dict:
        return {
            "obs": self.obs,
            "info": {},
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "spec_id": 0,
        }


def _make_model(logstd: float = -0.5) -> _SmokeModel:
    """stub 模型，logstd=-0.5（高于默认钳制测试值；PolicyHead 自身只 clamp 到 [-5,0]）。"""

    torch.manual_seed(0)
    model = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    with torch.no_grad():
        model.head_logstd.fill_(float(logstd))
    return model


def _obs() -> dict:
    return {"ego": np.full((1, 8), 0.5, dtype=np.float32)}


def _make_trainer(model, *, logstd_max: Optional[float], horizon: int = 2) -> PPOTrainer:
    trainer = PPOTrainer(
        model,
        _FixedObsPool(_obs()),
        PPOConfig(policy_logstd_max=logstd_max, seed=0, device="cpu", epochs=1, minibatch_size=4),
        reward_adapter=_RewardStub(),
        probe_batch=None,
    )
    trainer.adopt_obs([_obs()])
    trainer.collect_rollout(horizon)
    return trainer


def _expected_logprob(trainer: PPOTrainer, model, action: torch.Tensor, logstd_max: Optional[float]) -> float:
    with torch.no_grad():
        out = model({"ego": torch.as_tensor(_obs()["ego"])})
        logstd = clamp_logstd(out["action_logstd"], logstd_max)
        value = logprob_from_action(
            out["action_mu"], logstd, action, trainer.low, trainer.high, mode=trainer.config.action_mode
        )
    return float(value)


# --------------------------------------------------------------------------- #
# clamp_logstd 本体
# --------------------------------------------------------------------------- #

def test_clamp_logstd_none_identity_and_value() -> None:
    x = torch.tensor([-0.5, -3.0, 0.25], dtype=torch.float32)
    assert clamp_logstd(x, None) is x, "None 必须原样返回（位级不变）"
    got = clamp_logstd(x, -2.0)
    assert torch.equal(got, torch.tensor([-2.0, -3.0, -2.0]))
    assert torch.equal(x, torch.tensor([-0.5, -3.0, 0.25])), "不得原地修改输入"


def test_ppo_config_policy_logstd_max_validation() -> None:
    assert PPOConfig().policy_logstd_max is None
    assert PPOConfig(policy_logstd_max=-2).policy_logstd_max == -2.0
    with pytest.raises(ValueError):
        PPOConfig(policy_logstd_max=float("nan"))
    with pytest.raises(ValueError):
        PPOConfig(policy_logstd_max=float("inf"))


# --------------------------------------------------------------------------- #
# collect：采样消费钳制后 logstd
# --------------------------------------------------------------------------- #

def test_collect_samples_from_clamped_logstd(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: List[torch.Tensor] = []
    original = trainer_mod.sample_action

    def spy(mu, logstd, low, high, **kwargs):  # noqa: ANN001
        captured.append(logstd.detach().clone())
        return original(mu, logstd, low, high, **kwargs)

    monkeypatch.setattr(trainer_mod, "sample_action", spy)
    _make_trainer(_make_model(-0.5), logstd_max=-2.0)
    assert captured, "collect 必须经过 sample_action"
    for logstd in captured:
        assert torch.all(logstd <= -2.0 + 1e-6), "采样 logstd 必须已钳制"


def test_sample_and_logprob_use_same_clamped_distribution() -> None:
    """一致性（on-policy）：buffer 的 logprob == 用同一钳制后 logstd 重算的 logprob。"""

    model = _make_model(-0.5)
    trainer = _make_trainer(model, logstd_max=-2.0)
    assert trainer.buffer is not None
    action = torch.as_tensor(trainer.buffer.action[0], dtype=torch.float32)
    stored = float(trainer.buffer.logprob[0])
    clamped = _expected_logprob(trainer, model, action, -2.0)
    unclamped = _expected_logprob(trainer, model, action, None)
    assert abs(stored - clamped) < 1e-3, "采样与 logprob 必须同一钳制后分布"
    assert abs(unclamped - clamped) > 0.5, "构造的 logstd(-0.5) 与钳制值(-2.0) 应显著不同（测试有效性）"


def test_default_none_keeps_legacy_distribution() -> None:
    """默认 None：buffer 的 logprob == 未钳制重算值（旧行为不变）。"""

    model = _make_model(-0.5)
    trainer = _make_trainer(model, logstd_max=None)
    assert trainer.buffer is not None
    action = torch.as_tensor(trainer.buffer.action[0], dtype=torch.float32)
    stored = float(trainer.buffer.logprob[0])
    assert abs(stored - _expected_logprob(trainer, model, action, None)) < 1e-3


# --------------------------------------------------------------------------- #
# update：logprob / entropy 消费同一钳制后 logstd
# --------------------------------------------------------------------------- #

def _filled_trainer(model, *, logstd_max: Optional[float]) -> PPOTrainer:
    trainer = PPOTrainer(
        model,
        _FixedObsPool(_obs()),
        PPOConfig(policy_logstd_max=logstd_max, seed=0, device="cpu", epochs=1, minibatch_size=4, lr=0.0),
        reward_adapter=_RewardStub(),
        probe_batch=None,
    )
    rng = np.random.default_rng(0)
    buffer = RolloutBuffer(4, channels={"ego": (1, 8)}, history_channels=())
    for step in range(4):
        buffer.add_step(
            {"ego": rng.normal(size=(1, 8)).astype(np.float32)},
            action=np.array([3.0, 0.0], dtype=np.float32),
            logprob=0.0,
            value=float(rng.normal()),
            reward=float(rng.normal()),
            episode=0,
            step=step,
            spec_id=0,
        )
    trainer.buffer = buffer
    trainer._valid_mask = np.ones(4, dtype=bool)
    trainer._router_labels = np.zeros((4, 8), dtype=np.float32)
    trainer._has_router_labels = np.zeros(4, dtype=bool)
    return trainer


def test_update_consumes_clamped_logstd(monkeypatch: pytest.MonkeyPatch) -> None:
    lp_logstds: List[torch.Tensor] = []
    entropy_logstds: List[torch.Tensor] = []
    original_lp = trainer_mod.logprob_from_action
    original_entropy = trainer_mod.gaussian_entropy

    def spy_lp(mu, logstd, action, low, high, **kwargs):  # noqa: ANN001
        lp_logstds.append(logstd.detach().clone())
        return original_lp(mu, logstd, action, low, high, **kwargs)

    def spy_entropy(logstd):  # noqa: ANN001
        entropy_logstds.append(logstd.detach().clone())
        return original_entropy(logstd)

    monkeypatch.setattr(trainer_mod, "logprob_from_action", spy_lp)
    monkeypatch.setattr(trainer_mod, "gaussian_entropy", spy_entropy)

    metrics = _filled_trainer(_make_model(-0.5), logstd_max=-2.0).update()
    assert metrics["batches"] == 1
    assert lp_logstds and entropy_logstds
    for logstd in lp_logstds + entropy_logstds:
        assert torch.all(logstd <= -2.0 + 1e-6), "update 侧 logprob/entropy 必须同一钳制后 logstd"


def test_stage_c_policy_logstd_max_resolution_cli_over_config() -> None:
    args = _parse_args(["--stage", "C"])
    assert args.policy_logstd_max is None
    assert _resolve_stage_c_policy_logstd_max(args, {}) is None, "默认 None（不钳制）"
    assert _resolve_stage_c_policy_logstd_max(args, {"policy_logstd_max": -1.5}) == -1.5, "config 生效"
    args_cli = _parse_args(["--stage", "C", "--policy-logstd-max", "-2.0"])
    assert args_cli.policy_logstd_max == -2.0
    assert _resolve_stage_c_policy_logstd_max(args_cli, {"policy_logstd_max": -2.5}) == -2.0, "CLI 优先"
    with pytest.raises(SystemExit):
        _resolve_stage_c_policy_logstd_max(args, {"policy_logstd_max": "abc"})
    with pytest.raises(SystemExit):
        _resolve_stage_c_policy_logstd_max(args, {"policy_logstd_max": "nan"})
