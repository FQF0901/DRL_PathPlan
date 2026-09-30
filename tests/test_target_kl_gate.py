"""v4 ②：``target_kl`` 守门接线（``approx_kl > target_kl`` → 跳过该 epoch 剩余 minibatch）。

锁定：

- ``target_kl=None``（默认）：跑满 ``epochs × ceil(count/minibatch)``，``kl_early_stop=0``
  ——no-op/旧行为；
- ``target_kl`` 给值：首个超阈值 minibatch 后立刻停（``batches`` 只含已更新的 minibatch），
  ``kl_early_stop=1``、``kl_early_stop_total`` 跨 update 累计、日志一行；
- 触发判定按**单 minibatch** approx_kl（旧实现的 epoch 末 1.5× 均值启发式已替换）；
- ``PPOConfig`` 校验（非有限/非正 fail-fast）；
- 阶段 C 解析：CLI ``--target-kl`` > config ``train.ppo.target_kl`` > None；非法 fail-fast。

不建 env、不 import metadrive（stub 模型 + 手填 buffer）。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest
import torch

from pipeline.stages import _parse_args, _resolve_stage_c_target_kl
from pipeline.buffer import RolloutBuffer
from pipeline.trainer import PPOConfig, PPOTrainer, _SmokeModel

COUNT = 4
BATCH = 1


class _PoolStub:
    num_envs = 1


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index, info, obs, done, pool_reward, step_index=0):  # noqa: ANN001
        return float(pool_reward), {}


def _filled_trainer(*, target_kl: Optional[float], logprob: float = -20.0) -> PPOTrainer:
    """buffer 的 old_logprob 远离模型输出 → approx_kl 极大（守门必触发）。"""
    torch.manual_seed(0)
    model = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    logs: List[str] = []
    trainer = PPOTrainer(
        model,
        _PoolStub(),
        PPOConfig(
            target_kl=target_kl,
            epochs=4,
            minibatch_size=BATCH,
            lr=1e-3,
            device="cpu",
        ),
        reward_adapter=_RewardStub(),
        probe_batch=None,
        logger=logs.append,
    )
    trainer._logs = logs  # type: ignore[attr-defined]
    buffer = RolloutBuffer(COUNT, channels={"ego": (1, 8)}, history_channels=())
    rng = np.random.default_rng(0)
    for step in range(COUNT):
        buffer.add_step(
            {"ego": rng.normal(size=(1, 8)).astype(np.float32)},
            action=np.array([3.0, 0.0], dtype=np.float32),
            logprob=logprob,
            value=float(rng.normal()),
            reward=float(rng.normal()),
            episode=0,
            step=step,
            spec_id=0,
        )
    trainer.buffer = buffer
    trainer._valid_mask = np.ones(COUNT, dtype=bool)
    trainer._router_labels = np.zeros((COUNT, 8), dtype=np.float32)
    trainer._has_router_labels = np.zeros(COUNT, dtype=bool)
    return trainer


def test_default_none_runs_all_batches_legacy() -> None:
    trainer = _filled_trainer(target_kl=None)
    metrics = trainer.update()
    assert metrics["batches"] == 4 * COUNT, "None = 不守门：epochs×minibatch 全跑"
    assert metrics["kl_early_stop"] == 0.0
    assert metrics["kl_early_stop_total"] == 0.0
    assert metrics["target_kl"] is None
    assert not any("kl_early_stop" in line for line in trainer._logs)  # type: ignore[attr-defined]


def test_target_kl_gate_skips_remaining_minibatches_and_counts() -> None:
    trainer = _filled_trainer(target_kl=1e-6)
    first = trainer.update()
    assert first["batches"] == 1, "首个 minibatch 超阈值后必须跳过本 epoch 剩余（并终止后续 epoch）"
    assert first["kl_early_stop"] == 1.0
    assert first["kl_early_stop_total"] == 1.0
    assert first["target_kl"] == pytest.approx(1e-6)
    logs = trainer._logs  # type: ignore[attr-defined]
    assert any("kl_early_stop" in line and "target_kl" in line for line in logs), "必须打一行日志"
    # 第二次 update：累计计数递增（同一阈值仍触发）
    second = trainer.update()
    assert second["batches"] == 1
    assert second["kl_early_stop_total"] == 2.0


def test_target_kl_does_not_trigger_when_kl_below_threshold() -> None:
    """old_logprob 与模型输出一致 → approx_kl≈0 < 阈值 → 全跑（守门不误伤）。"""
    trainer = _filled_trainer(target_kl=10.0)
    with torch.no_grad():
        out = trainer.model(
            {"ego": torch.as_tensor(np.asarray(trainer.buffer.obs["ego"][0], dtype=np.float32))}
        )
    from pipeline.trainer import logprob_from_action

    action = torch.as_tensor(trainer.buffer.action[0], dtype=torch.float32)
    with torch.no_grad():
        logprob = float(
            logprob_from_action(
                out["action_mu"], out["action_logstd"], action, trainer.low, trainer.high
            )
        )
    for step in range(COUNT):
        trainer.buffer.logprob[step] = logprob
    metrics = trainer.update()
    assert metrics["batches"] == 4 * COUNT
    assert metrics["kl_early_stop"] == 0.0


def test_ppo_config_target_kl_validation() -> None:
    assert PPOConfig().target_kl is None
    assert PPOConfig(target_kl=0.02).target_kl == 0.02
    with pytest.raises(ValueError):
        PPOConfig(target_kl=float("nan"))
    with pytest.raises(ValueError):
        PPOConfig(target_kl=0.0)
    with pytest.raises(ValueError):
        PPOConfig(target_kl=-1.0)


def test_stage_c_target_kl_resolution_cli_over_config() -> None:
    args = _parse_args(["--stage", "C"])
    assert args.target_kl is None
    assert _resolve_stage_c_target_kl(args, {}) is None, "默认 None（不守门）"
    assert _resolve_stage_c_target_kl(args, {"ppo": {"target_kl": 0.02}}) == 0.02, "config 生效"
    args_cli = _parse_args(["--stage", "C", "--target-kl", "0.05"])
    assert args_cli.target_kl == 0.05
    assert _resolve_stage_c_target_kl(args_cli, {"ppo": {"target_kl": 0.02}}) == 0.05, "CLI 优先"
    with pytest.raises(SystemExit):
        _resolve_stage_c_target_kl(args, {"ppo": {"target_kl": "abc"}})
    with pytest.raises(SystemExit):
        _resolve_stage_c_target_kl(args, {"ppo": {"target_kl": -0.5}})
