"""critic warmup 回归：前 N 个 update 只拟合 value 头，策略/主干逐位不变；之后恢复 PPO。

锁定 ``PPOConfig.critic_warmup_updates``（trainer.update）语义：
- warmup update 指标 ``critic_warmup=True`` 且**不含** ``policy_loss``；
- 共享主干/policy 参数在 warmup 前后 ``torch.equal``（value 梯度不得回流共享主干）；
- value 头参数确实被更新（critic 在拟合）；
- 第 N+1 个 update ``critic_warmup=False``，恢复 ``policy_loss`` 与策略更新；
- ``critic_warmup_updates=0`` = 旧版常规 PPO（首轮即有 policy_loss）。
"""

from __future__ import annotations

import numpy as np
import torch

from pipeline.buffer import RolloutBuffer
from pipeline.trainer import PPOConfig, PPOTrainer, _SmokeModel

COUNT = 8
BATCH = 4


class _PoolStub:
    """update() 不触池，仅需 ``num_envs`` 供 trainer 初始化。"""

    num_envs = 1


def _fill_trainer(mathmodel: torch.nn.Module, warmup: int) -> PPOTrainer:
    model = mathmodel
    trainer = PPOTrainer(
        model,
        _PoolStub(),
        PPOConfig(
            critic_warmup_updates=warmup,
            epochs=2,
            minibatch_size=BATCH,
            lr=1e-3,
            device="cpu",
            normalize_advantage=True,
        ),
        probe_batch=None,
    )
    buffer = RolloutBuffer(
        COUNT,
        channels={"ego": (1, 8)},
        history_frames=6,
        history_interval=1,
        history_channels=(),
    )
    rng = np.random.default_rng(0)
    for step in range(COUNT):
        buffer.add_step(
            {"ego": rng.normal(size=(1, 8)).astype(np.float32), "ego_mask": np.ones((1,), dtype=np.float32)},
            pose=np.zeros(3, dtype=np.float32),
            action=np.array([3.0, 0.0], dtype=np.float32),
            logprob=0.0,
            value=float(rng.normal()),
            reward=float(rng.normal()),
            episode=0,
            step=step,
        )
    trainer.buffer = buffer
    trainer._valid_mask = np.ones(COUNT, dtype=bool)
    trainer._router_labels = np.zeros((COUNT, 8), dtype=np.float32)
    trainer._has_router_labels = np.zeros(COUNT, dtype=bool)
    return trainer


def _model() -> torch.nn.Module:
    return _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})


def _snapshot(model: torch.nn.Module) -> dict:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def test_warmup_updates_value_head_only_then_resumes_ppo() -> None:
    torch.manual_seed(0)
    model = _model()
    value_names = [name for name, _ in model.named_parameters() if "value" in name.lower()]
    policy_names = [name for name, _ in model.named_parameters() if "value" not in name.lower()]
    assert value_names and policy_names  # stub 的 head_value.* 必须可识别

    trainer = _fill_trainer(model, warmup=1)
    before = _snapshot(model)

    first = trainer.update()
    after_first = _snapshot(model)
    assert first["critic_warmup"] is True
    assert "policy_loss" not in first
    assert np.isfinite(first["value_loss"]) and first["value_loss"] > 0.0
    for name in value_names:
        assert not torch.equal(before[name], after_first[name]), f"value 参数 {name} 未更新"
    for name in policy_names:
        assert torch.equal(before[name], after_first[name]), f"warmup 期间策略参数 {name} 被改动"
    assert all(parameter.requires_grad for _, parameter in model.named_parameters()), (
        "warmup 结束后 requires_grad 未恢复"
    )

    second = trainer.update()
    after_second = _snapshot(model)
    assert second["critic_warmup"] is False
    assert "policy_loss" in second and np.isfinite(second["policy_loss"])
    assert any(
        not torch.equal(after_first[name], after_second[name]) for name in policy_names
    ), "warmup 结束后策略参数未恢复更新"


def test_warmup_zero_matches_plain_ppo() -> None:
    torch.manual_seed(0)
    model = _model()
    trainer = _fill_trainer(model, warmup=0)
    metrics = trainer.update()
    assert metrics["critic_warmup"] is False
    assert "policy_loss" in metrics
    assert "entropy" in metrics and "approx_kl" in metrics
