"""v4 ③：分参数组梯度/更新范数探针（``grad_group/<group>/{pre_clip,update}``，纯监控）。

锁定：

- 组划分：``policy.*`` / ``value.*``（含 ``head_value`` 兜底）/ ``plan_head.moe.experts.*`` /
  其余 = ``other``；
- 每 update 记录 pre-clip 梯度范数 + 实际更新范数（step 前后参数差），默认开（every=1）；
- **纯监控/no-op**：探针开（默认）与 ``group_probe_every=0`` 的训练参数逐位一致、核心指标一致；
- ``every=N``：只在该 update 产出（u1、u1+N、...）；``0`` = 关闭；
- ``PPOConfig`` 校验与阶段 C 解析（CLI > config ``train.ppo.group_probe_every`` > 1）。

不建 env、不 import metadrive（自建命名模型 + 手填 buffer）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.buffer import RolloutBuffer
from pipeline.stages import _parse_args, _resolve_stage_c_group_probe_every
from pipeline.trainer import (
    GRAD_PROBE_GROUPS,
    PPOConfig,
    PPOTrainer,
    _SmokeModel,
    grad_probe_group,
)

COUNT = 8
BATCH = 4


class _Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mu = torch.nn.Linear(16, 2)
        self.logstd = torch.nn.Parameter(torch.full((2,), -1.0))
        self.register_buffer("action_low", torch.tensor([0.0, -0.6]))
        self.register_buffer("action_high", torch.tensor([10.0, 0.6]))


class _ProbeModel(torch.nn.Module):
    """参数命名覆盖四个探针组，且四个组都进入前向（梯度非零）。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoders = torch.nn.ModuleDict({"ego": torch.nn.Linear(8, 16)})
        self.policy = _Policy()
        self.value = torch.nn.ModuleDict({"net": torch.nn.Linear(16, 1)})
        self.plan_head = torch.nn.ModuleDict(
            {"moe": torch.nn.ModuleDict({"experts": torch.nn.ModuleDict({"0": torch.nn.Linear(16, 16)})})}
        )

    def forward(self, obs, *, rollout=True, world_model=True, wm_detach=False):  # noqa: ANN001
        hidden = torch.tanh(self.encoders["ego"](torch.flatten(obs["ego"], start_dim=1)))
        hidden = hidden + 0.01 * self.plan_head["moe"]["experts"]["0"](hidden)
        span = self.policy.action_high - self.policy.action_low
        mu = self.policy.action_low + span * torch.sigmoid(self.policy.mu(hidden))
        return {
            "action_mu": mu,
            "action_logstd": self.policy.logstd.clamp(-5.0, 0.0).unsqueeze(0).expand(hidden.shape[0], -1),
            "value": self.value["net"](hidden),
            "plan": mu.unsqueeze(1).expand(-1, 6, -1).contiguous(),
        }


class _PoolStub:
    num_envs = 1


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index, info, obs, done, pool_reward, step_index=0):  # noqa: ANN001
        return float(pool_reward), {}


def _fill(trainer: PPOTrainer) -> PPOTrainer:
    buffer = RolloutBuffer(COUNT, channels={"ego": (1, 8)}, history_channels=())
    rng = np.random.default_rng(0)
    for step in range(COUNT):
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
    trainer._valid_mask = np.ones(COUNT, dtype=bool)
    trainer._router_labels = np.zeros((COUNT, 8), dtype=np.float32)
    trainer._has_router_labels = np.zeros(COUNT, dtype=bool)
    return trainer


def _make_trainer(model: torch.nn.Module, *, every: int = 1) -> PPOTrainer:
    return _fill(
        PPOTrainer(
            model,
            _PoolStub(),
            PPOConfig(group_probe_every=every, epochs=1, minibatch_size=BATCH, lr=1e-3, device="cpu"),
            reward_adapter=_RewardStub(),
            probe_batch=None,
            logger=lambda _line: None,
        )
    )


def _snapshot(model: torch.nn.Module) -> dict:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


# --------------------------------------------------------------------------- #
# 组划分 / 配置
# --------------------------------------------------------------------------- #

def test_grad_probe_group_classification() -> None:
    assert grad_probe_group("policy.mu.weight") == "policy"
    assert grad_probe_group("value.net.2.weight") == "value"
    assert grad_probe_group("plan_head.moe.experts.0.0.weight") == "experts"
    assert grad_probe_group("encoders.ego.weight") == "other"
    # 真实模型 value 前缀优先于 other；smoke stub 靠 value_names 兜底
    assert grad_probe_group("head_value.weight", frozenset({"head_value.weight"})) == "value"
    assert grad_probe_group("head_value.weight") == "other"


def test_ppo_config_group_probe_every_validation() -> None:
    assert PPOConfig().group_probe_every == 1
    assert PPOConfig(group_probe_every=0).group_probe_every == 0
    assert PPOConfig(group_probe_every=4).group_probe_every == 4
    with pytest.raises(ValueError):
        PPOConfig(group_probe_every=-1)
    with pytest.raises(ValueError):
        PPOConfig(group_probe_every="abc")


def test_stage_c_group_probe_every_resolution_cli_over_config() -> None:
    args = _parse_args(["--stage", "C"])
    assert args.group_probe_every is None
    assert _resolve_stage_c_group_probe_every(args, {}) == 1, "默认每 update"
    assert _resolve_stage_c_group_probe_every(args, {"ppo": {"group_probe_every": 4}}) == 4
    args_cli = _parse_args(["--stage", "C", "--group-probe-every", "0"])
    assert args_cli.group_probe_every == 0
    assert _resolve_stage_c_group_probe_every(args_cli, {"ppo": {"group_probe_every": 4}}) == 0, "CLI 优先"
    assert _resolve_stage_c_group_probe_every(args, {"ppo": {"group_probe_every": "abc"}}) == 1, "非法回退"


# --------------------------------------------------------------------------- #
# 指标内容
# --------------------------------------------------------------------------- #

def test_update_records_group_norms() -> None:
    torch.manual_seed(0)
    model = _ProbeModel()
    metrics = _make_trainer(model, every=1).update()
    group = metrics["grad_group"]
    assert set(group) == set(GRAD_PROBE_GROUPS), f"四个组必须齐全：{sorted(group)}"
    for name in GRAD_PROBE_GROUPS:
        assert np.isfinite(group[name]["pre_clip"]) and group[name]["pre_clip"] >= 0.0
        assert np.isfinite(group[name]["update"]) and group[name]["update"] >= 0.0
    # 四组都进入前向 → pre-clip 梯度与更新范数均 > 0
    for name in GRAD_PROBE_GROUPS:
        assert group[name]["pre_clip"] > 0.0, f"{name} 组应有梯度"
        assert group[name]["update"] > 0.0, f"{name} 组应被更新"
    assert metrics["grad_group_every"] == 1
    # 与现有总 grad_norm 并存
    assert "grad_norm" in metrics and metrics["grad_norm"] > 0.0


def test_group_probe_is_read_only_and_off_switch_matches() -> None:
    """探针默认开 vs every=0：参数逐位一致、核心指标一致（no-op 证据）。"""
    torch.manual_seed(0)
    model_on = _ProbeModel()
    torch.manual_seed(0)
    model_off = _ProbeModel()
    trainer_on = _make_trainer(model_on, every=1)
    trainer_off = _make_trainer(model_off, every=0)
    metrics_on = trainer_on.update()
    metrics_off = trainer_off.update()
    assert "grad_group" in metrics_on and "grad_group" not in metrics_off
    for name, parameter in model_on.named_parameters():
        assert torch.equal(parameter, model_off.get_parameter(name)), f"探针改变了参数 {name}"
    for key in ("total_loss", "policy_loss", "value_loss", "approx_kl", "grad_norm", "batches"):
        assert metrics_on[key] == pytest.approx(metrics_off[key], rel=0, abs=0), f"{key} 不应受探针影响"


def test_group_probe_every_interval() -> None:
    torch.manual_seed(0)
    trainer = _make_trainer(_ProbeModel(), every=2)
    assert "grad_group" in trainer.update(), "u1（基准）必须产出"
    assert "grad_group" not in trainer.update(), "u2 不在间隔内"
    assert "grad_group" in trainer.update(), "u3 = u1+2 必须产出"


def test_smoke_model_value_group_via_fallback() -> None:
    """smoke stub（``head_value.*``）走 value_names 兜底：value 组有 pre_clip 梯度。"""
    torch.manual_seed(0)
    model = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    metrics = _make_trainer(model, every=1).update()
    assert metrics["grad_group"]["value"]["pre_clip"] > 0.0
    assert metrics["grad_group"]["policy"]["pre_clip"] == 0.0, "stub 无 policy.* → 0（不虚报）"
