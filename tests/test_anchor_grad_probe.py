"""v4 ④：KL 锚梯度探针（``anchor_grad_norm/*``，只读、不进优化器）。

用于判定"锚是否真的作用于参数"（解 v3 A2/A3 异常）：探针**与主损失同口径**重算锚 loss
（``update`` 内锚的 mu 路径在 ``no_grad`` 内反解），因此：

- ``anchor_grad_norm/mu == 0`` = 实现上锚对 mu 无梯度（detach 事实的直接证据）；
- ``anchor_grad_norm/logstd > 0`` 且 ``policy > 0`` = 锚经 logstd 路径作用于参数；
- ``value/experts/other == 0`` = 锚不触及这些参数。

另锁定：``kl_anchor_coef=0`` 不产出；``--no-anchor-grad-probe``/``anchor_grad_probe=False``
与默认开逐位一致（no-op）；子批大小写入 ``n``；配置校验与阶段 C 解析。

不建 env、不 import metadrive（自建命名模型 + 手填 buffer）。
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pytest
import torch

from pipeline.buffer import RolloutBuffer
from pipeline.stages import (
    _parse_args,
    _resolve_stage_c_anchor_grad_probe,
    _resolve_stage_c_group_probe_every,
)
from pipeline.trainer import PPOConfig, PPOTrainer

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
        }


class _PoolStub:
    num_envs = 1


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index, info, obs, done, pool_reward, step_index=0):  # noqa: ANN001
        return float(pool_reward), {}


def _perturbed_ref(model: torch.nn.Module, *, logstd_delta: float = 0.5) -> torch.nn.Module:
    """参考模型 = 深拷贝 + policy.mu 扰动（KL > 0）+ logstd 扰动。

    ``logstd_delta != 0`` 时锚对 logstd 的梯度 ∝ ``σ1²/σ2² − 1`` 非零；``=0`` 时该路径也归零
    （见 :func:`test_anchor_probe_exposes_zero_gradient_when_sigma_matches`）。
    """
    ref = copy.deepcopy(model)
    with torch.no_grad():
        ref.policy.mu.weight.add_(0.2)
        if logstd_delta:
            ref.policy.logstd.add_(float(logstd_delta))
    return ref


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


def _make_trainer(
    model: torch.nn.Module,
    *,
    coef: float = 0.05,
    probe: bool = True,
    probe_size: int = 128,
    ref: Optional[torch.nn.Module] = None,
    logstd_delta: float = 0.5,
) -> PPOTrainer:
    return _fill(
        PPOTrainer(
            model,
            _PoolStub(),
            PPOConfig(
                kl_anchor_coef=coef,
                anchor_grad_probe=probe,
                anchor_grad_probe_size=probe_size,
                epochs=1,
                minibatch_size=BATCH,
                lr=1e-3,
                device="cpu",
            ),
            ref_model=_perturbed_ref(model, logstd_delta=logstd_delta) if ref is None else ref,
            reward_adapter=_RewardStub(),
            probe_batch=None,
            logger=lambda _line: None,
        )
    )


# --------------------------------------------------------------------------- #
# 指标内容（A2/A3 诊断口径）
# --------------------------------------------------------------------------- #

def test_anchor_grad_probe_reports_detached_mu_and_live_logstd() -> None:
    torch.manual_seed(0)
    model = _ProbeModel()
    metrics = _make_trainer(model).update()
    stats = metrics["anchor_grad_norm"]
    assert stats["available"] == 1.0
    assert stats["coef"] == pytest.approx(0.05)
    assert stats["n"] == 8.0, "n = min(probe_size, 有效行数)"
    # 实现事实：锚的 mu 路径 detach（主损失同口径）→ 对 mu 无梯度
    assert stats["mu"] == 0.0, "锚对 mu 的梯度必须为 0（detach 证据）"
    assert stats["logstd"] > 0.0, "锚必须经 logstd 路径产生梯度"
    # 参数分组：policy（logstd 是 policy.*）> 0；锚不触及 value/experts/other
    assert stats["policy"] > 0.0
    assert stats["value"] == 0.0
    assert stats["experts"] == 0.0
    assert stats["other"] == 0.0


def test_anchor_probe_exposes_zero_gradient_when_sigma_matches() -> None:
    """σ1==σ2（仅 mu 漂移）且 mu 路径 detach ⇒ 锚对**所有参数**梯度 = 0。

    KL(N1‖N2) 对 logstd1 的梯度 ∝ σ1²/σ2² − 1（σ 相等时归零），mu 路径又在 no_grad 内反解——
    这正是 v3 A2/A3 "锚存在但无梯度"异常的候选机制；探针必须如实报 0（而非掩盖）。
    """
    torch.manual_seed(0)
    stats = _make_trainer(_ProbeModel(), logstd_delta=0.0).update()["anchor_grad_norm"]
    assert stats["mu"] == 0.0
    assert stats["logstd"] == 0.0
    assert stats["policy"] == 0.0
    assert stats["available"] == 1.0, "探针可用性与梯度是否为零无关"


def test_anchor_probe_skipped_when_coef_zero() -> None:
    torch.manual_seed(0)
    metrics = _make_trainer(_ProbeModel(), coef=0.0).update()
    assert "anchor_grad_norm" not in metrics, "coef=0（锚未启用）不应产出探针"


def test_anchor_probe_size_limits_subset() -> None:
    torch.manual_seed(0)
    metrics = _make_trainer(_ProbeModel(), probe_size=2).update()
    assert metrics["anchor_grad_norm"]["n"] == 2.0


# --------------------------------------------------------------------------- #
# 只读 / no-op
# --------------------------------------------------------------------------- #

def test_anchor_probe_is_read_only() -> None:
    torch.manual_seed(0)
    model_on = _ProbeModel()
    torch.manual_seed(0)
    model_off = _ProbeModel()
    # 参考模型也必须逐位一致（各自扰动相同）
    ref_on = _perturbed_ref(model_on)
    ref_off = _perturbed_ref(model_off)
    metrics_on = _make_trainer(model_on, probe=True, ref=ref_on).update()
    metrics_off = _make_trainer(model_off, probe=False, ref=ref_off).update()
    assert "anchor_grad_norm" in metrics_on and "anchor_grad_norm" not in metrics_off
    for name, parameter in model_on.named_parameters():
        assert torch.equal(parameter, model_off.get_parameter(name)), f"探针改变了参数 {name}"
    for key in ("total_loss", "policy_loss", "value_loss", "kl_anchor", "approx_kl", "grad_norm"):
        assert metrics_on[key] == pytest.approx(metrics_off[key], rel=0, abs=0), f"{key} 不应受探针影响"


# --------------------------------------------------------------------------- #
# 配置 / 阶段 C 解析
# --------------------------------------------------------------------------- #

def test_ppo_config_anchor_probe_validation() -> None:
    assert PPOConfig().anchor_grad_probe is True
    assert PPOConfig().anchor_grad_probe_size == 128
    assert PPOConfig(anchor_grad_probe_size=4).anchor_grad_probe_size == 4
    with pytest.raises(ValueError):
        PPOConfig(anchor_grad_probe_size=0)
    with pytest.raises(ValueError):
        PPOConfig(anchor_grad_probe_size="abc")


def test_stage_c_anchor_probe_resolution_cli_over_config() -> None:
    args = _parse_args(["--stage", "C"])
    assert args.anchor_grad_probe is None
    assert _resolve_stage_c_anchor_grad_probe(args, {}) is True, "默认开"
    assert _resolve_stage_c_anchor_grad_probe(args, {"ppo": {"anchor_grad_probe": False}}) is False
    args_off = _parse_args(["--stage", "C", "--no-anchor-grad-probe"])
    assert args_off.anchor_grad_probe is False
    assert _resolve_stage_c_anchor_grad_probe(args_off, {"ppo": {"anchor_grad_probe": True}}) is False, "CLI 优先"
    # 探针与 ③ 的开关相互独立（anchor 探针只由自身开关/coef 决定）
    assert _resolve_stage_c_group_probe_every(args, {}) == 1
