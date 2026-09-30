"""v4 ④/⑨：KL 锚梯度探针（``anchor_grad_norm/*``，只读、不进优化器）。

用于判定"锚是否真的作用于参数"（解 v3 A2/A3 异常）：探针**与主损失同口径**重算锚 loss
（v4 ⑨ 后：``raw_mu`` 反解在梯度内、锚 KL 用 raw logstd）。因此：

- ``anchor_grad_norm/mu > 0`` = 锚真正经 mu 路径作用于策略参数（v4 ⑨ 修复前恒 0）；
- ``anchor_grad_norm/logstd > 0`` = 锚经 raw logstd 路径作用于参数（S2 钳制不再杀死该路径）；
- ``value == 0`` = 锚不触及价值头；``experts/other`` 反映共享主干（本 toy 模型故意让 experts
  参与 policy 输入 → 二者 > 0，说明锚会经共享主干回传）。

另锁定：``kl_anchor_coef=0`` 不产出且**逐位 no-op**（跨版本参数指纹）；锚 KL 数值与闭式手算
一致且用 raw logstd（非 S2 钳制值）；``--no-anchor-grad-probe``/``anchor_grad_probe=False``
与默认开逐位一致（no-op）；子批大小写入 ``n``；配置校验与阶段 C 解析。

不建 env、不 import metadrive（自建命名模型 + 手填 buffer）。

跨版本指纹（2026-10-01，脚本 ``/tmp/opencode/v4_9_fingerprint_coefpos.py``；同一确定性 CPU
训练 2 update × 4 帧、epochs=2、minibatch=4、lr=1e-3、ref=模型深拷贝）：

- ``coef=0``：参数 sha256 == ``LEGACY_PARAMS_SHA256``（与 v4 ⑧ 记录逐位一致，锚分支未进）；
- ``coef=0.05``：pre-fix ``b92af4133d0507b0d3e996e4198737576f5abfe692c59cc5b782700860f92f19``
  → post-fix ``b2b845227c99f5830a3bac7bbac26f5e1b62f779d7ae844e0f41fc7db4415e28``（mu 路径进
  梯度后锚确实改变参数；u2 起 total_loss/grad_norm/kl_anchor 随之变化）。
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Optional, Tuple

import numpy as np
import pytest
import torch

from pipeline.buffer import RolloutBuffer
from pipeline.stages import (
    _parse_args,
    _resolve_stage_c_anchor_grad_probe,
    _resolve_stage_c_group_probe_every,
)
from pipeline.trainer import PPOConfig, PPOTrainer, _SmokeModel

COUNT = 8
BATCH = 4
#: 跨版本指纹（v4 ⑧ 记录、v4 ⑨ 复核）：确定性 CPU 训练 2 update × 4 帧、coef=0 的参数 sha256。
LEGACY_PARAMS_SHA256 = "3e0e97f3327ba9a1eff2cb4d78036a91bcf78eaf9b884fc1c5568b82858af079"


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
    logstd_max: Optional[float] = None,
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
                policy_logstd_max=logstd_max,
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

def test_anchor_grad_probe_reports_live_mu_and_logstd() -> None:
    torch.manual_seed(0)
    model = _ProbeModel()
    metrics = _make_trainer(model).update()
    stats = metrics["anchor_grad_norm"]
    assert stats["available"] == 1.0
    assert stats["coef"] == pytest.approx(0.05)
    assert stats["n"] == 8.0, "n = min(probe_size, 有效行数)"
    # v4 ⑨：mu 反解在梯度内 + 锚用 raw logstd → 两条输出路径都有梯度
    assert stats["mu"] > 0.0, "锚必须对 mu 输出产生梯度（修复前恒 0）"
    assert stats["logstd"] > 0.0, "锚必须对 raw logstd 输出产生梯度"
    # 参数分组：policy（mu/logstd）> 0；toy 模型 experts/encoders 参与 policy 输入 → 共享主干 > 0
    assert stats["policy"] > 0.0
    assert stats["value"] == 0.0, "锚不触及价值头"
    assert stats["experts"] > 0.0, "toy 模型 experts 参与 policy 输入（共享主干回传）"
    assert stats["other"] > 0.0, "toy 模型 encoders 参与 policy 输入（共享主干回传）"


def test_anchor_probe_uses_raw_logstd_under_s2_clamp() -> None:
    """S2 钳制只服务采样/logprob：锚用 raw logstd ⇒ 钳制下 logstd 梯度仍 > 0。

    修复前锚消费 ``clamp_logstd`` 值：raw=-1.0 被钳到 max=-0.5 后与 ref（-0.5）相等，
    且 ``clamp`` 在越界处导数为 0 ⇒ ``anchor_grad_norm/logstd == 0``（锚被钳制杀死）。
    """
    torch.manual_seed(0)
    stats = _make_trainer(_ProbeModel(), logstd_max=-0.5).update()["anchor_grad_norm"]
    assert stats["mu"] > 0.0
    assert stats["logstd"] > 0.0, "raw logstd 路径不应被 S2 钳制切断"
    assert stats["policy"] > 0.0


def test_anchor_probe_nonzero_when_sigma_matches_but_mu_drifts() -> None:
    """σ1==σ2（仅 mu 漂移）时锚仍有梯度：mu 路径在梯度内（v4 ⑨ 核心修复）。

    修复前 mu 路径 detach + ∂KL/∂logstd ∝ σ1²/σ2²−1 = 0 ⇒ 锚对全部参数梯度 = 0
    （v3 A2/A3 "锚存在但无梯度"的机制）；修复后 mu 路径回传 → mu/policy > 0，
    仅 logstd 输出路径仍精确为 0（σ 相等）。
    """
    torch.manual_seed(0)
    stats = _make_trainer(_ProbeModel(), logstd_delta=0.0).update()["anchor_grad_norm"]
    assert stats["mu"] > 0.0, "σ 相等不再零化锚：mu 路径必须在梯度内"
    assert stats["logstd"] == 0.0, "σ1==σ2 ⇒ ∂KL/∂logstd 精确为 0（与 mu 路径无关）"
    assert stats["policy"] > 0.0
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
# 锚 KL 数值：闭式手算（小例）
# --------------------------------------------------------------------------- #

class _ConstantPolicyModel(torch.nn.Module):
    """mu/logstd 为直接参数（不依赖 obs）：便于按闭式手算锚 KL。"""

    def __init__(
        self,
        raw_mu: Tuple[float, float] = (0.0, 0.5),
        raw_logstd: Tuple[float, float] = (-0.2, -0.4),
    ) -> None:
        super().__init__()
        self.raw_mu = torch.nn.Parameter(torch.tensor(raw_mu, dtype=torch.float32))
        self.raw_logstd = torch.nn.Parameter(torch.tensor(raw_logstd, dtype=torch.float32))
        self.value_bias = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("action_low", torch.tensor([0.0, -0.6]))
        self.register_buffer("action_high", torch.tensor([10.0, 0.6]))

    def forward(self, obs, *, rollout=True, world_model=True, wm_detach=False):  # noqa: ANN001
        batch = int(torch.flatten(obs["ego"], start_dim=1).shape[0])
        span = self.action_high - self.action_low
        mu = self.action_low + span * torch.sigmoid(self.raw_mu)
        return {
            "action_mu": mu.unsqueeze(0).expand(batch, -1),
            "action_logstd": self.raw_logstd.clamp(-5.0, 0.0).unsqueeze(0).expand(batch, -1),
            "value": self.value_bias.expand(batch),
        }


def _hand_kl(mu1, logstd1, mu2, logstd2) -> float:
    """闭式 ``KL(N1‖N2)`` 逐维求和（纯 Python 手算，不依赖 torch）。"""
    return sum(
        (b - a) + (math.exp(2.0 * a) + (x - y) ** 2) / (2.0 * math.exp(2.0 * b)) - 0.5
        for a, b, x, y in zip(logstd1, logstd2, mu1, mu2)
    )


def test_anchor_kl_numeric_matches_hand_computed_raw_logstd() -> None:
    """锚 KL 数值 == 闭式手算（raw 参数），且用 raw logstd 而非 S2 钳制值。"""
    torch.manual_seed(0)
    model = _ConstantPolicyModel()
    ref = copy.deepcopy(model)
    with torch.no_grad():
        ref.raw_mu.copy_(torch.tensor([1.0, 0.5]))
        ref.raw_logstd.copy_(torch.tensor([-0.1, -0.4]))
    trainer = _fill(
        PPOTrainer(
            model,
            _PoolStub(),
            PPOConfig(
                kl_anchor_coef=0.05,
                epochs=1,
                minibatch_size=COUNT,
                lr=0.0,  # 参数不更新 → update 后可按同一模型状态独立复算
                device="cpu",
                policy_logstd_max=-1.0,  # S2 钳制生效：raw(-0.2/-0.4) → -1.0
            ),
            ref_model=ref,
            reward_adapter=_RewardStub(),
            probe_batch=None,
            logger=lambda _line: None,
        )
    )
    metrics = trainer.update()
    expected = 0.05 * _hand_kl(
        mu1=(0.0, 0.5), logstd1=(-0.2, -0.4), mu2=(1.0, 0.5), logstd2=(-0.1, -0.4)
    )
    assert metrics["kl_anchor"] == pytest.approx(expected, rel=1e-4, abs=1e-6)
    # 若锚误用 S2 钳制后的 logstd（-1.0/-1.0）→ 数值显著不同
    clamped = 0.05 * _hand_kl(
        mu1=(0.0, 0.5), logstd1=(-1.0, -1.0), mu2=(1.0, 0.5), logstd2=(-0.1, -0.4)
    )
    assert abs(metrics["kl_anchor"] - clamped) > 1e-2, "锚不得消费 S2 钳制值"


# --------------------------------------------------------------------------- #
# coef=0：逐位 no-op（跨版本指纹）
# --------------------------------------------------------------------------- #

class _ScriptedPool:
    """固定观测/奖励的确定性池（与 tests/test_trainer_noop_probes.py 同构）。"""

    num_envs = 1
    accepts_pre_step_labels = False

    def reset(self):  # noqa: ANN201
        return [self._record()]

    def step(self, actions, references=None):  # noqa: ANN001
        return [self._record()]

    def _record(self):  # noqa: ANN201
        return {
            "obs": {"ego": np.full((1, 8), 0.5, dtype=np.float32)},
            "info": {},
            "reward": 0.25,
            "terminated": False,
            "truncated": False,
            "spec_id": 0,
        }


def _param_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().astype(np.float32).tobytes())
    return digest.hexdigest()


def test_anchor_coef_zero_is_bitwise_noop_with_ref_model() -> None:
    """coef=0 时不进锚分支：带 ref_model 训练与跨版本指纹（v4 ⑧ 记录）逐位一致。"""
    torch.manual_seed(0)
    model = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    ref = copy.deepcopy(model)
    trainer = PPOTrainer(
        model,
        _ScriptedPool(),
        PPOConfig(epochs=2, minibatch_size=4, lr=1e-3, device="cpu", kl_anchor_coef=0.0),
        ref_model=ref,
        reward_adapter=_RewardStub(),
        probe_batch=None,
        logger=lambda _line: None,
    )
    trainer.adopt_obs([{"ego": np.full((1, 8), 0.5, dtype=np.float32)}])
    for _ in range(2):
        trainer.collect_rollout(4)
        metrics = trainer.update()
        assert "anchor_grad_norm" not in metrics, "coef=0（锚未启用）不应产出探针"
    assert _param_hash(model) == LEGACY_PARAMS_SHA256, "coef=0 必须与跨版本指纹逐位一致"


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
