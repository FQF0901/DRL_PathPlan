"""v6 E1 测试：router 监控 top-2 口径修复 + 专家输出范数探针（死键修复）+ no-op 证据。

- ``RouterMonitor``：门控权重必须是 **top-2 混合权重**（恰 2 非零、Σ=1），不再是
  ``sigmoid(logits)``；``router_mean_output_norm`` 由 ``_ExpertNormProbe`` 注入；
- 探针默认开、只读：开/关两侧参数逐位一致，且开启侧指标落盘（update() 返回值）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from torch import nn

from net.moe import top_k_softmax
from pipeline.stages import _router_monitor_summary
from pipeline.trainer import PPOConfig, PPOTrainer, RouterMonitor, _SmokeModel

OBS = {"ego": np.full((1, 8), 0.5, dtype=np.float32)}


class _ScriptedPool:
    num_envs = 1
    accepts_pre_step_labels = False

    def reset(self):  # noqa: ANN201
        return [self._record()]

    def step(self, actions, references=None):  # noqa: ANN001
        return [self._record()]

    @staticmethod
    def _record() -> Dict[str, Any]:
        return {
            "obs": OBS,
            "info": {},
            "reward": 0.25,
            "terminated": False,
            "truncated": False,
            "spec_id": 0,
        }


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:
        pass

    def step(self, env_index, info, obs, done, pool_reward, step_index=0):  # noqa: ANN001
        return float(pool_reward), {}


# --------------------------------------------------------------------------- #
# RouterMonitor：top-2 口径
# --------------------------------------------------------------------------- #

def test_router_monitor_uses_top2_gate_weights_not_sigmoid():
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.5, 0.0],  # top-2: e0,e1
            [0.1, 3.0, 2.0, 0.2],  # top-2: e1,e2
            [1.0, 0.9, 2.5, 0.3],  # top-2: e2,e0
        ]
    )
    weights = top_k_softmax(logits, 2)
    monitor = RouterMonitor(4)
    monitor.update({"router_logits": logits, "expert_weights": weights})
    summary = monitor.summary()

    assert summary["router_count"] == 3
    assert float(np.sum(summary["router_mean_weight"])) == pytest.approx(1.0, abs=1e-6)
    expected_active = (weights.numpy() > 0.0).mean(axis=0)
    np.testing.assert_allclose(summary["router_active_rate"], expected_active, atol=1e-6)
    # 旧 sigmoid 口径会给出全 1 的 active rate（所有 logits > 0）；top-2 口径为 2/3,2/3,2/3,0
    sigmoid_active = (torch.sigmoid(logits).numpy() > 0.5).mean(axis=0)
    assert not np.allclose(summary["router_active_rate"], sigmoid_active), "监控口径必须不再是 sigmoid"
    assert np.allclose(summary["router_active_rate"], [2 / 3, 2 / 3, 2 / 3, 0.0], atol=1e-6)
    # 全专家 softmax 的熵/有效专家数（监控面）
    assert 0.0 <= summary["router_mean_entropy"] <= 1.0
    assert 1.0 <= summary["router_effective_n"] <= 4.0


def test_router_monitor_derives_top2_from_logits_without_expert_weights():
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0], [0.1, 3.0, 2.0, 0.2]])
    with_weights = RouterMonitor(4)
    with_weights.update({"router_logits": logits, "expert_weights": top_k_softmax(logits, 2)})
    from_logits = RouterMonitor(4)
    from_logits.update({"router_logits": logits})
    np.testing.assert_allclose(
        with_weights.summary()["router_mean_weight"], from_logits.summary()["router_mean_weight"], atol=1e-6
    )


# --------------------------------------------------------------------------- #
# 专家输出范数探针：死键修复 + no-op
# --------------------------------------------------------------------------- #

class _MoeSmokeModel(_SmokeModel):
    """带 MoE 专家模块的 smoke stub：forward 时真实调用 experts（供 hook 采集）。"""

    def __init__(self, template: Dict[str, np.ndarray], num_experts: int = 8):
        super().__init__(template)
        self.plan_head = SimpleNamespace(
            moe=SimpleNamespace(experts=nn.ModuleList([nn.Linear(self.trunk[0].in_features, 64) for _ in range(num_experts)]))
        )

    def forward(self, obs, *, rollout: bool = True, world_model: bool = True, wm_detach: bool = False):  # noqa: ANN001
        out = super().forward(obs, rollout=rollout, world_model=world_model, wm_detach=wm_detach)
        hidden = out["latent"]
        for expert in self.plan_head.moe.experts:  # 输出被丢弃；hook 只读采集范数
            expert(hidden)
        return out


def _make_trainer(*, probe_on: bool) -> PPOTrainer:
    torch.manual_seed(0)
    model = _MoeSmokeModel(OBS)
    config = PPOConfig(
        epochs=2,
        minibatch_size=4,
        lr=1e-3,
        device="cpu",
        router_expert_norm_probe=probe_on,
        group_probe_every=0,
        anchor_grad_probe=False,
    )
    trainer = PPOTrainer(
        model,
        _ScriptedPool(),
        config,
        reward_adapter=_RewardStub(),
        probe_batch=None,
        logger=lambda _line: None,
    )
    trainer.adopt_obs([OBS])
    return trainer


def test_expert_norm_probe_populates_dead_key_and_is_bitwise_noop():
    trainer_on = _make_trainer(probe_on=True)
    trainer_off = _make_trainer(probe_on=False)
    history_on: List[Dict[str, Any]] = []
    history_off: List[Dict[str, Any]] = []
    for _ in range(2):
        trainer_on.collect_rollout(4)
        history_on.append(trainer_on.update())
        trainer_off.collect_rollout(4)
        history_off.append(trainer_off.update())

    # no-op 证据：参数逐位一致
    for name, parameter in trainer_on.model.named_parameters():
        assert torch.equal(parameter, trainer_off.model.get_parameter(name)), f"探针改变了参数 {name}"
    # 核心指标一致
    for metrics_on, metrics_off in zip(history_on, history_off):
        for key in ("total_loss", "policy_loss", "value_loss", "approx_kl", "grad_norm", "batches"):
            assert metrics_on[key] == metrics_off[key], f"{key} 不应受探针影响"

    # E1 指标落盘：router 负载 = top-2 口径；输出范数 = 探针注入（开启侧有值、关闭侧 None）
    for metrics in history_on:
        assert float(np.sum(metrics["router_mean_weight"])) == pytest.approx(1.0, abs=1e-6)
        norms = metrics["router_mean_output_norm"]
        assert norms is not None and len(norms) == 8 and all(value > 0.0 for value in norms)
    for metrics in history_off:
        assert metrics["router_mean_output_norm"] is None
    # 探针 hook 在 update 结束后必须摘除（不污染 collect/后续 forward）
    assert len(trainer_on.model.plan_head.moe.experts[0]._forward_hooks) == 0


# --------------------------------------------------------------------------- #
# E1 指标落盘：metrics.json::router_monitor 汇总
# --------------------------------------------------------------------------- #

def test_router_monitor_summary_persists_to_metrics_payload():
    history = [
        {
            "router_mean_weight": [0.5, 0.5, 0.0, 0.0],
            "router_active_rate": [0.5, 0.5, 0.0, 0.0],
            "router_mean_output_norm": [1.0, 2.0, 3.0, 4.0],
            "router_effective_n": 2.0,
            "router_mean_entropy": 0.5,
            "router_load_imbalance": 1.0,
        },
        {
            "router_mean_weight": [0.25, 0.75, 0.0, 0.0],
            "router_active_rate": [0.25, 0.75, 0.0, 0.0],
            "router_mean_output_norm": [3.0, 4.0, 5.0, 6.0],
            "router_effective_n": 3.0,
            "router_mean_entropy": 0.7,
            "router_load_imbalance": 2.0,
        },
    ]
    summary = _router_monitor_summary(history)
    assert summary["updates"] == 2
    assert summary["effective_n_mean"] == pytest.approx(2.5)
    assert summary["mean_entropy_mean"] == pytest.approx(0.6)
    assert summary["load_imbalance_mean"] == pytest.approx(1.5)
    assert summary["mean_weight_mean"] == pytest.approx([0.375, 0.625, 0.0, 0.0])  # 逐 expert 跨 update 均值
    np.testing.assert_allclose(summary["output_norm_mean"], [2.0, 3.0, 4.0, 5.0])
    assert summary["output_norm_updates"] == 2
    # 无数据（探针关/无 MoE）→ None，不抛异常
    empty = _router_monitor_summary([{"total_loss": 1.0}])
    assert empty["updates"] == 1
    assert empty["output_norm_mean"] is None and empty["effective_n_mean"] is None
