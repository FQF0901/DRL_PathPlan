"""P0-1 A-hold 回归：阶段 C 收集侧跟踪器参考只依赖被 PPO 记账的随机变量 ``a_t``。

固定契约（trainer.collect_rollout + ``PPOConfig.plan_reference``）：

- ``repeat_action``（默认）：``references == repeat(a_t, 6)``——执行侧只依赖 buffer 记账的
  ``(obs_t, a_t, r_t)``，与 on-policy 口径一致；
- 反事实：扰动 plan/WM 想象曲线（模型 ``plan`` 输出后段）**不改变** references；
- ``plan``（仅对照的旧行为）：首步 = a_t，后 5 步 = 规划预览——同一扰动会改变 references
  （证明扰动确实进入了被修复的旧路径）；
- V9（P2 性能）：``repeat_action`` 臂 collect 走 cheap path（``rollout=False``，不消费 plan），
  ``plan`` 对照臂保留 ``rollout=True``；
- 非法 ``plan_reference`` 值 fail-fast。

stub 池只捕获 ``references``（不建 MetaDrive 仿真）。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest
import torch

from pipeline.trainer import PPOConfig, PPOTrainer

HORIZON = 4


class _PlanStubModel(torch.nn.Module):
    """最小 PPO 模型：``action_mu`` 与 ``plan`` 后段解耦（``plan_tail_bias`` 模拟 WM 想象漂移）。"""

    def __init__(self) -> None:
        super().__init__()
        self.head_mu = torch.nn.Linear(8, 2)
        self.head_logstd = torch.nn.Parameter(torch.full((2,), -1.0))
        self.head_value = torch.nn.Linear(8, 1)
        self.plan_tail_bias = 0.0
        #: 逐次 forward 的 rollout 标志（V9：collect 路径选择守卫）
        self.rollout_flags: List[bool] = []

    def forward(  # noqa: D102 - stub 契约见类 docstring
        self, obs: dict, *, rollout: bool = True, world_model: bool = True, wm_detach: bool = False
    ) -> dict:
        self.rollout_flags.append(bool(rollout))
        x = obs["ego"].flatten(start_dim=1)
        mu = torch.sigmoid(self.head_mu(x))
        logstd = self.head_logstd.clamp(-5.0, 0.0).unsqueeze(0).expand(x.shape[0], -1)
        steps = torch.arange(6, dtype=mu.dtype, device=mu.device).reshape(1, 6, 1)
        # 步 0 = action_mu（与 cheap path 一致）；后段可扰动（legacy plan 会消费它）
        plan = mu.unsqueeze(1) + self.plan_tail_bias * steps
        return {"action_mu": mu, "action_logstd": logstd, "value": self.head_value(x), "plan": plan}


class _CapturePool:
    """捕获 ``pool.step`` 收到的 actions/references；不触环境。"""

    num_envs = 1

    def __init__(self) -> None:
        self.actions: List[np.ndarray] = []
        self.references: List[Optional[np.ndarray]] = []

    def reset(self) -> List[dict]:
        return [self._record()]

    def step(self, actions: np.ndarray, references: Optional[np.ndarray] = None) -> List[dict]:
        self.actions.append(np.asarray(actions, dtype=np.float32).copy())
        self.references.append(None if references is None else np.asarray(references, dtype=np.float32).copy())
        return [self._record()]

    @staticmethod
    def _record() -> dict:
        return {
            "obs": {"ego": np.zeros((1, 8), dtype=np.float32)},
            "info": {},
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index: int, info: dict, obs: dict, done: bool, pool_reward: float, step_index: int = 0):
        return float(pool_reward), {}


def _collect(
    plan_tail_bias: float, *, plan_reference: str, horizon: int = HORIZON
) -> tuple[_CapturePool, _PlanStubModel]:
    torch.manual_seed(0)
    model = _PlanStubModel()
    model.plan_tail_bias = float(plan_tail_bias)
    pool = _CapturePool()
    trainer = PPOTrainer(
        model,
        pool,
        PPOConfig(plan_reference=plan_reference, seed=0, device="cpu", epochs=1, minibatch_size=8),
        reward_adapter=_RewardStub(),
        probe_batch=None,
    )
    obs = {"ego": np.full((1, 8), 0.5, dtype=np.float32)}
    trainer.adopt_obs([obs])
    trainer.collect_rollout(horizon)
    return pool, model


def test_collect_path_follows_plan_reference() -> None:
    """V9（P2 性能）：repeat_action 臂 collect 走 cheap path；plan 对照臂保留 rollout 前向。"""
    _, cheap_model = _collect(0.0, plan_reference="repeat_action")
    assert cheap_model.rollout_flags[:HORIZON] == [False] * HORIZON, "repeat_action 臂必须 rollout=False"
    _, plan_model = _collect(0.0, plan_reference="plan")
    assert plan_model.rollout_flags[:HORIZON] == [True] * HORIZON, "plan 对照臂必须保留 rollout=True"


def test_references_are_repeated_actions() -> None:
    pool, _ = _collect(0.0, plan_reference="repeat_action")
    assert len(pool.references) == HORIZON
    for actions, references in zip(pool.actions, pool.references):
        assert references is not None and references.shape == (1, 6, 2)
        expected = np.repeat(actions[:, None, :], 6, axis=1)
        assert np.allclose(references, expected, atol=0.0), "references 必须逐位等于 repeat(a_t)"
        assert np.allclose(references[:, 0, :], actions, atol=0.0), "首步必须 = 采样动作"


def test_counterfactual_plan_tail_does_not_change_references() -> None:
    """扰动 plan 后段（WM 想象曲线）→ repeat_action 口径 references 不变。"""
    pool_a, _ = _collect(0.0, plan_reference="repeat_action")
    pool_b, _ = _collect(0.9, plan_reference="repeat_action")
    assert len(pool_a.actions) == len(pool_b.actions) == HORIZON
    for index in range(HORIZON):
        assert np.allclose(pool_a.actions[index], pool_b.actions[index]), "同种子动作应一致"
        assert np.allclose(pool_a.references[index], pool_b.references[index]), (
            f"step {index}: plan 后段扰动泄漏进 references"
        )


def test_plan_switch_is_sensitive_to_plan_tail() -> None:
    """legacy 对照开关：扰动确实作用在 plan 输出上（references 随 plan 后段变化）。"""
    pool_a, _ = _collect(0.0, plan_reference="plan")
    pool_b, _ = _collect(0.9, plan_reference="plan")
    for index in range(HORIZON):
        assert np.allclose(pool_a.actions[index], pool_b.actions[index])
        assert not np.allclose(pool_a.references[index], pool_b.references[index]), (
            "plan 对照口径下 references 应随 plan 后段变化（反事实敏感性自检）"
        )
        # 首步仍被采样动作覆盖（新旧口径共同约束）
        assert np.allclose(pool_a.references[index][:, 0, :], pool_a.actions[index], atol=0.0)


def test_plan_reference_validated() -> None:
    with pytest.raises(ValueError):
        PPOConfig(plan_reference="nope")
