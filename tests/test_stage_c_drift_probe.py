"""G4（v3 预注册 §0 必需项）：执行参考漂移探针 —— 固定探针批上的 plan/mu RMS 位移。

锁定：

- 漂移计算正确：u1 首次记录 = 基准（位移 0），此后已知位移 → ``plan[:,1:6]``（尾部 5 步）
  与 ``action_mu`` 的 RMS、``plan_tail_abs`` 与手算一致；
- interval 触发：u1（基准）+ 每个 ``probe_interval`` 的倍数；非记录点无 ``drift`` 指标；
  ``0`` = 关闭；
- 探针批缺失/加载失败：跳过 + 告警（不阻塞训练）；plan 形状不足 6 步 → 告警一次并停用；
- 纯监控无副作用：drift 前向不改模型参数 / 不写 buffer / 不改变同 seed 的训练数值；
- 解析：CLI ``--probe-interval`` > config ``stages.C.probe_interval`` > 默认 25。

不建 env、不 import metadrive（stub 模型 + 固定观测池）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pytest
import torch

from pipeline.stages import _parse_args, _resolve_stage_c_probe_interval
from pipeline.trainer import DEFAULT_PROBE_INTERVAL, PPOConfig, PPOTrainer, _SmokeModel


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
    """每次 reset/step 都返回同一观测（模型确定性 → 可离线重算期望值）。"""

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


def _obs() -> dict:
    return {"ego": np.full((1, 8), 0.5, dtype=np.float32)}


def _torch_obs() -> Dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value) for key, value in _obs().items()}


class _DriftModel(torch.nn.Module):
    """输出可控的 drift stub：``mu`` 直接可设，``plan[:,1:6] = mu + tail_shift``。"""

    def __init__(self, mu: np.ndarray, *, plan_steps: int = 6) -> None:
        super().__init__()
        self.plan_steps = int(plan_steps)
        self.register_buffer("base_mu", torch.as_tensor(np.asarray(mu, dtype=np.float32)))
        self.register_buffer(
            "tail_shift", torch.zeros((int(mu.shape[0]), max(0, self.plan_steps - 1), 2), dtype=torch.float32)
        )

    def forward(self, obs, *, rollout: bool = True, world_model: bool = True, **kwargs):  # noqa: ANN001
        mu = self.base_mu
        tail = mu[:, None, :] + self.tail_shift
        plan = torch.cat([mu[:, None, :], tail], dim=1)
        return {
            "action_mu": mu,
            "action_logstd": torch.full_like(mu, -1.0),
            "value": mu.sum(dim=-1, keepdim=True),
            "plan": plan,
        }


def _make_trainer(
    model: torch.nn.Module,
    *,
    probe_interval: int = DEFAULT_PROBE_INTERVAL,
    probe_batch: Optional[str] = None,
    logs: Optional[List[str]] = None,
    epochs: int = 1,
) -> PPOTrainer:
    trainer = PPOTrainer(
        model,
        _FixedObsPool(_obs()),
        PPOConfig(seed=0, device="cpu", epochs=epochs, minibatch_size=4),
        reward_adapter=_RewardStub(),
        probe_batch=probe_batch,
        probe_interval=probe_interval,
        logger=(logs.append if logs is not None else print),
    )
    trainer.adopt_obs([_obs()])
    return trainer


def _smoke_model() -> _SmokeModel:
    torch.manual_seed(0)
    return _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})


# --------------------------------------------------------------------------- #
# 漂移计算
# --------------------------------------------------------------------------- #

def test_drift_reference_then_known_displacement() -> None:
    mu1 = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]], dtype=np.float64)
    model = _DriftModel(mu1)
    logs: List[str] = []
    trainer = _make_trainer(model, probe_interval=25, logs=logs)
    trainer._probe_obs = _torch_obs()  # 注入固定探针批（不落盘数据集）

    # u1 = 首次记录 = 基准：位移 0；plan_tail_abs = 尾部 5 步平均绝对值（手算）
    trainer._updates_done = 0
    first = trainer._drift_metrics()["drift"]
    tail1 = np.repeat(mu1[:, None, :], 5, axis=1)
    assert first["available"] == 1.0 and first["update"] == 1.0
    assert first["plan_rms_vs_init"] == 0.0 and first["mu_rms_vs_init"] == 0.0
    assert first["plan_tail_abs"] == pytest.approx(float(np.mean(np.abs(tail1))))
    assert trainer._drift_ref_mu is not None and np.allclose(trainer._drift_ref_mu, mu1)

    # 非记录点（u2、u24）不触发；interval=0 亦不触发（见下）
    trainer._updates_done = 1
    assert trainer._drift_metrics() == {}
    trainer._updates_done = 23
    assert trainer._drift_metrics() == {}

    # 构造已知位移：mu 首行 +0.5，尾部 5 步再整体 +0.25
    model.base_mu.add_(torch.tensor([[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]]))
    model.tail_shift.add_(0.25)
    mu2 = mu1 + np.array([[0.5, 0.0], [0.0, 0.0], [0.0, 0.0]])
    tail2 = np.repeat(mu2[:, None, :], 5, axis=1) + 0.25
    expected_mu_rms = float(np.sqrt(np.mean(np.square(mu2 - mu1))))
    expected_plan_rms = float(np.sqrt(np.mean(np.square(tail2 - tail1))))
    expected_tail_abs = float(np.mean(np.abs(tail2)))

    trainer._updates_done = 24  # u25 = interval 倍数 → 记录点
    second = trainer._drift_metrics()["drift"]
    assert second["update"] == 25.0
    assert second["mu_rms_vs_init"] == pytest.approx(expected_mu_rms, rel=0, abs=1e-12)
    assert second["plan_rms_vs_init"] == pytest.approx(expected_plan_rms, rel=0, abs=1e-12)
    assert second["plan_tail_abs"] == pytest.approx(expected_tail_abs, rel=0, abs=1e-12)
    assert second["n"] == 3.0

    summary = trainer.drift_summary()
    assert summary["available"] == 1.0 and summary["interval"] == 25
    assert summary["updates"] == [1, 25]
    assert summary["mu_rms_vs_init"] == [0.0, pytest.approx(expected_mu_rms)]
    assert summary["plan_rms_vs_init"] == [0.0, pytest.approx(expected_plan_rms)]

    # 0 = 关闭：任何 update 编号都不再触发
    trainer.probe_interval = 0
    trainer._updates_done = 0
    assert trainer._drift_metrics() == {}
    trainer._updates_done = 49
    assert trainer._drift_metrics() == {}


def test_drift_metrics_inside_update_and_interval_trigger() -> None:
    model = _smoke_model()
    trainer = _make_trainer(model, probe_interval=25)
    trainer._probe_obs = _torch_obs()

    trainer.collect_rollout(2)
    first = trainer.update()
    assert first["drift"]["available"] == 1.0 and first["drift"]["update"] == 1.0
    assert first["drift"]["plan_rms_vs_init"] == 0.0 and first["drift"]["mu_rms_vs_init"] == 0.0

    trainer.collect_rollout(2)
    second = trainer.update()
    assert "drift" not in second, "u2 不是 interval 倍数 → 不触发（update_metrics 无副作用字段）"

    # 已知位移后，u25（interval 倍数）触发并与独立前向手算一致
    model.head_mu.bias.data.add_(0.2)
    trainer._updates_done = 24
    trainer.collect_rollout(2)
    third = trainer.update()
    assert third["drift"]["update"] == 25.0
    assert third["drift"]["mu_rms_vs_init"] > 0.0
    # 漂移前向发生在 optimizer step 之后：用同一（post-update）模型的独立前向手算对照
    model.eval()
    with torch.no_grad():
        direct = model(_torch_obs(), rollout=True, world_model=False)
        mu_after = direct["action_mu"].double().numpy()
    expected_mu_rms = float(np.sqrt(np.mean(np.square(mu_after - trainer._drift_ref_mu))))
    assert third["drift"]["mu_rms_vs_init"] == pytest.approx(expected_mu_rms, rel=1e-6)
    # stub 的 plan[:,1:6] = mu 重复 → plan RMS == mu RMS（同一位移广播）
    assert third["drift"]["plan_rms_vs_init"] == pytest.approx(expected_mu_rms, rel=1e-6)


# --------------------------------------------------------------------------- #
# 缺失 / 形状不符：跳过 + 告警
# --------------------------------------------------------------------------- #

def test_drift_probe_missing_batch_warns_and_skips() -> None:
    logs: List[str] = []
    trainer = _make_trainer(_smoke_model(), probe_interval=1, probe_batch=None, logs=logs)
    assert trainer._probe_obs is None
    assert sum("probe_batch=null" in line for line in logs) == 1, "启动打印探针不可用（一次）"

    for _ in range(2):
        trainer.collect_rollout(2)
        metrics = trainer.update()
        assert metrics["drift"]["available"] == 0.0
    assert sum("probe_batch=null" in line for line in logs) == 1, "告警只一次（不逐 update 刷屏）"
    assert trainer.drift_summary()["available"] == 0.0
    assert trainer.drift_summary()["updates"] == []

    # 路径不存在：启动告警一次（含 drift 探针关闭），后续静默跳过
    logs_missing: List[str] = []
    trainer_missing = _make_trainer(
        _smoke_model(), probe_interval=1, probe_batch="/nonexistent/probe_batch_dir", logs=logs_missing
    )
    assert sum("路径不存在" in line and "drift 探针关闭" in line for line in logs_missing) == 1
    trainer_missing.collect_rollout(2)
    metrics = trainer_missing.update()
    assert metrics["drift"]["available"] == 0.0
    assert sum("路径不存在" in line for line in logs_missing) == 1


def test_drift_probe_plan_shape_warns_once() -> None:
    model = _DriftModel(np.zeros((2, 2), dtype=np.float64), plan_steps=4)
    logs: List[str] = []
    trainer = _make_trainer(model, probe_interval=1, logs=logs)
    trainer._probe_obs = _torch_obs()

    assert trainer._drift_metrics()["drift"]["available"] == 0.0
    trainer._updates_done = 1
    assert trainer._drift_metrics()["drift"]["available"] == 0.0
    assert sum("[drift] 警告" in line for line in logs) == 1, "形状不符告警只一次"


# --------------------------------------------------------------------------- #
# 纯监控：不改模型 / 不写 buffer / 不改训练数值
# --------------------------------------------------------------------------- #

def _allclose_arrays(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and np.array_equal(left, right)


def test_drift_probe_has_no_side_effects() -> None:
    # 1) drift 前向本身不改模型参数、不写 buffer（专用 trainer）
    probe_only = _make_trainer(_smoke_model(), probe_interval=1)
    probe_only._probe_obs = _torch_obs()
    probe_only.collect_rollout(2)
    params_before = [parameter.detach().clone() for parameter in probe_only.model.parameters()]
    action_before = probe_only.buffer.action.copy()
    logprob_before = probe_only.buffer.logprob.copy()
    value_before = probe_only.buffer.value.copy()
    assert probe_only._drift_metrics()["drift"]["available"] == 1.0
    for before, after in zip(params_before, probe_only.model.parameters()):
        assert torch.equal(before, after.detach())
    assert _allclose_arrays(action_before, probe_only.buffer.action)
    assert _allclose_arrays(logprob_before, probe_only.buffer.logprob)
    assert _allclose_arrays(value_before, probe_only.buffer.value)

    # 2) 同 seed 同输入：drift on（interval=1）vs off 训练数值与权重逐位一致
    active = _make_trainer(_smoke_model(), probe_interval=1)
    reference = _make_trainer(_smoke_model(), probe_interval=0)
    active._probe_obs = _torch_obs()  # reference 探针批缺失 → drift 关闭
    for _ in range(2):
        active.collect_rollout(2)
        reference.collect_rollout(2)
        metrics_active = active.update()
        metrics_reference = reference.update()
        for key, value in metrics_reference.items():
            if key in ("drift", "update_timing_s"):
                continue
            assert key in metrics_active
            assert metrics_active[key] == value or (
                isinstance(value, dict) and metrics_active[key] == value
            ), f"指标 {key} 受 drift 探针影响"
    for left, right in zip(active.model.parameters(), reference.model.parameters()):
        assert torch.equal(left.detach(), right.detach()), "drift 探针不得改变训练后的权重"


# --------------------------------------------------------------------------- #
# 配置解析
# --------------------------------------------------------------------------- #

def test_resolve_stage_c_probe_interval_cli_over_config() -> None:
    args = _parse_args(["--stage", "C"])
    assert _resolve_stage_c_probe_interval(args, {}) == DEFAULT_PROBE_INTERVAL == 25, "默认 25"
    assert _resolve_stage_c_probe_interval(args, {"probe_interval": 10}) == 10
    assert _resolve_stage_c_probe_interval(args, {"probe_interval": 0}) == 0, "config 0=关"

    cli = _parse_args(["--stage", "C", "--probe-interval", "5"])
    assert _resolve_stage_c_probe_interval(cli, {"probe_interval": 10}) == 5, "CLI 优先"
    off = _parse_args(["--stage", "C", "--probe-interval", "0"])
    assert _resolve_stage_c_probe_interval(off, {"probe_interval": 25}) == 0
    assert _resolve_stage_c_probe_interval(args, {"probe_interval": "bad"}) == 25, "非法值回退默认"
