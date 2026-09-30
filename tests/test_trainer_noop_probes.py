"""v4 ⑧：默认开关 no-op 探针（探针/守门默认下训练数值与"旧行为等价路径"逐位一致）。

覆盖：

- 默认 ``PPOConfig``（③ 探针开、④ 锚探针开、② target_kl=None、① value_lr_scale=1.0）
  vs 显式关闭探针（``group_probe_every=0`` + ``anchor_grad_probe=False``）：collect+update
  两轮后**参数逐位一致**、核心指标一致；默认侧含全部新探针键；
- 优化器分组默认 = 旧版（value_lr_scale=1.0 不单独建组）；
- ``target_kl=None`` 跑满 epochs×minibatch（旧行为）；
- 指纹输出：参数 sha256（供跨版本对照）。

跨版本指纹（2026-09-30 实测，脚本 ``/tmp/opencode/v4_fingerprint.py``，同一确定性
CPU 训练 2 update × 4 帧）：

- pre-v4 基线（commit ``71712d8`` 工作树）与 v4 当前树参数 sha256 均 =
  ``3e0e97f3327ba9a1eff2cb4d78036a91bcf78eaf9b884fc1c5568b82858af079``，
  core 指标（total/policy/value loss、approx_kl、grad_norm、batches）逐位一致
  ⇒ 默认开关/探针 no-op；
- ⑥a 指纹：lane=8 vs ld slot0=10、v=10 → 旧 ``speed_limit=10.0/ratio=1.0`` →
  新 ``8.0/1.25``；
- ⑥b 指纹：终局帧（ld_mask 全 0、无 lane 源）→ 旧 ``None/None`` → 新 ``8.0/1.25``。

不建 env、不 import metadrive。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

import numpy as np
import torch

from pipeline.buffer import RolloutBuffer
from pipeline.trainer import PPOConfig, PPOTrainer, _SmokeModel, build_optimizer

CORE_KEYS = (
    "total_loss",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "clipfrac",
    "grad_norm",
    "batches",
    "entropy",
)
OBS = {"ego": np.full((1, 8), 0.5, dtype=np.float32)}


class _ScriptedPool:
    num_envs = 1
    accepts_pre_step_labels = False

    def reset(self):  # noqa: ANN201
        return [self._record()]

    def step(self, actions, references=None):  # noqa: ANN001
        return [self._record()]

    def _record(self) -> Dict[str, Any]:
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

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index, info, obs, done, pool_reward, step_index=0):  # noqa: ANN001
        return float(pool_reward), {}


def _make_trainer(*, probes_on: bool) -> PPOTrainer:
    torch.manual_seed(0)
    model = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    config = (
        PPOConfig(epochs=2, minibatch_size=4, lr=1e-3, device="cpu")
        if probes_on
        else PPOConfig(
            epochs=2,
            minibatch_size=4,
            lr=1e-3,
            device="cpu",
            group_probe_every=0,
            anchor_grad_probe=False,
        )
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


def _param_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().astype(np.float32).tobytes())
    return digest.hexdigest()


def _snapshot(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def test_default_switches_are_bitwise_noop_vs_probes_off() -> None:
    trainer_on = _make_trainer(probes_on=True)
    trainer_off = _make_trainer(probes_on=False)
    history_on: List[Dict[str, Any]] = []
    history_off: List[Dict[str, Any]] = []
    for _ in range(2):
        trainer_on.collect_rollout(4)
        history_on.append(trainer_on.update())
        trainer_off.collect_rollout(4)
        history_off.append(trainer_off.update())
    # 参数逐位一致
    for name, parameter in trainer_on.model.named_parameters():
        assert torch.equal(parameter, trainer_off.model.get_parameter(name)), f"默认探针改变了参数 {name}"
    assert _param_hash(trainer_on.model) == _param_hash(trainer_off.model)
    # 核心指标一致
    for metrics_on, metrics_off in zip(history_on, history_off):
        for key in CORE_KEYS:
            assert metrics_on[key] == metrics_off[key], f"{key} 不应受探针影响"
    # 默认侧含新探针键；关闭侧不含（③/④）
    assert "grad_group" in history_on[0] and "grad_group" not in history_off[0]
    assert "episodes" in history_on[0] and "episodes" in history_off[0], "⑤ 无开关、恒开（纯读）"
    assert history_on[0]["kl_early_stop"] == 0.0 and history_on[0]["target_kl"] is None


def test_default_optimizer_grouping_matches_legacy_signature() -> None:
    torch.manual_seed(0)
    model = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    legacy = build_optimizer(model, lr=1e-3, primary_lr_scale=0.1)
    default = build_optimizer(model, lr=1e-3, primary_lr_scale=0.1, value_lr_scale=1.0)
    assert [
        (float(group["lr"]), [id(parameter) for parameter in group["params"]])
        for group in default.param_groups
    ] == [
        (float(group["lr"]), [id(parameter) for parameter in group["params"]])
        for group in legacy.param_groups
    ], "① 默认必须与旧分组逐位一致"


def test_default_config_target_kl_none_runs_full_epochs() -> None:
    trainer = _make_trainer(probes_on=True)
    trainer.collect_rollout(4)
    metrics = trainer.update()
    assert metrics["batches"] == 2 * 1, "epochs=2 × minibatch=1（4 帧 / minibatch 4）"
    assert metrics["kl_early_stop"] == 0.0 and metrics["kl_early_stop_total"] == 0.0
