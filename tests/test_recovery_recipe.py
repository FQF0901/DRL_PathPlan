"""P1-2 recovery 配方（R1/R2）与混采/时间衰减机制单测（默认关闭；不跑真实训练）。

覆盖：
- R1/R2 冻结清单内容 + optimizer 组（trunk/primary/specific、LR、契约 fail-closed）；
- 探针 helper 四计数（tiny 模型 + 真配置 ``config/default.yaml`` 各一遍）；
- 混采比例控制（0% / 30% / 100%、确定性、放回采样、错误边界、epoch 生成器）；
- 六步链时间衰减（向量校验 + ``weighted_action_chain_loss`` 加权口径 + 默认逐位不变）；
- traj_aux recovery-only 掩码（默认关 = 旧行为）；
- 既有 specific_only/trunk_only/all 配方 optimizer 组逐位不变；
- 端到端 tiny phase3：recovery 冻结不变量 + 混采/衰减开关开启可跑通。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pipeline.stages import (
    FrameWindows,
    _BC_STEP_STRIDE,
    _PHASE3_ANCHOR_HEAD_PREFIXES,
    _PHASE3_SAFE_FREEZE,
    _RECOVERY_EXPERTS_FREEZE,
    _RECOVERY_FREEZE,
    _SPECIFIC_PHASE_FREEZE,
    _bc_channel_keys,
    _phase3_future_fn,
    build_model,
    load_config,
)
from pipeline.trainer import (
    BCConfig,
    BCDataset,
    DEFAULT_PRIMARY_LR_SCALE,
    DEFAULT_RECOVERY_CHAIN_DECAY,
    PHASE3_WM_LOSS_KEYS,
    Phase3Config,
    _PHASE3_SPECIFIC_PREFIXES,
    apply_freeze_prefixes,
    build_phase3_optimizer,
    chain_step_weights,
    effective_traj_aux_mask,
    mix_recovery_batch_indices,
    phase3_effective_config,
    pretrain_bc_phase3,
    recovery_mixed_batches,
    weighted_action_chain_loss,
)
from tests.v2_synthetic import TINY_MODEL_YAML, make_v2_arrays
from tools.controls.probe_freeze_audit import audit_model

_ANCHOR_YAML = TINY_MODEL_YAML + "plan_anchor:\n  enabled: true\n  num_anchors: 6\n"


def _tiny_model():
    from net.model import DrivingModel

    torch.manual_seed(0)
    return DrivingModel(hidden=16, num_experts=8, expert_hidden=16)


def _tiny_anchor_model():
    from net.model import DrivingModel

    torch.manual_seed(0)
    model = DrivingModel(hidden=16, num_experts=8, expert_hidden=16, num_anchors=6)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith(_PHASE3_ANCHOR_HEAD_PREFIXES) or name.startswith("plan_head.moe.experts."):
                parameter.normal_(0.0, 0.05)
        model.policy.mu.weight.normal_(0.0, 0.1)
        model.policy.mu.bias.normal_(0.0, 0.1)
    return model


def _dataset(episodes: int = 4, steps_per_episode: int = 6) -> BCDataset:
    arrays, _ = make_v2_arrays(episodes=episodes, steps_per_episode=steps_per_episode)
    return BCDataset(arrays, {"schema_version": 2, "label_names": None})


def _future_fn_for(dataset: BCDataset):
    arrays = dataset.arrays
    windows = FrameWindows(
        arrays,
        dataset.alignments,
        episode_key="episode_id",
        step_key="step",
        keys=_bc_channel_keys(),
        step_stride=_BC_STEP_STRIDE,
    )
    wm_valid = np.asarray(arrays["wm_valid"], dtype=np.float32) if "wm_valid" in arrays else None
    return _phase3_future_fn(dataset, windows, wm_valid)


def _phase3_cfg(**overrides) -> Phase3Config:
    base = dict(
        epochs=1, batch_size=8, micro_batch_size=4, device="cpu", shuffle=False, seed=0,
        val_indices=None,
    )
    base.update(overrides)
    return Phase3Config(**base)


# --------------------------------------------------------------------------- #
# 冻结清单 + optimizer 组
# --------------------------------------------------------------------------- #

def test_recovery_freeze_lists_content() -> None:
    always_frozen = ("encoders.", "mem_encoder.", "st_gnn.", "plan_head.moe.router.", "policy.", "value.")
    for prefix in always_frozen:
        assert prefix in _RECOVERY_FREEZE
        assert prefix in _RECOVERY_EXPERTS_FREEZE
    for prefix in ("plan_head.moe.experts.", "plan_head.moe.residual_scale"):
        assert prefix in _RECOVERY_FREEZE, "R1 冻结 experts/residual_scale"
        assert prefix not in _RECOVERY_EXPERTS_FREEZE, "R2 解冻 experts/residual_scale"
    for prefix in _PHASE3_ANCHOR_HEAD_PREFIXES:
        assert prefix in _RECOVERY_FREEZE and prefix in _RECOVERY_EXPERTS_FREEZE
    # train 前缀不得在冻结清单
    for prefix in ("plan_head.fusion.", "plan_head.norm.", "plan_head.ego_next.", "plan_head.moe.primary."):
        assert prefix not in _RECOVERY_FREEZE and prefix not in _RECOVERY_EXPERTS_FREEZE


def test_recovery_optimizer_groups_and_lr() -> None:
    model = _tiny_model()
    frozen = apply_freeze_prefixes(model, _RECOVERY_FREEZE)
    assert frozen
    optimizer = build_phase3_optimizer(model, 1e-3, freeze_mode="recovery")
    assert [group["name"] for group in optimizer.param_groups] == ["trunk", "primary"]
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([2.5e-4, 1.0e-4])
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert all(
        name.startswith(("plan_head.fusion.", "plan_head.norm.", "plan_head.ego_next.", "plan_head.moe.primary."))
        for name in trainable
    )
    assert not any(name.startswith(_PHASE3_SPECIFIC_PREFIXES) for name in trainable)

    model2 = _tiny_model()
    apply_freeze_prefixes(model2, _RECOVERY_EXPERTS_FREEZE)
    optimizer2 = build_phase3_optimizer(model2, 1e-3, freeze_mode="recovery_experts")
    assert [group["name"] for group in optimizer2.param_groups] == ["trunk", "primary", "specific"]
    assert [group["lr"] for group in optimizer2.param_groups] == pytest.approx([2.5e-4, 1.0e-4, 5.0e-4])
    specific_names = [
        name
        for name, parameter in model2.named_parameters()
        if parameter.requires_grad and name.startswith(_PHASE3_SPECIFIC_PREFIXES)
    ]
    assert specific_names and not any(name.startswith("plan_head.moe.router.") for name in specific_names)


def test_recovery_optimizer_primary_scale_override() -> None:
    model = _tiny_model()
    apply_freeze_prefixes(model, _RECOVERY_FREEZE)
    optimizer = build_phase3_optimizer(model, 1e-3, primary_scale=0.05, freeze_mode="recovery")
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([2.5e-4, 5.0e-5])


def test_recovery_optimizer_rejects_contract_violations() -> None:
    # R1 但 experts 被解冻 → 拒绝（specific 必须全冻）
    model = _tiny_model()
    apply_freeze_prefixes(model, _RECOVERY_FREEZE)
    for name, parameter in model.named_parameters():
        if name.startswith("plan_head.moe.experts."):
            parameter.requires_grad_(True)
            break
    with pytest.raises(ValueError, match="recovery"):
        build_phase3_optimizer(model, 1e-3, freeze_mode="recovery")
    # R2 但 router 被解冻 → 拒绝
    model2 = _tiny_model()
    apply_freeze_prefixes(model2, _RECOVERY_EXPERTS_FREEZE)
    for name, parameter in model2.named_parameters():
        if name.startswith("plan_head.moe.router."):
            parameter.requires_grad_(True)
            break
    with pytest.raises(ValueError, match="router"):
        build_phase3_optimizer(model2, 1e-3, freeze_mode="recovery_experts")


def test_existing_recipe_optimizers_unchanged() -> None:
    cases = (
        ("specific_only", _SPECIFIC_PHASE_FREEZE, ["specific"]),
        ("trunk_only", _PHASE3_SAFE_FREEZE, ["base"]),
        ("all", (), ["base", "specific"]),
    )
    for mode, prefixes, expected_groups in cases:
        model = _tiny_model()
        apply_freeze_prefixes(model, prefixes)
        optimizer = build_phase3_optimizer(model, 1e-3, freeze_mode=mode)
        assert [group["name"] for group in optimizer.param_groups] == expected_groups
        lrs = [group["lr"] for group in optimizer.param_groups]
        if mode == "all":
            assert lrs == pytest.approx([2.5e-4, 5.0e-4])
        elif mode == "trunk_only":
            assert lrs == pytest.approx([2.5e-4])
        else:
            assert lrs == pytest.approx([5.0e-4])


def test_probe_audit_real_config_recovery_recipes_pass() -> None:
    """验收（helper 路径）：探针 ``audit_model`` 对 R1/R2 四计数不变量 PASS。"""
    config = load_config("config/default.yaml")
    for mode, prefixes in (("recovery", _RECOVERY_FREEZE), ("recovery_experts", _RECOVERY_EXPERTS_FREEZE)):
        model = build_model(config)
        model.train()
        result = audit_model(model, prefixes, freeze_mode=mode, seed=0)
        assert result["status"] == "PASS", (mode, result["failures"])
        groups = result["groups"]
        for group in ("encoder", "mem", "st_gnn", "router", "policy", "value"):
            stat = groups[group]
            assert stat["requires_grad"] == 0 and stat["in_optimizer"] == 0
            assert stat["grad_nonzero"] == 0 and stat["changed"] == 0
        assert groups["fusion"]["changed"] > 0
        assert groups["plan_head.primary"]["changed"] > 0
        if mode == "recovery":
            assert groups["experts"]["requires_grad"] == 0 and groups["experts"]["changed"] == 0
        else:
            assert groups["experts"]["requires_grad"] > 0 and groups["experts"]["changed"] > 0
        assert not result["trainable_not_in_optimizer"] and not result["optimizer_not_trainable"]


# --------------------------------------------------------------------------- #
# 混采（默认关闭）
# --------------------------------------------------------------------------- #

def test_mix_recovery_batch_indices_ratio_boundaries() -> None:
    normal = np.arange(50)
    recovery = np.arange(50, 60)
    batch = mix_recovery_batch_indices(normal, recovery, 10, 0.0, seed=0, step=0)
    assert batch.shape == (10,)
    assert set(batch.tolist()) <= set(normal.tolist())
    batch = mix_recovery_batch_indices(normal, recovery, 10, 0.3, seed=0, step=0)
    assert sum(1 for index in batch if index >= 50) == 3
    assert len(set(batch.tolist())) == 10  # 池足够 → 无放回
    batch = mix_recovery_batch_indices(normal, recovery, 10, 1.0, seed=0, step=0)
    assert set(batch.tolist()) <= set(recovery.tolist())


def test_mix_recovery_batch_indices_deterministic() -> None:
    args = (np.arange(100), np.arange(100, 130), 12, 0.25)
    first = mix_recovery_batch_indices(*args, seed=0, step=0)
    again = mix_recovery_batch_indices(*args, seed=0, step=0)
    other_step = mix_recovery_batch_indices(*args, seed=0, step=1)
    other_seed = mix_recovery_batch_indices(*args, seed=1, step=0)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other_step)
    assert not np.array_equal(first, other_seed)


def test_mix_recovery_batch_indices_replacement_when_pool_small() -> None:
    recovery = np.array([100, 101])  # 与 normal 池（0..49）不相交
    batch = mix_recovery_batch_indices(np.arange(50), recovery, 10, 0.5, seed=0, step=0)
    assert sum(1 for index in batch if index >= 100) == 5  # 放回采样


def test_mix_recovery_batch_indices_errors() -> None:
    with pytest.raises(ValueError):
        mix_recovery_batch_indices(np.arange(10), [], 8, 0.5)  # recovery 池空
    with pytest.raises(ValueError):
        mix_recovery_batch_indices([], np.arange(3), 8, 0.5)  # normal 池空
    with pytest.raises(ValueError):
        mix_recovery_batch_indices(np.arange(10), np.arange(3), 0, 0.5)  # batch<1
    with pytest.raises(ValueError):
        mix_recovery_batch_indices(np.arange(10), np.arange(3), 4, 1.5)  # ratio 越界


def test_recovery_mixed_batches_epoch_generator() -> None:
    args = (np.arange(100), np.arange(100, 120), 10, 0.3)
    batches = list(recovery_mixed_batches(*args, seed=0, epoch=0, n_batches=4))
    assert len(batches) == 4
    for batch in batches:
        assert batch.shape == (10,)
        assert sum(1 for index in batch if index >= 100) == 3
    again = list(recovery_mixed_batches(*args, seed=0, epoch=0, n_batches=4))
    assert all(np.array_equal(a, b) for a, b in zip(batches, again))
    next_epoch = list(recovery_mixed_batches(*args, seed=0, epoch=1, n_batches=4))
    assert any(not np.array_equal(a, b) for a, b in zip(batches, next_epoch))
    with pytest.raises(ValueError):
        list(recovery_mixed_batches([], [], 4, 0.3))


# --------------------------------------------------------------------------- #
# 六步时间衰减（默认关闭）
# --------------------------------------------------------------------------- #

def test_chain_step_weights_validation() -> None:
    np.testing.assert_allclose(
        chain_step_weights(6, DEFAULT_RECOVERY_CHAIN_DECAY), [1.0, 0.8, 0.6, 0.4, 0.25, 0.15]
    )
    np.testing.assert_array_equal(chain_step_weights(6, None), np.ones(6))
    np.testing.assert_array_equal(chain_step_weights(6, ()), np.ones(6))
    with pytest.raises(ValueError):
        chain_step_weights(6, [1.0] * 5)
    with pytest.raises(ValueError):
        chain_step_weights(6, [1.0] * 5 + [-1.0])
    with pytest.raises(ValueError):
        chain_step_weights(0)


def test_weighted_action_chain_loss_default_bitwise_and_step_weights() -> None:
    torch.manual_seed(0)
    plan = torch.randn(4, 6, 2, dtype=torch.float64)
    target = torch.randn(4, 6, 2, dtype=torch.float64)
    valid = torch.tensor([[0, 1, 1, 0, 1, 1]] * 4, dtype=torch.float64)
    frame_weight = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float64)
    legacy, _ = weighted_action_chain_loss(plan, target, valid, frame_weight=frame_weight)
    same, _ = weighted_action_chain_loss(plan, target, valid, frame_weight=frame_weight, step_weights=None)
    ones, _ = weighted_action_chain_loss(
        plan, target, valid, frame_weight=frame_weight, step_weights=torch.ones(6, dtype=torch.float64)
    )
    assert torch.equal(legacy, same), "默认 None 必须逐位不变"
    assert torch.equal(legacy, ones), "全 1 权重必须逐位不变"

    decay = torch.tensor(DEFAULT_RECOVERY_CHAIN_DECAY, dtype=torch.float64)
    got, _ = weighted_action_chain_loss(
        plan, target, valid, frame_weight=frame_weight, step_weights=decay
    )
    diff = plan[:, 1:] - target[:, 1:]
    per_step = (diff ** 2).mean(dim=-1)
    weight = frame_weight.reshape(-1, 1) * valid[:, 1:] * decay[1:].reshape(1, -1)
    expected = (per_step * weight).sum() / weight.sum().clamp(min=1e-8)
    assert torch.allclose(got, expected, rtol=1e-12)
    # (K,) 与 (K-1,) 两种长度等价
    got_short, _ = weighted_action_chain_loss(
        plan, target, valid, frame_weight=frame_weight, step_weights=decay[1:]
    )
    assert torch.equal(got, got_short)
    with pytest.raises(ValueError):
        weighted_action_chain_loss(plan, target, valid, step_weights=torch.ones(4))


# --------------------------------------------------------------------------- #
# traj_aux recovery-only 掩码（默认关闭）
# --------------------------------------------------------------------------- #

def test_effective_traj_aux_mask_default_off_and_recovery_only() -> None:
    assert effective_traj_aux_mask(4) is None
    base = np.array([1, 0, 1, 1], dtype=np.float32)
    np.testing.assert_array_equal(effective_traj_aux_mask(4, traj_aux_valid=base), base)
    recovery = np.array([1, 0, 0, 1], dtype=np.float32)
    got = effective_traj_aux_mask(4, recovery_row_mask=recovery, recovery_only=True)
    np.testing.assert_array_equal(got, recovery)
    got2 = effective_traj_aux_mask(
        4, traj_aux_valid=base, recovery_row_mask=recovery, recovery_only=True
    )
    np.testing.assert_array_equal(got2, np.array([1, 0, 0, 1], dtype=np.float32))
    with pytest.raises(ValueError):
        effective_traj_aux_mask(4, recovery_only=True)  # 缺 recovery_row_mask
    with pytest.raises(ValueError):
        effective_traj_aux_mask(4, recovery_row_mask=[1, 0], recovery_only=True)  # 行数不符
    with pytest.raises(ValueError):
        effective_traj_aux_mask(4, traj_aux_valid=[1, 0, 1])  # 行数不符


def test_bc_config_recovery_switches_default_off() -> None:
    config = BCConfig()
    assert config.traj_aux_recovery_only is False
    assert config.recovery_row_mask is None


# --------------------------------------------------------------------------- #
# Phase3Config 默认/校验 + 端到端
# --------------------------------------------------------------------------- #

def test_phase3_config_defaults_and_validation() -> None:
    config = Phase3Config()
    assert config.recovery_mix_ratio == 0.0
    assert config.recovery_chain_decay == ()
    assert config.recovery_indices is None
    assert config.lr_primary_scale == pytest.approx(DEFAULT_PRIMARY_LR_SCALE)
    Phase3Config(freeze_mode="recovery")
    Phase3Config(freeze_mode="recovery_experts")
    with pytest.raises(ValueError):
        Phase3Config(freeze_mode="recovery_bogus")
    with pytest.raises(ValueError):
        Phase3Config(recovery_mix_ratio=0.3)  # 缺 recovery_indices
    with pytest.raises(ValueError):
        Phase3Config(recovery_chain_decay=(1.0, 0.5))  # 长度不符
    Phase3Config(recovery_mix_ratio=0.3, recovery_indices=np.array([0, 1]))


def test_phase3_effective_config_recovery_degrades_wm_and_anchor() -> None:
    config = Phase3Config(
        freeze_mode="recovery", anchor_ce_weight=1.0, anchor_wta_weight=1.0, od_weight=0.005
    )
    effective, zeroed = phase3_effective_config(config)
    assert tuple(zeroed) == PHASE3_WM_LOSS_KEYS + ("anchor_ce", "anchor_wta")
    assert float(effective.od_weight) == 0.0
    assert float(effective.anchor_ce_weight) == 0.0
    assert float(config.anchor_ce_weight) == pytest.approx(1.0), "helper 不得修改调用方 config"


def test_phase3_recovery_end_to_end_freezes_and_updates() -> None:
    dataset = _dataset()
    model = _tiny_anchor_model()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    cfg = _phase3_cfg(
        freeze_mode="recovery",
        freeze_prefixes=_RECOVERY_FREEZE,
        recovery_chain_decay=DEFAULT_RECOVERY_CHAIN_DECAY,
        anchor_ce_weight=1.0,
        anchor_wta_weight=1.0,
    )
    metrics = pretrain_bc_phase3(
        model, dataset, cfg, logger=lambda _: None, future_fn=_future_fn_for(dataset)
    )
    assert metrics["freeze_mode"] == "recovery"
    assert tuple(metrics["frozen_loss_keys"]) == PHASE3_WM_LOSS_KEYS + ("anchor_ce", "anchor_wta")
    assert metrics["trainable_param_groups"] == ["trunk", "primary"]
    assert math.isfinite(float(metrics["bc_loss"]))
    # 冻结参数逐位不变（含 experts/router/锚头/WM/encoders/policy/value）
    for name, parameter in model.named_parameters():
        if name.startswith(_RECOVERY_FREEZE):
            assert torch.equal(before[name], parameter.detach()), f"冻结参数 {name} 被更新"
    # trunk（plan 主干）有真实更新
    changed = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    }
    assert any(
        name.startswith(("plan_head.fusion.", "plan_head.norm.", "plan_head.ego_next."))
        for name in changed
    ), "recovery 必须更新 plan 主干"


def test_phase3_recovery_mix_end_to_end_logs_and_runs() -> None:
    dataset = _dataset(episodes=4, steps_per_episode=6)  # count = 24
    model = _tiny_anchor_model()
    messages: list = []
    cfg = _phase3_cfg(
        freeze_mode="recovery",
        freeze_prefixes=_RECOVERY_FREEZE,
        recovery_mix_ratio=0.25,
        recovery_indices=np.arange(12, 24),
    )
    metrics = pretrain_bc_phase3(
        model, dataset, cfg, logger=messages.append, future_fn=_future_fn_for(dataset)
    )
    assert math.isfinite(float(metrics["bc_loss"]))
    assert any("recovery 混采" in message and "12/24" in message for message in messages), messages


def test_phase3_recovery_mix_guard_fail_closed() -> None:
    dataset = _dataset()  # count = 24
    model = _tiny_anchor_model()
    # 越界索引 → 明确报错
    cfg = _phase3_cfg(
        freeze_mode="recovery",
        freeze_prefixes=_RECOVERY_FREEZE,
        recovery_mix_ratio=0.3,
        recovery_indices=np.array([999]),
    )
    with pytest.raises(ValueError, match="越界"):
        pretrain_bc_phase3(model, dataset, cfg, logger=lambda _: None, future_fn=_future_fn_for(dataset))
    # recovery 行与训练子集不相交 → 明确报错
    cfg2 = _phase3_cfg(
        freeze_mode="recovery",
        freeze_prefixes=_RECOVERY_FREEZE,
        train_indices=np.arange(0, 12),
        recovery_mix_ratio=0.3,
        recovery_indices=np.arange(12, 24),
    )
    with pytest.raises(ValueError, match="无 recovery 行"):
        pretrain_bc_phase3(model, dataset, cfg2, logger=lambda _: None, future_fn=_future_fn_for(dataset))


def test_stage_b_phase3_recovery_recipe_wiring(tmp_path) -> None:
    """stages 层配方接线：config freeze=recovery + lr_scale.primary + CLI 覆盖 R2。"""
    from pipeline.stages import _parse_args, run_stage_b_phase3
    from pipeline.trainer import save_checkpoint
    from tests.v2_synthetic import write_v2_dataset

    window = write_v2_dataset(tmp_path / "dagger_r1", episodes=3, steps_per_episode=6)
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(TINY_MODEL_YAML, encoding="utf-8")
    ckpt = tmp_path / "init.pt"
    save_checkpoint(ckpt, _tiny_model(), meta={"stage": "B", "phase": "phase2"})
    config = {
        "stages": {
            "B": {
                "phase3": {
                    "freeze": "recovery",
                    "lr_scale": {"base": 0.25, "specific": 0.5, "primary": 0.05},
                }
            }
        }
    }
    args = _parse_args([
        "--phase3", str(window), "--ckpt", str(ckpt), "--out", str(tmp_path / "out_r1"),
        "--model-config", str(model_cfg), "--device", "cpu", "--seed", "0",
        "--phase3-epochs", "1", "--batch-size", "8", "--val-frac", "0", "--no-monitor",
    ])
    metrics = run_stage_b_phase3(args, config)
    assert metrics["freeze"] == "recovery"
    assert metrics["lr_scale"] == {"base": 0.25, "specific": 0.5, "primary": 0.05}
    assert metrics["trainable_param_groups"] == ["trunk", "primary"]
    assert tuple(metrics["frozen_loss_keys"]) == PHASE3_WM_LOSS_KEYS
    assert metrics["frozen_params"] > 0 and metrics["trainable_params"] > 0

    args2 = _parse_args([
        "--phase3", str(window), "--ckpt", str(ckpt), "--out", str(tmp_path / "out_r2"),
        "--model-config", str(model_cfg), "--device", "cpu", "--seed", "0",
        "--phase3-epochs", "1", "--batch-size", "8", "--val-frac", "0", "--no-monitor",
        "--phase3-freeze", "recovery_experts",
    ])
    metrics2 = run_stage_b_phase3(args2, config)
    assert metrics2["freeze"] == "recovery_experts"
    assert metrics2["trainable_param_groups"] == ["trunk", "primary", "specific"]
