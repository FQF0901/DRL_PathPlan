"""fix-11 phase3 安全配方回归（freeze 契约修复 + trunk_only + clean150 守护）。

背景（根因诊断 ``/tmp/opencode/v7_struct_fail_diag.md`` §3）：

① ``_SPECIFIC_PHASE_FREEZE`` 曾漏 4 个 K-anchor 锚头前缀 → phase3 ``specific_only`` 下
   13 个锚头参数落入 base 组被训练（唯一监督 = chain L2）→ 锚计划左偏；契约修复 =
   锚头加入冻结清单，**测试断言 phase3 后锚头权重逐位不变（两 ckpt 比对）**；
② specific experts/router 训练是决定性破坏项 → 安全配方 ``freeze=trunk_only``：只训共享主干，
   冻结 experts/router/residual_scale + 锚头 + WM/value，**锚 CE/WTA 保留**（塑形共享特征）；
③ epoch 级 clean150 守护：``--phase3-guard-spec`` + ``--phase3-guard-min-success`` →
   每 epoch 评测、崩塌（或评测不可得）即中止（fail-closed）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline.stages import (
    FrameWindows,
    Phase3GuardCollapse,
    _BC_STEP_STRIDE,
    _PHASE3_ANCHOR_HEAD_PREFIXES,
    _PHASE3_SAFE_FREEZE,
    _SPECIFIC_PHASE_FREEZE,
    _bc_channel_keys,
    _parse_args,
    _phase3_future_fn,
    _phase3_guard_argv,
    _phase3_guard_decision,
    run_stage_b_phase3,
)
from pipeline.trainer import (
    BCDataset,
    PHASE3_WM_LOSS_KEYS,
    Phase3Config,
    _PHASE3_SPECIFIC_PREFIXES,
    apply_freeze_prefixes,
    build_phase3_optimizer,
    phase3_effective_config,
    pretrain_bc_phase3,
    save_checkpoint,
)
from tests.v2_synthetic import TINY_MODEL_YAML, make_v2_arrays, write_v2_dataset

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

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
        # 打散锚头/专家参数（零初始化末层会掩盖"未更新"断言）；policy.mu 非零（action 梯度可到主干）
        for name, parameter in model.named_parameters():
            if name.startswith(_PHASE3_ANCHOR_HEAD_PREFIXES) or name.startswith("plan_head.moe.experts."):
                parameter.normal_(0.0, 0.05)
        model.policy.mu.weight.normal_(0.0, 0.1)
        model.policy.mu.bias.normal_(0.0, 0.1)
    return model


def _model_cfg(tmp_path: Path, *, anchors: bool = True) -> Path:
    path = tmp_path / ("model_anchor.yaml" if anchors else "model.yaml")
    path.write_text(_ANCHOR_YAML if anchors else TINY_MODEL_YAML, encoding="utf-8")
    return path


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


def _is_anchor_param(name: str) -> bool:
    return name.startswith(_PHASE3_ANCHOR_HEAD_PREFIXES)


def _phase3_cfg(**overrides) -> Phase3Config:
    base = dict(
        epochs=1, batch_size=8, micro_batch_size=4, device="cpu", shuffle=False, seed=0,
        val_indices=None,
    )
    base.update(overrides)
    return Phase3Config(**base)


# --------------------------------------------------------------------------- #
# ① freeze 契约修复：锚头前缀进 specific_only 冻结清单 + 两 ckpt 逐位比对
# --------------------------------------------------------------------------- #

def test_specific_phase_freeze_includes_anchor_head_prefixes() -> None:
    assert _PHASE3_ANCHOR_HEAD_PREFIXES == (
        "plan_head.anchor_head.",
        "plan_head.speed_head.",
        "plan_head.residual_head.",
        "plan_head.anchor_embed",
    )
    for prefix in _PHASE3_ANCHOR_HEAD_PREFIXES:
        assert prefix in _SPECIFIC_PHASE_FREEZE, f"fix-11 契约：{prefix} 必须在冻结清单"
    model = _tiny_anchor_model()
    apply_freeze_prefixes(model, _SPECIFIC_PHASE_FREEZE)
    for name, parameter in model.named_parameters():
        if _is_anchor_param(name):
            assert not parameter.requires_grad, f"锚头参数 {name} 未冻结"
    # specific_only 语义不变：experts/router/residual_scale 仍可训
    assert any(
        parameter.requires_grad and name.startswith(_PHASE3_SPECIFIC_PREFIXES)
        for name, parameter in model.named_parameters()
    )


def test_phase3_specific_only_anchor_weights_bitwise_unchanged_two_ckpts(tmp_path: Path) -> None:
    """契约（fix-11）：phase3 specific_only 训练后锚头权重逐位不变（两 ckpt 比对）。"""
    dataset = _dataset()
    model = _tiny_anchor_model()
    before_ckpt = tmp_path / "before.pt"
    save_checkpoint(before_ckpt, model)
    cfg = _phase3_cfg(
        freeze_mode="specific_only",
        freeze_prefixes=_SPECIFIC_PHASE_FREEZE,
        val_indices=np.arange(dataset.count, dtype=np.int64),
        # 即使锚损失权重 > 0（配置层），specific_only 下也应自动降级 → 锚头无任何监督
        anchor_ce_weight=1.0,
        anchor_wta_weight=1.0,
    )
    metrics = pretrain_bc_phase3(
        model, dataset, cfg, logger=lambda _: None, future_fn=_future_fn_for(dataset)
    )
    after_ckpt = tmp_path / "after.pt"
    save_checkpoint(after_ckpt, model)

    assert tuple(metrics["frozen_loss_keys"]) == PHASE3_WM_LOSS_KEYS + ("anchor_ce", "anchor_wta")
    assert float(metrics["bc_anchor_ce_loss"] if "bc_anchor_ce_loss" in metrics else 0.0) == 0.0

    before = torch.load(before_ckpt, map_location="cpu", weights_only=False)["model"]
    after = torch.load(after_ckpt, map_location="cpu", weights_only=False)["model"]
    anchor_names = [name for name in before if _is_anchor_param(name)]
    assert anchor_names, "测试模型必须带锚头参数"
    for name in anchor_names:
        assert torch.equal(before[name], after[name]), f"锚头参数 {name} 被更新（契约违反）"
    # 对照：被冻结的锚头不更新，而 specific 专家确实更新（否则断言无信息量）
    assert any(
        not torch.equal(before[name], after[name])
        for name in before
        if name.startswith("plan_head.moe.experts.")
    )
    for name, parameter in model.named_parameters():
        if _is_anchor_param(name):
            assert parameter.grad is None, f"冻结锚头 {name} 不应有梯度"


# --------------------------------------------------------------------------- #
# ② 安全配方 trunk_only：只训共享主干 + 锚 CE/WTA 保留
# --------------------------------------------------------------------------- #

def test_phase3_trunk_only_freezes_specific_and_keeps_anchor_losses() -> None:
    dataset = _dataset()
    model = _tiny_anchor_model()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    cfg = _phase3_cfg(
        freeze_mode="trunk_only",
        freeze_prefixes=_PHASE3_SAFE_FREEZE,
        val_indices=np.arange(dataset.count, dtype=np.int64),
        anchor_ce_weight=1.0,
        anchor_wta_weight=1.0,
    )
    metrics = pretrain_bc_phase3(
        model, dataset, cfg, logger=lambda _: None, future_fn=_future_fn_for(dataset)
    )

    assert metrics["freeze_mode"] == "trunk_only"
    assert tuple(metrics["frozen_loss_keys"]) == PHASE3_WM_LOSS_KEYS
    assert metrics["frozen_params"] > 0 and metrics["trainable_params"] > 0
    assert metrics["trainable_param_groups"] == ["base"], "trunk_only 只应有 base 优化器组"
    # 可训集合 = 共享主干（不含 specific/锚头）
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert not name.startswith(_PHASE3_SAFE_FREEZE), f"{name} 不应在 trunk_only 下可训"
    # 冻结项逐位不变（experts/router/residual_scale/锚头/WM/value）
    for name, parameter in model.named_parameters():
        if name.startswith(_PHASE3_SAFE_FREEZE):
            assert torch.equal(before[name], parameter.detach()), f"冻结参数 {name} 被更新"
    # 共享主干有真实更新
    changed = {
        name for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    }
    assert any(
        name.startswith(("encoders.", "mem_encoder.", "plan_head.fusion.", "policy."))
        for name in changed
    ), "trunk_only 必须更新共享主干"
    # 锚 CE/WTA 保留（有梯度/有记录）；WM/ego_next 上游监督降级为 0
    assert float(metrics["bc_anchor_ce_loss"]) > 0.0
    assert float(metrics["bc_anchor_wta_loss"]) > 0.0
    for key in ("bc_ego_next_loss", "bc_od_loss", "bc_ld_loss", "bc_presence_loss", "bc_entry_loss"):
        assert float(metrics[key]) == 0.0, f"{key} 应在 trunk_only 下保守降级"
        assert float(metrics["val"][key]) == 0.0


def test_phase3_trunk_only_effective_config_keeps_anchor_keys() -> None:
    cfg = Phase3Config(
        freeze_mode="trunk_only", anchor_ce_weight=1.0, anchor_wta_weight=1.0, od_weight=0.005
    )
    effective, zeroed = phase3_effective_config(cfg)
    assert tuple(zeroed) == PHASE3_WM_LOSS_KEYS
    assert float(effective.anchor_ce_weight) == pytest.approx(1.0)
    assert float(effective.anchor_wta_weight) == pytest.approx(1.0)
    assert float(effective.od_weight) == 0.0
    assert cfg.anchor_ce_weight == pytest.approx(1.0), "helper 不得修改调用方 config"


def test_phase3_trunk_only_optimizer_contract() -> None:
    model = _tiny_anchor_model()
    frozen = apply_freeze_prefixes(model, _PHASE3_SAFE_FREEZE)
    assert frozen
    optimizer = build_phase3_optimizer(model, 1e-3, freeze_mode="trunk_only")
    assert [group.get("name") for group in optimizer.param_groups] == ["base"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2.5e-4)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable
    assert not any(
        name.startswith(_PHASE3_SPECIFIC_PREFIXES) or _is_anchor_param(name) for name in trainable
    )
    # 冻结契约被绕过（experts 被解冻）→ trunk_only 拒绝静默带病训练
    for name, parameter in model.named_parameters():
        if name.startswith("plan_head.moe.experts."):
            parameter.requires_grad_(True)
            break
    with pytest.raises(ValueError, match="trunk_only"):
        build_phase3_optimizer(model, 1e-3, freeze_mode="trunk_only")


# --------------------------------------------------------------------------- #
# ③ epoch 级 clean150 守护（纯函数 + argv + 端到端桩）
# --------------------------------------------------------------------------- #

def test_phase3_guard_decision_pure() -> None:
    assert _phase3_guard_decision(None, 0.13) == (False, "missing")
    collapsed, reason = _phase3_guard_decision(0.10, 0.13)
    assert collapsed and "0.1000" in reason
    assert _phase3_guard_decision(0.13, 0.13) == (False, ""), "阈值取严格小于（等于不停）"
    assert _phase3_guard_decision(0.26, 0.13) == (False, "")
    assert _phase3_guard_decision(float("nan"), 0.13) == (False, "non_finite")


def test_phase3_guard_argv_pins_protocol(tmp_path: Path) -> None:
    argv = _phase3_guard_argv(
        ckpt=tmp_path / "epoch001.pt",
        spec=tmp_path / "specs150.json",
        config="config/eval.yaml",
        out_root=tmp_path,
        name="guard/epoch001",
        workers=4,
        device="cuda",
    )
    joined = " ".join(argv)
    assert "--eval-reference plan" in joined
    assert "--tracker lqr" in joined
    assert "--workers 4" in joined
    assert "--device cuda" in joined
    assert "--name guard/epoch001" in joined


def _install_guard_stub(monkeypatch, success_by_epoch: dict, calls: list) -> None:
    """桩掉守护子进程：按 argv 解析 epoch，写假 metrics.json，返回 rc=0。"""

    def fake_run(argv, log_path):  # noqa: ANN001
        ckpt = Path(argv[argv.index("--ckpt") + 1])
        epoch = int(ckpt.stem.replace("epoch", ""))
        out_root = Path(argv[argv.index("--out") + 1])
        name = argv[argv.index("--name") + 1]
        metrics = out_root / name / "metrics.json"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text(
            json.dumps({"overall": {"success_rate": success_by_epoch[epoch]}}), encoding="utf-8"
        )
        calls.append(epoch)
        return 0

    monkeypatch.setattr("pipeline.stages._phase3_guard_run", fake_run)


def _guard_args(tmp_path: Path, window: Path, ckpt: Path, *, epochs: int) -> list:
    spec = tmp_path / "clean150.json"
    spec.write_text("[]", encoding="utf-8")
    return [
        "--phase3", str(window), "--ckpt", str(ckpt), "--out", str(tmp_path / "out"),
        "--model-config", str(_model_cfg(tmp_path, anchors=False)), "--device", "cpu",
        "--seed", "0", "--phase3-epochs", str(epochs), "--batch-size", "8", "--val-frac", "0",
        "--no-monitor", "--ckpt-every", "1",
        "--phase3-freeze", "specific_only",
        "--phase3-guard-spec", str(spec),
        "--phase3-guard-config", str(_model_cfg(tmp_path, anchors=False)),
        "--phase3-guard-workers", "1",
        "--phase3-guard-min-success", "0.5",
    ]


def test_phase3_epoch_guard_stops_on_collapse(monkeypatch, tmp_path: Path) -> None:
    window = write_v2_dataset(tmp_path / "dagger_r1", episodes=3, steps_per_episode=6)
    ckpt = tmp_path / "init.pt"
    save_checkpoint(ckpt, _tiny_model())
    calls: list = []
    _install_guard_stub(monkeypatch, {1: 0.0, 2: 1.0}, calls)
    args = _parse_args(_guard_args(tmp_path, window, ckpt, epochs=2))
    with pytest.raises(Phase3GuardCollapse, match="clean150"):
        run_stage_b_phase3(args, {"stages": {"B": {"phase3": {"freeze": "specific_only"}}}})
    assert calls == [1], "崩塌后不得继续下一 epoch"
    guard_json = tmp_path / "out" / "guard" / "guard.json"
    payload = json.loads(guard_json.read_text(encoding="utf-8"))
    assert payload["min_success"] == pytest.approx(0.5)
    assert len(payload["records"]) == 1
    assert payload["records"][0]["collapsed"] is True
    assert payload["records"][0]["success"] == pytest.approx(0.0)


def test_phase3_epoch_guard_passes_and_records(monkeypatch, tmp_path: Path) -> None:
    window = write_v2_dataset(tmp_path / "dagger_r1", episodes=3, steps_per_episode=6)
    ckpt = tmp_path / "init.pt"
    save_checkpoint(ckpt, _tiny_model())
    calls: list = []
    _install_guard_stub(monkeypatch, {1: 1.0, 2: 1.0}, calls)
    args = _parse_args(_guard_args(tmp_path, window, ckpt, epochs=2))
    metrics = run_stage_b_phase3(args, {"stages": {"B": {"phase3": {"freeze": "specific_only"}}}})
    assert calls == [1, 2]
    assert metrics["guard_clean150"]["min_success"] == pytest.approx(0.5)
    assert metrics["guard_clean150"]["workers"] == 1
    guard_json = tmp_path / "out" / "guard" / "guard.json"
    payload = json.loads(guard_json.read_text(encoding="utf-8"))
    assert [record["success"] for record in payload["records"]] == [1.0, 1.0]
    assert all(record["collapsed"] is False for record in payload["records"])


def test_phase3_guard_requires_min_success(tmp_path: Path) -> None:
    window = write_v2_dataset(tmp_path / "dagger_r1", episodes=3, steps_per_episode=6)
    ckpt = tmp_path / "init.pt"
    save_checkpoint(ckpt, _tiny_model())
    spec = tmp_path / "clean150.json"
    spec.write_text("[]", encoding="utf-8")
    args = _parse_args([
        "--phase3", str(window), "--ckpt", str(ckpt), "--out", str(tmp_path / "out"),
        "--model-config", str(_model_cfg(tmp_path, anchors=False)), "--device", "cpu",
        "--seed", "0", "--phase3-epochs", "1", "--batch-size", "8", "--val-frac", "0",
        "--no-monitor", "--phase3-guard-spec", str(spec),
    ])
    with pytest.raises(SystemExit, match="min-success"):
        run_stage_b_phase3(args, {"stages": {"B": {"phase3": {}}}})
