"""lane U1 回归：去聚类 + MoE 负载均衡 + 权重化 specific。

覆盖（用户定稿契约）：
① phase 1 MoE 关闭（输出严格 = primary；专家/router 冻结、step 后逐位不变）；
② phase 2 只训 experts+gate（primary/主干/policy 逐位不变）；
③ 行权重（worst=hard_weight / 其余=mild_weight；与 train_weight 相乘等价）；
④ 负载均衡 aux（Switch 式 α·E·Σ f_i·P_i）：全部路由到同一专家时上升；α 生效；
⑤ 无残留 cluster/gate 旧 tag（monitoring 映射为 None；新负载 tag 映射正确）；
⑥ worst-50% 权重 sidecar 往返 + 严格校验（rows/指纹/ckpt sha256）；
⑦ ``--val-dir``：train 行数 = train-dir 全部行、val 行数 = val-dir 全部行；
⑧ 端到端：phase 1 → 冻结 → phase 2（MoE 开 + 权重 + 负载指标）+ ``--dagger-dir`` 行合并（权重 1.0）；
⑨ **lane U4**：DAgger 行 traj-aux 逐行掩码（掩码行对 traj 损失贡献 = 0、非掩码行照旧、
   `_merge_dagger_rows` 掩码与 merged 行序对齐、掩码全 1 ≡ 无掩码）；
⑩ **lane U5**：多份 `--dagger-dir` 合并（episode_id 累积偏移 / 掩码全 0 / info 逐份行数）
   + 防线（dagger 记录的 spec 来源命中 eval/val → SystemExit）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from net.moe import MoEBlock, switch_load_balance_aux
from pipeline.hard_mining import (
    load_weight_sidecar,
    mine_hard_rows,
    row_weights_from_worst,
    write_weight_sidecar,
)
from pipeline.monitoring import _slim_tag
from pipeline.stages import (
    _PRIMARY_PHASE_FREEZE,
    _SPECIFIC_PHASE_FREEZE,
    _merge_dagger_rows,
    _parse_args,
    run_stage_b,
)
from pipeline.trainer import (
    BCConfig,
    BCDataset,
    apply_freeze_prefixes,
    pretrain_bc,
    row_scale_from_worst,
    traj_valid_for_rows,
)
from tests.v2_synthetic import TINY_MODEL_YAML, write_v2_dataset


def _tiny_model():
    from net.model import DrivingModel

    torch.manual_seed(0)
    return DrivingModel(hidden=16, num_experts=8, expert_hidden=16)


# ------------------------------------------------------------------ ① MoE 关闭
def test_moe_disabled_outputs_primary_only() -> None:
    torch.manual_seed(0)
    block = MoEBlock(hidden=8, num_experts=8, expert_hidden=8)
    with torch.no_grad():  # 让专家分支非零（默认 zero-init 时开关无差异）
        for expert in block.experts:
            expert[-1].weight.normal_(0.5)
    x = torch.randn(4, 8)
    primary = block.primary(x)
    block.set_enabled(False)
    out_off, aux_off = block(x)
    assert torch.allclose(out_off, primary, atol=1e-6), "MoE 关闭必须严格输出 primary"
    assert float(aux_off["moe_enabled"]) == 0.0
    assert "expert_load" not in aux_off and "load_balance_loss" not in aux_off
    block.set_enabled(True)
    out_on, aux_on = block(x)
    assert not torch.allclose(out_on, primary, atol=1e-4), "MoE 开启必须走 experts"
    assert "expert_load" in aux_on
    assert float(aux_on["expert_load"].sum()) == pytest.approx(1.0, rel=1e-5)


def test_phase1_freeze_keeps_experts_and_router_bitwise() -> None:
    model = _tiny_model()
    model.set_moe(enabled=False)
    frozen = apply_freeze_prefixes(model, _PRIMARY_PHASE_FREEZE)
    assert any(name.startswith("plan_head.moe.experts.") for name in frozen)
    assert any(name.startswith("plan_head.moe.router.") for name in frozen)
    assert not any(name.startswith("plan_head.moe.primary.") for name in frozen)
    assert not any(name.startswith("encoders.") for name in frozen)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    with torch.no_grad():
        model.plan_head.moe.primary[-1].weight.grad = torch.ones_like(
            model.plan_head.moe.primary[-1].weight
        )
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(before[name], parameter.detach()), f"phase 1 冻结参数被更新：{name}"


# ------------------------------------------------- ② phase 2 只训 experts+gate
def test_phase2_only_experts_router_trainable_primary_frozen() -> None:
    model = _tiny_model()
    frozen = apply_freeze_prefixes(model, _SPECIFIC_PHASE_FREEZE)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {
        name
        for name, _ in model.named_parameters()
        if name.startswith(
            ("plan_head.moe.experts.", "plan_head.moe.router.", "plan_head.moe.residual_scale")
        )
    }
    assert trainable == expected and trainable, "phase 2 可训练集合必须恰为 experts+router+residual_scale"
    assert any(name.startswith("plan_head.moe.primary.") for name in frozen)
    assert any(name.startswith("policy.") for name in frozen)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    with torch.no_grad():
        model.plan_head.moe.experts[0][-1].weight.grad = torch.ones_like(
            model.plan_head.moe.experts[0][-1].weight
        )
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(before[name], parameter.detach()), f"phase 2 冻结参数被更新：{name}"
    assert not torch.equal(
        before["plan_head.moe.experts.0.2.weight"],
        model.plan_head.moe.experts[0][-1].weight.detach(),
    ), "phase 2 experts 未更新"


# ------------------------------------------------------------------ ③ 行权重
def test_row_scale_weights_worst_and_mild() -> None:
    worst = np.asarray([1.0, 0.0, 1.0, -1.0], dtype=np.float32)
    scale = row_scale_from_worst(worst, hard_weight=1.0, mild_weight=0.1)
    assert np.allclose(scale, [1.0, 0.1, 1.0, 1.0]), "worst=1.0 / 其余=0.1 / 未知=1.0"
    sidecar_weights = row_weights_from_worst(np.asarray([1, 0, 1, 0]), hard_weight=1.0, mild_weight=0.1)
    assert np.allclose(sidecar_weights, [1.0, 0.1, 1.0, 0.1])
    # 有效质量 = 0.5·1.0 + 0.5·0.1 = 0.55（50/50 worst 划分）
    assert float(np.mean(sidecar_weights)) == pytest.approx(0.55)


def test_worst_weights_equivalent_to_premultiplied_train_weight(tmp_path: Path) -> None:
    """BCConfig 的 worst/mild 权重 ≡ 把 train_weight 预乘同样倍率（损失逐位同口径）。"""
    from tests.v2_synthetic import make_v2_arrays

    base_arrays, _ = make_v2_arrays(episodes=4, steps_per_episode=6)
    count = int(base_arrays["episode_id"].size)
    worst = np.zeros(count, dtype=np.float32)
    worst[: count // 2] = 1.0

    def _write(directory: Path, arrays) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "expert_bc.npz", **arrays)
        (directory / "expert_bc.meta.json").write_text(
            json.dumps({"schema_version": 2, "label_names": None}, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    dir_a = _write(tmp_path / "a", {k: np.array(v) for k, v in base_arrays.items()})
    premultiplied = {k: np.array(v) for k, v in base_arrays.items()}
    premultiplied["train_weight"] = (
        np.asarray(premultiplied["train_weight"], dtype=np.float32)
        * row_scale_from_worst(worst, hard_weight=1.0, mild_weight=0.1)
    )
    dir_b = _write(tmp_path / "b", premultiplied)

    def _run(directory: Path, cfg: BCConfig) -> float:
        ds = BCDataset.load(str(directory))
        model = _tiny_model()
        cfg = BCConfig(**{**cfg.__dict__, "epochs": 1, "batch_size": 16, "device": "cpu", "shuffle": False})
        metrics = pretrain_bc(model, ds, cfg, logger=lambda _: None)
        return float(metrics["bc_loss"])

    common = dict(
        epochs=1,
        batch_size=16,
        device="cpu",
        shuffle=False,
        moe_enabled=True,
        load_balance_coef=0.0,
        seed=0,
    )
    loss_a = _run(dir_a, BCConfig(**common, worst_flags=worst, hard_weight=1.0, mild_weight=0.1))
    loss_b = _run(dir_b, BCConfig(**common))
    assert loss_a == pytest.approx(loss_b, rel=1e-5), (loss_a, loss_b)


# ---------------------------------------------------------- ④ 负载均衡 aux
def test_switch_load_balance_aux_rises_when_all_to_one_expert() -> None:
    # Switch 式 aux 值域 = [1, E/k]（门控质量口径）：均匀路由 → 1；塌缩到同一 top-2 对 → 4
    rows = torch.arange(64)
    balanced = torch.full((64, 8), -0.05)
    balanced[rows, (rows * 2) % 8] = 0.02  # 近均匀 P + 逐行铺开的 top-2（f_i=1/4）
    balanced[rows, (rows * 2 + 1) % 8] = 0.01
    collapsed = torch.full((64, 8), -10.0)
    collapsed[:, 0] = 10.0  # 全部门控质量压到 e0 → f≈[1,0..]、P≈[1,0..] → aux ≈ E = 8
    a_balanced = float(switch_load_balance_aux(balanced, top_k=2))
    a_collapsed = float(switch_load_balance_aux(collapsed, top_k=2))
    assert a_balanced == pytest.approx(1.0, rel=0.05)
    assert a_collapsed == pytest.approx(8.0, rel=1e-2)
    assert a_collapsed > a_balanced, "负载塌缩时 aux 必须上升"


def test_moe_load_balance_loss_scaled_by_alpha() -> None:
    torch.manual_seed(0)
    block = MoEBlock(hidden=8, num_experts=8, expert_hidden=8)
    x = torch.randn(16, 8)
    _, aux_off = block(x)
    assert "load_balance_loss" not in aux_off  # α=0 → 不加 aux
    block.set_load_balance_coef(0.01)
    _, aux_on = block(x)
    raw = float(switch_load_balance_aux(block.router(x), top_k=2))
    assert float(aux_on["load_balance_loss"]) == pytest.approx(0.01 * raw, rel=1e-5)
    assert 0.9 <= raw <= 8.0  # 随机初始化：介于均衡下界（1）与塌缩上界（E=8）


def test_pretrain_bc_reports_load_metrics_and_aux(tmp_path: Path) -> None:
    ds_dir = tmp_path / "ds"
    write_v2_dataset(ds_dir, episodes=4, steps_per_episode=6)
    dataset = BCDataset.load(str(ds_dir))
    model = _tiny_model()
    cfg = BCConfig(
        epochs=1, batch_size=16, device="cpu", shuffle=False,
        moe_enabled=True, load_balance_coef=0.01, seed=0,
    )
    metrics = pretrain_bc(model, dataset, cfg, logger=lambda _: None)
    assert metrics["bc_moe_placeholder"] == 0.0
    assert int(metrics["bc_load_count"]) == dataset.count
    loads = [metrics[f"bc_expert_load_{i}"] for i in range(8)]
    assert sum(loads) == pytest.approx(1.0, rel=1e-4)
    assert np.isfinite(metrics["bc_load_cv"]) and np.isfinite(metrics["bc_gate_entropy"])
    assert metrics["bc_load_balance_loss"] > 0.0
    # MoE 关闭 → 无负载指标（占位）
    metrics_off = pretrain_bc(
        _tiny_model(), dataset,
        BCConfig(epochs=1, batch_size=16, device="cpu", shuffle=False, moe_enabled=False, seed=0),
        logger=lambda _: None,
    )
    assert metrics_off["bc_moe_placeholder"] == 1.0 and "bc_load_count" not in metrics_off


# ------------------------------------------------------- ⑤ 无残留 cluster/gate tag
def test_no_legacy_cluster_or_gate_tags() -> None:
    legacy = (
        "train/primary_bc_router_ce",
        "train/primary_bc_router_acc",
        "train/primary_bc_router_acc_majority",
        "val/specific_bc_router_ce",
        "train/primary_bc_router_cluster_loss",
        "train/specific_bc_router_cluster_loss",
        "train/specific_bc_gate_loss",
        "train/specific_bc_gate_ce",
        "train/specific_bc_gate_acc",
        "train/specific_bc_hard_rate",
    )
    for tag in legacy:
        assert _slim_tag(tag) is None, f"旧 cluster/gate tag 仍被保留：{tag}"
    assert _slim_tag("train/specific_bc_load_balance_loss") == "loss/planner/specific/load_balance"
    assert _slim_tag("val/specific_bc_load_balance_loss") == "val/loss/planner/specific/load_balance"
    assert _slim_tag("train/specific_bc_expert_load_0") == "router/expert_load/e0"
    assert _slim_tag("val/specific_bc_expert_load_7") == "val/router/expert_load/e7"
    assert _slim_tag("train/specific_bc_load_cv") == "router/load_cv"
    assert _slim_tag("train/specific_bc_gate_entropy") == "router/gate_entropy"


# ---------------------------------------------------------- ⑥ 权重 sidecar
def test_weight_sidecar_roundtrip_and_strict_validation(tmp_path: Path) -> None:
    ckpt = tmp_path / "primary.pt"
    ckpt.write_bytes(b"fake-primary")
    n = 8
    err = np.linspace(0.1, 0.8, n).astype(np.float32)
    worst = (err > 0.4).astype(np.uint8)
    path = write_weight_sidecar(
        tmp_path / "weight_sidecar.npz",
        worst=worst,
        err_l1=err,
        err_weighted=err * 2.0,
        weight=np.full(n, 2.0, dtype=np.float32),
        dataset_dir=str(tmp_path),
        dataset_meta={"obs_fingerprint": "v2-test-fp", "schema_version": 2},
        ckpt=str(ckpt),
        seed=0,
        hard_frac=0.5,
        hard_weight=1.0,
        mild_weight=0.1,
        threshold=0.4,
        order_rule="(-err_weighted, -err_l1, row_index asc)",
    )
    loaded = load_weight_sidecar(path, rows=n, obs_fingerprint="v2-test-fp")
    assert loaded["worst"].tolist() == worst.tolist()
    assert np.allclose(loaded["row_weight"], [0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0])
    assert loaded["meta"]["ckpt"]["sha256"] and loaded["meta"]["worst_rows"] == 4
    assert loaded["meta"]["hard_weight"] == 1.0 and loaded["meta"]["mild_weight"] == 0.1
    with pytest.raises(ValueError, match="rows"):
        load_weight_sidecar(path, rows=n + 1)
    with pytest.raises(ValueError, match="obs_fingerprint"):
        load_weight_sidecar(path, rows=n, obs_fingerprint="other")
    with pytest.raises(ValueError, match="ckpt.sha256"):
        load_weight_sidecar(path, rows=n, ckpt_sha256="deadbeef")
    with pytest.raises(ValueError, match="缺少权重 sidecar"):
        load_weight_sidecar(tmp_path / "missing.npz", rows=n)


def test_mine_hard_rows_worst_selection_rules() -> None:
    err = np.asarray([1.0, 5.0, 2.0, 4.0, 3.0, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
    mined = mine_hard_rows(err, err * 1.0, hard_frac=0.5)
    assert mined["hard_rows"] == 5 and mined["total_rows"] == 10
    assert np.flatnonzero(mined["worst"]).tolist() == [0, 1, 2, 3, 4]
    tie = mine_hard_rows(np.asarray([0.1, 0.3, 0.2, 0.4]), np.ones(4), hard_frac=0.5)
    assert np.flatnonzero(tie["worst"]).tolist() == [1, 3]
    assert mine_hard_rows(np.asarray([1.0]), np.asarray([1.0]), hard_frac=0.5)["hard_rows"] == 1
    with pytest.raises(ValueError):
        mine_hard_rows(err, err, hard_frac=0.0)


# --------------------------------------------------------------- ⑦ val-dir
def test_val_dir_takes_effect_train_rows_all(tmp_path: Path) -> None:
    train_dir = tmp_path / "BTC_test_expert5k"
    write_v2_dataset(train_dir, episodes=6, steps_per_episode=6)
    val_dir = tmp_path / "BTC_test_expert500val"
    write_v2_dataset(val_dir, episodes=2, steps_per_episode=6)
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(TINY_MODEL_YAML, encoding="utf-8")
    out = tmp_path / "stage_b"
    args = _parse_args([
        "--stage", "B", "--bc-dir", str(train_dir), "--out", str(out),
        "--model-config", str(model_yaml), "--device", "cpu", "--seed", "0",
        "--bc-epochs", "1", "--batch-size", "8", "--no-monitor", "--no-materialize",
        "--val-dir", str(val_dir),
    ])
    metrics = run_stage_b(args, {})
    train_rows = int(BCDataset.load(str(train_dir)).count)
    val_rows = int(BCDataset.load(str(val_dir)).count)
    assert metrics["train_frames"] == train_rows, "train 口径必须是 train-dir 全部行"
    assert metrics["val_frames"] == val_rows
    assert str(metrics["val_source"]).startswith("dir:")
    assert metrics["val_dir"] == str(val_dir)


# --------------------------------------------------- ⑧ 端到端（phase1→phase2）
def test_stage_b_two_phase_end_to_end_with_dagger(tmp_path: Path) -> None:
    ds_dir = tmp_path / "BTC_test_expert5k"
    write_v2_dataset(ds_dir, episodes=6, steps_per_episode=6)
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(TINY_MODEL_YAML, encoding="utf-8")
    dataset = BCDataset.load(str(ds_dir))

    # ① --mine-only：phase 1（MoE 关）+ worst 权重挖掘
    mine_out = tmp_path / "stage_b_mine"
    mine_metrics = run_stage_b(
        _parse_args([
            "--stage", "B", "--bc-dir", str(ds_dir), "--out", str(mine_out),
            "--model-config", str(model_yaml), "--device", "cpu", "--seed", "0",
            "--bc-epochs", "2", "--bc-phase-split", "0.5", "--batch-size", "8",
            "--no-monitor", "--no-materialize", "--mine-only",
        ]),
        {},
    )
    assert mine_metrics["mine_only"] is True
    n_worst = int(np.ceil(0.5 * dataset.count))
    assert mine_metrics["worst_rows"] == n_worst
    sidecar = mine_out / "weight_sidecar.npz"
    assert (mine_out / "primary.pt").is_file() and sidecar.is_file()
    side = load_weight_sidecar(sidecar, rows=dataset.count)
    assert side["meta"]["ckpt"]["sha256"] and side["meta"]["hard_weight"] == 1.0
    assert int(np.count_nonzero(side["worst"] > 0.5)) == n_worst

    # ② phase 2（MoE 开 + 权重 + 负载指标）；③ --dagger-dir 行合并（权重 1.0）
    dagger_dir = tmp_path / "BTC_test_dagger1"
    write_v2_dataset(dagger_dir, episodes=1, steps_per_episode=6)
    out = tmp_path / "stage_b_final"
    metrics = run_stage_b(
        _parse_args([
            "--stage", "B", "--bc-dir", str(ds_dir), "--out", str(out),
            "--model-config", str(model_yaml), "--device", "cpu", "--seed", "0",
            "--bc-epochs", "2", "--bc-phase-split", "0.5", "--batch-size", "8",
            "--no-monitor", "--no-materialize", "--weight-sidecar", str(sidecar),
            "--dagger-dir", str(dagger_dir), "--load-balance-coef", "0.01",
        ]),
        {},
    )
    assert metrics["moe_phase1_enabled"] is False
    assert metrics["load_balance_coef"] == 0.01
    assert metrics["worst_rows"] == n_worst
    assert metrics["dagger"]["dagger_rows"] == int(BCDataset.load(str(dagger_dir)).count)
    # lane U4：DAgger 行全部被 traj-aux 掩码（合成 traj6 不进 traj 损失）
    assert metrics["dagger"]["traj_aux_masked_rows"] == metrics["dagger"]["dagger_rows"]
    primary_phase = metrics["primary"]
    specific_phase = metrics["specific"]
    # phase 1：MoE 关闭 → 无负载指标（占位）
    assert primary_phase["bc_moe_placeholder"] == 1.0
    # phase 2：负载指标齐全（train）
    assert specific_phase["bc_moe_placeholder"] == 0.0
    loads = [specific_phase[f"bc_expert_load_{i}"] for i in range(8)]
    assert sum(loads) == pytest.approx(1.0, rel=1e-4)
    assert np.isfinite(specific_phase["bc_load_cv"]) and np.isfinite(specific_phase["bc_gate_entropy"])
    assert specific_phase["bc_load_balance_loss"] >= 0.0
    assert int(specific_phase["bc_load_count"]) > 0
    # phase 2：val 孪生（负载 + loss）
    val_metrics = metrics["specific"].get("val") or {}
    assert val_metrics.get("bc_load_count") == val_metrics.get("bc_load_count") and val_metrics
    assert "bc_load_cv" in val_metrics and "bc_gate_entropy" in val_metrics
    assert "bc_load_balance_loss" in val_metrics
    # 回归（2026-09-28 修复）：val 的"动作误差"必须来自 action_pred-target_action；
    # 此前误用轨迹误差（traj_pred-traj6）→ 与 traj_mae 数值雷同。此断言锁死口径。
    act_err = val_metrics.get("bc_action_err_weighted_mean")
    traj_mae = val_metrics.get("bc_traj_mae_m")
    assert act_err is not None and traj_mae is not None
    assert abs(float(act_err) - float(traj_mae)) > 1e-9, "val action_err 不应与 traj_mae 同值（口径回归）"
    # 冻结语义端到端：primary 段权重在 phase 2 逐位不变；experts 已更新
    # 同一 run 内对比（两次 run 的随机初始化不同；run 2 自己的 primary.pt vs final.pt）
    primary_ckpt = torch.load(out / "primary.pt", map_location="cpu", weights_only=False)["model"]
    final_ckpt = torch.load(out / "final.pt", map_location="cpu", weights_only=False)["model"]
    for name in ("plan_head.moe.primary.0.weight", "plan_head.moe.primary.2.weight", "policy.net.0.weight"):
        if name in primary_ckpt and name in final_ckpt:
            assert torch.equal(primary_ckpt[name], final_ckpt[name]), f"phase 2 不应更新 {name}"
    expert_names = [n for n in final_ckpt if ".moe.experts." in n]
    # v6 A1 新架构：top-2 路由可能不落 experts.0（实测落 experts.3/4，更新幅度同基线）→
    # 断言放宽为"任一 expert 被更新"；primary/policy 的逐位冻结断言保持不变。
    assert any(
        not torch.equal(primary_ckpt[n], final_ckpt[n]) for n in expert_names if n in primary_ckpt
    ), "phase 2 必须更新 experts（任一 expert 即可）"


# ------------------------------------------------- lane U4：DAgger 行 traj-aux 逐行掩码
def test_traj_valid_for_rows_alignment_and_bounds() -> None:
    flags = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert traj_valid_for_rows(None, np.asarray([0, 1])) is None, "None = 全 1（旧行为）"
    assert traj_valid_for_rows(flags, np.asarray([3, 0])).tolist() == [0.0, 1.0], "行索引对齐"
    with pytest.raises(ValueError, match="traj_aux_valid"):
        traj_valid_for_rows(flags, np.asarray([4]))


def test_traj_aux_mask_excludes_rows_from_traj_loss(tmp_path: Path) -> None:
    """① 掩码行对 traj 损失贡献 = 0（破坏其 traj6 也不变）；② 非掩码行仍出力。"""
    from tests.v2_synthetic import make_v2_arrays

    base, _ = make_v2_arrays(episodes=4, steps_per_episode=6)
    n = int(base["episode_id"].size)
    mask = np.ones(n, dtype=np.float32)
    mask[: n // 2] = 0.0  # 前一半 = DAgger 行（掩 0）

    def _write(directory: Path, arrays) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(directory / "expert_bc.npz", **arrays)
        (directory / "expert_bc.meta.json").write_text(
            json.dumps({"schema_version": 2, "label_names": None}, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    def _corrupt(source, rows) -> dict:
        arrays = {k: np.array(v) for k, v in source.items()}
        arrays["traj6"] = np.asarray(arrays["traj6"], dtype=np.float32).copy()
        arrays["traj6"][rows] += 100.0  # 100 m 级垃圾：若进 traj 损失必然可见
        return arrays

    clean = _write(tmp_path / "clean", {k: np.array(v) for k, v in base.items()})
    corrupt_masked = _write(tmp_path / "corrupt_masked", _corrupt(base, slice(0, n // 2)))
    corrupt_unmasked = _write(tmp_path / "corrupt_unmasked", _corrupt(base, slice(n // 2, n)))

    def _run(directory: Path) -> dict:
        dataset = BCDataset.load(str(directory))
        model = _tiny_model()
        cfg = BCConfig(
            epochs=1, batch_size=32, device="cpu", shuffle=False,
            moe_enabled=True, load_balance_coef=0.0, traj_aux_valid=mask, seed=0,
        )
        return pretrain_bc(model, dataset, cfg, logger=lambda _: None)

    metrics_clean = _run(clean)
    metrics_masked = _run(corrupt_masked)
    metrics_unmasked = _run(corrupt_unmasked)

    # ① 被掩码行的 traj6 是垃圾 → traj 项与总损失逐位不变（贡献 = 0）
    assert float(metrics_masked["bc_traj_loss"]) == pytest.approx(
        float(metrics_clean["bc_traj_loss"]), rel=1e-6
    ), "掩码行必须不吃 traj 损失"
    assert float(metrics_masked["bc_loss"]) == pytest.approx(
        float(metrics_clean["bc_loss"]), rel=1e-6
    ), "掩码行不应影响总损失"
    # ② 非掩码行照旧出力：破坏它们 → traj 项显著变化
    assert float(metrics_unmasked["bc_traj_loss"]) != pytest.approx(
        float(metrics_clean["bc_traj_loss"]), rel=1e-3
    ), "非掩码行必须仍吃 traj 损失"
    # 监控统计保持全量口径（报告照旧）：被掩码行的 traj6 仍进 traj_mse 统计
    assert float(metrics_masked["bc_traj_mse"]) != pytest.approx(
        float(metrics_clean["bc_traj_mse"]), rel=1e-3
    ), "bc_traj_mse 应保持全量口径（含被掩码行）"


def test_traj_aux_mask_all_ones_equals_none(tmp_path: Path) -> None:
    """掩码全 1 ≡ 无掩码（None）：旧行为逐位不变。"""
    ds_dir = tmp_path / "ds"
    write_v2_dataset(ds_dir, episodes=4, steps_per_episode=6)
    dataset = BCDataset.load(str(ds_dir))
    n = int(dataset.count)

    def _run(traj_aux_valid) -> dict:
        model = _tiny_model()
        cfg = BCConfig(
            epochs=1, batch_size=32, device="cpu", shuffle=False,
            moe_enabled=True, load_balance_coef=0.0, traj_aux_valid=traj_aux_valid, seed=0,
        )
        return pretrain_bc(model, dataset, cfg, logger=lambda _: None)

    metrics_none = _run(None)
    metrics_ones = _run(np.ones(n, dtype=np.float32))
    for key in ("bc_loss", "bc_traj_loss", "bc_action_loss", "bc_traj_mse", "bc_traj_mae_m"):
        assert float(metrics_ones[key]) == pytest.approx(float(metrics_none[key]), rel=1e-9), key


def test_merge_dagger_rows_traj_mask_alignment(tmp_path: Path) -> None:
    """③ `_merge_dagger_rows` 返回掩码：长度 = merged 行数；主集 = 1.0 / DAgger = 0.0。"""
    main_dir = tmp_path / "main"
    write_v2_dataset(main_dir, episodes=4, steps_per_episode=6)
    dagger_dir = tmp_path / "dagger"
    write_v2_dataset(dagger_dir, episodes=1, steps_per_episode=6)
    main = BCDataset.load(str(main_dir))
    n_main = int(main.count)
    n_dagger = int(BCDataset.load(str(dagger_dir)).count)
    worst = np.zeros(n_main, dtype=np.float32)
    worst[: n_main // 2] = 1.0

    merged, merged_idx, merged_worst, traj_valid, info = _merge_dagger_rows(
        main, np.arange(n_main, dtype=np.int64), worst, [str(dagger_dir)], logger=lambda _: None
    )
    assert merged.count == n_main + n_dagger
    assert len(traj_valid) == merged.count and merged_idx.size == merged.count, "长度对齐 merged 行序"
    assert np.allclose(traj_valid[:n_main], 1.0), "主集行 traj-aux = 1"
    assert np.allclose(traj_valid[n_main:], 0.0), "DAgger 行 traj-aux = 0"
    assert info["traj_aux_masked_rows"] == n_dagger and info["dagger_rows"] == n_dagger
    assert np.allclose(merged_worst[n_main:], 1.0), "DAgger 行权重 = worst（1.0）"


# ------------------------------------------- lane U5：多份 dagger 合并 + 隔离防线
def test_merge_multiple_dagger_rows_cumulative_offsets_and_mask(tmp_path: Path) -> None:
    """多份 `--dagger-dir`：episode_id 累积偏移、traj 掩码全 0、info 逐份行数、行序 = 主集→各份。"""
    main_dir = tmp_path / "main"
    write_v2_dataset(main_dir, episodes=4, steps_per_episode=6)
    d1_dir = tmp_path / "dagger1"
    write_v2_dataset(d1_dir, episodes=1, steps_per_episode=6)
    d2_dir = tmp_path / "dagger2"
    write_v2_dataset(d2_dir, episodes=2, steps_per_episode=6)
    main = BCDataset.load(str(main_dir))
    n_main = int(main.count)
    n1 = int(BCDataset.load(str(d1_dir)).count)
    n2 = int(BCDataset.load(str(d2_dir)).count)
    worst = np.zeros(n_main, dtype=np.float32)
    worst[: n_main // 2] = 1.0

    merged, merged_idx, merged_worst, traj_valid, info = _merge_dagger_rows(
        main, np.arange(n_main, dtype=np.int64), worst, [str(d1_dir), str(d2_dir)], logger=lambda _: None
    )
    assert merged.count == n_main + n1 + n2
    assert len(traj_valid) == merged.count and merged_idx.size == merged.count
    assert np.allclose(traj_valid[: n_main + n1 + n2], [1.0] * n_main + [0.0] * n1 + [0.0] * n2)
    assert [item["rows"] for item in info["dagger_dirs"]] == [n1, n2]
    assert info["dagger_rows"] == n1 + n2 and info["traj_aux_masked_rows"] == n1 + n2
    # episode_id 累积偏移：各份的 episode 集合互不相交（行内重复是同一 episode 的多帧）
    episodes = np.asarray(merged.arrays["episode_id"])
    parts = [episodes[:n_main], episodes[n_main : n_main + n1], episodes[n_main + n1 :]]
    assert len(np.unique(episodes)) == 4 + 1 + 2, "episode 总数 = 主集 4 + d1 1 + d2 2"
    assert not (set(parts[0].tolist()) & set(parts[1].tolist()))
    assert not (set(parts[1].tolist()) & set(parts[2].tolist()))
    assert not (set(parts[0].tolist()) & set(parts[2].tolist()))
    offsets = info["episode_id_offsets"]
    assert offsets[1] > offsets[0], "第二份偏移必须大于第一份（累积）"
    assert np.all(episodes[n_main : n_main + n1] >= offsets[0])
    assert np.all(episodes[n_main + n1 :] >= offsets[1])
    assert np.allclose(merged_worst[n_main:], 1.0)


def test_merge_dagger_rows_defense_blocks_eval_spec_source(tmp_path: Path) -> None:
    """防线：dagger 目录记录的 spec 来源命中 eval/val → SystemExit；来源缺失 → 仅警告。"""
    main_dir = tmp_path / "main"
    write_v2_dataset(main_dir, episodes=2, steps_per_episode=6)
    main = BCDataset.load(str(main_dir))
    worst = np.zeros(int(main.count), dtype=np.float32)

    bad_dir = tmp_path / "dagger_eval_source"
    write_v2_dataset(bad_dir, episodes=1, steps_per_episode=6)
    (bad_dir / "report.json").write_text(
        json.dumps({"provenance": {"spec_source": "env/specs/scenarios_eval500.json"}}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="隔离守卫"):
        _merge_dagger_rows(
            main, np.arange(int(main.count), dtype=np.int64), worst, [str(bad_dir)], logger=lambda _: None
        )

    # 来源字段缺失 → 警告日志、不阻塞（返回正常 5 元组）
    warn_dir = tmp_path / "dagger_no_source"
    write_v2_dataset(warn_dir, episodes=1, steps_per_episode=6)
    logged: list = []
    merged, _, _, traj_valid, info = _merge_dagger_rows(
        main, np.arange(int(main.count), dtype=np.int64), worst, [str(warn_dir)], logger=logged.append
    )
    assert merged.count > int(main.count) and float(np.sum(traj_valid)) == float(int(main.count))
    assert info["isolation_guard"] == []
    assert any("读不到 spec 来源" in str(line) for line in logged), "字段缺失必须留警告"
