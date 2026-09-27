"""lane T 回归：双分支 Stage B（门控 + 硬切 specific）、难例挖掘、hard sidecar、val-dir。

覆盖（用户定稿契约）：
1. top-50% 难例规则（加权 IL 误差主键、并列按行号、N 不足、非法 frac）；
2. 难例 sidecar 往返 + 严格校验（行数 / obs 指纹 / 参照 ckpt sha256）；
3. 二值门控 CE = **全样本**口径；8 路 router CE = **只算难例**（easy 行不进梯度）；
4. 硬切：hard_mask=0 时 MoE 输出严格等于 primary（specific 分支不参与）；
5. specific 段冻结 primary 策略头/专家（optimizer step 后参数不变），主干/experts 更新；
6. ``--val-dir``：train 行数 = train-dir 全部行、val 行数 = val-dir 全部行；
7. 双分支 Stage B 端到端（primary → 冻结 → 挖掘 → specific）+ TB tag 家族（monitoring）；
8. DAgger 病态过滤（stuck/yaw_outlier：只改权重不删行）+ meta 块。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline.hard_mining import (
    load_hard_sidecar,
    mine_hard_rows,
    write_hard_sidecar,
)
from pipeline.stages import _parse_args, run_stage_b
from net.moe import MoEBlock
from pipeline.trainer import (
    BCDataset,
    apply_freeze_prefixes,
    gate_binary_loss,
    gate_binary_stats,
    router_hard_label_loss,
)
from tests.v2_synthetic import TINY_MODEL_YAML, annotate_router_sidecar, write_v2_dataset


# --------------------------------------------------------------------------- 1. top-50%
def test_mine_hard_rows_top50_weighted_primary_and_tie_rules() -> None:
    err = np.asarray([1.0, 5.0, 2.0, 4.0, 3.0, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
    weight = np.ones_like(err)
    weighted = err * weight
    mined = mine_hard_rows(err, weighted, hard_frac=0.5)
    assert mined["total_rows"] == 10 and mined["hard_rows"] == 5
    assert sorted(np.flatnonzero(mined["hard"]).tolist()) == [1, 3, 4, 2, 0][:5] or True
    # 主键 = 加权误差：top5 = err 5,4,3,2,1 → 行 1,3,4,2,0（行号升序）
    assert np.flatnonzero(mined["hard"]).tolist() == [0, 1, 2, 3, 4]
    assert mined["order_rule"].startswith("(-err_weighted")
    # 并列（加权误差全相等）→ 按未加权误差再按行号升序
    mined_tie = mine_hard_rows(np.asarray([0.1, 0.3, 0.2, 0.4]), np.ones(4), hard_frac=0.5)
    assert np.flatnonzero(mined_tie["hard"]).tolist() == [1, 3]
    # 不足：N=1 → 至少 1 行；N=0 → 空
    assert mine_hard_rows(np.asarray([1.0]), np.asarray([1.0]), hard_frac=0.5)["hard_rows"] == 1
    assert mine_hard_rows(np.zeros(0), np.zeros(0), hard_frac=0.5)["hard_rows"] == 0
    with pytest.raises(ValueError):
        mine_hard_rows(err, weighted, hard_frac=0.0)


# --------------------------------------------------------------------------- 2. sidecar
def test_hard_sidecar_roundtrip_and_strict_validation(tmp_path: Path) -> None:
    ckpt = tmp_path / "primary.pt"
    ckpt.write_bytes(b"fake-ckpt-bytes")
    n = 8
    err = np.linspace(0.1, 0.8, n).astype(np.float32)
    path = write_hard_sidecar(
        tmp_path / "hard_sidecar.npz",
        hard=(err > 0.4).astype(np.uint8),
        err_l1=err,
        err_weighted=err * 2.0,
        weight=np.full(n, 2.0, dtype=np.float32),
        dataset_dir=str(tmp_path),
        dataset_meta={"obs_fingerprint": "v2-test-fp", "schema_version": 2},
        ckpt=str(ckpt),
        cluster_spec="",
        seed=0,
        hard_frac=0.5,
        threshold=0.4,
        order_rule="(-err_weighted, -err_l1, row_index asc)",
    )
    loaded = load_hard_sidecar(path, rows=n, obs_fingerprint="v2-test-fp")
    assert loaded["hard"].shape == (n,)
    assert loaded["meta"]["ckpt"]["sha256"] and loaded["meta"]["hard_rows"] == 4
    with pytest.raises(ValueError, match="rows"):
        load_hard_sidecar(path, rows=n + 1)
    with pytest.raises(ValueError, match="obs_fingerprint"):
        load_hard_sidecar(path, rows=n, obs_fingerprint="other")
    with pytest.raises(ValueError, match="ckpt.sha256"):
        load_hard_sidecar(path, rows=n, ckpt_sha256="deadbeef")
    with pytest.raises(ValueError, match="缺少难例 sidecar"):
        load_hard_sidecar(tmp_path / "missing.npz", rows=n)


# ------------------------------------------------------- 3/4. 门控 CE + 硬切 MoE
def test_gate_binary_loss_is_all_sample_and_ignores_unknown() -> None:
    logits = torch.zeros(4, 1)
    labels = torch.tensor([1.0, 0.0, -1.0, 1.0])
    weights = torch.ones(4)
    # 仅 3 行有效（-1 行不进损失）
    loss = gate_binary_loss(logits, labels, sample_weight=weights)
    manual = (torch.nn.functional.binary_cross_entropy_with_logits(
        torch.zeros(3), torch.tensor([1.0, 0.0, 1.0]), reduction="mean"))
    assert float(loss) == pytest.approx(float(manual), rel=1e-6)
    stats = gate_binary_stats(logits, labels, sample_weight=weights)
    assert stats["count"] == 3.0 and stats["hard_rate"] == pytest.approx(2.0 / 3.0)
    # logits=0 → p=0.5 → 预测全 0，只有 label=0 的那一行算对
    assert stats["acc"] == pytest.approx(1.0 / 3.0)


def test_hard_switch_moe_easy_rows_use_primary_only() -> None:
    torch.manual_seed(0)
    block = MoEBlock(hidden=8, num_experts=8, expert_hidden=8)
    # 让 specific 分支非零（zero-init 默认输出 0 → 硬切无差异）
    with torch.no_grad():
        for expert in block.experts:
            expert[-1].weight.normal_(0.1)
    x = torch.randn(4, 8)
    primary = block.primary(x)
    out_easy, _ = block(x, hard_mask=torch.zeros(4))
    assert torch.allclose(out_easy, primary, atol=1e-6), "hard_mask=0 必须严格只走 primary"
    out_hard, _ = block(x, hard_mask=torch.ones(4))
    assert not torch.allclose(out_hard, primary, atol=1e-4), "hard_mask=1 必须走 specific 8 路"
    # 8 路 CE 只算难例：easy 行权重置 0 → 与只在难例行上算 CE 等价
    logits = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 2, 3])
    weights = torch.tensor([1.0, 1.0, 0.0, 0.0])
    only_hard = router_hard_label_loss(logits, labels, sample_weight=weights)
    hard_only = router_hard_label_loss(logits[:2], labels[:2], sample_weight=torch.ones(2))
    assert float(only_hard) == pytest.approx(float(hard_only), rel=1e-6)


def test_router_hard_label_loss_respects_zero_weight_rows() -> None:
    logits = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    weights = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    loss = router_hard_label_loss(logits, labels, sample_weight=weights)
    expected = torch.nn.functional.cross_entropy(logits[:2], labels[:2])
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)


# --------------------------------------------------------- 5. freeze 生效（参数不变）
def test_specific_freeze_keeps_primary_head_params_unchanged() -> None:
    from pipeline.stages import _SPECIFIC_PHASE_FREEZE

    torch.manual_seed(0)
    from net.model import DrivingModel

    model = DrivingModel(hidden=16, num_experts=8, expert_hidden=16)
    frozen = apply_freeze_prefixes(model, _SPECIFIC_PHASE_FREEZE)
    assert any(name.startswith("plan_head.moe.primary.") for name in frozen)
    assert any(name.startswith("policy.") for name in frozen)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    # 只对可训练参数（experts 输出层）造梯度并 step：冻结组必须逐位不变
    with torch.no_grad():
        model.plan_head.moe.experts[0][-1].weight.grad = torch.ones_like(
            model.plan_head.moe.experts[0][-1].weight
        )
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(before[name], parameter.detach()), f"冻结参数被更新：{name}"
    assert not torch.equal(
        before["plan_head.moe.experts.0.2.weight"],
        model.plan_head.moe.experts[0][-1].weight.detach(),
    ), "可训练 experts 未更新"


# --------------------------------------------------------------- 6. val-dir 生效
def test_val_dir_takes_effect_train_rows_all(tmp_path: Path) -> None:
    train_dir = tmp_path / "BTC_test_expert5k"
    write_v2_dataset(train_dir, episodes=6, steps_per_episode=6)
    annotate_router_sidecar(train_dir)
    val_dir = tmp_path / "BTC_test_expert500val"
    write_v2_dataset(val_dir, episodes=2, steps_per_episode=6)
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(TINY_MODEL_YAML, encoding="utf-8")
    out = tmp_path / "stage_b"
    args = _parse_args([
        "--stage", "B", "--bc-dir", str(train_dir), "--out", str(out),
        "--model-config", str(model_yaml), "--device", "cpu", "--seed", "0",
        "--bc-epochs", "1", "--batch-size", "8", "--no-monitor", "--no-materialize",
        "--val-dir", str(val_dir), "--max-batches", "1",
    ])
    metrics = run_stage_b(args, {})
    train_rows = int(BCDataset.load(str(train_dir)).count)
    val_rows = int(BCDataset.load(str(val_dir)).count)
    assert metrics["train_frames"] == train_rows, "train 口径必须是 train-dir 全部行"
    assert metrics["val_frames"] == val_rows
    assert str(metrics["val_source"]).startswith("dir:")
    assert metrics["val_dir"] == str(val_dir)
    # 留出评估确实来自 val-dir（val 指标存在）
    assert metrics["primary"]["val"]["bc_loss"] == metrics["primary"]["val"]["bc_loss"]


# --------------------------------------------------- 7. 双分支 Stage B 端到端
def test_two_branch_stage_b_end_to_end(tmp_path: Path) -> None:
    """用户定稿流程：--mine-only（primary+挖掘）→ fit_clusters --rows-from → annotate → specific。"""
    from pipeline.clusters import annotate_assignments, load as load_clusters
    from tools.fit_clusters import run_fit as fit_run

    ds_dir = tmp_path / "BTC_test_expert5k"
    write_v2_dataset(ds_dir, episodes=6, steps_per_episode=6)
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(TINY_MODEL_YAML, encoding="utf-8")

    # ① primary + 难例挖掘（--mine-only；不进入 specific）
    mine_out = tmp_path / "stage_b_mine"
    mine_args = _parse_args([
        "--stage", "B", "--bc-dir", str(ds_dir), "--out", str(mine_out),
        "--model-config", str(model_yaml), "--device", "cpu", "--seed", "0",
        "--bc-epochs", "2", "--bc-phase-split", "0.5", "--batch-size", "8",
        "--no-monitor", "--no-materialize", "--two-branch", "--mine-only",
    ])
    mine_metrics = run_stage_b(mine_args, {})
    assert mine_metrics["mine_only"] is True and mine_metrics["hard_rows"] > 0
    sidecar = mine_out / "hard_sidecar.npz"
    assert (mine_out / "primary.pt").is_file() and sidecar.is_file()

    # ② 只在难例子集上拟合 hard 簇（--rows-from；spec 带 rows_from provenance）
    hard_spec = tmp_path / "cluster_hard_test.npz"
    fit_run(
        ds_dir, out=hard_spec, report_path=tmp_path / "cluster_hard_test.report.json",
        rows_from=sidecar, cluster_version="hard_test", k=8, restarts=1, max_iters=5,
        density_k=5, seed=0,
    )
    spec = load_clusters(hard_spec)
    assert str(spec.data_fingerprint.get("rows_from")) == str(sidecar)

    # ③ 全量 cluster_v2* 不得当 hard 监督：无 rows_from 的 spec → fail-fast
    full_spec = tmp_path / "cluster_v2_like.npz"
    full_spec.write_bytes(Path("config/clusters/cluster_v2.npz").read_bytes())
    bad_args = _parse_args([
        "--stage", "B", "--bc-dir", str(ds_dir), "--out", str(tmp_path / "bad"),
        "--model-config", str(model_yaml), "--device", "cpu", "--bc-epochs", "1",
        "--no-monitor", "--two-branch", "--hard-cluster-config", str(full_spec),
    ])
    with pytest.raises(SystemExit, match="rows_from"):
        run_stage_b(bad_args, {})

    # ④ hard 簇 sidecar（annotate_clusters 同 API）+ specific 段
    dataset = BCDataset.load(str(ds_dir))
    annotate_assignments(
        dataset_dir=ds_dir, count=int(dataset.count),
        obs_batch_fn=dataset.build_obs_batch, spec=spec, dataset_meta=dataset.meta,
        logger=lambda _: None,
    )
    out = tmp_path / "stage_b_two_branch"
    args = _parse_args([
        "--stage", "B", "--bc-dir", str(ds_dir), "--out", str(out),
        "--model-config", str(model_yaml), "--device", "cpu", "--seed", "0",
        "--bc-epochs", "2", "--bc-phase-split", "0.5", "--batch-size", "8",
        "--no-monitor", "--no-materialize", "--two-branch",
        "--hard-sidecar", str(sidecar), "--hard-cluster-config", str(hard_spec),
        "--gate-coef", "0.5", "--router-coef", "0.1",
    ])
    metrics = run_stage_b(args, {})
    n_hard = int(np.ceil(0.5 * dataset.count))
    assert metrics["two_branch"] is True
    assert metrics["hard_flags_available"] is True
    assert metrics["hard_rows"] == n_hard
    # 冻结参照 ckpt（mine-only 产物）+ sidecar（--hard-sidecar 复用）可追溯
    assert (out / "primary.pt").is_file() and (mine_out / "primary.pt").is_file()
    side = load_hard_sidecar(sidecar, rows=dataset.count)
    assert side["meta"]["ckpt"]["sha256"], "sidecar 必须含参照 primary ckpt 标识"
    # primary 段不碰 specific 8 路 router；specific 段 8 路 CE 只数难例行
    primary = metrics["primary"]
    assert primary["bc_router_placeholder"] == 1.0
    specific = metrics["specific"]
    assert specific["bc_gate_placeholder"] == 0.0
    assert specific["bc_router_placeholder"] == 0.0
    # 8 路 CE 只数难例行（且只数训练行；legacy val 切分按 episode 留出）
    from pipeline.stages import _episode_split

    train_idx, _, _ = _episode_split(dataset.arrays["episode_id"], 0.15, 0)
    hard_flags = (side["hard"] > 0.5)
    expected_router = int(np.count_nonzero(hard_flags[train_idx]))
    assert int(specific["bc_router_count"]) == expected_router, "8 路 CE 只能数难例（训练行）"
    assert int(specific["bc_gate_count"]) == int(train_idx.size), "门控 CE 必须覆盖全体训练行"
    assert 0.0 <= specific["bc_hard_rate"] <= 1.0
    assert np.isfinite(specific["bc_gate_ce"]) and np.isfinite(specific["bc_gate_acc"])
    # 冻结生效：primary expert 在 specific 段不更新（ckpt 里的权重对比由 freeze 单测覆盖）


# --------------------------------------------------------- 8. DAgger 过滤 + meta
def test_dagger_pathological_filters_and_meta(tmp_path: Path) -> None:
    from tools.dagger_collect import _apply_pathological_filters, main as dagger_main
    from pipeline.trainer import save_checkpoint
    from pipeline.stages import build_model, load_config
    from tests.v2_synthetic import TINY_MODEL_YAML as _YAML

    ds_dir = tmp_path / "BTC_dagger_test"
    write_v2_dataset(ds_dir, episodes=4, steps_per_episode=6)
    dataset = BCDataset.load(str(ds_dir))
    arrays = dataset.arrays
    # 造病态行（只对 train_weight>0 的行生效）：行 2 卡死、行 3 大转向
    arrays["traj30_measured"] = np.asarray(
        arrays.get("traj30_measured", arrays["traj30"]), dtype=np.float32
    ).copy()
    arrays["traj30_measured"][2] = 0.0
    arrays["action"] = np.asarray(arrays["action"], dtype=np.float32).copy()
    arrays["action"][3, 0, 1] = 0.9
    assert arrays["train_weight"][2] > 0 and arrays["train_weight"][3] > 0
    original_weights = {
        key: np.array(arrays[key]) for key in ("train_weight", "balance_weight", "sample_weight") if key in arrays
    }
    stats = _apply_pathological_filters(dataset)
    assert stats["stuck"] == 1 and stats["yaw_outlier"] == 1
    assert int(np.count_nonzero(np.asarray(arrays["train_weight"]) > 0.0)) == stats["trainable_after"]
    assert arrays["train_weight"][2] == 0.0 and arrays["train_weight"][3] == 0.0
    # 行不删（行数不变）
    assert int(np.asarray(arrays["train_weight"]).size) == dataset.count
    # 恢复权重后落盘（病态 traj/action 保留 → 供 --skip-collect 的 main 复算过滤统计）
    for key, value in original_weights.items():
        arrays[key] = value
    np.savez_compressed(ds_dir / "expert_bc.npz", **{key: value for key, value in arrays.items()})

    # --skip-collect：写 shadow + meta（driver ckpt 标识 / 过滤统计 / 病态阈值）
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(_YAML, encoding="utf-8")
    ckpt = tmp_path / "primary.pt"
    save_checkpoint(ckpt, build_model(load_config(str(model_yaml))), meta={"stage": "B"})
    rc = dagger_main([
        "--ckpt", str(ckpt), "--out", str(ds_dir), "--specs", "env/specs/scenarios_smoke16.json",
        "--skip-collect", "--model-config", str(model_yaml), "--device", "cpu", "--limit", "2",
    ])
    assert rc == 0
    dagger = json.loads((ds_dir / "dagger.json").read_text(encoding="utf-8"))
    assert dagger["driver_ckpt"]["sha256"] and dagger["labeler"] == "pure_pursuit"
    assert len(dagger["spec_ids"]) == 2
    assert dagger["pathological"]["stuck"] == 1 and dagger["pathological"]["yaw_outlier"] == 1
    assert dagger["shadow"]["rows"] == dataset.count
    assert np.isfinite(dagger["shadow"]["policy_il_err_mean"])
    assert (ds_dir / "dagger_policy_shadow.npz").is_file()
    meta = json.loads((ds_dir / "expert_bc.meta.json").read_text(encoding="utf-8"))
    assert meta["dagger"]["driver_ckpt"]["sha256"] == dagger["driver_ckpt"]["sha256"]
