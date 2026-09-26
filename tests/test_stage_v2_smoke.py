"""Stage A/B 小样本冒烟（schema v2 + 新 net 契约 + 逐 horizon/切片日志）。

覆盖交付项：

- Stage A：多步直接监督（LD 移除）、``train_weight × wm_valid`` 加权、逐 horizon
  loss/ADE/FDE/有效样本数、presence/entry BCE + AUC、监控序列落盘；
- Stage B：首步动作损失 + 6 点轨迹辅助（WM detach）+ router 软目标，动作误差
  mean/median/p95 + 分切片（急刹/急转/弯道）+ 逐 horizon ego 误差 + router 指标。

用小型 DrivingModel（hidden=16 / 8 experts）与合成 v2 数据集，CPU 数秒内完成。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from pipeline.stages import _parse_args, run_stage_a, run_stage_b
from tests.v2_synthetic import TINY_MODEL_YAML, write_v2_dataset

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _dataset(tmp_path: Path) -> Path:
    return write_v2_dataset(tmp_path / "bc_v2", episodes=6, steps_per_episode=6)


def _model_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(TINY_MODEL_YAML, encoding="utf-8")
    return path


def _csv_tags(directory: Path) -> dict:
    path = directory / "monitor" / "metrics.csv"
    assert path.exists(), f"监控 CSV 缺失：{path}"
    tags: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tags[row["tag"]] = float(row["value"])
    return tags


def test_stage_a_v2_smoke_per_horizon_presence_and_weights(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)
    out_dir = tmp_path / "stage_a"
    args = _parse_args([
        "--stage", "A", "--bc-dir", str(dataset_dir), "--out", str(out_dir),
        "--model-config", str(_model_cfg(tmp_path)), "--wm-epochs", "1",
        "--val-frac", "0.34", "--batch-size", "8", "--eval-frames", "12",
        "--plan-noise-p", "0.3", "--device", "cpu", "--seed", "0", "--monitor",
    ])
    metrics = run_stage_a(args, {})

    # 逐 horizon 损失/ADE/FDE + 每 horizon 有效样本数（count 与 weighted 双口径）
    per_horizon = metrics["per_horizon"]
    assert list(per_horizon) == [f"h{k}" for k in range(1, 7)]
    for name, item in per_horizon.items():
        for key in ("loss", "model_ade", "model_fde", "cv_ade", "cv_fde",
                    "valid_count", "valid_weight", "slot_count"):
            assert key in item, f"{name} 缺少 {key}"
    assert any(item["valid_count"] > 0 for item in per_horizon.values())
    assert metrics["val_valid_count"] > 0

    # 规格：未来 LD 损失移除；presence/entry 可用（新 net 有对应头）
    assert metrics["ld_loss"] == "removed"
    assert metrics["presence_available"] == 1.0
    assert np.isfinite(metrics["presence_auc"]) or np.isnan(metrics["presence_auc"])
    assert np.isfinite(metrics["presence_pos_rate"])
    assert 0.0 < metrics["presence_pos_rate"] < 1.0
    # 权重感知：数据集权重报告（train_weight=0 的帧计入 rows、不计入 rows_weighted）
    assert metrics["dataset/rows"] == 36.0
    assert metrics["dataset/train_weight_zero"] > 0.0  # train_weight=0 的帧存在
    assert metrics["dataset/weight_min"] == 0.0

    tags = _csv_tags(out_dir)
    for tag in ("train/wm_loss", "train/per_horizon/h1/loss", "train/per_horizon/h6/valid_count",
                "horizon/h1/loss/mean", "horizon/h1/ade/mean", "horizon/h6/valid_weight/mean"):
        assert tag in tags, f"Stage A 监控序列缺失：{tag}"
    assert (out_dir / "final.pt").exists() and (out_dir / "metrics.json").exists()


def test_stage_b_v2_smoke_slices_horizon_and_router(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)
    stage_a_dir = tmp_path / "stage_a"
    args_a = _parse_args([
        "--stage", "A", "--bc-dir", str(dataset_dir), "--out", str(stage_a_dir),
        "--model-config", str(_model_cfg(tmp_path)), "--wm-epochs", "1",
        "--batch-size", "8", "--eval-frames", "8", "--device", "cpu", "--seed", "0",
    ])
    run_stage_a(args_a, {})

    out_dir = tmp_path / "stage_b"
    args = _parse_args([
        "--stage", "B", "--bc-dir", str(dataset_dir), "--out", str(out_dir),
        "--ckpt", str(stage_a_dir / "final.pt"), "--model-config", str(_model_cfg(tmp_path)),
        "--bc-epochs", "1", "--bc-phase-split", "0.5", "--batch-size", "8",
        "--device", "cpu", "--seed", "0", "--monitor", "--router-coef", "0.1",
    ])
    metrics = run_stage_b(args, {})
    primary = metrics["primary"]

    # 动作误差 mean/median/p95 + 分切片（急刹/急转/弯道）
    for key in ("bc_action_err_mean", "bc_action_err_median", "bc_action_err_p95",
                "bc_action_err_weighted_mean", "bc_action_err_count", "bc_action_err_weight"):
        assert key in primary, f"缺少动作误差指标 {key}"
    assert primary["bc_action_err_weight"] > 0.0 and primary["bc_action_err_count"] == 36.0
    for slice_name in ("brake", "turn", "curve"):
        stats = primary[f"bc_action_err_slice_{slice_name}"]
        assert stats and stats["count"] > 0 and np.isfinite(stats["weighted_mean"]), f"切片 {slice_name} 无样本"

    # 逐 horizon 轨迹度量（B1 6 点）：加权 MSE（m²）与加权 MAE（m）双口径 + 旧 alias
    for k in range(1, 7):
        assert np.isfinite(primary[f"bc_traj_mse_h{k}"]), f"缺少逐 horizon MSE h{k}"
        assert np.isfinite(primary[f"bc_traj_mae_h{k}_m"]), f"缺少逐 horizon MAE h{k}"
        assert primary[f"bc_traj_mse_h{k}"] >= 0.0 and primary[f"bc_traj_mae_h{k}_m"] >= 0.0
        # Jensen：MAE ≤ sqrt(MSE)；旧 alias 在 l2 下等于 MSE
        assert primary[f"bc_traj_mae_h{k}_m"] ** 2 <= primary[f"bc_traj_mse_h{k}"] + 1e-9
        assert primary[f"bc_traj_err_h{k}"] == pytest.approx(primary[f"bc_traj_mse_h{k}"], rel=1e-9)

    # router 软目标（聚类 lane v1 artifact 计算 → 非占位）+ 路由指标
    assert primary["bc_router_soft_placeholder"] == 0.0
    assert primary["bc_router_cluster_version"] == "v1"
    assert primary["bc_router_cluster_k"] == 8
    assert np.isfinite(primary["bc_router_soft_ce"]) and np.isfinite(primary["bc_router_soft_kl"])
    for key in ("bc_router_top1_cluster_acc", "bc_router_nmi", "bc_router_entropy"):
        assert key in primary, f"缺少 router 指标 {key}"
    assert np.isfinite(primary["bc_router_entropy"])
    # 专家利用率（top-1 比例 / top-2 混合）与平均权重逐专家记录
    for index in range(8):
        assert f"bc_router_expert_util_{index}" in primary
        assert f"bc_router_expert_weight_{index}" in primary
        assert f"bc_router_expert_mix_util_{index}" in primary
    assert metrics["cluster_version"] == "v1" and metrics["cluster_k"] == 8
    assert metrics["cluster_soft_targets"] is True

    # 权重感知统计（Stage B）
    assert metrics["action_mu_count"] == 36.0
    assert metrics["action_mu_weight_sum"] > 0.0
    assert np.isfinite(metrics["action_mu_abs_err_weighted_mean"])
    # 权重 0 的帧不贡献误差统计分子：count 含全量、weight 仅计有效权重
    assert metrics["action_mu_abs_err_count"] == 36.0

    tags = _csv_tags(out_dir)
    for tag in ("train/primary_bc_action_err_weighted_mean", "train/primary_bc_traj_err_h1",
                "train/primary_bc_traj_mse", "train/primary_bc_traj_mae_m",
                "train/primary_bc_router_soft_ce", "train/primary_bc_router_cluster_version_num",
                "train/cluster_soft_targets",
                "horizon/h1/traj_mse_m2/mean", "horizon/h1/traj_mae_m/mean",
                "slice/brake/action_err/mean",
                "slice/turn/action_err/count", "label/on_curve/action_err/mean"):
        assert tag in tags, f"Stage B 监控序列缺失：{tag}"
    assert (out_dir / "final.pt").exists()
