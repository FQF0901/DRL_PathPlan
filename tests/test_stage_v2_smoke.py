"""Stage A/B 小样本冒烟（schema v2 + 新 net 契约 + Tier-1 监控 tag）。

覆盖交付项：

- Stage A：多步直接监督（LD 移除）、``train_weight × wm_valid`` 加权、逐 horizon
  loss/ADE + 匀速基线（``wm/od/*``）、presence/entry BCE + AUC、Tier-1 监控落盘；
- Stage B：首步动作损失 + 6 点轨迹辅助（WM detach）+ router 软目标，动作加权误差
  （``ego/action/err_weighted``）+ 逐 horizon ego 轨迹 MAE（``ego/traj/mae_m``）+
  router KPI；median/p95/slice/label 已按监控瘦身移除（``docs/metrics.md``）。

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

    # 逐 horizon 损失/ADE/FDE + 每 horizon 有效样本数（物理计数 = valid_samples/valid_weight_sum，
    # 与分组窗口的 n_updates 区分）
    per_horizon = metrics["per_horizon"]
    assert list(per_horizon) == [f"h{k}" for k in range(1, 7)]
    for name, item in per_horizon.items():
        for key in ("loss", "model_ade", "model_fde", "cv_ade", "cv_fde",
                    "valid_samples", "valid_weight_sum", "slot_count"):
            assert key in item, f"{name} 缺少 {key}"
        assert "valid_count" not in item and "valid_weight" not in item, "旧物理计数名必须移除"
    assert any(item["valid_samples"] > 0 for item in per_horizon.values())
    assert metrics["val_valid_samples"] > 0
    assert metrics["val_valid_weight_sum"] > 0.0

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
    for tag in ("wm/loss", "wm/od/loss/h1", "wm/od/ade_m/h1", "wm/od/ade_m/cv_h1",
                "wm/ego_next/loss/h1"):
        assert tag in tags, f"Stage A 监控序列缺失：{tag}"
    if np.isfinite(metrics["presence_auc"]):
        assert "wm/presence_auc" in tags
    # 监控瘦身（docs/metrics.md）：旧族（horizon/train/slice/label）与计数/n_updates 一个不留
    assert not [tag for tag in tags if tag.startswith(("horizon/", "train/", "slice/", "label/"))]
    assert not [tag for tag in tags if tag.endswith(("/count", "/n_updates"))]
    # val 集常量（cv 基线/AUC）与逐 horizon 曲线只在保留族；本用例 epochs=1 → 恰 1 行
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
        "--val-frac", "0.34",  # 与 Stage A 同比例 → 同一批留出 episode
        "--device", "cpu", "--seed", "0", "--monitor", "--router-coef", "0.1",
    ])
    metrics = run_stage_b(args, {})
    primary = metrics["primary"]

    # 留出集：按 episode 切分（A/B 同 seed/val_frac），val 不参与训练
    assert metrics["val_episodes"] > 0 and metrics["val_frames"] > 0
    assert metrics["train_frames"] + metrics["val_frames"] == 36
    assert primary["val"]["bc_loss"] == primary["val"]["bc_loss"]  # 非 NaN
    # 动作误差：加权主口径 + 计数/权重在 metrics.json；median/p95 与全部切片已瘦身移除
    for key in ("bc_action_err_mean", "bc_action_err_weighted_mean",
                "bc_action_err_count", "bc_action_err_weight"):
        assert key in primary, f"缺少动作误差指标 {key}"
    assert primary["bc_action_err_weight"] > 0.0
    assert primary["bc_action_err_count"] == float(metrics["train_frames"])
    for key in ("bc_action_err_median", "bc_action_err_p95",
                "bc_action_err_slice_brake", "bc_action_err_slice_brake_weighted_mean",
                "bc_router_expert_util_0", "bc_router_expert_mix_util_0"):
        assert key not in primary, f"已移除指标仍在 metrics.json：{key}"

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
    # 专家混合权重保留（利用率 util 已瘦身移除）
    for index in range(8):
        assert f"bc_router_expert_weight_{index}" in primary
        assert f"bc_router_expert_util_{index}" not in primary
        assert f"bc_router_expert_mix_util_{index}" not in primary
    assert metrics["cluster_version"] == "v1" and metrics["cluster_k"] == 8
    assert metrics["cluster_soft_targets"] is True

    # 权重感知统计（Stage B）
    assert metrics["action_mu_count"] == 36.0
    assert metrics["action_mu_weight_sum"] > 0.0
    assert np.isfinite(metrics["action_mu_abs_err_weighted_mean"])
    # 权重 0 的帧不贡献误差统计分子：count 含全量、weight 仅计有效权重
    assert metrics["action_mu_abs_err_count"] == 36.0

    tags = _csv_tags(out_dir)
    for tag in ("stageB/primary/loss_terms/loss", "stageB/primary/loss_terms/traj",
                "stageB/primary/loss_terms/action", "stageB/primary/loss_terms/router",
                "ego/traj/mae_m/h1", "ego/action/err_weighted",
                "router/soft_ce", "router/soft_kl", "router/top1_cluster_acc",
                "router/entropy", "router/primary/expert_mix_weight/e0",
                # 留出集（val_ 前缀；同族指标）
                "val_stageB/primary/loss_terms/loss", "val_ego/traj/mae_m/h1",
                "val_ego/action/err_weighted", "val_router/soft_ce"):
        assert tag in tags, f"Stage B 监控序列缺失：{tag}"
    # NMI 在极小数据集上可能因簇标签单一而为 NaN（NaN 静默跳过）→ 有值才断言
    if np.isfinite(primary["bc_router_nmi"]):
        assert "router/nmi" in tags
    # 监控瘦身：旧族（horizon/train/val/slice/label）与计数/n_updates 一个不留
    assert not [tag for tag in tags if tag.startswith(("horizon/", "train/", "val/", "slice/", "label/"))]
    assert not [tag for tag in tags if tag.endswith(("/count", "/n_updates"))]
    assert (out_dir / "final.pt").exists()
