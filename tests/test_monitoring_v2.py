"""监控回归：GroupedMetricStatistics + Tier-1 瘦身落盘（2026-09-27）。

- ``GroupedMetricStatistics`` 窗口统计本身不变（mean/n_updates/weighted_mean；legacy 模式使用）；
- 新口径 monitor 落盘前重命名：既有路径（KPI/episode/train/val/分组）经 ``_slim_tag``
  过滤，只保留 ``docs/metrics.md`` 的 Tier-1 tag；
- ``legacy_tags=True`` 保留旧 tag 全量（回退）。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pipeline.monitoring import GroupedMetricStatistics, TrainingMonitor


def _read_tags(csv_path: Path) -> dict:
    tags: dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tags[row["tag"]] = float(row["value"])
    return tags


def test_grouped_statistics_weighted_and_n_updates() -> None:
    stats = GroupedMetricStatistics("horizon")
    stats.update({"h1": {"loss": 2.0}, "h2": {"loss": 4.0}}, weights={"h1": 1.0, "h2": 3.0})
    stats.update({"h1": {"loss": 6.0}})
    out = stats.flush()
    assert out["horizon/h1/loss/n_updates"] == 2.0  # 贡献次数（不是样本数）
    assert "horizon/h1/loss/count" not in out, "旧 /count 名必须移除（语义易误读为样本数）"
    assert out["horizon/h1/loss/mean"] == 4.0
    assert out["horizon/h1/loss/weighted_mean"] == 4.0  # (1·2+1·6)/2（默认权重 1）
    assert out["horizon/h2/loss/weighted_mean"] == 4.0
    assert out["horizon/h2/loss/weight"] == 3.0
    assert stats.flush() == {}, "flush 后窗口必须清空"


def test_monitor_writes_slim_tier1_series(tmp_path: Path) -> None:
    monitor = TrainingMonitor(str(tmp_path), tensorboard=False, csv=True)
    # 既有路径 1：episode KPI / train step（嵌套 dict 自动展平）→ 非 Tier-1，全部丢弃
    monitor.on_episode({"success": 1.0, "collision": 0.0}, step=1)
    monitor.on_train_step({"loss": 0.5, "reward": {"total": 1.0, "dense": 0.7}}, step=1)
    # 既有路径 2：场景标签 / MoE 窗口 → 丢弃
    monitor.on_scene_step({"cutin_active": 1.0, "crowded": 0.0})
    monitor.on_moe_step(np.array([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    # Stage A：训练损失分解 + OD KPI（逐 horizon ADE/FDE + 匀速基线）
    monitor.on_train_step({"wm_loss": 0.8, "wm_loss_od": 0.5, "wm_loss_ego_next": 0.3,
                           "wm_loss_presence": 0.05, "wm_loss_entry": 0.02,
                           "presence_auc": 0.9, "entry_auc": 0.8,
                           "ego_action_err_weighted": 0.07, "ego_traj_fde_m": 0.9,
                           "grad_norm_plan_head": 1.0, "cv_ade": 0.1}, step=1)
    monitor.on_grouped_step(
        horizon={"h1": {"loss": 1.0, "ade": 2.0, "cv_ade": 0.5, "ego_next_loss": 0.3,
                        "fde": 3.0, "cv_fde": 3.3, "valid_samples": 12.0},
                 "h2": {"loss": 3.0, "ade": 4.0, "cv_ade": 0.6, "ego_next_loss": 0.4,
                        "fde": 4.0, "cv_fde": 4.4}},
        labels={"cutin_active": {"action_err": 0.4}},
        slices={"brake": {"action_err": 0.2}, "turn": {"action_err": 0.6}},
        step=1,
    )
    monitor.on_val_grouped_step(horizon={"h1": {"traj_mae_m": 0.11}, "h2": {"traj_mae_m": 0.22}}, step=1)
    # Stage B：相位损失分解 + 逐 horizon traj MAE/FDE + 留出族 + 硬标签 router
    monitor.on_train_step(
        {"primary_bc_loss": 0.4, "primary_bc_traj_loss": 0.3, "primary_bc_action_loss": 0.2,
         "primary_bc_router_loss": 0.1, "primary_bc_action_err_weighted_mean": 0.05,
         "primary_bc_traj_fde_m": 1.3,
         "primary_bc_action_err_median": 0.01, "primary_bc_router_ce": 0.7,
         "primary_bc_router_acc": 0.62, "primary_bc_router_acc_majority": 0.31,
         "primary_bc_router_cluster_loss": 0.09, "primary_bc_gate_loss": 0.07,
         "primary_bc_gate_ce": 0.66, "primary_bc_gate_acc": 0.71, "primary_bc_hard_rate": 0.52,
         "primary_bc_router_soft_kl": 0.3, "primary_bc_router_entropy": 1.2,
         "primary_bc_router_expert_mix_weight_0": 0.5},
        step=1,
    )
    monitor.on_grouped_step(horizon={"h1": {"traj_mae_m": 0.1, "traj_mse_m2": 2.0}}, step=1)
    monitor.on_val_step({"primary_bc_loss": 0.7, "primary_bc_traj_fde_m": 1.4,
                         "primary_bc_router_ce": 0.6, "primary_bc_router_acc": 0.5,
                         "primary_bc_gate_ce": 0.61, "primary_bc_gate_acc": 0.65,
                         "primary_bc_hard_rate": 0.5}, step=1)
    monitor.on_val_grouped_step(
        horizon={"h1": {"traj_mse_m2": 1.1, "traj_mae_m": 0.8}},
        labels={"cutin_active": {"action_err": 0.3}},
        slices={"brake": {"action_err": 0.1}},
        step=1,
    )
    monitor.flush(step=1)
    monitor.close()

    tags = _read_tags(tmp_path / "metrics.csv")
    for tag in ("loss/wm", "loss/od", "loss/ego_next", "loss/presence", "loss/entry",
                "val/od/presence_auc", "val/od/entry_auc",
                "val/od/ade_m/h1", "val/od/fde_m/h1", "val/od/ade_m/cv_h1", "val/od/fde_m/cv_h1",
                "loss/planner/primary/total", "loss/planner/primary/traj",
                "loss/planner/primary/action", "loss/planner/primary/router",
                "loss/planner/primary/router_cluster", "loss/planner/primary/gate",
                "ego/action/err_weighted", "ego/traj/mae_m/h1", "ego/traj/fde_m",
                "router/cluster/ce", "router/cluster/acc",
                "router/gate/ce", "router/gate/acc", "router/gate/hard_rate",
                "val/loss/planner/primary/total", "val/ego/traj/mae_m/h1",
                "val/router/cluster/ce", "val/router/cluster/acc",
                "val/router/gate/ce", "val/router/gate/hard_rate", "val/ego/traj/fde_m"):
        assert tag in tags, f"保留 tag 缺失：{tag}"
    # 已移除 tag 一个不留（含旧命名、软目标 router、n_updates/计数/slice/label/median/grad_norm）
    for tag in ("kpi/success", "train/loss", "train/reward/total", "scene_label/cutin_active/freq",
                "moe/effective_n", "horizon/h1/loss/mean", "horizon/h1/loss/n_updates",
                "horizon/h1/valid_samples/mean", "horizon/h1/traj_mse_m2/mean",
                "label/cutin_active/action_err/mean", "slice/brake/action_err/mean",
                "train/primary_bc_action_err_median", "train/grad_norm_plan_head",
                "train/cv_ade", "val/primary_bc_loss", "val/horizon/h1/traj_mse_m2/mean",
                "val/slice/brake/action_err/mean", "val/label/cutin_active/action_err/mean",
                "train/primary_bc_traj_mse",
                "wm/loss", "val/od/loss/h1", "val/ego_next/loss/h1",
                "planner/primary/loss_terms/loss", "val/planner/primary/loss_terms/loss",
                "router/soft_ce", "router/soft_kl", "router/entropy", "router/nmi",
                "router/top1_cluster_acc", "router/primary/expert_mix_weight/e0",
                "val_ego/traj/mae_m/h1", "val_router/ce", "val_stageB/primary/loss_terms/loss",
                "wm/od/loss/h1", "wm/presence_auc", "stageB/primary/loss_terms/loss",
                "val_ego/traj/mae_m/h1", "val_router/soft_ce"):
        assert tag not in tags, f"已移除 tag 仍写入：{tag}"


def test_monitor_legacy_flag_keeps_old_tags(tmp_path: Path) -> None:
    monitor = TrainingMonitor(str(tmp_path), tensorboard=False, csv=True, legacy_tags=True)
    monitor.on_train_step({"wm_loss": 0.8, "grad_norm_router": 1.0}, step=1)
    monitor.on_grouped_step(horizon={"h1": {"loss": 1.0, "traj_mse_m2": 2.0}}, step=1)
    monitor.flush(step=1)
    monitor.close()
    tags = _read_tags(tmp_path / "metrics.csv")
    for tag in ("train/wm_loss", "train/grad_norm_router", "horizon/h1/loss/mean",
                "horizon/h1/traj_mse_m2/mean", "horizon/h1/loss/n_updates"):
        assert tag in tags, f"legacy tag 缺失：{tag}"
    assert "wm/loss" not in tags and "wm/od/loss/h1" not in tags
