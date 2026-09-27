"""Stage B 监控/留出集回归（2026-09-26；瘦身口径 2026-09-27）：逐 epoch step 轴、留出、Tier-1 tag。

覆盖交付项：

- Stage B 的 monitor 逐 epoch flush（``step = phase_offset + epoch``，primary/specific
  全局单调），不再两相位撞点丢点；
- Stage B 按 episode 留出（与 Stage A 同 ``seed/val_frac`` → 同一批留出 episode），
  每 epoch 末 ``evaluate_bc`` 产出同族 val 指标，epoch 行打印 ``val=``；
- 监控密集口径（默认瘦身）：``stageB/<phase>/loss_terms`` / ``ego/traj/mae_m`` /
  ``ego/action/err_weighted`` / ``router/*``（留出 ``val_*``）；旧 tag、slice/label、
  ``n_updates`` 不再写；``--monitor-legacy-tags`` 回退旧全量 tag。
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from net.model import DrivingModel
from pipeline.stages import _episode_split, _parse_args, run_stage_b
from pipeline.trainer import BCConfig, BCDataset, evaluate_bc, pretrain_bc
from tests.v2_synthetic import TINY_MODEL_YAML, write_v2_dataset

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _dataset(tmp_path: Path, *, episodes: int = 6, steps_per_episode: int = 6) -> BCDataset:
    directory = write_v2_dataset(tmp_path / "bc_v2", episodes=episodes, steps_per_episode=steps_per_episode)
    return BCDataset.load(str(directory))


def _model_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(TINY_MODEL_YAML, encoding="utf-8")
    return path


def _csv_series(directory: Path) -> dict:
    series: dict[str, dict[int, float]] = {}
    with (directory / "monitor" / "metrics.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            series.setdefault(row["tag"], {})[int(row["step"])] = float(row["value"])
    return series


# ---------------------------------------------------------------------- 留出切分
def test_episode_split_deterministic_and_shared_across_stages() -> None:
    """同 ``seed/val_frac`` → A/B 同一批留出 episode；train/val 行索引完整互斥。"""
    episode_ids = np.repeat(np.arange(20, dtype=np.int64), 5)
    train_a, val_a, eps_a = _episode_split(episode_ids, 0.2, seed=3)
    train_b, val_b, eps_b = _episode_split(episode_ids, 0.2, seed=3)
    assert np.array_equal(eps_a, eps_b), "Stage A/B 必须留出同一批 episode"
    assert np.array_equal(train_a, train_b) and np.array_equal(val_a, val_b)
    assert sorted(np.concatenate([train_a, val_a]).tolist()) == list(range(episode_ids.size))
    assert not set(episode_ids[train_a].tolist()) & set(episode_ids[val_a].tolist())
    assert eps_a.size == 4  # round(0.2·20)


# ------------------------------------------------------------ pretrain_bc 回调/留出
def test_pretrain_bc_epoch_callback_train_increment_and_val_holdout(tmp_path: Path) -> None:
    """逐 epoch 回调：训练增量含切片/标签/router/逐 horizon；val 是真留出（计数=val 行数）。"""
    torch.manual_seed(0)
    dataset = _dataset(tmp_path)
    model = DrivingModel(hidden=16, num_experts=8, expert_hidden=16, wm_steps=6)
    train_idx, val_idx, val_episodes = _episode_split(dataset.arrays["episode_id"], 0.34, seed=0)
    assert val_idx.size > 0 and train_idx.size > 0

    calls: list = []
    config = BCConfig(
        epochs=2, batch_size=8, lr=1e-3, device="cpu", shuffle=False,
        moe_enabled=True, load_balance_coef=0.01,
        train_indices=train_idx, val_indices=val_idx,
        epoch_callback=lambda epoch, train_metrics, val_metrics: calls.append(
            (epoch, dict(train_metrics), dict(val_metrics))
        ),
    )
    metrics = pretrain_bc(model, dataset, config, logger=lambda _: None)

    assert [call[0] for call in calls] == [0, 1], "每 epoch 必须回调一次"
    train_metrics, val_metrics = calls[0][1], calls[0][2]
    # 训练增量：与全阶段汇总同族（动作/轨迹/MoE 负载 + 关键切分）
    for key in (
        "bc_loss", "bc_traj_loss", "bc_action_loss", "bc_load_balance_loss",
        "bc_action_err_weighted_mean", "bc_action_err_slice_brake_weighted_mean",
        "bc_action_err_label_on_curve", "bc_traj_mse_h1", "bc_traj_mae_h1_m",
        "bc_traj_err_h6", "bc_traj_fde_m", "bc_expert_load_0", "bc_load_cv",
        "bc_gate_entropy",
    ):
        assert key in train_metrics, f"训练 epoch 增量缺少 {key}"
    # val：同族标量 + 分组（per_horizon/slices/labels）
    for key in ("bc_loss", "bc_traj_mse", "bc_traj_mae_m", "bc_traj_fde_m",
                "bc_action_err_weighted_mean", "bc_load_cv", "bc_gate_entropy",
                "bc_action_err_count"):
        assert key in val_metrics, f"val 指标缺少 {key}"
    assert set(val_metrics["per_horizon"]) == {f"h{k}" for k in range(1, 7)}
    assert "brake" in val_metrics["slices"] and "on_curve" in val_metrics["labels"]
    # 真留出：统计计数 = val 行数（训练增量计数 = train 行数），且两集行索引互斥
    assert train_metrics["bc_action_err_count"] == float(train_idx.size)
    assert val_metrics["bc_action_err_count"] == float(val_idx.size)
    assert metrics["val"]["bc_action_err_count"] == float(val_idx.size)
    assert not (set(train_idx.tolist()) & set(val_idx.tolist()))
    assert metrics["val"]["bc_loss"] != train_metrics["bc_loss"]


def test_evaluate_bc_deterministic_and_side_effect_free(tmp_path: Path) -> None:
    """``evaluate_bc`` 无梯度、不改参数、同输入两次结果一致（evaluate 置 eval 且不恢复）。"""
    torch.manual_seed(0)
    dataset = _dataset(tmp_path)
    model = DrivingModel(hidden=16, num_experts=8, expert_hidden=16, wm_steps=6)
    config = BCConfig(epochs=1, batch_size=8, device="cpu", shuffle=False)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    indices = np.arange(8, dtype=np.int64)
    first = evaluate_bc(model, dataset, config, indices)
    second = evaluate_bc(model, dataset, config, indices)
    assert first["bc_loss"] == pytest.approx(second["bc_loss"], rel=1e-9)
    assert first["bc_traj_mse"] == pytest.approx(second["bc_traj_mse"], rel=1e-9)
    assert first["bc_action_err_count"] == 8.0
    for name, value in model.state_dict().items():
        assert torch.equal(before[name], value), f"evaluate_bc 修改了参数 {name}"
    assert model.training is False, "evaluate_bc 应把模型置为 eval（调用方负责恢复）"


# ------------------------------------------------------ Stage B 端到端 step 轴/命名
def test_stage_b_monitor_epoch_steps_and_val_family(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """BC_EPOCHS=4（2 primary + 2 specific）→ monitor step 1..4 单调，val 与 train 同族。"""
    dataset_dir = write_v2_dataset(tmp_path / "bc_v2", episodes=6, steps_per_episode=6)
    out_dir = tmp_path / "stage_b"
    args = _parse_args([
        "--stage", "B", "--bc-dir", str(dataset_dir), "--out", str(out_dir),
        "--ckpt", str(tmp_path / "missing_stage_a.pt"),  # 不读仓库既有 ckpt，保持测试自洽
        "--model-config", str(_model_cfg(tmp_path)),
        "--bc-epochs", "4", "--bc-phase-split", "0.5", "--val-frac", "0.34",
        "--batch-size", "8", "--device", "cpu", "--seed", "0", "--monitor",
        "--load-balance-coef", "0.01",
    ])
    metrics = run_stage_b(args, {})
    assert metrics["primary_epochs"] == 2 and metrics["specific_epochs"] == 2
    assert metrics["train_frames"] + metrics["val_frames"] == 36
    assert metrics["val_episodes"] > 0

    series = _csv_series(out_dir)
    # 逐 epoch step 轴：primary 1..2、specific 3..4（全局单调；step 0 = 阶段元数据）
    assert sorted(series["loss/planner/primary/total"]) == [1, 2]
    assert sorted(series["loss/planner/specific/total"]) == [3, 4]
    assert sorted(series["val/loss/planner/primary/total"]) == [1, 2]
    assert sorted(series["val/loss/planner/specific/total"]) == [3, 4]
    # 逐 horizon 族每相位每 epoch 一点（train 4 点 + val 4 点）
    assert sorted(series["ego/traj/mae_m/h1"]) == [1, 2, 3, 4]
    assert sorted(series["val/ego/traj/mae_m/h1"]) == [1, 2, 3, 4]
    # 真留出：val 与 train 数值不同（同 step 同族指标）
    for train_tag, val_tag in (
        ("loss/planner/primary/total", "val/loss/planner/primary/total"),
        ("ego/traj/mae_m/h1", "val/ego/traj/mae_m/h1"),
        ("ego/action/err_weighted", "val/ego/action/err_weighted"),
        ("ego/traj/fde_m", "val/ego/traj/fde_m"),
    ):
        for step in (1, 2):
            assert series[val_tag][step] != series[train_tag][step], (train_tag, step)
    # Tier-1 保留：MoE 负载 KPI + 动作误差主口径 + 末点 FDE
    for tag in ("router/expert_load/e0", "router/load_cv", "router/gate_entropy",
                "loss/planner/specific/load_balance",
                "ego/action/err_weighted", "ego/traj/fde_m"):
        assert tag in series, f"保留 tag 缺失：{tag}"
    # 去聚类：旧 cluster/gate tag 一个不留
    assert not [tag for tag in series if tag.startswith(("router/cluster/", "router/gate/"))]
    # 瘦身：旧 tag 族 / 软目标 router / n_updates / count / slice / label 一个不留
    assert not [tag for tag in series
                if tag.startswith(("horizon/", "slice/", "label/", "train/",
                                   "val/horizon/", "val/slice/", "val/label/",
                                   "planner/", "stageB/", "val_planner/", "val_stageB/"))]
    assert not [tag for tag in series
                if "soft_" in tag or "expert_mix" in tag or tag.endswith(("/count", "/n_updates"))]
    # epoch 行打印 val=
    captured = capsys.readouterr().out
    assert "val=" in captured, "epoch 行必须打印 val= 摘要"


def test_stage_b_monitor_legacy_tags_flag_restores_old_csv(tmp_path: Path) -> None:
    """``--monitor-legacy-tags``（回退）：旧 canonical tag 全量落 CSV。"""
    dataset_dir = write_v2_dataset(tmp_path / "bc_v2", episodes=4, steps_per_episode=6)
    out_dir = tmp_path / "stage_b_legacy"
    args = _parse_args([
        "--stage", "B", "--bc-dir", str(dataset_dir), "--out", str(out_dir),
        "--ckpt", str(tmp_path / "missing_stage_a.pt"),
        "--model-config", str(_model_cfg(tmp_path)),
        "--bc-epochs", "1", "--val-frac", "0.34",
        "--batch-size", "8", "--device", "cpu", "--seed", "0",
        "--monitor", "--monitor-legacy-tags",
    ])
    run_stage_b(args, {})
    series = _csv_series(out_dir)
    for tag in ("train/primary_bc_loss", "train/primary_bc_action_err_median",
                "horizon/h1/traj_mae_m/mean", "val/horizon/h1/traj_mae_m/mean",
                "slice/brake/action_err/mean"):
        assert tag in series, f"legacy tag 缺失：{tag}"
    assert "loss/planner/primary/total" not in series and "val/loss/planner/primary/total" not in series
