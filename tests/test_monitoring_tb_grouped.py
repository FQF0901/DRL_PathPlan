"""tensorboard 多线成图回归（2026-09-26）：canonical tag 自动归类 → ``add_scalars``。

需求：同一信号家族的 ``h1..h6`` / ``brake|turn|curve`` / ``e0..e7`` / loss 项在
tensorboard 里画在一张图内（与 ``tools/plot_curves.py`` 的 PNG 面板一致），
而不是散成多个独立 tag。

覆盖：
- 归类规则（train/val 各一图；``val/`` 族 main 加 ``val_`` 前缀）；
- 同族 <2 个 sub 不写（单点噪声）+ NaN/非族 tag 忽略；
- CSV 长表语义不变（只多 TB 事件，不加 CSV tag）；
- 真实 event 文件读回，重建 ``main_tag → subs`` 多线记录（EventAccumulator）。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pipeline.monitoring import TrainingMonitor, _grouped_scalars


def _ones(*tags: str) -> dict:
    return {tag: 1.0 for tag in tags}


# ---------------------------------------------------------------- 归类规则（helper）
def test_grouped_scalars_classification_all_families() -> None:
    scalars: dict = {}
    # Stage A：wm_terms / grad_norm / health
    scalars.update(_ones(
        "train/wm_loss", "train/wm_loss_od", "train/wm_loss_ego_next",
        "train/presence_loss", "train/entry_loss",
        "train/grad_norm_plan_head", "train/grad_norm_router", "train/grad_norm_st_gnn",
        "train/presence_auc", "train/entry_auc",
    ))
    # Stage B：loss 项 + 专家混合权重（primary/specific；train/val 各一张图）
    for phase in ("primary", "specific"):
        for term in ("loss", "traj_loss", "action_loss", "router_loss"):
            scalars[f"train/{phase}_bc_{term}"] = 1.0
            scalars[f"val/{phase}_bc_{term}"] = 1.0
        for index in range(8):
            scalars[f"train/{phase}_bc_router_expert_mix_weight_{index}"] = 1.0
            scalars[f"val/{phase}_bc_router_expert_mix_weight_{index}"] = 1.0
    # 分组窗口：逐 horizon / 分切片 / 逐标签（train + val）
    for k in range(1, 7):
        for metric in ("loss", "ade", "cv_ade", "ego_next_loss", "traj_mse_m2", "traj_mae_m"):
            scalars[f"horizon/h{k}/{metric}/mean"] = float(k)
            scalars[f"val/horizon/h{k}/{metric}/mean"] = float(k)
    for name in ("brake", "turn", "curve"):
        scalars[f"slice/{name}/action_err/mean"] = 0.5
        scalars[f"val/slice/{name}/action_err/mean"] = 0.5
    for name in ("cutin_active", "on_curve"):
        scalars[f"label/{name}/action_err/mean"] = 0.5
        scalars[f"val/label/{name}/action_err/mean"] = 0.5

    groups = _grouped_scalars(scalars)

    assert groups["wm_terms"] == {
        "wm_loss": 1.0, "wm_loss_od": 1.0, "wm_loss_ego_next": 1.0,
        "presence_loss": 1.0, "entry_loss": 1.0,
    }
    assert groups["grad_norm"] == {"plan_head": 1.0, "router": 1.0, "st_gnn": 1.0}
    assert groups["health"] == {"presence_auc": 1.0, "entry_auc": 1.0}
    assert groups["bc_terms_primary"] == {"loss": 1.0, "traj": 1.0, "action": 1.0, "router": 1.0}
    assert groups["bc_terms_specific"] == {"loss": 1.0, "traj": 1.0, "action": 1.0, "router": 1.0}
    assert groups["val_bc_terms_primary"] == {"loss": 1.0, "traj": 1.0, "action": 1.0, "router": 1.0}
    assert groups["val_bc_terms_specific"] == {"loss": 1.0, "traj": 1.0, "action": 1.0, "router": 1.0}
    for phase in ("primary", "specific"):
        assert groups[f"expert_mix_weight_{phase}"] == {f"e{i}": 1.0 for i in range(8)}
        assert groups[f"val_expert_mix_weight_{phase}"] == {f"e{i}": 1.0 for i in range(8)}
    expected_horizon = {f"h{k}": float(k) for k in range(1, 7)}
    horizon_mains = {
        "loss": "horizon_loss",
        "ade": "horizon_ade",
        "cv_ade": "horizon_cv_ade",
        "ego_next_loss": "horizon_ego_next",
        "traj_mse_m2": "horizon_traj_mse_m2",
        "traj_mae_m": "horizon_traj_mae_m",
    }
    for main in horizon_mains.values():
        assert groups[main] == expected_horizon
        assert groups[f"val_{main}"] == expected_horizon
    assert groups["slice_action_err"] == {"brake": 0.5, "turn": 0.5, "curve": 0.5}
    assert groups["val_slice_action_err"] == {"brake": 0.5, "turn": 0.5, "curve": 0.5}
    assert groups["label_action_err"] == {"cutin_active": 0.5, "on_curve": 0.5}
    assert groups["val_label_action_err"] == {"cutin_active": 0.5, "on_curve": 0.5}
    # canonical tag 本身不参与分组（不会自成就一个 1-sub 族）
    assert not any(main.startswith(("horizon/", "slice/", "label/", "val/")) for main in groups)


def test_grouped_scalars_min_two_rule_and_exclusions() -> None:
    """同族 <2 个 sub / NaN / 未列入族的指标 → 不写（避免单点噪声）。"""
    scalars = {
        # 未列入族的 horizon 指标（fde / legacy traj_err / 物理计数）
        "horizon/h1/fde/mean": 1.0,
        "horizon/h1/traj_err/mean": 1.0,
        "horizon/h1/valid_samples/mean": 1.0,
        "horizon/h1/valid_weight_sum/mean": 1.0,
        "horizon/h1/slot_count/mean": 1.0,
        # 只有 1 个 sub 的族
        "label/only_one/action_err/mean": 1.0,
        "slice/only_one/action_err/mean": 1.0,
        "train/grad_norm_sole": 1.0,
        "train/wm_loss": 1.0,
        "val/horizon/h1/traj_mae_m/mean": 1.0,
        # n_updates（不是 /mean）与标签名里的 _count 都不参与
        "label/on_curve/action_err/n_updates": 1.0,
        "label/on_curve_count/action_err/n_updates": 1.0,
        # 其他前缀/诊断序列不参与分组
        "train/primary_bc_traj_mse": 1.0,
        "train/reward/total": 1.0,
        "train/grad_norms_first_batch/plan_head": 1.0,
        "moe/effective_n": 1.0,
        "kpi/success": 1.0,
        "scene_label/cutin_active/freq": 1.0,
    }
    assert _grouped_scalars(scalars) == {}

    # NaN 静默跳过：h2/loss 有值但 h1 为 NaN → 有效 sub 只剩 1 个 → 整族跳过
    assert _grouped_scalars({
        "horizon/h1/loss/mean": float("nan"),
        "horizon/h2/loss/mean": 1.0,
        "horizon/h1/ade/mean": float("inf"),
    }) == {}

    # 达到 2 个 sub 即写；非标量值不参与
    groups = _grouped_scalars({"horizon/h1/loss/mean": 1.0, "horizon/h2/loss/mean": None})
    assert groups == {}
    groups = _grouped_scalars({"horizon/h1/loss/mean": 1.0, "horizon/h2/loss/mean": 2.0})
    assert groups == {"horizon_loss": {"h1": 1.0, "h2": 2.0}}


# ---------------------------------------------------------------- flush 端到端（假 writer）
class _RecordingWriter:
    """记录 ``add_scalar`` / ``add_scalars`` 调用的假 SummaryWriter。"""

    def __init__(self) -> None:
        self.calls: list = []
        self.scalars: list = []
        self.flushes = 0

    def add_scalar(self, tag, value, global_step=None) -> None:
        self.scalars.append((str(tag), value, global_step))

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None) -> None:
        self.calls.append((str(main_tag), dict(tag_scalar_dict), global_step))

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        pass


def _read_csv_tags(path: Path) -> dict:
    tags: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tags[row["tag"]] = float(row["value"])
    return tags


def test_flush_adds_grouped_tensorboard_without_touching_csv(tmp_path: Path) -> None:
    monitor = TrainingMonitor(str(tmp_path), tensorboard=False, csv=True)
    fake = _RecordingWriter()
    monitor._writer = fake  # 注入假 writer（构造时 tensorboard=False）

    monitor.on_train_step(
        {
            "wm_loss": 1.0, "wm_loss_od": 0.5, "wm_loss_ego_next": 0.25,
            "presence_loss": 0.1, "entry_loss": 0.2,
            "grad_norm_plan_head": 1.0, "grad_norm_router": 2.0, "grad_norm_st_gnn": 3.0,
            "presence_auc": 0.9, "entry_auc": 0.8,
            "primary_bc_loss": 0.4, "primary_bc_traj_loss": 0.3,
            "primary_bc_action_loss": 0.2, "primary_bc_router_loss": 0.1,
            "primary_bc_router_expert_mix_weight_0": 0.5,
            "primary_bc_router_expert_mix_weight_1": 0.5,
            "primary_bc_router_expert_mix_weight_2": float("nan"),  # NaN 不参与
        },
        step=1,
    )
    monitor.on_grouped_step(
        # h3 为 None（GroupedMetricStatistics 静默跳过）→ horizon_loss 只剩 h1/h2
        horizon={"h1": {"loss": 1.0}, "h2": {"loss": 2.0}, "h3": {"loss": None}},
        slices={"brake": {"action_err": 0.2}, "turn": {"action_err": 0.3}},
        labels={"on_curve": {"action_err": 0.4}},
        step=1,
    )
    monitor.on_val_step({"primary_bc_loss": 0.5, "primary_bc_traj_loss": 0.25}, step=1)
    monitor.on_val_grouped_step(
        horizon={"h1": {"traj_mae_m": 0.7}, "h2": {"traj_mae_m": 0.8}},
        step=1,
    )
    monitor.flush(step=1)

    grouped = {main: subs for main, subs, _ in fake.calls}
    assert grouped["wm_terms"] == {
        "wm_loss": 1.0, "wm_loss_od": 0.5, "wm_loss_ego_next": 0.25,
        "presence_loss": 0.1, "entry_loss": 0.2,
    }
    assert grouped["grad_norm"] == {"plan_head": 1.0, "router": 2.0, "st_gnn": 3.0}
    assert grouped["health"] == {"presence_auc": 0.9, "entry_auc": 0.8}
    assert grouped["bc_terms_primary"] == {"loss": 0.4, "traj": 0.3, "action": 0.2, "router": 0.1}
    assert grouped["expert_mix_weight_primary"] == {"e0": 0.5, "e1": 0.5}
    assert grouped["horizon_loss"] == {"h1": 1.0, "h2": 2.0}
    assert grouped["slice_action_err"] == {"brake": 0.2, "turn": 0.3}
    assert grouped["val_bc_terms_primary"] == {"loss": 0.5, "traj": 0.25}
    assert grouped["val_horizon_traj_mae_m"] == {"h1": 0.7, "h2": 0.8}
    # 单 sub 族不写：label_action_err（on_curve 一个）/ val_slice_action_err（无）
    assert "label_action_err" not in grouped
    assert "val_slice_action_err" not in grouped
    # 所有分组点都写在同一 step
    assert {step for _, _, step in fake.calls} == {1}
    assert fake.flushes == 1

    # 二次 flush（无新记录）→ 不重复写
    monitor.flush(step=1)
    assert len(fake.calls) == 9

    # CSV 长表语义不变：canonical tag 在、分组 main_tag 不在
    tags = _read_csv_tags(tmp_path / "metrics.csv")
    for tag in ("train/wm_loss", "train/grad_norm_router", "train/presence_auc",
                "train/primary_bc_loss", "train/primary_bc_router_expert_mix_weight_0",
                "horizon/h1/loss/mean", "slice/brake/action_err/mean",
                "val/horizon/h1/traj_mae_m/mean", "val/primary_bc_loss"):
        assert tag in tags, f"CSV 既有序列缺失：{tag}"
    assert not [tag for tag in tags if tag.startswith(("wm_terms", "horizon_loss", "slice_action_err",
                                                       "label_action_err", "bc_terms_", "expert_mix_weight_",
                                                       "val_bc_terms_", "val_horizon_"))]
    monitor.close()


# ---------------------------------------------------------------- 事件读回（真实 writer）
def _read_run(run_dir: Path) -> dict:
    """读回单个 run 目录的 scalar 事件：``{tag: [(step, value), ...]}``。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def test_summary_events_expose_grouped_multiline_tags(tmp_path: Path) -> None:
    """真实 event 读回：main_tag 在多个 sub-run 中各有数据点 = tensorboard 一张图多条线。"""
    monitor = TrainingMonitor(str(tmp_path), tensorboard=True, csv=True)
    if monitor._writer is None:
        pytest.skip("tensorboard 不可用")
    monitor.on_grouped_step(
        horizon={f"h{k}": {"traj_mae_m": 0.1 * k} for k in range(1, 7)},
        slices={"brake": {"action_err": 0.2}, "turn": {"action_err": 0.3}},
        labels={"only_one": {"action_err": 0.4}},  # 单 sub → 不成图
        step=3,
    )
    monitor.log_scalar("horizon/h1/fde/mean", 9.0, step=3)  # 未列入族 → 不成图
    monitor.flush(step=3)
    monitor.close()

    # 主 run：canonical tag 仍在（增量而非替换）；未列入族的 / 单 sub 族不写分组点
    top = _read_run(tmp_path)
    assert top["horizon/h1/traj_mae_m/mean"][0][0] == 3
    assert top["horizon/h1/traj_mae_m/mean"][0][1] == pytest.approx(0.1)
    assert "horizon/h6/traj_mae_m/mean" in top
    assert "slice/brake/action_err/mean" in top
    assert "horizon_fde/h1" not in top and "label_action_err/only_one" not in top

    # 分组：main_tag → {sub: value}（torch add_scalars：每 sub 一个 sub-run，文件内 tag = main_tag）
    grouped: dict[str, dict[str, float]] = {}
    for directory in sorted(path for path in tmp_path.iterdir() if path.is_dir()):
        for tag, points in _read_run(directory).items():
            sub = directory.name[len(tag) + 1:] if directory.name.startswith(f"{tag}_") else directory.name
            assert len(points) == 1 and points[0][0] == 3, (directory, tag, points)
            grouped.setdefault(tag, {})[sub] = points[0][1]

    assert grouped["horizon_traj_mae_m"] == pytest.approx({f"h{k}": 0.1 * k for k in range(1, 7)})
    assert grouped["slice_action_err"] == {"brake": pytest.approx(0.2), "turn": pytest.approx(0.3)}
    assert "label_action_err" not in grouped
    assert "horizon_fde" not in grouped
