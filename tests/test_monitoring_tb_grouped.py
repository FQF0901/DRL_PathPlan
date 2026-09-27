"""监控瘦身回归（2026-09-27）：Tier-1 tag 重命名 + tensorboard 同 run 多线。

需求：只保留 OD/EGO loss+KPI 与 router loss+KPI（清单 ``docs/metrics.md``）：
- 训练侧照旧喂 canonical tag，落盘前重命名（``wm/*`` / ``val/...`` / ``ego/*`` /
  ``router/*`` / ``planner/*``）；
- **未列入清单的 tag 直接丢弃**（TB/CSV 都不写）：slice/label/n_updates/计数/fde/traj_mse_m2/
  expert_util/grad_norm/kpi/moe/scene_label/计时/median/p95 ...；
- **不再用 torch ``add_scalars``**（会为每个 sub 建 ``<main>_<sub>/`` sub-run 目录）：
  多线族在主 run 内逐 sub 写 ``add_scalar("<main>/<sub>")``，TB 自动并成一张多线图；
  标量写单线 ``add_scalar``；同族 <2 个 sub 不写；
- ``legacy_tags=True`` 回退旧口径（原样落盘 + 旧多线分组）；
- 真实 event 文件读回：全部 tag 在主 run 的单个事件文件里（无 sub-run 子目录）。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pipeline.monitoring import (
    TrainingMonitor,
    _group_of_tag,
    _grouped_scalars,
    _slim_group_of_tag,
    _slim_tag,
)

#: 必须被丢弃的旧 tag（覆盖移除清单）
_REMOVED_TAGS = (
    "horizon/h1/fde/mean",
    "horizon/h1/cv_fde/mean",
    "horizon/h1/traj_mse_m2/mean",
    "horizon/h1/valid_samples/mean",
    "horizon/h1/valid_weight_sum/mean",
    "horizon/h1/slot_count/mean",
    "horizon/h1/loss/n_updates",
    "horizon/h1/loss/weighted_mean",
    "slice/brake/action_err/mean",
    "label/on_curve/action_err/mean",
    "train/primary_bc_action_err_median",
    "train/primary_bc_action_err_p95",
    "train/primary_bc_action_err_slice_brake_weighted_mean",
    "train/primary_bc_action_err_label_on_curve",
    "train/primary_bc_router_expert_util_0",
    "train/primary_bc_router_expert_mix_util_0",
    "train/primary_bc_traj_mse_h1",
    "train/grad_norm_router",
    "train/cluster_version_num",
    "train/cluster_k",
    "train/cluster_soft_targets",
    "train/update_timing_s/data_s",
    "kpi/success",
    "moe/effective_n",
    "scene_label/cutin_active/freq",
)


# ---------------------------------------------------------------- 瘦身规则（helper）
def test_slim_tag_renames_retained_and_drops_removed() -> None:
    expected = {
        "train/wm_loss": "wm/loss",
        "train/presence_auc": "val/od/presence_auc",
        "train/entry_auc": "val/od/entry_auc",
        "horizon/h1/loss/mean": "val/od/loss/h1",
        "horizon/h3/ade/mean": "val/od/ade_m/h3",
        "horizon/h2/cv_ade/mean": "val/od/ade_m/cv_h2",
        "horizon/h6/ego_next_loss/mean": "val/ego_next/loss/h6",
        "horizon/h4/traj_mae_m/mean": "ego/traj/mae_m/h4",
        "val/horizon/h2/traj_mae_m/mean": "val/ego/traj/mae_m/h2",
        "train/primary_bc_loss": "planner/primary/loss_terms/loss",
        "train/specific_bc_action_loss": "planner/specific/loss_terms/action",
        "val/primary_bc_router_loss": "val/planner/primary/loss_terms/router",
        "train/primary_bc_action_err_weighted_mean": "ego/action/err_weighted",
        "val/primary_bc_action_err_weighted_mean": "val/ego/action/err_weighted",
        "train/primary_bc_router_soft_ce": "router/soft_ce",
        "val/specific_bc_router_nmi": "val/router/nmi",
        "train/primary_bc_router_expert_mix_weight_3": "router/primary/expert_mix_weight/e3",
        "val/specific_bc_router_expert_mix_weight_7": "val/router/specific/expert_mix_weight/e7",
    }
    for old, new in expected.items():
        assert _slim_tag(old) == new, old
    for tag in _REMOVED_TAGS:
        assert _slim_tag(tag) is None, f"应丢弃：{tag}"


def test_slim_grouped_scalars_families_and_min_two_rule() -> None:
    scalars = {
        f"val/od/loss/h{k}": float(k) for k in range(1, 7)
    }
    scalars.update({f"val/od/ade_m/h{k}": float(k) for k in range(1, 7)})
    scalars.update({f"val/od/ade_m/cv_h{k}": 1.0 for k in range(1, 7)})
    scalars.update({f"val/ego/traj/mae_m/h{k}": float(k) for k in range(1, 7)})
    scalars.update({f"router/specific/expert_mix_weight/e{i}": 0.125 for i in range(8)})
    scalars.update({f"planner/primary/loss_terms/{term}": 1.0
                    for term in ("loss", "traj", "action", "router")})
    scalars["wm/loss"] = 1.0
    scalars["router/soft_ce"] = 0.5
    scalars["val/router/soft_ce"] = 0.4

    groups = _grouped_scalars(scalars)
    assert groups["val/od/loss"] == {f"h{k}": float(k) for k in range(1, 7)}
    assert groups["val/od/ade_m"] == {**{f"h{k}": float(k) for k in range(1, 7)},
                                      **{f"cv_h{k}": 1.0 for k in range(1, 7)}}
    assert groups["val/ego/traj/mae_m"] == {f"h{k}": float(k) for k in range(1, 7)}
    assert groups["router/specific/expert_mix_weight"] == {f"e{i}": 0.125 for i in range(8)}
    assert groups["planner/primary/loss_terms"] == {"loss": 1.0, "traj": 1.0,
                                                    "action": 1.0, "router": 1.0}
    # 独立标量不成族（TB 走 add_scalar）
    assert "wm/loss" not in groups and "router/soft_ce" not in groups
    assert _slim_group_of_tag("wm/loss") is None
    assert _slim_group_of_tag("router/soft_ce") is None

    # 单 sub / NaN → 不成图（避免单点噪声）
    assert _grouped_scalars({"val/od/loss/h1": 1.0}) == {}
    assert _grouped_scalars({"val/od/loss/h1": 1.0, "val/od/loss/h2": float("nan")}) == {}
    groups = _grouped_scalars({"val/od/loss/h1": 1.0, "val/od/loss/h2": 2.0})
    assert groups == {"val/od/loss": {"h1": 1.0, "h2": 2.0}}

    # legacy 规则仍可用（回退路径）
    legacy = _grouped_scalars({"horizon/h1/loss/mean": 1.0, "horizon/h2/loss/mean": 2.0},
                              group_fn=_group_of_tag)
    assert legacy == {"horizon_loss": {"h1": 1.0, "h2": 2.0}}


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


def test_flush_writes_slim_csv_and_same_run_multiline(tmp_path: Path) -> None:
    monitor = TrainingMonitor(str(tmp_path), tensorboard=False, csv=True)
    fake = _RecordingWriter()
    monitor._writer = fake  # 注入假 writer（构造时 tensorboard=False）

    monitor.on_train_step(
        {
            "wm_loss": 0.8, "wm_loss_od": 0.5, "presence_auc": 0.9, "entry_auc": 0.8,
            "grad_norm_router": 2.0, "cluster_k": 8.0, "update_timing_s": {"data_s": 0.1},
            "primary_bc_loss": 0.4, "primary_bc_traj_loss": 0.3,
            "primary_bc_action_loss": 0.2, "primary_bc_router_loss": 0.1,
            "primary_bc_action_err_weighted_mean": 0.05,
            "primary_bc_action_err_median": 0.01, "primary_bc_action_err_p95": 0.2,
            "primary_bc_router_soft_ce": 0.7, "primary_bc_router_soft_kl": 0.3,
            "primary_bc_router_top1_cluster_acc": 0.6, "primary_bc_router_nmi": 0.4,
            "primary_bc_router_entropy": 1.2,
            "primary_bc_router_expert_mix_weight_0": 0.5,
            "primary_bc_router_expert_mix_weight_1": 0.5,
            "primary_bc_router_expert_util_0": 0.9,
            "primary_bc_traj_mse_h1": 2.0,
        },
        step=1,
    )
    monitor.on_grouped_step(
        # h3 为 None（GroupedMetricStatistics 静默跳过）→ ego/traj/mae_m 只剩 h1/h2
        horizon={"h1": {"traj_mae_m": 0.1, "traj_mse_m2": 2.0}, "h2": {"traj_mae_m": 0.2},
                 "h3": {"traj_mae_m": None}},
        slices={"brake": {"action_err": 0.2}, "turn": {"action_err": 0.3}},
        labels={"on_curve": {"action_err": 0.4}},
        step=1,
    )
    monitor.on_val_step(
        {"primary_bc_loss": 0.5, "primary_bc_action_err_weighted_mean": 0.06,
         "primary_bc_router_soft_ce": 0.6},
        step=1,
    )
    monitor.on_val_grouped_step(
        horizon={"h1": {"traj_mae_m": 0.7, "traj_mse_m2": 3.0}, "h2": {"traj_mae_m": 0.8}},
        slices={"brake": {"action_err": 0.1}},
        step=1,
    )
    monitor.flush(step=1)

    # 多线族：同一 run 内逐 sub 写 add_scalar("<main>/<sub>")；不再用 add_scalars
    written = {tag: (value, step) for tag, value, step in fake.scalars}
    assert fake.calls == [], "不得调用 add_scalars（会建 sub-run 子目录）"
    assert written["planner/primary/loss_terms/loss"] == (0.4, 1)
    assert written["planner/primary/loss_terms/traj"] == (0.3, 1)
    assert written["planner/primary/loss_terms/action"] == (0.2, 1)
    assert written["planner/primary/loss_terms/router"] == (0.1, 1)
    assert written["ego/traj/mae_m/h1"] == (0.1, 1)
    assert written["ego/traj/mae_m/h2"] == (0.2, 1)
    assert written["val/ego/traj/mae_m/h1"] == (0.7, 1)
    assert written["val/ego/traj/mae_m/h2"] == (0.8, 1)
    assert written["router/primary/expert_mix_weight/e0"] == (0.5, 1)
    assert written["router/primary/expert_mix_weight/e1"] == (0.5, 1)
    assert "val/planner/primary/loss_terms/loss" not in written  # 单 sub → 不写
    # slice/label/被移除族绝不写
    assert not [tag for tag in written if "slice" in tag or "label" in tag]
    assert fake.flushes == 1

    # 标量（非多线族）走 add_scalar
    for tag in ("wm/loss", "val/od/presence_auc", "val/od/entry_auc", "ego/action/err_weighted",
                "val/ego/action/err_weighted", "router/soft_ce", "router/soft_kl",
                "router/top1_cluster_acc", "router/nmi", "router/entropy",
                "val/router/soft_ce"):
        assert (tag, 1) in {(t, s) for t, _, s in fake.scalars}, f"标量缺失：{tag}"
    # 移除 tag 绝不进 TB
    assert not [tag for tag, _, _ in fake.scalars
                if "median" in tag or "p95" in tag or "grad_norm" in tag or "expert_util" in tag]

    # 二次 flush（无新记录）→ 不重复写
    count = len(fake.scalars)
    monitor.flush(step=1)
    assert len(fake.scalars) == count

    # CSV：只有瘦身 tag；移除清单标签一个不留
    tags = _read_csv_tags(tmp_path / "metrics.csv")
    for tag in ("wm/loss", "val/od/presence_auc", "planner/primary/loss_terms/loss",
                "ego/traj/mae_m/h1", "val/ego/traj/mae_m/h1", "ego/action/err_weighted",
                "router/soft_ce", "router/primary/expert_mix_weight/e0"):
        assert tag in tags, f"CSV 保留 tag 缺失：{tag}"
    for tag in _REMOVED_TAGS:
        assert tag not in tags, f"CSV 出现已移除 tag：{tag}"
    assert not [tag for tag in tags
                if tag.startswith(("slice/", "label/", "horizon/", "train/", "kpi/", "moe/", "scene_label/"))
                or tag.startswith(("val/horizon/", "val/slice/", "val/label/"))]
    monitor.close()


def test_legacy_tags_flag_restores_old_surface(tmp_path: Path) -> None:
    """``legacy_tags=True``：旧 tag 原样落 CSV + 旧族 add_scalars（回退路径）。"""
    monitor = TrainingMonitor(str(tmp_path), tensorboard=False, csv=True, legacy_tags=True)
    fake = _RecordingWriter()
    monitor._writer = fake
    monitor.on_train_step({"wm_loss": 0.8, "wm_loss_od": 0.5, "grad_norm_router": 2.0}, step=1)
    monitor.on_grouped_step(
        horizon={f"h{k}": {"loss": float(k)} for k in range(1, 7)},
        slices={"brake": {"action_err": 0.2}, "turn": {"action_err": 0.3}},
        step=1,
    )
    monitor.flush(step=1)
    grouped = {main: subs for main, subs, _ in fake.calls}
    assert grouped["horizon_loss"] == {f"h{k}": float(k) for k in range(1, 7)}
    assert grouped["slice_action_err"] == {"brake": 0.2, "turn": 0.3}
    assert {tag for tag, _, _ in fake.scalars} >= {"train/wm_loss", "train/grad_norm_router"}
    tags = _read_csv_tags(tmp_path / "metrics.csv")
    assert "train/wm_loss" in tags and "horizon/h1/loss/mean" in tags
    assert "horizon/h1/loss/n_updates" in tags  # legacy 全量口径
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


def test_summary_events_write_multiline_tags_in_same_run(tmp_path: Path) -> None:
    """真实 event 读回：所有保留 tag（含多线族逐 sub）都落在主 run 的同一事件文件。"""
    monitor = TrainingMonitor(str(tmp_path), tensorboard=True, csv=True)
    if monitor._writer is None:
        pytest.skip("tensorboard 不可用")
    monitor.on_grouped_step(
        horizon={f"h{k}": {"traj_mae_m": 0.1 * k} for k in range(1, 7)},
        slices={"brake": {"action_err": 0.2}},
        step=3,
    )
    monitor.on_val_grouped_step(
        horizon={f"h{k}": {"traj_mae_m": 0.2 * k} for k in range(1, 7)},
        step=3,
    )
    monitor.on_train_step(
        {"wm_loss": 1.0, "wm_loss_od": 0.5, "primary_bc_loss": 0.2,
         "primary_bc_traj_loss": 0.1, "primary_bc_action_loss": 0.05,
         "primary_bc_router_loss": 0.05,
         "primary_bc_router_expert_mix_weight_0": 0.5,
         "primary_bc_router_expert_mix_weight_1": 0.5,
         "primary_bc_router_soft_ce": 0.3},
        step=3,
    )
    monitor.flush(step=3)
    monitor.close()

    # 同 run 单事件文件：多线族 tag 直接以 "<main>/<sub>" 出现；无 sub-run 子目录
    top = _read_run(tmp_path)
    assert not any(path.is_dir() for path in tmp_path.iterdir()), "不得再有 sub-run 子目录"
    assert top["wm/loss"][0] == (3, pytest.approx(1.0))
    assert top["router/soft_ce"][0] == (3, pytest.approx(0.3))
    assert [step for step, _ in top["ego/traj/mae_m/h1"]] == [3]
    assert top["ego/traj/mae_m/h1"][0][1] == pytest.approx(0.1)
    assert top["val/ego/traj/mae_m/h6"][0][1] == pytest.approx(1.2)
    assert top["planner/primary/loss_terms/loss"][0] == (3, pytest.approx(0.2))
    assert top["planner/primary/loss_terms/traj"][0] == (3, pytest.approx(0.1))
    assert top["planner/primary/loss_terms/action"][0] == (3, pytest.approx(0.05))
    assert top["planner/primary/loss_terms/router"][0] == (3, pytest.approx(0.05))
    assert top["router/primary/expert_mix_weight/e0"][0] == (3, pytest.approx(0.5))
    assert top["router/primary/expert_mix_weight/e1"][0] == (3, pytest.approx(0.5))
    # 旧 tag / 移除项不在
    assert "horizon/h1/traj_mae_m/mean" not in top
    assert "slice/brake/action_err/mean" not in top
    assert "val_ego/traj/mae_m/h1" not in top
