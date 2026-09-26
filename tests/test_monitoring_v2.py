"""监控回归：既有序列不退化 + v2 分组序列（per-horizon / per-label / 分切片）。

- 既有 42+ 序列（``train/...`` / ``kpi/...`` / ``scene_label/...`` / ``moe/...``）
  继续按原 tag 写 CSV；
- 新增 ``horizon/<h>/<m>``、``label/<name>/<m>``、``slice/<name>/<m>``
  （``mean``/``count``/``weighted_mean``）序列同时进 CSV。
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


def test_grouped_statistics_weighted_and_count() -> None:
    stats = GroupedMetricStatistics("horizon")
    stats.update({"h1": {"loss": 2.0}, "h2": {"loss": 4.0}}, weights={"h1": 1.0, "h2": 3.0})
    stats.update({"h1": {"loss": 6.0}})
    out = stats.flush()
    assert out["horizon/h1/loss/count"] == 2.0
    assert out["horizon/h1/loss/mean"] == 4.0
    assert out["horizon/h1/loss/weighted_mean"] == 4.0  # (1·2+1·6)/2（默认权重 1）
    assert out["horizon/h2/loss/weighted_mean"] == 4.0
    assert out["horizon/h2/loss/weight"] == 3.0
    assert stats.flush() == {}, "flush 后窗口必须清空"


def test_monitor_writes_legacy_and_grouped_series(tmp_path: Path) -> None:
    monitor = TrainingMonitor(str(tmp_path), tensorboard=False, csv=True)
    # 既有路径 1：episode KPI / train step（嵌套 dict 自动展平）
    monitor.on_episode({"success": 1.0, "collision": 0.0}, step=1)
    monitor.on_train_step({"loss": 0.5, "reward": {"total": 1.0, "dense": 0.7}}, step=1)
    # 既有路径 2：场景标签 / MoE 窗口
    monitor.on_scene_step({"cutin_active": 1.0, "crowded": 0.0})
    monitor.on_moe_step(np.array([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))
    # v2 分组窗口
    monitor.on_grouped_step(
        horizon={"h1": {"loss": 1.0, "ade": 2.0}, "h2": {"loss": 3.0, "ade": 4.0}},
        labels={"cutin_active": {"err": 0.4}},
        slices={"brake": {"action_err": 0.2}, "turn": {"action_err": 0.6}},
        step=1,
    )
    monitor.flush(step=1)
    monitor.close()

    tags = _read_tags(tmp_path / "metrics.csv")
    # 既有序列不退化
    for tag in ("kpi/success", "train/loss", "train/reward/total", "train/reward/dense",
                "scene_label/cutin_active/freq", "moe/effective_n"):
        assert tag in tags, f"既有序列缺失：{tag}"
    # 新增分组序列（CSV + tensorboard 同一套 tag）
    for tag in ("horizon/h1/loss/mean", "horizon/h1/loss/count", "horizon/h2/ade/mean",
                "label/cutin_active/err/mean", "slice/brake/action_err/mean",
                "slice/turn/action_err/count"):
        assert tag in tags, f"分组序列缺失：{tag}"
    # 窗口 flush 后再次 flush 不产生重复 tag（幂等 + 清空）
    monitor2 = TrainingMonitor(str(tmp_path / "second"), tensorboard=False, csv=True)
    monitor2.log_scalar("x", float("nan"), step=1)  # NaN 静默跳过
    monitor2.log_scalar("y", 3.0, step=1)
    monitor2.close()
    tags2 = _read_tags(tmp_path / "second" / "metrics.csv")
    assert "x" not in tags2 and tags2["y"] == 3.0
