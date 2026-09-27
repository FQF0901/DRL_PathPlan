"""``--limit-dataset`` 语义测试（2026-09-27）：读取阶段按完整 episode 前缀截断。

动机（2026-09-27 冒烟事故）：limit 必须**在物化之前**生效且内存 ∝ N：

- 旧实现先 ``BCDataset.load`` 全量解压（5k 数据集峰值 RSS ≈ 2.0 GB）再按行切片；
- 现在的 ``BCDataset.load(path, limit=N)`` 只流式解压 npz 各成员的**前 M 行**
  （M = :func:`episode_prefix_row_count`，完整 episode 前缀），
  物化/val 切分/权重统计都基于该子集，且**列语义不变**（只减行不改值）；
- CPU 冒烟（limit 设置）时训练侧 batch 收敛到 host 安全上限（见
  ``pipeline.stages._apply_smoke_batch_caps``），避免 macro 1024/micro 256 的 host
  前向图峰值（实测 ≈ 10.4 GB）挤爆机器。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pipeline.trainer as trainer_module
from pipeline.stages import (
    _apply_smoke_batch_caps,
    _parse_args,
    _smoke_cpu_guard,
    run_stage_a,
    run_stage_b,
)
from pipeline.trainer import BCDataset, MaterializedBCDataset, episode_prefix_row_count
from tests.v2_synthetic import TINY_MODEL_YAML, annotate_router_sidecar, make_v2_arrays, write_v2_dataset

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _dataset_dir(tmp_path: Path) -> Path:
    # 6 个 episode × 6 行 = 36 行；episode 边界 = [6, 12, 18, 24, 30]
    directory = write_v2_dataset(tmp_path / "bc_v2", episodes=6, steps_per_episode=6)
    annotate_router_sidecar(directory)  # Stage B 只读 sidecar（lane B ①）
    return directory


# --------------------------------------------------------------------------- #
# 截断口径（纯函数）
# --------------------------------------------------------------------------- #

def test_episode_prefix_row_count_boundaries() -> None:
    ids = np.array([0] * 100 + [1] * 50 + [2] * 80, dtype=np.int64)  # 边界 100/150
    assert episode_prefix_row_count(ids, 100) == 100  # 恰在边界
    assert episode_prefix_row_count(ids, 120) == 100
    assert episode_prefix_row_count(ids, 150) == 150
    assert episode_prefix_row_count(ids, 200) == 150  # 末集不完整 → 不含
    assert episode_prefix_row_count(ids, 99) == 100  # 首 episode 超限 → 保留完整首集
    assert episode_prefix_row_count(ids, 230) == 230
    assert episode_prefix_row_count(ids, 10_000) == 230
    assert episode_prefix_row_count(np.zeros(0, dtype=np.int64), 10) == 0
    # 单个 episode 长于 limit → 保留整集（冒烟子集非空）
    assert episode_prefix_row_count(np.zeros(50, dtype=np.int64), 10) == 50


# --------------------------------------------------------------------------- #
# 读取阶段前缀加载
# --------------------------------------------------------------------------- #

def test_load_limit_reads_episode_prefix_and_preserves_columns(tmp_path: Path) -> None:
    dataset_dir = _dataset_dir(tmp_path)
    full = BCDataset.load(str(dataset_dir))
    limited = BCDataset.load(str(dataset_dir), limit=20)
    assert limited.count == 18  # 3 个完整 episode（6 行/集）
    assert list(limited.arrays) == list(full.arrays)
    for key, value in full.arrays.items():
        assert value.dtype == limited.arrays[key].dtype, key
        assert limited.arrays[key].shape == value[:18].shape, key
        np.testing.assert_array_equal(limited.arrays[key], value[:18], err_msg=key)
    # 列语义不变形：权重/有效性原样（只截行；hist_valid 在合成 fixture 中缺省）
    for key in ("wm_valid", "train_weight", "balance_weight", "sample_weight", "od_id", "od_presence"):
        np.testing.assert_array_equal(limited.arrays[key], full.arrays[key][:18])
    assert limited.meta == full.meta
    assert limited.label_names == full.label_names
    assert limited.schema_version == full.schema_version
    # 边界规则
    assert BCDataset.load(str(dataset_dir), limit=4).count == 6  # 首集超限 → 保留整集
    assert BCDataset.load(str(dataset_dir), limit=999).count == 36
    assert BCDataset.load(str(dataset_dir), limit=None).count == 36


def test_load_limit_does_not_full_load_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """前缀路径不得走 ``np.load``（全量解压）；npz 可用时必须流式读前缀。"""
    dataset_dir = _dataset_dir(tmp_path)

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("limit 路径不应全量加载 npz（np.load 被调用）")

    monkeypatch.setattr(trainer_module.np, "load", _boom)
    limited = BCDataset.load(str(dataset_dir), limit=20)
    assert limited.count == 18


def test_load_limit_falls_back_for_unreadable_prefix(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """非前缀可读（如 episode_id 非整型）→ 全量加载 + 同口径内存截断，并打印警告。"""
    arrays, _ = make_v2_arrays(episodes=6, steps_per_episode=6)
    arrays["episode_id"] = arrays["episode_id"].astype(np.float64)
    dataset_dir = tmp_path / "bc_float_episode"
    dataset_dir.mkdir()
    np.savez_compressed(dataset_dir / "expert_bc.npz", **arrays)
    (dataset_dir / "expert_bc.meta.json").write_text(
        '{"schema_version": 2, "label_names": []}', encoding="utf-8"
    )

    limited = BCDataset.load(str(dataset_dir), limit=20)
    out = capsys.readouterr().out
    assert "不支持前缀读取" in out
    assert "limit=20 → rows=18/36" in out
    assert limited.count == 18


def test_materialized_rows_scale_with_limit(tmp_path: Path) -> None:
    dataset_dir = _dataset_dir(tmp_path)
    limited = BCDataset.load(str(dataset_dir), limit=20)
    lines: list[str] = []
    source = MaterializedBCDataset(limited, chunk_size=5, logger=lines.append)
    assert source.count == limited.count == 18
    assert any("18 行" in line for line in lines), lines
    assert source.obs_batch(np.arange(3, dtype=np.int64))["ego"].shape[0] == 3


# --------------------------------------------------------------------------- #
# CPU 冒烟 batch 收敛
# --------------------------------------------------------------------------- #

class _Args:
    def __init__(self, limit: int | None) -> None:
        self.limit_dataset = limit


class _Device:
    def __init__(self, type_: str) -> None:
        self.type = type_


def test_smoke_batch_caps_only_for_cpu_limit() -> None:
    cpu, cuda = _Device("cpu"), _Device("cuda")
    # 未设 limit / GPU → 原样
    assert _apply_smoke_batch_caps(_Args(None), cpu, 1024, 256, 10**6) == (1024, 256)
    assert _apply_smoke_batch_caps(_Args(2048), cuda, 1024, 256, 10**6) == (1024, 256)
    # CPU + limit → macro ≤128 / micro ≤8；macro 同时 ≤ 子集行数
    assert _apply_smoke_batch_caps(_Args(2048), cpu, 1024, 256, 10**6) == (128, 8)
    assert _apply_smoke_batch_caps(_Args(64), cpu, 1024, 256, 50) == (50, 8)
    assert _apply_smoke_batch_caps(_Args(2048), cpu, 64, 16, 10**6) == (64, 8)
    # micro 缺失（配置无梯度累积）→ 冒烟强制 micro=micro 上限（否则 macro 整批前向仍爆内存）
    assert _apply_smoke_batch_caps(_Args(2048), cpu, 1024, None, 10**6) == (128, 8)
    # 已足够小 → 不改动
    assert _apply_smoke_batch_caps(_Args(2048), cpu, 128, 8, 10**6) == (128, 8)
    assert _smoke_cpu_guard(_Args(2048), cpu) is True
    assert _smoke_cpu_guard(_Args(None), cpu) is False
    assert _smoke_cpu_guard(_Args(2048), cuda) is False


# --------------------------------------------------------------------------- #
# Stage A/B 集成：metrics/日志都基于子集
# --------------------------------------------------------------------------- #

def _model_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(TINY_MODEL_YAML, encoding="utf-8")
    return path


def test_stage_a_limit_uses_subset_and_logs_rows(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    dataset_dir = _dataset_dir(tmp_path)
    metrics = run_stage_a(
        _parse_args([
            "--stage", "A", "--bc-dir", str(dataset_dir), "--out", str(tmp_path / "a"),
            "--model-config", str(_model_cfg(tmp_path)), "--wm-epochs", "1",
            "--limit-dataset", "20", "--val-frac", "0.34", "--batch-size", "64",
            "--micro-batch-size", "32", "--device", "cpu", "--seed", "0",
        ]),
        {},
    )
    out = capsys.readouterr().out
    assert "limit=20 → rows=18/36" in out
    assert "[materialize] obs+targets 18 行" in out
    assert "冒烟内存保护" in out and "macro=18 micro=8 eval_frames=64" in out
    assert metrics["limit_dataset"] == 20 and metrics["samples"] == 18
    assert metrics["train_frames"] + metrics["val_frames"] == 18
    assert metrics["batch_size"] == 18 and metrics["micro_batch_size"] == 8  # 收敛后
    assert metrics["grad_accum"] is True


def test_stage_b_limit_uses_subset(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    dataset_dir = _dataset_dir(tmp_path)
    stage_a_dir = tmp_path / "stage_a"
    run_stage_a(
        _parse_args([
            "--stage", "A", "--bc-dir", str(dataset_dir), "--out", str(stage_a_dir),
            "--model-config", str(_model_cfg(tmp_path)), "--wm-epochs", "1",
            "--limit-dataset", "20", "--batch-size", "16", "--eval-frames", "6",
            "--device", "cpu", "--seed", "0",
        ]),
        {},
    )
    capsys.readouterr()

    metrics = run_stage_b(
        _parse_args([
            "--stage", "B", "--bc-dir", str(dataset_dir), "--out", str(tmp_path / "b"),
            "--ckpt", str(stage_a_dir / "final.pt"), "--model-config", str(_model_cfg(tmp_path)),
            "--bc-epochs", "1", "--limit-dataset", "20", "--val-frac", "0.34",
            "--batch-size", "64", "--micro-batch-size", "32", "--device", "cpu", "--seed", "0",
        ]),
        {},
    )
    out = capsys.readouterr().out
    assert "limit=20 → rows=18/36" in out
    assert "[materialize] obs+targets 18 行" in out
    assert "冒烟内存保护" in out
    assert metrics["limit_dataset"] == 20 and metrics["samples"] == 18
    assert metrics["train_frames"] + metrics["val_frames"] == 18
    assert metrics["batch_size"] == 18 and metrics["micro_batch_size"] == 8
