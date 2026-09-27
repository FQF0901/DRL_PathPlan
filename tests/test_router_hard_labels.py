"""lane B 机制回归：router 硬标签 sidecar（严格只读）+ 评测 KPI → TB。

- Stage B 只读 `<ds>/cluster_v<k>_assignments.npz`（annotate 生成）：缺失/行数不符 → fail-fast
  并打印生成命令；训练侧**禁止在线重算**；
- （lane U1 去聚类：训练侧不再消费 router 硬标签——`pipeline.trainer` 的 router CE/placeholder
  已删除，本文件只保留聚类 artifact 侧的 sidecar 契约测试）；
- `eval_runner.write_eval_tensorboard`：overall/by_primary/by_difficulty 全量 KPI 单点写 TB。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from dataclasses import replace

from pipeline.clusters import annotate_assignments, assignments_path, load as load_clusters, load_assignments
from pipeline.trainer import BCDataset
from tests.v2_synthetic import write_v2_dataset

_MODEL_CFG = "config/clusters/default.yaml"


def _dataset_dir(tmp_path, *, episodes: int = 4, steps_per_episode: int = 6):
    return write_v2_dataset(tmp_path / "bc_v2", episodes=episodes, steps_per_episode=steps_per_episode)


def _annotate(dataset_dir, **kwargs):
    dataset = BCDataset.load(str(dataset_dir))
    return annotate_assignments(
        dataset_dir=dataset_dir,
        count=int(dataset.count),
        obs_batch_fn=dataset.build_obs_batch,
        spec=load_clusters(_MODEL_CFG),
        dataset_meta=dataset.meta,
        logger=lambda _: None,
        **kwargs,
    )


def test_sidecar_roundtrip_and_strict_validation(tmp_path) -> None:
    dataset_dir = _dataset_dir(tmp_path)
    dataset = BCDataset.load(str(dataset_dir))
    spec = load_clusters(_MODEL_CFG)

    # 缺失 → fail-fast（附生成命令；禁止在线重算）
    with pytest.raises(RuntimeError, match="annotate_clusters.py"):
        load_assignments(dataset_dir, spec=spec, dataset_meta=dataset.meta)

    path = _annotate(dataset_dir)
    assert path == assignments_path(dataset_dir, spec=spec) and path.is_file()
    labels = load_assignments(dataset_dir, spec=spec, dataset_meta=dataset.meta)
    assert labels.shape == (dataset.count,) and labels.dtype == np.int64
    assert labels.min() >= 0 and labels.max() < spec.k
    with np.load(path) as data:
        assert set(("cluster", "top1_margin", "top1_prob", "rows", "cluster_k", "spec_hash",
                    "spec_path", "obs_fingerprint_dataset", "obs_fingerprint_spec",
                    "created_at", "row_alignment")) <= set(data.files)
        assert "router_soft_targets" not in data.files, "不得落地 (N,8) 软分布"
        assert str(np.asarray(data["obs_fingerprint_dataset"]).reshape(-1)[0]) == str(
            dataset.meta.get("obs_fingerprint") or ""
        )

    # 行数不符 → fail-fast
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload["rows"] = np.asarray([int(dataset.count) + 1], dtype=np.int64)
    np.savez_compressed(path, **payload)
    with pytest.raises(RuntimeError, match="校验不过"):
        load_assignments(dataset_dir, spec=spec, dataset_meta=dataset.meta)


def test_assignments_path_version_priority(tmp_path) -> None:
    """sidecar 路径只认版本对应文件（绝不回退到其它版本 → 不覆盖旧 sidecar）。"""
    from pipeline.clusters import assignments_path

    (tmp_path / "cluster_v1_assignments.npz").write_bytes(b"")
    spec_v2 = load_clusters(_MODEL_CFG)
    spec_v2 = replace(spec_v2, cluster_version="v2")
    assert assignments_path(tmp_path, spec=spec_v2).name == "cluster_v2_assignments.npz"
    spec_v1 = replace(spec_v2, cluster_version="v1")
    assert assignments_path(tmp_path, spec=spec_v1).name == "cluster_v1_assignments.npz"
    spec_nat = replace(spec_v2, cluster_version="v2_natural")
    assert assignments_path(tmp_path, spec=spec_nat).name == "cluster_v2_natural_assignments.npz"


def test_collect_expert_annotate_hook(tmp_path) -> None:
    """采集收尾钩子：有可用 spec 时自动批注 sidecar（生数据集时算一次）。"""
    from pathlib import Path as _Path

    from tests.v2_synthetic import make_v2_arrays
    from tools.collect_expert import _annotate_sidecar

    arrays, _ = make_v2_arrays(episodes=2, steps_per_episode=6)
    out_dir = tmp_path / "ds"
    out_dir.mkdir()
    path = _annotate_sidecar(out_dir, arrays, int(arrays["episode_id"].shape[0]))
    assert path is not None and _Path(path).is_file()
    labels = load_assignments(out_dir, spec=load_clusters(_MODEL_CFG))
    assert labels.shape == (int(arrays["episode_id"].shape[0]),)


def test_eval_tensorboard_overall_and_grouped(tmp_path) -> None:
    from pipeline.eval_runner import write_eval_tensorboard

    report = {
        "overall": {"n": 4, "success_rate": 0.5, "collision_rate": 0.25,
                    "success_wilson": [0.15, 0.85], "termination_counts": {"success": 2}},
        "by_primary": {"straight": {"n": 2, "success_rate": 1.0},
                       "curve": {"n": 2, "success_rate": 0.0}},
        "by_difficulty": {"easy": {"n": 4, "success_rate": 0.5}},
        "episodes": [{"id": "spec-0", "success": True}],
    }
    monitor_dir = tmp_path / "monitor"
    assert write_eval_tensorboard(report, monitor_dir) is True

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(monitor_dir))
    accumulator.Reload()
    scalars = {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }
    assert scalars["eval/success_rate"] == [(1, pytest.approx(0.5))]
    assert scalars["eval/collision_rate"] == [(1, pytest.approx(0.25))]
    assert scalars["eval/by_primary/success_rate/straight"] == [(1, pytest.approx(1.0))]
    assert scalars["eval/by_primary/success_rate/curve"] == [(1, pytest.approx(0.0))]
    assert scalars["eval/by_difficulty/success_rate/easy"] == [(1, pytest.approx(0.5))]
    # 非数值（wilson 列表/计数 dict）与逐 spec 明细不写 TB
    assert not [tag for tag in scalars if "wilson" in tag or "spec" in tag]
