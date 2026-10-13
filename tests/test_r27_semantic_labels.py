"""R21-27 契约测试：B 行标签消费守卫（NPZ 入口硬拒 v1）。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline.semantic_labels import V2_VERSION, load_samples_v2, load_v2_label_map, v1_exposure


def _write_dataset(tmp_path, *, b_rows=((7, 10), (7, 15)), missing_v2=False):
    n = 3 + len(b_rows)
    arrays = {
        "episode_id": np.array([1, 2] + [e for e, _ in b_rows], dtype=np.int64),
        "step": np.array([0, 0] + [s for _, s in b_rows], dtype=np.int64),
        "row_type": np.array([0, 2] + [1] * len(b_rows), dtype=np.int64),
        # B 行 expert_action = 旧 v1 值（哨兵 9.0，若被使用即为泄漏）
        "expert_action": np.array([[1.0, 0.1], [2.0, 0.2]] + [[9.0, 9.0]] * len(b_rows), dtype=np.float32),
        "geometry_code": np.zeros(n, dtype=np.int64),
        "split": np.zeros(n, dtype=np.int64),
        "speed": np.full(n, 8.0, dtype=np.float32),
    }
    np.savez_compressed(tmp_path / "samples.npz", **arrays)
    keys = [] if missing_v2 else list(b_rows)
    np.savez_compressed(
        tmp_path / "labels.npz",
        episode_id=np.array([e for e, _ in keys], dtype=np.int64),
        step=np.array([s for _, s in keys], dtype=np.int64),
        action=np.zeros((len(keys), 2), dtype=np.float32),
        perturbed_action=np.full((len(keys), 2), 5.0, dtype=np.float32),
        label_pose=np.zeros((len(keys), 3), dtype=np.float32),
        label_state_version=np.asarray([V2_VERSION]),
    )
    contract = {"v1_exposed_b_rows": 0, "excluded_b_rows": []}
    (tmp_path / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    return tmp_path


def test_b_rows_are_overridden_and_v1_never_returned(tmp_path) -> None:
    _write_dataset(tmp_path)
    loaded = load_samples_v2(tmp_path, tmp_path / "labels.npz", tmp_path / "contract.json")
    expert = loaded["arrays"]["expert_action"]
    assert loaded["v1_exposed"] == 0
    assert loaded["label_meta"]["covered_b_rows"] == 2
    # B 行 = v2 值（5.0），旧 v1 哨兵 9.0 绝不出现在返回数组
    assert float(expert[2:].min()) == 5.0
    assert not np.any(np.isclose(expert, 9.0))


def test_missing_v2_fails_fast(tmp_path) -> None:
    _write_dataset(tmp_path, missing_v2=True)
    with pytest.raises(SystemExit):
        load_samples_v2(tmp_path, tmp_path / "labels.npz", tmp_path / "contract.json")


def test_wrong_version_fails_fast(tmp_path) -> None:
    _write_dataset(tmp_path)
    with np.load(tmp_path / "labels.npz") as payload:
        data = {key: payload[key] for key in payload.files}
    data["label_state_version"] = np.asarray(["v1"])
    np.savez_compressed(tmp_path / "labels.npz", **data)
    with pytest.raises(SystemExit):
        load_v2_label_map(tmp_path / "labels.npz")


def test_excluded_row_is_dropped_and_marked(tmp_path) -> None:
    _write_dataset(tmp_path)
    contract = {"v1_exposed_b_rows": 0, "excluded_b_rows": [{"episode_id": 7, "step": 10}]}
    (tmp_path / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    loaded = load_samples_v2(tmp_path, tmp_path / "labels.npz", tmp_path / "contract.json")
    assert loaded["keep_mask"].tolist() == [True, True, False, True]
    assert np.isnan(loaded["arrays"]["expert_action"][2]).all()


def test_v1_exposure_counts_uncovered(tmp_path) -> None:
    _write_dataset(tmp_path)
    labels = load_v2_label_map(tmp_path / "labels.npz")
    with np.load(tmp_path / "samples.npz") as payload:
        arrays = {key: payload[key] for key in payload.files}
    assert v1_exposure(arrays, labels) == 0
    labels["perturbed"].pop((7, 10))
    assert v1_exposure(arrays, labels) == 1
