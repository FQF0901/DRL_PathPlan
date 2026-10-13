"""R21-27B 契约测试：N0/N1/C 定义锁定（纯函数）。"""

from __future__ import annotations

import numpy as np

from pipeline.residual_classes import (
    CLASS_C,
    CLASS_N0,
    CLASS_N1,
    class_stats,
    classify_rows,
    compute_thresholds,
    episode_consistency,
)


def _features(row_type, res_ds, res_dth, valid, lat_max, head_max, speed_min, episode):
    return {
        "row_type": np.asarray(row_type, dtype=np.int64),
        "res_ds": np.asarray(res_ds, dtype=np.float64),
        "res_dth": np.asarray(res_dth, dtype=np.float64),
        "future_valid": np.asarray(valid, dtype=bool),
        "future_lat_max": np.asarray(lat_max, dtype=np.float64),
        "future_head_max": np.asarray(head_max, dtype=np.float64),
        "future_speed_min": np.asarray(speed_min, dtype=np.float64),
        "episode_id": np.asarray(episode, dtype=np.int64),
        "geometry_code": np.zeros(len(row_type), dtype=np.int64),
        "speed": np.full(len(row_type), 8.0),
    }


def test_thresholds_are_teacher_quantiles() -> None:
    teacher = np.array([True, True, True, True, False])
    res_ds = np.array([0.1, 0.2, 0.3, 0.4, 99.0])
    res_dth = np.array([0.01, 0.02, 0.03, 0.04, 9.0])
    thresholds = compute_thresholds(res_ds, res_dth, teacher)
    assert abs(thresholds["tau_ds"] - 0.25) < 1e-12
    assert abs(thresholds["tau_dth"] - 0.025) < 1e-12
    assert thresholds["n_teacher"] == 4


def test_classify_n0_n1_c_and_guards() -> None:
    # 行 0/1/2：教师行（ep 0）残差递增；行 3：教师行但未来不安全；行 4：pert；行 5：learner
    f = _features(
        row_type=[0, 0, 0, 0, 1, 2],
        res_ds=[0.05, 0.30, 0.60, 0.10, 0.10, 0.10],
        res_dth=[0.001, 0.030, 0.050, 0.001, 0.001, 0.001],
        valid=[True, True, True, True, True, True],
        lat_max=[0.2, 0.4, 0.5, 2.5, 0.1, 0.1],   # 行 3 未来横偏超界 → 不安全
        head_max=[0.02, 0.03, 0.04, 0.02, 0.01, 0.01],
        speed_min=[5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        episode=[0, 0, 0, 0, 1, 2],
    )
    thresholds = compute_thresholds(f["res_ds"], f["res_dth"], np.asarray(f["row_type"]) == 0)
    labels = classify_rows(f, thresholds)
    assert labels[0] == CLASS_N0           # 残差最小 + 安全
    assert labels[1] == CLASS_N1           # 残差中等 + episode 符号一致（全正）
    assert labels[2] == CLASS_C            # 残差超 P90
    assert labels[3] == CLASS_C            # 教师行但未来不安全（lat 2.5）
    assert labels[4] == CLASS_C and labels[5] == CLASS_C  # pert/learner → C


def test_n1_requires_episode_consistency() -> None:
    # 同一 episode 残差符号来回（2:4 不达 0.7）→ 中等残差不判 N1，落 C
    f = _features(
        row_type=[0] * 8,
        res_ds=[0.01, 0.02, 0.03, -0.04, 0.5, 0.6, -0.5, -0.6],
        res_dth=[0.010] * 8,
        valid=[True] * 8, lat_max=[0.2] * 8, head_max=[0.02] * 8, speed_min=[5.0] * 8,
        episode=[0] * 8,
    )
    thresholds = compute_thresholds(f["res_ds"], f["res_dth"], np.asarray(f["row_type"]) == 0)
    labels = classify_rows(f, thresholds)
    assert list(labels[:4]) == [CLASS_N0] * 4, labels
    assert list(labels[4:]) == [CLASS_C] * 4, labels  # 中等残差但符号不一致（5:3）→ C
    consistency = episode_consistency(f["episode_id"], f["res_ds"], np.asarray(f["row_type"]) == 0)
    assert abs(consistency[0] - (5.0 / 8.0)) < 1e-12  # 5 正 : 3 负 → 主导符号比例 0.625 < 0.7


def test_invalid_future_forces_c() -> None:
    f = _features(
        row_type=[0, 0], res_ds=[0.0, 0.0], res_dth=[0.0, 0.0],
        valid=[False, True], lat_max=[0.1, 0.1], head_max=[0.01, 0.01], speed_min=[5.0, 5.0],
        episode=[0, 0],
    )
    thresholds = compute_thresholds(f["res_ds"], f["res_dth"], np.asarray(f["row_type"]) == 0)
    labels = classify_rows(f, thresholds)
    assert labels[0] == CLASS_C and labels[1] == CLASS_N0


def test_class_stats_shares_and_strata() -> None:
    f = _features(
        row_type=[0, 0, 1, 2], res_ds=[0.0, 0.5, 0.0, 0.0], res_dth=[0.0, 0.05, 0.0, 0.0],
        valid=[True] * 4, lat_max=[0.1] * 4, head_max=[0.01] * 4, speed_min=[5.0] * 4,
        episode=[0, 0, 1, 2],
    )
    thresholds = compute_thresholds(f["res_ds"], f["res_dth"], np.asarray(f["row_type"]) == 0)
    labels = classify_rows(f, thresholds)
    stats = class_stats(labels, f)
    assert stats["n"] == 4
    assert abs(sum(stats["share"].values()) - 1.0) < 1e-12
    assert sum(entry["n"] for entry in stats["by_row_type"]["teacher"].values()) == 2
    assert sum(entry["n"] for entry in stats["by_row_type"]["pert"].values()) == 1
    assert sum(sum(entry["n"] for entry in group.values()) for group in stats["by_row_type"].values()) == 4
