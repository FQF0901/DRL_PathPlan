"""P1-1 预备 recovery 采集器离线单测（不建 env、不跑 ckpt）。

覆盖：锚点计算、三档分类、行数上限、episode 级 split 无泄漏、一致性判定
（strict/relaxed/skip）、质量门接线、mode 标签、ego 系变换、产物写出布局。
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from env.tracking import interpolate
from tools.controls import collect_recovery as cr
from tools.controls import recovery_quality_gates as gates


# --------------------------------------------------------------------------- #
# 锚点 / 分类 / 行选择 / split
# --------------------------------------------------------------------------- #
def _ts(step: int | None) -> dict | None:
    return None if step is None else {"step": int(step), "t_s": 0.1 * int(step), "reason": "x"}


def test_compute_anchor_prevention() -> None:
    # T_plan−20=180, T_cross−30=180, first_low=50 → start=180 < T_plan → prevention
    anchor = cr.compute_anchor(
        {0: 0.0, 5: 0.2, 50: 1.5, 100: 0.1},
        {"T_plan": _ts(200), "T_cross": _ts(210)},
        low_margin_lat=1.0,
        context_steps=20,
        recovery_steps=30,
    )
    assert anchor is not None
    assert anchor["start_raw"] == 180
    assert anchor["takeover_step"] == 180
    assert anchor["class"] == "prevention"
    assert anchor["first_low_margin"] == 50


def test_compute_anchor_correction_and_snap() -> None:
    # T_plan−20=80, T_cross−30=170 → start=170（≥T_plan、<T_cross）→ correction
    anchor = cr.compute_anchor(
        {0: 0.0},
        {"T_plan": _ts(100), "T_cross": _ts(200)},
        context_steps=20,
        recovery_steps=30,
    )
    assert anchor["start_raw"] == 170
    assert anchor["class"] == "correction"

    # 网格对齐：start_raw=123 → takeover=120
    anchor = cr.compute_anchor(
        {123: 1.2}, {"T_plan": None, "T_cross": None}, context_steps=20, recovery_steps=30
    )
    assert anchor["start_raw"] == 123
    assert anchor["takeover_step"] == 120
    assert anchor["class"] == "prevention"


def test_compute_anchor_late_and_none() -> None:
    # first_low=90 ≥ T_cross=80 → late
    anchor = cr.compute_anchor(
        {90: 1.4}, {"T_plan": _ts(50), "T_cross": _ts(80)}, context_steps=20, recovery_steps=30
    )
    assert anchor["class"] == "late"
    assert anchor["start_raw"] == 90

    assert cr.compute_anchor({0: 0.1}, {"T_plan": None, "T_cross": None}) is None


def test_compute_anchor_t_cross_fallback() -> None:
    # footprint T_cross=None（线压线类 out_of_road）→ env 失败步 200 作 effective T_cross
    anchor = cr.compute_anchor(
        {100: 1.2},
        {"T_plan": _ts(150), "T_cross": None},
        context_steps=20,
        recovery_steps=30,
        t_cross_fallback=200,
    )
    assert anchor["T_cross"] == 200
    assert anchor["T_cross_footprint"] is None
    assert anchor["T_cross_source"] == "env_event"
    assert anchor["start_raw"] == 170  # T_cross−30
    assert anchor["class"] == "correction"  # 150 ≤ 170 < 200

    # first_low 晚于 effective T_cross → late
    anchor = cr.compute_anchor(
        {210: 1.3},
        {"T_plan": None, "T_cross": None},
        context_steps=20,
        recovery_steps=30,
        t_cross_fallback=200,
    )
    assert anchor["class"] == "late"

    # 无 fallback → 旧行为（T_cross 保持 None）
    anchor = cr.compute_anchor(
        {100: 1.2}, {"T_plan": _ts(150), "T_cross": None}, context_steps=20, recovery_steps=30
    )
    assert anchor["T_cross"] is None
    assert anchor["T_cross_source"] is None


def test_classify_anchor_direct() -> None:
    assert cr.classify_anchor(10, 100, 200) == "prevention"
    assert cr.classify_anchor(100, 100, 200) == "correction"
    assert cr.classify_anchor(150, 100, 200) == "correction"
    assert cr.classify_anchor(200, 100, 200) == "late"
    assert cr.classify_anchor(10, None, None) == "prevention"


def test_select_rows_cap_and_episode_end() -> None:
    assert cr.select_rows(10, 1000, expert_steps=6, max_rows=20) == [10, 15, 20, 25, 30, 35]
    assert cr.select_rows(10, 1000, expert_steps=6, max_rows=3) == [10, 15, 20]
    assert cr.select_rows(10, 22, expert_steps=6, max_rows=20) == [10, 15, 20]
    assert len(cr.select_rows(0, 1000, expert_steps=30, max_rows=20)) == 20  # ≤20/集


def test_split_episodes_no_leakage() -> None:
    ids = [251, 9, 15, 21, 23, 90, 132, 160, 187, 225]
    train, val = cr.split_episodes(ids)
    assert train == [9, 15, 21, 23, 132, 160, 187, 225]
    assert val == [90, 251]
    assert not set(train) & set(val)
    train2, val2 = cr.split_episodes(ids)
    assert (train, val) == (train2, val2)  # 确定性


def test_consistency_verdict() -> None:
    assert cr.consistency_verdict(1e-9, 1e-6, 1e-4) == "strict"
    assert cr.consistency_verdict(1e-5, 1e-6, 1e-4) == "relaxed"
    assert cr.consistency_verdict(1e-2, 1e-6, 1e-4) == "skip"
    assert cr.consistency_verdict(float("nan"), 1e-6, 1e-4) == "skip"


# --------------------------------------------------------------------------- #
# 几何 / mode
# --------------------------------------------------------------------------- #
def test_ego_relative_traj() -> None:
    base = [1.0, 1.0, np.pi / 2]
    poses = np.array([[1.0, 2.0, np.pi / 2], [1.0, 3.0, np.pi / 2]])
    local = cr.ego_relative_traj(poses, base)
    assert np.allclose(local, [[1.0, 0.0], [2.0, 0.0]], atol=1e-9)


def test_mode_label_rules() -> None:
    straight = np.stack([np.arange(1, 7) * 1.0, np.zeros(6)], axis=1)
    assert cr.mode_label(straight)[0] == "keep_lane"
    shift = np.array([[1.0, 0.3], [2.0, 0.9], [3.0, 1.4], [4.0, 1.5], [5.0, 1.5], [6.0, 1.5]])
    assert cr.mode_label(shift)[0] == "left_shift"
    right = shift * np.array([1.0, -1.0])
    assert cr.mode_label(right)[0] == "right_shift"
    curve = np.array([[1.0, 0.0], [2.0, 0.4], [3.0, 0.9], [4.0, 1.5], [5.0, 2.2], [6.0, 3.0]])
    assert cr.mode_label(curve)[0] == "left_curve"
    complex_traj = np.array([[1.0, 0.0], [0.5, 1.0], [-1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, -1.0]])
    assert cr.mode_label(complex_traj)[0] == "turnaround_or_complex"


# --------------------------------------------------------------------------- #
# 质量门接线 / 产物写出
# --------------------------------------------------------------------------- #
def _chain_points(chain: np.ndarray) -> np.ndarray:
    return interpolate(np.asarray(chain, dtype=np.float64), dt=0.5, hz=10)[
        [4, 9, 14, 19, 24, 29], :2
    ]


def _synthetic_row(*, usable: bool = True) -> dict:
    chain = np.array([[1.0, 0.0]] * 6)
    traj = _chain_points(chain)
    row = {
        "episode_id": 7,
        "step": 20,
        "class": "correction",
        "pose": [0.0, 0.0, 0.0],
        "student_plan": np.array([[1.0, 0.0]] * 6),
        "student_mu": np.array([1.0, 0.0]),
        "student_actual_traj": traj.copy(),
        "student_actual_mask": True,
        "expert_action_chain": chain,
        "expert_recovery_traj": traj,
        "expert_actual_recovery_traj": traj + 0.01,
        "expert_actual_mask": True,
        "expert_raw_action": np.array([0.05, 0.2]),
        "lane_lat": 0.5,
        "rc": 0.4,
        "T_plan": 60.0,
        "T_cross": 120.0,
        "first_low_margin": 40.0,
        "start_raw": 90,
        "takeover_step": 90,
        "margin_before": -0.5,
        "margin_after": -0.1,
        "rc_before": 0.4,
        "rc_after": 0.6,
        "takeover_off_road": False,
        "takeover_collision": False,
        "expert_recovered": True,
        "recovery_success_horizon": 30,
        "min_margin_after_takeover": -0.1,
        "route_progress_after_takeover": 0.2,
        "mode_label": "keep_lane",
        "mode_confidence": 0.9,
        "geometry_labels": {
            "surface_valid": True,
            "surface_inside_ratio": 1.0,
            "solid_left": False,
            "solid_right": True,
            "route_rc": 0.4,
        },
        "geom_surface_valid": True,
        "geom_surface_inside_ratio": 1.0,
        "geom_solid_left": False,
        "geom_solid_right": True,
        "geom_route_rc": 0.4,
        "obs": {"ego": np.zeros((1, 8), dtype=np.float32), "ld_mask": np.ones((16,), dtype=np.float32)},
        "termination": "out_of_road",
        "map_id": "9:SCS",
        "road_class": "curve",
    }
    if not usable:
        row["takeover_off_road"] = True
        row["expert_recovered"] = False
    return row


def test_quality_gate_wiring_usable() -> None:
    row = _synthetic_row()
    result = gates.evaluate_row(cr.build_gate_row(row))
    assert result["usable"] is True, result["reasons"]
    row["usable"] = True
    row["quality_reasons"] = []

    bad = _synthetic_row(usable=False)
    result = gates.evaluate_row(cr.build_gate_row(bad))
    assert result["usable"] is False
    assert any("post_takeover_failure" in reason for reason in result["reasons"])


def test_quality_gate_catches_action_traj_mismatch() -> None:
    row = _synthetic_row()
    row["expert_recovery_traj"] = row["expert_recovery_traj"] + np.array([0.0, 3.0])
    result = gates.evaluate_row(cr.build_gate_row(row))
    assert result["usable"] is False
    assert any("action_traj_inconsistent" in reason for reason in result["reasons"])


def _fake_args(tmp_path: Path) -> Namespace:
    return Namespace(
        ckpt=str(tmp_path / "missing_ckpt.pt"),
        config=str(tmp_path / "missing_config.yaml"),
        spec=str(tmp_path / "missing_spec.json"),
        max_steps=1000,
        device="cpu",
        workers=1,
        expert_steps=6,
        context_steps=20,
        recovery_steps=30,
        low_margin_lat=1.0,
        max_rows_per_episode=20,
        action_traj_tol=0.5,
        ids="7,11",
        limit=0,
        out=str(tmp_path / "out"),
    )


def test_write_outputs_layout(tmp_path: Path) -> None:
    row = _synthetic_row()
    row["usable"] = True
    row["quality_reasons"] = []
    row2 = dict(row)
    row2["episode_id"] = 11
    row2["class"] = "late"
    row2["takeover_off_road"] = True
    row2["expert_recovered"] = False
    row2["usable"] = False
    row2["quality_reasons"] = ["post_takeover_failure:post_takeover_off_road"]
    episodes = [
        {
            "id": 7,
            "status": "collected",
            "anchor": {"class": "correction"},
            "n_rows": 1,
            "n_usable": 1,
            "expert": {"recovered": True},
            "determinism": {"accepted": "strict", "attempts": []},
        },
        {
            "id": 11,
            "status": "collected",
            "anchor": {"class": "late"},
            "n_rows": 1,
            "n_usable": 0,
            "expert": {"recovered": False},
            "determinism": {"accepted": "relaxed", "attempts": []},
        },
    ]
    out = tmp_path / "out"
    cr._write_outputs(out, [row, row2], episodes, _fake_args(tmp_path))

    for name in ("manifest.json", "samples.npz", "schema.json", "quality_report.json", "split_manifest.json"):
        assert (out / name).is_file(), name

    data = np.load(out / "samples.npz", allow_pickle=False)
    assert data["episode_id"].tolist() == [7, 11]
    assert data["class_id"].tolist() == [1, 2]  # correction / late
    assert data["usable"].tolist() == [True, False]
    assert data["expert_recovery_traj"].shape == (2, 6, 2)
    assert data["obs__ego"].shape == (2, 1, 8)
    assert data["obs__ld_mask"].shape == (2, 16)
    assert data["mode_id"].tolist() == [0, 0]

    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    assert schema["schema_version"] == 1
    assert schema["class_definition"]["target_ratio"] == {"prevention": 0.5, "correction": 0.35, "late": 0.15}

    quality = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["n_rows"] == 2
    assert quality["n_usable"] == 1
    assert quality["gate_fail_counts"]["post_takeover_failure"] == 1
    assert quality["per_class"]["correction"] == {"rows": 1, "usable": 1}

    split = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["no_leakage"] is True
    assert set(split["train_episodes"]) & set(split["val_episodes"]) == set()
    assert split["train"]["rows"] + split["val"]["rows"] == 2

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["rows_total"] == 2
    assert manifest["counts"]["rows_usable"] == 1
    assert manifest["divergence"]["relaxed"] == 1
    assert manifest["field_gaps"]
