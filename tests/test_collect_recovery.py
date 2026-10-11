"""P1-1 recovery 采集器离线单测（不建 env、不跑 ckpt）。

覆盖：多窗口候选/实况三档分类/每档 ≤1 与 ≤20 行每集、配额截断、池选择、池规划外推、
episode 级 split 无泄漏（'pool:id' 键）、一致性判定（strict/relaxed/skip）、质量门接线、
mode 标签、ego 系变换、产物写出布局。
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
# 窗口候选 / 分类 / 选择 / 配额
# --------------------------------------------------------------------------- #
def test_generate_window_candidates() -> None:
    candidates = cr.generate_window_candidates(
        t_plan=130,
        t_cross=556,
        first_low_margin=125,
        context_steps=20,
        recovery_steps=30,
        late_margin_steps=5,
    )
    # {125, 110, 130, 525, 550, 555}（已对齐 5 网格）
    assert candidates == [110, 125, 130, 525, 550, 555]

    # 仅 first_low；以及 T_cross 邻域 clamp 到 0
    assert cr.generate_window_candidates(
        t_plan=None, t_cross=None, first_low_margin=17
    ) == [15]
    assert cr.generate_window_candidates(
        t_plan=None, t_cross=3, first_low_margin=None, recovery_steps=30, late_margin_steps=5
    ) == [0]

    # max_decision_step 截断
    assert cr.generate_window_candidates(
        t_plan=130, t_cross=None, first_low_margin=125, max_decision_step=100
    ) == [100]


def test_classify_window_three_classes() -> None:
    # prevention：计划可行 + margin 健康
    assert (
        cr.classify_window(
            100, t_plan=130, t_cross=None, lane_lat=0.4,
            plan_footprint_valid=True, exec_footprint_valid=True,
        )
        == "prevention"
    )
    # correction：plan footprint 不可行 / ≥T_plan / |lane_lat| ≥ low_lat
    assert (
        cr.classify_window(
            130, t_plan=130, t_cross=600, lane_lat=0.4,
            plan_footprint_valid=False, exec_footprint_valid=True,
        )
        == "correction"
    )
    assert (
        cr.classify_window(
            100, t_plan=None, t_cross=600, lane_lat=1.2,
            plan_footprint_valid=True, exec_footprint_valid=True,
        )
        == "correction"
    )
    # late：T_cross 邻域 / exec footprint 不可行 / |lane_lat| ≥ late_lat
    assert (
        cr.classify_window(
            551, t_plan=130, t_cross=556, lane_lat=0.4,
            plan_footprint_valid=True, exec_footprint_valid=True,
        )
        == "late"
    )
    assert (
        cr.classify_window(
            300, t_plan=130, t_cross=556, lane_lat=0.4,
            plan_footprint_valid=True, exec_footprint_valid=False,
        )
        == "late"
    )
    assert (
        cr.classify_window(
            300, t_plan=130, t_cross=556, lane_lat=1.6,
            plan_footprint_valid=True, exec_footprint_valid=True,
        )
        == "late"
    )


def test_select_windows_one_per_class_and_cap() -> None:
    candidates = [110, 125, 130, 525, 550, 555]
    states = {
        110: {"lane_lat": 0.3, "plan_footprint_valid": True, "exec_footprint_valid": True},
        125: {"lane_lat": 1.1, "plan_footprint_valid": True, "exec_footprint_valid": True},
        130: {"lane_lat": 0.9, "plan_footprint_valid": False, "exec_footprint_valid": True},
        525: {"lane_lat": 1.2, "plan_footprint_valid": False, "exec_footprint_valid": True},
        550: {"lane_lat": 1.6, "plan_footprint_valid": True, "exec_footprint_valid": True},
        555: {"lane_lat": 1.2, "plan_footprint_valid": True, "exec_footprint_valid": False},
    }
    windows = cr.select_windows(
        candidates, lambda step: states[step], t_plan=130, t_cross=556,
        low_lat=1.0, late_lat=1.5, late_margin_steps=5,
    )
    assert windows == [("prevention", 110), ("correction", 125), ("late", 555)]

    # max_windows 截断（prevention → correction → late 顺序）
    assert cr.select_windows(
        candidates, lambda step: states[step], t_plan=130, t_cross=556, max_windows=2
    ) == [("prevention", 110), ("correction", 125)]

    # 无候选 → 空
    assert cr.select_windows([], lambda step: {}, t_plan=None, t_cross=None) == []


def test_rows_per_episode_cap() -> None:
    windows = [("prevention", 10), ("correction", 60), ("late", 120)]
    total = sum(len(cr.select_rows(step, 1000, expert_steps=6, max_rows=20)) for _, step in windows)
    assert total == 18  # 3 窗口 × 6 行 ≤ 20/集
    assert len(cr.select_rows(10, 1000, expert_steps=6, max_rows=20)) == 6


def test_filter_windows_by_quota() -> None:
    windows = [("prevention", 10), ("correction", 60), ("late", 120)]
    targets = {"prevention": 10, "correction": 2500, "late": 1000}
    counts = {"prevention": 10, "correction": 5, "late": 0}
    assert cr.filter_windows_by_quota(windows, counts, targets) == [
        ("correction", 60), ("late", 120)
    ]


def test_parse_targets() -> None:
    assert cr.parse_targets("prevention=4000,correction=2500,late=1000") == {
        "prevention": 4000, "correction": 2500, "late": 1000
    }
    assert cr.parse_targets("late=7") == {"prevention": 0, "correction": 0, "late": 7}
    try:
        cr.parse_targets("unknown=1")
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("未知档位应报错")


def test_parse_shard() -> None:
    assert cr.parse_shard("0/1") == (0, 1)
    assert cr.parse_shard("3/6") == (3, 6)
    for bad in ("1", "6/6", "-1/3", "a/2", "0/0"):
        try:
            cr.parse_shard(bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"非法分片参数应报错: {bad}")


def test_shard_episodes_complete_disjoint_deterministic() -> None:
    selected = [
        ("b", _FakeSpec(2, "curve")),
        ("a", _FakeSpec(10, "merge")),
        ("b", _FakeSpec(1, "curve")),
        ("a", _FakeSpec(3, "curve")),
        ("a", _FakeSpec(7, "straight")),
    ]
    shards = [cr.shard_episodes(selected, index=i, count=3) for i in range(3)]
    # 稳定排序（池, id）→ round-robin：a:3 a:7 a:10 b:1 b:2
    assert [[(p, s.id) for p, s in shard] for shard in shards] == [
        [("a", 3), ("b", 1)],
        [("a", 7), ("b", 2)],
        [("a", 10)],
    ]
    union = [item for shard in shards for item in shard]
    assert len(union) == len(selected)
    assert {(p, s.id) for p, s in union} == {(p, s.id) for p, s in selected}
    for i in range(3):
        for j in range(3):
            if i != j:
                assert not ({(p, s.id) for p, s in shards[i]} & {(p, s.id) for p, s in shards[j]})
    # 确定性
    assert cr.shard_episodes(selected, index=1, count=3) == shards[1]
    # 空分片合法
    assert cr.shard_episodes(selected, index=5, count=6) == []


def test_apply_quota_policy() -> None:
    windows = [("prevention", 10), ("correction", 60), ("late", 120)]
    targets = {"prevention": 10, "correction": 2500, "late": 1000}
    counts = {"prevention": 10, "correction": 5, "late": 0}
    assert cr.apply_quota_policy(windows, counts, targets) == [("correction", 60), ("late", 120)]
    assert cr.apply_quota_policy(windows, counts, targets, stop_at_target=False) == windows


# --------------------------------------------------------------------------- #
# 池选择 / split
# --------------------------------------------------------------------------- #
class _FakeSpec:
    def __init__(self, spec_id: int, geometry: str):
        self.id = spec_id
        self.labels = {"geometry": geometry}


def _fake_pools() -> list[dict]:
    return [
        {
            "name": "a",
            "specs": [_FakeSpec(1, "curve"), _FakeSpec(2, "merge"), _FakeSpec(3, "curve")],
            "limit": None,
        },
        {
            "name": "b",
            "specs": [_FakeSpec(10, "curve"), _FakeSpec(11, "merge")],
            "limit": None,
        },
    ]


def test_select_episodes_by_ids() -> None:
    selected = cr.select_episodes(_fake_pools(), ids="2,11")
    assert [(name, spec.id) for name, spec in selected] == [("a", 2), ("b", 11)]


def test_select_episodes_stratified_round_robin() -> None:
    selected = cr.select_episodes(_fake_pools(), per_geometry=10)
    assert [(name, spec.id) for name, spec in selected] == [
        ("a", 1), ("a", 2), ("b", 10), ("b", 11), ("a", 3)
    ]
    limited = cr.select_episodes(_fake_pools(), per_geometry=10, limit=3)
    assert [(name, spec.id) for name, spec in limited] == [("a", 1), ("a", 2), ("b", 10)]


def test_select_episodes_pool_limit() -> None:
    pools = _fake_pools()
    pools[0]["limit"] = 2
    selected = cr.select_episodes(pools)
    assert [(name, spec.id) for name, spec in selected] == [("a", 1), ("a", 2), ("b", 10), ("b", 11)]


def test_split_episodes_no_leakage() -> None:
    keys = ["eval500:9", "eval500:15", "clean500:9", "clean500:90", "dagger_r1:21", "dagger_r1:23"]
    train, val = cr.split_episodes(keys)
    assert val == ["eval500:15"]  # sorted 后第 5 个（pos 4）
    assert train == ["clean500:9", "clean500:90", "dagger_r1:21", "dagger_r1:23", "eval500:9"]
    assert not set(train) & set(val)
    assert (train, val) == cr.split_episodes(keys)  # 确定性
    # int 键兼容
    train_i, val_i = cr.split_episodes([1, 2, 3, 4, 5])
    assert train_i == ["1", "2", "3", "4"] and val_i == ["5"]


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


def _synthetic_row(*, usable: bool = True, window_class: str = "correction") -> dict:
    chain = np.array([[1.0, 0.0]] * 6)
    traj = _chain_points(chain)
    row = {
        "episode_id": 7,
        "pool": "eval500",
        "geometry": "merge",
        "step": 20,
        "class": window_class,
        "window_index": 0,
        "takeover_step": 20,
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
        "T_cross_source": "env_event",
        "first_low_margin": 40.0,
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
        "road_class": "merge",
        "usable": usable,
        "quality_reasons": [],
    }
    if not usable:
        row["takeover_off_road"] = True
        row["expert_recovered"] = False
    return row


def test_quality_gate_wiring_usable() -> None:
    result = gates.evaluate_row(cr.build_gate_row(_synthetic_row()))
    assert result["usable"] is True, result["reasons"]
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


def test_build_pool_plan_extrapolation() -> None:
    rows = []
    for index in range(10):
        row = _synthetic_row(window_class="prevention")
        row["episode_id"] = index % 3
        row["pool"] = "eval500"
        row["geometry"] = "merge" if index < 4 else "curve"
        row["road_class"] = "merge" if index < 4 else "curve"
        rows.append(row)
    for index in range(5):
        row = _synthetic_row(window_class="correction")
        row["episode_id"] = index % 2
        row["pool"] = "dagger_r1"
        row["geometry"] = "tollgate"
        row["road_class"] = "tollgate"
        rows.append(row)
    episodes = [
        {"id": 0, "pool": "eval500", "status": "collected"},
        {"id": 1, "pool": "eval500", "status": "collected"},
        {"id": 2, "pool": "eval500", "status": "collected"},
        {"id": 0, "pool": "dagger_r1", "status": "collected"},
        {"id": 1, "pool": "dagger_r1", "status": "collected"},
    ]
    pools_info = [
        {"name": "eval500", "n_specs": 500, "selected": 3},
        {"name": "dagger_r1", "n_specs": 500, "selected": 2},
    ]
    plan = cr.build_pool_plan(rows, episodes, pools_info, {"prevention": 4000, "correction": 2500, "late": 1000})
    assert plan["class_rates_per_episode"]["prevention"] == 2.0
    assert plan["class_rates_per_episode"]["correction"] == 1.0
    assert plan["class_rates_per_episode"]["late"] == 0.0
    assert plan["required_episodes_by_class"] == {
        "prevention": 2000, "correction": 2500, "late": None
    }
    assert plan["required_episodes_total"] is None  # late 试点零产出 → 无法外推
    assert "late" in plan["recommendation"]
    assert plan["per_pool_yield"]["eval500"]["usable_rows"] == 10
    assert plan["per_pool_yield"]["dagger_r1"]["usable_rows"] == 5
    assert plan["projection_all_pools"]["eval500"]["projected_usable_rows"] == 1667  # 500×3.33
    hard = plan["hard_types"]
    assert hard["merge"]["usable_rows"] == 4
    assert hard["merge"]["rows_gap"] == 296
    assert hard["tollgate"]["usable_rows"] == 5
    assert hard["tollgate"]["episodes"] == 2
    assert hard["roundabout"]["usable_rows"] == 0
    assert hard["roundabout"]["required_episodes_by_rows"] is None


def _fake_args(tmp_path: Path) -> Namespace:
    return Namespace(
        ckpt=str(tmp_path / "missing_ckpt.pt"),
        config=str(tmp_path / "missing_config.yaml"),
        spec=["missing_spec.json"],
        max_steps=1000,
        device="cpu",
        workers=1,
        expert_steps=6,
        context_steps=20,
        recovery_steps=30,
        low_lat=1.0,
        late_lat=1.5,
        late_margin_steps=5,
        max_windows_per_episode=3,
        max_rows_per_episode=20,
        action_traj_tol=0.5,
        targets={"prevention": 4000, "correction": 2500, "late": 1000},
        ids="7,11",
        limit=0,
        stratify_per_geometry=0,
        shard=(0, 1),
        no_stop_at_target=False,
        out=str(tmp_path / "out"),
    )


def test_write_outputs_layout(tmp_path: Path) -> None:
    row = _synthetic_row(window_class="correction")
    row2 = _synthetic_row(window_class="late", usable=False)
    row2["episode_id"] = 11
    row2["pool"] = "clean500"
    row2["window_index"] = 1
    episodes = [
        {
            "id": 7,
            "pool": "eval500",
            "geometry": "merge",
            "status": "collected",
            "windows": [
                {"class": "prevention", "takeover_step": 5, "status": "collected",
                 "determinism": {"accepted": "strict"}, "n_rows": 6, "n_usable": 6},
                {"class": "correction", "takeover_step": 20, "status": "collected",
                 "determinism": {"accepted": "strict"}, "n_rows": 6, "n_usable": 6},
            ],
            "n_rows": 12,
            "n_usable": 12,
        },
        {
            "id": 11,
            "pool": "clean500",
            "geometry": "tollgate",
            "status": "collected",
            "windows": [
                {"class": "late", "takeover_step": 100, "status": "collected",
                 "determinism": {"accepted": "relaxed"}, "n_rows": 6, "n_usable": 0},
            ],
            "n_rows": 6,
            "n_usable": 0,
        },
        {"id": 12, "pool": "clean500", "status": "skipped", "reason": "no_window"},
    ]
    pools_info = [
        {"name": "eval500", "path": "a.json", "sha256": "x", "n_specs": 500, "selected": 1},
        {"name": "clean500", "path": "b.json", "sha256": "y", "n_specs": 500, "selected": 2},
    ]
    out = tmp_path / "out"
    cr._write_outputs(out, [row, row2], episodes, _fake_args(tmp_path), pools_info)

    for name in ("manifest.json", "samples.npz", "schema.json", "quality_report.json", "split_manifest.json"):
        assert (out / name).is_file(), name

    data = np.load(out / "samples.npz", allow_pickle=False)
    assert data["episode_id"].tolist() == [7, 11]
    assert data["pool"].tolist() == ["eval500", "clean500"]
    assert data["class_id"].tolist() == [1, 2]  # correction / late
    assert data["window_index"].tolist() == [0, 1]
    assert data["usable"].tolist() == [True, False]
    assert data["expert_recovery_traj"].shape == (2, 6, 2)
    assert data["obs__ego"].shape == (2, 1, 8)
    assert "start_raw" not in data.files

    schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
    assert schema["schema_version"] == 2
    assert "late" in schema["class_definition"]
    assert schema["class_definition"]["thresholds"]["late_lat"] == 1.5

    quality = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["n_rows"] == 2
    assert quality["n_usable"] == 1
    assert quality["gate_fail_counts"]["post_takeover_failure"] == 1
    assert quality["per_class"]["correction"] == {"rows": 1, "usable": 1}
    assert quality["quota"]["late"]["target"] == 1000
    assert quality["quota"]["late"]["usable"] == 0
    assert quality["quota"]["late"]["gap"] == 1000
    assert quality["per_pool"]["eval500"]["usable_rows"] == 1
    assert quality["road_type_coverage"]["hard_types"] == ["tollgate", "merge", "roundabout"]

    split = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["no_leakage"] is True
    assert set(split["train_episodes"]) & set(split["val_episodes"]) == set()
    assert split["train"]["rows"] + split["val"]["rows"] == 2

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["rows_total"] == 2
    assert manifest["counts"]["rows_usable"] == 1
    assert manifest["divergence"]["relaxed"] == 1
    assert manifest["quota"]["prevention"]["met"] is False
    assert manifest["pool_plan"]["class_rates_per_episode"]["correction"] == 0.5
    assert manifest["pools"][0]["name"] == "eval500"
    assert manifest["field_gaps"]
