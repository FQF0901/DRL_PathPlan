"""纯测试：主标签分组 / Wilson CI / 弱类 floor（P2/N5 ``pipeline/eval_runner.py``）。

只测纯函数与判定逻辑：不建 env、不导入 metadrive/torch（``eval_runner`` 模块 import 仅
stdlib+numpy+yaml，metadrive 全部延迟到 worker 内）。验收：
``tools/venv-python -m pytest tests/test_kpi_grouping.py -q``。
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from pipeline import eval_runner as er


# --------------------------------------------------------------------------- #
# Wilson CI
# --------------------------------------------------------------------------- #

def test_wilson_ci_reference_values() -> None:
    """与解析公式的既定数值对齐（±1e-4）。"""
    assert er.wilson_ci(0, 10) == pytest.approx((0.0, 0.27754), abs=1e-4)
    assert er.wilson_ci(10, 10) == pytest.approx((0.72246, 1.0), abs=1e-4)
    assert er.wilson_ci(5, 10) == pytest.approx((0.23659, 0.76341), abs=1e-4)
    assert er.wilson_ci(30, 100) == pytest.approx((0.21895, 0.39585), abs=1e-4)


def test_wilson_ci_bounds_and_center() -> None:
    """区间必须落在 [0,1]、包含点估计、且关于 k/n 对称（k 与 n-k 互换）。"""
    for k, n in ((0, 10), (3, 10), (7, 10), (30, 100), (99, 100), (1000, 1000)):
        lo, hi = er.wilson_ci(k, n)
        assert 0.0 <= lo <= k / n <= hi <= 1.0
        lo_swap, hi_swap = er.wilson_ci(n - k, n)
        assert lo == pytest.approx(1.0 - hi_swap, abs=1e-12)
        assert hi == pytest.approx(1.0 - lo_swap, abs=1e-12)


def test_wilson_ci_edge_cases() -> None:
    """n=0 / 非法 k → NaN，而不是抛异常或伪造区间。"""
    for k, n in ((0, 0), (1, 0), (-1, 10), (11, 10)):
        lo, hi = er.wilson_ci(k, n)
        assert math.isnan(lo) and math.isnan(hi)


# --------------------------------------------------------------------------- #
# 弱类 floor（config/eval.yaml::thresholds.per_category_success）
# --------------------------------------------------------------------------- #

def test_category_target_weak_and_non_weak() -> None:
    """弱类 target=max(baseline+0.15, 0.75)；非弱类 target=baseline-0.10。"""
    # 冻结基线的主标签 success（runs/baseline_eval/val_reference_by_primary.json）
    assert er.category_target(0.527) == pytest.approx(0.75)      # max(0.677, 0.75)
    assert er.category_target(0.648) == pytest.approx(0.798)
    assert er.category_target(0.692) == pytest.approx(0.842)
    assert er.category_target(0.802) == pytest.approx(0.702)
    assert er.category_target(0.846) == pytest.approx(0.746)
    # 边界：0.7 不算弱类（严格 < 0.7）
    assert er.category_target(0.7) == pytest.approx(0.6)
    assert er.category_target(0.699) == pytest.approx(0.849)  # max(0.699+0.15, 0.75)


# --------------------------------------------------------------------------- #
# primary 标签 / compound 分组
# --------------------------------------------------------------------------- #

def _spec(labels_geometry: str, geometry: list[str]) -> SimpleNamespace:
    return SimpleNamespace(labels={"geometry": labels_geometry}, geometry=list(geometry))


def test_primary_label_reads_labels_geometry() -> None:
    """primary = spec.labels['geometry']；缺失时退回 geometry[0] / unlabeled。"""
    assert er.primary_label(_spec("straight", ["straight", "straight"])) == "straight"
    assert er.primary_label(SimpleNamespace(labels={}, geometry=["curve", "straight"])) == "curve"
    assert er.primary_label(SimpleNamespace(labels={}, geometry=[])) == "unlabeled"
    assert er.primary_label(SimpleNamespace()) == "unlabeled"


def test_compound_spec_definition() -> None:
    """compound = 除 primary/straight 外还有其它几何标签（报告口径）。"""
    simple = _spec("straight", ["straight", "straight", "straight"])
    assert er.extra_geometry_labels(simple) == []
    assert er.is_compound_spec(simple) is False

    curve_extra = _spec("straight", ["straight", "curve", "straight"])
    assert er.extra_geometry_labels(curve_extra) == ["curve"]
    assert er.is_compound_spec(curve_extra) is True

    # 'straight' 是 ubiquitous 标签，不算 compound；primary 自身也不算
    assert er.is_compound_spec(_spec("curve", ["curve", "straight", "curve"])) is False
    assert er.extra_geometry_labels(
        _spec("intersection", ["straight", "curve", "intersection", "ramp_out", "roundabout"])
    ) == ["curve", "ramp_out", "roundabout"]


def test_group_episodes_by_field() -> None:
    episodes = [
        {"id": 0, "primary": "straight", "success": True},
        {"id": 1, "primary": "curve", "success": False},
        {"id": 2, "primary": "straight", "success": False},
        {"id": 3, "success": True},  # 缺字段 -> unlabeled
    ]
    groups = er.group_episodes(episodes, "primary")
    assert sorted(groups) == ["curve", "straight", "unlabeled"]
    assert [ep["id"] for ep in groups["straight"]] == [0, 2]
    assert groups["unlabeled"][0]["id"] == 3


# --------------------------------------------------------------------------- #
# 分组判定
# --------------------------------------------------------------------------- #

def _view(n: int, success_rate: float) -> dict:
    k = round(n * success_rate)
    return {"n": n, "success_rate": k / n, "success_wilson": list(er.wilson_ci(k, n))}


def test_judge_group_weak_floor_and_margin() -> None:
    # 弱类 baseline=0.527 -> target 0.75：0.90 过、0.70 不过
    passed = er.judge_group(_view(91, 0.90), 0.527)
    assert passed["judged"] and passed["passed"] and passed["weak"]
    assert passed["target"] == pytest.approx(0.75)
    failed = er.judge_group(_view(91, 0.70), 0.527)
    assert failed["judged"] and failed["passed"] is False

    # 非弱类 baseline=0.802 -> target 0.702
    non_weak = er.judge_group(_view(91, 0.71), 0.802)
    assert non_weak["judged"] and non_weak["passed"] and not non_weak["weak"]
    assert non_weak["target"] == pytest.approx(0.702)


def test_judge_group_n_min_and_missing_baseline() -> None:
    small = er.judge_group(_view(29, 0.99), 0.527)
    assert small["judged"] is False and small["passed"] is None
    assert "n=29" in small["reason"]

    at_min = er.judge_group(_view(30, 0.99), 0.527)
    assert at_min["judged"] is True

    missing = er.judge_group(_view(100, 0.99), None)
    assert missing["judged"] is False and missing["passed"] is None


# --------------------------------------------------------------------------- #
# summarize + 总判定
# --------------------------------------------------------------------------- #

def _episode(spec_id: int, primary: str, success: bool, *, compound: bool = False) -> dict:
    return {
        "id": spec_id,
        "primary": primary,
        "difficulty": "medium",
        "geometry": [primary, "straight"],
        "compound": compound,
        "success": success,
        "collision": not success,
        "off_road": False,
        "solid_line_crossing": not success,
        "speed_limit_violation": False,
        "termination": "arrive_dest" if success else "collision",
        "route_completion": 0.9 if success else 0.4,
        "steps": 300,
        "duration_s": 1.0,
        "min_ttc": 2.0 if success else 0.5,
        "a_lon_mean": 0.1,
        "a_lon_abs_p95": 1.0,
        "a_lat_mean": 0.0,
        "a_lat_abs_p95": 1.0,
        "jerk_mean": 0.0,
        "jerk_abs_p95": 2.0,
        "speed_ratio_mean": 0.8,
        "speed_ratio_p95": 1.0,
        "mean_speed_mps": 8.0,
        "_a_lon": [0.1, -0.2, 0.3],
        "_a_lat": [0.0, 0.1, -0.1],
        "_jerk": [0.0, 1.0, -1.0],
        "_speed_ratio": [0.8, 0.85, 0.75],
    }


def test_summarize_rates_and_wilson() -> None:
    episodes = [_episode(i, "straight", i % 2 == 0) for i in range(10)]
    view = er.summarize(episodes)
    assert view["n"] == 10
    assert view["success_rate"] == pytest.approx(0.5)
    lo, hi = view["success_wilson"]
    assert lo < 0.5 < hi
    assert view["route_completion_mean"] == pytest.approx(0.65)
    assert view["min_ttc_min"] == pytest.approx(0.5)
    assert view["a_lat_abs_p95"] > 0.0
    assert view["terminations"] == {"arrive_dest": 5, "collision": 5}
    assert er.summarize([]) == {"n": 0}


def test_evaluate_verdicts_against_frozen_baseline() -> None:
    """overall_success target = max(baseline-0.05, 0.70)；废弃缺参照时 passed=None。"""
    episodes = [_episode(i, "straight", True) for i in range(40)]
    overall = er.summarize(episodes)
    base_overall = {
        "success_rate": 0.748, "collision_rate": 0.16, "off_road_rate": 0.061,
        "route_completion_mean": 0.8734, "speed_ratio_mean": 0.7486,
        "a_lat_mean": -0.0504, "a_lat_abs_p95": 1.5322,
    }
    base_primary = {"straight": {"success": 0.802}}
    verdict = er.evaluate_verdicts(overall, {"straight": overall}, base_overall, base_primary)

    checks = verdict["checks"]
    assert checks["overall_success"]["target"] == pytest.approx(0.70)
    assert checks["overall_success"]["passed"] is True
    assert checks["collision_rate"]["target"] == pytest.approx(0.16)
    assert checks["collision_rate"]["passed"] is True   # 全 success -> collision_rate=0 <= 0.16
    assert checks["route_completion"]["passed"] is True
    assert checks["speed_ratio"]["passed"] is True
    assert checks["a_lat_p95"]["passed"] is True
    assert verdict["per_primary"]["straight"]["judged"] is True
    assert verdict["per_primary"]["straight"]["passed"] is True

    # 缺基线参照：判定为 None（不伪造 pass/fail）
    no_ref = er.evaluate_verdicts(overall, {"straight": overall}, None, None)
    assert no_ref["reference_available"] is False
    assert all(check["passed"] is None for check in no_ref["checks"].values())
    assert no_ref["per_primary"]["straight"]["judged"] is False
    assert no_ref["all_passed"] is False


def test_evaluate_verdicts_weak_group_failure() -> None:
    """弱类分组不达 floor 时 all_passed=False（即使整体指标全过）。"""
    episodes = [_episode(i, "ramp_out", i < 70) for i in range(100)]  # success 0.70 < 0.75
    by_primary = {"ramp_out": er.summarize(episodes)}
    base_overall = {
        "success_rate": 0.748, "collision_rate": 0.16, "off_road_rate": 0.061,
        "route_completion_mean": 0.8734, "speed_ratio_mean": 0.7486,
        "a_lat_mean": -0.05, "a_lat_abs_p95": 1.53,
    }
    verdict = er.evaluate_verdicts(
        by_primary["ramp_out"], by_primary, base_overall, {"ramp_out": {"success": 0.527}}
    )
    assert verdict["per_primary"]["ramp_out"]["weak"] is True
    assert verdict["per_primary"]["ramp_out"]["target"] == pytest.approx(0.75)
    assert verdict["per_primary"]["ramp_out"]["passed"] is False
    assert verdict["all_passed"] is False
