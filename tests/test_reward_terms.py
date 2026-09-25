"""reward_model 单元测试：全部用合成 dict 序列，不建 env、不 import metadrive。

覆盖契约 §3 验收点：
(a) 势能塑形不改变等终局轨迹的策略序（telescoping 不变性）；
(b) CaRL 式乘性/终止惩罚应用；
(c) KPI 主标签分组 + ``compound`` 桶 + Wilson CI + n>=30 / 弱类绝对 floor / overall 条款。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from reward_model import (
    AggregationConfig,
    CarlRule,
    KPI_NAMES,
    RewardAggregator,
    ShapingDecay,
    TermConfig,
    assign_credits,
    available_terms,
    build_terms,
    compute_kpis,
    compute_kpis_by_primary,
    discounted_return,
    episode_kpi,
    grouped_discounted_returns,
    make_term,
    wilson_ci,
)

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TERMS = {
    "crash",
    "out_of_road",
    "comfort_lon",
    "comfort_lat",
    "comfort_jerk",
    "speed_ratio",
    "solid_line",
    "speed_limit",
    "route_completion",
}


# --------------------------------------------------------------------------- #
# 注册表 / 项级行为
# --------------------------------------------------------------------------- #

def test_registry_lists_all_contract_terms() -> None:
    assert set(available_terms()) == EXPECTED_TERMS


def test_registry_unknown_term_and_flat_params() -> None:
    with pytest.raises(KeyError):
        make_term("not_a_term")
    terms = build_terms(
        [
            {"name": "comfort_lon", "weight": -0.1, "deadband": 1.0},
            {"name": "crash", "weight": -3.0, "enabled": False},
            TermConfig("speed_ratio", weight=2.0, params={"cap": 0.5}),
        ]
    )
    assert [term.name for term in terms] == ["comfort_lon", "speed_ratio"]
    assert terms[0].compute({"a_lon": 2.0}) == pytest.approx(1.0)  # (2-1)^2
    assert terms[1].compute({"speed_ratio": 0.9}) == pytest.approx(0.5)  # clip 到 cap


def test_crash_and_off_road_flags() -> None:
    crash = make_term("crash")
    assert crash.compute({"crash": True}) == 1.0
    assert crash.compute({"crash_vehicle": True}) == 1.0
    assert crash.compute({"collision": True}) == 1.0
    assert crash.compute({}) == 0.0
    assert crash.terminating and crash.reason == "collision"
    out = make_term("out_of_road")
    assert out.compute({"out_of_road": True}) == 1.0
    assert out.compute({}) == 0.0
    assert out.terminating and out.reason == "out_of_road"


def test_comfort_hinge_deadbands() -> None:
    lon = make_term("comfort_lon", deadband=2.5)
    assert lon.compute({"a_lon": 1.0}) == 0.0
    assert lon.compute({"a_lon": -3.5}) == pytest.approx(1.0)
    lat = make_term("comfort_lat", deadband=2.0)
    assert lat.compute({"a_lat": 3.0}) == pytest.approx(1.0)
    jerk = make_term("comfort_jerk", deadband=5.0)
    assert jerk.compute({"jerk": 5.0}) == 0.0
    assert jerk.compute({"jerk": 7.0}) == pytest.approx(4.0)


def test_speed_ratio_efficiency_term() -> None:
    term = make_term("speed_ratio", cap=1.0)
    assert term.compute({"speed": 4.0, "speed_limit_mps": 8.0}) == pytest.approx(0.5)
    assert term.compute({"speed_ratio": 1.25}) == pytest.approx(1.0)  # 超速不奖励
    capped = make_term("speed_ratio", cap=1.5)
    assert capped.compute({"speed_ratio": 1.25}) == pytest.approx(1.25)


def test_solid_line_flags_and_line_types() -> None:
    term = make_term("solid_line")
    assert term.compute({"on_white_continuous_line": True}) == 1.0
    assert term.compute({"on_yellow_continuous_line": True}) == 1.0
    assert term.compute({"solid_line_crossing": True}) == 1.0
    assert term.compute({"left_line_type_id": 2}) == 1.0  # 白实线
    assert term.compute({"right_line_type_id": 6}) == 1.0  # 黄实线
    assert term.compute({"left_line_type_id": 1}) == 0.0  # 虚线
    assert term.compute({}) == 0.0


def test_speed_limit_violation_penalty() -> None:
    term = make_term("speed_limit", tolerance=0.05)
    assert term.compute({"speed": 8.0, "speed_limit_mps": 8.0}) == 0.0
    assert term.compute({"speed": 10.0, "speed_limit_mps": 8.0}) == pytest.approx(0.2)
    assert term.compute({"speed_ratio": 1.0}) == 0.0


def test_route_completion_term_telescopes() -> None:
    term = make_term("route_completion", gamma=1.0)
    total = 0.0
    previous = 0.0
    for rc in (0.0, 0.25, 0.5, 1.0):
        total += term.compute({"route_completion": rc, "route_completion_prev": previous})
        previous = rc
    assert total == pytest.approx(1.0)  # Φ_T − Φ_0


# --------------------------------------------------------------------------- #
# (a) 势能塑形：不改变等终局轨迹的策略序
# --------------------------------------------------------------------------- #

GAMMA = 0.9


def _roll(rcs: list[float], dense: list[float], *, with_shaping: bool) -> float:
    """用聚合器滚一条轨迹，返回折扣总回报；dense 为与塑形无关的稠密收益。"""
    terms = [make_term("route_completion", gamma=GAMMA)] if with_shaping else []
    aggregator = RewardAggregator(
        terms, AggregationConfig(terminal_values={}, carl_rules={})
    )
    rewards: list[float] = []
    for index, rc in enumerate(rcs):
        ctx: dict[str, object] = {"route_completion": rc}
        if index == len(rcs) - 1:
            ctx["arrive_dest"] = True
        rewards.append(aggregator.step(ctx).reward + dense[index])
    return discounted_return(rewards, GAMMA)


def test_potential_shaping_preserves_policy_order() -> None:
    rc_a = [0.0, 0.1, 0.4, 0.9]
    rc_b = [0.0, 0.6, 0.85, 0.9]
    dense_a = [1.0, 1.0, 1.0, 1.0]
    dense_b = [0.9, 0.9, 0.9, 0.9]

    a_with = _roll(rc_a, dense_a, with_shaping=True)
    b_with = _roll(rc_b, dense_b, with_shaping=True)
    a_without = _roll([0.0] * 4, dense_a, with_shaping=False)
    b_without = _roll([0.0] * 4, dense_b, with_shaping=False)

    # 无塑形时 A 优；加塑形后序不变（等终局、等长 => 塑形总量相同）
    assert a_without > b_without
    assert a_with > b_with
    shaping_total = GAMMA**4 * 0.9  # γ^T Φ_T − Φ_0
    assert a_with - a_without == pytest.approx(shaping_total)
    assert b_with - b_without == pytest.approx(shaping_total)


# --------------------------------------------------------------------------- #
# (b) CaRL 式乘性 / 终止惩罚
# --------------------------------------------------------------------------- #

def test_carl_multiplier_and_terminating_penalty() -> None:
    terms = build_terms(
        [{"name": "speed_ratio", "weight": 1.0}, {"name": "crash", "weight": -10.0}]
    )
    aggregator = RewardAggregator(
        terms,
        AggregationConfig(
            terminal_values={"collision": -5.0},
            carl_rules={"crash": CarlRule(factor=0.0, terminate=True)},
        ),
    )
    normal = aggregator.step({"speed_ratio": 0.5})
    assert normal.reward == pytest.approx(0.5)
    assert normal.carl_multiplier == 1.0 and not normal.done

    crashed = aggregator.step({"speed_ratio": 1.0, "crash_vehicle": True})
    assert crashed.carl_multiplier == 0.0
    assert crashed.dense_sum == pytest.approx(1.0)
    assert crashed.components["crash"] == pytest.approx(-10.0)
    assert crashed.terminal_value == pytest.approx(-5.0)
    assert crashed.reward == pytest.approx(-15.0)
    assert crashed.done and crashed.reason == "collision"


def test_carl_partial_factor_and_penalty_only_rule() -> None:
    partial = RewardAggregator(
        build_terms([{"name": "speed_ratio", "weight": 1.0}]),
        AggregationConfig(
            terminal_values={},
            carl_rules={"crash": CarlRule(factor=0.25, terminate=False)},
        ),
    )
    step = partial.step({"speed_ratio": 1.0, "crash": True})
    assert step.reward == pytest.approx(0.25)
    assert step.done  # 碰撞按协议仍是终局

    penalty_only = RewardAggregator(
        [],
        AggregationConfig(
            terminal_values={},
            carl_rules={"out_of_road": CarlRule(factor=None, terminate=True, penalty=-3.0)},
        ),
    )
    out = penalty_only.step({"out_of_road": True})
    assert out.reward == pytest.approx(-3.0)
    assert out.done and out.reason == "out_of_road"


def test_terminal_values_arrival_and_timeout() -> None:
    config = AggregationConfig(
        terminal_values={"arrive_dest": 10.0, "max_step": -2.0, "collision": -5.0},
        carl_rules={},
    )
    aggregator = RewardAggregator([], config)
    arrived = aggregator.step({"arrive_dest": True})
    assert arrived.reward == pytest.approx(10.0)
    assert arrived.done and arrived.reason == "arrive_dest"
    aggregator.reset()
    timeout = aggregator.step({"max_step": True})
    assert timeout.reward == pytest.approx(-2.0)
    assert timeout.done and timeout.reason == "max_step"


# --------------------------------------------------------------------------- #
# shaping decay + 信用分配
# --------------------------------------------------------------------------- #

def test_shaping_decay_schedule_and_aggregation() -> None:
    decay = ShapingDecay(kind="linear", start=1.0, end=0.0, steps=10)
    assert decay(0) == pytest.approx(1.0)
    assert decay(5) == pytest.approx(0.5)
    assert decay(10) == pytest.approx(0.0)

    aggregator = RewardAggregator(
        [make_term("route_completion", gamma=1.0)],
        AggregationConfig(terminal_values={}, carl_rules={}, shaping_decay=decay),
    )
    step = aggregator.step({"route_completion": 0.5}, step_index=5)
    assert step.shaping_decay == pytest.approx(0.5)
    assert step.reward == pytest.approx(0.25)  # 0.5 * 0.5
    assert step.components["route_completion"] == pytest.approx(0.25)


def test_credit_modes() -> None:
    rewards = [1.0, 2.0, 3.0, 4.0]
    assert assign_credits(rewards, mode="dense") == rewards

    returns = grouped_discounted_returns(rewards, gamma=0.9, group_size=2)
    assert returns[1] == pytest.approx(6.6)  # 3 + 0.9*4
    assert returns[0] == pytest.approx(1 + 0.9 * 2 + 0.9**2 * returns[1])
    grouped = assign_credits(rewards, mode="grouped_discounted", gamma=0.9, group_size=2)
    assert grouped == pytest.approx([returns[0], returns[0], returns[1], returns[1]])

    ragged = grouped_discounted_returns([1.0, 2.0, 3.0], gamma=0.9, group_size=2)
    assert ragged[1] == pytest.approx(3.0)
    assert ragged[0] == pytest.approx(1 + 0.9 * 2 + 0.9**2 * 3.0)

    with pytest.raises(ValueError):
        assign_credits(rewards, mode="unknown")
    with pytest.raises(ValueError):
        grouped_discounted_returns(rewards, group_size=0)


# --------------------------------------------------------------------------- #
# (c) KPI：口径、Wilson CI、分组与判定
# --------------------------------------------------------------------------- #

def test_wilson_ci() -> None:
    low, high = wilson_ci(75, 100)
    assert low == pytest.approx(0.65695, abs=1e-3)
    assert high == pytest.approx(0.82455, abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 0.0)
    low, high = wilson_ci(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)
    low, high = wilson_ci(30, 30)
    assert high == 1.0 and low < 1.0


def _synthetic_episode() -> dict:
    return {
        "termination": "arrive_dest",
        "route_completion": 0.9,
        "steps": [
            {"speed": 4.0, "speed_limit_mps": 8.0, "heading": 0.0, "ttc": 4.0},
            {"speed": 6.0, "speed_limit_mps": 8.0, "heading": 0.0, "ttc": 2.5},
            {"speed": 8.0, "speed_limit_mps": 8.0, "heading": math.pi / 2,
             "on_white_continuous_line": True},
            {"speed": 9.0, "speed_limit_mps": 8.0, "heading": math.pi / 2, "ttc": 2.5},
        ],
    }


def test_episode_kpi_derived_quantities() -> None:
    record = episode_kpi(_synthetic_episode(), dt=0.5)
    assert record["success"] is True and record["collision"] is False
    assert record["route_completion"] == pytest.approx(0.9)
    assert record["min_ttc"] == pytest.approx(2.5)
    assert record["a_lon"] == pytest.approx([4.0, 4.0, 2.0])
    assert record["a_lat"][0] == pytest.approx(0.0)
    assert record["a_lat"][1] == pytest.approx(7.0 * math.pi)
    assert record["jerk"] == pytest.approx([0.0, -4.0])
    assert record["speed_ratio"] == pytest.approx([0.5, 0.75, 1.0, 1.125])
    assert record["solid_line_crossing"] is True
    assert record["speed_limit_violation"] is True
    assert record["speed_limit_violation_count"] == 1
    assert record["speed_limit_violation_rate"] == pytest.approx(0.25)


def test_compute_kpis_pooled_metrics() -> None:
    summary = compute_kpis([_synthetic_episode()], dt=0.5)
    assert summary["n"] == 1
    assert summary["success_rate"] == 1.0
    assert summary["speed_ratio_mean"] == pytest.approx(0.84375)
    assert summary["a_lon_mean"] == pytest.approx(10.0 / 3.0)
    assert summary["a_lon_abs_p95"] == pytest.approx(4.0)
    assert summary["jerk_abs_p95"] == pytest.approx(3.8)
    assert summary["solid_line_crossing_rate"] == 1.0
    assert summary["speed_limit_violation_episode_rate"] == 1.0
    assert summary["route_completion_mean"] == pytest.approx(0.9)
    assert summary["terminations"] == {"arrive_dest": 1}


def _labeled_episode(
    label: str,
    success: bool,
    *,
    control: str = "none",
    maneuver: tuple[str, ...] = ("straight",),
    geometry: tuple[str, ...] | None = None,
) -> dict:
    return {
        "labels": {
            "geometry": label,
            "traffic": [],
            "control": control,
            "maneuver": list(maneuver),
        },
        "geometry": list(geometry if geometry is not None else (label,)),
        "success": success,
        "collision": False,
        "off_road": False,
        "route_completion": 0.9,
        "termination": "arrive_dest" if success else "max_step",
    }


def test_kpi_primary_grouping_compound_and_overall_clause() -> None:
    straight = [_labeled_episode("straight", index < 30) for index in range(40)]
    ramp = [_labeled_episode("ramp_out", index < 5) for index in range(10)]
    compound = [
        _labeled_episode(
            "curve",
            False,
            control="cut_in",
            maneuver=("left",),
            geometry=("curve", "t_intersection"),
        )
        for _ in range(5)
    ]
    report = compute_kpis_by_primary(
        straight + ramp + compound,
        baselines={"straight": 0.802, "ramp_out": 0.527, "overall": 0.748},
    )

    # 主标签分组（n>=30 + 相对口径）
    straight_group = report["by_primary"]["straight"]
    assert straight_group["n"] == 40
    judgment = straight_group["judgment"]
    assert judgment["judged"] is True and judgment["rule"] == "relative"
    assert judgment["target"] == pytest.approx(0.702)
    assert judgment["passed"] is True  # 30/40 = 0.75
    assert judgment["wilson_ci"]["n"] == 40

    # n<30 只报告 + Wilson CI
    ramp_judgment = report["by_primary"]["ramp_out"]["judgment"]
    assert ramp_judgment["judged"] is False
    assert ramp_judgment["rule"] == "n_below_min"
    assert ramp_judgment["wilson_ci"]["lo"] > 0.0

    # 附带标签进 compound，单独报告、不判定
    assert report["compound"]["n"] == 5
    assert report["compound"]["judgment"]["judged"] is False
    assert report["compound"]["judgment"]["rule"] == "compound_report_only"
    assert report["by_primary"]["curve"]["n"] == 5

    # overall_success 条款：rate=35/55，baseline 0.748 -> target 0.70
    overall = report["overall"]
    assert overall["n"] == 55
    assert overall["judgment"]["clause"] == "overall_success"
    assert overall["judgment"]["target"] == pytest.approx(0.70)
    assert overall["judgment"]["passed"] is False


def test_kpi_weak_class_absolute_floor() -> None:
    episodes = [_labeled_episode("ramp_out", index < 28) for index in range(40)]
    report = compute_kpis_by_primary(episodes, baselines={"ramp_out": 0.527})
    judgment = report["by_primary"]["ramp_out"]["judgment"]
    assert judgment["judged"] is True
    assert judgment["rule"] == "weak_floor"
    assert judgment["target"] == pytest.approx(0.75)  # max(0.527+0.15, 0.75)
    assert judgment["passed"] is False  # 0.70 < 0.75


def test_kpi_overall_success_pass_and_no_baseline() -> None:
    passing = [_labeled_episode("straight", index < 72) for index in range(100)]
    report = compute_kpis_by_primary(passing, baselines={"overall": 0.748})
    assert report["overall"]["judgment"]["passed"] is True  # 0.72 >= 0.70
    assert report["by_primary"]["straight"]["judgment"]["rule"] == "no_baseline"
    assert report["by_primary"]["straight"]["judgment"]["judged"] is False

    failing = [_labeled_episode("straight", index < 68) for index in range(100)]
    report = compute_kpis_by_primary(failing, baselines={"overall": 0.748})
    assert report["overall"]["judgment"]["passed"] is False  # 0.68 < 0.70


def test_eval_yaml_kpi_names_match() -> None:
    config = yaml.safe_load((ROOT / "config" / "eval.yaml").read_text(encoding="utf-8"))
    assert tuple(config["kpis"]) == KPI_NAMES
