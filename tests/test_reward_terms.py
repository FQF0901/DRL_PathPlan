"""reward_model 单元测试：全部用合成 dict 序列，不建 env、不 import metadrive。

覆盖契约 §3 验收点：
(a) 势能塑形不改变等终局轨迹的策略序（telescoping 不变性）；
(b) CaRL 式乘性/终止惩罚应用；
(c) KPI 主标签分组 + ``compound`` 桶 + Wilson CI + n>=30 / 弱类绝对 floor / overall 条款。
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest
import yaml

from reward_model import (
    AggregationConfig,
    CarlRule,
    DEFAULT_TERM_CONFIGS,
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
    "lane_center",
    "lane_boundary",
    "ttc",
    "lead_gap",
    "low_speed",
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


def test_lane_center_deadband_clamp_and_default_off() -> None:
    """lane_center 数学：死区 0.25 / 截断 3.0 / 权重 -0.1；默认不在 DEFAULT_TERM_CONFIGS。"""
    term = make_term("lane_center", weight=-0.1)
    assert term.weight == pytest.approx(-0.1)
    assert term.deadband == pytest.approx(0.25)
    assert term.clamp == pytest.approx(3.0)
    for d_lat in (0.0, 0.1, 0.25, -0.25):
        assert term.compute({"d_lat": d_lat}) == 0.0
    assert term.compute({"d_lat": 1.25}) == pytest.approx(1.0)  # |1.25| - 0.25
    assert term.compute({"d_lat": -1.25}) == pytest.approx(1.0)
    assert term.compute({"d_lat": 3.0}) == pytest.approx(2.75)  # clamp 3 m
    assert term.compute({"d_lat": 10.0}) == pytest.approx(2.75)
    assert term.weight * term.compute({"d_lat": 10.0}) == pytest.approx(-0.275)
    # 单调不增（越偏越差）；死区边界取 0 起点
    rewards = [term.weight * term.compute({"d_lat": value}) for value in (0.25, 0.5, 1.0, 2.0, 3.0, 10.0)]
    assert all(later <= earlier for earlier, later in zip(rewards, rewards[1:]))
    # 默认关（不在默认 term 列表）
    assert "lane_center" not in {config["name"] for config in DEFAULT_TERM_CONFIGS}


def test_lane_center_boundary_fallback_and_no_lane() -> None:
    term = make_term("lane_center", weight=-0.1)
    # 回退：最近左右车道边界距离 → d_lat=(left-right)/2；|0.75| 超死区 0.5
    assert term.compute({"dist_to_left_side": 2.75, "dist_to_right_side": 1.25}) == pytest.approx(0.5)
    # 交替键名等价
    assert term.compute({"lateral_offset": 0.75}) == pytest.approx(0.5)
    assert term.compute({"lane_lateral_offset": 0.75}) == pytest.approx(0.5)
    assert term.compute({"dist_to_left_side": 1.5, "dist_to_right_side": 1.5}) == 0.0
    # 无车道信息 / 键非法 → 0，不崩溃
    assert term.compute({}) == 0.0
    assert term.compute({"dist_to_left_side": 2.0}) == 0.0
    assert term.compute({"d_lat": None}) == 0.0
    assert term.compute({"d_lat": "bad"}) == 0.0


def test_lane_center_enablement_via_term_weight_override() -> None:
    """``build_reward_adapter(term_weights={"lane_center": -0.1})`` = 追加启用（默认项不动）。"""
    from pipeline.trainer import build_reward_adapter

    before = [dict(term) for term in DEFAULT_TERM_CONFIGS]
    adapter, source = build_reward_adapter(term_weights={"lane_center": -0.1})
    assert source.endswith("+term_weights+append(lane_center)")
    aggregator = adapter.factory()
    weights = {term.name: term.weight for term in aggregator.terms}
    assert weights["lane_center"] == pytest.approx(-0.1)
    assert weights["route_completion"] == 1.0
    # 聚合正确：lane_center 贡献 = weight × raw（无横向键时 0）
    step = aggregator.step({"speed_ratio": 0.5, "d_lat": 1.25})
    assert step.components["lane_center"] == pytest.approx(-0.1)
    assert step.components["speed_ratio"] == pytest.approx(0.2)  # v5：权重 0.4
    assert step.reward == pytest.approx(0.1)
    centered = aggregator.step({"speed_ratio": 0.5, "d_lat": 0.1})
    assert centered.components["lane_center"] == 0.0
    # 未配置且未注册的名字仍 fail-fast；默认配置不被污染
    with pytest.raises(ValueError, match="未命中奖励项"):
        build_reward_adapter(term_weights={"not_a_term": 1.0})
    assert [dict(term) for term in DEFAULT_TERM_CONFIGS] == before


def test_ttc_no_lead_missing_keys_and_warn_once() -> None:
    """ttc：无前车 → 0（不告警）；缺键 → 0 且只告警一次。"""
    term = make_term("ttc", weight=-0.5)
    assert term.weight == pytest.approx(-0.5)
    assert term.ttc_threshold == pytest.approx(2.0)
    assert term.ttc_floor == pytest.approx(0.5)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # PP/IDM 约定：非正净距（含 −1 哨兵）= 无前车，属正常语义
        assert term.compute({"lead_gap_m": -1.0, "lead_speed_mps": 3.0, "velocity": 6.0}) == 0.0
        assert term.compute({"lead_gap_m": 0.0, "lead_speed_mps": 3.0, "velocity": 6.0}) == 0.0
        assert caught == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            assert term.compute({"velocity": 6.0}) == 0.0  # 两个前车键都缺
        assert term.compute({"lead_gap_m": 10.0, "lead_speed_mps": 2.0}) == 0.0  # 自车速度缺
        assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)
    with pytest.raises(ValueError):
        make_term("ttc", ttc_threshold=0.0)
    with pytest.raises(ValueError):
        make_term("ttc", ttc_floor=-1.0)
    # 默认关
    assert "ttc" not in {config["name"] for config in DEFAULT_TERM_CONFIGS}


def test_ttc_boundary_continuity_monotone_and_floor() -> None:
    """ttc：阈值处连续为 0、随 gap 单调、ttc_floor 封顶、未接近 → 0。"""
    term = make_term("ttc", weight=-0.5)
    # ttc = 2.0（gap=20, v_rel=10）恰在阈值 → 0；更安全恒为 0
    assert term.compute({"lead_gap_m": 20.0, "lead_speed_mps": 0.0, "velocity": 10.0}) == 0.0
    assert term.compute({"lead_gap_m": 200.0, "lead_speed_mps": 0.0, "velocity": 10.0}) == 0.0
    # 跨阈值连续：ttc 略小于 2 → 罚从 0 连续上升
    just_below = term.compute(
        {"lead_gap_m": 20.0 - 1e-3, "lead_speed_mps": 0.0, "velocity": 10.0}
    )
    assert 0.0 < just_below < 1e-3
    # 阈值内解析值：ttc=1 → raw = 1/1 − 1/2 = 0.5（加权 −0.25）
    assert term.compute({"lead_gap_m": 10.0, "lead_speed_mps": 0.0, "velocity": 10.0}) == pytest.approx(0.5)
    # 随 gap 单调不增
    raws = [
        term.compute({"lead_gap_m": gap, "lead_speed_mps": 0.0, "velocity": 10.0})
        for gap in (5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0)
    ]
    assert all(later <= earlier for earlier, later in zip(raws, raws[1:]))
    # ttc_floor=0.5s 封顶：ttc<=0.5 → raw = 2 − 0.5 = 1.5（单步加权 −0.75）
    assert term.compute({"lead_gap_m": 5.0, "lead_speed_mps": 0.0, "velocity": 10.0}) == pytest.approx(1.5)
    assert term.compute({"lead_gap_m": 0.1, "lead_speed_mps": 0.0, "velocity": 10.0}) == pytest.approx(1.5)
    # 未接近（v_ego <= v_lead）→ 0；speed 别名与 velocity 等价
    assert term.compute({"lead_gap_m": 5.0, "lead_speed_mps": 12.0, "velocity": 10.0}) == 0.0
    assert term.compute({"lead_gap_m": 5.0, "lead_speed_mps": 10.0, "velocity": 10.0}) == 0.0
    assert term.compute({"lead_gap_m": 10.0, "lead_speed_mps": 0.0, "speed": 10.0}) == pytest.approx(0.5)


def test_ttc_enablement_and_aggregation_components() -> None:
    """启用 ttc（默认项 + 追加项）时 components / reward 数值正确。"""
    configs = [dict(config) for config in DEFAULT_TERM_CONFIGS]
    configs.append({"name": "ttc", "weight": -0.5, "ttc_threshold": 2.0})
    aggregator = RewardAggregator(build_terms(configs))
    step = aggregator.step(
        {
            "speed_ratio": 0.5,
            "route_completion": 0.1,
            "lead_gap_m": 10.0,
            "lead_speed_mps": 0.0,
            "velocity": 10.0,
        }
    )
    assert step.raw_components["ttc"] == pytest.approx(0.5)
    assert step.components["ttc"] == pytest.approx(-0.25)
    # v5：speed_ratio 权重 0.4；low_speed 在 ctx 无速度键时为 0（只告警一次，不影响数值）
    assert step.reward == pytest.approx(0.4 * 0.5 + 0.1 - 0.25)


def test_lead_gap_boundary_monotone_and_default_off() -> None:
    """lead_gap（§7.5 候选 B）：gap_ref 处连续为 0、随 gap 单调、无需速度键、默认关。"""
    term = make_term("lead_gap", weight=-1.0, gap_ref=6.0, cap=1.0)
    assert term.gap_ref == pytest.approx(6.0)
    assert term.cap == pytest.approx(1.0)
    # 边界连续：gap = gap_ref → raw 恰为 0；更大恒为 0
    assert term.compute({"lead_gap_m": 6.0}) == 0.0
    assert term.compute({"lead_gap_m": 30.0}) == 0.0
    # 单调：gap 越小 raw 越大（不需要 velocity / lead_speed 键——与 ttc 的语义区别）
    raws = [term.compute({"lead_gap_m": gap}) for gap in (5.0, 4.0, 3.0, 2.0, 1.0)]
    assert all(later >= earlier for earlier, later in zip(raws, raws[1:]))
    assert raws[0] == pytest.approx(1.0 / 6.0)  # (6−5)/6
    assert raws[-1] == pytest.approx(5.0 / 6.0)
    # gap → 0 逼近 raw → 1（cap 封顶）；给出 speed 别名也不改变数值（速度无关）
    assert term.compute({"lead_gap_m": 0.1, "velocity": 10.0}) == pytest.approx(5.9 / 6.0)
    assert term.compute({"lead_gap_m": 0.1, "speed": 10.0}) == pytest.approx(5.9 / 6.0)
    # 无前车（含 −1 哨兵）→ 0 且不告警
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert term.compute({"lead_gap_m": -1.0}) == 0.0
        assert term.compute({"lead_gap_m": 0.0}) == 0.0
        assert caught == []
    # 缺键 / 非有限值 → 0 且只告警一次
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            assert term.compute({"velocity": 10.0}) == 0.0
        assert term.compute({"lead_gap_m": float("nan")}) == 0.0
        assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)
    with pytest.raises(ValueError):
        make_term("lead_gap", gap_ref=0.0)
    with pytest.raises(ValueError):
        make_term("lead_gap", cap=0.0)
    # 默认关
    assert "lead_gap" not in {config["name"] for config in DEFAULT_TERM_CONFIGS}


def test_lane_boundary_margin_threshold_and_default_off() -> None:
    """lane_boundary：居中 → 0、贴线 → 罚、压线封顶、单调、默认关（与 lane_center 互补）。"""
    term = make_term("lane_boundary", weight=-0.2)
    assert term.margin_threshold == pytest.approx(0.5)
    half = 1.75  # 3.5 m 车道
    # 余量 >= 阈值（|d_lat| <= 1.25）→ 0：车道内任意合法偏移不罚
    for d_lat in (0.0, 0.5, -1.25, 1.25):
        assert term.compute({"d_lat": d_lat, "lane_half_width_m": half}) == 0.0
    # 贴线：|d_lat| = 1.6 → 余量 0.15 → raw = 0.35 → 加权 −0.07
    assert term.compute({"d_lat": 1.6, "lane_half_width_m": half}) == pytest.approx(0.35)
    assert term.compute({"d_lat": -1.6, "lane_half_width_m": half}) == pytest.approx(0.35)
    assert term.weight * term.compute({"d_lat": 1.6, "lane_half_width_m": half}) == pytest.approx(-0.07)
    # 压线/越界：margin <= 0 → raw 封顶在 margin_threshold（单步最差 −0.1）
    assert term.compute({"d_lat": half, "lane_half_width_m": half}) == pytest.approx(0.5)
    assert term.compute({"d_lat": 5.0, "lane_half_width_m": half}) == pytest.approx(0.5)
    # 单调不减（越贴线 raw 越大；压线后封顶）
    raws = [
        term.compute({"d_lat": value, "lane_half_width_m": half})
        for value in (1.0, 1.25, 1.4, 1.6, 1.75, 3.0)
    ]
    assert all(later >= earlier for earlier, later in zip(raws, raws[1:]))
    # 默认关 + 参数非法 fail-fast
    assert "lane_boundary" not in {config["name"] for config in DEFAULT_TERM_CONFIGS}
    with pytest.raises(ValueError):
        make_term("lane_boundary", margin_threshold=0.0)


def test_lane_boundary_missing_keys_fallback_and_warn_once() -> None:
    """lane_boundary：缺键 → 0 且只告警一次；lane_half_width_m 可由左右边界距离回退。"""
    term = make_term("lane_boundary", weight=-0.2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            assert term.compute({"speed": 5.0}) == 0.0  # d_lat 与半宽都缺
        assert term.compute({"d_lat": 1.6}) == 0.0  # 有横向偏差但缺半宽
        assert term.compute({"lane_half_width_m": 1.75}) == 0.0  # 有半宽但缺 d_lat
        assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)
    # 回退：d_lat=(2.75−1.25)/2=0.75、half=(2.75+1.25)/2=2.0 → 余量 1.25 → 0（车道内）
    assert term.compute({"dist_to_left_side": 2.75, "dist_to_right_side": 1.25}) == 0.0
    # 显式半宽非正 → 走回退；d_lat=−1.7、half=2.0 → 余量 0.3 → raw 0.2
    assert term.compute(
        {"d_lat": -1.7, "lane_half_width_m": 0.0, "dist_to_left_side": 3.7, "dist_to_right_side": 0.3}
    ) == pytest.approx(0.2)
    # 非法 d_lat（None / 非数值）→ 0
    assert term.compute({"d_lat": None, "lane_half_width_m": 1.75}) == 0.0
    assert term.compute({"d_lat": "bad", "lane_half_width_m": 1.75}) == 0.0


def test_new_dense_terms_registered_and_default_list_frozen() -> None:
    """新项已注册（可经 config/CLI 启用）；DEFAULT_TERM_CONFIGS 逐项快照不变。"""
    assert {"ttc", "lane_boundary", "lead_gap"} <= set(available_terms())
    assert [dict(config) for config in DEFAULT_TERM_CONFIGS] == [
        {"name": "route_completion", "weight": 1.0, "gamma": 1.0},
        {"name": "speed_ratio", "weight": 0.4, "cap": 1.0},  # v5 §2
        {"name": "low_speed", "weight": -0.2},  # v5 §4 默认启用
        {"name": "comfort_lon", "weight": -0.05, "deadband": 2.5},
        {"name": "comfort_lat", "weight": -0.05, "deadband": 2.0},
        {"name": "comfort_jerk", "weight": -0.005, "deadband": 5.0},
        {"name": "solid_line", "weight": -2.0},
        {"name": "speed_limit", "weight": -5.0, "tolerance": 0.05},
        {"name": "crash", "weight": -10.0},
        {"name": "out_of_road", "weight": -8.0},
    ]


def test_new_terms_enablement_via_term_weight_override_and_aggregation() -> None:
    """``--reward-term-weight ttc=... lane_boundary=...`` 追加启用 + 聚合 components 数值。"""
    from pipeline.trainer import build_reward_adapter

    before = [dict(term) for term in DEFAULT_TERM_CONFIGS]
    adapter, source = build_reward_adapter(term_weights={"ttc": -0.5, "lane_boundary": -0.2})
    assert source.endswith("+term_weights+append(ttc,lane_boundary)")
    aggregator = adapter.factory()
    weights = {term.name: term.weight for term in aggregator.terms}
    assert weights["ttc"] == pytest.approx(-0.5)
    assert weights["lane_boundary"] == pytest.approx(-0.2)
    assert weights["route_completion"] == 1.0

    step = aggregator.step(
        {
            "speed_ratio": 0.5,
            "route_completion": 0.2,
            "lead_gap_m": 10.0,
            "lead_speed_mps": 0.0,
            "velocity": 10.0,
            "d_lat": 1.6,
            "lane_half_width_m": 1.75,
        }
    )
    assert step.raw_components["ttc"] == pytest.approx(0.5)
    assert step.components["ttc"] == pytest.approx(-0.25)
    assert step.raw_components["lane_boundary"] == pytest.approx(0.35)
    assert step.components["lane_boundary"] == pytest.approx(-0.07)
    assert step.reward == pytest.approx(0.4 * 0.5 + 0.2 - 0.25 - 0.07)
    assert [dict(term) for term in DEFAULT_TERM_CONFIGS] == before


def test_low_speed_boundary_continuity_and_magnitude() -> None:
    """v5 §4：``v<2`` 时 ``−0.2×(1−v/2)``；边界连续、单调、封顶；阈值与 KPI 对齐。"""
    from pipeline.eval_runner import CRAWL_SPEED_MPS
    from reward_model.terms import LOW_SPEED_THRESHOLD_MPS

    from reward_model import get_term_class

    term = make_term("low_speed", weight=-0.2)  # 默认权重 −0.2 / 阈值 2.0
    assert get_term_class("low_speed")().weight == pytest.approx(-0.2)  # 类默认值
    assert term.weight == pytest.approx(-0.2)
    assert term.threshold == pytest.approx(2.0)
    assert LOW_SPEED_THRESHOLD_MPS == pytest.approx(CRAWL_SPEED_MPS)  # 与评测 KPI 同阈值

    # 规格样例：v=0 → −0.2；v=1 → −0.1；v=2 → 0；v>2 → 0
    assert term.weight * term.compute({"speed": 0.0}) == pytest.approx(-0.2)
    assert term.weight * term.compute({"speed": 1.0}) == pytest.approx(-0.1)
    assert term.weight * term.compute({"speed": 2.0}) == pytest.approx(0.0)
    assert term.weight * term.compute({"speed": 2.5}) == pytest.approx(0.0)
    assert term.weight * term.compute({"velocity": 5.0}) == pytest.approx(0.0)
    # 连续性：阈值两侧 raw 从 0 连续上升；阈值下界夹到 0（不产生正奖励）
    assert term.compute({"speed": 2.0 - 1e-9}) == pytest.approx(0.0, abs=1e-8)
    assert term.compute({"speed": 1.0 + 1e-9}) == pytest.approx(0.5, abs=1e-8)
    # 单调：v 越小 raw 越大；v<0 封顶 1.0（加权不超权重）
    raws = [term.compute({"speed": v}) for v in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)]
    assert all(later <= earlier for earlier, later in zip(raws, raws[1:]))
    assert term.compute({"speed": -3.0}) == pytest.approx(1.0)
    # speed 与 velocity 同口径
    assert term.compute({"speed": 0.6}) == pytest.approx(term.compute({"velocity": 0.6}))
    # 非法阈值 fail-fast
    with pytest.raises(ValueError):
        make_term("low_speed", threshold=0.0)


def test_low_speed_missing_key_warn_once_and_default_enabled() -> None:
    """缺速度键 → 0 且只告警一次；默认启用（在 DEFAULT_TERM_CONFIGS）。"""
    term = make_term("low_speed", weight=-0.2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            assert term.compute({}) == 0.0
        assert term.compute({"speed": None}) == 0.0
        assert term.compute({"speed": "bad"}) == 0.0
        assert term.compute({"velocity": float("nan")}) == 0.0
        assert len(caught) == 1 and issubclass(caught[0].category, UserWarning)
    # 缺键帧在聚合器里贡献 0（不改变 reward）
    configs = [dict(config) for config in DEFAULT_TERM_CONFIGS]
    assert "low_speed" in {config["name"] for config in configs}
    aggregator = RewardAggregator(build_terms(configs))
    step = aggregator.step({"speed_ratio": 0.5})
    assert step.components["low_speed"] == 0.0
    assert step.reward == pytest.approx(0.2)


def test_v5_terminal_values_audited_table() -> None:
    """v5 §1 终局值表（P2 审计重解定稿；rc=1.0 基准档；报告 runs/reward_audit/report/）。"""
    from reward_model import default_terminal_values

    values = default_terminal_values()
    assert values == {
        "arrive_dest": 30.0,
        "collision": -19.0,
        "out_of_road": -15.0,
        "max_step": -23.0,
        "error": -5.0,
    }
    aggregator = RewardAggregator([])  # 默认 AggregationConfig 用该表
    assert aggregator.step({"arrive_dest": True}).reward == pytest.approx(30.0)
    assert aggregator.step({"collision": True}).reward == pytest.approx(-19.0)
    assert aggregator.step({"out_of_road": True}).reward == pytest.approx(-15.0)
    assert aggregator.step({"max_step": True}).reward == pytest.approx(-23.0)
    assert aggregator.step({"error": True}).reward == pytest.approx(-5.0)


def test_route_completion_tier_and_terminal_values_config_override_path() -> None:
    """rc 档（3/10/30）与终局值的 config/CLI 覆盖路径（P4 臂入口；**同档配对**）。"""
    from pipeline.trainer import build_reward_adapter

    # CLI 路径：--reward-term-weight route_completion=10 + 同档终局值（rc10 草案）
    rc10_values = {"arrive_dest": 22.0, "collision": -23.0, "out_of_road": -18.0,
                   "max_step": -29.0, "error": -5.0}
    adapter, source = build_reward_adapter(
        {"aggregation": {"terminal_values": rc10_values}},
        term_weights={"route_completion": 10.0},
    )
    assert source.endswith("+term_weights")
    weights = {term.name: term.weight for term in adapter.factory().terms}
    assert weights["route_completion"] == pytest.approx(10.0)
    assert weights["speed_ratio"] == pytest.approx(0.4)
    assert weights["low_speed"] == pytest.approx(-0.2)
    assert adapter.factory().step({"max_step": True}).reward == pytest.approx(-29.0)

    # 只改权重、终局值仍 rc=1 默认 → rc 档配对守卫 fail-fast（P4 前置-B，Gate2 发现④）
    with pytest.raises(ValueError, match="rc 档配对"):
        build_reward_adapter(term_weights={"route_completion": 10.0})

    # config 路径：stages.C.reward = {terms, aggregation.terminal_values}
    config = {
        "terms": [dict(term) for term in DEFAULT_TERM_CONFIGS],
        "aggregation": {"terminal_values": {"arrive_dest": 3.5, "collision": -7.25}},
    }
    adapter_cfg, _ = build_reward_adapter(config)
    aggregator = adapter_cfg.factory()
    assert aggregator.step({"arrive_dest": True}).reward == pytest.approx(3.5)
    aggregator.reset()
    # 默认 crash 终止项（−10）与覆盖后的 collision 终局值（−7.25）叠加
    assert aggregator.step({"collision": True}).reward == pytest.approx(-17.25)


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
# 奖励序不变量：蠕动 < 碰撞 < 正常行驶（lane_center 复活适配版）
# --------------------------------------------------------------------------- #

def _roll_synthetic_scenario(
    steps: int,
    speed_ratio: float,
    rc_target: float,
    terminal: str,
    *,
    d_lat: float | None = None,
) -> float:
    """真实 ``RewardAggregator`` 滚一条合成轨迹（含终局步，共 steps+1 步）。

    ``d_lat`` 非 None 时启用 lane_center（默认项 + 追加项），否则验证"无横向键 → 项为 0"。
    """
    configs = [dict(config) for config in DEFAULT_TERM_CONFIGS]
    if d_lat is not None:
        configs.append({"name": "lane_center", "weight": -0.1})
    aggregator = RewardAggregator(build_terms(configs))
    total = 0.0
    for index in range(steps + 1):
        ctx: dict[str, object] = {
            "speed_ratio": speed_ratio,
            "route_completion": rc_target * index / steps,
        }
        if d_lat is not None:
            ctx["d_lat"] = d_lat
        if index == steps:
            ctx[terminal] = True
        total += aggregator.step(ctx).reward
    return total


def test_reward_ordering_creep_collision_drive_with_and_without_lane_center() -> None:
    """排序不变量（f23b925 适配版，v4 CaRL 修复后重校）：蠕动 < 碰撞 < 正常行驶。

    - lane_center 关（现默认）：无横向键 → 项为 0，序保持；
    - lane_center 开（追加项）：偏航的蠕动被进一步压低，序仍保持。
    """
    creep = _roll_synthetic_scenario(1000, 0.06, 0.20, "max_step")
    collision = _roll_synthetic_scenario(150, 0.80, 0.45, "collision")
    drive = _roll_synthetic_scenario(300, 0.80, 1.0, "arrive_dest")
    assert creep < collision < drive

    creep_lc = _roll_synthetic_scenario(1000, 0.06, 0.20, "max_step", d_lat=1.5)
    collision_lc = _roll_synthetic_scenario(150, 0.80, 0.45, "collision", d_lat=0.5)
    drive_lc = _roll_synthetic_scenario(300, 0.80, 1.0, "arrive_dest", d_lat=0.5)
    assert creep_lc < collision_lc < drive_lc
    # 启用 lane_center 后每步罚 = weight×max(0,|d_lat|-0.25)；v4 修复后 CaRL 乘子只清零
    # **正向**稠密项，终局帧的 lane_center 罚（负向）保留，故步数 = steps + 1（含终局帧）。
    assert creep_lc == pytest.approx(creep - 1001 * 0.1 * (1.5 - 0.25), abs=0.02)
    assert collision_lc == pytest.approx(collision - 151 * 0.1 * (0.5 - 0.25), abs=0.02)
    assert drive_lc < drive


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


def test_carl_multiplier_zeroes_gains_but_keeps_same_frame_penalties() -> None:
    """v4 修复：违规帧乘子只清零正向稠密项；同帧的罚分项保留（不再被连带抹掉）。"""
    terms = build_terms(
        [
            {"name": "speed_ratio", "weight": 1.0},
            {"name": "solid_line", "weight": -2.0},
            {"name": "out_of_road", "weight": -8.0},
        ]
    )
    aggregator = RewardAggregator(
        terms,
        AggregationConfig(
            terminal_values={"out_of_road": -5.0},
            carl_rules={"out_of_road": CarlRule(factor=0.0, terminate=True)},
        ),
    )
    normal = aggregator.step({"speed_ratio": 0.6})
    assert normal.reward == pytest.approx(0.6)
    assert normal.dense_positive_sum == pytest.approx(0.6)
    assert normal.dense_negative_sum == 0.0

    off = aggregator.step({"speed_ratio": 0.8, "solid_line_crossing": True, "out_of_road": True})
    assert off.carl_multiplier == 0.0
    assert off.dense_positive_sum == pytest.approx(0.8)  # 正向：speed_ratio 被清零
    assert off.dense_negative_sum == pytest.approx(-2.0)  # 负向：solid_line 保留
    assert off.dense_sum == pytest.approx(-1.2)
    assert off.components["speed_ratio"] == pytest.approx(0.8)
    assert off.components["solid_line"] == pytest.approx(-2.0)
    assert off.components["out_of_road"] == pytest.approx(-8.0)
    assert off.terminal_value == pytest.approx(-5.0)
    assert off.reward == pytest.approx(-15.0)  # 0 + (−2.0) + (−8.0) + (−5.0)
    assert off.done and off.reason == "out_of_road"


def test_carl_fix_holds_on_term_weight_override_path() -> None:
    """``--reward-term-weight``（``build_reward_adapter``）路径与修复后的聚合口径一致。"""
    from pipeline.trainer import build_reward_adapter

    before = [dict(term) for term in DEFAULT_TERM_CONFIGS]
    adapter, source = build_reward_adapter(term_weights={"solid_line": -3.0})
    assert source.endswith("+term_weights")
    step = adapter.factory().step(
        {"speed_ratio": 0.5, "solid_line_crossing": True, "out_of_road": True}
    )
    assert step.components["speed_ratio"] == pytest.approx(0.2)  # 覆盖后仍按默认权重 0.4 计
    assert step.components["solid_line"] == pytest.approx(-3.0)
    # 0（正向清零）+ (−3.0) + (−8.0) + v5 定稿 out_of_road 终局值 (−15.0)
    assert step.reward == pytest.approx(-26.0)
    assert [dict(term) for term in DEFAULT_TERM_CONFIGS] == before


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
