"""v7 §13 奖励单变量臂队列 A/B/C 配置守卫（CPU；不开环境）。

锁定（v7 预注册 §13 的配置面）：

- 三个臂文件经训练侧 ``load_config``（一层平铺合并）可加载，``stages.C.reward`` 经
  ``build_reward_adapter`` 构造成功；
- **单变量**（其余 = ``v7_arm1_offroad.yaml`` 逐位）：
  - A：``off_road_edge.edge_scale_m`` 1.0 → 2.5（weight 保持 −0.5）；
  - B：追加 ``{name: speed_deficit, weight: -0.3}``；
  - C：追加 ``{name: comfort_jerk_win, weight: -0.1, deadband: 5.0, window_steps: 20}``；
- 终局值 / KL 锚键与 arm1 一致；新项在适配器内生效（B：v<限速扣分；C：20 步窗口 + reset）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml

from pipeline.stages import load_config
from pipeline.trainer import build_reward_adapter
from reward_model import DEFAULT_TERM_CONFIGS

_ROOT = Path(__file__).resolve().parents[1]
_ARMS = _ROOT / "config" / "arms"
_ARM1 = _ARMS / "v7_arm1_offroad.yaml"
_ARM_A = _ARMS / "v7_reward_A.yaml"
_ARM_B = _ARMS / "v7_reward_B.yaml"
_ARM_C = _ARMS / "v7_reward_C.yaml"

_INCLUDES = [
    "config/default.yaml",
    "config/env.yaml",
    "config/model.yaml",
    "config/train.yaml",
    "config/eval.yaml",
]

#: v7 预注册 §13：KL 锚 0.05 → 0.02（同 arm1/§9）
_KL_INITIAL = 0.05
_KL_FINAL = 0.02
#: A 臂单变量
_EDGE_SCALE_A = 2.5
_EDGE_WEIGHT = -0.5
#: B/C 臂单变量
_DEFICIT_WEIGHT = -0.3
_JERK_WIN_WEIGHT = -0.1
_JERK_WIN_DEADBAND = 5.0
_JERK_WIN_STEPS = 20

_TERMINAL_VALUES = {"arrive_dest": 29, "collision": -22, "out_of_road": -14,
                    "max_step": -46, "error": -5}


def _load_arm(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert raw.get("includes") == _INCLUDES, "臂配置 includes = default.yaml + 四个叶子子配置"
    return load_config(str(path))


def _terms(reward_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [dict(term) for term in reward_cfg["terms"]]


def _arm1_terms() -> List[Dict[str, Any]]:
    return _terms(_load_arm(_ARM1)["stages"]["C"]["reward"])


def _without(terms: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    return [dict(term) for term in terms if str(term.get("name")) != name]


def test_reward_A_single_variable_edge_scale_2p5() -> None:
    """A（§13）= arm1 逐位 + ``edge_scale_m`` 1.0 → 2.5（weight 保持 −0.5）。"""
    stage_c = _load_arm(_ARM_A)["stages"]["C"]
    reward_cfg = stage_c["reward"]
    adapter, source = build_reward_adapter(reward_cfg)
    assert source.startswith("reward_model.aggregation.RewardAggregator"), source

    terms = adapter.factory().terms
    weights = {term.name: term.weight for term in terms}
    assert weights["off_road_edge"] == pytest.approx(_EDGE_WEIGHT)
    edge = next(term for term in terms if term.name == "off_road_edge")
    assert edge.edge_scale_m == pytest.approx(_EDGE_SCALE_A)

    # 与 arm1 的差异仅 off_road_edge.edge_scale_m 一处
    base = _arm1_terms()
    got = _terms(reward_cfg)
    assert [t["name"] for t in got] == [t["name"] for t in base]
    diffs = [
        (a, b) for a, b in zip(got, base) if a != b
    ]
    assert diffs == [
        ({"name": "off_road_edge", "weight": -0.5, "edge_scale_m": 2.5},
         {"name": "off_road_edge", "weight": -0.5, "edge_scale_m": 1.0})
    ]
    assert reward_cfg["aggregation"]["terminal_values"] == _TERMINAL_VALUES
    assert stage_c["kl_anchor_coef"] == pytest.approx(_KL_INITIAL)
    assert stage_c["kl_anchor_final_coef"] == pytest.approx(_KL_FINAL)


def test_reward_A_edge_fires_from_2p5m_through_adapter() -> None:
    """A 适配器 dry-run：界内 ≥2.5 m → 0；1.5 m → −0.2；界上 → −0.5；越界 2.5 m 封顶 −1.0。"""
    adapter, _ = build_reward_adapter(
        _load_arm(_ARM_A)["stages"]["C"]["reward"], logger=lambda _msg: None
    )
    obs = {"ego": np.zeros((1, 8), dtype=np.float32)}
    aggregator = adapter.factory()

    def edge_component(d_edge: float) -> float:
        step = aggregator.step({"d_edge": d_edge})
        return float(step.components["off_road_edge"])

    assert edge_component(-2.5) == 0.0
    assert edge_component(-1.5) == pytest.approx(-0.2)  # 1 + (−1.5)/2.5 = 0.4
    assert edge_component(0.0) == pytest.approx(-0.5)
    assert edge_component(2.5) == pytest.approx(-1.0)  # 封顶 2 × 0.5
    assert edge_component(10.0) == pytest.approx(-1.0)


def test_reward_B_single_variable_speed_deficit() -> None:
    """B（§13）= arm1 逐位 + ``speed_deficit``（weight −0.3；其余逐位）。"""
    stage_c = _load_arm(_ARM_B)["stages"]["C"]
    reward_cfg = stage_c["reward"]
    adapter, source = build_reward_adapter(reward_cfg)
    assert source.startswith("reward_model.aggregation.RewardAggregator"), source

    terms = adapter.factory().terms
    weights = {term.name: term.weight for term in terms}
    assert weights["speed_deficit"] == pytest.approx(_DEFICIT_WEIGHT)
    assert weights["off_road_edge"] == pytest.approx(_EDGE_WEIGHT)
    edge = next(term for term in terms if term.name == "off_road_edge")
    assert edge.edge_scale_m == pytest.approx(1.0)

    got = _terms(reward_cfg)
    assert _without(got, "speed_deficit") == _arm1_terms(), "B = arm1 项集 + 恰一个 speed_deficit"
    assert got[3] == {"name": "speed_deficit", "weight": -0.3}
    assert reward_cfg["aggregation"]["terminal_values"] == _TERMINAL_VALUES
    assert stage_c["kl_anchor_coef"] == pytest.approx(_KL_INITIAL)
    assert stage_c["kl_anchor_final_coef"] == pytest.approx(_KL_FINAL)

    # 项在聚合器内生效：v=limit/2 → 0.5 × (−0.3) = −0.15；v=limit → 0；缺限速 → 0
    aggregator = adapter.factory()
    step = aggregator.step({"speed": 4.0, "speed_limit_mps": 8.0})
    assert step.raw_components["speed_deficit"] == pytest.approx(0.5)
    assert step.components["speed_deficit"] == pytest.approx(-0.15)
    assert aggregator.step({"speed": 8.0, "speed_limit_mps": 8.0}).components["speed_deficit"] == 0.0
    assert aggregator.step({"speed": 4.0}).components["speed_deficit"] == 0.0


def test_reward_C_single_variable_comfort_jerk_window() -> None:
    """C（§13）= arm1 逐位 + ``comfort_jerk_win``（−0.1 / deadband 5.0 / 20 步；其余逐位）。"""
    stage_c = _load_arm(_ARM_C)["stages"]["C"]
    reward_cfg = stage_c["reward"]
    adapter, source = build_reward_adapter(reward_cfg)
    assert source.startswith("reward_model.aggregation.RewardAggregator"), source

    terms = adapter.factory().terms
    weights = {term.name: term.weight for term in terms}
    assert weights["comfort_jerk_win"] == pytest.approx(_JERK_WIN_WEIGHT)
    win = next(term for term in terms if term.name == "comfort_jerk_win")
    assert win.deadband == pytest.approx(_JERK_WIN_DEADBAND)
    assert win.window_steps == _JERK_WIN_STEPS

    got = _terms(reward_cfg)
    assert _without(got, "comfort_jerk_win") == _arm1_terms(), "C = arm1 项集 + 恰一个 comfort_jerk_win"
    assert {"name": "comfort_jerk_win", "weight": -0.1, "deadband": 5.0, "window_steps": 20} in got
    assert reward_cfg["aggregation"]["terminal_values"] == _TERMINAL_VALUES
    assert stage_c["kl_anchor_coef"] == pytest.approx(_KL_INITIAL)
    assert stage_c["kl_anchor_final_coef"] == pytest.approx(_KL_FINAL)

    # 窗口生效：20 步 |jerk|=10 → mean 10 → raw 1.0 → −0.1；reset 后单步 5 → 0
    aggregator = adapter.factory()
    step = None
    for _ in range(20):
        step = aggregator.step({"jerk": 10.0})
    assert step is not None
    assert step.raw_components["comfort_jerk_win"] == pytest.approx(1.0)
    assert step.components["comfort_jerk_win"] == pytest.approx(-0.1)
    aggregator.reset()
    assert aggregator.step({"jerk": 5.0}).components["comfort_jerk_win"] == 0.0


def test_reward_arms_terminal_values_and_default_off() -> None:
    """三臂终局值与 arm1 一致；新项默认关（不在 DEFAULT_TERM_CONFIGS）。"""
    for path in (_ARM_A, _ARM_B, _ARM_C):
        cfg = _load_arm(path)
        assert cfg["stages"]["C"]["reward"]["aggregation"]["terminal_values"] == _TERMINAL_VALUES
    names = {term["name"] for term in DEFAULT_TERM_CONFIGS}
    assert "speed_deficit" not in names
    assert "comfort_jerk_win" not in names
