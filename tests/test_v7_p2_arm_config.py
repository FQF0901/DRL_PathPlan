"""v7 P2 首臂（``config/arms/v7_arm1_offroad.yaml``）dry-run 守卫（CPU；不开环境）。

锁定（v7 预注册 §9 的配置面）：

- 臂文件经训练侧 ``load_config``（一层平铺合并）可加载，``stages.C.reward`` 经
  ``build_reward_adapter`` 构造成功；
- 项集 = arm0 同档草案（rc=1）逐参数 + **恰一个**追加项 ``off_road_edge``
  （weight −0.5 / ``edge_scale_m`` 1.0）；终局值 = rc=1 草案逐位；
- ``off_road_edge`` 在适配器内生效（居中 → 0；压线 → −0.5；缺车道 ctx → 0）；
- KL 锚参数键（0.05 → 0.02）与 v7 预注册一致（记录用；driver CLI 实际生效）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from pipeline.stages import load_config
from pipeline.trainer import build_reward_adapter
from reward_model import DEFAULT_TERM_CONFIGS, make_term

_ROOT = Path(__file__).resolve().parents[1]
_ARM = _ROOT / "config" / "arms" / "v7_arm1_offroad.yaml"
_DRAFT = _ROOT / "docs" / "reward_audit" / "ebeta2" / "config_draft_rc1.yaml"

_INCLUDES = [
    "config/default.yaml",
    "config/env.yaml",
    "config/model.yaml",
    "config/train.yaml",
    "config/eval.yaml",
]

#: v7 预注册 §9（P2 首臂）：KL 锚初始 → 末值（慢衰减）
_KL_INITIAL = 0.05
_KL_FINAL = 0.02
#: off_road_edge 臂参数（单变量；BC-SAC 半量级）
_EDGE_WEIGHT = -0.5
_EDGE_SCALE_M = 1.0


def _load_arm() -> Dict[str, Any]:
    with open(_ARM, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert raw.get("includes") == _INCLUDES, "臂配置 includes = default.yaml + 四个叶子子配置"
    return load_config(str(_ARM))


def test_v7_arm1_reward_single_variable_and_terminal_pairs() -> None:
    config = _load_arm()
    reward_cfg = config["stages"]["C"]["reward"]
    adapter, source = build_reward_adapter(reward_cfg)
    assert source.startswith("reward_model.aggregation.RewardAggregator"), source

    terms = adapter.factory().terms
    weights = {term.name: term.weight for term in terms}
    assert weights["route_completion"] == pytest.approx(1.0)
    assert weights["speed_ratio"] == pytest.approx(0.4)
    assert weights["low_speed"] == pytest.approx(-0.2)
    assert weights["off_road_edge"] == pytest.approx(_EDGE_WEIGHT)
    edge = next(term for term in terms if term.name == "off_road_edge")
    assert edge.edge_scale_m == pytest.approx(_EDGE_SCALE_M)

    with open(_DRAFT, "r", encoding="utf-8") as handle:
        draft = yaml.safe_load(handle)["stages"]["C"]["reward"]
    draft_names = {term["name"] for term in draft["terms"]}
    assert "off_road_edge" not in draft_names, "off_road_edge 应为臂追加项（草案不含）"
    assert set(weights) == draft_names | {"off_road_edge"}
    rest = [term for term in reward_cfg["terms"] if str(term.get("name")) != "off_road_edge"]
    assert rest == draft["terms"], "去 off_road_edge 后项集应与 rc=1 草案逐参数一致"
    assert reward_cfg["aggregation"]["terminal_values"] == draft["aggregation"]["terminal_values"]

    # max_step 终局值经适配器实际结算（rc=1 定稿 −46）
    _, meta = adapter.step(
        0, {"max_step": True}, {"ego": np.zeros((1, 8), dtype=np.float32)}, True, 0.0
    )
    assert meta["terminal_key"] == "max_step"
    assert meta["terminal_value"] == pytest.approx(-46.0)


def test_v7_arm1_off_road_edge_fires_through_adapter() -> None:
    """适配器 dry-run：居中 → 0；压线（d_edge=0）→ −0.5；越界 1 m → 封顶 −1.0；缺车道 → 0。"""
    config = _load_arm()
    adapter, _ = build_reward_adapter(config["stages"]["C"]["reward"], logger=lambda _msg: None)
    obs = {"ego": np.zeros((1, 8), dtype=np.float32)}

    def edge_component(info: Dict[str, Any]) -> float:
        _, meta = adapter.step(0, dict(info), obs, False, 0.0)
        return float(meta["components"]["off_road_edge"])

    half = 1.75
    assert edge_component({"lane_lateral_offset": 0.0, "lane_half_width_m": half}) == 0.0
    assert edge_component({"lane_lateral_offset": -0.75, "lane_half_width_m": half}) == 0.0
    assert edge_component({"lane_lateral_offset": half, "lane_half_width_m": half}) == pytest.approx(-0.5)
    assert edge_component({"lane_lateral_offset": half + 1.0, "lane_half_width_m": half}) == pytest.approx(-1.0)
    assert edge_component({"lane_lateral_offset": 5.0, "lane_half_width_m": half}) == pytest.approx(-1.0)
    assert edge_component({"velocity": 5.0}) == 0.0  # 缺车道 ctx → 0（不炸、不罚）

    # 该项默认关：不在 DEFAULT_TERM_CONFIGS；单独实例化（不吃 make_term 的默认 weight=1.0）
    # 默认权重 = BC-SAC 原量级 −1.0
    assert "off_road_edge" not in {term["name"] for term in DEFAULT_TERM_CONFIGS}
    from reward_model import get_term_class

    assert get_term_class("off_road_edge")().weight == pytest.approx(-1.0)


def test_v7_arm1_kl_anchor_keys_recorded() -> None:
    """KL 锚参数（记录/预注册一致性；实际执行由 driver CLI 承载）。"""
    config = _load_arm()
    stage_c = config["stages"]["C"]
    assert stage_c["kl_anchor_coef"] == pytest.approx(_KL_INITIAL)
    assert stage_c["kl_anchor_final_coef"] == pytest.approx(_KL_FINAL)
    assert stage_c["kl_anchor_final_coef"] > 0.0, "v7 预注册 §3：KL 锚末值 > 0（方差控制）"
