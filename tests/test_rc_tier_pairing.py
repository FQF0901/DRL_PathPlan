"""P4 前置-B：rc 档配对守卫——``route_completion`` 权重与终局值必须同档（Gate2 发现④）。

锁定：

- 只改权重（CLI ``--reward-term-weight route_completion=X`` / config terms）而终局值仍
  rc=1 默认 → ``build_reward_adapter`` fail-fast（跨档复用风险）；
- 同档配对（``docs/reward_audit/config_draft_rc*.yaml``，P4 臂配置来源）→ 通过，且该档
  ``terminal_values`` 逐档生效（rc1 −23 / rc3 −24 / rc10 −29 / rc30 −42）；
- 非 rc 项权重覆盖（``speed_ratio`` 等）不受守卫影响。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from pipeline.trainer import build_reward_adapter

_DRAFTS = Path(__file__).resolve().parents[1] / "docs" / "reward_audit"
_RC1 = {
    "arrive_dest": 30.0,
    "collision": -19.0,
    "out_of_road": -15.0,
    "max_step": -23.0,
    "error": -5.0,
}
#: v5 §1 每档定稿（arrive/collision/out_of_road/max_step）与 max_step 单值
_TIER_TABLE = {
    1.0: (30, -19, -15, -23),
    3.0: (28, -20, -16, -24),
    10.0: (22, -23, -18, -29),
    30.0: (2, -33, -26, -42),
}


def _draft_reward_cfg(tier: float) -> Dict[str, Any]:
    with open(_DRAFTS / f"config_draft_rc{tier:g}.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["stages"]["C"]["reward"]


def _max_step_terminal_value(adapter) -> float:  # noqa: ANN001
    _, meta = adapter.step(
        0, {"max_step": True}, {"ego": np.zeros((1, 8), dtype=np.float32)}, True, 0.0
    )
    assert meta["terminal_key"] == "max_step"
    return float(meta["terminal_value"])


@pytest.mark.parametrize("tier", [3.0, 10.0, 30.0])
def test_rc_weight_with_rc1_defaults_fails_fast(tier: float) -> None:
    with pytest.raises(ValueError, match="rc 档配对"):
        build_reward_adapter(term_weights={"route_completion": tier})


def test_config_terms_rc_weight_without_values_fails_fast() -> None:
    with pytest.raises(ValueError, match="rc 档配对"):
        build_reward_adapter({"terms": [{"name": "route_completion", "weight": 10.0, "gamma": 1.0}]})


def test_explicit_rc1_values_with_other_weight_fail() -> None:
    with pytest.raises(ValueError, match="rc 档配对"):
        build_reward_adapter(
            {"aggregation": {"terminal_values": dict(_RC1)}},
            term_weights={"route_completion": 3.0},
        )


def test_rc1_default_and_draft_pass() -> None:
    adapter, _ = build_reward_adapter()
    assert _max_step_terminal_value(adapter) == pytest.approx(-23.0)
    adapter_rc1, _ = build_reward_adapter(_draft_reward_cfg(1.0))
    assert _max_step_terminal_value(adapter_rc1) == pytest.approx(-23.0)


@pytest.mark.parametrize("tier", [1.0, 3.0, 10.0, 30.0])
def test_draft_configs_pair_and_apply_tier_values(tier: float) -> None:
    """P4 臂配置来源：草案同时给出该档权重与该档终局值 → 守卫通过、值生效。"""
    cfg = _draft_reward_cfg(tier)
    adapter, source = build_reward_adapter(cfg)
    assert source.startswith("reward_model.aggregation.RewardAggregator")
    weights = {term.name: term.weight for term in adapter.factory().terms}
    assert weights["route_completion"] == pytest.approx(tier)
    values = cfg["aggregation"]["terminal_values"]
    expected = _TIER_TABLE[tier]
    assert (
        values["arrive_dest"], values["collision"], values["out_of_road"], values["max_step"]
    ) == expected, "草案终局值必须与 v5 §1 表逐档一致"
    assert values["error"] == -5
    assert _max_step_terminal_value(adapter) == pytest.approx(expected[3])


def test_cli_weight_plus_matching_values_passes() -> None:
    cfg = {"aggregation": {"terminal_values": dict(_draft_reward_cfg(3.0)["aggregation"]["terminal_values"])}}
    adapter, _ = build_reward_adapter(cfg, term_weights={"route_completion": 3.0})
    assert _max_step_terminal_value(adapter) == pytest.approx(-24.0)


def test_non_rc_weight_override_not_guarded() -> None:
    adapter, _ = build_reward_adapter(term_weights={"speed_ratio": 0.0})
    weights = {term.name: term.weight for term in adapter.factory().terms}
    assert weights["route_completion"] == pytest.approx(1.0)
    assert weights["speed_ratio"] == 0.0
    assert _max_step_terminal_value(adapter) == pytest.approx(-23.0)
