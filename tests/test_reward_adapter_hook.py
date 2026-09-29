"""P4 奖励钩子回归：``build_reward_adapter`` 接 config + 项权重覆盖（不改全局默认）。

锁定：
- ``term_weights={"speed_ratio": 0.0}`` → 聚合器该项权重为 0，其余默认项保留；
- ``DEFAULT_TERM_CONFIGS`` 不被污染（仍为 1.0）；
- 未命中项名 fail-fast（ValueError，防消融拼错静默失效）；
- config ``{"terms": [...]}`` 透传生效；
- ``stages._parse_reward_term_weights`` / CLI ``--reward-term-weight`` 解析。
"""

from __future__ import annotations

import pytest

from pipeline.stages import _parse_args, _parse_reward_term_weights
from pipeline.trainer import build_reward_adapter
from reward_model import DEFAULT_TERM_CONFIGS


def _weights(adapter) -> dict:
    return {term.name: term.weight for term in adapter.factory().terms}


def test_term_weights_override_is_applied_and_global_default_untouched() -> None:
    before = [dict(term) for term in DEFAULT_TERM_CONFIGS]
    adapter, source = build_reward_adapter(term_weights={"speed_ratio": 0.0})
    assert source.endswith("+term_weights")
    weights = _weights(adapter)
    assert weights["speed_ratio"] == 0.0
    assert weights["route_completion"] == 1.0, "其余项权重不应被改动"
    assert [dict(term) for term in DEFAULT_TERM_CONFIGS] == before, "不得污染全局默认"


def test_unknown_term_weight_fails_fast() -> None:
    with pytest.raises(ValueError, match="未命中奖励项"):
        build_reward_adapter(term_weights={"nope": 0.0})


def test_reward_config_terms_are_passed_through() -> None:
    adapter, _ = build_reward_adapter({"terms": [{"name": "speed_ratio", "weight": 2.0, "cap": 1.0}]})
    assert _weights(adapter) == {"speed_ratio": 2.0}


def test_cli_reward_term_weight_parsing() -> None:
    args = _parse_args(["--stage", "C", "--reward-term-weight", "speed_ratio=0.0",
                        "--reward-term-weight", "route_completion=0.5"])
    assert args.reward_term_weight == ["speed_ratio=0.0", "route_completion=0.5"]
    assert _parse_reward_term_weights(args.reward_term_weight) == {
        "speed_ratio": 0.0, "route_completion": 0.5,
    }
    assert _parse_reward_term_weights(None) == {}
    with pytest.raises(SystemExit):
        _parse_reward_term_weights(["speed_ratio"])
    with pytest.raises(SystemExit):
        _parse_reward_term_weights(["speed_ratio=abc"])
