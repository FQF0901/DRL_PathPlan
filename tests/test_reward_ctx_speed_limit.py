"""v4 ⑥ 奖励 ctx 限速修复（6a 来源同源 + 6b 终局帧掩码回退）——**行为修复，默认生效**。

背景：ctx 的 ``speed_limit_mps`` 旧口径 = obs LD slot0 第 4 维，与评测 KPI
（``pipeline/eval_runner.py::_lane_limit_mps`` → ``agent.lane.speed_limit``）不同源
（id86 实测 44 帧差异）；且终局到达帧 ``ld_mask[0]=0`` 时该键缺失 → ``speed_ratio`` /
``speed_limit`` 项在终局帧静默失效。

修复口径：

- **6a**：优先 ``info["lane_speed_limit_mps"]``（由 ``lane_speed_limit_info`` 从
  ``agent.lane.speed_limit`` 注入，与评测 KPI 同源、同兜底 13.9 m/s）；缺失回退 ld slot0；
  两源都缺 → 一次性告警 + 该帧无值；
- **6b**：终局帧无任何源时回退本 env **最后有效值**（口径连续）。

指纹（记录在案，测试显式断言）：

- 6a：lane=8.0 而 ld slot0=10.0 的帧，旧 ctx 用 10.0（speed=10 → ratio 1.0，超速惩罚 0.0）；
  新 ctx 用 8.0（ratio 1.25，惩罚项原始值 0.2 → ×(−5) = **−1.0**）；
- 6b：终局帧（ld_mask[0]=0）旧实现无 ``speed_limit_mps``/``speed_ratio``（惩罚项 0.0）；
  新实现回退最后有效值 8.0（speed=10 → ratio 1.25 → 惩罚 −1.0），不再静默失效。

不建 env、不 import metadrive（fake lane/agent + 纯 ctx 断言 + reward_model 项直接算值）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from pipeline.eval_runner import DEFAULT_SPEED_LIMIT_FALLBACK
from pipeline.trainer import (
    LANE_SPEED_LIMIT_FALLBACK_MPS,
    RewardAdapter,
    lane_speed_limit_info,
)
from reward_model import make_term

LD_SLOTS = 16
LD_FEATURES = 7


class _Result:
    """最小 StepReward 替身（RewardAdapter.step 只读这些字段）。"""

    reason = "none"
    terminal_key = ""
    terminal_value = 0.0
    dense_sum = 0.0
    terminating_sum = 0.0
    carl_multiplier = 1.0
    carl_penalty = 0.0
    shaping_decay = 1.0

    def __init__(self) -> None:
        self.components: Dict[str, float] = {}
        self.done = False
        self.reward = 0.0


class _CaptureAggregator:
    def __init__(self, sink: List[Dict[str, Any]]) -> None:
        self._sink = sink

    def step(self, ctx: Dict[str, Any], step_index: Optional[int] = None) -> _Result:
        self._sink.append(dict(ctx))
        return _Result()

    def reset(self) -> None:  # noqa: D102 - stub
        pass


def _make_adapter(logger: Optional[Any] = None) -> tuple[RewardAdapter, List[Dict[str, Any]]]:
    captured: List[Dict[str, Any]] = []
    adapter = RewardAdapter(factory=lambda: _CaptureAggregator(captured), logger=logger)
    return adapter, captured


def _obs(ld_limit: float, *, mask: float = 1.0, with_ld: bool = True) -> Dict[str, np.ndarray]:
    obs: Dict[str, np.ndarray] = {"ego": np.zeros((1, 8), dtype=np.float32)}
    if with_ld:
        ld = np.zeros((LD_SLOTS, LD_FEATURES), dtype=np.float32)
        ld[0, 4] = float(ld_limit)
        obs["ld"] = ld
        obs["ld_mask"] = np.array([mask] + [0.0] * (LD_SLOTS - 1), dtype=np.float32)
    return obs


def _step(
    adapter: RewardAdapter,
    captured: List[Dict[str, Any]],
    info: Dict[str, Any],
    obs: Dict[str, np.ndarray],
    *,
    done: bool = False,
) -> Dict[str, Any]:
    """执行一步并返回聚合器实际收到的 ctx（RewardAdapter.step 内部口径的最终结果）。"""
    adapter.step(0, info, obs, done, 0.0)
    return captured[-1]


# --------------------------------------------------------------------------- #
# 6a：来源对齐
# --------------------------------------------------------------------------- #

def test_lane_source_wins_over_ld_slot0_and_reward_fingerprint() -> None:
    adapter, captured = _make_adapter()
    ctx = _step(adapter, captured, {"velocity": 10.0, "lane_speed_limit_mps": 8.0}, _obs(ld_limit=10.0))
    assert ctx["speed_limit_mps"] == 8.0, "lane 源（评测 KPI 同源）必须优先于 ld slot0"
    assert ctx["speed_ratio"] == 10.0 / 8.0
    # 指纹：旧实现（ld slot0=10）ratio=1.0、惩罚 0.0；新实现 ratio=1.25 → 惩罚原始值 0.2
    term = make_term("speed_limit", weight=-5.0, tolerance=0.05)
    old_ctx = {"speed": 10.0, "speed_limit_mps": 10.0}
    assert term.weight * term.compute(old_ctx) == pytest.approx(0.0), "旧口径：不判超速"
    assert term.weight * term.compute(ctx) == pytest.approx(-1.0), "新口径：8 m/s 限速下 10 m/s 超速 25%"
    assert captured[-1]["speed_limit_mps"] == 8.0, "聚合器收到的 ctx 已是新口径"


def test_ld_slot0_fallback_preserved_when_lane_missing() -> None:
    adapter, captured = _make_adapter()
    ctx = _step(adapter, captured, {"velocity": 10.0}, _obs(ld_limit=10.0))
    assert ctx["speed_limit_mps"] == 10.0, "lane 缺失 → 旧 ld slot0 回退保留（兼容）"
    assert ctx["speed_ratio"] == 1.0


def test_info_speed_limit_key_untouched_and_cached() -> None:
    """info 自带 speed_limit_mps（MetaDrive 旧路径）不被覆盖，且纳入终局回退缓存。"""
    adapter, captured = _make_adapter()
    ctx = _step(adapter, captured, {"velocity": 10.0, "speed_limit_mps": 9.0}, _obs(ld_limit=10.0))
    assert ctx["speed_limit_mps"] == 9.0
    terminal = _step(adapter, captured, {"velocity": 10.0}, _obs(ld_limit=0.0, mask=0.0), done=True)
    assert terminal["speed_limit_mps"] == 9.0, "终局回退到 info 自带值"


# --------------------------------------------------------------------------- #
# 6b：终局帧掩码回退
# --------------------------------------------------------------------------- #

def test_terminal_mask_fallback_uses_last_valid_value() -> None:
    adapter, captured = _make_adapter()
    first = _step(adapter, captured, {"velocity": 8.0, "lane_speed_limit_mps": 8.0}, _obs(ld_limit=8.0))
    assert first["speed_limit_mps"] == 8.0
    # 终局到达帧：lane 源缺失 + ld_mask[0]=0（旧实现 → 无键、speed_ratio 失效）
    terminal = _step(adapter, captured, {"velocity": 10.0}, _obs(ld_limit=0.0, mask=0.0), done=True)
    assert terminal["speed_limit_mps"] == 8.0, "终局帧必须回退最后有效限速"
    assert terminal["speed_ratio"] == 10.0 / 8.0
    term = make_term("speed_limit", weight=-5.0, tolerance=0.05)
    assert term.weight * term.compute(terminal) == pytest.approx(-1.0), "终局帧 speed_limit 检查不再失效"


def test_non_terminal_masked_frame_keeps_legacy_no_value() -> None:
    """非终局帧的 ld_mask=0 不启用回退（最小改动面：仅终局帧口径连续）。"""
    adapter, captured = _make_adapter()
    _step(adapter, captured, {"velocity": 8.0, "lane_speed_limit_mps": 8.0}, _obs(ld_limit=8.0))
    mid = _step(adapter, captured, {"velocity": 8.0}, _obs(ld_limit=0.0, mask=0.0), done=False)
    assert "speed_limit_mps" not in mid
    assert "speed_ratio" not in mid


def test_missing_both_sources_warns_once() -> None:
    warnings: List[str] = []
    adapter, captured = _make_adapter(logger=warnings.append)
    first = _step(adapter, captured, {"velocity": 5.0}, _obs(ld_limit=0.0, mask=0.0))
    second = _step(adapter, captured, {"velocity": 5.0}, _obs(ld_limit=0.0, mask=0.0))
    assert "speed_limit_mps" not in first and "speed_limit_mps" not in second
    assert len(warnings) == 1 and "限速" in warnings[0], "两源都缺 → 一次性告警"


# --------------------------------------------------------------------------- #
# lane_speed_limit_info（来源与兜底）
# --------------------------------------------------------------------------- #

class _FakeLane:
    def __init__(self, speed_limit: float) -> None:
        self.speed_limit = float(speed_limit)


class _FakeAgent:
    def __init__(self, lane: Any = None, ref_lanes: Any = None) -> None:
        self.lane = lane
        if ref_lanes is not None:
            self.navigation = type("_Nav", (), {"current_ref_lanes": list(ref_lanes)})()


def test_lane_speed_limit_info_sources_and_fallback() -> None:
    assert lane_speed_limit_info(_FakeAgent(_FakeLane(8.3))) == {"lane_speed_limit_mps": 8.3}
    # MetaDrive 未设置（>=1000）→ 与评测 KPI 同兜底 13.9 m/s
    assert lane_speed_limit_info(_FakeAgent(_FakeLane(1000.0))) == {
        "lane_speed_limit_mps": LANE_SPEED_LIMIT_FALLBACK_MPS
    }
    # lane 缺失 → navigation.current_ref_lanes[0] 回退（与 lane_lateral_info 同链）
    agent = _FakeAgent(None, [_FakeLane(7.5)])
    assert lane_speed_limit_info(agent) == {"lane_speed_limit_mps": 7.5}
    assert lane_speed_limit_info(_FakeAgent(None)) == {}, "无 lane → 空（调用方回退 ld）"
    assert lane_speed_limit_info(object()) == {}, "异常属性链不外抛"


def test_fallback_constant_matches_eval_kpi() -> None:
    assert LANE_SPEED_LIMIT_FALLBACK_MPS == DEFAULT_SPEED_LIMIT_FALLBACK, "兜底值不得与评测漂移"
