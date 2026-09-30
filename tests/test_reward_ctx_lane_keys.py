"""v4 ⑦：ctx 新增键（``lead_gap_m`` / ``lead_speed_mps`` / ``lane_half_width_m``）接口契约。

- 来源：``lane_reward_info``（lane API + engine 对象扫描）→ ``LocalEnvPool._record`` 注入 info
  → ``RewardAdapter._build_ctx`` 透传到 ctx；
- 缺数据 = ``None``（键必须存在）+ 整体不可用一次性告警；
- ``lane_half_width_m`` 回退链：lane API → ``dist_to_left/right_side`` 折算；
- **默认计算、无行为影响**：现有奖励项（DEFAULT_TERM_CONFIGS）在有/无这三个键时 reward 逐位一致。

不建 env、不 import metadrive（fake lane/agent/engine）。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from pipeline.trainer import (
    LANE_CTX_KEYS,
    RewardAdapter,
    lane_half_width_m,
    lane_reward_info,
    lead_vehicle_info,
)
from reward_model import AggregationConfig, DEFAULT_TERM_CONFIGS, RewardAggregator, build_terms


class _FakeLane:
    def __init__(self, speed_limit: float = 8.0, width: float = 3.6) -> None:
        self.speed_limit = float(speed_limit)
        self.width = float(width)

    def width_at(self, longitudinal: float) -> float:  # noqa: ARG002
        return self.width

    def local_coordinates(self, position: Any) -> tuple:
        return (float(np.asarray(position, dtype=float).reshape(-1)[0]), 0.0)


class _FakeAgent:
    LENGTH = 4.5

    def __init__(self, lane: Any, position=(0.0, 0.0, 0.0), heading=(1.0, 0.0, 0.0)) -> None:
        self.lane = lane
        self.position = np.asarray(position, dtype=float)
        self.heading = np.asarray(heading, dtype=float)

    def convert_to_local_coordinates(self, point: Any, ref: Any) -> np.ndarray:
        return np.asarray(point, dtype=float) - np.asarray(ref, dtype=float)


class _FakeObject:
    def __init__(self, position: Any, velocity: Any, length: float = 4.0) -> None:
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.LENGTH = float(length)


class _FakeEngine:
    def __init__(self, objects: List[Any]) -> None:
        self._objects = {index: obj for index, obj in enumerate(objects)}

    def get_objects(self) -> Dict[int, Any]:
        return dict(self._objects)


class _CaptureAggregator:
    def __init__(self, sink: List[Dict[str, Any]]) -> None:
        self._sink = sink

    def step(self, ctx, step_index=None):  # noqa: ANN001
        self._sink.append(dict(ctx))

        class _R:
            reason = "none"
            terminal_key = ""
            terminal_value = 0.0
            dense_sum = 0.0
            terminating_sum = 0.0
            carl_multiplier = 1.0
            carl_penalty = 0.0
            shaping_decay = 1.0
            components: Dict[str, float] = {}
            done = False
            reward = 0.0

        return _R()

    def reset(self) -> None:  # noqa: D102 - stub
        pass


def _obs() -> Dict[str, np.ndarray]:
    return {"ego": np.zeros((1, 8), dtype=np.float32)}


# --------------------------------------------------------------------------- #
# lane_reward_info（来源）
# --------------------------------------------------------------------------- #

def test_lane_reward_info_full_context() -> None:
    lane = _FakeLane(speed_limit=8.0, width=3.6)
    agent = _FakeAgent(lane)
    engine = _FakeEngine(
        [
            _FakeObject(position=(20.0, 0.0, 0.0), velocity=(8.0, 0.0, 0.0), length=4.0),
            _FakeObject(position=(10.0, 5.0, 0.0), velocity=(5.0, 0.0, 0.0)),  # 邻车道（横向 5m）→ 过滤
            _FakeObject(position=(-30.0, 0.0, 0.0), velocity=(1.0, 0.0, 0.0)),  # 后方 → 过滤
        ]
    )
    info = lane_reward_info(agent, engine)
    assert info["lane_speed_limit_mps"] == 8.0
    assert info["lane_half_width_m"] == pytest.approx(1.8)
    assert info["lead_gap_m"] == pytest.approx(20.0 - 0.5 * (4.5 + 4.0)), "bumper 净距"
    assert info["lead_speed_mps"] == pytest.approx(8.0), "前车速度在自车航向投影"


def test_lane_reward_info_no_lead_or_engine() -> None:
    agent = _FakeAgent(_FakeLane())
    info = lane_reward_info(agent, _FakeEngine([]))
    assert info["lane_half_width_m"] == pytest.approx(1.8)
    assert info["lead_gap_m"] is None and info["lead_speed_mps"] is None, "无前车 = null（正常语义）"
    info_no_engine = lane_reward_info(agent, None)
    assert info_no_engine["lead_gap_m"] is None
    assert lane_reward_info(object(), None) == {key: None for key in LANE_CTX_KEYS}, "无 lane/engine 全 null"


def test_lead_vehicle_info_requires_vehicle_like_object() -> None:
    agent = _FakeAgent(_FakeLane())
    engine = _FakeEngine([_FakeObject(position=(5.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0))])
    assert lead_vehicle_info(agent, engine, 1.8)["lead_gap_m"] == pytest.approx(5.0 - 0.5 * (4.5 + 4.0))
    engine_bad = _FakeEngine([object()])
    assert lead_vehicle_info(agent, engine_bad, 1.8)["lead_gap_m"] is None, "无 LENGTH 的对象跳过"


def test_lane_half_width_none_when_unavailable() -> None:
    assert lane_half_width_m(object()) is None
    assert lane_half_width_m(_FakeAgent(_FakeLane(width=0.0))) is None


# --------------------------------------------------------------------------- #
# _build_ctx 透传 / 缺数据 null + 一次性告警
# --------------------------------------------------------------------------- #

def _ctx_of(adapter: RewardAdapter, info: Dict[str, Any], captured: List[Dict[str, Any]]) -> Dict[str, Any]:
    adapter.step(0, info, _obs(), False, 0.0)
    return captured[-1]


def test_ctx_lane_keys_default_null_and_warn_once() -> None:
    captured: List[Dict[str, Any]] = []
    warnings: List[str] = []
    adapter = RewardAdapter(factory=lambda: _CaptureAggregator(captured), logger=warnings.append)
    # lane 限速源存在（不触发 ⑥a 告警），仅 lane 上下文三键缺失 → 只应告警一次
    first = _ctx_of(adapter, {"velocity": 5.0, "lane_speed_limit_mps": 8.0}, captured)
    second = _ctx_of(adapter, {"velocity": 5.0, "lane_speed_limit_mps": 8.0}, captured)
    for key in LANE_CTX_KEYS:
        assert first[key] is None and second[key] is None, f"{key} 缺数据必须为 null（键存在）"
    assert len(warnings) == 1 and "车道上下文" in warnings[0], "整体不可用 → 一次性告警"


def test_ctx_lane_keys_passthrough_and_half_width_fallback() -> None:
    captured: List[Dict[str, Any]] = []
    adapter = RewardAdapter(factory=lambda: _CaptureAggregator(captured))
    ctx = _ctx_of(
        adapter,
        {
            "velocity": 5.0,
            "lane_speed_limit_mps": 8.0,
            "lead_gap_m": 12.5,
            "lead_speed_mps": 7.0,
            "lane_half_width_m": 1.75,
        },
        captured,
    )
    assert ctx["lead_gap_m"] == 12.5 and ctx["lead_speed_mps"] == 7.0
    assert ctx["lane_half_width_m"] == 1.75
    # lane API 缺失 → dist_to_left/right_side 折算（lane_lateral_info 回退键）
    ctx_fallback = _ctx_of(
        adapter, {"velocity": 5.0, "dist_to_left_side": 2.0, "dist_to_right_side": 1.6}, captured
    )
    assert ctx_fallback["lane_half_width_m"] == pytest.approx(1.8)


# --------------------------------------------------------------------------- #
# 无行为影响（现有奖励项不消费新键）
# --------------------------------------------------------------------------- #

def test_extra_ctx_keys_do_not_change_default_reward() -> None:
    base_ctx = {
        "speed": 6.0,
        "speed_limit_mps": 8.0,
        "route_completion": 0.5,
        "done": False,
        "a_lon": 0.3,
        "a_lat": 0.1,
        "jerk": 0.2,
    }
    plain = RewardAggregator(build_terms([dict(term) for term in DEFAULT_TERM_CONFIGS]), AggregationConfig())
    with_keys = RewardAggregator(build_terms([dict(term) for term in DEFAULT_TERM_CONFIGS]), AggregationConfig())
    result_plain = plain.step(dict(base_ctx), step_index=0)
    extended = dict(base_ctx)
    extended.update({"lead_gap_m": 12.0, "lead_speed_mps": 7.0, "lane_half_width_m": 1.8})
    result_extended = with_keys.step(extended, step_index=0)
    assert result_plain.reward == result_extended.reward
    assert dict(result_plain.components) == dict(result_extended.components)
