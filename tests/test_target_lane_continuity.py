"""R21-0：target lane 选择纯函数测试（优先级链 + 稳定性门）。

覆盖：
1. 优先级：route current_ref → route continuation → next_ref → ego.lane 仅兜底
   （**不得沿用 "ego.lane 最高优先"**）；
2. 门：非真实 transition 时 ego 横向偏移（±1 m）/ 航向偏移（±5°）**不引起 target 切换**；
3. 越过 target 车道末端（真实 transition）→ route continuation 切换；
4. 多 current_ref 时按续接端点选（不依赖 ego 横向位置）；平局保持 ego 路由车道（防横跳）；
5. 无 route 信息 → ego.lane 兜底（route_associated=False, confidence=0.25）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from env.obs.lane_semantic import (
    SOURCE_EGO_FALLBACK,
    SOURCE_NONE,
    SOURCE_ROUTE_CONTINUATION,
    SOURCE_ROUTE_CURRENT,
    SOURCE_ROUTE_NEXT,
    select_target_lane,
)


class FakeLane:
    """直道 stub（``position`` / ``local_coordinates`` / ``heading_theta_at`` / ``end``）。"""

    def __init__(self, *, x0=0.0, y0=0.0, heading=0.0, length=100.0, width=3.5, index=("road", "node", 0)):
        self.x0, self.y0, self.heading = float(x0), float(y0), float(heading)
        self.length, self.width = float(length), float(width)
        self.index = tuple(index)
        self.start = np.array([self.x0, self.y0, 0.0])
        self.end = np.array(
            [self.x0 + self.length * math.cos(self.heading), self.y0 + self.length * math.sin(self.heading), 0.0]
        )
        self.direction = np.array([math.cos(self.heading), math.sin(self.heading), 0.0])
        self.direction_lateral = np.array([self.direction[1], -self.direction[0], 0.0])

    def position(self, longitudinal, lateral=0.0):
        return np.asarray(
            self.start + float(longitudinal) * self.direction + float(lateral) * self.direction_lateral,
            dtype=np.float64,
        )

    def heading_theta_at(self, longitudinal):
        return self.heading

    def local_coordinates(self, position):
        delta = np.asarray(position, dtype=np.float64)[:2] - self.start[:2]
        return float(delta @ self.direction[:2]), float(delta @ self.direction_lateral[:2])


def _call(ego_lane=None, position=None, refs=(), nxt=()):
    return select_target_lane(
        ego_lane=ego_lane, ego_position=position, current_ref_lanes=refs, next_ref_lanes=nxt
    )


# --------------------------------------------------------------------------- #
# 1) 优先级链
# --------------------------------------------------------------------------- #
def test_route_current_wins_over_ego_lane() -> None:
    """ego 不在 route 上时，target 仍是 route 车道（ego.lane 不是最高优先）。"""
    route_lane = FakeLane(y0=0.0, index=("road", "node", 1))
    other_lane = FakeLane(y0=-3.5, index=("road", "node", 2))
    selection = _call(ego_lane=other_lane, position=(10.0, -3.5), refs=[route_lane])
    assert selection.lane is route_lane
    assert selection.source == SOURCE_ROUTE_CURRENT
    assert selection.route_associated is True
    assert selection.confidence == pytest.approx(1.0)


def test_ego_lane_on_route_is_selected_directly() -> None:
    lane = FakeLane()
    selection = _call(ego_lane=lane, position=(10.0, 0.5), refs=[lane])
    assert selection.lane is lane and selection.source == SOURCE_ROUTE_CURRENT


def test_ego_fallback_only_without_route() -> None:
    lane = FakeLane()
    selection = _call(ego_lane=lane, position=(10.0, 0.0))
    assert selection.lane is lane
    assert selection.source == SOURCE_EGO_FALLBACK
    assert selection.route_associated is False
    assert selection.confidence == pytest.approx(0.25)


def test_none_when_no_lane_at_all() -> None:
    selection = _call()
    assert selection.lane is None
    assert selection.source == SOURCE_NONE
    assert selection.route_associated is False
    assert selection.confidence == pytest.approx(0.0)


def test_next_ref_fallback_when_no_current_ref() -> None:
    next_lane = FakeLane(index=("next", "node", 0))
    selection = _call(ego_lane=None, position=(0.0, 0.0), nxt=[next_lane])
    assert selection.lane is next_lane
    assert selection.source == SOURCE_ROUTE_NEXT
    assert selection.confidence == pytest.approx(0.6)


def test_ego_lane_in_next_ref_is_route_continuation() -> None:
    next_lane = FakeLane(index=("next", "node", 0))
    selection = _call(ego_lane=next_lane, position=(5.0, 0.0), nxt=[next_lane])
    assert selection.lane is next_lane
    assert selection.source == SOURCE_ROUTE_CONTINUATION
    assert selection.route_associated is True


# --------------------------------------------------------------------------- #
# 2) 横向偏移 / 航向偏移不切换（非真实 transition）
# --------------------------------------------------------------------------- #
def test_lateral_and_heading_offsets_do_not_switch_target() -> None:
    lane = FakeLane()
    nxt = FakeLane(index=("next", "node", 0))
    base = _call(ego_lane=lane, position=lane.position(30.0, 0.0)[:2], refs=[lane], nxt=[nxt])
    for lat in (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0):
        for heading_delta in (-math.radians(10), -math.radians(2.5), math.radians(2.5), math.radians(10)):
            point = lane.position(30.0, lat)[:2]
            selection = _call(ego_lane=lane, position=point, refs=[lane], nxt=[nxt])
            assert selection.lane is base.lane
            assert selection.source == base.source
            assert selection.confidence == base.confidence
            # 航向不影响选择（纯函数只用位置投影 long 判越界）
            _ = heading_delta


def test_multi_ref_selection_independent_of_ego_lateral_position() -> None:
    """两条 current_ref：一条续接（gap=0），一条错位（gap=3.5）→ 无论 ego 在哪都选续接者。"""
    left = FakeLane(y0=3.5, index=("road", "node", 0))
    right = FakeLane(y0=0.0, index=("road", "node", 1))
    nxt = FakeLane(x0=100.0, y0=0.0, index=("next", "node", 1))
    for y in (-1.0, 0.0, 1.0, 3.5):
        selection = _call(ego_lane=left, position=(20.0, y), refs=[left, right], nxt=[nxt])
        assert selection.lane is right, f"y={y} 不应改变续接选择"
        assert selection.source == SOURCE_ROUTE_CURRENT


def test_tie_keeps_ego_route_lane() -> None:
    """续接端点差 ≤ 容差（平局）→ 保持 ego 所在路由车道（防横跳）。"""
    left = FakeLane(y0=3.5, index=("road", "node", 0))
    right = FakeLane(y0=0.0, index=("road", "node", 1))
    # gap(left)=1.5, gap(right)=2.0：最优=left，但与 ego 车道(right)差距 ≤ 容差 → 保持 right
    nxt = FakeLane(x0=100.0, y0=2.0, index=("next", "node", 0))
    selection = _call(ego_lane=right, position=(20.0, 0.0), refs=[left, right], nxt=[nxt])
    assert selection.lane is right
    assert selection.source == SOURCE_ROUTE_CURRENT


# --------------------------------------------------------------------------- #
# 3) route continuation（越过末端 = 真实 transition）
# --------------------------------------------------------------------------- #
def test_continuation_switch_after_lane_end() -> None:
    lane = FakeLane(length=100.0)
    nxt = FakeLane(index=("next", "node", 0))
    # 越过末端（long=100.5 > length）
    selection = _call(ego_lane=lane, position=lane.position(100.5, 0.0)[:2], refs=[lane], nxt=[nxt])
    assert selection.lane is nxt
    assert selection.source == SOURCE_ROUTE_CONTINUATION
    assert selection.confidence == pytest.approx(0.8)


def test_no_continuation_switch_before_lane_end() -> None:
    lane = FakeLane(length=100.0)
    nxt = FakeLane(index=("next", "node", 0))
    selection = _call(ego_lane=lane, position=lane.position(99.0, 0.0)[:2], refs=[lane], nxt=[nxt])
    assert selection.lane is lane and selection.source == SOURCE_ROUTE_CURRENT


def test_continuation_prefers_matching_lane_id() -> None:
    lane = FakeLane(index=("road", "node", 1))
    nxt_other = FakeLane(index=("next", "node", 0))
    nxt_match = FakeLane(index=("next", "node", 1))
    selection = _call(
        ego_lane=lane, position=lane.position(100.5, 0.0)[:2], refs=[lane], nxt=[nxt_other, nxt_match]
    )
    assert selection.lane is nxt_match
    assert selection.source == SOURCE_ROUTE_CONTINUATION
