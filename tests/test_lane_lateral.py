"""env 层横向偏移注入（``lane_center`` 项的输入）单元测试（f23b925 复活适配版）。

全部用鸭子类型 stub（不建 env；按仓库约定在用例内惰性 import ``env.metadrive_env``）。
覆盖 ``lane_lateral_info`` 的读取顺序与降级：
当前车道 ``local_coordinates`` → 导航参考车道 → ``dist_to_left/right_side`` 边界回退 → 键缺失。
"""

from __future__ import annotations

import pytest

from reward_model import make_term


class _StubLane:
    def __init__(self, lateral: float = 0.0, error: Exception | None = None) -> None:
        self.lateral = float(lateral)
        self.error = error
        self.calls = 0

    def local_coordinates(self, position):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return float(position[0]), self.lateral


class _StubAgent:
    def __init__(
        self,
        *,
        lane=None,
        position=(0.0, 0.0),
        ref_lanes=None,
        left=None,
        right=None,
    ) -> None:
        self.position = position
        self.lane = lane
        if ref_lanes is not None:
            self.navigation = type("_Nav", (), {"current_ref_lanes": ref_lanes})()
        self.dist_to_left_side = left
        self.dist_to_right_side = right


def _lane_lateral_info():
    pytest.importorskip("metadrive")
    from env.metadrive_env import lane_lateral_info  # 惰性：本文件不在顶层 import metadrive

    return lane_lateral_info


def test_lane_lateral_info_prefers_current_lane() -> None:
    info = _lane_lateral_info()(_StubAgent(lane=_StubLane(lateral=0.7), left=2.9, right=1.1))
    assert set(info) == {"lane_lateral_offset"}
    assert info["lane_lateral_offset"] == pytest.approx(0.7)
    # 只做一次投影查询（廉价性）
    lane = _StubLane(lateral=0.1)
    _lane_lateral_info()(_StubAgent(lane=lane))
    assert lane.calls == 1


def test_lane_lateral_info_navigation_reference_lane_fallback() -> None:
    lane = _StubLane(lateral=-0.4)
    info = _lane_lateral_info()(_StubAgent(lane=None, ref_lanes=[lane]))
    assert info == {"lane_lateral_offset": pytest.approx(-0.4)}


def test_lane_lateral_info_boundary_fallback_when_projection_fails() -> None:
    lane = _StubLane(error=ValueError("circular lane projection failed"))
    info = _lane_lateral_info()(_StubAgent(lane=lane, left=2.75, right=1.25))
    assert set(info) == {"dist_to_left_side", "dist_to_right_side"}
    assert info["dist_to_left_side"] == pytest.approx(2.75)
    assert info["dist_to_right_side"] == pytest.approx(1.25)


def test_lane_lateral_info_missing_lane_yields_missing_keys() -> None:
    lane_lateral_info = _lane_lateral_info()
    assert lane_lateral_info(_StubAgent()) == {}
    assert lane_lateral_info(None) == {}
    # 无导航时 MetaDrive 的 dist_to_* = (0, 0) 是占位而不是"居中"
    assert lane_lateral_info(_StubAgent(left=0.0, right=0.0)) == {}
    # 单侧边界 / 非有限值 → 缺失
    assert lane_lateral_info(_StubAgent(left=2.0, right=None)) == {}
    assert lane_lateral_info(_StubAgent(left=float("nan"), right=1.0)) == {}


def test_lane_center_term_consumes_injected_offset() -> None:
    lane_lateral_info = _lane_lateral_info()
    term = make_term("lane_center", weight=-0.1)
    beyond = lane_lateral_info(_StubAgent(lane=_StubLane(lateral=0.75)))
    assert term.compute(beyond) == pytest.approx(0.5)  # |0.75| - 0.25
    inside = lane_lateral_info(_StubAgent(lane=_StubLane(lateral=0.2)))
    assert term.compute(inside) == 0.0  # 死区内
    boundary = lane_lateral_info(_StubAgent(lane=_StubLane(lateral=3.0), left=3.0, right=1.0))
    assert term.compute(boundary) == pytest.approx(2.75)  # 截断 3 m
