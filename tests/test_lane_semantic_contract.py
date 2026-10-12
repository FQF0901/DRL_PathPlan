"""R21-0：lane_semantic 通道契约测试（schema v7）。

覆盖（门：几何查询误差 lat <0.05 m / heading <0.5° / curv 相对 <10%；坐标 round-trip；
禁 raw lane_id 数值输入）：

1. 形状 / 特征名 / dtype / mask 语义（无 target → mask=0；投影失败 → target_lane_valid=0）；
2. 直道解析几何：lat / heading_error / curvature / margins / distance_to_lane_end 与
   MetaDrive 口径一致（同号、同量纲）；
3. 圆弧解析曲率（右转负；near/far 采样）；
4. 左右可行驶 margin 跨同向相邻车道扩展、遇对向车道即停；
5. lane_change_required_left/right（同 road lane_id 大小 / 跨 road 横向位置）；
6. 坐标 round-trip（``lane.position(s, lat)`` → 投影回 ``(s, lat)``）；
7. 无 raw lane_id：仅改 lane index id（几何不变）→ 12 维特征逐位不变。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from env.obs.lane_semantic import (
    LANE_SEMANTIC_DIM,
    LANE_SEMANTIC_FEATURE_NAMES,
    LaneSemanticChannel,
    lane_curvature,
)


# --------------------------------------------------------------------------- #
# 假 env / 车道 / 车辆（不建 MetaDrive env）
# --------------------------------------------------------------------------- #
class FakeStraightLane:
    """直道（lat 正 = 车道方向右侧，与 MetaDrive StraightLane 一致）。"""

    def __init__(self, *, x0=0.0, y0=0.0, heading=0.0, length=300.0, width=3.5,
                 index=("road", "node", 0), speed_limit=1000.0):
        self.x0, self.y0, self.heading = float(x0), float(y0), float(heading)
        self.length, self.width = float(length), float(width)
        self.index = tuple(index)
        self.speed_limit = float(speed_limit)
        self.line_types = ["solid", "broken"]
        self.start = np.array([self.x0, self.y0, 0.0])
        self.end = np.array(
            [self.x0 + self.length * math.cos(self.heading), self.y0 + self.length * math.sin(self.heading), 0.0]
        )
        self.direction = np.array([math.cos(self.heading), math.sin(self.heading), 0.0])
        self.direction_lateral = np.array([self.direction[1], -self.direction[0], 0.0])

    def position(self, longitudinal, lateral=0.0):
        point = self.start + float(longitudinal) * self.direction + float(lateral) * self.direction_lateral
        return np.asarray(point, dtype=np.float64)

    def heading_theta_at(self, longitudinal):
        return self.heading

    def local_coordinates(self, position):
        delta = np.asarray(position, dtype=np.float64)[:2] - self.start[:2]
        return float(delta @ self.direction[:2]), float(delta @ self.direction_lateral[:2])


class FakeCircularLane:
    """右转圆弧（heading 随 s 递减：curvature = -1/R；center 在右侧）。"""

    def __init__(self, *, radius=50.0, length=80.0, width=3.5, index=("road", "arc", 0)):
        self.radius, self.length, self.width = float(radius), float(length), float(width)
        self.index = tuple(index)
        self.line_types = ["solid", "broken"]
        self.start = np.array([0.0, 0.0, 0.0])
        self.center = np.array([0.0, self.radius, 0.0])
        self.direction = np.array([1.0, 0.0, 0.0])
        self.end = self.position(self.length, 0.0)
        self.direction_lateral = np.array([0.0, -1.0, 0.0])  # 右侧（指向圆心）

    def position(self, longitudinal, lateral=0.0):
        phi = float(longitudinal) / self.radius
        r = self.radius - float(lateral)
        return np.array([r * math.sin(phi), self.radius - r * math.cos(phi), 0.0], dtype=np.float64)

    def heading_theta_at(self, longitudinal):
        return -float(longitudinal) / self.radius

    def local_coordinates(self, position):
        delta = np.asarray(position, dtype=np.float64)[:2] - self.center[:2]
        r = float(np.linalg.norm(delta))
        phi = math.atan2(delta[0], -delta[1])
        return float(phi * self.radius), float(self.radius - r)


class FakeNav:
    def __init__(self, current=(), nxt=()):
        self.current_ref_lanes = list(current)
        self.next_ref_lanes = list(nxt)


class FakeEgo:
    def __init__(self, *, position=(0.0, 0.0), heading=0.0, lane=None, nav=None):
        self.position = np.array([float(position[0]), float(position[1]), 0.0])
        self.heading_theta = float(heading)
        self.lane = lane
        self.navigation = nav

    def convert_to_local_coordinates(self, point, origin):
        delta = np.asarray(point, dtype=np.float64)[:2] - np.asarray(origin, dtype=np.float64)[:2]
        c, s = math.cos(self.heading_theta), math.sin(self.heading_theta)
        return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], 0.0])


class FakeEnv:
    def __init__(self, ego, graph=None):
        self.agent = ego
        self.current_map = type(
            "M", (), {"road_network": type("G", (), {"graph": dict(graph or {})})()}
        )()


def _feat(env) -> np.ndarray:
    feats, mask = LaneSemanticChannel().build(env, None)
    assert feats.shape == (1, LANE_SEMANTIC_DIM) and mask.shape == (1, )
    return feats[0]


# --------------------------------------------------------------------------- #
# 1) 契约 / 形状
# --------------------------------------------------------------------------- #
def test_feature_names_and_no_raw_lane_id_naming() -> None:
    assert LANE_SEMANTIC_DIM == 12
    assert len(LANE_SEMANTIC_FEATURE_NAMES) == 12
    assert not any("lane_id" in name for name in LANE_SEMANTIC_FEATURE_NAMES)
    assert LANE_SEMANTIC_FEATURE_NAMES[0] == "ego_to_target_lane_lateral_offset"


def test_no_target_yields_zero_mask() -> None:
    env = FakeEnv(FakeEgo(lane=None, nav=None))
    feats, mask = LaneSemanticChannel().build(env, None)
    assert mask[0] == 0.0
    np.testing.assert_allclose(feats, 0.0)


def test_ego_fallback_valid_when_no_route() -> None:
    lane = FakeStraightLane()
    env = FakeEnv(FakeEgo(position=(10.0, 0.0), lane=lane))
    feats = _feat(env)
    assert feats[6] == pytest.approx(1.0)  # target_lane_valid
    assert feats[7] == pytest.approx(0.25)  # route_association_confidence（ego 兜底）


def test_projection_failure_yields_invalid_features() -> None:
    class _BadLane(FakeStraightLane):
        def local_coordinates(self, position):
            raise ValueError("circular lane projection failed")

    lane = _BadLane()
    env = FakeEnv(FakeEgo(position=(10.0, 0.0), lane=lane))
    feats, mask = LaneSemanticChannel().build(env, None)
    assert mask[0] == 1.0  # target 已选出
    assert feats[0, 6] == 0.0  # target_lane_valid = 0
    np.testing.assert_allclose(feats[0], 0.0)


# --------------------------------------------------------------------------- #
# 2) 直道解析几何
# --------------------------------------------------------------------------- #
def test_straight_geometry_matches_metadrive_query() -> None:
    lane = FakeStraightLane(heading=0.3, length=200.0, width=3.5)
    s_ego, lat_ego = 40.0, 1.2  # lat 正 = 右侧
    point = lane.position(s_ego, lat_ego)
    ego = FakeEgo(position=point[:2], heading=0.3 + 0.05, lane=lane)
    feats = _feat(FakeEnv(ego))

    assert feats[0] == pytest.approx(lat_ego, abs=1e-5)  # lateral offset（MetaDrive 口径）
    assert feats[1] == pytest.approx(0.05, abs=1e-5)  # heading error（正 = 偏左/逆时针）
    assert feats[2] == pytest.approx(0.0, abs=1e-6)  # 直道曲率 0
    assert feats[3] == pytest.approx(0.0, abs=1e-6)
    assert feats[4] == pytest.approx(3.5 / 2.0 + lat_ego, abs=1e-5)  # left margin
    assert feats[5] == pytest.approx(3.5 / 2.0 - lat_ego, abs=1e-5)  # right margin
    assert feats[6] == pytest.approx(1.0)
    assert feats[10] == pytest.approx(200.0 - s_ego, abs=1e-5)  # distance_to_lane_end
    # 无 next 链 → distance_to_next_topology_change = 上限 cap（路线尽头）
    assert feats[11] == pytest.approx(200.0, abs=1e-5)


def test_topology_distance_uses_lane_end_when_route_continues() -> None:
    lane = FakeStraightLane(length=200.0)
    nxt = FakeStraightLane(index=("next", "node", 0))
    env = FakeEnv(FakeEgo(position=(40.0, 0.0), lane=lane, nav=FakeNav([lane], [nxt])))
    feats = _feat(env)
    assert feats[10] == pytest.approx(160.0, abs=1e-5)
    assert feats[11] == pytest.approx(160.0, abs=1e-5)


# --------------------------------------------------------------------------- #
# 3) 圆弧曲率（near/far 采样；右转负）
# --------------------------------------------------------------------------- #
def test_curvature_near_far_on_circular_lane() -> None:
    lane = FakeCircularLane(radius=50.0, length=80.0)
    ego = FakeEgo(position=lane.position(20.0, 0.0)[:2], heading=lane.heading_theta_at(20.0), lane=lane)
    feats = _feat(FakeEnv(ego))
    assert feats[2] == pytest.approx(-1.0 / 50.0, rel=1e-3)  # near = s+10
    assert feats[3] == pytest.approx(-1.0 / 50.0, rel=1e-3)  # far = s+30
    assert lane_curvature(lane, 20.0) == pytest.approx(-1.0 / 50.0, rel=1e-3)


def test_curvature_clamped_at_lane_end() -> None:
    lane = FakeCircularLane(radius=50.0, length=15.0)
    ego = FakeEgo(position=lane.position(10.0, 0.0)[:2], heading=lane.heading_theta_at(10.0), lane=lane)
    feats = _feat(FakeEnv(ego))
    # near(20) 与 far(40) 均 clamp 到 length=15 → 同一值
    assert feats[2] == pytest.approx(feats[3], abs=1e-9)
    assert feats[2] == pytest.approx(-1.0 / 50.0, rel=1e-3)


# --------------------------------------------------------------------------- #
# 4) 可行驶 margin（跨同向相邻车道；对向即停）
# --------------------------------------------------------------------------- #
def test_margins_expand_over_same_direction_siblings() -> None:
    left = FakeStraightLane(y0=3.5, index=("road", "node", 0))
    mid = FakeStraightLane(y0=0.0, index=("road", "node", 1))
    right = FakeStraightLane(y0=-3.5, index=("road", "node", 2))
    env = FakeEnv(
        FakeEgo(position=(10.0, 0.0), lane=mid),
        graph={"road": {"node": [left, mid, right]}},
    )
    feats = _feat(env)
    assert feats[4] == pytest.approx(3.5 / 2.0 + 3.5, abs=1e-5)  # 左：mid 半宽 + left 全宽
    assert feats[5] == pytest.approx(3.5 / 2.0 + 3.5, abs=1e-5)


def test_margins_stop_at_opposite_direction_lane() -> None:
    opposite = FakeStraightLane(y0=3.5, heading=math.pi, index=("road", "node", 0))
    mid = FakeStraightLane(y0=0.0, index=("road", "node", 1))
    env = FakeEnv(
        FakeEgo(position=(10.0, 0.0), lane=mid),
        graph={"road": {"node": [opposite, mid]}},
    )
    feats = _feat(env)
    assert feats[4] == pytest.approx(3.5 / 2.0, abs=1e-5)  # 对向车道不计入


# --------------------------------------------------------------------------- #
# 5) lane change 需求
# --------------------------------------------------------------------------- #
def test_lane_change_required_left_when_target_is_left() -> None:
    left = FakeStraightLane(y0=3.5, index=("road", "node", 0))
    mid = FakeStraightLane(y0=0.0, index=("road", "node", 1))
    # route 只包含 left（ego 在 mid，route 要求变到左道）
    env = FakeEnv(FakeEgo(position=(10.0, 0.0), lane=mid, nav=FakeNav([left], [])))
    feats = _feat(env)
    assert feats[6] == pytest.approx(1.0)
    assert feats[8] == pytest.approx(1.0) and feats[9] == pytest.approx(0.0)
    # left 车道中心在 y=+3.5，ego 在 y=0 → 车道系 lat = +3.5（ego 在中心线右侧）
    assert feats[0] == pytest.approx(3.5, abs=1e-5)


def test_lane_change_required_right_when_target_is_right() -> None:
    mid = FakeStraightLane(y0=0.0, index=("road", "node", 1))
    right = FakeStraightLane(y0=-3.5, index=("road", "node", 2))
    env = FakeEnv(FakeEgo(position=(10.0, 0.0), lane=mid, nav=FakeNav([right], [])))
    feats = _feat(env)
    assert feats[8] == pytest.approx(0.0) and feats[9] == pytest.approx(1.0)


def test_no_lane_change_flag_when_ego_on_target() -> None:
    lane = FakeStraightLane()
    env = FakeEnv(FakeEgo(position=(10.0, 0.4), lane=lane, nav=FakeNav([lane], [])))
    feats = _feat(env)
    assert feats[8] == 0.0 and feats[9] == 0.0


# --------------------------------------------------------------------------- #
# 6) 坐标 round-trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lane_cls,kwargs", [(FakeStraightLane, {"heading": 0.7}), (FakeCircularLane, {})])
def test_coordinate_round_trip(lane_cls, kwargs) -> None:
    lane = lane_cls(**kwargs)
    for s in (0.0, 5.0, 0.5 * lane.length, min(0.9 * lane.length, lane.length - 0.1)):
        for lat in (-1.5, 0.0, 1.5):
            point = lane.position(s, lat)
            long, lateral = lane.local_coordinates(point)
            assert long == pytest.approx(s, abs=1e-6)
            assert lateral == pytest.approx(lat, abs=1e-6)
            # 通道消费口径：把 ego 放到该点 → 特征还原 (s, lat)
            ego = FakeEgo(position=point[:2], heading=lane.heading_theta_at(s), lane=lane)
            feats = _feat(FakeEnv(ego))
            assert feats[0] == pytest.approx(lat, abs=1e-5)
            assert feats[10] == pytest.approx(lane.length - s, abs=1e-5)


# --------------------------------------------------------------------------- #
# 7) 无 raw lane_id：仅改 id（几何不变）→ 特征逐位不变
# --------------------------------------------------------------------------- #
def test_features_invariant_to_lane_id_relabeling() -> None:
    def _build(ids):
        lane = FakeStraightLane(index=("road", "node", ids[0]))
        env = FakeEnv(FakeEgo(position=(10.0, 0.5), lane=lane, nav=FakeNav([lane], [])))
        return _feat(env)

    a = _build([0])
    b = _build([7])
    np.testing.assert_array_equal(a, b)
