"""obs schema v6（v8 结构重构）单测（原 v5 lane/ttc 用例改写）。

覆盖：
1. LD 采样 offset ``{0,20,40,60,80}``：0 m 点 = ego 投影点（近场横向锚）、当前车道
   5 primary 槽 + 其余车道环填充、短车道降级；
2. lane/ttc 通道删除：模块文件 / DEFAULT_CHANNELS / builder / schema / collect_expert
   全链不存在；
3. schema v6 manifest/指纹与 ``tools/collect_expert.py`` 透传。

纯 NumPy + 假 env（不建 MetaDrive env；真实几何在 GPU 冒烟里验证）。
"""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

import env.obs.od as od_module
from env.obs.builder import DEFAULT_CHANNELS, ObservationBuilder
from env.obs.ld import LD_OFFSETS_M, LDChannel
from env.obs.schema import schema_manifest

# --------------------------------------------------------------------------- #
# 假 env / 车道 / 车辆
# --------------------------------------------------------------------------- #


class FakeStraightLane:
    """直道（车道系：lat 正 = 车道方向右侧，与 MetaDrive StraightLane 一致）。"""

    def __init__(
        self,
        *,
        x0: float = 0.0,
        y0: float = 0.0,
        heading: float = 0.0,
        length: float = 300.0,
        width: float = 3.5,
        index=("road", "node", 0),
        speed_limit: float = 1000.0,
    ):
        self.x0, self.y0, self.heading = float(x0), float(y0), float(heading)
        self.length, self.width = float(length), float(width)
        self.index = tuple(index)
        self.speed_limit = float(speed_limit)
        self.line_types = ["solid", "broken"]

    @property
    def _c(self) -> float:
        return math.cos(self.heading)

    @property
    def _s(self) -> float:
        return math.sin(self.heading)

    def position(self, longitudinal, lateral=0.0):
        # direction = (c, s)；direction_lateral = (s, -c)
        dx = float(longitudinal) * self._c + float(lateral) * self._s
        dy = float(longitudinal) * self._s - float(lateral) * self._c
        return np.array([self.x0 + dx, self.y0 + dy, 0.0], dtype=np.float64)

    def heading_theta_at(self, longitudinal):
        return self.heading

    def local_coordinates(self, position):
        px = float(position[0]) - self.x0
        py = float(position[1]) - self.y0
        return float(px * self._c + py * self._s), float(px * self._s - py * self._c)


class FakeEgo:
    def __init__(self, *, x: float = 0.0, y: float = 0.0, heading: float = 0.0,
                 vx: float = 0.0, vy: float = 0.0, lane=None, ref_lanes=None):
        self.id, self.name = "ego", "ego"
        self.position = np.array([x, y, 0.0], dtype=np.float64)
        self.velocity = np.array([vx, vy, 0.0], dtype=np.float64)
        self.heading_theta = float(heading)
        self.speed = float(math.hypot(vx, vy))
        self.last_speed = self.speed
        self.steering = 0.0
        self.last_heading_dir = np.array([math.cos(heading), math.sin(heading), 0.0])
        self.lane = lane
        if ref_lanes is not None:
            nav = type("_Nav", (), {"current_ref_lanes": list(ref_lanes), "next_ref_lanes": []})()
            self.navigation = nav
        else:
            self.navigation = None

    def convert_to_local_coordinates(self, point, origin):
        p = np.asarray(point, dtype=np.float64)[:2] - np.asarray(origin, dtype=np.float64)[:2]
        c, s = math.cos(self.heading_theta), math.sin(self.heading_theta)
        return np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1], 0.0], dtype=np.float64)


class FakeVehicle:
    """OD 所需最小车辆接口。"""

    def __init__(self, name: str, x: float = 0.0, y: float = 0.0, vx: float = 0.0,
                 vy: float = 0.0, heading: float = 0.0):
        self.id, self.name = str(name), str(name)
        self.position = np.array([x, y, 0.0], dtype=np.float64)
        self.velocity = np.array([vx, vy, 0.0], dtype=np.float64)
        self.heading_theta = float(heading)
        self.LENGTH, self.WIDTH = 4.5, 1.8


class FakeEngine:
    def __init__(self, objects):
        self.objects = {obj.id: obj for obj in objects}

    def get_objects(self):
        return self.objects


class FakeRoadNetwork:
    def __init__(self, graph):
        self.graph = graph


class FakeMap:
    def __init__(self, graph):
        self.road_network = FakeRoadNetwork(graph)


class FakeEnv:
    def __init__(self, ego: FakeEgo, others=(), step: int = 0, map_graph=None):
        self._objects = [ego, *others]
        self.engine = FakeEngine(self._objects)
        self.episode_step = int(step)
        self.config = {}
        self.prev_policy_action = None
        self.current_map = FakeMap(map_graph) if map_graph is not None else None

    @property
    def agent(self):
        return self._objects[0]

    def add(self, obj) -> None:
        self._objects.append(obj)
        self.engine.objects[obj.id] = obj


@pytest.fixture(autouse=True)
def _patch_base_vehicle(monkeypatch):
    monkeypatch.setattr(od_module, "BaseVehicle", FakeVehicle)


# --------------------------------------------------------------------------- #
# 1) LD：offset {0,20,40,60,80}（0 m = ego 投影点）
# --------------------------------------------------------------------------- #


def test_ld_offsets_include_zero_projection_point() -> None:
    """默认 offset {0,20,40,60,80}；0 m 点 = ego 投影点（dx=0，dy=-d_lat）。"""
    lane = FakeStraightLane(length=300.0)
    ego = FakeEgo(x=10.0, y=0.7, heading=0.0, lane=lane)  # ego 在中心线左侧 0.7 m
    feats, mask = LDChannel().build(FakeEnv(ego), None)

    assert LDChannel().offsets == tuple(LD_OFFSETS_M) == (0.0, 20.0, 40.0, 60.0, 80.0)
    assert mask[:5].tolist() == [1.0] * 5
    # slot 0：ego 投影点（s0 = ego 纵向投影）→ 自车系 dx=0、dy = -横向偏差
    assert feats[0, 0] == pytest.approx(0.0)
    assert feats[0, 1] == pytest.approx(-0.7)
    # slot 1..4：20..80 m 远场，同一车道中心线 → dy 恒为 -0.7
    for k, offset in enumerate(LD_OFFSETS_M[1:], start=1):
        assert feats[k, 0] == pytest.approx(offset)
        assert feats[k, 1] == pytest.approx(-0.7)
    assert mask[5:].sum() == 0.0


def test_ld_ring_fill_others_starts_at_zero_offset() -> None:
    """当前车道占 5 primary 槽；相邻车道从 0 m 环开始填充。"""
    lane = FakeStraightLane(length=300.0, index=("road", "node", 0))
    sibling = FakeStraightLane(y0=-3.5, length=300.0, index=("road", "node", 1))
    env = FakeEnv(FakeEgo(x=10.0, y=0.7, lane=lane), map_graph={"road": {"node": [lane, sibling]}})

    feats, mask = LDChannel().build(env, None)
    for k, offset in enumerate(LD_OFFSETS_M):  # primary：当前车道 0/20/40/60/80
        assert mask[k] == 1.0
        assert feats[k, 0] == pytest.approx(offset)
        assert feats[k, 1] == pytest.approx(-0.7)
    for k, offset in enumerate(LD_OFFSETS_M):  # secondary：相邻车道 0/20/40/60/80（环优先）
        assert mask[5 + k] == 1.0
        assert feats[5 + k, 0] == pytest.approx(offset)
        assert feats[5 + k, 1] == pytest.approx(-4.2)
    assert mask[10:].sum() == 0.0


def test_ld_short_lane_drops_far_offsets_and_ring_fills_others() -> None:
    """当前车道 <60 m：60/80 槽丢弃（mask=0）；相邻车道继续按环填充。"""
    lane = FakeStraightLane(length=50.0, index=("road", "node", 0))
    sibling = FakeStraightLane(y0=-3.5, length=300.0, index=("road", "node", 1))
    env = FakeEnv(FakeEgo(lane=lane), map_graph={"road": {"node": [lane, sibling]}})
    feats, mask = LDChannel().build(env, None)

    # primary：当前车道 0/20/40（60/80 超车道末端丢弃）
    assert [float(mask[k]) for k in (0, 1, 2)] == [1.0] * 3
    assert [float(feats[k, 0]) for k in (0, 1, 2)] == pytest.approx([0.0, 20.0, 40.0])
    # 后续槽：相邻车道 0/20/40/60/80（环优先；当前车道 60/80 已丢弃）
    assert [float(mask[k]) for k in (3, 4, 5, 6, 7)] == [1.0] * 5
    assert [float(feats[k, 0]) for k in (3, 4, 5, 6, 7)] == pytest.approx([0.0, 20.0, 40.0, 60.0, 80.0])
    assert [float(feats[k, 1]) for k in (3, 4, 5, 6, 7)] == pytest.approx([-3.5] * 5)
    assert mask[8:].sum() == 0.0


# --------------------------------------------------------------------------- #
# 2) lane/ttc 删除
# --------------------------------------------------------------------------- #


def test_lane_ttc_modules_and_channels_removed() -> None:
    """模块文件已删除；DEFAULT_CHANNELS / builder 不再有 lane/ttc；请求即报未知通道。"""
    assert importlib.util.find_spec("env.obs.lane") is None
    assert importlib.util.find_spec("env.obs.ttc") is None
    assert "lane" not in DEFAULT_CHANNELS and "ttc" not in DEFAULT_CHANNELS
    assert {"lane", "ttc"}.isdisjoint(ObservationBuilder().feature_spec)

    with pytest.raises(ValueError, match="未知观测通道"):
        ObservationBuilder({"channels": ["lane"]})
    with pytest.raises(ValueError, match="未知观测通道"):
        ObservationBuilder({"channels": ["ttc"]})


def test_builder_emits_v6_channels_without_lane_ttc() -> None:
    """v6 builder 输出：ld/od/others 形状不变，obs 中无 lane/ttc 键。"""
    lane = FakeStraightLane()
    sibling = FakeStraightLane(y0=-3.5, index=("road", "node", 1))
    obj = FakeVehicle("obj", 20.0, 0.0, vx=5.0)
    ego = FakeEgo(vx=10.0, lane=lane)
    env = FakeEnv(ego, [obj], map_graph={"road": {"node": [lane, sibling]}})
    builder = ObservationBuilder({"channels": ["ego", "od", "ld", "nav", "signal", "others"]})
    obs = builder.build(env, None)

    assert "lane" not in obs and "ttc" not in obs
    assert obs["ld"].shape == (16, 7) and obs["ld_mask"].shape == (16, )
    assert obs["ld"][0, 0] == pytest.approx(0.0)  # 0 m = ego 投影点
    assert obs["od"].shape == (16, 9)
    assert obs["others"].shape == (1, 33)
    assert builder.feature_spec["ld"] == (16, 7)
    assert set(builder.feature_spec) == {"ego", "od", "ld", "nav", "signal", "others"}


# --------------------------------------------------------------------------- #
# 3) schema v6 / collect_expert
# --------------------------------------------------------------------------- #


def test_schema_v6_manifest_and_fingerprint() -> None:
    from env.obs import OBS_SCHEMA_VERSION, obs_fingerprint

    manifest = schema_manifest()
    assert OBS_SCHEMA_VERSION == 6 and manifest["schema_version"] == 6
    assert manifest["frame"]["ld"]["shape"] == [16, 7]
    assert "lane" not in manifest["frame"] and "ttc" not in manifest["frame"]
    assert "lane_layout" not in manifest and "ttc_layout" not in manifest
    assert manifest["ld_layout"]["offsets_m"] == list(LD_OFFSETS_M) == [0.0, 20.0, 40.0, 60.0, 80.0]
    assert "0 m 点 = ego 投影点" in manifest["ld_layout"]["near_field"]
    assert obs_fingerprint().startswith("v6-")


def test_collect_expert_passthrough_channels_v6() -> None:
    from tools.collect_expert import CURRENT_CHANNELS, _channel_dim, _channel_slots, _npz_schema_manifest

    assert "ld" in CURRENT_CHANNELS
    assert {"lane", "ttc"}.isdisjoint(CURRENT_CHANNELS)
    assert _channel_dim("ld") == LDChannel.feature_dim
    assert _channel_slots("ld", 16) == 16 and _channel_slots("ego", 16) == 1
    manifest = _npz_schema_manifest(num_slots=16, frames=6, others_dim=33, label_count=8)
    assert manifest["ld"]["shape"] == ["<N>", 16, LDChannel.feature_dim]
    assert "lane" not in manifest and "lane_mask" not in manifest
    assert "ttc" not in manifest and "ttc_mask" not in manifest
