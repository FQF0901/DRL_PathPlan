"""R21-0：LD 点补充（role / offset_s / group / route 关联）拓扑门测试。

覆盖（门：同 lane offset_s 严格递增；同组不混入他 lane；role 分类正确；route 关联正确；
**禁 raw lane_id 数值输入**）：

1. ``ld_offset_s_norm`` 在同 group 内严格递增（同车道内按 offset 递增采样）；
2. ``ld_group`` 仅观测内局部 group：同组 = 同 lane 对象、异组 = 异 lane、无效槽 -1；
3. ``ld_lane_role``：target / target_continuation / left_adjacent / right_adjacent / other；
4. ``ld_route_associated``：∈ current_ref ∪ next_ref；
5. lane id 重标定（保序）不改变补充数组（无 raw lane_id 数值输入）；
6. 无 ego / 短车道降级时的默认值。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from env.obs.lane_semantic import (
    LD_SUPPLEMENT_KEYS,
    ROLE_LEFT_ADJACENT,
    ROLE_OTHER,
    ROLE_RIGHT_ADJACENT,
    ROLE_TARGET,
    ROLE_TARGET_CONTINUATION,
)
from env.obs.ld import LDChannel, LD_OFFSETS_M


class FakeStraightLane:
    def __init__(self, *, x0=0.0, y0=0.0, heading=0.0, length=300.0, width=3.5,
                 index=("road", "node", 0)):
        self.x0, self.y0, self.heading = float(x0), float(y0), float(heading)
        self.length, self.width = float(length), float(width)
        self.index = tuple(index)
        self.speed_limit = 1000.0
        self.line_types = ["solid", "broken"]
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


class FakeNav:
    def __init__(self, current=(), nxt=()):
        self.current_ref_lanes = list(current)
        self.next_ref_lanes = list(nxt)


class FakeEgo:
    def __init__(self, *, position=(10.0, 0.0), heading=0.0, lane=None, nav=None):
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


def _topology_env():
    """ego 在 mid（id 1）；左右同路邻居 id 0/2；next 路段有 id 0（other）与 id 1（续接）。"""
    left = FakeStraightLane(y0=3.5, index=("road", "node", 0))
    mid = FakeStraightLane(y0=0.0, index=("road", "node", 1))
    right = FakeStraightLane(y0=-3.5, index=("road", "node", 2))
    nxt_other = FakeStraightLane(x0=300.0, y0=3.5, index=("next", "node", 0))
    nxt_match = FakeStraightLane(x0=300.0, y0=0.0, index=("next", "node", 1))
    ego = FakeEgo(position=(200.0, 0.0), lane=mid, nav=FakeNav([mid], [nxt_other, nxt_match]))
    env = FakeEnv(ego, graph={"road": {"node": [left, mid, right]}})
    return env, left, mid, right, nxt_other, nxt_match


def _build(env, num_slots=16):
    channel = LDChannel(num_slots=num_slots, debug=True)
    feats, mask = channel.build(env, None)
    supplements = channel.companions()
    return channel, feats, mask, supplements


def test_supplement_keys_and_no_raw_lane_id() -> None:
    env, *_ = _topology_env()
    channel, _feats, _mask, supplements = _build(env)
    assert set(supplements) == set(LD_SUPPLEMENT_KEYS)
    assert not any("lane_id" in key for key in supplements)
    for key in LD_SUPPLEMENT_KEYS:
        assert supplements[key].shape == (channel.num_slots, )


def test_offset_s_strictly_increasing_within_group() -> None:
    env, *_ = _topology_env()
    _channel, _feats, mask, supplements = _build(env)
    groups, offsets = supplements["ld_group"], supplements["ld_offset_s_norm"]
    valid = mask > 0.5
    for group in np.unique(groups[valid]):
        values = offsets[valid & (groups == group)]
        assert values.size >= 1
        assert np.all(np.diff(values) > 0), f"group={group} offset_s 非严格递增：{values}"


def test_group_does_not_mix_lanes() -> None:
    env, *_ = _topology_env()
    channel, _feats, mask, supplements = _build(env)
    groups = supplements["ld_group"]
    meta = channel.last_debug
    assert meta is not None
    lane_of_slot = meta["lane_index"]
    valid = mask > 0.5
    group_to_lane: dict[int, tuple] = {}
    for slot in np.flatnonzero(valid):
        group = int(groups[slot])
        lane_index = lane_of_slot[slot]
        if group in group_to_lane:
            assert group_to_lane[group] == lane_index, "同组混入他 lane"
        else:
            group_to_lane[group] = lane_index
    assert len(set(group_to_lane.values())) == len(group_to_lane), "不同 group 指向同一 lane"
    assert set(group_to_lane) == set(range(len(group_to_lane))), "group id 应为首次出现顺序 0..K-1"
    # 无效槽 = -1
    assert np.all(groups[~valid] == -1)


def test_role_classification() -> None:
    env, left, mid, right, nxt_other, nxt_match = _topology_env()
    channel, _feats, mask, supplements = _build(env)
    roles = supplements["ld_lane_role"]
    meta = channel.last_debug
    assert meta is not None
    lane_index = meta["lane_index"]
    valid = mask > 0.5
    for slot in np.flatnonzero(valid):
        idx = lane_index[slot]
        if idx == mid.index:
            assert roles[slot] == ROLE_TARGET
        elif idx == left.index:
            assert roles[slot] == ROLE_LEFT_ADJACENT
        elif idx == right.index:
            assert roles[slot] == ROLE_RIGHT_ADJACENT
        elif idx == nxt_match.index:
            assert roles[slot] == ROLE_TARGET_CONTINUATION
        elif idx == nxt_other.index:
            assert roles[slot] == ROLE_OTHER
        else:  # pragma: no cover - 拓扑固定，不应出现
            raise AssertionError(f"未知槽 lane_index={idx}")
    # 目标/续接/左右邻居都至少出现一次（拓扑可观测）
    assert set(roles[valid].tolist()) >= {
        ROLE_TARGET,
        ROLE_LEFT_ADJACENT,
        ROLE_RIGHT_ADJACENT,
        ROLE_TARGET_CONTINUATION,
    }


def test_route_associated_flag() -> None:
    env, left, mid, right, nxt_other, nxt_match = _topology_env()
    channel, _feats, mask, supplements = _build(env)
    route = supplements["ld_route_associated"]
    meta = channel.last_debug
    assert meta is not None
    lane_index = meta["lane_index"]
    valid = mask > 0.5
    for slot in np.flatnonzero(valid):
        idx = lane_index[slot]
        expected = 1.0 if idx in (mid.index, nxt_other.index, nxt_match.index) else 0.0
        assert route[slot] == pytest.approx(expected)
    assert route[~valid].sum() == 0.0


def test_lane_id_relabeling_preserves_supplements() -> None:
    env_a, *_ = _topology_env()
    _c_a, _f_a, _m_a, sup_a = _build(env_a)
    # 保序 + 保序键重标定（整体 ×2：road 0/1/2 → 0/2/4；next 0/1 → 0/2）——
    # |Δ lane_id| 与左右关系（大小序）不变，槽序不变；若补充数组含 raw lane_id 数值则必然不等。
    left = FakeStraightLane(y0=3.5, index=("road", "node", 0))
    mid = FakeStraightLane(y0=0.0, index=("road", "node", 2))
    right = FakeStraightLane(y0=-3.5, index=("road", "node", 4))
    nxt_other = FakeStraightLane(x0=300.0, y0=3.5, index=("next", "node", 0))
    nxt_match = FakeStraightLane(x0=300.0, y0=0.0, index=("next", "node", 2))
    ego = FakeEgo(position=(200.0, 0.0), lane=mid, nav=FakeNav([mid], [nxt_other, nxt_match]))
    env_b = FakeEnv(ego, graph={"road": {"node": [left, mid, right]}})
    _c_b, _f_b, _m_b, sup_b = _build(env_b)
    for key in LD_SUPPLEMENT_KEYS:
        np.testing.assert_array_equal(sup_a[key], sup_b[key], err_msg=f"{key} 依赖了 raw lane_id")


def test_defaults_without_ego_or_on_short_lane() -> None:
    class _NoEgo:
        @property
        def agent(self):
            raise AssertionError("no agent")

        current_map = None

    channel = LDChannel()
    feats, mask = channel.build(_NoEgo(), None)
    assert mask.sum() == 0.0 and feats.sum() == 0.0
    supplements = channel.companions()
    assert np.all(supplements["ld_lane_role"] == -1)
    assert np.all(supplements["ld_group"] == -1)
    assert np.all(supplements["ld_offset_s_norm"] == 0.0)
    assert np.all(supplements["ld_route_associated"] == 0.0)

    # 短车道：只放得下 1 个 offset（length=6 < 10）→ 其余槽 role/group=-1
    short = FakeStraightLane(length=6.0, index=("road", "node", 0))
    env = FakeEnv(FakeEgo(position=(0.0, 0.0), lane=short))
    channel = LDChannel(debug=True)
    _feats, mask = channel.build(env, None)
    supplements = channel.companions()
    assert int(mask.sum()) == 1
    valid = mask > 0.5
    assert supplements["ld_lane_role"][valid][0] == ROLE_TARGET
    assert np.all(supplements["ld_lane_role"][~valid] == -1)
    assert np.all(supplements["ld_group"][~valid] == -1)
    # 采样 offset 与 mask 对应（第一个 offset = 5 m）
    assert channel.last_debug is not None and channel.last_debug["s_m"][0] == pytest.approx(5.0)
    assert LD_OFFSETS_M[0] == 5.0
