"""Round-18A：LD 旁路 debug 元数据（默认关、obs 输出不变）+ 注释漂移修正回归。

- ``debug=False``（默认）→ ``last_debug is None`` 且 feats/mask 与 debug 打开时**逐位相同**；
- ``debug=True`` → 逐槽元数据（lane_index/priority/order/offset/is_ego_lane/is_nav_current/is_nav_next/s）；
- 槽序：当前车道（offset 升序）在前、secondary 按 (offset, priority, order) 在后（与既有口径一致）。
"""

from __future__ import annotations

import numpy as np

from env.obs.ld import LDChannel, LD_OFFSETS_M


class _StubLane:
    def __init__(self, index, length=200.0, line_types=("broken", "solid"), speed_limit=8.0):
        self.index = index
        self.length = float(length)
        self.line_types = tuple(line_types)
        self.speed_limit = float(speed_limit)

    def position(self, s, lat=0.0):
        return np.asarray([float(s), 0.0], dtype=np.float32)

    def heading_theta_at(self, s):
        return 0.0

    def local_coordinates(self, position):
        return float(position[0]), 0.0


class _StubEgo:
    def __init__(self, lane, nav=None):
        self.lane = lane
        self.navigation = nav
        self.position = np.asarray([0.0, 0.0], dtype=np.float32)
        self.heading_theta = 0.0

    def convert_to_local_coordinates(self, point, origin=None):
        return np.asarray(point, dtype=np.float32)[:2]


class _StubNav:
    def __init__(self, current, nxt):
        self.current_ref_lanes = list(current)
        self.next_ref_lanes = list(nxt)


class _StubGraph:
    def __getitem__(self, key):
        return []


class _StubEnv:
    def __init__(self, ego, nav):
        self.agent = ego
        self.current_map = type(
            "M", (), {"road_network": type("G", (), {"graph": _StubGraph()})()}
        )()


def _make_env():
    ego_lane = _StubLane((0, 0, 0))
    next_lane = _StubLane((1, 0, 0))
    nav = _StubNav([ego_lane], [next_lane])
    return _StubEnv(_StubEgo(ego_lane, nav), nav), ego_lane, next_lane


def test_debug_off_by_default_and_bitwise_same() -> None:
    env, _ego, _nxt = _make_env()
    plain = LDChannel()
    assert plain.debug is False
    feats_a, mask_a = plain.build(env)
    assert plain.last_debug is None

    debug = LDChannel(debug=True)
    feats_b, mask_b = debug.build(env)
    assert np.array_equal(feats_a, feats_b)
    assert np.array_equal(mask_a, mask_b)
    assert debug.last_debug is not None


def test_debug_metadata_matches_slot_order() -> None:
    env, ego_lane, next_lane = _make_env()
    channel = LDChannel(debug=True)
    feats, mask = channel.build(env)
    meta = channel.last_debug
    assert meta is not None
    n_valid = int(mask.sum())
    assert meta["num_valid"] == n_valid == len(meta["lane_index"])
    # 当前车道 5 个 offset 在前（offset 升序）
    primary_offsets = meta["offset_m"][: len(LD_OFFSETS_M)]
    assert primary_offsets == list(LD_OFFSETS_M)
    assert meta["is_ego_lane"][: len(LD_OFFSETS_M)] == [True] * len(LD_OFFSETS_M)
    assert meta["is_nav_current"][: len(LD_OFFSETS_M)] == [True] * len(LD_OFFSETS_M)
    # secondary 为 next lane（is_nav_next=True，is_ego_lane=False）
    assert all(meta["is_ego_lane"][len(LD_OFFSETS_M):]) is False
    assert all(meta["is_nav_next"][len(LD_OFFSETS_M):]) is True
    assert meta["lane_index"][len(LD_OFFSETS_M)] == tuple(next_lane.index)
    assert meta["ego_lane_index"] == tuple(ego_lane.index)
    # 采样弧长 = s0 + offset（s0=0）
    assert meta["s_m"][0] == LD_OFFSETS_M[0]
    # feats 与元数据一致：primary 槽的 dy = 0（直道，ego 在中心线）
    assert np.allclose(feats[: len(LD_OFFSETS_M), 1], 0.0)
