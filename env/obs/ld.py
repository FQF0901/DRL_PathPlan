"""LD 通道：当前 / 相邻 / 下一参考车道的中心线采样 top-16 点。

采样与选取规则
--------------
- 候选车道（去重，优先级从高到低）：

  1. ego 当前车道（``env.agent.lane``；缺失时退化为 ``current_ref_lanes[0]``）；
  2. 下一参考车道中"同 lane id"的续接车道，否则 ``next_ref_lanes[0]``；
  3. 其余 ``next_ref_lanes``（按 |lane_id - ego_lane_id| 排序）；
  4. 同路相邻车道（``road_network.graph[from][to]`` 去掉当前车道，按 |Δlane_id| 排序，左侧优先）。

- 沿每条候选车道中心线按 ``s = clamp(ego 投影 long, 0, length) + offset`` 采样，
  offset ∈ {5,10,15,20,30} m（配置可改）；``s`` 超出车道末端或投影失败的点**丢弃**（mask=0），
  因此车道数不足 / 车道很短时通道自动降级——这就是"对少车道鲁棒"的机制。
- 16 个槽位的分配（固定预算下兼顾"自车前视"与"多车道近场"）：

  1. **自车当前车道**优先占满 5 个 offset（保证 5..30 m 前视，弯道/限速提前量）；
  2. 其余候选车道按"环优先"填充：先 5 m 环（所有车道），再 10 m、15 m、…，
     环内按车道优先级。车道多时远端环被裁掉，但每条车道都有近场几何。

特征（7 维，自车系 x 前向 / y 左向）::

    [dx, dy, heading_rel, curvature, speed_limit, left_line_type_id, right_line_type_id]

- ``heading_rel`` = 该采样点车道航向 - ego 航向（wrap 到 (-π,π]）；
- ``curvature`` = dθ/ds（数值中心差分；直道=0，圆弧=±1/R，右转（顺时针）为负）；
- ``speed_limit`` = ``lane.speed_limit`` **原始值**。注意上游单位混乱：PG block 传的是 m/s
  （``component/pgblock/ramp.py:33`` "12 m/s ~= 40 km/h"），而 map features 导出命名为
  ``speed_limit_kmh``（``component/map/pg_map.py:152``）。本项目约定：post-build 写入 m/s
  （见 ``env/metadrive_env.py``），本通道不做换算；
- ``left_line_type_id`` / ``right_line_type_id``：``lane.line_types`` 的左右线型离散 id
  （``component/pgblock/pg_block.py:257-258`` 确认 [0]=左、[1]=右）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from metadrive.type import MetaDriveType

from env.obs.base import FrameAlignment, ObservationChannel, make_empty, safe_ego, wrap_to_pi

#: 线型字符串 -> 稳定 id（0 保留给未知/无）。来源：metadrive/type.py:24-43
LINE_TYPE_IDS: dict[str, int] = {
    MetaDriveType.LINE_UNKNOWN: 0,
    MetaDriveType.LINE_BROKEN_SINGLE_WHITE: 1,
    MetaDriveType.LINE_SOLID_SINGLE_WHITE: 2,
    MetaDriveType.LINE_SOLID_DOUBLE_WHITE: 3,
    MetaDriveType.LINE_BROKEN_SINGLE_YELLOW: 4,
    MetaDriveType.LINE_BROKEN_DOUBLE_YELLOW: 5,
    MetaDriveType.LINE_SOLID_SINGLE_YELLOW: 6,
    MetaDriveType.LINE_SOLID_DOUBLE_YELLOW: 7,
    MetaDriveType.LINE_PASSING_DOUBLE_YELLOW: 8,
    MetaDriveType.BOUNDARY_LINE: 9,
    MetaDriveType.BOUNDARY_MEDIAN: 10,
    MetaDriveType.BOUNDARY_SIDEWALK: 11,
    MetaDriveType.GUARDRAIL: 12,
}
LINE_TYPE_ID_UNKNOWN = 0

_EPS = 1e-6


def line_type_id(value) -> float:
    """线型（字符串或已是数字 id）-> float id；未知一律 0。"""
    if value is None:
        return float(LINE_TYPE_ID_UNKNOWN)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(int(value))
    return float(LINE_TYPE_IDS.get(str(value), LINE_TYPE_ID_UNKNOWN))


def lane_curvature(lane, s: float) -> float:
    """dθ/ds 的数值中心差分（对任意 AbstractLane 实现都成立）。"""
    try:
        length = float(lane.length)
        lo = max(0.0, float(s) - 1.0)
        hi = min(length, float(s) + 1.0)
        if hi - lo < 1e-3:
            return 0.0
        dh = float(wrap_to_pi(lane.heading_theta_at(hi) - lane.heading_theta_at(lo)))
        return dh / (hi - lo)
    except Exception:  # noqa: BLE001 - 几何异常不应让观测通道崩掉
        return 0.0


def _lane_id(lane, default: int = 1 << 20) -> int:
    try:
        return int(lane.index[2])
    except Exception:  # noqa: BLE001
        return default


def _lane_s0(ego, lane) -> float:
    """ego 在候选车道上的纵向投影（clamp 到 [0, length]）。

    next_ref_lanes 位于当前路段之后，ego 投影通常为负 → 从 0 开始采样（即"进入下一路段后的 5..30 m"）；
    圆弧车道投影可能抛 ValueError（abs_lane.py:94）→ 退化为 0，保持通道可用。
    """
    try:
        long, _ = lane.local_coordinates(ego.position)
        s0 = float(long)
    except Exception:  # noqa: BLE001
        s0 = 0.0
    if not np.isfinite(s0):
        s0 = 0.0
    return min(max(s0, 0.0), float(lane.length))


def _siblings(env, lane) -> list:
    """同一路段（road）的所有车道；lane_id 0 为最左，越大越靠右（pg_block.py:253）。"""
    try:
        graph = env.current_map.road_network.graph
        idx = lane.index
        return list(graph[idx[0]][idx[1]])
    except Exception:  # noqa: BLE001
        return []


def candidate_lanes(env, ego, nav) -> list[tuple[int, object]]:
    """返回 ``[(优先级, lane)]``，已按优先级去重。"""
    current_ref = list(getattr(nav, "current_ref_lanes", None) or [])
    next_ref = list(getattr(nav, "next_ref_lanes", None) or [])
    ego_lane = getattr(ego, "lane", None)
    if ego_lane is None and current_ref:
        ego_lane = current_ref[0]
    if ego_lane is None:
        return []

    ego_id = _lane_id(ego_lane, default=-1)
    out: list[tuple[int, object]] = []
    seen: set[int] = set()

    def _add(priority: int, lane) -> None:
        if lane is None or id(lane) in seen:
            return
        seen.add(id(lane))
        out.append((priority, lane))

    _add(0, ego_lane)
    # 下一路段的续接车道：同 lane id 优先，否则用 next_ref_lanes[0]
    continuation = None
    for lane in next_ref:
        if ego_id >= 0 and _lane_id(lane, default=-2) == ego_id:
            continuation = lane
            break
    if continuation is None and next_ref:
        continuation = next_ref[0]
    _add(1, continuation)
    for lane in sorted(next_ref, key=lambda l: (abs(_lane_id(l) - ego_id), _lane_id(l))):
        _add(2, lane)
    for lane in sorted(
        _siblings(env, ego_lane), key=lambda l: (abs(_lane_id(l) - ego_id), 0 if _lane_id(l) < ego_id else 1)
    ):
        _add(3, lane)
    return out


class LDChannel(ObservationChannel):
    """top-16 车道点通道。"""

    name = "ld"
    feature_dim = 7
    # (dx,dy) 是点；heading_rel 是角度标量；curvature/speed_limit/线型不随坐标系变化
    alignment = FrameAlignment(point_pairs=((0, 1), ), angle_dims=(2, ))

    def __init__(self, *, num_slots: int = 16, offsets: Sequence[float] = (5.0, 10.0, 15.0, 20.0, 30.0)):
        self.num_slots = int(num_slots)
        self.offsets = tuple(float(o) for o in offsets)

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(self.num_slots, self.feature_dim)
        ego = safe_ego(env)
        if ego is None:
            return feats, mask
        nav = getattr(ego, "navigation", None)

        ego_theta = float(ego.heading_theta)
        # 当前车道（lane_priority==0）占满全部 offset；其余车道进入 secondary 按环优先排序
        # (offset_idx, lane_priority, lane_order, row)：lane_order 保证排序键唯一，避免 ndarray 参与比较
        primary: list[tuple[int, int, int, np.ndarray]] = []
        secondary: list[tuple[int, int, int, np.ndarray]] = []
        for lane_order, (lane_priority, lane) in enumerate(candidate_lanes(env, ego, nav)):
            s0 = _lane_s0(ego, lane)
            length = float(getattr(lane, "length", 0.0))
            if length <= 0.0:
                continue
            try:
                line_types = list(getattr(lane, "line_types", None) or [])
            except Exception:  # noqa: BLE001
                line_types = []
            left_type = line_type_id(line_types[0] if len(line_types) > 0 else None)
            right_type = line_type_id(line_types[1] if len(line_types) > 1 else None)
            speed_limit = float(getattr(lane, "speed_limit", 0.0))
            for offset_idx, offset in enumerate(self.offsets):
                s = s0 + offset
                if s > length + _EPS:
                    continue  # 超出车道末端：丢弃，交给 mask
                s = min(max(s, 0.0), length)
                try:
                    point = np.asarray(lane.position(s, 0.0), dtype=np.float32)[:2]
                    heading = float(lane.heading_theta_at(s))
                except Exception:  # noqa: BLE001
                    continue
                rel = np.asarray(ego.convert_to_local_coordinates(point, ego.position), dtype=np.float32)
                row = np.array(
                    [
                        rel[0],
                        rel[1],
                        float(wrap_to_pi(heading - ego_theta)),
                        lane_curvature(lane, s),
                        speed_limit,
                        left_type,
                        right_type,
                    ],
                    dtype=np.float32,
                )
                (primary if lane_priority == 0 else secondary).append((offset_idx, lane_priority, lane_order, row))

        primary.sort(key=lambda item: item[0])  # 当前车道：5 → 30 m
        secondary.sort(key=lambda item: (item[0], item[1], item[2]))  # 其余：环优先，环内按车道优先级
        for slot, (_, _, _, row) in enumerate((primary + secondary)[: self.num_slots]):
            feats[slot] = row
            mask[slot] = 1.0
        return feats, mask
