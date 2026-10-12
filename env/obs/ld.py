"""LD 通道：当前 / 相邻 / 下一参考车道的中心线采样 top-16 点。

v8 诊断臂 A1'（2026-10-07）：采样 offset = :data:`LD_OFFSETS_M` = {5,10,15,20,30} m
（回退 v4 近场口径；{0,20,40,60,80} 方案见 git 历史——闭环回归诊断中）
（5 档）——**0 m 点 = ego 投影点**（`s = clamp(ego 投影 long, 0, length)`），提供近场
横向锚定（d_lat 隐含在 ``dy`` 中），补偿 lane 通道删除后的近场信息缺口；远场保留
20/40/60/80 m 前视。v5 的独立 ``lane`` 通道已删除。

采样与选取规则
--------------
- 候选车道（去重，优先级从高到低）：

  1. ego 当前车道（``env.agent.lane``；缺失时退化为 ``current_ref_lanes[0]``）；
  2. 下一参考车道中"同 lane id"的续接车道，否则 ``next_ref_lanes[0]``；
  3. 其余 ``next_ref_lanes``（按 |lane_id - ego_lane_id| 排序）；
  4. 同路相邻车道（``road_network.graph[from][to]`` 去掉当前车道，按 |Δlane_id| 排序，左侧优先）。

- 沿每条候选车道中心线按 ``s = clamp(ego 投影 long, 0, length) + offset`` 采样，
  offset ∈ :data:`LD_OFFSETS_M` = {5,10,15,20,30} m（配置可改）；``s`` 超出车道末端或投影
  失败的点**丢弃**（mask=0），因此车道数不足 / 车道很短时通道自动降级——这就是"对少车道
  鲁棒"的机制。
- 16 个槽位的分配（固定预算下"自车近场锚 + 前视 + 多车道远场"）：

  1. **自车当前车道**优先占满 5 个 offset（0..80 m：近场锚 + 弯道/限速提前量）；
  2. 其余候选车道按"环优先"填充：先 0 m 环（所有车道近场），再 20 m、40 m、60 m、80 m，
     环内按车道优先级。车道多时远端环被裁掉，但每条车道都有 0/20 m 几何。

  v6 的取舍：当前车道 0 m 点即 ego 投影点（`dy` = 横向偏差的负值），横向控制关心的
  近场信息恢复；其余车道的近场由 0 m 环覆盖（只保底第一条环，被裁时相邻车道近场不可见，
  相邻车道近场对象仍由 OD 覆盖）。

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

R21（2026-10-12，语义重建，schema v7）
--------------------------------------
本通道新增**逐槽补充伴随数组**（不进 feats/mask，不做 SE(2) 对齐；见
``docs/r21_lane_semantic_contract.md``）：``ld_lane_role``（target / target_continuation /
left_adjacent / right_adjacent / other 编码）、``ld_offset_s_norm``（s/length，同车道内严格
递增）、``ld_group``（观测内局部同车道分组，不暴露全局 lane_id）、``ld_route_associated``
（∈ current_ref ∪ next_ref）。计算由 :mod:`env.obs.lane_semantic` 提供（target lane 选择为
唯一纯函数）。``lane_curvature`` 实现移至 :mod:`env.obs.lane_semantic`（此处 import 复用，
单一出处）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from metadrive.type import MetaDriveType

from env.obs.base import FrameAlignment, ObservationChannel, make_empty, safe_ego, wrap_to_pi
from env.obs.lane_semantic import (
    LD_SUPPLEMENT_KEYS,
    classify_slot_roles,
    lane_curvature,
    lane_id_of,
    select_target_lane,
    slot_group_ids,
    slot_offset_s_normalized,
    slot_route_associated,
)

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

#: v6/v8 默认采样 offset（m）：{5,10,15,20,30}（0 m 近场锚已回退，见模块 docstring 的 v8 诊断臂 A1';
#: 顺序即"当前车道 primary 槽位顺序"，也是其余车道的环填充顺序。
LD_OFFSETS_M: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 30.0)

_EPS = 1e-6


def line_type_id(value) -> float:
    """线型（字符串或已是数字 id）-> float id；未知一律 0。"""
    if value is None:
        return float(LINE_TYPE_ID_UNKNOWN)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(int(value))
    return float(LINE_TYPE_IDS.get(str(value), LINE_TYPE_ID_UNKNOWN))


def _lane_id(lane, default: int = 1 << 20) -> int:
    try:
        return int(lane.index[2])
    except Exception:  # noqa: BLE001
        return default


def _lane_s0(ego, lane) -> float:
    """ego 在候选车道上的纵向投影（clamp 到 [0, length]）。

    next_ref_lanes 位于当前路段之后，ego 投影通常为负 → 从 0 开始采样（即"从下一路段起点开始"）；
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
    #: R21 LD 点补充（逐槽伴随数组；不进 feats/mask、不做 SE(2) 对齐）：
    #: ``ld_lane_role`` / ``ld_offset_s_norm`` / ``ld_group`` / ``ld_route_associated``
    companion_names: tuple[str, ...] = LD_SUPPLEMENT_KEYS
    companion_fill: dict[str, object] = {
        "ld_lane_role": -1,
        "ld_offset_s_norm": 0.0,
        "ld_group": -1,
        "ld_route_associated": 0.0,
    }

    def __init__(
        self,
        *,
        num_slots: int = 16,
        offsets: Sequence[float] = LD_OFFSETS_M,
        front_m: float = 150.0,
        rear_m: float = 50.0,
        left_m: float = 25.0,
        right_m: float = 25.0,
        debug: bool = False,
    ):
        self.num_slots = int(num_slots)
        self.offsets = tuple(float(o) for o in offsets)
        self.front_m = float(front_m)
        self.rear_m = float(rear_m)
        self.left_m = float(left_m)
        self.right_m = float(right_m)
        #: Round-18A 旁路诊断（默认关）：``True`` 时把逐槽元数据写入 :attr:`last_debug`，
        #: **不改变** feats/mask 输出（obs fingerprint 不变）。运行时可由调用方置位。
        self.debug = bool(debug)
        #: ``build`` 后逐槽元数据（debug=True 时；否则 None）：
        #: ``lane_index``（3 元组）/``lane_priority``/``lane_order``/``offset_idx``/
        #: ``is_ego_lane``/``is_nav_current``/``is_nav_next``/``s``（采样弧长）
        self.last_debug: dict[str, object] | None = None
        #: ``build`` 后 R21 LD 点补充（伴随数组；见 :meth:`companions`）
        self._last_supplements: dict[str, np.ndarray] | None = None

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(self.num_slots, self.feature_dim)
        self._last_supplements = None
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
                # 盒式 scope（与 OD 同口径）：前 front / 后 rear / 左 left / 右 right
                if not (
                    -self.rear_m <= float(rel[0]) <= self.front_m
                    and -self.right_m <= float(rel[1]) <= self.left_m
                ):
                    continue
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
                (primary if lane_priority == 0 else secondary).append(
                    (offset_idx, lane_priority, lane_order, row, lane, float(s))
                )

        primary.sort(key=lambda item: item[0])  # 当前车道：5 → 30 m（offsets 顺序）
        secondary.sort(key=lambda item: (item[0], item[1], item[2]))  # 其余：环优先，环内按车道优先级
        assembled = (primary + secondary)[: self.num_slots]
        for slot, item in enumerate(assembled):
            feats[slot] = item[3]
            mask[slot] = 1.0
        self._last_supplements = self._build_supplements(ego, nav, assembled)
        if self.debug:
            ego_lane = getattr(ego, "lane", None)
            current_ref = list(getattr(nav, "current_ref_lanes", None) or [])
            next_ref = list(getattr(nav, "next_ref_lanes", None) or [])
            self.last_debug = {
                "num_valid": int(len(assembled)),
                "lane_index": [tuple(item[4].index) if hasattr(item[4], "index") else None for item in assembled],
                "lane_priority": [int(item[1]) for item in assembled],
                "lane_order": [int(item[2]) for item in assembled],
                "offset_idx": [int(item[0]) for item in assembled],
                "offset_m": [float(self.offsets[int(item[0])]) for item in assembled],
                "s_m": [float(item[5]) for item in assembled],
                "is_ego_lane": [bool(item[4] is ego_lane) for item in assembled],
                "is_nav_current": [bool(item[4] in current_ref) for item in assembled],
                "is_nav_next": [bool(item[4] in next_ref) for item in assembled],
                "ego_lane_index": tuple(ego_lane.index) if ego_lane is not None and hasattr(ego_lane, "index") else None,
                "nav_current_indices": [tuple(l.index) for l in current_ref if hasattr(l, "index")],
                "nav_next_indices": [tuple(l.index) for l in next_ref if hasattr(l, "index")],
            }
        else:
            self.last_debug = None
        return feats, mask

    # ------------------------------------------------------- R21 LD 点补充（伴随）
    def _build_supplements(self, ego, nav, assembled: list) -> dict[str, np.ndarray]:
        """逐槽补充数组（shape (num_slots,)；无效槽 role/group=-1、其余 0）。

        见 ``docs/r21_lane_semantic_contract.md``：``ld_lane_role`` / ``ld_offset_s_norm`` /
        ``ld_group`` / ``ld_route_associated``。任何异常 → 全默认（不崩观测）。
        """
        num_slots = self.num_slots
        role = np.full((num_slots, ), -1, dtype=np.int64)
        group = np.full((num_slots, ), -1, dtype=np.int64)
        offset_norm = np.zeros((num_slots, ), dtype=np.float32)
        route_assoc = np.zeros((num_slots, ), dtype=np.float32)
        if not assembled:
            return {
                "ld_lane_role": role,
                "ld_offset_s_norm": offset_norm,
                "ld_group": group,
                "ld_route_associated": route_assoc,
            }
        try:
            lane_objs = [item[4] for item in assembled]
            s_values = [float(item[5]) for item in assembled]
            current_ref = list(getattr(nav, "current_ref_lanes", None) or [])
            next_ref = list(getattr(nav, "next_ref_lanes", None) or [])
            ego_lane = getattr(ego, "lane", None)
            selection = select_target_lane(
                ego_lane=ego_lane,
                ego_position=getattr(ego, "position", None),
                current_ref_lanes=current_ref,
                next_ref_lanes=next_ref,
            )
            n = len(assembled)
            role[:n] = classify_slot_roles(selection.lane, next_ref, lane_objs)
            group[:n] = slot_group_ids(lane_objs)
            offset_norm[:n] = slot_offset_s_normalized(s_values, lane_objs)
            route_assoc[:n] = slot_route_associated(lane_objs, current_ref, next_ref)
        except Exception:  # noqa: BLE001 - 补充数组失败不应影响主观测
            pass
        return {
            "ld_lane_role": role,
            "ld_offset_s_norm": offset_norm,
            "ld_group": group,
            "ld_route_associated": route_assoc,
        }

    def companions(self, env=None, spec=None) -> dict[str, np.ndarray]:
        """返回最近一次 ``build`` 的 LD 点补充（未 build 时返回全默认；返回拷贝）。"""
        if self._last_supplements is None:
            return {
                "ld_lane_role": np.full((self.num_slots, ), -1, dtype=np.int64),
                "ld_offset_s_norm": np.zeros((self.num_slots, ), dtype=np.float32),
                "ld_group": np.full((self.num_slots, ), -1, dtype=np.int64),
                "ld_route_associated": np.zeros((self.num_slots, ), dtype=np.float32),
            }
        return {key: np.array(value, copy=True) for key, value in self._last_supplements.items()}

    def reset(self) -> None:
        self._last_supplements = None
