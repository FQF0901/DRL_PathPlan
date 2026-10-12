"""lane_semantic 通道（R21 语义重建，schema v7）+ target lane 纯函数 + LD 点补充。

契约全文见 ``docs/r21_lane_semantic_contract.md`` / ``docs/r21_target_lane_selection.md``。
本模块是唯一实现出处：

1. :func:`select_target_lane` —— **唯一可测试纯函数**（无 env / 无全局状态；输入 ego 车道、
   位姿、``nav.current_ref_lanes`` / ``nav.next_ref_lanes``，输出 :class:`TargetLaneSelection`）。
   优先级：**route current_ref_lane → route continuation → next_ref_lane → ego.lane 仅兜底**。
   绝不沿用"ego.lane 最高优先"；选择与 ego **横向**位置无关（横向偏移不引起 target 切换）。
2. :class:`LaneSemanticChannel` —— 12 维 lane-relative 语义（相对 target lane；自车系 1 槽）。
   **禁止 raw lane_id 数值输入**：通道内不含任何 lane id；lane 身份一律用
   role / same-road / left-right / route 关联表达。
3. :func:`classify_slot_roles` / :func:`slot_group_ids` —— LD 点补充（逐槽 role / 局部 group），
   由 :class:`env.obs.ld.LDChannel` 作为伴随数组输出（不进 net 输入）。
4. :func:`lane_curvature` —— dθ/ds 数值中心差分（``env.obs.ld`` 复用同一实现，单一出处）。

符号约定（与仓库既有口径一致，勿混）：

- ``lateral`` = MetaDrive 车道系横向坐标（``lane.local_coordinates`` 第 2 分量），
  **正 = 车道方向右侧**（``straight_lane.py:46`` ``direction_lateral=(dy,-dx)``；
  与 ``env/metadrive_env.lane_lateral_info`` / reward ``lane_center`` 同号）。
- ``heading_error`` = ``wrap(ego.heading_theta - lane.heading_theta_at(s))``，
  **正 = 自车航向相对车道切线逆时针偏左**。
- ``curvature`` = dθ/ds，**右转（顺时针）为负**（同 ``env.obs.ld`` 文档）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from env.obs.base import ObservationChannel, make_empty, safe_ego, wrap_to_pi

__all__ = [
    "LaneSemanticChannel",
    "TargetLaneSelection",
    "select_target_lane",
    "classify_slot_roles",
    "slot_group_ids",
    "slot_route_associated",
    "slot_offset_s_normalized",
    "lane_curvature",
    "lane_id_of",
    "lane_index_of",
    "LANE_SEMANTIC_FEATURE_NAMES",
    "LANE_SEMANTIC_DIM",
    "ROLE_OTHER",
    "ROLE_TARGET",
    "ROLE_TARGET_CONTINUATION",
    "ROLE_LEFT_ADJACENT",
    "ROLE_RIGHT_ADJACENT",
    "ROLE_INVALID",
    "LD_SUPPLEMENT_KEYS",
    "SOURCE_ROUTE_CURRENT",
    "SOURCE_ROUTE_CONTINUATION",
    "SOURCE_ROUTE_NEXT",
    "SOURCE_EGO_FALLBACK",
    "SOURCE_NONE",
    "LARGE_DISTANCE_CAP_M",
    "CURVATURE_NEAR_OFFSET_M",
    "CURVATURE_FAR_OFFSET_M",
    "LANE_CONNECT_TOL_M",
]

# --------------------------------------------------------------------------- #
# 常量 / 编码
# --------------------------------------------------------------------------- #

#: LD 点 role 编码（``ld_lane_role`` 伴随数组；-1 = 无效槽）
ROLE_INVALID = -1
ROLE_OTHER = 0
ROLE_TARGET = 1
ROLE_TARGET_CONTINUATION = 2
ROLE_LEFT_ADJACENT = 3
ROLE_RIGHT_ADJACENT = 4

#: target lane 来源（``TargetLaneSelection.source``；对应优先级链）
SOURCE_ROUTE_CURRENT = "route_current"
SOURCE_ROUTE_CONTINUATION = "route_continuation"
SOURCE_ROUTE_NEXT = "route_next"
SOURCE_EGO_FALLBACK = "ego_fallback"
SOURCE_NONE = "none"

#: route 关联置信度（按来源固定；纯函数输出的一部分，供下游直接消费）
_CONFIDENCE = {
    SOURCE_ROUTE_CURRENT: 1.0,
    SOURCE_ROUTE_CONTINUATION: 0.8,
    SOURCE_ROUTE_NEXT: 0.6,
    SOURCE_EGO_FALLBACK: 0.25,
    SOURCE_NONE: 0.0,
}

#: ``distance_to_next_topology_change`` 的"无拓扑变化"上限（m）：路线尽头/无下一段时填该值
LARGE_DISTANCE_CAP_M = 200.0
#: target lane 曲率采样前视偏移（m）：near/far 两档（超出车道长度时 clamp 到末端）
CURVATURE_NEAR_OFFSET_M = 10.0
CURVATURE_FAR_OFFSET_M = 30.0
#: route 续接/汇入端点容差（m）：端点距离差在该值内视为"共线续接"（保持锚定，避免横跳）
LANE_CONNECT_TOL_M = 1.5

_EPS = 1e-6

#: 12 维特征名（顺序即 ``LaneSemanticChannel`` 输出顺序）
LANE_SEMANTIC_FEATURE_NAMES: tuple[str, ...] = (
    "ego_to_target_lane_lateral_offset",
    "ego_heading_error_to_target_tangent",
    "target_lane_curvature_near",
    "target_lane_curvature_far",
    "left_drivable_margin",
    "right_drivable_margin",
    "target_lane_valid",
    "route_association_confidence",
    "lane_change_required_left",
    "lane_change_required_right",
    "distance_to_lane_end",
    "distance_to_next_topology_change",
)
LANE_SEMANTIC_DIM = len(LANE_SEMANTIC_FEATURE_NAMES)

#: LD 点补充伴随数组键（``LDChannel.companions()`` 输出；映射见契约文档）
LD_SUPPLEMENT_KEYS: tuple[str, ...] = (
    "ld_lane_role",
    "ld_offset_s_norm",
    "ld_group",
    "ld_route_associated",
)


# --------------------------------------------------------------------------- #
# 基础几何工具（纯函数，鸭子类型）
# --------------------------------------------------------------------------- #
def lane_index_of(lane: Any) -> tuple | None:
    """lane 的 ``index`` 元组（``(from_node, to_node, lane_id)``）；不可用返回 None。"""
    try:
        index = lane.index
    except Exception:  # noqa: BLE001
        return None
    if index is None:
        return None
    try:
        return tuple(index)
    except TypeError:
        return None


def lane_id_of(lane: Any) -> int:
    """lane 的 lane_id（``index[2]``）；不可用返回 -1。lane_id 0 = 最左，越大越右。"""
    index = lane_index_of(lane)
    if index is None or len(index) < 3:
        return -1
    try:
        return int(index[2])
    except (TypeError, ValueError):
        return -1


def _project(lane: Any, position: Any) -> tuple[float, float] | None:
    """ego 位置在车道上的 ``(long, lat)``；失败/非有限返回 None（圆弧投影会抛 ValueError）。"""
    try:
        long, lat = lane.local_coordinates(np.asarray(position, dtype=np.float64)[:2])
        long, lat = float(long), float(lat)
    except Exception:  # noqa: BLE001
        return None
    if not (np.isfinite(long) and np.isfinite(lat)):
        return None
    return long, lat


def _lane_length(lane: Any) -> float:
    try:
        return float(lane.length)
    except Exception:  # noqa: BLE001
        return 0.0


def _lane_width(lane: Any) -> float:
    for attr in ("width", ):
        try:
            value = float(getattr(lane, attr))
            if np.isfinite(value) and value > 0.0:
                return value
        except Exception:  # noqa: BLE001
            continue
    return 0.0


def _endpoint_gap(a: Any, b: Any) -> float:
    """``||a.end - b.start||``（续接端点距离）；几何不可用返回 inf。"""
    try:
        return float(np.linalg.norm(np.asarray(a.end, dtype=np.float64)[:2] - np.asarray(b.start, dtype=np.float64)[:2]))
    except Exception:  # noqa: BLE001
        return float("inf")


def _center_distance(lane: Any, position: Any) -> float:
    """ego 位置到车道中心线的近似距离（投影 |lat|；失败时用中点距离）。"""
    projection = _project(lane, position)
    if projection is not None:
        return abs(projection[1])
    try:
        middle = np.asarray(lane.position(_lane_length(lane) / 2.0, 0.0), dtype=np.float64)[:2]
        return float(np.linalg.norm(middle - np.asarray(position, dtype=np.float64)[:2]))
    except Exception:  # noqa: BLE001
        return float("inf")


def _id_delta(lane: Any, ego_lane: Any) -> int:
    """|lane_id(lane) - lane_id(ego_lane)|；任一缺失返回一个大数（稳定排序用）。"""
    if ego_lane is None:
        return 1 << 20
    lid, eid = lane_id_of(lane), lane_id_of(ego_lane)
    if lid < 0 or eid < 0:
        return 1 << 20
    return abs(lid - eid)


def lane_curvature(lane: Any, s: float) -> float:
    """dθ/ds 的数值中心差分（对任意 AbstractLane 实现都成立；右转（顺时针）为负）。"""
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


# --------------------------------------------------------------------------- #
# target lane 选择（唯一纯函数）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetLaneSelection:
    """target lane 选择结果。

    Attributes:
        lane: 选中的车道对象；无 route 且无 ego 车道时为 None。
        source: 来源（:data:`SOURCE_*`），对应优先级链。
        route_associated: 是否由 route（current_ref/next_ref）产生（False = ego 兜底）。
        confidence: 固定置信度（见 :data:`_CONFIDENCE`）。
    """

    lane: Any | None
    source: str
    route_associated: bool
    confidence: float


def _best_next(nxt: Sequence[Any], ego_lane: Any | None, ego_position: Any | None) -> Any:
    """next_ref_lanes 中的最优目标：同 lane_id 优先 → 距 ego 最近 → lane_id 小者稳定。"""
    if ego_lane is not None:
        ego_id = lane_id_of(ego_lane)
        if ego_id >= 0:
            for lane in nxt:
                if lane_id_of(lane) == ego_id:
                    return lane
    if ego_position is not None:
        return min(nxt, key=lambda lane: (_center_distance(lane, ego_position), lane_id_of(lane)))
    return nxt[0]


def _best_ref(
    refs: Sequence[Any],
    nxt: Sequence[Any],
    ego_lane: Any | None,
    ego_position: Any | None,
) -> Any:
    """current_ref_lanes 中的最优目标。

    有 next 链时按"续接端点距离"选（route continuation 语义），**不依赖 ego 横向位置**
    （端点距离差 ≤ :data:`LANE_CONNECT_TOL_M` 视为平局 → 退回 ego 车道 id / lane_id 稳定序）；
    无 next 链时按 ego 车道 id 距离；ego 车道也未知时才用横向距离。
    """
    if nxt:
        def key(lane: Any) -> tuple:
            gap = min(_endpoint_gap(lane, other) for other in nxt)
            return (gap, _id_delta(lane, ego_lane), lane_id_of(lane))

        best = min(refs, key=key)
        if ego_lane is not None and any(ego_lane is r for r in refs):
            # 平局保持：ego 所在路由车道与最优续接差距在容差内 → 保持 ego 车道（避免横跳）
            best_gap = key(best)[0]
            ego_gap = min(_endpoint_gap(ego_lane, other) for other in nxt)
            if ego_gap <= best_gap + LANE_CONNECT_TOL_M:
                return ego_lane
        return best
    if ego_lane is not None:
        return min(refs, key=lambda lane: (_id_delta(lane, ego_lane), lane_id_of(lane)))
    if ego_position is not None:
        return min(refs, key=lambda lane: (_center_distance(lane, ego_position), lane_id_of(lane)))
    return refs[0]


def _continuation_of(
    target: Any,
    nxt: Sequence[Any],
    ego_lane: Any | None,
    ego_position: Any | None,
) -> Any | None:
    """target 在 next 链上的续接车道：同 lane_id 优先 → ego 车道在 next 链 → 端点最近。"""
    if not nxt:
        return None
    target_id = lane_id_of(target)
    if target_id >= 0:
        for lane in nxt:
            if lane_id_of(lane) == target_id:
                return lane
    if ego_lane is not None and any(ego_lane is n for n in nxt):
        return ego_lane
    if ego_position is not None:
        return min(nxt, key=lambda lane: (_center_distance(lane, ego_position), lane_id_of(lane)))
    return min(nxt, key=lambda lane: (_endpoint_gap(target, lane), lane_id_of(lane)))


def select_target_lane(
    *,
    ego_lane: Any | None,
    ego_position: Any | None = None,
    current_ref_lanes: Iterable[Any] | None = None,
    next_ref_lanes: Iterable[Any] | None = None,
) -> TargetLaneSelection:
    """唯一 target lane 选择纯函数（契约见 ``docs/r21_target_lane_selection.md``）。

    优先级（不得调换）：

    1. **route current_ref_lane**：从 ``current_ref_lanes`` 中选续接最优者（端点距离 →
       ego 车道 id → lane_id 稳定序）；续接端点差 ≤ :data:`LANE_CONNECT_TOL_M` 视为平局 →
       保持 ego 所在路由车道（防横跳；同车道数直行时即"跟随 ego 车道"）。
    2. **route continuation**：目标车道已被越过末端（投影 long > length，nav 尚未更新）
       且存在 next 链 → 切到 target 的续接车道；或 current_ref 为空而 ego 车道 ∈ next_ref
       → 选 ego 车道（它就是路线续接）。
    3. **next_ref_lane**：无 current_ref 信息时从 next_ref 中选（同 lane_id → 最近 → 稳定序）。
    4. **ego.lane 仅兜底**：无任何 route 信息时才用 ego.lane（``route_associated=False``）。

    稳定性保证：存在 route 信息时，选择**不依赖 ego 横向位置**（横向偏移 ≤ 车道内 1 m 不改变
    选择）；只有真实越界（越过末端）或 lane 变化才可能切换。
    """
    refs = [lane for lane in (current_ref_lanes or ()) if lane is not None]
    nxt = [lane for lane in (next_ref_lanes or ()) if lane is not None]

    if refs:
        target = _best_ref(refs, nxt, ego_lane, ego_position)
        # route continuation：已越过目标车道末端（nav 尚未更新）
        if nxt:
            projection = _project(target, ego_position) if ego_position is not None else None
            if projection is not None and projection[0] > _lane_length(target) + _EPS:
                continuation = _continuation_of(target, nxt, ego_lane, ego_position)
                if continuation is not None and continuation is not target:
                    return TargetLaneSelection(
                        continuation, SOURCE_ROUTE_CONTINUATION, True, _CONFIDENCE[SOURCE_ROUTE_CONTINUATION]
                    )
        return TargetLaneSelection(target, SOURCE_ROUTE_CURRENT, True, _CONFIDENCE[SOURCE_ROUTE_CURRENT])

    if nxt:
        if ego_lane is not None and any(ego_lane is lane for lane in nxt):
            return TargetLaneSelection(
                ego_lane, SOURCE_ROUTE_CONTINUATION, True, _CONFIDENCE[SOURCE_ROUTE_CONTINUATION]
            )
        return TargetLaneSelection(
            _best_next(nxt, ego_lane, ego_position), SOURCE_ROUTE_NEXT, True, _CONFIDENCE[SOURCE_ROUTE_NEXT]
        )

    if ego_lane is not None:
        return TargetLaneSelection(ego_lane, SOURCE_EGO_FALLBACK, False, _CONFIDENCE[SOURCE_EGO_FALLBACK])
    return TargetLaneSelection(None, SOURCE_NONE, False, 0.0)


# --------------------------------------------------------------------------- #
# LD 点补充（逐槽 role / 局部 group / route 关联 / offset_s 归一化）
# --------------------------------------------------------------------------- #
def _continuation_lane(target: Any | None, next_ref_lanes: Iterable[Any] | None) -> Any | None:
    if target is None:
        return None
    nxt = [lane for lane in (next_ref_lanes or ()) if lane is not None]
    if not nxt:
        return None
    target_id = lane_id_of(target)
    if target_id >= 0:
        for lane in nxt:
            if lane_id_of(lane) == target_id:
                return lane
    return min(nxt, key=lambda lane: (_endpoint_gap(target, lane), lane_id_of(lane)))


def classify_slot_roles(
    target_lane: Any | None,
    next_ref_lanes: Iterable[Any] | None,
    slot_lanes: Sequence[Any],
) -> np.ndarray:
    """逐槽 role（:data:`ROLE_*`）。

    - target：与 target 同一车道对象；
    - target_continuation：next 链上的 target 续接车道（同 lane_id 优先，否则端点最近）；
    - left/right_adjacent：与 target **同 road**（``index[:2]`` 相等）且 lane_id 更小/更大
      （lane_id 0 = 最左）；
    - other：其余（含 target 缺失时全部槽）。
    """
    roles = np.full((len(slot_lanes), ), ROLE_OTHER, dtype=np.int64)
    if target_lane is None:
        return roles
    continuation = _continuation_lane(target_lane, next_ref_lanes)
    target_index = lane_index_of(target_lane)
    target_id = lane_id_of(target_lane)
    for k, lane in enumerate(slot_lanes):
        if lane is None:
            continue
        if lane is target_lane:
            roles[k] = ROLE_TARGET
            continue
        if continuation is not None and lane is continuation:
            roles[k] = ROLE_TARGET_CONTINUATION
            continue
        index = lane_index_of(lane)
        if (
            index is not None
            and target_index is not None
            and len(index) >= 2
            and len(target_index) >= 2
            and index[:2] == target_index[:2]
        ):
            lid = lane_id_of(lane)
            if target_id >= 0 and lid >= 0:
                roles[k] = ROLE_LEFT_ADJACENT if lid < target_id else ROLE_RIGHT_ADJACENT
    return roles


def slot_group_ids(slot_lanes: Sequence[Any]) -> np.ndarray:
    """同一 lane 对象 → 同一局部 group id（按首次出现顺序 0,1,2,...；None → -1）。

    "仅观测内局部 group"：不暴露全局 lane_id，只保证同车道同组、异车道异组。
    """
    groups = np.full((len(slot_lanes), ), -1, dtype=np.int64)
    mapping: dict[int, int] = {}
    next_id = 0
    for k, lane in enumerate(slot_lanes):
        if lane is None:
            continue
        key = id(lane)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
        groups[k] = mapping[key]
    return groups


def slot_route_associated(slot_lanes: Sequence[Any], current_ref_lanes: Iterable[Any] | None, next_ref_lanes: Iterable[Any] | None) -> np.ndarray:
    """逐槽 route 关联（1 = 该槽车道 ∈ current_ref ∪ next_ref；按对象身份判定）。"""
    route = [lane for lane in list(current_ref_lanes or ()) + list(next_ref_lanes or ()) if lane is not None]
    out = np.zeros((len(slot_lanes), ), dtype=np.float32)
    for k, lane in enumerate(slot_lanes):
        if lane is not None and any(lane is ref for ref in route):
            out[k] = 1.0
    return out


def slot_offset_s_normalized(slot_s: Sequence[float], slot_lanes: Sequence[Any]) -> np.ndarray:
    """逐槽 ``s / lane.length``（clamp [0,1]）；同车道内随 offset 严格递增。"""
    out = np.zeros((len(slot_s), ), dtype=np.float32)
    for k, (s, lane) in enumerate(zip(slot_s, slot_lanes)):
        length = _lane_length(lane) if lane is not None else 0.0
        if length <= 0.0:
            continue
        out[k] = float(min(max(float(s) / length, 0.0), 1.0))
    return out


# --------------------------------------------------------------------------- #
# lane_semantic 通道（1 槽 × 12 维）
# --------------------------------------------------------------------------- #
def _siblings(env: Any, lane: Any) -> list[Any]:
    """同 road 的所有车道（``road_network.graph[from][to]``）；不可用返回 []。"""
    try:
        graph = env.current_map.road_network.graph
        index = lane.index
        return list(graph[index[0]][index[1]])
    except Exception:  # noqa: BLE001
        return []


def _direction_aligned(a: Any, b: Any) -> bool:
    """两车道方向是否同向（点积 > 0）；方向不可用时保守视为同向（False 会让 margin 偏小，
    这里选择 True：与"未知道路按可行驶"的宽松语义一致，文档已声明）。"""
    try:
        da = np.asarray(a.direction, dtype=np.float64)[:2]
        db = np.asarray(b.direction, dtype=np.float64)[:2]
        if da.shape != (2, ) or db.shape != (2, ):
            return True
        return float(np.dot(da, db)) > 0.0
    except Exception:  # noqa: BLE001
        return True


def _drivable_margins(env: Any, lane: Any, lat: float) -> tuple[float, float]:
    """(left_margin, right_margin)：ego 到左右最近"不可行驶边界"的距离（m）。

    从 target lane 起向两侧扩展**同 road 且同向**的相邻车道（lane_id 递减 = 左、
    递增 = 右，0 = 最左）；遇到对向/不可用/缺失的相邻车道即停，边界 = 该处。
    几何不可用时退化为 target lane 自身宽度边界。两值均 ≥ 0。
    """
    width = _lane_width(lane)
    left = width / 2.0 + float(lat)
    right = width / 2.0 - float(lat)
    siblings = _siblings(env, lane)
    if not siblings or width <= 0.0:
        return max(left, 0.0), max(right, 0.0)
    target_id = lane_id_of(lane)
    if target_id < 0:
        return max(left, 0.0), max(right, 0.0)
    by_id: dict[int, Any] = {}
    for other in siblings:
        if other is lane or other is None:
            continue
        oid = lane_id_of(other)
        if oid >= 0:
            by_id.setdefault(oid, other)
    # 左：lane_id 递减（0 最左），同向可行驶才继续
    lid = target_id - 1
    while lid in by_id and _direction_aligned(lane, by_id[lid]):
        left += _lane_width(by_id[lid])
        lid -= 1
    rid = target_id + 1
    while rid in by_id and _direction_aligned(lane, by_id[rid]):
        right += _lane_width(by_id[rid])
        rid += 1
    return max(left, 0.0), max(right, 0.0)


class LaneSemanticChannel(ObservationChannel):
    """lane_semantic 通道：1 槽 × 12 维（相对 target lane 的语义特征；见契约文档）。

    - ``mask=1`` ⇔ 选出了 target lane（source ≠ none）；
    - ``target_lane_valid=1`` ⇔ mask=1 且 ego 投影成功、特征有限（否则该帧全 0，供
      有效率审计）；无 route 时走 ego.lane 兜底（``route_association_confidence=0.25``）；
    - 通道**不含 raw lane_id**；lane 身份由 role/group 伴随数组（LD 点补充）表达。
    """

    name = "lane_semantic"
    feature_dim = LANE_SEMANTIC_DIM

    def build(self, env: Any, spec: Any = None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(1, self.feature_dim)
        ego = safe_ego(env)
        if ego is None:
            return feats, mask
        nav = getattr(ego, "navigation", None)
        refs = list(getattr(nav, "current_ref_lanes", None) or [])
        nxt = list(getattr(nav, "next_ref_lanes", None) or [])
        ego_lane = getattr(ego, "lane", None)
        selection = select_target_lane(
            ego_lane=ego_lane,
            ego_position=getattr(ego, "position", None),
            current_ref_lanes=refs,
            next_ref_lanes=nxt,
        )
        if selection.lane is None:
            return feats, mask
        mask[0] = 1.0
        lane = selection.lane
        length = _lane_length(lane)
        projection = _project(lane, getattr(ego, "position", None))
        if projection is None or length <= 0.0:
            return feats, mask  # target_lane_valid=0（特征全 0）
        s_raw, lat = projection
        s = float(min(max(s_raw, 0.0), length))
        try:
            ego_theta = float(ego.heading_theta)
            heading_error = float(wrap_to_pi(ego_theta - float(lane.heading_theta_at(s))))
            near_s = min(s + CURVATURE_NEAR_OFFSET_M, length)
            far_s = min(s + CURVATURE_FAR_OFFSET_M, length)
            left_margin, right_margin = _drivable_margins(env, lane, lat)
            change_left, change_right = _lane_change_flags(ego_lane, lane, lat)
            remaining = max(length - s, 0.0)
            topology = remaining if nxt else LARGE_DISTANCE_CAP_M
            feats[0] = np.array(
                [
                    float(lat),
                    heading_error,
                    float(lane_curvature(lane, near_s)),
                    float(lane_curvature(lane, far_s)),
                    float(left_margin),
                    float(right_margin),
                    1.0,
                    float(selection.confidence),
                    float(change_left),
                    float(change_right),
                    float(remaining),
                    float(topology),
                ],
                dtype=np.float32,
            )
        except Exception:  # noqa: BLE001 - 单帧几何异常 → 该帧 invalid（不崩通道）
            feats[0] = 0.0
            return feats, mask
        if not np.all(np.isfinite(feats[0])):
            feats[0] = 0.0
        return feats, mask


def _lane_change_flags(ego_lane: Any | None, target: Any, lat: float) -> tuple[int, int]:
    """(left, right)：ego 是否需要向左/右变道到 target（ego 已在 target 上 → 全 0）。

    判定：ego 车道 ≠ target 时，若同 road 用 lane_id 大小（0 最左）；否则用 ego 在 target
    车道系中的横向位置（``lat`` 正 = ego 在 target 中心线右侧 → 需左移）。
    两者都不可用时不置位（保守）。
    """
    if ego_lane is None or ego_lane is target:
        return 0, 0
    target_index = lane_index_of(target)
    ego_index = lane_index_of(ego_lane)
    if target_index is not None and ego_index is not None and len(target_index) >= 2 and len(ego_index) >= 2:
        if target_index[:2] == ego_index[:2]:
            delta = lane_id_of(target) - lane_id_of(ego_lane)
            if delta < 0:
                return 1, 0
            if delta > 0:
                return 0, 1
            return 0, 0
    width = _lane_width(target)
    if width > 0.0 and abs(lat) > width / 2.0:
        return (1, 0) if lat > 0.0 else (0, 1)
    return 0, 0
