"""逐步可观测多标签（router 监督用）。

每个标签都只依赖**当前步可观测状态**：自车位置/速度/所在车道、地图结构（block 归属，来自
显式 `BIG_BLOCK_SEQUENCE` 建出的实际地图，不是 spec id）、其他车辆位置、以及 behaviors 记录的
脚本事件窗口。spec 只提供阈值与事件构造参数，**绝不**按 spec 类别直接置位。

标签定义（0/1，均为 float）
- ``in_intersection``   自车当前 lane 属于路口 block（X/T/U）。
- ``near_intersection`` 自车位置到任意路口 block 车道中心线 < 30 m。
- ``cutin_active``      事件窗口激活 且 脚本车在自车车道内（|lat| <= 0.5·w）且距离 <= 35 m
                        （-8 m <= 相对纵向 <= 60 m 仅作投影绕圈防护）。
- ``cutout_active``     事件窗口激活 且 脚本车已离开自车车道（|lat| >= 0.5·w）且距离 <= 45 m。
- ``crowded``           自车 30 m 内车辆数 > 阈值（默认 >2，即 ≥3 辆，可由 spec 覆盖）。
- ``car_following``     自车车道内前方存在 0 < 间距 <= 20 m 的车辆（横向 |lat| <= 0.75·w）。
- ``on_curve``          自车所在车道在当前纵向位置的曲率 > 0.01 1/m（等价半径 < 100 m）。
- ``merging``           自车位置到匝道/瓶颈（r/R/y/Y）block 车道中心线 < 25 m。
- ``roundabout_near``   自车位置到环岛（O）block 车道中心线 < 30 m。

同一类型可有多条脚本事件（``cut_in#2`` 等），标签取同类型事件的并集。

来源核对：block ID 见 metadrive 0.4.3 ``component/pgblock/*.py``（S/C/r/R/f/F/y/Y/X/T/U/O/...）；
几何工具（lane_point/lane_curvature/lane_projection/map_info/event_state）统一在
``env.scenario.behaviors`` 中实现并写明了源码依据。
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle

from env.scenario.behaviors import (
    INTERSECTION_BLOCK_IDS,
    MERGE_BLOCK_IDS,
    ROUNDABOUT_BLOCK_IDS,
    ego_lane,
    ego_vehicle,
    event_state,
    lane_curvature,
    lane_projection,
    map_info,
    nearest_distance,
    spec_field,
)

LABEL_ORDER: tuple[str, ...] = (
    "in_intersection",
    "near_intersection",
    "cutin_active",
    "cutout_active",
    "crowded",
    "car_following",
    "on_curve",
    "merging",
    "roundabout_near",
)

# 距离 / 数量阈值（契约 §3-L1b；除 crowded 阈值外均可用 spec.traffic 覆盖）
NEAR_INTERSECTION_M = 30.0
NEAR_ROUNDABOUT_M = 30.0
NEAR_MERGE_M = 25.0
CROWDED_RADIUS_M = 30.0
CROWDED_DEFAULT_THRESHOLD = 2  # 车辆数 > 2 -> ≥3 辆
CAR_FOLLOWING_GAP_M = 20.0
CAR_FOLLOWING_LATERAL_FRAC = 0.75
CURVATURE_THRESHOLD = 0.01  # 1/m
CUTIN_NEAR_M = 35.0
CUTIN_AHEAD_MAX_M = 60.0  # 仅作 CircularLane 投影绕圈的防护上界（gap_m 最大 30）
CUTIN_BEHIND_M = 8.0
CUTIN_LATERAL_FRAC = 0.5
CUTOUT_NEAR_M = 45.0
CUTOUT_LATERAL_FRAC = 0.5
CUTOUT_BEHIND_M = 10.0


def _traffic_config(spec: Any) -> dict:
    traffic = spec_field(spec, "traffic", {}) or {}
    return traffic if isinstance(traffic, dict) else {}


def _float_param(traffic: dict, key: str, default: float) -> float:
    try:
        return float(traffic.get(key, default))
    except (TypeError, ValueError):
        return default


def _int_param(traffic: dict, key: str, default: int) -> int:
    try:
        return int(traffic.get(key, default))
    except (TypeError, ValueError):
        return default


def _within(points: Optional[np.ndarray], position: np.ndarray, threshold: float) -> float:
    distance = nearest_distance(points, position)
    return 1.0 if distance is not None and distance < threshold else 0.0


def _other_vehicles(engine: Any, ego: BaseVehicle) -> list[BaseVehicle]:
    objects = getattr(engine, "get_objects", None)
    if objects is None:
        return []
    try:
        all_objects = objects()
    except Exception:  # noqa: BLE001
        return []
    return [obj for obj in all_objects.values() if isinstance(obj, BaseVehicle) and obj is not ego]


def _event_labels(state: dict, width: Optional[float]) -> tuple[float, float]:
    """由脚本事件窗口 + 实时几何计算 cut-in / cut-out 激活标签（同类型多事件取并集）。"""
    cut_in = 0.0
    cut_out = 0.0
    if width is None or width <= 0:
        return cut_in, cut_out
    events = state.get("events", {}) or {}
    actors = state.get("actors", {}) or {}
    for key, event in events.items():
        kind = str(event.get("type", "")).lower()
        if kind not in ("cut_in", "cut_out"):
            continue
        actor = actors.get(key) or {}
        if not event.get("active") or not actor.get("alive"):
            continue
        lateral = actor.get("lateral")
        long_rel = actor.get("long_rel")
        distance = actor.get("distance")
        if lateral is None or distance is None:
            continue
        if kind == "cut_in":
            if (
                abs(lateral) <= CUTIN_LATERAL_FRAC * width
                and distance <= CUTIN_NEAR_M
                and (long_rel is None or (-CUTIN_BEHIND_M <= long_rel <= CUTIN_AHEAD_MAX_M))
            ):
                cut_in = 1.0
        else:
            if (
                abs(lateral) >= CUTOUT_LATERAL_FRAC * width
                and distance <= CUTOUT_NEAR_M
                and (long_rel is None or long_rel >= -CUTOUT_BEHIND_M)
            ):
                cut_out = 1.0
    return cut_in, cut_out


def compute_step_labels(env: Any, spec: Any = None) -> dict[str, float]:
    """返回当前步的多标签（全 0/1 float），键顺序固定为 ``LABEL_ORDER``。"""
    labels = {key: 0.0 for key in LABEL_ORDER}
    ego = ego_vehicle(env)
    if ego is None:
        return labels
    engine = getattr(env, "engine", env)
    lane = ego_lane(engine, ego)
    try:
        position = np.asarray(ego.position, dtype=float)
    except Exception:  # noqa: BLE001
        return labels
    projection = lane_projection(lane, position) if lane is not None else None
    longitudinal = projection[0] if projection is not None else None
    try:
        width = float(lane.width) if lane is not None else None
    except Exception:  # noqa: BLE001
        width = None
    traffic = _traffic_config(spec)

    # ---- 几何 / 地图结构标签 ----
    info = map_info(env)
    if info is not None:
        try:
            lane_index = tuple(ego.lane_index)
        except Exception:  # noqa: BLE001
            lane_index = tuple(lane.index) if lane is not None and getattr(lane, "index", None) is not None else None
        block_id = info.lane_block.get(lane_index) if lane_index is not None else None
        labels["in_intersection"] = 1.0 if block_id in INTERSECTION_BLOCK_IDS else 0.0
        labels["near_intersection"] = _within(info.intersection_points, position, NEAR_INTERSECTION_M)
        labels["roundabout_near"] = _within(info.roundabout_points, position, NEAR_ROUNDABOUT_M)
        labels["merging"] = _within(info.merge_points, position, NEAR_MERGE_M)
    if lane is not None and longitudinal is not None:
        labels["on_curve"] = 1.0 if lane_curvature(lane, longitudinal) > CURVATURE_THRESHOLD else 0.0

    # ---- 交通密度 / 跟车标签（只用当前车辆位置，排除 ego）----
    vehicles = _other_vehicles(engine, ego)
    crowded_threshold = _int_param(traffic, "crowded_threshold", CROWDED_DEFAULT_THRESHOLD)
    crowded_radius = _float_param(traffic, "crowded_radius_m", CROWDED_RADIUS_M)
    follow_gap = _float_param(traffic, "car_following_gap_m", CAR_FOLLOWING_GAP_M)
    count = 0
    following = False
    ego_long = projection[0] if projection is not None else None
    for vehicle in vehicles:
        try:
            vehicle_position = np.asarray(vehicle.position, dtype=float)
        except Exception:  # noqa: BLE001
            continue
        if float(np.linalg.norm(vehicle_position[:2] - position[:2])) <= crowded_radius:
            count += 1
        if following or lane is None or ego_long is None or width is None:
            continue
        vehicle_projection = lane_projection(lane, vehicle_position)
        if vehicle_projection is None:
            continue
        rel_long = vehicle_projection[0] - ego_long
        euclidean = float(np.linalg.norm(vehicle_position[:2] - position[:2]))
        if (
            0.0 < rel_long <= follow_gap
            and abs(vehicle_projection[1]) <= CAR_FOLLOWING_LATERAL_FRAC * width
            and euclidean <= follow_gap + 5.0  # 防 CircularLane 投影绕圈产生的假"前车"
        ):
            following = True
    labels["crowded"] = 1.0 if count > crowded_threshold else 0.0
    labels["car_following"] = 1.0 if following else 0.0

    # ---- 脚本事件标签（窗口 + 实时几何；无 behaviors 时恒 0）----
    state = event_state(env)
    cut_in, cut_out = _event_labels(state, width)
    labels["cutin_active"] = cut_in
    labels["cutout_active"] = cut_out
    return labels
