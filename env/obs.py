"""Structured observation contract for the PNP planner.

This module is intentionally independent of the old Planner_RL observation schema.
It builds fixed-shape OD / LD / Nav / Ego tensors in the current ego frame and can
re-align a short history to the current frame for the memory bank.

Shapes:
  od:  [OD_SLOTS, OD_DIM]              OD_DIM = 11
  ld:  [LD_SLOTS, LD_POINTS, LD_DIM]   LD_DIM = 9
  nav: [NAV_DIM]                       NAV_DIM = 11
  ego: [EGO_DIM]                       EGO_DIM = 8
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from metadrive.component.buildings.base_building import BaseBuilding
from metadrive.component.static_object.traffic_object import TrafficCone
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils.math import wrap_to_pi

OD_SLOTS = 16
OD_DIM = 11
LD_SLOTS = 16
LD_POINTS = 5
LD_DIM = 9
NAV_DIM = 11
EGO_DIM = 8
HISTORY_FRAMES = 6

# OD feature layout
OD_FEATURES = (
    "dx", "dy", "cos_theta", "sin_theta", "vx_rel", "vy_rel",
    "length", "width", "type_id", "static_flag", "presence",
)
# LD point feature layout
LD_FEATURES = (
    "dx", "dy", "heading_rel", "curvature", "speed_limit",
    "left_type", "right_type", "lane_role", "lat_offset",
)
NAV_FEATURES = (
    "cp0_dx", "cp0_dy", "cp1_dx", "cp1_dy",
    "cmd_forward", "cmd_left", "cmd_right", "route_completion",
    "reserved0", "reserved1", "reserved2",
)
EGO_FEATURES = (
    "speed", "a_long", "a_lat", "yaw_rate", "steer", "curvature",
    "prev_dx", "prev_dpsi",
)

# MetaDrive line type -> stable id
LINE_TYPE_IDS = {
    "ROAD_LINE_UNKNOWN": 0,
    "ROAD_LINE_BROKEN_SINGLE_WHITE": 1,
    "ROAD_LINE_SOLID_SINGLE_WHITE": 2,
    "ROAD_LINE_SOLID_DOUBLE_WHITE": 3,
    "ROAD_LINE_BROKEN_SINGLE_YELLOW": 4,
    "ROAD_LINE_BROKEN_DOUBLE_YELLOW": 5,
    "ROAD_LINE_SOLID_SINGLE_YELLOW": 6,
    "ROAD_LINE_SOLID_DOUBLE_YELLOW": 7,
    "ROAD_LINE_PASSING_DOUBLE_YELLOW": 8,
    "ROAD_EDGE_BOUNDARY": 9,
    "ROAD_EDGE_MEDIAN": 10,
    "ROAD_EDGE_SIDEWALK": 11,
    "GUARDRAIL": 12,
}


def _line_type_id(value: Any) -> int:
    return int(LINE_TYPE_IDS.get(str(value), 0))


def _type_id(obj: Any) -> Tuple[int, float]:
    """Return (type id, static flag)."""
    name = type(obj).__name__
    if isinstance(obj, TrafficCone):
        return 5, 1.0
    if isinstance(obj, BaseBuilding):
        return 6, 1.0
    if isinstance(obj, BaseVehicle):
        vehicle_ids = {
            "DefaultVehicle": 0,
            "TrafficDefaultVehicle": 0,
            "StaticDefaultVehicle": 0,
            "SVehicle": 1,
            "MVehicle": 2,
            "LVehicle": 3,
            "XLVehicle": 4,
            "VaryingDynamicsVehicle": 0,
            "VaryingDynamicsBoundingBoxVehicle": 0,
        }
        return int(vehicle_ids.get(name, 7)), 0.0
    return 7, 1.0


def ego_pose(env: Any) -> np.ndarray:
    ego = env.agent
    p = np.asarray(ego.position, dtype=np.float64)
    return np.array([p[0], p[1], float(wrap_to_pi(ego.heading_theta))], dtype=np.float64)


def _to_local(ego: Any, xy: Sequence[float]) -> np.ndarray:
    p = np.asarray(ego.convert_to_local_coordinates(np.asarray(xy, dtype=np.float64), ego.position), dtype=np.float64)
    return p[:2]


def _rel_velocity(ego: Any, obj: Any) -> Tuple[float, float]:
    try:
        v_obj = np.asarray(obj.velocity, dtype=np.float64)[:2]
        v_ego = np.asarray(ego.velocity, dtype=np.float64)[:2]
        d = v_obj - v_ego
        th = float(ego.heading_theta)
        c, s = math.cos(th), math.sin(th)
        # rotate world vector into ego frame: x forward, y left
        return float(c * d[0] + s * d[1]), float(-s * d[0] + c * d[1])
    except Exception:
        return 0.0, 0.0


def _lane_curvature(lane: Any, s: float) -> float:
    try:
        s0 = max(0.0, float(s) - 1.0)
        s1 = min(float(lane.length), float(s) + 1.0)
        if s1 - s0 < 1e-3:
            return 0.0
        h0 = float(lane.heading_theta_at(s0))
        h1 = float(lane.heading_theta_at(s1))
        return float(wrap_to_pi(h1 - h0) / (s1 - s0))
    except Exception:
        return 0.0


def _lane_candidates(env: Any, ego: Any, max_n: int = LD_SLOTS) -> List[Tuple[float, Any, float, int]]:
    """Return (distance, lane, s0, role) sorted by distance.

    role: 0 current lane, 1 left adjacent, 2 right adjacent, 3 other.
    No target-lane / recommended-lane information is used.
    """
    try:
        lanes = list(env.current_map.road_network.get_all_lanes())
    except Exception:
        return []
    ego_lane = getattr(ego, "lane", None)
    ego_lane_idx = getattr(ego_lane, "index", None)
    ego_road = tuple(ego_lane_idx[:2]) if ego_lane_idx is not None else None
    ego_lane_id = int(ego_lane_idx[2]) if ego_lane_idx is not None else None
    out: List[Tuple[float, Any, float, int]] = []
    for lane in lanes:
        try:
            s0, lat = lane.local_coordinates(ego.position)
            s0 = float(np.clip(s0, 0.0, float(lane.length)))
            # longitudinal / lateral gate in the lane's own frame
            if s0 < -15.0 or s0 > 80.0:
                continue
            if abs(float(lat)) > 20.0:
                continue
            pos = lane.position(s0, 0.0)
            d = float(np.linalg.norm(np.asarray(pos)[:2] - np.asarray(ego.position)[:2]))
            role = 3
            try:
                idx = lane.index
                if ego_road is not None and tuple(idx[:2]) == ego_road:
                    if ego_lane_id is not None:
                        if int(idx[2]) < ego_lane_id:
                            role = 1
                        elif int(idx[2]) > ego_lane_id:
                            role = 2
                        else:
                            role = 0
            except Exception:
                pass
            out.append((d, lane, s0, role))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out[:max_n]


def build_observation(
    env: Any,
    *,
    prev_action: Optional[Sequence[float]] = None,
    prev_speed: Optional[float] = None,
    prev_heading: Optional[float] = None,
    dt: float = 0.5,
) -> Dict[str, np.ndarray]:
    """Build one current-frame observation from a live MetaDrive env."""
    ego = env.agent
    th = float(ego.heading_theta)
    speed = float(ego.speed)
    prev_action = np.zeros(4, dtype=np.float64) if prev_action is None else np.asarray(prev_action, dtype=np.float64)

    # ---- Ego ----
    a_long = 0.0 if prev_speed is None else (speed - float(prev_speed)) / max(dt, 1e-6)
    yaw_rate = 0.0 if prev_heading is None else float(wrap_to_pi(th - float(prev_heading))) / max(dt, 1e-6)
    a_lat = speed * yaw_rate
    curvature = yaw_rate / max(speed, 0.5)
    ego_feat = np.array([
        speed, a_long, a_lat, yaw_rate,
        float(getattr(ego, "steering", 0.0)),
        curvature,
        float(prev_action[0]) if len(prev_action) else 0.0,
        float(prev_action[2]) if len(prev_action) > 2 else 0.0,
    ], dtype=np.float32)

    # ---- OD ----
    od = np.zeros((OD_SLOTS, OD_DIM), dtype=np.float32)
    od_mask = np.zeros(OD_SLOTS, dtype=np.float32)
    objs: List[Tuple[float, Any]] = []
    try:
        for obj in env.engine.get_objects().values():
            if obj is ego:
                continue
            if not isinstance(obj, (BaseVehicle, TrafficCone, BaseBuilding)):
                continue
            rel = _to_local(ego, obj.position)
            if not (-20.0 <= rel[0] <= 80.0 and -20.0 <= rel[1] <= 20.0):
                continue
            objs.append((float(np.linalg.norm(rel)), obj))
    except Exception:
        objs = []
    objs.sort(key=lambda x: x[0])
    for i, (_, obj) in enumerate(objs[:OD_SLOTS]):
        rel = _to_local(ego, obj.position)
        try:
            rel_h = float(wrap_to_pi(float(obj.heading_theta) - th))
        except Exception:
            rel_h = 0.0
        vx, vy = _rel_velocity(ego, obj) if isinstance(obj, BaseVehicle) else (0.0, 0.0)
        tid, static = _type_id(obj)
        try:
            length = float(getattr(obj, "LENGTH", getattr(obj, "length", 2.0)))
            width = float(getattr(obj, "WIDTH", getattr(obj, "width", 1.5)))
        except Exception:
            length, width = 2.0, 1.5
        od[i] = np.array([
            rel[0], rel[1], math.cos(rel_h), math.sin(rel_h), vx, vy,
            length, width, tid, static, 1.0,
        ], dtype=np.float32)
        od_mask[i] = 1.0

    # ---- LD ----
    ld = np.zeros((LD_SLOTS, LD_POINTS, LD_DIM), dtype=np.float32)
    ld_mask = np.zeros((LD_SLOTS, LD_POINTS), dtype=np.float32)
    for i, (_, lane, s0, role) in enumerate(_lane_candidates(env, ego)):
        offsets = (0.0, 5.0, 10.0, 20.0, 30.0)
        try:
            lt = getattr(lane, "line_types", ("ROAD_LINE_UNKNOWN", "ROAD_LINE_UNKNOWN"))
            left_id = _line_type_id(lt[0] if len(lt) > 0 else 0)
            right_id = _line_type_id(lt[1] if len(lt) > 1 else 0)
            limit = float(getattr(lane, "speed_limit", 1000.0))
            if not (0.0 < limit < 1000.0):
                limit = 13.9
        except Exception:
            left_id = right_id = 0
            limit = 13.9
        try:
            _s_ego, _lat_ego = lane.local_coordinates(ego.position)
            lat_offset = float(_lat_ego)
        except Exception:
            lat_offset = 0.0
        for j, off in enumerate(offsets):
            s = float(np.clip(s0 + off, 0.0, float(lane.length)))
            try:
                p = np.asarray(lane.position(s, 0.0), dtype=np.float64)[:2]
                rel = _to_local(ego, p)
                head = float(wrap_to_pi(float(lane.heading_theta_at(s)) - th))
                ld[i, j] = np.array([
                    rel[0], rel[1], head, _lane_curvature(lane, s),
                    limit, left_id, right_id, role, lat_offset,
                ], dtype=np.float32)
                ld_mask[i, j] = 1.0
            except Exception:
                pass

    # ---- Nav ----
    nav = np.zeros(NAV_DIM, dtype=np.float32)
    nav_mask = np.zeros(1, dtype=np.float32)
    try:
        nv = ego.navigation
        cps = list(nv.get_checkpoints())
        for k, cp in enumerate(cps[:2]):
            rel = _to_local(ego, np.asarray(cp)[:2])
            nav[2 * k:2 * k + 2] = rel
        # command from navi_arrow_dir, same convention as old env.obs.nav
        cmd = "forward"
        try:
            arrow = getattr(nv, "navi_arrow_dir", None)
            if arrow is not None and len(arrow) >= 2:
                h0, h1 = float(arrow[0]), float(arrow[1])
                if abs(float(wrap_to_pi(h0 - h1))) >= math.radians(10.0):
                    d0 = np.array([math.cos(h0), math.sin(h0)])
                    d1 = np.array([math.cos(h1), math.sin(h1)])
                    cross = d1[0] * d0[1] - d1[1] * d0[0]
                    cmd = "left" if cross < 0 else "right"
        except Exception:
            pass
        nav[4 + {"forward": 0, "left": 1, "right": 2}.get(cmd, 0)] = 1.0
        nav[7] = float(getattr(nv, "route_completion", 0.0))
        nav_mask[0] = 1.0
    except Exception:
        pass

    return {
        "od": od, "od_mask": od_mask,
        "ld": ld, "ld_mask": ld_mask,
        "nav": nav, "nav_mask": nav_mask,
        "ego": ego_feat, "ego_mask": np.ones(1, dtype=np.float32),
        "pose": ego_pose(env),
    }


def _transform_xy(x: float, y: float, src_pose: np.ndarray, dst_pose: np.ndarray) -> Tuple[float, float]:
    # src ego frame -> world
    th = float(src_pose[2])
    c, s = math.cos(th), math.sin(th)
    wx = float(src_pose[0]) + c * x - s * y
    wy = float(src_pose[1]) + s * x + c * y
    # world -> dst ego frame
    dx, dy = wx - float(dst_pose[0]), wy - float(dst_pose[1])
    th2 = float(dst_pose[2])
    c2, s2 = math.cos(th2), math.sin(th2)
    return float(c2 * dx + s2 * dy), float(-s2 * dx + c2 * dy)


def _rotate_xy(x: float, y: float, angle: float) -> Tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return float(c * x - s * y), float(s * x + c * y)


def align_to_current(obs: Dict[str, np.ndarray], src_pose: np.ndarray, dst_pose: np.ndarray) -> Dict[str, np.ndarray]:
    """Align a historical observation into the current ego frame (geometry only)."""
    out = {k: np.array(v, copy=True) for k, v in obs.items() if k != "pose"}
    dth = float(wrap_to_pi(float(src_pose[2]) - float(dst_pose[2])))
    # OD: dx,dy (0,1); cos/sin (2,3); relative velocity (4,5)
    for i in range(OD_SLOTS):
        if out["od_mask"][i] <= 0:
            continue
        x, y = _transform_xy(float(out["od"][i, 0]), float(out["od"][i, 1]), src_pose, dst_pose)
        out["od"][i, 0], out["od"][i, 1] = x, y
        c, s = float(out["od"][i, 2]), float(out["od"][i, 3])
        out["od"][i, 2], out["od"][i, 3] = _rotate_xy(c, s, dth)
        vx, vy = _rotate_xy(float(out["od"][i, 4]), float(out["od"][i, 5]), dth)
        out["od"][i, 4], out["od"][i, 5] = vx, vy
    # LD: dx,dy (0,1); heading_rel (2)
    for i in range(LD_SLOTS):
        for j in range(LD_POINTS):
            if out["ld_mask"][i, j] <= 0:
                continue
            x, y = _transform_xy(float(out["ld"][i, j, 0]), float(out["ld"][i, j, 1]), src_pose, dst_pose)
            out["ld"][i, j, 0], out["ld"][i, j, 1] = x, y
            out["ld"][i, j, 2] = float(wrap_to_pi(float(out["ld"][i, j, 2]) + dth))
    # Nav checkpoints (0,1) and (2,3)
    if float(out["nav_mask"][0]) > 0:
        for k in (0, 2):
            x, y = _transform_xy(float(out["nav"][k]), float(out["nav"][k + 1]), src_pose, dst_pose)
            out["nav"][k], out["nav"][k + 1] = x, y
    return out


def stack_history(frames: Sequence[Dict[str, np.ndarray]], poses: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    """Stack up to HISTORY_FRAMES frames, oldest first, aligned to the newest pose.

    Missing warmup frames repeat the oldest real frame with valid=0.
    """
    n = min(len(frames), HISTORY_FRAMES)
    frames = list(frames[-n:])
    poses = list(poses[-n:])
    if not frames:
        raise ValueError("stack_history: no frames")
    dst = poses[-1]
    out = {
        "od": np.zeros((HISTORY_FRAMES, OD_SLOTS, OD_DIM), dtype=np.float32),
        "od_mask": np.zeros((HISTORY_FRAMES, OD_SLOTS), dtype=np.float32),
        "ld": np.zeros((HISTORY_FRAMES, LD_SLOTS, LD_POINTS, LD_DIM), dtype=np.float32),
        "ld_mask": np.zeros((HISTORY_FRAMES, LD_SLOTS, LD_POINTS), dtype=np.float32),
        "nav": np.zeros((HISTORY_FRAMES, NAV_DIM), dtype=np.float32),
        "nav_mask": np.zeros((HISTORY_FRAMES, 1), dtype=np.float32),
        "ego": np.zeros((HISTORY_FRAMES, EGO_DIM), dtype=np.float32),
        "ego_mask": np.zeros((HISTORY_FRAMES, 1), dtype=np.float32),
        "hist_valid": np.zeros(HISTORY_FRAMES, dtype=np.float32),
    }
    # left-pad by repeating oldest
    pad = HISTORY_FRAMES - n
    for t in range(HISTORY_FRAMES):
        src = 0 if t < pad else t - pad
        aligned = align_to_current(frames[src], poses[src], dst)
        out["od"][t] = aligned["od"]
        out["od_mask"][t] = aligned["od_mask"]
        out["ld"][t] = aligned["ld"]
        out["ld_mask"][t] = aligned["ld_mask"]
        out["nav"][t] = aligned["nav"]
        out["nav_mask"][t] = aligned["nav_mask"]
        out["ego"][t] = aligned["ego"]
        out["ego_mask"][t] = aligned["ego_mask"]
        out["hist_valid"][t] = 1.0 if t >= pad else 0.0
    return out
