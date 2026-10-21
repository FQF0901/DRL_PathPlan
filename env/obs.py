"""Forward structured observation schema v2 (OD / LD / road-level Nav / Ego).

This module is the *new* forward contract (``schema_version=2``).  It does not
modify and does not call the legacy ``env.obs.build_observation`` /
``env.obs.stack_history`` behavior; it only reuses the low-level geometry
helpers (ego pose, local transform, relative velocity, line-type mapping,
curvature, lane candidate filter).

Design facts / assumptions that are intentionally explicit
-----------------------------------------------------------
* Road-level navigation is built from ``navigation.current_road``,
  ``navigation.next_road``, ``navigation.final_road`` and
  ``navigation.checkpoints`` only.  ``get_checkpoints()``,
  ``navi_arrow_dir``, ``current_ref_lanes``, ``next_ref_lanes`` and
  ``final_lane`` are lane-level/legacy oracle APIs and are **never** read.
* MetaDrive 0.4.3 has no node-coordinate API.  A road is represented by the
  aggregate of *all* its lanes: mean entry / mid / exit positions and circular
  mean headings.  ``lanes[0]`` is never used as a target.
* Stable IDs are signed int64 values produced by BLAKE2b over
  ``scene_seed`` plus the entity/topology key.  They are alignment/diagnostic
  metadata, not policy inputs.
* OD / LD anchors below are ``(dx, dy, heading_rel)`` in the ego frame at the
  time the frame was captured.  ``stack_history`` SE(2)-compensates the
  anchors and the geometry features into the current (newest) ego frame, while
  IDs / taxonomy / masks are preserved.
* LD chunk anchor: ``floor(s0 / LD_CHUNK_METERS) * LD_CHUNK_METERS`` along the
  lane.  It is part of the LD stable ID (``ld_chunk_s``) so a local crop-window
  change does not silently reuse an old same-ID latent as if nothing changed.
  Polyline points remain ego-relative 0/5/10/20/30 m samples from the current
  longitudinal position; the separately stored chunk anchor makes the crop
  window explicit.
* Nav road IDs are ``[road_id, route_index]`` pairs.  ``road_id`` is
  ``hash(scene_seed, start_node, end_node)``; ``route_index`` is the index of
  the road in the current route (0 = current road, -1 = not on route).  LD
  road IDs are ``[road_id, lane_id_in_road]``.
* Nav slots are semantically ``[current, next, next-next, final]`` and are
  deduplicated by road ID; a short route therefore masks unused slots.
  Route-level padding is represented by ``nav_road_mask`` / ``nav_mask``.

Shapes
------
``od``             [OD_SLOTS, OD_DIM]                  = [16, 18]
``ld``             [LD_SLOTS, LD_POINTS, LD_DIM]       = [16, 5, 16]
``nav``            [NAV_ROADS * NAV_ROAD_DIM + NAV_GLOBAL_DIM] = [108]
``ego``            [EGO_DIM]                           = [12]
``stack_history`` prepends a history axis of ``HISTORY_FRAMES`` to every
tensor and returns all metadata tensors with the same history axis.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from metadrive.component.buildings.base_building import BaseBuilding
from metadrive.component.road_network.road import Road
from metadrive.component.static_object.traffic_object import TrafficCone
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils.math import wrap_to_pi

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


def _lane_candidates(env: Any, ego: Any, max_n: int = 16) -> List[Tuple[float, Any, float, int]]:
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



# ---------------------------------------------------------------------------
# Fixed schema sizes
# ---------------------------------------------------------------------------

OD_SLOTS = 16
OD_DIM = 18
LD_SLOTS = 16
LD_POINTS = 5
LD_DIM = 16
NAV_ROADS = 4
NAV_ROAD_DIM = 24
NAV_GLOBAL_DIM = 12
NAV_DIM = NAV_ROADS * NAV_ROAD_DIM + NAV_GLOBAL_DIM  # 108
EGO_DIM = 12
HISTORY_FRAMES = 6

SCHEMA_NAME = "drl_pathplan_obs"
SCHEMA_VERSION = 2
DEFAULT_DT = 0.5
LD_CHUNK_METERS = 20.0
LD_POINT_OFFSETS = (0.0, 5.0, 10.0, 20.0, 30.0)
OD_AGE_NORM_S = 3.0
#: Fallback if the environment cannot expose its physics timing.
OD_ACC_DT = 0.1


def _physics_dt(env: Any) -> float:
    """Derive the acceleration difference interval from MetaDrive config."""
    try:
        cfg = getattr(getattr(env, "engine", None), "global_config", None) or {}
        physics = float(cfg.get("physics_world_step_size", 0.02))
        repeat = int(cfg.get("decision_repeat", 5))
        dt = physics * repeat
        if math.isfinite(dt) and dt > 0.0:
            return dt
    except (AttributeError, TypeError, ValueError):
        pass
    return OD_ACC_DT
EGO_LENGTH_NORM = 5.0
EGO_WIDTH_NORM = 2.0

# OD feature layout: base fields first, then taxonomy/age fields.
OD_FEATURES = (
    "dx", "dy", "cos_theta", "sin_theta", "vx_rel", "vy_rel",
    "length", "width", "type_id", "static_flag", "presence",
    "acc_x", "acc_y", "age_norm",
    "semantic_type_id_norm", "mobility_type", "geometry_type", "state_type",
)
OD_INDEX = {name: i for i, name in enumerate(OD_FEATURES)}
OD_AGE_INDEX = OD_INDEX["age_norm"]
assert len(OD_FEATURES) == OD_DIM

# LD point feature layout: base fields first, then topology fields.
LD_FEATURES = (
    "dx", "dy", "heading_rel", "curvature", "speed_limit",
    "left_type", "right_type", "lane_role", "lat_offset",
    "has_left", "has_right", "has_prev", "has_next",
    "road_type_norm", "is_junction", "anchor_s_norm",
)
LD_INDEX = {name: i for i, name in enumerate(LD_FEATURES)}
assert len(LD_FEATURES) == LD_DIM

# Nav per-road layout = 24 features:
#   0:6   entry/mid/exit (dx,dy) / 100
#   6:8   entry/exit heading relative to ego (rad)
#   8:11  length / 100, width / 10, lane_count / 8
#   11:13 turn sin/cos, 13:15 out/in degree / 8, 15 type id / 15
#   16:21 is_current/is_next/is_negative/is_toll/is_junction
#   21:24 route_remaining_ratio, road_progress_frac, valid
NAV_ROAD_FEATURES = (
    "entry_dx", "entry_dy", "mid_dx", "mid_dy", "exit_dx", "exit_dy",
    "entry_heading_rel", "exit_heading_rel",
    "length_norm", "width_norm", "lane_count_norm",
    "turn_sin", "turn_cos", "out_degree_norm", "in_degree_norm",
    "type_id_norm",
    "is_current", "is_next", "is_negative", "is_toll", "is_junction",
    "route_remaining_ratio", "road_progress_frac", "valid",
)
NAV_ROAD_INDEX = {name: i for i, name in enumerate(NAV_ROAD_FEATURES)}
assert len(NAV_ROAD_FEATURES) == NAV_ROAD_DIM

# Nav global layout = 12 features:
#   route completion / remaining / total, current road length / progress,
#   remaining road count, junction ahead, destination dx/dy + heading sin/cos,
#   nav_valid.
NAV_GLOBAL_FEATURES = (
    "route_completion", "route_remaining_norm", "route_total_norm",
    "current_road_length_norm", "current_road_progress",
    "remaining_road_count_norm", "junction_ahead",
    "dest_dx", "dest_dy", "dest_heading_sin", "dest_heading_cos",
    "nav_valid",
)
NAV_GLOBAL_INDEX = {name: NAV_ROADS * NAV_ROAD_DIM + i for i, name in enumerate(NAV_GLOBAL_FEATURES)}
assert len(NAV_GLOBAL_FEATURES) == NAV_GLOBAL_DIM

EGO_FEATURES = (
    "speed", "a_long", "a_lat", "yaw_rate", "steer", "curvature",
    "prev_dx", "prev_dpsi", "v_long", "v_lat", "length_norm", "width_norm",
)
EGO_INDEX = {name: i for i, name in enumerate(EGO_FEATURES)}
assert len(EGO_FEATURES) == EGO_DIM

# Road block character -> stable road type id (0..15).
ROAD_TYPE_IDS = {
    "unknown": 0,
    "S": 1,
    "C": 2,
    "r": 3,
    "R": 4,
    "y": 5,
    "Y": 6,
    "X": 7,
    "T": 8,
    "O": 9,
    "U": 10,
    "B": 11,
    "$": 12,
    "I": 13,
    "d": 14,
}
JUNCTION_BLOCK_IDS = frozenset(("X", "T", "U", "O"))
TOLL_BLOCK_IDS = frozenset(("$",))

# OD semantic / geometry / state taxonomy ids.  Raw taxonomy metadata keeps
# these ids; the feature vector normalizes them.
OD_SEMANTIC_IDS = {
    "unknown": 0,
    "car": 1,
    "s_vehicle": 2,
    "m_vehicle": 3,
    "l_vehicle": 4,
    "xl_vehicle": 5,
    "cone": 6,
    "building": 7,
    "other": 8,
}
OD_SEMANTIC_NORM = 8.0
OD_GEOMETRY_IDS = {"unknown": 0, "point": 1, "box": 2, "building": 3}
OD_GEOMETRY_NORM = 3.0
OD_STATE_IDS = {"unknown": 0, "moving": 1, "stopped": 2, "static": 3}
OD_STATE_NORM = 3.0


# ---------------------------------------------------------------------------
# Stable IDs / small numeric helpers
# ---------------------------------------------------------------------------

def _stable_hash(scene_seed: int, *parts: Any) -> int:
    """Return a deterministic signed int64 BLAKE2b hash."""
    digest = hashlib.blake2b(digest_size=8, person=b"drlppv2")
    digest.update(struct.pack("<q", int(scene_seed)))
    for part in parts:
        digest.update(b"\x1f")
        if isinstance(part, (tuple, list)):
            for sub in part:
                digest.update(b"\x1e")
                digest.update(str(sub).encode("utf-8"))
        elif isinstance(part, float):
            # Avoid platform-dependent repr differences for chunk anchors.
            digest.update(f"{part:.6f}".encode("ascii"))
        else:
            digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little", signed=True)


def _scene_seed(env: Any) -> int:
    engine = getattr(env, "engine", None)
    candidates = (
        getattr(env, "current_seed", None),
        getattr(engine, "global_random_seed", None),
        getattr(getattr(env, "_spec", None), "seed", None),
    )
    for value in candidates:
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
    config = getattr(engine, "global_config", {}) if engine is not None else {}
    value = config.get("start_seed") if isinstance(config, dict) else None
    return int(value) if isinstance(value, (int, np.integer)) else 0


def _scene_timestamp(env: Any, policy_dt: float) -> Tuple[float, int]:
    """Return (simulation time, env-step index).

    ``episode_step`` counts ``env.step`` calls (0.1 s here), not policy
    decisions (0.5 s).  Read the simulator cadence from the config and fall
    back to 0.1 s; the policy dt is intentionally not used for this conversion.
    """
    engine = getattr(env, "engine", None)
    step = getattr(engine, "episode_step", 0) if engine is not None else 0
    try:
        step = int(step)
    except (TypeError, ValueError):
        step = 0
    sim_dt = 0.1
    try:
        config = engine.global_config
        sim_dt = float(config.get("physics_world_step_size", 0.02)) * int(config.get("decision_repeat", 5))
    except Exception:  # noqa: BLE001
        sim_dt = 0.1
    if not math.isfinite(sim_dt) or sim_dt <= 0.0:
        sim_dt = 0.1
    return float(step) * float(sim_dt), step


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _safe_positive(value: Any, fallback: float) -> float:
    out = _safe_float(value, fallback)
    return out if out > 1e-6 else float(fallback)


def _finite_float(array: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(dtype)


def _wrap_angle(value: Any) -> float:
    """Wrap to (-pi, pi] and canonicalize the +pi/-pi float32 ambiguity to -pi."""
    angle = float(wrap_to_pi(float(value)))
    if angle > math.pi - 1e-6:
        angle -= 2.0 * math.pi
    return float(angle)


def _circular_mean_heading(angles: Sequence[float]) -> float:
    if not angles:
        return 0.0
    sins = [math.sin(float(a)) for a in angles]
    coss = [math.cos(float(a)) for a in angles]
    return float(math.atan2(sum(sins), sum(coss)))


def _object_acceleration(ego: Any, obj: Any, dt: float) -> Tuple[float, float]:
    """World-frame acceleration rotated into the current ego frame.

    MetaDrive initialises ``last_velocity`` to the current velocity and updates
    it every physics step; if it is unavailable we return zero rather than
    inventing a finite difference from a stale value.
    """
    try:
        velocity = np.asarray(obj.velocity, dtype=np.float64)[:2]
        last = np.asarray(getattr(obj, "last_velocity", None), dtype=np.float64)[:2]
        if velocity.shape != (2,) or last.shape != (2,):
            return 0.0, 0.0
        if not (np.isfinite(velocity).all() and np.isfinite(last).all()):
            return 0.0, 0.0
        acc = (velocity - last) / max(float(dt), 1e-6)
        theta = _safe_float(getattr(ego, "heading_theta", 0.0))
        c, s = math.cos(theta), math.sin(theta)
        return float(c * acc[0] + s * acc[1]), float(-s * acc[0] + c * acc[1])
    except Exception:  # noqa: BLE001 - schema must always be finite
        return 0.0, 0.0


def _obj_stable_key(obj: Any) -> str:
    value = getattr(obj, "id", None)
    if value is None:
        value = getattr(obj, "name", None)
    if value is not None:
        return str(value)
    # Very defensive fallback for exotic engine objects.  It is stable within a
    # frame but not guaranteed across resets; supported objects have id/name.
    return f"{type(obj).__name__}@{getattr(obj, 'spawn_index', '?')}"


def _od_taxonomy(obj: Any, type_id: int, static_flag: float) -> Tuple[int, int, int, int]:
    if isinstance(obj, TrafficCone):
        return OD_SEMANTIC_IDS["cone"], 0, OD_GEOMETRY_IDS["point"], OD_STATE_IDS["static"]
    if isinstance(obj, BaseBuilding):
        return OD_SEMANTIC_IDS["building"], 0, OD_GEOMETRY_IDS["building"], OD_STATE_IDS["static"]
    if isinstance(obj, BaseVehicle):
        semantic = {
            0: OD_SEMANTIC_IDS["car"],
            1: OD_SEMANTIC_IDS["s_vehicle"],
            2: OD_SEMANTIC_IDS["m_vehicle"],
            3: OD_SEMANTIC_IDS["l_vehicle"],
            4: OD_SEMANTIC_IDS["xl_vehicle"],
        }.get(int(type_id), OD_SEMANTIC_IDS["car"])
        state = OD_STATE_IDS["moving"] if _safe_float(getattr(obj, "speed", 0.0)) > 0.1 else OD_STATE_IDS["stopped"]
        return semantic, 1, OD_GEOMETRY_IDS["box"], state
    semantic = OD_SEMANTIC_IDS["other"]
    mobility = 0 if static_flag > 0.5 else 1
    return semantic, mobility, OD_GEOMETRY_IDS["unknown"], OD_STATE_IDS["unknown"]


# ---------------------------------------------------------------------------
# Road topology / aggregate geometry
# ---------------------------------------------------------------------------

def _road_pair(road: Any) -> Optional[Tuple[str, str]]:
    if road is None:
        return None
    start = getattr(road, "start_node", None)
    end = getattr(road, "end_node", None)
    if start is None and isinstance(road, (tuple, list)) and len(road) >= 2:
        start, end = road[0], road[1]
    if start is None or end is None:
        return None
    return str(start), str(end)


def _checkpoint_pairs(checkpoints: Sequence[Any]) -> List[Tuple[str, str]]:
    return [(str(checkpoints[i]), str(checkpoints[i + 1])) for i in range(len(checkpoints) - 1)]


def _road_block_id(start: str, end: str) -> str:
    """Recover the MetaDrive block character for a road.

    ``Road.block_ID()`` already implements the convention used by MetaDrive
    0.4.3 (it inspects the positive/negative node name).  First-block nodes
    ``>``/``<`` are mapped to the straight block ``S``.
    """
    char = ""
    try:
        char = str(Road(start, end).block_ID())
    except Exception:  # noqa: BLE001
        char = ""
    if char in (">", "<"):
        return "S"
    if char in ROAD_TYPE_IDS:
        return char
    for node in (start, end):
        match = re.search(r"[SCrRyYXT OUB$I]", str(node))
        if match:
            return match.group(0)
    return "unknown"


def _road_id(scene_seed: int, pair: Tuple[str, str]) -> int:
    return _stable_hash(scene_seed, "road", pair[0], pair[1])


def _road_geometry(
    road_network: Any,
    pair: Tuple[str, str],
    cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """All-lane aggregate geometry for one road (never lanes[0] alone)."""
    if pair in cache:
        return cache[pair]
    start, end = pair
    lanes: List[Any] = []
    try:
        lanes = list(road_network.graph.get(start, {}).get(end, []))
    except Exception:  # noqa: BLE001
        lanes = []
    if not lanes:
        cache[pair] = None
        return None

    entries: List[np.ndarray] = []
    mids: List[np.ndarray] = []
    exits: List[np.ndarray] = []
    entry_headings: List[float] = []
    mid_headings: List[float] = []
    exit_headings: List[float] = []
    lengths: List[float] = []
    widths: List[float] = []
    for lane in lanes:
        try:
            length = _safe_positive(getattr(lane, "length", 0.0), 0.0)
            if length <= 0.0:
                continue
            entries.append(np.asarray(lane.position(0.0, 0.0), dtype=np.float64)[:2])
            mids.append(np.asarray(lane.position(length / 2.0, 0.0), dtype=np.float64)[:2])
            exits.append(np.asarray(lane.position(length, 0.0), dtype=np.float64)[:2])
            entry_headings.append(_safe_float(lane.heading_theta_at(0.0)))
            mid_headings.append(_safe_float(lane.heading_theta_at(length / 2.0)))
            exit_headings.append(_safe_float(lane.heading_theta_at(length)))
            lengths.append(float(length))
            widths.append(_safe_positive(getattr(lane, "width", None), 3.5))
        except Exception:  # noqa: BLE001
            continue
    if not entries:
        cache[pair] = None
        return None

    entry = np.mean(np.stack(entries, axis=0), axis=0)
    mid = np.mean(np.stack(mids, axis=0), axis=0)
    exit_point = np.mean(np.stack(exits, axis=0), axis=0)
    entry_heading = _circular_mean_heading(entry_headings)
    mid_heading = _circular_mean_heading(mid_headings)
    exit_heading = _circular_mean_heading(exit_headings)
    length = float(np.mean(lengths))
    width = float(np.mean(widths))
    block_id = _road_block_id(start, end)

    try:
        out_degree = len(road_network.graph.get(end, {}))
    except Exception:  # noqa: BLE001
        out_degree = 0
    in_degree = 0
    try:
        in_degree = sum(1 for pred in road_network.graph if start in road_network.graph[pred])
    except Exception:  # noqa: BLE001
        in_degree = 0

    info = {
        "pair": pair,
        "lanes": lanes,
        "entry": entry,
        "mid": mid,
        "exit": exit_point,
        "entry_heading": entry_heading,
        "mid_heading": mid_heading,
        "exit_heading": exit_heading,
        "length": length,
        "width": width,
        "lane_count": len(lanes),
        "block_id": block_id,
        "type_id": int(ROAD_TYPE_IDS.get(block_id, 0)),
        "out_degree": int(out_degree),
        "in_degree": int(in_degree),
        "is_negative": bool(_safe_road_negative(start, end)),
        "is_toll": block_id in TOLL_BLOCK_IDS,
        "is_junction": block_id in JUNCTION_BLOCK_IDS,
    }
    cache[pair] = info
    return info


def _safe_road_negative(start: str, end: str) -> bool:
    try:
        return bool(Road(start, end).is_negative_road())
    except Exception:  # noqa: BLE001
        return "-" in str(end)


def _road_progress(ego: Any, info: Dict[str, Any]) -> float:
    """Mean normalized longitudinal progress over all lanes of the road."""
    values: List[float] = []
    position = np.asarray(ego.position, dtype=np.float64)
    for lane in info.get("lanes", []):
        try:
            s, _ = lane.local_coordinates(position)
            length = _safe_positive(getattr(lane, "length", 0.0), 0.0)
            if length <= 0.0:
                continue
            values.append(float(np.clip(float(s) / length, 0.0, 1.0)))
        except Exception:  # noqa: BLE001
            continue
    return float(np.mean(values)) if values else 0.0


def _lane_topology(lane: Any, road_network: Any, cache: Dict[Tuple[Any, ...], Tuple[bool, bool]]) -> Tuple[bool, bool]:
    """Return (has_prev, has_next) from geometric lane-endpoint continuity."""
    try:
        index = tuple(lane.index)
    except Exception:  # noqa: BLE001
        return False, False
    if index in cache:
        return cache[index]
    has_prev = False
    has_next = False
    try:
        start_node, end_node = index[0], index[1]
        start = np.asarray(lane.position(0.0, 0.0), dtype=np.float64)[:2]
        end = np.asarray(lane.position(_safe_positive(lane.length, 0.0), 0.0), dtype=np.float64)[:2]
        for pred_node, edges in road_network.graph.items():
            if start_node not in edges:
                continue
            for pred_lane in edges[start_node]:
                try:
                    p = np.asarray(pred_lane.position(pred_lane.length, 0.0), dtype=np.float64)[:2]
                    if float(np.linalg.norm(p - start)) <= 2.0:
                        has_prev = True
                        break
                except Exception:  # noqa: BLE001
                    continue
            if has_prev:
                break
        for next_node in road_network.graph.get(end_node, {}):
            for next_lane in road_network.graph[end_node][next_node]:
                try:
                    p = np.asarray(next_lane.position(0.0, 0.0), dtype=np.float64)[:2]
                    if float(np.linalg.norm(p - end)) <= 2.0:
                        has_next = True
                        break
                except Exception:  # noqa: BLE001
                    continue
            if has_next:
                break
    except Exception:  # noqa: BLE001
        pass
    cache[index] = (has_prev, has_next)
    return has_prev, has_next


def _chunk_anchor(s0: float, lane_length: float) -> float:
    length = max(float(lane_length), 1e-6)
    s = float(np.clip(float(s0), 0.0, length))
    return float(math.floor(s / LD_CHUNK_METERS) * LD_CHUNK_METERS)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _build_ego(
    ego: Any,
    *,
    prev_action: Optional[Sequence[float]],
    prev_speed: Optional[float],
    prev_heading: Optional[float],
    dt: float,
) -> np.ndarray:
    theta = _safe_float(getattr(ego, "heading_theta", 0.0))
    speed = _safe_float(getattr(ego, "speed", 0.0))
    prev_action = np.zeros(4, dtype=np.float64) if prev_action is None else np.asarray(prev_action, dtype=np.float64)
    a_long = 0.0 if prev_speed is None else (speed - _safe_float(prev_speed)) / max(float(dt), 1e-6)
    yaw_rate = 0.0 if prev_heading is None else _wrap_angle(theta - _safe_float(prev_heading)) / max(float(dt), 1e-6)
    a_lat = speed * yaw_rate
    curvature = yaw_rate / max(speed, 0.5)

    v_long, v_lat = speed, 0.0
    try:
        velocity = np.asarray(ego.velocity, dtype=np.float64)[:2]
        if velocity.shape == (2,) and np.isfinite(velocity).all():
            c, s = math.cos(theta), math.sin(theta)
            v_long = float(c * velocity[0] + s * velocity[1])
            v_lat = float(-s * velocity[0] + c * velocity[1])
    except Exception:  # noqa: BLE001
        pass

    length = _safe_positive(getattr(ego, "LENGTH", getattr(ego, "length", 4.5)), 4.5)
    width = _safe_positive(getattr(ego, "WIDTH", getattr(ego, "width", 1.8)), 1.8)
    return np.array([
        speed,
        a_long,
        a_lat,
        yaw_rate,
        _safe_float(getattr(ego, "steering", 0.0)),
        curvature,
        float(prev_action[0]) if len(prev_action) else 0.0,
        float(prev_action[2]) if len(prev_action) > 2 else 0.0,
        v_long,
        v_lat,
        length / EGO_LENGTH_NORM,
        width / EGO_WIDTH_NORM,
    ], dtype=np.float32)


def _build_od(env: Any, ego: Any, scene_seed: int, dt: float) -> Dict[str, np.ndarray]:
    od = np.zeros((OD_SLOTS, OD_DIM), dtype=np.float32)
    od_mask = np.zeros(OD_SLOTS, dtype=np.float32)
    od_ids = np.zeros(OD_SLOTS, dtype=np.int64)
    od_anchor = np.zeros((OD_SLOTS, 3), dtype=np.float32)
    od_taxonomy = np.zeros((OD_SLOTS, 4), dtype=np.int64)
    od_age = np.zeros(OD_SLOTS, dtype=np.float32)

    theta = _safe_float(getattr(ego, "heading_theta", 0.0))
    objects: List[Tuple[float, Any]] = []
    try:
        for obj in env.engine.get_objects().values():
            if obj is ego:
                continue
            if not isinstance(obj, (BaseVehicle, TrafficCone, BaseBuilding)):
                continue
            rel = _to_local(ego, obj.position)
            if not (-20.0 <= rel[0] <= 80.0 and -20.0 <= rel[1] <= 20.0):
                continue
            objects.append((float(np.linalg.norm(rel)), obj))
    except Exception:  # noqa: BLE001
        objects = []
    objects.sort(key=lambda item: item[0])

    for i, (_, obj) in enumerate(objects[:OD_SLOTS]):
        rel = _to_local(ego, obj.position)
        rel_h = _safe_float(getattr(obj, "heading_theta", theta) - theta)
        rel_h = _wrap_angle(rel_h)
        vx, vy = _rel_velocity(ego, obj) if isinstance(obj, BaseVehicle) else (0.0, 0.0)
        acc_x, acc_y = _object_acceleration(ego, obj, _physics_dt(env))
        type_id, static_flag = _type_id(obj)
        length = _safe_positive(getattr(obj, "LENGTH", getattr(obj, "length", 2.0)), 2.0)
        width = _safe_positive(getattr(obj, "WIDTH", getattr(obj, "width", 1.5)), 1.5)
        semantic, mobility, geometry, state = _od_taxonomy(obj, type_id, static_flag)

        od[i] = np.array([
            rel[0], rel[1], math.cos(rel_h), math.sin(rel_h), vx, vy,
            length, width, float(type_id), float(static_flag), 1.0,
            acc_x, acc_y, 0.0,
            float(semantic) / OD_SEMANTIC_NORM,
            float(mobility),
            float(geometry) / OD_GEOMETRY_NORM,
            float(state) / OD_STATE_NORM,
        ], dtype=np.float32)
        od_mask[i] = 1.0
        od_ids[i] = _stable_hash(scene_seed, "od", _obj_stable_key(obj))
        od_anchor[i] = np.array([rel[0], rel[1], rel_h], dtype=np.float32)
        od_taxonomy[i] = np.array([semantic, mobility, geometry, state], dtype=np.int64)
    return {
        "od": od, "od_mask": od_mask, "od_ids": od_ids,
        "od_anchor": od_anchor, "od_taxonomy": od_taxonomy, "od_age": od_age,
    }


def _build_ld(
    env: Any,
    ego: Any,
    scene_seed: int,
    road_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]],
    topo_cache: Dict[Tuple[Any, ...], Tuple[bool, bool]],
) -> Dict[str, np.ndarray]:
    ld = np.zeros((LD_SLOTS, LD_POINTS, LD_DIM), dtype=np.float32)
    ld_mask = np.zeros((LD_SLOTS, LD_POINTS), dtype=np.float32)
    ld_ids = np.zeros(LD_SLOTS, dtype=np.int64)
    ld_road_ids = np.zeros((LD_SLOTS, 2), dtype=np.int64)
    ld_anchor = np.zeros((LD_SLOTS, 3), dtype=np.float32)
    ld_chunk_s = np.zeros(LD_SLOTS, dtype=np.float32)

    try:
        road_network = env.current_map.road_network
    except Exception:  # noqa: BLE001
        road_network = None
    candidates = _lane_candidates(env, ego, max_n=LD_SLOTS)
    for i, (_, lane, s0, role) in enumerate(candidates):
        try:
            lane_index = tuple(lane.index)
            lane_id = int(lane_index[2])
            lane_length = _safe_positive(getattr(lane, "length", 0.0), 0.0)
            if lane_length <= 0.0:
                continue
            s0 = float(np.clip(float(s0), 0.0, lane_length))
            chunk_s = _chunk_anchor(s0, lane_length)
            road_pair = (str(lane_index[0]), str(lane_index[1]))
            road_info = _road_geometry(road_network, road_pair, road_cache) if road_network is not None else None
            lane_count = int(road_info["lane_count"]) if road_info is not None else max(lane_id + 1, 1)
            has_left = lane_id > 0
            has_right = lane_id < max(lane_count - 1, 0)
            has_prev, has_next = (
                _lane_topology(lane, road_network, topo_cache) if road_network is not None else (False, False)
            )
            road_type_norm = float(road_info["type_id"]) / 15.0 if road_info is not None else 0.0

            try:
                line_types = getattr(lane, "line_types", ("ROAD_LINE_UNKNOWN", "ROAD_LINE_UNKNOWN"))
                left_id = _line_type_id(line_types[0] if len(line_types) > 0 else 0)
                right_id = _line_type_id(line_types[1] if len(line_types) > 1 else 0)
            except Exception:  # noqa: BLE001
                left_id = right_id = 0
            speed_limit = _safe_float(getattr(lane, "speed_limit", 13.9), 13.9)
            if not (0.0 < speed_limit < 1000.0):
                speed_limit = 13.9
            try:
                _, lat_offset = lane.local_coordinates(ego.position)
            except Exception:  # noqa: BLE001
                lat_offset = 0.0
            anchor_s_norm = float(np.clip(chunk_s / lane_length, 0.0, 1.0))

            for j, offset in enumerate(LD_POINT_OFFSETS):
                s = float(np.clip(s0 + offset, 0.0, lane_length))
                try:
                    point = np.asarray(lane.position(s, 0.0), dtype=np.float64)[:2]
                    rel = _to_local(ego, point)
                    heading = _wrap_angle(_safe_float(lane.heading_theta_at(s)) - _safe_float(ego.heading_theta))
                    ld[i, j] = np.array([
                        rel[0], rel[1], heading, _lane_curvature(lane, s),
                        speed_limit, left_id, right_id, role, float(lat_offset),
                        1.0 if has_left else 0.0, 1.0 if has_right else 0.0,
                        1.0 if has_prev else 0.0, 1.0 if has_next else 0.0,
                        road_type_norm, 1.0 if road_info is not None and road_info["is_junction"] else 0.0,
                        anchor_s_norm,
                    ], dtype=np.float32)
                    ld_mask[i, j] = 1.0
                except Exception:  # noqa: BLE001
                    pass

            ld_ids[i] = _stable_hash(scene_seed, "ld", lane_index[0], lane_index[1], lane_id)
            ld_road_ids[i] = np.array([
                _road_id(scene_seed, road_pair), lane_id,
            ], dtype=np.int64)
            try:
                anchor_point = np.asarray(lane.position(chunk_s, 0.0), dtype=np.float64)[:2]
                anchor_rel = _to_local(ego, anchor_point)
                anchor_heading = float(
                    _wrap_angle(_safe_float(lane.heading_theta_at(chunk_s)) - _safe_float(ego.heading_theta))
                )
                ld_anchor[i] = np.array([anchor_rel[0], anchor_rel[1], anchor_heading], dtype=np.float32)
            except Exception:  # noqa: BLE001
                pass
            ld_chunk_s[i] = float(chunk_s)
        except Exception:  # noqa: BLE001
            continue
    return {
        "ld": ld, "ld_mask": ld_mask, "ld_ids": ld_ids, "ld_road_ids": ld_road_ids,
        "ld_anchor": ld_anchor, "ld_chunk_s": ld_chunk_s,
    }


def _nav_slots(
    route_pairs: List[Tuple[str, str]],
    current_pair: Optional[Tuple[str, str]],
    next_pair: Optional[Tuple[str, str]],
    final_pair: Optional[Tuple[str, str]],
) -> List[Optional[Tuple[str, str]]]:
    slots: List[Optional[Tuple[str, str]]] = [current_pair]
    if next_pair is not None:
        slots.append(next_pair)
    elif len(route_pairs) > 1:
        slots.append(route_pairs[1])
    if len(route_pairs) > 2:
        slots.append(route_pairs[2])
    if final_pair is not None:
        slots.append(final_pair)
    unique: List[Optional[Tuple[str, str]]] = []
    for pair in slots:
        if pair is not None and pair not in unique:
            unique.append(pair)
    while len(unique) < NAV_ROADS:
        unique.append(None)
    return unique[:NAV_ROADS]


def _build_nav(
    ego: Any,
    scene_seed: int,
    road_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]],
) -> Dict[str, np.ndarray]:
    nav_vec = np.zeros(NAV_DIM, dtype=np.float32)
    nav_mask = np.zeros(1, dtype=np.float32)
    nav_road_mask = np.zeros(NAV_ROADS, dtype=np.float32)
    nav_road_ids = np.zeros((NAV_ROADS, 2), dtype=np.int64)
    nav_route_id = np.int64(0)

    try:
        navigation = ego.navigation
    except Exception:  # noqa: BLE001
        navigation = None
    if navigation is None:
        return {
            "nav": nav_vec, "nav_mask": nav_mask, "nav_road_mask": nav_road_mask,
            "nav_road_ids": nav_road_ids, "nav_route_id": nav_route_id,
        }

    try:
        road_network = ego.engine.current_map.road_network
    except Exception:  # noqa: BLE001
        road_network = None

    try:
        checkpoints = [str(node) for node in list(getattr(navigation, "checkpoints", []) or [])]
    except Exception:  # noqa: BLE001
        checkpoints = []
    pairs = _checkpoint_pairs(checkpoints)
    current_pair = _road_pair(getattr(navigation, "current_road", None))
    next_pair = _road_pair(getattr(navigation, "next_road", None))
    final_pair = _road_pair(getattr(navigation, "final_road", None))
    if current_pair is None and pairs:
        current_pair = pairs[0]
    if next_pair is None and len(pairs) > 1:
        next_pair = pairs[1]
    if final_pair is None and pairs:
        final_pair = pairs[-1]

    current_idx = 0
    if current_pair is not None:
        try:
            current_idx = pairs.index(current_pair)
        except ValueError:
            current_idx = 0
    route_pairs = pairs[current_idx:] if pairs else []
    if current_pair is not None and (not route_pairs or route_pairs[0] != current_pair):
        route_pairs = [current_pair] + [pair for pair in route_pairs if pair != current_pair]
    if not route_pairs:
        route_pairs = [pair for pair in (current_pair, next_pair, final_pair) if pair is not None]

    if checkpoints:
        nav_route_id = np.int64(_stable_hash(scene_seed, "route", checkpoints[0], checkpoints[-1]))

    if road_network is None or not route_pairs:
        return {
            "nav": nav_vec, "nav_mask": nav_mask, "nav_road_mask": nav_road_mask,
            "nav_road_ids": nav_road_ids, "nav_route_id": nav_route_id,
        }

    route_geoms = [_road_geometry(road_network, pair, road_cache) for pair in route_pairs]
    route_lengths = [float(info["length"]) if info is not None else 0.0 for info in route_geoms]
    route_total_from_graph = float(sum(route_lengths))
    total_length = _safe_float(getattr(navigation, "total_length", 0.0), 0.0)
    if total_length <= 0.0:
        total_length = route_total_from_graph
    route_index_map = {pair: index for index, pair in enumerate(route_pairs)}
    slots = _nav_slots(route_pairs, current_pair, next_pair, final_pair)

    ego_theta = _safe_float(getattr(ego, "heading_theta", 0.0))
    current_geom = _road_geometry(road_network, current_pair, road_cache) if current_pair is not None else None
    current_progress = _road_progress(ego, current_geom) if current_geom is not None else 0.0

    for slot, pair in enumerate(slots):
        if pair is None:
            continue
        info = _road_geometry(road_network, pair, road_cache)
        if info is None:
            continue
        entry = _to_local(ego, info["entry"])
        mid = _to_local(ego, info["mid"])
        exit_point = _to_local(ego, info["exit"])
        entry_heading = _wrap_angle(info["entry_heading"] - ego_theta)
        exit_heading = _wrap_angle(info["exit_heading"] - ego_theta)
        route_index = route_index_map.get(pair, -1)
        if route_index >= 0:
            remaining = float(sum(route_lengths[route_index:])) / max(total_length, 1e-6)
        else:
            remaining = 0.0
        progress = current_progress if pair == current_pair else 0.0
        block = np.array([
            entry[0] / 100.0, entry[1] / 100.0,
            mid[0] / 100.0, mid[1] / 100.0,
            exit_point[0] / 100.0, exit_point[1] / 100.0,
            entry_heading, exit_heading,
            info["length"] / 100.0, info["width"] / 10.0, info["lane_count"] / 8.0,
            math.sin(_wrap_angle(info["exit_heading"] - info["entry_heading"])),
            math.cos(_wrap_angle(info["exit_heading"] - info["entry_heading"])),
            info["out_degree"] / 8.0, info["in_degree"] / 8.0,
            info["type_id"] / 15.0,
            1.0 if slot == 0 else 0.0,
            1.0 if slot == 1 else 0.0,
            1.0 if info["is_negative"] else 0.0,
            1.0 if info["is_toll"] else 0.0,
            1.0 if info["is_junction"] else 0.0,
            float(np.clip(remaining, 0.0, 1.0)),
            float(np.clip(progress, 0.0, 1.0)),
            1.0,
        ], dtype=np.float32)
        nav_vec[slot * NAV_ROAD_DIM:(slot + 1) * NAV_ROAD_DIM] = block
        nav_road_mask[slot] = 1.0
        nav_road_ids[slot] = np.array([_road_id(scene_seed, pair), route_index], dtype=np.int64)

    # ---- global route / destination ----
    travelled = _safe_float(getattr(navigation, "travelled_length", 0.0), 0.0)
    completion = _safe_float(getattr(navigation, "route_completion", 0.0), 0.0)
    if total_length > 0.0 and (not (0.0 <= completion <= 1.0) or (completion <= 0.0 and travelled > 0.0)):
        completion = float(np.clip(travelled / total_length, 0.0, 1.0))
    if total_length > 0.0 and travelled <= 0.0 and completion > 0.0:
        travelled = completion * total_length
    remaining_length = max(total_length - travelled, 0.0)
    remaining_count = max(len(route_pairs) - 1, 0)
    junction_ahead = 1.0 if any(info is not None and info["is_junction"] for info in route_geoms) else 0.0

    dest_info = _road_geometry(road_network, final_pair, road_cache) if final_pair is not None else None
    if dest_info is None:
        dest_info = current_geom
    if dest_info is not None:
        dest_rel = _to_local(ego, dest_info["exit"])
        dest_heading = _wrap_angle(dest_info["exit_heading"] - ego_theta)
        dest_dx, dest_dy = dest_rel[0] / 100.0, dest_rel[1] / 100.0
        dest_sin, dest_cos = math.sin(dest_heading), math.cos(dest_heading)
    else:
        dest_dx = dest_dy = dest_sin = 0.0
        dest_cos = 1.0

    nav_vec[NAV_GLOBAL_INDEX["route_completion"]] = float(np.clip(completion, 0.0, 1.0))
    nav_vec[NAV_GLOBAL_INDEX["route_remaining_norm"]] = remaining_length / 100.0
    nav_vec[NAV_GLOBAL_INDEX["route_total_norm"]] = total_length / 100.0
    nav_vec[NAV_GLOBAL_INDEX["current_road_length_norm"]] = (
        float(current_geom["length"]) / 100.0 if current_geom is not None else 0.0
    )
    nav_vec[NAV_GLOBAL_INDEX["current_road_progress"]] = float(np.clip(current_progress, 0.0, 1.0))
    nav_vec[NAV_GLOBAL_INDEX["remaining_road_count_norm"]] = remaining_count / 8.0
    nav_vec[NAV_GLOBAL_INDEX["junction_ahead"]] = junction_ahead
    nav_vec[NAV_GLOBAL_INDEX["dest_dx"]] = dest_dx
    nav_vec[NAV_GLOBAL_INDEX["dest_dy"]] = dest_dy
    nav_vec[NAV_GLOBAL_INDEX["dest_heading_sin"]] = dest_sin
    nav_vec[NAV_GLOBAL_INDEX["dest_heading_cos"]] = dest_cos
    nav_valid = 1.0 if current_geom is not None else 0.0
    nav_vec[NAV_GLOBAL_INDEX["nav_valid"]] = nav_valid
    nav_mask[0] = nav_valid

    return {
        "nav": nav_vec, "nav_mask": nav_mask, "nav_road_mask": nav_road_mask,
        "nav_road_ids": nav_road_ids, "nav_route_id": nav_route_id,
    }


def build_observation(
    env: Any,
    *,
    prev_action: Optional[Sequence[float]] = None,
    prev_speed: Optional[float] = None,
    prev_heading: Optional[float] = None,
    dt: float = DEFAULT_DT,
) -> Dict[str, Any]:
    """Build one current-frame v2 observation from a live MetaDrive env."""
    ego = getattr(env, "agent", None)
    if ego is None or not hasattr(ego, "position"):
        raise ValueError("build_observation requires a reset env with env.agent")
    dt = max(_safe_float(dt, DEFAULT_DT), 1e-6)
    scene_seed = _scene_seed(env)
    road_cache: Dict[Tuple[str, str], Optional[Dict[str, Any]]] = {}
    topo_cache: Dict[Tuple[Any, ...], Tuple[bool, bool]] = {}

    ego_feat = _build_ego(ego, prev_action=prev_action, prev_speed=prev_speed,
                          prev_heading=prev_heading, dt=dt)
    od = _build_od(env, ego, scene_seed, dt)
    ld = _build_ld(env, ego, scene_seed, road_cache, topo_cache)
    nav = _build_nav(ego, scene_seed, road_cache)
    pose = np.asarray(ego_pose(env), dtype=np.float64)

    timestamp, step = _scene_timestamp(env, dt)
    out: Dict[str, Any] = {
        "od": _finite_float(od["od"]),
        "od_mask": _finite_float(od["od_mask"]),
        "od_ids": od["od_ids"].astype(np.int64),
        "od_anchor": _finite_float(od["od_anchor"]),
        "od_taxonomy": od["od_taxonomy"].astype(np.int64),
        "od_age": _finite_float(od["od_age"]),
        "ld": _finite_float(ld["ld"]),
        "ld_mask": _finite_float(ld["ld_mask"]),
        "ld_ids": ld["ld_ids"].astype(np.int64),
        "ld_road_ids": ld["ld_road_ids"].astype(np.int64),
        "ld_anchor": _finite_float(ld["ld_anchor"]),
        "ld_chunk_s": _finite_float(ld["ld_chunk_s"]),
        "nav": _finite_float(nav["nav"]),
        "nav_mask": _finite_float(nav["nav_mask"]),
        "nav_road_mask": _finite_float(nav["nav_road_mask"]),
        "nav_road_ids": nav["nav_road_ids"].astype(np.int64),
        "nav_route_id": np.int64(nav["nav_route_id"]),
        "ego": _finite_float(ego_feat),
        "ego_mask": np.ones(1, dtype=np.float32),
        "pose": _finite_float(pose),
        "meta": {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "source": "REAL",
            "scene_seed": int(scene_seed),
            "timestamp": float(timestamp),
            "step": int(step),
            "dt": float(dt),
        },
    }
    return out


# ---------------------------------------------------------------------------
# Stable-ID temporal alignment with SE(2) geometry compensation
# ---------------------------------------------------------------------------

def _as_pose(value: Any, fallback: np.ndarray) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64).reshape(-1)
        if pose.size >= 3 and np.isfinite(pose[:3]).all():
            return np.array([pose[0], pose[1], _wrap_angle(pose[2])], dtype=np.float64)
    except Exception:  # noqa: BLE001
        pass
    return np.array(fallback, dtype=np.float64)


def _compensate_od_row(
    row: np.ndarray,
    anchor: np.ndarray,
    src_pose: np.ndarray,
    dst_pose: np.ndarray,
    dtheta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    out = np.array(row, dtype=np.float32, copy=True)
    out[0], out[1] = _transform_xy(float(out[0]), float(out[1]), src_pose, dst_pose)
    c, s = _rotate_xy(float(out[OD_INDEX["cos_theta"]]), float(out[OD_INDEX["sin_theta"]]), dtheta)
    out[OD_INDEX["cos_theta"]], out[OD_INDEX["sin_theta"]] = c, s
    vx, vy = _rotate_xy(float(out[OD_INDEX["vx_rel"]]), float(out[OD_INDEX["vy_rel"]]), dtheta)
    out[OD_INDEX["vx_rel"]], out[OD_INDEX["vy_rel"]] = vx, vy
    ax, ay = _rotate_xy(float(out[OD_INDEX["acc_x"]]), float(out[OD_INDEX["acc_y"]]), dtheta)
    out[OD_INDEX["acc_x"]], out[OD_INDEX["acc_y"]] = ax, ay
    anchor_out = np.array(anchor, dtype=np.float32, copy=True)
    anchor_out[0], anchor_out[1] = _transform_xy(float(anchor[0]), float(anchor[1]), src_pose, dst_pose)
    anchor_out[2] = _wrap_angle(float(anchor[2]) + dtheta)
    return out, anchor_out


def _compensate_ld_point(row: np.ndarray, src_pose: np.ndarray, dst_pose: np.ndarray, dtheta: float) -> np.ndarray:
    out = np.array(row, dtype=np.float32, copy=True)
    out[LD_INDEX["dx"]], out[LD_INDEX["dy"]] = _transform_xy(
        float(out[LD_INDEX["dx"]]), float(out[LD_INDEX["dy"]]), src_pose, dst_pose
    )
    out[LD_INDEX["heading_rel"]] = _wrap_angle(float(out[LD_INDEX["heading_rel"]]) + dtheta)
    return out


def _compensate_nav_xy(x: float, y: float, src_pose: np.ndarray, dst_pose: np.ndarray) -> Tuple[float, float]:
    """Compensate a normalized Nav xy pair without mixing metres and /100 units.

    Nav features are stored divided by 100 for scale; pose translation is in
    metres.  Convert to metres for the SE(2) transform, then back.
    """
    mx, my = _transform_xy(float(x) * 100.0, float(y) * 100.0, src_pose, dst_pose)
    return float(mx) / 100.0, float(my) / 100.0


def _compensate_nav_road(block: np.ndarray, src_pose: np.ndarray, dst_pose: np.ndarray, dtheta: float) -> np.ndarray:
    out = np.array(block, dtype=np.float32, copy=True)
    for dx_i, dy_i in ((NAV_ROAD_INDEX["entry_dx"], NAV_ROAD_INDEX["entry_dy"]),
                       (NAV_ROAD_INDEX["mid_dx"], NAV_ROAD_INDEX["mid_dy"]),
                       (NAV_ROAD_INDEX["exit_dx"], NAV_ROAD_INDEX["exit_dy"])):
        out[dx_i], out[dy_i] = _compensate_nav_xy(float(out[dx_i]), float(out[dy_i]), src_pose, dst_pose)
    out[NAV_ROAD_INDEX["entry_heading_rel"]] = float(
        _wrap_angle(float(out[NAV_ROAD_INDEX["entry_heading_rel"]]) + dtheta)
    )
    out[NAV_ROAD_INDEX["exit_heading_rel"]] = float(
        _wrap_angle(float(out[NAV_ROAD_INDEX["exit_heading_rel"]]) + dtheta)
    )
    return out


def _compensate_nav_global(nav_vec: np.ndarray, src_pose: np.ndarray, dst_pose: np.ndarray, dtheta: float) -> None:
    dx_i = NAV_GLOBAL_INDEX["dest_dx"]
    dy_i = NAV_GLOBAL_INDEX["dest_dy"]
    nav_vec[dx_i], nav_vec[dy_i] = _compensate_nav_xy(float(nav_vec[dx_i]), float(nav_vec[dy_i]), src_pose, dst_pose)
    sin_i = NAV_GLOBAL_INDEX["dest_heading_sin"]
    cos_i = NAV_GLOBAL_INDEX["dest_heading_cos"]
    cos_new, sin_new = _rotate_xy(float(nav_vec[cos_i]), float(nav_vec[sin_i]), dtheta)
    nav_vec[cos_i], nav_vec[sin_i] = cos_new, sin_new


def _empty_history() -> Dict[str, np.ndarray]:
    return {
        "od": np.zeros((HISTORY_FRAMES, OD_SLOTS, OD_DIM), dtype=np.float32),
        "od_mask": np.zeros((HISTORY_FRAMES, OD_SLOTS), dtype=np.float32),
        "od_ids": np.zeros((HISTORY_FRAMES, OD_SLOTS), dtype=np.int64),
        "od_anchor": np.zeros((HISTORY_FRAMES, OD_SLOTS, 3), dtype=np.float32),
        "od_taxonomy": np.zeros((HISTORY_FRAMES, OD_SLOTS, 4), dtype=np.int64),
        "od_age": np.zeros((HISTORY_FRAMES, OD_SLOTS), dtype=np.float32),
        "ld": np.zeros((HISTORY_FRAMES, LD_SLOTS, LD_POINTS, LD_DIM), dtype=np.float32),
        "ld_mask": np.zeros((HISTORY_FRAMES, LD_SLOTS, LD_POINTS), dtype=np.float32),
        "ld_ids": np.zeros((HISTORY_FRAMES, LD_SLOTS), dtype=np.int64),
        "ld_road_ids": np.zeros((HISTORY_FRAMES, LD_SLOTS, 2), dtype=np.int64),
        "ld_anchor": np.zeros((HISTORY_FRAMES, LD_SLOTS, 3), dtype=np.float32),
        "ld_chunk_s": np.zeros((HISTORY_FRAMES, LD_SLOTS), dtype=np.float32),
        "nav": np.zeros((HISTORY_FRAMES, NAV_DIM), dtype=np.float32),
        "nav_mask": np.zeros((HISTORY_FRAMES, 1), dtype=np.float32),
        "nav_road_mask": np.zeros((HISTORY_FRAMES, NAV_ROADS), dtype=np.float32),
        "nav_road_ids": np.zeros((HISTORY_FRAMES, NAV_ROADS, 2), dtype=np.int64),
        "nav_route_id": np.zeros(HISTORY_FRAMES, dtype=np.int64),
        "ego": np.zeros((HISTORY_FRAMES, EGO_DIM), dtype=np.float32),
        "ego_mask": np.zeros((HISTORY_FRAMES, 1), dtype=np.float32),
        "pose": np.zeros((HISTORY_FRAMES, 3), dtype=np.float32),
        "hist_valid": np.zeros(HISTORY_FRAMES, dtype=np.float32),
    }


def stack_history(frames: Sequence[Dict[str, Any]], poses: Sequence[Any]) -> Dict[str, Any]:
    """Stack up to ``HISTORY_FRAMES`` v2 frames, oldest first.

    Alignment is by stable ID into the **current frame's slot order**, not by
    distance rank.  Geometry and geometry anchors are SE(2)-compensated from
    each historical ego frame into the current ego frame.  IDs / taxonomy /
    masks are not rotated.  A historical entity missing by ID gets mask=0 and
    zero geometry; warm-up padding is zero with ``hist_valid=0``.

    Metadata ID tensors (``od_ids`` / ``ld_ids`` / ``ld_road_ids`` /
    ``nav_road_ids``) are filled with the *current* slot IDs for every history
    frame so that a history tensor can be indexed with the current slot order;
    the corresponding mask says whether that entity exists in that frame.

    Nav is reordered by ``nav_road_ids[:, 0]`` (road ID) into the current nav
    slot order; slot 0 is the current frame's current road.  A route-ID change
    (which should only occur on reset) invalidates the historical nav.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("stack_history: no frames")
    n = min(len(frames), HISTORY_FRAMES)
    frames = frames[-n:]
    if poses is None:
        poses = [frame.get("pose") for frame in frames]
    poses = list(poses)[-n:]
    if len(poses) != n:
        raise ValueError("stack_history: frames and poses length mismatch")
    for key in ("od_ids", "ld_ids", "nav_road_ids"):
        if key not in frames[-1]:
            raise ValueError(f"stack_history requires v2 frames with {key!r}")

    current = frames[-1]
    dst_pose_raw = poses[-1]
    dst_pose = _as_pose(dst_pose_raw, np.asarray(current.get("pose", [0.0, 0.0, 0.0])))
    current_meta = current.get("meta", {}) if isinstance(current.get("meta"), dict) else {}
    dt = max(_safe_float(current_meta.get("dt", DEFAULT_DT), DEFAULT_DT), 1e-6)
    scene_seed = int(current_meta.get("scene_seed", 0) or 0)

    cur_od_ids = np.asarray(current["od_ids"], dtype=np.int64)
    cur_od_mask = np.asarray(current["od_mask"], dtype=np.float32)
    cur_ld_ids = np.asarray(current["ld_ids"], dtype=np.int64)
    cur_ld_mask = np.asarray(current["ld_mask"], dtype=np.float32)
    cur_ld_road_ids = np.asarray(current["ld_road_ids"], dtype=np.int64)
    cur_nav_road_ids = np.asarray(current["nav_road_ids"], dtype=np.int64)
    cur_nav_road_mask = np.asarray(current["nav_road_mask"], dtype=np.float32)
    cur_nav_mask = np.asarray(current["nav_mask"], dtype=np.float32)
    cur_route_id = int(np.asarray(current["nav_route_id"]).reshape(-1)[0])

    out = _empty_history()
    pad = HISTORY_FRAMES - n

    for t in range(HISTORY_FRAMES):
        if t < pad:
            continue
        src = t - pad
        frame = frames[src]
        src_pose = _as_pose(poses[src], np.asarray(frame.get("pose", [0.0, 0.0, 0.0])))
        dtheta = _wrap_angle(float(src_pose[2]) - float(dst_pose[2]))

        # ---- Ego (features are already ego-relative) ----
        out["ego"][t] = np.asarray(frame["ego"], dtype=np.float32)
        out["ego_mask"][t] = np.asarray(frame["ego_mask"], dtype=np.float32)
        out["pose"][t] = src_pose.astype(np.float32)
        out["hist_valid"][t] = 1.0

        # ---- OD ----
        src_od_ids = np.asarray(frame["od_ids"], dtype=np.int64)
        src_od_mask = np.asarray(frame["od_mask"], dtype=np.float32)
        src_od = np.asarray(frame["od"], dtype=np.float32)
        src_od_anchor = np.asarray(frame["od_anchor"], dtype=np.float32)
        src_od_taxonomy = np.asarray(frame["od_taxonomy"], dtype=np.int64)
        src_od_age = np.asarray(frame["od_age"], dtype=np.float32)
        od_map = {int(src_od_ids[i]): i for i in range(OD_SLOTS) if src_od_mask[i] > 0.5}
        out["od_ids"][t] = cur_od_ids
        for j in range(OD_SLOTS):
            if cur_od_mask[j] <= 0.5:
                continue
            src_i = od_map.get(int(cur_od_ids[j]))
            if src_i is None:
                continue
            row, anchor = _compensate_od_row(src_od[src_i], src_od_anchor[src_i], src_pose, dst_pose, dtheta)
            out["od"][t, j] = row
            out["od_mask"][t, j] = src_od_mask[src_i]
            out["od_anchor"][t, j] = anchor
            out["od_taxonomy"][t, j] = src_od_taxonomy[src_i]
            out["od_age"][t, j] = src_od_age[src_i]

        # ---- LD ----
        src_ld_ids = np.asarray(frame["ld_ids"], dtype=np.int64)
        src_ld_mask = np.asarray(frame["ld_mask"], dtype=np.float32)
        src_ld = np.asarray(frame["ld"], dtype=np.float32)
        src_ld_anchor = np.asarray(frame["ld_anchor"], dtype=np.float32)
        src_ld_chunk_s = np.asarray(frame["ld_chunk_s"], dtype=np.float32)
        ld_map = {int(src_ld_ids[i]): i for i in range(LD_SLOTS) if src_ld_mask[i].any()}
        out["ld_ids"][t] = cur_ld_ids
        out["ld_road_ids"][t] = cur_ld_road_ids
        for j in range(LD_SLOTS):
            if not cur_ld_mask[j].any():
                continue
            src_i = ld_map.get(int(cur_ld_ids[j]))
            if src_i is None:
                continue
            for k in range(LD_POINTS):
                if src_ld_mask[src_i, k] <= 0.5:
                    continue
                out["ld"][t, j, k] = _compensate_ld_point(src_ld[src_i, k], src_pose, dst_pose, dtheta)
                out["ld_mask"][t, j, k] = 1.0
            anchor = np.array(src_ld_anchor[src_i], dtype=np.float32, copy=True)
            anchor[0], anchor[1] = _transform_xy(float(anchor[0]), float(anchor[1]), src_pose, dst_pose)
            anchor[2] = _wrap_angle(float(anchor[2]) + dtheta)
            out["ld_anchor"][t, j] = anchor
            out["ld_chunk_s"][t, j] = src_ld_chunk_s[src_i]

        # ---- Nav ----
        src_nav_mask = np.asarray(frame["nav_mask"], dtype=np.float32)
        src_nav_road_mask = np.asarray(frame["nav_road_mask"], dtype=np.float32)
        src_nav_road_ids = np.asarray(frame["nav_road_ids"], dtype=np.int64)
        src_nav = np.asarray(frame["nav"], dtype=np.float32)
        src_route_id = int(np.asarray(frame["nav_route_id"]).reshape(-1)[0])
        out["nav_road_ids"][t] = cur_nav_road_ids
        out["nav_route_id"][t] = np.int64(src_route_id if src_nav_mask[0] > 0.5 else 0)
        route_match = (
            src_nav_mask[0] > 0.5
            and cur_nav_mask[0] > 0.5
            and src_route_id == cur_route_id
        )
        if not route_match:
            continue
        src_road_map = {
            int(src_nav_road_ids[i, 0]): i
            for i in range(NAV_ROADS)
            if src_nav_road_mask[i] > 0.5
        }
        for j in range(NAV_ROADS):
            if cur_nav_road_mask[j] <= 0.5:
                continue
            src_i = src_road_map.get(int(cur_nav_road_ids[j, 0]))
            if src_i is None:
                continue
            block = _compensate_nav_road(
                src_nav[src_i * NAV_ROAD_DIM:(src_i + 1) * NAV_ROAD_DIM], src_pose, dst_pose, dtheta
            )
            # Slot-role flags describe the destination/current slot order.
            block[NAV_ROAD_INDEX["is_current"]] = 1.0 if j == 0 else 0.0
            block[NAV_ROAD_INDEX["is_next"]] = 1.0 if j == 1 else 0.0
            out["nav"][t, j * NAV_ROAD_DIM:(j + 1) * NAV_ROAD_DIM] = block
            out["nav_road_mask"][t, j] = 1.0
        out["nav"][t, NAV_ROADS * NAV_ROAD_DIM:] = src_nav[NAV_ROADS * NAV_ROAD_DIM:]
        _compensate_nav_global(out["nav"][t], src_pose, dst_pose, dtheta)
        out["nav"][t, NAV_GLOBAL_INDEX["nav_valid"]] = 1.0
        out["nav_mask"][t, 0] = 1.0

    # ---- OD observation age (seconds tracked continuously in this window) ----
    for t in range(HISTORY_FRAMES):
        if out["hist_valid"][t] <= 0.5:
            continue
        if t > 0 and out["hist_valid"][t - 1] > 0.5:
            same = (out["od_mask"][t] > 0.5) & (out["od_ids"][t] == out["od_ids"][t - 1])
            ages = np.where(same, out["od_age"][t - 1] + dt, 0.0).astype(np.float32)
        else:
            ages = np.zeros(OD_SLOTS, dtype=np.float32)
        out["od_age"][t] = ages
        out["od"][t, :, OD_AGE_INDEX] = np.clip(ages / OD_AGE_NORM_S, 0.0, 1.0)

    # ---- finite guarantee ----
    for key in ("od", "od_mask", "od_anchor", "od_age", "ld", "ld_mask", "ld_anchor",
                "ld_chunk_s", "nav", "nav_mask", "nav_road_mask", "ego", "ego_mask",
                "pose", "hist_valid"):
        out[key] = _finite_float(np.asarray(out[key]))

    timestamps = np.zeros(HISTORY_FRAMES, dtype=np.float32)
    sources: List[str] = []
    for t in range(HISTORY_FRAMES):
        if out["hist_valid"][t] <= 0.5:
            sources.append("PAD")
            continue
        meta = frames[t - pad].get("meta", {})
        meta = meta if isinstance(meta, dict) else {}
        timestamps[t] = _safe_float(meta.get("timestamp", float(t - pad) * dt), 0.0)
        sources.append(str(meta.get("source", "REAL")))

    out["meta"] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "scene_seed": int(scene_seed),
        "dt": float(dt),
        "timestamps": timestamps,
        "sources": sources,
        "source": "REAL",
    }
    return out
