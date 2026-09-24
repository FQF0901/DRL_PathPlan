#!/usr/bin/env python3
"""场景俯视图抽检工具（matplotlib Agg，无 GL/3D 渲染）。

用途
----
把 spec 里的随机场景跑成一个真实 episode（规则专家驱动自车），再以世界系俯视图出图，
供人工目视核查：车道/线型/限速、自车、交通车、导航 route、脚本事件与逐步标签。

用法::

    tools/venv-python tools/visualize.py --specs env/specs/scenarios_train_slice200.json \\
        --random 10 --seed 0 --frames 3 --steps 120 --policy baseline --out runs/vis

选择方式（三选一）：``--random N``（``--seed`` 控制抽样，可复现）/ ``--ids 1,5,9`` /
``--all-in-file``。输出：每场景一张 1×N 帧拼图 ``scenario_<id>_<seed>.png``、
全体首帧总览 ``contact_sheet.png``、文字索引 ``index.md``。

关键实现事实（已对照 ``.venv/lib/python3.10/site-packages/metadrive/`` 源码）
----------------------------------------------------------------------------
- 车道取 ``current_map.road_network.get_all_lanes()``（node_road_network.py:325）；
  ``lane.line_types`` / ``lane.line_colors`` 为 ``[左侧, 右侧]`` 两元组
  （pg_map.py:141-149 对 side=0 取 ``-width/2``），线型是 ``MetaDriveType`` 字符串
  （``ROAD_LINE_BROKEN_SINGLE_WHITE`` / ``ROAD_LINE_SOLID_SINGLE_WHITE`` /
  ``ROAD_EDGE_BOUNDARY`` …，type.py:25-36）；断裂线画法沿用 ``LaneGraphics``
  的段长 3 m / 间距 5 m（top_down_obs_impl.py:269-270）。
- 自车/交通车姿态：``BaseVehicle.heading_theta`` 是**前进方向**世界航向
  （base_vehicle.py:1032-1033 覆写 +π/2），``position`` 为世界坐标，``LENGTH``/``WIDTH``
  属性可用；``engine.get_objects()`` 返回 id→对象（base_engine.py:219）。
- 驱动自车：``engine.add_policy(ego.id, PurePursuitIDMPolicy, ego, seed)`` 注册为动作来源
  （base_engine.py:98-101），随后 ``env.step([0.0, 0.0])`` 即由策略驾驶
  （与 tools/baseline_eval.py 同口径）。``--policy idle`` 时不注册策略。
- 导航 checkpoint 为世界坐标（base_navigation.py:153-160）。
- 图中文字一律英文（CJK 字体可能缺失）。

失败隔离：单场景任何异常都被捕获、打印并跳过，不影响其余场景；错误汇总进 ``index.md``。
"""

from __future__ import annotations

import argparse
import random
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 必须在 pyplot 之前：headless、不依赖 GL/DISPLAY

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 允许 `python tools/visualize.py` 直接运行
    sys.path.insert(0, str(ROOT))

from env.expert.pure_pursuit_idm import PurePursuitIDMPolicy  # noqa: E402
from env.metadrive_env import build_env  # noqa: E402
from env.scenario.labels import compute_step_labels  # noqa: E402
from env.scenario.spec import load_specs  # noqa: E402

try:  # 仅用于区分交通车/静态物（缺失时退化为按属性判定）
    from metadrive.component.vehicle.base_vehicle import BaseVehicle  # noqa: E402
except Exception:  # noqa: BLE001
    BaseVehicle = ()  # type: ignore[assignment]

# ======================================================================================
# 绘图常量
# ======================================================================================
LANE_SAMPLE_INTERVAL = 2.0  # 车道中心线/边界采样间隔（m），与 MetaDrive get_polyline 一致
STRIPE_LENGTH = 3.0  # 断裂线段长（m），LaneGraphics.STRIPE_LENGTH
STRIPE_SPACING = 5.0  # 断裂线间距（m），LaneGraphics.STRIPE_SPACING
UNSET_SPEED_LIMIT = 1000.0  # MetaDrive "未设置限速" 哨兵（abs_lane.py:22）

EGO_COLOR = "#D62728"
EGO_EDGE = "#7F1416"
TRAFFIC_COLOR = "#3B7DD8"
TRAFFIC_EDGE = "#1F4E8C"
STATIC_COLOR = "#9A9A9A"
STATIC_EDGE = "#6E6E6E"
ROUTE_COLOR = "#E07B00"
CENTERLINE_COLOR = "#BDBDBD"
YELLOW_LINE_COLOR = "#E0A51E"  # 黄线：白底上加深，语义仍是黄
GREY_LINE_COLOR = "#8A8A8A"  # PGLineColor.GREY=(1,1,1,1) 在白底不可见，映射为灰

_SKIP_LINE_TYPES = frozenset({None, "", "UNKNOWN_LINE", "UNKNOWN"})


# ======================================================================================
# spec 文本
# ======================================================================================
def _traffic_dict(spec: Any) -> dict:
    traffic = getattr(spec, "traffic", None)
    return traffic if isinstance(traffic, dict) else {}


def _events_brief(spec: Any) -> str:
    """脚本事件紧凑摘要：``cut_out/right@55+45 gap=24.4m v=6.6m/s``。"""
    parts = []
    for event in _traffic_dict(spec).get("events") or []:
        if not isinstance(event, dict):
            continue
        try:
            gap = float(event.get("gap_m", float("nan")))
            speed = float(event.get("speed_mps", float("nan")))
            parts.append(
                f"{event.get('type')}/{event.get('side')}@{event.get('trigger_step')}"
                f"+{event.get('duration_steps')} gap={gap:.1f}m v={speed:.1f}m/s"
            )
        except (TypeError, ValueError):
            parts.append(str(event))
    return "; ".join(parts)


def _formation(spec: Any) -> str:
    """场景文本：一行 spec 概要 + 一行交通 + 一行导航。"""
    traffic = _traffic_dict(spec)
    events = _events_brief(spec) or "-"
    patterns = ",".join(str(p) for p in (traffic.get("patterns") or [])) or "-"
    try:
        density = f"{float(traffic.get('density', 0.0)):.3f}"
    except (TypeError, ValueError):
        density = "?"
    nav = getattr(spec, "nav", None) if isinstance(getattr(spec, "nav", None), dict) else {}
    lines = [
        f"spec id={spec.id}  seed={spec.seed}  split={spec.split}  blocks={spec.blocks}  "
        f"geometry={'/'.join(str(g) for g in spec.geometry)}  difficulty={spec.difficulty}",
        f"traffic: density={density}  patterns={patterns}  "
        f"random_traffic={bool(traffic.get('random_traffic', False))}  events: {events}",
        f"nav: turns={nav.get('turns', '-')}  junction_blocks={nav.get('junction_blocks', '-')}",
    ]
    return "\n".join(lines)


def _annotation_line(spec: Any) -> str:
    """index.md 用的单行注释。"""
    traffic = _traffic_dict(spec)
    try:
        density = f"{float(traffic.get('density', 0.0)):.3f}"
    except (TypeError, ValueError):
        density = "?"
    patterns = ",".join(str(p) for p in (traffic.get("patterns") or [])) or "-"
    return (
        f"id={spec.id} seed={spec.seed} split={spec.split} blocks={spec.blocks} "
        f"geometry={'/'.join(str(g) for g in spec.geometry)} difficulty={spec.difficulty} "
        f"density={density} patterns={patterns} events={_events_brief(spec) or '-'}"
    )


def _format_labels(labels: Optional[dict], width: int = 78) -> str:
    """逐步标签紧凑单行（必要时折行）。"""
    if not labels:
        return "labels: n/a"
    parts = []
    for key, value in labels.items():
        try:
            number = float(value)
            parts.append(f"{key}={int(number)}" if number.is_integer() else f"{key}={number:.2f}")
        except (TypeError, ValueError):
            parts.append(f"{key}={value}")
    wrapped = textwrap.wrap("labels: " + " ".join(parts), width=width)
    return "\n".join(wrapped) if wrapped else "labels: n/a"


# ======================================================================================
# 场景数据提取（车道 + 每帧快照）
# ======================================================================================
def _sample_polyline(lane: Any, lateral_fn) -> np.ndarray:
    """沿车道弧长采样一条折线（世界坐标 Nx2）；``position`` 返回世界坐标。"""
    length = float(lane.length)
    samples = list(np.arange(0.0, length, LANE_SAMPLE_INTERVAL))
    if not samples or samples[-1] < length:
        samples.append(length)
    points = []
    for s in samples:
        xy = np.asarray(lane.position(float(s), float(lateral_fn(float(s)))), dtype=float).reshape(-1)
        points.append((float(xy[0]), float(xy[1])))
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _line_color(line_colors: Any, side: int) -> str:
    """把 ``lane.line_colors[side]``（RGBA，0-1 浮点）映射成白底可读的绘图色。"""
    try:
        rgba = np.asarray(line_colors[side], dtype=float).reshape(-1)
    except Exception:  # noqa: BLE001
        return GREY_LINE_COLOR
    if rgba.size < 3 or not np.all(np.isfinite(rgba[:3])):
        return GREY_LINE_COLOR
    red, green, blue = (float(c) for c in rgba[:3])
    if red > 1.0 or green > 1.0 or blue > 1.0:  # 兼容 0-255 表示
        red, green, blue = red / 255.0, green / 255.0, blue / 255.0
    if red > 0.7 and 0.45 < green < 0.95 and blue < 0.3:  # PGLineColor.YELLOW
        return YELLOW_LINE_COLOR
    if red > 0.85 and green > 0.85 and blue > 0.85:  # PGLineColor.GREY（白）不可见
        return GREY_LINE_COLOR
    return (red, green, blue)


def _extract_lanes(env: Any) -> tuple[list[dict], Optional[tuple[float, float, float, float]]]:
    """一次性提取所有车道的中心线 + 左右边界（地图在 episode 内静态）。

    返回 ``(lanes, map_bbox)``；``map_bbox = (min_x, max_x, min_y, max_y)`` 由采样点得到。
    """
    lanes = []
    seen: set[int] = set()
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    try:
        all_lanes = list(env.current_map.road_network.get_all_lanes())
    except Exception:  # noqa: BLE001 - 无地图时返回空，绘图退化为只画车辆
        return lanes, None
    for lane in all_lanes:
        if id(lane) in seen:
            continue
        seen.add(id(lane))
        try:
            length = float(lane.length)
            width = float(lane.width)
        except (AttributeError, TypeError, ValueError):
            continue
        if not (np.isfinite(length) and np.isfinite(width) and length > 0.0 and width > 0.0):
            continue
        try:
            center = _sample_polyline(lane, lambda _s: 0.0)
            sides = []
            line_types = list(getattr(lane, "line_types", ()) or ())
            line_colors = list(getattr(lane, "line_colors", ()) or ())
            for side in (0, 1):
                line_type = str(line_types[side]) if side < len(line_types) else None
                color = _line_color(line_colors, side)
                lateral_sign = -0.5 if side == 0 else 0.5  # side 0 = 左（pg_map.py:145-149）
                boundary = _sample_polyline(
                    lane, lambda s, sign=lateral_sign: sign * float(lane.width_at(s))
                )
                sides.append((line_type, color, boundary))
            limit = float(getattr(lane, "speed_limit", UNSET_SPEED_LIMIT))
        except Exception:  # noqa: BLE001 - 单条车道异常不影响其它车道
            continue
        lanes.append(
            {
                "index": str(tuple(lane.index)) if getattr(lane, "index", None) is not None else "<lane>",
                "center": center,
                "sides": sides,
                "limit": limit,
            }
        )
        for poly in [center] + [side[2] for side in sides if side[2] is not None]:
            if len(poly):
                min_x, max_x = min(min_x, float(poly[:, 0].min())), max(max_x, float(poly[:, 0].max()))
                min_y, max_y = min(min_y, float(poly[:, 1].min())), max(max_y, float(poly[:, 1].max()))
    bbox = None if min_x == float("inf") else (min_x, max_x, min_y, max_y)
    return lanes, bbox


def _broken_line_segments(poly: np.ndarray) -> list[np.ndarray]:
    """按弧长把一条边界折线切成断裂线段（段长 3 m / 间距 5 m，同 LaneGraphics）。"""
    if len(poly) < 2:
        return []
    distances = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = float(cumulative[-1])
    segments = []
    for start in np.arange(0.0, total, STRIPE_SPACING):
        end = min(start + STRIPE_LENGTH, total)
        if end - start < 0.4:
            continue
        xs = np.interp([start, end], cumulative, poly[:, 0])
        ys = np.interp([start, end], cumulative, poly[:, 1])
        segments.append(np.column_stack([xs, ys]))
    return segments


def _lane_artists(lane_data: Sequence[dict]) -> dict:
    """把车道数据预处理成按颜色分组的 LineCollection 素材（每场景算一次，各帧复用）。"""
    centers: list[np.ndarray] = []
    solid: dict[str, list[np.ndarray]] = {}
    broken: dict[str, list[np.ndarray]] = {}
    for lane in lane_data:
        centers.append(lane["center"])
        for line_type, color, boundary in lane["sides"]:
            if line_type in _SKIP_LINE_TYPES or boundary is None or len(boundary) < 2:
                continue
            if "BROKEN" in line_type:
                broken.setdefault(color, []).extend(_broken_line_segments(boundary))
            else:
                solid.setdefault(color, []).append(boundary)
    return {"centers": centers, "solid": solid, "broken": broken}


def _add_lane_artists(ax: Any, artists: dict) -> None:
    if artists["centers"]:
        ax.add_collection(
            LineCollection(
                artists["centers"], colors=CENTERLINE_COLOR, linewidths=0.5,
                linestyles=":", alpha=0.9, zorder=1,
            )
        )
    for color, segments in artists["solid"].items():
        ax.add_collection(
            LineCollection(segments, colors=color, linewidths=1.0, linestyles="solid", zorder=2)
        )
    for color, segments in artists["broken"].items():
        if segments:
            ax.add_collection(
                LineCollection(segments, colors=color, linewidths=1.0, linestyles="solid", zorder=2)
            )


def _box_corners(x: float, y: float, heading: float, length: float, width: float) -> np.ndarray:
    """车辆/物体矩形四角（世界坐标）；``heading`` 为前进方向世界航向。"""
    cos_h, sin_h = float(np.cos(heading)), float(np.sin(heading))
    forward = np.array([cos_h, sin_h])
    left = np.array([-sin_h, cos_h])
    center = np.array([x, y])
    corners = []
    for along in (-0.5 * length, 0.5 * length):
        for lateral in (-0.5 * width, 0.5 * width):
            corners.append(center + along * forward + lateral * left)
    return np.asarray(corners)  # [(-L,-W), (-L,+W), (+L,+W), (+L,-W)] 逆时针自洽


def _snapshot(env: Any, spec: Any, step: int, dt: float, status: str) -> dict:
    """抓取一帧的纯数据快照（世界对象随后会移动，必须先复制成 float/ndarray）。"""
    ego = env.vehicle
    ego_position = np.asarray(ego.position, dtype=float).reshape(-1)[:2]

    limit_text, limit_xy = None, None
    lane = getattr(ego, "lane", None)
    if lane is not None:
        try:
            limit = float(getattr(lane, "speed_limit", UNSET_SPEED_LIMIT))
            limit_text = f"ego lane limit {limit:.1f} m/s" if limit < UNSET_SPEED_LIMIT else "ego lane limit unset"
            longitudinal, _ = lane.local_coordinates(ego.position)
            s_ann = float(np.clip(float(longitudinal) + 6.0, 0.0, float(lane.length)))
            xy = np.asarray(lane.position(s_ann, 0.0), dtype=float).reshape(-1)
            limit_xy = (float(xy[0]), float(xy[1]))
        except Exception:  # noqa: BLE001
            limit_text, limit_xy = None, None

    traffic = []
    try:
        objects = list(env.engine.get_objects().values())
    except Exception:  # noqa: BLE001
        objects = []
    for obj in objects:
        if obj is ego:
            continue
        try:
            position = np.asarray(obj.position, dtype=float).reshape(-1)[:2]
            heading = float(obj.heading_theta)
            length = float(obj.LENGTH)
            width = float(obj.WIDTH)
        except Exception:  # noqa: BLE001 - 无尺寸属性的对象跳过
            continue
        if not (
            np.all(np.isfinite(position))
            and np.isfinite(heading)
            and np.isfinite(length)
            and np.isfinite(width)
            and length > 0.0
            and width > 0.0
        ):
            continue
        try:
            speed = float(getattr(obj, "speed", float("nan")))
        except Exception:  # noqa: BLE001
            speed = float("nan")
        traffic.append(
            {
                "name": str(getattr(obj, "name", "?")),
                "class_name": str(getattr(obj, "class_name", type(obj).__name__)),
                "x": float(position[0]),
                "y": float(position[1]),
                "heading": heading,
                "length": length,
                "width": width,
                "speed": speed,
                "is_vehicle": isinstance(obj, BaseVehicle),
            }
        )

    checkpoints = []
    try:
        for checkpoint in env.vehicle.navigation.get_checkpoints():
            xy = np.asarray(checkpoint, dtype=float).reshape(-1)
            checkpoints.append((float(xy[0]), float(xy[1])))
    except Exception:  # noqa: BLE001 - 无导航模块时省略 route
        checkpoints = []

    try:
        labels = compute_step_labels(env, spec)
    except Exception:  # noqa: BLE001 - 标签失败不应影响出图
        labels = None

    return {
        "step": int(step),
        "time_s": float(step) * float(dt),
        "status": status,
        "ego": {
            "x": float(ego_position[0]),
            "y": float(ego_position[1]),
            "heading": float(ego.heading_theta),
            "length": float(ego.LENGTH),
            "width": float(ego.WIDTH),
            "speed": float(getattr(ego, "speed", float("nan"))),
            "lane_index": str(getattr(ego, "lane_index", "?")),
        },
        "traffic": traffic,
        "checkpoints": checkpoints,
        "labels": labels,
        "limit_text": limit_text,
        "limit_xy": limit_xy,
    }


def _frame_steps(steps: int, frames: int) -> list[int]:
    """请求的帧步号：默认 t=0 / mid / end；去重并保序。"""
    if steps <= 0 or frames <= 1:
        return [0]
    requested = {int(round(index * steps / (frames - 1))) for index in range(frames)}
    requested.update((0, steps))
    return sorted(step for step in requested if 0 <= step <= steps)


def _data_span(snapshots: Sequence[dict], near_ego_m: float = 80.0) -> tuple[float, float, float, float]:
    """数据包围盒（自车轨迹 + route + 附近交通车），返回 ``(span_x, span_y, center_x, center_y)``。"""
    ego_points = [(snap["ego"]["x"], snap["ego"]["y"]) for snap in snapshots]
    xs = [point[0] for point in ego_points]
    ys = [point[1] for point in ego_points]
    for snap in snapshots:
        for checkpoint in snap["checkpoints"]:
            xs.append(checkpoint[0])
            ys.append(checkpoint[1])
        for item in snap["traffic"]:
            if not item["is_vehicle"]:
                continue
            if min((item["x"] - px) ** 2 + (item["y"] - py) ** 2 for px, py in ego_points) > near_ego_m**2:
                continue
            xs.append(item["x"])
            ys.append(item["y"])
    if not xs:
        return 60.0, 60.0, 0.0, 0.0
    margin = 25.0
    return (
        max(max(xs) - min(xs), 30.0) + margin,
        max(max(ys) - min(ys), 30.0) + margin,
        0.5 * (min(xs) + max(xs)),
        0.5 * (min(ys) + max(ys)),
    )


def _raw_spans(
    snapshots: Sequence[dict],
    map_bbox: Optional[tuple[float, float, float, float]],
    near_ego_m: float = 80.0,
) -> tuple[float, float, float, float]:
    """视图原始跨度与中心：优先整张地图（场景全貌），地图过大时退化为自车局部。"""
    data_x, data_y, data_cx, data_cy = _data_span(snapshots, near_ego_m)
    if map_bbox is not None:
        min_x, max_x, min_y, max_y = map_bbox
        map_x = (max_x - min_x) + 6.0
        map_y = (max_y - min_y) + 6.0
        if map_x <= 300.0 and map_y <= 300.0:
            return (
                max(map_x, data_x),
                max(map_y, data_y),
                0.5 * (min_x + max_x),
                0.5 * (min_y + max_y),
            )
    return data_x, data_y, data_cx, data_cy


def _view_bounds(
    snapshots: Sequence[dict],
    map_bbox: Optional[tuple[float, float, float, float]],
    *,
    aspect: float = 1.0,
    near_ego_m: float = 80.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """围绕自车 / route / 场景地图的视野；``aspect``=面板宽高比（equal aspect 下不浪费空白）。

    地图（车道采样包围盒）不大时展示场景全貌（含前方路口/环岛），过大时只取自车轨迹附近。
    """
    span_x, span_y, center_x, center_y = _raw_spans(snapshots, map_bbox, near_ego_m)
    if aspect > 0.0:  # 把短边撑到与面板长宽比一致（仍保证包含全部数据点）
        if span_x / span_y < aspect:
            span_x = span_y * aspect
        else:
            span_y = span_x / aspect
    span_x, span_y = min(span_x, 320.0), min(span_y, 320.0)
    return (center_x - 0.5 * span_x, center_x + 0.5 * span_x), (center_y - 0.5 * span_y, center_y + 0.5 * span_y)


# ======================================================================================
# 绘图
# ======================================================================================
def _draw_scene(
    ax: Any,
    spec: Any,
    lane_artists: dict,
    snap: dict,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    title: Optional[str] = None,
    legend: bool = False,
    show_traffic_speed: bool = True,
) -> None:
    """在一个 axes 上画一帧世界系俯视图。"""
    _add_lane_artists(ax, lane_artists)

    # ---- route（世界坐标）----
    if snap["checkpoints"]:
        route = np.asarray([(snap["ego"]["x"], snap["ego"]["y"])] + snap["checkpoints"], dtype=float)
        ax.plot(route[:, 0], route[:, 1], color=ROUTE_COLOR, ls="--", lw=1.2, zorder=3)
        ax.plot(
            route[1, 0], route[1, 1], marker="X", ms=7, color=ROUTE_COLOR,
            markeredgecolor="white", markeredgewidth=0.6, ls="none", zorder=5,
        )

    # ---- 交通 / 静态物 ----
    ego_xy = np.array([snap["ego"]["x"], snap["ego"]["y"]])
    for item in snap["traffic"]:
        corners = _box_corners(item["x"], item["y"], item["heading"], item["length"], item["width"])
        color = TRAFFIC_COLOR if item["is_vehicle"] else STATIC_COLOR
        edge = TRAFFIC_EDGE if item["is_vehicle"] else STATIC_EDGE
        ax.add_patch(
            Polygon(corners, closed=True, facecolor=color, edgecolor=edge, lw=0.6, alpha=0.85, zorder=4)
        )
        # 文字只标注自车附近物体：大范围视野下逐车标注会互相叠压不可读
        distance_to_ego = float(np.hypot(item["x"] - ego_xy[0], item["y"] - ego_xy[1]))
        label_radius = 40.0 if item["is_vehicle"] else 80.0
        if distance_to_ego <= label_radius:
            ax.text(
                item["x"], item["y"], item["class_name"], fontsize=5.2, color="white",
                ha="center", va="center", zorder=6, clip_on=True,
            )
            if item["is_vehicle"] and show_traffic_speed:
                label = f"v={item['speed']:.1f} m/s" if np.isfinite(item.get("speed", float("nan"))) else item["class_name"]
                ax.text(
                    item["x"], item["y"] + 0.6 * item["length"] + 0.7,
                    label, fontsize=5.0, color="#1F4E8C",
                    ha="center", va="bottom", zorder=6, clip_on=True,
                )

    # ---- ego：红框 + 航向箭头 ----
    ego = snap["ego"]
    corners = _box_corners(ego["x"], ego["y"], ego["heading"], ego["length"], ego["width"])
    ax.add_patch(Polygon(corners, closed=True, facecolor=EGO_COLOR, edgecolor=EGO_EDGE, lw=1.1, zorder=7))
    ax.annotate(
        "", xy=(ego["x"] + 1.6 * ego["length"] * np.cos(ego["heading"]),
                ego["y"] + 1.6 * ego["length"] * np.sin(ego["heading"])),
        xytext=(ego["x"], ego["y"]),
        arrowprops=dict(arrowstyle="-|>", color=EGO_EDGE, lw=1.4), zorder=8,
    )
    ax.text(
        ego["x"], ego["y"] + 0.6 * ego["length"] + 0.7, "EGO",
        fontsize=6.0, color=EGO_EDGE, ha="center", va="bottom", zorder=8, clip_on=True,
    )

    # ---- 自车车道限速标注 ----
    if snap["limit_text"] and snap["limit_xy"]:
        ax.text(
            snap["limit_xy"][0], snap["limit_xy"][1], snap["limit_text"],
            fontsize=5.6, color="#0B6B3A", ha="center", va="center", zorder=9,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#9CC5AE", lw=0.4, alpha=0.9),
        )

    # ---- 视野 / 轴 ----
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#FBFBFB")
    ax.grid(True, color="#DDDDDD", ls=":", lw=0.4, alpha=0.8)
    ax.set_xlabel("x [m]", fontsize=7)
    ax.set_ylabel("y [m]", fontsize=7)
    ax.tick_params(labelsize=6, length=2)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.6)

    if title:
        ax.set_title(title, fontsize=8, loc="left", family="DejaVu Sans")

    if legend:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=EGO_COLOR, edgecolor=EGO_EDGE, label="ego"),
            Patch(facecolor=TRAFFIC_COLOR, edgecolor=TRAFFIC_EDGE, label="traffic vehicle"),
            Patch(facecolor=STATIC_COLOR, edgecolor=STATIC_EDGE, label="static object"),
            Line2D([0], [0], color=ROUTE_COLOR, ls="--", marker="X", ms=5, label="route (ckpt 1 marked)"),
            Line2D([0], [0], color=GREY_LINE_COLOR, lw=1.0, label="lane line"),
            Line2D([0], [0], color=YELLOW_LINE_COLOR, lw=1.0, label="yellow line"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=5.4, framealpha=0.85, handlelength=1.6)


def _compact_lane_index(index_str: str) -> str:
    """车道 index 紧凑化：``('>>', '>>>', 1)`` -> ``>>:1``（只保留来源节点与车道号）。"""
    text = str(index_str)
    if not text.startswith("("):
        return text
    try:
        parts = [part.strip().strip("'\"") for part in text.strip("()").split(",")]
        return f"{parts[0]}->{parts[1]}:{parts[2]}"
    except IndexError:
        return text


def _frame_title(snap: dict, index: int, total: int) -> str:
    ego = snap["ego"]
    status = f"  [{snap['status']}]" if snap.get("status") else ""
    head = (
        f"frame {index + 1}/{total} - step {snap['step']} ({snap['time_s']:.1f} s)  "
        f"v={ego['speed']:.1f} m/s  ego_lane={_compact_lane_index(ego['lane_index'])}{status}"
    )
    return head + "\n" + _format_labels(snap["labels"], width=80)


def render_scenario_figure(
    spec: Any,
    snapshots: Sequence[dict],
    lane_data: Sequence[dict],
    map_bbox: Optional[tuple[float, float, float, float]],
    out_path: Path,
    *,
    policy: str,
    steps: int,
    dt: float,
) -> None:
    """单场景 1×N 帧拼图；面板高宽比跟随场景跨度（equal aspect 下不浪费版面）。"""
    artists = _lane_artists(lane_data)
    frame_count = max(len(snapshots), 1)
    raw_span_x, raw_span_y, _, _ = _raw_spans(snapshots, map_bbox)
    panel_aspect = float(np.clip(raw_span_x / max(raw_span_y, 1e-6), 0.6, 4.0))
    bounds = _view_bounds(snapshots, map_bbox, aspect=panel_aspect)

    figure_width = 5.9 * frame_count
    left, right, wspace = 0.045, 0.99, 0.16
    axes_width = figure_width * (right - left) / (frame_count + wspace * (frame_count - 1))
    axes_height = axes_width / panel_aspect
    bottom_in, top_in = 0.45, 1.25  # x 轴标签 / (suptitle 4 行 + 帧标题 3 行)
    figure_height = axes_height + bottom_in + top_in
    figure, axes = plt.subplots(1, frame_count, figsize=(figure_width, figure_height))
    if frame_count == 1:
        axes = [axes]
    for index, (ax, snap) in enumerate(zip(axes, snapshots)):
        _draw_scene(ax, spec, artists, snap, bounds, title=_frame_title(snap, index, frame_count), legend=index == 0)
    block = _formation(spec) + f"\nrun: policy={policy}  steps={steps} (dt={dt:.3f} s)  frames={frame_count}"
    figure.suptitle(block, fontsize=8.5, ha="left", x=0.005, y=0.995, va="top", family="DejaVu Sans")
    figure.subplots_adjust(
        top=1.0 - top_in / figure_height, bottom=bottom_in / figure_height,
        left=left, right=right, wspace=wspace,
    )
    figure.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def _contact_title(spec: Any, first: dict) -> str:
    """总览面板标题：3~5 行、每行不超过 ~92 字符，保证小面板可读。"""
    traffic = _traffic_dict(spec)
    try:
        density = f"{float(traffic.get('density', 0.0)):.3f}"
    except (TypeError, ValueError):
        density = "?"
    patterns = ",".join(str(p) for p in (traffic.get("patterns") or [])) or "-"
    lines = [
        f"#{spec.id}  seed={spec.seed}  blocks={spec.blocks}  difficulty={spec.difficulty}",
        f"geometry={'/'.join(str(g) for g in spec.geometry)}",
        f"traffic d={density}  patterns={patterns}",
        f"events: {_events_brief(spec) or '-'}",
    ]
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=92) or [line])
    return "\n".join(wrapped + [_format_labels(first["labels"], width=92)])


def render_contact_sheet(
    entries: Sequence[dict],
    out_path: Path,
    *,
    columns: int = 2,
) -> None:
    """全体场景首帧总览（2 列 × N/2 行，标题可读）。"""
    if not entries:
        return
    rows = (len(entries) + columns - 1) // columns
    panel_width, panel_height = 6.6, 4.1  # in/面板；aspect 与 _view_bounds 对齐
    figure, axes = plt.subplots(rows, columns, figsize=(panel_width * columns, panel_height * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax in axes[len(entries):]:
        ax.axis("off")
    for index, entry in enumerate(entries):
        ax = axes[index]
        spec = entry["spec"]
        snapshots = entry["snapshots"]
        artists = _lane_artists(entry["lanes"])
        bounds = _view_bounds(snapshots, entry.get("map_bbox"), aspect=panel_width / (panel_height * 0.78))
        _draw_scene(
            ax, spec, artists, snapshots[0], bounds,
            title=_contact_title(spec, snapshots[0]), legend=False, show_traffic_speed=False,
        )
    figure.suptitle(
        "MetaDrive scenario first frames (top-down, world frame)",
        fontsize=12, y=1.0, family="DejaVu Sans",
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))
    figure.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(figure)


# ======================================================================================
# episode 执行
# ======================================================================================
def _step_dt(env: Any) -> float:
    """单步物理时长（s）：MetaDrive 默认 0.02 s × 5 = 0.1 s（base_env.py:188-190）。"""
    config = getattr(env, "config", None)
    try:
        return float(config["physics_world_step_size"]) * float(config["decision_repeat"])
    except Exception:  # noqa: BLE001
        return 0.1


def _termination_status(terminated: bool, truncated: bool, info: dict) -> str:
    if terminated:
        for flag, label in (("crash", "collision"), ("out_of_road", "out_of_road"), ("arrive_dest", "arrive_dest")):
            if info.get(flag):
                return label
        return "terminated"
    if truncated:
        return "truncated"
    return ""


def run_episode(
    spec: Any, *, policy: str, steps: int, frames: int
) -> tuple[list[dict], list[dict], Optional[tuple[float, float, float, float]], float]:
    """建 env → reset → 驱动 → 抓帧；返回 ``(snapshots, lane_data, map_bbox, dt)``。"""
    env = None
    try:
        env = build_env(spec)
        env.reset()
        dt = _step_dt(env)
        lane_data, map_bbox = _extract_lanes(env)

        if policy == "baseline":
            ego = env.vehicle
            ego_policy = env.engine.add_policy(ego.id, PurePursuitIDMPolicy, ego, int(spec.seed))
            ego_policy.reset()

        requested = _frame_steps(int(steps), int(frames))
        snapshots: dict[int, dict] = {}
        status = ""
        executed = 0
        for step in range(int(steps) + 1):
            if step in requested:
                snapshots[step] = _snapshot(env, spec, step, dt, status)
            if step >= int(steps):
                break
            _, _, terminated, truncated, info = env.step([0.0, 0.0])
            executed = step + 1
            status = _termination_status(bool(terminated), bool(truncated), info)
            if status:
                snapshots.setdefault(executed, _snapshot(env, spec, executed, dt, status))
                break

        # 提前终止/未采集的帧：用不晚于该步的最近快照填充（保持 N 帧版面）
        ordered_steps = sorted(snapshots)
        filled = []
        for step in requested:
            chosen = max((candidate for candidate in ordered_steps if candidate <= step), default=ordered_steps[0])
            filled.append(snapshots[chosen])
        return filled, lane_data, map_bbox, dt
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


# ======================================================================================
# 选择与 CLI
# ======================================================================================
def _select_specs(specs: Sequence[Any], args: argparse.Namespace) -> list[Any]:
    """按 --random / --ids / --all-in-file 选择场景（保文件顺序、可复现）。"""
    if args.all_in_file:
        return list(specs)
    if args.ids:
        wanted = []
        for token in str(args.ids).split(","):
            token = token.strip()
            if not token:
                continue
            try:
                wanted.append(int(token))
            except ValueError:
                raise SystemExit(f"--ids 含非整数项: {token!r}")
        by_id = {int(getattr(spec, "id", -1)): spec for spec in specs}
        missing = [spec_id for spec_id in wanted if spec_id not in by_id]
        if missing:
            print(f"[warn] --ids 中不存在的 id: {missing}", flush=True)
        return [by_id[spec_id] for spec_id in wanted if spec_id in by_id]
    count = int(args.random)
    if count >= len(specs):
        print(f"[warn] --random {count} >= specs {len(specs)}，取全部", flush=True)
        return list(specs)
    rng = random.Random(int(args.seed))
    return [specs[index] for index in sorted(rng.sample(range(len(specs)), count))]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visualize.py",
        description="场景俯视图抽检（headless matplotlib；英文图内标注）",
    )
    parser.add_argument(
        "--specs", default="env/specs/scenarios_train_slice200.json",
        help="scenario-spec JSON 路径（load_specs 可读；默认 env/specs/scenarios_train_slice200.json）",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--random", type=int, metavar="N", help="随机抽 N 个场景（抽样由 --seed 决定）")
    group.add_argument("--ids", default="", help="显式选择 spec id，逗号分隔，如 1,5,9")
    group.add_argument("--all-in-file", action="store_true", help="选择文件中全部场景")
    parser.add_argument("--seed", type=int, default=0, help="抽样随机种子（默认 0；仅影响 --random）")
    parser.add_argument("--frames", type=int, default=3, help="每场景抓帧数（默认 3：t=0/mid/end）")
    parser.add_argument("--steps", type=int, default=120, help="每场景仿真步数（默认 120 ≈ 12 s）")
    parser.add_argument("--policy", choices=("baseline", "idle"), default="baseline",
                        help="baseline=PurePursuitIDMPolicy 驱动；idle=只 reset 后空动作")
    parser.add_argument("--out", default="runs/vis", help="输出目录（默认 runs/vis）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if int(args.frames) < 1:
        parser.error("--frames 必须 >= 1")
    if int(args.steps) < 0:
        parser.error("--steps 必须 >= 0")

    specs_path = Path(args.specs)
    if not specs_path.is_file():
        parser.error(f"spec 文件不存在: {specs_path}")
    specs = load_specs(specs_path)
    selected = _select_specs(specs, args)
    if not selected:
        print("[error] 没有选中任何场景", flush=True)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    entries: list[dict] = []
    failures: list[dict] = []
    for index, spec in enumerate(selected, start=1):
        tag = f"#{spec.id} seed={spec.seed} blocks={spec.blocks} {spec.difficulty}"
        scenario_started = time.time()
        try:
            snapshots, lane_data, map_bbox, dt = run_episode(
                spec, policy=str(args.policy), steps=int(args.steps), frames=int(args.frames)
            )
            image_path = out_dir / f"scenario_{spec.id}_{spec.seed}.png"
            render_scenario_figure(
                spec, snapshots, lane_data, map_bbox, image_path,
                policy=str(args.policy), steps=int(args.steps), dt=dt,
            )
            entries.append(
                {
                    "spec": spec,
                    "snapshots": snapshots,
                    "lanes": lane_data,
                    "map_bbox": map_bbox,
                    "image": image_path.name,
                }
            )
            elapsed = time.time() - scenario_started
            ended = snapshots[-1]["step"]
            status = snapshots[-1]["status"] or "ok"
            print(
                f"[{index}/{len(selected)}] {tag} -> {image_path.name} "
                f"(steps={ended}, {status}, {elapsed:.2f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - 单场景失败必须跳过而非终止整批
            failures.append({"spec": spec, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
            print(f"[{index}/{len(selected)}] {tag} -> FAILED: {type(exc).__name__}: {exc}", flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)

    contact_name = None
    if entries:
        contact_path = out_dir / "contact_sheet.png"
        try:
            render_contact_sheet(entries, contact_path)
            contact_name = contact_path.name
        except Exception as exc:  # noqa: BLE001 - 总览失败不影响单图
            print(f"[warn] contact_sheet 生成失败: {type(exc).__name__}: {exc}", flush=True)

    _write_index(
        out_dir / "index.md",
        specs_path=specs_path,
        args=args,
        entries=entries,
        failures=failures,
        contact_name=contact_name,
        dt=float("nan"),
        elapsed=time.time() - started_at,
    )
    total = time.time() - started_at
    print(
        f"done: {len(entries)} rendered, {len(failures)} failed, out={out_dir.resolve()} "
        f"(contact_sheet={contact_name}, {total:.2f}s)",
        flush=True,
    )
    return 0 if entries else 1


def _write_index(
    path: Path,
    *,
    specs_path: Path,
    args: argparse.Namespace,
    entries: Sequence[dict],
    failures: Sequence[dict],
    contact_name: Optional[str],
    dt: float,
    elapsed: float,
) -> None:
    """写 index.md：选择口径 + 每张图的注释行 + 失败清单。"""
    selection = (
        f"--all-in-file ({len(entries) + len(failures)} specs)"
        if args.all_in_file
        else (f"--ids {args.ids}" if args.ids else f"--random {args.random} --seed {args.seed}")
    )
    lines = [
        "# Scenario visualization index",
        "",
        f"- specs: `{specs_path}`",
        f"- selection: {selection}",
        f"- policy: {args.policy}; steps: {args.steps}; frames: {args.frames}",
        f"- runtime: {elapsed:.2f} s",
        "",
    ]
    if contact_name:
        lines += [f"![contact sheet]({contact_name})", "", f"- `{contact_name}` — first frame of every selected scenario", ""]
    lines += [
        "## Scenarios",
        "",
    ]
    if entries:
        for entry in entries:
            lines.append(f"- `{entry['image']}` — {_annotation_line(entry['spec'])}")
    else:
        lines.append("(no scenario rendered)")
    if failures:
        lines += ["", "## Errors", ""]
        for failure in failures:
            lines.append(f"- id={getattr(failure['spec'], 'id', '?')} seed={getattr(failure['spec'], 'seed', '?')}: {failure['error']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
