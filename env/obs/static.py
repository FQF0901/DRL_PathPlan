"""静态障碍（建筑/岗亭）走廊扫描（schema v4，P1-A）。

为什么需要：收费站岗亭是 MetaDrive 的 ``BaseBuilding``（静态对象）——不出现在 OD
（只编码动态车辆）与 LD（只编码车道线），而 tollgate 的变道决策点在上游 25–39 m，
彼时 ``others.road_class`` 仍为 0（未进入 ``$`` 块）→ 策略无法观测"本车道前方有亭"。
T3 triage 的确定性签名 S1–S4：9/9 失败 episode 的 plan 恒指向岗亭车道；IDM 参考实现
（``env/expert/pure_pursuit_idm.py::_static_blocker_gap``）单独扫描建筑并在 ~39.5 m
完成变道。本模块把同一几何口径编码进 ``others`` 通道的 ``static`` 段。

几何口径（与 IDM 对齐）
----------------------
- **走廊**：自车当前车道中心线两侧 ``lane.width/2 + lat_margin + lane_span·lane.width``
  （默认 ``lane_span=1`` → 覆盖左右各 1 条相邻车道）。比 IDM 的"单车道扫描"宽：
  变道决策点需要同时看到相邻车道的岗亭才能选自由车道；IDM 通过对每条候选车道分别
  扫描达到等价效果，obs 用"一次走廊扫描 + 相对车道 one-hot"表达。
- **净距**：``gap = s_obj - s_ego - 0.5·L_obj``（自车中心 → 障碍近面，IDM 同口径；
  仅接受 ``-0.5·L_obj <= gap <= scan_range``，报告值 clamp 到 >= 0）。
- **相对车道**：``rel = round(lat_left / lane.width)``，clamp 到 ``[-lane_span, lane_span]``；
  ``lat_left`` 为障碍中心相对当前车道中心线的横向偏移（**左正**，与 env 自车系
  y 左向 / ``_lane_lateral_left`` 同一约定，不依赖 MetaDrive lane local lat 的符号）。
- 取走廊内前方最近者；无 → 全 0（``present=0``）。

特征（``STATIC_DIM = 5``，写入 ``others`` 的 static 段）
--------------------------------------------------------
===== ============ =========================================================
dim   名称         语义
===== ============ =========================================================
0     present      1 = 走廊内找到前方静态障碍（在 range 内）
1     gap_norm     ``clamp(gap,0,scan_range)/scan_range``；无 → 0
2..4  rel one-hot  ``STATIC_REL_BUCKETS = (-1, 0, +1)``：障碍所在车道相对自车
                   当前车道（+1 = 左侧相邻车道；0 = 当前车道；-1 = 右侧相邻
                   车道）；无 → 全 0
===== ============ =========================================================

fallback：metadrive 不可用 / ego 无 lane / 场景无建筑 → 全 0（present=0，等价于
"无静态障碍"）；旧数据缺该段由 net 侧零填充（见 ``net.mem`` 的 legacy 重排）。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - 纯 NumPy 测试环境无 metadrive 时走占位类
    from metadrive.component.buildings.base_building import BaseBuilding
except Exception:  # noqa: BLE001 - 观测通道不得因 metadrive 缺失而崩

    class BaseBuilding:  # type: ignore[no-redef]
        """metadrive 不可用时的占位（任何对象都不是建筑）。"""


__all__ = [
    "STATIC_DIM",
    "STATIC_SCAN_RANGE_M",
    "STATIC_LAT_MARGIN_M",
    "STATIC_LANE_SPAN",
    "STATIC_REL_BUCKETS",
    "scan_static_blocker",
    "static_features",
]

#: 走廊前向搜索距离（m）；50 > IDM 的 40，覆盖 25–39 m 的变道决策点
STATIC_SCAN_RANGE_M = 50.0
#: 走廊横向余量（m；IDM ``static_obstacle_lat_margin`` 同值）
STATIC_LAT_MARGIN_M = 1.0
#: 走廊覆盖的相邻车道数（每侧）；1 = 当前车道 + 左右各 1 条
STATIC_LANE_SPAN = 1
#: 相对车道桶（+1 = 左侧相邻车道；0 = 当前车道；-1 = 右侧相邻车道）
STATIC_REL_BUCKETS: Tuple[int, ...] = (-1, 0, 1)
#: static 段特征维：present + gap_norm + rel one-hot
STATIC_DIM = 2 + len(STATIC_REL_BUCKETS)


def _lane_lateral_left(lane, point, s: float) -> Optional[float]:
    """``point`` 相对 ``lane`` 在弧长 ``s`` 处中心线的横向偏移（左正，m）。

    为什么不用 ``lane.local_coordinates`` 的 lat：部分车道（CircularLane 等）横向
    正方向随车道 ``direction`` 翻转（``env/expert/pure_pursuit_idm.py::_lane_lateral_left``
    同款处理）；这里统一投影到车道切向左侧（``[-sin h, cos h]``），与 env 自车系
    y 左向一致。
    """
    try:
        length = float(lane.length)
        s_clamped = float(min(max(float(s), 0.0), length)) if math.isfinite(length) else 0.0
        heading = float(lane.heading_theta_at(s_clamped))
        center = np.asarray(lane.position(s_clamped, 0.0), dtype=np.float64)[:2]
        offset = np.asarray(point, dtype=np.float64)[:2] - center
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    value = -offset[0] * math.sin(heading) + offset[1] * math.cos(heading)
    return float(value) if math.isfinite(value) else None


def _ego_lane(ego):
    """自车当前车道；``ego.lane`` 不可用时回退导航参考车道（与 speed_limit 同口径）。"""
    lane = getattr(ego, "lane", None)
    if lane is not None:
        return lane
    nav = getattr(ego, "navigation", None)
    refs = list(getattr(nav, "current_ref_lanes", None) or [])
    return refs[0] if refs else None


def scan_static_blocker(
    env,
    ego,
    *,
    scan_range_m: float = STATIC_SCAN_RANGE_M,
    lat_margin_m: float = STATIC_LAT_MARGIN_M,
    lane_span: int = STATIC_LANE_SPAN,
) -> Optional[Tuple[float, int]]:
    """走廊内前方最近静态障碍 → ``(gap_m, rel_bucket)``；无返回 ``None``。

    ``rel_bucket`` ∈ ``STATIC_REL_BUCKETS``（+1 左 / 0 当前 / -1 右）。
    任何异常/字段缺失都返回 ``None``（观测通道不得崩）。
    """
    if lane_span < 0 or scan_range_m <= 0.0:
        return None
    lane = _ego_lane(ego)
    if lane is None:
        return None
    try:
        objects = list(env.engine.get_objects().values())
    except (AttributeError, KeyError, TypeError):
        return None
    try:
        s_ego = float(lane.local_coordinates(ego.position)[0])
    except (AttributeError, TypeError, ValueError, IndexError):
        return None
    if not math.isfinite(s_ego):
        return None
    try:
        width = float(getattr(lane, "width", 0.0) or 0.0)
    except (TypeError, ValueError):
        width = 0.0
    if width <= 0.0:
        width = 3.5
    corridor = width / 2.0 + float(lat_margin_m) + float(lane_span) * width

    best_gap: Optional[float] = None
    best_bucket: Optional[int] = None
    for obj in objects:
        if not isinstance(obj, BaseBuilding):
            continue
        try:
            s_obj = float(lane.local_coordinates(obj.position)[0])
        except (AttributeError, TypeError, ValueError, IndexError):
            continue
        if not math.isfinite(s_obj):
            continue
        lat_left = _lane_lateral_left(lane, obj.position, s_obj)
        if lat_left is None or abs(lat_left) > corridor:
            continue
        try:
            half_length = 0.5 * float(getattr(obj, "LENGTH", 0.0) or 0.0)
        except (TypeError, ValueError):
            half_length = 0.0
        gap = s_obj - s_ego - half_length
        if gap < -half_length or gap > float(scan_range_m):
            continue
        bucket = int(round(lat_left / width))
        bucket = max(-int(lane_span), min(int(lane_span), bucket))
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_bucket = bucket
    if best_gap is None or best_bucket is None:
        return None
    return float(max(best_gap, 0.0)), int(best_bucket)


def static_features(
    env,
    ego,
    *,
    scan_range_m: float = STATIC_SCAN_RANGE_M,
    lat_margin_m: float = STATIC_LAT_MARGIN_M,
    lane_span: int = STATIC_LANE_SPAN,
) -> np.ndarray:
    """static 段特征 ``(STATIC_DIM,) float32``；无发现 → 全 0。"""
    feats = np.zeros((STATIC_DIM, ), dtype=np.float32)
    found = scan_static_blocker(
        env, ego, scan_range_m=scan_range_m, lat_margin_m=lat_margin_m, lane_span=lane_span
    )
    if found is None:
        return feats
    gap, bucket = found
    feats[0] = 1.0
    if scan_range_m > 0.0:
        feats[1] = float(min(max(gap, 0.0), float(scan_range_m)) / float(scan_range_m))
    # lane_span 配置大于 one-hot 桶范围时，归入最近桶（不出现"present=1 但 one-hot 全 0"）
    bucket = max(min(STATIC_REL_BUCKETS), min(max(STATIC_REL_BUCKETS), int(bucket)))
    try:
        index = STATIC_REL_BUCKETS.index(bucket)
    except ValueError:
        index = -1
    if index >= 0:
        feats[2 + index] = 1.0
    return feats
