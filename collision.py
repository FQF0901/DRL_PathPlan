#!/usr/bin/env python3
"""
碰撞检测模块 (Collision Detection for Parking Environment)
==========================================================
检测自车八边形轮廓与 FSD / OD 多边形之间的碰撞.

核心算法:
  - Broad phase: AABB (轴对齐包围盒) 快速拒绝
  - Narrow phase: SAT (Separating Axis Theorem) 精确碰撞

坐标系: 自车坐标系 (右手系, X前Y左). 所有多边形都已转换到该坐标系.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from mdf_reader import TimePointData, get_vehicle_params


# ============================================================
# 数据类型
# ============================================================

@dataclass
class CollisionResult:
    """一次碰撞检测的结果."""
    source: str               # 'OD' | 'FSD'
    index: int                # 在对应列表中的序号
    polygon: np.ndarray       # (N, 2) 多边形顶点
    collides: bool            # 是否碰撞
    penetration: float        # 穿透深度 (近似, 0 表示无碰撞)


# ============================================================
# 几何工具
# ============================================================

def _polygon_aabb(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """返回多边形轴对齐包围盒 (xmin, xmax, ymin, ymax)."""
    return (float(pts[:, 0].min()), float(pts[:, 0].max()),
            float(pts[:, 1].min()), float(pts[:, 1].max()))


def _aabb_overlap(a, b) -> bool:
    """检测两个 AABB 是否重叠. 每个 a/b = (xmin, xmax, ymin, ymax)."""
    return not (a[1] < b[0] or a[0] > b[1] or a[3] < b[2] or a[2] > b[3])


def _edge_normals(pts: np.ndarray) -> List[np.ndarray]:
    """返回凸多边形各边的单位法向量 (指向外侧, 假设 CCW)."""
    normals = []
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        edge = pts[j] - pts[i]
        # 垂直向量 (指向外侧, 假设顶点 CCW)
        nx, ny = edge[1], -edge[0]
        norm = np.hypot(nx, ny)
        if norm > 1e-12:
            normals.append(np.array([nx / norm, ny / norm]))
    return normals


def _project_polygon(pts: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    """将多边形顶点投影到轴上, 返回 (min, max)."""
    dots = pts @ axis
    return (float(dots.min()), float(dots.max()))


def _sat_intersect(pts_a: np.ndarray, pts_b: np.ndarray) -> Tuple[bool, float]:
    """
    SAT 检测两个凸多边形是否相交.
    返回 (collides, penetration_depth).
    """
    min_overlap = float('inf')

    # 收集两个多边形的所有边的法线
    axes = _edge_normals(pts_a) + _edge_normals(pts_b)

    for axis in axes:
        min_a, max_a = _project_polygon(pts_a, axis)
        min_b, max_b = _project_polygon(pts_b, axis)

        if max_a < min_b or max_b < min_a:
            return False, 0.0  # 分离轴 → 不相交

        overlap = min(max_a - min_b, max_b - min_a)
        if overlap < min_overlap:
            min_overlap = overlap

    return True, min_overlap


# ============================================================
# 碰撞检测器
# ============================================================

class CollisionDetector:
    """
    自车轮廓与 FSD/OD 多边形碰撞检测器.

    用法:
        detector = CollisionDetector(veh)
        results = detector.detect(tpd)
        for r in results:
            if r.collides:
                print(f"碰撞! {r.source}[{r.index}] 穿透={r.penetration:.3f}m")
    """

    def __init__(self, veh: dict):
        """
        参数:
            veh: get_vehicle_params() 返回的车辆参数字典.
        """
        # 自车轮廓顶点 (N, 2) — 已经在自车坐标系中
        self._ego_pts = np.column_stack([
            veh['contour_x'],
            veh['contour_y'],
        ])
        # 预计算 AABB 和边法线 (固定不变)
        self._ego_aabb = _polygon_aabb(self._ego_pts)
        self._ego_normals = _edge_normals(self._ego_pts)

    def detect(self, tpd: TimePointData) -> List[CollisionResult]:
        """检测 tpd 中所有 FSD/OD 与自车的碰撞."""
        results = []

        # --- OD ---
        for i, poly in enumerate(tpd.sifor_od):
            res = self._test_polygon(poly, 'OD', i)
            results.append(res)

        # --- FSD ---
        for i, poly in enumerate(tpd.sifor_fsd):
            res = self._test_polygon(poly, 'FSD', i)
            results.append(res)

        return results

    def detect_od(self, tpd: TimePointData) -> List[CollisionResult]:
        """仅检测 OD 碰撞."""
        return [self._test_polygon(poly, 'OD', i)
                for i, poly in enumerate(tpd.sifor_od)]

    def detect_fsd(self, tpd: TimePointData) -> List[CollisionResult]:
        """仅检测 FSD 碰撞."""
        return [self._test_polygon(poly, 'FSD', i)
                for i, poly in enumerate(tpd.sifor_fsd)]

    def _test_polygon(self, poly: np.ndarray, source: str,
                      index: int) -> CollisionResult:
        """检测单个多边形与自车轮廓的碰撞."""
        # 1. AABB broad phase
        poly_aabb = _polygon_aabb(poly)
        if not _aabb_overlap(self._ego_aabb, poly_aabb):
            return CollisionResult(source, index, poly, False, 0.0)

        # 2. SAT narrow phase
        collides, pen = _sat_intersect(self._ego_pts, poly)
        return CollisionResult(source, index, poly, collides, pen)


# ============================================================
# 便捷函数
# ============================================================

def detect_collisions(tpd: TimePointData,
                      veh: dict) -> List[CollisionResult]:
    """便捷函数: 一次调用完成所有碰撞检测."""
    detector = CollisionDetector(veh)
    return detector.detect(tpd)


def collision_summary(results: List[CollisionResult]) -> str:
    """生成可读的碰撞摘要."""
    colliding = [r for r in results if r.collides]
    if not colliding:
        return "无碰撞."

    lines = [f"检测到 {len(colliding)} 个碰撞:"]
    for r in colliding:
        lines.append(f"  {r.source}[{r.index}] 穿透深度={r.penetration:.3f}m")
    return "\n".join(lines)
