#!/usr/bin/env python3
"""
Geometric Path Planner for APA Parallel Parking
================================================
从自车位姿到 TargetPose 的几何规划算法.

策略:
  1. 直线 (heading 变化 < 20°)
  2. 单段圆弧
  3. 直线+圆弧 (倒车直行后切弧)
  4. 两段圆弧: 切触条件闭式解, R2 由 R1 一元线性解出 (无搜索采样)
  5. 三段: 直线 + 两段圆弧 (含前进/倒车两种初始方向)

约束:
  - 每段 >= 0.2m
  - 转弯半径 >= Rmin
  - FSD type=2 不参与碰撞
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from mdf_reader import TimePointData
from collision import _sat_intersect, _polygon_aabb, _aabb_overlap


@dataclass
class PathSegment:
    start_pose: Tuple[float, float, float]
    end_pose: Tuple[float, float, float]
    radius: float
    direction: str
    length: float
    num: int


@dataclass
class PathResult:
    segments: List[PathSegment]
    poses: np.ndarray
    n_maneuvers: int
    total_length: float
    valid: bool = False
    rejected_by_gate: bool = False


# ========== Geometry ==========

def _arc_center(pose, radius):
    x, y, yaw = pose
    if radius > 0:
        return (x - radius*np.sin(yaw), y + radius*np.cos(yaw))
    r = abs(radius)
    return (x + r*np.sin(yaw), y - r*np.cos(yaw))


def _sample_arc(start, radius, da, step=0.1, end_pos=None):
    cx, cy = _arc_center(start, radius)
    r = abs(radius)
    sa = np.arctan2(start[1]-cy, start[0]-cx)
    n = max(2, int(r*abs(da)/step)+1)
    da2 = da/n
    pts = []
    for i in range(n+1):
        a = sa + i*da2
        pts.append([cx+r*np.cos(a), cy+r*np.sin(a), start[2]+i*da2])
    pts = np.array(pts)
    if end_pos is not None:
        pts[-1, 0] = end_pos[0]
        pts[-1, 1] = end_pos[1]
        pts[-1, 2] = start[2] + da
        # 折角覆盖: 硬钳位末端点后, 若最后两个采样点间距 > 0.15m,
        # 在 pos[n-1] 与钳位点之间按 0.1m 步长插入线性插值中间点 (位置+heading).
        gap = np.hypot(pts[-1, 0]-pts[-2, 0], pts[-1, 1]-pts[-2, 1])
        if gap > 0.15:
            nins = int(np.ceil(gap / 0.1))
            seg = []
            for k in range(1, nins):
                frac = k / nins
                seg.append([pts[-2, 0] + frac*(pts[-1, 0]-pts[-2, 0]),
                            pts[-2, 1] + frac*(pts[-1, 1]-pts[-2, 1]),
                            pts[-2, 2] + frac*(pts[-1, 2]-pts[-2, 2])])
            pts = np.vstack([pts[:-1], np.array(seg), pts[-1:]])
    return pts


def _sample_line(start, length, step=0.1, end_pos=None, end_yaw=None):
    """采样直线, 位置从 start 线性插值到 end_pos, heading 从 start_yaw 到 end_yaw.
    
    参数:
      start: (x, y, yaw) 起点
      length: 路径长度 (用于决定采样点数)
      step: 采样步长
      end_pos: (ex, ey) 终点位置 (不传则用 heading 方向投影)
      end_yaw: 终点 heading (不传则保持起点头角)
    """
    x, y, yaw = start
    n = max(2, int(length/step)+1)
    if end_pos is None:
        end_pos = (x + length*np.cos(yaw), y + length*np.sin(yaw))
    if end_yaw is None:
        end_yaw = yaw
    ex, ey = end_pos
    poses = np.zeros((n + 1, 3))
    for i in range(n + 1):
        frac = i / n
        poses[i, 0] = x + frac * (ex - x)
        poses[i, 1] = y + frac * (ey - y)
        poses[i, 2] = yaw + frac * (end_yaw - yaw)
    # 硬钳位最后一点与终点完全重合
    poses[-1, 0] = ex
    poses[-1, 1] = ey
    poses[-1, 2] = end_yaw
    # 折角覆盖: 硬钳位末端点后, 若最后两个采样点间距 > 0.15m,
    # 在 pos[n-1] 与钳位点之间按 0.1m 步长插入线性插值中间点 (位置+heading).
    gap = np.hypot(poses[-1, 0]-poses[-2, 0], poses[-1, 1]-poses[-2, 1])
    if gap > 0.15:
        nins = int(np.ceil(gap / 0.1))
        seg = []
        for k in range(1, nins):
            frac = k / nins
            seg.append([poses[-2, 0] + frac*(poses[-1, 0]-poses[-2, 0]),
                        poses[-2, 1] + frac*(poses[-1, 1]-poses[-2, 1]),
                        poses[-2, 2] + frac*(poses[-1, 2]-poses[-2, 2])])
        poses = np.vstack([poses[:-1], np.array(seg), poses[-1:]])
    return poses


def _compute_arc(start, end):
    """
    Determine if a circular arc connects start_pose to end_pose.
    Returns (signed_radius, arc_angle, length) or None.
    radius > 0: left turn (heading increases), radius < 0: right turn (heading decreases).
    Unifies both turns: center = (x - R*sin(θ), y + R*cos(θ)) for any signed R.
    """
    x1, y1, y1aw = start
    x2, y2, y2aw = end
    dh = np.arctan2(np.sin(y2aw-y1aw), np.cos(y2aw-y1aw))
    if abs(dh) < 1e-6:
        return None  # straight line

    # For any signed R: center_x = x - R*sin(θ), center_y = y + R*cos(θ)
    # x1 - R*sin(y1aw) = x2 - R*sin(y2aw)  →  R = (x2-x1)/(sin(y2aw)-sin(y1aw))
    denom = np.sin(y2aw) - np.sin(y1aw)
    if abs(denom) < 1e-6:
        return None
    R = (x2 - x1) / denom

    # Verify y-consistency: y1 + R*cos(y1aw) == y2 + R*cos(y2aw)
    if abs((y1 + R*np.cos(y1aw)) - (y2 + R*np.cos(y2aw))) > 0.2:
        return None

    # Heading change must match radius sign
    da = np.arctan2(np.sin(y2aw-y1aw), np.cos(y2aw-y1aw))
    if R > 0 and da < 0:
        da += 2 * np.pi
    elif R < 0 and da > 0:
        da -= 2 * np.pi
    if (R > 0 and da <= 0) or (R < 0 and da >= 0):
        return None

    return (R, da, abs(R) * abs(da))


# ========== Collision ==========

def _collision_free(poses, tpd, contour):
    for px, py, pyaw in poses:
        c, s = np.cos(pyaw), np.sin(pyaw)
        ego = contour @ np.array([[c, -s], [s, c]]) + np.array([px, py])
        bb = _polygon_aabb(ego)
        for fi, poly in enumerate(tpd.sifor_fsd):
            if tpd.fsd_types[fi] == 2:
                continue
            if _aabb_overlap(bb, _polygon_aabb(poly)) and _sat_intersect(ego, poly)[0]:
                return False
        for poly in tpd.sifor_od:
            if _aabb_overlap(bb, _polygon_aabb(poly)) and _sat_intersect(ego, poly)[0]:
                return False
    return True


# ========== Planner ==========

class PathPlanner:
    def __init__(self, veh: dict):
        self.Rmin = float(veh.get('min_turn_radius', 5.4))
        cx = np.asarray(veh['contour_x'], dtype=float)
        cy = np.asarray(veh['contour_y'], dtype=float)
        self.contour = np.column_stack([cx, cy])

    def plan(self, tpd: TimePointData) -> PathResult:
        tp = tpd.tp
        if np.isnan(tp[0]) or np.isnan(tp[1]):
            return PathResult([], np.empty((0, 3)), 0, 0.0)
        target = (float(tp[0]), float(tp[1]), float(tp[2]))
        ego = (0., 0., 0.)
        candidates = []

        # Already at target?
        if np.hypot(target[0], target[1]) < 0.1:
            return PathResult([], np.empty((0, 3)), 0, 0.0, True)

        # 1. Straight line
        self._try_straight(ego, target, tpd, candidates)
        # 2. Single arc
        self._try_single_arc(ego, target, tpd, candidates)
        # 3. Line→arc (back straight then arc into spot)
        self._try_line_arc(ego, target, tpd, candidates)
        # 4. Two arcs (analytic circle-circle intersection)
        self._try_two(ego, target, tpd, candidates)
        # 5. Three-segment
        self._try_three(ego, target, tpd, candidates)

        if not candidates:
            return PathResult([], np.empty((0, 3)), 0, 0.0)
        best = min(candidates, key=lambda p: (p.n_maneuvers, p.total_length))
        # 质量门: detour 比 > 2.5×欧氏距离 → 诚实报"不可用解"
        dist = np.hypot(target[0], target[1])
        if best.total_length > 2.5 * dist:
            return PathResult([], np.empty((0, 3)), 0, 0.0,
                              rejected_by_gate=True)
        return best

    def _try_straight(self, start, end, tpd, candidates):
        dx, dy = end[0]-start[0], end[1]-start[1]
        dh = abs(np.arctan2(np.sin(end[2]-start[2]), np.cos(end[2]-start[2])))
        dist = np.hypot(dx, dy)
        if dist < 0.1 or dh > np.radians(20):
            return
        direction = 'forward' if dx*np.cos(start[2])+dy*np.sin(start[2]) > 0 else 'reverse'
        poses = _sample_line(start, dist, end_pos=(end[0], end[1]), end_yaw=end[2])
        if not _collision_free(poses, tpd, self.contour):
            return
        candidates.append(PathResult(
            [PathSegment(start, end, 0., direction, dist, 1)],
            poses, 1, dist, True))

    def _try_single_arc(self, start, end, tpd, candidates):
        arc = _compute_arc(start, end)
        if arc is None:
            return
        R, da, length = arc
        if abs(R) < self.Rmin or length < 0.2:
            return
        direction = 'forward' if da*R > 0 else 'reverse'
        poses = _sample_arc(start, R, da, end_pos=(end[0], end[1]))
        if not _collision_free(poses, tpd, self.contour):
            return
        candidates.append(PathResult(
            [PathSegment(start, end, R, direction, length, 1)],
            poses, 1, length, True))

    def _try_two(self, start, end, tpd, candidates, gate_len=None):
        """Two arcs: 圆-圆相交解析解 + 细 (R1,da1) 采样 (混合).

        A. 圆-圆相交: 弧1 起点 start 半径 R1 → int_p 在 C1=_arc_center(start,R1) 圆上;
           弧2 终点 end 半径 R2 → int_p 在 C2=_arc_center(end,R2) 圆上.
           int_p ∈ 两圆交点 (几何求交, 0/1/2 个).
        B. 细采样: R1 网格 × da1 步长 0.05, 弧2 半径由 _compute_arc 自由解出 —
           覆盖近切/终点松弛 (≤0.2m 容差) 的可避障解族.
        每 R1 找到首个 detour 达标候选即提前结束 (解集非空性不变, 大幅提速).
        gate_len: 达标长度门槛 (默认 2.5×欧氏距离); 上层组合 (如三段式) 可传入
        更紧的门槛以统一子门与最终质量门.
        """
        Rmin = self.Rmin
        step = max(0.5, Rmin * 0.1)
        R_pos = np.arange(Rmin, Rmin * 4, step)
        R_neg = np.arange(-Rmin * 4, -Rmin, step)
        R_vals = np.concatenate([R_neg, R_pos])
        seen = set()
        # 质量门感知早停: 每 R1 找到首个 detour 达标候选即停, 否则扫完该 R1
        if gate_len is None:
            gate_len = 2.5 * np.hypot(end[0]-start[0], end[1]-start[1])

        # ---- A. 圆-圆相交解析解 ----
        C2s = [_arc_center(end, R2) for R2 in R_vals]
        for R1 in R_vals:
            c1x, c1y = _arc_center(start, R1)
            r1 = abs(R1)
            sa1 = np.arctan2(start[1]-c1y, start[0]-c1x)
            n_before = len(candidates)
            r1_done = False
            for R2, (c2x, c2y) in zip(R_vals, C2s):
                r2 = abs(R2)
                d = np.hypot(c2x-c1x, c2y-c1y)
                if d > r1 + r2 + 1e-9 or d < abs(r1-r2) - 1e-9 or d < 1e-9:
                    continue
                a = (r1*r1 - r2*r2 + d*d) / (2.0 * d)
                h = np.sqrt(max(0.0, r1*r1 - a*a))
                ux = (c2x - c1x) / d
                uy = (c2y - c1y) / d
                mx, my = c1x + a*ux, c1y + a*uy
                for sgn in (1.0, -1.0):
                    ix, iy = mx + sgn*h*uy, my - sgn*h*ux
                    da1 = np.arctan2(iy-c1y, ix-c1x) - sa1
                    # 规范化到与 R1 符号一致 (±2π 调整)
                    if R1 > 0 and da1 < 0:
                        da1 += 2*np.pi
                    elif R1 < 0 and da1 > 0:
                        da1 -= 2*np.pi
                    if abs(da1) < 0.05:
                        continue
                    self._eval_arc_pair(
                        start, R1, da1, (float(ix), float(iy), start[2]+da1),
                        end, tpd, candidates, seen)
                    if len(candidates) > n_before and candidates[-1].total_length <= gate_len:
                        r1_done = True
                        break
                if r1_done:
                    break

        # ---- B. 细 (R1, da1) 采样, arc2 半径自由 ----
        da_vals = np.arange(-2.5, 2.5001, 0.05)
        for R1 in R_vals:
            cx, cy = _arc_center(start, R1)
            r = abs(R1)
            sa = np.arctan2(start[1]-cy, start[0]-cx)
            n_before = len(candidates)
            for da1 in da_vals:
                if abs(da1) < 0.05:
                    continue
                ix = cx + r*np.cos(sa + da1)
                iy = cy + r*np.sin(sa + da1)
                self._eval_arc_pair(
                    start, R1, da1, (ix, iy, start[2]+da1),
                    end, tpd, candidates, seen)
                if len(candidates) > n_before and candidates[-1].total_length <= gate_len:
                    break

    def _eval_arc_pair(self, start, R1, da1, int_p, end, tpd, candidates, seen):
        """两段弧候选共享校验: 去重 → 长度/半径/弧2 验证 → 碰撞 → append."""
        key = (round(int_p[0], 2), round(int_p[1], 2), round(da1, 2))
        if key in seen:
            return
        seen.add(key)
        ix, iy = int_p[0], int_p[1]
        d1 = abs(R1) * abs(da1)
        if d1 < 0.2:
            return
        d2 = np.hypot(end[0]-ix, end[1]-iy)
        if d2 < 0.2:
            return
        # 弧2: _compute_arc 验证 (y 一致性容差见改动 2)
        arc2 = _compute_arc(int_p, end)
        if arc2 is None:
            return
        R2c, da2, len2 = arc2
        if abs(R2c) < self.Rmin or len2 < 0.2:
            return

        dir1 = 'forward' if da1*R1 > 0 else 'reverse'
        dir2 = 'forward' if da2*R2c > 0 else 'reverse'

        p1 = _sample_arc(start, R1, da1, end_pos=(ix, iy))
        p2 = _sample_arc(int_p, R2c, da2, end_pos=(end[0], end[1]))
        full = np.vstack([p1[:-1], p2])
        if not _collision_free(full, tpd, self.contour):
            return

        s1 = PathSegment(start, int_p, R1, dir1, d1, 1)
        s2 = PathSegment(int_p, end, R2c, dir2, len2, 2)
        candidates.append(PathResult(
            [s1, s2], full, 2, d1+len2, True))

    def _try_line_arc(self, start, end, tpd, candidates):
        """Line→arc: back straight then arc into target. Classic parallel parking entry.

        d 由 y 一致性等式闭式解出: 直线段沿 x 轴 (int_p=(s·d,0,0)), 弧段需满足
        y2 = R·(cosθ2 − 1) 且 R = x2 − s·d / sinθ2 联立消 R 得
        d = s·(x2 − y2·sinθ2/(1−cosθ2)), 使弧精确过终点, 无钳位折角.
        """
        Rmin = self.Rmin
        x2, y2, th2 = end
        denom2 = 1.0 - np.cos(th2)
        if abs(denom2) < 1e-6 or abs(np.sin(th2)) < 1e-6:
            return
        for direction, sign in [('reverse', -1), ('forward', 1)]:
            max_d = 6.0 if direction == 'forward' else 12.0
            d = sign * (x2 - y2 * np.sin(th2) / denom2)
            if d <= 0 or d > max_d:
                continue
            int_p = (sign * d, 0.0, 0.0)
            lp = _sample_line(start, d, end_pos=(int_p[0], int_p[1]), end_yaw=int_p[2])
            if not _collision_free(lp, tpd, self.contour):
                continue
            # Now try arc from intermediate to target
            arc = _compute_arc(int_p, end)
            if arc is None:
                continue
            R, da, length = arc
            if abs(R) < Rmin or length < 0.2:
                continue
            dir2 = 'forward' if da*R > 0 else 'reverse'
            arc_poses = _sample_arc(int_p, R, da, end_pos=(end[0], end[1]))
            full = np.vstack([lp[:-1], arc_poses])
            if not _collision_free(full, tpd, self.contour):
                continue
            s1 = PathSegment(start, int_p, 0., direction, d, 1)
            s2 = PathSegment(int_p, end, R, dir2, length, 2)
            candidates.append(PathResult(
                [s1, s2], full, 2, d+length, True))

    def _try_three(self, start, end, tpd, candidates):
        """Three segments: straight + two-seg. Tries both forward and reverse.

        收集全部 d 的候选 (plan() 全局最小选择); 以已找到的最短总长为下界
        剪枝更长的 d (总长 ≥ d)."""
        for direction, sign in [('forward', 1), ('reverse', -1)]:
            max_d = 12.0 if direction == 'reverse' else 6.0
            best_total = None
            for dist in np.arange(0.5, max_d + 0.01, 0.5):
                if best_total is not None and dist >= best_total:
                    break
                int1 = (sign * dist, 0., 0.)
                lp = _sample_line(start, dist, end_pos=(sign * dist, 0.), end_yaw=start[2])
                if not _collision_free(lp, tpd, self.contour):
                    continue
                sub = []
                # 子门与最终质量门统一: 总长 = dist + 子长 ≤ 2.5×D_total
                self._try_two(int1, end, tpd, sub,
                              gate_len=2.5*np.hypot(end[0], end[1]) - dist)
                if sub:
                    best = min(sub, key=lambda p: p.total_length)
                    total = dist + best.total_length
                    full = np.vstack([lp[:-1], best.poses])
                    s0 = PathSegment(start, int1, 0., direction, dist, 1)
                    candidates.append(PathResult(
                        [s0]+best.segments, full,
                        best.n_maneuvers+1, total, True))
                    if best_total is None or total < best_total:
                        best_total = total


def plan_path(tpd, veh):
    return PathPlanner(veh).plan(tpd)


def path_summary(r):
    if not r.valid:
        return "未找到可行路径."
    lines = [f"规划成功: {r.n_maneuvers} 把, {r.total_length:.2f}m"]
    for s in r.segments:
        sp, ep = s.start_pose, s.end_pose
        lines.append(
            f"  第{s.num}把: {s.direction} R={s.radius:+.1f}m L={s.length:.2f}m "
            f"({sp[0]:.2f},{sp[1]:.2f},{np.degrees(sp[2]):.0f}°) → "
            f"({ep[0]:.2f},{ep[1]:.2f},{np.degrees(ep[2]):.0f}°)")
    return "\n".join(lines)
