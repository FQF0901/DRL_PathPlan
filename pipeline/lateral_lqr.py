#!/usr/bin/env python3
"""R41 第二批-F：**横向训练栈 / 固定 LQR 离线执行差距权重（P4 数据侧；非梯度）**。

规格出处
--------
- R41 复审（``docs/deepseekv4p1_argue.md`` §2 第 6 点）：LQR 可执行性重加权用
  **student−expert 执行差距** —— ``w ∝ max(0, sat_student − sat_expert)``，
  **不得**惩罚所有高 steer 样本（区分"场景本身需要高转向"与"计划抖动导致的不必要高转向"）；
- 权重口径 = 评测侧确定性 LQR 跟踪器（``env/tracking.LqrTracker``）的**横向反馈律**
  离线复刻：预瞄点姿态误差 ``e=[e_y, e_ψ]`` → ``δ = K(v)·e``，``sat = |δ|/δ_max``。

确定性/自包含
-------------
- 本模块为零依赖（numpy）的离线复刻：动力学 ``e' = A e − B δ``、
  ``A=[[1,v·dt],[0,1]]``、``B=[0; v·dt/L]``、``Q=diag(q_lat,q_head)``、``R=r_steer``、
  DARE 值迭代（固定迭代上限 + 容差）与速度分桶增益调度——**常量与
  ``env.tracking.LqrTracker`` 默认值一致**（守卫测试在 metadrive 可用时与
  ``env.tracking._solve_dare`` 对拍；本模块不 import metadrive）；
- 平面弧几何 = ``env/tracking.interpolate`` 的中点法常曲率弧（10 Hz 子步）复刻。

默认参数 = LqrTracker 构造默认（wheelbase 2.46894 = 1.05234+1.4166；δ_max = 40°）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LqrSaturationConfig:
    """离线 LQR 饱和读数参数（默认 = ``env.tracking.LqrTracker`` 默认）。"""

    wheelbase: float = 1.05234 + 1.4166
    max_steer_angle_rad: float = math.radians(40.0)
    lookahead_time: float = 0.8
    lookahead_base: float = 2.5
    lookahead_min: float = 3.0
    lookahead_max: float = 15.0
    q_lateral: float = 1.0
    q_heading: float = 6.0
    r_steer: float = 0.5
    steer_gain: float = 1.0
    reschedule_dv: float = 1.0
    riccati_iters: int = 200
    riccati_tol: float = 1e-12
    env_dt: float = 0.1
    reference_dt: float = 0.5
    reference_hz: int = 10

    def lookahead_m(self, speed: float) -> float:
        """预瞄距离 ``Ld = clip(0.8·v + 2.5, 3, 15)``（同 ``LqrTracker._lookahead``）。"""
        return float(min(max(self.lookahead_time * float(speed) + self.lookahead_base, self.lookahead_min), self.lookahead_max))


def dare_gain(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray, *, iters: int, tol: float) -> np.ndarray:
    """DARE 值迭代 → ``K = (R + BᵀPB)⁻¹ BᵀPA``（逐行复刻 ``env.tracking._solve_dare``）。"""
    p = np.array(q, dtype=np.float64, copy=True)
    k = np.zeros((r.shape[0], a.shape[0]), dtype=np.float64)
    for _ in range(int(iters)):
        s = r + b.T @ p @ b
        k = np.linalg.solve(s, b.T @ p @ a)
        p_next = q + a.T @ p @ a - a.T @ p @ b @ k
        if float(np.max(np.abs(p_next - p))) < float(tol):
            p = p_next
            break
        p = p_next
    return k


class SpeedGainSchedule:
    """按速度分桶的 LQR 增益（``LqrTracker._gain_for`` 口径；结果缓存 ⇒ 确定性）。"""

    def __init__(self, config: LqrSaturationConfig | None = None) -> None:
        self.config = config or LqrSaturationConfig()
        self._cache: dict[int, np.ndarray] = {}

    def gain(self, speed: float) -> np.ndarray:
        cfg = self.config
        bucket = int(round(max(abs(float(speed)), 0.5) / max(cfg.reschedule_dv, 1e-3)))
        cached = self._cache.get(bucket)
        if cached is not None:
            return cached
        v = max(bucket * max(cfg.reschedule_dv, 1e-3), 0.5)
        a = np.array([[1.0, v * cfg.env_dt], [0.0, 1.0]], dtype=np.float64)
        b = np.array([[0.0], [v * cfg.env_dt / cfg.wheelbase]], dtype=np.float64)
        q = np.diag([cfg.q_lateral, cfg.q_heading]).astype(np.float64)
        r = np.array([[cfg.r_steer]], dtype=np.float64)
        gain = dare_gain(a, b, q, r, iters=cfg.riccati_iters, tol=cfg.riccati_tol)
        self._cache[bucket] = gain
        return gain


def plan_polyline(plan: np.ndarray, config: LqrSaturationConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """计划 ``(P,6,2)`` ``(ds,dθ)`` → 10 Hz 参考位姿 ``(P,T,3)`` + 弧长 ``(P,T)``。

    几何 = 常曲率弧等时间采样（中点法；逐条复刻 ``env.tracking.interpolate``）：
    子步弦方向 ``θ + Δ/2``、弦长 ``2R·sin(Δ/2)``（``R = ds/dθ``；``|dθ|<eps`` ⇒ 直线段）。
    起点 = ego 系原点、theta=0。
    """
    cfg = config or LqrSaturationConfig()
    plan = np.asarray(plan, dtype=np.float64)
    if plan.ndim != 3 or plan.shape[2] != 2:
        raise SystemExit(f"[lateral] fail-fast：plan 形状应为 (P,K,2)，收到 {tuple(plan.shape)}")
    batch, steps = plan.shape[0], plan.shape[1]
    substeps = int(round(float(cfg.reference_dt) * int(cfg.reference_hz)))
    points = np.zeros((batch, steps * substeps, 3), dtype=np.float64)
    for row in range(batch):
        x = y = theta = 0.0
        out = 0
        for step in range(steps):
            ds, dtheta = float(plan[row, step, 0]), float(plan[row, step, 1])
            ds_sub, dtheta_sub = ds / substeps, dtheta / substeps
            for _ in range(substeps):
                if abs(dtheta_sub) < 1e-9 or abs(ds_sub) < 1e-9:
                    chord, direction = ds_sub, theta + 0.5 * dtheta_sub
                else:
                    radius = ds_sub / dtheta_sub
                    chord = 2.0 * radius * math.sin(0.5 * dtheta_sub)
                    direction = theta + 0.5 * dtheta_sub
                x += chord * math.cos(direction)
                y += chord * math.sin(direction)
                theta += dtheta_sub
                points[row, out, 0] = x
                points[row, out, 1] = y
                points[row, out, 2] = (theta + math.pi) % (2.0 * math.pi) - math.pi
                out += 1
    arc = np.zeros(points.shape[:2], dtype=np.float64)
    if points.shape[1] >= 2:
        deltas = np.linalg.norm(np.diff(points[:, :, :2], axis=1), axis=2)
        arc[:, 1:] = np.cumsum(deltas, axis=1)
    return points, arc


def preview_steer_raw(
    plan: np.ndarray,
    speed: np.ndarray,
    *,
    config: LqrSaturationConfig | None = None,
    schedule: SpeedGainSchedule | None = None,
) -> dict:
    """（P,6,2）计划 + 速度 → 预瞄 LQR 原始转向 ``δ`` 与饱和比 ``sat = |δ|/δ_max``。

    ego = 原点（计划起点自车系）；最近点 = 到原点的最小距离（离线无 cursor）；
    预瞄 = 最近点弧长 + ``Ld``（``searchsorted``），误差 ``e=[rel_y, wrap(rel_θ)]``，
    ``δ = K(v)·e·steer_gain``。返回 ``{"delta_raw", "sat", "steer_clipped", "lookahead_m",
    "preview_index", "error_y", "error_psi"}``（全部 (P,)）。
    """
    cfg = config or LqrSaturationConfig()
    schedule = schedule or SpeedGainSchedule(cfg)
    speed = np.asarray(speed, dtype=np.float64).reshape(-1)
    points, arc = plan_polyline(plan, cfg)
    if points.shape[0] != speed.shape[0]:
        raise SystemExit(
            f"[lateral] fail-fast：plan B={points.shape[0]} != speed B={speed.shape[0]}"
        )
    count = points.shape[0]
    delta = np.zeros(count, dtype=np.float64)
    lookahead = np.zeros(count, dtype=np.float64)
    preview = np.zeros(count, dtype=np.int64)
    error_y = np.zeros(count, dtype=np.float64)
    error_psi = np.zeros(count, dtype=np.float64)
    for row in range(count):
        dist = np.linalg.norm(points[row, :, :2], axis=1)
        nearest = int(np.argmin(dist))
        ld = cfg.lookahead_m(speed[row])
        target_arc = float(arc[row, nearest]) + ld
        index = int(np.searchsorted(arc[row], target_arc, side="left"))
        index = int(min(max(index, nearest), points.shape[1] - 1))
        pose = points[row, index]
        rel_y = float(pose[1])
        rel_psi = float((pose[2] + math.pi) % (2.0 * math.pi) - math.pi)
        gain = schedule.gain(float(speed[row]))
        error = np.array([rel_y, rel_psi], dtype=np.float64)
        delta[row] = float((gain @ error).reshape(-1)[0]) * float(cfg.steer_gain)
        lookahead[row] = ld
        preview[row] = index
        error_y[row] = rel_y
        error_psi[row] = rel_psi
    sat = np.abs(delta) / float(cfg.max_steer_angle_rad)
    steer = np.clip(delta / float(cfg.max_steer_angle_rad), -1.0, 1.0)
    return {
        "delta_raw": delta,
        "sat": sat,
        "steer_clipped": steer,
        "lookahead_m": lookahead,
        "preview_index": preview,
        "error_y": error_y,
        "error_psi": error_psi,
    }


def execution_gap_weights(
    sat_student: np.ndarray,
    sat_expert: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    clip_max: float = 1.0,
) -> dict:
    """``w ∝ max(0, sat_student − sat_expert)``（逐样本、非梯度）→ 归一化权重台账。

    归一化：``w = clip(gap / p95(gap | valid ∧ gap>0), 0, clip_max)``；无正 gap ⇒ 全 0。
    返回 ``{"gap", "w", "valid", "p95"}``（无 positive gap 时 ``w`` 全 0、``p95``=0）。
    """
    student = np.asarray(sat_student, dtype=np.float64).reshape(-1)
    expert = np.asarray(sat_expert, dtype=np.float64).reshape(-1)
    if student.shape != expert.shape:
        raise SystemExit(
            f"[lateral] fail-fast：sat_student 形状 {student.shape} != sat_expert 形状 {expert.shape}"
        )
    if valid is None:
        valid = np.ones(student.shape[0], dtype=bool)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    gap = np.maximum(0.0, student - expert)
    positive = valid & (gap > 0.0)
    if positive.any():
        scale = float(np.quantile(gap[positive], 0.95))
    else:
        scale = 0.0
    if scale > 0.0:
        weight = np.where(valid, np.clip(gap / scale, 0.0, float(clip_max)), 0.0)
    else:
        weight = np.zeros(student.shape[0], dtype=np.float64)
    return {"gap": gap, "w": weight, "valid": valid, "p95": scale}


__all__ = [
    "LqrSaturationConfig",
    "SpeedGainSchedule",
    "dare_gain",
    "execution_gap_weights",
    "plan_polyline",
    "preview_steer_raw",
]
