"""动作执行适配层（P2 契约 §1 / §8.5）：(Δs, Δθ) 圆弧插值 + Exact/LQR 两套跟踪器。

单一真源
--------
策略步动作 = ``(ds, dtheta)``：**下一 0.5 s 的弧长 + 航向变化**（2 维有界）。
:func:`interpolate` 是 ``(ds, dtheta) → 圆弧轨迹`` 的唯一实现：
每个 0.5 s 策略段内按**常曲率圆弧**推进，并以 0.1 s 子步（``hz=10``）输出位姿，因此
30 点参考在段内严格是同一段圆弧的等时间采样（不是"0.5 s 点之间直线插值"）。

- :class:`ExactTracker`（阶段 A）：每个 0.1 s 子步把 ego **运动学**置于插值位姿
  （写 position/heading/velocity/yaw-rate，不动动力学与控制器），同时驱动观测/奖励；
  位姿写入方式沿用 ``env/scenario/behaviors.py::WaypointPolicy``（0.4.3 无上游
  ``waypoint_policy``，已核对其 ``ReplayTrafficParticipantPolicy`` 式直写状态的做法）。
- :class:`LqrTracker`（阶段 C）：跟踪 10 Hz 参考点，输出 ``[steer, throttle]``，
  经 ``engine.add_policy`` 注册为 ego 策略（与 ``env/expert/pure_pursuit_idm.py`` 同路径）；
  横向 = 预瞄误差状态 ``[e_y, e_psi]`` 上的离散 LQR（Riccati 迭代，确定性），
  纵向上 = 参考速度的比例控制；前视距离/权重/增益均可配置。
- :func:`roundtrip_error`：§8.5 的 round-trip 校验——把稠密轨迹（10 Hz）反推成
  逐 0.5 s 动作，再用 :func:`interpolate` 重建并报误差（BC 数据采前的横向保真检查）。

坐标与角度约定
--------------
- 自车系 **x 前向 / y 左向**（``env/obs/base.py`` 顶部有源码级证据）；
- 角度一律 wrap 到 ``(-π, π]``（MetaDrive ``heading_theta`` 本身即 wrap，若返回未 wrap 的
  累积航向，与 ``set_heading_theta`` 的比较会差 2π，round-trip 断言会假失败）；
- 速度/位移单位 SI（m、m/s、rad）。
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.base_policy import BasePolicy
from metadrive.utils.math import clip, wrap_to_pi

__all__ = [
    "interpolate",
    "ego_to_world",
    "world_to_ego",
    "roundtrip_error",
    "ExactTracker",
    "LqrTracker",
]

#: 数值容差：弦长/转角小于该值时按直线段处理（避免 0/0）
_EPS = 1e-9


# ======================================================================================
# 基础几何
# ======================================================================================
def _as_point_array(values: Any, cols: int, name: str) -> np.ndarray:
    """把输入统一成 ``(N, cols)`` float64；空输入返回 ``(0, cols)``。"""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, cols), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != cols:
            raise ValueError(f"{name} 需要 (N,{cols})，收到 shape={arr.shape}")
        arr = arr.reshape(1, cols)
    if arr.ndim != 2 or arr.shape[1] != cols:
        raise ValueError(f"{name} 需要 (N,{cols})，收到 shape={arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含非有限值（NaN/Inf）")
    return arr


def interpolate(actions: Any, dt: float = 0.5, hz: int = 10) -> np.ndarray:
    """``(ds, dtheta)`` 序列 → 常曲率圆弧的 10 Hz 位姿序列 ``(N*dt*hz, 3)``。

    每个策略段 ``(ds, dtheta)`` 解释为：在 ``dt`` 秒内沿一段圆弧前进 ``ds`` 弧长、
    航向累计变化 ``dtheta``；段内按 ``1/hz`` 秒等时间切分，第 ``k`` 个子步的弦
    方向为 ``theta_k + delta/2``、弦长为 ``2R sin(delta/2)``（``R = ds/dtheta``，
    ``delta`` 为该子步的航向变化）——这正是圆弧的精确等时间采样（中点法），
    因此段内 C1 连续、且端点位移与转角严格一致。

    Args:
        actions: ``(N,2)`` 数组/列表，单位 m 与 rad（``dtheta`` 为相对量）。
        dt: 单个策略步时长（秒），默认 0.5。
        hz: 子步频率（Hz），默认 10（MetaDrive env.step = 0.1 s）。

    Returns:
        ``(N * dt * hz, 3)`` float64，``(x, y, theta)`` 位于**动作起点自车系**
        （起点为原点、theta=0）；``theta`` 已 wrap 到 ``(-π, π]``。

    Raises:
        ValueError: 形状/有限性非法，或 ``dt*hz`` 不是正整数。
    """
    seq = _as_point_array(actions, 2, "actions")
    dt = float(dt)
    hz = int(hz)
    if dt <= 0.0 or hz <= 0:
        raise ValueError(f"dt/hz 必须为正，收到 dt={dt}, hz={hz}")
    substeps_f = dt * hz
    substeps = int(round(substeps_f))
    if substeps < 1 or abs(substeps_f - substeps) > 1e-9:
        raise ValueError(f"dt*hz 必须是正整数个 0.1 s 子步，收到 dt={dt}, hz={hz}")
    if len(seq) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    out = np.zeros((len(seq) * substeps, 3), dtype=np.float64)
    x = y = 0.0
    theta = 0.0
    for seg_index, (ds, dtheta) in enumerate(seq):
        ds_sub = float(ds) / substeps
        dtheta_sub = float(dtheta) / substeps
        for k in range(substeps):
            if abs(dtheta_sub) < _EPS or abs(ds_sub) < _EPS:
                chord = ds_sub
                direction = theta + 0.5 * dtheta_sub
            else:
                radius = ds_sub / dtheta_sub  # 有符号半径；ds/dtheta 与子步同比例
                chord = 2.0 * radius * math.sin(0.5 * dtheta_sub)
                direction = theta + 0.5 * dtheta_sub
            x += chord * math.cos(direction)
            y += chord * math.sin(direction)
            theta = float(wrap_to_pi(theta + dtheta_sub))
            out[seg_index * substeps + k] = (x, y, theta)
    return out


def ego_to_world(points: Any, base_pose: Sequence[float]) -> np.ndarray:
    """自车系位姿 ``(N,3)`` → 世界系位姿（``base_pose = (x, y, theta)`` 为自车系原点/朝向）。"""
    arr = _as_point_array(points, 3, "points")
    bx, by, btheta = (float(base_pose[0]), float(base_pose[1]), float(base_pose[2]))
    cos_t, sin_t = math.cos(btheta), math.sin(btheta)
    out = np.array(arr, dtype=np.float64, copy=True)
    out[:, 0] = bx + arr[:, 0] * cos_t - arr[:, 1] * sin_t
    out[:, 1] = by + arr[:, 0] * sin_t + arr[:, 1] * cos_t
    out[:, 2] = wrap_to_pi(arr[:, 2] + btheta)
    return out


def world_to_ego(points: Any, base_pose: Sequence[float]) -> np.ndarray:
    """世界系位姿 ``(N,3)`` → 自车系（:func:`ego_to_world` 的逆变换）。"""
    arr = _as_point_array(points, 3, "points")
    bx, by, btheta = (float(base_pose[0]), float(base_pose[1]), float(base_pose[2]))
    cos_t, sin_t = math.cos(btheta), math.sin(btheta)
    dx = arr[:, 0] - bx
    dy = arr[:, 1] - by
    out = np.array(arr, dtype=np.float64, copy=True)
    out[:, 0] = dx * cos_t + dy * sin_t
    out[:, 1] = -dx * sin_t + dy * cos_t
    out[:, 2] = wrap_to_pi(arr[:, 2] - btheta)
    return out


def roundtrip_error(
    dense_poses: Any,
    *,
    dt: float = 0.5,
    hz: int = 10,
    max_segments: Optional[int] = None,
) -> dict:
    """§8.5 round-trip 校验：稠密轨迹 → 逐段 ``(ds, dtheta)`` → 重建 → 误差。

    反推口径（模型无关，先测"真实轨迹能否被圆弧模型复现"）：
    ``ds`` = 该 0.5 s 窗口内稠密点折线长度（真实弧长），
    ``dtheta`` = 窗口端点航向差（wrap）。

    Args:
        dense_poses: ``(T,3)`` 同一自车系下的 10 Hz 稠密位姿（t=0 为原点）。
        dt / hz: 与 :func:`interpolate` 相同。
        max_segments: 只校验前 N 个策略段（``None`` = 全部可整除部分）。

    Returns:
        含 ``actions (N,2)``、``reconstructed (N*sub,3)``、``actual (N*sub,3)``、
        ``err_xy``、``err_theta``、``ade``、``fde``、``max_xy``、``max_theta`` 的 dict。
    """
    poses = _as_point_array(dense_poses, 3, "dense_poses")
    substeps = int(round(float(dt) * int(hz)))
    n_segments = max(len(poses) - 1, 0) // substeps
    if max_segments is not None:
        n_segments = min(n_segments, int(max_segments))
    actions = np.zeros((n_segments, 2), dtype=np.float64)
    for seg in range(n_segments):
        window = poses[seg * substeps: (seg + 1) * substeps + 1]
        actions[seg, 0] = float(np.linalg.norm(np.diff(window[:, :2], axis=0), axis=1).sum())
        actions[seg, 1] = float(wrap_to_pi(window[-1, 2] - window[0, 2]))
    reconstructed = interpolate(actions, dt=dt, hz=hz)
    actual = poses[1: n_segments * substeps + 1]
    err_xy = np.linalg.norm(reconstructed[:, :2] - actual[:, :2], axis=1)
    err_theta = np.abs(wrap_to_pi(reconstructed[:, 2] - actual[:, 2]))
    return {
        "actions": actions,
        "reconstructed": reconstructed,
        "actual": actual,
        "err_xy": err_xy,
        "err_theta": err_theta,
        "ade": float(err_xy.mean()) if len(err_xy) else 0.0,
        "fde": float(err_xy[-1]) if len(err_xy) else 0.0,
        "max_xy": float(err_xy.max()) if len(err_xy) else 0.0,
        "max_theta": float(err_theta.max()) if len(err_theta) else 0.0,
    }


def _ego_vehicle(env: Any) -> Optional[BaseVehicle]:
    """兼容 env / engine 入参，取单智能体 ego（无则 None；不抛异常）。"""
    engine = getattr(env, "engine", env)
    if engine is None:
        return None
    manager = getattr(engine, "agent_manager", None)
    if manager is None:
        return None
    for attr in ("active_agents", "episode_created_agents"):
        agents = getattr(manager, attr, None)
        if isinstance(agents, dict) and agents:
            return next(iter(agents.values()))
    return None


def _ego_pose(vehicle: BaseVehicle) -> np.ndarray:
    """车辆世界位姿 ``(x, y, theta)``（theta wrap 到 (-π, π]）。"""
    position = np.asarray(vehicle.position, dtype=np.float64)
    return np.array(
        [float(position[0]), float(position[1]), float(wrap_to_pi(vehicle.heading_theta))],
        dtype=np.float64,
    )


# ======================================================================================
# 阶段 A：精确运动学跟踪
# ======================================================================================
class ExactTracker:
    """阶段 A 运动学执行器：每个 0.1 s 子步把 ego 置于插值位姿。

    为什么用"step 之后写状态"：
    MetaDrive 的 ``env.step`` = 决策（policy.act / ``before_step``）→ 物理积分 0.1 s →
    ``after_step``；在 ``before_step`` 写位姿仍会被随后的物理积分带偏。因此本类的
    标准用法是 **env.step 之后立即 :meth:`apply`**：先让物理/管理器走完一步，再把 ego
    瞬移到该子步的参考位姿（并写入参考速度/角速度），于是调用方随后构建的观测、标签、
    可视化看到的就是参考轨迹本身（round-trip 误差只来自浮点写入精度）。
    """

    def __init__(self, *, dt: float = 0.5, hz: int = 10):
        """
        Args:
            dt: 策略步时长（秒）。
            hz: 子步频率（Hz）；env.step = 1/hz 秒。
        """
        self.dt = float(dt)
        self.hz = int(hz)
        self.substeps = int(round(self.dt * self.hz))  # 需为整数（interpolate 会再校验）
        self.base_pose: Optional[np.ndarray] = None
        self.reference: Optional[np.ndarray] = None  # (N,3) 自车系（动作起点）
        self.index = 0

    # ------------------------------------------------------------------ 装配
    @property
    def n_points(self) -> int:
        return 0 if self.reference is None else int(self.reference.shape[0])

    @property
    def remaining(self) -> int:
        return max(self.n_points - self.index, 0)

    def arm(
        self,
        env: Any,
        actions: Any = None,
        *,
        points: Any = None,
        base_pose: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """装配参考并捕获自车系原点。

        Args:
            env: MetaDrive env（取 ego 位姿作 ``base_pose``）。
            actions: ``(N,2)`` ``(ds, dtheta)`` 序列（与 ``points`` 二选一）。
            points: ``(N,3)`` 自车系位姿序列（与 ``actions`` 二选一）。
            base_pose: 显式 ``(x, y, theta)``；``None`` 时取当前 ego 位姿。

        Returns:
            本次生效的自车系参考 ``(N,3)``。
        """
        if (actions is None) == (points is None):
            raise ValueError("ExactTracker.arm 需要且只需要 actions / points 之一")
        reference = interpolate(actions, dt=self.dt, hz=self.hz) if actions is not None else (
            _as_point_array(points, 3, "points")
        )
        reference = np.array(reference, dtype=np.float64, copy=True)
        if len(reference):
            reference[:, 2] = np.asarray(wrap_to_pi(reference[:, 2]), dtype=np.float64)
        if base_pose is None:
            vehicle = _ego_vehicle(env)
            if vehicle is None:
                raise RuntimeError("ExactTracker.arm: 无法获取 ego 位姿（env 未 reset?）")
            base_pose = _ego_pose(vehicle)
        self.base_pose = np.asarray(base_pose, dtype=np.float64).reshape(3)
        self.reference = reference  # 已 copy 且已 wrap，可直接持有
        self.index = 0
        # 世界系参考 + 逐点参考速度/角速度（相邻子步差分）
        dt_sub = 1.0 / float(self.hz)
        self._world = ego_to_world(self.reference, self.base_pose)
        origin = self.base_pose[:2]
        self._velocity = np.zeros((len(self._world), 2), dtype=np.float64)
        self._yaw_rate = np.zeros(len(self._world), dtype=np.float64)
        prev_xy = origin
        prev_theta = float(self.base_pose[2])
        for i, pose in enumerate(self._world):
            self._velocity[i] = (pose[:2] - prev_xy) / dt_sub
            self._yaw_rate[i] = float(wrap_to_pi(pose[2] - prev_theta)) / dt_sub
            prev_xy, prev_theta = pose[:2], pose[2]
        return self.reference

    # ------------------------------------------------------------------ 执行
    @staticmethod
    def set_pose(
        vehicle: BaseVehicle,
        position: Any,
        heading: float,
        velocity: Any = None,
        angular_velocity: float = 0.0,
    ) -> None:
        """把车辆运动学状态直写到给定位姿（``WaypointPolicy`` 的 set_pose 写法）。

        顺序与上游一致：position → heading → velocity → angular velocity。
        ``velocity`` 为世界系 2 维速度向量（``BaseVehicle.set_velocity`` 会写 ``last_velocity``）。
        """
        vehicle.set_position(np.asarray(position, dtype=np.float64)[:2])
        vehicle.set_heading_theta(float(heading))
        if velocity is None:
            velocity = np.zeros(2, dtype=np.float64)
        vehicle.set_velocity(np.asarray(velocity, dtype=np.float64)[:2])
        vehicle.set_angular_velocity(float(angular_velocity))

    def world_pose_at(self, index: int) -> np.ndarray:
        """第 ``index`` 个子步参考的世界位姿 ``(x, y, theta)``。"""
        if self.reference is None:
            raise RuntimeError("ExactTracker 未 arm")
        return np.array(self._world[int(index)], dtype=np.float64, copy=True)

    def apply(self, env: Any, index: Optional[int] = None) -> np.ndarray:
        """把 ego 置于第 ``index`` 个子步参考位姿；``index=None`` 时消费当前 cursor。

        Returns:
            从 env 读回的**实际**自车系位姿 ``(x, y, theta)``（用于 round-trip 断言）。
        """
        if self.reference is None or self.base_pose is None:
            raise RuntimeError("ExactTracker 未 arm")
        i = self.index if index is None else int(index)
        if not 0 <= i < self.n_points:
            raise IndexError(f"ExactTracker 子步越界：index={i}, n_points={self.n_points}")
        vehicle = _ego_vehicle(env)
        if vehicle is None:
            raise RuntimeError("ExactTracker.apply: 无法获取 ego")
        pose = self._world[i]
        self.set_pose(vehicle, pose[:2], float(pose[2]), self._velocity[i], float(self._yaw_rate[i]))
        if index is None:
            self.index += 1
        realized_world = _ego_pose(vehicle)
        return world_to_ego(realized_world.reshape(1, 3), self.base_pose)[0]

    def step(self, env: Any) -> np.ndarray:
        """:meth:`apply` 的别名（每调用一次前进一个 0.1 s 子步）。"""
        return self.apply(env)


# ======================================================================================
# 阶段 C：LQR 跟踪
# ======================================================================================
class LqrTracker(BasePolicy):
    """30 点参考 → ``[steer, throttle]`` 的确定性 LQR 跟踪器（阶段 C）。

    横向：以**预瞄点**（沿参考弧长前进 ``Ld = clip(lookahead_time*v + lookahead_base,
    lookahead_min, lookahead_max)``）的姿态误差 ``[e_y, e_psi]``（自车系 y 左正、航向
    wrap 差）为状态，反馈 ``delta = K · e``，``K`` 由离散 Riccati 迭代求解
    （动力学 ``e' = A e - B δ``，``A=[[1,v·dt],[0,1]]``、``B=[0; v·dt/L]``，
    ``Q=diag(q_lateral,q_heading)``、``R=r_steer``）；增益按速度分桶重算（确定性）。
    符号：``e_y>0``（参考在左）→ ``delta>0``（左转），与 MetaDrive ``action[0]>0=左``
    （``env/expert/pure_pursuit_idm.py`` 顶部三处源码证据）一致。

    纵向：参考点速度的前视比例控制 ``a = speed_gain·(v_ref - v)``，经
    ``accel/brake_action_scale`` 归一化为油门/刹车动作。参考短于 ``Ld`` 时预瞄点**钳制在
    参考末端**：`0.5 s` 单步参考（5 点）因此退化为"瞄准参考终点"的 4–5 m 前视控制。

    ``set_reference`` 每次策略步（0.5 s）调用一次：``(N,2)`` 动作或 ``(N,3)`` 自车系位姿。

    构造方式（两种都支持）：
    1. ``engine.add_policy(ego.id, LqrTracker, ego, seed)`` → 每个策略步 ``set_reference``
       （训练器 canonical 路径）；
    2. ``LqrTracker(ego, reference)``：第二个位置参数不是整数 seed 时视为**初始参考**
       （兼容 eval_runner 的构造约定，构造即捕获当前 ego 位姿为参考原点）。
    """

    DEFAULT_WHEELBASE = 1.05234 + 1.4166  # DefaultVehicle 前后轴距（vehicle_type.py:21-22）
    DEFAULT_MAX_STEERING_DEG = 40.0  # component/pg_space.py:232

    def __init__(
        self,
        control_object: BaseVehicle,
        random_seed: Optional[int] = None,
        config: Optional[dict] = None,
        *,
        wheelbase: Optional[float] = None,
        max_steer_angle_rad: Optional[float] = None,
        lookahead_time: float = 0.8,
        lookahead_base: float = 2.5,
        lookahead_min: float = 3.0,
        lookahead_max: float = 15.0,
        q_lateral: float = 1.0,
        q_heading: float = 6.0,
        r_steer: float = 0.5,
        steer_gain: float = 1.0,
        design_speed_mps: float = 10.0,
        reschedule_dv: float = 1.0,
        riccati_iters: int = 200,
        riccati_tol: float = 1e-12,
        speed_gain: float = 0.8,
        accel_action_scale: float = 3.0,
        brake_action_scale: float = 3.0,
        reference_dt: float = 0.5,
        reference_hz: int = 10,
    ) -> None:
        """
        Args:
            control_object: 被控车辆。
            random_seed / config: 与 ``BasePolicy`` 一致；本策略不用随机数。
            wheelbase: 轴距（m）；``None`` 时由车辆类常量推导。
            max_steer_angle_rad: action=±1 对应的前轮转角；``None`` 时由 ``max_steering`` 推导。
            lookahead_time / lookahead_base / lookahead_min / lookahead_max: 预瞄距离参数。
            q_lateral / q_heading / r_steer: LQR 权重（横向误差/航向误差/转向代价）。
            steer_gain: 转向总增益（1.0 = 纯 LQR）。
            design_speed_mps: 增益调度的标称速度（实际按 ``reschedule_dv`` 分桶重算）。
            reschedule_dv: 速度分桶宽度（m/s）；越大越少重算，越小越贴近当前速度。
            riccati_iters / riccati_tol: Riccati 值迭代上限/收敛阈值（固定迭代 → 确定性）。
            speed_gain: 纵向速度误差比例增益（1/s）。
            accel_action_scale / brake_action_scale: 加速度/减速度 → 动作的归一化尺度（m/s²）。
            reference_dt / reference_hz: 参考动作步长与子步频率（与 :func:`interpolate` 一致）。
        """
        # 兼容 eval_runner 构造约定：第二个位置参数是数组/列表时视为初始参考（见类 docstring）
        initial_reference: Optional[np.ndarray] = None
        if random_seed is not None and np.ndim(random_seed) >= 1:
            initial_reference = np.asarray(random_seed, dtype=np.float64)
            random_seed = None
        super().__init__(control_object=control_object, random_seed=random_seed, config=config)
        self.wheelbase = float(wheelbase) if wheelbase is not None else self._infer_wheelbase(control_object)
        self.max_steer_angle_rad = (
            float(max_steer_angle_rad)
            if max_steer_angle_rad is not None else self._infer_max_steer_angle_rad(control_object)
        )
        self.lookahead_time = float(lookahead_time)
        self.lookahead_base = float(lookahead_base)
        self.lookahead_min = float(lookahead_min)
        self.lookahead_max = float(lookahead_max)
        self.q_lateral = float(q_lateral)
        self.q_heading = float(q_heading)
        self.r_steer = float(r_steer)
        self.steer_gain = float(steer_gain)
        self.design_speed_mps = float(design_speed_mps)
        self.reschedule_dv = float(reschedule_dv)
        self.riccati_iters = int(riccati_iters)
        self.riccati_tol = float(riccati_tol)
        self.speed_gain = float(speed_gain)
        self.accel_action_scale = float(accel_action_scale)
        self.brake_action_scale = float(brake_action_scale)
        self.reference_dt = float(reference_dt)
        self.reference_hz = int(reference_hz)

        assert self.wheelbase > 0.0, "wheelbase must be positive"
        assert self.max_steer_angle_rad > 0.0, "max_steer_angle_rad must be positive"
        assert 0.0 < self.lookahead_min <= self.lookahead_max
        assert 0.0 < self.q_lateral and 0.0 < self.q_heading and 0.0 < self.r_steer
        assert 0.0 < self.steer_gain
        assert 0.0 < self.accel_action_scale and 0.0 < self.brake_action_scale
        assert 0.0 <= self.reschedule_dv
        assert self.riccati_iters >= 1

        self._ref_world: Optional[np.ndarray] = None  # (N,3) 世界系
        self._ref_arc: Optional[np.ndarray] = None  # (N,) 距原点的累计弧长
        self._ref_speed: Optional[np.ndarray] = None  # (N,) 参考速度 m/s
        self._cursor = 0
        self._gain_cache: dict[int, np.ndarray] = {}
        if initial_reference is not None:  # eval_runner 式构造：构造即 armed
            self.set_reference(initial_reference)

    # ------------------------------------------------------------------ 参考
    def set_reference(self, reference: Any, *, base_pose: Optional[Sequence[float]] = None) -> np.ndarray:
        """设置/刷新参考轨迹（自车系，起点为原点）。

        Args:
            reference: ``(N,2)`` 动作（``ds, dtheta``）或 ``(N,3)`` 自车系位姿。
            base_pose: 参考所依附的世界位姿；``None`` 时取当前 ego 位姿。

        Returns:
            生效的世界系参考位姿 ``(N,3)``。
        """
        arr = np.asarray(reference, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("reference 不能为空")
        if arr.ndim != 2 or arr.shape[1] not in (2, 3):
            raise ValueError(f"reference 需要 (N,2) 动作或 (N,3) 位姿，收到 shape={arr.shape}")
        points = (
            interpolate(arr, dt=self.reference_dt, hz=self.reference_hz)
            if arr.shape[1] == 2 else _as_point_array(arr, 3, "reference")
        )
        if base_pose is None:
            base_pose = _ego_pose(self.control_object)
        base = np.asarray(base_pose, dtype=np.float64).reshape(3)
        self._ref_world = ego_to_world(points, base)
        dt_sub = 1.0 / float(self.reference_hz)
        prev = np.vstack([base[:2], self._ref_world[:-1, :2]])
        chords = np.linalg.norm(self._ref_world[:, :2] - prev, axis=1)
        self._ref_speed = chords / dt_sub
        self._ref_arc = np.cumsum(chords)
        self._cursor = 0
        return self._ref_world

    @property
    def reference_world(self) -> Optional[np.ndarray]:
        """当前参考（世界系，``(N,3)``）；未设置时为 None。"""
        return self._ref_world

    # ------------------------------------------------------------------ 主入口
    def act(self, *args: Any, **kwargs: Any) -> list:
        """输出 ``[steer, throttle]``（引擎在 ``before_step`` 阶段调用）。"""
        ref = self._ref_world
        if ref is None or len(ref) == 0:
            action = [0.0, 0.0]
            self.action_info["action"] = action
            self.action_info["lqr_status"] = "no_reference"
            return action

        ego = self.control_object
        pos = np.asarray(ego.position, dtype=np.float64)[:2]
        theta = float(wrap_to_pi(ego.heading_theta))
        speed = float(ego.speed)

        preview = self._preview_index(pos, speed)
        pose = ref[preview]
        rel = world_to_ego(pose.reshape(1, 3), np.array([pos[0], pos[1], theta]))[0]
        error = np.array([float(rel[1]), float(rel[2])], dtype=np.float64)
        gain = self._gain_for(speed)
        delta = float(gain @ error) * self.steer_gain
        steer = float(clip(delta / self.max_steer_angle_rad, -1.0, 1.0))

        accel = self.speed_gain * (float(self._ref_speed[preview]) - speed)
        scale = self.accel_action_scale if accel >= 0.0 else self.brake_action_scale
        throttle = float(clip(accel / scale, -1.0, 1.0))

        action = [steer, throttle]
        self.action_info.update(
            {
                "action": action,
                "lqr_status": "tracking",
                "lqr_preview_index": int(preview),
                "lqr_error_y": float(error[0]),
                "lqr_error_psi": float(error[1]),
                "lqr_lookahead_m": float(self._lookahead(speed)),
                "lqr_ref_speed_mps": float(self._ref_speed[preview]),
            }
        )
        return action

    # ------------------------------------------------------------------ 内部
    def _lookahead(self, speed: float) -> float:
        return float(
            clip(self.lookahead_time * speed + self.lookahead_base, self.lookahead_min, self.lookahead_max)
        )

    def _preview_index(self, position: np.ndarray, speed: float) -> int:
        """最近点（cursor 起单调前搜）→ 沿弧长前进 ``Ld`` 的预瞄点索引。"""
        assert self._ref_world is not None and self._ref_arc is not None
        n = len(self._ref_world)
        start = int(min(max(self._cursor, 0), n - 1))
        dist = np.linalg.norm(self._ref_world[start:, :2] - position, axis=1)
        nearest = start + int(np.argmin(dist))
        self._cursor = nearest
        target_arc = float(self._ref_arc[nearest]) + self._lookahead(speed)
        index = int(np.searchsorted(self._ref_arc, target_arc, side="left"))
        return int(min(max(index, nearest), n - 1))

    def _gain_for(self, speed: float) -> np.ndarray:
        """按速度分桶的 LQR 增益（``_solve_dare``，结果缓存 → 确定性）。"""
        bucket = int(round(max(abs(float(speed)), 0.5) / max(self.reschedule_dv, 1e-3)))
        cached = self._gain_cache.get(bucket)
        if cached is not None:
            return cached
        v = max(bucket * max(self.reschedule_dv, 1e-3), 0.5)
        dt = self._env_dt()
        a = np.array([[1.0, v * dt], [0.0, 1.0]], dtype=np.float64)
        b = np.array([[0.0], [v * dt / self.wheelbase]], dtype=np.float64)
        q = np.diag([self.q_lateral, self.q_heading]).astype(np.float64)
        r = np.array([[self.r_steer]], dtype=np.float64)
        gain = _solve_dare(a, b, q, r, self.riccati_iters, self.riccati_tol)
        self._gain_cache[bucket] = gain
        return gain

    def _env_dt(self) -> float:
        """env.step 的物理步长（默认 0.02×5=0.1 s；读配置失败回退 0.1）。"""
        try:
            config = self.control_object.config
            dt = float(config["physics_world_step_size"]) * int(config["decision_repeat"])
            return dt if dt > 0.0 else 0.1
        except Exception:  # noqa: BLE001 - 车辆 config 不可读时用平台默认
            return 0.1

    def params(self) -> dict:
        """生效参数快照（写入评测 JSON / 日志，保证可追溯）。"""
        return {
            "wheelbase": self.wheelbase,
            "max_steer_angle_rad": self.max_steer_angle_rad,
            "lookahead_time": self.lookahead_time,
            "lookahead_base": self.lookahead_base,
            "lookahead_min": self.lookahead_min,
            "lookahead_max": self.lookahead_max,
            "q_lateral": self.q_lateral,
            "q_heading": self.q_heading,
            "r_steer": self.r_steer,
            "steer_gain": self.steer_gain,
            "design_speed_mps": self.design_speed_mps,
            "reschedule_dv": self.reschedule_dv,
            "speed_gain": self.speed_gain,
            "accel_action_scale": self.accel_action_scale,
            "brake_action_scale": self.brake_action_scale,
            "reference_dt": self.reference_dt,
            "reference_hz": self.reference_hz,
        }

    @classmethod
    def _infer_wheelbase(cls, control_object: Any) -> float:
        front = getattr(control_object, "FRONT_WHEELBASE", None)
        rear = getattr(control_object, "REAR_WHEELBASE", None)
        if front and rear:
            return float(front) + float(rear)
        return cls.DEFAULT_WHEELBASE

    @classmethod
    def _infer_max_steer_angle_rad(cls, control_object: Any) -> float:
        try:
            deg = float(control_object.config["max_steering"])
        except (AttributeError, KeyError, TypeError, ValueError):
            deg = cls.DEFAULT_MAX_STEERING_DEG
        return math.radians(deg) if deg > 0.0 else math.radians(cls.DEFAULT_MAX_STEERING_DEG)


def _solve_dare(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    iters: int,
    tol: float,
) -> np.ndarray:
    """离散代数 Riccati 值迭代 → 最优反馈增益 ``K = (R + BᵀPB)⁻¹ BᵀPA``。

    固定迭代上限 + 收敛阈值：结果与迭代次数无关（同一输入必然同一增益），
    满足"确定性优先"（契约 §7）。
    """
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
