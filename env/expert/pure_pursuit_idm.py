"""纯跟踪（pure pursuit）+ IDM 规则基线专家（KPI 基线 / BC 备选数据源）。

用途
----
1. 作为 P1a 的 **KPI 基线**：``tools/baseline_eval.py`` 用它批量跑场景，产出
   route_completion / 碰撞率 / 出界率 / 舒适性 / 速度比等指标；
2. 作为后续 BC 的备选数据源：动作口径与 RL 策略完全一致（``[steer, throttle]``）。

接口
----
``PurePursuitIDMPolicy(BasePolicy)``
- ``act() -> [steer, throttle]``，两维均在 ``[-1, 1]``（MetaDrive 连续动作空间，
  ``metadrive/policy/base_policy.py:86``）；
- 横向：以自车当前车道中心线为锚，沿导航路线（``current_ref_lanes`` →
  ``next_ref_lanes`` → 全图车道兜底）采样前视点，按纯跟踪律算期望前轮转角并归一化；
- 纵向：IDM 跟车（含前车速度差）+ 限速约束，输出归一化油门/刹车；
- 所有关键参数可配置；``None`` 表示“从被控车辆 / MetaDrive 配置推导默认值”。
  本类无随机数、无积分器状态，同一状态 + 同一参数必然给出同一动作（可复现基线）。

与 MetaDrive 0.4.3 已安装源码核对过的关键事实（file:line）
----------------------------------------------------------
- 动作 → 物理：``BaseVehicle._set_action`` 把 ``action[0]`` 乘以 ``max_steering``
  作为前轮转角（``component/vehicle/base_vehicle.py:473-481``）；平台默认
  ``max_steering=40``（``component/pg_space.py:232``），语义为 “action=±1 时的最大
  转向角（度）”（``envs/varying_dynamics_env.py:22`` 注释、增量转向上限 0.05/步
  亦以 [-1,1] 为归一化刻度，``base_vehicle.py:110``）。因此本策略把期望前轮转角
  （rad）除以 ``radians(max_steering)`` 得到归一化动作。
- 转向正负：``action[0] > 0`` 表示**左转**（世界航向角增大）。证据：键盘控制 a 键
  使 steering 增大（``engine/core/manual_controller.py:78-86``）；``PIDController``
  输出带负号（``component/vehicle/PID_controller.py:15-17``），官方 IDMPolicy 的
  ``pid(-Δheading)`` 对“需要左转”给出正输出（``policy/idm_policy.py:293-301``）；
  变道策略里 ``steering=+1`` 选中序号更小的车道（即左车道，
  ``policy/lange_change_policy.py:36-41`` + ``node_road_network.py:309-310`` 的
  左邻车道定义）。
- 速度单位：``ego.speed`` 为 m/s（``base_class/base_object.py:354-361``）。
- 限速单位：本项目统一用 **m/s**（``env/metadrive_env.py:24-26`` 明确“统一用 m/s 写入/读取”，
  L1a 的 spec ``limits`` 亦为 m/s）。MetaDrive 上游自身不一致——``overspeed``/``ScenarioLane``
  按 km/h（``base_vehicle.py:950-952``、``component/lane/scenario_lane.py:35-45``），PG
  ramp/tollgate 按 m/s 传入（``component/pgblock/ramp.py:33``、``tollgate.py:19``）。
  因此本模块用 :func:`lane_speed_limit_mps` 统一折算，``speed_limit_units`` 默认 ``"mps"``
  （项目口径），``"auto"``（按量级判定）与 ``"kmh"`` 可选。未设置（默认 1000，
  ``component/lane/abs_lane.py:22``）时用 ``fallback_speed_limit_mps``。
- 位姿：``ego.position``、``ego.heading_theta`` 为前进方向世界角
  （``base_object.py:307-315, 383-393``），世界系为右手系
  （``utils/coordinates_shift.py``）。
- 前车检索：直接复用官方 ``FrontBackObjects``（``policy/idm_policy.py:82-132``），
  与官方 IDM 同口径（中心距 + 前车速度投影）。
- 车道链：``lane.is_previous_lane_of``（``component/lane/abs_lane.py:84-89``）、
  ``road_network.get_all_lanes``（``node_road_network.py:325-334``）。
- ``navigation.get_checkpoints()`` 返回**世界坐标**：``base_navigation.py:153-160``
  内部调用 ``ref_lane.position(ref_lane.length, ...)``，而各 Lane 的 ``position()``
  返回世界坐标（``straight_lane.py:60-61`` 等）。仅作车道链不可用时的兜底。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from metadrive.component.buildings.base_building import BaseBuilding
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.policy.base_policy import BasePolicy
from metadrive.policy.idm_policy import FrontBackObjects
from metadrive.utils.math import clip, wrap_to_pi

__all__ = ["PurePursuitIDMPolicy", "lane_speed_limit_mps", "SPEED_LIMIT_KMH_THRESHOLD"]

# 二维点/向量（世界坐标或自车系，函数名/注释会指明）以 np.ndarray 表示，
# 统一由 _to_point 归一化。

# “auto” 单位判定阈值：项目限速（8.3~13.9 m/s）与 km/h 表达（30~50 km/h）量级天然可分；
# PG ramp/tollgate 的 12/3（m/s 语义）也会落到 m/s 一侧。
SPEED_LIMIT_KMH_THRESHOLD: float = 25.0

# MetaDrive 表示“未设置限速”的哨兵值（component/lane/abs_lane.py:22）
UNSET_SPEED_LIMIT: float = 1000.0


def _to_point(pos: Any) -> np.ndarray:
    """把 position / Vec3 / Vector / tuple 统一取成 shape=(2,) 的 float 数组。"""
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    return arr[:2].copy()


def _lane_key(lane: Any) -> Optional[str]:
    """车道诊断标识：``index``（如 ``('0S0', '0S1', 0)``）的紧凑字符串；无车道返回 None。"""
    if lane is None:
        return None
    index = getattr(lane, "index", None)
    if index is None:
        return f"<{type(lane).__name__}>"
    return str(tuple(index))


def lane_speed_limit_mps(
    lane: Any,
    *,
    units: str = "mps",
    fallback_mps: float = 13.9,
) -> float:
    """把 ``lane.speed_limit`` 统一折算成 m/s（策略与评测共用同一口径）。

    为什么需要它：MetaDrive 对 ``speed_limit`` 的单位并不统一（见模块 docstring），
    而本项目（``env/metadrive_env.py`` 与 L1a spec）统一用 m/s 写入。``units="mps"``
    为项目口径；``"kmh"`` 用于 ScenarioLane/``overspeed`` 语义；``"auto"`` 按数值量级
    判定（>25 视为 km/h）。

    Args:
        lane: 车道对象（可为 None）。
        units: ``"mps" | "auto" | "kmh"``。
        fallback_mps: 未设置（>=1000）或非法时的兜底限速（m/s）。

    Returns:
        限速（m/s），恒为正数。
    """
    limit = 0.0
    if lane is not None:
        try:
            limit = float(lane.speed_limit)
        except (AttributeError, TypeError, ValueError):
            limit = 0.0
    if not (0.0 < limit < UNSET_SPEED_LIMIT):
        return float(fallback_mps)
    if units == "kmh":
        return limit / 3.6
    if units == "mps":
        return limit
    return limit / 3.6 if limit > SPEED_LIMIT_KMH_THRESHOLD else limit


class PurePursuitIDMPolicy(BasePolicy):
    """横向纯跟踪 + 纵向 IDM 的规则基线专家。

    参数默认值全部来自 MetaDrive 自身车辆常量与 IDM 常用值，frozen 后不应在评测中
    随机化（保证基线可复现）。逐项含义见 ``__init__`` 与 ``params()``。
    """

    # ---- 车辆相关默认值（来自 MetaDrive 自身常量）----
    # DefaultVehicle 的 FRONT_WHEELBASE + REAR_WHEELBASE（component/vehicle/vehicle_type.py:21-22）
    DEFAULT_WHEELBASE: float = 1.05234 + 1.4166
    # 平台默认 max_steering=40（component/pg_space.py:232），单位：度
    DEFAULT_MAX_STEERING_DEG: float = 40.0
    DEFAULT_MAX_STEER_ANGLE_RAD: float = math.radians(DEFAULT_MAX_STEERING_DEG)

    # 车道链最多向前推进的段数（覆盖路口/短路段，防止病态地图死循环）
    MAX_CHAIN_SEGMENTS: int = 8
    # 跟车最小净距（m），防止 gap→0 时 IDM 项发散
    IDM_MIN_GAP: float = 0.5
    # 前视点已在自车身后时不再转向的阈值（m）
    MIN_LOOKAHEAD_FORWARD: float = 0.05
    # 认为两段车道“共线相接”的端点距离阈值（m）；超过则视为横向错位/车道数变化
    LANE_CONNECT_TOL: float = 0.5
    # 后继车道允许的最大航向突变（rad）：防止把对向车道选成后继（bidirection 反向车道
    # 与正向车道共享同一段中心线，纯几何端点相接会把行车方向掉头）
    MAX_SUCC_HEADING_DIFF: float = math.radians(100.0)

    def __init__(
        self,
        control_object: BaseVehicle,
        random_seed: Optional[int] = None,
        config: Optional[dict] = None,
        *,
        wheelbase: Optional[float] = None,
        max_steer_angle_rad: Optional[float] = None,
        steer_gain: float = 1.0,
        lookahead_time: float = 1.0,
        lookahead_base: float = 3.0,
        lookahead_min: float = 5.0,
        lookahead_max: float = 20.0,
        idm_desired_speed_kmh: Optional[float] = None,
        idm_a_max: float = 1.5,
        idm_b_comfort: float = 2.0,
        idm_delta: float = 4.0,
        idm_time_wanted: float = 1.5,
        idm_distance_wanted: float = 2.0,
        idm_detection_range: float = 50.0,
        accel_action_scale: float = 3.0,
        brake_action_scale: float = 3.0,
        fallback_speed_limit_mps: float = 13.9,
        speed_limit_units: str = "mps",
        lane_change_plan_time: float = 3.0,
        lane_change_plan_base: float = 12.0,
        lane_change_plan_min: float = 25.0,
        lane_change_plan_max: float = 80.0,
        lane_change_safety_gap: float = 10.0,
        lateral_correction_gain: float = 0.5,
        static_obstacle_range: float = 40.0,
        static_obstacle_lat_margin: float = 1.0,
    ) -> None:
        """
        Args:
            control_object: 被控车辆（``BaseVehicle``）。
            random_seed: 与 ``BasePolicy`` 一致；本策略不使用随机数。
            config: 与 ``BasePolicy`` 一致的可选配置。
            wheelbase: 轴距（m）；``None`` 时从车辆类常量推导，否则用 DefaultVehicle 值。
            max_steer_angle_rad: action=±1 对应的前轮转角（rad）；``None`` 时由车辆
                ``config["max_steering"]``（度）推导。
            steer_gain: 横向总增益（1.0 = 几何纯跟踪；>1 更激进，<1 更保守）。
            lookahead_time / lookahead_base / lookahead_min / lookahead_max: 前视距离
                ``Ld = clip(lookahead_time * v + lookahead_base, min, max)``（m）。
            idm_desired_speed_kmh: 显式目标速度（km/h）；``None`` 时按车道限速取
                （单位口径见 ``speed_limit_units``）。
            idm_a_max: IDM 最大加速度 ``a``（m/s²）。
            idm_b_comfort: IDM 舒适减速度 ``b``（m/s²）。
            idm_delta: IDM 速度指数。
            idm_time_wanted: 期望车头时距 ``T``（s）。
            idm_distance_wanted: 期望静止净距 ``s0``（m）。
            idm_detection_range: 前车搜索距离（m）。
            accel_action_scale: 正加速度 → 油门动作的归一化尺度（m/s²）。
            brake_action_scale: 负加速度 → 刹车动作的归一化尺度（m/s²）。
            fallback_speed_limit_mps: 车道未设置限速（MetaDrive 默认 1000）时的兜底限速
                （m/s，默认 13.9 ≈ 50 km/h，取自契约里高速直道的量级）。
            speed_limit_units: 车道 ``speed_limit`` 的单位口径：``"mps"``（项目口径，默认）、
                ``"auto"``（按量级判定）、``"kmh"``；见 :func:`lane_speed_limit_mps`。
            lane_change_plan_time / lane_change_plan_base / lane_change_plan_min / lane_change_plan_max:
                提前选目标车道的规划距离 ``d = clip(plan_time * v + plan_base, min, max)``（m）。
                当自车距当前路由段末端小于 ``d`` 时，沿 ``current_ref_lanes → next_ref_lanes``
                路由链预选下一段的目标车道；若该目标与当前车道不共线（车道数变化 / 横向错位），
                先瞄准路由内最接近的“汇入车道”，避免到路口末端才做一次不可实现的横向跳变。
            lane_change_safety_gap: 变道前对目标车道前后车的安全间距（m）；不满足则暂缓变道。
            lateral_correction_gain: 横向偏差修正增益（叠加在纯跟踪之外，0=关闭）。
                纯跟踪在持续曲率下存在与曲率成正比的稳态内切偏差（见模块诊断注释），
                该增益按 ``-gain * atan(2L·e/Ld²)`` 补偿（e 为自车相对锚车道中心线的横向偏差）。
            static_obstacle_range: 静态障碍（如收费站岗亭）的搜索距离（m）。
            static_obstacle_lat_margin: 判定“障碍在车道内”的横向余量（m）。
        """
        super().__init__(control_object=control_object, random_seed=random_seed, config=config)

        self.wheelbase = float(wheelbase) if wheelbase is not None else self._infer_wheelbase(control_object)
        self.max_steer_angle_rad = (
            float(max_steer_angle_rad)
            if max_steer_angle_rad is not None else self._infer_max_steer_angle_rad(control_object)
        )
        self.steer_gain = float(steer_gain)
        self.lookahead_time = float(lookahead_time)
        self.lookahead_base = float(lookahead_base)
        self.lookahead_min = float(lookahead_min)
        self.lookahead_max = float(lookahead_max)
        self.idm_desired_speed_kmh = None if idm_desired_speed_kmh is None else float(idm_desired_speed_kmh)
        self.idm_a_max = float(idm_a_max)
        self.idm_b_comfort = float(idm_b_comfort)
        self.idm_delta = float(idm_delta)
        self.idm_time_wanted = float(idm_time_wanted)
        self.idm_distance_wanted = float(idm_distance_wanted)
        self.idm_detection_range = float(idm_detection_range)
        self.accel_action_scale = float(accel_action_scale)
        self.brake_action_scale = float(brake_action_scale)
        self.fallback_speed_limit_mps = float(fallback_speed_limit_mps)
        self.speed_limit_units = str(speed_limit_units)
        self.lane_change_plan_time = float(lane_change_plan_time)
        self.lane_change_plan_base = float(lane_change_plan_base)
        self.lane_change_plan_min = float(lane_change_plan_min)
        self.lane_change_plan_max = float(lane_change_plan_max)
        self.lane_change_safety_gap = float(lane_change_safety_gap)
        self.lateral_correction_gain = float(lateral_correction_gain)
        self.static_obstacle_range = float(static_obstacle_range)
        self.static_obstacle_lat_margin = float(static_obstacle_lat_margin)

        # 参数自检：错误的默认值会让基线“静默变差”，这里直接失败
        assert self.wheelbase > 0.0, "wheelbase must be positive"
        assert self.max_steer_angle_rad > 0.0, "max_steer_angle_rad must be positive"
        assert 0.0 < self.steer_gain
        assert 0.0 < self.lookahead_min <= self.lookahead_max
        assert 0.0 < self.idm_a_max and 0.0 < self.idm_b_comfort and 0.0 < self.idm_delta
        assert 0.0 < self.accel_action_scale and 0.0 < self.brake_action_scale
        assert 0.0 < self.idm_detection_range
        assert self.fallback_speed_limit_mps > 0.0
        assert self.speed_limit_units in ("auto", "mps", "kmh")
        assert 0.0 <= self.lane_change_plan_base <= self.lane_change_plan_min <= self.lane_change_plan_max
        assert 0.0 <= self.lane_change_safety_gap
        assert 0.0 <= self.lateral_correction_gain
        assert 0.0 <= self.static_obstacle_range and 0.0 <= self.static_obstacle_lat_margin

    # ------------------------------------------------------------------ 生命周期

    def reset(self) -> None:
        """清空诊断信息。

        引擎在 episode 重置时**不会**自动调用策略 ``reset``（``engine/base_engine.py:316``
        只重置 managers），因此评测/训练脚本需在每次 reset 后显式调用。本策略无
        积分器状态，重复调用无副作用。
        """
        super().reset()

    def params(self) -> Dict[str, Any]:
        """导出生效参数快照（写入评测 JSON 的 meta，保证基线可追溯）。"""
        return {
            "wheelbase": self.wheelbase,
            "max_steer_angle_rad": self.max_steer_angle_rad,
            "steer_gain": self.steer_gain,
            "lookahead_time": self.lookahead_time,
            "lookahead_base": self.lookahead_base,
            "lookahead_min": self.lookahead_min,
            "lookahead_max": self.lookahead_max,
            "idm_desired_speed_kmh": self.idm_desired_speed_kmh,
            "idm_a_max": self.idm_a_max,
            "idm_b_comfort": self.idm_b_comfort,
            "idm_delta": self.idm_delta,
            "idm_time_wanted": self.idm_time_wanted,
            "idm_distance_wanted": self.idm_distance_wanted,
            "idm_detection_range": self.idm_detection_range,
            "accel_action_scale": self.accel_action_scale,
            "brake_action_scale": self.brake_action_scale,
            "fallback_speed_limit_mps": self.fallback_speed_limit_mps,
            "speed_limit_units": self.speed_limit_units,
            "lane_change_plan_time": self.lane_change_plan_time,
            "lane_change_plan_base": self.lane_change_plan_base,
            "lane_change_plan_min": self.lane_change_plan_min,
            "lane_change_plan_max": self.lane_change_plan_max,
            "lane_change_safety_gap": self.lane_change_safety_gap,
            "lateral_correction_gain": self.lateral_correction_gain,
            "static_obstacle_range": self.static_obstacle_range,
            "static_obstacle_lat_margin": self.static_obstacle_lat_margin,
        }

    # ------------------------------------------------------------------ 主入口

    def act(self, *args: Any, **kwargs: Any) -> List[float]:
        """算一步动作 ``[steer, throttle]``（引擎在 ``before_step`` 阶段调用）。

        为什么不用 PID/积分器：基线要求确定、可复现且动作口径统一；纯跟踪 + IDM
        都是无状态解析式，跨场景行为稳定，不需要调参积分器。
        """
        ego = self.control_object
        nav = getattr(ego, "navigation", None)
        ref_lanes = getattr(nav, "current_ref_lanes", None) if nav is not None else None
        if nav is None or not ref_lanes:
            # 导航尚未就绪（reset 初期 / 无导航模块）：安全直行，避免抛异常打断评测
            action = [0.0, 0.0]
            self.action_info["action"] = action
            return action

        # 诊断信息（车道链/前视点/锚车道）随 action_info 透出，供 --debug 轨迹定位问题
        diag: Dict[str, Any] = {}
        point, lookahead_dist = self._sample_lookahead(nav, ego, diag)
        if point is None:
            steer, alpha = 0.0, 0.0
        else:
            steer, alpha = self._lateral_action(ego, point, lookahead_dist, diag)
        throttle, long_diag = self._longitudinal_action(ego)

        action = [float(steer), float(throttle)]
        self.action_info["action"] = action
        self.action_info.update(long_diag)
        self.action_info.update(diag)
        self.action_info["pp_lookahead_m"] = float(lookahead_dist)
        self.action_info["pp_alpha_rad"] = float(alpha)
        if point is not None:
            self.action_info["pp_lookahead_xy"] = [float(point[0]), float(point[1])]
        return action

    # ------------------------------------------------------------------ 横向：纯跟踪

    def _sample_lookahead(
        self, nav: Any, ego: BaseVehicle, diag: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[np.ndarray], float]:
        """沿导航路线前视采样，返回 ``(世界坐标前视点, 前视距离[m])``。

        先选“目标车道”（见 :meth:`_plan_target_lane`：严格锚定
        ``current_ref_lanes → next_ref_lanes`` 路由链，并在接近路段末端时提前换到
        汇入/后继车道；遇到静态障碍则换到空闲车道），再沿后继车道链按弧长推进采样。
        ``diag`` 非空时写入锚车道 / 目标车道 / 路由车道 / 链 / 前视点等诊断字段。
        """
        lane = self._anchor_lane(nav, ego)
        v = max(float(ego.speed), 0.0)
        # 速度自适应前视：低速用小前视（跟线精度高），高速用大前视（稳定性好）
        lookahead_dist = float(
            clip(self.lookahead_time * v + self.lookahead_base, self.lookahead_min, self.lookahead_max)
        )
        if diag is not None:
            diag["pp_ref_lanes"] = [_lane_key(ref) for ref in (getattr(nav, "current_ref_lanes", None) or [])]
            diag["pp_next_ref_lanes"] = [_lane_key(ref) for ref in (getattr(nav, "next_ref_lanes", None) or [])]
            diag["pp_anchor_lane"] = _lane_key(lane)
            diag["pp_ego_lane"] = _lane_key(getattr(ego, "lane", None))
        if lane is None:
            return self._checkpoint_fallback(nav, ego), lookahead_dist

        long_ego, lat_ego = lane.local_coordinates(ego.position)
        s_anchor = float(clip(float(long_ego), 0.0, float(lane.length)))
        if diag is not None:
            diag["pp_anchor_lat"] = float(lat_ego)
            diag["pp_lat_left"] = self._lane_lateral_left(lane, ego, s_anchor)
        start_lane, plan_diag = self._plan_target_lane(nav, ego, lane, s_anchor, v)
        if diag is not None:
            diag.update(plan_diag)
            diag["pp_target_lane"] = _lane_key(start_lane)

        if start_lane is not lane:
            long_target, _ = start_lane.local_coordinates(ego.position)
            if long_target < 0.0:
                # 目标车道在自车前方（尚未到达）：保持锚车道，让车道链在端点自然过渡，
                # 避免用“远端目标点”提前切弯（诊断：T 字路口/U-turn 提前左/右转出界）
                start_lane = lane
                s = s_anchor
                if diag is not None:
                    diag["pp_target_lane"] = _lane_key(lane)
            else:
                s = float(clip(float(long_target), 0.0, float(start_lane.length)))
                # 变道进行中：前视点截断在目标车道内，不跨到更后面的路段
                end_s = min(s + lookahead_dist, float(start_lane.length))
                if diag is not None:
                    diag["pp_chain"] = [(_lane_key(start_lane), round(s, 2), round(end_s, 2))]
                return _to_point(start_lane.position(end_s, 0.0)), lookahead_dist
        else:
            s = s_anchor
        remaining = lookahead_dist
        cur: Optional[Any] = start_lane
        chain: List[Any] = []
        last_point = _to_point(start_lane.position(s, 0.0))
        for _ in range(self.MAX_CHAIN_SEGMENTS):
            if cur is None:
                if diag is not None:
                    diag["pp_chain"] = chain
                return last_point, lookahead_dist
            avail = max(float(cur.length) - s, 0.0)
            if remaining <= avail:
                point_s = min(s + remaining, float(cur.length))
                chain.append((_lane_key(cur), round(s, 2), round(point_s, 2)))
                if diag is not None:
                    diag["pp_chain"] = chain
                return _to_point(cur.position(point_s, 0.0)), lookahead_dist
            remaining -= avail
            chain.append((_lane_key(cur), round(s, 2), round(float(cur.length), 2)))
            last_point = _to_point(cur.position(float(cur.length), 0.0))
            cur = self._successor_lane(cur, nav, diag)
            s = 0.0
        if diag is not None:
            diag["pp_chain"] = chain
        return last_point, lookahead_dist

    def _plan_target_lane(
        self, nav: Any, ego: BaseVehicle, anchor: Any, s_anchor: float, v: float
    ) -> Tuple[Any, Dict[str, Any]]:
        """沿路由链选本步跟踪的目标车道（提前汇入 + 静态障碍绕行）。

        规则：
        1. 默认跟踪锚车道；当自车距锚车道末端小于规划距离
           ``d = clip(plan_time·v + plan_base, min, max)`` 时，取路由内后继车道；
        2. 若后继车道与锚车道端点不重合（车道数减少 / 横向错位，如 bidirection 的
           单车道偏置），先瞄准路由内端点离后继车道起点最近的“汇入车道”，
           使横向移动在路段内提前完成，而不是到端点做一次跳变；
        3. 目标/后继车道前方存在静态障碍（建筑/收费站岗亭）时，换到同一路由内
           空闲且可达的邻近车道。
        """
        refs = list(getattr(nav, "current_ref_lanes", None) or [])
        nxt = list(getattr(nav, "next_ref_lanes", None) or [])
        diag: Dict[str, Any] = {}
        target = anchor
        remaining = max(float(anchor.length) - s_anchor, 0.0)
        plan_dist = float(
            clip(
                self.lane_change_plan_time * v + self.lane_change_plan_base,
                self.lane_change_plan_min,
                self.lane_change_plan_max,
            )
        )
        diag["pp_plan_dist"] = round(plan_dist, 2)
        diag["pp_remaining"] = round(remaining, 2)
        succ = self._successor_lane(anchor, nav, None)
        diag["pp_plan_succ"] = _lane_key(succ)
        if succ is not None and remaining <= plan_dist:
            connect_gap = float(np.linalg.norm(_to_point(anchor.end) - _to_point(succ.start)))
            diag["pp_connect_gap"] = round(connect_gap, 2)
            if connect_gap <= self.LANE_CONNECT_TOL:
                # 共线续接：继续跟随当前车道，车道链在端点自然过渡
                target = anchor
            else:
                merge = self._merge_lane(refs, succ, ego, anchor=anchor)
                diag["pp_merge_lane"] = _lane_key(merge)
                target = succ if merge is None or merge is anchor else merge
        # 静态障碍（收费站岗亭/建筑）：目标或后继车道被挡住时换到空闲邻近路由车道
        blocker_lanes = [target] + ([succ] if succ is not None else [])
        blocker_gap = self._min_static_blocker_gap(blocker_lanes, ego, self.static_obstacle_range)
        diag["pp_blocker_gap"] = round(blocker_gap, 2) if blocker_gap is not None else None
        if blocker_gap is not None:
            alternate = self._free_alternate_lane(refs + nxt, ego, target)
            diag["pp_alternate_lane"] = _lane_key(alternate)
            if alternate is not None:
                target = alternate
        # 安全护栏：目标车道航向与锚车道末端航向相反（> MAX_SUCC_HEADING_DIFF）时拒绝，
        # 回到锚车道（绝不瞄准对向/掉头车道）
        if target is not anchor and not self._heading_consistent(anchor, target):
            diag["pp_target_rejected"] = _lane_key(target)
            target = anchor
        return target, diag

    def _min_static_blocker_gap(self, lanes: List[Any], ego: BaseVehicle, max_range: float) -> Optional[float]:
        """多条候选车道上最近的静态障碍净距（m）；全无则 ``None``。"""
        best: Optional[float] = None
        for lane in lanes:
            gap = self._static_blocker_gap(lane, ego, max_range)
            if gap is not None and (best is None or gap < best):
                best = gap
        return best

    def _merge_lane(
        self, refs: List[Any], succ: Any, ego: BaseVehicle, anchor: Any = None, exclude: Any = None
    ) -> Optional[Any]:
        """在 ``refs`` 中选端点离 ``succ.start`` 最近的汇入车道（优先无静态障碍者）。

        ``anchor`` 提供“平局/近平局保持”死区：锚车道与最优汇入车道的端点距离差
        ``<= LANE_CONNECT_TOL`` 时保持锚车道，避免在两条近等距车道间反复横跳
        （诊断：SBS 场景曾出现 target=lane0 → lane1 的 ping-pong）。
        """
        if not refs:
            return None
        succ_start = _to_point(succ.start)
        pool = [ref for ref in refs if ref is not exclude]
        if not pool:
            return None
        free = [ref for ref in pool if self._static_blocker_gap(ref, ego, self.static_obstacle_range) is None]
        candidates = free or pool
        best = min(candidates, key=lambda ref: float(np.linalg.norm(_to_point(ref.end) - succ_start)))
        if anchor is not None and anchor in candidates and anchor is not best:
            best_dist = float(np.linalg.norm(_to_point(best.end) - succ_start))
            anchor_dist = float(np.linalg.norm(_to_point(anchor.end) - succ_start))
            if anchor_dist <= best_dist + self.LANE_CONNECT_TOL:
                return anchor
        return best

    def _free_alternate_lane(self, route_lanes: List[Any], ego: BaseVehicle, blocked: Any) -> Optional[Any]:
        """在被 ``blocked`` 挡住时，从路由车道中选一条空闲、可达的邻近车道。

        优先级：车道序号差最小（横向最近）→ 前方静态障碍距离更远 → 车道序号更小（稳定）。
        """
        if blocked is None:
            return None
        blocked_idx = blocked.index[-1] if isinstance(blocked.index, tuple) else None
        candidates: List[Any] = []
        for lane in route_lanes:
            if lane is blocked or lane is None:
                continue
            gap = self._static_blocker_gap(lane, ego, self.static_obstacle_range)
            if gap is not None:
                continue
            if not self._lane_change_allowed(lane, ego):
                continue
            lane_idx = lane.index[-1] if isinstance(lane.index, tuple) else None
            index_delta = abs(lane_idx - blocked_idx) if (lane_idx is not None and blocked_idx is not None) else 99
            candidates.append((index_delta, lane_idx if lane_idx is not None else 99, lane))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]

    def _lane_change_allowed(self, target_lane: Any, ego: BaseVehicle) -> bool:
        """目标车道在自车前后 ``lane_change_safety_gap`` 内无车时允许变道。"""
        if self.lane_change_safety_gap <= 0.0:
            return True
        try:
            long_ego, _ = target_lane.local_coordinates(ego.position)
            found = FrontBackObjects.get_find_front_back_objs(
                self._surrounding_vehicles(ego),
                target_lane,
                ego.position,
                max_distance=float(self.lane_change_safety_gap),
            )
        except (AttributeError, KeyError, ValueError, TypeError, AssertionError):
            return True
        if found.has_front_object() and float(found.front_min_distance()) < self.lane_change_safety_gap:
            return False
        if found.has_back_object() and float(found.back_min_distance()) < self.lane_change_safety_gap:
            return False
        return True

    def _static_blocker_gap(self, lane: Any, ego: BaseVehicle, max_range: float) -> Optional[float]:
        """``lane`` 前方 ``max_range`` 内最近的静态障碍（建筑）净距（m）；无则 ``None``。

        用于收费站岗亭等“车道内硬障碍”：它们不出现在车辆的 lidar/前车检索里，
        必须单独扫描场景对象，否则会径直撞上（见诊断：tollgate id90 以 0.3 m/s 撞岗亭）。
        """
        if lane is None or max_range <= 0.0:
            return None
        try:
            objects = list(self.engine.get_objects().values())
        except (AttributeError, KeyError):
            return None
        try:
            s_ego = float(lane.local_coordinates(ego.position)[0])
        except (AttributeError, ValueError, TypeError):
            return None
        best: Optional[float] = None
        half_length = 0.0
        for obj in objects:
            if not isinstance(obj, BaseBuilding):
                continue
            try:
                s_obj, lat_obj = lane.local_coordinates(obj.position)
            except (AttributeError, ValueError, TypeError):
                continue
            if abs(float(lat_obj)) > float(lane.width) / 2.0 + self.static_obstacle_lat_margin:
                continue
            try:
                half_length = 0.5 * float(getattr(obj, "LENGTH", 0.0) or 0.0)
            except (TypeError, ValueError):
                half_length = 0.0
            gap = float(s_obj) - s_ego - half_length
            if -half_length <= gap <= max_range and (best is None or gap < best):
                best = max(gap, 0.0)
        return best

    @staticmethod
    def _anchor_lane(nav: Any, ego: BaseVehicle) -> Optional[Any]:
        """选跟踪锚车道：优先自车当前车道（在路由车道内），否则取横向最近的参考车道。"""
        lane = getattr(ego, "lane", None)
        refs = list(getattr(nav, "current_ref_lanes", None) or [])
        if lane is not None and any(lane is ref for ref in refs):
            return lane
        if refs:
            return min(refs, key=lambda ref: float(ref.distance(ego.position)))
        return lane

    def _successor_lane(self, lane: Any, nav: Any, diag: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """找 ``lane`` 沿行车方向的后继车道（**只在导航路由内**，多级兜底）。

        为什么去掉“全图几何连接”兜底：bidirection（B）块的正/反向车道共享同一段
        中心线，全图 ``is_previous_lane_of`` 会把**对向车道**选成后继（掉头），
        必须严格限制在 ``current_ref_lanes + next_ref_lanes`` 内并校验航向连续性。
        优先级：路由内几何连接 → 路由内道路节点相接 → 路由内几何最近。
        ``diag`` 非空时记录命中的兜底级别与候选集合（``pp_succ_source``）。
        """
        refs = list(getattr(nav, "current_ref_lanes", None) or [])
        nxt = list(getattr(nav, "next_ref_lanes", None) or [])

        # 1) 路由内的几何连接（端点重合 < 0.1 m，abs_lane.py:84-89）
        candidates = [
            other for other in refs + nxt
            if other is not lane and lane.is_previous_lane_of(other) and self._heading_consistent(lane, other)
        ]
        source = "route_geom"
        # 2) 道路端节点相接（路口/车道数变化处端点未必严格重合）
        if not candidates:
            candidates = [
                other for other in nxt
                if other is not lane and self._road_connected(lane, other) and self._heading_consistent(lane, other)
            ]
            source = "road_node"
        # 3) 路由内几何最近的下一路段车道（最后兜底）
        if not candidates and nxt:
            end = _to_point(lane.end)
            ordered = sorted(
                (other for other in nxt if other is not lane and self._heading_consistent(lane, other)),
                key=lambda other: float(np.linalg.norm(_to_point(other.start) - end)),
            )
            candidates = ordered[:1]
            source = "nearest_next"
        if diag is not None:
            diag["pp_succ_source"] = source if candidates else None
            diag["pp_succ_candidates"] = [_lane_key(other) for other in candidates]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # 多候选：优先同车道序号（MetaDrive 车道序号沿行车方向连续），再取航向最连续者
        lane_idx = lane.index[-1] if isinstance(lane.index, tuple) else None
        same_index = [
            other for other in candidates
            if lane_idx is not None and isinstance(other.index, tuple) and other.index[-1] == lane_idx
        ]
        pool = same_index or candidates
        end_heading = float(lane.heading_theta_at(float(lane.length)))
        return min(pool, key=lambda other: abs(float(wrap_to_pi(other.heading_theta_at(0.0) - end_heading))))

    @classmethod
    def _heading_consistent(cls, lane: Any, other: Any) -> bool:
        """后继车道起始航向与当前车道末端航向差 < ``MAX_SUCC_HEADING_DIFF``。

        用于剔除对向/掉头车道（bidirection 反向车道与正向车道端点相接但航向相反）。
        """
        try:
            end_heading = float(lane.heading_theta_at(float(lane.length)))
            start_heading = float(other.heading_theta_at(0.0))
        except (AttributeError, ValueError, TypeError):
            return True
        return abs(float(wrap_to_pi(start_heading - end_heading))) < cls.MAX_SUCC_HEADING_DIFF

    @staticmethod
    def _road_connected(lane: Any, other: Any) -> bool:
        """lane 所在道路的终点节点 == other 所在道路的起点节点（node_road_network.py:106-113）。"""
        idx_a, idx_b = getattr(lane, "index", None), getattr(other, "index", None)
        if not (isinstance(idx_a, tuple) and isinstance(idx_b, tuple)):
            return False
        return idx_a[1] == idx_b[0]

    def _checkpoint_fallback(self, nav: Any, ego: BaseVehicle) -> Optional[np.ndarray]:
        """车道对象完全不可用时的兜底：取导航 checkpoint 中最靠前的世界坐标点。"""
        try:
            checkpoints = nav.get_checkpoints()
        except (AttributeError, KeyError, ValueError, TypeError):
            return None
        theta = float(ego.heading_theta)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        ahead: List[Tuple[float, np.ndarray]] = []
        for ckpt in checkpoints:
            point = _to_point(ckpt)
            x_fwd = (point[0] - ego.position[0]) * cos_t + (point[1] - ego.position[1]) * sin_t
            if x_fwd > self.MIN_LOOKAHEAD_FORWARD:
                ahead.append((x_fwd, point))
        if ahead:
            ahead.sort(key=lambda item: item[0])
            return ahead[0][1]
        return _to_point(checkpoints[-1])

    @staticmethod
    def _lane_lateral_left(lane: Any, ego: BaseVehicle, s: Optional[float] = None) -> Optional[float]:
        """自车相对 ``lane`` 中心线的横向偏差（自车左向为正，世界系，m）。

        为什么不直接用 ``lane.local_coordinates`` 的 lat：部分车道（如 CircularLane）
        的横向正方向随车道 ``direction`` 翻转，直接用会搞反修正方向；这里统一投影到
        车道切向的左侧（``[-sin h, cos h]``），与转向动作“正=左”同一约定。
        """
        try:
            if s is None:
                long_, _ = lane.local_coordinates(ego.position)
                s = float(clip(float(long_), 0.0, float(lane.length)))
            heading = float(lane.heading_theta_at(float(s)))
            center = _to_point(lane.position(float(s), 0.0))
        except (AttributeError, ValueError, TypeError):
            return None
        offset = _to_point(ego.position) - center
        return float(-offset[0] * math.sin(heading) + offset[1] * math.cos(heading))

    def _lateral_action(
        self,
        ego: BaseVehicle,
        target_point: np.ndarray,
        lookahead_dist: float,
        diag: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """纯跟踪律 → 归一化转向动作，返回 ``(steer, alpha)``。

        ``α`` 为前视点相对自车纵轴的夹角（左正）；曲率 ``κ = 2 sin(α) / Ld``；
        期望前轮转角 ``δ = atan(κ · L)``；``steer = δ / max_steer_angle``。

        另叠加横向偏差修正 ``-gain · atan(2L·e/Ld²)``（e 为自车相对锚车道中心线的
        左向偏差）：纯跟踪在持续曲率下存在与曲率成正比的稳态内切偏差（诊断：
        roundabout R≈17 m 处内切 0.79 m，把相邻车道净距从 3.5 m 压到 1.9 m 并导致碰撞），
        该项把偏差压回中心线。
        """
        dx = float(target_point[0]) - float(ego.position[0])
        dy = float(target_point[1]) - float(ego.position[1])
        theta = float(ego.heading_theta)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        x_fwd = dx * cos_t + dy * sin_t  # 自车纵向分量
        y_left = -dx * sin_t + dy * cos_t  # 自车左向分量（正 = 左）
        if x_fwd <= self.MIN_LOOKAHEAD_FORWARD:
            # 目标点在身后（定位抖动 / 刚过弯）：纯跟踪无定义，交给纵向控制直行
            return 0.0, 0.0
        alpha = math.atan2(y_left, x_fwd)
        ld = max(float(lookahead_dist), 1e-3)
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), ld)
        steer_pp = delta / self.max_steer_angle_rad * self.steer_gain
        correction = 0.0
        if self.lateral_correction_gain > 0.0 and diag is not None:
            lat_left = diag.get("pp_lat_left")
            if lat_left is not None:
                correction = (
                    -self.lateral_correction_gain
                    * math.atan2(2.0 * self.wheelbase * float(lat_left), ld * ld)
                    / self.max_steer_angle_rad
                )
        if diag is not None:
            diag["pp_steer_pp"] = float(steer_pp)
            diag["pp_steer_corr"] = float(correction)
        steer = steer_pp + correction
        return float(clip(steer, -1.0, 1.0)), float(alpha)

    # ------------------------------------------------------------------ 纵向：IDM

    def _longitudinal_action(self, ego: BaseVehicle) -> Tuple[float, Dict[str, Any]]:
        """IDM（含前车速度差 + 静态障碍）+ 限速 → 归一化油门/刹车动作，返回 ``(throttle, diag)``。

        静态障碍（收费站岗亭等建筑）不出现在车辆 lidar/前车检索里，单独按“静止前车”
        并入 IDM 交互项，避免径直撞上（诊断：tollgate id90 以 0.3 m/s 撞岗亭）。
        """
        v = max(float(ego.speed), 0.0)
        v_target = self._target_speed_mps(ego)
        a_free = self.idm_a_max * (1.0 - (v / v_target) ** self.idm_delta)

        interaction = 0.0
        gap = float("inf")
        lead, dist_center = self._find_lead(ego)
        if lead is not None:
            lead_length = float(getattr(lead, "LENGTH", float(ego.LENGTH)))
            gap = max(dist_center - 0.5 * (float(ego.LENGTH) + lead_length), self.IDM_MIN_GAP)
            # 前车速度投影到自车纵轴（与官方 IDM 的 projected 口径一致，idm_policy.py:313-320）
            v_lead = float(np.dot(_to_point(lead.velocity), _to_point(ego.heading)))
            v_lead = max(v_lead, 0.0)
            interaction = max(interaction, (self._idm_s_star(v, v_lead) / gap) ** 2)

        static_gap = self._static_blocker_gap(getattr(ego, "lane", None), ego, self.static_obstacle_range)
        if static_gap is not None:
            static_gap = max(float(static_gap), self.IDM_MIN_GAP)
            interaction = max(interaction, (self._idm_s_star(v, 0.0) / static_gap) ** 2)
            if gap == float("inf") or static_gap < gap:
                gap = static_gap

        a_des = a_free - self.idm_a_max * interaction
        scale = self.accel_action_scale if a_des >= 0.0 else self.brake_action_scale
        throttle = float(clip(a_des / scale, -1.0, 1.0))
        diag = {
            "pp_target_speed_kmh": float(v_target * 3.6),
            "pp_a_des": float(a_des),
            "pp_lead_gap_m": float(gap) if math.isfinite(gap) else -1.0,
            "pp_static_gap_m": float(static_gap) if static_gap is not None else -1.0,
        }
        return throttle, diag

    def _idm_s_star(self, v: float, v_lead: float) -> float:
        """IDM 期望净距 ``s* = s0 + max(0, vT + v(v-v_lead)/(2√(ab)))``（m）。"""
        return self.idm_distance_wanted + max(
            0.0,
            v * self.idm_time_wanted
            + v * (v - v_lead) / (2.0 * math.sqrt(self.idm_a_max * self.idm_b_comfort))
        )

    def _target_speed_mps(self, ego: BaseVehicle) -> float:
        """目标速度（m/s）：显式配置 > 车道限速 > 兜底限速；再受车辆极速约束。"""
        if self.idm_desired_speed_kmh is not None:
            v_target = float(self.idm_desired_speed_kmh) / 3.6
        else:
            v_target = lane_speed_limit_mps(
                getattr(ego, "lane", None),
                units=self.speed_limit_units,
                fallback_mps=self.fallback_speed_limit_mps,
            )
        v_max = float(getattr(ego, "max_speed_m_s", 80.0 / 3.6))
        return max(min(v_target, v_max), 1.0)

    def _find_lead(self, ego: BaseVehicle) -> Tuple[Optional[BaseVehicle], float]:
        """返回自车所在车道前方的最近车辆及其中心距（m）；无前车返回 ``(None, 搜索距离)``。"""
        lane = getattr(ego, "lane", None)
        if lane is None:
            return None, float(self.idm_detection_range)
        try:
            found = FrontBackObjects.get_find_front_back_objs(
                self._surrounding_vehicles(ego), lane, ego.position, max_distance=self.idm_detection_range
            )
        except (AttributeError, KeyError, ValueError, TypeError, AssertionError):
            return None, float(self.idm_detection_range)
        return found.front_object(), float(found.front_min_distance())

    def _surrounding_vehicles(self, ego: BaseVehicle) -> List[BaseVehicle]:
        """自车探测半径内的其他车辆。

        为什么优先 lidar：官方 IDM 用 ``lidar.get_surrounding_objects`` 的 broad-phase
        结果（idm_policy.py:238），同口径可复现；lidar 不可用时回退扫全场车辆。
        """
        radius = int(math.ceil(self.idm_detection_range))
        try:
            lidar = ego.lidar
        except (AttributeError, ValueError, KeyError):
            lidar = None
        if lidar is not None and radius > 0:
            try:
                objs = lidar.get_surrounding_objects(ego, radius)
                return [obj for obj in objs if isinstance(obj, BaseVehicle) and obj is not ego]
            except (AttributeError, KeyError, ValueError, TypeError, AssertionError):
                pass
        objs = self.engine.get_objects().values()
        return [obj for obj in objs if isinstance(obj, BaseVehicle) and obj is not ego]

    # ------------------------------------------------------------------ 默认值推导

    @classmethod
    def _infer_wheelbase(cls, control_object: Any) -> float:
        """从车辆类常量推导轴距（DefaultVehicle: 1.05234 + 1.4166 = 2.46894 m）。"""
        front = getattr(control_object, "FRONT_WHEELBASE", None)
        rear = getattr(control_object, "REAR_WHEELBASE", None)
        if front and rear:
            return float(front) + float(rear)
        return cls.DEFAULT_WHEELBASE

    @classmethod
    def _infer_max_steer_angle_rad(cls, control_object: Any) -> float:
        """从 ``config["max_steering"]``（度，pg_space.py:232）推导 action=±1 的前轮转角。"""
        try:
            deg = float(control_object.config["max_steering"])
        except (AttributeError, KeyError, TypeError, ValueError):
            deg = cls.DEFAULT_MAX_STEERING_DEG
        return math.radians(deg) if deg > 0.0 else cls.DEFAULT_MAX_STEER_ANGLE_RAD
