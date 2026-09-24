"""ego 通道：自车动力学 8 维。

    [v, a_long, a_lat, yaw_rate, steer, curvature, reserved0, reserved1]

数据来源（都在 MetaDrive 车辆对象上，避免自己维护历史状态）：
- ``v``：``agent.speed``（m/s，``base_class/base_object.py:353-360``）；
- ``a_long``：``(speed - agent.last_speed) / dt``；``last_speed`` 在 ``before_step`` 写入
  （``component/vehicle/base_vehicle.py:224``），reset 时也已初始化（同文件 381-384），
  因此 a_long 就是"上一个 env step（0.1 s）内的平均纵向加速度"，无需通道内缓存；
- ``yaw_rate``：由 ``last_heading_dir``（before_step 写入）与当前 heading 的角差 / dt 得到，
  与上游 StateObservation 的做法一致（``obs/state_obs.py:124-130``）；
- ``a_lat`` = v * yaw_rate（向心加速度）；
- ``steer``：``agent.steering``（归一化到 [-1,1]，``base_vehicle.py:462-466``）；
- ``curvature``：当前车道的 dθ/ds（``env.obs.ld.lane_curvature``，直道 0、圆弧 ±1/R）。

dt 取 ``physics_world_step_size * decision_repeat``（默认 0.02×5=0.1 s）；上游 StateObservation
硬编码 0.1，这里从 env.config 读，避免 config 改动后失真。
"""

from __future__ import annotations

import math

import numpy as np

from env.obs.base import ObservationChannel, make_empty, safe_ego, wrap_to_pi
from env.obs.ld import lane_curvature

EGO_DIM = 8


class EgoChannel(ObservationChannel):
    """自车动力学通道（定长 1 槽）。"""

    name = "ego"
    feature_dim = EGO_DIM

    def __init__(self, *, physics_dt: float = 0.1):
        self.physics_dt = float(physics_dt)

    def _env_dt(self, env) -> float:
        """env.step 的物理时间步长；读不到配置时回退到构造参数。"""
        try:
            cfg = env.config
            dt = float(cfg["physics_world_step_size"]) * int(cfg["decision_repeat"])
            return dt if dt > 0 else self.physics_dt
        except Exception:  # noqa: BLE001
            return self.physics_dt

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(1, self.feature_dim)
        ego = safe_ego(env)
        if ego is None:
            return feats, mask
        dt = self._env_dt(env)

        speed = float(ego.speed)
        last_speed = float(getattr(ego, "last_speed", speed))
        a_long = (speed - last_speed) / dt

        heading = float(ego.heading_theta)
        last_dir = getattr(ego, "last_heading_dir", None)
        if last_dir is None:
            last_heading = heading
        else:
            last_heading = math.atan2(float(last_dir[1]), float(last_dir[0]))
        yaw_rate = float(wrap_to_pi(heading - last_heading)) / dt

        steer = float(getattr(ego, "steering", 0.0))
        curvature = 0.0
        lane = getattr(ego, "lane", None)
        if lane is None:
            nav = getattr(ego, "navigation", None)
            ref = getattr(nav, "current_ref_lanes", None) if nav is not None else None
            lane = ref[0] if ref else None
        if lane is not None:
            try:
                s, _ = lane.local_coordinates(ego.position)
                curvature = lane_curvature(lane, float(s))
            except Exception:  # noqa: BLE001
                curvature = 0.0

        feats[0, 0] = speed
        feats[0, 1] = a_long
        feats[0, 2] = speed * yaw_rate
        feats[0, 3] = yaw_rate
        feats[0, 4] = steer
        feats[0, 5] = curvature
        # feats[0, 6:8] 保留位保持 0
        mask[0] = 1.0
        return feats, mask
