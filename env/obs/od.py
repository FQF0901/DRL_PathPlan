"""OD 通道：100 m 内动态车辆 top-16（排除 ego）。

特征（9 维，自车系 x 前向 / y 左向，单位 SI）::

    [dx, dy, vx, vy, cosθ, sinθ, L, W, type_id]

- ``dx, dy``：对象位置相对 ego 的位置，自车系（``ego.convert_to_local_coordinates``）；
- ``vx, vy``：**相对速度**（对象速度 - ego 速度）旋入自车系，与 MetaDrive 自带 lidar 观测口径一致
  （``component/sensors/lidar.py:113-117``）；这样 TTC 可直接由 dx/vx 得到；
- ``cosθ, sinθ``：对象航向相对 ego 航向的余弦/正弦（世界角差，坐标旋转下等变）；
- ``L, W``：车长/车宽（``component/vehicle/base_vehicle.py:593-601``，单位 m）；
- ``type_id``：车型离散 id（见 ``VEHICLE_TYPE_IDS``）。

选取优先级："距离 + TTC"：**对每个候选取 ``min(TTC, ttc_cap_s)`` 作为键**（不接近/后方目标记 +∞ →
被压到 cap，与"TTC ≥ cap"的目标同组），组内再按欧氏距离升序。掩码 mask=1 表示槽位有效。

**scope = 盒式**：只在自车系 前 ``front_m``(默认 100m) / 后 ``rear_m``(50m) / 左 ``left_m``(25m) /
右 ``right_m``(25m) 的矩形内取候选（x 前 / y 左），盒内 top-16，不足补零 + mask=0；不做全图 top-k。

为什么 vx/vy 用相对速度而不是绝对速度：策略关心的是"是否会撞/何时撞"，
相对速度 + dx 即可直接构造 TTC，且与 MetaDrive 上游观测语义一致。
"""

from __future__ import annotations

import math

import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle

from env.obs.base import FrameAlignment, ObservationChannel, make_empty, safe_ego, wrap_to_pi

#: 车型 -> type_id（稳定映射；TrafficDefaultVehicle 等子类归入 DefaultVehicle=0）
VEHICLE_TYPE_IDS: dict[str, int] = {
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
VEHICLE_TYPE_ID_UNKNOWN = 5

_EPS = 1e-3


def vehicle_type_id(obj) -> float:
    """按 MRO 名称查表，未知车型返回 ``VEHICLE_TYPE_ID_UNKNOWN``。"""
    for cls in type(obj).__mro__:
        tid = VEHICLE_TYPE_IDS.get(cls.__name__)
        if tid is not None:
            return float(tid)
    return float(VEHICLE_TYPE_ID_UNKNOWN)


class ODChannel(ObservationChannel):
    """top-16 动态对象通道。"""

    name = "od"
    feature_dim = 9
    # 位置对 (dx,dy)、速度对 (vx,vy)、朝向单位向量对 (cosθ,sinθ)
    alignment = FrameAlignment(point_pairs=((0, 1), ), vector_pairs=((2, 3), (4, 5)))

    def __init__(
        self,
        *,
        num_slots: int = 16,
        front_m: float = 100.0,
        rear_m: float = 50.0,
        left_m: float = 25.0,
        right_m: float = 25.0,
        ttc_cap_s: float = 5.0,
    ):
        self.num_slots = int(num_slots)
        self.front_m = float(front_m)
        self.rear_m = float(rear_m)
        self.left_m = float(left_m)
        self.right_m = float(right_m)
        self.ttc_cap_s = float(ttc_cap_s)

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(self.num_slots, self.feature_dim)
        ego = safe_ego(env)
        engine = getattr(env, "engine", None)
        if ego is None or engine is None:
            return feats, mask

        ego_theta = float(ego.heading_theta)
        candidates: list[tuple[float, float, np.ndarray]] = []
        for obj in engine.get_objects().values():
            if obj is ego or not isinstance(obj, BaseVehicle):
                continue
            # 位置/速度都经 MetaDrive 自己的坐标变换，保证与当前帧其它通道同系
            rel_pos = np.asarray(ego.convert_to_local_coordinates(obj.position, ego.position), dtype=np.float32)
            # 盒式 scope：前 front / 后 rear / 左 left / 右 right（自车系 x 前 / y 左）
            if not (
                -self.rear_m <= float(rel_pos[0]) <= self.front_m
                and -self.right_m <= float(rel_pos[1]) <= self.left_m
            ):
                continue
            dist = float(math.hypot(float(rel_pos[0]), float(rel_pos[1])))
            if not np.isfinite(dist):
                continue
            rel_vel = np.asarray(ego.convert_to_local_coordinates(obj.velocity, ego.velocity), dtype=np.float32)
            heading_rel = float(wrap_to_pi(obj.heading_theta - ego_theta))
            ttc = math.inf
            if rel_pos[0] > 0.0 and rel_vel[0] < -_EPS:
                ttc = float(rel_pos[0]) / float(-rel_vel[0])
            row = np.array(
                [
                    rel_pos[0],
                    rel_pos[1],
                    rel_vel[0],
                    rel_vel[1],
                    math.cos(heading_rel),
                    math.sin(heading_rel),
                    float(obj.LENGTH),
                    float(obj.WIDTH),
                    vehicle_type_id(obj),
                ],
                dtype=np.float32,
            )
            candidates.append((min(ttc, self.ttc_cap_s), dist, row))

        candidates.sort(key=lambda item: (item[0], item[1]))
        for slot, (_, _, row) in enumerate(candidates[: self.num_slots]):
            feats[slot] = row
            mask[slot] = 1.0
        return feats, mask
