"""OD 通道：盒式 scope 内动态车辆，**固定槽位 = track id**（schema v2）。

特征（9 维，自车系 x 前向 / y 左向，单位 SI）::

    [dx, dy, vx, vy, cosθ, sinθ, L, W, type_id]

- ``dx, dy``：对象位置相对 ego 的位置，自车系（``ego.convert_to_local_coordinates``）；
- ``vx, vy``：**相对速度**（对象速度 - ego 速度）旋入自车系，与 MetaDrive 自带 lidar 观测口径一致
  （``component/sensors/lidar.py:113-117``）；这样 TTC 可直接由 dx/vx 得到；
- ``cosθ, sinθ``：对象航向相对 ego 航向的余弦/正弦（世界角差，坐标旋转下等变）；
- ``L, W``：车长/车宽（``component/vehicle/base_vehicle.py:593-601``，单位 m）；
- ``type_id``：车型离散 id（见 ``VEHICLE_TYPE_IDS``）。

固定槽位（v2；旧实现"逐帧按 (min(TTC,cap),距离) 重排"会让同一 slot 跨帧变成不同物体，
实测 12.5%/步）
------------------------------------------------------------------------------------------
每个 episode 维护 ``track id -> slot`` 表（``reset()`` 清空）：

1. 每帧对象按 track id 找槽；**新对象 → 第一个空槽**（新对象在候选内按紧迫度/距离排序：
   ``(min(TTC, ttc_cap_s), 距离)``，紧迫者优先占位）；
2. 槽满时：紧迫度最高的新对象**驱逐**"最长未出现且已出盒"的槽——即本帧不在盒内
   （含已销毁）且连续未出现时间最长的槽（并列取最小 slot 下标）；没有可驱逐槽时该对象丢弃；
3. 对象出盒 → ``presence=0``（槽保留，特征保持最近一次盒内观测值，供记忆库短期跟踪）；
   出盒连续超过 ``release_after_s``（默认 1.0 s）→ **释放**该槽（``od_id=-1``、特征清零）；
4. 释放计时用 ``env.episode_step`` 的真实时间轴（采集路径每 5 个 env step 才 build 一次，
   按"调用次数"计时会高估持续时间）；env 无 ``episode_step`` 时退化为按 build 次数 × dt。

输出（``build`` 返回 features/mask；id/presence 是 **companion** 数组，见 ``companions()``）::

    od         (16, 9) float32   # 特征；presence=0 的槽保留最近一次盒内观测（可能陈旧）
    od_mask    (16,)   float32   # 1 = 该槽本帧有效（已分配 track，``od_id >= 0``）
    od_id      (16,)   int64     # track id（episode 内稳定；-1 = 空槽）
    od_presence(16,)   float32   # 1 = 对象本帧在盒内被观测到（特征新鲜）

``od_mask`` 与 ``od_presence`` 的区别（下游必须都看）：``mask`` 是槽位级"该槽存在/身份有效"，
``presence`` 是实体级"本帧真的观测到"。出盒未释放期间 ``mask=1, presence=0``：身份与最近
观测保留（记忆库无缺口），但本帧几何**陈旧**，只有 ``presence=1`` 的槽才是新鲜观测。
``od_id``/``od_presence`` 随帧进入 6 帧历史（``od_id_hist``/``od_presence_hist``）。

scope = 盒式：默认 前 150 / 后 50 / 左 25 / 右 25 m（v2 由旧的 前 100 扩到 150：记忆库要预测
3 s 未来，150 m 前视覆盖 ~30 m/s 高速下的完整 3 s 窗口；后/侧向不变）。
"""

from __future__ import annotations

import math
from typing import Any

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

#: 空槽的 ``od_id`` 哨兵（track id 从 1 开始分配）
OD_ID_EMPTY = -1

#: 出盒多久后释放槽位（秒）；"持续 > 1.0 s"
DEFAULT_RELEASE_AFTER_S = 1.0

_EPS = 1e-3


def vehicle_type_id(obj) -> float:
    """按 MRO 名称查表，未知车型返回 ``VEHICLE_TYPE_ID_UNKNOWN``。"""
    for cls in type(obj).__mro__:
        tid = VEHICLE_TYPE_IDS.get(cls.__name__)
        if tid is not None:
            return float(tid)
    return float(VEHICLE_TYPE_ID_UNKNOWN)


class ODChannel(ObservationChannel):
    """盒式 scope 内动态对象通道（固定槽位 = track id）。"""

    name = "od"
    feature_dim = 9
    # 位置对 (dx,dy)、速度对 (vx,vy)、朝向单位向量对 (cosθ,sinθ)
    alignment = FrameAlignment(point_pairs=((0, 1), ), vector_pairs=((2, 3), (4, 5)))
    #: 随帧携带、不做 SE(2) 变换的槽位级伴随数组（进入历史/缓冲区）
    companion_names: tuple[str, ...] = ("od_id", "od_presence")
    #: 伴随数组的缺失填充值（见 FrameMemory / FrameLookup）
    companion_fill: dict[str, Any] = {"od_id": OD_ID_EMPTY, "od_presence": 0.0}

    def __init__(
        self,
        *,
        num_slots: int = 16,
        front_m: float = 150.0,
        rear_m: float = 50.0,
        left_m: float = 25.0,
        right_m: float = 25.0,
        ttc_cap_s: float = 5.0,
        physics_dt: float = 0.1,
        release_after_s: float = DEFAULT_RELEASE_AFTER_S,
    ):
        self.num_slots = int(num_slots)
        self.front_m = float(front_m)
        self.rear_m = float(rear_m)
        self.left_m = float(left_m)
        self.right_m = float(right_m)
        self.ttc_cap_s = float(ttc_cap_s)
        self.physics_dt = float(physics_dt)
        self.release_after_s = float(release_after_s)
        self._reset_table()

    # ------------------------------------------------------------------ 内部状态
    def _reset_table(self) -> None:
        self._slot_track: list[int] = [OD_ID_EMPTY] * self.num_slots  # slot -> track id
        self._track_slot: dict[int, int] = {}  # track id -> slot
        self._slot_row = np.zeros((self.num_slots, self.feature_dim), dtype=np.float32)
        self._slot_last_seen = np.full(self.num_slots, -(1 << 40), dtype=np.int64)
        self._track_ids: dict[str, int] = {}  # MetaDrive object id（字符串）-> 本 episode track id
        self._track_counter = 0
        self._last_step: int | None = None
        self._build_count = 0
        self._companions = {
            "od_id": np.full(self.num_slots, OD_ID_EMPTY, dtype=np.int64),
            "od_presence": np.zeros(self.num_slots, dtype=np.float32),
        }

    def reset(self) -> None:
        """episode 边界：清空 id->slot 表与全部计时（builder 在 ``episode_step==0`` 调用）。"""
        self._reset_table()

    # ------------------------------------------------------------------ track id / 时间轴
    def _track_id(self, obj: Any) -> int:
        """MetaDrive 对象 id（字符串）-> episode 内稳定整数 track id（从 1 递增）。"""
        key = str(getattr(obj, "id", None) or f"py{id(obj)}")
        track = self._track_ids.get(key)
        if track is None:
            self._track_counter += 1
            track = self._track_counter
            self._track_ids[key] = track
        return track

    def _current_step(self, env: Any) -> int:
        """真实时间轴（env.step 计数）；拿不到时退化为 build 次数。"""
        try:
            step = int(getattr(env, "episode_step"))
            if step < 0:
                raise ValueError
            return step
        except Exception:  # noqa: BLE001
            self._build_count += 1
            return self._build_count

    def _release_steps(self) -> float:
        if self.physics_dt <= 0.0:
            return float("inf")
        return self.release_after_s / self.physics_dt

    # ------------------------------------------------------------------ 槽位操作
    def _first_free_slot(self) -> int | None:
        for slot, track in enumerate(self._slot_track):
            if track == OD_ID_EMPTY:
                return slot
        return None

    def _pick_victim(self, observed: set[int]) -> int | None:
        """"最长未出现且已出盒"的槽（未观测且 last_seen 最早；并列取最小下标）。"""
        victim, victim_step = None, None
        for slot, track in enumerate(self._slot_track):
            if track == OD_ID_EMPTY or track in observed:
                continue
            last_seen = int(self._slot_last_seen[slot])
            if victim is None or last_seen < victim_step:
                victim, victim_step = slot, last_seen
        return victim

    def _assign(self, slot: int, track: int, step: int, row: np.ndarray) -> None:
        self._slot_track[slot] = track
        self._track_slot[track] = slot
        self._slot_row[slot] = row
        self._slot_last_seen[slot] = int(step)

    def _release(self, slot: int) -> None:
        track = self._slot_track[slot]
        if track != OD_ID_EMPTY:
            self._track_slot.pop(track, None)
        self._slot_track[slot] = OD_ID_EMPTY
        self._slot_row[slot] = 0.0
        self._slot_last_seen[slot] = -(1 << 40)

    # ------------------------------------------------------------------ 候选
    def _candidate_rows(self, env: Any, ego: Any) -> list[tuple[float, float, int, np.ndarray]]:
        """盒内候选 ``[(urgency, dist, track, row)]``（未排序）。"""
        engine = getattr(env, "engine", None)
        objects = getattr(engine, "get_objects", None)
        if objects is None:
            return []
        try:
            all_objects = objects()
        except Exception:  # noqa: BLE001
            return []
        ego_theta = float(ego.heading_theta)
        out: list[tuple[float, float, int, np.ndarray]] = []
        for obj in all_objects.values():
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
            track = self._track_id(obj)
            out.append((min(ttc, self.ttc_cap_s), dist, track, row))
        out.sort(key=lambda item: (item[0], item[1]))
        return out

    # ------------------------------------------------------------------ 构建
    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(self.num_slots, self.feature_dim)
        ego = safe_ego(env)
        if ego is None:
            self._companions = {
                "od_id": np.full(self.num_slots, OD_ID_EMPTY, dtype=np.int64),
                "od_presence": np.zeros(self.num_slots, dtype=np.float32),
            }
            return feats, mask

        step = self._current_step(env)
        if step == self._last_step:
            return self._render()
        self._last_step = step

        candidates = self._candidate_rows(env, ego)
        observed = {track for _, _, track, _ in candidates}
        fresh: dict[int, np.ndarray] = {}  # slot -> row（本帧观测）

        for _, _, track, row in candidates:
            slot = self._track_slot.get(track)
            if slot is not None:
                self._slot_row[slot] = row
                self._slot_last_seen[slot] = int(step)
                fresh[slot] = row
                continue
            # 新对象：第一个空槽；槽满则驱逐"最长未出现且已出盒"的槽
            free = self._first_free_slot()
            if free is None:
                free = self._pick_victim(observed)
                if free is None:
                    continue  # 16 槽全被在盒对象占满：丢弃该新对象
                self._release(free)
            self._assign(free, track, step, row)
            fresh[free] = row

        # 释放：出盒连续超过 release_after_s 的槽
        release_steps = self._release_steps()
        for slot, track in enumerate(self._slot_track):
            if track == OD_ID_EMPTY or slot in fresh:
                continue
            absent = int(step) - int(self._slot_last_seen[slot])
            if absent > release_steps + 1e-6:
                self._release(slot)

        return self._render()

    def _render(self) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(self.num_slots, self.feature_dim)
        ids = np.full(self.num_slots, OD_ID_EMPTY, dtype=np.int64)
        presence = np.zeros(self.num_slots, dtype=np.float32)
        return_step = self._last_step
        for slot, track in enumerate(self._slot_track):
            if track == OD_ID_EMPTY:
                continue
            ids[slot] = int(track)
            mask[slot] = 1.0  # 槽位有效（身份保留）
            feats[slot] = self._slot_row[slot]
            # presence=1 需要"最近一次观测发生在当前 step"（_slot_last_seen 在 build 里刷新）
            if return_step is not None and int(self._slot_last_seen[slot]) == int(return_step):
                presence[slot] = 1.0
        self._companions = {"od_id": ids, "od_presence": presence}
        return feats, mask

    # ------------------------------------------------------------------ 伴随数组
    def companions(self, env=None, spec=None) -> dict[str, np.ndarray]:
        """返回最近一次 ``build`` 的槽位级伴随数组（``od_id`` int64 / ``od_presence`` float32）。"""
        return {key: np.array(value, copy=True) for key, value in self._companions.items()}
