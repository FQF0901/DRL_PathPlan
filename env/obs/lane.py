"""当前车道块（schema v5，结构迭代 A）：ego 本车道的显式几何 token。

为什么需要
----------
v5 起 LD 主块的采样 offset 改为远场 ``{20,40,60,80} m``（见 :mod:`env.obs.ld`）：
- 近场（<20 m）横向锚定信息不再出现在 LD 的环填充里；
- 车道多 / LD 槽位被裁时，近场可能整段缺失（环填充是竞争关系）。

本通道只描述 **ego 当前车道**（``ego.lane``；缺失时退化为 ``navigation.current_ref_lanes[0]``），
单槽、不参与 LD 槽位竞争，因此近场横向锚定信息**显式且稳定可得**，与 LD 的 secondary
环填充解耦。

特征（17 维，自车系 x 前 / y 左；角度单位 rad）
-----------------------------------------------
===== ================ ====================================================
dim   名称              语义
===== ================ ====================================================
0     d_lat             ego 相对本车道中心线的横向偏差（m）。符号：**正 = ego 位于
                       中心线左侧**（沿车道方向看）。口径 = ``-lane.local_coordinates(
                       ego.position)[1]``（MetaDrive 车道系 lat 正 = 车道方向右侧，
                       与 ``env.metadrive_env.lane_lateral_info`` 的符号相反、绝对值相同）。
                       直道且航向对齐时恒有 ``d_lat == -near_dy``（符号锚定校验）。
1     heading_err       ``wrap(lane.heading_theta_at(s0) - ego.heading_theta)``：
                       车道方向相对 ego 航向的夹角（**正 = 车道方向在 ego 航向左侧**，
                       即 ego 相对车道右偏航）。与 LD 的 ``heading_rel`` 同号约定
                       （车道 - ego）。收敛目标：``heading_err ≈ +k·d_lat``（k>0 小）。
2     lane_width        车道宽 ``lane.width``（m）。
3     curvature         当前投影点的 dθ/ds（数值中心差分；右转（顺时针）为负，1/m）。
4     speed_limit       本车道限速**原始值**（m/s；与 LD 同口径，不做换算）。
5-7   near_*            s0+5 m 中心线点：自车系 ``(dx, dy)`` + ``heading_rel``（车道 - ego）。
8-10  mid_*             s0+15 m 中心线点：同上（近场-中距锚）。
11-13 far_*             s0+60 m 中心线点：同上（远场锚；LD 20–80 m 的补充摘要）。
14    near_valid        1 = s0+5 m 在车道长度内（几何字段有效）。
15    mid_valid         1 = s0+15 m 在车道长度内。
16    far_valid         1 = s0+60 m 在车道长度内。
===== ================ ====================================================

- ``s0`` = ego 在车道上的纵向投影（clamp 到 [0, length]；投影失败退化为 0）。
- 超车道末端的采样点：几何字段清零、``*_valid=0``（不丢弃整块）。
- 不套用 OD/LD 的盒式 scope：本块只描述自车所在车道，前 60 m 恒在 150 m 前视内。
- 无车道 / 无 ego：全 0 + ``lane_mask=0``。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from env.obs.base import FrameAlignment, ObservationChannel, make_empty, safe_ego, wrap_to_pi
from env.obs.ld import lane_curvature

#: 近场 / 中距 / 远场中心线采样 offset（m）
LANE_SAMPLE_OFFSETS: tuple[float, ...] = (5.0, 15.0, 60.0)

LANE_FEATURE_NAMES: tuple[str, ...] = (
    "d_lat",
    "heading_err",
    "lane_width",
    "curvature",
    "speed_limit",
    "near_dx",
    "near_dy",
    "near_heading_rel",
    "mid_dx",
    "mid_dy",
    "mid_heading_rel",
    "far_dx",
    "far_dy",
    "far_heading_rel",
    "near_valid",
    "mid_valid",
    "far_valid",
)
LANE_DIM = len(LANE_FEATURE_NAMES)

_EPS = 1e-6
_GEOM_BASE = 5  # 3 个采样点 × (dx, dy, heading_rel) 从 dim 5 开始
_VALID_BASE = _GEOM_BASE + 3 * len(LANE_SAMPLE_OFFSETS)


class LaneChannel(ObservationChannel):
    """当前车道块（单槽 17 维；近场/远场中心线摘要 + 横向锚）。"""

    name = "lane"
    feature_dim = LANE_DIM
    # 3 个中心线采样点是点对；heading_err / 3 个 heading_rel 是"车道 - ego"角（对齐时减 Δθ）
    alignment = FrameAlignment(
        point_pairs=tuple((_GEOM_BASE + 3 * k, _GEOM_BASE + 3 * k + 1) for k in range(len(LANE_SAMPLE_OFFSETS))),
        angle_dims=(1, ) + tuple(_GEOM_BASE + 3 * k + 2 for k in range(len(LANE_SAMPLE_OFFSETS))),
    )

    def __init__(
        self,
        *,
        offsets: Sequence[float] = LANE_SAMPLE_OFFSETS,
        num_slots: int = 1,
    ):
        offsets = tuple(float(o) for o in offsets)
        if len(offsets) != len(LANE_SAMPLE_OFFSETS):
            raise ValueError(
                f"lane 通道需要恰好 {len(LANE_SAMPLE_OFFSETS)} 个采样 offset（near/mid/far），收到 {offsets}"
            )
        self.offsets = offsets
        self.num_slots = int(num_slots)

    @staticmethod
    def _current_lane(ego):
        try:
            lane = getattr(ego, "lane", None)
        except Exception:  # noqa: BLE001 - lane 属性链异常一律降级
            lane = None
        if lane is None:
            nav = getattr(ego, "navigation", None)
            ref = getattr(nav, "current_ref_lanes", None) if nav is not None else None
            if ref:
                lane = ref[0]
        return lane

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(self.num_slots, self.feature_dim)
        ego = safe_ego(env)
        if ego is None:
            return feats, mask
        lane = self._current_lane(ego)
        if lane is None:
            return feats, mask
        try:
            length = float(lane.length)
        except Exception:  # noqa: BLE001
            length = 0.0
        if not np.isfinite(length) or length <= 0.0:
            return feats, mask

        ego_theta = float(ego.heading_theta)
        # 投影：d_lat 与 s0（失败退化 0，保持通道可用；与 ld._lane_s0 同哲学）
        s0, d_lat = 0.0, 0.0
        try:
            long, lat = lane.local_coordinates(ego.position)
            s0 = float(min(max(float(long), 0.0), length))
            d_lat = -float(lat)  # 车道系 lat 正 = 右侧 → 统一为"ego 在中心线左侧为正"
        except Exception:  # noqa: BLE001
            s0, d_lat = 0.0, 0.0
        if not np.isfinite(s0):
            s0 = 0.0
        if not np.isfinite(d_lat):
            d_lat = 0.0

        try:
            heading0 = float(lane.heading_theta_at(s0))
        except Exception:  # noqa: BLE001
            heading0 = ego_theta
        try:
            width = float(lane.width)
        except Exception:  # noqa: BLE001
            width = 0.0
        try:
            speed_limit = float(getattr(lane, "speed_limit", 0.0))
        except Exception:  # noqa: BLE001
            speed_limit = 0.0

        feats[0, 0] = d_lat
        feats[0, 1] = float(wrap_to_pi(heading0 - ego_theta))
        feats[0, 2] = width
        feats[0, 3] = lane_curvature(lane, s0)
        feats[0, 4] = speed_limit

        for k, offset in enumerate(self.offsets):
            base = _GEOM_BASE + 3 * k
            s_target = s0 + float(offset)
            if s_target > length + _EPS:
                continue  # 超车道末端：几何 0 + valid 0
            s = min(max(s_target, 0.0), length)
            try:
                point = np.asarray(lane.position(s, 0.0), dtype=np.float32)[:2]
                heading = float(lane.heading_theta_at(s))
            except Exception:  # noqa: BLE001
                continue
            rel = np.asarray(ego.convert_to_local_coordinates(point, ego.position), dtype=np.float32)
            feats[0, base] = float(rel[0])
            feats[0, base + 1] = float(rel[1])
            feats[0, base + 2] = float(wrap_to_pi(heading - ego_theta))
            feats[0, _VALID_BASE + k] = 1.0

        mask[0] = 1.0
        return feats, mask
