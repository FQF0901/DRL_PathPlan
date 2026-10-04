"""TTC 上下文 token（schema v5，结构迭代 A）：per-OD-slot TTC → 单槽上下文向量。

设计
----
- **输入只读 OD 槽位**（:class:`env.obs.od.ODChannel` 的当前渲染）：槽序、9 维字段、
  track-id 固定槽位与紧迫度排序策略一律不动；本通道只在渲染结果上做恒速 TTC 汇总。
- 每个"新鲜"槽（``od_presence=1``，即对象本帧在盒内被观测）计算两个 TTC 口径
  （自车系 x 前 / y 左，恒速假设）：

  1. ``ttc_x``（纵向轴）：``dx / (-vx)``（``dx>0`` 且 ``vx<0``），与 OD 槽位紧迫度
     ``min(TTC, cap)`` 的 TTC 同口径——"沿自车 x 轴（≈路径切线）的接近时间"；
  2. ``ttc_path``（**沿自车路径投影版**）：把对象视为以恒速运动的点、自车视为半径
     ``collision_radius_m``（默认 2.5 m ≈ 车道宽量级）的圆，解 ``|r + v·t| = R``
     的最小正根 = **首次进入自车路径走廊（碰撞圆）的时间**。它捕获横向切入/斜向逼近
     （``ttc_x`` 会漏检：``dx≤0`` 或纯横向相对速度），用于 cut-in 场景的提前量。

  两者都是"恒速、自车系"口径；``ttc_path`` 的投影是相对运动对自车（路径上的圆）的
  最近接近/进入时间，而不是对 ego 航向轴的标量投影（后者与 ``ttc_x`` 在 û=(1,0)
  时退化为同一量，见本模块设计说明）。

- 汇总成**独立 TTC 上下文 token**（与 nav 同组、在令牌集合中相邻；见
  ``net.model._head_tokens``）：min-TTC、<3 s 计数、**责任槽位**（两个口径合并后
  TTC 最小的新鲜槽）的相对位置/速度。
- 出盒未释放的陈旧槽（``presence=0``）不参与（几何不新鲜）；无任何接近对象时
  min 字段取 ``ttc_cap_s``、``ttc_valid=0``（mask 仍为 1 = "上下文可用且无风险"）。

特征（12 维）
-------------
===== =============== =====================================================
dim   名称            语义
===== =============== =====================================================
0     min_ttc_x       新鲜槽 ``ttc_x`` 最小值，clamp 到 [0, cap]；无 → cap（s）。
1     min_ttc_path    新鲜槽 ``ttc_path`` 最小值，clamp 到 [0, cap]；无 → cap（s）。
2     n_lt3_x         ``ttc_x < 3 s`` 的新鲜槽数（0..16）。
3     n_lt3_path      ``ttc_path < 3 s`` 的新鲜槽数。
4     resp_index      责任槽位下标（0..15，float）；无 → -1。
5     resp_dx         责任槽位相对位置 dx（m，自车系）。
6     resp_dy         责任槽位相对位置 dy（m）。
7     resp_vx         责任槽位相对速度 vx（m/s）。
8     resp_vy         责任槽位相对速度 vy（m/s）。
9     resp_ttc        责任槽位的获胜 TTC（= min(ttc_x, ttc_path)，clamp 到 cap）。
10    resp_is_path    1 = 责任槽位由 ``ttc_path`` 获胜（``ttc_path < ttc_x``），否则 0。
11    valid           1 = 至少一个新鲜槽有有限 TTC（两个口径任一）；0 = 无接近对象/无数据。
===== =============== =====================================================

``ttc_mask=1`` ⇔ ego 存在且 OD 通道可用（上下文可计算；``valid`` 区分"无风险"与"无数据"）。
责任槽位并列时取最小槽下标（确定性）。
"""

from __future__ import annotations

import math

import numpy as np

from env.obs.base import FrameAlignment, ObservationChannel, make_empty, safe_ego
from env.obs.od import ODChannel

DEFAULT_TTC_CAP_S = 5.0
DEFAULT_TTC_HORIZON_S = 3.0
DEFAULT_COLLISION_RADIUS_M = 2.5

TTC_FEATURE_NAMES: tuple[str, ...] = (
    "min_ttc_x",
    "min_ttc_path",
    "n_lt3_x",
    "n_lt3_path",
    "resp_index",
    "resp_dx",
    "resp_dy",
    "resp_vx",
    "resp_vy",
    "resp_ttc",
    "resp_is_path",
    "valid",
)
TTC_DIM = len(TTC_FEATURE_NAMES)

_EPS = 1e-3


def slot_ttc(
    dx: float,
    dy: float,
    vx: float,
    vy: float,
    *,
    collision_radius_m: float = DEFAULT_COLLISION_RADIUS_M,
) -> tuple[float, float]:
    """单槽 ``(ttc_x, ttc_path)``（未截断；``inf`` = 无接近）。

    - ``ttc_x``：``dx > 0 且 vx < -eps`` 时 ``dx / (-vx)``（与 OD 紧迫度同口径）；
    - ``ttc_path``：解 ``|r + v t| = R`` 的最小正根（首次进入碰撞圆）；已在该圆内且
      仍在接近（``r·v < 0``）→ 0.0；远离/无解 → ``inf``。
    """
    ttc_x = math.inf
    if dx > 0.0 and vx < -_EPS:
        ttc_x = float(dx) / float(-vx)

    ttc_path = math.inf
    rr = float(dx) * float(dx) + float(dy) * float(dy)
    vv = float(vx) * float(vx) + float(vy) * float(vy)
    rv = float(dx) * float(vx) + float(dy) * float(vy)
    radius = max(float(collision_radius_m), 0.0)
    if rr <= radius * radius:
        ttc_path = 0.0 if rv < 0.0 else math.inf
    elif vv > _EPS * _EPS and rv < 0.0:
        disc = rv * rv - vv * (rr - radius * radius)
        if disc >= 0.0:
            t = (-rv - math.sqrt(disc)) / vv
            if t >= 0.0:
                ttc_path = float(t)
    return ttc_x, ttc_path


class TTCChannel(ObservationChannel):
    """OD 槽位 TTC 汇总（单槽 12 维；与 nav 相邻的上下文 token）。"""

    name = "ttc"
    feature_dim = TTC_DIM
    # resp_dx/dy 是点；resp_vx/vy 是向量（历史对齐预留；ttc 默认不进 6 帧历史）
    alignment = FrameAlignment(point_pairs=((5, 6), ), vector_pairs=((7, 8), ))

    def __init__(
        self,
        od_channel: ODChannel | None = None,
        *,
        ttc_cap_s: float = DEFAULT_TTC_CAP_S,
        horizon_s: float = DEFAULT_TTC_HORIZON_S,
        collision_radius_m: float = DEFAULT_COLLISION_RADIUS_M,
    ):
        self._od = od_channel
        self.ttc_cap_s = float(ttc_cap_s)
        self.horizon_s = float(horizon_s)
        self.collision_radius_m = float(collision_radius_m)

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(1, self.feature_dim)
        ego = safe_ego(env)
        if ego is None or self._od is None:
            return feats, mask

        # 只读 OD 渲染：槽序/字段/排序策略不动（同一 step 重复 build 幂等）
        od_feats, od_mask = self._od.build(env, spec)
        presence = self._od.companions().get("od_presence")
        cap = max(self.ttc_cap_s, 0.0)

        min_x = min_p = math.inf
        n_x = n_p = 0
        best: tuple[float, int, bool] | None = None  # (ttc, slot, is_path)
        slots = int(od_feats.shape[0])
        for slot in range(slots):
            if od_mask[slot] < 0.5:
                continue
            if presence is None or float(presence[slot]) < 0.5:
                continue  # 陈旧槽（出盒未释放）几何不新鲜，不参与风险汇总
            dx, dy, vx, vy = (float(value) for value in od_feats[slot, :4])
            ttc_x, ttc_path = slot_ttc(
                dx, dy, vx, vy, collision_radius_m=self.collision_radius_m
            )
            if math.isfinite(ttc_x):
                min_x = min(min_x, ttc_x)
                if ttc_x < self.horizon_s:
                    n_x += 1
            if math.isfinite(ttc_path):
                min_p = min(min_p, ttc_path)
                if ttc_path < self.horizon_s:
                    n_p += 1
            candidate = min(ttc_x, ttc_path)
            if math.isfinite(candidate) and (best is None or candidate < best[0]):
                best = (candidate, slot, bool(ttc_path < ttc_x))

        feats[0, 0] = min(min_x, cap) if math.isfinite(min_x) else cap
        feats[0, 1] = min(min_p, cap) if math.isfinite(min_p) else cap
        feats[0, 2] = float(n_x)
        feats[0, 3] = float(n_p)
        if best is None:
            feats[0, 4] = -1.0
            feats[0, 9] = cap
            feats[0, 11] = 0.0
        else:
            ttc, slot, is_path = best
            feats[0, 4] = float(slot)
            feats[0, 5:9] = od_feats[slot, :4]
            feats[0, 9] = min(ttc, cap)
            feats[0, 10] = 1.0 if is_path else 0.0
            feats[0, 11] = 1.0
        mask[0] = 1.0
        return feats, mask
