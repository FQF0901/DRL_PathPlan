"""世界系地图状态通道（schema v3，A4 nav 修正的 env 侧）：``route_world`` + ``ego_world``。

为什么需要
----------
旧 nav 通道（:mod:`env.obs.nav`）只在当前自车系给出两个 checkpoint + 命令 + route
completion；rollout 中自车前进后没有任何"世界系地图状态"可用于**重投影**，导致
imagined 轨迹后 5 步一直使用 t0 的 nav（P0-3）。本模块把导航路线以**世界系折线**入库：

- ``route_world (M,2)``：世界系路线折线。顶点 = 首段起点 + 每个路段的**路段终点**
  （road 中心横向偏移 ``later_middle``，与 ``BaseNavigation.get_checkpoints`` 的
  checkpoint 逐点同口径；首段起点为 spawn 路段起点）。折线覆盖**整条路线**（含已驶过
  部分），因此下游可在任意位姿下重算"下一个/下下个 checkpoint、route completion、
  下一分叉命令"（实现见 ``net/mem.rebuild_nav_from_world``）。
- ``ego_world (1,3)``：t0 世界系位姿 ``(x, y, θ)``，rollout 的位姿链锚点。

形状与填充
----------
``M = ROUTE_WORLD_MAX_POINTS``（本项目 spec 限 3–5 个 block，实测路线顶点 6–16，
64 有充足余量）。不足 M 时用**最后一个真实顶点**重复填充，``route_world_mask`` 把
补位点标 0；路线不可用（无 navigation / 非 NodeNetworkNavigation 无 ``checkpoints``）
时全 0 + mask 全 0 —— 下游据此回退旧行为（t0 nav 冻结 + 一次性告警）。

坐标系：世界系（不做 SE(2) 对齐，不进 6 帧历史；``memory.channels`` 不含这两个通道）。
"""

from __future__ import annotations

import numpy as np

from env.obs.base import ObservationChannel, make_empty, safe_ego

__all__ = [
    "ROUTE_WORLD_MAX_POINTS",
    "EGO_WORLD_DIM",
    "ROUTE_WORLD_DIM",
    "route_world_polyline",
    "EgoWorldChannel",
    "RouteWorldChannel",
]

#: 路线折线最大点数（固定形状；多余截断、不足重复末点补位）
ROUTE_WORLD_MAX_POINTS = 64
#: ego 世界位姿维数 (x, y, θ)
EGO_WORLD_DIM = 3
#: 路线折线点维数 (x, y)
ROUTE_WORLD_DIM = 2


def route_world_polyline(nav) -> np.ndarray:
    """从导航对象提取世界系路线折线 ``(K,2)``；不可用时返回 ``(0,2)``。

    顶点 = [首段起点, 每段终点]，其中路段 ``(a,b)`` 的终点取
    ``graph[a][b][0].position(length, later_middle)``（``later_middle`` 按该路段车道数
    取 road 中心，与上游 checkpoint 同口径）。首段起点取 ``position(0, later_middle)``。
    任何异常（缺 ``checkpoints``/图边/几何）→ 返回空数组，绝不抛异常。
    """
    try:
        checkpoints = list(getattr(nav, "checkpoints", None) or [])
        if len(checkpoints) < 2:
            return np.zeros((0, ROUTE_WORLD_DIM), dtype=np.float32)
        graph = nav.map.road_network.graph
        points: list[np.ndarray] = []
        for index, (start, end) in enumerate(zip(checkpoints[:-1], checkpoints[1:])):
            lanes = graph[start][end]
            if not lanes:
                return np.zeros((0, ROUTE_WORLD_DIM), dtype=np.float32)
            ref = lanes[0]
            later_middle = (float(len(lanes)) / 2.0 - 0.5) * float(ref.width)
            if index == 0:
                points.append(np.asarray(ref.position(0.0, later_middle), dtype=np.float32)[:2])
            points.append(np.asarray(ref.position(ref.length, later_middle), dtype=np.float32)[:2])
        return np.asarray(points, dtype=np.float32)[:ROUTE_WORLD_MAX_POINTS]
    except Exception:  # noqa: BLE001 - 观测通道不得因导航/地图异常而崩
        return np.zeros((0, ROUTE_WORLD_DIM), dtype=np.float32)


class EgoWorldChannel(ObservationChannel):
    """t0 世界系位姿 ``(x, y, θ)``（定长 1 槽）。"""

    name = "ego_world"
    feature_dim = EGO_WORLD_DIM

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(1, self.feature_dim)
        ego = safe_ego(env)
        if ego is None:
            return feats, mask
        try:
            position = np.asarray(ego.position, dtype=np.float32)[:2]
            feats[0, 0] = position[0]
            feats[0, 1] = position[1]
            feats[0, 2] = float(ego.heading_theta)
            mask[0] = 1.0
        except Exception:  # noqa: BLE001
            return feats, mask
        return feats, mask


class RouteWorldChannel(ObservationChannel):
    """世界系路线折线 ``(M,2)``（定长 M 槽 + 逐点 mask）。"""

    name = "route_world"
    feature_dim = ROUTE_WORLD_DIM
    num_slots = ROUTE_WORLD_MAX_POINTS

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(self.num_slots, self.feature_dim)
        ego = safe_ego(env)
        nav = getattr(ego, "navigation", None) if ego is not None else None
        if nav is None:
            return feats, mask
        polyline = route_world_polyline(nav)
        if polyline.shape[0] < 2:
            return feats, mask
        count = int(min(polyline.shape[0], self.num_slots))
        feats[:count] = polyline[:count]
        mask[:count] = 1.0
        if count < self.num_slots:  # 末点重复填充，mask 保持 0（下游不得把补位点当真实顶点）
            feats[count:] = polyline[count - 1]
        return feats, mask
