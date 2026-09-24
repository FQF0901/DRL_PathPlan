"""nav 通道：2 个 checkpoint（自车系 x,y）+ 命令 one-hot（forward/left/right + 3 保留）+ route_completion。

- checkpoint 来自 ``vehicle.navigation.get_checkpoints()``（``component/navigation_module/base_navigation.py:153``），
  **返回世界系坐标**，这里用 ``ego.convert_to_local_coordinates`` 转成自车系（与 lidar 上游用法一致，
  ``component/sensors/lidar.py:84-90``）。不裁剪距离——下游（router/规划）需要真实距离；
- 命令与 ``info["navigation_command"]`` 同口径（``component/vehicle/base_vehicle.py:254-277``）：
  由 ``navigation.navi_arrow_dir`` 的两段航向差判断 forward / left / right，不计算"到转弯距离"
  （config/env.yaml `distance_to_turn: false`）；
- ``route_completion`` = ``navigation.route_completion``（travelled/total，node_network_navigation.py:367）。

维度 11 = 2×2 (checkpoints) + 6 (one-hot: forward,left,right,reserved×3) + 1 (route_completion)。
"""

from __future__ import annotations

import math

import numpy as np

from env.obs.base import FrameAlignment, ObservationChannel, make_empty, safe_ego, wrap_to_pi

COMMANDS: tuple[str, ...] = ("forward", "left", "right")
NUM_COMMANDS = 6  # 3 实际 + 3 保留
NUM_CHECKPOINTS = 2
TURN_EPS_RAD = math.radians(10.0)  # 与 base_vehicle.py:261 的 10° 阈值一致


def navigation_command(nav) -> str:
    """从 ``navi_arrow_dir`` 还原 forward/left/right（与上游 step_info 逻辑逐行一致）。"""
    arrow = getattr(nav, "navi_arrow_dir", None)
    if arrow is None or len(arrow) < 2:
        return "forward"
    h0, h1 = float(arrow[0]), float(arrow[1])
    if abs(float(wrap_to_pi(h0 - h1))) < TURN_EPS_RAD:
        return "forward"
    dir0 = np.array([math.cos(h0), math.sin(h0), 0.0])
    dir1 = np.array([math.cos(h1), math.sin(h1), 0.0])
    cross_z = float(np.cross(dir1, dir0)[-1])
    return "left" if cross_z < 0 else "right"


class NavChannel(ObservationChannel):
    """导航通道（定长 1 槽）。"""

    name = "nav"
    feature_dim = 2 * 2 + NUM_COMMANDS + 1
    alignment = FrameAlignment(point_pairs=((0, 1), (2, 3)))  # 两个 checkpoint 是点

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(1, self.feature_dim)
        ego = safe_ego(env)
        nav = getattr(ego, "navigation", None) if ego is not None else None
        if ego is None or nav is None:
            return feats, mask

        try:
            checkpoints = list(nav.get_checkpoints())
        except Exception:  # noqa: BLE001 - 导航不可用时返回 mask=0，不让观测通道崩掉
            checkpoints = []
        if not checkpoints:
            return feats, mask

        for k, ckpt in enumerate(checkpoints[:NUM_CHECKPOINTS]):
            ckpt = np.asarray(ckpt, dtype=np.float32)[:2]
            rel = np.asarray(ego.convert_to_local_coordinates(ckpt, ego.position), dtype=np.float32)
            feats[0, 2 * k:2 * k + 2] = rel

        cmd = navigation_command(nav)
        if cmd in COMMANDS:
            feats[0, 4 + COMMANDS.index(cmd)] = 1.0
        # 3 个保留位保持 0

        try:
            rc = float(nav.route_completion)
        except Exception:  # noqa: BLE001
            rc = 0.0
        feats[0, 4 + NUM_COMMANDS] = rc
        mask[0] = 1.0
        return feats, mask
