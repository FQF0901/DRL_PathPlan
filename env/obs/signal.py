"""signal 通道：4 维占位（恒为"无灯"）。

本项目不使用交通灯（config/env.yaml `traffic_lights: false`；PG 地图本来也不放红绿灯）。
占位语义与 MetaDrive 的 4 种灯态一一对应（``metadrive/type.py:57-60``）::

    [green, yellow, red, unknown] == [0, 0, 0, 1]

为什么不是全 0：全 0 与"网络没看到输入"不可区分；固定 [0,0,0,1] 明确表达"当前无灯/未知"，
未来接入信号灯时只需把真实灯态写入同一槽位，网络无需改输入维度。mask 恒为 1（这是有效观测）。
"""

from __future__ import annotations

import numpy as np

from env.obs.base import ObservationChannel, make_empty

SIGNAL_DIM = 4
NO_LIGHT_INDEX = 3  # unknown / 无灯


class SignalChannel(ObservationChannel):
    """信号占位通道（定长 1 槽）。"""

    name = "signal"
    feature_dim = SIGNAL_DIM

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(1, self.feature_dim)
        feats[0, NO_LIGHT_INDEX] = 1.0
        mask[0] = 1.0
        return feats, mask
