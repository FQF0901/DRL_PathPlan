"""观测组装：当前帧通道 + 6 帧历史堆叠。

``ObservationBuilder.build(env, spec)`` 输出（全部 float32）::

    ego (1,8)          ego_mask (1,)
    od  (16,9)         od_mask  (16,)
    ld  (16,7)         ld_mask  (16,)
    nav (1,11)         nav_mask (1,)
    signal (1,4)       signal_mask (1,)
    od_hist (6,16,9)   od_hist_mask (6,16)
    ld_hist (6,16,7)   ld_hist_mask (6,16)
    hist_valid (6,)    # 预热补位帧为 0

config（全部可选；``topk_objects``/``topk_lanes``/``history_frames`` 为 config/env.yaml 口径别名）::

    {
      "channels": ["ego", "od", "ld", "nav", "signal"],   # 顺序即输出顺序（可裁剪）
      "od": {"num_slots": 16, "range_m": 100.0, "ttc_cap_s": 5.0},
      "ld": {"num_slots": 16, "offsets": [5, 10, 15, 20, 30]},
      "memory": {"frames": 6, "interval": 5, "channels": ["od", "ld"]},
      "physics_dt": 0.1,
    }

episode 边界由 ``env.episode_step == 0`` 自动识别（env.reset 后第一次 build 会清空历史），
因此调用方只需要每步调用 ``build``；自定义通道用 :meth:`register` 挂载（不改已有通道）。
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from env.obs.base import ObservationChannel
from env.obs.ego import EgoChannel
from env.obs.ld import LDChannel
from env.obs.memory import FrameMemory
from env.obs.nav import NavChannel
from env.obs.od import ODChannel
from env.obs.signal import SignalChannel

DEFAULT_CHANNELS: tuple[str, ...] = ("ego", "od", "ld", "nav", "signal")
DEFAULT_MEMORY_CHANNELS: tuple[str, ...] = ("od", "ld")


class ObservationBuilder:
    """按 config 组装当前帧通道与历史堆叠。"""

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        od_cfg = dict(cfg.get("od") or {})
        ld_cfg = dict(cfg.get("ld") or {})
        mem_cfg = dict(cfg.get("memory") or {})
        # config/env.yaml 扁平键别名，方便 pipeline 直接透传
        if "topk_objects" in cfg:
            od_cfg.setdefault("num_slots", cfg["topk_objects"])
        if "topk_lanes" in cfg:
            ld_cfg.setdefault("num_slots", cfg["topk_lanes"])
        if "history_frames" in cfg:
            mem_cfg.setdefault("frames", cfg["history_frames"])
        self.physics_dt = float(cfg.get("physics_dt", 0.1))

        built_in: dict[str, ObservationChannel] = {
            "ego": EgoChannel(physics_dt=self.physics_dt),
            "od": ODChannel(**od_cfg),
            "ld": LDChannel(**ld_cfg),
            "nav": NavChannel(),
            "signal": SignalChannel(),
        }
        wanted = tuple(cfg.get("channels") or DEFAULT_CHANNELS)
        unknown = [name for name in wanted if name not in built_in]
        if unknown:
            raise ValueError(f"未知观测通道 {unknown}；内置通道：{sorted(built_in)}")
        self.channels: OrderedDict[str, ObservationChannel] = OrderedDict((name, built_in[name]) for name in wanted)

        self.memory = FrameMemory(
            frames=int(mem_cfg.get("frames", 6)),
            interval=int(mem_cfg.get("interval", 5)),
            channels=tuple(mem_cfg.get("channels") or DEFAULT_MEMORY_CHANNELS),
        )

    # ---------------------------------------------------------------- 组装
    def register(self, channel: ObservationChannel) -> None:
        """挂载自定义通道（实现 ObservationChannel 即可扩展，不改已有通道）。"""
        self.channels[channel.name] = channel

    def reset(self) -> None:
        """清空所有跨 step 状态（env.reset 后 build 会自动调用；也可手动调）。"""
        for channel in self.channels.values():
            channel.reset()
        self.memory.clear()

    def build(self, env, spec=None) -> dict[str, np.ndarray]:
        """构建一步观测；``spec`` 可为 None（通道不依赖 spec 时）。"""
        if int(getattr(env, "episode_step", 0)) == 0:
            self.reset()
        self.memory.bind(self.channels)

        built: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        for name, channel in self.channels.items():
            feats, mask = channel.build(env, spec)
            built[name] = (np.asarray(feats, dtype=np.float32), np.asarray(mask, dtype=np.float32))

        self.memory.push(env, spec, built)

        out: dict[str, np.ndarray] = {}
        for name, (feats, mask) in built.items():
            out[name] = feats
            out[f"{name}_mask"] = mask
        out.update(self.memory.stack(env))
        return out

    # ---------------------------------------------------------------- 元信息
    @property
    def feature_spec(self) -> dict[str, tuple[int, int]]:
        """``{通道名: (槽位数, 特征维)}``，供网络层核对输入维度。"""
        return {
            name: (int(getattr(channel, "num_slots", 1)), int(channel.feature_dim))
            for name, channel in self.channels.items()
        }
