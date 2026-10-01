"""观测组装：当前帧通道 + 6 帧历史堆叠（schema v2）。

``ObservationBuilder.build(env, spec)`` 输出（全部 float32，除标注外）::

    ego (1,8)          ego_mask (1,)
    od  (16,9)         od_mask  (16,)      od_id (16,) int64    od_presence (16,)
    ld  (16,7)         ld_mask  (16,)
    nav (1,11)         nav_mask (1,)       # 兼容保留（others 是规范输入）
    signal (1,4)       signal_mask (1,)    # 兼容保留
    others (1,16+K)    others_mask (1,)    # nav + speed_limit + signal + road_class one-hot(K)
    ego_world (1,3)    ego_world_mask (1,) # v3：t0 世界系位姿 (x,y,θ)
    route_world (64,2) route_world_mask (64,)  # v3：世界系路线折线（首段起点 + 各段终点）

    ego_hist (6,1,8)           ego_hist_mask (6,1)
    others_hist (6,1,16+K)     others_hist_mask (6,1)
    od_hist (6,16,9)           od_hist_mask (6,16)
    od_id_hist (6,16) int64    od_presence_hist (6,16)
    ld_hist (6,16,7)           ld_hist_mask (6,16)
    hist_valid (6,)            # 预热补位帧为 0；由真实缓冲长度计算

``od_id`` 是槽位身份（见 ``env/obs/od.py``）：OD 槽位 = track id，跨帧稳定；``od_id_hist``/
``od_presence_hist`` 与 ``od_hist`` 同槽位、身份一致且不做 SE(2) 变换。``od_mask`` = 槽位
本帧有效（已分配），``od_presence`` = 对象本帧在盒内被观测（详见 od 通道文档）。

config（全部可选；``topk_objects``/``topk_lanes``/``history_frames`` 为 config/env.yaml 口径别名）::

    {
      "channels": ["ego", "od", "ld", "nav", "signal", "others"],  # 顺序即输出顺序（可裁剪）
      "scope": {"front_m": 150.0, "rear_m": 50.0, "left_m": 25.0, "right_m": 25.0},
      "od": {"num_slots": 16, "ttc_cap_s": 5.0, "release_after_s": 1.0},
      "ld": {"num_slots": 16, "offsets": [5, 10, 15, 20, 30]},
      "others": {"speed_limit_norm_mps": 30.0},
      "memory": {"frames": 6, "interval": 5, "channels": ["ego", "others", "od", "ld"]},
      "physics_dt": 0.1,
    }

``scope`` 是 OD/LD 共用的盒式范围（v2 默认 前 150 / 后 50 / 左右 25 m；旧默认前 100，
扩展理由见 ``env/obs/od.py`` 文档）。episode 边界由 ``env.episode_step == 0`` 自动识别
（env.reset 后第一次 build 会清空历史与槽位表），因此调用方只需要每步调用 ``build``；
自定义通道用 :meth:`register` 挂载（不改已有通道）。
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from env.obs.base import ObservationChannel
from env.obs.ego import EgoChannel
from env.obs.ld import LDChannel
from env.obs.memory import DEFAULT_CHANNELS as DEFAULT_MEMORY_CHANNELS
from env.obs.memory import FrameMemory
from env.obs.nav import NavChannel
from env.obs.od import ODChannel
from env.obs.others import OthersChannel
from env.obs.signal import SignalChannel
from env.obs.world import EgoWorldChannel, RouteWorldChannel

DEFAULT_CHANNELS: tuple[str, ...] = (
    "ego",
    "od",
    "ld",
    "nav",
    "signal",
    "others",
    # schema v3（A4）：世界系地图状态（rollout 逐步重算 nav 的输入；不进 6 帧历史）
    "ego_world",
    "route_world",
)
#: OD/LD 共用的默认盒式 scope（v2：前 150 / 后 50 / 左右 25）
DEFAULT_SCOPE: dict[str, float] = {"front_m": 150.0, "rear_m": 50.0, "left_m": 25.0, "right_m": 25.0}
_SCOPE_KEYS = ("front_m", "rear_m", "left_m", "right_m")


class ObservationBuilder:
    """按 config 组装当前帧通道与历史堆叠。"""

    def __init__(self, config: dict | None = None):
        cfg = dict(config or {})
        od_cfg = dict(cfg.get("od") or {})
        ld_cfg = dict(cfg.get("ld") or {})
        others_cfg = dict(cfg.get("others") or {})
        mem_cfg = dict(cfg.get("memory") or {})
        # config/env.yaml 扁平键别名，方便 pipeline 直接透传
        if "topk_objects" in cfg:
            od_cfg.setdefault("num_slots", cfg["topk_objects"])
        if "topk_lanes" in cfg:
            ld_cfg.setdefault("num_slots", cfg["topk_lanes"])
        if "history_frames" in cfg:
            mem_cfg.setdefault("frames", cfg["history_frames"])
        scope_cfg = dict(DEFAULT_SCOPE)
        scope_cfg.update(dict(cfg.get("scope") or {}))
        for key in _SCOPE_KEYS:
            if key in cfg:  # 兼容把 scope 写成一层的旧配置
                scope_cfg[key] = cfg[key]
            od_cfg.setdefault(key, scope_cfg[key])
            ld_cfg.setdefault(key, scope_cfg[key])
        self.physics_dt = float(cfg.get("physics_dt", 0.1))

        built_in: dict[str, ObservationChannel] = {
            "ego": EgoChannel(physics_dt=self.physics_dt),
            "od": ODChannel(**{**od_cfg, "physics_dt": float(od_cfg.get("physics_dt", self.physics_dt))}),
            "ld": LDChannel(**ld_cfg),
            "nav": NavChannel(),
            "signal": SignalChannel(),
            "others": OthersChannel(**others_cfg),
            "ego_world": EgoWorldChannel(),
            "route_world": RouteWorldChannel(),
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
        companions: dict[str, dict[str, np.ndarray]] = {}
        for name, channel in self.channels.items():
            feats, mask = channel.build(env, spec)
            built[name] = (np.asarray(feats, dtype=np.float32), np.asarray(mask, dtype=np.float32))
            getter = getattr(channel, "companions", None)
            if getter is not None:
                try:
                    values = getter(env, spec) or {}
                except Exception:  # noqa: BLE001 - 伴随数组失败不应影响主观测
                    values = {}
                if values:
                    companions[name] = {key: np.asarray(value) for key, value in values.items()}

        self.memory.push(env, spec, built, companions=companions)

        out: dict[str, np.ndarray] = {}
        for name, (feats, mask) in built.items():
            out[name] = feats
            out[f"{name}_mask"] = mask
        for values in companions.values():
            out.update(values)
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
