"""6 帧 @0.5 s 历史记忆（每 5 个 env step 存 1 帧），SE(2) 对齐到当前自车系（schema v2）。

设计要点：
- ``push(env, spec, built, companions)``：**每个 env step 调用一次**（对同一 step 幂等）；
  只有 ``episode_step % interval == 0`` 的帧会入库，因此相邻入库帧严格相隔 interval 个 env step
  （0.1 s × 5 = 0.5 s）。入库内容 = 该帧自车系下的通道特征/掩码 + 槽位级伴随数组
  （``od_id``/``od_presence``，不做 SE(2) 变换）+ 该帧 ego 位姿 ``(x, y, θ)``；
- ``stack(env)``：把每帧特征用 :func:`env.obs.base.se2_align` 重表达到**当前**自车系；
  帧按"旧 → 新"排列，最新一帧是最近一次采样帧（episode_step 为 interval 整数倍时即当前帧）；
- 预热期（入库帧数 < frames）：重复**最旧的真实帧**补满窗口，并用 ``hist_valid``
  把这些补位帧标 0（真实帧标 1），避免把复制帧当成真实历史；
  ``hist_valid`` **由真实缓冲长度决定**（``valid[-num_stored:] = 1``），禁止"位置语义"；
- 时间戳异常（step 非严格递增）直接 raise：这通常意味着调用方绕过了 builder 的顺序，
  属于"与物理时间相悖"的早期信号，宁可报错。

**v2 历史通道**（默认 ``("ego", "others", "od", "ld")``）：``od_id`` 是槽位身份，历史里的
``od_id_hist``/``od_presence_hist`` 与 ``od_hist`` 同槽位、跨帧身份一致（OD 通道按 track id
固定槽位，见 ``env/obs/od.py``）；``ego_hist``/``others_hist`` 是单槽通道历史。
LD 保持 v1 语义（只作输入，无时序身份需求）。

``push(env)`` 单独调用时（不传 ``built``）会退化为用 :meth:`bind` 注入的通道对象自行构建，
以兼容契约里的简写签名；builder 路径始终传 ``built`` 以避免重复计算。
"""

from __future__ import annotations

from collections import deque
from typing import Mapping

import numpy as np

from env.obs.base import FrameAlignment, ObservationChannel, companion_fill_value, safe_ego, se2_align

#: v2 默认历史通道（与 builder.DEFAULT_MEMORY_CHANNELS 一致）
DEFAULT_CHANNELS: tuple[str, ...] = ("ego", "others", "od", "ld")


def ego_pose(env) -> np.ndarray:
    """当前 ego 位姿 ``(x, y, θ)``（世界系）。"""
    ego = safe_ego(env)
    if ego is None:
        return np.zeros(3, dtype=np.float32)
    pos = np.asarray(ego.position, dtype=np.float32)[:2]
    return np.array([pos[0], pos[1], float(ego.heading_theta)], dtype=np.float32)


class FrameMemory:
    """定长历史帧缓存 + 对齐输出（含槽位级伴随数组历史）。"""

    def __init__(
        self,
        *,
        frames: int = 6,
        interval: int = 5,
        channels: tuple[str, ...] = DEFAULT_CHANNELS,
    ):
        if frames < 1 or interval < 1:
            raise ValueError("frames/interval 必须 >= 1")
        self.frames = int(frames)
        self.interval = int(interval)
        self.channels = tuple(channels)
        self._buf: deque[dict] = deque(maxlen=self.frames)
        self._channel_map: dict[str, ObservationChannel] = {}
        self._last_step: int | None = None

    # ---------------------------------------------------------------- 绑定与清理
    def bind(self, channels: Mapping[str, ObservationChannel]) -> None:
        """注入通道对象（取它们的 alignment 元数据）；builder 每次 build 调用，幂等。"""
        self._channel_map = dict(channels)

    def clear(self) -> None:
        """清空历史（episode 边界）。"""
        self._buf.clear()
        self._last_step = None

    # ------------------------------------------------------------------- 入库
    def push(
        self,
        env,
        spec=None,
        built: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
        companions: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> None:
        """按 interval 采样入库；同一 ``episode_step`` 重复调用无副作用。"""
        step = int(getattr(env, "episode_step", 0))
        if step == 0:
            self.clear()  # env.reset 后首次调用：开新 episode
        if self._last_step == step:
            return
        self._last_step = step
        if step % self.interval != 0:
            return

        frame = self._capture(env, step, spec, built, companions)
        if frame is None:
            return
        if self._buf and step <= self._buf[-1]["step"]:
            raise RuntimeError(
                f"FrameMemory 时间戳异常：新帧 step={step} <= 上一帧 step={self._buf[-1]['step']}；"
                "请检查 push 调用顺序是否与 env.step 一致"
            )
        self._buf.append(frame)

    def _capture(
        self,
        env,
        step: int,
        spec,
        built: Mapping[str, tuple[np.ndarray, np.ndarray]] | None,
        companions: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ) -> dict | None:
        if built is None:
            if not self._channel_map:
                return None
            built = {
                name: self._channel_map[name].build(env, spec)
                for name in self.channels
                if name in self._channel_map
            }
        features: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        kept: dict[str, dict[str, np.ndarray]] = {}
        for name in self.channels:
            if name not in built:
                continue
            feats, mask = built[name]
            features[name] = np.array(feats, dtype=np.float32, copy=True)
            masks[name] = np.array(mask, dtype=np.float32, copy=True)
            values = (companions or {}).get(name)
            if values is None:
                channel = self._channel_map.get(name)
                getter = getattr(channel, "companions", None) if channel is not None else None
                if getter is not None:
                    try:
                        values = getter(env, spec) or {}
                    except Exception:  # noqa: BLE001 - 伴随数组失败不应让历史崩掉
                        values = {}
            if values:
                kept[name] = {key: np.array(value, copy=True) for key, value in values.items()}
        if not features:
            return None
        return {"step": step, "pose": ego_pose(env), "features": features, "masks": masks, "companions": kept}

    # ------------------------------------------------------------------- 读取
    def _alignment(self, name: str) -> FrameAlignment:
        channel = self._channel_map.get(name)
        return getattr(channel, "alignment", FrameAlignment())

    def _companion_spec(self, name: str, padded: list) -> dict[str, tuple[tuple, np.dtype, object]]:
        """发现该通道伴随数组规格 ``{key: (shape, dtype, fill)}``（取最新帧的样本）。"""
        channel = self._channel_map.get(name)
        fills = getattr(channel, "companion_fill", {}) or {}
        for frame in reversed(padded):
            values = (frame.get("companions") or {}).get(name)
            if not values:
                continue
            spec: dict[str, tuple[tuple, np.dtype, object]] = {}
            for key, value in values.items():
                arr = np.asarray(value)
                spec[key] = (tuple(arr.shape), arr.dtype, companion_fill_value(key, arr.dtype, fills))
            return spec
        return {}

    def stack(self, env) -> dict[str, np.ndarray]:
        """返回历史堆叠：``<ch>_hist``/``<ch>_hist_mask``/伴随 ``<key>_hist`` + ``hist_valid``。

        ``hist_valid`` 由**真实缓冲长度**决定：``valid[-num_stored:] = 1``，前面的补位帧为 0。
        """
        current = ego_pose(env)
        real = list(self._buf)
        out: dict[str, np.ndarray] = {}
        valid = np.zeros((self.frames, ), dtype=np.float32)
        if not real:
            out["hist_valid"] = valid
            return out
        # 预热：重复最旧真实帧补满窗口（补位帧 valid=0）
        padded = [real[0]] * (self.frames - len(real)) + real
        valid[self.frames - len(real):] = 1.0

        for name in self.channels:
            sample = None
            for frame in reversed(padded):
                if name in frame["features"]:
                    sample = frame["features"][name]
                    break
            if sample is None:
                continue
            num_slots, dim = int(sample.shape[0]), int(sample.shape[1])
            feats_hist = np.zeros((self.frames, num_slots, dim), dtype=np.float32)
            mask_hist = np.zeros((self.frames, num_slots), dtype=np.float32)
            alignment = self._alignment(name)
            for k, frame in enumerate(padded):
                feats = frame["features"].get(name)
                mask = frame["masks"].get(name)
                if feats is None or feats.shape != (num_slots, dim):
                    continue
                pose = frame["pose"]
                aligned = se2_align(
                    feats,
                    alignment=alignment,
                    delta_theta=float(current[2] - pose[2]),
                    delta_xy=current[:2] - pose[:2],
                    current_theta=float(current[2]),
                )
                feats_hist[k] = aligned
                mask_hist[k] = mask
            out[f"{name}_hist"] = feats_hist
            out[f"{name}_hist_mask"] = mask_hist

            for key, (shape, dtype, fill) in self._companion_spec(name, padded).items():
                history = np.full((self.frames, ) + shape, fill, dtype=dtype)
                for k, frame in enumerate(padded):
                    value = (frame.get("companions") or {}).get(name, {}).get(key)
                    if value is not None:
                        history[k] = value
                # 补位帧（hist_valid=0）不是真实观测：身份/存在性一律填缺省值，
                # 保证"存储的历史"与 pipeline.frames 精确查表结果逐位一致（v2 契约）
                for k in range(self.frames):
                    if valid[k] < 0.5:
                        history[k] = fill
                out[f"{key}_hist"] = history

        out["hist_valid"] = valid
        return out
