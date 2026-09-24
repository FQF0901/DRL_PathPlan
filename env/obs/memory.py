"""6 帧 @0.5 s 历史记忆（每 5 个 env step 存 1 帧），SE(2) 对齐到当前自车系。

设计要点（对应契约"每帧存自车系特征 + 该帧 ego 位姿，取用时 SE(2) 对齐到当前帧"）：

- ``push(env, spec, built)``：**每个 env step 调用一次**（对同一 step 幂等）；只有
  ``episode_step % interval == 0`` 的帧会入库，因此相邻入库帧严格相隔 interval 个 env step
  （0.1 s × 5 = 0.5 s）。入库内容 = 该帧自车系下的通道特征/掩码（``built``，由 builder 传入，
  不重复计算）+ 该帧 ego 位姿 ``(x, y, θ)``；
- ``stack(env)``：把每帧特征用 :func:`env.obs.base.se2_align` 重表达到**当前**自车系；
  帧按"旧 → 新"排列，最新一帧是最近一次采样帧（episode_step 为 interval 整数倍时即当前帧）；
- 预热期（入库帧数 < frames）：重复**最旧的真实帧**补满窗口，并用 ``hist_valid``
  把这些补位帧标 0（真实帧标 1），避免把复制帧当成真实历史；
- 时间戳异常（step 非严格递增）直接 raise：这通常意味着调用方绕过了 builder 的顺序，
  属于"与物理时间相悖"的早期信号，宁可报错。

``push(env)`` 单独调用时（不传 ``built``）会退化为用 :meth:`bind` 注入的通道对象自行构建，
以兼容契约里的简写签名；builder 路径始终传 ``built`` 以避免重复计算。
"""

from __future__ import annotations

from collections import deque
from typing import Mapping

import numpy as np

from env.obs.base import FrameAlignment, ObservationChannel, safe_ego, se2_align


def ego_pose(env) -> np.ndarray:
    """当前 ego 位姿 ``(x, y, θ)``（世界系）。"""
    ego = safe_ego(env)
    if ego is None:
        return np.zeros(3, dtype=np.float32)
    pos = np.asarray(ego.position, dtype=np.float32)[:2]
    return np.array([pos[0], pos[1], float(ego.heading_theta)], dtype=np.float32)


class FrameMemory:
    """定长历史帧缓存 + 对齐输出。"""

    def __init__(
        self,
        *,
        frames: int = 6,
        interval: int = 5,
        channels: tuple[str, ...] = ("od", "ld"),
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

    @property
    def num_stored(self) -> int:
        """已入库帧数（预热期 < frames）。"""
        return len(self._buf)

    # ------------------------------------------------------------------- 入库
    def push(self, env, spec=None, built: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None) -> None:
        """按 interval 采样入库；同一 ``episode_step`` 重复调用无副作用。"""
        step = int(getattr(env, "episode_step", 0))
        if step == 0:
            self.clear()  # env.reset 后首次调用：开新 episode
        if self._last_step == step:
            return
        self._last_step = step
        if step % self.interval != 0:
            return

        frame = self._capture(env, step, spec, built)
        if frame is None:
            return
        if self._buf and step <= self._buf[-1]["step"]:
            raise RuntimeError(
                f"FrameMemory 时间戳异常：新帧 step={step} <= 上一帧 step={self._buf[-1]['step']}；"
                "请检查 push 调用顺序是否与 env.step 一致"
            )
        self._buf.append(frame)

    def _capture(self, env, step: int, spec, built) -> dict | None:
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
        for name in self.channels:
            if name not in built:
                continue
            feats, mask = built[name]
            features[name] = np.array(feats, dtype=np.float32, copy=True)
            masks[name] = np.array(mask, dtype=np.float32, copy=True)
        if not features:
            return None
        return {"step": step, "pose": ego_pose(env), "features": features, "masks": masks}

    # ------------------------------------------------------------------- 读取
    def _alignment(self, name: str) -> FrameAlignment:
        channel = self._channel_map.get(name)
        return getattr(channel, "alignment", FrameAlignment())

    def stack(self, env) -> dict[str, np.ndarray]:
        """返回历史堆叠：``<ch>_hist (frames,N,F)``、``<ch>_hist_mask (frames,N)``、``hist_valid (frames,)``。"""
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

        out["hist_valid"] = valid
        return out
