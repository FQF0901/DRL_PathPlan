"""Rollout 缓冲（P2 契约 §4）：**按帧存储** + 历史窗口在线重建 + GAE(λ)。

RAM 关键设计（契约明确要求）
---------------------------
**绝不缓存 6 帧堆叠**。每帧只存当前帧通道与掩码 + 该帧 ego 位姿，历史窗口在
``build_history`` 里按需重建（与 ``env/obs/memory.FrameMemory`` 完全同口径：
每 ``interval`` 个 env step 采一帧、按 SE(2) 对齐到目标帧、预热期复制最旧真实帧
并把 ``hist_valid`` 置 0）。单帧 ≈ 323 个 float32 ≈ 1.3 KB；若直接存 od/ld 历史堆叠
要多出 ``6×16×(9+7)=1536`` 个 float（≈6 KB/帧，约 4.8×），万级 rollout 会显著吃掉
16GB 内存预算。

接口
----
- :meth:`RolloutBuffer.add_step`：逐帧入库（env-step 粒度，0.1 s）；
- :meth:`RolloutBuffer.build_history`：给定帧下标批量重建历史窗口
  （``<ch>_hist (B,6,N,F)`` / ``<ch>_hist_mask (B,6,N)`` / ``hist_valid (B,6)``）；
- :meth:`RolloutBuffer.compute_gae`：GAE(λ) 优势/回报（按 episode 切断，终局不 bootstrap，
  截断处按 ``truncation_bootstrap`` 处理）；
- :meth:`RolloutBuffer.episode_slices`：按 episode 的连续区间（PPO 分段用）。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = ["RolloutBuffer", "DEFAULT_CHANNELS"]

#: 与 ``env/obs/builder.ObservationBuilder`` 默认输出一致的通道形状（含槽位维）：
#: ego(1,8) / od(16,9) / ld(16,7) / nav(1,11) / signal(1,4)。
DEFAULT_CHANNELS: dict[str, tuple[int, ...]] = {
    "ego": (1, 8),
    "od": (16, 9),
    "ld": (16, 7),
    "nav": (1, 11),
    "signal": (1, 4),
}
#: 默认重建历史的通道（与 ``config/env.yaml`` 的 memory.channels 一致）
DEFAULT_HISTORY_CHANNELS: tuple[str, ...] = ("od", "ld")


def _frame_pose(env: Any) -> np.ndarray:
    """从 env 读当前 ego 位姿 ``(x, y, theta)``；不可用返回零（历史对齐退化为不平移）。"""
    try:
        ego = env.agent
        position = np.asarray(ego.position, dtype=np.float32)
        return np.array([position[0], position[1], float(ego.heading_theta)], dtype=np.float32)
    except Exception:  # noqa: BLE001 - 缓冲不应因观测失败而中断
        return np.zeros(3, dtype=np.float32)


def _alignments_for(channels: Sequence[str]) -> dict:
    """按通道取 SE(2) 对齐规则（惰性导入，避免缓冲模块把 MetaDrive 拉进导入链）。"""
    from env.obs.base import FrameAlignment
    from env.obs.ld import LDChannel
    from env.obs.nav import NavChannel
    from env.obs.od import ODChannel

    known = {
        "od": ODChannel.alignment,
        "ld": LDChannel.alignment,
        "nav": NavChannel.alignment,
        "ego": FrameAlignment(),
        "signal": FrameAlignment(),
    }
    return {name: known.get(name, FrameAlignment()) for name in channels}


class RolloutBuffer:
    """固定容量、按帧存储的 rollout 缓冲。"""

    def __init__(
        self,
        capacity: int,
        *,
        channels: Optional[Mapping[str, Sequence[int]]] = None,
        history_frames: int = 6,
        history_interval: int = 5,
        history_channels: Sequence[str] = DEFAULT_HISTORY_CHANNELS,
    ):
        """
        Args:
            capacity: 帧容量（env-step 数）。
            channels: ``{通道名: 槽位形状}``；``None`` = :data:`DEFAULT_CHANNELS`。
            history_frames: 历史窗口帧数（6 帧 @0.5 s）。
            history_interval: 采样间隔（env-step 数，5 = 0.5 s）。
            history_channels: 需要重建历史的通道（默认 od/ld，与 builder 一致）。
        """
        if int(capacity) < 1:
            raise ValueError(f"capacity 必须 >= 1，收到 {capacity}")
        if int(history_frames) < 1 or int(history_interval) < 1:
            raise ValueError("history_frames / history_interval 必须 >= 1")
        self.capacity = int(capacity)
        self.channels = {name: tuple(int(v) for v in shape) for name, shape in (channels or DEFAULT_CHANNELS).items()}
        self.history_frames = int(history_frames)
        self.history_interval = int(history_interval)
        self.history_channels = tuple(history_channels)

        self.obs: dict[str, np.ndarray] = {
            name: np.zeros((self.capacity, ) + shape, dtype=np.float32) for name, shape in self.channels.items()
        }
        self.obs_mask: dict[str, np.ndarray] = {
            name: np.zeros((self.capacity, ) + shape[:-1], dtype=np.float32)
            for name, shape in self.channels.items()
        }
        self.pose = np.zeros((self.capacity, 3), dtype=np.float32)
        self.action = np.zeros((self.capacity, 2), dtype=np.float32)
        self.logprob = np.zeros(self.capacity, dtype=np.float32)
        self.value = np.zeros(self.capacity, dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=bool)
        self.truncated = np.zeros(self.capacity, dtype=bool)
        self.episode = np.zeros(self.capacity, dtype=np.int32)
        self.step_index = np.zeros(self.capacity, dtype=np.int32)
        self._len = 0
        self._alignments: Optional[dict] = None

    # ------------------------------------------------------------------ 基本信息
    def __len__(self) -> int:
        return self._len

    @property
    def full(self) -> bool:
        return self._len >= self.capacity

    def clear(self) -> None:
        """清空（不释放底层数组）。"""
        self._len = 0

    # ------------------------------------------------------------------ 入库
    def add_step(
        self,
        obs: Mapping[str, np.ndarray],
        *,
        pose: Optional[Sequence[float]] = None,
        env: Any = None,
        action: Optional[Sequence[float]] = None,
        logprob: float = 0.0,
        value: float = 0.0,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        episode: Optional[int] = None,
        step: Optional[int] = None,
    ) -> int:
        """写入一帧（当前帧通道 + 掩码 + 位姿 + 动作/回报等元信息），返回帧下标。

        ``obs`` 只取**当前帧**通道（``ego/od/ld/nav/signal`` 与 ``*_mask``），
        ``*_hist`` 之类的键被忽略。``episode``/``step`` 缺省时按"上一帧终局则新
        episode、步号自增"自动维护，足以还原 FrameMemory 的采样节奏。
        """
        if self._len >= self.capacity:
            raise BufferError(f"RolloutBuffer 已满（capacity={self.capacity}）")
        index = self._len
        episode_id = self._resolve_episode(episode)
        step_id = self._resolve_step(step, episode_id)

        for name, shape in self.channels.items():
            self.obs[name][index] = self._frame_channel(obs, name, shape)
            mask = obs.get(f"{name}_mask")
            self.obs_mask[name][index] = self._frame_mask(mask, shape[:-1], name)

        resolved_pose = pose if pose is not None else (_frame_pose(env) if env is not None else None)
        self.pose[index] = np.zeros(3, dtype=np.float32) if resolved_pose is None else np.asarray(
            resolved_pose, dtype=np.float32
        ).reshape(3)
        if action is not None:
            self.action[index] = np.asarray(action, dtype=np.float32).reshape(2)
        self.logprob[index] = float(logprob)
        self.value[index] = float(value)
        self.reward[index] = float(reward)
        self.terminated[index] = bool(terminated)
        self.truncated[index] = bool(truncated)
        self.episode[index] = episode_id
        self.step_index[index] = step_id
        self._len += 1
        return index

    #: 别名（语义更直白）
    add = add_step

    def _resolve_episode(self, episode: Optional[int]) -> int:
        if episode is not None:
            return int(episode)
        if self._len == 0:
            return 0
        previous = self._len - 1
        ended = bool(self.terminated[previous] or self.truncated[previous])
        return int(self.episode[previous]) + (1 if ended else 0)

    def _resolve_step(self, step: Optional[int], episode_id: int) -> int:
        if step is not None:
            return int(step)
        if self._len == 0 or int(self.episode[self._len - 1]) != episode_id:
            return 0
        return int(self.step_index[self._len - 1]) + 1

    @staticmethod
    def _frame_channel(obs: Mapping[str, np.ndarray], name: str, shape: tuple) -> np.ndarray:
        value = obs.get(name)
        if value is None:
            return np.zeros(shape, dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32)
        if arr.shape != shape:
            raise ValueError(f"缓冲通道 {name} 期望 shape={shape}，收到 {arr.shape}")
        return arr

    @staticmethod
    def _frame_mask(mask: Any, shape: tuple, name: str) -> np.ndarray:
        if mask is None:
            return np.zeros(shape, dtype=np.float32)
        arr = np.asarray(mask, dtype=np.float32)
        if arr.shape != shape:
            raise ValueError(f"缓冲掩码 {name}_mask 期望 shape={shape}，收到 {arr.shape}")
        return arr

    # ------------------------------------------------------------------ 历史重建
    def build_history(
        self,
        indices: Any,
        *,
        frames: Optional[int] = None,
        interval: Optional[int] = None,
        current_pose: Optional[Sequence[float]] = None,
    ) -> dict[str, np.ndarray]:
        """在线重建指定帧的历史窗口（与 ``FrameMemory.stack`` 同口径）。

        Args:
            indices: 帧下标（int 或序列）。
            frames / interval: 覆盖默认窗口/采样间隔。
            current_pose: 对齐目标位姿；``None`` = 用各帧自身位姿（即"以该帧为当前帧"）。

        Returns:
            ``{"<ch>_hist": (B,F,N,Dim), "<ch>_hist_mask": (B,F,N), "hist_valid": (B,F)}``；
            标量 ``indices`` 时去掉 batch 维。
        """
        frames = self.history_frames if frames is None else int(frames)
        interval = self.history_interval if interval is None else int(interval)
        if frames < 1 or interval < 1:
            raise ValueError("frames / interval 必须 >= 1")
        if self._alignments is None:
            self._alignments = _alignments_for(self.history_channels)
        batched = np.ndim(indices) > 0
        index_array = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        out: dict[str, np.ndarray] = {}
        for name in self.history_channels:
            shape = self.channels.get(name)
            if shape is None:
                raise KeyError(f"history_channels 含未知通道 {name!r}")
            out[f"{name}_hist"] = np.zeros((len(index_array), frames) + shape, dtype=np.float32)
            out[f"{name}_hist_mask"] = np.zeros((len(index_array), frames) + shape[:-1], dtype=np.float32)
        out["hist_valid"] = np.zeros((len(index_array), frames), dtype=np.float32)

        for batch_index, frame_index in enumerate(index_array):
            if not 0 <= frame_index < self._len:
                raise IndexError(f"帧下标越界：{frame_index}（当前 {self._len} 帧）")
            real = self._sample_frames(int(frame_index), frames, interval)
            if not real:
                continue
            padded = [real[0]] * (frames - len(real)) + real
            out["hist_valid"][batch_index, frames - len(real):] = 1.0
            target_pose = (
                np.asarray(current_pose, dtype=np.float32).reshape(3) if current_pose is not None
                else self.pose[frame_index]
            )
            for slot, source in enumerate(padded):
                source_pose = self.pose[source]
                delta_theta = float(target_pose[2] - source_pose[2])
                delta_xy = target_pose[:2] - source_pose[:2]
                for name, alignment in self._alignments.items():
                    out[f"{name}_hist"][batch_index, slot] = _se2_align(
                        self.obs[name][source],
                        alignment=alignment,
                        delta_theta=delta_theta,
                        delta_xy=delta_xy,
                        current_theta=float(target_pose[2]),
                    )
                    out[f"{name}_hist_mask"][batch_index, slot] = self.obs_mask[name][source]
        if not batched:
            return {key: value[0] for key, value in out.items()}
        return out

    def _sample_frames(self, index: int, frames: int, interval: int) -> list:
        """取 ``index``（含）之前同 episode 的最近 ``frames`` 个采样帧（旧 → 新）。"""
        episode_id = int(self.episode[index])
        start = index
        while start > 0 and int(self.episode[start - 1]) == episode_id:
            start -= 1
        sampled: list = []
        cursor = index
        while cursor >= start and len(sampled) < frames:
            if int(self.step_index[cursor]) % interval == 0:
                sampled.append(cursor)
            cursor -= 1
        sampled.reverse()
        return sampled

    # ------------------------------------------------------------------ GAE
    def compute_gae(
        self,
        last_value: float = 0.0,
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
        truncation_bootstrap: float = 0.0,
        normalize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE(λ) → ``(advantages, returns)``（均 float32）。

        - 终局（``terminated``）不 bootstrap；episode 边界处 GAE 链切断；
        - 截断（``truncated``）在缓冲区末尾用 ``last_value`` bootstrap；
          中间被截断的 episode 用 ``truncation_bootstrap``（默认 0，因为 reset 后的
          "下一观测"不在本缓冲内）；
        - ``normalize=True`` 时对优势做零均值/单位方差（不改 returns）。
        """
        n = self._len
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in range(n - 1, -1, -1):
            is_last = t == n - 1
            same_next = (not is_last) and int(self.episode[t + 1]) == int(self.episode[t])
            if is_last:
                next_value = float(last_value)
            elif same_next:
                next_value = float(self.value[t + 1])
            else:
                next_value = float(truncation_bootstrap)
            terminal = bool(self.terminated[t]) or bool(self.truncated[t]) or (not same_next and not is_last)
            non_terminal = 0.0 if terminal else 1.0
            delta = float(self.reward[t]) + gamma * next_value * non_terminal - float(self.value[t])
            last_gae = delta + gamma * lam * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.value[:n]
        if normalize and n > 1:
            std = float(advantages.std())
            advantages = (advantages - float(advantages.mean())) / (std + 1e-8)
        return advantages, returns.astype(np.float32)

    # ------------------------------------------------------------------ 视图
    def episode_slices(self) -> list[tuple[int, int]]:
        """按 episode 的连续区间 ``[(start, stop), ...]``（PPO 分段/统计用）。"""
        slices: list[tuple[int, int]] = []
        start = 0
        for t in range(self._len):
            if t > start and int(self.episode[t]) != int(self.episode[start]):
                slices.append((start, t))
                start = t
        if self._len > start:
            slices.append((start, self._len))
        return slices

    def to_arrays(self, *, copy: bool = False) -> dict[str, np.ndarray]:
        """导出全部存储（默认返回视图，``copy=True`` 时返回副本）。"""
        data = {
            **{f"obs.{name}": value[: self._len] for name, value in self.obs.items()},
            **{f"mask.{name}": value[: self._len] for name, value in self.obs_mask.items()},
            "pose": self.pose[: self._len],
            "action": self.action[: self._len],
            "logprob": self.logprob[: self._len],
            "value": self.value[: self._len],
            "reward": self.reward[: self._len],
            "terminated": self.terminated[: self._len],
            "truncated": self.truncated[: self._len],
            "episode": self.episode[: self._len],
            "step_index": self.step_index[: self._len],
        }
        if copy:
            return {key: value.copy() for key, value in data.items()}
        return data


def _se2_align(features: np.ndarray, *, alignment: Any, delta_theta: float, delta_xy: np.ndarray, current_theta: float):
    """``env.obs.base.se2_align`` 的惰性包装（避免模块级导入 MetaDrive 依赖链）。"""
    from env.obs.base import se2_align

    return se2_align(
        features,
        alignment=alignment,
        delta_theta=delta_theta,
        delta_xy=delta_xy,
        current_theta=current_theta,
    )
