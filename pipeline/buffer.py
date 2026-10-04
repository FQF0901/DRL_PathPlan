"""Rollout 缓冲（P2 契约 §4）：**按帧存储** + 历史窗口精确查表重建 + GAE(λ)。

RAM 关键设计（契约明确要求）
--------------------------
**绝不缓存 6 帧堆叠**。每帧只存当前帧通道与掩码 + 槽位级伴随数组 + 该帧 ego 位姿，历史窗口在
``build_history`` 里按需重建（与 ``env/obs/memory.FrameMemory`` 同口径：每 ``interval`` 个
env step 采一帧、按 SE(2) 对齐到目标帧、窗口锚到 stride 网格、缺帧 ``hist_valid=0``）。
v2 起窗口由 ``pipeline.frames.FrameLookup`` 做 ``(episode, step)`` **精确查表**（不再按行位置
取"最近的 interval 倍数帧"再补最旧帧），并携带 ``od_id``/``od_presence`` 伴随历史；
slot 身份由 OD 通道固定（槽位 = track id，见 ``env/obs/od.py``）。

单帧 ≈ 323 个 float32（+2×16 伴随）≈ 1.4 KB；若直接存 od/ld 历史堆叠要多出
``6×16×(9+7)=1536`` 个 float（≈6 KB/帧，约 4.8×），万级 rollout 会显著吃掉 16GB 内存预算。

接口
----
- :meth:`RolloutBuffer.add_step`：逐帧入库（env-step 粒度，0.1 s）；
- :meth:`RolloutBuffer.build_history`：给定帧下标批量重建历史窗口
  （``<ch>_hist (B,6,N,F)`` / ``<ch>_hist_mask (B,6,N)`` / ``od_id_hist`` / ``hist_valid``）；
- :meth:`RolloutBuffer.compute_gae`：GAE(λ) 优势/回报（按 episode 切断，终局不 bootstrap，
  截断处按 ``truncation_bootstrap`` 处理；``valid_mask`` 给定时 padding/bootstrap 行
  不参与优势链、仅作前一有效帧的 bootstrap 价值容器）；
- :meth:`RolloutBuffer.episode_slices`：按 episode 的连续区间（PPO 分段用）。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from pipeline.frames import FrameLookup

__all__ = ["RolloutBuffer", "DEFAULT_CHANNELS"]


def _others_dim() -> int:
    """others 通道特征维（nav 11 + speed_limit 1 + signal 4 + road_class K）。"""
    try:
        from env.obs.others import OTHERS_HEAD_DIM, road_class_labels

        return int(OTHERS_HEAD_DIM) + len(road_class_labels())
    except Exception:  # noqa: BLE001 - 极端环境（缺 taxonomy）下退回默认 12 类
        return 28


def _struct_dims() -> tuple[int, int]:
    """v5 单槽上下文通道特征维 ``(lane, ttc)``（缺 env 依赖时退回常量 17/12）。"""
    try:
        from env.obs.lane import LANE_DIM
        from env.obs.ttc import TTC_DIM

        return int(LANE_DIM), int(TTC_DIM)
    except Exception:  # noqa: BLE001 - 极端环境（缺 env 依赖）下退回 v5 常量
        return 17, 12


_LANE_DIM, _TTC_DIM = _struct_dims()

#: 与 ``env/obs/builder.ObservationBuilder`` 默认输出一致的通道形状（含槽位维）：
#: ego(1,8) / od(16,9) / ld(16,7) / lane(1,17) / nav(1,11) / signal(1,4) / others(1,28) / ttc(1,12)。
#: lane/ttc 为 v5 当前帧上下文通道（不进 6 帧历史；obs 缺键时 :meth:`RolloutBuffer.add_step`
#: 填 0，net 侧按缺键回退语义处理）。
DEFAULT_CHANNELS: dict[str, tuple[int, ...]] = {
    "ego": (1, 8),
    "od": (16, 9),
    "ld": (16, 7),
    "lane": (1, _LANE_DIM),
    "nav": (1, 11),
    "signal": (1, 4),
    "others": (1, _others_dim()),
    "ttc": (1, _TTC_DIM),
}
#: 默认重建历史的通道（与 ``config/env.yaml`` / FrameMemory 一致）
DEFAULT_HISTORY_CHANNELS: tuple[str, ...] = ("ego", "others", "od", "ld")
#: OD 槽位级伴随数组（dtype, 缺省填充值）：身份 / 本帧观测标志
DEFAULT_COMPANIONS: dict[str, tuple[Any, Any]] = {
    "od_id": (np.int64, -1),
    "od_presence": (np.float32, 0.0),
}


def _frame_pose(env: Any) -> np.ndarray:
    """从 env 读当前 ego 位姿 ``(x, y, theta)``；不可用返回零（历史对齐退化为不平移）。"""
    try:
        ego = env.agent
        position = np.asarray(ego.position, dtype=np.float32)
        return np.array([position[0], position[1], float(ego.heading_theta)], dtype=np.float32)
    except Exception:  # noqa: BLE001 - 缓冲不应因观测失败而中断
        return np.zeros(3, dtype=np.float32)


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
            history_channels: 需要重建历史的通道（默认 ego/others/od/ld，与 builder 一致）。
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
        #: 槽位级伴随数组（OD 身份/存在性）；仅在 channels 含 od 时启用
        od_slots = self.channels.get("od", (0, ))[0]
        self.companions: dict[str, np.ndarray] = {}
        if od_slots:
            for name, (dtype, fill) in DEFAULT_COMPANIONS.items():
                self.companions[name] = np.full((self.capacity, int(od_slots)), fill, dtype=dtype)
        self.pose = np.zeros((self.capacity, 3), dtype=np.float32)
        self.action = np.zeros((self.capacity, 2), dtype=np.float32)
        self.logprob = np.zeros(self.capacity, dtype=np.float32)
        self.value = np.zeros(self.capacity, dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=bool)
        self.truncated = np.zeros(self.capacity, dtype=bool)
        self.episode = np.zeros(self.capacity, dtype=np.int32)
        self.step_index = np.zeros(self.capacity, dtype=np.int32)
        #: 帧级场景/环境标识（spec.id；未知 = -1）。per_scenario 优势归一化的分组键
        #: （见 ``PPOConfig.adv_norm``）；默认全 -1 → 该口径退化为单组（= global）。
        self.spec_id = np.full(self.capacity, -1, dtype=np.int32)
        self._len = 0
        self._lookup: Optional[FrameLookup] = None
        self._lookup_len = -1

    # ------------------------------------------------------------------ 基本信息
    def __len__(self) -> int:
        return self._len

    @property
    def full(self) -> bool:
        return self._len >= self.capacity

    def _invalidate_lookup(self) -> None:
        self._lookup = None
        self._lookup_len = -1

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
        spec_id: Optional[int] = None,
    ) -> int:
        """写入一帧（当前帧通道 + 掩码 + 伴随数组 + 位姿 + 动作/回报等元信息），返回帧下标。

        ``obs`` 只取**当前帧**通道（``ego/od/ld/lane/nav/signal/others/ttc`` 与 ``*_mask``）与伴随数组
        （``od_id``/``od_presence``），``*_hist`` 之类的键被忽略。``episode``/``step`` 缺省时按
        "上一帧终局则新 episode、步号自增"自动维护。``spec_id`` = 该帧所属场景（spec.id，
        未知/旧调用方缺省 -1），供 ``per_scenario`` 优势归一化分组。
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
        for name in self.companions:
            fill = DEFAULT_COMPANIONS[name][1]
            value_array = obs.get(name)
            if value_array is None:
                self.companions[name][index] = fill
            else:
                arr = np.asarray(value_array)
                if arr.shape != self.companions[name].shape[1:]:
                    raise ValueError(
                        f"伴随数组 {name} 期望 shape={self.companions[name].shape[1:]}，收到 {arr.shape}"
                    )
                self.companions[name][index] = arr

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
        self.spec_id[index] = -1 if spec_id is None else int(spec_id)
        self._len += 1
        self._invalidate_lookup()
        return index

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
    def _frame_lookup(self) -> FrameLookup:
        """按需构建精确查表的索引（数据网格 = 逐 env step → stride=1）。"""
        if self._lookup is not None and self._lookup_len == self._len:
            return self._lookup
        features = {name: self.obs[name][: self._len] for name in self.history_channels if name in self.obs}
        masks = {name: self.obs_mask[name][: self._len] for name in features}
        companions = {key: value[: self._len] for key, value in self.companions.items()} if "od" in features else {}
        self._lookup = FrameLookup(
            self.episode[: self._len],
            self.step_index[: self._len],
            pose=self.pose[: self._len],
            features=features,
            masks=masks,
            companions=companions,
            stride=1,
            check=True,
        )
        self._lookup_len = self._len
        return self._lookup

    def build_history(
        self,
        indices: Any,
        *,
        frames: Optional[int] = None,
        interval: Optional[int] = None,
        current_pose: Optional[Sequence[float]] = None,
    ) -> dict[str, np.ndarray]:
        """在线重建指定帧的历史窗口（``pipeline.frames`` 精确查表，与 FrameMemory 同口径）。

        窗口 = 以 ``floor(step/interval)*interval`` 为锚的 ``frames`` 个 stride 网格帧（旧→新）；
        ``hist_valid``/``<ch>_hist_mask`` 对缺帧为 0（buffer 逐 step 记录，缺帧只可能出现在
        episode 头部）。

        Args:
            indices: 帧下标（int 或序列）。
            frames / interval: 覆盖默认窗口/采样间隔。
            current_pose: 对齐目标位姿；``None`` = 用各帧自身位姿（即"以该帧为当前帧"）。

        Returns:
            ``{"<ch>_hist": (B,F,N,Dim), "<ch>_hist_mask": (B,F,N), "od_id_hist": (B,F,N),
            "hist_valid": (B,F)}``；标量 ``indices`` 时去掉 batch 维。
        """
        frames = self.history_frames if frames is None else int(frames)
        interval = self.history_interval if interval is None else int(interval)
        if frames < 1 or interval < 1:
            raise ValueError("frames / interval 必须 >= 1")
        batched = np.ndim(indices) > 0
        index_array = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        lookup = self._frame_lookup()
        out: Optional[dict[str, np.ndarray]] = None
        for position, frame_index in enumerate(index_array):
            if not 0 <= frame_index < self._len:
                raise IndexError(f"帧下标越界：{frame_index}（当前 {self._len} 帧）")
            target_pose = (
                np.asarray(current_pose, dtype=np.float32).reshape(3) if current_pose is not None else None
            )
            window = lookup.build_history(
                int(self.episode[frame_index]),
                int(self.step_index[frame_index]),
                stride=interval,
                k=frames,
                target_pose=target_pose,
            )
            if out is None:
                out = {key: np.zeros((len(index_array), ) + value.shape, dtype=value.dtype) for key, value in window.items()}
            for key, value in window.items():
                out[key][position] = value
        if out is None:  # 空输入：返回空 batch（保持键集合与正常路径一致）
            out = {
                f"{name}_hist": np.zeros((0, frames) + self.channels[name], dtype=np.float32)
                for name in self.history_channels
                if name in self.channels
            }
            out.update(
                {
                    f"{name}_hist_mask": np.zeros((0, frames) + self.channels[name][:-1], dtype=np.float32)
                    for name in self.history_channels
                    if name in self.channels
                }
            )
            out["hist_valid"] = np.zeros((0, frames), dtype=np.float32)
        if not batched:
            return {key: value[0] for key, value in out.items()}
        return out

    # ------------------------------------------------------------------ GAE
    def compute_gae(
        self,
        last_value: float = 0.0,
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
        truncation_bootstrap: float = 0.0,
        normalize: bool = False,
        valid_mask: Optional[Any] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE(λ) → ``(advantages, returns)``（均 float32）。

        - 终局（``terminated``）不 bootstrap；episode 边界处 GAE 链切断；
        - 截断（``truncated``）在缓冲区末尾用 ``last_value`` bootstrap；
          中间被截断的 episode 用 ``truncation_bootstrap``（默认 0，因为 reset 后的
          "下一观测"不在本缓冲内）；
        - ``normalize=True`` 时对优势做零均值/单位方差（不改 returns）。

        ``valid_mask``（trainer 口径，2026-09-30 G3 修复）：长度 >= ``len(self)`` 的布尔数组，
        ``False`` 行 = **padding/bootstrap 行**（env-major 布局里每 env 段末尾的
        "下一观测价值容器"；见 ``PPOTrainer.collect_rollout``）。给定时的语义：

        * padding 行不参与优势链——其优势记 0，且不把自身（伪）优势沿 ``γλ`` 传播给前一帧
          （旧实现把 padding 行当成真实一步，末段有效帧的优势会带 ``−γλ·V_bootstrap`` 的
          系统性偏差，且随 ``(γλ)^k`` 衰减污染整个尾部）；
        * padding 行仍作为**其前一有效帧的 bootstrap 价值容器**（``V(s_H)`` 供末帧
          ``δ = r + γV(s_H) − V(s_t)`` 使用）。

        ``None`` = 旧行为（所有行都参与链；纯 buffer 单测/无 padding 布局逐位不变）。
        """
        n = self._len
        advantages = np.zeros(n, dtype=np.float32)
        if valid_mask is None:
            valid = None
        else:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape[0] < n:
                raise ValueError(f"valid_mask 长度 {mask.shape[0]} < buffer 长度 {n}")
            valid = mask[:n]
        last_gae = 0.0
        for t in range(n - 1, -1, -1):
            if valid is not None and not valid[t]:
                # padding/bootstrap 行：仅价值容器，不参与优势链
                advantages[t] = 0.0
                last_gae = 0.0
                continue
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
            pool = advantages if valid is None else advantages[valid]
            if pool.size > 1:
                normalized = (pool - pool.mean()) / (pool.std() + 1e-8)
                if valid is None:
                    advantages = normalized
                else:
                    advantages = advantages.copy()
                    advantages[valid] = normalized
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
            **{f"companion.{name}": value[: self._len] for name, value in self.companions.items()},
            "pose": self.pose[: self._len],
            "action": self.action[: self._len],
            "logprob": self.logprob[: self._len],
            "value": self.value[: self._len],
            "reward": self.reward[: self._len],
            "terminated": self.terminated[: self._len],
            "truncated": self.truncated[: self._len],
            "episode": self.episode[: self._len],
            "step_index": self.step_index[: self._len],
            "spec_id": self.spec_id[: self._len],
        }
        if copy:
            return {key: value.copy() for key, value in data.items()}
        return data
