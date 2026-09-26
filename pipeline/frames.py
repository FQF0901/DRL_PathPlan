"""精确查表的历史 / 未来窗口（obs schema v2 的单一真源）。

问题背景（v1 缺陷）
------------------
历史窗口按**行位置**重建（取 ``index-5 .. index`` 的行），BC 数据集过滤掉行之后：
16.8% 的窗口时间不均匀（相邻帧间隔可能到 3 s），episode 头部复制最旧行且
``hist_valid`` 与真实帧对不上。本模块所有窗口都按 ``(episode_id, step)`` **精确查表**：

- 历史：``build_history(episode_id, step, stride=5, k=6)`` 取
  ``base - (k-1-j)*stride``（``base = floor(step/stride)*stride``，旧 → 新）的 k 帧；
  缺帧（过滤洞 / episode 之前 / 超出记录范围）→ 该 slot ``hist_valid=0``、特征全零、mask=0；
  存在的帧按 SE(2) 对齐到目标帧（``base`` 帧；buffer 场景可显式给 ``target_pose``）。
- 未来：``build_future(episode_id, step, stride=5, k=6)`` 取 ``base + (k+1)*stride``
  （k=1..K）的未来帧，对齐到 t0（``base``）帧；``valid`` = 目标帧存在，
  ``wm_valid`` = 目标帧存在 **且** 仍可用（``usable`` 数组，见下）。

为什么锚到 stride 网格而不是严格 ``step - j*stride``：PPO replay 逐 env step 记录，
采样历史必须落在"真实入库过的采样帧"（每 stride 个 env step 一帧）上；对 BC 数据集
（帧只记录在策略步边界，step 均为 stride 倍数）两者等价。无论目标 step 是否在网格上，
窗口都是均匀的 stride 网格，从根上消除"位置语义"的时间洞。

恒等与断言
----------
构造时断言 ``(episode_id, step)`` 唯一；``stride`` 断言每个 episode 内相邻帧 step 差
是 stride 的正整数倍（允许过滤洞，不允许无序/错格）。bc 数据集用 stride=5；
逐 env-step 的 rollout buffer 用 stride=1。

身份（slot）
------------
OD 槽位 = track id（``env/obs/od.py``），因此查表**不做任何槽位重排**：``od_hist`` 的第 s
槽在每帧都是同一个 track（``od_id_hist`` 标注身份；缺帧时 id 填 ``-1``）。未来目标的
``od_mask`` 同时要求"同 id 且在目标帧被观测到"（``presence``），所以 Stage A 不需要
v1 的最近邻槽位匹配。

接口
----
- :class:`FrameLookup`：``build_history`` / ``build_future``（签名与契约一致）；
- :func:`build_history` / :func:`build_future`：模块级薄封装（第一个参数是 lookup）；
- :func:`lookup_from_arrays`：从 BC / replay 数组字典直接构造 lookup（trainer 消费）。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "DEFAULT_K",
    "DEFAULT_STRIDE",
    "HISTORY_CHANNELS",
    "FrameLookup",
    "build_future",
    "build_history",
    "lookup_from_arrays",
]

DEFAULT_STRIDE = 5
DEFAULT_K = 6
#: v2 需要重建历史的通道（与 env.obs.builder / FrameMemory 一致）
HISTORY_CHANNELS: tuple[str, ...] = ("ego", "others", "od", "ld")
#: 默认伴随数组（进入历史的槽位级身份/存在性）
DEFAULT_COMPANION_KEYS: tuple[str, ...] = ("od_id", "od_presence")


def _fill_value(key: str, dtype) -> Any:
    """伴随数组缺帧填充值（整型 -1 / 浮点 0），与通道 ``companion_fill`` 口径一致。"""
    if key == "od_id":
        return -1
    return -1 if np.issubdtype(np.dtype(dtype), np.integer) else 0.0


def _default_alignments(names: Sequence[str]) -> dict[str, Any]:
    """按通道名取 SE(2) 对齐规则；导入失败退化为"不对齐"（frames 不强制依赖 env 实现）。"""
    from env.obs.base import FrameAlignment

    known: dict[str, Any] = {}
    try:
        from env.obs.ld import LDChannel
        from env.obs.nav import NavChannel
        from env.obs.od import ODChannel
        from env.obs.others import OthersChannel

        known = {
            "od": ODChannel.alignment,
            "ld": LDChannel.alignment,
            "nav": NavChannel.alignment,
            "others": OthersChannel.alignment,
            "ego": FrameAlignment(),
            "signal": FrameAlignment(),
        }
    except Exception:  # noqa: BLE001 - 纯 NumPy 环境（测试）下不导入 metadrive 依赖链
        pass
    return {name: known.get(name, FrameAlignment()) for name in names}


class FrameLookup:
    """按 ``(episode_id, step)`` 精确查表构建历史 / 未来窗口。"""

    def __init__(
        self,
        episode_id: Any,
        step: Any,
        *,
        pose: Optional[Any] = None,
        features: Optional[Mapping[str, np.ndarray]] = None,
        masks: Optional[Mapping[str, np.ndarray]] = None,
        companions: Optional[Mapping[str, np.ndarray]] = None,
        usable: Optional[Any] = None,
        stride: int = DEFAULT_STRIDE,
        check: bool = True,
    ):
        """
        Args:
            episode_id / step: 等长一维数组（每行一帧）。
            pose: ``(N,3)`` 每帧 ego 位姿 ``(x,y,θ)`` 世界系；查表对齐用。
            features: ``{通道名: (N,...)}`` 每帧特征。
            masks: ``{通道名: (N,...)}`` 每帧槽位掩码（形状 = features 去掉最后一维）。
            companions: ``{键: (N,...)}`` 槽位级伴随数组（如 ``od_id``/``od_presence``）。
            usable: ``(N,)`` 每帧是否可用于未来目标（``wm_valid`` 用）；None = 存在即可用。
            stride: 数据网格（断言相邻帧 step 差是它的正整数倍）。
            check: False 时跳过唯一性/网格断言（仅用于性能敏感且已知合法的场景）。
        """
        self.episode = np.asarray(episode_id, dtype=np.int64).reshape(-1)
        self.step = np.asarray(step, dtype=np.int64).reshape(-1)
        if self.episode.shape != self.step.shape:
            raise ValueError(f"episode_id/step 长度不一致：{self.episode.shape} vs {self.step.shape}")
        self.stride = max(1, int(stride))
        self.n = int(self.episode.shape[0])
        self.pose = None if pose is None else np.asarray(pose, dtype=np.float32).reshape(-1, 3)
        if self.pose is not None and self.pose.shape[0] != self.n:
            raise ValueError(f"pose 行数 {self.pose.shape[0]} != 帧数 {self.n}")
        self.features = {str(k): np.asarray(v) for k, v in dict(features or {}).items()}
        self.masks = {str(k): np.asarray(v) for k, v in dict(masks or {}).items()}
        self.companions = {str(k): np.asarray(v) for k, v in dict(companions or {}).items()}
        for key, value in {**self.features, **self.masks, **self.companions}.items():
            if value.shape[0] != self.n:
                raise ValueError(f"数组 {key!r} 行数 {value.shape[0]} != 帧数 {self.n}")
        self.usable = None if usable is None else np.asarray(usable).reshape(-1)
        if self.usable is not None and self.usable.shape[0] != self.n:
            raise ValueError(f"usable 行数 {self.usable.shape[0]} != 帧数 {self.n}")

        self._index: dict[tuple[int, int], int] = {}
        for row in range(self.n):
            key = (int(self.episode[row]), int(self.step[row]))
            if key in self._index:
                raise ValueError(
                    f"(episode_id, step) 不唯一：{key} 同时出现在行 {self._index[key]} 与 {row}"
                )
            self._index[key] = row
        if check:
            self._check_stride()

    # ------------------------------------------------------------------ 断言
    def _check_stride(self) -> None:
        """每个 episode 内 step 严格递增，且相邻差是 ``stride`` 的正整数倍（允许过滤洞）。"""
        for episode in np.unique(self.episode):
            rows = np.where(self.episode == episode)[0]
            values = np.sort(self.step[rows])
            if values.size < 2:
                continue
            diffs = np.diff(values)
            if np.any(diffs <= 0):
                raise ValueError(f"episode {int(episode)} 的 step 非严格递增：{values.tolist()[:12]}")
            bad = diffs[diffs % self.stride != 0]
            if bad.size:
                raise ValueError(
                    f"episode {int(episode)} 的 step 间隔 {bad.tolist()} 不是 stride={self.stride} 的正整数倍"
                    f"（step={values.tolist()[:12]}）"
                )

    # ------------------------------------------------------------------ 查询
    def index_of(self, episode_id: int, step: int) -> Optional[int]:
        """返回 ``(episode_id, step)`` 的行下标；不存在返回 None。"""
        return self._index.get((int(episode_id), int(step)))

    def _anchor(self, step: int, stride: int) -> int:
        """目标 step 锚到的 stride 网格点（floor；step 为负数时 Python 向下取整语义一致）。"""
        return (int(step) // int(stride)) * int(stride)

    def _target_pose(self, episode_id: int, step: int, stride: int, explicit: Optional[Any]) -> np.ndarray:
        """对齐目标位姿：显式 > 请求帧自身（离网时也以"当前帧"为准）> 网格锚点帧。"""
        if explicit is not None:
            return np.asarray(explicit, dtype=np.float32).reshape(3)
        for key in (int(step), self._anchor(step, stride)):
            row = self._index.get((int(episode_id), int(key)))
            if row is not None and self.pose is not None:
                return self.pose[row]
        return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _resolve(stride: Optional[int], k: Optional[int], default_stride: int) -> tuple[int, int]:
        stride = default_stride if stride is None else int(stride)
        k = DEFAULT_K if k is None else int(k)
        if stride < 1 or k < 1:
            raise ValueError(f"stride/k 必须 >= 1，收到 stride={stride}, k={k}")
        return stride, k

    def _align(self, features: np.ndarray, *, alignment: Any, target_pose: np.ndarray,
               source_pose: np.ndarray) -> np.ndarray:
        from env.obs.base import se2_align

        return se2_align(
            features,
            alignment=alignment,
            delta_theta=float(target_pose[2] - source_pose[2]),
            delta_xy=target_pose[:2] - source_pose[:2],
            current_theta=float(target_pose[2]),
        )

    def _alignment_map(self, overrides: Optional[Mapping[str, Any]], names: Sequence[str]) -> dict[str, Any]:
        resolved = dict(_default_alignments(tuple(names)))
        for name, alignment in dict(overrides or {}).items():
            if alignment is not None:
                resolved[str(name)] = alignment
        return resolved

    # ------------------------------------------------------------------ 历史
    def build_history(
        self,
        episode_id: int,
        step: int,
        stride: int = DEFAULT_STRIDE,
        k: int = DEFAULT_K,
        *,
        target_pose: Optional[Any] = None,
        alignments: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        """k 帧历史窗口（旧 → 新），精确查表 + per-frame valid。

        Returns（``k`` 维在前，单帧形状与 features 一致）::

            <ch>_hist (k,...) / <ch>_hist_mask (k,...)   # 存在的帧才填充，缺帧全零
            <key>_hist (k,...)                            # 伴随数组（缺帧填 -1 / 0）
            hist_valid (k,) float32                       # 该 slot 的真实帧是否存在
            frame_index_hist (k,) int64                   # 行下标（-1 = 缺帧）
            pose_hist (k,3) float32                       # 各帧原始位姿（对齐前）
        """
        stride, k = self._resolve(stride, k, self.stride)
        episode_id, step = int(episode_id), int(step)
        base = self._anchor(step, stride)
        target = self._target_pose(episode_id, step, stride, target_pose)
        alignments = self._alignment_map(alignments, tuple(self.features))
        keys = [(episode_id, base - (k - 1 - slot) * stride) for slot in range(k)]

        out: dict[str, np.ndarray] = {
            f"{name}_hist": np.zeros((k, ) + value.shape[1:], dtype=value.dtype)
            for name, value in self.features.items()
        }
        out.update(
            {
                f"{name}_hist_mask": np.zeros((k, ) + value.shape[1:], dtype=np.float32)
                for name, value in self.masks.items()
            }
        )
        for key, value in self.companions.items():
            out[f"{key}_hist"] = np.full((k, ) + value.shape[1:], _fill_value(key, value.dtype), dtype=value.dtype)
        out["hist_valid"] = np.zeros(k, dtype=np.float32)
        out["frame_index_hist"] = np.full(k, -1, dtype=np.int64)
        out["pose_hist"] = np.zeros((k, 3), dtype=np.float32)

        for slot, key in enumerate(keys):
            row = self._index.get(key)
            if row is None:
                continue
            out["hist_valid"][slot] = 1.0
            out["frame_index_hist"][slot] = row
            source_pose = self.pose[row] if self.pose is not None else np.zeros(3, dtype=np.float32)
            out["pose_hist"][slot] = source_pose
            for name, value in self.features.items():
                frame_value = value[row]
                if np.ndim(frame_value) != 2:  # 单槽/无槽维特征（如 (8,)）不做 SE(2) 变换，原样携带
                    out[f"{name}_hist"][slot] = frame_value
                else:
                    out[f"{name}_hist"][slot] = self._align(
                        frame_value, alignment=alignments[name], target_pose=target, source_pose=source_pose
                    )
            for name, value in self.masks.items():
                out[f"{name}_hist_mask"][slot] = value[row]
            for key_name, value in self.companions.items():
                out[f"{key_name}_hist"][slot] = value[row]
        return out

    # ------------------------------------------------------------------ 未来
    def build_future(
        self,
        episode_id: int,
        step: int,
        stride: int = DEFAULT_STRIDE,
        k: int = DEFAULT_K,
        *,
        target_pose: Optional[Any] = None,
        alignments: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, np.ndarray]:
        """k 步未来目标（``base + (k+1)*stride``），对齐到 t0 帧。

        Returns::

            od_fut (k,16,9) / ld_fut (k,16,7)         # 目标帧特征（对齐到 t0 自车系）
            od_mask (k,16)                            # wm_valid × 目标帧 presence × 与 t0 同 id
            od_mask_raw (k,16)                        # 目标帧自身 mask × valid（不做 id 匹配）
            ld_mask (k,16)                            # 目标帧 ld mask × wm_valid
            od_id_fut (k,16) int64                    # 目标槽位 id（缺帧 -1）
            od_presence_fut (k,16)                    # 目标帧观测标志
            od_id_t0 (16,) int64 / od_presence_t0 (16,)
            valid (k,)                                # 目标帧存在（查表命中）
            wm_valid (k,)                             # valid × usable（无 usable 时 = valid）
        """
        stride, k = self._resolve(stride, k, self.stride)
        episode_id, step = int(episode_id), int(step)
        base = self._anchor(step, stride)
        od = self.features.get("od")
        ld = self.features.get("ld")
        if od is None or ld is None:
            raise KeyError(f"build_future 需要 features['od']/features['ld']，现有 {sorted(self.features)}")
        alignments = self._alignment_map(alignments, tuple(self.features))
        target = self._target_pose(episode_id, step, stride, target_pose)

        t0_row = self._index.get((episode_id, base))
        if t0_row is None:
            t0_row = self._index.get((episode_id, step))
        od_id_t0 = (
            np.asarray(self.companions["od_id"][t0_row], dtype=np.int64)
            if t0_row is not None and "od_id" in self.companions
            else np.full(od.shape[1], -1, dtype=np.int64)
        )
        od_presence_t0 = (
            np.asarray(self.companions["od_presence"][t0_row], dtype=np.float32)
            if t0_row is not None and "od_presence" in self.companions
            else np.ones(od.shape[1], dtype=np.float32)
        )

        slots = od.shape[1]
        od_fut = np.zeros((k, ) + od.shape[1:], dtype=od.dtype)
        ld_fut = np.zeros((k, ) + ld.shape[1:], dtype=ld.dtype)
        od_mask = np.zeros((k, slots), dtype=np.float32)
        od_mask_raw = np.zeros((k, slots), dtype=np.float32)
        ld_mask = np.zeros((k, slots), dtype=np.float32)
        od_id_fut = np.full((k, slots), -1, dtype=np.int64)
        od_presence_fut = np.zeros((k, slots), dtype=np.float32)
        valid = np.zeros(k, dtype=np.float32)
        wm_valid = np.zeros(k, dtype=np.float32)

        for index in range(k):
            row = self._index.get((episode_id, base + (index + 1) * stride))
            if row is None:
                continue
            valid[index] = 1.0
            if self.usable is not None:
                wm_valid[index] = 1.0 if bool(self.usable[row]) else 0.0
            else:
                wm_valid[index] = 1.0
            source_pose = self.pose[row] if self.pose is not None else np.zeros(3, dtype=np.float32)
            od_fut[index] = self._align(
                od[row], alignment=alignments["od"], target_pose=target, source_pose=source_pose
            )
            ld_fut[index] = self._align(
                ld[row], alignment=alignments["ld"], target_pose=target, source_pose=source_pose
            )
            frame_od_mask = (
                np.asarray(self.masks["od"][row], dtype=np.float32) if "od" in self.masks else np.zeros(slots, np.float32)
            )
            frame_ld_mask = (
                np.asarray(self.masks["ld"][row], dtype=np.float32) if "ld" in self.masks else np.zeros(slots, np.float32)
            )
            if "od_id" in self.companions:
                od_id_fut[index] = np.asarray(self.companions["od_id"][row], dtype=np.int64)
            if "od_presence" in self.companions:
                od_presence_fut[index] = np.asarray(self.companions["od_presence"][row], dtype=np.float32)
            else:
                od_presence_fut[index] = frame_od_mask
            same_identity = (od_id_fut[index] == od_id_t0) & (od_id_t0 >= 0)
            od_mask[index] = wm_valid[index] * od_presence_fut[index] * same_identity.astype(np.float32)
            od_mask_raw[index] = frame_od_mask * valid[index]
            ld_mask[index] = frame_ld_mask * wm_valid[index]

        return {
            "od_fut": od_fut,
            "ld_fut": ld_fut,
            "od_mask": od_mask,
            "od_mask_raw": od_mask_raw,
            "ld_mask": ld_mask,
            "od_id_fut": od_id_fut,
            "od_presence_fut": od_presence_fut,
            "od_id_t0": od_id_t0,
            "od_presence_t0": od_presence_t0,
            "valid": valid,
            "wm_valid": wm_valid,
        }


# --------------------------------------------------------------------------- #
# 模块级薄封装 + 数组字典构造
# --------------------------------------------------------------------------- #

def build_history(
    lookup: FrameLookup, episode_id: int, step: int, stride: int = DEFAULT_STRIDE, k: int = DEFAULT_K, **kwargs
) -> dict[str, np.ndarray]:
    """``FrameLookup.build_history`` 的函数式入口。"""
    return lookup.build_history(episode_id, step, stride=stride, k=k, **kwargs)


def build_future(
    lookup: FrameLookup, episode_id: int, step: int, stride: int = DEFAULT_STRIDE, k: int = DEFAULT_K, **kwargs
) -> dict[str, np.ndarray]:
    """``FrameLookup.build_future`` 的函数式入口。"""
    return lookup.build_future(episode_id, step, stride=stride, k=k, **kwargs)


def lookup_from_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    episode_key: str = "episode_id",
    step_key: str = "step",
    pose_key: str = "pose",
    channels: Sequence[str] = HISTORY_CHANNELS,
    companion_keys: Sequence[str] = DEFAULT_COMPANION_KEYS,
    usable_key: Optional[str] = None,
    stride: int = DEFAULT_STRIDE,
    check: bool = True,
) -> FrameLookup:
    """从 BC / replay 数组字典构造 :class:`FrameLookup`（trainer 消费入口）。

    - 特征键 = ``channels``（缺的跳过）；掩码键 = ``f"{name}_mask"``；
    - 伴随键 = ``companion_keys``（缺的跳过）；
    - ``usable_key`` 显式传入每帧可用性数组（如采集侧 ``frame_usable``）→ 未来 ``wm_valid``；
      传 None 时自动探测 ``frame_usable`` / ``usable``。
    """
    features = {name: arrays[name] for name in channels if name in arrays}
    masks = {name: arrays[f"{name}_mask"] for name in features if f"{name}_mask" in arrays}
    companions = {key: arrays[key] for key in companion_keys if key in arrays}
    resolved_usable_key = usable_key
    if resolved_usable_key is None:
        for candidate in ("frame_usable", "usable"):
            if candidate in arrays:
                resolved_usable_key = candidate
                break
    usable = arrays.get(resolved_usable_key) if resolved_usable_key else None
    return FrameLookup(
        arrays[episode_key],
        arrays[step_key],
        pose=arrays.get(pose_key),
        features=features,
        masks=masks,
        companions=companions,
        usable=usable,
        stride=stride,
        check=check,
    )
