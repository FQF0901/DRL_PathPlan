"""others 通道（schema v2 规范上下文输入，单槽）::

    others = [nav(11), speed_limit(1, 归一化), signal(4), road_class one-hot(K)]

维度拆分（dim 索引）::

    0..10  nav：2 个 checkpoint 自车系 (x,y) + command one-hot(6) + route_completion
                  （与 :class:`env.obs.nav.NavChannel` 完全同口径，直接复用其 build）
    11     speed_limit：自车当前车道限速，**归一化**到 [0,1]
    12..15 signal：4 维灯态占位（本项目无灯 → [0,0,0,1]，未知/无灯）
    16..15+K road_class one-hot(K)：自车当前 lane 所属地图 block 的几何类别

road_class（K 与几何分类一致）
------------------------------
K = ``len(ROAD_CLASS_LABELS)``，默认取 ``env/scenario/taxonomy.GEOMETRY_LABELS``（12 类），
来源与 :func:`env.scenario.labels.compute_step_labels` 的 ``in_intersection`` 相同：
``env.scenario.behaviors.map_info(env).lane_block[ego.lane_index]`` 给出 block 字符，再经
``taxonomy.BLOCK_CHARS`` 反查几何标签。固定顺序（block 字符）::

    index  0    1    2       3        4     5     6            7            8         9     10          11
    label  straight curve ramp_in ramp_out merge split intersection t_intersection roundabout uturn bidirection tollgate
    char   S    C    r       R        y     Y     X            T            O         U     B           $

未知 / 缺失（lane 无 index、map_info 不可用）→ 12 维全 0（不设 unknown 维，保持 K 不变）。

speed_limit 单位与归一化
------------------------
``lane.speed_limit`` 原始单位按项目约定是 **m/s**（见 ``env/metadrive_env`` 限速后处理；
上游 PG block 也按 m/s）。归一化分母 ``speed_limit_norm_mps``（默认 30.0 m/s ≈ 108 km/h）::

    value = clamp(speed_limit_mps, 0, norm) / norm

未设置（>= ``SPEED_LIMIT_UNSET`` = 1000，MetaDrive 的 "未设置" 默认值）、<=0 或非有限 →
0.0（"未知"，与"限速为 0"在数值上不可分，文档标注）。

mask 语义
---------
``others_mask[0] = 1`` ⇔ 自车存在（通道整体可用）；各分量有确定性回退：nav 不可用 → 前 11 维 0、
限速未知 → 0、无灯 → [0,0,0,1]、block 未知 → road_class 全 0。旧 ``nav`` / ``signal`` 通道
仍保留在观测里以兼容旧消费者，但 **``others`` 是规范输入**。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from env.obs.base import FrameAlignment, ObservationChannel, make_empty, safe_ego
from env.obs.nav import NUM_CHECKPOINTS, NUM_COMMANDS, NavChannel
from env.obs.signal import SIGNAL_DIM, SignalChannel

#: nav 子向量维数（与 NavChannel.feature_dim 一致）
NAV_DIM = 2 * NUM_CHECKPOINTS + NUM_COMMANDS + 1
#: 归一化之前的分量之和 = nav + speed_limit + signal
OTHERS_HEAD_DIM = NAV_DIM + 1 + SIGNAL_DIM

#: taxonomy 不可用时的几何标签兜底（顺序必须与 taxonomy.GEOMETRY_LABELS 一致）
_FALLBACK_ROAD_CLASS_LABELS: tuple[str, ...] = (
    "straight",
    "curve",
    "ramp_in",
    "ramp_out",
    "merge",
    "split",
    "intersection",
    "t_intersection",
    "roundabout",
    "uturn",
    "bidirection",
    "tollgate",
)
#: taxonomy 不可用时的 block 字符兜底（taxonomy.BLOCK_CHARS）
_FALLBACK_BLOCK_CHARS: dict[str, str] = {
    "straight": "S",
    "curve": "C",
    "ramp_in": "r",
    "ramp_out": "R",
    "merge": "y",
    "split": "Y",
    "intersection": "X",
    "t_intersection": "T",
    "roundabout": "O",
    "uturn": "U",
    "bidirection": "B",
    "tollgate": "$",
}
#: AbstractLane 的"未设置"默认限速（与 env.metadrive_env.SPEED_LIMIT_UNSET 同值）
SPEED_LIMIT_UNSET = 1000.0
DEFAULT_SPEED_LIMIT_NORM_MPS = 30.0


def road_class_labels() -> tuple[str, ...]:
    """几何类别顺序；优先 ``taxonomy.GEOMETRY_LABELS``，缺失时用兜底常量。"""
    try:
        from env.scenario.taxonomy import GEOMETRY_LABELS

        labels = tuple(str(item) for item in GEOMETRY_LABELS)
        if labels:
            return labels
    except Exception:  # noqa: BLE001 - taxonomy 未就绪不影响通道
        pass
    return _FALLBACK_ROAD_CLASS_LABELS


def _road_class_by_char() -> dict[str, int]:
    """block 字符 -> one-hot 下标。"""
    labels = road_class_labels()
    mapping: dict[str, str] = dict(_FALLBACK_BLOCK_CHARS)
    try:
        from env.scenario.taxonomy import BLOCK_CHARS

        for label, char in dict(BLOCK_CHARS).items():
            if isinstance(char, str) and len(char) == 1:
                mapping[str(label)] = char
    except Exception:  # noqa: BLE001
        pass
    return {mapping[label]: index for index, label in enumerate(labels) if label in mapping}


def road_class_index(env) -> Optional[int]:
    """自车当前 lane 的 block 字符 -> 几何类别下标；不可用返回 None。"""
    try:
        from env.scenario.behaviors import map_info

        info = map_info(env)
        if info is None:
            return None
        ego = safe_ego(env)
        lane = getattr(ego, "lane", None) if ego is not None else None
        if lane is None:
            return None
        block = info.lane_block.get(tuple(lane.index))
        if block is None:
            return None
        return _road_class_by_char().get(str(block))
    except Exception:  # noqa: BLE001 - 观测通道不得因地图信息异常而崩
        return None


def speed_limit_value(ego, norm_mps: float = DEFAULT_SPEED_LIMIT_NORM_MPS) -> float:
    """自车当前车道限速 -> [0,1]（单位 m/s；未知/未设置 → 0.0）。"""
    if norm_mps <= 0.0:
        return 0.0
    lane = getattr(ego, "lane", None)
    if lane is None:
        nav = getattr(ego, "navigation", None)
        ref = getattr(nav, "current_ref_lanes", None) if nav is not None else None
        lane = ref[0] if ref else None
    if lane is None:
        return 0.0
    try:
        value = float(getattr(lane, "speed_limit", 0.0))
    except Exception:  # noqa: BLE001
        return 0.0
    if not np.isfinite(value) or value <= 0.0 or value >= SPEED_LIMIT_UNSET:
        return 0.0
    return float(min(max(value, 0.0), norm_mps) / norm_mps)


class OthersChannel(ObservationChannel):
    """规范上下文通道：nav + speed_limit + signal + road_class one-hot。"""

    name = "others"
    #: dim 0..3 是 nav 的两个 checkpoint（点），其余分量不随坐标系变化
    alignment = FrameAlignment(point_pairs=((0, 1), (2, 3)))

    def __init__(self, *, speed_limit_norm_mps: float = DEFAULT_SPEED_LIMIT_NORM_MPS):
        self.speed_limit_norm_mps = float(speed_limit_norm_mps)
        self._nav = NavChannel()
        self._signal = SignalChannel()

    @property
    def feature_dim(self) -> int:  # type: ignore[override]
        return OTHERS_HEAD_DIM + len(road_class_labels())

    def reset(self) -> None:
        return None

    def build(self, env, spec=None) -> tuple[np.ndarray, np.ndarray]:
        feats, mask = make_empty(1, self.feature_dim)
        ego = safe_ego(env)
        if ego is None:
            return feats, mask

        nav_feats, _nav_mask = self._nav.build(env, spec)
        feats[0, 0:NAV_DIM] = nav_feats[0]
        feats[0, NAV_DIM] = speed_limit_value(ego, self.speed_limit_norm_mps)
        signal_feats, _signal_mask = self._signal.build(env, spec)
        feats[0, NAV_DIM + 1:OTHERS_HEAD_DIM] = signal_feats[0]

        index = road_class_index(env)
        if index is not None:
            label_count = len(road_class_labels())
            if 0 <= index < label_count:
                feats[0, OTHERS_HEAD_DIM + index] = 1.0
        mask[0] = 1.0
        return feats, mask
