"""内置奖励项（N2 契约 §3）。

类别与项
--------
- 安全（终止型）：``crash``（任意 ``crash*`` 标志）/ ``out_of_road``。
- 舒适（死区二次软惩罚）：``comfort_lon`` / ``comfort_lat`` / ``comfort_jerk``。
- 车道保持（死区线性惩罚，**默认关**）：``lane_center``（偏离本车道中心线 ``|d_lat|``；无车道信息时为 0）。
- 效率：``speed_ratio``（``v / 车道限速`` 的截断收益，防爬行；超速交由合规项）。
- 合规：``solid_line``（连续实线跨越）/ ``speed_limit``（超速量）。
- 达成：``route_completion``（势能塑形 ``γΦ(s') − Φ(s)``，策略序保持）。

所有项只读 ``step_ctx`` 字典（键名与 MetaDrive ``info`` 对齐，见 ``env/metadrive_env.py``
与 ``env/obs/ld.py``），不 import env/metadrive。
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping

from reward_model.registry import TERMINATING, Term, register_term

__all__ = [
    "CRASH_FLAGS",
    "SOLID_LINE_TYPE_IDS",
    "CrashPenalty",
    "OutOfRoadPenalty",
    "LongitudinalAccelPenalty",
    "LateralAccelPenalty",
    "JerkPenalty",
    "LaneCenterPenalty",
    "SpeedRatioTerm",
    "SolidLineCrossingPenalty",
    "SpeedLimitViolationPenalty",
    "RouteCompletionShaping",
    "DEFAULT_TERM_CONFIGS",
    "lane_lateral_offset_from_ctx",
]

#: 任意一个为真即视为碰撞（MetaDrive ``info`` / P1a 契约 §0）
CRASH_FLAGS: tuple[str, ...] = (
    "crash",
    "crash_vehicle",
    "crash_object",
    "crash_building",
    "crash_sidewalk",
)

#: 连续实线类线型 id（来源 ``env/obs/ld.py::LINE_TYPE_IDS``：2/3 白实线，6/7/8 黄实线；
#: 1/4/5 为虚线）。只在本模块内使用，不 import env。
SOLID_LINE_TYPE_IDS: frozenset[int] = frozenset({2, 3, 6, 7, 8})


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hinge(value: float, deadband: float) -> float:
    """死区外二次惩罚：``max(0, |value| − deadband)²``（软惩罚，无硬截断）。"""
    excess = abs(float(value)) - float(deadband)
    return excess * excess if excess > 0.0 else 0.0


def speed_ratio_from_ctx(step_ctx: Mapping[str, Any]) -> float | None:
    """优先读 ``speed_ratio``；缺失时由 ``speed / speed_limit_mps`` 折算（m/s 口径）。"""
    ratio = _optional_float(step_ctx.get("speed_ratio"))
    if ratio is not None:
        return ratio
    speed = _optional_float(step_ctx.get("speed"))
    if speed is None:
        speed = _optional_float(step_ctx.get("velocity"))
    limit = _optional_float(step_ctx.get("speed_limit_mps"))
    if limit is None:
        limit = _optional_float(step_ctx.get("speed_limit"))
    if speed is None or limit is None or limit <= 0.0:
        return None
    return speed / limit


def lane_lateral_offset_from_ctx(step_ctx: Mapping[str, Any]) -> float | None:
    """读取横向偏差 ``d_lat``（m，自车相对当前车道中心线；本项只用 ``|d_lat|``）。

    读取顺序（只读 ``step_ctx``，不 import env/metadrive）：

    1. 显式键 ``d_lat`` / ``lane_lateral_offset`` / ``lateral_offset`` / ``lane_offset``
       （MetaDrive 车道系横向坐标 ``lane.local_coordinates(position)[1]`` 口径）；
    2. 回退：由最近的左右车道边界距离推导
       ``d_lat = (dist_to_left_side − dist_to_right_side) / 2``
       （MetaDrive ``BaseVehicle.dist_to_left_side / dist_to_right_side`` 同名口径：到本车道
       左 / 右边界的最近距离（m，>= 0）；两者之和 = 本车道宽度，故差值的一半即相对中心线偏移，
       与 ``local_coordinates`` 横向分量同号）；
    3. 键缺失 / 非法（None、非数值）→ ``None``（调用方按无车道信息处理）。
    """
    for key in ("d_lat", "lane_lateral_offset", "lateral_offset", "lane_offset"):
        value = _optional_float(step_ctx.get(key))
        if value is not None:
            return value
    left = _optional_float(step_ctx.get("dist_to_left_side"))
    right = _optional_float(step_ctx.get("dist_to_right_side"))
    if left is None or right is None:
        return None
    return 0.5 * (left - right)


@register_term
class CrashPenalty(Term):
    """安全：任意 ``crash*`` 标志（或 ``collision``）触发终止型惩罚。

    ``compute`` 返回 0/1；权重建议为负（默认聚合配置 -10）。
    """

    name: ClassVar[str] = "crash"
    kind: ClassVar[str] = TERMINATING
    reason: ClassVar[str] = "collision"

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        if _truthy(step_ctx.get("collision")):
            return 1.0
        return 1.0 if any(_truthy(step_ctx.get(flag)) for flag in CRASH_FLAGS) else 0.0


@register_term
class OutOfRoadPenalty(Term):
    """安全：``out_of_road`` 触发终止型惩罚（``compute`` 返回 0/1）。"""

    name: ClassVar[str] = "out_of_road"
    kind: ClassVar[str] = TERMINATING
    reason: ClassVar[str] = "out_of_road"

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        return 1.0 if _truthy(step_ctx.get("out_of_road")) else 0.0


@register_term
class LongitudinalAccelPenalty(Term):
    """舒适：纵向加速度 ``|a_lon|``（m/s²）超过死区后的二次软惩罚。"""

    name: ClassVar[str] = "comfort_lon"

    def __init__(self, weight: float = -0.05, deadband: float = 2.5) -> None:
        super().__init__(weight=weight, deadband=float(deadband))
        self.deadband = float(deadband)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        value = _optional_float(step_ctx.get("a_lon"))
        if value is None:
            value = _optional_float(step_ctx.get("accel_lon"))
        return 0.0 if value is None else _hinge(value, self.deadband)


@register_term
class LateralAccelPenalty(Term):
    """舒适：横向加速度 ``|a_lat|``（m/s²）超过死区后的二次软惩罚。"""

    name: ClassVar[str] = "comfort_lat"

    def __init__(self, weight: float = -0.05, deadband: float = 2.0) -> None:
        super().__init__(weight=weight, deadband=float(deadband))
        self.deadband = float(deadband)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        value = _optional_float(step_ctx.get("a_lat"))
        if value is None:
            value = _optional_float(step_ctx.get("accel_lat"))
        return 0.0 if value is None else _hinge(value, self.deadband)


@register_term
class JerkPenalty(Term):
    """舒适：jerk（m/s³）超过死区后的二次软惩罚。"""

    name: ClassVar[str] = "comfort_jerk"

    def __init__(self, weight: float = -0.005, deadband: float = 5.0) -> None:
        super().__init__(weight=weight, deadband=float(deadband))
        self.deadband = float(deadband)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        value = _optional_float(step_ctx.get("jerk"))
        return 0.0 if value is None else _hinge(value, self.deadband)


@register_term
class LaneCenterPenalty(Term):
    """车道保持：偏离本车道中心线的**线性死区**惩罚 ``max(0, min(|d_lat|, clamp) − deadband)``。

    - ``|d_lat|`` 先截断到 ``clamp``（默认 3.0 m）→ 单步原始值上界 ``clamp − deadband = 2.75``；
    - 死区默认 0.25 m（车道中心附近不惩罚）；默认权重 −0.1 → 单步最差 −0.275；
    - **无车道 / 无横向信息 → 0.0**（不惩罚、不崩溃）；横向偏差读取顺序见
      :func:`lane_lateral_offset_from_ctx`（显式 ``d_lat`` → 最近左右车道边界距离回退）；
    - **默认不在** :data:`DEFAULT_TERM_CONFIGS`（默认关）；经 config ``reward.terms`` 或
      CLI ``--reward-term-weight lane_center=-0.1`` 启用（``build_reward_adapter`` 的
      override 会把已注册但未配置的项按给定权重追加）。
    """

    name: ClassVar[str] = "lane_center"

    def __init__(self, weight: float = -0.1, deadband: float = 0.25, clamp: float = 3.0) -> None:
        super().__init__(weight=weight, deadband=float(deadband), clamp=float(clamp))
        self.deadband = float(deadband)
        self.clamp = float(clamp)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        value = lane_lateral_offset_from_ctx(step_ctx)
        if value is None:
            return 0.0
        excess = _clip(abs(value), 0.0, self.clamp) - self.deadband
        return excess if excess > 0.0 else 0.0


@register_term
class SpeedRatioTerm(Term):
    """效率：``clip(speed_ratio, 0, cap)`` 收益（默认权重 +1，鼓励贴近但不超过限速）。

    ``cap`` 默认 1.0：超过限速的部分由 :class:`SpeedLimitViolationPenalty` 惩罚，避免
    效率项奖励超速。
    """

    name: ClassVar[str] = "speed_ratio"

    def __init__(self, weight: float = 1.0, cap: float = 1.0, floor: float = 0.0) -> None:
        super().__init__(weight=weight, cap=float(cap), floor=float(floor))
        self.cap = float(cap)
        self.floor = float(floor)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        ratio = speed_ratio_from_ctx(step_ctx)
        if ratio is None:
            return 0.0
        return _clip(ratio, self.floor, self.cap)


@register_term
class SolidLineCrossingPenalty(Term):
    """合规：触碰/跨越连续实线（0/1 惩罚）。

    触发来源按优先级：

    1. 显式 ``solid_line_crossing`` / ``crossed_solid_line`` 标志；
    2. MetaDrive 逐帧标志 ``on_white_continuous_line`` / ``on_yellow_continuous_line``；
    3. ``left_line_type_id`` / ``right_line_type_id`` / ``line_type_id`` 命中连续实线 id
       （默认 ``SOLID_LINE_TYPE_IDS``，可用 ``solid_type_ids`` 覆盖）。
    """

    name: ClassVar[str] = "solid_line"

    def __init__(
        self,
        weight: float = -2.0,
        solid_type_ids: Any = SOLID_LINE_TYPE_IDS,
    ) -> None:
        ids = frozenset(int(item) for item in solid_type_ids)
        super().__init__(weight=weight, solid_type_ids=tuple(sorted(ids)))
        self.solid_type_ids = ids

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        if _truthy(step_ctx.get("solid_line_crossing")) or _truthy(
            step_ctx.get("crossed_solid_line")
        ):
            return 1.0
        if _truthy(step_ctx.get("on_white_continuous_line")) or _truthy(
            step_ctx.get("on_yellow_continuous_line")
        ):
            return 1.0
        for key in ("left_line_type_id", "right_line_type_id", "line_type_id"):
            value = _optional_float(step_ctx.get(key))
            if value is not None and int(value) in self.solid_type_ids:
                return 1.0
        return 0.0


@register_term
class SpeedLimitViolationPenalty(Term):
    """合规：超速量 ``max(0, speed_ratio − (1 + tolerance))``（容忍 5% 默认）。"""

    name: ClassVar[str] = "speed_limit"

    def __init__(self, weight: float = -5.0, tolerance: float = 0.05) -> None:
        super().__init__(weight=weight, tolerance=float(tolerance))
        self.tolerance = float(tolerance)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        ratio = speed_ratio_from_ctx(step_ctx)
        if ratio is None:
            return 0.0
        return max(0.0, ratio - (1.0 + self.tolerance))


@register_term
class RouteCompletionShaping(Term):
    """达成：势能塑形 ``γΦ(s′) − Φ(s)``（``Φ`` = 截断后的 ``route_completion``）。

    聚合器会把上一步的 ``route_completion`` 注入 ``route_completion_prev``；单独调用时
    若缺少该键则按 ``0.0``（episode 起点）处理。``γ=1`` 时同一 episode 的 shaping 总和只
    取决于首末势能，因此**不改变等终局轨迹的策略序**。
    """

    name: ClassVar[str] = "route_completion"
    shaping: ClassVar[bool] = True

    def __init__(
        self,
        weight: float = 1.0,
        gamma: float = 1.0,
        clip_min: float = 0.0,
        clip_max: float = 1.0,
    ) -> None:
        super().__init__(
            weight=weight, gamma=float(gamma), clip_min=float(clip_min), clip_max=float(clip_max)
        )
        self.gamma = float(gamma)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        current = _optional_float(step_ctx.get("route_completion"))
        if current is None:
            return 0.0
        previous = _optional_float(step_ctx.get("route_completion_prev"))
        if previous is None:
            previous = self.clip_min
        current = _clip(current, self.clip_min, self.clip_max)
        previous = _clip(previous, self.clip_min, self.clip_max)
        return self.gamma * current - previous


#: 默认奖励项配置（权重为先验初值，最终调参在后续 lane；惩罚项权重为负）。
DEFAULT_TERM_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "route_completion", "weight": 1.0, "gamma": 1.0},
    {"name": "speed_ratio", "weight": 1.0, "cap": 1.0},
    {"name": "comfort_lon", "weight": -0.05, "deadband": 2.5},
    {"name": "comfort_lat", "weight": -0.05, "deadband": 2.0},
    {"name": "comfort_jerk", "weight": -0.005, "deadband": 5.0},
    {"name": "solid_line", "weight": -2.0},
    {"name": "speed_limit", "weight": -5.0, "tolerance": 0.05},
    {"name": "crash", "weight": -10.0},
    {"name": "out_of_road", "weight": -8.0},
)
