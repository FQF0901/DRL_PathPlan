"""内置奖励项（N2 契约 §3）。

类别与项
--------
- 安全（终止型）：``crash``（任意 ``crash*`` 标志）/ ``out_of_road``。
- 安全（稠密近失，**默认关**）：``ttc``（前车时距危险度；无前车 / 缺键 = 0）。
- 舒适（死区二次软惩罚）：``comfort_lon`` / ``comfort_lat`` / ``comfort_jerk``。
- 车道保持（死区线性惩罚，**默认关**）：``lane_center``（偏离本车道中心线 ``|d_lat|``；无车道信息时为 0）。
- 车道保持（贴近边界罚，**默认关**）：``lane_boundary``（``lane_half_width_m − |d_lat|`` 小于
  阈值时线性罚；只罚"快出界"，不罚居中偏离）。
- 效率：``speed_ratio``（``v / 车道限速`` 的截断收益，防爬行；超速交由合规项）。
- 效率（低速蠕动，v5 默认启用）：``low_speed``（``v < 2 m/s`` 时按缺口线性罚，与评测 KPI
  ``CRAWL_SPEED_MPS`` 同阈值）。
- 合规：``solid_line``（连续实线跨越）/ ``speed_limit``（超速量）。
- 达成：``route_completion``（势能塑形 ``γΦ(s') − Φ(s)``，策略序保持）。

所有项只读 ``step_ctx`` 字典（键名与 MetaDrive ``info`` 对齐，见 ``env/metadrive_env.py``
与 ``env/obs/ld.py``），不 import env/metadrive。
"""

from __future__ import annotations

import math
import warnings

from typing import Any, ClassVar, Mapping

from reward_model.registry import TERMINATING, Term, register_term

__all__ = [
    "CRASH_FLAGS",
    "SOLID_LINE_TYPE_IDS",
    "CrashPenalty",
    "OutOfRoadPenalty",
    "TTCLeadPenalty",
    "LeadGapPenalty",
    "LongitudinalAccelPenalty",
    "LateralAccelPenalty",
    "JerkPenalty",
    "LaneCenterPenalty",
    "LaneBoundaryPenalty",
    "LowSpeedPenalty",
    "SpeedRatioTerm",
    "SolidLineCrossingPenalty",
    "SpeedLimitViolationPenalty",
    "RouteCompletionShaping",
    "DEFAULT_TERM_CONFIGS",
    "ego_speed_from_ctx",
    "lane_lateral_offset_from_ctx",
    "lane_half_width_from_ctx",
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

#: 低速蠕动阈值（m/s）：``low_speed`` 项的默认阈值，与评测 KPI
#: ``pipeline.eval_runner.CRAWL_SPEED_MPS = 2.0``（``low_speed_step_ratio`` / ``crawl_seconds``）
#: 对齐（本模块不 import pipeline，一致性由 ``tests/test_reward_terms.py`` 断言）。
LOW_SPEED_THRESHOLD_MPS: float = 2.0


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


def ego_speed_from_ctx(step_ctx: Mapping[str, Any]) -> float | None:
    """自车速度（m/s，标量）：读取顺序 ``speed`` → ``velocity``（与 :func:`speed_ratio_from_ctx` 一致）。

    MetaDrive ``info["velocity"]`` 即自车标量速度（m/s，``base_vehicle.py:245``）；``speed``
    为评测/合成序列的别名。非数值 / 非有限值 / 非标量（如向量列表）→ ``None``。
    """
    for key in ("speed", "velocity"):
        value = _optional_float(step_ctx.get(key))
        if value is not None:
            return value if math.isfinite(value) else None
    return None


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


def lane_half_width_from_ctx(step_ctx: Mapping[str, Any]) -> float | None:
    """车道半宽（m）：优先显式 ``lane_half_width_m``（pipeline 侧注入）。

    回退：由最近的左右车道边界距离推导 ``(dist_to_left_side + dist_to_right_side) / 2``
    （两者之和 = 本车道宽度，与 :func:`lane_lateral_offset_from_ctx` 的回退同源）。
    显式键非正 / 非有限，或回退值非正 → ``None``（调用方按无车道信息处理）。
    """
    explicit = _optional_float(step_ctx.get("lane_half_width_m"))
    if explicit is not None and math.isfinite(explicit) and explicit > 0.0:
        return explicit
    left = _optional_float(step_ctx.get("dist_to_left_side"))
    right = _optional_float(step_ctx.get("dist_to_right_side"))
    if (
        left is None
        or right is None
        or not math.isfinite(left)
        or not math.isfinite(right)
    ):
        return None
    half = 0.5 * (left + right)
    return half if half > 0.0 else None


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


class _WarnMissingInputMixin:
    """缺键诊断：每个项实例**只告警一次**（逐帧缺键时避免训练日志刷屏）。"""

    _warned_missing_input: bool = False

    def _warn_missing_input(self, message: str) -> None:
        if self._warned_missing_input:
            return
        self._warned_missing_input = True
        warnings.warn(message, UserWarning, stacklevel=3)


@register_term
class TTCLeadPenalty(Term, _WarnMissingInputMixin):
    """安全（稠密近失罚，**默认关**）：由前车时距（TTC）定义的危险度；无前车 / 缺键 = 0。

    定义
    ----
    ``ttc = gap / max(v_ego − v_lead, ε)``（s）。原始值取**逆时距超出阈值**的量并封顶：

    ``raw = max(0, 1 / max(ttc, ttc_floor) − 1 / ttc_threshold)``

    - 连续：``ttc = ttc_threshold``（默认 2.0 s）处 raw 恰为 0，``ttc`` 更大恒为 0；
    - 单调：``gap`` 越小 / 接近速度越大 → ``ttc`` 越小 → raw 越大；
    - 封顶：``ttc_floor``（默认 0.5 s）把 ``1/ttc`` 钉在 ≤ 2 /s → raw ≤ ``2 − 0.5 = 1.5``
      （默认权重 −0.5 → 单步最差 −0.75），避免贴车时 ``1/ttc`` 发散；
    - ``v_ego − v_lead <= 0``（未接近前车）→ 0；
    - 与 KPI 的 ``min_ttc`` 无关：本项只消费 step_ctx，不读 ctx 里可能存在的同名 ``ttc``
      键（评测口径由 ``kpi.py`` 从 eval 记录算）。

    输入键（pipeline 侧注入；与 PP/IDM expert 的 ``pp_lead_gap_m`` / 前车纵向速度投影同口径）
    ------------------------------------------------------------------------------------
    - ``lead_gap_m``：自车车头到前车车尾的净距（m）；``<= 0``（含 −1 哨兵）= 无前车 → 0；
    - ``lead_speed_mps``：前车速度在自车纵轴上的投影（m/s，>= 0）；
    - 自车速度：``speed`` → ``velocity``（见 :func:`ego_speed_from_ctx`）。

    必需键缺失 / 非有限值 → 0 且**只告警一次**（每实例）；``gap <= 0`` 的"无前车"路径属
    正常语义、不告警。启用：config ``stages.C.reward.terms`` 追加，或 CLI
    ``--reward-term-weight ttc=-0.5``；默认**不在** :data:`DEFAULT_TERM_CONFIGS`。
    """

    name: ClassVar[str] = "ttc"

    def __init__(
        self,
        weight: float = -0.5,
        ttc_threshold: float = 2.0,
        ttc_floor: float = 0.5,
    ) -> None:
        if ttc_threshold <= 0.0 or ttc_floor <= 0.0:
            raise ValueError(
                f"ttc_threshold / ttc_floor 必须 > 0，收到 {ttc_threshold} / {ttc_floor}"
            )
        super().__init__(
            weight=weight, ttc_threshold=float(ttc_threshold), ttc_floor=float(ttc_floor)
        )
        self.ttc_threshold = float(ttc_threshold)
        self.ttc_floor = float(ttc_floor)
        self.inverse_threshold = 1.0 / self.ttc_threshold

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        gap = _optional_float(step_ctx.get("lead_gap_m"))
        lead_speed = _optional_float(step_ctx.get("lead_speed_mps"))
        ego_speed = ego_speed_from_ctx(step_ctx)
        if (
            gap is None
            or lead_speed is None
            or ego_speed is None
            or not math.isfinite(gap)
            or not math.isfinite(lead_speed)
        ):
            self._warn_missing_input(
                "ttc: 缺少 lead_gap_m / lead_speed_mps / 自车速度(speed|velocity) 之一或值非法"
                " → 该项记 0（每个项实例只告警一次）"
            )
            return 0.0
        if gap <= 0.0:  # 无前车（PP/IDM 约定：非正净距含 −1 哨兵）
            return 0.0
        closing = ego_speed - lead_speed
        if closing <= 0.0:  # 未在接近前车 → 无近失
            return 0.0
        ttc = gap / closing
        raw = 1.0 / max(ttc, self.ttc_floor) - self.inverse_threshold
        return raw if raw > 0.0 else 0.0


@register_term
class LeadGapPenalty(Term, _WarnMissingInputMixin):
    """安全（稠密近碰/车距罚，**默认关**）：前车净距低于参考距时线性罚。

    定义
    ----
    ``raw = clamp((gap_ref − gap) / gap_ref, 0, cap)``（``gap = lead_gap_m``）：

    - 连续：``gap = gap_ref`` 处 raw 恰为 0，更大恒为 0；``gap → 0`` 时 raw → 1；
    - 单调：``gap`` 越小 → raw 越大；与接近速度**无关**（不需要 ``closing > 0``）——
      这是与 :class:`TTCLeadPenalty` 的语义区别：本项罚"近距/短头距"本身，
      在等速跟车（TTC = ∞）与减速接近时同样触发，属更早的安全裕度信号；
    - ``gap <= 0``（含 −1 哨兵）= 无前车 → 0（正常语义、不告警）；
    - 只读 ``step_ctx``；不消费自车速度 / 前车速度键（区别于 ttc 的输入面）。

    输入键：``lead_gap_m``（m；pipeline 侧注入，同 ttc）。必需键缺失 / 非有限值 →
    0 且**只告警一次**（每实例）。默认**不在** :data:`DEFAULT_TERM_CONFIGS`；P4 预注册臂
    （``docs/v6_program_prereg.md`` §7.5 候选 B）经 config 启用。
    """

    name: ClassVar[str] = "lead_gap"

    def __init__(self, weight: float = -1.0, gap_ref: float = 8.0, cap: float = 1.0) -> None:
        if gap_ref <= 0.0:
            raise ValueError(f"lead_gap gap_ref 必须 > 0，收到 {gap_ref}")
        if cap <= 0.0:
            raise ValueError(f"lead_gap cap 必须 > 0，收到 {cap}")
        super().__init__(weight=weight, gap_ref=float(gap_ref), cap=float(cap))
        self.gap_ref = float(gap_ref)
        self.cap = float(cap)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        gap = _optional_float(step_ctx.get("lead_gap_m"))
        if gap is None or not math.isfinite(gap):
            self._warn_missing_input(
                "lead_gap: 缺少 lead_gap_m 或值非法 → 该项记 0（每个项实例只告警一次）"
            )
            return 0.0
        if gap <= 0.0:  # 无前车（PP/IDM 约定：非正净距含 −1 哨兵）
            return 0.0
        return _clip((self.gap_ref - gap) / self.gap_ref, 0.0, self.cap)


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
class LaneBoundaryPenalty(Term, _WarnMissingInputMixin):
    """车道保持（贴近车道边界/路缘罚，**默认关**）：只罚"快出界"，不罚居中偏离。

    定义
    ----
    ``margin = lane_half_width_m − |d_lat|``（自车中心到本车道边界的剩余余量，m）；

    ``raw = max(0, margin_threshold − max(margin, 0))``

    - 余量 ≥ ``margin_threshold``（默认 0.5 m）→ 0：车道内任意合法偏移都不罚，与
      :class:`LaneCenterPenalty` 的"偏离中心"死区罚互补（后者从 ``|d_lat| > 0.25`` 起罚，
      本项只在 ``|d_lat| > lane_half_width − 0.5`` 时触发）；
    - 余量 < 阈值 → 线性罚，越贴线越大；``margin <= 0``（车中心已压线/越界）时
      ``max(margin, 0)`` 把 raw 封顶在 ``margin_threshold``（默认权重 −0.2 → 单步最差 −0.1），
      避免"越界越深罚越重"的无界斜坡；
    - 单调：同一 ``lane_half_width_m`` 下 ``|d_lat|`` 越大 raw 越大（不增）；
    - 语义边界（如实记录）：合法变道穿越边界带（``|d_lat| > lane_half_width_m − 阈值``）的
      少数帧会触发小额罚（约 0.5 m 带宽 × 单步 ≤ 0.1），属"贴近路缘"信号本身；ctx 无可靠
      信号可区分合法变道与跑偏，故不做变道豁免。

    输入键
    ------
    - ``d_lat``：同 :func:`lane_lateral_offset_from_ctx`（显式别名 → 左右边界距离回退）；
    - ``lane_half_width_m``：车道半宽（pipeline 侧注入）；缺失时回退
      ``(dist_to_left_side + dist_to_right_side) / 2``（见 :func:`lane_half_width_from_ctx`）。

    任一缺失 / 非法 → 0 且**只告警一次**（每实例）。启用：config
    ``stages.C.reward.terms`` 追加，或 CLI ``--reward-term-weight lane_boundary=-0.2``；
    默认**不在** :data:`DEFAULT_TERM_CONFIGS`。
    """

    name: ClassVar[str] = "lane_boundary"

    def __init__(self, weight: float = -0.2, margin_threshold: float = 0.5) -> None:
        if margin_threshold <= 0.0:
            raise ValueError(f"margin_threshold 必须 > 0，收到 {margin_threshold}")
        super().__init__(weight=weight, margin_threshold=float(margin_threshold))
        self.margin_threshold = float(margin_threshold)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        d_lat = lane_lateral_offset_from_ctx(step_ctx)
        half_width = lane_half_width_from_ctx(step_ctx)
        if d_lat is None or half_width is None or not math.isfinite(d_lat):
            self._warn_missing_input(
                "lane_boundary: 缺少 d_lat / lane_half_width_m 之一或值非法 → 该项记 0"
                "（每个项实例只告警一次）"
            )
            return 0.0
        margin = half_width - abs(d_lat)
        deficit = self.margin_threshold - max(margin, 0.0)
        return deficit if deficit > 0.0 else 0.0


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
class LowSpeedPenalty(Term, _WarnMissingInputMixin):
    """效率（低速蠕动罚，v5 默认启用）：``v < 2 m/s`` 时按速度缺口线性罚。

    定义
    ----
    ``raw = clamp(1 − v / threshold, 0, 1)``（默认 ``threshold = 2 m/s``，见
    :data:`LOW_SPEED_THRESHOLD_MPS`；与评测 KPI ``CRAWL_SPEED_MPS`` 同阈值）：

    - ``v = 0`` → raw 1.0（默认权重 −0.2 → 单步 −0.2）；``v = 1`` → 0.5（−0.1）；
    - ``v = threshold`` 处连续为 0，``v > threshold`` 恒为 0（不奖励高速，只罚蠕动）；
    - 负速度 / 异常值：``clamp`` 上界 1.0（不产生超过权重的发散惩罚）；
    - 只读 ``step_ctx``；速度口径 = :func:`ego_speed_from_ctx`（``speed`` → ``velocity``）。

    缺键：``speed`` / ``velocity`` 均缺失或非有限 → 0 且**只告警一次**（每实例）。
    默认启用：在 :data:`DEFAULT_TERM_CONFIGS` 中（v5 剖面 C 的一部分）。
    """

    name: ClassVar[str] = "low_speed"

    def __init__(
        self, weight: float = -0.2, threshold: float = LOW_SPEED_THRESHOLD_MPS
    ) -> None:
        if threshold <= 0.0:
            raise ValueError(f"low_speed threshold 必须 > 0，收到 {threshold}")
        super().__init__(weight=weight, threshold=float(threshold))
        self.threshold = float(threshold)

    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        speed = ego_speed_from_ctx(step_ctx)
        if speed is None:
            self._warn_missing_input(
                "low_speed: 缺少自车速度（speed|velocity）或值非法 → 该项记 0"
                "（每个项实例只告警一次）"
            )
            return 0.0
        return _clip(1.0 - speed / self.threshold, 0.0, 1.0)


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


#: 默认奖励项配置（v5 剖面 C 底座；惩罚项权重为负）。
#: - ``speed_ratio`` 权重 = **0.4**（v5 §2 唯一改动权重的保持项）；
#: - ``low_speed`` **默认启用**（v5 §4；阈值 2 m/s 与评测 KPI 对齐）；
#: - ``route_completion`` 默认权重保持 1.0（基准档）；P4 rc 扫档 3/10/30 经 config
#:   ``stages.C.reward.terms`` 或 CLI ``--reward-term-weight route_completion=<档>`` 覆盖，
#:   各档终局值由 ``tools/reward_audit.py`` 反解（配置草案见审计报告）。
DEFAULT_TERM_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "route_completion", "weight": 1.0, "gamma": 1.0},
    {"name": "speed_ratio", "weight": 0.4, "cap": 1.0},
    {"name": "low_speed", "weight": -0.2},
    {"name": "comfort_lon", "weight": -0.05, "deadband": 2.5},
    {"name": "comfort_lat", "weight": -0.05, "deadband": 2.0},
    {"name": "comfort_jerk", "weight": -0.005, "deadband": 5.0},
    {"name": "solid_line", "weight": -2.0},
    {"name": "speed_limit", "weight": -5.0, "tolerance": 0.05},
    {"name": "crash", "weight": -10.0},
    {"name": "out_of_road", "weight": -8.0},
)
