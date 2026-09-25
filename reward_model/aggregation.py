"""奖励聚合（N2 契约 §3）：逐步稠密 + 终局 outcome + CaRL 式乘性/终止惩罚。

组成
----
- **稠密项**（``kind="dense"``）：按权重求和；``shaping=True`` 的项再乘
  ``shaping_decay`` 退火系数（训练后期退火稠密塑形，契约 §3）。
- **终止型项**（``kind="terminating"``）：原始值 > 0 即 ``done=True`` 并记录原因
  （``crash`` → collision、``out_of_road``）；其加权值在乘性惩罚之后叠加。
- **CaRL 式乘性/终止惩罚**：``carl_rules`` 命中违规时把稠密和乘以 ``factor``
  （默认 0.0，即碰撞清零稠密奖励），并可 ``terminate`` / 追加 ``penalty``。
- **终局 outcome**：``arrive_dest`` / ``collision`` / ``out_of_road`` / ``max_step`` /
  ``error`` 对应 ``terminal_values``（到达加分、碰撞/出界/超时扣分），仅在终局步加一次。

信用分配
--------
- ``dense``：逐步奖励原样（PPO/GAE 口径）。
- ``grouped_discounted``：按 ``group_size`` 分组后计算组边界折扣回报（供 GRPO 消融）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from reward_model.registry import TERMINATING, Term
from reward_model.terms import CRASH_FLAGS

__all__ = [
    "CarlRule",
    "ShapingDecay",
    "AggregationConfig",
    "StepReward",
    "RewardAggregator",
    "default_terminal_values",
    "default_carl_rules",
    "discounted_return",
    "grouped_discounted_returns",
    "assign_credits",
]

#: 终局结果键（与 MetaDrive info 及 eval 协议一致）
TERMINAL_KEYS: tuple[str, ...] = ("arrive_dest", "collision", "out_of_road", "max_step", "error")

#: 这些键一旦命中即视为 episode 结束（max_step 需 info 显式给出标志）
_HARD_TERMINALS: frozenset[str] = frozenset({"arrive_dest", "collision", "out_of_road", "error"})


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def _crash_triggered(ctx: Mapping[str, Any]) -> bool:
    if _truthy(ctx.get("collision")):
        return True
    return any(_truthy(ctx.get(flag)) for flag in CRASH_FLAGS)


def _violation(ctx: Mapping[str, Any], key: str) -> bool:
    """CaRL 违规触发判定；``crash`` 聚合任意 ``crash*`` 标志。"""
    if key == "crash":
        return _crash_triggered(ctx)
    return _truthy(ctx.get(key))


def _carl_reason(key: str) -> str:
    return "collision" if key == "crash" else key


def default_terminal_values() -> dict[str, float]:
    """默认终局 outcome 奖惩（到达 +10、碰撞/出界 -5、超时 -2、基建错误 -5）。"""
    return {
        "arrive_dest": 10.0,
        "collision": -5.0,
        "out_of_road": -5.0,
        "max_step": -2.0,
        "error": -5.0,
    }


@dataclass(frozen=True)
class CarlRule:
    """CaRL 式违规规则。

    - ``factor``：命中后稠密和乘以该系数（``None`` 表示不乘；``0.0`` 即清零）。
    - ``terminate``：命中即终止 episode。
    - ``penalty``：命中后额外叠加的终止惩罚（加在稠密和之外）。
    - ``reason``：终止原因标签（缺省时 crash → ``"collision"``，其余用违规键）。
    """

    factor: float | None = None
    terminate: bool = False
    penalty: float = 0.0
    reason: str = ""


def default_carl_rules() -> dict[str, CarlRule]:
    """默认：碰撞/出界清零稠密奖励并终止（终止惩罚由 term/terminal outcome 承担）。"""
    return {
        "crash": CarlRule(factor=0.0, terminate=True),
        "out_of_road": CarlRule(factor=0.0, terminate=True),
    }


@dataclass(frozen=True)
class ShapingDecay:
    """稠密塑形退火调度：``constant`` / ``linear`` / ``exponential``。

    ``__call__(step)`` 返回 ``[0, 1]`` 区间系数：``linear`` 在 ``steps`` 步内从 ``start``
    线性降到 ``end``；``exponential`` 按几何插值。``steps<=0`` 时恒为 ``start``。
    """

    kind: str = "constant"
    start: float = 1.0
    end: float = 0.0
    steps: int = 0

    def __call__(self, step: int) -> float:
        start = float(self.start)
        end = float(self.end)
        steps = int(self.steps)
        if steps <= 0:
            return start
        progress = min(1.0, max(0.0, float(step) / float(steps)))
        if self.kind == "constant":
            return start
        if self.kind == "linear":
            return start + (end - start) * progress
        if self.kind == "exponential":
            if start <= 0.0 or end <= 0.0:
                return start + (end - start) * progress
            return start * (end / start) ** progress
        raise ValueError(f"未知 shaping decay 类型: {self.kind!r}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ShapingDecay":
        return cls(
            kind=str(config.get("kind", "linear")),
            start=float(config.get("start", 1.0)),
            end=float(config.get("end", 0.0)),
            steps=int(config.get("steps", 0)),
        )


DecayLike = float | ShapingDecay | Callable[[int], float] | Mapping[str, Any] | None


@dataclass
class AggregationConfig:
    """聚合配置（全部可选，默认值见 ``default_*``）。"""

    terminal_values: Mapping[str, float] = field(default_factory=default_terminal_values)
    carl_rules: Mapping[str, CarlRule] = field(default_factory=default_carl_rules)
    shaping_decay: DecayLike = None


@dataclass(frozen=True)
class StepReward:
    """单步聚合结果（``components`` 为各奖励项加权贡献）。"""

    reward: float
    components: dict[str, float]
    raw_components: dict[str, float]
    dense_sum: float
    carl_multiplier: float
    terminating_sum: float
    carl_penalty: float
    terminal_value: float
    terminal_key: str | None
    shaping_decay: float
    done: bool
    reason: str | None


def _resolve_decay(decay: DecayLike) -> Callable[[int], float]:
    if decay is None:
        return lambda step: 1.0
    if isinstance(decay, ShapingDecay):
        return decay
    if isinstance(decay, Mapping):
        return ShapingDecay.from_config(decay)
    if callable(decay):
        return decay
    constant = float(decay)
    return lambda step: constant


def _terminal_key(ctx: Mapping[str, Any]) -> str | None:
    if _truthy(ctx.get("arrive_dest")):
        return "arrive_dest"
    if _crash_triggered(ctx):
        return "collision"
    if _truthy(ctx.get("out_of_road")):
        return "out_of_road"
    if _truthy(ctx.get("error")):
        return "error"
    if _truthy(ctx.get("max_step")):
        return "max_step"
    reason = ctx.get("terminal_reason", ctx.get("reason"))
    if isinstance(reason, str) and reason in TERMINAL_KEYS:
        return reason
    return None


class RewardAggregator:
    """把奖励项按 step 聚合成 ``StepReward``，并维护跨步状态（上一 ``route_completion``）。"""

    def __init__(
        self,
        terms: Iterable[Term],
        config: AggregationConfig | None = None,
    ) -> None:
        self.terms: list[Term] = list(terms)
        names = [term.name for term in self.terms]
        if len(names) != len(set(names)):
            raise ValueError(f"奖励项重名: {names}")
        self.config = config if config is not None else AggregationConfig()
        self._decay = _resolve_decay(self.config.shaping_decay)
        self._prev_route_completion: float | None = None
        self._step_index = 0

    def reset(self) -> None:
        """episode 边界重置（清空上一进度、步计数）。"""
        self._prev_route_completion = None
        self._step_index = 0

    def step(self, step_ctx: Mapping[str, Any], *, step_index: int | None = None) -> StepReward:
        """聚合单步。

        ``step_ctx`` 为普通 ``Mapping``（env ``info`` 直传）；本方法不修改输入。
        ``route_completion_prev`` 缺失时由聚合器注入（首个 step 记 0.0）。
        """
        ctx: dict[str, Any] = dict(step_ctx)
        index = self._step_index if step_index is None else int(step_index)
        self._step_index = index + 1
        if "route_completion" in ctx and "route_completion_prev" not in ctx:
            ctx["route_completion_prev"] = (
                0.0 if self._prev_route_completion is None else self._prev_route_completion
            )

        decay = float(self._decay(index))
        raw_components: dict[str, float] = {}
        components: dict[str, float] = {}
        dense_sum = 0.0
        terminating_sum = 0.0
        done = _truthy(ctx.get("done"))
        reason: str | None = None

        for term in self.terms:
            value = float(term.compute(ctx))
            raw_components[term.name] = value
            contribution = term.weight * value
            if term.kind == TERMINATING:
                terminating_sum += contribution
                if value > 0.0 and reason is None:
                    done = True
                    reason = term.reason or term.name
            else:
                if term.shaping:
                    contribution *= decay
                dense_sum += contribution
            components[term.name] = contribution

        multiplier = 1.0
        carl_penalty = 0.0
        for key, rule in self.config.carl_rules.items():
            if not _violation(ctx, key):
                continue
            if rule.factor is not None:
                multiplier *= float(rule.factor)
            carl_penalty += float(rule.penalty)
            if rule.terminate and reason is None:
                done = True
                reason = rule.reason or _carl_reason(key)

        terminal_key = _terminal_key(ctx)
        terminal_value = (
            float(self.config.terminal_values.get(terminal_key, 0.0)) if terminal_key else 0.0
        )
        if terminal_key in _HARD_TERMINALS or terminal_key == "max_step":
            done = True
        if reason is None and terminal_key is not None:
            reason = terminal_key

        route_completion = ctx.get("route_completion")
        if route_completion is not None:
            try:
                self._prev_route_completion = float(route_completion)
            except (TypeError, ValueError):
                pass

        reward = (
            dense_sum * multiplier + terminating_sum + carl_penalty + terminal_value
        )
        return StepReward(
            reward=float(reward),
            components=components,
            raw_components=raw_components,
            dense_sum=float(dense_sum),
            carl_multiplier=float(multiplier),
            terminating_sum=float(terminating_sum),
            carl_penalty=float(carl_penalty),
            terminal_value=float(terminal_value),
            terminal_key=terminal_key,
            shaping_decay=decay,
            done=bool(done),
            reason=reason,
        )


# --------------------------------------------------------------------------- #
# 信用分配（PPO 用 dense；GRPO 消融用 grouped_discounted）
# --------------------------------------------------------------------------- #

def discounted_return(rewards: Sequence[float], gamma: float = 0.99) -> float:
    """标准折扣回报 ``Σ γ^t r_t``。"""
    return float(sum((float(gamma) ** index) * float(value) for index, value in enumerate(rewards)))


def grouped_discounted_returns(
    rewards: Sequence[float],
    *,
    gamma: float = 0.99,
    group_size: int = 6,
) -> list[float]:
    """把逐步奖励切成连续组，返回每个组的折扣回报（组边界处含跨组折扣）。

    对第 ``g`` 组（长度 ``m_g``）：``G_g = Σ_i γ^i r_{g,i} + γ^{m_g} · G_{g+1}``。
    供 GRPO 消融把序列信用压缩到组级标量。
    """
    group_size = int(group_size)
    if group_size < 1:
        raise ValueError(f"group_size 必须 >= 1，实际 {group_size}")
    gamma = float(gamma)
    if not 0.0 < gamma <= 1.0:
        raise ValueError(f"gamma 必须在 (0, 1]，实际 {gamma}")
    values = [float(value) for value in rewards]
    groups = [values[start : start + group_size] for start in range(0, len(values), group_size)]
    returns = [0.0] * len(groups)
    carry = 0.0
    for group_index in range(len(groups) - 1, -1, -1):
        group = groups[group_index]
        local = sum((gamma**offset) * value for offset, value in enumerate(group))
        returns[group_index] = local + (gamma ** len(group)) * carry
        carry = returns[group_index]
    return returns


def assign_credits(
    rewards: Sequence[float],
    *,
    mode: str = "dense",
    gamma: float = 0.99,
    group_size: int = 6,
) -> list[float]:
    """信用分配入口。

    - ``dense``：原样返回逐步奖励（PPO/GAE）。
    - ``grouped_discounted``：按组折扣回报并广播回组内每一步（组内各步拿同一组回报，
      形状与输入一致，便于批处理；组级标量用 :func:`grouped_discounted_returns`）。
    """
    values = [float(value) for value in rewards]
    if mode == "dense":
        return values
    if mode == "grouped_discounted":
        group_returns = grouped_discounted_returns(values, gamma=gamma, group_size=group_size)
        if not group_returns:
            return []
        size = int(group_size)
        return [
            group_returns[min(index // size, len(group_returns) - 1)] for index in range(len(values))
        ]
    raise ValueError(f"未知 credit mode: {mode!r}（可选 dense / grouped_discounted）")
