"""可插拔奖励项注册表（N2 契约 §3）。

约定
----
- 每个奖励项是 :class:`Term` 子类，暴露 ``name`` / ``weight`` /
  ``compute(step_ctx) -> float``；``compute`` 只读普通 ``Mapping``（step 级 info），
  不依赖 env / metadrive，因此可脱离环境单测。
- ``compute`` 返回**未加权原始值**：违规/代价项返回 ``>= 0`` 的程度量，进度塑形项返回
  有符号的势能差分；聚合器用 ``weight`` 加权后求和（惩罚项权重为负）。
- 注册通过 ``@register_term`` 显式完成；导入本模块无任何副作用（不建环境、不做 IO）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterable, Mapping

__all__ = [
    "Term",
    "TermConfig",
    "register_term",
    "available_terms",
    "get_term_class",
    "make_term",
    "build_terms",
]

#: ``Term.kind`` 的合法取值：稠密逐步项 / 触发即终止的安全项
DENSE = "dense"
TERMINATING = "terminating"


class Term(ABC):
    """奖励项基类。

    属性
    ----
    - ``name``：全局唯一的注册名（``register_term`` 会校验）。
    - ``kind``：``"dense"``（逐步稠密）或 ``"terminating"``（原始值 > 0 即终止 episode）。
    - ``shaping``：为 True 时，聚合器对该项施加 ``shaping_decay`` 退火系数。
    - ``reason``：终止原因标签（仅 ``terminating`` 项有意义，如 ``"collision"``）。
    """

    name: ClassVar[str] = ""
    kind: ClassVar[str] = DENSE
    shaping: ClassVar[bool] = False
    reason: ClassVar[str] = ""

    def __init__(self, weight: float = 1.0, **params: Any) -> None:
        self.weight = float(weight)
        self.params: dict[str, Any] = dict(params)

    @abstractmethod
    def compute(self, step_ctx: Mapping[str, Any]) -> float:
        """返回未加权原始值（惩罚 >= 0；势能塑形可为负）。"""

    @property
    def terminating(self) -> bool:
        """是否终止型项（``kind == "terminating"`` 的便捷属性）。"""
        return self.kind == TERMINATING

    def weighted(self, step_ctx: Mapping[str, Any]) -> float:
        """``weight * compute(step_ctx)``（已加权贡献）。"""
        return self.weight * float(self.compute(step_ctx))

    def is_triggered(self, step_ctx: Mapping[str, Any]) -> bool:
        """原始值 > 0 视为触发（终止型项据此结束 episode）。"""
        return float(self.compute(step_ctx)) > 0.0

    def describe(self) -> dict[str, Any]:
        """可序列化描述（配置回显/日志用）。"""
        return {
            "name": self.name,
            "weight": self.weight,
            "kind": self.kind,
            "shaping": self.shaping,
            "params": dict(self.params),
        }


#: 注册名 -> 奖励项类（同一进程内唯一）
_TERM_REGISTRY: dict[str, type[Term]] = {}


def register_term(cls: type[Term]) -> type[Term]:
    """把奖励项类登记进全局注册表（装饰器用法）。"""
    name = str(getattr(cls, "name", "") or "")
    if not name:
        raise ValueError(f"奖励项 {cls.__name__} 缺少非空 name")
    existing = _TERM_REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"奖励项重名: {name!r} 已注册为 {existing.__name__}")
    if cls.kind not in (DENSE, TERMINATING):
        raise ValueError(f"奖励项 {name!r} 的 kind 非法: {cls.kind!r}")
    _TERM_REGISTRY[name] = cls
    return cls


def available_terms() -> tuple[str, ...]:
    """已注册奖励项名（字典序，便于确定性遍历/测试）。"""
    return tuple(sorted(_TERM_REGISTRY))


def get_term_class(name: str) -> type[Term]:
    """按注册名取奖励项类；未知名字抛出 ``KeyError``（附可用名单）。"""
    try:
        return _TERM_REGISTRY[str(name)]
    except KeyError as exc:
        raise KeyError(f"未注册的奖励项 {name!r}；可用: {available_terms()}") from exc


def make_term(name: str, weight: float = 1.0, **params: Any) -> Term:
    """按注册名实例化奖励项（``params`` 透传给具体类）。"""
    return get_term_class(name)(weight=weight, **params)


@dataclass(frozen=True)
class TermConfig:
    """奖励项配置：``enabled=False`` 时 ``build`` 返回 None（单项可开关）。"""

    name: str
    weight: float = 1.0
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def build(self) -> "Term | None":
        if not self.enabled:
            return None
        return make_term(self.name, weight=self.weight, **dict(self.params))


_CONFIG_KEYS = frozenset({"name", "weight", "params", "enabled"})


def build_terms(configs: Iterable[TermConfig | Mapping[str, Any]]) -> list[Term]:
    """由配置序列构建奖励项列表（保持输入顺序，跳过 ``enabled=False``）。

    配置可以是 :class:`TermConfig`，也可以是普通 dict：

    - 识别键：``name`` / ``weight`` / ``enabled`` / ``params``；
    - 其余扁平键直接并入 ``params``（方便 YAML 直传，如
      ``{"name": "comfort_lon", "weight": -0.05, "deadband": 2.5}``）。
    """
    terms: list[Term] = []
    for config in configs:
        if isinstance(config, TermConfig):
            term = config.build()
        elif isinstance(config, Mapping):
            name = config.get("name")
            if name is None:
                raise ValueError(f"奖励项配置缺少 name: {dict(config)!r}")
            params = dict(config.get("params") or {})
            params.update({key: value for key, value in config.items() if key not in _CONFIG_KEYS})
            term = TermConfig(
                name=str(name),
                weight=float(config.get("weight", 1.0)),
                params=params,
                enabled=bool(config.get("enabled", True)),
            ).build()
        else:
            raise TypeError(f"不支持的奖励项配置类型: {type(config)!r}")
        if term is not None:
            terms.append(term)
    return terms
