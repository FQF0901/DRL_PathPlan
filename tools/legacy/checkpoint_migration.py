"""R38-P2a：checkpoint 迁移入口（最小实现；契约 ``docs/checkpoint_migration_contract.md`` §5）。

职责边界（硬约束）：
1. **只在装载路径被调用**：``net/`` 的 forward/rollout 不得 import 本模块；
2. **默认 fail-fast 不变**：未登记 unexpected / missing → 默认拒绝（``LoadError``）；
   本层只把**已登记族**（``legacy_key_mapper.IGNORE_PREFIXES``）从 unexpected 里移走，
   并逐族计数（``dropped_by_family``）；
3. **不参与 forward**：装载后模型输入/输出键集合与"干净加载"路径逐位一致；
4. 逃生门 ``on_partial="record"`` 必须显式给出，报告含 ``explicit_escape=True``。

当前接入状态（R38-P2a）：**模块与单测就位、loader 尚未接线**（eval_runner/trainer 的既有
fail-fast 行为零改动；接线属 D 组/后续清洗波，见 ``p2a_report.md`` ④）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from tools.legacy.legacy_key_mapper import classify_keys, summarize


class LoadError(RuntimeError):
    """默认 fail-fast：未登记 unexpected / missing / 形状不符。"""


@dataclass
class LoadReport:
    ckpt_keys: int = 0
    model_keys: int = 0
    n_load: int = 0
    n_drop: int = 0
    n_reject: int = 0
    n_missing: int = 0
    dropped_by_family: Mapping[str, int] = field(default_factory=dict)
    dropped_keys: tuple[str, ...] = ()
    reject_keys: tuple[str, ...] = ()
    missing_keys: tuple[str, ...] = ()
    shape_mismatch: tuple[str, ...] = ()
    explicit_escape: bool = False
    plan_summary: Mapping[str, Any] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ckpt_keys": self.ckpt_keys,
            "model_keys": self.model_keys,
            "n_load": self.n_load,
            "n_drop": self.n_drop,
            "n_reject": self.n_reject,
            "n_missing": self.n_missing,
            "dropped_by_family": dict(self.dropped_by_family),
            "dropped_keys": list(self.dropped_keys),
            "reject_keys": list(self.reject_keys),
            "missing_keys": list(self.missing_keys),
            "shape_mismatch": list(self.shape_mismatch),
            "explicit_escape": bool(self.explicit_escape),
            "plan_summary": dict(self.plan_summary),
        }


def migrate_state_dict(
    state: Mapping[str, Any], model_keys: Iterable[str]
) -> tuple[dict[str, Any], LoadReport]:
    """ckpt state_dict → 过滤后 state_dict + 报告（只应用**已登记**丢弃族；纯字典操作）。"""
    keys = tuple(str(key) for key in model_keys)
    plan = classify_keys(state.keys(), keys)
    drop_set = set(plan.drop)
    filtered = {key: value for key, value in state.items() if key not in drop_set}
    report = LoadReport(
        ckpt_keys=len(state),
        model_keys=len(keys),
        n_load=len(plan.load),
        n_drop=len(plan.drop),
        n_reject=len(plan.reject),
        n_missing=len(plan.missing),
        dropped_by_family=dict(plan.dropped_by_family),
        dropped_keys=tuple(plan.drop),
        reject_keys=tuple(plan.reject),
        missing_keys=tuple(plan.missing),
        plan_summary=summarize(plan),
        reasons=dict(plan.reasons),
    )
    return filtered, report


def load_checkpoint_state(
    model: Any,
    state: Mapping[str, Any],
    *,
    on_partial: str = "fail",
) -> LoadReport:
    """最小装载路径：classify → 丢弃已登记族 → 形状检查 → 默认 fail-fast → ``load_state_dict``。

    ``on_partial="fail"``（默认）：reject（未登记 unexpected）/ missing / 形状不符任一非零 →
    :class:`LoadError`（先交报告）；``on_partial="record"``（显式逃生门）：放行并置
    ``explicit_escape=True``。**已登记族的丢弃在两档下都只逐族计数，不阻塞。**
    """
    if on_partial not in ("fail", "record"):
        raise ValueError(f"on_partial 只能为 'fail'/'record'，收到 {on_partial!r}")
    model_keys = tuple(model.state_dict().keys())
    filtered, report = migrate_state_dict(state, model_keys)
    model_set = model.state_dict()
    shape_mismatch: list[str] = []
    loadable: dict[str, Any] = {}
    for key, value in filtered.items():
        target = model_set.get(key)
        if target is None:
            # filtered 里只可能剩 reject 类键（未登记 unexpected）——不放行
            continue
        try:
            same_shape = tuple(target.shape) == tuple(value.shape)
        except AttributeError:
            same_shape = False
        if same_shape:
            loadable[key] = value
        else:
            shape_mismatch.append(str(key))
    report.shape_mismatch = tuple(shape_mismatch)
    blocked = bool(report.reject_keys or report.missing_keys or shape_mismatch)
    if blocked and on_partial == "fail":
        raise LoadError(
            "ckpt 含未登记 unexpected / missing / 形状不符："
            f"reject={list(report.reject_keys)[:5]} missing={list(report.missing_keys)[:5]} "
            f"shape_mismatch={list(shape_mismatch)[:5]}（默认 fail-fast；"
            "确需放行请显式 on_partial='record'，并记录 explicit_escape）"
        )
    if blocked and on_partial == "record":
        report.explicit_escape = True
    model.load_state_dict(loadable, strict=False)
    return report


__all__ = ["LoadError", "LoadReport", "migrate_state_dict", "load_checkpoint_state"]
