"""R38-P2a：checkpoint 迁移入口（最小实现；契约 ``docs/repo_handbook.md`` 附录 D §5）。

职责边界（硬约束）：
1. **只在装载路径被调用**：``net/`` 的 forward/rollout 不得 import 本模块；
2. **默认 fail-fast 不变**：未登记 unexpected / missing → 默认拒绝（``LoadError``）；
   本层只把**已登记族**（``legacy_key_mapper.IGNORE_PREFIXES``）与**已登记改名族**
   （精确 ``RENAME_MAP`` + 前缀 ``PREFIX_RENAME_MAP``，目标键存在时）从 unexpected 里移走，
   并逐族计数（``dropped_by_family`` / ``renamed_by_family``）；
3. **不参与 forward**：装载后模型输入/输出键集合与"干净加载"路径逐位一致；
4. 逃生门 ``on_partial="record"`` 必须显式给出，报告含 ``explicit_escape=True``。

当前接入状态（R38-P2a/P2b）：**模块与单测就位、pipeline loader 尚未接线**（eval_runner/trainer 的既有
fail-fast 行为零改动）。R38-P2b 补：改名族（``policy.log_std→policy.logstd`` +
``path_head.→policy.structured_action.`` 双实体收编）在分类层应用；.slim 评估链的装载调用点
（``pipeline.trainer.load_checkpoint``）若需读旧 ckpt，可在调用点显式经本模块（见 p2b_report）。
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
    n_rename: int = 0
    rename_map: Mapping[str, str] = field(default_factory=dict)
    dropped_by_family: Mapping[str, int] = field(default_factory=dict)
    renamed_by_family: Mapping[str, int] = field(default_factory=dict)
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
            "n_rename": self.n_rename,
            "rename_map": dict(self.rename_map),
            "dropped_by_family": dict(self.dropped_by_family),
            "renamed_by_family": dict(self.renamed_by_family),
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
    """ckpt state_dict → 过滤后 state_dict + 报告（应用**已登记**丢弃族 + 改名族；纯字典操作）。

    R38-P2b：改名族（``policy.log_std→policy.logstd`` 精确 + ``path_head.→policy.structured_action.``
    前缀）在目标键存在于模型时应用（键名重写、张量原样）；改名目标缺席（条件族，如默认装配无
    结构化头）⇒ 该键进 ``dropped`` 并逐族计数（``rename_target_absent``，不静默）。
    """
    keys = tuple(str(key) for key in model_keys)
    plan = classify_keys(state.keys(), keys)
    drop_set = set(plan.drop)
    filtered: dict[str, Any] = {}
    for key, value in state.items():
        if key in drop_set:
            continue
        filtered[str(plan.rename.get(key, key))] = value
    report = LoadReport(
        ckpt_keys=len(state),
        model_keys=len(keys),
        n_load=len(plan.load),
        n_drop=len(plan.drop),
        n_reject=len(plan.reject),
        n_missing=len(plan.missing),
        n_rename=len(plan.rename),
        rename_map=dict(plan.rename),
        dropped_by_family=dict(plan.dropped_by_family),
        renamed_by_family=dict(plan.renamed_by_family),
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


def load_checkpoint_migrated(
    path: str, model: Any, *, strict: bool = False, state: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """实用装载入口（.slim 评估/训练链调用点 + **R39-A1 起接入正式 loader**）：旧键面经
    migration 后装载（R38-P2b）。

    流程：``torch.load``（``state`` 给出时跳过 —— 调用点已装载/已计 hash）→
    ``migrate_state_dict``（已登记族 drop + 改名族 rename）→
    ``model.load_state_dict(filtered, strict=strict)`` → 返回
    ``{"meta", "missing_keys", "unexpected_keys", "migration"}``。
    ``migration`` = :meth:`LoadReport.to_dict`（逐族计数；供调用点落盘/审计）。
    **注意**：未登记 unexpected 键仍会出现在 ``unexpected_keys``（默认 fail-fast 语义不弱化，
    由调用点决定拒绝或记录）；形状不符键在 ``load_state_dict(strict=False)`` 下同样抛
    ``RuntimeError``（本原语不做形状过滤 —— 正式 loader 的形状门在调用点，见
    ``pipeline/eval_runner._load_ckpt_model`` 的"干净 ⇒ strict=True 原语装载"路径）。
    """
    import torch

    if state is None:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        state = payload
        if isinstance(payload, Mapping):
            for key in ("model", "state_dict", "model_state_dict", "net"):
                if key in payload and isinstance(payload[key], Mapping):
                    state = payload[key]
                    break
        meta = dict(payload.get("meta") or {}) if isinstance(payload, Mapping) else {}
    else:
        meta = {}
    filtered, report = migrate_state_dict(state, model.state_dict().keys())
    missing, unexpected = model.load_state_dict(filtered, strict=bool(strict))
    return {
        "meta": meta,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "migration": report.to_dict(),
    }


__all__ = [
    "LoadError",
    "LoadReport",
    "load_checkpoint_migrated",
    "load_checkpoint_state",
    "migrate_state_dict",
]
