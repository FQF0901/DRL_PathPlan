"""R38-P2a：旧 ckpt 键面映射（纯函数；无 torch、无 IO、无副作用）。

出处：``docs/checkpoint_migration_contract.md`` §4.2（丢弃族常量）/ §4.4（改名族）/ §5（接口草案）。
范围（最小实现）：**B 组 K-anchor** 的 ignore 集 —— ``plan_head.anchor*``（16 键：``anchors``/
``anchor_ds``/``anchor_dth``/``anchor_embed``/``anchor_head.*``/``residual_head.*``/``speed_head.*``）
+ ``encoders.lane.*``/``encoders.ttc.*``（4 键；v8 已删通道）= 20 键（manifest B 组第 10 项）。

纪律（契约 §5 硬约束）：
1. 本模块只做"键面三档分类"（load/drop/reject），**不**决定 forward 语义、不 import ``net``；
2. 默认 fail-fast 不变：未登记的 unexpected 键 = ``reject``（装载方必须拒绝，不得静默丢弃）；
3. 每次丢弃逐族计数（``dropped_by_family``），禁止"总计 N 个 unexpected"一笔带过；
4. ``RENAME_MAP``/``INIT_FROM_CALIBRATION`` 为契约预留接口，B 组范围为空表（G/D 组再填）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: 已登记丢弃族（前缀匹配；顺序无关）。B 组：K-anchor 7 前缀 + v8 已删通道 2 前缀 = 20 键。
IGNORE_PREFIXES: tuple[str, ...] = (
    # K-anchor（13 个 ckpt 带此族；plan_head.py@9577078:120-123,91-117）
    "plan_head.anchors",
    "plan_head.anchor_ds",
    "plan_head.anchor_dth",
    "plan_head.anchor_embed",
    "plan_head.anchor_head.",
    "plan_head.residual_head.",
    "plan_head.speed_head.",
    # v8 已删通道（15 个 ckpt 带此族；net/encoders.py 注明 lane/ttc 已删除）
    "encoders.lane.",
    "encoders.ttc.",
)

#: 改名族（契约 §4.4：``policy.log_std → policy.logstd``；B 组范围不启用，预留）。
RENAME_MAP: dict[str, str] = {}

#: 标定初始化族（契约 §4.1：``path_head.*`` 缺键时来自 ``runs/round36/k1/calibration.json``；
#: B 组范围不启用，预留）。
INIT_FROM_CALIBRATION: dict[str, dict[str, str]] = {}


def family_of(key: str) -> str | None:
    """键 → 已登记族前缀（未登记 → ``None``）。"""
    for prefix in IGNORE_PREFIXES:
        if key.startswith(prefix):
            return prefix
    return None


@dataclass(frozen=True)
class MigrationPlan:
    """键面三档分类结果（契约 §5 ``classify_keys`` 的最小实现）。"""

    load: tuple[str, ...]
    drop: tuple[str, ...]
    reject: tuple[str, ...]          # 未登记 unexpected：装载方默认 fail-fast
    missing: tuple[str, ...]         # 模型需要但 ckpt 没有（装载方决定 init/拒载）
    rename: Mapping[str, str] = field(default_factory=dict)
    init: tuple[str, ...] = ()
    dropped_by_family: Mapping[str, int] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok_without_escape(self) -> bool:
        """默认口径可加载性：无 reject / 无 missing（丢弃族已登记 ⇒ 不算阻塞）。"""
        return not self.reject and not self.missing


def classify_keys(ckpt_keys: Iterable[str], model_keys: Iterable[str]) -> MigrationPlan:
    """ckpt 键集 × 模型键集 → :class:`MigrationPlan`（纯函数）。

    - ``load``：两侧都有（顺序按 ckpt）；
    - ``drop``：ckpt 有、模型没有、且属于已登记族（逐族计数）；
    - ``reject``：ckpt 有、模型没有、且**未登记**（默认口径仍 fail-fast）；
    - ``missing``：模型有、ckpt 没有（B 组不自动 init ⇒ 交给调用方）。
    """
    ckpt = tuple(str(key) for key in ckpt_keys)
    model = tuple(str(key) for key in model_keys)
    model_set = set(model)
    ckpt_set = set(ckpt)
    load: list[str] = []
    drop: list[str] = []
    reject: list[str] = []
    dropped_by_family: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for key in ckpt:
        if key in model_set:
            load.append(key)
            continue
        family = family_of(key)
        if family is None:
            reject.append(key)
            reasons[key] = "unregistered_unexpected"
        else:
            drop.append(key)
            dropped_by_family[family] = dropped_by_family.get(family, 0) + 1
            reasons[key] = f"registered_legacy:{family}"
    rename: dict[str, str] = {}
    init: list[str] = []
    for key, target in RENAME_MAP.items():
        if key in ckpt_set and target in model_set:
            rename[key] = target
            reasons[key] = f"rename:{target}"
    for key in model:
        if key in ckpt_set:
            continue
        for prefix, meta in INIT_FROM_CALIBRATION.items():
            if key.startswith(prefix):
                init.append(key)
                reasons[key] = f"init_from_calibration:{meta.get('source', '?')}"
                break
        else:
            reasons.setdefault(key, "missing")
    init_set = set(init)
    missing = tuple(key for key in model if key not in ckpt_set and key not in init_set)
    return MigrationPlan(
        load=tuple(load),
        drop=tuple(drop),
        reject=tuple(reject),
        missing=missing,
        rename=rename,
        init=tuple(init),
        dropped_by_family=dict(sorted(dropped_by_family.items())),
        reasons=reasons,
    )


def summarize(plan: MigrationPlan) -> dict[str, Any]:
    """分类结果 → 可落盘摘要（逐族计数；契约 §4.2 要求）。"""
    return {
        "n_load": len(plan.load),
        "n_drop": len(plan.drop),
        "n_reject": len(plan.reject),
        "n_missing": len(plan.missing),
        "n_init": len(plan.init),
        "dropped_by_family": dict(plan.dropped_by_family),
        "reject_keys": list(plan.reject[:20]),
        "ok_without_escape": plan.ok_without_escape,
    }


__all__ = [
    "IGNORE_PREFIXES",
    "RENAME_MAP",
    "INIT_FROM_CALIBRATION",
    "MigrationPlan",
    "family_of",
    "classify_keys",
    "summarize",
]
