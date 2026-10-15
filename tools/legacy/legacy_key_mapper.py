"""R38-P2a/P2b：旧 ckpt 键面映射（纯函数；无 torch、无 IO、无副作用）。

出处：``docs/repo_handbook.md`` 附录 D §4.2（丢弃族常量）/ §4.4（改名族）/ §5（接口）。
范围（R38-P2a）：**B 组 K-anchor** 的 ignore 集 —— ``plan_head.anchor*``（16 键：``anchors``/
``anchor_ds``/``anchor_dth``/``anchor_embed``/``anchor_head.*``/``residual_head.*``/``speed_head.*``）
+ ``encoders.lane.*``/``encoders.ttc.*``（4 键；v8 已删通道）= 20 键（manifest B 组第 10 项）。
范围（R38-P2b，D 组双实体收编）：``path_head.*``（18 键：结构化头 4 子模块 + 2 几何编码器）
→ ``policy.structured_action.*``（embedding/ownership fold；GPT §七-D + argue §5.63
"命名/归属迁移 + ckpt 键重映射，默认兼容"）；``policy.log_std → policy.logstd``（契约 §4.4 改名族）；
``meta.path_head_ablation`` / ``meta.path_head_mode``（弃用声明口径，:data:`LEGACY_META_KEYS`）。

纪律（契约 §5 硬约束）：
1. 本模块只做"键面三档分类"（load/rename/drop/reject），**不**决定 forward 语义、不 import ``net``；
2. 默认 fail-fast 不变：未登记的 unexpected 键 = ``reject``（装载方必须拒绝，不得静默丢弃）；
   已登记丢弃族 / 改名族从 unexpected 里移走，但**逐族计数**（``dropped_by_family`` /
   ``renamed_by_family``），且改名目标缺席时报 ``rename_target_absent``（条件族，装配决定）；
3. 每次丢弃逐族计数，禁止"总计 N 个 unexpected"一笔带过；
4. ``INIT_FROM_CALIBRATION`` 为契约预留接口（缺键侧标定初始化）；B/D 组范围为空表。
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

#: 改名族（精确键；契约 §4.4：``policy.log_std → policy.logstd``；8 个 ckpt 受影响）。
RENAME_MAP: dict[str, str] = {
    "policy.log_std": "policy.logstd",
}

#: 双实体收编改名族（前缀规则，最长前缀优先；R38-P2b D 组）：
#: 旧结构化头键面 ``path_head.*``（18 键）→ 收编后 ``policy.structured_action.*``。
#: 目标装配含该实体 ⇒ rename（load）；目标装配不含 ⇒ 条件族 drop + ``rename_target_absent``
#: （装配决定，逐族计数，不静默）；两侧都无 ⇒ 仍 reject（未登记）。
PREFIX_RENAME_MAP: tuple[tuple[str, str], ...] = (
    ("path_head.", "policy.structured_action."),
)

#: 旧 ckpt ``meta`` 里的弃用声明键（不进 state_dict；读出后丢弃，仅作装载守卫输入）。
#: 来源：A/D 组 ckpt 键面 ``meta.path_head_ablation`` / ``meta.path_head_mode``。
LEGACY_META_KEYS: tuple[str, ...] = ("path_head_ablation", "path_head_mode")

#: 标定初始化族（契约 §4.1：``path_head.*`` 缺键时来自 ``runs/round36/k1/calibration.json``；
#: B/D 组范围不启用，预留）。
INIT_FROM_CALIBRATION: dict[str, dict[str, str]] = {}


def family_of(key: str) -> str | None:
    """键 → 已登记丢弃族前缀（未登记 → ``None``）。"""
    for prefix in IGNORE_PREFIXES:
        if key.startswith(prefix):
            return prefix
    return None


def prefix_rename_of(key: str) -> tuple[str, str] | None:
    """键 → 命中的双实体改名规则 ``(旧前缀, 新前缀)``（未命中 → ``None``；最长前缀优先）。"""
    matched: tuple[str, str] | None = None
    for source, target in PREFIX_RENAME_MAP:
        if key.startswith(source) and (matched is None or len(source) > len(matched[0])):
            matched = (source, target)
    return matched


def renamed_key(key: str) -> str | None:
    """键 → 收编后目标键（精确改名优先，其次前缀改名；未命中 → ``None``）。"""
    if key in RENAME_MAP:
        return RENAME_MAP[key]
    rule = prefix_rename_of(key)
    if rule is None:
        return None
    source, target = rule
    return target + key[len(source):]


def classify_meta(meta: Mapping[str, Any] | None) -> dict[str, Any]:
    """ckpt ``meta`` → 弃用声明判定（读出后丢弃，不参与 forward；契约 §4.1 末行）。

    返回 ``{"recognized": {key: value}, "oneshot_declared": bool}``。
    ``oneshot_declared`` = 历史一次性六步臂 ckpt（R38-P1 后应 fail-fast 拒绝，除显式诊断）。
    """
    meta = dict(meta or {})
    recognized = {key: meta[key] for key in LEGACY_META_KEYS if key in meta}
    oneshot = str(meta.get("path_head_ablation", "")).lower() == "oneshot"
    return {"recognized": recognized, "oneshot_declared": bool(oneshot)}


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
    renamed_by_family: Mapping[str, int] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok_without_escape(self) -> bool:
        """默认口径可加载性：无 reject / 无 missing（丢弃族已登记 ⇒ 不算阻塞）。"""
        return not self.reject and not self.missing


def classify_keys(ckpt_keys: Iterable[str], model_keys: Iterable[str]) -> MigrationPlan:
    """ckpt 键集 × 模型键集 → :class:`MigrationPlan`（纯函数）。

    - ``load``：两侧都有（顺序按 ckpt）；
    - ``rename``：命中 :data:`RENAME_MAP`（精确）或 :data:`PREFIX_RENAME_MAP`（前缀），且目标键
      存在于模型（顺序按 ckpt）；
    - ``drop``：ckpt 有、模型没有、且属于已登记族（逐族计数；含"改名目标缺席"条件族）；
    - ``reject``：ckpt 有、模型没有、且**未登记**（默认口径仍 fail-fast）；
    - ``missing``：模型有、ckpt 没有（不自动 init ⇒ 交给调用方）。
    """
    ckpt = tuple(str(key) for key in ckpt_keys)
    model = tuple(str(key) for key in model_keys)
    model_set = set(model)
    ckpt_set = set(ckpt)
    load: list[str] = []
    rename: dict[str, str] = {}
    drop: list[str] = []
    reject: list[str] = []
    dropped_by_family: dict[str, int] = {}
    renamed_by_family: dict[str, int] = {}
    reasons: dict[str, str] = {}
    for key in ckpt:
        if key in model_set:
            load.append(key)
            continue
        target = renamed_key(key)
        if target is not None and target in model_set:
            rename[key] = target
            if key in RENAME_MAP:
                family = f"{key}->{target}"
            else:
                rule = prefix_rename_of(key)
                assert rule is not None
                family = f"{rule[0]}->{rule[1]}"
            renamed_by_family[family] = renamed_by_family.get(family, 0) + 1
            reasons[key] = f"rename:{target}"
            continue
        family = family_of(key)
        if family is not None:
            drop.append(key)
            dropped_by_family[family] = dropped_by_family.get(family, 0) + 1
            reasons[key] = f"registered_legacy:{family}"
            continue
        if target is not None:
            # 条件族：前缀改名规则已登记，但目标装配缺席该实体（如默认模型无结构化头）⇒
            # 显式 drop + 逐族计数（不静默）。**精确改名**（RENAME_MAP）目标缺席保持 P2a 口径
            # = ``reject``（默认 fail-fast；形状/存在性未决，不当条件族）。
            if key in RENAME_MAP:
                reject.append(key)
                reasons[key] = f"rename_target_absent:{target}"
                continue
            rule = prefix_rename_of(key)
            assert rule is not None
            family = f"{rule[0]}->{rule[1]}(assembly_missing)"
            drop.append(key)
            dropped_by_family[family] = dropped_by_family.get(family, 0) + 1
            reasons[key] = f"rename_target_absent:{target}"
            continue
        reject.append(key)
        reasons[key] = "unregistered_unexpected"
    init: list[str] = []
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
    rename_targets = set(rename.values())
    missing = tuple(
        key for key in model if key not in ckpt_set and key not in init_set and key not in rename_targets
    )
    return MigrationPlan(
        load=tuple(load),
        drop=tuple(drop),
        reject=tuple(reject),
        missing=missing,
        rename=rename,
        init=tuple(init),
        dropped_by_family=dict(sorted(dropped_by_family.items())),
        renamed_by_family=dict(sorted(renamed_by_family.items())),
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
        "n_rename": len(plan.rename),
        "dropped_by_family": dict(plan.dropped_by_family),
        "renamed_by_family": dict(plan.renamed_by_family),
        "reject_keys": list(plan.reject[:20]),
        "ok_without_escape": plan.ok_without_escape,
    }


__all__ = [
    "IGNORE_PREFIXES",
    "INIT_FROM_CALIBRATION",
    "LEGACY_META_KEYS",
    "PREFIX_RENAME_MAP",
    "RENAME_MAP",
    "MigrationPlan",
    "classify_keys",
    "classify_meta",
    "family_of",
    "prefix_rename_of",
    "renamed_key",
    "summarize",
]

