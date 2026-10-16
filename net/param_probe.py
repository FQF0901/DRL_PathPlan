#!/usr/bin/env python3
"""参数账本（R39-A2 起）：双配置五栏（total / reachable / trainable / frozen / dormant）+ 上限断言 + optimizer allowlist。

用法::

    tools/venv-python net/param_probe.py [--json-out PATH] [--skip-update-norm-demo] [--skip-smoke]

口径（GPT R39 §二-A 7 + §5.83 裁决 + GPT R40 §五.5 参数门）
----------------------------------------------------------
两配置（均 ``hidden=128``、``expert_hidden=76`` = ``config/model.yaml``，全量前向
``world_model_teacher_forcing``）：

- ``default``：``DrivingModel(hidden=128)``（结构化头**惰性构造**，此装配下**不存在**）；
- ``structured``：``set_path_head(True)`` + ``set_ld_mode("semantic")``（原生语义装配）。

键面 → 模块块（R38-P2b 收编口径）：``policy.structured_action.*`` = 结构化头；
``policy.*`` 其余 = 旧 PolicyHead；``plan_head.moe.*``；``plan_head.*``（fusion/norm/ego_next）；
``st_gnn.*``；``encoders.*``；``mem_encoder.*``；``value.*``；``semantic_wiring.*``
（含装配参数 ``beta_context``/``beta_token``/``target_proj.*``）。未知前缀 ⇒ ``ValueError``
（新模块不得绕过账本）。

五栏（GPT R40 §五.5「不能只报总参数」；每个参数恰好归一类）：

- ``dormant`` = **当前模式 forward 不可达**：关闭 MoE 专家（R39 运行口径：MoE 不训练/不启动）
  + 当前模式死路径（structured 模式不跑旧 PolicyHead；旧头模式不跑结构化头）；
- ``trainable``（= ``active``；同义别名）= 可达 ∧ 可训练（``requires_grad=True``，当前配方下）；
- ``frozen`` = 可达 ∧ 配方冻结（``requires_grad=False``）；
- ``reachable`` = ``trainable + frozen``；
- ``total`` = ``trainable + frozen + dormant``（全参数）。

可达性判定 = **实测模块 forward hook**（哪个模块在规范前向里被调用）∧ 声明死路径规则；
两者不一致 ⇒ ``RuntimeError`` fail-fast（账本与运行行为脱钩即失效）。声明死路径：
MoE 关闭时 ``plan_head.moe`` 整块记 dormant（router 仍为 TB 诊断产出 ``router_logits``，
但不参与任何主输出）。

断言（GPT 裁决；基线 = R39-A2 实测 HEAD；R41-D 处置）：

- ``total ≤ 2,000,000``（GPT R40 §五.5 绝对上限；R39「基线 ×1.02」乘性规则废止（正式语义已在
  R41-D 移除，历史常量见 ``tools/legacy/budget_constants.py``）。五栏口径
  不变：``total / reachable / trainable(=active) / frozen / dormant``——``trainable`` 即
  配方可训参数，``active`` 为同义别名）；
- ``budget_active = 可达参数（= active + frozen，MoE 关）`` 对 ``基线 × 1.05``（default
  871,478 / structured 994,717；与裁决 §一 "激活 ≈0.995M–1.103M" 同口径）：
  **R41-D 起仅为警告**（``warnings.active_budget_over``，五栏照报），**不作断言硬门**
  （若需恢复硬门须单独获批）；
- ``optimizer ⊆ 声明 allowlist``（structured = ``net.path_head.TRAIN_ALLOWLIST_PREFIXES``；
  default = 旧头配方占位）且 allowlist 参数全覆盖、全部可达。

另输出 **per-module 参数量** 与 **（训练时）update norm 接口**
（:func:`module_update_norms`；演示 = deepcopy 副本上一次性 AdamW step，不写回、不训练）。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from net.encoders import DEFAULT_OTHERS_DIM
from net.model import DrivingModel
from net.path_head import TRAIN_ALLOWLIST_PREFIXES

# --------------------------------------------------------------------------- #
# 参数门（R40 §五.5：总参数 ≤ 2,000,000 绝对上限；R39 「基线 ×1.02」乘性规则废止）
# --------------------------------------------------------------------------- #
#: 总参数绝对上限（GPT R40 `docs/chatgpt5p6sol_review.md` §五.5：「总参数 <=2,000,000」；
#: 五栏报告 total/reachable/trainable/frozen/dormant，``trainable`` 即配方可训参数）
TOTAL_CAP = 2_000_000
#: 总参数基线 = R39-A2 HEAD 实测（**回归对照/报告**用；R40 起不再参与乘性上限断言）
TOTAL_BASELINES: dict[str, int] = {
    "default": 1_278_879,
    "structured": 1_510_570,
}
#: 激活基线 = 同前向口径下 **可达参数**（MoE 关闭 + 当前模式单头）实测：
#: default = 1,278,879 − 407,401(MoE)；structured = 1,510,570 − 407,401 − 108,452(旧头)。
#: 与 GPT §一「当前激活参数约 0.995M–1.103M」下界同口径（旧头死路径不计入）。
ACTIVE_BASELINES: dict[str, int] = {
    "default": 871_478,
    "structured": 994_717,
}
#: 激活预算**警告阈值**系数（值 = GPT R39 裁决 1.05；**R41-D 起降级为警告**）：
#: `budget_active ≤ 基线 × 1.05` 仅作报告**警告**（``warnings.active_budget_over``），
#: **不作任何断言硬门**——参数预算只保留 ``total ≤ 2,000,000`` 绝对上限（任务书 R41-D：
#: “ACTIVE_LIMIT=1.05 降级为警告；若继续作硬门须单独获批”）。
ACTIVE_LIMIT_WARN = 1.05
#: （兼容别名）历史名 ``ACTIVE_LIMIT``；语义同 :data:`ACTIVE_LIMIT_WARN`（警告阈值，非硬门）。
ACTIVE_LIMIT = ACTIVE_LIMIT_WARN
#: （遗留说明）R39 总参数乘性上限系数 ``TOTAL_LIMIT=1.02`` 的正式语义**已于 R41-D 移除**
#: （R40 §五.5 起即不参与断言）；历史数值快照迁 ``tools/legacy/budget_constants.py``
#: （仅历史脚本 import 兼容；生产代码零引用）。

#: 历史绝对预算常量（H=96 旧口径；H=128 双配置改走 基线+容差 断言，保留供文档/对照指针）
PARAM_BUDGET = 1_500_000
#: 冒烟 batch（纯 CPU，固定种子）
SMOKE_BATCH = 2
#: 规范前向模式（两个配置的可达性判定都跑全量前向）
FORWARD_MODE = "world_model_teacher_forcing"

#: default（旧头）训练配方占位（R39 无 default 训练配方；与 K1 同构：单头 + fusion 末层 +
#: value 供 PPO）。structured 配方 = ``net.path_head.TRAIN_ALLOWLIST_PREFIXES``（代码声明）。
DEFAULT_MODE_TRAIN_ALLOWLIST: tuple[str, ...] = (
    "policy.",
    "value.",
    "plan_head.fusion.2.",
)
ALLOWLISTS_BY_CONFIG: dict[str, tuple[str, ...]] = {
    "default": DEFAULT_MODE_TRAIN_ALLOWLIST,
    "structured": TRAIN_ALLOWLIST_PREFIXES,
}

# --------------------------------------------------------------------------- #
# 键面 → 模块块（P2b 收编；长前缀优先；未知 ⇒ fail-fast）
# --------------------------------------------------------------------------- #
module_of_prefix: tuple[tuple[str, str], ...] = (
    ("policy.structured_action.", "policy.structured_action"),
    ("plan_head.moe.", "plan_head.moe"),
    ("semantic_wiring.", "semantic_wiring"),
    ("target_proj.", "semantic_wiring"),  # R21-25A 装配（beta 门控 + target 投影）
    ("plan_head.", "plan_head"),
    ("st_gnn.", "st_gnn"),
    ("encoders.", "encoders"),
    ("mem_encoder.", "mem_encoder"),
    ("policy.", "policy"),
    ("value.", "value"),
    ("ego_phys.", "ego_phys"),  # R32 canary（惰性构造，本探针两配置不构造）
    ("ld_set_attn.", "ld_set_attn"),  # Round-20 M2（惰性构造）
    ("ld_aggregator.", "ld_aggregator"),  # Round-22 M2a（惰性构造）
)
#: 直接挂在 ``DrivingModel`` 根上的 Parameter（无 owner 模块）→ 模块块
ROOT_PARAMETER_BLOCKS: dict[str, str] = {
    "beta_context": "semantic_wiring",
    "beta_token": "semantic_wiring",
}

#: 块级声明：当前模式下“死路径”（dormant）规则（与实测 hook 交叉验证）
MODE_DEAD_RULES: tuple[tuple[str, str], ...] = (
    ("policy.structured_action", "structured 头未启用（当前模式 = policy；默认关/旧头模式）"),
    ("policy", "structured 模式旧 PolicyHead 永不前向（R38-P2b 单动作出口）"),
    ("plan_head.moe", "MoE 关闭（R39 运行口径：MoE 不训练/不启动；primary/router 仍在位）"),
)

#: 「存活块」最少须有一个被执行的模块（实测背书；模块前缀）
LIVE_BLOCK_SIGNATURES: dict[str, tuple[str, ...]] = {
    "encoders": ("encoders.",),
    "mem_encoder": ("mem_encoder.",),
    "plan_head": ("plan_head.fusion.",),
    "plan_head.moe": ("plan_head.moe.primary.",),
    "st_gnn": ("st_gnn.spatial.",),
    "policy": ("policy.layers.",),
    "policy.structured_action": ("policy.structured_action.",),
    "value": ("value.net.",),
    "semantic_wiring": ("semantic_wiring.",),
}

#: 「死块」实测签名（这些模块前缀**不得**被调用；否则声明与运行脱钩）
DEAD_BLOCK_SIGNATURES: dict[str, tuple[str, ...]] = {
    "policy.structured_action": ("policy.structured_action.",),
    "policy": ("policy.layers.", "policy.trunk.", "policy.mu.", "policy.logstd.", "policy.query"),
    "plan_head.moe": ("plan_head.moe.experts.",),
}


def count_parameters(module: torch.nn.Module) -> int:
    """统计一个模块（含子模块）的可训练参数个数（兼容旧接口）。"""
    return sum(param.numel() for param in module.parameters())


def module_of(name: str) -> str:
    """参数全名 → 模块块（P2b 键面；未知前缀 ``ValueError`` fail-fast）。"""
    if name in ROOT_PARAMETER_BLOCKS:
        return ROOT_PARAMETER_BLOCKS[name]
    for prefix, block in module_of_prefix:
        if name.startswith(prefix):
            return block
    raise ValueError(
        f"参数 {name!r} 不在已知键面（P2b 收编模块块）——新模块须先登记到 "
        "net/param_probe.py::module_of_prefix（账本不得静默漏账）"
    )


def _owner_module(name: str) -> str:
    """参数的直接 owner 模块路径（``a.b.weight`` → ``a.b``；根参数 → ``""``）。"""
    return name.rsplit(".", 1)[0] if "." in name else ""


# --------------------------------------------------------------------------- #
# 观测构造（与 env schema v4/v7 同形状；新增 structured/semantic/world 键）
# --------------------------------------------------------------------------- #
def make_dummy_obs(
    batch: int = SMOKE_BATCH,
    seed: int = 0,
    others_dim: int = DEFAULT_OTHERS_DIM,
    *,
    structured: bool = False,
    semantic: bool = False,
    world: bool = False,
) -> dict[str, torch.Tensor]:
    """构造与 env schema / trainer 组装后同形状的假观测（形状冒烟 + 账本前向）。

    ``structured`` 追加 ``lane_anchor_xy`` + 纵向外源严格列（dynamic_distance/ttc）；
    ``semantic`` 追加 schema v7 语义键（lane_semantic + LD companions）；
    ``world`` 追加 A4 世界系键（ego_world/route_world/route_world_mask，直线路线）。
    """
    generator = torch.Generator().manual_seed(seed)
    frames, od_slots, ld_slots = 6, 16, 16

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    od_mask = (rand(batch, frames, od_slots) > -0.3).float()
    ld_mask = (rand(batch, frames, ld_slots) > -0.3).float()
    presence = od_mask * (rand(batch, frames, od_slots) > -0.3).float()
    hist_valid = torch.zeros(batch, frames)
    hist_valid[:, frames - 3 :] = 1.0  # 预热 3 帧（前 3 帧为补位）
    od_hist = rand(batch, frames, od_slots, 9) * od_mask.unsqueeze(-1)
    ld_hist = rand(batch, frames, ld_slots, 7) * ld_mask.unsqueeze(-1)
    ego_hist = rand(batch, frames, 1, 8)
    others_hist = rand(batch, frames, 1, others_dim)
    od_hist[:, : frames - 3] = od_hist[:, frames - 3 : frames - 2]  # 复制最旧真实帧
    ld_hist[:, : frames - 3] = ld_hist[:, frames - 3 : frames - 2]
    obs: dict[str, torch.Tensor] = {
        "ego_hist": ego_hist,
        "ego_hist_mask": torch.ones(batch, frames, 1),
        "od_hist": od_hist,
        "od_hist_mask": od_mask,
        "od_id_hist": (rand(batch, frames, od_slots).abs() * 10).long() + 1,
        "od_presence_hist": presence,
        "ld_hist": ld_hist,
        "ld_hist_mask": ld_mask,
        "others_hist": others_hist,
        "others_hist_mask": torch.ones(batch, frames, 1),
        "hist_valid": hist_valid,
        "ego": rand(batch, 1, 8),
        "od": od_hist[:, -1],
        "od_mask": od_mask[:, -1],
        "od_id": (rand(batch, od_slots).abs() * 10).long() + 1,
        "od_presence": presence[:, -1],
        "ld": ld_hist[:, -1],
        "ld_mask": ld_mask[:, -1],
        "others": rand(batch, 1, others_dim),
        "others_mask": torch.ones(batch, 1, 1),
        "nav": rand(batch, 1, 11),
        "nav_mask": torch.ones(batch, 1, 1),
        "signal": rand(batch, 1, 4),
        "signal_mask": torch.ones(batch, 1, 1),
    }
    if structured:
        s = torch.linspace(0.0, 40.0, 41)
        obs["lane_anchor_xy"] = (
            torch.stack([s, torch.zeros_like(s)], dim=-1).unsqueeze(0).expand(batch, -1, -1).contiguous()
        )
        obs["lane_anchor_mask"] = torch.ones(batch, 41)
        obs["path_dynamic_distance"] = torch.full((batch,), 30.0)
        obs["path_dynamic_ttc"] = torch.full((batch,), 8.0)
    if semantic:
        dim = 12
        lane_semantic = rand(batch, 1, dim)
        lane_semantic[..., 6] = 1.0  # target_lane_valid
        lane_semantic[..., 7] = 0.8  # route confidence
        lane_semantic[..., 8:10] = 0.0
        role = torch.zeros(batch, ld_slots, dtype=torch.long)
        role[:, :5] = 1  # target
        role[:, 5:7] = 2  # continuation
        role[:, 7] = 3
        role[:, 8] = 4
        obs.update(
            {
                "lane_semantic": lane_semantic,
                "lane_semantic_mask": torch.ones(batch, 1),
                "ld_lane_role": role,
                "ld_offset_s_norm": torch.rand(batch, ld_slots, generator=generator),
                "ld_route_associated": torch.ones(batch, ld_slots),
                "ld_group": torch.zeros(batch, ld_slots, dtype=torch.long),
            }
        )
    if world:
        points = 40
        route = torch.zeros(batch, points, 2)
        route[:, :, 0] = torch.linspace(0.0, 120.0, points)  # 直线路线（沿 ego x 轴）
        obs["ego_world"] = torch.zeros(batch, 3)
        obs["route_world"] = route
        obs["route_world_mask"] = torch.ones(batch, points)
    return obs


# --------------------------------------------------------------------------- #
# 模型装配（两配置）
# --------------------------------------------------------------------------- #
def make_obs_for(config: str, batch: int = SMOKE_BATCH, seed: int = 0) -> dict[str, torch.Tensor]:
    """按配置构造规范观测（structured = 头 + 语义 + 世界系键）。"""
    structured = config == "structured"
    return make_dummy_obs(
        batch, seed, structured=structured, semantic=structured, world=True
    )


def build_model(config: str) -> DrivingModel:
    """按配置装配模型（H=128 / expert_hidden=76 = config/model.yaml；seed 0）。"""
    torch.manual_seed(0)
    model = DrivingModel(hidden=128, expert_hidden=76)
    if config == "structured":
        model.set_path_head(True)
        model.set_ld_mode("semantic")
    elif config != "default":
        raise ValueError(f"未知配置 {config!r}（default | structured）")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# 可达性（实测 hook + 声明死路径规则）
# --------------------------------------------------------------------------- #
def forward_reachable(model: DrivingModel, obs: Mapping[str, torch.Tensor], *, moe_enabled: bool) -> dict[str, Any]:
    """规范前向实测：哪些模块被执行；与声明死路径规则交叉验证（不一致 ⇒ RuntimeError）。

    默认关/只读：不采集张量、不改参数（``set_moe`` 为运行模式开关，不消耗 RNG）。
    """
    model.set_moe(enabled=bool(moe_enabled))
    executed: set[str] = set()

    def _mark(name: str, module: torch.nn.Module, args: Any, output: Any) -> None:
        executed.add(name)

    handles = [
        module.register_forward_hook(partial(_mark, name))
        for name, module in model.named_modules()
        if name
    ]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(obs, forward_mode=FORWARD_MODE)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    mode = model.path_head_mode()
    dead: dict[str, str] = {}
    for block, reason in MODE_DEAD_RULES:
        if block == "policy.structured_action" and mode == "structured":
            continue
        if block == "policy" and mode != "structured":
            continue
        if block == "plan_head.moe" and bool(model.plan_head.moe.moe_enabled):
            continue
        dead[block] = reason

    # --- 交叉验证（声明 vs 实测；脱钩即账本失效） ---
    consistency: list[str] = []
    for block, signatures in DEAD_BLOCK_SIGNATURES.items():
        if block not in dead:
            continue
        leaked = sorted(name for name in executed if any(name.startswith(sig) for sig in signatures))
        if leaked:
            raise RuntimeError(
                f"[param_ledger] 声明死块 {block!r} 但模块被执行（{leaked[:4]}…）——"
                "账本与运行行为脱钩，拒绝出账"
            )
        consistency.append(f"dead:{block}（实测未执行）")
    present_blocks = _blocks_in_model(model)
    for block, signatures in LIVE_BLOCK_SIGNATURES.items():
        if block in dead or block not in present_blocks:
            continue
        matched = [name for name in executed if any(name.startswith(sig) for sig in signatures)]
        if not matched:
            raise RuntimeError(
                f"[param_ledger] 声明存活块 {block!r} 但签名模块未被调用——"
                "账本与运行行为脱钩，拒绝出账"
            )
        consistency.append(f"live:{block}（实测执行）")
    return {
        "moe_enabled": bool(moe_enabled),
        "mode": mode,
        "executed_modules": sorted(executed),
        "dead_blocks": dead,
        "consistency": consistency,
    }


def _blocks_in_model(model: torch.nn.Module) -> set[str]:
    """模型中实际出现的模块块集合（按键面归属）。"""
    return {module_of(name) for name, _ in model.named_parameters()}


def _param_reachable(model: DrivingModel, reach: dict[str, Any], name: str) -> bool:
    """单参数可达：声明死块 ⇒ False；否则 owner 模块（或任一祖先）须被实测执行。

    祖先口径的由来：``nn.MultiheadAttention`` 快路径内部走函数式调用，其 ``out_proj``
    子模块的 forward hook 不触发 ⇒ 以"祖先模块被执行"判定（父注意力块执行即视为其内部
    Linear 参与计算）。声明死块规则优先（结构化头/旧头按模式整体判死，不受祖先影响）。
    根级参数（``beta_*``）= semantic 装配（登记制，非模块 owner）。
    """
    block = module_of(name)
    if block in reach["dead_blocks"]:
        return False
    owner = _owner_module(name)
    if owner == "":
        if name not in ROOT_PARAMETER_BLOCKS:
            raise ValueError(f"根级参数 {name!r} 未登记（账本不得漏账）")
        return block not in reach["dead_blocks"]
    executed = set(reach["executed_modules"])
    parts = owner.split(".")
    for cut in range(len(parts), 0, -1):
        if ".".join(parts[:cut]) in executed:
            return True
    return False


def classify_parameters(
    model: DrivingModel,
    reach: dict[str, Any],
    *,
    recipe_allowlist: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """逐参数记录：模块块 / 参数数 / 可达 / 可训练（配方）/ 四分类归属。

    ``recipe_allowlist=None`` ⇒ 用当前 ``requires_grad`` 状态；给出 allowlist ⇒
    按 allowlist 前缀判定（与 ``pipeline.trainer.apply_trainable_allowlist`` 同口径，
    不修改参数）。
    """
    records: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        reachable = _param_reachable(model, reach, name)
        if recipe_allowlist is None:
            trainable = bool(parameter.requires_grad)
        else:
            trainable = name.startswith(tuple(recipe_allowlist))
        if not reachable:
            klass = "dormant"
        else:
            klass = "active" if trainable else "frozen"
        records.append(
            {
                "name": name,
                "module": module_of(name),
                "numel": int(parameter.numel()),
                "reachable": reachable,
                "trainable": bool(trainable),
                "class": klass,
            }
        )
    return records


def apply_recipe(model: DrivingModel, allowlist: Sequence[str]) -> tuple[str, ...]:
    """按 allowlist 设置 ``requires_grad``（只保留前缀内可训练）；返回冻结参数名。

    与 ``pipeline.trainer.apply_trainable_allowlist`` 同语义；param_probe 内联实现以免
    import 训练栈（pipeline/trainer 依赖重）。
    """
    frozen: list[str] = []
    prefixes = tuple(str(prefix) for prefix in allowlist)
    for name, parameter in model.named_parameters():
        trainable = name.startswith(prefixes)
        parameter.requires_grad_(trainable)
        if not trainable:
            frozen.append(name)
    return tuple(frozen)


def view_totals(records: Sequence[Mapping[str, Any]], total: int) -> dict[str, Any]:
    """五栏汇总 + 模块聚合（total/reachable/trainable/frozen/dormant + 可达）。

    ``trainable`` = ``active``（GPT R40 §五.5 栏名；同义别名，二者恒等）。
    """
    counts = {"active": 0, "frozen": 0, "dormant": 0}
    reachable = 0
    for record in records:
        counts[record["class"]] += int(record["numel"])
        if record["reachable"]:
            reachable += int(record["numel"])
    assert counts["active"] + counts["frozen"] + counts["dormant"] == total, "五栏不闭合"
    assert counts["active"] + counts["frozen"] == reachable, "trainable+frozen != 可达"
    return {
        "total": int(total),
        "active": counts["active"],
        "trainable": counts["active"],  # GPT R40 §五.5 栏名（= active 同义）
        "frozen": counts["frozen"],
        "dormant": counts["dormant"],
        "reachable": reachable,  # = trainable + frozen
        "budget_active": reachable,  # GPT §5.83 「激活」= 可达参数（含配方冻结）
    }


def per_module_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """per-module 参数与四分类（按模块块聚合；键面顺序）。"""
    order = [block for _, block in module_of_prefix] + ["semantic_wiring"]
    seen: dict[str, dict[str, Any]] = {}
    for record in records:
        entry = seen.setdefault(
            record["module"],
            {"module": record["module"], "params": 0, "active": 0, "frozen": 0, "dormant": 0, "reachable": 0},
        )
        entry["params"] += int(record["numel"])
        entry[record["class"]] += int(record["numel"])
        if record["reachable"]:
            entry["reachable"] += int(record["numel"])
    ordered: list[dict[str, Any]] = []
    for block in order:
        if block in seen:
            ordered.append(seen.pop(block))
    ordered.extend(seen.values())
    return ordered


# --------------------------------------------------------------------------- #
# optimizer allowlist 检查 / update norm 接口
# --------------------------------------------------------------------------- #
def module_update_norms(update_norms: Mapping[str, float]) -> dict[str, float]:
    """参数级 update norm 字典 → **per-module L2 聚合**（训练时接口）。

    ``update_norms`` = {参数全名: ‖Δθ‖₂}（如 ``D1 grad_ledger['update']['update_norms']``）；
    返回 {模块块: √Σ‖Δθ‖²}（块级 = 参数级范数的平方和开方）。
    """
    squared: dict[str, float] = {}
    for name, value in update_norms.items():
        block = module_of(name)
        squared[block] = squared.get(block, 0.0) + float(value) ** 2
    return {block: math.sqrt(value) for block, value in squared.items()}


def optimizer_allowlist_report(
    model: DrivingModel,
    allowlist: Sequence[str],
    *,
    reach: dict[str, Any] | None = None,
    lr: float = 1e-4,
) -> dict[str, Any]:
    """按 allowlist 构建 optimizer 并校验"optimizer 仅含声明模块"。

    - ``violations``：优化器内但不在 allowlist 的参数（必须为空）；
    - ``missing``：allowlist 内但不在优化器的参数（必须为空 = 无静默丢训练参数）；
    - ``dormant_in_optimizer``：allowlist 内但按当前模式不可达的参数（给 ``reach`` 时校验，
      必须为空 = 不允许给死路径建优化器组）。
    """
    prefixes = tuple(str(prefix) for prefix in allowlist)
    declared = [(name, parameter) for name, parameter in model.named_parameters() if name.startswith(prefixes)]
    optimizer = torch.optim.AdamW([parameter for _, parameter in declared], lr=lr)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    violations = [
        name
        for name, parameter in model.named_parameters()
        if id(parameter) in optimizer_ids and not name.startswith(prefixes)
    ]
    missing = [name for name, parameter in declared if id(parameter) not in optimizer_ids]
    dormant_in_optimizer: list[str] = []
    if reach is not None:
        for name, _ in declared:
            if not _param_reachable(model, reach, name):
                dormant_in_optimizer.append(name)
    return {
        "allowlist": list(prefixes),
        "n_optimizer_params": len(optimizer_ids),
        "numel": int(sum(int(parameter.numel()) for _, parameter in declared)),
        "violations": violations,
        "missing": missing,
        "dormant_in_optimizer": dormant_in_optimizer,
        "ok": not violations and not missing and not dormant_in_optimizer,
    }


def update_norm_demo(
    model: DrivingModel,
    obs: Mapping[str, torch.Tensor],
    allowlist: Sequence[str],
    *,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """（训练时）update norm 接口演示：**deepcopy 副本**上一次性 AdamW step（不写回、不训练）。

    损失 = 功能性输出 ``action_mu/value/latent/plan/traj_xy`` 的 ``Σ mean(·²)``；只对
    allowlist 参数建组（optimizer 隔离 = allowlist 外 update 恒 0，登记为断言）。
    返回参数级/模块级 update norm 与 ckpt 不变性读数。
    """
    probe = copy.deepcopy(model)
    params = [(name, parameter) for name, parameter in probe.named_parameters()]
    prefixes = tuple(str(prefix) for prefix in allowlist)
    declared = [(name, parameter) for name, parameter in params if name.startswith(prefixes)]
    optimizer = torch.optim.AdamW([parameter for _, parameter in declared], lr=lr)
    optimizer.zero_grad(set_to_none=True)
    out = probe(obs, forward_mode=FORWARD_MODE)
    loss = None
    for key in ("action_mu", "value", "latent", "plan", "traj_xy"):
        value = out[key]
        term = value.pow(2).mean()
        loss = term if loss is None else loss + term
    assert loss is not None
    loss.backward()
    grad_norm_total = math.sqrt(
        sum(
            float(parameter.grad.pow(2).sum())
            for _, parameter in params
            if parameter.grad is not None
        )
    )
    with torch.no_grad():
        before = {name: parameter.clone() for name, parameter in params}
    optimizer.step()
    update_norms: dict[str, float] = {}
    with torch.no_grad():
        for name, parameter in params:
            delta = float((parameter - before[name]).norm())
            if delta > 0.0:
                update_norms[name] = delta
    outside = sorted(name for name in update_norms if not name.startswith(prefixes))
    total = math.sqrt(sum(value**2 for value in update_norms.values()))
    return {
        "loss": float(loss),
        "grad_norm_total": grad_norm_total,
        "update_norm_total": total,
        "update_norms": update_norms,
        "by_module": module_update_norms(update_norms),
        "outside_allowlist_updates": outside,
        "optimizer_isolated": not outside,
        "n_updated_params": len(update_norms),
    }


# --------------------------------------------------------------------------- #
# 账本主流程
# --------------------------------------------------------------------------- #
def ledger_for_config(config: str, *, demo_update_norm: bool = True) -> dict[str, Any]:
    """一个配置的完整参数账本（两 MoE 视图 + 配方视图 + optimizer + update norm 接口）。"""
    if config not in TOTAL_BASELINES:
        raise ValueError(f"未知配置 {config!r}（default | structured）")
    model = build_model(config)
    obs = make_obs_for(config)
    total = count_parameters(model)
    allowlist = ALLOWLISTS_BY_CONFIG[config]

    reach_closed = forward_reachable(model, obs, moe_enabled=False)
    records_closed = classify_parameters(model, reach_closed)
    budget_closed = view_totals(records_closed, total)
    table_closed = per_module_table(records_closed)

    reach_open = forward_reachable(model, obs, moe_enabled=True)
    records_open = classify_parameters(model, reach_open)
    budget_open = view_totals(records_open, total)

    # 配方视图（按 allowlist 冻结；不修改模型，仅按前缀判定）
    records_recipe = classify_parameters(model, reach_closed, recipe_allowlist=allowlist)
    recipe_closed = view_totals(records_recipe, total)
    table_recipe = per_module_table(records_recipe)

    frozen_applied = apply_recipe(model, allowlist)
    optimizer = optimizer_allowlist_report(model, allowlist, reach=reach_closed)
    demo = update_norm_demo(model, obs, allowlist) if demo_update_norm else None

    assertion = {
        "total_limit": total <= TOTAL_CAP,
        "optimizer_ok": bool(optimizer["ok"]),
    }
    active_warn_threshold = math.floor(ACTIVE_BASELINES[config] * ACTIVE_LIMIT_WARN)
    warnings = {
        "active_budget_over": bool(budget_closed["budget_active"] > active_warn_threshold),
        "active_budget": int(budget_closed["budget_active"]),
        "active_warn_threshold": int(active_warn_threshold),
    }
    return {
        "config": config,
        "label": {"default": "default(H=128)", "structured": "structured（set_path_head + semantic）"}[config],
        "forward_mode": FORWARD_MODE,
        "total": total,
        "allowlist": list(allowlist),
        "baselines": {"total": TOTAL_BASELINES[config], "total_cap": TOTAL_CAP, "active": ACTIVE_BASELINES[config]},
        "views": {
            "budget_moe_closed": budget_closed,
            "budget_moe_open": budget_open,
            "recipe_moe_closed": recipe_closed,
        },
        "modules": {
            "budget_moe_closed": table_closed,
            "recipe_moe_closed": table_recipe,
        },
        "reach": {
            "moe_closed": {k: v for k, v in reach_closed.items() if k != "executed_modules"},
            "moe_open": {k: v for k, v in reach_open.items() if k != "executed_modules"},
            "executed_modules_moe_closed": reach_closed["executed_modules"],
        },
        "frozen_by_recipe": len(frozen_applied),
        "optimizer": optimizer,
        "update_norm_demo": demo,
        "assertion": assertion,
        "warnings": warnings,
        "assertion_detail": {
            "total_rule": f"total ≤ {TOTAL_CAP:,}（GPT R40 §五.5 绝对上限；R39 乘性规则正式语义"
                          "已移除，历史常量见 tools/legacy/budget_constants.py）",
            "active_warn_rule": (
                f"（警告，非硬门；R41-D 降级）budget_active {budget_closed['budget_active']:,} "
                f"≤ 基线 {ACTIVE_BASELINES[config]:,} × {ACTIVE_LIMIT_WARN} = {active_warn_threshold:,}"
            ),
        },
    }


def render_report(ledgers: Mapping[str, Mapping[str, Any]]) -> str:
    """人类可读摘要（五栏 + 断言 + per-module + optimizer + update norm）。"""
    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)

    emit("== R39-A2 参数账本（双配置五栏；MoE 关闭 = 运行口径；total 门 = GPT R40 §五.5 ≤2,000,000）==")
    emit(
        f"{'配置':<12}{'total':>11}{'reachable':>11}{'trainable':>11}{'frozen':>11}{'dormant':>11}"
    )
    for config, ledger in ledgers.items():
        view = ledger["views"]["recipe_moe_closed"]
        emit(
            f"{config:<12}{view['total']:>11,}{view['reachable']:>11,}{view['trainable']:>11,}"
            f"{view['frozen']:>11,}{view['dormant']:>11,}"
        )
    emit("（trainable = active 同义；reachable = trainable + frozen）")
    emit()
    emit("-- 预算视图（无配方；as-assembled；MoE 关/开）--")
    for config, ledger in ledgers.items():
        closed = ledger["views"]["budget_moe_closed"]
        opened = ledger["views"]["budget_moe_open"]
        emit(
            f"  {config:<12} MoE关: total {closed['total']:,} / 可达 {closed['reachable']:,} "
            f"/ dormant {closed['dormant']:,}    MoE开: 可达 {opened['reachable']:,} / dormant {opened['dormant']:,}"
        )
    emit()
    for config, ledger in ledgers.items():
        emit(f"-- per-module [{config}]（预算视图，MoE 关）--")
        for row in ledger["modules"]["budget_moe_closed"]:
            emit(
                f"  {row['module']:<28}{row['params']:>10,}  可达 {row['reachable']:>10,}"
                f"  dormant {row['dormant']:>9,}"
            )
        opt = ledger["optimizer"]
        emit(
            f"  optimizer: allowlist={opt['allowlist']} 组内参数 {opt['n_optimizer_params']} "
            f"({opt['numel']:,} numel)  violations={opt['violations']} missing={opt['missing']} "
            f"dormant={opt['dormant_in_optimizer']} ⇒ {'OK' if opt['ok'] else 'FAIL'}"
        )
        demo = ledger["update_norm_demo"]
        if demo is not None:
            by_module = ", ".join(f"{block}={value:.4g}" for block, value in sorted(demo["by_module"].items()))
            emit(
                f"  update_norm（副本一次 AdamW；loss={demo['loss']:.4g}）：total={demo['update_norm_total']:.4g} "
                f"by_module[{by_module}]  隔离={'OK' if demo['optimizer_isolated'] else 'FAIL'}"
            )
        assertion = ledger["assertion"]
        warnings = ledger["warnings"]
        emit(
            f"  断言: total(cap {TOTAL_CAP:,}) {assertion['total_limit']} · "
            f"optimizer {assertion['optimizer_ok']}"
        )
        emit(
            f"  警告（非硬门；R41-D）: active {warnings['active_budget']:,} "
            f"{'超阈' if warnings['active_budget_over'] else '≤阈'} "
            f"（阈值 {warnings['active_warn_threshold']:,} = 基线 × {ACTIVE_LIMIT_WARN}）"
        )
        emit()
    return "\n".join(lines)


def shape_smoke(configs: Sequence[str]) -> dict[str, dict[str, Any]]:
    """两配置形状/契约冒烟（forward + rollout 键形状）。"""
    report: dict[str, dict[str, Any]] = {}
    for config in configs:
        model = build_model(config)
        obs = make_obs_for(config)
        with torch.no_grad():
            out = model(obs, forward_mode=FORWARD_MODE)
            rolled = model.rollout(obs)
        assert tuple(out["traj_xy"].shape) == (SMOKE_BATCH, 6, 2)
        assert tuple(out["od_presence_pred"].shape) == (SMOKE_BATCH, 6, 16)
        assert tuple(out["od_entry_pred"].shape) == (SMOKE_BATCH, 6, 16)
        keys = ["action_mu", "value", "traj_xy", "plan", "latent"]
        if config == "structured":
            keys += ("structured_ds", "structured_kappa_near", "contract_runtime_ok")
            assert float(out["contract_runtime_ok"]) == 1.0
            assert "action_logstd" not in out  # structured：旧头不前向（无 logstd 出口）
        else:
            keys.append("action_logstd")  # 默认关路径：旧头出口
        report[config] = {
            key: {"forward": list(out[key].shape), "rollout": list(rolled[key].shape)}
            if torch.is_tensor(out[key])
            else out[key]
            for key in keys
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R39-A2 参数账本（双配置四分类 + 上限断言）")
    parser.add_argument("--json-out", type=Path, default=None, help="将完整账本写为 JSON")
    parser.add_argument("--skip-update-norm-demo", action="store_true", help="跳过副本一次性 AdamW 演示")
    parser.add_argument("--skip-smoke", action="store_true", help="跳过形状冒烟")
    args = parser.parse_args(argv)

    configs = ("default", "structured")
    ledgers = {
        config: ledger_for_config(config, demo_update_norm=not args.skip_update_norm_demo)
        for config in configs
    }
    print(render_report(ledgers))
    failures = [
        f"{config}:{name}"
        for config, ledger in ledgers.items()
        for name, ok in ledger["assertion"].items()
        if not ok
    ]
    if failures:
        raise AssertionError(f"R39-A2 参数账本断言失败：{failures}")
    smoke = None
    if not args.skip_smoke:
        smoke = shape_smoke(configs)
        print("== 形状冒烟 ==")
        for config, rows in smoke.items():
            print(f"  [{config}] " + "; ".join(f"{key}{value['forward']}" for key, value in rows.items()))
    summary = {
        config: {
            "total": ledger["total"],
            "total_cap": TOTAL_CAP,
            "recipe": ledger["views"]["recipe_moe_closed"],
            "budget_moe_closed": ledger["views"]["budget_moe_closed"],
            "budget_moe_open": ledger["views"]["budget_moe_open"],
            "assertion": ledger["assertion"],
            "warnings": ledger["warnings"],
        }
        for config, ledger in ledgers.items()
    }
    # R41-D：硬门只剩 total（≤2,000,000）；active×1.05 = 警告（非硬门）
    print(
        "OK: 参数预算门通过（total 门 = GPT R40 §五.5 ≤ 2,000,000；active×1.05 = 警告非硬门）—— "
        + " ｜ ".join(
            f"{config} total {ledger['total']:,}（上限 {TOTAL_CAP:,}）/ "
            f"reachable {ledger['views']['budget_moe_closed']['reachable']:,} / "
            f"trainable {ledger['views']['recipe_moe_closed']['trainable']:,} / "
            f"frozen {ledger['views']['recipe_moe_closed']['frozen']:,} / "
            f"dormant {ledger['views']['recipe_moe_closed']['dormant']:,}"
            + ("（active 警告：超阈！）" if ledger["warnings"]["active_budget_over"] else "")
            for config, ledger in ledgers.items()
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps({"ledgers": ledgers, "smoke": smoke}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON 已写：{args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
