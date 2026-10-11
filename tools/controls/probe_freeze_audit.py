#!/usr/bin/env python3
"""探针：freeze recipe 审计（P1-0C 批次 3a；§7.10-A4/A9 四层一致）。

配方 → requires_grad → optimizer → 实际更新（逐模块组四计数）：

a) ``requires_grad`` 参数数；b) optimizer 参数数；c) 小批 backward 后 nonzero-grad 计数；
d) ``optimizer.step()`` 后参数变化计数（前后 snapshot 比较）。

- **静态层**：``ast`` 只读解析 ``pipeline/stages.py`` 的冻结前缀清单
  （``_PRIMARY_PHASE_FREEZE`` / ``_SPECIFIC_PHASE_FREEZE`` / ``_PHASE3_ANCHOR_HEAD_PREFIXES``
  / ``_PHASE3_SAFE_FREEZE``），并与运行时导入的常量交叉核对（``static_match_runtime``）；
- **构建层**：``pipeline.stages.load_config`` + ``build_model``（只读调用）→
  ``pipeline.trainer.apply_freeze_prefixes`` → ``build_phase3_optimizer``（与 phase3
  训练完全同一接线）→ 随机张量小批（无需数据集/GPU；CPU 优先）前向 + backward + step；
- **判定**：freeze 组 a=b=c=d=0（且不在 optimizer）；train 组 a>0 且 d>0；否则 FAIL；
- **recovery 配方**：已实现（`_RECOVERY_FREEZE` / `_RECOVERY_EXPERTS_FREEZE`；P1-2）→ 与其它配方同路径四计数审计；`RECOVERY_SUGGESTION` 保留为对照清单。

合成小批 = 随机观测（与 env schema 同形状）+ 前向输出平方损失 + ``1e-3`` × 可训参数平方
正则项（打破专家零初始化导致 router/residual_scale/部分专家梯度退化为 0 的初始化退化；
探针目的 = 冻结/优化器/更新账本，不是损失语义保真）。

用法::

    tools/venv-python tools/controls/probe_freeze_audit.py --recipe specific_only \\
        [--config config/default.yaml] [--device cpu] [--batch-size 4] [--seed 0] \\
        [--out runs/p1_0/probe_freeze_audit.json]

退出码：PASS=0 / FAIL=1 / NOT_IMPLEMENTED=2。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

STAGES_PATH = _ROOT / "pipeline" / "stages.py"
DEFAULT_CONFIG = "config/default.yaml"
DEFAULT_OUT = "runs/p1_0/probe_freeze_audit.json"
_EXIT_BY_STATUS = {"PASS": 0, "FAIL": 1, "NOT_IMPLEMENTED": 2}

#: 逐模块组（固定组序；未命中任何前缀 → ``other``，参数名仍逐项可见于明细）
GROUP_ORDER = (
    "encoder",
    "mem",
    "st_gnn",
    "router",
    "experts",
    "plan_head.primary",
    "fusion",
    "policy",
    "value",
    "anchor",
    "other",
)
#: 组前缀（最长/更具体优先由组内互斥保证；``residual_scale`` 归入 experts——
#: 与 ``pipeline.trainer._PHASE3_SPECIFIC_PREFIXES`` 的 specific 划分同口径）
_GROUP_PREFIXES = (
    ("encoder", ("encoders.",)),
    ("mem", ("mem_encoder.",)),
    ("st_gnn", ("st_gnn.",)),
    ("router", ("plan_head.moe.router.",)),
    ("experts", ("plan_head.moe.experts.", "plan_head.moe.residual_scale")),
    ("plan_head.primary", ("plan_head.moe.primary.",)),
    ("fusion", ("plan_head.fusion.",)),
    ("policy", ("policy.",)),
    ("value", ("value.",)),
    (
        "anchor",
        (
            "plan_head.anchor_head.",
            "plan_head.speed_head.",
            "plan_head.residual_head.",
            "plan_head.anchor_embed",
        ),
    ),
)

#: 静态解析的模块级冻结常量（pipeline/stages.py）
STATIC_CONSTANT_NAMES = (
    "_PRIMARY_PHASE_FREEZE",
    "_SPECIFIC_PHASE_FREEZE",
    "_PHASE3_ANCHOR_HEAD_PREFIXES",
    "_PHASE3_SAFE_FREEZE",
    "_RECOVERY_FREEZE",
    "_RECOVERY_EXPERTS_FREEZE",
)
#: 配方 → 静态常量名（all = 空前缀）
RECIPE_CONSTANT = {
    "specific_only": "_SPECIFIC_PHASE_FREEZE",
    "trunk_only": "_PHASE3_SAFE_FREEZE",
    "recovery": "_RECOVERY_FREEZE",
    "recovery_experts": "_RECOVERY_EXPERTS_FREEZE",
}

#: 合成损失中的可训参数平方正则系数（打破零初始化梯度退化；见模块 docstring）
PARAM_REG_COEF = 1e-3

#: recovery 配方建议清单（尚未实现；§7.10-A6：train = fusion/norm/ego_next/primary（± experts/
#: residual，R2）；freeze = encoder/mem/WM(st_gnn)/router/anchor 头/policy/value；A9：
#: encoder/mem/WM/router 更新严格为零、fusion/primary 确实更新、experts 随 R1/R2 变化）
RECOVERY_SUGGESTION = {
    "doc_ref": "docs/deepseekv4p1_argue.md §7.10-A4/A6/A9 + round-7（R1/R2 两档消融）",
    "note": "R1 = 不训 experts；R2 = 带 experts/residual（R2 无显著增益 → 用 R1）",
    "R1": {
        "train_prefixes": [
            "plan_head.fusion.",
            "plan_head.norm.",
            "plan_head.ego_next.",
            "plan_head.moe.primary.",
        ],
        "freeze_prefixes": [
            "encoders.",
            "mem_encoder.",
            "st_gnn.",
            "plan_head.moe.router.",
            "plan_head.moe.experts.",
            "plan_head.moe.residual_scale",
            "plan_head.anchor_head.",
            "plan_head.speed_head.",
            "plan_head.residual_head.",
            "plan_head.anchor_embed",
            "policy.",
            "value.",
        ],
    },
    "R2": {
        "train_prefixes": [
            "plan_head.fusion.",
            "plan_head.norm.",
            "plan_head.ego_next.",
            "plan_head.moe.primary.",
            "plan_head.moe.experts.",
            "plan_head.moe.residual_scale",
        ],
        "freeze_prefixes": [
            "encoders.",
            "mem_encoder.",
            "st_gnn.",
            "plan_head.moe.router.",
            "plan_head.anchor_head.",
            "plan_head.speed_head.",
            "plan_head.residual_head.",
            "plan_head.anchor_embed",
            "policy.",
            "value.",
        ],
    },
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# 静态层：ast 只读解析冻结常量
# --------------------------------------------------------------------------- #

def parse_freeze_constants(path: str | Path) -> dict[str, tuple[str, ...]]:
    """解析 ``pipeline/stages.py`` 的模块级冻结常量（``Tuple[str, ...]`` 字面量 + 常量拼接）。

    支持 ``("a.",)`` 与 ``("a.",) + _OTHER_CONST`` 两种写法；非字符串元素/未知 Name → ValueError。
    """
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    raw: dict[str, ast.expr] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            raw[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            raw[node.target.id] = node.value  # ``_X: Tuple[str, ...] = (...)`` 形式

    def resolve(node: ast.expr) -> tuple[str, ...]:
        if isinstance(node, ast.Tuple):
            values = []
            for element in node.elts:
                if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
                    raise ValueError(f"冻结前缀元组含非字符串元素: {ast.dump(element)}")
                values.append(element.value)
            return tuple(values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return resolve(node.left) + resolve(node.right)
        if isinstance(node, ast.Name):
            if node.id not in raw:
                raise ValueError(f"引用了未定义常量: {node.id}")
            return resolve(raw[node.id])
        raise ValueError(f"不支持的冻结常量表达式: {ast.dump(node)}")

    out: dict[str, tuple[str, ...]] = {}
    for name in STATIC_CONSTANT_NAMES:
        if name in raw:
            out[name] = resolve(raw[name])
    return out


def group_of(name: str) -> str:
    """参数名 → 模块组（未命中 → ``other``）。"""
    for group, prefixes in _GROUP_PREFIXES:
        if any(name.startswith(prefix) for prefix in prefixes):
            return group
    return "other"


def _expected_kind(name: str, freeze_prefixes: tuple[str, ...]) -> str:
    return "freeze" if any(name.startswith(prefix) for prefix in freeze_prefixes) else "train"


# --------------------------------------------------------------------------- #
# 构建层：模型 / 随机小批 / 合成损失
# --------------------------------------------------------------------------- #

def _load_config(config_path: str | Path) -> dict:
    from pipeline.stages import load_config

    return load_config(str(config_path))


def _build_model(config: dict, *, seed: int = 0, device: str = "cpu"):
    import torch

    from pipeline.stages import build_model

    torch.manual_seed(int(seed))
    model = build_model(config)
    model.train()
    model.to(torch.device(str(device)))
    return model


def make_random_obs(model, *, batch_size: int = 4, seed: int = 0, device: str = "cpu") -> dict:
    """与 env schema 同形状的随机观测小批（含 warmup 复制帧语义；CPU 友好）。"""
    import torch

    from net.encoders import EGO_MEM_DIM, LD_MEM_DIM, OD_MEM_DIM, NAV_DIM, SIGNAL_DIM

    history = int(model.history_frames)
    od_slots = int(model.od_slots)
    ld_slots = int(model.ld_slots)
    others_dim = int(model.others_dim)
    dev = torch.device(str(device))
    generator = torch.Generator().manual_seed(int(seed))

    def rand(*shape: int):
        return torch.randn(*shape, generator=generator, dtype=torch.float32).to(dev)

    od_mask = (rand(batch_size, history, od_slots) > -0.4).float()
    ld_mask = (rand(batch_size, history, ld_slots) > -0.4).float()
    presence = od_mask * (rand(batch_size, history, od_slots) > -0.3).float()
    od_hist = rand(batch_size, history, od_slots, OD_MEM_DIM) * od_mask.unsqueeze(-1)
    ld_hist = rand(batch_size, history, ld_slots, LD_MEM_DIM) * ld_mask.unsqueeze(-1)
    ego_hist = rand(batch_size, history, 1, EGO_MEM_DIM)
    others_hist = rand(batch_size, history, 1, others_dim)
    # warmup 补位帧 = 复制最旧真实帧（与 memory.py / test_net_shapes.make_obs 同口径）
    valid_frames = min(3, history)
    oldest = history - valid_frames
    od_hist[:, :oldest] = od_hist[:, oldest: oldest + 1]
    ld_hist[:, :oldest] = ld_hist[:, oldest: oldest + 1]
    ego_hist[:, :oldest] = ego_hist[:, oldest: oldest + 1]
    others_hist[:, :oldest] = others_hist[:, oldest: oldest + 1]
    hist_valid = torch.zeros(batch_size, history, device=dev)
    hist_valid[:, oldest:] = 1.0
    ones = torch.ones(batch_size, 1, device=dev)
    obs = {
        "od_hist": od_hist,
        "od_hist_mask": od_mask,
        "ld_hist": ld_hist,
        "ld_hist_mask": ld_mask,
        "hist_valid": hist_valid,
        "ego": rand(batch_size, EGO_MEM_DIM),
        "od": od_hist[:, -1],
        "od_mask": od_mask[:, -1],
        "od_presence": presence[:, -1],
        "ld": ld_hist[:, -1],
        "ld_mask": ld_mask[:, -1],
        "nav": rand(batch_size, NAV_DIM),
        "nav_mask": ones,
        "signal": rand(batch_size, SIGNAL_DIM),
        "signal_mask": ones,
        "ego_hist": ego_hist,
        "ego_hist_mask": torch.ones(batch_size, history, 1, device=dev),
        "others_hist": others_hist,
        "others_hist_mask": torch.ones(batch_size, history, 1, device=dev),
        "od_id_hist": (rand(batch_size, history, od_slots).abs() * 10).long() + 1,
        "od_presence_hist": presence,
        "od_id": (rand(batch_size, od_slots).abs() * 10).long() + 1,
    }
    return obs


def make_loss_fn(model, *, batch_size: int = 4, seed: int = 0, device: str = "cpu"):
    """随机小批前向 + 输出平方损失 + 可训参数平方正则（见模块 docstring）。"""
    import torch

    def loss_fn():
        obs = make_random_obs(model, batch_size=batch_size, seed=seed, device=device)
        out = model(obs, rollout=True, world_model=False)
        loss = out["action_mu"].pow(2).mean() + out["value"].pow(2).mean()
        for key in ("plan", "ego_next"):
            if key in out:
                loss = loss + out[key].pow(2).mean()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if trainable:
            loss = loss + PARAM_REG_COEF * sum(p.pow(2).mean() for p in trainable)
        return loss

    return loss_fn


# --------------------------------------------------------------------------- #
# 单步账本（probe_train_account 复用）
# --------------------------------------------------------------------------- #

def account_step(model, optimizer, loss_fn) -> dict:
    """zero_grad → backward → snapshot → step → 逐参数/逐组四计数 + 逐组 grad-norm。"""
    import torch

    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    per_param: dict[str, dict] = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        per_param[name] = {
            "group": group_of(name),
            "requires_grad": bool(parameter.requires_grad),
            "in_optimizer": id(parameter) in optimizer_ids,
            "grad_nonzero": bool(grad is not None and int(torch.count_nonzero(grad)) > 0),
            "grad_norm": math.sqrt(float(grad.detach().pow(2).sum())) if grad is not None else 0.0,
            "changed": False,
        }
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.step()
    for name, parameter in model.named_parameters():
        per_param[name]["changed"] = not torch.equal(parameter.detach(), before[name])
    return {
        "loss": round(float(loss.detach()), 8),
        "per_param": per_param,
        "groups": _aggregate_groups(per_param),
    }


def _aggregate_groups(per_param: dict[str, dict]) -> dict[str, dict]:
    stats = {
        group: {
            "n_params": 0,
            "requires_grad": 0,
            "in_optimizer": 0,
            "grad_nonzero": 0,
            "grad_norm": 0.0,
            "changed": 0,
        }
        for group in GROUP_ORDER
    }
    squares: dict[str, float] = {group: 0.0 for group in GROUP_ORDER}
    for info in per_param.values():
        stat = stats[info["group"]]
        stat["n_params"] += 1
        stat["requires_grad"] += int(info["requires_grad"])
        stat["in_optimizer"] += int(info["in_optimizer"])
        stat["grad_nonzero"] += int(info["grad_nonzero"])
        stat["changed"] += int(info["changed"])
        squares[info["group"]] += float(info["grad_norm"]) ** 2
    for group in GROUP_ORDER:
        stats[group]["grad_norm"] = round(math.sqrt(squares[group]), 8)
    return stats


# --------------------------------------------------------------------------- #
# 审计核心（可注入 loss_fn / optimizer 供单测）
# --------------------------------------------------------------------------- #

def audit_model(
    model,
    freeze_prefixes,
    *,
    loss_fn=None,
    optimizer=None,
    freeze_mode: str = "all",
    batch_size: int = 4,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """构建层核心：冻结 → optimizer → 单步四计数 → 组期望 + 判定。

    ``loss_fn``/``optimizer`` 可注入（单测用 stub 模型；真实路径由 :func:`run_probe` 传
    ``build_phase3_optimizer`` 与随机小批损失）。
    """
    from pipeline.trainer import apply_freeze_prefixes, build_phase3_optimizer

    freeze_prefixes = tuple(str(prefix) for prefix in freeze_prefixes)
    frozen_names = list(apply_freeze_prefixes(model, freeze_prefixes))
    optimizer_error = None
    if optimizer is None:
        try:
            optimizer = build_phase3_optimizer(model, 1e-3, freeze_mode=str(freeze_mode))
        except ValueError as exc:
            optimizer = None
            optimizer_error = f"build_phase3_optimizer(freeze_mode={freeze_mode}) 拒绝: {exc}"

    if optimizer is not None:
        if loss_fn is None:
            loss_fn = make_loss_fn(model, batch_size=batch_size, seed=seed, device=device)
        step = account_step(model, optimizer, loss_fn)
        per_param = step["per_param"]
        groups = step["groups"]
        loss = step["loss"]
        optimizer_note = {
            "builder": "pipeline.trainer.build_phase3_optimizer",
            "freeze_mode": str(freeze_mode),
            "lr": 1e-3,  # 探针内部固定；base/specific scale 用 builder 默认（0.25 / 0.5）
            "n_param_groups": len(optimizer.param_groups),
            "group_names": [str(g.get("name", f"g{i}")) for i, g in enumerate(optimizer.param_groups)],
            "n_params": sum(len(g["params"]) for g in optimizer.param_groups),
        }
    else:
        per_param = {
            name: {
                "group": group_of(name),
                "requires_grad": bool(parameter.requires_grad),
                "in_optimizer": False,
                "grad_nonzero": False,
                "grad_norm": 0.0,
                "changed": False,
            }
            for name, parameter in model.named_parameters()
        }
        groups = _aggregate_groups(per_param)
        loss = None
        optimizer_note = None

    expected: dict[str, str] = {}
    for group in GROUP_ORDER:
        kinds = {
            _expected_kind(name, freeze_prefixes)
            for name, info in per_param.items()
            if info["group"] == group
        }
        if not kinds:
            expected[group] = "empty"
        elif len(kinds) == 1:
            expected[group] = kinds.pop()
        else:
            expected[group] = "mixed"

    trainable_names = {name for name, info in per_param.items() if info["requires_grad"]}
    optimizer_names = {name for name, info in per_param.items() if info["in_optimizer"]}
    trainable_not_in_optimizer = sorted(trainable_names - optimizer_names)
    optimizer_not_trainable = sorted(optimizer_names - trainable_names)

    failures = _judge(
        groups,
        expected,
        trainable_not_in_optimizer,
        optimizer_not_trainable,
        optimizer_error,
    )
    return {
        "status": "PASS" if not failures else "FAIL",
        "frozen_names": frozen_names,
        "optimizer": optimizer_note,
        "optimizer_error": optimizer_error,
        "loss": loss,
        "per_param": per_param,
        "groups": groups,
        "expected": expected,
        "trainable_not_in_optimizer": trainable_not_in_optimizer,
        "optimizer_not_trainable": optimizer_not_trainable,
        "failures": failures,
    }


def _judge(
    groups: dict[str, dict],
    expected: dict[str, str],
    trainable_not_in_optimizer: list[str],
    optimizer_not_trainable: list[str],
    optimizer_error: str | None,
) -> list[str]:
    failures: list[str] = []
    if optimizer_error:
        failures.append(optimizer_error)
    for group in GROUP_ORDER:
        kind = expected[group]
        stat = groups[group]
        if kind == "mixed":
            failures.append(f"组 {group} 同时含冻结与可训参数（mixed）")
        elif kind == "freeze":
            for key, label in (
                ("requires_grad", "requires_grad>0"),
                ("in_optimizer", "出现在 optimizer"),
                ("grad_nonzero", "有非零梯度"),
                ("changed", "被 step 更新"),
            ):
                if stat[key] > 0:
                    failures.append(f"freeze 组 {group} {label}: {stat[key]}")
        elif kind == "train":
            if stat["requires_grad"] == 0:
                failures.append(f"train 组 {group} 无可训参数（requires_grad=0）")
            if stat["changed"] == 0:
                failures.append(f"train 组 {group} 未被更新（changed=0）")
    if trainable_not_in_optimizer:
        failures.append(f"可训参数不在 optimizer（{len(trainable_not_in_optimizer)}）: {trainable_not_in_optimizer}")
    if optimizer_not_trainable:
        failures.append(f"optimizer 含不可训参数（{len(optimizer_not_trainable)}）: {optimizer_not_trainable}")
    return failures


# --------------------------------------------------------------------------- #
# run_probe / CLI
# --------------------------------------------------------------------------- #

def _write_payload(payload: dict, out: str | Path, *, quiet: bool) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not quiet:
        summary = (
            f"[probe] freeze_audit: {payload['status']} recipe={payload.get('recipe')} "
            f"failures={len(payload.get('failures') or [])}"
        )
        groups = payload.get("groups")
        if groups:
            frozen = [g for g, stat in groups.items() if stat.get("expected") == "freeze" and stat["n_params"]]
            train = [g for g, stat in groups.items() if stat.get("expected") == "train"]
            summary += f" freeze组={frozen} train组={train}"
        print(f"{summary} -> {out_path}", flush=True)


def run_probe(
    recipe: str,
    *,
    config: str | Path = DEFAULT_CONFIG,
    device: str = "cpu",
    out: str | Path = DEFAULT_OUT,
    batch_size: int = 4,
    seed: int = 0,
    quiet: bool = False,
) -> dict:
    """执行审计并把结果写入 out；返回输出 payload。"""
    recipe = str(recipe).strip().lower()
    if recipe not in ("specific_only", "trunk_only", "all", "recovery", "recovery_experts"):
        raise SystemExit(f"[probe] 非法 recipe: {recipe!r}（specific_only|trunk_only|all|recovery|recovery_experts）")

    static_error = None
    static: dict[str, tuple[str, ...]] = {}
    try:
        static = parse_freeze_constants(STAGES_PATH)
    except (OSError, SyntaxError, ValueError) as exc:
        static_error = f"解析 {STAGES_PATH} 失败: {exc}"

    base: dict = {
        "probe": "freeze_audit",
        "recipe": recipe,
        "config": str(config),
        "device": str(device),
        "static_constants": {name: list(values) for name, values in sorted(static.items())},
        "static_error": static_error,
        "generated_at": _now(),
    }

    # recovery / recovery_experts 已实现（P1-2）→ 与其它配方同路径四计数审计。

    if static_error is not None and recipe in RECIPE_CONSTANT:
        base.update({"status": "FAIL", "failures": [static_error]})
        _write_payload(base, out, quiet=quiet)
        return base

    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        base.update({"status": "FAIL", "failures": [f"device={device} 不可用（torch.cuda.is_available()=False）"]})
        _write_payload(base, out, quiet=quiet)
        return base

    prefixes: tuple[str, ...] = ()
    source_label = "（all：无冻结前缀）"
    if recipe in RECIPE_CONSTANT:
        prefixes = tuple(static.get(RECIPE_CONSTANT[recipe], ()))
        source_label = f"pipeline/stages.py::{RECIPE_CONSTANT[recipe]}"

    # 静态解析 vs 运行时导入常量交叉核对（同一源文件的独立路径）
    static_match_runtime = None
    try:
        import pipeline.stages as stages_mod

        runtime = tuple(getattr(stages_mod, RECIPE_CONSTANT[recipe])) if recipe in RECIPE_CONSTANT else ()
        static_match_runtime = runtime == prefixes
    except Exception as exc:  # noqa: BLE001 - 运行时导入失败 → FAIL（记录原因）
        static_match_runtime = None
        base["runtime_import_error"] = f"{type(exc).__name__}: {exc}"

    try:
        model = _build_model(_load_config(config), seed=seed, device=device)
    except Exception as exc:  # noqa: BLE001 - 构建失败 → FAIL（记录原因）
        import traceback

        base.update(
            {
                "status": "FAIL",
                "failures": [f"模型构建失败: {type(exc).__name__}: {exc}"],
                "traceback": traceback.format_exc(),
            }
        )
        _write_payload(base, out, quiet=quiet)
        return base

    result = audit_model(
        model,
        prefixes,
        freeze_mode=recipe,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )
    failures = list(result["failures"])
    if static_match_runtime is False:
        failures.append("静态解析前缀与运行时导入常量不一致（pipeline/stages.py）")
    elif static_match_runtime is None and "runtime_import_error" in base:
        failures.append(f"运行时常量导入失败: {base['runtime_import_error']}")

    base.update(
        {
            "status": "PASS" if not failures else "FAIL",
            "freeze_prefixes": list(prefixes),
            "freeze_prefixes_source": source_label,
            "static_match_runtime": static_match_runtime,
            "n_frozen_params": len(result["frozen_names"]),
            "optimizer": result["optimizer"],
            "optimizer_error": result["optimizer_error"],
            "loss": {
                "kind": "random_obs_forward+param_reg",
                "batch_size": int(batch_size),
                "seed": int(seed),
                "param_reg_coef": PARAM_REG_COEF,
                "value": result["loss"],
            },
            "groups": {
                group: {**result["groups"][group], "expected": result["expected"][group]}
                for group in GROUP_ORDER
            },
            "trainable_not_in_optimizer": result["trainable_not_in_optimizer"],
            "optimizer_not_trainable": result["optimizer_not_trainable"],
            "failures": failures,
        }
    )
    _write_payload(base, out, quiet=quiet)
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe_freeze_audit.py",
        description="探针：freeze recipe 审计（配方→requires_grad→optimizer→实际更新）",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="主配置（默认 config/default.yaml）")
    parser.add_argument(
        "--recipe",
        required=True,
        choices=("specific_only", "trunk_only", "all", "recovery", "recovery_experts"),
        help="冻结配方（recovery=R1 / recovery_experts=R2；P1-2 已实现）",
    )
    parser.add_argument("--device", default="cpu", help="torch device（默认 cpu）")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出 JSON 路径")
    parser.add_argument("--batch-size", type=int, default=4, help="随机小批大小（默认 4）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_probe(
        args.recipe,
        config=args.config,
        device=args.device,
        out=args.out,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    return _EXIT_BY_STATUS.get(payload["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
