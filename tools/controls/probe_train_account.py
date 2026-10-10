#!/usr/bin/env python3
"""探针：单步训练账本（P1-0C 批次 3a；§7.10-A9）。

同一随机小批单步（前向 → backward → optimizer.step）的完整账本：

- 模型**全部 trainable 参数是否都在 optimizer**（列出不在的；反向也列出 optimizer 里的
  不可训参数——一致性违规）；
- backward 后**逐组 grad-norm**（pre-step，未裁剪）；
- step 前后**逐组 updated 计数**（前后 snapshot 比较）；
- frozen 参数的非零梯度/被更新（应严格为空）。

与 ``probe_freeze_audit`` 共享私有 helper（``audit_model`` / ``GROUP_ORDER`` /
``RECIPE_CONSTANT`` 等；不新增公共模块）。

用法::

    tools/venv-python tools/controls/probe_train_account.py [--recipe all|specific_only|trunk_only] \\
        [--config config/default.yaml] [--device cpu] [--batch-size 4] [--seed 0] \\
        [--out runs/p1_0/probe_train_account.json]

退出码：PASS=0 / FAIL=1 / NOT_AVAILABLE=2。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.controls.probe_freeze_audit import (  # noqa: E402
    DEFAULT_CONFIG,
    GROUP_ORDER,
    RECIPE_CONSTANT,
    _EXIT_BY_STATUS,
    _build_model,
    _load_config,
    _now,
    audit_model,
)

DEFAULT_OUT = "runs/p1_0/probe_train_account.json"
RECIPES = ("all", "specific_only", "trunk_only")


def _write_payload(payload: dict, out: str | Path, *, quiet: bool) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not quiet:
        print(
            f"[probe] train_account: {payload['status']} recipe={payload.get('recipe')} "
            f"trainable={payload.get('n_trainable')}/{payload.get('n_params_total')} "
            f"not_in_optimizer={len(payload.get('trainable_not_in_optimizer') or [])} "
            f"failures={len(payload.get('failures') or [])} -> {out_path}",
            flush=True,
        )


def account_ledger(result: dict) -> dict:
    """``audit_model`` 结果 → 单步账本（membership 一致性 + 逐组 grad-norm/updated）。"""
    per_param = result["per_param"]
    trainable = {name for name, info in per_param.items() if info["requires_grad"]}
    optimizer_names = {name for name, info in per_param.items() if info["in_optimizer"]}
    trainable_not_in_optimizer = sorted(trainable - optimizer_names)
    optimizer_not_trainable = sorted(optimizer_names - trainable)
    trainable_not_updated = sorted(name for name in trainable if not per_param[name]["changed"])
    frozen_grad_nonzero = sorted(
        name for name, info in per_param.items() if not info["requires_grad"] and info["grad_nonzero"]
    )
    frozen_changed = sorted(
        name for name, info in per_param.items() if not info["requires_grad"] and info["changed"]
    )

    failures: list[str] = []
    if result["optimizer_error"]:
        failures.append(result["optimizer_error"])
    if not trainable:
        failures.append("无可训参数（配方在该模型上不适用）")
    if trainable_not_in_optimizer:
        failures.append(
            f"trainable 参数不在 optimizer（{len(trainable_not_in_optimizer)}）: {trainable_not_in_optimizer}"
        )
    if optimizer_not_trainable:
        failures.append(
            f"optimizer 含不可训参数（{len(optimizer_not_trainable)}）: {optimizer_not_trainable}"
        )
    if frozen_grad_nonzero:
        failures.append(f"frozen 参数有非零梯度（{len(frozen_grad_nonzero)}）: {frozen_grad_nonzero}")
    if frozen_changed:
        failures.append(f"frozen 参数被 step 更新（{len(frozen_changed)}）: {frozen_changed}")

    return {
        "status": "PASS" if not failures else "FAIL",
        "n_params_total": len(per_param),
        "n_trainable": len(trainable),
        "n_optimizer_params": len(optimizer_names),
        "optimizer_coverage": (
            round(len(trainable & optimizer_names) / len(trainable), 6) if trainable else None
        ),
        "trainable_not_in_optimizer": trainable_not_in_optimizer,
        "optimizer_not_trainable": optimizer_not_trainable,
        "trainable_not_updated": trainable_not_updated,
        "trainable_not_updated_note": (
            "信息项（非一致性违规）：随机小批未命中的专家/零初始化输出等可能无梯度 → 不更新；"
            "一致性判定只看 optimizer 归属与 frozen 不变量"
        ),
        "frozen_grad_nonzero": frozen_grad_nonzero,
        "frozen_changed": frozen_changed,
        "groups": {group: result["groups"][group] for group in GROUP_ORDER},
        "failures": failures,
    }


def run_probe(
    recipe: str = "all",
    *,
    config: str | Path = DEFAULT_CONFIG,
    device: str = "cpu",
    out: str | Path = DEFAULT_OUT,
    batch_size: int = 4,
    seed: int = 0,
    quiet: bool = False,
) -> dict:
    """执行单步账本并把结果写入 out；返回输出 payload。"""
    recipe = str(recipe).strip().lower()
    if recipe not in RECIPES:
        raise SystemExit(f"[probe] 非法 recipe: {recipe!r}（{'|'.join(RECIPES)}）")

    base: dict = {
        "probe": "train_account",
        "recipe": recipe,
        "config": str(config),
        "device": str(device),
        "generated_at": _now(),
    }

    import torch

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        base.update({"status": "FAIL", "failures": [f"device={device} 不可用（torch.cuda.is_available()=False）"]})
        _write_payload(base, out, quiet=quiet)
        return base

    try:
        import pipeline.stages as stages_mod

        prefixes: tuple[str, ...] = (
            tuple(getattr(stages_mod, RECIPE_CONSTANT[recipe])) if recipe in RECIPE_CONSTANT else ()
        )
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
    base.update(
        {
            "freeze_prefixes": list(prefixes),
            "optimizer": result["optimizer"],
            "loss": result["loss"],
            **account_ledger(result),
        }
    )
    _write_payload(base, out, quiet=quiet)
    return base


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="probe_train_account.py",
        description="探针：单步训练账本（trainable⊆optimizer + 逐组 grad-norm/updated）",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="主配置（默认 config/default.yaml）")
    parser.add_argument(
        "--recipe", default="all", choices=RECIPES, help="冻结配方（默认 all = 全参数可训）"
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
