#!/usr/bin/env python3
"""脚本取值工具（tools/train.sh / test.sh 以 ``eval "$(tools/venv-python tools/run_config.py ...)"`` 消费）。

路径/派生值的**唯一来源**在 ``config/train.yaml`` + ``pipeline/run_paths.py``（脚本里不写死路径）：

- ``run.work_dir`` auto = ``runs/BTC<北京戳>_<name>``（同轮 A/B 共用一个 run 根：Stage B 的 run 根
  取自 ``stage_a_ckpt`` 所属 Stage A run）
- ``run.bc_dir``   auto = 最新 ``datasets/BTC*_expert*``（须含 ``expert_bc.npz``）
- ``run.resume``   auto = 最新「有 ``stage_<x>/ckpt_epoch*.pt`` 且无 ``final.pt``」的 stage（原地续跑）
- ``run.stage_a_ckpt`` auto = 最新 Stage A ``stage_a/final.pt``；``run.eval_ckpt`` auto = 最新 B final，回退 A final

``--profile train`` 打印 ``BC_DIR WORK_DIR STAGE STAGE_DIR LOG DETACH_PID WM_EPOCHS BC_EPOCHS
CKPT_EVERY BATCH_SIZE MICRO_BATCH_SIZE TRAJ_AUX_WEIGHT CKPT RESUME``；``--profile eval`` 打印
``NAME WORK_DIR OUT_ROOT LOG DETACH_PID CKPT``；``--profile tb`` 打印
``TB_SPEC='name:path,...'``（run 名见 ``pipeline.run_paths.TB_RUN_ORDER``）。
同名环境变量优先（含显式置空：``RESUME=`` 关闭续跑）。
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import run_paths  # noqa: E402  （必须在 sys.path 处理之后）


def _over(key: str, default: Any) -> Any:
    """环境变量存在即覆盖（空串也算，便于显式关闭 resume/ckpt）。"""
    return os.environ[key] if key in os.environ else default


def _rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resume_stage_of(path: Path) -> str:
    return "B" if re.search(r"stage[_-]?b", str(path), re.IGNORECASE) else "A"


def resolve(root: Path, cfg: Mapping[str, Any], stage_arg: str = "") -> dict[str, str]:
    """train 侧取值（resume/阶段/work_dir/共戳全部按 run_paths 约定派生）。"""
    run = dict(cfg.get("run") or {})
    stages = dict(cfg.get("stages") or {})
    stage_a = dict(stages.get("A") or {})
    stage_b = dict(stages.get("B") or {})
    world_model = dict(stage_a.get("world_model") or {})
    bc = dict(stage_b.get("bc") or {})

    bc_dir = str(_over("BC_DIR", run.get("bc_dir", "auto")))
    if bc_dir == "auto":
        found = run_paths.latest_dataset(root)
        if found is None:
            print("[run_config] 错误：datasets/ 下找不到 BTC*_expert*（含 expert_bc.npz）", file=sys.stderr)
            raise SystemExit(2)
        bc_dir = _rel(found, root)

    # Stage A final（B 的 --ckpt；也是 B 复用 A run 根的依据）
    a_cfg = str(_over("STAGE_A_CKPT", run.get("stage_a_ckpt", "auto")))
    a_final: Path | None = None
    if a_cfg == "auto":
        a_final = run_paths.latest_final(root, "A")
    elif a_cfg:
        a_final = Path(a_cfg)
    a_final_str = _rel(a_final, root)

    # resume：auto = 最新未完成 stage（原地续跑，复用其 run 根）
    stage = stage_arg or str(_over("STAGE", ""))
    resume_cfg = _over("RESUME", run.get("resume", "auto"))
    resume_path, work_dir, out_dir = "", None, None
    if resume_cfg == "auto":
        found = run_paths.find_unfinished(root)
        if found is not None:
            found_wd, found_out, found_stage, ckpt = found
            if not stage or stage == found_stage:
                stage, work_dir, out_dir = stage or found_stage, found_wd, found_out
                resume_path = _rel(ckpt, root)
    elif resume_cfg:
        explicit = Path(str(resume_cfg))
        explicit_stage = _resume_stage_of(explicit)
        if not stage or stage == explicit_stage:
            stage = stage or explicit_stage
            resume_path, out_dir = str(explicit), explicit.parent
            work_dir = run_paths.work_dir_of_out(out_dir)
    if not stage:
        stage = "A"
    if out_dir is None:  # 全新 run：按约定生成 run 根（B 与 A 共用）
        work_dir_cfg = str(_over("WORK_DIR", run.get("work_dir", "auto")))
        if work_dir_cfg == "auto":
            if stage == "B" and a_final is not None:
                work_dir = run_paths.work_dir_of_out(a_final.parent)
            else:
                work_dir = run_paths.new_work_dir(str(_over("NAME", run.get("name", "train"))), root)
        else:
            work_dir = Path(work_dir_cfg)
        out_dir = run_paths.stage_dir(work_dir, stage)

    chosen = stage_a if stage == "A" else stage_b
    return {
        "BC_DIR": bc_dir,
        "WORK_DIR": str(work_dir),
        "STAGE": stage,
        "STAGE_DIR": str(out_dir),
        "LOG": str(run_paths.stage_log(work_dir, stage)),
        "DETACH_PID": str(run_paths.detach_pid_path(work_dir)),
        "WM_EPOCHS": str(_over("WM_EPOCHS", stage_a.get("epochs", world_model.get("epochs", 20)))),
        "BC_EPOCHS": str(_over("BC_EPOCHS", stage_b.get("epochs", bc.get("epochs", 20)))),
        "CKPT_EVERY": str(_over("CKPT_EVERY", run.get("checkpoint_interval_epochs", 5))),
        "BATCH_SIZE": str(_over("BATCH_SIZE", chosen.get("batch", 1024))),
        "MICRO_BATCH_SIZE": str(_over("MICRO_BATCH_SIZE", chosen.get("micro", 256))),
        "TRAJ_AUX_WEIGHT": (
            str(_over("TRAJ_AUX_WEIGHT", stage_b.get("traj_aux_weight", bc.get("traj_aux_weight", ""))))
            if stage == "B" else ""
        ),
        "CKPT": str(_over("CKPT", a_final_str)) if stage == "B" else "",
        "RESUME": resume_path,
    }


def resolve_eval(root: Path, cfg: Mapping[str, Any], policy: str = "ckpt", limit: str = "50",
                 tracker: str = "lqr") -> dict[str, str]:
    """test 侧取值（``CKPT=...`` / ``LIMIT=...`` 等环境变量覆盖仍可用）。"""
    run = dict(cfg.get("run") or {})
    value = _over("CKPT", _over("EVAL_CKPT", run.get("eval_ckpt", "auto")))
    if value == "auto":
        found = run_paths.latest_final(root, "B") or run_paths.latest_final(root, "A")
        value = _rel(found, root)
    name = str(_over("NAME", "") or "")
    if not name:
        tag = f"eval_baseline{limit}" if policy == "baseline" else f"eval_{tracker}{limit}"
        name = f"BTC{run_paths.beijing_stamp()}_{tag}"
    work_dir = Path(_over("WORK_DIR", f"runs/{name}"))
    return {
        "NAME": name,
        "WORK_DIR": str(work_dir),
        "OUT_ROOT": str(work_dir.parent),
        "LOG": str(run_paths.stage_log(work_dir, "eval")),
        "DETACH_PID": str(run_paths.detach_pid_path(work_dir)),
        "CKPT": str(value or ""),
    }


def resolve_tb(root: Path = ROOT) -> dict[str, str]:
    """TensorBoard 入口取值：``TB_SPEC='name:path,name:path'``（无候选 → 空串）。"""
    spec = run_paths.tb_spec(root)
    return {"TB_SPEC": ",".join(f"{name}:{path}" for name, path in spec)}


def emit(values: Mapping[str, str]) -> None:
    for key, value in values.items():
        print(f"{key}={shlex.quote(str(value))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="读 config/train.yaml → 打印脚本用 KEY=value（auto 自动解析）")
    parser.add_argument("--config", default="config/train.yaml", help="配置路径（相对仓库根，默认 config/train.yaml）")
    parser.add_argument("--profile", choices=("train", "eval", "tb"), default="train")
    parser.add_argument("--stage", default="", help="显式阶段 A|B（默认按 resume auto 推断，否则 A）")
    parser.add_argument("--policy", default="ckpt", help="eval profile：ckpt|baseline（命名用）")
    parser.add_argument("--limit", default="50", help="eval profile：评测条数（命名用）")
    parser.add_argument("--tracker", default="lqr", help="eval profile：跟踪器（命名用）")
    args = parser.parse_args(argv)

    path = Path(args.config)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        print(f"[run_config] 错误：配置不存在 {path}", file=sys.stderr)
        return 2
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if args.profile == "tb":
        values = resolve_tb(ROOT)
        if not values["TB_SPEC"]:
            print("[run_config] 无可用 monitor 事件目录（先跑 tools/train.sh）", file=sys.stderr)
        emit(values)
    elif args.profile == "eval":
        emit(resolve_eval(ROOT, cfg, policy=args.policy, limit=args.limit, tracker=args.tracker))
    else:
        emit(resolve(ROOT, cfg, stage_arg=args.stage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
