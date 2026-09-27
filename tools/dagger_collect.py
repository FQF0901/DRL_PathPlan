#!/usr/bin/env python3
"""DAgger-lite 采集（lane T）：冻结 primary 的失败场景池 → 规则专家完整 episode 数据 + 策略 shadow 查询。

协议（文档化；与用户定稿的差异见下）
------------------------------------
- **场景池 = 冻结 primary 的失败场景优先**：``--from-eval runs/*_eval*/episodes.csv`` 取
  ``success=False`` 的 spec（``--spec-source`` 提供 id→spec 映射），或 ``--specs`` 直接给池；
- **采集 = 规则专家 roll-out**（``--labeler pure_pursuit|idm``，复用 ``tools/collect_expert.py``
  的完整采集/过滤/配平/写出流水线，``--workers N`` 多核）；行/权重/过滤全部复用其共享函数，
  本工具不复制任何写行逻辑；
- **冻结 primary 的 shadow 查询**：对刚写出的数据集逐行跑冻结 primary 的 cheap path 前向
  （与评测同观测/同动作口径），把逐行 IL 误差（``w·|μ−专家首步动作|``）写进
  ``dagger_policy_shadow.npz``，并把统计（mean/median/分位）写进 ``dagger.json``；
- **病态过滤**（阈值写死并记录）：复用 ``extract_samples`` 规则 + 追加
  ``stuck``（速度 < 0.5 m/s）/ ``yaw_outlier``（|dθ| > 0.35 rad/0.5 s）——逐行
  ``train_weight`` 置 0（不删行），统计进 ``dagger.json``；
- **产物**：``expert_bc.npz`` + ``expert_bc.meta.json``（追加 ``dagger`` 块：driver ckpt
  sha256 / labeler / spec 清单 / 过滤统计）+ ``report.json`` + ``dagger.json``。

**与用户定稿的差异（需用户决定）**：MetaDrive 每进程单 engine，无法在"策略访问状态"上
执行专家动作取实测标签（标签必须来自专家在 env 内的实测执行）。本工具因此用
**失败场景池 + 专家 roll-out（标签）+ 策略 shadow 查询（访问分布诊断）**的 DAgger-lite 近似；
策略闭环 roll-in 的执行式 DAgger 需后续 lane（影子 env 或动作解析映射）落地。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: 病态过滤阈值（写死并记录进 dagger.json；勿按场景调参）
STUCK_SPEED_MPS = 0.5
YAW_OUTLIER_RAD = 0.35


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tools/dagger_collect.py", description="DAgger-lite 采集（lane T）")
    parser.add_argument("--ckpt", type=Path, required=True, help="冻结 primary 权重（shadow 查询参照）")
    parser.add_argument("--out", type=Path, required=True, help="输出数据集目录（datasets/BTC<ts>_daggerN）")
    parser.add_argument("--specs", type=str, default="", help="spec 池 json（与 --from-eval 二选一）")
    parser.add_argument("--from-eval", type=str, default="", help="失败清单：runs/*_eval*/episodes.csv")
    parser.add_argument("--spec-source", type=str, default="env/specs/scenarios_train.json",
                        help="--from-eval 的 spec id → spec 映射源")
    parser.add_argument("--labeler", choices=("pure_pursuit", "idm"), default="pure_pursuit")
    parser.add_argument("--workers", type=int, default=4, help="采集并行 worker 数（多核；透传 collect_expert）")
    parser.add_argument("--max-steps", type=int, default=600, help="单 episode 最大 env step")
    parser.add_argument("--limit", type=int, default=0, help="调试：spec 数上限（0=全部）")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--model-config", type=str, default="config/model.yaml")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--collect-expert", type=str, default="tools/collect_expert.py",
                        help="复用的采集入口（不改其行为）")
    parser.add_argument("--skip-collect", action="store_true", help="只做 shadow 查询/元数据（数据已存在）")
    return parser


def _select_specs(args: argparse.Namespace) -> "list":
    from env.scenario.spec import load_specs

    if args.specs:
        specs = list(load_specs(str(args.specs)))
        source = f"specs:{args.specs}"
    elif args.from_eval:
        csv_path = Path(args.from_eval)
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        rows = [line.split(",") for line in lines]
        header = {name: index for index, name in enumerate(rows[0])} if rows else {}
        failed = set()
        for row in rows[1:]:
            if len(row) <= max(header.get("spec_id", 0), header.get("success", 0)):
                continue
            if str(row[header["success"]]).strip().lower() in ("false", "0", "no"):
                failed.add(str(row[header["spec_id"]]).strip())
        specs = [
            spec for spec in load_specs(str(args.spec_source))
            if str(getattr(spec, "id", "")) in failed
        ]
        source = f"from_eval:{csv_path}（失败 {len(failed)} → 命中 {len(specs)}）"
    else:
        raise SystemExit("[dagger] 需要 --specs 或 --from-eval 指定场景池")
    if args.limit:
        specs = specs[: int(args.limit)]
    if not specs:
        raise SystemExit("[dagger] 场景池为空")
    print(f"[dagger] 场景池 {source} → {len(specs)} spec", flush=True)
    return specs


def _apply_pathological_filters(dataset, *, stuck_speed: float = STUCK_SPEED_MPS,
                                yaw_outlier: float = YAW_OUTLIER_RAD) -> dict:
    """病态行过滤（写死阈值）：卡死 / 大转向离群 → ``train_weight=0``（不删行）；返回统计。

    速度 = ``traj30_measured``（缺失回退 ``traj30``）首末点距离 / 2.5 s（0.1 s × 25 步的 30 点窗口）；dθ 取
    ``action[:, 0, 1]``（首步专家动作）。只改 ``train_weight/balance_weight``（配平权重同置 0，
    与 collect_expert 的过滤语义一致）。
    """
    arrays = dataset.arrays
    weights = np.asarray(arrays["train_weight"], dtype=np.float32)
    measured_key = "traj30_measured" if "traj30_measured" in arrays else "traj30"
    measured = np.asarray(arrays[measured_key], dtype=np.float64)
    speed = np.linalg.norm(measured[:, -1, :] - measured[:, 0, :], axis=-1) / 2.5
    dtheta = np.abs(np.asarray(arrays["action"], dtype=np.float64)[:, 0, 1])
    active = weights > 0.0
    stuck = active & (speed < float(stuck_speed))
    yaw = active & ~stuck & (dtheta > float(yaw_outlier))
    for key in ("train_weight", "balance_weight", "sample_weight"):
        if key in arrays:
            arr = np.asarray(arrays[key], dtype=np.float32).copy()
            arr[stuck | yaw] = 0.0
            arrays[key] = arr
    return {
        "rows": int(weights.size),
        "stuck": int(np.count_nonzero(stuck)),
        "yaw_outlier": int(np.count_nonzero(yaw)),
        "trainable_after": int(np.count_nonzero(np.asarray(arrays["train_weight"]) > 0.0)),
        "thresholds": {"stuck_speed_mps": float(stuck_speed), "yaw_outlier_rad": float(yaw_outlier)},
    }


def main(argv: "list[str] | None" = None) -> int:
    args = _build_parser().parse_args(argv)
    from pipeline.hard_mining import ckpt_identity, row_il_errors
    from pipeline.stages import build_model, load_config
    from pipeline.trainer import BCDataset, apply_thread_limits, load_checkpoint

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    specs = _select_specs(args)
    from env.scenario.spec import save_specs

    spec_path = out / "dagger_specs.json"
    save_specs(specs, spec_path)
    print(f"[dagger] spec 清单 → {spec_path}", flush=True)

    if not args.skip_collect:
        cmd = [
            sys.executable, str(_ROOT / args.collect_expert),
            "--specs", str(spec_path), "--out", str(out),
            "--expert", str(args.labeler), "--workers", str(int(args.workers)),
            "--max-steps", str(int(args.max_steps)),
        ]
        print(f"[dagger] 复用采集流水线：{' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(_ROOT), check=False)
        if proc.returncode != 0:
            print(f"[dagger] 采集失败（rc={proc.returncode}）", file=sys.stderr)
            return proc.returncode

    dataset = BCDataset.load(str(out))
    pathological = _apply_pathological_filters(dataset)
    if pathological["stuck"] or pathological["yaw_outlier"]:
        # 只改权重（行保留）；重写 npz（schema/键不变，meta 由 BCDataset 从 expert_bc.meta.json 读）
        np.savez_compressed(out / "expert_bc.npz", **{key: dataset.arrays[key] for key in dataset.arrays})
    print(
        f"[dagger] 病态过滤：stuck={pathological['stuck']} · yaw_outlier={pathological['yaw_outlier']}"
        f"（阈值 {pathological['thresholds']}；trainable={pathological['trainable_after']}/{pathological['rows']}）",
        flush=True,
    )
    config = load_config(str(args.config))
    apply_thread_limits(workers=1, config=config)
    model = build_model(load_config(str(args.model_config)))
    load_checkpoint(str(args.ckpt), model)
    print(
        f"[dagger] 冻结 primary shadow 查询：{args.ckpt} × {dataset.count} 行（cheap path，与评测同口径）",
        flush=True,
    )
    shadow = row_il_errors(model, dataset, device=args.device, batch_size=256, logger=print)
    np.savez_compressed(
        out / "dagger_policy_shadow.npz",
        policy_il_err=shadow["err_l1"],
        policy_il_err_weighted=shadow["err_weighted"],
        weight=shadow["weight"],
        ckpt_sha256=np.asarray([ckpt_identity(str(args.ckpt))["sha256"]]),
    )
    stats = {
        "rows": int(dataset.count),
        "policy_il_err_mean": float(np.mean(shadow["err_l1"])),
        "policy_il_err_weighted_mean": float(
            np.average(shadow["err_l1"], weights=np.maximum(shadow["weight"], 0.0))
        ),
        "policy_il_err_p50": float(np.percentile(shadow["err_l1"], 50)),
        "policy_il_err_p95": float(np.percentile(shadow["err_l1"], 95)),
    }
    filter_stats = {}
    report_path = out / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        filter_stats = dict((report.get("filters") or report.get("filter_counts") or {}))
    meta_path = out / "expert_bc.meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    meta["dagger"] = {
        "driver_ckpt": ckpt_identity(str(args.ckpt)),
        "labeler": str(args.labeler),
        "spec_source": str(args.specs or args.from_eval),
        "spec_ids": [str(getattr(spec, "id", "")) for spec in specs],
        "rows": int(dataset.count),
        "filter_counts": filter_stats,
        "pathological": pathological,
        "shadow": stats,
        "protocol": "expert-rollout labels + frozen-primary shadow query（见 tools/dagger_collect.py docstring）",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dagger.json").write_text(
        json.dumps(meta["dagger"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[dagger] DONE → {out}（rows={dataset.count} · shadow IL err mean={stats['policy_il_err_mean']:.4f}）",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
