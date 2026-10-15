#!/usr/bin/env python3
"""评估入口（P2/N5）：冻结验证集评测（委托 ``pipeline.eval_runner``）。

契约（本地档 ``.slim/deepwork/p2-contract.md`` §5，已随 CLEAN-1 清理）：``--config --spec --ckpt --out --policy``。
本入口只做参数校验与转发，真正的评测在 ``pipeline.eval_runner.main``：

- ``--policy baseline``：规则基线 ``PurePursuitIDMPolicy``（复现冻结参照，不需要 ``--ckpt``）；
- ``--policy ckpt``：加载 N1 ``DrivingModel`` 权重 + N3 跟踪器闭环（必须给 ``--ckpt``）。

未识别参数原样透传（``--workers/--limit/--name/--max-steps/--baseline-ref/--seed``
``--allow-partial-load/--tracker/--eval-reference/--device/--moe-off`` 等），
``--path-head {off,structured}``（R39-A1 结构化装配，声明于本入口后显式转发）同列；
run 名由代码自适应规范化（``pipeline/run_paths.py::eval_run_name``；调用方不手拼）：

- ``--name`` 缺省 → ``BTC<秒级北京戳>_eval500_<policy>``（kind 随 ``--spec`` 文件名自适应）；
- ``--name EXP_foo`` → ``BTC<秒级北京戳>_EXP_foo``；已是规范名则原样；
- ``--out`` 是输出根目录（路径型，不参与命名；实际写 ``<out>/<name>/``）。

例如薄切片冒烟::

    tools/venv-python tools/test.py --policy ckpt --ckpt runs/train/stageA/final.pt \\
        --limit 50 --workers 2 --name stageA_slice50
    tools/venv-python tools/test.py --policy baseline --limit 10 --workers 1

import 时只有 stdlib（``pipeline.eval_runner`` 延迟到 ``main()`` 内导入），``--help`` 无副作用。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许 `python tools/test.py` 直接运行（此时 sys.path[0] 是 tools/）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    """构造契约参数解析器（其余参数透传给 pipeline.eval_runner）。"""
    parser = argparse.ArgumentParser(
        prog="tools/test.py",
        description="DRL_PathPlan 评估入口（冻结 val 集：primary 分组 + Wilson CI + 弱类 floor）",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                        help="主配置路径（默认 config/default.yaml，includes 自动合并）")
    parser.add_argument("--spec", type=Path, default=None,
                        help="验证场景 spec（默认取 eval.yaml::eval.spec = 冻结 1000 条）")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="待评估权重路径（--policy ckpt 必填）")
    parser.add_argument("--out", type=Path, default=Path("runs/eval"),
                        help="评测输出根目录（实际写 <out>/<name>/，默认 runs/eval）")
    parser.add_argument("--policy", choices=("baseline", "ckpt"), default="ckpt",
                        help="baseline=规则基线参照；ckpt=策略权重（默认 ckpt）")
    parser.add_argument("--path-head", choices=("off", "structured"), default="off",
                        help="动作头装配（R39-A1；透传 pipeline.eval_runner）：off=默认旧 PolicyHead"
                             "（逐位不变）；structured=结构化头 policy.structured_action（口径对齐 "
                             ".slim/deepwork/p1_v2/a3/eval_a3.py）。结构化前向需要 obs['lane_anchor_xy']"
                             "：用 --lane-anchor inject 显式注入（否则缺键在 forward fail-fast）")
    parser.add_argument("--ld-mode", default="baseline",
                        choices=("baseline", "no_chain", "set_attn", "attn", "attn_no_chain",
                                 "semantic", "semantic_no_chain"),
                        help="LD 消融装配（R39-A1b；透传 runner；默认 baseline=生产口径）。"
                             "semantic 需 K1e2 血统语义键 ⇒ 配 --fill-from")
    parser.add_argument("--rollout-semantic", default="full",
                        choices=("full", "t0_only", "static_role_only"),
                        help="rollout semantic 注入口径（R39-A1b；默认 full=生产口径）")
    parser.add_argument("--lane-anchor", choices=("off", "inject"), default="off",
                        help="lane_anchor_xy 注入器（R39-A1b；默认 off=不触达）：inject=每决策步从"
                             "活 env 采目标车道中心线写入 obs（R36 C lane 注入器同源，与 a3 同口径）")
    parser.add_argument("--fill-from", type=Path, default=None,
                        help="缺键回填源 ckpt（R39-A1b；显式给出才触达；K1e2 血统例 = "
                             "runs/p1_v2/round28/K1/ckpt_epoch002.pt）")
    parser.add_argument("--world-state-api", default="off",
                        choices=("off", "analytic", "analytic_learned"),
                        help="统一 imagined WorldState API（R39-B2c；透传 pipeline.eval_runner）："
                             "off=默认不触达（与 A1b 口径逐字节同 hash）；analytic/analytic_learned"
                             "需 --path-head structured + --lane-anchor inject；learned 档需 "
                             "--world-residual-head")
    parser.add_argument("--world-residual-head", type=Path, default=None,
                        help="学习残差头 ckpt（R39-B2c；learned 档专用，如 "
                             "runs/round39/b2/residual_head.pt）")
    return parser


def _drop_name_arg(argv: "list[str]") -> "list[str]":
    """去掉透传 argv 里的 ``--name``/``--name=…``（命名已由本入口统一接管，避免 raw 值后到覆盖）。"""
    kept: "list[str]" = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == "--name":
            skip_next = True
            continue
        if item.startswith("--name="):
            continue
        kept.append(item)
    return kept


def main(argv: "list[str] | None" = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args, _extra = build_parser().parse_known_args(argv_list)

    if not args.config.is_file():
        print(f"[test] 配置不存在：{args.config}", file=sys.stderr)
        return 2
    if args.spec is not None and not args.spec.is_file():
        print(f"[test] spec 不存在：{args.spec}", file=sys.stderr)
        return 2
    if args.policy == "ckpt":
        if args.ckpt is None:
            print("[test] --policy ckpt 需要 --ckpt（或改用 --policy baseline）", file=sys.stderr)
            return 2
        if not args.ckpt.is_file():
            print(f"[test] ckpt 不存在：{args.ckpt}", file=sys.stderr)
            return 2

    # 运行目录布局（代码默认，见 pipeline/run_paths.py）：logs/ + manifest + 配置快照。
    # 自适应命名（runs/BTC<秒级戳>_<kind>_<tag>）：--out 是路径根不当名字；路径型 --name 原样透传。
    from pipeline import run_paths

    raw_name = run_paths.extra_value(argv_list, "--name")
    run_name = run_paths.eval_run_name(
        raw_name or None, spec=args.spec, policy=args.policy, default_kind="eval500"
    )
    work_dir = Path(args.out) / run_name
    run_paths.prepare_layout(work_dir)
    run_paths.write_manifest(
        work_dir, stage="eval", config=args.config, out=work_dir, argv=["tools/test.py", *argv_list]
    )

    forward = ["--config", str(args.config), "--policy", args.policy, "--out", str(args.out),
               "--name", run_name, "--path-head", str(args.path_head),
               "--ld-mode", str(args.ld_mode), "--rollout-semantic", str(args.rollout_semantic),
               "--lane-anchor", str(args.lane_anchor),
               "--world-state-api", str(args.world_state_api)]
    if args.fill_from is not None:
        forward += ["--fill-from", str(args.fill_from)]
    if args.world_residual_head is not None:
        forward += ["--world-residual-head", str(args.world_residual_head)]
    if args.spec is not None:
        forward += ["--spec", str(args.spec)]
    if args.ckpt is not None:
        forward += ["--ckpt", str(args.ckpt)]
    forward += _drop_name_arg(_extra)  # --workers/--limit/... 透传（--name 已规范化）

    # GL 修复：在 import metadrive/panda3d 之前预载 venv glvnd，并让 spawn worker 继承路径。
    # （即使本入口未经 tools/venv-python 启动也要生效；见 pipeline/gl_runtime.py）
    from pipeline.gl_runtime import ensure_gl_library_path

    ensure_gl_library_path(preload=True)

    try:
        from pipeline.eval_runner import main as eval_main  # N5 评测（延迟导入）
    except ImportError as exc:
        print(f"[test] 无法导入 pipeline.eval_runner：{exc}", file=sys.stderr)
        return 3
    return int(eval_main(forward) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
