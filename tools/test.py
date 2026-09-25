#!/usr/bin/env python3
"""评估入口（P2/N5）：冻结验证集评测（委托 ``pipeline.eval_runner``）。

契约（``.slim/deepwork/p2-contract.md`` §5）：``--config --spec --ckpt --out --policy``。
本入口只做参数校验与转发，真正的评测在 ``pipeline.eval_runner.main``：

- ``--policy baseline``：规则基线 ``PurePursuitIDMPolicy``（复现冻结参照，不需要 ``--ckpt``）；
- ``--policy ckpt``：加载 N1 ``DrivingModel`` 权重 + N3 跟踪器闭环（必须给 ``--ckpt``）。

未识别参数原样透传（``--workers/--limit/--name/--max-steps/--baseline-ref/--seed``），
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
    return parser


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

    forward = ["--config", str(args.config), "--policy", args.policy, "--out", str(args.out)]
    if args.spec is not None:
        forward += ["--spec", str(args.spec)]
    if args.ckpt is not None:
        forward += ["--ckpt", str(args.ckpt)]
    forward += list(_extra)  # --workers/--limit/--name/--max-steps/--baseline-ref/--seed 透传

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
