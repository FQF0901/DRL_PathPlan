"""评估入口（P0 占位，P1/P2 实现）。

用法：
    python3 tools/test.py --config config/default.yaml \
        --spec env/specs/scenarios_val.json --ckpt <ckpt>
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DRL_PathPlan 评估入口（P0 占位）")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                        help="主配置路径（默认 config/default.yaml）")
    parser.add_argument("--spec", type=Path, default=Path("env/specs/scenarios_val.json"),
                        help="验证场景 spec（默认 eval.yaml 中的 val spec）")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="待评估权重路径")
    parser.add_argument("--out", type=Path, default=Path("runs/eval"),
                        help="输出目录（默认 runs/eval）")
    return parser


def main() -> None:
    build_parser().parse_args()
    # P0 仅校验参数；确定性评估与 KPI 统计在 P1/P2 实现
    raise NotImplementedError("P1/P2")


if __name__ == "__main__":
    main()
