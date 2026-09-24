"""可视化入口（P0 占位，P1/P2 实现）。

用法：
    python3 tools/visualize.py --config config/default.yaml --ckpt <ckpt> --out runs/vis
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DRL_PathPlan 回放 / 轨迹可视化（P0 占位）")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                        help="主配置路径（默认 config/default.yaml）")
    parser.add_argument("--spec", type=Path, default=Path("env/specs/scenarios_val.json"),
                        help="可视化所用场景 spec")
    parser.add_argument("--ckpt", type=Path, required=True,
                        help="策略权重路径")
    parser.add_argument("--out", type=Path, default=Path("runs/vis"),
                        help="输出目录（默认 runs/vis）")
    return parser


def main() -> None:
    build_parser().parse_args()
    # P0 仅校验参数；回放渲染与图表输出在 P1/P2 实现
    raise NotImplementedError("P1/P2")


if __name__ == "__main__":
    main()
