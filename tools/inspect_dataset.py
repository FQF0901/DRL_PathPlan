"""数据集抽检入口（P0 占位，P1/P2 实现）。

用法：
    python3 tools/inspect_dataset.py --config config/default.yaml --out runs/inspect

检查项（对应 config/README.md 验收标准 1）：
    帧间 ego 位移 / 航向与记录运动一致、物体速度有界、无时间反转。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DRL_PathPlan 数据集物理时间一致性抽检（P0 占位）")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                        help="主配置路径（默认 config/default.yaml）")
    parser.add_argument("--spec", type=Path, default=Path("env/specs/scenarios_train.json"),
                        help="对应场景 spec（用于定位数据）")
    parser.add_argument("--out", type=Path, default=Path("runs/inspect"),
                        help="报告输出目录（默认 runs/inspect）")
    return parser


def main() -> None:
    build_parser().parse_args()
    # P0 仅校验参数；抽检逻辑与报告输出在 P1/P2 实现
    raise NotImplementedError("P1/P2")


if __name__ == "__main__":
    main()
