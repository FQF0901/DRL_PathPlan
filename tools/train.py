"""分阶段训练入口（P0 占位，P1/P2 实现）。

用法：
    python3 tools/train.py --config config/default.yaml --stage A
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DRL_PathPlan 训练入口（P0 占位）")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                        help="主配置路径（默认 config/default.yaml）")
    parser.add_argument("--stage", choices=["A", "B", "C"], required=True,
                        help="训练阶段：A=BC 预热+PPO，B=离线世界模型，C=联合微调")
    parser.add_argument("--spec", type=Path, default=None,
                        help="训练场景 spec（默认取 train.yaml 中 data.spec）")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="续训 / 初始化权重路径")
    parser.add_argument("--out", type=Path, default=Path("runs/train"),
                        help="输出目录（默认 runs/train）")
    return parser


def main() -> None:
    build_parser().parse_args()
    # P0 仅校验参数；配置加载、环境构建与训练循环在 P1/P2 实现
    raise NotImplementedError("P1/P2")


if __name__ == "__main__":
    main()
