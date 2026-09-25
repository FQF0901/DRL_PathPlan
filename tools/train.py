#!/usr/bin/env python3
"""分阶段训练入口（P2/N5）：阶段 A/B/C 的真实 CLI。

契约（``.slim/deepwork/p2-contract.md`` §5）：``--config --stage {A,B,C} --spec --ckpt --out``。
本入口只做参数校验 + 默认值解析，训练编排全部委托给 ``pipeline.stages``（N4）：

- ``A`` = world model 教师强制训练（ego 条件 = 专家 GT 动作序列；目标 = ``(episode, step+k)``
  查表重建的未来 OD/LD；直接多步损失 + ego plan 噪声增强），产出 ``final.pt``（含 WM）；
- ``B`` = planner BC（primary→specific；动作主损失 + 小权重 rollout 轨迹辅助（WM 冻结 + detach）
  + router BCE），产出 ``final.pt``（策略快照）；
- ``C`` = PPO RL（``LqrTracker`` 闭环 + **阶段 B 快照** KL 锚（系数衰减）+ primary lr ×0.1 +
  WM 冻结/解冻）。

未识别的参数原样透传给 ``pipeline.stages``（如 ``--envs/--updates/--pool/--bc-dir/--bc-epochs``），
因此 ``tools/train.py`` 与 ``python -m pipeline.stages`` 等价，只是固定了入口与常用默认值。

用法::

    tools/venv-python tools/train.py --stage A --bc-dir runs/bc_expert_full \\
        --wm-epochs 10 --out runs/train/stage_a
    tools/venv-python tools/train.py --stage B --ckpt runs/train/stage_a/final.pt \\
        --bc-dir runs/bc_expert_full --bc-epochs 10 --out runs/train/stage_b
    tools/venv-python tools/train.py --stage C --ckpt runs/train/stage_b/final.pt \\
        --spec env/specs/scenarios_train_slice200.json --envs 2 --updates 20 --out runs/train/stage_c

import 时只有 stdlib（``pipeline.stages`` 延迟到 ``main()`` 内导入），``--help`` 无副作用。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 允许 `python tools/train.py` 直接运行（此时 sys.path[0] 是 tools/）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    """构造契约参数解析器（其余参数透传给 pipeline.stages）。"""
    parser = argparse.ArgumentParser(
        prog="tools/train.py",
        description="DRL_PathPlan 分阶段训练入口（A=BC+PPO，B=离线世界模型，C=联合微调）",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                        help="主配置路径（默认 config/default.yaml，includes 自动合并）")
    parser.add_argument("--stage", choices=("A", "B", "C"), required=True,
                        help="训练阶段：A=WM 教师强制；B=planner BC；C=PPO RL")
    parser.add_argument("--spec", type=Path, default=None,
                        help="训练场景 spec（阶段 C；默认取 train.yaml::data.spec）")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="初始化权重：B=阶段 A 产物；C=阶段 B 策略快照")
    parser.add_argument("--out", type=Path, default=Path("runs/train"),
                        help="输出目录（默认 runs/train）")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args, _extra = build_parser().parse_known_args(argv_list)

    if not args.config.is_file():
        print(f"[train] 配置不存在：{args.config}", file=sys.stderr)
        return 2
    if args.spec is not None and not args.spec.is_file():
        print(f"[train] spec 不存在：{args.spec}", file=sys.stderr)
        return 2
    if args.ckpt is not None and not args.ckpt.is_file():
        print(f"[train] ckpt 不存在：{args.ckpt}", file=sys.stderr)
        return 2

    # GL 修复：在 import metadrive/panda3d 之前预载 venv glvnd，并让 spawn worker 继承路径。
    # （即使本入口未经 tools/venv-python 启动也要生效；见 pipeline/gl_runtime.py）
    from pipeline.gl_runtime import ensure_gl_library_path

    ensure_gl_library_path(preload=True)

    try:
        from pipeline.stages import main as stages_main  # N4 阶段编排（延迟导入）
    except ImportError as exc:
        print(f"[train] 无法导入 pipeline.stages：{exc}", file=sys.stderr)
        return 3
    # 原始 argv 透传：契约参数 + N4 扩展参数（--envs/--updates/--pool/--bc-dir/--replay/...）
    return int(stages_main(argv_list) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
