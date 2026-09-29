#!/usr/bin/env python3
"""分阶段训练入口（P2/N5）：阶段 A/B/C 的真实 CLI。

契约（``.slim/deepwork/p2-contract.md`` §5）：``--config --stage {A,B,C} --spec --ckpt --out``。
本入口只做参数校验 + 默认值解析，训练编排全部委托给 ``pipeline.stages``（N4）：

- ``A`` = world model 教师强制训练（ego 条件 = 专家 GT 动作序列；目标 = ``(episode, step+k)``
  查表重建的未来 OD；直接多步损失 + plan head ``ego_next`` 监督 + ego plan 噪声增强），
  产出 ``final.pt``（含 WM）；
- ``B`` = planner BC（primary→specific；动作主损失 + 小权重 rollout 轨迹辅助（WM 冻结 + detach）
  + router 软目标 CE/KL），产出 ``final.pt``（策略快照）；
- ``C`` = PPO RL（``LqrTracker`` 闭环 + **阶段 B 快照** KL 锚（系数衰减）+ primary lr ×0.1 +
  WM 冻结/解冻；可训练范围 ``--trainable-scope``（R2/P0-6，默认 ``design`` allowlist =
  policy/value + MoE specific，"干净 PPO 基线"的设计口径；``all`` = 旧行为仅冻 st_gnn））。

未识别的参数原样透传给 ``pipeline.stages``（如 ``--envs/--updates/--pool/--bc-dir/--bc-epochs``），
因此 ``tools/train.py`` 与 ``python -m pipeline.stages`` 等价，只是固定了入口与常用默认值。

用法::

    tools/venv-python tools/train.py --stage A --bc-dir runs/bc_expert_full \\
        --wm-epochs 10 --out runs/train/stage_a
    tools/venv-python tools/train.py --stage B --ckpt runs/train/stage_a/final.pt \\
        --bc-dir runs/bc_expert_full --bc-epochs 10 --out runs/train/stage_b
    tools/venv-python tools/train.py --stage C --ckpt runs/train/stage_b/final.pt \\
        --spec env/specs/scenarios_train_slice200.json --envs 2 --updates 20 --out runs/train/stage_c
    tools/venv-python tools/train.py --phase3 datasets/BTC<ts>_dagger_r1 --phase3-round 1 \\
        --ckpt runs/<run>/stage_b/final.phase2.pt --out runs/<run>/stage_b   # 迭代恢复训练
    tools/venv-python tools/train.py --phase3-loop [--phase3-rounds N]       # 全自动循环（采集→训练→评测×N）
    tools/venv-python tools/train.py --phase3-chain [--phase3-only]          # A→B→循环全链（PHASE3=1 用）

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
    parser.add_argument("--stage", choices=("A", "B", "C"), default=None,
                        help="训练阶段：A=WM 教师强制；B=planner BC；C=PPO RL"
                             "（--phase3 模式固定属于阶段 B，可省略）")
    parser.add_argument("--spec", type=Path, default=None,
                        help="训练场景 spec（阶段 C；默认取 train.yaml::data.spec）")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="初始化权重：B=阶段 A 产物；C=阶段 B 策略快照；"
                             "phase3=阶段 B phase 2 final（缺省读 config stages.B.phase3.init_ckpt）")
    parser.add_argument("--phase3", type=Path, default=None,
                        help="stage B phase 3 迭代恢复训练（lane P3-B）：当轮 dagger 数据集目录"
                             "（单独使用：无 5k/无 worst-mild 权重/无 mining）；给出即启用本模式")
    parser.add_argument("--phase3-loop", action="store_true",
                        help="phase 3 全自动循环（lane P3-C）：采集→训练→评测 ×N + 护栏；"
                             "由 tools/train.sh 的 PHASE3=1/PHASE3_ONLY=1 调用（也可直接使用）")
    parser.add_argument("--phase3-chain", action="store_true",
                        help="phase 3 全链（lane P3-C）：A→B→循环；PHASE3_ONLY=1/--phase3-only 跳过 A/B")
    parser.add_argument("--out", type=Path, default=Path("runs/train"),
                        help="输出目录（默认 runs/train）")
    return parser


def main(argv: "list[str] | None" = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    args, _extra = build_parser().parse_known_args(argv_list)

    if args.phase3_chain:
        # lane P3-C：A→B→phase3 全链（run_config 解析 + 循环编排都在 pipeline.phase3_loop）
        from pipeline.phase3_loop import run_chain

        return int(run_chain(argv_list) or 0)
    if args.phase3_loop:
        # lane P3-C：仅循环编排（采集/训练/评测的路径与状态由 pipeline.phase3_loop 自解析）
        from pipeline import phase3_loop

        return int(phase3_loop.main(argv_list) or 0)
    if args.stage is None and args.phase3 is None:
        print("[train] 必须给 --stage {A,B,C} 或 --phase3 <dagger_dir>（stage B phase 3 迭代恢复训练）",
              file=sys.stderr)
        return 2
    if args.phase3 is not None and args.stage not in (None, "B"):
        print(f"[train] --phase3 固定属于阶段 B（可省略 --stage），收到 --stage {args.stage}", file=sys.stderr)
        return 2
    stage = args.stage if args.stage is not None else "B"
    if args.phase3 is not None and not args.phase3.is_dir():
        print(f"[train] --phase3 dagger 目录不存在：{args.phase3}", file=sys.stderr)
        return 2
    if not args.config.is_file():
        print(f"[train] 配置不存在：{args.config}", file=sys.stderr)
        return 2
    if args.spec is not None and not args.spec.is_file():
        print(f"[train] spec 不存在：{args.spec}", file=sys.stderr)
        return 2
    if args.ckpt is not None and not args.ckpt.is_file():
        print(f"[train] ckpt 不存在：{args.ckpt}", file=sys.stderr)
        return 2

    # 运行目录布局（代码默认，见 pipeline/run_paths.py）：logs/ + stage 目录 + manifest + 配置快照。
    # 非侵入：训练写盘仍由 pipeline.stages 按 --out 决定（monitor/ckpt 保持在 --out 内）。
    from pipeline import run_paths

    work_dir = run_paths.work_dir_of_out(args.out)
    run_paths.prepare_layout(work_dir, stage)
    run_paths.write_manifest(
        work_dir,
        stage=stage,
        config=args.config,
        model_config=run_paths.extra_value(argv_list, "--model-config") or "config/model.yaml",
        out=args.out,
        argv=["tools/train.py", *argv_list],
    )

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
