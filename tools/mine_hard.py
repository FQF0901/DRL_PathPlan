#!/usr/bin/env python3
"""worst-50% 权重挖掘 CLI（lane U1）：冻结 primary → 逐行 IL 误差 → 行权重 sidecar。

用法::

    tools/venv-python tools/mine_hard.py --ckpt runs/train/stage_b/primary.pt \\
        --bc-dir datasets/BTC<ts>_expert5k --out runs/train/stage_b/weight_sidecar.npz

口径（用户定稿）：主键 = 动作加权 IL 误差 ``w·mean|μ−专家首步动作|``（w = train_weight×配平），
worst = top-50% 确定性选择（并列按未加权误差→行号升序）；行权重 = worst→``--hard-weight``（1.0）/
其余→``--mild-weight``（0.1），与 train_weight×balance_weight 相乘（全量曝光）；sidecar 记录参照
ckpt sha256 / 数据集指纹 / seed，训练侧只读 + 严格校验（见 ``pipeline.hard_mining``）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/mine_hard.py",
        description="冻结 primary 的 worst-50% 权重挖掘（动作加权 IL 误差）→ 行权重 sidecar npz",
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="冻结 primary 权重（.pt）")
    parser.add_argument("--bc-dir", type=str, required=True, help="BC 数据集目录（expert_bc.npz）")
    parser.add_argument("--out", type=Path, required=True, help="输出权重 sidecar npz（weight_sidecar.npz）")
    parser.add_argument("--hard-frac", type=float, default=0.5, help="worst 行占比（默认 0.5 = top-50 pct）")
    parser.add_argument("--hard-weight", type=float, default=1.0, help="worst 行权重（默认 1.0）")
    parser.add_argument("--mild-weight", type=float, default=0.1, help="其余行权重（默认 0.1）")
    parser.add_argument("--batch-size", type=int, default=256, help="逐行前向 batch")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--seed", type=int, default=0, help="记录到 sidecar（选择本身是确定性的）")
    parser.add_argument("--limit-dataset", type=int, default=None, help="调试：按 episode 前缀截断")
    parser.add_argument("--model-config", type=str, default="config/model.yaml", help="模型结构 yaml")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="主配置（线程/设备）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from pipeline.hard_mining import mine_hard_rows, row_il_errors, write_weight_sidecar
    from pipeline.stages import build_model, load_config
    from pipeline.trainer import BCDataset, apply_thread_limits, load_checkpoint

    config = load_config(str(args.config))
    apply_thread_limits(workers=1, config=config)
    if not Path(args.ckpt).is_file():
        print(f"[mine_hard] ckpt 不存在：{args.ckpt}", file=sys.stderr)
        return 2
    model = build_model(load_config(str(args.model_config)))
    meta = load_checkpoint(str(args.ckpt), model)
    print(
        f"[mine_hard] 载入 {args.ckpt}（missing={len(meta.get('missing_keys', []))}）"
        f" · 冻结参照 primary",
        flush=True,
    )
    dataset = BCDataset.load(str(args.bc_dir), limit=args.limit_dataset)
    errors = row_il_errors(
        model, dataset, device=args.device, batch_size=int(args.batch_size), logger=print
    )
    mined = mine_hard_rows(
        errors["err_l1"], errors["err_weighted"], hard_frac=float(args.hard_frac)
    )
    out = write_weight_sidecar(
        args.out,
        worst=mined["worst"],
        err_l1=errors["err_l1"],
        err_weighted=errors["err_weighted"],
        weight=errors["weight"],
        dataset_dir=str(args.bc_dir),
        dataset_meta=dataset.meta,
        ckpt=str(args.ckpt),
        seed=int(args.seed),
        hard_frac=float(args.hard_frac),
        hard_weight=float(args.hard_weight),
        mild_weight=float(args.mild_weight),
        threshold=float(mined["threshold"]),
        order_rule=str(mined["order_rule"]),
    )
    print(
        f"[mine_hard] DONE → {out}（worst={mined['hard_rows']}/{mined['total_rows']}"
        f" = {mined['hard_rows'] / max(1, mined['total_rows']):.1%} · "
        f"hard_weight={args.hard_weight} · mild_weight={args.mild_weight} · "
        f"threshold(err_weighted)={mined['threshold']:.6f}）",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
