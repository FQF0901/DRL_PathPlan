"""场景 spec 生成命令行入口（由 ``tools/gene_env.sh`` 调用）。

用法
----
    tools/venv-python -m env.scenario.cli                       # 全量 10k/1k -> env/specs/
    tools/venv-python -m env.scenario.cli --slice 200           # P1a 切片 + 全量
    tools/venv-python -m env.scenario.cli --slice 200 --no-validate

输出文件
--------
- 全量：``scenarios_train.json`` / ``scenarios_val.json``
- 切片：``scenarios_train_slice{N}.json``（N 条）+ ``scenarios_val_slice{M}.json``
  （M = max(1, N//4)，与 L1a 冒烟 ``build_specs(200, 50, ...)`` 口径一致）

生成后默认惰性调用 L1b 的 ``env.scenario.validator`` 做实例化校验；该模块尚不存在时
只打印警告并跳过（不影响 spec 生成），``--no-validate`` 可显式关闭。
"""

import argparse
import sys
from pathlib import Path

from .generator import build_specs
from .spec import ScenarioSpec, save_specs

DEFAULT_N_TRAIN = 10_000
DEFAULT_N_VAL = 1_000
DEFAULT_TRAIN_SEED_START = 1_000
DEFAULT_VAL_SEED_START = 5_000_000
DEFAULT_SLICE = 200
SLICE_VAL_RATIO = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m env.scenario.cli",
        description="生成 MetaDrive 场景 spec（几何分层 + 难度分层 + 脚本化事件窗口）",
    )
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN, help=f"全量训练 spec 条数（默认 {DEFAULT_N_TRAIN}）")
    parser.add_argument("--n-val", type=int, default=DEFAULT_N_VAL, help=f"全量验证 spec 条数（默认 {DEFAULT_N_VAL}）")
    parser.add_argument(
        "--slice",
        type=int,
        nargs="?",
        const=DEFAULT_SLICE,
        default=None,
        help=f"额外生成 P1a 切片：训练 N 条 + 验证 max(1, N//{SLICE_VAL_RATIO}) 条（不带值时 N={DEFAULT_SLICE}）",
    )
    parser.add_argument("--workers", type=int, default=8, help="实例化校验并行进程数（默认 8）")
    parser.add_argument("--out-dir", type=Path, default=Path("env/specs"), help="spec 输出目录（默认 env/specs）")
    parser.add_argument(
        "--seed-range",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help=f"训练 seed 区间 [START, STOP)（默认 [{DEFAULT_TRAIN_SEED_START}, ...)）",
    )
    parser.add_argument(
        "--val-seed-range",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        default=None,
        help=f"验证 seed 区间 [START, STOP)（默认 [{DEFAULT_VAL_SEED_START}, ...)）",
    )
    parser.add_argument("--rng-seed", type=int, default=0, help="生成器随机主种子（默认 0）")
    parser.add_argument("--no-validate", action="store_true", help="跳过实例化校验")
    return parser


def _seed_ranges(args: argparse.Namespace) -> tuple[tuple[int, int], tuple[int, int]]:
    """解析 seed 区间：未显式给出时按需求条数自动扩容，切片与全量共用同一前缀。"""
    slice_n = args.slice if args.slice is not None else 0
    if args.slice is not None and args.slice < 1:
        raise ValueError(f"--slice 必须 >= 1，实际 {args.slice}")
    train_needed = max(args.n_train, slice_n)
    val_needed = max(args.n_val, 0 if args.slice is None else max(1, slice_n // SLICE_VAL_RATIO))
    if args.seed_range is not None:
        train_seeds = (args.seed_range[0], args.seed_range[1])
    else:
        train_seeds = (DEFAULT_TRAIN_SEED_START, DEFAULT_TRAIN_SEED_START + train_needed)
    if args.val_seed_range is not None:
        val_seeds = (args.val_seed_range[0], args.val_seed_range[1])
    else:
        val_seeds = (DEFAULT_VAL_SEED_START, DEFAULT_VAL_SEED_START + val_needed)
    return train_seeds, val_seeds


def _write(name: str, specs: list[ScenarioSpec], out_dir: Path) -> Path:
    path = out_dir / name
    save_specs(specs, path)
    print(f"[gene_env] 写入 {path}（{len(specs)} 条）")
    return path


def _validate_outputs(outputs: list[tuple[str, list[ScenarioSpec]]], workers: int, out_dir: Path) -> int:
    """惰性调用 L1b validator；模块缺失/未实现时跳过。返回失败文件数。"""
    try:
        # 延迟导入：L1b 的 validator 可能尚未实现，缺失时跳过校验而不是让生成失败。
        from .validator import validate_specs, write_report
    except (ImportError, NotImplementedError) as exc:
        print(f"[gene_env] 跳过实例化校验（validator 不可用：{exc}）")
        return 0
    failed = 0
    for name, specs in outputs:
        report = validate_specs(specs, workers=workers)
        report_path = out_dir / f"validation_{name}.json"
        write_report(report, str(report_path))
        n_failed = 0
        if isinstance(report, dict):
            for key in ("n_failed", "num_failed", "failed"):
                value = report.get(key)
                if isinstance(value, int) and not isinstance(value, bool):
                    n_failed = max(n_failed, value)
        print(f"[gene_env] 校验 {name}: {len(specs)} 条，失败 {n_failed}，报告 {report_path}")
        if n_failed:
            failed += 1
    return failed


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train_seeds, val_seeds = _seed_ranges(args)
        outputs: list[tuple[str, list[ScenarioSpec]]] = []

        # 全量 spec：文件名与 config/{train,eval}.yaml 的 data.spec 对齐。
        train_specs, val_specs = build_specs(args.n_train, args.n_val, train_seeds, val_seeds, rng_seed=args.rng_seed)
        outputs.append(("scenarios_train", train_specs))
        outputs.append(("scenarios_val", val_specs))
        _write("scenarios_train.json", train_specs, args.out_dir)
        _write("scenarios_val.json", val_specs, args.out_dir)

        # P1a 切片：独立分层（保证切片内每类几何也有覆盖），seed 取同一区间前缀。
        if args.slice is not None:
            slice_val = max(1, args.slice // SLICE_VAL_RATIO)
            slice_train_specs, slice_val_specs = build_specs(
                args.slice, slice_val, train_seeds, val_seeds, rng_seed=args.rng_seed
            )
            outputs.append((f"scenarios_train_slice{args.slice}", slice_train_specs))
            outputs.append((f"scenarios_val_slice{slice_val}", slice_val_specs))
            _write(f"scenarios_train_slice{args.slice}.json", slice_train_specs, args.out_dir)
            _write(f"scenarios_val_slice{slice_val}.json", slice_val_specs, args.out_dir)
    except ValueError as exc:
        print(f"[gene_env] 错误: {exc}", file=sys.stderr)
        return 2

    if args.no_validate:
        print("[gene_env] --no-validate：跳过实例化校验")
        return 0
    return 1 if _validate_outputs(outputs, args.workers, args.out_dir) else 0


if __name__ == "__main__":
    raise SystemExit(main())
