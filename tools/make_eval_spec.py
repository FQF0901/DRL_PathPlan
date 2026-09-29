#!/usr/bin/env python3
"""生成 eval500 / smoke16 spec（Lane V-Data）：只读源文件，spec 字段原样保留，wrapper 附 provenance。

D3 守卫（G2 记账）：默认 DRY-RUN 只打印不写盘；`--help` 由 argparse 直接退出。必须显式
`--write` 才落盘（防误触重生成 eval500 之类的事故）。
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT, NOW = Path(__file__).resolve().parents[1], datetime.now().isoformat(timespec="seconds")


def _stratified(src, total, seed):
    """labels.geometry 分层：原比例最大余数配额（并列按标签名序）+ 层内无放回（seed）。"""
    specs = json.loads((ROOT / src).read_text(encoding="utf-8"))["specs"]
    groups = {}
    for spec in specs:
        groups.setdefault(spec["labels"]["geometry"], []).append(spec)
    labels = sorted(groups)
    quota = {lab: total * len(groups[lab]) / len(specs) for lab in labels}
    take = {lab: int(v) for lab, v in quota.items()}
    for lab in sorted(labels, key=lambda l: (-(quota[l] - take[l]), l))[: total - sum(take.values())]:
        take[lab] += 1
    rng, picked = np.random.default_rng(seed), []
    for lab in labels:
        idx = sorted(rng.choice(len(groups[lab]), size=take[lab], replace=False).tolist())
        picked += [groups[lab][i] for i in idx]
    return picked, take

def _write(rel, src, specs, per_geo, note, *, write=True):
    prov = {"source": src, "seed": 0, "per_geometry": per_geo, "generated_at": NOW,
            **({"note": note} if note else {}),
            "rule": "labels.geometry 分层 / 原比例最大余数配额（并列按标签名序）/ 层内无放回"}
    payload = {"schema_version": 1, "count": len(specs), "provenance": prov, "specs": specs}
    if not write:
        print(f"[make_eval_spec] DRY-RUN {rel}: {len(specs)} 条 {per_geo}（未写盘；加 --write 才写）")
        return
    (ROOT / rel).write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"[make_eval_spec] {rel}: {len(specs)} 条 {per_geo}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/make_eval_spec.py",
        description="生成 eval500 / smoke16 spec（D3 守卫：默认 DRY-RUN，不写盘）",
    )
    parser.add_argument("--write", action="store_true",
                        help="显式确认写盘（缺省只预览；防误触重生成已冻结的 spec）")
    parser.add_argument("--dry-run", action="store_true",
                        help="显式 DRY-RUN 预览（缺省行为；与 --write 同给报错）")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.write and args.dry_run:
        parser.error("--write 与 --dry-run 互斥（--write 才会写盘）")
    write = bool(args.write)
    val, geo = _stratified("env/specs/scenarios_val.json", 500, 0)
    _write("env/specs/scenarios_eval500.json", "env/specs/scenarios_val.json（1000 条冻结验证集）",
           val, geo, "", write=write)
    smoke, sgeo = _stratified("env/specs/scenarios_train.json", 16, 0)
    _write("env/specs/scenarios_smoke16.json", "env/specs/scenarios_train.json", smoke, sgeo,
           "非协议：仅代码冒烟用", write=write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
