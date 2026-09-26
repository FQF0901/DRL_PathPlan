#!/usr/bin/env python3
"""IL 评审报告生成器（design-v1.2 §5.1）。

读取一次训练的 `monitor/metrics.csv`（长表 step,tag,value）与 `metrics.json`，
产出 `il_report.md`（可直接评审的表格 + 判定门）与 `il_report.json`（机器可读摘要）。

用法:
    tools/venv-python tools/il_report.py --run runs/train/<stage_b_run> [--stage-a runs/train/<stage_a_run>] \\
        --out runs/il_report

设计原则：**按 tag 家族模式发现**，缺什么就明确写 "缺失"，绝不编造；判定门只在数据齐备时给 PASS/FAIL。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------- 基础读取
def load_monitor(path: str) -> Dict[str, Dict[int, float]]:
    series: Dict[str, Dict[int, float]] = defaultdict(dict)
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                series[row["tag"]][int(float(row["step"]))] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
    return series


def latest(series: Dict[int, float], n: int = 3) -> Optional[float]:
    if not series:
        return None
    steps = sorted(series)[-n:]
    return sum(series[s] for s in steps) / len(steps)


def last(series: Dict[int, float]) -> Optional[Tuple[int, float]]:
    if not series:
        return None
    step = max(series)
    return step, series[step]


def find(series: Dict[str, Dict[int, float]], pattern: str) -> Dict[str, Dict[int, float]]:
    rx = re.compile(pattern)
    return {tag: val for tag, val in series.items() if rx.search(tag)}


def table(rows: List[List[Any]], header: List[str]) -> List[str]:
    out = ["| " + " | ".join(str(c) for c in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in row) + " |")
    return out


def num(value: Optional[float], nd: int = 4) -> str:
    return "缺失" if value is None else f"{value:.{nd}f}"


# ---------------------------------------------------------------- 各段落
def stage_a_section(series: Dict[str, Dict[int, float]]) -> Tuple[List[str], Dict[str, Any]]:
    per_h = defaultdict(dict)  # k -> metric -> tag series
    for tag, s in find(series, r"horizon/h(\d+)/(\w+)/mean").items():
        match = re.search(r"horizon/h(\d+)/(\w+)/mean", tag)
        per_h[int(match.group(1))][match.group(2)] = s
    lines: List[str] = ["### Stage A（世界模型，teacher forcing）", ""]
    if not per_h:
        lines.append("> 未发现逐 horizon 序列（`horizon/h*/…/mean`）：Stage A 未跑或未开启 grouped 监控。")
        return lines, {"available": False}
    rows, gates = [], {}
    for k in sorted(per_h):
        m = per_h[k]
        ade, cv_ade = latest(m.get("ade", {})), latest(m.get("cv_ade", {}))
        fde, cv_fde = latest(m.get("fde", {})), latest(m.get("cv_fde", {}))
        valid = last(m.get("valid_count", {}))
        rows.append([f"h{k}", num(ade), num(cv_ade), num(fde), num(cv_fde),
                     "PASS" if (ade is not None and cv_ade is not None and ade < cv_ade) else
                     ("FAIL" if ade is not None and cv_ade is not None else "缺失"),
                     "" if valid is None else int(valid[1])])
        gates[f"h{k}"] = None if (ade is None or cv_ade is None) else bool(ade < cv_ade)
    lines += table(rows, ["horizon", "ADE", "CV ADE", "FDE", "CV FDE", "ADE<CV", "valid_count"])
    for name in ("presence_loss", "presence_auc", "entry_loss", "entry_auc"):
        s = find(series, rf"train/{name}$")
        if s:
            tag, val = next(iter(s.items()))
            lines.append(f"- `{tag}` = {num(latest(val))}")
    summary = {"available": True, "gates": gates}
    if gates and all(v is not None for v in gates.values()):
        summary["all_horizons_beat_cv"] = all(gates.values())
        lines.append("")
        lines.append(f"**判定：所有 horizon 的 ADE 均优于匀速 → "
                     f"{'PASS' if summary['all_horizons_beat_cv'] else 'FAIL'}**")
    return lines, summary


def stage_b_section(series: Dict[str, Dict[int, float]]) -> Tuple[List[str], Dict[str, Any]]:
    lines: List[str] = ["### Stage B（规划器 BC）", ""]
    summary: Dict[str, Any] = {}

    def scalar(name: str) -> Optional[float]:
        s = find(series, rf"train/primary_bc_{name}$")
        if not s:
            return None
        return latest(next(iter(s.values())))

    action = {key: scalar(f"action_err_{key}") for key in
              ("mean", "weighted_mean", "median", "p95", "count", "weight")}
    summary["action_err"] = action
    lines.append("**动作误差（首步 ds/dθ）**")
    lines += table([[k, num(v)] for k, v in action.items()], ["metric", "value"])
    slices = {m.group(1): latest(s) for m, s in
              ((re.search(r"_slice_(\w+)_weighted_mean$", t), v)
               for t, v in find(series, r"train/primary_bc_action_err_slice_(\w+)_weighted_mean").items())}
    if slices:
        lines += table([[k, num(v)] for k, v in sorted(slices.items())], ["slice", "weighted action err"])
    summary["slices"] = slices

    labels = {re.search(r"_label_(\w+?)(?:_count)?$", t).group(1): latest(v)
              for t, v in find(series, r"train/primary_bc_action_err_label_\w+$").items()}
    if labels:
        lines += table([[k, num(v)] for k, v in sorted(labels.items())], ["label", "action err"])
    summary["labels"] = labels

    horizon = {int(re.search(r"_traj_err_h(\d+)$", t).group(1)): latest(v)
               for t, v in find(series, r"train/primary_bc_traj_err_h\d+$").items()}
    if horizon:
        lines += table([[f"h{k}", num(v)] for k, v in sorted(horizon.items())],
                       ["traj horizon", "error (m)"])
    summary["traj_per_horizon"] = horizon

    router_soft = {k: scalar(f"router_soft_{k}") for k in ("ce", "kl", "placeholder")}
    router = {k: scalar(f"router_{k}") for k in ("top1_cluster_acc", "nmi", "entropy")}
    experts_w = {int(re.search(r"_(\d+)$", t).group(1)): latest(v) for t, v in
                 find(series, r"train/primary_bc_router_expert_mix_weight_\d+$").items()}
    experts_u = {int(re.search(r"_(\d+)$", t).group(1)): latest(v) for t, v in
                 find(series, r"train/primary_bc_router_expert_mix_util_\d+$").items()}
    lines.append("")
    lines.append("**路由（聚类软目标）**")
    lines += table([[k, num(v)] for k, v in {**router_soft, **router}.items()], ["metric", "value"])
    if experts_w:
        lines += table([[i, num(experts_w.get(i)), num(experts_u.get(i))] for i in sorted(experts_w)],
                       ["expert", "mix weight", "mix util"])
    cluster = {k: scalar(f"router_cluster_{k}") for k in ("version_num", "k")}
    summary.update({"router_soft": router_soft, "router": router,
                    "expert_mix_weight": experts_w, "expert_mix_util": experts_u,
                    "cluster": cluster})
    placeholder = router_soft.get("placeholder")
    if placeholder is not None:
        lines.append("")
        lines.append(f"**判定：router 软目标非占位 → "
                     f"{'PASS' if placeholder == 0 else 'FAIL（仍为占位实现）'}**")
        summary["soft_targets_real"] = (placeholder == 0)
    return lines, summary


def meta_section(metrics_path: str) -> Tuple[List[str], Dict[str, Any]]:
    lines = ["### 运行元数据", ""]
    if not os.path.exists(metrics_path):
        lines.append("> 未找到 metrics.json。")
        return lines, {}
    meta = json.load(open(metrics_path))
    keys = ["stage", "kind", "bc_dir", "samples", "device", "epochs", "primary_epochs", "specific_epochs",
            "batch_size", "lr", "action_weight", "traj_aux_weight", "router_coef", "wm_detach",
            "cluster_version", "cluster_k", "cluster_config", "obs_fingerprint", "dataset"]
    rows = [[k, str(meta[k])[:120]] for k in keys if k in meta]
    lines += table(rows, ["key", "value"]) if rows else ["（无匹配字段）"]
    return lines, {k: meta[k] for k in keys if k in meta}


def main() -> int:
    ap = argparse.ArgumentParser(description="IL report (design-v1.2 §5.1)")
    ap.add_argument("--run", required=True, help="Stage B 运行目录（含 monitor/metrics.csv）")
    ap.add_argument("--stage-a", default=None, help="可选：Stage A 运行目录")
    ap.add_argument("--out", default=None, help="输出目录（默认 <run>/il_report）")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.run, "il_report")
    os.makedirs(out_dir, exist_ok=True)

    series_b = load_monitor(os.path.join(args.run, "monitor", "metrics.csv"))
    meta_lines, meta = meta_section(os.path.join(args.run, "metrics.json"))

    lines: List[str] = ["# IL 评审报告", "",
                        f"- Stage B 运行：`{args.run}`",
                        f"- monitor 序列数：{len(series_b)}",
                        f"- Stage A 运行：`{args.stage_a or '（未提供）'}`", ""]
    payload: Dict[str, Any] = {"stage_b_run": args.run, "meta": meta,
                               "n_monitor_series": len(series_b)}

    b_lines, payload["stage_b"] = stage_b_section(series_b)
    lines += b_lines
    lines.append("")
    lines += meta_lines

    if args.stage_a:
        series_a = load_monitor(os.path.join(args.stage_a, "monitor", "metrics.csv"))
        lines.append("")
        a_lines, payload["stage_a"] = stage_a_section(series_a)
        lines += a_lines

    lines += ["", "## 备注", "",
              "- 闭环 KPI（50 条 LQR slice）由 `tools/test.sh` 单独评测，不在本报告内；",
              "- 判定门口径见 `docs/design-v1.2.md` §5.1；缺失项显示为「缺失」而不是默认通过。",
              f"- 生成时间：{__import__('datetime').datetime.now().isoformat(timespec='seconds')}"]

    md_path = os.path.join(out_dir, "il_report.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "il_report.json"), "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    print(f"wrote {md_path}")
    print(f"wrote {os.path.join(out_dir, 'il_report.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
