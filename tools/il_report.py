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
import math
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


def wilson_text(interval: Any) -> str:
    if (
        isinstance(interval, (list, tuple))
        and len(interval) == 2
        and all(isinstance(item, (int, float)) for item in interval)
    ):
        return f"[{interval[0]:.3f}, {interval[1]:.3f}]"
    return "缺失"


def closed_loop_section(eval_paths: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """可选：把冻结集闭环评测单独列出（design-v1.2 §5.1 第 6 条「单独列」）。

    读取 ``tools/test.sh`` 产物 ``<out>/<name>/metrics.json``（``--eval`` 可给目录或 json
    路径，可重复）。重点显示速度真均值/末步速度/低速（crawl）指标与动作口径；
    **不参与 IL 判定门**，缺失项显示「缺失」而不是默认通过。
    """
    lines: List[str] = ["### 闭环评测（单独列；不进 IL 判定门）", ""]
    payload: List[Dict[str, Any]] = []
    for raw in eval_paths:
        metrics_path = os.path.join(raw, "metrics.json") if os.path.isdir(raw) else raw
        if not os.path.exists(metrics_path):
            lines.append(f"- `{raw}`：未找到 metrics.json（缺失）")
            continue
        try:
            with open(metrics_path) as handle:
                doc = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            lines.append(f"- `{raw}`：metrics.json 读取失败（{type(exc).__name__}: {exc}）")
            continue
        meta: Dict[str, Any] = doc.get("meta") or {}
        overall: Dict[str, Any] = doc.get("overall") or {}
        verdict: Dict[str, Any] = doc.get("verdict") or {}
        name = os.path.basename(os.path.dirname(os.path.abspath(metrics_path)))
        rows = [
            ["n / n_error", f"{overall.get('n', '缺失')} / {overall.get('n_error', '缺失')}"],
            ["success（Wilson 95%）",
             f"{num(overall.get('success_rate'))} {wilson_text(overall.get('success_wilson'))}"],
            ["collision / off_road",
             f"{num(overall.get('collision_rate'))} / {num(overall.get('off_road_rate'))}"],
            ["speed_ratio_mean", num(overall.get("speed_ratio_mean"))],
            ["mean_speed_mps（逐 step 真均值）", num(overall.get("mean_speed_mps"))],
            ["final_speed_mps（末步；旧 mean_speed_mps 口径）", num(overall.get("final_speed_mps"))],
            ["low_speed_step_ratio（v < 2 m/s 步占比）", num(overall.get("low_speed_step_ratio"))],
            ["crawl_seconds（v < 2 m/s 累计秒数）", num(overall.get("crawl_seconds"))],
            ["steer_abs_mean / throttle_mean",
             f"{num(overall.get('steer_abs_mean'))} / {num(overall.get('throttle_mean'))}"],
            ["action_ds_mean_m / action_dtheta_abs_mean_rad",
             f"{num(overall.get('action_ds_mean_m'))} / {num(overall.get('action_dtheta_abs_mean_rad'))}"],
            ["verdict",
             f"all_passed={verdict.get('all_passed', '缺失')}（judged={verdict.get('n_checks_judged', '?')}）"],
        ]
        lines.append(
            f"**{name}**（policy={meta.get('policy', '?')}, tracker={meta.get('tracker') or '—'}, "
            f"ckpt={meta.get('ckpt') or '—'}）"
        )
        lines += table(rows, ["metric", "value"])
        if "final_speed_mps" not in overall:
            lines.append("")
            lines.append(
                "> 旧产物（2026-09-26 指标修复前）：`mean_speed_mps` 实为末步速度均值；"
                "`final_speed_mps`/crawl/动作字段缺失（需用修复后的 `tools/test.sh` 重跑）。"
            )
        if meta.get("policy") == "ckpt" and str(meta.get("tracker") or "exact") == "exact":
            lines.append("")
            lines.append(
                "> ckpt+exact 为运动学执行：无 `[steer, throttle]` 动作概念，`steer_abs_mean`/"
                "`throttle_mean` = N/A（NaN）；动作幅度看 `action_ds_mean_m`/"
                "`action_dtheta_abs_mean_rad`（m / rad）。"
            )
        lines.append("")
        payload.append({
            "name": name,
            "metrics_path": metrics_path,
            "policy": meta.get("policy"),
            "tracker": meta.get("tracker"),
            "ckpt": meta.get("ckpt"),
            "overall": {
                key: overall.get(key)
                for key in (
                    "n", "n_error", "success_rate", "success_wilson", "collision_rate",
                    "off_road_rate", "speed_ratio_mean", "mean_speed_mps", "final_speed_mps",
                    "low_speed_step_ratio", "crawl_seconds", "steer_abs_mean", "throttle_mean",
                    "action_ds_mean_m", "action_dtheta_abs_mean_rad",
                )
            },
            "verdict": {"all_passed": verdict.get("all_passed"),
                        "n_checks_judged": verdict.get("n_checks_judged")},
        })
    return lines, payload


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

    # 轨迹误差（2026-09-26 单位修正）：新键 traj_mae_h{k}_m（加权 MAE，m）与
    # traj_mse_h{k}（加权 MSE，m²）。旧 run 只有 traj_err_h{k}（loss_type=l2 下实为加权
    # MSE，m²）→ 回退：MSE 原样展示，MAE 用 sqrt(MSE) 近似（Jensen 上界，≥ 真 MAE）。
    def per_horizon(pattern: str) -> Dict[int, Optional[float]]:
        # 锚定 primary 相位（specific 同 tag 后缀，混入会按 CSV 顺序覆盖）
        return {int(re.search(pattern, t).group(1)): latest(v)
                for t, v in find(series, pattern).items()}

    mae_h = per_horizon(r"train/primary_bc_traj_mae_h(\d+)_m$")
    mse_h = per_horizon(r"train/primary_bc_traj_mse_h(\d+)$")
    legacy_err = per_horizon(r"train/primary_bc_traj_err_h(\d+)$")
    mae_estimated = False
    if not mse_h and legacy_err:
        mse_h = dict(legacy_err)
    for k, mse in sorted(mse_h.items()):
        if k not in mae_h and mse is not None and mse >= 0.0:
            mae_h[k] = math.sqrt(mse)
            mae_estimated = True
    horizon_rows: List[List[Any]] = []
    ratios: List[float] = []
    for k in sorted(set(mae_h) | set(mse_h)):
        mae, mse = mae_h.get(k), mse_h.get(k)
        horizon_rows.append([f"h{k}", num(mae), num(mse),
                             "" if mse is None else num(math.sqrt(mse))])
        # 换算校验只在 MAE/MSE 都是真实序列时有意义；估算的 MAE 与 sqrt(MSE) 恒等，不算校验。
        if not mae_estimated and mae and mae > 0 and mse is not None and mse >= 0:
            ratios.append(math.sqrt(mse) / mae)
    if horizon_rows:
        lines.append("**轨迹误差（逐 horizon，加权口径）**")
        if mae_estimated:
            lines.append("")
            lines.append("> 旧 run 无 MAE 序列：MAE 列由 sqrt(MSE) 近似（Jensen 上界，≥ 真 MAE）。")
        lines += table(horizon_rows, ["horizon", "MAE (m)", "MSE (m²)", "sqrt(MSE) (m)"])
        if ratios:
            lines.append("")
            lines.append(f"- 同 ckpt 换算校验：sqrt(MSE)/MAE ∈ [{min(ratios):.3f}, {max(ratios):.3f}]"
                         "（Jensen：≥1；=1 仅当误差幅度齐一，故不能用 sqrt(MSE) 当 MAE 读）")
    # 全局轨迹标量：旧 run 的 bc_traj_mae_m 实为**未加权 MSE（m²）**（2026-09-26 修正前语义），
    # 检测到旧口径时迁移到 bc_traj_mse_unweighted 展示，避免把 m² 读成 m。
    traj_global = {key: scalar(key) for key in
                   ("traj_mse", "traj_mae_m", "traj_mae_all_m", "traj_mse_unweighted")}
    legacy_global_mae = scalar("traj_mae_m")
    legacy_global_semantics = (
        traj_global["traj_mse"] is None
        and traj_global["traj_mae_all_m"] is None
        and legacy_global_mae is not None
    )
    if legacy_global_semantics:
        traj_global["traj_mse_unweighted"] = legacy_global_mae
        traj_global["traj_mae_m"] = None
    global_rows: List[List[Any]] = []
    if traj_global["traj_mse"] is not None:
        global_rows.append(["bc_traj_mse", f"{num(traj_global['traj_mse'])} m²（加权）"])
    if traj_global["traj_mae_m"] is not None:
        global_rows.append(["bc_traj_mae_m", f"{num(traj_global['traj_mae_m'])} m（加权）"])
    elif legacy_global_semantics:
        global_rows.append([
            "bc_traj_mae_m",
            "缺失（旧 run 该键实为未加权 MSE，口径见下行）",
        ])
    if traj_global["traj_mae_all_m"] is not None:
        global_rows.append([
            "bc_traj_mae_all_m",
            f"{num(traj_global['traj_mae_all_m'])} m（未加权，含被过滤帧，诊断）",
        ])
    if traj_global["traj_mse_unweighted"] is not None:
        global_rows.append([
            "bc_traj_mse_unweighted",
            f"{num(traj_global['traj_mse_unweighted'])} m²（未加权，旧 bc_traj_mae_m 口径 alias）",
        ])
    if global_rows:
        lines.append("")
        lines.append("**轨迹误差（全局）**")
        lines += table(global_rows, ["metric", "value"])
    horizon_summary = {k: {"mae_m": mae_h.get(k), "mse_m2": mse_h.get(k),
                           "sqrt_mse_m": (None if mse_h.get(k) is None else math.sqrt(mse_h[k]))}
                       for k in sorted(set(mae_h) | set(mse_h))}
    summary["traj_per_horizon"] = horizon_summary
    summary["traj_per_horizon_legacy"] = legacy_err  # 旧键原值（l2→MSE m² / l1→MAE m）
    summary["traj_global"] = traj_global
    summary["traj_global_legacy_bc_traj_mae_m"] = legacy_global_mae if legacy_global_semantics else None
    summary["traj_mae_estimated_from_mse"] = mae_estimated

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
    ap.add_argument("--eval", action="append", default=None,
                    help="可选：闭环评测目录（含 metrics.json）或 json 路径，可重复；"
                         "仅单独列出（速度/crawl/动作口径），不进 IL 判定门")
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

    if args.eval:
        lines.append("")
        eval_lines, payload["closed_loop"] = closed_loop_section(list(args.eval))
        lines += eval_lines

    lines += ["", "## 备注", "",
              "- 闭环 KPI（50 条 LQR slice）由 `tools/test.sh` 单独评测；用本脚本 "
              "`--eval <runs/eval/<name>>` 可单独列出（速度/crawl/动作口径），不进 IL 判定门；",
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
