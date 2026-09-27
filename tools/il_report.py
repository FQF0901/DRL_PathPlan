#!/usr/bin/env python3
"""IL 评审报告生成器（design-v1.2 §5.1）。

读取一次训练的 `monitor/metrics.csv`（长表 step,tag,value）与 `metrics.json`，
产出 `il_report.md`（可直接评审的表格 + 判定门）与 `il_report.json`（机器可读摘要）。

用法:
    tools/venv-python tools/il_report.py --run runs/train/<stage_b_run> [--stage-a runs/train/<stage_a_run>] \\
        --out runs/il_report

设计原则：**按 tag 家族模式发现**，缺什么就明确写 "缺失"，绝不编造；判定门只在数据齐备时给 PASS/FAIL。
tag 口径（2026-09-27 监控瘦身 v2）：新 run 用 `wm/*` / `val/...` / `ego/*` / `router/*` / `planner/*`
（见 `docs/metrics.md`）；瘦身 v1 名（`wm/od/*` / `stageB/*` / `val_*`）与更早的
`horizon/*` / `train/*_bc_*` / `val/*` 序列仍可读（回退）。
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


def find(series: Dict[str, Dict[int, float]], pattern: str) -> Dict[str, Dict[int, float]]:
    rx = re.compile(pattern)
    return {tag: val for tag, val in series.items() if rx.search(tag)}


def series_first(series: Dict[str, Dict[int, float]], *candidates: str) -> Dict[int, float]:
    """按候选顺序返回第一个存在的 tag 序列（新 tag 优先、旧 tag 回退）；都没有 → ``{}``。"""
    for tag in candidates:
        if tag and tag in series:
            return series[tag]
    return {}


#: Stage B 训练标量：瘦身 tag 候选（metrics.json 缺失时回退用；新名在前，v1/旧名回退）
_SLIM_TRAIN_SCALARS: Dict[str, Tuple[str, ...]] = {
    "loss": ("loss/planner/primary/total", "loss/planner/specific/total",
             "planner/primary/loss_terms/loss", "planner/specific/loss_terms/loss",
             "stageB/primary/loss_terms/loss", "stageB/specific/loss_terms/loss"),
    "traj_loss": ("loss/planner/primary/traj", "loss/planner/specific/traj",
                  "planner/primary/loss_terms/traj", "planner/specific/loss_terms/traj",
                  "stageB/primary/loss_terms/traj", "stageB/specific/loss_terms/traj"),
    "action_loss": ("loss/planner/primary/action", "loss/planner/specific/action",
                    "planner/primary/loss_terms/action", "planner/specific/loss_terms/action",
                    "stageB/primary/loss_terms/action", "stageB/specific/loss_terms/action"),
    "load_balance_loss": ("loss/planner/primary/load_balance", "loss/planner/specific/load_balance"),
    "action_err_weighted_mean": ("ego/action/err_weighted",),
    "traj_fde_m": ("ego/traj/fde_m",),
    "load_cv": ("router/load_cv",),
    "gate_entropy": ("router/gate_entropy",),
    "expert_load_0": ("router/expert_load/e0",),
}

#: Stage B 留出标量：瘦身 tag 候选（新名 `val/...` 在前，瘦身 v1 `val_*` 回退）
_SLIM_VAL_SCALARS: Dict[str, Tuple[str, ...]] = {
    "loss": ("val/loss/planner/primary/total", "val/loss/planner/specific/total",
             "val/planner/primary/loss_terms/loss", "val/planner/specific/loss_terms/loss",
             "val_stageB/primary/loss_terms/loss", "val_stageB/specific/loss_terms/loss"),
    "action_loss": ("val/loss/planner/primary/action", "val/loss/planner/specific/action",
                    "val/planner/primary/loss_terms/action", "val/planner/specific/loss_terms/action",
                    "val_stageB/primary/loss_terms/action", "val_stageB/specific/loss_terms/action"),
    "load_balance_loss": ("val/loss/planner/primary/load_balance", "val/loss/planner/specific/load_balance"),
    "traj_loss": ("val/loss/planner/primary/traj", "val/loss/planner/specific/traj",
                  "val/planner/primary/loss_terms/traj", "val/planner/specific/loss_terms/traj",
                  "val_stageB/primary/loss_terms/traj", "val_stageB/specific/loss_terms/traj"),
    "action_err_weighted_mean": ("val/ego/action/err_weighted", "val_ego/action/err_weighted"),
    "traj_fde_m": ("val/ego/traj/fde_m", "val_ego/traj/fde_m"),
    "load_cv": ("val/router/load_cv",),
    "gate_entropy": ("val/router/gate_entropy",),
    "expert_load_0": ("val/router/expert_load/e0",),
}


def is_slim_run(series: Dict[str, Dict[int, float]]) -> bool:
    """是否为新（lane B 瘦身）tag 口径的 run（含 v1 名回退识别）。"""
    return any(
        tag.startswith(("loss/", "val/loss/", "planner/", "ego/", "router/", "val/planner/",
                        "val/ego/", "val/router/", "val/od/", "val/ego_next/", "wm/", "stageB/",
                        "val_ego/", "val_router/", "val_stageB/"))
        for tag in series
    )


def latest_tag(series: Dict[str, Dict[int, float]], pattern: str) -> Optional[float]:
    """按 tag 正则取最近 ≤3 点均值；无匹配 → ``None``（缺失，不编造）。"""
    found = find(series, pattern)
    if not found:
        return None
    return latest(next(iter(found.values())))


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
    """Stage A 段落：**新（瘦身 v2）tag 优先，v1 名与旧 tag 回退**。

    - 新：``val/od/loss/h{k}``、``val/od/ade_m/h{k}``（+ ``cv_h{k}`` 匀速基线）、
      ``val/ego_next/loss/h{k}``、``wm/loss``、``val/od/presence_auc``、``val/od/entry_auc``；
    - 回退：瘦身 v1 ``wm/od/*`` / ``wm/presence_auc``；更早 ``horizon/h{k}/{loss,ade,cv_ade,fde,
      ego_next_loss}/mean`` + ``train/<scalar>``（旧 run 有 FDE / presence_loss / entry_loss）。
    """
    per_h: Dict[int, Dict[str, Dict[int, float]]] = {}
    for k in range(1, 7):
        entry = {
            "loss": series_first(series, f"val/od/loss/h{k}", f"wm/od/loss/h{k}",
                                 f"horizon/h{k}/loss/mean"),
            "ade": series_first(series, f"val/od/ade_m/h{k}", f"wm/od/ade_m/h{k}",
                                f"horizon/h{k}/ade/mean"),
            "cv_ade": series_first(series, f"val/od/ade_m/cv_h{k}", f"wm/od/ade_m/cv_h{k}",
                                   f"horizon/h{k}/cv_ade/mean"),
            "fde": series_first(series, f"val/od/fde_m/h{k}", f"horizon/h{k}/fde/mean"),
            "cv_fde": series_first(series, f"val/od/fde_m/cv_h{k}", f"horizon/h{k}/cv_fde/mean"),
        }
        if any(entry.values()):
            per_h[k] = entry
    lines: List[str] = ["### Stage A（世界模型，teacher forcing）", ""]
    if not per_h:
        lines.append("> 未发现逐 horizon 序列（新 tag `val/od/loss|ade_m/h*` 或旧 tag "
                     "`horizon/h*/.../mean`）：Stage A 未跑或未开启 grouped 监控。")
        return lines, {"available": False}
    slim = "val/od/ade_m/h1" in series or "wm/od/ade_m/h1" in series
    rows, gates = [], {}
    for k in sorted(per_h):
        m = per_h[k]
        ade, cv_ade = latest(m["ade"]), latest(m["cv_ade"])
        fde, cv_fde = latest(m["fde"]), latest(m["cv_fde"])
        rows.append([f"h{k}", num(ade), num(cv_ade), num(fde), num(cv_fde),
                     "PASS" if (ade is not None and cv_ade is not None and ade < cv_ade) else
                     ("FAIL" if ade is not None and cv_ade is not None else "缺失")])
        gates[f"h{k}"] = None if (ade is None or cv_ade is None) else bool(ade < cv_ade)
    lines += table(rows, ["horizon", "ADE", "CV ADE", "FDE", "CV FDE", "ADE<CV"])
    if slim:
        lines += ["", "> 监控瘦身（2026-09-27）：FDE / 有效样本计数已从曲线移除；ADE 与匀速基线"
                      "合并为 `val/od/ade_m`（`h*` / `cv_h*`）。见 `docs/metrics.md`。"]
    for label, candidates in (("wm_loss", ("wm/loss", "train/wm_loss")),
                              ("presence_auc", ("val/od/presence_auc", "wm/presence_auc",
                                                "train/presence_auc")),
                              ("entry_auc", ("val/od/entry_auc", "wm/entry_auc", "train/entry_auc")),
                              ("presence_loss", ("train/presence_loss",)),
                              ("entry_loss", ("train/entry_loss",))):
        found = series_first(series, *candidates)
        if found:
            tag = next((candidate for candidate in candidates if candidate in series), candidates[0])
            lines.append(f"- `{tag}` = {num(latest(found))}")
    summary = {"available": True, "gates": gates}
    if gates and all(v is not None for v in gates.values()):
        summary["all_horizons_beat_cv"] = all(gates.values())
        lines.append("")
        lines.append(f"**判定：所有 horizon 的 ADE 均优于匀速 → "
                     f"{'PASS' if summary['all_horizons_beat_cv'] else 'FAIL'}**")
    return lines, summary


def stage_b_section(
    series: Dict[str, Dict[int, float]], run_meta: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """Stage B 段落：**最终值优先读 metrics.json 的 ``primary`` 汇总**（新旧 run 都有），
    CSV 序列作为回退（新 tag 优先、旧 tag 回退）。

    新 tag（lane B）：``loss/planner/<phase>/{total,traj,action,router}``、
    ``ego/action/err_weighted``、``ego/traj/mae_m/h*``、``ego/traj/fde_m``、
    ``router/{ce,acc,acc_majority}``；留出同族进 ``val/`` 命名空间（``val/loss/planner/...`` /
    ``val/ego/...`` / ``val/router/...``）。瘦身 v1 名（``stageB/*`` / ``val_stageB/*`` /
    ``val_ego/*`` / ``val_router/*``）与更早 ``train/<phase>_bc_*`` / ``horizon|slice|label/*``
    自动回退；软目标 KL/温度/专家混合权重已删除（lane B B3）。
    median/p95、全部 slice/label、expert_util 已移除 → 明确写「已移除」。
    """
    lines: List[str] = ["### Stage B（规划器 BC）", ""]
    summary: Dict[str, Any] = {}
    primary_summary: Dict[str, Any] = {}
    if isinstance(run_meta, dict) and isinstance(run_meta.get("primary"), dict):
        primary_summary = run_meta["primary"]
    slim = is_slim_run(series)

    def scalar(name: str) -> Optional[float]:
        value = primary_summary.get(f"bc_{name}")
        if isinstance(value, (int, float)):
            return float(value)
        if slim:
            found = series_first(series, *_SLIM_TRAIN_SCALARS.get(name, ()))
            if found:
                return latest(found)
        s = find(series, rf"train/primary_bc_{name}$")
        if not s:
            return None
        return latest(next(iter(s.values())))

    action = {key: scalar(f"action_err_{key}") for key in
              ("mean", "weighted_mean", "median", "p95", "count", "weight")}
    summary["action_err"] = action
    lines.append("**动作误差（首步 ds/dθ）**")
    rows = [[key, ("已移除" if slim and key in ("median", "p95") and value is None else num(value))]
            for key, value in action.items()]
    lines += table(rows, ["metric", "value"])
    if slim:
        lines += ["", "> 监控瘦身（2026-09-27）：动作误差 median/p95 与全部 slice/label 切片已移除；"
                      "主口径 = `ego/action/err_weighted`（留出 `val/ego/action/err_weighted`）。"
                      "见 `docs/metrics.md`。"]
    slices: Dict[str, Optional[float]] = {}
    labels: Dict[str, Optional[float]] = {}
    if not slim:
        slices = {m.group(1): latest(s) for m, s in
                  ((re.search(r"_slice_(\w+)_weighted_mean$", t), v)
                   for t, v in find(series, r"train/primary_bc_action_err_slice_(\w+)_weighted_mean").items())}
        if slices:
            lines += table([[k, num(v)] for k, v in sorted(slices.items())], ["slice", "weighted action err"])
        # 逐标签动作误差：排除 `_count` 伴生 tag（旧实现里 `\w+$` 会把它当误差值，值≈样本数）
        labels = {re.search(r"_label_(\w+?)$", t).group(1): latest(v)
                  for t, v in find(series, r"train/primary_bc_action_err_label_\w+$").items()
                  if not t.endswith("_count")}
        if labels:
            lines += table([[k, num(v)] for k, v in sorted(labels.items())], ["label", "action err"])
    summary["slices"] = slices
    summary["labels"] = labels

    # 轨迹误差（逐 horizon，加权口径）：新 tag `ego/traj/mae_m/h{k}`（留出 `val/ego/...`）；
    # 瘦身 v1 `val_ego/...` / 旧 `train/primary_bc_traj_mae_h{k}_m` / `val/horizon/h{k}/...` 回退；
    # metrics.json（`bc_traj_mse_h{k}` / val `per_horizon`）为最后回退（MSE 只在此保留）。
    mae_h: Dict[int, Optional[float]] = {}
    mse_h: Dict[int, Optional[float]] = {}
    legacy_err: Dict[int, Optional[float]] = {}
    for k in range(1, 7):
        found = series_first(series, f"ego/traj/mae_m/h{k}", f"train/primary_bc_traj_mae_h{k}_m")
        mae_h[k] = latest(found) if found else None
        if mae_h[k] is None and isinstance(primary_summary.get(f"bc_traj_mae_h{k}_m"), (int, float)):
            mae_h[k] = float(primary_summary[f"bc_traj_mae_h{k}_m"])
        found = series_first(series, f"train/primary_bc_traj_mse_h{k}")
        mse_h[k] = latest(found) if found else None
        if mse_h[k] is None and isinstance(primary_summary.get(f"bc_traj_mse_h{k}"), (int, float)):
            mse_h[k] = float(primary_summary[f"bc_traj_mse_h{k}"])
        found = series_first(series, f"train/primary_bc_traj_err_h{k}")
        legacy_err[k] = latest(found) if found else None
    # 旧 run（2026-09-26 单位修正前）只有 traj_err_h{k}（l2 下实为加权 MSE，m²）→ 作为 MSE 回退
    if all(value is None for value in mse_h.values()):
        mse_h = {k: value for k, value in legacy_err.items() if value is not None}
    mae_estimated = False
    for k, mse in sorted(mse_h.items()):
        if mae_h.get(k) is None and mse is not None and mse >= 0.0:
            mae_h[k] = math.sqrt(mse)
            mae_estimated = True
    horizon_rows: List[List[Any]] = []
    ratios: List[float] = []
    for k in sorted(set(mae_h) | set(mse_h)):
        mae, mse = mae_h.get(k), mse_h.get(k)
        if mae is None and mse is None:
            continue
        horizon_rows.append([f"h{k}", num(mae), num(mse),
                             "" if mse is None else num(math.sqrt(mse))])
        # 换算校验只在 MAE/MSE 都是真实序列时有意义；估算的 MAE 与 sqrt(MSE) 恒等，不算校验。
        if not mae_estimated and mae and mae > 0 and mse is not None and mse >= 0:
            ratios.append(math.sqrt(mse) / mae)
    if horizon_rows:
        lines.append("**轨迹误差（逐 horizon，加权口径）**")
        if slim:
            lines.append("")
            lines.append("> `ego/traj/mae_m`（留出 `val/ego/traj/mae_m`）为唯一逐 horizon 曲线；"
                         "MSE 仅作损失口径保留在 metrics.json（不再是曲线，见 `docs/metrics.md`）。")
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

    # lane U3：去聚类后 router 只保留 MoE 负载 KPI（无簇标签 CE/acc；旧 run 缺失 → 显示缺失）
    router = {
        k: scalar(f"router_{k}")
        for k in ("expert_load_0", "load_cv", "gate_entropy")
    }
    lines.append("")
    lines.append("**MoE 负载（lane U1：无聚类监督；phase 1 MoE 关闭 → 缺失）**")
    lines += table([[k, num(v)] for k, v in router.items()], ["metric", "value"])
    lines.append("")
    lines.append("> `expert_load_0` = e0 门控权重质量占比；`load_cv` = 负载变异系数（0 = 均衡）；"
                 "`gate_entropy` = 逐 token 路由分布归一化熵（1 = 均匀）。"
                 "聚类硬标签 CE/acc 与二值门控指标已删除（lane U1/U3）。")
    summary.update({"router": router})

    # ---- 留出集（按 episode 留出；新 tag = val_*/新族；旧 run = val/*；缺失 → 明确「缺失」）----
    val_snapshot = primary_summary.get("val") if isinstance(primary_summary.get("val"), dict) else {}
    lines.append("")
    lines.append("**留出集（按 episode 留出；train/val 同族指标对照）**")

    def val_scalar(name: str) -> Optional[float]:
        value = val_snapshot.get(f"bc_{name}")
        if isinstance(value, (int, float)):
            return float(value)
        if slim:
            found = series_first(series, *_SLIM_VAL_SCALARS.get(name, ()))
            if found:
                return latest(found)
        return latest_tag(series, rf"val/primary_bc_{name}$")

    compare_names = {
        "bc_loss": "loss",
        "bc_traj_mse (m²)": "traj_mse",
        "bc_action_loss": "action_loss",
        "bc_action_err_weighted_mean": "action_err_weighted_mean",
        "bc_traj_fde_m": "traj_fde_m",
        "bc_load_cv": "load_cv",
        "bc_gate_entropy": "gate_entropy",
    }
    train_compare = {label: scalar(name) for label, name in compare_names.items()}
    val_compare = {label: val_scalar(name) for label, name in compare_names.items()}
    if all(value is None for value in val_compare.values()):
        lines.append("> 未发现留出集序列（旧 run 或 `--val-frac 0`）：训练曲线均为训练集数字，"
                     "需用修复后的 Stage B 重跑（与 Stage A 同 `--seed/--val-frac` 即同一批留出 episode）。")
        summary["val"] = {"available": False}
        return lines, summary
    lines += table([[label, num(train_compare[label]), num(val_compare[label])]
                    for label in compare_names], ["metric", "train", "val"])

    # 逐 horizon（val）：新 tag `val/ego/traj/mae_m/h{k}` → 瘦身 v1 `val_ego/...` →
    # 旧 `val/horizon/*` → metrics.json val 快照
    val_mae_h: Dict[int, Optional[float]] = {}
    val_mse_h: Dict[int, Optional[float]] = {}
    for k in range(1, 7):
        found = series_first(series, f"val/ego/traj/mae_m/h{k}", f"val_ego/traj/mae_m/h{k}",
                             f"val/horizon/h{k}/traj_mae_m/mean")
        if found:
            val_mae_h[k] = latest(found)
        found = series_first(series, f"val/horizon/h{k}/traj_mse_m2/mean")
        if found:
            val_mse_h[k] = latest(found)
    for key, item in (val_snapshot.get("per_horizon") or {}).items():
        if not isinstance(item, dict):
            continue
        index = int(str(key).lstrip("h"))
        val_mae_h.setdefault(index, item.get("traj_mae_m"))
        val_mse_h.setdefault(index, item.get("traj_mse_m2"))
    if val_mae_h or val_mse_h:
        lines += table([[f"h{k}", num(val_mae_h.get(k)), num(val_mse_h.get(k))]
                        for k in sorted(set(val_mae_h) | set(val_mse_h))],
                       ["horizon", "MAE (m)", "MSE (m²)"])

    # 关键切分（val）：旧 run 才有 slice/label（新 run 已按瘦身清单移除）
    val_slices: Dict[str, Optional[float]] = {}
    val_labels: Dict[str, Optional[float]] = {}
    if not slim:
        val_slices = {m.group(1): latest(s) for m, s in
                      ((re.search(r"val/slice/(\w+)/action_err/mean", tag), values)
                       for tag, values in find(series, r"val/slice/\w+/action_err/mean").items())}
        val_labels = {m.group(1): latest(s) for m, s in
                      ((re.search(r"val/label/(\w+)/action_err/mean", tag), values)
                       for tag, values in find(series, r"val/label/\w+/action_err/mean").items())}
    for name, stats in (val_snapshot.get("slices") or {}).items():
        if isinstance(stats, dict):
            val_slices.setdefault(str(name), stats.get("weighted_mean"))
    for name, stats in (val_snapshot.get("labels") or {}).items():
        if isinstance(stats, dict):
            val_labels.setdefault(str(name), stats.get("weighted_mean"))
    if val_slices:
        lines += table([[k, num(v)] for k, v in sorted(val_slices.items())],
                       ["slice", "val weighted action err"])
    if val_labels:
        lines += table([[k, num(v)] for k, v in sorted(val_labels.items())],
                       ["label", "val weighted action err"])
    summary["val"] = {
        "available": True,
        "compare": {"train": train_compare, "val": val_compare},
        "per_horizon": {
            f"h{k}": {"mae_m": val_mae_h.get(k), "mse_m2": val_mse_h.get(k)}
            for k in sorted(set(val_mae_h) | set(val_mse_h))
        },
        "slices": val_slices,
        "labels": val_labels,
        "action_err_count": val_scalar("action_err_count"),
        "action_err_weight": val_scalar("action_err_weight"),
    }
    return lines, summary


def meta_section(metrics_path: str) -> Tuple[List[str], Dict[str, Any]]:
    lines = ["### 运行元数据", ""]
    if not os.path.exists(metrics_path):
        lines.append("> 未找到 metrics.json。")
        return lines, {}
    meta = json.load(open(metrics_path))
    keys = ["stage", "kind", "bc_dir", "samples", "device", "epochs", "primary_epochs", "specific_epochs",
            "batch_size", "lr", "action_weight", "traj_aux_weight", "wm_detach",
            "load_balance_coef", "hard_weight", "mild_weight", "obs_fingerprint", "dataset",
            # 留出集（2026-09-26；旧 run 缺失 → 不显示）
            "val_frac", "train_frames", "val_frames", "val_episodes"]
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
    metrics_path = os.path.join(args.run, "metrics.json")
    meta_lines, meta = meta_section(metrics_path)
    run_meta: Dict[str, Any] = {}
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path) as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                run_meta = loaded
        except (OSError, json.JSONDecodeError):
            run_meta = {}

    lines: List[str] = ["# IL 评审报告", "",
                        f"- Stage B 运行：`{args.run}`",
                        f"- monitor 序列数：{len(series_b)}",
                        f"- Stage A 运行：`{args.stage_a or '（未提供）'}`", ""]
    payload: Dict[str, Any] = {"stage_b_run": args.run, "meta": meta,
                               "n_monitor_series": len(series_b)}

    b_lines, payload["stage_b"] = stage_b_section(series_b, run_meta)
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
              "- Stage B 自 2026-09-26 起按 episode 留出（与 Stage A 同 `--seed/--val-frac`）："
              "训练/留出曲线分别在 `train/<phase>_*` / `val/<phase>_*`；本报告「留出集」段落"
              "优先读 `val/*` 序列与 metrics.json 的 `primary.val` 快照，旧 run 无 val → 显示缺失；",
              "- 监控瘦身（2026-09-27）：新 run 只记录 OD/EGO loss+KPI 与 router loss+KPI，tag 为 "
              "`wm/*` / `val/...` / `ego/*` / `router/*` / `planner/*`（见 `docs/metrics.md`）；"
              "本报告对新 tag 优先读取、旧 tag（含瘦身 v1 名）自动回退，动作误差 median/p95、"
              "slice/label 等已移除项显示为「已移除」；",
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
