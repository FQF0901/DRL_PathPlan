#!/usr/bin/env python3
"""离线曲线绘图（不依赖 tensorboard；matplotlib Agg → PNG）。

用法::

    tools/venv-python tools/plot_curves.py \
        --stage-a runs/train/<runA>/stage_a --stage-b runs/train/<runB>/stage_b \
        --out runs/train/<runB>/plots

读取各 stage 的 ``monitor/metrics.csv``（长表 step,tag,value），按预设"面板"出图：

- Stage A：`A_horizon_loss.png`（逐 horizon loss）、`A_ade_vs_cv.png`（逐 horizon ADE vs 匀速）、
  `A_health.png`（presence/entry AUC + grad_norm）、`A_wm_terms.png`（wm_loss_*）
- Stage B：`B_loss_terms.png`（bc/traj/action/router，train vs val）、`B_action_err.png`
  （mean/median/p95 + 分切片，train vs val）、`B_traj_horizon.png`（逐 horizon MAE）、
  `B_router.png`（CE/KL/entropy/top1/NMI + 专家混合权重）

缺失的 tag 自动跳过；每个面板一个 PNG。tag 命名兼容 2026-09-26 前后的两套（`/count`→`/n_updates`、
`valid_count`→`valid_samples`、`train/per_horizon/*` 已移除）。
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_monitor(stage_dir: str) -> Dict[str, Dict[int, float]]:
    path = os.path.join(stage_dir, "monitor", "metrics.csv")
    series: Dict[str, Dict[int, float]] = defaultdict(dict)
    if not os.path.exists(path):
        print(f"[plot] 缺少 {path}")
        return series
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                series[row["tag"]][int(float(row["step"]))] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
    return series


def pick(series: Dict[str, Dict[int, float]], *candidates: str) -> Optional[Tuple[str, Dict[int, float]]]:
    """按候选顺序返回第一个存在的 tag。"""
    for tag in candidates:
        if tag in series:
            return tag, series[tag]
    return None


def short(tag: str) -> str:
    return tag.replace("train/", "").replace("val/", "val:").replace("horizon/", "").replace("/mean", "")


def panel(series: Dict[str, Dict[int, float]], specs: List[Tuple[str, Tuple[str, ...]]], title: str, out: str) -> bool:
    """specs: [(label, (candidate tags...))]；一个面板里每个 spec 一条曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawn = 0
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=130)
    for label, candidates in specs:
        found = pick(series, *candidates)
        if found is None:
            continue
        tag, values = found
        xs = sorted(values)
        ys = [values[x] for x in xs]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2, label=f"{label} [{short(tag)}]")
        drawn += 1
    if drawn == 0:
        plt.close(fig)
        print(f"[plot] 跳过 {title}（无匹配 tag）")
        return False
    ax.set_title(title)
    ax.set_xlabel("step / epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[plot] {out}（{drawn} 条曲线）")
    return True


def stage_a_panels(series: Dict[str, Dict[int, float]], out_dir: str) -> None:
    panel(series, [(f"h{k} loss", (f"horizon/h{k}/loss/mean",)) for k in range(1, 7)],
          "Stage A · per-horizon WM loss", os.path.join(out_dir, "A_horizon_loss.png"))
    panel(series, [(f"h{k} ADE", (f"horizon/h{k}/ade/mean",)) for k in range(1, 7)]
          + [("h1 CV", ("horizon/h1/cv_ade/mean",)), ("h6 CV", ("horizon/h6/cv_ade/mean",))],
          "Stage A · per-horizon ADE vs constant-velocity", os.path.join(out_dir, "A_ade_vs_cv.png"))
    panel(series, [("presence AUC", ("train/presence_auc",)), ("entry AUC", ("train/entry_auc",)),
                   ("grad plan_head", ("train/grad_norm_plan_head",)), ("grad st_gnn", ("train/grad_norm_st_gnn",)),
                   ("grad router", ("train/grad_norm_router",))],
          "Stage A · health", os.path.join(out_dir, "A_health.png"))
    panel(series, [("wm_loss", ("train/wm_loss",)), ("od", ("train/wm_loss_od",)),
                   ("ego_next", ("train/wm_loss_ego_next",)), ("presence", ("train/presence_loss",)),
                   ("entry", ("train/entry_loss",))],
          "Stage A · WM loss terms", os.path.join(out_dir, "A_wm_terms.png"))
    # 逐 horizon ego_next（若存在）
    panel(series, [(f"h{k} ego_next", (f"horizon/h{k}/ego_next_loss/mean",)) for k in range(1, 7)],
          "Stage A · per-horizon ego_next loss", os.path.join(out_dir, "A_ego_next.png"))


def stage_b_panels(series: Dict[str, Dict[int, float]], out_dir: str) -> None:
    panel(series, [("bc(t)", ("train/primary_bc_loss", "train/specific_bc_loss")),
                   ("bc(v)", ("val/primary_bc_loss", "val/specific_bc_loss")),
                   ("traj(t)", ("train/primary_bc_traj_loss", "train/specific_bc_traj_loss")),
                   ("traj(v)", ("val/primary_bc_traj_loss", "val/specific_bc_traj_loss")),
                   ("action(t)", ("train/primary_bc_action_loss", "train/specific_bc_action_loss")),
                   ("router(t)", ("train/primary_bc_router_loss", "train/specific_bc_router_loss")),
                   ("router(v)", ("val/primary_bc_router_loss", "val/specific_bc_router_loss"))],
          "Stage B · loss terms (train vs val)", os.path.join(out_dir, "B_loss_terms.png"))
    panel(series, [("err mean(t)", ("train/primary_bc_action_err_weighted_mean", "train/specific_bc_action_err_weighted_mean")),
                   ("err mean(v)", ("val/primary_bc_action_err_weighted_mean", "val/specific_bc_action_err_weighted_mean")),
                   ("median(t)", ("train/primary_bc_action_err_median", "train/specific_bc_action_err_median")),
                   ("p95(t)", ("train/primary_bc_action_err_p95", "train/specific_bc_action_err_p95")),
                   ("brake", ("train/primary_bc_action_err_slice_brake_weighted_mean",)),
                   ("turn", ("train/primary_bc_action_err_slice_turn_weighted_mean",)),
                   ("curve", ("train/primary_bc_action_err_slice_curve_weighted_mean",))],
          "Stage B · action error (weighted L1) + slices", os.path.join(out_dir, "B_action_err.png"))
    panel(series, [(f"h{k} traj MAE(t)", (f"train/primary_bc_traj_mae_h{k}_m", f"train/specific_bc_traj_mae_h{k}_m")) for k in range(1, 7)]
          + [(f"h{k} traj MAE(v)", (f"val/primary_bc_traj_mae_h{k}_m", f"val/specific_bc_traj_mae_h{k}_m")) for k in (1, 3, 6)],
          "Stage B · per-horizon trajectory MAE (m)", os.path.join(out_dir, "B_traj_horizon.png"))
    panel(series, [("soft CE(t)", ("train/primary_bc_router_soft_ce", "train/specific_bc_router_soft_ce")),
                   ("soft CE(v)", ("val/primary_bc_router_soft_ce", "val/specific_bc_router_soft_ce")),
                   ("soft KL(t)", ("train/primary_bc_router_soft_kl", "train/specific_bc_router_soft_kl")),
                   ("top1(t)", ("train/primary_bc_router_top1_cluster_acc", "train/specific_bc_router_top1_cluster_acc")),
                   ("NMI(t)", ("train/primary_bc_router_nmi", "train/specific_bc_router_nmi")),
                   ("entropy(t)", ("train/primary_bc_router_entropy", "train/specific_bc_router_entropy"))],
          "Stage B · router (cluster soft targets)", os.path.join(out_dir, "B_router.png"))
    panel(series, [(f"expert {i} mix", (f"train/primary_bc_router_expert_mix_weight_{i}", f"train/specific_bc_router_expert_mix_weight_{i}")) for i in range(8)],
          "Stage B · expert mix weights", os.path.join(out_dir, "B_experts.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description="离线曲线绘图（monitor CSV → PNG）")
    ap.add_argument("--stage-a", default=None, help="Stage A 运行目录（含 monitor/metrics.csv）")
    ap.add_argument("--stage-b", default=None, help="Stage B 运行目录")
    ap.add_argument("--out", required=True, help="PNG 输出目录")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.stage_a:
        stage_a_panels(load_monitor(args.stage_a), args.out)
    if args.stage_b:
        stage_b_panels(load_monitor(args.stage_b), args.out)
    print(f"[plot] done → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
