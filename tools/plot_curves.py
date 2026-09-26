#!/usr/bin/env python3
"""离线曲线绘图（不依赖 tensorboard；matplotlib Agg → PNG）。

用法::

    tools/venv-python tools/plot_curves.py \
        --stage-a runs/train/<runA>/stage_a --stage-b runs/train/<runB>/stage_b \
        --out runs/train/<runB>/plots

读取各 stage 的 ``monitor/metrics.csv``（长表 step,tag,value），按预设"面板"出图。
**每个面板内同族多线同图**（与 tensorboard 一致），只画保留清单（``docs/metrics.md``）：

- Stage A：`A_wm_od_loss.png`（`wm/od/loss` h1..h6）、`A_wm_od_ade.png`
  （`wm/od/ade_m` h1..h6 + cv_h1..cv_h6 共 12 线）、`A_wm_ego_next.png`
  （`wm/ego_next/loss` h1..h6）、`A_wm_kpi.png`（`wm/loss` + presence/entry AUC）
- Stage B：`B_loss_terms_{primary,specific}.png`（`stageB/<phase>/loss_terms` 4 项，
  train vs val）、`B_ego_traj_mae.png`（`ego/traj/mae_m` + `val_ego/traj/mae_m` h1..h6）、
  `B_ego_action_err.png`（`ego/action/err_weighted` train/val）、
  `B_router.png`（soft CE/KL + top1/NMI/entropy train/val）、
  `B_router_experts_{primary,specific}.png`（`router/<phase>/expert_mix_weight` e0..e7）

已删除的面板：slice/label 切片、median/p95、expert_util（见 ``docs/metrics.md``）。
旧 run（2026-09-27 前 tag）自动回退到 ``horizon/*`` / ``train|val/<phase>_bc_*``；
缺失的 tag 自动跳过；每个面板一个 PNG。
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


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
    """按候选顺序返回第一个存在的 tag（新 tag 优先、旧 tag 回退）。"""
    for tag in candidates:
        if tag and tag in series:
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
    panel(series, [(f"h{k}", (f"wm/od/loss/h{k}", f"horizon/h{k}/loss/mean")) for k in range(1, 7)],
          "Stage A · WM OD loss（逐 horizon）", os.path.join(out_dir, "A_wm_od_loss.png"))
    panel(series,
          [(f"h{k} ADE", (f"wm/od/ade_m/h{k}", f"horizon/h{k}/ade/mean")) for k in range(1, 7)]
          + [(f"cv h{k}", (f"wm/od/ade_m/cv_h{k}", f"horizon/h{k}/cv_ade/mean")) for k in range(1, 7)],
          "Stage A · WM OD ADE vs 匀速基线（12 线同图）", os.path.join(out_dir, "A_wm_od_ade.png"))
    panel(series,
          [(f"h{k}", (f"wm/ego_next/loss/h{k}", f"horizon/h{k}/ego_next_loss/mean")) for k in range(1, 7)],
          "Stage A · plan head ego_next loss（逐 horizon）", os.path.join(out_dir, "A_wm_ego_next.png"))
    panel(series, [("wm loss", ("wm/loss", "train/wm_loss")),
                   ("presence AUC", ("wm/presence_auc", "train/presence_auc")),
                   ("entry AUC", ("wm/entry_auc", "train/entry_auc"))],
          "Stage A · WM loss / presence / entry KPI", os.path.join(out_dir, "A_wm_kpi.png"))


#: 旧 tag 的 loss 项后缀（term → ``bc_*`` 键）
_BC_TERM_KEYS = {"loss": "loss", "traj": "traj_loss", "action": "action_loss", "router": "router_loss"}
_ROUTER_KPIS = ("soft_ce", "soft_kl", "top1_cluster_acc", "nmi", "entropy")


def stage_b_panels(series: Dict[str, Dict[int, float]], out_dir: str) -> None:
    for phase in ("primary", "specific"):
        specs: List[Tuple[str, Tuple[str, ...]]] = []
        for split in ("train", "val"):
            new_prefix = "val_stageB" if split == "val" else "stageB"
            old_prefix = f"{split}/{phase}_bc_"
            for term, key in _BC_TERM_KEYS.items():
                specs.append((f"{term}({split[0]})",
                              (f"{new_prefix}/{phase}/loss_terms/{term}", f"{old_prefix}{key}")))
        panel(series, specs, f"Stage B · {phase} loss terms（train vs val）",
              os.path.join(out_dir, f"B_loss_terms_{phase}.png"))
    panel(series,
          [(f"h{k} MAE(t)", (f"ego/traj/mae_m/h{k}", f"horizon/h{k}/traj_mae_m/mean",
                             f"train/primary_bc_traj_mae_h{k}_m")) for k in range(1, 7)]
          + [(f"h{k} MAE(v)", (f"val_ego/traj/mae_m/h{k}", f"val/horizon/h{k}/traj_mae_m/mean",
                               f"val/primary_bc_traj_mae_h{k}_m")) for k in range(1, 7)],
          "Stage B · ego 6 点轨迹逐 horizon 加权 MAE（m，train vs val）",
          os.path.join(out_dir, "B_ego_traj_mae.png"))
    panel(series, [("err weighted(t)", ("ego/action/err_weighted", "train/primary_bc_action_err_weighted_mean")),
                   ("err weighted(v)", ("val_ego/action/err_weighted", "val/primary_bc_action_err_weighted_mean"))],
          "Stage B · 首步动作加权误差（weighted L1）", os.path.join(out_dir, "B_ego_action_err.png"))
    panel(series, [(f"{name}({split[0]})",
                    (f"{'val_router' if split == 'val' else 'router'}/{name}",
                     f"{split}/primary_bc_router_{name}"))
                   for split in ("train", "val") for name in _ROUTER_KPIS],
          "Stage B · router（soft CE/KL + top1/NMI/entropy）", os.path.join(out_dir, "B_router.png"))
    for phase in ("primary", "specific"):
        specs = [
            (f"e{i}(t)", (f"router/{phase}/expert_mix_weight/e{i}",
                          f"train/{phase}_bc_router_expert_mix_weight_{i}"))
            for i in range(8)
        ] + [
            (f"e{i}(v)", (f"val_router/{phase}/expert_mix_weight/e{i}",
                          f"val/{phase}_bc_router_expert_mix_weight_{i}"))
            for i in range(8)
        ]
        panel(series, specs, f"Stage B · {phase} 专家混合权重（e0..e7，train vs val）",
              os.path.join(out_dir, f"B_router_experts_{phase}.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description="离线曲线绘图（monitor CSV → PNG；Tier-1 精简面板）")
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
