#!/usr/bin/env python3
"""R41 第二批-F：**横向训练栈 / 薄装配层（数据入口）**。

把 ``pipeline.residual_dataset.build_pairs``（配对 + episode 级 split + obs 组装，**复用不改**）
与横向数据侧（``lateral_labels`` / ``lateral_replan``）装配为单一输入字典：

``rows / main_ids / succ_ids / train_mask / val_mask``（= build_pairs 原样；split 口径一致）
+ ``labels``（κ/arc/step_valid 全行）+ ``step_mask``（**仅 main_ids 行**：步有效 ∧ 锚视界）
+ ``station_target/station_mask``（仅 main_ids；站位三列）+ ``speed/speed_present/speed_weight/
bucket_id``（仅 main_ids）+ ``anchor_horizon``（全行）+ ``replan``（跨帧对，行号索引）。

口径约束
--------
- **禁自产状态回灌**：目标只来自 ``path_exec_v1`` 真实列（action 派生 κ；锚折线几何）；
- 抽样/split 与残差链同源（``--seed`` / ``--val-frac`` / ``--max-pairs`` 同一 ``build_pairs``），
  保证同一 episode 在两个训练栈之间不产生切分泄漏差异；
- 本模块零研究工作区依赖：只 import numpy 与 ``pipeline`` 内部模块（torch 由 ``build_obs`` 惰性）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import lateral_anchor, lateral_labels, lateral_replan, residual_dataset


def build_lateral_data(args) -> dict:
    """``args``（data/shards/max_pairs/seed/val_frac）→ 横向装配输入字典（见模块 docstring）。"""
    pairs = residual_dataset.build_pairs(args)
    rows = pairs["rows"]
    main_ids = np.asarray(pairs["main_ids"], dtype=np.int64)
    succ_ids = np.asarray(pairs["succ_ids"], dtype=np.int64)
    action = rows.get("action")
    if action is None:
        raise SystemExit("[lateral] fail-fast：path_exec_v1 缺 action (N,6,2)（κ 目标唯一来源）")

    labels = lateral_labels.plan_targets_from_action(np.asarray(action))
    horizon = lateral_labels.anchor_horizon_lengths(rows["anchor"])
    horizon_margin_m = float(getattr(args, "horizon_margin_m", 0.0) or 0.0)
    step_mask_full = lateral_labels.plan_step_masks(
        labels["step_valid"], labels["arc"], horizon, horizon_margin_m=horizon_margin_m
    )
    step_mask = step_mask_full[main_ids]
    station_target, station_mask = lateral_labels.station_targets(
        labels["kappa"][main_ids], labels["arc"][main_ids], step_mask
    )
    exog = np.asarray(pairs["exog"], dtype=np.float64)
    present = np.asarray(pairs["present"], dtype=np.float64)
    speed = exog[main_ids, 0]
    speed_present = present[main_ids, 0]
    speed_weight, bucket_id = lateral_labels.speed_bucket_weights(speed, speed_present)
    replan = lateral_replan.build_replan_pairs(
        rows,
        present,
        exog,
        main_ids,
        succ_ids,
        kappa=labels["kappa"],
        arc=labels["arc"],
        step_mask=step_mask_full,
    )
    meta: dict[str, Any] = {
        **pairs["meta"],
        "lateral": {
            "space": "kappa = dtheta/ds",
            "ds_min_m": lateral_labels.DS_MIN,
            "horizon_margin_m": horizon_margin_m,
            "stations_m": list(lateral_labels.KAPPA_STATIONS),
            "n_replan_pairs": int(replan["pair_valid"].size),
            "n_replan_pairs_valid": int(replan["pair_valid"].sum()),
            "station_mask_rates": [float(v) for v in station_mask.mean(axis=0)],
            "step_mask_rates": [float(v) for v in step_mask.mean(axis=0)],
        },
    }
    return {
        "rows": rows,
        "main_ids": main_ids,
        "succ_ids": succ_ids,
        "train_mask": pairs["train_mask"],
        "val_mask": pairs["val_mask"],
        "exog": exog,
        "present": present,
        "action": np.asarray(action, dtype=np.float64),
        "labels": labels,
        "anchor_horizon": horizon,
        "step_mask_full": step_mask_full,
        "step_mask": step_mask,
        "station_target": station_target,
        "station_mask": station_mask,
        "speed": speed,
        "speed_present": speed_present,
        "speed_weight": speed_weight,
        "bucket_id": bucket_id,
        "replan": replan,
        "horizon_margin_m": horizon_margin_m,
        "anchor_min_ds": float(
            getattr(args, "anchor_min_ds", None) or lateral_anchor.MIN_RESAMPLE_DS_M
        ),
        "anchor_noise_sigma_m": float(getattr(args, "anchor_noise_sigma_m", 0.0) or 0.0),
        "build_obs": residual_dataset.build_obs,
        "meta": meta,
    }


__all__ = ["build_lateral_data"]
