#!/usr/bin/env python3
"""R41 第二批-F 正式 runner 骨架：**横向训练栈装配验证（--dry-run / --assemble-smoke；零训练）**。

规格出处
--------
- R41 复审（``docs/deepseekv4p1_argue.md`` §2 批次：第二批 P0–P4；**不训练**，仅装配验证）；
- 数据侧 = ``pipeline/lateral_labels.py``（κ/计划目标 + 掩码 + 速度桶）、
  ``pipeline/lateral_replan.py``（跨帧对 + Warp + 门控）、``pipeline/lateral_lqr.py``
  （固定 LQR 执行差距权重；非梯度）、``pipeline/lateral_anchor.py``（锚修正/噪声增强，默认关）；
- loss 侧 = ``pipeline/lateral_losses.py``（κ 监督 / L_Δκ / L_replan / 批重加权钩子；默认关）；
- 既有项不动：``L_plan`` = ``pipeline.trainer.weighted_action_chain_loss``（(ds,dθ) 空间链损失）、
  既有曲率正则 = ``pipeline.trainer.plan_curvature_loss``（本 runner 只读引用/记录，不修改）。

本 runner 的职责（本 lane = 第二批准备 F）
------------------------------------------
1. ``--dry-run``：数据/目标/掩码/权重/跨帧对装配 + 账本打印（**不建模型、不写 runs/**）；
2. ``--check-features``：在 dry-run 上做**逐位复算契约**（κ 公式、掩码公式、warp 行覆盖、
   锚修正 off 逐位、噪声增强 σ 读数）；
3. ``--assemble-smoke``：在真实帧上（canary ckpt 冻结前向）计算 **arm 的每项损失值**
   （κ 监督/L_Δκ/L_replan/LQR gap 权重）+ 梯度流装配断言（仅到预测输入，无 optimizer step），
   写 ``<out>/smoke_<arm>.json``；
4. **不执行任何训练/oracle/闭环**；正式训练循环（P0–P4）待预注册批准后另立。

产物与写域：``runs/round41/f/**``（smoke 证据）；``.slim/deepwork/p1_v2/round41/f*``（报告）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import lateral_anchor, lateral_labels, lateral_lqr, lateral_losses, lateral_replan
from pipeline.lateral_dataset import build_lateral_data
from pipeline.residual_dataset import load_base_dataset, resolve_path

DEFAULT_CONFIG = ROOT / "config/lateral_train.yaml"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def harness_info() -> dict[str, Any]:
    """横向栈模块指纹（入库自包含；零研究工作区文件）。"""
    info: dict[str, Any] = {}
    for label, module in (
        ("lateral_labels", lateral_labels),
        ("lateral_replan", lateral_replan),
        ("lateral_lqr", lateral_lqr),
        ("lateral_anchor", lateral_anchor),
        ("lateral_losses", lateral_losses),
    ):
        path = Path(module.__file__).resolve()
        info[label] = {"module": module.__name__, "path": str(path), "sha256": _sha256(path)}
    info["runner"] = {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())}
    return info


# --------------------------------------------------------------------------- #
# 配置（config/lateral_train.yaml；CLI 同名键优先）
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    import yaml

    config_path = Path(path)
    if not config_path.is_file():
        raise SystemExit(f"[lateral] fail-fast：config 不存在：{config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"[lateral] fail-fast：config 顶层应为映射：{config_path}")
    return {str(key): value for key, value in payload.items()}


def build_parser(config: Mapping[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_lateral.py",
        description="横向训练栈装配验证（R41 第二批-F；零训练）",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 配置（键 = 本 parser dest）")
    parser.add_argument("--data", default="datasets/path_exec_v1")
    parser.add_argument("--base-dataset", default="datasets/BTC20261007-2202_expert5k_v8")
    parser.add_argument("--init", default="runs/p1_v2/round22/M0/ckpt_epoch002.pt")
    parser.add_argument("--model-config", default="runs/p1_v2/round22/M0/model.snapshot.yaml")
    parser.add_argument("--ckpt", default="runs/round36/j2/t1_keep_res/ckpt_epoch001.pt")
    parser.add_argument("--out", default=str(ROOT / "runs/round41/f"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shards", nargs="*", default=None, help="path_exec_v1 分片子集（默认全量）")
    parser.add_argument("--max-pairs", type=int, default=None, help="配对抽样上限（dry-run/smoke 用）")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--arm", default="p1", choices=lateral_losses.ARM_ORDER, help="P0–P4（预置见 f_prereg）")
    # 几何轴（R41-D 模型实例配置；默认哨兵 = 现行为）
    parser.add_argument(
        "--readout-lane-geometry",
        default="default",
        choices=("default", "corrected", "legacy_aligned"),
    )
    # 锚输入结构（默认关；AC 轴）
    parser.add_argument("--anchor-correction", default="off", choices=lateral_anchor.ANCHOR_CORRECTION_MODES)
    parser.add_argument("--anchor-min-ds", type=float, default=lateral_anchor.MIN_RESAMPLE_DS_M)
    parser.add_argument("--anchor-noise-sigma-m", type=float, default=0.0, help="锚噪声增强 σ（m；0=关；C 建议 0.02–0.05）")
    # κ 监督 / 掩码
    parser.add_argument("--horizon-margin-m", type=float, default=0.0, help="锚视界掩码安全边际（m）")
    parser.add_argument("--kappa-coef", type=float, default=None, help="覆盖 arm 预置的 κ 监督系数")
    parser.add_argument("--delta-kappa-coef", type=float, default=None, help="L_Δκ 系数（默认 0；备件）")
    parser.add_argument("--replan-coef", type=float, default=None, help="L_replan 系数")
    parser.add_argument("--dk-thresh", type=float, default=lateral_losses.DK_THRESH_DEFAULT)
    # P2 速度桶
    parser.add_argument("--speed-weight-mode", default=None, choices=("off", "bucket"))
    # P4 LQR
    parser.add_argument("--lqr-reweight", action="store_true", default=None)
    parser.add_argument("--lqr-probe", type=int, default=0, help="学生计划 LQR 读数的 probe 帧数（0=关）")
    parser.add_argument("--lqr-clip-max", type=float, default=1.0)
    parser.add_argument("--lqr-alpha", type=float, default=1.0)
    # 运行模式
    parser.add_argument("--dry-run", action="store_true", help="装配验证：不建模型、不写 runs/")
    parser.add_argument("--check-features", action="store_true", help="--dry-run + 逐位复算契约")
    parser.add_argument("--assemble-smoke", action="store_true", help="真实帧损失装配 smoke（写 runs/round41/f/**）")
    parser.add_argument("--json-out", default=None, help="报告 JSON 输出路径（--dry-run 默认不写）")
    parser.add_argument("--all-terms", action="store_true", help="smoke：附加全项开启（含 L_Δκ）的诊断行")
    if config:
        parser.set_defaults(**dict(config))
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG))
    known, _ = pre.parse_known_args(argv)
    config = load_config(known.config)
    parser = build_parser(config)
    args = parser.parse_args(argv)
    declared = {action.dest for action in parser._actions}
    unknown = sorted(set(config) - declared)
    if unknown:
        raise SystemExit(f"[lateral] fail-fast：{known.config} 含未知键 {unknown}（键名 = CLI dest）")
    if args.lqr_reweight is None:
        args.lqr_reweight = False
    if not isinstance(args.anchor_correction, str) or args.anchor_correction not in lateral_anchor.ANCHOR_CORRECTION_MODES:
        raise SystemExit(
            f"[lateral] fail-fast：anchor_correction={args.anchor_correction!r} 非法"
            f"（{' | '.join(lateral_anchor.ANCHOR_CORRECTION_MODES)}；YAML 中请加引号：\"off\"）"
        )
    return args


def resolve_loss_config(args) -> lateral_losses.LateralLossConfig:
    """arm 预置 + CLI 覆盖 → :class:`LateralLossConfig`（evaluate_arms 的单一出处）。"""
    overrides: dict[str, Any] = {
        "dk_thresh": float(args.dk_thresh),
        "lqr_clip_max": float(args.lqr_clip_max),
        "lqr_alpha": float(args.lqr_alpha),
    }
    if args.kappa_coef is not None:
        overrides["kappa_coef"] = float(args.kappa_coef)
    if args.delta_kappa_coef is not None:
        overrides["delta_kappa_coef"] = float(args.delta_kappa_coef)
    if args.replan_coef is not None:
        overrides["replan_coef"] = float(args.replan_coef)
    if args.speed_weight_mode is not None:
        overrides["speed_weight_mode"] = str(args.speed_weight_mode)
    config = lateral_losses.config_for_arm(args.arm, **overrides)
    if bool(args.lqr_reweight):
        config.lqr_reweight = True
    return config


# --------------------------------------------------------------------------- #
# 数据/账本（dry-run 核心；零训练）
# --------------------------------------------------------------------------- #
def _json_sanitize(value: Any) -> Any:
    """JSON 化：张量/ndarray → 类型摘要（原始值不落盘；标量透传）。"""
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return {"tensor": str(tuple(value.shape))}
    if isinstance(value, np.ndarray):
        return {"ndarray": str(tuple(value.shape))}
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def distribution_report(values: np.ndarray, *, mask: np.ndarray | None = None) -> dict:
    """分布读数（P50/P90/max/mean，掩码后）。"""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool).reshape(-1)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
        "max": float(arr.max()),
    }


def assemble_report(data: dict) -> dict:
    """横向数据侧账本（κ 目标/掩码/速度桶/跨帧对/门控/权重）。"""
    labels = data["labels"]
    step_mask = data["step_mask"]
    station_target = data["station_target"]
    station_mask = data["station_mask"]
    speed_weight = data["speed_weight"]
    bucket_ids = data["bucket_id"]
    speed = data["speed"]
    speed_present = data["speed_present"]
    main_ids = data["main_ids"]
    replan = data["replan"]

    valid_rows = speed_present > 0.5
    kappa = labels["kappa"][main_ids]
    step_valid = labels["step_valid"][main_ids]
    mask = step_mask
    report: dict[str, Any] = {
        "n_pairs": int(main_ids.size),
        "labels": {
            "space": "kappa = dtheta/ds（rad/m；理由见 pipeline/lateral_labels docstring）",
            "step_valid_rate": [float(v) for v in step_valid.mean(axis=0)],
            "plan_mask_rate": [float(v) for v in mask.mean(axis=0)],
            "kappa_abs_mean_valid": [
                float(np.abs(kappa[:, k][mask[:, k] > 0.5]).mean()) if bool((mask[:, k] > 0.5).any()) else None
                for k in range(kappa.shape[1])
            ],
            "kappa_p95_valid": [
                float(np.quantile(np.abs(kappa[:, k][mask[:, k] > 0.5]), 0.95))
                if bool((mask[:, k] > 0.5).any()) else None
                for k in range(kappa.shape[1])
            ],
            "all_zero_action_rows": int((np.abs(data["action"]).sum(axis=(1, 2)) == 0).sum()),
        },
        "anchor_horizon_m": distribution_report(data["anchor_horizon"], mask=np.ones_like(data["anchor_horizon"], dtype=bool)),
        "anchor_horizon_frac_lt_29_75": float((data["anchor_horizon"] < 29.75).mean()),
        "stations": {
            "names": ["kappa_near", "kappa_mid", "kappa_far"],
            "stations_m": list(lateral_labels.KAPPA_STATIONS),
            "mask_rate": [float(v) for v in station_mask.mean(axis=0)],
            "target_abs_mean_valid": [
                float(np.abs(station_target[:, s][station_mask[:, s] > 0.5]).mean())
                if bool((station_mask[:, s] > 0.5).any()) else None
                for s in range(station_mask.shape[1])
            ],
        },
        "speed": {
            "edges": list(lateral_labels.SPEED_BUCKET_EDGES),
            "weights_table": list(lateral_labels.SPEED_BUCKET_WEIGHTS_DEFAULT),
            "bucket_counts": [int(v) for v in np.bincount(bucket_ids[valid_rows], minlength=len(lateral_labels.SPEED_BUCKET_EDGES))],
            "weight_mean_valid": float(speed_weight[valid_rows].mean()) if valid_rows.any() else None,
            "speed_mps": distribution_report(speed, mask=valid_rows),
        },
        "replan_pairs": {
            "count": int(replan["pair_valid"].size),
            "pair_valid_rate": float(replan["pair_valid"].mean()) if replan["pair_valid"].size else None,
            "ds_exec_m": distribution_report(replan["ds_exec"], mask=replan["pair_valid"]),
            "delta_psi_abs": distribution_report(np.abs(replan["delta_psi"]), mask=replan["pair_valid"]),
            "gate_safety": distribution_report(replan["gate_safety"], mask=replan["pair_valid"]),
            "gate_align": distribution_report(replan["gate_align"], mask=replan["pair_valid"]),
            "w_replan": distribution_report(replan["w_replan"], mask=replan["pair_valid"]),
            "lam_positive_rate": float((replan["lam"] > 0).mean()) if replan["lam"].size else None,
            "warp_valid_rate_per_step": [float(v) for v in (replan["warp_index"] >= 0).mean(axis=0)]
            if replan["warp_index"].size else [],
        },
        "horizon_present_note": (
            "状态侧 present 掩码由 residual_targets（base_present ∧ real_present）承担；"
            "本账本报告 lateral 目标侧掩码（步有效 + 锚视界）"
        ),
    }
    return report


# --------------------------------------------------------------------------- #
# 逐位复算契约（--check-features）
# --------------------------------------------------------------------------- #
def feature_contract(data: dict) -> dict:
    """κ/掩码/warp/锚修正 的逐位复算（单一出处 = 数据侧模块）。"""
    labels = data["labels"]
    main_ids = data["main_ids"]
    action = np.asarray(data["action"], dtype=np.float64)
    recomputed = lateral_labels.plan_targets_from_action(action)
    report: dict[str, Any] = {
        "kappa_formula_bitwise": bool(np.array_equal(recomputed["kappa"], labels["kappa"])),
        "arc_formula_bitwise": bool(np.array_equal(recomputed["arc"], labels["arc"])),
        "step_valid_bitwise": bool(np.array_equal(recomputed["step_valid"], labels["step_valid"])),
    }
    mask_re = lateral_labels.plan_step_masks(
        recomputed["step_valid"][main_ids],
        recomputed["arc"][main_ids],
        data["anchor_horizon"][main_ids],
        horizon_margin_m=float(data["horizon_margin_m"]),
    )
    report["plan_mask_bitwise"] = bool(np.array_equal(mask_re, data["step_mask"]))
    replan = data["replan"]
    long_prev = replan["warp_index"]
    report["warp_index_range_ok"] = bool(
        (long_prev <= 5).all() and (long_prev >= -1).all()
    ) if long_prev.size else True
    # warp 值复算：apply_warp ≡ gather
    warped, warp_mask = lateral_replan.apply_warp(
        replan["warp_index"], labels["kappa"][replan["prev_row_ids"]]
    )
    manual = np.zeros_like(warped)
    hit = replan["warp_index"] >= 0
    if hit.any():
        rows = np.arange(hit.shape[0])[:, None].repeat(hit.shape[1], axis=1)
        manual = np.where(
            hit, labels["kappa"][replan["prev_row_ids"]][rows, np.clip(replan["warp_index"], 0, 5)], 0.0
        )
    report["warp_gather_bitwise"] = bool(np.array_equal(warped, manual))
    report["warp_mask_pairs"] = bool(np.array_equal(warp_mask > 0.5, hit))
    # 锚修正 off 逐位 / 噪声增强读数
    import torch

    first = np.asarray(data["rows"]["anchor"][int(main_ids[0])], dtype=np.float32)
    tensor = torch.as_tensor(first)[None]
    tensor_mask = torch.ones(1, tensor.shape[1])
    off_xy, off_mask = lateral_anchor.correct_anchor_tensor(tensor, tensor_mask, mode="off")
    report["anchor_correction_off_bitwise"] = bool(torch.equal(off_xy, tensor) and torch.equal(off_mask, tensor_mask))
    on_xy, on_mask = lateral_anchor.correct_anchor_tensor(
        tensor, tensor_mask, mode="resample", min_ds_m=float(data["anchor_min_ds"])
    )
    report["anchor_resample_points_orig"] = int(tensor.shape[1])
    report["anchor_resample_points_new"] = int(on_xy.shape[1])
    # 点间距 ≥ min_ds（内部点）
    pts = on_xy[0, : int(on_mask[0].sum())].numpy()
    if len(pts) >= 2:
        spacing = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        report["anchor_resample_min_spacing_m"] = float(spacing.min()) if spacing.size else None
    report["anchor_noise_sigma_m"] = float(data["anchor_noise_sigma_m"])
    report["pass"] = bool(
        report["kappa_formula_bitwise"]
        and report["arc_formula_bitwise"]
        and report["step_valid_bitwise"]
        and report["plan_mask_bitwise"]
        and report["warp_index_range_ok"]
        and report["warp_gather_bitwise"]
        and report["warp_mask_pairs"]
        and report["anchor_correction_off_bitwise"]
    )
    return report


# --------------------------------------------------------------------------- #
# 学生计划 probe（冻结前向；LQR 差距权重；零梯度）
# --------------------------------------------------------------------------- #
def probe_student_plans(args, data: dict, device) -> dict:
    """canary 冻结前向 → 学生 κ_near/mid/far + 6 步 plan + LQR 饱和/差距权重。

    帧 = 重规划对的**当前帧**（``replan['curr_row_ids']``）与**上一帧**
    （``replan['prev_row_ids']``）各取前 ``--lqr-probe`` 行；全程 ``torch.no_grad``，
    **零 optimizer**。
    """
    import torch

    from tools.train_residual import build_model

    count = int(args.lqr_probe)
    replan = data["replan"]
    if count <= 0 or replan["curr_ids"].size == 0:
        return {}
    take = min(count, int(replan["curr_ids"].size))
    positions = np.arange(take, dtype=np.int64)
    curr_rows = replan["curr_row_ids"][:take].astype(np.int64)
    prev_rows = replan["prev_row_ids"][:take].astype(np.int64)

    model, load_info = build_model(args, device)
    if str(args.readout_lane_geometry) != "default":
        model.set_readout_lane_geometry(str(args.readout_lane_geometry))
    base_dataset = load_base_dataset(resolve_path(Path(args.base_dataset)))
    generator = torch.Generator(device=str(device)).manual_seed(int(args.seed))
    with torch.no_grad():
        outputs = {}
        for label, rows_idx in (("curr", curr_rows), ("prev", prev_rows)):
            obs = data["build_obs"](data["rows"], rows_idx, device=device, base_dataset=base_dataset)
            lateral_anchor.apply_anchor_options(
                obs,
                correction=str(args.anchor_correction),
                noise_sigma_m=float(args.anchor_noise_sigma_m),
                generator=generator if float(args.anchor_noise_sigma_m) > 0.0 else None,
            )
            path_out = model.encode(obs)["path"]
            outputs[label] = {
                "kappa_stations": torch.stack(
                    [path_out["kappa_near"], path_out["kappa_mid"], path_out["kappa_far"]], dim=-1
                ).cpu(),
                "plan": path_out["plan_aux"].cpu(),
                "obs_head": {key: float(obs[key].abs().mean().item()) for key in ("lane_anchor_xy", "lane_anchor_mask")},
            }
    curr = outputs["curr"]
    prev = outputs["prev"]
    speed = np.asarray(data["speed"])[positions]
    speed_present = np.asarray(data["speed_present"])[positions] > 0.5
    expert_plan = np.asarray(data["action"], dtype=np.float64)[curr_rows]
    sat_expert = lateral_lqr.preview_steer_raw(expert_plan, np.where(speed_present, speed, 0.5))
    sat_student = lateral_lqr.preview_steer_raw(curr["plan"].numpy().astype(np.float64), np.where(speed_present, speed, 0.5))
    gap = lateral_lqr.execution_gap_weights(sat_student["sat"], sat_expert["sat"], valid=speed_present)
    return {
        "n": int(take),
        "load_info_ckpt_sha256": load_info["ckpt_sha256"],
        "readout_lane_geometry": str(model.readout_lane_geometry),
        "kappa_stations_pred": curr["kappa_stations"],
        "plan_pred_curr": curr["plan"],
        "plan_pred_prev": prev["plan"],
        "sat_expert": sat_expert["sat"],
        "sat_student": sat_student["sat"],
        "sat_expert_dist": distribution_report(sat_expert["sat"]),
        "sat_student_dist": distribution_report(sat_student["sat"]),
        "gap": gap,
        "gap_dist": distribution_report(gap["gap"], mask=gap["valid"]),
        "steer_sat_student_rate": float((np.abs(sat_student["steer_clipped"]) >= 1.0).mean()),
        "steer_sat_expert_rate": float((np.abs(sat_expert["steer_clipped"]) >= 1.0).mean()),
        "obs_head_anchor_absmean": curr["obs_head"],
    }


# --------------------------------------------------------------------------- #
# 损失装配（real tensors；逐项值 + 梯度流断言；零 optimizer）
# --------------------------------------------------------------------------- #
def assemble_loss_terms(args, data: dict, probe: dict) -> dict:
    """arm 损失项在真实帧上的装配读数（κ 监督 / L_Δκ / L_replan / 批重加权）。

    - 预测 = canary 冻结前向（probe）；目标/掩码/权重 = 数据侧；LQR gap = 非梯度；
    - 另跑 ``all_terms`` 诊断行（L_Δκ 强制开；全项同时开）验证梯度可达（只到预测输入）。
    """
    import torch

    if not probe:
        return {"skipped": "无 probe（--lqr-probe 0）；本项需要学生计划"}
    take = int(probe["n"])
    # take = replan pairs 前 N（与 probe 同取法）；probe 的各读数按 pair 位置排序
    station_pos = np.arange(take, dtype=np.int64)
    replan = data["replan"]

    kappa_pred = probe["kappa_stations_pred"]  # (N,3)
    kappa_target = torch.as_tensor(data["station_target"][station_pos], dtype=torch.float32)
    kappa_mask = torch.as_tensor(data["station_mask"][station_pos], dtype=torch.float32)
    plan_pred = probe["plan_pred_curr"]  # (N,6,2)
    kappa_prev_pred = lateral_losses.plan_kappa_from_plan(probe["plan_pred_prev"])[0]

    warp_index = torch.as_tensor(replan["warp_index"][:take], dtype=torch.long)
    warp_mask = torch.as_tensor((replan["warp_index"][:take] >= 0).astype(np.float32))
    warp_weight = torch.as_tensor(replan["w_replan"][:take], dtype=torch.float32)
    base_weights = torch.as_tensor(data["speed_weight"][station_pos], dtype=torch.float32)
    lqr_weights = torch.as_tensor(probe["gap"]["w"][:take], dtype=torch.float32)

    plan_target_mask = torch.as_tensor(
        data["step_mask"][station_pos], dtype=torch.float32
    )

    def run(config) -> dict:
        pred = kappa_pred.clone().requires_grad_(True)
        plan = plan_pred.clone().requires_grad_(True)
        prev = kappa_prev_pred.clone().requires_grad_(True)
        out = lateral_losses.lateral_loss_terms(
            config,
            kappa_pred=pred,
            kappa_target=kappa_target,
            kappa_mask=kappa_mask,
            plan_pred=plan,
            plan_target_mask=plan_target_mask,
            kappa_prev_pred=prev,
            warp_index=warp_index,
            warp_mask=warp_mask,
            warp_weight=warp_weight,
            base_sample_weights=base_weights,
            lqr_weights=lqr_weights,
        )
        gradients = {}
        if out["terms"] and float(out["total"].detach()) != 0.0:
            grads = torch.autograd.grad(out["total"], [pred, plan, prev], retain_graph=False, allow_unused=True)
            gradients = {
                "kappa_pred_grad_norm": float(grads[0].norm()) if grads[0] is not None else 0.0,
                "plan_pred_grad_norm": float(grads[1].norm()) if grads[1] is not None else 0.0,
                "kappa_prev_pred_grad_norm": float(grads[2].norm()) if grads[2] is not None else 0.0,
            }
        return {
            "active": list(out["active"]),
            "terms": {key: float(value.detach()) for key, value in out["terms"].items()},
            "total": float(out["total"].detach()),
            "sample_weights": (
                {
                    "mean": float(out["sample_weights"].mean()),
                    "min": float(out["sample_weights"].min()),
                    "max": float(out["sample_weights"].max()),
                }
                if out["sample_weights"] is not None else None
            ),
            "gradients": gradients,
        }

    arm_config = resolve_loss_config(args)
    report = {"arm": args.arm, "config": arm_config.__dict__, "arm_terms": run(arm_config)}
    if bool(args.all_terms):
        all_on = lateral_losses.LateralLossConfig(
            kappa_coef=0.5, delta_kappa_coef=0.1, replan_coef=0.5,
            speed_weight_mode="bucket", lqr_reweight=True, lqr_alpha=float(args.lqr_alpha),
            lqr_clip_max=float(args.lqr_clip_max), dk_thresh=float(args.dk_thresh),
        )
        report["all_terms"] = run(all_on)
    report["default_off_zero"] = run(lateral_losses.LateralLossConfig())
    report["pass"] = bool(
        report["default_off_zero"]["total"] == 0.0
        and report["default_off_zero"]["terms"] == {}
        and all(np.isfinite(value) for value in report["arm_terms"]["terms"].values())
    )
    return report


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if not (args.dry_run or args.check_features or args.assemble_smoke):
        raise SystemExit(
            "[lateral] fail-fast：本 runner 不执行训练（R41-F 纪律）；"
            "请用 --dry-run / --check-features / --assemble-smoke"
        )
    data = build_lateral_data(args)
    print(f"[lateral] pairs {data['meta']}", flush=True)
    report = assemble_report(data)
    report["meta"] = {
        "runner": "R41 第二批-F（横向训练栈装配；零训练）",
        "arm": args.arm,
        "device": str(device),
        "data": data["meta"],
        "geometry_axis": {
            "readout_lane_geometry": str(args.readout_lane_geometry),
            "anchor_correction": str(args.anchor_correction),
            "anchor_min_ds_m": float(args.anchor_min_ds),
            "anchor_noise_sigma_m": float(args.anchor_noise_sigma_m),
        },
        "harness": harness_info(),
    }
    contract = feature_contract(data) if args.check_features else None
    probe: dict = {}
    if args.assemble_smoke or int(args.lqr_probe) > 0:
        probe = probe_student_plans(args, data, device)
        report["probe"] = _json_sanitize(
            {key: value for key, value in probe.items() if not torch.is_tensor(value)}
        )
    if args.assemble_smoke:
        report["loss_assembly"] = assemble_loss_terms(args, data, probe)
    report["contract"] = contract
    report["elapsed_s"] = round(time.time() - started, 1)
    report["pass"] = bool(
        (contract is None or contract["pass"])
        and report.get("loss_assembly", {}).get("pass", True)
    )
    out_path: Path | None = None
    if args.assemble_smoke:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.json_out) if args.json_out else out_dir / f"smoke_{args.arm}.json"
    elif args.json_out:
        out_path = Path(args.json_out)
    if out_path is not None:
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[lateral] → {out_path}", flush=True)
    print(json.dumps({k: report[k] for k in ("pass", "elapsed_s")}, ensure_ascii=False), flush=True)
    if not report["pass"]:
        raise SystemExit("[lateral] fail-fast：装配验证未通过")
    print("[lateral] 装配验证通过（零训练、零 optimizer step）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
