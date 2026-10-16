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
4. **R41 第二批执行-G：正式训练循环**（``--train`` / ``--smoke-steps N``；F 预注册 §1 共同面）：
   数据 = ``pipeline.lateral_dataset``（episode 级 split，复用残差链）；模型装配 =
   ``tools.train_residual.build_model`` 口径（M0 snapshot + legacy 迁移 + ``--ckpt``
   冻结基座）+ 几何轴实例配置；**冻结面 = 除 allowlist 外全部**（``net.path_head.
   TRAIN_ALLOWLIST_PREFIXES`` = ``policy.structured_action.`` + ``plan_head.fusion.2.``）；
   loss = ``L_action``（首步 (ds,dθ)；= ``pipeline.trainer._phase3_loss_terms`` 的 action 项口径）
   + ``L_plan``（``trainer.weighted_action_chain_loss``；6 步 (ds,dθ) 链，k=1..5）+ 横向项
   （``lateral_losses.ARM_PRESETS``；p0 = 零触达）；frame_weight = ``train_weight×balance_weight``
   （= ``trainer.row_action_weights`` 口径；`g_probe` 已验证 path_exec≡base 同键行）；
   val（episode 级）只用于**选点**；ckpt → ``runs/dev/round41/<arm>/``；
   账本 = loss 曲线/分项/时长/参数五栏/四 hash/冻结面逐位守卫/首步梯度探针。

**禁**：P3/P4 训练期接线（跨帧双前向 / LQR 权重量产）不在本批（F 预注册 §8-3/§6 展开规则）
⇒ ``replan_coef>0`` / ``lqr_reweight`` 在训练模式下 fail-fast。

产物与写域：``runs/round41/f/**``（smoke 证据）、``runs/dev/round41/<arm>/**``（训练产物）；
``.slim/deepwork/p1_v2/round41/{f,g}*``（报告）。
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
    # ---- R41 第二批执行-G：正式训练循环 ----
    parser.add_argument("--train", action="store_true", help="正式训练循环（写 runs/dev/round41/<arm>/**）")
    parser.add_argument("--smoke-steps", type=int, default=0, help="训练循环但最多 N 个 optimizer step（成本标定；不写 ckpt）")
    parser.add_argument("--train-out", default=None, help="训练产物目录（默认 runs/dev/round41/<arm>）")
    parser.add_argument("--epochs", type=int, default=2, help="≤2（F 预注册 §4）")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=5.0, help="0 = 关（非法值 fail-fast）")
    parser.add_argument("--max-steps", type=int, default=8000, help="optimizer step 总上限（F 预注册 §4）")
    parser.add_argument("--max-wall-s", type=float, default=5400.0, help="wall 预算（s；触顶即停并显式记账）")
    parser.add_argument("--action-weight", type=float, default=1.0, help="L_action 权重（= phase3 action_weight 默认）")
    parser.add_argument("--chain-weight", type=float, default=0.2, help="L_plan 权重（= phase3 action_chain 默认）")
    parser.add_argument(
        "--forward-mode", default="chain", choices=("chain", "t0"),
        help="chain = policy_wm_rollout 6 步想象链（= 既有 phase3 口径；默认）；t0 = 仅 encode（诊断臂）",
    )
    parser.add_argument("--calib-steps", type=int, default=20, help="步时标定窗口：投影超预算 ⇒ 降 step 并记账")
    parser.add_argument("--log-every", type=int, default=100, help="loss 曲线落点间隔（step）")
    parser.add_argument("--no-grad-probe", action="store_false", dest="grad_probe", default=True,
                        help="关首步分项梯度探针（默认开）")
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
# 正式训练循环（R41 第二批执行-G；F 预注册 §1 共同面）
# --------------------------------------------------------------------------- #
def _net_tree_digest() -> dict:
    """``net/**/*.py`` 源树指纹（口径 = m2：逐文件 sha16 → 排序连接 sha256 首 16 hex；只读）。"""
    import hashlib

    root = ROOT / "net"
    per_file = {}
    for path in sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts):
        per_file[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    tree = hashlib.sha256()
    for rel in sorted(per_file):
        tree.update(f"{rel}\0{per_file[rel]}\n".encode("utf-8"))
    return {"n_files": len(per_file), "tree_sha16": tree.hexdigest()[:16]}


def _combined_harness_sha(harness: Mapping[str, Any]) -> str:
    import hashlib

    payload = json.dumps(
        {key: value["sha256"] for key, value in harness.items() if isinstance(value, Mapping) and "sha256" in value},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def training_mode(args) -> bool:
    """是否走训练循环（``--train`` / ``--smoke-steps``）。"""
    return bool(args.train) or int(args.smoke_steps) > 0


def train_model_assembly(args):
    """模型装配（= ``tools.train_residual.build_model`` 口径）+ 几何轴实例配置。"""
    from tools.train_residual import build_model

    model, load_info = build_model(args, _device_of(args))
    model.set_readout_lane_geometry(str(args.readout_lane_geometry))
    return model, load_info


def _device_of(args):
    import torch

    return torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")


def freeze_to_allowlist(model) -> tuple[list, list[str]]:
    """除 allowlist（``net.path_head.TRAIN_ALLOWLIST_PREFIXES``）外全冻；返回可训参数与键名。"""
    from net.path_head import TRAIN_ALLOWLIST_PREFIXES

    prefixes = tuple(TRAIN_ALLOWLIST_PREFIXES)
    trainable = []
    for name, parameter in model.named_parameters():
        allowed = name.startswith(prefixes)
        parameter.requires_grad_(allowed)
        if allowed:
            trainable.append((name, parameter))
    if not trainable:
        raise SystemExit("[lateral] fail-fast：allowlist 未命中任何参数（结构化头未构造？）")
    foreign = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith(prefixes)
    ]
    if foreign:  # pragma: no cover - 结构性排除
        raise SystemExit(f"[lateral] fail-fast：存在 allowlist 外未冻结参数 {foreign[:3]}")
    return trainable, sorted(name for name, _ in trainable)


def training_data_arrays(args, data: dict) -> dict:
    """训练张量视图：``action``（仅 main_ids）+ 掩码/目标/权重 + frame_weight（train×balance）。"""
    import numpy as np

    from pipeline.residual_dataset import load_base_dataset, resolve_path

    index = np.asarray(data["main_ids"], dtype=np.int64)
    base = load_base_dataset(resolve_path(Path(args.base_dataset)))
    arrays = base["dataset"].arrays
    rows = data["rows"]
    base_rows = np.asarray(
        [
            base["lookup"][(int(rows["spec_id"][i]), int(rows["episode_id"][i]), int(rows["step"][i]))]
            for i in index
        ],
        dtype=np.int64,
    )
    train_w = np.asarray(arrays["train_weight"], dtype=np.float32)[base_rows]
    balance_w = np.asarray(arrays["balance_weight"], dtype=np.float32)[base_rows]
    if not bool((train_w > 0.5).all()):  # pragma: no cover - 结构性排除（ok 掩码已含 train_weight>0.5）
        raise SystemExit("[lateral] fail-fast：训练行出现 train_weight≤0.5")
    return {
        "index": index,
        "action": np.asarray(data["action"], dtype=np.float32)[index],
        "step_mask": np.asarray(data["step_mask"], dtype=np.float32),
        "station_target": np.asarray(data["station_target"], dtype=np.float32),
        "station_mask": np.asarray(data["station_mask"], dtype=np.float32),
        "speed_weight": np.asarray(data["speed_weight"], dtype=np.float32),
        "frame_weight": (train_w * balance_w).astype(np.float32),
        "base": base,
    }


def batch_tensors(view: dict, index: "np.ndarray", device):
    """行索引 → device 张量批（目标/掩码/权重；obs 由 ``build_obs`` 另组装）。"""
    import torch

    def take(array):
        return torch.as_tensor(np.asarray(array)[index], dtype=torch.float32, device=device)

    return {
        "action": take(view["action"]),
        "step_mask": take(view["step_mask"]),
        "station_target": take(view["station_target"]),
        "station_mask": take(view["station_mask"]),
        "speed_weight": take(view["speed_weight"]),
        "frame_weight": take(view["frame_weight"]),
    }


def batch_loss_terms(model, obs, batch: dict, config: lateral_losses.LateralLossConfig, args) -> dict:
    """单批前向 + 损失（L_action + L_plan + 横向项；单一出处 = trainer / lateral_losses）。

    ``--forward-mode chain``（默认；= 既有 ``trainer._phase3_loss_terms`` 口径）= 模型自身
    ``policy_wm_rollout`` 6 步想象链：``plan`` = 链 plan（评测 ``--tail chain`` 的同一产物），
    ``action_mu`` = t0 动作（≡ ``encode`` 的 ``action``，逐位）；``t0`` = 仅 ``encode``（诊断臂；
    G-iter1 已证：t0-only 目标不管 k≥1 想象latent ⇒ 链尾失控 ⇒ 不作正式臂）。
    """
    import torch

    from pipeline.trainer import weighted_action_chain_loss

    forward_mode = str(getattr(args, "forward_mode", "chain") or "chain")
    if forward_mode == "chain":
        out = model(obs, forward_mode="policy_wm_rollout")
        action_pred = out["action_mu"]
        plan_pred = out["plan"]
        kappa_pred = torch.stack(
            [
                out["structured_kappa_near"][:, 0],
                out["structured_aux_kappa_mid"][:, 0],
                out["structured_aux_kappa_far"][:, 0],
            ],
            dim=-1,
        )
    elif forward_mode == "t0":
        out = model.encode(obs)["path"]
        action_pred = out["action"]
        plan_pred = out["plan_aux"]
        kappa_pred = torch.stack([out["kappa_near"], out["kappa_mid"], out["kappa_far"]], dim=-1)
    else:
        raise SystemExit(f"[lateral] fail-fast：--forward-mode 只支持 chain | t0，收到 {forward_mode!r}")
    frame_w = batch["frame_weight"]
    diff = action_pred - batch["action"][:, 0]
    per_sample = (diff ** 2).mean(dim=-1)
    l_action = (per_sample * frame_w).sum() / frame_w.sum().clamp_min(1e-8)
    l_plan, per_step = weighted_action_chain_loss(
        plan_pred, batch["action"], batch["step_mask"], frame_weight=frame_w, loss_type="l2"
    )
    lateral = lateral_losses.lateral_loss_terms(
        config,
        kappa_pred=kappa_pred,
        kappa_target=batch["station_target"],
        kappa_mask=batch["station_mask"],
        plan_pred=plan_pred,
        plan_target_mask=batch["step_mask"],
        base_sample_weights=batch["speed_weight"],
    )
    total = float(args.action_weight) * l_action + float(args.chain_weight) * l_plan + lateral["total"]
    return {
        "total": total,
        "l_action": float(args.action_weight) * l_action,
        "l_plan": float(args.chain_weight) * l_plan,
        "lateral_terms": {key: value for key, value in lateral["terms"].items()},
        "lateral_total": lateral["total"],
        "lateral_active": tuple(lateral["active"]),
        "parts": {
            "action": float(args.action_weight) * l_action,
            "chain": float(args.chain_weight) * l_plan,
            "lateral": lateral["total"],
        },
        "plan_per_step": per_step,
        "forward_mode": forward_mode,
    }


def _grad_probe(model, obs, batch, config, args, trainable, *, terms: dict | None = None) -> dict:
    """首步分项梯度探针：三项各自对 allowlist 的梯度范数 + 两两 cosine（retain_graph）。"""
    import torch

    params = [parameter for _, parameter in trainable]
    if terms is None:
        terms = batch_loss_terms(model, obs, batch, config, args)
    vectors = {}
    for label in ("action", "chain", "lateral"):
        part = terms["parts"][label]
        if not torch.is_tensor(part) or not part.requires_grad:
            vectors[label] = None  # 零图项（如 p0 的横向项）无梯度
            continue
        grads = torch.autograd.grad(part, params, retain_graph=True, allow_unused=True)
        vectors[label] = torch.cat(
            [
                (grad.detach().reshape(-1) if grad is not None else torch.zeros_like(parameter).reshape(-1))
                for grad, (_, parameter) in zip(grads, trainable)
            ]
        )

    def _cos(a, b) -> float:
        if a is None or b is None:
            return float("nan")
        denom = float(a.norm()) * float(b.norm())
        return float(torch.dot(a, b) / denom) if denom > 1e-12 else float("nan")

    return {
        "grad_norm": {
            label: (None if vector is None else float(vector.norm())) for label, vector in vectors.items()
        },
        "cos": {
            "action_chain": _cos(vectors["action"], vectors["chain"]),
            "action_lateral": _cos(vectors["action"], vectors["lateral"]),
            "chain_lateral": _cos(vectors["chain"], vectors["lateral"]),
        },
        "loss_values": {key: float(value.detach()) for key, value in terms["parts"].items()},
    }


def train_loop(args, data: dict) -> int:
    """正式训练循环（F 预注册共同面；写 ``--train-out``）。"""
    import hashlib
    import numpy as np
    import torch

    from pipeline.residual_dataset import build_obs

    if float(args.replan_coef if args.replan_coef is not None else 0.0) > 0.0 or bool(args.lqr_reweight):
        raise SystemExit(
            "[lateral] fail-fast：P3/P4 训练期接线不在本批（F 预注册 §8-3；跨帧双前向 / LQR 权重量产"
            " 待 P1 过门后另立）"
        )
    config = resolve_loss_config(args)
    if config.replan_coef or config.lqr_reweight:  # pragma: no cover - 同上（arm 预置路径）
        raise SystemExit("[lateral] fail-fast：P3/P4 训练期接线不在本批（F 预注册 §8-3）")
    smoke = int(args.smoke_steps) > 0
    if float(args.grad_clip) < 0.0:
        raise SystemExit(f"[lateral] fail-fast：--grad-clip 非法（{args.grad_clip}）")

    started = time.time()
    torch.manual_seed(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))

    model, load_info = train_model_assembly(args)
    trainable, allow_names = freeze_to_allowlist(model)
    optimizer = torch.optim.Adam(
        [parameter for _, parameter in trainable],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimizer_ids != {id(p) for _, p in trainable}:  # pragma: no cover - allowlist 断言
        raise SystemExit("[lateral] fail-fast：optimizer 参数集 ≠ allowlist（冻结面违约）")
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not name.startswith(tuple(__import__("net.path_head", fromlist=["x"]).TRAIN_ALLOWLIST_PREFIXES))
    }

    view = training_data_arrays(args, data)
    main_ids = view["index"]
    train_idx = np.flatnonzero(np.asarray(data["train_mask"], dtype=bool))
    val_idx = np.flatnonzero(np.asarray(data["val_mask"], dtype=bool))
    n_train = int(train_idx.size)

    plan = {
        "arm": args.arm,
        "forward_mode": str(args.forward_mode),
        "loss_config": {key: value for key, value in config.__dict__.items()},
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "max_steps": int(args.max_steps),
        "max_wall_s": float(args.max_wall_s),
        "action_weight": float(args.action_weight),
        "chain_weight": float(args.chain_weight),
        "steps_per_epoch": int(np.ceil(n_train / max(1, int(args.batch_size)))),
        "n_train_pairs": n_train,
        "n_val_pairs": int(val_idx.size),
        "seed": int(args.seed),
    }
    ledger: dict[str, Any] = {
        "mode": "smoke-steps" if smoke else "train",
        "runner": "R41 第二批执行-G（训练循环；F 预注册 §1 共同面）",
        "plan": plan,
        "data": data["meta"],
        "geometry": {
            "readout_lane_geometry": str(model.readout_lane_geometry),
            "note": "训练前向（encode→path）对几何轴逐位不敏感（g_probe 实测：default≡legacy_aligned；"
                    "corrected 只改 rollout 步几何，t0 头输出 maxΔ=0）",
        },
        "model": {
            "ckpt": load_info["ckpt"],
            "ckpt_sha256": load_info["ckpt_sha256"],
            "init": str(args.init),
            "init_sha256": _sha256(Path(resolve_path(Path(args.init)))),
            "model_config": str(args.model_config),
            "model_config_sha256": _sha256(Path(resolve_path(Path(args.model_config)))),
            "missing_keys": load_info["missing_keys"],
            "unexpected_keys": load_info["unexpected_keys"],
            "total_params": int(sum(p.numel() for p in model.parameters())),
            "allowlist_tensors": len(trainable),
            "allowlist_params": int(sum(p.numel() for _, p in trainable)),
            "frozen_params": int(
                sum(
                    p.numel()
                    for name, p in model.named_parameters()
                    if not name.startswith(tuple(__import__("net.path_head", fromlist=["x"]).TRAIN_ALLOWLIST_PREFIXES))
                )
            ),
            "allowlist_names": allow_names,
        },
        "epochs": [],
        "steps": [],
        "grad_probe": None,
        "stop": {"reason": "running", "steps_done": 0, "wall_s": None},
    }

    def evaluate(split: str) -> dict:
        index_pool = val_idx if split == "val" else train_idx
        totals = {"total": 0.0, "l_action": 0.0, "l_plan": 0.0, "lateral": 0.0}
        lateral_terms: dict[str, float] = {}
        batches = 0
        with torch.no_grad():
            for start in range(0, index_pool.size, int(args.val_batch_size)):
                index = index_pool[start : start + int(args.val_batch_size)]
                obs = build_obs(data["rows"], main_ids[index], device=_device_of(args), base_dataset=view["base"])
                terms = batch_loss_terms(model, obs, batch_tensors(view, index, _device_of(args)), config, args)
                totals["total"] += float(terms["total"])
                totals["l_action"] += float(terms["l_action"])
                totals["l_plan"] += float(terms["l_plan"])
                totals["lateral"] += float(terms["lateral_total"])
                for key, value in terms["lateral_terms"].items():
                    lateral_terms[key] = lateral_terms.get(key, 0.0) + float(value)
                batches += 1
        scale = 1.0 / max(1, batches)
        return {
            "split": split,
            "n_pairs": int(index_pool.size),
            "batches": batches,
            **{key: value * scale for key, value in totals.items()},
            "lateral_terms": {key: value * scale for key, value in lateral_terms.items()},
        }

    def save_ckpt(tag: str, epoch_label: int, val: dict) -> str:
        from pipeline.stages import save_checkpoint

        out_dir = train_out_dir(args)
        payload = {
            "arm": args.arm,
            "round": "round41",
            "lane": "g",
            "task": f"lateral_{args.arm}",
            "epoch": int(epoch_label),
            "loss_config": plan["loss_config"],
            "weights": {"action": plan["action_weight"], "chain": plan["chain_weight"]},
            "plan": plan,
            "val": val,
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "data": data["meta"],
            "allowlist": allow_names,
            "load_info_ckpt_sha256": load_info["ckpt_sha256"],
            "geometry": {"readout_lane_geometry": str(model.readout_lane_geometry)},
            "harness": _json_sanitize(harness_info()),
        }
        path = str(out_dir / tag)
        save_checkpoint(path, model, meta=payload, optimizer=optimizer, epoch=int(epoch_label), val_metrics=val)
        return path

    train_out_dir(args).mkdir(parents=True, exist_ok=True)
    stop_reason = "completed"
    steps_done = 0
    step_times: list[float] = []
    budget_steps = int(args.max_steps)
    budget_reduced: dict | None = None
    ckpt_points: list[dict] = []
    selected: dict | None = None
    epoch_index = 0
    run_started = time.time()
    if smoke:
        budget_steps = min(budget_steps, int(args.smoke_steps))
    while epoch_index < int(args.epochs) and stop_reason == "completed":
        epoch_index += 1
        order = train_idx[np_rng.permutation(n_train)]
        sums = {"total": 0.0, "l_action": 0.0, "l_plan": 0.0, "lateral": 0.0}
        epoch_batches = 0
        for start in range(0, order.size, int(args.batch_size)):
            index = order[start : start + int(args.batch_size)]
            device = _device_of(args)
            obs = build_obs(data["rows"], main_ids[index], device=device, base_dataset=view["base"])
            batch = batch_tensors(view, index, device)
            step_started = time.time()
            terms = batch_loss_terms(model, obs, batch, config, args)
            if ledger["grad_probe"] is None and bool(args.grad_probe):
                # 首步分项梯度探针（同图 3 次 autograd.grad；不写参数、不吞图：retain_graph ⇒ 之后可正常 backward）
                ledger["grad_probe"] = _grad_probe(
                    model, obs, batch, config, args, trainable, terms=terms
                )
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], float(args.grad_clip))
            optimizer.step()
            steps_done += 1
            epoch_batches += 1
            for key in sums:
                value = terms["lateral_total"] if key == "lateral" else terms[key]
                sums[key] += float(value)
            step_times.append(time.time() - step_started)
            if steps_done % max(1, int(args.log_every)) == 0 or steps_done == 1:
                recent = float(np.mean(step_times[-min(50, len(step_times)) :]))
                ledger["steps"].append(
                    {
                        "step": steps_done,
                        "epoch": epoch_index,
                        "total": float(terms["total"]),
                        "l_action": float(terms["l_action"]),
                        "l_plan": float(terms["l_plan"]),
                        "lateral": float(terms["lateral_total"]),
                        "lateral_terms": {k: float(v) for k, v in terms["lateral_terms"].items()},
                        "step_time_s": recent,
                    }
                )
                print(
                    f"[lateral] epoch {epoch_index} step {steps_done} total={float(terms['total']):.6f} "
                    f"action={float(terms['l_action']):.6f} plan={float(terms['l_plan']):.6f} "
                    f"lateral={float(terms['lateral_total']):.6f}（{recent:.3f} s/step）",
                    flush=True,
                )
            if steps_done == int(args.calib_steps) and step_times:
                mean_step = float(np.mean(step_times))
                remaining_steps = int(min(plan["max_steps"], plan["epochs"] * plan["steps_per_epoch"])) - steps_done
                projected = mean_step * remaining_steps
                print(
                    f"[lateral] 成本标定：步时 {mean_step:.4f} s（n={len(step_times)}）⇒ 预计剩余 "
                    f"{projected/60:.1f} min；预算 {float(args.max_wall_s)/60:.1f} min",
                    flush=True,
                )
                ledger["calibration"] = {
                    "steps": steps_done,
                    "mean_step_s": mean_step,
                    "projected_remaining_s": projected,
                    "budget_s": float(args.max_wall_s),
                }
                if projected > float(args.max_wall_s):
                    fit = max(1, int(float(args.max_wall_s) / mean_step) - steps_done)
                    budget_reduced = {
                        "reason": "projected_over_budget",
                        "max_steps_before": budget_steps,
                        "max_steps_after": steps_done + fit,
                        "projected_s": projected,
                        "budget_s": float(args.max_wall_s),
                    }
                    budget_steps = steps_done + fit
                    print(f"[lateral] WARN：投影超预算 ⇒ 降 step 至 {budget_steps}（记账不静默）", flush=True)
            if steps_done >= budget_steps:
                stop_reason = "max_steps"
            if (time.time() - run_started) > float(args.max_wall_s):
                stop_reason = "wall_budget"
            if stop_reason != "completed":
                print(
                    f"[lateral] 训练停止：{stop_reason}（steps={steps_done}, wall={time.time()-run_started:.1f}s）",
                    flush=True,
                )
                break
        scale = 1.0 / max(1, epoch_batches)
        train_eval = {key: value * scale for key, value in sums.items()}
        val_eval = evaluate("val")
        epoch_record = {
            "epoch": epoch_index,
            "steps_done": steps_done,
            "train_loss_mean": train_eval,
            "val": val_eval,
            "elapsed_s": round(time.time() - run_started, 1),
        }
        ledger["epochs"].append(epoch_record)
        print(
            f"[lateral] epoch {epoch_index} 完：train total={train_eval['total']:.6f} / "
            f"val total={val_eval['total']:.6f}"
            f"（action={val_eval['l_action']:.6f} plan={val_eval['l_plan']:.6f} lateral={val_eval['lateral']:.6f}）",
            flush=True,
        )
        if not smoke:
            tag = f"ckpt_epoch{epoch_index:03d}.pt"
            path = save_ckpt(tag, epoch_index, val_eval)
            ckpt_points.append({"epoch": epoch_index, "path": path, "val_total": val_eval["total"]})
            if selected is None or val_eval["total"] < selected["val_total"]:
                selected = {"epoch": epoch_index, "path": path, "val_total": val_eval["total"]}
        if stop_reason != "completed":
            break
    if stop_reason == "completed" and steps_done >= budget_steps:
        stop_reason = "max_steps"

    # ---- 冻结面逐位守卫（训练后） ----
    frozen_after = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name not in set(allow_names)
    }
    max_frozen_delta = 0.0
    changed = []
    for name, before in frozen_before.items():
        after = frozen_after.get(name)
        if after is None:
            changed.append(name)
            continue
        delta = float((before - after).abs().max())
        if delta > 0.0:
            changed.append(name)
        max_frozen_delta = max(max_frozen_delta, delta)
    ledger["frozen_guard"] = {
        "n_frozen_tensors": len(frozen_before),
        "changed_tensors": changed[:10],
        "max_abs_delta": max_frozen_delta,
        "bitwise_unchanged": not changed,
    }
    ledger["stop"] = {
        "reason": stop_reason,
        "steps_done": steps_done,
        "wall_s": round(sum(step_times), 1),
        "budget": {"max_steps": budget_steps, "max_wall_s": float(args.max_wall_s), "reduction": budget_reduced},
    }
    ledger["selection"] = (
        {"rule": "val 最优（episode 级；仅 val）", "points": ckpt_points, "best": selected}
        if not smoke
        else {"rule": "smoke-steps：无选点", "points": [], "best": None}
    )
    if not smoke and selected is not None:
        from shutil import copyfile

        best_path = train_out_dir(args) / "ckpt_selected.pt"
        copyfile(selected["path"], best_path)
        selected["selected_copy"] = str(best_path)
    ledger["hashes"] = {
        "net_tree": _net_tree_digest(),
        "harness_sha256": _combined_harness_sha(harness_info()),
        "harness": _json_sanitize(harness_info()),
        "data_identity_sha256": hashlib.sha256(
            (
                json.dumps(data["meta"], sort_keys=True, default=str)
                + "|"
                + hashlib.sha256(np.asarray(main_ids, dtype=np.int64).tobytes()).hexdigest()
            ).encode("utf-8")
        ).hexdigest(),
        "ckpt_sha256": {Path(point["path"]).name: _sha256(Path(point["path"])) for point in ckpt_points},
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    ledger["elapsed_s"] = round(sum(step_times), 1)
    ledger["wall_s"] = round(time.time() - run_started, 1)
    out_dir = train_out_dir(args)
    log_name = "smoke_steps.json" if smoke else "train_log.json"
    log_path = out_dir / log_name
    log_path.write_text(json.dumps(_json_sanitize(ledger), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[lateral] → {log_path}", flush=True)
    if smoke:
        per_step = float(np.mean(step_times)) if step_times else None
        print(
            json.dumps(
                {
                    "mode": "smoke-steps",
                    "arm": args.arm,
                    "steps": steps_done,
                    "mean_step_s": per_step,
                    "s_per_1000_steps": None if per_step is None else per_step * 1000,
                    "projected_2epoch_min": None
                    if per_step is None
                    else per_step * plan["steps_per_epoch"] * plan["epochs"] / 60.0,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    if not ledger["frozen_guard"]["bitwise_unchanged"]:  # pragma: no cover - 结构性排除
        raise SystemExit("[lateral] fail-fast：冻结面出现改动（见 train_log.frozen_guard）")
    print(
        f"[lateral] 训练完成：steps={steps_done} 选点 epoch={selected['epoch'] if selected else None} "
        f"val={selected['val_total'] if selected else None}",
        flush=True,
    )
    return 0


def train_out_dir(args) -> Path:
    """训练产物目录（默认 ``runs/dev/round41/<arm>``；F 预注册 §4 ckpt 命名域）。"""
    raw = args.train_out or f"runs/dev/round41/{args.arm}"
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    import torch

    started = time.time()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if not (args.dry_run or args.check_features or args.assemble_smoke or training_mode(args)):
        raise SystemExit(
            "[lateral] fail-fast：未知运行模式；"
            "请用 --dry-run / --check-features / --assemble-smoke / --train / --smoke-steps N"
        )
    data = build_lateral_data(args)
    print(f"[lateral] pairs {data['meta']}", flush=True)
    if training_mode(args):
        # R41 第二批执行-G：正式训练循环（写 runs/dev/round41/<arm>/**）
        return train_loop(args, data)
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
