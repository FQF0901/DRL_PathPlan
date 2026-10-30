"""Phase 2b joint policy + history-WM fine-tuning.

The encoder and target encoder stay frozen.  The shared TSB is unfrozen with a
separate small learning rate; the policy head and WM/decoder are trained
jointly on the canonical train/val split.  This is an optional post-training
stage, not the default IL/DAgger trainer.
"""

from __future__ import annotations


if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from env.action.action_space import DynamicsLimits
from net.encoder import TargetEncoder
from pipeline.checkpointing import load_drive_model_from_checkpoint, save_drive_model_checkpoint
from pipeline.losses import wm_losses
from pipeline.train_bc import (
    LossConfig,
    _batch_from_indices,
    build_targets,
    compute_losses,
    load_data,
)
from pipeline.train_config import parse_train_args, timestamped_run_dir
from pipeline.train_wm import (
    _build_pairs,
    _make_training_arrays,
    _persistence,
)


def _weighted_hard_limits(drive) -> DynamicsLimits | None:
    limits = getattr(drive, "action_config", {}).get("limits", {}).get("hard")
    return DynamicsLimits(**limits) if limits else None


def _batch_tensors(arrays: Mapping[str, np.ndarray], idx: np.ndarray, device: str) -> Dict[str, torch.Tensor]:
    keys = (
        "od_target", "ld_target", "od_tmask", "ld_tmask",
        "od_geometry", "ld_geometry", "od_gmask", "ld_gmask", "action",
    )
    return {
        key: torch.from_numpy(np.asarray(arrays[key][idx], dtype=np.float32)).to(device)
        for key in keys
    }


def _joint_step(
    drive,
    policy_batch: Mapping[str, torch.Tensor],
    wm_batch: Mapping[str, torch.Tensor],
    cfg: LossConfig,
    device: str,
    *,
    hard_limits,
    content_weight: float,
    geometry_weight: float,
    policy_weight: float,
    wm_weight: float,
):
    # One TSB pass feeds both heads.  The policy batch already contains the
    # 6-frame history and stable IDs.
    mode = policy_batch.get("history_mode")
    zero_pose = torch.ones(policy_batch["od"].shape[0], dtype=torch.bool, device=device)
    if torch.is_tensor(mode):
        mode = mode.reshape(-1)
        if mode.numel() == zero_pose.numel():
            zero_pose = mode < 0.5
    window, last_enc = drive.build_history_window(policy_batch, zero_pose=zero_pose)
    h = drive.temporal_spatial(window, last_enc["ego"], ego_valid=last_enc.get("ego_valid"))
    policy_out = drive.planner(
        h["od"], h["ld"], last_enc["nav"], last_enc["ego"],
        window.od_valid[:, -1], window.ld_valid[:, -1],
    )
    policy_loss, policy_parts = compute_losses(
        policy_out, policy_batch, cfg,
        action_low=drive.action_low.detach().cpu().numpy(),
        action_high=drive.action_high.detach().cpu().numpy(),
        hard_limits=hard_limits,
    )
    wm_out = drive.world_model(
        h["od"], h["ld"], wm_batch["action"],
        od_base=window.od_content[:, -1],
        ld_base=window.ld_content[:, -1],
        od_valid=window.od_valid[:, -1],
        ld_valid=window.ld_valid[:, -1],
    )
    wm_parts = wm_losses(
        wm_out,
        od_target=wm_batch["od_target"],
        ld_target=wm_batch["ld_target"],
        od_valid=wm_batch["od_tmask"],
        ld_valid=wm_batch["ld_tmask"],
        od_geometry_target=wm_batch["od_geometry"],
        ld_geometry_target=wm_batch["ld_geometry"],
        od_geometry_mask=wm_batch["od_gmask"],
        ld_geometry_mask=wm_batch["ld_gmask"],
        content_weight=content_weight,
        od_geometry_weight=geometry_weight,
        ld_geometry_weight=geometry_weight,
    )
    total = float(policy_weight) * policy_loss + float(wm_weight) * wm_parts["total"]
    return total, policy_parts, wm_parts


def _evaluate_joint(
    drive, data, targets, pairs, wm_arrays, cfg, device, *,
    hard_limits, content_weight, geometry_weight, policy_weight, wm_weight,
    batch_size,
):
    drive.eval()
    total_policy = total_wm = 0.0
    n = 0
    idxs = np.arange(len(pairs), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(idxs), int(batch_size)):
            idx = idxs[start : start + int(batch_size)]
            rows = pairs[idx]
            policy_batch = _batch_from_indices(data, targets, rows, device)
            wm_batch = _batch_tensors(wm_arrays, idx, device)
            _, pp, wp = _joint_step(
                drive, policy_batch, wm_batch, cfg, device,
                hard_limits=hard_limits,
                content_weight=content_weight,
                geometry_weight=geometry_weight,
                policy_weight=policy_weight,
                wm_weight=wm_weight,
            )
            total_policy += float(pp["total"].detach())
            total_wm += float(wp["total"].detach())
            n += 1
    persistence = _persistence(
        wm_arrays, idxs,
        content_weight=content_weight, geometry_weight=geometry_weight,
    )
    return total_policy / max(1, n), total_wm / max(1, n), persistence


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Optional joint policy + WM fine-tune")
    p.add_argument(
        "--config",
        default="config/train_joint.yaml",
        help="YAML training config (CLI flags override it)",
    )
    p.add_argument("--data", nargs="+", default=None)
    p.add_argument("--val-data", nargs="+", default=None)
    p.add_argument("--val-split", default="data/splits/val_500.json")
    p.add_argument("--ckpt", default=None, help="DriveModel policy/WM checkpoint")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-pairs", type=int, default=60000)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--tsb-lr", type=float, default=1e-6)
    p.add_argument("--policy-weight", type=float, default=1.0)
    p.add_argument("--wm-weight", type=float, default=1.0)
    p.add_argument(
        "--policy-tol", type=float, default=0.05,
        help="max relative val-policy degradation allowed for a best checkpoint",
    )
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--geometry-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parse_train_args(p, argv, "config/train_joint.yaml")
    if not args.data:
        raise SystemExit("--data is required (CLI or config)")
    if not args.val_data:
        raise SystemExit("--val-data is required (CLI or config)")
    if not args.ckpt:
        raise SystemExit("--ckpt is required (CLI or config)")
    if not args.out:
        raise SystemExit("--out is required (CLI or config)")
    args.out = str(timestamped_run_dir(args.out))
    print(f"[joint] config={args.config} run_dir={args.out}")
    if args.epochs < 1:
        raise SystemExit("--epochs must be >= 1")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    train_data = load_data(args.data)
    val_data = load_data(args.val_data)
    expected = {int(x) for x in json.loads(Path(args.val_split).read_text())["seeds"]}
    actual = {int(x) for x in np.unique(val_data["scene_seed"])}
    if actual != expected:
        raise ValueError(
            f"validation scene seeds do not match {args.val_split}: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    overlap = {int(x) for x in np.unique(train_data["scene_seed"])} & actual
    if overlap:
        raise ValueError(f"joint train/val scene overlap: {sorted(overlap)[:10]}")

    drive = load_drive_model_from_checkpoint(args.ckpt, args.device, strict=True)
    drive.to(args.device)
    encoder = drive.encoder
    for p_ in encoder.parameters():
        p_.requires_grad_(False)
    encoder.eval()
    target = TargetEncoder(encoder).to(args.device).eval()

    # TSB is intentionally unfrozen only in this optional joint stage.
    for p_ in drive.temporal_spatial.parameters():
        p_.requires_grad_(True)
    for p_ in drive.planner.parameters():
        p_.requires_grad_(True)
    for p_ in drive.world_model.parameters():
        p_.requires_grad_(True)

    action_low = drive.action_low.detach().cpu().numpy()
    action_high = drive.action_high.detach().cpu().numpy()
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw_loss = (payload.get("config", {}) or {}).get("loss", {})
    allowed = {f.name for f in fields(LossConfig)}
    cfg = LossConfig(**{k: float(v) for k, v in raw_loss.items() if k in allowed}) if isinstance(raw_loss, Mapping) else LossConfig()
    policy_targets = build_targets(train_data, cfg, action_low=action_low, action_high=action_high)
    val_targets = build_targets(val_data, cfg, action_low=action_low, action_high=action_high)
    train_pairs, _ = _build_pairs(train_data, int(args.max_pairs))
    val_pairs, _ = _build_pairs(val_data, int(args.max_pairs))
    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise ValueError("joint training needs consecutive-frame pairs")
    wm_train = _make_training_arrays(
        train_data, drive, target, args.device, int(args.max_pairs),
        action_low=action_low, action_high=action_high,
    )
    wm_val = _make_training_arrays(
        val_data, drive, target, args.device, int(args.max_pairs),
        action_low=action_low, action_high=action_high,
    )
    if len(wm_train["scene_seed"]) != len(train_pairs) or len(wm_val["scene_seed"]) != len(val_pairs):
        raise ValueError("joint policy/WM pair alignment mismatch")

    tsb_params = list(drive.temporal_spatial.parameters())
    other_params = list(drive.planner.parameters()) + list(drive.world_model.parameters())
    opt = torch.optim.AdamW([
        {"params": tsb_params, "lr": float(args.tsb_lr)},
        {"params": other_params, "lr": float(args.lr)},
    ], weight_decay=1e-4)
    hard = _weighted_hard_limits(drive)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    initial_policy, initial_wm, initial_persistence = _evaluate_joint(
        drive, val_data, val_targets, val_pairs, wm_val, cfg, args.device,
        hard_limits=hard,
        content_weight=args.content_weight,
        geometry_weight=args.geometry_weight,
        policy_weight=args.policy_weight,
        wm_weight=args.wm_weight,
        batch_size=args.batch_size,
    )
    best_val = initial_wm
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in drive.state_dict().items()}
    for epoch in range(1, int(args.epochs) + 1):
        drive.train()
        drive.encoder.eval()
        perm = np.random.default_rng(int(args.seed) + epoch).permutation(len(train_pairs))
        sums = {"policy": 0.0, "wm": 0.0}
        n_batches = 0
        for start in range(0, len(perm), int(args.batch_size)):
            idx = perm[start : start + int(args.batch_size)]
            rows = train_pairs[idx]
            policy_batch = _batch_from_indices(train_data, policy_targets, rows, args.device)
            wm_batch = _batch_tensors(wm_train, idx, args.device)
            total, pp, wp = _joint_step(
                drive, policy_batch, wm_batch, cfg, args.device,
                hard_limits=hard,
                content_weight=args.content_weight,
                geometry_weight=args.geometry_weight,
                policy_weight=args.policy_weight,
                wm_weight=args.wm_weight,
            )
            policy_loss = float(args.policy_weight) * pp["total"]
            wm_loss = float(args.wm_weight) * wp["total"]
            opt.zero_grad(set_to_none=True)
            policy_loss.backward(retain_graph=True)
            policy_grads = {}
            for name, params in (("tsb", tsb_params), ("heads", other_params)):
                sq = sum(float(p.grad.detach().pow(2).sum()) for p in params if p.grad is not None)
                policy_grads[name] = sq ** 0.5
                for p in params:
                    if p.grad is not None:
                        p._joint_policy_grad = p.grad.detach().clone()
            opt.zero_grad(set_to_none=True)
            wm_loss.backward()
            wm_grads = {}
            for name, params in (("tsb", tsb_params), ("heads", other_params)):
                sq = sum(float(p.grad.detach().pow(2).sum()) for p in params if p.grad is not None)
                wm_grads[name] = sq ** 0.5
                for p in params:
                    policy_grad = getattr(p, "_joint_policy_grad", None)
                    if policy_grad is None:
                        continue
                    if p.grad is None:
                        p.grad = policy_grad
                    else:
                        p.grad = p.grad + policy_grad
            for name, params in (("tsb", tsb_params), ("heads", other_params)):
                for p in params:
                    if hasattr(p, "_joint_policy_grad"):
                        del p._joint_policy_grad
            torch.nn.utils.clip_grad_norm_(tsb_params + other_params, 1.0)
            opt.step()
            sums["policy"] += float(pp["total"].detach())
            sums["wm"] += float(wp["total"].detach())
            for key, val in policy_grads.items():
                sums[f"policy_grad_{key}"] = sums.get(f"policy_grad_{key}", 0.0) + val
            for key, val in wm_grads.items():
                sums[f"wm_grad_{key}"] = sums.get(f"wm_grad_{key}", 0.0) + val
            n_batches += 1
        val_policy, val_wm, persistence = _evaluate_joint(
            drive, val_data, val_targets, val_pairs, wm_val, cfg, args.device,
            hard_limits=hard,
            content_weight=args.content_weight,
            geometry_weight=args.geometry_weight,
            policy_weight=args.policy_weight,
            wm_weight=args.wm_weight,
            batch_size=args.batch_size,
        )
        record = {
            "epoch": epoch,
            "train_policy": sums["policy"] / max(1, n_batches),
            "train_wm": sums["wm"] / max(1, n_batches),
            "val_policy": val_policy,
            "val_wm": val_wm,
            "persistence": persistence,
            "grad_norm_policy_tsb": sums.get("policy_grad_tsb", 0.0) / max(1, n_batches),
            "grad_norm_policy_heads": sums.get("policy_grad_heads", 0.0) / max(1, n_batches),
            "grad_norm_wm_tsb": sums.get("wm_grad_tsb", 0.0) / max(1, n_batches),
            "grad_norm_wm_heads": sums.get("wm_grad_heads", 0.0) / max(1, n_batches),
        }
        history.append(record)
        print(f"[joint] {record}", flush=True)
        policy_budget = float(args.policy_tol) * max(abs(initial_policy), 1e-6)
        policy_ok = (val_policy - initial_policy) <= policy_budget
        if val_wm < best_val and policy_ok:
            best_val = val_wm
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in drive.state_dict().items()}

    extra = {
        "loss": asdict(cfg),
        "target_encoder": args.ckpt,
        "train_scene_seeds": sorted({int(x) for x in train_data["scene_seed"]}),
        "val_scene_seeds": sorted({int(x) for x in val_data["scene_seed"]}),
        "joint_phase": "2b",
        "config_file": getattr(args, "config", None),
        "run_dir": str(out),
    }
    save_drive_model_checkpoint(drive, str(out / "last_drive.pt"), extra=extra)
    drive.load_state_dict(best_state, strict=True)
    save_drive_model_checkpoint(drive, str(out / "best_drive.pt"), extra=extra)
    (out / "metrics.json").write_text(
        json.dumps({
            "history": history,
            "data": list(args.data),
            "train_scene_seeds": sorted({int(x) for x in train_data["scene_seed"]}),
            "val_scene_seeds": sorted({int(x) for x in val_data["scene_seed"]}),
            "initial": {"val_policy": initial_policy, "val_wm": initial_wm, "persistence": initial_persistence},
            "best_epoch": best_epoch,
            "best_val_wm": best_val,
            "ckpt": args.ckpt,
            "config_file": args.config,
            "run_dir": str(out),
            "val_data": list(args.val_data),
            "val_split": args.val_split,
            "policy_weight": float(args.policy_weight),
            "wm_weight": float(args.wm_weight),
            "content_weight": float(args.content_weight),
            "geometry_weight": float(args.geometry_weight),
            "tsb_lr": float(args.tsb_lr),
            "lr": float(args.lr),
            "max_pairs": int(args.max_pairs),
        }, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
