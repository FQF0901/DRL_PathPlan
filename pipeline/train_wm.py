"""Train the bank-native one-step latent world model.

The online encoder is frozen; a separate target encoder supplies stop-grad
next-frame latents.  The WM predicts ``r_z`` and direct OD/LD geometry, and the
losses live in :mod:`pipeline.losses`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from dataset.splits import split_by_scene_seed
from env.action.config import load_action_contract
from env.obs import HISTORY_FRAMES
from net.encoder import TargetEncoder
from net.world_model import EgoConditionedLatentWM
from pipeline.checkpointing import load_drive_model_from_checkpoint
from pipeline.losses import wm_losses
from pipeline.train_bc import _resolve_npz_paths, load_data


def _load_ids(paths: Sequence[str]) -> Dict[str, np.ndarray]:
    od, ld = [], []
    for raw in paths:
        for npz_path in _resolve_npz_paths(Path(raw)):
            with np.load(npz_path) as z:
                od.append(np.asarray(z["od_ids"], dtype=np.int64))
                ld.append(np.asarray(z["ld_ids"], dtype=np.int64))
    if not od:
        raise ValueError("no datasets for WM metadata")
    return {"od_ids": np.concatenate(od, axis=0), "ld_ids": np.concatenate(ld, axis=0)}


def _rows_to_batch(data: Dict[str, np.ndarray], idx: np.ndarray, device: str) -> Dict[str, Tensor]:
    """Current-frame encoder batch from history rows."""
    keys = ("od", "od_mask", "ld", "ld_mask", "nav", "nav_mask", "ego", "ego_mask")
    out: Dict[str, Tensor] = {}
    for key in keys:
        arr = np.asarray(data[key][idx, -1], dtype=np.float32)
        out[key] = torch.from_numpy(arr).to(device)
    out["od_ids"] = torch.from_numpy(np.asarray(data["od_ids"][idx, -1], dtype=np.int64)).to(device)
    out["ld_ids"] = torch.from_numpy(np.asarray(data["ld_ids"][idx, -1], dtype=np.int64)).to(device)
    if "pose" in data:
        out["pose"] = torch.from_numpy(np.asarray(data["pose"][idx, -1], dtype=np.float32)).to(device)
    return out


@torch.no_grad()
def _encode_rows(encoder, data: Dict[str, np.ndarray], rows: np.ndarray, device: str, batch_size: int = 256):
    chunks: List[Dict[str, np.ndarray]] = []
    for start in range(0, len(rows), batch_size):
        idx = rows[start : start + batch_size]
        out = encoder.encode_current(_rows_to_batch(data, idx, device))
        chunks.append(
            {
                key: out[key].detach().cpu().numpy()
                for key in ("od_content", "ld_content", "od_geometry", "ld_geometry", "od_valid", "ld_valid")
            }
        )
    if not chunks:
        raise ValueError("no rows to encode")
    return {key: np.concatenate([c[key] for c in chunks], axis=0) for key in chunks[0]}


def _build_pairs(data: Dict[str, np.ndarray], max_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    pairs: List[Tuple[int, int]] = []
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    step = np.asarray(data["step"], dtype=np.int64)
    for eid in np.unique(episode_id):
        idx = np.flatnonzero(episode_id == eid)
        idx = idx[np.argsort(step[idx])]
        for a, b in zip(idx[:-1], idx[1:]):
            if int(step[b]) == int(step[a]) + 1:
                pairs.append((int(a), int(b)))
    if max_pairs > 0 and len(pairs) > int(max_pairs):
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(len(pairs), size=int(max_pairs), replace=False))
        pairs = [pairs[int(i)] for i in keep]
    return np.asarray([p[0] for p in pairs], np.int64), np.asarray([p[1] for p in pairs], np.int64)


def _slot_align(cur, cur_mask, tgt, tgt_mask) -> Tuple[np.ndarray, np.ndarray]:
    out = np.zeros_like(cur, dtype=np.float32)
    mask = np.zeros_like(cur_mask, dtype=np.float32)
    for j in range(cur.shape[0]):
        if cur_mask[j] > 0.5 and tgt_mask[j] > 0.5:
            out[j] = tgt[j]
            mask[j] = 1.0
    return out, mask


def _align_by_id(cur, cur_ids, cur_mask, tgt, tgt_ids, tgt_mask) -> Tuple[np.ndarray, np.ndarray]:
    valid_cur = cur_mask > 0.5
    valid_tgt = tgt_mask > 0.5
    usable = sorted({int(x) for x in cur_ids[valid_cur]}) if valid_cur.any() else []
    # Legacy/synthetic data often has 0 for every slot.  Slot alignment is the
    # only meaningful fallback there; do not map all entities to target[0].
    if not usable or set(usable) == {0}:
        return _slot_align(cur, cur_mask, tgt, tgt_mask)
    out = np.zeros_like(cur, dtype=np.float32)
    mask = np.zeros_like(cur_mask, dtype=np.float32)
    for j in range(cur.shape[0]):
        if cur_mask[j] <= 0.5:
            continue
        ident = int(cur_ids[j])
        hit = np.flatnonzero(tgt_ids == ident)
        if hit.size == 0 or tgt_mask[int(hit[0])] <= 0.5:
            continue
        out[j] = tgt[int(hit[0])]
        mask[j] = 1.0
    return out, mask


def _align_geometry_by_id(cur_ids, cur_mask, tgt, tgt_ids, tgt_mask) -> Tuple[np.ndarray, np.ndarray]:
    valid_cur = cur_mask > 0.5
    valid_tgt = tgt_mask > 0.5
    usable = sorted({int(x) for x in cur_ids[valid_cur]}) if valid_cur.any() else []
    if not usable or set(usable) == {0}:
        out = np.zeros_like(tgt, dtype=np.float32)
        mask = np.zeros_like(cur_mask, dtype=np.float32)
        for j in range(len(cur_ids)):
            if cur_mask[j] > 0.5 and tgt_mask[j] > 0.5:
                out[j] = tgt[j]
                mask[j] = 1.0
        return out, mask
    out = np.zeros_like(tgt, dtype=np.float32)
    mask = np.zeros_like(cur_mask, dtype=np.float32)
    for j in range(len(cur_ids)):
        if cur_mask[j] <= 0.5:
            continue
        hit = np.flatnonzero(tgt_ids == int(cur_ids[j]))
        if hit.size == 0 or tgt_mask[int(hit[0])] <= 0.5:
            continue
        out[j] = tgt[int(hit[0])]
        mask[j] = 1.0
    return out, mask


def _normalize_executed(actions: np.ndarray, low=None, high=None) -> np.ndarray:
    if low is None or high is None:
        contract = load_action_contract()
        low, high = contract.bounds.low, contract.bounds.high
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    width = np.maximum(high - low, 1e-6)
    return ((actions - low) / width).astype(np.float32)


def _make_training_arrays(
    data: Dict[str, np.ndarray], encoder, target_encoder, device: str, max_pairs: int,
    *, action_low=None, action_high=None,
):
    cur_rows, next_rows = _build_pairs(data, max_pairs)
    if len(cur_rows) == 0:
        raise ValueError("no consecutive-frame pairs found")
    cur = _encode_rows(encoder, data, cur_rows, device)
    nxt = _encode_rows(target_encoder, data, next_rows, device)
    out = {
        "od": cur["od_content"], "ld": cur["ld_content"],
        "od_valid": cur["od_valid"], "ld_valid": cur["ld_valid"],
        "od_target": np.zeros_like(cur["od_content"]),
        "ld_target": np.zeros_like(cur["ld_content"]),
        "od_tmask": np.zeros_like(cur["od_valid"]),
        "ld_tmask": np.zeros_like(cur["ld_valid"]),
        "od_geometry": np.zeros_like(cur["od_geometry"]),
        "ld_geometry": np.zeros_like(cur["ld_geometry"]),
        "od_gmask": np.zeros_like(cur["od_valid"]),
        "ld_gmask": np.zeros_like(cur["ld_valid"]),
        "scene_seed": np.asarray(data["scene_seed"][cur_rows], dtype=np.int64),
    }
    for p in range(len(cur_rows)):
        cur_ids_od = data["od_ids"][cur_rows[p], -1]
        tgt_ids_od = data["od_ids"][next_rows[p], -1]
        cur_ids_ld = data["ld_ids"][cur_rows[p], -1]
        tgt_ids_ld = data["ld_ids"][next_rows[p], -1]
        out["od_target"][p], out["od_tmask"][p] = _align_by_id(
            cur["od_content"][p], cur_ids_od, cur["od_valid"][p],
            nxt["od_content"][p], tgt_ids_od, nxt["od_valid"][p],
        )
        out["ld_target"][p], out["ld_tmask"][p] = _align_by_id(
            cur["ld_content"][p], cur_ids_ld, cur["ld_valid"][p],
            nxt["ld_content"][p], tgt_ids_ld, nxt["ld_valid"][p],
        )
        out["od_geometry"][p], out["od_gmask"][p] = _align_geometry_by_id(
            cur_ids_od, cur["od_valid"][p], nxt["od_geometry"][p], tgt_ids_od, nxt["od_valid"][p],
        )
        out["ld_geometry"][p], out["ld_gmask"][p] = _align_geometry_by_id(
            cur_ids_ld, cur["ld_valid"][p], nxt["ld_geometry"][p], tgt_ids_ld, nxt["ld_valid"][p],
        )
        # New target-only entities have no current slot.  Put their direct
        # geometry into free current slots so they receive supervision.
        for prefix, cur_ids, tgt_ids, cur_mask, tgt_mask, geom_key, gmask_key in (
            ("od", cur_ids_od, tgt_ids_od, cur["od_valid"][p], nxt["od_valid"][p], "od_geometry", "od_gmask"),
            ("ld", cur_ids_ld, tgt_ids_ld, cur["ld_valid"][p], nxt["ld_valid"][p], "ld_geometry", "ld_gmask"),
        ):
            known = {int(x) for x, m in zip(cur_ids, cur_mask) if m > 0.5}
            free = [i for i, m in enumerate(cur_mask) if m <= 0.5]
            for j, (ident, valid) in enumerate(zip(tgt_ids, tgt_mask)):
                if valid <= 0.5 or int(ident) in known or not free:
                    continue
                slot = free.pop(0)
                out[geom_key][p, slot] = nxt[geom_key][p, j]
                out[gmask_key][p, slot] = 1.0
    phys = (
        np.where(
            np.isfinite(np.asarray(data.get("executed_action", np.full((len(data["action"]), 4), np.nan)), dtype=np.float32)[cur_rows]),
            np.asarray(data.get("executed_action", np.full((len(data["action"]), 4), np.nan)), dtype=np.float32)[cur_rows],
            np.asarray(data["action"][cur_rows], dtype=np.float32),
        )
        if "executed_action" in data
        else np.asarray(data["action"][cur_rows], dtype=np.float32)
    )
    out["action"] = _normalize_executed(phys, action_low, action_high)
    return out


def _tensors(arrays: Dict[str, np.ndarray], idx: np.ndarray, device: str) -> Dict[str, Tensor]:
    keys = (
        "od", "ld", "od_valid", "ld_valid", "od_target", "ld_target",
        "od_tmask", "ld_tmask", "od_geometry", "ld_geometry", "od_gmask", "ld_gmask",
        "action",
    )
    return {key: torch.from_numpy(np.asarray(arrays[key][idx], dtype=np.float32)).to(device) for key in keys}


def _loss(wm, batch: Dict[str, Tensor]):
    pred = wm(batch["od"], batch["ld"], batch["action"], od_valid=batch["od_valid"], ld_valid=batch["ld_valid"])
    return wm_losses(
        pred,
        od_target=batch["od_target"], ld_target=batch["ld_target"],
        od_valid=batch["od_tmask"], ld_valid=batch["ld_tmask"],
        od_geometry_target=batch["od_geometry"], ld_geometry_target=batch["ld_geometry"],
        od_geometry_mask=batch["od_gmask"], ld_geometry_mask=batch["ld_gmask"],
    )


def _persistence(arrays: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, float]:
    def mse(a, b, m):
        mask = np.asarray(m, dtype=np.float32)
        while mask.ndim < a.ndim:
            mask = mask[..., None]
        mask = np.broadcast_to(mask, a.shape)
        denom = max(float(mask.sum()), 1.0)
        return float((((a - b) ** 2) * mask).sum() / denom)

    def mae(a, b, m):
        mask = np.asarray(m, dtype=np.float32)
        while mask.ndim < a.ndim:
            mask = mask[..., None]
        mask = np.broadcast_to(mask, a.shape)
        denom = max(float(mask.sum()), 1.0)
        return float((np.abs(a - b) * mask).sum() / denom)

    return {
        "od_content": mse(arrays["od"][idx], arrays["od_target"][idx], arrays["od_tmask"][idx]),
        "ld_content": mse(arrays["ld"][idx], arrays["ld_target"][idx], arrays["ld_tmask"][idx]),
        # Contract geometry has no persistence baseline; report the zero
        # decoder baseline separately, using the same L1 as the loss.
        "od_geometry_zero_baseline": mae(np.zeros_like(arrays["od_geometry"][idx]), arrays["od_geometry"][idx], arrays["od_gmask"][idx]),
        "ld_geometry_zero_baseline": mae(np.zeros_like(arrays["ld_geometry"][idx]), arrays["ld_geometry"][idx], arrays["ld_gmask"][idx]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bank-native one-step latent WM")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--ckpt", required=True, help="bank-native DriveModel checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-pairs", type=int, default=60000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    data = load_data(args.data)
    ids = _load_ids(args.data)
    if len(ids["od_ids"]) != len(data["action"]):
        raise ValueError("WM ID metadata length does not match loaded rows")
    data["od_ids"], data["ld_ids"] = ids["od_ids"], ids["ld_ids"]

    drive = load_drive_model_from_checkpoint(args.ckpt, args.device, strict=True)
    encoder = drive.encoder
    for p_ in encoder.parameters():
        p_.requires_grad_(False)
    encoder.eval()
    target = TargetEncoder(encoder).to(args.device).eval()
    if getattr(drive, "action_low", None) is None or getattr(drive, "action_high", None) is None:
        raise ValueError("encoder checkpoint has no action bounds; cannot normalize WM actions")
    action_low = drive.action_low.detach().cpu().numpy()
    action_high = drive.action_high.detach().cpu().numpy()
    arrays = _make_training_arrays(
        data, encoder, target, args.device, int(args.max_pairs),
        action_low=action_low, action_high=action_high,
    )
    train_mask, val_mask = split_by_scene_seed(arrays["scene_seed"], args.val_frac, args.seed)
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    if len(train_idx) == 0:
        raise ValueError("empty WM training split")
    if len(val_idx) == 0:
        raise ValueError("empty WM validation split; choose --val-frac > 0 with disjoint scenes")
    if set(int(x) for x in arrays["scene_seed"][train_idx]) & set(int(x) for x in arrays["scene_seed"][val_idx]):
        raise ValueError("WM train/validation scene seeds overlap")
    train_seeds = sorted({int(x) for x in arrays["scene_seed"][train_idx]})
    val_seeds = sorted({int(x) for x in arrays["scene_seed"][val_idx]})

    hidden = int(getattr(drive, "hidden", 128))
    wm = EgoConditionedLatentWM(hidden=hidden, action_dim=4).to(args.device)
    opt = torch.optim.AdamW(wm.parameters(), lr=args.lr, weight_decay=1e-4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    best_state = None
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        wm.train()
        perm = np.random.default_rng(int(args.seed) + epoch).permutation(train_idx)
        total = 0.0
        n_batches = 0
        for start in range(0, len(perm), int(args.batch_size)):
            batch = _tensors(arrays, perm[start : start + int(args.batch_size)], args.device)
            losses = _loss(wm, batch)
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
            opt.step()
            total += float(losses["total"].detach())
            n_batches += 1
        wm.eval()
        val_total = 0.0
        parts: Dict[str, float] = {}
        with torch.no_grad():
            for start in range(0, len(val_idx), int(args.batch_size)):
                batch = _tensors(arrays, val_idx[start : start + int(args.batch_size)], args.device)
                losses = _loss(wm, batch)
                val_total += float(losses["total"].detach())
                for key, value in losses.items():
                    parts[key] = parts.get(key, 0.0) + float(value.detach())
        n_val = max(1, math.ceil(len(val_idx) / int(args.batch_size)))
        record = {
            "epoch": epoch,
            "train_loss": total / max(1, n_batches),
            "val_loss": val_total / n_val,
            "val_parts": {k: v / n_val for k, v in parts.items()},
            "persistence": _persistence(arrays, val_idx),
        }
        history.append(record)
        if record["val_loss"] < best:
            best = record["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in wm.state_dict().items()}
            torch.save(
                {
                    "model": wm.state_dict(),
                    "config": {
                        "schema": "bank_native_wm_d1",
                        "hidden": hidden,
                        "action_dim": 4,
                        "target_encoder": args.ckpt,
                        "train_scene_seeds": train_seeds,
                        "val_scene_seeds": val_seeds,
                        "val_frac": float(args.val_frac),
                        "split_seed": int(args.seed),
                        "max_pairs": int(args.max_pairs),
                        "data": list(args.data),
                        "action_contract": {
                            "low": action_low.tolist() if action_low is not None else [],
                            "high": action_high.tolist() if action_high is not None else [],
                        },
                    },
                    "epoch": epoch,
                    "metrics": record,
                },
                out / "best.pt",
            )
    if best_state is not None:
        drive.world_model.load_state_dict(best_state, strict=True)
        from pipeline.checkpointing import save_drive_model_checkpoint
        save_drive_model_checkpoint(drive, str(out / "best_drive.pt"))
    (out / "metrics.json").write_text(
        json.dumps({
            "history": history,
            "train_scene_seeds": train_seeds,
            "val_scene_seeds": val_seeds,
            "val_frac": float(args.val_frac),
            "split_seed": int(args.seed),
            "max_pairs": int(args.max_pairs),
            "data": list(args.data),
        }, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
