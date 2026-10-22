"""Train the Stage D1 ego-conditioned one-step latent world model.

The planner/encoder is frozen.  Targets are the target encoder's next-frame
per-entity latents, aligned to the current slot order by stable ID.  The
script reports persistence and one-step metrics; multi-step D2 evaluation is a
separate gate.
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
from env.obs import HISTORY_FRAMES, LD_SLOTS, OD_SLOTS
from net.world_model import EgoConditionedLatentWM, masked_mse
from pipeline.checkpointing import load_model_from_checkpoint
from pipeline.train_bc import _resolve_npz_paths, load_data


def _load_ids(paths: Sequence[str]) -> Dict[str, np.ndarray]:
    """Load stable IDs that the planner deliberately excludes from model input."""
    od, ld = [], []
    for raw in paths:
        for npz_path in _resolve_npz_paths(Path(raw)):
            with np.load(npz_path) as z:
                od.append(np.asarray(z["od_ids"], dtype=np.int64))
                ld.append(np.asarray(z["ld_ids"], dtype=np.int64))
    if not od:
        raise ValueError("no datasets for WM metadata")
    return {
        "od_ids": np.concatenate(od, axis=0),
        "ld_ids": np.concatenate(ld, axis=0),
    }


def _rows_to_batch(data: Dict[str, np.ndarray], idx: np.ndarray, device: str) -> Dict[str, Tensor]:
    keys = ("od", "od_mask", "ld", "ld_mask", "nav", "nav_mask", "ego", "ego_mask", "hist_valid")
    out: Dict[str, Tensor] = {}
    for key in keys:
        arr = np.asarray(data[key][idx], dtype=np.float32)
        out[key] = torch.from_numpy(arr).to(device)
    return out


@torch.no_grad()
def _encode_rows(
    encoder,
    data: Dict[str, np.ndarray],
    rows: np.ndarray,
    device: str,
    *,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    chunks: List[Dict[str, np.ndarray]] = []
    for start in range(0, len(rows), batch_size):
        idx = rows[start : start + batch_size]
        batch = _rows_to_batch(data, idx, device)
        out = encoder.encode_entities(batch)
        chunks.append(
            {
                "od": out["od"].detach().cpu().numpy(),
                "ld": out["ld"].detach().cpu().numpy(),
                "nav": out["nav"].detach().cpu().numpy(),
                "ego": out["ego"].detach().cpu().numpy(),
                "od_mask": out["od_mask"].detach().cpu().numpy(),
                "ld_mask": out["ld_mask"].detach().cpu().numpy(),
            }
        )
    if not chunks:
        raise ValueError("no rows to encode")
    keys = chunks[0].keys()
    return {key: np.concatenate([c[key] for c in chunks], axis=0) for key in keys}


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
                if len(pairs) >= int(max_pairs):
                    return (
                        np.asarray([p[0] for p in pairs], dtype=np.int64),
                        np.asarray([p[1] for p in pairs], dtype=np.int64),
                    )
    return (
        np.asarray([p[0] for p in pairs], dtype=np.int64),
        np.asarray([p[1] for p in pairs], dtype=np.int64),
    )


def _align_by_id(
    cur_lat: np.ndarray,
    cur_ids: np.ndarray,
    cur_mask: np.ndarray,
    target_lat: np.ndarray,
    target_ids: np.ndarray,
    target_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align next-frame tokens to current slots by stable ID."""
    out = np.zeros_like(cur_lat, dtype=np.float32)
    mask = np.zeros_like(cur_mask, dtype=np.float32)
    for j in range(cur_lat.shape[0]):
        ident = int(cur_ids[j])
        if ident == 0 or float(cur_mask[j]) <= 0.5:
            continue
        hit = np.flatnonzero(target_ids == ident)
        if hit.size == 0:
            continue
        k = int(hit[0])
        if float(target_mask[k]) <= 0.5:
            continue
        out[j] = target_lat[k]
        mask[j] = 1.0
    return out, mask


def _make_training_arrays(
    data: Dict[str, np.ndarray],
    encoder,
    device: str,
    max_pairs: int,
) -> Dict[str, np.ndarray]:
    cur_rows, next_rows = _build_pairs(data, max_pairs)
    if len(cur_rows) == 0:
        raise ValueError("no consecutive-frame pairs found")
    unique_rows = np.unique(np.concatenate([cur_rows, next_rows]))
    encoded = _encode_rows(encoder, data, unique_rows, device)
    row_map = {int(row): i for i, row in enumerate(unique_rows)}
    ci = np.asarray([row_map[int(x)] for x in cur_rows], dtype=np.int64)
    ni = np.asarray([row_map[int(x)] for x in next_rows], dtype=np.int64)
    cur_ids_od = np.asarray(data["od_ids"][cur_rows, -1], dtype=np.int64)
    tgt_ids_od = np.asarray(data["od_ids"][next_rows, -1], dtype=np.int64)
    cur_ids_ld = np.asarray(data["ld_ids"][cur_rows, -1], dtype=np.int64)
    tgt_ids_ld = np.asarray(data["ld_ids"][next_rows, -1], dtype=np.int64)
    od_target = np.zeros_like(encoded["od"][ci], dtype=np.float32)
    od_tmask = np.zeros_like(encoded["od_mask"][ci], dtype=np.float32)
    ld_target = np.zeros_like(encoded["ld"][ci], dtype=np.float32)
    ld_tmask = np.zeros_like(encoded["ld_mask"][ci], dtype=np.float32)
    for p in range(len(ci)):
        od_target[p], od_tmask[p] = _align_by_id(
            encoded["od"][ci[p]], cur_ids_od[p], encoded["od_mask"][ci[p]],
            encoded["od"][ni[p]], tgt_ids_od[p], encoded["od_mask"][ni[p]],
        )
        ld_target[p], ld_tmask[p] = _align_by_id(
            encoded["ld"][ci[p]], cur_ids_ld[p], encoded["ld_mask"][ci[p]],
            encoded["ld"][ni[p]], tgt_ids_ld[p], encoded["ld_mask"][ni[p]],
        )
    return {
        "od": encoded["od"][ci],
        "ld": encoded["ld"][ci],
        "nav": encoded["nav"][ci],
        "ego": encoded["ego"][ci],
        "od_mask": encoded["od_mask"][ci],
        "ld_mask": encoded["ld_mask"][ci],
        "od_target": od_target,
        "ld_target": ld_target,
        "od_tmask": od_tmask,
        "ld_tmask": ld_tmask,
        "nav_target": encoded["nav"][ni],
        "ego_target": encoded["ego"][ni],
        "action": np.asarray(data["action"][cur_rows], dtype=np.float32),
        "scene_seed": np.asarray(data["scene_seed"][cur_rows], dtype=np.int64),
    }


def _tensors(arrays: Dict[str, np.ndarray], idx: np.ndarray, device: str) -> Dict[str, Tensor]:
    keys = (
        "od", "ld", "nav", "ego", "od_mask", "ld_mask", "od_target", "ld_target",
        "od_tmask", "ld_tmask", "nav_target", "ego_target", "action",
    )
    return dict(
        **{key: torch.from_numpy(np.asarray(arrays[key][idx], dtype=np.float32)).to(device) for key in keys}
    )


def _loss(wm, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, float]]:
    pred = wm(
        od=batch["od"], ld=batch["ld"], nav=batch["nav"], ego=batch["ego"],
        od_mask=batch["od_mask"], ld_mask=batch["ld_mask"], action=batch["action"],
    )
    losses = {
        "od": masked_mse(pred["od"], batch["od_target"], batch["od_tmask"]),
        "ld": masked_mse(pred["ld"], batch["ld_target"], batch["ld_tmask"]),
        "nav": masked_mse(pred["nav"], batch["nav_target"], torch.ones_like(batch["nav"][..., 0])),
        "ego": masked_mse(pred["ego"], batch["ego_target"], torch.ones_like(batch["ego"][..., 0])),
    }
    total = sum(losses.values())
    return total, {key: float(value.detach()) for key, value in losses.items()}


def _persistence(arrays: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, float]:
    def mse(a, b, m):
        mask = np.asarray(m, dtype=np.float32)
        denom = max(float(mask.sum()), 1.0)
        return float((((a - b) ** 2) * mask[..., None]).sum() / denom)
    return {
        "od": mse(arrays["od"][idx], arrays["od_target"][idx], arrays["od_tmask"][idx]),
        "ld": mse(arrays["ld"][idx], arrays["ld_target"][idx], arrays["ld_tmask"][idx]),
        "nav": mse(arrays["nav"][idx], arrays["nav_target"][idx], np.ones((len(idx), 1))),
        "ego": mse(arrays["ego"][idx], arrays["ego_target"][idx], np.ones((len(idx), 1))),
    }


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage D1 one-step latent world model")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--ckpt", required=True, help="frozen planner checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-pairs", type=int, default=60000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    data = load_data(args.data)
    ids = _load_ids(args.data)
    if len(ids["od_ids"]) != len(data["action"]):
        raise ValueError("WM ID metadata length does not match loaded rows")
    data["od_ids"] = ids["od_ids"]
    data["ld_ids"] = ids["ld_ids"]
    encoder = load_model_from_checkpoint(args.ckpt, args.device, strict=True)
    for p_ in encoder.parameters():
        p_.requires_grad_(False)
    encoder.eval()
    arrays = _make_training_arrays(data, encoder, args.device, int(args.max_pairs))
    train_mask, val_mask = split_by_scene_seed(arrays["scene_seed"], args.val_frac, args.seed)
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask) if val_mask.any() else train_idx
    if len(val_idx) == 0:
        raise ValueError("empty WM validation split")

    model = EgoConditionedLatentWM(hidden=128, od_slots=OD_SLOTS, ld_slots=LD_SLOTS).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = np.random.default_rng(args.seed + epoch).permutation(train_idx)
        train_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), args.batch_size):
            idx = perm[start : start + args.batch_size]
            batch = _tensors(arrays, idx, args.device)
            loss, parts = _loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.detach())
            n_batches += 1
        model.eval()
        val_loss = 0.0
        val_parts: Dict[str, float] = {}
        with torch.no_grad():
            for start in range(0, len(val_idx), args.batch_size):
                idx = val_idx[start : start + args.batch_size]
                batch = _tensors(arrays, idx, args.device)
                loss, parts = _loss(model, batch)
                val_loss += float(loss.detach())
                for key, value in parts.items():
                    val_parts[key] = val_parts.get(key, 0.0) + value
        val_loss /= max(1, math.ceil(len(val_idx) / args.batch_size))
        persistence = _persistence(arrays, val_idx)
        record = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, n_batches),
            "val_loss": val_loss,
            "val_parts": {k: v / max(1, math.ceil(len(val_idx) / args.batch_size)) for k, v in val_parts.items()},
            "persistence": persistence,
        }
        history.append(record)
        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {
                        "schema": "pnp_latent_wm_d1",
                        "hidden": 128,
                        "od_slots": OD_SLOTS,
                        "ld_slots": LD_SLOTS,
                        "action_dim": 4,
                        "target_encoder": args.ckpt,
                        "target_drift": 0.0,
                    },
                    "epoch": epoch,
                    "metrics": record,
                },
                out / "best.pt",
            )
        print(f"[wm] epoch {epoch:03d} train={record['train_loss']:.6f} val={val_loss:.6f} persistence={persistence}", flush=True)

    summary = {
        "schema": "pnp_latent_wm_d1",
        "data": list(args.data),
        "frozen_encoder": args.ckpt,
        "pairs": int(len(arrays["scene_seed"])),
        "train_pairs": int(len(train_idx)),
        "val_pairs": int(len(val_idx)),
        "history": history,
        "best_val_loss": best,
        "best": {"epoch": int(np.argmin([h["val_loss"] for h in history]) + 1)},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
