"""P1 BC trainer for the structured planner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from net.model import ACTION_HIGH, ACTION_LOW, StructuredPlanner, count_parameters

KEYS = ("od", "od_mask", "ld", "ld_mask", "nav", "nav_mask", "ego", "ego_mask", "hist_valid")


def _split_episodes(episode_id: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.unique(episode_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac))) if len(ids) > 1 else 0
    val = set(ids[-n_val:].tolist()) if n_val else set()
    train_mask = np.array([int(e) not in val for e in episode_id])
    val_mask = ~train_mask
    return train_mask, val_mask


def _tensor_batch(z: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, torch.Tensor]:
    out = {}
    for k in KEYS:
        out[k] = torch.from_numpy(np.asarray(z[k][mask], dtype=np.float32))
    return out


def _norm_action(a: torch.Tensor) -> torch.Tensor:
    low = ACTION_LOW.to(a.device)
    high = ACTION_HIGH.to(a.device)
    return (a - low) / (high - low).clamp_min(1e-6)


def _denorm_action(a: torch.Tensor) -> torch.Tensor:
    low = ACTION_LOW.to(a.device)
    high = ACTION_HIGH.to(a.device)
    return low + a.clamp(0.0, 1.0) * (high - low)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--init", default=None)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    parts = []
    for d in args.data:
        zz = np.load(Path(d) / "expert_bc.npz")
        parts.append({k: zz[k] for k in zz.files})
    z = {}
    for k in parts[0]:
        if k == "episode_id":
            arrs = []
            off = 0
            for p_ in parts:
                arr = p_[k].copy()
                arr = arr + off
                off = int(arr.max()) + 1 if len(arr) else off
                arrs.append(arr)
            z[k] = np.concatenate(arrs, axis=0)
        else:
            z[k] = np.concatenate([p_[k] for p_ in parts], axis=0)
    action = z["action"].astype(np.float32)
    episode_id = z["episode_id"]
    train_mask, val_mask = _split_episodes(episode_id, args.val_frac, args.seed)
    if not val_mask.any():
        val_mask = train_mask
    train = _tensor_batch(z, train_mask)
    val = _tensor_batch(z, val_mask)
    target = torch.from_numpy(action)
    target_norm = _norm_action(target)
    train_ds = TensorDataset(*(train[k] for k in KEYS), target_norm[train_mask])
    val_ds = TensorDataset(*(val[k] for k in KEYS), target_norm[val_mask])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = StructuredPlanner().to(args.device)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[bc] initialized from {args.init}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bc] rows train={int(train_mask.sum())} val={int(val_mask.sum())} params={count_parameters(model)} device={args.device}")

    best = math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for *xs, y in train_loader:
            batch = {k: v.to(args.device) for k, v in zip(KEYS, xs)}
            y = y.to(args.device)
            out = model(batch)
            loss = loss_fn(out["action_mu"], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(y)
            n += len(y)
        train_loss = total / max(n, 1)

        model.eval()
        total = 0.0
        n = 0
        mae = torch.zeros(4, dtype=torch.float64)
        with torch.no_grad():
            for *xs, y in val_loader:
                batch = {k: v.to(args.device) for k, v in zip(KEYS, xs)}
                y = y.to(args.device)
                out = model(batch)
                total += float(loss_fn(out["action_mu"], y)) * len(y)
                n += len(y)
                mae += (out["action_mu"] - y).abs().sum(dim=0).double().cpu()
        val_loss = total / max(n, 1)
        mae = (mae / max(n, 1)).numpy()
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mae_norm": mae.tolist()}
        history.append(rec)
        print(f"[bc] epoch {epoch:03d} train={train_loss:.6f} val={val_loss:.6f} mae={np.round(mae,4).tolist()}")
        if val_loss < best:
            best = val_loss
            torch.save({
                "model": model.state_dict(),
                "config": {"hidden": model.hidden},
                "epoch": epoch,
                "val_loss": best,
                "action_low": ACTION_LOW.tolist(),
                "action_high": ACTION_HIGH.tolist(),
            }, out_dir / "best.pt")
    (out_dir / "metrics.json").write_text(json.dumps({"history": history, "params": count_parameters(model)}, indent=2), encoding="utf-8")
    print(f"[bc] wrote {out_dir/'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
