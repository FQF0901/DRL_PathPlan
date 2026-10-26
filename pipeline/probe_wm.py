"""Autoregressive probe for the bank-native latent world model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from dataset.splits import split_by_scene_seed
from net.encoder import TargetEncoder
from net.world_model import EgoConditionedLatentWM
from pipeline.checkpointing import load_drive_model_from_checkpoint
from pipeline.losses import masked_mse
from pipeline.train_bc import load_data
from pipeline.train_wm import _align_by_id, _build_pairs, _encode_rows, _load_ids, _rows_to_batch


def _sequences(data: Dict[str, np.ndarray], horizon: int) -> List[np.ndarray]:
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    step = np.asarray(data["step"], dtype=np.int64)
    out = []
    need = horizon + 1
    for eid in np.unique(episode_id):
        idx = np.flatnonzero(episode_id == eid)
        idx = idx[np.argsort(step[idx])]
        for start in range(0, len(idx) - need + 1):
            window = idx[start : start + need]
            if np.all(np.diff(step[window]) == 1):
                out.append(window)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bank-native autoregressive WM probe")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--encoder-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--max-sequences", type=int, default=1000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    data = load_data(args.data)
    ids = _load_ids(args.data)
    data["od_ids"], data["ld_ids"] = ids["od_ids"], ids["ld_ids"]
    seqs = _sequences(data, int(args.horizon))
    if not seqs:
        raise SystemExit("no sequences")
    payload = torch.load(args.wm_ckpt, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    heldout = set(int(x) for x in cfg.get("val_scene_seeds", []))
    split_val_frac = float(cfg.get("val_frac", args.val_frac))
    split_seed = int(cfg.get("split_seed", args.seed))
    if heldout:
        seqs = [s for s in seqs if int(data["scene_seed"][s[0]]) in heldout]
        heldout_seeds = sorted(heldout)
    else:
        # Split the complete sequence set before any max-sequence subsampling.
        seq_seeds = np.asarray([int(data["scene_seed"][s[0]]) for s in seqs], dtype=np.int64)
        _train_m, val_m = split_by_scene_seed(seq_seeds, split_val_frac, split_seed)
        heldout_seeds = sorted({int(x) for x in seq_seeds[val_m]})
        seqs = [s for s, keep in zip(seqs, val_m) if keep]
    if len(seqs) > int(args.max_sequences):
        rng = np.random.default_rng(int(args.seed))
        keep = np.sort(rng.choice(len(seqs), size=int(args.max_sequences), replace=False))
        seqs = [seqs[int(i)] for i in keep]
    if not seqs:
        raise SystemExit("no held-out WM sequences")

    drive = load_drive_model_from_checkpoint(args.encoder_ckpt, args.device, strict=True)
    encoder = drive.encoder.eval()
    rows = np.unique(np.concatenate(seqs))
    enc = _encode_rows(encoder, data, rows, args.device)
    target = TargetEncoder(encoder).to(args.device).eval()
    tgt_enc = _encode_rows(target, data, rows, args.device)
    row_map = {int(r): i for i, r in enumerate(rows)}
    si = np.asarray([[row_map[int(r)] for r in seq] for seq in seqs], dtype=np.int64)

    wm = EgoConditionedLatentWM(hidden=int(cfg.get("hidden", getattr(drive, "hidden", 128))), action_dim=4)
    wm.load_state_dict(payload["model"])
    wm.to(args.device).eval()
    if getattr(drive, "action_low", None) is None or getattr(drive, "action_high", None) is None:
        raise RuntimeError("encoder checkpoint has no action bounds")
    low = drive.action_low.detach().cpu().numpy().astype(np.float32)
    high = drive.action_high.detach().cpu().numpy().astype(np.float32)
    recorded = cfg.get("action_contract", {})
    if recorded:
        rec_low = np.asarray(recorded.get("low", []), dtype=np.float32)
        rec_high = np.asarray(recorded.get("high", []), dtype=np.float32)
        if rec_low.size and not (np.allclose(rec_low, low) and np.allclose(rec_high, high)):
            raise ValueError("encoder/WM action bounds mismatch")
    width = np.maximum(high - low, 1e-6)
    executed = data.get("executed_action")
    actions = data["action"] if executed is None else np.where(
        np.isfinite(np.asarray(executed, dtype=np.float32)), executed, data["action"]
    )
    action_norm = (np.asarray(actions, dtype=np.float32) - low) / width

    result = {
        "horizon": int(args.horizon),
        "sequences": len(seqs),
        "data": list(args.data),
        "encoder_ckpt": args.encoder_ckpt,
        "wm_ckpt": args.wm_ckpt,
        "held_out_split": {
            "val_frac": split_val_frac,
            "seed": split_seed,
            "scene_seeds": heldout_seeds,
        },
        "depths": [],
    }
    with torch.no_grad():
        for depth in range(1, int(args.horizon) + 1):
            parts = {"od_content": [], "ld_content": []}
            persistence = {"od_content": [], "ld_content": []}
            for n, seq in enumerate(seqs):
                z_od = torch.from_numpy(enc["od_content"][si[n, 0]]).to(args.device).unsqueeze(0)
                z_ld = torch.from_numpy(enc["ld_content"][si[n, 0]]).to(args.device).unsqueeze(0)
                odv = torch.from_numpy(enc["od_valid"][si[n, 0]]).to(args.device).unsqueeze(0)
                ldv = torch.from_numpy(enc["ld_valid"][si[n, 0]]).to(args.device).unsqueeze(0)
                for t in range(depth):
                    a = torch.from_numpy(action_norm[seq[t]]).to(args.device).unsqueeze(0)
                    step = wm(z_od, z_ld, a, od_valid=odv, ld_valid=ldv)
                    z_od = step["od"].detach()
                    z_ld = step["ld"].detach()
                target = si[n, depth]
                tgt_od, tm_od = _align_by_id(
                    z_od[0].cpu().numpy(), data["od_ids"][seq[0], -1], odv[0].cpu().numpy(),
                    tgt_enc["od_content"][target], data["od_ids"][seq[depth], -1], tgt_enc["od_valid"][target],
                )
                tgt_ld, tm_ld = _align_by_id(
                    z_ld[0].cpu().numpy(), data["ld_ids"][seq[0], -1], ldv[0].cpu().numpy(),
                    tgt_enc["ld_content"][target], data["ld_ids"][seq[depth], -1], tgt_enc["ld_valid"][target],
                )
                parts["od_content"].append(float(masked_mse(
                    z_od, torch.from_numpy(tgt_od).to(args.device).unsqueeze(0),
                    torch.from_numpy(tm_od).to(args.device).unsqueeze(0),
                )))
                parts["ld_content"].append(float(masked_mse(
                    z_ld, torch.from_numpy(tgt_ld).to(args.device).unsqueeze(0),
                    torch.from_numpy(tm_ld).to(args.device).unsqueeze(0),
                )))
                persistence["od_content"].append(float(masked_mse(
                    torch.from_numpy(enc["od_content"][si[n, 0]]).to(args.device).unsqueeze(0),
                    torch.from_numpy(tgt_od).to(args.device).unsqueeze(0),
                    torch.from_numpy(tm_od).to(args.device).unsqueeze(0),
                )))
                persistence["ld_content"].append(float(masked_mse(
                    torch.from_numpy(enc["ld_content"][si[n, 0]]).to(args.device).unsqueeze(0),
                    torch.from_numpy(tgt_ld).to(args.device).unsqueeze(0),
                    torch.from_numpy(tm_ld).to(args.device).unsqueeze(0),
                )))
            result["depths"].append({
                "depth": depth,
                "wm": {k: float(np.mean(v)) for k, v in parts.items()},
                "persistence": {k: float(np.mean(v)) for k, v in persistence.items()},
            })
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
