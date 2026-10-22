"""Autoregressive multi-step probe for the Stage D1 latent WM.

This is an evaluation probe, not an imagined-RL path: actions are the recorded
real actions, and the target encoder remains frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from net.world_model import EgoConditionedLatentWM, masked_mse, rollout_latents
from pipeline.checkpointing import load_model_from_checkpoint
from pipeline.train_bc import load_data
from pipeline.train_wm import (
    _align_by_id,
    _encode_rows,
    _load_ids,
    _rows_to_batch,
)


def _build_sequences(data: Dict[str, np.ndarray], horizon: int, max_sequences: int):
    episode_id = np.asarray(data["episode_id"], dtype=np.int64)
    step = np.asarray(data["step"], dtype=np.int64)
    seqs: List[np.ndarray] = []
    need = horizon + 1
    for eid in np.unique(episode_id):
        idx = np.flatnonzero(episode_id == eid)
        idx = idx[np.argsort(step[idx])]
        for start in range(0, len(idx) - need + 1):
            window = idx[start : start + need]
            if np.all(np.diff(step[window]) == 1):
                seqs.append(window)
                if len(seqs) >= max_sequences:
                    return np.asarray(seqs, dtype=np.int64)
    return np.asarray(seqs, dtype=np.int64)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Autoregressive WM probe")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--encoder-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--max-sequences", type=int, default=1000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    data = load_data(args.data)
    ids = _load_ids(args.data)
    data["od_ids"], data["ld_ids"] = ids["od_ids"], ids["ld_ids"]
    seqs = _build_sequences(data, int(args.horizon), int(args.max_sequences))
    if len(seqs) == 0:
        raise SystemExit("no sequences")
    rows = np.unique(seqs)
    encoder = load_model_from_checkpoint(args.encoder_ckpt, args.device, strict=True)
    encoder.eval()
    enc = _encode_rows(encoder, data, rows, args.device)
    row_map = {int(r): i for i, r in enumerate(rows)}
    si = np.asarray([[row_map[int(r)] for r in seq] for seq in seqs], dtype=np.int64)

    wm = EgoConditionedLatentWM(hidden=128, od_slots=16, ld_slots=16).to(args.device)
    payload = torch.load(args.wm_ckpt, map_location="cpu", weights_only=False)
    wm.load_state_dict(payload["model"])
    wm.eval()

    # Initial state, actions, and aligned targets for each horizon.
    init = {
        "od": torch.from_numpy(enc["od"][si[:, 0]]).to(args.device),
        "ld": torch.from_numpy(enc["ld"][si[:, 0]]).to(args.device),
        "nav": torch.from_numpy(enc["nav"][si[:, 0]]).to(args.device),
        "ego": torch.from_numpy(enc["ego"][si[:, 0]]).to(args.device),
    }
    masks = {
        "od": torch.from_numpy(enc["od_mask"][si[:, 0]]).to(args.device),
        "ld": torch.from_numpy(enc["ld_mask"][si[:, 0]]).to(args.device),
    }
    actions = np.asarray(
        [[data["action"][seq[t]] for t in range(args.horizon)] for seq in seqs],
        dtype=np.float32,
    )
    action_t = torch.from_numpy(actions).to(args.device)

    targets: List[Dict[str, torch.Tensor]] = []
    for t in range(1, args.horizon + 1):
        od = np.zeros_like(enc["od"][si[:, 0]])
        ld = np.zeros_like(enc["ld"][si[:, 0]])
        odm = np.zeros_like(enc["od_mask"][si[:, 0]])
        ldm = np.zeros_like(enc["ld_mask"][si[:, 0]])
        for n in range(len(seqs)):
            ci = si[n, 0]
            ti = si[n, t]
            od[n], odm[n] = _align_by_id(
                enc["od"][ci], data["od_ids"][seqs[n, 0], -1], enc["od_mask"][ci],
                enc["od"][ti], data["od_ids"][seqs[n, t], -1], enc["od_mask"][ti],
            )
            ld[n], ldm[n] = _align_by_id(
                enc["ld"][ci], data["ld_ids"][seqs[n, 0], -1], enc["ld_mask"][ci],
                enc["ld"][ti], data["ld_ids"][seqs[n, t], -1], enc["ld_mask"][ti],
            )
        targets.append(
            {
                "od": torch.from_numpy(od).to(args.device),
                "ld": torch.from_numpy(ld).to(args.device),
                "od_mask": torch.from_numpy(odm).to(args.device),
                "ld_mask": torch.from_numpy(ldm).to(args.device),
                "nav": torch.from_numpy(enc["nav"][si[:, t]]).to(args.device),
                "ego": torch.from_numpy(enc["ego"][si[:, t]]).to(args.device),
            }
        )

    with torch.no_grad():
        final, history = rollout_latents(wm, init, action_t, masks)
    result = {"horizon": int(args.horizon), "sequences": int(len(seqs)), "depths": []}
    prev = init
    for t, pred in enumerate(history, start=1):
        target = targets[t - 1]
        parts = {
            "od": float(masked_mse(pred["od"], target["od"], target["od_mask"])),
            "ld": float(masked_mse(pred["ld"], target["ld"], target["ld_mask"])),
            "nav": float(masked_mse(pred["nav"], target["nav"], torch.ones_like(pred["nav"][..., 0]))),
            "ego": float(masked_mse(pred["ego"], target["ego"], torch.ones_like(pred["ego"][..., 0]))),
        }
        # persistence: initial state repeated at every depth
        persistence = {
            "od": float(masked_mse(init["od"], target["od"], target["od_mask"])),
            "ld": float(masked_mse(init["ld"], target["ld"], target["ld_mask"])),
            "nav": float(masked_mse(init["nav"], target["nav"], torch.ones_like(pred["nav"][..., 0]))),
            "ego": float(masked_mse(init["ego"], target["ego"], torch.ones_like(pred["ego"][..., 0]))),
        }
        result["depths"].append({"depth": t, "wm": parts, "persistence": persistence})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
