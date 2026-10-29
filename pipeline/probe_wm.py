"""Autoregressive probe for the bank-native history-conditioned WM."""

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
from pipeline.losses import masked_l1, masked_mse
from pipeline.train_bc import load_data
from pipeline.train_wm import _align_by_id, _align_geometry_by_id, _encode_rows, _history_batch, _load_ids


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


def _encode_target_rows(encoder, data, rows, device):
    return _encode_rows(encoder, data, rows, device)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Bank-native autoregressive WM probe")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--encoder-ckpt", required=True)
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--max-sequences", type=int, default=1000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--allow-random-val", action="store_true")
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
    recorded_encoder = cfg.get("target_encoder")
    if recorded_encoder:
        if Path(str(recorded_encoder)).resolve() != Path(args.encoder_ckpt).resolve():
            raise ValueError(
                "probe encoder_ckpt does not match the WM's target_encoder: "
                f"{recorded_encoder} vs {args.encoder_ckpt}"
            )
    heldout = set(int(x) for x in cfg.get("val_scene_seeds", []))
    split_val_frac = float(cfg.get("val_frac", args.val_frac))
    split_seed = int(cfg.get("split_seed", args.seed))
    if heldout:
        seqs = [s for s in seqs if int(data["scene_seed"][s[0]]) in heldout]
        heldout_seeds = sorted(heldout)
    elif args.allow_random_val:
        seq_seeds = np.asarray([int(data["scene_seed"][s[0]]) for s in seqs], dtype=np.int64)
        _train_m, val_m = split_by_scene_seed(seq_seeds, split_val_frac, split_seed)
        heldout_seeds = sorted({int(x) for x in seq_seeds[val_m]})
        seqs = [s for s, keep in zip(seqs, val_m) if keep]
    else:
        raise SystemExit(
            "WM checkpoint has no val_scene_seeds; pass --allow-random-val only for legacy debug"
        )
    if len(seqs) > int(args.max_sequences):
        rng = np.random.default_rng(int(args.seed))
        keep = np.sort(rng.choice(len(seqs), size=int(args.max_sequences), replace=False))
        seqs = [seqs[int(i)] for i in keep]
    if not seqs:
        raise SystemExit("no held-out WM sequences")

    drive = load_drive_model_from_checkpoint(args.encoder_ckpt, args.device, strict=True)
    encoder = drive.encoder.eval()
    target = TargetEncoder(encoder).to(args.device).eval()
    rows = np.unique(np.concatenate(seqs))
    row_map = {int(r): i for i, r in enumerate(rows)}
    si = np.asarray([[row_map[int(r)] for r in seq] for seq in seqs], dtype=np.int64)
    tgt_enc = _encode_target_rows(target, data, rows, args.device)

    wm = EgoConditionedLatentWM(
        hidden=int(cfg.get("hidden", getattr(drive, "hidden", 128))),
        action_dim=4,
    )
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

    def _mode_zero(row: int) -> bool:
        mode = data.get("history_mode")
        if mode is None:
            return False
        return bool(float(np.asarray(mode)[row]) < 0.5)

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
            parts = {"od_content": [], "ld_content": [], "od_geometry": [], "ld_geometry": []}
            persistence = {"od_content": [], "ld_content": [], "od_geometry_zero": [], "ld_geometry_zero": []}
            for n, seq in enumerate(seqs):
                # Initial real 6-frame window, encoded by the frozen TSB.
                first_batch = _history_batch(data, np.asarray([seq[0]], dtype=np.int64), args.device)
                window, last_enc = drive.build_history_window(
                    first_batch, zero_pose=_mode_zero(int(seq[0]))
                )
                base_od = window.od_content[:, -1].clone()
                base_ld = window.ld_content[:, -1].clone()
                z_od = base_od
                z_ld = base_ld
                ego_hidden = last_enc["ego"]
                ego_features = torch.from_numpy(
                    np.asarray(data["ego"][seq[0], -1], dtype=np.float32)
                ).to(args.device).unsqueeze(0)
                source_pose = torch.from_numpy(
                    np.asarray(data["pose"][seq[0], -1], dtype=np.float32)
                ).to(args.device).unsqueeze(0)
                h = drive.temporal_spatial(window, ego_hidden, ego_valid=last_enc.get("ego_valid"))
                for t in range(depth):
                    a = torch.from_numpy(action_norm[seq[t]]).to(args.device).unsqueeze(0)
                    step = wm(
                        h["od"],
                        h["ld"],
                        a,
                        od_base=z_od,
                        ld_base=z_ld,
                        od_valid=window.od_valid[:, -1],
                        ld_valid=window.ld_valid[:, -1],
                    )
                    z_od = step["od"].detach()
                    z_ld = step["ld"].detach()
                    if t + 1 < depth:
                        # Use the same action/ego recurrence as model.imagine;
                        # never teacher-force future ego/pose.
                        requested = drive._to_requested(a)
                        next_pose, next_ego_features = drive._ego_step(
                            ego_features, requested, source_pose
                        )
                        ego_features = next_ego_features
                        source_pose = next_pose
                        window = window.advance(
                            od_content=z_od,
                            od_geometry=step["od_geometry"],
                            od_type=window.od_type[:, -1],
                            od_valid=step["od_valid"],
                            od_stable_id=window.od_stable_id[:, -1],
                            ld_content=z_ld,
                            ld_geometry=step["ld_geometry"],
                            ld_type=window.ld_type[:, -1],
                            ld_valid=step["ld_valid"],
                            ld_stable_id=window.ld_stable_id[:, -1],
                            ego_pose=next_pose,
                            source=1,
                            rollout_depth=window.rollout_depth[:, -1] + 1,
                        )
                        ego_hidden = drive.encoder.encode_ego(ego_features)
                        h = drive.temporal_spatial(
                            window, ego_hidden,
                            ego_valid=torch.ones((1, 1), dtype=torch.bool, device=args.device),
                        )
                target_row = si[n, depth]
                tgt_od, tm_od = _align_by_id(
                    z_od[0].cpu().numpy(), window.od_stable_id[0, -1].cpu().numpy(),
                    window.od_valid[0, -1].cpu().numpy(),
                    tgt_enc["od_content"][target_row], data["od_ids"][seq[depth], -1],
                    tgt_enc["od_valid"][target_row],
                )
                tgt_ld, tm_ld = _align_by_id(
                    z_ld[0].cpu().numpy(), window.ld_stable_id[0, -1].cpu().numpy(),
                    window.ld_valid[0, -1].cpu().numpy(),
                    tgt_enc["ld_content"][target_row], data["ld_ids"][seq[depth], -1],
                    tgt_enc["ld_valid"][target_row],
                )
                cur_od_ids = window.od_stable_id[0, -1].cpu().numpy()
                cur_ld_ids = window.ld_stable_id[0, -1].cpu().numpy()
                cur_od_mask = window.od_valid[0, -1].cpu().numpy()
                cur_ld_mask = window.ld_valid[0, -1].cpu().numpy()
                tgt_geo_od, gm_od = _align_geometry_by_id(
                    cur_od_ids, cur_od_mask,
                    tgt_enc["od_geometry"][target_row],
                    data["od_ids"][seq[depth], -1], tgt_enc["od_valid"][target_row],
                )
                tgt_geo_ld, gm_ld = _align_geometry_by_id(
                    cur_ld_ids, cur_ld_mask,
                    tgt_enc["ld_geometry"][target_row],
                    data["ld_ids"][seq[depth], -1], tgt_enc["ld_valid"][target_row],
                )
                pred_geo_od = step["od_geometry"][0].cpu().numpy()
                pred_geo_ld = step["ld_geometry"][0].cpu().numpy()
                gm_od_t = torch.from_numpy(gm_od).to(args.device).unsqueeze(0)
                gm_ld_t = torch.from_numpy(gm_ld).to(args.device).unsqueeze(0)
                parts["od_geometry"].append(float(masked_l1(
                    step["od_geometry"],
                    torch.from_numpy(tgt_geo_od).to(args.device).unsqueeze(0),
                    gm_od_t,
                )))
                parts["ld_geometry"].append(float(masked_l1(
                    step["ld_geometry"],
                    torch.from_numpy(tgt_geo_ld).to(args.device).unsqueeze(0),
                    gm_ld_t,
                )))
                persistence["od_geometry_zero"].append(float(masked_l1(
                    torch.zeros_like(step["od_geometry"]),
                    torch.from_numpy(tgt_geo_od).to(args.device).unsqueeze(0),
                    gm_od_t,
                )))
                persistence["ld_geometry_zero"].append(float(masked_l1(
                    torch.zeros_like(step["ld_geometry"]),
                    torch.from_numpy(tgt_geo_ld).to(args.device).unsqueeze(0),
                    gm_ld_t,
                )))
                parts["od_content"].append(float(masked_mse(
                    z_od, torch.from_numpy(tgt_od).to(args.device).unsqueeze(0),
                    torch.from_numpy(tm_od).to(args.device).unsqueeze(0),
                )))
                parts["ld_content"].append(float(masked_mse(
                    z_ld, torch.from_numpy(tgt_ld).to(args.device).unsqueeze(0),
                    torch.from_numpy(tm_ld).to(args.device).unsqueeze(0),
                )))
                persistence["od_content"].append(float(masked_mse(
                    base_od, torch.from_numpy(tgt_od).to(args.device).unsqueeze(0),
                    torch.from_numpy(tm_od).to(args.device).unsqueeze(0),
                )))
                persistence["ld_content"].append(float(masked_mse(
                    base_ld, torch.from_numpy(tgt_ld).to(args.device).unsqueeze(0),
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
