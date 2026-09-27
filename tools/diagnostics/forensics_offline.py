#!/usr/bin/env python3
"""离线取证（只读分析，不修改任何行为代码）。

回答"curve/roundabout/uturn/tollgate 四类 0% 成功率"的数据侧/预瞄头侧问题：

1. ``--section counts``：逐几何（+难度）训练样本量：行数 / train_weight 和 / 非零权重行数；
2. ``--section plan``：逐几何的**预瞄头误差**（stage_b final.pt 前向）：
   - ``traj_xy`` vs 数据集 ``traj6``（专家关键轨迹，自车系 m）→ 逐 horizon 横向 MAE；
   - ``plan[:,k]`` vs 数据集 ``action[:,k]``（专家 6 步动作）→ ds / dtheta MAE。

用法::

    tools/venv-python tools/diagnostics/forensics_offline.py --section counts
    tools/venv-python tools/diagnostics/forensics_offline.py --section plan --rows 20000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

FOCUS = ("curve", "roundabout", "uturn", "tollgate")
EASY = ("straight",)


def _load_arrays(npz: str) -> dict:
    with np.load(npz) as payload:
        return {key: payload[key] for key in payload.files}


def section_counts(arrays: dict) -> dict:
    geom = np.asarray(arrays["geometry"])
    diff = np.asarray(arrays["difficulty"])
    weight = np.asarray(arrays["train_weight"], dtype=np.float64)
    balance = np.asarray(arrays["balance_weight"], dtype=np.float64) if "balance_weight" in arrays else np.ones_like(weight)
    report = {"n_rows": int(len(geom)), "per_geometry": {}, "per_difficulty_geometry": {}}
    uniq = sorted({str(x) for x in geom.tolist()})
    for name in uniq:
        sel = geom == name
        report["per_geometry"][name] = {
            "rows": int(sel.sum()),
            "trainable_rows": int(((weight > 0) & sel).sum()),
            "weight_sum": float(weight[sel].sum()),
            "balance_weight_sum": float(balance[sel].sum()),
            "share_of_rows": float(sel.mean()),
        }
    for name in FOCUS + EASY:
        for level in ("easy", "medium", "hard"):
            sel = (geom == name) & (diff == level)
            if sel.any():
                report["per_difficulty_geometry"][f"{level}/{name}"] = {
                    "rows": int(sel.sum()),
                    "weight_sum": float(weight[sel].sum()),
                }
    return report


def _build_model(ckpt: str, device: str):
    import torch
    from pipeline.stages import build_model

    model = build_model({"hidden_dim": 128, "moe": {}, "world_model": {}})
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict", "net"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    filtered = {k: v for k, v in state.items() if k in model.state_dict() and tuple(model.state_dict()[k].shape) == tuple(v.shape)}
    missing = [k for k in model.state_dict() if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    model.to(device).eval()
    print(f"[forensics] ckpt={ckpt} loaded={len(filtered)} missing={len(missing)} device={device}", flush=True)
    return model


def section_plan(arrays: dict, *, ckpt: str, device: str, rows: int, batch: int, npz: str) -> dict:
    import torch

    from pipeline.trainer import BCDataset

    dataset = BCDataset(arrays, {})
    geom = np.asarray(arrays["geometry"])
    focus_sel = np.isin(geom, list(FOCUS + EASY))
    idx_focus = np.where(focus_sel)[0]
    rng = np.random.default_rng(0)
    if rows > 0 and idx_focus.size > rows:
        idx_focus = rng.choice(idx_focus, size=rows, replace=False)
    idx_focus = np.sort(idx_focus)

    model = _build_model(ckpt, device)
    weight = np.asarray(arrays["train_weight"], dtype=np.float64)
    action = np.asarray(arrays["action"], dtype=np.float64)
    traj6 = np.asarray(arrays["traj6"], dtype=np.float64)

    acc: dict = {}
    with torch.no_grad():
        for start in range(0, idx_focus.size, batch):
            idx = idx_focus[start:start + batch]
            obs = dataset.build_obs_batch(idx)
            tensor = {k: torch.as_tensor(np.asarray(v, dtype=np.float32), device=device) for k, v in obs.items()}
            out = model(tensor, rollout=True, world_model=False)
            traj = out["traj_xy"].detach().float().cpu().numpy().astype(np.float64)   # (B,6,2) xy
            plan = out["plan"].detach().float().cpu().numpy().astype(np.float64)     # (B,6,2) ds,dtheta
            mu = out["action_mu"].detach().float().cpu().numpy().astype(np.float64)  # (B,2)
            w = weight[idx]
            g = geom[idx]
            for name in FOCUS + EASY:
                sel = g == name
                if not sel.any():
                    continue
                slot = acc.setdefault(name, {"n": 0, "w": 0.0,
                                             "xy_n": np.zeros(6), "xy_w": np.zeros(6),
                                             "ds_w": np.zeros(6), "dth_w": np.zeros(6),
                                             "ds_wsum": np.zeros(6), "dth_wsum": np.zeros(6),
                                             "exp_ds": np.zeros(6), "exp_absdth": np.zeros(6),
                                             "mu": [], "plan1": [], "exp1": []})
                slot["n"] += int(sel.sum())
                slot["w"] += float(w[sel].sum())
                err = np.linalg.norm(traj[sel] - traj6[idx[sel]], axis=-1)  # (b,6) m
                slot["xy_n"] += (err * w[sel, None]).sum(axis=0)
                slot["xy_w"] += w[sel].sum()
                ds_err = np.abs(plan[sel][:, :, 0] - action[idx[sel]][:, :, 0])
                dth_err = np.abs(plan[sel][:, :, 1] - action[idx[sel]][:, :, 1])
                slot["ds_w"] += (ds_err * w[sel, None]).sum(axis=0)
                slot["dth_w"] += (dth_err * w[sel, None]).sum(axis=0)
                slot["ds_wsum"] += w[sel].sum()
                slot["dth_wsum"] += w[sel].sum()
                slot["exp_ds"] += (action[idx[sel]][:, :, 0] * w[sel, None]).sum(axis=0)
                slot["exp_absdth"] += (np.abs(action[idx[sel]][:, :, 1]) * w[sel, None]).sum(axis=0)
                # 转向响应诊断：只保留权重>0 的行，逐步累积（最后统一算相关/分位）
                keep = sel & (w > 0)
                if keep.any():
                    slot["mu"].append(mu[keep])
                    slot["plan1"].append(plan[keep][:, 0, :])
                    slot["exp1"].append(action[idx[keep]][:, 0, :])

    out: dict = {"rows_used": int(idx_focus.size), "per_geometry": {}}
    for name, slot in acc.items():
        wn = np.maximum(slot["xy_w"], 1e-9)
        wd = np.maximum(slot["ds_wsum"], 1e-9)
        entry = {
            "n": slot["n"],
            "weight_sum": round(slot["w"], 1),
            "traj_xy_mae_m": [round(float(v), 4) for v in (slot["xy_n"] / wn).tolist()],
            "plan_ds_mae_action": [round(float(v), 4) for v in (slot["ds_w"] / wd).tolist()],
            "plan_dtheta_mae_action": [round(float(v), 4) for v in (slot["dth_w"] / wd).tolist()],
            "expert_ds_mean": [round(float(v), 3) for v in (slot["exp_ds"] / wd).tolist()],
            "expert_abs_dtheta_mean": [round(float(v), 3) for v in (slot["exp_absdth"] / wd).tolist()],
        }
        if slot["plan1"]:
            mu = np.concatenate(slot["mu"], axis=0)
            plan1 = np.concatenate(slot["plan1"], axis=0)
            exp1 = np.concatenate(slot["exp1"], axis=0)
            dth_p, dth_e = plan1[:, 1], exp1[:, 1]
            ds_p, ds_e = plan1[:, 0], exp1[:, 0]
            # 相关性 / 过原点斜率（转弯响应）
            def _corr(a, b):
                if a.std() < 1e-9 or b.std() < 1e-9:
                    return float("nan")
                return float(np.corrcoef(a, b)[0, 1])

            def _slope(a, b):
                den = float(np.dot(b, b))
                return float(np.dot(a, b) / den) if den > 1e-12 else float("nan")

            q = float(np.percentile(np.abs(dth_e), 90))
            strong = np.abs(dth_e) >= max(q, 0.05)
            mid = (np.abs(dth_e) > 0.05) & ~strong
            buckets = {}
            for thr in (0.05, 0.1, 0.2, 0.3, 0.5):
                sel_b = np.abs(dth_e) >= thr
                if int(sel_b.sum()) < 5:
                    continue
                buckets[f"|dth|>={thr}"] = {
                    "n": int(sel_b.sum()),
                    "share": round(float(sel_b.mean()), 5),
                    "expert_dtheta_mean": round(float(dth_e[sel_b].mean()), 4),
                    "plan_dtheta_mean": round(float(dth_p[sel_b].mean()), 4),
                    "plan_over_expert": round(
                        float(np.dot(dth_p[sel_b], dth_e[sel_b])
                              / max(float(np.dot(dth_e[sel_b], dth_e[sel_b])), 1e-12)), 4
                    ),
                    "expert_ds_mean": round(float(ds_e[sel_b].mean()), 3),
                    "plan_ds_mean": round(float(ds_p[sel_b].mean()), 3),
                }
            entry["turn_response"] = {
                "n": int(len(dth_e)),
                "corr_plan_dtheta_h1": round(_corr(dth_p, dth_e), 4),
                "slope_plan_on_expert_dtheta": round(_slope(dth_p, dth_e), 4),
                "p90_abs_expert_dtheta": round(q, 4),
                "strong_turn_n": int(strong.sum()),
                "strong_turn_expert_dtheta_mean": round(float(dth_e[strong].mean()), 4),
                "strong_turn_plan_dtheta_mean": round(float(dth_p[strong].mean()), 4),
                "strong_turn_plan_over_expert": round(
                    float(np.dot(dth_p[strong], dth_e[strong]) / max(float(np.dot(dth_e[strong], dth_e[strong])), 1e-12)), 4
                ),
                "strong_turn_abs_err_mean": round(float(np.abs(dth_p[strong] - dth_e[strong]).mean()), 4),
                "mid_turn_n": int(mid.sum()),
                "mid_turn_plan_over_expert": round(
                    float(np.dot(dth_p[mid], dth_e[mid]) / max(float(np.dot(dth_e[mid], dth_e[mid])), 1e-12)), 4
                ) if mid.any() else float("nan"),
                "corr_mu_dtheta": round(_corr(mu[:, 1], dth_e), 4),
                "slope_mu_on_expert_dtheta": round(_slope(mu[:, 1], dth_e), 4),
                "strong_turn_mu_over_expert": round(
                    float(np.dot(mu[strong][:, 1], dth_e[strong]) / max(float(np.dot(dth_e[strong], dth_e[strong])), 1e-12)), 4
                ),
                "corr_plan_ds": round(_corr(ds_p, ds_e), 4),
                "slope_plan_ds": round(_slope(ds_p, ds_e), 4),
                "buckets": buckets,
            }
        out["per_geometry"][name] = entry
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="forensics_offline.py")
    parser.add_argument("--npz", default="runs/bc_expert_2k_v2/expert_bc.npz")
    parser.add_argument("--ckpt", default="runs/train/il_v2_10x10_b_fixed/stage_b/final.pt")
    parser.add_argument("--section", default="counts", choices=("counts", "plan", "all"))
    parser.add_argument("--rows", type=int, default=20000, help="plan 段每类几何最多采样行数（0=全部）")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="runs/forensics/offline.json")
    args = parser.parse_args(argv)

    arrays = _load_arrays(args.npz)
    report: dict = {"npz": args.npz, "ckpt": args.ckpt}
    if args.section in ("counts", "all"):
        report["counts"] = section_counts(arrays)
        print(json.dumps(report["counts"], ensure_ascii=False, indent=1), flush=True)
    if args.section in ("plan", "all"):
        report["plan"] = section_plan(
            arrays, ckpt=args.ckpt, device=args.device, rows=args.rows, batch=args.batch, npz=args.npz
        )
        print(json.dumps(report["plan"], ensure_ascii=False, indent=1), flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[forensics] → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
