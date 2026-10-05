#!/usr/bin/env python
"""拟合 K-anchor 计划头的形状锚字典（v7 结构迭代 B，离线/CPU，只读数据）。

口径（与 fix-3 探针 `/tmp/opencode/kanchor_shape.py` 同源，见
`docs/v7_program_prereg.md` §11）：

1. 数据：expert 5k（默认 `datasets/BTC20261002-0941_expert5k_v41/expert_bc.npz`）；
   行过滤 = ``train_weight>0`` 且 ``action (N,6,2)`` 全有限；moving = 6 步积分路径
   长度 ≥ 1 m（与 fix-3 一致）。
2. 聚类：``cumdtheta = cumsum(action[...,1])``（ego 系），120k 子样（seed=0），
   ``KMeans(K, n_init=10, random_state=0)`` —— **与 fix-3 逐位一致**（簇大小可比对）。
3. 原型：每簇 ``ds`` = 簇内 ``action[...,0]`` 均值；``dtheta_lane`` = 簇内
   ``action[...,1] − lane_follow`` 均值（lane 帧 dθ 原型）。``lane_follow`` 用
   ``ld`` 槽位 0 的 ``heading_rel``（做 ``−κ·5 m`` 曲率修正，口径见 §11）与
   ``curvature`` + 该行自身 ds 剖面；车道无效行退化为 ego 系（不加修正）。
4. 输出：``config/plan_anchors_k6.json``（``anchors[k] = {ds[6], dtheta_lane[6]}`` +
   来源元数据），供 ``net.anchor.load_anchor_dictionary`` 加载。

用法::

    tools/venv-python tools/fit_plan_anchors.py \
        --dataset datasets/BTC20261002-0941_expert5k_v41/expert_bc.npz \
        --k 6 --out config/plan_anchors_k6.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

#: ld 槽位 0 的采样 offset（m；见 env/obs/ld.py）——曲率修正用
LD_SLOT0_OFFSET_M = 5.0


def integrate(chain: np.ndarray) -> np.ndarray:
    """``(N,6,2)`` ds,dθ → 6 个端点在 t0 ego 系的位置（前向 Euler，fix-3 同式）。"""
    dth = chain[..., 1]
    th_prev = np.cumsum(dth, axis=1) - dth
    ds = chain[..., 0]
    x = np.cumsum(ds * np.cos(th_prev), axis=1)
    y = np.cumsum(ds * np.sin(th_prev), axis=1)
    return np.stack([x, y], axis=-1)


def lane_geometry(ld: np.ndarray, ld_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``ld (N,16,7)`` 槽位 0..4 → 最近有效槽位的 ``(dpsi_s0, curvature, valid)``。

    ``dpsi_s0`` = ``heading_rel − curvature·5 m``（槽位 0/1 的采样 offset 为 5/10 m；
    统一按 5 m 修正——槽位 1 时残差 ≤ 5 m 的曲率误差，fix-3 §4.2 同口径）。
    """
    n = int(ld.shape[0])
    dpsi = np.zeros(n, dtype=np.float64)
    kappa = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for slot in range(5):
        ok = (ld_mask[:, slot] > 0.5) & (~valid)
        if not ok.any():
            continue
        offset = 5.0 * (slot + 1)
        k = ld[ok, slot, 3]
        dpsi[ok] = ld[ok, slot, 2] - k * offset
        kappa[ok] = k
        valid[ok] = True
    return dpsi, kappa, valid


def lane_follow_dtheta(ds: np.ndarray, dpsi: np.ndarray, kappa: np.ndarray) -> np.ndarray:
    """lane 帧 → ego 系的"沿车道跟随"dθ 剖面 ``(N,6)``。

    ``heading_i = dpsi + κ·s_i``（s = cumsum(ds)）；``dθ_0 = dpsi + κ·ds_0``、
    ``dθ_i = κ·ds_i``（i≥1）。与 ``net.anchor.lane_follow_dtheta`` 同一口径。
    """
    s = np.cumsum(ds, axis=1)
    follow = kappa[:, None] * ds
    follow[:, 0] = dpsi + kappa * ds[:, 0]
    return follow


def main() -> None:
    parser = argparse.ArgumentParser(description="拟合 K-anchor 形状锚字典（离线，只读）")
    parser.add_argument(
        "--dataset",
        default="datasets/BTC20261002-0941_expert5k_v41/expert_bc.npz",
        help="expert npz（需含 action/train_weight/ld/ld_mask）",
    )
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--out", default="config/plan_anchors_k6.json")
    parser.add_argument("--subsample", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-length-m", type=float, default=1.0)
    args = parser.parse_args()

    payload = np.load(args.dataset)
    action = np.asarray(payload["action"], dtype=np.float64)
    weight = np.asarray(payload["train_weight"], dtype=np.float64).reshape(-1)
    ld = np.asarray(payload["ld"], dtype=np.float64)
    ld_mask = np.asarray(payload["ld_mask"], dtype=np.float64)
    finite = np.isfinite(action).all(axis=(1, 2))
    base = (weight > 0.0) & finite
    path_len = np.linalg.norm(integrate(action), axis=-1)[:, -1]
    moving = base & (path_len >= float(args.min_length_m))
    dpsi, kappa, lane_valid = lane_geometry(ld, ld_mask)
    follow = lane_follow_dtheta(action[..., 0], dpsi, kappa)
    dth_lane = action[..., 1] - follow  # lane 帧 dθ（无效车道行 = ego 系，valid 掩码排除）

    idx_moving = np.where(moving)[0]
    rng = np.random.default_rng(int(args.seed))
    sub = rng.choice(
        idx_moving, size=min(int(args.subsample), idx_moving.size), replace=False
    )
    cum = np.cumsum(action[sub, :, 1], axis=1)
    km = KMeans(n_clusters=int(args.k), n_init=10, random_state=int(args.seed)).fit(cum)
    labels = km.labels_

    anchors = []
    for k in range(int(args.k)):
        member = labels == k
        ds_mean = action[sub[member], :, 0].mean(axis=0)
        lane_ok = member & lane_valid[sub]
        if lane_ok.any():
            dth_mean = dth_lane[sub[lane_ok]].mean(axis=0)
        else:
            dth_mean = action[sub[member], :, 1].mean(axis=0)
        anchors.append(
            {
                "index": int(k),
                "n": int(member.sum()),
                "ds": [float(v) for v in ds_mean],
                "dtheta_lane": [float(v) for v in dth_mean],
                "dtheta_ego_mean": [
                    float(v) for v in action[sub[member], :, 1].mean(axis=0)
                ],
                "lane_valid_frac": float(lane_ok.sum() / max(int(member.sum()), 1)),
            }
        )
    sizes = [int((labels == k).sum()) for k in range(int(args.k))]
    result = {
        "version": 1,
        "feature": "cumdtheta_lane",
        "k": int(args.k),
        "dt": 0.5,
        "dataset": str(args.dataset),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fit": {
            "rows_total": int(action.shape[0]),
            "rows_moving": int(moving.sum()),
            "subsample": int(sub.size),
            "kmeans": {"n_init": 10, "random_state": int(args.seed), "feature": "ego cumdtheta"},
            "cluster_sizes": sizes,
            "fix3_reference_sizes": [92519, 7823, 3012, 8461, 4515, 3670],
            "lane_valid_frac": float(lane_valid[moving].mean()),
        },
        "anchors": anchors,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[fit-anchor] K={args.k} sizes={sizes} → {out}")
    for a in anchors:
        print(
            f"  C{a['index']}: n={a['n']:6d} ds={np.round(a['ds'], 3).tolist()} "
            f"dth_lane={np.round(a['dtheta_lane'], 4).tolist()} "
            f"lane_valid={a['lane_valid_frac']:.2f}"
        )
    print(f"[fit-anchor] sha256={hashlib.sha256(text.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
