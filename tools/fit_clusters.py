#!/usr/bin/env python3
"""拟合场景原型聚类 spec（lane C / fix-4）并输出体检报告。

用法::

    tools/venv-python tools/fit_clusters.py --bc-dir runs/bc_expert_2k \
        --out config/clusters/cluster_v1.npz \
        --report config/clusters/cluster_v1.report.json

流程（``pipeline/clusters.py`` 的确定性实现）：
1. 读 ``expert_bc.npz`` + ``expert_bc.meta.json`` → 当前帧 obs → 原始特征（v2 固定槽位
   语义：``od_presence`` 优先、无效槽位清零）；
2. 去常数/近零方差维 → 标准化 → PCA 白化 48 维；
3. 白化空间 kNN 局部密度 → 稀有度权重 ``w = clip((ρ_med/ρ)^α, 0.1, 10)``（rule-free）；
4. 容量约束加权 k-means（k=8、每簇 ≤25%）+ 体检后处理（>30% 二分 / <2% 并入）；
5. 冻结 spec（质心/PCA/特征口径/数据指纹/git hash）→ ``cluster_v1.npz``；
6. 体检报告（``*.report.json`` + 控制台表）：簇规模/熵/半径、难度/几何/规则标签占比、
   稀有结构富集倍数（**规则标签只用于体检，绝不进入监督**）、FLAG 与两段式退路开关。

数据就绪后全量拟合（v2 数据集）::

    tools/venv-python tools/fit_clusters.py --bc-dir <V2_BC_DIR> \
        --out config/clusters/cluster_v1.npz --report config/clusters/cluster_v1.report.json

若体检 FLAG（无簇对稀有结构富集 ≥3×），追加两段式退路::

    tools/venv-python tools/fit_clusters.py --bc-dir <V2_BC_DIR> --two-stage \
        --out config/clusters/cluster_v1.npz --report config/clusters/cluster_v1.report.json

重聚类对齐旧编号::

    tools/venv-python tools/fit_clusters.py --bc-dir <V2_BC_DIR> \
        --align-to config/clusters/cluster_v1.npz --out config/clusters/cluster_v2.npz
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.clusters import (  # noqa: E402
    ACTIVITY_STATS,
    CLUSTER_VERSION,
    DEFAULT_ALPHA,
    DEFAULT_BOUNDARY_MARGIN,
    DEFAULT_CAP_RATIO,
    DEFAULT_DENSITY_K,
    DEFAULT_K,
    DEFAULT_MAX_SHARE,
    DEFAULT_MIN_SHARE,
    DEFAULT_PCA_DIM,
    DEFAULT_SMOOTH_STRENGTH,
    DEFAULT_SPEC,
    DEFAULT_TEMPERATURE,
    FEATURE_CONTRACT,
    ClusterSpec,
    activity_statistics,
    align_clusters,
    density_weights,
    encode_obs,
    enforce_size_bounds,
    fit_capacity_kmeans,
    fit_feature_pipeline,
    remap_assignment,
)

#: 当前帧 obs 键（含 v2）；``od_id`` 只做形状校验，不进入特征
OBS_KEYS: Tuple[str, ...] = (
    "ego",
    "ego_mask",
    "od",
    "od_mask",
    "od_presence",
    "od_id",
    "ld",
    "ld_mask",
    "nav",
    "nav_mask",
    "signal",
    "signal_mask",
    "others",
    "others_mask",
)


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:  # noqa: BLE001 - 指纹失败不阻断
        return ""


def _current_obs_fingerprint() -> str:
    try:
        from env.obs import obs_fingerprint

        return str(obs_fingerprint())
    except Exception:  # noqa: BLE001 - 指纹失败不阻断
        return ""


def load_bc_dataset(bc_dir: str | Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """读取 ``expert_bc.npz`` + ``expert_bc.meta.json``（目录或 npz 路径）。"""
    target = Path(bc_dir)
    if target.is_dir():
        target = target / "expert_bc.npz"
    if not target.is_file():
        raise FileNotFoundError(f"BC 数据集不存在：{target}")
    meta_path = target.parent / target.name.replace(".npz", ".meta.json")
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(target, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return arrays, meta


def current_frame_obs(arrays: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """从数据集 arrays 取当前帧 obs 键（batch 维在前）。"""
    obs: Dict[str, np.ndarray] = {}
    for key in OBS_KEYS:
        if key in arrays:
            obs[key] = np.asarray(arrays[key])
    if "ego" not in obs:
        raise ValueError("数据集中缺少 'ego'（不是 BC 数据集？）")
    return obs


def rare_labels_for_gate(
    label_names: Sequence[str],
    labels: np.ndarray,
    *,
    rare_max_rate: float = 0.05,
) -> Tuple[str, ...]:
    """稀有标签集合：全局占比 ≤ ``rare_max_rate``；至少含全局占比最低的标签。"""
    rates = np.asarray(labels, dtype=np.float64).mean(axis=0)
    limit = max(float(rare_max_rate), float(rates.min()) * 1.25)
    rare = tuple(name for name, rate in zip(label_names, rates) if rate <= limit)
    return rare or (str(label_names[int(np.argmin(rates))]),)


def _shared_table(rows: Sequence[Sequence[Any]], headers: Sequence[str]) -> str:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    lines = ["  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        lines.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _fit_two_stage(
    z: np.ndarray,
    raw: np.ndarray,
    w: np.ndarray,
    *,
    k: int,
    cap_ratio: float,
    max_share: float,
    min_share: float,
    max_bucket: float,
    seed: int,
    max_iters: int,
    contract: Mapping[str, Any] = FEATURE_CONTRACT,
) -> Dict[str, Any]:
    """两段式退路：活跃度 2-means 门控 + 活跃子集 K-1 簇（容量按全局占比折算）。

    1-D 加权 2-means 自然阈值可能极不均衡（平坦密度 → 阈值贴近众数，活跃子集过小），
    因此加容量钳制：每桶 ≤ ``max_bucket``（默认 65%），超出时把阈值移到加权分位点。
    """
    act = activity_statistics(raw, contract=contract)
    mean = act.mean(axis=0)
    std = act.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    weights = np.full(len(ACTIVITY_STATS), 1.0 / len(ACTIVITY_STATS))
    score = ((act - mean) / std) @ weights
    c0, c1 = np.percentile(score, [25.0, 75.0])
    if c0 > c1:
        c0, c1 = c1, c0
    for _ in range(50):
        assign_gate = np.abs(score - c1) < np.abs(score - c0)
        w0 = float(w[~assign_gate].sum())
        w1 = float(w[assign_gate].sum())
        c0 = float((score[~assign_gate] * w[~assign_gate]).sum() / w0) if w0 > 0 else c0
        c1 = float((score[assign_gate] * w[assign_gate]).sum() / w1) if w1 > 0 else c1
    if c0 > c1:
        c0, c1 = c1, c0
    threshold = 0.5 * (c0 + c1)
    active = score > threshold
    clamped = False
    # 容量钳制：2-means 自然阈值可能极不均衡；两桶都 ≤ max_bucket（按权重分位点移阈值）
    order = np.argsort(score, kind="stable")
    cumulative = np.cumsum(w[order])
    total = float(cumulative[-1])
    active_share = float(w[active].sum() / max(total, 1e-12))
    if active_share < 1.0 - max_bucket - 1e-9 or active_share > max_bucket + 1e-9:
        target_cum = max_bucket if active_share < 1.0 - max_bucket else 1.0 - max_bucket
        cut = int(np.searchsorted(cumulative, target_cum * total, side="left"))
        cut = min(max(cut, 0), score.size - 1)
        threshold = float(score[order[cut]])
        active = score > threshold
        clamped = True
    c0 = float((score[~active] * w[~active]).sum() / max(w[~active].sum(), 1e-12))
    c1 = float((score[active] * w[active]).sum() / max(w[active].sum(), 1e-12))
    if c0 > c1:
        c0, c1 = c1, c0
        active = ~active
    p_active = float(w[active].sum() / max(w.sum(), 1e-12))
    if p_active < 0.05 or p_active > 0.95:  # 退化为平凡分割 → 不做两段式
        raise ValueError(f"活跃度 2-means 分割退化（p_active={p_active:.3f}），改用 flat 模式")
    cap_ratio_active = float(np.clip(cap_ratio / max(p_active, 0.05), 0.05, 0.9))
    max_share_active = float(np.clip(max_share / max(p_active, 0.05), 0.05, 0.99))
    min_share_active = float(np.clip(min_share / max(p_active, 0.05), 0.0, 0.5))
    km = fit_capacity_kmeans(
        z[active], w[active], k=k - 1, cap_ratio=cap_ratio_active, seed=seed, max_iters=max_iters
    )
    centers, assign_active, log = enforce_size_bounds(
        z[active],
        w[active],
        km.centroids,
        km.assign,
        target_k=k - 1,
        cap_ratio=cap_ratio_active,
        max_share=max_share_active,
        min_share=min_share_active,
    )
    calm_reference = (
        (z[~active] * w[~active][:, None]).sum(axis=0) / w[~active].sum()
        if np.any(~active)
        else np.zeros(z.shape[1])
    )
    return {
        "active": active,
        "centers": centers.astype(np.float64),
        "assign_active": assign_active.astype(np.int64),
        "activity_mean": mean,
        "activity_std": std,
        "activity_weights": weights,
        "activity_centers": np.array([c0, c1], dtype=np.float64),
        "calm_reference": np.asarray(calm_reference, dtype=np.float64),
        "p_active": p_active,
        "clamped": clamped,
        "threshold": float(threshold),
        "log": log,
        "objective": float(km.objective),
        "iterations": int(km.iterations),
    }


def _capacity_report(
    assign: np.ndarray, w: np.ndarray, *, k: int, max_share: float, min_share: float
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """簇**计数占比**（lane D 均衡门口径，与 v1 的 c0=84.7% 同口径）+ 加权占比（诊断）。

    返回 ``(capacity_ok, count_shares, weighted_shares)``；`capacity_ok` = 计数占比在
    ``[min_share, max_share]`` 内（"每簇 ≈12.5%" 的直观口径；加权占比只作报告）。
    """
    counts = np.bincount(np.asarray(assign, dtype=np.int64), minlength=int(k)).astype(np.float64)
    count_shares = counts / max(float(counts.sum()), 1.0)
    weighted = np.array([float(w[assign == c].sum()) for c in range(int(k))], dtype=np.float64)
    weighted = weighted / max(float(w.sum()), 1e-12)
    ok = bool(
        count_shares.min() >= float(min_share) - 1e-9
        and count_shares.max() <= float(max_share) + 1e-9
    )
    return ok, count_shares, weighted


def _fit_balanced_split(
    z: np.ndarray,
    raw: np.ndarray,
    w: np.ndarray,
    *,
    k: int,
    cap_ratio: float,
    max_share: float,
    min_share: float,
    max_bucket: float,
    seed: int,
    max_iters: int,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """lane D 变体（b）：活跃度分桶（每桶 ≤ ``max_bucket``）+ **两个桶都做容量约束 k-means**。

    与变体（a）flat 的区别只在初始化/分桶：最终仍是 **flat spec**（soft=全 8 质心 top-2），
    因此 annotate 的 argmax 硬标签与 spec 分配一致；分桶只用于杜绝"平稳桶"变巨簇。
    """
    act = activity_statistics(raw, contract=contract)
    mean = act.mean(axis=0)
    std = act.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    weights = np.full(len(ACTIVITY_STATS), 1.0 / len(ACTIVITY_STATS))
    score = ((act - mean) / std) @ weights
    order = np.argsort(score, kind="stable")
    cumulative = np.cumsum(w[order])
    total = float(cumulative[-1])
    # 分桶：按权重分位点切在 max_bucket（默认 50%）→ 两桶都 ≤ max_bucket
    cut = int(np.searchsorted(cumulative, (1.0 - float(max_bucket)) * total, side="left"))
    cut = min(max(cut, 0), score.size - 1)
    threshold = float(score[order[cut]])
    calm = score <= threshold
    if calm.all() or (~calm).all():
        raise ValueError("活跃度分桶退化（全 calm/全 active），改用 flat 模式")
    share_calm = float(w[calm].sum() / max(total, 1e-12))
    share_active = 1.0 - share_calm
    # k 分配：按桶权重，且每桶簇数必须能容纳全局 cap/floor
    def _alloc(share: float) -> int:
        low = int(np.ceil(share / max(float(max_share), 1e-6) - 1e-9))
        high = int(np.floor(share / max(float(min_share), 1e-6) + 1e-9))
        high = max(high, low)
        return int(np.clip(int(round(k * share)), max(low, 1), max(high, 1)))
    k_calm = min(max(_alloc(share_calm), 1), k - 1)
    k_active = k - k_calm
    if k_active < 1:
        raise ValueError("分桶后 active 桶无簇可分配，改用 flat 模式")
    log: list = []
    centers_all: list = []
    assign = np.full(z.shape[0], -1, dtype=np.int64)
    for bucket, count in ((calm, k_calm), (~calm, k_active)):
        if int(bucket.sum()) < count:
            raise ValueError("桶内样本数少于分配簇数，改用 flat 模式")
        bucket_share = float(w[bucket].sum() / max(total, 1e-12))
        cap_b = float(np.clip(cap_ratio / max(bucket_share, 1e-6), 1e-3, 1.0))
        max_b = float(np.clip(max_share / max(bucket_share, 1e-6), 1e-3, 1.0))
        min_b = float(np.clip(min_share / max(bucket_share, 1e-6), 0.0, 0.5))
        km = fit_capacity_kmeans(z[bucket], w[bucket], k=int(count), cap_ratio=cap_b, seed=seed, max_iters=max_iters)
        centers_b, assign_b, log_b = enforce_size_bounds(
            z[bucket], w[bucket], km.centroids, km.assign,
            target_k=int(count), cap_ratio=cap_b, max_share=max_b, min_share=min_b,
        )
        assign[bucket] = assign_b + len(centers_all)
        centers_all.append(np.asarray(centers_b, dtype=np.float64))
        log.extend([{**item, "bucket": "calm" if bucket is calm else "active"} for item in log_b])
    # 全局收尾：分桶只做初始化，最终仍按全局 cap/floor 精修一次（杜绝桶内折算误差残留）
    centers_merged = np.concatenate(centers_all, axis=0)
    centers_merged, assign, log_global = enforce_size_bounds(
        z, w, centers_merged, assign, target_k=k,
        cap_ratio=cap_ratio, max_share=max_share, min_share=min_share,
    )
    log.extend([{**item, "bucket": "global"} for item in log_global])
    return {
        "centers": centers_merged,
        "assign": assign,
        "log": log,
        "objective": float("nan"),
        "iterations": -1,
        "activity_mean": mean,
        "activity_std": std,
        "activity_weights": weights,
        "activity_centers": np.array([0.0, 0.0], dtype=np.float64),
        "calm_reference": np.zeros(z.shape[1], dtype=np.float64),
        "p_active": float(share_active),
        "clamped": True,
        "threshold": threshold,
        "k_calm": int(k_calm),
        "k_active": int(k_active),
    }


def fit_cluster_spec_from_dataset(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
    *,
    cluster_version: str = CLUSTER_VERSION,
    k: int = DEFAULT_K,
    pca_dim: int = DEFAULT_PCA_DIM,
    density_k: int = DEFAULT_DENSITY_K,
    alpha: float = DEFAULT_ALPHA,
    cap_ratio: float = DEFAULT_CAP_RATIO,
    max_share: float = DEFAULT_MAX_SHARE,
    min_share: float = DEFAULT_MIN_SHARE,
    temperature: float = DEFAULT_TEMPERATURE,
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
    smooth_strength: float = DEFAULT_SMOOTH_STRENGTH,
    seed: int = 0,
    max_iters: int = 50,
    max_rows: int = 0,
    two_stage: bool = False,
    two_stage_max_bucket: float = 0.65,
    rare_max_rate: float = 0.05,
    enrichment_min: float = 3.0,
    dataset_path: str = "",
    align_to: Optional[str] = None,
    feature_contract: Mapping[str, Any] = FEATURE_CONTRACT,
    restarts: int = 8,
    balanced_split: bool = False,
    natural: bool = False,
) -> Dict[str, Any]:
    """核心拟合：返回 ``{"spec", "report", "arrays"（诊断中间量）}``（不落盘）。

    ``natural=True``（对照审阅用）：**不加容量/份额约束**的密度加权 k-means（cap/floor 置为
    1.0/0.0，仍走同一确定性实现）；生产候选用默认（cap 15% / floor 8% + restarts）。
    """
    if natural:
        cap_ratio, max_share, min_share = 1.0, 1.0, 0.0
    obs = current_frame_obs(arrays)
    n_total = int(obs["ego"].shape[0])
    if max_rows and int(max_rows) < n_total:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(n_total, size=int(max_rows), replace=False))
        obs = {key: value[keep] for key, value in obs.items()}
        label_source: Optional[np.ndarray] = (
            np.asarray(arrays["labels"])[keep] if "labels" in arrays else None
        )
        difficulty = np.asarray(arrays["difficulty"])[keep] if "difficulty" in arrays else None
        geometry = np.asarray(arrays["geometry"])[keep] if "geometry" in arrays else None
    else:
        label_source = np.asarray(arrays["labels"]) if "labels" in arrays else None
        difficulty = np.asarray(arrays["difficulty"]) if "difficulty" in arrays else None
        geometry = np.asarray(arrays["geometry"]) if "geometry" in arrays else None
    n = int(obs["ego"].shape[0])

    raw = encode_obs(obs, contract=feature_contract)  # 契约驱动（v2 = ego+od top6+nav cmd+road_class）
    others_source = "raw" if "others" in obs else "fallback"
    features = fit_feature_pipeline(raw, pca_dim=pca_dim)
    z = features["z"].astype(np.float64)
    w = density_weights(z, k=density_k, alpha=alpha)

    log: list = []
    mode = "two_stage" if two_stage else "flat"
    activity: Dict[str, Any] = {}
    if mode == "flat":
        if balanced_split:
            bal = _fit_balanced_split(
                z, raw, w,
                k=k, cap_ratio=cap_ratio, max_share=max_share, min_share=min_share,
                max_bucket=two_stage_max_bucket, seed=seed, max_iters=max_iters,
                contract=feature_contract,
            )
            centers, assign, log = bal["centers"], bal["assign"], bal["log"]
            activity = {
                "activity_mean": bal["activity_mean"], "activity_std": bal["activity_std"],
                "activity_weights": bal["activity_weights"], "activity_centers": bal["activity_centers"],
                "calm_reference": bal["calm_reference"], "p_active": bal["p_active"],
                "activity_clamped": True, "k_calm": bal["k_calm"], "k_active": bal["k_active"],
            }
            diag = {"objective": float("nan"), "iterations": -1, "history": []}
        else:
            # 变体（a）：flat + 严格容量，多次 restart 取「满足约束且 objective 最低」
            best: Optional[Dict[str, Any]] = None
            for attempt in range(max(1, int(restarts))):
                km = fit_capacity_kmeans(
                    z, w, k=k, cap_ratio=cap_ratio, seed=int(seed) + attempt, max_iters=max_iters
                )
                centers_a, assign_a, log_a = enforce_size_bounds(
                    z, w, km.centroids, km.assign, target_k=k,
                    cap_ratio=cap_ratio, max_share=max_share, min_share=min_share,
                )
                ok_a, _, _ = _capacity_report(assign_a, w, k=k, max_share=max_share, min_share=min_share)
                candidate = {
                    "ok": ok_a, "objective": float(km.objective),
                    "centers": centers_a, "assign": assign_a, "log": log_a,
                    "iterations": int(km.iterations), "history": km.history[-8:],
                }
                if best is None or (candidate["ok"], -candidate["objective"]) > (best["ok"], -best["objective"]):
                    best = candidate
            centers, assign, log = best["centers"], best["assign"], best["log"]
            diag = {"objective": best["objective"], "iterations": best["iterations"], "history": best["history"]}
    else:
        two = _fit_two_stage(
            z,
            raw,
            w,
            k=k,
            cap_ratio=cap_ratio,
            max_share=max_share,
            min_share=min_share,
            max_bucket=two_stage_max_bucket,
            seed=seed,
            max_iters=max_iters,
            contract=feature_contract,
        )
        centers = two["centers"]
        active = two["active"]
        assign = np.zeros(n, dtype=np.int64)
        assign[active] = 1 + two["assign_active"]
        log = two["log"]
        diag = {"objective": two["objective"], "iterations": two["iterations"], "history": []}
        activity = {
            "activity_mean": two["activity_mean"],
            "activity_std": two["activity_std"],
            "activity_weights": two["activity_weights"],
            "activity_centers": two["activity_centers"],
            "calm_reference": two["calm_reference"],
            "p_active": two["p_active"],
        }
    spec = ClusterSpec(
        cluster_version=str(cluster_version),
        mode=mode,
        k=int(k),
        feature_contract=dict(feature_contract),
        others_source=others_source,
        raw_dim=int(raw.shape[1]),
        keep_dims=features["keep_dims"],
        feature_mean=features["feature_mean"],
        feature_std=features["feature_std"],
        components=features["components"],
        whiten_scale=features["whiten_scale"],
        centroids=np.asarray(centers, dtype=np.float64),
        temperature=float(temperature),
        boundary_margin=float(boundary_margin),
        smooth_strength=float(smooth_strength),
        explained_variance=features["explained_variance"],
        activity_mean=np.asarray(activity.get("activity_mean", np.zeros(0))),
        activity_std=np.asarray(activity.get("activity_std", np.ones(0))),
        activity_weights=np.asarray(activity.get("activity_weights", np.zeros(0))),
        activity_centers=np.asarray(activity.get("activity_centers", np.zeros(0))),
        calm_reference=np.asarray(activity.get("calm_reference", np.zeros(0))),
        fit_params={
            "alpha": float(alpha),
            "density_k": int(density_k),
            "cap_ratio": float(cap_ratio),
            "max_share": float(max_share),
            "min_share": float(min_share),
            "pca_dim": int(pca_dim),
            "pca_dim_actual": int(features["components"].shape[0]),
            "seed": int(seed),
            "max_iters": int(max_iters),
            "max_rows": int(max_rows),
            "two_stage": bool(two_stage),
            "two_stage_max_bucket": float(two_stage_max_bucket),
            "balanced_split": bool(balanced_split),
            "natural": bool(natural),
            "restarts": int(restarts),
            "feature_contract": str(feature_contract.get("name") or ""),
            "p_active": activity.get("p_active"),
            "activity_clamped": activity.get("clamped"),
            "objective": diag["objective"],
            "iterations": diag["iterations"],
            "postprocess_log": log,
        },
        data_fingerprint={
            "source": str(dataset_path),
            "rows": int(n),
            "rows_total": int(n_total),
            "obs_fingerprint": str(meta.get("obs_fingerprint") or ""),
            "obs_fingerprint_current": _current_obs_fingerprint(),
            "schema_version": meta.get("schema_version"),
            "history_storage": meta.get("history_storage"),
            "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_hash": _git_hash(),
            "seed": int(seed),
        },
    )

    # 重聚类对齐（Hungarian）：把新簇编号对齐冻结 spec
    alignment: Dict[str, Any] = {}
    if align_to:
        frozen = ClusterSpec.load(align_to)
        if frozen.mode != spec.mode or int(frozen.k) != int(spec.k):
            raise ValueError(
                f"--align-to 要求 mode/k 一致：frozen {frozen.mode}/{frozen.k} vs new {spec.mode}/{spec.k}"
            )
        perm = align_clusters(frozen.centroids, spec.centroids)
        before = float(np.mean(np.min(np.linalg.norm(
            np.asarray(frozen.centroids)[:, None, :] - np.asarray(spec.centroids)[None, :, :], axis=2), axis=1)))
        spec.centroids = np.asarray(spec.centroids, dtype=np.float64)[perm]
        if spec.mode == "two_stage":
            inverse = np.empty_like(perm)
            inverse[perm] = np.arange(perm.size, dtype=perm.dtype)
            active_values = assign >= 1
            assign[active_values] = 1 + inverse[assign[active_values] - 1]
        else:
            assign = remap_assignment(assign, perm)
        after = float(np.mean(np.linalg.norm(
            np.asarray(frozen.centroids) - np.asarray(spec.centroids), axis=1)))
        alignment = {"perm": perm.tolist(), "old_cluster": list(range(int(frozen.k))),
                     "mean_cost_before": before, "mean_cost_after": after}

    capacity_ok, capacity_shares, capacity_weighted = _capacity_report(
        assign, w, k=int(k), max_share=max_share, min_share=min_share
    )
    report = build_report(
        spec=spec,
        raw=raw,
        z=z,
        w=w,
        assign=assign,
        diag=diag,
        capacity={"ok": bool(capacity_ok), "shares": capacity_shares.tolist(),
                  "weighted_shares": capacity_weighted.tolist(),
                  "max_share": float(max_share), "min_share": float(min_share)},
        labels=label_source,
        label_names=tuple(meta.get("label_names") or ()),
        difficulty=difficulty,
        geometry=geometry,
        rare_max_rate=rare_max_rate,
        enrichment_min=enrichment_min,
        alignment=alignment,
    )
    return {
        "spec": spec,
        "report": report,
        "arrays": {"raw": raw, "z": z, "weights": w, "assign": assign},
    }


def build_report(
    *,
    spec: ClusterSpec,
    raw: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
    assign: np.ndarray,
    diag: Mapping[str, Any],
    labels: Optional[np.ndarray],
    label_names: Sequence[str],
    difficulty: Optional[np.ndarray],
    geometry: Optional[np.ndarray],
    rare_max_rate: float,
    enrichment_min: float,
    alignment: Mapping[str, Any],
    capacity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """体检报告：簇规模/熵/半径 + 难度/几何/规则标签占比 + 稀有富集 + FLAG + 均衡门。"""
    n = raw.shape[0]
    k = int(spec.k)
    soft = spec.assign_soft(raw.astype(np.float32))
    dist = spec.distances(z.astype(np.float32)).astype(np.float64)
    if spec.mode == "flat":
        d_assigned = dist[np.arange(n), assign]
        gap = (np.sort(dist, axis=1)[:, 1] - np.sort(dist, axis=1)[:, 0]) / (
            np.sort(dist, axis=1)[:, 1] + np.sort(dist, axis=1)[:, 0] + 1e-12
        )
    else:  # two_stage：活跃行算到活跃质心的距离，平稳行指向 calm_reference
        d_assigned = np.zeros(n, dtype=np.float64)
        gap = np.ones(n, dtype=np.float64)
        active_rows = assign >= 1
        if np.any(active_rows):
            d_active = dist[active_rows]
            d_assigned[active_rows] = d_active[np.arange(int(active_rows.sum())), assign[active_rows] - 1]
            order = np.sort(d_active, axis=1)
            gap[active_rows] = (order[:, 1] - order[:, 0]) / (order[:, 1] + order[:, 0] + 1e-12)
        if np.any(~active_rows) and spec.calm_reference.size:
            d_assigned[~active_rows] = np.linalg.norm(
                z[~active_rows] - np.asarray(spec.calm_reference, dtype=np.float64), axis=1
            )
    counts = np.bincount(assign, minlength=k).astype(np.int64)
    shares = counts / max(n, 1)

    rare_names: Tuple[str, ...] = ()
    global_rates: Dict[str, float] = {}
    enrichment: Dict[str, Dict[str, Any]] = {}
    if labels is not None and label_names:
        rare_names = rare_labels_for_gate(label_names, labels, rare_max_rate=rare_max_rate)
        rates = np.asarray(labels, dtype=np.float64).mean(axis=0)
        global_rates = {name: float(rate) for name, rate in zip(label_names, rates)}
        for name in rare_names:
            index = list(label_names).index(name)
            per_cluster = np.array(
                [float(np.asarray(labels)[assign == j, index].mean()) if counts[j] else 0.0 for j in range(k)]
            )
            best = int(np.argmax(per_cluster))
            global_rate = max(float(rates[index]), 1e-12)
            enrichment[name] = {
                "global_rate": float(rates[index]),
                "cluster_rates": [float(v) for v in per_cluster],
                "max_cluster": best,
                "max_rate": float(per_cluster[best]),
                "max_enrichment": float(per_cluster[best] / global_rate),
            }

    clusters: list = []
    for j in range(k):
        mask = assign == j
        cluster: Dict[str, Any] = {
            "id": j,
            "size": int(counts[j]),
            "share": float(shares[j]),
            "weight_share": float(w[mask].sum() / max(w.sum(), 1e-12)),
            "radius_mean": float(d_assigned[mask].mean()) if counts[j] else 0.0,
            "radius_p95": float(np.percentile(d_assigned[mask], 95)) if counts[j] else 0.0,
            "radius_max": float(d_assigned[mask].max()) if counts[j] else 0.0,
            "soft_top1_mean": float(soft[mask].max(axis=1).mean()) if counts[j] else 0.0,
            "soft_top2_mass_mean": float(np.sort(soft[mask], axis=1)[:, -2:].sum(axis=1).mean())
            if counts[j]
            else 0.0,
            "soft_entropy_norm_mean": float(
                (-(soft[mask] * np.log(np.clip(soft[mask], 1e-12, 1.0))).sum(axis=1)
                 / np.log(max(k, 2))).mean()
            ) if counts[j] else 0.0,
        }
        if difficulty is not None:
            values, freq = np.unique(difficulty[mask].astype(str), return_counts=True)
            cluster["difficulty"] = {
                str(v): float(c / max(int(counts[j]), 1)) for v, c in zip(values, freq)
            }
        if geometry is not None:
            values, freq = np.unique(geometry[mask].astype(str), return_counts=True)
            top = sorted(zip(freq.tolist(), values.tolist()), reverse=True)[:3]
            cluster["geometry_top3"] = [[str(name), float(c / max(int(counts[j]), 1))] for c, name in top]
        if labels is not None and label_names:
            cluster["rule_label_rates"] = {
                str(name): float(np.asarray(labels)[mask, i].mean()) if counts[j] else 0.0
                for i, name in enumerate(label_names)
            }
        if enrichment:
            cluster["rare_enrichment"] = {
                name: float(enrichment[name]["cluster_rates"][j] / max(global_rates[name], 1e-12))
                for name in rare_names
            }
        clusters.append(cluster)

    share_entropy = float(-(shares[shares > 0] * np.log(shares[shares > 0])).sum() / np.log(max(k, 2)))
    max_share = float(shares.max()) if shares.size else 0.0
    min_share = float(shares.min()) if shares.size else 0.0
    if enrichment:
        best_name = max(enrichment, key=lambda name: enrichment[name]["max_enrichment"])
        max_enrichment = float(enrichment[best_name]["max_enrichment"])
        best_cluster = int(enrichment[best_name]["max_cluster"])
    else:
        best_name, max_enrichment, best_cluster = "", float("nan"), -1
    flag = bool(enrichment) and max_enrichment < float(enrichment_min)
    flag_reason = (
        f"无簇对稀有结构富集 ≥{enrichment_min:g}×：max={max_enrichment:.2f}×（{best_name} @ cluster "
        f"{best_cluster}）；建议 --two-stage 两段式退路（先 2-means 分活跃/平稳，7 簇只花在活跃子集）"
        if flag
        else (
            f"稀有富集达标：{best_name} {max_enrichment:.2f}× @ cluster {best_cluster}"
            if enrichment
            else "规则标签缺失，无法做稀有富集体检（FLAG）"
        )
    )
    top1 = np.argmax(soft, axis=1)
    face = int(np.argmax(soft.max(axis=1)))
    boundary = int(np.argmin(gap))
    examples = []
    for tag, row in (("face", face), ("boundary", boundary)):
        order = np.argsort(soft[row])[::-1][:3]
        examples.append(
            {
                "kind": tag,
                "row": int(row),
                "assigned": int(assign[row]),
                "top3": [[int(c), float(soft[row, c])] for c in order],
                "top1_hard_match": bool(int(top1[row]) == int(assign[row])),
            }
        )
    fallback_command = (
        "tools/venv-python tools/fit_clusters.py --bc-dir <BC_DIR> --two-stage "
        "--out config/clusters/cluster_v1.npz --report config/clusters/cluster_v1.report.json"
    )
    return {
        "cluster_version": spec.cluster_version,
        "mode": spec.mode,
        "k": k,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": dict(spec.data_fingerprint),
        "feature_contract": spec.feature_contract,
        "fit_params": spec.fit_params,
        "objective": {"value": diag.get("objective"), "iterations": diag.get("iterations")},
        "size": {
            "counts": [int(v) for v in counts],
            "shares": [float(v) for v in shares],
            "max_share": max_share,
            "min_share": min_share,
            "capacity_ok": bool((capacity or {}).get("ok", max_share <= spec.fit_params.get("cap_ratio", 1.0) + 1e-9)),
            "capacity_bounds": {
                "max_share": float((capacity or {}).get("max_share", spec.fit_params.get("max_share", 1.0))),
                "min_share": float((capacity or {}).get("min_share", spec.fit_params.get("min_share", 0.0))),
            },
            "capacity_weighted_shares": [float(v) for v in (capacity or {}).get("weighted_shares", [])],
            "share_entropy_norm": share_entropy,
        },
        "soft_target": {
            "top1_mean": float(soft.max(axis=1).mean()),
            "top2_mass_mean": float(np.sort(soft, axis=1)[:, -2:].sum(axis=1).mean()),
            "gap_p05": float(np.percentile(gap, 5)),
            "gap_p50": float(np.percentile(gap, 50)),
            "examples": examples,
        },
        "weight": {
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "p50": float(np.percentile(w, 50)),
            "clip_low_frac": float(np.mean(w <= 0.1 + 1e-9)),
            "clip_high_frac": float(np.mean(w >= 10.0 - 1e-9)),
        },
        "clusters": clusters,
        "rare": {
            "labels": list(rare_names),
            "global_rates": global_rates,
            "max_enrichment": max_enrichment,
            "max_enrichment_label": best_name,
            "max_enrichment_cluster": best_cluster,
            "enrichment_min": float(enrichment_min),
            "details": enrichment,
        },
        "flag": flag,
        "flag_reason": flag_reason,
        "fallback": {
            "two_stage_available": True,
            "two_stage_command": fallback_command,
            "activity_stats": list(ACTIVITY_STATS),
            "p_active": spec.fit_params.get("p_active"),
        },
        "alignment": dict(alignment),
        "postprocess_log": list(spec.fit_params.get("postprocess_log") or []),
    }


def print_report(report: Mapping[str, Any]) -> None:
    """控制台体检表。"""
    print(f"[cluster] version={report['cluster_version']} mode={report['mode']} k={report['k']}")
    data = report.get("data") or {}
    print(
        f"[cluster] 数据: {data.get('source')} rows={data.get('rows')} "
        f"obs_fingerprint={data.get('obs_fingerprint')} git={data.get('git_hash')}"
    )
    headers = ["cluster", "size", "share", "w-share", "radius", "r_p95", "H_norm", "top1", "top-label", "max-enrich"]
    rows = []
    for cluster in report["clusters"]:
        rates = cluster.get("rule_label_rates") or {}
        top_label = max(rates, key=rates.get) if rates else "-"
        enrich = cluster.get("rare_enrichment") or {}
        max_enrich = max(enrich.values()) if enrich else float("nan")
        rows.append(
            [
                cluster["id"],
                cluster["size"],
                f"{cluster['share']:.3f}",
                f"{cluster['weight_share']:.3f}",
                f"{cluster['radius_mean']:.2f}",
                f"{cluster['radius_p95']:.2f}",
                f"{cluster['soft_entropy_norm_mean']:.3f}",
                f"{cluster['soft_top1_mean']:.3f}",
                top_label,
                "-" if not enrich else f"{max_enrich:.2f}x",
            ]
        )
    print(_shared_table(rows, headers))
    rare = report.get("rare") or {}
    print(
        f"[cluster] 稀有结构：{rare.get('labels')} → max_enrichment="
        f"{rare.get('max_enrichment')}×（{rare.get('max_enrichment_label')} @ cluster "
        f"{rare.get('max_enrichment_cluster')}）"
    )
    examples = (report.get("soft_target") or {}).get("examples") or []
    for example in examples:
        print(f"[cluster] soft 样例[{example['kind']}]: row={example['row']} top3={example['top3']}")
    if report.get("flag"):
        print(f"[cluster] FLAG: {report['flag_reason']}")
        print(f"[cluster] 退路命令: {report['fallback']['two_stage_command']}")
    else:
        print(f"[cluster] 体检通过: {report['flag_reason']}")


def dump_soft_targets(bc_dir: str | Path, spec: ClusterSpec, out: str | Path) -> Path:
    """把冻结 spec 的软目标/硬簇离线物化到 npz（可选；trainer 也可在线调用 ``soft_targets_from_obs``）。

    产物键：``router_soft_targets (N,8)``、``router_cluster (N,)``、``cluster_version``。
    """
    arrays, _ = load_bc_dataset(bc_dir)
    obs = current_frame_obs(arrays)
    soft = spec.soft_targets_from_obs(obs)
    hard = spec.assign_hard(spec.encode(obs))
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        router_soft_targets=np.asarray(soft, dtype=np.float32),
        router_cluster=np.asarray(hard, dtype=np.int64),
        cluster_version=np.asarray(spec.cluster_version),
    )
    print(f"[cluster] 已物化软目标：{target}（{soft.shape[0]} 行 × {soft.shape[1]} 簇）")
    return target


def run_fit(
    bc_dir: str | Path,
    *,
    out: str | Path = DEFAULT_SPEC,
    report_path: str | Path | None = None,
    dump_targets: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """端到端：读数据集 → 拟合 → 落盘 npz + report.json；返回 report 路径。"""
    arrays, meta = load_bc_dataset(bc_dir)
    result = fit_cluster_spec_from_dataset(
        arrays, meta, dataset_path=str(bc_dir), **kwargs
    )
    spec: ClusterSpec = result["spec"]
    out_path = Path(out)
    spec.save(out_path)
    report_target = Path(report_path) if report_path else out_path.with_suffix(".report.json")
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(
        json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print_report(result["report"])
    print(f"[cluster] 已写入 {out_path} 与 {report_target}")
    if dump_targets:
        dump_soft_targets(bc_dir, spec, dump_targets)
    return report_target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/fit_clusters.py",
        description="拟合场景原型聚类 spec（无监督 k-means + 容量约束 + 软目标）并输出体检报告",
    )
    parser.add_argument("--bc-dir", type=str, required=True, help="BC 数据集目录（expert_bc.npz）")
    parser.add_argument("--out", type=Path, default=Path(DEFAULT_SPEC), help="输出 fix 聚类 spec npz")
    parser.add_argument("--report", type=Path, default=None, help="体检报告 json（默认 <out>.report.json）")
    parser.add_argument("--cluster-version", type=str, default=CLUSTER_VERSION)
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="簇数（默认 8）")
    parser.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM, help="白化维数（默认 32；实际 min(32, 去常量后维数, N-1)）")
    parser.add_argument("--density-k", type=int, default=DEFAULT_DENSITY_K, help="kNN 密度邻居数")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="稀有度加权指数")
    parser.add_argument("--cap-ratio", type=float, default=DEFAULT_CAP_RATIO, help="每簇容量上限占比")
    parser.add_argument("--max-share", type=float, default=DEFAULT_MAX_SHARE, help="体检二分阈值")
    parser.add_argument("--min-share", type=float, default=DEFAULT_MIN_SHARE, help="体检并入阈值")
    parser.add_argument("--tau", type=float, default=DEFAULT_TEMPERATURE, help="top-2 softmax 温度")
    parser.add_argument("--boundary-margin", type=float, default=DEFAULT_BOUNDARY_MARGIN)
    parser.add_argument("--smooth-strength", type=float, default=DEFAULT_SMOOTH_STRENGTH)
    parser.add_argument("--rare-max-rate", type=float, default=0.05, help="稀有标签全局占比上限")
    parser.add_argument("--enrichment-min", type=float, default=3.0, help="稀有富集体检门限")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=0, help="调试用截断行数（0=全量）")
    parser.add_argument("--two-stage", action="store_true", help="两段式退路（活跃 2-means + 7 簇活跃子集）")
    parser.add_argument("--two-stage-max-bucket", type=float, default=0.65,
                        help="两段式活跃度分桶上限（自然 2-means 过于不均衡时钳制）")
    parser.add_argument("--restarts", type=int, default=8, help="flat 模式 restart 次数（取满足容量约束且 objective 最低）")
    parser.add_argument("--balanced-split", action="store_true",
                        help="lane D 变体（b）：活跃度分桶 + 两桶都做容量约束 k-means（仍是 flat spec）")
    parser.add_argument("--natural", action="store_true",
                        help="对照审阅：不加容量/份额约束的自然版（cap/floor 1.0/0.0）")
    parser.add_argument("--align-to", type=str, default=None, help="把新簇编号 Hungarian 对齐到旧 spec npz")
    parser.add_argument("--dump-targets", type=Path, default=None,
                        help="可选：把 soft_targets/cluster 物化到该 npz（trainer 也可在线调用）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run_fit(
            args.bc_dir,
            out=args.out,
            report_path=args.report,
            cluster_version=args.cluster_version,
            k=args.k,
            pca_dim=args.pca_dim,
            density_k=args.density_k,
            alpha=args.alpha,
            cap_ratio=args.cap_ratio,
            max_share=args.max_share,
            min_share=args.min_share,
            temperature=args.tau,
            boundary_margin=args.boundary_margin,
            smooth_strength=args.smooth_strength,
            rare_max_rate=args.rare_max_rate,
            enrichment_min=args.enrichment_min,
            seed=args.seed,
            max_iters=args.max_iters,
            max_rows=args.max_rows,
            two_stage=args.two_stage,
            two_stage_max_bucket=args.two_stage_max_bucket,
            restarts=args.restarts,
            balanced_split=args.balanced_split,
            natural=args.natural,
            align_to=args.align_to,
            dump_targets=args.dump_targets,
        )
    except Exception as exc:  # noqa: BLE001 - CLI 失败给出可读原因
        print(f"[cluster] 拟合失败：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
