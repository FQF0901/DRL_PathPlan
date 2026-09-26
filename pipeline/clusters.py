"""场景原型聚类（lane C，fix-4）：无监督 k-means 软目标，替代手工规则标签监督 router。

契约（``docs/design-v1.2.md`` §4）
=================================

1. **特征口径（只用当前帧 obs；禁用任何学习表征/latent/历史；簇 = 当前帧 obs 的可判函数）**::

       ego(8) + od(16×9) + ld(16×7) + others(F)

   - 展平顺序固定：ego → od（槽 0..15 行序）→ ld（槽 0..15 行序）→ others；
   - **od 有效性 = ``od_presence``（v2 优先）否则 ``od_mask``；无效槽位特征全部清零**
     （v2 固定槽位语义：对象出盒 ``presence=0`` 但槽位保留、原始特征可能非零 —— 不清零
     簇就不是当前帧的函数）；od 槽位顺序**原样保留**（v2 槽位 = track id，槽位序号本身
     是当前帧可观测信息，禁止重排/TTC 重选）；
   - ld 有效性 = ``ld_mask``，无效槽位清零；
   - others 规范来源 = v2 ``others`` 通道（nav(11)+speed_limit(1)+signal(4)+road_class one-hot）；
     v2 数据未就绪时的**等价 fallback** = ``nav(11) + speed_limit(ld 槽0 第4维) + signal(4)``；
     拟合时选定来源写入 spec（``others_source``），加载/推理严格按 spec 复现，禁止漂移。
   - 与 router 输入的一致性：``net`` 的 router 输入是同一份当前帧 obs 经编码器得到的 latent
     （固定槽位、同 mask 语义）；本模块给出的是同一帧的**确定性特征函数**，训练时作为软目标
     （router 学 latent → 簇），因此特征口径必须与在线 obs 完全同源。

2. 预处理：去常数/近零方差维 → 标准化 → PCA 白化到 ``pca_dim``（默认 48）。
3. 稀有度加权（rule-free）：白化空间 kNN 局部密度 ρ_i，
   ``w_i = clip((ρ_median/ρ_i)^α, 0.1, 10)``（α 默认 0.75）。
4. 容量约束加权 k-means：k=8，每簇 ≤ ``cap_ratio·N``（默认 25%），逐轮贪心 + 松弛填充 +
   成对交换改进；体检后处理：>``max_share``（30%）沿第一主方向二分、<``min_share``（2%）
   并入最近簇（全程记日志）。
5. 软目标：top-2 距离 ``softmax(-d/τ)``（τ 默认 0.5）→ 形如 0.65/0.35；top-2 相对间隔
   小于 ``boundary_margin`` 的边界样本向 top-3/均匀平滑；输出 ``(N,K)`` 行和=1。
6. 两段式退路（体检 FLAG 时启用）：先按连续活跃度统计量 2-means 分"活跃 vs 平稳"，
   K-1=7 个簇只花在活跃子集（见 :func:`activity_statistics`）。
7. 冻结与版本：``config/clusters/cluster_v1.npz``（质心/预处理参数/特征口径/数据指纹/git hash）；
   重聚类用 :func:`align_clusters`（Hungarian）对齐旧簇编号。

trainer（Stage B）调用契约
=========================

::

    from pipeline.clusters import load, soft_targets_from_obs

    soft = soft_targets_from_obs(obs_batch)        # (B,8) float32，行和=1
    # obs_batch 与 BCDataset.build_obs_batch 同口径：
    #   ego (B,8) / od (B,16,9) / od_mask (B,16) / ld (B,16,7) / ld_mask (B,16)
    #   nav (B,11) / signal (B,4)；v2 追加 od_presence (B,16) / others (B,F_o)
    # 软目标只用于体检之外的 router 监督（BCE / KL），规则标签绝不进入监督。

``load(spec)``：``spec`` 可为 npz 路径、``config/clusters/*.yaml|json``（取 ``spec`` 键 +
``cluster_version`` 校验 + τ 等覆盖）或 mapping；``None`` = ``config/clusters/default.yaml``。
返回 :class:`ClusterSpec`，进程内按路径缓存（``refresh=True`` 强制重读）。

形状约定：所有输入按 **batch 维在前**；单槽通道允许 ``(B,1,F)`` 或 ``(B,F)``。
torch 输入 → torch 输出（同 device；内部转 numpy 计算，确定性），numpy 输入 → numpy 输出。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:  # torch 仅用于输入/输出直通，缺失不影响聚类
    import torch as _torch
except Exception:  # noqa: BLE001 - 环境裁剪时退化为 numpy-only
    _torch = None  # type: ignore[assignment]

__all__ = [
    "CLUSTER_VERSION",
    "DEFAULT_SPEC",
    "DEFAULT_CONFIG",
    "DEFAULT_K",
    "DEFAULT_PCA_DIM",
    "DEFAULT_ALPHA",
    "DEFAULT_DENSITY_K",
    "DEFAULT_CAP_RATIO",
    "DEFAULT_MAX_SHARE",
    "DEFAULT_MIN_SHARE",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_BOUNDARY_MARGIN",
    "DEFAULT_SMOOTH_STRENGTH",
    "OD_SLOTS",
    "OD_DIM",
    "LD_SLOTS",
    "LD_DIM",
    "EGO_DIM",
    "FALLBACK_OTHERS_DIM",
    "FEATURE_CONTRACT",
    "ACTIVITY_STATS",
    "ClusterSpec",
    "KMeansResult",
    "load",
    "assign_soft",
    "soft_targets_from_obs",
    "encode_obs",
    "clear_cache",
    "fit_feature_pipeline",
    "density_weights",
    "fit_capacity_kmeans",
    "enforce_size_bounds",
    "align_clusters",
    "remap_assignment",
    "activity_statistics",
    "activity_score",
    "top2_soft_assignments",
]

# --------------------------------------------------------------------------- #
# 常量与特征口径
# --------------------------------------------------------------------------- #

CLUSTER_VERSION = "v1"
DEFAULT_SPEC = "config/clusters/cluster_v1.npz"
DEFAULT_CONFIG = "config/clusters/default.yaml"

DEFAULT_K = 8
DEFAULT_PCA_DIM = 48
DEFAULT_ALPHA = 0.75
DEFAULT_DENSITY_K = 16
DEFAULT_CAP_RATIO = 0.25
DEFAULT_MAX_SHARE = 0.30
DEFAULT_MIN_SHARE = 0.02
DEFAULT_TEMPERATURE = 0.5
DEFAULT_BOUNDARY_MARGIN = 0.05
DEFAULT_SMOOTH_STRENGTH = 0.5

OD_SLOTS = 16
OD_DIM = 9
LD_SLOTS = 16
LD_DIM = 7
EGO_DIM = 8
FALLBACK_OTHERS_DIM = 16  # nav(11) + speed_limit(1) + signal(4)

#: 特征口径（原样写入冻结 npz 的 ``meta.feature_contract``，加载时校验）
FEATURE_CONTRACT: Dict[str, Any] = {
    "name": "clusters.feature.v1",
    "frame": "current-frame obs only（禁用 latent/历史）",
    "order": ["ego", "od", "ld", "others"],
    "ego": {"dims": EGO_DIM},
    "od": {
        "slots": OD_SLOTS,
        "dims": OD_DIM,
        "validity": "od_presence if present else od_mask",
        "invalid_policy": "zero",
        "slot_order": "as_is（v2 固定槽位 = track id，禁止重排）",
    },
    "ld": {"slots": LD_SLOTS, "dims": LD_DIM, "validity": "ld_mask", "invalid_policy": "zero"},
    "others": {
        "raw_key": "others",
        "raw_composition": "nav(11)+speed_limit(1, 归一化)+signal(4)+road_class one-hot(12)（v2 §3.2，共 28 维）",
        "fallback": "nav(11)+speed_limit(ld slot0 dim4, 原始 m/s)+signal(4)（v2 前的旧数据；v2 数据集必须用 raw）",
    },
}

#: 两段式退路的连续活跃度统计量（无规则标签）
ACTIVITY_STATS: Tuple[str, ...] = ("od_present_count", "od_interaction", "ego_maneuver")

#: 模块根（相对路径解析基准 = 仓库根）
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

_OD_VALIDITY_KEYS = ("od_presence", "od_mask")


def _project_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else _PROJECT_ROOT / target


# --------------------------------------------------------------------------- #
# obs → 原始特征
# --------------------------------------------------------------------------- #


def _to_numpy(value: Any) -> np.ndarray:
    if _torch is not None and isinstance(value, _torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _squeeze_single_slot(array: np.ndarray, name: str) -> np.ndarray:
    """``(B,1,F) → (B,F)``；已是 ``(B,F)`` 原样返回。"""
    if array.ndim == 3 and array.shape[1] == 1:
        return array[:, 0]
    return array


def _channel(array: np.ndarray, name: str, slots: int, dims: int) -> np.ndarray:
    if array.ndim != 3 or array.shape[1] != slots or array.shape[2] != dims:
        raise ValueError(
            f"obs[{name!r}] 形状应为 (B,{slots},{dims})，收到 {tuple(array.shape)}"
        )
    return array


def _mask_channel(array: np.ndarray, name: str, slots: int) -> np.ndarray:
    """``(B,slots)`` 或 ``(B,slots,1)`` → ``(B,slots)``。"""
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2 or array.shape[1] != slots:
        raise ValueError(f"obs[{name!r}] 形状应为 (B,{slots})，收到 {tuple(array.shape)}")
    return array


def encode_obs(
    obs: Mapping[str, Any],
    *,
    others_source: str = "auto",
    od_slots: int = OD_SLOTS,
    ld_slots: int = LD_SLOTS,
) -> np.ndarray:
    """当前帧 obs → 原始特征 ``(N, ego+od+ld+others)``（float32）。

    ``others_source``：``"auto"``（有 ``others`` 键用 raw，否则 fallback）、``"raw"``、
    ``"fallback"``。无效 od/ld 槽位一律清零（od 有效性优先 ``od_presence``）。
    不修改输入数组（od/ld 掩码乘法在副本上做）。
    """
    if not isinstance(obs, Mapping):
        raise TypeError(f"obs 必须是映射（dict），收到 {type(obs).__name__}")
    if "ego" not in obs:
        raise ValueError("obs 缺少必需通道 'ego'")
    ego = _squeeze_single_slot(_to_numpy(obs["ego"]).astype(np.float64, copy=False), "ego")
    if ego.ndim != 2 or ego.shape[1] != EGO_DIM:
        raise ValueError(f"obs['ego'] 形状应为 (B,{EGO_DIM})，收到 {tuple(ego.shape)}")
    n = int(ego.shape[0])

    od = _channel(_to_numpy(obs.get("od")) if "od" in obs else np.zeros((n, od_slots, OD_DIM)), "od", od_slots, OD_DIM)
    od = od.astype(np.float64, copy=True)
    od_valid: Optional[np.ndarray] = None
    for key in _OD_VALIDITY_KEYS:
        if key in obs:
            od_valid = _mask_channel(_to_numpy(obs[key]).astype(np.float64, copy=False), key, od_slots)
            break
    if od_valid is not None and od_valid.shape[0] == n:
        od *= (od_valid != 0.0)[..., None]

    ld = _channel(_to_numpy(obs.get("ld")) if "ld" in obs else np.zeros((n, ld_slots, LD_DIM)), "ld", ld_slots, LD_DIM)
    ld = ld.astype(np.float64, copy=True)
    if "ld_mask" in obs:
        ld_valid = _mask_channel(_to_numpy(obs["ld_mask"]).astype(np.float64, copy=False), "ld_mask", ld_slots)
        ld *= (ld_valid != 0.0)[..., None]

    if others_source == "auto":
        others_source = "raw" if "others" in obs else "fallback"
    if others_source == "raw":
        if "others" not in obs:
            raise ValueError("others_source='raw' 但 obs 缺少 'others' 通道")
        others = _squeeze_single_slot(_to_numpy(obs["others"]).astype(np.float64, copy=False), "others")
        if others.ndim != 2:
            raise ValueError(f"obs['others'] 形状应为 (B,F)，收到 {tuple(others.shape)}")
    elif others_source == "fallback":
        nav = _squeeze_single_slot(_to_numpy(obs["nav"]).astype(np.float64, copy=False), "nav")
        signal = _squeeze_single_slot(_to_numpy(obs["signal"]).astype(np.float64, copy=False), "signal")
        if nav.ndim != 2 or nav.shape[1] != 11:
            raise ValueError(f"obs['nav'] 形状应为 (B,11)，收到 {tuple(nav.shape)}")
        if signal.ndim != 2 or signal.shape[1] != 4:
            raise ValueError(f"obs['signal'] 形状应为 (B,4)，收到 {tuple(signal.shape)}")
        others = np.concatenate([nav, ld[:, 0, 4:5], signal], axis=1)  # speed_limit = ld 槽0 第4维
    else:
        raise ValueError(f"未知 others_source {others_source!r}（应为 auto/raw/fallback）")

    if any(array.shape[0] != n for array in (od, ld, others)):
        raise ValueError(
            f"obs 各通道 batch 维不一致：ego={n}, od={od.shape[0]}, ld={ld.shape[0]}, others={others.shape[0]}"
        )
    raw = np.concatenate([ego, od.reshape(n, -1), ld.reshape(n, -1), others], axis=1)
    return raw.astype(np.float32)


# --------------------------------------------------------------------------- #
# 特征预处理（去近零方差 → 标准化 → PCA 白化）
# --------------------------------------------------------------------------- #


def fit_feature_pipeline(
    raw: np.ndarray,
    *,
    pca_dim: int = DEFAULT_PCA_DIM,
    var_floor: float = 1e-8,
    rel_var_frac: float = 1e-3,
) -> Dict[str, Any]:
    """拟合预处理参数并返回白化特征。

    去常数/近零方差：``std > var_floor`` 且 ``std > rel_var_frac · max(std)``。
    PCA 用协方差特征分解（确定性符号约定：最大绝对分量取正）。
    """
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError(f"特征矩阵形状非法：{tuple(x.shape)}（需 (N,D), N≥2）")
    std = x.std(axis=0)
    max_std = float(std.max()) if std.size else 0.0
    keep = (std > var_floor) & (std > rel_var_frac * max_std)
    if not np.any(keep):
        keep = std > var_floor
    if not np.any(keep):
        keep = np.ones_like(std, dtype=bool)  # 全常数兜底（后续标准化 std→1）
    keep_dims = np.flatnonzero(keep).astype(np.int64)
    xk = x[:, keep_dims]
    mean = xk.mean(axis=0)
    scale = xk.std(axis=0)
    scale = np.where(scale > var_floor, scale, 1.0)
    xs = (xk - mean) / scale

    d = int(min(max(int(pca_dim), 1), xs.shape[1], max(xs.shape[0] - 1, 1)))
    cov = (xs.T @ xs) / max(xs.shape[0] - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:d]
    components = evecs[:, order].T.copy()  # (d, D_kept)
    for row in components:  # 符号约定：最大绝对分量取正
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0.0:
            row *= -1.0
    explained = np.maximum(evals[order], 1e-12)
    whiten_scale = np.sqrt(explained)
    z = (xs @ components.T) / whiten_scale
    return {
        "keep_dims": keep_dims,
        "feature_mean": mean,
        "feature_std": scale,
        "components": components,
        "whiten_scale": whiten_scale,
        "explained_variance": explained,
        "z": z.astype(np.float32),
    }


# --------------------------------------------------------------------------- #
# 稀有度加权（kNN 局部密度，rule-free）
# --------------------------------------------------------------------------- #


def _knn_mean_distance(x: np.ndarray, k: int) -> np.ndarray:
    """每点到最近 k 个邻居（不含自身）的平均欧氏距离。优先 scipy cKDTree。"""
    n = x.shape[0]
    kk = int(min(max(int(k), 1), n - 1))
    if kk < 1:
        return np.zeros(n, dtype=np.float64)
    x64 = np.asarray(x, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree  # 延迟导入（缺失时走分块暴力）

        tree = cKDTree(x64)
        dist, _ = tree.query(x64, k=kk + 1)
        return np.asarray(dist[:, 1:]).mean(axis=1)
    except Exception:  # noqa: BLE001 - scipy 缺失：分块暴力，小数据可用
        out = np.empty(n, dtype=np.float64)
        for start in range(0, n, 2048):
            stop = min(start + 2048, n)
            block = x64[start:stop]
            d2 = (
                (block * block).sum(axis=1)[:, None]
                + (x64 * x64).sum(axis=1)[None, :]
                - 2.0 * (block @ x64.T)
            )
            np.maximum(d2, 0.0, out=d2)
            dist = np.sqrt(d2)
            for row in range(stop - start):
                dist[row, start + row] = np.inf  # 排除自身
            out[start:stop] = np.sort(dist, axis=1)[:, :kk].mean(axis=1)
        return out


def density_weights(
    x: np.ndarray,
    *,
    k: int = DEFAULT_DENSITY_K,
    alpha: float = DEFAULT_ALPHA,
    w_min: float = 0.1,
    w_max: float = 10.0,
) -> np.ndarray:
    """``w_i = clip((ρ_median/ρ_i)^α, w_min, w_max)``；ρ 为 kNN 局部密度（1/平均邻居距离）。"""
    z = np.asarray(x, dtype=np.float64)
    mean_dist = _knn_mean_distance(z, k)
    rho = 1.0 / np.maximum(mean_dist, 1e-9)
    median = float(np.median(rho))
    if not np.isfinite(median) or median <= 0.0:
        return np.ones(z.shape[0], dtype=np.float64)
    ratio = median / np.maximum(rho, 1e-12)
    return np.clip(ratio ** float(alpha), float(w_min), float(w_max))


# --------------------------------------------------------------------------- #
# 容量约束加权 k-means
# --------------------------------------------------------------------------- #


@dataclass
class KMeansResult:
    """拟合产物：质心/硬分配/目标值/迭代历史/后处理日志。"""

    centroids: np.ndarray
    assign: np.ndarray
    objective: float
    iterations: int
    history: list = field(default_factory=list)
    log: list = field(default_factory=list)
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def _pairwise_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    c64 = np.asarray(centers, dtype=np.float64)
    d2 = (x64 * x64).sum(axis=1)[:, None] + (c64 * c64).sum(axis=1)[None, :] - 2.0 * (x64 @ c64.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _weighted_centers(
    x: np.ndarray, w: np.ndarray, assign: np.ndarray, k: int
) -> np.ndarray:
    dim = x.shape[1]
    centers = np.zeros((k, dim), dtype=np.float64)
    for j in range(k):
        mask = assign == j
        if not np.any(mask):
            d = _pairwise_distances(x, centers)[:, j] if j else np.zeros(x.shape[0])
            fallback = int(np.argmax(w * d)) if np.any(d > 0) else int(np.argmax(w))
            centers[j] = x[fallback]
            continue
        wj = w[mask]
        total = float(wj.sum())
        centers[j] = (x[mask] * wj[:, None]).sum(axis=0) / total if total > 0 else x[mask].mean(axis=0)
    return centers


def _assign_capacity(x: np.ndarray, centers: np.ndarray, cap: int) -> Tuple[np.ndarray, np.ndarray]:
    """逐轮贪心容量分配：第 r 轮每个未分配点申请第 r 偏好簇，簇内按距离升序填到 cap。

    总容量 ``k·cap ≥ N`` ⇒ 必然全部分配（最后一轮兜底按最近未满簇）。
    """
    dist = _pairwise_distances(x, centers)
    n, k = dist.shape
    order = np.argsort(dist, axis=1, kind="stable")
    assign = np.full(n, -1, dtype=np.int64)
    counts = np.zeros(k, dtype=np.int64)
    for r in range(k):
        pending = np.flatnonzero(assign < 0)
        if pending.size == 0:
            break
        cand = order[pending, r]
        for c in range(k):
            room = int(cap - counts[c])
            if room <= 0:
                continue
            sel = pending[cand == c]
            if sel.size == 0:
                continue
            take = sel[np.argsort(dist[sel, c], kind="stable")[:room]]
            assign[take] = c
            counts[c] += take.size
    if np.any(assign < 0):  # 理论不可达（k·cap≥N）；保险：最近未满
        for i in np.flatnonzero(assign < 0):
            order_i = np.argsort(dist[i], kind="stable")
            for c in order_i:
                if counts[c] < cap:
                    assign[i] = c
                    counts[c] += 1
                    break
            else:
                assign[i] = int(order_i[0])
                counts[assign[i]] += 1
    return assign, dist


def _improve_assignment(
    dist: np.ndarray, assign: np.ndarray, cap: int, *, max_rounds: int = 2
) -> int:
    """容量保持的局部改进：① 未满簇吸收更近的点；② 成对交换（双方都更近才换）。"""
    n, k = dist.shape
    moved_total = 0
    rows = np.arange(n)
    for _ in range(max_rounds):
        moved = 0
        counts = np.bincount(assign, minlength=k)
        for b in range(k):
            room = int(cap - counts[b])
            if room <= 0:
                continue
            others = assign != b
            if not np.any(others):
                continue
            gain = dist[others, b] - dist[rows[others], assign[others]]
            idx = np.flatnonzero(others)[gain > 1e-9]
            if idx.size == 0:
                continue
            g = dist[idx, b] - dist[idx, assign[idx]]
            take = idx[np.argsort(g, kind="stable")[::-1][:room]]
            assign[take] = b
            counts[b] += take.size
            moved += take.size
        for a in range(k):
            for b in range(a + 1, k):
                ia = np.flatnonzero(assign == a)
                ib = np.flatnonzero(assign == b)
                if ia.size == 0 or ib.size == 0:
                    continue
                gain_a = dist[ia, a] - dist[ia, b]
                gain_b = dist[ib, b] - dist[ib, a]
                sel_a = ia[gain_a > 1e-9]
                sel_b = ib[gain_b > 1e-9]
                if sel_a.size == 0 or sel_b.size == 0:
                    continue
                ga = dist[sel_a, a] - dist[sel_a, b]
                gb = dist[sel_b, b] - dist[sel_b, a]
                m = int(min(ga.size, gb.size))
                oa = np.argsort(ga, kind="stable")[::-1][:m]
                ob = np.argsort(gb, kind="stable")[::-1][:m]
                take = int(np.sum((ga[oa] + gb[ob]) > 1e-9))
                if take <= 0:
                    continue
                pa = sel_a[oa[:take]]
                pb = sel_b[ob[:take]]
                assign[pa] = b
                assign[pb] = a
                moved += take
        moved_total += moved
        if moved == 0:
            break
    return moved_total


def _objective(dist: np.ndarray, assign: np.ndarray, w: np.ndarray) -> float:
    rows = np.arange(assign.shape[0])
    return float(np.sum(w * dist[rows, assign] ** 2))


def _kmeans_plus_plus(x: np.ndarray, w: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    first = int(rng.integers(n))
    centers = [x[first]]
    closest = ((x - centers[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        prob = w * closest
        total = float(prob.sum())
        if not np.isfinite(total) or total <= 0.0:
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=prob / total))
        centers.append(x[idx])
        closest = np.minimum(closest, ((x - x[idx]) ** 2).sum(axis=1))
    return np.stack(centers).astype(np.float64)


def _refine(
    x: np.ndarray, w: np.ndarray, centers: np.ndarray, cap: int, iters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """固定质心数的容量约束 Lloyd 精修（后处理重跑用；结束时保证无空簇）。"""
    centers = np.asarray(centers, dtype=np.float64)
    k = centers.shape[0]
    assign, dist = _assign_capacity(x, centers, cap)
    for _ in range(max(int(iters), 1)):
        centers = _weighted_centers(x, w, assign, k)
        assign, dist = _assign_capacity(x, centers, cap)
        if _improve_assignment(dist, assign, cap, max_rounds=1) == 0:
            break
    counts = np.bincount(assign, minlength=k)
    if np.any(counts == 0):
        rows = np.arange(x.shape[0])
        far = np.argsort(-(w * dist[rows, assign] ** 2), kind="stable")
        used = np.zeros(x.shape[0], dtype=bool)
        for j in np.flatnonzero(counts == 0):
            pick = next((int(i) for i in far if not used[i]), int(far[0]))
            used[pick] = True
            centers[j] = x[pick]
        assign, dist = _assign_capacity(x, centers, cap)
        _improve_assignment(dist, assign, cap, max_rounds=1)
    return centers, assign


def fit_capacity_kmeans(
    x: np.ndarray,
    w: np.ndarray,
    *,
    k: int = DEFAULT_K,
    cap_ratio: float = DEFAULT_CAP_RATIO,
    seed: int = 0,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> KMeansResult:
    """加权 + 每簇容量上限的 k-means（k-means++ 初始化，确定性种子）。"""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if w.shape[0] != n:
        raise ValueError(f"权重形状 {w.shape} 与特征 {x.shape} 不一致")
    cap = max(int(math.ceil(float(cap_ratio) * n)), int(math.ceil(n / max(k, 1))))
    rng = np.random.default_rng(int(seed))
    centers = _kmeans_plus_plus(x, w, k, rng)
    assign, dist = _assign_capacity(x, centers, cap)
    objective = _objective(dist, assign, w)
    history = [objective]
    iterations = 0
    for iteration in range(1, int(max_iters) + 1):
        iterations = iteration
        centers = _weighted_centers(x, w, assign, k)
        assign, dist = _assign_capacity(x, centers, cap)
        _improve_assignment(dist, assign, cap, max_rounds=1)
        new_objective = _objective(dist, assign, w)
        history.append(new_objective)
        if new_objective >= objective - tol * max(1.0, abs(objective)):
            objective = min(objective, new_objective)
            break
        objective = new_objective
    centers = _weighted_centers(x, w, assign, k)
    counts = np.bincount(assign, minlength=k).astype(np.int64)
    return KMeansResult(
        centroids=centers.astype(np.float32),
        assign=assign.astype(np.int64),
        objective=float(objective),
        iterations=int(iterations),
        history=[float(v) for v in history],
        log=[],
        counts=counts,
    )


# --------------------------------------------------------------------------- #
# 体检后处理：>max_share 二分、<min_share 并入最近簇
# --------------------------------------------------------------------------- #


def _weighted_mean(x: np.ndarray, w: np.ndarray, idx: np.ndarray) -> np.ndarray:
    wi = w[idx]
    total = float(wi.sum())
    if total <= 0.0:
        return x[idx].mean(axis=0)
    return (x[idx] * wi[:, None]).sum(axis=0) / total


def _split_cluster(
    x: np.ndarray, w: np.ndarray, centers: np.ndarray, assign: np.ndarray, j: int
) -> Tuple[np.ndarray, np.ndarray]:
    """沿簇内加权第一主方向二分；返回质心表（新簇追加在尾部）与重分配后的 assign。"""
    idx = np.flatnonzero(assign == j)
    if idx.size < 2:
        return centers, assign
    mu = _weighted_mean(x, w, idx)
    x0 = x[idx] - mu
    wc = w[idx]
    cov = (x0 * wc[:, None]).T @ x0 / max(float(wc.sum()), 1e-12)
    evals, evecs = np.linalg.eigh(cov)
    direction = evecs[:, -1]
    proj = x0 @ direction
    order = np.argsort(proj, kind="stable")
    cum = np.cumsum(wc[order])
    half = 0.5 * float(cum[-1])
    split_at = int(np.searchsorted(cum, half, side="left")) + 1
    split_at = min(max(split_at, 1), idx.size - 1)
    left = idx[order[:split_at]]
    right = idx[order[split_at:]]
    new_center = _weighted_mean(x, w, right)
    centers = np.vstack([np.asarray(centers, dtype=np.float64), new_center])
    centers[j] = _weighted_mean(x, w, left)
    assign = np.array(assign, dtype=np.int64, copy=True)
    assign[right] = centers.shape[0] - 1
    return centers, assign


def _merge_cluster(
    x: np.ndarray, w: np.ndarray, centers: np.ndarray, assign: np.ndarray, j: int
) -> Tuple[np.ndarray, np.ndarray]:
    """把簇 j 的成员并入最近簇，删除簇 j 并重编号。"""
    k = centers.shape[0]
    if k <= 1:
        return centers, assign
    others = np.delete(np.arange(k), j)
    members = np.flatnonzero(assign == j)
    rest = np.asarray(centers, dtype=np.float64)[others]
    if members.size:
        dj = _pairwise_distances(x[members], rest)
        assign = np.array(assign, dtype=np.int64, copy=True)
        assign[members] = others[np.argmin(dj, axis=1)]
    else:
        assign = np.array(assign, dtype=np.int64, copy=True)
    centers = np.delete(np.asarray(centers, dtype=np.float64), j, axis=0)
    assign[assign > j] -= 1
    return centers, assign


def enforce_size_bounds(
    x: np.ndarray,
    w: np.ndarray,
    centers: np.ndarray,
    assign: np.ndarray,
    *,
    target_k: Optional[int] = None,
    cap_ratio: float = DEFAULT_CAP_RATIO,
    max_share: float = DEFAULT_MAX_SHARE,
    min_share: float = DEFAULT_MIN_SHARE,
    max_rounds: int = 3,
    repair_iters: int = 10,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """体检后处理：>max_share 沿第一主方向二分；<min_share 并入最近簇；保持 K 不变并重精修。"""
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers, dtype=np.float64)
    assign = np.asarray(assign, dtype=np.int64)
    n = x.shape[0]
    target = int(target_k if target_k is not None else centers.shape[0])
    log: list = []

    def _shares() -> np.ndarray:
        return np.bincount(assign, minlength=centers.shape[0]).astype(np.float64) / max(n, 1)

    def _size_pass(label: str) -> bool:
        """二分过大簇 / 并入过小簇 / 保持 K=target；返回是否有改动。"""
        nonlocal centers, assign
        shares = _shares()
        changed = False
        guard = 0
        while shares.max() > max_share + 1e-12 and centers.shape[0] < 2 * target and guard < 2 * target:
            j = int(np.argmax(shares))
            before = int(np.sum(assign == j))
            centers, assign = _split_cluster(x, w, centers, assign, j)
            log.append({"op": "split" + label, "cluster": j, "size_before": before})
            shares = _shares()
            changed = True
            guard += 1
        guard = 0
        while shares.min() < min_share - 1e-12 and centers.shape[0] > 1 and guard < 2 * target:
            j = int(np.argmin(shares))
            before = int(np.sum(assign == j))
            centers, assign = _merge_cluster(x, w, centers, assign, j)
            log.append({"op": "merge_tiny" + label, "cluster": j, "size_before": before})
            shares = _shares()
            changed = True
            guard += 1
        while centers.shape[0] > target:
            j = int(np.argmin(shares))
            centers, assign = _merge_cluster(x, w, centers, assign, j)
            log.append({"op": "merge_to_k" + label, "cluster": j})
            shares = _shares()
            changed = True
        while centers.shape[0] < target:
            j = int(np.argmax(shares))
            centers, assign = _split_cluster(x, w, centers, assign, j)
            log.append({"op": "split_to_k" + label, "cluster": j})
            shares = _shares()
            changed = True
        return changed

    #: 精修用的硬容量：min(cap_ratio, max_share) —— refine 后必然 ≤ 该上限（且无空簇）
    def _tight_cap() -> int:
        return max(
            int(math.ceil(min(cap_ratio, max_share) * n)),
            int(math.ceil(n / max(target, 1))),
        )

    for _ in range(int(max_rounds)):
        if not _size_pass(f""):
            break
        centers, assign = _refine(x, w, centers, _tight_cap(), repair_iters)
    # 最后一次纯分配清理（保证 K/规模约束）→ 按最终 assign 同步质心（不再 refine，避免空簇）
    _size_pass("_final")
    centers = _weighted_centers(x, w, assign, centers.shape[0])
    return centers.astype(np.float32), assign.astype(np.int64), log


# --------------------------------------------------------------------------- #
# 软目标（top-2 softmax + 边界平滑）
# --------------------------------------------------------------------------- #


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def top2_soft_assignments(
    dist: np.ndarray,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN,
    smooth_strength: float = DEFAULT_SMOOTH_STRENGTH,
) -> np.ndarray:
    """``(N,k)`` 距离 → ``(N,k)`` 软目标（行和=1）。

    - top-2：``softmax([-d1, -d2]/τ)``；
    - 边界（相对间隔 ``(d2-d1)/(d1+d2) < boundary_margin``）：按 β 向
      ``0.5·top-3 softmax + 0.5·uniform(top-3)`` 平滑，β = smooth_strength·(1-gap/margin)。
    """
    dist = np.asarray(dist, dtype=np.float64)
    n, k = dist.shape
    tau = max(float(temperature), 1e-6)
    if k == 1:
        return np.ones((n, 1), dtype=np.float32)
    top_n = min(3, k)
    order = np.argsort(dist, axis=1, kind="stable")[:, :top_n]
    top_d = np.take_along_axis(dist, order, axis=1)
    d1, d2 = top_d[:, 0], top_d[:, 1]
    p2 = _softmax(np.stack([-d1, -d2], axis=1) / tau)
    gap = (d2 - d1) / (d2 + d1 + 1e-12)
    beta = np.clip(1.0 - gap / max(float(boundary_margin), 1e-9), 0.0, 1.0) * float(smooth_strength)
    beta = np.where(d2 > d1 + 1e-9, beta, 0.0)  # 完全等距无偏好时不平滑（避免任意偏置）
    out = np.zeros((n, k), dtype=np.float64)
    rows = np.arange(n)
    if top_n >= 3 and float(smooth_strength) > 0.0:
        p3 = _softmax(-top_d / tau)
        p_smooth = 0.5 * p3 + 0.5 * (1.0 / top_n)
    else:
        p_smooth = np.full((n, top_n), 0.5, dtype=np.float64)
    for col in range(top_n):
        out[rows, order[:, col]] = (1.0 - beta) * (p2[:, col] if col < 2 else 0.0) + beta * p_smooth[:, col]
    out /= out.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# 冻结 spec：加载/保存/推理
# --------------------------------------------------------------------------- #


@dataclass
class ClusterSpec:
    """冻结的聚类 spec（npz 可序列化）：预处理 + 质心 + 软目标参数 + 数据指纹。"""

    cluster_version: str = CLUSTER_VERSION
    mode: str = "flat"  # flat | two_stage
    k: int = DEFAULT_K
    feature_contract: Dict[str, Any] = field(default_factory=lambda: dict(FEATURE_CONTRACT))
    others_source: str = "fallback"
    raw_dim: int = 0
    keep_dims: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    feature_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    feature_std: np.ndarray = field(default_factory=lambda: np.ones(0))
    components: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    whiten_scale: np.ndarray = field(default_factory=lambda: np.ones(0))
    centroids: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    temperature: float = DEFAULT_TEMPERATURE
    boundary_margin: float = DEFAULT_BOUNDARY_MARGIN
    smooth_strength: float = DEFAULT_SMOOTH_STRENGTH
    explained_variance: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: two_stage：活跃度统计量（(3,) 均值/标准差/权重）；标量簇心在活动度轴上
    activity_mean: np.ndarray = field(default_factory=lambda: np.zeros(0))
    activity_std: np.ndarray = field(default_factory=lambda: np.ones(0))
    activity_weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    activity_centers: np.ndarray = field(default_factory=lambda: np.zeros(0))
    calm_reference: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fit_params: Dict[str, Any] = field(default_factory=dict)
    data_fingerprint: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ 文件
    @property
    def pca_dim(self) -> int:
        return int(self.centroids.shape[1])

    def to_params(self) -> Dict[str, Any]:
        """npz 键值（数组 + ``meta`` JSON 字符串）。"""
        meta = {
            "cluster_version": self.cluster_version,
            "mode": self.mode,
            "k": int(self.k),
            "others_source": self.others_source,
            "raw_dim": int(self.raw_dim),
            "temperature": float(self.temperature),
            "boundary_margin": float(self.boundary_margin),
            "smooth_strength": float(self.smooth_strength),
            "feature_contract": self.feature_contract,
            "fit_params": self.fit_params,
            "data_fingerprint": self.data_fingerprint,
        }
        return {
            "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True),
            "keep_dims": np.asarray(self.keep_dims, dtype=np.int64),
            "feature_mean": np.asarray(self.feature_mean, dtype=np.float64),
            "feature_std": np.asarray(self.feature_std, dtype=np.float64),
            "components": np.asarray(self.components, dtype=np.float64),
            "whiten_scale": np.asarray(self.whiten_scale, dtype=np.float64),
            "centroids": np.asarray(self.centroids, dtype=np.float64),
            "explained_variance": np.asarray(self.explained_variance, dtype=np.float64),
            "activity_mean": np.asarray(self.activity_mean, dtype=np.float64),
            "activity_std": np.asarray(self.activity_std, dtype=np.float64),
            "activity_weights": np.asarray(self.activity_weights, dtype=np.float64),
            "activity_centers": np.asarray(self.activity_centers, dtype=np.float64),
            "calm_reference": np.asarray(self.calm_reference, dtype=np.float64),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **self.to_params())
        return target

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "ClusterSpec":
        meta = json.loads(str(params["meta"]))
        spec = cls(
            cluster_version=str(meta.get("cluster_version") or CLUSTER_VERSION),
            mode=str(meta.get("mode") or "flat"),
            k=int(meta.get("k") or DEFAULT_K),
            feature_contract=dict(meta.get("feature_contract") or FEATURE_CONTRACT),
            others_source=str(meta.get("others_source") or "fallback"),
            raw_dim=int(meta.get("raw_dim") or 0),
            keep_dims=np.asarray(params["keep_dims"], dtype=np.int64),
            feature_mean=np.asarray(params["feature_mean"], dtype=np.float64),
            feature_std=np.asarray(params["feature_std"], dtype=np.float64),
            components=np.asarray(params["components"], dtype=np.float64),
            whiten_scale=np.asarray(params["whiten_scale"], dtype=np.float64),
            centroids=np.asarray(params["centroids"], dtype=np.float64),
            temperature=float(meta.get("temperature", DEFAULT_TEMPERATURE)),
            boundary_margin=float(meta.get("boundary_margin", DEFAULT_BOUNDARY_MARGIN)),
            smooth_strength=float(meta.get("smooth_strength", DEFAULT_SMOOTH_STRENGTH)),
            explained_variance=np.asarray(params.get("explained_variance", np.zeros(0)), dtype=np.float64),
            activity_mean=np.asarray(params.get("activity_mean", np.zeros(0)), dtype=np.float64),
            activity_std=np.asarray(params.get("activity_std", np.ones(0)), dtype=np.float64),
            activity_weights=np.asarray(params.get("activity_weights", np.zeros(0)), dtype=np.float64),
            activity_centers=np.asarray(params.get("activity_centers", np.zeros(0)), dtype=np.float64),
            calm_reference=np.asarray(params.get("calm_reference", np.zeros(0)), dtype=np.float64),
            fit_params=dict(meta.get("fit_params") or {}),
            data_fingerprint=dict(meta.get("data_fingerprint") or {}),
        )
        spec.validate()
        return spec

    @classmethod
    def load(cls, path: str | Path) -> "ClusterSpec":
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(
                f"聚类 spec 不存在：{target}；先用 tools/fit_clusters.py 拟合（见 config/clusters/README.md）"
            )
        with np.load(target, allow_pickle=False) as payload:
            params = {key: payload[key] for key in payload.files}
        return cls.from_params(params)

    def validate(self) -> None:
        """形状/口径自检（损坏文件立即失败，避免静默错位）。"""
        k = int(self.centroids.shape[0])
        if self.mode == "flat" and k != int(self.k):
            raise ValueError(f"centroids 行数 {k} != k={self.k}（mode=flat）")
        if self.mode == "two_stage" and k != int(self.k) - 1:
            raise ValueError(f"two_stage 活跃质心 {k} != k-1={int(self.k) - 1}")
        if self.components.shape[1] != self.keep_dims.shape[0]:
            raise ValueError("components 列数与 keep_dims 不一致（spec 损坏）")
        if self.centroids.shape[1] != self.components.shape[0]:
            raise ValueError("centroids 维数与 PCA 维数不一致（spec 损坏）")
        if self.raw_dim and self.keep_dims.size and int(self.keep_dims.max()) >= self.raw_dim:
            raise ValueError("keep_dims 越界（spec 损坏）")
        if self.mode == "two_stage":
            for name in ("activity_mean", "activity_std", "activity_weights"):
                array = getattr(self, name)
                if array.shape[0] != len(ACTIVITY_STATS):
                    raise ValueError(f"two_stage 缺少 {name}（应为 {len(ACTIVITY_STATS)} 维）")
            if self.activity_centers.shape[0] != 2:
                raise ValueError("two_stage 需要 2 个活动度簇心")

    # ------------------------------------------------------------------ 推理
    def encode(self, obs: Mapping[str, Any]) -> np.ndarray:
        return encode_obs(obs, others_source=self.others_source)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """原始特征 → 白化特征 ``(N,d)``（严格按冻结参数）。"""
        x = np.asarray(raw, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"原始特征应为 (N,D)，收到 {tuple(x.shape)}")
        if self.raw_dim and x.shape[1] != int(self.raw_dim):
            raise ValueError(
                f"特征维数 {x.shape[1]} != spec.raw_dim {int(self.raw_dim)}"
                "（数据口径/others 来源不一致，需重新拟合或改用匹配的 spec）"
            )
        x = x[:, self.keep_dims]
        standardized = (x - self.feature_mean) / self.feature_std
        return ((standardized @ self.components.T) / self.whiten_scale).astype(np.float32)

    def distances(self, z: np.ndarray) -> np.ndarray:
        """白化特征 → 到各质心的欧氏距离 ``(N,k)``。"""
        return _pairwise_distances(z, np.asarray(self.centroids, dtype=np.float64)).astype(np.float32)

    def assign_hard(self, raw: np.ndarray) -> np.ndarray:
        z = self.transform(raw)
        return np.argmin(self.distances(z).astype(np.float64), axis=1).astype(np.int64)

    def activity(self, raw: np.ndarray) -> np.ndarray:
        """两段式活跃度得分（标准化的 3 统计量加权平均）。"""
        stats = activity_statistics(np.asarray(raw, dtype=np.float64))
        z = (stats - self.activity_mean) / np.maximum(self.activity_std, 1e-8)
        return z @ self.activity_weights

    def assign_soft(
        self,
        raw: np.ndarray,
        *,
        temperature: Optional[float] = None,
        boundary_margin: Optional[float] = None,
        smooth_strength: Optional[float] = None,
    ) -> np.ndarray:
        """原始特征 ``(N,D)`` → 软目标 ``(N,K)``（float32，行和=1）。"""
        tau = self.temperature if temperature is None else float(temperature)
        margin = self.boundary_margin if boundary_margin is None else float(boundary_margin)
        strength = self.smooth_strength if smooth_strength is None else float(smooth_strength)
        z = self.transform(raw)
        dist = self.distances(z).astype(np.float64)
        if self.mode == "flat":
            return top2_soft_assignments(
                dist, temperature=tau, boundary_margin=margin, smooth_strength=strength
            )
        # two_stage：活跃度门控 × 活跃子集 top-2
        score = self.activity(np.asarray(raw, dtype=np.float64))
        calm_center, active_center = float(self.activity_centers[0]), float(self.activity_centers[1])
        pair = np.stack([-np.abs(score - calm_center), -np.abs(score - active_center)], axis=1) / max(tau, 1e-6)
        p_pair = _softmax(pair)
        active_soft = top2_soft_assignments(
            dist, temperature=tau, boundary_margin=margin, smooth_strength=strength
        )
        out = np.zeros((raw.shape[0], int(self.k)), dtype=np.float64)
        out[:, 0] = p_pair[:, 0]
        out[:, 1:] = p_pair[:, 1:2] * active_soft
        out /= out.sum(axis=1, keepdims=True)
        return out.astype(np.float32)

    def soft_targets_from_obs(self, obs: Mapping[str, Any]) -> np.ndarray:
        """obs → 软目标 ``(N,K)``（encode + assign_soft）。"""
        return self.assign_soft(self.encode(obs))


# --------------------------------------------------------------------------- #
# load / 模块级便捷入口（含 torch 直通）
# --------------------------------------------------------------------------- #

_CACHE: Dict[str, ClusterSpec] = {}


def _load_structured_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"聚类配置不存在：{path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - 环境缺失时给出可操作提示
            raise ImportError(f"读取 {path} 需要 PyYAML；或直接把 spec 指到 .npz") from exc
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


def _resolve_spec(
    spec: Any,
) -> Tuple[Path, Dict[str, float], Optional[str]]:
    """→ (npz 路径, 覆盖参数, 期望 cluster_version)。"""
    if spec is None:
        return _resolve_spec(_load_structured_file(_project_path(DEFAULT_CONFIG)))
    if isinstance(spec, Mapping):
        target = spec.get("spec") or spec.get("path")
        if target is None:
            target = DEFAULT_SPEC
        overrides = {
            key: float(spec[key])
            for key in ("temperature", "boundary_margin", "smooth_strength")
            if key in spec
        }
        expected = spec.get("cluster_version")
        return _project_path(str(target)), overrides, (str(expected) if expected else None)
    path = _project_path(str(spec))
    if path.suffix.lower() in (".yaml", ".yml", ".json"):
        return _resolve_spec(_load_structured_file(path))
    return path, {}, None


def load(spec: Any = None, *, refresh: bool = False) -> ClusterSpec:
    """加载冻结聚类 spec（进程内缓存）。

    ``spec``：npz 路径 / ``config/clusters/*.yaml|json`` / mapping / 已加载的
    :class:`ClusterSpec` / ``None``（默认配置）。
    yaml/json 可带 ``cluster_version``（与 artifact 不一致直接报错）及
    ``temperature/boundary_margin/smooth_strength`` 覆盖。
    """
    if isinstance(spec, ClusterSpec):
        return spec
    path, overrides, expected = _resolve_spec(spec)
    key = str(path.resolve())
    if refresh:
        _CACHE.pop(key, None)
    model = _CACHE.get(key)
    if model is None:
        model = ClusterSpec.load(path)
        _CACHE[key] = model
    if expected is not None and expected != model.cluster_version:
        raise ValueError(
            f"cluster_version 不一致：配置 {expected!r} != artifact {model.cluster_version!r}"
            f"（{path}）；请重拟合或修正配置"
        )
    if overrides:
        model = replace(model, **overrides)
    return model


def clear_cache() -> None:
    """清空 ``load`` 缓存（测试/重拟合后使用）。"""
    _CACHE.clear()


def _torch_input(obs: Mapping[str, Any]) -> Optional[Any]:
    if _torch is None:
        return None
    for value in obs.values():
        if isinstance(value, _torch.Tensor):
            return value
    return None


def assign_soft(features: np.ndarray, *, spec: Any = None, **kwargs: Any) -> np.ndarray:
    """模块级便捷入口：**原始特征** ``(N,D)`` → 软目标 ``(N,K)``。"""
    model = load(spec)
    return model.assign_soft(features, **kwargs)


def soft_targets_from_obs(obs_batch: Mapping[str, Any], *, spec: Any = None, **kwargs: Any) -> np.ndarray:
    """模块级便捷入口：obs batch → 软目标 ``(N,K)``；torch 输入 → torch 输出。"""
    model = load(spec)
    out = model.soft_targets_from_obs(obs_batch, **kwargs)
    tensor = _torch_input(obs_batch)
    if tensor is not None:
        return _torch.as_tensor(out, dtype=_torch.float32, device=tensor.device)
    return out


# --------------------------------------------------------------------------- #
# 两段式退路：活跃度统计 + 对齐工具
# --------------------------------------------------------------------------- #


def activity_statistics(raw: np.ndarray) -> np.ndarray:
    """连续活跃度统计量 ``(N,3)``：od 在场数 / OD 交互强度 / 自车机动强度。

    只用当前帧特征（od 无效槽位已清零）：
    - ``od_present_count``：od 行非零槽位数；
    - ``od_interaction``：``Σ presence · max(0,-vx_rel)/max(|dx|,1)``（接近速度/距离）；
    - ``ego_maneuver``：``|a_lat| + |v·yaw_rate| + |curvature|·v²``。
    """
    x = np.asarray(raw, dtype=np.float64)
    edp = EGO_DIM
    od = x[:, edp : edp + OD_SLOTS * OD_DIM].reshape(-1, OD_SLOTS, OD_DIM)
    present = (np.abs(od).sum(axis=-1) > 0.0).astype(np.float64)
    count = present.sum(axis=1)
    dx = od[..., 0]
    vx = od[..., 2]
    closing = np.clip(-vx, 0.0, None)
    interaction = (present * closing / np.clip(np.abs(dx), 1.0, None)).sum(axis=1)
    ego = x[:, :EGO_DIM]
    v, a_lat, yaw_rate, curvature = ego[:, 0], ego[:, 2], ego[:, 3], ego[:, 5]
    maneuver = np.abs(a_lat) + np.abs(v * yaw_rate) + np.abs(curvature) * v * v
    return np.stack([count, interaction, maneuver], axis=1)


def activity_score(raw: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """按冻结参数（mean/std/weights）计算标量活跃度得分。"""
    mean = np.asarray(params["activity_mean"], dtype=np.float64)
    std = np.asarray(params["activity_std"], dtype=np.float64)
    weights = np.asarray(params["activity_weights"], dtype=np.float64)
    stats = activity_statistics(raw)
    return ((stats - mean) / np.maximum(std, 1e-8)) @ weights


def align_clusters(source_centroids: np.ndarray, target_centroids: np.ndarray) -> np.ndarray:
    """Hungarian 对齐：返回 ``perm`` 使 ``target[perm]`` 与 ``source`` 一一最小代价对应。

    重聚类流程：拟合新 spec → ``perm = align_clusters(old.centroids, new.centroids)`` →
    ``new_aligned = new.centroids[perm]`` → ``new_assign = remap_assignment(new_assign, perm)``。
    """
    a = np.asarray(source_centroids, dtype=np.float64)
    b = np.asarray(target_centroids, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"簇数不一致：source {a.shape[0]} vs target {b.shape[0]}")
    cost = _pairwise_distances(a, b)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
    except Exception:  # noqa: BLE001 - 贪心兜底（小规模，或 scipy 缺失）
        rows, cols = np.arange(a.shape[0]), np.full(a.shape[0], -1, dtype=np.int64)
        used = np.zeros(b.shape[0], dtype=bool)
        for row in rows:
            order = np.argsort(cost[row], kind="stable")
            for col in order:
                if not used[col]:
                    cols[row] = col
                    used[col] = True
                    break
    perm = np.empty(a.shape[0], dtype=np.int64)
    perm[rows] = cols
    return perm


def remap_assignment(assign: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """把旧编号下的硬分配映射到新编号：``new[perm[i]] = old[i]``。"""
    a = np.asarray(assign, dtype=np.int64)
    p = np.asarray(perm, dtype=np.int64)
    inverse = np.empty_like(p)
    inverse[p] = np.arange(p.shape[0], dtype=p.dtype)
    return inverse[a]
