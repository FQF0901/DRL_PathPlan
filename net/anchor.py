"""K-anchor 计划头工具（v7 结构迭代 B）：形状锚字典 / WTA 分配 / 车道系变换。

设计（fix-3 可行性报告 §5.2 方案 A；预注册见 ``docs/v7_program_prereg.md`` §11）
-------------------------------------------------------------------------------
- **形状锚**：K=6 条 ``(ds, dθ)`` 6 步链，锚 = expert ``cumdtheta`` KMeans 簇的物理原型
  （`tools/fit_plan_anchors.py` 拟合；簇大小与 fix-3 逐位一致）。
- **车道系**：锚的 dθ 以 **lane 帧**表达（lane 航向 − ego 航向 = ``heading_err``，
  ``curvature`` = κ）。ego 系 dθ = lane 帧 dθ + "沿车道跟随"剖面
  （``heading_i = Δψ + κ·s_i``；``dθ_0 = Δψ + κ·ds_0``、``dθ_i = κ·ds_i``）。
  ``(ds, dθ)`` 是车体量：对整条路径的旋转/平移不变——lane 帧的唯一作用就是这层
  航向剖面修正（fix-3 §4.3 口径）。车道无效 → Δψ=κ=0 → 恒等（旧行为兼容）。
- **软混合（方案 A）**：``plan = Σ_k p_k·(anchor_k + residual_k)``，
  ``p = softmax(logits/τ)``；``ds`` 由连续速度头给出（形状锚不含速度档位），
  residual 为逐锚 6×2 修正（WTA 训练）。
- **WTA 分配**：形状空间 = lane 帧累积 dθ；按逐 step 有效掩码加权 RMS 最近锚分配。

本模块不 import torch.nn（纯函数 + 张量），供 net / pipeline 共用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import torch
from torch import Tensor

#: 默认锚数（fix-3 推荐 K=6）
K_DEFAULT = 6
#: 锚特征口径标识（写入锚文件）
ANCHOR_FEATURE = "cumdtheta_lane"
#: 默认锚文件（相对仓库根；缺省/缺失时回退 :data:`DEFAULT_ANCHOR_DS/_DTH_LANE`）
ANCHOR_FILE_DEFAULT = "config/plan_anchors_k6.json"

# --------------------------------------------------------------------------- #
# 内置默认字典（K=6；与 config/plan_anchors_k6.json 同源）
# --------------------------------------------------------------------------- #
#: 来源：`tools/fit_plan_anchors.py` 在 `datasets/BTC20261002-0941_expert5k_v41` 上拟合
#: （KMeans K=6, n_init=10, random_state=0；簇大小 [92519,7823,3012,8461,4515,3670]，
#: 与 fix-3 `kanchor_shape_results.json` 逐位一致）。dθ 为 **lane 帧**原型。
DEFAULT_ANCHOR_DS: tuple[tuple[float, ...], ...] = (
    (3.547, 3.547, 3.553, 3.559, 3.560, 3.556),
    (3.485, 3.516, 3.529, 3.513, 3.508, 3.507),
    (3.259, 3.231, 3.064, 2.975, 2.971, 2.955),
    (3.132, 3.130, 3.119, 3.104, 3.076, 3.063),
    (3.212, 3.221, 3.210, 3.106, 2.945, 2.924),
    (3.223, 3.152, 3.139, 3.178, 3.227, 3.268),
)
DEFAULT_ANCHOR_DTH_LANE: tuple[tuple[float, ...], ...] = (
    (0.0000, -0.0000, -0.0000, 0.0003, 0.0002, 0.0005),
    (0.0951, -0.0376, -0.0354, -0.0210, 0.0009, 0.0111),
    (-0.1141, 0.0501, 0.0860, 0.0802, -0.0016, -0.0352),
    (0.0350, -0.0191, -0.0064, -0.0070, -0.0169, -0.0259),
    (-0.0381, -0.0017, 0.0131, 0.0439, 0.1109, 0.1096),
    (0.0242, 0.0779, 0.0001, -0.0420, -0.0575, -0.0457),
)
#: 簇语义（fix-3 §2.2；仅用于文档/监控标签）
ANCHOR_SEMANTICS: tuple[str, ...] = (
    "straight",
    "right_curve",
    "left_then_back",
    "gentle_right",
    "left_curve",
    "roundabout_left_back",
)


# --------------------------------------------------------------------------- #
# 锚字典加载 / 校验
# --------------------------------------------------------------------------- #
def anchors_from_payload(payload: Mapping[str, object], expected_k: Optional[int] = None) -> Tensor:
    """锚文件 payload → ``(K,6,2)`` 张量（lane 帧 ``ds``/``dtheta_lane``）；非法即报错。"""
    anchors = payload.get("anchors")
    if not isinstance(anchors, list) or not anchors:
        raise ValueError("锚文件缺少非空 'anchors' 列表")
    rows: list[list[list[float]]] = []
    for index, entry in enumerate(anchors):
        if not isinstance(entry, Mapping):
            raise ValueError(f"anchors[{index}] 应为映射（含 ds/dtheta_lane）")
        ds = entry.get("ds")
        dth = entry.get("dtheta_lane")
        if not isinstance(ds, (list, tuple)) or len(ds) != 6:
            raise ValueError(f"anchors[{index}]['ds'] 应为 6 维")
        if not isinstance(dth, (list, tuple)) or len(dth) != 6:
            raise ValueError(f"anchors[{index}]['dtheta_lane'] 应为 6 维")
        rows.append([[float(ds[i]), float(dth[i])] for i in range(6)])
    tensor = torch.tensor(rows, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("锚字典含非有限值")
    if bool((tensor[..., 0] < 0.0).any()):
        raise ValueError("锚字典 ds 必须非负")
    if expected_k is not None and int(tensor.shape[0]) != int(expected_k):
        raise ValueError(f"锚字典 K={int(tensor.shape[0])} != 期望 {int(expected_k)}")
    return tensor


def load_anchor_dictionary(
    path: Optional[str] = None, *, expected_k: Optional[int] = None
) -> Tensor:
    """加载锚字典 ``(K,6,2)``。

    ``path`` 为 None 或文件不存在 → 回退内置默认（K=6，见
    :data:`DEFAULT_ANCHOR_DS`/:data:`DEFAULT_ANCHOR_DTH_LANE`）；文件存在但非法 → 报错
    （不静默回退，避免实验口径漂移）。
    """
    if path:
        target = Path(path)
        if not target.exists() and not target.is_absolute():
            # 相对路径兜底：相对仓库根解析（eval worker 的 cwd 可能不同）
            target = Path(__file__).resolve().parents[1] / target
        if target.exists():
            payload = json.loads(target.read_text(encoding="utf-8"))
            return anchors_from_payload(payload, expected_k=expected_k)
    return default_anchors(expected_k=expected_k)


def default_anchors(*, expected_k: Optional[int] = None) -> Tensor:
    """内置默认锚字典 ``(6,6,2)``（fix-3 K=6 原型；lane 帧 dθ）。"""
    rows = [
        [[DEFAULT_ANCHOR_DS[k][i], DEFAULT_ANCHOR_DTH_LANE[k][i]] for i in range(6)]
        for k in range(len(DEFAULT_ANCHOR_DS))
    ]
    tensor = torch.tensor(rows, dtype=torch.float32)
    if expected_k is not None and int(tensor.shape[0]) != int(expected_k):
        raise ValueError(
            f"内置默认锚字典 K={int(tensor.shape[0])} != 期望 {int(expected_k)}"
            "（请提供 --anchor-path 指定锚文件）"
        )
    return tensor


# --------------------------------------------------------------------------- #
# 车道系变换（lane 帧 ↔ ego 帧 dθ 剖面）
# --------------------------------------------------------------------------- #
def _column(x: Optional[Tensor]) -> Optional[Tensor]:
    if x is None:
        return None
    if x.ndim == 2:
        return x.reshape(-1)
    if x.ndim != 1:
        raise ValueError(f"期望 (B,) 或 (B,1)，收到 {tuple(x.shape)}")
    return x


def lane_follow_dtheta(
    ds: Tensor,
    heading_err: Tensor,
    curvature: Tensor,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """ego 系"沿车道跟随"dθ 剖面 ``(B,6)``。

    ``heading_i = Δψ + κ·s_i``（``s = cumsum(ds)``）⇒ ``dθ_0 = Δψ + κ·ds_0``、
    ``dθ_i = κ·ds_i``（i≥1）。``valid`` 为 0 的行恒 0（恒等变换）。
    """
    if ds.ndim != 2 or int(ds.shape[-1]) != 6:
        raise ValueError(f"ds 应为 (B,6)，收到 {tuple(ds.shape)}")
    dpsi = _column(heading_err)
    kappa = _column(curvature)
    if dpsi is None or kappa is None:
        raise ValueError("heading_err/curvature 不能为 None")
    follow = kappa.reshape(-1, 1) * ds
    follow = torch.cat(
        [(dpsi + kappa * ds[:, 0]).reshape(-1, 1), follow[:, 1:]], dim=-1
    )
    if valid is not None:
        follow = follow * _column(valid).reshape(-1, 1).to(follow.dtype)
    return follow


def to_lane_dtheta(
    chain: Tensor,
    heading_err: Tensor,
    curvature: Tensor,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """ego 系 ``(B,6,2)`` 链 → lane 帧 dθ（``dθ_lane = dθ_ego − follow``）。"""
    if chain.ndim != 3 or int(chain.shape[-1]) != 2:
        raise ValueError(f"chain 应为 (B,6,2)，收到 {tuple(chain.shape)}")
    out = chain.clone()
    out[..., 1] = chain[..., 1] - lane_follow_dtheta(
        chain[..., 0], heading_err, curvature, valid
    )
    return out


def from_lane_dtheta(
    chain_lane: Tensor,
    ds: Tensor,
    heading_err: Tensor,
    curvature: Tensor,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """lane 帧 dθ + 速度剖面 ``ds`` → ego 系链（``dθ_ego = dθ_lane + follow``）。

    ``chain_lane`` 的 ds 维不参与（调用方以 ``ds`` 为准；本函数只做航向剖面回投）。
    """
    if chain_lane.ndim != 3 or int(chain_lane.shape[-1]) != 2:
        raise ValueError(f"chain_lane 应为 (B,6,2)，收到 {tuple(chain_lane.shape)}")
    out = chain_lane.clone()
    out[..., 0] = ds
    out[..., 1] = chain_lane[..., 1] + lane_follow_dtheta(ds, heading_err, curvature, valid)
    return out


def anchor_lane_context(lane_feat: Tensor, lane_mask: Optional[Tensor] = None) -> Tensor:
    """lane 原始块 ``(B,17)`` + mask → ``(B,3)`` = ``[heading_err, curvature, valid]``。

    口径：``heading_err`` = lane 航向 − ego 航向（v5 ``lane`` 块 dim 1；正 = 车道在左），
    ``curvature`` = 投影点 dθ/ds（dim 3）；``valid`` = ``lane_mask × near_valid``
    （车道末端几何清零 → 恒等变换，旧行为兼容）。
    """
    if lane_feat.ndim != 2 or int(lane_feat.shape[-1]) < 17:
        raise ValueError(f"lane_feat 应为 (B,17)，收到 {tuple(lane_feat.shape)}")
    valid = torch.ones(
        int(lane_feat.shape[0]), dtype=lane_feat.dtype, device=lane_feat.device
    )
    if lane_mask is not None:
        valid = (lane_mask.reshape(-1).to(device=lane_feat.device) > 0.5).to(lane_feat.dtype)
    valid = valid * (lane_feat[:, 14] > 0.5).to(lane_feat.dtype)
    return torch.stack([lane_feat[:, 1], lane_feat[:, 3], valid], dim=-1)


def lane_shape_features(
    chain: Tensor,
    heading_err: Tensor,
    curvature: Tensor,
    valid: Optional[Tensor] = None,
) -> Tensor:
    """形状特征 = lane 帧 dθ 的累积和 ``(B,6)``（WTA 分配空间；帧不变表示）。"""
    lane = to_lane_dtheta(chain, heading_err, curvature, valid)
    return torch.cumsum(lane[..., 1], dim=-1)


# --------------------------------------------------------------------------- #
# WTA 分配
# --------------------------------------------------------------------------- #
def assign_anchors(
    chain: Tensor,
    anchor_dth: Tensor,
    chain_valid: Tensor,
    heading_err: Tensor,
    curvature: Tensor,
    lane_valid: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    """按形状空间（lane 帧累积 dθ）最近锚分配（WTA）。

    距离 = 逐 step 有效掩码加权 RMS：``sqrt(Σ v·(cumdθ − cumA_k)² / Σ v)``。
    返回 ``{index (B,), dist (B,), margin (B,), row_valid (B,), target (B,6)}``；
    无有效 step 的行 ``row_valid=False``、``index=0``（训练侧按行掩码跳过）。
    """
    if chain_valid.ndim != 2 or int(chain_valid.shape[1]) != 6:
        raise ValueError(f"chain_valid 应为 (B,6)，收到 {tuple(chain_valid.shape)}")
    anchor_dth = anchor_dth.to(dtype=chain.dtype, device=chain.device)
    target = lane_shape_features(chain, heading_err, curvature, lane_valid)
    anchor_cum = torch.cumsum(anchor_dth, dim=-1)  # (K,6)
    diff = target.unsqueeze(1) - anchor_cum.unsqueeze(0)  # (B,K,6)
    weight = chain_valid.to(dtype=chain.dtype, device=chain.device)
    denominator = weight.sum(dim=-1).clamp(min=1e-6)  # (B,)
    dist = torch.sqrt(
        ((diff ** 2) * weight.unsqueeze(1)).sum(dim=-1) / denominator.unsqueeze(-1)
    )  # (B,K)
    row_valid = weight.sum(dim=-1) > 0.5
    index = dist.argmin(dim=-1)
    if int(dist.shape[1]) >= 2:
        top2 = dist.topk(2, dim=-1, largest=False).values
        margin = top2[:, 1] - top2[:, 0]
    else:
        margin = torch.zeros_like(dist[:, 0])
    return {
        "index": index,
        "dist": dist.gather(1, index.unsqueeze(1)).squeeze(1),
        "margin": margin,
        "row_valid": row_valid,
        "target": target,
    }


# --------------------------------------------------------------------------- #
# 软混合 / 指派计划
# --------------------------------------------------------------------------- #
def mixture_probabilities(logits: Tensor, temperature: float = 1.0, hard: bool = False) -> Tensor:
    """``softmax(logits/τ)``；``hard=True``（推理 argmax）→ one-hot。"""
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature 必须 > 0，收到 {temperature}")
    probs = torch.softmax(logits / float(temperature), dim=-1)
    if hard:
        index = probs.argmax(dim=-1, keepdim=True)
        probs = torch.zeros_like(probs).scatter_(1, index, 1.0)
    return probs


def anchor_mixture(
    logits: Tensor,
    residual: Tensor,
    anchor_dth: Tensor,
    speed_ds: Tensor,
    heading_err: Tensor,
    curvature: Tensor,
    lane_valid: Optional[Tensor] = None,
    *,
    temperature: float = 1.0,
    hard: bool = False,
) -> tuple[Tensor, Tensor]:
    """方案 A 软混合：``plan = Σ_k p_k·(anchor_k + residual_k)`` → ``(plan (B,6,2), probs (B,K))``。

    ``anchor_k`` 的 ds 维由连续速度头 ``speed_ds (B,6)`` 承担（形状锚不含速度档位）；
    故 ``plan_ds = speed_ds + Σ p_k·res_ds_k``、``plan_dθ_lane = Σ p_k·(anchor_dθ_k + res_dθ_k)``，
    最后经 lane 帧 → ego 帧回投。
    """
    probs = mixture_probabilities(logits, temperature, hard)
    residual = residual.to(dtype=speed_ds.dtype, device=speed_ds.device)
    res_ds = torch.einsum("bk,bkd->bd", probs, residual[..., 0])
    res_dth = torch.einsum("bk,bkd->bd", probs, residual[..., 1])
    anchor_dth = anchor_dth.to(dtype=speed_ds.dtype, device=speed_ds.device)
    anchor_mix = torch.einsum("bk,kd->bd", probs, anchor_dth)
    plan_ds = speed_ds + res_ds
    plan_dth_lane = anchor_mix + res_dth
    plan = from_lane_dtheta(
        torch.stack([plan_ds, plan_dth_lane], dim=-1),
        plan_ds,
        heading_err,
        curvature,
        lane_valid,
    )
    return plan, probs


def assigned_plan(
    index: Tensor,
    residual: Tensor,
    anchor_dth: Tensor,
    speed_ds: Tensor,
    heading_err: Tensor,
    curvature: Tensor,
    lane_valid: Optional[Tensor] = None,
) -> Tensor:
    """WTA 指派计划（只取被分配锚的 residual）→ ``(B,6,2)`` ego 系。"""
    batch = int(index.shape[0])
    residual = residual.to(dtype=speed_ds.dtype, device=speed_ds.device)
    row = torch.arange(batch, device=residual.device)
    picked = residual[row, index.reshape(-1).to(residual.device)]  # (B,6,2)
    ds = speed_ds + picked[..., 0]
    anchor_dth = anchor_dth.to(dtype=speed_ds.dtype, device=speed_ds.device)
    dth_lane = anchor_dth[index] + picked[..., 1]
    return from_lane_dtheta(
        torch.stack([ds, dth_lane], dim=-1), ds, heading_err, curvature, lane_valid
    )


__all__ = [
    "K_DEFAULT",
    "ANCHOR_FEATURE",
    "ANCHOR_FILE_DEFAULT",
    "ANCHOR_SEMANTICS",
    "DEFAULT_ANCHOR_DS",
    "DEFAULT_ANCHOR_DTH_LANE",
    "anchors_from_payload",
    "load_anchor_dictionary",
    "default_anchors",
    "anchor_lane_context",
    "lane_follow_dtheta",
    "to_lane_dtheta",
    "from_lane_dtheta",
    "lane_shape_features",
    "assign_anchors",
    "mixture_probabilities",
    "anchor_mixture",
    "assigned_plan",
]
