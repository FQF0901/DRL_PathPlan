"""Round-36 H1：执行优先骨架——车道锚定横向曲率头 + 路径条件化纵向头 + 外部兼容组合层。

规格出处：``docs/chatgpt5p6sol_review.md`` §二/§三（GPT ``7c0186f``；round-36 journal）。

设计（一图流）
--------------
1. **锚**（lane-anchored）：外部给定目标车道后，数据/环境侧把其中心线采样为 ego t0 系折线
   ``lane_anchor_xy (B,P,2)``（首点 ≈ 自车在车道上的投影点，允许横向偏移；点序沿车道前进）。
   :func:`anchor_reference` 把它重采样为**三个曲率控制站点** ``s_ctrl``
   （``KAPPA_CTRL_FRACTIONS × horizon``，固定横向前瞻 ⇒ 与锚长度解耦）上的参考曲率
   ``kappa_ref (B,3)``，并给出横向偏移/航向误差（自车相对锚线；**只作特征**，不强行把
   路径起点搬到锚线上——见 caveats）。
2. **横向头**：``latent + 几何编码`` → ``κ_near/mid/far`` **残差**（``tanh × kappa_limit``）。
   ``κ(s) = κ_ref(s) + residual_κ(s)``，分段线性插值（C0、控制点处连续；s<s_near 取 κ_near、
   s>s_far 取 κ_far 常数延伸）。末层零初始化 ⇒ **初始 κ(s) 严格 = 锚中心线曲率**（不从零生成）。
3. **纵向头**：输入 = 组合后横向路径曲率 ``κ_ctrl(3)`` + 外源几何 ``long_exog``
   （当前速度/左右边界余量/静态距/动态距/TTC/nav 进度/目标车道末端距，见
   :data:`LONG_EXOG_FEATURE_NAMES`；缺键默认 0 并返回 presence 掩码）→ 6 步 ``ds``
   （``sigmoid × DS_HIGH``，软下界 0 ⇒ ``s`` 单调；零初始化末层 + 偏置 logit(3.5/10)
   ⇒ 初始恒速 3.5 m/步）。
4. **组合层**（外部兼容；:func:`compose_path_plan`）：``ds_k = s_k − s_{k−1}``（``s_0≡0``）、
   ``dθ_k = κ̄_k·ds_k``，其中 ``κ̄_k = (1/ds_k)∫_{s_{k−1}}^{s_k} κ(s) ds``（分段线性精确积分）
   ⇒ ``(B,6,2)`` 的 ``(ds,dθ)`` plan —— **与现 rollout/LQR/环境/评测接口逐位同构**：
   ``plan[:,0] == action_mu``、``LQR set_reference(plan)``、``interpolate`` 6→30 点不改。
   ``traj_xy/traj_theta`` 用 :func:`net.model.arc_step`/``compose_pose`` 逐步重放 plan
   （与 ``_rollout`` 同一口径；段内恒曲率圆弧）。

默认关 / 逐位兼容
------------------
- 本模块**不被** :class:`net.model.DrivingModel` 默认构造（``set_path_head(True)`` 惰性构造，
  构造期保存/恢复 CPU RNG ⇒ 不扰动既有初始化抽样序列）；默认关时模型键集/前向与 HEAD 逐位一致。
- ``forward`` 只在模型启用 path head 时被调用；纯函数（几何/组合）无全局状态、不消耗 RNG。

输入契约（模型启用时；``obs`` 键）
----------------------------------
- ``lane_anchor_xy (B,P,2)``（必需；容忍 ``(B,1,P,2)`` 单例维）——目标车道中心线折线（ego t0 系）；
- ``lane_anchor_mask (B,P)``（可选；默认全 1）；
- ``target_lane_valid (B,)``（可选；缺省 = 锚有效）；
- ``path_left_margin/path_right_margin/path_static_distance/path_dynamic_distance/
  path_dynamic_ttc/path_nav_progress/path_distance_to_target_lane_end (B,)``（可选，缺省 0）；
  当前速度取 ``obs['ego']`` 第 0 维（v, m/s）；``lane_semantic`` 存在时 margin/d2end/valid
  回退到其列 4/5/11/6。

训练/冻结集（H1 manifest，见 ``.slim/deepwork/p1_v2/round36/h1_manifest.json``）
-----------------------------------------------------------------------------
- **可训**：``path_head.``（横向头/纵向头/两个几何特征编码层）+ ``plan_head.fusion.2.``
  （PolicyBackbone 末层，:data:`TRAIN_ALLOWLIST_PREFIXES`）；
- **冻结**：WM(``st_gnn.``)/``encoders.`` 主体/``mem_encoder.`` 主体/旧 ``policy.``/``value.``/
  ``plan_head.moe.``/LQR（LQR 不在模型内）。
- target-lane token dropout：训练开关（默认关；0.2→0.5 斜坡由训练侧调度），实现在
  ``DrivingModel.set_target_lane_dropout``。

Caveats（骨架口径，H2/H3 后续收口）
-----------------------------------
- **曲率空间锚定**，非位置锚定：路径从自车原点/自车航向出发，横向偏移与航向误差只作为
  特征喂头（首版不把路径起点搬到锚线上；offset 的修正由残差头学习）。
- ``long_exog`` 可选键缺省 0 且 presence 掩码随输出返回（防静默数据缺口由 H3 hook 审计）。
- 组合层把 ``ds`` 截断到 ``[0,10]``、``dθ`` 截断到 ``[−0.6,0.6]``（policy 动作界），
  截断掩码 ``ds_clamped/dtheta_clamped`` 随输出返回；截断后 ``κ̄`` 按 ``dθ/ds`` 反算
  （``kappa_seg``），截断前积分值在 ``kappa_seg_raw``。
"""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from net.encoders import H
from net.mem import squeeze_batch_singletons
from net.policy import ACTION_HIGH, ACTION_LOW
from net.spatial import wrap_angle

__all__ = [
    "ANCHOR_FEATURE_NAMES",
    "ANCHOR_FEATURE_NORMALIZATION",
    "DS_HIGH",
    "DS_INIT",
    "DTHETA_HIGH",
    "DTHETA_LOW",
    "DT",
    "KAPPA_CTRL_FRACTIONS",
    "KAPPA_CTRL_S",
    "KAPPA_LIMIT",
    "LAT_HORIZON_M",
    "LONG_EXOG_FEATURE_NAMES",
    "LONG_EXOG_NORMALIZATION",
    "LONG_EXOG_OBS_KEYS",
    "LONG_HEAD_FEATURE_NAMES",
    "LONG_HEAD_NORMALIZATION",
    "NUM_STEPS",
    "TRAIN_ALLOWLIST_PREFIXES",
    "PathExecutionHead",
    "PathGeometryEncoder",
    "anchor_from_obs",
    "anchor_reference",
    "compose_path_plan",
    "kappa_at",
    "long_exog_from_obs",
    "replay_plan",
    "target_lane_valid_from_obs",
]

_EPS = 1e-6
#: 组合/纵向步数（与 WM rollout/plan 契约一致）
NUM_STEPS = 6
#: 单步时长（s）；``ds`` 与速度换算 ``v = ds/dt``
DT = 0.5
#: 横向曲率路径的固定前瞻长度（m）：≈ 3 s × 11.7 m/s；控制站位与之挂钩
LAT_HORIZON_M = 35.0
#: 曲率控制点站位 = ``fractions × horizon`` → (5.25, 17.5, 29.75) m
KAPPA_CTRL_FRACTIONS: tuple[float, float, float] = (0.15, 0.50, 0.85)
KAPPA_CTRL_S: tuple[float, float, float] = tuple(
    round(fraction * LAT_HORIZON_M, 6) for fraction in KAPPA_CTRL_FRACTIONS
)
#: 横向残差限幅（1/m；``tanh × kappa_limit``）
KAPPA_LIMIT = 0.1
#: 纵向头初始速度（m/步；expert ds 中位 ≈3.5 m → 偏置 logit(3.5/10)）
DS_INIT = 3.5
#: 动作上界（与 :data:`net.policy.ACTION_HIGH` 同口径；ds ∈ [0,10] m/步）
DS_HIGH = float(ACTION_HIGH[0])
DTHETA_LOW = float(ACTION_LOW[1])
DTHETA_HIGH = float(ACTION_HIGH[1])

#: 横向头输入的锚几何特征（顺序即列序；归一化由 :data:`ANCHOR_FEATURE_NORMALIZATION` 固定）
ANCHOR_FEATURE_NAMES: tuple[str, ...] = (
    "lateral_offset",
    "heading_error",
    "kappa_ref_near",
    "kappa_ref_mid",
    "kappa_ref_far",
    "anchor_total_s",
    "anchor_valid",
    "target_lane_valid",
)
#: 锚几何特征固定归一化除数（入口口径，与 ``env.obs.lane_semantic`` 同范式）
ANCHOR_FEATURE_NORMALIZATION: tuple[float, ...] = (
    5.0,
    math.pi,
    0.1,
    0.1,
    0.1,
    100.0,
    1.0,
    1.0,
)

#: 纵向头**外源**几何特征（顺序即列序；当前速度取 ``obs['ego']`` 第 0 维）
LONG_EXOG_FEATURE_NAMES: tuple[str, ...] = (
    "speed",
    "left_margin",
    "right_margin",
    "static_distance",
    "dynamic_distance",
    "dynamic_ttc",
    "nav_progress",
    "distance_to_target_lane_end",
)
#: 外源几何固定归一化除数
LONG_EXOG_NORMALIZATION: tuple[float, ...] = (10.0, 25.0, 25.0, 50.0, 50.0, 10.0, 1.0, 200.0)
#: 外源几何的 obs 键（顺序同 :data:`LONG_EXOG_FEATURE_NAMES`；``None`` = 由 ``obs['ego']`` 求速度）
LONG_EXOG_OBS_KEYS: tuple[str | None, ...] = (
    None,
    "path_left_margin",
    "path_right_margin",
    "path_static_distance",
    "path_dynamic_distance",
    "path_dynamic_ttc",
    "path_nav_progress",
    "path_distance_to_target_lane_end",
)

#: 纵向头完整输入 = 组合后横向曲率 κ_ctrl(3) + 外源几何(8)
LONG_HEAD_FEATURE_NAMES: tuple[str, ...] = (
    "kappa_near",
    "kappa_mid",
    "kappa_far",
) + LONG_EXOG_FEATURE_NAMES
LONG_HEAD_NORMALIZATION: tuple[float, ...] = (0.1, 0.1, 0.1) + LONG_EXOG_NORMALIZATION

#: H1 训练 allowlist（manifest 同步声明）：**只有这些前缀参数可训练**，其余全冻。
#: - ``path_head.`` = 横向曲率头 + 纵向路径头 + 两个几何特征编码层；
#: - ``plan_head.fusion.2.`` = PolicyBackbone（plan head fusion）末层；其余 fusion/norm/ego_next
#:   与 MoE/旧 PolicyHead/ValueHead/WM/Encoder 主体全部冻结（§三）。
TRAIN_ALLOWLIST_PREFIXES: tuple[str, ...] = (
    "path_head.",
    "plan_head.fusion.2.",
)

_GEOM_EMB_DIM = 64


# ---------------------------------------------------------------------- 几何纯函数
def _interp_nodes(s_nodes: Tensor, values: Tensor, s_query: Tensor) -> Tensor:
    """逐行线性插值（常数延伸）：``s_nodes (B,P)`` 非降、``values (B,P)``、``s_query (B,Q)``。

    ``s_query`` 先按行 clamp 到 ``[s_nodes[0], s_nodes[-1]]``（超出端点 = 端点值常数延伸）。
    P 很小（锚折线 ≤ 64 点），逐区间掩码实现，避免对 torch 版本的依赖。
    """
    batch, points = int(s_nodes.shape[0]), int(s_nodes.shape[1])
    lo = s_nodes[:, :1]
    hi = s_nodes[:, -1:]
    s = torch.clamp(s_query, min=lo, max=hi)
    out = torch.zeros_like(s_query)
    prev = s_nodes[:, :1]
    for index in range(points - 1):
        nxt = s_nodes[:, index + 1 : index + 2]
        span = (nxt - prev).clamp_min(_EPS)
        t = ((s - prev) / span).clamp(0.0, 1.0)
        inside = (s >= prev) & (s <= nxt)
        segment = values[:, index : index + 1] * (1.0 - t) + values[:, index + 1 : index + 2] * t
        out = torch.where(inside, segment, out)
        prev = nxt
    out = torch.where(s >= hi, values[:, -1:], out)
    return out


def anchor_reference(
    anchor_xy: Tensor,
    anchor_mask: Tensor | None = None,
    *,
    horizon: float = LAT_HORIZON_M,
) -> dict[str, Tensor]:
    """目标车道中心线折线 → 锚参考量（纯函数；无 NaN、无 RNG）。

    Args:
        anchor_xy: ``(B,P,2)`` 车道中心线采样点（ego t0 系：x 前向 / y 左向；点序沿车道前进；
            首点 ≈ 自车在车道上的投影点，允许横向偏移）。
        anchor_mask: ``(B,P)`` 1=有效点（可选；默认全 1）。有效点 <3 或总长 <1 m 的行
            ``valid=0``（``kappa_ref`` 归零 = 直锚兜底；**不报错**，由调用方决定是否 fail-fast）。
        horizon: 控制站位前瞻（m）；``s_ctrl = KAPPA_CTRL_FRACTIONS × horizon``（固定）。

    Returns:
        dict（全部 (B,·) / float32）：
        - ``kappa_ref (B,3)``：三控制站位的锚曲率（顶点中心差分 + 线性插值 + 端点常数延伸）；
        - ``s_ctrl (B,3)``：控制站位 [m]（常数，供组合层积分用）；
        - ``total_s (B,)``、``valid (B,)``、``kappa_vertex (B,P)``（诊断）、``s_nodes (B,P)``；
        - ``lateral_offset (B,)`` = ``-anchor_xy[:,0,1]``（自车相对锚线横向偏移；lane 系口径
          正 = 自车偏左）；``heading_error (B,)`` = ``-θ_anchor(0)``（正 = 自车航向偏左）。
          无效行上上述量全部置 0（``valid`` 门控）。
    """
    if not torch.is_tensor(anchor_xy):
        raise TypeError(f"anchor_xy 必须是 Tensor，收到 {type(anchor_xy).__name__}")
    xy = anchor_xy.float()
    if xy.ndim != 3 or int(xy.shape[-1]) != 2:
        raise ValueError(f"anchor_xy 形状应为 (B,P,2)，收到 {tuple(xy.shape)}")
    batch, points, _ = (int(size) for size in xy.shape)
    if points < 2:
        raise ValueError(f"anchor_xy 至少需要 2 个点，收到 P={points}")
    device, dtype = xy.device, xy.dtype
    if anchor_mask is None:
        mask = torch.ones((batch, points), dtype=dtype, device=device)
    else:
        mask = anchor_mask.float()
        if mask.ndim != 2 or tuple(mask.shape) != (batch, points):
            raise ValueError(f"anchor_mask 形状应为 {(batch, points)}，收到 {tuple(mask.shape)}")

    segment = xy[:, 1:] - xy[:, :-1]
    segment_len = segment.norm(dim=-1)
    segment_valid = (mask[:, 1:] > 0.5) & (mask[:, :-1] > 0.5)
    segment_len = torch.where(segment_valid, segment_len, torch.zeros_like(segment_len))
    s_nodes = torch.cat(
        [torch.zeros((batch, 1), dtype=dtype, device=device), torch.cumsum(segment_len, dim=1)],
        dim=1,
    )
    theta_segment = torch.atan2(segment[..., 1], segment[..., 0])
    if points >= 3:
        pair_len = 0.5 * (segment_len[:, 1:] + segment_len[:, :-1])
        pair_valid = segment_valid[:, 1:] & segment_valid[:, :-1] & (pair_len > _EPS)
        pair_dtheta = wrap_angle(theta_segment[:, 1:] - theta_segment[:, :-1])
        kappa_inner = torch.where(
            pair_valid, pair_dtheta / pair_len.clamp_min(_EPS), torch.zeros_like(pair_len)
        )
        kappa_vertex = torch.cat(
            [kappa_inner[:, :1], kappa_inner, kappa_inner[:, -1:]], dim=1
        )
    else:
        kappa_vertex = torch.zeros_like(s_nodes)

    total_s = s_nodes[:, -1]
    valid = ((mask.sum(dim=1) >= 3) & (total_s >= 1.0)).to(dtype)
    s_ctrl = (
        torch.as_tensor(KAPPA_CTRL_FRACTIONS, dtype=dtype, device=device)
        .mul(float(horizon))
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    kappa_ref = _interp_nodes(s_nodes, kappa_vertex, s_ctrl) * valid.unsqueeze(-1)
    lateral_offset = -xy[:, 0, 1] * valid
    heading_error = -theta_segment[:, 0] * valid
    return {
        "kappa_ref": kappa_ref,
        "s_ctrl": s_ctrl,
        "total_s": total_s * valid,
        "valid": valid,
        "kappa_vertex": kappa_vertex,
        "s_nodes": s_nodes,
        "lateral_offset": lateral_offset,
        "heading_error": heading_error,
    }


def _check_ctrl_stations(s_ctrl: Tensor) -> None:
    """控制站位校验：``(B,3)``、升序、最小间隔 ≥1e-3（分块积分前提）。"""
    if s_ctrl.ndim != 2 or int(s_ctrl.shape[-1]) != 3:
        raise ValueError(f"s_ctrl 形状应为 (B,3)，收到 {tuple(s_ctrl.shape)}")
    gaps = s_ctrl[:, 1:] - s_ctrl[:, :-1]
    if not bool((gaps > 1e-3).all()):
        raise ValueError(f"s_ctrl 必须严格升序且间隔 ≥1e-3，收到 {s_ctrl.detach().tolist()[:2]}")


def kappa_at(kappa_ctrl: Tensor, s_ctrl: Tensor, s_query: Tensor) -> Tensor:
    """分段线性 κ(s)：``s<s0`` 取 κ0、``s>s2`` 取 κ2（常数延伸）。

    ``kappa_ctrl (B,3)``、``s_ctrl (B,3)``（升序）、``s_query`` 任意尾维（钳制到 [s0,s2] 后求值）。
    """
    s = torch.clamp(s_query, min=s_ctrl[..., 0:1], max=s_ctrl[..., 2:3])
    k0, k1, k2 = kappa_ctrl.unbind(-1)
    s0, s1, s2 = s_ctrl.unbind(-1)
    t01 = ((s - s0.unsqueeze(-1)) / (s1 - s0).clamp_min(_EPS).unsqueeze(-1)).clamp(0.0, 1.0)
    t12 = ((s - s1.unsqueeze(-1)) / (s2 - s1).clamp_min(_EPS).unsqueeze(-1)).clamp(0.0, 1.0)
    kappa01 = k0.unsqueeze(-1) + (k1 - k0).unsqueeze(-1) * t01
    kappa12 = k1.unsqueeze(-1) + (k2 - k1).unsqueeze(-1) * t12
    return torch.where(s <= s1.unsqueeze(-1), kappa01, kappa12)


def _integrate_kappa(kappa_ctrl: Tensor, s_ctrl: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """``∫_lo^hi κ(s) ds``（分段线性**精确**；要求 ``lo ≤ hi``）。

    口径 = :func:`kappa_at`：``s<s0`` 常数 κ0、``s>s2`` 常数 κ2、``[s0,s2]`` 分段线性。
    实现 = 三段拼装（下常数段 + 内部在 ``s1`` 切分的两段梯形 + 上常数段）；
    内部子区间内 κ 严格线性 ⇒ 梯形=精确积分。
    """
    s0, s1, s2 = (station.unsqueeze(-1) for station in s_ctrl.unbind(-1))
    k0, k1, k2 = (value.unsqueeze(-1) for value in kappa_ctrl.unbind(-1))
    below = (torch.minimum(hi, s0) - lo).clamp_min(0.0) * k0
    above = (hi - torch.maximum(lo, s2)).clamp_min(0.0) * k2
    inner_lo = torch.maximum(lo, s0)
    inner_hi = torch.minimum(hi, s2)
    mid = torch.minimum(torch.maximum(s1, inner_lo), inner_hi)

    def _trapezoid(k_a: Tensor, k_b: Tensor, s_a: Tensor, s_b: Tensor, start: Tensor, end: Tensor) -> Tensor:
        span = (s_b - s_a).clamp_min(_EPS)
        value_start = k_a + (k_b - k_a) * (start - s_a) / span
        value_end = k_a + (k_b - k_a) * (end - s_a) / span
        return 0.5 * (value_start + value_end) * (end - start).clamp_min(0.0)

    first = _trapezoid(k0, k1, s0, s1, inner_lo, mid)
    second = _trapezoid(k1, k2, s1, s2, mid, inner_hi)
    return below + first + second + above


def replay_plan(plan: Tensor, theta0: Tensor | None = None) -> dict[str, Tensor]:
    """``(B,K,2)`` 的 ``(ds,dθ)`` plan → 位姿链（与 ``_rollout`` 同一口径：段内恒曲率圆弧）。

    ``net.model.arc_step``/``compose_pose`` 是运动学唯一实现出处（函数内导入避免循环依赖）。
    返回 ``traj_xy (B,K,2)``、``traj_theta (B,K)``（t0 自车系；``theta0`` 缺省 0 = 自车航向）。
    """
    from net.model import arc_step, compose_pose

    if plan.ndim != 3 or int(plan.shape[-1]) != 2:
        raise ValueError(f"plan 形状应为 (B,K,2)，收到 {tuple(plan.shape)}")
    batch, steps = int(plan.shape[0]), int(plan.shape[1])
    dtype, device = plan.dtype, plan.device
    pose = torch.zeros((batch, 3), dtype=dtype, device=device)
    if theta0 is not None:
        if torch.is_tensor(theta0):
            pose[:, 2] = theta0.reshape(batch)
        else:
            pose[:, 2] = float(theta0)
    xy: list[Tensor] = []
    theta: list[Tensor] = []
    for index in range(steps):
        ds, dtheta = plan[:, index, 0], plan[:, index, 1]
        dx, dy = arc_step(ds, dtheta)
        pose = compose_pose(pose, dx, dy, dtheta)
        xy.append(pose[:, :2])
        theta.append(pose[:, 2])
    return {"traj_xy": torch.stack(xy, dim=1), "traj_theta": torch.stack(theta, dim=1)}


def compose_path_plan(
    kappa_ctrl: Tensor,
    s_ctrl: Tensor,
    ds: Tensor,
    *,
    dt: float = DT,
) -> dict[str, Tensor]:
    """**外部兼容组合层**：横向 κ 控制点 + 纵向逐段 ``ds`` → LQR/环境/评测同构的 6 步 plan。

    契约（与现 rollout/LQR 消费一致，冻结）：
    - ``plan (B,K,2)``，``plan[:,k] = (ds_k, dθ_k)``，``ds_k = s_k − s_{k−1}``（``s_0 ≡ 0``）；
    - ``dθ_k = κ̄_k·ds_k``，``κ̄_k = (1/ds_k)∫ κ(s)ds``（分段线性精确积分）——**段内恒曲率**，
      故用 :func:`replay_plan`（``arc_step``/``compose_pose``）重放 plan 的端点与 κ(s)
      连续积分端点一致（同 ``_rollout`` 口径）；
    - ``action_mu = plan[:,0]``（rollout step0 / cheap path 同一值 ⇒ ``plan[:,0]==action_mu``）；
    - ``ds`` 截断到 ``[0,10]``、``dθ`` 截断到 ``[−0.6,0.6]``（policy 动作界），截断掩码返回；
    - ``progress_s (B,K)`` = 累计沿路径进度 ``s_k``（截断后口径）。

    Returns: dict（``plan/action_mu/ds/dtheta/progress_s/traj_xy/traj_theta/kappa_seg/
    kappa_seg_raw/ds_clamped/dtheta_clamped/s_clamped``）。
    """
    if kappa_ctrl.ndim != 2 or int(kappa_ctrl.shape[-1]) != 3:
        raise ValueError(f"kappa_ctrl 形状应为 (B,3)，收到 {tuple(kappa_ctrl.shape)}")
    if ds.ndim != 2 or int(ds.shape[-1]) < 1:
        raise ValueError(f"ds 形状应为 (B,K)（K≥1），收到 {tuple(ds.shape)}")
    if int(kappa_ctrl.shape[0]) != int(ds.shape[0]):
        raise ValueError(
            f"kappa_ctrl B={int(kappa_ctrl.shape[0])} != ds B={int(ds.shape[0])}"
        )
    _check_ctrl_stations(s_ctrl)
    if int(s_ctrl.shape[0]) != int(ds.shape[0]):
        raise ValueError(f"s_ctrl B={int(s_ctrl.shape[0])} != ds B={int(ds.shape[0])}")

    ds_raw = ds.float()
    ds_clamped = torch.where(
        torch.isfinite(ds_raw), ds_raw.clamp(0.0, DS_HIGH), torch.zeros_like(ds_raw)
    )
    ds_clamped_mask = (ds_raw < 0.0) | (ds_raw > DS_HIGH) | ~torch.isfinite(ds_raw)
    s_cum = torch.cat(
        [torch.zeros((int(ds.shape[0]), 1), dtype=ds.dtype, device=ds.device), torch.cumsum(ds_clamped, dim=1)],
        dim=1,
    )
    lo, hi = s_cum[:, :-1], s_cum[:, 1:]
    dtheta_raw = _integrate_kappa(kappa_ctrl.float(), s_ctrl.float(), lo, hi)
    dtheta = dtheta_raw.clamp(DTHETA_LOW, DTHETA_HIGH)
    dtheta_clamped_mask = (dtheta_raw < DTHETA_LOW) | (dtheta_raw > DTHETA_HIGH)
    kappa_seg = torch.where(
        ds_clamped > 1e-3, dtheta / ds_clamped.clamp_min(_EPS), torch.zeros_like(dtheta)
    )
    kappa_seg_raw = torch.where(
        ds_clamped > 1e-3, dtheta_raw / ds_clamped.clamp_min(_EPS), torch.zeros_like(dtheta_raw)
    )
    plan = torch.stack([ds_clamped, dtheta], dim=-1)
    replay = replay_plan(plan)
    return {
        "plan": plan,
        "action_mu": plan[:, 0, :],
        "ds": ds_clamped,
        "dtheta": dtheta,
        "progress_s": s_cum[:, 1:],
        "speed": ds_clamped / float(dt),
        "kappa_seg": kappa_seg,
        "kappa_seg_raw": kappa_seg_raw,
        "ds_clamped": ds_clamped_mask,
        "dtheta_clamped": dtheta_clamped_mask,
        "s_clamped": (lo, hi),
        "traj_xy": replay["traj_xy"],
        "traj_theta": replay["traj_theta"],
    }


# ---------------------------------------------------------------------- obs → 输入
def anchor_from_obs(obs: Mapping[str, Tensor]) -> tuple[Tensor, Tensor | None]:
    """``obs`` → ``(lane_anchor_xy (B,P,2), lane_anchor_mask (B,P)|None)``；缺键 fail-fast。"""
    if not isinstance(obs, Mapping):
        raise TypeError(f"obs 必须是映射，收到 {type(obs).__name__}")
    value = obs.get("lane_anchor_xy")
    if not torch.is_tensor(value):
        raise ValueError(
            "path head 启用时 obs 必须提供 'lane_anchor_xy'（目标车道中心线折线，(B,P,2)，"
            "ego t0 系；首点 ≈ 自车在车道上的投影点）"
        )
    xy = squeeze_batch_singletons(value.float(), 3, "lane_anchor_xy")
    if int(xy.shape[-1]) != 2:
        raise ValueError(f"lane_anchor_xy 形状应为 (B,P,2)，收到 {tuple(xy.shape)}")
    mask_value = obs.get("lane_anchor_mask")
    mask: Tensor | None = None
    if torch.is_tensor(mask_value):
        mask = squeeze_batch_singletons(mask_value.float(), 2, "lane_anchor_mask")
        if tuple(mask.shape) != tuple(xy.shape[:2]):
            raise ValueError(
                f"lane_anchor_mask 形状应为 {tuple(xy.shape[:2])}，收到 {tuple(mask.shape)}"
            )
    return xy, mask


def target_lane_valid_from_obs(
    obs: Mapping[str, Tensor], *, batch: int, device: torch.device | None = None
) -> Tensor | None:
    """``obs['target_lane_valid']``（或 ``lane_semantic`` 列 6）→ ``(B,)``；都缺 → ``None``。

    ``None`` 时 :class:`PathExecutionHead` 以锚有效位兜底（``anchor_valid``）。
    """
    value = obs.get("target_lane_valid")
    if torch.is_tensor(value):
        vector = squeeze_batch_singletons(value.float(), 1, "target_lane_valid")
        if int(vector.shape[0]) != batch:
            raise ValueError(f"target_lane_valid B={int(vector.shape[0])} != {batch}")
        return vector.to(device=device) if device is not None else vector
    semantic = obs.get("lane_semantic")
    if torch.is_tensor(semantic):
        sem = squeeze_batch_singletons(semantic.float(), 2, "lane_semantic")
        if int(sem.shape[0]) == batch and int(sem.shape[1]) >= 12:
            column = sem[:, 6]
            return column.to(device=device) if device is not None else column
    return None


def _vector_from_obs(
    obs: Mapping[str, Tensor],
    key: str,
    *,
    batch: int,
    device: torch.device,
    default: float = 0.0,
) -> tuple[Tensor, float]:
    """可选 ``(B,)``（或 ``(B,1)``）标量通道 → ``(values (B,), present)``。"""
    value = obs.get(key)
    if not torch.is_tensor(value):
        return torch.full((batch,), float(default), dtype=torch.float32, device=device), 0.0
    vector = squeeze_batch_singletons(value.float(), 1, key)
    if int(vector.shape[0]) != batch:
        raise ValueError(f"obs[{key!r}] B={int(vector.shape[0])} != {batch}")
    return vector.to(device=device), 1.0


def long_exog_from_obs(
    obs: Mapping[str, Tensor], *, batch: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """``obs`` → 纵向头外源几何 ``(long_exog (B,8), present (B,8))``（缺键默认 0，记录掩码）。

    列序 = :data:`LONG_EXOG_FEATURE_NAMES`。速度优先取 ``obs['ego']`` 第 0 维（v, m/s），
    否则取 ``path_speed``；margin/d2end/target 有效位在缺键时回退 ``lane_semantic``
    （列 4/5/11/6，semantic 模式数据）；仍缺 → 0。
    """
    names = LONG_EXOG_FEATURE_NAMES
    values: list[Tensor] = []
    present = torch.zeros((batch, len(names)), dtype=torch.float32, device=device)

    semantic = obs.get("lane_semantic")
    semantic_ok = torch.is_tensor(semantic)
    semantic_batch_ok = False
    if semantic_ok:
        sem = squeeze_batch_singletons(semantic.float(), 2, "lane_semantic")
        semantic_batch_ok = sem.ndim == 2 and int(sem.shape[0]) == batch and int(sem.shape[1]) >= 12
        semantic = sem if semantic_batch_ok else None

    def _push(index: int, vector: Tensor, has: float) -> None:
        values.append(vector)
        present[:, index] = float(has)

    # 0) speed：obs['ego'] 第 0 维（当前帧；容忍 (B,8)/(B,1,8)/(B,T,8)）
    ego = obs.get("ego")
    speed_ok = False
    if torch.is_tensor(ego):
        ego_t = ego.float()
        while ego_t.ndim > 2 and int(ego_t.shape[1]) == 1:
            ego_t = ego_t.squeeze(1)
        if ego_t.ndim == 3:
            ego_t = ego_t[:, -1, :]
        if ego_t.ndim == 2 and tuple(ego_t.shape) == (batch, 8):
            _push(0, ego_t[:, 0].to(device=device), 1.0)
            speed_ok = True
    if not speed_ok:
        value, has = _vector_from_obs(obs, "path_speed", batch=batch, device=device)
        _push(0, value, has)

    # 1) left_margin / 2) right_margin
    left, left_has = _vector_from_obs(obs, str(LONG_EXOG_OBS_KEYS[1]), batch=batch, device=device)
    if left_has == 0.0 and semantic_batch_ok:
        left, left_has = semantic[:, 4].to(device=device), 1.0
    _push(1, left, left_has)
    right, right_has = _vector_from_obs(obs, str(LONG_EXOG_OBS_KEYS[2]), batch=batch, device=device)
    if right_has == 0.0 and semantic_batch_ok:
        right, right_has = semantic[:, 5].to(device=device), 1.0
    _push(2, right, right_has)

    # 3) static / 4) dynamic / 5) ttc
    for index in (3, 4, 5):
        value, has = _vector_from_obs(
            obs, str(LONG_EXOG_OBS_KEYS[index]), batch=batch, device=device
        )
        _push(index, value, has)

    # 6) nav_progress：显式键 → obs['nav'] 末维（route_completion）→ 0
    progress, progress_has = _vector_from_obs(
        obs, str(LONG_EXOG_OBS_KEYS[6]), batch=batch, device=device
    )
    if progress_has == 0.0:
        nav = obs.get("nav")
        if torch.is_tensor(nav):
            nav_t = nav.float()
            while nav_t.ndim > 2 and int(nav_t.shape[1]) == 1:
                nav_t = nav_t.squeeze(1)
            if nav_t.ndim == 2 and int(nav_t.shape[0]) == batch and int(nav_t.shape[1]) >= 11:
                progress, progress_has = nav_t[:, -1].to(device=device), 1.0
    _push(6, progress, progress_has)

    # 7) distance_to_target_lane_end
    distance, distance_has = _vector_from_obs(
        obs, str(LONG_EXOG_OBS_KEYS[7]), batch=batch, device=device
    )
    if distance_has == 0.0 and semantic_batch_ok:
        distance, distance_has = semantic[:, 11].to(device=device), 1.0
    _push(7, distance, distance_has)

    return torch.stack(values, dim=-1).to(device=device, dtype=torch.float32), present


# ---------------------------------------------------------------------- 模块
class PathGeometryEncoder(nn.Module):
    """几何特征编码层：raw 几何 → 隐向量（固定归一化 buffer + 最小 2 层 MLP）。"""

    def __init__(self, in_dim: int, out_dim: int = _GEOM_EMB_DIM, normalization: tuple[float, ...] | None = None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if normalization is None:
            normalization = (1.0,) * self.in_dim
        if len(normalization) != self.in_dim:
            raise ValueError(f"normalization 长度 {len(normalization)} != in_dim {self.in_dim}")
        self.register_buffer(
            "normalization", torch.tensor(normalization, dtype=torch.float32), persistent=True
        )
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, feats: Tensor) -> Tensor:
        if feats.ndim != 2 or int(feats.shape[-1]) != self.in_dim:
            raise ValueError(f"几何特征形状应为 (B,{self.in_dim})，收到 {tuple(feats.shape)}")
        normalized = feats / self.normalization.clamp_min(_EPS)
        return self.net(normalized)


class PathExecutionHead(nn.Module):
    """执行优先骨架：车道锚定横向曲率头 + 路径条件化纵向头 + 外部兼容组合层。

    **默认不构造**（``DrivingModel.set_path_head``）。初始化口径：
    - 横向末层零初始化 ⇒ 初始 ``κ(s) = κ_ref(s)``（= 锚中心线；直锚 ⇒ 直行）；
    - 纵向末层零权重 + 偏置 ``logit(DS_INIT/DS_HIGH)`` ⇒ 初始恒速 :data:`DS_INIT`（3.5 m/步）。
    """

    def __init__(
        self,
        hidden: int = H,
        *,
        kappa_limit: float = KAPPA_LIMIT,
        geom_emb: int = _GEOM_EMB_DIM,
        ds_init: float = DS_INIT,
        horizon: float = LAT_HORIZON_M,
        num_steps: int = NUM_STEPS,
        dt: float = DT,
    ):
        super().__init__()
        hidden = int(hidden)
        if hidden <= 0:
            raise ValueError(f"hidden 必须为正，收到 {hidden}")
        if int(num_steps) != NUM_STEPS:
            raise ValueError(
                f"num_steps 必须 = {NUM_STEPS}（LQR/rollout 6 步契约），收到 {num_steps}"
            )
        if not 0.0 < float(ds_init) < DS_HIGH:
            raise ValueError(f"ds_init 必须在 (0,{DS_HIGH})，收到 {ds_init}")
        self.hidden = hidden
        self.kappa_limit = float(kappa_limit)
        self.horizon = float(horizon)
        self.num_steps = int(num_steps)
        self.dt = float(dt)
        self.lateral_geom = PathGeometryEncoder(
            len(ANCHOR_FEATURE_NAMES), geom_emb, ANCHOR_FEATURE_NORMALIZATION
        )
        self.longitudinal_geom = PathGeometryEncoder(
            len(LONG_HEAD_FEATURE_NAMES), geom_emb, LONG_HEAD_NORMALIZATION
        )
        self.lateral = nn.Sequential(
            nn.Linear(hidden + int(geom_emb), hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.lateral[-1].weight)
        nn.init.zeros_(self.lateral[-1].bias)
        self.longitudinal = nn.Sequential(
            nn.Linear(hidden + int(geom_emb), max(8, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(8, hidden // 2), self.num_steps),
        )
        nn.init.zeros_(self.longitudinal[-1].weight)
        nn.init.constant_(
            self.longitudinal[-1].bias,
            math.log(float(ds_init) / (DS_HIGH - float(ds_init))),
        )

    def parameter_names(self) -> tuple[str, ...]:
        """本模块参数/buffer 名（供 ckpt allowlist/冻结声明核对）。"""
        names = [name for name, _ in self.named_parameters()]
        names += [name for name, _ in self.named_buffers()]
        return tuple(sorted(names))

    def forward(
        self,
        latent: Tensor,
        *,
        anchor_xy: Tensor,
        anchor_mask: Tensor | None = None,
        long_exog: Tensor | None = None,
        target_lane_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """``latent (B,H)`` + 锚折线 + 外源几何 → 执行优先输出 dict（见模块 docstring）。

        返回键（全部 Tensor）：``plan/action_mu/traj_xy/traj_theta/ds/dtheta/progress_s/
        speed/kappa_seg/kappa_seg_raw/ds_clamped/dtheta_clamped/kappa_ctrl/kappa_residual/
        kappa_ref/s_ctrl/anchor_lateral_offset/anchor_heading_error/anchor_total_s/
        anchor_valid/target_lane_valid/anchor_features/long_features``。
        重复调用（同输入）逐位一致（无 RNG/全局状态）。
        """
        if latent.ndim != 2 or int(latent.shape[-1]) != self.hidden:
            raise ValueError(f"latent 形状应为 (B,{self.hidden})，收到 {tuple(latent.shape)}")
        batch = int(latent.shape[0])
        anchor_xy = anchor_xy.to(device=latent.device)
        if anchor_mask is not None:
            anchor_mask = anchor_mask.to(device=latent.device)
        anchor = anchor_reference(anchor_xy, anchor_mask, horizon=self.horizon)
        if int(anchor["valid"].shape[0]) != batch:
            raise ValueError(
                f"lane_anchor_xy B={int(anchor['valid'].shape[0])} != latent B={batch}"
            )
        if long_exog is None:
            long_exog = torch.zeros(
                (batch, len(LONG_EXOG_FEATURE_NAMES)), dtype=latent.dtype, device=latent.device
            )
        else:
            long_exog = long_exog.float()
            if tuple(long_exog.shape) != (batch, len(LONG_EXOG_FEATURE_NAMES)):
                raise ValueError(
                    f"long_exog 形状应为 {(batch, len(LONG_EXOG_FEATURE_NAMES))}，"
                    f"收到 {tuple(long_exog.shape)}"
                )
            long_exog = long_exog.to(device=latent.device)
        if target_lane_valid is None:
            target_lane_valid = anchor["valid"]
        else:
            tlv = squeeze_batch_singletons(target_lane_valid.float(), 1, "target_lane_valid")
            if int(tlv.shape[0]) != batch:
                raise ValueError(f"target_lane_valid B={int(tlv.shape[0])} != latent B={batch}")
            target_lane_valid = tlv.to(device=latent.device)

        anchor_features = torch.stack(
            [
                anchor["lateral_offset"],
                anchor["heading_error"],
                anchor["kappa_ref"][:, 0],
                anchor["kappa_ref"][:, 1],
                anchor["kappa_ref"][:, 2],
                anchor["total_s"],
                anchor["valid"],
                target_lane_valid,
            ],
            dim=-1,
        ).to(dtype=latent.dtype)
        lateral_input = torch.cat([latent, self.lateral_geom(anchor_features)], dim=-1)
        residual = torch.tanh(self.lateral(lateral_input)) * self.kappa_limit
        kappa_ctrl = anchor["kappa_ref"].to(dtype=latent.dtype) + residual

        long_features = torch.cat([kappa_ctrl, long_exog.to(dtype=latent.dtype)], dim=-1)
        long_input = torch.cat([latent, self.longitudinal_geom(long_features)], dim=-1)
        ds = DS_HIGH * torch.sigmoid(self.longitudinal(long_input))

        composed = compose_path_plan(kappa_ctrl, anchor["s_ctrl"], ds, dt=self.dt)
        output: dict[str, Tensor] = {
            "plan": composed["plan"],
            "action_mu": composed["action_mu"],
            "traj_xy": composed["traj_xy"],
            "traj_theta": composed["traj_theta"],
            "ds": composed["ds"],
            "dtheta": composed["dtheta"],
            "progress_s": composed["progress_s"],
            "speed": composed["speed"],
            "kappa_seg": composed["kappa_seg"],
            "kappa_seg_raw": composed["kappa_seg_raw"],
            "ds_clamped": composed["ds_clamped"],
            "dtheta_clamped": composed["dtheta_clamped"],
            "kappa_ctrl": kappa_ctrl,
            "kappa_residual": residual,
            "kappa_ref": anchor["kappa_ref"],
            "s_ctrl": anchor["s_ctrl"],
            "anchor_lateral_offset": anchor["lateral_offset"],
            "anchor_heading_error": anchor["heading_error"],
            "anchor_total_s": anchor["total_s"],
            "anchor_valid": anchor["valid"],
            "target_lane_valid": target_lane_valid,
            "anchor_features": anchor_features,
            "long_features": long_features,
        }
        return output
