"""OD/LD 显式 anchor + ego-motion warp（R36 L-D 最小实现；R37 M1 接入 ``net/model.py`` 主路径）。

依据：GPT ``7dcf6bd`` §四.3/§五 阶段 1（坐标层先行）+ R33-B 已验证口径（``net/model.py``
``set_ego_warp`` / ``DrivingModel._ego_warp_pose_k``）。本模块**只含纯函数/无状态**：导入无
副作用、无全局状态。R37 M1 集成（GPT ``66497a2`` §三.3/§三.4）后：``DrivingModel`` 的
``set_ego_warp_mode("analytic")`` 经 :func:`assemble_pose_k`（``w2`` + :func:`od_velocity_t0`
场景系速度）接管 rollout ``pose_k``；``off``（默认）经 :func:`assemble_pose_k`（``off``）逐位
等于旧内联分支；5 项 hook / 分项读数由模型侧 ``set_warp_hooks`` / ``set_warp_readout`` 门控。

语义（与 R33-B **逐位**一致；等价性单测见 ``.slim/deepwork/p1_v2/round36/d1_ego_warp_tests.py``）
----------------------------------------------------------------------------------------
把 t0 帧（ego_t0 系）表达的张量重表达进 ``ego_k`` 系（k 步后的自车系），``T_k = (Δp_k, Δθ_k)``
= t0 帧下 ego 的累积位姿：

- **静态 LD anchor 逆变换**：``p' = R(−Δθ)(p − Δp)``、``ψ' = ψ − Δθ``（``R`` = 2×2 旋转矩阵）；
- **OD anchor 先恒速推进再同一逆变换**：``p' = T_k^{-1}(p + v·(k·dt))``（GPT §四.3 的 ``v_OD``）。
  **``v`` 必须是对象在 t0（场景）系的速度**——即 ``od`` 通道的**相对速度 + ego t0 速度**（见
  :func:`od_velocity_t0`）：warp 把 t0 系当作**静态场景**（LD 静态锚成立的前提），因此节点自身
  运动要按世界速度推进；直接用通道相对速度会把 ego 自身位移重复扣一次（本仓 round34 录制实测
  同速前车系统性偏移 ≈ ``|Δp|``：P50 0.98 m / P90 1.95 m，正确口径 P50 0.05 m / P90 0.43 m）。
  注：``mode="off"``（生产路径）把 t0 系当作**随体（body）坐标系**消费，那里 ``v_rel`` 先验才是
  正确的漂移模型（同速对象保持随体坐标），两种口径不可混用；
- **整图等价**：同一刚体变换施加于 ego/OD/LD 全部节点 ⇒ ego 节点恒 ``(0,0,0)``；任意两节点
  距离不变、相对航向不变、ego↔节点相对位移按 ``R(−Δθ)`` 旋转（round33 ``b_warp_compat.py``
  自洽核验同口径）。

anchor schema（``AnchorBatch``；**坐标系 = ego t0 系，x 前 / y 左，单位 m / m·s⁻¹ / rad**）
---------------------------------------------------------------------------------------
==== =========== ========================================================== =======================================
字段    形状         语义                                                       既有契约映射
==== =========== ========================================================== =======================================
xy   (B,N,2)     位置 [m]                                                   ``FrameEncoding.pose[..., :2]``；
                                                                           OD 预测空间前 2 维（``od_state_from_features``）
theta (B,N)      航向 [rad]（相对 ego t0 航向，已 wrap）                      ``pose[..., 2]``（OD = ``atan2(sinθ, cosθ)``；
                                                                           LD = ``heading_rel`` 原始 7 维第 2 维）
vel  (B,N,2)     **t0 场景系速度** [m/s]（OD 用；静态 LD 恒 0）              ``od_feat[..., 2:4]``（通道给的是
                 = 通道相对速度 + ego t0 速度（od_velocity_t0）             **相对速度**，需 + ego 速度换算）
size (B,N,2)     (L, W) [m]（LD 无尺寸 ⇒ 恒 0）                           ``od_feat[..., 6:8]``（``L, W``）
live (B,N)       0/1 有效掩码                                                OD = ``od_mask ∧ od_presence``；
                                                                           LD = ``ld_mask``
slot_id (B,N) int64 OD = track id（-1 = 空槽）；LD 恒 -1（固定几何槽位，   ``obs['od_id']`` / companion 数组
                     无稳定身份）
==== =========== ========================================================== =======================================

字段来源（现网 obs 通道）：OD = ``env/obs/od.py`` 9 维 ``[dx,dy,vx,vy,cosθ,sinθ,L,W,type_id]``；
LD = ``env/obs/ld.py`` 7 维 ``[dx,dy,heading_rel,curvature,speed_limit,left_line_type,right_line_type]``。
**注意区分** 已退役的 K-anchor plan 形状锚 ``(ds,dθ)``（历史快照
``tools/legacy/k_anchor/anchor.py``；动作形状锚，非几何 anchor，不属本 schema）。
路点/车道锚的 ``type_id``、``curvature``、线型等语义字段**不进**本
schema（保持坐标层最小：位姿 + 速度 + 尺寸 + 掩码 + 身份）。

5 项 hook（``WarpHooks``；**默认关 + bit-exact**）
----------------------------------------------
① ``anchor_pre``：warp 前 anchor（t0 帧）；② ``anchor_analytic``：解析 warp 后 anchor（ego_k 帧）；
③ ``anchor_actual_next``：**真实下一帧** anchor（ego_k 帧；由调用方提供）；④ ``residual_learned``：
learned 残差（结构化 WM 输出；当前无 ⇒ ``None``）；⑤ ``anchor_final``：最终预测 anchor
（= ② + ④；④ 缺省时 = ②）。``enabled=False``（默认）时 :meth:`WarpHooks.capture` 立即返回
``None``：零张量操作、不消耗 RNG、不改写输入。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
from torch import Tensor

from net.spatial import wrap_angle

__all__ = [
    "ANCHOR_SCHEMA_VERSION",
    "FIELD_MAP",
    "AnchorBatch",
    "WarpHookRecord",
    "WarpHooks",
    "WARP_MODES",
    "advance_od_pose",
    "assemble_pose_k",
    "masked_pose_error",
    "od_velocity_t0",
    "validate_anchor",
    "warp_ld_anchors",
    "warp_od_anchors",
    "warp_pose_k",
    "warp_pose_t0_to_ego_k",
]

#: anchor schema 版本（本模块局部标识；与 obs ``schema_version`` / 录制 schema 独立）
ANCHOR_SCHEMA_VERSION = "r36d.anchor.v1"

#: 图位姿组装模式：``off``（现状 = R33-B 改动前口径）/ ``w1``（LD 静止锚逆变换，OD 不带恒速
#: 先验）/ ``w2``（w1 + OD 恒速先验 = ``off`` 整图的 ego_k 系重表达）。
WARP_MODES: tuple[str, ...] = ("off", "w1", "w2")

#: anchor schema ↔ 既有 pose/anchor 契约字段映射（文档化数据；见模块 docstring 表）。
FIELD_MAP: Mapping[str, Mapping[str, str]] = {
    "xy": {"frame.pose[..., :2]": "FrameEncoding.pose 前两维（ego t0 系位置）"},
    "theta": {
        "frame.pose[..., 2]": "FrameEncoding.pose 第三维（相对 ego 航向，rad）",
        "st_gnn.od_anchor[..., 4]": "od_state_from_features 的 heading",
        "st_gnn.ld_anchor[..., 2]": "ld_state_from_features 的 heading（= ld_feat[..., 2]）",
    },
    "vel": {
        "od_feat[..., 2:4]": "od 原始 9 维的 (vx, vy)（**相对速度**，自车系；warp 口径需 + ego t0 速度）",
        "st_gnn.od_anchor[..., 2:4]": "od_state_from_features 的 (vx, vy)（同上，相对速度）",
    },
    "size": {"od_feat[..., 6:8]": "od 原始 9 维的 (L, W)"},
    "live": {
        "od_mask∧od_presence": "OD 有效 = 槽有效 ∧ 本帧观测新鲜",
        "ld_mask": "LD 有效 = 采样点有效",
    },
    "slot_id": {"obs['od_id'] / od_id_hist": "OD track id（-1 = 空槽）；LD 固定几何槽无 id"},
}


# --------------------------------------------------------------------------- #
# 核心：t0 帧 → ego_k 系（与 R33-B ``_ego_warp_pose_k`` 同 op 序 ⇒ 逐位一致）
# --------------------------------------------------------------------------- #
def warp_pose_t0_to_ego_k(ego_pose: Tensor, pose_t0: Tensor) -> Tensor:
    """节点位姿 ``(B,N,3)``（t0 帧）→ ego_k 系：``p' = R(−Δθ)(p − Δp)``、``ψ' = ψ − Δθ``。

    ``ego_pose (B,3)`` = 第 k 步 ego 累积位姿（t0 帧；即 ``(Δp, Δθ)``）。与
    ``DrivingModel._ego_warp_pose_k`` 的逐节点分量实现同 op 序（同一 :func:`net.spatial.wrap_angle`），
    供逐位等价单测。
    """
    dtheta = ego_pose[:, 2:3]
    cos_t, sin_t = torch.cos(dtheta), torch.sin(dtheta)
    rel = pose_t0[..., :2] - ego_pose[:, None, :2]
    x = cos_t * rel[..., 0] + sin_t * rel[..., 1]
    y = -sin_t * rel[..., 0] + cos_t * rel[..., 1]
    theta = wrap_angle(pose_t0[..., 2] - dtheta)
    return torch.stack([x, y, theta], dim=-1)


def warp_pose_k(ego_pose: Tensor, od_pose: Tensor, ld_pose: Tensor) -> Tensor:
    """R33-B ``DrivingModel._ego_warp_pose_k`` 的等价纯函数（ego + OD + LD 同一刚体变换）。

    Args:
        ego_pose: ``(B,3)`` 第 k 步 ego 累积位姿（t0 帧）。
        od_pose: ``(B,S,3)`` **warp 前** OD 位姿（w1: 无恒速先验；w2: 含先验——由调用方决定）。
        ld_pose: ``(B,L,3)`` warp 前 LD 位姿（t0 静止）。

    Returns:
        ``(B,1+S+L,3)``；ego 槽恒 ``(0,0,0)``（同一变换作用于 ego 自身的自洽结果）。
    """
    slots = torch.cat([od_pose, ld_pose], dim=1)
    warped = warp_pose_t0_to_ego_k(ego_pose, slots)
    ego_zero = torch.zeros_like(ego_pose)
    return torch.cat([ego_zero.unsqueeze(1), warped], dim=1)


# --------------------------------------------------------------------------- #
# OD：恒速先验推进 + 组装（off / w1 / w2）
# --------------------------------------------------------------------------- #
def advance_od_pose(od_pose: Tensor, od_vel: Tensor, dt: float, steps: int = 1) -> Tensor:
    """OD 恒速先验推进（t0 场景系）：``p' = p + v·(steps·dt)``，航向不变。

    ``od_pose (B,S,3)`` / ``od_vel (B,S,2)`` = **对象在 t0 场景系的速度**（先经
    :func:`od_velocity_t0` 由通道相对速度换算；warp 口径下不可直接传相对速度）。
    ``mode="off"`` 组装与 rollout 内联口径同 op 序：``od_pose_t0[..., :2] + od_velocity * (float(k)·dt)``。
    """
    offset = float(int(steps)) * float(dt)
    return torch.cat([od_pose[..., :2] + od_vel * offset, od_pose[..., 2:]], dim=-1)


def od_velocity_t0(od_rel_vel: Tensor, ego_vel_t0: Tensor) -> Tensor:
    """od 通道**相对速度** → **t0 场景系对象速度**（warp 口径的 ``v_OD``）：``v_rel + v_ego``。

    ``od_rel_vel (B,S,2)`` = ``od_now[..., 2:4]``（对象速度 − ego 速度，自车系）；
    ``ego_vel_t0 (B,2)`` = ego 在 t0 系的速度（通常 ``cat([ego_speed, 0])``，ego 速度沿自身 x 轴）。
    返回 ``(B,S,2)``；掩码外槽位由调用方先清零。
    """
    return od_rel_vel + ego_vel_t0[:, None, :]


def masked_pose_error(pred: Tensor, actual: Tensor, live: Tensor) -> Tensor:
    """节点位姿误差的**位置 L2 掩码均值**（``(B,N,3)`` × ``live (B,N)`` → ``(B,)``，单位 m）。

    R37 M1 分项读数用（解析先验误差 / 学习残差贡献 / 最终 anchor 误差）：仅统计 ``live=1``
    槽位；全无效（分母 0）⇒ 0。只度量位置（航向误差由 hook 全张量留档离线算）。
    """
    diff = pred[..., :2] - actual[..., :2]
    err = torch.sqrt(diff.pow(2).sum(dim=-1).clamp_min(0.0))
    weight = live.to(err.dtype)
    return (err * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)


def assemble_pose_k(
    ego_pose: Tensor,
    od_pose_t0: Tensor,
    od_vel: Tensor,
    ld_pose_t0: Tensor,
    *,
    mode: str = "off",
    dt: float = 0.5,
    steps: int = 1,
) -> Tensor:
    """按 ``mode`` 组装 ``st_gnn`` 图位姿 ``(B,1+S+L,3)``（R33-B 三种口径）。

    - ``off``（默认；**逐位等于 rollout 现状**）：``[Δp_k, od_t0 + k·dt·v_t0, ld_t0]``（t0 帧）；
    - ``w1``：``off`` 的整图 ego_k 系重表达，且 OD **不带**恒速先验（t0 静止位姿）；
    - ``w2``：``off`` 的整图 ego_k 系重表达（OD 含恒速先验）。

    Args:
        ego_pose: ``(B,3)`` 第 k 步 ego 累积位姿（t0 帧）= ``(Δp_k, Δθ_k)``。
        od_pose_t0/ld_pose_t0: ``(B,S,3)/(B,L,3)`` t0 帧节点位姿。
        od_vel: ``(B,S,2)`` 恒速先验速度（**off**：守卫生产路径 bit-exact ⇒ 传 od 通道的相对速度
            ``od_now[..., 2:4] * od_live``；**w1/w2**：warp 把 t0 系当静态场景 ⇒ 应传 t0 场景系
            对象速度 :func:`od_velocity_t0`）。
        dt/steps: 步长 [s] / 步数 k（``off`` 的先验视界 = ``steps·dt``）。
    """
    mode = _check_mode(mode)
    if mode == "off":
        return torch.cat(
            [ego_pose.unsqueeze(1), advance_od_pose(od_pose_t0, od_vel, dt, steps), ld_pose_t0],
            dim=1,
        )
    od_pose_pre = od_pose_t0 if mode == "w1" else advance_od_pose(od_pose_t0, od_vel, dt, steps)
    return warp_pose_k(ego_pose, od_pose_pre, ld_pose_t0)


def _check_mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in WARP_MODES:
        raise ValueError(f"未知 warp mode={mode!r}（off | w1 | w2）")
    return mode


# --------------------------------------------------------------------------- #
# anchor schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AnchorBatch:
    """显式 anchor 张量集（ego t0 系；字段/单位/契约映射见模块 docstring 表）。

    ``kind`` 仅用于校验与文档（``"od"``：速度/尺寸/身份语义有效；``"ld"``：静态锚，
    ``vel ≡ 0``、``size ≡ 0``、``slot_id ≡ -1``）。``vel`` = **t0 场景系速度**（warp 口径；
    od 通道给的相对速度需先经 :func:`od_velocity_t0` 换算）。
    """

    xy: Tensor
    theta: Tensor
    vel: Tensor
    size: Tensor
    live: Tensor
    slot_id: Tensor
    kind: str = "od"

    def to_pose(self) -> Tensor:
        """→ ``(B,N,3)`` 节点位姿（与 ``FrameEncoding.pose`` / ``st_gnn`` pose 契约一致）。"""
        return torch.cat([self.xy, self.theta.unsqueeze(-1)], dim=-1)


def validate_anchor(anchor: AnchorBatch) -> None:
    """schema 形状/类型校验（失败即 ``ValueError``；坐标层 fail-fast）。"""
    if anchor.kind not in ("od", "ld"):
        raise ValueError(f"anchor.kind 必须为 'od' | 'ld'，收到 {anchor.kind!r}")
    xy = anchor.xy
    if xy.ndim != 3 or int(xy.shape[-1]) != 2:
        raise ValueError(f"anchor.xy 形状应为 (B,N,2)，收到 {tuple(xy.shape)}")
    batch, n = int(xy.shape[0]), int(xy.shape[1])
    for name, t, shape in (
        ("theta", anchor.theta, (batch, n)),
        ("vel", anchor.vel, (batch, n, 2)),
        ("size", anchor.size, (batch, n, 2)),
        ("live", anchor.live, (batch, n)),
        ("slot_id", anchor.slot_id, (batch, n)),
    ):
        if tuple(t.shape) != shape:
            raise ValueError(f"anchor.{name} 形状应为 {shape}，收到 {tuple(t.shape)}")
    if anchor.slot_id.dtype != torch.long:
        raise ValueError(f"anchor.slot_id 必须为 torch.long，收到 {anchor.slot_id.dtype}")
    if anchor.kind == "ld" and anchor.vel.numel() and bool(anchor.vel.abs().max() > 0):
        raise ValueError("LD anchor 为静态锚：vel 必须恒 0")


def _warp_anchor_fields(anchor: AnchorBatch, xy: Tensor, theta: Tensor) -> AnchorBatch:
    """保持 ``vel/size/live/slot_id`` 不变、替换 ``xy/theta`` 的 anchor 拷贝。"""
    return AnchorBatch(
        xy=xy,
        theta=theta,
        vel=anchor.vel,
        size=anchor.size,
        live=anchor.live,
        slot_id=anchor.slot_id,
        kind=anchor.kind,
    )


def warp_ld_anchors(anchor: AnchorBatch, ego_pose: Tensor) -> AnchorBatch:
    """静态 LD anchor 逆变换（``p' = R(−Δθ)(p − Δp)``、``ψ' = ψ − Δθ``）。

    尺寸/掩码/身份不随刚体变换改变（``size ≡ 0``、``slot_id ≡ -1`` 原样保留）。
    """
    validate_anchor(anchor)
    warped = warp_pose_t0_to_ego_k(ego_pose, anchor.to_pose())
    return _warp_anchor_fields(anchor, warped[..., :2], warped[..., 2])


def warp_od_anchors(
    anchor: AnchorBatch, ego_pose: Tensor, *, dt: float = 0.5, steps: int = 1
) -> AnchorBatch:
    """OD anchor：先恒速推进 ``p + v·(steps·dt)``，再同一逆变换（``p' = T_k^{-1}(p + v·Δt)``）。

    ``anchor.vel`` 必须为 **t0 场景系速度**（见 :func:`od_velocity_t0`；直接传 od 通道相对速度
    会让同速对象系统性落后 ``|Δp|``）。
    """
    validate_anchor(anchor)
    advanced = torch.cat([anchor.xy + anchor.vel * (float(int(steps)) * float(dt)),
                          anchor.theta.unsqueeze(-1)], dim=-1)
    warped = warp_pose_t0_to_ego_k(ego_pose, advanced)
    return _warp_anchor_fields(anchor, warped[..., :2], warped[..., 2])


# --------------------------------------------------------------------------- #
# 5 项 hook（默认关；probe 侧只读）
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WarpHookRecord:
    """单步 5 项 hook 快照（①..⑤；见模块 docstring）。"""

    step: int
    anchor_pre: Tensor | None = None
    anchor_analytic: Tensor | None = None
    anchor_actual_next: Tensor | None = None
    residual_learned: Tensor | None = None
    anchor_final: Tensor | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "anchor_pre": self.anchor_pre,
            "anchor_analytic": self.anchor_analytic,
            "anchor_actual_next": self.anchor_actual_next,
            "residual_learned": self.residual_learned,
            "anchor_final": self.anchor_final,
        }


@dataclass
class WarpHooks:
    """warp 5 项 hook 容器（**默认关**；开启后只读，不写回任何输入/输出）。

    ``enabled=False``（默认）⇒ :meth:`capture` 立即返回 ``None``：零张量操作、不消耗 RNG、
    不改写输入张量（前向逐位不变）。``max_records>0`` 时超出上限的 capture 只计数不存张量
    （长 rollout 探针的内存护栏）。
    """

    enabled: bool = False
    max_records: int = 0
    records: list[WarpHookRecord] = field(default_factory=list)
    capture_calls: int = 0
    skipped_records: int = 0

    def capture(
        self,
        *,
        step: int,
        anchor_pre: Tensor,
        anchor_analytic: Tensor,
        anchor_actual_next: Tensor | None = None,
        residual_learned: Tensor | None = None,
    ) -> WarpHookRecord | None:
        """记录 ①..⑤；返回快照（``enabled=False`` ⇒ ``None``，不触碰任何张量）。

        ⑤ = ② + ④（④ 缺省 ⇒ ⑤ = ②）。所有张量 ``detach().clone()`` 后存储（只读、防后续
        原地改写污染；不改变调用方张量）。
        """
        if not self.enabled:
            return None
        self.capture_calls += 1
        if self.max_records > 0 and len(self.records) >= int(self.max_records):
            self.skipped_records += 1
            return None
        final = anchor_analytic if residual_learned is None else anchor_analytic + residual_learned
        record = WarpHookRecord(
            step=int(step),
            anchor_pre=anchor_pre.detach().clone(),
            anchor_analytic=anchor_analytic.detach().clone(),
            anchor_actual_next=None if anchor_actual_next is None else anchor_actual_next.detach().clone(),
            residual_learned=None if residual_learned is None else residual_learned.detach().clone(),
            anchor_final=final.detach().clone(),
        )
        self.records.append(record)
        return record

    def clear(self) -> None:
        self.records.clear()
        self.capture_calls = 0
        self.skipped_records = 0
