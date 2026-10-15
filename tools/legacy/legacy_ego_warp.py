"""R38-P2b-C：legacy ``set_ego_warp``（w1/w2）快照（**逐行口径拷贝**，仅历史复现/审计）。

出处（快照 commit = ``dd0a8c2``（R38-P2b 前 HEAD），行号为快照时点）：

- ``net/model.py:593-628`` → :meth:`DrivingModel.set_ego_warp`（模式 setter + 与 ``ego_warp_mode``
  的互斥守卫）；本模块的 :func:`set_legacy_ego_warp` 保留守卫语义但**不能驱动当前 net 的
  rollout**（w1/w2 已从前向路径删除；见下"复现约束"）；
- ``net/model.py:697-717`` → ``DrivingModel._ego_warp_pose_k``（静态方法，逐节点逆变换实现）
  = :func:`legacy_ego_warp_pose_k`；
- ``net/ego_warp.py:189-227``（快照时点）→ ``assemble_pose_k``（``off``/``w1``/``w2``）+
  ``_check_mode`` = :func:`legacy_assemble_pose_k` / :func:`legacy_check_mode`；
- ``net/ego_warp.py:154-164`` → ``advance_od_pose``（本模块重实现为 :func:`_advance_od_pose`，
  op 序逐行一致）。

**判定**：`LEGACY_REPRO`（`docs/deprecation_manifest.yaml` C 组：w1/w2 = 历史 warp 实验口径；
R37 M1 后正式接口 = ``set_ego_warp_mode(off|analytic)``，v_OD 语义修正后旧 w2（相对速度先验）
不再是生产口径）。

复现约束
--------
1. **逐位参照物 = fixture**：w1/w2 的逐位期望已冻结在 ``tests/fixtures/r38_p2b_warp/``
   （生成器 ``.slim/deepwork/p1_v2/round38/p2b_oracle_freeze.py``；含 pure_cases / model_poses /
   m1_arms / b_compat_arms + manifest sha256）。Gate/单测**读 fixture**，不再调用模型侧 w1/w2；
2. 本模块纯函数（:func:`legacy_ego_warp_pose_k` / :func:`legacy_assemble_pose_k`）可在当前 net 上
   复算 w1/w2 的 pose_k 口径（用于 fixture 完整性核验与历史分析）；
3. **模型内整臂复现**（把 w1/w2 接回 rollout 产出 plan/traj）需删除前 net：
   ``git worktree add /tmp/legacy_net dd0a8c2``（``net/`` 单树快照即可；见 README）；
4. :func:`set_legacy_ego_warp` 对当前 net **fail-fast**：w1/w2 不再进入前向路径，静默设置属性
   只会误导（生产 net 只有一套 warp 状态机 ``ego_warp_mode ∈ {off, analytic}``）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = [
    "LEGACY_WARP_MODES",
    "legacy_assemble_pose_k",
    "legacy_check_mode",
    "legacy_ego_warp_pose_k",
    "set_legacy_ego_warp",
]

#: 快照时点 ``WARP_MODES``（legacy）：``off`` / ``w1`` / ``w2``。
#: 正式接口现为 ``("off", "analytic")``（``net/ego_warp.py``）；``analytic`` 的 op 序 = 旧 ``w2``。
LEGACY_WARP_MODES: tuple[str, ...] = ("off", "w1", "w2")


def _wrap_angle(theta: Tensor) -> Tensor:
    """``net/spatial.py::wrap_angle`` 的口径拷贝（``atan2(sin, cos)``；逐位同 op 序）。"""
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def _advance_od_pose(od_pose: Tensor, od_vel: Tensor, dt: float, steps: int = 1) -> Tensor:
    """出处 ``net/ego_warp.py:154-164``（``advance_od_pose``）：``p' = p + v·(steps·dt)``。"""
    offset = float(int(steps)) * float(dt)
    return torch.cat([od_pose[..., :2] + od_vel * offset, od_pose[..., 2:]], dim=-1)


def legacy_check_mode(mode: str) -> str:
    """出处 ``net/ego_warp.py:223-227``（快照时点）：legacy 三档校验（含旧报错文案）。"""
    mode = str(mode).lower()
    if mode not in LEGACY_WARP_MODES:
        raise ValueError(f"未知 warp mode={mode!r}（off | w1 | w2）")
    return mode


def legacy_ego_warp_pose_k(ego_pose: Tensor, od_pose: Tensor, ld_pose: Tensor) -> Tensor:
    """出处 ``net/model.py:697-717``（``DrivingModel._ego_warp_pose_k``，逐行拷贝）。

    t0 帧节点位姿 → ego_k 自车系（``p' = R(−Δθ)(p − Δp)``、``ψ' = ψ − Δθ``），
    ego 槽恒 ``(0,0,0)``。
    """
    slots = torch.cat([od_pose, ld_pose], dim=1)
    dtheta = ego_pose[:, 2:3]
    cos_t, sin_t = torch.cos(dtheta), torch.sin(dtheta)
    rel = slots[..., :2] - ego_pose[:, None, :2]
    x = cos_t * rel[..., 0] + sin_t * rel[..., 1]
    y = -sin_t * rel[..., 0] + cos_t * rel[..., 1]
    theta = _wrap_angle(slots[..., 2] - dtheta)
    ego_zero = torch.zeros_like(ego_pose)
    return torch.cat([ego_zero.unsqueeze(1), torch.stack([x, y, theta], dim=-1)], dim=1)


def legacy_assemble_pose_k(
    ego_pose: Tensor,
    od_pose_t0: Tensor,
    od_vel: Tensor,
    ld_pose_t0: Tensor,
    *,
    mode: str = "off",
    dt: float = 0.5,
    steps: int = 1,
) -> Tensor:
    """出处 ``net/ego_warp.py:189-227``（快照时点 ``assemble_pose_k``，逐行拷贝）。

    - ``off``：``[Δp_k, od_t0 + k·dt·v_t0, ld_t0]``（t0 帧；v 语义 = 调用方传入）；
    - ``w1``：``off`` 的整图 ego_k 系重表达，OD **不带**恒速先验；
    - ``w2``：``off`` 的整图 ego_k 系重表达（OD 含恒速先验）。

    注：旧 w2 的模型内调用点用**通道相对速度**推进（随体语义泄漏；R36 L-D 已证系统性重复扣
    ``|Δp|``）——本快照只复算数学，不改变语义；正式口径见 :func:`net.ego_warp.assemble_pose_k`
    ``mode="analytic"``（同 op 序，速度 = t0 场景系）。
    """
    mode = legacy_check_mode(mode)
    if mode == "off":
        return torch.cat(
            [ego_pose.unsqueeze(1), _advance_od_pose(od_pose_t0, od_vel, dt, steps), ld_pose_t0],
            dim=1,
        )
    od_pose_pre = od_pose_t0 if mode == "w1" else _advance_od_pose(od_pose_t0, od_vel, dt, steps)
    slots = torch.cat([od_pose_pre, ld_pose_t0], dim=1)
    dtheta = ego_pose[:, 2:3]
    cos_t, sin_t = torch.cos(dtheta), torch.sin(dtheta)
    rel = slots[..., :2] - ego_pose[:, None, :2]
    x = cos_t * rel[..., 0] + sin_t * rel[..., 1]
    y = -sin_t * rel[..., 0] + cos_t * rel[..., 1]
    theta = _wrap_angle(slots[..., 2] - dtheta)
    ego_zero = torch.zeros_like(ego_pose)
    return torch.cat([ego_zero.unsqueeze(1), torch.stack([x, y, theta], dim=-1)], dim=1)


def set_legacy_ego_warp(model: Any, mode: str = "off") -> None:
    """出处 ``net/model.py:593-628``（``set_ego_warp`` 守卫语义）——对当前 net **fail-fast**。

    w1/w2 已从生产前向路径删除（R38-P2b）：生产 net 只有一套 warp 状态机
    ``set_ego_warp_mode(off|analytic)``。本函数保留为"旧入口不再可用"的显式报错点；
    逐位参照读 fixture，纯函数复算用 :func:`legacy_assemble_pose_k`。
    """
    legacy_check_mode(mode)
    raise RuntimeError(
        "set_ego_warp（w1/w2）已退役（R38-P2b，LEGACY_REPRO）：生产 net 不保留第二套 warp "
        "状态机；逐位参照 = tests/fixtures/r38_p2b_warp/（含 manifest sha256），"
        "纯函数复算用 tools.legacy.legacy_ego_warp.legacy_assemble_pose_k。"
        f"（收到 mode={mode!r}，model={type(model).__name__}）"
    )
