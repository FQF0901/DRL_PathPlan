"""R38-P1 减枝迁移快照：**H1 一次性六步路径**（``oneshot`` 诊断臂）。

出处（快照 commit = ``74f0bd4``；源 = 删除前的 ``net/``，行号按该 commit 计）
-----------------------------------------------------------------------
- ``net/path_head.py:641-702`` → :func:`compose_path_plan`（**一次性 6 步组合层**，逐行拷贝）；
- ``net/path_head.py:525-531`` → :func:`_check_ctrl_stations`（拷贝）；
- ``net/path_head.py:549-572`` → :func:`_integrate_kappa`（分段线性 κ 精确积分；拷贝）；
- ``net/path_head.py:575-601`` → :func:`replay_plan`（含 ``net/model.py:226-247`` 的
  ``arc_step``/``compose_pose`` 圆弧数学，改为本模块内联 ``_arc_step``/``_compose_pose``，
  避免依赖当前的 ``net.model``）；
- ``net/path_head.py:1053-1054`` → :func:`one_shot_plan`（``ComposeLayer.one_shot_plan`` 方法体）；
- ``net/path_head.py:1288-1305`` → :func:`oneshot_head_forward`（
  ``StructuredPolicyHead.forward(..., one_shot=True)`` 的叠加块：用一次性 plan 覆盖
  ``plan/action/action_mu/dtheta/progress_s/speed/kappa_seg*/traj_*``）；
- ``net/model.py:1911-1916`` → :func:`pinned_tail_actions`（``plan_override`` 钉住尾段语义：
  ``action = plan[:, k, :]``，step0 仍 = ``action_mu``）；
- ``net/model.py:986-993`` → :func:`require_oneshot_optin`（R36 L-A 装载守卫：默认拒绝）。

判定出处
--------
R38 弃用清单 A 组（``ARCHITECTURE_FORBIDDEN``：唯一动作源 + 每步单动作）：
一次性六步 = 第二个六步轨迹生成器、绕过「决策 → WM 推进一步 → 同一头重新决策」。
R38-P1 把实现迁到本快照并从 ``net/`` 删除（``tools/legacy/README.md`` 登记）。

**仅历史复现，需旧 net 版本**
-----------------------------
本模块自包含（只依赖 ``torch``），可直接复算 H1 口径的 6 步 plan；
但要在**模型内**复现 ``oneshot`` 诊断臂（``forward(..., one_shot=True)`` + ``plan_override``
钉住）必须用删除前的 net 版本，例如::

    git worktree add /tmp/old_net 74f0bd4      # 或 git show 74f0bd4:net/path_head.py

本模块不保证与当前 ``net`` 接口兼容，也**不被生产代码导入**（CI 守卫 #1 的扫描面）。
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "DTHETA_HIGH",
    "DTHETA_LOW",
    "DS_HIGH",
    "DT",
    "compose_path_plan",
    "one_shot_plan",
    "oneshot_head_forward",
    "pinned_tail_actions",
    "replay_plan",
    "require_oneshot_optin",
]

# --- 常量快照（出处 net/path_head.py:144-159；动作界与 net.policy.ACTION_HIGH/LOW 同值）
_EPS = 1e-6
#: 单步时长（s）
DT = 0.5
#: 动作上界：ds ∈ [0,10] m/步、dθ ∈ [−0.6,0.6] rad/步
DS_HIGH = 10.0
DTHETA_LOW = -0.6
DTHETA_HIGH = 0.6


def _wrap_angle(theta: Tensor) -> Tensor:
    """出处 ``net.spatial.wrap_angle``（快照内联，避免依赖当前 net）。"""
    return torch.remainder(theta + torch.pi, 2.0 * torch.pi) - torch.pi


def _arc_step(ds: Tensor, dtheta: Tensor, eps: float = _EPS) -> tuple[Tensor, Tensor]:
    """出处 ``net/model.py:226-238``（``arc_step``，逐行拷贝）。"""
    small = dtheta.abs() < eps
    safe = torch.where(small, torch.ones_like(dtheta), dtheta)
    radius = ds / safe
    dx = torch.where(small, ds, radius * torch.sin(dtheta))
    dy = torch.where(small, torch.zeros_like(ds), radius * (1.0 - torch.cos(dtheta)))
    return dx, dy


def _compose_pose(pose: Tensor, dx: Tensor, dy: Tensor, dtheta: Tensor) -> Tensor:
    """出处 ``net/model.py:241-247``（``compose_pose``，逐行拷贝）。"""
    theta = pose[..., 2]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    new_x = pose[..., 0] + cos_t * dx - sin_t * dy
    new_y = pose[..., 1] + sin_t * dx + cos_t * dy
    return torch.stack([new_x, new_y, _wrap_angle(theta + dtheta)], dim=-1)


def _check_ctrl_stations(s_ctrl: Tensor) -> None:
    """出处 ``net/path_head.py:525-531``（拷贝）。"""
    if s_ctrl.ndim != 2 or int(s_ctrl.shape[-1]) != 3:
        raise ValueError(f"s_ctrl 形状应为 (B,3)，收到 {tuple(s_ctrl.shape)}")
    gaps = s_ctrl[:, 1:] - s_ctrl[:, :-1]
    if not bool((gaps > 1e-3).all()):
        raise ValueError(f"s_ctrl 必须严格升序且间隔 ≥1e-3，收到 {s_ctrl.detach().tolist()[:2]}")


def _integrate_kappa(kappa_ctrl: Tensor, s_ctrl: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
    """出处 ``net/path_head.py:549-572``（拷贝；分段线性 κ 的精确积分）。"""
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
    """出处 ``net/path_head.py:575-601``（拷贝；``(B,K,2)`` 的 ``(ds,dθ)`` → 位姿链）。"""
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
        dx, dy = _arc_step(ds, dtheta)
        pose = _compose_pose(pose, dx, dy, dtheta)
        xy.append(pose[:, :2])
        theta.append(pose[:, 2])
    return {"traj_xy": torch.stack(xy, dim=1), "traj_theta": torch.stack(theta, dim=1)}


def compose_path_plan(kappa_ctrl: Tensor, s_ctrl: Tensor, ds: Tensor, *, dt: float = DT) -> dict[str, Tensor]:
    """出处 ``net/path_head.py:641-702``（拷贝）：H1 口径的一次性 6 步组合层。

    ``plan (B,K,2)``，``ds_k = s_k − s_{k−1}``（``s_0≡0``）、``dθ_k = κ̄_k·ds_k``
    （``κ̄_k`` = 段内平均曲率）；``action_mu = plan[:,0]``。
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


def one_shot_plan(kappa_ctrl: Tensor, s_ctrl: Tensor, ds: Tensor, *, dt: float = DT) -> dict[str, Tensor]:
    """出处 ``net/path_head.py:1053-1054``（``StructuredComposeLayer.one_shot_plan`` 方法体）。"""
    return compose_path_plan(kappa_ctrl, s_ctrl, ds, dt=dt)


def oneshot_head_forward(
    head: "torch.nn.Module",
    latent: Tensor,
    *,
    anchor_xy: Tensor,
    anchor_mask: Tensor | None = None,
    long_exog: Tensor | None = None,
    target_lane_valid: Tensor | None = None,
) -> dict[str, Tensor]:
    """出处 ``net/path_head.py:1288-1305``：H1 一次性 6 步叠加块（需旧 net 版本）。

    需要旧的 ``StructuredPolicyHead``（其 ``forward`` 有 ``one_shot`` 形参、``compose`` 有
    ``one_shot_plan`` 方法）。当前 net 已删除该路径 ⇒ 传当前 head 会 ``AttributeError``；
    复现请用删除前的 net（``git worktree add /tmp/old_net 74f0bd4``）并调用旧
    ``head.forward(..., one_shot=True)``。
    """
    if not hasattr(getattr(head, "compose", None), "one_shot_plan"):
        raise AttributeError(
            "当前 net 的 StructuredPolicyHead 已删除一次性 6 步路径（R38-P1，"
            "见 tools/legacy/README.md）：请在删除前的 net 上调用 "
            "head.forward(..., one_shot=True)（旧版本 = git show 74f0bd4:net/path_head.py）。"
        )
    output = head(
        latent,
        anchor_xy=anchor_xy,
        anchor_mask=anchor_mask,
        long_exog=long_exog,
        target_lane_valid=target_lane_valid,
        one_shot=True,
    )
    return output


def pinned_tail_actions(plan: Tensor, action_mu: Tensor) -> Tensor:
    """出处 ``net/model.py:1911-1916``：``plan_override`` 钉住尾段的动作序列（语义快照）。

    旧 ``_rollout`` 里 ``k ≥ 1`` 时 ``action = plan[:, k, :]``（step0 仍 = ``action_mu``）；
    返回 ``(B,K,2)``：``[action_mu, plan[:,1], …, plan[:,K-1]]``。
    """
    if plan.ndim != 3 or int(plan.shape[-1]) != 2:
        raise ValueError(f"plan 形状应为 (B,K,2)，收到 {tuple(plan.shape)}")
    if action_mu.shape != plan[:, 0, :].shape:
        raise ValueError(
            f"action_mu 形状应为 {tuple(plan[:, 0, :].shape)}，收到 {tuple(action_mu.shape)}"
        )
    tail = plan[:, 1:, :]
    return torch.cat([action_mu.unsqueeze(1), tail], dim=1)


def require_oneshot_optin(allow_oneshot: bool = False) -> None:
    """出处 ``net/model.py:986-993``（R36 L-A 装载守卫；语义快照）。

    一次性 6 步**默认拒绝**（不接入生产）；确需历史复现必须显式 ``allow_oneshot=True``。
    """
    if not allow_oneshot:
        raise ValueError(
            "oneshot 一次性 6 步诊断臂（不接入生产）：默认拒绝。"
            "确需历史复现请显式 allow_oneshot=True（旧工具侧 --diagnostic-oneshot）。"
        )


def _selftest() -> int:
    """合成输入自检：形状/契约（``plan[:,0]==action_mu``）/ 钉住语义/装载守卫。"""
    batch, steps = 2, 6
    kappa_ctrl = torch.tensor([[0.02, 0.0, -0.01], [0.0, 0.01, 0.0]])
    s_ctrl = torch.tensor([[5.25, 17.5, 29.75]]).expand(batch, -1).contiguous()
    ds = torch.full((batch, steps), 3.0)
    out = compose_path_plan(kappa_ctrl, s_ctrl, ds)
    assert tuple(out["plan"].shape) == (batch, steps, 2)
    assert torch.equal(out["action_mu"], out["plan"][:, 0, :])
    pinned = pinned_tail_actions(out["plan"], out["action_mu"])
    assert torch.equal(pinned, out["plan"])
    try:
        require_oneshot_optin()
    except ValueError:
        pass
    else:  # pragma: no cover - 守卫失效即报错
        raise SystemExit("[legacy oneshot] 装载守卫失效：默认未拒绝")
    require_oneshot_optin(True)
    print(
        f"[legacy oneshot] selftest ok: plan={tuple(out['plan'].shape)} "
        f"ds_sum={float(out['ds'].sum()):.3f} dtheta_max={float(out['dtheta'].abs().max()):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
