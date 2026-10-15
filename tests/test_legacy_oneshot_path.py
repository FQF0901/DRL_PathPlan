"""R38-P1：``tools/legacy/oneshot_path.py``（H1 一次性六步路径快照）单测。

覆盖：快照可加载（不被生产导入）/ ``compose_path_plan`` 的 H1 口径不变量（
``plan[:,0]==action_mu``、``ds_k = s_k−s_{k−1}``、dθ 截断）/ ``plan_override`` 钉住语义
``pinned_tail_actions`` / 装载守卫默认拒绝 / 模块自检可跑通。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tools.legacy.oneshot_path import (
    DTHETA_HIGH,
    DT,
    compose_path_plan,
    one_shot_plan,
    pinned_tail_actions,
    replay_plan,
    require_oneshot_optin,
)

ROOT = Path(__file__).resolve().parents[1]


def _inputs(batch: int = 2, steps: int = 6, ds_value: float = 3.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kappa_ctrl = torch.tensor([[0.02, 0.0, -0.01], [0.0, 0.01, 0.0]])[:batch]
    s_ctrl = torch.tensor([[5.25, 17.5, 29.75]]).expand(batch, -1).contiguous()
    ds = torch.full((batch, steps), float(ds_value))
    return kappa_ctrl, s_ctrl, ds


def test_compose_path_plan_contract() -> None:
    """H1 口径：plan[:,0]==action_mu、ds 前缀和 = progress、dθ 分段平均曲率积分。"""
    kappa_ctrl, s_ctrl, ds = _inputs()
    out = compose_path_plan(kappa_ctrl, s_ctrl, ds)

    assert tuple(out["plan"].shape) == (2, 6, 2)
    assert torch.equal(out["action_mu"], out["plan"][:, 0, :])
    assert torch.allclose(out["ds"], ds)
    assert torch.allclose(out["progress_s"][:, -1], ds.sum(dim=1), atol=1e-6)
    # dθ_k = κ̄_k·ds_k ⇒ κ_seg 反解回段内平均曲率
    assert torch.allclose(out["kappa_seg"], out["dtheta"] / out["ds"].clamp_min(1e-6), atol=1e-6)
    assert out["dtheta"].abs().max() <= DTHETA_HIGH
    # 轨迹 = 同一 plan 的圆弧复算
    replay = replay_plan(out["plan"])
    assert torch.allclose(replay["traj_xy"], out["traj_xy"], atol=1e-6)
    assert torch.allclose(replay["traj_theta"], out["traj_theta"], atol=1e-6)
    # 直锚 + 零曲率 ⇒ 直线前进（dy ≡ 0）
    straight = compose_path_plan(torch.zeros(2, 3), s_ctrl, ds)
    assert torch.allclose(straight["plan"][..., 1], torch.zeros(2, 6), atol=1e-6)
    assert torch.allclose(straight["traj_xy"][..., 1], torch.zeros(2, 6), atol=1e-6)


def test_compose_path_plan_clamps_and_validation() -> None:
    """动作界截断（ds<0 / dθ 超界）+ 形状校验 fail-fast。"""
    kappa_ctrl, s_ctrl, _ = _inputs()
    ds = torch.tensor([[-1.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.0] * 6])
    out = compose_path_plan(torch.full((2, 3), 0.5), s_ctrl, ds)
    assert bool(out["ds_clamped"][0, 0])
    assert torch.allclose(out["ds"][0, 0], torch.tensor(0.0))
    assert bool(out["dtheta_clamped"].any())
    with pytest.raises(ValueError):
        compose_path_plan(kappa_ctrl, s_ctrl, torch.ones(2, 6, 1))
    with pytest.raises(ValueError):
        compose_path_plan(kappa_ctrl[:1], s_ctrl, ds)
    bad_stations = torch.tensor([[5.25, 5.25, 29.75]]).expand(2, -1).contiguous()
    with pytest.raises(ValueError):
        compose_path_plan(kappa_ctrl, bad_stations, ds)


def test_one_shot_plan_is_compose_wrapper() -> None:
    """``ComposeLayer.one_shot_plan`` 快照 = compose_path_plan（同值）。"""
    kappa_ctrl, s_ctrl, ds = _inputs()
    assert torch.equal(
        one_shot_plan(kappa_ctrl, s_ctrl, ds)["plan"], compose_path_plan(kappa_ctrl, s_ctrl, ds)["plan"]
    )
    assert DT == 0.5


def test_pinned_tail_actions_semantics() -> None:
    """``plan_override`` 钉住语义：step0 = action_mu，尾段 = plan[:,1:]；形状不符 fail-fast。"""
    kappa_ctrl, s_ctrl, ds = _inputs()
    out = compose_path_plan(kappa_ctrl, s_ctrl, ds)
    plan, action_mu = out["plan"], out["action_mu"]
    pinned = pinned_tail_actions(plan, action_mu)
    assert torch.equal(pinned, plan)
    fake_mu = torch.zeros_like(action_mu)
    pinned_fake = pinned_tail_actions(plan, fake_mu)
    assert torch.equal(pinned_fake[:, 0, :], fake_mu)
    assert torch.equal(pinned_fake[:, 1:, :], plan[:, 1:, :])
    with pytest.raises(ValueError):
        pinned_tail_actions(plan, action_mu.unsqueeze(0))


def test_oneshot_optin_guard_default_rejects() -> None:
    """装载守卫：默认拒绝；显式 opt-in 通过。"""
    with pytest.raises(ValueError, match="一次性 6 步"):
        require_oneshot_optin()
    require_oneshot_optin(True)


def test_legacy_module_selftest_runs() -> None:
    """模块 ``__main__`` 自检可在仓库根目录跑通（README 记录的运行方式）。"""
    completed = subprocess.run(
        [sys.executable, "-m", "tools.legacy.oneshot_path"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "[legacy oneshot] selftest ok" in completed.stdout
