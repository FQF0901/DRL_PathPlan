"""R39-B2b：目标口径对齐损失（``column_residual_loss_aligned``）单测。

出处：``.slim/deepwork/p1_v2/round39/b2b_prereg.md`` §3（预注册：ds/dt 口径 speed 目标 +
逐列加权 + TTC 事件式）。断言面：

1. **speed_target**：仅替换第 0 列目标（其余列逐位不变）；形状 fail-fast；
2. **掩码/权重**：掩码 = ``base_present ∧ real_present``；逐列权重生效（speed 加权）；
3. **TTC 事件式**：非事件行 ``pred=cap`` ⇒ cap 项 0；事件行 ``pred=real`` ⇒ 回归项 0；
   BCE 有界且方向正确；``ttc_event_weight`` 生效；
4. **旧函数不动**：``column_residual_loss`` 语义逐位不变（本文件同时回归）。
"""

from __future__ import annotations

import math

import pytest
import torch

from net.world_residual import (
    COLUMN_LOSS_WEIGHTS,
    TARGET_LOSS_SCALE,
    column_residual_loss,
    column_residual_loss_aligned,
)


def _payload(batch: int = 4, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    base = torch.rand(batch, 4, generator=generator) * torch.tensor([6.0, 20.0, 20.0, 60.0])
    real = torch.rand(batch, 4, generator=generator) * torch.tensor([6.0, 20.0, 20.0, 60.0])
    delta = torch.randn(batch, 4, generator=generator)
    present = torch.ones(batch, 4)
    return {"base": base, "real": real, "delta": delta, "present": present}


def test_speed_target_replaces_only_first_column() -> None:
    payload = _payload()
    speed = payload["real"][:, 0] + 1.0
    base_out = column_residual_loss_aligned(
        payload["delta"], base_exog=payload["base"], base_present=payload["present"],
        real_exog=payload["real"], real_present=payload["present"], speed_target=None,
    )
    out = column_residual_loss_aligned(
        payload["delta"], base_exog=payload["base"], base_present=payload["present"],
        real_exog=payload["real"], real_present=payload["present"], speed_target=speed,
    )
    assert torch.allclose(out["per_column"][1:], base_out["per_column"][1:], atol=0.0)
    assert not torch.allclose(out["per_column"][0], base_out["per_column"][0])
    assert out["target"][:, 0].equal(speed)
    assert out["target"][:, 1:].equal(payload["real"][:, 1:])


def test_speed_target_shape_fail_fast() -> None:
    payload = _payload()
    with pytest.raises(ValueError):
        column_residual_loss_aligned(
            payload["delta"], base_exog=payload["base"], base_present=payload["present"],
            real_exog=payload["real"], real_present=payload["present"],
            speed_target=torch.zeros(3),
        )
    with pytest.raises(ValueError):
        column_residual_loss_aligned(
            payload["delta"], base_exog=payload["base"], base_present=payload["present"],
            real_exog=payload["real"], real_present=payload["present"],
            column_weights=(1.0, 1.0),
        )
    with pytest.raises(ValueError):
        column_residual_loss_aligned(
            payload["delta"], base_exog=payload["base"], base_present=payload["present"],
            real_exog=payload["real"], real_present=payload["present"], mode="huber",
        )


def test_zero_delta_equals_masked_base_vs_target() -> None:
    payload = _payload()
    present = payload["present"].clone()
    present[0, 1] = 0.0
    out = column_residual_loss_aligned(
        torch.zeros_like(payload["delta"]),
        base_exog=payload["base"], base_present=present,
        real_exog=payload["real"], real_present=present,
        scale=TARGET_LOSS_SCALE,
    )
    # 掩码行（present=0）不计：列 1 只统计 3 行
    diff = (payload["base"][:, 1] - payload["real"][:, 1]) / TARGET_LOSS_SCALE[1]
    expected = torch.nn.functional.smooth_l1_loss(
        diff[1:], torch.zeros_like(diff[1:])
    )
    assert math.isclose(float(out["per_column"][1]), float(expected), rel_tol=1e-6)
    assert out["mask"][0, 1].item() == 0.0


def test_column_weights_shift_weighted_loss() -> None:
    payload = _payload()
    out_speed_heavy = column_residual_loss_aligned(
        payload["delta"], base_exog=payload["base"], base_present=payload["present"],
        real_exog=payload["real"], real_present=payload["present"],
        column_weights=(100.0, 1.0, 1.0, 1.0),
    )
    out_equal = column_residual_loss_aligned(
        payload["delta"], base_exog=payload["base"], base_present=payload["present"],
        real_exog=payload["real"], real_present=payload["present"],
        column_weights=(1.0, 1.0, 1.0, 1.0),
    )
    assert float(out_speed_heavy["loss"]) >= float(
        out_speed_heavy["per_column"][0].item()
    ) * 0.9
    assert not math.isclose(
        float(out_speed_heavy["loss"]), float(out_equal["loss"]), rel_tol=1e-6
    )
    assert COLUMN_LOSS_WEIGHTS[0] == 2.0


def test_ttc_event_is_bounded_and_directional() -> None:
    batch = 4
    base = torch.zeros(batch, 4)
    real = torch.zeros(batch, 4)
    base[:, 3] = 60.0
    real[:, 3] = torch.tensor([60.0, 60.0, 10.0, 10.0])  # 事件 = 行 2/3
    present = torch.ones(batch, 4)
    # 纯 BCE 口径：pred=cap（全部）⇒ 每行 logits=0 ⇒ ln2；非事件行 2 + 事件行 2
    out = column_residual_loss_aligned(
        torch.zeros(batch, 4), base_exog=base, base_present=present,
        real_exog=real, real_present=present,
        ttc_event_weight=1.0, ttc_regression_weight=0.0,
    )
    assert out["n_ttc_event"] == 2 and out["n_ttc_noevent"] == 2
    assert math.isclose(float(out["per_column"][3]), math.log(2.0), rel_tol=1e-5)
    # 方向：把**事件行** pred 推到远低于 cap（判为"有冲突"）⇒ BCE↓（非事件行不动）；
    # τ 归一 ⇒ 有界族（错判代价 ∝ |cap−pred|/τ，非线性爆炸）
    delta_down = torch.zeros(batch, 4)
    delta_down[2:, 3] = -50.0
    out_down = column_residual_loss_aligned(
        delta_down, base_exog=base, base_present=present,
        real_exog=real, real_present=present,
        ttc_event_weight=1.0, ttc_regression_weight=0.0,
    )
    expected_down = (2.0 * math.log(2.0) + 2.0 * math.log1p(math.exp(-5.0))) / 4.0
    assert float(out_down["per_column"][3]) == pytest.approx(expected_down, rel=1e-5)
    assert float(out_down["per_column"][3]) < float(out["per_column"][3])
    # 回归/ cap 分档：事件行 pred=real ⇒ reg_event=0；非事件行 pred=cap ⇒ reg_cap=0
    out_terms = column_residual_loss_aligned(
        torch.zeros(batch, 4), base_exog=base, base_present=present,
        real_exog=real, real_present=present, ttc_event_weight=0.0,
        scale=TARGET_LOSS_SCALE,
    )
    # delta=0 ⇒ pred=base：事件行 pred=60 vs real=10（|Δ|/10 = 5 ⇒ smooth_l1 = 4.5）；
    # 非事件行 pred=cap ⇒ 0；ℓ_ttc = 0.5·0.5·(4.5 + 0) = 1.125
    assert float(out_terms["per_column"][3]) == pytest.approx(1.125, rel=1e-5)


def test_ttc_cap_rows_push_to_cap_not_beyond() -> None:
    batch = 2
    base = torch.zeros(batch, 4)
    real = torch.zeros(batch, 4)
    base[:, 3] = 60.0
    real[:, 3] = 60.0  # 全部非事件行
    present = torch.ones(batch, 4)
    # pred > cap（学习残差把 ttc 推过 cap）⇒ min(pred,cap)=cap ⇒ cap 项仍 0
    delta = torch.zeros(batch, 4)
    delta[:, 3] = +30.0
    out = column_residual_loss_aligned(
        delta, base_exog=base, base_present=present,
        real_exog=real, real_present=present, ttc_event_weight=0.0,
    )
    assert float(out["per_column"][3]) == pytest.approx(0.0, abs=1e-7)


def test_legacy_loss_unchanged_regression() -> None:
    payload = _payload()
    out = column_residual_loss(
        payload["delta"], base_exog=payload["base"], base_present=payload["present"],
        real_exog=payload["real"], real_present=payload["present"],
        scale=TARGET_LOSS_SCALE,
    )
    # 逐列 = 掩码 smooth_l1（scale 归一）均值；loss = 逐列均值（旧语义）
    expected = []
    for column in range(4):
        diff = (
            payload["base"][:, column] + payload["delta"][:, column] - payload["real"][:, column]
        ) / TARGET_LOSS_SCALE[column]
        expected.append(float(torch.nn.functional.smooth_l1_loss(diff, torch.zeros_like(diff))))
    assert float(out["loss"].item()) == pytest.approx(sum(expected) / 4.0, rel=1e-6)
    for column in range(4):
        assert float(out["per_column"][column]) == pytest.approx(expected[column], rel=1e-6)
