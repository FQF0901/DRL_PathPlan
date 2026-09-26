"""轨迹度量单位/口径回归测试（2026-09-26）。

背景：``bc_trajectory_loss`` 旧第二返回值是**未加权 MSE（m²）**却被命名为 ``mae``，
epoch 日志的 ``mae=`` 随之误导（h6 的 5.50 实为 m²，真实 MAE ≈ 2.3 m）。本文件锁定
修正后的口径契约：

- ``traj_mse_weighted`` = 加权 MSE（m²），与 ``loss_type`` 无关；
- ``traj_mae_weighted_m`` = 加权 MAE（m），与 ``loss_type`` 无关；
  误差幅度齐一时二者满足 ``MAE = sqrt(MSE)``，一般情形 ``MAE ≤ sqrt(MSE)``（Jensen）；
- ``train_weight=0`` 的被过滤帧不进入任何加权值；
- ``traj_mae_all_m`` 是**未加权**诊断口径，含被过滤帧。
"""

from __future__ import annotations

import math

import pytest
import torch

from pipeline.trainer import bc_trajectory_loss

#: 被测样本权重：样本 0/3 是 ``train_weight=0`` 的"被过滤帧"（近崩溃/离道，误差极大）
FILTERED_WEIGHTS = torch.tensor([0.0, 1.0, 1.0, 0.0])
UNIFORM_MAGNITUDES = [10.0, 2.0, 2.0, 10.0]


def _constant_error_pair(magnitudes: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """构造每样本所有 ``(K,2)`` 元素误差幅度相同的样本：``MAE_i = |c_i|``，``MSE_i = c_i²``。"""
    pred = torch.zeros(len(magnitudes), 3, 2)
    for index, value in enumerate(magnitudes):
        pred[index] = float(value)
    return pred, torch.zeros_like(pred)


def test_weighted_mae_equals_sqrt_mse_and_zero_weight_excluded() -> None:
    """l2 下误差幅度齐一：加权 MAE = sqrt(加权 MSE)；w=0 帧（误差 10 m）不进入加权值。"""
    pred, target = _constant_error_pair(UNIFORM_MAGNITUDES)
    loss, metrics = bc_trajectory_loss(pred, target, loss_type="l2", weights=FILTERED_WEIGHTS)

    assert float(loss) == pytest.approx(4.0)  # 加权 MSE = (2²+2²)/2
    assert metrics["traj_mse_weighted"] == pytest.approx(4.0)
    assert metrics["traj_mae_weighted_m"] == pytest.approx(2.0)
    assert metrics["traj_mae_weighted_m"] == pytest.approx(math.sqrt(metrics["traj_mse_weighted"]))
    # 未加权诊断口径含被过滤帧：(10+2+2+10)/4 = 6 m，远大于加权 MAE —— 旧接口正是这个口径
    assert metrics["traj_mae_all_m"] == pytest.approx(6.0)
    assert metrics["traj_mae_all_m"] > metrics["traj_mae_weighted_m"]


def test_metrics_are_loss_type_independent_and_loss_matches_units() -> None:
    """``loss`` 随 ``loss_type`` 变（l2→MSE m²，l1→MAE m），三个诊断指标两种口径一致。"""
    pred, target = _constant_error_pair(UNIFORM_MAGNITUDES)
    loss_l2, metrics_l2 = bc_trajectory_loss(pred, target, loss_type="l2", weights=FILTERED_WEIGHTS)
    loss_l1, metrics_l1 = bc_trajectory_loss(pred, target, loss_type="l1", weights=FILTERED_WEIGHTS)

    assert float(loss_l2) == pytest.approx(4.0)
    assert float(loss_l1) == pytest.approx(2.0)  # 加权 MAE（米）
    assert float(loss_l1) == pytest.approx(metrics_l2["traj_mae_weighted_m"])
    for key in ("traj_mse_weighted", "traj_mae_weighted_m", "traj_mae_all_m"):
        assert metrics_l1[key] == pytest.approx(metrics_l2[key]), f"{key} 不应随 loss_type 改变"


def test_jensen_bound_when_error_magnitudes_vary() -> None:
    """一般情形：MAE ≤ sqrt(MSE)（等号只在误差幅度齐一时成立，sqrt 不能当 MAE 用）。"""
    pred, target = _constant_error_pair([1.0, 3.0])  # 无权重 → 加权=未加权
    loss, metrics = bc_trajectory_loss(pred, target, loss_type="l2")

    assert metrics["traj_mse_weighted"] == pytest.approx(5.0)  # (1²+3²)/2
    assert metrics["traj_mae_weighted_m"] == pytest.approx(2.0)  # (1+3)/2
    assert metrics["traj_mae_all_m"] == pytest.approx(2.0)
    assert metrics["traj_mae_weighted_m"] < math.sqrt(metrics["traj_mse_weighted"])
    assert float(loss) == pytest.approx(metrics["traj_mse_weighted"])


def test_all_zero_weights_stay_finite() -> None:
    """权重全 0（极端过滤）时加权指标兜底为 0 而不是 NaN；未加权诊断仍反映真实误差。"""
    pred, target = _constant_error_pair([2.0])
    loss, metrics = bc_trajectory_loss(pred, target, loss_type="l2", weights=torch.zeros(1))

    assert float(loss) == pytest.approx(0.0)
    assert metrics["traj_mse_weighted"] == pytest.approx(0.0)
    assert metrics["traj_mae_weighted_m"] == pytest.approx(0.0)
    assert metrics["traj_mae_all_m"] == pytest.approx(2.0)
