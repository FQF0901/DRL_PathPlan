"""R41 第二批-F：**横向训练栈守卫**（数据侧 κ/掩码/速度桶/跨帧对 + loss 侧 + LQR + 锚修正）。

自包含（fixture ``tests/fixtures/residual_mini`` + 合成数组；CPU；零研究工作区依赖）。
锁定口径（与 ``pipeline/lateral_*.py`` docstring 逐条对应）：
1. κ 目标 = ``dθ/ds``（ds < ds_min ⇒ 0 + mask 0）；arc = 段末端累计；
2. 掩码 = 步有效 ∧ 锚视界（arc_k ≤ horizon − margin）；站位插值分段常数、越计划末弧 ⇒ 0；
3. 速度桶权重（桶界 = KAPPA_SPEED_BUCKETS；有效行归一化均值 1；缺速度中性 1）；
4. 跨帧对 = 连续决策窗（t → t+5 且两帧在可训练集）；Warp = κ_prev@(s_curr + ΔS)；门控权重
   （位移/安全/位姿对齐）；
5. 固定 LQR = ``env.tracking`` 反馈律离线复刻（DARE 对拍；metadrive 可用时）；
6. loss 默认关（全 0 项不建图）；掩码加权值 = 手算；L_replan = w·|κ_t − gather(κ_prev)|；
7. 锚修正 off 逐位；resample 点距 ≥ min_ds 且 1/Δs² 噪声增益显著下降；噪声增强 σ 正确；
8. arm 预置 P0–P4 增量堆叠。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/residual_mini"

from pipeline import lateral_anchor, lateral_labels, lateral_losses, lateral_lqr, lateral_replan  # noqa: E402


# --------------------------------------------------------------------------- #
# 0) 常量单源 / 数据侧基础
# --------------------------------------------------------------------------- #
def test_constants_single_source() -> None:
    """站位/桶界/步数 = ``net.path_head`` 单一出处（防止两处漂移）。"""
    import net.path_head as ph

    assert lateral_labels.KAPPA_STATIONS == tuple(ph.KAPPA_CTRL_S)
    assert lateral_labels.SPEED_BUCKET_EDGES == tuple(ph.KAPPA_SPEED_BUCKETS)
    assert lateral_labels.PLAN_STEPS == int(ph.NUM_STEPS)


def test_plan_targets_kappa_formula_and_step_valid() -> None:
    action = np.zeros((3, 6, 2), dtype=np.float64)
    action[0] = np.tile([2.0, 0.02], (6, 1))
    action[1, 0] = [0.05, 0.01]  # ds < ds_min ⇒ 无效
    action[1, 1:] = np.tile([1.0, -0.03], (5, 1))
    action[2, 0] = [1.0, np.nan]  # 非有限 ⇒ 无效
    action[2, 1:] = np.tile([1.0, 0.01], (5, 1))
    out = lateral_labels.plan_targets_from_action(action)
    assert np.allclose(out["kappa"][0], 0.01)
    assert out["step_valid"][0].all()
    assert not out["step_valid"][1, 0] and out["kappa"][1, 0] == 0.0
    assert np.allclose(out["kappa"][1, 1:], -0.03)
    assert not out["step_valid"][2, 0] and out["kappa"][2, 0] == 0.0
    assert np.allclose(out["arc"][0], np.arange(1, 7) * 2.0)
    assert out["kappa"].shape == (3, 6)


def test_plan_step_masks_anchor_horizon_edges() -> None:
    step_valid = np.ones((2, 6), dtype=bool)
    arc = np.tile(np.arange(1, 7, dtype=np.float64) * 2.0, (2, 1))  # 2..12
    horizon = np.array([6.0, 10.0])
    mask = lateral_labels.plan_step_masks(step_valid, arc, horizon)
    assert mask[0].tolist() == [1, 1, 1, 0, 0, 0]  # arc<=6（含端点）
    assert mask[1].tolist() == [1, 1, 1, 1, 1, 0]
    margin = lateral_labels.plan_step_masks(step_valid, arc, horizon, horizon_margin_m=2.0)
    assert margin[0].tolist() == [1, 1, 0, 0, 0, 0]
    invalid = step_valid.copy()
    invalid[1, 0] = False
    mask2 = lateral_labels.plan_step_masks(invalid, arc, horizon)
    assert mask2[1, 0] == 0.0


def test_station_targets_piecewise_constant_and_beyond_plan_mask() -> None:
    kappa = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
    arc = np.array([[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]])
    mask = np.ones((1, 6), dtype=np.float32)
    target, out_mask = lateral_labels.station_targets(
        kappa, arc, mask, stations=(1.0, 3.0, 5.0, 11.0, 12.0, 13.0)
    )
    assert target[0, :5].tolist() == [0.1, 0.2, 0.3, 0.6, 0.6]
    assert out_mask[0, :5].tolist() == [1, 1, 1, 1, 1]
    assert out_mask[0, 5] == 0.0  # 超计划末弧 ⇒ 不外推
    mask[0, 2] = 0.0
    target2, mask2 = lateral_labels.station_targets(kappa, arc, mask, stations=(5.0,))
    assert mask2[0, 0] == 0.0 and target2[0, 0] == 0.0


def test_speed_bucket_weights_boundaries_normalize_and_invalid() -> None:
    edges = lateral_labels.SPEED_BUCKET_EDGES
    speed = np.array([0.0, 2.99, 3.0, 5.99, 6.0, 8.99, 9.0, 11.99, 12.0, np.nan])
    ids = lateral_labels.speed_bucket_ids(speed)
    assert ids.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, -1]
    present = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=np.float64)
    weights, ids2 = lateral_labels.speed_bucket_weights(speed, present, edges=edges, normalize=False)
    table = np.asarray(lateral_labels.SPEED_BUCKET_WEIGHTS_DEFAULT)
    assert np.allclose(weights[:9], table[ids2[:9]])
    assert weights[9] == 1.0  # present=0 ⇒ 中性
    weights_n, _ = lateral_labels.speed_bucket_weights(speed, present, edges=edges, normalize=True)
    assert abs(float(weights_n[:9].mean()) - 1.0) < 1e-6
    assert weights_n[9] == 1.0


# --------------------------------------------------------------------------- #
# 1) 跨帧重规划对 + warp + 门控
# --------------------------------------------------------------------------- #
def _synthetic_replan_inputs():
    n = 3
    kappa = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]), (n, 1))
    arc = np.tile(np.arange(1, 7, dtype=np.float64) * 4.0, (n, 1))  # 4..24
    step_mask = np.ones((n, 6), dtype=np.float32)
    present = np.zeros((n, 8), dtype=np.float64)
    present[:, 0] = 1.0
    exog = np.zeros((n, 8), dtype=np.float64)
    main_ids = np.array([0, 1], dtype=np.int64)
    succ_ids = np.array([1, 2], dtype=np.int64)
    return kappa, arc, step_mask, present, exog, main_ids, succ_ids


def test_replan_pairs_ds_exec_and_warp_index() -> None:
    kappa, arc, step_mask, present, exog, main_ids, succ_ids = _synthetic_replan_inputs()
    rows = {"ego_world": np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.02], [8.0, 0.0, 0.05]])}
    pairs = lateral_replan.build_replan_pairs(
        rows, present, exog, main_ids, succ_ids, kappa=kappa, arc=arc, step_mask=step_mask
    )
    assert pairs["prev_ids"].tolist() == [0]
    assert pairs["curr_ids"].tolist() == [1]
    assert float(pairs["ds_exec"][0]) == pytest.approx(4.0)
    assert float(pairs["delta_psi"][0]) == pytest.approx(0.02)
    # query = s_k + ΔS（=4）⇒ 分段常数映射（段界含端点）
    index = pairs["warp_index"][0]
    assert index.tolist()[:5] == [1, 2, 3, 4, 5]
    assert index[5] == -1  # 超上一帧计划末弧 ⇒ 无效
    warped, wmask = lateral_replan.apply_warp(pairs["warp_index"], kappa[pairs["prev_row_ids"]])
    assert wmask[0].tolist() == [1, 1, 1, 1, 1, 0]
    assert warped[0, :5].tolist() == [0.2, 0.3, 0.4, 0.5, 0.6]
    assert warped[0, 5] == 0.0


def test_replan_weights_gates_monotonic() -> None:
    # 安全收缩 ⇒ gate 下降；位移增大 ⇒ w_ds 下降；航向失配 ⇒ g_align 下降
    # 行 0/2/4 = prev，行 1/3/5 = curr；对 = (0→1), (2→3), (4→5)（succ 有效仅此三处）
    present = np.zeros((6, 8), dtype=np.float64)
    present[:, [1, 2, 3, 4, 5]] = 1.0
    exog = np.zeros((6, 8), dtype=np.float64)
    exog[[0, 2, 4], 1], exog[[0, 2, 4], 2] = 5.0, 5.0
    exog[[0, 2, 4], 3], exog[[0, 2, 4], 4], exog[[0, 2, 4], 5] = 30.0, 30.0, 20.0
    exog[1] = exog[0]                     # 无变化
    exog[3] = exog[2].copy(); exog[3, 1] = 5.0 - 3.5   # width shrink 3.5 ⇒ Λ=1
    exog[5] = exog[4].copy(); exog[5, 3] = 30.0 - 50.0  # static drop 50 ⇒ Λ=1（clip）
    ego_world = np.array(
        [
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],   # 对 1
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],   # 对 2
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.3],   # 对 3（Δψ=0.3）
        ]
    )
    rows = {"ego_world": ego_world}
    kappa = np.zeros((6, 6))
    arc = np.tile(np.arange(1, 7, dtype=np.float64) * 4.0, (6, 1))
    mask = np.ones((6, 6), dtype=np.float32)
    pairs = lateral_replan.build_replan_pairs(
        rows, present, exog, np.arange(6), np.array([1, -1, 3, -1, 5, -1]),
        kappa=kappa, arc=arc, step_mask=mask,
    )
    assert pairs["prev_ids"].tolist() == [0, 2, 4]
    assert pairs["curr_ids"].tolist() == [1, 3, 5]
    gates = pairs["gate_safety"]
    assert gates[0] == pytest.approx(1.0)
    assert gates[1] == pytest.approx(np.exp(-1.0))
    assert gates[2] == pytest.approx(np.exp(-1.0))
    w = pairs["w_replan"]
    base = np.exp(-2.0 / lateral_replan.DS_REF_M)
    assert w[0] > w[1]
    assert float(w[0]) == pytest.approx(base * np.exp(0.0))
    assert float(w[1]) == pytest.approx(base * np.exp(-1.0))
    # 航向失配门（对 3：Δψ=0.3 且 κ_prev(ego)=0 ⇒ mismatch=0.3）
    assert pairs["gate_align"][2] == pytest.approx(np.exp(-0.3 / lateral_replan.PSI_ALIGN_TAU_RAD))
    assert float(w[2]) == pytest.approx(
        base * np.exp(-1.0) * np.exp(-0.3 / lateral_replan.PSI_ALIGN_TAU_RAD)
    )


# --------------------------------------------------------------------------- #
# 2) 固定 LQR（离线复刻）
# --------------------------------------------------------------------------- #
def test_dare_gain_matches_env_tracking() -> None:
    pytest.importorskip("metadrive")
    from env.tracking import _solve_dare

    from pipeline.lateral_lqr import LqrSaturationConfig

    cfg = LqrSaturationConfig()
    v = 8.0
    a = np.array([[1.0, v * cfg.env_dt], [0.0, 1.0]])
    b = np.array([[0.0], [v * cfg.env_dt / cfg.wheelbase]])
    q = np.diag([cfg.q_lateral, cfg.q_heading]).astype(np.float64)
    r = np.array([[cfg.r_steer]])
    reference = _solve_dare(a, b, q, r, cfg.riccati_iters, cfg.riccati_tol)
    ours = lateral_lqr.dare_gain(a, b, q, r, iters=cfg.riccati_iters, tol=cfg.riccati_tol)
    assert np.allclose(ours, reference, atol=1e-12)
    schedule = lateral_lqr.SpeedGainSchedule(cfg)
    gain = schedule.gain(v)
    assert np.allclose(gain, reference, atol=1e-12)
    # 闭环稳定（A − B K 谱半径 < 1）
    closed = a - b @ gain
    assert float(np.max(np.abs(np.linalg.eigvals(closed)))) < 1.0


def test_lqr_saturation_monotonic_in_kappa_and_straight_zero() -> None:
    plan_straight = np.zeros((1, 6, 2))
    plan_straight[0, :, 0] = 4.0
    speed = np.array([8.0])
    straight = lateral_lqr.preview_steer_raw(plan_straight, speed)
    assert abs(straight["delta_raw"][0]) < 1e-9
    assert straight["sat"][0] < 1e-9
    saturations = []
    for kappa in (0.005, 0.02, 0.05):
        plan = np.zeros((1, 6, 2))
        plan[0, :, 0] = 4.0
        plan[0, :, 1] = kappa * 4.0
        saturations.append(float(lateral_lqr.preview_steer_raw(plan, speed)["sat"][0]))
    assert saturations[0] < saturations[1] < saturations[2]


def test_execution_gap_weights_formula() -> None:
    student = np.array([0.2, 0.5, 0.1, 1.0])
    expert = np.array([0.2, 0.5, 0.3, 0.0])
    out = lateral_lqr.execution_gap_weights(student, expert, valid=np.array([1, 1, 1, 0], dtype=bool))
    assert out["gap"].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert float(out["w"].max()) == 0.0  # valid 行无正 gap
    out2 = lateral_lqr.execution_gap_weights(
        np.array([0.4, 0.8]), np.array([0.2, 0.2]), valid=np.array([1, 1], dtype=bool)
    )
    assert np.allclose(out2["gap"], [0.2, 0.6])
    assert float(out2["w"].max()) == pytest.approx(1.0)  # p95 归一化后 cap=1
    assert float(out2["w"].min()) > 0.0


# --------------------------------------------------------------------------- #
# 3) 损失项（默认关 + 数值口径）
# --------------------------------------------------------------------------- #
def test_kappa_supervision_loss_masked_value() -> None:
    pred = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    target = torch.zeros_like(pred)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    out = lateral_losses.kappa_supervision_loss(pred, target, mask)
    manual = (torch.nn.functional.smooth_l1_loss(pred, target, beta=1.0, reduction="none") * mask).sum() / mask.sum()
    assert float(out["loss"]) == pytest.approx(float(manual))
    weights = torch.tensor([2.0, 1.0])
    out_w = lateral_losses.kappa_supervision_loss(pred, target, mask, sample_weights=weights)
    element = torch.nn.functional.smooth_l1_loss(pred, target, beta=1.0, reduction="none")
    manual_w = (element * mask * weights.reshape(-1, 1)).sum() / (mask * weights.reshape(-1, 1)).sum()
    assert float(out_w["loss"]) == pytest.approx(float(manual_w))
    empty = lateral_losses.kappa_supervision_loss(pred, target, torch.zeros_like(mask))
    assert float(empty["loss"]) == 0.0


def test_plan_delta_kappa_hinge_and_pair_mask() -> None:
    kappa = torch.tensor([[0.0, 0.05, 0.4], [0.0, 1.0, 1.0]])
    mask = torch.ones(2, 3)
    out = lateral_losses.plan_delta_kappa_loss(kappa, mask, dk_thresh=0.1)
    # 行 0 hinges: [0, 0.25]；行 1: [0.9, 0] ⇒ 均值 = 1.15/4
    assert float(out["loss"]) == pytest.approx(1.15 / 4.0)
    mask[0, 1] = 0.0
    out2 = lateral_losses.plan_delta_kappa_loss(kappa, mask, dk_thresh=0.1)
    # 步 1 失效 ⇒ 其两侧配对 (0,1)/(1,2) 同时失效；只余行 1 的 (0.9, 0) ⇒ 0.9/2
    assert float(out2["loss"]) == pytest.approx(0.9 / 2.0)


def test_replan_consistency_loss_gather_and_weight() -> None:
    kappa_t = torch.tensor([[0.0, 0.1, 0.2]])
    kappa_prev = torch.tensor([[0.5, 0.3, 0.1]])
    warp_index = torch.tensor([[2, 0, -1]])
    warp_mask = torch.tensor([[1.0, 1.0, 0.0]])
    weight = torch.tensor([0.5])
    out = lateral_losses.replan_consistency_loss(kappa_t, kappa_prev, warp_index, warp_mask, weight)
    expected = (abs(0.0 - 0.1) + abs(0.1 - 0.5)) / 2.0  # gather: [0.1, 0.5, *]
    assert float(out["loss"]) == pytest.approx(expected)
    zero = lateral_losses.replan_consistency_loss(kappa_t, kappa_prev, warp_index, torch.zeros_like(warp_mask), weight)
    assert float(zero["loss"]) == 0.0


def test_combine_sample_weights_product_clip_normalize() -> None:
    a = torch.tensor([1.0, 2.0, 0.0])
    b = torch.tensor([1.0, 1.0, 5.0])
    out = lateral_losses.combine_sample_weights([a, b], normalize=False)
    assert torch.allclose(out, torch.tensor([1.0, 2.0, 0.0]))
    out_clip = lateral_losses.combine_sample_weights([a, b], cap=1.5, normalize=False)
    assert torch.allclose(out_clip, torch.tensor([1.0, 1.5, 0.0]))
    out_norm = lateral_losses.combine_sample_weights([a, b], normalize=True)
    positive = out_norm > 0
    assert abs(float(out_norm[positive].mean()) - 1.0) < 1e-6
    assert float(out_norm[2]) == 0.0


def test_lateral_loss_terms_default_off_and_incremental_arms() -> None:
    b, s, k = 4, 3, 6
    kappa_pred = torch.randn(b, s, requires_grad=True)
    kappa_target = torch.zeros(b, s)
    kappa_mask = torch.ones(b, s)
    plan_pred = torch.zeros(b, k, 2, requires_grad=True)
    plan_pred.data[..., 0] = 4.0
    plan_pred.data[..., 1] = 0.1
    prev = torch.zeros(b, k, requires_grad=True)
    warp_index = torch.zeros(b, k, dtype=torch.long)
    warp_mask = torch.ones(b, k)
    warp_weight = torch.full((b,), 0.5)
    base = torch.ones(b)
    lqr_w = torch.zeros(b)
    p0 = lateral_losses.lateral_loss_terms(lateral_losses.config_for_arm("p0"))
    assert p0["terms"] == {} and float(p0["total"]) == 0.0
    p1 = lateral_losses.lateral_loss_terms(
        lateral_losses.config_for_arm("p1"),
        kappa_pred=kappa_pred, kappa_target=kappa_target, kappa_mask=kappa_mask,
        base_sample_weights=base,
    )
    assert set(p1["active"]) == {"kappa"} and float(p1["total"]) > 0.0
    p3 = lateral_losses.lateral_loss_terms(
        lateral_losses.config_for_arm("p3"),
        kappa_pred=kappa_pred, kappa_target=kappa_target, kappa_mask=kappa_mask,
        plan_pred=plan_pred, kappa_prev_pred=prev, warp_index=warp_index, warp_mask=warp_mask,
        warp_weight=warp_weight, base_sample_weights=base,
    )
    assert set(p3["active"]) == {"kappa", "replan", "speed_bucket"}
    p4 = lateral_losses.lateral_loss_terms(
        lateral_losses.config_for_arm("p4"),
        kappa_pred=kappa_pred, kappa_target=kappa_target, kappa_mask=kappa_mask,
        plan_pred=plan_pred, kappa_prev_pred=prev, warp_index=warp_index, warp_mask=warp_mask,
        warp_weight=warp_weight, base_sample_weights=base, lqr_weights=lqr_w,
    )
    assert "lqr_reweight" in p4["active"]
    with pytest.raises(SystemExit):
        lateral_losses.config_for_arm("p9")
    # p1 缺输入 ⇒ fail-fast
    with pytest.raises(SystemExit):
        lateral_losses.lateral_loss_terms(lateral_losses.config_for_arm("p1"))


def test_plan_kappa_from_plan_matches_data_caliber() -> None:
    plan = torch.zeros(2, 6, 2)
    plan[0, :, 0] = 2.0
    plan[0, :, 1] = 0.02
    plan[1, 0, 0] = 0.05  # 无效步
    plan[1, :, 1] = 0.03
    kappa, valid = lateral_losses.plan_kappa_from_plan(plan)
    assert np.allclose(kappa[0].numpy(), 0.01)
    assert valid[0].all()
    assert kappa[1, 0] == 0.0 and valid[1, 0] == 0.0


# --------------------------------------------------------------------------- #
# 4) 锚修正 + 噪声增强（默认关逐位；1/Δs² 增益下降）
# --------------------------------------------------------------------------- #
def test_anchor_correction_off_bitwise_and_resample_spacing() -> None:
    xy = torch.tensor([[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0], [2.0, 0.2]]])
    mask = torch.ones(1, 5)
    off_xy, off_mask = lateral_anchor.correct_anchor_tensor(xy, mask, mode="off")
    assert torch.equal(off_xy, xy) and torch.equal(off_mask, mask)
    on_xy, on_mask = lateral_anchor.correct_anchor_tensor(xy, mask, mode="resample", min_ds_m=2.0)
    points = on_xy[0, : int(on_mask[0].sum())].numpy()
    spacing = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert spacing.min() >= 2.0 - 1e-6
    assert np.allclose(points[0], xy[0, 0].numpy())
    assert np.allclose(points[-1], xy[0, -1].numpy())
    with pytest.raises(SystemExit):
        lateral_anchor.correct_anchor_tensor(xy, mask, mode="bogus")


def test_anchor_noise_augmentation_sigma_and_padding() -> None:
    xy = torch.zeros(1, 4, 2)
    xy[0, :, 0] = torch.arange(4, dtype=torch.float32) * 0.5
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    generator = torch.Generator().manual_seed(0)
    noisy, out_mask = lateral_anchor.perturb_anchor_tensor(xy, mask, sigma_m=0.03, generator=generator)
    assert torch.equal(out_mask, mask)
    assert torch.equal(noisy[0, 3], xy[0, 3])  # padding 点不动（原值保留）
    assert not torch.equal(noisy[0, :3], xy[0, :3])
    diff = (noisy[0, :3] - xy[0, :3]).numpy()
    assert abs(float(np.sqrt((diff ** 2).mean())) - 0.03) < 0.03  # 单点样本弱断言
    with pytest.raises(SystemExit):
        lateral_anchor.perturb_anchor_tensor(xy, mask, sigma_m=0.0)


def test_anchor_resample_reduces_kappa_noise_amplification() -> None:
    """2 cm 锚点噪声 → κ_ref 扰动：raw ≫ ≥2 m 重采样（C §6-4/§6-1 的 1/Δs² 增益证据）。"""
    from net.path_head import anchor_reference

    rng = np.random.default_rng(0)
    arc = np.arange(0.0, 30.0, 0.5)
    xy = np.stack([arc, 0.5 * np.sin(arc / 6.0)], axis=1).astype(np.float32)
    tensor = torch.as_tensor(xy)[None]
    mask = torch.ones(1, xy.shape[0])
    ref = anchor_reference(tensor, mask)
    generator = torch.Generator().manual_seed(1)
    noisy, noisy_mask = lateral_anchor.perturb_anchor_tensor(tensor, mask, sigma_m=0.02, generator=generator)
    raw = anchor_reference(noisy, noisy_mask)
    raw_delta = float((raw["kappa_ref"] - ref["kappa_ref"]).abs().max())
    corrected, corrected_mask = lateral_anchor.correct_anchor_tensor(noisy, noisy_mask, mode="resample", min_ds_m=2.0)
    ref_corrected = anchor_reference(corrected, corrected_mask)
    base_corrected = anchor_reference(
        *lateral_anchor.correct_anchor_tensor(tensor, mask, mode="resample", min_ds_m=2.0)
    )
    corrected_delta = float((ref_corrected["kappa_ref"] - base_corrected["kappa_ref"]).abs().max())
    assert raw_delta > 1e-3
    assert corrected_delta < 0.25 * raw_delta


# --------------------------------------------------------------------------- #
# 5) fixture 级数据装配（复用残差链配对）
# --------------------------------------------------------------------------- #
def _fixture_args(**overrides) -> Namespace:
    base = dict(
        data=str(FIXTURE), shards=None, max_pairs=None, seed=0, val_frac=0.25, chunk=64,
        horizon_margin_m=0.0, anchor_min_ds=2.0, anchor_noise_sigma_m=0.0,
    )
    base.update(overrides)
    return Namespace(**base)


def test_fixture_lateral_data_end_to_end() -> None:
    from pipeline.lateral_dataset import build_lateral_data

    data = build_lateral_data(_fixture_args())
    assert data["main_ids"].size > 0
    assert data["step_mask"].shape == (data["main_ids"].size, 6)
    assert data["station_target"].shape == (data["main_ids"].size, 3)
    assert data["speed_weight"].shape == (data["main_ids"].size,)
    assert data["replan"]["pair_valid"].size > 0
    # 掩码包含性：κ_far 站位（29.75m）在当前 6 步数据上几乎不可达 ⇒ 低/零有效数（结构性，报告登记）
    assert float(data["station_mask"][:, 0].mean()) > 0.5
    # 跨帧对：warp 逐列覆盖随步号单调不增（位移向前 ⇒ 远列先失效）
    rates = (data["replan"]["warp_index"] >= 0).mean(axis=0)
    assert np.all(np.diff(rates) <= 1e-9)


def test_runner_dry_run_and_check_features_pass(tmp_path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("train_lateral_runner", ROOT / "tools/train_lateral.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report_path = tmp_path / "dry_run.json"
    code = module.main(
        [
            "--dry-run", "--check-features", "--arm", "p0", "--device", "cpu",
            "--data", str(FIXTURE), "--max-pairs", "64", "--json-out", str(report_path),
        ]
    )
    assert code == 0
    import json

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["contract"]["pass"] is True
    assert report["contract"]["anchor_correction_off_bitwise"] is True
