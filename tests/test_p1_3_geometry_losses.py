"""P1-3 三项损失（ld_line_margin / curvature / progress）与梯度校准探针单测（默认关）。

覆盖：
- 三项各自数值（直线/弯曲/越线/推进合成用例，含手算对照）；
- mask 语义（LD 无效槽、极小 ds、无效 route 行）；
- 默认关逐位不变（weight=0 → term=0、不落 metrics 键、总损失=旧组件和）；
- 梯度回传到 plan/traj 链（``requires_grad`` 链检查）；
- 探针输出结构（grad-norm / cosine / conflict 键）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.trainer import (
    SOLID_LINE_TYPE_IDS,
    Phase3Config,
    _phase3_loss_terms,
    geometry_grad_probe,
    ld_line_margin_loss,
    plan_curvature_loss,
    pretrain_bc_phase3,
    route_progress_loss,
)
from tests.test_phase3_safe_recipe import _future_fn_for
from tests.test_recovery_wiring import _run_stage3, _tiny_model, _write_window_dir
from tests.v2_synthetic import make_v2_arrays
from pipeline.trainer import BCDataset


def test_solid_line_type_ids_match_env() -> None:
    """``{2,3,6,7}`` 必须等于 env/obs/ld.py::LINE_TYPE_IDS 的 SOLID_* 四项（防漂移）。"""
    from env.obs.ld import LINE_TYPE_IDS

    derived = tuple(sorted(v for k, v in LINE_TYPE_IDS.items() if "SOLID" in str(k).upper()))
    assert tuple(sorted(SOLID_LINE_TYPE_IDS)) == derived == (2, 3, 6, 7)


# --------------------------------------------------------------------------- #
# ld_line_margin_loss
# --------------------------------------------------------------------------- #

def _ld_line(batch: int = 1, slots: int = 16, *, step: float = 2.0, n_valid: int = 5) -> torch.Tensor:
    """沿 x 轴、y=0 的中心线（前 n_valid 槽有效，其余零值）。"""
    ld = torch.zeros(batch, slots, 7)
    ld[..., 0] = torch.arange(slots).float() * step
    ld[..., 5] = 1.0  # left broken
    ld[..., 6] = 1.0  # right broken
    return ld


def _ld_mask(batch: int = 1, slots: int = 16, n_valid: int = 5) -> torch.Tensor:
    mask = torch.zeros(batch, slots)
    mask[:, :n_valid] = 1.0
    return mask


def test_ld_margin_on_lane_zero_and_far_positive() -> None:
    ld = _ld_line()
    mask = _ld_mask()
    on_lane = torch.zeros(1, 6, 2)
    on_lane[0, :, 0] = torch.linspace(0.0, 8.0, 6)  # x∈[0,8] 全在中心线上
    loss0, diag0 = ld_line_margin_loss(on_lane, ld, mask, target_margin_m=1.0)
    assert float(loss0) == pytest.approx(0.0, abs=1e-6)
    assert diag0["ld_plan_covered_ratio"] == pytest.approx(1.0)
    far = on_lane.clone()
    far[0, :, 1] = 5.0
    loss1, diag1 = ld_line_margin_loss(far, ld, mask, target_margin_m=1.0)
    assert float(loss1) > 1.0
    assert diag1["ld_margin_mean_m"] > diag0["ld_margin_mean_m"]


def test_ld_margin_masked_slot_is_ignored() -> None:
    ld = _ld_line()
    mask = _ld_mask()
    plan = torch.zeros(1, 6, 2)
    plan[0, :, 0] = torch.linspace(0.0, 8.0, 6)
    plan[0, :, 1] = 2.0  # 距中心线 2m > target 1m → base > 0
    base, _ = ld_line_margin_loss(plan, ld, mask, target_margin_m=1.0)
    assert float(base) > 0.5
    # 近处"陷阱"槽（靠近 plan 路径、mask=0）→ 不生效
    trap = ld.clone()
    trap[0, 10, :2] = torch.tensor([4.0, 0.5])
    masked, _ = ld_line_margin_loss(plan, trap, mask, target_margin_m=1.0)
    assert float(masked) == pytest.approx(float(base), abs=1e-6)
    # 同一槽置为有效（mask=1）→ 生效（最近距被拉近 → surface 下降）
    mask2 = mask.clone()
    mask2[0, 10] = 1.0
    active, _ = ld_line_margin_loss(plan, trap, mask2, target_margin_m=1.0)
    assert float(active) < float(base) - 0.05


def test_ld_margin_solid_crossing_penalty() -> None:
    ld = _ld_line()
    mask = _ld_mask()
    plan = torch.zeros(1, 6, 2)
    plan[0, :, 0] = torch.arange(6).float() * 2.0
    plan[0, :, 1] = 2.5  # 横向 2.5 > solid_halfwidth 1.75
    # 大 target → surface/coverage 为 0，隔离 solid 项：relu(2.5−1.75)=0.75
    solid = ld.clone()
    solid[..., 5] = 2.0  # left solid（面向侧：plan 在左 → left）
    loss_solid, diag = ld_line_margin_loss(
        plan, solid, mask, target_margin_m=10.0, solid_weight=1.0
    )
    assert float(loss_solid) == pytest.approx(0.75, abs=1e-5)
    assert diag["ld_solid_active_ratio"] == pytest.approx(1.0)
    # 破折线（id=1）→ solid 项 0
    broken = ld.clone()
    loss_broken, diag_b = ld_line_margin_loss(
        plan, broken, mask, target_margin_m=10.0, solid_weight=1.0
    )
    assert float(loss_broken) == pytest.approx(0.0, abs=1e-6)
    assert diag_b["ld_solid_active_ratio"] == pytest.approx(0.0)
    # 横向在自由半宽内 → 0
    inside = plan.clone()
    inside[0, :, 1] = 1.0
    loss_inside, _ = ld_line_margin_loss(
        inside, solid, mask, target_margin_m=10.0, solid_weight=1.0
    )
    assert float(loss_inside) == pytest.approx(0.0, abs=1e-6)


def test_ld_margin_grad_flows_to_traj() -> None:
    ld = _ld_line()
    mask = _ld_mask()
    traj = torch.zeros(1, 6, 2, requires_grad=True)
    with torch.no_grad():
        traj[0, :, 0] = torch.arange(6).float() * 2.0
        traj[0, :, 1] = 2.5
    solid = ld.clone()
    solid[..., 5] = 2.0
    loss, _ = ld_line_margin_loss(traj, solid, mask, target_margin_m=10.0)
    loss.backward()
    assert traj.grad is not None and float(traj.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# plan_curvature_loss
# --------------------------------------------------------------------------- #

def test_curvature_straight_zero_and_hand_computed_zigzag() -> None:
    ds = torch.full((1, 6), 2.0)
    straight = torch.stack([ds, torch.zeros(1, 6)], dim=-1)
    loss0, diag0 = plan_curvature_loss(straight)
    assert float(loss0) == pytest.approx(0.0, abs=1e-9)
    assert diag0["curv_kappa_abs_mean"] == pytest.approx(0.0)
    # κ = [0,0,0.2,0,0,0]（dθ=0.4/2m）→ Δκ 对 = [0,0.2,0.2,0,0] → relu(Δκ−0.1) 均值 = 0.04
    dth = torch.zeros(1, 6)
    dth[0, 2] = 0.4
    zigzag = torch.stack([ds, dth], dim=-1)
    loss1, diag1 = plan_curvature_loss(zigzag, kappa_thresh=0.2, dk_thresh=0.1, strong_weight=2.0)
    assert float(loss1) == pytest.approx(0.04, abs=1e-6)
    assert diag1["curv_over_dk_ratio"] == pytest.approx(2.0 / 5.0)
    assert diag1["curv_over_kappa_ratio"] == pytest.approx(0.0)
    # 超阈值 κ：dθ=0.6 → κ=0.3 → over=0.1×2.0=0.2；Δκ 对=[0,0.3,0.3,0,0]→relu 均值 0.08 → 0.28
    dth2 = torch.zeros(1, 6)
    dth2[0, 2] = 0.6
    sharp = torch.stack([ds, dth2], dim=-1)
    loss2, diag2 = plan_curvature_loss(sharp, kappa_thresh=0.2, dk_thresh=0.1, strong_weight=2.0)
    # Δκ 项 = 0.08；超阈值 κ 强罚 = 2.0 × (relu(0.3−0.2)/6) = 0.0333…
    assert float(loss2) == pytest.approx(0.08 + 2.0 * (0.1 / 6.0), abs=1e-6)
    assert diag2["curv_over_kappa_ratio"] == pytest.approx(1.0 / 6.0)


def test_curvature_min_ds_mask() -> None:
    ds = torch.full((1, 6), 2.0)
    ds[0, 2] = 0.01  # 极小 ds → 该步 κ/Δκ 掩码
    dth = torch.zeros(1, 6)
    dth[0, 2] = 1.0  # 巨大 dθ 若未掩码会爆
    plan = torch.stack([ds, dth], dim=-1)
    loss, diag = plan_curvature_loss(plan, min_ds_m=0.1)
    assert float(loss) == pytest.approx(0.0, abs=1e-9)
    assert diag["curv_valid_ds_ratio"] == pytest.approx(5.0 / 6.0)
    # 掩码对相邻 Δκ 对：只有 (1,2),(2,3) 对被掩 → 无其它 κ
    assert diag["curv_dk_abs_mean"] == pytest.approx(0.0)


def test_curvature_grad_flows_to_plan() -> None:
    plan = torch.zeros(1, 6, 2, requires_grad=True)
    with torch.no_grad():
        plan[0, :, 0] = 2.0
        plan[0, 2, 1] = 0.6
    loss, _ = plan_curvature_loss(plan)
    loss.backward()
    assert plan.grad is not None and float(plan.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# route_progress_loss
# --------------------------------------------------------------------------- #

def _route_world(batch: int = 1, points: int = 10, step: float = 3.0) -> torch.Tensor:
    route = torch.zeros(batch, points, 2)
    route[..., 0] = torch.arange(points).float() * step
    return route


def test_progress_good_zero_and_slow_positive() -> None:
    route = _route_world()
    mask = torch.ones(1, 10)
    ego = torch.zeros(1, 3)
    good = torch.zeros(1, 6, 2)
    good[0, :, 0] = torch.linspace(1.0, 12.0, 6)  # 末点 12m ≥ target 8
    loss0, diag0 = route_progress_loss(good, ego, route, mask, target_m=8.0)
    assert float(loss0) == pytest.approx(0.0, abs=1e-6)
    assert diag0["progress_mean_m"] > 8.0
    slow = good * 0.15  # 末点 1.8m < 8 → hinge ≈ 6.2
    loss1, diag1 = route_progress_loss(slow, ego, route, mask, target_m=8.0)
    assert float(loss1) > 5.0
    assert diag1["progress_shortfall_ratio"] == pytest.approx(1.0)


def test_progress_world_frame_transform() -> None:
    """ego 在 (5,0,θ=0)，route 世界系 x=5..32：ego 系末点 7m → 世界 12m → 推进 ≈7。"""
    route = _route_world()
    route[..., 0] = route[..., 0] + 5.0
    mask = torch.ones(1, 10)
    ego = torch.tensor([[5.0, 0.0, 0.0]])
    plan = torch.zeros(1, 6, 2)
    plan[0, :, 0] = torch.linspace(1.0, 7.0, 6)
    loss, diag = route_progress_loss(plan, ego, route, mask, target_m=8.0)
    assert diag["progress_mean_m"] == pytest.approx(7.0, abs=0.5)
    assert float(loss) == pytest.approx(8.0 - 7.0, abs=0.5)


def test_progress_invalid_route_rows_excluded() -> None:
    route = _route_world(batch=2)
    mask = torch.ones(2, 10)
    mask[1] = 0.0  # 第 2 行无有效 route
    ego = torch.zeros(2, 3)
    plan = torch.zeros(2, 6, 2)
    plan[..., 0] = torch.linspace(1.0, 2.0, 6)  # 慢 → hinge > 0
    loss, diag = route_progress_loss(plan, ego, route, mask, target_m=8.0)
    assert diag["progress_route_valid_ratio"] == pytest.approx(0.5)
    # 无效行不进均值 → loss 等于有效行的 hinge（末点 2m → 6.0）
    assert float(loss) == pytest.approx(6.0, abs=0.5)


# --------------------------------------------------------------------------- #
# 默认关：term=0、不落键、总损失=旧组件和
# --------------------------------------------------------------------------- #

class _StubModel(torch.nn.Module):
    def __init__(self, batch: int = 4) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(0)
        self.traj_xy = torch.nn.Parameter(torch.randn(batch, 6, 2, generator=gen))
        self.plan = torch.nn.Parameter(torch.randn(batch, 6, 2, generator=gen))
        self.mu = torch.nn.Parameter(torch.randn(batch, 2, generator=gen))

    def forward(self, obs, *, rollout=True, world_model=False, **kwargs):  # noqa: D102 - stub
        return {"traj_xy": self.traj_xy, "plan": self.plan, "action_mu": self.mu}


def _stub_bundle(batch: int = 4, *, with_geometry_obs: bool = True):
    model = _StubModel(batch)
    gen = torch.Generator().manual_seed(1)
    targets = {
        "action": torch.randn(batch, 6, 2, generator=gen),
        "traj6": torch.randn(batch, 6, 2, generator=gen),
        "train_weight": torch.ones(batch),
    }
    future_t = {
        "action_chain": targets["action"],
        "action_chain_valid": torch.ones(batch, 6),
    }
    obs = {}
    if with_geometry_obs:
        route = torch.zeros(batch, 10, 2)
        route[..., 0] = torch.arange(10).float() * 3.0
        obs = {
            "ld": torch.zeros(batch, 16, 7),
            "ld_mask": torch.ones(batch, 16),
            "ego_world": torch.zeros(batch, 3),
            "route_world": route,
            "route_world_mask": torch.ones(batch, 10),
        }
    return model, obs, future_t, targets


def _zero_cfg(**overrides) -> Phase3Config:
    base = dict(
        bias_calib_weight=0.0, ego_next_weight=0.0, od_weight=0.0, ld_weight=0.0,
        latent_weight=0.0, presence_weight=0.0, entry_weight=0.0,
        anchor_ce_weight=0.0, anchor_wta_weight=0.0,
    )
    base.update(overrides)
    return Phase3Config(**base)


def test_default_off_terms_are_zero_and_raw_empty() -> None:
    model, obs, future_t, targets = _stub_bundle()
    cfg = _zero_cfg()
    terms = _phase3_loss_terms(model, obs, future_t, targets, targets["train_weight"], cfg)
    for key in ("ld_margin", "curvature", "progress"):
        assert torch.equal(terms[key], torch.zeros(()))
    assert terms["_raw"] == {}
    assert terms["geom_diag"] == {}
    # 总损失 = 旧组件和（三项贡献 0）
    total = terms["action"] + terms["action_chain"] + terms["bias_calib"] + terms["anchor_ce"]
    total = total + terms["anchor_wta"] + terms["ego_next"] + terms["od"] + terms["ld"]
    total = total + terms["latent"] + terms["presence"] + terms["entry"] + terms["load_balance"]
    for key in ("ld_margin", "curvature", "progress"):
        total = total + terms[key]
    assert torch.equal(total, terms["action"] + terms["action_chain"])


def test_weight_positive_terms_positive_and_diag() -> None:
    model, obs, future_t, targets = _stub_bundle()
    # 让 curvature/progress 有非零项：plan 弯曲 + 慢推进
    with torch.no_grad():
        model.plan[0, :, 0] = 2.0
        model.plan[0, 2, 1] = 0.6
        model.traj_xy.zero_()
    cfg = _zero_cfg(
        ld_margin_weight=0.1, curvature_weight=0.1, progress_weight=0.1,
        progress_target_m=8.0,
    )
    terms = _phase3_loss_terms(model, obs, future_t, targets, targets["train_weight"], cfg)
    assert float(terms["curvature"]) > 0.0
    assert float(terms["progress"]) > 0.0
    assert set(terms["geom_diag"]) == {"ld_margin", "curvature", "progress"}
    # 未乘权重的原始项在 _raw 中（仅 probe_terms）
    assert terms["_raw"] == {}


def test_missing_obs_channels_fail_closed() -> None:
    model, _, future_t, targets = _stub_bundle(with_geometry_obs=False)
    cfg = _zero_cfg(ld_margin_weight=0.1)
    with pytest.raises(ValueError, match="obs 通道"):
        _phase3_loss_terms(model, {}, future_t, targets, targets["train_weight"], cfg)


# --------------------------------------------------------------------------- #
# 梯度探针结构
# --------------------------------------------------------------------------- #

def test_geometry_grad_probe_structure() -> None:
    model, obs, future_t, targets = _stub_bundle()
    with torch.no_grad():
        model.plan[0, :, 0] = 2.0
        model.plan[0, 2, 1] = 0.6
        model.traj_xy[0, :, 0] = torch.linspace(1.0, 2.0, 6)
    cfg = _zero_cfg()
    probe = geometry_grad_probe(model, obs, future_t, targets, targets["train_weight"], cfg)
    for key in (
        "grad_norm_traj", "grad_norm_ld_margin", "grad_norm_curvature", "grad_norm_progress",
        "cos_traj_geometry", "cos_traj_curvature", "cos_progress_geometry",
        "conflict_traj_geometry", "conflict_traj_curvature", "conflict_progress_geometry",
    ):
        assert key in probe, key
    assert np.isfinite(probe["grad_norm_curvature"]) and probe["grad_norm_curvature"] > 0.0
    assert np.isfinite(probe["grad_norm_progress"]) and probe["grad_norm_progress"] > 0.0
    assert isinstance(probe["conflict_traj_geometry"], bool)
    # 探针不改 .grad（纯诊断）
    assert all(parameter.grad is None for parameter in model.parameters())


# --------------------------------------------------------------------------- #
# 端到端 tiny phase3：默认关不落键；开启落键且有限
# --------------------------------------------------------------------------- #

def _tiny_dataset() -> BCDataset:
    arrays, _ = make_v2_arrays(episodes=4, steps_per_episode=6)
    n = int(arrays["episode_id"].size)
    arrays["ego_world"] = arrays["pose"].astype(np.float32)[:, None, :]
    arrays["ego_world_mask"] = np.ones((n, 1), dtype=np.float32)
    arrays["route_world"] = np.zeros((n, 64, 2), dtype=np.float32)
    arrays["route_world_mask"] = np.ones((n, 64), dtype=np.float32)
    return BCDataset(arrays, {"schema_version": 2, "label_names": None})


def _run_tiny(*, cfg_kwargs: dict, max_batches: int = 1) -> dict:
    dataset = _tiny_dataset()
    model = _tiny_model()
    config = Phase3Config(
        epochs=1, batch_size=24, micro_batch_size=None, device="cpu", shuffle=False, seed=0,
        val_indices=None, max_batches=max_batches, freeze_mode="all", freeze_prefixes=(),
        **cfg_kwargs,
    )
    return pretrain_bc_phase3(
        model, dataset, config, logger=lambda _: None, future_fn=_future_fn_for(dataset)
    )


def test_tiny_phase3_default_off_no_new_metrics() -> None:
    metrics = _run_tiny(cfg_kwargs={})
    for key in ("bc_ld_margin_loss", "bc_curvature_loss", "bc_progress_loss"):
        assert key not in metrics
    assert not any(str(key).startswith("geom_") for key in metrics)
    old_sum = (
        metrics["bc_action_loss"] + metrics["bc_action_chain_loss"] + metrics["bc_bias_calib_loss"]
        + metrics["bc_ego_next_loss"] + metrics["bc_od_loss"] + metrics["bc_ld_loss"]
        + metrics["bc_latent_loss"] + metrics["bc_presence_loss"] + metrics["bc_entry_loss"]
        + metrics["bc_load_balance_loss"]
    )
    # 组件逐项 float32 求和 vs 总损失 float32 单次求和 → 容差按 float32 量级
    assert float(metrics["bc_loss"]) == pytest.approx(old_sum, rel=1e-5)


def test_tiny_phase3_new_terms_and_probe_enabled() -> None:
    metrics = _run_tiny(
        cfg_kwargs=dict(
            ld_margin_weight=0.1, curvature_weight=0.1, progress_weight=0.1,
            progress_target_m=20.0,  # 保证 hinge 激活（tiny 模型 plan 远小于 20m）
            geom_grad_probe=True, geom_grad_probe_every=1, geom_grad_probe_batch=8,
        ),
        max_batches=2,
    )
    for key in ("bc_ld_margin_loss", "bc_curvature_loss", "bc_progress_loss"):
        assert key in metrics and np.isfinite(float(metrics[key]))
    assert float(metrics["bc_progress_loss"]) > 0.0
    assert float(metrics["bc_ld_margin_loss"]) >= 0.0 and float(metrics["bc_curvature_loss"]) >= 0.0
    assert any(str(key).startswith("geom_") for key in metrics)
    assert metrics["geom_grad_probe"], "探针必须落记录"
    record = metrics["geom_grad_probe"][0]
    assert "grad_norm_curvature" in record and "cos_traj_geometry" in record
    assert "geom_grad_conflict_counts" in metrics


# --------------------------------------------------------------------------- #
# stages 接线（CLI → Phase3Config → 损失/metrics）
# --------------------------------------------------------------------------- #

def test_stages_cli_wiring_and_default_off(tmp_path) -> None:
    window_dir, _ = _write_window_dir(tmp_path, episodes=4, steps_per_episode=6)
    metrics_on = _run_stage3(
        tmp_path, window_dir,
        extra=[
            "--phase3-ld-margin-weight", "0.1",
            "--phase3-curvature-weight", "0.1",
            "--phase3-progress-weight", "0.1",
            "--phase3-geom-grad-probe",
            "--phase3-geom-grad-probe-batch", "8",
        ],
        out_name="out_p13_on",
    )
    for key in ("bc_ld_margin_loss", "bc_curvature_loss", "bc_progress_loss"):
        assert key in metrics_on, key
    assert metrics_on["geom_grad_probe"]
    assert float(metrics_on["losses"]["ld_margin"]) == pytest.approx(0.1)

    metrics_off = _run_stage3(tmp_path, window_dir, extra=[], out_name="out_p13_off")
    for key in ("bc_ld_margin_loss", "bc_curvature_loss", "bc_progress_loss"):
        assert key not in metrics_off
    assert "geom_grad_probe" not in metrics_off
    assert float(metrics_off["losses"]["ld_margin"]) == 0.0
