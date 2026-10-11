"""P1-2 规格补口单测：phase3 traj（pose 链）损失项（默认关、逐位不变；掩码只作用于 traj）。

覆盖：
- ``_phase3_loss_terms`` 的 ``traj`` 项：weight=0 → 恒 0（掩码无关）；weight>0 → 与手算
  ``Σ w·mask·MSE / Σ w·mask`` 一致；全 0 掩码 → 0；None 掩码 → 全行；
- 端到端 tiny phase3：weight>0 时 ``bc_traj_aux_loss`` 落键且 >0、总损失差值 ≈ traj 项、
  traj 监控值（单 batch 同前向）与 weight=0 完全一致；weight=0 时 metrics 不落 traj 键；
- stages 接线：``--phase3-traj-aux-weight`` → Phase3Config/损失/metrics；无真实 traj 行
  （无锚/recovery）→ fail-closed；默认关闭 metrics 不变。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.trainer import (
    Phase3Config,
    _phase3_loss_terms,
    pretrain_bc_phase3,
)
from pipeline.trainer import BCDataset
from tests.test_phase3_safe_recipe import _future_fn_for
from tests.test_recovery_wiring import (
    _run_stage3,
    _tiny_model,
    _write_recovery_dir,
    _write_window_dir,
)
from tests.v2_synthetic import make_v2_arrays


class _StubPhase3Model(torch.nn.Module):
    """固定输出 stub（traj_xy/plan/action_mu 为 buffer；前向与输入无关）。"""

    def __init__(self, batch: int, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer(
            "traj_xy", torch.randn(batch, 6, 2, generator=generator, dtype=torch.float32)
        )
        self.register_buffer(
            "plan", torch.randn(batch, 6, 2, generator=generator, dtype=torch.float32)
        )
        self.register_buffer("action_mu", torch.randn(batch, 2, generator=generator, dtype=torch.float32))

    def forward(self, obs, *, rollout=True, world_model=False, **kwargs):  # noqa: D102 - stub
        return {"traj_xy": self.traj_xy, "plan": self.plan, "action_mu": self.action_mu}


def _stub_terms(traj_aux_weight: float, traj_row_mask=None):
    batch = 4
    torch.manual_seed(0)
    model = _StubPhase3Model(batch=batch, seed=1)
    generator = torch.Generator().manual_seed(2)
    targets = {
        "action": torch.randn(batch, 6, 2, generator=generator, dtype=torch.float32),
        "traj6": torch.randn(batch, 6, 2, generator=generator, dtype=torch.float32),
        "train_weight": torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.float32),
    }
    future_t = {
        "action_chain": targets["action"],
        "action_chain_valid": torch.ones(batch, 6, dtype=torch.float32),
    }
    config = Phase3Config(
        traj_aux_weight=traj_aux_weight,
        bias_calib_weight=0.0,
        ego_next_weight=0.0,
        od_weight=0.0,
        ld_weight=0.0,
        latent_weight=0.0,
        presence_weight=0.0,
        entry_weight=0.0,
        anchor_ce_weight=0.0,
        anchor_wta_weight=0.0,
    )
    terms = _phase3_loss_terms(
        model,
        {},
        future_t,
        targets,
        targets["train_weight"],
        config,
        traj_row_mask=traj_row_mask,
    )
    return terms, model, targets


def _manual_traj(weight: float, model, targets, mask=None) -> torch.Tensor:
    diff = model.traj_xy - targets["traj6"]
    per_sample = (diff ** 2).mean(dim=(-1, -2))
    weights = targets["train_weight"]
    if mask is not None:
        weights = weights * mask
    return weight * (per_sample * weights).sum() / weights.sum().clamp(min=1e-8)


def test_traj_term_zero_when_weight_zero_and_mask_irrelevant() -> None:
    for mask in (None, torch.zeros(4), torch.tensor([1.0, 0.0, 1.0, 0.0])):
        terms, _, _ = _stub_terms(0.0, mask)
        assert torch.equal(terms["traj"], torch.zeros((), dtype=terms["traj"].dtype))


def test_traj_term_matches_manual_with_mask() -> None:
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    terms, model, targets = _stub_terms(0.1, mask)
    expected = _manual_traj(0.1, model, targets, mask)
    assert torch.allclose(terms["traj"], expected, rtol=1e-6, atol=1e-8)
    # 掩码行只算被放行行：与"只保留掩码行再全量"等价
    terms_all, _, _ = _stub_terms(0.1, None)
    assert not torch.allclose(terms_all["traj"], terms["traj"])
    assert terms["traj"] > 0.0


def test_traj_term_all_masked_is_zero_and_none_mask_uses_all_rows() -> None:
    terms_zero, _, _ = _stub_terms(0.5, torch.zeros(4))
    assert float(terms_zero["traj"]) == 0.0
    terms_none, model, targets = _stub_terms(0.5, None)
    assert torch.allclose(terms_none["traj"], _manual_traj(0.5, model, targets, None), rtol=1e-6)


# --------------------------------------------------------------------------- #
# 端到端 tiny phase3
# --------------------------------------------------------------------------- #

def _tiny_phase3_dataset() -> BCDataset:
    arrays, _ = make_v2_arrays(episodes=4, steps_per_episode=6)
    return BCDataset(arrays, {"schema_version": 2, "label_names": None})


def _run_tiny_phase3(*, traj_aux_weight: float, traj_aux_valid=None, max_batches: int = 1) -> dict:
    dataset = _tiny_phase3_dataset()
    model = _tiny_model()
    config = Phase3Config(
        epochs=1, batch_size=24, micro_batch_size=None, device="cpu", shuffle=False, seed=0,
        val_indices=None, max_batches=max_batches,
        freeze_mode="all", freeze_prefixes=(),
        traj_aux_weight=traj_aux_weight, traj_aux_valid=traj_aux_valid,
    )
    return pretrain_bc_phase3(
        model, dataset, config, logger=lambda _: None, future_fn=_future_fn_for(dataset)
    )


def test_tiny_phase3_traj_weight_positive_monitor_equal_and_loss_delta() -> None:
    base = _run_tiny_phase3(traj_aux_weight=0.0)
    with_traj = _run_tiny_phase3(traj_aux_weight=0.5)
    # 默认（weight=0）不落 traj 损失键
    assert "bc_traj_aux_loss" not in base
    # weight>0：traj 项落键且 >0
    assert "bc_traj_aux_loss" in with_traj
    assert float(with_traj["bc_traj_aux_loss"]) > 0.0
    # 总损失差值 = traj 项贡献（单 batch、同前向 → 其余项逐位相同）
    for key in ("bc_action_loss", "bc_action_chain_loss", "bc_od_loss", "bc_ld_loss"):
        assert float(base[key]) == pytest.approx(float(with_traj[key]), rel=0, abs=1e-12)
    delta = float(with_traj["bc_loss"]) - float(base["bc_loss"])
    assert delta == pytest.approx(float(with_traj["bc_traj_aux_loss"]), rel=1e-5, abs=1e-9)
    # traj 监控口径不变：单 batch 同前向 → 完全一致
    for key in ("bc_traj_mse", "bc_traj_mae_m", "bc_traj_fde_m", "bc_traj_loss"):
        assert float(base[key]) == pytest.approx(float(with_traj[key]), rel=0, abs=1e-12)


def test_tiny_phase3_traj_all_masked_equals_weight_zero() -> None:
    base = _run_tiny_phase3(traj_aux_weight=0.0)
    masked = _run_tiny_phase3(
        traj_aux_weight=0.5, traj_aux_valid=np.zeros(24, dtype=np.float32)
    )
    assert float(masked["bc_traj_aux_loss"]) == 0.0
    assert float(masked["bc_loss"]) == pytest.approx(float(base["bc_loss"]), rel=0, abs=1e-12)


# --------------------------------------------------------------------------- #
# stages 接线 + fail-closed + 默认关闭
# --------------------------------------------------------------------------- #

def test_stages_traj_aux_wiring(tmp_path) -> None:
    window_dir, window_arrays = _write_window_dir(tmp_path, episodes=4, steps_per_episode=6)
    recovery_dir, _ = _write_recovery_dir(tmp_path, window_arrays, n_train=4, n_val=2, unusable_train=0)
    metrics = _run_stage3(
        tmp_path, window_dir,
        extra=[
            "--recovery-dir", str(recovery_dir),
            "--recovery-mix-ratio", "0.5",
            "--phase3-traj-aux-weight", "0.1",
        ],
        out_name="out_traj",
    )
    assert float(metrics["losses"]["traj_aux"]) == pytest.approx(0.1)
    assert metrics["traj_monitor_only"] is False
    assert "bc_traj_aux_loss" in metrics and float(metrics["bc_traj_aux_loss"]) > 0.0
    assert float(metrics["bc_traj_mse"]) >= 0.0


def test_stages_traj_aux_fail_closed_without_real_rows(tmp_path) -> None:
    window_dir, _ = _write_window_dir(tmp_path, episodes=4, steps_per_episode=6)
    with pytest.raises(SystemExit, match="无真实 traj6 行"):
        _run_stage3(
            tmp_path, window_dir,
            extra=["--phase3-traj-aux-weight", "0.1"],  # 无 --recovery-dir/锚
            out_name="out_fail",
        )


def test_stages_traj_aux_default_off_metrics_unchanged(tmp_path) -> None:
    window_dir, _ = _write_window_dir(tmp_path, episodes=4, steps_per_episode=6)
    metrics = _run_stage3(tmp_path, window_dir, extra=[], out_name="out_off")
    assert metrics["traj_monitor_only"] is True
    assert "bc_traj_aux_loss" not in metrics
    assert float(metrics["losses"]["traj_aux"]) == 0.0
