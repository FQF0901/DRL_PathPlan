"""Round-17 A3 配方开关单测：尾链正则（默认关、逐位不变）+ 六步衰减 CLI。

覆盖：
- ``Phase3Config``/``BCConfig`` 默认 ``tail_smooth_weight=tail_curv_weight=0.0``（默认关）；
- ``a3_tail_regularizers`` 数学：手算 l2/l1 平滑 + |Δκ|（含 ds floor）；
- ``_phase3_loss_terms``：权重 0 → ``terms["tail_smooth"]/["tail_curv"]`` 恒 0；
  权重 >0 → 与手算一致（行加权均值口径）；
- stages 接线：``--phase3-tail-smooth-weight/--phase3-tail-curv-weight`` 默认 None；
  ``--phase3-chain-decay`` 解析（``_parse_chain_decay_cli``）与 fail-closed。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.stages import _parse_args, _parse_chain_decay_cli
from pipeline.trainer import Phase3Config, _phase3_loss_terms, a3_tail_regularizers
from tests.test_phase3_traj_aux import _StubPhase3Model


def test_defaults_off() -> None:
    config = Phase3Config()
    assert float(config.tail_smooth_weight) == 0.0
    assert float(config.tail_curv_weight) == 0.0


def _manual(plan: torch.Tensor, weight: torch.Tensor, *, loss_type: str = "l2", floor: float = 0.5):
    diff = plan[:, 1:, :] - plan[:, :-1, :]
    per_sample = (diff ** 2).sum(dim=-1) if loss_type == "l2" else diff.abs().sum(dim=-1)
    smooth = (per_sample.mean(dim=-1) * weight).sum() / weight.sum()
    kappa = plan[:, :, 1] / plan[:, :, 0].abs().clamp(min=floor)
    curv = (kappa[:, 1:] - kappa[:, :-1]).abs().mean(dim=-1)
    return smooth, (curv * weight).sum() / weight.sum()


def test_regularizer_math() -> None:
    plan = torch.zeros(2, 6, 2, dtype=torch.float64)
    plan[0, :, 0] = torch.arange(6, dtype=torch.float64)  # ds 0..5，dθ=0
    plan[1, :, 0] = 1.0
    plan[1, :, 1] = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
    weight = torch.tensor([1.0, 3.0], dtype=torch.float64)
    smooth, curv = a3_tail_regularizers(plan, frame_weight=weight)
    manual_smooth, manual_curv = _manual(plan, weight)
    assert torch.allclose(smooth, manual_smooth, rtol=1e-9)
    assert torch.allclose(curv, manual_curv, rtol=1e-9)
    # 行 0：ds 每步 +1、dθ=0 → 逐差分平方=1（5 个差分的均值=1）；行 1：差分 (0,0.1) → 0.01
    # 行加权均值 = (1.0×1.0 + 3.0×0.01)/4 = 0.2575
    assert abs(float(manual_smooth) - (1.0 * 1.0 + 3.0 * 0.01) / 4.0) < 1e-9
    assert float(curv) > 0.0  # 行 1 的 dθ 递增 → |Δκ| > 0


def test_regularizer_l1_and_floor() -> None:
    plan = torch.zeros(1, 6, 2, dtype=torch.float64)
    plan[0, :, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)  # |ds|<floor
    plan[0, :, 1] = torch.tensor([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=torch.float64)
    weight = torch.ones(1, dtype=torch.float64)
    smooth_l2, curv = a3_tail_regularizers(plan, frame_weight=weight)
    smooth_l1, _ = a3_tail_regularizers(plan, frame_weight=weight, loss_type="l1")
    # 每差分 (Δds,Δdθ)=(0.1,0.2) → l2 逐差分 0.01+0.04=0.05；l1 = 0.1+0.2=0.3
    assert abs(float(smooth_l2) - 0.05) < 1e-12
    assert abs(float(smooth_l1) - 0.3) < 1e-12
    # κ = dθ / max(|ds|, 0.5) = 2·dθ（ds<floor）→ Δκ = 0.4
    assert abs(float(curv) - 0.4) < 1e-12


def _stub_terms(smooth_weight: float, curv_weight: float):
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
        tail_smooth_weight=smooth_weight,
        tail_curv_weight=curv_weight,
        bias_calib_weight=0.0, ego_next_weight=0.0, od_weight=0.0, ld_weight=0.0,
        latent_weight=0.0, presence_weight=0.0, entry_weight=0.0,
    )
    terms = _phase3_loss_terms(
        model, {}, future_t, targets, targets["train_weight"], config, traj_row_mask=None
    )
    return terms, model, targets


def test_terms_zero_when_weight_zero() -> None:
    terms, _, _ = _stub_terms(0.0, 0.0)
    assert float(terms["tail_smooth"]) == 0.0
    assert float(terms["tail_curv"]) == 0.0


def test_terms_match_manual_when_enabled() -> None:
    terms, model, targets = _stub_terms(0.5, 0.25)
    smooth, curv = a3_tail_regularizers(
        model.plan, frame_weight=targets["train_weight"]
    )
    assert torch.allclose(terms["tail_smooth"], 0.5 * smooth, rtol=1e-6, atol=1e-8)
    assert torch.allclose(terms["tail_curv"], 0.25 * curv, rtol=1e-6, atol=1e-8)
    assert float(terms["tail_smooth"]) > 0.0


def test_cli_flags_default_none() -> None:
    args = _parse_args(["--phase3", "dummy"])
    assert getattr(args, "phase3_tail_smooth_weight") is None
    assert getattr(args, "phase3_tail_curv_weight") is None
    assert getattr(args, "phase3_chain_decay") is None
    args2 = _parse_args(
        ["--phase3", "dummy", "--phase3-tail-smooth-weight", "0.1",
         "--phase3-tail-curv-weight", "0.2", "--phase3-chain-decay", "1.0,0.8,0.6,0.4,0.25,0.15"]
    )
    assert float(args2.phase3_tail_smooth_weight) == 0.1
    assert float(args2.phase3_tail_curv_weight) == 0.2
    assert args2.phase3_chain_decay == "1.0,0.8,0.6,0.4,0.25,0.15"


def test_chain_decay_cli_parsing() -> None:
    assert _parse_chain_decay_cli("1.0,0.8,0.6,0.4,0.25,0.15") == pytest.approx(
        (1.0, 0.8, 0.6, 0.4, 0.25, 0.15)
    )
    assert _parse_chain_decay_cli("1;0.5") == pytest.approx((1.0, 0.5))
    with pytest.raises(SystemExit):
        _parse_chain_decay_cli("abc")
    with pytest.raises(SystemExit):
        _parse_chain_decay_cli(" , ")
