"""Contract tests for the canonical feature-governance planner."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from env.obs import (
    EGO_DIM,
    HISTORY_FRAMES,
    LD_DIM,
    LD_POINTS,
    LD_SLOTS,
    NAV_DIM,
    OD_DIM,
    OD_SLOTS,
)
from net.model import StructuredPlanner, count_parameters

B, T = 2, HISTORY_FRAMES
HIDDEN = 128


def _batch() -> dict:
    g = torch.Generator().manual_seed(7)
    od_mask = (torch.rand(B, T, OD_SLOTS, generator=g) > 0.35).float()
    ld_mask = (torch.rand(B, T, LD_SLOTS, LD_POINTS, generator=g) > 0.25).float()
    # guarantee at least one live entity per frame for the normal-path test
    od_mask[:, :, 0] = 1.0
    ld_mask[:, :, 0, 0] = 1.0
    return {
        "od": torch.randn(B, T, OD_SLOTS, OD_DIM, generator=g),
        "od_mask": od_mask,
        "ld": torch.randn(B, T, LD_SLOTS, LD_POINTS, LD_DIM, generator=g),
        "ld_mask": ld_mask,
        "nav": torch.randn(B, T, NAV_DIM, generator=g),
        "nav_mask": torch.ones(B, T, 1),
        "ego": torch.randn(B, T, EGO_DIM, generator=g),
        "ego_mask": torch.ones(B, T, 1),
        "hist_valid": torch.ones(B, T),
        "od_ids": torch.arange(B * T * OD_SLOTS).reshape(B, T, OD_SLOTS),
        "ld_ids": torch.arange(B * T * LD_SLOTS).reshape(B, T, LD_SLOTS),
        "od_anchor": torch.randn(B, T, OD_SLOTS, 3, generator=g),
        "ld_anchor": torch.randn(B, T, LD_SLOTS, 3, generator=g),
    }


def _assert_finite(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)
        return
    assert torch.isfinite(value).all(), value


def test_forward_shapes_and_finiteness():
    model = StructuredPlanner()
    out = model(_batch())

    assert out["action_mu"].shape == (B, 4)
    assert out["action_logstd"].shape == (B, 4)
    assert out["value"].shape == (B, 1)
    assert out["risk"].shape == (B, 5)
    assert out["cost_value"].shape == (B, 1)
    assert out["latent"].shape == (B, HIDDEN)
    assert torch.all(out["action_mu"] >= 0.0) and torch.all(out["action_mu"] <= 1.0)
    assert torch.all(out["action_logstd"] >= -5.0) and torch.all(out["action_logstd"] <= 0.0)
    _assert_finite(out)


def test_plan_input_is_exactly_six_latents():
    model = StructuredPlanner()
    assert model.plan[0].in_features == 6 * HIDDEN

    seen = []
    handle = model.plan[0].register_forward_pre_hook(lambda _m, inp: seen.append(tuple(inp[0].shape)))
    try:
        model(_batch())
    finally:
        handle.remove()
    assert seen == [(B, 6 * HIDDEN)]


def test_encode_entities_interface():
    model = StructuredPlanner()
    batch = _batch()
    ent = model.encode_entities(batch)

    expected = {
        "od": (B, OD_SLOTS, HIDDEN),
        "ld": (B, LD_SLOTS, HIDDEN),
        "nav": (B, HIDDEN),
        "ego": (B, HIDDEN),
        "od_mask": (B, OD_SLOTS),
        "ld_mask": (B, LD_SLOTS),
        "ld_point_mask": (B, LD_SLOTS, LD_POINTS),
        "nav_mask": (B, 1),
        "ego_mask": (B, 1),
        "hist_valid": (B, T),
        "od_ids": (B, OD_SLOTS),
        "ld_ids": (B, LD_SLOTS),
        "od_anchor": (B, OD_SLOTS, 3),
        "ld_anchor": (B, LD_SLOTS, 3),
    }
    for key, shape in expected.items():
        assert ent[key].shape == shape, (key, ent[key].shape, shape)
        _assert_finite(ent[key])

    torch.testing.assert_close(ent["od_ids"], batch["od_ids"][:, -1])
    torch.testing.assert_close(ent["ld_ids"], batch["ld_ids"][:, -1])
    torch.testing.assert_close(ent["od_anchor"], batch["od_anchor"][:, -1])
    torch.testing.assert_close(ent["ld_anchor"], batch["ld_anchor"][:, -1])
    torch.testing.assert_close(ent["ld_mask"], batch["ld_mask"][:, -1].any(dim=-1).float())


def test_aux_head_shapes():
    model = StructuredPlanner()
    aux = model(_batch())["aux"]
    assert aux["lane_role_logits"].shape == (B, LD_SLOTS, 4)
    assert aux["od_gap"].shape == (B, OD_SLOTS)
    assert aux["od_ttc"].shape == (B, OD_SLOTS)
    assert aux["od_urgent"].shape == (B, OD_SLOTS)
    _assert_finite(aux)


def test_all_invalid_entity_rows_are_safe():
    model = StructuredPlanner()
    batch = _batch()
    # row 0 has no entities at all; row 1 keeps a valid scene.
    batch["od_mask"][0] = 0.0
    batch["ld_mask"][0] = 0.0
    batch["nav_mask"][0] = 0.0

    ent = model.encode_entities(batch)
    out = model(batch)

    _assert_finite(ent)
    _assert_finite(out)
    assert torch.count_nonzero(ent["od"][0]) == 0
    assert torch.count_nonzero(ent["ld"][0]) == 0
    assert torch.count_nonzero(ent["od"][1]) > 0


def test_parameter_budget():
    total = count_parameters(StructuredPlanner())
    assert 800_000 <= total <= 1_200_000, total


def test_no_oracle_features():
    source = Path(inspect.getsourcefile(StructuredPlanner)).read_text()
    for name in ("current_ref_lanes", "get_checkpoints", "navi_arrow_dir"):
        assert name not in source, name
