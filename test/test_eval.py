"""Tests for the clean-schema closed-loop evaluator.

The real-env smoke is opt-in (``RUN_EVAL_CLEAN_SMOKE=1``) so the default test
run stays fast and never requires a checkpoint.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from env.metadrive_env import build_env
from env.obs import (
    EGO_DIM,
    HISTORY_FRAMES,
    LD_DIM,
    LD_POINTS,
    LD_SLOTS,
    NAV_DIM,
    OD_DIM,
    OD_SLOTS,
    _empty_history,
)
from env.scenario.spec import load_specs
from env.action.action_space import CLEAN_BOUNDS
from net.model import StructuredPlanner
from pipeline.eval import (
    HARD_LIMITS,
    _batch_from_history,
    _trajectory,
    load_model,
    run_episode,
    safe_brake_action,
)
from pipeline.eval_metrics import summarize_rows

SPEC_PATH = "env/specs/scenarios_train_slice200.json"


def test_load_model_random_warns(capsys):
    model = load_model(None, "cpu")
    captured = capsys.readouterr().out.lower()

    assert isinstance(model, StructuredPlanner)
    assert not model.training
    assert "warning" in captured and "smoke" in captured
    assert model.eval_config["hidden"] == model.hidden


def _synthetic_history():
    hist = _empty_history()
    hist["hist_valid"][:] = 1.0
    return hist


def test_load_model_from_checkpoint(tmp_path):
    state = StructuredPlanner(hidden=64, heads=4, spatial_layers=1)
    ckpt = tmp_path / "clean.pt"
    torch.save(
        {
            "model": state.state_dict(),
            "config": {"hidden": 64, "heads": 4, "spatial_layers": 1},
        },
        ckpt,
    )

    model = load_model(str(ckpt), "cpu")
    assert isinstance(model, StructuredPlanner)
    assert not model.training
    assert model.hidden == 64
    assert model.heads == 4
    assert model.eval_config == {"hidden": 64, "heads": 4, "spatial_layers": 1}


def test_batch_from_history_shapes_and_forward():
    batch = _batch_from_history(_synthetic_history(), device="cpu")

    assert batch["od"].shape == (1, HISTORY_FRAMES, OD_SLOTS, OD_DIM)
    assert batch["od_mask"].shape == (1, HISTORY_FRAMES, OD_SLOTS)
    assert batch["ld"].shape == (1, HISTORY_FRAMES, LD_SLOTS, LD_POINTS, LD_DIM)
    assert batch["ld_mask"].shape == (1, HISTORY_FRAMES, LD_SLOTS, LD_POINTS)
    assert batch["nav"].shape == (1, HISTORY_FRAMES, NAV_DIM)
    assert batch["nav_mask"].shape == (1, HISTORY_FRAMES, 1)
    assert batch["ego"].shape == (1, HISTORY_FRAMES, EGO_DIM)
    assert batch["ego_mask"].shape == (1, HISTORY_FRAMES, 1)
    assert batch["hist_valid"].shape == (1, HISTORY_FRAMES)
    for value in batch.values():
        assert value.dtype == torch.float32

    out = StructuredPlanner()(batch)
    assert out["action_mu"].shape == (1, 4)
    assert torch.isfinite(out["action_mu"]).all()
    assert torch.all(out["action_mu"] >= 0.0)
    assert torch.all(out["action_mu"] <= 1.0)


def test_safe_brake_action_is_feasible():
    for v0 in (0.0, 2.0, 10.0, 14.0):
        action = safe_brake_action(v0)
        expected_v1 = max(0.0, v0 - 3.0)
        assert action[1] == 0.0 and action[2] == 0.0
        assert action[3] == pytest.approx(expected_v1)
        assert action[0] == pytest.approx((v0 + expected_v1) * 0.5 / 2.0)
        assert np.all(action >= CLEAN_BOUNDS.low)
        assert np.all(action <= CLEAN_BOUNDS.high)

        tr = _trajectory(action, v0)
        assert tr.feasible, tr.codes
        assert tr.speeds[-1] == pytest.approx(expected_v1)
        assert not tr.codes


def test_clean_limits_are_provisional_values():
    assert HARD_LIMITS.a_long_max == 3.5
    assert HARD_LIMITS.kappa_max == 0.345
    assert HARD_LIMITS.jerk_long_max == 20.0


def test_summarize_rows_accepts_clean_schema():
    rows = [
        {
            "success": True,
            "collision": False,
            "off_road": False,
            "wrong_route": False,
            "stuck": False,
            "timeout": False,
            "comfort_violation": False,
            "feasible": True,
            "fallback": False,
            "route_completion": 1.0,
            "tracking_error_xy": 0.10,
            "tracking_error_heading": 0.02,
            "tracking_error_speed": 0.05,
            "steer_saturation": 0.0,
            "throttle_saturation": 0.2,
            "max_curvature": 0.10,
            "max_lateral_accel": 1.0,
            "max_jerk": 2.0,
            "geometry": "straight",
        },
        {
            "success": False,
            "collision": True,
            "off_road": False,
            "wrong_route": False,
            "stuck": False,
            "timeout": False,
            "comfort_violation": True,
            "feasible": False,
            "fallback": True,
            "route_completion": 0.4,
            "tracking_error_xy": 0.30,
            "tracking_error_heading": 0.06,
            "tracking_error_speed": 0.15,
            "steer_saturation": 0.5,
            "throttle_saturation": 0.5,
            "max_curvature": 0.20,
            "max_lateral_accel": 2.0,
            "max_jerk": 4.0,
            "geometry": "curve",
        },
    ]
    summary = summarize_rows(rows)

    assert summary["n"] == 2
    assert summary["success_rate"] == pytest.approx(0.5)
    assert summary["collision_rate"] == pytest.approx(0.5)
    assert summary["fallback_rate"] == pytest.approx(0.5)
    assert summary["route_completion_mean"] == pytest.approx(0.7)
    assert summary["tracking_error_xy_mean"] == pytest.approx(0.2)
    assert summary["max_jerk_mean"] == pytest.approx(3.0)
    assert summary["by_group"]["straight"]["success_rate"] == pytest.approx(1.0)
    assert summary["by_group"]["curve"]["success_rate"] == pytest.approx(0.0)


@pytest.mark.skipif(
    os.environ.get("RUN_EVAL_CLEAN_SMOKE") != "1",
    reason="set RUN_EVAL_CLEAN_SMOKE=1 to run the 1-scene MetaDrive smoke",
)
def test_run_episode_one_scene_smoke():
    spec = load_specs(SPEC_PATH)[0]
    model = load_model(None, "cpu")
    env = build_env(spec, use_render=False, lru_size=0)
    try:
        row = run_episode(
            env, model, spec, controller="exact", max_steps=10, device="cpu"
        )
    finally:
        env.close()

    assert row["controller"] == "exact"
    assert 0 < row["steps"] <= 10
    assert row["policy_steps"] >= 1
    assert isinstance(row["success"], bool)
    assert isinstance(row["collision"], bool)
    assert isinstance(row["feasible"], bool)
    assert isinstance(row["fallback"], bool)
    assert row["preview"].startswith("NOT_IMPLEMENTED")
    assert "requested_action_codes" in row
    assert "executed_action_codes" in row
    assert len(row["requested_action"]) == 4
    assert len(row["executed_action"]) == 4
