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
from env.action.config import load_action_contract
from net.build_net import build_net
from net.model import DriveModel
from pipeline.common.checkpointing import save_drive_model_checkpoint

_C = load_action_contract()
CLEAN_BOUNDS = _C.bounds
from pipeline.eval.closed_loop import (
    HARD_LIMITS,
    _batch_from_history,
    _resolve_exact_execution,
    _trajectory,
    load_model,
    run_episode,
)
from pipeline.eval.metrics import summarize_rows

SPEC_PATH = "data/specs/mini_train_100.json"


def test_load_model_random_warns(capsys):
    model = load_model(None, "cpu")
    captured = capsys.readouterr().out.lower()

    assert isinstance(model, DriveModel)
    assert not model.training
    assert "warning" in captured and "smoke" in captured
    assert model.eval_config["hidden"] == model.hidden


def _synthetic_history():
    hist = _empty_history()
    hist["hist_valid"][:] = 1.0
    return hist


def test_load_model_from_checkpoint(tmp_path):
    state = build_net()
    ckpt = tmp_path / "clean.pt"
    save_drive_model_checkpoint(state, str(ckpt))

    model = load_model(str(ckpt), "cpu")
    assert isinstance(model, DriveModel)
    assert not model.training
    assert model.hidden == 128
    assert model.heads == 4
    assert model.eval_config == {"hidden": 128, "heads": 4, "spatial_layers": 2}

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
    for key, value in batch.items():
        if key in ("od_ids", "ld_ids"):
            assert value.dtype == torch.int64
        else:
            assert value.dtype == torch.float32

    out = build_net()(batch)
    assert out["action_mu"].shape == (1, 4)
    assert torch.isfinite(out["action_mu"]).all()
    assert torch.all(out["action_mu"] >= 0.0)
    assert torch.all(out["action_mu"] <= 1.0)


def test_infeasible_requested_action_is_executed_faithfully():
    requested = np.array([0.0, 0.0, 0.0, 10.0])
    executed, trajectory, feasible, codes = _resolve_exact_execution(
        requested, v0=10.0
    )
    np.testing.assert_allclose(executed, requested)
    assert trajectory is not None
    assert feasible is False
    assert "ZERO_LENGTH_WITH_SPEED" in codes


def test_invalid_requested_action_has_no_execution():
    requested = np.array([np.nan, 0.0, 0.0, 1.0])
    executed, trajectory, feasible, codes = _resolve_exact_execution(
        requested, v0=1.0
    )
    assert trajectory is None
    assert feasible is False
    assert codes == ("INVALID_ACTION",)
    np.testing.assert_allclose(executed, requested, equal_nan=True)


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
    assert "fallback" not in row
    assert row["preview"].startswith("NOT_IMPLEMENTED")
    assert "requested_action_codes" in row
    assert "executed_action_codes" in row
    assert len(row["requested_action"]) == 4
    assert len(row["executed_action"]) == 4
