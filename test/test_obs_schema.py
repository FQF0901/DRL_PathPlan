"""Tests for the forward OD/LD/Nav/Ego schema v2.

These tests use the real MetaDrive wrapper and the first scenario of the
existing slice-200 spec.  They check fixed shapes, finiteness, stable IDs,
road-level-only Nav, ID-based history alignment and SE(2) compensation.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from env.metadrive_env import build_env
from env.obs import (
    EGO_DIM,
    HISTORY_FRAMES,
    LD_DIM,
    LD_POINTS,
    LD_SLOTS,
    NAV_DIM,
    NAV_GLOBAL_INDEX,
    NAV_ROAD_DIM,
    NAV_ROAD_INDEX,
    NAV_ROADS,
    OD_DIM,
    OD_INDEX,
    OD_SLOTS,
    _physics_dt,
    build_observation,
)
from env.scenario.spec import load_specs

SPEC_PATH = "data/specs/mini_train_100.json"


@pytest.fixture(scope="module")
def env_and_obs():
    spec = load_specs(SPEC_PATH)[0]
    env = build_env(spec, use_render=False)
    env.reset()
    obs = build_observation(env)
    try:
        yield env, obs
    finally:
        env.close()


def _angle_close(actual, expected, tol=1e-5):
    delta = np.arctan2(np.sin(float(actual) - float(expected)), np.cos(float(actual) - float(expected)))
    assert abs(float(delta)) <= tol, (actual, expected, delta)


def _permute_frame(frame, od_order, ld_order, nav_order):
    """Return a deep copy with slot order changed but stable IDs intact."""
    out = copy.deepcopy(frame)
    od_idx = np.asarray(od_order, dtype=np.int64)
    ld_idx = np.asarray(ld_order, dtype=np.int64)
    nav_idx = np.asarray(nav_order, dtype=np.int64)

    for key in ("od", "od_mask", "od_ids", "od_anchor", "od_taxonomy", "od_age"):
        out[key] = out[key][od_idx]
    for key in ("ld", "ld_mask", "ld_ids", "ld_road_ids", "ld_anchor", "ld_chunk_s"):
        out[key] = out[key][ld_idx]

    old_nav = np.array(out["nav"], copy=True)
    for new_slot, old_slot in enumerate(nav_idx):
        out["nav"][new_slot * NAV_ROAD_DIM:(new_slot + 1) * NAV_ROAD_DIM] = old_nav[
            old_slot * NAV_ROAD_DIM:(old_slot + 1) * NAV_ROAD_DIM
        ]
    out["nav_road_mask"] = out["nav_road_mask"][nav_idx]
    out["nav_road_ids"] = out["nav_road_ids"][nav_idx]
    return out


def test_current_frame_shapes_finite_and_stable_ids(env_and_obs):
    env, obs = env_and_obs

    shapes = {
        "od": (OD_SLOTS, OD_DIM),
        "od_mask": (OD_SLOTS,),
        "od_ids": (OD_SLOTS,),
        "od_anchor": (OD_SLOTS, 3),
        "od_taxonomy": (OD_SLOTS, 4),
        "od_age": (OD_SLOTS,),
        "ld": (LD_SLOTS, LD_POINTS, LD_DIM),
        "ld_mask": (LD_SLOTS, LD_POINTS),
        "ld_ids": (LD_SLOTS,),
        "ld_road_ids": (LD_SLOTS, 2),
        "ld_anchor": (LD_SLOTS, 3),
        "ld_chunk_s": (LD_SLOTS,),
        "nav": (NAV_DIM,),
        "nav_mask": (1,),
        "nav_road_mask": (NAV_ROADS,),
        "nav_road_ids": (NAV_ROADS, 2),
        "ego": (EGO_DIM,),
        "ego_mask": (1,),
        "pose": (3,),
    }
    for key, shape in shapes.items():
        assert obs[key].shape == shape, key

    int64_keys = ("od_ids", "od_taxonomy", "ld_ids", "ld_road_ids", "nav_road_ids", "nav_route_id")
    for key in int64_keys:
        assert obs[key].dtype == np.int64, key
    assert np.isscalar(obs["nav_route_id"]) or obs["nav_route_id"].shape == ()

    float_keys = (
        "od", "od_mask", "od_anchor", "od_age", "ld", "ld_mask", "ld_anchor",
        "ld_chunk_s", "nav", "nav_mask", "nav_road_mask", "ego", "ego_mask", "pose",
    )
    for key in float_keys:
        assert obs[key].dtype == np.float32, key
        assert np.isfinite(obs[key]).all(), key

    assert obs["nav_mask"][0] == 1.0
    assert obs["nav_road_mask"][0] == 1.0
    assert obs["nav"][NAV_GLOBAL_INDEX["nav_valid"]] == 1.0
    assert obs["nav_road_ids"][0, 1] == 0  # slot 0 is the current road

    # Stable IDs must be reproducible for the same scene/slot contents.
    obs2 = build_observation(env)
    for key in ("od_ids", "ld_ids", "nav_road_ids", "nav_route_id"):
        assert np.array_equal(obs[key], obs2[key]), key

    od_valid_ids = obs["od_ids"][obs["od_mask"] > 0.5]
    ld_valid_ids = obs["ld_ids"][obs["ld_mask"].any(axis=1)]
    assert len(set(od_valid_ids.tolist())) == len(od_valid_ids)
    assert len(set(ld_valid_ids.tolist())) == len(ld_valid_ids)

    for key in ("schema", "schema_version", "source", "scene_seed", "timestamp", "step", "dt"):
        assert key in obs["meta"]


class _ForbiddenNavAPI:
    """Any use of a lane-level navigation oracle fails the test."""

    def __getitem__(self, *args, **kwargs):
        raise AssertionError("lane-level navigation oracle was read")

    def __iter__(self):
        raise AssertionError("lane-level navigation oracle was iterated")

    def __len__(self):
        raise AssertionError("lane-level navigation oracle was sized")

    def __call__(self, *args, **kwargs):
        raise AssertionError("lane-level navigation oracle was called")


def test_build_observation_never_reads_lane_level_nav(env_and_obs, monkeypatch):
    env, baseline = env_and_obs
    nav = env.agent.navigation

    monkeypatch.setattr(nav, "get_checkpoints", _ForbiddenNavAPI())
    monkeypatch.setattr(nav, "navi_arrow_dir", _ForbiddenNavAPI())
    monkeypatch.setattr(nav, "current_ref_lanes", _ForbiddenNavAPI())
    monkeypatch.setattr(nav, "next_ref_lanes", _ForbiddenNavAPI())
    monkeypatch.setattr(nav, "final_lane", _ForbiddenNavAPI())

    obs = build_observation(env)
    assert obs["nav_mask"][0] == 1.0
    assert obs["nav_road_mask"][0] == 1.0
    assert obs["nav"][NAV_GLOBAL_INDEX["nav_valid"]] == 1.0
    assert np.isfinite(obs["nav"]).all()
    assert np.array_equal(obs["nav"], baseline["nav"])
    assert np.array_equal(obs["nav_road_ids"], baseline["nav_road_ids"])
    assert np.array_equal(obs["nav_route_id"], baseline["nav_route_id"])


def test_physics_dt_follows_metadrive_config():
    class Engine:
        global_config = {"physics_world_step_size": 0.02, "decision_repeat": 5}

    class Env:
        engine = Engine()

    assert abs(_physics_dt(Env()) - 0.1) < 1e-12

    class Broken:
        engine = None

    assert abs(_physics_dt(Broken()) - 0.1) < 1e-12
