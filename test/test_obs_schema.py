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
    stack_history,
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


def test_history_shapes_and_metadata(env_and_obs):
    env, obs = env_and_obs
    hist = stack_history([obs] * HISTORY_FRAMES, [obs["pose"]] * HISTORY_FRAMES)

    shapes = {
        "od": (HISTORY_FRAMES, OD_SLOTS, OD_DIM),
        "od_mask": (HISTORY_FRAMES, OD_SLOTS),
        "od_ids": (HISTORY_FRAMES, OD_SLOTS),
        "od_anchor": (HISTORY_FRAMES, OD_SLOTS, 3),
        "od_taxonomy": (HISTORY_FRAMES, OD_SLOTS, 4),
        "od_age": (HISTORY_FRAMES, OD_SLOTS),
        "ld": (HISTORY_FRAMES, LD_SLOTS, LD_POINTS, LD_DIM),
        "ld_mask": (HISTORY_FRAMES, LD_SLOTS, LD_POINTS),
        "ld_ids": (HISTORY_FRAMES, LD_SLOTS),
        "ld_road_ids": (HISTORY_FRAMES, LD_SLOTS, 2),
        "ld_anchor": (HISTORY_FRAMES, LD_SLOTS, 3),
        "ld_chunk_s": (HISTORY_FRAMES, LD_SLOTS),
        "nav": (HISTORY_FRAMES, NAV_DIM),
        "nav_mask": (HISTORY_FRAMES, 1),
        "nav_road_mask": (HISTORY_FRAMES, NAV_ROADS),
        "nav_road_ids": (HISTORY_FRAMES, NAV_ROADS, 2),
        "nav_route_id": (HISTORY_FRAMES,),
        "ego": (HISTORY_FRAMES, EGO_DIM),
        "ego_mask": (HISTORY_FRAMES, 1),
        "pose": (HISTORY_FRAMES, 3),
        "hist_valid": (HISTORY_FRAMES,),
    }
    for key, shape in shapes.items():
        assert hist[key].shape == shape, key
    for key in ("od_ids", "od_taxonomy", "ld_ids", "ld_road_ids", "nav_road_ids", "nav_route_id"):
        assert hist[key].dtype == np.int64, key
    for key in shapes:
        if hist[key].dtype.kind == "f":
            assert np.isfinite(hist[key]).all(), key

    assert hist["hist_valid"].sum() == HISTORY_FRAMES
    last = HISTORY_FRAMES - 1
    assert np.array_equal(hist["od_ids"][last], obs["od_ids"])
    assert np.array_equal(hist["ld_ids"][last], obs["ld_ids"])
    assert np.array_equal(hist["nav_road_ids"][last], obs["nav_road_ids"])
    assert int(hist["nav_route_id"][last]) == int(obs["nav_route_id"])

    # Warm-up frames are zero/masked, not fake repeated history.
    short = stack_history([obs, obs], [obs["pose"], obs["pose"]])
    assert short["hist_valid"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert short["od_mask"][:4].sum() == 0.0
    assert short["ld_mask"][:4].sum() == 0.0
    assert short["nav_mask"][:4].sum() == 0.0
    assert short["nav_road_mask"][:4].sum() == 0.0


def test_history_aligns_by_stable_id_not_slot(env_and_obs):
    env, obs = env_and_obs
    order_od = list(reversed(range(OD_SLOTS)))
    order_ld = list(reversed(range(LD_SLOTS)))
    order_nav = list(reversed(range(NAV_ROADS)))
    shuffled = _permute_frame(obs, order_od, order_ld, order_nav)

    hist = stack_history([shuffled, obs], [obs["pose"], obs["pose"]])
    t = HISTORY_FRAMES - 2

    for j in range(OD_SLOTS):
        if obs["od_mask"][j] <= 0.5:
            continue
        assert hist["od_mask"][t, j] == 1.0
        assert hist["od_ids"][t, j] == obs["od_ids"][j]
        assert np.allclose(hist["od"][t, j], obs["od"][j])
        assert np.allclose(hist["od_anchor"][t, j], obs["od_anchor"][j])

    for j in range(LD_SLOTS):
        if not obs["ld_mask"][j].any():
            continue
        assert np.array_equal(hist["ld_mask"][t, j], obs["ld_mask"][j])
        assert hist["ld_ids"][t, j] == obs["ld_ids"][j]
        assert np.allclose(hist["ld"][t, j], obs["ld"][j])
        assert np.allclose(hist["ld_anchor"][t, j], obs["ld_anchor"][j])
        assert np.allclose(hist["ld_chunk_s"][t, j], obs["ld_chunk_s"][j])

    for j in range(NAV_ROADS):
        if obs["nav_road_mask"][j] <= 0.5:
            continue
        assert hist["nav_road_mask"][t, j] == 1.0
        assert np.array_equal(hist["nav_road_ids"][t, j], obs["nav_road_ids"][j])
        assert np.allclose(
            hist["nav"][t, j * NAV_ROAD_DIM:(j + 1) * NAV_ROAD_DIM],
            obs["nav"][j * NAV_ROAD_DIM:(j + 1) * NAV_ROAD_DIM],
        )
    assert hist["nav_mask"][t, 0] == 1.0


def test_history_missing_ids_are_masked(env_and_obs):
    env, obs = env_and_obs
    order_od = list(reversed(range(OD_SLOTS)))
    order_ld = list(reversed(range(LD_SLOTS)))
    order_nav = list(reversed(range(NAV_ROADS)))
    shuffled = _permute_frame(obs, order_od, order_ld, order_nav)
    t = HISTORY_FRAMES - 2

    od_valid = np.where(obs["od_mask"] > 0.5)[0]
    assert len(od_valid) > 0
    od_slot = int(od_valid[0])
    shuffled["od_mask"][order_od.index(od_slot)] = 0.0
    shuffled["od_ids"][order_od.index(od_slot)] = 0
    shuffled["od"][order_od.index(od_slot)] = 0.0

    ld_valid = np.where(obs["ld_mask"].any(axis=1))[0]
    assert len(ld_valid) > 0
    ld_slot = int(ld_valid[0])
    shuffled["ld_mask"][order_ld.index(ld_slot)] = 0.0
    shuffled["ld"][order_ld.index(ld_slot)] = 0.0

    nav_slot = 1
    assert obs["nav_road_mask"][nav_slot] > 0.5
    shuffled["nav_road_mask"][order_nav.index(nav_slot)] = 0.0
    shuffled["nav_road_ids"][order_nav.index(nav_slot)] = 0

    hist = stack_history([shuffled, obs], [obs["pose"], obs["pose"]])
    assert hist["od_mask"][t, od_slot] == 0.0
    assert np.allclose(hist["od"][t, od_slot], 0.0)
    assert hist["ld_mask"][t, ld_slot].sum() == 0.0
    assert np.allclose(hist["ld"][t, ld_slot], 0.0)
    assert hist["nav_road_mask"][t, nav_slot] == 0.0
    assert np.allclose(hist["nav"][t, nav_slot * NAV_ROAD_DIM:(nav_slot + 1) * NAV_ROAD_DIM], 0.0)
    # Route-level nav is still valid even when one historical slot is missing.
    assert hist["nav_mask"][t, 0] == 1.0


def test_history_compensates_geometry_into_current_ego_frame(env_and_obs):
    env, obs = env_and_obs
    t = HISTORY_FRAMES - 2

    # Pure translation: historical ego was 10 m ahead on +x.
    hist = stack_history([obs, obs], [np.array([10.0, 0.0, 0.0]), np.zeros(3)])
    for j in range(OD_SLOTS):
        if obs["od_mask"][j] <= 0.5:
            continue
        assert np.allclose(hist["od"][t, j, 0], obs["od"][j, 0] + 10.0)
        assert np.allclose(hist["od"][t, j, 1], obs["od"][j, 1])
        assert np.allclose(hist["od_anchor"][t, j, 0], obs["od_anchor"][j, 0] + 10.0)
        assert np.allclose(hist["od_anchor"][t, j, 1], obs["od_anchor"][j, 1])
    for j in range(LD_SLOTS):
        if not obs["ld_mask"][j].any():
            continue
        assert np.allclose(hist["ld"][t, j, :, 0], obs["ld"][j, :, 0] + 10.0)
        assert np.allclose(hist["ld"][t, j, :, 1], obs["ld"][j, :, 1])
        assert np.allclose(hist["ld_anchor"][t, j, 0], obs["ld_anchor"][j, 0] + 10.0)

    # Nav pure translation: 10 m ahead must add 10/100 in normalized units,
    # not 10 (the old 100x bug) or 1000.
    for j in range(NAV_ROADS):
        if obs["nav_road_mask"][j] <= 0.5:
            continue
        block = obs["nav"][j * NAV_ROAD_DIM:(j + 1) * NAV_ROAD_DIM]
        got = hist["nav"][t, j * NAV_ROAD_DIM:(j + 1) * NAV_ROAD_DIM]
        for dx_i, dy_i in ((NAV_ROAD_INDEX["entry_dx"], NAV_ROAD_INDEX["entry_dy"]),
                           (NAV_ROAD_INDEX["mid_dx"], NAV_ROAD_INDEX["mid_dy"]),
                           (NAV_ROAD_INDEX["exit_dx"], NAV_ROAD_INDEX["exit_dy"])):
            assert np.allclose(got[dx_i], float(block[dx_i]) + 0.10)
            assert np.allclose(got[dy_i], float(block[dy_i]))
    assert np.allclose(hist["nav"][t, NAV_GLOBAL_INDEX["dest_dx"]],
                       float(obs["nav"][NAV_GLOBAL_INDEX["dest_dx"]]) + 0.10)
    assert np.allclose(hist["nav"][t, NAV_GLOBAL_INDEX["dest_dy"]],
                       float(obs["nav"][NAV_GLOBAL_INDEX["dest_dy"]]))

    # Pure rotation by +90 degrees: (x, y) -> (-y, x).
    rot = stack_history([obs, obs], [np.array([0.0, 0.0, np.pi / 2.0]), np.zeros(3)])
    for j in range(OD_SLOTS):
        if obs["od_mask"][j] <= 0.5:
            continue
        x, y = float(obs["od"][j, 0]), float(obs["od"][j, 1])
        assert np.allclose(rot["od"][t, j, 0], -y)
        assert np.allclose(rot["od"][t, j, 1], x)
        c, s = float(obs["od"][j, 2]), float(obs["od"][j, 3])
        assert np.allclose(rot["od"][t, j, 2], -s)
        assert np.allclose(rot["od"][t, j, 3], c)
        assert np.allclose(rot["od_anchor"][t, j, 0], -float(obs["od_anchor"][j, 1]))
        assert np.allclose(rot["od_anchor"][t, j, 1], float(obs["od_anchor"][j, 0]))
        _angle_close(rot["od_anchor"][t, j, 2], float(obs["od_anchor"][j, 2]) + np.pi / 2.0)

    for j in range(LD_SLOTS):
        if not obs["ld_mask"][j].any():
            continue
        x = obs["ld"][j, :, 0]
        y = obs["ld"][j, :, 1]
        assert np.allclose(rot["ld"][t, j, :, 0], -y)
        assert np.allclose(rot["ld"][t, j, :, 1], x)
        for k in range(LD_POINTS):
            _angle_close(rot["ld"][t, j, k, 2], float(obs["ld"][j, k, 2]) + np.pi / 2.0)
        assert np.allclose(rot["ld_anchor"][t, j, 0], -float(obs["ld_anchor"][j, 1]))
        assert np.allclose(rot["ld_anchor"][t, j, 1], float(obs["ld_anchor"][j, 0]))
        _angle_close(rot["ld_anchor"][t, j, 2], float(obs["ld_anchor"][j, 2]) + np.pi / 2.0)

    # Nav per-road entry/mid/exit and the destination are also compensated.
    for j in range(NAV_ROADS):
        if obs["nav_road_mask"][j] <= 0.5:
            continue
        block = obs["nav"][j * NAV_ROAD_DIM:(j + 1) * NAV_ROAD_DIM]
        got = rot["nav"][t, j * NAV_ROAD_DIM:(j + 1) * NAV_ROAD_DIM]
        for dx_i, dy_i in ((NAV_ROAD_INDEX["entry_dx"], NAV_ROAD_INDEX["entry_dy"]),
                           (NAV_ROAD_INDEX["mid_dx"], NAV_ROAD_INDEX["mid_dy"]),
                           (NAV_ROAD_INDEX["exit_dx"], NAV_ROAD_INDEX["exit_dy"])):
            assert np.allclose(got[dx_i], -float(block[dy_i]))
            assert np.allclose(got[dy_i], float(block[dx_i]))
        _angle_close(
            got[NAV_ROAD_INDEX["entry_heading_rel"]],
            float(block[NAV_ROAD_INDEX["entry_heading_rel"]]) + np.pi / 2.0,
        )
        _angle_close(
            got[NAV_ROAD_INDEX["exit_heading_rel"]],
            float(block[NAV_ROAD_INDEX["exit_heading_rel"]]) + np.pi / 2.0,
        )

    dest_dx = obs["nav"][NAV_GLOBAL_INDEX["dest_dx"]]
    dest_dy = obs["nav"][NAV_GLOBAL_INDEX["dest_dy"]]
    assert np.allclose(rot["nav"][t, NAV_GLOBAL_INDEX["dest_dx"]], -float(dest_dy))
    assert np.allclose(rot["nav"][t, NAV_GLOBAL_INDEX["dest_dy"]], float(dest_dx))

    # Ego features are ego-relative and must not be rotated by stack_history.
    assert np.allclose(rot["ego"][t], obs["ego"])
    assert np.allclose(rot["ego_mask"][t], obs["ego_mask"])


def test_physics_dt_follows_metadrive_config():
    class Engine:
        global_config = {"physics_world_step_size": 0.02, "decision_repeat": 5}

    class Env:
        engine = Engine()

    assert abs(_physics_dt(Env()) - 0.1) < 1e-12

    class Broken:
        engine = None

    assert abs(_physics_dt(Broken()) - 0.1) < 1e-12


def test_ld_stable_id_ignores_chunk_s(env_and_obs):
    _, obs = env_and_obs
    nxt = copy.deepcopy(obs)
    nxt["ld_chunk_s"] = np.asarray(nxt["ld_chunk_s"], dtype=np.float32) + 1.0
    hist = stack_history([nxt, obs], [nxt["pose"], obs["pose"]])
    last = HISTORY_FRAMES - 1
    # IDs identify the lane, not the local 20 m crop; chunk_s is a feature.
    assert np.array_equal(hist["ld_ids"][last], obs["ld_ids"])
    assert np.array_equal(hist["ld_ids"][last - 1], obs["ld_ids"])
    assert np.allclose(hist["ld_chunk_s"][last - 1], nxt["ld_chunk_s"])
    assert np.allclose(hist["ld_chunk_s"][last], obs["ld_chunk_s"])
