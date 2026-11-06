"""Tests for the schema-v2 clean-planner BC trainer.

No MetaDrive environment or collection is built here.  The fixture is a tiny
synthetic ``expert_bc.npz`` with the exact schema-v2 history shapes, two
scenes observed by four episodes, one out-of-bounds label, one failed-episode
row set and one unknown comfort label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from env.obs import (
    EGO_DIM,
    HISTORY_FRAMES,
    LD_DIM,
    LD_POINTS,
    LD_SLOTS,
    NAV_DIM,
    NAV_ROADS,
    OD_DIM,
    OD_SLOTS,
)
from net.build_net import build_net
from pipeline.train_bc import (
    INPUT_KEYS,
    LossConfig,
    _batch_from_indices,
    build_targets,
    compute_losses,
    load_data,
    main,
    split_masks,
)


def _history_arrays(n: int) -> dict:
    t = HISTORY_FRAMES
    zeros_f = lambda *shape: np.zeros(shape, dtype=np.float32)
    zeros_i = lambda *shape: np.zeros(shape, dtype=np.int64)
    return {
        "od": zeros_f(n, t, OD_SLOTS, OD_DIM),
        "od_mask": zeros_f(n, t, OD_SLOTS),
        "od_ids": zeros_i(n, t, OD_SLOTS),
        "od_anchor": zeros_f(n, t, OD_SLOTS, 3),
        "od_taxonomy": zeros_i(n, t, OD_SLOTS, 4),
        "od_age": zeros_f(n, t, OD_SLOTS),
        "ld": zeros_f(n, t, LD_SLOTS, LD_POINTS, LD_DIM),
        "ld_mask": zeros_f(n, t, LD_SLOTS, LD_POINTS),
        "ld_ids": zeros_i(n, t, LD_SLOTS),
        "ld_road_ids": zeros_i(n, t, LD_SLOTS, 2),
        "ld_anchor": zeros_f(n, t, LD_SLOTS, 3),
        "ld_chunk_s": zeros_f(n, t, LD_SLOTS),
        "nav": zeros_f(n, t, NAV_DIM),
        "nav_mask": zeros_f(n, t, 1),
        "nav_road_mask": zeros_f(n, t, NAV_ROADS),
        "nav_road_ids": zeros_i(n, t, NAV_ROADS, 2),
        "nav_route_id": zeros_i(n, t),
        "ego": zeros_f(n, t, EGO_DIM),
        "ego_mask": zeros_f(n, t, 1),
        "pose": zeros_f(n, t, 3),
        "hist_valid": zeros_f(n, t),
    }


def _write_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    n = 8
    arrays = _history_arrays(n)

    for i in range(n):
        for t in range(HISTORY_FRAMES):
            # Two observed objects: a lead vehicle and a close slower object.
            arrays["od"][i, t, 0, :11] = np.array(
                [10.0, 0.2, 1.0, 0.0, -1.5, 0.0, 4.5, 1.8, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            arrays["od"][i, t, 1, :11] = np.array(
                [3.0, 0.5, 1.0, 0.0, -3.0, 0.0, 2.0, 1.6, 0.0, 0.0, 1.0],
                dtype=np.float32,
            )
            arrays["od_mask"][i, t, :2] = 1.0

            # Two valid lanes with different lane roles.
            arrays["ld"][i, t, 0, :4, 0] = np.array([5.0, 10.0, 15.0, 25.0])
            arrays["ld"][i, t, 0, :4, 1] = 0.1
            arrays["ld"][i, t, 0, :4, 2] = 0.0
            arrays["ld"][i, t, 0, :4, 7] = 0.0
            arrays["ld_mask"][i, t, 0, :4] = 1.0

            arrays["ld"][i, t, 1, :2, 0] = np.array([4.0, 9.0])
            arrays["ld"][i, t, 1, :2, 1] = -0.1
            arrays["ld"][i, t, 1, :2, 7] = 3.0
            arrays["ld_mask"][i, t, 1, :2] = 1.0

            arrays["nav"][i, t, :] = 0.0
            arrays["nav"][i, t, NAV_DIM - 1] = 1.0  # nav_valid
            arrays["nav_mask"][i, t, 0] = 1.0
            arrays["nav_road_mask"][i, t, 0] = 1.0

            arrays["ego"][i, t] = np.array(
                [8.0, -0.5, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 8.0, 0.0, 0.9, 0.36],
                dtype=np.float32,
            )
            arrays["ego_mask"][i, t, 0] = 1.0
            arrays["hist_valid"][i, t] = 1.0

    arrays["action"] = np.array(
        [
            [3.0, 0.1, 0.02, 7.0],
            [3.5, 0.0, -0.01, 7.5],
            [2.5, 0.2, 0.03, 6.0],
            [4.0, -0.1, -0.02, 8.0],
            [1.0, 0.0, 0.0, 1.0],
            [8.0, 0.0, 0.0, 9.0],  # dx outside CLEAN_BOUNDS high (7.5)
            [2.0, 0.1, 0.01, 5.0],
            [0.5, 0.0, 0.0, 0.5],
        ],
        dtype=np.float32,
    )

    # Scene 100 spans episodes 10 and 11: a split by episode would leak it.
    arrays["scene_seed"] = np.array([100, 100, 100, 100, 101, 101, 102, 102], dtype=np.int64)
    arrays["episode_id"] = np.array([10, 10, 11, 11, 12, 12, 13, 13], dtype=np.int64)
    arrays["step"] = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    for i, eid in enumerate(arrays["episode_id"]):
        base = 1000 + 10 * int(eid)
        arrays["od_ids"][i, :, 0] = base + 1
        arrays["od_ids"][i, :, 1] = base + 2
        arrays["ld_ids"][i, :, 0] = base + 3
        arrays["ld_ids"][i, :, 1] = base + 4
    arrays["success"] = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    arrays["collision"] = np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=np.int8)
    arrays["off_road"] = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    arrays["wrong_route"] = np.zeros(n, dtype=np.int8)
    arrays["stuck"] = np.zeros(n, dtype=np.int8)
    arrays["timeout"] = np.zeros(n, dtype=np.int8)
    arrays["expert_plan_valid"] = arrays["success"].copy()
    arrays["label_valid"] = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=np.int8)
    arrays["source"] = np.zeros(n, dtype=np.int8)
    arrays["privileged_expert_sample"] = np.ones(n, dtype=np.int8)
    arrays["residual_xy"] = np.zeros((n, 2), dtype=np.float32)
    arrays["residual_xy_norm"] = np.array(
        [0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    arrays["residual_dpsi"] = np.array(
        [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    arrays["comfortable"] = np.array([1, 1, 0, 1, 0, 1, -1, 1], dtype=np.int8)
    arrays["intervened"] = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    # `intervened` above is legacy replay; `shield` is the safety intervention.
    arrays["shield"] = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    arrays["done"] = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    arrays["feasible"] = np.array([1, 1, 1, 1, 1, 1, 0, 1], dtype=np.int8)
    arrays["label_pass"] = arrays["feasible"].copy()
    arrays["action_in_bounds"] = np.array([1, 1, 1, 1, 1, 0, 1, 1], dtype=np.int8)
    arrays["label_source"] = np.ones(n, dtype=np.int8)
    arrays["student_failure"] = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)

    arrays["history_mode"] = np.ones(n, dtype=np.float32)
    np.savez_compressed(root / "expert_bc.npz", **arrays)
    return root


def _fixture(tmp_path: Path) -> Path:
    return _write_fixture(tmp_path / "data")


def test_targets_oob_value_risk_and_plan_weights(tmp_path):
    data = load_data([_fixture(tmp_path)])
    cfg = LossConfig()
    targets = build_targets(data, cfg)

    assert targets["action_phys"].shape == (8, 4)
    assert targets["action_norm"].shape == (8, 4)
    # Raw target is not clipped; only the NLL copy is.
    assert targets["action_norm"][5, 0] > 1.0
    assert targets["label_oob"][5, 0]
    assert np.isclose(targets["action_norm_clipped"][5, 0], 1.0)
    assert np.isclose(targets["label_oob"].any(axis=1).mean(), 0.125)

    # Plan weights: invalid labels/action bounds are excluded; failed-expert
    # downweight is terminal-segment only; shield is separate from replay; the
    # residual and infeasible downweights apply to actual masks.
    np.testing.assert_allclose(
        targets["plan_weight"],
        np.array([1.0, 3.0, 0.2, 0.0, 2.0, 0.0, 0.4, 0.4], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )


def test_split_by_scene_seed_not_episode(tmp_path):
    data = load_data([_fixture(tmp_path)])
    train_mask, val_mask = split_masks(data, 0.5, 0)

    assert np.all(train_mask | val_mask)
    assert not np.any(train_mask & val_mask)
    train_seeds = set(data["scene_seed"][train_mask].tolist())
    val_seeds = set(data["scene_seed"][val_mask].tolist())
    assert train_seeds.isdisjoint(val_seeds)
    # Every row of a scene must land on one side, even when episodes differ.
    for seed in np.unique(data["scene_seed"]):
        in_train = train_mask[data["scene_seed"] == seed]
        in_val = val_mask[data["scene_seed"] == seed]
        assert in_train.all() or in_val.all()
    # Scene 100 is shared by episodes 10 and 11 and must remain intact.
    assert (train_mask[data["scene_seed"] == 100].all()
            or val_mask[data["scene_seed"] == 100].all())


def test_losses_are_finite_and_differentiable(tmp_path):
    data = load_data([_fixture(tmp_path)])
    cfg = LossConfig()
    targets = build_targets(data, cfg)
    idx = np.arange(8)
    batch = _batch_from_indices(data, targets, idx, "cpu")

    assert not ({"scene_seed", "episode_id", "expert_target_lane_id"} & set(batch))
    assert set(INPUT_KEYS) <= set(batch)

    model = build_net()
    out = model(batch)
    assert set(out["aux"]) == {"collision", "offroad", "timeout", "urgency", "progress"}

    total, components = compute_losses(out, batch, cfg)
    expected = {"total", "nll", "point", "heading", "velocity", "curvature", "accel"}
    assert set(components) == expected
    for key, value in components.items():
        assert value.ndim == 0, (key, value.shape)
        assert torch.isfinite(value), (key, value)

    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_zero_plan_weight_batch_has_zero_plan_loss(tmp_path):
    data = load_data([_fixture(tmp_path)])
    cfg = LossConfig(w_failed_expert=0.0)
    targets = build_targets(data, cfg)
    # Rows with no usable plan label: plan terms must be zero, value/risk flow.
    idx = np.array([3, 5])
    batch = _batch_from_indices(data, targets, idx, "cpu")
    model = build_net()
    out = model(batch)
    total, components = compute_losses(out, batch, cfg)

    assert float(components["nll"]) == 0.0
    assert float(components["point"]) == 0.0
    assert float(components["total"]) == 0.0


def test_one_epoch_smoke(tmp_path):
    data_dir = _fixture(tmp_path)
    out_dir = tmp_path / "out"
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("{}\n", encoding="utf-8")
    rc = main(
        [
            "--config", str(cfg_path),
            "--data", str(data_dir),
            "--out", str(out_dir),
            "--epochs", "1",
            "--batch-size", "4",
            "--lr", "1e-3",
            "--val-frac", "0.5",
            "--seed", "1",
            "--device", "cpu",
            "--nll-coef", "1.0",
        ]
    )
    assert rc == 0
    run_dirs = list(tmp_path.glob("out_*"))
    assert len(run_dirs) == 1, run_dirs
    out_dir = run_dirs[0]
    for name in ("best.pt", "last.pt", "metrics.json"):
        assert (out_dir / name).exists(), name

    ckpt = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    config = ckpt["config"]
    assert config["model_type"] == "clean"
    assert config["schema_version"] == 2
    assert len(config["action_low"]) == 4 and len(config["action_high"]) == 4
    assert config["decision"]["dt"] == 0.5
    assert config["decision"]["steps_per_policy"] == 5
    assert "hard" in config["limits"] and "comfort" in config["limits"]
    assert config["params"] > 0
    assert config["split"]["by"] == "scene_seed"
    assert set(config["split"]["train_seeds"]).isdisjoint(config["split"]["val_seeds"])
    assert config["split"]["train_seeds"]
    assert config["split"]["val_seeds"]
    assert "nll_coef" in config["loss"]
    assert "selection" in config and "single-MSE" in config["selection"]

    last = torch.load(out_dir / "last.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 1
    from pipeline.checkpointing import load_model_from_checkpoint
    loaded = load_model_from_checkpoint(str(out_dir / "best.pt"), "cpu", strict=True)
    assert loaded.hidden == 128

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["label_oob_rate"] == 0.125
    assert metrics["label_oob_rate_by_split"]["val"] > 0.0
    assert metrics["history"][0]["epoch"] == 1
    assert metrics["history"][0]["eval_split"] == "val"
    assert metrics["history"][0]["val_loss"] is not None
    assert len(metrics["history"][0]["eval_mae_physical"]) == 4
    assert metrics["best"]["split"] == "val"


def test_canonical_shard_directory_loader(tmp_path):
    source = _fixture(tmp_path)
    canonical = tmp_path / "train_bc_2k"
    canonical.mkdir()
    (canonical / "shard_0000.npz").write_bytes((source / "expert_bc.npz").read_bytes())
    data = load_data([canonical])
    assert len(data["action"]) == 8
    assert len(set(data["scene_seed"].tolist())) == 3

def test_missing_history_mode_is_rejected(tmp_path):
    data_dir = _fixture(tmp_path)
    path = data_dir / "expert_bc.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files if k != "history_mode"}
    np.savez_compressed(path, **arrays)
    with np.testing.assert_raises(ValueError):
        load_data([data_dir])


def test_fast_history_window_matches_record_path(tmp_path):
    from net.build_net import build_net
    from pipeline.scene_batches import make_scene_batches

    data = load_data([_fixture(tmp_path)])
    cfg = LossConfig()
    targets = build_targets(data, cfg)
    idx = np.arange(len(data["action"]))
    batch = _batch_from_indices(data, targets, idx, "cpu")
    model = build_net().eval()
    zero = torch.zeros(len(idx), dtype=torch.bool)
    slow, slow_enc = model.build_history_window(batch)
    fast, fast_enc = model.build_history_window_fast(batch)
    assert model._can_use_fast_history(batch)
    for key in slow.__dataclass_fields__:
        a, c = getattr(slow, key), getattr(fast, key)
        if a.dtype.is_floating_point:
            assert torch.allclose(a, c, atol=5e-6, rtol=0), key
        else:
            assert torch.equal(a, c), key
    for key in ("nav", "ego", "nav_valid", "ego_valid"):
        if key in slow_enc:
            a, c = slow_enc[key], fast_enc[key]
            if a.dtype.is_floating_point:
                assert torch.allclose(a, c, atol=5e-6, rtol=0), key
            else:
                assert torch.equal(a, c), key

    sequences = make_scene_batches(data, idx, scenes_per_batch=2)
    assert len(sequences) == 2
    assert all(sb.rows.shape[0] <= 2 for sb in sequences)
    for sb in sequences:
        for b, length in enumerate(sb.lengths):
            steps = data["step"][sb.rows[b, :length]]
            assert np.all(np.diff(steps) > 0)


def test_sequence_epoch_smoke(tmp_path):
    data_dir = _fixture(tmp_path)
    out = tmp_path / "seq"
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("{}\n", encoding="utf-8")
    rc = main([
        "--config", str(cfg_path),
        "--data", str(data_dir),
        "--out", str(out),
        "--epochs", "1",
        "--batch-size", "4",
        "--sequence-batch",
        "--scenes-per-batch", "2",
        "--tbptt-chunk", "2",
        "--val-frac", "0.5",
        "--device", "cpu",
    ])
    assert rc == 0
    assert list(tmp_path.glob("seq_*"))


def test_sequence_fifo_alignment_matches_fresh_window(tmp_path):
    from net.build_net import build_net

    data = load_data([_fixture(tmp_path)])
    cfg = LossConfig()
    targets = build_targets(data, cfg)
    # Fixture rows are ordered; use two adjacent rows from one episode.
    ep = data["episode_id"]
    rows = np.flatnonzero(ep == ep[np.flatnonzero(np.diff(data["step"]) == 1)[0]])
    rows = rows[np.argsort(data["step"][rows])]
    assert len(rows) >= 2
    model = build_net().eval()
    b0 = _batch_from_indices(data, targets, rows[:1], "cpu")
    b1 = _batch_from_indices(data, targets, rows[1:2], "cpu")
    w0, _ = model.build_history_window_fast(b0)
    enc1 = model.prepare_current_encoding(model.encoder.encode_current(b1))
    advanced = model.align_window_to_current(w0, enc1)
    pose = b1["pose"][:, -1].clone()
    pose[:] = 0.0  # projected-history fixture
    advanced = advanced.advance(
        od_content=enc1["od_content"], od_geometry=enc1["od_geometry"],
        od_type=enc1["od_type"], od_valid=enc1["od_valid"],
        od_stable_id=enc1["od_stable_id"],
        ld_content=enc1["ld_content"], ld_geometry=enc1["ld_geometry"],
        ld_type=enc1["ld_type"], ld_valid=enc1["ld_valid"],
        ld_stable_id=enc1["ld_stable_id"], ego_pose=pose,
        source=0, rollout_depth=0,
    )
    fresh, _ = model.build_history_window_fast(b1)
    for key in fresh.__dataclass_fields__:
        a, c = getattr(advanced, key), getattr(fresh, key)
        if a.dtype.is_floating_point:
            assert torch.allclose(a, c, atol=5e-6, rtol=0), key
        else:
            assert torch.equal(a, c), key


def test_fifo_append_safe_detects_reentering_entity(tmp_path):
    from net.build_net import build_net

    data = load_data([_fixture(tmp_path)])
    cfg = LossConfig()
    targets = build_targets(data, cfg)
    rows = np.flatnonzero(data["episode_id"] == data["episode_id"][0])
    rows = rows[np.argsort(data["step"][rows])]
    assert len(rows) >= 2
    model = build_net().eval()
    b0 = _batch_from_indices(data, targets, rows[:1], "cpu")
    b1 = _batch_from_indices(data, targets, rows[1:2], "cpu")
    window, _ = model.build_history_window_fast(b0)
    enc = model.prepare_current_encoding(model.encoder.encode_current(b1))
    assert model.fifo_append_safe(window, enc)
    # Inject a current entity ID that was absent from the previous frame.
    enc["od_stable_id"] = enc["od_stable_id"].clone()
    enc["od_valid"] = enc["od_valid"].clone()
    enc["od_valid"][0, 0] = True
    enc["od_stable_id"][0, 0] = 987654321
    assert not model.fifo_append_safe(window, enc)


def test_closed_loop_score_is_lexicographic_success_first():
    from pipeline.train_bc import _closed_loop_score

    good = {
        "success_rate": 0.9,
        "route_completion_mean": 1.0,
        "collision_rate": 0.0,
        "off_road_rate": 0.0,
        "infeasible_count_mean": 10.0,
    }
    bad = {
        "success_rate": 0.1,
        "route_completion_mean": 0.1,
        "collision_rate": 0.0,
        "off_road_rate": 0.0,
        "infeasible_count_mean": 0.0,
    }
    assert _closed_loop_score(good) > _closed_loop_score(bad)


def test_load_data_row_cap_keeps_whole_episodes_and_is_deterministic(tmp_path):
    fixture = _fixture(tmp_path)
    full = load_data([fixture])
    capped_a = load_data([fixture], max_rows=[4], seed=0)
    capped_b = load_data([fixture], max_rows=[4], seed=0)

    assert len(capped_a["action"]) == 4
    assert len(capped_a["action"]) < len(full["action"])
    np.testing.assert_array_equal(capped_a["episode_id"], capped_b["episode_id"])
    np.testing.assert_array_equal(capped_a["action"], capped_b["action"])
    # Every selected episode keeps all of its rows.
    for episode in np.unique(capped_a["episode_id"]):
        rows = np.flatnonzero(capped_a["episode_id"] == episode)
        source_rows = np.flatnonzero(full["episode_id"] == full["episode_id"][rows[0]])
        assert len(rows) == len(source_rows)


def test_load_data_row_cap_count_must_match_paths(tmp_path):
    with pytest.raises(ValueError, match="row caps"):
        load_data([_fixture(tmp_path)], max_rows=[1, 2])
