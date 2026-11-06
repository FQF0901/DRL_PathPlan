"""Smoke tests for the bank-native WM trainer/probe."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from pipeline.checkpointing import load_drive_model_from_checkpoint, save_drive_model_checkpoint
from net.build_net import build_net
from pipeline.probe_wm import main as probe_main
from pipeline.train_wm import main as wm_main
from test_train_bc import _write_fixture


def test_train_wm_and_probe_smoke(tmp_path):
    data = _write_fixture(tmp_path / "data")
    drive = build_net()
    drive_path = tmp_path / "encoder.pt"
    save_drive_model_checkpoint(drive, str(drive_path))

    out = tmp_path / "wm"
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("{}\n", encoding="utf-8")
    rc = wm_main([
        "--config", str(cfg_path),
        "--data", str(data),
        "--ckpt", str(drive_path),
        "--out", str(out),
        "--epochs", "1",
        "--batch-size", "4",
        "--max-pairs", "20",
        "--val-frac", "0.5",
        "--allow-random-val",
        "--seed", "1",
        "--device", "cpu",
    ])
    assert rc == 0
    run_dirs = list(tmp_path.glob("wm_*"))
    assert len(run_dirs) == 1, run_dirs
    out = run_dirs[0]
    assert (out / "best.pt").exists()
    assert (out / "best_drive.pt").exists()
    payload = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    cfg = payload["config"]
    assert set(cfg["train_scene_seeds"]).isdisjoint(cfg["val_scene_seeds"])
    drive = load_drive_model_from_checkpoint(str(out / "best_drive.pt"), "cpu", strict=True)
    assert drive.world_model is not None

    probe_out = tmp_path / "probe"
    rc = probe_main([
        "--data", str(data),
        "--encoder-ckpt", str(drive_path),
        "--wm-ckpt", str(out / "best.pt"),
        "--out", str(probe_out),
        "--horizon", "1",
        "--max-sequences", "2",
        "--device", "cpu",
    ])
    assert rc == 0
    metrics = json.loads((probe_out / "metrics.json").read_text())
    assert "held_out_split" in metrics
    assert metrics["depths"][0]["depth"] == 1


def test_wm_alignment_rejects_zero_and_target_duplicates():
    import numpy as np
    import pytest
    from pipeline.train_wm import _align_by_id, _usable_current_ids

    with pytest.raises(ValueError, match="stable_id=0"):
        _usable_current_ids(np.array([0, 2]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="duplicate stable"):
        _usable_current_ids(np.array([2, 2]), np.array([1.0, 1.0]))

    cur = np.zeros((1, 4), dtype=np.float32)
    tgt = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="duplicate target stable ID"):
        _align_by_id(
            cur, np.array([2]), np.array([1.0]),
            tgt, np.array([2, 2]), np.array([1.0, 1.0]),
        )
