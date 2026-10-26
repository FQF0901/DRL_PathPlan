"""Smoke tests for the bank-native WM trainer/probe."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from pipeline.checkpointing import load_drive_model_from_checkpoint, save_drive_model_checkpoint
from pipeline.legacy_model import StructuredPlanner
from pipeline.probe_wm import main as probe_main
from pipeline.train_wm import main as wm_main
from test_train_bc import _write_fixture


def test_train_wm_and_probe_smoke(tmp_path):
    data = _write_fixture(tmp_path / "data")
    adapter = StructuredPlanner(hidden=32, heads=4, spatial_layers=1)
    drive_path = tmp_path / "encoder.pt"
    save_drive_model_checkpoint(adapter.drive, str(drive_path))

    out = tmp_path / "wm"
    rc = wm_main([
        "--data", str(data),
        "--ckpt", str(drive_path),
        "--out", str(out),
        "--epochs", "1",
        "--batch-size", "4",
        "--max-pairs", "20",
        "--val-frac", "0.5",
        "--seed", "1",
        "--device", "cpu",
    ])
    assert rc == 0
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
