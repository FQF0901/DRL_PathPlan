"""Smoke test for the optional Phase 2b joint trainer."""

from __future__ import annotations

import json
import torch

from net.build_net import build_net
from pipeline.checkpointing import load_drive_model_from_checkpoint, save_drive_model_checkpoint
from pipeline.train_joint import main as joint_main


def test_joint_trainer_smoke(tmp_path):
    ckpt = tmp_path / "drive.pt"
    save_drive_model_checkpoint(build_net(), str(ckpt))
    out = tmp_path / "joint"
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("{}\n", encoding="utf-8")
    rc = joint_main([
        "--config", str(cfg_path),
        "--data", "data/episodes/mini/il_train_100",
        "--val-data", "data/episodes/mini/val_20_expert",
        "--val-split", "data/splits/mini_val_20.json",
        "--ckpt", str(ckpt),
        "--out", str(out),
        "--epochs", "1",
        "--batch-size", "64",
        "--max-pairs", "100",
        "--tsb-lr", "1e-6",
        "--lr", "2e-5",
        "--device", "cpu",
    ])
    assert rc == 0
    run_dirs = list(tmp_path.glob("joint_*"))
    assert len(run_dirs) == 1, run_dirs
    out = run_dirs[0]
    assert (out / "best_drive.pt").exists()
    assert (out / "last_drive.pt").exists()
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["best_epoch"] >= 0
    assert len(metrics["val_scene_seeds"]) == 20
    loaded = load_drive_model_from_checkpoint(str(out / "best_drive.pt"), "cpu", strict=True)
    initial = load_drive_model_from_checkpoint(str(ckpt), "cpu", strict=True)
    # Joint training must actually step policy/TSB heads, not only measure them.
    changed = any(
        not torch.equal(a, b)
        for a, b in zip(initial.planner.parameters(), loaded.planner.parameters())
    )
    assert changed, "joint trainer did not update planner parameters"
    assert all(
        torch.equal(a, b)
        for a, b in zip(initial.encoder.parameters(), loaded.encoder.parameters())
    )
