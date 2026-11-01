"""Smoke test for the optional Phase 2b joint trainer."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch

from net.build_net import build_net
from pipeline.checkpointing import load_drive_model_from_checkpoint, save_drive_model_checkpoint
from pipeline.train_joint import main as joint_main
from test_train_bc import _fixture


def _subset_raw(src: Path, dst: Path, seeds: set[int]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    with np.load(next(src.glob("*.npz"))) as z:
        arrays = {k: z[k] for k in z.files}
    keep = np.isin(arrays["scene_seed"], list(seeds))
    arrays = {k: v[keep] for k, v in arrays.items()}
    arrays["history_mode"] = np.ones(len(arrays["action"]), dtype=np.float32)
    np.savez_compressed(dst / "expert_bc.npz", **arrays)


def test_joint_trainer_smoke(tmp_path):
    ckpt = tmp_path / "drive.pt"
    save_drive_model_checkpoint(build_net(), str(ckpt))
    out = tmp_path / "joint"
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("{}\n", encoding="utf-8")
    source = _fixture(tmp_path)
    train = tmp_path / "train_raw"
    val = tmp_path / "val_raw"
    _subset_raw(source, train, {100})
    _subset_raw(source, val, {101, 102})
    split = tmp_path / "val.json"
    split.write_text(json.dumps({"seeds": [101, 102]}), encoding="utf-8")
    rc = joint_main([
        "--config", str(cfg_path),
        "--data", str(train),
        "--val-data", str(val),
        "--val-split", str(split),
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
    assert len(metrics["val_scene_seeds"]) == 2
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
