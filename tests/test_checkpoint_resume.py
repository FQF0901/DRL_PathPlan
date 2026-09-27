"""周期检查点 + resume（2026-09-27）：payload 键集/同格式、A/B 周期保存与续跑。

动机（两次真实事故）：训练只在阶段末保存 ckpt → 跑到 19/20 epoch 被静默杀掉后无任何
中间态。本测试覆盖：

- ``save_checkpoint`` 固定键集（周期 ckpt 与 ``final.pt`` 可跨文件比较）；
- RNG 状态捕获/恢复（数据顺序来源的 ``default_rng`` 生成器）；
- Stage A：``--ckpt-every`` 周期保存 + ``--resume`` 从 epoch 3 续跑；
- Stage B：周期保存 + 跨 primary/specific 相位边界的 resume（primary 跳过、specific 续跑）。
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline.stages import (
    _load_yaml,
    _parse_args,
    _resolve_ckpt_every,
    build_model,
    run_stage_a,
    run_stage_b,
)
from pipeline.trainer import (
    capture_rng_state,
    config_snapshot_hash,
    load_training_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from tests.v2_synthetic import TINY_MODEL_YAML, write_v2_dataset

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

_CKPT_KEYS = {"model", "meta", "optimizer", "epoch", "val_metrics", "rng_state", "config_hash"}


def _dataset(tmp_path: Path) -> Path:
    directory = write_v2_dataset(tmp_path / "bc_v2", episodes=6, steps_per_episode=6)
    return directory


def _model_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(TINY_MODEL_YAML, encoding="utf-8")
    return path


def _ckpt_keys(path: Path) -> set:
    return set(torch.load(path, map_location="cpu", weights_only=False))


def test_save_checkpoint_fixed_keys_and_rng_roundtrip(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()

    generator = np.random.default_rng(7)
    generator.random(4)
    state = capture_rng_state(generator)
    expected = generator.random(3)

    path = tmp_path / "ckpt_epoch002.pt"
    save_checkpoint(
        path,
        model,
        meta={"stage": "T"},
        optimizer=optimizer,
        epoch=2,
        val_metrics={"val_loss": 1.0},
        rng_state=state,
        config_hash=config_snapshot_hash({"train": {"ckpt_every": 5}}),
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) == _CKPT_KEYS  # 固定键集：周期 ckpt 与 final.pt 同格式
    assert payload["epoch"] == 2 and payload["optimizer"] is not None

    # RNG 状态恢复：生成器后续序列与捕获时一致（global 状态先存后还原，避免污染其它测试）
    numpy_state, python_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    try:
        replica = np.random.default_rng(0)
        restore_rng_state(state, numpy_generator=replica)
        np.testing.assert_allclose(replica.random(3), expected)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)

    info = load_training_checkpoint(str(path), model)
    assert info["epoch"] == 2
    assert info["val_metrics"] == {"val_loss": 1.0}
    assert info["config_hash"] == config_snapshot_hash({"train": {"ckpt_every": 5}})
    assert info["optimizer"] is not None and info["rng_state"]


def test_resolve_ckpt_every_priority() -> None:
    args_default = _parse_args(["--stage", "A"])
    assert _resolve_ckpt_every(args_default, {}) == 5  # 默认 5
    assert _resolve_ckpt_every(args_default, {"train": {"ckpt_every": 3}}) == 3  # config 覆盖默认
    args_off = _parse_args(["--stage", "A", "--ckpt-every", "0"])
    assert _resolve_ckpt_every(args_off, {"train": {"ckpt_every": 3}}) == 0  # CLI 优先且 0=关


def test_stage_a_periodic_ckpt_then_resume(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    dataset_dir = _dataset(tmp_path)
    model_cfg = _model_cfg(tmp_path)
    common = ["--bc-dir", str(dataset_dir), "--model-config", str(model_cfg),
              "--device", "cpu", "--seed", "0", "--batch-size", "8", "--eval-frames", "8",
              "--val-frac", "0.34", "--ckpt-every", "2"]

    out = tmp_path / "stage_a"
    metrics = run_stage_a(_parse_args(["--stage", "A", "--out", str(out), "--wm-epochs", "4", *common]), {})
    assert metrics["ckpt_every"] == 2 and metrics["last_epoch"] == 4
    assert (out / "ckpt_epoch002.pt").exists() and (out / "ckpt_epoch004.pt").exists()
    assert not (out / "ckpt_epoch003.pt").exists()
    assert _ckpt_keys(out / "ckpt_epoch002.pt") == _ckpt_keys(out / "final.pt") == _CKPT_KEYS
    info = load_training_checkpoint(str(out / "ckpt_epoch002.pt"), build_model(_load_yaml(str(model_cfg))))
    assert info["epoch"] == 2 and info["optimizer"] is not None and info["rng_state"]

    # resume：从 epoch 2 的 ckpt 续跑到 6（每 2 轮保存 → 只在 4/6 落盘）
    out_resume = tmp_path / "stage_a_resume"
    metrics_resume = run_stage_a(
        _parse_args(["--stage", "A", "--out", str(out_resume), "--wm-epochs", "6",
                     "--resume", str(out / "ckpt_epoch002.pt"), *common]),
        {},
    )
    printed = capsys.readouterr().out
    assert "从 epoch 3 继续" in printed, printed
    assert metrics_resume["resume_epoch"] == 2 and metrics_resume["last_epoch"] == 6
    assert metrics_resume["resumed_from"] == str(out / "ckpt_epoch002.pt")
    assert (out_resume / "ckpt_epoch004.pt").exists() and (out_resume / "ckpt_epoch006.pt").exists()
    assert not (out_resume / "ckpt_epoch002.pt").exists()
    assert (out_resume / "final.pt").exists()


def test_stage_b_periodic_ckpt_then_resume_across_phase(tmp_path: Path) -> None:
    dataset_dir = _dataset(tmp_path)
    model_cfg = _model_cfg(tmp_path)
    stage_a = tmp_path / "stage_a"
    run_stage_a(
        _parse_args(["--stage", "A", "--bc-dir", str(dataset_dir), "--out", str(stage_a),
                     "--model-config", str(model_cfg), "--wm-epochs", "1", "--batch-size", "8",
                     "--eval-frames", "8", "--device", "cpu", "--seed", "0"]),
        {},
    )

    # 4 epochs = primary 2 + specific 2 → ckpt_epoch003 落在 specific 段中间
    common = ["--bc-dir", str(dataset_dir), "--model-config", str(model_cfg), "--bc-epochs", "4",
              "--bc-phase-split", "0.5", "--batch-size", "8", "--val-frac", "0.34",
              "--device", "cpu", "--seed", "0", "--ckpt-every", "1"]
    out = tmp_path / "stage_b"
    metrics = run_stage_b(
        _parse_args(["--stage", "B", "--out", str(out), "--ckpt", str(stage_a / "final.pt"), *common]),
        {},
    )
    assert metrics["ckpt_every"] == 1 and metrics["primary"]["last_epoch"] == 2
    for index in range(1, 5):
        assert (out / f"ckpt_epoch{index:03d}.pt").exists()
    assert _ckpt_keys(out / "ckpt_epoch003.pt") == _ckpt_keys(out / "final.pt") == _CKPT_KEYS

    # 场景 1（相位边界）：ckpt_epoch002 = primary 完成 → primary 跳过、specific 从本地 0 全跑
    out_resume = tmp_path / "stage_b_resume"
    metrics_resume = run_stage_b(
        _parse_args(["--stage", "B", "--out", str(out_resume), "--resume", str(out / "ckpt_epoch002.pt"), *common]),
        {},
    )
    assert metrics_resume["resume_epoch"] == 2
    assert metrics_resume["primary"]["skipped"] is True
    assert metrics_resume["specific"]["last_epoch"] == 2
    # specific 全段重跑 → 与原始 specific 汇总逐值一致（数据顺序/超参确定）
    for key in ("bc_loss", "bc_traj_loss", "bc_action_loss", "bc_action_mu_ds_mean"):
        assert metrics_resume["specific"][key] == pytest.approx(metrics["specific"][key], rel=1e-9)
    assert (out_resume / "ckpt_epoch003.pt").exists() and (out_resume / "ckpt_epoch004.pt").exists()

    # 场景 2（相位内 mid-specific）：ckpt_epoch003 = specific 本地 1/2 → 本地 epoch 2 续跑后
    # 模型/优化器状态应与原始 ckpt_epoch004 逐位一致（RNG + optimizer 状态恢复的精度边界）
    out_mid = tmp_path / "stage_b_mid"
    metrics_mid = run_stage_b(
        _parse_args(["--stage", "B", "--out", str(out_mid), "--resume", str(out / "ckpt_epoch003.pt"), *common]),
        {},
    )
    assert metrics_mid["resume_epoch"] == 3
    assert metrics_mid["specific"]["last_epoch"] == 2  # 只跑了 specific 本地 epoch 2
    original = torch.load(out / "ckpt_epoch004.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(out_mid / "ckpt_epoch004.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(original["model"][key], resumed["model"][key]) for key in original["model"])
    assert (out_mid / "final.pt").exists()
