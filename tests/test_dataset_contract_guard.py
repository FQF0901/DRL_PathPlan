"""Stage A/B 启动数据集契约守卫（2026-09-26 事故防复发）。

事故：``STAGE=B BC_EPOCHS=20 STAGE_A_OUT=... bash tools/train.sh`` 漏设 ``BC_DIR``
→ 静默回退默认 ``runs/bc_expert_full``（schema v1，无 ``others``）→ router 损失全程 0，
只打印一行 ``[bc] router 软目标不可用（...）`` 提示，结果作废却不易察觉。本文件锁定：

1. schema<2 → 训练前 ``SystemExit``，消息含当前 BC_DIR、实际内容与正确示例命令；
2. schema=2 但缺 v2 通道（``others``/``od_id``/``od_presence``）→ 同样硬失败；
3. ``--allow-legacy-dataset`` / ``ALLOW_LEGACY_DATASET=1`` → 放行 + 返回 ``legacy_dataset=True``；
4. v2 正常通过 + 启动横幅 ``[stageX] dataset=... rows=... fingerprint=... schema=v2``；
5. Stage A/B runner 接线（真实落盘 legacy fixture：训练前失败；放行路径跑完 Stage B 并在
   ``metrics.json`` 落盘 ``legacy_dataset=true``）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pipeline.stages import (
    _allow_legacy_dataset,
    _parse_args,
    run_stage_a,
    run_stage_b,
    validate_bc_dataset,
)
from pipeline.trainer import BCDataset, SUPERVISED_LABELS
from tests.v2_synthetic import TINY_MODEL_YAML, make_v2_arrays

#: 合成 fixture 中 v2 独有的键（v1 数据集没有；剔除后 = 近似真实 legacy 数据集）。
_LEGACY_DROP_KEYS = (
    "others",
    "others_mask",
    "od_id",
    "od_presence",
    "od_id_hist",
    "od_presence_hist",
    "train_weight",
    "balance_weight",
    "wm_valid",
    "frame_usable",
    "filter_reason",
)


def _legacy_arrays(episodes: int = 6, steps_per_episode: int = 6) -> dict:
    arrays, _ = make_v2_arrays(episodes=episodes, steps_per_episode=steps_per_episode)
    for key in _LEGACY_DROP_KEYS:
        arrays.pop(key, None)
    return arrays


def _legacy_dataset(**kwargs: int) -> BCDataset:
    return BCDataset(_legacy_arrays(**kwargs), {"schema_version": 1, "kind": "bc_expert"})


def _v2_dataset(fingerprint: str = "v2-testfingerprint", **kwargs: int) -> BCDataset:
    arrays, _ = make_v2_arrays(**kwargs)
    return BCDataset(
        arrays,
        {
            "schema_version": 2,
            "obs_fingerprint": fingerprint,
            "label_names": list(SUPERVISED_LABELS),
        },
    )


def _write_legacy_dataset(directory: Path, **kwargs: int) -> Path:
    """落盘近似 v1 的 fixture（无 v2 通道 + meta schema_version=1）。"""
    arrays = _legacy_arrays(**kwargs)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "expert_bc.npz", **arrays)
    meta = {
        "schema_version": 1,
        "kind": "bc_expert",
        "label_names": list(SUPERVISED_LABELS),
        "count": int(len(arrays["episode_id"])),
    }
    (directory / "expert_bc.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


def _model_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(TINY_MODEL_YAML, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 校验函数：硬失败 / 放行 / 通过 + 横幅
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stage", ["A", "B"])
def test_legacy_dataset_hard_fails_with_actionable_message(stage: str) -> None:
    dataset = _legacy_dataset()
    with pytest.raises(SystemExit) as excinfo:
        validate_bc_dataset(dataset, "runs/bc_expert_full", stage, allow_legacy=False)
    msg = str(excinfo.value)
    # 当前 BC_DIR + 它实际是什么（schema/缺通道/行数）
    assert "runs/bc_expert_full" in msg
    assert "schema_version=1" in msg
    for channel in ("others", "od_id", "od_presence"):
        assert channel in msg
    assert f"rows={dataset.count}" in msg
    # 正确示例命令 + 逃生门提示
    assert "BC_DIR=runs/bc_expert_5k_v2" in msg
    assert f"STAGE={stage}" in msg
    assert "--allow-legacy-dataset" in msg


def test_schema2_missing_channel_hard_fails() -> None:
    arrays, _ = make_v2_arrays(episodes=2, steps_per_episode=6)
    arrays.pop("others")
    arrays.pop("others_mask")
    dataset = BCDataset(arrays, {"schema_version": 2})
    with pytest.raises(SystemExit) as excinfo:
        validate_bc_dataset(dataset, "runs/bc_partial_v2", "B", allow_legacy=False)
    msg = str(excinfo.value)
    assert "runs/bc_partial_v2" in msg
    assert "others" in msg


def test_allow_legacy_releases_and_marks(capsys: pytest.CaptureFixture[str]) -> None:
    dataset = _legacy_dataset()
    marker = validate_bc_dataset(dataset, "runs/bc_expert_full", "B", allow_legacy=True)
    assert marker["legacy_dataset"] is True
    out = capsys.readouterr().out
    assert "警告" in out and "legacy" in out and "legacy_dataset=true" in out


@pytest.mark.parametrize("stage", ["A", "B"])
def test_v2_dataset_passes_and_prints_banner(
    stage: str, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = _v2_dataset()
    marker = validate_bc_dataset(dataset, "runs/bc_expert_5k_v2", stage, allow_legacy=False)
    assert marker["legacy_dataset"] is False
    assert marker["dataset_fingerprint"] == "v2-testfingerprint"
    out_lines = capsys.readouterr().out.splitlines()
    banner = (
        f"[stage{stage}] dataset=runs/bc_expert_5k_v2 rows={dataset.count} "
        "fingerprint=v2-testfingerprint schema=v2"
    )
    assert banner in out_lines, f"启动横幅格式不符：{out_lines}"


def test_cli_and_env_escape_hatch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _allow_legacy_dataset(_parse_args(["--stage", "B"])) is False
    assert _allow_legacy_dataset(_parse_args(["--stage", "B", "--allow-legacy-dataset"])) is True
    monkeypatch.setenv("ALLOW_LEGACY_DATASET", "1")
    assert _allow_legacy_dataset(_parse_args(["--stage", "A"])) is True
    monkeypatch.setenv("ALLOW_LEGACY_DATASET", "0")
    assert _allow_legacy_dataset(_parse_args(["--stage", "A"])) is False


# --------------------------------------------------------------------------- #
# Stage A/B runner 接线（真实落盘 fixture）
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stage,runner", [("A", run_stage_a), ("B", run_stage_b)])
def test_stage_runner_hard_fails_before_training(
    stage: str, runner, tmp_path: Path
) -> None:
    dataset_dir = _write_legacy_dataset(tmp_path / "bc_legacy")
    out_dir = tmp_path / f"stage_{stage.lower()}"
    args = _parse_args([
        "--stage", stage, "--bc-dir", str(dataset_dir), "--out", str(out_dir),
        "--model-config", str(_model_cfg(tmp_path)),
        "--ckpt", str(tmp_path / "missing.pt"),  # 不载入真实 checkpoint
        "--device", "cpu",
    ])
    with pytest.raises(SystemExit) as excinfo:
        runner(args, {})
    msg = str(excinfo.value)
    assert str(dataset_dir) in msg
    assert "BC_DIR=runs/bc_expert_5k_v2" in msg and f"STAGE={stage}" in msg
    assert not (out_dir / "final.pt").exists(), "契约失败不得产出 checkpoint"


def test_allow_legacy_stage_b_runs_and_marks_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset_dir = _write_legacy_dataset(tmp_path / "bc_legacy")
    out_dir = tmp_path / "stage_b"
    args = _parse_args([
        "--stage", "B", "--bc-dir", str(dataset_dir), "--out", str(out_dir),
        "--ckpt", str(tmp_path / "missing.pt"),
        "--model-config", str(_model_cfg(tmp_path)),
        "--bc-epochs", "1", "--batch-size", "8", "--val-frac", "0.34",
        "--cluster-config", str(tmp_path / "no_such_cluster.yaml"),  # 跳过聚类软目标
        "--allow-legacy-dataset", "--device", "cpu", "--seed", "0",
    ])
    metrics = run_stage_b(args, {})
    assert metrics["legacy_dataset"] is True
    payload = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["legacy_dataset"] is True
    out = capsys.readouterr().out
    assert "放行 legacy" in out
