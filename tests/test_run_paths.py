"""入口工具只读契约：``pipeline/run_paths.py`` 布局/发现 + ``tools/run_config.py`` 取值 + 两个 .sh 的行数。

纯离线（不建 env、不跑训练）：合成 datasets/ 与 runs/ 目录验证 work_dir/共戳/resume/ckpt 发现。
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from pipeline import run_paths
from tools import run_config

ROOT = Path(__file__).resolve().parents[1]
_ENV_KEYS = ("STAGE", "RESUME", "BC_DIR", "WORK_DIR", "CKPT", "STAGE_A_CKPT", "EVAL_CKPT", "NAME", "WM_EPOCHS")


def _dataset(root: Path, name: str) -> Path:
    path = root / "datasets" / name
    path.mkdir(parents=True)
    (path / "expert_bc.npz").write_bytes(b"")
    return path


def _run(root: Path, stamp: str, name: str = "train") -> Path:
    path = root / "runs" / f"BTC{stamp}_{name}"
    path.mkdir(parents=True)
    return path


def test_layout_naming_and_paths(tmp_path: Path) -> None:
    work = run_paths.new_work_dir("train", tmp_path, stamp="20260101-0000")
    assert work == tmp_path / "runs/BTC20260101-0000_train"
    assert run_paths.stage_dir(work, "A") == work / "stage_a"
    assert run_paths.stage_dir(work, "B") == work / "stage_b"
    assert run_paths.stage_log(work, "B") == work / "logs/stage_b.log"
    assert run_paths.detach_pid_path(work) == work / "detach.pid"
    assert run_paths.work_dir_of_out(work / "stage_a") == work
    layout = run_paths.prepare_layout(work, "A")
    assert layout["stage"].is_dir() and layout["logs"].is_dir() and layout["detach_pid"].parent == work


def test_dataset_and_run_discovery(tmp_path: Path) -> None:
    _dataset(tmp_path, "BTC20260101-0000_expert5k")
    _dataset(tmp_path, "BTC20260202-0000_expert5k")
    assert run_paths.latest_dataset(tmp_path).name == "BTC20260202-0000_expert5k"

    finished = _run(tmp_path, "20260101-0000") / "stage_a"
    finished.mkdir()
    (finished / "ckpt_epoch005.pt").write_bytes(b"")
    (finished / "final.pt").write_bytes(b"")
    unfinished = _run(tmp_path, "20260202-0000") / "stage_b"
    unfinished.mkdir()
    (unfinished / "ckpt_epoch010.pt").write_bytes(b"")

    assert run_paths.latest_final(tmp_path, "A") == finished / "final.pt"
    assert run_paths.latest_final(tmp_path, "B") is None
    found = run_paths.find_unfinished(tmp_path)
    assert found is not None
    work_dir, out_dir, stage, ckpt = found
    assert (work_dir.name, out_dir.name, stage, ckpt.name) == (
        "BTC20260202-0000_train", "stage_b", "B", "ckpt_epoch010.pt",
    )

    # 仍在运行（detach.pid 存活）的 run 不被 resume auto 选中
    live = work_dir / "detach.pid"
    live.write_text(str(os.getpid()), encoding="utf-8")
    assert run_paths.find_unfinished(tmp_path) is None
    live.write_text("0", encoding="utf-8")
    assert run_paths.find_unfinished(tmp_path) is not None


def test_resolve_train_shares_run_root_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    _dataset(tmp_path, "BTC20260101-0000_expert5k")
    stage_a = _run(tmp_path, "20260101-0000") / "stage_a"
    stage_a.mkdir()
    (stage_a / "final.pt").write_bytes(b"")
    cfg = {
        "run": {"name": "train", "work_dir": "auto", "bc_dir": "auto", "resume": "auto",
                "stage_a_ckpt": "auto", "eval_ckpt": "auto", "checkpoint_interval_epochs": 5},
        "stages": {"A": {"epochs": 20, "batch": 1024, "micro": 256},
                   "B": {"epochs": 20, "batch": 1024, "micro": 512, "traj_aux_weight": 0.1}},
    }

    fresh = run_config.resolve(tmp_path, cfg)
    assert fresh["STAGE"] == "A"
    assert re.fullmatch(r".*runs/BTC\d{8}-\d{4}_train", fresh["WORK_DIR"]), fresh["WORK_DIR"]
    assert fresh["STAGE_DIR"].endswith("/stage_a") and fresh["LOG"].endswith("/logs/stage_a.log")
    assert fresh["BC_DIR"].endswith("BTC20260101-0000_expert5k") and fresh["CKPT"] == "" and fresh["RESUME"] == ""

    stage_b = run_config.resolve(tmp_path, cfg, stage_arg="B")
    assert stage_b["WORK_DIR"].endswith("runs/BTC20260101-0000_train")   # B 复用 A run 根（同戳）
    assert stage_b["STAGE_DIR"].endswith("/stage_b") and stage_b["CKPT"].endswith("stage_a/final.pt")
    assert stage_b["TRAJ_AUX_WEIGHT"] == "0.1" and stage_b["RESUME"] == ""
    assert run_config.resolve_eval(tmp_path, cfg)["CKPT"].endswith("stage_a/final.pt")

    # resume auto：最新未完成 stage 原地续跑（work_dir/out 都来自该 run）
    pending = _run(tmp_path, "20260202-0000") / "stage_a"
    pending.mkdir()
    (pending / "ckpt_epoch003.pt").write_bytes(b"")
    resumed = run_config.resolve(tmp_path, cfg)
    assert resumed["STAGE"] == "A" and resumed["WORK_DIR"].endswith("runs/BTC20260202-0000_train")
    assert resumed["RESUME"].endswith("ckpt_epoch003.pt")

    monkeypatch.setenv("RESUME", "")   # 显式置空 = 关闭续跑
    assert run_config.resolve(tmp_path, cfg)["RESUME"] == ""


def test_entry_scripts_thin_and_parse() -> None:
    for rel in ("tools/train.sh", "tools/test.sh"):
        path = ROOT / rel
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 30, rel
