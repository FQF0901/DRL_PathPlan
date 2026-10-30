"""Training YAML config merge tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.train_bc import build_arg_parser
from pipeline.train_config import load_train_config, parse_train_args, timestamped_run_dir


def test_train_bc_yaml_defaults_and_cli_override(tmp_path):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "data:\n  - from-config\nout: runs/from-config\nepochs: 3\nbatch_size: 8\n"
        "device: auto\n",
        encoding="utf-8",
    )
    args = parse_train_args(build_arg_parser(), ["--config", str(cfg)], "config/train_bc.yaml")
    assert args.data == ["from-config"]
    assert args.out == "runs/from-config"
    assert args.epochs == 3
    assert args.batch_size == 8
    assert args.device in {"cpu", "cuda"}

    args = parse_train_args(
        build_arg_parser(),
        ["--config", str(cfg), "--epochs", "9", "--out", "runs/cli"],
        "config/train_bc.yaml",
    )
    assert args.epochs == 9
    assert args.out == "runs/cli"


def test_checked_in_bc_config_is_loadable():
    cfg = load_train_config("config/train_bc.yaml")
    assert cfg["epochs"] >= 1
    assert isinstance(cfg["data"], list)
    assert cfg["val_data"]
    args = parse_train_args(build_arg_parser(), [], "config/train_bc.yaml")
    assert args.data == cfg["data"]
    assert args.out == cfg["out"]
    assert args.val_data == cfg["val_data"]


def test_timestamped_run_dir_suffix():
    out = timestamped_run_dir("runs/example")
    assert out.parent == Path("runs")
    assert out.name.startswith("example_")
    assert len(out.name.rsplit("_", 1)[1]) == 12


def test_unknown_config_key_fails(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("not_a_training_argument: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown training config"):
        parse_train_args(build_arg_parser(), ["--config", str(cfg)], "config/train_bc.yaml")


def test_missing_config_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_train_args(
            build_arg_parser(),
            ["--config", str(tmp_path / "missing.yaml")],
            "config/train_bc.yaml",
        )
