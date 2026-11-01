"""Small YAML-config layer shared by the training entry points.

The YAML file supplies argparse defaults; explicit CLI flags still win.  This
keeps the existing argparse API usable while making the checked-in config files
the normal source of training defaults.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


def _find_config_arg(argv: Sequence[str], default: str | Path) -> Path:
    value = None
    for i, arg in enumerate(argv):
        if arg == "--config":
            if i + 1 >= len(argv):
                raise ValueError("--config needs a path")
            value = argv[i + 1]
        elif arg.startswith("--config="):
            value = arg.split("=", 1)[1]
    return Path(default if value is None else value)


def load_train_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"training config must be a mapping: {path}")
    return dict(raw)


def parse_train_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    default_config: str | Path,
) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    path = _find_config_arg(argv, default_config)
    cfg = load_train_config(path)

    known = {action.dest for action in parser._actions if action.dest != "help"}
    unknown = sorted(set(cfg) - known)
    if unknown:
        raise ValueError(f"unknown training config keys in {path}: {unknown}")

    parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    args.config = str(path)
    if getattr(args, "device", None) == "auto":
        args.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    return args


def timestamped_run_dir(out: str | Path) -> Path:
    """Append ``_yymmddhhmmss`` to a configured run directory."""
    path = Path(out)
    stamp = datetime.now().strftime("%y%m%d%H%M%S")
    return path.with_name(f"{path.name}_{stamp}")


def latest_run_alias(run_dir: str | Path) -> Path:
    """Create/replace a stable sibling symlink for the timestamped run.

    The actual run directory keeps its timestamp; ``runs/name`` points at the
    newest run so downstream YAML can use a stable checkpoint path.
    """
    run = Path(run_dir)
    name = run.name
    suffix = name[-13:] if len(name) > 13 else ""
    if not (suffix.startswith("_") and suffix[1:].isdigit()):
        return run
    alias = run.with_name(name[:-13])
    try:
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.is_symlink():
            alias.unlink()
        elif alias.exists():
            return run
        alias.symlink_to(run.name, target_is_directory=True)
    except OSError:
        # A filesystem without symlink permission still leaves the real run
        # directory usable; operators can pass --ckpt explicitly.
        pass
    return alias
