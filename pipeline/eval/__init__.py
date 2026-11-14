"""Closed-loop evaluation package."""

from .closed_loop import load_model, run_episode
from .metrics import (
    bootstrap_ci,
    gate_report,
    paired_bootstrap,
    paired_scene_table,
    summarize_rows,
)

__all__ = [
    "bootstrap_ci",
    "gate_report",
    "load_model",
    "paired_bootstrap",
    "paired_scene_table",
    "run_episode",
    "summarize_rows",
]
