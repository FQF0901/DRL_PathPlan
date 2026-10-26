"""Backward-compatible import location for the action contract loader.

The canonical loader lives in :mod:`env.action.config`; this module only
re-exports it so older imports keep working without creating a second parser.
"""

from env.action.config import ActionContract, DEFAULT_CONFIG_PATH, load_action_contract

__all__ = ["ActionContract", "load_action_contract", "DEFAULT_CONFIG_PATH"]
