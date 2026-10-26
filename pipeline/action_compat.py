"""Config-backed compatibility names for legacy pipeline imports.

The canonical objects come from :func:`env.action.config.load_action_contract`.
New code should import the contract directly; these aliases exist only so the
legacy live path does not carry a second hard-coded action table.
"""

from env.action.config import load_action_contract

_CONTRACT = load_action_contract()

CLEAN_BOUNDS = _CONTRACT.bounds
DEFAULT_BOUNDS = CLEAN_BOUNDS
HARD_LIMITS = _CONTRACT.hard
COMFORT_LIMITS = _CONTRACT.comfort

__all__ = ["CLEAN_BOUNDS", "DEFAULT_BOUNDS", "HARD_LIMITS", "COMFORT_LIMITS"]
