"""4D action ``(dx, dy, dpsi, v_next)`` -> PSTL trajectory / speed execution layer.

Scope
-----
This is the canonical P0/P1 execution layer.  The previous 3D/legacy
implementation is available only from git history (tag ``pre-consolidation``),
not as a second live path.

PROVISIONAL data-audit facts (do not treat as final; re-audit before training)
-----------------------------------------------------------------------------
Expert BC 2k rows (provisional)::

    dx     p99.9 = 6.94, max 6.95
    dy     p0.1/p99.9 = -0.79 / +0.57, max +/-1.25
    dpsi   p99.9 = -0.28 / +0.18, DAgger max +/-0.789
    v_next max = 13.90
    stop rows = 0.975 %; stop-to-go rows = 45

The canonical action box is ``[0,7.5],[-1.5,1.5],[-0.8,0.8],[0,15]``
(:data:`CLEAN_BOUNDS` / :data:`DEFAULT_BOUNDS`).

Hard/comfort feasibility limits below are provisional audit recommendations::

    hard:   a_long [-10, 3.5], |a_lat| <= 8, |kappa| <= 0.345,
            |kappa_rate| <= 1.0, |jerk_long| <= 20, |jerk_lat| <= 20
    comfort:a_long [-4, 2],  |a_lat| <= 3, |kappa| <= 0.20,
            |kappa_rate| <= 0.05, |jerk_long| <= 5, |jerk_lat| <= 5

Curvature / lateral-acceleration checks are only applied when the path arc
length is at least 0.25 m (short segments are dominated by numerical noise).
Stop class: ``arc_length < 0.10 and v_next < 0.10 and |dpsi| <= 0.05``.

PSTL (path + speed time law)
----------------------------
A 4D action is executed in two decoupled pieces.

1. **Geometric path** ``P(u)``, ``u in [0, 1]``: cubic Hermite

   ``p0 = (0, 0)``, ``p1 = (dx, dy)``,
   ``m0 = rho * ||p1|| * (1, 0)``,
   ``m1 = rho * ||p1|| * (cos dpsi, sin dpsi)``, ``rho = 1.0``.

   ``P(u) = h00 p0 + h10 m0 + h01 p1 + h11 m1``.  The path arc length ``L``
   is the 16-point Gauss-Legendre integral of ``||dP/du||`` over ``[0, 1]``.

2. **Time law** for ``tau = t / T in [0, 1]`` (``T = dt``)::

       s(tau) = T*v0*h10(tau) + L*h01(tau) + T*v1*h11(tau)

   with ``h10 = tau^3 - 2 tau^2 + tau``, ``h01 = -2 tau^3 + 3 tau^2``,
   ``h11 = tau^3 - tau^2``.  Therefore

       s(0) = 0,          s(1) = L,
       s_dot(0)/T = v0,   s_dot(1)/T = v1,
       accel(tau) = s_ddot(tau) / T^2,
       jerk(tau)  = s_dddot / T^3 = 6 * (T*(v0+v1) - 2L) / T^3.

   Samples are taken at ``tau = 1/hz, ..., 1``; positions are ``P(s/L)``,
   heading is the path tangent (endpoint forced to ``dpsi``), longitudinal
   acceleration is the time-law acceleration, and lateral acceleration is
   ``kappa * speed^2``.

Endpoint proof
--------------
Because ``P(1) = p1`` and the sampled time law ends at ``s = L``, the last
sample is exactly ``(dx, dy)``; its tangent is ``m1 ∝ (cos dpsi, sin dpsi)``
for non-degenerate paths, and that heading is forced explicitly.  The speed
at the last sample is ``s_dot(1)/T = v1`` exactly.  Thus ``v_next`` is an
absolute endpoint speed, not a chord-averaged approximation.

Degenerate actions
------------------
* full stop (``v0 = v1 = 0``, ``||p1|| = 0``, ``|dpsi| <= 0.05``): hold at the
  origin, zero speed, heading ramps to ``dpsi``; flag ``FULL_STOP_HOLD``.
* zero length with speed (``||p1|| = 0`` and ``v0 > 0`` or ``v1 > 0``):
  rejected with code ``ZERO_LENGTH_WITH_SPEED`` (no invented geometry).
* rotation in place (``||p1|| = 0``, ``v0 = v1 = 0``, ``|dpsi| > 0.05``):
  rejected with code ``ROTATION_IN_PLACE``.

Nothing here rewrites ``v_next`` as a function of ``v0``: projection only
clips to the static :class:`ActionBounds` box and reports the result.  The
state-dependent part is the *feasibility diagnosis* (speeds/accelerations),
never a silent action rewrite.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

__all__ = [
    "ActionBounds",
    "DynamicsLimits",
    "ActionTrajectory",
    "Trajectory",
    "ProjectionResult",
    "V1Interval",
    "GateResult",
    "CLEAN_BOUNDS",
    "DEFAULT_BOUNDS",
    "HARD_LIMITS",
    "COMFORT_LIMITS",
    "normalize",
    "denormalize",
    "clip_components",
    "action_to_trajectory",
    "feasible_v1_interval",
    "gate",
    "project",
]

#: Numerical tolerance for path/tangent degenerate checks.
_EPS = 1e-9
#: ``|dpsi|`` above which a zero-length stop is a rotation-in-place reject.
_EPS_ROTATION = 0.05
#: Curvature/lateral checks apply only to paths at least this long (m).
_MIN_CURVATURE_PATH = 0.25
#: Hard/comfort comparison slack (floating-point noise only).
_LIMIT_TOL = 1e-9
#: Gauss-Legendre nodes/weights for the arc-length integral.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)


# ======================================================================================
# Bounds and limits
# ======================================================================================
@dataclass(frozen=True, eq=False)
class ActionBounds:
    """Static box for ``(dx, dy, dpsi, v_next)``.

    Bounds are *static*: they never depend on ``v0``.  ``DEFAULT_BOUNDS`` is
    the canonical box (:data:`CLEAN_BOUNDS`).
    """

    low: np.ndarray
    high: np.ndarray
    name: str = "custom"

    def __post_init__(self) -> None:
        low = np.array(self.low, dtype=np.float64).reshape(-1)
        high = np.array(self.high, dtype=np.float64).reshape(-1)
        if low.shape != (4,) or high.shape != (4,):
            raise ValueError(f"ActionBounds 需要 4 维 low/high，收到 {low.shape}/{high.shape}")
        if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
            raise ValueError("ActionBounds 含非有限值")
        if np.any(low > high):
            raise ValueError(f"ActionBounds low > high: {low} / {high}")
        low.setflags(write=False)
        high.setflags(write=False)
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ActionBounds):
            return NotImplemented
        return (
            self.name == other.name
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
        )

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.low.tolist()), tuple(self.high.tolist())))

    @property
    def width(self) -> np.ndarray:
        return self.high - self.low

    def normalize(self, action: Any, *, clip: bool = False) -> np.ndarray:
        return normalize(action, self, clip=clip)

    def denormalize(self, normalized: Any, *, clip: bool = True) -> np.ndarray:
        return denormalize(normalized, self, clip=clip)

    def clip(self, action: Any) -> np.ndarray:
        return clip_components(action, self)

    def project(
        self,
        action: Any,
        v0: float,
        *,
        dt: float = 0.5,
        hz: int = 10,
        limits: "DynamicsLimits | None" = None,
        comfort_limits: "DynamicsLimits | None" = None,
    ) -> "ProjectionResult":
        """Static-box projection bound to this action box."""
        return project(
            action,
            v0,
            dt=dt,
            hz=hz,
            bounds=self,
            limits=HARD_LIMITS if limits is None else limits,
            comfort_limits=COMFORT_LIMITS if comfort_limits is None else comfort_limits,
        )

    def contains(self, action: Any, *, atol: float = _LIMIT_TOL) -> Any:
        arr = np.asarray(action, dtype=np.float64)
        result = (arr >= self.low - atol) & (arr <= self.high + atol)
        return bool(result.all()) if arr.ndim == 1 else result

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "low": self.low.tolist(), "high": self.high.tolist()}


CLEAN_BOUNDS = ActionBounds(
    low=np.array([0.0, -1.5, -0.8, 0.0]),
    high=np.array([7.5, 1.5, 0.8, 15.0]),
    name="clean",
)
DEFAULT_BOUNDS = CLEAN_BOUNDS


@dataclass(frozen=True)
class DynamicsLimits:
    """Acceleration/curvature/jerk limits (provisional, canonical calibration)."""

    a_long_min: float = -10.0
    a_long_max: float = 3.5
    a_lat_max: float = 8.0
    kappa_max: float = 0.345
    kappa_rate_max: float = 1.0
    jerk_long_max: float = 20.0
    jerk_lat_max: float = 20.0


HARD_LIMITS = DynamicsLimits()
COMFORT_LIMITS = DynamicsLimits(
    a_long_min=-4.0,
    a_long_max=2.0,
    a_lat_max=3.0,
    kappa_max=0.20,
    kappa_rate_max=0.05,
    jerk_long_max=5.0,
    jerk_lat_max=5.0,
)

# ======================================================================================
# Small helpers
# ======================================================================================
def _wrap(angle: Any) -> Any:
    """Wrap angle(s) to ``(-pi, pi]``."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def _as_actions(value: Any, name: str = "action") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 4:
        raise ValueError(f"{name} 需要形状 (...,4)，收到 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} 含非有限值（NaN/Inf）")
    return arr


def _as_single_action(value: Any, name: str = "action") -> np.ndarray:
    arr = _as_actions(value, name)
    if arr.ndim != 1:
        raise ValueError(f"{name} 需要单条形状 (4,)，收到 {arr.shape}")
    return arr


def _validate_time_grid(dt: float, hz: int) -> Tuple[float, int, int]:
    dt = float(dt)
    hz = int(hz)
    if dt <= 0.0 or hz <= 0:
        raise ValueError(f"dt/hz 必须为正，收到 dt={dt}, hz={hz}")
    substeps_f = dt * hz
    substeps = int(round(substeps_f))
    if substeps < 1 or abs(substeps_f - substeps) > 1e-9:
        raise ValueError(f"dt*hz 必须是正整数个子步，收到 dt={dt}, hz={hz}")
    return dt, hz, substeps


def _component_masks(action: np.ndarray, bounds: ActionBounds) -> np.ndarray:
    return (action < bounds.low - _LIMIT_TOL) | (action > bounds.high + _LIMIT_TOL)


def _bounds_codes(action: np.ndarray, bounds: ActionBounds) -> Tuple[list, list]:
    names = ("dx", "dy", "dpsi", "v_next")
    codes, reasons = [], []
    bad = _component_masks(action, bounds)
    for i, name in enumerate(names):
        if bad[i]:
            codes.append(f"OUT_OF_BOUNDS_{name.upper()}")
            reasons.append(
                f"{name}={action[i]:.6g} outside {bounds.name} bounds "
                f"[{bounds.low[i]:.6g}, {bounds.high[i]:.6g}]"
            )
    return codes, reasons


# ======================================================================================
# Normalization
# ======================================================================================
def normalize(action: Any, bounds: ActionBounds = DEFAULT_BOUNDS, *, clip: bool = False) -> np.ndarray:
    """Affine ``(action - low) / (high - low)``; optional static clipping first."""
    arr = _as_actions(action)
    if clip:
        arr = clip_components(arr, bounds)
    width = bounds.width
    if np.any(width <= 0.0):
        raise ValueError(f"{bounds.name} bounds 宽度必须为正，收到 {width}")
    return (arr - bounds.low) / width


def denormalize(
    normalized: Any,
    bounds: ActionBounds = DEFAULT_BOUNDS,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Affine ``low + normalized * (high - low)``.

    ``clip=True`` (default) clamps normalized values to
    ``[0, 1]``; it is a static box clamp, not a state-dependent rewrite.
    """
    arr = np.asarray(normalized, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[-1] != 4:
        raise ValueError(f"normalized action 需要形状 (...,4)，收到 {arr.shape}")
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    return bounds.low + arr * bounds.width


def clip_components(action: Any, bounds: ActionBounds = DEFAULT_BOUNDS) -> np.ndarray:
    """Clip each component to the static bounds (no state-dependent logic)."""
    arr = _as_actions(action)
    return np.minimum(np.maximum(arr, bounds.low), bounds.high)


# ======================================================================================
# PSTL geometry
# ======================================================================================
def _hermite_basis(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    return h00, h10, h01, h11


def _path_quantities(
    u: np.ndarray,
    p1: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    dpsi: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(P, P_u, P_uu, speed_u, heading, kappa)`` in ego frame.

    ``speed_u = ||dP/du||``, ``kappa`` is signed (left positive).  Degenerate
    tangents fall back to the straight interpolation ``dpsi * u``.
    """
    u = np.asarray(u, dtype=np.float64)
    _, h10, h01, h11 = _hermite_basis(u)
    p = h10[:, None] * m0 + h01[:, None] * p1 + h11[:, None] * m1

    d10 = 3.0 * u**2 - 4.0 * u + 1.0
    d01 = -6.0 * u**2 + 6.0 * u
    d11 = 3.0 * u**2 - 2.0 * u
    p_u = d10[:, None] * m0 + d01[:, None] * p1 + d11[:, None] * m1

    dd10 = 6.0 * u - 4.0
    dd01 = -12.0 * u + 6.0
    dd11 = 6.0 * u - 2.0
    p_uu = dd10[:, None] * m0 + dd01[:, None] * p1 + dd11[:, None] * m1

    speed_u = np.linalg.norm(p_u, axis=1)
    good = speed_u > _EPS
    safe = np.where(good, speed_u, 1.0)
    heading = np.where(good, np.arctan2(p_u[:, 1], p_u[:, 0]), dpsi * u)
    cross = p_u[:, 0] * p_uu[:, 1] - p_u[:, 1] * p_uu[:, 0]
    kappa = np.where(good, cross / safe**3, 0.0)
    return p, p_u, p_uu, speed_u, heading, kappa


def _arc_length(p1: np.ndarray, m0: np.ndarray, m1: np.ndarray) -> float:
    u = 0.5 * (_GL_NODES + 1.0)
    _, p_u, _, _, _, _ = _path_quantities(u, p1, m0, m1, 0.0)
    return float(0.5 * np.sum(_GL_WEIGHTS * np.linalg.norm(p_u, axis=1)))


def _path_endpoint_curvature(p1: np.ndarray, m0: np.ndarray, m1: np.ndarray) -> Tuple[float, float]:
    """Curvature and ``d kappa / ds`` at the path endpoint (u=1)."""
    u = np.array([1.0])
    _, p_u, p_uu, speed_u, _, kappa = _path_quantities(u, p1, m0, m1, 0.0)
    g = float(speed_u[0])
    if g <= _EPS:
        return 0.0, 0.0
    # P_uuu = 6*m0 - 12*p1 + 6*m1 (p0 = 0)
    p_uuu = 6.0 * m0 - 12.0 * p1 + 6.0 * m1
    cross = float(p_u[0, 0] * p_uu[0, 1] - p_u[0, 1] * p_uu[0, 0])
    cross3 = float(p_u[0, 0] * p_uuu[1] - p_u[0, 1] * p_uuu[0])
    dkappa_du = cross3 / g**3 - 3.0 * (cross / g**3) * float(np.dot(p_u[0], p_uu[0])) / g**2
    return float(kappa[0]), float(dkappa_du / g)


def _path_start_curvature(p1: np.ndarray, m0: np.ndarray, m1: np.ndarray) -> Tuple[float, float]:
    """Curvature and ``d kappa / ds`` at the path start (u=0)."""
    u = np.array([0.0])
    _, p_u, p_uu, speed_u, _, kappa = _path_quantities(u, p1, m0, m1, 0.0)
    g = float(speed_u[0])
    if g <= _EPS:
        return 0.0, 0.0
    p_uuu = 6.0 * m0 - 12.0 * p1 + 6.0 * m1
    cross = float(p_u[0, 0] * p_uu[0, 1] - p_u[0, 1] * p_uu[0, 0])
    cross3 = float(p_u[0, 0] * p_uuu[1] - p_u[0, 1] * p_uuu[0])
    dkappa_du = cross3 / g**3 - 3.0 * (cross / g**3) * float(np.dot(p_u[0], p_uu[0])) / g**2
    return float(kappa[0]), float(dkappa_du / g)


# ======================================================================================
# Time-law helpers
# ======================================================================================
def _time_law(
    tau: np.ndarray,
    v0: float,
    v1: float,
    length: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(s, s_dot, s_ddot, s_dddot)`` for the PSTL cubic time law."""
    tau = np.asarray(tau, dtype=np.float64)
    _, h10, h01, h11 = _hermite_basis(tau)
    d10 = 3.0 * tau**2 - 4.0 * tau + 1.0
    d01 = -6.0 * tau**2 + 6.0 * tau
    d11 = 3.0 * tau**2 - 2.0 * tau
    dd10 = 6.0 * tau - 4.0
    dd01 = -12.0 * tau + 6.0
    dd11 = 6.0 * tau - 2.0

    s = dt * v0 * h10 + length * h01 + dt * v1 * h11
    s_dot = dt * v0 * d10 + length * d01 + dt * v1 * d11
    s_ddot = dt * v0 * dd10 + length * dd01 + dt * v1 * dd11
    s_dddot = 6.0 * (dt * (v0 + v1) - 2.0 * length)
    return s, s_dot, s_ddot, float(s_dddot)


def _time_derivative_min(v0: float, v1: float, length: float, dt: float) -> float:
    """Exact minimum of ``s_dot(tau)`` on ``[0, 1]``.

    ``s_dot`` is quadratic, so its minimum is at an endpoint or at the single
    interior vertex.
    """
    a = 3.0 * dt * (v0 + v1) - 6.0 * length
    b = -4.0 * dt * v0 + 6.0 * length - 2.0 * dt * v1
    c = dt * v0
    values = [c, a + b + c]
    if abs(a) > _EPS:
        tau = -b / (2.0 * a)
        if 0.0 < tau < 1.0:
            values.append(a * tau * tau + b * tau + c)
    return float(min(values))


# ======================================================================================
# Result containers
# ======================================================================================
@dataclass
class ActionTrajectory:
    """PSTL trajectory and diagnostics for one 0.5 s segment.

    Arrays are ego-frame samples at ``t = 1/hz, ..., dt``.  ``positions`` is
    ``(N,3)`` ``(x, y, heading)``; ``speeds`` is ``(N,)`` m/s.  The final
    sample is forced to the requested endpoint position/heading/speed.

    ``codes`` contains problem codes (hard violations / degenerate action);
    ``flags`` contains informational tags.  ``feasible`` is true only when the
    action is valid, within bounds, and hard-feasible.
    """

    action: np.ndarray
    v0: float
    dt: float
    hz: int
    positions: np.ndarray
    speeds: np.ndarray
    a_long: np.ndarray
    a_lat: np.ndarray
    kappa: np.ndarray
    kappa_rate: np.ndarray
    jerk_long: np.ndarray
    jerk_lat: np.ndarray
    endpoint_errors: Dict[str, float]
    feasible: bool
    comfortable: bool
    codes: Tuple[str, ...]
    flags: Tuple[str, ...]
    reasons: Tuple[str, ...]
    metrics: Dict[str, Any]
    valid: bool = True
    comfort_codes: Tuple[str, ...] = ()
    comfort_reasons: Tuple[str, ...] = ()

    @property
    def n_points(self) -> int:
        return int(self.positions.shape[0])

    @property
    def points(self) -> np.ndarray:
        return self.positions

    @property
    def endpoint_error(self) -> Dict[str, float]:
        return self.endpoint_errors

    @property
    def start_speed(self) -> float:
        return float(self.metrics["start_speed_mps"])

    @property
    def end_speed(self) -> float:
        return float(self.metrics["end_speed_mps"])

    @property
    def jerk(self) -> np.ndarray:
        return self.jerk_long

    @property
    def jerk_long_scalar(self) -> float:
        return float(self.metrics["jerk_long_scalar"])

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in ("positions", "points", "xytheta", "trajectory"):
                return self.positions
            if key == "speeds":
                return self.speeds
            if hasattr(self, key):
                return getattr(self, key)
            return self.metrics[key]
        return self.positions[key]

    def keys(self) -> Tuple[str, ...]:
        return (
            "positions", "speeds", "a_long", "a_lat", "kappa", "kappa_rate",
            "jerk_long", "jerk_lat", "endpoint_errors", "feasible", "codes",
            "flags", "metrics",
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.copy(),
            "v0": self.v0,
            "dt": self.dt,
            "hz": self.hz,
            "positions": self.positions.copy(),
            "points": self.positions.copy(),
            "speeds": self.speeds.copy(),
            "a_long": self.a_long.copy(),
            "a_lat": self.a_lat.copy(),
            "kappa": self.kappa.copy(),
            "kappa_rate": self.kappa_rate.copy(),
            "jerk_long": self.jerk_long.copy(),
            "jerk_lat": self.jerk_lat.copy(),
            "endpoint_errors": dict(self.endpoint_errors),
            "feasible": self.feasible,
            "comfortable": self.comfortable,
            "valid": self.valid,
            "codes": tuple(self.codes),
            "flags": tuple(self.flags),
            "reasons": tuple(self.reasons),
            "comfort_codes": tuple(self.comfort_codes),
            "comfort_reasons": tuple(self.comfort_reasons),
            "metrics": dict(self.metrics),
        }


@dataclass
class V1Interval:
    """Hard-feasible ``v_next`` interval for a fixed ``(dx, dy, dpsi, v0)``.

    Derived from: action bounds, longitudinal jerk, start/end acceleration and
    time-law monotonicity.  Lateral/curvature feasibility is checked by
    :func:`action_to_trajectory` / :func:`gate`; it is not folded into this
    interval because it is not affine in ``v1``.
    """

    low: float
    high: float
    nonempty: bool
    constraints: Dict[str, Tuple[float, float]]
    codes: Tuple[str, ...]
    reasons: Tuple[str, ...]

    @property
    def width(self) -> float:
        return max(0.0, float(self.high - self.low))

    @property
    def lo(self) -> float:
        return self.low

    @property
    def hi(self) -> float:
        return self.high

    @property
    def is_empty(self) -> bool:
        return not self.nonempty

    def contains(self, v1: float, *, atol: float = 1e-9) -> bool:
        return bool(self.nonempty and self.low - atol <= float(v1) <= self.high + atol)

    def __iter__(self):
        return iter((self.low, self.high))

    def __getitem__(self, index: int) -> float:
        return (self.low, self.high)[index]


#: Short alias used by callers that prefer ``Trajectory``.
Trajectory = ActionTrajectory


@dataclass
class ProjectionResult:
    """Result of static component clipping + PSTL feasibility diagnosis."""

    requested_action: np.ndarray
    executed_action: np.ndarray
    component_clipped: np.ndarray
    clipped_components: Tuple[str, ...]
    trajectory: ActionTrajectory
    feasible: bool
    comfortable: bool
    v1_interval: V1Interval
    v1_in_interval: bool
    codes: Tuple[str, ...]
    reasons: Tuple[str, ...]
    metrics: Dict[str, Any]

    @property
    def action(self) -> np.ndarray:
        return self.executed_action

    @property
    def executed(self) -> np.ndarray:
        return self.executed_action

    @property
    def requested(self) -> np.ndarray:
        return self.requested_action

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested_action": self.requested_action.copy(),
            "executed_action": self.executed_action.copy(),
            "component_clipped": self.component_clipped.copy(),
            "clipped_components": tuple(self.clipped_components),
            "trajectory": self.trajectory,
            "feasible": self.feasible,
            "comfortable": self.comfortable,
            "v1_interval": self.v1_interval,
            "v1_in_interval": self.v1_in_interval,
            "codes": tuple(self.codes),
            "reasons": tuple(self.reasons),
            "metrics": dict(self.metrics),
        }


@dataclass
class GateResult:
    """Accept/reject gate for one action (no rewriting)."""

    accepted: bool
    codes: Tuple[str, ...]
    reasons: Tuple[str, ...]
    trajectory: ActionTrajectory
    v1_interval: V1Interval
    metrics: Dict[str, Any]

    @property
    def action(self) -> np.ndarray:
        return self.trajectory.action


# ======================================================================================
# Main PSTL builder
# ======================================================================================
def action_to_trajectory(
    action: Any,
    v0: float,
    dt: float = 0.5,
    hz: int = 10,
    bounds: ActionBounds = DEFAULT_BOUNDS,
    limits: DynamicsLimits = HARD_LIMITS,
    comfort_limits: DynamicsLimits = COMFORT_LIMITS,
) -> ActionTrajectory:
    """Execute a 4D action as a PSTL trajectory.

    Args:
        action: ``(dx, dy, dpsi, v_next)`` in metres / radians / m/s.
        v0: current speed (m/s, non-negative); used by the time law only.
        dt: policy segment duration (s); default 0.5.
        hz: sample rate (Hz); default 10 -> 5 samples per segment.
        bounds: static action box used only for out-of-bounds reporting.
        limits: hard feasibility limits.
        comfort_limits: comfort limits.

    Returns:
        :class:`ActionTrajectory`.  The action is **not** clipped here; use
        :func:`project` for the static-box projection path.
    """
    a = _as_single_action(action)
    v0 = float(v0)
    if not np.isfinite(v0) or v0 < 0.0:
        raise ValueError(f"v0 必须是非负有限速度，收到 {v0}")
    v1 = float(a[3])
    if not np.isfinite(v1) or v1 < 0.0:
        raise ValueError(f"v_next 必须是非负有限速度，收到 {v1}")
    dt, hz, n = _validate_time_grid(dt, hz)
    tau = np.arange(1, n + 1, dtype=np.float64) / float(n)
    dpsi = float(_wrap(a[2]))
    p1 = np.array([float(a[0]), float(a[1])], dtype=np.float64)
    chord = float(np.linalg.norm(p1))

    codes: list = []
    flags: list = []
    reasons: list = []

    # ---- geometry ------------------------------------------------------------
    zero_length = chord <= _EPS
    if zero_length:
        length = 0.0
        m0 = np.zeros(2, dtype=np.float64)
        m1 = np.zeros(2, dtype=np.float64)
        positions = np.zeros((n, 3), dtype=np.float64)
        positions[:, 2] = dpsi * tau
        headings = positions[:, 2].copy()
        speed_u = np.zeros(n, dtype=np.float64)
        kappa = np.zeros(n, dtype=np.float64)
        kappa_rate = np.zeros(n, dtype=np.float64)
        if v0 > _EPS or v1 > _EPS:
            codes.append("ZERO_LENGTH_WITH_SPEED")
            reasons.append(
                f"zero-length path (dx=dy=0) with v0={v0:.6g}, v_next={v1:.6g}"
            )
        elif abs(dpsi) > _EPS_ROTATION:
            codes.append("ROTATION_IN_PLACE")
            reasons.append(
                f"zero-length stop requests rotation in place (|dpsi|={abs(dpsi):.6g} rad)"
            )
        else:
            flags.append("FULL_STOP_HOLD")
        kappa_start = kappa_end = 0.0
        dkappa_ds_start = dkappa_ds_end = 0.0
    else:
        m = chord  # rho = 1.0
        m0 = np.array([m, 0.0], dtype=np.float64)
        m1 = m * np.array([math.cos(dpsi), math.sin(dpsi)], dtype=np.float64)
        length = _arc_length(p1, m0, m1)
        u = np.zeros(n, dtype=np.float64)
        positions = np.zeros((n, 3), dtype=np.float64)
        headings = np.zeros(n, dtype=np.float64)
        speed_u = np.zeros(n, dtype=np.float64)
        kappa = np.zeros(n, dtype=np.float64)
        kappa_rate = np.zeros(n, dtype=np.float64)
        kappa_start, dkappa_ds_start = _path_start_curvature(p1, m0, m1)
        kappa_end, dkappa_ds_end = _path_endpoint_curvature(p1, m0, m1)

    # ---- time law ------------------------------------------------------------
    s, s_dot, s_ddot, s_dddot = _time_law(tau, v0, v1, length, dt)
    speeds = s_dot / dt
    a_long = s_ddot / (dt * dt)
    jerk_long_scalar = s_dddot / (dt * dt * dt)

    if not zero_length:
        u = np.clip(s / length, 0.0, 1.0) if length > _EPS else np.ones(n)
        (
            positions[:, :2],
            p_u,
            p_uu,
            speed_u,
            headings,
            kappa,
        ) = _path_quantities(u, p1, m0, m1, dpsi)
        positions[:, 2] = headings
        # d kappa / ds = (d kappa / du) / ||dP/du||; exact third-derivative formula
        p_uuu = 6.0 * m0 - 12.0 * p1 + 6.0 * m1
        cross = p_u[:, 0] * p_uu[:, 1] - p_u[:, 1] * p_uu[:, 0]
        cross3 = p_u[:, 0] * p_uuu[1] - p_u[:, 1] * p_uuu[0]
        safe = np.where(speed_u > _EPS, speed_u, 1.0)
        dkappa_du = cross3 / safe**3 - 3.0 * (cross / safe**3) * (
            (p_u[:, 0] * p_uu[:, 0] + p_u[:, 1] * p_uu[:, 1]) / safe**2
        )
        dkappa_ds = np.where(speed_u > _EPS, dkappa_du / safe, 0.0)
        kappa_rate = dkappa_ds * speeds
    else:
        dkappa_ds = np.zeros(n, dtype=np.float64)
        # zero-length heading ramp is already set; kappa/rates stay zero

    # ---- derived dynamics ----------------------------------------------------
    a_lat = kappa * speeds**2
    jerk_long = np.full(n, jerk_long_scalar, dtype=np.float64)
    jerk_lat = kappa_rate * speeds**2 + 2.0 * kappa * speeds * a_long

    # endpoint start/end diagnostics (start is not part of the sampled arrays)
    a_start = (-4.0 * v0 + 6.0 * length / dt - 2.0 * v1) / dt
    a_end = (2.0 * v0 - 6.0 * length / dt + 4.0 * v1) / dt
    v_consistent = 2.0 * length / dt - v0  # speed that makes jerk zero
    r_v = v1 - v_consistent
    j_req = 6.0 * r_v / (dt * dt)
    a_lat_start = kappa_start * v0**2
    a_lat_end = kappa_end * v1**2
    kappa_rate_start = dkappa_ds_start * v0
    kappa_rate_end = dkappa_ds_end * v1
    jerk_lat_start = kappa_rate_start * v0**2 + 2.0 * kappa_start * v0 * a_start
    jerk_lat_end = kappa_rate_end * v1**2 + 2.0 * kappa_end * v1 * a_end

    # ---- force exact endpoint ------------------------------------------------
    raw_pos_err = float(np.linalg.norm(positions[-1, :2] - p1))
    raw_head_err = float(abs(float(_wrap(headings[-1] - dpsi))))
    raw_speed_err = float(abs(speeds[-1] - v1))
    positions[-1, 0] = p1[0]
    positions[-1, 1] = p1[1]
    positions[-1, 2] = dpsi
    headings[-1] = dpsi
    speeds[-1] = v1

    pos_error = float(np.linalg.norm(positions[-1, :2] - p1))
    head_error = float(abs(float(_wrap(headings[-1] - dpsi))))
    speed_error = float(abs(speeds[-1] - v1))
    endpoint_errors = {
        "position_m": pos_error,
        "heading_rad": head_error,
        "speed_mps": speed_error,
        # compact aliases for callers that use the shorter names
        "position": pos_error,
        "heading": head_error,
        "speed": speed_error,
        "raw_position_m": raw_pos_err,
        "raw_heading_rad": raw_head_err,
        "raw_speed_mps": raw_speed_err,
    }

    # ---- applicability / classification --------------------------------------
    curvature_checked = length >= _MIN_CURVATURE_PATH
    stop_class = bool(length < 0.10 and v1 < 0.10 and abs(dpsi) <= 0.05)
    full_stop = bool(chord <= _EPS and v0 <= _EPS and v1 <= _EPS)
    stop_to_go = bool(v0 <= _EPS and v1 > _EPS and length > _EPS)
    dx_negative = bool(a[0] < 0.0)
    min_time_derivative = _time_derivative_min(v0, v1, length, dt)
    speed_nonnegative = bool(min_time_derivative >= -_LIMIT_TOL)

    if stop_class:
        flags.append("STOP_CLASS")
    if full_stop:
        flags.append("FULL_STOP")
    if stop_to_go:
        flags.append("STOP_TO_GO")
    if dx_negative:
        flags.append("DX_NEGATIVE")
    if not curvature_checked:
        flags.append("CURVATURE_SKIPPED")
    if not speed_nonnegative:
        codes.append("TIME_LAW_NON_MONOTONE")
        reasons.append(
            f"time law speed becomes negative (min s_dot={min_time_derivative:.6g} m)"
        )

    bounds_bad = _component_masks(a, bounds)
    b_codes, b_reasons = _bounds_codes(a, bounds)
    codes.extend(b_codes)
    reasons.extend(b_reasons)
    action_in_bounds = not bool(np.any(bounds_bad))

    # ---- feasibility check arrays (include path endpoints) -------------------
    a_long_check = np.concatenate([a_long, np.array([a_start, a_end])])
    a_lat_check = np.concatenate([a_lat, np.array([a_lat_start, a_lat_end])])
    kappa_check = np.concatenate([kappa, np.array([kappa_start, kappa_end])])
    kappa_rate_check = np.concatenate(
        [kappa_rate, np.array([kappa_rate_start, kappa_rate_end])]
    )
    jerk_lat_check = np.concatenate([jerk_lat, np.array([jerk_lat_start, jerk_lat_end])])
    jerk_long_check = np.array([jerk_long_scalar], dtype=np.float64)

    hard_codes, hard_reasons = _check_limits(
        a_long_check,
        a_lat_check,
        kappa_check,
        kappa_rate_check,
        jerk_long_check,
        jerk_lat_check,
        limits,
        prefix="HARD",
        curvature_checked=curvature_checked,
    )
    comfort_codes, comfort_reasons = _check_limits(
        a_long_check,
        a_lat_check,
        kappa_check,
        kappa_rate_check,
        jerk_long_check,
        jerk_lat_check,
        comfort_limits,
        prefix="COMFORT",
        curvature_checked=curvature_checked,
    )
    codes.extend(hard_codes)
    reasons.extend(hard_reasons)

    # Invalid means the geometry itself is unusable (zero-length/rotation).
    invalid_codes = {"ZERO_LENGTH_WITH_SPEED", "ROTATION_IN_PLACE"}
    valid = not any(c in invalid_codes for c in codes)
    feasible = bool(valid and not hard_codes and not b_codes and speed_nonnegative)
    comfortable = bool(valid and not comfort_codes)

    metrics: Dict[str, Any] = {
        "path_length_m": float(length),
        "chord_length_m": float(chord),
        "duration_s": float(dt),
        "substeps": int(n),
        "v0_mps": float(v0),
        "v1_mps": float(v1),
        "start_speed_mps": float(v0),
        "end_speed_mps": float(v1),
        "start_accel_mps2": float(a_start),
        "end_accel_mps2": float(a_end),
        "min_speed_mps": float(np.min(speeds)),
        "max_speed_mps": float(np.max(speeds)),
        "min_time_derivative": float(min_time_derivative),
        "speed_nonnegative": speed_nonnegative,
        "stop_class": stop_class,
        "full_stop": full_stop,
        "stop_to_go": stop_to_go,
        "curvature_checked": bool(curvature_checked),
        "dx_negative": dx_negative,
        "action_in_bounds": bool(action_in_bounds),
        "out_of_bounds_components": [
            name for name, bad in zip(("dx", "dy", "dpsi", "v_next"), bounds_bad) if bad
        ],
        "max_abs_a_long": float(np.max(np.abs(a_long_check))),
        "min_a_long": float(np.min(a_long_check)),
        "max_a_long": float(np.max(a_long_check)),
        "max_abs_a_lat": float(np.max(np.abs(a_lat_check))),
        "max_abs_kappa": float(np.max(np.abs(kappa_check))),
        "max_abs_kappa_rate": float(np.max(np.abs(kappa_rate_check))),
        "max_abs_jerk_long": float(np.max(np.abs(jerk_long_check))),
        "max_abs_jerk_lat": float(np.max(np.abs(jerk_lat_check))),
        "kappa_start": float(kappa_start),
        "kappa_end": float(kappa_end),
        "a_lat_start": float(a_lat_start),
        "a_lat_end": float(a_lat_end),
        "jerk_long_scalar": float(jerk_long_scalar),
        "jerk_scalar": float(jerk_long_scalar),
        "jerk": float(jerk_long_scalar),
        "v_consistent_mps": float(v_consistent),
        "r_v_mps": float(r_v),
        "j_req_mps3": float(j_req),
    }

    return ActionTrajectory(
        action=a.copy(),
        v0=v0,
        dt=dt,
        hz=hz,
        positions=positions,
        speeds=speeds,
        a_long=a_long,
        a_lat=a_lat,
        kappa=kappa,
        kappa_rate=kappa_rate,
        jerk_long=jerk_long,
        jerk_lat=jerk_lat,
        endpoint_errors=endpoint_errors,
        feasible=feasible,
        comfortable=comfortable,
        codes=tuple(codes),
        flags=tuple(flags),
        reasons=tuple(reasons),
        metrics=metrics,
        valid=bool(valid),
        comfort_codes=tuple(comfort_codes),
        comfort_reasons=tuple(comfort_reasons),
    )


def _check_limits(
    a_long: np.ndarray,
    a_lat: np.ndarray,
    kappa: np.ndarray,
    kappa_rate: np.ndarray,
    jerk_long: np.ndarray,
    jerk_lat: np.ndarray,
    limits: DynamicsLimits,
    *,
    prefix: str,
    curvature_checked: bool,
) -> Tuple[list, list]:
    codes: list = []
    reasons: list = []

    lo = float(np.min(a_long))
    hi = float(np.max(a_long))
    if lo < limits.a_long_min - _LIMIT_TOL:
        codes.append(f"{prefix}_A_LONG_MIN")
        reasons.append(f"a_long min {lo:.6g} < {prefix.lower()} min {limits.a_long_min:.6g}")
    if hi > limits.a_long_max + _LIMIT_TOL:
        codes.append(f"{prefix}_A_LONG_MAX")
        reasons.append(f"a_long max {hi:.6g} > {prefix.lower()} max {limits.a_long_max:.6g}")

    max_jl = float(np.max(np.abs(jerk_long)))
    if max_jl > limits.jerk_long_max + _LIMIT_TOL:
        codes.append(f"{prefix}_JERK_LONG")
        reasons.append(
            f"|jerk_long| max {max_jl:.6g} > {prefix.lower()} max {limits.jerk_long_max:.6g}"
        )

    if curvature_checked:
        max_alat = float(np.max(np.abs(a_lat)))
        if max_alat > limits.a_lat_max + _LIMIT_TOL:
            codes.append(f"{prefix}_A_LAT")
            reasons.append(
                f"|a_lat| max {max_alat:.6g} > {prefix.lower()} max {limits.a_lat_max:.6g}"
            )
        max_k = float(np.max(np.abs(kappa)))
        if max_k > limits.kappa_max + _LIMIT_TOL:
            codes.append(f"{prefix}_KAPPA")
            reasons.append(
                f"|kappa| max {max_k:.6g} > {prefix.lower()} max {limits.kappa_max:.6g}"
            )
        max_kr = float(np.max(np.abs(kappa_rate)))
        if max_kr > limits.kappa_rate_max + _LIMIT_TOL:
            codes.append(f"{prefix}_KAPPA_RATE")
            reasons.append(
                f"|kappa_rate| max {max_kr:.6g} > {prefix.lower()} max "
                f"{limits.kappa_rate_max:.6g}"
            )
        max_jlat = float(np.max(np.abs(jerk_lat)))
        if max_jlat > limits.jerk_lat_max + _LIMIT_TOL:
            codes.append(f"{prefix}_JERK_LAT")
            reasons.append(
                f"|jerk_lat| max {max_jlat:.6g} > {prefix.lower()} max "
                f"{limits.jerk_lat_max:.6g}"
            )
    return codes, reasons


# ======================================================================================
# Feasible v_next interval
# ======================================================================================
def _contiguous_runs(mask: np.ndarray) -> list:
    runs = []
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _shrink_to_monotone(
    low: float,
    high: float,
    v0: float,
    length: float,
    dt: float,
    preferred: float,
    samples: int = 2049,
) -> Tuple[float, float, bool]:
    """Shrink ``[low, high]`` to the contiguous region with non-negative speed.

    Returns ``(new_low, new_high, changed)``.  If no sample is feasible the
    interval is returned with ``low > high`` (empty).
    """
    if high < low:
        return low, high, False
    grid = np.linspace(low, high, int(samples))
    ok = np.array(
        [_time_derivative_min(v0, float(v), length, dt) >= -_LIMIT_TOL for v in grid],
        dtype=bool,
    )
    if ok.all():
        return low, high, False
    if not ok.any():
        return high, low, True
    runs = _contiguous_runs(ok)
    chosen = None
    for i0, i1 in runs:
        if grid[i0] - _LIMIT_TOL <= preferred <= grid[i1] + _LIMIT_TOL:
            chosen = (i0, i1)
            break
    if chosen is None:
        chosen = max(runs, key=lambda r: grid[r[1]] - grid[r[0]])
    i0, i1 = chosen

    def feasible(v: float) -> bool:
        return _time_derivative_min(v0, float(v), length, dt) >= -_LIMIT_TOL

    # refine the left boundary between a known-bad sample and a known-good one
    new_low = float(grid[i0])
    if i0 > 0:
        bad, good = float(grid[i0 - 1]), float(grid[i0])
        for _ in range(60):
            mid = 0.5 * (bad + good)
            if feasible(mid):
                good = mid
            else:
                bad = mid
        new_low = good
    new_high = float(grid[i1])
    if i1 < len(grid) - 1:
        good, bad = float(grid[i1]), float(grid[i1 + 1])
        for _ in range(60):
            mid = 0.5 * (good + bad)
            if feasible(mid):
                good = mid
            else:
                bad = mid
        new_high = good
    return new_low, new_high, True


def feasible_v1_interval(
    action: Any,
    v0: float,
    dt: float = 0.5,
    bounds: ActionBounds = DEFAULT_BOUNDS,
    limits: DynamicsLimits = HARD_LIMITS,
) -> V1Interval:
    """Hard-feasible interval for ``v_next`` given ``(dx, dy, dpsi, v0)``.

    The action's own ``v_next`` is ignored except as the preferred branch when
    monotonicity splits the interval.  This function does not rewrite the
    action; :func:`gate` uses it to diagnose, and a caller may choose a new
    ``v_next`` explicitly from the returned interval.

    Constraints folded in: static bounds, longitudinal jerk, start/end
    acceleration, and non-negative time-law speed.  Lateral/curvature limits
    remain a post-hoc check in :func:`action_to_trajectory`.
    """
    a = _as_single_action(action)
    v0 = float(v0)
    if not np.isfinite(v0) or v0 < 0.0:
        raise ValueError(f"v0 必须是非负有限速度，收到 {v0}")
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt 必须为正，收到 {dt}")
    dpsi = float(_wrap(a[2]))
    p1 = np.array([float(a[0]), float(a[1])], dtype=np.float64)
    chord = float(np.linalg.norm(p1))
    preferred = float(a[3])

    constraints: Dict[str, Tuple[float, float]] = {}
    codes: list = []
    reasons: list = []

    if chord <= _EPS:
        length = 0.0
        if v0 > _EPS or preferred > _EPS:
            codes.append("V1_INTERVAL_EMPTY")
            reasons.append("zero-length path with speed has no feasible v_next")
            return V1Interval(float("nan"), float("nan"), False, constraints, tuple(codes), tuple(reasons))
        if abs(dpsi) > _EPS_ROTATION:
            codes.append("V1_INTERVAL_EMPTY")
            reasons.append("rotation in place has no feasible v_next")
            return V1Interval(float("nan"), float("nan"), False, constraints, tuple(codes), tuple(reasons))
    else:
        m = chord
        m0 = np.array([m, 0.0], dtype=np.float64)
        m1 = m * np.array([math.cos(dpsi), math.sin(dpsi)], dtype=np.float64)
        length = _arc_length(p1, m0, m1)

    low = float(bounds.low[3])
    high = float(bounds.high[3])
    constraints["bounds"] = (low, high)

    if length <= _EPS:
        # full stop: only a zero target speed is physically meaningful
        low = max(low, 0.0)
        high = min(high, 0.0)
        constraints["stop"] = (low, high)
    else:
        # longitudinal jerk: |6*(T*(v0+v1) - 2L)/T^3| <= J
        alpha = 6.0 / dt**2
        beta = 12.0 * length / dt**3
        j_lo = (-limits.jerk_long_max + beta) / alpha - v0
        j_hi = (limits.jerk_long_max + beta) / alpha - v0
        low, high = max(low, j_lo), min(high, j_hi)
        constraints["jerk_long"] = (j_lo, j_hi)

        # start acceleration: a_start = (-4v0 + 6L/T - 2v1)/T
        s_lo = (6.0 * length / dt - 4.0 * v0 - limits.a_long_max * dt) / 2.0
        s_hi = (6.0 * length / dt - 4.0 * v0 - limits.a_long_min * dt) / 2.0
        low, high = max(low, s_lo), min(high, s_hi)
        constraints["accel_start"] = (s_lo, s_hi)

        # end acceleration: a_end = (2v0 - 6L/T + 4v1)/T
        e_lo = (limits.a_long_min * dt - 2.0 * v0 + 6.0 * length / dt) / 4.0
        e_hi = (limits.a_long_max * dt - 2.0 * v0 + 6.0 * length / dt) / 4.0
        low, high = max(low, e_lo), min(high, e_hi)
        constraints["accel_end"] = (e_lo, e_hi)

        if low <= high:
            low, high, changed = _shrink_to_monotone(low, high, v0, length, dt, preferred)
            constraints["monotonicity"] = (low, high)
            if changed:
                codes.append("V1_MONOTONICITY_LIMIT")
                reasons.append(
                    f"v_next interval limited by non-negative speed: [{low:.6g}, {high:.6g}]"
                )

    nonempty = bool(np.isfinite(low) and np.isfinite(high) and low <= high + 1e-9)
    if not nonempty:
        codes.append("V1_INTERVAL_EMPTY")
        reasons.append(
            f"no feasible v_next under hard limits (raw interval [{low:.6g}, {high:.6g}])"
        )
        low, high = float("nan"), float("nan")
    return V1Interval(float(low), float(high), nonempty, constraints, tuple(codes), tuple(reasons))


# ======================================================================================
# Gate and projection
# ======================================================================================
def gate(
    action: Any,
    v0: float,
    dt: float = 0.5,
    hz: int = 10,
    bounds: ActionBounds = DEFAULT_BOUNDS,
    limits: DynamicsLimits = HARD_LIMITS,
    comfort_limits: DynamicsLimits = COMFORT_LIMITS,
    *,
    require_comfort: bool = False,
) -> GateResult:
    """Accept/reject an action without rewriting it.

    Rejects on out-of-bounds components, degenerate geometry, hard dynamics
    violations, or ``v_next`` outside :func:`feasible_v1_interval`.  Codes are
    stable strings; reasons are human-readable.
    """
    a = _as_single_action(action)
    traj = action_to_trajectory(a, v0, dt=dt, hz=hz, bounds=bounds, limits=limits, comfort_limits=comfort_limits)
    interval = feasible_v1_interval(a, v0, dt=dt, bounds=bounds, limits=limits)
    codes = list(traj.codes)
    reasons = list(traj.reasons)

    v1 = float(a[3])
    if not interval.nonempty:
        codes.append("V1_INTERVAL_EMPTY")
        reasons.append("no feasible v_next interval for this geometry/state")
    elif v1 < interval.low - 1e-9:
        codes.append("V1_BELOW_INTERVAL")
        reasons.append(f"v_next={v1:.6g} < interval low {interval.low:.6g}")
    elif v1 > interval.high + 1e-9:
        codes.append("V1_ABOVE_INTERVAL")
        reasons.append(f"v_next={v1:.6g} > interval high {interval.high:.6g}")

    if require_comfort and not traj.comfortable:
        codes.extend(traj.comfort_codes)
        reasons.extend(traj.comfort_reasons)

    accepted = not codes
    metrics = dict(traj.metrics)
    metrics["v1_interval_low"] = interval.low
    metrics["v1_interval_high"] = interval.high
    metrics["v1_interval_nonempty"] = interval.nonempty
    metrics["v1_interval_width"] = interval.width
    return GateResult(
        accepted=accepted,
        codes=tuple(codes),
        reasons=tuple(reasons),
        trajectory=traj,
        v1_interval=interval,
        metrics=metrics,
    )


def project(
    action: Any,
    v0: float,
    dt: float = 0.5,
    hz: int = 10,
    bounds: ActionBounds = DEFAULT_BOUNDS,
    limits: DynamicsLimits = HARD_LIMITS,
    comfort_limits: DynamicsLimits = COMFORT_LIMITS,
) -> ProjectionResult:
    """Static component clipping + PSTL feasibility diagnosis.

    Only the static :class:`ActionBounds` box is applied.  In particular,
    ``v_next`` is **not** rewritten from ``v0``, path length, acceleration or
    curvature; an infeasible request is reported, not silently repaired.
    ``component_clipped`` records every per-component clip, including a tiny
    negative ``dx`` drift clipped to zero.
    """
    requested = _as_single_action(action)
    executed = clip_components(requested, bounds)
    clipped_mask = np.abs(executed - requested) > 0.0
    names = ("dx", "dy", "dpsi", "v_next")
    clipped_components = tuple(name for name, bad in zip(names, clipped_mask) if bad)

    reasons: list = []
    clip_codes: list = []
    for i, name in enumerate(names):
        if clipped_mask[i]:
            clip_codes.append(f"CLIPPED_{name.upper()}")
            reasons.append(
                f"{name} clipped from {requested[i]:.6g} to {executed[i]:.6g} "
                f"({bounds.name} bounds [{bounds.low[i]:.6g}, {bounds.high[i]:.6g}])"
            )
    if clipped_mask[0] and -0.10 <= requested[0] < 0.0:
        reasons.append("negative dx drift clipped to zero")

    traj = action_to_trajectory(
        executed, v0, dt=dt, hz=hz, bounds=bounds, limits=limits, comfort_limits=comfort_limits
    )
    interval = feasible_v1_interval(executed, v0, dt=dt, bounds=bounds, limits=limits)
    v1 = float(executed[3])
    v1_in_interval = bool(interval.contains(v1))

    codes = list(clip_codes) + list(traj.codes)
    reasons.extend(traj.reasons)
    if not interval.nonempty:
        codes.append("V1_INTERVAL_EMPTY")
        reasons.append("no feasible v_next interval for this geometry/state")
    elif not v1_in_interval:
        if v1 < interval.low:
            codes.append("V1_BELOW_INTERVAL")
            reasons.append(f"v_next={v1:.6g} < feasible interval low {interval.low:.6g}")
        else:
            codes.append("V1_ABOVE_INTERVAL")
            reasons.append(f"v_next={v1:.6g} > feasible interval high {interval.high:.6g}")

    feasible = bool(traj.feasible and interval.nonempty and v1_in_interval)
    metrics = dict(traj.metrics)
    metrics.update(
        {
            "component_clipped": clipped_mask.copy(),
            "clipped_components": list(clipped_components),
            "v1_interval_low": interval.low,
            "v1_interval_high": interval.high,
            "v1_interval_nonempty": interval.nonempty,
            "v1_interval_width": interval.width,
            "v1_in_interval": v1_in_interval,
        }
    )
    return ProjectionResult(
        requested_action=requested.copy(),
        executed_action=executed.copy(),
        component_clipped=clipped_mask,
        clipped_components=clipped_components,
        trajectory=traj,
        feasible=feasible,
        comfortable=bool(traj.comfortable),
        v1_interval=interval,
        v1_in_interval=v1_in_interval,
        codes=tuple(codes),
        reasons=tuple(reasons),
        metrics=metrics,
    )
