"""Bounded action distribution used by BC and PPO.

The policy emits a mean and log-std for a *normalized* 4D action in ``[0,1]``.
A plain Normal plus a post-hoc clamp violates the PPO probability-action
contract (the stored log-prob belongs to the unclamped sample while the
environment executes the clamped action).  A diagonal truncated Normal on
``[0,1]`` keeps every sample executable and gives an exact log-density.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor


def _std_normal_cdf(x: Tensor) -> Tensor:
    return torch.special.ndtr(x)


def _std_normal_icdf(p: Tensor) -> Tensor:
    # torch.special.ndtri is available in the project's torch 2.3 build.
    return torch.special.ndtri(p.clamp(1e-7, 1.0 - 1e-7))


class TruncatedNormal:
    """Diagonal Normal truncated to ``[low, high]`` (default ``[0,1]``)."""

    def __init__(
        self,
        loc: Tensor,
        scale: Tensor,
        *,
        low: float = 0.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        if loc.shape != scale.shape:
            raise ValueError(f"loc/scale shape mismatch: {loc.shape} vs {scale.shape}")
        self.loc = loc
        self.scale = scale.clamp_min(float(eps))
        self.low = float(low)
        self.high = float(high)
        if not self.high > self.low:
            raise ValueError("high must be > low")
        z0 = (self.low - self.loc) / self.scale
        z1 = (self.high - self.loc) / self.scale
        self._cdf0 = _std_normal_cdf(z0)
        self._cdf1 = _std_normal_cdf(z1)
        self._mass = (self._cdf1 - self._cdf0).clamp_min(float(eps))

    def cdf(self, x: Tensor) -> Tensor:
        return ((_std_normal_cdf((x - self.loc) / self.scale) - self._cdf0) / self._mass).clamp(0.0, 1.0)

    def sample(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        if sample_shape is None:
            shape = self.loc.shape
        else:
            shape = torch.Size(tuple(sample_shape) + tuple(self.loc.shape))
        u = torch.rand(shape, device=self.loc.device, dtype=self.loc.dtype)
        z = self.loc + self.scale * _std_normal_icdf(self._cdf0 + u * self._mass)
        return z.clamp(self.low, self.high)

    def log_prob(self, value: Tensor) -> Tensor:
        inside = (value >= self.low) & (value <= self.high)
        z = (value - self.loc) / self.scale
        normal = -0.5 * z * z - torch.log(self.scale) - 0.5 * math.log(2.0 * math.pi)
        log_mass = torch.log(self._mass)
        out = normal - log_mass
        # Outside support is -inf, but never silently map a different value to it.
        return torch.where(inside, out, torch.full_like(out, -float("inf")))

    def entropy(self) -> Tensor:
        """Approximate entropy via a fixed 33-point quadrature on [low,high]."""
        n = 33
        grid = torch.linspace(self.low, self.high, n, device=self.loc.device, dtype=self.loc.dtype)
        # expand grid to loc.shape: [*shape, n]
        view = (1,) * self.loc.ndim + (n,)
        g = grid.view(*view).expand(*self.loc.shape, n)
        p = self.log_prob(g).exp()
        # trapezoid; broad enough for monitoring, not used in gradients
        dx = (self.high - self.low) / (n - 1)
        return -(p[..., 1:-1].sum(-1) + 0.5 * p[..., 0] + 0.5 * p[..., -1]) * dx

    def mode(self) -> Tensor:
        return self.loc.clamp(self.low, self.high)

    def mean(self) -> Tensor:
        """Numerical mean (33-point quadrature); monitoring only."""
        n = 33
        grid = torch.linspace(self.low, self.high, n, device=self.loc.device, dtype=self.loc.dtype)
        view = (1,) * self.loc.ndim + (n,)
        g = grid.view(*view).expand(*self.loc.shape, n)
        p = self.log_prob(g).exp()
        dx = (self.high - self.low) / (n - 1)
        return (g * p).sum(-1) * dx


def truncated_normal_nll(
    target: Tensor,
    loc: Tensor,
    log_scale: Tensor,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Return ``(nll_per_row, outside_mask)`` for a normalized target."""
    outside = (target < low) | (target > high)
    target_c = target.clamp(low, high)
    dist = TruncatedNormal(loc, log_scale.exp(), low=low, high=high)
    return -dist.log_prob(target_c).sum(dim=-1), outside
