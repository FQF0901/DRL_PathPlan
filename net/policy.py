"""策略头与价值头：高斯策略输出有界动作 ``(ds, dθ)``，并给出 PPO 所需的对数概率。

动作约定（p2-contract §0）
-------------------------
``(ds, dθ)`` = 下一个 0.5 s 的弧长（m）与航向变化（rad）。
动作有界：``ds ∈ [0, 10] m``、``dθ ∈ [-0.6, 0.6] rad``（0.5 s 内 ≈ ±1.2 rad/s，
覆盖路口转向），由 ``tanh`` 压缩保证；默认界可在构造时覆盖。

- ``forward`` 返回**已压缩且落在动作界内**的均值与**有界** logstd
  （``clamp(log_std_min, log_std_max)``，默认 ``[-5, 0]``）；
- :meth:`log_prob` 使用 tanh 变换的雅可比修正，PPO 可直接用；
- ``log_std`` 是可学习参数（每维一个），初始化 ``-1.0``（std≈0.37，动作尺度友好）。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from net.encoders import H

#: 动作下界/上界（ds, dθ）
ACTION_LOW: tuple[float, float] = (0.0, -0.6)
ACTION_HIGH: tuple[float, float] = (10.0, 0.6)
LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.0
_EPS = 1e-6


class PolicyHead(nn.Module):
    """2 维均值 + 有界 logstd 的策略头。"""

    def __init__(
        self,
        hidden: int = H,
        action_low: tuple[float, float] = ACTION_LOW,
        action_high: tuple[float, float] = ACTION_HIGH,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        log_std_init: float = -1.0,
    ):
        super().__init__()
        if not (len(action_low) == len(action_high) == 2):
            raise ValueError("action_low/action_high 必须各为 2 维")
        self.trunk = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        self.mu = nn.Linear(hidden, 2)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    # ---------------------------------------------------------------- 基础量
    def raw(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """返回未压缩均值 ``raw_mu (B,2)`` 与已 clamp 的 ``log_std (B,2)``。"""
        raw_mu = self.mu(self.trunk(latent))
        log_std = self.log_std.clamp(self.log_std_min, self.log_std_max).expand_as(raw_mu)
        return raw_mu, log_std

    def squash(self, raw_mu: Tensor) -> Tensor:
        """tanh 压缩到 ``[low, high]``。"""
        unit = 0.5 * (torch.tanh(raw_mu) + 1.0)
        return self.action_low + (self.action_high - self.action_low) * unit

    def forward(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """``latent (B,H)`` -> ``(action_mu (B,2), action_logstd (B,2))``，二者均有界。"""
        raw_mu, log_std = self.raw(latent)
        return self.squash(raw_mu), log_std

    # ---------------------------------------------------------------- 分布操作
    def sample(self, latent: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """重参数化采样（用于 PPO rollout）。"""
        raw_mu, log_std = self.raw(latent)
        noise = torch.randn(raw_mu.shape, generator=generator, dtype=raw_mu.dtype, device=raw_mu.device)
        return self.squash(raw_mu + noise * log_std.exp())

    def log_prob(self, latent: Tensor, action: Tensor) -> Tensor:
        """tanh 压缩后的对数概率 ``(B,)``（含雅可比修正）。"""
        raw_mu, log_std = self.raw(latent)
        span = (self.action_high - self.action_low).clamp(min=_EPS)
        unit = (2.0 * (action - self.action_low) / span - 1.0).clamp(-1.0 + _EPS, 1.0 - _EPS)
        raw_action = torch.atanh(unit)
        base = -0.5 * (
            ((raw_action - raw_mu) / log_std.exp()) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_det = torch.log(span * 0.5 * (1.0 - unit**2) + _EPS)
        return (base - log_det).sum(dim=-1)


class ValueHead(nn.Module):
    """状态价值 ``V(s) (B,1)``（PPO+GAE 必需）。"""

    def __init__(self, hidden: int = H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent)
