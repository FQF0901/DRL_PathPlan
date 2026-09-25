"""策略头与价值头：高斯策略输出有界动作 ``(ds, dθ)``，并给出 PPO 所需的对数概率。

动作约定（p2-contract §0）
-------------------------
``(ds, dθ)`` = 下一个 0.5 s 的弧长（m）与航向变化（rad）。
动作有界：``ds ∈ [0, 10] m``、``dθ ∈ [-0.6, 0.6] rad``（0.5 s 内 ≈ ±1.2 rad/s，
覆盖路口转向），由 ``sigmoid`` 压缩保证；默认界可在构造时覆盖。

为什么不是 tanh（实测缺陷，2026-09-25）：tanh 在 ``raw < -2`` 区域的导数
``1-tanh²(raw)`` 指数消失（raw=-2.6 时 ≈0.011），BC/PPO 梯度不足以拉回；
薄切片训练中策略 raw(ds) 从 -0.9 漂到 -2.6 → ds≈0.07 m 的蠕动策略。
``sigmoid`` 在专家数据所在的中间区间（ds≈3 m ⇔ raw≈-0.85）导数 ≈0.16，
且末层 ``mu`` 零初始化 → 初始 raw=0 → 初始动作 = 界中点 ``(5.0 m, 0 rad)``，
新策略从中速巡航起步、初始梯度健康。

- ``forward`` 返回**已压缩且落在动作界内**的均值与**有界** logstd
  （``clamp(log_std_min, log_std_max)``，默认 ``[-5, 0]``）；
- :meth:`log_prob` 使用 sigmoid 变换的雅可比修正，PPO 可直接用；
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
    """2 维均值 + 有界 logstd 的策略头（sigmoid 压缩，见模块 docstring）。"""

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
        # 末层零初始化：初始 raw=0 → 动作 = 界中点 (5 m, 0 rad)，远离饱和区
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
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
        """sigmoid 压缩到 ``[low, high]``（``low + span·σ(raw)``）。"""
        unit = torch.sigmoid(raw_mu)
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
        """sigmoid 压缩后的对数概率 ``(B,)``（含雅可比修正）。"""
        raw_mu, log_std = self.raw(latent)
        span = (self.action_high - self.action_low).clamp(min=_EPS)
        unit = ((action - self.action_low) / span).clamp(_EPS, 1.0 - _EPS)
        raw_action = torch.log(unit) - torch.log1p(-unit)  # logit(unit)
        base = -0.5 * (
            ((raw_action - raw_mu) / log_std.exp()) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_det = torch.log(span * unit * (1.0 - unit) + _EPS)
        return (base - log_det).sum(dim=-1)


class ValueHead(nn.Module):
    """状态价值 ``V(s) (B,1)``（PPO+GAE 必需）。"""

    def __init__(self, hidden: int = H):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, latent: Tensor) -> Tensor:
        return self.net(latent)
