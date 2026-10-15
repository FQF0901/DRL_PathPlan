"""R38-P2b：旧 ``PolicyHead`` 快照（**逐行口径拷贝**，仅供 legacy 装载器/离线诊断使用）。

出处（快照 commit = ``dd0a8c2``，R38-P2b 前 HEAD）：

- ``net/policy.py`` 整文件（``PolicyHead`` / ``ValueHead`` / ``CrossAttnHead`` / ``_CrossAttnLayer``
  + 动作界与 logstd 常量）；
- 唯一改动：``from net.encoders import H`` → 本地常量 :data:`H`（R38 快照值 128），
  使快照在 net 结构演进后仍可独立实例化旧 ckpt 的 ``policy.*`` 键面。

用途（D 组"旧 PolicyHead 迁 legacy 模型加载器"落点）
-----------------------------------------------------
1. **离线诊断**：``tools/diagnostics/legacy_policy_diag.py`` 复算旧头 ``policy_action_mu`` /
   ``action_logstd``（R38-P2b 后正式 forward 不再并行计算旧头诊断）；
2. **旧 ckpt 键面检查**：以 :class:`PolicyHead` 实例化 ``policy.*``（19–21 键）做形状/语义审计；
3. **历史复现**：与 ``docs/repo_handbook.md`` 附录 D §4.4 的 ``policy.log_std →
   policy.logstd`` 改名族对账。

**注意**：生产 net 的 ``net/policy.py`` **未**删除（默认关路径的唯一动作出口仍用它）；
本快照是"旧头独立可实例化"的冻结副本，**不被生产代码导入**（CI 守卫 #1 扫描面）。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

#: ``net/encoders.py::H`` 的快照值（快照 commit 时点 = 128；本快照自包含）。
H = 128

#: 动作下界/上界（ds, dθ）
ACTION_LOW: tuple[float, float] = (0.0, -0.6)
ACTION_HIGH: tuple[float, float] = (10.0, 0.6)
LOG_STD_MIN = -5.0
LOG_STD_MAX = 0.0
_EPS = 1e-6

#: 交叉注意力头默认超参（v6 冻结：4 头、默认 1 层；head_dim = H/num_heads）
ATTN_HEADS = 4
ATTN_LAYERS = 1


class _CrossAttnLayer(nn.Module):
    """单层 pre-LN 交叉注意力（1 查询 × 令牌集合）+ 残差（出处 ``net/policy.py:54-73``）。"""

    def __init__(self, hidden: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)

    def forward(self, query: Tensor, tokens: Tensor, key_mask: Tensor) -> Tensor:
        valid = key_mask > 0.5
        row_valid = valid.any(dim=1)
        safe = valid.clone()
        safe[:, 0] = safe[:, 0] | ~row_valid
        out = self.attn(
            self.norm(query), tokens, tokens, key_padding_mask=~safe, need_weights=False
        )[0]
        return query + out * row_valid.to(out.dtype).view(-1, 1, 1)


class CrossAttnHead(nn.Module):
    """v6 交叉注意力头基类（出处 ``net/policy.py:76-116``）。"""

    def __init__(self, hidden: int = H, num_heads: int = ATTN_HEADS, layers: int = ATTN_LAYERS):
        super().__init__()
        hidden, num_heads, layers = int(hidden), int(num_heads), int(layers)
        if hidden <= 0 or hidden % num_heads != 0:
            raise ValueError(f"hidden({hidden}) 必须为正且能被 num_heads({num_heads}) 整除")
        if layers < 1:
            raise ValueError("layers 必须 >= 1")
        self.hidden = hidden
        self.num_heads = num_heads
        self.num_layers = layers
        self.query = nn.Parameter(torch.zeros(1, hidden))
        nn.init.normal_(self.query, std=0.02)
        self.layers = nn.ModuleList(
            [_CrossAttnLayer(hidden, num_heads) for _ in range(layers)]
        )
        self.norm_out = nn.LayerNorm(hidden)

    def forward(self, tokens: Tensor, key_mask: Tensor) -> Tensor:
        if tokens.ndim != 3 or key_mask.ndim != 2:
            raise ValueError(
                f"tokens 应为 (B,T,H)、key_mask 应为 (B,T)，收到 {tuple(tokens.shape)}/"
                f"{tuple(key_mask.shape)}"
            )
        if int(tokens.shape[0]) != int(key_mask.shape[0]) or int(tokens.shape[1]) != int(key_mask.shape[1]):
            raise ValueError(
                f"tokens {tuple(tokens.shape)} 与 key_mask {tuple(key_mask.shape)} 的 B/T 不一致"
            )
        query = self.query.unsqueeze(0).expand(int(tokens.shape[0]), -1, -1)
        for layer in self.layers:
            query = layer(query, tokens, key_mask)
        row_valid = (key_mask > 0.5).any(dim=1)
        return self.norm_out(query) * row_valid.to(query.dtype).view(-1, 1, 1)


class PolicyHead(CrossAttnHead):
    """交叉注意力策略头（出处 ``net/policy.py:119-200``；快照逐行拷贝）。"""

    def __init__(
        self,
        hidden: int = H,
        action_low: tuple[float, float] = ACTION_LOW,
        action_high: tuple[float, float] = ACTION_HIGH,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        log_std_init: float = -1.0,
        num_heads: int = ATTN_HEADS,
        layers: int = ATTN_LAYERS,
        trunk_hidden: int = 160,
    ):
        super().__init__(hidden, num_heads, layers)
        if not (len(action_low) == len(action_high) == 2):
            raise ValueError("action_low/action_high 必须各为 2 维")
        self.trunk = nn.Sequential(
            nn.Linear(self.hidden, int(trunk_hidden)),
            nn.GELU(),
            nn.Linear(int(trunk_hidden), self.hidden),
        )
        self.mu = nn.Linear(self.hidden, 2)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        self.logstd = nn.Linear(self.hidden, 2)
        nn.init.normal_(self.logstd.weight, std=0.01)
        nn.init.constant_(self.logstd.bias, float(log_std_init))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    def head_features(self, tokens: Tensor, key_mask: Tensor) -> Tensor:
        return self.trunk(super().forward(tokens, key_mask)[:, 0])

    def raw(self, tokens: Tensor, key_mask: Tensor) -> tuple[Tensor, Tensor]:
        features = self.head_features(tokens, key_mask)
        raw_mu = self.mu(features)
        log_std = self.logstd(features).clamp(self.log_std_min, self.log_std_max)
        return raw_mu, log_std

    def squash(self, raw_mu: Tensor) -> Tensor:
        unit = torch.sigmoid(raw_mu)
        return self.action_low + (self.action_high - self.action_low) * unit

    def forward(self, tokens: Tensor, key_mask: Tensor) -> tuple[Tensor, Tensor]:
        raw_mu, log_std = self.raw(tokens, key_mask)
        return self.squash(raw_mu), log_std

    def sample(
        self, tokens: Tensor, key_mask: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        raw_mu, log_std = self.raw(tokens, key_mask)
        noise = torch.randn(
            raw_mu.shape, generator=generator, dtype=raw_mu.dtype, device=raw_mu.device
        )
        return self.squash(raw_mu + noise * log_std.exp())

    def log_prob(self, tokens: Tensor, key_mask: Tensor, action: Tensor) -> Tensor:
        raw_mu, log_std = self.raw(tokens, key_mask)
        span = (self.action_high - self.action_low).clamp(min=_EPS)
        unit = ((action - self.action_low) / span).clamp(_EPS, 1.0 - _EPS)
        raw_action = torch.log(unit) - torch.log1p(-unit)
        base = -0.5 * (
            ((raw_action - raw_mu) / log_std.exp()) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_det = torch.log(span * unit * (1.0 - unit) + _EPS)
        return (base - log_det).sum(dim=-1)


class ValueHead(CrossAttnHead):
    """交叉注意力价值头（出处 ``net/policy.py:203-221``；快照随 PolicyHead 一并冻结）。"""

    def __init__(
        self,
        hidden: int = H,
        num_heads: int = ATTN_HEADS,
        layers: int = ATTN_LAYERS,
        net_hidden: int = 256,
    ):
        super().__init__(hidden, num_heads, layers)
        self.net = nn.Sequential(
            nn.Linear(self.hidden, int(net_hidden)),
            nn.GELU(),
            nn.Linear(int(net_hidden), 1),
        )

    def forward(self, tokens: Tensor, key_mask: Tensor) -> Tensor:
        return self.net(super().forward(tokens, key_mask)[:, 0])


__all__ = [
    "ACTION_HIGH",
    "ACTION_LOW",
    "ATTN_HEADS",
    "ATTN_LAYERS",
    "CrossAttnHead",
    "H",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "PolicyHead",
    "ValueHead",
]
