"""策略头与价值头（v6）：1 学习查询 × 令牌集合的交叉注意力头 → 有界动作分布 / V(s)。

v6 规格（docs/v6_net_design.md §1，冻结）
----------------------------------------
- 旧实现把 OD/LD 掩码均值池化成单 token（``_masked_mean``）再喂小 MLP；v6 改为
  **交叉注意力头**（policy/value 各一份，不共享权重）：
  - 查询 K=1（学习参数 ``query (1,H)``；Gate0 裁定 K=8 时 q1..q7 无监督 → 死参数）；
  - 键/值令牌集合（≤37）：OD 16 + LD 16 + others 1 + ego 1 + nav 1 + signal 1 +
    **plan_head 融合 latent 1**（保 experts → policy/value 条件通路）；
  - 4 头（d=128 时 head_dim=32）、pre-LN + 残差、默认 1 层（1–2 层可配）；
  - 注意力输出 → pre-LN → 小 MLP（``H → trunk_hidden → H``）→ 输出层（policy:
    ``mu``/``logstd`` 各 H→2；value: ``net → net_hidden → 1``）。
- 掩码：``key_mask (B,T)``（1=有效）拼自 ``od_live``/``ld_live``/others/``nav_mask``/
  ``signal_mask``；ego 与融合 latent 恒有效；**全无效行输出严格 0 且不产生 NaN**
  （数值安全：强制一个 key 参与 softmax，再按行置 0）。
- 参数预算（v8 参数再分配；d=128、1 层）：cross-attn 投影 4×(128²+128)=66,048；查询
  1×128=128；pre-LN 2×2×128=512；trunk ``128→160→128``=41,248；输出层 policy
  2×(128×2+2)=516 / value ``128→256→1``=33,281 ⇒ **policy ≈108.5k / value ≈100.0k**。

动作约定（p2-contract §0，不变）
-------------------------------
``(ds, dθ)`` = 下一个 0.5 s 的弧长（m）与航向变化（rad）；动作有界：
``ds ∈ [0,10] m``、``dθ ∈ [-0.6,0.6] rad``，由 ``sigmoid`` 压缩保证（为什么不是 tanh
见旧注释：tanh 在 ``raw<-2`` 区域导数指数消失，薄切片曾坍缩成蠕动策略）。

分布接口（不变）：``raw/squash/log_prob/sample`` 与 logstd clamp ``[-5,0]`` 全部保留。
- ``mu`` 末层零初始化 ⇒ 初始 ``raw_mu=0`` → 初始动作 = 界中点 ``(5.0 m, 0 rad)``，
  新策略从中速巡航起步（BC 回归契约，见 tests/test_bc_pretrain.py）；
- ``logstd`` 由头输出（``raw_logstd``）：末层小权重（std=0.01）+ bias=``log_std_init``
  ⇒ 初始 logstd ≈ -1.0（std≈0.37）；权重非零保证输出层上游参数有梯度（无死参数）。
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

#: 交叉注意力头默认超参（v6 冻结：4 头、默认 1 层；head_dim = H/num_heads）
ATTN_HEADS = 4
ATTN_LAYERS = 1


class _CrossAttnLayer(nn.Module):
    """单层 pre-LN 交叉注意力（1 查询 × 令牌集合）+ 残差。"""

    def __init__(self, hidden: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads, batch_first=True)

    def forward(self, query: Tensor, tokens: Tensor, key_mask: Tensor) -> Tensor:
        """``query (B,1,H)``, ``tokens (B,T,H)``, ``key_mask (B,T)``（1=有效）→ ``(B,1,H)``。"""
        valid = key_mask > 0.5
        row_valid = valid.any(dim=1)  # (B,)：该行是否有任何有效 key
        # 全无效行：强制第 0 个 key 有效，避免 softmax 分母为 0 → NaN；输出随后置 0。
        # （模型组装时 ego/latent 恒有效，此分支只服务"直接以全无效 mask 调头"的边界用例。）
        safe = valid.clone()
        safe[:, 0] = safe[:, 0] | ~row_valid
        out = self.attn(
            self.norm(query), tokens, tokens, key_padding_mask=~safe, need_weights=False
        )[0]
        return query + out * row_valid.to(out.dtype).view(-1, 1, 1)


class CrossAttnHead(nn.Module):
    """v6 交叉注意力头基类：K=1 学习查询 × 令牌集合 → 头特征 ``(B,1,H)``。

    结构 = 查询嵌入 + ``layers`` 层 pre-LN 交叉注意力（残差）+ 输出 pre-LN；
    子类在此之上接任务头（policy: trunk+mu/logstd；value: net→V）。
    """

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
        #: K=1 学习查询（Gate0：K=8 时 q1..q7 无监督信号 → 零梯度死参数）
        self.query = nn.Parameter(torch.zeros(1, hidden))
        nn.init.normal_(self.query, std=0.02)
        self.layers = nn.ModuleList(
            [_CrossAttnLayer(hidden, num_heads) for _ in range(layers)]
        )
        self.norm_out = nn.LayerNorm(hidden)

    def forward(self, tokens: Tensor, key_mask: Tensor) -> Tensor:
        """``tokens (B,T,H)`` + ``key_mask (B,T)`` → 头特征 ``(B,1,H)``。"""
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
        row_valid = (key_mask > 0.5).any(dim=1)  # (B,)：全无效行输出严格 0（规格 §1.2）
        return self.norm_out(query) * row_valid.to(query.dtype).view(-1, 1, 1)


class PolicyHead(CrossAttnHead):
    """交叉注意力策略头：令牌集合 → 有界动作 ``(ds,dθ)`` 分布（sigmoid 压缩）。"""

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
        #: 小 MLP（v8 参数再分配：``H → trunk_hidden → H``；默认 128 → 160 → 128）
        self.trunk = nn.Sequential(
            nn.Linear(self.hidden, int(trunk_hidden)),
            nn.GELU(),
            nn.Linear(int(trunk_hidden), self.hidden),
        )
        self.mu = nn.Linear(self.hidden, 2)
        # 末层零初始化：初始 raw=0 → 动作 = 界中点 (5 m, 0 rad)，远离饱和区
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        #: logstd 头（raw_logstd）；小权重 + bias=log_std_init ⇒ 初始 logstd≈-1.0，
        #: 且权重非零 ⇒ trunk/注意力栈经此路径有梯度（无死参数）
        self.logstd = nn.Linear(self.hidden, 2)
        nn.init.normal_(self.logstd.weight, std=0.01)
        nn.init.constant_(self.logstd.bias, float(log_std_init))
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    # ---------------------------------------------------------------- 基础量
    def head_features(self, tokens: Tensor, key_mask: Tensor) -> Tensor:
        """注意力栈 + 小 MLP → ``(B,H)``（policy 输出层的输入）。"""
        return self.trunk(super().forward(tokens, key_mask)[:, 0])

    def raw(self, tokens: Tensor, key_mask: Tensor) -> tuple[Tensor, Tensor]:
        """返回未压缩均值 ``raw_mu (B,2)`` 与已 clamp 的 ``log_std (B,2)``。"""
        features = self.head_features(tokens, key_mask)
        raw_mu = self.mu(features)
        log_std = self.logstd(features).clamp(self.log_std_min, self.log_std_max)
        return raw_mu, log_std

    def squash(self, raw_mu: Tensor) -> Tensor:
        """sigmoid 压缩到 ``[low, high]``（``low + span·σ(raw)``）。"""
        unit = torch.sigmoid(raw_mu)
        return self.action_low + (self.action_high - self.action_low) * unit

    def forward(self, tokens: Tensor, key_mask: Tensor) -> tuple[Tensor, Tensor]:
        """``(tokens, key_mask)`` -> ``(action_mu (B,2), action_logstd (B,2))``，二者均有界。"""
        raw_mu, log_std = self.raw(tokens, key_mask)
        return self.squash(raw_mu), log_std

    # ---------------------------------------------------------------- 分布操作
    def sample(
        self, tokens: Tensor, key_mask: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        """重参数化采样（用于 PPO rollout）。"""
        raw_mu, log_std = self.raw(tokens, key_mask)
        noise = torch.randn(
            raw_mu.shape, generator=generator, dtype=raw_mu.dtype, device=raw_mu.device
        )
        return self.squash(raw_mu + noise * log_std.exp())

    def log_prob(self, tokens: Tensor, key_mask: Tensor, action: Tensor) -> Tensor:
        """sigmoid 压缩后的对数概率 ``(B,)``（含雅可比修正）。"""
        raw_mu, log_std = self.raw(tokens, key_mask)
        span = (self.action_high - self.action_low).clamp(min=_EPS)
        unit = ((action - self.action_low) / span).clamp(_EPS, 1.0 - _EPS)
        raw_action = torch.log(unit) - torch.log1p(-unit)  # logit(unit)
        base = -0.5 * (
            ((raw_action - raw_mu) / log_std.exp()) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_det = torch.log(span * unit * (1.0 - unit) + _EPS)
        return (base - log_det).sum(dim=-1)


class ValueHead(CrossAttnHead):
    """交叉注意力价值头：令牌集合 → ``V(s) (B,1)``（PPO+GAE 必需）。"""

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
