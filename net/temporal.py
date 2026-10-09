"""时序聚合：对 mem 的 T 帧做**掩码注意力池化**（v2 mem-bank 契约）。

为什么是注意力而不是 GRU
------------------------
v2 的输入是 4 个 per-modality mem（``(B,T,...)``，最后一帧 = 当前帧），网络对
OD / Ego / Others **各自**做一次"6 帧 + validity mask"的注意力池化；LD 不做时序
（见 :mod:`net.mem` 的说明）。注意力相对 GRU 的好处：

- 每帧对当前决策的贡献可解释（输出权重即"看哪几帧"），且不引入跨帧隐状态；
- ``hist_valid`` 的预热补位帧可直接用 mask 屏蔽（补位帧复制最旧真实帧但 mask=1）；
- rollout 里 mem 窗口是"滑动 + 合成帧"的，注意力不需要维护/重置隐状态。

掩码语义
--------
``ok = slot_mask & valid``（``slot_mask`` 对 ego/others 为 None）：

- 补位帧（``valid=0``）整帧不参与（不引入复制帧语义，兼容旧 §8.4 的 must-fix）；
- 无效槽位不参与；
- **整列全无效**（某槽在窗口内没有任何有效帧）时 pooled=0 → 经 ``out`` 层（带 bias）输出 = bias（非严格 0）；
  反向有限（softmax 用 ``-1e4`` 屏蔽 + 乘 ``ok`` 归零，避免 ``-inf`` 产生 NaN）。

帧龄嵌入
--------
同样的特征出现在不同帧龄上应当可区分；键上加入可学的帧龄嵌入
（``age = T-1-pos``，0 = 最新）。LD 不走本模块（见 :mod:`net.mem`）。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from net.encoders import H

#: softmax 屏蔽值（不用 ``-inf``：避免全屏蔽列在反向产生 NaN）
_MASK_FILL = -1e4


class TemporalAttention(nn.Module):
    """单查询键值注意力池化：``(B,T,N,H) -> (B,N,H)``（N=1 或槽位数）。"""

    def __init__(self, hidden: int = H, key_dim: int | None = None):
        super().__init__()
        self.hidden = int(hidden)
        self.key_dim = int(key_dim or max(8, hidden // 4))
        self.query = nn.Parameter(torch.zeros(self.key_dim))
        self.key = nn.Linear(hidden, self.key_dim)
        self.value = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, hidden)
        self.age = nn.Embedding(64, hidden)

    def forward(
        self,
        x: Tensor,
        slot_mask: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> Tensor:
        """``x (B,T,N,H)`` -> ``(B,N,H)``。

        Args:
            x: 编码后的 mem 帧（无效槽位应为 0，由编码器保证）。
            slot_mask: ``(B,T,N)`` 槽位掩码；``None`` 表示无槽位维（ego/others）。
            valid: ``(B,T)`` 帧有效性（``hist_valid``）；``None`` 表示全部有效。
        """
        if x.ndim != 4:
            raise ValueError(f"TemporalAttention 期望 (B,T,N,H)，收到 {tuple(x.shape)}")
        batch, frames, slots, _ = x.shape
        if self.age.num_embeddings < frames:
            raise ValueError(f"帧数 {frames} 超过帧龄嵌入容量 {self.age.num_embeddings}")
        age_ids = torch.arange(frames - 1, -1, -1, dtype=torch.long, device=x.device)
        x = x + self.age(age_ids).view(1, frames, 1, self.hidden)

        ok = torch.ones((batch, frames, slots), dtype=torch.bool, device=x.device)
        if slot_mask is not None:
            ok = ok & (slot_mask.reshape(batch, frames, slots) > 0.5)
        if valid is not None:
            ok = ok & (valid.reshape(batch, frames) > 0.5).unsqueeze(-1)

        logits = (self.key(x) * self.query).sum(dim=-1) / math.sqrt(self.key_dim)
        logits = torch.where(ok, logits, torch.full_like(logits, _MASK_FILL))
        alpha = torch.softmax(logits, dim=1) * ok.to(logits.dtype)
        pooled = (alpha.unsqueeze(-1) * self.value(x)).sum(dim=1)
        return self.out(pooled)
