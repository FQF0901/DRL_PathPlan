"""时序编码：逐节点 GRU（6 帧 @0.5 s），``hist_valid`` 显式门控预热补位帧。

为什么必须用 ``hist_valid`` 而不是 ``*_hist_mask``
------------------------------------------------
``env/obs/memory.py`` 在预热期（真实帧不足 6 帧）会**复制最旧的真实帧**补满窗口，
且复制帧的 ``*_hist_mask`` 仍为 1（它确实是"有效槽位"）。若只用 mask，
网络会把同一帧当成多帧历史（p2-contract §8.4 的 must-fix）。
本模块的处理：

- ``hist_valid[t] == 0`` 的帧：输入清零、GRU 隐状态保持不变（等价于整帧跳过）；
- ``hist_valid[t] == 1`` 的帧：正常更新隐状态。

由于 ``memory.stack`` 产生的是"前缀补位、后缀真实"的窗口，
上述规则退化为"只从最早真实帧开始累积"，既不会引入复制帧，也不依赖补位位置。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from net.encoders import H


class TemporalEncoder(nn.Module):
    """逐节点共享的 GRUCell（对 6 帧循环调用）。

    使用 ``GRUCell`` 手写时间循环而非 ``nn.GRU``：需要在时间维按帧插入
    ``hist_valid`` 门控（整帧无效时冻结隐状态），``nn.GRU`` 无法表达该操作。
    """

    def __init__(self, hidden: int = H):
        super().__init__()
        self.hidden = int(hidden)
        self.cell = nn.GRUCell(hidden, hidden)

    def step(self, x: Tensor, hidden: Tensor, valid: Tensor | None = None) -> Tensor:
        """单帧推进。

        Args:
            x: ``(B, N, H)`` 当前帧节点特征（无效槽位应为 0）。
            hidden: ``(B, N, H)`` 上一帧隐状态。
            valid: ``(B,)`` 帧有效标志；0 表示该帧整帧跳过（隐状态冻结）。
        """
        batch, num_nodes, _ = x.shape
        updated = self.cell(
            x.reshape(batch * num_nodes, -1),
            hidden.reshape(batch * num_nodes, -1),
        ).reshape(batch, num_nodes, self.hidden)
        if valid is None:
            return updated
        return torch.where(valid.view(batch, 1, 1) > 0, updated, hidden)

    def forward(
        self,
        feat: Tensor,
        frame_mask: Tensor | None = None,
        valid: Tensor | None = None,
    ) -> Tensor:
        """``feat (B, T, N, H)`` -> ``(B, N, H)``。

        Args:
            feat: 历史节点特征（编码器已按槽位 mask 清零）。
            frame_mask: ``(B, T, N)`` 槽位掩码；无效槽位的输入再乘一次 0。
            valid: ``(B, T)`` = ``hist_valid``；**决定哪些帧参与时序聚合**。
        """
        if feat.ndim != 4:
            raise ValueError(f"TemporalEncoder 期望 (B,T,N,H)，收到 {tuple(feat.shape)}")
        batch, frames, num_nodes, _ = feat.shape
        hidden = torch.zeros((batch, num_nodes, self.hidden), dtype=feat.dtype, device=feat.device)
        for t in range(frames):
            x = feat[:, t]
            if frame_mask is not None:
                x = x * frame_mask[:, t].unsqueeze(-1)
            frame_valid = None if valid is None else valid[:, t]
            hidden = self.step(x, hidden, frame_valid)
        return hidden
