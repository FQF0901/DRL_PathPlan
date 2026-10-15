"""R38-P3-G：旧 others 布局（28 维，schema v2/v3）→ v4（33 维）**显式适配器**。

出处（快照 commit = ``2cfd8e8``（R38-P3 前 HEAD），行号为快照时点）：

- ``net/mem.py:110-138`` → ``_remap_legacy_others``（旧 28→33 静默零填充重排 + 一次性
  RuntimeWarning）= :func:`remap_legacy_others_28_to_33`（纯函数，无告警）；
- ``net/encoders.py:59-61`` → ``OTHERS_LEGACY_DIM`` / ``OTHERS_HEAD_LEGACY_DIM``（随迁）。

**判定**：`LEGACY_REPRO`（R38 弃用清单 G 组：旧布局重排违反"缺失静默
填充"守则——static 段补 0 = "无静态障碍"，会静默错读）。R38-P3 起 ``net/mem.py`` 对 28 维
输入 **fail-fast**（不再静默重排），迁移路径二选一：

1. **重采数据**（推荐；当前 env schema v7 / others = 33 维）；
2. **显式转换**：本模块 :func:`remap_legacy_others_28_to_33` 把旧数据集/旧缓存一次性转成
   33 维后再喂 ``mem_from_obs``（显式选择 ⇒ 语义可见，"补 0 static 段"由调用方负责）。

布局（与 ``net/mem.py`` docstring 同源）：

- 旧（28）：``[nav(11), speed_limit(1), signal(4), road_class(12)]``；
- 新（33）：``[nav(11), speed_limit(1), signal(4), static(5), road_class(12)]``；
- 未迁移的假设：旧数据无 static 段 ⇒ 转换后 static 全 0（= "无静态障碍"），该语义由调用方
  显式承担（R38-P3 后不再由 net 静默补全）。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

__all__ = [
    "OTHERS_LEGACY_DIM",
    "OTHERS_HEAD_LEGACY_DIM",
    "OTHERS_STATIC_DIM",
    "remap_legacy_others_28_to_33",
]

#: 旧 others 布局（env schema v2/v3）：``nav(11)+speed_limit(1)+signal(4)+road_class(12)`` = 28。
OTHERS_LEGACY_DIM = 28
#: 旧布局中 static 段之前的公共前缀维数（nav + speed_limit + signal）。
OTHERS_HEAD_LEGACY_DIM = 16
#: v4 新增 static 段维数（present + gap_norm + 相对车道 one-hot(3)）；转换时填空补 0。
OTHERS_STATIC_DIM = 5

#: v4 目标布局维数（= ``net.encoders.DEFAULT_OTHERS_DIM``；不 import net 以保持 legacy 自包含）。
OTHERS_V4_DIM = OTHERS_LEGACY_DIM + OTHERS_STATIC_DIM


def remap_legacy_others_28_to_33(tensor: Tensor) -> Tensor:
    """:func:`remap_legacy_others`（``net/mem.py:110-138`` @ 2cfd8e8）的纯函数快照。

    ``(..., 28)`` → ``(..., 33)``：``[0,16)`` 原样、``[16,21)`` 补 0（static 未知 = 无静态
    障碍）、``[21,33)`` ← 旧 ``[16,28)``（road_class 平移）。已是 33 维 ⇒ 原样返回；
    其他维数 ⇒ ``ValueError``（不静默）。
    """
    if not torch.is_tensor(tensor):
        raise TypeError(f"remap_legacy_others_28_to_33 需要 torch.Tensor，收到 {type(tensor).__name__}")
    dim = int(tensor.shape[-1])
    if dim == OTHERS_V4_DIM:
        return tensor
    if dim != OTHERS_LEGACY_DIM:
        raise ValueError(
            f"others 特征维 {dim} 既不是旧布局 {OTHERS_LEGACY_DIM}（schema v2/v3）也不是 v4 布局"
            f" {OTHERS_V4_DIM}：拒绝静默转换（R38-P3-G）。"
        )
    out = torch.zeros(
        tuple(tensor.shape[:-1]) + (OTHERS_V4_DIM, ),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    out[..., :OTHERS_HEAD_LEGACY_DIM] = tensor[..., :OTHERS_HEAD_LEGACY_DIM]
    out[..., OTHERS_HEAD_LEGACY_DIM + OTHERS_STATIC_DIM:] = tensor[..., OTHERS_HEAD_LEGACY_DIM:]
    return out


def _self_check() -> int:
    """模块自检（CPU；仅 torch）：逐位对齐旧 ``_remap_legacy_others`` 语义。"""
    legacy = torch.arange(OTHERS_LEGACY_DIM, dtype=torch.float32).reshape(1, 1, OTHERS_LEGACY_DIM)
    out = remap_legacy_others_28_to_33(legacy)
    assert tuple(out.shape) == (1, 1, OTHERS_V4_DIM)
    assert torch.equal(out[..., :OTHERS_HEAD_LEGACY_DIM], legacy[..., :OTHERS_HEAD_LEGACY_DIM])
    assert torch.equal(
        out[..., OTHERS_HEAD_LEGACY_DIM:OTHERS_HEAD_LEGACY_DIM + OTHERS_STATIC_DIM],
        torch.zeros(1, 1, OTHERS_STATIC_DIM),
    )
    assert torch.equal(out[..., OTHERS_HEAD_LEGACY_DIM + OTHERS_STATIC_DIM:], legacy[..., OTHERS_HEAD_LEGACY_DIM:])
    v4 = torch.zeros(2, 4, OTHERS_V4_DIM)
    assert remap_legacy_others_28_to_33(v4) is v4
    print("[tools.legacy.others_layout] self-check OK")
    for bad in (torch.zeros(1, 8), torch.zeros(1, 16)):
        try:
            remap_legacy_others_28_to_33(bad)
        except ValueError:
            continue
        raise AssertionError("非 28/33 维必须 ValueError")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
