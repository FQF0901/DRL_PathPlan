"""``tools.legacy.k_anchor`` 快照自检（CPU、只依赖 torch；``python -m`` 可跑）。

覆盖：锚字典文件加载/内置默认一致性 → legacy ``PlanHead(num_anchors=6)`` 构造 →
``set_anchors``/``plan_anchors`` 前向（键集/形状/有限性）→ 软混合与 WTA 的几何不变量。
不 import 当前 ``net.model``（模型级复现需旧 net，见 README）。
"""

from __future__ import annotations

from pathlib import Path

import torch

from tools.legacy.k_anchor.anchor import (
    K_DEFAULT,
    anchor_mixture,
    assign_anchors,
    from_lane_dtheta,
    load_anchor_dictionary,
)
from tools.legacy.k_anchor.plan_head import PlanHead

ROOT = Path(__file__).resolve().parents[3]
ANCHOR_FILE = Path(__file__).resolve().parent / "plan_anchors_k6.json"


def main() -> int:
    anchors = load_anchor_dictionary(str(ANCHOR_FILE), expected_k=K_DEFAULT)
    assert tuple(anchors.shape) == (K_DEFAULT, 6, 2), anchors.shape
    head = PlanHead(hidden=32, num_experts=2, expert_hidden=16, num_anchors=K_DEFAULT)
    head.set_anchors(anchors)
    latent = torch.randn(4, 32)
    lane_ctx = torch.zeros(4, 3)  # v8 后 lane 通道删除 → 恒 zeros（valid=0 → 恒等）
    out = head.plan_anchors(latent, lane_ctx)
    for key in ("plan", "logits", "probs", "speed_ds", "residual", "anchor_dth", "ctx"):
        assert key in out, key
    assert tuple(out["plan"].shape) == (4, 6, 2)
    assert bool(torch.isfinite(out["plan"]).all())
    # 残差末层零初始化 ⇒ 初始 plan = 速度头 + 锚 dθ 混合（lane_ctx 全零 = 恒等回投）
    plan, probs = anchor_mixture(
        out["logits"], out["residual"], head.anchor_dth, out["speed_ds"],
        lane_ctx[:, 0], lane_ctx[:, 1], lane_ctx[:, 2],
    )
    assert torch.allclose(plan, out["plan"], atol=1e-6)
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-6)
    # WTA：目标链 = 锚 2 的形状 ⇒ 分配到锚 2
    chain = torch.stack([torch.full((6,), 3.5), anchors[2, :, 1]], dim=-1).unsqueeze(0)
    assignment = assign_anchors(
        chain, anchors[..., 1], torch.ones(1, 6), torch.zeros(1), torch.zeros(1), torch.ones(1)
    )
    assert int(assignment["index"][0]) == 2
    # 车道系往返：恒等（valid=0）
    roundtrip = from_lane_dtheta(chain, chain[..., 0], torch.zeros(1), torch.zeros(1))
    assert torch.allclose(roundtrip, chain, atol=0.0)
    print(f"[k_anchor selfcheck] OK · K={K_DEFAULT} · plan={tuple(out['plan'].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
