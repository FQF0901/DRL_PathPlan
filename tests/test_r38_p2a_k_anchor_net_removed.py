"""R38-P2a：K-anchor 减枝移除守卫（net 实现面；S3 = 删实现）。

锁定（`docs/deprecation_manifest.yaml` B 组）：

- ``net/anchor.py`` 与 ``tools/fit_plan_anchors.py`` 已删除（实现只存在于 `tools/legacy/k_anchor/`）；
- ``PlanHead`` 无锚参数/无锚 buffer/无 ``set_anchors``/``set_anchor_mode``/``plan_anchors``；
- ``num_anchors=0``（历史默认关）的键集不变：默认模型 state_dict 无任何锚键。
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_net_anchor_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("net.anchor")
    assert not (ROOT / "net" / "anchor.py").exists()
    assert not (ROOT / "tools" / "fit_plan_anchors.py").exists()
    assert not (ROOT / "config" / "plan_anchors_k6.json").exists()


def test_plan_head_has_no_anchor_api() -> None:
    from net.plan_head import PlanHead

    signature = inspect.signature(PlanHead.__init__)
    for name in ("num_anchors", "anchor_embed_dim", "anchor_temperature", "anchor_hard"):
        assert name not in signature.parameters, f"K-anchor 构造参数 {name} 不得存在"
    head = PlanHead(hidden=16, num_experts=2, expert_hidden=8)
    for attr in ("num_anchors", "set_anchors", "set_anchor_mode", "plan_anchors",
                 "anchor_head", "speed_head", "residual_head", "anchor_embed"):
        assert not hasattr(head, attr), f"K-anchor API {attr} 不得存在"
    state_keys = list(head.state_dict())
    assert not any("anchor" in key or "speed_head" in key or "residual_head" in key
                   for key in state_keys), state_keys


def test_plan_head_forward_shape_contract() -> None:
    """无锚分支的 forward 契约不变：(latent, ego_next, moe_aux)。"""
    import torch

    from net.plan_head import EGO_NEXT_DIM, PlanHead

    head = PlanHead(hidden=16, num_experts=2, expert_hidden=8).eval()
    batch = 2
    tokens = [torch.randn(batch, 16) for _ in range(6)]
    with torch.no_grad():
        latent, ego_next, moe_aux = head(*tokens)
    assert tuple(latent.shape) == (batch, 16)
    assert tuple(ego_next.shape) == (batch, EGO_NEXT_DIM)
    for key in ("router_logits", "expert_weights", "moe_enabled"):
        assert key in moe_aux, key
