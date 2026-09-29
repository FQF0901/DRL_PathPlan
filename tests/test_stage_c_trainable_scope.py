"""R2/P0-6 回归：阶段 C 冻结范围（design allowlist）与可训练参数组表。

``design``（默认）= 只保留 ``policy.*`` / ``value.*`` / ``plan_head.moe.experts.*`` /
``plan_head.moe.router.*`` / ``plan_head.moe.residual_scale`` 可训；其余（encoders /
mem_encoder / plan_head 主干 fusion·norm·ego_next·primary / st_gnn）全部冻结。
组表 ``(前缀 → 参数量/LR)`` 与 ``build_optimizer`` 的 LR 分组同口径。
"""

from __future__ import annotations

from pipeline.stages import _load_yaml, build_model
from pipeline.trainer import (
    STAGE_C_DESIGN_PREFIXES,
    apply_trainable_allowlist,
    trainable_param_groups,
)

_FROZEN_PROBES = (
    "encoders.ego.weight",
    "mem_encoder.ego_attn.value.weight",
    "plan_head.fusion.0.weight",
    "plan_head.norm.weight",
    "plan_head.ego_next.0.weight",
    "plan_head.moe.primary.0.weight",
    "st_gnn.spatial.layers.0.node_mlp.0.weight",
)
_TRAINABLE_PROBES = (
    "policy.mu.weight",
    "policy.trunk.0.weight",
    "value.net.2.weight",
    "plan_head.moe.experts.0.0.weight",
    "plan_head.moe.router.2.weight",
    "plan_head.moe.residual_scale",
)


def _model():
    return build_model(_load_yaml("config/model.yaml"))


def test_design_scope_allowlist_keeps_only_specific_and_heads() -> None:
    model = _model()
    frozen = apply_trainable_allowlist(model, STAGE_C_DESIGN_PREFIXES)

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names, "design allowlist 不得为空"
    assert all(name.startswith(STAGE_C_DESIGN_PREFIXES) for name in trainable_names), trainable_names
    for name in _FROZEN_PROBES:
        assert name in frozen and not model.get_parameter(name).requires_grad, f"{name} 应冻结"
    for name in _TRAINABLE_PROBES:
        assert name not in frozen and model.get_parameter(name).requires_grad, f"{name} 应可训"


def test_trainable_param_groups_match_optimizer_lr_grouping() -> None:
    model = _model()
    apply_trainable_allowlist(model, STAGE_C_DESIGN_PREFIXES)
    groups = trainable_param_groups(model, lr=1e-4, primary_lr_scale=0.1)
    by_prefix = {group["prefix"]: group for group in groups}
    assert set(by_prefix) == {
        "policy.",
        "value.",
        "plan_head.moe.experts.",
        "plan_head.moe.router.",
        "plan_head.moe.residual_scale",
    }
    # design 下 primary 冻结 → 无 ×primary_lr_scale 组
    assert all(group["lr"] == 1e-4 for group in groups)
    assert sum(int(group["params"]) for group in groups) == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    expected = {
        "plan_head.moe.experts.": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("plan_head.moe.experts.")
        ),
        "plan_head.moe.router.": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("plan_head.moe.router.")
        ),
    }
    for prefix, params in expected.items():
        assert by_prefix[prefix]["params"] == params
