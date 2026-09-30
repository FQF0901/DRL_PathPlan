"""v4 ①：``--value-lr-scale``（``value.*`` 参数组 lr = lr × scale；默认 1.0 = 现状）。

锁定：

- ``build_optimizer(value_lr_scale != 1.0)``：value 组 lr 正确、组序 ``[other, value?, primary]``；
- ``value_lr_scale == 1.0``（默认）：**不单独建 value 组**（组序/参数展平序与旧版逐位一致，
  旧 ckpt 优化器状态可原样恢复）——no-op 证据；
- ``trainable_param_groups`` 与优化器 LR 分组同口径（value. 组显示缩放后 lr）；
- ``PPOConfig`` 校验（非有限/负数 fail-fast）；
- 阶段 C 解析：CLI ``--value-lr-scale`` > config ``train.value_lr_scale`` > 1.0；非法 fail-fast。

不建 env、不 import metadrive（自建小模型 + 纯解析）。
"""

from __future__ import annotations

import pytest
import torch

from pipeline.stages import _parse_args, _resolve_stage_c_value_lr_scale
from pipeline.trainer import PPOConfig, build_optimizer, trainable_param_groups


class _TinyModel(torch.nn.Module):
    """参数命名覆盖四个组：policy / value / experts（specific）/ other（encoders）。"""

    def __init__(self) -> None:
        super().__init__()
        self.encoders = torch.nn.ModuleDict({"ego": torch.nn.Linear(4, 8)})
        self.policy = torch.nn.ModuleDict({"mu": torch.nn.Linear(8, 2)})
        self.value = torch.nn.ModuleDict({"net": torch.nn.Linear(8, 1)})
        self.plan_head = torch.nn.ModuleDict(
            {"moe": torch.nn.ModuleDict({"experts": torch.nn.ModuleDict({"0": torch.nn.Linear(8, 8)})})}
        )


def _named_groups(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> dict:
    """优化器参数组 → ``{组内参数名集合: lr}``（按参数 identity 反查名字）。"""
    index = {id(parameter): name for name, parameter in model.named_parameters()}
    out = {}
    for group in optimizer.param_groups:
        names = frozenset(index[id(parameter)] for parameter in group["params"])
        out[names] = float(group["lr"])
    return out


def test_value_group_lr_and_group_order() -> None:
    model = _TinyModel()
    optimizer = build_optimizer(model, lr=1e-3, primary_lr_scale=0.1, value_lr_scale=0.25)
    groups = _named_groups(optimizer, model)
    value_names = frozenset({"value.net.weight", "value.net.bias"})
    other_names = frozenset(
        {
            "encoders.ego.weight",
            "encoders.ego.bias",
            "policy.mu.weight",
            "policy.mu.bias",
            "plan_head.moe.experts.0.weight",
            "plan_head.moe.experts.0.bias",
        }
    )
    assert groups[value_names] == pytest.approx(2.5e-4), "value 组 lr = lr × value_lr_scale"
    assert groups[other_names] == pytest.approx(1e-3)
    assert [len(group["params"]) for group in optimizer.param_groups] == [6, 2], (
        "组序应为 [other, value, primary?]（本模型无 primary → 2 组；experts 属 other）"
    )


def test_default_scale_keeps_legacy_grouping_no_value_group() -> None:
    """默认 1.0：不建 value 组（与旧版逐位一致的 no-op 证据）。"""
    model = _TinyModel()
    legacy = build_optimizer(model, lr=1e-3, primary_lr_scale=0.1)
    current = build_optimizer(model, lr=1e-3, primary_lr_scale=0.1, value_lr_scale=1.0)
    legacy_signature = [
        (float(group["lr"]), [id(parameter) for parameter in group["params"]])
        for group in legacy.param_groups
    ]
    current_signature = [
        (float(group["lr"]), [id(parameter) for parameter in group["params"]])
        for group in current.param_groups
    ]
    assert current_signature == legacy_signature, "value_lr_scale=1.0 必须与旧分组逐位一致"
    assert len(current.param_groups) == 1, "默认不单独建 value 组（无 primary → 仅 other 组）"


def test_trainable_param_groups_value_lr_matches_optimizer() -> None:
    model = _TinyModel()
    groups = trainable_param_groups(model, lr=1e-3, primary_lr_scale=0.1, value_lr_scale=0.5)
    by_prefix = {group["prefix"]: group for group in groups}
    assert by_prefix["value."]["lr"] == pytest.approx(5e-4)
    assert by_prefix["policy."]["lr"] == pytest.approx(1e-3)
    assert by_prefix["plan_head.moe.experts."]["lr"] == pytest.approx(1e-3)
    assert by_prefix["encoders."]["lr"] == pytest.approx(1e-3)
    # 默认 1.0：value. 组 lr 不缩放（旧口径）
    legacy = {group["prefix"]: group for group in trainable_param_groups(model, lr=1e-3, primary_lr_scale=0.1)}
    assert legacy["value."]["lr"] == pytest.approx(1e-3)


def test_ppo_config_value_lr_scale_validation() -> None:
    assert PPOConfig().value_lr_scale == 1.0
    assert PPOConfig(value_lr_scale=0.25).value_lr_scale == 0.25
    assert PPOConfig(value_lr_scale=0.0).value_lr_scale == 0.0
    with pytest.raises(ValueError):
        PPOConfig(value_lr_scale=float("nan"))
    with pytest.raises(ValueError):
        PPOConfig(value_lr_scale=float("inf"))
    with pytest.raises(ValueError):
        PPOConfig(value_lr_scale=-0.1)


def test_stage_c_value_lr_scale_resolution_cli_over_config() -> None:
    args = _parse_args(["--stage", "C"])
    assert args.value_lr_scale is None
    assert _resolve_stage_c_value_lr_scale(args, {}) == 1.0, "默认 1.0（现状）"
    assert _resolve_stage_c_value_lr_scale(args, {"value_lr_scale": 0.25}) == 0.25, "config 生效"
    args_cli = _parse_args(["--stage", "C", "--value-lr-scale", "0.5"])
    assert args_cli.value_lr_scale == 0.5
    assert _resolve_stage_c_value_lr_scale(args_cli, {"value_lr_scale": 0.25}) == 0.5, "CLI 优先"
    with pytest.raises(SystemExit):
        _resolve_stage_c_value_lr_scale(args, {"value_lr_scale": "abc"})
    with pytest.raises(SystemExit):
        _resolve_stage_c_value_lr_scale(args, {"value_lr_scale": -1.0})
