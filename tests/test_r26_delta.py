"""R21-26B 契约测试：`semantic_off` 基底前向 + delta 监督绑定语义路径。

1. `semantic_off=True` 输出 = baseline（legacy）路径逐位（β 任意）；
2. β=0 时 `delta_model = mu_on − mu_off = 0`（严格）；
3. `L_delta = SmoothL1(mu_on − mu_off.detach(), target)` 的梯度只落在语义路径
   （β_context/β_token/wiring/target_proj），不落非语义参数（基底冻结/detach 语义）；
4. β≠0 时 delta_model ≠ 0（残差路径存在）。
"""

from __future__ import annotations

import torch

from tests.test_r21_3a_semantic_wiring import _activate_residual, _semantic_model, _semantic_obs

OUT_KEYS = ("action_mu", "value", "plan", "traj_xy", "router_logits")


def _forward(model, obs, **kwargs):
    return model(obs, forward_mode="policy_wm_rollout", **kwargs)


def _max_diff(a, b) -> float:
    return float((a - b).abs().max())


def test_semantic_off_equals_baseline_bitwise() -> None:
    model = _semantic_model()
    obs = _semantic_obs(batch=2, seed=31)
    baseline = _forward(model, obs)  # semantic 模式 β=0
    _activate_residual(model, context=0.7, token=0.5)
    off = _forward(model, obs, semantic_off=True)
    for key in OUT_KEYS:
        assert _max_diff(baseline[key], off[key]) < 1e-6, f"semantic_off 应与 baseline 逐位（{key}）"
    on = _forward(model, obs)
    assert _max_diff(on["action_mu"], off["action_mu"]) > 1e-6, "β≠0 时 on/off 应有差异"


def test_delta_zero_at_beta_zero_and_nonzero_after() -> None:
    model = _semantic_model()
    obs = _semantic_obs(batch=2, seed=32)
    on = _forward(model, obs)["action_mu"]
    off = _forward(model, obs, semantic_off=True)["action_mu"]
    assert float((on - off).abs().max()) == 0.0, "β=0 ⇒ delta 严格 0"
    _activate_residual(model, context=0.5, token=0.5)
    delta = _forward(model, obs)["action_mu"] - _forward(model, obs, semantic_off=True)["action_mu"]
    assert float(delta.abs().max()) > 1e-6


def test_cheap_path_consistent_and_off_zero() -> None:
    """R21-26B：cheap path（rollout=False）的 on/off 差在 β=0 严格 0；β≠0 与 full 路径一致。"""
    model = _semantic_model()
    obs = _semantic_obs(batch=2, seed=34)
    with torch.no_grad():
        full_on = _forward(model, obs)["action_mu"]
        cheap_on = model(obs, forward_mode="policy_single_step")["action_mu"]
        cheap_off = model(obs, forward_mode="policy_single_step", semantic_off=True)["action_mu"]
        assert float((full_on - cheap_on).abs().max()) == 0.0, "cheap 与 full 的 action_mu 应逐位一致"
        assert float((cheap_on - cheap_off).abs().max()) == 0.0, "β=0 ⇒ cheap on/off 严格 0"
        _activate_residual(model, context=0.3, token=0.3)
        cheap_on2 = model(obs, forward_mode="policy_single_step")["action_mu"]
        cheap_off2 = model(obs, forward_mode="policy_single_step", semantic_off=True)["action_mu"]
        full_on2 = _forward(model, obs)["action_mu"]
        assert float((cheap_on2 - cheap_off2).abs().max()) > 1e-6
        assert float((full_on2 - cheap_on2).abs().max()) == 0.0


def test_delta_loss_gradient_binds_semantic_only() -> None:
    """基底冻结（requires_grad=False，= Stage 1 语义路径 only）时，L_delta 梯度只落语义路径。"""
    model = _semantic_model()
    _activate_residual(model, context=0.3, token=0.3)
    for param in model.parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if name == "beta_context" or name == "beta_token" or name.startswith("semantic_wiring.") or name.startswith("target_proj."):
            param.requires_grad_(True)
    obs = _semantic_obs(batch=2, seed=33)
    target = torch.zeros(2, 2)  # 任意 delta 目标
    mu_on = _forward(model, obs)["action_mu"]
    mu_off = _forward(model, obs, semantic_off=True)["action_mu"].detach()
    delta_model = mu_on - mu_off
    loss = torch.nn.functional.smooth_l1_loss(delta_model, target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert model.beta_context.grad is not None and float(model.beta_context.grad.abs()) > 0
    assert model.beta_token.grad is not None and float(model.beta_token.grad.abs()) > 0
    # 冻结的非语义参数在 L_delta 下无梯度（基底冻结/detach 语义）
    for name in ("policy.mu.weight", "policy.mu.bias", "plan_head.fusion.2.weight"):
        param = dict(model.named_parameters())[name]
        assert param.grad is None or float(param.grad.abs().max()) == 0.0, f"{name} 不应有 L_delta 梯度"
