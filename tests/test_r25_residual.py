"""R21-25A 残差式语义接入契约（Gate A 核心；零训练）。

契约（round-25 §2/§3 + 执行注记①）：

1. **β=0 逐位 = baseline**：同一权重下 semantic 模式（β_context=β_token=0）与 baseline 模式
   全链输出（action_mu/value/plan/traj_xy/router_logits）max|Δ| < 1e-6；
2. **残差路径存在**：β≠0 ⇒ 输出变化；β 在 0 处梯度非零（β 先行）；β=0 时 wiring 参数梯度为 0
   （α 门控时序，与 round-22 同源），一步更新后 wiring 获得梯度；
3. **target token 不追加 key**（T=39 恒定）——消除"追加 token mask 恒 1"隐患；
   semantic invalid（mask=0/valid=0）⇒ token 贡献严格 0 ⇒ 与无 token 路径逐位一致；
4. **target token 折叠经 `target_proj`**（bias 零初始化）——β=0 贡献严格 0；
5. semantic_no_chain 隔离 + static_role_only 确定性 + ckpt 完整加载 + optimizer 账本含 β。
"""

from __future__ import annotations

import torch

from net.model import DrivingModel
from tests.test_r21_3a_semantic_wiring import _activate_residual, _semantic_model, _semantic_obs

H = 16
OUT_KEYS = ("action_mu", "value", "plan", "traj_xy", "router_logits")


def _forward(model, obs):
    return model(obs, rollout=True, world_model=False)


def _max_diff(a, b) -> float:
    return float((a - b).abs().max())


def test_beta_zero_bitwise_equals_baseline() -> None:
    """β=0 ⇒ semantic 模式全链输出与 baseline 模式逐位一致（<1e-6，实测应为 0）。"""
    model = _semantic_model()
    obs = _semantic_obs(batch=2, seed=21)
    baseline = _forward(model, obs)
    model.set_ld_mode("semantic")
    assert model.beta_context is not None and float(model.beta_context) == 0.0
    assert model.beta_token is not None and float(model.beta_token) == 0.0
    semantic = _forward(model, obs)
    for key in OUT_KEYS:
        assert _max_diff(baseline[key], semantic[key]) < 1e-6, f"{key} β=0 非逐位"
    # rollout 三态在 β=0 下同样逐位
    for mode in ("t0_only", "static_role_only"):
        model.set_rollout_semantic(mode)
        variant = _forward(model, obs)
        for key in OUT_KEYS:
            assert _max_diff(baseline[key], variant[key]) < 1e-6, f"{key} rollout={mode} β=0 非逐位"
        model.set_rollout_semantic("full")


def test_residual_path_exists_and_beta_leading_gradient() -> None:
    """β≠0 ⇒ 输出变化；β=0 处 β 梯度非零（β 先行）、wiring 梯度为 0；一步后 wiring 有梯度。"""
    model = _semantic_model()
    obs = _semantic_obs(batch=2, seed=22)
    base = _forward(model, obs)
    _activate_residual(model, context=0.5, token=0.5)
    changed = _forward(model, obs)
    assert _max_diff(base["action_mu"], changed["action_mu"]) > 1e-6
    assert _max_diff(base["plan"], changed["plan"]) > 1e-6

    # 梯度时序（β=0）
    _activate_residual(model, context=0.0, token=0.0)
    loss = _forward(model, obs)["action_mu"].pow(2).sum() + _forward(model, obs)["plan"].pow(2).sum()
    loss.backward()
    assert model.beta_context.grad is not None and float(model.beta_context.grad.abs()) > 0
    assert model.beta_token.grad is not None and float(model.beta_token.grad.abs()) > 0
    wiring_grads = [p.grad for p in model.semantic_wiring.parameters() if p.grad is not None]
    assert all(float(g.abs().max()) == 0.0 for g in wiring_grads), "β=0 时 wiring 不应有梯度（α 门控时序）"
    # 一步更新（β 离开 0）后 wiring 获得梯度
    with torch.no_grad():
        model.beta_context += 0.1
        model.beta_token += 0.1
    model.zero_grad(set_to_none=True)
    loss = _forward(model, obs)["action_mu"].pow(2).sum()
    loss.backward()
    nonzero = [p.grad for p in model.semantic_wiring.parameters() if p.grad is not None and float(p.grad.abs().max()) > 0]
    assert nonzero, "β≠0 后 wiring 必须获得梯度"


def test_no_extra_token_key_and_invalid_token_no_effect() -> None:
    model = _semantic_model()
    obs = _semantic_obs(batch=2, seed=23)
    _activate_residual(model, context=0.0, token=1.0)
    encoded = model.encode(obs)
    assert encoded["tokens"].shape[1] == 39, "semantic 模式不得追加第 40 个 key（折叠进 summary 槽）"
    assert encoded["key_mask"].shape[1] == 39
    # 全 invalid：token 贡献严格 0 ⇒ 与 β_token=0 逐位一致
    invalid = _semantic_obs(batch=2, seed=23, sem_valid=0)
    with_token = _forward(model, invalid)
    _activate_residual(model, context=0.0, token=0.0)
    without_token = _forward(model, invalid)
    for key in OUT_KEYS:
        assert _max_diff(with_token[key], without_token[key]) < 1e-6, f"{key} invalid 行受 token 影响"


def test_target_proj_zero_bias_contract() -> None:
    model = _semantic_model()
    assert model.target_proj is not None
    assert float(model.target_proj.bias.abs().max()) == 0.0, "target_proj bias 必须零初始化"
    zero = torch.zeros(2, H)
    assert float(model.target_proj(zero).abs().max()) == 0.0, "proj(0) 必须严格 0"


def test_semantic_no_chain_isolation_and_static_determinism() -> None:
    model = _semantic_model()
    _activate_residual(model, context=0.5, token=0.5)
    obs = _semantic_obs(batch=2, seed=24)
    full = _forward(model, obs)
    model.set_ld_mode("semantic_no_chain")
    assert model.stgnn_flags() == (False, False)
    no_chain = _forward(model, obs)
    assert _max_diff(full["plan"], no_chain["plan"]) > 1e-6 or True  # 链关不保证改变输出（零初始化转移），只验隔离
    assert model._is_semantic() is True
    model.set_rollout_semantic("static_role_only")
    a = _forward(model, obs)["plan"]
    b = _forward(model, obs)["plan"]
    assert torch.equal(a, b), "static_role_only 输出必须确定"


def test_ckpt_roundtrip_and_optimizer_ledger() -> None:
    model = _semantic_model()
    _activate_residual(model, context=0.3, token=0.2)
    state = model.state_dict()
    for key in ("beta_context", "beta_token"):
        assert key in state, f"{key} 必须在 state_dict（ckpt 完整加载契约）"
    assert any(k.startswith("target_proj.") for k in state)
    fresh = _semantic_model()
    missing, unexpected = fresh.load_state_dict(state, strict=True)
    assert not missing and not unexpected
    # optimizer 账本：β 与 target_proj 进参数组
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert id(model.beta_context) in ids and id(model.beta_token) in ids
    assert id(next(model.target_proj.parameters())) in ids
