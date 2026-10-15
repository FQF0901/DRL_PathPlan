"""R21-3A 语义接线契约测试（round-23 §6A 十项 + 归一化逐位 + 改名一致性）。

契约（要点见 ``docs/repo_handbook.md`` 附录 D）：

1. semantic invalid（``lane_semantic_mask=0 ∨ target_lane_valid=0``）→ ``target_lane_token`` 严格 0；
2. 入口固定归一化逐位（``env.obs.lane_semantic.LANE_SEMANTIC_NORMALIZATION`` 单一出处；
   编码器 buffer / schema layout / 手算除值三者逐位一致）；
3. role/group permutation 一致性（槽序置换不改变 context）；
4. target-role 改变 ⇒ target pool/context 变；
5. PlanHead t0 与 rollout 每一步都消费 semantic context（扰动 lane_semantic ⇒ 6 步 plan 全变）；
6. policy/value 消费一致（同一令牌集合）；
7. 旧 schema（缺键）fail-fast；旧 ckpt fail-fast（strict 报错 / missing ⊆ semantic 白名单）；
8. 无 hidden mean fallback（semantic 模式 plan_step 缺 context 报错；forward 不调用 aggregate_ld）；
9. 全灭（LD 全无效 + semantic invalid）forward/backward 有限；
10. route/ego 切换仅契约允许处（token 只由 lane_semantic/mask/valid 决定；role 变化不动 token）；
+ 改名一致性（distance_to_target_lane_end / same-direction road-surface margin）与
   ``ld_group`` 仅 equality（group 数值重标定不改 context）。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from env.obs.lane_semantic import (
    LANE_SEMANTIC_DIM,
    LANE_SEMANTIC_FEATURE_NAMES,
    LANE_SEMANTIC_NORMALIZATION,
    normalize_lane_semantic,
)
from env.obs.schema import LANE_SEMANTIC_LAYOUT
from net.model import DrivingModel
from net.semantic import LaneSemanticEncoder, SemanticWiring
from tests.test_round22_m2ab import NUM_LD, _obs as _base_obs
from tests.test_recovery_wiring import _tiny_model

H = 16


def _semantic_obs(batch: int = 2, seed: int = 0, *, ld_valid: int = NUM_LD, sem_valid: int = 2):
    """round-22 obs + schema v7 语义键（lane_semantic + LD companions）。"""
    obs = _base_obs(batch=batch, seed=seed, ld_valid=ld_valid)
    torch.manual_seed(seed + 100)
    lane_semantic = torch.randn(batch, 1, LANE_SEMANTIC_DIM)
    lane_semantic[..., 6] = 1.0  # target_lane_valid
    lane_semantic[..., 7] = 0.8  # route confidence
    lane_semantic[..., 8:10] = 0.0
    mask = torch.zeros(batch, 1)
    mask[: int(sem_valid)] = 1.0
    role = torch.zeros(batch, NUM_LD, dtype=torch.long)
    role[:, :5] = 1  # target（5 primary 槽）
    role[:, 5:7] = 2  # continuation
    role[:, 7] = 3  # left
    role[:, 8] = 4  # right
    obs.update(
        {
            "lane_semantic": lane_semantic,
            "lane_semantic_mask": mask,
            "ld_lane_role": role,
            "ld_offset_s_norm": torch.rand(batch, NUM_LD),
            "ld_route_associated": torch.ones(batch, NUM_LD),
            "ld_group": torch.zeros(batch, NUM_LD, dtype=torch.long),
        }
    )
    return obs


def _semantic_model() -> DrivingModel:
    model = _tiny_model()
    # net/policy.py 的 mu 末层零初始化（初始动作 = 界中点，对任何输入不敏感）——
    # 动作级消费断言前先打散（测试内局部扰动，不改生产代码；round-22 同口径）。
    torch.nn.init.normal_(model.policy.mu.weight, std=0.1)
    torch.nn.init.normal_(model.policy.mu.bias, std=0.05)
    model.set_ld_mode("semantic")
    return model


def _forward(model, obs, **kwargs):
    return model(obs, rollout=True, world_model=False, **kwargs)


def _activate_residual(model, *, context: float = 1.0, token: float = 1.0) -> None:
    """R21-25A：β=0 为 baseline 逐位；消费/一致性断言前把 β 置非零（残差路径存在）。"""
    with torch.no_grad():
        model.beta_context.fill_(float(context))
        model.beta_token.fill_(float(token))


# --------------------------------------------------------------------------- #
# 1) invalid → token 严格 0
# --------------------------------------------------------------------------- #
def test_semantic_invalid_token_strict_zero() -> None:
    model = _semantic_model()
    _activate_residual(model, context=0.0, token=1.0)
    obs = _semantic_obs(batch=2, sem_valid=1)
    out = _forward(model, obs)
    token = out["target_lane_token"]
    assert token.shape == (2, H)
    assert torch.equal(token[1], torch.zeros_like(token[1])), "mask=0 行 token 必须严格 0"
    assert float(token[0].abs().sum()) > 0.0
    # 内部 valid=0（mask=1 但 target_lane_valid=0）→ 严格 0
    obs2 = _semantic_obs(batch=2, sem_valid=2)
    obs2["lane_semantic"][1, 0, 6] = 0.0
    out2 = _forward(model, obs2)
    assert torch.equal(out2["target_lane_token"][1], torch.zeros_like(out2["target_lane_token"][1]))


def test_encoder_token_zero_exact_under_invalid() -> None:
    torch.manual_seed(0)
    encoder = LaneSemanticEncoder(H)
    feats = torch.randn(3, LANE_SEMANTIC_DIM)
    valid = torch.tensor([[1.0], [0.0], [1.0]])
    token = encoder(feats, valid)
    assert torch.equal(token[1], torch.zeros(H))
    assert not torch.equal(token[0], torch.zeros(H))


# --------------------------------------------------------------------------- #
# 2) 归一化逐位
# --------------------------------------------------------------------------- #
def test_normalization_constants_single_source() -> None:
    assert len(LANE_SEMANTIC_NORMALIZATION) == LANE_SEMANTIC_DIM == 12
    expected = (5.0, math.pi, 0.1, 0.1, 25.0, 25.0, 1.0, 1.0, 1.0, 1.0, 200.0, 200.0)
    assert LANE_SEMANTIC_NORMALIZATION == expected
    layout = LANE_SEMANTIC_LAYOUT["normalization"]
    for name, divisor in zip(LANE_SEMANTIC_FEATURE_NAMES, expected):
        assert layout[name] == divisor
    # 编码器 buffer 与常量同源（float32 精度下逐位）
    encoder = LaneSemanticEncoder(H)
    expected_t = torch.tensor(expected, dtype=torch.float32)
    assert torch.equal(encoder.normalization, expected_t)


def test_normalize_lane_semantic_bitwise() -> None:
    feats = np.array([1.0, math.pi, 0.05, -0.05, 12.5, 25.0, 1.0, 0.25, 1.0, 0.0, 100.0, 200.0])
    out = normalize_lane_semantic(feats)
    manual = feats / np.asarray(LANE_SEMANTIC_NORMALIZATION)
    np.testing.assert_array_equal(out, manual)
    # 编码器前向 = MLP(feats/constants)（逐位）
    torch.manual_seed(0)
    encoder = LaneSemanticEncoder(H)
    t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    valid = torch.ones(1, 1)
    got = encoder(t, valid)
    expected_token = encoder.net(t / encoder.normalization) * valid
    assert torch.equal(got, expected_token)


# --------------------------------------------------------------------------- #
# 3) role/group permutation 一致性
# --------------------------------------------------------------------------- #
def test_role_group_permutation_consistency() -> None:
    model = _semantic_model()
    _activate_residual(model, context=1.0, token=0.0)
    obs = _semantic_obs(batch=1, seed=3)
    base = _forward(model, obs)
    perm = torch.randperm(NUM_LD)
    permuted = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in obs.items()}
    for key in ("ld", "ld_mask", "ld_lane_role", "ld_offset_s_norm", "ld_route_associated", "ld_group"):
        permuted[key] = obs[key][:, perm].clone()
    permuted["ld_hist"][:, -1] = obs["ld_hist"][:, -1][:, perm]
    permuted["ld_hist_mask"][:, -1] = obs["ld_hist_mask"][:, -1][:, perm]
    out = _forward(model, permuted)
    assert torch.allclose(base["geometry_context"], out["geometry_context"], atol=1e-6)
    assert torch.allclose(base["target_lane_token"], out["target_lane_token"], atol=1e-6)


def test_ld_group_only_equality() -> None:
    """group 数值重标定（保 equality 结构）不改 context——证明未作数值使用。"""
    model = _semantic_model()
    obs = _semantic_obs(batch=1, seed=4)
    base = _forward(model, obs)["geometry_context"]
    relabeled = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in obs.items()}
    groups = obs["ld_group"].clone()
    relabeled["ld_group"] = torch.where(groups == 0, torch.full_like(groups, 7), groups)
    out = _forward(model, relabeled)["geometry_context"]
    assert torch.allclose(base, out, atol=1e-6)


# --------------------------------------------------------------------------- #
# 4) target-role 改变 ⇒ target pool/context 变
# --------------------------------------------------------------------------- #
def test_target_role_change_changes_context() -> None:
    model = _semantic_model()
    _activate_residual(model, context=1.0, token=0.0)
    obs = _semantic_obs(batch=1, seed=5)
    base = _forward(model, obs)["geometry_context"]
    moved = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in obs.items()}
    role = obs["ld_lane_role"].clone()
    role[:, :5] = 0  # 原 target 变 other
    role[:, 10:15] = 1  # 新 target
    moved["ld_lane_role"] = role
    out = _forward(model, moved)["geometry_context"]
    assert not torch.allclose(base, out, atol=1e-6), "target-role 改变必须改变 context"
    # 直接检查 target pool：相同 context 输入下只改 target 槽 ⇒ 读出输入变化
    wiring = model.semantic_wiring
    assert wiring is not None
    z_ld = torch.randn(1, NUM_LD, H)
    live = torch.ones(1, NUM_LD)
    kwargs = dict(
        lane_semantic=obs["lane_semantic"][:, 0],
        lane_semantic_mask=obs["lane_semantic_mask"],
        offset_s=obs["ld_offset_s_norm"],
        route_assoc=obs["ld_route_associated"],
        group=obs["ld_group"],
    )
    ctx_a, _ = wiring(z_ld, live, role=obs["ld_lane_role"], **kwargs)
    ctx_b, _ = wiring(z_ld, live, role=role, **kwargs)
    assert not torch.allclose(ctx_a, ctx_b, atol=1e-6)


# --------------------------------------------------------------------------- #
# 5) PlanHead t0 + rollout 每步消费；6) policy/value 一致
# --------------------------------------------------------------------------- #
def test_planhead_t0_and_rollout_consume_semantic() -> None:
    model = _semantic_model()
    _activate_residual(model, context=1.0, token=1.0)
    obs = _semantic_obs(batch=2, seed=6)
    base = _forward(model, obs)
    perturbed = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in obs.items()}
    perturbed["lane_semantic"][0, 0, 0] += 1.0  # lateral offset +1（归一化前）
    out = _forward(model, perturbed)
    assert (out["latent"][0] - base["latent"][0]).abs().max() > 1e-6, "t0 PlanHead 未消费"
    plan_delta = (out["plan"] - base["plan"]).abs().amax(dim=(0, 2))
    assert plan_delta.shape == (6, )
    assert float(plan_delta.min()) > 1e-7, f"rollout 步未消费 semantic：{plan_delta.tolist()}"
    assert (out["action_mu"][0] - base["action_mu"][0]).abs().max() > 1e-8
    assert (out["value"][0] - base["value"][0]).abs().max() > 1e-8, "value 未消费"


# --------------------------------------------------------------------------- #
# 7) 旧 schema / 旧 ckpt fail-fast
# --------------------------------------------------------------------------- #
def test_missing_semantic_keys_fail_fast() -> None:
    model = _semantic_model()
    obs = _base_obs(batch=1)  # 无 semantic 键
    with pytest.raises(ValueError, match="semantic 模式缺少观测键"):
        _forward(model, obs)


def test_old_ckpt_fail_fast_whitelist() -> None:
    baseline = _tiny_model()
    state = baseline.state_dict()
    semantic = _semantic_model()
    with pytest.raises(RuntimeError):
        semantic.load_state_dict(state, strict=True)
    missing, unexpected = semantic.load_state_dict(state, strict=False)
    allowed = set(semantic.semantic_parameter_names())
    assert set(missing) <= allowed, f"旧 ckpt 缺非 semantic 键：{sorted(set(missing) - allowed)}"
    assert not unexpected
    # baseline 模式严格加载旧 ckpt 仍逐位可行（默认关契约）
    baseline.load_state_dict(state, strict=True)


# --------------------------------------------------------------------------- #
# 8) 无 hidden mean fallback
# --------------------------------------------------------------------------- #
def test_residual_base_is_legacy_and_plan_step_requires_context() -> None:
    """R21-25A 契约改写：semantic 模式**以 legacy aggregate_ld 为残差基底**（旧"禁 masked_mean"
    语义反转）；仍禁止"缺 context 的静默回退"。"""
    model = _semantic_model()
    obs = _semantic_obs(batch=1, seed=7)
    # semantic 模式正常前向会调用 aggregate_ld（legacy 基底）——不应抛错
    _forward(model, obs)
    with pytest.raises(RuntimeError, match="禁止静默回退"):
        model.plan_step(
            torch.randn(1, H), torch.randn(1, NUM_LD, H), torch.randn(1, NUM_LD, H),
            torch.ones(1, NUM_LD), torch.ones(1, NUM_LD), torch.randn(1, H),
            torch.randn(1, H), torch.randn(1, H),
        )


# --------------------------------------------------------------------------- #
# 9) 全灭 forward/backward 有限
# --------------------------------------------------------------------------- #
def test_all_invalid_forward_backward_finite() -> None:
    model = _semantic_model()
    obs = _semantic_obs(batch=2, seed=8, ld_valid=0, sem_valid=0)
    out = _forward(model, obs)
    for key in ("action_mu", "plan", "value", "geometry_context", "target_lane_token"):
        assert torch.isfinite(out[key]).all(), f"{key} 含非有限值"
    loss = out["action_mu"].sum() + out["plan"].sum() + out["geometry_context"].sum() + out["target_lane_token"].sum()
    loss.backward()
    grads = [p.grad for p in model.semantic_wiring.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------- #
# 10) route/ego 切换仅契约允许处
# --------------------------------------------------------------------------- #
def test_token_depends_only_on_lane_semantic_and_validity() -> None:
    """token 只由 lane_semantic（12 维）与 mask/valid 决定：role/route/offset/group 变化不动 token。"""
    model = _semantic_model()
    _activate_residual(model, context=0.0, token=1.0)
    obs = _semantic_obs(batch=1, seed=9)
    token_a = _forward(model, obs)["target_lane_token"]
    changed = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in obs.items()}
    changed["ld_lane_role"] = torch.full_like(obs["ld_lane_role"], 0)
    changed["ld_route_associated"] = 1.0 - obs["ld_route_associated"]
    changed["ld_offset_s_norm"] = torch.rand_like(obs["ld_offset_s_norm"])
    changed["ld_group"] = torch.full_like(obs["ld_group"], 3)
    token_b = _forward(model, changed)["target_lane_token"]
    assert torch.allclose(token_a, token_b, atol=1e-7)
    # 切换只发生在 valid 门：valid 0→1 / 1→0 精确切换零/非零
    obs_valid0 = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in obs.items()}
    obs_valid0["lane_semantic"][0, 0, 6] = 0.0
    assert torch.equal(_forward(model, obs_valid0)["target_lane_token"][0], torch.zeros(H))
    assert not torch.equal(token_a[0], torch.zeros(H))


# --------------------------------------------------------------------------- #
# + 改名一致性
# --------------------------------------------------------------------------- #
def test_round23_renames_consistent() -> None:
    names = LANE_SEMANTIC_FEATURE_NAMES
    assert "distance_to_target_lane_end" in names
    assert "distance_to_next_topology_change" not in names
    assert "left_same_direction_road_surface_margin" in names
    assert "right_same_direction_road_surface_margin" in names
    assert "left_drivable_margin" not in names and "right_drivable_margin" not in names
    renames = LANE_SEMANTIC_LAYOUT["renames_round23"]
    assert "distance_to_next_topology_change" in renames
    assert "left_drivable_margin" in renames and "right_drivable_margin" in renames
    assert "禁作监督真值" in renames["left_drivable_margin"]


def test_semantic_wiring_param_count_and_names() -> None:
    model = _semantic_model()
    names = model.semantic_parameter_names()
    assert names and all(
        name.startswith("semantic_wiring.") or name in ("beta_context", "beta_token")
        or name.startswith("target_proj.")
        for name in names
    )
    total = sum(p.numel() for p in model.semantic_wiring.parameters())
    assert total > 0
    # 归一化常数随 ckpt 持久化（state_dict 含 buffer）
    assert "semantic_wiring.lane_encoder.normalization" in model.state_dict()


def test_ld_augmentation_uses_role_route_offset_group() -> None:
    """四项增广都实际生效：逐一置零/改变 ⇒ context 变。"""
    torch.manual_seed(0)
    wiring = SemanticWiring(H)
    # offset 编码末层生产上零初始化（稳定起步）——本测试先打散以验证接线
    torch.nn.init.normal_(wiring.offset_encoder[-1].weight, std=0.1)
    torch.nn.init.normal_(wiring.offset_encoder[-1].bias, std=0.05)
    z_ld = torch.randn(1, NUM_LD, H)
    live = torch.ones(1, NUM_LD)
    base_kwargs = dict(
        lane_semantic=torch.randn(1, LANE_SEMANTIC_DIM),
        lane_semantic_mask=torch.ones(1, 1),
        role=torch.cat([torch.full((1, 8), 1), torch.zeros(1, 8, dtype=torch.long)], dim=1),
        offset_s=torch.rand(1, NUM_LD),
        route_assoc=torch.ones(1, NUM_LD),
        group=torch.zeros(1, NUM_LD, dtype=torch.long),
    )
    base, _ = wiring(z_ld, live, **base_kwargs)
    for key, value in (
        ("role", torch.zeros(1, NUM_LD, dtype=torch.long)),
        ("route_assoc", torch.zeros(1, NUM_LD)),
        ("offset_s", torch.zeros(1, NUM_LD)),
        # group：改 equality 结构（target=0 / 其余=1）——只改数值不改 equality 的变体不算生效
        ("group", torch.cat([torch.zeros(1, 8), torch.ones(1, 8)], dim=1).to(torch.long)),
    ):
        variant = dict(base_kwargs)
        variant[key] = value
        out, _ = wiring(z_ld, live, **variant)
        assert not torch.allclose(base, out, atol=1e-6), f"{key} 增广未生效"
