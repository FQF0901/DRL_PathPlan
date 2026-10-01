"""v6 net 验收：交叉注意力头（A1/A2）、t0 单次 st_gnn 主路径（A3）、nav 逐步重建（A4）、
Stage C allowlist 命名契约（§5 #7）。

规格：``docs/v6_net_design.md``（冻结，@031cc1c）§1/§2/§3/§5 #7。纯 CPU、固定种子。
"""

from __future__ import annotations

import pytest
import torch

import net.model as net_model_module
from net.encoders import NAV_DIM
from net.mem import mem_from_obs
from net.model import DrivingModel, arc_step, compose_pose
from net.policy import CrossAttnHead, PolicyHead, ValueHead
from net.st_gnn import SpatioTemporalGNN
from pipeline.stages import _load_yaml, build_model
from pipeline.trainer import STAGE_C_DESIGN_PREFIXES, apply_trainable_allowlist
from tests.test_net_shapes import clone_obs, make_obs

H = 128
#: 冻结规格的精确参数算式（docs/v6_net_design.md §1.2；默认 1 层、d=128）
POLICY_PARAMS = 83_716  # ≈83.7k
VALUE_PARAMS = 83_329  # ≈83.3k
PARAM_RANGE = (60_000, 100_000)
TOKEN_LENGTH = 37  # OD16 + LD16 + others1 + ego1 + nav1 + signal1 + latent1


def _tokens(batch: int = 4, hidden: int = H) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(batch, TOKEN_LENGTH, hidden), torch.ones(batch, TOKEN_LENGTH)


# ---------------------------------------------------------------- 形状/参数量（config 口径）
def test_config_model_hidden_and_expert_hidden_are_frozen_values() -> None:
    """H=128 落点 = config/model.yaml（不得用 net/encoders.py 默认 96）；expert_hidden 维持 256。"""
    config = _load_yaml("config/model.yaml")
    model = build_model(config)
    assert int(config["hidden_dim"]) == H
    assert int(config["moe"]["experts"]["hidden_dim"]) == 256
    assert model.hidden == H
    assert model.plan_head.moe.experts[0][0].out_features == 256
    assert model.plan_head.moe.num_experts == 8 and model.plan_head.moe.top_k == 2
    # 新头按 H=128 构造（4 头 → head_dim=32）
    assert model.policy.hidden == H and model.policy.num_heads == 4 and model.policy.num_layers == 1
    assert sum(p.numel() for p in model.parameters()) <= 1_500_000


def test_attn_head_param_budget_matches_frozen_formula() -> None:
    """policy ≈83.7k / value ≈83.3k（[60k,100k]）——按 config/ckpt 口径（H=128）。"""
    model = build_model(_load_yaml("config/model.yaml"))
    policy_params = sum(p.numel() for p in model.policy.parameters())
    value_params = sum(p.numel() for p in model.value.parameters())
    assert policy_params == POLICY_PARAMS, f"policy 参数量 {policy_params} != 规格 {POLICY_PARAMS}"
    assert value_params == VALUE_PARAMS, f"value 参数量 {value_params} != 规格 {VALUE_PARAMS}"
    for label, count in (("policy", policy_params), ("value", value_params)):
        assert PARAM_RANGE[0] <= count <= PARAM_RANGE[1], f"{label} 参数量 {count} 不在 {PARAM_RANGE}"


@pytest.mark.parametrize("batch", [1, 8])
def test_attn_head_shapes_masks_and_stability(batch: int) -> None:
    """B=1/8：输出形状/dtype 与现行一致；全无效 mask 行严格 0（不 NaN）；log_prob 有限。"""
    policy = PolicyHead(hidden=H)
    value = ValueHead(hidden=H)
    tokens, key_mask = _tokens(batch)

    mu, log_std = policy(tokens, key_mask)
    v = value(tokens, key_mask)
    assert tuple(mu.shape) == (batch, 2) and mu.dtype == torch.float32
    assert tuple(log_std.shape) == (batch, 2) and log_std.dtype == torch.float32
    assert tuple(v.shape) == (batch, 1) and v.dtype == torch.float32
    assert torch.isfinite(mu).all() and torch.isfinite(log_std).all() and torch.isfinite(v).all()

    # 头级全无效 mask：注意力输出严格 0（数值安全：无 softmax 0/0）
    head = CrossAttnHead(hidden=H)
    zero_mask = torch.zeros(batch, TOKEN_LENGTH)
    out = head(tokens, zero_mask)
    assert torch.equal(out, torch.zeros_like(out)), "全无效行头输出必须为 0"
    mu0, log_std0 = policy(tokens, zero_mask)
    assert torch.isfinite(mu0).all() and torch.isfinite(log_std0).all()

    # log_prob 数值稳定（采样 + 记分，无 NaN/Inf）
    sample = policy.sample(tokens, key_mask, generator=torch.Generator().manual_seed(2))
    log_prob = policy.log_prob(tokens, key_mask, sample)
    assert tuple(log_prob.shape) == (batch, ) and torch.isfinite(log_prob).all()


def test_single_learned_query_and_layer_configuration() -> None:
    """K=1（Gate0 裁定：K>1 无监督信号 → 死参数）；层数 1/2 可配。"""
    policy = PolicyHead(hidden=H)
    assert tuple(policy.query.shape) == (1, H), "查询必须是 K=1 学习查询"
    assert len(policy.layers) == 1

    two_layer = PolicyHead(hidden=H, layers=2)
    assert len(two_layer.layers) == 2
    tokens, key_mask = _tokens(batch=2)
    mu, log_std = two_layer(tokens, key_mask)
    assert tuple(mu.shape) == (2, 2) and tuple(log_std.shape) == (2, 2)
    # 参数预算 [60k,100k] 只约束默认 1 层（每层 cross-attn 投影 +66k，规格明示"默认 1 层"）
    assert sum(p.numel() for p in policy.parameters()) == POLICY_PARAMS


def test_attn_heads_have_no_dead_parameters() -> None:
    """单次 backward 后头内全部参数有非零梯度（K=1 查询 + 全链无死参数）。"""
    policy = PolicyHead(hidden=H)
    value = ValueHead(hidden=H)
    tokens, key_mask = _tokens()
    mu, log_std = policy(tokens, key_mask)
    loss = mu.sum() + log_std.sum() + value(tokens, key_mask).sum()
    loss.backward()
    for label, module in (("policy", policy), ("value", value)):
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, f"{label}.{name} 无梯度（死参数）"
            assert float(parameter.grad.abs().sum()) > 0.0, f"{label}.{name} 梯度为零（死参数）"


def test_key_mask_gates_tokens() -> None:
    """key_mask=0 的令牌（任意垃圾值）不得影响头输出。"""
    policy = PolicyHead(hidden=32)
    tokens, key_mask = _tokens(batch=2, hidden=32)
    key_mask[:, 1] = 0.0
    baseline = policy(tokens, key_mask)[0]
    corrupted = tokens.clone()
    corrupted[:, 1] = 100.0
    assert torch.allclose(baseline, policy(corrupted, key_mask)[0], atol=0.0)


# ---------------------------------------------------------------- A3：t0 单次 st_gnn 主路径
def test_encode_runs_single_t0_st_gnn_message_passing() -> None:
    """encode() 对 t0 帧跑单次 st_gnn 消息传递；令牌集合 T=37 且 ego/latent 恒有效。"""
    model = DrivingModel(hidden=16).eval()
    counter = {"n": 0}
    handle = model.st_gnn.spatial.register_forward_hook(
        lambda *_: counter.__setitem__("n", counter["n"] + 1)
    )
    encoded = model.encode(make_obs(batch=2))
    handle.remove()

    assert counter["n"] == 1, "encode 必须恰好执行一次 st_gnn 消息传递（A3）"
    assert tuple(encoded["tokens"].shape) == (2, TOKEN_LENGTH, 16)
    assert tuple(encoded["key_mask"].shape) == (2, TOKEN_LENGTH)
    assert encoded["key_mask"][:, 33].all(), "ego_ctx token 恒有效"
    assert encoded["key_mask"][:, 36].all(), "plan_head 融合 latent token 恒有效"
    # §5 #4：nav/signal mask 随 token 一起返回
    assert tuple(encoded["nav_mask"].shape) == (2, 1) and tuple(encoded["signal_mask"].shape) == (2, 1)


# ---------------------------------------------------------------- A3：st_gnn 公共委托
def test_st_gnn_node_features_matches_spatial_reference() -> None:
    """``node_features`` = forward 解码器前的节点输出（同一 spatial/step_embed 实现）。"""
    torch.manual_seed(0)
    st_gnn = SpatioTemporalGNN(hidden=16, od_slots=16, ld_slots=16)
    ego = torch.randn(2, 16)
    od = torch.randn(2, 16, 16)
    ld = torch.randn(2, 16, 16)
    node_mask = torch.ones(2, 33)
    pose = torch.zeros(2, 33, 3)
    h_od, h_ld = st_gnn.node_features(
        ego_ctx=ego, od_ctx=od, ld_ctx=ld, node_mask=node_mask, pose=pose
    )
    assert tuple(h_od.shape) == (2, 16, 16) and tuple(h_ld.shape) == (2, 16, 16)
    # 参考：历史 _t0_object_features 的内联实现（step_index=1 → step_embed(0)）
    index = torch.zeros(2, dtype=torch.long)
    nodes = torch.cat([ego.unsqueeze(1) + st_gnn.step_embed(index).unsqueeze(1), od, ld], dim=1)
    nodes = st_gnn.spatial(nodes, node_mask, pose)
    assert torch.equal(h_od, nodes[:, 1:17]) and torch.equal(h_ld, nodes[:, 17:])
    # step_index 决定步嵌入（forward 每步调用同一委托）
    h_od3, _ = st_gnn.node_features(
        ego_ctx=ego, od_ctx=od, ld_ctx=ld, node_mask=node_mask, pose=pose, step_index=3
    )
    assert not torch.equal(h_od, h_od3)


def test_t0_object_features_delegates_to_public_node_features(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_t0_object_features`` 走 ``st_gnn.node_features``（不再直连 spatial/step_embed）。"""
    model = DrivingModel(hidden=16).eval()
    encoded = model.encode(make_obs(batch=2))["encoded"]
    calls: list[int] = []
    real = model.st_gnn.node_features

    def spy(**kwargs: object):
        calls.append(int(kwargs.get("step_index", -1)))  # type: ignore[arg-type]
        return real(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model.st_gnn, "node_features", spy)
    od_obj, ld_obj = model._t0_object_features(encoded)
    assert calls == [1], "t0 单次 pass 必须以 step_index=1 经公共委托"
    assert tuple(od_obj.shape) == (2, 16, 16) and tuple(ld_obj.shape) == (2, 16, 16)


# ---------------------------------------------------------------- A4：nav 逐步重建
def _obs_with_world(batch: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    obs = make_obs(batch=batch, seed=seed)
    # 世界系直道折线（非退化 → 真实重建，不触发 helper 的 fallback 告警）
    route = torch.zeros(batch, 8, 2)
    route[:, :, 0] = torch.arange(8, dtype=torch.float32) * 5.0
    obs["route_world"] = route
    obs["route_world_mask"] = torch.ones(batch, 8)
    obs["ego_world"] = torch.tensor([[10.0, 5.0, 0.3], [20.0, -3.0, -0.2]], dtype=torch.float32)[:batch]
    return obs


def test_rollout_rebuilds_nav_every_step_from_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """有 route_world/ego_world 时：rollout 每步重建 nav（6 次），位姿逐步推进，mask 透传。"""
    model = DrivingModel(hidden=16).eval()
    obs = _obs_with_world(batch=2)
    seen: list[torch.Tensor] = []
    seen_masks: list[object] = []
    real = net_model_module.rebuild_nav_from_world

    def spy(mem: object, ego_pose_world: torch.Tensor, route_world: object, mask: object = None):
        seen.append(ego_pose_world.detach().clone())
        seen_masks.append(mask.detach().clone() if torch.is_tensor(mask) else mask)
        return real(mem, ego_pose_world, route_world, mask)

    monkeypatch.setattr(net_model_module, "rebuild_nav_from_world", spy)
    model(obs)
    assert len(seen) == 6, f"rollout 应逐步重建 nav（6 步），实际 {len(seen)}"
    assert all(torch.isfinite(pose).all() for pose in seen)
    assert not torch.allclose(seen[0], seen[-1], atol=1e-6), "ego 世界位姿必须逐步推进"
    # route_world_mask 必须逐步原值到达重建 helper（不得在 model 内被丢弃）
    assert all(torch.is_tensor(mask) for mask in seen_masks)
    assert all(torch.equal(mask, obs["route_world_mask"]) for mask in seen_masks)  # type: ignore[arg-type]


def test_encode_canonicalizes_and_passes_route_world_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """route_world_mask 透传契约：encode 规范为 ``(B,M)``（容忍 ``(B,1,M)``），原值到达重建。"""
    model = DrivingModel(hidden=16).eval()
    obs = _obs_with_world(batch=2)
    mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    )
    obs["route_world_mask"] = mask.unsqueeze(1)  # (B,1,M) 单例槽位维
    seen: list[torch.Tensor] = []
    real = net_model_module._mem_nav_features_from_world

    def spy(pose: torch.Tensor, route: torch.Tensor, route_mask: object = None):
        seen.append(route_mask.detach().clone())  # type: ignore[union-attr]
        return real(pose, route, route_mask)

    monkeypatch.setattr(net_model_module, "_mem_nav_features_from_world", spy)
    encoded = model.encode(obs)
    assert tuple(encoded["route_world_mask"].shape) == (2, 8), "encode 必须把 mask 规范为 (B,M)"
    assert torch.equal(encoded["route_world_mask"], mask)
    model(obs)  # rollout 重建必须收到同一 mask
    assert seen and all(torch.equal(item, mask) for item in seen)
    # 非法形状 fail-fast（而不是静默丢弃 mask）
    with pytest.raises(ValueError, match="route_world_mask"):
        model.encode({**obs, "route_world_mask": torch.ones(3, 8)})


def test_rollout_without_world_inputs_keeps_t0_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """无世界系键（旧数据集/旧测试）→ 不重建，回退 t0 冻结上下文（逐位兼容）。"""
    model = DrivingModel(hidden=16).eval()
    calls: list[int] = []
    monkeypatch.setattr(
        net_model_module, "rebuild_nav_from_world", lambda *args, **kwargs: calls.append(1)
    )
    model(make_obs(batch=2))
    assert calls == [], "缺少 route_world/ego_world 时不得触发 nav 重建"


def test_step_context_uses_world_pose_and_syncs_others_nav(monkeypatch: pytest.MonkeyPatch) -> None:
    """_step_context：世界位姿直传 helper；重建结果重编码为 token 且写回 others 的 nav 维。"""
    model = DrivingModel(hidden=16).eval()
    mem = mem_from_obs(make_obs(batch=2), others_dim=28, history_frames=6)
    pose_world = torch.tensor([[13.0, 5.0, 0.3], [20.0, -3.0, -0.2]])
    world = {
        "route_world": torch.zeros(2, 8, 2),
        "route_world_mask": torch.ones(2, 8),
        "ego_world": pose_world,
    }
    seen: list[torch.Tensor] = []

    def spy(mem_arg: object, ego_pose_world: torch.Tensor, route_world: object, mask: object = None):
        seen.append(ego_pose_world.detach().clone())
        return torch.zeros(2, NAV_DIM), torch.ones(2, 1)

    monkeypatch.setattr(net_model_module, "rebuild_nav_from_world", spy)
    nav_token, nav_mask, signal_token, signal_mask = model._step_context(
        mem,
        pose_world,
        world,
        torch.zeros(2, 16),
        torch.zeros(2, 1),
        torch.zeros(2, 16),
        torch.zeros(2, 1),
        1,
    )
    assert torch.allclose(seen[0], pose_world, atol=1e-6), "世界位姿必须原样传给重建 helper"
    assert torch.allclose(mem.others[:, -1, :NAV_DIM], torch.zeros(2, NAV_DIM)), "others nav 维未同步重建"
    assert tuple(nav_token.shape) == (2, 16) and tuple(nav_mask.shape) == (2, 1)
    assert tuple(signal_token.shape) == (2, 16) and tuple(signal_mask.shape) == (2, 1)


def test_advance_pose_world_matches_compose_pose() -> None:
    """世界位姿推进与 ``compose_pose(arc_step)`` 等价（rollout 与教师强制同一步进语义）。"""
    pose_world = torch.tensor([[1.0, 2.0, 0.3], [-4.0, 0.5, -1.2]])
    ds = torch.tensor([3.0, 1.5])
    dtheta = torch.tensor([0.2, -0.4])
    advanced = net_model_module.advance_pose_world(pose_world, ds, dtheta)
    dx, dy = arc_step(ds, dtheta)
    expected = compose_pose(pose_world, dx, dy, dtheta)
    assert torch.allclose(advanced, expected, atol=1e-6)


def test_rollout_with_world_inputs_supports_backward() -> None:
    """A4 回归：带世界系键的 rollout 反向必须可用（nav 重建不得原地写 autograd 图内张量）。"""
    torch.manual_seed(0)
    model = DrivingModel(hidden=16, expert_hidden=16)
    with torch.no_grad():
        model.policy.mu.weight.normal_(0.0, 0.01)
    obs = _obs_with_world(batch=2)
    out = model(obs)
    (out["traj_xy"].pow(2).mean() + out["action_mu"].pow(2).mean()).backward()
    encoder_grads = [p.grad for p in model.encoders.parameters()]
    assert any(g is not None and float(g.abs().sum()) > 0.0 for g in encoder_grads), "编码器应收到梯度"


def test_nav_rebuild_result_is_actually_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """重建出的 nav 必须真正进入 rollout（改重建值 → 轨迹/计划改变），不是"接了不用"。"""
    torch.manual_seed(0)
    model = DrivingModel(hidden=16).eval()
    with torch.no_grad():  # mu 零初始化会挡住上游 → 给非零权重让动作链对令牌敏感
        model.policy.mu.weight.normal_(0.0, 0.05)
    obs = _obs_with_world(batch=2)

    baseline = model(obs)

    def shifted(mem: object, ego_pose_world: torch.Tensor, route_world: object, mask: object = None):
        nav = mem.others[:, -1, :NAV_DIM].clone()
        nav = nav + 1.0
        return nav, torch.ones(2, 1)

    monkeypatch.setattr(net_model_module, "rebuild_nav_from_world", shifted)
    changed = model(obs)
    assert not torch.allclose(baseline["traj_xy"], changed["traj_xy"], atol=1e-6)
    assert not torch.allclose(baseline["plan"], changed["plan"], atol=1e-6)


# ---------------------------------------------------------------- 信息充分性（toy）
def test_policy_distinguishes_left_right_cutin_tokens() -> None:
    """去池化 smoke：同 ego/地图，cut-in 目标在左/右（仅 OD 槽 0 位置不同）→ policy 输出不同。

    正式信息充分性对照（样本数/阈值预注册）属 Gate1 证据；本测试只锁定"逐槽位置信息
    不被抹平"这一去池化动机（池化版对左右镜像不可区分）。
    """
    torch.manual_seed(0)
    model = DrivingModel(hidden=128, expert_hidden=256).eval()
    with torch.no_grad():
        model.policy.mu.weight.normal_(0.0, 0.1)
        model.policy.mu.bias.zero_()

    def _with_target(side: float) -> dict[str, torch.Tensor]:
        obs = clone_obs(make_obs(batch=1, seed=3))
        obs["od_hist_mask"][:, -1, 0] = 1.0
        obs["od_presence_hist"][:, -1, 0] = 1.0
        obs["od_mask"][:, 0] = 1.0
        obs["od_presence"][:, 0] = 1.0
        obs["od_id_hist"][:, -1, 0] = 7
        obs["od_id"][:, 0] = 7
        obs["od_hist"][:, -1, 0, :2] = torch.tensor([8.0, side])
        obs["od"][:, 0, :2] = torch.tensor([8.0, side])
        return obs

    left = model(_with_target(3.5), rollout=False, world_model=False)["action_mu"]
    right = model(_with_target(-3.5), rollout=False, world_model=False)["action_mu"]
    assert float((left - right).abs().max()) > 1e-4, "左/右 cut-in 目标不可区分（去池化失效？）"


# ---------------------------------------------------------------- Stage C allowlist 命名契约
def test_stage_c_allowlist_keeps_attn_heads_trainable() -> None:
    """§5 #7：新头保 ``policy.``/``value.`` 前缀 → allowlist 冻结后头内全部参数可训，其余全冻结。"""
    model = build_model(_load_yaml("config/model.yaml"))
    frozen = apply_trainable_allowlist(model, STAGE_C_DESIGN_PREFIXES)

    head_names = [name for name, _ in model.named_parameters() if name.startswith(("policy.", "value."))]
    assert head_names, "新头参数必须挂在 policy./value. 前缀下（否则被静默冻结）"
    assert all(model.get_parameter(name).requires_grad for name in head_names)
    assert not any(name in frozen for name in head_names)

    for name, parameter in model.named_parameters():
        if name.startswith(
            ("policy.", "value.", "plan_head.moe.experts.", "plan_head.moe.residual_scale")
        ):
            continue
        assert not parameter.requires_grad, f"{name} 应被 Stage C 冻结"

    trainable_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    extras = sum(
        p.numel()
        for name, p in model.named_parameters()
        if name.startswith(("plan_head.moe.experts.", "plan_head.moe.residual_scale"))
    )
    assert trainable_total == POLICY_PARAMS + VALUE_PARAMS + extras, "allowlist 下头参数量不完整"
