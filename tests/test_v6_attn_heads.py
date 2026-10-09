"""v6/v8 net 验收：交叉注意力头（A1/A2）、A4 nav 逐步重建、Stage C allowlist 命名契约（§5 #7）。

规格：``docs/v6_net_design.md``（冻结，@031cc1c）§1/§2/§3/§5 #7；**v8 修订**（参数再分配 +
latent 重构）覆盖其中被取代的口径（t0 单次 MP 进头、参数预算、令牌组成）。纯 CPU、固定种子。
"""

from __future__ import annotations

import pytest
import torch

import net.model as net_model_module
from net.encoders import NAV_DIM
from net.mem import advance_pose_world, mem_from_obs, nav_features_from_world
from net.model import DrivingModel, arc_step, compose_pose
from net.policy import CrossAttnHead, PolicyHead, ValueHead
from net.st_gnn import SpatioTemporalGNN
from pipeline.stages import _load_yaml, build_model
from pipeline.trainer import STAGE_C_DESIGN_PREFIXES, apply_trainable_allowlist
from tests.test_net_shapes import clone_obs, make_obs

H = 128
#: v8 参数再分配后的精确参数算式（config/model.yaml 口径；见 tests/test_net_shapes.py）
POLICY_PARAMS = 108_452  # ≈108.5k
VALUE_PARAMS = 99_969  # ≈100.0k
PARAM_RANGE = (60_000, 120_000)
TOKEN_LENGTH = 39  # v8：z_od16 + z_ld16 + od_pool + ld_pool + others + z_ego + nav + signal + latent


def _tokens(batch: int = 4, hidden: int = H) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.randn(batch, TOKEN_LENGTH, hidden), torch.ones(batch, TOKEN_LENGTH)


# ---------------------------------------------------------------- 形状/参数量（config 口径）
def test_config_model_hidden_and_expert_hidden_are_frozen_values() -> None:
    """H=128 落点 = config/model.yaml（不得用 net/encoders.py 默认 96）；expert_hidden=76（v8）。"""
    config = _load_yaml("config/model.yaml")
    model = build_model(config)
    assert int(config["hidden_dim"]) == H
    assert int(config["moe"]["experts"]["hidden_dim"]) == 76
    assert model.hidden == H
    assert model.plan_head.moe.experts[0][0].out_features == 76
    assert model.plan_head.moe.num_experts == 8 and model.plan_head.moe.top_k == 2
    # 新头按 H=128 构造（4 头 → head_dim=32）
    assert model.policy.hidden == H and model.policy.num_heads == 4 and model.policy.num_layers == 1
    assert sum(p.numel() for p in model.parameters()) <= 1_500_000


def test_attn_head_param_budget_matches_frozen_formula() -> None:
    """policy ≈108.5k / value ≈100.0k（v8 参数再分配；config/ckpt 口径 H=128）。"""
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


# ---------------------------------------------------------------- v8：令牌集合（无 t0 MP）
def test_encode_builds_t39_tokens_without_t0_message_passing() -> None:
    """encode() 不跑 st_gnn 消息传递（v8 B3）；令牌集合 T=39 且池化/ego/nav/signal/latent 恒有效。"""
    model = DrivingModel(hidden=16).eval()
    counter = {"n": 0}
    handle = model.st_gnn.spatial.register_forward_hook(
        lambda *_: counter.__setitem__("n", counter["n"] + 1)
    )
    encoded = model.encode(make_obs(batch=2))
    handle.remove()

    assert counter["n"] == 0, "v8：encode 不得执行 st_gnn 消息传递（t0 单次 MP 进头已删除）"
    assert tuple(encoded["tokens"].shape) == (2, TOKEN_LENGTH, 16)
    assert tuple(encoded["key_mask"].shape) == (2, TOKEN_LENGTH)
    # [z_od16, z_ld16, od_pool, ld_pool, others, z_ego, nav, signal, latent]
    assert encoded["key_mask"][:, 32:].all(), "池化/others/z_ego/nav/signal/latent 恒有效"
    assert encoded["key_mask"][:, 38].all(), "plan_head 融合 latent token 恒有效"
    # §5 #4：nav/signal mask 随 token 一起返回
    assert tuple(encoded["nav_mask"].shape) == (2, 1) and tuple(encoded["signal_mask"].shape) == (2, 1)
    # latent 状态（v8）：当前帧编码
    assert tuple(encoded["z_ego"].shape) == (2, 16)
    assert tuple(encoded["z_od"].shape) == (2, 16, 16)
    assert tuple(encoded["z_ld"].shape) == (2, 16, 16)


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


def test_step_context_uses_world_pose_and_rebuilds_nav(monkeypatch: pytest.MonkeyPatch) -> None:
    """_step_context：世界位姿直传 helper；重建结果重编码为 nav token（v8：不再同步 others）。"""
    model = DrivingModel(hidden=16).eval()
    obs = make_obs(batch=2)
    mem = mem_from_obs(obs, others_dim=33, history_frames=6)
    others_before = mem.others.clone()
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
    # v8：others 上下文为 t0 静态（不再回写 nav 维；nav 由独立 token 承载）
    assert torch.equal(mem.others, others_before), "others 不得被逐步改写"
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


def test_rollout_switches_route_command_after_displacement(monkeypatch: pytest.MonkeyPatch) -> None:
    """A4 验收（docs/v6_net_design.md §3）：**model.rollout 级** route-command 切换单测。

    场景：路线顶点 (0,0)→(10,0)→(50,0)→(60,10)（分叉顶点 (10,0)；下一段 45° 左转）。
    ego 起点 (6.43,0,0)：t0 命令 = forward；默认动作 ds=5 m/步 ⇒ 第 1 步位移跨过 (10,0)，
    命令应切换为 left。断言（不只 helper 层）：

    1. ``rebuild_nav_from_world`` 在 rollout 内被调用 6 次，第 1 次的世界位姿 = 起点 +5 m；
    2. t0 特征命令 = forward、第 1 次重建特征命令 = left（位移触发切换）；
    3. 第 1 次重建出的 nav 特征 → token 后**被策略头实际消费**（policy 第 2 次调用的
       nav token 逐位 = ``embed_nav(重建特征)``），且与 t0 token 不同（命令切换生效）。
    """
    model = DrivingModel(hidden=16).eval()
    obs = make_obs(batch=1, seed=0)
    route = torch.zeros(1, 8, 2)
    route[0, :4] = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [50.0, 0.0], [60.0, 10.0]], dtype=torch.float32
    )
    route_mask = torch.zeros(1, 8)
    route_mask[0, :4] = 1.0
    obs["route_world"] = route
    obs["route_world_mask"] = route_mask
    obs["ego_world"] = torch.tensor([[6.43, 0.0, 0.0]], dtype=torch.float32)
    t0_feats, _ = nav_features_from_world(obs["ego_world"], route, route_mask)
    obs["nav"] = t0_feats  # 真实 env 口径：t0 nav 由世界系重建得到（forward 命令）
    obs["nav_mask"] = torch.ones(1, 1)

    poses: list[torch.Tensor] = []
    rebuilds: list[torch.Tensor] = []
    real_rebuild = net_model_module.rebuild_nav_from_world

    def rebuild_spy(mem: object, ego_pose_world: torch.Tensor, route_world: object, mask: object = None):
        poses.append(ego_pose_world.detach().clone())
        feats, nav_mask = real_rebuild(mem, ego_pose_world, route_world, mask)
        rebuilds.append(feats.detach().clone())
        return feats, nav_mask

    nav_tokens: list[torch.Tensor] = []
    real_policy_forward = model.policy.forward

    def policy_spy(tokens: torch.Tensor, key_mask: torch.Tensor):
        nav_tokens.append(tokens[:, 36].detach().clone())  # [z_od16, z_ld16, od_pool, ld_pool, others, z_ego, nav, ...]
        return real_policy_forward(tokens, key_mask)

    monkeypatch.setattr(net_model_module, "rebuild_nav_from_world", rebuild_spy)
    monkeypatch.setattr(model.policy, "forward", policy_spy)
    model.rollout(obs)

    assert len(rebuilds) == 6, f"rollout 应逐步重建 6 次，实际 {len(rebuilds)}"
    assert len(nav_tokens) == 6, f"策略头调用 6 次（t0 + rollout 5 步），实际 {len(nav_tokens)}"
    # t0 命令 forward；位移 5 m（默认动作）后第 1 次重建跨过 (10,0) → 命令 left
    assert float(t0_feats[0, 4]) == pytest.approx(1.0) and float(t0_feats[0, 4:7].sum()) == pytest.approx(1.0)
    assert float(rebuilds[0][0, 5]) == pytest.approx(1.0) and float(rebuilds[0][0, 4:7].sum()) == pytest.approx(1.0)
    advanced = advance_pose_world(obs["ego_world"], torch.tensor([5.0]), torch.tensor([0.0]))
    assert torch.allclose(poses[0], advanced, atol=1e-6), "第 1 次重建必须用位移后的世界位姿"
    # 切换后的 nav token 被策略头实际消费（t0 = forward 命令，第 1 步 = left 命令）
    expected_t0 = model.encoders.embed_nav(t0_feats.unsqueeze(1), torch.ones(1, 1))[:, 0]
    expected_step1 = model.encoders.embed_nav(rebuilds[0].unsqueeze(1), torch.ones(1, 1))[:, 0]
    assert torch.equal(nav_tokens[0], expected_t0), "t0 nav token 必须来自 forward 命令特征"
    assert torch.equal(nav_tokens[1], expected_step1), "第 1 步 nav token 必须来自切换后的 left 命令特征"
    assert not torch.allclose(nav_tokens[0], nav_tokens[1]), "位移后 nav 不得停留在 t0（命令切换必须生效）"


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


# ---------------------------------------------------------------- 信息充分性：均值令牌消融对照
def _mean_token_ablate(
    tokens: torch.Tensor, key_mask: torch.Tensor, *, start: int = 0, stop: int = 16
) -> torch.Tensor:
    """均值令牌消融（池化基线口径）：把 OD 槽令牌 ``[start, stop)`` 替换为其掩码均值广播。

    掩码均值 = ``Σ live 槽令牌 / live 槽数``；非 live 槽不参与（也不影响结果）。
    """
    ablated = tokens.clone()
    slot_mask = key_mask[:, start:stop]
    denom = slot_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    mean = (tokens[:, start:stop] * slot_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
    ablated[:, start:stop] = mean
    return ablated


def test_mean_token_ablation_removes_slot_distribution_distinguishability() -> None:
    """Gate1 §6.1 信息充分性判据（本测试重定义/落地口径）——**均值令牌消融对照**。

    扰动 = **掩码均值不变的 OD 槽位分布扰动**：3 个 live OD 槽，把 ``+δ/-δ`` 分别加到
    槽 0/槽 1（均值逐位不变，token 集合改变）。判据：

    1. **去池化头（原始令牌）**：该扰动可区分（``Δmu > 1e-4``）⇒ 逐槽分布/身份信息真实
       进入头（与真机 cut_in 探针 §5 的镜像/置换对照互补）；
    2. **均值令牌消融（池化基线）**：同一扰动 ``Δmu ≤ 1e-6`` 且 ``≤ 原始 Δmu 的 1e-3``
       （预期不可区分）⇒ 池化只保留掩码均值、逐槽分布信息被抹平（消融对照有区分度）；
    3. **消融有效性对照**：均值级扰动（live 槽整体取反 ⇒ 掩码均值取反）在消融后仍可区分
       （``Δmu > 1e-4``）⇒ 消融不是常数函数，池化保留的是"均值级"信息。
    """
    torch.manual_seed(0)
    policy = PolicyHead(hidden=H)
    with torch.no_grad():
        policy.mu.weight.normal_(0.0, 0.1)
        policy.mu.bias.zero_()

    torch.manual_seed(7)
    tokens = torch.randn(1, TOKEN_LENGTH, H)
    key_mask = torch.ones(1, TOKEN_LENGTH)
    live = (0, 1, 2)
    for slot in range(16):
        if slot not in live:
            key_mask[0, slot] = 0.0

    delta = torch.zeros(H)
    delta[0] = 0.5
    moved = tokens.clone()
    moved[0, live[0]] += delta
    moved[0, live[1]] -= delta

    with torch.no_grad():
        mu_base = policy(tokens, key_mask)[0]
        mu_moved = policy(moved, key_mask)[0]
        delta_orig = float((mu_base - mu_moved).abs().max())

        base_abl = _mean_token_ablate(tokens, key_mask)
        moved_abl = _mean_token_ablate(moved, key_mask)
        mean_gap = float((base_abl[:, :16] - moved_abl[:, :16]).abs().max())
        mu_base_abl = policy(base_abl, key_mask)[0]
        mu_moved_abl = policy(moved_abl, key_mask)[0]
        delta_abl = float((mu_base_abl - mu_moved_abl).abs().max())
        ablation_loss = float((mu_base - mu_base_abl).abs().max())

        flipped = tokens.clone()
        flipped[0, list(live)] *= -1.0  # 掩码均值 = -原均值（均值级扰动）
        mu_flipped_abl = policy(_mean_token_ablate(flipped, key_mask), key_mask)[0]
        delta_mean_level = float((mu_base_abl - mu_flipped_abl).abs().max())

    assert delta_orig > 1e-4, f"原始令牌对均值不变的槽位分布扰动不可区分（Δmu={delta_orig:.3e}）"
    assert mean_gap <= 1e-7, f"消融后两输入的掩码均值必须一致（gap={mean_gap:.3e}）"
    assert delta_abl <= 1e-6 and delta_abl <= 1e-3 * delta_orig, (
        f"均值令牌消融后不得保留槽位分布可区分性（Δmu_abl={delta_abl:.3e} vs Δmu_orig={delta_orig:.3e}）"
    )
    assert ablation_loss > 0.0, "消融必须实际改变头输入（逐槽令牌 ≠ 均值令牌）"
    assert delta_mean_level > 1e-4, "消融后均值级扰动仍应可区分（消融不是常数函数）"


def test_mirror_distinguishability_under_mean_token_ablation_on_model_obs() -> None:
    """Gate1 §6.1 信息充分性判据（真机探针 §5 同款镜像扰动 + 均值令牌消融）。

    在 ``make_obs`` 真实形状 obs（多 live OD 槽）上镜像全部 OD 槽（y/vy/sinθ 取反）：

    1. **原始令牌**：``Δmu > 1e-4`` ⇒ 逐槽位置/身份信息进入头（与真机 cut_in 探针一致）；
    2. **消融后**：镜像仍可区分（``Δmu_abl > 0``）——镜像改变 OD 掩码均值，池化基线仅剩
       均值级可区分性；"池化不可区分"只对**均值不变**的槽位分布成立（见上一测试）；
    3. **消融不是恒等**：消融前后头输出差 ``> 0``（逐槽令牌 ≠ 均值令牌）。
    """
    torch.manual_seed(0)
    model = DrivingModel(hidden=H, expert_hidden=256).eval()
    with torch.no_grad():
        model.policy.mu.weight.normal_(0.0, 0.1)
        model.policy.mu.bias.zero_()

    base = make_obs(batch=2, seed=3)
    mirrored = clone_obs(base)
    for key in ("od", "od_hist"):
        mirrored[key][..., 1] *= -1.0  # dy
        mirrored[key][..., 3] *= -1.0  # vy
        mirrored[key][..., 5] *= -1.0  # sinθ

    with torch.no_grad():
        encoded_base = model.encode(base)
        encoded_mirror = model.encode(mirrored)
        tokens_base, mask_base = encoded_base["tokens"], encoded_base["key_mask"]
        tokens_mirror, mask_mirror = encoded_mirror["tokens"], encoded_mirror["key_mask"]

        mu_base = model.policy(tokens_base, mask_base)[0]
        mu_mirror = model.policy(tokens_mirror, mask_mirror)[0]
        delta_orig = float((mu_base - mu_mirror).abs().max())

        mu_base_abl = model.policy(_mean_token_ablate(tokens_base, mask_base), mask_base)[0]
        mu_mirror_abl = model.policy(_mean_token_ablate(tokens_mirror, mask_mirror), mask_mirror)[0]
        delta_abl = float((mu_base_abl - mu_mirror_abl).abs().max())
        ablation_loss = float((mu_base - mu_base_abl).abs().max())

    assert delta_orig > 1e-4, f"镜像 L/R 在真实形状 obs 上不可区分（Δmu={delta_orig:.3e}）"
    assert delta_abl > 0.0, "镜像改变掩码均值 → 消融后仍应保留均值级可区分性（消融不是常数函数）"
    assert ablation_loss > 0.0, "消融必须实际改变头输入（逐槽令牌 ≠ 均值令牌）"


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
