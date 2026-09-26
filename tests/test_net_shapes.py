"""net 线验收测试（v2）：mem-bank 形状/掩码/validity、MoE top-2、ST-GNN 先验、参数预算。

约定（design-v1.2 §2/§3、p2-contract §2/§8.4）：
- 纯 CPU、小 batch、固定种子，不加载 MetaDrive；
- ``hist_valid`` 必须门控预热补位帧（补位帧复制最旧真实帧且 ``*_hist_mask=1``）；
- 无效槽位（``mask=0`` 或 ``presence=0``）不得影响任何输出；ego reserved 两维是"上一动作"；
- LD 不做时序：历史帧（除当前帧）不得影响任何输出（design-v1.2 §2）。

rollout 语义（拷贝隔离 / detach / 梯度表）在 ``tests/test_mem_rollout.py``。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from net.encoders import EGO_MEM_DIM, H, LD_MEM_DIM, OD_MEM_DIM
from net.mem import mem_from_obs
from net.moe import MoEBlock, top_k_softmax
from net.model import (
    SUPERVISED_LABELS,
    DrivingModel,
    arc_step,
    compose_pose,
    interpolate_actions,
)
from net.policy import LOG_STD_MAX, LOG_STD_MIN, PolicyHead
from net.st_gnn import SpatioTemporalGNN

ROOT = Path(__file__).resolve().parents[1]
PARAM_BUDGET = 1_500_000
BATCH = 3
HISTORY = 6
VALID_FRAMES = 3  # 前 3 帧为 warmup 补位（复制最旧真实帧，mask=1）
OD_SLOTS = 16
LD_SLOTS = 16
OTHERS_DIM = 28  # env schema v2：nav(11)+speed_limit(1)+signal(4)+road_class(12)


# ---------------------------------------------------------------------- 工具
def _model(seed: int = 0) -> DrivingModel:
    """固定种子的默认配置模型（eval 模式，确定性）。"""
    torch.manual_seed(seed)
    return DrivingModel().eval()


def make_obs(
    batch: int = BATCH,
    seed: int = 0,
    valid_frames: int = VALID_FRAMES,
    *,
    with_companions: bool = True,
    with_ego_hist: bool = True,
    with_others_hist: bool = True,
) -> dict[str, torch.Tensor]:
    """构造与 env schema v2 同形状的观测（含 warmup 复制帧语义 + 伴随数组）。

    默认贴近 trainer 组装后的形状：单槽历史 ``(B,6,1,F)``、od/ld ``(B,6,S,F)``；
    ``with_companions=False`` 时省略 ``od_id_hist/od_presence_hist``（测回退路径）。
    """
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    od_mask = (rand(batch, HISTORY, OD_SLOTS) > -0.4).float()
    ld_mask = (rand(batch, HISTORY, LD_SLOTS) > -0.4).float()
    presence = od_mask * (rand(batch, HISTORY, OD_SLOTS) > -0.3).float()
    od_hist = rand(batch, HISTORY, OD_SLOTS, OD_MEM_DIM) * od_mask.unsqueeze(-1)
    ld_hist = rand(batch, HISTORY, LD_SLOTS, LD_MEM_DIM) * ld_mask.unsqueeze(-1)
    ego_hist = rand(batch, HISTORY, 1, EGO_MEM_DIM)
    others_hist = rand(batch, HISTORY, 1, OTHERS_DIM)
    if valid_frames < HISTORY:
        oldest = HISTORY - valid_frames
        od_hist[:, :oldest] = od_hist[:, oldest : oldest + 1]  # memory.py 的补位方式
        ld_hist[:, :oldest] = ld_hist[:, oldest : oldest + 1]
        ego_hist[:, :oldest] = ego_hist[:, oldest : oldest + 1]
        others_hist[:, :oldest] = others_hist[:, oldest : oldest + 1]
    hist_valid = torch.zeros(batch, HISTORY)
    hist_valid[:, HISTORY - valid_frames :] = 1.0
    obs: dict[str, torch.Tensor] = {
        "od_hist": od_hist,
        "od_hist_mask": od_mask,
        "ld_hist": ld_hist,
        "ld_hist_mask": ld_mask,
        "hist_valid": hist_valid,
        # 当前帧（trainer 组装后的挤压形状）
        "ego": rand(batch, EGO_MEM_DIM),
        "od": od_hist[:, -1],
        "od_mask": od_mask[:, -1],
        "od_presence": presence[:, -1],
        "ld": ld_hist[:, -1],
        "ld_mask": ld_mask[:, -1],
        "nav": rand(batch, 11),
        "nav_mask": torch.ones(batch, 1),
        "signal": rand(batch, 4),
        "signal_mask": torch.ones(batch, 1),
    }
    if with_ego_hist:
        obs["ego_hist"] = ego_hist
        obs["ego_hist_mask"] = torch.ones(batch, HISTORY, 1)
    if with_others_hist:
        obs["others_hist"] = others_hist
        obs["others_hist_mask"] = torch.ones(batch, HISTORY, 1)
    if with_companions:
        obs["od_id_hist"] = (rand(batch, HISTORY, OD_SLOTS).abs() * 10).long() + 1
        obs["od_presence_hist"] = presence
        obs["od_id"] = obs["od_id_hist"][:, -1].clone()
    return obs


def clone_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in obs.items()}


#: 全量前向必须包含的键（输出契约）
REQUIRED_OUTPUTS = (
    "action_mu",
    "action_logstd",
    "value",
    "traj_xy",
    "od_pred",
    "ld_pred",
    "od_presence_pred",
    "od_entry_pred",
    "router_logits",
    "expert_weights",
    "latent",
)


# ---------------------------------------------------------------------- 形状/边界
def test_forward_shapes_and_bounds() -> None:
    """前向输出形状与动作/对数标准差边界。"""
    model = _model()
    out = model(make_obs())
    assert out["action_mu"].shape == (BATCH, 2)
    assert out["action_logstd"].shape == (BATCH, 2)
    assert out["value"].shape == (BATCH, 1)
    assert out["traj_xy"].shape == (BATCH, 6, 2)
    assert out["traj_theta"].shape == (BATCH, 6)
    assert out["plan"].shape == (BATCH, 6, 2)
    assert out["od_pred"].shape == (BATCH, 6, 16, 5)
    assert out["ld_pred"].shape == (BATCH, 6, 16, 4)
    assert out["od_presence_pred"].shape == (BATCH, 6, 16)
    assert out["od_entry_pred"].shape == (BATCH, 6, 16)
    assert out["router_logits"].shape == (BATCH, 8)
    assert out["expert_weights"].shape == (BATCH, 8)
    assert out["latent"].shape == (BATCH, H)

    assert torch.all(out["action_mu"] >= model.policy.action_low - 1e-6)
    assert torch.all(out["action_mu"] <= model.policy.action_high + 1e-6)
    assert torch.all(out["action_logstd"] >= LOG_STD_MIN - 1e-6)
    assert torch.all(out["action_logstd"] <= LOG_STD_MAX + 1e-6)
    for key in REQUIRED_OUTPUTS:
        assert torch.isfinite(out[key]).all(), f"{key} 含非有限值"


def test_plan_first_action_matches_action_mu() -> None:
    """``plan[:,0]`` 必须与 ``action_mu`` 逐位一致（同一 plan-head 计算）。"""
    model = _model()
    out = model(make_obs())
    assert torch.equal(out["plan"][:, 0], out["action_mu"])


def test_rollout_shapes_and_no_grad() -> None:
    """rollout 与 forward 形状一致，且全程不建图。"""
    model = _model()
    out = model.rollout(make_obs())
    assert tuple(out["traj_xy"].shape) == (BATCH, 6, 2)
    assert tuple(out["od_pred"].shape) == (BATCH, 6, 16, 5)
    assert tuple(out["ld_pred"].shape) == (BATCH, 6, 16, 4)
    assert tuple(out["od_presence_pred"].shape) == (BATCH, 6, 16)
    assert tuple(out["od_entry_pred"].shape) == (BATCH, 6, 16)
    assert all(not tensor.requires_grad for tensor in out.values())


def test_ego_reserved_dims_are_consumed() -> None:
    """ego mem 当前帧最后 2 维（上一动作）必须作为普通特征进入网络（§8.4）。"""
    model = _model()
    obs = make_obs()
    baseline = model(obs)["latent"]
    changed = clone_obs(obs)
    changed["ego_hist"][:, -1, 0, 6:] += 1.0
    assert not torch.allclose(baseline, model(changed)["latent"], atol=1e-6)


# ---------------------------------------------------------------------- 门控/掩码
def test_hist_valid_gates_padded_frames() -> None:
    """hist_valid=0 的补位帧不得参与时序聚合；置 1 后必须改变输出。"""
    model = _model()
    obs = make_obs()
    baseline = model(obs)

    corrupted = clone_obs(obs)
    oldest = HISTORY - VALID_FRAMES
    for key in ("od_hist", "ld_hist", "ego_hist", "others_hist"):
        corrupted[key][:, :oldest] = 100.0 * torch.randn_like(corrupted[key][:, :oldest])
    out_corrupted = model(corrupted)
    for key in REQUIRED_OUTPUTS:
        assert torch.equal(baseline[key], out_corrupted[key]), f"hist_valid=0 的补位帧仍影响了 {key}"

    ungated = clone_obs(obs)
    ungated["hist_valid"][:, :oldest] = 1.0
    out_ungated = model(ungated)["latent"]
    assert not torch.allclose(baseline["latent"], out_ungated, atol=1e-5), "hist_valid 未生效：真实历史被忽略"


def test_masked_slots_do_not_affect_outputs() -> None:
    """无效槽位（mask=0 / 当前帧 presence=0）填任意垃圾都不得改变输出。

    注意：``presence=0`` 的**历史帧**特征按 env 语义是"最近一次盒内观测"（stale but valid），
    会参与 live 槽位的时间注意力——这是设计意图；不变量只针对当前帧无效槽位与全帧 mask=0。
    """
    model = _model()
    obs = make_obs()
    baseline = model(obs)

    corrupted = clone_obs(obs)
    dead = obs["od_hist_mask"] < 0.5  # 全帧 mask=0
    corrupted["od_hist"][dead] = 99.0
    corrupted["od_id_hist"][dead] = 12345
    now_absent = (obs["od_hist_mask"][:, -1] > 0.5) & (obs["od_presence_hist"][:, -1] < 0.5)
    corrupted["od_hist"][:, -1][now_absent] = -99.0
    corrupted["od_id_hist"][:, -1][now_absent] = 23456
    corrupted["ld_hist"][obs["ld_hist_mask"] < 0.5] = -77.0
    out = model(corrupted)
    for key in REQUIRED_OUTPUTS:
        assert torch.equal(baseline[key], out[key]), f"掩码外垃圾值改变了 {key}"


def test_ld_history_is_not_temporal() -> None:
    """LD 不做时序：改动 ld 历史（除当前帧）不得改变任何输出（design-v1.2 §2）。"""
    model = _model()
    obs = make_obs()
    baseline = model(obs)

    corrupted = clone_obs(obs)
    corrupted["ld_hist"][:, :-1] = 100.0 * torch.randn_like(corrupted["ld_hist"][:, :-1])
    out = model(corrupted)
    for key in baseline:
        assert torch.equal(baseline[key], out[key]), f"LD 历史（非当前帧）仍影响了 {key}"


# ---------------------------------------------------------------------- 输入契约/回退
def test_mem_fallbacks_and_squeeze_equivalence() -> None:
    """缺失通道回退（无伴随数组 / 无 ego_hist / 无 others_hist）与单槽挤压等价。"""
    model = _model()
    out_full = model(make_obs())

    for kwargs in ({"with_companions": False}, {"with_ego_hist": False}, {"with_others_hist": False}):
        out = model(make_obs(**kwargs))
        for key in REQUIRED_OUTPUTS:
            assert torch.isfinite(out[key]).all(), f"{kwargs} 下 {key} 非有限"

    # 伴随数组缺失时必须回退到当前帧 od_id/od_presence（有则语义等价；这里只验证有限/可跑）
    no_companions = make_obs(with_companions=False)
    assert "od_id_hist" not in no_companions and "od_presence_hist" not in no_companions
    out_nc = model(no_companions)
    assert out_nc["od_pred"].shape == (BATCH, 6, 16, 5)

    # 单槽通道挤压与否等价（env builder 原始 (B,6,1,F) vs trainer 挤压后的 (B,6,F)）
    squeezed = clone_obs(make_obs())
    for key in ("ego_hist", "others_hist", "ego_hist_mask", "others_hist_mask"):
        squeezed[key] = squeezed[key].squeeze(2)
    out_squeezed = model(squeezed)
    for key in REQUIRED_OUTPUTS:
        assert torch.allclose(out_full[key], out_squeezed[key], atol=0.0), f"{key} 挤压前后不等价"

    # 当前帧单槽通道的 env 原始形状（(B,1,F)/(B,1,1)）与 trainer 挤压形状等价
    raw = clone_obs(make_obs())
    for key in ("nav", "signal"):
        raw[key] = raw[key].unsqueeze(1)
    for key in ("nav_mask", "signal_mask"):
        raw[key] = raw[key].unsqueeze(1)
    raw["others"] = torch.randn(BATCH, 1, OTHERS_DIM)
    raw["others_mask"] = torch.ones(BATCH, 1, 1)
    out_raw = model(raw)
    for key in REQUIRED_OUTPUTS:
        assert torch.isfinite(out_raw[key]).all()


def test_mem_parser_rejects_bad_obs() -> None:
    """mem 解析器：缺失键 / 特征维错 / others 维不匹配必须显式报错。"""
    obs = make_obs()
    with pytest.raises(ValueError):
        mem_from_obs(
            {k: v for k, v in obs.items() if k != "hist_valid"}, others_dim=OTHERS_DIM, history_frames=HISTORY
        )
    bad_shape = clone_obs(obs)
    bad_shape["od_hist"] = bad_shape["od_hist"][..., : OD_MEM_DIM - 1]
    with pytest.raises(ValueError):
        mem_from_obs(bad_shape, others_dim=OTHERS_DIM, history_frames=HISTORY)
    bad_ld = clone_obs(obs)
    bad_ld["ld_hist"] = bad_ld["ld_hist"][:, :5]
    with pytest.raises(ValueError):
        mem_from_obs(bad_ld, others_dim=OTHERS_DIM, history_frames=HISTORY)
    with pytest.raises(ValueError):
        mem_from_obs(clone_obs(obs), others_dim=8, history_frames=HISTORY)


def test_model_rejects_bad_obs() -> None:
    """模型输入校验：缺通道 / 槽位数不匹配 / dtype 错必须显式报错。"""
    model = _model()
    obs = make_obs()
    missing = {key: value for key, value in obs.items() if key != "hist_valid"}
    with pytest.raises(ValueError):
        model(missing)

    bad_slots = clone_obs(obs)
    bad_slots["od_hist"] = bad_slots["od_hist"][:, :, :15]
    bad_slots["od_hist_mask"] = bad_slots["od_hist_mask"][:, :, :15]
    bad_slots["od_id_hist"] = bad_slots["od_id_hist"][:, :, :15]
    bad_slots["od_presence_hist"] = bad_slots["od_presence_hist"][:, :, :15]
    with pytest.raises(ValueError):
        model(bad_slots)

    bad_dtype = clone_obs(obs)
    bad_dtype["ego_hist"] = bad_dtype["ego_hist"].double()
    with pytest.raises(ValueError):
        model(bad_dtype)


# ---------------------------------------------------------------------- MoE
def test_moe_primary_is_not_router_gated() -> None:
    """primary 不受门控影响；expert 零初始化 + 门控可显式覆盖（v2：top-2 软混合）。"""
    torch.manual_seed(0)
    moe = MoEBlock(hidden=8, num_experts=3, expert_hidden=16, router_hidden=8, top_k=2)
    x = torch.randn(4, 8)
    primary = moe.primary(x)

    out_default, aux_default = moe(x)
    assert torch.allclose(out_default, primary, atol=0.0), "零初始化 expert 不应改变初始输出"
    assert aux_default["router_logits"].shape == (4, 3)
    assert aux_default["expert_weights"].shape == (4, 3)
    assert torch.allclose(aux_default["effective_experts"], torch.ones(4), atol=1e-6)

    with torch.no_grad():  # 让 expert 非零，再验证 primary 与门控严格解耦
        for expert in moe.experts:
            expert[-1].weight.normal_()
            expert[-1].bias.normal_()
    out_zero, aux_zero = moe(x, gates=torch.zeros(4, 3))
    out_one, _ = moe(x, gates=torch.ones(4, 3))
    assert torch.allclose(out_zero, primary, atol=0.0)
    assert not torch.allclose(out_one, primary, atol=1e-6)
    assert "effective_experts" in aux_zero


def test_top_k_softmax_is_sparse_and_normalized() -> None:
    """top-2 软混合：每行恰好 2 个非零、和为 1，且等于 top-2 上的 softmax。"""
    torch.manual_seed(0)
    logits = torch.randn(5, 8)
    weights = top_k_softmax(logits, 2)
    assert torch.all((weights > 0).sum(dim=-1) == 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5), atol=1e-6)
    top = logits.topk(2, dim=-1)
    expected = torch.softmax(top.values, dim=-1)
    assert torch.allclose(weights.gather(-1, top.indices), expected, atol=1e-6)

    # 与全 8 维 softmax 在 top-2 上重归一化等价
    full = torch.softmax(logits, dim=-1)
    selected = full.gather(-1, top.indices)
    renorm = selected / selected.sum(dim=-1, keepdim=True)
    assert torch.allclose(expected, renorm, atol=1e-6)


def test_moe_gradient_flows_to_router_and_experts() -> None:
    """top-2 混合对 router / expert 均有梯度（可被训练侧软目标监督）。"""
    torch.manual_seed(0)
    moe = MoEBlock(hidden=8, num_experts=4, expert_hidden=8, router_hidden=8, top_k=2)
    x = torch.randn(6, 8)
    with torch.no_grad():
        for expert in moe.experts:
            expert[-1].weight.normal_(0.0, 0.05)
    out, _ = moe(x)
    out.sum().backward()
    assert float(moe.router[-1].weight.grad.abs().sum()) > 0.0
    assert any(float(expert[-1].weight.grad.abs().sum()) > 0.0 for expert in moe.experts)


def test_supervised_labels_match_config() -> None:
    """router 监督标签与 config/model.yaml 固定顺序一致，且与 expert 数一一对应。"""
    config = yaml.safe_load((ROOT / "config" / "model.yaml").read_text(encoding="utf-8"))
    labels = tuple(config["moe"]["router"]["supervised_labels"])
    assert labels == SUPERVISED_LABELS
    assert len(SUPERVISED_LABELS) == int(config["moe"]["experts"]["count"]) == 8
    moe = MoEBlock(hidden=8, num_experts=len(SUPERVISED_LABELS), expert_hidden=8, router_hidden=8)
    assert moe.num_experts == len(SUPERVISED_LABELS)


# ---------------------------------------------------------------------- ST-GNN
def test_st_gnn_untrained_is_constant_velocity_prior() -> None:
    """零初始化解码器 ⇒ 未训练 ST-GNN = OD 匀速 / LD 静止先验（Stage A 对照下界）。"""
    torch.manual_seed(0)
    model = SpatioTemporalGNN(hidden=16, od_slots=4, ld_slots=4, steps=6, spatial_layers=1).eval()
    batch = 2
    ego_ctx = torch.randn(batch, 16)
    od_ctx = torch.randn(batch, 4, 16)
    ld_ctx = torch.randn(batch, 4, 16)
    node_mask = torch.ones(batch, 1 + 4 + 4)
    pose = torch.randn(batch, 1 + 4 + 4, 3)
    pose[:, 0] = 0.0
    od_anchor = torch.randn(batch, 4, 5)
    ld_anchor = torch.randn(batch, 4, 4)

    od_pred, ld_pred, presence, entry = model(
        ego_ctx=ego_ctx, od_ctx=od_ctx, ld_ctx=ld_ctx, node_mask=node_mask, pose=pose,
        step_index=2, od_anchor=od_anchor, ld_anchor=ld_anchor,
    )
    expected = torch.stack(
        [
            od_anchor[..., 0] + od_anchor[..., 2] * 1.0,
            od_anchor[..., 1] + od_anchor[..., 3] * 1.0,
            od_anchor[..., 2],
            od_anchor[..., 3],
            od_anchor[..., 4],
        ],
        dim=-1,
    )
    assert torch.allclose(od_pred, expected, atol=1e-6)
    assert torch.allclose(ld_pred, ld_anchor, atol=1e-6)
    # presence 先验 ≈0.88（持续存在）、entry 先验 ≈0.12（新进入罕见）：不与输入耦合
    assert torch.allclose(presence, torch.full_like(presence, presence[0, 0].item()), atol=1e-6)
    assert float(torch.sigmoid(presence).mean()) > 0.8
    assert float(torch.sigmoid(entry).mean()) < 0.2


def test_st_gnn_step_index_controls_prior_horizon() -> None:
    """先验视界随 step_index 增大：k=6 的外推位移必须是 k=1 的 6 倍（匀速先验）。"""
    torch.manual_seed(0)
    model = SpatioTemporalGNN(hidden=16, od_slots=4, ld_slots=4, steps=6, spatial_layers=1).eval()
    batch = 1
    od_anchor = torch.zeros(batch, 4, 5)
    od_anchor[..., 2] = 2.0  # vx
    kwargs = dict(
        ego_ctx=torch.randn(batch, 16), od_ctx=torch.randn(batch, 4, 16), ld_ctx=torch.randn(batch, 4, 16),
        node_mask=torch.ones(batch, 9), pose=torch.zeros(batch, 9, 3),
        od_anchor=od_anchor, ld_anchor=torch.zeros(batch, 4, 4),
    )
    od_1, _, _, _ = model(step_index=1, **kwargs)
    od_6, _, _, _ = model(step_index=6, **kwargs)
    assert od_1[..., 0].mean().item() == pytest.approx(1.0, abs=1e-6)
    assert od_6[..., 0].mean().item() == pytest.approx(6.0, abs=1e-6)


# 注：v1 ``net.world_model`` 已删除（无调用方；Stage A 走 mem + ST-GNN 教师强制），
# 其"未训练=匀速先验 / 多步掩码损失"覆盖分别由上面的 ST-GNN 先验测试与
# ``pipeline.trainer.weighted_od_multi_step_loss``（tests/test_weight_accounting.py）承担。


# ---------------------------------------------------------------------- 运动学/策略
def test_arc_step_and_interpolation() -> None:
    """圆弧运动学解析值与 6→30 插值端点一致性（§8.5 单一真源）。"""
    import math

    ds = torch.tensor([2.0])
    dtheta = torch.tensor([0.5])
    dx, dy = arc_step(ds, dtheta)
    radius = 2.0 / 0.5
    assert dx.item() == pytest.approx(radius * math.sin(0.5), abs=1e-6)
    assert dy.item() == pytest.approx(radius * (1.0 - math.cos(0.5)), abs=1e-6)

    straight_dx, straight_dy = arc_step(torch.tensor([2.0]), torch.tensor([0.0]))
    assert straight_dx.item() == pytest.approx(2.0, abs=1e-6)
    assert straight_dy.item() == pytest.approx(0.0, abs=1e-6)

    actions = torch.randn(2, 6, 2) * 0.3
    actions[..., 0] = actions[..., 0].abs()
    poses = interpolate_actions(actions, dt=0.5, hz=10)
    assert tuple(poses.shape) == (2, 30, 3)

    pose = torch.zeros(2, 3)
    endpoints = []
    for k in range(6):
        step_dx, step_dy = arc_step(actions[:, k, 0], actions[:, k, 1])
        pose = compose_pose(pose, step_dx, step_dy, actions[:, k, 1])
        endpoints.append(pose)
    assert torch.allclose(poses[:, 4::5], torch.stack(endpoints, dim=1), atol=1e-5)
    assert torch.isfinite(poses).all()


def test_policy_sample_and_log_prob() -> None:
    """策略采样有界、固定 generator 可复现、log_prob 有限。"""
    torch.manual_seed(0)
    policy = PolicyHead(hidden=16)
    latent = torch.randn(5, 16)
    mu, log_std = policy(latent)
    assert torch.all(mu >= policy.action_low - 1e-6) and torch.all(mu <= policy.action_high + 1e-6)
    assert torch.all(log_std >= LOG_STD_MIN - 1e-6) and torch.all(log_std <= LOG_STD_MAX + 1e-6)

    sample_a = policy.sample(latent, generator=torch.Generator().manual_seed(1))
    sample_b = policy.sample(latent, generator=torch.Generator().manual_seed(1))
    assert torch.equal(sample_a, sample_b)
    assert torch.all(sample_a >= policy.action_low - 1e-6) and torch.all(sample_a <= policy.action_high + 1e-6)
    log_prob = policy.log_prob(latent, sample_a)
    assert log_prob.shape == (5,) and torch.isfinite(log_prob).all()


# ---------------------------------------------------------------------- 装配
def test_param_budget_and_module_counts() -> None:
    """参数总量 ≤1.5M（默认 H=96 与训练配置 H=128/256），且各子模块均非空。"""
    torch.manual_seed(0)
    model = DrivingModel()
    total = sum(param.numel() for param in model.parameters())
    assert total <= PARAM_BUDGET, f"参数超预算: {total}"
    # others 维 = env schema v2 规范值 28（nav 11 + speed_limit 1 + signal 4 + road_class 12）
    assert model.others_dim == 28
    for module in (
        model.encoders,
        model.mem_encoder,
        model.plan_head,
        model.st_gnn,
        model.policy,
        model.value,
    ):
        assert sum(param.numel() for param in module.parameters()) > 0

    torch.manual_seed(0)
    configured = DrivingModel(hidden=128, expert_hidden=256)
    configured_total = sum(param.numel() for param in configured.parameters())
    assert configured_total <= PARAM_BUDGET, f"H=128 配置超预算: {configured_total}"

    # MoE 只在 plan head 内（规格第 3 条），且 top-2 软混合
    assert configured.plan_head.moe.top_k == 2
    assert configured.plan_head.moe.num_experts == 8
    assert not hasattr(model, "world_model")


def test_deterministic_same_seed() -> None:
    """同种子两次构建输出逐位一致（纯 CPU 确定性）。"""
    model_a = _model(seed=7)
    model_b = _model(seed=7)
    obs = make_obs(seed=1)
    out_a, out_b = model_a(obs), model_b(obs)
    for key in REQUIRED_OUTPUTS + ("traj_theta", "plan"):
        assert torch.equal(out_a[key], out_b[key]), f"{key} 不确定"
