"""net 线验收测试：形状 / hist_valid 门控 / 掩码 / MoE / 世界模型损失 / 运动学 / 参数预算。

约定（p2-contract §2、§8.4）：
- 纯 CPU、小 batch、固定种子，不加载 MetaDrive；
- ``hist_valid`` 必须门控预热补位帧（补位帧复制最旧真实帧且 ``*_hist_mask=1``）；
- 无效槽位（mask=0）不得影响任何输出；ego reserved 两维是"上一动作"，按普通特征消费。
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from net.encoders import EGO_DIM, H, LD_DIM, OD_DIM
from net.moe import MoEBlock
from net.model import (
    SUPERVISED_LABELS,
    DrivingModel,
    arc_step,
    compose_pose,
    interpolate_actions,
)
from net.policy import LOG_STD_MAX, LOG_STD_MIN, PolicyHead
from net.world_model import WorldModel, direct_multi_step_loss

ROOT = Path(__file__).resolve().parents[1]
PARAM_BUDGET = 1_500_000
BATCH = 3
HISTORY = 6
VALID_FRAMES = 3  # 前 3 帧为 warmup 补位（复制最旧真实帧，mask=1）


# ---------------------------------------------------------------------- 工具
def _model(seed: int = 0) -> DrivingModel:
    """固定种子的默认配置模型（eval 模式，确定性）。"""
    torch.manual_seed(seed)
    return DrivingModel().eval()


def make_obs(batch: int = BATCH, seed: int = 0, valid_frames: int = VALID_FRAMES) -> dict[str, torch.Tensor]:
    """构造与 env.obs.builder 同形状的观测（含 warmup 复制帧语义）。"""
    generator = torch.Generator().manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32)

    od_mask = (rand(batch, 16) > -0.4).float()
    ld_mask = (rand(batch, 16) > -0.4).float()
    od_hist = rand(batch, HISTORY, 16, OD_DIM)
    ld_hist = rand(batch, HISTORY, 16, LD_DIM)
    od_hist_mask = od_mask.unsqueeze(1).expand(batch, HISTORY, 16).clone()
    ld_hist_mask = ld_mask.unsqueeze(1).expand(batch, HISTORY, 16).clone()
    if valid_frames < HISTORY:
        oldest = HISTORY - valid_frames
        od_hist[:, :oldest] = od_hist[:, oldest : oldest + 1]  # memory.py 的补位方式
        ld_hist[:, :oldest] = ld_hist[:, oldest : oldest + 1]
    hist_valid = torch.zeros(batch, HISTORY)
    hist_valid[:, HISTORY - valid_frames :] = 1.0
    return {
        "ego": rand(batch, EGO_DIM),
        "od": rand(batch, 16, OD_DIM) * od_mask.unsqueeze(-1),
        "od_mask": od_mask,
        "ld": rand(batch, 16, LD_DIM) * ld_mask.unsqueeze(-1),
        "ld_mask": ld_mask,
        "nav": rand(batch, 11),
        "nav_mask": torch.ones(batch, 1),
        "signal": rand(batch, 4),
        "signal_mask": torch.ones(batch, 1),
        "od_hist": od_hist,
        "od_hist_mask": od_hist_mask,
        "ld_hist": ld_hist,
        "ld_hist_mask": ld_hist_mask,
        "hist_valid": hist_valid,
    }


def clone_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in obs.items()}


REQUIRED_OUTPUTS = (
    "action_mu",
    "action_logstd",
    "value",
    "traj_xy",
    "od_pred",
    "ld_pred",
    "router_logits",
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
    assert out["od_pred"].shape == (BATCH, 6, 16, 5)
    assert out["ld_pred"].shape == (BATCH, 6, 16, 4)
    assert out["router_logits"].shape == (BATCH, 8)
    assert out["latent"].shape == (BATCH, H)

    assert torch.all(out["action_mu"] >= model.policy.action_low - 1e-6)
    assert torch.all(out["action_mu"] <= model.policy.action_high + 1e-6)
    assert torch.all(out["action_logstd"] >= LOG_STD_MIN - 1e-6)
    assert torch.all(out["action_logstd"] <= LOG_STD_MAX + 1e-6)
    for key in REQUIRED_OUTPUTS:
        assert torch.isfinite(out[key]).all(), f"{key} 含非有限值"


def test_rollout_shapes_and_no_grad() -> None:
    """rollout 与 forward 形状一致，且全程不建图。"""
    model = _model()
    out = model.rollout(make_obs())
    assert tuple(out["traj_xy"].shape) == (BATCH, 6, 2)
    assert tuple(out["od_pred"].shape) == (BATCH, 6, 16, 5)
    assert tuple(out["ld_pred"].shape) == (BATCH, 6, 16, 4)
    assert all(not tensor.requires_grad for tensor in out.values())


def test_ego_reserved_dims_are_consumed() -> None:
    """ego 最后 2 维（上一动作）必须作为普通特征进入网络（§8.4）。"""
    model = _model()
    obs = make_obs()
    baseline = model(obs)["latent"]
    changed = clone_obs(obs)
    changed["ego"][:, 6:] += 1.0
    assert not torch.allclose(baseline, model(changed)["latent"], atol=1e-6)


# ---------------------------------------------------------------------- 门控/掩码
def test_hist_valid_gates_padded_frames() -> None:
    """hist_valid=0 的补位帧不得参与时序聚合；置 1 后必须改变输出。"""
    model = _model()
    obs = make_obs()
    baseline = model(obs)

    corrupted = clone_obs(obs)
    oldest = HISTORY - VALID_FRAMES
    corrupted["od_hist"][:, :oldest] = 100.0 * torch.randn_like(corrupted["od_hist"][:, :oldest])
    corrupted["ld_hist"][:, :oldest] = 100.0 * torch.randn_like(corrupted["ld_hist"][:, :oldest])
    # 补位帧 mask 本来就是 1，唯一能屏蔽它们的就是 hist_valid
    out_corrupted = model(corrupted)
    for key in REQUIRED_OUTPUTS:
        assert torch.equal(baseline[key], out_corrupted[key]), f"hist_valid=0 的补位帧仍影响了 {key}"

    ungated = clone_obs(obs)
    ungated["hist_valid"][:, :oldest] = 1.0
    out_ungated = model(ungated)["latent"]
    assert not torch.allclose(baseline["latent"], out_ungated, atol=1e-5), "hist_valid 未生效：真实历史被忽略"


def test_masked_slots_do_not_affect_outputs() -> None:
    """无效槽位填任意垃圾值都不得改变输出（当前帧 + 历史帧）。"""
    model = _model()
    obs = make_obs()
    baseline = model(obs)

    corrupted = clone_obs(obs)
    bad_od = obs["od_mask"] < 0.5
    bad_ld = obs["ld_mask"] < 0.5
    corrupted["od"][bad_od] = 99.0
    corrupted["ld"][bad_ld] = -99.0
    corrupted["od_hist"][obs["od_hist_mask"] < 0.5] = 77.0
    corrupted["ld_hist"][obs["ld_hist_mask"] < 0.5] = -77.0
    out = model(corrupted)
    for key in REQUIRED_OUTPUTS:
        assert torch.equal(baseline[key], out[key]), f"掩码外垃圾值改变了 {key}"


# ---------------------------------------------------------------------- MoE
def test_moe_primary_is_not_router_gated() -> None:
    """primary 不受门控影响；expert 零初始化 + 门控可显式覆盖。"""
    torch.manual_seed(0)
    moe = MoEBlock(hidden=8, num_experts=3, expert_hidden=16, router_hidden=8)
    x = torch.randn(4, 8)
    primary = moe.primary(x)

    out_default, aux_default = moe(x)
    assert torch.allclose(out_default, primary, atol=0.0), "零初始化 expert 不应改变初始输出"
    assert aux_default["router_logits"].shape == (4, 3)
    assert aux_default["expert_weights"].shape == (4, 3)
    assert torch.all((aux_default["expert_weights"] > 0) & (aux_default["expert_weights"] < 1))

    with torch.no_grad():  # 让 expert 非零，再验证 primary 与门控严格解耦
        for expert in moe.experts:
            expert[-1].weight.normal_()
            expert[-1].bias.normal_()
    out_zero, aux_zero = moe(x, gates=torch.zeros(4, 3))
    out_one, _ = moe(x, gates=torch.ones(4, 3))
    assert torch.allclose(out_zero, primary, atol=0.0)
    assert not torch.allclose(out_one, primary, atol=1e-6)
    assert "effective_experts" in aux_zero


def test_supervised_labels_match_config() -> None:
    """router 监督标签与 config/model.yaml 固定顺序一致，且与 expert 数一一对应。"""
    config = yaml.safe_load((ROOT / "config" / "model.yaml").read_text(encoding="utf-8"))
    labels = tuple(config["moe"]["router"]["supervised_labels"])
    assert labels == SUPERVISED_LABELS
    assert len(SUPERVISED_LABELS) == int(config["moe"]["experts"]["count"]) == 8
    moe = MoEBlock(hidden=8, num_experts=len(SUPERVISED_LABELS), expert_hidden=8, router_hidden=8)
    assert moe.num_experts == len(SUPERVISED_LABELS)


# ---------------------------------------------------------------------- 世界模型
def test_world_model_untrained_is_constant_velocity_prior() -> None:
    """零初始化解码器 ⇒ 未训练世界模型 = OD 匀速 / LD 静止先验（Stage B 对照下界）。"""
    torch.manual_seed(0)
    model = WorldModel(hidden=16, od_slots=4, ld_slots=4, steps=6).eval()
    latent = torch.randn(2, 16)
    action = torch.randn(2, 2) * 0.1
    od_state = torch.randn(2, 4, 5)
    ld_state = torch.randn(2, 4, 4)
    od_pred, ld_pred, _ = model.step(latent, action, od_state, ld_state, 2)
    expected = torch.stack(
        [
            od_state[..., 0] + od_state[..., 2] * 1.0,
            od_state[..., 1] + od_state[..., 3] * 1.0,
            od_state[..., 2],
            od_state[..., 3],
            od_state[..., 4],
        ],
        dim=-1,
    )
    assert torch.allclose(od_pred, expected, atol=1e-6)
    assert torch.allclose(ld_pred, ld_state, atol=1e-6)


def test_world_model_multistep_shapes_and_loss_mask() -> None:
    """直接多步形状 + 掩码损失：完全预测=0，掩码外垃圾不影响损失。"""
    torch.manual_seed(0)
    model = WorldModel(hidden=16, od_slots=4, ld_slots=4, steps=6).eval()
    latent = torch.randn(2, 16)
    plan = torch.randn(2, 6, 2) * 0.2
    od_state = torch.randn(2, 4, 5)
    ld_state = torch.randn(2, 4, 4)
    od_pred, ld_pred, latent_seq = model(latent, plan, od_state, ld_state)
    assert tuple(od_pred.shape) == (2, 6, 4, 5)
    assert tuple(ld_pred.shape) == (2, 6, 4, 4)
    assert tuple(latent_seq.shape) == (2, 6, 16)

    od_mask = (torch.rand(2, 6, 4) > 0.3).float()
    ld_mask = (torch.rand(2, 6, 4) > 0.3).float()
    loss_perfect = direct_multi_step_loss(od_pred, ld_pred, od_pred.clone(), ld_pred.clone(), od_mask, ld_mask)
    assert loss_perfect.item() == pytest.approx(0.0, abs=1e-8)

    # 有效槽位给出非零误差，掩码外填垃圾：损失必须只由有效槽位决定
    od_target = od_pred.detach().clone()
    od_target[..., 0] += 0.5
    ld_target = ld_pred.detach().clone()
    ld_target[..., 0] -= 0.5
    loss_base = direct_multi_step_loss(od_pred, ld_pred, od_target, ld_target, od_mask, ld_mask)
    assert loss_base.item() > 0.0

    bad_od = od_pred.detach().clone()
    bad_ld = ld_pred.detach().clone()
    bad_od[od_mask < 0.5] = 1e3
    bad_ld[ld_mask < 0.5] = -1e3
    bad_target_od = od_target.clone()
    bad_target_ld = ld_target.clone()
    bad_target_od[od_mask < 0.5] = -1e3
    bad_target_ld[ld_mask < 0.5] = 1e3
    loss_masked = direct_multi_step_loss(
        bad_od, bad_ld, bad_target_od, bad_target_ld, od_mask, ld_mask
    )
    assert torch.allclose(loss_masked, loss_base, atol=1e-8)

    valid = torch.ones(2, 6)
    valid[:, 3:] = 0.0
    loss_partial = direct_multi_step_loss(
        od_pred, ld_pred, od_target + 1.0, ld_target + 1.0, od_mask, ld_mask, valid
    )
    assert torch.isfinite(loss_partial) and loss_partial.item() > 0.0

    loss_grad = direct_multi_step_loss(
        od_pred, ld_pred, od_target + 0.1, ld_target + 0.1, od_mask, ld_mask
    )
    loss_grad.backward()
    assert model.od_head[-1].weight.grad is not None
    assert model.od_head[-1].weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------- 运动学/策略
def test_arc_step_and_interpolation() -> None:
    """圆弧运动学解析值与 6→30 插值端点一致性（§8.5 单一真源）。"""
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


# ---------------------------------------------------------------------- 装配/输入
def test_param_budget_and_module_counts() -> None:
    """参数总量 ≤1.5M，且各子模块均非空。"""
    torch.manual_seed(0)
    model = DrivingModel()
    total = sum(param.numel() for param in model.parameters())
    assert total <= PARAM_BUDGET, f"参数超预算: {total}"
    for module in (
        model.encoders,
        model.temporal,
        model.spatial,
        model.moe,
        model.world_model,
        model.policy,
        model.value,
    ):
        assert sum(param.numel() for param in module.parameters()) > 0


def test_deterministic_same_seed() -> None:
    """同种子两次构建输出逐位一致（纯 CPU 确定性）。"""
    model_a = _model(seed=7)
    model_b = _model(seed=7)
    obs = make_obs(seed=1)
    out_a, out_b = model_a(obs), model_b(obs)
    for key in REQUIRED_OUTPUTS:
        assert torch.equal(out_a[key], out_b[key]), f"{key} 不确定"


def test_input_validation_rejects_bad_obs() -> None:
    """缺通道 / 形状错 / dtype 错必须显式报错。"""
    model = _model()
    obs = make_obs()
    missing = {key: value for key, value in obs.items() if key != "hist_valid"}
    with pytest.raises(ValueError):
        model(missing)

    bad_shape = clone_obs(obs)
    bad_shape["od"] = bad_shape["od"][:, :15]
    with pytest.raises(ValueError):
        model(bad_shape)

    bad_dtype = clone_obs(obs)
    bad_dtype["ego"] = bad_dtype["ego"].double()
    with pytest.raises(ValueError):
        model(bad_dtype)
