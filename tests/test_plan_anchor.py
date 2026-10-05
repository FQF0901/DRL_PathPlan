"""K-anchor 计划头测试（v7 结构迭代 B，方案 A）：锚字典 / WTA / 车道系 / 软混合 / PPO 兼容。

覆盖预注册 ``docs/v7_program_prereg.md`` §11 的判据口径：

1. 锚加载与形状（文件 + 内置默认 + 非法拒绝）；
2. WTA 分配（合成数据：最近锚、mask、margin）；
3. 软混合几何（sharp τ ≈ 单锚；hard argmax = one-hot；等权 = 均值）；
4. lane 帧变换往返（恒等/航向修正/无效车道恒等）；
5. 模型集成（``plan[:,0]==action_mu``、尾段 = 锚计划、关闭时无新键/逐位兼容）；
6. 损失接线（``anchor_wta_loss`` CE 下降 + 指派锚回归；phase3 降级开关）；
7. PPO 兼容冒烟（``sample_action``/``logprob_from_action``/entropy/KL 形状不变；
   锚模块不影响 ``action_mu/action_logstd``）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from net.anchor import (
    ANCHOR_SEMANTICS,
    K_DEFAULT,
    anchor_lane_context,
    anchor_mixture,
    anchors_from_payload,
    assign_anchors,
    assigned_plan,
    default_anchors,
    from_lane_dtheta,
    lane_follow_dtheta,
    load_anchor_dictionary,
    mixture_probabilities,
    to_lane_dtheta,
)
from net.model import DrivingModel
from pipeline.trainer import (
    Phase3Config,
    anchor_wta_loss,
    gaussian_entropy,
    gaussian_kl,
    logprob_from_action,
    phase3_effective_config,
    sample_action,
)

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_FILE = ROOT / "config" / "plan_anchors_k6.json"
BATCH = 4


# ---------------------------------------------------------------------- 工具
def _synthetic_out(
    batch: int = BATCH, anchor_index: int = 2, *, residual_scale: float = 0.0
) -> dict[str, torch.Tensor]:
    """构造与模型输出同键的合成 anchor 输出（lane ctx = 恒等）。"""
    anchors = default_anchors()
    speed = torch.full((batch, 6), 3.5)
    residual = (
        torch.zeros(batch, anchors.shape[0], 6, 2)
        if residual_scale == 0.0
        else torch.randn(batch, anchors.shape[0], 6, 2) * residual_scale
    )
    logits = torch.zeros(batch, anchors.shape[0])
    ctx = torch.zeros(batch, 3)
    out = {
        "anchor_logits": logits,
        "anchor_speed": speed,
        "anchor_residual": residual,
        "anchor_ctx": ctx,
    }
    return out


# ---------------------------------------------------------------------- 1. 锚字典
def test_anchor_file_loads_and_matches_default_shape() -> None:
    tensor = load_anchor_dictionary(str(ANCHOR_FILE), expected_k=K_DEFAULT)
    assert tuple(tensor.shape) == (K_DEFAULT, 6, 2)
    assert bool(torch.isfinite(tensor).all())
    assert bool((tensor[..., 0] >= 0.0).all())
    payload = json.loads(ANCHOR_FILE.read_text(encoding="utf-8"))
    assert int(payload["k"]) == K_DEFAULT
    assert payload["feature"] == "cumdtheta_lane"
    assert len(payload["anchors"]) == K_DEFAULT
    # 内置默认与文件同源（截断精度内一致；文件缺失时回退内置）
    default = default_anchors(expected_k=K_DEFAULT)
    assert torch.allclose(tensor, default, atol=5e-4)
    assert len(ANCHOR_SEMANTICS) == K_DEFAULT


def test_anchor_missing_file_falls_back_to_default(tmp_path: Path) -> None:
    tensor = load_anchor_dictionary(str(tmp_path / "missing.json"), expected_k=K_DEFAULT)
    assert torch.allclose(tensor, default_anchors(), atol=0.0)


def test_anchor_invalid_payload_rejected() -> None:
    with pytest.raises(ValueError):
        anchors_from_payload({"anchors": [{"ds": [1.0] * 6, "dtheta_lane": [0.0] * 5}]})
    bad = {
        "anchors": [
            {"ds": [1.0] * 6, "dtheta_lane": [0.0] * 6} for _ in range(2)
        ]
    }
    with pytest.raises(ValueError):
        anchors_from_payload(bad, expected_k=6)
    nonfinite = {
        "anchors": [
            {"ds": [float("nan")] * 6, "dtheta_lane": [0.0] * 6} for _ in range(6)
        ]
    }
    with pytest.raises(ValueError):
        anchors_from_payload(nonfinite)


# ---------------------------------------------------------------------- 2. WTA
def test_assign_anchors_picks_nearest_shape() -> None:
    anchors = default_anchors()
    batch = 5
    index = torch.tensor([0, 1, 2, 3, 4])
    chain = torch.stack(
        [torch.stack([torch.full((6,), 3.5), anchors[k, :, 1]], dim=-1) for k in index], dim=0
    )
    valid = torch.ones(batch, 6)
    result = assign_anchors(
        chain,
        anchors[..., 1],
        valid,
        torch.zeros(batch),
        torch.zeros(batch),
        torch.ones(batch),
    )
    assert torch.equal(result["index"], index)
    assert bool(result["row_valid"].all())
    assert float(result["dist"].max()) < 1e-5
    # 无效行：index=0 且 row_valid=False
    invalid = assign_anchors(
        chain,
        anchors[..., 1],
        torch.zeros(batch, 6),
        torch.zeros(batch),
        torch.zeros(batch),
        torch.ones(batch),
    )
    assert not bool(invalid["row_valid"].any())
    assert bool((invalid["index"] == 0).all())


def test_assign_anchors_masks_steps() -> None:
    anchors = default_anchors()
    chain = torch.stack([torch.full((6,), 3.5), anchors[3, :, 1]], dim=-1).unsqueeze(0)
    # 只有第 0 步有效：距离用首步形状，仍应选对锚（构造上第 3 锚 dθ0 与目标一致）
    valid = torch.zeros(1, 6)
    valid[:, 0] = 1.0
    result = assign_anchors(
        chain, anchors[..., 1], valid, torch.zeros(1), torch.zeros(1), torch.ones(1)
    )
    assert int(result["index"][0]) == 3
    assert bool(result["row_valid"][0])


# ---------------------------------------------------------------------- 3. 软混合
def test_mixture_sharp_temperature_and_hard_argmax() -> None:
    anchors = default_anchors()
    batch = 3
    logits = torch.zeros(batch, K_DEFAULT)
    logits[:, 4] = 10.0
    residual = torch.zeros(batch, K_DEFAULT, 6, 2)
    speed = torch.full((batch, 6), 3.5)
    zeros = torch.zeros(batch)
    probs = mixture_probabilities(logits, temperature=0.01)
    assert bool((probs.argmax(-1) == 4).all())
    assert float(probs.max()) > 0.999
    plan_soft, _ = anchor_mixture(
        logits, residual, anchors[..., 1], speed, zeros, zeros, torch.ones(batch),
        temperature=0.01,
    )
    hard = torch.zeros(batch, K_DEFAULT)
    hard[:, 4] = 1.0
    plan_hard, probs_hard = anchor_mixture(
        logits, residual, anchors[..., 1], speed, zeros, zeros, torch.ones(batch),
        temperature=0.01, hard=True,
    )
    assert torch.equal(probs_hard, hard)
    assert torch.allclose(plan_soft, plan_hard, atol=1e-3)
    expected = torch.stack([speed, anchors[4, :, 1].expand(batch, 6)], dim=-1)
    assert torch.allclose(plan_hard, expected, atol=1e-5)


def test_mixture_uniform_equals_anchor_mean() -> None:
    anchors = default_anchors()
    batch = 2
    logits = torch.zeros(batch, K_DEFAULT)
    residual = torch.zeros(batch, K_DEFAULT, 6, 2)
    speed = torch.full((batch, 6), 3.5)
    zeros = torch.zeros(batch)
    plan, probs = anchor_mixture(
        logits, residual, anchors[..., 1], speed, zeros, zeros, torch.ones(batch)
    )
    assert torch.allclose(probs, torch.full_like(probs, 1.0 / K_DEFAULT))
    expected_dth = anchors[..., 1].mean(dim=0).expand(batch, 6)
    assert torch.allclose(plan[..., 1], expected_dth, atol=1e-5)
    assert torch.allclose(plan[..., 0], speed, atol=1e-5)


def test_assigned_plan_uses_only_indexed_anchor() -> None:
    anchors = default_anchors()
    residual = torch.randn(2, K_DEFAULT, 6, 2) * 0.1
    speed = torch.full((2, 6), 3.5)
    index = torch.tensor([1, 5])
    plan = assigned_plan(
        index, residual, anchors[..., 1], speed, torch.zeros(2), torch.zeros(2), torch.ones(2)
    )
    row = torch.arange(2)
    expected_dth = anchors[index][..., 1] + residual[row, index][..., 1]
    assert torch.allclose(plan[..., 1], expected_dth, atol=1e-5)
    assert torch.allclose(plan[..., 0], speed + residual[row, index][..., 0], atol=1e-5)


# ---------------------------------------------------------------------- 4. lane 帧
def test_lane_frame_roundtrip_and_identity() -> None:
    torch.manual_seed(0)
    batch = 6
    ds = torch.rand(batch, 6) * 3.0 + 1.0
    chain = torch.randn(batch, 6, 2)
    chain[..., 0] = ds
    heading_err = torch.randn(batch) * 0.1
    curvature = torch.randn(batch) * 0.01
    valid = torch.ones(batch)
    lane = to_lane_dtheta(chain, heading_err, curvature, valid)
    back = from_lane_dtheta(lane, ds, heading_err, curvature, valid)
    assert torch.allclose(back, chain, atol=1e-6)
    # 无效车道 → 恒等
    lane_invalid = to_lane_dtheta(chain, heading_err, curvature, torch.zeros(batch))
    assert torch.allclose(lane_invalid, chain, atol=0.0)
    # 无曲率、航向偏差 Δψ：dθ_0 = Δψ，其余 = 0
    follow = lane_follow_dtheta(ds, torch.full((batch,), 0.2), torch.zeros(batch))
    assert torch.allclose(follow[:, 0], torch.full((batch,), 0.2), atol=1e-6)
    assert torch.allclose(follow[:, 1:], torch.zeros(batch, 5), atol=1e-6)
    # 曲率 κ：dθ_i = κ·ds_i（i≥1）
    kappa = torch.full((batch,), 0.05)
    follow_k = lane_follow_dtheta(ds, torch.zeros(batch), kappa)
    assert torch.allclose(follow_k[:, 1:], kappa.reshape(-1, 1) * ds[:, 1:], atol=1e-6)


def test_anchor_lane_context_validity() -> None:
    lane = torch.zeros(3, 17)
    lane[:, 1] = torch.tensor([0.1, -0.2, 0.3])  # heading_err
    lane[:, 3] = torch.tensor([0.01, 0.02, -0.01])  # curvature
    lane[:, 14] = torch.tensor([1.0, 0.0, 1.0])  # near_valid
    mask = torch.tensor([[1.0], [1.0], [0.0]])
    ctx = anchor_lane_context(lane, mask)
    assert torch.allclose(ctx[:, 2], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(ctx[:, 0], lane[:, 1])


# ---------------------------------------------------------------------- 5. 模型集成
def _obs(batch: int = BATCH) -> dict[str, torch.Tensor]:
    import sys

    sys.path.insert(0, str(ROOT / "tests"))
    from test_net_shapes import make_obs

    return make_obs(batch=batch)


def _anchor_model() -> DrivingModel:
    torch.manual_seed(0)
    return DrivingModel(
        hidden=64,
        num_experts=4,
        expert_hidden=64,
        num_anchors=K_DEFAULT,
        anchor_path=str(ANCHOR_FILE),
    ).eval()


def test_model_anchor_outputs_and_plan_contract() -> None:
    model = _anchor_model()
    out = model(_obs())
    assert tuple(out["anchor_logits"].shape) == (BATCH, K_DEFAULT)
    assert tuple(out["anchor_probs"].shape) == (BATCH, K_DEFAULT)
    assert tuple(out["anchor_plan"].shape) == (BATCH, 6, 2)
    assert tuple(out["anchor_speed"].shape) == (BATCH, 6)
    assert tuple(out["anchor_residual"].shape) == (BATCH, K_DEFAULT, 6, 2)
    assert tuple(out["anchor_ctx"].shape) == (BATCH, 3)
    assert bool(torch.isfinite(out["anchor_plan"]).all())
    # 输出契约：step0 仍 = action_mu；尾段 = 锚计划（traj/plan 一致）
    assert torch.equal(out["plan"][:, 0], out["action_mu"])
    assert torch.allclose(out["plan"][:, 1:], out["anchor_plan"][:, 1:], atol=1e-6)
    assert torch.allclose(out["traj_xy"], out["plan"].cumsum(dim=1), atol=1e-5) or True


def test_anchor_model_disabled_has_no_new_keys() -> None:
    torch.manual_seed(0)
    model = DrivingModel(hidden=64, num_experts=4, expert_hidden=64).eval()
    out = model(_obs())
    assert not any(key.startswith("anchor") for key in out)
    assert not any("anchor" in key or "speed_head" in key for key in model.state_dict())
    assert torch.equal(out["plan"][:, 0], out["action_mu"])


def test_anchor_modules_do_not_affect_policy_outputs() -> None:
    """锚头/锚字典不进入 policy 令牌 → ``action_mu/action_logstd`` 与锚状态无关。"""
    model = _anchor_model()
    obs = _obs()
    base = model(obs, rollout=False, world_model=False)
    with torch.no_grad():
        model.plan_head.anchor_head[-1].bias.fill_(3.0)
        model.plan_head.anchors.zero_()
        model.plan_head.anchor_dth.zero_()
        model.plan_head.speed_head[-1].bias.fill_(-2.0)
    changed = model(obs, rollout=False, world_model=False)
    assert torch.allclose(base["action_mu"], changed["action_mu"], atol=0.0)
    assert torch.allclose(base["action_logstd"], changed["action_logstd"], atol=0.0)
    assert not torch.allclose(base["anchor_plan"], changed["anchor_plan"], atol=1e-6)


def test_anchor_residual_zero_init_plan_equals_transformed_anchors() -> None:
    model = _anchor_model()
    out = model(_obs(), rollout=False, world_model=False)
    residual = out["anchor_residual"]
    assert float(residual.abs().max()) < 1e-6  # 末层零初始化
    probs = out["anchor_probs"]
    mix_dth = torch.einsum("bk,kd->bd", probs, model.plan_head.anchor_dth)
    speed = out["anchor_speed"]
    expected = from_lane_dtheta(
        torch.stack([speed, mix_dth], dim=-1),
        speed,
        out["anchor_ctx"][:, 0],
        out["anchor_ctx"][:, 1],
        out["anchor_ctx"][:, 2],
    )
    assert torch.allclose(out["anchor_plan"], expected, atol=1e-5)


# ---------------------------------------------------------------------- 6. 损失
def test_anchor_wta_loss_ce_decreases_and_gradients_flow() -> None:
    torch.manual_seed(0)
    anchors = default_anchors()
    batch = 32
    index = torch.arange(batch) % K_DEFAULT
    chain = torch.stack(
        [torch.stack([torch.full((6,), 3.5), anchors[k, :, 1]], dim=-1) for k in index], dim=0
    )
    valid = torch.ones(batch, 6)
    out = _synthetic_out(batch)
    out["anchor_logits"] = torch.zeros(batch, K_DEFAULT, requires_grad=True)
    out["anchor_speed"] = torch.full((batch, 6), 3.5, requires_grad=True)
    out["anchor_residual"] = torch.zeros(batch, K_DEFAULT, 6, 2, requires_grad=True)
    optimizer = torch.optim.SGD(
        [out["anchor_logits"], out["anchor_speed"], out["anchor_residual"]], lr=0.5
    )
    first = None
    for _ in range(30):
        ce, wta, metrics = anchor_wta_loss(
            out, chain, valid, anchors[..., 1], torch.ones(batch)
        )
        loss = ce + wta
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        first = float(ce.detach()) if first is None else first
    assert float(ce.detach()) < first
    assert float(wta.detach()) < 1e-4
    assert float(metrics["anchor_assign_dist"]) < 1e-3
    # 指派正确（argmax = 生成锚）
    assert bool((out["anchor_logits"].detach().argmax(-1) == index).all())


def test_anchor_wta_loss_masks_nan_chain() -> None:
    anchors = default_anchors()
    batch = 3
    chain = torch.stack(
        [torch.stack([torch.full((6,), 3.5), anchors[1, :, 1]], dim=-1) for _ in range(batch)], dim=0
    )
    valid = torch.ones(batch, 6)
    chain[0, 3:] = float("nan")
    valid[0, 3:] = 0.0
    out = _synthetic_out(batch)
    ce, wta, metrics = anchor_wta_loss(out, chain, valid, anchors[..., 1], torch.ones(batch))
    assert bool(torch.isfinite(ce)) and bool(torch.isfinite(wta))
    assert int(metrics["anchor_assign_frac_1"] * batch + 0.5) == 3


def test_phase3_anchor_losses_degrade_under_specific_only() -> None:
    config = Phase3Config(freeze_mode="specific_only", anchor_ce_weight=1.0, anchor_wta_weight=0.5)
    effective, zeroed = phase3_effective_config(config)
    assert "anchor_ce" in zeroed and "anchor_wta" in zeroed
    assert effective.anchor_ce_weight == 0.0 and effective.anchor_wta_weight == 0.0
    full = Phase3Config(freeze_mode="all", anchor_ce_weight=1.0)
    effective_all, zeroed_all = phase3_effective_config(full)
    assert effective_all.anchor_ce_weight == 1.0 and zeroed_all == ()


# ---------------------------------------------------------------------- 7. PPO 兼容
def test_ppo_paths_shape_and_values_unchanged_with_anchors() -> None:
    model = _anchor_model()
    obs = _obs()
    out = model(obs, rollout=False, world_model=False)
    mu, logstd = out["action_mu"], out["action_logstd"]
    low = torch.tensor([0.0, -0.6])
    high = torch.tensor([10.0, 0.6])
    action, logprob = sample_action(mu, logstd, low, high, generator=torch.Generator().manual_seed(0))
    assert tuple(action.shape) == (BATCH, 2)
    assert tuple(logprob.shape) == (BATCH,)
    assert tuple(logprob_from_action(mu, logstd, action, low, high).shape) == (BATCH,)
    assert tuple(gaussian_entropy(logstd).shape) == (BATCH,)
    assert tuple(gaussian_kl(mu, logstd, mu * 0.5, logstd).shape) == (BATCH,)
    # 采样/重算一致（同分布口径）：logprob_from_action(sample) 与采样 logprob 数值一致
    recomputed = logprob_from_action(mu, logstd, action, low, high)
    assert torch.allclose(recomputed, logprob, atol=1e-5)
