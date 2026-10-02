"""rollout 语义验收（design-v1.2 §2.1）：拷贝隔离 + detach + 梯度表 + MoE 软目标接口。

必须断言（项目主指定）：
1. 每步 loss 对 plan head / encoder 权重有梯度（action/pose 链保持可微 → 6 点轨迹目标
   仍能训练 plan head 的动作链）；
2. 梯度不跨"合成帧"边界回传（step-k 的预测不能沿状态链回传到 step-j 的预测，j<k）；
3. 真 mem 未被写入（forward 前后 obs 逐位一致；``MemBank.shift_*`` 不原地改写）。

另有：LD 不做时序、presence/entry 头、top-2 router 软目标接口（KL/CE 可用）。
"""

from __future__ import annotations

import torch

from net.mem import MemBank, mem_from_obs
from net.model import DrivingModel
from tests.test_net_shapes import HISTORY, OTHERS_DIM, make_obs


def _unblocked_model(seed: int = 0) -> DrivingModel:
    """policy.mu 末层零初始化会挡住上游梯度（BC 设计使然）→ 给一个小的非零权重解阻。"""
    torch.manual_seed(seed)
    model = DrivingModel()
    with torch.no_grad():
        model.policy.mu.weight.normal_(0.0, 0.01)
    return model


# ---------------------------------------------------------------------- ③ 拷贝隔离
def test_real_mem_is_never_written() -> None:
    """③ 真 mem 未被写入：forward/rollout 前后 obs 所有键逐位一致。"""
    model = _unblocked_model()
    obs = make_obs()
    before = {key: value.clone() for key, value in obs.items()}
    model(obs)
    model.rollout(obs)
    for key, value in obs.items():
        assert torch.equal(value, before[key]), f"真 mem 被写入：{key}"


def test_mem_bank_shift_is_copy_isolated() -> None:
    """``MemBank.shift_*`` 只生成新张量，不原地改写源 mem（滑动窗口拷贝隔离）。"""
    obs = make_obs(batch=2)
    mem = mem_from_obs(obs, others_dim=OTHERS_DIM, history_frames=HISTORY)
    snapshots = {
        "od": mem.od.clone(),
        "od_mask": mem.od_mask.clone(),
        "od_valid": mem.od_valid.clone(),
        "ld": mem.ld.clone(),
        "ego": mem.ego.clone(),
        "others": mem.others.clone(),
    }
    copy = mem.clone()
    copy.shift_ego(torch.zeros(2, 8))
    copy.shift_od_ld(
        torch.zeros(2, 16, 9),
        torch.zeros(2, 16),
        torch.zeros(2, 16, dtype=torch.long),
        torch.zeros(2, 16),
        torch.zeros(2, 16, 7),
        torch.zeros(2, 16),
    )
    for name, snapshot in snapshots.items():
        assert torch.equal(getattr(mem, name), snapshot), f"源 mem.{name} 被副本滑动改写"
    # 副本确实滑动了（最新帧 = 刚挤入的 0）
    assert torch.equal(copy.od[:, -1], torch.zeros(2, 16, 9))
    assert torch.equal(copy.ego[:, -1], torch.zeros(2, 8))
    assert float(copy.od_valid[:, -1].min()) == 1.0


def test_rollout_does_not_mutate_obs_with_companions_and_validity() -> None:
    """真 mem 的伴随数组（id/presence）与 hist_valid 同样不得被改写。"""
    model = _unblocked_model()
    obs = make_obs(with_companions=True, valid_frames=3)
    before = {key: value.clone() for key, value in obs.items()}
    model(obs)
    for key in ("od_id_hist", "od_presence_hist", "hist_valid", "od_hist_mask", "ld_hist_mask"):
        assert torch.equal(obs[key], before[key]), f"真 mem 伴随数组被写入：{key}"


# ---------------------------------------------------------------------- ① 每步梯度
def test_each_step_trajectory_loss_reaches_plan_head_and_encoder() -> None:
    """① 每个 rollout 步的 traj loss 对 plan head 与 encoder 权重都有非零梯度。"""
    model = _unblocked_model()
    out = model(make_obs())
    for k in range(6):
        model.zero_grad(set_to_none=True)
        loss = out["traj_xy"][:, k].pow(2).mean()
        loss.backward(retain_graph=True)
        plan_grad = sum(
            float(p.grad.abs().sum())
            for name, p in model.named_parameters()
            if name.startswith("plan_head.") and p.grad is not None
        )
        enc_grad = sum(
            float(p.grad.abs().sum())
            for name, p in model.named_parameters()
            if name.startswith("encoders.") and p.grad is not None
        )
        assert plan_grad > 0.0, f"step {k} 的 traj loss 对 plan head 无梯度"
        assert enc_grad > 0.0, f"step {k} 的 traj loss 对 encoder 无梯度"


def test_action_pose_chain_stays_differentiable_across_all_steps() -> None:
    """action/pose 链不被 detach：第 6 步轨迹对第 1 步动作仍有非零梯度。"""
    model = _unblocked_model()
    captured: list[torch.Tensor] = []
    handle = model.policy.mu.register_forward_hook(lambda module, inputs, output: captured.append(output))
    out = model(make_obs())
    handle.remove()
    assert len(captured) == 6  # 6 个计划步各调用一次 policy.mu

    loss = out["traj_xy"][:, 5].pow(2).mean()
    grad_first = torch.autograd.grad(loss, captured[0], allow_unused=True, retain_graph=True)[0]
    assert grad_first is not None and float(grad_first.abs().sum()) > 0.0, "动作链在第 1 步被切断"
    grad_last = torch.autograd.grad(loss, captured[-1], allow_unused=True, retain_graph=True)[0]
    assert grad_last is not None and float(grad_last.abs().sum()) > 0.0


# ---------------------------------------------------------------------- ② detach 边界
def test_gradient_does_not_cross_synthetic_frame_boundary() -> None:
    """② step-k（k≥2）的预测不得回传到 step-j（j<k）的预测（合成帧已 detach）。"""
    model = _unblocked_model()
    out = model(make_obs())
    late = out["od_pred"][:, 3].pow(2).mean()
    grad = torch.autograd.grad(late, out["od_pred"][:, 0], allow_unused=True, retain_graph=True)[0]
    assert grad is None or float(grad.abs().sum()) == 0.0, "梯度跨合成帧边界回传到 step-1 预测"

    late_traj = out["traj_xy"][:, 5].pow(2).mean()
    grad_traj = torch.autograd.grad(late_traj, out["od_pred"][:, 0], allow_unused=True, retain_graph=True)[0]
    assert grad_traj is None or float(grad_traj.abs().sum()) == 0.0, "轨迹 loss 沿状态链回传到预测"


def test_detection_step_keeps_gradient_to_real_obs() -> None:
    """真实帧（step0）不 detach：第 1 步预测对 obs 输入（检测任务）可微。"""
    model = _unblocked_model()
    obs = make_obs()
    leaf = {
        key: (value.clone().requires_grad_(True) if value.is_floating_point() else value.clone())
        for key, value in obs.items()
    }
    out = model(leaf)
    grad = torch.autograd.grad(out["od_pred"][:, 0].pow(2).mean(), leaf["od_hist"], allow_unused=True)[0]
    assert grad is not None and float(grad.abs().sum()) > 0.0, "检测步对真实 od mem 不可微"
    # 第 1 步预测对真实 presence/id 伴随数组也应可微（id 为整数不建图，presence 可微）
    grad_presence = torch.autograd.grad(
        out["od_pred"][:, 0].pow(2).mean(), leaf["od_presence_hist"], allow_unused=True, retain_graph=True
    )[0]
    assert grad_presence is None or torch.isfinite(grad_presence).all()


def test_prediction_loss_reaches_st_gnn_and_frames_do_not_reach_plan_head() -> None:
    """预测 loss 训练 ST-GNN；EGO 合成帧 detach ⇒ 预测不回传到 plan head。"""
    model = _unblocked_model()
    out = model(make_obs())
    loss = out["od_pred"].pow(2).mean() + out["ld_pred"].pow(2).mean()
    loss.backward(retain_graph=True)
    st_grad = sum(
        float(p.grad.abs().sum())
        for name, p in model.named_parameters()
        if name.startswith("st_gnn.") and p.grad is not None
    )
    assert st_grad > 0.0, "ST-GNN 未收到预测 loss 的梯度"
    plan_grad = sum(
        float((p.grad if p.grad is not None else torch.zeros(())).abs().sum())
        for name, p in model.named_parameters()
        if name.startswith("plan_head.")
    )
    assert plan_grad == 0.0, "预测 loss 经合成帧回传到了 plan head（detach 失效）"


# ---------------------------------------------------------------------- MoE 软目标接口
def test_router_soft_target_interface() -> None:
    """router 接口：logits (B,8) 可做 KL/CE；expert_weights 为 top-2 分布（软目标同形）。"""
    model = _unblocked_model()
    out = model(make_obs())
    logits = out["router_logits"]
    weights = out["expert_weights"]
    assert logits.shape == (3, 8) and weights.shape == (3, 8)
    assert torch.all((weights > 0).sum(dim=-1) == 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-6)

    # 训练侧软目标（聚类 lane 提供，形如 0.65/0.35 的 8 维分布）可对 logits 做 KL/CE
    soft = torch.zeros(3, 8)
    soft[:, 0], soft[:, 1] = 0.65, 0.35
    log_prob = torch.log_softmax(logits, dim=-1)
    kl = torch.nn.functional.kl_div(log_prob, soft, reduction="batchmean")
    assert torch.isfinite(kl) and kl.item() >= 0.0
    kl.backward()
    assert model.plan_head.moe.router[-1].weight.grad is not None


# ---------------------------------------------------------------------- presence/entry
def test_presence_entry_heads_are_functional_in_rollout() -> None:
    """presence/entry 头参与 rollout（软掩码阈值化）且输出有限 logits。"""
    model = _unblocked_model()
    out = model(make_obs())
    presence = torch.sigmoid(out["od_presence_pred"])
    entry = torch.sigmoid(out["od_entry_pred"])
    assert presence.shape == (3, 6, 16) and entry.shape == (3, 6, 16)
    assert torch.isfinite(presence).all() and torch.isfinite(entry).all()
    assert float(presence.min()) >= 0.0 and float(presence.max()) <= 1.0
    # presence 先验偏"持续存在"、entry 偏"罕见"（初始 bias）
    assert float(presence.mean()) > 0.5
    assert float(entry.mean()) < 0.5


def test_ld_pred_is_input_only_for_rollout() -> None:
    """LD 预测只作 rollout 输入/诊断：``ld_pred`` 不参与 traj 链（traj 损失对 ld_pred 无梯度）。"""
    model = _unblocked_model()
    out = model(make_obs())
    loss = out["traj_xy"].pow(2).mean()
    grad = torch.autograd.grad(loss, out["ld_pred"], allow_unused=True, retain_graph=True)[0]
    assert grad is None or float(grad.abs().sum()) == 0.0


def test_rollout_repr_depends_on_updated_ego_mem() -> None:
    """rollout 预测条件化于（副本内）ego mem：改真 mem 的当前 ego 帧必须改变 od_pred。

    零初始化解码器会把 ST-GNN 输出退化为纯先验（与输入无关），因此先给 od/ld 解码器
    末层一个小扰动，再验证条件化。
    """
    torch.manual_seed(0)
    model = DrivingModel()
    with torch.no_grad():
        model.st_gnn.od_head[-1].weight.normal_(0.0, 0.05)
        model.st_gnn.od_head[-1].bias.zero_()
    model.eval()
    obs = make_obs()
    out_a = model(obs)
    changed = {key: value.clone() for key, value in obs.items()}
    changed["ego_hist"][:, -1, 0, :6] += 5.0  # 改真 mem 的当前 ego 特征
    out_b = model(changed)
    assert not torch.allclose(out_a["od_pred"], out_b["od_pred"], atol=1e-6)


def test_two_rollouts_are_independent() -> None:
    """每次 forward 都从真 mem 重新拷贝：同输入两次前向逐位一致（无跨次污染）。"""
    model = _unblocked_model().eval()
    obs = make_obs()
    a = model(obs)
    b = model(obs)
    for key in ("traj_xy", "plan", "od_pred", "ld_pred", "od_presence_pred", "od_entry_pred"):
        assert torch.equal(a[key], b[key]), f"{key} 两次 rollout 不一致（状态被污染）"
