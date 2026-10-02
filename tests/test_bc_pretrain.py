"""BC 预训练回归测试：时间对齐的轨迹目标、即时动作损失、策略头非饱和参数化。

背景（trainer 线，2026-09-25）：薄切片策略坍缩成蠕动（ds≈0.067 m/0.5 s）有三个叠加缺陷：

1. ``pretrain_bc`` 取 ``out.get("action_seq", out.get("actions"))``，而模型只输出
   ``action_mu`` → ``bc_action_loss`` 恒为 0；
2. 轨迹损失把 6 点预测对 30 点 ``traj30`` 用 ``linspace(0,29,6)`` 重采样 → 时间错位
   （t=0.1/0.7/1.3/1.9/2.5/3.0 s vs 模型的 0.5..3.0 s）→ 策略学会"更短的轨迹"；
3. tanh 压缩在 raw 偏负时梯度消失 → 一旦漂到 ``raw<-2`` 无法恢复。

本文件锁定：目标取点公式 = ``traj6``、动作损失对即时动作非零、sigmoid 参数化的
初始中点与梯度健康性、采样/logprob 与 ``PolicyHead`` 一致（PPO ratio 契约）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from net.model import DrivingModel
from net.policy import ACTION_HIGH, ACTION_LOW, PolicyHead
from pipeline.trainer import (
    BCConfig,
    BCDataset,
    SUPERVISED_LABELS,
    bc_trajectory_loss,
    logprob_from_action,
    pretrain_bc,
    sample_action,
    trajectory_target_indices,
)

COUNT = 8
BATCH = 4
HISTORY = 6
#: 数据集 schema 里 traj6 在 traj30（0.1 s 采样，t=0.1..3.0 s）中的下标（t=0.5..3.0 s）
TRAJ6_INDICES = [4, 9, 14, 19, 24, 29]


# ---------------------------------------------------------------------- 数据集
def _synthetic_dataset() -> BCDataset:
    """构造与 ``tools/collect_expert.py`` 同 schema 的小型合成数据集。

    - 专家即时动作 ds=3.0 m/0.5 s（未来步 ds=0.0，用于区分"是否只监督即时动作"）；
    - 轨迹目标：沿 +x 直线每 0.5 s 前进 3.0 m，``traj6`` 严格来自 ``traj30`` 下标。
    """
    rng = np.random.default_rng(0)
    arrays = {
        "episode_id": np.zeros(COUNT, dtype=np.int64),
        "pose": np.zeros((COUNT, 3), dtype=np.float32),
        "hist_valid": np.ones((COUNT, HISTORY), dtype=np.float32),
        "ego": rng.normal(size=(COUNT, 1, 8)).astype(np.float32),
        "ego_mask": np.ones((COUNT, 1), dtype=np.float32),
        "od": rng.normal(size=(COUNT, 16, 9)).astype(np.float32),
        "od_mask": np.ones((COUNT, 16), dtype=np.float32),
        "ld": rng.normal(size=(COUNT, 16, 7)).astype(np.float32),
        "ld_mask": np.ones((COUNT, 16), dtype=np.float32),
        "nav": rng.normal(size=(COUNT, 1, 11)).astype(np.float32),
        "nav_mask": np.ones((COUNT, 1), dtype=np.float32),
        "signal": rng.normal(size=(COUNT, 1, 4)).astype(np.float32),
        "signal_mask": np.ones((COUNT, 1), dtype=np.float32),
        "labels": np.zeros((COUNT, 8), dtype=np.float32),
        "sample_weight": np.ones(COUNT, dtype=np.float32),
    }
    # 30 点密集轨迹：t=(k+1)·0.1 s 时 x=3.0·(k+1)/5（0.5 s 走 3.0 m 的匀速直线）
    traj30 = np.zeros((COUNT, 30, 2), dtype=np.float32)
    traj30[:, :, 0] = np.arange(1, 31, dtype=np.float32)[None, :] * (3.0 / 5.0)
    arrays["traj30"] = traj30
    arrays["traj6"] = traj30[:, TRAJ6_INDICES].copy()
    action = np.zeros((COUNT, 6, 2), dtype=np.float32)
    action[:, 0, 0] = 3.0  # 即时动作（唯一受监督步）
    arrays["action"] = action
    return BCDataset(arrays, {"label_names": list(SUPERVISED_LABELS), "history_storage": "per_frame"})


# ---------------------------------------------------------------------- 缺陷 2：时间对齐
def test_trajectory_target_indices_match_dataset_traj6() -> None:
    """6 点预测 → 30 点密集目标的下标必须是 traj6 的确切来源 [4,9,14,19,24,29]。"""
    assert trajectory_target_indices(6, 30).tolist() == TRAJ6_INDICES


def test_bc_trajectory_loss_uses_time_aligned_targets() -> None:
    """pred = 真实 traj6 时，对 traj30 的损失必须为 0（旧 linspace 取点会时间错位）。"""
    dataset = _synthetic_dataset()
    traj6 = torch.as_tensor(dataset.arrays["traj6"])
    traj30 = torch.as_tensor(dataset.arrays["traj30"])

    loss, metrics = bc_trajectory_loss(traj6, traj30)
    assert float(loss) == pytest.approx(0.0, abs=1e-10)
    assert metrics["traj_mae_weighted_m"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["traj_mse_weighted"] == pytest.approx(0.0, abs=1e-10)
    assert metrics["traj_mae_all_m"] == pytest.approx(0.0, abs=1e-10)

    # 旧实现的取点（错误）确实与 traj6 不同 —— 保证上面的断言有意义
    old_idx = torch.linspace(0, 29, 6).round().long()
    assert not torch.allclose(traj30[:, old_idx], traj6)


# ---------------------------------------------------------------------- 缺陷 1：动作损失
def test_pretrain_bc_action_loss_nonzero_and_immediate_action() -> None:
    """BC 动作损失必须非零，且监督的是即时动作 ``action[:, 0, :]``。

    合成数据：即时 ds=3.0、未来 ds=0.0。零初始化策略头初始 ds=5.0 ⇒ 正确监督的
    加权 MSE（对 2 个动作维取均值）= 0.5·(5-3)²/2 = 1.0；若错取 ``action[:, 1]``
    则为 0.5·(5-0)²/2 = 6.25。
    """
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    model = DrivingModel()
    config = BCConfig(epochs=1, batch_size=BATCH, lr=3e-4, device="cpu", shuffle=False)
    metrics = pretrain_bc(model, dataset, config, logger=lambda _: None)

    assert metrics["bc_action_loss"] > 0.0, "BC 动作损失必须实际运行（缺陷 1 回归）"
    assert metrics["bc_action_loss"] == pytest.approx(1.0, abs=0.3), "动作目标必须是即时动作 action[:,0,:]"
    assert np.isfinite(metrics["bc_action_loss"])


def test_pretrain_bc_action_dim_weights_scale_dtheta_term() -> None:
    """P1 iter2：``action_dim_weights=(1, w)`` 只放大 dθ 项（ds 项不变）。

    lr=0 ⇒ 参数不动，损失可精确对账：初始 (5.0, 0)；目标 (3.0, 0.4)。
    默认 (1,1)：0.5·(2² + 0.4²)/2 = 1.04；w=8.33：0.5·(2² + 8.33·0.4²)/2 ≈ 1.333。
    """
    torch.manual_seed(0)
    dataset = _synthetic_dataset()
    dataset.arrays["action"][:, 0, 1] = 0.4

    def _loss(dim_weights):
        torch.manual_seed(0)
        model = DrivingModel()
        config = BCConfig(
            epochs=1, batch_size=BATCH, lr=0.0, device="cpu", shuffle=False,
            action_dim_weights=dim_weights,
        )
        return pretrain_bc(model, dataset, config, logger=lambda _: None)["bc_action_loss"]

    base = _loss((1.0, 1.0))
    scaled = _loss((1.0, 8.33))
    assert base == pytest.approx(1.04, abs=1e-4)
    assert scaled == pytest.approx((4.0 + 8.33 * 0.16) / 4.0, abs=1e-4)
    assert scaled > base


# ---------------------------------------------------------------------- 缺陷 3：参数化
def _tokens(batch: int = 5, hidden: int = 16, length: int = 37) -> tuple[torch.Tensor, torch.Tensor]:
    """v6 交叉注意力头的玩具令牌集合 + 全有效 key mask。"""
    return torch.randn(batch, length, hidden), torch.ones(batch, length)


def test_policy_head_init_is_midrange_and_trainable() -> None:
    """新策略头初始 raw=0 → 动作 = 界中点 (5 m, 0)，且专家工作点附近梯度不饱和。

    v6：头输入 = 令牌集合（OD/LD 逐槽 + others + ego + nav + signal + 融合 latent）+ key mask。
    """
    torch.manual_seed(0)
    policy = PolicyHead(hidden=16)
    tokens, key_mask = _tokens()
    raw_mu, _ = policy.raw(tokens, key_mask)
    assert torch.allclose(raw_mu, torch.zeros_like(raw_mu), atol=1e-6)
    action_mu, _ = policy(tokens, key_mask)
    assert torch.allclose(action_mu[:, 0], torch.full((5,), 5.0), atol=1e-5)
    assert torch.allclose(action_mu[:, 1], torch.zeros(5), atol=1e-5)
    assert torch.all(action_mu >= policy.action_low - 1e-6)
    assert torch.all(action_mu <= policy.action_high + 1e-6)

    # 专家工作点 ds=3 m（raw=logit(0.3)）：ds 对 raw 的局部导数 ≈ span·σ(1-σ) ≈ 2.1，
    # 远大于 tanh 在 raw=-2.6 处的 ≈0.05（饱和陷阱）。
    raw = torch.tensor([[float(np.log(0.3 / 0.7))]], requires_grad=True)
    ds = policy.squash(raw)[0, 0]
    grad = torch.autograd.grad(ds, raw)[0]
    assert float(grad) > 0.5, f"ds 工作点梯度饱和：{float(grad)}"

    loss = (policy(tokens, key_mask)[0][:, 0] - 3.0).pow(2).mean()
    loss.backward()
    assert policy.mu.weight.grad is not None
    assert float(policy.mu.weight.grad.abs().sum()) > 0.0
    # v6：logstd 头权重非零 ⇒ 注意力栈经 logstd 路径也有梯度（mu 零初始化不再挡死上游）
    policy.zero_grad(set_to_none=True)
    policy(tokens, key_mask)[1].sum().backward()
    assert float(policy.layers[0].attn.in_proj_weight.grad.abs().sum()) > 0.0


def test_trainer_sampling_matches_policy_head_log_prob() -> None:
    """trainer 的采样/logprob 与 ``PolicyHead`` 同分布（PPO ratio 契约）。"""
    torch.manual_seed(0)
    policy = PolicyHead(hidden=16)
    tokens, key_mask = _tokens(batch=4)
    mu, log_std = policy(tokens, key_mask)
    low = torch.tensor(ACTION_LOW)
    high = torch.tensor(ACTION_HIGH)

    action, logprob = sample_action(
        mu, log_std, low, high, mode="sigmoid_squashed", generator=torch.Generator().manual_seed(3)
    )
    assert torch.all(action >= low - 1e-6) and torch.all(action <= high + 1e-6)

    assert torch.allclose(logprob, policy.log_prob(tokens, key_mask, action), atol=1e-4)
    assert torch.allclose(
        logprob, logprob_from_action(mu, log_std, action, low, high, mode="sigmoid_squashed"), atol=1e-4
    )
