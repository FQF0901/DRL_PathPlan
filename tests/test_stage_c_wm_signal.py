"""W1 回归：阶段 C 的 WM（``st_gnn.*``）全期冻结 ⇒ 无训练信号。

锁定 ``apply_freeze_prefixes(("st_gnn.",))`` + 一次 PPO ``update()`` 的语义：

- ``st_gnn.*`` 参数在 update 前后 ``torch.equal``（逐位不变），且梯度为 ``None``；
- 其余参数确实被更新（证明 update 真实发生）；
- ``stages._wm_trainable`` fail-fast 判据：解冻后立刻为 True（阶段 C 期望恒 False）。
"""

from __future__ import annotations

import numpy as np
import torch

from pipeline.buffer import RolloutBuffer
from pipeline.stages import _wm_trainable
from pipeline.trainer import PPOConfig, PPOTrainer, apply_freeze_prefixes

COUNT = 8
BATCH = 4


class _StageCModel(torch.nn.Module):
    """最小 PPO 模型 + ``st_gnn`` 子模块（参数名前缀可被 apply_freeze_prefixes 识别）。"""

    def __init__(self) -> None:
        super().__init__()
        self.st_gnn = torch.nn.Linear(8, 8)
        self.head_mu = torch.nn.Linear(8, 2)
        self.head_logstd = torch.nn.Parameter(torch.full((2,), -1.0))
        self.head_value = torch.nn.Linear(8, 1)
        self.head_router = torch.nn.Linear(8, 8)
        self.head_traj = torch.nn.Linear(8, 12)

    def forward(  # noqa: D102 - stub 契约
        self, obs: dict, *, rollout: bool = True, world_model: bool = True, wm_detach: bool = False
    ) -> dict:
        x = obs["ego"].flatten(start_dim=1)
        latent = self.st_gnn(x)
        mu = torch.sigmoid(self.head_mu(latent))
        logstd = self.head_logstd.clamp(-5.0, 0.0).unsqueeze(0).expand(x.shape[0], -1)
        return {
            "action_mu": mu,
            "action_logstd": logstd,
            "value": self.head_value(latent),
            "plan": mu.unsqueeze(1).expand(-1, 6, -1).contiguous(),
            "router_logits": self.head_router(latent),
            "latent": latent,
        }


def _fill_trainer(model: torch.nn.Module) -> PPOTrainer:
    trainer = PPOTrainer(
        model,
        type("_PoolStub", (), {"num_envs": 1})(),
        PPOConfig(epochs=2, minibatch_size=BATCH, lr=1e-3, device="cpu", normalize_advantage=True),
        probe_batch=None,
    )
    buffer = RolloutBuffer(
        COUNT,
        channels={"ego": (1, 8)},
        history_frames=6,
        history_interval=1,
        history_channels=(),
    )
    rng = np.random.default_rng(0)
    for step in range(COUNT):
        buffer.add_step(
            {"ego": rng.normal(size=(1, 8)).astype(np.float32), "ego_mask": np.ones((1,), dtype=np.float32)},
            pose=np.zeros(3, dtype=np.float32),
            action=np.array([3.0, 0.0], dtype=np.float32),
            logprob=0.0,
            value=float(rng.normal()),
            reward=float(rng.normal()),
            episode=0,
            step=step,
        )
    trainer.buffer = buffer
    trainer._valid_mask = np.ones(COUNT, dtype=bool)
    trainer._router_labels = np.zeros((COUNT, 8), dtype=np.float32)
    trainer._has_router_labels = np.zeros(COUNT, dtype=bool)
    return trainer


def test_wm_unchanged_after_one_update() -> None:
    torch.manual_seed(0)
    model = _StageCModel()
    frozen = apply_freeze_prefixes(model, ("st_gnn.",))
    assert "st_gnn.weight" in frozen and "st_gnn.bias" in frozen
    assert not _wm_trainable(model)

    trainer = _fill_trainer(model)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    metrics = trainer.update()

    wm_names = [name for name, _ in model.named_parameters() if name.startswith("st_gnn.")]
    assert wm_names
    for name, parameter in model.named_parameters():
        if name in wm_names:
            assert parameter.grad is None, f"{name} 梯度非 None（WM 不应有信号）"
            assert torch.equal(before[name], parameter), f"{name} 被更新（冻结失效）"
            assert not parameter.requires_grad, f"{name} requires_grad 被恢复为 True"
    # 参与损失的策略/value 头确实被更新（证明 update 真实发生；router/traj 头无损失项不检查）
    for name in ("head_mu.weight", "head_mu.bias", "head_logstd", "head_value.weight", "head_value.bias"):
        assert not torch.equal(before[name], dict(model.named_parameters())[name]), f"{name} 未更新"
    assert metrics["critic_warmup"] is False
    assert np.isfinite(metrics["total_loss"])


def test_wm_trainable_flag_tracks_freeze() -> None:
    model = _StageCModel()
    assert _wm_trainable(model)
    apply_freeze_prefixes(model, ("st_gnn.",))
    assert not _wm_trainable(model)
    apply_freeze_prefixes(model, ())
    assert _wm_trainable(model)
