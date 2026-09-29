"""R7 回归（G1 复核）：阶段 C collect 与 update 的观测口径一致（历史四通道 ego/others/od/ld）。

背景（G1 复核）：``PPOTrainer.collect_rollout`` 里 ``RolloutBuffer.history_channels`` 曾只给
``("od", "ld")``，而 RolloutBuffer 默认 / env builder / Stage A/B 物化全部用四通道
``("ego", "others", "od", "ld")``。update 侧 ``_assemble_obs_batch`` 只重建 od/ld 历史，
ego/others 历史缺失 → ``net.mem`` 回退"当前帧复制 6 帧"（与 collect 的真历史不同）→
no-op 优化器下首 update 的 ``approx_kl`` 不为 0（G1 实测 ≈ 0.384）。

回归口径：
- **强版**：monkeypatch ``optimizer.step`` 为 no-op，跑 1 个 update，断言
  ``|approx_kl| < 1e-6`` 且 ``clipfrac == 0``（权重不变 ⇒ ratio=1）；旧行为下必红。
- **弱版**：collect 后取若干帧，断言 update 侧 ``_assemble_obs_batch`` 重建的
  ``<ch>_hist``/``hist_valid`` 与 collect 同帧逐位一致，且模型前向 ``action_mu/value`` 一致。
"""

from __future__ import annotations

import numpy as np
import torch

from pipeline.trainer import PPOConfig, PPOTrainer, _to_device_obs

HORIZON = 6
_FRAMES = 6
_EGO_DIM = 8
_OTHERS_DIM = 28
_OD_SLOTS = 16
_OD_DIM = 9
_LD_DIM = 7
_HISTORY_CHANNELS = ("ego", "others", "od", "ld")


class _HistPool:
    """逐帧生成与 buffer 重建同口径的四通道历史；``pose`` 恒零 ⇒ SE(2) 对齐恒等。

    历史语义与 ``pipeline.frames.FrameLookup.build_history`` 一致：slot 0 = 最老
    （step-5）、slot -1 = 当前帧；episode 头部缺帧 → 特征全零 / mask=0 / hist_valid=0。
    """

    num_envs = 1

    def __init__(self) -> None:
        self._step = 0
        self._frames: list[dict[str, np.ndarray]] = []

    @staticmethod
    def _frame(step: int) -> dict[str, np.ndarray]:
        return {
            "ego": np.full((1, _EGO_DIM), 0.1 * (step + 1), dtype=np.float32),
            "ego_mask": np.ones((1,), dtype=np.float32),
            "others": np.full((1, _OTHERS_DIM), 0.02 * (step + 1), dtype=np.float32),
            "others_mask": np.ones((1,), dtype=np.float32),
            "od": np.full((_OD_SLOTS, _OD_DIM), 0.01 * (step + 1), dtype=np.float32),
            "od_mask": np.ones((_OD_SLOTS,), dtype=np.float32),
            "ld": np.full((_OD_SLOTS, _LD_DIM), 0.03 * (step + 1), dtype=np.float32),
            "ld_mask": np.ones((_OD_SLOTS,), dtype=np.float32),
        }

    def _obs(self, step: int) -> dict[str, np.ndarray]:
        obs = dict(self._frames[step])
        for name, feature_shape, mask_shape in (
            ("ego", (1, _EGO_DIM), (1,)),
            ("others", (1, _OTHERS_DIM), (1,)),
            ("od", (_OD_SLOTS, _OD_DIM), (_OD_SLOTS,)),
            ("ld", (_OD_SLOTS, _LD_DIM), (_OD_SLOTS,)),
        ):
            history = np.zeros((_FRAMES, ) + feature_shape, dtype=np.float32)
            history_mask = np.zeros((_FRAMES, ) + mask_shape, dtype=np.float32)
            for slot in range(_FRAMES):
                source = step - (_FRAMES - 1 - slot)
                if source < 0:
                    continue
                history[slot] = self._frames[source][name]
                history_mask[slot] = self._frames[source][f"{name}_mask"]
            obs[f"{name}_hist"] = history
            obs[f"{name}_hist_mask"] = history_mask
        obs["hist_valid"] = np.array(
            [1.0 if step - (_FRAMES - 1 - slot) >= 0 else 0.0 for slot in range(_FRAMES)],
            dtype=np.float32,
        )
        obs["pose"] = np.zeros(3, dtype=np.float32)
        return obs

    def _record(self) -> dict:
        frame = self._frame(self._step)
        self._frames.append(frame)
        obs = self._obs(self._step)
        self._step += 1
        return {"obs": obs, "info": {}, "reward": 0.0, "terminated": False, "truncated": False}

    def reset(self) -> list[dict]:
        return [self._record()]

    def step(self, actions: np.ndarray, references=None) -> list[dict]:
        return [self._record()]


class _HistoryDependentModel(torch.nn.Module):
    """最小 PPO 模型：动作/价值显式消费 ``ego_hist``/``others_hist``（缺失时按 net.mem 回退复制当前帧）。"""

    def __init__(self) -> None:
        super().__init__()
        self.head_mu = torch.nn.Linear(_EGO_DIM + _OTHERS_DIM, 2)
        self.head_logstd = torch.nn.Parameter(torch.full((2,), -1.0))
        self.head_value = torch.nn.Linear(_EGO_DIM + _OTHERS_DIM, 1)

    @staticmethod
    def _mem(obs: dict, name: str) -> torch.Tensor:
        current = obs[name].reshape(obs[name].shape[0], -1)
        history = obs.get(f"{name}_hist")
        if history is None:
            # net.mem 回退：当前帧复制 6 帧（旧 history_channels=("od","ld") 的 update 侧行为）
            return current.unsqueeze(1).expand(-1, _FRAMES, -1).mean(dim=1)
        return history.reshape(history.shape[0], history.shape[1], -1).mean(dim=1)

    def forward(  # noqa: D102 - stub 契约
        self, obs: dict, *, rollout: bool = True, world_model: bool = True, wm_detach: bool = False
    ) -> dict:
        x = torch.cat([self._mem(obs, "ego"), self._mem(obs, "others")], dim=-1)
        mu = torch.sigmoid(self.head_mu(x))
        return {
            "action_mu": mu,
            "action_logstd": self.head_logstd.clamp(-5.0, 0.0).unsqueeze(0).expand(x.shape[0], -1),
            "value": self.head_value(x),
            "plan": mu.unsqueeze(1).expand(-1, 6, -1).contiguous(),
        }


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index: int, info: dict, obs: dict, done: bool, pool_reward: float, step_index: int = 0):
        return float(pool_reward), {}


def _trainer() -> PPOTrainer:
    torch.manual_seed(0)
    return PPOTrainer(
        _HistoryDependentModel(),
        _HistPool(),
        PPOConfig(seed=0, device="cpu", epochs=2, minibatch_size=4),
        reward_adapter=_RewardStub(),
        probe_batch=None,
    )


def _forward(model: torch.nn.Module, batch: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = model(_to_device_obs(batch, torch.device("cpu")), rollout=False, world_model=False)
    return out["action_mu"].detach().cpu().numpy(), out["value"].detach().cpu().numpy()


def test_noop_update_has_zero_approx_kl_and_clipfrac(monkeypatch) -> None:
    """强版：collect≠update 口径差 ⇒ no-op 优化器下 approx_kl ≠ 0（旧行为）；修复后必须归零。"""
    trainer = _trainer()
    trainer.collect_rollout(HORIZON)
    assert trainer.buffer is not None
    assert trainer.buffer.history_channels == _HISTORY_CHANNELS, (
        f"PPO buffer 必须与 RolloutBuffer 默认/ builder / 物化同口径（四通道），"
        f"收到 {trainer.buffer.history_channels}"
    )
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, *args, **kwargs: None)
    metrics = trainer.update()
    assert abs(float(metrics["approx_kl"])) < 1e-6, (
        f"no-op 优化器下 update 的观测与 collect 不一致：approx_kl={metrics['approx_kl']}"
    )
    assert float(metrics["clipfrac"]) == 0.0


def test_update_obs_rebuild_matches_collect_frame() -> None:
    """弱版：update 侧重建的历史/前向与 collect 同帧逐位一致。"""
    trainer = _trainer()
    collect_batches: list[dict[str, np.ndarray]] = []
    original = PPOTrainer._to_tensor_obs

    def spy(self, obs_list):  # noqa: ANN001 - 测试探针
        batch = original(self, obs_list)
        collect_batches.append({key: value.detach().cpu().clone() for key, value in batch.items()})
        return batch

    PPOTrainer._to_tensor_obs = spy
    try:
        trainer.collect_rollout(HORIZON)
    finally:
        PPOTrainer._to_tensor_obs = original
    assert trainer.buffer is not None
    assert trainer.buffer.history_channels == _HISTORY_CHANNELS

    for frame in (3, HORIZON - 1):
        assembled = trainer._assemble_obs_batch(np.array([frame]))
        collected = {key: value.numpy() for key, value in collect_batches[frame].items()}
        for name in _HISTORY_CHANNELS:
            assert f"{name}_hist" in assembled, f"update 重建缺少 {name}_hist（{frame}）"
            assert np.allclose(assembled[f"{name}_hist"], collected[f"{name}_hist"], atol=0.0), name
            assert np.allclose(
                assembled[f"{name}_hist_mask"], collected[f"{name}_hist_mask"], atol=0.0
            ), f"{name}_hist_mask"
        assert np.allclose(assembled["hist_valid"], collected["hist_valid"], atol=0.0)
        mu_update, value_update = _forward(trainer.model, assembled)
        mu_collect, value_collect = _forward(trainer.model, collected)
        assert np.allclose(mu_update, mu_collect, atol=1e-6), f"frame {frame}: action_mu 不一致"
        assert np.allclose(value_update, value_collect, atol=1e-6), f"frame {frame}: value 不一致"
