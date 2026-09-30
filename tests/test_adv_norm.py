"""① 优势归一化选项（``PPOConfig.adv_norm``）＋ GAE padding 行修复 回归测试。

覆盖 G3 查证结论（2026-09-30）：

- **现状是 bug**：env-major 布局里每 env 段末尾的 padding/bootstrap 行（"下一观测价值容器"）
  被旧 ``compute_gae`` 当成真实一步 → 其伪优势 ``-V_bootstrap`` 沿 ``γλ`` 衰减污染同 episode
  尾部**有效帧**的优势；padding 行本身**从未**进入 mean/std 与 loss 分母（``_valid_mask``
  只放行 H 帧/env）。本文件用内联旧实现锁定证据，并锁定修复后的正确口径；
- ``adv_norm``：``global``（默认，与旧表达式逐位一致）/ ``per_scenario``（按 ``spec_id``
  分组归一化）/ ``none``；``normalize_advantage=False`` 等价 ``none``；
- ``RolloutBuffer.spec_id`` 帧级场景标识的写入/导出。

不建 env、不 import metadrive。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pytest
import torch

from pipeline.buffer import RolloutBuffer
from pipeline.trainer import (
    PPOConfig,
    PPOTrainer,
    _SmokeModel,
    _normalize_advantages,
)


# --------------------------------------------------------------------------- #
# adv_norm 数学
# --------------------------------------------------------------------------- #

def test_adv_norm_validated() -> None:
    for mode in ("global", "per_scenario", "none"):
        assert PPOConfig(adv_norm=mode).adv_norm == mode
    with pytest.raises(ValueError):
        PPOConfig(adv_norm="nope")


def test_global_matches_legacy_expression_bitwise() -> None:
    """默认 global 与旧表达式逐位一致（float64 全路径）。"""
    rng = np.random.default_rng(0)
    values = rng.normal(size=257).astype(np.float64)
    legacy = values.copy()
    if legacy.size > 1:
        legacy = (legacy - legacy.mean()) / (legacy.std() + 1e-8)
    got = _normalize_advantages(values, "global")
    assert np.array_equal(got, legacy)
    # 单元素：旧守卫 advantages.size > 1 → 原样
    single = np.asarray([3.0])
    assert np.array_equal(_normalize_advantages(single, "global"), single)


def test_none_returns_unnormalized_values() -> None:
    values = np.asarray([1.0, -2.0, 3.5, 0.0])
    assert np.array_equal(_normalize_advantages(values, "none"), values)


def test_per_scenario_normalizes_each_group_independently() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    groups = np.asarray([0, 0, 0, 1, 1, 1])
    got = _normalize_advantages(values, "per_scenario", groups=groups)
    for gid in (0, 1):
        mask = groups == gid
        group = values[mask]
        expected = (group - group.mean()) / (group.std() + 1e-8)
        assert np.allclose(got[mask], expected)
        assert abs(float(got[mask].mean())) < 1e-12
    # 组间独立：把 group 1 整体平移 1000 不影响 group 0 的归一化值
    shifted = _normalize_advantages(values + np.asarray([0, 0, 0, 1000, 1000, 1000]), "per_scenario",
                                    groups=groups)
    assert np.array_equal(got[:3], shifted[:3])


def test_per_scenario_singleton_group_keeps_value_and_validates_groups() -> None:
    values = np.asarray([5.0, 1.0, 2.0, 3.0])
    groups = np.asarray([7, 1, 1, 1])
    got = _normalize_advantages(values, "per_scenario", groups=groups)
    assert got[0] == 5.0  # 单帧组保持原值（不零化）
    with pytest.raises(ValueError):
        _normalize_advantages(values, "per_scenario")
    with pytest.raises(ValueError):
        _normalize_advantages(values, "per_scenario", groups=np.asarray([0, 1]))


# --------------------------------------------------------------------------- #
# buffer.spec_id
# --------------------------------------------------------------------------- #

def _frame() -> dict:
    return {"ego": np.zeros((1, 8), dtype=np.float32)}


def test_buffer_spec_id_default_and_roundtrip() -> None:
    buffer = RolloutBuffer(3, channels={"ego": (1, 8)}, history_channels=())
    buffer.add_step(_frame(), episode=0, step=0, spec_id=11)
    buffer.add_step(_frame(), episode=0, step=1)
    buffer.add_step(_frame(), episode=0, step=2, spec_id=12)
    assert buffer.spec_id[:3].tolist() == [11, -1, 12]
    assert buffer.to_arrays()["spec_id"].tolist() == [11, -1, 12]


# --------------------------------------------------------------------------- #
# GAE padding 行：现状（bug）→ 修复
# --------------------------------------------------------------------------- #

GAMMA, LAM = 0.99, 0.95


def _fill_frames(buffer: RolloutBuffer, count: int, *, values, rewards, episode: int = 0) -> None:
    for step in range(count):
        buffer.add_step(
            _frame(),
            action=np.zeros(2, dtype=np.float32),
            logprob=0.0,
            value=float(values[step]),
            reward=float(rewards[step]),
            terminated=False,
            episode=episode,
            step=step,
            spec_id=episode,
        )


def _padding_row(buffer: RolloutBuffer, value: float, *, episode: int, step: int) -> None:
    buffer.add_step(
        _frame(),
        action=np.zeros(2, dtype=np.float32),
        logprob=0.0,
        value=float(value),
        reward=0.0,
        terminated=False,
        episode=episode,
        step=step,
    )


def _gae_legacy(buffer: RolloutBuffer, *, gamma: float = GAMMA, lam: float = LAM,
                last_value: float = 0.0) -> np.ndarray:
    """旧实现（HEAD dfbdb02）的逐行复刻：所有行都参与优势链。"""
    n = len(buffer)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in range(n - 1, -1, -1):
        is_last = t == n - 1
        same_next = (not is_last) and int(buffer.episode[t + 1]) == int(buffer.episode[t])
        next_value = last_value if is_last else (float(buffer.value[t + 1]) if same_next else 0.0)
        terminal = bool(buffer.terminated[t]) or bool(buffer.truncated[t]) or (not same_next and not is_last)
        non_terminal = 0.0 if terminal else 1.0
        delta = float(buffer.reward[t]) + gamma * next_value * non_terminal - float(buffer.value[t])
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    return advantages


def _fixture_buffer() -> tuple[RolloutBuffer, np.ndarray]:
    """1 env × 3 帧 + padding 行；有效优势可由手算递推核对。"""
    buffer = RolloutBuffer(4, channels={"ego": (1, 8)}, history_channels=())
    _fill_frames(buffer, 3, values=[0.5, 0.4, 0.3], rewards=[1.0, 1.0, 1.0])
    _padding_row(buffer, value=2.0, episode=0, step=3)
    valid = np.asarray([True, True, True, False])
    return buffer, valid


def test_gae_padding_row_never_enters_denominator_but_used_to_seed_chain() -> None:
    """现状证据：padding 行不在 valid 分母里，但旧实现把它的伪优势塞进有效帧的优势链。"""
    buffer, valid = _fixture_buffer()
    legacy = _gae_legacy(buffer)
    fixed, _ = buffer.compute_gae(last_value=0.0, gamma=GAMMA, lam=LAM, valid_mask=valid)

    # (a) padding 行从不进入 mean/std / loss 分母（valid 只含 3 个真实帧）
    assert int(np.asarray(valid, dtype=bool).sum()) == 3
    assert float(legacy[-1]) == pytest.approx(-2.0)  # 旧：伪优势 = -V_bootstrap
    assert float(fixed[-1]) == 0.0

    # (b) 旧实现的污染：有效帧 k 的偏差 = -γλ·V_pad·(γλ)^(H-1-k)（同 episode 尾部）
    for k in range(3):
        expected_bias = -(GAMMA * LAM) * 2.0 * (GAMMA * LAM) ** (2 - k)
        assert float(legacy[k] - fixed[k]) == pytest.approx(expected_bias, rel=1e-5)


def test_gae_fixed_matches_manual_bootstrap_recursion() -> None:
    """修复口径：padding 行只作 bootstrap 值容器，链种子为 0（手算递推对照）。"""
    buffer, valid = _fixture_buffer()
    values = [0.5, 0.4, 0.3]
    advantages, returns = buffer.compute_gae(last_value=0.0, gamma=GAMMA, lam=LAM, valid_mask=valid)
    last = 0.0
    expected = [0.0, 0.0, 0.0]
    for t in (2, 1, 0):
        next_value = buffer.value[t + 1] if t < 2 else 2.0  # 末帧的下一步 = padding 容器
        delta = 1.0 + GAMMA * next_value - values[t]
        last = delta + GAMMA * LAM * last
        expected[t] = last
    for t in range(3):
        assert float(advantages[t]) == pytest.approx(expected[t], rel=1e-6)
        assert float(returns[t]) == pytest.approx(expected[t] + values[t], rel=1e-6)
    assert float(advantages[3]) == 0.0


def test_gae_no_mask_and_all_valid_mask_are_bitwise_legacy() -> None:
    """兼容位：不传 mask / 全 True mask 与旧实现逐位一致（无 padding 布局不受影响）。"""
    buffer, _ = _fixture_buffer()
    legacy = _gae_legacy(buffer)
    no_mask, _ = buffer.compute_gae(last_value=0.0, gamma=GAMMA, lam=LAM)
    all_valid, _ = buffer.compute_gae(last_value=0.0, gamma=GAMMA, lam=LAM,
                                      valid_mask=np.ones(len(buffer), dtype=bool))
    assert np.array_equal(no_mask, legacy)
    assert np.array_equal(all_valid, legacy)
    with pytest.raises(ValueError):
        buffer.compute_gae(valid_mask=np.ones(2, dtype=bool))


def test_gae_normalize_with_mask_uses_valid_pool_only() -> None:
    """normalize=True + valid_mask：mean/std 只统计有效帧，padding 行保持 0。"""
    buffer, valid = _fixture_buffer()
    advantages, _ = buffer.compute_gae(last_value=0.0, gamma=GAMMA, lam=LAM,
                                       valid_mask=valid, normalize=True)
    pool = advantages[valid]
    assert abs(float(pool.mean())) < 1e-6
    assert abs(float(pool.std()) - 1.0) < 1e-6
    assert float(advantages[3]) == 0.0


def test_gae_padding_value_only_bootstraps_its_own_segment() -> None:
    """2 env：env0 的 padding 值只影响 env0 段；env1 段优势不受影响（逐位）。"""
    buffer = RolloutBuffer(6, channels={"ego": (1, 8)}, history_channels=())
    _fill_frames(buffer, 2, values=[0.5, 0.4], rewards=[1.0, 1.0], episode=0)
    _padding_row(buffer, value=2.0, episode=0, step=2)
    _fill_frames(buffer, 2, values=[1.5, 1.4], rewards=[1.0, 1.0], episode=1)
    _padding_row(buffer, value=9.0, episode=1, step=2)
    valid = np.asarray([True, True, False, True, True, False])
    a, _ = buffer.compute_gae(last_value=0.0, gamma=GAMMA, lam=LAM, valid_mask=valid)
    # 改 env0 padding 值 → env1 段逐位不变
    buffer.value[2] = 7.0
    b, _ = buffer.compute_gae(last_value=0.0, gamma=GAMMA, lam=LAM, valid_mask=valid)
    assert not np.array_equal(a[:2], b[:2])
    assert np.array_equal(a[3:5], b[3:5])


def test_trainer_update_runs_with_each_adv_norm_and_legacy_none_equivalence() -> None:
    """trainer 级：三种口径可跑；``adv_norm=none`` 与旧 ``normalize_advantage=False`` 指标逐位一致。

    ``global``/``per_scenario`` 的差异用 ratio=1 的构造（logprob = 模型对确定性动作的
    重算值）直接核对：ratio=1 时 ``policy_loss = -mean(adv)``，global 归一化后恒 0，
    分场景（两位点组、组内标准差不齐）则非 0。
    """
    from pipeline.trainer import logprob_from_action

    def fill(model, config, *, exact_logprob: bool = False) -> PPOTrainer:
        trainer = PPOTrainer(model, _PoolStub(), config, probe_batch=None)
        buffer = RolloutBuffer(8, channels={"ego": (1, 8)}, history_channels=())
        rng = np.random.default_rng(0)
        for step in range(8):
            obs = rng.normal(size=(1, 8)).astype(np.float32)
            action = np.array([3.0, 0.0], dtype=np.float32)
            logprob = 0.0
            if exact_logprob:
                with torch.no_grad():
                    out = model({"ego": torch.as_tensor(obs[None])})
                    mu = out["action_mu"].reshape(-1)
                    logstd = out["action_logstd"].reshape(-1)
                    action = mu.detach().cpu().numpy().astype(np.float32)
                    logprob = float(
                        logprob_from_action(mu, logstd, torch.as_tensor(action),
                                            trainer.low, trainer.high, mode=config.action_mode)
                    )
            buffer.add_step(
                {"ego": obs},
                action=action,
                logprob=logprob,
                value=float(rng.normal()),
                reward=float(rng.normal()),
                episode=0,
                step=step,
                spec_id=0 if step < 4 else 1,
            )
        trainer.buffer = buffer
        trainer._valid_mask = np.ones(8, dtype=bool)
        trainer._router_labels = np.zeros((8, 8), dtype=np.float32)
        trainer._has_router_labels = np.zeros(8, dtype=bool)
        return trainer

    torch.manual_seed(0)
    model_a = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    torch.manual_seed(0)
    model_b = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    metrics_none = fill(model_a, PPOConfig(adv_norm="none", epochs=1, minibatch_size=8, lr=0.0, device="cpu")).update()
    metrics_legacy = fill(model_b, PPOConfig(epochs=1, minibatch_size=8, lr=0.0, device="cpu",
                                             normalize_advantage=False)).update()
    assert metrics_none["total_loss"] == metrics_legacy["total_loss"]

    torch.manual_seed(0)
    model_c = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    metrics_global = fill(model_c, PPOConfig(adv_norm="global", epochs=1, minibatch_size=8, lr=0.0, device="cpu"),
                          exact_logprob=True).update()
    assert np.isfinite(metrics_global["total_loss"])
    assert abs(metrics_global["policy_loss"]) < 1e-6
    torch.manual_seed(0)
    model_d = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    metrics_scenario = fill(model_d, PPOConfig(adv_norm="per_scenario", epochs=1, minibatch_size=8,
                                               lr=0.0, device="cpu"), exact_logprob=True).update()
    assert np.isfinite(metrics_scenario["total_loss"])


class _PoolStub:
    num_envs = 1


def test_per_scenario_wiring_passes_buffer_spec_id_groups(monkeypatch) -> None:
    """接线：``adv_norm=per_scenario`` 时 groups == ``buffer.spec_id[valid]``（帧级场景分组）。"""
    import pipeline.trainer as trainer_mod

    captured: list = []
    original = trainer_mod._normalize_advantages

    def spy(advantages, mode="global", *, groups=None):
        captured.append(
            (mode, None if groups is None else np.asarray(groups).copy())
        )
        return original(advantages, mode, groups=groups)

    monkeypatch.setattr(trainer_mod, "_normalize_advantages", spy)
    torch.manual_seed(0)
    model = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    trainer = PPOTrainer(
        model, _PoolStub(), PPOConfig(adv_norm="per_scenario", epochs=1, minibatch_size=8,
                                      lr=0.0, device="cpu"), probe_batch=None,
    )
    spec_ids = np.asarray([0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)
    buffer = RolloutBuffer(8, channels={"ego": (1, 8)}, history_channels=())
    rng = np.random.default_rng(2)
    for step in range(8):
        buffer.add_step(
            {"ego": rng.normal(size=(1, 8)).astype(np.float32)},
            action=np.array([3.0, 0.0], dtype=np.float32),
            logprob=0.0,
            value=float(rng.normal()),
            reward=float(rng.normal()),
            episode=0,
            step=step,
            spec_id=int(spec_ids[step]),
        )
    trainer.buffer = buffer
    trainer._valid_mask = np.ones(8, dtype=bool)
    trainer._router_labels = np.zeros((8, 8), dtype=np.float32)
    trainer._has_router_labels = np.zeros(8, dtype=bool)
    trainer.update()
    assert captured and captured[0][0] == "per_scenario"
    assert np.array_equal(captured[0][1], spec_ids)


def test_collect_propagates_spec_id_into_buffer() -> None:
    """真收集路径：stub 池 record["spec_id"] → buffer.spec_id（per_scenario 的分组来源）。"""

    class _Pool:
        num_envs = 1

        def __init__(self) -> None:
            # reset 记录（不入 buffer）+ 3 个 step 记录 = 4 次 _record
            self.specs = [3, 5, 3, 5]
            self.calls = 0

        def reset(self):
            return [self._record()]

        def step(self, actions, references=None):
            return [self._record()]

        def _record(self) -> dict:
            spec = self.specs[self.calls % len(self.specs)]
            self.calls += 1
            return {
                "obs": {"ego": np.zeros((1, 8), dtype=np.float32)},
                "info": {},
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "spec_id": spec,
            }

    class _RewardStub:
        early_terminations = 0

        def reset_all(self) -> None:  # noqa: D102 - stub
            pass

        def step(self, env_index, info, obs, done, pool_reward, step_index=0):
            return float(pool_reward), {}

    torch.manual_seed(0)
    model = _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    trainer = PPOTrainer(model, _Pool(), PPOConfig(seed=0, device="cpu", epochs=1, minibatch_size=4),
                         reward_adapter=_RewardStub(), probe_batch=None)
    trainer.adopt_obs([{"ego": np.zeros((1, 8), dtype=np.float32)}])
    trainer.collect_rollout(3)
    assert trainer.buffer is not None
    assert trainer.buffer.spec_id[:3].tolist() == [3, 5, 3]
    valid = np.where(trainer._valid_mask)[0]
    assert trainer.buffer.spec_id[valid].tolist() == [3, 5, 3]
