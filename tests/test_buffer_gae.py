"""``pipeline.buffer``（RolloutBuffer / GAE / 历史窗口重建）纯 NumPy 单测。

覆盖契约 §4 与 §8.4 的关键行为：

- GAE(λ) 手算对照（γ/λ 显式给定）：终局不 bootstrap、截断切断、episode 边界切断、末端
  用 ``last_value`` bootstrap、``normalize`` 只改优势；
- 按帧入库：形状/掩码/容量校验，``episode``/``step_index`` 自动维护；
- 历史窗口在线重建：每 ``interval`` 帧采样、预热复制最旧帧 + ``hist_valid=0``、
  SE(2) 对齐到目标帧平面、掩码随源帧复制；
- ``episode_slices`` / ``to_arrays`` 视图。

纯 NumPy：不 import torch，不建 env（``build_history`` 只在需要时惰性 import
``env.obs`` 的对齐规则）。
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.buffer import RolloutBuffer


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #

def _obs_template() -> dict:
    return {
        "ego": (1, 8),
        "od": (2, 9),
        "ld": (2, 7),
        "nav": (1, 11),
        "signal": (1, 4),
    }


def _make_frame(seed: int = 0, *, od_point=(0.0, 0.0)) -> dict:
    rng = np.random.default_rng(seed)
    obs = {}
    for name, shape in _obs_template().items():
        obs[name] = rng.normal(size=shape).astype(np.float32)
        obs[f"{name}_mask"] = np.ones(shape[:-1], dtype=np.float32)
    obs["od"][0, 0:2] = np.asarray(od_point, dtype=np.float32)
    obs["ego_hist"] = rng.normal(size=(6, 1, 8)).astype(np.float32)  # 非当前帧键，应被忽略
    obs["hist_valid"] = np.ones(6, dtype=np.float32)
    return obs


def _add(
    buffer: RolloutBuffer,
    seed: int,
    *,
    reward: float = 0.0,
    value: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    episode: int | None = None,
    step: int | None = None,
    pose=(0.0, 0.0, 0.0),
    od_point=(0.0, 0.0),
):
    return buffer.add_step(
        _make_frame(seed, od_point=od_point),
        pose=np.asarray(pose, dtype=np.float32),
        action=np.array([1.0, 0.1], dtype=np.float32),
        logprob=-0.5,
        value=value,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        episode=episode,
        step=step,
    )


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #

def test_gae_two_step_terminal_matches_manual():
    """r=[1,2] v=[0.5,0.2] done=[0,1] γ=0.9 λ=0.8 → A=[1.976, 1.8], ret=A+v。"""
    buffer = RolloutBuffer(2, channels=_obs_template())
    _add(buffer, 0, reward=1.0, value=0.5)
    _add(buffer, 1, reward=2.0, value=0.2, terminated=True)
    advantages, returns = buffer.compute_gae(last_value=0.0, gamma=0.9, lam=0.8)
    np.testing.assert_allclose(advantages, [1.976, 1.8], atol=1e-6)
    np.testing.assert_allclose(returns, [2.476, 2.0], atol=1e-6)


def test_gae_bootstraps_at_buffer_end():
    """末帧非终局时用 last_value bootstrap：v1 的下一步价值 5.0。"""
    buffer = RolloutBuffer(2, channels=_obs_template())
    _add(buffer, 0, reward=1.0, value=0.5)
    _add(buffer, 1, reward=2.0, value=0.2)
    advantages, returns = buffer.compute_gae(last_value=5.0, gamma=0.9, lam=0.8)
    # delta1 = 2 + 0.9*5 - 0.2 = 6.3; delta0 = 1 + 0.9*0.2 - 0.5 = 0.68
    expected_1 = 6.3
    expected_0 = 0.68 + 0.9 * 0.8 * 1.0 * expected_1
    np.testing.assert_allclose(advantages, [expected_0, expected_1], atol=1e-6)
    np.testing.assert_allclose(returns, [expected_0 + 0.5, expected_1 + 0.2], atol=1e-6)


def test_gae_cuts_at_episode_boundary_and_truncation():
    """episode 变化处与中间截断处都切断 GAE（不跨 episode 传播优势）。"""
    buffer = RolloutBuffer(4, channels=_obs_template())
    _add(buffer, 0, reward=1.0, value=0.0, episode=0)
    _add(buffer, 1, reward=1.0, value=0.0, episode=0)
    _add(buffer, 2, reward=1.0, value=0.0, episode=1)  # 新 episode：边界在 1→2
    _add(buffer, 3, reward=1.0, value=0.0, episode=1, truncated=True)
    advantages, _ = buffer.compute_gae(last_value=100.0, gamma=0.9, lam=0.8)
    # episode 1：末帧截断不 bootstrap → A3=1, A2=1+0.72*1=1.72
    np.testing.assert_allclose(advantages[2:], [1.72, 1.0], atol=1e-6)
    # episode 0 的末帧（index1）后是别的 episode → 无 bootstrap，A1=1, A0=1+0.72=1.72
    np.testing.assert_allclose(advantages[:2], [1.72, 1.0], atol=1e-6)


def test_gae_normalize_only_advantages():
    buffer = RolloutBuffer(3, channels=_obs_template())
    for index in range(3):
        _add(buffer, index, reward=1.0, value=0.0)
    advantages, returns = buffer.compute_gae(last_value=0.0, gamma=1.0, lam=1.0, normalize=True)
    assert abs(float(advantages.mean())) < 1e-5
    assert abs(float(advantages.std()) - 1.0) < 1e-5
    # returns = 未归一化优势 + value = [3,2,1]（λ=γ=1、无终局、last_value=0）
    np.testing.assert_allclose(returns, [3.0, 2.0, 1.0], atol=1e-6)


# --------------------------------------------------------------------------- #
# 按帧入库
# --------------------------------------------------------------------------- #

def test_add_step_keeps_single_frame_and_ignores_hist_keys():
    buffer = RolloutBuffer(2, channels=_obs_template())
    index = _add(buffer, 3, step=7)
    assert index == 0 and len(buffer) == 1
    np.testing.assert_allclose(buffer.obs["ego"][0], _make_frame(3)["ego"])
    assert buffer.step_index[0] == 7
    assert buffer.obs_mask["od"][0].shape == (2,)
    # step 自动维护（未显式给 step 时递增）
    _add(buffer, 4)
    assert buffer.step_index[1] == 8


def test_add_step_validates_shapes_and_capacity():
    buffer = RolloutBuffer(1, channels=_obs_template())
    bad = _make_frame(0)
    bad["od"] = np.zeros((3, 9), dtype=np.float32)
    with pytest.raises(ValueError):
        buffer.add_step(bad, pose=np.zeros(3))
    _add(buffer, 0)
    with pytest.raises(BufferError):
        _add(buffer, 1)


def test_episode_slices_and_to_arrays():
    buffer = RolloutBuffer(4, channels=_obs_template())
    _add(buffer, 0)
    _add(buffer, 1)
    _add(buffer, 2, terminated=True)  # 下一帧自动进入新 episode
    _add(buffer, 3)
    assert buffer.episode_slices() == [(0, 3), (3, 4)]
    arrays = buffer.to_arrays()
    assert set(arrays) >= {"obs.ego", "mask.ego", "pose", "action", "reward", "episode", "step_index"}
    assert arrays["obs.ego"].shape == (4, 1, 8)
    assert arrays["reward"].shape == (4,)


# --------------------------------------------------------------------------- #
# 历史窗口在线重建
# --------------------------------------------------------------------------- #

def test_build_history_sampling_warmup_and_valid_mask():
    """interval=5：index=11 → 采样帧 step∈{0,5,10}，预热补 3 帧且 hist_valid 标记。"""
    buffer = RolloutBuffer(12, channels=_obs_template(), history_frames=6, history_interval=5)
    for step in range(12):
        _add(buffer, step, step=step)
    history = buffer.build_history(11)
    assert history["od_hist"].shape == (6, 2, 9)
    assert history["od_hist_mask"].shape == (6, 2)
    np.testing.assert_allclose(history["hist_valid"], [0, 0, 0, 1, 1, 1])
    # 最后三个槽位来自 index 0 / 5 / 10 的真实帧
    np.testing.assert_allclose(history["od_hist"][-1], buffer.obs["od"][10])
    np.testing.assert_allclose(history["od_hist"][-2], buffer.obs["od"][5])
    np.testing.assert_allclose(history["od_hist"][-3], buffer.obs["od"][0])
    # 预热槽复制最旧的采样帧（index 0）
    np.testing.assert_allclose(history["od_hist"][0], buffer.obs["od"][0])


def test_build_history_batch_and_episode_boundary():
    """跨 episode 只取同 episode 的采样帧；批量输入返回 batch 维。"""
    buffer = RolloutBuffer(13, channels=_obs_template(), history_frames=6, history_interval=5)
    for step in range(6):
        _add(buffer, step, step=step, episode=0)
    for step in range(7):
        _add(buffer, 100 + step, step=step, episode=1)
    history = buffer.build_history(np.array([5, 12]))
    assert history["od_hist"].shape == (2, 6, 2, 9)
    np.testing.assert_allclose(history["hist_valid"][0], [0, 0, 0, 0, 1, 1])  # episode0: step 0/5
    np.testing.assert_allclose(history["hist_valid"][1], [0, 0, 0, 0, 1, 1])  # episode1: step 0/5
    # 越界报错
    with pytest.raises(IndexError):
        buffer.build_history(99)


def test_build_history_se2_alignment_translation():
    """目标帧相对源帧平移 +x：世界点 (1,0) 在目标帧自车系下应回到 (0,0)。"""
    buffer = RolloutBuffer(2, channels=_obs_template(), history_frames=2, history_interval=1)
    _add(buffer, 0, step=0, pose=(0.0, 0.0, 0.0), od_point=(1.0, 0.0))
    _add(buffer, 1, step=1, pose=(1.0, 0.0, 0.0), od_point=(1.0, 0.0))
    history = buffer.build_history(1, frames=1)
    # frames=1 → 只有目标帧自身，不需要平移；这里验证窗口含源帧时的对齐
    history_two = buffer.build_history(1, frames=2)
    # 槽位 0 是源帧（index0，pose 原点），对齐到 index1 的 pose (1,0,0)
    np.testing.assert_allclose(history_two["od_hist"][0, 0, 0:2], [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(history["od_hist"][0, 0, 0:2], [1.0, 0.0], atol=1e-6)


def test_build_history_rotation_is_equivariant():
    """源帧与目标帧相差 90°：世界点 (1,0) 在目标帧下旋转为 (0,-1)（y 左向）。"""
    buffer = RolloutBuffer(2, channels=_obs_template(), history_frames=2, history_interval=1)
    _add(buffer, 0, step=0, pose=(0.0, 0.0, 0.0), od_point=(1.0, 0.0))
    _add(buffer, 1, step=1, pose=(0.0, 0.0, np.pi / 2), od_point=(0.0, 0.0))
    history = buffer.build_history(1, frames=2)
    # 源帧 od 的位置分量 (1,0) 应被旋转到目标系：R(-90°)·(1,0) = (0,-1)
    np.testing.assert_allclose(history["od_hist"][0, 0, 0:2], [0.0, -1.0], atol=1e-6)
    # 速度向量同样旋转
    np.testing.assert_allclose(
        history["od_hist"][0, 0, 2:4],
        [buffer.obs["od"][0, 0, 3], -buffer.obs["od"][0, 0, 2]],
        atol=1e-6,
    )
