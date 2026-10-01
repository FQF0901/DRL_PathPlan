"""P4 前置-B：max_step 接线（Gate2 实锤）——截断 → ``info["max_step"]`` → 终局值 −23。

锁定：

- ``LocalEnvPool`` 到 ``max_episode_steps`` 截断且 info 无终局键 → 终局 record
  ``info["max_step"]=True``（reward 口径与监视口径 ``episode_termination_reason`` 对齐）；
- ``collect_rollout`` 对 Vector/旧 stub 池的截断 record 统一兜底注入；
- 端到端（池 → collect_rollout → ``RewardAdapter``/``RewardAggregator``）：终局步
  ``terminal_key=max_step`` / ``terminal_value=−23.0``（rc=1 定稿）、episode 探针 reason 同档；
- **对照**：屏蔽注入（接线前现场）→ ``terminal_key=None`` / ``terminal_value=0.0``，
  两者奖励差 = −23（证明区分度）；
- 既有终局键（arrive_dest/crash/…）不被 ``max_step`` 覆盖。

不建 MetaDrive env、不 import metadrive（fake env + 真实奖励聚合器 + ``_SmokeModel``）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from pipeline import trainer as trainer_mod
from pipeline.trainer import (
    LocalEnvPool,
    PPOConfig,
    PPOTrainer,
    _SmokeModel,
    build_reward_adapter,
    has_terminal_outcome,
)

OBS = {"ego": np.full((1, 8), 0.5, dtype=np.float32)}
_ACTION = np.array([1.0, 0.0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# fake env / pool 脚手架（与 tests/test_spec_rotation.py 同风格）
# --------------------------------------------------------------------------- #

class _FakeBuilder:
    def build(self, env, spec):  # noqa: ANN001
        return {"ego": np.full((1, 8), 0.5, dtype=np.float32)}


class _FakeEnv:
    """永不自然终局的最小 env（截断只由池的 max_episode_steps 触发）。"""

    def __init__(self) -> None:
        self.agent = SimpleNamespace(
            position=np.array([0.0, 0.0]),
            heading_theta=0.0,
            speed=5.0,
            on_white_continuous_line=False,
            on_yellow_continuous_line=False,
        )
        self.prev_policy_action = np.zeros(2)
        self.steps = 0

    def step(self, action):  # noqa: ANN001
        self.steps += 1
        self.agent.position = np.array([float(self.steps), 0.0])
        return None, 0.0, False, False, {"route_completion": 0.0}

    def reset(self):
        self.steps = 0
        self.agent.position = np.array([0.0, 0.0])
        return None, {}


def _make_pool(*, max_episode_steps: int = 2) -> LocalEnvPool:
    pool = LocalEnvPool.__new__(LocalEnvPool)
    pool.specs = [SimpleNamespace(id=1)]
    pool.tracker_kind = "kinematic"
    pool.max_episode_steps = int(max_episode_steps)
    pool.traffic_density = None
    pool.obs_config = {}
    pool.logger = lambda *args, **kwargs: None
    pool.num_envs = 1
    pool.spec_rotation = "off"  # 固定 spec（不重建 env；截断 auto-reset 复用 fake env）
    pool._builder = _FakeBuilder()
    pool._env = _FakeEnv()
    pool._spec = pool.specs[0]
    pool._index = 0
    pool._steps = 0
    pool._order = None  # 无 router 标签（order=None → 标签计算走 None 分支）
    pool._tracker = None
    pool._tracker_policy = None
    pool._current_info = {}
    pool._build = lambda spec: None  # 显式 reset 不重建（fake env 已就位）
    return pool


class _CaptureAdapter:
    """记录每次 ``RewardAdapter.step`` 的 info/done/meta，再委托真实适配器。"""

    early_terminations = 0

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: List[Dict[str, Any]] = []

    def reset_all(self) -> None:
        self.inner.reset_all()
        self.calls.clear()

    def step(self, env_index, info, obs, done, pool_reward, step_index=None):  # noqa: ANN001
        reward, meta = self.inner.step(
            env_index, info, obs, done, pool_reward, step_index=step_index
        )
        self.calls.append(
            {
                "info": dict(info),
                "done": bool(done),
                "meta": dict(meta),
                "reward": float(reward),
            }
        )
        return reward, meta


def _make_trainer(pool: Any) -> tuple[PPOTrainer, _CaptureAdapter]:
    torch.manual_seed(0)
    inner, _ = build_reward_adapter(logger=lambda _line: None)
    capture = _CaptureAdapter(inner)
    trainer = PPOTrainer(
        _SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)}),
        pool,
        PPOConfig(epochs=1, minibatch_size=4, lr=0.0, device="cpu"),
        reward_adapter=capture,
        probe_batch=None,
        logger=lambda _line: None,
    )
    return trainer, capture


def _done_call(capture: _CaptureAdapter) -> Dict[str, Any]:
    done_calls = [call for call in capture.calls if call["done"]]
    assert len(done_calls) == 1, "小 max_episode_steps 下应恰有一个终局步"
    return done_calls[0]


# --------------------------------------------------------------------------- #
# 池层：截断注入
# --------------------------------------------------------------------------- #

def test_local_pool_truncation_injects_max_step() -> None:
    pool = _make_pool(max_episode_steps=2)
    pool.reset()
    first = pool.step(_ACTION)[0]
    assert first["truncated"] is False and "max_step" not in first["info"]
    terminal = pool.step(_ACTION)[0]
    assert terminal["truncated"] is True and terminal["terminated"] is False
    assert terminal["info"]["max_step"] is True, "截断终局 record 必须注入 max_step"
    assert "next_obs" in terminal, "终局 auto-reset 语义不变（P0-2）"


def test_mark_truncation_max_step_skips_existing_outcome() -> None:
    assert has_terminal_outcome({"max_step": True}) is True
    assert has_terminal_outcome({"crash_vehicle": True}) is True
    assert has_terminal_outcome({"route_completion": 0.5}) is False
    arrived = {"arrive_dest": True}
    assert trainer_mod._mark_truncation_max_step(arrived) is False
    assert "max_step" not in arrived, "既有终局 outcome 不得被 max_step 覆盖"
    plain: Dict[str, Any] = {}
    assert trainer_mod._mark_truncation_max_step(plain) is True
    assert plain["max_step"] is True


# --------------------------------------------------------------------------- #
# 端到端：collect_rollout → terminal_key / terminal_value
# --------------------------------------------------------------------------- #

def test_collect_rollout_truncation_terminal_value_minus_23() -> None:
    trainer, capture = _make_trainer(_make_pool(max_episode_steps=2))
    trainer.collect_rollout(3)
    call = _done_call(capture)
    assert call["info"]["max_step"] is True
    assert call["meta"]["terminal_key"] == "max_step"
    assert call["meta"]["terminal_value"] == pytest.approx(-23.0), "rc=1 定稿终局值"
    assert call["meta"]["reason"] == "max_step"
    # 监视口径与奖励口径一致（v4 episode 探针）
    assert trainer._episode_records[-1]["reason"] == "max_step"
    assert trainer._episode_records[-1]["steps"] == 2


def test_control_without_wiring_terminal_value_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """对照（接线前现场）：截断不注入 → terminal_key=None / terminal_value=0（超时变正收益）。"""
    monkeypatch.setattr(trainer_mod, "_mark_truncation_max_step", lambda info: False)
    trainer, capture = _make_trainer(_make_pool(max_episode_steps=2))
    trainer.collect_rollout(3)
    call = _done_call(capture)
    assert "max_step" not in call["info"]
    assert call["meta"]["terminal_key"] is None
    assert call["meta"]["terminal_value"] == pytest.approx(0.0)
    assert trainer._episode_records[-1]["reason"] == "max_step", "监视口径仍按 truncated 报 max_step"


def test_wiring_shifts_terminal_reward_by_minus_23() -> None:
    """同一 fake env：接线 vs 对照仅差终局值（−23），证明注入是唯一变量。"""
    pool_a = _make_pool(max_episode_steps=2)
    trainer_a, capture_a = _make_trainer(pool_a)
    trainer_a.collect_rollout(3)
    wired = _done_call(capture_a)

    original = trainer_mod._mark_truncation_max_step
    trainer_mod._mark_truncation_max_step = lambda info: False
    try:
        trainer_b, capture_b = _make_trainer(_make_pool(max_episode_steps=2))
        trainer_b.collect_rollout(3)
        control = _done_call(capture_b)
    finally:
        trainer_mod._mark_truncation_max_step = original
    assert wired["reward"] - control["reward"] == pytest.approx(-23.0)


# --------------------------------------------------------------------------- #
# Vector/旧 stub 池：collect_rollout 统一兜底
# --------------------------------------------------------------------------- #

class _TruncatingStubPool:
    """Vector 路径替身：第 2 步 ``truncated=True`` 且 info 无 max_step（接线前现场）。"""

    num_envs = 1
    accepts_pre_step_labels = False

    def __init__(self) -> None:
        self._step_index = 0

    def reset(self) -> List[Dict[str, Any]]:
        return [self._record(truncated=False)]

    def step(self, actions, references=None):  # noqa: ANN001
        record = self._record(truncated=True)
        self._step_index += 1
        return [record]

    def _record(self, *, truncated: bool) -> Dict[str, Any]:
        return {
            "obs": OBS,
            "info": {},
            "reward": 0.0,
            "terminated": False,
            "truncated": truncated,
            "spec_id": 0,
        }


def test_collect_rollout_fallback_injects_for_vector_records() -> None:
    trainer, capture = _make_trainer(_TruncatingStubPool())
    trainer.collect_rollout(1)
    call = _done_call(capture)
    assert call["info"]["max_step"] is True, "Vector 截断 record 必须在 collect_rollout 兜底注入"
    assert call["meta"]["terminal_key"] == "max_step"
    assert call["meta"]["terminal_value"] == pytest.approx(-23.0)
