"""S1（v3）：local pool 每 episode 轮换 spec + ``off`` 旧行为回归。

锁定（依据 Step 0 实证：``--pool local --envs 1`` 旧行为整个 run 只训 spec[0]）：

- ``LocalEnvPool(spec_rotation="episode")``（默认）：终局 auto-reset 按 round-robin
  ``_index`` 轮换到下一条 spec（复用 ``_build``/``_setup_episode`` 公共路径）；
- ``spec_rotation="off"``：终局 auto-reset 复用当前 spec（旧行为）；显式 ``reset()``
  在两种口径下都轮换（旧语义不变）；
- P0-2 不回归：终局 record 取终局帧（旧 spec），新 episode 首帧只经 ``record["next_obs"]``
  返回（新 spec/env）；``_current_info`` 清空、``prev_policy_action`` 清零、``_steps`` 归零；
- S2 的 ``collect_rollout`` 场景覆盖：stub 池 record["spec_id"] → ``episode_spec_ids`` /
  ``spec_ids_seen``（逐 episode 记录）；
- ``build_pool`` 透传 ``spec_rotation``；阶段 C 解析（CLI > config > 默认 episode）与非法值 fail-fast。

不建真实 env、不 import metadrive（fake env + stub builder）。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np
import pytest
import torch

from pipeline import trainer as trainer_mod
from pipeline.stages import (
    _parse_args,
    _resolve_stage_c_spec_rotation,
)
from pipeline.trainer import LocalEnvPool, PPOConfig, PPOTrainer, build_pool


# --------------------------------------------------------------------------- #
# fake env / pool 脚手架（与 tests/test_router_labels_protocol.py 同风格）
# --------------------------------------------------------------------------- #

class _FakeBuilder:
    """obs = ``{'ego': [[env.uid, ...]]}``：uid 即当前 spec id（可观察 env 身份）。"""

    def build(self, env, spec):  # noqa: ANN001
        return {"ego": np.full((1, 8), float(env.uid), dtype=np.float32)}


class _FakeEnv:
    def __init__(self, uid: int, *, terminate_after: int = 1) -> None:
        self.uid = int(uid)
        self.agent = SimpleNamespace(
            position=np.array([float(uid), 0.0]),
            heading_theta=0.0,
            speed=5.0,
            on_white_continuous_line=False,
            on_yellow_continuous_line=False,
        )
        self.prev_policy_action = np.zeros(2)
        self.episode_step = 0
        self.steps = 0
        self.terminate_after = int(terminate_after)

    def step(self, action):  # noqa: ANN001
        self.steps += 1
        self.episode_step = self.steps
        terminated = self.steps >= self.terminate_after
        return None, 1.0, terminated, False, {"episode_steps": self.steps}

    def reset(self):
        self.steps = 0
        self.episode_step = 0
        return None, {}


def _make_pool(
    spec_ids: List[int], *, rotation: str, build_log: List[int], envs: List[_FakeEnv]
) -> LocalEnvPool:
    pool = LocalEnvPool.__new__(LocalEnvPool)
    pool.specs = [SimpleNamespace(id=int(sid)) for sid in spec_ids]
    pool.tracker_kind = "kinematic"
    pool.max_episode_steps = 50
    pool.traffic_density = None
    pool.obs_config = {}
    pool.logger = lambda *args, **kwargs: None
    pool.num_envs = 1
    pool.spec_rotation = rotation
    pool._builder = _FakeBuilder()
    pool._env = _FakeEnv(-1)
    pool._spec = None
    pool._index = -1
    pool._steps = 0
    pool._order = ()  # 无 router 标签：_record 走 labels=None 分支
    pool._tracker = None
    pool._tracker_policy = None
    pool._current_info = {}

    def _build(spec):  # noqa: ANN001
        build_log.append(int(spec.id))
        env = _FakeEnv(int(spec.id))
        envs.append(env)
        pool._env = env
        pool._spec = spec

    pool._build = _build
    return pool


_ACTION = np.array([3.0, 0.0], dtype=np.float64)


# --------------------------------------------------------------------------- #
# 轮换顺序 / off 不变
# --------------------------------------------------------------------------- #

def test_episode_rotation_round_robin_and_wrap() -> None:
    build_log: List[int] = []
    envs: List[_FakeEnv] = []
    pool = _make_pool([10, 20, 30], rotation="episode", build_log=build_log, envs=envs)

    first = pool.reset()[0]
    assert first["spec_id"] == 10 and build_log == [10]

    # 每次终局：record 取旧 spec 终局帧；next_obs 来自轮换后的新 spec/env
    expected_next = [20, 30, 10]
    previous = 10
    for want in expected_next:
        record = pool.step(_ACTION)[0]
        assert record["terminated"] is True
        assert record["spec_id"] == previous, "终局 record 必须仍是旧 spec（P0-2）"
        assert "episode_steps" in record["info"], "终局 info 取自终局帧"
        assert float(record["next_obs"]["ego"][0, 0]) == want, "next_obs 必须来自轮换后的新 env"
        previous = want

    # 三次终局轮换后游标回到 spec 10；显式 reset 继续同一游标 → 20
    assert build_log == [10, 20, 30, 10], "每次终局轮换都重建 env（_build 路径）"
    after = pool.reset()[0]
    assert after["spec_id"] == 20 and build_log[-1] == 20


def test_off_mode_keeps_spec_and_explicit_reset_still_rotates() -> None:
    build_log: List[int] = []
    envs: List[_FakeEnv] = []
    pool = _make_pool([10, 20], rotation="off", build_log=build_log, envs=envs)

    assert pool.reset()[0]["spec_id"] == 10
    for _ in range(3):
        record = pool.step(_ACTION)[0]
        assert record["terminated"] is True
        assert record["spec_id"] == 10
        assert float(record["next_obs"]["ego"][0, 0]) == 10, "off：auto-reset 复用同一 spec/env"
    assert build_log == [10], "off：auto-reset 不重建 env（旧行为）"

    assert pool.reset()[0]["spec_id"] == 20, "显式 reset 在 off 下仍轮换（旧语义）"


def test_terminal_auto_reset_clears_episode_state() -> None:
    build_log: List[int] = []
    envs: List[_FakeEnv] = []
    pool = _make_pool([10, 20], rotation="episode", build_log=build_log, envs=envs)
    pool.reset()
    pool._current_info = {"stale": True}
    pool.step(_ACTION)
    assert pool._current_info == {}, "终局 reset 后 _current_info 必须清空（P0-2）"
    assert pool._steps == 0 and pool._env.episode_step == 0, "新 episode 步计数归零"
    assert pool._env.prev_policy_action.tolist() == [0.0, 0.0], "§8.4：新 episode 上一动作清零"


def test_invalid_spec_rotation_rejected() -> None:
    with pytest.raises(ValueError):
        LocalEnvPool([SimpleNamespace(id=0)], spec_rotation="nope")


# --------------------------------------------------------------------------- #
# build_pool 透传 / 阶段 C 解析
# --------------------------------------------------------------------------- #

def test_build_pool_forwards_spec_rotation_to_local_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _SpyPool:
        def __init__(self, specs, **kwargs):  # noqa: ANN001
            captured.update(kwargs)
            self.num_envs = 1

    monkeypatch.setattr(trainer_mod, "LocalEnvPool", _SpyPool)
    build_pool([SimpleNamespace(id=0)], kind="local", num_envs=1)
    assert captured["spec_rotation"] == "episode", "默认 = episode（修正后的预期行为）"
    build_pool([SimpleNamespace(id=0)], kind="local", num_envs=1, spec_rotation="off")
    assert captured["spec_rotation"] == "off"
    with pytest.raises(ValueError):
        build_pool([SimpleNamespace(id=0)], kind="local", num_envs=1, spec_rotation="nope")


def test_stage_c_spec_rotation_resolution_cli_over_config() -> None:
    args = _parse_args(["--stage", "C"])
    assert args.spec_rotation is None
    assert _resolve_stage_c_spec_rotation(args, {}) == "episode", "默认 episode"
    assert _resolve_stage_c_spec_rotation(args, {"spec_rotation": "off"}) == "off", "config 可关"
    args_cli = _parse_args(["--stage", "C", "--spec-rotation", "off"])
    assert args_cli.spec_rotation == "off"
    assert _resolve_stage_c_spec_rotation(args_cli, {"spec_rotation": "episode"}) == "off", "CLI 优先"
    with pytest.raises(SystemExit):
        _resolve_stage_c_spec_rotation(args, {"spec_rotation": "nope"})


# --------------------------------------------------------------------------- #
# trainer 侧：每 episode spec 记录（metrics 证据通道）
# --------------------------------------------------------------------------- #

class _RotatingPoolStub:
    """每步一条记录且**每步终止**（模拟逐 episode 首帧）；spec 序 [3, 5, 7, 9]。"""

    num_envs = 1

    def __init__(self) -> None:
        self.specs = [3, 5, 7, 9]
        self.calls = 0
        self.accepts_pre_step_labels = False

    def reset(self):
        return [self._record()]

    def step(self, actions, references=None):  # noqa: ANN001
        return [self._record()]

    def _record(self) -> dict:
        spec = self.specs[self.calls % len(self.specs)]
        self.calls += 1
        return {
            "obs": {"ego": np.zeros((1, 8), dtype=np.float32)},
            "info": {},
            "reward": 0.0,
            "terminated": True,
            "truncated": False,
            "spec_id": spec,
        }


class _RewardStub:
    early_terminations = 0

    def reset_all(self) -> None:  # noqa: D102 - stub
        pass

    def step(self, env_index, info, obs, done, pool_reward, step_index=0):  # noqa: ANN001
        return float(pool_reward), {}


def test_trainer_records_episode_spec_ids() -> None:
    torch.manual_seed(0)
    model = trainer_mod._SmokeModel({"ego": np.zeros((1, 8), dtype=np.float32)})
    trainer = PPOTrainer(
        model,
        _RotatingPoolStub(),
        PPOConfig(seed=0, device="cpu", epochs=1, minibatch_size=4),
        reward_adapter=_RewardStub(),
        probe_batch=None,
    )
    assert trainer.episode_spec_ids == [] and trainer.spec_ids_seen == set()
    trainer.adopt_obs([{"ego": np.zeros((1, 8), dtype=np.float32)}])
    trainer.collect_rollout(3)
    assert trainer.episode_spec_ids == [3, 5, 7], "逐 episode 首帧记录 spec_id"
    assert trainer.spec_ids_seen == {3, 5, 7}
