"""P0-2 协议回归：LocalEnvPool 步前标签 + 终局 record/reset 分离；Vector worker 步开始快照。

- ``LocalEnvPool.labels_now()``：不推进状态、反映当前（步前）帧；
- 终局 record 的 obs/pose/info 取自终局帧，reset 后的新 episode 首帧只经 ``next_obs`` 返回；
- 终局 reset 清空 ``_current_info``：下一 episode 首帧不得复用上一 episode 的 step info；
- ``ResidentEnv.step`` 记录 ``labels_at_step_start``（子步推进前快照）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipeline import trainer as trainer_mod
from pipeline.trainer import LocalEnvPool
from pipeline.vector_env import ResidentEnv


class _FakeBuilder:
    def build(self, env, spec):  # noqa: ANN001
        return {"ego": np.zeros((1, 8), dtype=np.float32)}


class _FakeEnv:
    """最小 env：step 推进位置并翻转标签值；终止后 reset 到"新 episode"位置。"""

    def __init__(self, *, terminate_after: int = 1) -> None:
        self.agent = SimpleNamespace(
            position=np.array([0.0, 0.0]),
            heading_theta=0.0,
            speed=5.0,
            on_white_continuous_line=False,
            on_yellow_continuous_line=False,
        )
        self.prev_policy_action = np.zeros(2)
        self.label_value = 1.0
        self.steps = 0
        self.terminate_after = int(terminate_after)

    def step(self, action):  # noqa: ANN001
        self.steps += 1
        self.label_value = 2.0
        self.agent.position = np.array([float(self.steps), 0.0])
        terminated = self.steps >= self.terminate_after
        return None, 0.0, terminated, False, {"episode_steps": self.steps}

    def reset(self):
        self.steps = 0
        self.label_value = 10.0
        self.agent.position = np.array([100.0, 0.0])
        return None, {}


def _make_pool(env: _FakeEnv) -> LocalEnvPool:
    pool = LocalEnvPool.__new__(LocalEnvPool)
    pool.specs = [SimpleNamespace(id=1)]
    pool.tracker_kind = "kinematic"
    pool.max_episode_steps = 5
    pool.traffic_density = None
    pool.obs_config = {}
    pool.logger = lambda *args, **kwargs: None
    pool.num_envs = 1
    pool._builder = _FakeBuilder()
    pool._env = env
    pool._spec = pool.specs[0]
    pool._index = 0
    # 本文件锁定 P0-2 记录协议（终局 record/reset 分离）：固定 spec（S1 轮换关闭）
    pool.spec_rotation = "off"
    pool._steps = 0
    pool._order = ("cutin_active",)
    pool._tracker = None
    pool._tracker_policy = None
    pool._current_info = {}
    return pool


@pytest.fixture()
def label_env(monkeypatch: pytest.MonkeyPatch):
    """``_router_labels_from_env`` → 直接读 fake env 的 ``label_value``。"""

    def fake_labels(env, spec, order):  # noqa: ANN001
        return np.full((len(order),), float(env.label_value), dtype=np.float32)

    monkeypatch.setattr(trainer_mod, "_router_labels_from_env", fake_labels)


def test_labels_now_reflects_pre_step_state(label_env) -> None:
    env = _FakeEnv(terminate_after=6)  # 一个策略步 = 5 个子步；6 保证本步不终局
    pool = _make_pool(env)
    assert np.allclose(pool.labels_now(), 1.0)
    pool.step(np.array([3.0, 0.0], dtype=np.float64))
    assert env.label_value == 2.0
    assert np.allclose(pool.labels_now(), 2.0), "labels_now 应反映实时（步前）状态"


def test_terminal_record_reset_separation(label_env) -> None:
    env = _FakeEnv(terminate_after=1)
    pool = _make_pool(env)
    record = pool.step(np.array([3.0, 0.0], dtype=np.float64))[0]

    assert record["terminated"] is True
    # 终局 record = 终局帧（位置 1.0；info 来自终局 env.step）
    assert float(record["obs"]["pose"][0]) == 1.0
    assert record["info"]["episode_steps"] == 1
    assert np.allclose(record["info"]["router_labels"], 2.0)
    # reset 后首帧只作为 next_obs（位置 100.0，标签 10.0），不混进终局 record
    assert float(record["next_obs"]["pose"][0]) == 100.0
    assert env.prev_policy_action.tolist() == [0.0, 0.0]
    assert pool._current_info == {}, "终局 reset 后必须清空 _current_info"


def test_reset_first_frame_drops_stale_info(label_env) -> None:
    env = _FakeEnv(terminate_after=1)
    pool = _make_pool(env)
    pool.step(np.array([3.0, 0.0], dtype=np.float64))
    # 模拟旧行为现场：上一 episode 的 step info 残留
    pool._current_info = {"episode_steps": 99}
    pool._build = lambda spec: None  # 绕过 MetaDrive 建图，仅验证记录协议
    record = pool.reset()[0]
    assert "episode_steps" not in record["info"], "reset 首帧复用了上一 episode 的 step info"
    assert np.allclose(record["info"]["router_labels"], 10.0), "reset 首帧标签应取自新 episode"


class _FlipWorkerEnv:
    def __init__(self) -> None:
        self.label_state = 1.0
        self.prev_policy_action = np.zeros(2)

    def step(self, action):  # noqa: ANN001
        self.label_state = 2.0  # 子步推进后翻转
        return None, 0.5, False, False, {"route_completion": 0.1}


def test_worker_records_labels_at_step_start() -> None:
    env = ResidentEnv(worker_id=0, build_obs=False, label_order=("cutin_active",))
    fake = _FlipWorkerEnv()
    env._env = fake
    env._router_labels = lambda: np.full((1,), fake.label_state, dtype=np.float32)

    record = env.step([(0.5, 0.2)], prev_action=(1.5, -0.1))

    assert np.allclose(record["labels_at_step_start"], 1.0), "必须是子步推进前的标签快照"
    assert np.allclose(record["router_labels"], 2.0), "router_labels 保持步后口径（兼容）"
    assert np.allclose(record["prev_action"], (1.5, -0.1))
