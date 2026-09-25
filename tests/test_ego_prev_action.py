"""ego 通道 reserved 维（上一策略动作）注入测试。

用 stub env（不建 MetaDrive 仿真）：Gate 3 §8.4 —— reserved0/1 = 上一策略步 (ds, dtheta)，
由调用方通过 ``env.prev_policy_action`` 注入；未注入或格式非法时保持 0。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from env.obs.ego import EgoChannel


class _StubLane:
    """最小车道桩：触发 curvature 计算失败路径（build 内 try/except → 0）。"""

    def local_coordinates(self, position):  # noqa: ANN001
        return (0.0, 0.0)


def _stub_env(*, prev_action=None) -> SimpleNamespace:
    ego = SimpleNamespace(
        speed=10.0,
        last_speed=9.8,
        heading_theta=0.2,
        last_heading_dir=(1.0, 0.0),
        steering=0.1,
        lane=_StubLane(),
        position=np.zeros(2, dtype=np.float32),
    )
    env = SimpleNamespace(
        agent=ego,
        config={"physics_world_step_size": 0.02, "decision_repeat": 5},
    )
    if prev_action is not None:
        env.prev_policy_action = prev_action
    return env


def test_reserved_dims_zero_by_default() -> None:
    feats, mask = EgoChannel().build(_stub_env())
    assert feats.shape == (1, 8)
    assert mask[0] == 1.0
    assert feats[0, 6] == 0.0 and feats[0, 7] == 0.0


def test_reserved_dims_carry_prev_action() -> None:
    feats, _ = EgoChannel().build(_stub_env(prev_action=(3.5, -0.12)))
    assert abs(float(feats[0, 6]) - 3.5) < 1e-6
    assert abs(float(feats[0, 7]) + 0.12) < 1e-6


def test_malformed_prev_action_ignored() -> None:
    feats, _ = EgoChannel().build(_stub_env(prev_action=("x", None)))
    assert feats[0, 6] == 0.0 and feats[0, 7] == 0.0
