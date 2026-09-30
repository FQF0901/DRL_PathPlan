"""③ 评测语义开关（``--eval-reference``）回归测试（stub，不建 env / 不加载 ckpt）。

锁定：
- CLI 默认 ``plan``（现状逐位不变）；合法值 {plan, repeat_action}，非法值 fail-fast；
- ``build_eval_references``：``repeat_action`` = ``repeat(mu,6)``（与训练 A-hold 同构）；
  ``plan`` = plan 预览且首步强制 = mu；
- ``_CkptController.action``：``repeat_action`` 走 cheap path（``rollout=False``，不消费 plan），
  ``plan`` 保留 ``rollout=True``；两者下发给跟踪器的参考形状/取值正确。
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest
import torch

from pipeline.eval_runner import (
    EVAL_REFERENCES,
    _CkptController,
    build_eval_references,
    build_parser,
)


# --------------------------------------------------------------------------- #
# 解析与纯函数
# --------------------------------------------------------------------------- #

def test_cli_default_is_plan_and_choices() -> None:
    args = build_parser().parse_args([])
    assert args.eval_reference == "plan"  # 默认 = 现状
    assert build_parser().parse_args(["--eval-reference", "repeat_action"]).eval_reference == "repeat_action"
    assert EVAL_REFERENCES == ("plan", "repeat_action")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--eval-reference", "nope"])


def test_build_eval_references_repeat_action_matches_training_ahold() -> None:
    mu = np.asarray([3.25, -0.125])
    references = build_eval_references(mu, None, "repeat_action")
    assert references.shape == (6, 2)
    assert np.array_equal(references, np.repeat(mu[None, :], 6, axis=0))
    # plan 输出即使是"漂移的想象曲线"，repeat_action 也完全不消费它
    biased = build_eval_references(mu, np.full((6, 2), 99.0), "repeat_action")
    assert np.array_equal(biased, references)


def test_build_eval_references_plan_forces_first_row_to_mu() -> None:
    mu = np.asarray([2.0, 0.1])
    plan = np.arange(12, dtype=np.float64).reshape(6, 2) * 0.5
    references = build_eval_references(mu, plan, "plan")
    assert references.shape == (6, 2)
    assert np.array_equal(references[0], mu)  # 首步 = 执行动作（确定性评测）
    assert np.array_equal(references[1:], plan[1:])  # 后 5 步 = 规划预览（旧行为）
    with pytest.raises(ValueError):
        build_eval_references(mu, None, "plan")  # plan 口径必须给 plan
    with pytest.raises(ValueError):
        build_eval_references(mu, None, "nope")


# --------------------------------------------------------------------------- #
# _CkptController.action（stub 模型/池）
# --------------------------------------------------------------------------- #

class _StubModel(torch.nn.Module):
    """最小契约：action_mu 与 plan 后段解耦；记录 rollout 标志。"""

    def __init__(self) -> None:
        super().__init__()
        self.rollout_flags: List[bool] = []

    def forward(self, obs, *, rollout: bool = True, world_model: bool = True, wm_detach: bool = False):
        self.rollout_flags.append(bool(rollout))
        mu = torch.tensor([[3.0, 0.2]], dtype=torch.float64)
        plan = mu.unsqueeze(1) + torch.arange(6, dtype=torch.float64).reshape(1, 6, 1) * 10.0
        return {"action_mu": mu, "action_logstd": torch.zeros(1, 2, dtype=torch.float64), "plan": plan}


class _StubTracker:
    def __init__(self) -> None:
        self.references: Optional[np.ndarray] = None

    def set_reference(self, reference) -> None:
        self.references = np.asarray(reference, dtype=np.float64).copy()


class _StubBuilder:
    def build(self, env, spec):  # noqa: D102 - stub
        return {"ego": np.zeros((1, 8), dtype=np.float32)}


class _StubAgent:
    position = (0.0, 0.0)
    heading_theta = 0.0


class _StubEnv:
    agent = _StubAgent()


def _controller(eval_reference: str) -> _CkptController:
    controller = object.__new__(_CkptController)
    controller.spec = type("_Spec", (), {"id": 0, "seed": 0})()
    controller.device = torch.device("cpu")
    controller.model = _StubModel()
    controller.builder = _StubBuilder()
    controller.tracker_kind = "lqr"
    controller.tracker = _StubTracker()
    controller.decision_interval = 5
    controller._steps = 0
    controller._action = [0.0, 0.0]
    controller._plan_action = np.zeros(2, dtype=np.float64)
    controller._pose_history = []
    controller.eval_reference = eval_reference
    return controller


def test_controller_repeat_action_uses_cheap_path_and_repeated_reference() -> None:
    controller = _controller("repeat_action")
    controller.action(_StubEnv())
    assert controller.model.rollout_flags == [False], "repeat_action 必须走 cheap path（rollout=False）"
    expected = np.repeat(np.asarray([3.0, 0.2])[None, :], 6, axis=0)
    assert np.allclose(controller.tracker.references, expected, atol=0.0)
    assert np.allclose(controller._plan_action, [3.0, 0.2])
    # 逐 env step 递增；未到决策边界不再前向
    controller.action(_StubEnv())
    assert controller.model.rollout_flags == [False]


def test_controller_plan_uses_rollout_path_and_plan_tail() -> None:
    controller = _controller("plan")
    controller.action(_StubEnv())
    assert controller.model.rollout_flags == [True], "plan 口径保留 rollout 前向"
    references = controller.tracker.references
    assert references.shape == (6, 2)
    assert np.allclose(references[0], [3.0, 0.2])  # 首步 = mu
    assert np.allclose(references[1:, 0], [13.0, 23.0, 33.0, 43.0, 53.0])  # 后 5 步 = plan 预览
