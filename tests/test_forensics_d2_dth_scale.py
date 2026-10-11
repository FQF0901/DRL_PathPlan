"""P1-v2 D：forensics d2 ``--d2-dth-scale``（dθ-only 反事实）CPU 单测。

不建 env、不加载 ckpt：``scale_expert_action`` 纯函数逐位/缩放断言；控制器接线用
fake env/labeler/tracker；CLI 参数默认与解析。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.diagnostics.forensics_closed_loop import (
    D2RepeatExpertController,
    build_parser,
    scale_expert_action,
)


# --------------------------------------------------------------------------- #
# 纯函数：默认逐位不变 / 缩放
# --------------------------------------------------------------------------- #
def test_scale_default_bitwise_unchanged() -> None:
    action = np.array([2.0, 0.4], dtype=np.float64)
    out = scale_expert_action(action)
    assert out.tobytes() == action.tobytes()  # 逐位不变（无复制/运算）
    assert scale_expert_action([2.0, 0.4]).tolist() == [2.0, 0.4]


def test_scale_dth_only() -> None:
    action = np.array([2.0, 0.4], dtype=np.float64)
    out = scale_expert_action(action, dth_scale=0.85)
    assert out[0] == 2.0  # ds 不动
    assert out[1] == pytest.approx(0.4 * 0.85)
    # 默认（dth=1.0）不受影响
    assert scale_expert_action(action, dth_scale=1.0).tobytes() == action.tobytes()


def test_scale_v1_v2_and_composition() -> None:
    action = np.array([2.0, 0.4], dtype=np.float64)
    assert scale_expert_action(action, ds_scale=1.1).tolist() == pytest.approx([2.2, 0.4])
    assert scale_expert_action(action, ds_dth_scale=1.1).tolist() == pytest.approx([2.2, 0.44])
    # dth-only 与 V2 乘性组合
    assert scale_expert_action(action, ds_dth_scale=1.1, dth_scale=0.9).tolist() == pytest.approx(
        [2.2, 0.44 * 0.9]
    )
    # dth-only 与 V1 乘性组合
    assert scale_expert_action(action, ds_scale=1.1, dth_scale=0.9).tolist() == pytest.approx(
        [2.2, 0.36]
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_parser_d2_dth_scale() -> None:
    parser = build_parser()
    assert parser.parse_args(["--out", "x"]).d2_dth_scale == 1.0
    args = parser.parse_args(["--mode", "d2", "--d2-dth-scale", "0.85", "--out", "x"])
    assert args.d2_dth_scale == pytest.approx(0.85)
    assert args.d2_ds_scale == 1.0 and args.d2_ds_dth_scale == 1.0


# --------------------------------------------------------------------------- #
# 控制器接线（fake env / labeler / tracker）
# --------------------------------------------------------------------------- #
class _FakeAgent:
    position = [0.0, 0.0]
    heading_theta = 0.0
    speed = 5.0
    lane = None
    lane_index = None
    navigation = None


class _FakeEnv:
    agent = _FakeAgent()


def _make_controller(*, dth_scale: float) -> tuple[D2RepeatExpertController, dict]:
    controller = D2RepeatExpertController(
        None, None, {"device": "cpu"}, dth_scale=dth_scale
    )
    captured: dict = {}
    controller.labeler = SimpleNamespace(
        label=lambda ego: {
            "action": np.array([2.0, 0.4], dtype=np.float64),
            "raw_action": np.array([0.1, 0.2], dtype=np.float64),
        }
    )
    controller.tracker = SimpleNamespace(
        set_reference=lambda plan: captured.update(plan=np.asarray(plan, dtype=np.float64)),
        _ref_world=None,
    )
    controller._steps = 0
    controller.decision_interval = 5
    return controller, captured


def test_controller_wiring_default_unchanged() -> None:
    controller, captured = _make_controller(dth_scale=1.0)
    controller.action(_FakeEnv())
    plan = captured["plan"]
    assert plan.shape == (6, 2)
    assert np.array_equal(plan, np.repeat(np.array([[2.0, 0.4]]), 6, axis=0))  # 逐位不变
    assert controller.records[-1]["reference"] == "expert_repeat"


def test_controller_wiring_dth_only() -> None:
    controller, captured = _make_controller(dth_scale=0.85)
    controller.action(_FakeEnv())
    plan = captured["plan"]
    assert np.allclose(plan[:, 0], 2.0)  # ds 不动
    assert np.allclose(plan[:, 1], 0.4 * 0.85)
    assert controller.dth_scale == pytest.approx(0.85)
