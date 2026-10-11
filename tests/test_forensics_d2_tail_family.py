"""Round-15A2：forensics d2 尾族（``--d2-tail``）CPU 单测。

不建 env、不加载 ckpt：``parse_tail_spec`` / ``build_tail_plan`` 纯函数断言（默认
repeat 逐位不变；衰减/保曲率/航向衰减形状）+ CLI + 控制器接线（fake env/labeler/tracker）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tools.diagnostics.forensics_closed_loop import (
    D2RepeatExpertController,
    build_parser,
    build_tail_plan,
    parse_tail_spec,
    scale_expert_action,
)


def test_parse_tail_spec_repeat() -> None:
    factors, dims = parse_tail_spec("repeat")
    assert dims == "both"
    assert np.array_equal(factors, np.ones(6))


def test_parse_tail_spec_decay() -> None:
    factors, dims = parse_tail_spec("decay:geo:0.8")
    assert dims == "both"
    assert factors.tolist() == pytest.approx([0.8 ** k for k in range(6)])
    factors, dims = parse_tail_spec("decay:lin:0.5")
    assert factors.tolist() == pytest.approx([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
    factors, dims = parse_tail_spec("dtheta_decay:geo:0.8")
    assert dims == "dtheta"
    assert factors.tolist() == pytest.approx([0.8 ** k for k in range(6)])


def test_parse_tail_spec_errors() -> None:
    for bad in ("bad", "decay:foo:0.5", "decay:geo:1.5", "decay:geo:0", "decay:geo:x", "repeat:extra"):
        with pytest.raises(ValueError):
            parse_tail_spec(bad)


def test_build_tail_plan_repeat_bitwise() -> None:
    action = np.array([2.0, 0.4], dtype=np.float64)
    plan = build_tail_plan(action, "repeat")
    expected = np.repeat(action[None, :], 6, axis=0)
    assert plan.tobytes() == expected.tobytes()  # 默认逐位不变


def test_build_tail_plan_decay_preserves_curvature() -> None:
    action = np.array([2.0, 0.4], dtype=np.float64)
    plan = build_tail_plan(action, "decay:geo:0.8")
    factors = np.array([0.8 ** k for k in range(6)])
    assert np.allclose(plan[:, 0], 2.0 * factors)
    assert np.allclose(plan[:, 1], 0.4 * factors)
    ratio = plan[:, 1] / plan[:, 0]
    assert np.allclose(ratio, 0.2)  # 保曲率（ds/dθ 恒定）


def test_build_tail_plan_dtheta_decay() -> None:
    action = np.array([2.0, 0.4], dtype=np.float64)
    plan = build_tail_plan(action, "dtheta_decay:lin:0.5")
    assert np.allclose(plan[:, 0], 2.0)  # ds 不动
    assert np.allclose(plan[:, 1], 0.4 * np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]))


def test_parser_d2_tail() -> None:
    parser = build_parser()
    assert parser.parse_args(["--out", "x"]).d2_tail == "repeat"
    args = parser.parse_args(["--mode", "d2", "--d2-tail", "decay:geo:0.8", "--out", "x"])
    assert args.d2_tail == "decay:geo:0.8"
    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "d2", "--d2-tail", "nonsense", "--out", "x"])


class _FakeAgent:
    position = [0.0, 0.0]
    heading_theta = 0.0
    speed = 5.0
    lane = None
    lane_index = None
    navigation = None


class _FakeEnv:
    agent = _FakeAgent()


def _make_controller(*, tail_spec: str):
    controller = D2RepeatExpertController(
        None, None, {"device": "cpu"}, tail_spec=tail_spec
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


def test_controller_tail_spec_wiring() -> None:
    controller, captured = _make_controller(tail_spec="decay:geo:0.8")
    controller.action(_FakeEnv())
    plan = captured["plan"]
    expected = build_tail_plan(np.array([2.0, 0.4]), "decay:geo:0.8")
    assert np.array_equal(plan, expected)
    # 与显式缩放（V1）组合：先缩放再套尾族
    controller2, captured2 = _make_controller(tail_spec="dtheta_decay:geo:0.8")
    controller2.ds_scale = 1.1
    controller2.action(_FakeEnv())
    scaled = scale_expert_action(np.array([2.0, 0.4]), ds_scale=1.1)
    assert np.array_equal(captured2["plan"], build_tail_plan(scaled, "dtheta_decay:geo:0.8"))
