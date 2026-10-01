"""P4 臂配置 dry-run（Gate4 准备）：逐臂 ``build_reward_adapter`` + 终局值/权重与 E-β″ 草案一致。

锁定（CPU；不开环境）：

- ``config/arms/arm0–7`` 经 ``pipeline.stages.load_config``（训练侧平铺合并）可加载，
  且 ``stages.C.reward`` 经 ``build_reward_adapter`` 构造成功（rc 档配对守卫通过）；
- 终局值与 ``docs/reward_audit/ebeta2/config_draft_rc*.yaml`` 逐档一致；
- 单变量：arm1–3 rc 档、arm4 ``train.ppo.lam=0.98``、arm5 ``ttc``、arm6 ``lane_boundary``、
  arm7 ``lane_center``；``speed_ratio`` 0.4 / ``low_speed`` 启用为 bundle 公共项；
- ``max_step`` 经适配器实际结算（``terminal_key=max_step``）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import yaml

from pipeline.stages import load_config
from pipeline.trainer import build_reward_adapter

_ROOT = Path(__file__).resolve().parents[1]
_ARMS = _ROOT / "config" / "arms"
_DRAFTS = _ROOT / "docs" / "reward_audit" / "ebeta2"

#: 臂文件 includes（训练侧 load_config 只解析一层 → 同时列 default.yaml 与其叶子子配置）
_INCLUDES = [
    "config/default.yaml",
    "config/env.yaml",
    "config/model.yaml",
    "config/train.yaml",
    "config/eval.yaml",
]

#: 臂文件 → 期望（tier=rc 档；lam=λ 单变量；extra=追加项 (name, weight)）
_ARM_CASES: Dict[str, Dict[str, Any]] = {
    "arm0_bundle_rc1.yaml": {"tier": 1.0},
    "arm1_rc3.yaml": {"tier": 3.0},
    "arm2_rc10.yaml": {"tier": 10.0},
    "arm3_rc30.yaml": {"tier": 30.0},
    "arm4_lam098.yaml": {"tier": 1.0, "lam": 0.98},
    "arm5_ttc.yaml": {"tier": 1.0, "extra": ("ttc", -0.5)},
    "arm6_lane_boundary.yaml": {"tier": 1.0, "extra": ("lane_boundary", -0.2)},
    "arm7_lane_center.yaml": {"tier": 1.0, "extra": ("lane_center", -0.1)},
}


def _draft_reward(tier: float) -> Dict[str, Any]:
    with open(_DRAFTS / f"config_draft_rc{tier:g}.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["stages"]["C"]["reward"]


def _max_step_terminal_value(adapter) -> float:  # noqa: ANN001
    _, meta = adapter.step(
        0, {"max_step": True}, {"ego": np.zeros((1, 8), dtype=np.float32)}, True, 0.0
    )
    assert meta["terminal_key"] == "max_step"
    return float(meta["terminal_value"])


def _load_arm(name: str) -> Dict[str, Any]:
    with open(_ARMS / name, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    assert raw.get("includes") == _INCLUDES, "臂配置 includes = default.yaml + 四个叶子子配置"
    return load_config(str(_ARMS / name))


@pytest.mark.parametrize("arm_name,spec", list(_ARM_CASES.items()))
def test_arm_reward_pairs_and_matches_draft(arm_name: str, spec: Dict[str, Any]) -> None:
    config = _load_arm(arm_name)
    reward_cfg = config["stages"]["C"]["reward"]
    adapter, source = build_reward_adapter(reward_cfg)
    assert source.startswith("reward_model.aggregation.RewardAggregator"), source

    weights = {term.name: term.weight for term in adapter.factory().terms}
    assert weights["route_completion"] == pytest.approx(spec["tier"])
    assert weights["speed_ratio"] == pytest.approx(0.4)
    assert weights["low_speed"] == pytest.approx(-0.2)

    draft = _draft_reward(spec["tier"])
    values = reward_cfg["aggregation"]["terminal_values"]
    assert values == draft["aggregation"]["terminal_values"], "终局值必须与 E-β″ 草案逐档一致"
    assert _max_step_terminal_value(adapter) == pytest.approx(values["max_step"])

    draft_names = {term["name"] for term in draft["terms"]}
    extra = spec.get("extra")
    if extra is None:
        # Gate4 加固：不只看项名集合——逐参数（weight/gamma/deadband/cap/…）与同档草案全等
        assert reward_cfg["terms"] == draft["terms"], "非单变量臂项集应与同档草案逐参数一致"
    else:
        name, weight = extra
        assert name not in draft_names, f"{name} 应为追加项（草案不含）"
        assert weights[name] == pytest.approx(weight)
        # Gate4 加固：去掉 extra 项后的项集逐参数 == 同档草案（顺序亦一致）
        rest = [term for term in reward_cfg["terms"] if str(term.get("name")) != name]
        assert len(rest) == len(reward_cfg["terms"]) - 1, "臂项集应恰含一个 extra 项"
        assert rest == draft["terms"], "去 extra 项集应与同档草案逐参数一致"
        assert set(weights) == draft_names | {name}

    if "lam" in spec:
        assert config["train"]["ppo"]["lam"] == pytest.approx(spec["lam"])


def test_flat_merge_semantics_documented() -> None:
    """训练侧 ``load_config`` 一层平铺合并：arm 的 ``stages`` 段整体替换；其余顶层段保留。

    臂文件同时列 ``config/default.yaml`` 与四个叶子子配置——只写 default.yaml 会丢
    ``train``/``data``/``run`` 等段（一层 includes、不递归；dry-run 实测）。非 reward 的
    ``stages.C`` 键回落代码默认（本提交与 ``config/train.yaml`` 逐键一致，见
    ``config/arms/README.md``「加载语义」）——本测试把该语义钉住，防无声漂移。
    """
    arm_cfg = _load_arm("arm0_bundle_rc1.yaml")
    assert set(arm_cfg["stages"]["C"]) == {"reward"}
    default_cfg = load_config("config/default.yaml")
    for key, value in default_cfg.items():
        if key == "stages":
            continue  # stages 由臂文件整体替换（仅 C.reward）
        assert key in arm_cfg, f"平铺合并应保留 {key}"
        assert arm_cfg[key] == value, f"{key} 应与解析 config/default.yaml 逐位一致"


def test_arm4_lambda_pins_train_block() -> None:
    """arm4 显式钉整段 ``train``（仅 lam 改 0.98）——避免平铺合并把其余 train 键打回默认。"""
    arm_cfg = _load_arm("arm4_lam098.yaml")
    default_train = load_config("config/default.yaml")["train"]
    assert default_train["ppo"]["lam"] == pytest.approx(0.95)
    expected = {**default_train, "ppo": {**dict(default_train["ppo"]), "lam": 0.98}}
    assert arm_cfg["train"] == expected, "arm4 的 train 段应 = 默认逐位一致，唯一变更 lam=0.98"


def test_extra_term_params() -> None:
    """追加项参数（阈值/死区）与 §5 口径一致。"""
    ttc = build_reward_adapter(_load_arm("arm5_ttc.yaml")["stages"]["C"]["reward"])[0]
    ttc_term = {term.name: term for term in ttc.factory().terms}["ttc"]
    assert ttc_term.ttc_threshold == pytest.approx(2.0)
    assert ttc_term.ttc_floor == pytest.approx(0.5)

    boundary = build_reward_adapter(_load_arm("arm6_lane_boundary.yaml")["stages"]["C"]["reward"])[0]
    boundary_term = {term.name: term for term in boundary.factory().terms}["lane_boundary"]
    assert boundary_term.margin_threshold == pytest.approx(0.5)

    center = build_reward_adapter(_load_arm("arm7_lane_center.yaml")["stages"]["C"]["reward"])[0]
    center_term = {term.name: term for term in center.factory().terms}["lane_center"]
    assert center_term.deadband == pytest.approx(0.25)
    assert center_term.clamp == pytest.approx(3.0)
