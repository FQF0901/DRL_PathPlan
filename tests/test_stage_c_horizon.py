"""P4/Gate4 **horizon 对齐**：Stage C 训练截断 = 200 策略步（=100 s，与审计/评测一致）。

锁定：

- ``--max-episode-steps`` 解析序：CLI 优先 → config ``stages.C.max_episode_steps`` →
  :data:`DEFAULT_STAGE_C_MAX_EPISODE_STEPS`（200）；非正整数 fail-fast；
- ``config/train.yaml`` 默认 200（= 旧 600 的 1/3）；臂文件 ``stages`` 段整体替换后
  回落代码默认，解析结果同为 200（arm 路径不漂移）；
- ``build_pool(max_episode_steps=...)`` 透传到 ``LocalEnvPool``（截断执行值）；
- 算术：200 策略步 × ``_POLICY_DT``（0.5 s）= 100 s = 审计/评测 1000 物理步 × 0.1 s。

背景（Gate4）：旧默认 600 策略步 = 300 s = 审计 3× —— max_step 的 dense（实测 +0.168/步、
不衰减）在训练口径被放大 ≈3× → 超时 total ≈ +54 ≈ 到达（"超时正收益"）。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.stages import (
    DEFAULT_STAGE_C_MAX_EPISODE_STEPS,
    _parse_args,
    _resolve_stage_c_max_episode_steps,
    build_pool,
    load_config,
)
from pipeline.trainer import _PHYSICS_HZ, _POLICY_DT

_ROOT = Path(__file__).resolve().parents[1]


def test_resolve_max_episode_steps_cli_config_default() -> None:
    args = _parse_args(["--stage", "C"])
    assert args.max_episode_steps is None, "未给 CLI → 走 config/默认"
    assert DEFAULT_STAGE_C_MAX_EPISODE_STEPS == 200
    assert _resolve_stage_c_max_episode_steps(args, {}) == 200, "默认 200（horizon 对齐）"
    assert _resolve_stage_c_max_episode_steps(args, {"max_episode_steps": 300}) == 300, "config 覆盖默认"

    cli = _parse_args(["--stage", "C", "--max-episode-steps", "150"])
    assert cli.max_episode_steps == 150
    assert _resolve_stage_c_max_episode_steps(cli, {"max_episode_steps": 300}) == 150, "CLI 优先"


def test_resolve_max_episode_steps_fail_fast() -> None:
    for bad in ("0", "-1"):
        args = _parse_args(["--stage", "C", "--max-episode-steps", bad])
        with pytest.raises(SystemExit):
            _resolve_stage_c_max_episode_steps(args, {})
    # config 非法值（CLI 缺省时）
    args = _parse_args(["--stage", "C"])
    with pytest.raises(SystemExit):
        _resolve_stage_c_max_episode_steps(args, {"max_episode_steps": "bad"})
    with pytest.raises(SystemExit):
        _resolve_stage_c_max_episode_steps(args, {"max_episode_steps": None})


def test_config_train_yaml_default_is_aligned() -> None:
    config = load_config("config/default.yaml")
    assert config["stages"]["C"]["max_episode_steps"] == 200, "config 默认 = 200 策略步"
    args = _parse_args(["--stage", "C"])
    assert _resolve_stage_c_max_episode_steps(args, config["stages"]["C"]) == 200


def test_arm_configs_resolve_to_aligned_default() -> None:
    """臂文件 ``stages`` 段整体替换（仅 reward）→ 回落代码默认；解析结果仍 = 200。"""
    for name in ("arm0_bundle_rc1.yaml", "arm4_lam098.yaml", "arm5_ttc.yaml"):
        config = load_config(str(_ROOT / "config" / "arms" / name))
        assert "max_episode_steps" not in config["stages"]["C"], "臂文件不显式携带（回落代码默认）"
        assert _resolve_stage_c_max_episode_steps(_parse_args(["--stage", "C"]), config["stages"]["C"]) == 200


def test_build_pool_passes_max_episode_steps_to_local_pool() -> None:
    specs = [SimpleNamespace(id=0)]
    pool = build_pool(specs, kind="local", num_envs=1, max_episode_steps=200)
    try:
        assert type(pool).__name__ == "LocalEnvPool"
        assert int(pool.max_episode_steps) == 200, "截断执行值 = 解析值"
    finally:
        pool.close()


def test_horizon_arithmetic_matches_audit_and_eval() -> None:
    """200 策略步 × 0.5 s = 100 s = 1000 物理步 × 0.1 s（审计/评测口径）；旧 600 = 3×。"""
    assert _POLICY_DT == pytest.approx(0.5)
    assert _PHYSICS_HZ == pytest.approx(10.0)
    policy_steps = DEFAULT_STAGE_C_MAX_EPISODE_STEPS
    assert policy_steps * _POLICY_DT == pytest.approx(100.0)
    assert policy_steps * _POLICY_DT == pytest.approx(1000 / _PHYSICS_HZ)
    assert 600 * _POLICY_DT == pytest.approx(3 * policy_steps * _POLICY_DT), "旧默认 = 3× 审计口径"
