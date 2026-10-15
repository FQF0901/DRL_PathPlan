"""R38-Q4A：`path_*` 生产者补全（env 侧）+ 禁静默 0（net 侧）单元测试。

覆盖（零训练、零 GPU）：
1. ``ObservationBuilder.build`` 在 t0 段产出 8 列 ``path_*`` + 8 个 ``<key>__present``；
2. 生产者动态三列与 R37-M7 ``truth_fill`` **逐位相同**（同一公式出处 = ``env.obs.long_exog``）；
3. margin/nav/d2end 生产者值 = head 既有回退口径（``lane_semantic`` 列 4/5/11、``nav[:,-1]``）；
4. ``net/path_head.long_exog_from_obs``：dynamic 两列缺键 ⇒ fail-fast；显式 present 掩码被消费；
5. static 缺键 ⇒ 0 + present=0（诊断列；不报错）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import env.obs.od as od_module
from env.obs.builder import ObservationBuilder
from env.obs.long_exog import static_gap, truth_fill
from env.obs.path_field import PATH_LONG_EXOG_KEYS, path_long_exog_fields, path_present_key
from net.path_head import LONG_EXOG_FEATURE_NAMES, long_exog_from_obs
from tests.test_obs_schema_v2 import FakeEnv, FakeEgo, FakeVehicle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _patch_base_vehicle(monkeypatch):
    monkeypatch.setattr(od_module, "BaseVehicle", FakeVehicle)


def _build_default(ego: FakeEgo, others=(), step: int = 0):
    builder = ObservationBuilder({})  # 全默认通道（ego/od/ld/nav/signal/others/.../lane_semantic）
    env = FakeEnv(ego, list(others), step=step)
    return builder, env, builder.build(env, None)


# --------------------------------------------------------------------------- #
# 1) 生产者：8 列 + present 掩码
# --------------------------------------------------------------------------- #
def test_builder_produces_eight_path_columns_with_present_masks():
    ego = FakeEgo(vx=10.0)
    obj = FakeVehicle("front", 20.0, 0.0, vx=-5.0)
    _builder, _env, obs = _build_default(ego, [obj])

    for key in PATH_LONG_EXOG_KEYS:
        assert key in obs, f"builder 未产出 {key}"
        assert obs[key].shape == (1, ), f"{key} 形状应为 (1,)，收到 {obs[key].shape}"
        assert obs[key].dtype == np.float32
        present_key = path_present_key(key)
        assert present_key in obs and obs[present_key].shape == (1, )

    # 前车 20 m、**相对**接近速度 15 m/s（ego 10 − 前车 5）⇒ ttc = 20/15 s；
    # 矩形最近距 = 20 − 0.5·4.5 = 17.75 m
    assert float(obs["path_dynamic_ttc"][0]) == pytest.approx(20.0 / 15.0, abs=1e-6)
    assert float(obs["path_dynamic_distance"][0]) == pytest.approx(20.0 - 0.5 * 4.5, abs=1e-6)
    assert float(obs[path_present_key("path_dynamic_ttc")][0]) == 1.0
    assert float(obs[path_present_key("path_dynamic_distance")][0]) == 1.0
    # 无静态障碍（假 env 无 lane → scan 返回 None）⇒ 0 + present=0（显式主态）
    assert float(obs["path_static_distance"][0]) == 0.0
    assert float(obs[path_present_key("path_static_distance")][0]) == 0.0
    # 速度 = ego[:,0]
    assert float(obs["path_speed"][0]) == pytest.approx(10.0, abs=1e-6)


def test_producer_dynamic_columns_bitwise_match_m7_truth_fill():
    """生产者 vs R37-M7 注入器：同帧同公式（逐位）。"""
    ego = FakeEgo(vx=12.0)
    objs = [
        FakeVehicle("a", 25.0, 1.5, vx=-4.0),
        FakeVehicle("b", -8.0, 0.5, vx=1.0),
        FakeVehicle("c", 60.0, -2.0, vx=-0.5),
    ]
    _builder, env, obs = _build_default(ego, objs)

    record = truth_fill(dict(obs), env, env.agent)  # 只读快照（键已存在 ⇒ 不写）
    assert record["filled"] == {}, "生产者已产出三键时 M7 回填应为无操作"
    truth = record["truth"]
    assert float(obs["path_dynamic_ttc"][0]) == float(np.float32(truth["ttc"]))
    assert float(obs["path_dynamic_distance"][0]) == float(np.float32(truth["dyn"]))
    # 无静态障碍 ⇒ M7 truth 为 NaN（present=0），生产者写 0 + present=0
    assert not np.isfinite(truth["static"])
    assert float(obs["path_static_distance"][0]) == 0.0
    assert static_gap(env, env.agent) is None


def test_producer_static_column_matches_scan_when_blocker_present(monkeypatch):
    """静态列有真值 ⇒ 写真值（与 ``scan_static_blocker`` 同函数/同值）。"""
    _builder, env, obs = _build_default(FakeEgo(vx=8.0))
    monkeypatch.setattr("env.obs.path_field.static_gap", lambda _env, _ego: 31.5)
    fields = path_long_exog_fields(env, obs)
    assert float(fields["path_static_distance"][0]) == pytest.approx(31.5, abs=1e-6)
    assert float(fields[path_present_key("path_static_distance")][0]) == 1.0


# --------------------------------------------------------------------------- #
# 2) 派生列 = head 既有回退口径（逐位）
# --------------------------------------------------------------------------- #
def _fused_obs() -> dict[str, np.ndarray]:
    """合成单帧 t0 obs：ego/lane_semantic/nav + od（无 path 键）。"""
    sem = np.zeros((1, 12), dtype=np.float32)
    sem[0, 4] = 5.25
    sem[0, 5] = 5.24
    sem[0, 6] = 1.0
    sem[0, 11] = 22.0
    nav = np.zeros((1, 11), dtype=np.float32)
    nav[0, -1] = 0.12
    return {
        "ego": np.array([[9.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
        "ego_mask": np.ones((1, ), dtype=np.float32),
        "lane_semantic": sem,
        "lane_semantic_mask": np.ones((1, ), dtype=np.float32),
        "nav": nav,
        "nav_mask": np.ones((1, ), dtype=np.float32),
        "od": np.zeros((1, 16, 9), dtype=np.float32),
        "od_mask": np.zeros((1, 16), dtype=np.float32),
        "od_presence": np.zeros((1, 16), dtype=np.float32),
    }


def test_derived_columns_and_fallback_are_bitwise_identical():
    """生产者产出的 margin/nav/d2end = ``long_exog_from_obs`` 既有回退口径（逐位）。"""
    raw = _fused_obs()
    env = FakeEnv(FakeEgo())
    fields = path_long_exog_fields(env, raw)

    obs_before = {key: torch.as_tensor(np.asarray(value)) for key, value in raw.items()}
    obs_before["path_dynamic_distance"] = torch.zeros(1)
    obs_before["path_dynamic_ttc"] = torch.full((1, ), 60.0)
    exog_fallback, present_fallback = long_exog_from_obs(
        obs_before, batch=1, device=torch.device("cpu")
    )

    obs_after = {key: torch.as_tensor(np.asarray(value)) for key, value in raw.items()}
    obs_after.update({key: torch.as_tensor(np.asarray(value)) for key, value in fields.items()})
    exog_produced, present_produced = long_exog_from_obs(
        obs_after, batch=1, device=torch.device("cpu")
    )

    assert torch.equal(exog_fallback, exog_produced)          # 8 列值逐位
    # present 位：除列 4（dynamic_distance）外逐位相同 —— 旧回退口径把"键在场"记为 present=1，
    # 生产者用同源 ``__present`` 掩码（本帧无有效 OD 槽 ⇒ 0）。head 输入不含 present，
    # 故数值路径不受影响（闭环臂逐位回归见 q4a 报告）。
    assert torch.equal(
        present_fallback[:, [0, 1, 2, 3, 5, 6, 7]], present_produced[:, [0, 1, 2, 3, 5, 6, 7]]
    )
    assert float(present_fallback[0, 4]) == 1.0 and float(present_produced[0, 4]) == 0.0
    # 列值确实来自 lane_semantic / nav（非 0 静默）
    assert float(exog_produced[0, 1]) == pytest.approx(5.25, abs=1e-6)
    assert float(exog_produced[0, 2]) == pytest.approx(5.24, abs=1e-6)
    assert float(exog_produced[0, 6]) == pytest.approx(0.12, abs=1e-6)
    assert float(exog_produced[0, 7]) == pytest.approx(22.0, abs=1e-6)
    assert float(exog_produced[0, 5]) == pytest.approx(60.0, abs=1e-6)  # 无 OD 槽 ⇒ cap = 无冲突


# --------------------------------------------------------------------------- #
# 3) 禁静默 0：fail-fast + 显式 present
# --------------------------------------------------------------------------- #
def _obs_without_path(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "ego": torch.zeros((batch, 8), dtype=torch.float32),
        "lane_semantic": torch.zeros((batch, 12), dtype=torch.float32),
        "lane_semantic_mask": torch.ones(batch),
        "nav": torch.zeros((batch, 11), dtype=torch.float32),
    }


def test_missing_dynamic_columns_fail_fast():
    obs = _obs_without_path()
    with pytest.raises(ValueError, match="obs 缺 'path_dynamic_distance'"):
        long_exog_from_obs(obs, batch=2, device=torch.device("cpu"))

    obs_dyn_only = dict(obs)
    obs_dyn_only["path_dynamic_distance"] = torch.zeros(2)
    with pytest.raises(ValueError, match="obs 缺 'path_dynamic_ttc'"):
        long_exog_from_obs(obs_dyn_only, batch=2, device=torch.device("cpu"))


def test_explicit_present_masks_are_consumed_and_zero_is_allowed():
    """值 0 + ``__present=0`` = 合法「本帧无观测」（不是缺键、不是静默 0）。"""
    obs = _obs_without_path()
    obs["path_dynamic_ttc"] = torch.full((2, ), 60.0)
    obs["path_dynamic_ttc__present"] = torch.tensor([1.0, 1.0])
    obs["path_dynamic_distance"] = torch.zeros(2)
    obs["path_dynamic_distance__present"] = torch.tensor([0.0, 1.0])
    obs["path_static_distance"] = torch.zeros(2)
    obs["path_static_distance__present"] = torch.tensor([0.0, 0.0])
    obs["path_left_margin"] = torch.full((2, ), 3.0)
    obs["path_left_margin__present"] = torch.tensor([1.0, 0.0])

    exog, present = long_exog_from_obs(obs, batch=2, device=torch.device("cpu"))
    assert list(LONG_EXOG_FEATURE_NAMES[4:6]) == ["dynamic_distance", "dynamic_ttc"]
    assert present[:, 4].tolist() == [0.0, 1.0]
    assert present[:, 3].tolist() == [0.0, 0.0]
    assert present[:, 1].tolist() == [1.0, 0.0]
    assert float(exog[0, 4]) == 0.0 and float(exog[0, 3]) == 0.0


def test_static_column_missing_stays_zero_with_present_zero():
    """static 缺键 ⇒ 0 + present=0（诊断列；文档化行为，不报错）。"""
    obs = _obs_without_path()
    obs["path_dynamic_ttc"] = torch.full((2, ), 60.0)
    obs["path_dynamic_distance"] = torch.zeros(2)
    exog, present = long_exog_from_obs(obs, batch=2, device=torch.device("cpu"))
    assert float(exog[0, 3]) == 0.0
    assert float(present[0, 3]) == 0.0
