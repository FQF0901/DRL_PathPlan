"""R21-29 契约测试：κ 真掩码 / 低 ds 三态 / 负 ds fail-fast / LQR 耦合遥测与前馈（默认关）。"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from metadrive.policy.base_policy import BasePolicy

from env.tracking import LqrTracker
from pipeline.residual_classes import (
    CLASS_C,
    CLASS_N0,
    CURV_INVALID,
    CURV_LOW_YAW,
    CURV_VALID,
    classify_rows,
    compute_thresholds,
    curvature_state,
    low_ds_dtheta_threshold,
)
from pipeline.semantic_labels import NEG_DS_EPS, kinematics_metadata


# --------------------------------------------------------------------------- #
# 1) plan_valid_delta_kappa（pairwise ds ≥ 0.5）
# --------------------------------------------------------------------------- #
def test_plan_valid_delta_kappa_pairwise() -> None:
    from tests.test_r21_3a_semantic_wiring import _activate_residual, _semantic_model, _semantic_obs

    model = _semantic_model()
    _activate_residual(model, context=0.2, token=0.2)
    obs = _semantic_obs(batch=2, seed=51)
    with torch.no_grad():
        out = model(obs, rollout=True, world_model=False)
    ds = out["plan"][..., 0]
    expected = (ds.abs() >= 0.5)[..., :-1] & (ds.abs() >= 0.5)[..., 1:]
    assert out["plan_valid_delta_kappa"].dtype == torch.bool
    assert torch.equal(out["plan_valid_delta_kappa"], expected)
    # 手工构造：把某步 ds 置 0.1 → 相邻两个 Δκ 均无效
    plan = out["plan"].clone()
    plan[:, 2, 0] = 0.1
    valid = (plan[..., 0].abs() >= 0.5)[..., :-1] & (plan[..., 0].abs() >= 0.5)[..., 1:]
    assert not bool(valid[0, 1]) and not bool(valid[0, 2]) and bool(valid[0, 0])


# --------------------------------------------------------------------------- #
# 2) 低 ds 三态 + 分类门
# --------------------------------------------------------------------------- #
def test_curvature_state_three_states() -> None:
    expert_ds = np.array([2.0, 0.2, 0.2, 2.0])
    expert_dth = np.array([0.05, 0.001, 0.9, 0.05])
    legacy_ds = np.array([2.0, 0.2, 0.2, 0.2])
    state = curvature_state(expert_ds, expert_dth, legacy_ds, theta_low=0.01)
    assert list(state) == [CURV_VALID, CURV_LOW_YAW, CURV_INVALID, CURV_INVALID]


def test_low_ds_dtheta_threshold_from_expert() -> None:
    ds = np.array([2.0, 0.1, 0.2, 0.3, 1.0])
    dth = np.array([0.5, 0.001, 0.002, 0.003, 0.4])
    theta = low_ds_dtheta_threshold(ds, dth)
    assert abs(theta - float(np.quantile([0.001, 0.002, 0.003], 0.95))) < 1e-12  # 低 ds 行的 |dθ| P95


def test_low_ds_large_dtheta_forces_c() -> None:
    """低 ds + 大 dθ（invalid_or_unknown）→ 禁 N0/N1（落 C），即使 |Δds|/|Δdθ| 很小。"""
    n = 4
    features = {
        "row_type": np.zeros(n, dtype=np.int64),
        "res_ds": np.full(n, 0.001),
        "res_dth": np.full(n, 0.0001),
        "future_valid": np.ones(n, dtype=bool),
        "future_lat_max": np.full(n, 0.1),
        "future_head_max": np.full(n, 0.01),
        "future_speed_min": np.full(n, 5.0),
        "episode_id": np.zeros(n, dtype=np.int64),
        "legacy_rollout_safe": np.ones(n, dtype=bool),
        "res_kappa": np.zeros(n),
        "kappa_valid": np.ones(n, dtype=bool),
        "curvature_state": np.array([CURV_VALID, CURV_LOW_YAW, CURV_INVALID, CURV_INVALID], dtype=object),
    }
    thresholds = compute_thresholds(features["res_ds"], features["res_dth"], np.ones(n, dtype=bool),
                                    res_kappa=features["res_kappa"], kappa_valid=features["kappa_valid"])
    labels = classify_rows(features, thresholds)
    assert labels[0] == CLASS_N0
    assert labels[1] == CLASS_N0          # 低速小转向：可 N0
    assert labels[2] == CLASS_C and labels[3] == CLASS_C  # 低 ds 大转向 → C


# --------------------------------------------------------------------------- #
# 3) semantic_labels：负 ds fail-fast + Δκ 字段
# --------------------------------------------------------------------------- #
def test_negative_ds_fail_fast(tmp_path) -> None:
    arrays = {"expert_action": np.array([[2.0, 0.1], [-0.5, 0.2]], dtype=np.float32)}
    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(legacy_path, action=np.array([[2.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    with pytest.raises(SystemExit):
        kinematics_metadata(arrays, legacy_path)
    # legacy 负 ds 同样 fail-fast
    arrays2 = {"expert_action": np.array([[2.0, 0.1], [2.0, 0.2]], dtype=np.float32)}
    np.savez_compressed(legacy_path, action=np.array([[2.0, 0.0], [-0.1, 0.0]], dtype=np.float32))
    with pytest.raises(SystemExit):
        kinematics_metadata(arrays2, legacy_path)
    assert NEG_DS_EPS > 0


def test_delta_kappa_fields(tmp_path) -> None:
    expert = np.array([[2.0, 0.10], [1.0, -0.20]], dtype=np.float32)
    legacy = np.array([[2.0, 0.02], [1.0, -0.05]], dtype=np.float32)
    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(legacy_path, action=legacy)
    kin = kinematics_metadata({"expert_action": expert}, legacy_path)
    assert "expert_delta_kappa" in kin and "legacy_delta_kappa" in kin and "valid_delta_kappa" in kin
    assert np.allclose(kin["expert_delta_kappa"], expert[:, 1] / expert[:, 0] - legacy[:, 1] / legacy[:, 0], atol=1e-6)


# --------------------------------------------------------------------------- #
# 4) LQR 耦合遥测 / 曲率前馈（默认关）
# --------------------------------------------------------------------------- #
class _StubVehicle:
    FRONT_WHEELBASE = 1.05234
    REAR_WHEELBASE = 1.4166
    config = {"max_steering": 40.0}

    def __init__(self, position=(0.0, 0.0), heading=0.0, speed=6.0):
        self.position = np.array(position, dtype=np.float64)
        self.heading_theta = float(heading)
        self.speed = float(speed)


@pytest.fixture(autouse=True)
def _no_policy_mark(monkeypatch):
    monkeypatch.setattr(BasePolicy, "show_policy_mark", lambda self: None)


def test_coupling_telemetry_default_off_and_enabled() -> None:
    actions = np.array([[2.0, 0.10]] * 6)
    off = LqrTracker(_StubVehicle(), 0)
    off.set_reference(actions, base_pose=(0.0, 0.0, 0.0))
    off.act()
    for key in ("lqr_speed_error", "lqr_curvature_error", "preview_index_delta", "preview_plan_step_transition"):
        assert key not in off.action_info  # 默认关
    on = LqrTracker(_StubVehicle(), 0, telemetry_coupling=True)
    on.set_reference(actions, base_pose=(0.0, 0.0, 0.0))
    on.act()
    on.act()
    for key in ("lqr_speed_error", "lqr_curvature_error", "preview_index_delta", "preview_plan_step_transition"):
        assert key in on.action_info, key
    assert abs(on.action_info["lqr_speed_error"] - (on.action_info["ref_chord_speed"] - 6.0)) < 1e-6


def test_curvature_feedforward_default_off_bitwise() -> None:
    """默认关：同一参考下动作逐位一致；开启后（反馈增益置极小以避开饱和）动作改变。"""
    actions = np.array([[2.0, 0.10]] * 6)
    tiny = dict(q_lateral=1e-9, q_heading=1e-9, r_steer=1e9)
    base = LqrTracker(_StubVehicle(), 0, **tiny)
    base.set_reference(actions, base_pose=(0.0, 0.0, 0.0))
    a0 = np.asarray(base.act(), dtype=np.float64)
    base2 = LqrTracker(_StubVehicle(), 0, **tiny)
    base2.set_reference(actions, base_pose=(0.0, 0.0, 0.0))
    assert np.array_equal(a0, np.asarray(base2.act(), dtype=np.float64))  # 默认关 → 确定性逐位
    ff = LqrTracker(_StubVehicle(), 0, curvature_feedforward=True, **tiny)
    ff.set_reference(actions, base_pose=(0.0, 0.0, 0.0))
    a1 = np.asarray(ff.act(), dtype=np.float64)
    assert not np.allclose(a0, a1), "曲率前馈应改变动作（非默认配置）"
    expected = float(np.arctan(ff.wheelbase * ff.action_info["lqr_ref_curvature"]) / ff.max_steer_angle_rad)
    assert abs(float(a1[0]) - expected) < 1e-9
