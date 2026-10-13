"""R21-28 契约测试：横纵耦合审计（N0/N1/C 的 κ/legacy-rollout 升级、标签元数据/manifest、模型只读诊断、LQR 遥测）。"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from metadrive.policy.base_policy import BasePolicy

from env.tracking import LqrTracker
from pipeline.residual_classes import CLASS_C, CLASS_N0, classify_rows, compute_thresholds
from pipeline.semantic_labels import KAPPA_VALID_DS_M, build_manifest, kinematics_metadata, load_samples_v2


def _features(**overrides):
    base = {
        "row_type": np.zeros(4, dtype=np.int64),
        "res_ds": np.array([0.05, 0.05, 0.05, 0.05]),
        "res_dth": np.array([0.002, 0.002, 0.002, 0.002]),
        "future_valid": np.ones(4, dtype=bool),
        "future_lat_max": np.full(4, 0.2),
        "future_head_max": np.full(4, 0.02),
        "future_speed_min": np.full(4, 5.0),
        "episode_id": np.zeros(4, dtype=np.int64),
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# 28B：res_kappa / legacy_rollout_safe
# --------------------------------------------------------------------------- #
def test_legacy_rollout_safe_gates_n0() -> None:
    """legacy 同态 rollout 不安全 → 即使动作残差小也不得判 N0（C）。"""
    features = _features(legacy_rollout_safe=np.array([True, False, True, False]))
    thresholds = compute_thresholds(features["res_ds"], features["res_dth"], np.ones(4, dtype=bool))
    labels = classify_rows(features, thresholds)
    assert list(labels) == [CLASS_N0, CLASS_C, CLASS_N0, CLASS_C]


def test_res_kappa_gates_n0() -> None:
    """κ 残差超阈（小 ds 放大）→ 不得判 N0，即使 |Δds|/|Δdθ| 都很小。"""
    features = _features(
        res_kappa=np.array([0.001, 0.5, 0.001, 0.5]),
        kappa_valid=np.ones(4, dtype=bool),
        legacy_rollout_safe=np.ones(4, dtype=bool),
    )
    thresholds = compute_thresholds(
        features["res_ds"], features["res_dth"], np.ones(4, dtype=bool),
        res_kappa=features["res_kappa"], kappa_valid=features["kappa_valid"],
    )
    labels = classify_rows(features, thresholds)
    # κ 残差超阈（P50）⇒ 至少不得判 N0（落 N1 或 C）
    assert labels[0] == CLASS_N0 and labels[2] == CLASS_N0
    assert labels[1] != CLASS_N0 and labels[3] != CLASS_N0
    assert "tau_kappa" in thresholds
    # κ 残差超 P90 ⇒ 落 C（不允许小残差）
    features_c = _features(
        res_kappa=np.array([0.001, 99.0, 0.001, 0.001]),
        kappa_valid=np.ones(4, dtype=bool),
        legacy_rollout_safe=np.ones(4, dtype=bool),
    )
    thresholds_c = compute_thresholds(
        features_c["res_ds"], features_c["res_dth"], np.ones(4, dtype=bool),
        res_kappa=features_c["res_kappa"], kappa_valid=features_c["kappa_valid"],
    )
    labels_c = classify_rows(features_c, thresholds_c)
    assert labels_c[1] == CLASS_C
    assert labels_c[0] == CLASS_N0 and labels_c[2] == CLASS_N0 and labels_c[3] == CLASS_N0


def test_kappa_invalid_rows_skip_kappa_gate() -> None:
    """kappa_valid=False（ds < 0.5）→ κ 门不生效（数值不可信）。"""
    features = _features(
        res_kappa=np.array([9.9, 9.9, 0.001, 0.001]),
        kappa_valid=np.array([False, False, True, True]),
        legacy_rollout_safe=np.ones(4, dtype=bool),
    )
    thresholds = compute_thresholds(
        features["res_ds"], features["res_dth"], np.ones(4, dtype=bool),
        res_kappa=features["res_kappa"], kappa_valid=features["kappa_valid"],
    )
    labels = classify_rows(features, thresholds)
    assert list(labels) == [CLASS_N0, CLASS_N0, CLASS_N0, CLASS_N0]


# --------------------------------------------------------------------------- #
# 28B：semantic_labels 元数据 / manifest
# --------------------------------------------------------------------------- #
def test_kinematics_metadata_and_manifest(tmp_path) -> None:
    expert = np.array([[2.0, 0.10], [1.0, -0.20], [0.2, 0.05]], dtype=np.float32)  # 末行 ds<0.5
    legacy = np.array([[2.0, 0.02], [0.8, -0.05], [0.2, 0.01]], dtype=np.float32)
    arrays = {"expert_action": expert}
    legacy_path = tmp_path / "legacy.npz"
    np.savez_compressed(legacy_path, action=legacy)
    kin = kinematics_metadata(arrays, legacy_path)
    assert np.allclose(kin["expert_kappa"], expert[:, 1] / expert[:, 0], atol=1e-6)
    assert np.allclose(kin["legacy_kappa"], legacy[:, 1] / legacy[:, 0], atol=1e-6)
    assert kin["kappa_valid"].tolist() == [True, True, False]
    assert KAPPA_VALID_DS_M == 0.5
    labels_path = tmp_path / "labels.npz"
    np.savez_compressed(labels_path, label_state_version=np.asarray(["same_state_v2"]))
    manifest = build_manifest(labels_path, arrays, rows=3, covered_b_rows=2, excluded_b_rows=1)
    assert manifest["backend"] and manifest["query_timing"]
    assert len(manifest["label_hash"]) == 16 and len(manifest["expert_hash"]) == 16
    assert manifest["rows"] == 3 and manifest["covered_b_rows"] == 2


def test_load_samples_v2_returns_kinematics_and_manifest(tmp_path) -> None:
    n = 4
    arrays = {
        "episode_id": np.array([1, 1, 7, 7], dtype=np.int64),
        "step": np.array([0, 1, 10, 15], dtype=np.int64),
        "row_type": np.array([0, 2, 1, 1], dtype=np.int64),
        "expert_action": np.array([[2.0, 0.1], [2.0, 0.2], [9.0, 9.0], [9.0, 9.0]], dtype=np.float32),
        "geometry_code": np.zeros(n, dtype=np.int64),
        "split": np.zeros(n, dtype=np.int64),
        "speed": np.full(n, 8.0, dtype=np.float32),
    }
    np.savez_compressed(tmp_path / "samples.npz", **arrays)
    np.savez_compressed(
        tmp_path / "labels.npz",
        episode_id=np.array([7, 7], dtype=np.int64), step=np.array([10, 15], dtype=np.int64),
        action=np.zeros((2, 2), dtype=np.float32),
        perturbed_action=np.array([[1.5, 0.3], [1.2, -0.1]], dtype=np.float32),
        label_pose=np.zeros((2, 3), dtype=np.float32),
        label_state_version=np.asarray(["same_state_v2"]),
    )
    np.savez_compressed(tmp_path / "legacy.npz", action=np.full((n, 2), 2.0, dtype=np.float32))
    (tmp_path / "contract.json").write_text(json.dumps({"v1_exposed_b_rows": 0, "excluded_b_rows": []}), encoding="utf-8")
    loaded = load_samples_v2(tmp_path, tmp_path / "labels.npz", tmp_path / "contract.json",
                             legacy_actions_path=tmp_path / "legacy.npz")
    kin = loaded["kinematics"]
    assert set(("expert_ds", "expert_dtheta", "expert_kappa", "legacy_ds", "legacy_dtheta",
                "legacy_kappa", "kappa_valid")) <= set(kin)
    assert loaded["label_meta"]["kappa_valid_rows"] == int(kin["kappa_valid"].sum())
    assert loaded["manifest"]["label_state_version"] == "same_state_v2"
    # B 行 v2 覆盖生效（旧哨兵 9.0 不出现）
    assert not np.any(np.isclose(loaded["arrays"]["expert_action"], 9.0))


# --------------------------------------------------------------------------- #
# 28A：模型只读诊断
# --------------------------------------------------------------------------- #
def test_model_forward_physics_diagnostics() -> None:
    from tests.test_r21_3a_semantic_wiring import _activate_residual, _semantic_model, _semantic_obs

    model = _semantic_model()
    obs = _semantic_obs(batch=3, seed=41)
    _activate_residual(model, context=0.2, token=0.2)
    with torch.no_grad():
        out = model(obs, rollout=True, world_model=False)
        assert out["final_ds"].shape == (3,)
        assert torch.allclose(out["final_ds"], out["action_mu"][:, 0], atol=0)
        assert torch.allclose(out["final_kappa"], out["action_mu"][:, 1] / out["action_mu"][:, 0].clamp_min(1e-6), atol=1e-6)
        assert out["valid_kappa"].dtype == torch.bool and out["valid_kappa"].shape == (3,)
        assert out["plan_valid_kappa"].shape == out["plan_kappa"].shape
        dec = model.semantic_action_decomposition(obs)
        assert torch.allclose(dec["final_dtheta"], dec["final_action"][:, 1], atol=0)
        assert torch.allclose(dec["legacy_ds"], dec["legacy_action"][:, 0], atol=0)
        assert torch.allclose(dec["semantic_delta_kappa"], dec["final_kappa"] - dec["legacy_kappa"], atol=1e-6)
        assert dec["valid_kappa"].dtype == torch.bool
        # final = legacy + delta（逐位）
        assert torch.allclose(dec["final_action"] - dec["legacy_action"], dec["semantic_delta_action"], atol=1e-6)


# --------------------------------------------------------------------------- #
# 28A：LQR 遥测
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


def test_lqr_tracker_coupling_telemetry() -> None:
    tracker = LqrTracker(_StubVehicle(), 0)
    straight = np.array([[2.0, 0.0]] * 6)
    tracker.set_reference(straight, base_pose=(0.0, 0.0, 0.0))
    tracker.act()
    info = tracker.action_info
    for key in ("lqr_ref_curvature", "lqr_actual_curvature", "lqr_preview_plan_step",
                "throttle_raw", "throttle_clipped", "ref_chord_speed", "steer_raw"):
        assert key in info, key
    assert abs(info["lqr_ref_curvature"]) < 1e-6  # 直行 → 曲率 ~0
    assert info["lqr_preview_plan_step"] == info["lqr_preview_index"] // 5
    assert abs(info["throttle_clipped"]) <= 1.0
    # 转弯参考：κ_ref ≈ dθ/ds（每步 0.1 rad / 2 m = 0.05）
    turning = np.array([[2.0, 0.10]] * 6)
    tracker2 = LqrTracker(_StubVehicle(), 0)
    tracker2.set_reference(turning, base_pose=(0.0, 0.0, 0.0))
    tracker2.act()
    assert abs(tracker2.action_info["lqr_ref_curvature"] - 0.05) < 0.01
