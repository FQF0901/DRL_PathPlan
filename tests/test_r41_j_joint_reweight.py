#!/usr/bin/env python3
"""R41-J 回归测试：① 评测侧 LqrTracker 覆盖白名单 + ② 联合区重加权/过采样（默认关/逐位）。

覆盖：
1. ``pipeline.eval_runner.lqr_tracker_kwargs``：空 ⇒ {}（默认调用逐位不变）；白名单透传；
   未知键 fail-fast；
2. ``tools.train_lateral.joint_region_mask``：region 判定（v<5 ∧ |lat|>0.2）、缺行/NaN 排除、
   统计块字段（训练行口径）；
3. ``tools.train_lateral.joint_oversample_indices``：factor 取整、region 命中、factor<2 ⇒ 空；
4. CLI 守卫：``--joint-reweight`` 非 off 需 ``--obs-source v7obs``（legacy fail-fast）、
   factor ≤1 fail-fast、默认 off/5.0/0.2/4.0。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.eval_runner import LQR_TRACKER_KEYS, lqr_tracker_kwargs  # noqa: E402
from tools.train_lateral import joint_oversample_indices, joint_region_mask, parse_args  # noqa: E402


class _StubStore:
    """``V7ObsStore`` 最小契约：``n`` + ``_arrays['lane_semantic']``（(N,1,12) 落盘形）。"""

    def __init__(self, lat: np.ndarray):
        n = int(lat.size)
        sem = np.zeros((n, 1, 12), dtype=np.float32)
        sem[:, 0, 0] = lat
        self.n = n
        self._arrays = {"lane_semantic": sem}


def _args(**kwargs) -> SimpleNamespace:
    base = {"joint_reweight": "weight", "joint_factor": 4.0, "joint_speed_max": 5.0, "joint_lat_min": 0.2}
    base.update(kwargs)
    return SimpleNamespace(**base)


def test_lqr_tracker_kwargs_default_is_empty_bitwise() -> None:
    assert lqr_tracker_kwargs(None) == {}
    assert lqr_tracker_kwargs({}) == {}
    # 白名单 = LqrTracker 构造键；J 的预瞄口径旋钮必须在列
    for key in ("lookahead_time", "lookahead_base", "lookahead_min", "lookahead_max",
                "q_lateral", "q_heading", "r_steer", "steer_gain", "reschedule_dv"):
        assert key in LQR_TRACKER_KEYS


def test_lqr_tracker_kwargs_passthrough_and_fail_fast() -> None:
    overrides = {"lookahead_time": 0.4, "r_steer": 2.0}
    assert lqr_tracker_kwargs(overrides) == overrides
    with pytest.raises(ValueError):
        lqr_tracker_kwargs({"not_a_key": 1.0})


def test_joint_region_mask_definition_and_stats() -> None:
    # rows 0..4：lat = [-0.5, -0.1, 0.3, 0.0, -0.9]；行 4 缺失（rows=-1）；speed = [3, 3, 3, 3, 3]
    store = _StubStore(np.array([-0.5, -0.1, 0.3, 0.0, -0.9], dtype=np.float32))
    view = {"v7_store": store, "v7_rows": np.array([0, 1, 2, 3, -1], dtype=np.int64)}
    data = {
        "speed": np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float64),
        "train_mask": np.array([True, True, True, False, True]),
    }
    region, joint = joint_region_mask(_args(), data, view)
    assert region.tolist() == [True, False, True, False, False]  # 缺行/|lat|≤0.2/高速 排除
    assert joint["n_region"] == 2 and joint["n_region_train"] == 2
    assert joint["lat_min"] == 0.2 and joint["speed_max"] == 5.0
    assert joint["v_p50_region_train"] == 3.0
    assert abs(joint["abs_lat_p50_region_train"] - 0.4) < 1e-6  # (0.5, 0.3) → p50 ≈ 0.4
    assert abs(joint["region_train_frac"] - 2 / 4) < 1e-9  # train_mask 4 行（含缺行行的 mask）


def test_joint_region_speed_gate() -> None:
    store = _StubStore(np.array([-0.5, -0.5, -0.5], dtype=np.float32))
    view = {"v7_store": store, "v7_rows": np.array([0, 1, 2], dtype=np.int64)}
    data = {"speed": np.array([4.9, 5.1, 3.0]), "train_mask": np.array([True, True, True])}
    region, _ = joint_region_mask(_args(), data, view)
    assert region.tolist() == [True, False, True]


def test_joint_oversample_indices() -> None:
    # train_idx = main_ids 行序内的训练行位置；region 与 main_ids 同长
    region = np.array([True, False, True, False, True, False])
    train_idx = np.array([1, 3, 4], dtype=np.int64)
    assert joint_oversample_indices(train_idx, region, 1.0).tolist() == []
    assert joint_oversample_indices(train_idx, region, 3.0).tolist() == [4, 4]  # hit=[4]，×2 份
    assert joint_oversample_indices(train_idx, region, 2.4).tolist() == [4]  # round(2.4)=2 ⇒ 1 份
    assert joint_oversample_indices(train_idx, np.zeros(6, dtype=bool), 4.0).tolist() == []


def test_cli_joint_flags_defaults_and_guards() -> None:
    args = parse_args(["--dry-run"])
    assert args.joint_reweight == "off" and args.joint_factor == 4.0
    assert args.joint_speed_max == 5.0 and args.joint_lat_min == 0.2
    # aligned 装配下允许（obs_source 自动 = v7obs）
    args2 = parse_args(["--dry-run", "--assembly", "aligned", "--joint-reweight", "weight"])
    assert args2.joint_reweight == "weight" and args2.obs_source == "v7obs"
    # legacy（base obs）⇒ fail-fast
    with pytest.raises(SystemExit):
        parse_args(["--dry-run", "--joint-reweight", "weight"])
    # factor ≤ 1 ⇒ fail-fast
    with pytest.raises(SystemExit):
        parse_args(["--dry-run", "--assembly", "aligned", "--joint-reweight", "weight", "--joint-factor", "1.0"])
