"""R41-I 守卫：``path_*`` 真口径列（``--long-exog-source v7obs``）与两口径的**逐位不变**默认。

依据：``.slim/deepwork/p1_v2/round41/i_report.md`` §1
- 真口径（env 生产者，``i_truth.py`` 仿真判定 100% 命中）= ``lane_semantic`` 列 4/5/11；
- dataset 口径（``datasets/path_exec_v1`` 的 I1 ``long_exog`` 列 1/2）**左右反号**：
  I1 用**左正** lat 代入 ``left = W/2 + lat``（真值 = 右正 lat）；
  d2end = 未封顶 remaining vs env = 拓扑封顶列 11。

覆盖（自包含 CPU；不读 ``runs/`` 重资产）：
1. ``build_obs`` 默认（``long_exog_override=None``）与显式覆盖前后的键值语义；
2. ``long_exog_override_from_store``：取值对齐 + 缺行 **fail-fast**（禁静默回退）；
3. CLI：``--long-exog-source`` 值域 + 与 ``--obs-source v7obs`` 的冗余组合 fail-fast。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.residual_dataset import build_obs  # noqa: E402
from tools.train_lateral import long_exog_override_from_store, parse_args  # noqa: E402


class _StubStore:
    """``V7ObsStore`` 最小契约：``map_rows`` + ``._arrays``（本 lane 只消费 3 列 + present）。"""

    def __init__(self, arrays: dict, keys: list[tuple[int, int]]):
        self._arrays = {k: np.asarray(v) for k, v in arrays.items()}
        self.lookup = {key: i for i, key in enumerate(keys)}

    def map_rows(self, keys):
        rows = np.full(len(keys), -1, dtype=np.int64)
        present = np.zeros(len(keys), dtype=bool)
        for i, key in enumerate(keys):
            row = self.lookup.get((int(key[0]), int(key[1])))
            if row is not None:
                rows[i] = int(row)
                present[i] = True
        return rows, present


def _rows(n: int = 4) -> dict:
    anchor = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    return {
        "n": n,
        "episode_id": np.arange(n, dtype=np.int64),
        "step": np.arange(n, dtype=np.int64) * 5,
        "spec_id": np.zeros(n, dtype=np.int64),
        "anchor": [anchor.copy() for _ in range(n)],
    }


def _stub_store_values() -> tuple[_StubStore, list[tuple[int, int]]]:
    keys = [(0, 0), (1, 5), (2, 10)]
    arrays = {
        "path_left_margin": [5.10, 5.20, 5.30],
        "path_right_margin": [5.40, 5.50, 5.60],
        "path_distance_to_target_lane_end": [12.0, 13.0, 14.0],
        "path_left_margin__present": [1.0, 1.0, 0.0],
        "path_right_margin__present": [1.0, 1.0, 0.0],
        "path_distance_to_target_lane_end__present": [1.0, 0.0, 1.0],
    }
    return _StubStore(arrays, keys), keys


def test_build_obs_default_bitwise_unchanged() -> None:
    """默认路径（无覆盖）逐位不变；覆盖只在显式给出时生效。"""
    rows = _rows()
    idx = np.arange(4, dtype=np.int64)
    rows["long_exog"] = np.tile(np.array([8.0, 5.25, 5.25, 0.0, 20.0, 30.0, 0.5, 17.5], np.float32), (4, 1))
    rows["long_present"] = np.ones((4, 8), np.float32)
    first = build_obs(rows, idx, device="cpu", base_dataset=None)
    again = build_obs(rows, idx, device="cpu", base_dataset=None)
    for key in first:
        assert torch.equal(first[key], again[key]), key
    assert float(first["path_left_margin"][0]) == pytest.approx(5.25)

    override = {"path_left_margin": np.array([5.10, 5.20, 5.30, 5.40], np.float32),
                "path_left_margin__present": np.array([1.0, 1.0, 0.0, 1.0], np.float32)}
    third = build_obs(rows, idx, device="cpu", base_dataset=None, long_exog_override=override)
    assert torch.equal(third["path_left_margin"], torch.tensor([5.10, 5.20, 5.30, 5.40]))
    assert torch.equal(third["path_left_margin__present"], torch.tensor([1.0, 1.0, 0.0, 1.0]))
    assert torch.equal(third["path_right_margin"], first["path_right_margin"])  # 未覆盖键不变


def test_build_obs_override_length_fail_fast() -> None:
    rows = _rows()
    idx = np.arange(4, dtype=np.int64)
    rows["long_exog"] = np.zeros((4, 8), np.float32)
    rows["long_present"] = np.ones((4, 8), np.float32)
    with pytest.raises(SystemExit):
        build_obs(rows, idx, device="cpu", base_dataset=None,
                  long_exog_override={"path_left_margin": np.zeros(3, np.float32)})


def test_override_from_store_values_and_missing_fail_fast() -> None:
    store, keys = _stub_store_values()
    rows = _rows()
    out = long_exog_override_from_store(store, rows, np.array([0, 1, 2], dtype=np.int64))
    assert set(out) == {
        "path_left_margin", "path_right_margin", "path_distance_to_target_lane_end",
        "path_left_margin__present", "path_right_margin__present",
        "path_distance_to_target_lane_end__present",
    }
    assert np.allclose(out["path_left_margin"], [5.10, 5.20, 5.30])
    assert np.allclose(out["path_right_margin"], [5.40, 5.50, 5.60])
    assert np.allclose(out["path_left_margin__present"], [1.0, 1.0, 0.0])
    # 缺行（row 3 不在 store）⇒ fail-fast（禁静默回退 dataset 口径）
    with pytest.raises(SystemExit):
        long_exog_override_from_store(store, rows, np.array([0, 1, 2, 3], dtype=np.int64))


def test_cli_long_exog_source_axis() -> None:
    cfg = str(ROOT / "config/lateral_train.yaml")
    base = parse_args(["--config", cfg])
    assert base.long_exog_source == "dataset"
    yaml_default = parse_args(["--config", cfg, "--long-exog-source", "v7obs"])
    assert yaml_default.long_exog_source == "v7obs"
    assert yaml_default.obs_source == "base"
    # 冗余组合（aligned 已整批取 store）⇒ fail-fast
    with pytest.raises(SystemExit):
        parse_args(["--config", cfg, "--assembly", "aligned", "--long-exog-source", "v7obs"])
    with pytest.raises(SystemExit):
        parse_args(["--config", cfg, "--long-exog-source", "bogus"])


# =========================================================================== #
# 4. 几何轴：``fixed_t0margins`` / ``corrected_t0margins``（R41-I 新增正确档）
# =========================================================================== #
def test_t0margins_axis_keeps_t0_identity_while_fixed_swaps() -> None:
    """``fixed_t0margins``：footer 修复照旧（foot 走 mask），margin = t0 真口径（== production）；
    ``fixed``（历史档）在 k=0 与 t0 row 相差 ``2·lat_k``（= 缺陷，冻结登记）。"""
    from net.path_head import step_geometry

    y_lane = 0.285  # 车道中心线 ego t0 系 y（lat = −y 左正 ⇒ lat = −0.285）
    anchor = torch.zeros((1, 4, 2))
    anchor[0, :, 0] = torch.arange(4, dtype=torch.float32) * 3.0
    anchor[0, :, 1] = y_lane
    mask = torch.ones((1, 4))
    # t0 row 值（env/真口径）：left = W/2 + lat = 5.25 − 0.285；right = 5.25 + 0.285
    exog = torch.zeros((1, 8)); exog[0, 1] = 5.25 - 0.285; exog[0, 2] = 5.25 + 0.285
    present = torch.ones((1, 8))
    pose = torch.zeros((1, 3))
    progress = torch.zeros((1,))
    prod = step_geometry(anchor, mask, exog, present, pose, progress, geometry_mode="production")
    t0 = step_geometry(anchor, mask, exog, present, pose, progress, geometry_mode="fixed_t0margins")
    fixed = step_geometry(anchor, mask, exog, present, pose, progress, geometry_mode="fixed")
    # k0：t0margins == production（t0 恒等）；fixed 互换（差 2·lat = 0.57）
    for key in ("long_exog", "long_present"):
        assert torch.equal(t0[key], prod[key]), key
    lat_k0 = float(t0["lateral_offset"][0])
    assert lat_k0 == pytest.approx(-0.285, abs=1e-5)
    assert float(fixed["long_exog"][0, 1]) == pytest.approx(float(prod["long_exog"][0, 2]), abs=1e-5)
    assert float(fixed["long_exog"][0, 1]) - float(t0["long_exog"][0, 1]) == pytest.approx(
        2.0 * lat_k0, abs=1e-5)
    # 键集一致
    assert set(prod) == set(t0) == set(fixed)


def test_t0margins_axis_accepted_by_canonical_entry_and_instance_config() -> None:
    from net.path_head import (
        normalize_readout_lane_geometry,
        readout_lane_geometry_internal,
    )

    assert normalize_readout_lane_geometry("corrected_t0margins") == "corrected_t0margins"
    assert normalize_readout_lane_geometry(" CORRECTED_T0MARGINS ") == "corrected_t0margins"
    assert readout_lane_geometry_internal("corrected_t0margins") == "fixed_t0margins"
