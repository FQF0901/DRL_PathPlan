"""精确查表窗口回归（schema v2）：过滤洞、头部 valid、wm_valid 门控、身份匹配。

对应交付项「查表正确性（含过滤洞）」：

- **禁止按行位置取窗口**：帧被过滤后行距 ≠ 时间距；本测试构造 step 洞（缺 5、缺 15），
  断言 ``hist_valid``/``valid`` 精确反映缺帧，episode 头部不得伪造 ``valid=1``；
- ``wm_valid`` 逐 horizon 门控目标帧可用性（``od_mask`` 必须清零）；
- OD 槽位用 ``od_id`` 身份匹配（同 id 才算目标；离场/入场不串位）。
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.stages import build_future, build_history
from tests.v2_synthetic import make_v2_arrays


class _ArraysWrapper(dict):
    """让 build_* 的 lane 缓存键稳定（同对象身份）。"""


def _fixture() -> dict:
    arrays, _ = make_v2_arrays(episodes=2, steps_per_episode=7, stride=5, seed=0)
    return arrays


def _fixture(episodes: int = 2, steps_per_episode: int = 7) -> dict:
    arrays, _ = make_v2_arrays(episodes=episodes, steps_per_episode=steps_per_episode, stride=5, seed=0)
    return arrays


def test_history_exact_lookup_with_filtered_hole_and_head_valid() -> None:
    arrays = _ArraysWrapper(_fixture(episodes=1, steps_per_episode=7))
    # 过滤掉 step=15（第 4 行）→ 时间洞；行距不再等于时间距
    keep = np.array([0, 1, 2, 4, 5, 6])
    sliced = _ArraysWrapper(
        {key: (value[keep] if value.shape[0] == 7 else value) for key, value in arrays.items()}
    )
    # 当前 step=25：窗口网格 0,5,10,15,20,25 → 15 缺失
    history_hole = build_history(sliced, np.array([4]), episode_key="episode_id", step_key="step", stride=5)
    valid_hole = np.asarray(history_hole["hist_valid"])[0]
    assert valid_hole.tolist() == [1.0, 1.0, 1.0, 0.0, 1.0, 1.0], "过滤洞必须 valid=0（精确查表）"
    mask = np.asarray(history_hole["od_hist_mask"])[0]
    assert mask[3].sum() == 0.0, "缺帧槽位掩码必须清零"
    # 越界帧（step=10 的窗口 -15..10）→ 前 3 个 slot 无效、掩码清零
    history_edge = build_history(sliced, np.array([2]), episode_key="episode_id", step_key="step", stride=5)
    assert np.asarray(history_edge["hist_valid"])[0].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert np.asarray(history_edge["od_hist_mask"])[0, :3].sum() == 0.0
    # 头部：step=0 的帧只有 1 个真实历史槽（不得按行位置伪造 valid=1）
    history_head = build_history(sliced, np.array([0]), episode_key="episode_id", step_key="step", stride=5)
    assert np.asarray(history_head["hist_valid"])[0].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_future_exact_lookup_and_wm_valid_gating() -> None:
    arrays = _fixture()
    future = build_future(
        arrays, np.array([0]), episode_key="episode_id", step_key="step", stride=5
    )
    assert np.asarray(future["valid"])[0].tolist() == [1.0] * 6
    # 显式 wm_valid 只允许 h2/h4 → od_mask 只在对应 horizon 非零（并被查表存在性取交）
    wm = np.zeros((1, 6), dtype=np.float32)
    wm[0, 1] = 1.0
    wm[0, 3] = 1.0
    gated = build_future(
        arrays, np.array([0]), episode_key="episode_id", step_key="step", stride=5, wm_valid=wm
    )
    assert np.asarray(gated["wm_valid"])[0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    assert np.asarray(gated["od_mask"])[0, 0].sum() == 0.0
    assert np.asarray(gated["od_mask"])[0, 1].sum() > 0.0
    assert np.asarray(gated["od_mask"])[0, 3].sum() > 0.0
    assert np.asarray(gated["valid"])[0].tolist() == [1.0] * 6, "valid=步存在性，不受 wm_valid 影响"


def test_future_step_holes_truncate_valid() -> None:
    arrays = _fixture()
    # 只保留 episode 0 的前 3 帧（step 0/5/10）→ 未来只有 k=1,2 存在
    keep = np.isin(arrays["episode_id"], [0]) & (arrays["step"] <= 10)
    sliced = {key: (value[keep] if value.shape[0] == keep.size else value) for key, value in arrays.items()}
    future = build_future(sliced, np.array([0]), episode_key="episode_id", step_key="step", stride=5)
    assert np.asarray(future["valid"])[0].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert np.asarray(future["od_mask"])[0, 2:].sum() == 0.0
    assert np.asarray(future["wm_valid"])[0].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_future_identity_matching_by_od_id() -> None:
    """v2 OD 槽位 = track id：目标帧同槽位换车（id 不同）不得作为该槽位的目标。"""
    arrays = _fixture(episodes=1, steps_per_episode=7)
    target_rows = np.where((arrays["episode_id"] == 0) & (arrays["step"] == 5))[0]
    assert target_rows.size == 1
    row = int(target_rows[0])
    arrays["od_id"][row, 0] = 999  # 车 1 离场，槽位被新车（3999）占用
    arrays["od"][row, 0, 0] = 123.0
    arrays["od_presence"][row, 0] = 1.0

    future = build_future(arrays, np.array([0]), episode_key="episode_id", step_key="step", stride=5)
    mask0 = np.asarray(future["od_mask"])[0, 0]
    assert mask0[0] == 0.0, "不同 id 的槽位不得作为目标（防串位）"
    assert mask0[1] == 1.0, "同 id 槽位仍应是有效目标"
    # 空槽/离场（presence=0）同样清零
    arrays["od_presence"][row, 1] = 0.0
    future2 = build_future(arrays, np.array([0]), episode_key="episode_id", step_key="step", stride=5)
    assert np.asarray(future2["od_mask"])[0, 0, 1] == 0.0


def test_local_equivalence_without_lane_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """lane A 不可用时本地等价实现语义一致（valid/wm_valid 口径）。"""
    import pipeline.stages as stages

    arrays = _fixture()
    lane_future = build_future(arrays, np.array([1, 3]), episode_key="episode_id", step_key="step", stride=5)
    monkeypatch.setattr(stages, "_lane_frames_api", lambda: None)
    stages._FRAMES_LOOKUP_CACHE.clear()
    local_future = build_future(arrays, np.array([1, 3]), episode_key="episode_id", step_key="step", stride=5)
    np.testing.assert_allclose(lane_future["valid"], local_future["valid"])
    np.testing.assert_allclose(lane_future["wm_valid"], local_future["wm_valid"])
    np.testing.assert_allclose(lane_future["od_mask"], local_future["od_mask"])
    # 缺帧特征补位方式不同（lane 全零 / 本地最近帧），但 mask=0 → 只比较有效步
    valid_gate = np.asarray(lane_future["valid"])[:, :, None, None]
    np.testing.assert_allclose(
        lane_future["od_fut"] * valid_gate, local_future["od_fut"] * valid_gate, atol=1e-4
    )

    lane_hist = build_history(arrays, np.array([2]), episode_key="episode_id", step_key="step", stride=5)
    stages._FRAMES_LOOKUP_CACHE.clear()
    local_hist = build_history(arrays, np.array([2]), episode_key="episode_id", step_key="step", stride=5)
    np.testing.assert_allclose(lane_hist["hist_valid"], local_hist["hist_valid"])
    np.testing.assert_allclose(lane_hist["od_hist_mask"], local_hist["od_hist_mask"])
