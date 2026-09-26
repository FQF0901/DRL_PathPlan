"""``pipeline.frames`` 精确查表单测：过滤洞 / 网格锚定 / 唯一性与 stride 断言 /
未来 id 匹配与 wm_valid / SE(2) 对齐 / ``lookup_from_arrays``。

纯 NumPy：不 import MetaDrive（对齐规则由传入的 ``alignments`` 指定或惰性退化）。
"""

from __future__ import annotations

import numpy as np
import pytest

from env.obs.base import FrameAlignment
from pipeline.frames import (
    DEFAULT_K,
    DEFAULT_STRIDE,
    FrameLookup,
    build_future as frames_build_future,
    build_history as frames_build_history,
    lookup_from_arrays,
)
from pipeline.stages import build_future, build_history
from tests.v2_synthetic import make_v2_arrays

OD_ALIGN = FrameAlignment(point_pairs=((0, 1), ), vector_pairs=((2, 3), (4, 5)))
ALIGNMENTS = {"od": OD_ALIGN, "ld": FrameAlignment(point_pairs=((0, 1), ))}


def _arrays(episodes):
    """episodes: {episode_id: [steps]}；世界系常量目标（dx 换算见测试注释）。"""
    rows = [(eid, step) for eid, steps in episodes.items() for step in steps]
    n = len(rows)
    episode = np.array([row[0] for row in rows], dtype=np.int64)
    step = np.array([row[1] for row in rows], dtype=np.int64)
    pose = np.zeros((n, 3), dtype=np.float32)
    pose[:, 0] = step.astype(np.float32)  # ego 沿世界 x 每 step 前进 1 m
    od = np.zeros((n, 16, 9), dtype=np.float32)
    od[:, 0, 0] = 100.0  # 世界目标 x = 100 + step -> 存储系 dx 恒 100
    od[:, 0, 4] = 1.0
    od_mask = np.zeros((n, 16), dtype=np.float32)
    od_mask[:, 0] = 1.0
    ld = np.zeros((n, 16, 7), dtype=np.float32)
    ld_mask = np.zeros((n, 16), dtype=np.float32)
    od_id = np.full((n, 16), -1, dtype=np.int64)
    od_id[:, 0] = 7
    od_presence = np.zeros((n, 16), dtype=np.float32)
    od_presence[:, 0] = 1.0
    return {
        "episode_id": episode,
        "step": step,
        "pose": pose,
        "od": od,
        "od_mask": od_mask,
        "ld": ld,
        "ld_mask": ld_mask,
        "od_id": od_id,
        "od_presence": od_presence,
        "frame_usable": np.ones(n, dtype=np.float32),
    }


def _lookup(arrays, stride=5):
    return lookup_from_arrays(arrays, stride=stride, check=True)


# --------------------------------------------------------------------------- #
# 历史：过滤洞 / 网格锚定 / 对齐
# --------------------------------------------------------------------------- #

def test_history_exact_lookup_under_filtered_holes():
    """episode 0 有 15 的洞：历史按 (episode, step-5j) 精确查表，缺帧 valid=0/零填充。"""
    arrays = _arrays({0: [0, 5, 10, 20]})
    lookup = _lookup(arrays)
    window = lookup.build_history(0, 20, stride=5, k=6, alignments=ALIGNMENTS)
    steps = [-5, 0, 5, 10, 15, 20]
    expected_valid = [1.0 if t in (0, 5, 10, 20) else 0.0 for t in steps]
    np.testing.assert_allclose(window["hist_valid"], expected_valid)
    # 缺帧：特征 0、mask 0、od_id=-1；存在帧：id=7、mask=1
    np.testing.assert_allclose(window["od_hist"][0], 0.0)
    np.testing.assert_allclose(window["od_hist_mask"][0], 0.0)
    assert window["od_id_hist"][0, 0] == -1
    assert window["od_id_hist"][3, 0] == 7
    # 对齐到 base=20：aligned dx = 100 - (20 - step)
    for slot, step in enumerate(steps):
        if step in (0, 5, 10, 20):
            assert window["od_hist"][slot, 0, 0] == pytest.approx(100.0 - (20 - step), abs=1e-4)
    # 最新 slot 即目标帧自身
    assert window["od_hist"][-1, 0, 0] == pytest.approx(100.0)
    assert window["frame_index_hist"].tolist() == [-1, 0, 1, 2, -1, 3]


def test_history_anchors_to_stride_grid_for_offgrid_steps():
    """逐 env-step 数据（stride=1 网格）查询 step=11、采样 stride=5 → 锚到 floor 10。"""
    arrays = _arrays({0: list(range(12))})
    lookup = _lookup(arrays, stride=1)
    window = lookup.build_history(0, 11, stride=5, k=6, alignments=ALIGNMENTS)
    assert window["frame_index_hist"].tolist() == [-1, -1, -1, 0, 5, 10]
    np.testing.assert_allclose(window["hist_valid"], [0, 0, 0, 1, 1, 1])
    # 对齐到请求帧 step=11（当前帧），不是锚点帧 step=10：aligned dx = 100 + step - 11
    assert window["od_hist"][-1, 0, 0] == pytest.approx(99.0)
    assert window["od_hist"][-2, 0, 0] == pytest.approx(94.0)
    # pose_hist 保留各源帧自己的位姿（对齐前）
    assert window["pose_hist"][-1, 0] == pytest.approx(10.0)


def test_history_rollout_buffer_style_interval_1():
    arrays = _arrays({0: [100, 101, 102, 103, 104, 105, 106]})
    lookup = _lookup(arrays, stride=1)
    window = lookup.build_history(0, 106, stride=1, k=6, alignments=ALIGNMENTS)
    assert window["frame_index_hist"].tolist() == [1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(window["hist_valid"], np.ones(6))


# --------------------------------------------------------------------------- #
# 断言
# --------------------------------------------------------------------------- #

def test_uniqueness_assertion():
    arrays = _arrays({0: [0, 5]})
    arrays["step"] = np.array([0, 0], dtype=np.int64)
    with pytest.raises(ValueError, match="不唯一"):
        _lookup(arrays)


def test_stride_assertion_allows_holes_but_rejects_misaligned_gap():
    arrays = _arrays({0: [0, 5, 15, 20]})  # 洞为 stride 倍数：允许
    _lookup(arrays, stride=5)
    bad = _arrays({0: [0, 5, 12]})  # 7 不是 5 的倍数：拒绝
    with pytest.raises(ValueError, match="stride"):
        _lookup(bad, stride=5)
    FrameLookup(bad["episode_id"], bad["step"], check=False)  # check=False 不抛


# --------------------------------------------------------------------------- #
# 未来：id 匹配 / presence / wm_valid
# --------------------------------------------------------------------------- #

def _future_arrays():
    steps = list(range(0, 35, 5))
    arrays = _arrays({1: steps})
    n = len(steps)
    # slot 1 的对象在 step>=15 出现（t0 不存在 -> 不应被监督），id=9
    for row, step in enumerate(steps):
        if step >= 15:
            arrays["od"][row, 1, 0] = 42.0
            arrays["od_mask"][row, 1] = 1.0
            arrays["od_id"][row, 1] = 9
            arrays["od_presence"][row, 1] = 1.0
    # step=10 的目标帧标记不可用（模拟过滤洞外还存在的"脏帧"）
    arrays["frame_usable"][steps.index(10)] = 0.0
    assert n == len(steps)
    return arrays


def test_future_identity_matching_and_wm_valid():
    arrays = _future_arrays()
    lookup = _lookup(arrays)
    future = lookup.build_future(1, 0, stride=5, k=6, alignments=ALIGNMENTS)
    assert future["valid"].tolist() == [1.0] * 6
    # step=10 不可用 -> 第 2 个 horizon 的 wm_valid=0，其 mask 清零
    assert future["wm_valid"].tolist() == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert future["od_mask"][1].tolist() == [0.0] * 16
    # t0 已存在的对象（slot0, id=7）：同 id 命中，全部 horizon 可监督
    assert future["od_mask"][:, 0].tolist() == [1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    # 未来才出现的对象（slot1, id=9；t0 od_id=-1）：id 不匹配 -> 永远不监督
    assert future["od_mask"][:, 1].tolist() == [0.0] * 6
    assert future["od_id_t0"][0] == 7 and future["od_id_t0"][1] == -1
    assert future["od_id_fut"][3, 1] == 9  # 原始 id 仍可见（调试/其它用途）
    assert future["od_presence_fut"][3, 1] == 1.0
    # 对齐：未来第 1 帧（step=5，世界 x=105）在 t0（step=0）系 dx=105
    assert future["od_fut"][0, 0, 0] == pytest.approx(105.0, abs=1e-4)
    assert future["od_fut"][1, 0, 0] == pytest.approx(110.0, abs=1e-4)
    # 缺 horizon（超出 episode）valid=0、特征 0
    short = lookup.build_future(1, 25, stride=5, k=6, alignments=ALIGNMENTS)
    assert short["valid"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(short["od_fut"][1:], 0.0)


def test_future_mask_raw_keeps_presence_free_slot_mask():
    arrays = _future_arrays()
    lookup = _lookup(arrays)
    future = lookup.build_future(1, 0, stride=5, k=6, alignments=ALIGNMENTS)
    # od_mask_raw 不做 id 匹配；slot1 在 step>=15 的 horizon 上有值
    assert future["od_mask_raw"][3, 1] == 1.0
    assert future["od_mask_raw"][0, 1] == 0.0


# --------------------------------------------------------------------------- #
# 模块级函数入口
# --------------------------------------------------------------------------- #

def test_module_level_helpers_delegate():
    arrays = _arrays({0: [0, 5, 10]})
    lookup = _lookup(arrays)
    window = frames_build_history(lookup, 0, 10, stride=5, k=3, alignments=ALIGNMENTS)
    assert window["hist_valid"].tolist() == [1.0, 1.0, 1.0]
    future = frames_build_future(lookup, 0, 0, stride=5, k=2, alignments=ALIGNMENTS)
    assert future["valid"].tolist() == [1.0, 1.0]
    assert DEFAULT_STRIDE == 5 and DEFAULT_K == 6


# =========================================================================== #
# schema v2 回归（由独立 v2 测试文件合并而来，pipeline.stages 入口）：
# 过滤洞 / 头部 valid / wm_valid 门控 / od_id 身份匹配 / lane 本地等价。
# =========================================================================== #

class _ArraysWrapper(dict):
    """让 build_* 的 lane 缓存键稳定（同对象身份）。"""


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
