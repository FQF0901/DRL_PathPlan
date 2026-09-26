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
    build_future,
    build_history,
    lookup_from_arrays,
)

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
    window = build_history(lookup, 0, 10, stride=5, k=3, alignments=ALIGNMENTS)
    assert window["hist_valid"].tolist() == [1.0, 1.0, 1.0]
    future = build_future(lookup, 0, 0, stride=5, k=2, alignments=ALIGNMENTS)
    assert future["valid"].tolist() == [1.0, 1.0]
    assert DEFAULT_STRIDE == 5 and DEFAULT_K == 6
