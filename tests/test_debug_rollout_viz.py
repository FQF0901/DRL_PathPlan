"""回归：``tools/debug_rollout_viz.py`` 的纯几何/偏差函数（不依赖 env/torch/matplotlib）。

lane P3-L 的回灌可视化把"GT 未来（各帧自车系）"与"模型推演（t0 系）"统一到 t0 自车系；
本测试锁定坐标变换、OD/LD 投影与 ADE/FDE 口径，防止渲染/provenance 静默错系。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tools.debug_rollout_viz import (
    box_corners,
    gt_future,
    gt_ld_at,
    gt_od_boxes,
    ld_segments,
    plan_deviation,
    se2_apply,
    wm_ld_at,
    wm_od_boxes,
    world_to_frame,
    wrap_to_pi,
)


def test_se2_apply_and_world_to_frame_roundtrip() -> None:
    rng = np.random.default_rng(7)
    pose = np.array([3.0, -2.0, 0.7])
    pts = rng.normal(size=(9, 2)) * 5.0
    world = se2_apply(pose, pts)
    back = world_to_frame(pose, world)
    assert np.allclose(back, pts, atol=1e-9)


def test_gt_future_is_in_t0_frame() -> None:
    # t0 在原点朝 +x：未来沿 +x 直行 → t0 系点 = 位移本身
    poses = [np.array([0.0, 0.0, 0.0]) for _ in range(40)]
    for k in range(40):
        poses[k][0] = 0.5 * k  # 每 env step 0.5 m
    gt_xy, valid = gt_future(poses, 0, interval=5, horizon=6)
    assert valid.all()
    assert np.allclose(gt_xy[:, 0], [2.5, 5.0, 7.5, 10.0, 12.5, 15.0])
    assert np.allclose(gt_xy[:, 1], 0.0)

    # t0 朝 +y（θ=π/2）、未来沿世界 +y：t0 系里仍是 +x 前进
    poses2 = [np.array([0.0, 0.0, math.pi / 2]) for _ in range(40)]
    for k in range(40):
        poses2[k][1] = 0.5 * k
    gt2, valid2 = gt_future(poses2, 0, interval=5, horizon=6)
    assert valid2.all()
    assert np.allclose(gt2[:, 0], [2.5, 5.0, 7.5, 10.0, 12.5, 15.0], atol=1e-9)
    assert np.allclose(gt2[:, 1], 0.0, atol=1e-9)


def test_gt_future_truncates_when_episode_ends() -> None:
    poses = [np.array([float(k), 0.0, 0.0]) for k in range(7)]  # 只有 7 步
    gt_xy, valid = gt_future(poses, 0, interval=5, horizon=6)
    assert int(valid.sum()) == 1  # k=5 存在；k=10.. 不存在
    assert np.isnan(gt_xy[1:, 0]).all()


def test_gt_od_boxes_transform_into_t0_frame() -> None:
    # 未来帧 ego 在 (10,0,0)，对象在其自车系 (5,2)，t0 在原点 θ=0 → t0 系 (15,2)
    records = [
        {
            "od": np.zeros((16, 9), dtype=np.float64),
            "od_presence": np.zeros(16, dtype=np.float64),
        }
        for _ in range(6)
    ]
    records[5]["od"][0] = [5.0, 2.0, 0.0, 0.0, 1.0, 0.0, 4.5, 1.8, 2.0]
    records[5]["od_presence"][0] = 1.0
    poses = [np.array([0.0, 0.0, 0.0])] * 6
    poses[5] = np.array([10.0, 0.0, 0.0])
    boxes = gt_od_boxes(records, poses, start=0, interval=5, horizon=1)
    assert len(boxes) == 1
    assert boxes[0]["x"] == pytest.approx(15.0)
    assert boxes[0]["y"] == pytest.approx(2.0)
    assert boxes[0]["heading"] == pytest.approx(0.0)
    assert boxes[0]["L"] == pytest.approx(4.5)

    # presence=0（陈旧槽）不画
    records[5]["od_presence"][0] = 0.0
    assert gt_od_boxes(records, poses, start=0, interval=5, horizon=1) == []
    # 帧不存在（episode 提前结束）不画
    assert gt_od_boxes(records, poses, start=0, interval=5, horizon=6) == []


def test_gt_ld_at_rotates_positions_and_headings() -> None:
    records = []
    for _ in range(6):
        rec = {"ld": np.zeros((16, 7)), "ld_mask": np.zeros(16)}
        records.append(rec)
    # 未来帧 ego 在原点 θ=π/2；LD 采样点在自车系 (3,0)、航向 0（=世界 π/2）
    records[5]["ld"][0] = [3.0, 0.0, 0.0, 0.01, 12.0, 2.0, 2.0]
    records[5]["ld_mask"][0] = 1.0
    poses = [np.array([0.0, 0.0, math.pi / 2])] * 6
    rows, valid = gt_ld_at(records, poses, start=0, interval=5, horizon=1)
    assert valid[0]
    assert rows[0, 0] == pytest.approx(3.0, abs=1e-9)  # 同朝向 → 自车系坐标原样过继
    assert rows[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert rows[0, 2] == pytest.approx(0.0, abs=1e-9)  # 与 ego 航向同向 → t0 系相对角 0
    assert not valid[1:].any()


def test_plan_deviation_ade_fde_only_on_valid_points() -> None:
    plan = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    gt = np.array([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    valid = np.array([True, True, False])
    ade, fde, dists, gt_dists = plan_deviation(plan, gt, valid)
    assert ade == pytest.approx(0.5)  # |1-0|, |2-2|
    assert fde == pytest.approx(0.0)
    assert dists == pytest.approx([1.0, 0.0])
    # 全部无效 → nan
    nan_ade, nan_fde, empty, _ = plan_deviation(plan, gt, np.zeros(3, dtype=bool))
    assert math.isnan(nan_ade) and math.isnan(nan_fde) and empty == []


def test_wm_od_boxes_gate_and_size_carry() -> None:
    od_pred = np.zeros((16, 5))
    od_pred[0] = [10.0, 1.0, 0.0, 0.0, 0.3]
    od_pred[1] = [20.0, 0.0, 0.0, 0.0, 0.0]
    od_pred[2] = [30.0, 0.0, 0.0, 0.0, 0.0]
    presence = np.full(16, -10.0)
    presence[0] = 10.0  # sigmoid > 0.5：高置信
    presence[1] = 10.0  # 高置信但无静态尺寸可继承 → 跳过
    od_now = np.zeros((16, 9))
    od_now[0, 6:8] = [4.5, 1.8]
    od_now[2, 6:8] = [4.5, 1.8]
    mask = np.zeros(16)
    mask[2] = 1.0  # t0 槽位存在（身份可继承）→ 低置信也画（虚线淡显）
    boxes = wm_od_boxes(od_pred, presence, od_now, mask)
    assert [int(box["slot"]) for box in boxes] == [0, 2]
    assert boxes[0]["presence"] > 0.5
    assert boxes[1]["presence"] < 0.5
    assert boxes[0]["L"] == pytest.approx(4.5)
    # 既无 mask 也无置信、或无尺寸 → 不画
    assert wm_od_boxes(od_pred, presence * 0 - 10.0, np.zeros((16, 9)), np.zeros(16)) == []


def test_wm_ld_at_gates_by_slot_mask() -> None:
    pred = np.zeros((16, 4))
    pred[0] = [5.0, -1.0, 0.2, 0.0]
    pred[1] = [9.0, 9.0, 0.0, 0.0]
    mask = np.zeros(16)
    mask[0] = 1.0
    rows, valid = wm_ld_at(pred, mask)
    assert valid[0] and not valid[1:].any()
    assert rows[0, 0] == pytest.approx(5.0)
    assert rows[0, 3] == pytest.approx(0.0)


def test_box_corners_and_ld_segments() -> None:
    corners = box_corners(0.0, 0.0, 4.0, 2.0, 0.0)
    assert corners.shape == (4, 2)
    assert np.allclose(corners, [[-2, -1], [2, -1], [2, 1], [-2, 1]])
    # 旋转 90°：长轴指向 +y（y 范围 = ±L/2）
    rotated = box_corners(0.0, 0.0, 4.0, 2.0, math.pi / 2)
    assert np.allclose(np.sort(rotated[:, 1]), [-2, -2, 2, 2])

    rows = np.zeros((2, 7))
    rows[0] = [1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    mask = np.array([1.0, 0.0])
    segs = ld_segments(rows, mask, seg_len=2.0)
    assert len(segs) == 1
    x0, y0, x1, y1 = segs[0]
    assert (x0, y0, x1, y1) == pytest.approx((0.0, 2.0, 2.0, 2.0))


def test_wrap_to_pi() -> None:
    assert wrap_to_pi(3 * math.pi) == pytest.approx(math.pi)
    assert wrap_to_pi(-3 * math.pi) == pytest.approx(math.pi)
    assert wrap_to_pi(0.1) == pytest.approx(0.1)
