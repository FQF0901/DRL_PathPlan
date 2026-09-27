"""权重感知会计回归：count vs weighted 双口径、权重 0 不出力、软目标温度。

对应交付项「权重大小写」：

- ``weighted_stats``：count/mean 与 weight/weighted_mean 两套口径同时给出；
- ``weighted_od_multi_step_loss``：逐槽位显式加权（``Σ w·mask·err / Σ w·mask``），
  权重 0 的帧不贡献分子/分母，且不做"有效项总数"隐性重加权；
- ``row_action_weights``：``train_weight × balance_weight``（缺失时的退化链）；
- ``presence_entry_loss``：权重 0 的 (帧,horizon) 完全剔除；AUC 单类 → nan；
- ``router_hard_label_loss``：硬标签 CE（F.cross_entropy 同式；权重 0 不出力；标签 <0 忽略）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pipeline.trainer import (
    BCDataset,
    binary_auc,
    dataset_weight_report,
    presence_entry_loss,
    row_action_weights,
    router_hard_label_loss,
    weighted_od_multi_step_loss,
    weighted_stats,
)


# ------------------------------------------------------------------ weighted_stats
def test_weighted_stats_count_vs_weighted_dual_caliber() -> None:
    stats = weighted_stats(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 1.0]), prefix="err_")
    assert stats["err_count"] == 3.0
    assert stats["err_weight"] == 2.0
    assert stats["err_mean"] == pytest.approx(2.0)  # 计数口径
    assert stats["err_weighted_mean"] == pytest.approx(2.5)  # 加权口径
    assert stats["err_median"] == pytest.approx(2.0)  # 零权重样本不影响分位数
    assert stats["err_p95"] == pytest.approx(3.0)
    # 全零权重：统计量为 nan，但 count/weight 仍保留（诊断）
    empty = weighted_stats(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
    assert empty["count"] == 2.0 and empty["weight"] == 0.0
    assert np.isnan(empty["weighted_mean"])


def test_weighted_quantile_ignores_zero_weight_samples() -> None:
    values = np.array([0.0, 10.0, 20.0, 30.0])
    weights = np.array([0.0, 1.0, 1.0, 1.0])
    stats = weighted_stats(values, weights)
    assert stats["median"] == pytest.approx(20.0)
    assert stats["p95"] == pytest.approx(30.0)


# ------------------------------------------------------------------ 行权重
def _tiny_dataset(arrays: dict) -> BCDataset:
    from pipeline.trainer import SUPERVISED_LABELS

    return BCDataset(arrays, {"label_names": list(SUPERVISED_LABELS)})


def test_row_action_weights_precedence_and_product() -> None:
    count = 4
    base = {"episode_id": np.zeros(count, dtype=np.int64)}
    dataset = _tiny_dataset({**base, "sample_weight": np.full(count, 0.5, dtype=np.float32)})
    assert row_action_weights(dataset, np.arange(count)).tolist() == [0.5] * count
    dataset = _tiny_dataset({
        **base,
        "sample_weight": np.full(count, 0.5, dtype=np.float32),
        "train_weight": np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32),
        "balance_weight": np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float32),
    })
    weights = row_action_weights(dataset, np.arange(count))
    assert weights.tolist() == [0.0, 2.0, 1.0, 0.0]  # train_weight × balance_weight（不含 sample_weight）


def test_dataset_weight_report_counts_and_weighted() -> None:
    count = 4
    dataset = _tiny_dataset({
        "episode_id": np.zeros(count, dtype=np.int64),
        "train_weight": np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "balance_weight": np.ones(count, dtype=np.float32),
    })
    report = dataset_weight_report(dataset, prefix="dataset")
    assert report["dataset/rows"] == 4.0
    assert report["dataset/rows_weighted"] == 3.0
    assert report["dataset/train_weight_zero"] == pytest.approx(0.25)


# ------------------------------------------------------------------ OD 多步加权损失
def _od_case(slot_counts, errors, frame_weights):
    """每帧 slot 数/误差/权重可配；构造 pred/target 使每槽 4 维误差 = errors[f]。"""
    horizon = 1
    slots = 16
    frames = len(slot_counts)
    od_pred = torch.zeros((frames, horizon, slots, 5))
    od_target = torch.zeros((frames, horizon, slots, 5))
    od_mask = torch.zeros((frames, horizon, slots))
    for frame, (count, err) in enumerate(zip(slot_counts, errors)):
        od_mask[frame, 0, :count] = 1.0
        od_target[frame, 0, :count, :4] = err  # 4 个线性维都有误差 → smooth_l1=0.5·err²（err<=1）
    return od_pred, od_target, od_mask, torch.tensor(frame_weights, dtype=torch.float32)


def test_weighted_od_loss_does_not_implicitly_reweight_by_valid_count() -> None:
    """帧 A：1 槽误差 1.0；帧 B：16 槽误差 0。等权 → 显式加权 = 0.5/17（不是逐帧均值的 0.25）。"""
    od_pred, od_target, od_mask, weights = _od_case([1, 16], [1.0, 0.0], [1.0, 1.0])
    loss, per_h = weighted_od_multi_step_loss(od_pred, od_target, od_mask, frame_weight=weights)
    # smooth_l1(beta=1) 在 err=1.0 处 = 0.5（4 维合计/4 = 0.5）→ 1 槽贡献 0.5，16 槽 0 → 0.5/17
    assert float(loss) == pytest.approx(0.5 / 17.0, rel=1e-4), "必须逐槽位显式加权而非逐帧均值"
    assert per_h[0]["count"] == 17.0


def test_weighted_od_loss_zero_weight_frames_are_excluded() -> None:
    base = dict(slot_counts=[4, 4], frame_weights=[0.0, 1.0])
    od_pred, od_target, od_mask, weights = _od_case([4, 4], [1.0, 3.0], [0.0, 1.0])
    loss_zero, _ = weighted_od_multi_step_loss(od_pred, od_target, od_mask, frame_weight=weights)
    # 只保留第二帧 → 与单独计算第二帧一致
    od_pred_single, od_target_single, od_mask_single, weights_single = _od_case([4], [3.0], [1.0])
    loss_single, _ = weighted_od_multi_step_loss(
        od_pred_single, od_target_single, od_mask_single, frame_weight=weights_single
    )
    assert float(loss_zero) == pytest.approx(float(loss_single), rel=1e-5)
    # valid 门控同样清零（wm_valid=0 的 horizon 不出力）
    od_pred2, od_target2, od_mask2, weights2 = _od_case([4], [3.0], [1.0])
    valid = torch.zeros((1, 1))
    loss_masked, per_h = weighted_od_multi_step_loss(
        od_pred2, od_target2, od_mask2, frame_weight=weights2, valid=valid
    )
    assert float(loss_masked) == 0.0
    assert per_h[0]["count"] == 0.0 and per_h[0]["weight"] < 1e-6


# ------------------------------------------------------------------ presence/entry
def test_presence_entry_loss_weights_and_auc() -> None:
    logits_presence = torch.tensor([[5.0, -5.0, 5.0, -5.0]])
    logits_entry = torch.tensor([[-5.0, 5.0, -5.0, 5.0]])
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    terms = presence_entry_loss(logits_presence, logits_entry, target, 1.0 - target)
    assert terms["presence_auc"] == 1.0
    assert terms["entry_auc"] == 1.0
    assert float(terms["presence"]) < 0.1
    # 权重 0 的帧（第 2 行）不贡献损失
    zeros = presence_entry_loss(
        torch.cat([logits_presence, torch.full((1, 4), 100.0)], dim=0),
        torch.cat([logits_entry, torch.full((1, 4), 100.0)], dim=0),
        torch.cat([target, target], dim=0),
        torch.cat([1.0 - target, 1.0 - target], dim=0),
        frame_weight=torch.tensor([1.0, 0.0]),
    )
    assert float(zeros["presence"]) == pytest.approx(float(terms["presence"]), rel=1e-5)

    single_class = presence_entry_loss(logits_presence[:, :2], logits_entry[:, :2],
                                       torch.ones(1, 2), torch.zeros(1, 2))
    assert np.isnan(single_class["presence_auc"]) and np.isnan(single_class["entry_auc"])


def test_binary_auc_rank_based() -> None:
    assert binary_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert binary_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert np.isnan(binary_auc([0.5, 0.6], [1, 1]))


# ------------------------------------------------------------------ router 硬标签 CE（lane B B3）
def test_router_hard_label_loss_weighting_and_missing_labels() -> None:
    from pipeline.trainer import router_hard_label_loss, router_hard_label_stats

    logits = torch.tensor([[2.0, -1.0, 0.5, 0.0, 0.1, -0.2, 0.3, 0.4],
                           [0.0, 1.0, -1.0, 0.5, 0.2, 0.1, -0.3, 0.4]])
    labels = torch.tensor([0, 2])
    reference = torch.nn.functional.cross_entropy(logits, labels)
    assert float(router_hard_label_loss(logits, labels)) == pytest.approx(float(reference), rel=1e-6)
    # 权重 0 的样本不出力：weight=[0,1] 时等于只对第 2 行求 CE
    weighted = router_hard_label_loss(logits, labels, sample_weight=torch.tensor([0.0, 2.0]))
    single = torch.nn.functional.cross_entropy(logits[1:], labels[1:])
    assert float(weighted) == pytest.approx(float(single), rel=1e-6)
    # 标签缺失（<0）被忽略；全缺失 → loss=0、stats 为 nan/0（placeholder 语义由调用方记）
    missing = router_hard_label_loss(logits, torch.tensor([-1, 2]))
    assert float(missing) == pytest.approx(float(single), rel=1e-6)
    assert float(router_hard_label_loss(logits, torch.tensor([-1, -1]))) == 0.0
    stats = router_hard_label_stats(logits, labels)
    assert stats["ce"] == pytest.approx(float(reference), rel=1e-6)
    assert stats["acc"] == pytest.approx(0.5)  # 第 1 行 argmax=0=标签；第 2 行 argmax=1 ≠ 标签 2
    assert stats["count"] == 2.0
    assert np.isnan(router_hard_label_stats(logits, torch.tensor([-1, -1]))["acc"])
