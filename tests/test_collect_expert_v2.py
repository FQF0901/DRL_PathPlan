"""``tools/collect_expert.py`` v2 契约测试（不建 env / 不跑 worker）。

覆盖：
- 全部 policy 帧入库 + 过滤命中 → ``train_weight=0``（不删行）；
- ``wm_valid[6]`` 语义（目标帧存在且仍可用）；
- 权重感知统计（计数=行数 / 加权=权重和）与配平；
- ``save_dataset`` 的 v2 schema（od_id int64 / wm_valid / meta schema 清单 / per_frame_v2）。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from env.obs.schema import schema_manifest
from tools.collect_expert import (
    CURRENT_CHANNELS,
    INT64_KEYS,
    MEM_HISTORY_KEYS,
    SUPERVISED_LABELS,
    WINDOW_POLICIES,
    _npz_schema_manifest,
    _parse_args,
    apply_balance,
    arc_interpolate,
    extract_samples,
    label_statistics,
    save_dataset,
)
from env.scenario.labels import LABEL_ORDER


# --------------------------------------------------------------------------- #
# 合成 episode
# --------------------------------------------------------------------------- #

class _Spec:
    id = 3
    seed = 1003
    split = "train"
    difficulty = "easy"
    labels = {"geometry": "straight"}


def _episode(
    n_steps: int = 80,
    *,
    bad_at: int | None = None,
    off_lane: tuple[int, ...] = (),
    cut_unverified: tuple[int, ...] = (),
    ended_by_env: bool = False,
):
    """直线 episode：poses[i] = (0.05*i, 0, 0)（每 env step 0.05 m，运动学 round-trip 精确）。

    frames 只在策略步边界（每 5 step）记录；``bad_at`` 之后的窗口不再可用。
    """
    poses = [np.array([0.05 * i, 0.0, 0.0], dtype=np.float64) for i in range(n_steps + 1)]
    flags = [None] + [
        {"bad": False, "crash": False, "out_of_road": False, "arrive": False, "max_step": False}
        for _ in range(n_steps)
    ]
    if bad_at is not None:
        flags[bad_at] = {"bad": True, "crash": False, "out_of_road": True, "arrive": False, "max_step": False}
    frames = {}
    for t in range(0, n_steps, 5):
        obs = {
            "ego": np.zeros((1, 8), dtype=np.float32),
            "od": np.zeros((16, 9), dtype=np.float32),
            "od_mask": np.zeros((16, ), dtype=np.float32),
            "od_id": np.full((16, ), -1, dtype=np.int64),
            "od_presence": np.zeros((16, ), dtype=np.float32),
            "ld": np.zeros((16, 7), dtype=np.float32),
            "ld_mask": np.zeros((16, ), dtype=np.float32),
        }
        obs["ego"][0, 0] = 0.5
        labels_raw = np.zeros(len(LABEL_ORDER), dtype=np.float32)
        if t in cut_unverified:
            labels_raw[LABEL_ORDER.index("cutin_active")] = 1.0
        frames[t] = {
            "obs": obs,
            "hist_valid": np.ones(6, dtype=np.float32),
            "od_id_hist": np.full((6, 16), -1, dtype=np.int64),
            "od_presence_hist": np.zeros((6, 16), dtype=np.float32),
            "labels_raw": labels_raw,
            "events": [],
            "pose": poses[t].copy(),
            "on_lane": t not in off_lane,
            "lane_lat": 0.0,
            "lane_width": 3.5,
        }
    return {
        "frames": frames,
        "poses": poses,
        "flags": flags,
        "termination": "out_of_road" if bad_at is not None else "max_step",
        "steps": n_steps,
        "ended_by_env": ended_by_env,
    }


def _extract(episode, interpolate_fn=arc_interpolate, require_dense=False):
    from collections import Counter

    counter = Counter()
    rows = extract_samples(
        episode,
        _Spec(),
        episode_id=0,
        label_order=SUPERVISED_LABELS,
        interpolate_fn=interpolate_fn,
        on_lane_frac=0.5,
        on_lane_margin=0.3,
        roundtrip_key_mean=0.25,
        roundtrip_key_max=0.5,
        filter_counter=counter,
        require_dense=require_dense,
    )
    return rows, counter


# --------------------------------------------------------------------------- #
# 存全部帧 + train_weight
# --------------------------------------------------------------------------- #

def test_all_policy_frames_are_stored_and_filtered_rows_get_zero_weight():
    episode = _episode(80, bad_at=60, off_lane=(10,), cut_unverified=(15,))
    rows, counter = _extract(episode)
    # v2：行数 = 候选 policy 帧数（不再逐帧删除）
    assert len(rows) == len(episode["frames"]) == 16
    assert [row["step"] for row in rows] == sorted(row["step"] for row in rows)

    by_step = {row["step"]: row for row in rows}
    # limit = 59 -> t+30 > 59 的行（t>=30）全部 terminal_window=0
    assert by_step[30]["train_weight"] == 0.0
    assert by_step[75]["train_weight"] == 0.0
    # 命中优先级：terminal_window > not_on_lane > cut > roundtrip
    assert by_step[10]["filter_reason"] == "not_on_lane"
    assert by_step[15]["filter_reason"] == "cut_label_unverified"
    # 干净行 train_weight=1
    assert by_step[5]["train_weight"] == 1.0
    assert by_step[25]["train_weight"] == 1.0

    assert counter["terminal_window"] == 10  # t=30..75
    assert counter["not_on_lane"] == 1
    assert counter["cut_label_unverified"] == 1
    assert "roundtrip_fail" not in counter
    # 过滤计数 = 零权重行数（不是总行数；干净行不计数）
    assert sum(counter.values()) == sum(1 for row in rows if row["train_weight"] == 0.0) == 12


def test_wm_valid_target_exists_and_usable():
    episode = _episode(80, bad_at=45, off_lane=(20,))
    rows, _ = _extract(episode)
    by_step = {row["step"]: row for row in rows}
    # t=0：目标 5..30 都 <= limit=44；t=20 目标帧离道 -> k=4 掩码 0
    assert by_step[0]["wm_valid"].tolist() == [1, 1, 1, 0, 1, 1]
    # t=25：自身终末窗口（weight=0），目标 30..55 -> 45 起越限
    assert by_step[25]["train_weight"] == 0.0
    assert by_step[25]["wm_valid"].tolist() == [1, 1, 1, 0, 0, 0]
    # t=35：目标 40..65，仅 40 可用
    assert by_step[35]["wm_valid"].tolist() == [1, 0, 0, 0, 0, 0]
    # frame_usable：t=20 自己离道 -> 0
    assert by_step[20]["frame_usable"] == 0.0
    assert by_step[15]["frame_usable"] == 1.0


def test_roundtrip_failure_sets_weight_zero_without_dropping_rows():
    episode = _episode(80)

    def broken_interpolate(seq, **kwargs):
        return np.zeros((30, 3), dtype=np.float64)

    rows, counter = _extract(episode, interpolate_fn=broken_interpolate)
    assert len(rows) == 16  # 仍全部入库
    # 可计算窗口的行（t+30 <= last_state=80 -> t<=50）roundtrip_fail；其余 terminal_window
    assert counter["roundtrip_fail"] == 11  # t=0..50 共 11 行
    trainable = [row for row in rows if row["train_weight"] > 0.0]
    assert trainable == []
    # 不可计算窗口的行：错误字段 NaN、动作/轨迹零填充
    last = rows[-1]
    assert np.isnan(last["roundtrip_key_err"])
    np.testing.assert_allclose(last["action"], 0.0)


def test_rows_carry_companions_and_mem_history():
    episode = _episode(30)
    rows, _ = _extract(episode)
    for row in rows:
        assert row["obs"]["od_id"].dtype == np.int64
        assert row["od_id_hist"].dtype == np.int64
        assert row["od_presence_hist"].shape == (6, 16)
        assert row["hist_valid"].shape == (6, )
        assert row["wm_valid"].shape == (WINDOW_POLICIES, )


# --------------------------------------------------------------------------- #
# 权重感知统计 / 配平
# --------------------------------------------------------------------------- #

def _fake_sample(group, label_index, train_weight=1.0):
    labels = np.zeros(len(SUPERVISED_LABELS), dtype=np.float32)
    if label_index is not None:
        labels[label_index] = 1.0
    return {
        "difficulty": group[0],
        "geometry": group[1],
        "spec_id": 0,
        "step": 0,
        "labels": labels,
        "train_weight": float(train_weight),
    }


def test_label_statistics_count_and_weighted_conventions():
    samples = [
        _fake_sample(("easy", "straight"), 0, 1.0),
        _fake_sample(("easy", "straight"), 0, 1.0),
        _fake_sample(("hard", "curve"), 2, 0.0),  # 过滤行：两套口径都不计
        _fake_sample(("hard", "curve"), 2, 1.0),
    ]
    for sample in samples:
        sample["sample_weight"] = 2.0 if sample["difficulty"] == "hard" else 0.5
    counts, weighted = label_statistics(samples, SUPERVISED_LABELS)
    assert counts["cutin_active"] == 2  # 行数口径
    assert weighted["cutin_active"] == pytest.approx(2 * 1.0 * 0.5)  # 权重和口径
    assert counts["crowded"] == 1  # train_weight=0 的行不计
    assert weighted["crowded"] == pytest.approx(2.0)


def test_balance_weights_use_trainable_rows_only():
    samples = [
        _fake_sample(("easy", "straight"), 0),
        _fake_sample(("easy", "straight"), 0),
        _fake_sample(("easy", "straight"), 0),
        _fake_sample(("easy", "straight"), 0, train_weight=0.0),
        _fake_sample(("hard", "curve"), 1),
    ]
    balance = apply_balance(samples, mode="weights", ratio=3.0, seed=0)
    weights = balance["sample_weight"]
    # 可训练总数 4、两个非空组：A(3) -> 4/(2*3)，B(1) -> 4/(2*1)
    assert weights[0] == pytest.approx(4.0 / 6.0)
    assert weights[4] == pytest.approx(2.0)
    assert weights[3] == 1.0  # 过滤行权重保持 1（下游先乘 train_weight）
    stats = balance["stats"]
    assert stats["counts"]["rows"] == 5 and stats["counts"]["trainable_rows"] == 4
    assert stats["group_counts_after"] == {"easy/straight": 3, "hard/curve": 1}


def test_balance_cap_zeroes_weights_without_deleting_rows():
    samples = [_fake_sample(("easy", "straight"), 0) for _ in range(4)]
    samples += [_fake_sample(("hard", "curve"), 1) for _ in range(2)]
    balance = apply_balance(samples, mode="cap", ratio=1.0, seed=0)
    # min_group=2 -> cap=2；easy 组 4 行中 2 行被置 0，行数不变
    assert len(samples) == 6
    weights = balance["sample_weight"]
    assert int((weights == 0.0).sum()) == 2
    assert balance["stats"]["cap"] == 2
    assert balance["stats"]["counts"]["trainable_after_cap"] == 4


# --------------------------------------------------------------------------- #
# save_dataset / meta schema
# --------------------------------------------------------------------------- #

def test_save_dataset_v2_schema_roundtrip(tmp_path):
    episode = _episode(80)
    rows, _ = _extract(episode)
    balance = apply_balance(rows, mode="weights", ratio=3.0, seed=0)
    from env.obs.builder import ObservationBuilder

    builder = ObservationBuilder({})  # 仅用于 _alignment_meta（不建 env）
    paths = save_dataset(
        tmp_path,
        rows,
        label_order=SUPERVISED_LABELS,
        builder=builder,
        config={"limit": None},
        report={"kinematics_source": "test"},
        sample_weight=balance["sample_weight"],
        balance_group=balance["balance_group"],
    )
    with np.load(paths["npz"]) as payload:
        arrays = {key: payload[key] for key in payload.files}
    for key in ("od_id", "od_id_hist", "od_presence", "od_presence_hist", "wm_valid", "train_weight",
                "frame_usable", "hist_valid", "episode_id", "step", "pose", "action", "balance_weight",
                "filter_reason"):
        assert key in arrays, f"npz 缺少 {key}"
    np.testing.assert_allclose(arrays["balance_weight"], arrays["sample_weight"])
    assert set(np.unique(arrays["filter_reason"]).tolist()) <= {
        "", "terminal_window", "not_on_lane", "cut_label_unverified", "roundtrip_fail", "roundtrip_dense_fail",
    }
    assert arrays["od_id"].dtype == np.int64
    assert arrays["od_id_hist"].dtype == np.int64
    assert arrays["od_presence"].dtype == np.float32
    assert arrays["wm_valid"].shape == (len(rows), 6)
    assert arrays["od_id_hist"].shape == (len(rows), 6, 16)
    assert arrays["episode_id"].tolist() == [0] * len(rows)

    meta = json.loads(open(paths["meta"], encoding="utf-8").read())
    assert meta["schema_version"] == 2
    assert meta["history_storage"] == "per_frame_v2"
    assert meta["obs_fingerprint"].startswith("v2-")
    manifest = schema_manifest()
    for key, spec in manifest["frame"].items():
        assert meta["schema"]["frame"][key]["dtype"] == spec["dtype"]
    assert meta["schema"]["od_slot_policy"]["presence"].startswith("对象出盒")
    assert meta["dataset_schema"]["od_id"]["dtype"] == "int64"
    assert meta["dataset_schema"]["wm_valid"]["shape"][1] == 6
    assert "计数=行数" in meta["weight_semantics"]["effective_weight"] or "train_weight" in meta["weight_semantics"]["effective_weight"]


def test_v2_constants_and_cli_defaults():
    assert "others" in CURRENT_CHANNELS
    assert set(MEM_HISTORY_KEYS) == {"od_id_hist", "od_presence_hist"}
    assert INT64_KEYS == frozenset({"od_id", "od_id_hist"})
    args = _parse_args(["--specs", "x.json", "--out", "runs/x"])
    assert args.balance == "weights"
    assert args.workers == 0  # 0 = auto（CPU 核数取半、上限 8）
    manifest = _npz_schema_manifest(num_slots=16, frames=6, others_dim=28, label_count=8)
    assert manifest["train_weight"]["semantics"].startswith("过滤门")
    assert manifest["frame_usable"]["shape"] == ["<N>"]
