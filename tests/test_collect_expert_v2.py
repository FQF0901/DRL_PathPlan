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
    _concat_shards,
    _npz_schema_manifest,
    _parse_args,
    _samples_to_arrays,
    _shard_segments,
    _spec_entries,
    _summarize_records,
    apply_balance,
    arc_interpolate,
    balance_from_specs,
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
    world: bool = False,
):
    """直线 episode：poses[i] = (0.05*i, 0, 0)（每 env step 0.05 m，运动学 round-trip 精确）。

    frames 只在策略步边界（每 5 step）记录；``bad_at`` 之后的窗口不再可用。
    ``world=True`` 时逐帧带 schema v3 世界系键（``ego_world``/``route_world`` + mask）。
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
        if world:  # schema v3：A4 nav 逐步重建输入（世界系键 + 逐点 mask）
            obs["ego_world"] = np.zeros((1, 3), dtype=np.float32)
            obs["ego_world_mask"] = np.ones((1,), dtype=np.float32)
            obs["route_world"] = np.zeros((64, 2), dtype=np.float32)
            obs["route_world_mask"] = np.zeros((64,), dtype=np.float32)
            obs["route_world_mask"][:4] = 1.0
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


def test_balance_from_specs_matches_apply_balance():
    """分片路径的配平（摘要 + train_weight 数组）与 apply_balance 逐行同值/同统计。"""
    rng = np.random.default_rng(7)
    samples, entries = [], []
    start = 0
    for spec_index in range(8):
        difficulty, geometry = ("easy", "straight") if spec_index % 2 == 0 else ("hard", "curve")
        n_rows = 4 + spec_index % 3
        for step in range(n_rows):
            labels = np.zeros(len(SUPERVISED_LABELS), dtype=np.float32)
            if step == 0:
                labels[spec_index % len(SUPERVISED_LABELS)] = 1.0
            samples.append(
                {
                    "difficulty": difficulty,
                    "geometry": geometry,
                    "spec_id": spec_index,  # 唯一 → (spec_id, step) 序 = spec 序 + step 序
                    "step": step,
                    "labels": labels,
                    "train_weight": float(rng.random() > 0.3),
                }
            )
        entries.append(
            {
                "spec_index": spec_index,
                "start": start,
                "rows": n_rows,
                "trainable": int(sum(1 for row in samples[start:] if row["train_weight"] > 0.0)),
                "report": {"id": spec_index, "difficulty": difficulty, "geometry": geometry},
            }
        )
        start += n_rows
    train_weight = np.array([row["train_weight"] for row in samples], dtype=np.float32)
    for mode, ratio in (("weights", 3.0), ("cap", 1.0), ("none", 3.0)):
        reference = apply_balance(samples, mode=mode, ratio=ratio, seed=0)
        got = balance_from_specs(entries, train_weight, mode=mode, ratio=ratio)
        np.testing.assert_allclose(got["sample_weight"], reference["sample_weight"])
        np.testing.assert_array_equal(got["balance_group"], reference["balance_group"])
        reference_stats = dict(reference["stats"])
        if "dropped_indices" in reference_stats:
            reference_stats["dropped_indices_count"] = len(reference_stats.pop("dropped_indices"))
        assert got["stats"] == reference_stats


def test_balance_from_specs_zero_trainable_group_keeps_weight_one():
    """回归：整组 0 可训练行（如 easy/merge）时不得 KeyError；该组行权重保持 1（与 apply_balance 一致）。"""
    samples, entries = [], []
    start = 0
    plan = [("easy", "merge", 3, 0.0), ("easy", "straight", 2, 1.0), ("hard", "curve", 5, 1.0)]
    for spec_index, (difficulty, geometry, n_rows, tw) in enumerate(plan):
        for step in range(n_rows):
            samples.append(
                {
                    "difficulty": difficulty,
                    "geometry": geometry,
                    "spec_id": spec_index,
                    "step": step,
                    "labels": np.zeros(len(SUPERVISED_LABELS), dtype=np.float32),
                    "train_weight": tw,
                }
            )
        entries.append(
            {
                "spec_index": spec_index,
                "start": start,
                "rows": n_rows,
                "trainable": int(round(tw * n_rows)),
                "report": {"id": spec_index, "difficulty": difficulty, "geometry": geometry},
            }
        )
        start += n_rows
    train_weight = np.array([row["train_weight"] for row in samples], dtype=np.float32)
    reference = apply_balance(samples, mode="weights", ratio=3.0, seed=0)
    got = balance_from_specs(entries, train_weight, mode="weights", ratio=3.0)
    np.testing.assert_allclose(got["sample_weight"], reference["sample_weight"])
    np.testing.assert_array_equal(got["balance_group"], reference["balance_group"])
    # easy/merge（0 可训练行）→ 1.0；两个可训练组 2/5 行 → 7/(2*2)=1.75 / 7/(2*5)=0.7（不归一化 dead 组）
    assert got["sample_weight"][:3].tolist() == [1.0, 1.0, 1.0]
    assert got["sample_weight"][3] == pytest.approx(1.75)
    assert got["sample_weight"][5] == pytest.approx(0.7)
    assert "easy/merge" not in got["stats"]["group_trainable_before"]


def test_shard_roundtrip_merge_matches_direct_arrays(tmp_path):
    """分片乱序落盘 → 按 spec 序逐键拼接 == 直接对全量行建数组（键/值/dtype 一致）。"""
    rows_a, counter_a = _extract(_episode(30))
    rows_b = [dict(row, episode_id=1) for row in _extract(_episode(25))[0]]
    records = [
        {
            "spec_index": 1,
            "kept": rows_b,
            "filter_counts": {},
            "candidates": len(rows_b),
            "report": {"id": 200, "difficulty": "hard", "geometry": "curve"},
        },
        {
            "spec_index": 0,
            "kept": rows_a,
            "filter_counts": counter_a,
            "candidates": len(rows_a),
            "report": {"id": 100, "difficulty": "easy", "geometry": "straight"},
        },
    ]
    # 分片乱序到达：spec 1 的摘要先入列，行序仍须按 spec 下标
    summary_b = _summarize_records(1, 0, [records[0]], tmp_path)
    summary_a = _summarize_records(0, 0, [records[1]], tmp_path)
    entries, total_rows, missing = _spec_entries([summary_b, summary_a], 2)
    assert missing == []
    assert total_rows == len(rows_a) + len(rows_b)
    assert [entry["spec_index"] for entry in entries] == [0, 1]
    arrays = _concat_shards(tmp_path, _shard_segments(entries), total_rows)
    expected = _samples_to_arrays(rows_a + rows_b)
    assert set(arrays) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(arrays[key], expected[key])


def test_shard_writer_skips_empty_chunk(tmp_path):
    summary = _summarize_records(
        0, 0, [{"spec_index": 0, "kept": [], "candidates": 0, "filter_counts": {}, "report": {"id": 1, "termination": "error"}, "error": "Boom: x"}],
        tmp_path,
    )
    assert summary["shard"] is None
    assert summary["rows"] == 0
    assert summary["specs"][0]["error"] == "Boom: x"
    assert list(tmp_path.iterdir()) == []  # 无行不落分片


# --------------------------------------------------------------------------- #
# save_dataset / meta schema
# --------------------------------------------------------------------------- #

def test_save_dataset_v2_schema_roundtrip(tmp_path):
    episode = _episode(80, world=True)  # v3：带上世界系键走完整 save_dataset 路径
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
                "filter_reason", "ego_world", "ego_world_mask", "route_world", "route_world_mask"):
        assert key in arrays, f"npz 缺少 {key}"
    assert arrays["route_world"].shape == (len(rows), 64, 2)
    assert arrays["route_world_mask"].shape == (len(rows), 64)
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
    # v3（A4 nav 修正）：obs_fingerprint 前缀随 OBS_SCHEMA_VERSION 升为 v3-（数据集 schema_version
    # 是 collect_expert 自己的契约版本，保持 2）
    assert meta["obs_fingerprint"].startswith("v3-")
    assert meta["obs_schema_version"] == 3
    manifest = schema_manifest()
    for key, spec in manifest["frame"].items():
        assert meta["schema"]["frame"][key]["dtype"] == spec["dtype"]
    assert meta["schema"]["od_slot_policy"]["presence"].startswith("对象出盒")
    assert meta["dataset_schema"]["od_id"]["dtype"] == "int64"
    assert meta["dataset_schema"]["wm_valid"]["shape"][1] == 6
    assert meta["dataset_schema"]["route_world"]["shape"] == ["<N>", 64, 2]
    assert meta["dataset_schema"]["ego_world"]["shape"] == ["<N>", 1, 3]
    assert meta["channel_shapes"]["route_world"] == [64, 2]
    assert meta["channel_shapes"]["ego_world"] == [1, 3]
    assert "计数=行数" in meta["weight_semantics"]["effective_weight"] or "train_weight" in meta["weight_semantics"]["effective_weight"]


def test_v2_constants_and_cli_defaults():
    assert "others" in CURRENT_CHANNELS
    # v3（A4）：世界系键必须逐帧入库，否则 Stage A/B 缺 nav 重建输入
    assert {"ego_world", "route_world"} <= set(CURRENT_CHANNELS)
    assert set(MEM_HISTORY_KEYS) == {"od_id_hist", "od_presence_hist"}
    assert INT64_KEYS == frozenset({"od_id", "od_id_hist"})
    args = _parse_args(["--specs", "x.json", "--out", "runs/x"])
    assert args.balance == "weights"
    assert args.workers == 0  # 0 = auto（CPU 核数取半、上限 8）
    manifest = _npz_schema_manifest(num_slots=16, frames=6, others_dim=28, label_count=8)
    assert manifest["train_weight"]["semantics"].startswith("过滤门")
    assert manifest["frame_usable"]["shape"] == ["<N>"]
    # world 键的形状契约（非 OD/LD 槽位数）：ego_world (1,3) / route_world (64,2)
    assert manifest["ego_world"]["shape"] == ["<N>", 1, 3]
    assert manifest["ego_world_mask"]["shape"] == ["<N>", 1]
    assert manifest["route_world"]["shape"] == ["<N>", 64, 2]
    assert manifest["route_world_mask"]["shape"] == ["<N>", 64]


def test_world_channels_are_stored_with_masks():
    """v3：``_samples_to_arrays`` 逐帧存 ``ego_world``/``route_world`` + 逐点 mask。"""
    episode = _episode(80, world=True)
    rows, _ = _extract(episode)
    arrays = _samples_to_arrays(rows)
    assert arrays["ego_world"].shape == (len(rows), 1, 3)
    assert arrays["ego_world_mask"].shape == (len(rows), 1)
    assert arrays["route_world"].shape == (len(rows), 64, 2)
    assert arrays["route_world_mask"].shape == (len(rows), 64)
    assert float(arrays["ego_world_mask"].min()) == 1.0
    assert float(arrays["route_world_mask"].sum(axis=1).min()) == 4.0


def test_legacy_samples_without_world_channels_are_unchanged():
    """旧 episode（无世界系键）→ 不新增 npz 键，逐位兼容旧数据集契约。"""
    rows, _ = _extract(_episode(80))
    arrays = _samples_to_arrays(rows)
    assert "ego_world" not in arrays and "route_world" not in arrays
