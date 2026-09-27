"""``collect_expert --workers`` 的纯逻辑测试（不建 env / 不起 worker 进程）。

并行采集的正确性依赖两条与 worker 数无关的性质：
1. spec 轮转分配（``index % N``）确定、互斥、覆盖全部；
2. 父进程只收**逐 spec 标量摘要**，按全局 spec 下标展开 → 样本顺序 / ``episode_id`` /
   过滤统计与单进程逐条累加一致（``episode_id`` 就是全局下标），逐帧数据留在分片里。
"""

from __future__ import annotations

import os

from tools.collect_expert import (
    MAX_RECOMMENDED_WORKERS,
    RECYCLE_EVERY_SPECS,
    _chunk_tasks,
    _merge_filter_counts,
    _parse_args,
    _resolve_workers,
    _spec_entries,
    _worker_spec_indices,
)


def test_worker_spec_indices_partition_and_determinism() -> None:
    for total, workers in ((1, 1), (10, 3), (7, 7), (5, 8)):
        groups = [_worker_spec_indices(total, index, workers) for index in range(workers)]
        assert groups == [_worker_spec_indices(total, index, workers) for index in range(workers)]
        flat = sorted(index for group in groups for index in group)
        assert flat == list(range(total)), (total, workers, groups)


def test_chunk_tasks_sizes_and_coverage() -> None:
    tasks = [(index, None) for index in range(10)]
    chunks = _chunk_tasks(tasks, 4)
    assert [len(chunk) for chunk in chunks] == [4, 4, 2]
    assert [index for chunk in chunks for index, _ in chunk] == list(range(10))
    assert _chunk_tasks(tasks, 0) == [tasks]  # 0 = 不回收
    assert _chunk_tasks([], 4) == []


def _summary(worker_index, chunk_id, specs, shard=None):
    """构造一个块摘要；``specs = [(spec_index, rows, candidates, filter_counts), ...]``。"""
    return {
        "worker_index": worker_index,
        "chunk_id": chunk_id,
        "shard": shard,
        "rows": sum(rows for _, rows, _, _ in specs),
        "specs": [
            {
                "spec_index": index,
                "rows": rows,
                "candidates": candidates,
                "trainable": rows,
                "filter_counts": dict(counts),
                "report": {
                    "id": index,
                    "termination": "max_step",
                    "difficulty": "easy",
                    "geometry": "straight",
                },
                "error": None,
            }
            for index, rows, candidates, counts in specs
        ],
    }


def test_spec_entries_follow_global_spec_order() -> None:
    # 乱序到达（w0 的 spec 2 先到）
    summaries = [
        _summary(0, 0, [(2, 2, 10, {"b": 1})]),
        _summary(1, 0, [(0, 1, 5, {"a": 2})]),
        _summary(2, 0, [(1, 3, 7, {"a": 1, "c": 3})]),
    ]
    entries, total_rows, missing = _spec_entries(summaries, 3)
    assert [entry["spec_index"] for entry in entries] == [0, 1, 2]
    assert [entry["rows"] for entry in entries] == [1, 3, 2]
    assert [entry["start"] for entry in entries] == [0, 1, 4]
    assert total_rows == 6
    assert missing == []
    # 过滤计数合并 = 单进程顺序累加；键顺序也按 spec 原始顺序（a → c → b）
    counter = _merge_filter_counts(entries)
    assert dict(counter) == {"a": 3, "b": 1, "c": 3}
    assert list(counter) == ["a", "c", "b"]


def test_spec_entries_reports_missing_specs() -> None:
    summaries = [_summary(0, 0, [(0, 1, 1, {})]), _summary(1, 0, [(2, 1, 1, {})])]
    _, _, missing = _spec_entries(summaries, 3)
    assert missing == [1]


def test_spec_entries_shard_offsets_accumulate_per_shard() -> None:
    summaries = [
        _summary(0, 0, [(0, 2, 2, {}), (1, 1, 1, {})], shard="shard_w0_c0.npz"),
        _summary(0, 1, [(2, 3, 3, {})], shard="shard_w0_c1.npz"),
    ]
    entries, total_rows, _ = _spec_entries(summaries, 3)
    assert [(entry["shard"], entry["shard_offset"]) for entry in entries] == [
        ("shard_w0_c0.npz", 0),
        ("shard_w0_c0.npz", 2),
        ("shard_w0_c1.npz", 0),
    ]
    assert total_rows == 6


def test_workers_cli_default_is_auto() -> None:
    """默认 0=auto（CPU 取半 ∧ 可用内存折算，上限 ``MAX_RECOMMENDED_WORKERS``）；显式值原样；1 仍走单进程路径。"""
    args = _parse_args(["--specs", "env/specs/x.json", "--out", "runs/x"])
    assert int(args.workers) == 0
    assert int(args.recycle_every) == RECYCLE_EVERY_SPECS
    auto = _resolve_workers(0)
    assert 1 <= auto <= MAX_RECOMMENDED_WORKERS
    assert auto <= max(1, (os.cpu_count() or 4) // 2)  # CPU 上限
    assert _resolve_workers(-1) == auto  # 负值等同 auto
    assert _resolve_workers(1) == 1
    assert _resolve_workers(4) == 4
    assert int(_parse_args(["--specs", "x", "--out", "y", "--workers", "4"]).workers) == 4
    assert int(_parse_args(["--specs", "x", "--out", "y", "--workers", "1"]).workers) == 1
    assert int(_parse_args(["--specs", "x", "--out", "y", "--recycle-every", "0"]).recycle_every) == 0
    assert _parse_args(["--specs", "x", "--out", "y"]).keep_shards is False
    assert _parse_args(["--specs", "x", "--out", "y", "--keep-shards"]).keep_shards is True
