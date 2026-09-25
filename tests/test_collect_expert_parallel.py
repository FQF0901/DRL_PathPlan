"""``collect_expert --workers`` 的纯逻辑测试（不建 env / 不起 worker 进程）。

并行采集的正确性依赖两条与 worker 数无关的性质：
1. spec 轮转分配（``index % N``）确定、互斥、覆盖全部；
2. 父进程按**全局 spec 下标**合并记录 → 样本顺序 / ``episode_id`` / 过滤统计与
   单进程逐条累加一致（``episode_id`` 就是全局下标）。
"""

from __future__ import annotations

from collections import Counter

from tools.collect_expert import (
    RECYCLE_EVERY_SPECS,
    _chunk_tasks,
    _merge_records,
    _parse_args,
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


def _record(spec_index: int, steps, filter_counts, candidates, error=None):
    return {
        "spec_index": spec_index,
        "kept": [{"episode_id": spec_index, "step": step} for step in steps],
        "filter_counts": Counter(filter_counts),
        "candidates": candidates,
        "steps": 1,
        "elapsed_s": 0.0,
        "report": (
            {"id": spec_index, "termination": "error", "error": error}
            if error
            else {"id": spec_index, "termination": "max_step"}
        ),
        **({"error": error} if error else {}),
    }


def test_merge_records_follows_global_spec_order() -> None:
    # 乱序到达（w0 的 spec 2 先到）
    records = [
        _record(2, [0, 1], {"b": 1}, 10),
        _record(0, [0], {"a": 2}, 5),
        _record(1, [0, 1, 2], {"a": 1, "c": 3}, 7, error="Boom: x"),
    ]
    samples, spec_reports, filter_counts, total_candidates, missing = _merge_records(records, 3)
    assert [sample["episode_id"] for sample in samples] == [0, 1, 1, 1, 2, 2]
    assert [sample["step"] for sample in samples] == [0, 0, 1, 2, 0, 1]
    assert [report["id"] for report in spec_reports] == [0, 1, 2]
    # 计数合并 = 单进程顺序累加；键顺序也按 spec 原始顺序（a → c → b）
    assert dict(filter_counts) == {"a": 3, "b": 1, "c": 3}
    assert list(filter_counts) == ["a", "c", "b"]
    assert total_candidates == 22
    assert missing == []


def test_merge_records_reports_missing_specs() -> None:
    records = [_record(0, [0], {}, 1), _record(2, [0], {}, 1)]
    _, _, _, _, missing = _merge_records(records, 3)
    assert missing == [1]


def test_workers_cli_default_is_single_process() -> None:
    args = _parse_args(["--specs", "env/specs/x.json", "--out", "runs/x"])
    assert int(args.workers) == 1
    assert int(args.recycle_every) == RECYCLE_EVERY_SPECS
    assert int(_parse_args(["--specs", "x", "--out", "y", "--workers", "4"]).workers) == 4
    assert int(_parse_args(["--specs", "x", "--out", "y", "--recycle-every", "0"]).recycle_every) == 0
