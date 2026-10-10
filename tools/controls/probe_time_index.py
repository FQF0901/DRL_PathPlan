#!/usr/bin/env python3
"""探针：数据集时间索引契约（P1-0C）。

检查 `episode_id` 分组内 `step` 单调递增、相邻行差恒为 5（0.5s 策略步、0.1s 物理步）。
数据集 = 专家 BC npz（默认 ``datasets/BTC20261007-2202_expert5k_v8/expert_bc.npz``）。

输出 JSON::

    {status, dataset, stride, n_violations, n_episodes, checks, generated_at}

``n_violations`` = 同 episode 相邻行中 ``diff != 5`` 的对数（应恒为 0）。

用法::

    tools/venv-python tools/controls/probe_time_index.py \\
        [--dataset datasets/BTC20261007-2202_expert5k_v8/expert_bc.npz] \\
        [--out runs/p1_0/probe_time_index.json]

退出码：PASS=0 / FAIL=1 / NOT_AVAILABLE=2。
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

STRIDE_EXPECTED = 5
REQUIRED_KEYS = ("episode_id", "step")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_probe(dataset: str | Path, out: str | Path, *, quiet: bool = False) -> dict:
    """执行探针并写入 out；返回输出 payload。"""
    dataset_path = Path(dataset)
    out_path = Path(out)
    payload: dict = {"dataset": str(dataset), "generated_at": _now()}

    try:
        with np.load(dataset_path, mmap_mode="r") as data:
            missing = [key for key in REQUIRED_KEYS if key not in data.files]
            if missing:
                payload.update(
                    {
                        "status": "NOT_AVAILABLE",
                        "reason": f"数据集缺字段: {missing}",
                        "available_keys": sorted(data.files),
                    }
                )
                _write(payload, out_path, quiet)
                return payload
            episode_id = np.array(data["episode_id"], dtype=np.int64)
            step = np.array(data["step"], dtype=np.int64)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        payload.update({"status": "NOT_AVAILABLE", "reason": f"数据集不可读: {exc}"})
        _write(payload, out_path, quiet)
        return payload

    if len(episode_id) != len(step):
        payload.update({"status": "NOT_AVAILABLE", "reason": "episode_id/step 长度不一致"})
        _write(payload, out_path, quiet)
        return payload

    n_episodes = int(np.unique(episode_id).size)
    order = np.lexsort((step, episode_id))
    episode_sorted = episode_id[order]
    step_sorted = step[order]
    same_episode = episode_sorted[1:] == episode_sorted[:-1]
    diffs = (step_sorted[1:] - step_sorted[:-1])[same_episode]
    n_pairs = int(diffs.size)

    if n_pairs == 0:
        payload.update(
            {
                "status": "NOT_AVAILABLE",
                "reason": "无同 episode 相邻行可检（每 episode ≤1 行）",
                "stride": None,
                "n_violations": 0,
                "n_episodes": n_episodes,
                "checks": {"stride_expected": STRIDE_EXPECTED, "n_pairs_checked": 0},
            }
        )
        _write(payload, out_path, quiet)
        return payload

    values, counts = np.unique(diffs, return_counts=True)
    stride = int(values[int(np.argmax(counts))])
    n_violations = int((diffs != STRIDE_EXPECTED).sum())
    monotonic = bool((diffs > 0).all())
    all_equal_mode = bool((diffs == stride).all())
    episodes_with_pairs = int(np.unique(episode_sorted[:-1][same_episode]).size)

    payload.update(
        {
            "status": "PASS" if n_violations == 0 else "FAIL",
            "stride": stride,
            "n_violations": n_violations,
            "n_episodes": n_episodes,
            "checks": {
                "stride_expected": STRIDE_EXPECTED,
                "stride_mode": stride,
                "monotonic_increasing": monotonic,
                "all_diffs_equal_mode": all_equal_mode,
                "n_pairs_checked": n_pairs,
                "n_episodes_with_pairs": episodes_with_pairs,
            },
        }
    )
    _write(payload, out_path, quiet)
    return payload


def _write(payload: dict, out_path: Path, quiet: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if quiet:
        return
    print(f"[probe] time_index: {payload['status']} -> {out_path}")
    if payload["status"] == "NOT_AVAILABLE":
        print(f"  reason: {payload.get('reason')}")
        return
    checks = payload["checks"]
    print(
        f"  stride={payload['stride']} n_episodes={payload['n_episodes']} "
        f"n_violations={payload['n_violations']} pairs={checks['n_pairs_checked']} "
        f"monotonic={checks['monotonic_increasing']} all_equal_mode={checks['all_diffs_equal_mode']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探针：数据集时间索引契约（step 单调 + 相邻差 5）")
    parser.add_argument(
        "--dataset",
        default="datasets/BTC20261007-2202_expert5k_v8/expert_bc.npz",
        help="专家 BC 数据集 npz",
    )
    parser.add_argument("--out", default="runs/p1_0/probe_time_index.json", help="输出 JSON 路径")
    args = parser.parse_args(argv)

    payload = run_probe(args.dataset, args.out)
    return {"PASS": 0, "FAIL": 1, "NOT_AVAILABLE": 2}.get(payload["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
