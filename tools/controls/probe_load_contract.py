#!/usr/bin/env python3
"""探针：checkpoint 加载契约（P1-0C；加载报告 = ``metrics.json::meta.ckpt_load`` 提取）。

读 ``<run>/metrics.json::meta.ckpt_load`` → 写
``<run>/probe_results.json::probes.checkpoint_load_contract``::

    {"status": ..., "counts": {...}, "loaded_ratio": ..., "ckpt": ..., "sha256": ...}

status：PASS = counts 全 0 且 loaded_keys == model_keys；counts 非全 0 → FAIL；
``meta.ckpt_load`` 缺失 → NOT_AVAILABLE。摘要打到 stdout；
退出码 PASS=0 / FAIL=1 / NOT_AVAILABLE=2。

用法::

    tools/venv-python tools/controls/probe_load_contract.py --run-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.controls.artifacts import build_load_report  # noqa: E402

PROBE_NAME = "checkpoint_load_contract"
_EXIT_BY_STATUS = {"PASS": 0, "FAIL": 1, "NOT_AVAILABLE": 2}


def run_probe(run_dir: str | Path, *, quiet: bool = False) -> dict:
    """执行探针并合并写入 probe_results.json；返回探针 payload。"""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.json"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metrics = None

    report = build_load_report(metrics)
    payload = {key: report.get(key) for key in ("status", "counts", "loaded_ratio", "ckpt", "sha256")}

    probe_results_path = run_dir / "probe_results.json"
    if probe_results_path.is_file():
        data = json.loads(probe_results_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"probe_results.json 顶层不是对象: {probe_results_path}")
    else:
        data = {}
    probes = data.setdefault("probes", {})
    if not isinstance(probes, dict):
        raise ValueError(f"probe_results.json::probes 不是对象: {probe_results_path}")
    probes[PROBE_NAME] = payload
    probe_results_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not quiet:
        print(
            f"[probe] {PROBE_NAME}: {payload['status']} "
            f"counts={payload['counts']} loaded_ratio={payload['loaded_ratio']} "
            f"ckpt={payload['ckpt']}"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探针：checkpoint 加载契约（写入 probe_results.json）")
    parser.add_argument("--run-dir", required=True, help="实验 run 目录（需含 metrics.json）")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[error] run-dir 不存在: {run_dir}", file=sys.stderr)
        return 2
    payload = run_probe(run_dir)
    return _EXIT_BY_STATUS.get(payload["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
