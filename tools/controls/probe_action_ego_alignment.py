#!/usr/bin/env python3
"""探针：跨行 action 链 ↔ 真实 ego_world 对齐（P1-0C）。

数据集行 = 0.5s 策略步；同一 episode 内相邻行 ``step`` 差恒为 stride（参考 5）。
本探针独立复核：

1. stride 推断：episode 内相邻 ``step`` 差分众数；非 5 → FAIL。
2. 零链清单：``action`` 链全零行计数；「全零链 ∩ ``train_weight>0``」存在 → FAIL
   （全零链只允许出现在 weight=0 占位行）。
3. 对齐：从 ``train_weight>0`` 行抽样（默认 6000、seed 固定），用 ``(episode_id, step)``
   映射找 ``step + stride*k``（k=1..6）未来行，未来世界位姿按 ``-theta`` 旋入当前行
   坐标系，与 ``traj6[i][k-1]`` 比 Euclidean；逐 k 记 coverage/median/p90/p99/max/frac>0.5m。

判定（PASS）::

    零链∩weight>0 == 0 且 stride == 5 且 各 k: median ≤ 0.05m 且 p99 ≤ 1.0m 且 coverage ≥ 99%

输出 JSON::

    {status, dataset, n_total, n_weight_pos, stride, zero_chain:{count, with_weight_pos},
     per_k:{"1".."6": {...}}, thresholds, sample, seed, generated_at}

用法::

    tools/venv-python tools/controls/probe_action_ego_alignment.py \\
        [--dataset datasets/BTC20261007-2202_expert5k_v8/expert_bc.npz] \\
        [--sample 6000] [--seed 0] [--out runs/p1_0/action_ego_alignment.json]

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

REQUIRED_KEYS = ("episode_id", "step", "pose", "ego_world", "traj6", "action", "train_weight")
STRIDE_EXPECTED = 5
HORIZON = 6
MEDIAN_MAX_M = 0.05
P99_MAX_M = 1.0
COVERAGE_MIN = 0.99
FRAC_GT_0P5M_REF = 0.0042  # 参考值（不判，仅记录）


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _world_xy_theta(array: np.ndarray) -> np.ndarray:
    """(N,1,3) 或 (N,3) → (N,3) 世界位姿。"""
    if array.ndim == 3:
        array = array[:, 0, :]
    return np.array(array, dtype=np.float64)


def _write_and_report(payload: dict, out_path: Path, quiet: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if quiet:
        return
    print(f"[probe] action_ego_alignment: {payload['status']} -> {out_path}")
    if payload["status"] == "NOT_AVAILABLE":
        print(f"  reason: {payload.get('reason')}")
        return
    zero = payload["zero_chain"]
    print(
        f"  stride={payload['stride']} n_total={payload['n_total']} "
        f"n_weight_pos={payload['n_weight_pos']} zero_chain={zero['count']} "
        f"(with_weight_pos={zero['with_weight_pos']}) sample={payload['sample']} seed={payload['seed']}"
    )
    print("  k  coverage    median     p90        p99        max        frac>0.5m")
    for k, stats in payload["per_k"].items():
        if stats["n_pairs"]:
            print(
                f"  {k}  {stats['coverage']:.4f}    {stats['median']:.5f}   {stats['p90']:.5f}"
                f"    {stats['p99']:.5f}    {stats['max']:.4f}    {stats['frac_gt_0p5m']:.5f}"
            )
        else:
            print(f"  {k}  {stats['coverage']:.4f}    (无配对行)")
    if payload.get("fail_reasons"):
        print("  fail_reasons: " + "; ".join(payload["fail_reasons"]))


def run_probe(
    dataset: str | Path,
    out: str | Path,
    *,
    sample: int = 6000,
    seed: int = 0,
    quiet: bool = False,
) -> dict:
    """执行探针并写入 out；返回输出 payload。"""
    dataset_path = Path(dataset)
    out_path = Path(out)
    payload: dict = {
        "dataset": str(dataset),
        "sample_requested": int(sample),
        "seed": int(seed),
        "generated_at": _now(),
    }

    try:
        with np.load(dataset_path, mmap_mode="r") as data:
            available = sorted(data.files)
            missing = [key for key in REQUIRED_KEYS if key not in data.files]
            if missing:
                payload.update(
                    {
                        "status": "NOT_AVAILABLE",
                        "reason": f"数据集缺字段: {missing}",
                        "available_keys": available,
                    }
                )
                _write_and_report(payload, out_path, quiet)
                return payload
            episode_id = np.array(data["episode_id"], dtype=np.int64)
            step = np.array(data["step"], dtype=np.int64)
            pose = _world_xy_theta(np.asarray(data["pose"]))
            ego_world = _world_xy_theta(np.asarray(data["ego_world"]))
            traj6 = np.array(data["traj6"], dtype=np.float64)
            action = np.array(data["action"], dtype=np.float64)
            weight = np.array(data["train_weight"], dtype=np.float64)
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        payload.update({"status": "NOT_AVAILABLE", "reason": f"数据集不可读: {exc}"})
        _write_and_report(payload, out_path, quiet)
        return payload

    shape_errors: list[str] = []
    if pose.ndim != 2 or pose.shape[1] < 3:
        shape_errors.append(f"pose.shape={pose.shape}")
    if ego_world.ndim != 2 or ego_world.shape[1] < 3:
        shape_errors.append(f"ego_world.shape={ego_world.shape}")
    if traj6.ndim != 3 or traj6.shape[1] != HORIZON or traj6.shape[2] < 2:
        shape_errors.append(f"traj6.shape={traj6.shape}")
    if action.ndim != 3 or action.shape[1] != HORIZON:
        shape_errors.append(f"action.shape={action.shape}")
    lengths = {len(episode_id), len(step), len(pose), len(ego_world), len(traj6), len(action), len(weight)}
    if len(lengths) != 1:
        shape_errors.append(f"各字段长度不一致: {sorted(lengths)}")
    if shape_errors:
        payload.update({"status": "NOT_AVAILABLE", "reason": "字段形状不符: " + "; ".join(shape_errors)})
        _write_and_report(payload, out_path, quiet)
        return payload

    n_total = int(len(episode_id))
    n_weight_pos = int((weight > 0).sum())

    # ---- stride 推断（episode 内相邻 step 差分众数） ----
    order = np.lexsort((step, episode_id))
    episode_sorted = episode_id[order]
    step_sorted = step[order]
    same_episode = episode_sorted[1:] == episode_sorted[:-1]
    diffs = (step_sorted[1:] - step_sorted[:-1])[same_episode]
    if diffs.size:
        values, counts = np.unique(diffs, return_counts=True)
        stride: int | None = int(values[int(np.argmax(counts))])
    else:
        stride = None

    # ---- 零链清单 ----
    zero_mask = np.all(action == 0, axis=(1, 2))
    zero_count = int(zero_mask.sum())
    zero_with_pos = int((zero_mask & (weight > 0)).sum())

    fail_reasons: list[str] = []
    if stride != STRIDE_EXPECTED:
        fail_reasons.append(f"stride={stride} != {STRIDE_EXPECTED}")
    if zero_with_pos > 0:
        fail_reasons.append(f"全零 action 链 ∩ train_weight>0 = {zero_with_pos} 行")

    # ---- 抽样 + 未来行对齐 ----
    candidates = np.flatnonzero(weight > 0)
    if candidates.size == 0:
        payload.update(
            {
                "status": "NOT_AVAILABLE",
                "reason": "无 train_weight>0 行可抽样",
                "n_total": n_total,
                "n_weight_pos": 0,
                "stride": stride,
                "zero_chain": {"count": zero_count, "with_weight_pos": zero_with_pos},
            }
        )
        _write_and_report(payload, out_path, quiet)
        return payload

    rng = np.random.default_rng(seed)
    sampled = rng.choice(candidates, size=min(int(sample), int(candidates.size)), replace=False)
    per_k: dict[str, dict] = {}

    if stride is not None and stride > 0:
        key_base = int(step.max()) + stride * HORIZON + 2
        keys = episode_sorted * key_base + step_sorted
        for k in range(1, HORIZON + 1):
            query = episode_id[sampled] * key_base + (step[sampled] + stride * k)
            positions = np.searchsorted(keys, query)
            valid = positions < keys.size
            positions_clipped = np.clip(positions, 0, keys.size - 1)
            valid &= keys[positions_clipped] == query
            n_pairs = int(valid.sum())
            coverage = n_pairs / len(sampled) if len(sampled) else 0.0
            if n_pairs:
                future = order[positions_clipped[valid]]
                current = sampled[valid]
                delta_x = ego_world[future, 0] - ego_world[current, 0]
                delta_y = ego_world[future, 1] - ego_world[current, 1]
                theta = pose[current, 2]
                local_x = delta_x * np.cos(theta) + delta_y * np.sin(theta)
                local_y = -delta_x * np.sin(theta) + delta_y * np.cos(theta)
                errors = np.hypot(
                    local_x - traj6[current, k - 1, 0], local_y - traj6[current, k - 1, 1]
                )
                stats = {
                    "n_pairs": n_pairs,
                    "coverage": coverage,
                    "median": float(np.median(errors)),
                    "p90": float(np.percentile(errors, 90)),
                    "p99": float(np.percentile(errors, 99)),
                    "max": float(errors.max()),
                    "frac_gt_0p5m": float((errors > 0.5).mean()),
                }
                if stats["coverage"] < COVERAGE_MIN:
                    fail_reasons.append(f"k={k} coverage={coverage:.4f} < {COVERAGE_MIN}")
                if stats["median"] > MEDIAN_MAX_M:
                    fail_reasons.append(f"k={k} median={stats['median']:.5f} > {MEDIAN_MAX_M}")
                if stats["p99"] > P99_MAX_M:
                    fail_reasons.append(f"k={k} p99={stats['p99']:.5f} > {P99_MAX_M}")
            else:
                stats = {
                    "n_pairs": 0,
                    "coverage": coverage,
                    "median": None,
                    "p90": None,
                    "p99": None,
                    "max": None,
                    "frac_gt_0p5m": None,
                }
                fail_reasons.append(f"k={k} coverage=0（无未来配对行）")
            per_k[str(k)] = stats
    else:
        fail_reasons.append("无法推断 stride（无同 episode 相邻行）")

    payload.update(
        {
            "status": "FAIL" if fail_reasons else "PASS",
            "n_total": n_total,
            "n_weight_pos": n_weight_pos,
            "stride": stride,
            "zero_chain": {"count": zero_count, "with_weight_pos": zero_with_pos},
            "per_k": per_k,
            "thresholds": {
                "stride_expected": STRIDE_EXPECTED,
                "median_max_m": MEDIAN_MAX_M,
                "p99_max_m": P99_MAX_M,
                "coverage_min": COVERAGE_MIN,
                "frac_gt_0p5m_ref": FRAC_GT_0P5M_REF,
            },
            "sample": int(len(sampled)),
            "fail_reasons": fail_reasons,
        }
    )
    _write_and_report(payload, out_path, quiet)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探针：跨行 action 链 ↔ 真实 ego_world 对齐")
    parser.add_argument(
        "--dataset",
        default="datasets/BTC20261007-2202_expert5k_v8/expert_bc.npz",
        help="专家 BC 数据集 npz",
    )
    parser.add_argument("--sample", type=int, default=6000, help="train_weight>0 行抽样数")
    parser.add_argument("--seed", type=int, default=0, help="抽样随机种子")
    parser.add_argument(
        "--out", default="runs/p1_0/action_ego_alignment.json", help="输出 JSON 路径"
    )
    args = parser.parse_args(argv)

    payload = run_probe(args.dataset, args.out, sample=args.sample, seed=args.seed)
    return {"PASS": 0, "FAIL": 1, "NOT_AVAILABLE": 2}.get(payload["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
