#!/usr/bin/env python3
"""探针：几何代理覆盖（LD 通道）（P1-0C）。

数据 = 专家 BC npz（默认 ``datasets/BTC20261007-2202_expert5k_v8/expert_bc.npz``）。

指标：

- ``ld_valid_ratio``：``ld_mask`` 全表均值（有效槽位占比；16 槽/帧）。
- ``solid_available_ratio``：帧级"≥1 个有效实线槽位"占比。实线 id 集合从
  ``env/obs/ld.py::LINE_TYPE_IDS``（只读）按名称含 ``SOLID`` 派生（= {2,3,6,7}；
  ``PASSING_DOUBLE_YELLOW``(8) 与边界/护栏(9..12) 不计入）。
- ``solid_slot_ratio``：有效槽位中实线槽位占比（附加）。
- LD 点距（自车原点）分布摘要：min/p10/median/p90/p99/max（有效槽位）；
  ``near_field_coverage_ratio`` = 帧级"≥1 个有效 LD 点距 ≤ ``--near-field``"占比。
- ``plan_coverage_ratio``：6 点链（``traj6``）中至少一点距任意有效 LD 点 ≤
  ``--plan-radius``（默认 2m）的帧占比；另报 ``train_weight>0`` 子集。

输出 ``{status, ld_valid_ratio, solid_available_ratio, plan_coverage_ratio, ...}``
→ ``runs/p1_0/probe_geometry_coverage.json``。
status 判定为**诊断性下限**（默认 ld_valid≥0.5、solid_available≥0.5、
plan_coverage≥0.8，均可用 CLI 调整；非 §7.10 硬门）。

用法::

    tools/venv-python tools/controls/probe_geometry_coverage.py [--dataset ...] \\
        [--out runs/p1_0/probe_geometry_coverage.json] [--chunk 50000] \\
        [--plan-radius 2.0] [--near-field 5.0]

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

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:  # 只读引用线型 id 表（env/obs/ld.py）
    from env.obs.ld import LINE_TYPE_IDS

    SOLID_LINE_IDS: tuple[int, ...] = tuple(
        sorted(value for name, value in LINE_TYPE_IDS.items() if "SOLID" in name)
    )
except Exception:  # noqa: BLE001 - 导入失败时探针报 NOT_AVAILABLE
    LINE_TYPE_IDS = None
    SOLID_LINE_IDS = ()

REQUIRED_KEYS = ("ld", "ld_mask", "traj6")
DEFAULT_MIN_LD_VALID = 0.5
DEFAULT_MIN_SOLID_AVAILABLE = 0.5
DEFAULT_MIN_PLAN_COVERAGE = 0.8


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_and_report(payload: dict, out_path: Path, quiet: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if quiet:
        return
    print(f"[probe] geometry_coverage: {payload['status']} -> {out_path}")
    if payload["status"] == "NOT_AVAILABLE":
        print(f"  reason: {payload.get('reason')}")
        return
    print(
        f"  n_frames={payload['n_frames']} ld_valid_ratio={payload['ld_valid_ratio']:.4f} "
        f"solid_available_ratio={payload['solid_available_ratio']:.4f} "
        f"(solid_slot_ratio={payload['solid_slot_ratio']:.4f})"
    )
    weight_ratio = payload["plan_coverage_ratio_weight_pos"]
    weight_text = f"{weight_ratio:.4f}" if weight_ratio is not None else "n/a"
    print(
        f"  plan_coverage_ratio={payload['plan_coverage_ratio']:.4f} "
        f"(weight>0: {weight_text}) "
        f"near_field_coverage_ratio={payload['near_field_coverage_ratio']:.4f}"
    )
    dist = payload["ld_point_distance_m"]

    def _fmt(value: float | None) -> str:
        return f"{value:.3f}" if value is not None else "n/a"

    print(
        f"  ld point dist m: min={_fmt(dist['min'])} p10={_fmt(dist['p10'])} "
        f"median={_fmt(dist['median'])} p90={_fmt(dist['p90'])} "
        f"p99={_fmt(dist['p99'])} max={_fmt(dist['max'])}"
    )


def run_probe(
    dataset: str | Path,
    out: str | Path,
    *,
    chunk: int = 50000,
    plan_radius: float = 2.0,
    near_field: float = 5.0,
    min_ld_valid: float = DEFAULT_MIN_LD_VALID,
    min_solid_available: float = DEFAULT_MIN_SOLID_AVAILABLE,
    min_plan_coverage: float = DEFAULT_MIN_PLAN_COVERAGE,
    quiet: bool = False,
) -> dict:
    """执行探针并写入 out；返回输出 payload。"""
    dataset_path = Path(dataset)
    out_path = Path(out)
    payload: dict = {
        "dataset": str(dataset),
        "plan_radius_m": float(plan_radius),
        "near_field_m": float(near_field),
        "generated_at": _now(),
    }
    if not SOLID_LINE_IDS:
        payload.update(
            {"status": "NOT_AVAILABLE", "reason": "env.obs.ld::LINE_TYPE_IDS 不可导入"}
        )
        _write_and_report(payload, out_path, quiet)
        return payload

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
                _write_and_report(payload, out_path, quiet)
                return payload
            n_frames_total = int(data["ld_mask"].shape[0])

            n_frames = 0
            n_valid_slots = 0
            n_solid_slots = 0
            n_solid_frames = 0
            n_near_frames = 0
            n_plan_ok = 0
            n_weight_pos = 0
            n_plan_ok_weight_pos = 0
            distance_parts: list[np.ndarray] = []
            has_weight = "train_weight" in data.files

            solid_ids = np.array(SOLID_LINE_IDS, dtype=np.float32)
            for start in range(0, n_frames_total, max(int(chunk), 1)):
                end = min(start + max(int(chunk), 1), n_frames_total)
                ld = np.asarray(data["ld"][start:end], dtype=np.float32)
                mask = np.asarray(data["ld_mask"][start:end], dtype=np.float32) > 0
                traj = np.asarray(data["traj6"][start:end], dtype=np.float32)[:, :, :2]
                weight = (
                    np.asarray(data["train_weight"][start:end], dtype=np.float32)
                    if has_weight
                    else np.ones(end - start, dtype=np.float32)
                )

                n_frames += end - start
                n_valid_slots += int(mask.sum())

                line_ids = ld[:, :, 5:7]
                solid = mask & np.isin(line_ids, solid_ids).any(axis=2)
                n_solid_slots += int(solid.sum())
                n_solid_frames += int((solid.sum(axis=1) > 0).sum())

                points = ld[:, :, :2]
                distances = np.linalg.norm(points, axis=2)
                distance_parts.append(distances[mask])
                n_near_frames += int(
                    (np.where(mask, distances, np.inf).min(axis=1) <= near_field).sum()
                )

                plan_dist = np.linalg.norm(
                    traj[:, :, None, :] - points[:, None, :, :], axis=3
                )  # (B,6,16)
                plan_dist = np.where(mask[:, None, :], plan_dist, np.inf)
                plan_ok = plan_dist.min(axis=(1, 2)) <= plan_radius
                n_plan_ok += int(plan_ok.sum())
                weight_pos = weight > 0
                n_weight_pos += int(weight_pos.sum())
                n_plan_ok_weight_pos += int(plan_ok[weight_pos].sum())
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        payload.update({"status": "NOT_AVAILABLE", "reason": f"数据集不可读: {exc}"})
        _write_and_report(payload, out_path, quiet)
        return payload

    if n_frames == 0:
        payload.update({"status": "NOT_AVAILABLE", "reason": "数据集为空"})
        _write_and_report(payload, out_path, quiet)
        return payload

    ld_valid_ratio = n_valid_slots / (n_frames * 16)
    solid_slot_ratio = n_solid_slots / n_valid_slots if n_valid_slots else 0.0
    solid_available_ratio = n_solid_frames / n_frames
    plan_coverage_ratio = n_plan_ok / n_frames
    plan_coverage_ratio_weight_pos = n_plan_ok_weight_pos / n_weight_pos if n_weight_pos else None

    all_distances = (
        np.concatenate(distance_parts) if distance_parts else np.zeros(0, dtype=np.float32)
    )
    if all_distances.size:
        distance_summary = {
            "min": float(all_distances.min()),
            "p10": float(np.percentile(all_distances, 10)),
            "median": float(np.median(all_distances)),
            "p90": float(np.percentile(all_distances, 90)),
            "p99": float(np.percentile(all_distances, 99)),
            "max": float(all_distances.max()),
        }
    else:
        distance_summary = dict.fromkeys(("min", "p10", "median", "p90", "p99", "max"))

    checks = {
        "ld_valid": ld_valid_ratio >= min_ld_valid,
        "solid_available": solid_available_ratio >= min_solid_available,
        "plan_coverage": plan_coverage_ratio >= min_plan_coverage,
    }
    payload.update(
        {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "n_frames": n_frames,
            "n_valid_slots": n_valid_slots,
            "ld_valid_ratio": ld_valid_ratio,
            "solid_available_ratio": solid_available_ratio,
            "solid_slot_ratio": solid_slot_ratio,
            "plan_coverage_ratio": plan_coverage_ratio,
            "plan_coverage_ratio_weight_pos": plan_coverage_ratio_weight_pos,
            "near_field_coverage_ratio": n_near_frames / n_frames,
            "ld_point_distance_m": distance_summary,
            "solid_line_type_ids": list(SOLID_LINE_IDS),
            "checks": checks,
            "thresholds": {
                "ld_valid_min": min_ld_valid,
                "solid_available_min": min_solid_available,
                "plan_coverage_min": min_plan_coverage,
            },
        }
    )
    _write_and_report(payload, out_path, quiet)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探针：几何代理覆盖（LD 有效槽/实线/plan 覆盖）")
    parser.add_argument(
        "--dataset",
        default="datasets/BTC20261007-2202_expert5k_v8/expert_bc.npz",
        help="专家 BC 数据集 npz",
    )
    parser.add_argument("--out", default="runs/p1_0/probe_geometry_coverage.json", help="输出 JSON 路径")
    parser.add_argument("--chunk", type=int, default=50000, help="分块行数（控内存）")
    parser.add_argument("--plan-radius", type=float, default=2.0, help="plan 点- LD 点距离阈值（m）")
    parser.add_argument("--near-field", type=float, default=5.0, help="近场覆盖半径（m）")
    parser.add_argument("--min-ld-valid", type=float, default=DEFAULT_MIN_LD_VALID)
    parser.add_argument("--min-solid-available", type=float, default=DEFAULT_MIN_SOLID_AVAILABLE)
    parser.add_argument("--min-plan-coverage", type=float, default=DEFAULT_MIN_PLAN_COVERAGE)
    args = parser.parse_args(argv)

    payload = run_probe(
        args.dataset,
        args.out,
        chunk=args.chunk,
        plan_radius=args.plan_radius,
        near_field=args.near_field,
        min_ld_valid=args.min_ld_valid,
        min_solid_available=args.min_solid_available,
        min_plan_coverage=args.min_plan_coverage,
    )
    return {"PASS": 0, "FAIL": 1, "NOT_AVAILABLE": 2}.get(payload["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
