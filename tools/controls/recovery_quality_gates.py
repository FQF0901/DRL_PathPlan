#!/usr/bin/env python3
"""恢复样本质量门（P1-0C；§7.10 A5 质量门 = 纯函数 + CLI）。

输入：JSONL（每行一个样本行）或 JSON（数组 / ``{"samples": [...]}``）。
输出：summary JSON（P1-1 采集器可直接复用）+ 可选逐行 JSONL（``--rows-out``）。

样本行契约（P1-1 recovery 采集器应写出；别名兼容见 ``_ALIASES``）::

    episode_id: int, step: int, dt: float           # 时间契约（dt 应 = 0.5 s）
    pose: [x, y, theta]                             # student 真实状态（世界系）
    recovery_traj: [[x, y], ...×6]                  # 专家恢复轨迹（自车系，k=1..6 步）
    action_chain: [[ds, dtheta], ...×6]             # 专家空问动作链（BC 口径）
    margin_before / margin_after: float             # 恢复窗 min_signed_margin（前/后）
    rc_before / rc_after: float                     # route completion（前/后）
    takeover_off_road / takeover_collision: bool    # 接管后是否仍 off/collision
    geometry_labels: 数值列表/字典（含有限值）       # 几何标签
    recovery_traj_world: 可选 [[x, y], ...×6]       # 世界系恢复轨迹；存在时校验坐标变换

门（任一命中 → 样本不可用）::

    post_takeover_failure       专家接管后仍 off/collision
    margin_no_improvement       恢复窗 margin 无改善（gain ≤ --min-margin-gain）
    route_no_improvement        route progress 无改善（gain ≤ --min-route-gain）
    action_traj_inconsistent    action 链积分（env.tracking.interpolate 单一真源）
                                与 recovery_traj 最大偏差 > --action-traj-tol
    coordinate_time_contract    step/dt/pose/traj 契约失败（含 world↔local 交叉校验）
    geometry_labels_unavailable 几何标签缺失/全 NaN

用法::

    tools/venv-python tools/controls/recovery_quality_gates.py \\
        --input <samples.jsonl> [--out runs/p1_0/recovery_quality_gates_summary.json] \\
        [--rows-out <decisions.jsonl>] [--action-traj-tol 0.5] [--min-margin-gain 0.0] \\
        [--min-route-gain 0.0] [--dt 0.5] [--dt-tol 1e-6] [--coord-cross-tol 1e-3] [--limit N]

退出码：PASS=0（全部行可用）/ FAIL=1（存在不可用行）/ NOT_AVAILABLE=2（输入不可读或零行）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.tracking import interpolate  # noqa: E402  # §8.5 单一真源（只读引用）

HORIZON = 6
POLICY_HZ = 10  # interpolate 子步频率（0.1 s）

_ALIASES: dict[str, tuple[str, ...]] = {
    "episode_id": ("episode_id", "ep"),
    "step": ("step", "t"),
    "pose": ("pose", "ego_pose"),
    "takeover_off_road": ("takeover_off_road", "post_takeover_off_road", "recovery_off_road"),
    "takeover_collision": ("takeover_collision", "post_takeover_collision", "recovery_collision"),
    "takeover_termination": (
        "takeover_termination",
        "post_takeover_termination",
        "recovery_termination",
    ),
    "margin_before": ("margin_before", "min_signed_margin_before", "margin_window_before"),
    "margin_after": ("margin_after", "min_signed_margin_after", "margin_window_after"),
    "rc_before": ("rc_before", "route_completion_before", "route_progress_before"),
    "rc_after": ("rc_after", "route_completion_after", "route_progress_after"),
    "recovery_traj": ("recovery_traj", "expert_recovery_traj", "recovery_trajectory"),
    "recovery_traj_world": ("recovery_traj_world", "expert_recovery_traj_world"),
    "action_chain": ("action_chain", "action"),
    "dt": ("dt", "policy_dt", "dt_s"),
    "geometry_labels": ("geometry_labels", "labels", "geom_labels"),
    "geometry_labels_valid": ("geometry_labels_valid", "labels_valid"),
}

_FAIL_TERMINATIONS = {"collision", "out_of_road", "off_road", "offroad"}


@dataclass(frozen=True)
class GateParams:
    action_traj_tol_m: float = 0.5
    min_margin_gain: float = 0.0
    min_route_gain: float = 0.0
    dt: float = 0.5
    dt_tol: float = 1e-6
    coord_cross_tol_m: float = 1e-3


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _get(row: dict, name: str) -> Any:
    for key in _ALIASES[name]:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _to_array(value: Any, *, rows: int, cols: int) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] != rows or arr.shape[1] < cols:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def gate_post_takeover(row: dict, params: GateParams) -> list[str]:
    off = _get(row, "takeover_off_road")
    collision = _get(row, "takeover_collision")
    termination = _get(row, "takeover_termination")
    if off is None and collision is None and termination is None:
        return ["missing:takeover_outcome"]
    reasons: list[str] = []
    if off is True:
        reasons.append("post_takeover_off_road")
    if collision is True:
        reasons.append("post_takeover_collision")
    if isinstance(termination, str) and termination.lower() in _FAIL_TERMINATIONS:
        reasons.append(f"post_takeover_termination:{termination}")
    return reasons


def _improvement_gate(
    row: dict, before_key: str, after_key: str, min_gain: float, label: str
) -> list[str]:
    """label 仅用于调用方语义；返回 detail（evaluate_row 会加门名前缀）。"""
    before = _as_float(_get(row, before_key))
    after = _as_float(_get(row, after_key))
    if before is None or after is None:
        return [f"missing:{before_key}/{after_key}"]
    gain = after - before
    if gain <= min_gain:
        return [f"gain={gain:.6f}"]
    return []


def gate_margin(row: dict, params: GateParams) -> list[str]:
    return _improvement_gate(
        row, "margin_before", "margin_after", params.min_margin_gain, "margin_no_improvement"
    )


def gate_route(row: dict, params: GateParams) -> list[str]:
    return _improvement_gate(
        row, "rc_before", "rc_after", params.min_route_gain, "route_no_improvement"
    )


def gate_action_traj(row: dict, params: GateParams) -> list[str]:
    actions_raw = _get(row, "action_chain")
    traj_raw = _get(row, "recovery_traj")
    if actions_raw is None:
        return ["missing:action_chain"]
    if traj_raw is None:
        return ["missing:recovery_traj"]
    actions = _to_array(actions_raw, rows=HORIZON, cols=2)
    traj = _to_array(traj_raw, rows=HORIZON, cols=2)
    if actions is None:
        return ["action_chain_invalid"]
    if traj is None:
        return ["recovery_traj_invalid"]
    substeps = int(round(params.dt * POLICY_HZ))
    if substeps < 1 or abs(params.dt * POLICY_HZ - substeps) > 1e-9:
        return [f"dt_not_integratable:{params.dt}"]
    try:
        dense = interpolate(actions, dt=params.dt, hz=POLICY_HZ)
    except ValueError as exc:
        return [f"action_integration_failed:{exc}"]
    key = dense[[i * substeps - 1 for i in range(1, HORIZON + 1)], :2]
    max_err = float(np.max(np.linalg.norm(key - traj[:, :2], axis=1)))
    if max_err > params.action_traj_tol_m:
        return [f"max_err={max_err:.6f}>{params.action_traj_tol_m}"]
    return []


def gate_coordinate_time(row: dict, params: GateParams) -> list[str]:
    reasons: list[str] = []
    if _get(row, "episode_id") is None:
        reasons.append("missing:episode_id")
    step = _as_float(_get(row, "step"))
    if step is None:
        reasons.append("missing:step")
    elif step < 0:
        reasons.append(f"step_negative:{step}")
    dt = _as_float(_get(row, "dt"))
    if dt is None:
        reasons.append("missing:dt")
    elif abs(dt - params.dt) > params.dt_tol:
        reasons.append(f"dt_contract:{dt}!={params.dt}")

    pose_raw = _get(row, "pose")
    pose: np.ndarray | None = None
    if pose_raw is None:
        reasons.append("missing:pose")
    else:
        try:
            pose = np.asarray(pose_raw, dtype=np.float64)
        except (TypeError, ValueError):
            pose = None
        if pose is None or pose.ndim != 1 or pose.size < 3 or not np.isfinite(pose).all():
            reasons.append("pose_invalid")
            pose = None

    traj_raw = _get(row, "recovery_traj")
    if traj_raw is None:
        reasons.append("missing:recovery_traj")
    else:
        traj = _to_array(traj_raw, rows=HORIZON, cols=2)
        if traj is None:
            reasons.append("recovery_traj_invalid")
        else:
            world_raw = _get(row, "recovery_traj_world")
            if world_raw is not None:
                world = _to_array(world_raw, rows=HORIZON, cols=2)
                if world is None or pose is None:
                    reasons.append("recovery_traj_world_invalid")
                else:
                    theta = float(pose[2])
                    cos_t, sin_t = math.cos(theta), math.sin(theta)
                    delta_x = world[:, 0] - float(pose[0])
                    delta_y = world[:, 1] - float(pose[1])
                    local = np.stack(
                        [delta_x * cos_t + delta_y * sin_t, -delta_x * sin_t + delta_y * cos_t],
                        axis=1,
                    )
                    cross_err = float(np.max(np.linalg.norm(local - traj[:, :2], axis=1)))
                    if cross_err > params.coord_cross_tol_m:
                        reasons.append(f"coord_cross_check:{cross_err:.6f}")
    return reasons


def gate_geometry_labels(row: dict, params: GateParams) -> list[str]:
    raw = _get(row, "geometry_labels")
    if raw is None:
        return ["missing:geometry_labels"]
    if _get(row, "geometry_labels_valid") is False:
        return ["geometry_labels_unavailable"]
    if isinstance(raw, dict):
        if not raw:
            return ["geometry_labels_empty"]
        numbers = [
            value
            for value in raw.values()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if numbers and np.isfinite(np.asarray(numbers, dtype=np.float64)).any():
            return []
        if any(value is not None for value in raw.values()):
            return []
        return ["geometry_labels_unavailable"]
    try:
        arr = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return ["geometry_labels_invalid"]
    if arr.size == 0:
        return ["geometry_labels_empty"]
    if not np.isfinite(arr).any():
        return ["geometry_labels_all_nan"]
    return []


GATES: tuple[tuple[str, Callable[[dict, GateParams], list[str]]], ...] = (
    ("post_takeover_failure", gate_post_takeover),
    ("margin_no_improvement", gate_margin),
    ("route_no_improvement", gate_route),
    ("action_traj_inconsistent", gate_action_traj),
    ("coordinate_time_contract", gate_coordinate_time),
    ("geometry_labels_unavailable", gate_geometry_labels),
)


def evaluate_row(row: Any, params: GateParams = GateParams()) -> dict:
    """单行过全部门；返回 ``{"usable": bool, "reasons": [...], "gates": {...}}``。"""
    if not isinstance(row, dict):
        return {
            "usable": False,
            "reasons": ["row_not_object"],
            "gates": {name: ["row_not_object"] for name, _ in GATES},
        }
    gates: dict[str, list[str]] = {}
    reasons: list[str] = []
    for name, gate in GATES:
        gate_reasons = gate(row, params)
        gates[name] = gate_reasons
        reasons.extend(f"{name}:{item}" for item in gate_reasons)
    return {"usable": not reasons, "reasons": reasons, "gates": gates}


def _load_rows(path: Path, limit: int | None) -> list:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            rows = next(
                (data[key] for key in ("samples", "rows", "records") if isinstance(data.get(key), list)),
                [data],
            )
        else:
            raise ValueError("JSON 顶层应为数组或含 samples/rows/records 的对象")
    if limit is not None:
        rows = rows[: max(int(limit), 0)]
    return rows


def build_summary(
    rows: list, params: GateParams, input_path: Path
) -> tuple[dict, list[dict]]:
    results = [evaluate_row(row, params) for row in rows]
    n_usable = sum(1 for result in results if result["usable"])
    gate_fail_counts = {
        name: sum(1 for result in results if result["gates"][name]) for name, _ in GATES
    }
    reason_counts: Counter[str] = Counter(
        reason.split("=", 1)[0] for result in results for reason in result["reasons"]
    )
    summary = {
        "status": "PASS" if n_usable == len(rows) else "FAIL",
        "input": str(input_path),
        "n_rows": len(rows),
        "n_usable": n_usable,
        "usable_ratio": (n_usable / len(rows)) if rows else None,
        "gate_fail_counts": gate_fail_counts,
        "reason_counts": dict(reason_counts),
        "params": asdict(params),
        "gate_order": [name for name, _ in GATES],
        "generated_at": _now(),
    }
    return summary, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="恢复样本质量门（§7.10 A5 纯函数 + CLI）")
    parser.add_argument("--input", required=True, help="样本 JSONL 或 JSON")
    parser.add_argument(
        "--out", default="runs/p1_0/recovery_quality_gates_summary.json", help="summary JSON 路径"
    )
    parser.add_argument("--rows-out", default=None, help="可选：逐行判定 JSONL")
    parser.add_argument("--action-traj-tol", type=float, default=0.5, help="action 链-轨迹容差（m）")
    parser.add_argument("--min-margin-gain", type=float, default=0.0, help="margin 最小改善量")
    parser.add_argument("--min-route-gain", type=float, default=0.0, help="route 最小改善量")
    parser.add_argument("--dt", type=float, default=0.5, help="期望策略步长（s）")
    parser.add_argument("--dt-tol", type=float, default=1e-6, help="dt 容差（s）")
    parser.add_argument("--coord-cross-tol", type=float, default=1e-3, help="world↔local 交叉校验容差（m）")
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 行")
    args = parser.parse_args(argv)

    params = GateParams(
        action_traj_tol_m=args.action_traj_tol,
        min_margin_gain=args.min_margin_gain,
        min_route_gain=args.min_route_gain,
        dt=args.dt,
        dt_tol=args.dt_tol,
        coord_cross_tol_m=args.coord_cross_tol,
    )
    input_path = Path(args.input)
    out_path = Path(args.out)

    if not input_path.is_file():
        summary = {
            "status": "NOT_AVAILABLE",
            "input": str(input_path),
            "reason": "输入文件不存在",
            "params": asdict(params),
            "generated_at": _now(),
        }
        _write_summary(summary, out_path)
        print(f"[gates] recovery_quality_gates: NOT_AVAILABLE ({summary['reason']}) -> {out_path}")
        return 2
    try:
        rows = _load_rows(input_path, args.limit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        summary = {
            "status": "NOT_AVAILABLE",
            "input": str(input_path),
            "reason": f"输入不可读: {exc}",
            "params": asdict(params),
            "generated_at": _now(),
        }
        _write_summary(summary, out_path)
        print(f"[gates] recovery_quality_gates: NOT_AVAILABLE ({summary['reason']}) -> {out_path}")
        return 2
    if not rows:
        summary = {
            "status": "NOT_AVAILABLE",
            "input": str(input_path),
            "reason": "输入零行",
            "params": asdict(params),
            "generated_at": _now(),
        }
        _write_summary(summary, out_path)
        print(f"[gates] recovery_quality_gates: NOT_AVAILABLE ({summary['reason']}) -> {out_path}")
        return 2

    summary, results = build_summary(rows, params, input_path)
    _write_summary(summary, out_path)
    if args.rows_out:
        rows_out = Path(args.rows_out)
        rows_out.parent.mkdir(parents=True, exist_ok=True)
        with rows_out.open("w", encoding="utf-8") as handle:
            for index, result in enumerate(results):
                handle.write(
                    json.dumps(
                        {"index": index, "usable": result["usable"], "reasons": result["reasons"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(
        f"[gates] recovery_quality_gates: {summary['status']} "
        f"n_rows={summary['n_rows']} n_usable={summary['n_usable']} -> {out_path}"
    )
    for name, count in summary["gate_fail_counts"].items():
        if count:
            print(f"  {name}: {count} 行")
    return 0 if summary["status"] == "PASS" else 1


def _write_summary(summary: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
