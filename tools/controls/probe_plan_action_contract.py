#!/usr/bin/env python3
"""探针：plan/action 契约（P1-0C）。

数据 = forensics JSON 的 ``episodes[].records[]``（每条含 ``decision`` / ``mu`` / ``plan``）。
抽样 ``decision=true`` 步，断言 ``plan[0] ≈ mu``（逐维 ``|Δ| ≤ tol``）；
另统计 plan 全列有限比例（附加统计，不参与判定）。

输出 JSON::

    {"status": "PASS" 当且仅当 violations == 0, "n_checked": ..., "violations": ...,
     "tol": ..., "source": ...}

数据实际结构不符（无 episodes / 无 decision=true / 无 plan·mu 字段）时优雅返回
``{"status": "NOT_AVAILABLE", ...}`` 并在 ``structure`` 中报告实际结构。
大文件（>64MB）流式解析，不整文件载入内存。

用法::

    tools/venv-python tools/controls/probe_plan_action_contract.py \\
        --forensics-json runs/forensics/p0_d2.json [--sample-rate 0.01] [--tol 1e-6] \\
        [--out runs/p1_0/probe_plan_action_contract.json]

退出码：PASS=0 / FAIL=1 / NOT_AVAILABLE=2。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from datetime import datetime
from pathlib import Path

_CHUNK = 1 << 16
_KEY_RE = re.compile(r'"episodes"\s*:')
_WHOLE_FILE_LIMIT = 64 * 1024 * 1024  # NOT_AVAILABLE 结构诊断时的整体解析上限


def _iter_episode_texts(path: Path):
    """流式产出 episodes[] 中每个 episode 对象的 JSON 文本。"""
    with path.open("r", encoding="utf-8") as handle:
        head = ""
        match = None
        while match is None:
            chunk = handle.read(_CHUNK)
            if not chunk:
                raise ValueError("未找到 episodes 字段")
            head += chunk
            match = _KEY_RE.search(head)
        pending = head[match.end():]

        state = "seek_arr"  # seek_arr → items
        depth = 0
        in_str = False
        escape = False
        started = False
        current: list[str] = []
        done = False

        while not done:
            for ch in pending:
                if state == "seek_arr":
                    if ch == "[":
                        state = "items"
                    elif not ch.isspace():
                        raise ValueError(f"episodes 不是数组（首字符 {ch!r}）")
                    continue
                if in_str:
                    if started:
                        current.append(ch)
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                    if started:
                        current.append(ch)
                    continue
                if ch == "{":
                    if depth == 0:
                        started = True
                    if started:
                        current.append(ch)
                    depth += 1
                    continue
                if ch == "}":
                    depth -= 1
                    if started:
                        current.append(ch)
                    if started and depth == 0:
                        yield "".join(current)
                        current = []
                        started = False
                    continue
                if depth > 0:
                    current.append(ch)
                    continue
                if ch == "]":
                    done = True
                    break
            else:
                pending = handle.read(_CHUNK)
                if not pending:
                    done = True
                continue
            break


def _structure_report(path: Path, reason: str) -> dict:
    """NOT_AVAILABLE 时的结构诊断（大文件只报可观察信息）。"""
    structure: dict = {"reason": reason}
    try:
        size = path.stat().st_size
    except OSError:
        structure["file"] = "不存在或不可读"
        return structure
    structure["file_size_bytes"] = size
    if size <= _WHOLE_FILE_LIMIT:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            structure["parse_error"] = str(exc)
            return structure
        if isinstance(data, dict):
            structure["top_level_keys"] = sorted(data.keys())
            episodes = data.get("episodes")
            if isinstance(episodes, list) and episodes and isinstance(episodes[0], dict):
                records = episodes[0].get("records")
                if isinstance(records, list) and records and isinstance(records[0], dict):
                    structure["sample_record_keys"] = sorted(records[0].keys())
    else:
        structure["note"] = "文件过大，跳过整体解析"
    return structure


def run_probe(
    forensics_json: str | Path,
    out: str | Path,
    *,
    sample_rate: float = 0.01,
    tol: float = 1e-6,
    quiet: bool = False,
) -> dict:
    """执行探针并把结果写入 out；返回输出 payload。"""
    source_path = Path(forensics_json)
    out_path = Path(out)

    n_episodes = 0
    n_records = 0
    n_decision_true = 0
    n_with_plan = 0
    n_missing_fields = 0
    n_checked = 0
    violations = 0
    max_abs_delta = 0.0
    finite_plans = 0
    plan_shape_bad = 0
    rng = random.Random(0)

    reason: str | None = None
    try:
        for episode_text in _iter_episode_texts(source_path):
            try:
                episode = json.loads(episode_text)
            except json.JSONDecodeError:
                continue
            n_episodes += 1
            records = episode.get("records") if isinstance(episode, dict) else None
            if not isinstance(records, list):
                continue
            for record in records:
                if not isinstance(record, dict):
                    continue
                n_records += 1
                if record.get("decision") is not True:
                    continue
                n_decision_true += 1
                plan = record.get("plan")
                mu = record.get("mu")
                if plan is None or mu is None:
                    n_missing_fields += 1
                    if rng.random() < sample_rate:
                        n_checked += 1
                        violations += 1
                    continue
                n_with_plan += 1

                if isinstance(plan, list) and plan and all(isinstance(p, list) for p in plan):
                    finite = all(
                        isinstance(v, (int, float)) and math.isfinite(v) for p in plan for v in p
                    )
                else:
                    finite = False
                    plan_shape_bad += 1
                if finite:
                    finite_plans += 1

                if rng.random() < sample_rate:
                    n_checked += 1
                    first = plan[0] if isinstance(plan, list) and plan else None
                    if (
                        not isinstance(first, (list, tuple))
                        or not isinstance(mu, (list, tuple))
                        or len(first) != len(mu)
                    ):
                        violations += 1
                        continue
                    try:
                        delta = max(
                            (abs(float(a) - float(b)) for a, b in zip(first, mu)), default=0.0
                        )
                    except (TypeError, ValueError):
                        violations += 1
                        continue
                    max_abs_delta = max(max_abs_delta, delta)
                    if delta > tol:
                        violations += 1
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        reason = str(exc)

    if reason is None and n_episodes == 0:
        reason = "未发现 episodes[] 数组"
    if reason is None and n_decision_true == 0:
        reason = "records[] 中无 decision=true 记录"
    if reason is None and n_with_plan == 0:
        reason = "decision=true 记录缺 plan/mu 字段"

    if reason is not None:
        payload: dict = {
            "status": "NOT_AVAILABLE",
            "n_checked": n_checked,
            "violations": None,
            "tol": tol,
            "source": str(forensics_json),
            "sample_rate": sample_rate,
            "n_episodes": n_episodes,
            "n_records": n_records,
            "n_decision_true": n_decision_true,
            "structure": _structure_report(source_path, reason),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
    else:
        payload = {
            "status": "PASS" if violations == 0 else "FAIL",
            "n_checked": n_checked,
            "violations": violations,
            "tol": tol,
            "source": str(forensics_json),
            "sample_rate": sample_rate,
            "n_episodes": n_episodes,
            "n_records": n_records,
            "n_decision_true": n_decision_true,
            "n_with_plan": n_with_plan,
            "n_missing_fields": n_missing_fields,
            "finite_plan_ratio": (finite_plans / n_with_plan) if n_with_plan else None,
            "plan_shape_bad": plan_shape_bad,
            "max_abs_delta": max_abs_delta,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if n_checked == 0:
            payload["warning"] = "n_checked == 0（抽样未命中，判定为 vacuous PASS）"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not quiet:
        print(
            f"[probe] plan_action_contract: {payload['status']} "
            f"n_checked={payload['n_checked']} violations={payload['violations']} "
            f"n_decision_true={payload['n_decision_true']} tol={tol} -> {out_path}"
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="探针：plan/action 契约（plan[0] ≈ mu）")
    parser.add_argument("--forensics-json", required=True, help="forensics JSON（episodes[].records[]）")
    parser.add_argument("--sample-rate", type=float, default=0.01, help="decision=true 步抽样率")
    parser.add_argument("--tol", type=float, default=1e-6, help="逐维 |Δ| 容差")
    parser.add_argument(
        "--out",
        default="runs/p1_0/probe_plan_action_contract.json",
        help="输出 JSON 路径（目录不存在则创建）",
    )
    args = parser.parse_args(argv)

    payload = run_probe(
        args.forensics_json, args.out, sample_rate=args.sample_rate, tol=args.tol
    )
    return {"PASS": 0, "FAIL": 1, "NOT_AVAILABLE": 2}.get(payload["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
