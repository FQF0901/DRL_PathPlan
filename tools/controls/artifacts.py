#!/usr/bin/env python3
"""统一产物补齐器（P1 执行手册 §7.10-B；正式实验目录 11 项统一产物）。

用法::

    tools/venv-python tools/controls/artifacts.py --run-dir <dir> [--force]

补齐产物（**绝不覆盖已存在文件**；``--force`` 仅允许重写 ``health_report.json``）::

    manifest.json                 manifest.txt 的 key:value 结构化
    resolved_config.yaml          config.snapshot.yaml 原文 + model.snapshot.yaml 说明头
    git_state.json                git rev-parse HEAD + status --porcelain
    checkpoint.sha256             manifest argv 中 --ckpt 的 sha256（找不到 → 空文件）
    checkpoint_load_report.json   metrics.json::meta.ckpt_load 提取（无 → NOT_AVAILABLE）
    probe_results.json            探针统一输出（不存在时 {"probes": {}}）
    health_report.json            探针 + load report 聚合（PASS/WARN/FAIL）
    decision.md                   九字段决策模板（不存在时）
    stdout.log                    占位文件（不存在时）

已有产物 metrics.json / episodes.csv 原样保持、不碰。
幂等：默认连续两次运行，第二次零改动（health_report 默认不重写）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

COUNTS_KEYS = ("missing", "unexpected", "shape_mismatch")

# decision.md 九字段（P1 执行手册 §7.10-B）
DECISION_FIELDS = (
    ("hypothesis", "假设：本 run 要验证什么、预期方向"),
    ("change", "变更：相对基线/起点改了什么（配置、数据、代码）"),
    ("result", "结果：关键数字（success/off-road/collision/rc/...）"),
    ("paired comparison", "配对比较：同集/同 seed/统计口径（McNemar/bootstrap）"),
    ("failure composition", "失败构成：plan/tracker/recovery/... 分类计数"),
    ("probe status", "探针状态：probe_results.json / health_report.json 结论"),
    ("go", "Go 条件：满足什么才继续"),
    ("no-go", "No-Go 条件：触发即停"),
    ("next allowed action", "下一步允许动作：no-go 时唯一允许的动作"),
)

_STDOUT_PLACEHOLDER = (
    "# stdout.log 占位（由 tools/controls/artifacts.py 创建）\n"
    "# 本 run 未接入统一 stdout tee；正式 run 应由驱动脚本把 stdout 写入此文件。\n"
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def _write_if_missing(path: Path, text: str) -> bool:
    if path.exists():
        return False
    path.write_text(text, encoding="utf-8")
    return True


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def parse_manifest_text(text: str) -> dict[str, str]:
    """解析 manifest.txt 的 ``key: value`` 行（值含 ``:`` 只切第一个）。"""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_.-]*)\s*:\s*(.*)$", line)
        if match:
            fields[match.group(1)] = match.group(2)
    return fields


def manifest_argv_tokens(text: str) -> list[str]:
    argv = parse_manifest_text(text).get("argv", "")
    try:
        return shlex.split(argv)
    except ValueError:
        return argv.split()


def extract_ckpt(argv_tokens: list[str]) -> str | None:
    """从 argv 提取 ``--ckpt <path>`` / ``--ckpt=<path>``。"""
    for i, token in enumerate(argv_tokens):
        if token == "--ckpt" and i + 1 < len(argv_tokens):
            return argv_tokens[i + 1]
        if token.startswith("--ckpt="):
            return token.split("=", 1)[1]
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(run_dir: Path) -> dict[str, Any]:
    """在 run_dir 所在仓库执行 rev-parse HEAD + status --porcelain。"""

    def _git(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(run_dir), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout

    commit_out = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain")
    return {
        "commit": commit_out.strip() if commit_out else None,
        "dirty_files": [line for line in (porcelain or "").splitlines() if line.strip()],
        "generated_at": _now(),
    }


def build_load_report(metrics: Any) -> dict[str, Any]:
    """metrics.json::meta.ckpt_load → checkpoint_load_report.json 结构。"""
    meta = metrics.get("meta") if isinstance(metrics, dict) else None
    load = meta.get("ckpt_load") if isinstance(meta, dict) else None
    if not isinstance(load, dict):
        return {"status": "NOT_AVAILABLE", "reason": "metrics.json::meta.ckpt_load 缺失"}
    counts_raw = load.get("counts") if isinstance(load.get("counts"), dict) else {}
    counts = {key: int(counts_raw.get(key) or 0) for key in COUNTS_KEYS}
    loaded = int(load.get("loaded_keys") or 0)
    model = int(load.get("model_keys") or 0)
    ratio = (loaded / model) if model else None
    ok = all(value == 0 for value in counts.values()) and loaded == model
    return {
        "status": "PASS" if ok else "FAIL",
        "loaded_ratio": ratio,
        "counts": counts,
        "ckpt": load.get("ckpt"),
        "sha256": load.get("sha256"),
    }


def build_health_report(run_dir: Path) -> dict[str, Any]:
    """聚合 probe_results.json 的探针 + checkpoint_load_report.json。"""
    probe_results = _load_json(run_dir / "probe_results.json")
    probes = probe_results.get("probes") if isinstance(probe_results, dict) else None
    probes = probes if isinstance(probes, dict) else {}
    load_report = _load_json(run_dir / "checkpoint_load_report.json")

    checks: list[dict[str, str]] = []
    for name in sorted(probes):
        entry = probes[name]
        status = entry.get("status") if isinstance(entry, dict) else None
        checks.append(
            {"name": f"probe:{name}", "status": status or "NOT_AVAILABLE", "source": "probe_results.json"}
        )
    load_status = load_report.get("status") if isinstance(load_report, dict) else None
    checks.append(
        {
            "name": "checkpoint_load_report",
            "status": load_status or "NOT_AVAILABLE",
            "source": "checkpoint_load_report.json",
        }
    )

    statuses = [check["status"] for check in checks]
    if any(status == "FAIL" for status in statuses):
        overall = "FAIL"
    elif all(status == "PASS" for status in statuses):
        overall = "PASS"
    else:
        overall = "WARN"
    return {"status": overall, "generated_at": _now(), "checks": checks}


def _decision_template(run_dir: Path) -> str:
    lines = [
        f"# Decision Record — {run_dir.name}",
        "",
        "> 模板由 tools/controls/artifacts.py 生成（P1 执行手册 §7.10-B 统一产物）；",
        "> 每个字段保留 TODO，实验负责人填写后不要删除字段名。",
        "",
    ]
    for name, hint in DECISION_FIELDS:
        lines += [f"## {name}", f"<!-- {hint} -->", "TODO", ""]
    return "\n".join(lines)


def run_artifacts(run_dir: str | Path, force: bool = False) -> list[dict[str, Any]]:
    """补齐统一产物；返回逐项动作清单。"""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir 不存在: {run_dir}")
    actions: list[dict[str, Any]] = []

    def _add(artifact: str, action: str, detail: str | None = None) -> None:
        record: dict[str, Any] = {"artifact": artifact, "action": action}
        if detail:
            record["detail"] = detail
        actions.append(record)

    manifest_txt = run_dir / "manifest.txt"
    manifest_text = manifest_txt.read_text(encoding="utf-8") if manifest_txt.is_file() else None

    # 1) manifest.json
    if (run_dir / "manifest.json").exists():
        _add("manifest.json", "exists")
    elif manifest_text is not None:
        fields = parse_manifest_text(manifest_text)
        (run_dir / "manifest.json").write_text(_dump(fields), encoding="utf-8")
        _add("manifest.json", "created", f"keys={','.join(sorted(fields))}")
    else:
        _warn(f"缺 manifest.txt，跳过 manifest.json: {run_dir}")
        _add("manifest.json", "skipped", "manifest.txt 缺失")

    # 2) resolved_config.yaml
    config_snapshot = run_dir / "config.snapshot.yaml"
    if (run_dir / "resolved_config.yaml").exists():
        _add("resolved_config.yaml", "exists")
    elif config_snapshot.is_file():
        model_snapshot = run_dir / "model.snapshot.yaml"
        header = [
            "# resolved_config.yaml（统一产物，由 tools/controls/artifacts.py 生成）",
            "# 正文 = config.snapshot.yaml 原文（逐字保留，未合并、未解析）。",
            "# model.snapshot.yaml 为模型侧补充快照（网络结构/训练超参），与本文件合并阅读"
            "即完整 resolved config；本文件不重复其内容。",
            f"# model.snapshot.yaml: {'present' if model_snapshot.is_file() else 'absent'}",
            "",
        ]
        text = "\n".join(header) + config_snapshot.read_text(encoding="utf-8")
        (run_dir / "resolved_config.yaml").write_text(text, encoding="utf-8")
        _add(
            "resolved_config.yaml",
            "created",
            f"model.snapshot.yaml={'present' if model_snapshot.is_file() else 'absent'}",
        )
    else:
        _warn(f"缺 config.snapshot.yaml，跳过 resolved_config.yaml: {run_dir}")
        _add("resolved_config.yaml", "skipped", "config.snapshot.yaml 缺失")

    # 3) git_state.json
    if (run_dir / "git_state.json").exists():
        _add("git_state.json", "exists")
    else:
        state = git_state(run_dir)
        (run_dir / "git_state.json").write_text(_dump(state), encoding="utf-8")
        _add(
            "git_state.json",
            "created",
            f"commit={state['commit']} dirty={len(state['dirty_files'])}",
        )

    # 4) checkpoint.sha256
    if (run_dir / "checkpoint.sha256").exists():
        _add("checkpoint.sha256", "exists")
    else:
        ckpt = extract_ckpt(manifest_argv_tokens(manifest_text)) if manifest_text is not None else None
        if ckpt and Path(ckpt).is_file():
            (run_dir / "checkpoint.sha256").write_text(
                f"{sha256_file(Path(ckpt))}  {ckpt}\n", encoding="utf-8"
            )
            _add("checkpoint.sha256", "created", f"ckpt={ckpt}")
        else:
            (run_dir / "checkpoint.sha256").write_text("", encoding="utf-8")
            _add("checkpoint.sha256", "created", "未找到 --ckpt 或文件缺失 → 空文件")

    # 5) checkpoint_load_report.json
    if (run_dir / "checkpoint_load_report.json").exists():
        _add("checkpoint_load_report.json", "exists")
    else:
        report = build_load_report(_load_json(run_dir / "metrics.json"))
        (run_dir / "checkpoint_load_report.json").write_text(_dump(report), encoding="utf-8")
        _add("checkpoint_load_report.json", "created", f"status={report['status']}")

    # 6) probe_results.json
    if _write_if_missing(run_dir / "probe_results.json", _dump({"probes": {}})):
        _add("probe_results.json", "created", 'probes={}')
    else:
        _add("probe_results.json", "exists")

    # 7) decision.md
    if _write_if_missing(run_dir / "decision.md", _decision_template(run_dir)):
        _add("decision.md", "created", f"fields={len(DECISION_FIELDS)}")
    else:
        _add("decision.md", "exists")

    # 8) stdout.log
    if _write_if_missing(run_dir / "stdout.log", _STDOUT_PLACEHOLDER):
        _add("stdout.log", "created", "占位")
    else:
        _add("stdout.log", "exists")

    # 9) health_report.json（--force 仅允许覆盖此项）
    health_path = run_dir / "health_report.json"
    health_existed = health_path.exists()
    if health_existed and not force:
        _add("health_report.json", "exists", "--force 才允许覆盖")
    else:
        report = build_health_report(run_dir)
        health_path.write_text(_dump(report), encoding="utf-8")
        _add(
            "health_report.json",
            "overwritten" if health_existed else "created",
            f"status={report['status']}",
        )
    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="统一产物补齐（§7.10-B 11 项；默认不覆盖已存在文件）")
    parser.add_argument("--run-dir", required=True, help="正式实验 run 目录")
    parser.add_argument("--force", action="store_true", help="仅允许覆盖 health_report.json")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"[error] run-dir 不存在: {run_dir}", file=sys.stderr)
        return 2

    actions = run_artifacts(run_dir, force=args.force)
    print(f"[artifacts] run-dir: {run_dir}")
    for record in actions:
        detail = f"  ({record['detail']})" if record.get("detail") else ""
        print(f"  {record['action']:<10} {record['artifact']}{detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
