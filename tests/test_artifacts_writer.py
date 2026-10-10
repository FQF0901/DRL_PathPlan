"""P1-0C 统一产物补齐 + 契约探针 离线单测。

覆盖：artifacts.py 九项产物生成/关键键/不覆盖/幂等；checkpoint_load_report 的
PASS/FAIL/NOT_AVAILABLE 与 health 聚合；probe_load_contract 的 PASS/FAIL/NA；
probe_plan_action_contract 的 PASS/FAIL/NA（含结构不符诊断）。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from tools.controls import artifacts, probe_load_contract, probe_plan_action_contract


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_run(tmp_path: Path, load_kind: str = "pass", *, with_manifest: bool = True) -> Path:
    """伪造 run dir：manifest.txt + config/model 快照 + metrics.json + 假 ckpt。"""
    run = tmp_path / "runs" / "BTC20260101-0000_fake"
    run.mkdir(parents=True)
    ckpt = tmp_path / "fake_ckpt.pt"
    ckpt.write_bytes(b"fake-weights-0123456789")
    ckpt_sha = _sha256(ckpt.read_bytes())

    if with_manifest:
        (run / "manifest.txt").write_text(
            "\n".join(
                [
                    "created_at: 2026-01-01T00:00:00+08:00",
                    "stage: eval",
                    "git: deadbeefdeadbeefdeadbeefdeadbeefdeadbeef @main dirty=0",
                    "config: config/default.yaml",
                    f"work_dir: {run}",
                    f"out: {run}",
                    f"argv: tools/test.py --policy ckpt --ckpt {ckpt} --tracker lqr",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    (run / "config.snapshot.yaml").write_text("# cfg snapshot\nseed: 0\n", encoding="utf-8")
    (run / "model.snapshot.yaml").write_text("model:\n  hidden: 64\n", encoding="utf-8")

    meta: dict = {"ckpt_sha256": ckpt_sha}
    if load_kind == "pass":
        meta["ckpt_load"] = {
            "ckpt": str(ckpt),
            "sha256": ckpt_sha,
            "device": "cpu",
            "allow_partial_load": False,
            "loaded_keys": 168,
            "model_keys": 168,
            "ckpt_keys": 168,
            "counts": {"missing": 0, "unexpected": 0, "shape_mismatch": 0},
            "missing": [],
            "unexpected": [],
            "shape_mismatch": [],
        }
    elif load_kind == "fail":
        meta["ckpt_load"] = {
            "ckpt": str(ckpt),
            "sha256": ckpt_sha,
            "device": "cpu",
            "allow_partial_load": False,
            "loaded_keys": 166,
            "model_keys": 168,
            "ckpt_keys": 166,
            "counts": {"missing": 2, "unexpected": 0, "shape_mismatch": 0},
            "missing": ["a", "b"],
            "unexpected": [],
            "shape_mismatch": [],
        }
    (run / "metrics.json").write_text(json.dumps({"meta": meta}), encoding="utf-8")
    return run


def _git(run: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(run), *args], capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _init_repo_with_dirty(run: Path) -> str:
    _git(run, "init", "-q")
    _git(run, "add", "manifest.txt")
    _git(run, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-q", "-m", "init")
    (run / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")
    return _git(run, "rev-parse", "HEAD")


def _snapshot(run: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(run.iterdir()) if p.is_file()}


def test_artifacts_generates_all_and_keys(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    head = _init_repo_with_dirty(run)
    ckpt = tmp_path / "fake_ckpt.pt"

    actions = artifacts.run_artifacts(run)
    created = {rec["artifact"] for rec in actions if rec["action"] == "created"}
    assert created == {
        "manifest.json",
        "resolved_config.yaml",
        "git_state.json",
        "checkpoint.sha256",
        "checkpoint_load_report.json",
        "probe_results.json",
        "decision.md",
        "stdout.log",
        "health_report.json",
    }

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["created_at"] == "2026-01-01T00:00:00+08:00"
    assert manifest["stage"] == "eval"
    assert manifest["config"] == "config/default.yaml"
    assert manifest["work_dir"] == str(run)
    assert manifest["out"] == str(run)
    assert "--ckpt" in manifest["argv"]

    resolved = (run / "resolved_config.yaml").read_text(encoding="utf-8")
    assert resolved.startswith("# resolved_config.yaml")
    assert "model.snapshot.yaml" in resolved.split("seed: 0")[0]
    assert resolved.endswith("# cfg snapshot\nseed: 0\n")

    state = json.loads((run / "git_state.json").read_text(encoding="utf-8"))
    assert state["commit"] == head
    assert any("dirty.txt" in line for line in state["dirty_files"])
    assert state["generated_at"]

    expected_sha = _sha256(ckpt.read_bytes())
    assert (run / "checkpoint.sha256").read_text(encoding="utf-8") == f"{expected_sha}  {ckpt}\n"

    report = json.loads((run / "checkpoint_load_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["loaded_ratio"] == 1.0
    assert report["counts"] == {"missing": 0, "unexpected": 0, "shape_mismatch": 0}
    assert report["ckpt"] == str(ckpt)
    assert report["sha256"] == expected_sha

    assert json.loads((run / "probe_results.json").read_text(encoding="utf-8")) == {"probes": {}}

    health = json.loads((run / "health_report.json").read_text(encoding="utf-8"))
    assert health["status"] == "PASS"
    assert any(c["name"] == "checkpoint_load_report" for c in health["checks"])
    assert health["generated_at"]

    decision = (run / "decision.md").read_text(encoding="utf-8")
    headings = [ln for ln in decision.splitlines() if ln.startswith("## ")]
    assert len(headings) == 9
    for name in (
        "hypothesis",
        "change",
        "result",
        "paired comparison",
        "failure composition",
        "probe status",
        "go",
        "no-go",
        "next allowed action",
    ):
        assert f"## {name}" in decision
    assert (run / "stdout.log").is_file()


def test_artifacts_idempotent_and_never_overwrites(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    artifacts.run_artifacts(run)
    first = _snapshot(run)

    # 第二次默认运行：零改动（含 health_report 默认不重写）
    artifacts.run_artifacts(run)
    assert _snapshot(run) == first

    # 哨兵内容：默认运行不得覆盖任何已有文件
    sentinels = {
        "stdout.log": b"SENTINEL-STDOUT\n",
        "decision.md": b"SENTINEL-DECISION\n",
        "manifest.json": b'{"sentinel": true}\n',
        "resolved_config.yaml": b"SENTINEL-CONFIG\n",
        "checkpoint.sha256": b"SENTINEL-SHA\n",
        "checkpoint_load_report.json": b'{"sentinel": true}\n',
        "probe_results.json": b'{"sentinel": true}\n',
        "health_report.json": b'{"sentinel": true}\n',
    }
    for name, blob in sentinels.items():
        (run / name).write_bytes(blob)

    actions = artifacts.run_artifacts(run)
    assert all(rec["action"] == "exists" for rec in actions)
    for name, blob in sentinels.items():
        assert (run / name).read_bytes() == blob

    # --force 只允许覆盖 health_report.json
    artifacts.run_artifacts(run, force=True)
    for name, blob in sentinels.items():
        if name == "health_report.json":
            continue
        assert (run / name).read_bytes() == blob
    health = json.loads((run / "health_report.json").read_text(encoding="utf-8"))
    assert health["status"] == "WARN"  # 哨兵 load report 不可解析 → NOT_AVAILABLE → WARN


def test_artifacts_without_manifest_skips_manifest_json(tmp_path: Path) -> None:
    run = _make_run(tmp_path, with_manifest=False)
    artifacts.run_artifacts(run)
    assert not (run / "manifest.json").exists()
    assert not (run / "checkpoint.sha256").exists() or (run / "checkpoint.sha256").stat().st_size == 0
    for name in ("probe_results.json", "decision.md", "stdout.log", "health_report.json"):
        assert (run / name).exists()


def test_load_report_fail_and_not_available(tmp_path: Path) -> None:
    fail_run = _make_run(tmp_path / "a", load_kind="fail")
    artifacts.run_artifacts(fail_run)
    report = json.loads((fail_run / "checkpoint_load_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["counts"]["missing"] == 2
    assert report["loaded_ratio"] == 166 / 168
    health = json.loads((fail_run / "health_report.json").read_text(encoding="utf-8"))
    assert health["status"] == "FAIL"

    na_run = _make_run(tmp_path / "b", load_kind="none")
    artifacts.run_artifacts(na_run)
    report = json.loads((na_run / "checkpoint_load_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "NOT_AVAILABLE"
    health = json.loads((na_run / "health_report.json").read_text(encoding="utf-8"))
    assert health["status"] == "WARN"


def test_health_aggregates_probe_fail(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    artifacts.run_artifacts(run)
    probes = json.loads((run / "probe_results.json").read_text(encoding="utf-8"))
    probes["probes"]["some_probe"] = {"status": "FAIL"}
    (run / "probe_results.json").write_text(json.dumps(probes), encoding="utf-8")
    artifacts.run_artifacts(run, force=True)
    health = json.loads((run / "health_report.json").read_text(encoding="utf-8"))
    assert health["status"] == "FAIL"
    assert any(c["name"] == "probe:some_probe" and c["status"] == "FAIL" for c in health["checks"])


def test_probe_load_contract_pass_fail_na(tmp_path: Path, capsys) -> None:
    run = _make_run(tmp_path / "pass")
    (run / "probe_results.json").write_text(
        json.dumps({"probes": {"other": {"status": "PASS"}}}), encoding="utf-8"
    )
    payload = probe_load_contract.run_probe(run)
    assert payload["status"] == "PASS"
    assert payload["counts"] == {"missing": 0, "unexpected": 0, "shape_mismatch": 0}
    assert payload["loaded_ratio"] == 1.0
    stored = json.loads((run / "probe_results.json").read_text(encoding="utf-8"))
    assert stored["probes"]["checkpoint_load_contract"] == payload
    assert stored["probes"]["other"] == {"status": "PASS"}  # 不覆盖其他探针
    assert "checkpoint_load_contract: PASS" in capsys.readouterr().out
    assert probe_load_contract.main(["--run-dir", str(run)]) == 0

    fail_run = _make_run(tmp_path / "fail", load_kind="fail")
    payload = probe_load_contract.run_probe(fail_run, quiet=True)
    assert payload["status"] == "FAIL"
    assert probe_load_contract.main(["--run-dir", str(fail_run)]) == 1

    na_run = _make_run(tmp_path / "na", load_kind="none")
    payload = probe_load_contract.run_probe(na_run, quiet=True)
    assert payload["status"] == "NOT_AVAILABLE"
    assert probe_load_contract.main(["--run-dir", str(na_run)]) == 2


def _forensics(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "forensics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"mode": "d2", "episodes": [{"id": 1, "records": records}]}), encoding="utf-8"
    )
    return path


def test_plan_action_probe_pass_fail_na(tmp_path: Path) -> None:
    good_records = [
        {"step": 0, "decision": True, "mu": [1.0, 2.0], "plan": [[1.0, 2.0], [3.0, 4.0]]},
        {"step": 1, "decision": False},
        {"step": 2, "decision": True, "mu": [0.5, -0.5], "plan": [[0.5, -0.5], [0.6, -0.6]]},
    ]
    out = tmp_path / "nested" / "probe.json"
    payload = probe_plan_action_contract.run_probe(
        _forensics(tmp_path / "ok", good_records), out, sample_rate=1.0, quiet=True
    )
    assert payload["status"] == "PASS"
    assert payload["n_checked"] == 2
    assert payload["violations"] == 0
    assert payload["finite_plan_ratio"] == 1.0
    assert out.is_file()  # 目标目录自动创建
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["status"] == "PASS"
    assert set(("status", "n_checked", "violations", "tol", "source")) <= set(stored)
    assert probe_plan_action_contract.main(
        [
            "--forensics-json",
            str(_forensics(tmp_path / "ok", good_records)),
            "--sample-rate",
            "1.0",
            "--out",
            str(tmp_path / "cli.json"),
        ]
    ) == 0

    bad_records = [
        {"step": 0, "decision": True, "mu": [1.0, 2.0], "plan": [[1.0, 2.001], [3.0, 4.0]]}
    ]
    payload = probe_plan_action_contract.run_probe(
        _forensics(tmp_path / "bad", bad_records), tmp_path / "bad.json", sample_rate=1.0, quiet=True
    )
    assert payload["status"] == "FAIL"
    assert payload["violations"] == 1
    assert payload["max_abs_delta"] > 1e-6

    na_records = [{"step": 0, "x": 1.0}]  # 无 decision/mu/plan
    payload = probe_plan_action_contract.run_probe(
        _forensics(tmp_path / "na", na_records), tmp_path / "na.json", sample_rate=1.0, quiet=True
    )
    assert payload["status"] == "NOT_AVAILABLE"
    assert payload["violations"] is None
    assert "structure" in payload
    assert payload["structure"]["reason"]
