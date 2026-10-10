"""P1-0C 恢复质量门单测（合成 fixture 每门 True/False + CLI）。"""

from __future__ import annotations

import json
from pathlib import Path

from tools.controls import recovery_quality_gates as gates


def _good_row(**overrides) -> dict:
    row = {
        "episode_id": 7,
        "step": 10,
        "dt": 0.5,
        "pose": [0.0, 0.0, 0.0],
        "recovery_traj": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        "recovery_traj_world": [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
        ],
        "action_chain": [[1.0, 0.0]] * 6,
        "margin_before": 0.1,
        "margin_after": 0.4,
        "rc_before": 0.5,
        "rc_after": 0.7,
        "takeover_off_road": False,
        "takeover_collision": False,
        "geometry_labels": [0.1, 0.2, 0.3],
    }
    row.update(overrides)
    return row


def test_good_row_usable() -> None:
    result = gates.evaluate_row(_good_row())
    assert result["usable"] is True
    assert result["reasons"] == []


def test_each_gate_failure_flagged() -> None:
    cases = [
        ({"takeover_off_road": True}, "post_takeover_failure"),
        ({"takeover_collision": True}, "post_takeover_failure"),
        ({"takeover_termination": "out_of_road"}, "post_takeover_failure"),
        ({"margin_after": 0.1}, "margin_no_improvement"),
        ({"margin_after": 0.05}, "margin_no_improvement"),
        ({"rc_after": 0.5}, "route_no_improvement"),
        (
            {"recovery_traj": [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [9.0, 0.0]]},
            "action_traj_inconsistent",
        ),
        ({"dt": 0.4}, "coordinate_time_contract"),
        ({"pose": [0.0, 0.0, float("nan")]}, "coordinate_time_contract"),
        (
            {"recovery_traj_world": [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0], [7.0, 0.0]]},
            "coordinate_time_contract",
        ),
        ({"geometry_labels": [float("nan")] * 3}, "geometry_labels_unavailable"),
        ({"geometry_labels": None}, "geometry_labels_unavailable"),
    ]
    for override, gate in cases:
        result = gates.evaluate_row(_good_row(**override))
        assert result["usable"] is False, override
        assert any(reason.startswith(gate) for reason in result["reasons"]), (override, result["reasons"])


def test_action_traj_tolerance_parameterized() -> None:
    traj = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.3, 0.0]]
    row = _good_row(recovery_traj=traj, recovery_traj_world=traj)
    assert gates.evaluate_row(row, gates.GateParams(action_traj_tol_m=0.5))["usable"] is True
    assert gates.evaluate_row(row, gates.GateParams(action_traj_tol_m=0.1))["usable"] is False


def test_alias_fields_supported() -> None:
    row = _good_row()
    row["min_signed_margin_before"] = row.pop("margin_before")
    row["min_signed_margin_after"] = row.pop("margin_after")
    row["route_completion_before"] = row.pop("rc_before")
    row["route_completion_after"] = row.pop("rc_after")
    row["action"] = row.pop("action_chain")
    row["labels"] = row.pop("geometry_labels")
    assert gates.evaluate_row(row)["usable"] is True


def test_sparse_row_reports_gaps() -> None:
    result = gates.evaluate_row({"episode_id": 1, "step": 0})
    assert result["usable"] is False
    joined = " ".join(result["reasons"])
    for expected in (
        "missing:takeover_outcome",
        "missing:margin_before/margin_after",
        "missing:action_chain",
        "missing:dt",
        "missing:geometry_labels",
    ):
        assert expected in joined


def test_cli_summary_and_rows_out(tmp_path: Path) -> None:
    good = _good_row()
    bad = _good_row(margin_after=0.1)
    samples = tmp_path / "samples.jsonl"
    samples.write_text(json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    out = tmp_path / "summary.json"
    rows_out = tmp_path / "rows.jsonl"

    assert gates.main(["--input", str(samples), "--out", str(out), "--rows-out", str(rows_out)]) == 1
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["status"] == "FAIL"
    assert summary["n_rows"] == 2
    assert summary["n_usable"] == 1
    assert summary["usable_ratio"] == 0.5
    assert summary["gate_fail_counts"]["margin_no_improvement"] == 1
    assert "margin_no_improvement:gain" in summary["reason_counts"]
    decisions = [json.loads(line) for line in rows_out.read_text(encoding="utf-8").splitlines()]
    assert decisions[0]["usable"] is True
    assert decisions[1]["usable"] is False

    # 全可用 → PASS / exit 0；JSON 数组输入同样支持
    all_good = tmp_path / "all_good.json"
    all_good.write_text(json.dumps([good]), encoding="utf-8")
    assert gates.main(["--input", str(all_good), "--out", str(tmp_path / "s2.json")]) == 0
    assert json.loads((tmp_path / "s2.json").read_text(encoding="utf-8"))["status"] == "PASS"


def test_cli_not_available(tmp_path: Path) -> None:
    assert gates.main(["--input", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path / "s.json")]) == 2
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert gates.main(["--input", str(empty), "--out", str(tmp_path / "s2.json")]) == 2
