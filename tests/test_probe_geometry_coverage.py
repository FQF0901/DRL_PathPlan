"""P1-0C 几何代理覆盖探针单测（合成 npz：LD 槽/实线/plan 覆盖）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.controls import probe_geometry_coverage as probe


def _make_npz(
    path: Path,
    *,
    n_frames: int = 3,
    valid_slots: int = 8,
    solid: bool = True,
    point_x: float = 5.0,
    plan_hit: bool = True,
) -> Path:
    ld = np.zeros((n_frames, 16, 7), dtype=np.float32)
    mask = np.zeros((n_frames, 16), dtype=np.float32)
    for slot in range(valid_slots):
        ld[:, slot, 0] = point_x + slot  # dx
        ld[:, slot, 1] = 0.0  # dy
        ld[:, slot, 5] = 2.0 if solid else 1.0  # left line type id
        mask[:, slot] = 1.0
    traj = np.zeros((n_frames, 6, 2), dtype=np.float32)
    traj[:, :, 0] = point_x if plan_hit else 100.0
    np.savez(
        path,
        ld=ld,
        ld_mask=mask,
        traj6=traj,
        train_weight=np.ones(n_frames, dtype=np.float32),
    )
    return path


def test_solid_id_derivation() -> None:
    assert set(probe.SOLID_LINE_IDS) == {2, 3, 6, 7}
    assert 8 not in probe.SOLID_LINE_IDS  # PASSING_DOUBLE_YELLOW 不计入实线


def test_geometry_coverage_pass(tmp_path: Path) -> None:
    dataset = _make_npz(tmp_path / "geom.npz")
    payload = probe.run_probe(dataset, tmp_path / "nested" / "geom.json", quiet=True)

    assert payload["status"] == "PASS"
    assert payload["n_frames"] == 3
    assert payload["ld_valid_ratio"] == 8 / 16
    assert payload["solid_available_ratio"] == 1.0
    assert payload["solid_slot_ratio"] == 1.0
    assert payload["plan_coverage_ratio"] == 1.0
    assert payload["plan_coverage_ratio_weight_pos"] == 1.0
    assert payload["near_field_coverage_ratio"] == 1.0
    assert payload["ld_point_distance_m"]["min"] == 5.0
    assert payload["ld_point_distance_m"]["median"] == 8.5  # 槽位 x=5..12
    assert (tmp_path / "nested" / "geom.json").is_file()

    assert probe.main(["--dataset", str(dataset), "--out", str(tmp_path / "cli.json")]) == 0


def test_geometry_coverage_fail_without_valid_slots(tmp_path: Path) -> None:
    dataset = _make_npz(tmp_path / "empty.npz", valid_slots=0)
    payload = probe.run_probe(dataset, tmp_path / "empty.json", quiet=True)

    assert payload["status"] == "FAIL"
    assert payload["ld_valid_ratio"] == 0.0
    assert payload["solid_available_ratio"] == 0.0
    assert payload["plan_coverage_ratio"] == 0.0
    assert payload["checks"]["ld_valid"] is False
    assert probe.main(["--dataset", str(dataset), "--out", str(tmp_path / "empty_cli.json")]) == 1


def test_geometry_coverage_not_available(tmp_path: Path) -> None:
    payload = probe.run_probe(tmp_path / "nope.npz", tmp_path / "na.json", quiet=True)
    assert payload["status"] == "NOT_AVAILABLE"
    assert probe.main(["--dataset", str(tmp_path / "nope.npz"), "--out", str(tmp_path / "na2.json")]) == 2

    # 缺字段（无 ld_mask）→ NOT_AVAILABLE
    partial = tmp_path / "partial.npz"
    np.savez(partial, ld=np.zeros((2, 16, 7), dtype=np.float32))
    payload = probe.run_probe(partial, tmp_path / "partial.json", quiet=True)
    assert payload["status"] == "NOT_AVAILABLE"
    assert "ld_mask" in payload["reason"]
