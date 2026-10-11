"""P1-1 分片合并器离线单测（合成分片目录 → merge → 断言拼接/split/数据门）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tests.test_collect_recovery import _fake_args, _synthetic_row
from tools.controls import collect_recovery as cr
from tools.controls import merge_recovery_shards as merge


def _episode(episode_id: int, pool: str, *, n_rows: int, n_usable: int, geometry: str = "merge") -> dict:
    return {
        "id": episode_id,
        "pool": pool,
        "geometry": geometry,
        "status": "collected",
        "windows": [
            {
                "class": "correction",
                "takeover_step": 10,
                "status": "collected",
                "determinism": {"accepted": "strict"},
                "n_rows": n_rows,
                "n_usable": n_usable,
            }
        ],
        "n_rows": n_rows,
        "n_usable": n_usable,
    }


def _make_shard(
    tmp_path: Path,
    name: str,
    episode_ids: list[int],
    *,
    pool: str = "eval500",
    shard: tuple[int, int] = (0, 1),
    geometry: str = "merge",
) -> Path:
    rows = []
    episodes = []
    for index, episode_id in enumerate(episode_ids):
        usable = index != len(episode_ids) - 1  # 最后一集放一行不可用，制造通过率 <1
        row = _synthetic_row(usable=usable, window_class="correction")
        row["episode_id"] = episode_id
        row["pool"] = pool
        row["geometry"] = geometry
        row["road_class"] = geometry
        rows.append(row)
        episodes.append(
            _episode(episode_id, pool, n_rows=1, n_usable=int(usable), geometry=geometry)
        )
    args = _fake_args(tmp_path)
    args.shard = shard
    pools_info = [{"name": pool, "path": "spec.json", "sha256": "x", "n_specs": 500, "selected": len(episode_ids)}]
    shard_dir = tmp_path / name
    cr._write_outputs(shard_dir, rows, episodes, args, pools_info)
    return shard_dir


def test_merge_two_shards_concat_split_and_gate(tmp_path: Path) -> None:
    shard_a = _make_shard(tmp_path, "shard0", [1, 2, 3], shard=(0, 2))
    shard_b = _make_shard(tmp_path, "shard1", [4, 5, 6], shard=(1, 2))
    out = tmp_path / "merged"

    summary = merge.merge_shards([shard_b, shard_a], out)  # 输入乱序 → 按名排序合并
    for name in ("manifest.json", "samples.npz", "schema.json", "quality_report.json",
                 "split_manifest.json", "data_gate_report.json"):
        assert (out / name).is_file(), name

    data = np.load(out / "samples.npz", allow_pickle=False)
    assert data["episode_id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert data["usable"].tolist() == [True, True, False, True, True, False]
    assert data["expert_recovery_traj"].shape == (6, 6, 2)

    split = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["no_leakage"] is True
    assert set(split["train_episodes"]) & set(split["val_episodes"]) == set()
    assert len(split["train_episodes"]) + len(split["val_episodes"]) == 6
    assert split["train"]["rows"] + split["val"]["rows"] == 6
    assert split["train"]["usable_rows"] + split["val"]["usable_rows"] == 4

    gate = json.loads((out / "data_gate_report.json").read_text(encoding="utf-8"))
    assert gate["status"] == "FAIL"  # 6 行远未达 10k 门
    checks = {check["name"]: check for check in gate["checks"]}
    assert checks["usable_rows"]["observed"] == 4
    assert checks["usable_rows"]["pass"] is False
    assert checks["independent_episodes"]["observed"] == 4  # 含有效行的 episode（保守口径）
    assert gate["observed"]["episodes_collected"] == 6
    assert gate["observed"]["episodes_with_usable_rows"] == 4
    assert checks["max_rows_per_episode"]["observed"] == 1
    assert checks["max_rows_per_episode"]["pass"] is True
    assert abs(checks["quality_pass_rate"]["observed"] - 4 / 6) < 1e-3
    assert checks["class_prevention"]["pass"] is False
    assert checks["class_correction"]["observed"] == 4
    assert "hard_merge_rows" in checks and "hard_merge_episodes" in checks
    assert gate["observed"]["class_usable_counts"]["correction"] == 4

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_shards"] == 2
    assert [source["name"] for source in manifest["sources"]] == ["shard0", "shard1"]
    assert all(source["samples_sha256"] for source in manifest["sources"])
    assert manifest["counts"]["rows_total"] == 6
    assert manifest["counts"]["rows_usable"] == 4
    assert manifest["counts"]["episodes_collected"] == 6
    assert manifest["pools"][0]["selected"] == 6
    assert manifest["data_gate"]["status"] == "FAIL"
    assert manifest["divergence"]["strict"] == 6  # 每片 3 个窗口 × 2 片

    quality = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert quality["n_rows"] == 6 and quality["n_usable"] == 4
    assert quality["per_class"]["correction"] == {"rows": 6, "usable": 4}

    assert summary["data_gate"]["status"] == "FAIL"
    # CLI：数据门未过 → exit 1
    assert merge.main(["--shards", str(shard_a), str(shard_b), "--out", str(tmp_path / "merged2")]) == 1


def test_merge_rejects_episode_overlap(tmp_path: Path) -> None:
    shard_a = _make_shard(tmp_path, "shard0", [1, 2, 3])
    shard_b = _make_shard(tmp_path, "shard1", [3, 4, 5])  # id=3 重叠
    assert merge.main(["--shards", str(shard_a), str(shard_b), "--out", str(tmp_path / "bad")]) == 2
    assert not (tmp_path / "bad" / "manifest.json").exists()


def test_merge_rejects_schema_mismatch(tmp_path: Path) -> None:
    shard_a = _make_shard(tmp_path, "shard0", [1, 2])
    shard_b = _make_shard(tmp_path, "shard1", [3, 4])
    schema_path = shard_b / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["schema_version"] = 999
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    assert merge.main(["--shards", str(shard_a), str(shard_b), "--out", str(tmp_path / "bad")]) == 2


def test_merge_missing_shard_file(tmp_path: Path) -> None:
    shard_a = _make_shard(tmp_path, "shard0", [1, 2])
    (shard_a / "quality_report.json").unlink()
    assert merge.main(["--shards", str(shard_a), "--out", str(tmp_path / "bad")]) == 2


def test_data_gate_report_hard_types_and_pass(tmp_path: Path) -> None:
    rows = []
    episodes = []
    for index in range(4):
        row = _synthetic_row(window_class="prevention")
        row["episode_id"] = index
        row["geometry"] = "tollgate" if index < 2 else "curve"
        row["road_class"] = row["geometry"]
        rows.append(row)
        episodes.append(_episode(index, "eval500", n_rows=1, n_usable=1, geometry=row["geometry"]))
    report = merge.build_data_gate_report(
        [
            {"usable": row["usable"], "class": row["class"], "pool": row["pool"],
             "episode_id": row["episode_id"], "geometry": row["geometry"], "road_class": row["road_class"]}
            for row in rows
        ],
        episodes,
        targets={"prevention": 4, "correction": 0, "late": 0},
        min_rows=4,
        min_episodes=4,
        hard_types=("tollgate",),
        hard_min_rows=2,
        hard_min_episodes=2,
    )
    assert report["status"] == "PASS"
    assert report["observed"]["hard_types"]["tollgate"] == {"usable_rows": 2, "episodes": 2}
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["hard_tollgate_rows"]["pass"] is True
    assert checks["hard_tollgate_episodes"]["pass"] is True
    assert checks["class_prevention"]["pass"] is True
    assert checks["quality_pass_rate"]["pass"] is True
