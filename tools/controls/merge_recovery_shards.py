#!/usr/bin/env python3
"""合并 recovery 分片 → 正式数据集目录 + 数据门核验（P1-1）。

用法::

    tools/venv-python tools/controls/merge_recovery_shards.py \\
        --shards runs/p1_0/recovery_formal_shard0 ... runs/p1_0/recovery_formal_shard5 \\
        --out datasets/recovery_v1

行为
----
- 校验各片 ``schema.json`` 一致、npz 键/形状/类型一致、episode ``(pool,id)`` 互斥；
- 按分片名排序拼接 ``samples.npz``；合并 ``manifest.json``（记录分片来源与全部参数）；
- 重算全局 ``split_manifest.json``（episode 80/20、无泄漏）与 ``quality_report.json``；
- 输出 ``data_gate_report.json``（§7.10-A5 数据门逐项核验：有效行 ≥10k、独立 episode ≥500、
  单 episode ≤20 行、质量通过率 ≥80%、三档 ≥4000/2500/1000、困难路型每类 ≥300 行且
  ≥30 eps——困难路型按 spec geometry / records road_class 汇总）；
- ``--out`` 在 ``datasets/`` 下时补数据集卡模板（``docs/recovery_v1_dataset_card.md``，
  不存在才写）。

退出码：0 = 合并成功且数据门全过；1 = 合并成功但数据门未过；2 = 合并失败（不一致/重叠等）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.controls import collect_recovery as cr  # noqa: E402

REQUIRED_FILES = (
    "manifest.json",
    "samples.npz",
    "schema.json",
    "quality_report.json",
    "split_manifest.json",
)


class MergeError(RuntimeError):
    """分片不一致 / 重叠 / 缺文件等致命错误。"""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_shard(shard_dir: Path) -> dict:
    if not shard_dir.is_dir():
        raise MergeError(f"分片目录不存在: {shard_dir}")
    missing = [name for name in REQUIRED_FILES if not (shard_dir / name).is_file()]
    if missing:
        raise MergeError(f"分片缺文件 {missing}: {shard_dir}")
    return {
        "dir": shard_dir,
        "name": shard_dir.name,
        "manifest": json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8")),
        "schema": json.loads((shard_dir / "schema.json").read_text(encoding="utf-8")),
        "quality": json.loads((shard_dir / "quality_report.json").read_text(encoding="utf-8")),
        "arrays": dict(np.load(shard_dir / "samples.npz", allow_pickle=False)),
        "samples_sha256": cr._sha256_file(shard_dir / "samples.npz"),
    }


def _check_schema_consistency(shards: list[dict]) -> None:
    canonical = json.dumps(shards[0]["schema"], ensure_ascii=False, sort_keys=True)
    for shard in shards[1:]:
        if json.dumps(shard["schema"], ensure_ascii=False, sort_keys=True) != canonical:
            raise MergeError(f"schema.json 不一致: {shard['dir']} vs {shards[0]['dir']}")
    first_keys = set(shards[0]["arrays"])
    for shard in shards[1:]:
        if set(shard["arrays"]) != first_keys:
            raise MergeError(f"npz 键集不一致: {shard['dir']}")
    for key in sorted(first_keys):
        reference = shards[0]["arrays"][key]
        for shard in shards[1:]:
            array = shard["arrays"][key]
            if array.dtype != reference.dtype or array.shape[1:] != reference.shape[1:]:
                raise MergeError(
                    f"npz 数组 {key} 形状/类型不一致: {shard['dir']} "
                    f"{array.shape}/{array.dtype} vs {reference.shape}/{reference.dtype}"
                )
    pools_reference = [
        (pool.get("name"), pool.get("n_specs")) for pool in shards[0]["manifest"].get("pools", [])
    ]
    for shard in shards[1:]:
        pools_current = [
            (pool.get("name"), pool.get("n_specs")) for pool in shard["manifest"].get("pools", [])
        ]
        if pools_current != pools_reference:
            raise MergeError(f"分片 pools 列表不一致: {shard['dir']}")


def _episode_keys(shard: dict) -> set[str]:
    keys = {
        f"{episode.get('pool')}:{episode['id']}"
        for episode in shard["manifest"].get("episodes", [])
        if episode.get("status") == "collected"
    }
    npz_keys = {
        f"{pool}:{episode_id}"
        for pool, episode_id in zip(
            shard["arrays"]["pool"].tolist(), shard["arrays"]["episode_id"].tolist()
        )
    }
    if not npz_keys <= keys:
        raise MergeError(f"npz 行含 manifest 未列 episode: {shard['dir']}")
    return keys


def _check_episode_disjoint(shards: list[dict]) -> None:
    seen: dict[str, Path] = {}
    for shard in shards:
        for key in _episode_keys(shard):
            if key in seen:
                raise MergeError(f"episode 重叠: {key}（{seen[key]} 与 {shard['dir']}）")
            seen[key] = shard["dir"]


def _concat_arrays(shards: list[dict]) -> dict[str, np.ndarray]:
    keys = sorted(shards[0]["arrays"])
    return {
        key: np.concatenate([shard["arrays"][key] for shard in shards], axis=0) for key in keys
    }


def _minimal_rows(arrays: dict[str, np.ndarray]) -> list[dict]:
    return [
        {
            "usable": bool(arrays["usable"][index]),
            "class": str(arrays["class_label"][index]),
            "pool": str(arrays["pool"][index]),
            "episode_id": int(arrays["episode_id"][index]),
            "geometry": str(arrays["geometry"][index]),
            "road_class": str(arrays["road_class"][index]),
        }
        for index in range(len(arrays["usable"]))
    ]


def build_data_gate_report(
    rows: list[dict],
    episodes: list[dict],
    *,
    targets: dict[str, int] | None = None,
    min_rows: int = 10000,
    min_episodes: int = 500,
    max_rows_per_episode: int = 20,
    min_pass_rate: float = 0.80,
    hard_types: tuple[str, ...] = cr.HARD_ROAD_TYPES,
    hard_min_rows: int = cr.HARD_TYPE_MIN_ROWS,
    hard_min_episodes: int = cr.HARD_TYPE_MIN_EPISODES,
) -> dict:
    """§7.10-A5 数据门逐项核验（纯函数）。"""
    usable = [row for row in rows if row["usable"]]
    n_rows, n_usable = len(rows), len(usable)
    pass_rate = (n_usable / n_rows) if n_rows else 0.0
    episode_all = {
        f"{episode.get('pool')}:{episode['id']}"
        for episode in episodes
        if episode.get("status") == "collected"
    }
    episode_usable = {f"{row['pool']}:{row['episode_id']}" for row in usable}
    per_episode = Counter(f"{row['pool']}:{row['episode_id']}" for row in rows)
    max_per_episode = max(per_episode.values()) if per_episode else 0
    class_counts = {window_class: 0 for window_class in cr.WINDOW_CLASSES}
    for row in usable:
        class_counts[row["class"]] = class_counts.get(row["class"], 0) + 1

    checks: list[dict] = [
        {
            "name": "usable_rows",
            "requirement": f">= {min_rows}",
            "observed": n_usable,
            "pass": n_usable >= int(min_rows),
        },
        {
            "name": "independent_episodes",
            "requirement": f">= {min_episodes}",
            "observed": len(episode_usable),
            "pass": len(episode_usable) >= int(min_episodes),
        },
        {
            "name": "max_rows_per_episode",
            "requirement": f"<= {max_rows_per_episode}",
            "observed": max_per_episode,
            "pass": max_per_episode <= int(max_rows_per_episode),
        },
        {
            "name": "quality_pass_rate",
            "requirement": f">= {min_pass_rate:.2f}",
            "observed": round(pass_rate, 4),
            "pass": pass_rate >= float(min_pass_rate),
        },
    ]
    for window_class, target in (targets or {}).items():
        checks.append(
            {
                "name": f"class_{window_class}",
                "requirement": f">= {int(target)}",
                "observed": int(class_counts.get(window_class, 0)),
                "pass": int(class_counts.get(window_class, 0)) >= int(target),
            }
        )
    hard: dict[str, dict] = {}
    for hard_type in hard_types:
        type_rows = [
            row for row in usable if row["geometry"] == hard_type or row["road_class"] == hard_type
        ]
        type_episodes = {f"{row['pool']}:{row['episode_id']}" for row in type_rows}
        hard[hard_type] = {"usable_rows": len(type_rows), "episodes": len(type_episodes)}
        checks.append(
            {
                "name": f"hard_{hard_type}_rows",
                "requirement": f">= {hard_min_rows}",
                "observed": len(type_rows),
                "pass": len(type_rows) >= int(hard_min_rows),
            }
        )
        checks.append(
            {
                "name": f"hard_{hard_type}_episodes",
                "requirement": f">= {hard_min_episodes}",
                "observed": len(type_episodes),
                "pass": len(type_episodes) >= int(hard_min_episodes),
            }
        )
    return {
        "status": "PASS" if all(check["pass"] for check in checks) else "FAIL",
        "checks": checks,
        "observed": {
            "n_rows": n_rows,
            "n_usable": n_usable,
            "usable_ratio": round(pass_rate, 4),
            "episodes_collected": len(episode_all),
            "episodes_with_usable_rows": len(episode_usable),
            "max_rows_per_episode": max_per_episode,
            "class_usable_counts": class_counts,
            "hard_types": hard,
        },
        "targets": dict(targets or {}),
        "generated_at": _now(),
    }


def merge_shards(
    shard_dirs: list[str | Path], out_dir: str | Path, *, val_every: int = 5
) -> dict:
    """合并分片并写全部产物；返回摘要（含 data_gate）。"""
    shards = [_load_shard(Path(path)) for path in sorted(str(path) for path in shard_dirs)]
    if not shards:
        raise MergeError("未提供分片")
    _check_schema_consistency(shards)
    _check_episode_disjoint(shards)
    arrays = _concat_arrays(shards)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "samples.npz", **arrays)
    (out_dir / "schema.json").write_text(
        json.dumps(shards[0]["schema"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows = _minimal_rows(arrays)
    episodes = [episode for shard in shards for episode in shard["manifest"].get("episodes", [])]
    skipped = [episode for shard in shards for episode in shard["manifest"].get("skipped", [])]
    targets = dict(shards[0]["manifest"].get("targets") or {})
    collected = [episode for episode in episodes if episode.get("status") == "collected"]

    pools_info: list[dict] = []
    for pool in shards[0]["manifest"].get("pools", []):
        name = pool.get("name")
        selected = sum(
            int(entry.get("selected") or 0)
            for shard in shards
            for entry in shard["manifest"].get("pools", [])
            if entry.get("name") == name
        )
        pools_info.append(
            {
                "name": name,
                "path": pool.get("path"),
                "sha256": pool.get("sha256"),
                "n_specs": pool.get("n_specs"),
                "selected": selected,
            }
        )

    class_counts = {window_class: 0 for window_class in cr.WINDOW_CLASSES}
    usable_class_counts = {window_class: 0 for window_class in cr.WINDOW_CLASSES}
    for row in rows:
        class_counts[row["class"]] = class_counts.get(row["class"], 0) + 1
        if row["usable"]:
            usable_class_counts[row["class"]] = usable_class_counts.get(row["class"], 0) + 1
    counts = {
        "episodes_attempted": sum(
            int(shard["manifest"].get("counts", {}).get("episodes_attempted", 0))
            for shard in shards
        ),
        "episodes_collected": len(collected),
        "episodes_skipped": len([episode for episode in episodes if episode.get("status") == "skipped"]),
        "windows_collected": sum(
            int(shard["manifest"].get("counts", {}).get("windows_collected", 0))
            for shard in shards
        ),
        "rows_total": len(rows),
        "rows_usable": sum(1 for row in rows if row["usable"]),
        "class_counts": class_counts,
        "usable_class_counts": usable_class_counts,
    }
    quota = {
        window_class: {
            "target": int(targets.get(window_class, 0)),
            "usable": int(usable_class_counts.get(window_class, 0)),
            "met": int(usable_class_counts.get(window_class, 0)) >= int(targets.get(window_class, 0)),
            "gap": max(0, int(targets.get(window_class, 0)) - int(usable_class_counts.get(window_class, 0))),
        }
        for window_class in cr.WINDOW_CLASSES
    }
    divergence: dict = {
        key: sum(
            int(shard["manifest"].get("divergence", {}).get(key, 0)) for shard in shards
        )
        for key in ("windows_collected", "strict", "relaxed", "skipped_state_mismatch")
    }
    divergence["divergence_rate"] = (
        (divergence["windows_collected"] - divergence["strict"])
        / max(1, divergence["windows_collected"])
    )

    def _coverage(key_name: str) -> dict[str, dict]:
        coverage: dict[str, dict] = {}
        for index in range(len(arrays["usable"])):
            key = str(arrays[key_name][index] or "?")
            bucket = coverage.setdefault(key, {"episodes": set(), "rows": 0, "usable_rows": 0})
            bucket["episodes"].add(f"{arrays['pool'][index]}:{arrays['episode_id'][index]}")
            bucket["rows"] += 1
            bucket["usable_rows"] += int(arrays["usable"][index])
        return {
            key: {
                "episodes": len(value["episodes"]),
                "rows": value["rows"],
                "usable_rows": value["usable_rows"],
            }
            for key, value in coverage.items()
        }

    road_type_coverage = {
        "by_episode_geometry": _coverage("geometry"),
        "by_row_road_class": _coverage("road_class"),
        "hard_types": list(cr.HARD_ROAD_TYPES),
    }
    pool_plan = cr.build_pool_plan(rows, episodes, pools_info, targets)

    episode_keys = [f"{episode.get('pool')}:{episode['id']}" for episode in collected]
    train, val = cr.split_episodes(episode_keys, val_every=val_every)
    train_set, val_set = set(train), set(val)
    if train_set & val_set:
        raise MergeError("split 泄漏（不应发生）")

    def _split_counts(keys: set[str]) -> dict[str, int]:
        subset = [row for row in rows if f"{row['pool']}:{row['episode_id']}" in keys]
        return {
            "episodes": len(keys),
            "rows": len(subset),
            "usable_rows": sum(1 for row in subset if row["usable"]),
        }

    split_manifest = {
        "rule": f"sorted('pool:id') 每 {val_every} 个末位进 val（episode 级；行不跨集）",
        "no_leakage": len(train_set & val_set) == 0,
        "train_episodes": train,
        "val_episodes": val,
        "train": _split_counts(train_set),
        "val": _split_counts(val_set),
        "generated_at": _now(),
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    gate_fail_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    for shard in shards:
        for key, value in shard["quality"].get("gate_fail_counts", {}).items():
            gate_fail_counts[key] = gate_fail_counts.get(key, 0) + int(value)
        for key, value in shard["quality"].get("reason_counts", {}).items():
            reason_counts[key] = reason_counts.get(key, 0) + int(value)
    per_class: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = per_class.setdefault(row["class"], {"rows": 0, "usable": 0})
        bucket["rows"] += 1
        bucket["usable"] += int(row["usable"])
    quality = {
        "status": "PASS" if counts["rows_usable"] == counts["rows_total"] else "FAIL",
        "input": f"merged {len(shards)} shards",
        "n_rows": counts["rows_total"],
        "n_usable": counts["rows_usable"],
        "usable_ratio": counts["rows_usable"] / max(1, counts["rows_total"]),
        "gate_fail_counts": gate_fail_counts,
        "reason_counts": reason_counts,
        "params": shards[0]["quality"].get("params"),
        "gate_order": shards[0]["quality"].get("gate_order"),
        "generated_at": _now(),
        "per_class": per_class,
        "quota": quota,
        "per_pool": pool_plan["per_pool_yield"],
        "road_type_coverage": road_type_coverage,
        "per_episode": [
            {
                "id": episode["id"],
                "pool": episode.get("pool"),
                "windows": [
                    {"class": window["class"], "takeover_step": window["takeover_step"], "status": window["status"]}
                    for window in episode.get("windows", [])
                ],
                "n_rows": episode.get("n_rows", 0),
                "n_usable": episode.get("n_usable", 0),
            }
            for episode in collected
        ],
    }
    (out_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    data_gate = build_data_gate_report(rows, episodes, targets=targets)
    (out_dir / "data_gate_report.json").write_text(
        json.dumps(data_gate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "tool": "tools/controls/merge_recovery_shards.py",
        "created_at": _now(),
        "git_commit": cr._git_commit(),
        "n_shards": len(shards),
        "sources": [
            {
                "name": shard["name"],
                "path": str(shard["dir"]),
                "samples_sha256": shard["samples_sha256"],
                "shard": shard["manifest"].get("shard"),
                "counts": shard["manifest"].get("counts", {}),
                "targets": shard["manifest"].get("targets"),
                "args": shard["manifest"].get("args"),
            }
            for shard in shards
        ],
        "args": {"shards": [str(path) for path in shard_dirs], "out": str(out_dir), "val_every": val_every},
        "ckpt": shards[0]["manifest"].get("ckpt"),
        "config": shards[0]["manifest"].get("config"),
        "pools": pools_info,
        "targets": targets,
        "protocol": shards[0]["manifest"].get("protocol"),
        "window_params": shards[0]["manifest"].get("window_params"),
        "counts": counts,
        "quota": quota,
        "divergence": divergence,
        "road_type_coverage": road_type_coverage,
        "pool_plan": pool_plan,
        "data_gate": data_gate,
        "field_gaps": shards[0]["manifest"].get("field_gaps", []),
        "episodes": episodes,
        "skipped": skipped,
        "generated_at": _now(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    cr._maybe_write_dataset_card(out_dir)
    return {
        "out": str(out_dir),
        "counts": counts,
        "data_gate": data_gate,
        "split": split_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merge_recovery_shards.py", description="合并 recovery 分片 + 数据门核验"
    )
    parser.add_argument("--shards", nargs="+", required=True, help="分片目录列表（≥1）")
    parser.add_argument("--out", required=True, help="输出目录（正式 datasets/recovery_v1）")
    parser.add_argument("--val-every", type=int, default=5, help="episode 级 val 比例分母（默认 5=20%%）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = merge_shards(args.shards, args.out, val_every=args.val_every)
    except MergeError as exc:
        print(f"[merge] ERROR: {exc}", file=sys.stderr)
        return 2
    gate = summary["data_gate"]
    counts = summary["counts"]
    print(
        f"[merge] shards={len(args.shards)} → {summary['out']} | "
        f"rows={counts['rows_total']} usable={counts['rows_usable']} "
        f"episodes={counts['episodes_collected']} | data_gate={gate['status']}",
        flush=True,
    )
    for check in gate["checks"]:
        if not check["pass"]:
            print(
                f"  [GATE FAIL] {check['name']}: {check['observed']} (需 {check['requirement']})",
                flush=True,
            )
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
