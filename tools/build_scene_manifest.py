#!/usr/bin/env python
"""Build a static RL scene manifest from one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from env.scenario.spec import load_specs
from pipeline.eval.closed_loop import load_model
from pipeline.train.rl import run_rollout_episode
from pipeline.train.rl_reward import RewardConfig

_MODEL = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--specs", default="data/specs/train_2000.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--action-mode", choices=("4d_exact",), default="4d_exact")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--std-min", type=float, nargs="+", default=[0.005, 0.0, 0.0, 0.005])
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-decisions", type=int, default=120)
    parser.add_argument(
        "--expert-meta",
        default="data/episodes/train_2000/il_expert_exact/expert_bc.meta.json",
    )
    return parser.parse_args()


def _worker_init(ckpt: str, device: str) -> None:
    global _MODEL
    _MODEL = load_model(ckpt, device, strict=True)


def _worker(task: Dict[str, Any]) -> Dict[str, Any]:
    episode = run_rollout_episode(
        _MODEL,
        task["spec"],
        device=task["device"],
        scene_seed=int(task["spec"].seed),
        member=int(task["member"]),
        sampling_seed=int(task["sampling_seed"]),
        max_steps=int(task["max_steps"]),
        max_decisions=int(task["max_decisions"]),
        reward_config=RewardConfig(),
        std_min=task["std_min"],
    )
    summary = episode["summary"]
    return {
        "seed": int(summary["scene_seed"]),
        "terminal": str(summary["terminal_event"]),
        "success": bool(summary["success"]),
        "route_completion": float(summary["route_completion"]),
        "reward_sum": float(summary["reward_sum"]),
        "decisions": int(summary["decision_count"]),
    }


def _entropy(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def _expert_success(path: str) -> Dict[int, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"expert meta has no episodes: {path}")
    return {
        int(item["scene_seed"]): int(bool(item["success"]))
        for item in episodes
    }


def _select(stats: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
    eligible = [
        item
        for item in stats
        if item["expert_success"] and item["members_observed"] == item["members"]
    ]
    frontier = sorted(
        [item for item in eligible if 0 < item["success_count"] < item["members"]],
        key=lambda item: (-item["outcome_entropy"], item["seed"]),
    )
    failure = sorted(
        [item for item in eligible if item["success_count"] == 0],
        key=lambda item: (-item["outcome_entropy"], item["seed"]),
    )
    success = sorted(
        [item for item in eligible if item["success_count"] == item["members"]],
        key=lambda item: item["seed"],
    )
    selected: List[Dict[str, Any]] = []
    selected += frontier[: min(len(frontier), int(target * 0.6))]
    selected += failure[: min(len(failure), int(target * 0.3))]
    selected += success[: min(len(success), int(target * 0.1))]
    seen = {item["seed"] for item in selected}
    if len(selected) < target:
        for item in sorted(eligible, key=lambda item: (-item["value"], item["seed"])):
            if item["seed"] in seen:
                continue
            selected.append(item)
            seen.add(item["seed"])
            if len(selected) >= target:
                break
    if not selected:
        raise ValueError("no eligible scenes selected")
    return selected


def main() -> int:
    args = _parse_args()
    if int(args.members) < 3:
        raise ValueError("--members must be >= 3")
    if int(args.workers) < 1:
        raise ValueError("--workers must be >= 1")
    if int(args.target) < 8:
        raise ValueError("--target must be >= scenes_per_update (8)")
    if len(args.std_min) not in (1, 4):
        raise ValueError("--std-min must be scalar or 4-vector")
    if any(not (0.0 <= float(value) <= 1.0) for value in args.std_min):
        raise ValueError("--std-min values must be in [0, 1]")
    specs = load_specs(args.specs)
    if int(args.limit) > 0:
        specs = specs[: int(args.limit)]
    expert = _expert_success(args.expert_meta)
    tasks: List[Dict[str, Any]] = []
    for spec in specs:
        for member in range(int(args.members)):
            tasks.append(
                {
                    "spec": spec,
                    "member": member,
                    "device": args.device,
                    "sampling_seed": int(
                        (int(spec.seed) * 1_000_003 + member * 17 + 1) % (2**31 - 1)
                    ),
                    "max_steps": int(args.max_steps),
                    "max_decisions": int(args.max_decisions),
                    "std_min": args.std_min,
                }
            )
    context = mp.get_context("spawn")
    rows: List[Dict[str, Any]] = []
    with context.Pool(
        processes=int(args.workers),
        initializer=_worker_init,
        initargs=(args.ckpt, args.device),
    ) as pool:
        for result in pool.imap_unordered(_worker, tasks, chunksize=1):
            rows.append(result)
    by_seed: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    stats: List[Dict[str, Any]] = []
    for spec in specs:
        group = by_seed.get(int(spec.seed), [])
        counts = Counter(row["terminal"] for row in group)
        members = int(args.members)
        observed = len(group)
        success_count = int(counts.get("success", 0))
        returns = np.asarray([row["reward_sum"] for row in group], dtype=np.float64)
        item = {
            "seed": int(spec.seed),
            "spec_id": int(spec.id),
            "difficulty": spec.difficulty,
            "members": members,
            "members_observed": observed,
            "success_count": success_count,
            "p_success": success_count / members,
            "terminal_histogram": dict(sorted(counts.items())),
            "outcome_entropy": _entropy(
                np.asarray([counts.get(event, 0) for event in ("success", "collision", "off_road", "timeout", "invalid_action")])
            ),
            "return_mean": float(returns.mean()) if len(returns) else 0.0,
            "return_std": float(returns.std()) if len(returns) else 0.0,
            "route_completion_mean": float(
                np.mean([row["route_completion"] for row in group]) if group else 0.0
            ),
            "decisions_mean": float(
                np.mean([row["decisions"] for row in group]) if group else 0.0
            ),
            "expert_success": int(expert.get(int(spec.seed), 0)),
        }
        item["value"] = float(item["outcome_entropy"])
        stats.append(item)
    selected = _select(stats, int(args.target))
    payload = {
        "schema": "drl_pathplan_rl_scene_manifest_v1",
        "ckpt": str(args.ckpt),
        "specs": str(args.specs),
        "members": int(args.members),
        "std_min": args.std_min,
        "action_mode": args.action_mode,
        "target": int(args.target),
        "selected": [int(item["seed"]) for item in selected],
        "stats": stats,
        "selected_stats": selected,
        "spec_sha256": hashlib.sha256(Path(args.specs).read_bytes()).hexdigest(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out)
    print(
        f"[scene-manifest] evaluated={len(stats)} selected={len(selected)} -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
