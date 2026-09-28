#!/usr/bin/env python3
"""生成 DAgger 采集池：从 train spec（5k 已覆盖的 5000 条）分层随机抽 3×500（互不重叠）。

协议（2026-09-28 定，v2）：
- 池只允许来自 train spec；`tools/dagger_collect.py` 启动时会做二次隔离校验（∩eval/val=∅）；
- 分层字段 = ``labels.geometry``；配额 = 最大余数法（与 ``make_eval_spec`` 同规则）；
- 层内无放回、固定种子（SEED + r）；三轮依次从剩余池中抽取 → 互不重叠、可复现；
- 产物：``env/specs/scenarios_train_dagger_r{1,2,3}.json``（wrapper 附 provenance）。
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now().isoformat(timespec="seconds")
SEED = 20260928
ROUNDS = 3
PER_ROUND = 500
COVERED_BY = "datasets/BTC20260926-2343_expert5k"


def _load(rel: str):
    data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    return data["specs"] if isinstance(data, dict) and "specs" in data else data


def _stratified_sample(pool, total: int, seed: int):
    """labels.geometry 分层 + 最大余数配额 + 层内无放回。返回 (picked, take)。"""
    groups = {}
    for spec in pool:
        groups.setdefault(spec["labels"]["geometry"], []).append(spec)
    labels = sorted(groups)
    quota = {lab: total * len(groups[lab]) / len(pool) for lab in labels}
    take = {lab: int(v) for lab, v in quota.items()}
    for lab in sorted(labels, key=lambda l: (-(quota[l] - take[l]), l))[: total - sum(take.values())]:
        take[lab] += 1
    rng = np.random.default_rng(seed)
    picked = []
    for lab in labels:
        idx = sorted(rng.choice(len(groups[lab]), size=take[lab], replace=False).tolist())
        picked += [groups[lab][i] for i in idx]
    return picked, take


def main() -> int:
    train = _load("env/specs/scenarios_train.json")
    rep = json.loads((ROOT / COVERED_BY / "report.json").read_text(encoding="utf-8"))
    covered = {(s["id"], s["seed"]) for s in rep["per_spec"]}
    pool = [s for s in train if (s["id"], s["seed"]) in covered]
    if len(pool) != 5000:
        raise SystemExit(f"[pools] 5k 覆盖集大小异常：{len(pool)}（期望 5000）")

    # 冻结隔离：候选池与 eval500 / val 的交集必须为空
    pool_keys = {(s["id"], s["seed"]) for s in pool}
    for name in ("scenarios_eval500.json", "scenarios_val.json"):
        other = {(s["id"], s["seed"]) for s in _load(f"env/specs/{name}")}
        inter = pool_keys & other
        if inter:
            raise SystemExit(f"[pools] 隔离失败：候选池 ∩ {name} = {sorted(inter)[:3]} …")

    remaining = list(pool)
    round_keys = []
    for r in range(1, ROUNDS + 1):
        picked, take = _stratified_sample(remaining, PER_ROUND, SEED + r)
        picked_keys = {(s["id"], s["seed"]) for s in picked}
        remaining = [s for s in remaining if (s["id"], s["seed"]) not in picked_keys]
        round_keys.append(picked_keys)
        rel = f"env/specs/scenarios_train_dagger_r{r}.json"
        prov = {
            "source": "env/specs/scenarios_train.json（5k 已覆盖的 5000 条）",
            "covered_by": COVERED_BY,
            "seed": SEED + r,
            "round": r,
            "rule": "labels.geometry 分层 / 最大余数配额 / 层内无放回 / 三轮互不重叠",
            "quota": take,
            "remaining_after": len(remaining),
            "generated_at": NOW,
        }
        payload = {"schema_version": 1, "count": len(picked), "provenance": prov, "specs": picked}
        (ROOT / rel).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        print(f"[pools] r{r}: n={len(picked)} 配额={take} 剩余={len(remaining)} → {rel}")

    for i in range(ROUNDS):
        for j in range(i + 1, ROUNDS):
            if round_keys[i] & round_keys[j]:
                raise SystemExit(f"[pools] r{i + 1} 与 r{j + 1} 重叠：{sorted(round_keys[i] & round_keys[j])[:3]}")
    print("[pools] ✓ 三轮互不重叠；候选池 ∩ eval/val = 空")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
