#!/usr/bin/env python3
"""P2 奖励审计工具（Gate2）：pre-v6 rollout 采集 + HEAD 完整 v5 项集离线重放 + 终局值反解。

规格：`docs/rl_reward_v5.md` §3/§7、`docs/v6_program_prereg.md` §3（P2/Gate2）。

结构
----
1. **采集**（`collect`）：E-β′（旧架构）无法在 HEAD 加载（v6 新头 missing=20），rollout 必须在
   pre-v6 代码快照（`031cc1c`，`git worktree`/`git archive` 到 /tmp/opencode/v6_pre）下执行；
   每 episode 记录**每策略步完整 ctx**（真实 `RewardAdapter._build_ctx` + `LocalEnvPool._record`
   同口径 info 注入，见 `tools/reward_audit_collect.py`）。分层：按终局类补足每类 n ≥ 目标。
2. **重放/审计**（`analyze`）：用 HEAD 的**真实 `RewardAggregator`（完整 v5 项集，含
   `low_speed`）**逐策略步离线重放采集到的 ctx；分类剖面 vs 目标 C（+50 / −20 / −15 / −10）、
   dense 正负拆分、CaRL/终局掩码命中、折扣/未折扣两层、`low_speed` 有无前车拆分、rc 各档剖面。
3. **终局值反解**（`analyze` 内）：样本与审计样本**互斥**（按终局类**分层随机**划分，
   `--split-seed` 固定可复现；`--swap-ab` 交换两半做 A/B 互换交叉验证；记录 id+hash）；
   每档 rc（1.0 基准 / 3 / 10 / 30）按 `终局值 = 目标 − 稠密贡献 − 终止项` 反解一套
   `terminal_values`，产出配置草案与复算差（容差 ≤ 0.5）。
4. `dry-run`：合成小样本（每类 ≥ 2）端到端自检，不建 env、不依赖快照。
5. **排除 eval500（默认开启）**：collect 池过滤 + analyze 兜底过滤，默认排除
   `env/specs/scenarios_eval500.json` 的 (id, seed) 集；`--no-exclude` 显式关闭；
   `--exclude ""` 报错（空值曾导致排除静默失效）。报告 meta 记录 **HEAD commit sha**、
   split seed、exclude 与每类最小样本数（E-β″ 复算用 `--min-per-class 50`）。

产物（`runs/reward_audit/report/`，`runs/` 为 gitignore）：`reward_audit.md`、
`reward_audit.json`、`config_draft_rc{tier}.yaml`、`episodes/`（采集缓存）。`--mirror-md`
可另写验收路径。入库副本（tracked）见 `docs/reward_audit/`（原始 runs/ 路径 + sha256
对照见该目录 `MANIFEST.md`）。
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 直接以脚本运行（非 -m）时可 import reward_model/pipeline
    sys.path.insert(0, str(ROOT))
DEFAULT_PRE_ROOT = Path("/tmp/opencode/v6_pre")
DEFAULT_PRE_COMMIT = "031cc1c"
DEFAULT_CKPT = ROOT / "runs/_refs_rlbase/e_beta_prime/final.pt"
DEFAULT_POOL = ROOT / "env/specs/scenarios_val.json"
DEFAULT_EXCLUDE = ROOT / "env/specs/scenarios_eval500.json"

#: 目标剖面 C（docs/rl_reward_v5.md §3；error 不在剖面内）
TARGETS: Dict[str, float] = {
    "arrive_dest": 50.0,
    "collision": -20.0,
    "out_of_road": -15.0,
    "max_step": -10.0,
}
CLASS_ORDER: Tuple[str, ...] = ("arrive_dest", "collision", "out_of_road", "max_step")
#: 默认 rc 档：1.0 = 代码默认（基准档，回填 default_terminal_values）；3/10/30 = P4 扫档
DEFAULT_TIERS: Tuple[float, ...] = (1.0, 3.0, 10.0, 30.0)
#: 反解复算差冻结容差（§3）
TOLERANCE = 0.5
#: 审计剖面 vs 目标判定容差（§7「建议绝对 ≤ 5」）
PROFILE_TOLERANCE = 5.0
#: 审计每类最小样本数（§7）
MIN_PER_CLASS = 10
#: 审计/反解各自最小总样本数（§3/§7）
MIN_TOTAL = 50
#: 分层随机划分默认 seed（固定；报告记录；E-β″ 复算用 --split-seed 显式覆盖）
DEFAULT_SPLIT_SEED = 20261001
GAMMA = 0.99


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not math.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}" if abs(value) >= 1e-12 or value == 0 else f"{value:.{digits}f}"


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(statistics.fmean(values)) if values else None


def _std(values: Sequence[float]) -> Optional[float]:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0 if values else None


def _load_spec_pairs(path: Path) -> List[Tuple[int, int]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    specs = doc["specs"] if isinstance(doc, Mapping) else doc
    return [(int(spec["id"]), int(spec["seed"])) for spec in specs]


def _pair_key(spec_id: int, seed: int) -> str:
    return f"{spec_id}:{seed}"


def _head_commit(repo: Path = ROOT) -> Optional[str]:
    """当前 HEAD commit sha（记录进报告 meta；git 不可用/非仓库时 None）。"""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except Exception:  # noqa: BLE001 - 记录性字段，不阻塞审计
        return None
    text = proc.stdout.strip()
    return text if proc.returncode == 0 and text else None


def _resolve_exclude(args: argparse.Namespace) -> Optional[Path]:
    """解析排除文件：默认 `--exclude`（eval500）；`--no-exclude` 关闭；空值报错（防静默失效）。"""
    if getattr(args, "no_exclude", False):
        return None
    raw = getattr(args, "exclude", None)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        raise SystemExit(
            "--exclude 需为非空路径（空值曾导致排除静默失效）；如需关闭排除请显式 --no-exclude"
        )
    return Path(text).resolve()


def _load_excluded(path: Optional[Path]) -> set:
    """排除集的 (id, seed) 集合（`scenarios_eval500.json` 场景 id 集；val 池下即其 500 条）。

    fail-closed：排除文件缺失时报错（排除是安全要求）；确需关闭请显式 `--no-exclude`。
    """
    if path is None:
        return set()
    if not path.is_file():
        raise SystemExit(f"排除文件不存在：{path}（生成 spec 或显式 --no-exclude）")
    return set(_load_spec_pairs(path))


def _candidate_pairs(pool_path: Path, exclude_path: Optional[Path]) -> List[Tuple[int, int]]:
    """池候选 (id, seed)：排序 + 排除 exclude 集（默认 eval500）。"""
    pairs = sorted(_load_spec_pairs(pool_path))
    excluded = _load_excluded(exclude_path)
    if excluded:
        pairs = [pair for pair in pairs if pair not in excluded]
    return pairs


def _filter_excluded(
    episodes: Sequence[Mapping[str, Any]], excluded: set
) -> List[Mapping[str, Any]]:
    """analyze 兜底过滤：采集未排除时，复算也不得包含 exclude 集（如 eval500）。"""
    if not excluded:
        return list(episodes)
    out: List[Mapping[str, Any]] = []
    for episode in episodes:
        meta = episode["meta"]
        pair = (int(meta.get("spec_id", -1)), int(meta.get("spec_seed", -1)))
        if pair not in excluded:
            out.append(episode)
    return out


# --------------------------------------------------------------------------- #
# 1) pre-v6 快照
# --------------------------------------------------------------------------- #
def ensure_snapshot(pre_root: Path, commit: str, *, repo: Path = ROOT) -> Dict[str, Any]:
    """确保 pre-v6 代码快照存在（git worktree 优先，失败回退 git archive）。

    任务书要求 `git worktree add /tmp/opencode/v6_pre 031cc1c`；`git worktree` 不可用时
    （受限环境）回退 `git archive <commit> | tar -x`——两者内容逐位等价（快照不含 .git）。
    """
    marker = pre_root / ".v6_pre_commit"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == commit:
        return {"pre_root": str(pre_root), "commit": commit, "method": "reuse"}
    if pre_root.is_dir() and any(pre_root.iterdir()) and marker.is_file():
        raise SystemExit(f"快照 {pre_root} 已存在但 commit 不匹配（期望 {commit}）")

    pre_root.parent.mkdir(parents=True, exist_ok=True)
    method = ""
    if not pre_root.exists() or not any(pre_root.iterdir()):
        proc = subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(pre_root), commit],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            method = "worktree"
        else:
            print(f"[audit] git worktree 不可用（{proc.stderr.strip()[:200]}）→ 回退 git archive", flush=True)
    if not method:
        if not pre_root.exists():
            pre_root.mkdir(parents=True)
        tar_path = pre_root.parent / f".{pre_root.name}.tar"
        proc = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", f"--output={tar_path}", commit],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"git archive {commit} 失败：{proc.stderr.strip()}")
        proc = subprocess.run(["tar", "-xf", str(tar_path), "-C", str(pre_root)], capture_output=True, text=True)
        tar_path.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise SystemExit(f"解包快照失败：{proc.stderr.strip()}")
        method = "archive"
    marker.write_text(commit + "\n", encoding="utf-8")
    return {"pre_root": str(pre_root), "commit": commit, "method": method}


# --------------------------------------------------------------------------- #
# 2) 采集
# --------------------------------------------------------------------------- #
def collect(args: argparse.Namespace) -> Dict[str, Any]:
    pre_root = Path(args.pre_root).resolve()
    snap = ensure_snapshot(pre_root, args.pre_commit)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 路径按调用侧 cwd 解析（collector 子进程 cwd=pre_root，不能依赖相对路径）
    pool_path = Path(args.pool).resolve()
    exclude_path = _resolve_exclude(args)
    pairs = _candidate_pairs(pool_path, exclude_path)  # 确定性顺序（id, seed）+ 排除 eval500
    if args.max_episodes:
        pairs = pairs[: int(args.max_episodes)]

    per_class_target = int(args.per_class_target)
    print(
        f"[audit] collect：pool={pool_path}（排除 {exclude_path or '无'}）候选 {len(pairs)} 条；"
        f"每类目标 n≥{per_class_target}；workers={args.workers}",
        flush=True,
    )
    collected = _existing_outcomes(out_dir)
    counts = _class_counts(collected)
    started = time.time()
    attempted = 0

    def _run(pair: Tuple[int, int]) -> Dict[str, Any]:
        spec_id, seed = pair
        out_path = out_dir / f"id{spec_id}_seed{seed}.json"
        if out_path.is_file():
            return {"pair": pair, "path": str(out_path), "skipped": True}
        cmd = [
            args.python,
            str(ROOT / "tools/reward_audit_collect.py"),
            "--pre-root",
            str(pre_root),
            "--pre-commit",
            args.pre_commit,
            "--ckpt",
            str(Path(args.ckpt).resolve()),
            "--specs",
            str(pool_path),
            "--spec-id",
            str(spec_id),
            "--spec-seed",
            str(seed),
            "--config",
            str(Path(args.config).resolve()),
            "--tracker",
            args.tracker,
            "--eval-reference",
            args.eval_reference,
            "--device",
            args.device,
            "--max-steps",
            str(args.max_steps),
            "--out",
            str(out_path),
        ]
        env = dict(os.environ, PYTHONPATH=str(pre_root))
        proc = subprocess.run(cmd, cwd=str(pre_root), env=env, capture_output=True, text=True)
        log_path = log_dir / f"id{spec_id}_seed{seed}.log"
        log_path.write_text(
            (proc.stdout or "") + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else ""),
            encoding="utf-8",
        )
        if proc.returncode != 0:
            return {"pair": pair, "path": str(out_path), "error": proc.stderr.strip()[-400:]}
        return {"pair": pair, "path": str(out_path), "skipped": False}

    pending = [pair for pair in pairs if not (out_dir / f"id{pair[0]}_seed{pair[1]}.json").is_file()]
    batch_size = max(1, int(args.batch))
    done_total = len(pairs) - len(pending)
    with cf.ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            for result in pool.map(_run, chunk):
                attempted += 1
                if "error" in result:
                    print(f"[audit] collect 失败 {result['pair']}: {result['error']}", flush=True)
            done_total += len(chunk)
            collected = _existing_outcomes(out_dir)
            counts = _class_counts(collected)
            print(
                f"[audit] collect 进度 {done_total}/{len(pairs)}："
                + " ".join(f"{cls}={counts.get(cls, 0)}" for cls in CLASS_ORDER)
                + f" other={counts.get('other', 0)}（{time.time() - started:.0f}s）",
                flush=True,
            )
            if all(counts.get(cls, 0) >= per_class_target for cls in CLASS_ORDER):
                print(f"[audit] collect 达到每类目标（{per_class_target}），提前停止", flush=True)
                break
    collected = _existing_outcomes(out_dir)
    counts = _class_counts(collected)
    summary = {
        "pre_snapshot": snap,
        "pool": str(pool_path),
        "exclude": str(exclude_path) if exclude_path else None,
        "exclude_pairs": len(_load_excluded(exclude_path)),
        "head_commit": _head_commit(),
        "episodes_dir": str(out_dir),
        "attempted": attempted,
        "collected": len(collected),
        "class_counts": dict(counts),
        "workers": int(args.workers),
        "max_steps": int(args.max_steps),
        "ckpt": str(Path(args.ckpt).resolve()),
        "ckpt_sha256": _sha256_file(Path(args.ckpt)),
        "wall_time_s": time.time() - started,
    }
    (out_dir.parent / "collect_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[audit] collect 完成：{summary['class_counts']}（{summary['wall_time_s']:.0f}s）", flush=True)
    return summary


def _existing_outcomes(out_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for path in sorted(out_dir.glob("id*_seed*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            meta = doc["meta"]
        except Exception:  # noqa: BLE001 - 半成品/损坏文件跳过（采集可 resume 重跑）
            continue
        out.append({"path": str(path), "meta": meta, "steps": doc.get("steps", [])})
    return out


def _class_counts(episodes: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for episode in episodes:
        cls = str(episode["meta"].get("termination", "other"))
        counts[cls] = counts.get(cls, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# 3) 重放（HEAD 真实聚合器）
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeReplay:
    spec_id: int
    seed: int
    cls: str
    termination: str
    steps: int
    policy_steps: int
    ctx_sha256: str
    file_sha256: Optional[str]
    total: float
    discounted: float
    dense_positive_sum: float
    dense_negative_sum: float
    dense_effective: float
    dense_positive_only: float
    terminating_sum: float
    carl_penalty: float
    terminal_value: float
    carl_hits: int
    terminal_hits: int
    mask_fallback_hits: int
    term_sums: Dict[str, float] = field(default_factory=dict)
    low_speed_lead_sum: float = 0.0
    low_speed_nolead_sum: float = 0.0
    low_speed_lead_steps: int = 0
    low_speed_nolead_steps: int = 0
    rewards: List[float] = field(default_factory=list)
    carl_zeroed_dense: float = 0.0  # 被乘子清零的正向稠密（= dense_pos − dense_pos×mult）
    triggered_frames: Dict[str, int] = field(default_factory=dict)


def _replay_episode(
    episode: Mapping[str, Any],
    *,
    rc_weight: float,
    terminal_values: Mapping[str, float],
    inject_max_step: bool = True,
) -> EpisodeReplay:
    from reward_model import DEFAULT_TERM_CONFIGS, RewardAggregator, build_terms
    from reward_model.aggregation import AggregationConfig, default_carl_rules, discounted_return

    configs = [dict(term) for term in DEFAULT_TERM_CONFIGS]
    for term in configs:
        if term["name"] == "route_completion":
            term["weight"] = float(rc_weight)
    aggregator = RewardAggregator(
        build_terms(configs),
        AggregationConfig(terminal_values=dict(terminal_values), carl_rules=default_carl_rules()),
    )
    meta = episode["meta"]
    cls = str(meta.get("termination", "other"))
    steps = list(episode.get("steps", []))
    rewards: List[float] = []
    dense_pos = dense_neg = terminating = carl_penalty = terminal_value = 0.0
    dense_effective = dense_positive_only = 0.0
    carl_hits = terminal_hits = mask_fallback_hits = 0
    term_sums: Dict[str, float] = {}
    ls_lead = ls_nolead = 0.0
    ls_lead_steps = ls_nolead_steps = 0
    carl_zeroed = 0.0
    triggered_frames: Dict[str, int] = {}
    for index, record in enumerate(steps):
        ctx = {key: value for key, value in dict(record.get("ctx", {})).items() if not str(key).startswith("__")}
        is_last = index == len(steps) - 1
        if inject_max_step and is_last and cls == "max_step" and not ctx.get("max_step"):
            # rollout 循环在 max_steps 截断时 env 未置 info.max_step（eval 侧同口径按步数分类）
            ctx["max_step"] = True
        result = aggregator.step(ctx, step_index=int(record.get("policy_step", index)))
        rewards.append(float(result.reward))
        dense_pos += float(result.dense_positive_sum)
        dense_neg += float(result.dense_negative_sum)
        dense_effective += float(result.dense_positive_sum) * float(result.carl_multiplier) + float(
            result.dense_negative_sum
        )
        dense_positive_only += float(result.dense_positive_sum) + float(result.dense_negative_sum)
        terminating += float(result.terminating_sum)
        carl_penalty += float(result.carl_penalty)
        terminal_value += float(result.terminal_value)
        carl_zeroed += float(result.dense_positive_sum) * (1.0 - float(result.carl_multiplier))
        if abs(float(result.carl_multiplier) - 1.0) > 1e-9:
            carl_hits += 1
        if result.terminal_key is not None:
            terminal_hits += 1
        if ctx.get("__speed_limit_source") == "terminal_mask_fallback":
            mask_fallback_hits += 1
        for name, value in result.components.items():
            term_sums[name] = term_sums.get(name, 0.0) + float(value)
        for name, value in result.raw_components.items():
            if abs(float(value)) > 1e-12:
                triggered_frames[name] = triggered_frames.get(name, 0) + 1
        low_speed = float(result.components.get("low_speed", 0.0))
        lead = ctx.get("lead_gap_m")
        has_lead = isinstance(lead, (int, float)) and float(lead) > 0.0
        if has_lead:
            ls_lead += low_speed
            ls_lead_steps += 1
        else:
            ls_nolead += low_speed
            ls_nolead_steps += 1
    return EpisodeReplay(
        spec_id=int(meta.get("spec_id", -1)),
        seed=int(meta.get("spec_seed", -1)),
        cls=cls,
        termination=str(meta.get("termination", "other")),
        steps=int(meta.get("steps", 0)),
        policy_steps=len(steps),
        ctx_sha256=str(meta.get("ctx_sha256", "")),
        file_sha256=_sha256_file(Path(str(episode["path"]))) if episode.get("path") else None,
        total=float(sum(rewards)),
        discounted=float(discounted_return(rewards, GAMMA)),
        dense_positive_sum=dense_pos,
        dense_negative_sum=dense_neg,
        dense_effective=dense_effective,
        dense_positive_only=dense_positive_only,
        terminating_sum=terminating,
        carl_penalty=carl_penalty,
        terminal_value=terminal_value,
        carl_hits=carl_hits,
        terminal_hits=terminal_hits,
        mask_fallback_hits=mask_fallback_hits,
        term_sums=term_sums,
        low_speed_lead_sum=ls_lead,
        low_speed_nolead_sum=ls_nolead,
        low_speed_lead_steps=ls_lead_steps,
        low_speed_nolead_steps=ls_nolead_steps,
        rewards=rewards,
        carl_zeroed_dense=carl_zeroed,
        triggered_frames=triggered_frames,
    )


def _profile(replays: Sequence[EpisodeReplay], targets: Mapping[str, float]) -> Dict[str, Any]:
    """按终局类聚合剖面（mean±std，未折扣 + 折扣两层）。"""
    out: Dict[str, Any] = {}
    for cls in CLASS_ORDER:
        rows = [row for row in replays if row.cls == cls]
        if not rows:
            out[cls] = {"n": 0}
            continue
        totals = [row.total for row in rows]
        discounted = [row.discounted for row in rows]
        target = targets.get(cls)
        mean_total = _mean(totals)
        out[cls] = {
            "n": len(rows),
            "target": target,
            "total_mean": mean_total,
            "total_std": _std(totals),
            "total_min": min(totals),
            "total_max": max(totals),
            "delta": (mean_total - target) if (mean_total is not None and target is not None) else None,
            "discounted_mean": _mean(discounted),
            "discounted_std": _std(discounted),
            "dense_positive_sum_mean": _mean([row.dense_positive_sum for row in rows]),
            "dense_negative_sum_mean": _mean([row.dense_negative_sum for row in rows]),
            "dense_effective_mean": _mean([row.dense_effective for row in rows]),
            "dense_positive_only_mean": _mean([row.dense_positive_only for row in rows]),
            "terminating_sum_mean": _mean([row.terminating_sum for row in rows]),
            "carl_penalty_mean": _mean([row.carl_penalty for row in rows]),
            "terminal_value_mean": _mean([row.terminal_value for row in rows]),
            "carl_zeroed_dense_mean": _mean([row.carl_zeroed_dense for row in rows]),
            "carl_hit_episodes": sum(1 for row in rows if row.carl_hits > 0),
            "carl_hit_steps": sum(row.carl_hits for row in rows),
            "terminal_hit_steps": sum(row.terminal_hits for row in rows),
            "mask_fallback_steps": sum(row.mask_fallback_hits for row in rows),
            "steps_mean": _mean([float(row.policy_steps) for row in rows]),
            "term_means": {
                name: _mean([row.term_sums.get(name, 0.0) for row in rows])
                for name in sorted({key for row in rows for key in row.term_sums})
            },
            "triggered_frames": {
                name: sum(row.triggered_frames.get(name, 0) for row in rows)
                for name in sorted({key for row in rows for key in row.triggered_frames})
            },
            "low_speed_lead_sum_mean": _mean([row.low_speed_lead_sum for row in rows]),
            "low_speed_nolead_sum_mean": _mean([row.low_speed_nolead_sum for row in rows]),
            "low_speed_lead_steps": sum(row.low_speed_lead_steps for row in rows),
            "low_speed_nolead_steps": sum(row.low_speed_nolead_steps for row in rows),
            "episodes": [
                {
                    "spec_id": row.spec_id,
                    "seed": row.seed,
                    "ctx_sha256": row.ctx_sha256,
                    "file_sha256": row.file_sha256,
                    "total": row.total,
                    "discounted": row.discounted,
                }
                for row in sorted(rows, key=lambda item: (item.spec_id, item.seed))
            ],
        }
    out["other"] = {
        "n": sum(1 for row in replays if row.cls == "other"),
        "total_mean": _mean([row.total for row in replays if row.cls == "other"]),
    }
    return out


def _solve_terminal_values(
    replays: Sequence[EpisodeReplay], targets: Mapping[str, float]
) -> Dict[str, Any]:
    """按 `终局值 = 目标 − 稠密贡献 − 终止项 − carl_penalty` 反解每类终局值（整数定稿）。"""
    out: Dict[str, Any] = {}
    for cls in CLASS_ORDER:
        rows = [row for row in replays if row.cls == cls]
        if not rows:
            out[cls] = {"n": 0, "resolved": None}
            continue
        dense = _mean([row.dense_effective for row in rows]) or 0.0
        dense_pos_only = _mean([row.dense_positive_only for row in rows]) or 0.0
        terminating = _mean([row.terminating_sum for row in rows]) or 0.0
        carl = _mean([row.carl_penalty for row in rows]) or 0.0
        raw = float(targets[cls]) - dense - terminating - carl
        rounded = float(round(raw))
        recomputed = dense + terminating + carl + rounded
        out[cls] = {
            "n": len(rows),
            "target": targets[cls],
            "dense_effective_mean": dense,
            "dense_positive_only_mean": dense_pos_only,
            "terminating_mean": terminating,
            "carl_penalty_mean": carl,
            "resolved_raw": raw,
            "resolved": rounded,
            "recomputed": recomputed,
            "recompute_delta": recomputed - targets[cls],
            "positive_only_resolved_raw": float(targets[cls]) - dense_pos_only - terminating - carl,
        }
    return out


# --------------------------------------------------------------------------- #
# 4) 报告
# --------------------------------------------------------------------------- #
def _render_markdown(result: Mapping[str, Any]) -> str:
    lines: List[str] = []
    meta = result["meta"]
    min_per_class = meta.get("min_per_class", MIN_PER_CLASS)
    lines.append("# P2 奖励审计报告（v5 剖面 C）\n")
    lines.append(
        f"- 生成：{meta['generated_at']}（工具 `tools/reward_audit.py`；HEAD commit `{meta.get('head_commit')}`）"
    )
    snapshot = meta.get("pre_snapshot") or {"pre_root": "—", "commit": "—", "method": "—"}
    lines.append(f"- pre-v6 快照：`{snapshot['pre_root']}` @ `{snapshot['commit']}`（{snapshot['method']}）")
    lines.append(f"- ckpt：`{meta.get('ckpt', '—')}`（sha256 `{meta.get('ckpt_sha256')}`）")
    lines.append(f"- 池：`{meta.get('pool', '—')}`，排除 `{meta.get('exclude')}`；采集 episode {meta['collected']} 条"
                 + (f"（原始 {meta.get('collected_raw')}，剔除 {meta.get('excluded_episodes')}）"
                    if meta.get("excluded_episodes") else ""))
    lines.append(f"- 分类计数：{meta['class_counts']}")
    lines.append(f"- 划分：分层随机（split seed `{meta.get('split_seed')}`；A/B 互换 `{meta.get('swap_ab')}`）；"
                 f"每类 n ≥ {min_per_class}")
    lines.append(f"- 审计样本 {meta['audit']['n']} 条 / 反解样本 {meta['solve']['n']} 条（互斥，见 §样本清单）")
    lines.append(f"- 重放口径：HEAD `RewardAggregator`（完整 v5 项集含 `low_speed`；`ttc`/`lane_boundary`/"
                 f"`lane_center` 默认关）；γ={GAMMA}；反解复算容差 ≤ {TOLERANCE}，"
                 f"剖面判定容差 ≤ {PROFILE_TOLERANCE}\n")

    lines.append("## 0. 结论\n")
    lines.append(f"- 目标剖面达标（审计样本，未折扣均值 |Δ| ≤ {PROFILE_TOLERANCE}）："
                 + ("**是**" if result["meta"]["profile_pass"] else "**否**"))
    lines.append(f"- 每类 n ≥ {min_per_class}："
                 + ("**是**" if result["meta"]["per_class_pass"] else "**否**"))
    lines.append(f"- 总样本 ≥ {MIN_TOTAL}（审计/反解各自）："
                 + ("**是**" if result["meta"]["total_pass"] else "**否**"))
    for tier in result["meta"]["tiers"]:
        entry = result["tiers"][str(tier)]
        lines.append(
            f"- rc={tier:g} 档：剖面判定 "
            + ("**通过**" if entry.get("profile_pass") else "**未通过**")
            + f"（|Δ|max={_fmt(entry.get('delta_max'))}，每类 n≥{min_per_class}："
            + ("是" if entry.get("per_class_pass") else "否")
            + "）"
        )
    for cls in CLASS_ORDER:
        item = result["tiers"][str(result["meta"]["tiers"][0])]["audit_profile"].get(cls, {})
        if item.get("n"):
            lines.append(
                f"- 基准档（rc={result['meta']['tiers'][0]:g}）{cls}：n={item['n']} "
                f"mean={_fmt(item['total_mean'])}（目标 {_fmt(item['target'], 0)}，"
                f"Δ={_fmt(item['delta'])}）"
            )
    lines.append("")

    lines.append("## 1. 样本清单（id + hash）\n")
    for label in ("audit", "solve"):
        lines.append(f"### {label}（{result[label]['n']} 条）\n")
        lines.append("| 类 | id | seed | ctx_sha256 | 文件 sha256 | total |")
        lines.append("|---|---|---|---|---|---|")
        for cls in CLASS_ORDER:
            for row in result[label]["manifest"].get(cls, []):
                lines.append(
                    f"| {cls} | {row['spec_id']} | {row['seed']} | `{row['ctx_sha256'][:18]}…` | "
                    f"`{str(row['file_sha256'])[:18]}…` | {_fmt(row['total'])} |"
                )
        lines.append("")

    lines.append("## 2. 分类剖面 vs 目标（未折扣 + 折扣两层）\n")
    lines.append("| 档 | 类 | n | 未折扣 mean±std | 目标 | |Δ| | 折扣 mean±std | dense+ | dense− | 终止项 | carl_pen | 终局值 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tier in result["meta"]["tiers"]:
        profile = result["tiers"][str(tier)]["audit_profile"]
        for cls in CLASS_ORDER:
            item = profile.get(cls, {})
            if not item.get("n"):
                continue
            lines.append(
                f"| rc={tier:g} | {cls} | {item['n']} | "
                f"{_fmt(item['total_mean'])}±{_fmt(item['total_std'], 2)} | {_fmt(item['target'], 0)} | "
                f"{_fmt(abs(item['delta']) if item['delta'] is not None else None)} | "
                f"{_fmt(item['discounted_mean'])}±{_fmt(item['discounted_std'], 2)} | "
                f"{_fmt(item['dense_positive_sum_mean'])} | {_fmt(item['dense_negative_sum_mean'])} | "
                f"{_fmt(item['terminating_sum_mean'])} | {_fmt(item['carl_penalty_mean'])} | "
                f"{_fmt(item['terminal_value_mean'])} |"
            )
    lines.append("")

    lines.append("## 3. 终局值反解（每档一套；样本与审计互斥）\n")
    lines.append("| 档 | 类 | n | 稠密贡献(effective) | 终止项 | carl_pen | 反解 raw | 定稿(整数) | 复算 | 复算差 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for tier in result["meta"]["tiers"]:
        solve = result["tiers"][str(tier)]["solve"]
        for cls in CLASS_ORDER:
            item = solve.get(cls, {})
            if not item.get("n"):
                continue
            lines.append(
                f"| rc={tier:g} | {cls} | {item['n']} | {_fmt(item['dense_effective_mean'])} | "
                f"{_fmt(item['terminating_mean'])} | {_fmt(item['carl_penalty_mean'])} | "
                f"{_fmt(item['resolved_raw'])} | {_fmt(item['resolved'], 1)} | {_fmt(item['recomputed'])} | "
                f"{_fmt(item['recompute_delta'])} |"
            )
    lines.append("")
    lines.append("各档 `terminal_values`（`error` 保留 −5）：\n")
    for tier in result["meta"]["tiers"]:
        values = result["tiers"][str(tier)]["terminal_values"]
        lines.append(f"- rc={tier:g}：`{json.dumps(values, ensure_ascii=False)}`")
    lines.append("")

    lines.append("## 4. dense 语义拆分与 CaRL / 终局掩码命中\n")
    lines.append("| 档 | 类 | n | pos×mult+neg | pos+neg（忽略乘子） | 被 CaRL 清零 | carl 命中 episode/step | 终局帧命中 step | 终局掩码回退 step |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for tier in result["meta"]["tiers"]:
        profile = result["tiers"][str(tier)]["audit_profile"]
        for cls in CLASS_ORDER:
            item = profile.get(cls, {})
            if not item.get("n"):
                continue
            lines.append(
                f"| rc={tier:g} | {cls} | {item['n']} | {_fmt(item['dense_effective_mean'])} | "
                f"{_fmt(item['dense_positive_only_mean'])} | {_fmt(item['carl_zeroed_dense_mean'])} | "
                f"{item['carl_hit_episodes']}/{item['carl_hit_steps']} | {item['terminal_hit_steps']} | "
                f"{item['mask_fallback_steps']} |"
            )
    lines.append("")

    lines.append("### 4b. 违规/触发帧统计（审计样本，各类合计触发帧数）\n")
    lines.append("| 档 | 类 | n | 触发帧（raw ≠ 0） |")
    lines.append("|---|---|---|---|")
    for tier in result["meta"]["tiers"]:
        profile = result["tiers"][str(tier)]["audit_profile"]
        for cls in CLASS_ORDER:
            item = profile.get(cls, {})
            if not item.get("n"):
                continue
            triggered = item.get("triggered_frames") or {}
            top = sorted(triggered.items(), key=lambda kv: kv[1], reverse=True)
            lines.append(
                f"| rc={tier:g} | {cls} | {item['n']} | "
                + ("; ".join(f"`{name}`={count}" for name, count in top) if top else "—")
                + " |"
            )
    lines.append("")

    lines.append("## 5. low_speed 拆分（有无前车；审计样本）\n")
    lines.append("| 档 | 类 | n | 有前车 Σ/步 | 无前车 Σ/步 | 有前车步 | 无前车步 |")
    lines.append("|---|---|---|---|---|---|---|")
    for tier in result["meta"]["tiers"]:
        profile = result["tiers"][str(tier)]["audit_profile"]
        for cls in CLASS_ORDER:
            item = profile.get(cls, {})
            if not item.get("n"):
                continue
            lead_steps = max(1, item["low_speed_lead_steps"])
            nolead_steps = max(1, item["low_speed_nolead_steps"])
            lines.append(
                f"| rc={tier:g} | {cls} | {item['n']} | "
                f"{_fmt(item['low_speed_lead_sum_mean'])}/"
                f"{_fmt((item['low_speed_lead_sum_mean'] or 0.0) / lead_steps)} | "
                f"{_fmt(item['low_speed_nolead_sum_mean'])}/"
                f"{_fmt((item['low_speed_nolead_sum_mean'] or 0.0) / nolead_steps)} | "
                f"{item['low_speed_lead_steps']} | {item['low_speed_nolead_steps']} |"
            )
    lines.append("")

    lines.append("## 6. 逐项贡献 top（审计样本；各类 mean）\n")
    lines.append("| 类 | 项贡献 mean（按 |mean| 降序） |")
    lines.append("|---|---|")
    for cls in CLASS_ORDER:
        item = result["tiers"][str(DEFAULT_TIERS[0])]["audit_profile"].get(cls, {})
        if not item.get("n"):
            continue
        terms = item["term_means"]
        top = sorted(terms.items(), key=lambda kv: abs(kv[1] or 0.0), reverse=True)[:8]
        lines.append(f"| {cls} | " + "; ".join(f"`{name}`={_fmt(value)}" for name, value in top) + " |")
    lines.append("")

    lines.append("## 7. 未决问题 / 说明\n")
    for item in result["meta"].get("notes", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _config_draft(tier: float, terminal_values: Mapping[str, float]) -> str:
    terms = [
        ("route_completion", f"{{name: route_completion, weight: {tier:g}, gamma: 1.0}}"),
        ("speed_ratio", "{name: speed_ratio, weight: 0.4, cap: 1.0}"),
        ("low_speed", "{name: low_speed, weight: -0.2}"),
        ("comfort_lon", "{name: comfort_lon, weight: -0.05, deadband: 2.5}"),
        ("comfort_lat", "{name: comfort_lat, weight: -0.05, deadband: 2.0}"),
        ("comfort_jerk", "{name: comfort_jerk, weight: -0.005, deadband: 5.0}"),
        ("solid_line", "{name: solid_line, weight: -2.0}"),
        ("speed_limit", "{name: speed_limit, weight: -5.0, tolerance: 0.05}"),
        ("crash", "{name: crash, weight: -10.0}"),
        ("out_of_road", "{name: out_of_road, weight: -8.0}"),
    ]
    lines = [
        f"# P2 审计配置草案：rc 档 = {tier:g}（v5 剖面 C；终局值反解自审计工具）",
        "# 用法：合并进 config/train.yaml 的 stages.C，或按 CLI --reward-term-weight route_completion=<档> 覆盖权重；",
        "# 终局值经 stages.C.reward.aggregation.terminal_values 透传（AggregationConfig）。",
        "stages:",
        "  C:",
        "    reward:",
        "      terms:",
    ]
    lines += [f"        - {text}" for _, text in terms]
    lines.append("      aggregation:")
    lines.append("        terminal_values:")
    for key in ("arrive_dest", "collision", "out_of_road", "max_step", "error"):
        lines.append(f"          {key}: {terminal_values.get(key, 0.0):g}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 5) analyze（分区 + 重放 + 反解 + 报告）
# --------------------------------------------------------------------------- #
def _partition(
    episodes: Sequence[Mapping[str, Any]],
    *,
    split_seed: int = DEFAULT_SPLIT_SEED,
    swap_ab: bool = False,
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    """按终局类**分层随机**划分审计/反解（互斥、seed 可复现、与输入顺序无关）。

    - 每类先按 (id, seed) 排序，再用由 `split_seed` + 类名派生的 RNG shuffle，交替分配；
    - 替代 P2 的"按序号交替"（两半难度不可交换，Gate2 发现 ③）；
    - `swap_ab=True` 交换两半（A/B 互换交叉验证：两向各跑一次）。
    """
    audit: List[Mapping[str, Any]] = []
    solve: List[Mapping[str, Any]] = []
    for cls in CLASS_ORDER:
        rows = sorted(
            (episode for episode in episodes if str(episode["meta"].get("termination")) == cls),
            key=lambda item: (int(item["meta"].get("spec_id", -1)), int(item["meta"].get("spec_seed", -1))),
        )
        random.Random(f"{int(split_seed)}:{cls}").shuffle(rows)
        for index, episode in enumerate(rows):
            (audit if index % 2 == 0 else solve).append(episode)
    if swap_ab:
        audit, solve = solve, audit
    return audit, solve


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    from reward_model import default_terminal_values

    episodes_dir = Path(args.episodes_dir).resolve()
    episodes_all = _existing_outcomes(episodes_dir)
    if not episodes_all:
        raise SystemExit(f"未找到采集 episode：{episodes_dir}")
    exclude_path = _resolve_exclude(args)
    excluded = _load_excluded(exclude_path)
    episodes = _filter_excluded(episodes_all, excluded)
    if not episodes:
        raise SystemExit(f"排除后无可用 episode：{episodes_dir}（exclude={exclude_path}）")
    split_seed = int(getattr(args, "split_seed", DEFAULT_SPLIT_SEED))
    swap_ab = bool(getattr(args, "swap_ab", False))
    min_per_class = int(getattr(args, "min_per_class", MIN_PER_CLASS))
    audit_set, solve_set = _partition(episodes, split_seed=split_seed, swap_ab=swap_ab)
    tiers = [float(value) for value in str(args.tiers).split(",") if value.strip()]

    provisional = default_terminal_values()
    collect_summary: Dict[str, Any] = {}
    summary_path = episodes_dir.parent / "collect_summary.json"
    if summary_path.is_file():
        try:
            collect_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            collect_summary = {}
    result: Dict[str, Any] = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tool": "tools/reward_audit.py",
            "head_commit": _head_commit(),
            "episodes_dir": str(episodes_dir),
            "pre_snapshot": collect_summary.get("pre_snapshot"),
            "ckpt": collect_summary.get("ckpt"),
            "ckpt_sha256": collect_summary.get("ckpt_sha256"),
            "pool": collect_summary.get("pool"),
            "exclude": str(exclude_path) if exclude_path else None,
            "exclude_pairs": len(excluded),
            "collected": len(episodes),
            "collected_raw": len(episodes_all),
            "excluded_episodes": len(episodes_all) - len(episodes),
            "class_counts": _class_counts(episodes),
            "split_seed": split_seed,
            "swap_ab": swap_ab,
            "min_per_class": min_per_class,
            "tiers": tiers,
            "audit": {"n": len(audit_set)},
            "solve": {"n": len(solve_set)},
            "targets": TARGETS,
            "gamma": GAMMA,
            "tolerance": TOLERANCE,
            "profile_tolerance": PROFILE_TOLERANCE,
            "provisional_terminal_values": provisional,
            "notes": [],
        },
        "audit": {"n": len(audit_set)},
        "solve": {"n": len(solve_set)},
        "tiers": {},
    }

    # 反解：每档用 solve 样本的稠密贡献 + 该档 rc 权重重放 → terminal_values
    for tier in tiers:
        solve_replays = [
            _replay_episode(episode, rc_weight=tier, terminal_values=provisional)
            for episode in solve_set
        ]
        solve = _solve_terminal_values(solve_replays, TARGETS)
        terminal_values = {
            key: float(value) for key, value in provisional.items()
        }
        for cls in CLASS_ORDER:
            item = solve.get(cls, {})
            if item.get("resolved") is not None:
                terminal_values[cls] = float(item["resolved"])
        audit_replays = [
            _replay_episode(episode, rc_weight=tier, terminal_values=terminal_values)
            for episode in audit_set
        ]
        profile = _profile(audit_replays, TARGETS)
        tier_deltas = [
            abs(profile[cls]["delta"])
            for cls in CLASS_ORDER
            if profile.get(cls, {}).get("n") and profile[cls]["delta"] is not None
        ]
        tier_per_class = all(profile.get(cls, {}).get("n", 0) >= min_per_class for cls in CLASS_ORDER)
        result["tiers"][str(tier)] = {
            "rc_weight": tier,
            "terminal_values": terminal_values,
            "solve": solve,
            "audit_profile": profile,
            "profile_pass": bool(tier_deltas) and all(delta <= PROFILE_TOLERANCE for delta in tier_deltas),
            "per_class_pass": tier_per_class,
            "delta_max": max(tier_deltas) if tier_deltas else None,
            "audit_replays": [
                {
                    "spec_id": row.spec_id,
                    "seed": row.seed,
                    "cls": row.cls,
                    "total": row.total,
                    "discounted": row.discounted,
                }
                for row in audit_replays
            ],
        }
        del solve_replays  # 只保留聚合结果，控制 JSON 体积

    # 样本清单（审计/反解各自 id + hash）
    for label, rows in (("audit", audit_set), ("solve", solve_set)):
        manifest: Dict[str, List[Dict[str, Any]]] = {cls: [] for cls in CLASS_ORDER}
        for episode in sorted(
            rows,
            key=lambda item: (
                str(item["meta"].get("termination")),
                int(item["meta"].get("spec_id", -1)),
                int(item["meta"].get("spec_seed", -1)),
            ),
        ):
            cls = str(episode["meta"].get("termination"))
            if cls not in manifest:
                continue
            manifest[cls].append(
                {
                    "spec_id": int(episode["meta"].get("spec_id", -1)),
                    "seed": int(episode["meta"].get("spec_seed", -1)),
                    "ctx_sha256": str(episode["meta"].get("ctx_sha256", "")),
                    "file_sha256": _sha256_file(Path(str(episode["path"]))),
                    "total": None,  # 由重放结果回填
                }
            )
        # 回填 total（基准档的定稿终局值口径）
        base = result["tiers"][str(tiers[0])]
        base_terminal = base["terminal_values"]
        if label == "audit":
            totals = {(row["spec_id"], row["seed"]): row["total"] for row in base["audit_replays"]}
        else:
            solve_replays = [
                _replay_episode(episode, rc_weight=tiers[0], terminal_values=base_terminal)
                for episode in solve_set
            ]
            totals = {(row.spec_id, row.seed): row.total for row in solve_replays}
        for cls in CLASS_ORDER:
            for entry in manifest[cls]:
                entry["total"] = totals.get((entry["spec_id"], entry["seed"]))
        result[label]["manifest"] = manifest

    # 判定
    base_profile = result["tiers"][str(tiers[0])]["audit_profile"]
    per_class_pass = all(
        base_profile.get(cls, {}).get("n", 0) >= min_per_class for cls in CLASS_ORDER
    )
    deltas = [
        abs(base_profile[cls]["delta"])
        for cls in CLASS_ORDER
        if base_profile.get(cls, {}).get("n") and base_profile[cls]["delta"] is not None
    ]
    profile_pass = bool(deltas) and all(delta <= PROFILE_TOLERANCE for delta in deltas)
    result["meta"]["per_class_pass"] = per_class_pass
    result["meta"]["total_pass"] = len(audit_set) >= MIN_TOTAL and len(solve_set) >= MIN_TOTAL
    result["meta"]["profile_pass"] = profile_pass

    result["meta"]["notes"] = [
        f"审计/反解样本按终局类分层随机划分（split_seed={split_seed}，swap_ab={swap_ab}；互斥）；"
        "每 episode 记录 id/seed/ctx_sha256/文件 sha256。",
        "max_step 类：rollout 循环在 max_steps 截断时 env 未置 info.max_step（与 eval 分类同口径按步数），"
        "重放时对终局帧补 max_step=True 以结算终局值；**E-β″ 复算须在训练侧 max_step 接线后执行**。",
        "rc=1.0 为代码默认基准档（回填 default_terminal_values）；3/10/30 为 P4 扫档（每档一套终局值，不得跨档复用）。",
        "dense 语义两列：pos×mult+neg（真实聚合器口径，用于反解）与 pos+neg（忽略 CaRL 乘子，仅对照）。",
        "E-β′ vs E-β″：本报告 = E-β′（旧架构，P2/Gate2，池含 eval500 重叠，见 docs/rl_reward_v5.md §7）；"
        "E-β″ 在 P3 重训后按 docs/v6_program_prereg.md §7.1 用同一工具复算（排除 eval500、分层随机、A/B 互换、"
        "每类 n≥50、max_step 接线后；P3 后、P4 前，作为 P4 奖励口径）。",
    ]
    if excluded:
        result["meta"]["notes"].append(
            f"排除集生效：{exclude_path}（{len(excluded)} 个 (id, seed)）；"
            f"原始采集 {len(episodes_all)} → 过滤后 {len(episodes)}（剔除 {len(episodes_all) - len(episodes)}）。"
        )
    if not result["meta"]["class_counts"].get("error"):
        result["meta"]["notes"].append("error 类 n=0（不在剖面目标内；按 §7 只报 n 与区间）。")
    if not result["meta"]["total_pass"]:
        result["meta"]["notes"].append(
            f"样本量不足：审计 {len(audit_set)} / 反解 {len(solve_set)}（目标 ≥ {MIN_TOTAL}）→ 需补采。"
        )
    if not per_class_pass:
        result["meta"]["notes"].append(
            f"分类样本不足（每类目标 ≥ {min_per_class}）："
            + ", ".join(f"{cls}={base_profile.get(cls, {}).get('n', 0)}" for cls in CLASS_ORDER)
            + " → 该类只报 n 与区间，不做达标判定。"
        )

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reward_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    markdown = _render_markdown(result)
    (out_dir / "reward_audit.md").write_text(markdown, encoding="utf-8")
    for tier in tiers:
        draft = _config_draft(tier, result["tiers"][str(tier)]["terminal_values"])
        (out_dir / f"config_draft_rc{tier:g}.yaml").write_text(draft, encoding="utf-8")
    if args.mirror_md:
        Path(args.mirror_md).write_text(markdown, encoding="utf-8")
    print(
        f"[audit] analyze 完成 → {out_dir}/reward_audit.md（审计 n={len(audit_set)}，"
        f"反解 n={len(solve_set)}，profile_pass={profile_pass}）",
        flush=True,
    )
    return result


# --------------------------------------------------------------------------- #
# 6) dry-run（合成小样本端到端）
# --------------------------------------------------------------------------- #
def _synthetic_episode(cls: str, *, spec_id: int, steps: int = 8) -> Dict[str, Any]:
    """合成 episode：密集项 + 终局类；用于 dry-run（n≥2/类）与口径单测。"""
    records = []
    for index in range(steps):
        last = index == steps - 1
        ctx: Dict[str, Any] = {
            "speed": 5.0,
            "speed_limit_mps": 8.0,
            "speed_ratio": 0.625,
            "route_completion": 1.0 * index / steps,
            "a_lon": 0.1,
            "a_lat": 0.1,
            "jerk": 0.1,
            "lead_gap_m": 20.0,
            "lead_speed_mps": 5.0,
            "lane_half_width_m": 1.75,
            "d_lat": 0.1,
            "done": last,
            "__speed_limit_source": "info.lane_speed_limit_mps",
        }
        if last:
            if cls == "arrive_dest":
                ctx["arrive_dest"] = True
            elif cls == "collision":
                ctx["collision"] = True
            elif cls == "out_of_road":
                ctx["out_of_road"] = True
            elif cls == "max_step":
                ctx["max_step"] = False  # 模拟 rollout 截断（重放注入 True）
        records.append({"policy_step": index, "frame": index * 5, "ctx": ctx})
    termination = "max_step" if cls == "max_step" else cls
    return {
        "path": None,
        "meta": {
            "spec_id": spec_id,
            "spec_seed": 5000000 + spec_id,
            "termination": termination,
            "steps": steps * 5,
            "policy_steps": steps,
            "ctx_sha256": _sha256_text(json.dumps(records, sort_keys=True)),
        },
        "steps": records,
    }


def dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    episodes: List[Dict[str, Any]] = []
    for index, cls in enumerate(CLASS_ORDER):
        for repeat in range(3):  # 每类 3 条（≥2）
            episodes.append(_synthetic_episode(cls, spec_id=index * 10 + repeat))
    # 用一个临时目录承载合成 episode（manifest 的 file_sha256 为 None）
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = out_dir / "synthetic_episodes"
    episodes_dir.mkdir(exist_ok=True)
    for episode in episodes:
        path = episodes_dir / f"id{episode['meta']['spec_id']}_seed{episode['meta']['spec_seed']}.json"
        path.write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
        episode["path"] = str(path)
    ns = argparse.Namespace(
        episodes_dir=str(episodes_dir),
        out=str(out_dir),
        tiers="1,3,10,30",
        mirror_md=None,
        exclude=None,  # 合成样本不排除
        no_exclude=True,
        split_seed=DEFAULT_SPLIT_SEED,
        swap_ab=False,
        min_per_class=MIN_PER_CLASS,
    )
    result = analyze(ns)
    print(f"[audit] dry-run OK：{out_dir}/reward_audit.md（每类 3 条合成样本）", flush=True)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P2 奖励审计（采集 + 重放 + 终局值反解）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_snap = sub.add_parser("snapshot", help="创建/校验 pre-v6 代码快照")
    p_snap.add_argument("--pre-root", default=str(DEFAULT_PRE_ROOT))
    p_snap.add_argument("--pre-commit", default=DEFAULT_PRE_COMMIT)

    p_collect = sub.add_parser("collect", help="pre-v6 快照下采集 rollout（分层补足）")
    p_collect.add_argument("--pre-root", default=str(DEFAULT_PRE_ROOT))
    p_collect.add_argument("--pre-commit", default=DEFAULT_PRE_COMMIT)
    p_collect.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    p_collect.add_argument("--pool", default=str(DEFAULT_POOL))
    p_collect.add_argument(
        "--exclude", default=str(DEFAULT_EXCLUDE), help="排除的场景 (id, seed) 文件（默认 eval500）"
    )
    p_collect.add_argument(
        "--no-exclude", action="store_true", help="显式关闭排除（默认排除 eval500；空值 --exclude '' 会报错）"
    )
    p_collect.add_argument("--out-dir", default=str(ROOT / "runs/reward_audit/episodes"))
    p_collect.add_argument("--workers", type=int, default=6)
    p_collect.add_argument("--batch", type=int, default=40, help="每批 episode 数（批后检查分层配额）")
    p_collect.add_argument("--max-steps", type=int, default=1000)
    p_collect.add_argument("--per-class-target", type=int, default=20)
    p_collect.add_argument("--max-episodes", type=int, default=0, help="0 = 不限制（跑完候选池）")
    p_collect.add_argument("--python", default=sys.executable)
    p_collect.add_argument("--config", default=str(ROOT / "config/default.yaml"))
    p_collect.add_argument("--tracker", default="lqr")
    p_collect.add_argument("--eval-reference", default="plan")
    p_collect.add_argument("--device", default="auto")

    p_analyze = sub.add_parser("analyze", help="HEAD 完整 v5 项集离线重放 + 反解 + 报告")
    p_analyze.add_argument("--episodes-dir", default=str(ROOT / "runs/reward_audit/episodes"))
    p_analyze.add_argument("--out", default=str(ROOT / "runs/reward_audit/report"))
    p_analyze.add_argument("--tiers", default="1,3,10,30")
    p_analyze.add_argument("--mirror-md", default=None)
    p_analyze.add_argument(
        "--exclude", default=str(DEFAULT_EXCLUDE), help="分析前剔除的场景 (id, seed) 文件（默认 eval500）"
    )
    p_analyze.add_argument(
        "--no-exclude", action="store_true", help="显式关闭排除（默认排除 eval500）"
    )
    p_analyze.add_argument(
        "--split-seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help="审计/反解分层随机划分 seed（固定记录，可复现）",
    )
    p_analyze.add_argument(
        "--swap-ab", action="store_true", help="交换审计/反解两半（A/B 互换交叉验证）"
    )
    p_analyze.add_argument(
        "--min-per-class",
        type=int,
        default=MIN_PER_CLASS,
        help=f"每类最小样本数（默认 {MIN_PER_CLASS}；E-β″ 复算用 50）",
    )

    p_dry = sub.add_parser("dry-run", help="合成小样本端到端自检（n≥2/类）")
    p_dry.add_argument("--out", default=str(ROOT / "runs/reward_audit/dry_run"))

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "snapshot":
        print(json.dumps(ensure_snapshot(Path(args.pre_root), args.pre_commit), ensure_ascii=False))
        return 0
    if args.command == "collect":
        collect(args)
        return 0
    if args.command == "analyze":
        analyze(args)
        return 0
    if args.command == "dry-run":
        dry_run(args)
        return 0
    raise SystemExit(f"未知命令 {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
