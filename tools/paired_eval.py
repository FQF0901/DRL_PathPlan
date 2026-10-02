#!/usr/bin/env python3
"""v7-P0 配对评测协议工具：同场景配对 McNemar + bootstrap CI + 多 run 汇总。

用途
----
输入两组（或多组）``episodes.csv``（``pipeline/eval_runner.py`` 输出；同 spec、同
``(id, seed)`` 场景集），按场景配对，输出：

- **2×2 翻牌矩阵**：``fixed``（基线失败 → agent 成功）/ ``broken``（基线成功 → agent 失败）
  / ``both_pass`` / ``both_fail``；
- ``net = fixed − broken``、``z = |net| / √(fixed + broken)``（与
  ``docs/v6_program_prereg.md`` §7.3 / ``docs/rl_stage_c_experiments.md`` 同口径）；
- **McNemar 精确检验 p**（二项双尾精确法，无连续性校正）；
- **配对差 95% CI**（按场景 bootstrap；默认 10000 次、固定 seed 可复现）；
- **分层**（几何 primary × 难度，及 primary / difficulty 边际）配对差；
- **分项**（collision / off-road / max_step）配对差；
- **多 run 汇总**（多 seed：run 级均值 / 中位数 / IQM / min / max + bootstrap 区间）。

方向约定：``success`` 越高越好；``collision`` / ``off_road`` / ``max_step`` 越低越好。
事件项的 ``net > 0`` 表示 agent 事件更多（更差），报告内逐项标注。

选点纪律（v7 预注册 §4）：本工具只做**评估**，不参与选点；测试集（eval500）只应评估一次。

与现有工具的关系
----------------
已核对 ``tools/reward_audit.py``、``tools/il_report.py`` 等现有工具：仓库内此前**没有**可
复用的配对统计函数（fixed/broken/net/z 只在文档与一次性脚本中出现）；本工具按上述冻结
口径独立实现，不改动既有调用点。

用法
----
    tools/venv-python tools/paired_eval.py --baseline RUN/episodes.csv \\
        --agent RUN_A/episodes.csv RUN_B/episodes.csv RUN_C/episodes.csv \\
        --out-dir runs/paired_eval/v7_p0
    # --baseline 可给多个 run（基线多 evaluation seed），与每个 agent run 交叉配对；
    # 多个 baseline 时主结论应 pin 单一 baseline run（交叉配对用于基线稳定性检查）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BOOTSTRAP = 10000
DEFAULT_SEED = 20261002
DEFAULT_ALPHA = 0.05
#: 分层判定最小 n（与 ``pipeline/eval_runner.py::PRIMARY_GROUP_MIN_N`` 同口径；n < 只报不判）
DEFAULT_MIN_STRATUM = 30
#: 单 run 参考判据（v7 计划 §1：Δ ≥ +3pt 且 z ≥ 1.96）
POSITIVE_DELTA_PP = 3.0
POSITIVE_Z = 1.96
#: 安全闸（v7 预注册 §5）：绝对上限 / 相对 IDM 允许增量
COLLISION_GATE_ABS = 0.10
OFF_ROAD_GATE_ABS = 0.10
OFF_ROAD_GATE_MARGIN_PP = 2.0

TRUTHY = frozenset({"true", "1", "yes", "y", "t"})
FALSY = frozenset({"", "false", "0", "no", "n", "f", "none", "nan", "null"})
REQUIRED_COLUMNS = ("id", "seed", "success", "collision", "off_road", "termination")

#: 事件名 → 方向（报告/JSON 逐项标注）
EVENTS: Dict[str, str] = {
    "success": "higher_is_better",
    "collision": "lower_is_better",
    "off_road": "lower_is_better",
    "max_step": "lower_is_better",
}

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_MIN_STRATUM",
    "DEFAULT_SEED",
    "EVENTS",
    "Episode",
    "aggregate_runs",
    "analyze",
    "analyze_pair",
    "bootstrap_mean_ci",
    "iqm",
    "load_episodes_csv",
    "main",
    "mcnemar_exact_p",
    "pair_episodes",
    "paired_stats",
    "render_markdown",
]


# --------------------------------------------------------------------------- #
# 基础：读取 episodes.csv / Episode 记录
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Episode:
    """``episodes.csv`` 一行的解析结果（只保留配对/分层所需字段）。"""

    spec_id: int
    seed: int
    split: str
    difficulty: str
    primary: str
    geometry: Tuple[str, ...]
    success: bool
    collision: bool
    off_road: bool
    termination: str
    route_completion: Optional[float]
    steps: Optional[int]
    policy: str
    error: str

    @property
    def key(self) -> Tuple[int, int]:
        return (self.spec_id, self.seed)

    def has_event(self, event: str) -> bool:
        if event == "success":
            return self.success
        if event == "collision":
            return self.collision
        if event == "off_road":
            return self.off_road
        if event == "max_step":
            return self.termination == "max_step"
        raise KeyError(f"未知事件：{event!r}（可选 {sorted(EVENTS)}）")


def _parse_bool(text: Any, *, column: str, row_no: int) -> bool:
    value = str(text if text is not None else "").strip().lower()
    if value in TRUTHY:
        return True
    if value in FALSY:
        return False
    raise SystemExit(f"episodes.csv 第 {row_no} 行 {column} 非布尔值：{text!r}")


def _opt_float(text: Any) -> Optional[float]:
    value = str(text if text is not None else "").strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _opt_int(text: Any) -> Optional[int]:
    number = _opt_float(text)
    return None if number is None else int(number)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _head_commit(repo: Path = ROOT) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except Exception:  # noqa: BLE001 - 记录性字段，不阻塞分析
        return None
    text = proc.stdout.strip()
    return text if proc.returncode == 0 and text else None


def load_episodes_csv(path: Path) -> Dict[Tuple[int, int], Episode]:
    """读取 ``episodes.csv`` → ``{(id, seed): Episode}``（fail-closed）。

    - 必需列缺失 / 布尔值非法 / 同一文件内 ``(id, seed)`` 重复 → ``SystemExit``；
    - 空行跳过；``geometry`` 为 ``;`` 连接的多标签。
    """
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"episodes.csv 不存在：{path}")
    out: Dict[Tuple[int, int], Episode] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        missing = [name for name in REQUIRED_COLUMNS if name not in fields]
        if missing:
            raise SystemExit(f"{path} 缺少必需列：{missing}（现有列：{fields}）")
        for row_no, row in enumerate(reader, start=2):
            if not any(str(value or "").strip() for value in row.values()):
                continue
            try:
                spec_id = int(float(str(row["id"]).strip()))
                seed = int(float(str(row["seed"]).strip()))
            except ValueError as exc:
                raise SystemExit(f"{path} 第 {row_no} 行 id/seed 非数值：{exc}") from exc
            key = (spec_id, seed)
            if key in out:
                raise SystemExit(f"{path} 内 (id, seed)={key} 重复（配对键必须唯一）")
            geometry = tuple(
                part.strip() for part in str(row.get("geometry") or "").split(";") if part.strip()
            )
            out[key] = Episode(
                spec_id=spec_id,
                seed=seed,
                split=str(row.get("split") or "").strip(),
                difficulty=str(row.get("difficulty") or "").strip() or "unlabeled",
                primary=str(row.get("primary") or "").strip() or "unlabeled",
                geometry=geometry,
                success=_parse_bool(row.get("success"), column="success", row_no=row_no),
                collision=_parse_bool(row.get("collision"), column="collision", row_no=row_no),
                off_road=_parse_bool(row.get("off_road"), column="off_road", row_no=row_no),
                termination=str(row.get("termination") or "").strip() or "other",
                route_completion=_opt_float(row.get("route_completion")),
                steps=_opt_int(row.get("steps")),
                policy=str(row.get("policy") or "").strip(),
                error=str(row.get("error") or "").strip(),
            )
    if not out:
        raise SystemExit(f"episodes.csv 无数据行：{path}")
    return out


def pair_episodes(
    baseline: Mapping[Tuple[int, int], Episode],
    agent: Mapping[Tuple[int, int], Episode],
    *,
    allow_mismatch: bool = False,
) -> Tuple[List[Tuple[Episode, Episode]], Dict[str, Any]]:
    """按 ``(id, seed)`` 配对两组 episode（fail-closed）。

    - 键集合不一致：默认 ``SystemExit``；``allow_mismatch=True`` 时取交集并在 meta 记录缺失数；
    - 同键的 ``primary`` / ``difficulty`` / ``split`` 不一致 → ``SystemExit``（spec 口径不一致）。
    """
    baseline_keys, agent_keys = set(baseline), set(agent)
    common = sorted(baseline_keys & agent_keys)
    baseline_only = sorted(baseline_keys - agent_keys)
    agent_only = sorted(agent_keys - baseline_keys)
    info = {
        "n_common": len(common),
        "n_baseline_only": len(baseline_only),
        "n_agent_only": len(agent_only),
        "baseline_only_examples": baseline_only[:5],
        "agent_only_examples": agent_only[:5],
        "allow_mismatch": bool(allow_mismatch),
    }
    if (baseline_only or agent_only) and not allow_mismatch:
        raise SystemExit(
            "配对键不一致（fail-closed）："
            f"baseline 独有 {len(baseline_only)} 条、agent 独有 {len(agent_only)} 条；"
            f"示例 baseline={baseline_only[:3]} agent={agent_only[:3]}。"
            "确需交集配对请显式 --allow-mismatch（记录缺失数）。"
        )
    mismatched: List[str] = []
    for key in common:
        base_ep, agent_ep = baseline[key], agent[key]
        if (base_ep.primary, base_ep.difficulty, base_ep.split) != (
            agent_ep.primary,
            agent_ep.difficulty,
            agent_ep.split,
        ):
            mismatched.append(f"{key[0]}:{key[1]}")
    if mismatched:
        raise SystemExit(
            f"同键场景标签不一致（spec 口径不一致）：{len(mismatched)} 条，示例 {mismatched[:5]}"
        )
    rows = [(baseline[key], agent[key]) for key in common]
    return rows, info


# --------------------------------------------------------------------------- #
# 统计：McNemar 精确检验 / bootstrap CI / IQM
# --------------------------------------------------------------------------- #
def mcnemar_exact_p(fixed: int, broken: int) -> float:
    """McNemar 精确检验（二项双尾，p=0.5，无连续性校正）；``n=0`` 返回 1.0。

    等价于 ``scipy.stats.binomtest(min(fixed, broken), fixed + broken, 0.5).pvalue``
    （单侧尾概率 ×2 截断到 1），不引入 scipy 依赖。
    """
    fixed, broken = int(fixed), int(broken)
    n = fixed + broken
    if n <= 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / float(2**n)
    return min(1.0, 2.0 * tail)


def _derive_seed(base_seed: int, tag: str) -> int:
    """由 base seed + 标签派生确定性 bootstrap seed（跨进程/顺序可复现）。"""
    return (int(base_seed) * 1_000_003 + zlib.crc32(tag.encode("utf-8"))) & 0x7FFFFFFF


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[Optional[float], Optional[float]]:
    """均值的 bootstrap 百分位 CI（按元素重采样；固定 seed 可复现）。"""
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0:
        return (None, None)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), dtype=np.float64)
    chunk = max(1, int(2_000_000 // max(1, array.size)))
    for start in range(0, int(n_boot), chunk):
        stop = min(int(n_boot), start + chunk)
        index = rng.integers(0, array.size, size=(stop - start, array.size))
        means[start:stop] = array[index].mean(axis=1)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return (float(lo), float(hi))


def iqm(values: Sequence[float]) -> Optional[float]:
    """Interquartile mean（rliable 口径：排序后两侧各裁 ``int(n*0.25)`` 再取均值）。"""
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    n = len(finite)
    if n == 0:
        return None
    trim = int(n * 0.25)
    core = finite[trim : n - trim] if n - 2 * trim > 0 else finite
    return float(sum(core) / len(core))


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(finite) / len(finite)) if finite else None


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    finite = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    n = len(finite)
    if n == 0:
        return None
    mid = n // 2
    return finite[mid] if n % 2 else (finite[mid - 1] + finite[mid]) / 2.0


def paired_stats(
    rows: Sequence[Tuple[Episode, Episode]],
    event: str,
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    tag: str = "",
) -> Dict[str, Any]:
    """单事件配对统计：2×2、net/z、McNemar 精确 p、配对差 + bootstrap 95% CI。"""
    if event not in EVENTS:
        raise KeyError(f"未知事件：{event!r}（可选 {sorted(EVENTS)}）")
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "fixed": 0,
            "broken": 0,
            "both_pass": 0,
            "both_fail": 0,
            "net": 0,
            "z": None,
            "mcnemar_exact_p": 1.0,
            "rate_base": None,
            "rate_agent": None,
            "delta": None,
            "delta_pp": None,
            "ci95": [None, None],
            "ci95_pp": [None, None],
            "direction": EVENTS[event],
        }
    base = np.asarray([1 if base_ep.has_event(event) else 0 for base_ep, _ in rows], dtype=np.int64)
    agent = np.asarray([1 if agent_ep.has_event(event) else 0 for _, agent_ep in rows], dtype=np.int64)
    fixed = int(np.sum((base == 0) & (agent == 1)))
    broken = int(np.sum((base == 1) & (agent == 0)))
    both_pass = int(np.sum((base == 1) & (agent == 1)))
    both_fail = int(np.sum((base == 0) & (agent == 0)))
    net = fixed - broken
    denominator = fixed + broken
    z = abs(net) / math.sqrt(denominator) if denominator > 0 else None
    diffs = (agent - base).astype(np.float64)
    delta = float(diffs.mean())
    lo, hi = bootstrap_mean_ci(
        diffs, n_boot=n_boot, seed=_derive_seed(seed, f"{tag}|{event}"), alpha=alpha
    )
    return {
        "n": n,
        "fixed": fixed,
        "broken": broken,
        "both_pass": both_pass,
        "both_fail": both_fail,
        "net": net,
        "z": z,
        "mcnemar_exact_p": mcnemar_exact_p(fixed, broken),
        "rate_base": float(base.mean()),
        "rate_agent": float(agent.mean()),
        "delta": delta,
        "delta_pp": delta * 100.0,
        "ci95": [lo, hi],
        "ci95_pp": [None if lo is None else lo * 100.0, None if hi is None else hi * 100.0],
        "direction": EVENTS[event],
    }


def _strata_stats(
    rows: Sequence[Tuple[Episode, Episode]],
    key_fn,
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    min_stratum: int,
    tag: str,
) -> Dict[str, Any]:
    groups: Dict[str, List[Tuple[Episode, Episode]]] = {}
    for base_ep, agent_ep in rows:
        groups.setdefault(str(key_fn(base_ep, agent_ep)), []).append((base_ep, agent_ep))
    out: Dict[str, Any] = {}
    for name in sorted(groups):
        stats = paired_stats(
            groups[name], "success", n_boot=n_boot, seed=seed, alpha=alpha, tag=f"{tag}|{name}"
        )
        stats["group"] = name
        stats["judged"] = bool(stats["n"] >= int(min_stratum))
        out[name] = stats
    return out


def analyze_pair(
    baseline: Mapping[Tuple[int, int], Episode],
    agent: Mapping[Tuple[int, int], Episode],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    min_stratum: int = DEFAULT_MIN_STRATUM,
    allow_mismatch: bool = False,
) -> Dict[str, Any]:
    """单对（baseline run × agent run）完整分析：总体 + 分项 + 分层 + 单 run 判读。"""
    rows, pair_info = pair_episodes(baseline, agent, allow_mismatch=allow_mismatch)
    result: Dict[str, Any] = {
        "pair": pair_info,
        "n_pairs": len(rows),
        "n_error": {
            "baseline": sum(1 for ep in baseline.values() if ep.error),
            "agent": sum(1 for ep in agent.values() if ep.error),
        },
        "items": {},
        "strata": {},
    }
    for event in EVENTS:
        result["items"][event] = paired_stats(
            rows, event, n_boot=n_boot, seed=seed, alpha=alpha, tag=f"items|{event}"
        )
    result["strata"] = {
        "primary_x_difficulty": _strata_stats(
            rows,
            lambda b, a: f"{b.primary}×{b.difficulty}",
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            min_stratum=min_stratum,
            tag="pxd",
        ),
        "by_primary": _strata_stats(
            rows,
            lambda b, a: b.primary,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            min_stratum=min_stratum,
            tag="primary",
        ),
        "by_difficulty": _strata_stats(
            rows,
            lambda b, a: b.difficulty,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
            min_stratum=min_stratum,
            tag="difficulty",
        ),
    }
    success = result["items"]["success"]
    ci_pp = success["ci95_pp"]
    result["verdict"] = {
        "delta_pp": success["delta_pp"],
        "z": success["z"],
        "mcnemar_exact_p": success["mcnemar_exact_p"],
        "ci95_pp": ci_pp,
        "ci_lower_gt_zero": (ci_pp[0] > 0.0) if ci_pp[0] is not None else None,
        "positive_3pt_z196": bool(
            success["delta_pp"] is not None
            and success["delta_pp"] >= POSITIVE_DELTA_PP
            and success["z"] is not None
            and success["z"] >= POSITIVE_Z
        ),
    }
    return result


# --------------------------------------------------------------------------- #
# 多 run 汇总（多 seed：均值 / 中位数 / IQM / 区间）
# --------------------------------------------------------------------------- #
#: 汇总指标：JSON 路径 → 展示名
AGGREGATE_METRICS: Dict[str, Tuple[str, ...]] = {
    "agent_success_rate": ("items", "success", "rate_agent"),
    "baseline_success_rate": ("items", "success", "rate_base"),
    "delta_pp": ("items", "success", "delta_pp"),
    "net": ("items", "success", "net"),
    "z": ("items", "success", "z"),
    "mcnemar_exact_p": ("items", "success", "mcnemar_exact_p"),
    "collision_rate": ("items", "collision", "rate_agent"),
    "collision_delta_pp": ("items", "collision", "delta_pp"),
    "off_road_rate": ("items", "off_road", "rate_agent"),
    "off_road_delta_pp": ("items", "off_road", "delta_pp"),
    "max_step_delta_pp": ("items", "max_step", "delta_pp"),
}


def _dig(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _aggregate_metric(
    values: Sequence[Optional[float]],
    *,
    n_boot: int,
    seed: int,
    alpha: float,
    tag: str,
) -> Dict[str, Any]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    lo, hi = bootstrap_mean_ci(
        finite, n_boot=n_boot, seed=_derive_seed(seed, f"agg|{tag}"), alpha=alpha
    )
    return {
        "n": len(finite),
        "values": finite,
        "mean": _mean(finite),
        "median": _median(finite),
        "iqm": iqm(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "ci95": [lo, hi],
    }


def aggregate_runs(
    run_pairs: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """多 run 汇总：run 级指标 → 均值/中位数/IQM/min/max + bootstrap 95% CI（按 run 重采样）。"""
    metrics: Dict[str, Any] = {}
    for name, path in AGGREGATE_METRICS.items():
        values = [_dig(run_pair, path) for run_pair in run_pairs]
        metrics[name] = _aggregate_metric(
            values, n_boot=n_boot, seed=seed, alpha=alpha, tag=name
        )
    delta = metrics["delta_pp"]
    collision = metrics["collision_delta_pp"]
    collision_rate = metrics["collision_rate"]
    off_road = metrics["off_road_delta_pp"]
    off_road_rate = metrics["off_road_rate"]

    def _gate(rate: Optional[float], delta_pp: Optional[float], *, abs_limit: float, margin_pp: float) -> Dict[str, Any]:
        passed: Optional[bool] = None
        if rate is not None or delta_pp is not None:
            passed = bool(
                (rate is not None and rate <= abs_limit)
                or (delta_pp is not None and delta_pp <= margin_pp)
            )
        return {
            "agent_rate_mean": rate,
            "delta_pp_mean": delta_pp,
            "absolute_limit": abs_limit,
            "margin_pp": margin_pp,
            "passed": passed,
        }

    ci_lo = delta["ci95"][0]
    verdict = {
        "n_runs": len(run_pairs),
        "mean_delta_pp": delta["mean"],
        "median_delta_pp": delta["median"],
        "iqm_delta_pp": delta["iqm"],
        "ci95_pp": delta["ci95"],
        "primary_ci_lower_gt_zero": bool(ci_lo is not None and ci_lo > 0.0),
        "primary_iqm_positive": bool(delta["iqm"] is not None and delta["iqm"] > 0.0),
        "positive_runs": sum(
            1
            for run_pair in run_pairs
            if bool(_dig(run_pair, ("verdict", "positive_3pt_z196")))
        ),
        "collision_gate": _gate(
            collision_rate["mean"], collision["mean"], abs_limit=COLLISION_GATE_ABS, margin_pp=0.0
        ),
        "off_road_gate": _gate(
            off_road_rate["mean"],
            off_road["mean"],
            abs_limit=OFF_ROAD_GATE_ABS,
            margin_pp=OFF_ROAD_GATE_MARGIN_PP,
        ),
    }
    return {"metrics": metrics, "verdict": verdict}


def analyze(
    baseline_runs: Sequence[Mapping[str, Any]],
    agent_runs: Sequence[Mapping[str, Any]],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    min_stratum: int = DEFAULT_MIN_STRATUM,
    allow_mismatch: bool = False,
) -> Dict[str, Any]:
    """入口：多 baseline run × 多 agent run 交叉配对 + 汇总（agent 外层遍历）。"""
    run_pairs: List[Dict[str, Any]] = []
    for agent_run in agent_runs:
        for baseline_run in baseline_runs:
            pair = analyze_pair(
                baseline_run["episodes"],
                agent_run["episodes"],
                n_boot=n_boot,
                seed=seed,
                alpha=alpha,
                min_stratum=min_stratum,
                allow_mismatch=allow_mismatch,
            )
            pair["baseline"] = {
                "label": baseline_run["label"],
                "path": baseline_run["path"],
                "sha256": baseline_run["sha256"],
            }
            pair["agent"] = {
                "label": agent_run["label"],
                "path": agent_run["path"],
                "sha256": agent_run["sha256"],
            }
            run_pairs.append(pair)
    report = {
        "meta": {
            "tool": "tools/paired_eval.py",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "head_commit": _head_commit(),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "alpha": float(alpha),
            "min_stratum": int(min_stratum),
            "allow_mismatch": bool(allow_mismatch),
            "pairing": "同 (id, seed) 场景配对；同键 primary/difficulty/split 一致性校验（fail-closed）",
            "success_definition": "episodes.csv::success（= arrive_dest）",
            "net_z_definition": "net = fixed − broken；z = |net| / √(fixed + broken)（v6 §7.3）",
            "direction": dict(EVENTS),
            "baseline_runs": [
                {
                    "label": run["label"],
                    "path": run["path"],
                    "sha256": run["sha256"],
                    **run["summary"],
                }
                for run in baseline_runs
            ],
            "agent_runs": [
                {
                    "label": run["label"],
                    "path": run["path"],
                    "sha256": run["sha256"],
                    **run["summary"],
                }
                for run in agent_runs
            ],
        },
        "run_pairs": run_pairs,
        "aggregate": aggregate_runs(run_pairs, n_boot=n_boot, seed=seed, alpha=alpha),
    }
    return report


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def _fmt(value: Any, digits: int = 3, sign: bool = False) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:+.{digits}f}" if sign else f"{number:.{digits}f}"


def _fmt_ci(ci: Sequence[Any], digits: int = 2) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "—"
    return f"[{_fmt(ci[0], digits, sign=True)}, {_fmt(ci[1], digits, sign=True)}]"


def _fmt_p(value: Any) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.4f}" if number >= 1e-4 else f"{number:.2e}"


def _stats_row(label: str, stats: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {stats['n']} | {_fmt(stats['rate_base'])} | {_fmt(stats['rate_agent'])} | "
        f"{_fmt(stats['delta_pp'], 2, sign=True)} | {stats['fixed']} | {stats['broken']} | "
        f"{stats['both_pass']} | {stats['both_fail']} | {stats['net']:+d} | {_fmt(stats['z'], 2)} | "
        f"{_fmt_p(stats['mcnemar_exact_p'])} | {_fmt_ci(stats['ci95_pp'])} |"
    )


def render_markdown(report: Mapping[str, Any]) -> str:
    meta = report["meta"]
    aggregate = report["aggregate"]
    run_pairs = report["run_pairs"]
    lines: List[str] = []
    lines.append("# v7 配对评测报告（配对 McNemar + bootstrap CI）\n")
    lines.append(
        f"- 生成：{meta['generated_at']}（工具 `{meta['tool']}`；分析 HEAD commit `{meta['head_commit']}`）"
    )
    lines.append(
        f"- 配对口径：{meta['pairing']}；success = `{meta['success_definition']}`；`{meta['net_z_definition']}`"
    )
    lines.append(
        f"- bootstrap：n={meta['n_boot']}，seed={meta['seed']}，α={meta['alpha']:g}"
        f"（单对按场景重采样；多 run 汇总按 run 重采样）；分层判定最小 n={meta['min_stratum']}"
    )
    lines.append(
        "- 方向：success ↑ 好；collision / off_road / max_step ↓ 好（事件项 net > 0 = agent 事件更多 = 更差）\n"
    )
    lines.append("## 0. 结论\n")
    verdict = aggregate["verdict"]
    lines.append(
        f"- 单 run 参考判据（Δ ≥ +3pt 且 z ≥ 1.96）：**{verdict['positive_runs']}/{verdict['n_runs']}** run 满足"
    )
    lines.append(
        f"- 多 run 汇总（n={verdict['n_runs']}）：配对差均值 **{_fmt(verdict['mean_delta_pp'], 2, sign=True)}pt**"
        f"，95% CI **{_fmt_ci(verdict['ci95_pp'])}pt**，下界 > 0："
        f"**{'是' if verdict['primary_ci_lower_gt_zero'] else '否'}**；IQM "
        f"{_fmt(verdict['iqm_delta_pp'], 2, sign=True)}pt（> 0：{'是' if verdict['primary_iqm_positive'] else '否'}）"
    )
    collision_gate = verdict["collision_gate"]
    off_road_gate = verdict["off_road_gate"]
    lines.append(
        f"- 安全闸：collision 均值 {_fmt(collision_gate['agent_rate_mean'])} / Δ "
        f"{_fmt(collision_gate['delta_pp_mean'], 2, sign=True)}pt → "
        f"**{'通过' if collision_gate['passed'] else '未通过' if collision_gate['passed'] is False else '—'}**"
        f"（≤10% 或 Δ≤0）；off-road 均值 {_fmt(off_road_gate['agent_rate_mean'])} / Δ "
        f"{_fmt(off_road_gate['delta_pp_mean'], 2, sign=True)}pt → "
        f"**{'通过' if off_road_gate['passed'] else '未通过' if off_road_gate['passed'] is False else '—'}**"
        f"（≤10% 或 Δ≤+2pt）"
    )
    lines.append("")

    lines.append("## 1. 总体（success）\n")
    lines.append(
        "| agent run | baseline run | n | base succ | agent succ | Δpt | fixed | broken | both_pass | both_fail | net | z | McNemar p | 95% CI (Δpt) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run_pair in run_pairs:
        stats = run_pair["items"]["success"]
        lines.append(
            f"| {run_pair['agent']['label']} | {run_pair['baseline']['label']} | "
            + _stats_row("", stats).lstrip("| ").rstrip(" |")
            + " |"
        )
    lines.append("")

    lines.append("## 2. 分项配对差（collision / off-road / max_step）\n")
    lines.append(
        "| agent run | 项 | n | base | agent | Δpt | fixed | broken | net | McNemar p | 95% CI (Δpt) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for run_pair in run_pairs:
        for event in ("collision", "off_road", "max_step"):
            stats = run_pair["items"][event]
            lines.append(
                f"| {run_pair['agent']['label']} | {event} | {stats['n']} | "
                f"{_fmt(stats['rate_base'])} | {_fmt(stats['rate_agent'])} | "
                f"{_fmt(stats['delta_pp'], 2, sign=True)} | {stats['fixed']} | {stats['broken']} | "
                f"{stats['net']:+d} | {_fmt_p(stats['mcnemar_exact_p'])} | {_fmt_ci(stats['ci95_pp'])} |"
            )
    lines.append("")

    lines.append("## 3. 分层：几何（primary）× 难度\n")
    lines.append(
        "| agent run | 层 | n | base succ | agent succ | Δpt | fixed | broken | net | McNemar p | 95% CI (Δpt) | 判定 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for run_pair in run_pairs:
        for name, stats in run_pair["strata"]["primary_x_difficulty"].items():
            lines.append(
                f"| {run_pair['agent']['label']} | {name} | {stats['n']} | {_fmt(stats['rate_base'])} | "
                f"{_fmt(stats['rate_agent'])} | {_fmt(stats['delta_pp'], 2, sign=True)} | "
                f"{stats['fixed']} | {stats['broken']} | {stats['net']:+d} | "
                f"{_fmt_p(stats['mcnemar_exact_p'])} | {_fmt_ci(stats['ci95_pp'])} | "
                f"{'判定' if stats['judged'] else '仅报告'} |"
            )
    lines.append("")

    lines.append("### 3b. 边际分层（primary / difficulty）\n")
    lines.append("| agent run | 维度 | 层 | n | base succ | agent succ | Δpt | net | McNemar p |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for run_pair in run_pairs:
        for dimension in ("by_primary", "by_difficulty"):
            for name, stats in run_pair["strata"][dimension].items():
                lines.append(
                    f"| {run_pair['agent']['label']} | {dimension} | {name} | {stats['n']} | "
                    f"{_fmt(stats['rate_base'])} | {_fmt(stats['rate_agent'])} | "
                    f"{_fmt(stats['delta_pp'], 2, sign=True)} | {stats['net']:+d} | "
                    f"{_fmt_p(stats['mcnemar_exact_p'])} |"
                )
    lines.append("")

    lines.append("## 4. 多 run 汇总（run 级；均值 / 中位数 / IQM / 区间）\n")
    lines.append("| 指标 | n | mean | median | IQM | min | max | 95% CI |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, stats in aggregate["metrics"].items():
        lines.append(
            f"| {name} | {stats['n']} | {_fmt(stats['mean'], 3, sign=True)} | "
            f"{_fmt(stats['median'], 3, sign=True)} | {_fmt(stats['iqm'], 3, sign=True)} | "
            f"{_fmt(stats['min'], 3, sign=True)} | {_fmt(stats['max'], 3, sign=True)} | "
            f"{_fmt_ci(stats['ci95'], 3)} |"
        )
    lines.append("")

    lines.append("## 5. 说明 / 边界\n")
    lines.append(
        "- 配对完整性：默认 fail-closed（键集合必须一致、同键 primary/difficulty/split 一致）；"
        "`--allow-mismatch` 仅用于显式记录的交集配对。"
    )
    if len(meta["baseline_runs"]) > 1 and len(meta["agent_runs"]) > 1:
        lines.append(
            "- 多个 baseline run × 多个 agent run：run pair 为交叉配对；主结论应 pin 单一 baseline run"
            "（交叉配对用于基线稳定性检查）。"
        )
    lines.append(
        "- 选点纪律（v7 预注册 §4）：测试集（eval500）每个最终 run 只评估一次；本报告只做评估，不参与选点。"
    )
    lines.append(
        "- 分层 n < 最小判定数只报不判（与 eval_runner `per_category_n_min` 同口径）；单 run 仅方向性，"
        "主判据以多 run 汇总 95% CI 下界 > 0 为准。"
    )
    lines.append("")
    return "\n".join(lines)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(key): _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(value) for value in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _label_for(path: Path) -> str:
    return path.parent.name if path.parent.name else path.stem


def _load_run(path_text: str, label: Optional[str]) -> Dict[str, Any]:
    path = Path(path_text).resolve()
    episodes = load_episodes_csv(path)
    n = len(episodes)
    summary = {
        "n": n,
        "n_error": sum(1 for ep in episodes.values() if ep.error),
        "success_rate": sum(1 for ep in episodes.values() if ep.success) / n if n else None,
    }
    return {
        "label": str(label) if label else _label_for(path),
        "path": str(path),
        "sha256": _sha256_file(path),
        "episodes": episodes,
        "summary": summary,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/paired_eval.py",
        description="v7 配对评测：同场景配对 McNemar + bootstrap CI + 多 run 汇总",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        metavar="EPISODES.csv",
        help="基线 run 的 episodes.csv（可多个：基线多 evaluation seed）",
    )
    parser.add_argument(
        "--agent",
        nargs="+",
        required=True,
        metavar="EPISODES.csv",
        help="agent run 的 episodes.csv（多 seed：≥3 个独立 run）",
    )
    parser.add_argument("--baseline-labels", nargs="+", default=None, help="与 --baseline 一一对应的标签")
    parser.add_argument("--agent-labels", nargs="+", default=None, help="与 --agent 一一对应的标签")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help="bootstrap 次数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap 固定 seed")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="显著性水平（默认 0.05）")
    parser.add_argument(
        "--min-stratum",
        type=int,
        default=DEFAULT_MIN_STRATUM,
        help="分层最小判定 n（n < 只报不判；默认与 eval_runner 同口径 30）",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="键集合不一致时取交集（默认 fail-closed 报错；缺失数记入报告）",
    )
    parser.add_argument("--out-dir", default=None, help="写 paired_eval.md / paired_eval.json 的目录")
    parser.add_argument("--quiet", action="store_true", help="不向 stdout 打印 markdown")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.bootstrap <= 0:
        raise SystemExit("--bootstrap 必须 > 0")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("--alpha 必须在 (0, 1) 内")
    if args.min_stratum < 0:
        raise SystemExit("--min-stratum 必须 ≥ 0")
    for flag, labels, paths in (
        ("--baseline-labels", args.baseline_labels, args.baseline),
        ("--agent-labels", args.agent_labels, args.agent),
    ):
        if labels is not None and len(labels) != len(paths):
            raise SystemExit(f"{flag} 数量（{len(labels)}）须与对应输入文件数（{len(paths)}）一致")

    baseline_runs = [
        _load_run(path, label)
        for path, label in zip(args.baseline, args.baseline_labels or [None] * len(args.baseline))
    ]
    agent_runs = [
        _load_run(path, label)
        for path, label in zip(args.agent, args.agent_labels or [None] * len(args.agent))
    ]
    report = analyze(
        baseline_runs,
        agent_runs,
        n_boot=int(args.bootstrap),
        seed=int(args.seed),
        alpha=float(args.alpha),
        min_stratum=int(args.min_stratum),
        allow_mismatch=bool(args.allow_mismatch),
    )
    markdown = render_markdown(report)
    if not args.quiet:
        print(markdown)
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "paired_eval.md").write_text(markdown, encoding="utf-8")
        (out_dir / "paired_eval.json").write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"[paired_eval] 报告 → {out_dir / 'paired_eval.md'} / {out_dir / 'paired_eval.json'}")
    verdict = report["aggregate"]["verdict"]
    print(
        f"[paired_eval] runs={verdict['n_runs']} meanΔ={_fmt(verdict['mean_delta_pp'], 2, sign=True)}pt "
        f"CI95={_fmt_ci(verdict['ci95_pp'])} 下界>0={verdict['primary_ci_lower_gt_zero']} "
        f"正向 run={verdict['positive_runs']}/{verdict['n_runs']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
