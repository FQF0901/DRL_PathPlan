#!/usr/bin/env python3
"""Recovery 采集器（P1-1 预备；两遍确定性重放 + 专家接管，干跑默认）。

机制（详见 §7.10-A5 与 docs/p1_probe_spec.md）
-----------------------------------------------
forensics 为"单控制器整集"架构、无接管/快照 → 本工具用**两遍确定性重放**：

- **Pass 1（锚点）**：学生（ckpt + LqrTracker + plan 参考）跑完整 episode，逐 env-step /
  逐决策步记录（复用 ``tools/diagnostics/forensics_closed_loop.py`` 的控制器/记录/后处理，
  只读 import，不改该文件）。候选窗口起点沿用锚点触发源：``first_low_margin``
  （首个 ``|lane_lat| > --low-lat`` 的决策步；**代理**：forensics 的 ``min_signed_margin``
  恒 None——引擎只给布尔 on-lane 查询，不伪造 signed distance）、``T_plan−context`` /
  ``T_plan``、``T_cross−recovery`` / ``T_cross−late_margin`` / ``T_cross``（T_cross 可回退
  env out_of_road/crash 首步，见 field_gaps）。
  **多窗口**：每个失败 episode 至多 3 个窗口（每档 ≤1），**按接管时刻实况**分类：
  ``late`` = 已越界/接近压线（exec footprint 不可行 / T_cross 邻域 / |lane_lat| ≥ --late-lat）；
  ``correction`` = 计划已不可行或 margin 退化（plan footprint 不可行 / ≥T_plan / |lane_lat| ≥ --low-lat）；
  ``prevention`` = 计划可行且 margin 健康。窗口选取：prevention 取最晚、correction 取最早、
  late 取最晚；≤20 行/集（3×6）。
- **配额**：``--targets prevention=4000,correction=2500,late=1000``（默认即此）；按池遍历，
  某档有效行数达标后该档停止加采；全档达标提前停止（不复制样本，自然过量即停）。
- **Pass 2（接管）**：同 spec 重建 env，学生重放到窗口 ``takeover_step``（对齐 0.5s 决策
  网格），随后切换为 **d2 同款 expert 接管**（``D2RepeatExpertController`` 语义：当前状态
  专家空问动作 repeat 6 步 + LQR；直接调用其 ``action()``，不改代码）。接管后继续
  ``--expert-steps``（默认 6 = 3s）个决策步；为给每行 6 点未来轨迹，Pass 2 实际多跑 30 步。
- **一致性核验（逐窗口）**：Pass 2 接管步状态 vs Pass 1（x/y/θ，含 wrap）；容差先
  ``--determinism-tol 1e-6``；不通过 → 重跑 Pass 2 一次 → 仍 >1e-6 但
  ≤ ``--determinism-tol-relaxed 1e-4`` → 接受并标注 relaxed；仍不一致 → **跳过该窗口并
  计数**（不整 lane 硬停）。d2 评测栈存在 run-to-run 非确定性（已知证据：spec 299 12 次
  3 种结局），故必须容错。

样本行（schema 契约 = ``tools/controls/recovery_quality_gates.py`` docstring）
--------------------------------------------------------------------------
每行 = Pass 2 **实际接管状态**（③：样本一律与 expert 接管同源）：
``episode_id/step/pose/dt``；student obs+history、student plan、student actual trajectory
（Pass 1 同 step 对齐，k≥1 行属"同时间索引反事实对照"）；expert current-state action
（raw + (ds,dθ)）、expert recovery 6 步 plan、expert recovery 轨迹（积分）、expert actual
recovery 轨迹（Pass 2 实际未来 6 决策步，自车系）；T_plan/T_cross/first_low_margin；
geometry labels（surface=plan footprint inside_ratio/valid；legal-line=当前车道
left/right 线型 id ∈ 实线集合（``env/obs/ld.py::LINE_TYPE_IDS`` 只读派生）；route=rc）；
expert_recovered / recovery_success_horizon / min_margin_after_takeover /
route_progress_after_takeover（专家接管 3s 窗口内实测）；mode_label/mode_confidence
（恢复轨迹粗标签，仅分析用）；validity masks。
质量门：逐行调用 ``recovery_quality_gates.evaluate_row``；不合格行仍落盘但 ``usable=False``
（"可用集" = usable 行）。

产物::

    <out>/                     # 试点 runs/p1_0/recovery_pilot/；正式 datasets/recovery_v1/
      manifest.json         provenance + 逐 episode 窗口/一致性/质量 + 配额 + 池规划
      samples.npz           样本行（数组；含 pool/class/window_index）
      schema.json           字段/形状/语义 + 质量门映射
      quality_report.json   质量门聚合 + 每档配额 + 每池产出 + 路型覆盖
      split_manifest.json   episode 级 train/val（'pool:id' 键，禁泄漏）

正式采集（另行授权）：``--out datasets/recovery_v1`` 并附
``docs/recovery_v1_dataset_card.md`` 模板（本工具自动写，若不存在）。

用法（100 集试点，多池按几何分层）::

    tools/venv-python tools/controls/collect_recovery.py \\
        --ckpt runs/BTC20261009-0824_sw_pri512/stage_b/ckpt_epoch020.pt \\
        --config runs/p1_0/configs/p1a_pri512.yaml \\
        --spec env/specs/scenarios_eval500.json \\
        --spec docs/v7_reports/specs/specs_val_only500.json \\
        --spec env/specs/scenarios_train_dagger_r1.json \\
        --stratify-per-geometry 10 --limit 100 --workers 1 --device cpu \\
        --out runs/p1_0/recovery_pilot

正式采集（分片 + 合并；6 分片示例）::

    for i in 0 1 2 3 4 5; do
      tools/venv-python tools/controls/collect_recovery.py \\
        --spec env/specs/scenarios_eval500.json \\
        --spec docs/v7_reports/specs/specs_val_only500.json \\
        --spec env/specs/scenarios_train_dagger_r1.json \\
        --spec env/specs/scenarios_train_dagger_r2.json \\
        --spec env/specs/scenarios_train_dagger_r3.json \\
        --stratify-per-geometry 30 --limit 1500 --no-stop-at-target \\
        --shard $i/6 --out runs/p1_0/recovery_formal_shard$i --workers 1 --device cpu &
    done; wait
    tools/venv-python tools/controls/merge_recovery_shards.py \\
        --shards runs/p1_0/recovery_formal_shard{0,1,2,3,4,5} --out datasets/recovery_v1

``--shard i/n``：在池选择/分层完成后对稳定排序的 ``(池, id)`` 列表做 round-robin
划分（完备互斥、确定性）。``--no-stop-at-target``：targets 仅报告、按全池自然分布采集
（避免"三档和 7.5k < 10k 总行门"的冲突；默认仍为达标停采）。

退出码：0 = 完成（含跳过计数）；2 = 致命错误（参数/环境/ckpt 等）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.obs.ld import LINE_TYPE_IDS, line_type_id  # noqa: E402  # 只读引用线型 id 表
from env.tracking import interpolate  # noqa: E402  # §8.5 弧线积分单一真源
from tools.controls import recovery_quality_gates as gates  # noqa: E402

POLICY_DT = 0.5
STEPS_PER_POLICY = 5
HORIZON = 6
DEFAULT_CKPT = "runs/BTC20261009-0824_sw_pri512/stage_b/ckpt_epoch020.pt"
DEFAULT_CONFIG = "runs/p1_0/configs/p1a_pri512.yaml"
DEFAULT_SPEC = "env/specs/scenarios_eval500.json"
DEFAULT_OUT = "runs/p1_0/recovery_dryrun"

SOLID_LINE_IDS: tuple[int, ...] = tuple(
    sorted(value for name, value in LINE_TYPE_IDS.items() if "SOLID" in name)
)  # = (2, 3, 6, 7)

CLASS_IDS = {"prevention": 0, "correction": 1, "late": 2}
WINDOW_CLASSES = ("prevention", "correction", "late")
HARD_ROAD_TYPES = ("tollgate", "merge", "roundabout")
HARD_TYPE_MIN_ROWS = 300
HARD_TYPE_MIN_EPISODES = 30
MODE_IDS = {
    "keep_lane": 0,
    "left_shift": 1,
    "right_shift": 2,
    "left_curve": 3,
    "right_curve": 4,
    "merge": 5,
    "turnaround_or_complex": 6,
}
CLASS_TARGET_RATIO = {"prevention": 0.50, "correction": 0.35, "late": 0.15}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- #
# 纯函数（离线单测覆盖）
# --------------------------------------------------------------------------- #
def _ts_step(timestamps: dict | None, key: str) -> int | None:
    entry = (timestamps or {}).get(key)
    if isinstance(entry, dict) and entry.get("step") is not None:
        return int(entry["step"])
    return None


def first_low_margin_step(
    decision_lane_lat: dict[int, float | None], low_lat: float
) -> int | None:
    """首个 ``|lane_lat| > low_lat`` 的决策步（margin 代理，见 field_gaps）。"""
    for step in sorted(decision_lane_lat):
        lat = decision_lane_lat[step]
        if lat is not None and math.isfinite(float(lat)) and abs(float(lat)) > float(low_lat):
            return int(step)
    return None


def generate_window_candidates(
    *,
    t_plan: int | None,
    t_cross: int | None,
    first_low_margin: int | None,
    context_steps: int = 20,
    recovery_steps: int = 30,
    late_margin_steps: int = 5,
    max_decision_step: int | None = None,
) -> list[int]:
    """候选接管点（env steps，对齐 0.5s 决策网格、升序去重）。

    触发源沿用锚点体系：``first_low_margin`` / ``T_plan−context`` / ``T_plan`` /
    ``T_cross−recovery`` / ``T_cross−late_margin`` / ``T_cross``（T_cross 可来自 env 事件回退）。
    """
    raw: list[int] = []
    if first_low_margin is not None:
        raw.append(int(first_low_margin))
    if t_plan is not None:
        raw.append(max(0, int(t_plan) - int(context_steps)))
        raw.append(int(t_plan))
    if t_cross is not None:
        raw.append(max(0, int(t_cross) - int(recovery_steps)))
        raw.append(max(0, int(t_cross) - int(late_margin_steps)))
        raw.append(int(t_cross))
    steps: set[int] = set()
    for value in raw:
        step = STEPS_PER_POLICY * (max(0, int(value)) // STEPS_PER_POLICY)
        if max_decision_step is not None:
            step = min(step, int(max_decision_step))
        steps.add(step)
    return sorted(steps)


def classify_window(
    takeover_step: int,
    *,
    t_plan: int | None,
    t_cross: int | None,
    lane_lat: float | None,
    plan_footprint_valid: Any,
    exec_footprint_valid: Any,
    low_lat: float = 1.0,
    late_lat: float = 1.5,
    late_margin_steps: int = 5,
) -> str:
    """按接管时刻实况（margin 严重度）三档分类。

    - ``late``：已越界/接近压线 —— exec footprint 不可行、T_cross 邻域
      （``takeover ≥ T_cross−late_margin``）或 ``|lane_lat| ≥ late_lat``；
    - ``correction``：计划已不可行或 margin 已退化 —— plan footprint 不可行、
      ``takeover ≥ T_plan`` 或 ``|lane_lat| ≥ low_lat``；
    - ``prevention``：其余（计划可行且 margin 健康）。
    """
    if t_cross is not None and int(takeover_step) >= int(t_cross) - int(late_margin_steps):
        return "late"
    if exec_footprint_valid is False:
        return "late"
    if lane_lat is not None and abs(float(lane_lat)) >= float(late_lat):
        return "late"
    if t_plan is not None and int(takeover_step) >= int(t_plan):
        return "correction"
    if plan_footprint_valid is False:
        return "correction"
    if lane_lat is not None and abs(float(lane_lat)) >= float(low_lat):
        return "correction"
    return "prevention"


def select_windows(
    candidates: list[int],
    state_at: Any,
    *,
    t_plan: int | None = None,
    t_cross: int | None = None,
    low_lat: float = 1.0,
    late_lat: float = 1.5,
    late_margin_steps: int = 5,
    max_windows: int = 3,
) -> list[tuple[str, int]]:
    """候选点按接管时刻实况分类，每档至多 1 个窗口。

    ``state_at(step) -> {"lane_lat", "plan_footprint_valid", "exec_footprint_valid"}``。
    选取：prevention 取最晚（贴近退化起点）、correction 取最早（贴近计划失效起点）、
    late 取最晚（贴近/越过压线点）。
    """
    by_class: dict[str, list[int]] = {}
    for step in candidates:
        state = state_at(int(step)) or {}
        window_class = classify_window(
            int(step),
            t_plan=t_plan,
            t_cross=t_cross,
            lane_lat=state.get("lane_lat"),
            plan_footprint_valid=state.get("plan_footprint_valid"),
            exec_footprint_valid=state.get("exec_footprint_valid"),
            low_lat=low_lat,
            late_lat=late_lat,
            late_margin_steps=late_margin_steps,
        )
        by_class.setdefault(window_class, []).append(int(step))
    chosen: list[tuple[str, int]] = []
    if by_class.get("prevention"):
        chosen.append(("prevention", max(by_class["prevention"])))
    if by_class.get("correction"):
        chosen.append(("correction", min(by_class["correction"])))
    if by_class.get("late"):
        chosen.append(("late", max(by_class["late"])))
    return chosen[: max(1, int(max_windows))]


def filter_windows_by_quota(
    windows: list[tuple[str, int]], usable_counts: dict[str, int], targets: dict[str, int]
) -> list[tuple[str, int]]:
    """达到目标（有效行数）的档位不再加采（本集该档窗口移除）。"""
    return [
        (window_class, step)
        for window_class, step in windows
        if int(usable_counts.get(window_class, 0)) < int(targets.get(window_class, 0))
    ]


def parse_targets(text: str) -> dict[str, int]:
    """``prevention=4000,correction=2500,late=1000`` → dict（未列出的档位目标 0）。"""
    targets = {window_class: 0 for window_class in WINDOW_CLASSES}
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        key = key.strip()
        if key not in WINDOW_CLASSES:
            raise ValueError(f"未知档位 {key!r}（可选 {WINDOW_CLASSES}）")
        targets[key] = int(value)
    return targets


def apply_quota_policy(
    windows: list[tuple[str, int]],
    usable_counts: dict[str, int],
    targets: dict[str, int],
    *,
    stop_at_target: bool = True,
) -> list[tuple[str, int]]:
    """配额策略：``stop_at_target=True`` 按档位过滤（默认）；False 全采（targets 仅报告）。"""
    if not stop_at_target:
        return list(windows)
    return filter_windows_by_quota(windows, usable_counts, targets)


def parse_shard(text: str) -> tuple[int, int]:
    """``i/n``（0-based）→ (index, count)；越界/格式错误抛 ValueError。"""
    head, sep, tail = str(text).partition("/")
    if not sep:
        raise ValueError(f"--shard 需为 i/n 格式，收到 {text!r}")
    try:
        index, count = int(head), int(tail)
    except ValueError as exc:
        raise ValueError(f"--shard 需为整数 i/n，收到 {text!r}") from exc
    if count < 1 or index < 0 or index >= count:
        raise ValueError(f"--shard 越界: {text!r}（要求 0 ≤ i < n）")
    return index, count


def shard_episodes(
    selected: list[tuple[str, Any]], *, index: int, count: int
) -> list[tuple[str, Any]]:
    """在池选择/分层之后按稳定排序 ``(池, id)`` 做 i/n 划分（round-robin stride）。

    完备且互斥、确定性（同参数两次划分相同）；``count > len`` 时部分分片为空。
    """
    index, count = int(index), int(count)
    if count < 1 or index < 0 or index >= count:
        raise ValueError(f"分片参数越界: {index}/{count}（要求 0 ≤ i < n）")
    ordered = sorted(
        selected, key=lambda item: (str(item[0]), int(getattr(item[1], "id", -1)))
    )
    return ordered[index::count]


def _pool_name(path: str | Path) -> str:
    stem = Path(path).stem
    for prefix in ("scenarios_", "specs_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return stem


def select_episodes(
    pool_specs: list[dict],
    *,
    ids: str = "",
    per_geometry: int = 0,
    limit: int = 0,
) -> list[tuple[str, Any]]:
    """从多池选择 episode（确定性）：ids > 按几何分层 round-robin > 池内前 N。

    ``pool_specs`` = ``[{"name", "specs", "limit"}]``（spec 对象需有 ``id``/``labels``）。
    返回 ``[(pool_name, spec), ...]``。
    """
    selected: list[tuple[str, Any]] = []
    wanted_ids = [int(token) for token in str(ids).split(",") if token.strip()] if str(ids).strip() else []
    if wanted_ids:
        want = set(wanted_ids)
        for pool in pool_specs:
            for spec in pool["specs"]:
                if int(getattr(spec, "id", -1)) in want:
                    selected.append((pool["name"], spec))
    elif int(per_geometry) > 0:
        groups: list[tuple[str, str, list]] = []
        for pool in pool_specs:
            by_geometry: dict[str, list] = {}
            for spec in pool["specs"]:
                geometry = str(getattr(spec, "labels", {}).get("geometry", "?"))
                by_geometry.setdefault(geometry, []).append(spec)
            for geometry in sorted(by_geometry):
                groups.append((pool["name"], geometry, by_geometry[geometry]))
        for round_index in range(int(per_geometry)):
            for pool_name, _geometry, specs in groups:
                if round_index < len(specs):
                    selected.append((pool_name, specs[round_index]))
                    if limit and len(selected) >= int(limit):
                        return selected[: int(limit)]
    else:
        for pool in pool_specs:
            cap = pool.get("limit")
            specs = pool["specs"][: int(cap)] if cap else list(pool["specs"])
            for spec in specs:
                selected.append((pool["name"], spec))
                if limit and len(selected) >= int(limit):
                    return selected[: int(limit)]
    if limit and len(selected) > int(limit):
        selected = selected[: int(limit)]
    return selected


def split_episodes(episode_keys: list, *, val_every: int = 5) -> tuple[list, list]:
    """episode 级 train/val 划分（确定性、无泄漏）：排序后每 val_every 个末位进 val。

    ``episode_keys`` 可为 int 或 ``"pool:id"`` 字符串等任意可排序键。
    """
    ordered = sorted({str(key) for key in episode_keys})
    val: list[str] = []
    if val_every and val_every > 0:
        val = [key for pos, key in enumerate(ordered) if pos % val_every == val_every - 1]
    val_set = set(val)
    train = [key for key in ordered if key not in val_set]
    return train, val


def select_rows(
    takeover_step: int, episode_len_steps: int, *, expert_steps: int = 6, max_rows: int = 20
) -> list[int]:
    """样本行锚点（决策步）：接管步 + 专家阶段每个决策步；≤ max_rows 且不超 episode 长度。"""
    rows: list[int] = []
    limit = min(int(expert_steps), int(max_rows))
    for k in range(limit):
        step = int(takeover_step) + STEPS_PER_POLICY * k
        if step >= int(episode_len_steps):
            break
        rows.append(step)
    return rows


def consistency_verdict(max_diff: float, tol: float, relaxed: float) -> str:
    """strict（≤tol）/ relaxed（≤relaxed）/ skip。"""
    value = float(max_diff)
    if not math.isfinite(value) or value > float(relaxed):
        return "skip"
    return "strict" if value <= float(tol) else "relaxed"


def ego_relative_traj(poses: np.ndarray, base_pose: np.ndarray | list[float]) -> np.ndarray:
    """世界系位姿 ``(K,3|2)`` → base_pose=(x,y,θ) 自车系相对位置 ``(K,2)``。"""
    arr = np.asarray(poses, dtype=np.float64).reshape(-1, 3) if np.asarray(poses).ndim > 1 and np.asarray(poses).shape[-1] == 3 else None
    if arr is None:
        pts = np.asarray(poses, dtype=np.float64).reshape(-1, 2)
        arr = np.concatenate([pts, np.zeros((len(pts), 1))], axis=1)
    base = np.asarray(base_pose, dtype=np.float64).reshape(-1)
    theta = float(base[2])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx = arr[:, 0] - float(base[0])
    dy = arr[:, 1] - float(base[1])
    return np.stack([dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t], axis=1)


def mode_label(
    traj: np.ndarray, *, nav_cps: Any = None, pose: Any = None
) -> tuple[str, float]:
    """恢复轨迹粗标签（分析用）：keep_lane/left_shift/right_shift/left_curve/right_curve/merge/turnaround_or_complex。"""
    pts = np.asarray(traj, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 2 or not np.isfinite(pts).all():
        return "turnaround_or_complex", 0.0
    segments = np.diff(np.vstack([[0.0, 0.0], pts]), axis=0)
    angles = np.arctan2(segments[:, 1], segments[:, 0])
    turn = float(sum(_wrap_pi(a - b) for a, b in zip(angles[1:], angles[:-1])))
    lateral = float(pts[-1, 1])

    merge = False
    if nav_cps is not None and pose is not None and abs(lateral) >= 1.0:
        try:
            cps = np.asarray(nav_cps, dtype=np.float64).reshape(-1, 2)
            theta = float(np.asarray(pose, dtype=np.float64).reshape(-1)[2])
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            dx = cps[:, 0] - float(np.asarray(pose).reshape(-1)[0])
            dy = cps[:, 1] - float(np.asarray(pose).reshape(-1)[1])
            local_x = dx * cos_t + dy * sin_t
            local_y = -dx * sin_t + dy * cos_t
            near = (local_x > 0.0) & (local_x <= 25.0)
            same_side = np.sign(local_y) == math.copysign(1.0, lateral)
            if np.any(near & (np.abs(local_y) >= 1.0) & same_side):
                merge = True
        except Exception:  # noqa: BLE001 - 标签尽力而为
            merge = False

    if abs(turn) >= 1.2 or abs(lateral) >= 4.0:
        label = "turnaround_or_complex"
        confidence = min(1.0, max(abs(turn) / 1.2, abs(lateral) / 4.0))
    elif abs(lateral) >= 1.0 and abs(turn) < 0.35:
        if merge:
            label = "merge"
            confidence = min(1.0, abs(lateral) / 1.0)
        else:
            label = "left_shift" if lateral > 0 else "right_shift"
            confidence = min(1.0, abs(lateral) / 1.0) * (1.0 - min(1.0, abs(turn) / 0.35))
    elif abs(turn) >= 0.35:
        label = "left_curve" if turn > 0 else "right_curve"
        confidence = min(1.0, abs(turn) / 0.35)
    else:
        label = "keep_lane"
        confidence = 1.0 - min(1.0, abs(lateral) / 1.0 + abs(turn) / 0.35)
    return label, float(max(0.0, min(1.0, confidence)))


def build_gate_row(row: dict) -> dict:
    """样本行 → recovery_quality_gates.evaluate_row 输入（质量门接线）。"""
    return {
        "episode_id": int(row["episode_id"]),
        "step": int(row["step"]),
        "dt": POLICY_DT,
        "pose": [float(v) for v in row["pose"]],
        "recovery_traj": np.asarray(row["expert_recovery_traj"], dtype=np.float64).tolist(),
        "action_chain": np.asarray(row["expert_action_chain"], dtype=np.float64).tolist(),
        "margin_before": row["margin_before"],
        "margin_after": row["margin_after"],
        "rc_before": row["rc_before"],
        "rc_after": row["rc_after"],
        "takeover_off_road": bool(row["takeover_off_road"]),
        "takeover_collision": bool(row["takeover_collision"]),
        "geometry_labels": row["geometry_labels"],
    }


def _lane_line_types(env: Any) -> dict[str, float]:
    """当前车道 left/right 线型 id（与 env/obs/ld.py 同源读取；失败 → 0）。"""
    try:
        lane = getattr(env.agent, "lane", None)
        types = list(getattr(lane, "line_types", None) or [])
        return {
            "left_id": float(line_type_id(types[0] if len(types) > 0 else None)),
            "right_id": float(line_type_id(types[1] if len(types) > 1 else None)),
        }
    except Exception:  # noqa: BLE001
        return {"left_id": 0.0, "right_id": 0.0}


# --------------------------------------------------------------------------- #
# 控制器（只读复用 forensics；不改其代码）
# --------------------------------------------------------------------------- #
def _make_controllers():
    """延迟 import forensics（避免 import 期副作用）并返回 (StudentRecorder, TakeoverController)。"""
    from tools.diagnostics import forensics_closed_loop as fcl
    from tools.dagger_collect import _make_labeler

    class StudentRecorder(fcl.InstrumentedCkpt):
        """Pass 1：学生控制器 + 逐决策步 obs / 车道线型记录。"""

        def __init__(self, env, spec, task, *, tracker_json=None):
            super().__init__(env, spec, task, tracker_json=tracker_json, eval_reference="plan")
            self.obs_by_step: dict[int, dict[str, np.ndarray]] = {}
            self.lane_lines_by_step: dict[int, dict[str, float]] = {}

        def action(self, env):
            result = super().action(env)
            rec = self.records[-1] if self.records else None
            if rec is not None and rec.get("decision"):
                step = int(rec["step"])
                obs = self.builder.build(env, self.spec)  # 同 step 二次 build（幂等）
                self.obs_by_step[step] = {
                    key: np.array(value, copy=True) for key, value in obs.items()
                }
                self.lane_lines_by_step[step] = _lane_line_types(env)
            return result

    class TakeoverController(fcl.D2RepeatExpertController):
        """Pass 2：前 takeover_step 步 = 学生（InstrumentedCkpt 语义），之后 = D2 expert repeat。"""

        def __init__(self, env, spec, task, *, takeover_step: int, tracker_json=None):
            # 跳过 D2.__init__ 的 load_model=False：学生阶段需要模型
            fcl.InstrumentedCkpt.__init__(
                self, env, spec, task, tracker_json=tracker_json, eval_reference="plan"
            )
            self.takeover_step = int(takeover_step)
            self.expert_kind = "pure_pursuit"
            self.ds_scale = 1.0
            self.ds_dth_scale = 1.0
            self.labeler = None
            self.records_control_saturation = True
            self.lane_lines_by_step: dict[int, dict[str, float]] = {}

        def bind(self, env) -> None:
            fcl.InstrumentedCkpt.bind(self, env)
            self.labeler = _make_labeler(env, self.spec, self.expert_kind)

        def action(self, env):
            if self._steps < self.takeover_step:
                result = fcl.InstrumentedCkpt.action(self, env)
            else:
                result = fcl.D2RepeatExpertController.action(self, env)
            rec = self.records[-1] if self.records else None
            if rec is not None and rec.get("decision"):
                self.lane_lines_by_step[int(rec["step"])] = _lane_line_types(env)
            return result

    return StudentRecorder, TakeoverController


# --------------------------------------------------------------------------- #
# Pass 运行
# --------------------------------------------------------------------------- #
def _run_pass1(spec, config, task, *, max_steps: int):
    from tools.diagnostics import forensics_closed_loop as fcl

    StudentRecorder, _ = _make_controllers()
    env = fcl._build_env(spec, config)
    try:
        controller = StudentRecorder(env, spec, task)
        outcome = fcl.run_episode(env, spec, controller, max_steps=max_steps)
        analysis = fcl._analyze_episode(controller, outcome)
    finally:
        try:
            env.close()
        except BaseException:
            pass
    return controller, outcome, analysis


def _run_pass2(spec, config, task, *, takeover_step: int, expert_steps: int, horizon_steps: int):
    from tools.diagnostics import forensics_closed_loop as fcl

    _, TakeoverController = _make_controllers()
    env = fcl._build_env(spec, config)
    try:
        controller = TakeoverController(env, spec, task, takeover_step=takeover_step)
        max_steps = takeover_step + STEPS_PER_POLICY * (expert_steps - 1) + horizon_steps + 1
        outcome = fcl.run_episode(env, spec, controller, max_steps=max_steps)
        analysis = fcl._analyze_episode(controller, outcome)
    finally:
        try:
            env.close()
        except BaseException:
            pass
    return controller, outcome, analysis


def compare_prefix(pass1, pass2, takeover_step: int) -> dict:
    """Pass2 前段（< takeover_step）与 Pass1 的逐位比较。

    返回 ``takeover_state_diff``（接管步状态 x/y/θ 最大偏差；takeover_step=0 → 0）、
    ``prefix_max_diff``（全部前段 env-step 位置/航向最大偏差）、``plan_max_diff``（决策
    plan/mu 最大偏差）、``n_compared``。
    """
    n = min(int(takeover_step), len(pass1.steps), len(pass2.steps))
    prefix_max = 0.0
    for index in range(n):
        a, b = pass1.steps[index], pass2.steps[index]
        prefix_max = max(
            prefix_max,
            abs(float(a["x"]) - float(b["x"])),
            abs(float(a["y"]) - float(b["y"])),
            abs(_wrap_pi(float(a["theta"]) - float(b["theta"]))),
        )
    if n > 0:
        a, b = pass1.steps[n - 1], pass2.steps[n - 1]
        takeover_state_diff = max(
            abs(float(a["x"]) - float(b["x"])),
            abs(float(a["y"]) - float(b["y"])),
            abs(_wrap_pi(float(a["theta"]) - float(b["theta"]))),
        )
    else:
        takeover_state_diff = 0.0
    plan_max = 0.0
    for index in range(min(int(takeover_step), len(pass1.records), len(pass2.records))):
        rec1, rec2 = pass1.records[index], pass2.records[index]
        if not rec1.get("decision"):
            continue
        for key in ("plan", "mu"):
            arr1 = np.asarray(rec1.get(key), dtype=np.float64)
            arr2 = np.asarray(rec2.get(key), dtype=np.float64)
            if arr1.shape == arr2.shape and arr1.size:
                plan_max = max(plan_max, float(np.max(np.abs(arr1 - arr2))))
    return {
        "takeover_state_diff": float(takeover_state_diff),
        "prefix_max_diff": float(prefix_max),
        "plan_max_diff": float(plan_max),
        "n_compared_steps": int(n),
    }


# --------------------------------------------------------------------------- #
# episode 采集
# --------------------------------------------------------------------------- #
def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _record_pose(rec: dict) -> list[float]:
    return [float(rec["x"]), float(rec["y"]), float(rec["theta"])]


def _window_quality(controller2, outcome2, takeover_step: int, args) -> dict:
    """专家接管阶段质量（3s 相位 + Pass2 观测 horizon 失败判定）。"""
    phase_len = STEPS_PER_POLICY * int(args.expert_steps)
    phase_steps = controller2.steps[takeover_step: takeover_step + phase_len]
    outcome2_term = str(outcome2.get("termination"))
    # 失败判定 = 相位内 off/crash 或 Pass 2 观测 horizon 内 env 以 off/collision 终止
    # （只看 3s 窗会漏掉窗后立即失败，如 +3.2s 越界 → 假 recovered）
    takeover_off_road = (
        any(bool(row.get("out_of_road")) for row in phase_steps) or outcome2_term == "out_of_road"
    )
    takeover_collision = (
        any(bool(row.get("crash")) for row in phase_steps) or outcome2_term == "collision"
    )
    expert_recovered = not (takeover_off_road or takeover_collision)
    lane_lats = [_safe_float(row.get("lane_lat")) for row in phase_steps]
    lane_lats = [value for value in lane_lats if value is not None]
    min_margin_after = -max(abs(value) for value in lane_lats) if lane_lats else None
    rc_takeover = (
        _safe_float(controller2.steps[takeover_step - 1].get("rc")) if takeover_step > 0 else 0.0
    )
    rc_end = _safe_float(phase_steps[-1].get("rc")) if phase_steps else rc_takeover
    route_gain = (
        float(rc_end - rc_takeover)
        if rc_end is not None and rc_takeover is not None
        else None
    )
    # 恢复窗 margin/route 改善 = episode 级端点对比（gate 用）：
    # before = 接管前 context 窗内最差 -|lane_lat|；after = 相位末 5 步 -|lane_lat| 均值。
    pre_window = controller2.steps[max(0, takeover_step - int(args.context_steps)): takeover_step]
    pre_lats = [_safe_float(row.get("lane_lat")) for row in pre_window]
    pre_lats = [value for value in pre_lats if value is not None]
    margin_before_ep = -max(abs(value) for value in pre_lats) if pre_lats else None
    tail = lane_lats[-min(5, len(lane_lats)):] if lane_lats else []
    margin_after_ep = -sum(abs(value) for value in tail) / len(tail) if tail else None
    return {
        "recovered": bool(expert_recovered),
        "success_horizon_steps": int(len(phase_steps)),
        "phase_steps": int(phase_len),
        "off_road": bool(takeover_off_road),
        "collision": bool(takeover_collision),
        "min_margin_after_takeover": min_margin_after,
        "margin_before_window": margin_before_ep,
        "margin_after_tail": margin_after_ep,
        "route_progress_after_takeover": route_gain,
        "rc_before": rc_takeover,
        "rc_after": rc_end,
        "termination": outcome2_term,
    }


def _collect_one_window(spec, config, task, controller1, *, takeover_step: int, args) -> dict:
    """单窗口：Pass 2（含一次重跑）+ 接管步状态一致性核验 + 质量字段。"""
    attempts: list[dict] = []
    accepted = None
    accepted_verdict = None
    for attempt in (1, 2):
        controller2, outcome2, _analysis2 = _run_pass2(
            spec,
            config,
            task,
            takeover_step=takeover_step,
            expert_steps=args.expert_steps,
            horizon_steps=STEPS_PER_POLICY * HORIZON,
        )
        if len(controller2.steps) <= takeover_step or len(controller2.records) <= takeover_step:
            return {"status": "skipped", "reason": "takeover_beyond_episode", "attempts": attempts}
        diff = compare_prefix(controller1, controller2, takeover_step)
        verdict = consistency_verdict(
            diff["takeover_state_diff"], args.determinism_tol, args.determinism_tol_relaxed
        )
        attempts.append({"attempt": attempt, "verdict": verdict, **diff})
        if verdict == "strict":
            accepted = (controller2, outcome2)
            accepted_verdict = "strict"
            break
        if verdict == "relaxed" and attempt == 2:
            accepted = (controller2, outcome2)
            accepted_verdict = "relaxed"
            break
    if accepted is None:
        return {"status": "skipped", "reason": "pass2_state_mismatch", "attempts": attempts}
    controller2, outcome2 = accepted
    return {
        "status": "collected",
        "determinism": {"accepted": accepted_verdict, "attempts": attempts},
        "expert": _window_quality(controller2, outcome2, takeover_step, args),
        "controller2": controller2,
    }


def _build_window_rows(
    controller1,
    controller2,
    *,
    window_class: str,
    window_index: int,
    takeover_step: int,
    pool: str,
    info: dict,
    quality: dict,
    args,
) -> list[dict]:
    """窗口样本行（Pass 2 实际状态；每行 6 点未来）。"""
    anchors = info["anchors"]
    row_steps = select_rows(
        takeover_step,
        len(controller2.records),
        expert_steps=args.expert_steps,
        max_rows=args.max_rows_per_episode,
    )
    rows: list[dict] = []
    for step in row_steps:
        rec2 = controller2.records[step]
        pose = _record_pose(rec2)
        plan2 = np.asarray(rec2.get("plan"), dtype=np.float64).reshape(-1, 2)
        recovery_traj = interpolate(plan2, dt=POLICY_DT, hz=10)[
            [i * STEPS_PER_POLICY - 1 for i in range(1, HORIZON + 1)], :2
        ]
        raw_action = np.asarray(
            rec2.get("expert_raw_action", [np.nan, np.nan]), dtype=np.float64
        ).reshape(-1)[:2]
        # 实际未来 6 决策步（Pass 2 实际状态，自车系）
        actual_future = np.full((HORIZON, 2), np.nan, dtype=np.float64)
        actual_mask = False
        future_poses = []
        for k in range(1, HORIZON + 1):
            index = step + STEPS_PER_POLICY * k
            if index < len(controller2.records):
                future_poses.append(_record_pose(controller2.records[index]))
            else:
                future_poses.append(None)
        if all(p is not None for p in future_poses):
            actual_future = ego_relative_traj(np.asarray(future_poses, dtype=np.float64), pose)
            actual_mask = True
        # student 侧（Pass 1 同 step 对齐；k≥1 行为同时间索引反事实对照）
        student_plan = np.full((HORIZON, 2), np.nan, dtype=np.float64)
        student_mu = np.full(2, np.nan, dtype=np.float64)
        student_actual = np.full((HORIZON, 2), np.nan, dtype=np.float64)
        student_actual_mask = False
        obs = None
        if step < len(controller1.records):
            rec1 = controller1.records[step]
            plan1 = rec1.get("plan")
            if plan1 is not None:
                student_plan = np.asarray(plan1, dtype=np.float64).reshape(-1, 2)
            mu1 = rec1.get("mu")
            if mu1 is not None:
                student_mu = np.asarray(mu1, dtype=np.float64).reshape(-1)[:2]
            future1 = []
            for k in range(1, HORIZON + 1):
                index = step + STEPS_PER_POLICY * k
                future1.append(
                    _record_pose(controller1.records[index])
                    if index < len(controller1.records)
                    else None
                )
            if all(p is not None for p in future1):
                student_actual = ego_relative_traj(np.asarray(future1, dtype=np.float64), pose)
                student_actual_mask = True
            obs = controller1.obs_by_step.get(step)
        line_types = controller2.lane_lines_by_step.get(step, {"left_id": 0.0, "right_id": 0.0})
        footprint = rec2.get("plan_footprint") or {}
        surface_inside = _safe_float(footprint.get("inside_ratio"))
        surface_valid = footprint.get("valid")
        lane_lat = _safe_float(rec2.get("lane_lat"))
        rc_step = _safe_float(controller2.steps[step - 1].get("rc")) if step > 0 else 0.0
        nav_cps = rec2.get("nav_cps")
        label, confidence = mode_label(recovery_traj, nav_cps=nav_cps, pose=pose)

        row: dict[str, Any] = {
            "episode_id": info["id"],
            "pool": pool,
            "geometry": info["geometry"],
            "step": int(step),
            "class": window_class,
            "window_index": int(window_index),
            "pose": pose,
            "student_plan": student_plan,
            "student_mu": student_mu,
            "student_actual_traj": student_actual,
            "student_actual_mask": bool(student_actual_mask),
            "expert_action_chain": plan2,
            "expert_recovery_traj": recovery_traj,
            "expert_actual_recovery_traj": actual_future,
            "expert_actual_mask": bool(actual_mask),
            "expert_raw_action": raw_action,
            "lane_lat": lane_lat,
            "rc": rc_step,
            "T_plan": anchors["T_plan"],
            "T_cross": anchors["T_cross"],
            "T_cross_source": anchors["T_cross_source"] or "",
            "first_low_margin": anchors["first_low_margin"],
            "takeover_step": int(takeover_step),
            "margin_before": quality["margin_before_window"],
            "margin_after": quality["margin_after_tail"],
            "rc_before": quality["rc_before"],
            "rc_after": quality["rc_after"],
            "takeover_off_road": bool(quality["off_road"]),
            "takeover_collision": bool(quality["collision"]),
            "expert_recovered": bool(quality["recovered"]),
            "recovery_success_horizon": int(quality["success_horizon_steps"]),
            "min_margin_after_takeover": quality["min_margin_after_takeover"],
            "route_progress_after_takeover": quality["route_progress_after_takeover"],
            "mode_label": label,
            "mode_confidence": float(confidence),
            "geometry_labels": {
                "surface_valid": surface_valid,
                "surface_inside_ratio": surface_inside,
                "solid_left": int(line_types["left_id"]) in SOLID_LINE_IDS,
                "solid_right": int(line_types["right_id"]) in SOLID_LINE_IDS,
                "route_rc": rc_step,
            },
            "geom_surface_valid": surface_valid,
            "geom_surface_inside_ratio": surface_inside,
            "geom_solid_left": int(line_types["left_id"]) in SOLID_LINE_IDS,
            "geom_solid_right": int(line_types["right_id"]) in SOLID_LINE_IDS,
            "geom_route_rc": rc_step,
            "obs": obs,
            "termination": info["termination"],
            "map_id": rec2.get("map_id"),
            "road_class": rec2.get("road_class"),
        }
        gate_result = gates.evaluate_row(
            build_gate_row(row), gates.GateParams(action_traj_tol_m=args.action_traj_tol)
        )
        row["usable"] = bool(gate_result["usable"])
        row["quality_reasons"] = list(gate_result["reasons"])
        rows.append(row)
    return rows


def collect_episode(spec, pool: str, config, task, *, args, usable_counts: dict[str, int]) -> dict:
    """一条 episode：Pass 1 → 多窗口（每档 ≤1，配额感知）Pass 2 → 行。

    返回 ``{"status": "collected"|"skipped", ...}``；``collected`` 时含 ``rows``。
    """
    started = time.time()
    episode_id = int(spec.id)
    info: dict[str, Any] = {
        "id": episode_id,
        "pool": pool,
        "seed": int(getattr(spec, "seed", -1)),
        "geometry": str(getattr(spec, "labels", {}).get("geometry", "?")),
        "difficulty": str(getattr(spec, "difficulty", "?")),
    }
    controller1, outcome1, analysis1 = _run_pass1(
        spec, config, task, max_steps=args.max_steps
    )
    info["termination"] = str(outcome1.get("termination"))
    info["rc_final"] = _safe_float(outcome1.get("rc_final"))
    info["episode_len_steps"] = len(controller1.steps)

    decision_lane_lat = {
        int(rec["step"]): _safe_float(rec.get("lane_lat"))
        for rec in controller1.records
        if rec.get("decision")
    }
    env_failure_step = None
    for row in controller1.steps:
        if row.get("out_of_road") or row.get("crash"):
            env_failure_step = int(row["step"])
            break
    t_plan = _ts_step(analysis1.get("timestamps"), "T_plan")
    t_cross_footprint = _ts_step(analysis1.get("timestamps"), "T_cross")
    if t_cross_footprint is not None:
        t_cross, t_cross_source = t_cross_footprint, "footprint"
    elif env_failure_step is not None:
        t_cross, t_cross_source = env_failure_step, "env_event"
    else:
        t_cross, t_cross_source = None, None
    first_low = first_low_margin_step(decision_lane_lat, args.low_lat)
    info["anchors"] = {
        "T_plan": t_plan,
        "T_cross": t_cross,
        "T_cross_footprint": t_cross_footprint,
        "T_cross_source": t_cross_source,
        "first_low_margin": first_low,
    }

    max_decision = max(0, (len(controller1.steps) - 1) // STEPS_PER_POLICY) * STEPS_PER_POLICY
    candidates = generate_window_candidates(
        t_plan=t_plan,
        t_cross=t_cross,
        first_low_margin=first_low,
        context_steps=args.context_steps,
        recovery_steps=args.recovery_steps,
        late_margin_steps=args.late_margin_steps,
        max_decision_step=max_decision,
    )

    def _state_at(step: int) -> dict:
        record = controller1.records[step] if 0 <= step < len(controller1.records) else {}
        return {
            "lane_lat": _safe_float(record.get("lane_lat")),
            "plan_footprint_valid": record.get("plan_footprint_valid"),
            "exec_footprint_valid": record.get("exec_footprint_valid"),
        }

    windows = select_windows(
        candidates,
        _state_at,
        t_plan=t_plan,
        t_cross=t_cross,
        low_lat=args.low_lat,
        late_lat=args.late_lat,
        late_margin_steps=args.late_margin_steps,
        max_windows=args.max_windows_per_episode,
    )
    if not windows:
        info.update(
            {"status": "skipped", "reason": "no_window", "candidates": candidates,
             "wall_s": round(time.time() - started, 1)}
        )
        return info
    windows = apply_quota_policy(
        windows,
        usable_counts,
        args.targets,
        stop_at_target=not getattr(args, "no_stop_at_target", False),
    )
    if not windows:
        info.update(
            {"status": "skipped", "reason": "quota_satisfied",
             "wall_s": round(time.time() - started, 1)}
        )
        return info

    windows_info: list[dict] = []
    all_rows: list[dict] = []
    for index, (window_class, takeover_step) in enumerate(windows):
        result = _collect_one_window(
            spec, config, task, controller1, takeover_step=takeover_step, args=args
        )
        entry: dict[str, Any] = {
            "index": index,
            "class": window_class,
            "takeover_step": int(takeover_step),
            "status": result["status"],
        }
        if result["status"] == "collected":
            controller2 = result.pop("controller2")
            rows = _build_window_rows(
                controller1,
                controller2,
                window_class=window_class,
                window_index=index,
                takeover_step=takeover_step,
                pool=pool,
                info=info,
                quality=result["expert"],
                args=args,
            )
            entry.update(
                {
                    "determinism": result["determinism"],
                    "expert": result["expert"],
                    "n_rows": len(rows),
                    "n_usable": sum(1 for row in rows if row["usable"]),
                }
            )
            for row in rows:
                if row["usable"]:
                    usable_counts[window_class] = usable_counts.get(window_class, 0) + 1
            all_rows.extend(rows)
        else:
            entry.update({"reason": result.get("reason"), "attempts": result.get("attempts")})
        windows_info.append(entry)

    if not all_rows:
        info.update(
            {
                "status": "skipped",
                "reason": "all_windows_skipped",
                "windows": windows_info,
                "wall_s": round(time.time() - started, 1),
            }
        )
        return info
    info.update(
        {
            "status": "collected",
            "windows": windows_info,
            "n_rows": len(all_rows),
            "n_usable": sum(1 for row in all_rows if row["usable"]),
            "rows": all_rows,
            "wall_s": round(time.time() - started, 1),
        }
    )
    return info


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
_SCHEMA_FIELDS: list[dict[str, Any]] = [
    {"name": "episode_id", "shape": [], "dtype": "int64", "semantics": "spec id（池内）"},
    {"name": "pool", "shape": [], "dtype": "str", "semantics": "池名（spec 文件名 stem）；与 episode_id 组成全局键"},
    {"name": "geometry", "shape": [], "dtype": "str", "semantics": "episode 几何标签（spec.labels.geometry）"},
    {"name": "step", "shape": [], "dtype": "int64", "semantics": "窗口锚点 env step（决策步，5 的倍数）"},
    {"name": "class_id", "shape": [], "dtype": "int8", "semantics": "0=prevention/1=correction/2=late（窗口接管时刻实况分类）"},
    {"name": "window_index", "shape": [], "dtype": "int8", "semantics": "episode 内窗口序号（0..2）"},
    {"name": "takeover_step", "shape": [], "dtype": "int64", "semantics": "窗口接管 env step（学生→专家）"},
    {"name": "T_plan", "shape": [], "dtype": "float32", "semantics": "首次计划 footprint 不可行（NaN=无）"},
    {"name": "T_cross", "shape": [], "dtype": "float32", "semantics": "effective 首次越界步：footprint 越界，缺省回退 env out_of_road/crash 首步（NaN=无）"},
    {"name": "T_cross_source", "shape": [], "dtype": "str", "semantics": "T_cross 来源：footprint / env_event / 空"},
    {"name": "first_low_margin", "shape": [], "dtype": "float32", "semantics": "首个 |lane_lat|>阈值 决策步（margin 代理；NaN=无）"},
    {"name": "lane_lat", "shape": [], "dtype": "float32", "semantics": "锚点状态横向偏移（Pass 2 实际）"},
    {"name": "rc", "shape": [], "dtype": "float32", "semantics": "锚点 route completion"},
    {"name": "student_plan", "shape": [6, 2], "dtype": "float32", "semantics": "学生 plan (ds,dθ)（Pass 1 对齐；NaN=缺）"},
    {"name": "student_mu", "shape": [2], "dtype": "float32", "semantics": "学生 action_mu"},
    {"name": "student_actual_traj", "shape": [6, 2], "dtype": "float32", "semantics": "学生实际未来 6 决策步（自车系；NaN=缺）"},
    {"name": "student_actual_mask", "shape": [], "dtype": "bool", "semantics": "student_actual_traj 是否完整"},
    {"name": "expert_action_chain", "shape": [6, 2], "dtype": "float32", "semantics": "专家接管 plan (ds,dθ)（Pass 2 实际）"},
    {"name": "expert_recovery_traj", "shape": [6, 2], "dtype": "float32", "semantics": "专家恢复轨迹（plan 积分，自车系）"},
    {"name": "expert_actual_recovery_traj", "shape": [6, 2], "dtype": "float32", "semantics": "专家实际未来 6 决策步（自车系；NaN=截断）"},
    {"name": "expert_actual_mask", "shape": [], "dtype": "bool", "semantics": "expert_actual_recovery_traj 是否完整"},
    {"name": "expert_raw_action", "shape": [2], "dtype": "float32", "semantics": "专家原始 [steer, throttle]"},
    {"name": "margin_before", "shape": [], "dtype": "float32", "semantics": "恢复窗 before：接管前 context 窗最差 -|lane_lat|（episode 级；margin 代理）"},
    {"name": "margin_after", "shape": [], "dtype": "float32", "semantics": "恢复窗 after：相位末 5 步 -|lane_lat| 均值（episode 级）"},
    {"name": "rc_before", "shape": [], "dtype": "float32", "semantics": "接管时 rc（episode 级）"},
    {"name": "rc_after", "shape": [], "dtype": "float32", "semantics": "接管 3s 末 rc（episode 级）"},
    {"name": "takeover_off_road", "shape": [], "dtype": "bool", "semantics": "接管后 off-road（相位内或 Pass2 horizon 内 env 终止）"},
    {"name": "takeover_collision", "shape": [], "dtype": "bool", "semantics": "接管后碰撞（同上）"},
    {"name": "expert_recovered", "shape": [], "dtype": "bool", "semantics": "接管后未以 off/collision 失败（含 Pass2 horizon）"},
    {"name": "recovery_success_horizon", "shape": [], "dtype": "int32", "semantics": "接管后存活 env steps（≤30）"},
    {"name": "min_margin_after_takeover", "shape": [], "dtype": "float32", "semantics": "见 margin_after"},
    {"name": "route_progress_after_takeover", "shape": [], "dtype": "float32", "semantics": "rc_end − rc_takeover"},
    {"name": "mode_id", "shape": [], "dtype": "int8", "semantics": "恢复轨迹粗标签 id（分析用）"},
    {"name": "mode_confidence", "shape": [], "dtype": "float32", "semantics": "标签置信（启发式）"},
    {"name": "geom_surface_valid", "shape": [], "dtype": "bool", "semantics": "plan footprint 全可行（NaN→False）"},
    {"name": "geom_surface_inside_ratio", "shape": [], "dtype": "float32", "semantics": "plan footprint inside_ratio"},
    {"name": "geom_solid_left", "shape": [], "dtype": "bool", "semantics": "当前车道左线为实线"},
    {"name": "geom_solid_right", "shape": [], "dtype": "bool", "semantics": "当前车道右线为实线"},
    {"name": "geom_route_rc", "shape": [], "dtype": "float32", "semantics": "route completion"},
    {"name": "usable", "shape": [], "dtype": "bool", "semantics": "通过 recovery_quality_gates 全部六门"},
    {"name": "quality_reasons_json", "shape": [], "dtype": "str", "semantics": "质量门 reasons（JSON 数组）"},
    {"name": "class_label", "shape": [], "dtype": "str", "semantics": "prevention/correction/late"},
    {"name": "mode_label", "shape": [], "dtype": "str", "semantics": "恢复轨迹粗标签"},
    {"name": "termination", "shape": [], "dtype": "str", "semantics": "Pass 1 episode 终止"},
    {"name": "map_id", "shape": [], "dtype": "str", "semantics": "地图标识"},
    {"name": "road_class", "shape": [], "dtype": "str", "semantics": "道路类别"},
    {"name": "obs__<key>", "shape": "varied", "dtype": "float32", "semantics": "student obs 全键（含 history/mask；Pass 1 对齐）"},
]


def _stack_rows(rows: list[dict]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    n = len(rows)

    def stack(name: str, dtype: Any) -> np.ndarray:
        return np.asarray([row.get(name) for row in rows], dtype=dtype)

    arrays["episode_id"] = stack("episode_id", np.int64)
    arrays["pool"] = np.asarray([str(row.get("pool") or "") for row in rows], dtype="<U32")
    arrays["geometry"] = np.asarray([str(row.get("geometry") or "") for row in rows], dtype="<U24")
    arrays["step"] = stack("step", np.int64)
    arrays["class_id"] = np.asarray([CLASS_IDS[row["class"]] for row in rows], dtype=np.int8)
    arrays["window_index"] = stack("window_index", np.int8)
    arrays["takeover_step"] = stack("takeover_step", np.int64)
    for name in ("T_plan", "T_cross", "first_low_margin"):
        arrays[name] = np.asarray(
            [np.nan if row.get(name) is None else float(row[name]) for row in rows], dtype=np.float32
        )
    arrays["T_cross_source"] = np.asarray(
        [str(row.get("T_cross_source") or "") for row in rows], dtype="<U12"
    )
    arrays["lane_lat"] = np.asarray(
        [np.nan if row.get("lane_lat") is None else float(row["lane_lat"]) for row in rows], dtype=np.float32
    )
    arrays["rc"] = np.asarray(
        [np.nan if row.get("rc") is None else float(row["rc"]) for row in rows], dtype=np.float32
    )
    arrays["student_plan"] = np.asarray([row["student_plan"] for row in rows], dtype=np.float32)
    arrays["student_mu"] = np.asarray([row["student_mu"] for row in rows], dtype=np.float32)
    arrays["student_actual_traj"] = np.asarray(
        [row["student_actual_traj"] for row in rows], dtype=np.float32
    )
    arrays["student_actual_mask"] = np.asarray(
        [row["student_actual_mask"] for row in rows], dtype=bool
    )
    arrays["expert_action_chain"] = np.asarray(
        [row["expert_action_chain"] for row in rows], dtype=np.float32
    )
    arrays["expert_recovery_traj"] = np.asarray(
        [row["expert_recovery_traj"] for row in rows], dtype=np.float32
    )
    arrays["expert_actual_recovery_traj"] = np.asarray(
        [row["expert_actual_recovery_traj"] for row in rows], dtype=np.float32
    )
    arrays["expert_actual_mask"] = np.asarray([row["expert_actual_mask"] for row in rows], dtype=bool)
    arrays["expert_raw_action"] = np.asarray([row["expert_raw_action"] for row in rows], dtype=np.float32)
    for name in ("margin_before", "margin_after", "rc_before", "rc_after", "min_margin_after_takeover", "route_progress_after_takeover"):
        arrays[name] = np.asarray(
            [np.nan if row.get(name) is None else float(row[name]) for row in rows], dtype=np.float32
        )
    arrays["takeover_off_road"] = np.asarray([row["takeover_off_road"] for row in rows], dtype=bool)
    arrays["takeover_collision"] = np.asarray([row["takeover_collision"] for row in rows], dtype=bool)
    arrays["expert_recovered"] = np.asarray([row["expert_recovered"] for row in rows], dtype=bool)
    arrays["recovery_success_horizon"] = np.asarray(
        [row["recovery_success_horizon"] for row in rows], dtype=np.int32
    )
    arrays["mode_id"] = np.asarray([MODE_IDS[row["mode_label"]] for row in rows], dtype=np.int8)
    arrays["mode_confidence"] = np.asarray([row["mode_confidence"] for row in rows], dtype=np.float32)
    arrays["geom_surface_valid"] = np.asarray(
        [bool(row["geom_surface_valid"]) if row.get("geom_surface_valid") is not None else False for row in rows],
        dtype=bool,
    )
    arrays["geom_surface_inside_ratio"] = np.asarray(
        [np.nan if row.get("geom_surface_inside_ratio") is None else float(row["geom_surface_inside_ratio"]) for row in rows],
        dtype=np.float32,
    )
    arrays["geom_solid_left"] = np.asarray([row["geom_solid_left"] for row in rows], dtype=bool)
    arrays["geom_solid_right"] = np.asarray([row["geom_solid_right"] for row in rows], dtype=bool)
    arrays["geom_route_rc"] = np.asarray(
        [np.nan if row.get("geom_route_rc") is None else float(row["geom_route_rc"]) for row in rows],
        dtype=np.float32,
    )
    arrays["usable"] = np.asarray([row["usable"] for row in rows], dtype=bool)
    arrays["quality_reasons_json"] = np.asarray(
        [json.dumps(row["quality_reasons"], ensure_ascii=False) for row in rows], dtype="<U512"
    )
    arrays["class_label"] = np.asarray([row["class"] for row in rows], dtype="<U16")
    arrays["mode_label"] = np.asarray([row["mode_label"] for row in rows], dtype="<U24")
    arrays["termination"] = np.asarray([str(row.get("termination") or "") for row in rows], dtype="<U24")
    arrays["map_id"] = np.asarray([str(row.get("map_id") or "") for row in rows], dtype="<U64")
    arrays["road_class"] = np.asarray([str(row.get("road_class") or "") for row in rows], dtype="<U64")

    obs_keys: list[str] = sorted({key for row in rows for key in (row.get("obs") or {})})
    for key in obs_keys:
        values = []
        shape = None
        for row in rows:
            value = (row.get("obs") or {}).get(key)
            if value is None:
                if shape is None:
                    values.append(None)
                    continue
                values.append(np.full(shape, np.nan, dtype=np.float32))
            else:
                arr = np.asarray(value, dtype=np.float32)
                shape = arr.shape if shape is None else shape
                values.append(arr)
        if values and values[0] is not None:
            arrays[f"obs__{key}"] = np.stack(values).astype(np.float32)
        else:
            arrays[f"obs__{key}"] = np.full((n, 1), np.nan, dtype=np.float32)
    return arrays


def build_pool_plan(
    rows: list[dict],
    episodes: list[dict],
    pools_info: list[dict],
    targets: dict[str, int],
    *,
    hard_types: tuple[str, ...] = HARD_ROAD_TYPES,
    min_hard_rows: int = HARD_TYPE_MIN_ROWS,
    min_hard_episodes: int = HARD_TYPE_MIN_EPISODES,
) -> dict:
    """试点 → 满门外推与池规划（纯函数；写入 manifest.pool_plan）。"""
    collected = [ep for ep in episodes if ep.get("status") == "collected"]
    usable_rows = [row for row in rows if row["usable"]]
    n_episodes = len(collected)

    per_pool: dict[str, dict] = {}
    for pool in pools_info:
        name = str(pool.get("name"))
        pool_rows = [row for row in usable_rows if row.get("pool") == name]
        pool_episodes = [ep for ep in collected if ep.get("pool") == name]
        class_counts = {window_class: 0 for window_class in WINDOW_CLASSES}
        for row in pool_rows:
            class_counts[row["class"]] = class_counts.get(row["class"], 0) + 1
        per_pool[name] = {
            "available_episodes": pool.get("n_specs"),
            "selected_episodes": pool.get("selected"),
            "collected_episodes": len(pool_episodes),
            "usable_rows": len(pool_rows),
            "usable_rows_per_episode": (
                (len(pool_rows) / len(pool_episodes)) if pool_episodes else None
            ),
            "class_usable_rows": class_counts,
        }

    class_rates = {
        window_class: (
            (sum(1 for row in usable_rows if row["class"] == window_class) / n_episodes)
            if n_episodes
            else 0.0
        )
        for window_class in WINDOW_CLASSES
    }
    required_by_class: dict[str, int | None] = {}
    for window_class, target in targets.items():
        rate = class_rates.get(window_class, 0.0)
        if int(target) <= 0:
            required_by_class[window_class] = 0
        elif rate > 0:
            required_by_class[window_class] = int(math.ceil(int(target) / rate))
        else:
            required_by_class[window_class] = None
    values = [value for value in required_by_class.values() if value is not None]
    required_total = max(values) if values and len(values) == len(required_by_class) else None

    projection: dict[str, dict] = {}
    for name, pool_info in per_pool.items():
        rate = pool_info["usable_rows_per_episode"] or 0.0
        available = pool_info["available_episodes"] or 0
        collected_here = pool_info["collected_episodes"]
        projection[name] = {
            "available_episodes": available,
            "projected_usable_rows": int(round(available * rate)),
            "projected_class_rows": {
                window_class: (
                    int(round(available * (pool_info["class_usable_rows"].get(window_class, 0) / collected_here)))
                    if collected_here
                    else 0
                )
                for window_class in WINDOW_CLASSES
            },
        }
    projected_total = sum(info["projected_usable_rows"] for info in projection.values())

    hard: dict[str, dict] = {}
    for hard_type in hard_types:
        type_rows = [
            row
            for row in usable_rows
            if row.get("geometry") == hard_type or row.get("road_class") == hard_type
        ]
        type_episodes = {(row.get("pool"), row["episode_id"]) for row in type_rows}
        rate_rows = (len(type_rows) / n_episodes) if n_episodes else 0.0
        rate_episodes = (len(type_episodes) / n_episodes) if n_episodes else 0.0
        hard[hard_type] = {
            "usable_rows": len(type_rows),
            "episodes": len(type_episodes),
            "rows_gap": max(0, int(min_hard_rows) - len(type_rows)),
            "episodes_gap": max(0, int(min_hard_episodes) - len(type_episodes)),
            "required_episodes_by_rows": (
                int(math.ceil(int(min_hard_rows) / rate_rows)) if rate_rows > 0 else None
            ),
            "required_episodes_by_episodes": (
                int(math.ceil(int(min_hard_episodes) / rate_episodes)) if rate_episodes > 0 else None
            ),
        }

    missing_classes = [
        window_class for window_class, value in required_by_class.items() if value is None
    ]
    recommendation = (
        f"按试点速率三档所需 episode={required_by_class}（总需求={required_total}）；"
        f"全池投影（可用 episode × 试点可用行/集）={projected_total} 行；"
        + (f"警告：{missing_classes} 试点零产出，需先解决该类窗口生成；" if missing_classes else "")
        + f"困难路型每类需 ≥{min_hard_rows} 行且 ≥{min_hard_episodes} eps（见 hard_types）。"
    )
    return {
        "episodes_collected": n_episodes,
        "class_rates_per_episode": class_rates,
        "required_episodes_by_class": required_by_class,
        "required_episodes_total": required_total,
        "per_pool_yield": per_pool,
        "projection_all_pools": projection,
        "projected_usable_rows_all_pools": projected_total,
        "hard_types": hard,
        "recommendation": recommendation,
    }


def _write_outputs(
    out_dir: Path,
    rows: list[dict],
    episodes: list[dict],
    args,
    pools_info: list[dict] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = _stack_rows(rows)
    np.savez_compressed(out_dir / "samples.npz", **arrays)

    schema = {
        "schema_version": 2,
        "created_by": "tools/controls/collect_recovery.py",
        "row_semantics": (
            "每行 = 某窗口 Pass 2 实际接管阶段的一个决策步状态；expert 侧数据与 expert 接管同源"
            "（Pass 2 实际），student 侧为 Pass 1 同 step 对齐（k≥1 行为同时间索引反事实对照）。"
            "同一 episode 最多 3 个窗口（每档 ≤1），≤20 行/集。"
        ),
        "horizon": {"steps": HORIZON, "dt_s": POLICY_DT, "ego_frame": "x 前向 / y 左向"},
        "quality_gates": {
            "module": "tools/controls/recovery_quality_gates.py",
            "mapping": "usable = evaluate_row(build_gate_row(row)) 全门通过；不合格行保留 usable=False",
            "params": {"action_traj_tol_m": args.action_traj_tol},
        },
        "class_definition": {
            "prevention": "接管时刻计划可行且 margin 健康（|lane_lat| < low_lat 且 plan footprint 可行）",
            "correction": "接管时刻计划已不可行（plan footprint 不可行 / ≥T_plan）或 |lane_lat| ≥ low_lat",
            "late": "接管时刻已越界/接近压线（exec footprint 不可行 / ≥T_cross−late_margin / |lane_lat| ≥ late_lat）",
            "thresholds": {
                "low_lat": args.low_lat,
                "late_lat": args.late_lat,
                "late_margin_steps": args.late_margin_steps,
            },
            "target_ratio": CLASS_TARGET_RATIO,
            "window_selection": "prevention 取最晚、correction 取最早、late 取最晚；每档 ≤1，≤3 窗口/集",
        },
        "fields": _SCHEMA_FIELDS,
    }
    (out_dir / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    collected = [ep for ep in episodes if ep.get("status") == "collected"]
    skipped = [ep for ep in episodes if ep.get("status") == "skipped"]
    all_windows = [window for ep in collected for window in ep.get("windows", [])]
    gate_rows = [build_gate_row(row) for row in rows]
    summary, _results = gates.build_summary(
        gate_rows, gates.GateParams(action_traj_tol_m=args.action_traj_tol), Path("pass2_takeover")
    )
    class_counts: dict[str, int] = {window_class: 0 for window_class in WINDOW_CLASSES}
    usable_class_counts: dict[str, int] = {window_class: 0 for window_class in WINDOW_CLASSES}
    for row in rows:
        class_counts[row["class"]] = class_counts.get(row["class"], 0) + 1
        if row["usable"]:
            usable_class_counts[row["class"]] = usable_class_counts.get(row["class"], 0) + 1
    quota = {
        window_class: {
            "target": int(args.targets.get(window_class, 0)),
            "usable": int(usable_class_counts.get(window_class, 0)),
            "met": int(usable_class_counts.get(window_class, 0)) >= int(args.targets.get(window_class, 0)),
            "gap": max(0, int(args.targets.get(window_class, 0)) - int(usable_class_counts.get(window_class, 0))),
        }
        for window_class in WINDOW_CLASSES
    }
    per_class: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = per_class.setdefault(row["class"], {"rows": 0, "usable": 0})
        bucket["rows"] += 1
        bucket["usable"] += int(row["usable"])

    def _coverage(row_subset: list[dict], key_fn) -> dict[str, dict]:
        coverage: dict[str, dict] = {}
        for row in row_subset:
            key = str(key_fn(row) or "?")
            bucket = coverage.setdefault(key, {"episodes": set(), "rows": 0, "usable_rows": 0})
            bucket["episodes"].add(f"{row['pool']}:{row['episode_id']}")
            bucket["rows"] += 1
            bucket["usable_rows"] += int(row["usable"])
        return {
            key: {
                "episodes": len(value["episodes"]),
                "rows": value["rows"],
                "usable_rows": value["usable_rows"],
            }
            for key, value in coverage.items()
        }

    road_type_coverage = {
        "by_episode_geometry": _coverage(rows, lambda row: row.get("geometry")),
        "by_row_road_class": _coverage(rows, lambda row: row.get("road_class")),
        "hard_types": list(HARD_ROAD_TYPES),
    }
    pools_info = pools_info or []
    pool_plan = build_pool_plan(rows, episodes, pools_info, args.targets)

    quality = {
        **summary,
        "input": "Pass 2 expert takeover rows (in-memory)",
        "per_class": per_class,
        "quota": quota,
        "class_target_ratio": CLASS_TARGET_RATIO,
        "per_pool": pool_plan["per_pool_yield"],
        "road_type_coverage": road_type_coverage,
        "per_episode": [
            {
                "id": ep["id"],
                "pool": ep.get("pool"),
                "windows": [
                    {"class": window["class"], "takeover_step": window["takeover_step"], "status": window["status"]}
                    for window in ep.get("windows", [])
                ],
                "n_rows": ep.get("n_rows", 0),
                "n_usable": ep.get("n_usable", 0),
            }
            for ep in collected
        ],
    }
    (out_dir / "quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def _episode_key(ep: dict) -> str:
        return f"{ep.get('pool')}:{ep['id']}"

    episode_keys = [_episode_key(ep) for ep in collected]
    train, val = split_episodes(episode_keys)
    train_set, val_set = set(train), set(val)

    def _counts(keys: set[str]) -> dict[str, int]:
        subset = [row for row in rows if f"{row['pool']}:{row['episode_id']}" in keys]
        return {
            "episodes": len(keys),
            "rows": len(subset),
            "usable_rows": sum(1 for row in subset if row["usable"]),
        }

    split_manifest = {
        "rule": "sorted('pool:id') 每 5 个末位进 val（episode 级；行不跨集）",
        "no_leakage": len(train_set & val_set) == 0,
        "train_episodes": train,
        "val_episodes": val,
        "train": _counts(train_set),
        "val": _counts(val_set),
        "generated_at": _now(),
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    divergence = {
        "windows_collected": sum(1 for window in all_windows if window.get("status") == "collected"),
        "strict": sum(
            1 for window in all_windows if window.get("determinism", {}).get("accepted") == "strict"
        ),
        "relaxed": sum(
            1 for window in all_windows if window.get("determinism", {}).get("accepted") == "relaxed"
        ),
        "skipped_state_mismatch": sum(
            1 for window in all_windows if window.get("reason") == "pass2_state_mismatch"
        ),
        "divergence_rate": (
            (
                sum(1 for window in all_windows if window.get("status") == "collected")
                - sum(1 for window in all_windows if window.get("determinism", {}).get("accepted") == "strict")
            )
            / max(1, sum(1 for window in all_windows if window.get("status") == "collected"))
        ),
    }
    manifest = {
        "tool": "tools/controls/collect_recovery.py",
        "created_at": _now(),
        "git_commit": _git_commit(),
        "shard": {"index": int(getattr(args, "shard", (0, 1))[0]), "count": int(getattr(args, "shard", (0, 1))[1])},
        "args": vars(args),
        "ckpt": {"path": str(args.ckpt), "sha256": _sha256_file(Path(args.ckpt)) if Path(args.ckpt).is_file() else None},
        "config": {"path": str(args.config), "sha256": _sha256_file(Path(args.config)) if Path(args.config).is_file() else None},
        "pools": pools_info,
        "targets": args.targets,
        "protocol": {
            "tracker": "lqr",
            "eval_reference": "plan",
            "max_steps": args.max_steps,
            "device": args.device,
            "workers": args.workers,
            "expert_steps": args.expert_steps,
        },
        "window_params": {
            "context_steps": args.context_steps,
            "recovery_steps": args.recovery_steps,
            "low_lat": args.low_lat,
            "late_lat": args.late_lat,
            "late_margin_steps": args.late_margin_steps,
            "max_windows_per_episode": args.max_windows_per_episode,
            "max_rows_per_episode": args.max_rows_per_episode,
        },
        "counts": {
            "episodes_attempted": len(episodes),
            "episodes_collected": len(collected),
            "episodes_skipped": len(skipped),
            "windows_collected": divergence["windows_collected"],
            "rows_total": len(rows),
            "rows_usable": sum(1 for row in rows if row["usable"]),
            "class_counts": class_counts,
            "usable_class_counts": usable_class_counts,
        },
        "quota": quota,
        "divergence": divergence,
        "road_type_coverage": road_type_coverage,
        "pool_plan": pool_plan,
        "field_gaps": [
            "min_signed_margin 恒 None（forensics 设计：引擎只有布尔 on-lane 查询）→ margin 用 -|lane_lat| 代理；"
            "质量门 margin_before/after = 窗口级恢复窗端点（context 窗最差 / 相位末 5 步均值），"
            "min_margin_after_takeover 单列为相位内最差值",
            "T_cross 的 footprint 口径对'线压线类 out_of_road'恒 None（on-lane 射线查询不触发）→ "
            "effective T_cross 回退到 env out_of_road/crash 首步（T_cross_source=env_event），"
            "否则 late 类结构上不可达",
            "窗口分类为接管时刻实况（|lane_lat|/plan·exec footprint/T_cross 邻域），非时序位置；"
            "prevention 取最晚、correction 取最早、late 取最晚，每档 ≤1",
            "k≥1 行的 student 字段为 Pass 1 同 step 反事实对照（该时刻 Pass 2 已由专家驱动）",
            "expert_recovery_traj = expert_action_chain 的运动学积分（标签）；实际执行见 expert_actual_recovery_traj",
            "困难路型计数：episode 级用 spec.labels.geometry，行级用 road_class（coverage 两列都报）",
        ],
        "episodes": [ep for ep in episodes],
        "skipped": [
            {"id": ep["id"], "pool": ep.get("pool"), "reason": ep.get("reason"), "attempts": ep.get("attempts")}
            for ep in skipped
        ],
        "generated_at": _now(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _git_commit() -> str | None:
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return proc.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _maybe_write_dataset_card(out_dir: Path) -> None:
    """正式采集（--out 在 datasets/ 下）时补数据集卡模板（不存在才写）。"""
    try:
        if "datasets" not in out_dir.parts:
            return
        card = _ROOT / "docs" / "recovery_v1_dataset_card.md"
        if card.exists():
            return
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(
            "# recovery_v1 数据集卡（模板，由 collect_recovery.py 生成）\n\n"
            "> TODO：正式采集完成后填写。字段契约见 `<dataset>/schema.json` 与\n"
            "> `tools/controls/recovery_quality_gates.py` docstring。\n\n"
            "## 采集 provenance\n\n- ckpt / spec / config sha256：见 `<dataset>/manifest.json`\n"
            "- 日期 / 操作者：TODO\n\n"
            "## 数据门（§7.10-A5）\n\n"
            "- 有效行 ≥10,000 / 独立 episode ≥500 / 单 episode ≤20 行 / 质量门通过率 ≥80%\n"
            "- prevention ≥4,000 / correction ≥2,500 / late ≥1,000\n"
            "- 困难路型每类 ≥300 行且 ≥30 episodes\n\n"
            "## 已知缺口\n\n- TODO（对照 manifest.field_gaps）\n",
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001 - 数据集卡尽力而为
        pass


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="collect_recovery.py", description="recovery 采集器（多窗口两遍重放 + 专家接管 + 配额）")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="学生起点 ckpt")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="臂配置（含 moe.primary.hidden_dim 等）")
    parser.add_argument(
        "--spec",
        action="append",
        default=None,
        help="spec 文件，可重复；entry = path 或 path=N（该池前 N）。缺省 eval500",
    )
    parser.add_argument("--ids", default="", help="逗号分隔 spec id（跨池按 id 选择）")
    parser.add_argument(
        "--stratify-per-geometry",
        type=int,
        default=0,
        help="每 (池, 几何) 最多 N 个（round-robin），配合 --limit 做均衡抽样",
    )
    parser.add_argument("--limit", type=int, default=0, help="全局 episode 上限（0=不限）")
    parser.add_argument("--out", default=DEFAULT_OUT, help="输出目录（试点默认 runs/p1_0/recovery_dryrun）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1, help="试点仅支持 1（顺序）；正式采集另行扩展")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--context-steps", type=int, default=20, help="T_plan − context（env steps，默认 2s）")
    parser.add_argument("--recovery-steps", type=int, default=30, help="T_cross − recovery（env steps，默认 3s）")
    parser.add_argument(
        "--low-lat", "--low-margin-lat", dest="low_lat", type=float, default=1.0,
        help="correction 阈值 / first_low_margin 的 |lane_lat|（m）",
    )
    parser.add_argument("--late-lat", type=float, default=1.5, help="late 阈值 |lane_lat|（m，接近压线）")
    parser.add_argument("--late-margin-steps", type=int, default=5, help="T_cross 邻域宽度（env steps，默认 0.5s）")
    parser.add_argument("--max-windows-per-episode", type=int, default=3)
    parser.add_argument("--max-rows-per-episode", type=int, default=20)
    parser.add_argument("--expert-steps", type=int, default=6, help="每窗口接管后决策步数（默认 6 = 3s）")
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=parse_targets("prevention=4000,correction=2500,late=1000"),
        help="每档目标有效行数（达到后该档停止加采）",
    )
    parser.add_argument(
        "--shard",
        type=parse_shard,
        default=parse_shard("0/1"),
        help="分片 i/n（0-based；在池选择/分层完成后对稳定排序的 (池,id) 列表划分）",
    )
    parser.add_argument(
        "--no-stop-at-target",
        action="store_true",
        help="按全池自然分布采集（targets 仅作报告；默认达到目标后该档停止加采）",
    )
    parser.add_argument("--determinism-tol", type=float, default=1e-6)
    parser.add_argument("--determinism-tol-relaxed", type=float, default=1e-4)
    parser.add_argument("--action-traj-tol", type=float, default=0.5, help="质量门 action↔traj 容差（m）")
    return parser


def _parse_spec_entries(entries: list[str]) -> list[dict]:
    """``path[=N]`` → 池描述（含已加载 specs）。"""
    from env.scenario.spec import load_specs

    pools: list[dict] = []
    for entry in entries:
        text = str(entry)
        path_text, limit = text, None
        if "=" in text:
            head, _, tail = text.rpartition("=")
            if head and tail.strip().isdigit():
                path_text, limit = head, int(tail)
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"spec 不存在: {path}")
        pools.append(
            {
                "name": _pool_name(path),
                "path": str(path),
                "specs": list(load_specs(str(path))),
                "limit": limit,
            }
        )
    return pools


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.workers) != 1:
        print("[recovery] 试点阶段仅支持 --workers 1（正式采集另行扩展分片）", file=sys.stderr)
        return 2

    from pipeline import eval_runner as ev
    from pipeline.gl_runtime import ensure_gl_library_path

    ensure_gl_library_path(preload=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:  # noqa: BLE001
        pass

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        print(f"[recovery] ckpt 不存在: {ckpt_path}", file=sys.stderr)
        return 2

    spec_entries = list(args.spec) if args.spec else [DEFAULT_SPEC]
    try:
        pools = _parse_spec_entries(spec_entries)
    except FileNotFoundError as exc:
        print(f"[recovery] {exc}", file=sys.stderr)
        return 2
    pools_info = [
        {
            "name": pool["name"],
            "path": pool["path"],
            "sha256": _sha256_file(Path(pool["path"])),
            "n_specs": len(pool["specs"]),
            "limit": pool["limit"],
            "selected": 0,
        }
        for pool in pools
    ]

    config = ev.load_config(args.config)
    task = {
        "policy": "ckpt",
        "ckpt": str(ckpt_path),
        "model_config": ev.task_model_config(config),
        "obs_config": dict((config.get("env") or {}).get("obs") or {}),
        "device": str(args.device),
        "tracker": "lqr",
        "eval_reference": "plan",
    }

    selected = select_episodes(
        pools, ids=args.ids, per_geometry=args.stratify_per_geometry, limit=args.limit
    )
    shard_index, shard_count = args.shard
    selected = shard_episodes(selected, index=shard_index, count=shard_count)
    for pool_info in pools_info:
        pool_info["selected"] = sum(1 for name, _spec in selected if name == pool_info["name"])

    out_dir = Path(args.out)
    print(
        f"[recovery] ckpt={ckpt_path} config={args.config} pools={[p['name'] for p in pools]} "
        f"selected={len(selected)} shard={shard_index}/{shard_count} "
        f"targets={args.targets} stop_at_target={not args.no_stop_at_target} out={out_dir}",
        flush=True,
    )
    episodes: list[dict] = []
    rows: list[dict] = []
    usable_counts = {window_class: 0 for window_class in WINDOW_CLASSES}
    for pool_name, spec in selected:
        if not args.no_stop_at_target and all(
            usable_counts[cls] >= args.targets.get(cls, 0) for cls in WINDOW_CLASSES
        ):
            print(f"[recovery] 全部档位达到目标，停止加采（已跑 {len(episodes)} 集）", flush=True)
            break
        try:
            info = collect_episode(
                spec, pool_name, config, task, args=args, usable_counts=usable_counts
            )
        except SystemExit:
            raise
        except BaseException as exc:  # noqa: BLE001 - 单集失败不终止整 lane
            import traceback

            info = {
                "id": int(spec.id),
                "pool": pool_name,
                "status": "skipped",
                "reason": f"exception:{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        episodes.append(info)
        if info.get("status") == "collected":
            rows.extend(info.pop("rows", []))
            window_text = ",".join(
                f"{window['class']}@{window['takeover_step']}({window.get('n_rows', 0)}/{window.get('n_usable', 0)})"
                for window in info.get("windows", [])
            )
            det_text = ",".join(
                str(window.get("determinism", {}).get("accepted"))
                for window in info.get("windows", [])
            )
            print(
                f"[recovery] pool={info['pool']} id={info['id']} windows=[{window_text}] "
                f"rows={info['n_rows']} usable={info['n_usable']} det={det_text} "
                f"({info.get('wall_s')}s)",
                flush=True,
            )
        else:
            print(
                f"[recovery] pool={info['pool']} id={info['id']} SKIP reason={info.get('reason')} "
                f"({info.get('wall_s')}s)",
                flush=True,
            )
        if len(episodes) % 10 == 0:
            print(
                f"[recovery] progress episodes={len(episodes)} usable_class={usable_counts}",
                flush=True,
            )

    _write_outputs(out_dir, rows, episodes, args, pools_info)
    _maybe_write_dataset_card(out_dir)
    collected = sum(1 for ep in episodes if ep.get("status") == "collected")
    quota_gap = {
        cls: max(0, int(args.targets.get(cls, 0)) - usable_counts.get(cls, 0))
        for cls in WINDOW_CLASSES
    }
    print(
        f"[recovery] DONE episodes={len(episodes)} collected={collected} "
        f"rows={len(rows)} usable={sum(1 for row in rows if row['usable'])} "
        f"usable_class={usable_counts} quota_gap={quota_gap} → {out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
