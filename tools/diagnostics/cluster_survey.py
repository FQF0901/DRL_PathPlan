#!/usr/bin/env python3
"""Lane C 诊断：聚类现状排查 + 每簇 100 帧可视化（只读脚本；CPU-only）。

用法::

    tools/venv-python tools/diagnostics/cluster_survey.py              # 默认数据/spec/输出
    tools/venv-python tools/diagnostics/cluster_survey.py --per-cluster 100 --seed 0 \
        --out runs/BTC20260927-1300_cluster_survey

产物（全部落 ``--out``，默认 ``runs/BTC<北京戳>_cluster_survey/``；不写任何其它位置）：

- ``summary.md`` / ``summary.json``：占比（全量 / ``train_weight>0``）、与体检报告 60k 子采样
  shares 的对照、每簇画像（difficulty/geometry/labels）、指纹告警、字段语义、抽样清单、运行信息；
- ``cluster_0{k}/``：每簇 ``--per-cluster``（默认 100）张 5 联面板 PNG + ``contact_sheet.png``
  （25 图拼版；``--no-contact-sheet`` 关闭）；
- ``assignment.npz``：本次运行的全量 top-1 簇号 / top-1 软概率 / top1-top2 间隔副本。

Sidecar（供 train/eval 直接读，避免每次重算）
============================================
全量分配另写一份 sidecar 到数据集目录 ``<bc_dir>/<spec_stem>_assignments.npz``（默认
``datasets/BTC20260926-2343_expert5k/cluster_v1_assignments.npz``），键：

- ``cluster`` int16(N)：top-1 硬簇号（``argmax`` 软目标；two_stage 含活跃度门控）；
- ``top1_margin`` float32(N)：``top1_prob - second_best_prob``（软目标 top-2 间隔）；
- ``top1_prob`` float32(N)：最大软目标概率（附加键，便于直接筛"边界样本"）；
- ``spec_hash``（spec npz 内容 sha256）/ ``spec_path`` / ``rows`` / ``created_at``；
- ``obs_fingerprint_dataset`` / ``obs_fingerprint_spec``：数据集 vs spec 拟合时的
  ``env/obs`` 源码指纹（不一致必须显式告警，见下）；
- ``git_hash`` / ``git_dirty_entries`` / ``script_path``：生成时的仓库状态与脚本；
- ``row_alignment``：说明 ``cluster[i]`` 对应数据集第 i 行（与 ``expert_bc.npz`` 各数组行序一致）。

复用规则：sidecar 的 ``rows``+``spec_hash``+``obs_fingerprint_dataset`` 与当前一致 → 直接复用
（不重算）；``--refresh-assignments`` 强制重算。写入用临时文件 + 原子 rename；目录不可写时
只在 summary 记 ``written=false`` 并继续（不阻塞诊断）。

簇口径
======
簇 = **当前帧 obs 的确定性函数**（冻结 spec ``config/clusters/cluster_v1.npz``，k=8，
``mode=two_stage``：簇 0 = 活跃度门控的"平稳"原型，簇 1..7 = 活跃子集 k-means 原型）。
本脚本按 spec 逐 batch 调 ``pipeline.clusters.soft_targets_from_obs`` 后 argmax 得硬簇号；
输入 = 数据集当前帧通道（与训练 ``BCDataset.build_obs_batch`` 的当前帧逐位一致，已离线校验）。

字段语义（先读 ``env/obs/`` 源码确认；只在**当前帧 obs**上作图，自车系 = x 前向 / y 左向）
=========================================================================================
- ``ego`` (1,8) = ``[v, a_long, a_lat, yaw_rate, steer, curvature, prev_ds, prev_dtheta]``
  （``env/obs/ego.py``；v m/s、a m/s²、yaw_rate rad/s、curvature 1/m、prev 两维=上一策略步动作）。
  作图：自车在原点、朝向 +x（heading），另标 +y（左）；把 8 个数值写成面板文字。
- ``od`` (16,9) = ``[dx, dy, vx, vy, cosθ, sinθ, L, W, type_id]``（``env/obs/od.py``）。
  ``dx,dy`` = 目标中心在自车系的位置；``vx,vy`` = **相对速度**（目标−ego，旋入自车系）；
  ``cosθ,sinθ`` = 目标航向相对 ego 航向；``L,W`` = 车长/车宽（m）；``type_id`` = 车型枚举
  （0 Default/1 S/2 M/3 L/4 XL/5 unknown，见 ``od.py::VEHICLE_TYPE_IDS``）。
  有效性：``od_mask=1`` 仅表示"槽位已分配 track id"；``od_presence=1`` 才是"本帧真的观测到"，
  出盒未释放期间 ``mask=1, presence=0``（特征陈旧）。**作图只画 ``presence>0`` 的槽**（其余
  按"无效槽位不上色"处理），矩形 = 中心 (dx,dy) + 朝向 ``atan2(sinθ,cosθ)`` + 尺寸 L×W。
- ``ld`` (16,7) = ``[dx, dy, heading_rel, curvature, speed_limit, left_type_id, right_type_id]``
  （``env/obs/ld.py``）。车道中心线采样点（offset ∈ {5,10,15,20,30} m）：``dx,dy`` 自车系、
  ``heading_rel`` 相对 ego 航向、``curvature`` dθ/ds、``speed_limit`` m/s、左右线型 id。
  作图：每个 ``ld_mask>0`` 槽画"点 + 沿 heading_rel 的短线段"（数据集不含采样间距/长度字段，
  固定半长 2.5 m 仅用于可视化，不表示真实车道长度）。
- ``nav`` (1,11) = ``[c0_x, c0_y, c1_x, c1_y, one-hot(6), route_completion]``（``env/obs/nav.py``）：
  前 4 维 = 2 个 checkpoint（自车系）；one-hot = forward/left/right + 3 保留；末维 = route_completion。
  作图：两个 checkpoint 画星标 + ego→c0→c1 虚线（连线仅表示顺序），文字标注命令与完成度。
- ``signal`` (1,4)：占位恒 ``[0,0,0,1]``（无交通灯，``env/obs/signal.py``），不单独作图。
- ``others`` (1,28) = ``nav(11) + speed_limit(1, 归一化) + signal(4) + road_class one-hot(12)``
  （``env/obs/others.py``；聚类 spec 的 ``others_source=raw`` 用的就是它），不单独作图。
- 每帧角注：difficulty / geometry / lane_lat / spec_id / episode_id / step / 有效槽位数
  （od fresh=presence>0 / stale=mask=1&presence=0 / empty=mask=0；ld valid=ld_mask 计数）。
- 面板 ①scene/②ld/③od 固定视窗（默认 x∈[-20,60]、y∈[-30,30] m，`--view-window` 可调，跨帧可比）；
  视窗外实体不画、面板角注 out-of-view 计数；④ego ±8 m；⑤nav 自适应 checkpoint 范围。

内存/确定性：只解压聚类与作图需要的 22 个 npz 成员（约 0.6 GB），encode 按 ``--batch-size``
分批（默认 8192）→ 峰值 RSS < 1 GB；抽样用 ``np.random.default_rng(seed)`` 按簇序取
``train_weight>0`` 行（先排序再取），同 seed + 同数据 → 同抽样、同图。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.clusters import load as load_clusters, soft_targets_from_obs  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # CPU-only / 无显示环境
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

DEFAULT_BC_DIR = "datasets/BTC20260926-2343_expert5k"
DEFAULT_SPEC = "config/clusters/cluster_v1.npz"
DEFAULT_PER_CLUSTER = 100
DEFAULT_BATCH_SIZE = 8192
DEFAULT_SEED = 0
DEFAULT_DPI = 110
#: 面板 ①–③（scene/ld/od）固定视窗 xmin,xmax,ymin,ymax（m，自车系）：跨帧可比；窗外实体不画并角注计数。
DEFAULT_VIEW_WINDOW = "-20,60,-30,30"

#: 聚类特征需要的 obs 通道（与 ``pipeline.clusters.encode_obs`` 口径一致；others=raw）。
FEATURE_KEYS: Tuple[str, ...] = (
    "ego", "od", "od_mask", "od_presence", "ld", "ld_mask", "nav", "nav_mask", "signal", "others",
)
#: 作图/统计额外需要的数组（其余 npz 成员不读，控制内存）。
EXTRA_KEYS: Tuple[str, ...] = (
    "episode_id", "step", "difficulty", "geometry", "lane_lat", "spec_id", "seed",
    "train_weight", "balance_weight", "frame_usable", "labels",
)

#: 无效槽位（mask=0 / presence=0）不上色：OD 只画 presence>0；LD 只画 ld_mask>0。
_LD_HALF_SEGMENT_M = 2.5  # 仅可视化用的固定半段长（数据集无采样间距字段）

#: 车型 id 名称（``env/obs/od.py::VEHICLE_TYPE_IDS`` + UNKNOWN）。
VEHICLE_TYPE_NAMES: Dict[int, str] = {0: "default", 1: "S", 2: "M", 3: "L", 4: "XL", 5: "unknown"}
NAV_COMMANDS: Tuple[str, ...] = ("forward", "left", "right")


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #

def _jsonable(value: Any) -> Any:
    """numpy 标量/数组 → 原生 Python（JSON 序列化用）。"""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _wrap_pi(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _box_corners(dx: float, dy: float, yaw: float, length: float, width: float) -> np.ndarray:
    """OD 矩形四角（自车系）：中心 (dx,dy)、朝向 yaw、尺寸 L×W。"""
    half_l, half_w = max(float(length), 0.1) / 2.0, max(float(width), 0.1) / 2.0
    local = np.array([[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]])
    cos_y, sin_y = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return local @ rotation.T + np.array([float(dx), float(dy)])


def _set_frame_limits(
    ax: Any, xs: Sequence[float], ys: Sequence[float], *, window: Optional[Tuple[float, float, float, float]] = None, min_span: float = 10.0
) -> None:
    """坐标轴：``window`` 给定则用固定视窗（面板可比），否则按点集自适应（含原点）。"""
    if window is not None:
        x0, x1, y0, y1 = (float(value) for value in window)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.3, alpha=0.3)
        return
    values_x = [float(v) for v in xs if np.isfinite(v)]
    values_y = [float(v) for v in ys if np.isfinite(v)]
    values_x.append(0.0)
    values_y.append(0.0)
    x0, x1 = min(values_x), max(values_x)
    y0, y1 = min(values_y), max(values_y)
    span = max(x1 - x0, y1 - y0, float(min_span)) * 1.15
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ax.set_xlim(cx - span / 2.0, cx + span / 2.0)
    ax.set_ylim(cy - span / 2.0, cy + span / 2.0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.3)


def _in_window(x: float, y: float, window: Optional[Tuple[float, float, float, float]]) -> bool:
    if window is None:
        return True
    x0, x1, y0, y1 = window
    return bool(float(x0) <= float(x) <= float(x1) and float(y0) <= float(y) <= float(y1))


def _draw_ld(
    ax: Any,
    ld: np.ndarray,
    ld_mask: np.ndarray,
    *,
    window: Optional[Tuple[float, float, float, float]] = None,
    annotate: bool = True,
) -> Tuple[List[float], List[float], int]:
    """LD：``ld_mask>0`` 的槽位画点 + 沿 heading_rel 的短线段（半长 2.5 m，仅可视化）。"""
    xs: List[float] = []
    ys: List[float] = []
    out_of_view = 0
    for slot in np.flatnonzero(np.asarray(ld_mask) > 0.5):
        dx, dy, heading_rel = float(ld[slot, 0]), float(ld[slot, 1]), float(ld[slot, 2])
        if not _in_window(dx, dy, window):
            out_of_view += 1
            continue
        half = np.array([math.cos(heading_rel), math.sin(heading_rel)]) * _LD_HALF_SEGMENT_M
        point = np.array([dx, dy])
        ax.plot(
            [point[0] - half[0], point[0] + half[0]],
            [point[1] - half[1], point[1] + half[1]],
            "-", color="tab:green", linewidth=1.4, alpha=0.85,
        )
        ax.plot(dx, dy, "o", color="tab:green", markersize=2.8)
        if annotate:
            ax.text(dx, dy, f" l{slot}", color="tab:green", fontsize=6, alpha=0.8)
        xs.extend([point[0] - half[0], point[0] + half[0]])
        ys.extend([point[1] - half[1], point[1] + half[1]])
    return xs, ys, out_of_view


def _draw_od(
    ax: Any,
    od: np.ndarray,
    od_mask: np.ndarray,
    od_presence: np.ndarray,
    *,
    window: Optional[Tuple[float, float, float, float]] = None,
    annotate: bool = True,
) -> Tuple[List[float], List[float], int]:
    """OD：只画 ``presence>0``（本帧新鲜观测）；矩形 = (dx,dy)+朝向+ L×W。"""
    xs: List[float] = []
    ys: List[float] = []
    out_of_view = 0
    for slot in np.flatnonzero(np.asarray(od_presence) > 0.5):
        dx, dy, vx, vy, cos_t, sin_t, length, width = (float(v) for v in od[slot, :8])
        type_id = int(round(float(od[slot, 8])))
        if not (_in_window(dx, dy, window) or any(_in_window(px, py, window) for px, py in _box_corners(dx, dy, math.atan2(sin_t, cos_t), length, width))):
            out_of_view += 1
            continue
        yaw = math.atan2(sin_t, cos_t)
        corners = _box_corners(dx, dy, yaw, length, width)
        ax.add_patch(
            Polygon(corners, closed=True, facecolor="tab:orange", edgecolor="tab:orange", alpha=0.28, linewidth=1.0)
        )
        ax.plot(dx, dy, ".", color="tab:orange", markersize=3.0)
        if annotate:
            ax.text(
                dx, dy, f"s{slot}/{VEHICLE_TYPE_NAMES.get(type_id, str(type_id))}",
                fontsize=5.5, color="tab:orange", alpha=0.95,
            )
        xs.extend(corners[:, 0].tolist())
        ys.extend(corners[:, 1].tolist())
    stale = int(np.sum((np.asarray(od_mask) > 0.5) & (np.asarray(od_presence) <= 0.5)))
    if annotate and stale:
        ax.text(
            0.02, 0.98, f"od stale(mask=1,presence=0,not drawn)={stale}",
            transform=ax.transAxes, fontsize=6, va="top", color="gray",
        )
    return xs, ys, out_of_view


def _draw_ego(ax: Any, ego: np.ndarray, *, length: float = 4.0, annotate: bool = True) -> None:
    """自车：原点 + 朝向（+x）+ 左向（+y）坐标基；``annotate`` 时写 8 维状态（两行）。"""
    ax.annotate(
        "", xy=(length, 0.0), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="tab:blue", linewidth=2.0),
    )
    ax.annotate(
        "", xy=(0.0, length * 0.6), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="tab:blue", linewidth=1.0, alpha=0.6),
    )
    ax.plot([0.0], [0.0], "o", color="tab:blue", markersize=4.0)
    if annotate:
        names = ("v", "a_long", "a_lat", "yaw_rate", "steer", "curv", "prev_ds", "prev_dtheta")
        values = [float(value) for value in ego]
        line_one = ", ".join(f"{name}={value:.2f}" for name, value in zip(names[:4], values[:4]))
        line_two = ", ".join(f"{name}={value:.2f}" for name, value in zip(names[4:], values[4:]))
        ax.text(0.02, 0.02, line_one + "\n" + line_two, transform=ax.transAxes, fontsize=6.2,
                va="bottom", color="tab:blue")
        ax.text(0.02, 0.92, "ego at origin, +x = heading, +y = left", transform=ax.transAxes, fontsize=6, color="gray")


def _draw_nav(
    ax: Any,
    nav: np.ndarray,
    nav_mask: np.ndarray,
    *,
    window: Optional[Tuple[float, float, float, float]] = None,
    annotate: bool = True,
) -> Tuple[List[float], List[float], int]:
    """nav：2 个 checkpoint（自车系）+ 命令 one-hot + route_completion；ego→c0→c1 虚线仅表顺序。"""
    xs: List[float] = []
    ys: List[float] = []
    out_of_view = 0
    valid = bool(float(nav_mask[0]) > 0.5)
    if valid:
        first = (float(nav[0]), float(nav[1]))
        second = (float(nav[2]), float(nav[3]))
        if window is None:
            draw_first, draw_second = True, True
        else:
            draw_first = _in_window(*first, window)
            draw_second = _in_window(*second, window)
            out_of_view = int(not draw_first) + int(not draw_second)
        if draw_first or draw_second:
            path_x = [0.0] + ([first[0]] if draw_first else []) + ([second[0]] if draw_second else [])
            path_y = [0.0] + ([first[1]] if draw_first else []) + ([second[1]] if draw_second else [])
            ax.plot(path_x, path_y, "--", color="tab:purple", linewidth=0.9, alpha=0.8)
        for point, label, visible in ((first, " c0", draw_first), (second, " c1", draw_second)):
            if not visible:
                continue
            ax.plot(point[0], point[1], "*", color="tab:purple", markersize=9.0)
            if annotate:
                ax.text(point[0], point[1], label, color="tab:purple", fontsize=6.5)
            xs.append(point[0])
            ys.append(point[1])
    if annotate:
        one_hot = np.asarray(nav[4:10], dtype=float)
        command_index = int(np.argmax(one_hot))
        command = NAV_COMMANDS[command_index] if command_index < len(NAV_COMMANDS) and one_hot[command_index] > 0.5 else "none"
        ax.text(
            0.02, 0.92,
            f"cmd={command} route_completion={float(nav[10]):.3f} mask={float(nav_mask[0]):.0f}",
            transform=ax.transAxes, fontsize=6.5, color="tab:purple",
        )
    return xs, ys, out_of_view


# --------------------------------------------------------------------------- #
# 聚类分配
# --------------------------------------------------------------------------- #

def assign_clusters(
    spec: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    batch_size: int,
    logger: Any = print,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """全体行分批 encode + 软目标 → ``(hard, top1, margin)``；``hard = argmax``（two_stage 含门控）。"""
    total = int(len(arrays["episode_id"]))
    hard = np.zeros(total, dtype=np.int8)
    top1 = np.zeros(total, dtype=np.float32)
    margin = np.zeros(total, dtype=np.float32)
    started = time.perf_counter()
    for start in range(0, total, int(batch_size)):
        stop = min(start + int(batch_size), total)
        obs_batch = {key: np.asarray(arrays[key])[start:stop] for key in FEATURE_KEYS}
        soft = np.asarray(soft_targets_from_obs(obs_batch, spec=spec), dtype=np.float32)
        partition = np.partition(soft, -2, axis=1)
        hard[start:stop] = soft.argmax(axis=1).astype(np.int8)
        top1[start:stop] = partition[:, -1]
        margin[start:stop] = partition[:, -1] - partition[:, -2]
    logger(
        f"[survey] cluster assign: {total} rows in {(total + batch_size - 1) // batch_size} batches "
        f"({time.perf_counter() - started:.1f}s, batch={batch_size})"
    )
    return hard, top1, margin


# --------------------------------------------------------------------------- #
# 全量分配 sidecar（供 train/eval 直接读；避免每次重算）
# --------------------------------------------------------------------------- #

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info() -> Dict[str, str]:
    """仓库 HEAD / dirty 计数（只读 git 子命令；不可用 → unknown）。"""

    def _run(args: Sequence[str]) -> Optional[subprocess.CompletedProcess]:
        try:
            return subprocess.run(
                list(args), cwd=str(_PROJECT_ROOT), capture_output=True, text=True, timeout=10, check=False
            )
        except Exception:  # noqa: BLE001
            return None

    head = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {
        "git_hash": head.stdout.strip() if head is not None and head.returncode == 0 else "unknown",
        "git_dirty_entries": str(len(status.stdout.splitlines())) if status is not None and status.returncode == 0 else "unknown",
    }


def default_assignments_path(bc_dir: Path, spec_path: Path) -> Path:
    """sidecar 路径：``<bc_dir>/<spec_stem>_assignments.npz``。"""
    return Path(bc_dir) / f"{Path(spec_path).stem}_assignments.npz"


def write_assignments_sidecar(
    path: Path,
    *,
    hard: np.ndarray,
    top1: np.ndarray,
    margin: np.ndarray,
    spec_hash: str,
    spec_path: str,
    dataset_fingerprint: str,
    spec_fingerprint: str,
    logger: Any = print,
) -> Dict[str, Any]:
    """原子写 sidecar（临时文件 + ``os.replace``）；失败只报告，不抛。"""
    path = Path(path)
    info: Dict[str, Any] = {"written": False, "path": str(path), "error": ""}
    git = _git_info()
    created_at = datetime.now().isoformat(timespec="seconds")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".partial.npz")
        np.savez_compressed(
            tmp,
            cluster=np.asarray(hard, dtype=np.int16),
            top1_margin=np.asarray(margin, dtype=np.float32),
            top1_prob=np.asarray(top1, dtype=np.float32),
            spec_hash=np.array(str(spec_hash)),
            spec_path=np.array(str(spec_path)),
            obs_fingerprint_dataset=np.array(str(dataset_fingerprint)),
            obs_fingerprint_spec=np.array(str(spec_fingerprint)),
            rows=np.array(int(np.asarray(hard).size), dtype=np.int64),
            created_at=np.array(str(created_at)),
            git_hash=np.array(str(git["git_hash"])),
            git_dirty_entries=np.array(str(git["git_dirty_entries"])),
            script_path=np.array("tools/diagnostics/cluster_survey.py"),
            top1_margin_definition=np.array("top1_prob - second_best_prob（软目标 top-2 间隔）"),
            row_alignment=np.array("cluster[i]/top1_prob[i]/top1_margin[i] 对应数据集（expert_bc.npz）第 i 行"),
        )
        os.replace(tmp, path)
        info.update({"written": True, "created_at": created_at, **git})
        logger(f"[survey] assignments sidecar 写入：{path}（spec_hash={spec_hash[:12]}，git={git['git_hash'][:8]}）")
    except Exception as exc:  # noqa: BLE001 - 数据集目录只读等场景不应阻塞诊断
        info["error"] = f"{type(exc).__name__}: {exc}"
        logger(f"[survey] ⚠️ assignments sidecar 写入失败（{info['error']}）→ 仅本次运行内可用")
    return info


def load_or_compute_assignments(
    *,
    arrays: Mapping[str, np.ndarray],
    spec: Any,
    spec_path: Path,
    dataset_fingerprint: str,
    spec_fingerprint: str,
    sidecar_path: Path,
    batch_size: int,
    refresh: bool,
    logger: Any = print,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """复用有效 sidecar（spec_hash + 数据集指纹 + rows 匹配）否则重算并写 sidecar。"""
    total = int(len(arrays["episode_id"]))
    spec_hash = _sha256_file(spec_path)
    info: Dict[str, Any] = {"path": str(sidecar_path), "spec_hash": spec_hash, "reused": False}
    if sidecar_path.is_file() and not refresh:
        try:
            with np.load(sidecar_path, allow_pickle=False) as payload:
                fields = {key: payload[key] for key in payload.files}
            required = ("cluster", "top1_margin", "top1_prob", "rows", "spec_hash", "obs_fingerprint_dataset")
            missing = [key for key in required if key not in fields]
            ok = not missing and (
                int(fields["rows"]) == total
                and str(fields["spec_hash"]) == spec_hash
                and str(fields["obs_fingerprint_dataset"]) == str(dataset_fingerprint)
                and fields["cluster"].shape == (total,)
                and fields["top1_margin"].shape == (total,)
                and fields["top1_prob"].shape == (total,)
            )
            if ok:
                info.update(
                    {
                        "reused": True,
                        "created_at": str(fields.get("created_at", "")),
                        "rows": int(fields["rows"]),
                        "obs_fingerprint_dataset": str(fields["obs_fingerprint_dataset"]),
                        "obs_fingerprint_spec": str(fields.get("obs_fingerprint_spec", "")),
                        "git_hash": str(fields.get("git_hash", "")),
                    }
                )
                logger(
                    f"[survey] assignments sidecar 复用：{sidecar_path}"
                    f"（spec_hash={spec_hash[:12]}，created_at={info['created_at']}）"
                )
                return (
                    fields["cluster"].astype(np.int64),
                    fields["top1_prob"].astype(np.float32),
                    fields["top1_margin"].astype(np.float32),
                    info,
                )
            reason = f"缺键 {missing}" if missing else "rows/spec_hash/obs_fingerprint 不匹配"
            logger(f"[survey] assignments sidecar 失效（{reason}）→ 重算：{sidecar_path}")
        except Exception as exc:  # noqa: BLE001
            logger(f"[survey] assignments sidecar 读取失败（{type(exc).__name__}: {exc}）→ 重算")
    hard, top1, margin = assign_clusters(spec, arrays, batch_size=batch_size, logger=logger)
    write_info = write_assignments_sidecar(
        sidecar_path,
        hard=hard,
        top1=top1,
        margin=margin,
        spec_hash=spec_hash,
        spec_path=str(spec_path.resolve()),
        dataset_fingerprint=dataset_fingerprint,
        spec_fingerprint=spec_fingerprint,
        logger=logger,
    )
    info.update({"rows": total, **{key: value for key, value in write_info.items() if key != "path"}})
    return hard, top1, margin, info


def report_comparison(report_path: Path, full_shares: np.ndarray, train_shares: np.ndarray) -> Dict[str, Any]:
    """与冻结体检报告（``cluster_v1.report.json``，60k 子采样口径）的 shares 对照。"""
    out: Dict[str, Any] = {"report_path": str(report_path), "available": False}
    if not Path(report_path).is_file():
        out["note"] = "体检报告缺失 → 无法对照"
        return out
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        out["note"] = f"体检报告读取失败：{type(exc).__name__}: {exc}"
        return out
    size = report.get("size") or {}
    data = report.get("data") or {}
    shares = [float(value) for value in (size.get("shares") or [])]
    if not shares or len(shares) != len(full_shares):
        out["note"] = f"体检报告 share 长度 {len(shares)} != 簇数 {len(full_shares)} → 无法对照"
        return out
    out.update(
        {
            "available": True,
            "fit_rows": int(data.get("rows") or 0),
            "fit_rows_total": int(data.get("rows_total") or 0),
            "fit_source": str(data.get("source") or ""),
            "fit_obs_fingerprint": str(data.get("obs_fingerprint") or ""),
            "fit_git_hash": str(data.get("git_hash") or ""),
            "report_shares": shares,
            "survey_full_shares": [float(value) for value in full_shares],
            "survey_train_shares": [float(value) for value in train_shares],
            "delta_full_minus_report": [float(a) - float(b) for a, b in zip(full_shares, shares)],
            "note": "报告 shares = 拟合子采样（fit_rows）口径；survey = 当前数据集全量口径；"
                    "spec/data 指纹不一致时差异同时含观测版本漂移与子采样因素。",
        }
    )
    return out



# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #

def _value_distribution(values: np.ndarray, indices: np.ndarray) -> Dict[str, float]:
    subset = np.asarray(values)[indices]
    if subset.size == 0:
        return {}
    unique, counts = np.unique(subset, return_counts=True)
    total = float(counts.sum())
    return {str(key): float(count) / total for key, count in zip(unique, counts)}


def _top_labels(label_means: np.ndarray, label_names: Sequence[str], k: int = 3) -> List[Tuple[str, float]]:
    order = np.argsort(-np.asarray(label_means))[:k]
    return [
        (str(label_names[i]) if i < len(label_names) else f"label_{i}", float(label_means[i]))
        for i in order
    ]


def cluster_stats(
    *,
    arrays: Mapping[str, np.ndarray],
    hard: np.ndarray,
    top1: np.ndarray,
    margin: np.ndarray,
    num_clusters: int,
    train_mask: np.ndarray,
    label_names: Sequence[str],
    boundary_margin: float,
) -> List[Dict[str, Any]]:
    """每簇画像（全量口径的分布 + 训练子集计数）。"""
    od_presence = np.asarray(arrays["od_presence"])
    od_mask = np.asarray(arrays["od_mask"])
    ld_mask = np.asarray(arrays["ld_mask"])
    fresh = (od_presence > 0.5).sum(axis=1).astype(np.float32)
    stale = ((od_mask > 0.5) & (od_presence <= 0.5)).sum(axis=1).astype(np.float32)
    ld_valid = (ld_mask > 0.5).sum(axis=1).astype(np.float32)
    labels = np.asarray(arrays["labels"], dtype=np.float64)
    lane_lat = np.asarray(arrays["lane_lat"], dtype=np.float64)
    geometry = arrays["geometry"]
    difficulty = arrays["difficulty"]

    stats: List[Dict[str, Any]] = []
    for cluster in range(int(num_clusters)):
        rows = np.flatnonzero(hard == cluster)
        train_rows = np.flatnonzero((hard == cluster) & train_mask)
        entry: Dict[str, Any] = {
            "id": int(cluster),
            "count_full": int(rows.size),
            "count_train": int(train_rows.size),
            "difficulty": _value_distribution(difficulty, rows),
            "lane_lat_mean": float(np.nanmean(lane_lat[rows])) if rows.size else float("nan"),
            "od_fresh_mean": float(fresh[rows].mean()) if rows.size else float("nan"),
            "od_stale_mean": float(stale[rows].mean()) if rows.size else float("nan"),
            "ld_valid_mean": float(ld_valid[rows].mean()) if rows.size else float("nan"),
            "top1_prob_mean": float(top1[rows].mean()) if rows.size else float("nan"),
            "boundary_share": float((margin[rows] < float(boundary_margin)).mean()) if rows.size else float("nan"),
        }
        # geometry top-3（全量）
        if rows.size:
            unique, counts = np.unique(geometry[rows], return_counts=True)
            order = np.argsort(-counts)[:3]
            entry["geometry_top3"] = [[str(unique[i]), float(counts[i]) / float(rows.size)] for i in order]
        else:
            entry["geometry_top3"] = []
        # labels top-3（全量均值；labels 为 0/1，均值=占比）
        entry["label_top3"] = (
            _top_labels(labels[rows].mean(axis=0) if rows.size else np.zeros(labels.shape[1]), label_names)
            if rows.size else []
        )
        if rows.size:
            entry["label_rates"] = {
                str(name): float(labels[rows, index].mean()) for index, name in enumerate(label_names)
            }
        else:
            entry["label_rates"] = {}
        stats.append(entry)
    return stats


# --------------------------------------------------------------------------- #
# 出图
# --------------------------------------------------------------------------- #

def plot_frame(
    *,
    row: int,
    arrays: Mapping[str, np.ndarray],
    cluster: int,
    out_path: Path,
    dpi: int,
    top1_prob: float,
    margin: float,
    view: Tuple[float, float, float, float],
) -> None:
    """单帧 5 联面板：场景叠加（ld+od+ego+nav）/ ld / od / ego / nav。

    ①③（scene/ld/od）用固定视窗 ``view``（跨帧可比），视窗外实体不画、右上角注 out-of-view 计数；
    ④ ego 用 ±8 m 小窗；⑤ nav 自适应到 checkpoint 范围（route 点可能很远）。
    """
    ego = np.asarray(arrays["ego"][row], dtype=np.float32).reshape(-1)
    od = np.asarray(arrays["od"][row], dtype=np.float32)
    od_mask = np.asarray(arrays["od_mask"][row], dtype=np.float32)
    od_presence = np.asarray(arrays["od_presence"][row], dtype=np.float32)
    ld = np.asarray(arrays["ld"][row], dtype=np.float32)
    ld_mask = np.asarray(arrays["ld_mask"][row], dtype=np.float32)
    nav = np.asarray(arrays["nav"][row], dtype=np.float32).reshape(-1)
    nav_mask = np.asarray(arrays["nav_mask"][row], dtype=np.float32).reshape(-1)
    episode_id = int(arrays["episode_id"][row])
    step = int(arrays["step"][row])
    difficulty = str(arrays["difficulty"][row])
    geometry = str(arrays["geometry"][row])
    lane_lat = float(arrays["lane_lat"][row])
    spec_id = int(arrays["spec_id"][row])
    fresh_n = int(np.sum(od_presence > 0.5))
    stale_n = int(np.sum((od_mask > 0.5) & (od_presence <= 0.5)))
    empty_n = int(np.sum(od_mask <= 0.5))
    ld_n = int(np.sum(ld_mask > 0.5))

    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.4))
    fig.suptitle(
        f"cluster {cluster} | {difficulty} | {geometry} | lane_lat={lane_lat:+.2f} | spec_id={spec_id} | "
        f"episode={episode_id} step={step} | od(fresh/stale/empty)={fresh_n}/{stale_n}/{empty_n} | ld={ld_n} | row={row} "
        f"| top1={top1_prob:.2f} margin={margin:.2f}",
        fontsize=8.5,
    )
    view_text = f"view x[{view[0]:.0f},{view[1]:.0f}] y[{view[2]:.0f},{view[3]:.0f}] m; out-of-view not drawn"

    # panel 1: scene（固定视窗；只画几何，文字集中到各专用面板，避免小面板文字堆叠）
    ld_x, ld_y, ld_out = _draw_ld(axes[0], ld, ld_mask, window=view, annotate=False)
    od_x, od_y, od_out = _draw_od(axes[0], od, od_mask, od_presence, window=view, annotate=False)
    nav_x, nav_y, nav_out = _draw_nav(axes[0], nav, nav_mask, window=view, annotate=False)
    _draw_ego(axes[0], ego, annotate=False)
    axes[0].set_title("scene: ld+od+ego+nav", fontsize=8.5, loc="left")
    _set_frame_limits(axes[0], ld_x + od_x + nav_x, ld_y + od_y + nav_y, window=view)
    axes[0].text(0.02, 0.02, "ld=green  od=orange  ego=blue  nav=purple", transform=axes[0].transAxes,
                 fontsize=6, va="bottom", color="gray")
    axes[0].text(0.98, 0.98, f"out-of-view: ld={ld_out} od={od_out} nav={nav_out}", transform=axes[0].transAxes,
                 fontsize=6, ha="right", va="top", color="gray")

    # panel 2: ld（固定视窗）
    x, y, ld_out = _draw_ld(axes[1], ld, ld_mask, window=view)
    axes[1].set_title("ld (mask>0; 2.5 m segments)", fontsize=8.5, loc="left")
    _set_frame_limits(axes[1], x, y, window=view)
    axes[1].text(0.98, 0.02, f"out-of-view: {ld_out}", transform=axes[1].transAxes, fontsize=6, ha="right", va="bottom", color="gray")

    # panel 3: od（固定视窗）
    x, y, od_out = _draw_od(axes[2], od, od_mask, od_presence, window=view)
    axes[2].set_title("od (presence>0; box=dx,dy,yaw,L,W)", fontsize=8.5, loc="left")
    _set_frame_limits(axes[2], x, y, window=view)
    axes[2].text(0.98, 0.02, f"out-of-view: {od_out}", transform=axes[2].transAxes, fontsize=6, ha="right", va="bottom", color="gray")

    # panel 4: ego（±8 m）
    _draw_ego(axes[3], ego)
    axes[3].set_title("ego (state text at origin)", fontsize=8.5, loc="left")
    _set_frame_limits(axes[3], [0.0], [0.0], window=(-8.0, 8.0, -8.0, 8.0))

    # panel 5: nav（自适应 checkpoint 范围）
    x, y, _nav_out = _draw_nav(axes[4], nav, nav_mask)
    axes[4].set_title("nav (2 ckpts + command + completion)", fontsize=8.5, loc="left")
    _set_frame_limits(axes[4], x + [0.0], y + [0.0])

    for axis in axes:
        axis.set_xlabel("x forward (m)", fontsize=7)
        axis.set_ylabel("y left (m)", fontsize=7)
        axis.tick_params(labelsize=6.5)
    fig.text(
        0.005, 0.005,
        f"render: dx/dy in ego frame (x fwd / y left); od boxes only when presence>0; ld segments centered at sample points; panels 1-3 {view_text}",
        fontsize=6, color="gray",
    )
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def contact_sheet(png_paths: Sequence[Path], out_path: Path, *, max_images: int = 25) -> None:
    """25 图拼版（5 列；每张 = 一帧的 5 联面板 strip，原生像素拼接）。"""
    selected = [Path(path) for path in list(png_paths)[: int(max_images)]]
    if not selected:
        return
    try:
        from PIL import Image  # matplotlib 依赖 Pillow；缺失则回退 matplotlib 拼版
    except ImportError:
        _contact_sheet_matplotlib(selected, out_path)
        return
    with Image.open(selected[0]) as first:
        tile_w, tile_h = first.size
    columns = 5
    rows = int(math.ceil(len(selected) / columns))
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "white")
    for index, path in enumerate(selected):
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB"), ((index % columns) * tile_w, (index // columns) * tile_h))
    sheet.save(out_path)
    sheet.close()


def _contact_sheet_matplotlib(png_paths: Sequence[Path], out_path: Path) -> None:
    """PIL 缺失时的回退拼版（matplotlib imshow；分辨率受限）。"""
    images = [plt.imread(str(path)) for path in png_paths]
    columns = 5
    rows = int(math.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 6.0, rows * 1.25))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for index, axis in enumerate(axes_array.ravel()):
        axis.axis("off")
        if index < len(images):
            axis.imshow(images[index], aspect="auto")
    fig.suptitle(f"contact sheet: {len(images)} sampled frames (each strip = 5-panel view)", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(out_path, dpi=60)
    plt.close(fig)
    del images


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def _load_arrays(npz_path: Path, keys: Sequence[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as payload:
        missing = [key for key in keys if key not in payload.files]
        if missing:
            raise SystemExit(f"[survey] 数据集缺少字段：{missing}（{npz_path}）")
        arrays = {key: payload[key] for key in keys}
    return arrays, {}


def _fingerprint_report(spec: Any, meta: Mapping[str, Any]) -> Dict[str, Any]:
    spec_fp = str((spec.data_fingerprint or {}).get("obs_fingerprint") or "")
    spec_fp_current = str((spec.data_fingerprint or {}).get("obs_fingerprint_current") or "")
    dataset_fp = str(meta.get("obs_fingerprint") or "")
    match = bool(spec_fp and dataset_fp and spec_fp == dataset_fp)
    return {
        "spec_obs_fingerprint": spec_fp,
        "spec_obs_fingerprint_current_at_fit": spec_fp_current,
        "dataset_obs_fingerprint": dataset_fp,
        "match": match,
        "spec_fitted_at": str((spec.data_fingerprint or {}).get("fitted_at") or ""),
        "spec_fit_git_hash": str((spec.data_fingerprint or {}).get("git_hash") or ""),
        "spec_fit_source": str((spec.data_fingerprint or {}).get("source") or ""),
        "spec_fit_rows": int((spec.data_fingerprint or {}).get("rows") or 0),
        "warning": (
            "" if match else
            "spec 拟合时的 obs_fingerprint（env/obs 源码哈希）与当前数据集不一致："
            "聚类特征口径可能已漂移（spec 拟合于旧观测版本），占比/画像需按此前提解读，"
            "不可直接当作当前观测版本的权威聚类。"
        ),
    }


def survey(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    bc_dir = Path(args.bc_dir)
    if not (bc_dir / "expert_bc.npz").is_file():
        raise SystemExit(f"[survey] 数据集不存在：{bc_dir / 'expert_bc.npz'}")
    meta = {}
    meta_path = bc_dir / "expert_bc.meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    label_names = list(meta.get("label_names") or [])

    out_dir = Path(args.out) if args.out else (
        Path("runs") / f"BTC{datetime.now().strftime('%Y%m%d-%H%M')}_cluster_survey"
    )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(f"[survey] 输出目录非空：{out_dir}（加 --force 覆盖/续写）")
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = load_clusters(args.spec)
    arrays, _ = _load_arrays(bc_dir / "expert_bc.npz", tuple(FEATURE_KEYS) + tuple(EXTRA_KEYS))
    fingerprint = _fingerprint_report(spec, meta)
    total_rows = int(len(arrays["episode_id"]))
    train_mask = np.asarray(arrays["train_weight"], dtype=np.float64) > 0.0
    print(
        f"[survey] data={bc_dir} rows={total_rows} train_weight>0={int(train_mask.sum())} "
        f"| spec={args.spec} mode={spec.mode} k={spec.k} others={spec.others_source}"
    )
    print(
        f"[survey] obs_fingerprint: spec={fingerprint['spec_obs_fingerprint']} "
        f"dataset={fingerprint['dataset_obs_fingerprint']} match={fingerprint['match']}"
    )
    if not fingerprint["match"]:
        print("[survey] ⚠️ 指纹不一致：聚类 spec 拟合于旧观测版本，占比/画像按此前提解读（见 summary.md）")

    spec_path = Path(args.spec)
    sidecar_path = (
        Path(args.assignments_path)
        if args.assignments_path
        else default_assignments_path(bc_dir, spec_path)
    )
    hard, top1, margin, sidecar_info = load_or_compute_assignments(
        arrays=arrays,
        spec=spec,
        spec_path=spec_path,
        dataset_fingerprint=fingerprint["dataset_obs_fingerprint"],
        spec_fingerprint=fingerprint["spec_obs_fingerprint"],
        sidecar_path=sidecar_path,
        batch_size=int(args.batch_size),
        refresh=bool(args.refresh_assignments),
    )
    np.savez_compressed(
        out_dir / "assignment.npz",
        hard=hard, top1=top1, margin=margin,
        row=np.arange(total_rows, dtype=np.int64),
    )

    num_clusters = int(spec.k)
    stats = cluster_stats(
        arrays=arrays,
        hard=hard,
        top1=top1,
        margin=margin,
        num_clusters=num_clusters,
        train_mask=train_mask,
        label_names=label_names,
        boundary_margin=float(spec.boundary_margin),
    )
    counts_full = np.array([entry["count_full"] for entry in stats], dtype=np.int64)
    counts_train = np.array([entry["count_train"] for entry in stats], dtype=np.int64)
    shares_full = counts_full / max(int(counts_full.sum()), 1)
    shares_train = counts_train / max(int(counts_train.sum()), 1)
    print("[survey] top-1 share (full): " + ", ".join(f"c{k}={v:.3f}" for k, v in enumerate(shares_full)))
    print("[survey] top-1 share (train): " + ", ".join(f"c{k}={v:.3f}" for k, v in enumerate(shares_train)))
    # 与冻结体检报告（60k 子采样）对照
    comparison = report_comparison(
        Path(args.report) if args.report else Path(args.spec).with_suffix(".report.json"),
        shares_full,
        shares_train,
    )
    if comparison["available"]:
        delta_text = ", ".join(
            f"c{k}={value:+.4f}" for k, value in enumerate(comparison["delta_full_minus_report"])
        )
        print(f"[survey] share vs report（60k，{comparison['fit_rows']} rows）：{delta_text}")

    # ------------------------------------------------------------------ 抽样出图
    rng = np.random.default_rng(int(args.seed))
    per_cluster = int(args.per_cluster)
    view_window = tuple(float(value) for value in str(args.view_window).split(","))
    if len(view_window) != 4:
        raise SystemExit(f"[survey] --view-window 需要 4 个数（xmin,xmax,ymin,ymax），收到 {args.view_window!r}")
    sample_records: List[List[Dict[str, Any]]] = []
    for cluster in range(num_clusters):
        pool = np.flatnonzero((hard == cluster) & train_mask)
        pool_kind = "train_weight>0"
        if pool.size == 0:  # 兜底：训练子集为空时退回全量该簇（summary 记录）
            pool = np.flatnonzero(hard == cluster)
            pool_kind = "all(fallback)"
        if pool.size > per_cluster:
            picked = np.sort(rng.choice(pool, size=per_cluster, replace=False))
        else:
            picked = pool
        records: List[Dict[str, Any]] = []
        for row in picked.tolist():
            record = {
                "row": int(row),
                "episode_id": int(arrays["episode_id"][row]),
                "step": int(arrays["step"][row]),
                "spec_id": int(arrays["spec_id"][row]),
                "difficulty": str(arrays["difficulty"][row]),
                "geometry": str(arrays["geometry"][row]),
                "lane_lat": float(arrays["lane_lat"][row]),
                "top1_prob": float(top1[row]),
                "margin": float(margin[row]),
                "pool": pool_kind,
            }
            record["png"] = (
                f"cluster_{cluster:02d}/c{cluster:02d}_ep{record['episode_id']:04d}"
                f"_step{record['step']:04d}_row{record['row']:06d}.png"
            )
            records.append(record)
            plot_frame(
                row=int(row), arrays=arrays, cluster=cluster,
                out_path=out_dir / record["png"], dpi=int(args.dpi),
                top1_prob=record["top1_prob"], margin=record["margin"], view=view_window,
            )
        sample_records.append(records)
        cluster_dir = out_dir / f"cluster_{cluster:02d}"
        if not args.no_contact_sheet:
            contact_sheet(
                [out_dir / record["png"] for record in records],
                cluster_dir / "contact_sheet.png",
            )
        print(f"[survey] cluster {cluster}: {len(records)} figures → {cluster_dir}")

    # ------------------------------------------------------------------ summary
    runtime = time.perf_counter() - started
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    field_semantics = {
        "ego": "[v, a_long, a_lat, yaw_rate, steer, curvature, prev_ds, prev_dtheta]（env/obs/ego.py；自车系）",
        "od": "[dx, dy, vx, vy, cosθ, sinθ, L, W, type_id]（env/obs/od.py；dx/dy 自车系位置，vx/vy 相对速度，cos/sin 相对航向，L/W 车长宽 m，type_id 见 VEHICLE_TYPE_IDS）；od_mask=槽位分配，od_presence=本帧观测（画图只用 presence>0）",
        "ld": "[dx, dy, heading_rel, curvature, speed_limit, left_type_id, right_type_id]（env/obs/ld.py；车道中心线采样点 offset∈{5,10,15,20,30} m；画点+沿 heading_rel 的 2.5 m 短线段）",
        "nav": "[c0_x, c0_y, c1_x, c1_y, one-hot(6), route_completion]（env/obs/nav.py；checkpoint 自车系，one-hot=forward/left/right+3 保留；画星标+ego→c0→c1 虚线）",
        "signal": "[0,0,0,1] 占位（env/obs/signal.py；无交通灯，不单独作图）",
        "others": "nav(11)+speed_limit(1 归一化)+signal(4)+road_class one-hot(12)（env/obs/others.py；聚类 spec others_source=raw，不单独作图）",
        "axes": "自车系 x 前向 / y 左向（env/obs/base.py）；面板 ①scene/②ld/③od 用固定视窗（跨帧可比），"
                "视窗外实体不画并在面板角注 out-of-view 计数；④ego ±8 m；⑤nav 自适应 checkpoint 范围；无效槽位不上色",
    }
    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "script": "tools/diagnostics/cluster_survey.py",
        "bc_dir": str(bc_dir),
        "spec": str(args.spec),
        "spec_mode": str(spec.mode),
        "spec_k": int(spec.k),
        "spec_others_source": str(spec.others_source),
        "boundary_margin": float(spec.boundary_margin),
        "rows_total": total_rows,
        "rows_train_weight_gt0": int(train_mask.sum()),
        "obs_fingerprint": fingerprint,
        "counts_full": counts_full.tolist(),
        "shares_full": shares_full.tolist(),
        "counts_train": counts_train.tolist(),
        "shares_train": shares_train.tolist(),
        "sampling": {
            "seed": int(args.seed),
            "per_cluster": per_cluster,
            "pool": "train_weight>0（为空退回 all，见样本 pool 字段）",
            "deterministic": "同一 seed + 同一数据/顺序 → 同抽样",
        },
        "assignment_artifact": "assignment.npz（运行副本：hard/top1/margin/row）",
        "assignments_sidecar": sidecar_info,
        "report_comparison": comparison,
        "view_window_m": list(view_window),
        "field_semantics": field_semantics,
        "clusters": [
            {**{key: value for key, value in entry.items()}, "share_full": float(shares_full[entry["id"]]),
             "share_train": float(shares_train[entry["id"]]), "samples": sample_records[entry["id"]]}
            for entry in stats
        ],
        "runtime_s": runtime,
        "peak_rss_mb": peak_rss_mb,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(
        f"[survey] DONE → {out_dir}（rows={total_rows}，figures={num_clusters * per_cluster}，"
        f"runtime={runtime:.1f}s，peak_rss={peak_rss_mb:.0f}MB）"
    )
    return 0


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    fingerprint = summary["obs_fingerprint"]
    lines: List[str] = []
    lines.append(f"# 聚类现状排查（cluster survey）— {summary['generated_at']}")
    lines.append("")
    lines.append(f"- 脚本：`{summary['script']}`（只读；CPU-only）")
    lines.append(f"- 数据集：`{summary['bc_dir']}`（rows={summary['rows_total']}，train_weight>0={summary['rows_train_weight_gt0']}）")
    lines.append(
        f"- spec：`{summary['spec']}`（mode={summary['spec_mode']}，k={summary['spec_k']}，"
        f"others_source={summary['spec_others_source']}，boundary_margin={summary['boundary_margin']}）"
    )
    lines.append(
        f"- 抽样：seed={summary['sampling']['seed']} · 每簇 {summary['sampling']['per_cluster']} 帧 · "
        f"池={summary['sampling']['pool']}（{summary['sampling']['deterministic']}）"
    )
    lines.append(f"- 运行：{summary['runtime_s']:.1f}s，峰值 RSS ≈ {summary['peak_rss_mb']:.0f} MB")
    sidecar = summary.get("assignments_sidecar") or {}
    if sidecar:
        if sidecar.get("reused"):
            lines.append(
                f"- 全量分配：复用 sidecar `{sidecar.get('path')}`（spec_hash={str(sidecar.get('spec_hash'))[:12]}，"
                f"created_at={sidecar.get('created_at', '')}）"
            )
        elif sidecar.get("written"):
            lines.append(
                f"- 全量分配：本次重算并写 sidecar `{sidecar.get('path')}`"
                f"（spec_hash={str(sidecar.get('spec_hash'))[:12]}，created_at={sidecar.get('created_at', '')}，"
                f"git={str(sidecar.get('git_hash', ''))[:8]}）"
            )
        else:
            lines.append(
                f"- 全量分配：本次重算；sidecar 写入失败（{sidecar.get('error', '')}）→ `{sidecar.get('path')}`"
            )
    lines.append("")
    if not fingerprint["match"]:
        lines.append("## ⚠️ 指纹告警（必须阅读）")
        lines.append("")
        lines.append(
            f"- spec 拟合时 `obs_fingerprint` = `{fingerprint['spec_obs_fingerprint']}`"
            f"（fitted_at={fingerprint['spec_fitted_at']}，git={fingerprint['spec_fit_git_hash']}，"
            f"rows={fingerprint['spec_fit_rows']}，source={fingerprint['spec_fit_source']}）"
        )
        lines.append(f"- 当前数据集 `obs_fingerprint` = `{fingerprint['dataset_obs_fingerprint']}`")
        lines.append(f"- 结论：**不一致**。{fingerprint['warning']}")
        lines.append("")
    else:
        lines.append(
            f"## 指纹一致性：✅ match（spec={fingerprint['spec_obs_fingerprint']}，"
            f"dataset={fingerprint['dataset_obs_fingerprint']}）"
        )
        lines.append("")
    lines.append("## 占比（top-1 硬簇号 = argmax 软目标）")
    lines.append("")
    lines.append("| 簇 | 全量 count | 全量 share | train_weight>0 count | train share |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for cluster in summary["clusters"]:
        lines.append(
            f"| c{cluster['id']} | {cluster['count_full']} | {cluster['share_full']:.4f} | "
            f"{cluster['count_train']} | {cluster['share_train']:.4f} |"
        )
    lines.append("")
    lines.append("## 每簇画像（全量口径）")
    lines.append("")
    lines.append(
        "| 簇 | difficulty (easy/medium/hard) | geometry top3 | labels top3（占比） | lane_lat mean | "
        "od fresh/stale | ld valid | top1 prob | boundary share |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for cluster in summary["clusters"]:
        difficulty = cluster.get("difficulty") or {}
        difficulty_text = "/".join(f"{difficulty.get(name, 0.0):.2f}" for name in ("easy", "medium", "hard"))
        geometry_text = ", ".join(f"{name}={share:.2f}" for name, share in (cluster.get("geometry_top3") or []))
        labels_text = ", ".join(f"{name}={rate:.3f}" for name, rate in (cluster.get("label_top3") or []))
        lines.append(
            f"| c{cluster['id']} | {difficulty_text} | {geometry_text} | {labels_text} | "
            f"{cluster['lane_lat_mean']:+.3f} | {cluster['od_fresh_mean']:.2f}/{cluster['od_stale_mean']:.2f} | "
            f"{cluster['ld_valid_mean']:.2f} | {cluster['top1_prob_mean']:.3f} | {cluster['boundary_share']:.3f} |"
        )
    lines.append("")
    lines.append("## 与体检报告（60k 子采样）的 shares 对照")
    lines.append("")
    comparison = summary.get("report_comparison") or {}
    if comparison.get("available"):
        lines.append(
            f"- 报告：`{comparison['report_path']}`（fit_rows={comparison['fit_rows']} / rows_total={comparison['fit_rows_total']}，"
            f"source={comparison['fit_source']}，git={comparison['fit_git_hash']}，"
            f"fit obs_fingerprint={comparison['fit_obs_fingerprint']}）"
        )
        lines.append("")
        lines.append("| 簇 | 报告 share（60k 子采样） | 全量 share（本次，360,501 行） | Δ(全量−报告) | train share（本次） |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for index, cluster in enumerate(summary["clusters"]):
            lines.append(
                f"| c{index} | {comparison['report_shares'][index]:.4f} | {comparison['survey_full_shares'][index]:.4f} | "
                f"{comparison['delta_full_minus_report'][index]:+.4f} | {comparison['survey_train_shares'][index]:.4f} |"
            )
        lines.append("")
        lines.append(f"> {comparison['note']}")
    else:
        lines.append(f"- 不可用：{comparison.get('note', '未知原因')}")
    lines.append("")
    lines.append("## 字段语义（读自 `env/obs/` 源码；字段 → 绘图对应）")
    lines.append("")
    for key, text in summary["field_semantics"].items():
        lines.append(f"- `{key}`：{text}")
    lines.append("")
    lines.append("## 抽样清单（文件名规则）")
    lines.append("")
    lines.append(
        "- 目录：`cluster_0{k}/`；文件名：`c{k:02d}_ep{episode_id:04d}_step{step:04d}_row{row:06d}.png`；"
        "每簇附 `contact_sheet.png`（前 25 张 5 联面板拼版）。"
    )
    lines.append(
        "- 帧面板（左→右）：① scene（ld+od+ego+nav 叠加）② ld ③ od ④ ego ⑤ nav；"
        "①–③ 固定视窗 x[{:.0f},{:.0f}] y[{:.0f},{:.0f}] m（视窗外实体不画，面板角注 out-of-view 计数；"
        "`--view-window` 可调），④ ±8 m，⑤ 自适应 checkpoint 范围；"
        "角注含簇号/difficulty/geometry/lane_lat/spec_id/episode/step/有效槽位数/top1 概率/top1-top2 间隔。".format(
            *summary.get("view_window_m", [float("nan")] * 4)
        )
    )
    lines.append("- 完整抽样清单（row/episode/step/spec_id/png/top1_prob）见 `summary.json.clusters[*].samples`。")
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    lines.append("- `summary.md` / `summary.json`：本文件与机器可读版")
    lines.append("- `assignment.npz`：本次运行副本（`hard`/`top1`/`margin`/`row`）")
    lines.append(
        "- sidecar（供 train/eval 直接读）：`<bc_dir>/<spec_stem>_assignments.npz`，键 `cluster`(int16)/"
        "`top1_margin`(float32,=top1−top2)/`top1_prob`(float32)/`spec_hash`/`spec_path`/`rows`/"
        "`created_at`/`obs_fingerprint_dataset`/`obs_fingerprint_spec`/`git_hash`；"
        "另有说明键 `top1_margin_definition`/`row_alignment`/`git_dirty_entries`/`script_path`"
    )
    lines.append("- `cluster_00..07/`：抽样 PNG + contact sheet")
    lines.append("")
    lines.append("## 复算命令")
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"tools/venv-python tools/diagnostics/cluster_survey.py --bc-dir {summary['bc_dir']} "
        f"--spec {summary['spec']} --seed {summary['sampling']['seed']} "
        f"--per-cluster {summary['sampling']['per_cluster']} --out <out_dir>"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/diagnostics/cluster_survey.py",
        description="聚类现状排查 + 每簇 N 帧 5 联面板可视化（只读；CPU-only）",
    )
    parser.add_argument("--bc-dir", default=DEFAULT_BC_DIR, help=f"BC 数据集目录（默认 {DEFAULT_BC_DIR}）")
    parser.add_argument("--spec", default=DEFAULT_SPEC, help=f"冻结聚类 spec（默认 {DEFAULT_SPEC}）")
    parser.add_argument(
        "--report", default=None,
        help="体检报告（对照 60k 子采样 shares；默认 spec 同目录同名 .report.json）",
    )
    parser.add_argument(
        "--assignments-path", default=None,
        help="全量分配 sidecar 路径（默认 <bc_dir>/<spec_stem>_assignments.npz）",
    )
    parser.add_argument(
        "--refresh-assignments", action="store_true",
        help="忽略已有 sidecar，强制重算全量 top-1 分配并覆盖写",
    )
    parser.add_argument(
        "--out", default=None,
        help="输出目录（默认 runs/BTC<北京戳>_cluster_survey）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="抽样 seed（默认 0，确定性）")
    parser.add_argument("--per-cluster", type=int, default=DEFAULT_PER_CLUSTER, help="每簇抽帧数（默认 100）")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="encode batch（默认 8192）")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="单帧 PNG dpi（默认 110）")
    parser.add_argument(
        "--view-window", default=DEFAULT_VIEW_WINDOW,
        help=f"面板 ①–③ 固定视窗 xmin,xmax,ymin,ymax（m；默认 {DEFAULT_VIEW_WINDOW}）",
    )
    parser.add_argument("--no-contact-sheet", action="store_true", help="不生成每簇 25 图拼版")
    parser.add_argument("--force", action="store_true", help="输出目录非空时允许覆盖/续写")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return survey(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
