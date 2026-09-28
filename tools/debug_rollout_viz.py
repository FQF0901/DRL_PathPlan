#!/usr/bin/env python3
"""闭环保车道失败"回灌"可视化（Lane P3-L）：一个失败 val 场景 → 逐帧 GT vs 模型推演 PNG。

做什么
------
1. 按 ``tools/test.py --tracker lqr`` 同协议闭环驾驶所选 spec（``_CkptController`` +
   LQR tracker，0.5 s 决策，6 点 plan）并**录制每个 env step 的观测**（ObservationBuilder
   原样输出）与自车位姿；
2. 离线逐**决策帧**（0.5 s）重跑 ``model(obs, rollout=True, world_model=True)``；
3. 每帧渲染一张 PNG（左右两子图，均统一到 **t0 自车系**、同尺度同朝向）：
   - 左 = GT 真实未来 3 s：ego 真实轨迹（t0→t+3 s）+ t+3 s 的 OD 框（真实观测）+
     t+3 s 的 LD 采样点（真实观测）；
   - 右 = 模型推演：``traj_xy``（6 点 plan 端点）+ WM ``od_pred`` 第 6 步 +
     WM ``ld_pred`` 第 6 步；
4. 输出每帧 plan vs GT 轨迹偏差 ADE/FDE（控制台 + ``deviations.json`` / ``deviations.csv``）。

用法::

    tools/venv-python tools/debug_rollout_viz.py \
        --ckpt runs/BTC20260928-1630_train/stage_b/final.pt \
        --spec-id 10 \
        --out runs/BTC20260928-2100_debug_viz

默认单进程、``--device auto``（与评测同口径：cuda 可用则 cuda；GPU 被并发评测占用时
等待/回退 cpu——cpu 与 cuda 的浮点尾差可能改变临界场景的结局）。
产物：``<out>/frames/frame_XXX_step_XXXX.png``、``deviations.json``、``deviations.csv``、
``episode.json``、``README.md``。

复用：``pipeline.eval_runner``（配置加载 / ``_CkptController`` / ``_unwrap_env`` /
``_step_dt``）、``env.metadrive_env.build_env``、``net.model.DrivingModel`` 前向。
不改 ``env/``、不改训练/评测主链路。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

# 允许 `python tools/debug_rollout_viz.py` 直接运行（此时 sys.path[0] 是 tools/）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

#: 决策周期（env step，0.1 s）：与 ``_CkptController.decision_interval`` 一致
DEFAULT_DECISION_INTERVAL = 5
#: plan 视界（点）：6 点 × 0.5 s = 3.0 s
PLAN_HORIZON = 6


# --------------------------------------------------------------------------- #
# 纯几何（无 env/torch 依赖；可单测）
# --------------------------------------------------------------------------- #
def wrap_to_pi(angle: float) -> float:
    """回绕到 ``(-π, π]``。"""
    value = float(angle) % (2.0 * math.pi)
    if value > math.pi:
        value -= 2.0 * math.pi
    return value


def se2_apply(pose: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``pose=(x,y,θ)`` 自车系坐标 → 世界坐标（``points (N,2)``）。"""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    c, s = math.cos(theta), math.sin(theta)
    pts = np.asarray(points, dtype=np.float64)
    return np.stack(
        [x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], axis=1
    )


def world_to_frame(pose0: np.ndarray, points: np.ndarray) -> np.ndarray:
    """世界坐标 → ``pose0`` 自车系（t0 系）坐标。"""
    x0, y0, th0 = float(pose0[0]), float(pose0[1]), float(pose0[2])
    c, s = math.cos(th0), math.sin(th0)
    pts = np.asarray(points, dtype=np.float64)
    dx = pts[:, 0] - x0
    dy = pts[:, 1] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def box_corners(cx: float, cy: float, length: float, width: float, heading: float) -> np.ndarray:
    """t0 系里的 OD 框四角（heading=0 指向 +x；y 左向）。"""
    c, s = math.cos(heading), math.sin(heading)
    half_l, half_w = 0.5 * float(length), 0.5 * float(width)
    local = np.asarray([[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]])
    return np.stack(
        [cx + c * local[:, 0] - s * local[:, 1], cy + s * local[:, 0] + c * local[:, 1]], axis=1
    )


def ld_segments(feats: np.ndarray, mask: np.ndarray, seg_len: float = 3.0) -> list[tuple[float, float, float, float]]:
    """LD 采样点 → ``[(x0, y0, x1, y1), ...]``（每点画 ``seg_len`` 米短线段表达朝向）。"""
    out: list[tuple[float, float, float, float]] = []
    feats = np.asarray(feats)
    mask = np.asarray(mask).reshape(-1)
    for row, valid in zip(feats, mask):
        if float(valid) < 0.5:
            continue
        x, y, heading = float(row[0]), float(row[1]), float(row[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(heading)):
            continue
        dx, dy = 0.5 * seg_len * math.cos(heading), 0.5 * seg_len * math.sin(heading)
        out.append((x - dx, y - dy, x + dx, y + dy))
    return out


def plan_deviation(plan_xy: np.ndarray, gt_xy: np.ndarray, valid: np.ndarray) -> tuple[float, float, list[float], list[float]]:
    """plan vs GT 的 ADE/FDE（只在有效 GT 点上算）。返回 ``(ade, fde, 逐步距离, GT 有效位移)``。"""
    plan = np.asarray(plan_xy, dtype=np.float64).reshape(-1, 2)
    gt = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    n = int(min(len(plan), len(gt), len(valid)))
    if n == 0 or not bool(valid[:n].any()):
        return float("nan"), float("nan"), [], []
    plan, gt, valid = plan[:n], gt[:n], valid[:n]
    dist = np.linalg.norm(plan[valid] - gt[valid], axis=1)
    gt_dist = np.linalg.norm(gt[valid], axis=1)
    return float(dist.mean()), float(dist[-1]), [float(v) for v in dist], [float(v) for v in gt_dist]


def gt_future(poses: Sequence[np.ndarray], start: int, interval: int, horizon: int = PLAN_HORIZON) -> tuple[np.ndarray, np.ndarray]:
    """真实未来 3 s 的 ego 位置（t0 系）与有效性（episode 提前结束 → 尾部无效）。"""
    gt_xy = np.full((horizon, 2), np.nan, dtype=np.float64)
    valid = np.zeros(horizon, dtype=bool)
    for j in range(horizon):
        k = int(start) + interval * (j + 1)
        if 0 <= k < len(poses):
            gt_xy[j] = world_to_frame(np.asarray(poses[start]), np.asarray(poses[k])[None, :2])[0]
            valid[j] = True
    return gt_xy, valid


def gt_od_boxes(
    records: Sequence[Mapping[str, np.ndarray]],
    poses: Sequence[np.ndarray],
    start: int,
    interval: int,
    horizon: int = PLAN_HORIZON,
) -> list[dict[str, float]]:
    """``t0+3 s`` 帧的 GT OD 框（转到 t0 系；只取 ``od_presence=1`` 的新鲜观测）。"""
    k = int(start) + interval * horizon
    if k >= len(records):
        return []
    rec = records[k]
    od = np.asarray(rec["od"], dtype=np.float64)
    presence = np.asarray(rec.get("od_presence", np.zeros(len(od))), dtype=np.float64)
    pose_f = np.asarray(poses[k], dtype=np.float64)
    pose_0 = np.asarray(poses[start], dtype=np.float64)
    boxes: list[dict[str, float]] = []
    for slot, row in enumerate(od):
        if float(presence[slot]) <= 0.5:
            continue
        dx, dy = float(row[0]), float(row[1])
        if not (math.isfinite(dx) and math.isfinite(dy)):
            continue
        p0 = world_to_frame(pose_0, se2_apply(pose_f, np.asarray([[dx, dy]])))[0]
        heading = wrap_to_pi(pose_f[2] + math.atan2(float(row[5]), float(row[4])) - pose_0[2])
        boxes.append(
            {
                "x": float(p0[0]),
                "y": float(p0[1]),
                "heading": float(heading),
                "L": float(row[6]),
                "W": float(row[7]),
                "slot": float(slot),
            }
        )
    return boxes


def gt_ld_at(
    records: Sequence[Mapping[str, np.ndarray]],
    poses: Sequence[np.ndarray],
    start: int,
    interval: int,
    horizon: int = PLAN_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """``t0+3 s`` 帧的 GT LD 采样点（位置/朝向转到 t0 系；curvature/线型不变）。"""
    k = int(start) + interval * horizon
    if k >= len(records):
        return np.zeros((0, 7), dtype=np.float64), np.zeros(0, dtype=bool)
    rec = records[k]
    ld = np.asarray(rec["ld"], dtype=np.float64)
    mask = np.asarray(rec["ld_mask"], dtype=np.float64).reshape(-1)
    pose_f = np.asarray(poses[k], dtype=np.float64)
    pose_0 = np.asarray(poses[start], dtype=np.float64)
    rows = np.zeros_like(ld)
    valid = np.zeros(len(ld), dtype=bool)
    for slot, row in enumerate(ld):
        if float(mask[slot]) <= 0.5 or not math.isfinite(float(row[0])):
            continue
        p0 = world_to_frame(pose_0, se2_apply(pose_f, np.asarray([[row[0], row[1]]])))[0]
        rows[slot] = row
        rows[slot, 0], rows[slot, 1] = float(p0[0]), float(p0[1])
        rows[slot, 2] = wrap_to_pi(pose_f[2] + float(row[2]) - pose_0[2])
        valid[slot] = True
    return rows, valid


def wm_od_boxes(
    od_pred_step: np.ndarray,
    presence_step: np.ndarray,
    od_now: np.ndarray,
    od_mask_now: np.ndarray,
) -> list[dict[str, float]]:
    """WM 第 6 步 OD 预测 → 框（位置/朝向已是 t0 系；L/W 沿用 t0 帧对应槽位）。

    槽位门控：t0 帧 ``od_mask=1``（槽位存在，身份可继承）或模型 presence>0.5（预测存在）。
    ``presence`` 概率随框返回，渲染时用它区分"高置信 / 低置信"（低置信虚线淡显）。
    """
    od_pred = np.asarray(od_pred_step, dtype=np.float64)
    prob = 1.0 / (1.0 + np.exp(-np.asarray(presence_step, dtype=np.float64)))
    od_now = np.asarray(od_now, dtype=np.float64)
    mask = np.asarray(od_mask_now, dtype=np.float64).reshape(-1)
    boxes: list[dict[str, float]] = []
    for slot, pred in enumerate(od_pred):
        if float(mask[slot]) <= 0.5 and float(prob[slot]) <= 0.5:
            continue
        x, y, heading = float(pred[0]), float(pred[1]), float(pred[4])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(heading)):
            continue
        length, width = float(od_now[slot, 6]), float(od_now[slot, 7])
        if not (length > 0.0 and width > 0.0):  # 槽位无静态尺寸可继承（t0 无观测）
            continue
        boxes.append(
            {
                "x": x,
                "y": y,
                "heading": heading,
                "L": length,
                "W": width,
                "slot": float(slot),
                "presence": float(prob[slot]),
            }
        )
    return boxes


def wm_ld_at(ld_pred_step: np.ndarray, ld_mask_now: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """WM 第 6 步 LD 预测（``[x,y,heading,curvature]``，t0 系）；用 t0 帧槽位 mask 门控。"""
    pred = np.asarray(ld_pred_step, dtype=np.float64)
    mask = np.asarray(ld_mask_now, dtype=np.float64).reshape(-1)
    rows = np.zeros((len(pred), 7), dtype=np.float64)
    valid = np.zeros(len(pred), dtype=bool)
    for slot, row in enumerate(pred):
        if float(mask[slot]) <= 0.5:
            continue
        if not all(math.isfinite(float(v)) for v in row[:3]):
            continue
        rows[slot, 0], rows[slot, 1], rows[slot, 2], rows[slot, 3] = (
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
        )
        valid[slot] = True
    return rows, valid


# --------------------------------------------------------------------------- #
# 录制（在线闭环）
# --------------------------------------------------------------------------- #
def record_closed_loop(
    spec: Any,
    task: Mapping[str, Any],
    *,
    max_steps: int,
    traffic_density: Optional[float] = None,
) -> dict[str, Any]:
    """闭环跑一个 spec 并录制逐 env step 观测/位姿（LQR 协议与评测一致）。"""
    from pipeline.eval_runner import _CkptController, _step_dt, _unwrap_env  # 延迟导入（需 torch）
    from env.metadrive_env import build_env

    env = _unwrap_env(build_env(spec, traffic_density=traffic_density, use_render=False))
    try:
        controller = _CkptController(env, spec, task)
        reset_out = env.reset()
        reset_info = reset_out[1] if isinstance(reset_out, tuple) and len(reset_out) == 2 else {}
        controller.bind(env)

        records: list[dict[str, np.ndarray]] = []
        original_build = controller.builder.build

        def _recording_build(env_: Any, spec_: Any) -> dict[str, np.ndarray]:
            obs = original_build(env_, spec_)
            records.append(
                {
                    key: (np.array(value, copy=True) if isinstance(value, np.ndarray) else value)
                    for key, value in obs.items()
                }
            )
            return obs

        controller.builder.build = _recording_build  # type: ignore[method-assign]

        poses: list[np.ndarray] = []
        speeds: list[float] = []
        info: dict[str, Any] = dict(reset_info)
        terminated = truncated = False
        for step_index in range(int(max_steps)):
            ego = env.agent
            poses.append(
                np.asarray(
                    [float(ego.position[0]), float(ego.position[1]), float(ego.heading_theta)],
                    dtype=np.float64,
                )
            )
            action = controller.action(env)
            if len(records) != step_index + 1:
                raise RuntimeError(
                    f"obs 录制错位：step={step_index} records={len(records)}（ObservationBuilder 调用次数异常）"
                )
            _, _, terminated, truncated, info = env.step(action)
            info = info if isinstance(info, dict) else {}
            speeds.append(float(info.get("velocity", float(ego.speed))))
            if terminated or truncated:
                break
        dt = float(_step_dt(env))
    finally:
        try:
            env.close()
        except BaseException:  # noqa: BLE001 - 关闭失败不影响已录制数据
            pass

    return {
        "records": records,
        "poses": poses,
        "speeds": speeds,
        "dt": dt,
        "steps": len(poses),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
        "decision_interval": int(getattr(controller, "decision_interval", DEFAULT_DECISION_INTERVAL)),
        "controller": controller,
    }


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #
def _plot_boxes(ax: Any, boxes: Sequence[Mapping[str, float]], *, edge: str, label: str) -> None:
    from matplotlib.patches import Polygon

    for index, box in enumerate(boxes):
        corners = box_corners(box["x"], box["y"], box["L"], box["W"], box["heading"])
        presence = float(box.get("presence", 1.0))
        confident = presence > 0.5
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                fill=False,
                edgecolor=edge,
                linewidth=1.6 if confident else 1.0,
                linestyle="-" if confident else "--",
                alpha=0.95 if confident else 0.45,
                label=label if index == 0 else None,
            )
        )


def _plot_ld(ax: Any, rows: np.ndarray, valid: np.ndarray, *, color: str, label: str) -> None:
    segments = ld_segments(rows, valid)
    for index, (x0, y0, x1, y1) in enumerate(segments):
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.2, alpha=0.75,
                label=label if index == 0 else None)


def _plot_path(ax: Any, xy: np.ndarray, *, color: str, label: str, marker: str = "o") -> None:
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    ax.plot(pts[:, 0], pts[:, 1], color=color, marker=marker, markersize=3.5,
            linewidth=1.6, label=label)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite):
        ax.annotate("t+3s", xy=(finite[-1, 0], finite[-1, 1]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=color)


def _plot_ego(ax: Any) -> None:
    ax.plot([0.0], [0.0], marker=">", color="black", markersize=9, label="ego (t0)")
    ax.annotate("", xy=(3.0, 0.0), xytext=(0.0, 0.0),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2})
    ax.text(3.2, 0.3, "x fwd", fontsize=7, color="black")


def _plot_offscreen_boxes(ax: Any, boxes: Sequence[Mapping[str, float]], *, lim: float,
                          color: str, label: str) -> None:
    """视窗外的 OD 框：在边界画一个方向标记 + "nearest Xm" 说明（保持近场缩放不丢失上下文）。"""
    off = [box for box in boxes if max(abs(float(box["x"])), abs(float(box["y"]))) > lim * 0.98]
    if not off:
        return
    nearest = min(off, key=lambda box: math.hypot(float(box["x"]), float(box["y"])))
    x, y = float(nearest["x"]), float(nearest["y"])
    scale = (lim * 0.98) / max(abs(x), abs(y), 1e-9)
    bx, by = x * scale, y * scale
    marker = ">" if abs(bx) >= abs(by) and bx > 0 else "<" if abs(bx) >= abs(by) else "^" if by > 0 else "v"
    ax.plot([bx], [by], marker=marker, color=color, markersize=6, alpha=0.65,
            label=label, clip_on=False)
    ax.annotate(
        f"{len(off)} OD off-view (nearest {math.hypot(x, y):.0f}m)",
        xy=(bx, by), xytext=(-6 if bx > 0 else 6, 6), textcoords="offset points",
        fontsize=7, color=color, ha="right" if bx > 0 else "left",
    )


def _view_limit(
    gt_xy: np.ndarray,
    wm_xy: np.ndarray,
    *,
    boxes: Sequence[Sequence[Mapping[str, float]]] = (),
    ld_points: Sequence[np.ndarray] = (),
    context_cap: float = 35.0,
    hard_cap: float = 45.0,
) -> float:
    """统一视窗半径（m）：ego 轨迹（GT + plan）+ **近场**上下文（≤ ``context_cap`` 的 OD/LD）。

    远端的 OD 框 / LD 点在 150 m 观测 scope 内可达 40–60 m，若全量入窗会把近场轨迹压扁；
    因此只把 ``context_cap`` 内的上下文纳入窗口，之外的照旧绘制（由 matplotlib 裁剪）。
    """
    values: list[float] = [0.0]
    for points in (gt_xy, wm_xy):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        finite = pts[np.isfinite(pts).all(axis=1)]
        if len(finite):
            values.extend(np.abs(finite).ravel().tolist())
    for group in boxes:
        for box in group:
            reach = max(abs(float(box["x"])), abs(float(box["y"])))
            if reach <= context_cap:
                values.append(reach)
    for rows in ld_points:
        pts = np.asarray(rows, dtype=np.float64).reshape(-1, 2)
        finite = pts[np.isfinite(pts).all(axis=1)]
        for x, y in finite:
            reach = max(abs(float(x)), abs(float(y)))
            if reach <= context_cap:
                values.append(reach)
    return min(hard_cap, max(12.0, max(values) + 8.0))


def render_frame(
    path: Path,
    *,
    header: str,
    gt_xy: np.ndarray,
    gt_valid: np.ndarray,
    gt_od: Sequence[Mapping[str, float]],
    gt_ld: tuple[np.ndarray, np.ndarray],
    wm_xy: np.ndarray,
    wm_od: Sequence[Mapping[str, float]],
    wm_ld: tuple[np.ndarray, np.ndarray],
    ade: float,
    fde: float,
) -> None:
    """一帧一张 PNG：左 GT 真实未来 / 右 模型推演（同为 t0 自车系、同尺度）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=110)
    lim = _view_limit(
        gt_xy,
        wm_xy,
        boxes=(gt_od, wm_od),
        ld_points=(
            np.asarray(gt_ld[0], dtype=np.float64)[np.asarray(gt_ld[1], dtype=bool)][:, :2],
            np.asarray(wm_ld[0], dtype=np.float64)[np.asarray(wm_ld[1], dtype=bool)][:, :2],
        ),
    )

    for ax, title in zip(axes, ("GT (realized) t0 -> t+3 s", "Model rollout (plan + WM @ step 6)")):
        _plot_ego(ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.set_xlabel("x [m] (ego forward)", fontsize=9)
        ax.set_ylabel("y [m] (ego left)", fontsize=9)

    _plot_ld(axes[0], gt_ld[0], gt_ld[1], color="0.55", label="LD GT @ t+3s")
    _plot_path(axes[0], gt_xy, color="#1f77b4", label="ego GT path")
    _plot_boxes(axes[0], gt_od, edge="#2ca02c", label="OD GT @ t+3s")
    _plot_offscreen_boxes(axes[0], gt_od, lim=lim, color="#2ca02c", label="OD GT off-view")

    _plot_ld(axes[1], wm_ld[0], wm_ld[1], color="#9467bd", label="LD WM @ step 6")
    _plot_path(axes[1], wm_xy, color="#ff7f0e", label="traj_xy (plan)")
    _plot_boxes(axes[1], wm_od, edge="#d62728", label="OD WM @ step 6")
    _plot_offscreen_boxes(axes[1], wm_od, lim=lim, color="#d62728", label="OD WM off-view")

    for ax in axes:
        legend = ax.legend(loc="upper left", fontsize=8, framealpha=0.85)
        legend.set_zorder(10)

    ade_text = "ADE=nan FDE=nan" if not math.isfinite(ade) else f"ADE={ade:.2f}m FDE={fde:.2f}m"
    fig.suptitle(f"{header} | {ade_text} | GT pts={int(np.sum(gt_valid))}/{PLAN_HORIZON}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 输出辅助
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_output_readme(path: Path, *, ckpt: str, spec_id: int, out_dir: Path) -> None:
    path.write_text(
        f"""# debug_rollout_viz 输出说明（spec {spec_id}）

本目录由 `tools/debug_rollout_viz.py` 生成；ckpt = `{ckpt}`。

## 怎么跑

```bash
tools/venv-python tools/debug_rollout_viz.py \\
    --ckpt {ckpt} \\
    --spec-id {spec_id} \\
    --out {out_dir}
```

- 默认 `--device auto`（= 评测同口径：cuda 可用则 cuda）。GPU 空闲显存不足（可能有并发评测）时
  先等待、仍不足则自动回退 cpu 并告警——单进程低并发，不会与评测抢资源。
- 注意：cpu 与 cuda 的浮点尾差会改变**临界**场景的结局（实测 spec 23：cpu=max_step / cuda=out_of_road@226
  与评测一致）。要复现 episodes.csv 的失败轨迹，请确保实际用的是 cuda（`episode.json::device`）。
- 场景默认取 `config/eval.yaml::eval.spec`（`scenarios_eval500.json`）；`--spec-file` 可改。
- `--max-steps` 默认 1000（与评测一致）；`--limit-frames N` 只渲染前 N 帧（调试用）。

## 怎么看

每个决策帧（0.5 s）一张 `frames/frame_XXX_step_XXXX.png`，左右两子图**同为 t0 自车系**
（ego 在原点、x 前向、y 左向）、同尺度同朝向：

- 左（GT，真实发生的未来 3 s）：蓝线 = ego 真实轨迹 t0→t+3 s；绿框 = t+3 s 真实 OD 观测；
  灰短线 = t+3 s 真实 LD 采样点（短线方向 = 采样点车道朝向）。
- 右（模型推演）：橙线 = `traj_xy`（rollout 的 6 点 plan 端点，t=0.5…3.0 s）；
  红框 = WM `od_pred` 第 6 步（实线 = presence>0.5；虚线淡显 = t0 槽位存在但模型 presence 低）；
  紫短线 = WM `ld_pred` 第 6 步。
- 图题 = spec id / 帧号 / env step（0.1 s 为单位）/ 时间 / 该帧 plan vs GT 的 ADE、FDE。
- 视窗按 ego 轨迹（GT + plan）+ 近场上下文（≤35 m 的 OD/LD）自动取半径、上限 45 m；更远的 OD/LD 会被裁剪
  （OD 框会在边界画方向标记 + `nearest Xm` 说明，不丢失"远处还有多少目标"的信息）。

## 量化偏差

- `deviations.json`：逐帧 `{{frame, env_step, t_s, ade_m, fde_m, n_gt_points, dists_m, gt_dists_m}}` + 汇总；
- `deviations.csv`：同数据的平铺表；
- `episode.json`：场景/ckpt/终止类型/步数等元信息。

注意：episode 若在 t+3 s 前结束（如 out_of_road），GT 未来按有效点数截断（`n_gt_points<6`），
帧题会显示 `GT pts=k/6`；对应帧 ADE/FDE 只在有效点上计算。
""",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _gpu_free_mib() -> Optional[float]:
    """当前 CUDA 设备空闲显存（MiB）；不可用时 None。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return float(free) / (1024.0 * 1024.0)
    except Exception:  # noqa: BLE001 - 读取失败按"未知"处理
        return None


def _resolve_device_with_guard(
    requested: str,
    config: Mapping[str, Any],
    *,
    floor_mib: float = 2048.0,
    wait_s: float = 60.0,
) -> tuple[str, dict[str, Any]]:
    """解析设备（默认 auto = 与评测同口径）；并发评测占用 GPU 时等待/回退 cpu（单进程低并发）。"""
    from pipeline.trainer import resolve_device

    device = resolve_device(requested, config)
    report: dict[str, Any] = {"requested": str(requested), "resolved": str(device)}
    if not str(device).startswith("cuda"):
        return str(device), report
    free = _gpu_free_mib()
    report["free_mib_before"] = free
    deadline = time.time() + max(0.0, float(wait_s))
    while free is not None and free < float(floor_mib) and time.time() < deadline:
        print(f"[viz] GPU 空闲显存 {free:.0f} MiB < {floor_mib:.0f} MiB（可能有并发评测）→ 等待…", flush=True)
        time.sleep(5.0)
        free = _gpu_free_mib()
    report["free_mib_after"] = free
    if free is not None and free < float(floor_mib):
        report["fallback_to_cpu"] = True
        print(f"[viz] 等待后显存仍 {free:.0f} MiB < {floor_mib:.0f} MiB → 回退 cpu（数值可能与 cuda 评测有尾差）", flush=True)
        return "cpu", report
    return str(device), report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tools/debug_rollout_viz.py",
        description="失败场景回灌可视化：LQR 闭环录制 + 离线 rollout/WM 推演 + 逐帧 GT 对比 PNG",
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="模型权重（如 runs/.../stage_b/final.pt）")
    parser.add_argument("--spec-id", type=int, required=True, help="val 场景 spec id（如 eval500 episodes.csv 里的 id）")
    parser.add_argument("--spec-file", type=Path, default=None,
                        help="场景 spec 文件（默认取 config eval.spec = scenarios_eval500.json）")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                        help="主配置（默认 config/default.yaml）")
    parser.add_argument("--out", type=Path, default=None,
                        help="输出目录（默认 runs/BTC<北京戳>_debug_viz_spec<id>）")
    parser.add_argument("--seed", type=int, default=0, help="torch/numpy 随机种子（默认 0；env 用 spec.seed）")
    parser.add_argument("--max-steps", type=int, default=None, help="单 episode 最大 env step（默认评测口径 1000）")
    parser.add_argument("--device", default="auto",
                        help="策略设备（默认 auto：与评测同口径；GPU 显存不足时自动回退 cpu 并告警）")
    parser.add_argument("--torch-threads", type=int, default=4, help="torch CPU 线程上限（默认 4）")
    parser.add_argument("--traffic-density", type=float, default=None, help="覆盖 spec 交通密度（默认按 spec）")
    parser.add_argument("--limit-frames", type=int, default=None, help="只渲染前 N 个决策帧（默认全部）")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.ckpt.is_file():
        print(f"[viz] ckpt 不存在：{args.ckpt}", file=sys.stderr)
        return 2
    if not args.config.is_file():
        print(f"[viz] 配置不存在：{args.config}", file=sys.stderr)
        return 2

    # GL 修复：在 import metadrive/panda3d 之前预载 venv glvnd（与 tools/test.py 同路径）。
    from pipeline.gl_runtime import ensure_gl_library_path

    ensure_gl_library_path(preload=True)

    import torch

    from pipeline.eval_runner import DEFAULT_MAX_STEPS, load_config
    from pipeline.run_paths import beijing_stamp

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.set_num_threads(max(1, int(args.torch_threads)))

    config = load_config(args.config)
    device, device_report = _resolve_device_with_guard(args.device, config)
    eval_cfg = dict(config.get("eval") or {})
    spec_file = Path(args.spec_file or eval_cfg.get("spec") or "env/specs/scenarios_eval500.json")
    if not spec_file.is_file():
        print(f"[viz] spec 文件不存在：{spec_file}", file=sys.stderr)
        return 2

    from env.scenario.spec import load_specs

    specs = {int(getattr(spec, "id", -1)): spec for spec in load_specs(spec_file)}
    if args.spec_id not in specs:
        print(f"[viz] spec id {args.spec_id} 不在 {spec_file}（共 {len(specs)} 条）", file=sys.stderr)
        return 2
    spec = specs[args.spec_id]

    out_dir = Path(args.out) if args.out is not None else Path(
        f"runs/BTC{beijing_stamp()}_debug_viz_spec{args.spec_id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = dict(config.get("env") or {})
    model_config = {
        "hidden_dim": config.get("hidden_dim", 128),
        "moe": dict(config.get("moe") or {}),
        "world_model": dict(config.get("world_model") or {}),
    }
    max_steps = int(args.max_steps) if args.max_steps is not None else int(DEFAULT_MAX_STEPS)
    task = {
        "ckpt": str(args.ckpt),
        "model_config": model_config,
        "obs_config": dict(env_cfg.get("obs") or {}),
        "tracker_config": dict(env_cfg.get("tracking") or {}),
        "device": str(device),
        "tracker": "lqr",
        "moe_off": False,
    }

    print(
        f"[viz] spec={args.spec_id} seed={getattr(spec, 'seed', None)} "
        f"labels={getattr(spec, 'labels', None)} ckpt={args.ckpt} device={device} out={out_dir}",
        flush=True,
    )

    started = time.time()
    episode = record_closed_loop(spec, task, max_steps=max_steps, traffic_density=args.traffic_density)
    records = episode["records"]
    poses = episode["poses"]
    interval = int(episode["decision_interval"])
    steps = int(episode["steps"])
    dt = float(episode["dt"])
    info = dict(episode["info"])

    termination = (
        "arrive_dest" if bool(info.get("arrive_dest", False))
        else "collision" if bool(info.get("crash", False))
        else "out_of_road" if bool(info.get("out_of_road", False))
        else "max_step"
    )
    print(
        f"[viz] 录制完成：steps={steps} ({steps * dt:.1f}s) termination={termination} "
        f"rc={info.get('route_completion')} records={len(records)}",
        flush=True,
    )
    if termination != "out_of_road":
        print(
            "[viz] 注意：本场景实际 termination != out_of_road（与源 episodes.csv 选择可能不一致），"
            "仍继续出图。",
            flush=True,
        )

    # ---- 离线：逐决策帧重跑 rollout + WM（复用在线 controller 的模型与组 batch 口径）----
    controller = episode["controller"]
    model = controller.model
    decision_steps = list(range(0, steps, interval))
    if args.limit_frames is not None:
        decision_steps = decision_steps[: max(0, int(args.limit_frames))]
    frames: list[dict[str, Any]] = []
    rendered = 0
    for frame_index, start in enumerate(decision_steps):
        obs = records[start]
        tensors = controller._tensors(obs)  # 与评测 `_CkptController.action` 完全同口径
        with torch.no_grad():
            output = model(tensors, rollout=True, world_model=True)

        traj_xy = output["traj_xy"][0].detach().cpu().numpy()
        od_step = output["od_pred"][0, PLAN_HORIZON - 1].detach().cpu().numpy()
        ld_step = output["ld_pred"][0, PLAN_HORIZON - 1].detach().cpu().numpy()
        pres_step = output["od_presence_pred"][0, PLAN_HORIZON - 1].detach().cpu().numpy()

        gt_xy, gt_valid = gt_future(poses, start, interval)
        gy_od = gt_od_boxes(records, poses, start, interval)
        gy_ld = gt_ld_at(records, poses, start, interval)
        wm_od = wm_od_boxes(od_step, pres_step, obs["od"], obs["od_mask"])
        wm_ld = wm_ld_at(ld_step, obs["ld_mask"])
        ade, fde, dists, gt_dists = plan_deviation(traj_xy, gt_xy, gt_valid)
        presence_prob = 1.0 / (1.0 + np.exp(-pres_step))
        frame_pmax = float(np.max(presence_prob)) if presence_prob.size else 0.0

        t_s = start * dt
        header = (
            f"spec {args.spec_id} | {getattr(spec, 'labels', {}).get('difficulty', '?')}"
            f"/{getattr(spec, 'labels', {}).get('geometry', '?')} | "
            f"frame {frame_index + 1}/{len(decision_steps)} | env_step {start} | t={t_s:.1f}s"
        )
        png = out_dir / "frames" / f"frame_{frame_index:03d}_step_{start:04d}.png"
        render_frame(
            png,
            header=header,
            gt_xy=gt_xy,
            gt_valid=gt_valid,
            gt_od=gy_od,
            gt_ld=gy_ld,
            wm_xy=traj_xy,
            wm_od=wm_od,
            wm_ld=wm_ld,
            ade=ade,
            fde=fde,
        )
        rendered += 1
        frames.append(
            {
                "frame": frame_index,
                "env_step": int(start),
                "t_s": float(t_s),
                "ade_m": ade,
                "fde_m": fde,
                "n_gt_points": int(np.sum(gt_valid)),
                "dists_m": dists,
                "gt_dists_m": gt_dists,
                "plan_xy": [[float(v) for v in row] for row in np.asarray(traj_xy).reshape(-1, 2)],
                "gt_xy": [[None if not math.isfinite(float(v)) else float(v) for v in row]
                          for row in np.asarray(gt_xy).reshape(-1, 2)],
                "n_gt_od": len(gy_od),
                "n_wm_od": len(wm_od),
                "n_wm_od_conf": int(sum(1 for box in wm_od if float(box.get("presence", 1.0)) > 0.5)),
                "gt_od_boxes": [[round(float(b["x"]), 2), round(float(b["y"]), 2)] for b in gy_od],
                "wm_od_boxes": [[round(float(b["x"]), 2), round(float(b["y"]), 2),
                                 round(float(b.get("presence", 1.0)), 3)] for b in wm_od],
                "wm_presence_max": frame_pmax,
                "wm_presence_n_gt05": int(np.sum(presence_prob > 0.5)),
                "od_now_with_size": int(np.sum(np.asarray(obs["od"])[:, 6] > 0.0)),
                "n_gt_ld": int(np.sum(gy_ld[1])),
                "n_wm_ld": int(np.sum(wm_ld[1])),
                "png": str(png.relative_to(out_dir)),
            }
        )
        ade_text = "ADE=nan FDE=nan" if not math.isfinite(ade) else f"ADE={ade:.2f}m FDE={fde:.2f}m"
        frame_conf = sum(1 for box in wm_od if float(box.get("presence", 1.0)) > 0.5)
        print(
            f"[frame {frame_index:03d}] env_step={start:>4} t={t_s:5.1f}s GTpts={int(np.sum(gt_valid))}/6 "
            f"{ade_text} od(gt/wm/conf)={len(gy_od)}/{len(wm_od)}/{frame_conf} "
            f"ld(gt/wm)={int(np.sum(gy_ld[1]))}/{int(np.sum(wm_ld[1]))} wm_pmax={frame_pmax:.2f}",
            flush=True,
        )

    # ---- 汇总输出 ----
    valid_ades = [f["ade_m"] for f in frames if math.isfinite(float(f["ade_m"]))]
    valid_fdes = [f["fde_m"] for f in frames if math.isfinite(float(f["fde_m"]))]
    summary = {
        "n_frames": len(frames),
        "n_frames_with_gt": len(valid_ades),
        "ade_mean_m": float(np.mean(valid_ades)) if valid_ades else None,
        "ade_min_m": float(np.min(valid_ades)) if valid_ades else None,
        "ade_max_m": float(np.max(valid_ades)) if valid_ades else None,
        "fde_mean_m": float(np.mean(valid_fdes)) if valid_fdes else None,
        "ade_first3_m": [f["ade_m"] for f in frames[:3]],
        "ade_last3_m": [f["ade_m"] for f in frames[-3:]],
    }
    deviations = {
        "spec_id": int(args.spec_id),
        "ckpt": str(args.ckpt),
        "spec_file": str(spec_file),
        "termination": termination,
        "frames": frames,
        "summary": summary,
    }
    (out_dir / "deviations.json").write_text(
        json.dumps(_jsonable(deviations), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "deviations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "env_step", "t_s", "ade_m", "fde_m", "n_gt_points", "n_gt_od", "n_wm_od", "n_gt_ld", "n_wm_ld"])
        for frame in frames:
            writer.writerow(
                [frame["frame"], frame["env_step"], f"{frame['t_s']:.1f}",
                 "" if frame["ade_m"] is None else f"{frame['ade_m']:.4f}",
                 "" if frame["fde_m"] is None else f"{frame['fde_m']:.4f}",
                 frame["n_gt_points"], frame["n_gt_od"], frame["n_wm_od"], frame["n_gt_ld"], frame["n_wm_ld"]]
            )

    episode_doc = {
        "spec": {
            "id": int(args.spec_id),
            "seed": int(getattr(spec, "seed", -1)),
            "split": getattr(spec, "split", None),
            "labels": getattr(spec, "labels", None),
        },
        "ckpt": str(args.ckpt),
        "tracker": "lqr",
        "device": str(device),
        "device_report": dict(device_report),
        "max_steps": max_steps,
        "steps": steps,
        "duration_s": steps * dt,
        "termination": termination,
        "route_completion": info.get("route_completion"),
        "collision": bool(info.get("crash", False)),
        "out_of_road": bool(info.get("out_of_road", False)),
        "mean_speed_mps": float(np.mean(episode["speeds"])) if episode["speeds"] else None,
        "final_speed_mps": float(episode["speeds"][-1]) if episode["speeds"] else None,
        "decision_interval": interval,
        "n_frames_rendered": rendered,
        "wall_time_s": float(time.time() - started),
    }
    (out_dir / "episode.json").write_text(
        json.dumps(_jsonable(episode_doc), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_output_readme(out_dir / "README.md", ckpt=str(args.ckpt), spec_id=int(args.spec_id), out_dir=out_dir)

    print(
        f"[viz] 完成：PNG={rendered} → {out_dir / 'frames'} | ADE mean={summary['ade_mean_m']} | "
        f"termination={termination}@{steps} | wall={episode_doc['wall_time_s']:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
