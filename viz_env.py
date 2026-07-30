#!/usr/bin/env python3
"""
MF4 环境可视化工具 (Environment Visualizer for Parking Data)
============================================================
读取 MF4 文件和 PAR 标定文件, 绘制某一时刻的泊车环境:
  - 车位 (Parking Slot)
  - 障碍物 (OD, SIFOR1)
  - 自由空间边界 (FSD, SIFOR1)
  - 自车位姿 (Ego Pose)
  - 目标位姿/起始位姿 (TargetPose / StartPose)

坐标系: 右手系, 自车纵轴为 X, 横轴为 Y, 前正左正.
"""

import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from asammdf import MDF

from mdf_reader import (
    DATASET_DIR, PAR_FILE, TIME_STEP,
    get_vehicle_params, build_time_grid,
    MdfData, TimePointData,
)


# ============================================================
# GLOBALS
# ============================================================
AXIS_LIM = 15
SAFE_MARGIN_EN = True


# ============================================================
# 1. PLOTTING UTILITIES
# ============================================================

def vehicle_box_safety(veh, cx, cy, yaw):
    """生成安全裕度矩形轮廓坐标 (含 0.1/0.139 margin)."""
    sx, sy = 0.1, 0.139
    C2R, L, W = veh['C2R'], veh['length'], veh['width']
    hL, hW = L / 2, W / 2
    lx = np.array([C2R + hL + sx, C2R + hL + sx, C2R - hL - sx, C2R - hL - sx, C2R + hL + sx])
    ly = np.array([-hW - sy, hW + sy, hW + sy, -hW - sy, -hW - sy])
    c, s = np.cos(yaw), np.sin(yaw)
    return c * lx - s * ly + cx, s * lx + c * ly + cy


def vehicle_box_rect(veh, cx, cy, yaw):
    """生成车辆矩形轮廓坐标."""
    C2R, L, W = veh['C2R'], veh['length'], veh['width']
    hL, hW = L / 2, W / 2
    lx = np.array([C2R + hL, C2R + hL, C2R - hL, C2R - hL, C2R + hL])
    ly = np.array([-hW, hW, hW, -hW, -hW])
    c, s = np.cos(yaw), np.sin(yaw)
    return c * lx - s * ly + cx, s * lx + c * ly + cy


def draw_pose(ax, label, x, y, yaw, veh, color='#EDB120', marker='^', show_safety=True):
    """绘制目标位姿 (TP/SP)."""
    if np.isnan(x) or np.isnan(y) or (abs(x) < 1e-6 and abs(y) < 1e-6):
        return
    ax.plot([x, x + np.cos(yaw)], [y, y + np.sin(yaw)], '-', color='#FFFF00', lw=1, marker='x')
    if show_safety:
        bx, by = vehicle_box_safety(veh, x, y, yaw)
        ax.plot(bx, by, '-.', color=color, lw=1, marker=marker, label=label + ' Safety')
    else:
        bx, by = vehicle_box_rect(veh, x, y, yaw)
        ax.plot(bx, by, '-', color=color, lw=1, marker=marker, label=label)


def plot_fsd_group(ax, polys, types, confs, label_prefix):
    """绘制一组 FSD 多边形, 每种类型只进一次图例."""
    shown = set()
    for xy, ft, fc in zip(polys, types, confs):
        if ft in (100, 5):
            c, m, ls, n = '#006400', 'o', '-', f'{label_prefix}: Edge Boundary'
        elif ft == 2:
            c, m, ls, n = '#77AC30', '^', '-', f'{label_prefix}: Partially Drivable'
        elif ft == 3 and fc == 1:
            c, m, ls, n = '#00FF00', 'x', '-', f'{label_prefix}: USS'
        elif ft == 3 and fc == 2:
            c, m, ls, n = '#FF7F50', 'x', '-', f'{label_prefix}: Vision-OD'
        elif ft == 3 and fc == 3:
            c, m, ls, n = '#BDB713', 'x', '-', f'{label_prefix}: USS && Vision-OD'
        elif ft == 3 and fc == 4:
            c, m, ls, n = '#556B2F', 'x', '-', f'{label_prefix}: Vision-OD High'
        elif ft == 55:
            c, m, ls, n = '#00FFF1', 'x', '-', f'{label_prefix}: Unconfirmed USS'
        elif ft == 128:
            c, m, ls, n = '#A2142F', 'x', '-', f'{label_prefix}: Road Geo'
        elif ft == 33:
            c, m, ls, n = '#024500', 's', '--', f'{label_prefix}: Virtual Wall'
        else:
            c, m, ls, n = 'm', '*', '-', f'{label_prefix}: Other'
        lbl = n if n not in shown else ''
        shown.add(n)
        ax.plot(xy[:, 0], xy[:, 1], color=c, lw=1.5, marker=m,
                linestyle=ls, label=lbl)


def draw_path(ax, path_result):
    """Draw planned trajectory on the axis as a single highlighted path.
    Direction arrows indicate travel direction (from difference between consecutive poses).
    Start (blue circle) and end (purple square) are marked.
    """
    if not path_result or not path_result.valid or len(path_result.poses) < 2:
        return
    poses = path_result.poses

    ax.plot(poses[:, 0], poses[:, 1], '-', color='#FF6F00', lw=3.0,
            alpha=0.9, label='Planned path')

    # Arrows along travel direction (from pose differences, not heading)
    step = max(1, len(poses) // 10)
    for i in range(0, len(poses) - 1, step):
        p0, p1 = poses[i], poses[min(i+1, len(poses)-1)]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        ax.arrow(p0[0], p0[1], dx, dy, head_width=0.12, head_length=0.15,
                 fc='#FF6F00', ec='#FF6F00', alpha=0.8)

    ax.plot(poses[0, 0], poses[0, 1], 'o', color='blue', ms=6, label='Path start')
    ax.plot(poses[-1, 0], poses[-1, 1], 's', color='purple', ms=6, label='Path end')


def plot_frame(tpd: TimePointData, veh, title_str="", path_result=None):
    """绘制某一时刻的完整俯视图."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Parking Slots
    for i, xy in enumerate(tpd.ps):
        ax.plot(xy[:, 0], xy[:, 1], '-', color='#000000', lw=2.5,
                label='Parking Slot' if i == 0 else '')

    # 2. OD (SIFOR1)
    n_od = len(tpd.sifor_od)
    for i, xy in enumerate(tpd.sifor_od):
        lbl = f'OD: {n_od}' if i == 0 else ''
        ax.plot(xy[:, 0], xy[:, 1], '+-', color='#0072BD', lw=1, label=lbl)
        if tpd.sifor_od_types[i] == 19:
            lbl2 = 'Wheel Stopper' if i == 0 and n_od > 0 and tpd.sifor_od_types[0] == 19 else ''
            ax.plot(xy[:, 0], xy[:, 1], '+-', color='#0000FF', lw=1, label=lbl2)

    # 3. FSD (SIFOR1)
    plot_fsd_group(ax, tpd.sifor_fsd, tpd.fsd_types, tpd.fsd_confs, 'FSD')

    # 4. Ego Vehicle
    cx, cy = veh['contour_x'], veh['contour_y']
    ax.plot(cx, cy, '-', color='#4DBEEE', lw=1.5, label='Ego Vehicle')
    sx, sy = 0.1, 0.139
    C2R, L, W = veh['C2R'], veh['length'], veh['width']
    hL, hW = L / 2, W / 2
    lsx = np.array([C2R + hL + sx, C2R + hL + sx, C2R - hL - sx, C2R - hL - sx, C2R + hL + sx])
    lsy = np.array([-hW - sy, hW + sy, hW + sy, -hW - sy, -hW - sy])
    ax.plot(lsx, lsy, '-.', color='#4DBEEE', lw=1.5)
    ax.plot(0, 0, marker='*', color='blue', ms=10)

    # 5. TargetPose: TP + SP
    draw_pose(ax, 'TP', *tpd.tp, veh, show_safety=SAFE_MARGIN_EN)
    draw_pose(ax, 'SP', *tpd.sp, veh, color='#EDB120', marker='v', show_safety=SAFE_MARGIN_EN)

    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    ax.set_xlabel('X / m')
    ax.set_ylabel('Y / m')
    ax.set_aspect('equal')
    ax.grid(True)
    draw_path(ax, path_result)
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title(title_str)
    plt.tight_layout()
    return fig, ax


# ============================================================
# 2. TRIGGER DETECTION
# ============================================================

def find_trigger_times(trig, t_grid, min_gap=2.0):
    """从触发信号中提取事件时刻."""
    if trig is None:
        return []
    sig = np.nan_to_num(trig, nan=0)
    rising = np.where((sig[:-1] < 0.5) & (sig[1:] >= 0.5))[0] + 1
    times = t_grid[rising]
    if len(times) == 0:
        return []
    filt = [times[0]]
    for t in times[1:]:
        if t - filt[-1] >= min_gap:
            filt.append(t)
    return filt


# ============================================================
# 3. MAIN
# ============================================================

def select_mf4_file(directory=DATASET_DIR):
    """打开文件选择对话框."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    fp = filedialog.askopenfilename(
        title="选择 MF4 文件",
        initialdir=directory,
        filetypes=[("MF4 files", "*.mf4"), ("All files", "*.*")]
    )
    root.destroy()
    return fp


def main():
    print("=" * 60)
    print("  MF4 环境可视化工具")
    print("=" * 60)

    print("\n[1] 选择 MF4 文件...")
    mf4_path = select_mf4_file()
    if not mf4_path:
        print("未选择文件, 退出.")
        return
    print(f"  文件: {os.path.basename(mf4_path)}")

    print("\n[2] 读取车辆参数...")
    veh = get_vehicle_params(PAR_FILE)

    print("\n[3] 打开 MF4...")
    mdf = MDF(mf4_path)
    t_grid = build_time_grid(mdf)
    print(f"  时间范围: {t_grid[0]:.2f} ~ {t_grid[-1]:.2f} s ({len(t_grid)} pts)")

    print("\n[4] 读取数据...")
    md = MdfData(mdf, t_grid, veh)

    while True:
        mode = input("\n模式: 1-手动时刻  2-规划触发  3-退出: ").strip()
        if mode == '1':
            _mode_manual(md, veh)
        elif mode == '2':
            _mode_triggers(md, veh)
        elif mode == '3':
            break

    mdf.close()
    print("再见!")


def _mode_manual(md, veh):
    try:
        t = float(input("输入时刻 (秒): "))
    except ValueError:
        print("无效数字.")
        return
    idx = int(np.argmin(np.abs(md.t - t)))
    print(f"  最近: t={md.t[idx]:.3f}s")
    tpd = TimePointData(md, idx)
    plot_frame(tpd, veh, f"t={md.t[idx]:.2f}s")
    plt.show()


def _mode_triggers(md, veh):
    trigs = find_trigger_times(md.trigger, md.t)
    if not trigs:
        print("未找到触发.")
        return
    print(f"找到 {len(trigs)} 个触发:")
    for i, t in enumerate(trigs):
        print(f"  [{i+1}] t={t:.3f}s")
    inp = input("\n编号 (0=全部, Enter=返回): ").strip()
    if not inp:
        return
    try:
        sel = int(inp)
    except ValueError:
        return
    if sel == 0:
        for i, t in enumerate(trigs):
            idx = int(np.argmin(np.abs(md.t - t)))
            tpd = TimePointData(md, idx)
            fig, ax = plot_frame(tpd, veh, f"[{i+1}/{len(trigs)}] t={md.t[idx]:.2f}s")
            plt.show(block=False)
            input(f"  Enter 下一帧...")
            plt.close(fig)
    elif 1 <= sel <= len(trigs):
        t = trigs[sel - 1]
        idx = int(np.argmin(np.abs(md.t - t)))
        tpd = TimePointData(md, idx)
        plot_frame(tpd, veh, f"Trigger [{sel}] t={md.t[idx]:.2f}s")
        plt.show()


if __name__ == '__main__':
    main()
