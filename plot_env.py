#!/usr/bin/env python3
"""
MF4 环境可视化工具 (Environment Visualizer for Parking Data)
============================================================
读取 MF4 文件和 PAR 标定文件, 绘制某一时刻的泊车环境:
  - 车位 (Parking Slot, IFOR Fusion)
  - 障碍物 (OD, SIFOR1)
  - 自由空间边界 (FSD, SIFOR1)
  - 虚拟边界 (Virtual Boundary)
  - 自车位姿 (Ego Pose)
  - 目标位姿/起始位姿 (TargetPose / StartPose / ITP / ISP)
  - 规划轨迹 (Trajectory)

坐标系: 右手系, 自车纵轴为 X, 横轴为 Y, 前正左正.
"""

import os
import re
import logging
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from asammdf import MDF

# Suppress asammdf verbose duplicate-channel errors
logging.getLogger('asammdf').setLevel(logging.WARNING)

# ============================================================
# GLOBALS
# ============================================================
DATASET_DIR = "/workspace/00_Dataset/Parking_P417"
PAR_FILE = os.path.join(DATASET_DIR, "01_P417_APA_PSI_PSF.par")
TIME_STEP = 0.05
AXIS_LIM = 15
SAFE_MARGIN_EN = True


# ============================================================
# 1. PAR FILE PARSER
# ============================================================
def parse_par(filepath):
    params = {}
    if not os.path.exists(filepath):
        print(f"[WARN] PAR 文件不存在: {filepath}")
        return params
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('CANape PAR'):
                continue
            m = re.match(r'^([\w\.\[\]]+)\s+\[(\w+)\]\s+([-\d.e+]+)\s*;', line)
            if m:
                params[m.group(1)] = float(m.group(3))
    return params


def get_vehicle_params(par_path):
    p = parse_par(par_path)
    veh = {}
    veh['C2R'] = p.get('Psi_VehicleParam_apv.Center2RearAxle_f32', 1.4666)
    veh['length'] = p.get('Psi_VehicleParam_apv.VehicleLength_f32', 4.8312)
    veh['width'] = p.get('Psi_VehicleParam_apv.VehicleWidth_f32', 1.978)
    veh['min_turn_radius'] = p.get('Psi_VehicleParam_apv.MinTurnRadius_f32', 5.4)
    cx, cy = [], []
    for i in range(8):
        xk = p.get(f'Psi_VehicleParam_apv.ContourPoints_af32[{i}].X_f32', None)
        yk = p.get(f'Psi_VehicleParam_apv.ContourPoints_af32[{i}].Y_f32', None)
        if xk is not None and yk is not None:
            cx.append(xk)
            cy.append(yk)
    if len(cx) >= 4:
        veh['contour_x'] = np.array(cx + [cx[0]])  # close polygon
        veh['contour_y'] = np.array(cy + [cy[0]])
    else:
        L, W, C2R = veh['length'], veh['width'], veh['C2R']
        hL, hW = L / 2, W / 2
        veh['contour_x'] = np.array([C2R + hL, C2R + hL, C2R - hL, C2R - hL, C2R + hL])
        veh['contour_y'] = np.array([-hW, hW, hW, -hW, -hW])
    print(f"[INFO] 车辆: C2R={veh['C2R']:.3f}, L={veh['length']:.3f}, W={veh['width']:.3f}")
    return veh


# ============================================================
# 2. MF4 HELPERS
# ============================================================

def find_channel(mdf, pattern):
    """按子串搜通道, 返回 (完整名, group_idx, channel_idx) 或 (None, None, None)."""
    for ch_name in mdf.channels_db:
        if pattern in ch_name:
            grp_idx, ch_idx = mdf.channels_db[ch_name][0]
            return ch_name, grp_idx, ch_idx
    return None, None, None


def find_indexed_channels(mdf, base_field):
    """
    查找带 `._0_` 或 `[00]` 索引的通道.
    返回 (prefix, field_basename, fmt_str, group_idx, channel_idx).
    """
    # _0_ style: Prefix.Field._0_
    for ch_name in mdf.channels_db:
        if base_field not in ch_name:
            continue
        m = re.search(r'\._(\d+)_$', ch_name)
        if m:
            num = m.group(1)
            pos = ch_name.rfind('._' + num + '_')
            pf = ch_name[:pos]
            dot = pf.rfind('.')
            prefix = pf[:dot] if dot >= 0 else ''
            field = pf[dot + 1:] if dot >= 0 else pf
            grp, cidx = mdf.channels_db[ch_name][0]
            return prefix, field, '._%d_', grp, cidx
    # [00] style: Prefix.Field[00]
    for ch_name in mdf.channels_db:
        if base_field not in ch_name:
            continue
        m = re.search(r'\[(\d+)\]$', ch_name)
        if m:
            num = m.group(1)
            pos = ch_name.rfind('[' + num + ']')
            pf = ch_name[:pos]
            dot = pf.rfind('.')
            prefix = pf[:dot] if dot >= 0 else ''
            field = pf[dot + 1:] if dot >= 0 else pf
            grp, cidx = mdf.channels_db[ch_name][0]
            pad = len(num)
            if pad > 1 and num.startswith('0'):
                fmt = f'[%0{pad}d]'
            else:
                fmt = '[%d]'
            return prefix, field, fmt, grp, cidx
    return None, None, None, None, None


def read_structured(mdf, ch_name, group_idx, channel_idx=0):
    """
    读取结构化 dtype 通道 (如包含数组的 struct).
    返回 (timestamps, data_2d) 其中 data_2d.shape = (N, K).
    """
    try:
        sig = mdf.get(ch_name, group=group_idx, index=channel_idx)
    except Exception:
        return None, None
    if sig is None:
        return None, None
    ts = np.asarray(sig.timestamps, dtype=float).ravel()
    sd = sig.samples
    if sd.dtype.names:
        field = sd.dtype.names[0]
        data = np.asarray(sd[field], dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        return ts, data
    else:
        vals = np.asarray(sd, dtype=float).ravel()
        return ts, vals[:, None]


def read_simple(mdf, ch_name, group_idx, channel_idx=0):
    """读取简单 (非结构化) 通道, 返回 (ts, vals_1d)."""
    try:
        sig = mdf.get(ch_name, group=group_idx, index=channel_idx)
    except Exception:
        return None, None
    if sig is None:
        return None, None
    ts = np.asarray(sig.timestamps, dtype=float).ravel()
    vals = np.asarray(sig.samples, dtype=float).ravel()
    return ts, vals


def nearest_interp_1d(ts, vals, t_grid):
    """一维最近邻插值."""
    if len(vals) == 0:
        return np.full_like(t_grid, np.nan, dtype=float)
    idx = np.searchsorted(ts, t_grid)
    idx = np.clip(idx, 0, len(ts) - 1)
    ip = np.clip(idx - 1, 0, len(ts) - 1)
    use_p = np.abs(ts[ip] - t_grid) < np.abs(ts[idx] - t_grid)
    idx = np.where(use_p, ip, idx)
    return vals[idx]


def nearest_interp_2d(ts, data_2d, t_grid):
    """二维最近邻插值 (data_2d: N_ts x K)."""
    if data_2d.shape[1] == 0:
        return np.full((len(t_grid), data_2d.shape[1]), np.nan)
    idx = np.searchsorted(ts, t_grid)
    idx = np.clip(idx, 0, len(ts) - 1)
    ip = np.clip(idx - 1, 0, len(ts) - 1)
    use_p = np.abs(ts[ip] - t_grid) < np.abs(ts[idx] - t_grid)
    idx = np.where(use_p, ip, idx)
    return data_2d[idx]


def build_time_grid(mdf, step=TIME_STEP):
    tmn, tmx = [], []
    for ch in list(mdf.channels_db.keys())[:200]:
        try:
            sig = mdf.get(ch)
            if sig is not None and len(sig.timestamps) > 1:
                tmn.append(float(sig.timestamps[0]))
                tmx.append(float(sig.timestamps[-1]))
        except Exception:
            pass
    if not tmn:
        return np.array([])
    return np.arange(min(tmn), max(tmx) + step, step)


# ============================================================
# 3. COORDINATE TRANSFORM
# ============================================================

def h_transform(pts, ego_x, ego_y, ego_yaw):
    """全局 -> 自车 坐标变换 (右手系, X前Y左)."""
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    x, y = pts[..., 0], pts[..., 1]
    tx = c * x - s * y + (-ego_x * c + ego_y * s)
    ty = s * x + c * y + (-ego_x * s - ego_y * c)
    r = np.zeros_like(pts)
    r[..., 0] = tx
    r[..., 1] = ty
    return r


def h_transform_single(px, py, ego_x, ego_y, ego_yaw):
    t = h_transform(np.array([[px, py]]), ego_x, ego_y, ego_yaw)
    return float(t[0, 0]), float(t[0, 1])


# ============================================================
# 4. MF4 DATA READER
# ============================================================

class MdfData:
    def __init__(self, mdf, t_grid, veh):
        self.mdf = mdf
        self.t = t_grid
        self.N = len(t_grid)
        self.veh = veh
        self._read_all()

    # ---- helpers ----
    def _sig1(self, pattern):
        """简单 1D 信号 -> 插值到 t_grid."""
        ch, grp, cidx = find_channel(self.mdf, pattern)
        if ch is None:
            return np.full(self.N, np.nan)
        ts, vals = read_simple(self.mdf, ch, grp, cidx)
        if ts is None:
            return np.full(self.N, np.nan)
        return nearest_interp_1d(ts, vals, self.t)

    def _struct(self, pattern):
        """结构化通道 -> 返回 (N, K) 数组."""
        ch, grp, cidx = find_channel(self.mdf, pattern)
        if ch is None:
            return None
        ts, data = read_structured(self.mdf, ch, grp, cidx)
        if ts is None:
            return None
        return nearest_interp_2d(ts, data, self.t)

    def _idx(self, prefix, field, fmt, grp, cidx, idx_val, sub_field=None):
        """读取索引通道."""
        if sub_field:
            name = f"{prefix}.{sub_field}{fmt % idx_val}"
        else:
            name = f"{prefix}.{field}{fmt % idx_val}"
        ts, vals = read_simple(self.mdf, name, grp, cidx)
        if ts is None:
            return np.full(self.N, np.nan)
        return nearest_interp_1d(ts, vals, self.t)

    # ---- main read ----
    def _read_all(self):
        print("  [1/5] EgoPose...")
        self._read_egopose()
        print("  [2/5] Trigger...")
        self._read_trigger()
        print("  [3/5] ParkingSlot...")
        self._read_ps()
        print("  [4/5] SIFOR1 OD/FSD...")
        self._read_sifor()
        print("  [5/5] TargetPose...")
        self._read_tp()

    def _read_egopose(self):
        self.ego_x = self._sig1('EPE_Global_Estimated_X_m_o_obv')
        self.ego_y = self._sig1('EPE_Global_Estimated_Y_m_o_obv')
        self.ego_yaw = self._sig1('EPE_Global_Estimated_Yaw_rad_o_obv')

    def _read_trigger(self):
        self.trigger = self._sig1('HAS_Event_Request_Trajectory_Trigger')

    # ---- Parking Slot (structured dtype, mirror SIFOR1 approach) ----
    def _read_ps(self):
        self.ps_px, self.ps_py = [], []
        prefix = 'Rte_Irv_SWC_APA_GLY_B1XE_HI_ZF_Fusion_Parking_Slot_Set_IRV'
        corners_x, corners_y = [], []
        ok = True
        for ci in range(4):
            d = self._struct(f'{prefix}.PS_P{ci}_X_ISO8855_Rear_Axle_m')
            if d is None:
                ok = False
                break
            corners_x.append(d)
            d = self._struct(f'{prefix}.PS_P{ci}_Y_ISO8855_Rear_Axle_m')
            corners_y.append(d)
        if not ok:
            print("  [WARN] 未找到 ParkingSlot 数据")
            self.ps_count = np.zeros(self.N, dtype=int)
            return

        # Count valid slots by non-zero P0_X per timestamp
        self.ps_count = np.sum(np.abs(corners_x[0]) > 1e-4, axis=1).astype(int)
        max_slots = corners_x[0].shape[1]
        for si in range(max_slots):
            pts_x, pts_y = [], []
            for ci in range(4):
                pts_x.append(corners_x[ci][:, si])
                pts_y.append(corners_y[ci][:, si])
            pts_x.append(pts_x[0])
            pts_y.append(pts_y[0])
            self.ps_px.append(np.column_stack(pts_x))
            self.ps_py.append(np.column_stack(pts_y))

    # ---- SIFOR1 OD / FSD (structured dtypes) ----
    def _read_sifor(self):
        # --- OD ---
        self.od_px, self.od_py, self.od_type = [], [], []
        cnt = self._sig1('SIFOR1_Valid_Fusion_Object_Set.Number_Of_Valid_Objects')
        self.od_count = np.nan_to_num(cnt, nan=0).astype(int)

        prefix = 'SIFOR1_Valid_Fusion_Object_Set'
        # Read corner points (each is (N, 64) structured)
        corners_x, corners_y = [], []
        for ci in range(4):
            d = self._struct(f'{prefix}.Object_Point_{ci}_X')
            if d is None:
                print("  [WARN] 未找到 SIFOR1 OD 数据")
                return
            corners_x.append(d)
            d = self._struct(f'{prefix}.Object_Point_{ci}_Y')
            corners_y.append(d)
        # Read type
        d = self._struct(f'{prefix}.Object_Type')
        if d is not None:
            self.od_type_data = d  # (N, 64)
        else:
            self.od_type_data = np.zeros((self.N, 64))

        # Build per-object time-series: transpose to (64, N, 5) -> list of (N, 5)
        max_od = 64
        for oi in range(max_od):
            pts_x, pts_y = [], []
            for ci in range(4):
                pts_x.append(corners_x[ci][:, oi])
                pts_y.append(corners_y[ci][:, oi])
            pts_x.append(pts_x[0])
            pts_y.append(pts_y[0])
            self.od_px.append(np.column_stack(pts_x))
            self.od_py.append(np.column_stack(pts_y))

        # --- FSD ---
        self.fsd_px, self.fsd_py, self.fsd_type, self.fsd_conf = [], [], [], []
        cnt = self._sig1('SIFOR1_Valid_Free_Space_Boundary_Set.Number_Of_Valid_Objects')
        self.fsd_count = np.nan_to_num(cnt, nan=0).astype(int)

        prefix = 'SIFOR1_Valid_Free_Space_Boundary_Set'
        corners_x, corners_y = [], []
        for ci in range(4):
            d = self._struct(f'{prefix}.Freespace_Boundary_Point_{ci}_X')
            if d is None:
                print("  [WARN] 未找到 SIFOR1 FSD 数据")
                return
            corners_x.append(d)
            d = self._struct(f'{prefix}.Freespace_Boundary_Point_{ci}_Y')
            corners_y.append(d)
        d = self._struct(f'{prefix}.Freespace_Boundary_Type')
        self.fsd_type_data = d if d is not None else np.zeros((self.N, 64))
        d = self._struct(f'{prefix}.Freespace_Boundary_Confidence')
        self.fsd_conf_data = d if d is not None else np.zeros((self.N, 64))

        max_fsd = 64
        for fi in range(max_fsd):
            pts_x, pts_y = [], []
            for ci in range(4):
                pts_x.append(corners_x[ci][:, fi])
                pts_y.append(corners_y[ci][:, fi])
            pts_x.append(pts_x[0])
            pts_y.append(pts_y[0])
            self.fsd_px.append(np.column_stack(pts_x))
            self.fsd_py.append(np.column_stack(pts_y))

    # ---- TargetPose (HAS) ----
    def _read_tp(self):
        for attr, pat in [
            ('tp_x', 'HAS_Selected_Target_Position_X_m'),
            ('tp_y', 'HAS_Selected_Target_Position_Y_m'),
            ('tp_yaw', 'HAS_Selected_Target_Yaw_Angle_rad'),
            ('sp_x', 'HAS_Selected_Start_Position_X_m'),
            ('sp_y', 'HAS_Selected_Start_Position_Y_m'),
            ('sp_yaw', 'HAS_Selected_Start_Yaw_Angle_rad'),
        ]:
            setattr(self, attr, self._sig1(pat))


# ============================================================
# 5. TIME POINT EXTRACTOR
# ============================================================

class TimePointData:
    def __init__(self, md: MdfData, idx: int):
        self.idx = idx
        self.t = md.t[idx]
        self.ego_x = float(md.ego_x[idx])
        self.ego_y = float(md.ego_y[idx])
        self.ego_yaw = float(md.ego_yaw[idx])

        def _extract_arrays(arr_list, count_arr, idx):
            """从信号数组列表中提取 idx 时刻的多边形点集."""
            result = []
            n = int(count_arr[idx]) if idx < len(count_arr) else 0
            for si in range(min(n, len(arr_list))):
                pts = arr_list[si][idx, :]
                if np.all(np.abs(pts) < 1e-6):
                    continue
                result.append(pts)
            return result

        def _extract_od(arr_list, type_data, count_arr, idx):
            """提取 OD 对象."""
            polys, types = [], []
            n = int(count_arr[idx]) if idx < len(count_arr) else 0
            for si in range(min(n, len(arr_list))):
                gx = arr_list[si][idx, :]
                if np.all(np.abs(gx) < 1e-6):
                    continue
                polys.append(gx)
                types.append(int(type_data[idx, si]) if type_data is not None else 0)
            return polys, types

        # -- parking slots (already in ego frame) --
        self.ps = []
        n_ps = int(md.ps_count[idx]) if idx < len(md.ps_count) else 0
        for si in range(min(n_ps, len(md.ps_px))):
            gx = md.ps_px[si][idx, :]
            gy = md.ps_py[si][idx, :]
            if np.any(np.abs(gx) > 1e-6):
                self.ps.append(np.column_stack([gx, gy]))

        # -- SIFOR1 OD / FSD (already in ego frame) --
        self.sifor_od_polys, self.sifor_od_types = _extract_od(md.od_px, md.od_type_data, md.od_count, idx)
        self.sifor_fsd_polys = _extract_arrays(md.fsd_px, md.fsd_count, idx)
        self.sifor_od = [np.column_stack([p, md.od_py[oi][idx, :]])
                        for oi, p in enumerate(self.sifor_od_polys)]
        self.sifor_fsd = [np.column_stack([md.fsd_px[fi][idx, :], md.fsd_py[fi][idx, :]])
                         for fi in range(len(self.sifor_fsd_polys))]

        # FSD type/conf
        self.fsd_types = [int(md.fsd_type_data[idx, fi]) if md.fsd_type_data is not None else 0 for fi in range(len(self.sifor_fsd_polys))]
        self.fsd_confs = [int(md.fsd_conf_data[idx, fi]) if md.fsd_conf_data is not None else 0 for fi in range(len(self.sifor_fsd_polys))]

        # -- TargetPose: TP + SP (already in ego frame) --
        def _tpv(name):
            a = getattr(md, name, None)
            return float(a[idx]) if a is not None and idx < len(a) else np.nan
        self.tp = (_tpv('tp_x'), _tpv('tp_y'), _tpv('tp_yaw'))
        self.sp = (_tpv('sp_x'), _tpv('sp_y'), _tpv('sp_yaw'))


# ============================================================
# 6. PLOTTING
# ============================================================

def vehicle_box_safety(veh, cx, cy, yaw):
    sx, sy = 0.1, 0.139
    C2R, L, W = veh['C2R'], veh['length'], veh['width']
    hL, hW = L / 2, W / 2
    lx = np.array([C2R + hL + sx, C2R + hL + sx, C2R - hL - sx, C2R - hL - sx, C2R + hL + sx])
    ly = np.array([-hW - sy, hW + sy, hW + sy, -hW - sy, -hW - sy])
    c, s = np.cos(yaw), np.sin(yaw)
    return c * lx - s * ly + cx, s * lx + c * ly + cy


def vehicle_box_rect(veh, cx, cy, yaw):
    C2R, L, W = veh['C2R'], veh['length'], veh['width']
    hL, hW = L / 2, W / 2
    lx = np.array([C2R + hL, C2R + hL, C2R - hL, C2R - hL, C2R + hL])
    ly = np.array([-hW, hW, hW, -hW, -hW])
    c, s = np.cos(yaw), np.sin(yaw)
    return c * lx - s * ly + cx, s * lx + c * ly + cy


def draw_pose(ax, label, x, y, yaw, veh, color='#EDB120', marker='^', show_safety=True):
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


def plot_frame(tpd: TimePointData, veh, title_str=""):
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
    ax.legend(loc='upper right', fontsize=7)
    ax.set_title(title_str)
    plt.tight_layout()
    return fig, ax


# ============================================================
# 7. TRIGGER DETECTION
# ============================================================

def find_trigger_times(trig, t_grid, min_gap=2.0):
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
# 8. MAIN
# ============================================================

def select_mf4_file(directory=DATASET_DIR):
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
