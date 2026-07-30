#!/usr/bin/env python3
"""
MF4 信号读取层 (Signal Reader for Parking Data)
================================================
读取 MF4 文件和 PAR 标定文件, 提取泊车环境信号:
  - 自车位姿 (EgoPose)
  - 车位 (Parking Slot)
  - 障碍物 (OD, SIFOR1)
  - 自由空间边界 (FSD, SIFOR1)
  - 目标/起始位姿 (TargetPose / StartPose)

坐标系: 右手系, 自车纵轴为 X, 横轴为 Y, 前正左正.
"""

import os
import re
import logging

import numpy as np
from asammdf import MDF

# Suppress asammdf verbose duplicate-channel errors and missing-attachment warnings
asammdf_log = logging.getLogger('asammdf')
asammdf_log.setLevel(logging.WARNING)
class AsammdfFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if 'not found' in msg or 'Attachment' in msg or 'Exception during attachment' in msg:
            return False
        return True
asammdf_log.addFilter(AsammdfFilter())

# ============================================================
# GLOBALS
# ============================================================
DATASET_DIR = "/workspace/00_Dataset/Parking_P417"
PAR_FILE = os.path.join(DATASET_DIR, "01_P417_APA_PSI_PSF.par")
TIME_STEP = 0.05


# ============================================================
# 1. PAR FILE PARSER
# ============================================================
def parse_par(filepath):
    """解析 PAR 标定文件, 返回 {键: 值} 字典."""
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
    """从 PAR 文件提取车辆参数 (轮廓, 尺寸, 转弯半径等)."""
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
    # [00] style
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
    """根据 MF4 中关键信号的时间范围构建等步长时间网格."""
    # Probe known high-frequency signals for time range
    known = [
        'EPE_Global_Estimated_X_m_o_obv',
        'EPE_Global_Estimated_Y_m_o_obv',
        'HAS_Event_Request_Trajectory_Trigger',
        'SIFOR1_Valid_Fusion_Object_Set',
    ]
    tmn, tmx = [], []
    for pat in known:
        ch, grp, cidx = find_channel(mdf, pat)
        if ch is None:
            continue
        try:
            sig = mdf.get(ch, group=grp, index=cidx)
            if sig is not None and len(sig.timestamps) > 1:
                tmn.append(float(sig.timestamps[0]))
                tmx.append(float(sig.timestamps[-1]))
        except Exception:
            pass
    # Fallback: scan all channels if known signals not found
    if not tmn:
        for ch in list(mdf.channels_db.keys()):
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
    """单点变换."""
    t = h_transform(np.array([[px, py]]), ego_x, ego_y, ego_yaw)
    return float(t[0, 0]), float(t[0, 1])


# ============================================================
# 4. MF4 DATA READER
# ============================================================

class MdfData:
    """从 MF4 读取并插值所有信号到统一时间网格."""

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

    def _read_ps(self):
        """读取 Parking Slot (structured dtype, 镜像 SIFOR1 方法)."""
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
        self.ps_px, self.ps_py = [], []
        for si in range(max_slots):
            pts_x, pts_y = [], []
            for ci in range(4):
                pts_x.append(corners_x[ci][:, si])
                pts_y.append(corners_y[ci][:, si])
            pts_x.append(pts_x[0])
            pts_y.append(pts_y[0])
            self.ps_px.append(np.column_stack(pts_x))
            self.ps_py.append(np.column_stack(pts_y))

    def _read_sifor(self):
        """读取 SIFOR1 OD / FSD (structured dtypes)."""
        # --- OD ---
        self.od_px, self.od_py, self.od_type = [], [], []
        cnt = self._sig1('SIFOR1_Valid_Fusion_Object_Set.Number_Of_Valid_Objects')
        self.od_count = np.nan_to_num(cnt, nan=0).astype(int)

        prefix = 'SIFOR1_Valid_Fusion_Object_Set'
        corners_x, corners_y = [], []
        for ci in range(4):
            d = self._struct(f'{prefix}.Object_Point_{ci}_X')
            if d is None:
                print("  [WARN] 未找到 SIFOR1 OD 数据")
                return
            corners_x.append(d)
            d = self._struct(f'{prefix}.Object_Point_{ci}_Y')
            corners_y.append(d)
        d = self._struct(f'{prefix}.Object_Type')
        self.od_type_data = d if d is not None else np.zeros((self.N, 64))

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

    def _read_tp(self):
        """读取 TargetPose / StartPose."""
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

# ============================================================
# 6. SLOT TYPE LOOKUP
# ============================================================

def get_selected_slot_type(mdf, t_query):
    """
    读取目标库位的类型.

    方法: 先尝试 PFSM*Driver_Selected_Parking_Space_ID → PS_SlotType array lookup,
          回退到 SA_Ego_PS_SlotType_NU.
    
    返回: (slot_type_int, source_string) 或 (None, 'no_signal').
    """
    import numpy as np

    # Method 1: PFSM ID → slot type array lookup
    try:
        id_ch, id_g, id_c = find_channel(mdf, 'Driver_Selected_Parking_Space_ID')
        if id_ch is not None and id_g is not None and id_c is not None:
            id_ts, id_v = read_simple(mdf, id_ch, id_g, id_c)
            if id_ts is not None and len(id_ts) > 0:
                id_idx = int(np.argmin(np.abs(id_ts - t_query)))
                sel_id = int(id_v[id_idx])
                if sel_id > 0:  # valid ID
                    # Look up TP_100ms PS_SlotType._sel_id_
                    st_ch_name = f'TP_100ms_ZF_Fusion_Parking_Slot_Set_i_obv.PS_SlotType._{sel_id}_'
                    for ch_n in mdf.channels_db:
                        if st_ch_name == ch_n:
                            st_ch, st_g, st_c = ch_n, mdf.channels_db[ch_n][0][0], mdf.channels_db[ch_n][0][1]
                            st_ts, st_v = read_simple(mdf, st_ch, st_g, st_c)
                            if st_ts is not None and len(st_ts) > 0:
                                st_idx = int(np.argmin(np.abs(st_ts - t_query)))
                                st_val = int(st_v[st_idx])
                                if st_val > 0:  # only return valid type
                                    return (st_val, 'pfsm_lookup')
    except Exception:
        pass

    # Method 2: Fallback to SA_Ego_PS_SlotType_NU
    try:
        st_ch, st_g, st_c = find_channel(mdf, 'SA_Ego_PS_SlotType_NU')
        if st_ch is not None and st_g is not None and st_c is not None:
            st_ts, st_v = read_simple(mdf, st_ch, st_g, st_c)
            if st_ts is not None and len(st_ts) > 0:
                st_idx = int(np.argmin(np.abs(st_ts - t_query)))
                return (int(st_v[st_idx]), 'ego_slottype')
    except Exception:
        pass

    return (None, 'no_signal')


class TimePointData:
    """从 MdfData 中提取某一时刻的快照数据."""

    def __init__(self, md: MdfData, idx: int):
        self.idx = idx
        self.t = md.t[idx]
        self.ego_x = float(md.ego_x[idx])
        self.ego_y = float(md.ego_y[idx])
        self.ego_yaw = float(md.ego_yaw[idx])

        def _extract_arrays(arr_list, count_arr, idx):
            result = []
            n = int(count_arr[idx]) if idx < len(count_arr) else 0
            for si in range(min(n, len(arr_list))):
                pts = arr_list[si][idx, :]
                if np.all(np.abs(pts) < 1e-6):
                    continue
                result.append(pts)
            return result

        def _extract_od(arr_list, type_data, count_arr, idx):
            polys, types = [], []
            n = int(count_arr[idx]) if idx < len(count_arr) else 0
            for si in range(min(n, len(arr_list))):
                gx = arr_list[si][idx, :]
                if np.all(np.abs(gx) < 1e-6):
                    continue
                polys.append(gx)
                types.append(int(type_data[idx, si]) if type_data is not None else 0)
            return polys, types

        # -- parking slots --
        self.ps = []
        n_ps = int(md.ps_count[idx]) if idx < len(md.ps_count) else 0
        for si in range(min(n_ps, len(md.ps_px))):
            gx = md.ps_px[si][idx, :]
            gy = md.ps_py[si][idx, :]
            if np.any(np.abs(gx) > 1e-6):
                self.ps.append(np.column_stack([gx, gy]))

        # -- SIFOR1 OD / FSD --
        self.sifor_od_polys, self.sifor_od_types = _extract_od(md.od_px, md.od_type_data, md.od_count, idx)
        self.sifor_fsd_polys = _extract_arrays(md.fsd_px, md.fsd_count, idx)
        self.sifor_od = [np.column_stack([p, md.od_py[oi][idx, :]])
                        for oi, p in enumerate(self.sifor_od_polys)]
        self.sifor_fsd = [np.column_stack([md.fsd_px[fi][idx, :], md.fsd_py[fi][idx, :]])
                         for fi in range(len(self.sifor_fsd_polys))]

        # FSD type / conf
        self.fsd_types = [int(md.fsd_type_data[idx, fi]) if md.fsd_type_data is not None else 0
                         for fi in range(len(self.sifor_fsd_polys))]
        self.fsd_confs = [int(md.fsd_conf_data[idx, fi]) if md.fsd_conf_data is not None else 0
                         for fi in range(len(self.sifor_fsd_polys))]

        # -- TargetPose (transform from global to ego-vehicle frame) --
        def _tpv(name):
            a = getattr(md, name, None)
            return float(a[idx]) if a is not None and idx < len(a) else np.nan
        tp_gx, tp_gy, tp_gyaw = _tpv('tp_x'), _tpv('tp_y'), _tpv('tp_yaw')
        if not any(np.isnan(v) for v in (tp_gx, tp_gy, tp_gyaw, self.ego_x, self.ego_y, self.ego_yaw)):
            dx = tp_gx - self.ego_x
            dy = tp_gy - self.ego_y
            eyaw = self.ego_yaw
            self.tp = (
                dx * np.cos(eyaw) + dy * np.sin(eyaw),
                -dx * np.sin(eyaw) + dy * np.cos(eyaw),
                float(np.arctan2(np.sin(tp_gyaw - eyaw), np.cos(tp_gyaw - eyaw)))
            )
        else:
            # Any NaN → return all NaNs so planner's early-return catches it
            self.tp = (np.nan, np.nan, np.nan)
        sp_gx, sp_gy, sp_gyaw = _tpv('sp_x'), _tpv('sp_y'), _tpv('sp_yaw')
        if not any(np.isnan(v) for v in (sp_gx, sp_gy, sp_gyaw, self.ego_x, self.ego_y, self.ego_yaw)):
            dx = sp_gx - self.ego_x
            dy = sp_gy - self.ego_y
            eyaw = self.ego_yaw
            self.sp = (
                dx * np.cos(eyaw) + dy * np.sin(eyaw),
                -dx * np.sin(eyaw) + dy * np.cos(eyaw),
                float(np.arctan2(np.sin(sp_gyaw - eyaw), np.cos(sp_gyaw - eyaw)))
            )
        else:
            self.sp = (np.nan, np.nan, np.nan)
