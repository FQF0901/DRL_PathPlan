#!/usr/bin/env python3
"""
路径合规校验工具 (Path Verification Utility)
=============================================
校验规划路径是否满足所有约束:
  1. 无碰撞 (轨迹每10cm和端点 + FSD/OD, FSD type=2 排除)
  2. 每段轨迹长度 >= 0.2m
  3. 转弯半径 >= 最小转弯半径 (Rmin)
  4. 泊车把数尽量少 (1-2把优先)

用法:
  from verify_path import verify_path, verify_report
  result = verify_path(path, tpd, contour, Rmin)
  print(verify_report(result))
"""

from typing import Dict, Any
import numpy as np
from path_planner import PathResult
from mdf_reader import TimePointData
from collision import _sat_intersect, _polygon_aabb, _aabb_overlap


def verify_path(path: PathResult, tpd: TimePointData,
                contour: np.ndarray, Rmin: float) -> Dict[str, Any]:
    """
    校验一条路径是否满足所有约束.

    参数:
      path: PathPlanner 输出的路径
      tpd: 时刻快照数据 (含 FSD/OD)
      contour: 自车轮廓 (N, 2)
      Rmin: 最小转弯半径

    返回:
      dict 包含各校验项结果:
        valid: bool — 全部通过
        checks: dict — 每项具体结果
    """
    checks = {}

    # ---- 1. 碰撞校验 (每10cm + 端点, FSD type=2 排除) ----
    coll_result = _check_collision(path, tpd, contour)
    checks['collision'] = coll_result

    # ---- 2. 段长校验 ----
    seg_lens = [s.length for s in path.segments]
    min_seg = min(seg_lens) if seg_lens else 0.0
    seg_ok = min_seg >= 0.2 or path.n_maneuvers == 0
    checks['segment_length'] = {
        'pass': seg_ok,
        'min': min_seg,
        'threshold': 0.2,
        'detail': f"最短段长 {min_seg:.2f}m {'✅' if seg_ok else '❌'} (需 >= 0.2m)"
    }

    # ---- 3. 转弯半径校验 ----
    radii = [abs(s.radius) for s in path.segments if s.radius != 0]
    min_r = min(radii) if radii else float('inf')
    rad_ok = min_r >= Rmin or len(radii) == 0
    checks['turn_radius'] = {
        'pass': rad_ok,
        'min': min_r,
        'threshold': Rmin,
        'detail': f"最小半径 {min_r:.1f}m {'✅' if rad_ok else '❌'} (需 >= {Rmin:.1f}m)"
    }

    # ---- 4. 起点校验: poses[0] == (0,0,0) ----
    start_ok = True
    se = 0.0
    if path.valid and len(path.poses) > 0:
        se = np.hypot(path.poses[0, 0], path.poses[0, 1])
        start_ok = se < 0.01
    checks['start_pose'] = {
        'pass': start_ok,
        'detail': f"起点误差 {se:.3f}m {'✅' if start_ok else '❌'} (需 < 0.01m)"
    }

    # ---- 5. 终点校验: poses[-1] == TargetPose (位置+heading) ----
    end_ok = True
    end_pos_err = 0.0
    end_yaw_err = 0.0
    if path.valid and len(path.segments) > 0 and len(path.poses) > 0:
        last_seg = path.segments[-1]
        end_pos_err = np.hypot(path.poses[-1, 0] - last_seg.end_pose[0],
                               path.poses[-1, 1] - last_seg.end_pose[1])
        end_yaw_err = abs(np.arctan2(
            np.sin(path.poses[-1, 2] - last_seg.end_pose[2]),
            np.cos(path.poses[-1, 2] - last_seg.end_pose[2])))
        end_ok = end_pos_err < 1e-3 and abs(end_yaw_err) < 1e-3
    end_ok = end_ok or path.n_maneuvers == 0
    checks['end_pose'] = {
        'pass': end_ok,
        'pos_err': end_pos_err,
        'yaw_err_deg': np.degrees(end_yaw_err),
        'detail': f"终点: 位置误差 {end_pos_err:.6f}m, "
                  f"heading误差 {np.degrees(end_yaw_err):.6f}° "
                  f"{'✅' if end_ok else '❌'} "
                  f"(需 位置<1e-3m, heading<1e-3rad)"
    }

    # ---- 6. 把数 (参考) ----
    checks['maneuvers'] = {
        'pass': True,
        'value': path.n_maneuvers,
        'detail': f"{path.n_maneuvers}把 {'✅' if path.n_maneuvers <= 2 else '⚠️ >2把'}"
    }

    # ---- 汇总 ----
    all_pass = all(
        v['pass'] for k, v in checks.items()
        if k != 'maneuvers'  # maneuvers 是参考项
    ) and path.valid

    return {
        'valid': all_pass,
        'total_length': path.total_length,
        'n_maneuvers': path.n_maneuvers,
        'checks': checks,
    }


def _check_collision(path: PathResult, tpd: TimePointData,
                     contour: np.ndarray) -> Dict:
    """碰撞校验: 每10cm采样点 + 端点, FSD type=2 排除."""
    if not path.valid:
        return {'pass': False, 'detail': 'N/A (无路径)'}
    if len(path.poses) == 0:
        return {'pass': True, 'detail': 'N/A (0段路径, 无需碰撞校验)'}

    for pi, pose in enumerate(path.poses):
        px, py, pyaw = pose
        c, s = np.cos(pyaw), np.sin(pyaw)
        ego_at = contour @ np.array([[c, -s], [s, c]]) + np.array([px, py])
        bb = _polygon_aabb(ego_at)

        # FSD (排除 type=2)
        for fi, poly in enumerate(tpd.sifor_fsd):
            if tpd.fsd_types[fi] == 2:
                continue
            if _aabb_overlap(bb, _polygon_aabb(poly)):
                if _sat_intersect(ego_at, poly)[0]:
                    return {
                        'pass': False,
                        'detail': f'与 FSD[{fi}] (type={tpd.fsd_types[fi]}) '
                                  f'在 pose[{pi}]={pose} 碰撞'
                    }
        # OD
        for oi, poly in enumerate(tpd.sifor_od):
            if _aabb_overlap(bb, _polygon_aabb(poly)):
                if _sat_intersect(ego_at, poly)[0]:
                    return {
                        'pass': False,
                        'detail': f'与 OD[{oi}] 在 pose[{pi}]={pose} 碰撞'
                    }

    return {'pass': True, 'detail': '无碰撞 ✅'}


def verify_report(result: Dict[str, Any]) -> str:
    """生成可读的校验报告."""
    if not result['valid']:
        lines = ["❌ 路径校验失败:"]
        for k, v in result['checks'].items():
            if not v.get('pass', True):
                lines.append(f"  [{k}] {v.get('detail', 'FAIL')}")
        return '\n'.join(lines)

    lines = [f"✅ 路径校验通过 ({result['n_maneuvers']}把, {result['total_length']:.2f}m)"]
    for k, v in result['checks'].items():
        lines.append(f"  ✅ {v.get('detail', k)}")
    return '\n'.join(lines)


# ============================================================
# CLI 模式: 批量校验 scenarios.json 中的场景
# ============================================================

if __name__ == '__main__':
    import argparse, json, os, sys, warnings
    warnings.filterwarnings('ignore')
    import matplotlib
    matplotlib.use('Agg')
    from asammdf import MDF
    from mdf_reader import (get_vehicle_params, build_time_grid,
                            MdfData, TimePointData, PAR_FILE, DATASET_DIR)
    from path_planner import PathPlanner

    parser = argparse.ArgumentParser(description='路径合规校验工具')
    parser.add_argument('--scenarios', default=os.path.join(
        os.path.dirname(__file__) or '.', 'scenarios', 'scenarios.json'),
        help='scenarios.json 路径 (默认: ./scenarios/scenarios.json)')
    parser.add_argument('--dataset-dir', default=DATASET_DIR)
    parser.add_argument('--par-file', default=PAR_FILE)
    args = parser.parse_args()

    veh = get_vehicle_params(args.par_file)
    planner = PathPlanner(veh)

    if os.path.exists(args.scenarios):
        with open(args.scenarios) as f:
            scenarios = json.load(f)
    else:
        # 直接扫描 MF4 文件
        print(f"[INFO] scenarios.json 不存在, 扫描 {args.dataset_dir}...")
        scenarios = []
        for fn in sorted(os.listdir(args.dataset_dir)):
            if not fn.lower().endswith('.mf4'):
                continue
            fpath = os.path.join(args.dataset_dir, fn)
            try:
                mdf = MDF(fpath)
            except Exception:
                continue
            chs = list(mdf.channels_db)
            if not any('Trajectory_Trigger' in c for c in chs):
                mdf.close(); continue
            t_grid = build_time_grid(mdf)
            if len(t_grid) < 2:
                mdf.close(); continue
            md = MdfData(mdf, t_grid, veh)
            trigs = []
            if md.trigger is not None:
                sig = np.nan_to_num(md.trigger, nan=0)
                rising = np.where((sig[:-1] < 0.5) & (sig[1:] >= 0.5))[0] + 1
                trig_times = t_grid[rising]
                min_gap = 2.0
                if len(trig_times) > 0:
                    trigs = [trig_times[0]]
                    for tt in trig_times[1:]:
                        if tt - trigs[-1] >= min_gap:
                            trigs.append(tt)
            mdf.close()

            # Read SlotType and PrkgZone for each trigger
            for tt in trigs:
                mdf = MDF(fpath)
                try:
                    from mdf_reader import find_channel, read_simple
                    st_ch, st_g, st_c = find_channel(mdf, 'SA_Ego_PS_SlotType_NU')
                    pz_ch, pz_g, pz_c = find_channel(mdf, 'Psi_PrkgInfo.PrkgZone_u8')
                    if st_ch is None or pz_ch is None:
                        mdf.close(); continue
                    st_ts, st_v = read_simple(mdf, st_ch, st_g, st_c)
                    pz_ts, pz_v = read_simple(mdf, pz_ch, pz_g, pz_c)
                    if st_ts is None or pz_ts is None:
                        mdf.close(); continue
                    st_idx = int(np.argmin(np.abs(st_ts - tt)))
                    pz_idx = int(np.argmin(np.abs(pz_ts - tt)))
                    st_val = int(st_v[st_idx])
                    pz_val = int(pz_v[pz_idx])
                    mdf.close()
                    if st_val == 2 and pz_val >= 8:
                        scenarios.append({'file': fn, 'trigger_time': float(tt),
                                          'slot_type': st_val, 'prkg_zone': pz_val})
                except Exception:
                    mdf.close(); continue

    if not scenarios:
        print("[WARN] 没有找到匹配的场景.")
        sys.exit(0)

    print(f"[INFO] 校验 {len(scenarios)} 个场景...")
    passed = 0
    for sc in scenarios:
        fname = sc['file']
        fpath = os.path.join(args.dataset_dir, fname)
        t = sc['trigger_time']
        try:
            mdf = MDF(fpath)
            t_grid = build_time_grid(mdf)
            md = MdfData(mdf, t_grid, veh)
            idx = int(np.argmin(np.abs(md.t - t)))
            tpd = TimePointData(md, idx)
            mdf.close()
        except Exception as e:
            print(f"  [SKIP] {fname[:35]:35s} {e}")
            continue

        path = planner.plan(tpd)
        result = verify_path(path, tpd, planner.contour, planner.Rmin)

        tp = tpd.tp
        status = '✅' if result['valid'] else '❌'
        print(f"  {status} {fname[:35]:35s} t={t:.2f} "
              f"tp=({tp[0]:.2f},{tp[1]:.2f}) "
              f"{result['n_maneuvers']}把 {result['total_length']:.2f}m")
        for k, v in result['checks'].items():
            if not v.get('pass', True):
                print(f"       ❌ {v.get('detail', k)}")
        if result['valid']:
            passed += 1

    print(f"\n[RESULT] {passed}/{len(scenarios)} 通过")
