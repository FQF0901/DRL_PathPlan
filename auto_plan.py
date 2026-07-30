#!/usr/bin/env python3
"""
自动规划流水线 (Auto Planning Pipeline)
=========================================
整合 extract_scenarios.py / path_planner.py / viz_env.py:
  1. 遍历数据集中的所有 MF4 文件
  2. 筛选 SlotType=2 && PrkgZone>=8 的触发时刻
  3. 调用 PathPlanner 规划路径
  4. 调用 verify_path 校验合规
  5. 调用 plot_frame 绘制环境 + 轨迹并保存 PNG

保留 viz_env.py 原有的交互式功能不变 (选择 MF4 并绘图).
"""

import os
import argparse
import warnings
import numpy as np

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from asammdf import MDF

from mdf_reader import (
    get_vehicle_params, build_time_grid, MdfData, TimePointData,
    PAR_FILE, DATASET_DIR, find_channel, read_simple, get_selected_slot_type,
)
from path_planner import PathPlanner, path_summary
from viz_env import plot_frame, find_trigger_times
from verify_path import verify_path, verify_report


# ============================================================
# 管线入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='自动规划流水线')
    parser.add_argument('--dataset-dir', default=DATASET_DIR,
                        help='MF4 数据集目录')
    parser.add_argument('--par-file', default=PAR_FILE,
                        help='PAR 标定文件')
    parser.add_argument('--output-dir', default='./auto_plan_out',
                        help='输出目录 (默认: ./auto_plan_out)')
    parser.add_argument('--slot-type', default=0, type=int,
                        help='筛选: 目标库位类型 (0=不限, 2=水平库位)')
    parser.add_argument('--prkgzone-min', default=8, type=int,
                        help='筛选: 最小 PrkgZone')
    parser.add_argument('--interactive', action='store_true',
                        help='以交互模式运行 (调用 viz_env.py 的主循环)')
    parser.add_argument('--format', default='png', choices=['png', 'pdf'],
                        help='输出格式: png (位图) 或 pdf (矢量, 缩放不失真)')
    args = parser.parse_args()

    if args.interactive:
        _run_interactive()
        return

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    par_file = args.par_file
    slot_type_target = args.slot_type
    prkgzone_min = args.prkgzone_min

    os.makedirs(output_dir, exist_ok=True)

    # 加载车辆参数
    print(f"[INFO] 车辆参数: {par_file}")
    veh = get_vehicle_params(par_file)
    planner = PathPlanner(veh)

    # 扫描 MF4 文件
    mf4_files = sorted([
        f for f in os.listdir(dataset_dir) if f.lower().endswith('.mf4')
    ])
    print(f"[INFO] 找到 {len(mf4_files)} 个 MF4 文件")

    total_planned = 0
    total_passed = 0

    for fn in mf4_files:
        fpath = os.path.join(dataset_dir, fn)
        try:
            mdf = MDF(fpath)
        except Exception:
            continue

        chs = list(mdf.channels_db)
        # 检查所需信号
        if not any('Trajectory_Trigger' in c for c in chs):
            mdf.close()
            continue
        if not any('SIFOR1_Valid_Free_Space' in c for c in chs):
            mdf.close()
            continue

        # 读取数据
        try:
            t_grid = build_time_grid(mdf)
            if len(t_grid) < 2:
                mdf.close(); continue
            md = MdfData(mdf, t_grid, veh)
        except Exception:
            mdf.close(); continue

        # 查找触发时刻
        trig_times = find_trigger_times(md.trigger, md.t)
        if not trig_times:
            mdf.close(); continue

        # 读取 PrkgZone 原始信号 (SlotType 通过 get_selected_slot_type 实时查询)
        try:
            pz_ch, pz_g, pz_c = find_channel(mdf, 'Psi_PrkgInfo.PrkgZone_u8')
            if pz_ch is None:
                mdf.close(); continue
            pz_ts, pz_v = read_simple(mdf, pz_ch, pz_g, pz_c)
            if pz_ts is None:
                mdf.close(); continue
        except Exception:
            mdf.close(); continue

        # 筛选匹配的触发
        matches = []
        for tt in trig_times:
            st_result = get_selected_slot_type(mdf, tt)
            if st_result[0] is None:
                continue
            st_val = st_result[0]
            pz_idx = int(np.argmin(np.abs(pz_ts - tt)))
            pz_val = int(pz_v[pz_idx])
            if (slot_type_target == 0 or st_val == slot_type_target) and pz_val >= prkgzone_min:
                matches.append((tt, pz_val, st_result[1]))

        mdf.close()

        if not matches:
            continue

        # 对每个匹配场景: 规划 → 校验 → 绘图 → 保存
        for tt, pz, st_src in matches:
            # 重新打开 MDF 构建 TimePointData
            mdf = MDF(fpath)
            t_grid = build_time_grid(mdf)
            md = MdfData(mdf, t_grid, veh)
            idx = int(np.argmin(np.abs(md.t - tt)))
            tpd = TimePointData(md, idx)
            mdf.close()

            # 规划
            path = planner.plan(tpd)
            tp = tpd.tp

            # 校验
            v_result = verify_path(path, tpd, planner.contour, planner.Rmin)
            v_report = verify_report(v_result)

            # 打印摘要
            safe_name = fn.replace('.mf4', '').replace('#', '_')
            status = '✅' if v_result['valid'] else '❌'
            print(f"  {status} {safe_name[:40]:40s} t={tt:.2f}s  "
                  f"tp=({tp[0]:.2f},{tp[1]:.2f},{np.degrees(tp[2]):.0f}°)  "
                  f"{v_result['n_maneuvers']}把 {v_result['total_length']:.2f}m")

            # 绘图
            title = f"{safe_name} t={tt:.2f}s Zone={pz} src={st_src}"
            if path.valid:
                fig, ax = plot_frame(tpd, veh, title, path)
            else:
                fig, ax = plot_frame(tpd, veh, title + " (NO-PATH)")

            out_name = f"{safe_name}_t{tt:.2f}.{args.format}"
            out_path = os.path.join(output_dir, out_name)
            fig.savefig(out_path, dpi=150, bbox_inches='tight', format=args.format)
            plt.close(fig)

            total_planned += 1
            if v_result['valid']:
                total_passed += 1

    print(f"\n{'='*60}")
    print(f"  规划完成: {total_planned} 个场景")
    print(f"  校验通过: {total_passed}/{total_planned}")
    print(f"  输出目录: {output_dir}/")
    print(f"{'='*60}")


def _run_interactive():
    """交互模式: 调用 viz_env.py 的 main 函数."""
    from viz_env import main as viz_main
    viz_main()


if __name__ == '__main__':
    main()
