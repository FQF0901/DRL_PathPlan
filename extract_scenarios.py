#!/usr/bin/env python3
"""
Extract scenario frames from MF4 files based on planning trigger conditions.

Filters: selected parking slot type == 2 (horizontal/parallel) AND PrkgZone_u8 >= 8.
Saves rendered environment plots as PNG/PDF files.
"""

import json
import os
import argparse
import logging

import numpy as np

# Must be set before importing pyplot (headless rendering)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from asammdf import MDF

from mdf_reader import (
    get_vehicle_params, build_time_grid, MdfData, TimePointData,
    DATASET_DIR, PAR_FILE, find_channel, read_simple, nearest_interp_1d,
    get_selected_slot_type,
)
from viz_env import plot_frame, find_trigger_times

# Suppress asammdf verbose logs
logging.getLogger('asammdf').setLevel(logging.WARNING)


def sanitize_filename(basename):
    """Sanitize filename: replace # with _, remove .mf4 extension."""
    name = basename.replace('#', '_')
    if name.lower().endswith('.mf4'):
        name = name[:-4]
    return name


def has_required_signals(mdf):
    """Check if the MF4 file has the required signals for processing."""
    # Trigger signal
    trig_ch, _, _ = find_channel(mdf, 'HAS_Event_Request_Trajectory_Trigger')
    if trig_ch is None:
        return False, "missing trigger signal (HAS_Event_Request_Trajectory_Trigger)"

    # SIFOR1 FSD signals
    fsd_ch, _, _ = find_channel(mdf, 'SIFOR1_Valid_Free_Space_Boundary_Set')
    if fsd_ch is None:
        return False, "missing SIFOR1 FSD data"

    # Parking Slot signals
    ps_ch, _, _ = find_channel(mdf, 'Parking_Slot_Set_IRV')
    if ps_ch is None:
        return False, "missing Parking Slot data"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="Extract scenario frames from MF4 files based on planning trigger conditions."
    )
    parser.add_argument('--dataset-dir', default=DATASET_DIR,
                        help=f"Directory containing MF4 files (default: {DATASET_DIR})")
    parser.add_argument('--output-dir', default='./scenarios',
                        help="Output directory for rendered images (default: ./scenarios)")
    parser.add_argument('--par-file', default=PAR_FILE,
                        help=f"PAR calibration file path (default: {PAR_FILE})")
    parser.add_argument('--slot-type', default=0, type=int,
                        help="Selected slot type to filter (0=any, 2=horizontal/parallel)")
    parser.add_argument('--prkgzone-min', default=8, type=int,
                        help="Minimum PrkgZone value (default: 8)")
    parser.add_argument('--format', default='png', choices=['png', 'pdf'],
                        help='输出格式: png (位图) 或 pdf (矢量, 缩放不失真)')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    par_file = args.par_file
    slot_type_target = args.slot_type
    prkgzone_min = args.prkgzone_min

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load vehicle parameters
    print(f"[INFO] Loading vehicle params from: {par_file}")
    veh = get_vehicle_params(par_file)

    # Find all .mf4 files in dataset directory
    mf4_files = sorted([
        f for f in os.listdir(dataset_dir)
        if f.lower().endswith('.mf4')
    ])
    if not mf4_files:
        print(f"[ERROR] No .mf4 files found in {dataset_dir}")
        return

    print(f"[INFO] Found {len(mf4_files)} MF4 files in {dataset_dir}")

    total_saved = 0
    results_meta = []

    for mf4_name in mf4_files:
        mf4_path = os.path.join(dataset_dir, mf4_name)
        print(f"\n{'=' * 60}")
        print(f"[FILE] {mf4_name}")

        # Open MF4
        try:
            mdf = MDF(mf4_path)
        except Exception as e:
            print(f"  [SKIP] Cannot open MF4: {e}")
            continue

        # Check required signals
        ok, reason = has_required_signals(mdf)
        if not ok:
            print(f"  [SKIP] {reason}")
            mdf.close()
            continue

        print(f"  [OK] Required signals present")

        # Build time grid
        t_grid = build_time_grid(mdf, step=0.05)
        if len(t_grid) == 0:
            print("  [SKIP] Empty time grid")
            mdf.close()
            continue

        print(f"  Time grid: {t_grid[0]:.2f} ~ {t_grid[-1]:.2f} s ({len(t_grid)} pts)")

        # Read all signals via MdfData
        try:
            md = MdfData(mdf, t_grid, veh)
        except Exception as e:
            print(f"  [SKIP] Error reading MdfData: {e}")
            mdf.close()
            continue

        # Find trigger times
        if md.trigger is None:
            print("  [SKIP] No trigger signal data")
            mdf.close()
            continue

        trigger_times = find_trigger_times(md.trigger, md.t)
        if not trigger_times:
            print("  [SKIP] No trigger events found")
            mdf.close()
            continue

        print(f"  Found {len(trigger_times)} trigger event(s)")

        # Read signals for filtering
        ch_pz, pz_grp, pz_cidx = find_channel(mdf, 'Psi_PrkgInfo.PrkgZone_u8')
        if ch_pz is None:
            print("  [SKIP] Cannot find channel 'Psi_PrkgInfo.PrkgZone_u8'")
            mdf.close()
            continue
        
        pz_ts, pz_vals = read_simple(mdf, ch_pz, pz_grp, pz_cidx)
        if pz_ts is None:
            print("  [SKIP] Cannot read PrkgZone signal data")
            mdf.close()
            continue

        # Sanitized base name for output files
        base_name = sanitize_filename(mf4_name)
        file_saved = 0

        for i, trig_t in enumerate(trigger_times):
            # Get selected slot type
            st_result = get_selected_slot_type(mdf, trig_t)
            if st_result[0] is None:
                continue
            st_val = st_result[0]
            
            # PrkgZone
            pz_vals_grid = nearest_interp_1d(pz_ts, pz_vals, np.array([trig_t]))
            pz_val = pz_vals_grid[0]
            if pz_val is None or np.isnan(pz_val):
                continue
            
            # Apply filters
            if slot_type_target != 0 and int(st_val) != slot_type_target:
                continue
            if int(pz_val) < prkgzone_min:
                continue
            
            print(f"  [MATCH] t={trig_t:.2f}s  SlotType={int(st_val)}  PrkgZone={int(pz_val)}  "
                  f"source={st_result[1]}")

            # Find index in the interpolated time grid
            idx = int(np.argmin(np.abs(md.t - trig_t)))

            # Create TimePointData snapshot
            try:
                tpd = TimePointData(md, idx)
            except Exception as e:
                print(f"    [SKIP] Error creating TimePointData: {e}")
                continue

            # Render plot
            title = (f"{base_name} t={md.t[idx]:.2f}s  "
                     f"SlotType={int(st_val)}({st_result[1]})  PrkgZone={int(pz_val)}")
            fig, ax = plot_frame(tpd, veh, title)

            # Save plot
            out_name = f"{base_name}_t{md.t[idx]:.2f}.{args.format}"
            out_path = os.path.join(output_dir, out_name)
            fig.savefig(out_path, dpi=150, bbox_inches='tight', format=args.format)
            plt.close(fig)
            print(f"    -> Saved: {out_path}")
            file_saved += 1
            total_saved += 1
            results_meta.append({
                "file": mf4_name,
                "trigger_time": float(trig_t),
                "slot_type": int(st_val),
                "slot_type_source": st_result[1],
                "prkg_zone": int(pz_val),
                "ego_pose": {"x": tpd.ego_x, "y": tpd.ego_y, "yaw": tpd.ego_yaw},
                "tp_pose": {"x": tpd.tp[0], "y": tpd.tp[1], "yaw": tpd.tp[2]},
                "image": out_name,
            })

        if file_saved == 0:
            print(f"  [INFO] No matching scenarios "
                  f"(SlotType={slot_type_target}, PrkgZone>={prkgzone_min})")

        mdf.close()

    if total_saved > 0:
        meta_path = os.path.join(output_dir, 'scenarios.json')
        with open(meta_path, 'w') as f:
            json.dump(results_meta, f, indent=2)
        print(f"  Metadata saved to {meta_path}")

    print(f"\n{'=' * 60}")
    print(f"Saved {total_saved} scenarios to {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
