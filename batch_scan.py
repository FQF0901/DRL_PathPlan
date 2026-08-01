#!/usr/bin/env python3
"""
Batch scan: run the path planner over every parking scenario in the P417 dataset.

Per file:
  1. Detect scenarios (fallback chain):
     a. "trigger"     : rising edges of HAS_Event_Request_Trajectory_Trigger
                        (reuses viz_env.find_trigger_times), kept only when
                        Psi_PrkgInfo.PrkgZone_u8 >= 8 (extract_scenarios filter).
                        No slot-type filter. Multiple triggers per file allowed.
     b. "tp_fallback" : if no trigger scenario passes, use the first sample where
                        the raw target pose is valid (|x|>1e-6 && |y|>1e-6) AND
                        PrkgZone >= 8 — the "target active in parking zone" moment.
     c. "tp_forced"   : if neither exists, force a scenario at sample idx 0.
  2. For each scenario: TimePointData(md, idx) snapshot + plan_path(tpd, veh).
  3. Write results JSON (structure matches .slim/deepwork/batch_scan_results.json)
     and print a per-file summary table.
"""

import argparse
import json
import logging
import multiprocessing
import os
import re
import sys
import time
from functools import partial

import numpy as np

import mdf_cache
from mdf_reader import (
    get_vehicle_params, TimePointData,
    DATASET_DIR, PAR_FILE,
)
from viz_env import find_trigger_times
from path_planner import plan_path, _arc_center

logging.getLogger('asammdf').setLevel(logging.WARNING)

PRKGZONE_MIN = 8
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '.slim', 'deepwork', 'batch_scan_results.json')


def log_num_from_file(fname):
    """Extract trailing log number, e.g. '...#CANape_log_060.mf4' -> '060'."""
    m = re.search(r'log_(\d+)\.mf4$', fname)
    return m.group(1) if m else ''


def tp_valid(x, y):
    """Raw TP valid = not NaN and not the (0,0) cleared-target sentinel."""
    return not (np.isnan(x) or np.isnan(y) or (abs(x) < 1e-6 and abs(y) < 1e-6))


def strategy_label(plan):
    """Derive strategy label from a PathResult (mirrors planner preference order)."""
    if not plan.valid:
        return 'no_path'
    if plan.n_maneuvers == 0:
        return 'already_at_target'
    if plan.n_maneuvers == 1:
        return 'straight' if abs(plan.segments[0].radius) < 1e-6 else 'single_arc'
    if plan.n_maneuvers == 2:
        return 'line_arc' if abs(plan.segments[0].radius) < 1e-6 else 'two_seg'
    if plan.n_maneuvers == 3:
        return 'three_seg'
    return f'{plan.n_maneuvers}_seg'


def detect_scenarios(md, t_grid):
    """
    Return list of (time, idx, source, pz) scenario anchors.
    Fallback chain: trigger -> tp_fallback -> tp_forced (see module docstring).
    pz = PrkgZone value at the anchor time (None if signal unavailable).
    """
    # PrkgZone signal (cached by mdf_cache; None if signal unavailable)
    pz_grid = getattr(md, '_pz_grid', None)

    def pz_ok(idx):
        if pz_grid is None:
            return True
        v = pz_grid[idx]
        return not (v is None or np.isnan(v)) and int(v) >= PRKGZONE_MIN

    def pz_at(idx):
        if pz_grid is None:
            return None
        v = pz_grid[idx]
        return None if (v is None or np.isnan(v)) else int(v)

    # 1. trigger rising edges (extract_scenarios/viz_env logic), PrkgZone filter
    scenarios = []
    for t in find_trigger_times(md.trigger, md.t):
        idx = int(np.argmin(np.abs(md.t - t)))
        if pz_ok(idx):
            scenarios.append((float(t), idx, 'trigger', pz_at(idx)))
    if scenarios:
        return scenarios

    # 2. tp_fallback: first sample with valid target pose inside parking zone
    if md.tp_x is not None:
        ok = (np.abs(md.tp_x) > 1e-6) & (np.abs(md.tp_y) > 1e-6)
        if pz_grid is not None:
            ok = ok & (np.nan_to_num(pz_grid, nan=0) >= PRKGZONE_MIN)
        idxs = np.where(ok)[0]
        if len(idxs):
            i = int(idxs[0])
            return [(float(md.t[i]), i, 'tp_fallback', pz_at(i))]

    # 3. tp_forced: no usable signal -> force scenario at first sample
    return [(float(md.t[0]), 0, 'tp_forced', pz_at(0))]


def process_file(mf4_name, veh, use_cache=True):
    """Load one MF4 (via mdf_cache), detect scenarios, plan each scenario.

    Returns (fentry, trigs): fentry is the per-file summary dict (errors kept in
    fentry["errors"]), trigs the list of per-scenario entries (same structure as
    before). No prints here — caller prints per-file lines.
    """
    mf4_path = os.path.join(DATASET_DIR, mf4_name)
    fentry = {"file": mf4_name, "log_num": log_num_from_file(mf4_name),
              "extraction_ok": False, "n_scenarios": 0,
              "plan_ok": 0, "plan_fail": 0, "errors": []}

    try:
        md, err = mdf_cache.load_md(mf4_path, veh, use_cache=use_cache)
    except Exception as e:
        fentry["errors"].append(f"open: {e}")
        return fentry, []
    if md is None:
        fentry["errors"].append(err)
        return fentry, []
    t_grid = md.t

    fentry["extraction_ok"] = True

    scenarios = detect_scenarios(md, t_grid)
    fentry["n_scenarios"] = len(scenarios)

    trigs = []
    for t, idx, source, pz in scenarios:
        entry = {"file": mf4_name, "log_num": fentry["log_num"],
                 "time": float(t), "idx": int(idx), "source": source,
                 "slot_type": -1, "pz": pz, "tp_valid": False,
                 "excluded": False, "quality_gate_rejected": False,
                 "valid": False, "strategy_label": "no_path",
                 "n_segments": 0, "total_length": 0.0, "errors": [],
                 "ego": None, "target": None}
        tpd = None
        try:
            tpd = TimePointData(md, idx)
            plan = plan_path(tpd, veh)
            entry["valid"] = bool(plan.valid)
            entry["strategy_label"] = strategy_label(plan)
            entry["n_segments"] = int(plan.n_maneuvers)
            entry["total_length"] = float(plan.total_length)
            entry["quality_gate_rejected"] = bool(plan.rejected_by_gate)
            entry["ego"] = [float(tpd.ego_x), float(tpd.ego_y), float(tpd.ego_yaw)]
            tp = tpd.tp
            entry["target"] = [float(tp[0]), float(tp[1]), float(tp[2])]
            entry["tp_valid"] = bool(not np.isnan(tp[0]))
            st_grid = getattr(md, '_slot_type_grid', None)
            if st_grid is not None and not np.isnan(st_grid[idx]):
                entry["slot_type"] = int(st_grid[idx])
            if entry["valid"] and len(plan.poses) > 0:
                # 端点精度: 末端采样位姿 vs 目标
                px, py, pyaw = plan.poses[-1]
                entry["endpoint_err"] = float(np.hypot(px - tp[0], py - tp[1]))
                entry["endpoint_herr"] = float(np.arctan2(
                    np.sin(pyaw - tp[2]), np.cos(pyaw - tp[2])))
                # 残差指标: 各段几何重构末端 vs 声明 end_pose.
                # 弧段: C=_arc_center(start_pose, R), 重构点 = C + R·(sin(θ+da), −cos(θ+da)),
                #   da = (length/|R|)·sign(direction=='forward'?+1:−1)
                # 直段: 残差 = | |end−start| − length |
                resids = []
                for s in plan.segments:
                    if abs(s.radius) < 1e-6:
                        resids.append(abs(
                            np.hypot(s.end_pose[0]-s.start_pose[0],
                                     s.end_pose[1]-s.start_pose[1]) - s.length))
                    else:
                        cx, cy = _arc_center(s.start_pose, s.radius)
                        # direction 编码 sign(da·R): da 真符号 = sign(R)·(forward?+1:−1)
                        da = np.sign(s.radius) * (
                            1.0 if s.direction == 'forward' else -1.0) * (
                            s.length / abs(s.radius))
                        th = s.start_pose[2]
                        rx = cx + s.radius * np.sin(th + da)
                        ry = cy - s.radius * np.cos(th + da)
                        resids.append(float(np.hypot(
                            s.end_pose[0] - rx, s.end_pose[1] - ry)))
                entry["max_residual_m"] = float(max(resids)) if resids else 0.0
                entry["exact_tangent"] = bool(entry["max_residual_m"] < 1e-6)
        except Exception as e:
            entry["errors"].append(str(e))
        if tpd is not None:
            # EPE 未初始化 (如 #061: tp_forced 且 idx 0 且 ego=(0,0)) → 排除
            entry["excluded"] = bool(
                source == 'tp_forced' and idx == 0 and
                abs(tpd.ego_x) < 1e-6 and abs(tpd.ego_y) < 1e-6)
        if entry["valid"]:
            fentry["plan_ok"] += 1
        else:
            fentry["plan_fail"] += 1
        trigs.append(entry)

    return fentry, trigs


def main():
    t_start = time.time()
    parser = argparse.ArgumentParser(description='Batch scan over P417 dataset')
    parser.add_argument('--workers', type=int, default=4,
                        help='parallel workers (default 4, capped at cpu count)')
    parser.add_argument('--no-cache', action='store_true',
                        help='force re-parsing of all MF4 signals (rebuild cache)')
    args = parser.parse_args()
    workers = max(1, min(args.workers, 4, os.cpu_count() or 1))
    use_cache = not args.no_cache

    veh = get_vehicle_params(PAR_FILE)

    mf4_files = sorted(f for f in os.listdir(DATASET_DIR) if f.lower().endswith('.mf4'))
    print(f"[INFO] Found {len(mf4_files)} MF4 files in {DATASET_DIR}")
    print(f"[INFO] workers={workers}  cache={'on' if use_cache else 'off'}")

    files_out, triggers_out = [], []
    n_files_with_scenarios = 0
    n_files_extraction_ok = 0

    with multiprocessing.Pool(workers, maxtasksperchild=1) as pool:
        worker = partial(process_file, veh=veh, use_cache=use_cache)
        for fentry, trigs in pool.imap(worker, mf4_files):
            files_out.append(fentry)
            triggers_out.extend(trigs)
            if fentry["extraction_ok"]:
                n_files_extraction_ok += 1
            if fentry["n_scenarios"] > 0:
                n_files_with_scenarios += 1
            print(f"  {fentry['log_num']:>4s}  n_scenarios={fentry['n_scenarios']:2d}  "
                  f"plan_ok={fentry['plan_ok']:2d}  plan_fail={fentry['plan_fail']:2d}")

    n_plan_ok = sum(f["plan_ok"] for f in files_out)
    n_plan_fail = sum(f["plan_fail"] for f in files_out)
    n_total = n_plan_ok + n_plan_fail
    n_excluded = sum(1 for t in triggers_out if t.get("excluded"))
    n_scenarios_valid = n_total - n_excluded
    n_gate_rejected = sum(1 for t in triggers_out if t.get("quality_gate_rejected"))
    detours = []
    for t in triggers_out:
        if t["valid"] and t["total_length"] > 0 and t.get("target"):
            tx, ty = t["target"][0], t["target"][1]
            dist = np.hypot(tx, ty)
            if dist > 0.05:
                detours.append(t["total_length"] / dist)
    strat_counter = {}
    for t in triggers_out:
        strat_counter[t["strategy_label"]] = strat_counter.get(t["strategy_label"], 0) + 1
    result = {
        "n_files": len(files_out),
        "n_files_with_scenarios": n_files_with_scenarios,
        "n_files_extraction_ok": n_files_extraction_ok,
        "n_scenarios_total": n_total,
        "n_scenarios_valid": n_scenarios_valid,
        "n_excluded": n_excluded,
        "n_plan_ok": n_plan_ok,
        "n_plan_fail": n_plan_fail,
        "success_rate_pct": round(100.0 * n_plan_ok / n_total, 1) if n_total else 0.0,
        "success_rate_valid_pct": round(100.0 * n_plan_ok / n_scenarios_valid, 1) if n_scenarios_valid else 0.0,
        "n_quality_gate_rejected": n_gate_rejected,
        "detour_ratio_max": round(max(detours), 2) if detours else 0.0,
        "detour_ratio_mean": round(float(np.mean(detours)), 2) if detours else 0.0,
        "strategy_distribution": strat_counter,
        "files": files_out,
        "triggers": triggers_out,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] Results written to {OUT_JSON}")
    print(f"[INFO] n_files={result['n_files']}  n_scenarios={n_total}  "
          f"plan_ok={n_plan_ok}  plan_fail={n_plan_fail}  "
          f"success_rate={result['success_rate_pct']}%  "
          f"valid_scope={n_scenarios_valid}  "
          f"success_rate_valid={result['success_rate_valid_pct']}%  "
          f"excluded={n_excluded}  gate_rejected={n_gate_rejected}")
    print(f"[INFO] detour ratio (valid paths): max={result['detour_ratio_max']}  "
          f"mean={result['detour_ratio_mean']}")
    ep_errs = [t["endpoint_err"] for t in triggers_out
               if t.get("valid") and "endpoint_err" in t]
    ep_herrs = [abs(t["endpoint_herr"]) for t in triggers_out
                if t.get("valid") and "endpoint_herr" in t]
    if ep_errs:
        print(f"[INFO] endpoint err (valid paths): max={max(ep_errs):.5f} m  "
              f"mean={float(np.mean(ep_errs)):.5f} m  "
              f"max heading diff={max(ep_herrs):.5f} rad")
    resids = [t["max_residual_m"] for t in triggers_out
              if t.get("valid") and "max_residual_m" in t]
    n_exact = sum(1 for t in triggers_out
                  if t.get("valid") and t.get("exact_tangent"))
    if resids:
        print(f"[INFO] residual (valid paths): max={max(resids):.6f} m  "
              f"exact_tangent(<1e-6)={n_exact}/{len(resids)}")
    print(f"[INFO] strategy distribution: {strat_counter}")
    print(f"[INFO] total wall time: {time.time()-t_start:.1f}s")
    fails = [t for t in triggers_out if not t["valid"] and not t.get("excluded")]
    if fails:
        print(f"[INFO] failures ({len(fails)}):")
        for t in fails:
            print(f"  #{t['log_num']} t={t['time']:.2f} idx={t['idx']} src={t['source']} "
                  f"ego={t['ego']} target={t['target']} gate={t['quality_gate_rejected']}")


if __name__ == '__main__':
    main()
