#!/usr/bin/env python3
"""闭环取证结果分析（只读）：plan-vs-执行偏差、tracker 误差、失败形态。

用法::

    tools/venv-python tools/forensics_report.py --in runs/forensics/closed_lqr.json \
        --out runs/forensics/closed_lqr_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

FOCUS = ("curve", "roundabout", "uturn", "tollgate")


def _wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def plan_deviation(records, steps):
    """逐策略步：plan（自车系）→ 世界系参考，与实际执行轨迹比对。

    返回 per-decision 偏差与聚合统计（横向 e_lat、航向 e_psi、同时刻 xy 距离）。
    """
    from env.tracking import ego_to_world, interpolate

    by_step = {int(s["step"]): s for s in steps}
    per_decision = []
    for rec in records:
        if not rec.get("decision") or "plan" not in rec:
            continue
        s0 = int(rec["step"])
        ref = interpolate(np.asarray(rec["plan"], dtype=np.float64), dt=0.5, hz=10)
        ref_w = ego_to_world(ref, (rec["x"], rec["y"], rec["theta"]))
        realized, e_lat, e_psi, e_xy = [], [], [], []
        for i in range(5):  # 本 plan 覆盖的 5 个 env step（0.1 s 子步）
            step = by_step.get(s0 + i)
            if step is None:
                break
            p = np.array([step["x"], step["y"]], dtype=np.float64)
            realized.append([float(p[0]), float(p[1])])
            # 最近点横向/航向误差
            d = np.linalg.norm(ref_w[:, :2] - p, axis=1)
            j = int(np.argmin(d))
            th = float(ref_w[j, 2])
            left = np.array([-math.sin(th), math.cos(th)])
            e_lat.append(float(np.dot(p - ref_w[j, :2], left)))
            e_psi.append(float(_wrap_to_pi(float(step["theta"]) - th)))
            e_xy.append(float(np.linalg.norm(ref_w[i, :2] - p)))  # 同时刻索引对齐
        per_decision.append({
            "step": s0,
            "rc": rec.get("env_after", {}).get("rc"),
            "speed": rec.get("speed"),
            "mu_ds": rec.get("mu", [float("nan")])[0],
            "mu_dtheta": rec.get("mu", [float("nan"), float("nan")])[1],
            "plan_ds_mean": float(np.mean([a for a, _ in rec["plan"]])),
            "plan_dtheta_sum": float(np.sum([b for _, b in rec["plan"]])),
            "e_lat_max": float(np.max(np.abs(e_lat))) if e_lat else float("nan"),
            "e_lat_end": float(e_lat[-1]) if e_lat else float("nan"),
            "e_psi_max": float(np.max(np.abs(e_psi))) if e_psi else float("nan"),
            "e_psi_end": float(e_psi[-1]) if e_psi else float("nan"),
            "e_xy_end": float(e_xy[-1]) if e_xy else float("nan"),
            "realized": realized,
        })
    return per_decision


def episode_summary(ep):
    steps = ep.get("steps") or []
    records = ep.get("records") or []
    out = {"id": ep["id"], "geometry": ep["geometry"], "difficulty": ep["difficulty"],
           "termination": ep["outcome"].get("termination"), "rc_final": ep["outcome"].get("rc_final"),
           "steps": ep["outcome"].get("steps"), "speed_mean": ep["outcome"].get("speed_mean")}
    if not steps:
        return out
    last = steps[-1]
    speeds = [s["speed"] for s in steps]
    out.update({
        "rc_fail": last.get("rc"),
        "speed_final": last.get("speed"),
        "speed_p95": float(np.percentile(speeds, 95)),
        "block_at_fail": last.get("block"),
        "lane_lat_at_fail": last.get("lane_lat"),
        "max_abs_lane_lat": float(np.nanmax(np.abs([s.get("lane_lat") for s in steps if s.get("lane_lat") is not None] or [float("nan")]))),
        "block_seq": "".join(dict.fromkeys([str(s.get("block")) for s in steps])),
    })
    # 失败前 0.5 s 窗口的速度/横向加速度
    tail = steps[-5:]
    if len(tail) >= 2:
        v = np.array([s["speed"] for s in tail], dtype=np.float64)
        a_lon = np.diff(v) / 0.1
        out["a_lon_tail_mean"] = float(np.mean(a_lon))
        th = np.array([s["theta"] for s in tail], dtype=np.float64)
        dth = np.array([_wrap_to_pi(th[i + 1] - th[i]) for i in range(len(th) - 1)])
        out["a_lat_tail_p95"] = float(np.percentile(np.abs(0.5 * (v[1:] + v[:-1]) * dth / 0.1), 95))
        out["yaw_rate_tail"] = float(np.mean(np.abs(dth)) / 0.1)
    ey = [s.get("lqr_error_y") for s in steps if s.get("lqr_error_y") is not None]
    ep_ = [s.get("lqr_error_psi") for s in steps if s.get("lqr_error_psi") is not None]
    la = [s.get("lqr_lookahead_m") for s in steps if s.get("lqr_lookahead_m") is not None]
    if ey:
        out["tracker_e_y_abs_p95"] = float(np.percentile(np.abs(ey), 95))
        out["tracker_e_y_abs_max"] = float(np.max(np.abs(ey)))
        out["tracker_e_psi_abs_p95"] = float(np.percentile(np.abs(ep_), 95))
        out["tracker_e_psi_abs_max"] = float(np.max(np.abs(ep_)))
        out["lookahead_mean"] = float(np.mean(la)) if la else float("nan")
        out["lookahead_min"] = float(np.min(la)) if la else float("nan")
    # 出界/碰撞前 1 s 的 rc 轨迹
    out["rc_tail"] = [round(float(s.get("rc") or 0.0), 4) for s in steps[-12:]]
    out["block_tail"] = "".join([str(s.get("block")) for s in steps[-12:]])
    dev = plan_deviation(records, steps)
    if dev:
        out["plan_e_lat_max_mean"] = float(np.mean([d["e_lat_max"] for d in dev]))
        out["plan_e_psi_max_mean"] = float(np.mean([d["e_psi_max"] for d in dev]))
        out["plan_e_xy_end_mean"] = float(np.mean([d["e_xy_end"] for d in dev]))
        out["plan_dtheta_sum_mean"] = float(np.mean([d["plan_dtheta_sum"] for d in dev]))
        out["mu_dtheta_abs_mean"] = float(np.mean([abs(d["mu_dtheta"]) for d in dev]))
        out["_dev"] = dev
    return out


def _trace_theta_over(trace, t_idx, horizon_steps=30):
    """从 trace（list of steps，每步 0.1 s）取 t_idx 起 horizon 步的航向变化（rad，wrap）。"""
    if t_idx + horizon_steps >= len(trace):
        return float("nan")
    th0 = trace[t_idx]["theta"]
    th1 = trace[t_idx + horizon_steps]["theta"]
    return float(_wrap_to_pi(th1 - th0))


def _trace_steer_mean(trace, t_idx, horizon_steps=30):
    vals = [trace[i].get("steer") for i in range(t_idx, min(t_idx + horizon_steps, len(trace)))]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return float(np.mean(np.abs(vals))) if vals else float("nan")


def _trace_speed(trace, t_idx, horizon_steps=30):
    vals = [trace[i].get("speed") for i in range(t_idx, min(t_idx + horizon_steps, len(trace)))]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else float("nan")


def compare_route(docs: dict, rc_grid=None):
    """按 rc 对齐比较各模式：路线需求（baseline 实测航向变化） vs ckpt plan 航向变化。

    ``docs`` = {"baseline": doc, "lqr": doc, "exact": doc}；返回逐 spec/rc 桶的对比。
    """
    if rc_grid is None:
        rc_grid = [round(x, 2) for x in np.arange(0.1, 1.0, 0.1)]
    by_mode = {}
    for mode, doc in docs.items():
        by_mode[mode] = {int(ep["id"]): ep for ep in doc["episodes"]}

    rows = []
    for sid, base_ep in by_mode.get("baseline", {}).items():
        base_trace = base_ep.get("steps") or []
        if not base_trace:
            continue
        ck = by_mode.get("lqr", {}).get(sid)
        ex = by_mode.get("exact", {}).get(sid)
        for r in rc_grid:
            row = {"id": sid, "geometry": base_ep["geometry"], "rc": r}
            for mode, ep in (("baseline", base_ep), ("lqr", ck), ("exact", ex)):
                if ep is None:
                    continue
                trace = ep.get("steps") or []
                if not trace:
                    continue
                idx = int(np.argmin([abs((s.get("rc") or 0.0) - r) for s in trace]))
                if abs((trace[idx].get("rc") or 0.0) - r) > 0.05:
                    continue
                row[f"{mode}_dtheta_3s"] = _trace_theta_over(trace, idx)
                row[f"{mode}_speed"] = _trace_speed(trace, idx)
                row[f"{mode}_steer_abs_mean"] = _trace_steer_mean(trace, idx)
                row[f"{mode}_lane_curv"] = trace[idx].get("lane_curv")
                row[f"{mode}_block"] = trace[idx].get("block")
            # ckpt plan 航向变化（决策点最接近该 rc）
            for mode, ep in (("lqr", ck), ("exact", ex)):
                if ep is None:
                    continue
                recs = [x for x in (ep.get("records") or []) if x.get("decision") and "plan" in x]
                if not recs:
                    continue
                ridx = int(np.argmin([abs(((x.get("env_after") or {}).get("rc") or 0.0) - r) for x in recs]))
                rc_r = (recs[ridx].get("env_after") or {}).get("rc") or 0.0
                if abs(rc_r - r) > 0.05:
                    continue
                plan = recs[ridx]["plan"]
                row[f"{mode}_plan_dtheta_3s"] = float(np.sum([b for _, b in plan]))
                row[f"{mode}_plan_kappa"] = float(
                    np.mean([b for _, b in plan]) / max(np.mean([a for a, _ in plan]), 1e-6)
                )
            rows.append(row)
    return rows


def aggregate_compare(rows):
    """按 mode 汇总：plan/executed 航向变化与 baseline 的比率、速度、|steer|、lane_curv。"""
    out = {}
    for mode in ("lqr", "exact"):
        num = den = 0.0
        spd, steer, curv = [], [], []
        for row in rows:
            base = row.get("baseline_dtheta_3s")
            plan = row.get(f"{mode}_plan_dtheta_3s")
            if isinstance(base, (int, float)) and isinstance(plan, (int, float)) and abs(base) > 0.05:
                num += abs(plan) * abs(base)
                den += abs(base) ** 2
            for key, store in (("speed", spd), ("steer_abs_mean", steer), ("lane_curv", curv)):
                v = row.get(f"{mode}_{key}")
                if isinstance(v, (int, float)) and v == v:
                    store.append(v)
        base_spd = [r.get("baseline_speed") for r in rows if isinstance(r.get("baseline_speed"), (int, float))]
        base_steer = [r.get("baseline_steer_abs_mean") for r in rows if isinstance(r.get("baseline_steer_abs_mean"), (int, float))]
        base_curv = [r.get("baseline_lane_curv") for r in rows if isinstance(r.get("baseline_lane_curv"), (int, float))]
        out[mode] = {
            "n_rc_points": len(rows),
            "dtheta_gain_vs_baseline": (num / den) if den > 0 else float("nan"),
            "speed_mean": float(np.mean(spd)) if spd else float("nan"),
            "steer_abs_mean": float(np.mean(steer)) if steer else float("nan"),
            "lane_curv_mean": float(np.mean(curv)) if curv else float("nan"),
        }
        out.setdefault("baseline", {})
    out["baseline"] = {
        "speed_mean": float(np.mean(base_spd)) if base_spd else float("nan"),
        "steer_abs_mean": float(np.mean(base_steer)) if base_steer else float("nan"),
        "lane_curv_mean": float(np.mean(base_curv)) if base_curv else float("nan"),
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="forensics_report.py")
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--baseline-in", default=None)
    parser.add_argument("--exact-in", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    doc = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    summaries = [episode_summary(ep) for ep in doc["episodes"]]
    if args.baseline_in:
        docs = {"lqr": doc, "baseline": json.loads(Path(args.baseline_in).read_text(encoding="utf-8"))}
        if args.exact_in:
            docs["exact"] = json.loads(Path(args.exact_in).read_text(encoding="utf-8"))
        rows = compare_route(docs)
        table = [r for r in rows]
    else:
        rows = None
        table = []
    report = {"mode": doc.get("mode"), "tracker_json": doc.get("tracker_json"), "episodes": summaries}
    if rows is not None:
        report["compare_rows"] = rows
        report["compare_aggregate"] = aggregate_compare(rows)
    focus = [s for s in summaries if s["geometry"] in FOCUS]
    easy = [s for s in summaries if s["geometry"] == "straight"]

    def _agg(rows, key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    report["aggregate"] = {}
    for name, rows in (("focus", focus), ("easy", easy)):
        report["aggregate"][name] = {
            "n": len(rows),
            "speed_mean": _agg(rows, "speed_mean"),
            "speed_final": _agg(rows, "speed_final"),
            "tracker_e_y_abs_p95": _agg(rows, "tracker_e_y_abs_p95"),
            "tracker_e_psi_abs_p95": _agg(rows, "tracker_e_psi_abs_p95"),
            "lookahead_mean": _agg(rows, "lookahead_mean"),
            "plan_e_lat_max_mean": _agg(rows, "plan_e_lat_max_mean"),
            "plan_e_psi_max_mean": _agg(rows, "plan_e_psi_max_mean"),
            "plan_e_xy_end_mean": _agg(rows, "plan_e_xy_end_mean"),
            "plan_dtheta_sum_mean": _agg(rows, "plan_dtheta_sum_mean"),
        }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'id':>4} {'geometry':<11} {'term':<11} {'rc_fail':>7} {'steps':>5} "
          f"{'spd_mean':>8} {'spd_end':>7} {'ey95':>6} {'epsi95':>6} {'blk_fail':>8} {'blk_tail':<14} "
          f"{'dev_lat':>7} {'dev_psi':>7} {'dth_sum':>7}")
    for s in summaries:
        print(f"{s['id']:>4} {s['geometry']:<11} {str(s.get('termination')):<11} "
              f"{_fmt(s.get('rc_fail')):>7} {str(s.get('steps')):>5} {_fmt(s.get('speed_mean')):>8} "
              f"{_fmt(s.get('speed_final')):>7} {_fmt(s.get('tracker_e_y_abs_p95')):>6} "
              f"{_fmt(s.get('tracker_e_psi_abs_p95')):>6} {str(s.get('block_at_fail')):>8} "
              f"{str(s.get('block_tail')):<14} {_fmt(s.get('plan_e_lat_max_mean')):>7} "
              f"{_fmt(s.get('plan_e_psi_max_mean')):>7} {_fmt(s.get('plan_dtheta_sum_mean')):>7}")
    print()
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=1))
    if rows is not None:
        print()
        print(f"{'id':>4} {'geom':<11} {'rc':>4} {'blk':>3} | {'base dth3s':>10} {'lqr plan':>9} {'exact plan':>10} "
              f"| {'base v':>7} {'lqr v':>7} {'exact v':>7} | {'base|st|':>8} {'lqr|st|':>8} | {'base curv':>9} {'lqr curv':>9}")
        for r in table:
            print(f"{r['id']:>4} {r['geometry']:<11} {r['rc']:>4.2f} {str(r.get('baseline_block') or r.get('lqr_block')):>3} | "
                  f"{_fmt(r.get('baseline_dtheta_3s')):>10} {_fmt(r.get('lqr_plan_dtheta_3s')):>9} {_fmt(r.get('exact_plan_dtheta_3s')):>10} | "
                  f"{_fmt(r.get('baseline_speed'),2):>7} {_fmt(r.get('lqr_speed'),2):>7} {_fmt(r.get('exact_speed'),2):>7} | "
                  f"{_fmt(r.get('baseline_steer_abs_mean')):>8} {_fmt(r.get('lqr_steer_abs_mean')):>8} | "
                  f"{_fmt(r.get('baseline_lane_curv'),4):>9} {_fmt(r.get('lqr_lane_curv'),4):>9}")
        print()
        print("compare_aggregate:", json.dumps(report["compare_aggregate"], ensure_ascii=False))
    print(f"[forensics] → {out_path}")
    return 0


def _fmt(v, nd=3):
    if isinstance(v, (int, float)) and v == v:
        return f"{v:.{nd}f}"
    return "  n/a"


if __name__ == "__main__":
    raise SystemExit(main())
