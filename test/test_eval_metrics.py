import numpy as np
from pipeline.eval_metrics import bootstrap_ci, paired_bootstrap, summarize_rows, gate_report


def test_bootstrap_and_paired():
    x = np.arange(100, dtype=float)
    ci = bootstrap_ci(x, n_boot=200, seed=1)
    assert ci["n"] == 100 and ci["lo"] <= ci["mean"] <= ci["hi"]
    d = paired_bootstrap(x, x - 1.0, n_boot=200, seed=1)
    assert d["n"] == 100 and abs(d["mean_diff"] - 1.0) < 1e-9


def test_summary_and_gate():
    rows_a = [
        {"scene_seed": 1, "geometry": "straight", "success": True, "collision": False, "route_completion": 1.0},
        {"scene_seed": 2, "geometry": "curve", "success": False, "collision": True, "route_completion": 0.5},
    ]
    rows_b = [
        {"scene_seed": 1, "geometry": "straight", "success": True, "collision": False, "route_completion": 1.0},
        {"scene_seed": 2, "geometry": "curve", "success": True, "collision": False, "route_completion": 0.8},
    ]
    s = summarize_rows(rows_a)
    assert s["n"] == 2 and s["success_count"] == 1
    g = gate_report(rows_a, rows_b, seed=0)
    assert g["success"]["n_paired"] == 2
    assert abs(g["success"]["diff"]["mean_diff"] + 0.5) < 1e-9
