# 锁版清理 C —— dry-run 清单（执行前快照，2026-10-06）

- 删除 runs 目录：63 项 / 772.9 MB
- 删除 datasets：4 项 / 46.3 MB
- 删除 ckpt：142 文件 / 902.6 MB
- 删除 log：4181 文件 / 3.06 MB
- 删除 monitor：468 文件 / 57.1 MB

## runs 删除项（按体积降序）

- `runs/BTC20260928-1630_train`（155.0 MB）refs=['docs/v6_reports/v6_cleanup_report.md', 'docs/version_ledger.md', 'tools/debug_rollout_viz.py']
- `runs/BTC20260929-1105_train`（117.9 MB）refs=['docs/v6_reports/v6_cleanup_report.md', 'docs/version_ledger.md']
- `runs/BTC20260928-2201_ctrl_A20`（61.1 MB）refs=['docs/v6_reports/v6_cleanup_report.md', 'docs/version_ledger.md']
- `runs/BTC20260930-0543_stage_c_p4prep_a4ckpt`（43.1 MB）refs=无 tracked 引用
- `runs/BTC20260930-0617_stage_c_p4prep_nospeed`（43.1 MB）refs=无 tracked 引用
- `runs/BTC20260930-1625_stage_c_v3_a1`（23.9 MB）refs=['docs/v6_reports/v6_cleanup_report.md']
- `runs/BTC20260930-1725_stage_c_v3_sa`（23.9 MB）refs=无 tracked 引用
- `runs/BTC20260930-1645_stage_c_v3_a2`（23.9 MB）refs=['docs/v6_reports/v6_cleanup_report.md']
- `runs/BTC20260930-1816_diag_a2rep`（23.9 MB）refs=无 tracked 引用
- `runs/BTC20260930-1608_stage_c_v3_p0`（23.9 MB）refs=['docs/v6_reports/v6_cleanup_report.md']
- `runs/BTC20260930-1706_stage_c_v3_a3`（23.9 MB）refs=无 tracked 引用
- `runs/BTC20260930-0551_stage_c_p3_holdplan`（23.9 MB）refs=无 tracked 引用
- `runs/BTC20260929-1357_stageB_phase3_r5`（18.0 MB）refs=无 tracked 引用
- `runs/BTC20260929-1357_stageB_phase3_r4`（18.0 MB）refs=无 tracked 引用
- `runs/BTC20260929-1357_stageB_phase3_r3`（18.0 MB）refs=无 tracked 引用
- `runs/BTC20260929-1357_stageB_phase3_r1`（18.0 MB）refs=无 tracked 引用
- `runs/BTC20260929-1357_stageB_phase3_r2`（18.0 MB）refs=无 tracked 引用
- `runs/BTC20260930-0541_p4prep_c1_smoke`（13.9 MB）refs=无 tracked 引用
- `runs/BTC20260930-0359_stage_c_p3_a3`（5.9 MB）refs=['docs/rl_stage_c_experiments.md']
- `runs/BTC20260930-0342_stage_c_p3_a2`（5.9 MB）refs=['docs/rl_stage_c_experiments.md']
- `runs/BTC20260930-0322_stage_c_p3_a1`（5.9 MB）refs=['docs/rl_stage_c_experiments.md']
- `runs/BTC20260930-0418_stage_c_p3_a4`（5.9 MB）refs=['docs/rl_stage_c_experiments.md']
- `runs/BTC20260930-0501_stage_c_p3_a6`（5.9 MB）refs=['docs/rl_stage_c_experiments.md']
- `runs/BTC20260930-0442_stage_c_p3_a5`（5.9 MB）refs=['docs/rl_stage_c_experiments.md']
- `runs/BTC20261005-2137_v7p4extra_smoke_u2`（5.2 MB）refs=无 tracked 引用
- `runs/BTC20261006-0614_v7s14_smoke_u2`（5.2 MB）refs=无 tracked 引用
- `runs/BTC20261003-0638_v7p2_smoke_u2`（5.2 MB）refs=无 tracked 引用
- `runs/BTC20261005-1731_v7p2_smoke_u2`（5.2 MB）refs=无 tracked 引用
- `runs/BTC20260930-0554_p4prep_c4_smoke`（4.6 MB）refs=无 tracked 引用
- `runs/BTC20260927-2202_eval500_phase1`（1.2 MB）refs=['docs/experiments.md']
- `runs/BTC20260927-2209_eval500_phase2`（1.2 MB）refs=['docs/experiments.md']
- `runs/BTC20260930-1434_eval500_ebeta_repeat`（1.1 MB）refs=无 tracked 引用
- `runs/BTC20260929-1357_eval500_phase3_r4`（1.1 MB）refs=['docs/phase3_rootcause_analysis.md']
- `runs/BTC20260930-1441_eval500_ebeta_plan_ctrl`（1.1 MB）refs=无 tracked 引用
- `runs/BTC20260930-1437_eval500_clean_ebeta_repeat`（1.1 MB）refs=无 tracked 引用
- `runs/BTC20260929-1357_eval500_phase3_r3`（1.1 MB）refs=['docs/phase3_rootcause_analysis.md']
- `runs/BTC20260929-1357_eval500_phase3_r2`（1.1 MB）refs=['docs/phase3_rootcause_analysis.md']
- `runs/BTC20260929-1357_eval500_phase3_r1`（1.1 MB）refs=['docs/phase3_rootcause_analysis.md']
- `runs/BTC20260930-1449_eval500_clean_ebeta_plan_ctrl`（1.1 MB）refs=无 tracked 引用
- `runs/BTC20260929-1357_eval500_phase3_r5`（1.1 MB）refs=['docs/phase3_rootcause_analysis.md']
- `runs/BTC20260930-1743_posthoc_v3sa_u100_sub150_rep`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3sa_u050_sub150_rep`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3sa_u025_sub150_rep`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3a2_u075_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1645_eval500_v3a2_u50_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1608_eval500_v3p0_u50_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3sa_u075_sub150_rep`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1604_eval500_v3base_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1645_eval500_v3a2_u25_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1625_eval500_v3a1_u25_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3a1_u075_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3sa_u075_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1625_eval500_v3a1_u50_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3a2_u100_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3p0_u075_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3a3_u075_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1608_eval500_v3p0_u25_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3a3_u100_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_ebeta_rep_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3a1_u100_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3sa_u100_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20260930-1743_posthoc_v3p0_u100_sub150`（0.4 MB）refs=无 tracked 引用
- `runs/BTC20261002-091128_v7p1_t1_arm0u200_exact_sub135`（0.3 MB）refs=无 tracked 引用

## datasets 删除项

- `datasets/BTC20260927-1734_expert500val`（29.7 MB）refs=['docs/experiments.md', 'docs/v7_reports/v7_p0_gap_decomposition.md']
- `datasets/BTC20260928-1006_dagger_r1`（5.9 MB）refs=['docs/experiments.md']
- `datasets/BTC20260928-1109_dagger_r2`（5.5 MB）refs=['docs/experiments.md']
- `datasets/BTC20260928-1154_dagger_r3`（5.2 MB）refs=['docs/experiments.md']

## ckpt / log / monitor 删除项

逐文件清单见 `CLEANUP_C_DELETED.txt`（ckpt 142 / log 4181 / monitor 468）。

