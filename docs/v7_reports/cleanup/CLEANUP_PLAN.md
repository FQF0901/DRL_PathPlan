# repo 清理清单（CLEANUP_PLAN）—— 只出清单，不删

> 生成：2026-10-06（锁版清理 A）；范围：`/workspace/01_Proj/DRL_PathPlan`（repo 内，排除 `.venv`）。
> 前置：`docs/v7_reports/` + `docs/v6_reports/` 已入仓（commit `8762a22`）；`/tmp/opencode` 已清理（759M → 1.2M，仅 KEEP/）。
> 方法：体积 = `du -sb` 逐目录；docs 引用 = 全 `docs/` 文本（含新入仓报告）搜索目录全名，命中不到再试去 `BTC<日期>-` 前缀短名（短名命中已逐项抽查，仅 2 项）。
> 原则：**被 docs 引用或属最佳产物链 → 保留**；未引用陈旧 → 建议删除；同批次未引用 → 复核。本文件仅为建议，删除前需人工确认。

## 0. 汇总

| 区域 | 体积 | 保留 | 建议删除 | 复核 |
|---|---|---|---|---|
| `runs/`（510 子目录） | 3.8G | 365 个（3.30G） | 105 个（**357.9M**） | 40 个（96.2M） |
| `datasets/`（24 目录 + 1 日志） | 1.8G | 16 项（1.35G） | 7 项（114.2M） | 2 项（295.3M） |
| `docs/`（26 文件 + reward_audit/） | ~9.5M | 15 | 0（10 项归档候选） | — |
| 缓存/杂项 | ~5M | — | 已直接清理（见 §D） | — |

**建议删除总量（保守口径）**：runs ≈ 357.9M + datasets ≈ 114.2M ≈ **472.1M**（不含复核项）。

## A. `runs/`（510 子目录，3.8G）

### A.1 建议保留 = 最佳产物链（任务指定）+ docs 引用

指定保留链（均在清单中逐项标记）：w1 线、s11 线、P4-extra、§14、§13 奖励臂、E-β″（v6 P3）、IDM 基线（eval500/clean500/tg45）、v6 P4 关键臂（arm0/arm5/seed11）、P3 四 seed、v7 结构线（已关闭，报告入仓）、`_refs_rlbase`（工具硬引用）、`_refs_oldgen`、`reward_audit*`/`reward_viz`。

> 注：v3/v4 代 `stage_c_*`/`p3_*` 等 runs 多数仍被 `docs/rl_stage_c_*` 引用 → 按"被引用"保留；若这些文档后续归档，可连带复核删除（其体积主要在 v3/v4 批）。

### A.2 建议删除（未引用；105 个，357.9M）

均为 09-27～10-02 的中间/重复 eval 与一次性实验目录（v3/v4 perf/smoke、p3 中间 eval、p4prep smoke、eval500 逐臂副本等）。完整逐项见 §A.4。

### A.3 复核（未引用但与保留项同批次；40 个，96.2M）

见 §A.4 中"复核"行；多为保留 run 的同期 eval 子目录，建议人工抽查后决定。

### A.4 完整清单（按体积降序）

| 路径 | 大小 | docs 引用 | 建议 | 理由 |
|---|---|---|---|---|
| `runs/reward_audit_ebeta2` | 426.5M | 全名 | 保留 | 最佳产物链：奖励审计产物（docs/reward_audit 引用） |
| `runs/reward_viz` | 203.4M | 全名 | 保留 | 最佳产物链：奖励可视化产物 |
| `runs/reward_audit` | 176.1M | 全名 | 保留 | 最佳产物链：奖励审计产物（docs/reward_audit 引用） |
| `runs/BTC20260928-1630_train` | 147.8M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-0941_v7p1b` | 128.5M | 全名 | 保留 | 最佳产物链：v7 P1-B 链 |
| `runs/BTC20261005-0856_v7struct_v5` | 125.4M | 全名 | 保留 | 最佳产物链：v7 结构线（已关闭；报告入仓） |
| `runs/BTC20261002-1837_v7p1i3` | 120.3M | 全名 | 保留 | 最佳产物链：v7 P1 iter3 |
| `runs/BTC20261001-1631_v6retrain` | 120.2M | 全名 | 保留 | 最佳产物链：v6 全量重训 |
| `runs/BTC20260929-1105_train` | 112.5M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260929-0425_fixA_nold` | 112.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260928-2201_ctrl_A20` | 58.3M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261005-0856_v7struct_v5_p3fix` | 58.1M | 全名 | 保留 | 最佳产物链：v7 结构线（已关闭；报告入仓） |
| `runs/BTC20261005-1732_v7p2_s1_arm1` | 46.4M | 全名 | 保留 | 最佳产物链：P3 种子分布 s1 |
| `runs/BTC20261005-1856_v7p2_s2_arm1` | 46.4M | 全名 | 保留 | 最佳产物链：P3 种子分布 s2 |
| `runs/BTC20261003-0641_v7p2_s0_arm1` | 46.4M | 全名 | 保留 | 最佳产物链：P3 种子分布 s0 |
| `runs/BTC20261005-0601_v7p2_s11_arm1` | 46.4M | 全名 | 保留 | 最佳产物链：s11（P2/P3 关键 seed；最佳成功读数） |
| `runs/BTC20261002-0303_p4_arm5` | 46.4M | 全名 | 保留 | 最佳产物链：v6 P4 臂（Gate4） |
| `runs/BTC20261002-0732_p4_arm0_seed11` | 46.4M | 全名 | 保留 | 最佳产物链：v6 P4 臂（Gate4） |
| `runs/BTC20261002-0206_p4_arm2` | 46.4M | 全名 | 保留 | 最佳产物链：v6 P4 臂（Gate4） |
| `runs/BTC20261002-0004_p4_arm0` | 46.4M | 全名 | 保留 | 最佳产物链：v6 P4 臂（Gate4） |
| `runs/BTC20261002-0026_p4_arm0` | 46.4M | 全名 | 保留 | 最佳产物链：v6 P4 臂（Gate4） |
| `runs/BTC20261002-0241_p4_arm4` | 46.4M | 全名 | 保留 | 最佳产物链：v6 P4 臂（Gate4） |
| `runs/BTC20261002-0148_p4_arm1` | 46.4M | 全名 | 保留 | 最佳产物链：v6 P4 臂（Gate4） |
| `runs/BTC20261002-0223_p4_arm3` | 46.4M | 全名 | 保留 | 最佳产物链：v6 P4 臂（Gate4） |
| `runs/BTC20261001-1619_v6mini` | 43.8M | 全名 | 保留 | 最佳产物链：v6 mini 闸 |
| `runs/BTC20260930-0543_stage_c_p4prep_a4ckpt` | 41.1M | — | 保留 | 最佳产物链：v6 P4 前置 |
| `runs/BTC20260930-0617_stage_c_p4prep_nospeed` | 41.1M | — | 保留 | 最佳产物链：v6 P4 前置 |
| `runs/BTC20261002-2329_v7p1dagger_w1` | 36.5M | 全名 | 保留 | 最佳产物链：w1 最佳 IL 线（P1 冠军 / P2-P3 base） |
| `runs/BTC20261002-2329_v7p1dagger_w2` | 36.5M | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261002-2329_v7p1dagger_w3` | 36.5M | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261002-2329_v7p1dagger_w4` | 36.5M | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261006-0233_v7reward_B_s0` | 25.7M | 全名 | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-0342_v7reward_B_s11` | 25.7M | 全名 | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-0445_v7reward_C_s0` | 25.7M | 全名 | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-0711_v7s14_s11_ttc` | 25.7M | 全名 | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261006-0615_v7s14_s0_ttc` | 25.7M | 全名 | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261005-2243_v7p4extra_s11_colls` | 25.7M | 全名 | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-0027_v7reward_A_s0` | 25.7M | 全名 | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-0127_v7reward_A_s11` | 25.7M | 全名 | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261005-2138_v7p4extra_s0_colls` | 25.7M | 全名 | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261001-0109_stage_c_v4_vanchor` | 23.1M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0350_stage_c_v41pa` | 23.1M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0443_stage_c_v41pk` | 23.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0243_stage_c_v4_vttc` | 23.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0131_stage_c_v4_vdata` | 23.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0012_stage_c_v4_v0` | 23.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0323_stage_c_v4_vboundary` | 23.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0043_stage_c_v4_v1` | 23.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0220_stage_c_v4_vlr` | 23.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1625_stage_c_v3_a1` | 22.8M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1725_stage_c_v3_sa` | 22.8M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1645_stage_c_v3_a2` | 22.8M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1816_diag_a2rep` | 22.8M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1608_stage_c_v3_p0` | 22.8M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1706_stage_c_v3_a3` | 22.8M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-0551_stage_c_p3_holdplan` | 22.8M | — | 保留 | docs 缩写引用（rl_stage_c_report：a4ckpt/holdplan/nospeed） |
| `runs/BTC20261005-0856_v7struct_v5_p3` | 19.4M | 全名 | 保留 | 最佳产物链：v7 结构线（已关闭；报告入仓） |
| `runs/BTC20261001-1631_v6p3` | 18.2M | 全名 | 保留 | 最佳产物链：E-β″ 线（v6 P3 采纳 / E3 零点） |
| `runs/BTC20260929-182033_stageB_phase3_exp_beta_anchor1` | 17.2M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260929-191846_stageB_phase3_exp_beta_anchor1_repro` | 17.2M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260929-1357_stageB_phase3_r5` | 17.2M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260929-1357_stageB_phase3_r4` | 17.2M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260929-1357_stageB_phase3_r3` | 17.2M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260929-1357_stageB_phase3_r1` | 17.2M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260929-1357_stageB_phase3_r2` | 17.2M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260929-185120_stageB_phase3_exp_ctrl_expert5k` | 17.2M | 全名 | 保留 | docs 引用（全名） |
| `runs/_refs_oldgen` | 15.7M | 全名 | 保留 | 最佳产物链：旧代参照（v6_cleanup_report 引用） |
| `runs/BTC20260930-0541_p4prep_c1_smoke` | 13.3M | — | 保留 | 最佳产物链：v6 P4 前置 |
| `runs/_refs_rlbase` | 9.6M | 全名 | 保留 | 最佳产物链：E-β′ 参照基线（tools/reward_audit.py 硬引用） |
| `runs/BTC20260930-0359_stage_c_p3_a3` | 5.7M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-0342_stage_c_p3_a2` | 5.7M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-0322_stage_c_p3_a1` | 5.7M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-0418_stage_c_p3_a4` | 5.7M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-0501_stage_c_p3_a6` | 5.7M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-0442_stage_c_p3_a5` | 5.7M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261001-2356_p4_smoke_u5_arm0` | 5.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-2137_v7p4extra_smoke_u2` | 5.0M | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-0614_v7s14_smoke_u2` | 5.0M | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261003-0638_v7p2_smoke_u2` | 5.0M | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-1731_v7p2_smoke_u2` | 5.0M | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261006-0026_v7reward_smoke_C_u2` | 5.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-0026_v7reward_smoke_B_u2` | 5.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261001-2356_p4_smoke_u2_arm5` | 5.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-2357_p4_smoke_u2_arm0_nocli` | 5.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-2355_p4_smoke_u2_arm0` | 5.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-2356_p4_smoke_u2_arm0` | 5.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0226_stage_c_smoke2_e1` | 4.6M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0221_stage_c_smoke2_curves` | 4.6M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0146_stage_c_smoke` | 4.5M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1601_stage_c_drift_smoke` | 4.5M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1602_stage_c_drift_e1_on` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1602_stage_c_drift_e1_off` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0218_stage_c_smoke2` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0554_p4prep_c4_smoke` | 4.4M | — | 保留 | 最佳产物链：v6 P4 前置 |
| `runs/BTC20260930-0300_stage_c_perf_v12` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0300_stage_c_perf_v11` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0300_stage_c_perf_v10` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0300_stage_c_perf_v9` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0300_stage_c_perf_v10b` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0300_stage_c_perf_v12_r256` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0300_stage_c_perf_v11_r256` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0300_stage_c_perf_v9_r256` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0300_stage_c_perf_v10_r256` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0152_stage_c_perf` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0152_stage_c_perf_r256` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0152_stage_c_perf_r512` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-0152_stage_c_perf_kl0` | 4.4M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-071840_v7p2_s11_eval500` | 1.6M | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-071745_v7p2_s11_u125_clean500` | 1.6M | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20260927-2202_eval500_phase1` | 1.1M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260929-095624_eval500_L2p1` | 1.1M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260929-100314_eval500_L2p2` | 1.1M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260927-2209_eval500_phase2` | 1.1M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260927-1839_eval500_baseline` | 1.1M | 全名 | 保留 | 最佳产物链：IDM eval500 基线（0.756） |
| `runs/BTC20261001-0350_eval500_v41pa_eval500` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0350_eval500_v41pa_full` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-163243_v7p3fix_best_eval500` | 1.0M | — | 保留 | 最佳产物链：v7 fix-11 补救（trunk_only） |
| `runs/BTC20261005-120728_v7sb_b_ckpt_epoch010_clean500` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261001-0243_eval500_v4vttc_eval500` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-161813_v7p3fix_best_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：v7 fix-11 补救（trunk_only） |
| `runs/BTC20261002-084501_v6p4_seed11_u175_clean500` | 1.0M | — | 保留 | docs 缩写引用（v6 seed11 复现/复检序列） |
| `runs/BTC20261005-121932_v7sb_b_ckpt_epoch015_clean500` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0131_eval500_v4vdata_eval500` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0243_eval500_v4vttc_full` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0131_eval500_v4vdata_full` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260929-190640_eval500_exp_ctrl_expert5k` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-214503_v7p1i3_screen_final` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P1 iter3 |
| `runs/BTC20260930-1435_eval500_l2_repeat` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-094238_v6p4_seed11_u200_eval500` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1439_eval500_clean_l2_repeat` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-212641_v7p1i3_screen_epoch010` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P1 iter3 |
| `runs/BTC20261003-052024_v7p1dagger_w3_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261003-073541_v7p2_s0_u050_clean500` | 1.0M | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-213524_v7p1i3_screen_epoch015` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P1 iter3 |
| `runs/BTC20260929-195151_eval500_phase3_base` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-0617_eval500_p4nospeed_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261003-051010_v7p1dagger_w2_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261002-083129_v6p4_seed11_u200_clean500` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261003-054238_v7p1dagger_w4_eval500` | 1.0M | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory` | 1.0M | 全名 | 保留 | 最佳产物链：w1 最佳 IL 线（P1 冠军 / P2-P3 base） |
| `runs/BTC20261001-200331_v6p3_e3_eval500` | 1.0M | 全名 | 保留 | 最佳产物链：E-β″ 线（v6 P3 采纳 / E3 零点） |
| `runs/BTC20261003-045912_v7p1dagger_w1_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：w1 最佳 IL 线（P1 冠军 / P2-P3 base） |
| `runs/BTC20261001-195447_v6p3_e3_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：E-β″ 线（v6 P3 采纳 / E3 零点） |
| `runs/BTC20261001-192937_v6p3_keepbest_ckpt_epoch005` | 1.0M | 全名 | 保留 | 最佳产物链：E-β″ 线（v6 P3 采纳 / E3 零点） |
| `runs/BTC20261001-193831_v6p3_keepbest_final` | 1.0M | 全名 | 保留 | 最佳产物链：E-β″ 线（v6 P3 采纳 / E3 零点） |
| `runs/BTC20261002-164209_v7p1b_eval500_sel` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P1-B 链 |
| `runs/BTC20261002-160948_v7p1b_eval500` | 1.0M | — | 保留 | 最佳产物链：v7 P1-B 链 |
| `runs/BTC20261001-194721_v6p3_diag_b_final_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：E-β″ 线（v6 P3 采纳 / E3 零点） |
| `runs/BTC20261003-053208_v7p1dagger_w4_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261005-195358_v7p2_s2_u050_clean500` | 1.0M | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20260929-225323_eval500_clean_l2` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-160206_v7p1b_clean500` | 1.0M | — | 保留 | 最佳产物链：v7 P1-B 链 |
| `runs/BTC20261002-0303_p4arm5_u175_clean500` | 1.0M | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20260930-1434_eval500_ebeta_repeat` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-162639_v7p1b_screen_epoch010` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P1-B 链 |
| `runs/BTC20261003-074819_v7p2_s0_eval500` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20260929-183716_eval500_exp_beta_anchor1` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260929-1357_eval500_phase3_r4` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260929-193529_eval500_exp_beta_anchor1_repro` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1441_eval500_ebeta_plan_ctrl` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261003-072348_v7p2_s0_u100_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-163338_v7p1b_screen_epoch015` | 1.0M | — | 保留 | 最佳产物链：v7 P1-B 链 |
| `runs/BTC20261005-140006_v7sb_final_eval500` | 1.0M | — | 复核 | 6 位时间戳出现在 docs（可能缩写引用） |
| `runs/BTC20261001-045536_diag_keepbest_v0_u075_clean500` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1437_eval500_clean_ebeta_repeat` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-0026_p4arm0_eval500` | 1.0M | 全名 | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20260929-1357_eval500_phase3_r3` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261002-215634_v7p1i3_eval500_sel` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P1 iter3 |
| `runs/BTC20260929-1357_eval500_phase3_r2` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261005-135523_v7sb_p3_final_clean500` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-134825_v7sb_p3_ckpt_epoch005_clean500` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261005-182951_v7p2_s1_u200_clean500` | 1.0M | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-184357_v7p2_s1_eval500` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-162002_v7p1b_screen_epoch005` | 1.0M | — | 保留 | 最佳产物链：v7 P1-B 链 |
| `runs/BTC20260930-0543_eval500_p3a4p_u200_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260929-1357_eval500_phase3_r1` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-0418_eval500_p3a4_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261005-181715_v7p2_s1_u050_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20260930-1449_eval500_clean_ebeta_plan_ctrl` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260929-224519_eval500_clean_ebeta` | 1.0M | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-0418_eval500_p3a4` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260929-1357_eval500_phase3_r5` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261002-211752_v7p1i3_screen_epoch005` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P1 iter3 |
| `runs/BTC20261002-0026_p4arm0_u200_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261005-200952_v7p2_s2_eval500` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-0026_p4arm0_u175_clean500` | 1.0M | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261001-050729_diag_keepbest_v41pa_u075_clean500` | 1.0M | — | 复核 | 6 位时间戳出现在 docs（可能缩写引用） |
| `runs/BTC20261006-043243_v7reward_B_s11_eval500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20260930-0442_eval500_p3a5` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261005-194156_v7p2_s2_u200_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-0303_p4arm5_eval500` | 1.0M | 全名 | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261005-074755_v7p2_s11_eval500_clean` | 1.0M | 全名 | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261006-022029_v7reward_A_s11_eval500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261002-0303_p4arm5_u200_clean500` | 1.0M | 短名 | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261006-032820_v7reward_B_s0_eval500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20260930-0551_eval500_p3holdplan_clean` | 1.0M | — | 保留 | docs 缩写引用（rl_stage_c_report：a4ckpt/holdplan/nospeed） |
| `runs/BTC20261006-075920_v7s14_s11_eval500` | 1.0M | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261006-011533_v7reward_A_s0_eval500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261005-233629_v7p4extra_s11_eval500` | 1.0M | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-040633_v7reward_B_s11_u025_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20260930-0442_eval500_p3a5_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261006-053907_v7reward_C_s0_eval500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20260930-0543_eval500_p3a4p_u025_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261005-232159_v7p4extra_s11_u075_clean500` | 1.0M | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-030017_v7reward_B_s0_u025_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261005-223034_v7p4extra_s0_eval500` | 1.0M | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261005-230803_v7p4extra_s11_u050_clean500` | 1.0M | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261005-073128_v7p2_s11_u150_clean500_clean` | 1.0M | 全名 | 保留 | 最佳产物链：s11 洁净复验（clean500/eval500/tg45） |
| `runs/BTC20261006-015301_v7reward_A_s11_u100_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-052250_v7reward_C_s0_u025_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-070046_v7s14_s0_eval500` | 1.0M | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261006-073318_v7s14_s11_u025_clean500` | 1.0M | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261005-070504_v7p2_s11_u150_clean500` | 1.0M | — | 保留 | 最佳产物链：s11 洁净复验（clean500/eval500/tg45） |
| `runs/BTC20261001-051833_diag_keepbest_vanchor_u100_clean500` | 1.0M | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261006-031432_v7reward_B_s0_u100_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20260930-0501_eval500_p3a6` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261006-010209_v7reward_A_s0_u100_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-005000_v7reward_A_s0_u025_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20260930-0543_eval500_p3a4p_u050_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261005-221530_v7p4extra_s0_u050_clean500` | 1.0M | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261005-220214_v7p4extra_s0_u075_clean500` | 1.0M | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-074550_v7s14_s11_u050_clean500` | 1.0M | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261006-050959_v7reward_C_s0_u075_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-041925_v7reward_B_s11_u100_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-020628_v7reward_A_s11_u050_clean500` | 1.0M | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20260930-0359_eval500_p3a3` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260930-0322_eval500_p3a1` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260930-0501_eval500_p3a6_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261006-063713_v7s14_s0_u075_clean500` | 1.0M | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20260930-0543_eval500_p3a4p_u100_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261006-064832_v7s14_s0_u050_clean500` | 1.0M | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20260930-0359_eval500_p3a3_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260930-0342_eval500_p3a2` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260930-0322_eval500_p3a1_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260930-0342_eval500_p3a2_clean` | 1.0M | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261002-100757_v7p0_idm_eval500_seedB` | 1.0M | 全名 | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261002-101123_v7p0_idm_baseline_eval500` | 1.0M | — | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261002-100221_v7p0_idm_eval500_rerun` | 1.0M | 全名 | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261002-102146_v7p0_idm_eval500_hash0_r2` | 1.0M | — | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261002-101954_v7p0_idm_eval500_hash0_r1` | 1.0M | 全名 | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261002-101320_v7p0_idm_eval500_rerun_clean` | 1.0M | 全名 | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261002-100606_v7p0_idm_eval500_seedA` | 1.0M | 全名 | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261002-100413_v7p0_idm_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261002-102339_v7p0_idm_clean500_hash0` | 1.0M | 全名 | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261002-101322_v7p0_idm_baseline_clean500` | 1.0M | — | 保留 | 最佳产物链：IDM 基线（现口径） |
| `runs/BTC20261001-201519_v6p3_ref_ebeta_clean500` | 1.0M | 全名 | 保留 | 最佳产物链：E-β″ 线（v6 P3 采纳 / E3 零点） |
| `runs/BTC20260930-1743_posthoc_v3sa_u100_sub150_rep` | 386K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-0026_p4arm0_u050_sub150` | 386K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261002-0303_p4arm5_u050_sub150` | 386K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261003-070745_v7p2_s0_u075_sub150` | 386K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-080015_v6p4_recheck_arm3_u200_sub150` | 386K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-0026_p4arm0_u100_sub150` | 385K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20260930-1743_posthoc_v3sa_u050_sub150_rep` | 385K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261005-192024_v7p2_s2_u025_sub150` | 385K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-115035_v7sb_b_ckpt_epoch010_sub150` | 385K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261001-042120_indist_v0_slice200_150` | 385K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-115424_v7sb_b_ckpt_epoch015_sub150` | 385K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0344_diag_v4post2_vboundary_u100_sub150` | 385K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1743_posthoc_v3sa_u025_sub150_rep` | 385K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-0026_p4arm0_u150_sub150` | 385K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261005-115843_v7sb_b_ckpt_epoch020_sub150` | 385K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0244_diag_v4post2_vlr_u100_sub150` | 385K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-120306_v7sb_b_final_sub150` | 385K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-081746_v6p4_seed11_u150_sub150` | 385K | — | 保留 | docs 缩写引用（v6 seed11 复现/复检序列） |
| `runs/BTC20261002-0241_p4arm4_u050_sub150` | 385K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261003-045300_v7p1dagger_w4_e010` | 384K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261002-080336_v6p4_recheck_arm4_u200_sub150` | 384K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261001-0043_eval500_v4v1_u025_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-0148_p4arm1_u050_sub150` | 384K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261002-0303_p4arm5_u100_sub150` | 384K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261003-071123_v7p2_s0_u125_sub150` | 384K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261001-0131_eval500_v4vdata_u100_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0350_eval500_v41pa_u100_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-082236_v6p4_seed11_u175_sub150` | 384K | — | 保留 | docs 缩写引用（v6 seed11 复现/复检序列） |
| `runs/BTC20261001-0220_eval500_v4vlr_u050_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0243_eval500_v4vttc_u025_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261003-005853_v7p1dagger_w1_e015` | 384K | 全名 | 保留 | 最佳产物链：w1 最佳 IL 线（P1 冠军 / P2-P3 base） |
| `runs/BTC20261001-0012_eval500_v4v0_u050_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0012_eval500_v4v0_u025_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-004759_diag_v4post_v0_u100_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-223731_v7p1probe_i3_clean150_repeat_action` | 384K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261002-0223_p4arm3_u050_sub150` | 384K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261001-0244_diag_v4post2_vlr_u075_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261003-033454_v7p1dagger_w3_e010` | 384K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261001-0323_eval500_v4vboundary_u050_sub150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-0026_p4arm0_u125_sub150` | 384K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261003-033755_v7p1dagger_w3_e015` | 384K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261002-0303_p4arm5_u150_sub150` | 384K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261001-042120_indist_vttc_slice200_150` | 384K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261003-005609_v7p1dagger_w1_e010` | 384K | 全名 | 保留 | 最佳产物链：w1 最佳 IL 线（P1 冠军 / P2-P3 base） |
| `runs/BTC20261003-033138_v7p1dagger_w3_e005` | 384K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261005-061935_v7p2_s11_u050_sub150` | 383K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261003-021410_v7p1dagger_w2_e010` | 383K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261003-065707_v7p2_s0_u050_sub150` | 383K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-0303_p4arm5_u125_sub150` | 383K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261002-082712_v6p4_seed11_u200_sub150` | 383K | — | 保留 | docs 缩写引用（v6 seed11 复现/复检序列） |
| `runs/BTC20261003-005306_v7p1dagger_w1_e005` | 383K | 全名 | 保留 | 最佳产物链：w1 最佳 IL 线（P1 冠军 / P2-P3 base） |
| `runs/BTC20261001-0243_eval500_v4vttc_u100_sub150` | 383K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-223344_v7p1probe_p1b_clean150_repeat_action` | 383K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261003-021700_v7p1dagger_w2_e015` | 383K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261001-0043_eval500_v4v1_u050_sub150` | 383K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-191614_v7p2_s2_u100_sub150` | 383K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261003-045608_v7p1dagger_w4_e015` | 383K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20260930-1706_eval500_v3a3_u50_sub150` | 383K | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20260930-1743_posthoc_v3a2_u075_sub150` | 383K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261003-070430_v7p2_s0_u025_sub150` | 383K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261001-0323_eval500_v4vboundary_u025_sub150` | 383K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1645_eval500_v3a2_u50_sub150` | 383K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1608_eval500_v3p0_u50_sub150` | 383K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-0303_p4arm5_u175_sub150` | 383K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261001-0220_eval500_v4vlr_u025_sub150` | 383K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261003-045010_v7p1dagger_w4_e005` | 383K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261005-191148_v7p2_s2_u050_sub150` | 383K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-223120_v7p1probe_p1b_clean150_plan` | 383K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261003-021124_v7p1dagger_w2_e005` | 383K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261001-010942_diag_v4post_v1_u075_sub150` | 383K | — | 复核 | 6 位时间戳出现在 docs（可能缩写引用） |
| `runs/BTC20261005-062635_v7p2_s11_u100_sub150` | 383K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20260930-1743_posthoc_v3sa_u075_sub150_rep` | 382K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261001-0131_eval500_v4vdata_u050_sub150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-0026_p4arm0_u075_sub150` | 382K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261002-0303_p4arm5_u075_sub150` | 382K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261005-193220_v7p2_s2_u150_sub150` | 382K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-064432_v7p2_s11_u125_sub150` | 382K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261001-0344_diag_v4post2_vboundary_u075_sub150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-003158_diag_rollback_a3_both_sub150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-004759_diag_v4post_v0_u075_sub150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261003-070038_v7p2_s0_u100_sub150` | 382K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-175535_v7p2_s1_u025_sub150` | 382K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-0026_p4arm0_u025_sub150` | 382K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261002-0303_p4arm5_u025_sub150` | 382K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261001-042120_indist_ebeta_slice200_150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-001834_diag_rollback_p0_all_sub150` | 382K | — | 复核 | 6 位时间戳出现在 docs（可能缩写引用） |
| `runs/BTC20261001-003555_diag_rollback_a3_all_sub150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-063319_v7p2_s11_u025_sub150` | 382K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261001-002702_diag_rollback_a1_all_sub150` | 382K | — | 复核 | 6 位时间戳出现在 docs（可能缩写引用） |
| `runs/BTC20260930-1604_eval500_v3base_sub150` | 382K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261001-0243_eval500_v4vttc_u050_sub150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-001518_diag_rollback_p0_both_sub150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-002349_diag_rollback_a1_both_sub150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-141117_v7sb_diag_p3_repeat150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-192829_v7p2_s2_u125_sub150` | 382K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20260930-1645_eval500_v3a2_u25_sub150` | 382K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1706_eval500_v3a3_u25_sub150` | 382K | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261002-0026_p4arm0_u200_sub150` | 382K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261003-071530_v7p2_s0_u150_sub150` | 382K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-114756_v7sb_b_ckpt_epoch005_sub150` | 382K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-180928_v7p2_s1_u175_sub150` | 382K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-174815_v7p2_s1_u050_sub150` | 382K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-0732_p4arm0s11_u050_sub150` | 381K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20261005-141152_v7sb_diag_b_repeat150` | 381K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-181322_v7p2_s1_u200_sub150` | 381K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261001-0131_eval500_v4vdata_u075_sub150` | 381K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1625_eval500_v3a1_u25_sub150` | 381K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-223438_v7p1probe_i3_clean150_plan` | 381K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261001-0350_eval500_v41pa_u075_sub150` | 381K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1743_posthoc_v3a1_u075_sub150` | 381K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261005-065550_v7p2_s11_u175_sub150` | 381K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-193810_v7p2_s2_u200_sub150` | 381K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-0026_p4arm0_u175_sub150` | 381K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20260930-1743_posthoc_v3sa_u075_sub150` | 381K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1725_eval500_v3sa_u50_sub150` | 381K | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261006-014500_v7reward_A_s11_u025_sub150` | 381K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20260930-1625_eval500_v3a1_u50_sub150` | 381K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1743_posthoc_v3a2_u100_sub150` | 381K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-0303_p4arm5_u200_sub150` | 381K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20260930-1743_posthoc_v3p0_u075_sub150` | 381K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261001-0350_eval500_v41pa_u050_sub150` | 381K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-063852_v7p2_s11_u075_sub150` | 381K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-175850_v7p2_s1_u075_sub150` | 381K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-225950_v7p4extra_s11_u025_sub150` | 380K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261005-134526_v7sb_p3_ckpt_epoch005_sub150` | 380K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-075618_v6p4_recheck_arm2_u200_sub150` | 380K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20260930-1743_posthoc_v3a3_u075_sub150` | 380K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261005-134656_v7sb_p3_final_sub150` | 380K | — | 复核 | 6 位时间戳出现在 docs（可能缩写引用） |
| `runs/BTC20260930-1608_eval500_v3p0_u25_sub150` | 380K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261002-0206_p4arm2_u050_sub150` | 380K | — | 保留 | 最佳产物链：v6 P4 终评/复检 |
| `runs/BTC20260930-1743_posthoc_v3a3_u100_sub150` | 380K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261005-175210_v7p2_s1_u100_sub150` | 380K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261006-035858_v7reward_B_s11_u025_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261001-010942_diag_v4post_v1_u100_sub150` | 380K | — | 复核 | 6 位时间戳出现在 docs（可能缩写引用） |
| `runs/BTC20261005-192452_v7p2_s2_u075_sub150` | 380K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-070025_v7p2_s11_u200_sub150` | 380K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20260930-1725_eval500_v3sa_u25_sub150` | 380K | — | 复核 | 与保留项同批次（可能是其 eval/子 run） |
| `runs/BTC20261006-014839_v7reward_A_s11_u075_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261005-065008_v7p2_s11_u150_sub150` | 380K | — | 保留 | 最佳产物链：s11 洁净复验（clean500/eval500/tg45） |
| `runs/BTC20261003-071904_v7p2_s0_u175_sub150` | 380K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261006-025138_v7reward_B_s0_u025_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261001-0243_eval500_v4vttc_u075_sub150` | 380K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0443_eval500_v41pk_u025_sub150` | 380K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20260930-1743_posthoc_ebeta_rep_sub150` | 380K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261005-230407_v7p4extra_s11_u075_sub150` | 380K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261001-0350_eval500_v41pa_u025_sub150` | 380K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-180232_v7p2_s1_u125_sub150` | 380K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-075202_v6p4_recheck_arm1_u200_sub150` | 380K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261006-062959_v7s14_s0_u025_sub150` | 380K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261005-225205_v7p4extra_s11_u050_sub150` | 380K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261005-215420_v7p4extra_s0_u025_sub150` | 380K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-062631_v7s14_s0_u100_sub150` | 380K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261006-024402_v7reward_B_s0_u050_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-025602_v7reward_B_s0_u075_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261001-0131_eval500_v4vdata_u025_sub150` | 380K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0137_diag_v4post2_vanchor_u100_sub150` | 380K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261006-014049_v7reward_A_s11_u100_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-040258_v7reward_B_s11_u075_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-003921_v7reward_A_s0_u100_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-035224_v7reward_B_s11_u050_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261005-214625_v7p4extra_s0_u050_sub150` | 380K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-035521_v7reward_B_s11_u100_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20260930-1743_posthoc_v3a1_u100_sub150` | 380K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261006-004648_v7reward_A_s0_u075_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-045743_v7reward_C_s0_u100_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-024755_v7reward_B_s0_u100_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-072547_v7s14_s11_u025_sub150` | 380K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261006-013655_v7reward_A_s11_u050_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261001-003027_diag_rollback_a3_policy_sub150` | 380K | — | 复核 | 6 位时间戳出现在 docs（可能缩写引用） |
| `runs/BTC20261005-180551_v7p2_s1_u150_sub150` | 380K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261001-0137_diag_v4post2_vanchor_u075_sub150` | 380K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261006-072303_v7s14_s11_u100_sub150` | 380K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261006-004302_v7reward_A_s0_u025_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-050601_v7reward_C_s0_u075_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261001-0109_eval500_v4vanchor_u025_sub150` | 380K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261006-071927_v7s14_s11_u050_sub150` | 380K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261006-072944_v7s14_s11_u075_sub150` | 380K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261005-225625_v7p4extra_s11_u100_sub150` | 380K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-063340_v7s14_s0_u075_sub150` | 380K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261006-050139_v7reward_C_s0_u025_sub150` | 380K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261006-003558_v7reward_A_s0_u050_sub150` | 379K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261005-215032_v7p4extra_s0_u100_sub150` | 379K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261005-215822_v7p4extra_s0_u075_sub150` | 379K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-062311_v7s14_s0_u050_sub150` | 379K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20260930-1743_posthoc_v3sa_u100_sub150` | 379K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261006-045357_v7reward_C_s0_u050_sub150` | 379K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261001-0443_eval500_v41pk_u050_sub150` | 379K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-0109_eval500_v4vanchor_u050_sub150` | 379K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-193658_v7p2_s2_u175_sub150` | 379K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261001-003051_diag_rollback_a3_experts_sub150` | 378K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-001451_diag_rollback_p0_experts_sub150` | 377K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-002224_diag_rollback_a1_experts_sub150` | 376K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261003-072227_v7p2_s0_u200_sub150` | 376K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20260930-1743_posthoc_v3p0_u100_sub150` | 374K | 全名 | 保留 | docs 引用（全名） |
| `runs/BTC20261001-002200_diag_rollback_a1_policy_sub150` | 374K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-001427_diag_rollback_p0_policy_sub150` | 373K | — | 复核 | 6 位时间戳出现在 docs（可能缩写引用） |
| `runs/BTC20261001-042120_indist_v41pa_dagger150` | 350K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-042120_indist_vdata_dagger150` | 349K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261001-042120_indist_ebeta_dagger150` | 347K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261002-091128_v7p1_t1_arm0u200_exact_sub135` | 299K | — | 保留 | 最佳产物链：v7 P1 其他 |
| `runs/BTC20261005-071821_v7p2_s11_tg45` | 180K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261001-2358_p4_smoke_eval30` | 148K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261005-163040_v7p3fix_best_tg45` | 131K | — | 保留 | 最佳产物链：v7 fix-11 补救（trunk_only） |
| `runs/BTC20261002-223004_v7p1probe_i3_tg45_repeat_action` | 131K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261003-053043_v7p1dagger_w3_tg45` | 131K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261005-184214_v7p2_s1_tg45` | 131K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-222857_v7p1probe_p1b_tg45_repeat_action` | 130K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261003-054109_v7p1dagger_w4_tg45` | 130K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261005-075959_v7p2_s11_tg45_clean` | 130K | 全名 | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261005-200818_v7p2_s2_tg45` | 130K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261002-215549_v7p1i3_tg45_sel` | 130K | 全名 | 保留 | 最佳产物链：v7 P1 iter3 |
| `runs/BTC20261002-222917_v7p1probe_i3_tg45_plan` | 130K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261003-051910_v7p1dagger_w2_tg45` | 130K | 全名 | 保留 | 最佳产物链：DAgger 四窗口（P1 训练数据） |
| `runs/BTC20261006-011357_v7reward_A_s0_tg45` | 130K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261003-074639_v7p2_s0_tg45` | 130K | — | 保留 | 最佳产物链：v7 P2/P3 其他 |
| `runs/BTC20261006-043100_v7reward_B_s11_tg45` | 130K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261003-050855_v7p1dagger_w1_tg45` | 130K | 全名 | 保留 | 最佳产物链：w1 最佳 IL 线（P1 冠军 / P2-P3 base） |
| `runs/BTC20261002-164134_v7p1b_tg45_sel` | 130K | 短名 | 保留 | 最佳产物链：v7 P1-B 链 |
| `runs/BTC20261002-222821_v7p1probe_p1b_tg45_plan` | 130K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261006-075739_v7s14_s11_tg45` | 130K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261005-140438_v7sb_final_tg45` | 130K | — | 建议删除 | 未引用陈旧 |
| `runs/BTC20261006-065914_v7s14_s0_tg45` | 130K | — | 保留 | 最佳产物链：§14 ttc 臂（s11 单点 PASS） |
| `runs/BTC20261002-160114_v7p1b_tg45` | 130K | 全名 | 保留 | 最佳产物链：v7 P1-B 链 |
| `runs/BTC20261006-032630_v7reward_B_s0_tg45` | 130K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261005-222851_v7p4extra_s0_tg45` | 130K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-021843_v7reward_A_s11_tg45` | 130K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261005-233442_v7p4extra_s11_tg45` | 129K | — | 保留 | 最佳产物链：P4-extra 碰撞抑制（安全改善版） |
| `runs/BTC20261006-053711_v7reward_C_s0_tg45` | 129K | — | 保留 | 最佳产物链：§13 奖励单变量臂 A/B/C |
| `runs/BTC20261002-160154_v7p1b_tg45_idm` | 129K | 全名 | 保留 | 最佳产物链：IDM tg45 基线 |
| `runs/BTC20261001-162637_v6mini_eval16` | 70K | 全名 | 保留 | 最佳产物链：v6 mini 闸 |
| `runs/BTC20261002-223041_v7p1probe_p1b_t3_repeat_action` | 61K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261002-223109_v7p1probe_i3_t3_repeat_action` | 61K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261002-223051_v7p1probe_i3_t3_plan` | 61K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261002-223027_v7p1probe_p1b_t3_plan` | 61K | 全名 | 保留 | 最佳产物链：v7 P1 探针 |
| `runs/BTC20261002-101824_v7p0_probe8_h0_r1` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-101847_v7p0_probe8_h0_r3` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-101859_v7p0_probe8_h0_r4` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102811_v7p0_probe8_w1_r3` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102839_v7p0_probe8_w1_r5` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-101836_v7p0_probe8_h0_r2` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-101910_v7p0_probe8_h0_r5` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-101802_v7p0_probe8_r5` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102743_v7p0_probe8_w1_r1` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102757_v7p0_probe8_w1_r2` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-101750_v7p0_probe8_r4` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-101727_v7p0_probe8_r2` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-101739_v7p0_probe8_r3` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102825_v7p0_probe8_w1_r4` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-101716_v7p0_probe8_r1` | 57K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-103011_v7p0_probe299_seeded_r4` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-103020_v7p0_probe299_seeded_r6` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102731_v7p0_probe299_w1_r2` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102728_v7p0_probe299_w1_r1` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102738_v7p0_probe299_w1_r5` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102958_v7p0_probe299_seeded_r1` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-103016_v7p0_probe299_seeded_r5` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-103006_v7p0_probe299_seeded_r3` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102733_v7p0_probe299_w1_r3` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-103002_v7p0_probe299_seeded_r2` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102741_v7p0_probe299_w1_r6` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261002-102736_v7p0_probe299_w1_r4` | 35K | — | 保留 | 最佳产物链：v7 P0 其他 |
| `runs/BTC20261005-1724_v7p2_s1_arm1` | 19K | 全名 | 保留 | 最佳产物链：P3 种子分布 s1 |
| `runs/BTC20261002-085952_v6p4_seed11_u200_eval500` | 9K | — | 复核 | v6 seed11 同批复评（docs 未逐字引用） |
| `runs/BTC20261002-091127_v6p4_seed11_u200_eval500` | 9K | — | 复核 | v6 seed11 同批复评（docs 未逐字引用） |

## B. `datasets/`（24 目录 + 1 日志，1.8G）

| 路径 | 大小 | docs 引用 | 建议 | 理由 |
|---|---|---|---|---|
| `datasets/BTC20261002-0941_expert5k_v4` | 286.3M | 全名 | 保留 | 当前最佳线复现（v4/v4.1 + DAgger 窗口） |
| `datasets/BTC20261002-0941_expert5k_v41` | 286.2M | 全名 | 保留 | 当前最佳线复现（v4/v4.1 + DAgger 窗口） |
| `datasets/BTC20261001-1327_expert5k` | 286.1M | 全名 | 保留 | docs 引用（全名） |
| `datasets/BTC20260926-2343_expert5k` | 283.3M | 全名 | 保留 | docs 引用（全名） |
| `datasets/BTC20261005-0814_expert5k_v5` | 275.3M | 全名 | 复核 | v5 线已关闭；仅历史文档引用（非复现依赖）→ 倾向删除（-297M） |
| `datasets/BTC20261002-0941_expert500val_v4` | 28.9M | 全名 | 保留 | 当前最佳线复现（v4/v4.1 + DAgger 窗口） |
| `datasets/BTC20261002-0941_expert500val_v41` | 28.9M | 全名 | 保留 | 当前最佳线复现（v4/v4.1 + DAgger 窗口） |
| `datasets/BTC20261001-1327_expert500val` | 28.9M | 全名 | 保留 | docs 引用（全名） |
| `datasets/BTC20260927-1734_expert500val` | 28.3M | 全名 | 保留 | docs 引用（全名） |
| `datasets/BTC20261005-0814_expert500val_v5` | 27.8M | — | 建议删除 | v5 线已关闭且未引用 |
| `datasets/BTC20261002-2329_v7p1dagger_w1` | 22.2M | 全名 | 保留 | 当前最佳线复现（v4/v4.1 + DAgger 窗口） |
| `datasets/BTC20261002-2329_v7p1dagger_w3` | 22.1M | 全名 | 保留 | 当前最佳线复现（v4/v4.1 + DAgger 窗口） |
| `datasets/BTC20261002-2329_v7p1dagger_w2` | 22.1M | 全名 | 保留 | 当前最佳线复现（v4/v4.1 + DAgger 窗口） |
| `datasets/BTC20261002-2329_v7p1dagger_w4` | 22.0M | 全名 | 保留 | 当前最佳线复现（v4/v4.1 + DAgger 窗口） |
| `datasets/BTC20260929-1357_phase3_dagger_r2` | 21.5M | — | 建议删除 | 未引用陈旧（v2/v3 DAgger 中间轮） |
| `datasets/BTC20260929-1357_phase3_dagger_r1` | 21.4M | — | 建议删除 | 未引用陈旧（v2/v3 DAgger 中间轮） |
| `datasets/BTC20260929-1357_phase3_dagger_r4` | 21.2M | — | 建议删除 | 未引用陈旧（v2/v3 DAgger 中间轮） |
| `datasets/BTC20260929-1357_phase3_dagger_r3` | 20.8M | — | 建议删除 | 未引用陈旧（v2/v3 DAgger 中间轮） |
| `datasets/BTC20260929-1357_phase3_dagger_r5` | 20.7M | 全名 | 保留 | v6 phase3 冻结池（prereg 显式接受；被引用） |
| `datasets/BTC20261005-0856_phase3_dagger_v5` | 20.0M | 全名 | 复核 | v5 线已关闭；仅历史文档引用（非复现依赖）→ 倾向删除（-297M） |
| `datasets/BTC20260928-1006_dagger_r1` | 5.7M | 全名 | 保留 | docs 引用（全名） |
| `datasets/BTC20260928-1109_dagger_r2` | 5.2M | 全名 | 保留 | docs 引用（全名） |
| `datasets/BTC20260928-1154_dagger_r3` | 5.0M | 全名 | 保留 | docs 引用（全名） |
| `datasets/BTC20260926-2343_expert5k.log` | 993K | — | 建议删除 | 生成日志（无引用） |
| `datasets/BTC20260929-1103_phase3_dagger_r1` | 669K | — | 建议删除 | 未引用陈旧（v2/v3 DAgger 中间轮） |

## C. `docs/`（陈旧/归档候选；只列不改）

| 路径 | 大小 | 建议 | 理由 |
|---|---|---|---|
| `docs/v7_program_prereg.md` | 66K | 保留 | v7 现行预注册（含 §12-§14） |
| `docs/v7_program_report.md` | 43K | 保留 | v7 现行主报告（本次已更新引用） |
| `docs/v6_program_prereg.md` | 30K | 保留 | v6 现行预注册（§7.4 pin 表等） |
| `docs/v6_program_report.md` | 23K | 保留 | v6 现行主报告（本次已更新引用） |
| `docs/version_ledger.md` | 7K | 保留 | 版本台账（被 v6 报告引用） |
| `docs/experiments.md` | 34K | 保留 | 实验总台账（被 v7/v6 prereg 引用；DAgger 事故更正） |
| `docs/design-v1.2.md` | 19K | 保留 | 设计基线 v1.2（后台长任务纪律） |
| `docs/metrics.md` | 9K | 保留 | TB/度量操作文档 |
| `docs/LOCKS.md` | 811B | 保留 | 锁纪律（操作） |
| `docs/dataset_stats.md` | 3K | 保留 | 数据集统计（操作参考；无引用） |
| `docs/v6_net_design.md` | 13K | 保留 | v6 网络设计（被 v6 报告/prereg 引用） |
| `docs/rl_reward_v5.md` | 15K | 保留 | 奖励 v5 规范（被 v6 报告/prereg 引用） |
| `docs/reward_audit/` | 7.7M | 保留 | 奖励审计定稿（7.7M；被 v6/v7 文档引用） |
| `docs/phase3_rootcause_analysis.md` | 31K | 保留 | P3 根因分析（被 v6 prereg/台账引用） |
| `docs/rl_stage_c_experiments.md` | 26K | 保留 | Stage C v3/v4 协议+结果（被 v7/v6 prereg 引用） |
| `docs/rl_stage_c_v4_report.md` | 8K | 保留/归档候选 | v4 代报告（被 v6 prereg 引用；v4 线已关） |
| `docs/rl_stage_c_report.md` | 11K | 归档候选 | v3 代报告（v3 线已关；仅 1 处历史引用） |
| `docs/rl_stage_c_v3_prereg.md` | 5K | 归档候选 | v3 代预注册（已关线） |
| `docs/rl_stage_c_v4_prereg.md` | 4K | 归档候选 | v4 代预注册（已关线；0 引用） |
| `docs/rl_stage_c_v4p1_prereg.md` | 3K | 归档候选 | v4.1 代预注册（已关线；0 引用） |
| `docs/rl_v3_evidence.sha256` | 10K | 归档候选 | v3 证据哈希清单（已关线） |
| `docs/p0-measurements.md` | 8K | 归档候选 | v3 期 P0 测量（已关线） |
| `docs/feasibility-analysis.md` | 39K | 归档候选 | 早期可行性分析（无引用） |
| `docs/forensics-2026-09-26.md` | 9K | 归档候选 | 早期取证（1 处历史引用） |
| `docs/db44fefe-system-review.md` | 31K | 归档候选 | 系统 review（2 处历史引用） |
| `docs/architecture-BTC20260925-2234.svg` | 70K | 归档候选 | 早期架构图（无引用） |

> 归档候选建议动作：移入 `docs/archive/`（或保持现状，仅标记）；删除前需确认无引用。

## D. 可直接清理类（缓存/杂项）

| 项 | 体积 | 状态 | 说明 |
|---|---|---|---|
| `__pycache__/` ×12（env/net/pipeline/tools/tests/reward_model 等） | 4.5M | **已清理**（本次） | 忽略文件，安全 |
| `*.pyc`（197 个，非 `.venv`） | — | **已清理**（本次） | 同上 |
| `.pytest_cache/` | 96K | **已清理**（本次） | 同上 |
| `datasets/BTC20260926-2343_expert5k.log` | 1000K | 建议删除 | expert5k 生成日志；无引用 |
| 未跟踪文件（`git status --porcelain`） | — | 无 | 本次清理后仍为干净（仅 docs 提交） |

## E. 注意事项

1. `runs/`、`datasets/` 均在 `.gitignore` 中（未跟踪），删除不影响 git 历史；但**最佳产物链 ckpt 是唯一副本**，删除前务必按 §A.1/§B 复核。
2. `runs/reward_audit_ebeta2`（443M）与 `runs/reward_audit`（182M）被 `docs/reward_audit/` 引用（定稿证据），且 `tools/reward_audit*.py` 硬引用 `_refs_rlbase`。
3. v7 结构线（`v7struct*`/`v7p3fix*`）虽"失败关闭"，其报告已入仓且被主报告引用，建议保留对应 runs 以便复算 §2/§3 数字。
4. 本清单的引用检查仅覆盖 `docs/`；`tools/`/`tests/` 硬引用已单独核对（`_refs_rlbase`）。

