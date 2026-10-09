# V8 清理-2：datasets/ 与 runs/ 删除执行记录（DONE，保守范围）

> 执行：2026-10-09 18:38–18:39（CST）｜依据：`docs/cleanup/V8_CLEANUP_data_runs_WILL_DELETE.md`（用户批准保守范围：datasets 拟删 16 项 + runs Tier-1 拟删 170 项）
> 授权边界（严格执行）：仅上述 186 项；**Tier-2（135 项）、KEEP 列表、任何 `20261009-*`、`.slim/`、`config/model.yaml`、`opencode_mobile.md` 均未触碰**；未 commit。
> 执行方式：清单内逐项 `rm -rf '<目录>'`（**无通配符**），执行前逐项核对（存在性/非符号链接/近 2h 无写入/不在 KEEP/前缀正确）。

## 0. 结果摘要

| 区域 | 计划 | 成功 | 跳过 | 实际释放 |
|---|---|---|---|---|
| `datasets/` | 16 | **16** | 0 | 1,215,024,324 B（1.1G） |
| `runs/` Tier-1 | 170 | **170** | 0 | 553,217,401 B（527.6M） |
| **合计** | **186** | **186** | **0** | **1,768,241,725 B（1.6G ≈ 1.65 GiB）** |

体积基线（`du -sb`）：

| 区域 | 执行前（18:38:51） | 执行后（18:39:13） | du 差 |
|---|---|---|---|
| `datasets/` | 2,142,518,377 B | 927,494,053 B | −1,215,024,324 B |
| `runs/` | 3,201,407,481 B | 2,648,190,080 B | −553,217,401 B |
| 合计 | 5,343,925,858 B | 3,575,684,133 B | **−1,768,241,725 B** |

> 校验：逐项 `du -sb` 求和 = **1,768,241,725 B**，与 `du` 前后差**逐字节一致**；186/186 目录删除后确认不存在。

## 1. 执行前核对（186/186 通过，0 跳过）

- 目录名与清单完全一致（从 WILL_DELETE 的 A/B 段命令块机械提取：16 + 170），且均属于盘点清单；
- 与 KEEP 集无交集：datasets 3 KEEP + `v41`（人工确认）全排除；runs 的 `fixA_nold`、`v6p3`（硬引用移 KEEP）、`v7p1dagger_w1`、`v7p2_s11_arm1` 全排除；
- 全部为真实目录（非符号链接）；路径前缀正确、无通配符；
- `find <dir> -newermt '-2 hours' -type f` → **186/186 为空**（无近期写入）；
- 结果：无需跳过任何项，无异常。

## 2. 逐项结果

### 2.1 `datasets/`（16 项 / 1.1G）

| # | 路径 | 体积 | 结果 |
|---|---|---|---|
| 1 | `datasets/BTC20260929-1357_phase3_dagger_r1` | 21.4M | ✅ 成功 |
| 2 | `datasets/BTC20260929-1357_phase3_dagger_r2` | 21.5M | ✅ 成功 |
| 3 | `datasets/BTC20260929-1357_phase3_dagger_r3` | 20.8M | ✅ 成功 |
| 4 | `datasets/BTC20260929-1357_phase3_dagger_r4` | 21.2M | ✅ 成功 |
| 5 | `datasets/BTC20260929-1357_phase3_dagger_r5` | 20.7M | ✅ 成功 |
| 6 | `datasets/BTC20261001-1327_expert500val` | 28.9M | ✅ 成功 |
| 7 | `datasets/BTC20261001-1327_expert5k` | 286.1M | ✅ 成功 |
| 8 | `datasets/BTC20261002-0941_expert500val_v4` | 28.9M | ✅ 成功 |
| 9 | `datasets/BTC20261002-0941_expert500val_v41` | 28.9M | ✅ 成功 |
| 10 | `datasets/BTC20261002-0941_expert5k_v4` | 286.3M | ✅ 成功 |
| 11 | `datasets/BTC20261002-2329_v7p1dagger_w1` | 22.2M | ✅ 成功 |
| 12 | `datasets/BTC20261002-2329_v7p1dagger_w2` | 22.1M | ✅ 成功 |
| 13 | `datasets/BTC20261002-2329_v7p1dagger_w3` | 22.1M | ✅ 成功 |
| 14 | `datasets/BTC20261002-2329_v7p1dagger_w4` | 22.0M | ✅ 成功 |
| 15 | `datasets/BTC20261007-1838_expert500val_v8` | 27.9M | ✅ 成功 |
| 16 | `datasets/BTC20261007-1838_expert5k_v8` | 277.9M | ✅ 成功 |

### 2.2 `runs/` Tier-1（170 项 / 527.6M）

| # | 路径 | 体积 | 结果 |
|---|---|---|---|
| 1 | `runs/BTC20260927-1839_eval500_baseline` | 999.4K | ✅ 成功 |
| 2 | `runs/BTC20260929-095624_eval500_L2p1` | 1011.5K | ✅ 成功 |
| 3 | `runs/BTC20260929-100314_eval500_L2p2` | 1011.2K | ✅ 成功 |
| 4 | `runs/BTC20260929-182033_stageB_phase3_exp_beta_anchor1` | 8.6M | ✅ 成功 |
| 5 | `runs/BTC20260929-183716_eval500_exp_beta_anchor1` | 1008.3K | ✅ 成功 |
| 6 | `runs/BTC20260929-185120_stageB_phase3_exp_ctrl_expert5k` | 8.6M | ✅ 成功 |
| 7 | `runs/BTC20260929-190640_eval500_exp_ctrl_expert5k` | 1013.3K | ✅ 成功 |
| 8 | `runs/BTC20260929-191846_stageB_phase3_exp_beta_anchor1_repro` | 8.6M | ✅ 成功 |
| 9 | `runs/BTC20260929-193529_eval500_exp_beta_anchor1_repro` | 1008.2K | ✅ 成功 |
| 10 | `runs/BTC20260929-195151_eval500_phase3_base` | 1011.5K | ✅ 成功 |
| 11 | `runs/BTC20260929-224519_eval500_clean_ebeta` | 1005.7K | ✅ 成功 |
| 12 | `runs/BTC20260929-225323_eval500_clean_l2` | 1009.0K | ✅ 成功 |
| 13 | `runs/BTC20260930-0551_eval500_p3holdplan_clean` | 1001.9K | ✅ 成功 |
| 14 | `runs/BTC20261001-1619_v6mini` | 43.7M | ✅ 成功 |
| 15 | `runs/BTC20261001-162637_v6mini_eval16` | 52.3K | ✅ 成功 |
| 16 | `runs/BTC20261001-1631_v6retrain` | 47.6M | ✅ 成功 |
| 17 | `runs/BTC20261001-192937_v6p3_keepbest_ckpt_epoch005` | 1009.7K | ✅ 成功 |
| 18 | `runs/BTC20261001-193831_v6p3_keepbest_final` | 1009.7K | ✅ 成功 |
| 19 | `runs/BTC20261001-194721_v6p3_diag_b_final_clean500` | 1009.4K | ✅ 成功 |
| 20 | `runs/BTC20261001-195447_v6p3_e3_clean500` | 1009.8K | ✅ 成功 |
| 21 | `runs/BTC20261001-200331_v6p3_e3_eval500` | 1010.3K | ✅ 成功 |
| 22 | `runs/BTC20261001-201519_v6p3_ref_ebeta_clean500` | 989.5K | ✅ 成功 |
| 23 | `runs/BTC20261002-0004_p4_arm0` | 9.9M | ✅ 成功 |
| 24 | `runs/BTC20261002-0026_p4_arm0` | 14.9M | ✅ 成功 |
| 25 | `runs/BTC20261002-0026_p4arm0_eval500` | 1006.6K | ✅ 成功 |
| 26 | `runs/BTC20261002-0026_p4arm0_u025_sub150` | 344.2K | ✅ 成功 |
| 27 | `runs/BTC20261002-0026_p4arm0_u050_sub150` | 348.2K | ✅ 成功 |
| 28 | `runs/BTC20261002-0026_p4arm0_u075_sub150` | 344.5K | ✅ 成功 |
| 29 | `runs/BTC20261002-0026_p4arm0_u100_sub150` | 347.5K | ✅ 成功 |
| 30 | `runs/BTC20261002-0026_p4arm0_u125_sub150` | 345.8K | ✅ 成功 |
| 31 | `runs/BTC20261002-0026_p4arm0_u150_sub150` | 346.8K | ✅ 成功 |
| 32 | `runs/BTC20261002-0026_p4arm0_u175_clean500` | 1003.6K | ✅ 成功 |
| 33 | `runs/BTC20261002-0026_p4arm0_u175_sub150` | 342.8K | ✅ 成功 |
| 34 | `runs/BTC20261002-0026_p4arm0_u200_clean500` | 1004.9K | ✅ 成功 |
| 35 | `runs/BTC20261002-0026_p4arm0_u200_sub150` | 343.7K | ✅ 成功 |
| 36 | `runs/BTC20261002-0148_p4_arm1` | 9.9M | ✅ 成功 |
| 37 | `runs/BTC20261002-0148_p4arm1_u050_sub150` | 346.5K | ✅ 成功 |
| 38 | `runs/BTC20261002-0206_p4_arm2` | 9.9M | ✅ 成功 |
| 39 | `runs/BTC20261002-0206_p4arm2_u050_sub150` | 342.5K | ✅ 成功 |
| 40 | `runs/BTC20261002-0223_p4_arm3` | 9.9M | ✅ 成功 |
| 41 | `runs/BTC20261002-0223_p4arm3_u050_sub150` | 346.0K | ✅ 成功 |
| 42 | `runs/BTC20261002-0241_p4_arm4` | 9.9M | ✅ 成功 |
| 43 | `runs/BTC20261002-0241_p4arm4_u050_sub150` | 346.6K | ✅ 成功 |
| 44 | `runs/BTC20261002-0303_p4_arm5` | 14.9M | ✅ 成功 |
| 45 | `runs/BTC20261002-0303_p4arm5_eval500` | 1002.8K | ✅ 成功 |
| 46 | `runs/BTC20261002-0303_p4arm5_u025_sub150` | 344.2K | ✅ 成功 |
| 47 | `runs/BTC20261002-0303_p4arm5_u050_sub150` | 348.2K | ✅ 成功 |
| 48 | `runs/BTC20261002-0303_p4arm5_u075_sub150` | 344.5K | ✅ 成功 |
| 49 | `runs/BTC20261002-0303_p4arm5_u100_sub150` | 346.5K | ✅ 成功 |
| 50 | `runs/BTC20261002-0303_p4arm5_u125_sub150` | 345.4K | ✅ 成功 |
| 51 | `runs/BTC20261002-0303_p4arm5_u150_sub150` | 345.7K | ✅ 成功 |
| 52 | `runs/BTC20261002-0303_p4arm5_u175_clean500` | 1009.0K | ✅ 成功 |
| 53 | `runs/BTC20261002-0303_p4arm5_u175_sub150` | 344.8K | ✅ 成功 |
| 54 | `runs/BTC20261002-0303_p4arm5_u200_clean500` | 1002.1K | ✅ 成功 |
| 55 | `runs/BTC20261002-0303_p4arm5_u200_sub150` | 342.7K | ✅ 成功 |
| 56 | `runs/BTC20261002-0732_p4_arm0_seed11` | 9.9M | ✅ 成功 |
| 57 | `runs/BTC20261002-0732_p4arm0s11_u050_sub150` | 343.5K | ✅ 成功 |
| 58 | `runs/BTC20261002-075202_v6p4_recheck_arm1_u200_sub150` | 342.1K | ✅ 成功 |
| 59 | `runs/BTC20261002-075618_v6p4_recheck_arm2_u200_sub150` | 342.5K | ✅ 成功 |
| 60 | `runs/BTC20261002-080015_v6p4_recheck_arm3_u200_sub150` | 347.7K | ✅ 成功 |
| 61 | `runs/BTC20261002-080336_v6p4_recheck_arm4_u200_sub150` | 346.5K | ✅ 成功 |
| 62 | `runs/BTC20261002-081746_v6p4_seed11_u150_sub150` | 346.6K | ✅ 成功 |
| 63 | `runs/BTC20261002-082236_v6p4_seed11_u175_sub150` | 346.3K | ✅ 成功 |
| 64 | `runs/BTC20261002-082712_v6p4_seed11_u200_sub150` | 345.3K | ✅ 成功 |
| 65 | `runs/BTC20261002-083129_v6p4_seed11_u200_clean500` | 1011.1K | ✅ 成功 |
| 66 | `runs/BTC20261002-084501_v6p4_seed11_u175_clean500` | 1015.3K | ✅ 成功 |
| 67 | `runs/BTC20261002-0941_v7p1b` | 55.9M | ✅ 成功 |
| 68 | `runs/BTC20261002-100221_v7p0_idm_eval500_rerun` | 999.4K | ✅ 成功 |
| 69 | `runs/BTC20261002-100413_v7p0_idm_clean500` | 998.0K | ✅ 成功 |
| 70 | `runs/BTC20261002-100606_v7p0_idm_eval500_seedA` | 999.2K | ✅ 成功 |
| 71 | `runs/BTC20261002-100757_v7p0_idm_eval500_seedB` | 999.5K | ✅ 成功 |
| 72 | `runs/BTC20261002-101123_v7p0_idm_baseline_eval500` | 999.4K | ✅ 成功 |
| 73 | `runs/BTC20261002-101320_v7p0_idm_eval500_rerun_clean` | 999.3K | ✅ 成功 |
| 74 | `runs/BTC20261002-101322_v7p0_idm_baseline_clean500` | 998.0K | ✅ 成功 |
| 75 | `runs/BTC20261002-101716_v7p0_probe8_r1` | 40.1K | ✅ 成功 |
| 76 | `runs/BTC20261002-101727_v7p0_probe8_r2` | 40.1K | ✅ 成功 |
| 77 | `runs/BTC20261002-101739_v7p0_probe8_r3` | 40.1K | ✅ 成功 |
| 78 | `runs/BTC20261002-101750_v7p0_probe8_r4` | 40.3K | ✅ 成功 |
| 79 | `runs/BTC20261002-101802_v7p0_probe8_r5` | 40.3K | ✅ 成功 |
| 80 | `runs/BTC20261002-101824_v7p0_probe8_h0_r1` | 40.3K | ✅ 成功 |
| 81 | `runs/BTC20261002-101836_v7p0_probe8_h0_r2` | 40.3K | ✅ 成功 |
| 82 | `runs/BTC20261002-101847_v7p0_probe8_h0_r3` | 40.3K | ✅ 成功 |
| 83 | `runs/BTC20261002-101859_v7p0_probe8_h0_r4` | 40.3K | ✅ 成功 |
| 84 | `runs/BTC20261002-101910_v7p0_probe8_h0_r5` | 40.3K | ✅ 成功 |
| 85 | `runs/BTC20261002-101954_v7p0_idm_eval500_hash0_r1` | 999.4K | ✅ 成功 |
| 86 | `runs/BTC20261002-102146_v7p0_idm_eval500_hash0_r2` | 999.4K | ✅ 成功 |
| 87 | `runs/BTC20261002-102339_v7p0_idm_clean500_hash0` | 998.0K | ✅ 成功 |
| 88 | `runs/BTC20261002-102728_v7p0_probe299_w1_r1` | 22.2K | ✅ 成功 |
| 89 | `runs/BTC20261002-102731_v7p0_probe299_w1_r2` | 22.2K | ✅ 成功 |
| 90 | `runs/BTC20261002-102733_v7p0_probe299_w1_r3` | 22.2K | ✅ 成功 |
| 91 | `runs/BTC20261002-102736_v7p0_probe299_w1_r4` | 22.2K | ✅ 成功 |
| 92 | `runs/BTC20261002-102738_v7p0_probe299_w1_r5` | 22.2K | ✅ 成功 |
| 93 | `runs/BTC20261002-102741_v7p0_probe299_w1_r6` | 22.2K | ✅ 成功 |
| 94 | `runs/BTC20261002-102743_v7p0_probe8_w1_r1` | 40.3K | ✅ 成功 |
| 95 | `runs/BTC20261002-102757_v7p0_probe8_w1_r2` | 40.3K | ✅ 成功 |
| 96 | `runs/BTC20261002-102811_v7p0_probe8_w1_r3` | 40.3K | ✅ 成功 |
| 97 | `runs/BTC20261002-102825_v7p0_probe8_w1_r4` | 40.1K | ✅ 成功 |
| 98 | `runs/BTC20261002-102839_v7p0_probe8_w1_r5` | 40.3K | ✅ 成功 |
| 99 | `runs/BTC20261002-102958_v7p0_probe299_seeded_r1` | 22.2K | ✅ 成功 |
| 100 | `runs/BTC20261002-103002_v7p0_probe299_seeded_r2` | 22.2K | ✅ 成功 |
| 101 | `runs/BTC20261002-103006_v7p0_probe299_seeded_r3` | 22.2K | ✅ 成功 |
| 102 | `runs/BTC20261002-103011_v7p0_probe299_seeded_r4` | 22.2K | ✅ 成功 |
| 103 | `runs/BTC20261002-103016_v7p0_probe299_seeded_r5` | 22.2K | ✅ 成功 |
| 104 | `runs/BTC20261002-103020_v7p0_probe299_seeded_r6` | 22.2K | ✅ 成功 |
| 105 | `runs/BTC20261002-160114_v7p1b_tg45` | 111.9K | ✅ 成功 |
| 106 | `runs/BTC20261002-160154_v7p1b_tg45_idm` | 112.2K | ✅ 成功 |
| 107 | `runs/BTC20261002-160206_v7p1b_clean500` | 1009.0K | ✅ 成功 |
| 108 | `runs/BTC20261002-160948_v7p1b_eval500` | 1009.5K | ✅ 成功 |
| 109 | `runs/BTC20261002-162002_v7p1b_screen_epoch005` | 1006.2K | ✅ 成功 |
| 110 | `runs/BTC20261002-162639_v7p1b_screen_epoch010` | 1008.5K | ✅ 成功 |
| 111 | `runs/BTC20261002-163338_v7p1b_screen_epoch015` | 1007.6K | ✅ 成功 |
| 112 | `runs/BTC20261002-164134_v7p1b_tg45_sel` | 112.3K | ✅ 成功 |
| 113 | `runs/BTC20261002-164209_v7p1b_eval500_sel` | 1009.7K | ✅ 成功 |
| 114 | `runs/BTC20261002-1837_v7p1i3` | 73.2M | ✅ 成功 |
| 115 | `runs/BTC20261002-211752_v7p1i3_screen_epoch005` | 1004.9K | ✅ 成功 |
| 116 | `runs/BTC20261002-212641_v7p1i3_screen_epoch010` | 1012.4K | ✅ 成功 |
| 117 | `runs/BTC20261002-213524_v7p1i3_screen_epoch015` | 1011.7K | ✅ 成功 |
| 118 | `runs/BTC20261002-214503_v7p1i3_screen_final` | 1012.9K | ✅ 成功 |
| 119 | `runs/BTC20261002-215549_v7p1i3_tg45_sel` | 112.6K | ✅ 成功 |
| 120 | `runs/BTC20261002-215634_v7p1i3_eval500_sel` | 1006.5K | ✅ 成功 |
| 121 | `runs/BTC20261002-222821_v7p1probe_p1b_tg45_plan` | 112.3K | ✅ 成功 |
| 122 | `runs/BTC20261002-222857_v7p1probe_p1b_tg45_repeat_action` | 112.8K | ✅ 成功 |
| 123 | `runs/BTC20261002-222917_v7p1probe_i3_tg45_plan` | 112.5K | ✅ 成功 |
| 124 | `runs/BTC20261002-223004_v7p1probe_i3_tg45_repeat_action` | 113.5K | ✅ 成功 |
| 125 | `runs/BTC20261002-223027_v7p1probe_p1b_t3_plan` | 43.6K | ✅ 成功 |
| 126 | `runs/BTC20261002-223041_v7p1probe_p1b_t3_repeat_action` | 43.7K | ✅ 成功 |
| 127 | `runs/BTC20261002-223051_v7p1probe_i3_t3_plan` | 43.6K | ✅ 成功 |
| 128 | `runs/BTC20261002-223109_v7p1probe_i3_t3_repeat_action` | 43.7K | ✅ 成功 |
| 129 | `runs/BTC20261002-223120_v7p1probe_p1b_clean150_plan` | 344.7K | ✅ 成功 |
| 130 | `runs/BTC20261002-223344_v7p1probe_p1b_clean150_repeat_action` | 345.1K | ✅ 成功 |
| 131 | `runs/BTC20261002-223438_v7p1probe_i3_clean150_plan` | 343.2K | ✅ 成功 |
| 132 | `runs/BTC20261002-223731_v7p1probe_i3_clean150_repeat_action` | 346.0K | ✅ 成功 |
| 133 | `runs/BTC20261002-2329_v7p1dagger_w2` | 36.4M | ✅ 成功 |
| 134 | `runs/BTC20261002-2329_v7p1dagger_w3` | 36.4M | ✅ 成功 |
| 135 | `runs/BTC20261002-2329_v7p1dagger_w4` | 36.4M | ✅ 成功 |
| 136 | `runs/BTC20261003-005306_v7p1dagger_w1_e005` | 345.2K | ✅ 成功 |
| 137 | `runs/BTC20261003-005609_v7p1dagger_w1_e010` | 345.6K | ✅ 成功 |
| 138 | `runs/BTC20261003-005853_v7p1dagger_w1_e015` | 346.2K | ✅ 成功 |
| 139 | `runs/BTC20261003-021124_v7p1dagger_w2_e005` | 344.7K | ✅ 成功 |
| 140 | `runs/BTC20261003-021410_v7p1dagger_w2_e010` | 345.5K | ✅ 成功 |
| 141 | `runs/BTC20261003-021700_v7p1dagger_w2_e015` | 345.1K | ✅ 成功 |
| 142 | `runs/BTC20261003-033138_v7p1dagger_w3_e005` | 345.6K | ✅ 成功 |
| 143 | `runs/BTC20261003-033454_v7p1dagger_w3_e010` | 346.0K | ✅ 成功 |
| 144 | `runs/BTC20261003-033755_v7p1dagger_w3_e015` | 345.7K | ✅ 成功 |
| 145 | `runs/BTC20261003-045010_v7p1dagger_w4_e005` | 344.7K | ✅ 成功 |
| 146 | `runs/BTC20261003-045300_v7p1dagger_w4_e010` | 346.5K | ✅ 成功 |
| 147 | `runs/BTC20261003-045608_v7p1dagger_w4_e015` | 345.0K | ✅ 成功 |
| 148 | `runs/BTC20261003-045912_v7p1dagger_w1_clean500` | 1009.9K | ✅ 成功 |
| 149 | `runs/BTC20261003-050855_v7p1dagger_w1_tg45` | 112.3K | ✅ 成功 |
| 150 | `runs/BTC20261003-051010_v7p1dagger_w2_clean500` | 1011.3K | ✅ 成功 |
| 151 | `runs/BTC20261003-051910_v7p1dagger_w2_tg45` | 112.5K | ✅ 成功 |
| 152 | `runs/BTC20261003-052024_v7p1dagger_w3_clean500` | 1012.4K | ✅ 成功 |
| 153 | `runs/BTC20261003-053043_v7p1dagger_w3_tg45` | 113.2K | ✅ 成功 |
| 154 | `runs/BTC20261003-053208_v7p1dagger_w4_clean500` | 1009.3K | ✅ 成功 |
| 155 | `runs/BTC20261003-054109_v7p1dagger_w4_tg45` | 112.7K | ✅ 成功 |
| 156 | `runs/BTC20261003-054238_v7p1dagger_w4_eval500` | 1011.1K | ✅ 成功 |
| 157 | `runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory` | 1010.7K | ✅ 成功 |
| 158 | `runs/BTC20261003-0641_v7p2_s0_arm1` | 9.9M | ✅ 成功 |
| 159 | `runs/BTC20261003-065707_v7p2_s0_u050_sub150` | 345.5K | ✅ 成功 |
| 160 | `runs/BTC20261003-070038_v7p2_s0_u100_sub150` | 344.3K | ✅ 成功 |
| 161 | `runs/BTC20261003-070430_v7p2_s0_u025_sub150` | 344.8K | ✅ 成功 |
| 162 | `runs/BTC20261003-070745_v7p2_s0_u075_sub150` | 347.9K | ✅ 成功 |
| 163 | `runs/BTC20261003-071123_v7p2_s0_u125_sub150` | 346.5K | ✅ 成功 |
| 164 | `runs/BTC20261003-071530_v7p2_s0_u150_sub150` | 343.7K | ✅ 成功 |
| 165 | `runs/BTC20261003-071904_v7p2_s0_u175_sub150` | 342.3K | ✅ 成功 |
| 166 | `runs/BTC20261003-072227_v7p2_s0_u200_sub150` | 337.8K | ✅ 成功 |
| 167 | `runs/BTC20261003-072348_v7p2_s0_u100_clean500` | 1008.2K | ✅ 成功 |
| 168 | `runs/BTC20261003-073541_v7p2_s0_u050_clean500` | 1012.0K | ✅ 成功 |
| 169 | `runs/BTC20261003-074639_v7p2_s0_tg45` | 112.5K | ✅ 成功 |
| 170 | `runs/BTC20261003-074819_v7p2_s0_eval500` | 1008.4K | ✅ 成功 |

## 3. 删除后结构概览

- `datasets/` 剩 **4 项**：KEEP 3 项（`BTC20260926-2343_expert5k`、`BTC20261007-2202_expert5k_v8`、`BTC20261007-2202_expert500val_v8`）+ 人工确认 1 项（`BTC20261002-0941_expert5k_v41`）。
- `runs/` 顶层剩 **159 项** = Tier-2 **135 项（完整未动）** + KEEP 24 项：
  - `BTC20261007-*` × 8、`BTC20261009-*` × 6；
  - 非 BTC 6 项：`eval`、`_refs_oldgen`、`_refs_rlbase`、`reward_viz`、`reward_audit`、`reward_audit_ebeta2`；
  - 用户口径 2 项：`BTC20261002-2329_v7p1dagger_w1`、`BTC20261005-0601_v7p2_s11_arm1`；
  - 硬引用移 KEEP 2 项：`BTC20260929-0425_fixA_nold`、`BTC20261001-1631_v6p3`。
- 保护核验（执行后）：Tier-2 135/135 存在；KEEP 全部存在；`20261007`×8 / `20261009`×6 完整；`runs/eval` 内 36 个评测子目录未动。

## 4. 备注

- 本执行只删除了 WILL_DELETE 清单中 A/B 段的 186 个目录；未触碰任何文件级内容（不涉及 `.slim/`、config、文档）。
- 并行会话（config/docs 归档）的仓库改动与本执行无关，未干预；本执行**未产生 git commit**。
- Tier-2 的 135 条 `rm -rf` 命令仍保留在 WILL_DELETE §5-C 段（未执行，后续需另行授权）。
