# /tmp → repo 路径迁移说明（MIGRATION_NOTE）

> 生成：2026-10-06 锁版清理 B。用途：解释 `docs/` 正文中历史 `/tmp/opencode/...` 引用，以及本日归档/删除路径的对应关系。
> **本说明不修改任何冻结文档正文**；`docs/v7_program_prereg.md`、`docs/v6_program_prereg.md` 仅在顶部加了 1 行路径注记（指向本文件）。
> 清理记录（权威）：`docs/v7_reports/cleanup/`（`CLEANUP_PLAN.md`、`DELETED_MANIFEST.txt`、`CLEANUP_A_REPORT.md`、`CLEANUP_B_REPORT.md`、`KEEP_README.md`）。

## 1. `/tmp/opencode/` 证据档 → 已入仓（in-repo 权威副本）

清理 A（2026-10-06，commit `8762a22`）将 v7/v6 关键报告、图、规格、时间线**先复制后删除**入仓；逐项原路径/新路径/sha256 见：

- v7：`docs/v7_reports/README.md` §1（报告 24）、§2（图 4）、§3（规格 3）、§4（时间线 1）
- v6：`docs/v6_reports/README.md`（报告 7）
- KEEP 未决项（清理 B 入仓）：见本文 §3

## 2. `/tmp/opencode/` 其余工作档 → 已删除（无副本）

一次性驱动脚本、日志、中间 JSON、工作目录（`phase3_diag/`、`v7_pre_v5/` worktree、`rollback/`、`oldcode/`、`research/` 等）已随清理 A 删除，逐项清单见 `docs/v7_reports/cleanup/DELETED_MANIFEST.txt`。docs 正文中的此类 `/tmp/opencode/<脚本|日志|目录>` 引用**视为历史路径**，不再解析。

例外（已转存）：

| /tmp 前缀 | 去向 |
|---|---|
| `/tmp/opencode/phase3_diag/exp/specs_val_only{500,150}.json` | `docs/v7_reports/specs/`（sha256 见 README §3） |
| `/tmp/opencode/v7_p1b_chain/specs_tollgate45.json` | `docs/v7_reports/specs/` |
| `/tmp/opencode/v7_q2_tollgate_figure/*.png` | `docs/v7_reports/figures/` |
| `/tmp/opencode/v7_night_watch.log` | `docs/v7_reports/v7_night_watch.log` |
| `/tmp/opencode/v7_p0_specs/*` | `docs/v7_reports/specs/p0_variants/`（清理 B，见 §3） |
| `/tmp/opencode/v7_q6q2/*.json` | `docs/v7_reports/evidence/`（清理 B，见 §3） |
| `/tmp/opencode/v7_pre_v5/`（worktree） | 已移除；复现见 §6 |

## 3. KEEP 复核入仓（清理 B，8 文件）

原 `/tmp/opencode/KEEP/`（清理 A 未决项，均被入仓报告直接引用）已搬入 repo，sha256 为入仓副本实测：

| 原路径 | 新路径 | sha256 |
|---|---|---|
| `KEEP/v7_struct_retrain_v5.md` | `docs/v7_reports/v7_struct_retrain_v5.md` | `1b0b6194ae8c9d70f0a47e71832593f9e4f11aacf2132a6e53148191135d9149` |
| `KEEP/v7_p0_specs/scenarios_eval500_seedA.json` | `docs/v7_reports/specs/p0_variants/scenarios_eval500_seedA.json` | `e31a87b232777ccc6f565e44a1d83e85ad2cdb5b909ed9e093098b029d2b6635` |
| `KEEP/v7_p0_specs/scenarios_eval500_seedB.json` | `docs/v7_reports/specs/p0_variants/scenarios_eval500_seedB.json` | `fa644eb49d85999c1a651e44fc7ecab6b33dc259a2fbddacc8d8b645ace5e7dc` |
| `KEEP/v7_p0_specs/variants_manifest.json` | `docs/v7_reports/specs/p0_variants/variants_manifest.json` | `489326e8a8f1ecb5a5a70e2c391b6389bd45168fa4e8ff5ec4b7c3d93963625f` |
| `KEEP/v7_p0_specs/determinism_probe8.json` | `docs/v7_reports/specs/p0_variants/determinism_probe8.json` | `f548cb02f6881baab02b68d8c315825f31199ee64306241c4408ad2d88260777` |
| `KEEP/v7_p0_specs/probe_299.json` | `docs/v7_reports/specs/p0_variants/probe_299.json` | `62e6b73656bc2f1075fc2ebdb373ea955890ccd11055afbfa6549fb7d54a07bd` |
| `KEEP/v7_q6q2/collision_replay.json` | `docs/v7_reports/evidence/collision_replay.json` | `8413de0baff9473b1fb9eef7258d82bf37914d0ec1eb51d9b7487cfafdbcf8ab` |
| `KEEP/v7_q6q2/tollgate_viz.json` | `docs/v7_reports/evidence/tollgate_viz.json` | `55d09fda13618d8e415963981c00ad4037d3a3ff4a7d84becec751b6d8d1a91b` |

## 4. repo 内删除（清理 B；`runs/`、`datasets/` 为 gitignore 未跟踪项，不影响 git 历史）

- `runs/`：删除 **133 项 / 377.3 MiB**（v3/v4 代 perf/smoke/中间 eval、p4prep smoke、`_repeat` 等未引用陈旧项）。
- `datasets/`：删除 **5 项 / 324.7 MiB**，其中 **v5 线 3 项 / 323.1 MiB**（`BTC20261005-0814_expert5k_v5`、`BTC20261005-0856_phase3_dagger_v5`、`BTC20261005-0814_expert500val_v5`；v5 线已关闭、报告已归档，删除前引用情况记录在案）。
- **复核后保留 16 项 / 161.6 MiB**（有引用）：`runs/BTC20260929-1357_{stageB,eval500}_phase3_r2..r5`（`docs/phase3_rootcause_analysis.md` `r{1..5}` + `.slim` 引用）、`runs/BTC20260930-1434/1437/1441/1449_eval500_*`（`docs/v6_reports/v6_cleanup_report.md:43` 零点引用清单）、`datasets/BTC20260929-1357_phase3_dagger_r1..r4`（同 `r{1..5}` 引用）。
- 逐项清单与引用复核结果：`docs/v7_reports/cleanup/CLEANUP_B_REPORT.md`。

## 5. `docs/` → `docs/archive/`（清理 B，移动不删）

10 项 v3/v4 代归档候选移入 `docs/archive/`（映射表见 `docs/archive/README.md`）：`rl_stage_c_report.md`、`rl_stage_c_v3_prereg.md`、`rl_stage_c_v4_prereg.md`、`rl_stage_c_v4p1_prereg.md`、`rl_v3_evidence.sha256`、`p0-measurements.md`、`feasibility-analysis.md`、`forensics-2026-09-26.md`、`db44fefe-system-review.md`、`architecture-BTC20260925-2234.svg`。
其他文档若引用上述原 `docs/<名>` 路径，改指 `docs/archive/<名>` 即可（内容逐字未变）。

## 6. 复现提示

- **pre-v5 worktree 已移除**（原 `/tmp/opencode/v7_pre_v5` @ `2f4450e`）：如需复现 P3 s1/s2 与 P4-extra/奖励/§14 臂（同代码期）→ `git worktree add <dir> 2f4450e`（主树 venv 绝对解释器 + `runs`/`datasets` 软链主树）。
- **规格文件**：clean500/tg45 派生件只在 `docs/v7_reports/specs/`（`env/specs/` 不含），复算须用该目录三文件（sha256 逐位核对）。
- 最佳产物 ckpt 仍在 `runs/`（保留链）：w1 `runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt`、s11 `runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt`、p4e s0u75 `runs/BTC20261005-2138_v7p4extra_s0_colls/ckpt_u075.pt`、s14 s11u25 `runs/BTC20261006-0711_v7s14_s11_ttc/ckpt_u025.pt`。
