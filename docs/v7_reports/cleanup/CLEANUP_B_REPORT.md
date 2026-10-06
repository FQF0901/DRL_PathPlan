# 锁版清理 B 报告（repo 清理 + KEEP 搬运 + 锁版 tag）

> 执行：fixer 子代理；日期：2026-10-06；范围：`/workspace/01_Proj/DRL_PathPlan`（排除 `.venv`）。
> 前置：`CLEANUP_PLAN.md`（清理 A 清单）、`DELETED_MANIFEST.txt`（清理 A /tmp 删除逐项）。
> 提交：`93338a1`（主提交：删除/归档/入仓/说明）+ 本报告提交；锁版 tag：`v7-lock-20261006`。
> 约束遵守：**不删被引用项**（复核后保留 16 项并记录）；删前 dry-run + 逐项记录；未动 `env/`、`net/`、`pipeline/` 代码；冻结文档正文未改（v7/v6 prereg 仅顶部各加 1 行路径注记）。

## 0. 执行摘要（体积前后）

| 区域 | 前 | 后 | 变化 | 说明 |
|---|---|---|---|---|
| `runs/` | 4,021,246,915 B（510 项） | 3,625,662,343 B（377 项） | **−395,584,572 B（−377.3 MiB）** | 删 133 项；复核保留 12 项（76.9 MiB，有引用） |
| `datasets/` | 1,878,340,170 B（25 项） | 1,537,847,008 B（20 项） | **−340,493,162 B（−324.7 MiB）** | 删 5 项（含 v5 线 3 项 / 323.1 MiB）；复核保留 4 项（84.8 MiB，有引用） |
| `docs/` | 10,906,924 B | 12,184,127 B | +1,277,203 B | 归档 10 项（移动不删）+ KEEP/清理记录/迁移说明入仓 |
| repo 总计（排除 `.venv`/`.git`） | 5,928,914,918 B | 5,197,517,693 B | **−731,397,225 B（≈−697.5 MiB）** | 差额含 docs 增重与 pytest `__pycache__` 再生 |
| `/tmp/opencode/` | 1.6M（记录+KEEP） | ≈0 | −1.6M | 记录已存档入仓（§6） |

**删除总量**：`runs/` 133 项 + `datasets/` 5 项 = **138 项 / 736,077,734 B（702.0 MiB）**（脚本逐项 `du -sb` 求和；删除 138/138 成功，0 失败）。

## 1. 删前复核（强制）方法与结果

方法：对 `CLEANUP_PLAN.md` §A/§B 全部候选（145 runs + 9 datasets）逐项重新检索 **tracked 全仓**（`git grep`，覆盖 `docs/` 含 `docs/v7_reports/`、`docs/v6_reports/`，以及 `tools/`/`tests/`/`config/` 等）：先全名、再去 `BTC<日期>-` 短名；补充花括号族引用（`r{1..5}`/`w{1..4}`）、通配（`BTC20260930-1434*`）与缩写（`-1437*`）人工判读。逐项结果：`cleanup/CLEANUP_B_WILL_DELETE.md`。

**复核发现“有引用” → 保留 16 项（161.6 MiB），并记录**：

| 保留项 | 体积 | 引用 |
|---|---|---|
| `runs/BTC20260929-1357_stageB_phase3_r2..r5` | 68.8M | `.slim/deepwork/phase3-rl-base-analysis.md:19`（`r{1..5}`）；`docs/v6_reports/v6_cleanup_report.md:211` 已记录保留 |
| `runs/BTC20260929-1357_eval500_phase3_r2..r5` | 4.0M | `docs/phase3_rootcause_analysis.md:25,250`（`r{k}`/`r{1..5}`）；`v6_cleanup_report.md:210` |
| `datasets/BTC20260929-1357_phase3_dagger_r1..r4` | 84.8M | `docs/phase3_rootcause_analysis.md:251` + `.slim/...:19`（`r{1..5}`，5 轮窗口数据族） |
| `runs/BTC20260930-1434/1437/1441/1449_eval500_*` | 4.0M | `docs/v6_reports/v6_cleanup_report.md:43`（零点引用 run 清单，通配/缩写） |

**v5 线（显式删除，引用情况记录在案）**：`expert5k_v5` / `expert500val_v5` / `phase3_dagger_v5` 共 3 项 / 323.1 MiB —— 已关闭路线、报告已归档；引用点（`docs/v7_program_report.md:150`、`docs/v7_program_prereg.md:342-343`）为历史叙述，非复现依赖。

**其余删除项复核为无引用**；少量命中为假阳性（如 `BTC20260930-0418_eval500_p3a4*` 的时间戳命中实为保留父 run `stage_c_p3_a4`；`datasets/BTC20260926-2343_expert5k.log` 命中实为目录名），已逐项判读后删除。

## 2. `runs/` 删除明细（133 项 / 377.3 MiB）

类别（逐项见 `cleanup/CLEANUP_B_DELETED.txt`）：

- v3/v4 代实验目录（未引用）：`stage_c_v4_*` 9 项（≈208M）、`stage_c_v41pa/pk`、`v7sb_*` 中间项、`indist_*`、`diag_v4post*`、`diag_rollback*`、`diag_keepbest*` 等；
- perf/smoke/中间 eval：`stage_c_perf*` 14 项、`stage_c_smoke*`/`drift_*` 7 项、`p4_smoke_*` 5 项、`p4prep`/`p3a4p`/`p3a*` 中间 eval、`v7p0/v7p1` 探针中间项等；
- `_repeat` 重复评测：`eval500_{l2,ebeta,clean_l2,clean_ebeta}_repeat` 等（其中 1434/1437/1441/1449 因有引用保留，见 §1）。

## 3. `datasets/` 删除明细（5 项 / 324.7 MiB）

| 项 | 体积 | 原因 |
|---|---|---|
| `BTC20261005-0814_expert5k_v5` | 275.3M | v5 线已关闭（显式删除） |
| `BTC20261005-0814_expert500val_v5` | 27.8M | v5 线已关闭（显式删除） |
| `BTC20261005-0856_phase3_dagger_v5` | 20.0M | v5 线已关闭（显式删除） |
| `BTC20260926-2343_expert5k.log` | 993K | 生成日志，无引用（命中为目录名假阳性） |
| `BTC20260929-1103_phase3_dagger_r1` | 669K | 未引用陈旧（1103 轮 runs 已在 v6 清理删除） |

保留 16 项 / 1.35G 不动（含 §1 的 4 项 + v4/v4.1/v7dagger 等最佳线数据）。

## 4. `docs/` 归档（10 项 → `docs/archive/`，移动不删）

`rl_stage_c_report.md`、`rl_stage_c_v3_prereg.md`、`rl_stage_c_v4_prereg.md`、`rl_stage_c_v4p1_prereg.md`、`rl_v3_evidence.sha256`、`p0-measurements.md`、`feasibility-analysis.md`、`forensics-2026-09-26.md`、`db44fefe-system-review.md`、`architecture-BTC20260925-2234.svg`。索引/映射：`docs/archive/README.md`。`rl_stage_c_v4_report.md`（被 v6 prereg 引用）按计划保留原位。

## 5. KEEP 搬运（8 文件 / 1.1M → `docs/v7_reports/`，sha256 逐位校验一致）

- `v7_struct_retrain_v5.md` → `docs/v7_reports/`；
- P0 变体 5 件 → `docs/v7_reports/specs/p0_variants/`；
- `collision_replay.json`、`tollgate_viz.json` → `docs/v7_reports/evidence/`。

README 索引行已补：`docs/v7_reports/README.md` §6（未决→已解决）、§7（8 项含 sha256）。

## 6. 迁移说明与记录存档

- 新增 `docs/v7_reports/MIGRATION_NOTE.md`：`/tmp` 引用 → 入仓/删除映射、KEEP 去向、repo 删除摘要、docs→archive 映射、复现提示。
- `docs/v7_program_prereg.md`、`docs/v6_program_prereg.md` 顶部各加 **1 行路径注记**（正文未改）。
- 清理记录存档 `docs/v7_reports/cleanup/`：`CLEANUP_PLAN.md`、`DELETED_MANIFEST.txt`、`CLEANUP_A_REPORT.md`、`CLEANUP_B_REPORT.md`（本文件）、`KEEP_README.md`、`CLEANUP_B_WILL_DELETE.md`、`CLEANUP_B_DELETED.txt`。
- `/tmp/opencode/` 清空（仅本报告生成前的暂存；完成后删除）。

## 7. 验证

- **测试**：`.venv/bin/python -m pytest -q` → **694 passed**, 145 warnings in 32.89s（exit 0；代码未动，全绿）。
- **git**：`git status --porcelain` 干净（提交 `93338a1` + 本报告提交）。
- **体积**：见 §0（runs −377.3 MiB / datasets −324.7 MiB / 总 −702.0 MiB）。
- **完整性抽查**：最佳产物链 ckpt 在位且 sha256 前缀吻合（w1 `fdfe0808…`、s11 `a7cc091f…`、p4e s0u75 `4217abe0…`、s14 s11u25 `229bbc1e…`）。

## 8. 锁版 tag

`git tag -a v7-lock-20261006`（指向本报告提交）：说明锁版状态 = 代码 + docs + 关键 ckpt/数据保留；清理记录（A/B）；最佳产物 sha256 全值：

| 产物 | ckpt | sha256 |
|---|---|---|
| w1（P1 冠军 / P2-P3 base） | `runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt` | `fdfe080818eb355b388692269454538eed8b5cc318a9eb4e73b206f8cbd688d9` |
| s11（程序最佳 RL，u150） | `runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt` | `a7cc091fcbda670b25c396a43e18dc39abe089053e50aa292b5fc0f19164ba2e` |
| p4e s0 u75（安全改善版） | `runs/BTC20261005-2138_v7p4extra_s0_colls/ckpt_u075.pt` | `4217abe0d67cc231e48b1fe783c3d2fbd87dac5022a091c71d7b1cf4f54dd342` |
| s14 s11 u25（单点 PASS） | `runs/BTC20261006-0711_v7s14_s11_ttc/ckpt_u025.pt` | `229bbc1e37c6e6fbfbfcdcc96fa6426c6fd0f9f86412d78cecd1c2f088e29b7f` |

## 9. 与计划口径的偏差（如实报告）

1. `runs/` 计划“建议删除 105 + 复核 40”：实删 **133**，复核后保留 12 项（77M，phase3 族 + 零点清单）——花括号/通配/缩写引用是计划方法未覆盖的引用形式。
2. `datasets/` 计划“建议删除 7 项/≈114M”：实删 3 项（≈29.5M），**r1–r4 复核发现被 `r{1..5}` 族引用 → 保留**（84.8M）；v5 3 项按任务显式删除。
3. v5 线删除项存在历史文档引用（`v7_program_report.md:150` 等）——按任务“已关闭路线、报告已归档 → 删除并记录”执行。
