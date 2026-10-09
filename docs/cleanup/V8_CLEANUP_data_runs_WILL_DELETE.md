# V8 清理-2：datasets/ 与 runs/ 清理 dry-run 清单（WILL-DELETE，本阶段未执行）

> **✅ 已执行（保守范围：datasets 16 + runs Tier-1 170；Tier-2 未执行）→ 见 [V8_CLEANUP_data_runs_DONE.md](V8_CLEANUP_data_runs_DONE.md)**
> （注：本清单正文保留为执行前 dry-run 快照；下方 §5-C 段 Tier-2 命令仍未执行。）

> 生成：2026-10-09（CST）｜仓库：`/workspace/01_Proj/DRL_PathPlan`
> **阶段纪律**：本阶段**严禁任何删除/移动/改名**；除本文件外不新增/修改任何文件；不触碰任何 `20261009-*` 目录与 `.slim/` 内文件（`.slim/` 仅只读检查）。
> **删除不可回滚**：`datasets/`、`runs/` 均在 `.gitignore` 中（被忽略，git 无法恢复）。
> **方法**：
> - 体积 = `du -sb` 逐目录；mtime = `stat -c %y`（多数目录 mtime 为 2026-10-06 11:05 的历史迁移戳，不代表内容新旧）；
> - 引用检查 = **边界匹配** grep（正则 `名字([^A-Za-z0-9_-]|$)`，防止 `..._v4` 误配 `..._v41`、`..._v5` 误配 `..._v5_p3` 等前缀污染），范围 = tracked 文件 `tests/ tools/ pipeline/ net/ env/ reward_model/ config/ docs/` + `.slim/deepwork/*.sh`；另做 `.slim/` 全目录与全仓 `runs/BTC*` 引用扫描作为补充警告；
> - 近 1h 写入 = 逐候选 `find <dir> -type f -newermt '-60 minutes'`，另加 `runs/` 全局扫描交叉验证。
> **核验结果**：327 个候选目录**近 1h 写入 = 0**；当前活跃写入只在 KEEP 区（`runs/eval/`、`runs/BTC20261009-1545_sw_exp128`、`runs/BTC20261009-1545_sw_attn2`）。

## 0. 汇总

| 区域 | 候选总量 | KEEP / 移出 | 拟删 | 拟删体积 | 口径 |
|---|---|---|---|---|---|
| `datasets/`（20 项） | 2.0G | KEEP 3 项 598.3M + 人工确认 1 项 286.2M | 16 项 | **1.1G** | 硬引用检查 |
| `runs/` Tier-1（0927~1003 的 BTC*） | 172 项 590.3M | 硬引用移 KEEP 2 项 62.7M | 170 项 | **527.6M** | 明确过期（v6 代 + v7 p1 代） |
| `runs/` Tier-2（1005/1006 的 BTC*） | 135 项 349.4M | —（s11_arm1 10M 已按任务 KEEP，不在候选内） | 135 项 | **349.4M** | v7 复现链证据，口径待定 |
| **合计拟删** | | | **321 项** | **≈ 2.0G** | Tier-2 建议暂缓 |

细分（供口径决策）：

- **Tier-1 拟删 170 项**：96 项有非台账文档叙述引用（503.0M），74 项仅历史台账引用（24.6M）；代码/配置/测试/`.sh` 硬引用 = 0。
- **Tier-2 拟删 135 项**：27 项被 v7 复现链报告/证据文件引用（284.3M），108 项仅历史台账引用（65.0M）。
- **datasets 拟删 16 项**：其中 `BTC20261007-1838` 对（305.8M）仅被 `.slim` 历史日志引用（警告项）；`BTC20261002-0941_expert5k_v4`（286.3M）是 v7 P1 DAgger cycle 的锚数据集（叙述引用）。

> **⚠️ 口径冲突警告（重要）**：Tier-1 拟删 **170/170**、Tier-2 拟删 **135/135** 在 2026-10-06 的上一版 `docs/v7_reports/cleanup/CLEANUP_PLAN.md` 中**均被明确列为「保留」**（理由：最佳产物链 / 被 v6/v7 报告引用；该锁版 tag = `v7-lock-20261006`）。本阶段的「拟删」口径与之**全面冲突**——执行删除 = 放弃 v6/v7 复现链证据，须由人工明确签署后再执行。

## 1. KEEP 总清单（含理由）

### 1.1 `datasets/` KEEP（3 项 / 598.3M）

| 目录 | 体积 | 依据（已逐条 grep 核对） | 口径 |
|---|---|---|---|
| `datasets/BTC20260926-2343_expert5k` | 283.3M | `config/train.yaml::train.probe_batch`；`pipeline/trainer.py::DEFAULT_PROBE_BATCH`；`pipeline/stages.py`；`tools/make_dagger_pools.py::COVERED_BY`；`config/arms/arm4_lam098.yaml` | **任务指定 KEEP + 代码/配置硬依赖** |
| `datasets/BTC20261007-2202_expert5k_v8` | 286.3M | 当前 v8 排摸训练数据；`.slim/deepwork/v8b_sweep_a.sh`、`v8b_sweep_b.sh`、`v8b_retry_stg3.sh` 硬引用 | **任务指定 KEEP（当前排摸）** |
| `datasets/BTC20261007-2202_expert500val_v8` | 28.7M | 同上（验证集）；3 个 `.slim/deepwork/*.sh` 硬引用 | **任务指定 KEEP（当前排摸）** |

引用检查与近 1h 写入明细（自动生成）：

| 目录 | 体积 | mtime | 近1h写 | 引用检查 |
|---|---|---|---|---|
| `datasets/BTC20260926-2343_expert5k` | 283.3M | 2026-09-27 19:32:03 | 无 | **硬引用**：config/README.md；config/arms/arm4_lam098.yaml；config/train.yaml；pipeline/stages.py；pipeline/trainer.py；tools/make_dagger_pools.py<br>叙述：archive/rl_stage_c_experiments.md；archive/rl_stage_c_v3_prereg.md；experiments.md；v6_program_prereg.md；v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_B_DELETED.txt；v7_reports/cleanup/CLEANUP_B_REPORT.md；v7_reports/cleanup/CLEANUP_B_WILL_DELETE.md；v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261007-2202_expert500val_v8` | 28.7M | 2026-10-07 22:30:41 | 无 | **硬引用**：.slim/deepwork/v8b_retry_stg3.sh；.slim/deepwork/v8b_sweep_a.sh；.slim/deepwork/v8b_sweep_b.sh |
| `datasets/BTC20261007-2202_expert5k_v8` | 286.3M | 2026-10-07 22:28:13 | 无 | **硬引用**：.slim/deepwork/v8b_retry_stg3.sh；.slim/deepwork/v8b_sweep_a.sh；.slim/deepwork/v8b_sweep_b.sh |

### 1.2 `runs/` KEEP（22 项 / ≈2.1G）

| 目录 | 体积 | 理由 | 口径 |
|---|---|---|---|
| `runs/BTC20261007-*`（8 个：1838_v8chain、1940_v8smoke、2202_v8chain、2202_a2_featurewm、2202_a3_oldparams、2202_arm_a4_oldlatent、2202_arm_p、2202_arm_t） | 774M | 当前 v8 排摸链；`.slim/deepwork/v8_run/` 台账持续引用 | 任务强制 |
| `runs/BTC20261009-*`（6 个：0824_sw_pri128/pri512/rou64/stg3、1545_sw_attn2/exp128） | 433M | 当前排摸 sweep；近 1h 仍有活跃写入（sw_attn2、sw_exp128） | 任务强制 |
| `runs/eval/`（36 个子目录） | 24M | 全部评测产物（含 20261009-* 活跃评测；近 1h 有写入） | 任务强制 |
| `runs/_refs_oldgen` | 16M | 参考产物 | 任务强制 |
| `runs/_refs_rlbase` | 9.6M | **工具硬引用**：`tools/reward_audit.py::DEFAULT_CKPT`、`tools/reward_audit_collect.py` 默认 `--ckpt`（均指向 `e_beta_prime/final.pt`） | 任务强制 + 硬引用 |
| `runs/reward_viz` | 207M | 奖励可视化最佳产物链 | 任务强制 |
| `runs/reward_audit` | 178M | 奖励审计最佳产物链 | 任务强制 |
| `runs/reward_audit_ebeta2` | 430M | 奖励审计（E-β″）最佳产物链 | 任务强制 |
| `runs/BTC20261002-2329_v7p1dagger_w1` | 37M | s11 臂的 `--ckpt` 依赖；`config/arms/v7_arm2_bundle_kl.yaml`、`v7_arm3_bundle_only.yaml` 注释引用 `w1/ckpt_epoch005.pt`（注：该两文件在清单生成期间被并行清理会话归档至 `docs/archive/config_arms_legacy/`，引用关系不变） | **用户既有复现链口径** |
| `runs/BTC20261005-0601_v7p2_s11_arm1` | 10M | 当前 RL 配方证据；`.slim/` 引用 8 处 | **用户既有复现链口径** |

## 2. `datasets/` 分类（20 项 / 2.0G）

### 2.1 KEEP（3 项 / 598.3M）

见 §1.1。

### 2.2 需人工确认（1 项 / 286.2M）：`datasets/BTC20261002-0941_expert5k_v41`

硬引用命中（3 处，均为「记录/工具默认」性质，非运行时读取）：

| 文件 | 性质 | 影响 |
|---|---|---|
| `tools/fit_plan_anchors.py:89` | **默认 `--dataset`** = `datasets/BTC20261002-0941_expert5k_v41/expert_bc.npz` | 删除后重拟合锚字典（`config/plan_anchors_k6.json`）的默认命令失效，须先重采数据 |
| `config/plan_anchors_k6.json:6` | `"dataset"` provenance 字段 | 锚字典本体（tracked）不受影响；运行时加载不读 npz |
| `net/anchor.py:39` | 注释（来源记录） | 文档断链 |

判定：删除**不破坏**运行时训练/评测/测试（`config/model.yaml` 加载的是锚字典 JSON），但会使锚字典 provenance 与重拟合路径断链 → **不进删除命令**，交人工裁决。若确认 v7 结构线已关闭且不再重拟合锚，可降级为可删（286.2M）。

| 目录 | 体积 | mtime | 近1h写 | 引用检查 |
|---|---|---|---|---|
| `datasets/BTC20261002-0941_expert5k_v41` | 286.2M | 2026-10-02 17:58:21 | 无 | **硬引用**：config/plan_anchors_k6.json；net/anchor.py；tools/fit_plan_anchors.py<br>叙述：v7_program_prereg.md；v7_reports/v7_kanchor_feasibility.md；v7_reports/v7_p1_iter2.md；v7_reports/v7_p1_iter3.md；v7_reports/v7_struct_b_kanchor.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_D_DEADCODE.md；v7_reports/cleanup/CLEANUP_PLAN.md |

### 2.3 拟删（16 项 / 1.1G）

逐项引用检查见下表（`叙述` = 文档引用；`历史台账` = `docs/v7_reports/cleanup/*` 等清理记录；**硬引用 = 0**）。逐项备注：

- **`BTC20261002-0941_expert5k_v4`（286.3M）**：无硬引用。是 **v7 P1 DAgger cycle 的锚数据集**（`docs/v7_reports/v7_p1_dagger_cycle.md` §1：anchor = `…_v4`，mild=1.0）；删后 w1..w4 训练链不可从数据复现（w1 run ckpt 仍 KEEP）。
- **`BTC20261002-2329_v7p1dagger_w1..w4`（各 ~22M）**：仅叙述引用（`v7_p1_dagger_cycle.md` 等）。注：w1 行引用中的「硬引用(裸名)」实为 `config/arms/*.yaml` 注释指向 **run** `runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt`（KEEP），**不是**本数据集路径 → 本数据集无硬引用。
- **`BTC20261007-1838_expert5k_v8` + `_expert500val_v8`（305.8M）**：tracked 引用 = 无；`.slim` 非 `.sh` 引用 = 有（`v8_watch.log`、`v8_backlog.md`、`v8_run/logs/*.log`、`v8_run/chain.log`，均为历史执行日志/台账）。它们是 KEEP 的 `runs/BTC20261007-1838_v8chain` 的训练数据 → 删除后该链**不可复训**（已有 ckpt/eval 仍可评）。**警告项**。
- **`BTC20261001-1327_expert5k/expert500val`（315.0M）**：叙述引用（v6 报告；`.slim/deepwork/v6-net-retrain.md` 记录采集）；无硬引用。
- **`BTC20260929-1357_phase3_dagger_r1..r5`（~107M）**：叙述引用（`v6_program_prereg.md`、`v6_p3_report.md`、`.slim/deepwork/phase3-rl-base-analysis.md` 以 `r{1..5}` 花括号族引用）；无硬引用。删后 v6 P3 phase3 五轮链不可复现。

| 目录 | 体积 | mtime | 近1h写 | 引用检查 |
|---|---|---|---|---|
| `datasets/BTC20260929-1357_phase3_dagger_r1` | 21.4M | 2026-09-29 14:17:49 | 无 | 叙述：v7_reports/MIGRATION_NOTE.md<br>历史台账：v7_reports/cleanup/CLEANUP_B_REPORT.md；v7_reports/cleanup/CLEANUP_B_WILL_DELETE.md；v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20260929-1357_phase3_dagger_r2` | 21.5M | 2026-09-29 14:41:13 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_B_WILL_DELETE.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20260929-1357_phase3_dagger_r3` | 20.8M | 2026-09-29 15:06:40 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_B_WILL_DELETE.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20260929-1357_phase3_dagger_r4` | 21.2M | 2026-09-29 15:30:55 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_B_WILL_DELETE.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20260929-1357_phase3_dagger_r5` | 20.7M | 2026-09-29 15:56:57 | 无 | 叙述：v6_program_prereg.md；v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261001-1327_expert500val` | 28.9M | 2026-10-01 13:57:47 | 无 | 叙述：v6_program_prereg.md；v6_program_report.md；v6_reports/v6_p3_report.md；v7_program_prereg.md<br>历史台账：v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261001-1327_expert5k` | 286.1M | 2026-10-01 13:55:39 | 无 | 叙述：v6_program_prereg.md；v6_program_report.md；v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261002-0941_expert500val_v4` | 28.9M | 2026-10-02 10:59:27 | 无 | 叙述：v7_reports/v7_p1b_chain.md；v7_reports/v7_p1b_failure_diag.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261002-0941_expert500val_v41` | 28.9M | 2026-10-02 18:00:25 | 无 | 叙述：v7_reports/v7_p1_iter2.md；v7_reports/v7_p1_iter3.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261002-0941_expert5k_v4` | 286.3M | 2026-10-02 10:57:19 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p1_dagger_cycle.md；v7_reports/v7_p1b_chain.md；v7_reports/v7_p1b_failure_diag.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261002-2329_v7p1dagger_w1` | 22.2M | 2026-10-02 23:51:35 | 无 | **硬引用(裸名)**：config/arms/v7_arm2_bundle_kl.yaml；config/arms/v7_arm3_bundle_only.yaml<br>叙述：v7_reports/v7_p1_dagger_cycle.md<br>叙述(裸名)：v7_program_prereg.md；v7_program_report.md；v7_reports/MIGRATION_NOTE.md；v7_reports/v7_p1_w1_eval500.md<br>历史台账：v7_reports/cleanup/CLEANUP_B_REPORT.md；v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261002-2329_v7p1dagger_w2` | 22.1M | 2026-10-03 01:22:07 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261002-2329_v7p1dagger_w3` | 22.1M | 2026-10-03 02:41:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261002-2329_v7p1dagger_w4` | 22.0M | 2026-10-03 04:01:09 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>叙述(裸名)：v7_reports/v7_struct_a2_wiring_probe.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `datasets/BTC20261007-1838_expert500val_v8` | 27.9M | 2026-10-07 19:09:30 | 无 | 无 |
| `datasets/BTC20261007-1838_expert5k_v8` | 277.9M | 2026-10-07 19:06:46 | 无 | 无 |

### 2.4 复现说明：被删 datasets 的生成工具线索（按名核对存在性）

| 数据集族 | 生成工具（存在性已核对） | 线索/口径 |
|---|---|---|
| expert5k / expert500val 全部（1327、0941_v4/v41、1838_v8、2202_v8） | **`tools/collect_expert.py`（存在，98KB；写 `expert_bc.npz`）** | 示例命令：`docs/v7_reports/v7_p1_iter2.md` L214-217（`--specs env/specs/scenarios_train.json --limit 5000 --out datasets/…_expert5k_v41`；eval500 用 `scenarios_eval500.json`）；1838 的 v8 采集命令见 `.slim/deepwork/v8_watch.log` |
| phase3_dagger_r1..r5（v6 P3 五轮窗口） | **`tools/dagger_collect.py`（存在，72KB）** + 池工具 **`tools/make_dagger_pools.py`（存在，8KB）** | 池 = `scenarios_train.json` ∩ expert5k `report.json`；三道隔离守卫口径见 `docs/archive/phase3_rootcause_analysis.md`、`docs/experiments.md` §359、`config/train.yaml::stages.B.phase3.collect` |
| v7p1dagger_w1..w4（v7 P1 窗口） | 同上（DAgger 链） | 生成/校验 provenance 见 `docs/v7_reports/v7_p1_dagger_cycle.md`（§2.1b 窗口校验 JSON；池 `scenarios_train_5k.json`、target-fails=1000、窗口 10s） |
| `tools/fit_plan_anchors.py`（存在） | 是 v41 的**消费者**（拟合 `config/plan_anchors_k6.json`），非生成者 | — |

复现成本提示：expert5k 采集 ~1683s/5k 行（1327 记录）；DAgger 窗口需重跑学生闭环（依赖 ckpt 与失败率），非即时可复现。

## 3. `runs/` Tier-1（时间戳 20260927~20261003 的 BTC*；拟删 170 项 / 527.6M）

按日分布：0927×1、0929×12、0930×1、1001×10、1002×113、1003×35。

### 3.1 硬引用命中 → 移入 KEEP（2 项 / 62.7M，**不放入删除计划**）

| 目录 | 体积 | mtime | 近1h写 | 引用检查 |
|---|---|---|---|---|
| `runs/BTC20260929-0425_fixA_nold` | 44.5M | 2026-10-06 11:05:46 | 无 | **硬引用**：config/arms/v7_struct_b_v5_train.yaml；config/train.yaml<br>叙述：LOCKS.md；archive/phase3_rootcause_analysis.md；archive/rl_stage_c_experiments.md；v6_program_prereg.md<br>叙述(裸名)：version_ledger.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-1631_v6p3` | 18.2M | 2026-10-01 19:12:08 | 无 | **硬引用**：config/arms/README.md；tools/reward_audit_collect.py<br>叙述：reward_audit/ebeta2/MANIFEST.md；reward_audit/ebeta2/reward_audit.json；reward_audit/ebeta2/reward_audit.md；reward_audit/ebeta2/reward_audit_swap.json；v6_program_prereg.md；v6_program_report.md；v6_reports/v6_p3_report.md；v6_reports/v6_seed11_attribution.md；v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |

- **`runs/BTC20260929-0425_fixA_nold`（44.5M）**：`config/train.yaml::stages.B.phase3.init_ckpt` 指向 `…/stage_b/final.pt`（v6/v7 phase3 基座）；`config/arms/v7_struct_b_v5_train.yaml` 同样引用（该 yaml 生成期间被并行会话归档至 `docs/archive/config_arms_legacy/`）。**属用户既有复现链口径（phase3 基座）**。
- **`runs/BTC20261001-1631_v6p3`（18.2M）**：`tools/reward_audit_collect.py` 示例命令（E-β″ 审计）指向 `…/stage_b/ckpt_epoch005.pt`；`config/arms/README.md` 引用。

### 3.2 拟删（170 项 / 527.6M）

- 引用检查结论：**硬引用 = 0**；96 项有非台账文档叙述引用（503.0M），74 项仅历史台账引用（24.6M）。叙述引用按任务口径不阻断删除，引用文档已逐项列出。
- **⚠️ 前版口径**：170/170 均被 2026-10-06 `CLEANUP_PLAN.md` 列为「保留」；其中体量最大的族：v6 P4/P3 族 53 项（203.9M）、1003 DAgger w 族 25 项（119.5M）、1002 v7p1i3 族 7 项（78.3M）、1002 v7p1b 族 10 项（62.1M）、1003 v7p2_s0 族 13 项（15.7M）、1002 v7p0 族 37 项（10.6M）。
- 单项目录全部列出如下：

| 目录 | 体积 | mtime | 近1h写 | 引用检查 |
|---|---|---|---|---|
| `runs/BTC20260927-1839_eval500_baseline` | 999.4K | 2026-10-06 11:05:46 | 无 | 叙述：experiments.md；v7_program_prereg.md；v7_program_report.md；v7_reports/v7_p0_gap_decomposition.md；v7_reports/v7_p0_idm_baseline.md；v7_reports/v7_p1_iter3.md；v7_reports/v7_p3fix_retry.md<br>叙述(裸名)：v7_reports/v7_p1_w1_eval500.md；v7_reports/v7_p2_s11_verify.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260929-095624_eval500_L2p1` | 1011.5K | 2026-10-06 11:05:46 | 无 | 叙述(裸名)：v6_reports/v6_cleanup_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md；v7_reports/cleanup/DELETED_MANIFEST.txt |
| `runs/BTC20260929-100314_eval500_L2p2` | 1011.2K | 2026-10-06 11:05:46 | 无 | 叙述：archive/phase3_rootcause_analysis.md；archive/rl_stage_c_experiments.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md；v7_reports/cleanup/DELETED_MANIFEST.txt |
| `runs/BTC20260929-182033_stageB_phase3_exp_beta_anchor1` | 8.6M | 2026-10-06 11:05:46 | 无 | 叙述：archive/phase3_rootcause_analysis.md；v6_program_prereg.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260929-183716_eval500_exp_beta_anchor1` | 1008.3K | 2026-10-06 11:05:46 | 无 | 叙述：archive/phase3_rootcause_analysis.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260929-185120_stageB_phase3_exp_ctrl_expert5k` | 8.6M | 2026-10-06 11:05:46 | 无 | 叙述：archive/phase3_rootcause_analysis.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260929-190640_eval500_exp_ctrl_expert5k` | 1013.3K | 2026-10-06 11:05:46 | 无 | 叙述：archive/phase3_rootcause_analysis.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260929-191846_stageB_phase3_exp_beta_anchor1_repro` | 8.6M | 2026-10-06 11:05:46 | 无 | 叙述(裸名)：v6_reports/v6_cleanup_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260929-193529_eval500_exp_beta_anchor1_repro` | 1008.2K | 2026-10-06 11:05:46 | 无 | 叙述：archive/phase3_rootcause_analysis.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260929-195151_eval500_phase3_base` | 1011.5K | 2026-10-06 11:05:46 | 无 | 叙述(裸名)：archive/rl_stage_c_experiments.md；v6_reports/v6_cleanup_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260929-224519_eval500_clean_ebeta` | 1005.7K | 2026-10-06 11:05:46 | 无 | 叙述：archive/rl_stage_c_experiments.md；v6_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260929-225323_eval500_clean_l2` | 1009.0K | 2026-10-06 11:05:46 | 无 | 叙述：archive/rl_stage_c_experiments.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20260930-0551_eval500_p3holdplan_clean` | 1001.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-1619_v6mini` | 43.7M | 2026-10-06 11:05:46 | 无 | 叙述：v6_program_report.md；v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-162637_v6mini_eval16` | 52.3K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-1631_v6retrain` | 47.6M | 2026-10-01 18:08:03 | 无 | 叙述：v6_program_report.md；v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-192937_v6p3_keepbest_ckpt_epoch005` | 1009.7K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-193831_v6p3_keepbest_final` | 1009.7K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-194721_v6p3_diag_b_final_clean500` | 1009.4K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-195447_v6p3_e3_clean500` | 1009.8K | 2026-10-06 11:05:46 | 无 | 叙述：v6_program_prereg.md；v6_program_report.md；v6_reports/v6_collision_arm_design.md；v6_reports/v6_p3_report.md；v6_reports/v6_p4_recheck.md；v6_reports/v6_p4_report.md；v7_program_report.md；v7_reports/v7_p1_iter3.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-200331_v6p3_e3_eval500` | 1010.3K | 2026-10-06 11:05:46 | 无 | 叙述：v6_program_prereg.md；v6_program_report.md；v6_reports/v6_collision_arm_design.md；v6_reports/v6_p3_report.md；v6_reports/v6_p4_report.md；v7_program_report.md；v7_reports/v7_p0_gap_decomposition.md；v7_reports/v7_p1_iter3.md<br>叙述(裸名)：v7_reports/v7_p1_w1_eval500.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261001-201519_v6p3_ref_ebeta_clean500` | 989.5K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p3_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0004_p4_arm0` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v6_program_report.md；v6_reports/v6_p4_incident.md；v6_reports/v6_p4_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4_arm0` | 14.9M | 2026-10-06 11:05:46 | 无 | 叙述：v6_program_report.md；v6_reports/v6_p4_report.md；v6_reports/v6_seed11_attribution.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_eval500` | 1006.6K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p0_gap_decomposition.md<br>叙述(裸名)：v6_reports/v6_p4_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u025_sub150` | 344.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u050_sub150` | 348.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u075_sub150` | 344.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u100_sub150` | 347.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u125_sub150` | 345.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u150_sub150` | 346.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u175_clean500` | 1003.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u175_sub150` | 342.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u200_clean500` | 1004.9K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p0_gap_decomposition.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0026_p4arm0_u200_sub150` | 343.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0148_p4_arm1` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p4_recheck.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0148_p4arm1_u050_sub150` | 346.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0206_p4_arm2` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p4_recheck.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0206_p4arm2_u050_sub150` | 342.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0223_p4_arm3` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p4_recheck.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0223_p4arm3_u050_sub150` | 346.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0241_p4_arm4` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p4_recheck.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0241_p4arm4_u050_sub150` | 346.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4_arm5` | 14.9M | 2026-10-06 11:05:46 | 无 | 叙述：v6_program_report.md；v6_reports/v6_p4_report.md；v6_reports/v6_seed11_attribution.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_eval500` | 1002.8K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p0_gap_decomposition.md<br>叙述(裸名)：v6_reports/v6_p4_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u025_sub150` | 344.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u050_sub150` | 348.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u075_sub150` | 344.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u100_sub150` | 346.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u125_sub150` | 345.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u150_sub150` | 345.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u175_clean500` | 1009.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u175_sub150` | 344.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u200_clean500` | 1002.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0303_p4arm5_u200_sub150` | 342.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0732_p4_arm0_seed11` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v6_program_report.md；v6_reports/v6_seed11_attribution.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0732_p4arm0s11_u050_sub150` | 343.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-075202_v6p4_recheck_arm1_u200_sub150` | 342.1K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p4_recheck.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-075618_v6p4_recheck_arm2_u200_sub150` | 342.5K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p4_recheck.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-080015_v6p4_recheck_arm3_u200_sub150` | 347.7K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p4_recheck.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-080336_v6p4_recheck_arm4_u200_sub150` | 346.5K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_p4_recheck.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-081746_v6p4_seed11_u150_sub150` | 346.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-082236_v6p4_seed11_u175_sub150` | 346.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-082712_v6p4_seed11_u200_sub150` | 345.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-083129_v6p4_seed11_u200_clean500` | 1011.1K | 2026-10-06 11:05:46 | 无 | 叙述：v6_reports/v6_seed11_attribution.md；v7_reports/v7_p0_gap_decomposition.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-084501_v6p4_seed11_u175_clean500` | 1015.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-0941_v7p1b` | 55.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p1_dagger_cycle.md；v7_reports/v7_p1_probe_v1v2.md；v7_reports/v7_p1b_chain.md；v7_reports/v7_p1b_failure_diag.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-100221_v7p0_idm_eval500_rerun` | 999.4K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p0_idm_baseline.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-100413_v7p0_idm_clean500` | 998.0K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p0_idm_baseline.md；v7_reports/v7_p1_iter3.md；v7_reports/v7_p3fix_retry.md<br>叙述(裸名)：v7_reports/v7_p2_s11_verify.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-100606_v7p0_idm_eval500_seedA` | 999.2K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p0_idm_baseline.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-100757_v7p0_idm_eval500_seedB` | 999.5K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p0_idm_baseline.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101123_v7p0_idm_baseline_eval500` | 999.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101320_v7p0_idm_eval500_rerun_clean` | 999.3K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p0_idm_baseline.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101322_v7p0_idm_baseline_clean500` | 998.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101716_v7p0_probe8_r1` | 40.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101727_v7p0_probe8_r2` | 40.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101739_v7p0_probe8_r3` | 40.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101750_v7p0_probe8_r4` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101802_v7p0_probe8_r5` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101824_v7p0_probe8_h0_r1` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101836_v7p0_probe8_h0_r2` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101847_v7p0_probe8_h0_r3` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101859_v7p0_probe8_h0_r4` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101910_v7p0_probe8_h0_r5` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-101954_v7p0_idm_eval500_hash0_r1` | 999.4K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p0_idm_baseline.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102146_v7p0_idm_eval500_hash0_r2` | 999.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102339_v7p0_idm_clean500_hash0` | 998.0K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p0_idm_baseline.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102728_v7p0_probe299_w1_r1` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102731_v7p0_probe299_w1_r2` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102733_v7p0_probe299_w1_r3` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102736_v7p0_probe299_w1_r4` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102738_v7p0_probe299_w1_r5` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102741_v7p0_probe299_w1_r6` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102743_v7p0_probe8_w1_r1` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102757_v7p0_probe8_w1_r2` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102811_v7p0_probe8_w1_r3` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102825_v7p0_probe8_w1_r4` | 40.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102839_v7p0_probe8_w1_r5` | 40.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-102958_v7p0_probe299_seeded_r1` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-103002_v7p0_probe299_seeded_r2` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-103006_v7p0_probe299_seeded_r3` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-103011_v7p0_probe299_seeded_r4` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-103016_v7p0_probe299_seeded_r5` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-103020_v7p0_probe299_seeded_r6` | 22.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-160114_v7p1b_tg45` | 111.9K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p3fix_retry.md<br>叙述(裸名)：v7_reports/v7_p2_s11_verify.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-160154_v7p1b_tg45_idm` | 112.2K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p1_iter3.md；v7_reports/v7_p3fix_retry.md<br>叙述(裸名)：v7_reports/v7_p2_s11_verify.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-160206_v7p1b_clean500` | 1009.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-160948_v7p1b_eval500` | 1009.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-162002_v7p1b_screen_epoch005` | 1006.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-162639_v7p1b_screen_epoch010` | 1008.5K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p1_dagger_cycle.md；v7_reports/v7_p1_iter3.md；v7_reports/v7_p3fix_retry.md<br>叙述(裸名)：v7_reports/v7_p2_s11_verify.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-163338_v7p1b_screen_epoch015` | 1007.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-164134_v7p1b_tg45_sel` | 112.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-164209_v7p1b_eval500_sel` | 1009.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p1_dagger_cycle.md；v7_reports/v7_p1_iter3.md；v7_reports/v7_p3fix_retry.md<br>叙述(裸名)：v7_reports/v7_p1_w1_eval500.md；v7_reports/v7_p2_s11_verify.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-1837_v7p1i3` | 73.2M | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_iter3.md；v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-211752_v7p1i3_screen_epoch005` | 1004.9K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_iter3.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-212641_v7p1i3_screen_epoch010` | 1012.4K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_iter3.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-213524_v7p1i3_screen_epoch015` | 1011.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_iter3.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-214503_v7p1i3_screen_final` | 1012.9K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_iter3.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-215549_v7p1i3_tg45_sel` | 112.6K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_iter3.md；v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-215634_v7p1i3_eval500_sel` | 1006.5K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_iter3.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-222821_v7p1probe_p1b_tg45_plan` | 112.3K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-222857_v7p1probe_p1b_tg45_repeat_action` | 112.8K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-222917_v7p1probe_i3_tg45_plan` | 112.5K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-223004_v7p1probe_i3_tg45_repeat_action` | 113.5K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-223027_v7p1probe_p1b_t3_plan` | 43.6K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-223041_v7p1probe_p1b_t3_repeat_action` | 43.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-223051_v7p1probe_i3_t3_plan` | 43.6K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-223109_v7p1probe_i3_t3_repeat_action` | 43.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-223120_v7p1probe_p1b_clean150_plan` | 344.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-223344_v7p1probe_p1b_clean150_repeat_action` | 345.1K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-223438_v7p1probe_i3_clean150_plan` | 343.2K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-223731_v7p1probe_i3_clean150_repeat_action` | 346.0K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_probe_v1v2.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-2329_v7p1dagger_w2` | 36.4M | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-2329_v7p1dagger_w3` | 36.4M | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261002-2329_v7p1dagger_w4` | 36.4M | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md；v7_reports/v7_struct_a2_wiring_probe.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-005306_v7p1dagger_w1_e005` | 345.2K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-005609_v7p1dagger_w1_e010` | 345.6K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-005853_v7p1dagger_w1_e015` | 346.2K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-021124_v7p1dagger_w2_e005` | 344.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-021410_v7p1dagger_w2_e010` | 345.5K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-021700_v7p1dagger_w2_e015` | 345.1K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-033138_v7p1dagger_w3_e005` | 345.6K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-033454_v7p1dagger_w3_e010` | 346.0K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-033755_v7p1dagger_w3_e015` | 345.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-045010_v7p1dagger_w4_e005` | 344.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-045300_v7p1dagger_w4_e010` | 346.5K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-045608_v7p1dagger_w4_e015` | 345.0K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-045912_v7p1dagger_w1_clean500` | 1009.9K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_prereg.md；v7_program_report.md；v7_reports/v7_p1_dagger_cycle.md；v7_reports/v7_p1_w1_eval500.md；v7_reports/v7_p3fix_retry.md<br>叙述(裸名)：v7_reports/v7_p2_s11_verify.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-050855_v7p1dagger_w1_tg45` | 112.3K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md；v7_reports/v7_p1_w1_eval500.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-051010_v7p1dagger_w2_clean500` | 1011.3K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-051910_v7p1dagger_w2_tg45` | 112.5K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-052024_v7p1dagger_w3_clean500` | 1012.4K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-053043_v7p1dagger_w3_tg45` | 113.2K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-053208_v7p1dagger_w4_clean500` | 1009.3K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-054109_v7p1dagger_w4_tg45` | 112.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-054238_v7p1dagger_w4_eval500` | 1011.1K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p1_dagger_cycle.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory` | 1010.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_prereg.md；v7_program_report.md；v7_reports/v7_p1_w1_eval500.md；v7_reports/v7_p3fix_retry.md<br>叙述(裸名)：v7_reports/v7_p2_s11_verify.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-0641_v7p2_s0_arm1` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p2_arm1.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-065707_v7p2_s0_u050_sub150` | 345.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-070038_v7p2_s0_u100_sub150` | 344.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-070430_v7p2_s0_u025_sub150` | 344.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-070745_v7p2_s0_u075_sub150` | 347.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-071123_v7p2_s0_u125_sub150` | 346.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-071530_v7p2_s0_u150_sub150` | 343.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-071904_v7p2_s0_u175_sub150` | 342.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-072227_v7p2_s0_u200_sub150` | 337.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-072348_v7p2_s0_u100_clean500` | 1008.2K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-073541_v7p2_s0_u050_clean500` | 1012.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-074639_v7p2_s0_tg45` | 112.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261003-074819_v7p2_s0_eval500` | 1008.4K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |

## 4. `runs/` Tier-2（时间戳 20261005/20261006 的 BTC*；135 项 / 349.4M；口径待定）

按日分布：20261005×70、20261006×65。`runs/BTC20261005-0601_v7p2_s11_arm1`（10M）已按任务 KEEP，不在表内。

- 引用检查结论：**硬引用 = 0**；27 项被 v7 复现链报告/证据文件引用（284.3M）——含 `v7_p2_s11_verify.md`、`v7_p3fix_retry.md`、`v7_struct_retrain_v5.md`、`v7_struct_fail_diag.md`、`v7_p4extra_collision.md`、`v7_p4extra_rewards.md`、`v7_s14_ttc.md`、`v7_night_watch.log`、`v7_reports/evidence/collision_replay.json` 等；108 项仅历史台账引用（65.0M）。
- **⚠️ 前版口径**：135/135 均被 2026-10-06 `CLEANUP_PLAN.md` 列为「保留」。
- **建议**：Tier-2 暂缓执行；若最终决定执行，建议先仅删「仅历史台账」的 108 项（65.0M），27 项有报告/证据引用者待 v7 报告归档决策后再处理。
- 单项目录全部列出如下：

| 目录 | 体积 | mtime | 近1h写 | 引用检查 |
|---|---|---|---|---|
| `runs/BTC20261005-061935_v7p2_s11_u050_sub150` | 345.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-062635_v7p2_s11_u100_sub150` | 344.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-063319_v7p2_s11_u025_sub150` | 344.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-063852_v7p2_s11_u075_sub150` | 342.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-064432_v7p2_s11_u125_sub150` | 344.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-065008_v7p2_s11_u150_sub150` | 342.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-065550_v7p2_s11_u175_sub150` | 343.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-070025_v7p2_s11_u200_sub150` | 342.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-070504_v7p2_s11_u150_clean500` | 1000.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-071745_v7p2_s11_u125_clean500` | 1.6M | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-071821_v7p2_s11_tg45` | 168.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-071840_v7p2_s11_eval500` | 1.6M | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-073128_v7p2_s11_u150_clean500_clean` | 1001.0K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p2_s11_verify.md；v7_reports/v7_p3fix_retry.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-074755_v7p2_s11_eval500_clean` | 1002.4K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/evidence/collision_replay.json；v7_reports/v7_night_watch.log；v7_reports/v7_p2_s11_verify.md；v7_reports/v7_p3fix_retry.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-075959_v7p2_s11_tg45_clean` | 112.7K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p2_s11_verify.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-0856_v7struct_v5` | 58.7M | 2026-10-05 10:39:49 | 无 | 叙述：v7_program_prereg.md；v7_program_report.md；v7_reports/v7_p3fix_retry.md；v7_reports/v7_struct_fail_diag.md；v7_reports/v7_struct_retrain_v5.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_D_DEADCODE.md；v7_reports/cleanup/CLEANUP_D_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-0856_v7struct_v5_p3` | 19.4M | 2026-10-05 13:06:47 | 无 | 叙述：v7_program_report.md；v7_reports/v7_struct_fail_diag.md；v7_reports/v7_struct_retrain_v5.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_C_REPORT.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-0856_v7struct_v5_p3fix` | 57.8M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p3fix_retry.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_D_DEADCODE.md；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-115035_v7sb_b_ckpt_epoch010_sub150` | 347.4K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p3fix_retry.md；v7_reports/v7_struct_fail_diag.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-120728_v7sb_b_ckpt_epoch010_clean500` | 1016.0K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-134526_v7sb_p3_ckpt_epoch005_sub150` | 342.5K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_struct_fail_diag.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-134825_v7sb_p3_ckpt_epoch005_clean500` | 1006.3K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_struct_retrain_v5.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-161813_v7p3fix_best_clean500` | 1015.4K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-163040_v7p3fix_best_tg45` | 113.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-163243_v7p3fix_best_eval500` | 1016.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-1724_v7p2_s1_arm1` | 15.3K | 2026-10-06 11:05:46 | 无 | 叙述：v7_reports/v7_p2_arm1.md<br>历史台账：v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-1732_v7p2_s1_arm1` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-174815_v7p2_s1_u050_sub150` | 343.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-175210_v7p2_s1_u100_sub150` | 342.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-175535_v7p2_s1_u025_sub150` | 344.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-175850_v7p2_s1_u075_sub150` | 342.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-180232_v7p2_s1_u125_sub150` | 342.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-180551_v7p2_s1_u150_sub150` | 341.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-180928_v7p2_s1_u175_sub150` | 343.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-181322_v7p2_s1_u200_sub150` | 343.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-181715_v7p2_s1_u050_clean500` | 1005.9K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-182951_v7p2_s1_u200_clean500` | 1006.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-184214_v7p2_s1_tg45` | 112.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-184357_v7p2_s1_eval500` | 1006.2K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-1856_v7p2_s2_arm1` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-191148_v7p2_s2_u050_sub150` | 344.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-191614_v7p2_s2_u100_sub150` | 345.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-192024_v7p2_s2_u025_sub150` | 347.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-192452_v7p2_s2_u075_sub150` | 342.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-192829_v7p2_s2_u125_sub150` | 343.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-193220_v7p2_s2_u150_sub150` | 344.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-193658_v7p2_s2_u175_sub150` | 340.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-193810_v7p2_s2_u200_sub150` | 342.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-194156_v7p2_s2_u200_clean500` | 1002.9K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-195358_v7p2_s2_u050_clean500` | 1009.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-200818_v7p2_s2_tg45` | 112.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-200952_v7p2_s2_eval500` | 1004.0K | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-2138_v7p4extra_s0_colls` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/MIGRATION_NOTE.md；v7_reports/v7_p4extra_collision.md<br>历史台账：v7_reports/cleanup/CLEANUP_B_REPORT.md；v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-214625_v7p4extra_s0_u050_sub150` | 341.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-215032_v7p4extra_s0_u100_sub150` | 341.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-215420_v7p4extra_s0_u025_sub150` | 342.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-215822_v7p4extra_s0_u075_sub150` | 341.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-220214_v7p4extra_s0_u075_clean500` | 1000.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-221530_v7p4extra_s0_u050_clean500` | 1000.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-222851_v7p4extra_s0_tg45` | 111.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-223034_v7p4extra_s0_eval500` | 1001.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-2243_v7p4extra_s11_colls` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_p4extra_collision.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-225205_v7p4extra_s11_u050_sub150` | 342.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-225625_v7p4extra_s11_u100_sub150` | 341.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-225950_v7p4extra_s11_u025_sub150` | 342.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-230407_v7p4extra_s11_u075_sub150` | 342.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-230803_v7p4extra_s11_u050_clean500` | 1001.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-232159_v7p4extra_s11_u075_clean500` | 1001.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-233442_v7p4extra_s11_tg45` | 111.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261005-233629_v7p4extra_s11_eval500` | 1001.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-0026_v7reward_smoke_B_u2` | 5.0M | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-0026_v7reward_smoke_C_u2` | 5.0M | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-0027_v7reward_A_s0` | 14.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_night_watch.log；v7_reports/v7_p4extra_rewards.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-003558_v7reward_A_s0_u050_sub150` | 341.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-003921_v7reward_A_s0_u100_sub150` | 341.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-004302_v7reward_A_s0_u025_sub150` | 341.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-004648_v7reward_A_s0_u075_sub150` | 341.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-005000_v7reward_A_s0_u025_clean500` | 1000.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-010209_v7reward_A_s0_u100_clean500` | 1000.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-011357_v7reward_A_s0_tg45` | 112.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-011533_v7reward_A_s0_eval500` | 1001.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-0127_v7reward_A_s11` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_night_watch.log；v7_reports/v7_p4extra_rewards.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-013655_v7reward_A_s11_u050_sub150` | 341.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-014049_v7reward_A_s11_u100_sub150` | 341.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-014500_v7reward_A_s11_u025_sub150` | 342.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-014839_v7reward_A_s11_u075_sub150` | 342.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-015301_v7reward_A_s11_u100_clean500` | 1000.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-020628_v7reward_A_s11_u050_clean500` | 1000.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-021843_v7reward_A_s11_tg45` | 111.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-022029_v7reward_A_s11_eval500` | 1002.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-0233_v7reward_B_s0` | 19.8M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_night_watch.log；v7_reports/v7_p4extra_rewards.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-024402_v7reward_B_s0_u050_sub150` | 342.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-024755_v7reward_B_s0_u100_sub150` | 341.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-025138_v7reward_B_s0_u025_sub150` | 342.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-025602_v7reward_B_s0_u075_sub150` | 342.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-030017_v7reward_B_s0_u025_clean500` | 1001.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-031432_v7reward_B_s0_u100_clean500` | 1000.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-032630_v7reward_B_s0_tg45` | 111.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-032820_v7reward_B_s0_eval500` | 1002.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-0342_v7reward_B_s11` | 19.8M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_night_watch.log；v7_reports/v7_p4extra_rewards.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-035224_v7reward_B_s11_u050_sub150` | 341.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-035521_v7reward_B_s11_u100_sub150` | 341.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-035858_v7reward_B_s11_u025_sub150` | 342.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-040258_v7reward_B_s11_u075_sub150` | 341.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-040633_v7reward_B_s11_u025_clean500` | 1001.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-041925_v7reward_B_s11_u100_clean500` | 1000.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-043100_v7reward_B_s11_tg45` | 112.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-043243_v7reward_B_s11_eval500` | 1003.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-0445_v7reward_C_s0` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_night_watch.log；v7_reports/v7_p4extra_rewards.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-045357_v7reward_C_s0_u050_sub150` | 341.3K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-045743_v7reward_C_s0_u100_sub150` | 341.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-050139_v7reward_C_s0_u025_sub150` | 341.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-050601_v7reward_C_s0_u075_sub150` | 341.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-050959_v7reward_C_s0_u075_clean500` | 1000.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-052250_v7reward_C_s0_u025_clean500` | 1000.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-053711_v7reward_C_s0_tg45` | 111.5K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-053907_v7reward_C_s0_eval500` | 1001.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-0615_v7s14_s0_ttc` | 14.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/v7_night_watch.log；v7_reports/v7_s14_ttc.md<br>历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-062311_v7s14_s0_u050_sub150` | 341.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-062631_v7s14_s0_u100_sub150` | 342.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-062959_v7s14_s0_u025_sub150` | 342.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-063340_v7s14_s0_u075_sub150` | 341.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-063713_v7s14_s0_u075_clean500` | 999.4K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-064832_v7s14_s0_u050_clean500` | 999.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-065914_v7s14_s0_tg45` | 112.0K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-070046_v7s14_s0_eval500` | 1000.9K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-0711_v7s14_s11_ttc` | 9.9M | 2026-10-06 11:05:46 | 无 | 叙述：v7_program_report.md；v7_reports/MIGRATION_NOTE.md；v7_reports/v7_s14_ttc.md<br>历史台账：v7_reports/cleanup/CLEANUP_B_REPORT.md；v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-071927_v7s14_s11_u050_sub150` | 341.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-072303_v7s14_s11_u100_sub150` | 341.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-072547_v7s14_s11_u025_sub150` | 341.7K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-072944_v7s14_s11_u075_sub150` | 341.6K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-073318_v7s14_s11_u025_clean500` | 1000.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-074550_v7s14_s11_u050_clean500` | 1000.1K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-075739_v7s14_s11_tg45` | 112.2K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |
| `runs/BTC20261006-075920_v7s14_s11_eval500` | 1001.8K | 2026-10-06 11:05:46 | 无 | 历史台账：v7_reports/cleanup/CLEANUP_C_DELETED.txt；v7_reports/cleanup/CLEANUP_PLAN.md |

## 5. 建议删除命令（逐项 `rm -rf`；严禁通配符批量命令）

> **本阶段不执行**。执行前须：① 人工确认范围与口径冲突；② 再次确认无 20261009/活跃进程触碰目标；③ 建议逐段执行并留存本清单为删除记录。Tier-2 段默认**暂缓**。

### A. datasets/（16 项，1.1G）

```bash
rm -rf 'datasets/BTC20260929-1357_phase3_dagger_r1'
rm -rf 'datasets/BTC20260929-1357_phase3_dagger_r2'
rm -rf 'datasets/BTC20260929-1357_phase3_dagger_r3'
rm -rf 'datasets/BTC20260929-1357_phase3_dagger_r4'
rm -rf 'datasets/BTC20260929-1357_phase3_dagger_r5'
rm -rf 'datasets/BTC20261001-1327_expert500val'
rm -rf 'datasets/BTC20261001-1327_expert5k'
rm -rf 'datasets/BTC20261002-0941_expert500val_v4'
rm -rf 'datasets/BTC20261002-0941_expert500val_v41'
rm -rf 'datasets/BTC20261002-0941_expert5k_v4'
rm -rf 'datasets/BTC20261002-2329_v7p1dagger_w1'
rm -rf 'datasets/BTC20261002-2329_v7p1dagger_w2'
rm -rf 'datasets/BTC20261002-2329_v7p1dagger_w3'
rm -rf 'datasets/BTC20261002-2329_v7p1dagger_w4'
rm -rf 'datasets/BTC20261007-1838_expert500val_v8'
rm -rf 'datasets/BTC20261007-1838_expert5k_v8'
```

### B. runs/ Tier-1（170 项，527.6M）

```bash
rm -rf 'runs/BTC20260927-1839_eval500_baseline'
rm -rf 'runs/BTC20260929-095624_eval500_L2p1'
rm -rf 'runs/BTC20260929-100314_eval500_L2p2'
rm -rf 'runs/BTC20260929-182033_stageB_phase3_exp_beta_anchor1'
rm -rf 'runs/BTC20260929-183716_eval500_exp_beta_anchor1'
rm -rf 'runs/BTC20260929-185120_stageB_phase3_exp_ctrl_expert5k'
rm -rf 'runs/BTC20260929-190640_eval500_exp_ctrl_expert5k'
rm -rf 'runs/BTC20260929-191846_stageB_phase3_exp_beta_anchor1_repro'
rm -rf 'runs/BTC20260929-193529_eval500_exp_beta_anchor1_repro'
rm -rf 'runs/BTC20260929-195151_eval500_phase3_base'
rm -rf 'runs/BTC20260929-224519_eval500_clean_ebeta'
rm -rf 'runs/BTC20260929-225323_eval500_clean_l2'
rm -rf 'runs/BTC20260930-0551_eval500_p3holdplan_clean'
rm -rf 'runs/BTC20261001-1619_v6mini'
rm -rf 'runs/BTC20261001-162637_v6mini_eval16'
rm -rf 'runs/BTC20261001-1631_v6retrain'
rm -rf 'runs/BTC20261001-192937_v6p3_keepbest_ckpt_epoch005'
rm -rf 'runs/BTC20261001-193831_v6p3_keepbest_final'
rm -rf 'runs/BTC20261001-194721_v6p3_diag_b_final_clean500'
rm -rf 'runs/BTC20261001-195447_v6p3_e3_clean500'
rm -rf 'runs/BTC20261001-200331_v6p3_e3_eval500'
rm -rf 'runs/BTC20261001-201519_v6p3_ref_ebeta_clean500'
rm -rf 'runs/BTC20261002-0004_p4_arm0'
rm -rf 'runs/BTC20261002-0026_p4_arm0'
rm -rf 'runs/BTC20261002-0026_p4arm0_eval500'
rm -rf 'runs/BTC20261002-0026_p4arm0_u025_sub150'
rm -rf 'runs/BTC20261002-0026_p4arm0_u050_sub150'
rm -rf 'runs/BTC20261002-0026_p4arm0_u075_sub150'
rm -rf 'runs/BTC20261002-0026_p4arm0_u100_sub150'
rm -rf 'runs/BTC20261002-0026_p4arm0_u125_sub150'
rm -rf 'runs/BTC20261002-0026_p4arm0_u150_sub150'
rm -rf 'runs/BTC20261002-0026_p4arm0_u175_clean500'
rm -rf 'runs/BTC20261002-0026_p4arm0_u175_sub150'
rm -rf 'runs/BTC20261002-0026_p4arm0_u200_clean500'
rm -rf 'runs/BTC20261002-0026_p4arm0_u200_sub150'
rm -rf 'runs/BTC20261002-0148_p4_arm1'
rm -rf 'runs/BTC20261002-0148_p4arm1_u050_sub150'
rm -rf 'runs/BTC20261002-0206_p4_arm2'
rm -rf 'runs/BTC20261002-0206_p4arm2_u050_sub150'
rm -rf 'runs/BTC20261002-0223_p4_arm3'
rm -rf 'runs/BTC20261002-0223_p4arm3_u050_sub150'
rm -rf 'runs/BTC20261002-0241_p4_arm4'
rm -rf 'runs/BTC20261002-0241_p4arm4_u050_sub150'
rm -rf 'runs/BTC20261002-0303_p4_arm5'
rm -rf 'runs/BTC20261002-0303_p4arm5_eval500'
rm -rf 'runs/BTC20261002-0303_p4arm5_u025_sub150'
rm -rf 'runs/BTC20261002-0303_p4arm5_u050_sub150'
rm -rf 'runs/BTC20261002-0303_p4arm5_u075_sub150'
rm -rf 'runs/BTC20261002-0303_p4arm5_u100_sub150'
rm -rf 'runs/BTC20261002-0303_p4arm5_u125_sub150'
rm -rf 'runs/BTC20261002-0303_p4arm5_u150_sub150'
rm -rf 'runs/BTC20261002-0303_p4arm5_u175_clean500'
rm -rf 'runs/BTC20261002-0303_p4arm5_u175_sub150'
rm -rf 'runs/BTC20261002-0303_p4arm5_u200_clean500'
rm -rf 'runs/BTC20261002-0303_p4arm5_u200_sub150'
rm -rf 'runs/BTC20261002-0732_p4_arm0_seed11'
rm -rf 'runs/BTC20261002-0732_p4arm0s11_u050_sub150'
rm -rf 'runs/BTC20261002-075202_v6p4_recheck_arm1_u200_sub150'
rm -rf 'runs/BTC20261002-075618_v6p4_recheck_arm2_u200_sub150'
rm -rf 'runs/BTC20261002-080015_v6p4_recheck_arm3_u200_sub150'
rm -rf 'runs/BTC20261002-080336_v6p4_recheck_arm4_u200_sub150'
rm -rf 'runs/BTC20261002-081746_v6p4_seed11_u150_sub150'
rm -rf 'runs/BTC20261002-082236_v6p4_seed11_u175_sub150'
rm -rf 'runs/BTC20261002-082712_v6p4_seed11_u200_sub150'
rm -rf 'runs/BTC20261002-083129_v6p4_seed11_u200_clean500'
rm -rf 'runs/BTC20261002-084501_v6p4_seed11_u175_clean500'
rm -rf 'runs/BTC20261002-0941_v7p1b'
rm -rf 'runs/BTC20261002-100221_v7p0_idm_eval500_rerun'
rm -rf 'runs/BTC20261002-100413_v7p0_idm_clean500'
rm -rf 'runs/BTC20261002-100606_v7p0_idm_eval500_seedA'
rm -rf 'runs/BTC20261002-100757_v7p0_idm_eval500_seedB'
rm -rf 'runs/BTC20261002-101123_v7p0_idm_baseline_eval500'
rm -rf 'runs/BTC20261002-101320_v7p0_idm_eval500_rerun_clean'
rm -rf 'runs/BTC20261002-101322_v7p0_idm_baseline_clean500'
rm -rf 'runs/BTC20261002-101716_v7p0_probe8_r1'
rm -rf 'runs/BTC20261002-101727_v7p0_probe8_r2'
rm -rf 'runs/BTC20261002-101739_v7p0_probe8_r3'
rm -rf 'runs/BTC20261002-101750_v7p0_probe8_r4'
rm -rf 'runs/BTC20261002-101802_v7p0_probe8_r5'
rm -rf 'runs/BTC20261002-101824_v7p0_probe8_h0_r1'
rm -rf 'runs/BTC20261002-101836_v7p0_probe8_h0_r2'
rm -rf 'runs/BTC20261002-101847_v7p0_probe8_h0_r3'
rm -rf 'runs/BTC20261002-101859_v7p0_probe8_h0_r4'
rm -rf 'runs/BTC20261002-101910_v7p0_probe8_h0_r5'
rm -rf 'runs/BTC20261002-101954_v7p0_idm_eval500_hash0_r1'
rm -rf 'runs/BTC20261002-102146_v7p0_idm_eval500_hash0_r2'
rm -rf 'runs/BTC20261002-102339_v7p0_idm_clean500_hash0'
rm -rf 'runs/BTC20261002-102728_v7p0_probe299_w1_r1'
rm -rf 'runs/BTC20261002-102731_v7p0_probe299_w1_r2'
rm -rf 'runs/BTC20261002-102733_v7p0_probe299_w1_r3'
rm -rf 'runs/BTC20261002-102736_v7p0_probe299_w1_r4'
rm -rf 'runs/BTC20261002-102738_v7p0_probe299_w1_r5'
rm -rf 'runs/BTC20261002-102741_v7p0_probe299_w1_r6'
rm -rf 'runs/BTC20261002-102743_v7p0_probe8_w1_r1'
rm -rf 'runs/BTC20261002-102757_v7p0_probe8_w1_r2'
rm -rf 'runs/BTC20261002-102811_v7p0_probe8_w1_r3'
rm -rf 'runs/BTC20261002-102825_v7p0_probe8_w1_r4'
rm -rf 'runs/BTC20261002-102839_v7p0_probe8_w1_r5'
rm -rf 'runs/BTC20261002-102958_v7p0_probe299_seeded_r1'
rm -rf 'runs/BTC20261002-103002_v7p0_probe299_seeded_r2'
rm -rf 'runs/BTC20261002-103006_v7p0_probe299_seeded_r3'
rm -rf 'runs/BTC20261002-103011_v7p0_probe299_seeded_r4'
rm -rf 'runs/BTC20261002-103016_v7p0_probe299_seeded_r5'
rm -rf 'runs/BTC20261002-103020_v7p0_probe299_seeded_r6'
rm -rf 'runs/BTC20261002-160114_v7p1b_tg45'
rm -rf 'runs/BTC20261002-160154_v7p1b_tg45_idm'
rm -rf 'runs/BTC20261002-160206_v7p1b_clean500'
rm -rf 'runs/BTC20261002-160948_v7p1b_eval500'
rm -rf 'runs/BTC20261002-162002_v7p1b_screen_epoch005'
rm -rf 'runs/BTC20261002-162639_v7p1b_screen_epoch010'
rm -rf 'runs/BTC20261002-163338_v7p1b_screen_epoch015'
rm -rf 'runs/BTC20261002-164134_v7p1b_tg45_sel'
rm -rf 'runs/BTC20261002-164209_v7p1b_eval500_sel'
rm -rf 'runs/BTC20261002-1837_v7p1i3'
rm -rf 'runs/BTC20261002-211752_v7p1i3_screen_epoch005'
rm -rf 'runs/BTC20261002-212641_v7p1i3_screen_epoch010'
rm -rf 'runs/BTC20261002-213524_v7p1i3_screen_epoch015'
rm -rf 'runs/BTC20261002-214503_v7p1i3_screen_final'
rm -rf 'runs/BTC20261002-215549_v7p1i3_tg45_sel'
rm -rf 'runs/BTC20261002-215634_v7p1i3_eval500_sel'
rm -rf 'runs/BTC20261002-222821_v7p1probe_p1b_tg45_plan'
rm -rf 'runs/BTC20261002-222857_v7p1probe_p1b_tg45_repeat_action'
rm -rf 'runs/BTC20261002-222917_v7p1probe_i3_tg45_plan'
rm -rf 'runs/BTC20261002-223004_v7p1probe_i3_tg45_repeat_action'
rm -rf 'runs/BTC20261002-223027_v7p1probe_p1b_t3_plan'
rm -rf 'runs/BTC20261002-223041_v7p1probe_p1b_t3_repeat_action'
rm -rf 'runs/BTC20261002-223051_v7p1probe_i3_t3_plan'
rm -rf 'runs/BTC20261002-223109_v7p1probe_i3_t3_repeat_action'
rm -rf 'runs/BTC20261002-223120_v7p1probe_p1b_clean150_plan'
rm -rf 'runs/BTC20261002-223344_v7p1probe_p1b_clean150_repeat_action'
rm -rf 'runs/BTC20261002-223438_v7p1probe_i3_clean150_plan'
rm -rf 'runs/BTC20261002-223731_v7p1probe_i3_clean150_repeat_action'
rm -rf 'runs/BTC20261002-2329_v7p1dagger_w2'
rm -rf 'runs/BTC20261002-2329_v7p1dagger_w3'
rm -rf 'runs/BTC20261002-2329_v7p1dagger_w4'
rm -rf 'runs/BTC20261003-005306_v7p1dagger_w1_e005'
rm -rf 'runs/BTC20261003-005609_v7p1dagger_w1_e010'
rm -rf 'runs/BTC20261003-005853_v7p1dagger_w1_e015'
rm -rf 'runs/BTC20261003-021124_v7p1dagger_w2_e005'
rm -rf 'runs/BTC20261003-021410_v7p1dagger_w2_e010'
rm -rf 'runs/BTC20261003-021700_v7p1dagger_w2_e015'
rm -rf 'runs/BTC20261003-033138_v7p1dagger_w3_e005'
rm -rf 'runs/BTC20261003-033454_v7p1dagger_w3_e010'
rm -rf 'runs/BTC20261003-033755_v7p1dagger_w3_e015'
rm -rf 'runs/BTC20261003-045010_v7p1dagger_w4_e005'
rm -rf 'runs/BTC20261003-045300_v7p1dagger_w4_e010'
rm -rf 'runs/BTC20261003-045608_v7p1dagger_w4_e015'
rm -rf 'runs/BTC20261003-045912_v7p1dagger_w1_clean500'
rm -rf 'runs/BTC20261003-050855_v7p1dagger_w1_tg45'
rm -rf 'runs/BTC20261003-051010_v7p1dagger_w2_clean500'
rm -rf 'runs/BTC20261003-051910_v7p1dagger_w2_tg45'
rm -rf 'runs/BTC20261003-052024_v7p1dagger_w3_clean500'
rm -rf 'runs/BTC20261003-053043_v7p1dagger_w3_tg45'
rm -rf 'runs/BTC20261003-053208_v7p1dagger_w4_clean500'
rm -rf 'runs/BTC20261003-054109_v7p1dagger_w4_tg45'
rm -rf 'runs/BTC20261003-054238_v7p1dagger_w4_eval500'
rm -rf 'runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory'
rm -rf 'runs/BTC20261003-0641_v7p2_s0_arm1'
rm -rf 'runs/BTC20261003-065707_v7p2_s0_u050_sub150'
rm -rf 'runs/BTC20261003-070038_v7p2_s0_u100_sub150'
rm -rf 'runs/BTC20261003-070430_v7p2_s0_u025_sub150'
rm -rf 'runs/BTC20261003-070745_v7p2_s0_u075_sub150'
rm -rf 'runs/BTC20261003-071123_v7p2_s0_u125_sub150'
rm -rf 'runs/BTC20261003-071530_v7p2_s0_u150_sub150'
rm -rf 'runs/BTC20261003-071904_v7p2_s0_u175_sub150'
rm -rf 'runs/BTC20261003-072227_v7p2_s0_u200_sub150'
rm -rf 'runs/BTC20261003-072348_v7p2_s0_u100_clean500'
rm -rf 'runs/BTC20261003-073541_v7p2_s0_u050_clean500'
rm -rf 'runs/BTC20261003-074639_v7p2_s0_tg45'
rm -rf 'runs/BTC20261003-074819_v7p2_s0_eval500'
```

### C. runs/ Tier-2（135 项，349.4M）—— 口径待定，暂缓执行

```bash
rm -rf 'runs/BTC20261005-061935_v7p2_s11_u050_sub150'
rm -rf 'runs/BTC20261005-062635_v7p2_s11_u100_sub150'
rm -rf 'runs/BTC20261005-063319_v7p2_s11_u025_sub150'
rm -rf 'runs/BTC20261005-063852_v7p2_s11_u075_sub150'
rm -rf 'runs/BTC20261005-064432_v7p2_s11_u125_sub150'
rm -rf 'runs/BTC20261005-065008_v7p2_s11_u150_sub150'
rm -rf 'runs/BTC20261005-065550_v7p2_s11_u175_sub150'
rm -rf 'runs/BTC20261005-070025_v7p2_s11_u200_sub150'
rm -rf 'runs/BTC20261005-070504_v7p2_s11_u150_clean500'
rm -rf 'runs/BTC20261005-071745_v7p2_s11_u125_clean500'
rm -rf 'runs/BTC20261005-071821_v7p2_s11_tg45'
rm -rf 'runs/BTC20261005-071840_v7p2_s11_eval500'
rm -rf 'runs/BTC20261005-073128_v7p2_s11_u150_clean500_clean'
rm -rf 'runs/BTC20261005-074755_v7p2_s11_eval500_clean'
rm -rf 'runs/BTC20261005-075959_v7p2_s11_tg45_clean'
rm -rf 'runs/BTC20261005-0856_v7struct_v5'
rm -rf 'runs/BTC20261005-0856_v7struct_v5_p3'
rm -rf 'runs/BTC20261005-0856_v7struct_v5_p3fix'
rm -rf 'runs/BTC20261005-115035_v7sb_b_ckpt_epoch010_sub150'
rm -rf 'runs/BTC20261005-120728_v7sb_b_ckpt_epoch010_clean500'
rm -rf 'runs/BTC20261005-134526_v7sb_p3_ckpt_epoch005_sub150'
rm -rf 'runs/BTC20261005-134825_v7sb_p3_ckpt_epoch005_clean500'
rm -rf 'runs/BTC20261005-161813_v7p3fix_best_clean500'
rm -rf 'runs/BTC20261005-163040_v7p3fix_best_tg45'
rm -rf 'runs/BTC20261005-163243_v7p3fix_best_eval500'
rm -rf 'runs/BTC20261005-1724_v7p2_s1_arm1'
rm -rf 'runs/BTC20261005-1732_v7p2_s1_arm1'
rm -rf 'runs/BTC20261005-174815_v7p2_s1_u050_sub150'
rm -rf 'runs/BTC20261005-175210_v7p2_s1_u100_sub150'
rm -rf 'runs/BTC20261005-175535_v7p2_s1_u025_sub150'
rm -rf 'runs/BTC20261005-175850_v7p2_s1_u075_sub150'
rm -rf 'runs/BTC20261005-180232_v7p2_s1_u125_sub150'
rm -rf 'runs/BTC20261005-180551_v7p2_s1_u150_sub150'
rm -rf 'runs/BTC20261005-180928_v7p2_s1_u175_sub150'
rm -rf 'runs/BTC20261005-181322_v7p2_s1_u200_sub150'
rm -rf 'runs/BTC20261005-181715_v7p2_s1_u050_clean500'
rm -rf 'runs/BTC20261005-182951_v7p2_s1_u200_clean500'
rm -rf 'runs/BTC20261005-184214_v7p2_s1_tg45'
rm -rf 'runs/BTC20261005-184357_v7p2_s1_eval500'
rm -rf 'runs/BTC20261005-1856_v7p2_s2_arm1'
rm -rf 'runs/BTC20261005-191148_v7p2_s2_u050_sub150'
rm -rf 'runs/BTC20261005-191614_v7p2_s2_u100_sub150'
rm -rf 'runs/BTC20261005-192024_v7p2_s2_u025_sub150'
rm -rf 'runs/BTC20261005-192452_v7p2_s2_u075_sub150'
rm -rf 'runs/BTC20261005-192829_v7p2_s2_u125_sub150'
rm -rf 'runs/BTC20261005-193220_v7p2_s2_u150_sub150'
rm -rf 'runs/BTC20261005-193658_v7p2_s2_u175_sub150'
rm -rf 'runs/BTC20261005-193810_v7p2_s2_u200_sub150'
rm -rf 'runs/BTC20261005-194156_v7p2_s2_u200_clean500'
rm -rf 'runs/BTC20261005-195358_v7p2_s2_u050_clean500'
rm -rf 'runs/BTC20261005-200818_v7p2_s2_tg45'
rm -rf 'runs/BTC20261005-200952_v7p2_s2_eval500'
rm -rf 'runs/BTC20261005-2138_v7p4extra_s0_colls'
rm -rf 'runs/BTC20261005-214625_v7p4extra_s0_u050_sub150'
rm -rf 'runs/BTC20261005-215032_v7p4extra_s0_u100_sub150'
rm -rf 'runs/BTC20261005-215420_v7p4extra_s0_u025_sub150'
rm -rf 'runs/BTC20261005-215822_v7p4extra_s0_u075_sub150'
rm -rf 'runs/BTC20261005-220214_v7p4extra_s0_u075_clean500'
rm -rf 'runs/BTC20261005-221530_v7p4extra_s0_u050_clean500'
rm -rf 'runs/BTC20261005-222851_v7p4extra_s0_tg45'
rm -rf 'runs/BTC20261005-223034_v7p4extra_s0_eval500'
rm -rf 'runs/BTC20261005-2243_v7p4extra_s11_colls'
rm -rf 'runs/BTC20261005-225205_v7p4extra_s11_u050_sub150'
rm -rf 'runs/BTC20261005-225625_v7p4extra_s11_u100_sub150'
rm -rf 'runs/BTC20261005-225950_v7p4extra_s11_u025_sub150'
rm -rf 'runs/BTC20261005-230407_v7p4extra_s11_u075_sub150'
rm -rf 'runs/BTC20261005-230803_v7p4extra_s11_u050_clean500'
rm -rf 'runs/BTC20261005-232159_v7p4extra_s11_u075_clean500'
rm -rf 'runs/BTC20261005-233442_v7p4extra_s11_tg45'
rm -rf 'runs/BTC20261005-233629_v7p4extra_s11_eval500'
rm -rf 'runs/BTC20261006-0026_v7reward_smoke_B_u2'
rm -rf 'runs/BTC20261006-0026_v7reward_smoke_C_u2'
rm -rf 'runs/BTC20261006-0027_v7reward_A_s0'
rm -rf 'runs/BTC20261006-003558_v7reward_A_s0_u050_sub150'
rm -rf 'runs/BTC20261006-003921_v7reward_A_s0_u100_sub150'
rm -rf 'runs/BTC20261006-004302_v7reward_A_s0_u025_sub150'
rm -rf 'runs/BTC20261006-004648_v7reward_A_s0_u075_sub150'
rm -rf 'runs/BTC20261006-005000_v7reward_A_s0_u025_clean500'
rm -rf 'runs/BTC20261006-010209_v7reward_A_s0_u100_clean500'
rm -rf 'runs/BTC20261006-011357_v7reward_A_s0_tg45'
rm -rf 'runs/BTC20261006-011533_v7reward_A_s0_eval500'
rm -rf 'runs/BTC20261006-0127_v7reward_A_s11'
rm -rf 'runs/BTC20261006-013655_v7reward_A_s11_u050_sub150'
rm -rf 'runs/BTC20261006-014049_v7reward_A_s11_u100_sub150'
rm -rf 'runs/BTC20261006-014500_v7reward_A_s11_u025_sub150'
rm -rf 'runs/BTC20261006-014839_v7reward_A_s11_u075_sub150'
rm -rf 'runs/BTC20261006-015301_v7reward_A_s11_u100_clean500'
rm -rf 'runs/BTC20261006-020628_v7reward_A_s11_u050_clean500'
rm -rf 'runs/BTC20261006-021843_v7reward_A_s11_tg45'
rm -rf 'runs/BTC20261006-022029_v7reward_A_s11_eval500'
rm -rf 'runs/BTC20261006-0233_v7reward_B_s0'
rm -rf 'runs/BTC20261006-024402_v7reward_B_s0_u050_sub150'
rm -rf 'runs/BTC20261006-024755_v7reward_B_s0_u100_sub150'
rm -rf 'runs/BTC20261006-025138_v7reward_B_s0_u025_sub150'
rm -rf 'runs/BTC20261006-025602_v7reward_B_s0_u075_sub150'
rm -rf 'runs/BTC20261006-030017_v7reward_B_s0_u025_clean500'
rm -rf 'runs/BTC20261006-031432_v7reward_B_s0_u100_clean500'
rm -rf 'runs/BTC20261006-032630_v7reward_B_s0_tg45'
rm -rf 'runs/BTC20261006-032820_v7reward_B_s0_eval500'
rm -rf 'runs/BTC20261006-0342_v7reward_B_s11'
rm -rf 'runs/BTC20261006-035224_v7reward_B_s11_u050_sub150'
rm -rf 'runs/BTC20261006-035521_v7reward_B_s11_u100_sub150'
rm -rf 'runs/BTC20261006-035858_v7reward_B_s11_u025_sub150'
rm -rf 'runs/BTC20261006-040258_v7reward_B_s11_u075_sub150'
rm -rf 'runs/BTC20261006-040633_v7reward_B_s11_u025_clean500'
rm -rf 'runs/BTC20261006-041925_v7reward_B_s11_u100_clean500'
rm -rf 'runs/BTC20261006-043100_v7reward_B_s11_tg45'
rm -rf 'runs/BTC20261006-043243_v7reward_B_s11_eval500'
rm -rf 'runs/BTC20261006-0445_v7reward_C_s0'
rm -rf 'runs/BTC20261006-045357_v7reward_C_s0_u050_sub150'
rm -rf 'runs/BTC20261006-045743_v7reward_C_s0_u100_sub150'
rm -rf 'runs/BTC20261006-050139_v7reward_C_s0_u025_sub150'
rm -rf 'runs/BTC20261006-050601_v7reward_C_s0_u075_sub150'
rm -rf 'runs/BTC20261006-050959_v7reward_C_s0_u075_clean500'
rm -rf 'runs/BTC20261006-052250_v7reward_C_s0_u025_clean500'
rm -rf 'runs/BTC20261006-053711_v7reward_C_s0_tg45'
rm -rf 'runs/BTC20261006-053907_v7reward_C_s0_eval500'
rm -rf 'runs/BTC20261006-0615_v7s14_s0_ttc'
rm -rf 'runs/BTC20261006-062311_v7s14_s0_u050_sub150'
rm -rf 'runs/BTC20261006-062631_v7s14_s0_u100_sub150'
rm -rf 'runs/BTC20261006-062959_v7s14_s0_u025_sub150'
rm -rf 'runs/BTC20261006-063340_v7s14_s0_u075_sub150'
rm -rf 'runs/BTC20261006-063713_v7s14_s0_u075_clean500'
rm -rf 'runs/BTC20261006-064832_v7s14_s0_u050_clean500'
rm -rf 'runs/BTC20261006-065914_v7s14_s0_tg45'
rm -rf 'runs/BTC20261006-070046_v7s14_s0_eval500'
rm -rf 'runs/BTC20261006-0711_v7s14_s11_ttc'
rm -rf 'runs/BTC20261006-071927_v7s14_s11_u050_sub150'
rm -rf 'runs/BTC20261006-072303_v7s14_s11_u100_sub150'
rm -rf 'runs/BTC20261006-072547_v7s14_s11_u025_sub150'
rm -rf 'runs/BTC20261006-072944_v7s14_s11_u075_sub150'
rm -rf 'runs/BTC20261006-073318_v7s14_s11_u025_clean500'
rm -rf 'runs/BTC20261006-074550_v7s14_s11_u050_clean500'
rm -rf 'runs/BTC20261006-075739_v7s14_s11_tg45'
rm -rf 'runs/BTC20261006-075920_v7s14_s11_eval500'
```

## 6. 无法判断 / 需人工确认清单

1. **`datasets/BTC20261002-0941_expert5k_v41`（286.2M）**：硬引用（`tools/fit_plan_anchors.py` 默认数据路径 + `config/plan_anchors_k6.json` provenance + `net/anchor.py` 注释）。裁决问题：v7 结构线是否彻底关闭、是否还需重拟合锚字典。若不删，建议在工具注释中改默认路径以免长期断链。
2. **`runs/` Tier-2 全量（135 项 / 349.4M）**：v7 复现链证据边界（27 项有报告/证据引用，含证据 JSON `collision_replay.json`）；口径待定。
3. **`datasets/BTC20261007-1838_expert5k_v8` + `_expert500val_v8`（305.8M）**：当前 KEEP 的 `runs/BTC20261007-1838_v8chain` 的训练数据；被 `.slim` 历史日志/台账引用；删除 = 该链不可复训。虽已按任务列入拟删（A 段命令），建议人工再裁决是否保留为 v8 首链证据。
4. **Tier-1 全量与上一版锁版口径全面冲突（170/170 曾被列为保留）**：特别是大项 `BTC20261002-1837_v7p1i3`（73.2M）、`BTC20261002-0941_v7p1b`（55.9M）、`BTC20261001-1631_v6retrain`（47.6M）、`BTC20261001-1619_v6mini`（43.7M）、`v7p1dagger_w2..w4`（各 36.4M）、1003 的 v7p2_s0 族（15.7M，属 v7 P2 代而非 p1，按时间戳落入 Tier-1）。若 v6/v7 报告将随本次清理一并归档/降级，则冲突消解；否则建议保留大项。
5. **`runs/BTC20261005-074755_v7p2_s11_eval500_clean`（1.7M，Tier-2）**：被证据文件 `docs/v7_reports/evidence/collision_replay.json` 与 `v7_night_watch.log` 引用 → Tier-2 内优先人工确认项。
6. **`.slim` 非 `.sh` 文件叙述引用（警告，不阻断）**：`runs/BTC20261002-1837_v7p1i3`、`runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory`（`v7-beat-idm.md`）；`datasets/BTC20261001-1327_expert5k`（`v6-net-retrain.md`）；`datasets/BTC20260929-1357_phase3_dagger_r*`（`phase3-rl-base-analysis.md`，`r{1..5}`）；`datasets/BTC20261002-0941_expert5k_v4`（`v7-beat-idm.md`）。

## 7. 附录：核验记录

- 引用检查命令（对每个候选）：`grep -rEl -- "<名字>([^A-Za-z0-9_-]|$)" $(git ls-files tests tools pipeline net env reward_model config docs) .slim/deepwork/*.sh`；补充裸名匹配与 `.slim/` 全目录扫描。
- 近 1h 写入检查：`find <候选目录> -type f -newermt '-60 minutes' -print -quit` → 327/327 无命中；全局 `find runs datasets -type f -newermt '-60 minutes'` 仅命中 KEEP 区（`runs/eval/`、`runs/BTC20261009-1545_sw_exp128`、`runs/BTC20261009-1545_sw_attn2`）。
- `runs/` 全量对账（329 个顶层目录）：Tier-1 候选 173（含 w1 KEEP）= 172 候选 + 1 KEEP；Tier-2 候选 136（含 s11_arm1 KEEP）= 135 候选 + 1 KEEP；KEEP 非候选 20 = 8×20261007 + 6×20261009 + 6 个非 BTC（`_refs_oldgen`、`_refs_rlbase`、`eval`、`reward_audit`、`reward_audit_ebeta2`、`reward_viz`）。
- **并发说明**：引用检查基线 = 2026-10-09 18:20–18:30 仓库状态。期间并行会话（config/docs 清理）移动/归档了部分 `config/`、`docs/` 文件（如 `config/arms/v7_*.yaml` → `docs/archive/config_arms_legacy/`、`docs/experiments.md` → `docs/archive/experiments.md`、`docs/v6_program_*.md` → `docs/v6_reports/`）；本清单中引用的文件路径可能已随归档变化，**引用关系本身不变**。
- 本清单为 dry-run 快照；**未执行任何删除/移动/改名，未修改 `.slim/` 内任何文件**。
