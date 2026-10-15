# docs/archive/ —— 历史归档桶（只读）

> 收录规则：已关闭实验线（v3/v4 代）的文档、已被现行口径取代的文档，以及锁版/清理时移入的历史快照。
> 归档件正文一律原样保留、不修改；本 README 仅作索引。现行口径以 `docs/` 根（如 `docs/rl_reward_v5.md`、`docs/v7_program_prereg.md`、`docs/v7_reports/v7_program_report.md`）、`docs/v6_reports/`、`docs/v7_reports/` 为准。
> 相关记录：`docs/cleanup/`（2026-10-09 V8 清理）、`docs/v7_reports/cleanup/`（2026-10-06 锁版清理 A–D）、`docs/v7_reports/MIGRATION_NOTE.md`（/tmp → repo 迁移说明）。

## 索引（2026-10-09 实况：18 个文件 + `config_arms_legacy/`）

### A. 2026-10-09 V8 清理移入（2 文件 + 1 目录）

| 现路径 | 原路径 | 说明 |
|---|---|---|
| `docs/archive/experiments.md` | `docs/experiments.md` | 2026-09-25 期实验记录（v3 代 IL 台账；文中 `runs/*` 产物多已清理，可按文内命令重生成） |
| `docs/archive/version_ledger.md` | `docs/version_ledger.md` | A/B 链路版本台账（Review 用；含 E-β′ 定档与窗口条件性记录） |
| `docs/archive/config_arms_legacy/` | `config/arms/`（21 个 `.yaml`）+ `tests/`（2 个退役测试） | v6 P4 奖励臂 / v7 奖励与结构臂快照 + `.py.disabled` 测试；目录内 `README.md` 为逐项权威清单 |

> `config_arms_legacy/` 现状（`ls`）：21 个 `.yaml` + `test_p4_arm_configs.py.disabled` + `test_v7_reward_arms.py.disabled` + `README.md`，共 24 文件。当前 `config/arms/` 仅保留 `README.md` 与 `v7_arm1_offroad.yaml`。
> 移入依据与引用检查：`docs/cleanup/V8_CLEANUP_config_docs.md`（§1.1–§1.3、§7）。

### B. 2026-10-06 锁版清理 B 批（10 项）

| 原路径 | 现路径 | 说明 |
|---|---|---|
| `docs/rl_stage_c_report.md` | `docs/archive/rl_stage_c_report.md` | v3 代 Stage C 报告（v3 线已关） |
| `docs/rl_stage_c_v3_prereg.md` | `docs/archive/rl_stage_c_v3_prereg.md` | v3 代预注册（已关线） |
| `docs/rl_stage_c_v4_prereg.md` | `docs/archive/rl_stage_c_v4_prereg.md` | v4 代预注册（已关线） |
| `docs/rl_stage_c_v4p1_prereg.md` | `docs/archive/rl_stage_c_v4p1_prereg.md` | v4.1 代预注册（已关线） |
| `docs/rl_v3_evidence.sha256` | `docs/archive/rl_v3_evidence.sha256` | v3 证据哈希清单（内部路径相对 repo 根，移动后仍可校验） |
| `docs/p0-measurements.md` | `docs/archive/p0-measurements.md` | v3 期 P0 实测 |
| `docs/feasibility-analysis.md` | `docs/archive/feasibility-analysis.md` | 早期可行性分析 |
| `docs/forensics-2026-09-26.md` | `docs/archive/forensics-2026-09-26.md` | 早期场景级失败取证（2026-09-26） |
| `docs/db44fefe-system-review.md` | `docs/archive/db44fefe-system-review.md` | 早期系统 review（P0-1/P0-2/P0-3 依据） |
| `docs/architecture-BTC20260925-2234.svg` | `docs/archive/architecture-BTC20260925-2234.svg` | 早期系统架构图 |

### C. 2026-10-06 锁版清理 C 批（6 项）

| 原路径 | 现路径 | 说明 |
|---|---|---|
| `docs/design-v1.2.md` | `docs/archive/design-v1.2.md` | v3 代目录/命名纪律设计（§3.5 仍被 README 引用，路径已更新为归档件） |
| `docs/dataset_stats.md` | `docs/archive/dataset_stats.md` | v3 代数据集统计（历史产物清单） |
| `docs/metrics.md` | `docs/archive/metrics.md` | v3 代指标口径说明（被 v6/v7 报告取代） |
| `docs/phase3_rootcause_analysis.md` | `docs/archive/phase3_rootcause_analysis.md` | v3 phase3 根因分析（E-β′ 定档证据；v6/v7 报告已接续） |
| `docs/rl_stage_c_experiments.md` | `docs/archive/rl_stage_c_experiments.md` | v3/v4 协议 + 结果台账（历史 fixed/broken/net/z 记录） |
| `docs/rl_stage_c_v4_report.md` | `docs/archive/rl_stage_c_v4_report.md` | v4 代报告（教训引用；v6 起已被取代） |

### D. 2026-10-15 P1-v2 代归档（18 项：报告/审计/契约快照）

| 原路径 | 现路径 | 说明 |
|---|---|---|
| `docs/p1_0_report.md` | `docs/archive/p1v2_reports/p1_0_report.md` | P1-v2 执行纪要（Step 1–3 压缩存档源） |
| `docs/p1_2_diagnosis.md` | `docs/archive/p1v2_reports/p1_2_diagnosis.md` | P1-v2 阶段诊断 |
| `docs/p1_implementation_contract.md` | `docs/archive/p1v2_reports/p1_implementation_contract.md` | P1 实现契约 v1 |
| `docs/stage_a_config_consumption.md` | `docs/archive/p1v2_reports/stage_a_config_consumption.md` | Stage-A 配置消费审计 |
| `docs/round31_audit_summary.md` | `docs/archive/p1v2_reports/round31_audit_summary.md` | R31 审计包摘要（R30 Reality Audit 审计包） |
| `docs/round31_stagec_contract_audit.md` | `docs/archive/p1v2_reports/round31_stagec_contract_audit.md` | R31 Stage-C 契约审计 |
| `docs/round32_canary_report.md` | `docs/archive/p1v2_reports/round32_canary_report.md` | R32 canary Gate 0–4 全测 |
| `docs/round32_decision.md` | `docs/archive/p1v2_reports/round32_decision.md` | R32 决策记录（零引用件） |
| `docs/round32_audit_summary.md` | `docs/archive/p1v2_reports/round32_audit_summary.md` | R32 审计摘要（零引用件） |
| `docs/round33_report.md` | `docs/archive/p1v2_reports/round33_report.md` | R33 报告（33A 三问 + 33B warp NO-GO） |
| `docs/round33c_report.md` | `docs/archive/p1v2_reports/round33c_report.md` | R33C 报告（B1 稳定器：硬门 6/11 未过） |
| `docs/round34a_report.md` | `docs/archive/p1v2_reports/round34a_report.md` | R34A 路径级失败归因（覆盖 94.7%） |
| `docs/round34b0_report.md` | `docs/archive/p1v2_reports/round34b0_report.md` | R34B0 覆盖门 |
| `docs/round35_report.md` | `docs/archive/p1v2_reports/round35_report.md` | R35 导航范围审计收口 |
| `docs/navigation_scope_audit.md` | `docs/archive/p1v2_reports/navigation_scope_audit.md` | R35 审计正文（32KB；生成脚本 `.slim/deepwork/p1_v2/round35/f1_nav_audit.py` 的 `DOCS_MD` 已同步） |
| `docs/net_architecture.md` | `docs/archive/v8_reports/net_architecture.md` | v8 代网络架构（42KB） |
| `docs/v8_net_review.md` | `docs/archive/v8_reports/v8_net_review.md` | v8 代网络 review（25KB） |
| `docs/v7_program_report.md` | `docs/v7_reports/v7_program_report.md` | v7 代程序报告（归入既有 v7 桶） |

> 归档依据：2026-10-15 全仓引用扫描（727 文件）——上述 18 件**零代码引用**（`round30_audit.md`/`navigation_scope_audit.md` 之外仅文档互引；后者的代码引用方 `f1_nav_audit.py` 路径常量已同步更新）。**保留在根**的文档 = 被 `net/`/`env/`/`tools/`/`tests/`/`.slim` 代码按路径引用者（清单见 `docs/README.md`）。

## 引用路径注记（按时间）

1. 清理 B/C 期间：`README.md`（根）、`docs/v6_reports/v6_net_design.md`、`docs/rl_reward_v5.md`、`docs/v7_program_prereg.md`、`docs/v7_reports/README.md` 中指向本目录的路径已改为 `docs/archive/...`；`docs/v6_reports/v6_program_prereg.md` 顶部加 1 行路径注记（正文未改）。
2. 2026-10-09 V8 清理：`experiments.md`、`version_ledger.md` 移入本目录；`v6_net_design.md`、`v6_program_prereg.md`、`v6_program_report.md` 移入 `docs/v6_reports/`。本次**未同步更新**相关引用；已知失效链接清单见 `docs/cleanup/V8_CLEANUP_config_docs.md` §5.3。
3. 2026-10-15 P1-v2 代归档（本批）：**已同步更新**全部 `docs/…` 路径引用（25 个文件：`docs/*.md`、根 `README.md`、`docs/cleanup/`、`docs/v7_reports/*`、`docs/archive/README.md` 等）+ 代码引用（`.slim/deepwork/p1_v2/round35/f1_nav_audit.py::DOCS_MD`、`.slim/deepwork/p1_0/journal.md` 注记）；**未改**历史 run 产物（`runs/round36/agent_review/*`）与 `.slim/**/journal.md` 中的旧路径提及（日志留痕，不在校验范围）。

## 历史文档引用项的数据删除注记

- 清理 C（2026-10-06）：本目录文档所引用的部分 `runs/`/`datasets/` 项已删除（v3/v4 代实验线关闭，无现行复现依赖）。逐项清单见 `docs/v7_reports/cleanup/CLEANUP_C_DELETED.txt` 与 `CLEANUP_C_REPORT.md` §2。
- V8 清理-2（2026-10-09）：`datasets/` 16 项 + `runs/` Tier-1 170 项删除；逐项见 `docs/cleanup/V8_CLEANUP_data_runs_DONE.md`。
- **保留**的现行证据链（w1→s11→p4extra/s14、E-β′、E-β″、v6 P4、IDM 基线）不受影响。

## 现行文档（未归档，便于对照）

**权威索引见 `docs/README.md`**（现行核心 / 契约与规格 / 数据集卡 / 归档桶 四区 + 维护规则）。`docs/` 根保留：决策台账 `deepseekv4p1_argue.md`、GPT 往来 `chatgpt5p6sol_review.md`、架构契约 `architecture_contract.md`、近两轮报告（`round36_report.md`/`round37_report.md`/`r36_*.md`）、被代码按路径引用的契约与卡片（`p1_v2_decision.md`/`p1_probe_spec.md`/`p0_eval_compat_design.md`/`recovery_*.md`/`semantic_v1_dataset_card.md`/`r21_*.md`/`rl_reward_v5.md`/`v7_program_prereg.md`/`round30_audit.md`/`LOCKS.md`）；证据档：`docs/v6_reports/`、`docs/v7_reports/`；奖励审计：`docs/reward_audit/`；清理记录：`docs/cleanup/`。
