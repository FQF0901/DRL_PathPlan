# docs/ 索引与维护规则（2026-10-15 整理）

> 本文件 = `docs/` 的**权威导航**。整理原则：**被代码按路径引用的文档一律留在 `docs/` 根**（不移动）；历史代际/已关闭线的文档移入归档桶（`archive/`、`v6_reports/`、`v7_reports/`），并同步更新全部引用。归档件**正文不改**，只由 README 索引。

## 1. 现行核心（常读）

| 文件 | 角色 | 引用方 |
|---|---|---|
| `deepseekv4p1_argue.md` | **决策台账/主文档**（§0 一页纸 + 历轮 §5.x；GPT↔我方往来结论） | GPT、各 lane 脚本（13 处代码引用） |
| `chatgpt5p6sol_review.md` | GPT 审阅往来原文（每轮拉取后入库） | `net/path_head.py` 等（出处引用 12 处） |
| `architecture_contract.md` | 架构契约（生成器/辅助边界/调用图/训练部署同构） | `net/model.py`、`net/path_head.py` |
| `round37_report.md` | 本轮全量报告（阶段 1：三层根因 + lane 索引 + 玩家讨论纪要） | 本索引 |
| `repo_handbook.md` | **仓库手册**：网络架构 / 训练流程 / 当前问题（第 1/2 章代码核验 + 第 3 章台账依据；附录含文档-代码漂移表） | 本索引、台账 |
| `r37_p0b_approval_packet.md` | **CG 审批包**：P0-B（IDM+PP 蒸馏）放行 + 7 项决策（D1–D7）+ 风险与放行条件（含 M8 取消披露） | 台账 §5.67 |
| `round36_report.md`、`r36_implementation_evidence.md`、`r36_warp_layer_note.md` | 上一轮报告与证据/注记 | 台账、round36 工具 |
| `LOCKS.md` | 锁版/冻结清单 | 清理记录 |

## 2. 契约与规格（代码按路径引用，**勿移动**）

`p1_v2_decision.md`（P1-v2 决策）· `p1_probe_spec.md`（探针规格 v1.1）· `p0_eval_compat_design.md`（P0 评测兼容设计）· `recovery_v2_contract.md` / `recovery_v1_dataset_card.md`（恢复线契约与卡片）· `semantic_v1_dataset_card.md`（语义数据集卡）· `r21_lane_semantic_contract.md` / `r21_target_lane_selection.md` / `r21_3a_semantic_wiring.md`（R21 语义三件套）· `rl_reward_v5.md`（奖励口径）· `v7_program_prereg.md`（v7 预注册，仍被 `net/anchor.py` 引用）· `round30_audit.md`（R30 Reality Audit，现行问题框架依据）

**清理/治理（R38 减枝四交付物，2026-10-15）**：`code_reachability_audit.md`（67 符号可达性审计 + 逐组判定/迁移顺序 + CI 守卫清单）· `deprecation_manifest.yaml`（46 项：ARCHITECTURE_FORBIDDEN 19 / LEGACY_REPRO 24 / CHECKPOINT_ONLY 3）· `runtime_entrypoint_matrix.md`（正式入口 × `.slim` 链 × legacy 触达）· `checkpoint_migration_contract.md`（ckpt 键面普查 + 迁移映射草案）

**R38 裁决/审计（2026-10-15）**：`r38_path_field_classification.md`（`path_*` 七字段最终分类 + 缺失语义 + schema 冻结建议；编排裁定已结案）· `r38_unified_benchmark.md`（protocol hash + Clean500 定义 + 重跑表，草稿待收口）

## 3. 归档桶（只读；正文不改）

| 桶 | 内容 | 索引 |
|---|---|---|
| `archive/p1v2_reports/`（15 件，2026-10-15 入） | R31–R35 报告/审计 + P1-v2 早期（p1_0/p1_2/p1_impl/stage_a） | `archive/README.md` §D |
| `archive/v8_reports/`（2 件，2026-10-15 入） | v8 代网络架构与 review | `archive/README.md` §D |
| `v7_reports/`（含 `v7_program_report.md`） | v7 代全部报告/清理记录/迁移说明 | `v7_reports/README.md` |
| `v6_reports/` | v6 代报告 | `v6_reports/README.md` |
| `cleanup/`、`reward_audit/`、`archive/`（A–C 批） | V8 清理/奖励审计/更早档案 | 各目录 README/MANIFEST |

## 4. 迁移表（2026-10-15 批，18 项）

完整"原路径 → 现路径"见 **`archive/README.md` §D**。摘要：`p1_0_report / p1_2_diagnosis / p1_implementation_contract / stage_a_config_consumption / round31_{audit_summary,stagec_contract_audit} / round32_{canary_report,decision,audit_summary} / round33_report / round33c_report / round34a_report / round34b0_report / round35_report / navigation_scope_audit` → `archive/p1v2_reports/`；`net_architecture / v8_net_review` → `archive/v8_reports/`；`v7_program_report` → `v7_reports/`。

**引用同步**：迁移时已更新全部 `docs/…` 引用——**tracked 侧 19 个文件**（随迁移入 commit `d18f148`，该 commit 共 33 项 = 18 重命名 + 14 修改 + 1 新增）＋ **`.slim` 本地 7 个**（`p1_0`/`round30`/`round31`/`round32`/`round33` journal、`round35/f1_nav_audit.py::DOCS_MD`、`round34/d3_leakage_causality.py`）；`runs/round36/agent_review/*` 中的旧路径提及（历史留痕）未改，不在校验范围。（2026-10-15 勘误：原记"25 个文件"系脚本口径〔tracked+本地混计〕，已按上述分列。）

## 5. 维护规则（新增文档时遵守）

1. **命名**：轮次报告 `roundNN_report.md`；证据/注记 `rNN_*.md`；契约 `*_contract.md`/`*_spec.md`；数据集卡 `*_dataset_card.md`。
2. **归位**：当轮文档放 `docs/` 根；轮次关闭 ≥2 轮后，若无代码引用 → 移入 `archive/`（更新本索引 + `archive/README.md` + 全部引用）。
3. **移动前必跑引用扫描**（模板）：
   ```bash
   grep -rn "<basename>" --include=*.py --include=*.sh --include=*.md --include=*.json \
     net/ env/ tools/ pipeline/ tests/ config/ docs/ .slim/ 2>/dev/null | grep -v "<file itself>"
   ```
   有代码引用 → 不移动（或先同步更新引用方）。
4. **归档件正文不改**；索引只增补。
5. 台账（`deepseekv4p1_argue.md`）的 §6 索引与本文件保持同步。
