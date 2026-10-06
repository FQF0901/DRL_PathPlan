# 锁版清理 A 报告（/tmp/opencode 产物搬运 + 清理 + repo 清单）

> 执行：fixer 子代理；日期：2026-10-06；commit：`8762a22`（docs 入仓 + 引用更新）
> 产出：`/tmp/opencode/CLEANUP_A_REPORT.md`（本文件）、`/tmp/opencode/CLEANUP_PLAN.md`（repo 清单）、`/tmp/opencode/DELETED_MANIFEST.txt`（删除逐项清单）、`/tmp/opencode/KEEP/`（未决暂存）

## 0. 前置检查与体积

- **进程检查**：`pgrep -af "tools/(train|test)\.py|v7_.*driver|v6_supervise"` → 仅 pgrep 自身，**无依赖 /tmp 的在跑进程**。
- 体积：`/tmp/opencode` **759M → 1.3M**（净删 ≈ 758M）；repo **6.3G → 6.3G**（本次仅清缓存 ~4.6M，四舍五入不变；runs/datasets 未删）。
- 约束遵守：先复制后删除；逐项记录删除清单；未动 `env/`、`net/`、`pipeline/`；未改 `.gitignore`；`git status` 全程可解释（收尾干净）。

## 1. 任务 1：必须保留项入仓（先复制后删除）

| 类别 | 数量 | 目标 | 备注 |
|---|---|---|---|
| v7 报告 | 24 | `docs/v7_reports/` | 清单中 `v7_seed11_attribution.md` 实际不存在（未产出），无入仓 |
| v6 报告 | 7 | `docs/v6_reports/` | 全部存在 |
| 图 | 4 | `docs/v7_reports/figures/` | tollgate spec34 四帧；`git add -f`（`*.png` 在 .gitignore，未改规则） |
| 规格 | 3 | `docs/v7_reports/specs/` | clean500 / clean150 / tg45；sha256 已交叉核对（见 §6） |
| 时间线 | 1 | `docs/v7_reports/v7_night_watch.log` | 夜间队列存档 |

- README 索引：`docs/v7_reports/README.md`、`docs/v6_reports/README.md`（每项含 原路径/新路径/sha256/一句话说明；规格附与 `env/specs/` 生成物关系）。
- 复制后逐文件 size 校验一致；sha256 清单见 README（入仓副本实测）。

## 2. 任务 2：repo 内引用更新

- `docs/v7_program_report.md`：§4.4 / §8.7 证据档清单改为 in-repo 路径（未入仓项标注"已清理"）；§2 汇总表 4 行、§3 表示层行、§4.2 worktree、§6 配对目录、§8.3/§8.7 图与 JSON 引用同步更新。**数字与结论未改**。
- `docs/v6_program_report.md`：§5.4 证据档清单、§4 时间线（seed11 归因）、§5.2/§6 引用同步更新。
- 其他 docs（`v7_program_prereg.md`、`rl_stage_c_*` 等）**未动**——冻结预注册/历史文档保完整性（见未决 3）。

## 3. 任务 3：/tmp/opencode 清理

- **删除 1157 个顶层项（746M）**：一次性脚本/日志/JSON/缓存、`phase3_diag/`（规格已搬）、`research/`（论文 PDF）、`rollback/`（swap ckpt）、`oldcode/`（旧代码副本）、各代工作目录等；逐项（路径/体积/文件数）见 `DELETED_MANIFEST.txt`。
- **worktree**：`git worktree remove --force /tmp/opencode/v7_pre_v5` + `git worktree prune` 成功；移除前该 worktree dirty 的 8 个文件（`reward_model/*.py`、`config/arms/v7_*`）与主树**逐位一致**（无独有改动，已在后续 commit 中）。复现提示已写入 v7 报告 §4.2 与 v7_reports README：**如需复现 s1/s2/s11 与 P4-extra/奖励/§14 臂（同代码期 `2f4450e`），重建 worktree**。
- **KEEP（未决）**：8 个文件 1.1M（P0 spec-seed 变体、q6q2 碰撞/tollgate JSON、v7_struct_retrain_v5.md）——均被入仓报告直接引用但不在原保留清单；说明见 `KEEP/README.md`。
- **最终状态**：`/tmp/opencode/` 仅 4 项：`KEEP/`、`CLEANUP_PLAN.md`、`DELETED_MANIFEST.txt`、本报告。

## 4. 任务 4：repo 清理清单（只出清单，未删）

详见 `CLEANUP_PLAN.md`（逐项表）。摘要：

| 区域 | 保留 | 建议删除 | 复核 |
|---|---|---|---|
| `runs/`（510 子目录，3.8G） | 365 个 / 3.30G（被引用 + 最佳产物链） | **105 个 / 358M**（未引用陈旧：v3/v4 perf/smoke、p3 中间 eval、p4prep smoke 等） | 40 个 / 96M（同批次/缩写引用疑似） |
| `datasets/`（24 目录 + 1 日志，1.8G） | 16 项 / 1.35G（v4/v4.1 + DAgger 窗口 + 被引用） | **7 项 / 114M**（phase3_dagger r1–r4 中间轮、expert500val_v5、expert5k.log） | 2 项 / 295M（v5 线：expert5k_v5 + phase3_dagger_v5，已关闭，倾向删） |
| `docs/` | 15（现行 + 被引用） | 0 | 10 项归档候选（v3/v4 代报告与预注册、早期文档/架构图） |

- **已直接清理（本次）**：`__pycache__/` ×12、`*.pyc` ×197、`.pytest_cache/`（共 ~4.6M，忽略文件）。
- 未跟踪文件：`git status --porcelain` 干净（无）。
- 注意：`runs/`、`datasets/` 为 `.gitignore` 未跟踪项，删除不影响 git 历史，但**最佳产物链 ckpt 是唯一副本**，删除前务必按清单复核。

## 5. 未决项

1. **KEEP/ 8 文件**（1.1M）：建议复核后搬入 `docs/v7_reports/`（`v7_struct_retrain_v5.md` → 报告；P0 变体 → `specs/p0_variants/`；q6q2 JSON → `evidence/`）或明确删除。
2. **datasets v5 线**（expert5k_v5 275M + phase3_dagger_v5 20M）：已关闭线，仅历史文档引用（非复现依赖）→ 建议删除；`expert500val_v5`（27.8M）未引用，建议删除。
3. **repo 其他 docs 仍引用已删除的 /tmp 路径**（未在本次更新范围）：`v7_program_prereg.md`（30 处）、`rl_stage_c_experiments.md`（9）、`rl_stage_c_report.md`（6）、`rl_stage_c_v4_report.md`（4）、`phase3_rootcause_analysis.md`（4）、`version_ledger.md`（2）、`rl_reward_v5.md`（2）、`v6_net_design.md`（2）、v4/v4p1 prereg（各 2）、`reward_audit/`（2）、`metrics/feasibility/experiments/design-v1.2`（各 1）。冻结预注册未改；如需统一标注/补搬，建议列入"清理 B"。
4. 入仓报告正文内的历史 `/tmp` 路径保持原样（存档完整性，README 已记录原→新映射）。
5. 结构线未入仓报告 `v7_struct_retrain_v5.md` 在 KEEP；其余未入仓报告（`v7_p1b_report.md`、`v7_p0_gap_tables.md` 等）已按清单删除。

## 6. 验证

- **规格 sha256 交叉核对**（入仓副本 vs repo 内既有记录）：`specs_val_only500` `087db3f5…` = `docs/rl_stage_c_experiments.md`；`specs_val_only150` `81f0f958…` = `docs/v7_program_prereg.md` §12；`specs_tollgate45` `27010b0e…` = `docs/v7_reports/v7_p1_probe_v1v2.md`。全部一致。
- **搬运完整性**：39 个报告/图/规格/日志复制后 size 逐项一致；sha256 已写入 README。
- **git**：`git status --porcelain` 干净；commit `8762a22`（43 files changed，含 4 图 `-f`）。
- **未运行测试**：本任务为文档搬运/文件清理，未改代码；无需代码测试（`env/`、`net/`、`pipeline/` 未动）。
