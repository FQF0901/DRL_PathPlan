# v6 证据/报告档（in-repo 存档）

> 来源：v6 程序执行期证据档原写于 `/tmp/opencode/`（不入 repo）。2026-10-06 锁版清理时按保留清单**先复制后删除**入仓；本目录为权威副本，sha256 为入仓副本实测值（`sha256sum` 可复验）。
> 原 `/tmp/opencode/` 其余 v6 工作档（驱动脚本、逐臂 JSON、`v6_p4_results.{md,json}`、`v6_p4_seed11_report.md`、`v6_gate4_fixes.md`、`v6_p1_validation.md`、`v6_reward_audit*.md`、`v6_ttc_falsification.md`、`v6_p3_data.md`、`v6_stageA_cost.md` 等）未在保留清单内，已随同日清理删除；其中部分内容已由 repo 内 `docs/reward_audit/`、`docs/v6_reports/v6_net_design.md`、`docs/v6_reports/v6_program_prereg.md` 承接。
> 索引用途：主报告 `docs/v6_reports/v6_program_report.md` 的 §5.4 / §6 引用。
> **状态：冻结**。本目录正文（报告/prereg/net 设计）为 v6 程序冻结证据档，只读；本 README 为索引。

## /tmp 入仓报告（7 项）

| # | 原路径 | 新路径 | sha256 | 说明 |
|---|---|---|---|---|
| 1 | `/tmp/opencode/v6_p3_report.md` | `docs/v6_reports/v6_p3_report.md` | `676530c5b44af43138ef355fdd7441c972877933e9b5b18c8fa417f593824254` | P3 执行报告：mini 闸 → 全量重训链 → E3（E-β″ 零点） |
| 2 | `/tmp/opencode/v6_p4_report.md` | `docs/v6_reports/v6_p4_report.md` | `1346fbeae9bfaa46a26fabe66908568391307df824305ea95ccde228365e4dfc` | P4 Stage C 臂链（Gate4）批报告：arm0–5 完成、arm6/7 预算门截断 |
| 3 | `/tmp/opencode/v6_seed11_attribution.md` | `docs/v6_reports/v6_seed11_attribution.md` | `fd5e2ea13d618cd710feb7b9e9f1e5300746e0e8d3fe3b21d00a66ad7c404808` | seed=11 复现失败归因（arm0：seed 0 vs 11；含 §10 复算命令） |
| 4 | `/tmp/opencode/v6_p4_recheck.md` | `docs/v6_reports/v6_p4_recheck.md` | `14eb398d492f979ec845c03f05de3c5acf7d4dfc09b85675fc4d0cf22024f024` | P4 arm1–4 · u200 复检（Gate4 终审建议；post-hoc 探索性标注） |
| 5 | `/tmp/opencode/v6_collision_arm_design.md` | `docs/v6_reports/v6_collision_arm_design.md` | `1db4655d05d695c76136b474d67fc562272d445691c7c48e186f8e9e81cd2d09` | 碰撞抑制臂诊断 + 设计 + 预注册（P5 前置；v7 P4-extra/§14 依据） |
| 6 | `/tmp/opencode/v6_p4_incident.md` | `docs/v6_reports/v6_p4_incident.md` | `65dafc5162091d4eb09b4e0190899d7787b53fbfb8af3509932b0f00f7f6439e` | 事件记录：arm0 首臂后复核误停（driver tag 前缀 bug） |
| 7 | `/tmp/opencode/v6_cleanup_report.md` | `docs/v6_reports/v6_cleanup_report.md` | `9be2b681c27274e0a01507acb7fc6634dcaa44a9d03276631d6fc514f4fb6e3b` | v6 前置清理报告（runs/、/tmp、/tmp/opencode 范围与保留项） |

> 上表 7 项 sha256 已于 2026-10-09 逐文件复验，与磁盘一致。

## V8 清理移入（2026-10-09，3 项；原 `docs/` 根）

| 原路径 | 现路径 | sha256 | 说明 |
|---|---|---|---|
| `docs/v6_net_design.md` | `docs/v6_reports/v6_net_design.md` | `830bf15e95edbc283a5f816ecebe570065828b4fe4dc035877cef86b40d1651d` | v6 Net 设计冻结（P1 架构规格） |
| `docs/v6_program_prereg.md` | `docs/v6_reports/v6_program_prereg.md` | `3079e27b5489dabeb34b77a79c0d1204d431497749a8502b2660a2a9e9fa6ed3` | v6 程序级预注册（6 阶段 + Oracle 门；冻结） |
| `docs/v6_program_report.md` | `docs/v6_reports/v6_program_report.md` | `2dd0aa7d6f409c4e0a5bcea1c404909098e03372fbed74833001c36e962a723f` | v6 程序收尾报告（主报告） |

> 移入依据：`docs/cleanup/V8_CLEANUP_config_docs.md` §1.5（正文未改）。sha256 为 2026-10-09 入仓副本实测（`sha256sum` 可复验）。移入后正文内相对链接按移动前原位书写（如 `rl_reward_v5.md`），未改；映射见 `docs/v7_reports/MIGRATION_NOTE.md`。

## 复现提示

- 关键数字复算入口在 repo：`runs/*/episodes.csv`（逐 (id,seed) 配对）、`runs/*/metrics.json`（overall）；seed11 归因复算命令见 `v6_seed11_attribution.md` §10。
- v6 P4 驱动/脚本（`v6_p4_driver.py` 等）不入 repo，已随清理删除；执行口径与 pin 表见 `docs/v6_reports/v6_program_prereg.md` §7.4。
