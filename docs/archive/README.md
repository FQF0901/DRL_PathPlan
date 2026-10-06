# docs/archive/ —— v3/v4 代历史归档（2026-10-06 锁版清理 B/C）

> B 批（10 项）与 C 批（6 项）均**保留历史，不删**；文件内容原样，未做任何修改。
> 原因：v3/v4 代实验线已关闭，这些文档仅作历史追溯，不再是现行口径。
> 其他文档中若仍引用原 `docs/<文件名>` 路径，对应本目录同名文件；映射总说明见 `docs/v7_reports/MIGRATION_NOTE.md` 与 `docs/v7_reports/cleanup/CLEANUP_C_REPORT.md`。

## B 批（2026-10-06 锁版清理 B 移入）

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

## C 批（2026-10-06 锁版清理 C 移入）

| 原路径 | 现路径 | 说明 |
|---|---|---|
| `docs/design-v1.2.md` | `docs/archive/design-v1.2.md` | v3 代目录/命名纪律设计（§3.5 仍被 README 引用，路径已更新为归档件） |
| `docs/dataset_stats.md` | `docs/archive/dataset_stats.md` | v3 代数据集统计（历史产物清单） |
| `docs/metrics.md` | `docs/archive/metrics.md` | v3 代指标口径说明（被 v6/v7 报告取代） |
| `docs/phase3_rootcause_analysis.md` | `docs/archive/phase3_rootcause_analysis.md` | v3 phase3 根因分析（E-β′ 定档证据；v6/v7 报告已接续） |
| `docs/rl_stage_c_experiments.md` | `docs/archive/rl_stage_c_experiments.md` | v3/v4 协议 + 结果台账（历史 fixed/broken/net/z 记录） |
| `docs/rl_stage_c_v4_report.md` | `docs/archive/rl_stage_c_v4_report.md` | v4 代报告（教训引用；v6 起已被取代） |

> 路径更新：`README.md`、`docs/v6_net_design.md`、`docs/rl_reward_v5.md`、`docs/v7_program_prereg.md`、`docs/v7_reports/README.md` 中的指向已改为 `docs/archive/...`；`docs/v6_program_prereg.md` 顶部加 1 行路径注记（正文未改）。

**历史文档引用项的数据删除注记（清理 C）**：本目录内文档所引用的部分 `runs/`/`datasets/` 项已于清理 C 删除（v3/v4 代实验线关闭，无现行复现依赖）。逐项清单见 `docs/v7_reports/cleanup/CLEANUP_C_DELETED.txt` 与 `CLEANUP_C_REPORT.md` §2。**保留**的现行证据链（w1→s11→p4extra/s14、E-β′、E-β″、v6 P4、IDM 基线）不受影响。

**未归档（仍留 `docs/`）的 v4 代相关文档**：

- `docs/version_ledger.md`：A/B 版本台账，被 `docs/v6_program_report.md` 引为上游依据 → 保留原位。
