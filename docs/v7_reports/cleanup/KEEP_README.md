# KEEP（未决暂存，待复核）

> 2026-10-06 锁版清理 A：以下文件**不在原保留清单**，但被已入仓报告直接引用（含 sha256），且生成脚本/环境已随本次清理删除 → 删除将使引用悬空或不可复现，故暂存于此待复核。
> 注意：`/tmp` 非持久存储；确认后请搬入 repo（建议 `docs/v7_reports/`）或明确删除。

| 文件 | 大小 | 引用处 | 说明 |
|---|---|---|---|
| `v7_struct_retrain_v5.md` | 11K | `docs/v7_program_report.md` §3（表示层主判据行）、§4.4 | v5 重训链 Stage B 读数来源；入仓报告直接引用 |
| `v7_p0_specs/scenarios_eval500_seedA.json` | 398K | `docs/v7_reports/v7_p0_idm_baseline.md`（sha `e31a87b2…`） | P0 fix-15 spec-seed 变体 A |
| `v7_p0_specs/scenarios_eval500_seedB.json` | 398K | `docs/v7_reports/v7_p0_idm_baseline.md`（sha `fa644eb4…`） | P0 fix-15 spec-seed 变体 B |
| `v7_p0_specs/variants_manifest.json` | 1K | `docs/v7_reports/v7_p0_idm_baseline.md`（sha `489326e8…`） | 变体 manifest（seed 偏移 +100000/+200000） |
| `v7_p0_specs/determinism_probe8.json` | 6K | `docs/v7_reports/v7_p0_idm_baseline.md` 异常深挖段 | 8 条确定性探针子集 |
| `v7_p0_specs/probe_299.json` | 1K | `docs/v7_reports/v7_p0_idm_baseline.md` 异常深挖段 | id 299 单条探针 |
| `v7_q6q2/collision_replay.json` | 239K | `docs/v7_program_report.md` §8.7、`docs/v7_reports/v7_q6_collision_types.md` | 101 条碰撞仪器化重放原始 JSON（生成脚本已删） |
| `v7_q6q2/tollgate_viz.json` | 9K | `docs/v7_program_report.md` §8.3、`docs/v7_reports/v7_q2_tollgate_figure.md` | tollgate 运行时度量 JSON（生成脚本已删） |

合计约 1.1M。
