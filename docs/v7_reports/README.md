# v7 证据/报告档（in-repo 存档）

> 来源：v7 程序执行期证据档原写于 `/tmp/opencode/`（不入 repo）。2026-10-06 锁版清理时按保留清单**先复制后删除**入仓；本目录为权威副本，sha256 为入仓副本实测值（`sha256sum` 可复验）。
> 原 `/tmp/opencode/` 其余工作档（一次性脚本/日志/JSON/中间报告）已随同日清理删除；原未决 KEEP 项已于同日清理 B 复核入仓（见 §7）。索引用途：主报告 `docs/v7_program_report.md` 的 §4.4 / §8.7 引用。

## 1. 报告（24 项）

| # | 原路径 | 新路径 | sha256 | 说明 |
|---|---|---|---|---|
| 1 | `/tmp/opencode/v7_p0_gap_decomposition.md` | `docs/v7_reports/v7_p0_gap_decomposition.md` | `70ae1804f44b1f03f4bc5faa51830c73e8a2f236be1c02d16854aba7dfb7ab3c` | P0 差距分解：我方策略 vs IDM 分层剖面（含可复算脚本口径） |
| 2 | `/tmp/opencode/v7_p0_idm_baseline.md` | `docs/v7_reports/v7_p0_idm_baseline.md` | `7ae59f8d6ddedb3fcb25dd6f4efbf351926161b8273672ead9c9b31dfc94f5c4` | P0 fix-15：IDM 现口径复测 + spec-seed 变体（Gate A 补证②） |
| 3 | `/tmp/opencode/v7_p1_dagger_cycle.md` | `docs/v7_reports/v7_p1_dagger_cycle.md` | `d1ef561cf91e9e12a925f8cb918ab1607b1f2a54359be7ed4dc457b327439b63` | P1 DAgger 唯一一轮闭环重标注 cycle（4 窗口）执行报告 |
| 4 | `/tmp/opencode/v7_p1b_failure_diag.md` | `docs/v7_reports/v7_p1b_failure_diag.md` | `2e6575dbdb8eb14afcd1bdd3617380ed44e1568681f84320aaaa4b50aad8d5e7` | P1-B 失败诊断：obs v4 static 段为何没修好 tollgate |
| 5 | `/tmp/opencode/v7_p1_iter2.md` | `docs/v7_reports/v7_p1_iter2.md` | `6df64d7f54cf95bc9a269ea78e16f3b5cc1deb354a17616e67956079922c9f78` | P1 迭代 2：roundtrip 过滤修复 + 横向表达修复 + 教师强制指标 |
| 6 | `/tmp/opencode/v7_p1_iter3.md` | `docs/v7_reports/v7_p1_iter3.md` | `8109341fbf3acada54f4513b044fe01486ebd80279d6c35ace8b69839962342f` | P1 iter3：v4.1 数据 + 横向权重 [1.0,69.4] 的 A20+B20 重训与闭环对照 |
| 7 | `/tmp/opencode/v7_p1_probe_v1v2.md` | `docs/v7_reports/v7_p1_probe_v1v2.md` | `1d8346280d06dae90e64d02fca5b67811ae1c6c5dd5b63b58685baa2d6cd92e3` | P1 复盘探针 V1+V2：plan 尾链 vs μ，定位闭环执行量断点 |
| 8 | `/tmp/opencode/v7_p1_w1_eval500.md` | `docs/v7_reports/v7_p1_w1_eval500.md` | `1e850c13b626030ca7e362f42bbbefa341b6936fb1b5ea9e5625ae47f8dfd85a` | P1 w1 e005 · eval500 探索性复评（Gate B 补证；非独立一次评估） |
| 9 | `/tmp/opencode/v7_p2_arm1.md` | `docs/v7_reports/v7_p2_arm1.md` | `7fd724068f10e62130f2001c9a568de36c069bb97ddb83493456844342634c40` | P2 首臂报告：arm1（w1 e005 + off_road_edge + KL 锚；seeds 0/11） |
| 10 | `/tmp/opencode/v7_p2_s11_verify.md` | `docs/v7_reports/v7_p2_s11_verify.md` | `def321b29230c8085f59fd1ba1f8343bb82d1ca105e2f5c1590ad8cec3181a64` | arm1 seed11 u150 洁净复验（clean500 / eval500 / tg45 / T3） |
| 11 | `/tmp/opencode/v7_p3_arm1_seeds.md` | `docs/v7_reports/v7_p3_arm1_seeds.md` | `8c733d15e8bdbaa7a723a9393ae8a2049fb1b9da55bc4dcf684ba58f9f47d4d5` | P3 arm1 n=4 种子分布读数（seeds 0/11/1/2） |
| 12 | `/tmp/opencode/v7_p4extra_collision.md` | `docs/v7_reports/v7_p4extra_collision.md` | `c0c896902deaa9cbd0110df8dee3322b64be56ecfce79ef2abf169cfba7e4f5d` | P4-extra 碰撞抑制单变量（terminal collision −22→−32；seeds 0/11） |
| 13 | `/tmp/opencode/v7_p4extra_rewards.md` | `docs/v7_reports/v7_p4extra_rewards.md` | `85de502a24d95298f9469d0b4573c27f595e1fd1928bf165a12b90af77d97e30` | §13 奖励单变量臂 A/B/C 队列报告（base = s11 u150） |
| 14 | `/tmp/opencode/v7_s14_ttc.md` | `docs/v7_reports/v7_s14_ttc.md` | `55e62c5d82433e5e479d6c4e81da812cdb6cbfdadc560d747e5d348ffb17d4af` | §14 ttc 稠密近失罚单变量（seeds 0/11）报告 |
| 15 | `/tmp/opencode/v7_q6_collision_types.md` | `docs/v7_reports/v7_q6_collision_types.md` | `d00beb235ca249de03ecf0a5158c09be5c172c1c8a5d5b67f0ef1d8cd65ad412` | Q6：s11 eval500 碰撞类型仪器化重放（101 条 collision 分类） |
| 16 | `/tmp/opencode/v7_q2_tollgate_figure.md` | `docs/v7_reports/v7_q2_tollgate_figure.md` | `91fa6aac012df43ffe45471032b62a0ead45ebddf9d4d94849c00c0c7f150b73` | Q2：tollgate 双面板可视化（spec 34 / s11 u150）分析 |
| 17 | `/tmp/opencode/v7_morning_brief.md` | `docs/v7_reports/v7_morning_brief.md` | `8bd533e13b2d15bced82cfc23601f229ee2abb2fdea2274a28701e4b9b338b00` | 夜间 23:00–06:10 产出汇总晨报（碰撞抑制/类型/tollgate 图/奖励臂/§14） |
| 18 | `/tmp/opencode/v7_kanchor_feasibility.md` | `docs/v7_reports/v7_kanchor_feasibility.md` | `9c66282b4c7a4a83f80472fe30eb26288e68af7bf6f4c164328111b9d49ce8a1` | K-anchor 可行性探针：轨迹聚类 anchor + 选择头（fix-3 依据） |
| 19 | `/tmp/opencode/v7_struct_a_obs_v5.md` | `docs/v7_reports/v7_struct_a_obs_v5.md` | `540c4cc22566621687357cf22a28fd528615d297fc448cd3a19b4c268b44374b` | 结构迭代 A：obs v5（LD 远场 + 当前车道块 + TTC token） |
| 20 | `/tmp/opencode/v7_struct_a2_wiring_probe.md` | `docs/v7_reports/v7_struct_a2_wiring_probe.md` | `bd886e3bcd3a5ed21ecf80c8357eb13d675de63fa4d0bc18b5d42b663c3eda51` | A2：v5 训练管线接线 + 选择头可学性探针 |
| 21 | `/tmp/opencode/v7_struct_b_kanchor.md` | `docs/v7_reports/v7_struct_b_kanchor.md` | `cd9e871f85923341c18714abe1803524522b1446ea6a258057a66d8962a65485` | 结构迭代 B：K-anchor 计划头（K=6 形状锚 + WTA + 选择头）实现与测试 |
| 22 | `/tmp/opencode/v7_struct_fail_diag.md` | `docs/v7_reports/v7_struct_fail_diag.md` | `468419602846686400605b88238b671092a6d4a797ac7bbb4c86c64ae52619ad` | v5+K-anchor 链失败根因诊断（specific experts/router 训练 = 决定性破坏项） |
| 23 | `/tmp/opencode/v7_p3fix_retry.md` | `docs/v7_reports/v7_p3fix_retry.md` | `0dbda1c0c930e4379dbbc9eb9a501db5230608806a16e3e772a3c9424d9e6665` | fix-11 phase3 安全配方重跑原始数据（trunk_only + clean150 守护） |
| 24 | `/tmp/opencode/v7_p1b_chain.md` | `docs/v7_reports/v7_p1b_chain.md` | `b487f583c2efba6777a58e5508cd230b0e304a525f81bdfb37d6bfea36d4d8d3` | P1-B 链报告：obs v4 重采 → Stage A/B 重训 → tollgate 回归（aborted:B） |

> `v7_seed11_attribution.md`：原清单标注"若在"，实际未产出（不存在），无入仓。
> 原 `/tmp/opencode/` 的其余 v7 报告（如 `v7_p0_gap_tables.md`、`v7_p1b_report.md`、`v7_struct_retrain_v5.md` 等）未在保留清单内；其中被主报告引用者见文末"未决项"。

## 2. 图（`figures/`，4 张，来自 Q2 tollgate 双面板）

| 原路径 | 新路径 | sha256 | 说明 |
|---|---|---|---|
| `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_A_first_sighting_step0821.png` | `docs/v7_reports/figures/…A_first_sighting_step0821.png` | `c2e5f1d73fb59af4249dc18b2aae194abba255fc506456a6d88494cd599b7351` | spec 34：static 首次出现（step 821 / 54.71 m） |
| `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_B_decision_zone_39m_step0869.png` | `docs/v7_reports/figures/…B_decision_zone_39m_step0869.png` | `7f07c00b3d126c77f01e141ad63c2074f3b1a984d65fbea4b9ce60b03d0e1686` | 变道决策区 25–39 m（step 869） |
| `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_C_gate_entry_step0929.png` | `docs/v7_reports/figures/…C_gate_entry_step0929.png` | `96f63d1be8e9688de60f1f423b76c5fe87bfcaf9bbd89fa6f405c54c8f5a4b4f` | 进入 `$` block（step 929 / 17.26 m） |
| `/tmp/opencode/v7_q2_tollgate_figure/tollgate_spec34_D_final_crash_step0961.png` | `docs/v7_reports/figures/…D_final_crash_step0961.png` | `c1b7e80f565948541c2679ba59d433af940907e4fee8cdb352bad1f734630365` | 撞岗亭终局（step 961；ego_v 0.45 m/s） |

## 3. 规格（`specs/`，3 项；外部依赖，必须保）

| 原路径 | 新路径 | sha256 | 说明 |
|---|---|---|---|
| `/tmp/opencode/phase3_diag/exp/specs_val_only500.json` | `docs/v7_reports/specs/specs_val_only500.json` | `087db3f5c02e9cedccb8313733207f59dfa90162002b0d9bf002e5b80bcc366e` | **clean500 留出集**（500 条）；v7 主判据/多臂评测规格 |
| `/tmp/opencode/phase3_diag/exp/specs_val_only150.json` | `docs/v7_reports/specs/specs_val_only150.json` | `81f0f95814e3ea8bb8c349e47b2a19121ea78d578a59264fcc8bb1a660d4afa6` | clean500 前 150 条（file-order first 150）；P1-B/复检子集 |
| `/tmp/opencode/v7_p1b_chain/specs_tollgate45.json` | `docs/v7_reports/specs/specs_tollgate45.json` | `27010b0ebfb59d1daf45570d5851cf47452ab3d5eddbc9d9f3ddfc6d98c523e6` | **tg45**：eval500 中 `labels.geometry == tollgate` 的 45 条 |

**与 `env/specs/` 生成物的关系**（均为 repo 冻结文件的离线派生，`env/` 未改动、未新增）：

- `specs_val_only500.json` = `env/specs/scenarios_val.json`（1000）− `env/specs/scenarios_eval500.json`（500），按 (id,seed) 差集、保序；provenance 内嵌。sha256 与 `docs/archive/rl_stage_c_experiments.md` 的 clean500 记录一致（`087db3f5…`）。
- `specs_val_only150.json` = 上述前 150 条（`provenance.subset.parent_raw_sha256 = 087db3f5…`）。sha256 与 `docs/v7_program_prereg.md` §12 记录一致（`81f0f958…`）。
- `specs_tollgate45.json` = `env/specs/scenarios_eval500.json` 中 tollgate 45 条（provenance 含 id 列表：34, 76, …924）。sha256 与 `docs/v7_reports/v7_p1_probe_v1v2.md` 记录一致（`27010b0e…c523e6`）。
- 复现注意：`env/specs/` **不含** clean500/tg45 派生件，复算/重跑须使用本目录三个文件（sha256 逐位核对）。

## 4. 时间线存档

| 原路径 | 新路径 | sha256 | 说明 |
|---|---|---|---|
| `/tmp/opencode/v7_night_watch.log` | `docs/v7_reports/v7_night_watch.log` | `0b26e81d5cc865d0307067cb2772ddee8a5148734d79e6b4635179e6eee2cb2d` | 2026-10-05 夜间队列（P4-extra → fix-3/4/5）时间线日志 |

## 5. 复现提示

- **pre-v5 worktree 已移除**（原 `/tmp/opencode/v7_pre_v5` @ `2f4450e`；移除前 dirty 文件与主树逐位一致，无独有改动）。如需复现 P3 s1/s2 与 P4-extra/奖励/§14 臂（同代码期）：`git worktree add <dir> 2f4450e`（主树 venv 绝对解释器 + `runs`/`datasets` 软链主树）。
- 原 `/tmp/opencode/` 下的驱动脚本/日志/JSON 已删除；一次性复现口径见各报告正文与 `docs/v7_program_prereg.md`。

## 6. 未决项（原暂留 `/tmp/opencode/KEEP/`）——**已于 2026-10-06 清理 B 解决**

以下文件曾因不在原保留清单、但被入仓报告直接引用而暂存 `/tmp/opencode/KEEP/`；清理 B 已逐项复核并搬入本目录（见 §7），`/tmp/opencode/KEEP/` 已清空：

| 文件 | 大小 | 引用处 | 处理 |
|---|---|---|---|
| `v7_struct_retrain_v5.md` | 11 KB | 主报告 §3 表示层主判据行、§4.4 | 入仓 `docs/v7_reports/` |
| `v7_p0_specs/scenarios_eval500_seedA.json` | 398 KB | `v7_p0_idm_baseline.md`（sha `e31a87b2…`） | 入仓 `docs/v7_reports/specs/p0_variants/` |
| `v7_p0_specs/scenarios_eval500_seedB.json` | 398 KB | `v7_p0_idm_baseline.md`（sha `fa644eb4…`） | 同上 |
| `v7_p0_specs/variants_manifest.json` | 1 KB | `v7_p0_idm_baseline.md`（sha `489326e8…`） | 同上 |
| `v7_p0_specs/determinism_probe8.json`、`probe_299.json` | 7 KB | `v7_p0_idm_baseline.md` 异常深挖段 | 同上 |
| `v7_q6q2/collision_replay.json`、`tollgate_viz.json` | 245 KB / 9 KB | 主报告 §8.3/§8.7、`v7_q6_collision_types.md`、`v7_q2_tollgate_figure.md` | 入仓 `docs/v7_reports/evidence/` |

## 7. KEEP 复核入仓（2026-10-06 清理 B；8 文件；sha256 = 入仓副本实测）

| # | 原路径 | 新路径 | sha256 | 说明 |
|---|---|---|---|---|
| 1 | `/tmp/opencode/KEEP/v7_struct_retrain_v5.md` | `docs/v7_reports/v7_struct_retrain_v5.md` | `1b0b6194ae8c9d70f0a47e71832593f9e4f11aacf2132a6e53148191135d9149` | v5 重训链 Stage B 读数来源（主报告 §3 表示层主判据行、§4.4） |
| 2 | `/tmp/opencode/KEEP/v7_p0_specs/scenarios_eval500_seedA.json` | `docs/v7_reports/specs/p0_variants/scenarios_eval500_seedA.json` | `e31a87b232777ccc6f565e44a1d83e85ad2cdb5b909ed9e093098b029d2b6635` | P0 fix-15 spec-seed 变体 A |
| 3 | `/tmp/opencode/KEEP/v7_p0_specs/scenarios_eval500_seedB.json` | `docs/v7_reports/specs/p0_variants/scenarios_eval500_seedB.json` | `fa644eb49d85999c1a651e44fc7ecab6b33dc259a2fbddacc8d8b645ace5e7dc` | P0 fix-15 spec-seed 变体 B |
| 4 | `/tmp/opencode/KEEP/v7_p0_specs/variants_manifest.json` | `docs/v7_reports/specs/p0_variants/variants_manifest.json` | `489326e8a8f1ecb5a5a70e2c391b6389bd45168fa4e8ff5ec4b7c3d93963625f` | 变体 manifest（seed 偏移 +100000/+200000） |
| 5 | `/tmp/opencode/KEEP/v7_p0_specs/determinism_probe8.json` | `docs/v7_reports/specs/p0_variants/determinism_probe8.json` | `f548cb02f6881baab02b68d8c315825f31199ee64306241c4408ad2d88260777` | 8 条确定性探针子集 |
| 6 | `/tmp/opencode/KEEP/v7_p0_specs/probe_299.json` | `docs/v7_reports/specs/p0_variants/probe_299.json` | `62e6b73656bc2f1075fc2ebdb373ea955890ccd11055afbfa6549fb7d54a07bd` | id 299 单条探针 |
| 7 | `/tmp/opencode/KEEP/v7_q6q2/collision_replay.json` | `docs/v7_reports/evidence/collision_replay.json` | `8413de0baff9473b1fb9eef7258d82bf37914d0ec1eb51d9b7487cfafdbcf8ab` | 101 条碰撞仪器化重放原始 JSON（生成脚本已清理，不可原地重生成） |
| 8 | `/tmp/opencode/KEEP/v7_q6q2/tollgate_viz.json` | `docs/v7_reports/evidence/tollgate_viz.json` | `55d09fda13618d8e415963981c00ad4037d3a3ff4a7d84becec751b6d8d1a91b` | tollgate 运行时度量 JSON（同上） |

> 清理记录存档：`docs/v7_reports/cleanup/`（CLEANUP_PLAN / DELETED_MANIFEST / CLEANUP_A_REPORT / CLEANUP_B_REPORT / KEEP_README / **CLEANUP_C_REPORT / CLEANUP_C_DELETED / CLEANUP_C_WILL_DELETE**）；/tmp 引用映射总说明：`docs/v7_reports/MIGRATION_NOTE.md`。
