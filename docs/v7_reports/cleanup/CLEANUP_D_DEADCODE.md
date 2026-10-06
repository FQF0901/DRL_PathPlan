# 锁版清理 D —— 生产 dead-code 候选报告（只报告，不删）

> 生成：2026-10-06（清理 D）。范围：`env/`、`net/`、`pipeline/`、`tools/`、`config/`、`reward_model/` 中的**可疑废弃项**（未使用模块/函数/臂配置；已关闭路线代码；未跑/被取代臂配置）。
> 判定：全仓引用检索（`rg`，排除 `tests/`）+ 调用可达性（入口/配置/文档/证据链）+ 默认开关状态。**本批未删除任何生产代码/配置**。
> 建议口径：`保留[默认关+有测试+文档]` / `归档` / `删除待批`（需用户裁决；删除前须同步测试与证据说明）。
> 关联：`CLEANUP_D_REPORT.md`（tests 审计）、`CLEANUP_C_REPORT.md`（runs/datasets 处置先例）。

## 0. 候选汇总

| # | 候选 | 默认状态 | 引用/调用检查 | 风险 | 建议 |
|---|---|---|---|---|---|
| DC-1 | v5 结构线 obs（lane/ttc 通道全链） | **默认开** | 默认通道 + 证据链 | 中（证据 ckpt 复现） | 保留（默认开+有测试+文档）；路线关闭待用户裁决 |
| DC-2 | K-anchor 计划头 | **默认关**（enabled=false） | 配置开关 + 证据链 | 低 | 保留[默认关+有测试+文档] |
| DC-3 | fix-11 phase3 安全配方 | 默认关（显式 CLI） | 显式开关 + 证据链 | 低 | 保留[默认关+有测试+文档] |
| DC-4 | 未运行臂配置（v6 arm6/7/8、v7 arm2/3） | 不参与运行 | 仅预注册/报告/dry-run 测试 | 低 | 归档 / 删除待批 |
| DC-5 | 未跑臂专用奖励项（lead_gap/lane_boundary/lane_center） | 默认关 | 注册表 + dry-run 测试 | 低 | 保留（默认关+有测试+文档）；随 DC-4 裁决 |
| DC-6 | `tools/collect_expert.py` 串行旧路径三函数 | 无调用 | 0 生产调用（详见 §6） | 低 | label_statistics 删除待批；另两项保留或删除待批 |
| DC-7 | 文档小项（tools/README `measure/smoke_env.py` 条目） | — | 指向不存在文件 | 极低 | 修正 README（待批） |

**非候选（复核后排除）**：所有默认关但现行可开的 v3/v4/v6 选项（`target_kl` / `value_lr_scale` / `policy_logstd_max` / `critic_warmup_updates` / `adv_norm=per_scenario` / `spec_rotation=off` / `plan_reference=plan` / `freeze=all` / `legacy_tags` / `--allow-legacy-dataset`）——均被现行 CLI/config 解析链引用且有测试+文档，属"现行能力"而非废弃。

---

## DC-1 v5 结构线：obs v5 lane/ttc 通道（已关闭路线，代码默认开）

- **位置**：
  - 实现：`env/obs/lane.py`、`env/obs/ttc.py`；
  - 接线：`env/obs/builder.py::DEFAULT_CHANNELS`（含 `lane`/`ttc`）、`tools/collect_expert.py::CURRENT_CHANNELS`、`pipeline/buffer.py::DEFAULT_CHANNELS`、`pipeline/trainer.py::SINGLE_SLOT_CHANNELS`/`BCDataset._obs_keys`、`net/encoders.py`（`embed_lane/embed_ttc`）、`net/model.py::_struct_context_tokens`。
- **引用/调用检查**：
  - 主树**默认开**：`rg -n 'DEFAULT_CHANNELS' env/obs/builder.py pipeline/buffer.py` → lane/ttc 在默认元组；新建采集/训练/评测会产/消费 v5 键。
  - 现行实验（w1/arm1/P4-extra/§13/§14）在 **pre-v5 worktree（`2f4450e`）**执行（`docs/v7_program_prereg.md` §14："期间禁改代码"），即主线刻意避开主树 v5 代码以保持与 w1 基座可比。
  - 证据链：`runs/BTC20261005-0856_v7struct_v5{,_p3,_p3fix}`（保留）+ `docs/v7_reports/v7_struct_a_obs_v5.md`、`v7_struct_a2_wiring_probe.md`、`v7_struct_fail_diag.md`、`v7_p3fix_retry.md`。
  - 测试：`tests/test_obs_v5_lane_ttc.py`(16)、`tests/test_v5_pipeline_wiring.py`(5)。
- **风险**：若删除，① v5struct 证据 ckpt 无法复现/评测；② 主树默认观测口径变化，需同步裁决 v4 数据缺键回退边界（现行主线 ckpt 依赖该回退）；③ 16+5 个测试需同步删除。
- **建议**：**保留**（默认开+有测试+文档；证据 ckpt 复现依赖）。如用户裁决"结构线资产仅留报告不留复现"，再连同 DC-2/DC-3 一起归档/删除（前置：改写证据说明）。

## DC-2 v5 结构线 B：K-anchor 计划头（已关闭路线，默认关）

- **位置**：`net/anchor.py`（371 行）、`net/plan_head.py::plan_anchors`、`net/model.py`（`num_anchors>0` 分支）、`pipeline/stages.py::build_model`（plan_anchor 解析）、`pipeline/trainer.py::anchor_wta_loss`、`config/model.yaml::plan_anchor`（`enabled: false`）、`config/plan_anchors_k6.json`、`tools/fit_plan_anchors.py`、`config/arms/v7_struct_b_v5_{model,train,eval}.yaml`。
- **引用/调用检查**：
  - `enabled=false` → `num_anchors=0` → 无新参数/新输出，与旧版逐位一致（`pipeline/stages.py:169-181` 注释与实现）；旧 ckpt 严格加载不受影响。
  - 启用仅 v5 结构线（已关闭）；保留 run 的 `model.snapshot.yaml`/`config.snapshot.yaml` 引用（证据复现）。
  - `datasets/BTC20261002-0941_expert5k_v41` 的保留理由之一 = anchor 默认引用（`CLEANUP_C_REPORT.md` §5）。
  - 测试：`tests/test_plan_anchor.py`(18)、`tests/test_phase3_safe_recipe.py`(10)。
- **风险**：默认关，保留成本低；删除需同步 28 个测试 + 证据复现说明 + `plan_anchors_k6.json` 的 dataset 保留理由更新。
- **建议**：**保留[默认关+有测试+文档]**（默认推荐）。

## DC-3 fix-11 phase3 安全配方（已关闭路线的补救机制，默认关）

- **位置**：`pipeline/stages.py::_PHASE3_SAFE_FREEZE`、`_PHASE3_ANCHOR_HEAD_PREFIXES`、`_phase3_guard_argv/_phase3_guard_decision`、`run_stage_b_phase3` 的 `freeze=trunk_only` 分支（`--phase3-freeze`/`--phase3-guard-spec`/`--phase3-guard-min-success`）。
- **引用/调用检查**：仅显式 CLI 启用（默认 `specific_only`/无 guard）；`rg -n 'trunk_only' config/*.yaml config/arms/*.yaml` → 无配置默认启用；测试 `tests/test_phase3_safe_recipe.py`(10) 覆盖；证据 run `runs/BTC20261005-0856_v7struct_v5_p3fix`（保留）。
- **风险**：低（默认关；但含 `_SPECIFIC_PHASE_FREEZE` 锚头冻结契约修复——该修复对启用 K-anchor 的 phase3 是正确性修复）。
- **建议**：**保留[默认关+有测试+文档]**；随 DC-1/DC-2 一并裁决。

## DC-4 未运行臂配置（失败/降级臂）

- **位置**：
  - v6 P4（预算门截断未跑，`docs/v6_reports/v6_p4_report.md` §1）：`config/arms/arm6_lane_boundary.yaml`、`arm7_lane_center.yaml`、`arm8_collision_suppress_{gap,term,term46}.yaml`（arm8 设计后被 v7 P4-extra 碰撞抑制取代）；
  - v7 §10（预注册后降级未跑，`docs/v7_program_report.md` §1）：`config/arms/v7_arm2_bundle_kl.yaml`、`v7_arm3_bundle_only.yaml`。
- **引用/调用检查**：仅预注册/报告（v6_prereg §7、v6_p4_report、v7_prereg §10、v7_report）+ dry-run 测试（`tests/test_p4_arm_configs.py` 14、`tests/test_v7_p2_arm_config.py` 6）；无对应 run 目录（`ls runs/ | grep p4_arm[678]` 为空）。
- **风险**：删除需同步 dry-run 用例的臂清单参数化；配置体量 KB 级，保留成本低。
- **建议**：**归档 / 删除待批**（v6 arm6/7 已明确"NOT RUN"；arm8 被 v7 P4-extra 取代；arm2/3 预注册降级）。删除前置：同步测试清单 + 预注册文档注记。

## DC-5 未跑臂专用奖励项

- **位置**：
  - `reward_model/terms.py::LeadGapPenalty`（`lead_gap`）——仅 `arm8_collision_suppress_gap.yaml` 启用；
  - `LaneBoundaryPenalty`（`lane_boundary`）——仅 `arm6_lane_boundary.yaml`；
  - `LaneCenterPenalty`（`lane_center`）——仅 `arm7_lane_center.yaml`（`config/train.yaml` 注释作为"默认关车道保持项"文档化）。
- **引用/调用检查**：注册表 + terms 实现 + 测试（`tests/test_reward_terms.py`、`tests/test_p4_arm_configs.py`、`tests/test_lane_lateral.py`）；上下文键 `lead_gap_m/lead_speed_mps/lane_half_width_m` 为现行 v4⑦ 接口（`pipeline/trainer.py::LANE_CTX_KEYS`）。
- **风险**：低（默认关；依赖项 ctx 键仍为现行）。
- **建议**：**保留（默认关+有测试+文档）**；若 DC-4 归档，可一并裁决（删除需同步 terms 用例与 dry-run 用例）。

## DC-6 `tools/collect_expert.py` 串行旧路径三函数（0 生产调用）

> 背景：`71dd19f`（分片/并行采集重构）后生产路径 = `_collect_* → _concat_shards → balance_from_specs → _write_dataset`（`main` 与 `tools/dagger_collect.py` 均如此）；三个串行旧函数失去生产调用。

| 函数 | 位置 | 生产调用 | 现有引用 | 建议 |
|---|---|---|---|---|
| `apply_balance` | `tools/collect_expert.py:1144` | **0**（分片路径=`balance_from_specs`） | `tests/test_collect_expert_v2.py` 3 用例：2 个手算语义 + 1 个与 live 的三档差分等价（weights/cap/none） | **保留**（live 差分测试参考实现）或删除待批（需改写 3 用例） |
| `label_statistics` | `tools/collect_expert.py:1336` | **0**（报告统计已内联向量化） | 仅 1 个用例（本批已删除） | **删除待批** |
| `save_dataset` | `tools/collect_expert.py:1678` | **0**（生产=`_write_dataset`） | docstring 自述"单进程旧路径 / 测试用"；tests 3 处（1 个 schema roundtrip + 2 个 dagger fixture） | **保留**（测试/单进程封装）或删除待批（需改写 3 处测试调用） |

- **引用检查命令**：`rg -n '\b(apply_balance|label_statistics|save_dataset)\b' -g '!tests/**' .` → 除定义行与注释/docstring 外无调用。
- **风险**：低；`apply_balance` 若删除会同时移除 live 配平路径的手算 ground truth 与差分参考（建议保留或先补 live 手算用例）。
- **建议**：`label_statistics` 删除待批；其余两项按上表（保留优先）。

## DC-7 文档小项（tools/README 过期条目）

- `tools/README.md` 表格含 `measure/smoke_env.py`，但 `tools/measure/` 实际只有 `check_determinism.py`、`obs_smoke.py`（`tools/measure/README.md` 同样列出 smoke_env.py）。**建议**：修正两处 README（本批未动，避免与 tests 审计范围混淆）。
- `tools/diagnostics/{forensics_offline,forensics_report}.py` 仅被 `docs/archive/forensics-2026-09-26.md` 引用；`forensics_closed_loop.py` 被现行 `docs/v7_reports/v7_p0_gap_decomposition.md` 引为"建议下一步复用" → 三项均**保留**（诊断工具，成本低）。

---

## 附：与 tests 审计的交叉

- DC-1/2/3 的测试均保留（`CLEANUP_D_REPORT.md` §2/§4 边界项）。
- DC-4 的配置被 dry-run 测试锁定（测试保留）；删除配置时需同步。
- DC-6 的 `label_statistics` 用例已随本批删除；`apply_balance`/`save_dataset` 用例保留（差分/契约职责）。
