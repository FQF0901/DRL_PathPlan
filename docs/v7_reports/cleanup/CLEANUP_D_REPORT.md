# 锁版清理 D 报告 —— tests 审计与废弃测试清理（+ dead-code 候选报告）

> 执行：fixer 子代理；日期：2026-10-06；范围：`/workspace/01_Proj/DRL_PathPlan/tests/`（全 74 个测试文件 + 2 个 helper）。
> 前置：清理 A/B/C（`docs/v7_reports/cleanup/CLEANUP_{PLAN,A,B,C}_*`）。本批：**tests 逐文件审计**（分类 a 现行 / b 遗留兼容 / c 废弃 / d 噪声）+ **只删"被测对象已死"的测试** + **生产 dead-code 候选报告**（`CLEANUP_D_DEADCODE.md`，只报告不删）。
> 约束遵守：未动 `env/`、`net/`、`pipeline/`、`reward_model/`、`tools/`、`config/` 生产代码/配置；未动兼容/证据链；临时文件在 `/tmp/opencode`（结束后清理）。

## 0. 摘要

| 项 | 前 | 后 | 变化 |
|---|---|---|---|
| 测试文件 | 74 | 74 | 0（未删文件） |
| 用例（`pytest` collected） | **694** | **693** | **−1**（1 个 c 类用例） |
| 分类结果 | — | a 现行 74 文件（其中 9 文件含 b 遗留兼容子项）/ c 废弃 1 用例 / d 噪声 1 项（README 过期） | — |

- **c 类删除（1 用例）**：`tests/test_collect_expert_v2.py::test_label_statistics_count_and_weighted_conventions` —— 被测函数 `tools/collect_expert.py::label_statistics` **生产零调用**（报告统计已改内联向量化；全仓唯一引用就是该用例），属"代码路径被取代且无兼容职责"。
- **未删除 v5 结构线 / v6 未跑臂 / v4 旧开关测试**（边界项）：其被测代码仍在且可达（默认开或默认关+配置开关），按判定方法"符号仍被 import/调用 → 保留"，全部列入 `CLEANUP_D_DEADCODE.md` 供裁决。典型：
  - obs v5 lane/ttc：`DEFAULT_CHANNELS` 默认包含（现行默认观测），且 v5struct 证据 ckpt 复现依赖 → 保留；
  - K-anchor / fix-11：默认关、有测试+文档、证据链引用 → 保留；
  - v6 arm6/7/8、v7 arm2/3 未跑臂配置：配置存在且被 dry-run 测试锁定 → 测试保留，配置列入候选。
- **d 类清理**：`tests/README.md` 过期计数（"240 passed"→"693 passed"）与索引补充（本批唯一 d 项；全仓无无用例的 skip/xfail、无重复用例名、无空断言用例）。

## 1. 判定方法（可复核）

1. **符号级引用检索**：对每个测试文件用 AST 提取其 import 的生产符号（`env/net/pipeline/tools/reward_model`；含函数内惰性 import；共 312 个唯一符号），逐个在**全仓（排除 `tests/`）**检索 `\b<symbol>\b`，并区分"定义/`__all__`/字符串引用"与"真实调用"；零调用者即"测试对象已死"候选。
   - 结果：仅 3 个符号零生产调用 —— `tools.collect_expert.{apply_balance, label_statistics, save_dataset}`（详见 §4 与 dead-code 报告 DC-6）。
2. **特性级复核**（对"版本命名"文件）：`test_stage_v11` / `test_integration_v12` / `test_obs_schema_v2` / `test_monitoring_v2` / `test_collect_expert_v2` / `test_v5_pipeline_wiring` / `test_stage_c_*` / `test_v6_*` 等逐一核对被测行为是否仍属现行程序（配置默认、入口调用、证据链），而非仅凭文件名判断。
3. **兼容职责复核**：凡被测行为属 v4 代 ckpt/数据回退（`net/mem.py::_remap_legacy_others` 28→33、obs 缺键回退、旧 schema 数据集、`legacy_tags`、`freeze=all`、`spec_rotation=off`、`--allow-legacy-dataset` 等）→ 一律 b 类保留。
4. **不确定 → 保留并列入报告**（本批边界项全部按此处理）。

复核命令示例（报告内结论均可按此重放）：

```bash
# c 类删除依据（唯一引用 = 测试）
rg -n '\blabel_statistics\b' -g '!tests/**' .        # 仅 tools/collect_expert.py:1336 定义
# v5 默认开（保留依据）
rg -n 'DEFAULT_CHANNELS' env/obs/builder.py pipeline/buffer.py   # lane/ttc 在默认元组内
# K-anchor 默认关（保留依据）
rg -n 'plan_anchor' config/model.yaml pipeline/stages.py        # enabled: false → num_anchors=0
```

## 2. 执行与 dry-run（删除清单）

**dry-run 候选 → 判定 → 执行**：

| 候选 | 被测符号生产调用 | 判定 | 处置 |
|---|---|---|---|
| `test_label_statistics_count_and_weighted_conventions` | `label_statistics`：0（报告已内联向量化；唯一引用=该用例） | **c 废弃** | **删除（1 用例）** |
| `test_balance_weights_use_trainable_rows_only` / `test_balance_cap_zeroes_weights_without_deleting_rows` | `apply_balance`：0 生产调用，但被 `test_balance_from_specs_matches_apply_balance` 用作 **live `balance_from_specs` 的差分测试参考实现**（weights/cap/none 三档等价 + 手算 ground truth） | 边界（有现行测试职责） | 保留（列入 DC-6，待裁决） |
| `test_save_dataset_v2_schema_roundtrip`（+ `test_dagger_collect` 2 处 fixture 调用） | `save_dataset`：0 生产调用，但 docstring 标注"单进程旧路径 / 测试用"，且用例锁定 **live `_write_dataset` 的 v2 npz/meta schema 契约** | 边界（测试封装） | 保留（列入 DC-6，待裁决） |
| v5 结构线测试（`test_plan_anchor` 18 / `test_obs_v5_lane_ttc` 16 / `test_v5_pipeline_wiring` 5 / `test_phase3_safe_recipe` 10） | 符号均被生产 import/调用（默认开或默认关+配置）；证据 ckpt 复现依赖 | a（含 b 兼容子项） | 保留（列入 DC-1/2/3） |
| v6 未跑臂配置测试（`test_p4_arm_configs` 14 / `test_v7_p2_arm_config` 6） | 配置 loader/`build_reward_adapter` 为现行代码 | a（配置本身入候选 DC-4/5） | 保留 |

**实际删除**（逐项见 `CLEANUP_D_DELETED.txt`）：

- `tests/test_collect_expert_v2.py::test_label_statistics_count_and_weighted_conventions`（1 用例）+ 同步移除该文件对 `label_statistics` 的 import（避免悬空 import）。

## 3. 逐文件分类（74 文件 / 693 用例；a=现行，b=遗留兼容，c=废弃，d=噪声）

> 计数为删除后 `pytest --collect-only` 实测。b 类为文件内部分用例的兼容职责（文件同时含 a）。判定证据以生产代码引用/配置默认/入口调用为准。

### 3.1 数据/观测/缓冲/帧

| 文件 | 用例 | 分类 | 覆盖对象与判定依据 |
|---|---:|---|---|
| `test_obs_schema_v2.py` | 8 | a | OD 固定槽位/驱逐/presence/6 帧顺序：`env.obs.od` 现行实现 |
| `test_obs_v5_lane_ttc.py` | 16 | a+b | v5 lane/ttc 通道（builder/buffer/collect 默认通道，net 消费）+ **v4 缺键回退**（现行 ckpt 依赖） |
| `test_p1a_static_obs.py` | 11 | a+b | schema v4 static 段现行 + **28→33 `_remap_legacy_others` 回退**（v4 数据加载依赖） |
| `test_frames_lookup.py` | 13 | a | `pipeline.frames` 精确查表：现行 |
| `test_buffer_gae.py` | 11 | a | RolloutBuffer/GAE/历史重建：现行 |
| `test_fast_data_path.py` | 6 | a | 物化快路径 vs 旧逐批：现行训练路径 |
| `test_limit_dataset.py` | 8 | a | `--limit-dataset` 前缀截断：现行 |
| `test_dataset_contract_guard.py` | 10 | a+b | schema<2 硬失败（现行）+ `--allow-legacy-dataset` 逃生门（兼容） |
| `test_ego_prev_action.py` | 3 | a | ego reserved 维注入：现行 |
| `test_v5_pipeline_wiring.py` | 5 | a+b | BCDataset/SINGLE_SLOT/缓冲区 lane/ttc 接线（现行默认）+ **v4 数据无 struct 键加载**（兼容） |

### 3.2 网络/策略

| 文件 | 用例 | 分类 | 覆盖对象与判定依据 |
|---|---:|---|---|
| `test_net_shapes.py` | 21 | a | net 形状/mask/MoE/WM/参数预算：现行 |
| `test_mem_rollout.py` | 13 | a | rollout 拷贝隔离/detach/梯度表：现行 |
| `test_od_pose_grad.py` | 3 | a | 空 OD 槽位 NaN 梯度修复：现行 |
| `test_v6_attn_heads.py` | 22 | a | v6 交叉注意力头/t0 单次 st_gnn/nav 重建（现行 net 结构，v6_net_design 冻结） |
| `test_v6_world_nav.py` | 15 | a | `env.obs.world` + nav 重建 + buffer 透传：现行（A4） |
| `test_plan_anchor.py` | 18 | a | K-anchor：默认关（`plan_anchor.enabled=false`，关闭逐位兼容）——闭合路线，见 DC-2 |

### 3.3 训练器/阶段 A/B/phase3/Stage C

| 文件 | 用例 | 分类 | 覆盖对象与判定依据 |
|---|---:|---|---|
| `test_adv_norm.py` | 14 | a | `adv_norm` 选项 + GAE padding 修复：现行（train.yaml `adv_norm: global`） |
| `test_anchor_grad_probe.py` | 10 | a | v4④ 锚梯度探针：默认开、只读监控 |
| `test_grad_group_probe.py` | 7 | a | v4③ 分参数组梯度探针：默认开 |
| `test_episode_probes.py` | 5 | a | v4⑤ episode 数据探针：现行监控 |
| `test_trainer_noop_probes.py` | 3 | a | v4⑧ 默认开关 no-op 指纹：现行 |
| `test_stage_c_drift_probe.py` | 6 | a | G4 漂移探针（`probe_interval` 默认 25）：现行 |
| `test_stage_c_effective_action.py` | 5 | a+b | P0-1 A-hold（默认 `repeat_action`）+ `plan` 旧对照 |
| `test_stage_c_label_alignment.py` | 3 | a+b | 标签同帧契约 + 旧接口回退 |
| `test_stage_c_lam.py` | 3 | a | `--lam` 0.95/0.98：现行选项 |
| `test_stage_c_horizon.py` | 6 | a | P4 horizon 200 步：现行（train.yaml 默认） |
| `test_stage_c_obs_consistency.py` | 3 | a | R7 collect/update 观测口径：现行 |
| `test_stage_c_periodic_ckpt.py` | 4 | a | C1 周期 ckpt：现行（`ckpt_every`） |
| `test_stage_c_trainable_scope.py` | 2 | a | R2 冻结 allowlist：现行 |
| `test_stage_c_wm_signal.py` | 2 | a | W1 WM 冻结语义：现行 |
| `test_target_kl_gate.py` | 5 | a | v4② `target_kl` 守门：现行选项（默认 null） |
| `test_value_lr_scale.py` | 5 | a | v4① `value_lr_scale`：现行选项（默认 1.0） |
| `test_policy_logstd_max.py` | 7 | a | S2 `logstd_max` 钳制：现行选项（默认 null） |
| `test_critic_warmup.py` | 2 | a | `critic_warmup_updates`：现行选项（默认 0） |
| `test_bc_pretrain.py` | 6 | a | BC 目标对齐/动作损失：现行 |
| `test_traj_metrics.py` | 4 | a | traj 度量口径修复：现行 |
| `test_weight_accounting.py` | 9 | a | 权重感知会计：现行 |
| `test_checkpoint_resume.py` | 4 | a | 周期 ckpt/resume：现行 |
| `test_resume_device.py` | 4 | a | 跨设备 resume：现行 |
| `test_stage_v2_smoke.py` | 2 | a | Stage A/B 冒烟（schema v2 契约）：现行 |
| `test_stage_v11.py` | 8 | a | v1.1 契约：GL 守卫/未来窗口 stride/WM detach/freeze 前缀：现行 |
| `test_stage_a_ld_loss.py` | 4 | a | A 的 `ld_coef=0` 新语义（LD 仅监控）：现行 |
| `test_stage_b_monitor_holdout.py` | 5 | a | B 监控/留出：现行 |
| `test_stage_b_phase3.py` | 13 | a | phase3 机制（锁版 Stage B）：现行 |
| `test_phase3_recipe_switches.py` | 10 | a+b | freeze/anchor 开关（`freeze=all`/anchor off 旧行为回归）：现行+兼容 |
| `test_phase3_safe_recipe.py` | 10 | a | fix-11 安全配方 + 锚头冻结契约修复：默认关，见 DC-3 |
| `test_phase3_loop.py` | 45 | a | phase3 全自动循环（R1/R4/keep-best）：现行编排 |
| `test_integration_v12.py` | 9 | a | v1.2 集成契约 + `net.world_model` 删除守卫：现行 |
| `test_v6_e1_router_monitor.py` | 4 | a | v6 E1 router 监控 top-2 口径修复 + 专家范数探针（默认开、no-op）：现行 |

### 3.4 奖励/评测/KPI/工具

| 文件 | 用例 | 分类 | 覆盖对象与判定依据 |
|---|---:|---|---|
| `test_reward_terms.py` | 43 | a | 奖励项/势能塑形/CaRL/KPI：现行 |
| `test_reward_ctx_lane_keys.py` | 7 | a | v4⑦ ctx 键（lead_gap_m 等）：现行 |
| `test_reward_ctx_speed_limit.py` | 8 | a | v4⑥ 限速来源/终局回退：现行（默认生效修复） |
| `test_reward_adapter_hook.py` | 4 | a | `build_reward_adapter` 钩子：现行 |
| `test_rc_tier_pairing.py` | 12 | a | rc 档配对守卫：现行（docs/reward_audit 引用） |
| `test_kpi_grouping.py` | 14 | a | Wilson CI/分组/弱类 floor：现行 |
| `test_paired_eval.py` | 21 | a | v7 配对评测工具（Gate A 方差闸）：现行 |
| `test_eval_reference.py` | 5 | a | `--eval-reference`（默认 plan，pin 协议）：现行 |
| `test_reward_audit.py` | 17 | a | P2 奖励审计工具：现行（P4 臂配置来源） |
| `test_lane_lateral.py` | 5 | a | `lane_lateral_info` → 默认关 `lane_center` 项输入：注册表在册（DC-5） |
| `test_lane_u1_moe_balance.py` | 19 | a | lane U1 去聚类/负载均衡/权重化 specific：现行 |
| `test_p4_arm_configs.py` | 14 | a | v6 P4 臂 dry-run（含未跑 arm6/7/8，配置入 DC-4） |
| `test_v7_p2_arm_config.py` | 6 | a | v7 §9/§10 臂配置 dry-run（arm2/3 未跑，入 DC-4） |
| `test_v7_reward_arms.py` | 5 | a | v7 §13 奖励臂 A/B/C dry-run：现行 |
| `test_max_step_wiring.py` | 6 | a | P4 max_step 接线：现行 |
| `test_spec_rotation.py` | 7 | a+b | S1 spec 轮换（默认 episode）+ `off` 旧行为 |
| `test_router_labels_protocol.py` | 4 | a | P0-2 标签协议：现行 |
| `test_monitoring_v2.py` | 3 | a+b | 监控瘦身 + `legacy_tags` 回退 |
| `test_monitoring_tb_grouped.py` | 5 | a | Tier-1 tag/多线 TB：现行 |
| `test_run_naming.py` | 8 | a | runs 命名规范化：现行 |
| `test_run_paths.py` | 5 | a | 入口布局契约：现行 |
| `test_dagger_collect.py` | 32 | a | DAgger 工具（P1 闭环重标注）：现行 |
| `test_collect_expert_v2.py` | 15 | a（c×1 已删） | collect_expert v2 契约；删除 1 个死函数用例（§2） |
| `test_collect_expert_parallel.py` | 6 | a | `--workers` 分配/合并：现行 |
| `test_debug_rollout_viz.py` | 10 | a | 回灌可视化纯几何（tools/README 在册工具）：现行 |

### 3.5 helper / 文档

| 文件 | 分类 | 说明 |
|---|---|---|
| `tests/v2_synthetic.py` | a | 合成数据集 helper（20 处引用）：现行 |
| `tests/__init__.py` | a | 包标记 |
| `tests/README.md` | d | 过期计数与索引 → 本批更新（见 §5） |

## 4. 边界项（保留理由，逐条）

1. **"版本命名"≠废弃**：`test_stage_v11`（v1.1 契约）、`test_integration_v12`（v1.2 契约）、`test_obs_schema_v2`（schema v2 槽位语义）、`test_monitoring_v2`（Tier-1 瘦身 + legacy 回退）、`test_collect_expert_v2`（数据集 schema 版本号仍为 2）、`test_stage_v2_smoke`（Stage A/B 冒烟）——文件名中的版本号是**契约版本**，被测行为全部仍属现行代码/兼容职责。
2. **v5 结构线（已关闭路线）**：代码默认开（lane/ttc）或默认关（K-anchor/fix-11），且 v5struct 证据 run 与报告在仓（`runs/BTC20261005-0856_v7struct_v5*`、`docs/v7_reports/v7_struct_*.md`）→ 被测对象未死；**保留测试**，生产代码列入 dead-code 候选（DC-1/2/3）。
3. **未跑臂配置**：v6 arm6/7/8、v7 arm2/3 从未运行，但其配置被 dry-run 测试锁定（loader/adapter 为现行代码）→ 测试保留；配置列入候选（DC-4/5）。
4. **v4 代兼容**：`_remap_legacy_others`、obs 缺键回退、旧 schema 数据集、`legacy_tags`、`freeze=all`、`spec_rotation=off`、`--allow-legacy-dataset`、`plan_reference=plan`、旧标签接口回退 → b 类保留（任务指定保护对象）。
5. **默认关但现行选项**：`target_kl`/`value_lr_scale`/`logstd_max`/`critic_warmup`/`adv_norm=per_scenario`/`plan_anchor`/fix-11 → 默认关+配置可开+有测试+有文档 → 保留。

## 5. d 类噪声清理

- `tests/README.md`：运行计数 "当前 240 passed" → **"当前 693 passed"**；补充清理 D 审计索引段（指向本报告）。
- 全仓扫描确认无以下噪声：无用例的 `skip/xfail`（现有 skip 均为条件式：CUDA / metadrive / scipy / tensorboard / env specs）；重复测试函数名（0）；无断言用例（0）；悬空 fixture/helper（0，helper 均被引用）。
- 本批未发现需要删除的重复用例；`test_balance_from_specs_matches_apply_balance` 与 2 个 `apply_balance` 用例构成差分测试，非重复。

## 6. 验证

- **测试**：`.venv/bin/python -m pytest tests/ -q` → **693 passed**, 145 warnings（exit 0）。删除前基线：694 passed。
- **删除范围**：仅 `tests/`（1 用例 + 1 行 import）；`git status` 确认未触碰生产代码/配置。
- **引用复核**：删除后全仓（排除 tests）对 `label_statistics` 的检索仅剩定义行 `tools/collect_expert.py:1336`。

## 7. 与任务口径的偏差（如实）

1. **未删文件、仅删 1 用例**：按任务指定方法（符号仍被生产 import/调用 → 非死），v5 结构线/未跑臂/旧开关的测试全部命中"保留"分支；真正"被测对象已死"的仅 `label_statistics` 1 个用例。其余 2 个死函数（`apply_balance`/`save_dataset`）的测试承担 live 差分/契约职责 → 保留并列入报告。
2. **未动 `tests/` 之外的任何代码**（含 `tools/collect_expert.py` 死函数本体）——按约束 dead-code 只报告。
3. **v5 线处置待裁决**：DC-1/2/3 给出保留建议与删除前置条件（证据 ckpt 复现、测试同步），未擅自删除。
