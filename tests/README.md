# tests

目的：骨架 / 契约 / 回归测试。**全部为 CPU 离线测试，不打开 MetaDrive 仿真环境**：env/obs 层用
stub / fake 对象，训练与数据链路用合成数据集、小型模型与 stub 池。

## 运行方式

统一使用 `tools/venv-python`（包装 `.venv/bin/python` 并注入 glvnd 运行库）；裸 `pytest` 可能落到
系统 Python。命令均在**仓库根目录**执行：

    # 单文件
    tools/venv-python -m pytest tests/test_arm_configs.py -q

    # 子集（多文件 / 关键字过滤）
    tools/venv-python -m pytest tests/test_stage_c_horizon.py tests/test_run_paths.py -q
    tools/venv-python -m pytest tests/ -q -k reward

    # 全量（2026-10-15 核对：107 个 test_*.py，collect 966 项）
    tools/venv-python -m pytest tests/ -q

    # 可选并行（pytest-xdist 已安装）
    tools/venv-python -m pytest tests/ -q -n 2

仓库无 pytest.ini / pyproject 测试配置，也没有 CI 配置（无 `.github/` 等）；测试全部为本地手动运行。

## 环境依赖与自动 skip

- `tests/test_lane_lateral.py` 与 `tests/test_forensics_footprint.py`（oracle 分支）在用例内
  `pytest.importorskip("metadrive")` 并惰性 import `env.metadrive_env`（仍不建 env）；缺 metadrive 包时
  这两个文件自动 skip。`test_paired_eval.py`（scipy.stats）、`test_v5_pipeline_wiring.py`（torch）同样用
  `importorskip` 惰性导入。真实几何 / 环境的验证由 `tools/measure`、`tools/diagnostics` 等脚本与冒烟流程承担，不在 `tests/`。
- 无 CUDA 时 `test_resume_device.py` 的 cuda 用例 skip。
- 无 `setsid` 时 `test_phase3_loop.py` 的后台化用例 skip。
- `test_reward_audit.py` 的排除集用例在 `env/specs/*.json`（gitignored 产物）缺失时 skip。
- `test_stage_v11.py` 的 GL 守卫用例在 venv 本地 glvnd 目录不存在时 skip。

## 测试清单（按主题）

### env / obs 契约

| 文件 | 覆盖 |
| --- | --- |
| `test_obs_schema_v2.py` | OD 固定槽位分配/驱逐/presence、6 帧 mem 顺序与 `hist_valid`、schema 键与 dtype |
| `test_obs_v5_lane_ttc.py` | obs schema/manifest（当前 v7，断言 v7 前缀）：LD offset `{5,10,15,20,30}`、lane/ttc 通道删除 |
| `test_p1a_static_obs.py` | 静态障碍几何（走廊/净距/相对车道/身后）、`others` static 段落位、旧数据回退 |
| `test_v6_world_nav.py` | obs 世界系键、nav 逐步重建、teacher forcing 同步、buffer 透传 |
| `test_ego_prev_action.py` | ego reserved 维（上一策略动作）注入；未注入/非法值保持 0 |
| `test_lane_lateral.py` | `lane_lateral_info` 读取顺序与降级（stub；缺 metadrive 时 skip） |
| `test_max_step_wiring.py` | max_step 截断 → `info["max_step"]` → 终局值结算接线 |
| `test_v5_pipeline_wiring.py` | obs v6 通道表在 BCDataset / RolloutBuffer 的接线 |
| `test_lane_semantic_contract.py` | R21-0 `lane_semantic` 通道契约（schema v7）：形状/特征名/dtype/mask、几何查询门、禁 raw lane_id |
| `test_target_lane_continuity.py` | target lane 选择纯函数：优先级链（route → next_ref → ego.lane）+ 稳定性门 |
| `test_ld_group_topology.py` | R21-0 LD 点补充拓扑门（role / offset_s 严格递增 / 同组不混 lane / route 关联） |
| `test_r21_3a_semantic_wiring.py` | R21-3A 语义接线契约（round-23 §6A 十项 + 归一化逐位 + 改名一致性） |
| `test_round18a_ld_debug.py` | LD 旁路 debug 元数据（默认关；debug 开关下 feats/mask 逐位不变） |
| `test_round20_ld_ablation.py` | Round-20 消融开关：M1（关 LD-LD 链）/ M2（LD set-attention）默认关、既有路径逐位不变 |
| `test_round22_m2ab.py` | Round-22 Gate A：M2a/M2b（`LdAggregate`）`alpha=0` 与 baseline 逐位一致 |

### net 模型 / 网络

| 文件 | 覆盖 |
| --- | --- |
| `test_net_shapes.py` | net 形状/掩码/`hist_valid` 门控、MoE top-2、ST-GNN 先验、参数预算、PPO cheap path |
| `test_od_pose_grad.py` | 空 OD 槽位 `atan2(0,0)` 的 NaN 不得污染 `traj_xy` 梯度 |
| `test_v6_attn_heads.py` | 交叉注意力头 A1/A2、A4 nav 逐步重建、Stage C allowlist 命名契约 |
| `test_v6_e1_router_monitor.py` | router 监控 top-2 口径修复 + 专家输出范数探针 |
| `test_lane_u1_moe_balance.py` | 去聚类 + MoE 负载均衡 + 权重化 specific |
| `test_plan_anchor.py` | K-anchor 计划头：锚字典 / WTA / 车道系 / 软混合 / PPO 兼容 |
| `test_mem_rollout.py` | rollout 语义：拷贝隔离、detach、梯度表、MoE 软目标接口 |
| `test_r21_3d_modes.py` | R21-3D `semantic_no_chain` 单变量模式 + rollout semantic 三种口径 |
| `test_r25_residual.py` | R21-25A 残差式语义接入契约（Gate A；零训练） |
| `test_r26_delta.py` | R21-26B `semantic_off` 基底前向逐位 + delta 监督绑定语义路径 |
| `test_r28_coupling.py` | R21-28 横纵耦合审计（κ/legacy-rollout 升级、标签元数据/manifest、LQR 遥测） |
| `test_r29_fixes.py` | R21-29 修复：κ 真掩码 / 低 ds 三态 / 负 ds fail-fast / LQR 前馈（默认关） |

### pipeline：阶段 A/B 与数据

| 文件 | 覆盖 |
| --- | --- |
| `test_stage_v11.py` | v1.1 阶段契约：GL 守卫、未来窗口 stride 查表、WM detach、freeze 前缀 |
| `test_stage_v2_smoke.py` | Stage A/B 小样本冒烟（合成 v2 数据集 + tiny 模型，CPU 数秒） |
| `test_stage_a_ld_loss.py` | Stage A WM latent consistency 主损失 + 物理解码诊断 + 掩码语义 |
| `test_stage_b_monitor_holdout.py` | Stage B 逐 epoch 监控、episode 留出、Tier-1 tag |
| `test_stage_b_phase3.py` | stage B phase3：dagger 单数据集、全参数可训、LR 分组 |
| `test_bc_pretrain.py` | BC 目标对齐 / 动作损失非零 / 策略头参数化与 logprob 契约 |
| `test_critic_warmup.py` | critic warmup 只拟合 value 头，之后恢复 PPO |
| `test_dataset_contract_guard.py` | Stage A/B 启动数据集契约守卫（防静默回退） |
| `test_limit_dataset.py` | `--limit-dataset` 按完整 episode 前缀截断（不整包解压） |
| `test_fast_data_path.py` | 物化快路径（切片 + pin H2D）vs 旧逐 batch 重建等价 |
| `test_frames_lookup.py` | 帧精确查表：过滤洞 / 网格锚定 / stride / `wm_valid` / SE(2) |
| `test_weight_accounting.py` | count vs weighted 双口径、权重 0 不出力、软目标温度 |
| `test_traj_metrics.py` | 轨迹损失单位/口径（加权 MSE vs MAE） |
| `test_p1_3_geometry_losses.py` | P1-3 三项损失（`ld_line_margin` / `curvature` / `progress`）与梯度校准探针（默认关） |
| `test_phase3_traj_aux.py` | P1-2 phase3 traj（pose 链）损失项：默认关逐位不变、掩码只作用于 traj |
| `test_r27_residual_classes.py` | R21-27B N0/N1/C 定义锁定（`pipeline.residual_classes` 纯函数） |
| `test_r27_semantic_labels.py` | R21-27 B 行标签消费守卫（`pipeline.semantic_labels` 入口硬拒 v1） |
| `test_collect_recovery.py` | P1-1 recovery 采集器离线单测（窗口候选/三档分类/配额/池规划/一致性判定） |
| `test_merge_recovery_shards.py` | P1-1 recovery 分片合并器（合成分片 → merge → 拼接/split/数据门） |

### pipeline：PPO 与训练机制

| 文件 | 覆盖 |
| --- | --- |
| `test_buffer_gae.py` | 按帧缓冲 + 在线历史重建 + GAE(λ) 手算对照 |
| `test_adv_norm.py` | 优势归一化选项 + GAE padding 行修复 |
| `test_target_kl_gate.py` | `target_kl` 守门：超阈值跳过 epoch 剩余 minibatch |
| `test_policy_logstd_max.py` | 策略 logstd 上界钳制（采样与 logprob 同分布） |
| `test_value_lr_scale.py` | `--value-lr-scale`：value 参数组 lr 与组序 |
| `test_trainer_noop_probes.py` | 探针/守门默认开启与旧行为逐位一致 |
| `test_checkpoint_resume.py` | 周期 ckpt payload/RNG 捕获/续跑 |
| `test_resume_device.py` | 跨设备 resume 优化器状态搬移（无 CUDA 时部分 skip） |

### pipeline：协议 / 轮转 / 集成

| 文件 | 覆盖 |
| --- | --- |
| `test_router_labels_protocol.py` | LocalEnvPool 步前标签 + 终局 record/reset 分离 |
| `test_spec_rotation.py` | spec 每 episode 轮换 + `off` 旧行为回归 |
| `test_integration_v12.py` | 跨 lane 集成：阶段 B 冻结前缀、presence/entry id 轴 |

### pipeline：phase3 / 阶段 C

| 文件 | 覆盖 |
| --- | --- |
| `test_phase3_loop.py` | phase3 全自动循环编排（fake train.sh / venv-python 桩；setsid 用例可 skip） |
| `test_phase3_recipe_switches.py` | phase3 配方开关（`freeze=specific_only` 等） |
| `test_phase3_safe_recipe.py` | fix-11 安全配方：freeze 契约修复 + trunk_only + clean150 守护 |
| `test_stage_c_drift_probe.py` | 执行参考漂移探针（plan/mu RMS 位移） |
| `test_stage_c_effective_action.py` | A-hold：收集侧参考只依赖被 PPO 记账的 `a_t` |
| `test_stage_c_horizon.py` | horizon 对齐 200 策略步（CLI/config/臂回落 + `build_pool` 透传） |
| `test_stage_c_label_alignment.py` | router 标签与 buffer `obs_current` 同帧 |
| `test_stage_c_lam.py` | GAE λ 选项（CLI/config，0.95/0.98 档） |
| `test_stage_c_obs_consistency.py` | 阶段 C collect 与 update 观测口径一致 |
| `test_stage_c_periodic_ckpt.py` | 阶段 C 周期 ckpt 调度与 payload |
| `test_stage_c_trainable_scope.py` | 阶段 C 冻结范围（design allowlist）与参数组表 |
| `test_stage_c_wm_signal.py` | 阶段 C WM 全期冻结 ⇒ 无训练信号 |
| `test_recovery_recipe.py` | P1-2 recovery 配方（R1/R2）与混采/时间衰减机制（默认关；不跑真实训练） |
| `test_recovery_wiring.py` | P1-2 recovery_v1 → 训练集合并/混采/历史覆盖（默认关、逐位不变） |
| `test_recovery_quality_gates.py` | P1-0C 恢复质量门（合成 fixture 每门 True/False + CLI） |
| `test_a3_recipe_switches.py` | Round-17 A3 配方开关：尾链正则（默认关逐位不变）+ 六步衰减 CLI |

### reward（奖励模型 / 适配器 / 审计）

| 文件 | 覆盖 |
| --- | --- |
| `test_reward_terms.py` | 奖励项 / 势能塑形不变性 / CaRL / KPI 分组与判定 |
| `test_reward_adapter_hook.py` | `build_reward_adapter` config 接线与项权重覆盖 |
| `test_reward_ctx_lane_keys.py` | ctx 车道键（lead_gap / lead_speed / lane_half_width）透传 |
| `test_reward_ctx_speed_limit.py` | ctx 限速来源同源 + 终局帧掩码回退 |
| `test_rc_tier_pairing.py` | route_completion 权重与终局值同档守卫 |
| `test_arm_configs.py` | 当前臂 `v7_arm1_offroad.yaml` 加载语义 + `off_road_edge` 适配器生效 |
| `test_reward_audit.py` | 奖励审计工具：dry-run / 分区互斥 / 重放语义 |

### 监控与探针

| 文件 | 覆盖 |
| --- | --- |
| `test_monitoring_tb_grouped.py` | Tier-1 tag 重命名 + tensorboard 同 run 多线 |
| `test_monitoring_v2.py` | GroupedMetricStatistics + Tier-1 瘦身落盘 |
| `test_episode_probes.py` | 训练侧 episode 统计探针（termination/return/steps） |
| `test_grad_group_probe.py` | 分参数组梯度/更新范数探针（纯监控） |
| `test_anchor_grad_probe.py` | KL 锚梯度探针（只读、不进优化器） |
| `test_probe_action_ego_alignment.py` | P1-0C 跨行 action 链 ↔ `ego_world` 对齐探针（直行链 PASS；篡改 FAIL；缺数据集 NOT_AVAILABLE） |
| `test_probe_freeze_audit.py` | P1-0C freeze recipe 审计探针（静态层解析 + stub 模型，四层一致） |
| `test_probe_geometry_coverage.py` | P1-0C 几何代理覆盖探针（LD 槽/实线/plan 覆盖） |
| `test_probe_time_index.py` | P1-0C 时间索引探针（含恒速合成 episode 的链积分锚点） |
| `test_probe_train_account.py` | P1-0C 单步训练账本探针（trainable ⊆ optimizer、逐组 grad-norm/updated 计数） |

### 评测与工具

| 文件 | 覆盖 |
| --- | --- |
| `test_kpi_grouping.py` | 主标签分组 / Wilson CI / 弱类 floor（纯函数） |
| `test_eval_reference.py` | `--eval-reference` 语义（plan / repeat_action） |
| `test_paired_eval.py` | 配对评测统计 / CI / 分层 / 多 run 汇总 |
| `test_run_naming.py` | runs/ 命名规范化（自适应命名） |
| `test_run_paths.py` | run 布局/发现、共戳/resume、入口脚本 ≤30 行 + `bash -n` |
| `test_debug_rollout_viz.py` | 可视化纯几何/偏差函数（坐标变换、ADE/FDE 口径） |
| `test_collect_expert_parallel.py` | `collect_expert --workers` 轮转分配与合并确定性 |
| `test_collect_expert_v2.py` | collect_expert v2 数据集契约（权重、`wm_valid`、schema） |
| `test_dagger_collect.py` | DAgger-lite：策略 roll-in、专家空问标签、分片/扫描/报告 |
| `test_artifacts_writer.py` | P1-0C 统一产物补齐（九项产物生成/关键键/不覆盖/幂等）+ 契约探针 PASS/FAIL/NA |
| `test_eval_load_failfast.py` | 评测 ckpt 加载 fail-fast / `--allow-partial-load` / 记录字段 |
| `test_eval_model_config_forwarding.py` | 评测侧 model_config（`spatial`）透传守卫（漏传事故回归） |
| `test_forensics_d2_dth_scale.py` | forensics d2 `--d2-dth-scale`（dθ-only 反事实）纯函数 + 控制器接线 |
| `test_forensics_d2_tail_family.py` | forensics d2 尾族 `--d2-tail`（decay / dtheta_decay 形状）+ CLI |
| `test_forensics_footprint.py` | forensics footprint / 时间戳（D1/D2/repeat_action）CPU 单测（metadrive 仅 oracle 分支 skip） |
| `test_forensics_ref_world_head.py` | 探针 12：forensics 三轨 `ref_world_head` / `ref_arc_len` |
| `test_tracker_ref_speed_scale.py` | `LqrTracker.ref_speed_scale`（P1-0B V3 配速）stub 单测 |

## 约定

- 测试放在 `tests/`，命名 `test_*.py`；`tests/__init__.py` 使其成为包，共享夹具在
  `tests/v2_synthetic.py`（非测试模块，不被 pytest 收集）。
- 断言以契约为准（形状、门控、因果链、判定逻辑），不做快照式回归。
- 一次性诊断/临时脚本不放进 `tests/`；可放 `tools/diagnostics/` 或临时目录。
- 退役测试归档进 git 历史（原 `docs/archive/` 已随 docs 收敛删除；取档见 `docs/repo_handbook.md` 附录 D），
  以 `.disabled` 后缀防收集（例：`test_p4_arm_configs.py.disabled`、`test_v7_reward_arms.py.disabled`）。
  当前臂守卫是 `tests/test_arm_configs.py`
  （由 `test_v7_p2_arm_config.py` 重命名重写而来）。
- `__pycache__/`、`.pytest_cache/` 已在 `.gitignore`，不提交。
