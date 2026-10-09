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

    # 全量（2026-10-09 核对：72 个 test_*.py，collect 666 项）
    tools/venv-python -m pytest tests/ -q

    # 可选并行（pytest-xdist 已安装）
    tools/venv-python -m pytest tests/ -q -n 2

仓库无 pytest.ini / pyproject 测试配置，也没有 CI 配置（无 `.github/` 等）；测试全部为本地手动运行。

## 环境依赖与自动 skip

- 只有 `tests/test_lane_lateral.py` 在用例内 `pytest.importorskip("metadrive")` 并惰性 import
  `env.metadrive_env`（仍不建 env）；缺 metadrive 包时该文件自动 skip。真实几何 / 环境的验证
  由 `tools/measure`、`tools/diagnostics` 等脚本与冒烟流程承担，不在 `tests/`。
- 无 CUDA 时 `test_resume_device.py` 的 cuda 用例 skip。
- 无 `setsid` 时 `test_phase3_loop.py` 的后台化用例 skip。
- `test_reward_audit.py` 的排除集用例在 `env/specs/*.json`（gitignored 产物）缺失时 skip。
- `test_stage_v11.py` 的 GL 守卫用例在 venv 本地 glvnd 目录不存在时 skip。

## 测试清单（按主题）

### env / obs 契约

| 文件 | 覆盖 |
| --- | --- |
| `test_obs_schema_v2.py` | OD 固定槽位分配/驱逐/presence、6 帧 mem 顺序与 `hist_valid`、schema 键与 dtype |
| `test_obs_v5_lane_ttc.py` | obs v6：LD offset `{5,10,15,20,30}`、lane/ttc 通道删除、schema/manifest |
| `test_p1a_static_obs.py` | 静态障碍几何（走廊/净距/相对车道/身后）、`others` static 段落位、旧数据回退 |
| `test_v6_world_nav.py` | obs 世界系键、nav 逐步重建、teacher forcing 同步、buffer 透传 |
| `test_ego_prev_action.py` | ego reserved 维（上一策略动作）注入；未注入/非法值保持 0 |
| `test_lane_lateral.py` | `lane_lateral_info` 读取顺序与降级（stub；缺 metadrive 时 skip） |
| `test_max_step_wiring.py` | max_step 截断 → `info["max_step"]` → 终局值结算接线 |
| `test_v5_pipeline_wiring.py` | obs v6 通道表在 BCDataset / RolloutBuffer 的接线 |

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

## 约定

- 测试放在 `tests/`，命名 `test_*.py`；`tests/__init__.py` 使其成为包，共享夹具在
  `tests/v2_synthetic.py`（非测试模块，不被 pytest 收集）。
- 断言以契约为准（形状、门控、因果链、判定逻辑），不做快照式回归。
- 一次性诊断/临时脚本不放进 `tests/`；可放 `tools/diagnostics/` 或临时目录。
- 退役测试归档到 `docs/archive/`，用 `.disabled` 后缀防收集。例：
  `docs/archive/config_arms_legacy/test_p4_arm_configs.py.disabled`、
  `test_v7_reward_arms.py.disabled`。当前臂守卫是 `tests/test_arm_configs.py`
  （由 `test_v7_p2_arm_config.py` 重命名重写而来）。
- `__pycache__/`、`.pytest_cache/` 已在 `.gitignore`，不提交。
