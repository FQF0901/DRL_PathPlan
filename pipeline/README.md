# pipeline

训练 / 评估流水线：阶段 A/B/C、阶段 B phase 3 迭代恢复、全自动循环、数据窗口 / 缓冲 / 环境池 / 评测 / 监控。
编排主入口 `pipeline.stages.main`（`pipeline/stages.py:5172-5183`；`--phase3` 优先于 `--stage`），`tools/train.py` 是薄入口（透传参数）。

## 模块职责

| 文件 | 职责 |
| --- | --- |
| `stages.py` | 阶段编排 + CLI（`run_stage_a/b/b_phase3/c`）、`build_model`（`stages.py:152`）、BC 数据集契约校验、未来/历史窗口组装、阶段级损失与评估循环。 |
| `trainer.py` | PPO/GAE 训练器、BC/phase3 预训练循环、损失函数实现、冻结/allowlist/参数组、奖励适配器、环境池（Local/Vector 适配）、BCDataset、checkpoint/RNG。 |
| `buffer.py` | `RolloutBuffer`：按帧存储（不缓存 6 帧堆叠）+ 历史窗口在线重建 + GAE(λ)（按 episode 切断；`valid_mask` 修 padding 行污染）（`buffer.py:1-33`）。 |
| `frames.py` | `FrameLookup`：按 `(episode_id, step)` **精确查表**的历史/未来窗口（stride 网格、缺帧 `hist_valid=0`、SE(2) 对齐到目标帧）（`frames.py:1-42`）。 |
| `vector_env.py` | spawn 常驻 env 池（每进程一个 engine）、按 spec 复用键分组、按 spec 数回收、`MemAvailable` 下限检查、`env.prev_policy_action` 注入（`vector_env.py:1-45`）。 |
| `eval_runner.py` | 独立进程、确定性冻结验证集评测；KPI/分组/Wilson CI/弱类 floor；内存纪律与 worker 回收（`eval_runner.py:1-45`）。 |
| `monitoring.py` | tensorboard + CSV，hook 式；默认 Tier-1 瘦身（非白名单 tag 直接丢弃），`--monitor-legacy-tags` 回退全量（`monitoring.py:1-45`）。 |
| `hard_mining.py` | worst-50% 行权重挖掘（冻结 primary 的加权 IL 误差 → 行权重 sidecar，可追溯 ckpt/数据集指纹）（`hard_mining.py:1-24`）。 |
| `phase3_loop.py` | phase 3 全自动循环：采集 → 训练 → 评测 ×N + 基座/轮间护栏 + keep-best + 状态文件；OOM/显存守卫（`phase3_loop.py:1-45`）。 |
| `run_paths.py` | 运行目录布局唯一来源（`runs/BTC<北京戳>_<name>`、stage 目录、日志/清单/配置快照命名）（`run_paths.py:1-17`）。 |
| `gl_runtime.py` | venv glvnd 库路径保障（spawn worker 的 GL 修复；`LD_LIBRARY_PATH` + 进程内预载）（`gl_runtime.py:1-20`）。 |

## 阶段语义（`config/train.yaml::stages`）

### A — WM 教师强制（latent 世界模型）

- ego 条件 = 专家 GT `(ds,dθ)` 链（累积位姿 + A4 nav 逐步重建，可加 plan 噪声）；目标 = `(episode, step+5k)` 精确查表的未来 OD/LD/ego 帧（stride=5：BC 帧只在策略步边界记录，`_BC_STEP_STRIDE=5`，`stages.py:827`；FrameWindows `stages.py:1760`；`frames.py:344-345` `base+(index+1)*stride`）（t0 对齐、`od_id` 身份匹配、`wm_valid` 门控）（`stages.py:1717-1719,1949-2099`）。
- 损失（代码组合 `stages.py:2311-2317`）：**latent consistency 主监督**（`z_*` vs 未来帧编码目标 `.detach()`，掩码加权 smooth_l1，权重 1.0）+ 物理小解码（od 0.1 / ld 0.02，诊断）+ plan head `ego_next`（0.1）+ presence/entry BCE（各 0.1）（`stages.py:1863-1875`）。
- 可训练 = 除 `policy./value.` 外全部；**MoE 关闭**（输出严格 = primary）（`stages.py:1742-1745,1794-1802`）。policy/value 头不参与。

### B — planner BC（primary → specific）

- 轮数按 `bc.primary_phase_split` 切两段（`stages.py:2982-2985`）：
  - **phase 1（primary）**：MoE 关闭、冻结 `st_gnn/value/experts/router`（`_PRIMARY_PHASE_FREEZE`，`stages.py:2734-2738`）；产出 `primary.pt`。
  - **phase 2（specific）**：MoE 开 + **worst/mild 行权重**（全量曝光）+ Switch 负载均衡 aux；冻结 `st_gnn/value/encoders/mem_encoder/plan_head 主干(primary/fusion/norm/ego_next)/policy` + K-anchor 锚头（`_SPECIFIC_PHASE_FREEZE`，`stages.py:2754-2765`）；可训 = experts/router/residual_scale。
- 损失（`trainer.pretrain_bc`，`trainer.py:4042-4047`）：`traj_weight·轨迹 + action_weight·首步动作 + load_balance + anchor_ce/wta（默认 0）`；轨迹目标 = `traj6`（6 点端点，与 rollout 对齐），动作目标 = 专家即时 `action[:,0]`（`trainer.py:3522-3547`）。
- **router 无聚类监督**（lane U1）：无簇标签/router CE；`--dagger-dir` 可向 specific 段合并 DAgger 行并掩掉其 traj 损失（`stages.py:3396-3443`）。

### B phase 3 — 迭代恢复训练（`--phase3 <dagger_dir>`）

- 数据 = 当轮 DAgger 失败窗口目录单独使用（无 worst/mild、无 mining）；`anchor.enabled=true`（默认关）时并入 5k 锚行（`stages.py:3694-3724,3796-3803`）。
- **冻结配方** `freeze`：`specific_only`（`config/train.yaml:138` 默认；代码兜底 `all`，`stages.py:3752`）= 只训 experts/router/residual_scale，WM/ego_next 与锚 CE/WTA 自动降级为 0；`trunk_only` = 只训共享主干、冻结 WM/value/specific/锚头（锚 CE/WTA 保留塑形）；`all` = 全参数解冻（`stages.py:3746-3780`；降级逻辑 `trainer.py:4587-4613`）。
- 损失项（`trainer._phase3_loss_terms`，`trainer.py:4616-4758`；总装 `4997-5010`）：action + action_chain + bias_calib（默认 0）+ anchor_ce/wta（默认 0）+ ego_next + od + ld + latent + presence + entry + load_balance；`traj` 只监控不进损失（`traj_aux_weight` 默认 0）。
- 动作链目标来源 `action_chain_source=auto`：新 dagger 标签用逐行 `action[0:6]` 真实教师链；旧数据回退同 episode 未来行查表（`stages.py:3715-3717`）。
- 可选 epoch 级 clean150 守护（`--phase3-guard-spec`，fail-closed）；LR 分组 base ×`lr_scale.base` / specific ×`lr_scale.specific`。

### C — PPO RL（实验口径）

- 收集/更新走 cheap path（`rollout=False, world_model=False`），`plan_reference=repeat_action` 时执行参考 = `repeat(a_t, 6)`（P0-1 A-hold）；`plan` 仅旧行为对照（`trainer.py:659-663,6493-6503`）。
- **WM（`st_gnn.*`）全期冻结**；`wm_freeze_updates` 已弃用（仅兼容打印）；`wm_loss_enabled=true` 直接 fail-fast（`stages.py:4629-4647`）。
- `trainable_scope=design`（默认）allowlist = `policy./value./plan_head.moe.experts./plan_head.moe.residual_scale`，其余含共享主干/primary/router/WM 全冻（`trainer.py:910-923`）；`all` = 仅冻 st_gnn 的旧行为。
- 优化：PPO clip 0.2 / γ0.99 / λ0.95（可覆盖）、value/entropy 项、梯度裁剪；KL 锚到阶段 B 快照（`kl_anchor_coef` 线性衰减到 `kl_anchor_final_coef`）；可选 BC 锚；critic warmup（前 N 个 update 只拟合 value 头）；`target_kl` 守门；`adv_norm` global/per_scenario/none（`trainer.py:608-702,6718+`）。
- 周期 ckpt `ckpt_u<NNN>.pt`；drift 探针（`plan[:,1:6]`/`action_mu` RMS 位移，纯监控）；horizon 默认 200 策略步（`stages.py:4440-4466`）。

### 全自动循环

`tools/train.py --phase3-loop`（或 `--phase3-chain`，`tools/train.sh` 的 `PHASE3=1/PHASE3_ONLY=1`）→ `pipeline.phase3_loop`：每轮 采集 → 训练（独立子进程）→ 评测；基座自评（fail-closed）+ 护栏（`guard.mode=base|prev|off`）+ keep-best 导出 `best/final.pt`；状态文件 `phase3_status.json` 原子重写。

## 训练模式与损失构成（代码口径汇总）

| 阶段 | 前向路径 | 损失项（代码名） |
| --- | --- | --- |
| A | `model.encode` + 手写 latent 教师强制（`stages.py:1949`） | `weighted_latent_consistency_loss`（主）、`weighted_od/ld_multi_step_loss`（诊断）、`ego_next` smooth_l1、`presence_entry_loss`（`trainer.py:2031/2092/2160/2537`） |
| B | `model(obs, rollout=True, world_model=False)`（`trainer.py:3943`） | `bc_trajectory_loss` + 首步动作回归 + MoE `load_balance_loss` + `anchor_wta_loss`（可选）（`trainer.py:3952-4047,2301`） |
| phase3 | 同上 + `wm_teacher_forcing_predictions`（`trainer.py:3354`） | `_phase3_loss_terms` 12 项（`trainer.py:4739-4757`） |
| C | collect（默认 `repeat_action` 臂）/update 走 cheap path | PPO clip/value/entropy + KL 锚 + 可选 BC 锚（`trainer.py:6820-6882`） |

## 配置入口（`config/train.yaml`）

| 段 / 键 | 用途 |
| --- | --- |
| `runtime.seed` | 全局种子 |
| `train.device / threads.*` | 设备与线程上限 |
| `train.ppo.lam / minibatch_size / target_kl / group_probe_every / anchor_grad_probe*` | Stage C PPO 与探针（CLI 覆盖） |
| `train.wm.batch_size / micro_batch_size` | Stage A 宏/micro batch 默认 |
| `train.bc.batch_size / micro_batch_size` | Stage B 宏/micro batch 默认 |
| `train.probe_batch` | 固定探针批（动作漂移 + 低速告警；null 关闭） |
| `train.critic_warmup_updates / adv_norm / value_lr_scale / trim_memory_every / ckpt_every` | Stage C 优化与保存 |
| `data.spec / steps_per_epoch / buffer.*` | 训练 spec 与 buffer |
| `run.name / work_dir / bc_dir / resume / stage_a_ckpt / eval_ckpt` | auto 路径解析（`pipeline.run_paths`） |
| `stages.A.*` | `epochs/batch/micro`、`world_model.{loss,ego_condition,targets,slot_matching,noise,ego_next_coef,presence_coef,entry_coef}` |
| `stages.B.bc.*` | `epochs/primary_phase_split/action_weight/traj_aux_weight/load_balance_coef/hard_weight/mild_weight/loss_type/action_dim_weights/anchor_*` |
| `stages.B.phase3.*` | `rounds/spec_pool/fail_target/losses.*/freeze/lr_scale.*/anchor.*/eval.*/guard.*` |
| `stages.C.*` | `tracker/kl_anchor*/plan_reference/trainable_scope/spec_rotation/max_episode_steps/policy_logstd_max/primary_lr_scale/probe_interval/ckpt_every`；可选 `reward.{terms,aggregation}` |
| `monitoring.*` | tensorboard/csv/scene_labels/moe_routing |

入口：`tools/venv-python tools/train.py --stage A|B|C ...`（`--phase3 <dir>` / `--phase3-loop` / `--phase3-chain`）；`--model-config` 默认 `config/model.yaml`；`--config` 默认 `config/default.yaml`（includes 一层平铺合并，`stages.py:138-149`）。注意：`stages.A.world_model.trainable` 当前未被 `run_stage_a` 读取（实际范围见上）。

## 数据流与接口

- **BC 数据集**：`BCDataset.load`（npz + meta；`trainer.py:2875-3109`）；`MaterializedBCDataset` 物化快路径（切片 + pin H2D，与旧路径数值等价，`trainer.py:3113`）；schema<2/缺 v2 通道默认硬失败（`--allow-legacy-dataset` 放行）。
- **窗口**：`FrameWindows`（阶段 A 用）与 `FrameLookup`（buffer/trainer 用）都按 `(episode_id, step)` 查表；`stride` BC=5、rollout=1。
- **缓冲**：`RolloutBuffer.add_step` 逐帧入库 → `build_history` 重拼 6 帧 → `compute_gae(valid_mask=...)`；bootstrap 行不进 minibatch。
- **环境池**：`LocalEnvPool`（单进程精确 tracker，**仅阶段 C**；`build_pool` 只在 `stages.py:4682` 调用，阶段 A/B 为离线 BC 训练、不建 env 池）与 `VectorEnvPool`（spawn 多进程，`--pool vector`）；MetaDrive 每进程只能一个 engine；策略动作经 `env.prev_policy_action` 注入 obs 的 ego reserved 维。
- **奖励**：`build_reward_adapter`（`trainer.py:1595-1662`）构造 `RewardAdapter`（`trainer.py:1333-1352`）；每 env 一份聚合器；rc 档配对守卫（权重≠1 而终局值仍为 rc=1 默认 → fail-fast，`trainer.py:1569-1592`）。
- **评测**：`eval_runner` 冻结 val 集、独立进程、确定性；`--tracker exact|lqr`、`--eval-reference plan|repeat_action`（`eval_runner.py:770-792`）；KPI 口径见 `config/eval.yaml`。

## 测试入口

`tools/venv-python -m pytest tests/ -q`（tests/README.md）。与本目录直接相关的核心用例：

- 阶段 A/B：`test_stage_v11.py`、`test_stage_v2_smoke.py`、`test_bc_pretrain.py`、`test_fast_data_path.py`、`test_stage_b_monitor_holdout.py`、`test_stage_a_ld_loss.py`
- phase3：`test_stage_b_phase3.py`、`test_phase3_recipe_switches.py`、`test_phase3_safe_recipe.py`、`test_phase3_loop.py`、`test_dagger_collect.py`
- 阶段 C：`test_critic_warmup.py`、`test_adv_norm.py`、`test_target_kl_gate.py`、`test_stage_c_trainable_scope.py`、`test_stage_c_drift_probe.py`、`test_stage_c_horizon.py`、`test_stage_c_periodic_ckpt.py`、`test_stage_c_effective_action.py`、`test_stage_c_wm_signal.py`
- 缓冲/窗口/恢复：`test_buffer_gae.py`、`test_frames_lookup.py`、`test_checkpoint_resume.py`、`test_resume_device.py`
- 评测/监控/入口：`test_eval_reference.py`、`test_paired_eval.py`、`test_kpi_grouping.py`、`test_monitoring_v2.py`、`test_run_paths.py`

## 约定与坑

- MetaDrive 每进程只能有一个 engine；任一时刻只跑一个 env-heavy 任务；评测前查 `MemAvailable`。
- `wm_detach` 是 no-op（合成帧 detach 固定语义）；BC/phase3 前向 `rollout=True, world_model=False`，PPO collect/update 走 cheap path。
- `compute_gae(valid_mask=...)` 修复了 padding/bootstrap 行对 episode 尾部优势的污染（`test_adv_norm.py`）。
- 监控默认只落 Tier-1 白名单 tag（OD/EGO 的 loss/KPI + router）；需要旧 tag 用 `--monitor-legacy-tags`。
- phase3 `specific_only` 下上游 WM/ego_next/锚损失自动置 0；不要据此判断“损失未接线”。
- 数据集 schema/标签顺序有硬校验（标签顺序必须等于 `config/model.yaml::moe.router.supervised_labels`）。
