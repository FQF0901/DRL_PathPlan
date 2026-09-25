# config

目的：集中声明环境、模型、训练与评估参数（YAML，键均带中文注释）。

- `default.yaml`：主配置，`includes` 按 env → model → train → eval 顺序平铺合并。
- `env.yaml`：MetaDrive / 步长（策略 0.5 s、MPC/插值 0.1 s）/ 观测 / 场景；`store_map=false`、可选 LRU ≤32。
- `model.yaml`：H=128、GRU 时序、2 层空间消息传递、MoE（primary + 8 experts 128→256→128）、
  WM 6 步直接多步损失；`moe.router.supervised_labels` 8 个标签（固定顺序，与 expert 一一对应）。
- `train.yaml`：阶段 A/B/C、数据与监控。
- `eval.yaml`：冻结 val 集、KPI 与阈值、`dataset_gate`。

## 关键冻结值
- `train.device: auto`（CUDA 可用则 cuda，否则 cpu）；torch 线程 ≤4 且 `OMP/MKL=1`（防 eval worker 线程风暴）。
- PPO `minibatch_size: 1024`（cheap path，GPU）；BC `batch_size: 256`（rollout path，显存峰值 ~8GB，并发时降到 128）。
- `stages.B.bc.traj_aux_weight`：**已验证配方 = 0.3**（闭环优于 0.1，p2-contract §11）；
  当前 yaml 默认仍为 0.1，正式跑用 CLI `--traj-aux-weight 0.3` 覆盖。
- `train.critic_warmup_updates: 0`（默认关，= 旧行为）；阶段 C 推荐 `--critic-warmup-updates 10`。
- `train.probe_batch: runs/bc_expert_full`（null=关闭）：PPO 固定探针诊断（动作漂移 + 低速告警）的输入。

## 评估协议（eval.yaml）
- `recycle_every_specs: 150`：worker 按 spec 数回收（build_env 泄漏 ≈3.5MB/spec）；
  `mem_available_floor_mb: 3000` 运行前程序化检查；`train_pool_policy: pause`（评测期间训练池暂停）。
- 基线 `pure_pursuit_idm` 冻结于 2026-09-25；按 primary 标签（`labels.geometry`）分组，附带标签单列
  `compound`；n≥30 + Wilson CI；弱类 floor = `max(baseline+0.15, 0.75)`。

## 验收标准（两条）
1. **数据侧** `dataset_gate`：物理时间一致性违例 = 0；BC 保留产出率 `bc_retained_step_yield ≥ 0.60`；
   每个受监督标签正样本 ≥ 50。
2. **策略 KPI（相对冻结规则基线）**：`overall_success ≥ max(baseline − 0.05, 0.70)`；collision/offroad 零基线保护 ε；route_completion、speed_ratio（≥0.90×baseline）、a_lat_mean/p95 条款；per-category 弱类 floor。
