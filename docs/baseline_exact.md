# Exact-execution BC/DAgger baseline

> 日期：2026-11-07
> 状态：**current pre-RL baseline**
> 合同：`docs/arch_lock.md`；动作执行 exact tracking，`requested_action == executed_action`。
> 旧 `docs/baseline_r3.md` 是 with safety fallback 的历史 baseline，不能与本文件混用。

## 0. 结论

当前最佳 checkpoint：

```text
runs/bc_dagger_exact_r3_261107031440/best_closed_loop.pt
epoch 15
```

完整 val500（500 scenes，exact execution，max_steps=600）：

| 指标 | 值 |
|---|---:|
| success | **0.664** |
| collision | **0.178** |
| off_road | 0.120 |
| route_completion | 0.851 |
| feasible | 0.812 |
| timeout | 0.048 |
| stuck | 0.006 |
| comfort_violation | 0.996 |

终止分布：

```text
arrive:      332
collision:    89
out_of_road:  55
timeout:      24
```

## 1. 四阶段 exact baseline

| 阶段 | checkpoint | success | collision | off_road | route | feasible | timeout | stuck |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| IL | `runs/bc_il_exact_261106232455` | 0.056 | 0.044 | 0.902 | 0.414 | 0.330 | 0.000 | 0.000 |
| DAgger r1 | `runs/bc_dagger_exact_r1_261107003244` | 0.510 | 0.202 | 0.268 | 0.794 | 0.904 | 0.024 | 0.000 |
| DAgger r2 | `runs/bc_dagger_exact_r2_261107013643` | 0.532 | 0.212 | 0.216 | 0.810 | 0.906 | 0.042 | 0.000 |
| **DAgger r3** | `runs/bc_dagger_exact_r3_261107031440` | **0.664** | **0.178** | **0.120** | **0.851** | 0.812 | 0.048 | 0.006 |

`best_closed_loop.pt` 的 epoch：

```text
IL:         20
DAgger r1:  15
DAgger r2:   5
DAgger r3:  15
```

## 2. DAgger r3 按难度

| difficulty | n | success | collision | off-road | timeout | stuck |
|---|---:|---:|---:|---:|---:|---:|
| easy | 180 | 0.850 | 0.067 | 0.106 | 0.006 | 0.000 |
| medium | 176 | 0.648 | 0.119 | 0.199 | 0.034 | 0.006 |
| hard | 144 | **0.451** | **0.389** | 0.042 | 0.118 | 0.014 |

hard 仍然是主要失败集中区：collision 38.9%。  
easy/medium 已明显改善，整体 success 的提升主要来自 easy/medium 和 hard collision 的下降。

## 3. 数据

| 数据集 | 路径 | 规模 |
|---|---|---|---|
| IL expert | `data/episodes/train_2000/il_expert_exact` | 139,620 rows / 2,000 scenes |
| val expert | `data/episodes/val_500/expert_exact` | 34,368 rows / 500 scenes |
| DAgger r1 | `data/episodes/train_2000/il_dagger_exact_r1` | 135,897 rows / 2,000 episodes |
| DAgger r2 | `data/episodes/train_2000/il_dagger_exact_r2` | 96,034 rows / 1,268 episodes |
| DAgger r3 | `data/episodes/train_2000/il_dagger_exact_r3` | 96,706 rows / 1,268 episodes |

DAgger 采集：

```text
r1: full train_2000, expert_ratio=0.50, shield_ttc=2.0, stuck_shield_steps=20
r2: medium+hard,   expert_ratio=0.60, shield_ttc=2.5, stuck_shield_steps=20
r3: medium+hard,   expert_ratio=0.65, shield_ttc=3.5, stuck_shield_steps=20
```

r3 采集内 student outcome（含 expert replay，不是纯 student）：

```text
success 0.563 / collision 0.226 / off-road 0.094 / max_step 0.117
```

## 4. 训练 recipe

```text
IL:
  data: il_expert_exact
  init: none
  epochs: 20
  lr: 1e-3
  rows: 139,620

r1:
  data: IL 20k + r1 cap 100k
  init: IL best_closed_loop
  epochs: 20
  lr: 1e-3
  rows: 119,985

r2:
  data: IL 20k + r1 20k + r2 cap 100k
  init: r1 best_closed_loop
  epochs: 20
  lr: 5e-4
  near_weight: 1.5
  w_shield: 4.0
  rows: 136,016

r3:
  data: IL 20k + r2 25k + r3 cap 100k
  init: r2 best_closed_loop
  epochs: 15
  lr: 5e-4
  near_weight: 1.5
  w_shield: 4.0
  rows: 141,688
```

通用设置：

```text
sequence_batch: true
scenes_per_batch: 256
clip_length: 100
tbptt_chunk: 8
closed_loop_every: 5
closed_loop_limit: 100
closed_loop_final_limit: 500
closed_loop_patience: 3
```

## 5. 产物

```text
runs/bc_il_exact_261106232455/closed_loop/best_final.json
runs/bc_dagger_exact_r1_261107003244/closed_loop/best_final.json
runs/bc_dagger_exact_r2_261107013643/closed_loop/best_final.json
runs/bc_dagger_exact_r3_261107031440/closed_loop/best_final.json
```

训练/采集日志：

```text
runs/logs/exact_pipeline.log
runs/logs/exact_r3.log
```

## 6. 备注

- 本 baseline 的 eval/selection 都是 exact execution，无 v1 projection、safe-brake、postprocess 或 fallback mask。
- `feasible` 仍是诊断字段；有限但 PSTL-infeasible 的动作被忠实执行，不会终止或改写。
- `test500` 未参与本 baseline 的任何调参或选择。
- 旧 `runs/bc_dagger_raw_r3_261101235342` / `docs/baseline_r3.md` 是 with safety fallback 的历史结果，不能与 exact 数字直接比较。
