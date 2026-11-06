# DAgger r3 baseline

> 日期：2026-11-03
> 状态：**historical baseline — with safety fallback，不是 exact baseline**
> 说明：该 run 的 eval/selection 使用了已移除的 v1 projection / safe-brake；
> `fallback=0.094` 表示 9.4% episode 发生过动作改写。exact 重跑后由新 baseline 取代。
> 架构合同见 `docs/arch_lock.md`。

## Checkpoint

```text
runs/bc_dagger_raw_r3_261101235342/best_closed_loop.pt
epoch 10
```

模型：`DriveModel` 1,270,748 parameters；action 为 normalized `[0,1]^4` TruncatedNormal；raw history，6 帧 / 16 OD + 16 LD slot。

训练入口：

```text
runs/logs/train_bc_dagger_raw_r3.log
```

## val500 gate

```text
specs: data/specs/val_500.json
split_manifest: data/splits/val_500.json
controller: exact
max_steps: 600
n: 500
```

结果：

| 指标 | 值 |
|---|---:|
| success | 0.610 |
| collision | 0.180 |
| off_road | 0.112 |
| route_completion | 0.832 |
| feasible | 0.856 |
| fallback | 0.094 |
| timeout | 0.106 |
| stuck | 0.086 |
| comfort_violation | 0.970 |

按难度：

| difficulty | n | success | collision | off-road | timeout | stuck |
|---|---:|---:|---:|---:|---:|---:|
| easy | 180 | 0.839 | 0.056 | 0.111 | 0.017 | 0.006 |
| medium | 176 | 0.568 | 0.165 | 0.199 | 0.068 | 0.040 |
| hard | 144 | 0.375 | 0.354 | 0.007 | 0.264 | 0.243 |

hard termination：

```text
arrive:      54
collision:   51
timeout:     38
out_of_road:  1
```

按 geometry：

| geometry | n | success | collision | off-road | timeout | stuck |
|---|---:|---:|---:|---:|---:|---:|
| straight | 331 | 0.607 | 0.196 | 0.094 | 0.115 | 0.094 |
| curve | 169 | 0.615 | 0.148 | 0.148 | 0.089 | 0.071 |

## 相对 DAgger r1

`runs/bc_dagger_raw_r1_261101173703/eval_val500/metrics.json`：

```text
r1: success 0.512 / collision 0.176 / off-road 0.192
    route 0.785 / feasible 0.480 / fallback 0.160
    timeout 0.122 / stuck 0.096
```

差值：

```text
success   +9.8 pt
collision +0.4 pt
off-road  -8.0 pt
timeout   -1.6 pt
stuck     -1.0 pt
```

## 训练 recipe

初始化：

```text
runs/bc_dagger_raw_r2_safety_261101225359/best_closed_loop.pt
```

数据混合：

```text
IL expert:       20,000 rows
previous DAgger: 25,000 rows
latest DAgger:   all rows
```

训练：

```text
sequence_batch: true
scenes_per_batch: 256
clip_length: 100
tbptt_chunk: 8
epochs: 15
lr: 5e-4
near_weight: 1.5
w_shield: 4.0
```

run config：

```text
train rows: 141,616
train scenes: 1,389
val rows: 34,386
val scenes: 500
```

## DAgger r3 采集

```text
data: data/episodes/train_2000/il_dagger_raw_r3
student: runs/bc_dagger_raw_r2_safety_261101225359/best_closed_loop.pt
focus: medium + hard
scenes: 1,268
expert_ratio: 0.65
shield_ttc: 3.5
stuck_shield_steps: 20
round: 3
rows: 96,627
episodes: 1,268
failed: 0
expert replay: 63,639
lift: 32,988
valid: 95,728
```

采集内 student outcomes：

```text
success:   0.563
collision: 0.226
off-road:  0.094
max_step:  0.117
stuck:     0.015
```

按难度：

| difficulty | n | success | collision | off-road | max-step | stuck |
|---|---:|---:|---:|---:|---:|---:|
| medium | 702 | 0.657 | 0.111 | 0.167 | 0.066 | 0.003 |
| hard | 566 | 0.447 | 0.369 | 0.004 | 0.180 | 0.030 |

## 备注

- 本文件是 run-specific baseline，不是架构不变量；下一次 DAgger/RL run 应新建对应 baseline 文档或更新本文件。
- comfort_violation 在当前 episode 级定义下接近饱和（expert 自身每段也只有约 0.69 comfortable），不作为 gate。
- `test500` 未参与本 baseline 的任何调参。
