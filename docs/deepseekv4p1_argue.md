# DeepSeek V4.1 RL：当前状态与 r5 改动方案

> 日期：2026-11-17
> 状态：r4b 已结束；test500 已消费；r5 **尚未启动**。
> 合同：`docs/arch_lock.md`。本文件只记录当前事实、根因和即将实施的改动，不代替合同。

## 0. 结论先行

- r4b 的 val500 最好 **0.746 @ update 40**，final 0.730；one-shot test500 **0.708（354/500）< 0.728**，gate 未通过。
- test500 已消费，不再用于调参、选 checkpoint 或二次 gate。r5 只做 val500 selection/research。
- 探索不足不是单一问题：BC 的 `logstd` 接近 clamp、动作空间 `(dx, dy, dpsi, v_next)` 有 time-law 耦合、i.i.d. floor 一大就 invalid、scene 覆盖只有 18.1%。
- GMM 离线诊断已做：动作分布基本单峰；额外分量是零动作/停车行，不是 maneuver。**不做 GMM，不换 head，不上 flow matching。**
- r5 只做三件事：
  1. **reward 课程**：先压掉 comfort，保留 progress/terminal；
  2. **静态 scene 训练集**：一次性评估 train_2000，筛高价值 scene，固定 manifest；
  3. **structured action（临时）**：保持 `(v_next, dy, dpsi)`，反推 `dx`，保证 time-law 一致；后期再改 `(x, y, yaw)`。

## 1. r4b 事实

| 项目 | 结果 |
|---|---:|
| val500 best | **0.746 @ update 40** |
| val500 final | 0.730 |
| val500 10 次均值 | 0.724 |
| test500 strict | **0.708（354/500）** |
| test500 collision | 0.130 |
| test500 off_road | 0.070 |
| test500 timeout | 0.094 |
| test500 route | 0.862 |

失败结构：

- hard：collision 34.2%，timeout 19.9%，success 41.1%；
- medium：off_road 15.4%，success 71.6%；
- straight collision 16.8%，curve 5.8%。

test500 与 val500 的 0.746 差 3.8pp，说明 val500 的 best-of-10 + BC/DAgger selection 偏差合计高估约 3–4pp。

## 2. 根因

### 2.1 探索分四层，不能只看 entropy

1. **动作分布熵**：BC `logstd` 在固定 hard scene 上已有 49% 决策卡在 clamp -5；u50 是 81%。`clamp(-5,0)` 在 -5 以下梯度为 0，entropy bonus 拉不动。
2. **动作可行性**：动作是 `(dx, dy, dpsi, v_next)`，执行层用 0.5s 三次 Hermite 路径 + 时间律。`dx` 和 `v_next` 独立加噪声会破坏 `dx ≈ v·dt`，直接 `invalid_action`。
3. **状态/场景覆盖**：r4b 50 updates × 8 scenes = 400 draws，只覆盖 362/2000 = 18.1% 的 train scene。
4. **信用分配**：group 内全 success 或全 fail 时 terminal advantage 为 0；只有 mixed outcome 或 failure-type 差异才有信号。

### 2.2 探索 floor 的实测上限

在同一 u1 batch（8 scenes × 8 members）上扫 `std_min`：

| floor | success | off_road | collision | invalid | decisions |
|---|---:|---:|---:|---:|---:|
| r4b `[.005,0,0,.005]` | 0.297 | 0.422 | 0.219 | 0.047 | 58.4 |
| `[.01,.001,.001,.01]` | 0.297 | 0.438 | 0.141 | 0.063 | 59.2 |
| `[.02,.002,.002,.02]` | 0.188 | 0.375 | 0.250 | 0.172 | 49.1 |
| `[.03,.003,.003,.03]` | 0.141 | 0.422 | 0.219 | 0.203 | 47.3 |
| `[.05,.005,.005,.05]` | 0.000 | 0.422 | 0.234 | 0.344 | 27.0 |

结论：0.01 是第一个安全点；0.02 以上开始 invalid；0.05 直接退化。**继续加 i.i.d. floor 不是可行的探索方向。**

### 2.3 GMM 排模结果

冻结 BC encoder，对 95,617 条有效 DAgger 行拟合条件 GMM（K=1..8），按 scene seed 80/20：

| K | held-out NLL | 有效分量 | active (>2%) | max weight >0.9 | MI(nav turn) | episode purity | switch rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -13.275 | 1.00 | 1 | 1.00 | 0 | 1.000 | 0 |
| 2 | **-13.482** | 1.04 | 1 | 1.00 | 0 | 1.000 | 0 |
| 4 | -13.475 | 1.21 | 2 | 0.90 | 0 | 0.981 | 0.008 |
| 8 | **-13.492** | 1.29 | 2 | 0.87 | 0.002 | 0.994 | 0.002 |

关键细节：

- K=2 的 usage 是 **99.4% / 0.6%**，第二分量 action mean 是 `[0,0,0,0]`，即零动作/停车行；
- K=8 有 5 个分量是零动作，主分量占 94.5%；
- 与 nav turn 的互信息接近 0；episode purity ~1；switch rate ~0；
- 排除零动作行后，K=1 最好，K>1 没有稳定改善。

**结论：当前数据/特征下动作分布基本单峰，GMM 分量不是 maneuver。不要为多模态重训 IL/DAgger。**

## 3. r5 改动方案

### 3.1 Reward 课程

保持 `pipeline/train/rl_reward.py` 的公式不变，只加权重 schedule。

| 项 | 阶段 A（1–15） | 阶段 B（16–30） | 阶段 C（31–50） |
|---|---:|---:|---:|
| progress | 10 | 10 | 10 |
| comfort | **0** | 0 → 0.05 | 0.05 |
| low_speed | 0.1 | 0.1 → 0.2 | 0.2 |
| wrong_route | 2 | 2 | 2 |
| stuck | 2 | 2 | 2 |
| success | 20 | 20 | 20 |
| collision | 20 | 20 | 20 |
| off_road | 15 | 15 | 15 |
| timeout | 10 | 10 | 10 |
| invalid | 20 | 20 | 20 |

原则：

- **progress 必须保留**；它是 credit assignment，不是 comfort 类探索抑制项。
- 阶段 A 只去掉 comfort，low_speed 保留很小；先学 collision/off-road/timeout/success。
- 阶段 B/C 再把 comfort 加回来，避免从头到尾被平顺性压制。
- 不做纯 episode-level reward；否则全 fail group 没有 dense 信号，success 太稀疏。

新增纯模块：

```text
pipeline/train/reward_schedule.py
```

接口：

```python
resolve_reward_config(base: RewardConfig, phases, update: int) -> RewardConfig
```

- 只覆盖配置里列出的权重；其余沿用 base；
- 数值字段在阶段端点间线性插值；
- 不依赖 env/model/trainer；
- 无 `reward_schedule` 时行为与当前完全一致。

配置块（`config/train_rl.yaml`，可选）：

```yaml
reward_schedule:
  interpolate: true
  phases:
    - until_update: 15
      w_comfort: 0.0
      w_low_speed: 0.1
    - until_update: 30
      w_comfort: 0.05
      w_low_speed: 0.2
    - until_update: 50
      w_comfort: 0.05
      w_low_speed: 0.2
```

Trainer 记录每个 update 解析后的权重到 `train_log.jsonl`，checkpoint 记录 schedule 和 current weights。

### 3.2 静态 scene 训练集

不做在线 value sampler / 热刷新；一次性评估 train_2000，生成固定 manifest。

流程：

1. r3 exact checkpoint 在 train_2000 上跑 deterministic 1 episode/scene（约 1–3h，4 workers）。
2. 对候选 scene 跑 8 members（failure + 10–20% success）：
   - 约 500–800 scenes × 8 = 4000–6400 episodes；
   - 记录 success_count、terminal histogram、return std、route、expert success。
3. 选择：
   - frontier：`0 < success_count < 8`，优先；
   - deterministic failure：`success_count == 0`，按 collision/off_road/timeout 分层；
   - success anchor：`success_count == 8`，取 10–20%；
   - 排除 expert 也失败的 scene；
   - 目标 **400–600 scenes**。
4. 输出 manifest（固定 schema）：

```json
{
  "schema": "drl_pathplan_rl_scene_manifest_v1",
  "base_ckpt": "runs/bc_dagger_exact_r3_.../best_closed_loop.pt",
  "specs": "data/specs/train_2000.json",
  "members": 8,
  "selected": [1001, 1007],
  "stats": [],
  "sha256": "..."
}
```

5. Trainer 读取 manifest，过滤 `train_2000`，继续用现有 difficulty 采样在子集上抽 8 scenes/update。
   - 没有 manifest：走完整 train_2000；
   - 有 manifest：assert 与 val500/test500 无 scene seed 交集；日志记录 manifest SHA、selected count、实际采样 seeds。

不新增 `scene_sampler.py`，不做在线 value 表；manifest 位置待定（`runs/` 或 `data/manifests/`）。

### 3.3 Structured action（临时）

保留 4D head，但训练期行为分布改为 3 维：

```text
policy samples: (v_next, dy, dpsi)
derive:         dx from time law
execute:        (dx, dy, dpsi, v_next)
```

- `dx` 由 0.5s 时间律推导；目标是让路径长度/速度曲线一致，避免 `dx` 与 `v_next` 独立噪声导致的 invalid。
- PPO 的 log-prob 必须是 **3 维** `(v_next, dy, dpsi)`；`dx` 是确定推导，不能采样后再改 4D 动作却沿用 4D log-prob。
- head 的第 4 维 `dx` 输出暂时忽略或做辅助 loss；初期可从 r3 checkpoint 直接初始化 3 维分量，不需要完整重训 IL/DAgger。
- 横向/曲率仍由 `action_to_trajectory` post-hoc 检查；推导出的 `dx` 越界时拒绝/重采样，不能 projection。
- 后期把动作正式改为 `(x, y, yaw)` 参数化；这需要 BC/DAgger 重训和合同更新。

低风险替代：先做 **A（只把 `v_next` 限制在 `feasible_v1_interval`）**，它精确且不改 head；如果 B 的 3D log-prob/几何推导成本太高，可先用 A 做 10-update pilot。

## 4. 自动化与最小落地范围

不做大而全框架；只加必要模块和脚本：

```text
pipeline/train/reward_schedule.py       # 纯 reward 阶段解析
tools/build_r5_scene_manifest.py        # 一次性 scene 评估 + manifest
tools/run_r5.sh                         # setsid nohup 启动
tools/monitor_r5.sh                     # 20 分钟资源/进度监控
```

Trainer 只增加：

- `reward_schedule` 配置解析；
- `scene_manifest` 读取/过滤；
- 日志/checkpoint 记录 schedule、resolved weights、manifest SHA；
- structured action 的 3D 分布与 log-prob 一致性。

测试：

- reward schedule 阶段边界/插值/非法值；
- scene manifest schema、seed 交集、缺失 fallback；
- structured action 的 time-law、dx 边界、3D log-prob audit；
- 无新配置时旧行为不变。

## 5. 约束

- conda/python/官方库只读，不改 site-packages；
- 充分利用 CPU/MEM/SWAP/GPU；4 workers + GPU；
- task 与 session 解耦：`setsid nohup` + PID/日志；
- 每 20 分钟监控一次进程、RSS、swap、GPU、update/holdout；
- 早停：val500 < 0.62；u10 val500 < 0.670；invalid >0.10 或 decisions <45 连续 2 个 update；
- test500 已消费，r5 不再使用；没有新 clean split 前不宣称 promotion。

## 6. 待拍板

1. scene manifest 放 `runs/`（run artifact）还是 `data/manifests/`（正式训练集）？
2. structured action 直接做 B（3D policy + 反推 dx），还是先做 A（只约束 `v_next`）？
3. 目标 scene 数取 400–600 中的哪个值（建议 500）？

## 7. P1 reward curriculum 结果（2026-11-18）

Run：`runs/rl_group_r5_p1_261118000410`，15 updates，只把 `w_comfort` 从 0.05 设为 0；其余权重、`std_min=[0.005,0,0,0.005]`、scene 采样、PPO 超参不变。

| 指标 | P1 u5 | r4b u5 | P1 u10 | r4b u10 | P1 u15 | r4b u15 |
|---|---:|---:|---:|---:|---:|---:|
| strict success | 0.660 | 0.688 | 0.676 | 0.698 | **0.692** | **0.722** |
| collision | 0.172 | 0.132 | 0.162 | 0.134 | 0.160 | 0.118 |
| off_road | 0.134 | 0.114 | 0.116 | 0.098 | 0.100 | 0.088 |
| timeout | 0.044 | 0.070 | 0.046 | 0.072 | 0.048 | 0.072 |
| route | 0.852 | 0.868 | 0.854 | 0.873 | 0.858 | 0.880 |

结论：**comfort=0 不是有效探索。** success 三个检查点全部低于 r4b 同期（−2.8 / −2.2 / −3.0pp），collision/off_road 都更高；timeout 更低只是把失败换成了碰撞。训练 rollout 的 raw `max_abs_a_long_mean` 约 4.2–4.9（comfort 阈值 2.0），说明动作更激进，但没有转化为更高的 success。

决定：**停止在 P1 reward 上叠加 P2/P3。** comfort 保留；P2/P3 若要继续，必须回到 r3 + 默认 reward 单独验证，不能把 P1 当 baseline。
