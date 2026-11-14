# exact execution / stable-ID review — current position

> 日期：2026-11-07
> 对照：`docs/chatgpt5p6sol_review.md`、`docs/arch_lock.md`、`docs/baseline_exact.md`、当前代码
> 结论：**代码主线接受；审查中的 P1/P2 已修；`arch_lock` 可以锁为 pre-RL contract，剩余问题是 P3 级或历史文档问题。**

## 0. 总判断

这轮 review 的质量比上一版高：它没有再纠缠 exact execution / stable-ID 的方向，而是抓
invalid-action 语义、target-only ID、PPO 终止回填、registry 和文档范围这些收口问题。

我的判断：

| 类别 | 结论 |
|---|---|
| exact execution | 接受，代码已按 `requested_action == executed_action` 收口 |
| stable ID | 接受，生成/loader/BankWindow/fast path 已 fail-fast |
| P1-1 `h_valid` | 接受，doc 已改为 `current_valid ∪ target_mask` |
| P1-2 invalid `feasible` | 接受，code 已统一 invalid 分支计数 |
| P2-1 target-only ID | 接受，`train_wm` 已对整个 target frame 校验 |
| P2-2/P2-3 PPO invalid | 接受，已回填 terminal 并记录 `invalid_action_count` |
| P2-4 registry | 接受，`data/manifests/datasets.json` 已更新 |
| P2-5 “所有使用端” | **部分接受**：doc 已改成 canonical 入口/窗口构建 fail-fast，不宣称 encoder 每一行都校验 |
| P2-6 DAgger replay | 接受，doc 已限定为 student/lift 行 exact |
| P2-7 LD chunk | 接受，doc 已改成 chunk anchor 是 feature，不是 ID |
| P3-1..P3-5, P3-7, P3-8 | 接受，已做文档/小代码清理 |
| P3-6 sentinel | **部分接受**：numpy validator 已去掉 sentinel；torch fast path 仍保留极低概率的 sentinel 边界，暂不作为 blocker |

## 1. 我接受并已经修的

### 1.1 P1-1 `h_valid` 语义

`arch_lock` 原来写：

```text
h_valid  # current valid mask
```

实际 WM 训练是：

```text
h_valid       = current_valid ∪ target_mask
current_valid = current_valid
```

已在 §7.1 修正。这个不是可选解释：target-only 新实体必须进入 WM 预测/监督，否则
content/geometry 会被 mask 掉。

### 1.2 P1-2 invalid action 的 `feasible` 语义

`pipeline/eval/closed_loop.py` 现在对三个 invalid 分支统一递增 `infeasible_count`：

- `requested_tr is None`
- trajectory speeds 非有限/负值
- `ExactTracker.arm` 抛 `TypeError/ValueError`

因此 `termination=invalid_action` 不再出现 `feasible=True`。同时记录
`execution_unavailable=True`，明确这一行没有真正执行。

### 1.3 P2-1 target-only stable ID

`pipeline/train/wm.py::_align_by_id` / `_align_geometry_by_id` 现在对整个 target
frame 调用 `validate_stable_ids`，再做 `_target_hit`。target-only zero/duplicate
不再需要靠 `load_data` 兜底。

### 1.4 P2-2 / P2-3 PPO invalid

`pipeline/train_ppo.py`（历史记录；该文件已删除）现在：

- invalid action 发生时，如果已有 transition，最后一条标记 `terminated=True`；
- 记录 `invalid_action_count`；
- summary 的 `infeasible_count` 包含 invalid action episode。

### 1.5 P2-4 registry

`data/manifests/datasets.json` 已改为 exact 路径和当前 rows/success：

```text
il_expert_exact
val_500/expert_exact
il_dagger_exact_r1/r2/r3
```

mini episode pools 标记为 optional/not collected。

### 1.6 P2-6 / P2-7 / P3-1 / P3-8

- DAgger replay 行明确是 expert action，不属于 student exact execution；
- LD chunk anchor 明确是 feature，不是 stable ID 的一部分；
- 删除重复的 geometry 句子；
- WM 训练 action 明确为“有 executed_action 用它，否则用实测 action”。

### 1.7 P3 清理

- `baseline_r3.md` 标注为 with-fallback 历史文件，旧 checkpoint/data/log/metrics 已删除；
- `baseline_exact.md` 说明第一次 r3 采集 `96,632/1267 failed=1`，当前 baseline 使用
  修复后的 `96,706/1268 failed=0` 重采；
- `data/README.md` 的 mini 路径改成 optional；
- `pipeline/train/wm.py` 去掉未使用的 `usable`；
- eval invalid row 增加 `execution_unavailable`。

## 2. 我部分反驳或限定解释的

### 2.1 P2-5 “所有使用端 fail-fast”过强

这条批评是对的：`net/encoder.py` 的 `_ids` 只做 shape/key 检查，`net/model.py` 的
record path 在 shape 不匹配时会 `continue`。说“所有使用端”不准确。

但我不打算把 `validate_stable_ids` 塞进每一个辅助函数，因为那会造成三套重复校验。
合同改成：

```text
canonical 生成端、loader、BankWindow 构建、模型 fast path fail-fast
辅助/诊断函数可以使用显式 test-only fixture
```

当前生产数据的真正入口是 `build_observation` / `load_data` / `BankWindowBuilder` /
`_sanitize_ids` / `train_wm`，这些已经覆盖。doc 已按这个边界收紧。

### 2.2 P3-6 sentinel 碰撞

numpy 侧已经改成：

```python
order = np.lexsort((ids_mat, ~valid_mat))
```

不再依赖 `np.iinfo(np.int64).min` sentinel。

torch 侧 `_sanitize_ids` 仍用 `-(2**62)` 作为 invalid sentinel。理论上 valid ID
恰好等于该值、且 invalid slot 夹在两个相同 valid ID 之间时，重复检测可能漏掉。
概率是 `1/2^63` 量级，且 stable hash 是 64-bit。我的判断：

- 接受它是真实边界；
- 不把它作为 pre-RL blocker；
- 如果未来要彻底消除，改成 per-row valid-ID unique 检查，代价是 fast path 变慢。

### 2.3 P3-5 invalid row 的 `executed_action`

invalid row 现在保留 `executed_action = requested_action` 作为审计值，同时增加
`execution_unavailable=True`。我不改成 `None`，因为：

- 现有 summary/consumer 期望 `[4]` 数组；
- `requested_action` 本身仍是有价值的诊断；
- `execution_unavailable` 已明确表达“没有执行”。

这比把字段改成 `None` 更兼容，也更符合“requested 原样、executed 不可用”的事实。

### 2.4 P3-3 / P3-4 历史文档

`baseline_r3.md` 的旧 path 不可复现，这是事实；但它的价值是保留“with safety fallback”
的历史指标，不是提供可复现 artifact。已在头部明确“旧 checkpoint/data/logs/metrics
已清理，不能复现”。

`baseline_exact.md` 的 r3 第一次采集失败 1 个 episode，是实际发生过的；当前 baseline
使用修复后的重采结果，已在日志说明。

## 3. 结论

- **没有 P0。**
- P1-1、P1-2、P2-1、P2-2、P2-3、P2-4 已修。
- P2-5、P2-6、P2-7 是文档边界问题，已收紧。
- P3-6 是极低概率的 torch fast-path 边界，已记录，不作为 blocker。
- 当前完整测试：**141 passed, 1 skipped**（`LD_LIBRARY_PATH=.venv/gl/usr/lib/x86_64-linux-gnu`）。
- 当前 exact baseline：r3 `success 0.664 / collision 0.178 / route 0.851`，见
  `docs/baseline_exact.md`。

因此：**代码主线 ACCEPT；`docs/arch_lock.md` 可以锁为 pre-RL contract。**
后续真正的工作不是继续补 review 文档，而是 WM gate / RL。

---

# 2026-11-11 — RL 现状、问题与猜想（deepseekv4p1）

> 更新时间：2026-11-11 11:30（训练完成；owner 决定停止 RL 线）
> Run：`runs/ppo_group_r3_261111015244`
> 合同：`docs/arch_lock.md` §12.4（本文件不是合同，只记录现状、问题、猜想和待拍板问题）
> 结论先行：**实现链路已跑通；holdout best 0.688 / final 0.680，但 0.688 是 best-of-10 的乐观值，且 holdout 的 medium/hard 全部被 r3 IL 训练见过（P0 exposure confound），不能外推到 test500 的 unseen medium/hard。owner 决定停止 RL 线；test500 未消费。**
> 2026-11-14 清理：删除 calibration 200、legacy PPO/joint/selector、`dataset/rl_splits.py`；`pipeline/` 重组为 `common/eval/losses/rl/train/wm` 子包。本文中的旧路径/calibration 内容如与代码冲突，以 `docs/arch_lock.md` 和当前代码为准。

---

## 1. 目标和硬约束

- exact DAgger r3 → real-env group-relative PPO → train_2000 holdout 选择 → **test500 one-shot final gate（strict success ≥ 72.8%）**。
- RL split：train seeds 1000..2499（1500）；holdout seeds 2500..2999（500）；val500 只做 diagnostic；test500 只跑一次。
- RL 只用 real env；WM 不参与 rollout/reward/advantage。
- 所有奖励/惩罚进一个规则 reward model；没有 cost vector/Lagrangian。
- 单流 return；per-step active-set leave-one-out advantage；不用 episode-level advantage 广播。
- planner + TSB 可训练；encoder/aux 冻结；exact execution，requested == executed。
- 每个实现阶段都做了 adversarial review；剩余 P0 已清零后才启动正式训练。

---

## 2. 已实现并验证的部分

| 文件 | 内容 | 状态 |
|---|---|---|
| `pipeline/rl/reward.py` | 规则 reward：dense + terminal、事件优先级、wrong_route/stuck one-shot、5-substep 事件聚合 | 已实现，单测通过 |
| `dataset/splits.py` | RL split manifest 加载、role/sha/无重叠校验（与通用 split 工具合并） | 已实现 |
| `data/splits/train_2000_rl_train_1500.json` | seeds 1000..2499 | 已实现 |
| `data/splits/train_2000_rl_holdout_500.json` | seeds 2500..2999 | 已实现 |
| `pipeline/rl/train_group.py` | spawn 多进程 rollout、exact 0.5s segment、active-set LOO、PPO、holdout selection | 已实现 |
| `config/train_rl_group.yaml` | 50 updates / 8 scenes × 8 group / 4 workers / eval 500 every 5 | 已实现 |
| `pipeline/eval/closed_loop.py` / `pipeline/eval/metrics.py` | 新增 `strict_success`；eval 显式 `reset(seed=spec.seed)` | 已实现 |

验证证据：

- 全量测试最后一次结果：`185 passed, 2 skipped`；trainer smoke（`RUN_PPO_GROUP_SMOKE=1`）`11 passed`。
- PPO old-logprob audit：实测 max abs diff `2e-5 ~ 3.4e-5`，低于 `1e-3` 容差。
- 对抗 review 已覆盖：文档合同、reward/split、trainer。剩余 P0 已修；没有发现 NaN、非法 ratio、invalid 爆炸。
- 基线 holdout（新 eval 口径，seed 显式）：`strict_success = 0.646`。

已知偏差/缺口：

- **reward calibration 已取消**（owner 2026-11-14）：200-scene calibration manifest 和相关 loader/trainer 参数已删除；正式训练用的是 provisional 默认权重。
- `entropy_proxy` 实际记录的是 mean log-std，不是 entropy；指标命名有误导，但不影响训练。
- baseline holdout 最初只保存了汇总；后来用独立进程补跑并保存 `baseline_holdout_rows.json`，paired bootstrap 已做（见 §3.2/§4.8）。
- `runs/` 被 gitignore；训练产物不在本次 commit 里，文档只记录快照。

---

## 3. 正式训练快照（已完成 50/50 updates）

### 3.1 运行配置

```text
run: runs/ppo_group_r3_261111015244
start: 2026-11-11 01:52
updates: 50
scenes_per_update: 8
group_size: 8
workers: 4
max_steps: 600 (env step cap)
max_decisions: 120
gamma: 0.99
clip: 0.2
epochs: 4
minibatch: 256
lr: 3e-5 (planner), 3e-6 (TSB)
eval_every: 5
eval_limit: 500 (full holdout)
device: cuda
```

### 3.2 Holdout 曲线

| update | strict_success | collision | off_road | timeout | route_completion |
|---:|---:|---:|---:|---:|---:|
| 0 (r3 baseline) | 0.646 | 0.186 | 0.100 | — | 0.8423 |
| 5 | 0.672 | 0.174 | 0.090 | 0.066 | 0.8479 |
| 10 | 0.664 | 0.174 | 0.086 | 0.080 | 0.8490 |
| 15 | 0.676 | 0.156 | 0.078 | 0.092 | 0.8526 |
| 20 | **0.688** | 0.136 | 0.090 | 0.086 | 0.8605 |
| 25 | 0.674 | 0.150 | 0.092 | 0.086 | 0.8570 |
| 30 | 0.674 | 0.148 | 0.090 | 0.088 | 0.8572 |
| 35 | 0.676 | 0.148 | 0.092 | 0.084 | 0.8560 |
| 40 | 0.678 | 0.148 | 0.092 | 0.084 | 0.8561 |
| 45 | 0.666 | 0.162 | 0.084 | 0.090 | 0.8506 |
| 50 | 0.680 | 0.150 | 0.084 | 0.086 | 0.8568 |

- 最好 `0.688 @ update 20`；此后 6 次在 `0.666 ~ 0.680`。最后 5 次均值与最前 5 次均值相同（0.6748），没有持续上升趋势。
- `best_holdout.pt` 初始化自 r3 baseline，所以训练不会 promote 一个比 0.646 更差的 checkpoint；最终 best 是 update 20，不是 update 50。
- 500-scene 单次评测的标准误约 `sqrt(0.68*0.32/500) ≈ 2.1pp`；单点差异不能直接当真实提升。

**最终 paired 统计（baseline per-scene rows 已补，`baseline_holdout_rows.json`）：**

| 对比 | base | RL | diff | McNemar exact p | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| baseline vs update 20 | 0.646 | 0.688 | **+4.2pp** | **0.0099** | [+1.2pp, +7.2pp] |
| baseline vs update 50 | 0.646 | 0.680 | +3.4pp | 0.033 | [+0.4pp, +6.4pp] |
| update 20 vs update 50 | 0.688 | 0.680 | -0.8pp | 0.60 | [-3.0pp, +1.4pp] |

- 按 difficulty（update 20）：easy 0.872→0.904、medium 0.607→0.653、hard 0.388→0.439；三个分层都涨，但 hard 仍只有 0.439。
- **P0 caveat：holdout 不是 IL-clean。** r3 `config.train_seeds` 有 1389 个 scene，与 RL holdout 交集 **350/500**：173 medium + 139 hard + 38 easy；IL-unseen 的 150 个 **全部是 easy**。test500 的 medium/hard 是 IL-unseen，所以当前 holdout 的 medium/hard 增益不能外推到 test500。详见 §4.11。
- 修正结论：**在这个 IL-exposed holdout 上，r3 → RL 的提升是统计显著的，但幅度只有 3~4pp；update 20 与 50 无法区分，0.688 是 best-of-10 的乐观值。** 它不能证明 unseen medium/hard 会同步提升，也不能证明接近 72.8%。

### 3.3 训练 rollout（50 个 update 的统计）

| 指标 | 值 |
|---|---:|
| train success mean | 0.536 |
| train success std | 0.155 |
| train success range | 0.188 ~ 0.828 |
| mean decisions / episode | 71.4 |
| mean reward / episode | -1.74 |
| collision mean / max | 0.198 / 0.406 |
| off-road mean / max | 0.155 / 0.375 |
| timeout mean / max | 0.099 / 0.328 |

- 训练 scene 上的成功率波动很大，最高到 0.83；但 holdout 没有同步上升。
- update 50 的 reward 分量（per transition 均值）：`progress +0.111`、`comfort -0.131`、`terminal -0.039`、`low_speed -0.005`、`wrong_route/stuck = 0`。
- update 50 的 terminal 分布：64 episodes 中 success 24、collision 24、off_road 9、timeout 7；训练 batch 的终局波动很大。
- PPO：`approx_kl` 0.009~0.018、`clip_frac` 0.11~0.20，均低于 stop 阈值；没有触发提前停 epoch。
- 日志里的 `entropy_proxy` 约 `-4.83 ~ -4.91`，即访问状态上的 log-std ≈ -4.85，对应 std ≈ **0.008**（normalized action `[0,1]`）。

### 3.4 资源与运行时间

- 进程已正常退出；RSS 峰值约 **9.3GB**，swap 峰值约 **25GB**，GPU 约 1.3GB；未观察到 OOM 或非零退出。
- 训练总运行约 **8h40m**（01:52 ~ 10:32）。
- holdout 评测耗时在明显变长：相邻两次 eval 的间隔 `35.5 → 37.6 → 44.5 → 47.4 → 52.6 → 59.3 → 62.8 → 65.4 → 69.9 分钟`，9 个间隔合计约 **475 分钟**；评测是主要时间开销。
- 背景监控脚本每 20 分钟写一次 `runs/logs/ppo_group_monitor.log`；记录到 8 小时窗口结束。训练进程与 session 解耦。

---

## 4. 观察到的问题

### 4.1 Holdout 平台化，train rollout 没有平台化

**事实**：train success 从 0.19 到 0.81 大幅波动，且 update 36/39 能到 0.73/0.81；holdout 在 update 20 之后一直 0.674~0.678。
**含义**：至少有两种可能：策略只记住了部分 train scene；或 group-relative 信号在 holdout 上不迁移；或 holdout 已经接近 r3 表征的能力上限。
**不能下结论**：现在还不能说“过拟合”，因为训练只覆盖了 286 个 unique scenes，且每 update 只有 8 个 scene。

### 4.2 探索可能已经塌缩（最可疑）

**事实**：PPO 重算时 `action_logstd` 均值约 -4.85，std ≈ 0.008；normalized action 范围是 `[0,1]`。
**推断**：同一 scene 的 8 个 group 成员动作几乎相同。若真是这样，group 内 terminal 结果差异（success/collision/timeout）更多来自系统混沌放大，而不是动作质量差异；LOO advantage 会变成噪声。
**目前没有验证**：还没有记录 group 内动作 L2 距离、状态分叉时间、或“同一动作重放”的 terminal 一致性。这是下一步最便宜、最值得做的诊断。
**潜在修法**：entropy bonus、log-std floor/最小值、增大 group、动作扰动、或先做 deterministic group + 值基线（但 no critic 是合同约束，需要 owner 批准）。

### 4.3 Dense reward 几乎相互抵消，且 comfort 未校准

**事实**：update 39 的 `progress +0.106` 与 `comfort -0.110` 基本抵消；terminal 才是主要信号。
**推断**：comfort 的 jerk/加速度阈值可能过于敏感，持续惩罚必要的刹车/变道；在未校准的情况下，策略可能学到“少动、慢动”而不是“更快到达”。
**合同偏差**：config 里 `w_comfort=0.05` 是 provisional；calibration split 已删除，不能声称权重已经合理。

### 4.4 Timeout 可能过便宜

**事实**：timeout reward 是 `-w_timeout * (1 - rc_T)`，`w_timeout=10`。训练 rollout 中出现过 31%~33% 的 timeout（update 33/35），holdout timeout 约 8.4%。`low_speed` 的均值只有 -0.004。
**推断**：当 `rc_T` 较高时，timeout 只罚很少；策略可能选择“慢慢开、别撞”而不是冒险完成。高 timeout 也可能是 scene route 本身很长，需要区分。
**待验证**：按 route length、timeout 时的 rc、以及 timeout 前的速度分层，确认是策略保守还是 scene 长度。

### 4.5 数据覆盖小、梯度方差大

**事实**：40 个 update × 8 scene = 320 次 draw，unique 286 个 scene（train 共 1500）；difficulty 抽样为 hard 128、medium 113、easy 71（hard 约 41%），符合 `difficulty_weights` 的过采样。
**推断**：每 update 只看到 8 个 scene，64 条 episode；相对 1500 个 train scene，覆盖不足。策略可能在少量 scene 上反复优化，holdout 泛化有限。
**代价**：增加 `scenes_per_update` 会线性增加 rollout 时间，但能显著降低方差。

### 4.6 全量 holdout 评测成本过高，且在变慢

**事实**：每 5 个 update 跑一次 500-scene holdout，单次从 35 分钟涨到 63 分钟；7 次已经约 5.5 小时。
**推断**：评测本身成了主要时间开销，也挤占了 update 数；评测变慢与 RSS/swap 增长相关，可能是内存压力/分配器碎片。
**运维风险**：8 小时监控窗口可能在训练结束前到期；训练是 detached 的，监控停止不等于训练停止。

### 4.7 内存/swap 持续增长，根因未确认

**事实**：RSS 从启动时约 1.9GB 涨到峰值约 8.7GB；swap 从 6.8GB 涨到 20GB。`evaluate_holdout` 每个 episode 都 `build_env` + `env.close()`，表面上没有保留 env。
**推断**：可能是 Python/CUDA/分配器碎片，或 MetaDrive/panda3d 的全局状态没有完全释放；也可能有隐藏引用。现在没有证据指向单一 leak 点。
**风险**：继续跑更长训练可能 OOM 或严重 swap thrash；需要在下一轮前做一次独立的内存 profile。

### 4.8 Selection 的 winner's curse

**事实**：best 是在 8 次 holdout eval 里取最大值；500-scene 评测的 SE 约 2.1pp。
**推断**：`0.688` 很可能包含 1~2pp 的乐观偏差；真实水平更接近 0.67。用这个 checkpoint 去跑 test500，期望值不高。
**缺口**：baseline 没有保存 per-scene rows，无法直接做 paired bootstrap；只能做独立样本区间，或者重跑一次 r3 baseline（额外成本）。

### 4.9 评测口径变化，历史数字不可直接比较

**事实**：`pipeline/eval/closed_loop.py` 现在显式 `reset(seed=spec.seed)`，并新增 `strict_success = success and not collision and not off_road`。
**含义**：holdout `0.646` 是新口径；历史 val500 `0.664` 是不同 split、可能不同 reset/终止语义，不能直接说“r3 退化了”。
- test500 没有历史 baseline；gate 只能报绝对 success + CI，不能伪造 delta。
- **gate 指标未统一**：`arch_lock` §12.4/§12.8 写 `success >= 72.8%`，而 `pipeline/eval/closed_loop.py` 的 `success` 是原始 `arrive_dest`；`strict_success` 才排除 collision/off_road。本轮 holdout 的 `success == strict_success`（没有差异），但 test500 协议必须先预注册用 `strict_success`。

### 4.10 合同执行上的偏差

- calibration 200 已删除；reward 权重仍是 provisional。
- `entropy_proxy` 命名错误（实际是 log-std）。
- baseline rows 最初未落盘；已用独立进程补跑并精确复现 0.646，paired 统计现在可信。
- oracle review 的 P2：`evaluate_holdout` rows 丢了 `invalid_action`/`infeasible`/`difficulty`；`_timeout_terminal_transition` 分支没有把 reward 加进 `reward_sum`（正常流程不可达）；`run_meta.json` 没有 git sha / reward weight hash / eval split sha。
- 没有 failure-window、没有 cost/Lagrangian、没有 WM 参与，这些是符合合同的。

### 4.11 Holdout IL exposure confound（P0，oracle review 发现）

**事实**：r3 IL checkpoint 的 `config.train_seeds` 有 1389 个 scene；RL holdout 500 中有 **350 个**在 IL 训练集里：

| 子集 | n | baseline | update 20 | update 50 |
|---|---:|---:|---:|---:|
| IL-seen | 350 | 0.551 | 0.600 | 0.586 |
| IL-unseen | 150（全 easy） | 0.867 | 0.893 | 0.900 |
| easy | 188 | 0.872 | 0.904 | 0.910 |
| medium | 173 | 0.607 | 0.653 | 0.647 |
| hard | 139 | 0.388 | 0.439 | 0.410 |

- IL-seen 的 350 = 173 medium + 139 hard + 38 easy；IL-unseen 的 150 全部是 easy。
- **test500 的 medium/hard 是 IL-unseen**；当前 holdout 的 medium/hard 全部是 IL-seen。
- 这不是 RL trainer 的 split violation：RL 训练确实没看 holdout seeds；问题在 **初始化 r3 的 IL exposure**。但它意味着 holdout 的 medium/hard 增益不能外推到 test500。
- 影响：按 holdout per-difficulty 外推 test500 会得到约 0.687，但这是乐观的；真实 unseen medium/hard 可能更低。**不能用当前 holdout 证明 test500 会过 72.8%。**

---

## 5. 猜想（按优先级，含验证方法）

> 标记：[事实] 有日志/代码证据；[推断] 目前最合理的解释；[待验证] 还没有数据。

### H1（最高）探索塌缩导致 group-relative 信号主要是噪声

- [事实] log-std ≈ -4.85，std ≈ 0.008。
- [推断] 8 个成员动作几乎相同；terminal 差异由混沌放大，advantage 不反映动作质量。
- 验证：记录每个 decision 的 group 内动作 L2；统计“动作距离 → 终局差异”的关系；用固定动作重放同 scene，估计纯环境分叉率。
- 修法：entropy bonus / log-std floor / 更大 group / 动作扰动；需要 owner 批准合同变更。

### H2 Reward 误设（comfort 过强、timeout 过便宜）

- [事实] progress 与 comfort 抵消；timeout 曾达 31%~33%；low_speed 几乎为 0。
- [推断] 策略可能偏向保守/慢速，或 comfort 抑制必要机动。
- 验证：在固定 holdout / 诊断集上分解 success/collision/timeout 的 return；对比 dense 与 terminal 占比；按 timeout 时 rc 分层。
- 修法：校准一次权重（不扫网格），必要时把 comfort 降为诊断或更小权重；timeout shaping 改为更强的固定惩罚或 `-w_timeout*(1-rc)^2`。需要 owner 批准。

### H3 数据覆盖不足 / 更新太少

- [事实] 286 unique train scenes；每 update 8 scene；50 updates。
- [推断] 策略在少量 scene 上过拟合，holdout 泛化不足。
- 验证：在“已见 train scene”与“未见 holdout”上分别评估，比较 gap；统计 scene 重复次数与收益。
- 修法：`scenes_per_update` 16~32、updates 100+；或 hard-scene curriculum。不改合同方向。

### H4 表征/学习率瓶颈

- [事实] encoder 冻结；TSB lr 是 planner 的 1/10；TSB 可训练 546k，planner 363k。
- [推断] planner 在固定表征上调整，可能很快到达 r3 特征的上限；TSB 太慢，难以改变表征。
- 验证：看 TSB/planner 的 grad norm 与参数变化；做 planner-only vs TSB 提高 lr 的短 ablation。
- 修法：提高 TSB lr 或解冻部分 encoder（后者违反当前合同，需要 owner 批准）。

### H5 统计噪声 / winner's curse

- [事实] holdout 10 次评测，best 0.688；baseline rows 已补，paired McNemar：baseline→update 20 **+4.2pp, p=0.0099**；baseline→update 50 +3.4pp, p=0.033；update 20→50 **-0.8pp, p=0.60**。
- [推断] 在这个 IL-exposed holdout 上 r3→RL 的提升是真实的，但幅度只有 3~4pp；0.688 是 10 次选择中的最大值，真实期望更接近 0.675；update 20 与 50 无法区分；由于 medium/hard 全部 IL-seen，不能外推到 test500。
- 验证：已做 paired bootstrap/McNemar；下一步用更多 eval 或 lower-confidence selection 降低 winner's curse。
- 修法：下一轮保存 baseline rows；selection 用更稳的指标（如 3 次 eval 均值或 lower confidence bound）。

### H6 Timeout/max_decisions 是 scene 长度伪影

- [事实] holdout timeout 8.4%；训练 rollout 有些 update 31%~33%；平均 70 decisions。
- [推断] 部分 timeout 是 route 太长/路口等待，不是策略失败；当前 timeout penalty 对这些 scene 不公平。
- 验证：timeout 按 route length、rc、速度分层；看 timeout 是否集中在长 route。
- 修法：timeout 归一化到 route length；或对 rc 已很高的 timeout 减小惩罚。需要 owner 批准。

---

## 6. 当前能确定和不能确定的

**能确定：**

- 训练 pipeline 已跑通；reward/advantage/PPO/eval 都有测试和日志；50/50 正常结束。
- holdout 最好 0.688（update 20），最终 0.680（update 50），未达到 72.8%。
- 在这个 IL-exposed holdout 上，r3→RL 的提升统计显著（+4.2pp，p=0.0099）；update 20 与 50 无法区分。
- **holdout 不是 IL-clean**：350/500 被 r3 IL 训练见过；medium/hard 全部 IL-seen；IL-unseen 只有 easy。test500 的 medium/hard 是 unseen，不能外推。
- test500 没有跑过，没有被污染。
- reward 未校准；探索 std 很小；评测很贵；内存/swap 在增长（进程已结束，无法确认是否 leak）。

**不能确定：**

- 平台化到底是探索、reward、数据覆盖、表征还是统计噪声导致；目前没有足够证据把锅扣在单一原因上。
- 0.688 是选择偏差下的乐观值；真实期望约 0.675，不能声称“接近 72.8%”。
- 当前 holdout 的 medium/hard 增益有多少是 RL 泛化、多少是 r3 IL exposure 的记忆效应；没有 clean holdout 无法拆分。
- 内存增长是否是真 leak；没有 profile。

**不要做的事：**

- 不要用 test500 反复试错；不要用 val500 调 RL；不要在看到 holdout 0.688 时宣称接近 72.8%。
- 不要在训练中途改 reward/config；当前 run 只能作为一次“未校准 baseline run”。

---

## 7. 最终决定与交付

**Owner 决定（2026-11-11）：停止 RL 线；不消费 test500。**

理由（基于 oracle adversarial review + 本文件数据）：

- best holdout 0.688 是 best-of-10 的乐观值；10 次均值 0.6748，诚实估计约 0.675；即使按 0.688 的 Wilson CI，上界 0.727 仍低于 72.8%。
- holdout 有 IL exposure confound：350/500 被 r3 IL 训练见过，medium/hard 全部 IL-seen，IL-unseen 只有 easy；test500 的 medium/hard 是 unseen。当前 holdout 不能证明 test500 会过 gate。
- 探索塌缩（std≈0.008，logstd 贴近 clamp -5）使 group-relative 信号主要来自系统混沌，而不是动作质量；这次 run 没有真正测试 group-relative PPO 的设计意图。
- reward calibration 已取消（owner 决定），comfort 与 progress 抵消，timeout 在高 rc 时几乎免费；这是一次 **provisional-weight baseline run**，不能当作方法上限。

**交付物**：

- run：`runs/ppo_group_r3_261111015244`
- best：`best_holdout.pt`（update 20，0.688）
- last：`last.pt`（update 50，0.680）
- 日志：`train_log.jsonl`（50 updates）、`runs/logs/ppo_group_train.log`、`runs/logs/ppo_group_monitor.log`
- baseline per-scene rows：`runs/ppo_group_r3_261111015244/baseline_holdout_rows.json`
- 代码/文档 commits：`19cba26`、`aa45aa8`、`6385292`（`pnp_ilrl`）
- **test500：未消费，保持干净。**

**如果未来重启 RL 线（预注册清单，未执行）**：

1. 从 `data/specs/source/train_10000.json` seeds 3000–3999 生成 clean holdout，匹配 test500 difficulty mix；先用 r3 和当前 best 保存 baseline rows。
2. 修探索：logstd 下限从 -5 提高到约 -2.5（或 entropy bonus），记录 group 动作 L2 / clamp 饱和比例 / 同 group terminal 分歧。
3. reward 权重如需调整，先由 owner 批准；不做 calibration split，不在 val500/test500 上调。
4. 增大 `scenes_per_update`（8→16）、updates（50→60–80）；eval 降频到 200-scene subset，最后 full clean holdout。
5. 预注册 selection：报告 3 个候选的均值和 selected，不把 best-of-N 当 headline；gate 指标统一 `strict_success`；checkpoint/run_meta 写入 git sha、reward weight hash、eval split sha。

---

## 8. 已关闭的问题 / 未来重启才需要拍板

**已关闭：**

1. **test500 是否现在消费？** Owner 决定：停止 RL 线，不消费 test500；保留给未来可能的 clean run。
2. **是否继续第二轮？** 本轮不继续；先解决 clean holdout、探索两个前提，再谈预算。

**未来重启才需要拍板：**

1. 是否批准新的 clean RL holdout（seeds 3000–3999），替换当前 IL-contaminated holdout 做 selection。
2. 是否批准 exploration 合同变更（logstd floor / entropy bonus）。
3. 第二轮预算：`scenes_per_update=16`、`updates=60–80`、eval 每 10 update 一次 200-scene subset。
4. 是否允许提高 TSB lr 或解冻部分 encoder（H4，合同变更）。

---

## 9. 复现命令

训练（已完成 50/50）：

```bash
cd /workspace/01_Proj/DRL_PathPlan
PYTHONPATH=. LD_LIBRARY_PATH=.venv/gl/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m pipeline.rl.train_group \
  --config config/train_rl_group.yaml \
  --out runs/ppo_group_r3
```

监控：

```bash
tail -f runs/logs/ppo_group_monitor.log
tail -f runs/logs/ppo_group_train.log
```

test500（**未执行**；owner 已决定停止 RL 线，保留给未来 clean run）：

```bash
PYTHONPATH=. LD_LIBRARY_PATH=.venv/gl/usr/lib/x86_64-linux-gnu \
  .venv/bin/python -m pipeline.eval \
  --specs data/specs/test_500.json \
  --split-manifest data/splits/test_500.json \
  --split test \
  --allow-final-test \
  --limit 500 \
  --ckpt runs/ppo_group_r3_261111015244/best_holdout.pt \
  --device cuda --max-steps 600 \
  --out runs/eval_test500_rl_once
```

注意：`--split test` 必须显式 `--allow-final-test`；`--limit` 默认是 10，gate 必须写 `--limit 500`。

跑完后只看 `metrics.json` 的 `strict_success_rate`，不要把 `success_rate` 当 gate。

---

## 10. 诊断证据：为什么提升有限（2026-11-11）

> 只读诊断，产物在 `runs/analysis/rl_diagnostics/`（README、`metrics.json`、`fixed_scene_sweep.json`、`reward_trends.json`、`traces.pkl`、10 张 `scene_*.png`）。没有跑 test500/val500，没有训练。

### 10.1 训练 reward 曲线

- `reward_mean` slope **+0.437/update**（p=5.9e-8），first10 **-13.80** → last10 **+5.91**；范围 [-26.69, +16.78]。
- 主要来自 **comfort**：-0.341 → -0.116（p=3.2e-17）；progress 基本不变（0.115→0.108，p=0.11）。
- success +0.13pp/update（p=0.14）；off-road -0.19pp/update（p=0.026）；collision 与 timeout 的趋势不显著。
- holdout strict_success 基本持平（slope p=0.656，10 次均值 0.6748，范围 0.664–0.688）。

### 10.2 固定 10 场景 × 8 members 的 checkpoint sweep

同一组 holdout 场景、同一组 sampling seed，只换 checkpoint：

| update | reward | success | collision | off_road | timeout |
|---:|---:|---:|---:|---:|---:|
| 1 | -22.16 | 0.350 | 0.300 | 0.275 | 0.063 |
| 10 | -14.19 | 0.338 | 0.188 | 0.275 | 0.200 |
| 20 | -5.67 | 0.388 | 0.175 | 0.200 | 0.238 |
| 30 | -3.17 | 0.412 | 0.175 | 0.200 | 0.212 |
| 40 | -1.12 | 0.425 | 0.150 | 0.188 | 0.225 |
| 50 | **+0.89** | **0.425** | **0.125** | **0.175** | **0.275** |

**Reward +23 分，success 只 +7.5pp，timeout +21pp。** 策略主要在用 timeout 换 collision/off-road 的下降，而不是完成路线。

### 10.3 10 个场景的终局

| seed | difficulty | success/8 | terminal |
|---|---|---:|---|
| 2500 | easy | 0 | off_road ×8 |
| 2600 | easy | 0 | timeout ×8 |
| 2750 | easy | 8 | success ×8 |
| 2950 | easy | 8 | success ×8 |
| 2525 | medium | 0 | off_road ×8 |
| 2675 | medium | 8 | success ×8 |
| 2900 | medium | 0 | collision ×8 |
| 2550 | hard | 0 | collision ×2 + timeout ×6 |
| 2700 | hard | 0 | collision ×3 + timeout ×5 |
| 2850 | hard | 7 | success ×7 + collision ×1 |

- **9/10 场景的 8 个 member 终局同生共死**；只有 2850 同时有 success 和 failure。
- 每场景图：`runs/analysis/rl_diagnostics/scene_{seed}.png`（3×3：8 个 member + overlay）。

### 10.4 Policy 坍缩

- rollout mean logstd **-4.818**（normalized std≈0.0081）；训练 `entropy_proxy` -4.699 → -4.889（p=1.7e-14）。
- **66.3% 的 logstd 元素卡在 clamp 下限 -5.0**；四维饱和比例 59.5% / 63.4% / 70.4% / 72.0%。
- t=0 时 8 个 member 的 `action_mu` 完全相同（pairwise L2=0）；初始 sampled L2 仅 0.016–0.070，后续分叉是状态混沌，不是主动探索。
- 物理采样 std 约 accel 0.06 m/s²、steer 0.023、curvature 0.013、speed 0.12 m/s。

### 10.5 组内 advantage 区分度

- 没有 GAE/critic；用的是 per-step active-set LOO advantage。
- 9/10 场景终局一致 → terminal advantage 对组内成员相同，只能优化 dense reward。
- 唯一混合场景 2850：success vs failure 的 mean advantage 差 **+31.05**（collision member ≈ -28.55），说明混合终局时 LOO 能强区分。
- 全样本：success 与 episode mean advantage 的 point-biserial **r=0.19（p=0.09）**，AUC **0.531**；group/global std=0.757，但主要来自 dense 分量，不是 success。
- skipped（active set <3）1.08%。

### 10.6 结论

这次 RL 的提升有限，主要不是“PPO 没跑通”，而是三个因素叠加：

1. **Reward 结构**：timeout 惩罚远小于 collision/off-road，comfort 又奖励平滑；策略学会“少撞、少越野、更多 timeout”，success 只吃到边际收益。
2. **探索坍缩**：logstd 贴近 clamp，group 成员几乎是同一策略；没有足够探索去找成功轨迹。
3. **组内终局同质化**：9/10 场景全成功或全失败，per-step LOO advantage 没有 terminal 对比，只能优化 comfort/progress；这与 reward 的 comfort 项叠加，进一步把策略推向“平滑但慢/超时”。

再叠加 holdout 的 IL exposure confound（§4.11），当前 holdout 的 medium/hard 增益不能外推到 test500 的 unseen medium/hard。故停止 RL 线、保留 test500 是符合数据的决定。
