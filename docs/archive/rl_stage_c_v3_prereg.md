# Stage C v3 · 预注册（2026-09-30，开跑前冻结）

> 目的：在"GAE 已修 + 语义锁定 plan + 多场景轮换已修 + 降噪已验证"的新基线上，检验**单轮 RL 是否能做到"不损伤"并给出可复现的方向性证据**。
> 纪律：clean SHA；先声明后跑；内容哈希 pin；双评测；逐臂回填；失败臂保留现场；**训练 reward 不作正判据**。

## 0. 口径与代码基线

- **语义 = plan（B）**（与评测一致）。**G4 限定语（必读）**：执行参考生成器（plan 预览尾 5 步 = policy 输出）在本阶段**未被优化**（no_grad、无 likelihood，且会随 PPO 漂移）；PPO 梯度**只覆盖首动作**。⇒ 结果只能声称"策略网络在 plan 执行口径下的变化"，**不得**声称规划器/WM/多步执行被改进；P0-3（rollout 复用 t0 nav token）未修 ⇒ 对 nav 敏感场景（路口/汇入/环岛）的解释受限；**退出条件**：下一阶段前完成 (C) plan-as-action 可行性路线，或补 (A) repeat 干净重训对照。
- **代码基线**（HEAD）：`d69a9d3`（adv_norm + GAE padding 修复）、`3731846`（S1 spec 轮换，默认 episode）、`e5f34e6`（S2 logstd 钳制，默认 None）、`b3ecdab`（eval `--eval-reference`，默认 plan）、`223a361`（lane_center，默认关）、`17efc2f`（ckpt_every）、`93164ea`（奖励钩子）。
- **评测**：clean500 主（`specs_val_only500.json`，内容哈希 pin）+ eval500 辅；`--eval-reference plan`；workers=6；零点引用 E-β′（0.446/0.436）/ L2（0.318/0.328）复用（plan 语义已证逐位可复现）。
- **训练 pins（全臂一致）**：pool=local / envs=1 / tracker=lqr / H=256 / updates=100 / seed=0 / `trainable_scope=design` / probe-on（`datasets/BTC20260926-2343_expert5k`）/ `trim_memory_every=4` / `ckpt_every=25`（暴露 u25/u50）/ `adv_norm=global`（注：envs=1 下 `per_scenario≡global`，组式归一化属 Step 2）/ `plan_reference=plan`。
- **漂移探针**（G4 必需项）：每 25 upd 在固定探针批上记录 `plan[1..5]` 相对 u1 的 RMS 位移与 `action_mu` RMS（monitor 字段）；异常抬升 ⇒ 结论标注受限。

## 1. 臂集（嵌套单变量 + 1 侧臂）

| 臂 | 相对 P0 的变量 | 目的 |
|---|---|---|
| **P0** | `--spec-rotation off` + KL 0.05→0 + logstd None | v3 自校准（**不得复用 P3 数值**） |
| **A1** | + `--spec-rotation episode` | 多场景轮换单变量 |
| **A2** | + `--policy-logstd-max -2.0` | 降噪单变量（plan 下已实测有效：off-road 90%→70%、Δ回报 +26.3） |
| **A3** | + `--kl-anchor-coef 0.01 --kl-anchor-final-coef 0.01` | KL floor 单变量（与 C3 的 −45 预算可比） |
| **S-A（侧）** | A1 参数 + 训练 `--plan-reference repeat_action`，**双语义评测** | 检验 (A) 干净路径是否仍活（~8min 训练） |

下一轮（不在本预注册）：奖励对齐臂（lane_center / TTC 稠密 / 量纲归一化 / terminal 结构）、组式 baseline（Step 2，须**显式锁 spec**——S1 后 auto-reset 会推进游标，1-spec 池或组期 `spec_rotation=off`）、per_scenario 归一化（需组模式）、KL floor 数值（0.01 vs 0.02/0.05）。

## 2. 判据（先声明）

1. **软护栏（训练侧）**：approx_kl / clipfrac / entropy|logstd / value EV 相对 u1 的区间 + **漂移探针**；越界仅标注，不自动停。
2. **中期硬闸**：u25 与 u50 ckpt 在 **clean500 预注册子集（150 条）**配对 net ≥ **−20** 且 off_road 无大幅抬升；破线 ⇒ 判"早期崩"，该臂停（省时）。
3. **终闸**：全量 clean500 配对 net ≥ **−10**（主）且 eval500 同向（辅）；报 fixed/broken/z。
4. **采纳**需 `seed=11` 复现（单 seed 仅方向性证据）。
5. **净增益**另立门槛：net ≥ **+20** 且 z ≥ 1.96 才称"优于 init"。
6. 采纳点 = 过闸臂中 clean500 net 最优的 ckpt（中间点按 ckpt_every 全记录）。

## 3. 预算

- plan 训练 ≈15.3 s/upd（C3 标定）×100 ≈ **26 min/臂**；评测 ≈ 子集闸 2×3min + 终评 2×7min ≈ 20min；4 主臂 ≈ **3.1h** + S-A ≈ 25min ⇒ 约 3.5h（GPU 串行；中期闸失败可提前终止）。

## 4. 已知边界 / 风险（必随结果引用）

- (B) 中间态限定语（§0）；P0-3 未修（解释边界）；WM 想象帧质量；
- S1 下每 spec 曝光 ≈1 episode（"多场景更好"需更大预算或块轮换；块轮换列 Step 2 备选）；
- 奖励-KPI 错配未在本轮处理——若复现（训练 reward↑/闭环↓），预声明下一步 = 奖励对齐臂，而非否定 RL；
- `0.01 floor` 未优化；ICC（plan）≈0.18（K=10 0.22）⇒ 组式以 **critic-free** 定位，不以消场景方差为收益预期；
- 证据卫生：每次证据 run 的 manifest 附 `git status --porcelain` 或 patch 哈希。

## 5. 变更管理

本文件开跑前冻结并提交（证据 run 引其哈希）；任何 pin 变更需注明并重跑受影响臂；结果表与判定逐臂回填到 `docs/rl_stage_c_experiments.md`（v3 章节）。
