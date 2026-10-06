# Stage C v4 · 预注册（冻结版；开跑前提交，本文件 hash 见实验记录）

> 目的：在 v4 修复栈（critic/优化修复 + 奖励口径修复 + 训练侧探针 + 锚修复）上，检验 **先"不损伤"（clean net ≥ −10）、再"方向性正向"（net ≥ +20 且 z ≥ 1.96）**；为"RL 提升基座闭环"建立正确底座。

## 0. 口径与零点
- 语义：**plan**（G4 限定语照旧：执行参考生成器未优化、PPO 梯度只覆盖首动作）。
- **代码基线**：`cf74b1c`（①–⑧）+ `3590d30`（**⑨ 锚修复**：mu/raw logstd 进梯度；coef=0 逐位不变，CPU+GPU 指纹已证）。
- 零点：E-β′ **sub150 succ 0.4467**（canonical `6adbe0af…`）/ clean500 0.446 / eval500 0.436；子集150 = clean500 文件序前 150。
- 评测：clean500 主 + eval500 辅；`--eval-reference plan`；workers 6；内容哈希 pin。

## 1. 共同 pins（全臂一致）
- `--spec env/specs/scenarios_train_slice200.json`（**V-data 例外**：`scenarios_train_dagger_r1.json`，500 条，train-only、与 eval 零重叠）、`--pool local --envs 1`、tracker lqr、rollout 256、`--updates 100`、`--seed 0`、`--trainable-scope design`、`--ckpt-every 25`、`--adv-norm global`、**`--kl-anchor-coef 0.0 --kl-anchor-final-coef 0.0`**（无锚；仅 V-anchor 例外）、奖励默认（含已修：CaRL 乘子 / 限速源 / 终局掩码）、探针默认开（grad_group / anchor_grad / episodes / drift）。

## 2. 臂集（7 臂，单变量嵌套于 V0）
| 臂 | 变量（相对 V0） | 目的 |
|---|---|---|
| **V0** | ＋critic bundle：`--critic-warmup-updates 10 --value-lr-scale 5 --target-kl 0.05` | 全修复基线 |
| **V1** | －critic bundle（warmup 0 / value-lr 1 / 无 target-kl） | 归因 critic 修复 |
| **V-anchor** | ＋`--kl-anchor-coef 0.01 --kl-anchor-final-coef 0.01`（⑨ 后=真锚） | 真 KL 约束 |
| **V-data** | ＋`--spec env/specs/scenarios_train_dagger_r1.json`（500 池） | 数据规模 |
| **V-lr** | `--lr 1e-4` | 稳定化 |
| **V-ttc** | ＋`--reward-term-weight ttc=-0.5` | 失败前预警项 |
| **V-boundary** | ＋`--reward-term-weight lane_boundary=-0.2` | 贴边稠密项 |

## 3. 判据（先声明，不事后改）
1. **训练侧软护栏**（只标注）：grad_group/*、anchor_grad_norm/*、episodes/*、EV、approx_kl、kl_early_stop、drift。
2. **中期**：u25 **只观察**（不再停链）；**u50 破线**（net < −20 或 offΔ ≥ +0.10）⇒ 该臂判 early-collapse、跳过终评、继续下一臂。
3. **终评**（过 u50 才跑）：u100/final clean500 全量 + eval500；**net ≥ −10 = 不损伤**；**net ≥ +20 且 z ≥ 1.96 = 正向**。
4. **任何正向 ⇒ seed=11 复现**（优先于后续边际臂）。
5. 报 fixed/broken/z、offΔ/collΔ 与探针读数；单 seed 仅方向性证据。

## 4. 顺序与预算
- 顺序：**V0 → V1 → V-anchor → V-data → V-lr → V-ttc → V-boundary**（假设优先级）；正向复现与模块归因实验插队。
- 每臂 ≈ 训练 16–18min + 闸 2.5min×2 + 终评 2×~7min ≈ **30–35min**；7 臂 ≈ 3.5–4.5h。
- 8h 窗口内其余：seed 复现、**模块回滚归因**（policy/experts 分别还原到初始，测"哪个模块把闭环练坏"）、可视化重建（`runs/reward_viz/`）。

## 5. 已知边界
- plan 语义限定语（同 v3/G4）；单 seed；**v3 数值不可直接比**（奖励修复 + 新探针 + 锚修复均默认生效）；锚修复对 coef=0 臂逐位无影响（已证）；GPU 串行=臂间可比性（刻意；诊断类评测允许并行并标注）。

## 6. 变更管理
- 开跑前冻结提交（本文件）；任何 pin 变更须注明并重跑受影响臂；结果逐臂回填 `docs/rl_stage_c_experiments.md`（v4 章节）。

## 7. 链与监控
- 驱动：`/tmp/opencode/rl_v4_driver.py` + `rl_v4_chain.sh`（复用 `rl_v3_lib.py`；预检 clean tree/零点/GPU；逐臂 `.done`+结果档 `/tmp/opencode/rl_v4_results.{md,json}`）。
- 监控：`/tmp/opencode/v4_supervise.sh`（20min 一发）。
