# v6 程序级预注册（Net 重构 → 重训 → 奖励 v5 → RL → 扩量）

> **状态：冻结（2026-10-01；Gate0 复核修正版）**。6 阶段（P0–P5）、每阶段 Oracle 门；Oracle 评审总额 = **6 次初评 + 每门 ≤ 2 次重审**。
> 关联：[`docs/v6_net_design.md`](v6_net_design.md)（架构规格）、[`docs/rl_reward_v5.md`](rl_reward_v5.md)（奖励规格）。
> 上游：`.slim/deepwork/v6-net-retrain.md`；目标 = **RL 大幅提升基座闭环性能**。

## 0. 阶段表（P0–P5）

| 阶段 | 内容 | 归属 | Oracle 门（一句话理由） |
|---|---|---|---|
| **P0 前置** | ① 保守清理（runs/、/tmp、/tmp/opencode，保留证据类）；② **设计冻结文档**（本三件）；③ Stage A/B 吞吐与重训成本测量 | fixer×3 | **Gate0**：设计冻结评审——防止带错误规格开工 |
| **P1 Net 实现** | A1–A4（交叉注意力头 / 去池化 / st_gnn 主路径 / nav）+ E1（router 监控修复 + 专家范数探针）；单测/形状/参数量/吞吐 | fixer×2（`net/model+policy+plan_head` 一 lane；`env/*+net/mem` 一 lane） | **Gate1**：实现正确性与吞吐——新架构是重训前提 |
| **P2 奖励 v5 实现 + 审计** | `low_speed`、终局值、权重、rc 档反解工具、λ 选项、**奖励审计工具**（真实聚合器 ≥ 50 rollout 重放） | fixer×1 | **Gate2**：实现 + 审计剖面 vs 目标——防止带错误奖励开训 |
| **P3 Stage A/B 重训** | 新基座（E-β″）从 Stage A 重训 + mini 闸 + 配方链（r5 dagger + expert5k 锚 mild=1.0）+ 验收（E3 硬闸）+ keep-best | fixer（驱动）+ 监控 | **Gate3**：新基座验收证据——不达标不进 RL |
| **P4 Stage C RL** | v6 臂：奖励 bundle、rc 扫档、λ 对照、`ttc`/`lane_boundary`/`lane_center`；臂预算/顺序/止损；keep-best 闸 | fixer（链驱动）+ 监控 | **Gate4**：臂结果与采纳判定 |
| **P5 收尾/扩量** | 数据扩量（5k / steps）、group baseline、文档回填、最终报告 | fixer + orchestrator | **Gate5**：最终复核 |

## 1. P0 前置（进行中）

- ① 保守清理（候选清单待用户确认；仅清 runs/、/tmp、/tmp/opencode，证据类保留）。
- ② 设计冻结文档 = 本文件 + [`docs/v6_net_design.md`](v6_net_design.md) + [`docs/rl_reward_v5.md`](rl_reward_v5.md)。
- ③ Stage A/B 吞吐与重训成本测量（决定 P3 预算）。
- **Gate0**：设计冻结评审——防止带错误规格开工。

## 2. P1 Net 实现

- A1 交叉注意力头 / A2 去池化 / A3 st_gnn 主路径 / A4 nav（P0-3）+ E1（router 监控修复 + 专家范数探针）。
- 验收：形状/参数量单测、信息充分性对照（左/右 cut-in 可区分）、吞吐（steps/s）——规格见 [`docs/v6_net_design.md`](v6_net_design.md) §1.3/§4。
- **Gate1**：实现正确性与吞吐——新架构是重训前提。

## 3. P2 奖励 v5 实现 + 审计

- 内容：`low_speed`、终局值（**重解定稿**：临时值 → 完整项集重放重解）、权重（`speed_ratio` 0.4 / rc 扫档 3/10/30）、rc 档反解工具、λ 选项、奖励审计工具（真实 `RewardAggregator` ≥ 50 rollout 重放 → 分类剖面 vs 目标；样本与反解互斥、每类 n ≥ 10）。
- 规格见 [`docs/rl_reward_v5.md`](rl_reward_v5.md)（§1 临时值、§3 反解程序、§7 审计工具）。
- **Gate2**：实现 + 审计剖面 vs 目标——防止带错误奖励开训。

## 4. P3 Stage A/B 重训（E3 硬闸）

### 4.1 配方链（E-β′ 参照；E-β″ 同链重训）

- **E-β′（旧架构参照）** = `fixA_nold`（A20+B20，micro512）→ **phase3 r5 dagger + expert5k 锚 mild=1.0（micro256）**。
  - 锚定 run/argv（`runs/BTC20260929-182033_stageB_phase3_exp_beta_anchor1/manifest.txt`）：
    `tools/train.py --phase3 datasets/BTC20260929-1357_phase3_dagger_r5 --phase3-round 5 --ckpt runs/BTC20260929-0425_fixA_nold/stage_b/final.pt --config config/default.yaml --model-config config/model.yaml --micro-batch-size 256 --phase3-anchor --phase3-anchor-bc-dir datasets/BTC20260926-2343_expert5k --phase3-anchor-mild-weight 1.0`
  - 零点：clean500 **0.446** / eval500 **0.436**（`docs/rl_stage_c_experiments.md` §3）。
- **E-β″（v6 新基座）** = **同配方链重训**：新架构从 Stage A 重训（A20 micro256 → B20 micro512）→ 同 argv 的 phase3 r5 dagger + expert5k 锚 mild=1.0（micro256）。
- **数据窗口 sha256 清单前置**：P3 开跑前冻结并落 repo：`datasets/BTC20260926-2343_expert5k`、`datasets/BTC20260929-1357_phase3_dagger_r5`、`datasets/BTC20260927-1734_expert500val` 及所用 spec json 的 sha256 清单；清单写入 run manifest。

### 4.2 mini 重训闸（32k 行，先于全量）

先跑 32k 行切片 mini 重训（A 2ep + B 1ep；命令口径见 `/tmp/opencode/v6_stageA_cost.md` §4），验收 5 条：

1. 无 OOM，Stage A micro256 峰值 **≤ 10.0 GiB**；
2. Stage A 单 epoch **≤ 28 s**（32k 切片；实测基线 24–25 s + 新架构增量）；
3. Stage B **micro 选定档无 OOM**，且 **`st_gnn` 在主路径实际执行**（耗时/图证据）；
4. 损失有限（无 NaN/Inf），曲线无异常；
5. mini ckpt 可评测（**≥ 16 条**）且 run manifest 完整 + 工作树 clean。

### 4.3 micro / 可训性预案

- Stage B **保持 `st_gnn` 冻结**（现口径；frozen micro512 可行）。
- 如需解冻（仅当 WM loss 接线）：**退 micro256**（micro512 会 OOM，`/tmp/opencode/v6_stageA_cost.md` §7）**+ 约 1.8× 时间**，且解冻前须过 mini 验证。

### 4.4 abort 判据与预算

- **abort**：单 epoch 墙钟超基线 **+25%** 即停（检查是否落 +7–23% 增量带外）。
- **预算 ≤ 5 h/轮**（含 A+B 全量 2.6–3.1 h + phase3 链 + 评测/keep-best；超预算按优先级截断）。
- **E3 硬闸**：clean500 / eval500 ≥ **0.446 / 0.436**（细则 §6）。
- **Gate3**：新基座验收证据——不达标不进 RL。

## 5. keep-best 协议（P3/P4 通用）

- 周期 ckpt（`ckpt_every`）+ 全轨迹评测；采纳 **最优 ckpt** 而非末点。
- 已知偏差：子集150 最优点对全量系统性偏乐观 **2–4×**（`docs/rl_stage_c_v4_report.md` §3.4）⇒ 候选 ckpt 必须 **全量 clean500 复评** 后才可采纳；keep-best 是**减损工具、不承诺增益**（同上 §4）。
- **全量复评时间计入预算**：每候选 ckpt 一次全量 clean500 复评 ≈ **3–8 min**（实测 185–484 s；`docs/rl_stage_c_experiments.md` §6.1），计入 P3/P4 预算（§4.4 / §7.1）。
- 采纳记录：ckpt 路径 + 子集/全量读数 + sha256。

## 6. E3 新基座验收（硬闸，细则）

- 判据：**clean500 succ ≥ 0.446 且 eval500 succ ≥ 0.436**（与 E-β′ 同 harness / 同 spec / 同 seed 口径；spec 哈希与零点引用见 `docs/rl_stage_c_experiments.md`；E-β′ 零点 0.446 / 0.436 见 `docs/phase3_rootcause_analysis.md`）。
- 辅助：配对 net / z、fixed/broken、offΔ / collΔ（全轨迹，不仅末点）。
- **统计口径**：全量 500 条的 succ 单点差在 **±2pt** 内不作胜负结论（历史零点/复评差异量级）；以**配对 net**（逐 (id,seed) 配对）辅助读法为主。
- **迭代/回退**：不达标 → ≤ 2 轮迭代（**不改冻结架构**：数据 / 超参 / 种子层面）；仍不达标 → **回退 E-β′（0.436）作为 P4 init**（= **旧架构代码路径**，见 [`docs/v6_net_design.md`](v6_net_design.md) §5 #9；保留旧实现与 git tag，不得覆写删除），E-β″ 降级为对照并记录。
- 证据：`runs/` 全量评测档 + `episodes.csv` + sha256。

## 7. P4 Stage C RL（臂预算 / 顺序 / 止损；λ 对照）

### 7.1 臂顺序、预算与止损

- **spec 池 pin = `env/specs/scenarios_train_dagger_r1.json`（500 条；pool 是唯一防塌杠杆，`docs/rl_stage_c_v4_report.md` §4）**；`--pool local --envs 1`，GPU 串行（臂间可比）。
- **奖励 bundle（定义）**：剖面 C 的默认奖励改动集合作为**一个整体底座臂**——`speed_ratio` 0.4 + rc 选定档 + `low_speed` 启用 + 终局值定稿表；其余臂在 bundle 之上单变量。
- **开臂前置**：P3 后、P4 前先跑一次奖励审计/重解（E-β″ 上定稿终局值与剖面；[`docs/rl_reward_v5.md`](rl_reward_v5.md) §7）；未复算不得开臂。
- **顺序（每臂单变量，前臂通过再开下臂）**：
  1. **bundle 底座臂**（v5 奖励默认全量）；
  2. **rc 扫档 3 / 10 / 30**（固定剖面、终局值为因变量）；
  3. **λ 0.95 vs 0.98**（细则 7.2）；
  4. **`ttc` 臂**（**前置**：离线证伪，would-be 触发率 ≈ 0 则改项或不做——[`docs/rl_reward_v5.md`](rl_reward_v5.md) §5）；
  5. **`lane_boundary` 臂**；
  6. **`lane_center` 臂**（两文档口径一致：`docs/rl_reward_v5.md` §5 与本表同为 −0.1 / deadband 0.25 m）。
- **预算（按 v4 实测外推）**：单臂 ≈ 训练（u200 ≈ 8 min）+ 双评测（clean500 + eval500 各 ≈ 3–8 min，实测 185–484 s）≈ **20–25 min**（典型）；keep-best 候选全量复评 ≈ 6–8 min/候选（**计入**）；P4 整批 ≤ **4 h**（串行）。
- **止损**：每臂全轨迹监控；若 u25/u50 出现早崩（子集配对 net 低于 init 且 z 显著；具体阈值随该臂预注册冻结），提前停臂；末段崩解候选不采纳（keep-best 协议，§5）。

### 7.2 λ 0.95 vs 0.98 对照（细则）

- GAE λ：现行默认 **0.95**（`pipeline/trainer.py` gamma 0.99 / lam 0.95）vs **0.98** 对照臂。
- 单变量：其余 pins 完全一致；GPU 串行（臂间可比）；记录 returns / EV / 崩解窗口。

## 8. 数据扩量条件（P5；顺序约束）

- **steps 上调（100→200/400）以崩解修复为前提**（keep-best 采纳协议 + 末段崩解诊断先行）。
- **distinct 500→5k**：重训完成后扩量（P5）。
- 曝光监控：以**单轮采集的 spec 池**为窗口统计 per-spec episode 计数（来源 `tools/dagger_collect.py` report `per_spec` / `tools/make_dagger_pools.py` 覆盖口径）；判据：零曝光 = 0、per-spec ≤ 2。

## 9. 监控纪律

- **20 min × 8 h** 监督（`/tmp/opencode/v6_supervise.sh`，orchestrator 每轮重挂）。
- 训练进程 **setsid** 解耦 task/session。
- **GPU 串行 = 臂间可比**；诊断评测允许并行并标注。
- 单 seed 仅方向性；每批预注册冻结；正向结论 ⇒ seed=11 复现（`docs/rl_stage_c_v4_report.md` §5.6）。

## 10. 协议与边界

- 闸：全轨迹 + 分层子集（v4 闸弱筛选教训，`docs/rl_stage_c_v4_report.md` §4）。
- 奖励审计（Gate2）通过后才开 P3；**P3 后、P4 前复算一次**（[`docs/rl_reward_v5.md`](rl_reward_v5.md) §7）；E3 不达标不进 P4。
- 已裁定：st_gnn 在 Stage C **冻结**（无 WM loss 解冻无梯度）；Stage A/B 按原 WM 损失训练（[`docs/v6_net_design.md`](v6_net_design.md) §2）。
- 开放项：rc 档默认档位（P4 臂后定）；清理候选清单待用户确认。

## 11. 修订锚（Gate0 remediation）

- 三份冻结文档（本文件 + `docs/v6_net_design.md` + `docs/rl_reward_v5.md`）的 Gate0 修正版 commit：`d27395dd0e1590d2151c42830f6a930bb5aafd2c`（短 `d27395d`，2026-10-01，首次 commit）。本锚行由第二次小 commit 写入（不改动其余内容）。
