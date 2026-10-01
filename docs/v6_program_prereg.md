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
- **数据窗口 sha256 清单前置（Gate1 补证-A 修订，2026-10-01）**：P3 开跑前冻结并落 repo，清单写入 run manifest：
  - **v3 补采（obs schema v3，世界键 → A4 生效；`tools/collect_expert.py` 同 seed/配置重采）**：
    - train：`datasets/BTC20261001-1327_expert5k`（360,407 行 / 259,522 可训；raw `024bc02e39b5ea4397f559a239b3ddd745e486a5ff5ef8baf582230cf0c75e97`；canonical `499c3f94c008ea9ebb57b54572969ceda98e71e6b2bcc79cc44fb680ab17b967`）
    - val：`datasets/BTC20261001-1327_expert500val`（36,147 行 / 25,855 可训；raw `3843ba86e0ea57de581beacef2158f91dc5d9a4b121fc98456cdcc513c566c3d`；canonical `e7592e83f22d7e5a2f32ff552481c8c67ce6290efc9d949219bc97dc8071a4e8`）
    - 逐位对照：expert5k 4988/5000、val 499/500 episode 与旧 v2 基础字段逐位一致；13 个 episode 为采集侧罕见 run-to-run 非确定性（非 v3 变更；差异记录见 `/tmp/opencode/v6_p3_data.md` §2）。
  - **phase3 冻结池（显式接受 v2 + 断言）**：`datasets/BTC20260929-1357_phase3_dagger_r5`（raw `45e38b3ccb1bc3d805f08db500bc64ec645d01b5e3d932a78a51eb3b38199056`）+ 锚 `datasets/BTC20260926-2343_expert5k`（raw `2b4b0d41b300044289fe4a6df7237a05b8f8a6b33932425596221bf0829a2bef`）。r5 driver 为 pre-v6 ckpt（新 policy/value 头 missing=20）→ 忠实 v3 重采不可行；v2 下 A4 rollout nav 回退 t0（影响仅 action_chain 第 2..6 步上下文，与 E-β′ 训练口径一致）。开跑前 preflight 断言（`/tmp/opencode/v6_p3_dagger_check.py --expect-v2`）+ manifest 记录 `a4_nav_rebuild=false (accepted)`。
  - spec json：`env/specs/scenarios_train.json` `1595acaf77c0c29226e01f8598d14a59d8aec3a0cba75180ce674f087483595b`、`env/specs/scenarios_eval500.json` `98856105eca17461bbdabbf88f203102be7b585fd3de82820b17460358dd4595`、`env/specs/scenarios_train_5k.json` `211575f09675975c200a6bac73cd50090b7836a7eb3515bc93393780814fb91e`（phase3 池）。
  - 旧 v2 `expert5k`/`expert500val` 仅作 E-β′ 参照，不用于 E-β″ 的 A/B 训练。

### 4.2 mini 重训闸（32k 行，先于全量）

先跑 32k 行切片 mini 重训（A 2ep + B 1ep；命令口径见 `/tmp/opencode/v6_stageA_cost.md` §4），验收 5 条：

1. 无 OOM，Stage A micro256 峰值 **≤ 10.0 GiB**；
2. Stage A 单 epoch **≤ 28 s**（32k 切片；实测基线 24–25 s + 新架构增量）；
3. Stage B **micro 选定档无 OOM**，且 **`st_gnn` 在主路径实际执行**（耗时/图证据）；
4. 损失有限（无 NaN/Inf），曲线无异常；
5. mini ckpt 可评测（**≥ 16 条**）且 run manifest 完整 + 工作树 clean；
6. **A4 nav 世界键生效断言（fail-closed）**：mini 训练（Stage A/B，数据 = §4.1 v3 数据集）**首 batch 无** `[stageA]`/`[trainer]` nav 回退 WARN，且数据契约含 `ego_world`/`route_world`（`obs_schema_version=3`；等价证据：rollout/教师强制逐步重建生效）；不满足 → 停（训练/评测口径不一致）。

### 4.3 micro / 可训性预案

- Stage B **保持 `st_gnn` 冻结**（现口径；frozen micro512 可行）。
- 如需解冻（仅当 WM loss 接线）：**退 micro256**（micro512 会 OOM，`/tmp/opencode/v6_stageA_cost.md` §7）**+ 约 1.8× 时间**，且解冻前须过 mini 验证。

### 4.4 abort 判据与预算

- **abort**：单 epoch 墙钟超基线 **+30%** 即停（A4 生效口径 full rollout @B=256 = **102.6 ms** vs 旧架构基线 84.93 ms = **+20.9%**（中心）；+25% 只剩 ~4pt 余量、带内抖动即误停 → 放宽至 +30%，仍在旧增量带宽 [+7%, +23%] 之外，可捕获真异常）。
- **预算 ≤ 5 h/轮**（含 A+B 全量 **≈2.9 h**（2.4 h 基线 × 1.21；带宽 2.6–3.0 h）+ phase3 链 + 评测/keep-best；超预算按优先级截断）。
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
- **开臂前置（E-β″ 奖励复算规格；Gate2 P4 MUST 修订，2026-10-01）**：P3 后、P4 前在 E-β″ 上跑一次奖励审计/重解
  （定稿该基座终局值与剖面，作为 P4 奖励口径；工具 [`docs/rl_reward_v5.md`](rl_reward_v5.md) §7）；**未复算不得开臂**：
  1. **排除 eval500**：审计池必须排除 `env/specs/scenarios_eval500.json` 的全部 500 条（P2 池与其全量重叠的教训；
     工具默认排除，`--no-exclude` 仅显式关闭，`--exclude ""` 报错）；
  2. **分层随机划分**：审计/反解样本按终局类分层随机划分（工具 `--split-seed` 固定并记录；替代 P2 的按序号交替
     ——两半难度不可交换：P2 collision 两半 5.34 vs 8.85）；
  3. **A/B 互换交叉验证**：`--swap-ab` 两向各跑一次（两半角色互换），两向结论一致方可作为 P4 奖励口径；
  4. **每类 n ≥ 50**：审计与反解各自按终局类计（`--min-per-class 50`；不足则该类只报 n 与区间、复算不通过）。
     注：排除 eval500 后 val 池仅剩 500 条（E-β′ 实测 collision/max_step 各 20 条 < 50）⇒ **须用与 eval500
     不相交的补充池补足**，池组成与来源写入复算报告 meta；
  5. **max_step 接线后执行**：须先完成训练侧 max_step 接线（P4 前置-B；P2 已证未接线时"超时"被误读为正收益）
     再复算；口径断言 `terminal_key=max_step`（终局值按复算档取值，当前代码默认基准档 −23）；
  6. **记录 HEAD commit sha + split seed**：复算报告 meta 必含（工具已支持），报告入库 tracked 路径
     （`docs/reward_audit/`；原始 `runs/` 路径 + sha256 对照见其 `MANIFEST.md`）。
- **max_step 裁定 + horizon 对齐（Orchestrator，2026-10-01；E-β″ 复算定稿后；Gate4 修正版）**：
  - **max_step 终局值采用 E-β″ 点估计**：rc=1 **−46** / rc=3 −48 / rc=10 −54 / rc=30 −71
    （落点 `docs/reward_audit/ebeta2/config_draft_rc*.yaml`）。依据：profile 一致性——rc=1 审计半区
    mean **−15.050**、Δ **−5.050**（边界；方向 B mean −4.750、Δ +5.250）；若沿用 P2 −23 则
    mean ≈ **+8.05**、Δ +18（= 超时正收益）。标注 **n=8/向、不判通过**（第 4 条）、
    **首臂后复核**（见下）、**不扩采**（~5-6 h 超预算）。
  - **horizon 单位更正（Gate4）**：审计 `--max-steps 1000` = 1000 物理步（dt=0.1 s）= 100 s
    = **200 策略步**（策略步 dt=0.5 s）；训练 `LocalEnvPool.max_episode_steps=600` = **600 策略步
    = 300 s = 审计/评测的 3×**（审计、训练两侧此前均误按"1000 步 vs 600 步"读，单位混淆）。
    Gate4 实测 16 条 max_step episode：dense 均值 **+33.6**（0.168/策略步、**速率不衰减**、
    均速比 0.434 = 巡航型 loiterer）⇒ **旧 600 步口径** dense ≈ +100（3×）→ 超时
    total ≈ **+54** ≈ 到达（"超时正收益"在训练口径重现，−46 被稀释）。
  - **horizon 对齐裁定**：**Stage C 训练截断对齐 200 策略步（=100 s，与审计/评测一致）**；
    −46 在 200 步口径下即精确校准（审计 rc=1：dense +33.920 / terminating −2.970 / terminal
    −46 → mean −15.050，Δ −5.050，边界 n=8）。实现（最简净路径）：`config/train.yaml::
    stages.C.max_episode_steps=200`（代码默认同值）+ CLI `--max-episode-steps` 覆盖 +
    `metrics.json` 记录（`max_episode_steps`/`horizon_s`/`pool.max_episode_steps`/`rollout_steps`）+
    单测 `tests/test_stage_c_horizon.py`；审计/评测侧不变。**若将来改 `LocalEnvPool` 侧默认值，
    必须先核查所有调用方**（本修订不动 `LocalEnvPool` 代码默认 600）。
  - **首臂后复核（arm0 训练完成后、开 arm1 前）**：driver 自动读训练监视（legacy tags 逐 update
    序列），人工按下表处置——这是 horizon 裁定的运行时验证：
    | 读数 | 期望/阈值 | 处置 |
    |---|---|---|
    | `episodes/termination_counts/max_step`（n） | n ≥ 2 | n=0/1 → 训练窗口证据不足 → **300 s 探针**（100–200 episode；`tools/test.py --max-steps 3000`）后再裁 |
    | `episodes/mean_return_by_reason/max_step`（R_max，按 count 加权） | **R_max ≤ 0 且 R_max ≤ 0.5 × `mean_return_by_reason/arrive_dest`** | **R_max > 0 或 > 0.5×arrive → 停批重裁**（horizon/终局值/奖励复审；不得继续 arm1） |
    | `episodes/mean_steps_by_reason/max_step`（S_max） | ≈ 200 | 明显 ≠200 → 停批排查截断接线 |
    | `train/reward/terminal`（终局结算均值） | 显著为负（−46 档生效） | 若 ≈0/为正 → 停批排查 terminal 键 |
  - **非 max_step 类两向一致性（补冻结）**：E-β″ 复算 A/B 互换两向，**非 max_step 类各档
    |Δ| ≤ 5 视为结论一致**（E-β″ 实测最大 ≈2.2）；max_step 类因 n<50 只报点估计、不判通过
    （结构性不足，见 `docs/reward_audit/ebeta2/MANIFEST.md` §已知偏差）。
- **顺序（每臂单变量，前臂通过再开下臂）**：
  1. **bundle 底座臂**（v5 奖励默认全量）；
  2. **rc 扫档 3 / 10 / 30**（固定剖面、终局值为因变量）；
  3. **λ 0.95 vs 0.98**（细则 7.2）；
  4. **`ttc` 臂**（**前置**：离线证伪，would-be 触发率 ≈ 0 则改项或不做——[`docs/rl_reward_v5.md`](rl_reward_v5.md) §5）；
  5. **`lane_boundary` 臂**；
  6. **`lane_center` 臂**（两文档口径一致：`docs/rl_reward_v5.md` §5 与本表同为 −0.1 / deadband 0.25 m）。
- **预算实测（Gate4 修正）**：训练 u200/H256（`--updates 200 --rollout-steps 256`）实测 **≈19 min/臂**
  （v6 架构；旧外推 8 min 为 v4 旧口径）；加 u50 sub150 闸（≈1–2 min）+ keep-best（8 候选 sub150
  ≈12 min + top1–2 全量 clean500 复评 ≈5 min/候选）+ 终评 eval500 ≈3–8 min ⇒ **≈45–55 min/臂；
  8 臂 ≈3.7–6.7 h > 4 h**。**截断优先级（冻结）**：**arm0–4 必跑**（bundle + rc 扫档 3/10/30 + λ，
  ≈4 h 内）；**arm5–7（ttc / lane_boundary / lane_center）预算允许才后置**。driver 以 u1–u5
  实跑外推 u200 成本，并在批级预算门（≤4 h）处截断。
- **止损（Gate4 冻结）**：**u50 子集闸** = clean500 前 150 行（sub150；零点 64/150=0.4267）配对
  **net < −20 或 offΔ ≥ +0.10 → 判 early-collapse**：跳过 keep-best 与终评、保留现场、继续下一臂。
  arm0 另加**首臂后复核**（max_step 裁定块）。末段崩解候选不采纳（keep-best 协议，§5）。

### 7.2 λ 0.95 vs 0.98 对照（细则）

- GAE λ：现行默认 **0.95**（`pipeline/trainer.py` gamma 0.99 / lam 0.95）vs **0.98** 对照臂。
- 单变量：其余 pins 完全一致；GPU 串行（臂间可比）；记录 returns / EV / 崩解窗口。

### 7.3 P4 终评判据 + keep-best 候选规则（Gate4 冻结）

- **终评判据（全量 clean500 逐 (id,seed) 配对 vs E-β″ E3 零点；`net = fixed − broken`、
  `z = |net| / √(fixed+broken)`）**：
  - **net ≥ −10 = 不损伤**（可作"无损伤"结论；net ≥ 0 方可作"采纳"）；
  - **net ≥ +20 且 z ≥ 1.96 = 正向**（增益结论）；
  - **net < 0 = 不采纳**（−10 ≤ net < 0 = 不损伤但不采纳，记负向边缘）。
  - eval500 用同阈值为**辅**（辅助一致性）；两集冲突（如 clean500 正向、eval500 < −10）记偏差、
    不判正向。
- **零点 pin（不得混用 E-β′；Gate4）**：clean500 `runs/BTC20261001-195447_v6p3_e3_clean500`
  （**0.440**；`episodes.csv` sha256 `d63db75e…`）/ eval500 `runs/BTC20261001-200331_v6p3_e3_eval500`
  （**0.440**；`episodes.csv` sha256 `3d949398…`）；训练 init = `runs/BTC20261001-1631_v6p3/
  stage_b/ckpt_epoch005.pt`（sha256 `723db1c2…`）。
- **keep-best 候选规则（禁止 sub150 直采）**：臂内候选 = 周期 ckpt `ckpt_u025…u200`
  （`--ckpt-every 25`，**8 个**）→ **sub150 速评**（sub150 = E-β″ clean500 `episodes.csv` 前 150 行；
  spec `/tmp/opencode/phase3_diag/exp/specs_val_only150.json`，canonical sha `6adbe0af…`；
  零点 **64/150 = 0.4267**）→ 按子集配对 net 取 **top1–2** → **全量 clean500 复评** → 以**全量配对
  net** 采纳最高者（并列取 z 高者）。**禁止 sub150 直采**（子集最优点对全量系统性偏乐观 2–4×，§5）。
  采纳候选的全量 clean500 复评 = 该臂终评主读数（不另跑一次）；eval500 对采纳 ckpt 跑一次（辅）。

### 7.4 P4 driver pin 表（Gate4 冻结；`/tmp/opencode/v6_p4_driver.py`，执行脚本不入 repo）

| pin | 值（逐字） |
|---|---|
| `--config` | `config/arms/arm{0..7}_*.yaml`（rc 档 / λ / 追加项**仅由臂文件承载**） |
| `--spec` | `env/specs/scenarios_train_dagger_r1.json`（500 条；raw sha256 `d204e803…`） |
| `--ckpt` | `runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt`（sha256 `723db1c2…`；启动断言） |
| `--pool` / `--envs` | `local` / `1`（tracker lqr） |
| `--trainable-scope` / `--plan-reference` / `--adv-norm` | `design` / `repeat_action` / `global` |
| `--updates` / `--rollout-steps` | `200` / `256` |
| `--ppo-epochs` / `--minibatch-size` / `--seed` | `2` / `1024` / `0` |
| `--critic-warmup-updates` | `0` |
| `--kl-anchor-coef` / `--kl-anchor-final-coef` | `0.05` / `0.0` |
| `--max-episode-steps` | `200`（horizon 对齐；收尾回读 `metrics.json` 断言） |
| `--ckpt-every` / `--probe-interval` | `25` / `25` |
| `--device` / 记录 | `cuda`；`--monitor --monitor-legacy-tags`；每臂独立 `<out>` + log |
| **禁用** | `--lam`、`--reward-term-weight route_completion=…`（rc/λ 由臂文件承载；driver 启动断言拒绝） |
| 零点/评测 pin | §7.3；clean500/sub150 spec 在 `/tmp` 易失 → driver 启动按 **sha256（raw + canonical）断言** |

- **fail-closed 断言（driver 启动 + 收尾）**：启动——ckpt / spec / 零点 `episodes.csv` / spec 文件
  sha256 逐项匹配，`specs=500`，臂 reward 权重与终局值逐档配对（`build_reward_adapter` 通过），
  命令不含禁用 pin；收尾——`metrics.json` 回读 `reward_config` 逐键 = 臂配置、`specs=500`、
  `rollout_steps=256`、`max_episode_steps=200`、`pool.kind=LocalEnvPool`，加**逐臂"extra 项均值 ≠ 0、
  terminal 档位值生效"**断言。任一断言失败 → 停臂保留现场（不得继续下一臂）。

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

## 11. 修订锚（Gate0 remediation / Gate1 补证-A）

- 三份冻结文档（本文件 + `docs/v6_net_design.md` + `docs/rl_reward_v5.md`）的 Gate0 修正版 commit：`d27395dd0e1590d2151c42830f6a930bb5aafd2c`（短 `d27395d`，2026-10-01，首次 commit）。本锚行由第二次小 commit 写入（不改动其余内容）。
- Gate1 补证-A（P3 数据窗口 v3 补采 + mini 闸 A4 断言 + abort/预算校准）内容 commit：`0863186`（2026-10-01）。本锚行由第二次小 commit 写入（不改动其余内容）。
- Gate2 P4 前置-A（审计工具 exclude 默认/分层随机/HEAD+seed 记录 + 产物入库 + v5 池重叠声明 + §7.1 E-β″ 复算规格）内容 commit：`0120ac0`（工具/单测）、`b0fa3b1`（文档/产物，2026-10-01）。本锚行由第二次小 commit 写入（不改动其余内容）。

- Gate2 P4 收尾（max_step 裁定 + horizon 显式接受 + P4 臂配置组装 `config/arms/` + dry-run 单测）内容 commit：`920395e`（2026-10-01）。本锚行由第二次小 commit 写入（不改动其余内容）。
