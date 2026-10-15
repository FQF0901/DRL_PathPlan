# R31-3 Stage C 执行契约审计（collect / update / eval / deploy 四映射一致性）

> 依据：GPT R31-3（`docs/chatgpt5p6sol_review.md:381-393`：确认 collect / update / eval / deploy 四个 action mapping 是否一致；同文 `:411-418` 根因排序 P0=「训练链和部署链不一致，Stage C ≠ Deploy」）；`.slim/deepwork/p1_v2/round31/journal.md:10`（lane D 定义）；`docs/deepseekv4p1_argue.md:277`（R31 审计包开工声明）。
> 纪律：**零训练、零生产代码改动**；只读分析（grep/read + ckpt 参数只读抽查）；本文为唯一新增产物。
> 口径：全部行号在**当前 HEAD `440790b`** 逐条复核（2026-10-14）；术语 S1/S2/S3 沿用 `docs/round30_audit.md` §1（S1=Exact+mu、S2=LQR+plan、S3=LQR+repeat）。
> 边界：仓内**没有独立的部署执行器代码**；「deploy」在本审计中 = 部署协议口径 S2（六步 WM/plan rollout → LQR 预览跟踪），生产侧由 phase3 守护评测显式 pin（`pipeline/stages.py:4125-4137`），评测侧由 `--tracker lqr --eval-reference plan` 代表。

---

## 0. 结论速览

- **collect（默认）与 update 内部一致**（同一 cheap path、同一 logstd 钳制、buffer 记账自洽）——两者共同构成 **S3（repeat+LQR）** 训练闭环。
- **collect（S3）≠ deploy（S2）是结构性不一致**：R30-1 矩阵已实测严格劣（K1e2 0.52→0.02、pri512 0.36→0.00，跨策略 0 翻正；`docs/round30_audit.md:52-57,64-66`）⇒ GPT 的 P0 指控成立。
- **eval 存在两个默认口径**：官方栈 `eval_runner` 默认 `exact + plan`（`pipeline/eval_runner.py:1716,1719`），诊断栈 `eval_a3` 默认 `lqr + repeat`（`.slim/deepwork/p1_v2/a3/eval_a3.py:419-420`）；前者默认口径 **≠ deploy**（exact 仅消费 `plan[0]`）。
- **eval 独有开关**（`kappa-compose`、`--moe-off`）在采集/训练链路不存在或未对齐——评测分布与训练分布存在可量化偏差（见 I3/I4）。

---

## 1. 四链路 action mapping 主表

| 链路 | 输入状态来源 | 动作（采样/确定性） | 执行参考构造 | 执行器 | 观测与奖励/记账 |
|---|---|---|---|---|---|
| **collect**（默认，repeat） | `LocalEnvPool`（默认 `--pool local`，`stages.py:5829`）真实 env 状态；`ObservationBuilder`（`trainer.py:6872,6939-6949`） | **采样** `a_t ~ sigmoid(raw_mu+σ·ε)`：cheap path 前向（`trainer.py:7683`）→ `clamp_logstd`（`:7686`）→ `sample_action`（`:7690-7692`；分布约定 `trainer.py:8-11`） | `repeat(a_t,6)`（`:7703`）；**plan 不计算**（`need_plan=False`，`:7678`） | `LqrTracker`（`stages.py:5380` 硬编码 `tracker="lqr"`；池装配 `trainer.py:6903-6911`；每策略步 `set_reference((N,2))` `:7031-7035`） | reward=RewardAdapter（`:7739-7742`）；buffer 记 `(obs,a_t,logprob,value,reward)`（`:7761-7774,7830-7842`）；bootstrap 行（`:7794-7803`） |
| **collect**（plan 对照臂） | 同上 | 同上 | `plan` 输出且 `plan[:,0]=a_t`（`:7688-7689,7699-7701`）；尾 5 步 = WM/plan 预览（**记账外变量**，`:7696-7698` 注释） | 同 LqrTracker | 同上（plan 不进入 buffer） |
| **update**（PPO） | buffer 帧 + `build_history` 重拼 6 帧历史（`trainer.py:7879-7888,7974`；四通道口径 `:7807-7812`） | **无采样/无执行**：cheap path 前向（`:8000`）→ 同钳制（`:8013`）→ 对存储 `a_t` 重算 logprob（`:8014-8018`）；可训范围 = design allowlist（policy/value + MoE experts/residual_scale，`:1147-1152`；应用 `stages.py:5353-5354`） | 不构造参考 | 无 | 消费 buffer 的 reward/value/advantage；无 env 交互 |
| **eval**（`eval_runner`，官方栈；`tools/test.py` 透传） | env 真实状态 + `ObservationBuilder`（`eval_runner.py:964-967`） | **确定性** `action_mu`（`:1038-1048`；`build_eval_references` docstring `:889`） | `--eval-reference {plan,repeat_action}` 默认 `plan`（`:1719`；`:856-859`）：`plan` 首步强制=mu（`:903-907`）；`repeat_action`=`repeat(mu,6)`（`:897-898`，cheap path） | `--tracker {lqr,exact}` **默认 `exact`**（`:1716`）：exact→`ExactTracker`（`:975-978`）+ `arm(references)`（`:1052-1054`）+ post_step（`:1062-1065`）；lqr→`LqrTracker`（`:980-990`）+ `set_reference`（`:1055-1057`） | KPI=succ/off/coll/tmo/rc；**env reward 被丢弃**（`:1131` `_, _, terminated, truncated, info = env.step(action)`；旁录见 R30 §1.5 `round30_audit.md:89-108`） |
| **eval**（`eval_a3`，诊断栈） | 同 `eval_runner`（子类 `_CkptController`，`eval_a3.py:96`）；另有 `--oracle-action`/semantic 遥测 | 确定性 mu；`--kappa-compose` 时 **ds 用 `semantic_off` 前向替换**（`:207-218`） | `--tail {repeat,scale,chain}` 默认 `repeat`（`:419`）：repeat/scale=`tail_plan(mu)` 纯函数（`a3_tails.py:23-38`）；chain=模型 rollout 链 + `build_eval_references(mu,raw_plan,"plan")`（`:214-219`）；`"eval_reference":"plan"` **硬编码**（`:478`）；plan0 守卫（`:248-253`，计 `_plan0_mismatch` `:342`） | `--tracker {lqr,exact}` **默认 `lqr`**（`:420`）；exact 语义同 eval_runner（docstring `:8-13`） | 同 eval_runner + `trace.npz`/`dec_trace`（`:141-156`） |
| **deploy**（S2 口径） | 真实/仿真环境状态（仓内代表=评测链） | 确定性 mu（同 eval plan 臂） | 六步 WM/plan rollout（`rollout=True`）→ `plan[0]=mu`（`eval_runner.py:1042-1049` + `:903-907`） | `LqrTracker` 预览跟踪（30 点插值；`env/tracking.py:478-506,585-595`） | 部署 KPI（无 PPO 记账）；仓内 pin：phase3 守护 `--tracker lqr --eval-reference plan`（`stages.py:4134-4136`）+ clean150 success 门（`:4099-4112`） |

---

## 2. 逐链路证据（复核明细）

### 2.1 collect（`trainer.py:7660-7876`）

- 默认 `plan_reference=repeat_action`：CLI 优先，config `stages.C.plan_reference`（`stages.py:5256`；`config/train.yaml:205`）。
- cheap path 与完整前向的 `action_mu/action_logstd/value` **逐位一致**（`net/model.py:77-79,1209-1212`；测试 `tests/test_net_shapes.py:663::test_cheap_path_matches_full_forward`）。
- 执行侧只依赖记账随机变量 `a_t`：`references[:,0]=a_t` 由构造保证（repeat 即 `:7703`；plan 臂 `:7701` 显式覆写）。
- 池执行分支：lqr 消费 `references`（`trainer.py:7031-7035`）；exact 分支只 arm 单动作（`:7024-7030`）；kinematic 分支**忽略 references**（`:7159`）。`VectorPoolAdapter` 请求 `lqr` 时降级为近似运动学（`trainer.py:7123-7124`）——默认 `--pool local/--envs 1`（`stages.py:5829,5839`）不触发；实测全部 Stage C runs metrics 为 `pool.kind=LocalEnvPool` + `tracker=lqr`（§4 I1 对账）。
- 记账口径：`pending` 帧含 `action/logprob/value/reward`（`:7761-7774`）→ `buffer.add_step`（`:7830-7842`）；GAE 由 `pipeline.buffer` 计算。

### 2.2 update（`trainer.py:7986-8062`）

- 每 update 只走 cheap path（`rollout=False, world_model=False`，`:8000`）；`clamp_logstd`（`:8013`）与 collect 同函数（`:7686`）→ ratio 口径一致（S2 修复）。
- **不消费 plan、不构造参考、不调用 tracker**——PPO 梯度只作用于 policy/value（+design allowlist 的 MoE experts/residual_scale，`1147-1152`）；plan_head 主干（fusion/norm/ego_next/primary）与 st_gnn 冻结（`stages.py:5353-5354`；WM 全期冻结 `:5333-5345`）。

### 2.3 eval（两条栈）

- `eval_runner`：`--tracker` 默认 **exact**、`--eval-reference` 默认 **plan**（`eval_runner.py:1716-1721`）；`tools/test.py` 仅做转发（`tools/test.py:11-12`），不覆写默认。
- exact 执行器结构判定（R30 F1）：`ExactTracker.substeps=5`（`env/tracking.py:247`），`arm` 6 步→30 点（`:257-290`），每决策步 5 子步后 re-arm ⇒ **只有 `plan[0]` 被真正执行**，尾段丢弃（`eval_a3.py:8-13` docstring 同判；R30 `docs/round30_audit.md:42-43`）。
- `eval_a3`：`--tail repeat/scale` 走 cheap path（`eval_a3.py:195-199` `rollout=(tail=="chain")`）；`--kappa-compose`（`:443-444`）在动作（`:207-213`）与 plan（`:216-218`）的 **ds 维**替换为 `semantic_off` 前向结果；`--moe-off`（`:431-432`）。
- 两栈执行器/lqr 装配一致：`engine.add_policy(..., LqrTracker, ...)`（`eval_runner.py:980-990`）与训练池同款（`trainer.py:6903-6911`）。

### 2.4 deploy（口径 pin）

- 仓内代表：phase3 守护 argv 固定 `--tracker lqr --eval-reference plan`（`stages.py:4134-4136`，注释 `:4125` "口径与 §2.5 pin 表一致：LQR + eval-reference plan"）；评测执行语义 = eval plan 臂（§2.3）。
- 部署链的状态条件化 rollout（WM 六步）即 `rollout=True` 前向（`net/model.py:1257-1260` `_rollout` → `plan`）。

---

## 3. 一致性矩阵（两两对照）

维度：①动作分布 ②参考尾段 ③执行器 ④随机性来源 ⑤是否消费 `plan[0]` 之外的点。

| 对照对 | ①动作分布 | ②参考尾段 | ③执行器 | ④随机性 | ⑤plan 尾消费 | 判定 |
|---|---|---|---|---|---|---|
| collect(repeat) vs update | 同分布函数（cheap path 逐位一致 `net/model.py:1209-1212`；同钳制 `trainer.py:7686/8013`） | update 无参考（N/A） | update 无执行（N/A） | collect 采样 / update 对 `a_t` 重算 | 均不消费（`:7678,8000`） | **一致** |
| collect(repeat) vs eval_a3(repeat) | 采样 `a_t` vs 确定性 `mu`（标准 train/eval 差） | 同为 repeat 族（`:7703` vs `a3_tails.py:23-38`） | 同 LqrTracker | 采样 vs 确定 | 均不消费 plan | **部分一致**（kappa/MoE 开关差见 I3/I4） |
| collect(repeat) vs eval_runner 默认 | 同上 | repeat vs **plan**（`eval_runner.py:1719`） | lqr vs **exact**（`:1716`） | 采样 vs 确定 | collect 无 / exact 仅 `plan[0]`（`env/tracking.py:247`） | **不一致** |
| collect(repeat) vs deploy(lqr+plan) | 同上 | **repeat vs plan**（`trainer.py:7703` vs `eval_runner.py:903-907`） | 同 LqrTracker | 采样 vs 确定 | 无 vs 有 | **不一致（结构级，S3≠S2）** |
| update vs eval | 同 `mu` 分布函数（eval 用 mu 确定执行） | N/A | N/A | 无（update 无采样） | 默认臂均不消费（eval 仅 lqr+plan 臂消费） | **部分一致** |
| update vs deploy | 同 `mu`（update 只优化 mu 路径） | update 不消费 plan ⇒ 部署尾段=冻结 plan 输出 | N/A | — | 否 vs 是 | **部分一致** |
| eval(lqr+plan) vs deploy | 同（确定性 mu） | 同（plan 首步=mu，`eval_runner.py:903-907`） | 同 LqrTracker | 同（确定） | 同 | **一致** |
| eval_runner 默认(exact) vs deploy | 同（mu） | plan vs plan | **exact vs lqr** | 同 | **仅 `plan[0]` vs 全尾** | **不一致** |

---

## 4. 不一致清单

### I1（结构级）S3（repeat+LQR）≠ S2（plan+LQR）——训练口径 ≠ 部署口径

- 采集默认参考 = `repeat(a_t,6)`（`trainer.py:7703`）+ LqrTracker（`stages.py:5380`）= **S3**；部署口径 = 六步 plan → LQR（`stages.py:4134-4136`；`eval_runner.py:903-907`）= **S2**。
- R30-1 实测（calib50，同 ckpt/episode；`docs/round30_audit.md:50-57`）：K1e2 succ **0.520→0.020**（tmo 0.28→0.84）、pri512 **0.360→0.000**（off 1.00）；配对 0 翻正 / 25、18 翻负（`:64-66`）⇒ "repeat 参考无加速预览 ⇒ 匀速跟踪耗速 ⇒ 策略 ds 反馈收缩"的 S3 低速平衡（`:73-76`）。
- 实跑对账：本审计读取全部 Stage C `metrics.json`（**14 个 run**，`runs/BTC20261005-*`/`runs/BTC20261006-*` 族）`plan_reference` 全为 `repeat_action`、实际 `tracker`/`pool.tracker` 全为 `lqr`、`pool.kind` 全为 `LocalEnvPool` ⇒ **S3 是唯一在用的训练口径**，plan 臂仅代码对照。

### I2 exact 仅消费 `plan[0]`，且 `eval_runner` 默认就是 exact

- 结构判定：`ExactTracker.substeps=5`（`env/tracking.py:247`）→ 每决策 0.5s 消费完 30 点参考的前 5 点，下个决策 re-arm 丢弃尾段（`eval_a3.py:8-13`；R30 `round30_audit.md:42-43`）。
- 官方栈默认 `--tracker exact`（`eval_runner.py:1716`）⇒ **默认官方评测 = S1，不测部署口径**；R30 矩阵亦证实 exact×plan ≡ exact×repeat 逐位一致（`:42-43,52-57`）。

### I3 `kappa-compose` 仅存在于 eval_a3（采集/训练链路无此开关）

- 生产代码（`pipeline/`、`net/`、`env/`、`tools/`）grep `kappa_compose|kappa-compose` **零命中**；唯一实现 = `eval_a3.py:207-218,443-444`（及 R4/M2 诊断脚本）。
- K1 语义臂训练语义 = "ds_final ≡ ds_legacy，**部署组合在 eval 层**"（`.slim/deepwork/p1_v2/round28/r28_train.py:5-6`）⇒ 评测动作分布 = 训练动作分布的 eval-only 重组合。
- R31-M4 原文证据：`docs/deepseekv4p1_argue.md:247`「`kappa-compose` 目前只存在于诊断侧 `eval_a3`，采集/训练链路无此开关 ⇒ 需移植」。

### I4 MoE on/off 口径差（`--moe-off`）

- 评测侧开关：`eval_runner.py:1714-1715,941-944`（primary-only 臂）；`eval_a3.py:431-432`（help 明说"与语义臂训练 set_moe(False) 一致；**默认开启=部署口径**"）。
- 训练侧：K1 语义臂训练 `model.set_moe(enabled=False)`（`r28_train.py:216`）；而 R30 矩阵 K1e2 四格 **未传 `--moe-off`**（`.slim/deepwork/p1_v2/round30/run_r30_matrix_v2.sh:15,27-30`）⇒ 以 MoE-on 评测 MoE-off 训练的策略。
- 量化（本审计只读抽查 ckpt）：K1e2 与 M0 的 8 个 expert 末层权重非零（`Σ|W|`=1803.54；`residual_scale=1.1494`）⇒ MoE-on 会叠加非零专家残差，**不是数值 no-op**；R26 已记录同族坑（`.slim/deepwork/p1_v2/round26/r26_report.py:273`）。
- Stage C 侧无开关：`PPOTrainer` 从不调用 `set_moe`（训练侧 `set_moe` 调用仅 Stage A 强制关 `stages.py:1760`、BC/phase3 读 `config.moe_enabled` `trainer.py:4091/5752`），MoE 状态 = 构造默认 on（`net/moe.py:115`），且 design allowlist 训练 experts/residual_scale（`trainer.py:1147-1152`）——**训练开、评测可关，但无一致性守卫**。

### I5 `stages.C.tracker=mpc_lqr` 键未被读取（死配置）

- 定义：`config/train.yaml:198` `tracker: mpc_lqr`；实际：`stages.py:5380` **硬编码** `tracker="lqr"`；`run_stage_c` 从不读取该键（grep `stage_cfg.get("tracker")` 零命中）。
- 且 `mpc_lqr` 不是合法值：`LocalEnvPool` 仅接受 `exact/lqr/kinematic`（`trainer.py:6863-6864`）；`build_pool` 对非 lqr/exact 静默转 `kinematic`（`:7261`）。
- metrics 只记实际值：`pool.tracker`/`tracker` 取 `pool.tracker_kind`（`stages.py:5433,5437`）。

### I6 update 不消费 plan + design allowlist 冻结 plan 主干 ⇒ "PPO 收益"作用域不含部署参考尾段

- update 走 cheap path（`trainer.py:8000`），plan/rollout 零消费（`net/model.py:77-79,1209-1212`）；plan_head 主干不在 allowlist（`trainer.py:1147-1152`）⇒ Stage C 期间 **plan 输出（部署参考的尾 5 步）基本冻结**。
- 与 R30 结论合读：WM 固有 TF 能力 OK（0.02–0.04），部署链误差主因是链路语义（`docs/round30_audit.md:181-183`）——PPO 只能改善 `mu/value/experts`，尾段缺陷不归 PPO 管辖。

### I7（旁记）eval 丢弃 env reward，与训练 return 不可直接比

- `eval_runner._run_episode` 丢弃 `env.step` 返回的 reward（`eval_runner.py:1131`）；R30 §1.5 只能由 `eval_a3` 只读旁录补 `reward.csv`（`docs/round30_audit.md:89-108`）。评测 KPI（succ/off/coll/tmo/rc）与训练 return 属不同记账口径。

---

## 5. 影响与开放问题（PPO 收益可迁移性）

### 5.1 逐条含义

- **I1**：PPO 在 S3 闭环里收集/评估 advantage——S3 与 S2 是不同动力系统（R30 严格劣、跨策略一致）。即使训练 return 上升，也首先反映"S3 平衡态"的改善（如低速爬行稳定性），对 S2 部署的迁移性**不可先验假设**；应先做口径切换/双口径评估（R30 §6 候选裁决 2，`docs/round30_audit.md:296`）。
- **I2**：默认官方评测（exact）测的是单步运动学执行 + 尾段丢弃；任何在 exact 下可见的收益都可能不迁移到 LQR+plan，反之 deploy 回归可能在默认评测中不可见。
- **I3**：K1 类语义 ckpt 的评测数字含 eval-only 组合（ds 替换）；训练/采集分布中不存在该组合 ⇒ 语义收益归因被污染（对 Stage C PPO 而言同族：若未来对 semantic ckpt 做 PPO，训练分布与部署组合不一致）。
- **I4**：MoE on/off 是动作分布级的开关（参数级已证非 no-op）；语义臂（K1）训练/评测口径不一致使 R30 数字与"训练时策略"存在系统偏差；Stage C（MoE on）与 pri512 系评测（MoE on）一致，但无守卫防止未来错配。
- **I5**：配置给人"可切 tracker"的假象；若有人以 `stages.C.tracker=exact` 意图做 Exact 主训练，会被静默忽略（仍是 lqr）——fail-fast 建议列入后置修复批次（本轮不改代码）。
- **I6**：PPO 收益只能作用于 mu/value/experts；部署参考尾段来自冻结 plan。迁移性问题应拆成"mu 迁移"与"尾段链路语义"两部分独立验证。

### 5.2 不训练前提下可进一步验证的只读检查（2–3 项）

1. **`--moe-off` A/B（calib50，确定性、零训练）**：对 K1e2 复跑 eval_a3 repeat 格，`--moe-off` vs 默认，量化 I4 对 R30 数字的影响（只读评测，沿用 `run_r30_matrix_v2.sh:17-22` 命令形态）。
2. **ckpt 参数/前向抽查（本审计已做参数级）**：可扩展到 experts 逐专家范数与 router logits 分布，确认 MoE-on 残差的量级（只读 `torch.load`）。
3. **run 清单对账**：对所有 Stage C `metrics.json` 核对 `plan_reference/tracker/pool.kind`（本审计已做一次：全 `repeat_action`+`lqr`）；新 run 落地后重跑该 grep 即可守住 I1/I5 无回归。

---

## 6. 结论表

| 对照（四映射两两） | 一致？ | 一句话 |
|---|---|---|
| collect vs update | **是** | 同一 cheap path + 同 logstd 钳制 + buffer 记账自洽（`net/model.py:1209-1212`；`trainer.py:7686/8000/8013`） |
| collect vs eval | **部分** | `eval_a3 repeat+lqr` 与采集同构；官方栈默认 `exact+plan` 与采集不同；`kappa-compose`/MoE 开关仅 eval（I3/I4） |
| collect vs deploy | **否** | S3（repeat+LQR）≠ S2（plan+LQR）；R30 实测严格劣、0 翻正（`round30_audit.md:52-57,64-66`） |
| update vs deploy | **部分** | update 只优化 mu/value/experts 且不消费 plan；部署参考尾段 = 冻结 plan 输出（I6） |
| update vs eval | **部分** | 分布函数同源（`mu`）；执行/参考语义在 update 无对应物 |
| eval vs deploy | **部分** | 显式 `lqr+plan` 时**一致**（守护 pin `stages.py:4134-4136`）；`eval_runner` 默认 exact 时不一致（I2） |

**总判定**：四个 action mapping **不是同一系统**——collect/update 构成 S3 训练闭环；eval 有 exact（默认）与 lqr 两个口径；deploy=S2。与 GPT R31-3 的 P0 怀疑一致：**Stage C ≠ Deploy 成立且为结构级**（已由 R30-1 实测量化）；其余差异（kappa-compose、MoE、exact 默认、死键）为口径级不一致，均可在不训练前提下进一步量化。

---

## 7. 复核说明

- 行号复核：本文全部 `file:line` 于 HEAD `440790b`（2026-10-14）逐条 grep/read 复核；R30 数字引自 `docs/round30_audit.md`（`acf1b8f` 后未变更，本文未复算矩阵）。
- 只读抽查：K1e2/M0 ckpt 的 experts 末层范数与 `residual_scale` 为 `torch.load` 只读读取（不写盘、不改代码）。
- 未做：任何训练/评测复跑、任何生产代码或配置改动、任何 commit。
