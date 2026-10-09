# v6 程序收尾报告（Net 重构 → E-β″ 重训 → 奖励 v5 → P4 RL 臂批）

> 生成：2026-10-02（v6 程序收尾；写作时 HEAD `5a26411`）。上游：[`v6_program_prereg.md`](v6_program_prereg.md)（§0–§11）、[`v6_net_design.md`](v6_net_design.md)、[`rl_reward_v5.md`](rl_reward_v5.md)；执行档 `.slim/deepwork/v6-net-retrain.md`。
>
> **一句话结论：E-β″ 与 E-β′ 等价（±2pt 带内、逐 (id,seed) 配对 z≈0）；P4 arm0 单 seed clean500 +99 / eval500 +91，但 seed=11 复现失败（clean500 −59，z=5.00），2-seed 均值 ≈+20、方差主导 ⇒ RL 增益不可采纳；P5 阻断；碰撞抑制臂已立项（§7.5 / arm8）未跑。**

## 0. 摘要

- **基座（P3）**：E-β″（v6 新架构）全链重训 + E3 验收判「不损伤（带偏差通过）」——clean500 0.440 vs E-β′ 0.446、eval500 0.440 vs 0.436；逐 (id,seed) 配对 net **−3 / +2 counts**（z 0.457 / 0.272）⇒ 统计等价（跨代码代配对，E-β′ 归档 episodes 逐位复核）。采纳 `ckpt_epoch005.pt`（sha `723db1c2…`）为 P4 init。
- **P4 臂批（6/8 臂，预算门截断于 arm6）**：arm0 bundle（rc1）单 seed **terminal:positive**——clean500 **net +99**（z=9.08，0.440→0.638）/ eval500 **net +91**（z=8.64，0.440→0.622）；arm5（ttc）同向 **+77 / +59**；arm1–4 按 u50 闸 early-collapse（其中 3/4 为假阳性，§2.3）。
- **复现（§9.1 三分法）**：arm0 seed=11 u50 闸触发（offΔ +0.14），u200 clean500 **net −59**（z=5.00，0.440→0.322；fixed/broken=40/99）⇒ **失败分支**；seed0 vs seed11 终局差 **158 counts = 批内混沌标尺（22–32）的 5–7×**；2-seed clean500 均值 ≈ **+20**，方差主导。
- **归因**：轨迹混沌 + 亚稳吸引子分叉——u1 起权重即分叉、u46–62 首处行为分离、**u101–105 起持续分叉（与 KL 锚衰减窗重合）**、u150 锁定；无 NaN / 通道爆炸 / 熵塌 / 基线漂移。单 seed +99/+91 为高侧抽样，不可外推。
- **处置**：RL 增益降级为「单 seed 方向性」；**P5（扩量）阻断**；碰撞抑制臂（arm8 候选 A/A′/B + 辅助碰撞闸）已立项并预注册（§7.5），开跑前置 = seed=11 闭环完成（已满足）且排在 P5 之前。
- **资产/tag**：ckpt/run/配置/审计清单见 §5；版本 tag `v6-p4-closed-20261002`（§7）。

## 1. 阶段表（P0–P4）与各门结论

| 阶段 | 内容 | 门结论 | 处置 / 关键证据 |
|---|---|---|---|
| **P0 前置** | 保守清理（≈4.3G）+ 三份设计冻结文档 + Stage A/B 吞吐/成本测量 | **Gate0：需修正 → 已修** | 6 处规格缺陷 + 奖励侧 + 预注册侧共 17 条修正；文档冻结 commit `d27395d`（+锚 `031cc1c`）；修正后直接进 P1、无需重审 |
| **P1 Net 实现** | A1 交叉注意力头（K=1）/ A2 去池化 / A3 t0 st_gnn / A4 nav + E1（router 监控修复 + 专家范数探针） | **Gate1：需补证 → 已补** | 3 项必补：v3 数据补采（`datasets/BTC20261001-1327_expert5k` / `datasets/BTC20261001-1327_expert500val`）、A4 生效口径吞吐 **102.6ms = +20.9%**（abort 放宽 +30%）、均值令牌消融对照；`0863186` / `f66c866` / `e8fa69f`；527 passed |
| **P2 奖励 v5** | `low_speed`、`speed_ratio` 0.4、rc 扫档 3/10/30、终局值重解、审计工具 | **Gate2：通过**（附 P4 前置 MUST） | 540 passed 独立复核；四档反解 |Δ|max ≤5（3.66/3.57/2.76/1.88）；关键发现 max_step 未接线（超时被误读为正收益）→ fix-14 `fb4f8c1`（563 passed） |
| **P3 重训** | E-β″ 同链重训（A20 micro256 → B20 micro512 → phase3 r5 dagger + expert5k 锚 mild=1.0）+ mini 闸 + E3 + keep-best | **Gate3：通过（带偏差）** | E3 点估计 clean500 **0.440 < 0.446**（−3 条/500，记偏差）；eval500 0.440 ≥ 0.436；配对 net ≈0 ⇒ 判「不损伤（带偏差通过）」；不迭代、不回退；接受 E-β″ 为 P4 init |
| **P4 RL 臂批** | 8 臂（bundle / rc3/10/30 / λ0.98 / ttc / lane_boundary / lane_center）+ 终评 + seed 复现 + u200 复检 | **Gate4：通过（结论修正）** | 初评「需修正」→ 修正集 `0c725db`（horizon 对齐 200 策略步等）→ 开臂 → 终审「需补证」→ 闭环：seed11 失败归因、u50 闸假阳性 3/4、碰撞臂立项 `8b95be5`；**结论修正：RL 增益降级为单 seed 方向性** |

### 1.1 各门要点（详）

- **Gate0（需修正）**：方向通过；缺陷 = A1 去池化去向未定义、H=128 落点写错、K=8/q0 死参数、分布 API 矛盾、A3 语义未收敛、A4 依赖面低估；奖励侧反解输入无归档（终局值标"临时值"，P2 重解）；预注册缺 phase3 配方链 / mini 闸 / abort / P4 细节。修正后 17 条全部落文。
- **Gate1（需补证）**：实现逐项核对通过、无需改码；A1/A3 衔接接受不改（rollout 语义原样，副作用 = t0 MP `no_grad` 弱化 OD/LD 通路，mini 重训监控）。补证：v3 数据补采 + A4 生效吞吐 + 消融基线；dagger 池忠实重采不可行 ⇒ 方案 A（显式接受 v2 + preflight 断言）。
- **Gate2（通过）**：四档反解复算、分区互斥（501/499）、剖面统计复核；max_step 接线列为 P4 开工前置（后由 fix-14 完成）；另有 6 项发现（审计池含 eval500、采集帧错位、分区不可交换、CLI 权重覆盖不带终局值、horizon 一致性、折扣口径）在 P4 前逐条闭环。
- **Gate3（带偏差通过）**：5 条偏差记录——点估计缺口 / CI 宽 ±2.5–3.3pt（"无显著退化"≠"证明等价"）/ 跨代不可量化 / collision +2pt 同向（不显著）/ 本判定非先例。残余风险：E-β′ 无法在新代码复评（旧 ckpt missing=20 → succ 0.000），硬闸为预注册的跨代比较口径。
- **Gate4（通过，附结论修正）**：初评需修正 3 项（horizon 单位错误 → 训练截断对齐 200 策略步；预注册补丁；driver pin 表）→ 修正集 `0c725db` 后开臂；终审可信性高（配对逐位复算一致、训练池 vs eval500 交集 0、eval500 未参与选择），但复现判据缺口 → 揭盲前冻结 §9.1 三分法；最终闭环通过，**结论修正为**：RL 增益 = 单 seed 方向性（2-seed 均值 +20、方差主导），不可采纳。

## 2. 关键数字表

### 2.1 基座：E-β′ vs E-β″（等价性）

| 项 | E-β′（旧架构，冻结零点） | E-β″（v6 新架构） | 配对（E-β″ vs E-β′，逐 (id,seed)） |
|---|---|---|---|
| clean500 succ | **0.446**（`runs/BTC20260929-224519_eval500_clean_ebeta`） | **0.440**（同权重三次读数 0.438/0.438/0.440） | net **−3 counts**（z 0.457）；offΔ +0.022 / collΔ +0.020 |
| eval500 succ | **0.436**（`runs/_refs_rlbase/e_beta_prime/eval_episodes.csv`） | **0.440** | net **+2 counts**（z 0.272）；offΔ −0.008 / collΔ +0.022 |
| 采纳 ckpt | `runs/_refs_rlbase/e_beta_prime/final.pt`（sha `c9e2d31e…`） | `runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt`（sha `723db1c2…`） | 跨代码代配对（零点 episodes 归档；E-β″ 载入 missing=0） |

> 备注：E-β″ E3 硬闸 **clean500 fail-closed 判 FAIL**（0.440 < 0.446，−0.6pt）、**eval500 PASS**（0.440 ≥ 0.436）；按 §6 统计口径（±2pt 带内不作胜负、配对为主）+ Gate3 裁决判「不损伤（带偏差通过）」。phase3 链增益 +0.118（B final 0.322 → E-β″ 0.440）。

### 2.2 P4 臂总表（arm0–arm7）

> u50 = sub150（n=150，零点 64/150=0.4267）；终评 = 全量 clean500 逐 (id,seed) 配对；`—` = 未执行。

| 臂（变量） | verdict | u50 闸读数（net/z/offΔ） | 终评 clean500 | eval500 | 采纳 ckpt |
|---|---|---|---|---|---|
| **arm0 · bundle rc1** | **terminal:positive** | pass（−6 / 1.9 / +0.013） | **+99**（z 9.08；0.440→0.638；offΔ −0.330 / collΔ +0.108） | **+91**（z 8.64；0.440→0.622） | u200 sha `4ed0d90e…` |
| arm1 · rc3 | early-collapse | TRIP（−24 / 4.38 / +0.173） | —（闸跳过） | — | — |
| arm2 · rc10 | early-collapse | TRIP（−20 / 4.08 / +0.127） | — | — | — |
| arm3 · rc30 | early-collapse | TRIP（−26 / 5.10 / +0.087） | — | — | — |
| arm4 · λ0.98 | early-collapse | TRIP（−64 / 8.00 / −0.073；succ→0.0） | — | — | — |
| **arm5 · ttc** | **terminal:positive** | pass（−6 / 1.9 / +0.013） | **+77**（z 7.59；0.440→0.594；offΔ −0.274 / collΔ +0.122） | **+59**（z 5.93；0.440→0.558） | u200 sha `a72a31dc…` |
| arm6 · lane_boundary | NOT RUN | — | — | — | —（预算门截断） |
| arm7 · lane_center | NOT RUN | — | — | — | —（批止于 arm6） |

- 预算：主批 elapsed **14159.6 s / 14400 s**（3.93h），arm6 启动前预算门截断；逐臂墙钟 arm0 4928.5s / arm1 1066.9 / arm2 1027.3 / arm3 1081.0 / arm4 1303.4 / arm5 4752.4。
- arm5 视作 bundle 第二轨迹（ttc 近零激活 `mean_last20≈−0.001/update`，且与 arm0 共享 u≤50 权重，`ckpt_u025/u050` sha256 逐位相同）⇒ §9.1 豁免独立 seed=11。
- arm1–4 的 u50 读数与 u200 复检见 §2.3；arm6/7 显式降级（§9.1）。

### 2.3 u50 闸 vs u200 复检（假阳性 3/4）

| 臂 | u50 net / z / offΔ | u200 复检 net / z / offΔ | 判定（post-hoc） |
|---|---|---|---|
| arm1 · rc3 | −24 / 4.38 / +0.173 | **+32 / 4.82 / −0.327** | **u50 假阳性**（u200 已恢复） |
| arm2 · rc10 | −20 / 4.08 / +0.127 | **+30 / 4.74 / −0.313** | **u50 假阳性** |
| arm3 · rc30 | −26 / 5.10 / +0.087 | **−29 / 5.21 / +0.173** | **真崩**（维持 u50 止损） |
| arm4 · λ0.98 | −64 / 8.00 / −0.073（succ 0.4267→0.0） | **+3 / 0.58 / −0.180** | u50 崩解形态为爬行/超时吸引子（置信度高）；u200 仅回到不损伤区（z 低、offΔ 负），不作正向 |

- 复检为 **post-hoc 探索性标注**（每臂单点 u200 / 单 seed / 子集），不构成新预注册结论；keep-best 协议下不得直接表述为「可行/正向」（禁 sub150 直采）。
- 含义：u50 单点闸假阳性 3/4 ⇒ §7.5 已修订止损为 **u50+u100 双点闸**（两点均触发才停）或 u200 sub150 复检 + 全量复评；rc3/10 的 u200 collΔ（+0.087/+0.073）低于 arm0（+0.113），但 rc 与碰撞抑制不混臂。

### 2.4 seed=11 复现（arm0，§9.1）

| 项 | seed=0（`BTC20261002-0026_p4_arm0`） | seed=11（`BTC20261002-0732_p4_arm0_seed11`） |
|---|---|---|
| 训练 wall / rc / 断言 | 932.2 s / 0 / ok | 936.6 s / 0 / ok（唯一变量 `--seed 11`） |
| u50 闸 | pass（net −6，offΔ +0.013） | **TRIP**（net −16，z 3.41，offΔ **+0.140**） |
| u200 clean500（正式判据） | **+99**（z 9.08；0.440→0.638） | **−59**（z 5.00；0.440→0.322；fixed/broken=40/99；offΔ −0.028 / collΔ +0.014） |
| sub150 代理 u150/u175/u200 | —（u175 +29 / u200 +30） | **−30 / −28 / −19**（同向） |
| 权重交叉距离（u200） | — | 5.43% rel L2（≈ arm0–arm5 5.30%；自身漂移 4.5%/4.1%） |
| 末段吸引子（末 25 窗） | 快 / 低熵（entropy 0.632；EV +0.181） | 慢 / 高熵（entropy 1.369；EV −0.016） |
| §9.1 判定 | terminal:positive | **失败（< −10）⇒ 停线归因** |

- 终局差 **158 counts**（+99 vs −59）= 批内混沌标尺（22–32）的 5–7×；2-seed clean500 均值 ≈ **+20**，方差主导。
- seed11 u175 clean500 的 post-hoc 复评写作时进行中（快照原 `/tmp/opencode/v6_p4_seed11_posthoc_status.txt`，已清理）；不改变失败分支与归因结论。

### 2.5 碰撞代价（arm0 行为构成；§7.5 立项依据）

| 集 | 成功增益 | 碰撞率 | 净多撞 | 救援而撞（base OOR） | 原成功现撞（base arrive） | 交换率 |
|---|---|---|---|---|---|---|
| clean500 | +19.8pt（0.440→0.638） | 6.0% → **16.8%**（+10.8pt） | +54 | 54 | 10 | 99/54 = **1.83** |
| eval500 | +18.2pt（0.440→0.622） | 5.0% → **15.4%**（+10.4pt） | +52 | 55 | 9 | 91/52 = **1.75** |

- 失败经济学（E-β″ rc=1 审计，`docs/reward_audit/ebeta2/`）：arrive **+49.652**（n=564）/ collision **−21.395**（n=73）/ OOR **−14.995**（n=606）/ max_step **−15.050**（n=8）；碰撞 vs 出界分离仅 **6.4**；救援盈亏平衡成功率 `p*`：9.0%（collision −22）→ **20.2%**（−32）→ **32.0%**（−46）。
- arm8 候选（已预注册、未跑）：A = collision −22→−32（首选）；A′ = −46（条件升级）；B = `lead_gap` dense（−1.0 / 6.0m，后备）；辅助碰撞闸 = clean500 `collΔ ≤ +3pt 或 碰撞率 ≤ 10%`。
- arm0 终评为 terminal:positive 但**本批无碰撞率闸** ⇒ Gate4 建议碰撞抑制臂 + 辅助闸作为 P5 前置（已立项）。

## 3. 归因与教训

### 3.1 seed 分叉归因：轨迹混沌 + 亚稳吸引子分叉

- **时间线**（数值见 `docs/v6_reports/v6_seed11_attribution.md`）：权重自 u1 即分叉（单 seed 改变 rollout 采样；u25 交叉距离 1.24% rel L2）；第一处显著行为分离 **u46–62**（确定性探针 ds@2–4 最大差 1.43 m/s，u50 闸窗正落其中）；u65–100 部分回敛；**持久分叉自 u101–105**（ds@4–8 / ds@8+ / ds_std / ds mean 依次分离且不再回敛）；logstd u135、returns/EV u140–172 跟进；**终局 u150 前后锁定**（post-hoc u150 sub150 已 −30）。
- **排除系统性不稳定**：无 NaN/Inf；grad norm 峰值相当（142.6 vs 121.1）；approx_kl 除 u1 外 <0.05；value_loss / advantage / router 无爆炸或漂移；权重交叉距离近似线性增长（u200 5.43%），与各 run 自身漂移同量级，分歧方向 cos 仅 0.43（共享梯度漂移 + 独立混沌分量）——**不是"某通道爆炸 / 熵塌 / 基线漂移"型故障**。
- **吸引子证据**：两 run 落入定性不同的晚段吸引子——seed0 快 / 低熵 / value EV↑；seed11 慢 / 高熵 / EV≈0。arm5 在 u100–140 与 seed11 同向（4–8 档下行、entropy 上行）但 u150 后部分恢复并终局正向 ⇒ 系统存在**放大窗口**且至少两个晚段吸引子（"快/锐化"与"慢/探索"）；arm0 与 arm5 落前者、seed11 落后者。
- **KL 锚衰减窗重合**：u100 时锚系数 ≈0.025、u150 ≈0.013；三条轨迹都在锚弱化后分叉 ⇒ v7 方差控制首选「KL 锚末值 > 0（0.01–0.02）或更慢衰减」。
- **定量含义**：单 seed 净效应精度 ±20–30 counts 仅适用于**同吸引子内**扰动；跨吸引子分叉可达 100+ counts（本例 158）。arm0 +99/+91 = 单 seed 高侧抽样，不可外推。

### 3.2 教训（已入 v7 档）

1. **多 seed 强制**：每臂 n≥3–5（至少 0/11/23），报分布（min/median/max）与分层；单 seed 不判通过/失败以外的结论。
2. **方差控制优先**：KL 锚末值 > 0 或更慢衰减；增大 rollout / 等效 batch（当前 1 env × 256 steps/update，梯度噪声大）。
3. **评测协议**：u50 单点闸假阳性 3/4 ⇒ **u50+u100 双点闸**（均触发才停）或 u200 复检 + 全量复评；保留 keep-best；晚段 EV/entropy 监控（诊断用，非闸）。
4. **选点纪律**：禁止 sub150 直采（子集最优点对全量系统性偏乐观 2–4×）；采纳须全量 clean500 配对 + seed 复现。
5. **预注册精度假设修正**：功效分析按最坏情况（bimodal）设计；臂间比较同 seed 不保证可比（arm0 vs arm5 同 seed 仍分叉）⇒ 多种子平均或配对评估 + 置信区间。
6. **保留现场策略正确**：每 25 update ckpt 足以事后吸引子归因；后续臂沿用。

## 4. 事故与修复记录

| # | 事件 | 根因 | 修复 | 影响/旁证 |
|---|---|---|---|---|
| ① | **P4 首跑 arm0 首臂后复核误停**（2026-10-02 00:04:55–00:20:23，927.8 s；arm1–7 未启动、无 GPU 消耗） | driver `first_arm_review()` 按**无前缀** tag 读 monitor；实际展平加 `train/` 前缀（`pipeline/trainer.py:59`）→ n=0 误判 `needs_probe300` | 最小补丁：三行 tag 加 `train/` 前缀（driver sha `7b9e0201…` → `32928a6c…`，不入 repo）；离线回放正确读数 **n=140 / R_max −48.5104 / S_max 200.0 / trip=false** | 00:26:33 整批重跑；首跑/重跑 arm0 `ckpt_u200.pt` sha 相同（`4ed0d90e…`）= 确定性旁证；现场保留 `runs/BTC20261002-0004_p4_arm0/` |
| ② | **horizon 单位错误**（Gate4 关键发现；终局值标定 3× 偏差风险） | 审计 `--max-steps 1000` = 1000 **物理步** = 100s = **200 策略步**；训练 `LocalEnvPool` 600 = 600 **策略步** = 300s = 3×（两侧单位混淆） | 裁定：训练截断**对齐 200 策略步**；`config/train.yaml::stages.C.max_episode_steps=200` + CLI + metrics 记录 + 6 单测（`0c725db`，586 passed） | 旧 600 步口径下 max_step dense（+0.168/步、不衰减、均速比 0.434）≈ +100 → 超时 total ≈ +54 ≈ 到达（超时正收益重现）；200 步口径下 −46 精确校准；首臂后复核实测通过（n=140、R_max −48.51、S_max 200.0） |
| ③ | **u50 单点闸假阳性 3/4**（arm1/2/4） | 单点 / 单子集止损对中途行为翻转敏感（seed11 真阳性但机制中途翻转） | §7.5 修订：u50+u100 双点闸或 u200 复检 + 全量复评；保留现场与逐臂日志 | arm1 +32 / arm2 +30 / arm4 +3（u200 复检 ≥−10）；arm3 −29 真崩维持 |
| ④ | **P3 链收尾脚本 bug** | 配对统计函数 `off_road` 布尔列解析崩溃 | 修复后收尾脚本重算 | 训练与全部评测已完成，不影响任何读数 |
| ⑤ | **规格冲突（P3）** | 任务书写"A micro512" vs 冻结 prereg "A micro256 / ≤10GiB" | 按冻结规格执行 A256/B512（12GB 卡 A512 必 OOM） | 如实记录（P3 报告 §6） |

## 5. 资产清单

### 5.1 ckpt sha256（均对盘上文件复核）

| 资产 | 路径 | sha256 |
|---|---|---|
| E-β′ 参照 | `runs/_refs_rlbase/e_beta_prime/final.pt` | `c9e2d31e4b9d4f784335ffe693b70ddecd4b0ff0e2d4fb062c82aa947f13ea87` |
| E-β″ A final | `runs/BTC20261001-1631_v6retrain/stage_a/final.pt` | `546cb62f62db34aa1b6aebcd7063dace1d7ae39176f0d860768d003cf5e765da` |
| E-β″ B final | `runs/BTC20261001-1631_v6retrain/stage_b/final.pt` | `069aa158207d05d4e83d180cd2230c39d7906e3a1418a65afd7acc74109c8234` |
| **E-β″ P3 采纳（P4 init）** | `runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt` | `723db1c27d9da33ed8da17375af718cb421058e000a54555a2e5c667f5a56fa5` |
| **arm0 采纳 u200** | `runs/BTC20261002-0026_p4_arm0/ckpt_u200.pt` | `4ed0d90e7144ad46373da03bc56841cb3ac3de05d474f598175f1dc690253902` |
| **arm5 采纳 u200** | `runs/BTC20261002-0303_p4_arm5/ckpt_u200.pt` | `a72a31dcd7d27d7753bfffc4b05a6e8c3c7fbdd32ebf0db1051f6e114a26ca41` |
| arm0 seed11 u200 | `runs/BTC20261002-0732_p4_arm0_seed11/ckpt_u200.pt` | `a0b91b4323237f6497efd1bd49051b50f8b7acdaad738ab55c5ce16f28b09406` |
| arm1–4 u200（复检） | `runs/BTC20261002-0{148,206,223,241}_p4_arm{1..4}/ckpt_u200.pt` | `d50ed7e6…` / `6f772acf…` / `37798ffe…` / `9e6ef69f…` |
| mini A/B final | `runs/BTC20261001-1619_v6mini/stage_{a,b}/final.pt` | `2ab11cde…` / `4fc54437…` |

### 5.2 runs/ 关键路径

- **P3**：`runs/BTC20261001-1619_v6mini`（mini 闸 6/6）、`runs/BTC20261001-1631_v6retrain`（A/B）、`runs/BTC20261001-1631_v6p3`（phase3 + 采纳 ckpt）。
- **E3 零点**：`runs/BTC20261001-195447_v6p3_e3_clean500`（episodes sha `d63db75e…`）、`runs/BTC20261001-200331_v6p3_e3_eval500`（sha `3d949398…`）。
- **P4 臂**：`runs/BTC20261002-0026_p4_arm0` / `-0148_p4_arm1` / `-0206_p4_arm2` / `-0223_p4_arm3` / `-0241_p4_arm4` / `-0303_p4_arm5`；首跑现场 `-0004_p4_arm0`。
- **终评**：`BTC20261002-0026_p4arm0_{u200_clean500,eval500}`、`BTC20261002-0303_p4arm5_{u200_clean500,eval500}`。
- **复现/复检**：`-0732_p4_arm0_seed11`；seed11 sub150 `-081746/-082236/-082712`；seed11 u200 clean500 `-083129_v6p4_seed11_u200_clean500`；seed11 u175 clean500 `-084501_…`（写作时进行中）；u200 复检 `-075202/-075618/-080015/-080336`；u50 闸 `BTC20261002-00*_p4arm*_u050_sub150`。

### 5.3 配置与审计

- `config/arms/`：`arm0_bundle_rc1` / `arm1_rc3` / `arm2_rc10` / `arm3_rc30` / `arm4_lam098` / `arm5_ttc` / `arm6_lane_boundary` / `arm7_lane_center` / `arm8_collision_suppress_{term,term46,gap}` + `README.md`。
- `docs/reward_audit/ebeta2/`：`MANIFEST.md`、`reward_audit.{md,json}`、`reward_audit_swap.json`、`ab_comparison.{md,json}`、`config_draft_rc{1,3,10,30}.yaml`（E-β″ 定稿：rc=1 +29/−22/−14/−46/−5；rc=3 +27/−23/−15/−48；rc=10 +20/−28/−18/−54；rc=30 +0/−41/−26/−71）。

### 5.4 证据/报告档（已入仓 `docs/v6_reports/`；索引/sha256 见 `docs/v6_reports/README.md`）

- 执行报告（入仓）：`docs/v6_reports/v6_p3_report.md`、`docs/v6_reports/v6_p4_report.md`、`docs/v6_reports/v6_p4_incident.md`、`docs/v6_reports/v6_p4_recheck.md`、`docs/v6_reports/v6_seed11_attribution.md`、`docs/v6_reports/v6_collision_arm_design.md`；`v6_p4_results.{md,json}`、`v6_p4_seed11_report.md`、`v6_p4_seed11_posthoc.{json,md,status}`（未入仓，已随 2026-10-06 `/tmp` 清理删除）。
- 门/专项：`docs/v6_reports/v6_cleanup_report.md`（入仓）；`v6_gate4_fixes.md`、`v6_p1_validation.md`、`v6_reward_audit.md`、`v6_reward_audit_ebeta2.md`、`v6_ttc_falsification.md`、`v6_p3_data.md`、`v6_stageA_cost.md`（未入仓，已清理；部分内容由 `docs/reward_audit/`、`docs/v6_net_design.md` 承接）。
- 驱动/脚本（不入 repo；已随 2026-10-06 清理删除）：`v6_p4_driver.py`（补丁后 sha `32928a6c…`）、`v6_p4_seed11.py`、`v6_p4_recheck.py`、`v6_collision_diag.py` 等。

### 5.5 文档（repo tracked）

`docs/v6_program_prereg.md`（§0–§11）、`docs/v6_net_design.md`、`docs/rl_reward_v5.md`、`docs/version_ledger.md`、`.slim/deepwork/v6-net-retrain.md`、本报告。

## 6. 现场与复现

- **本报告 §2 关键数字已由 fixer 从盘上 `episodes.csv` / `metrics.json` 独立复算**（脚本口径 `net=fixed−broken`、`z=|net|/√(fixed+broken)`）：
  - E-β″ vs E-β′ 配对（−3/+2）；arm0/arm5 终评（+99/+91、+77/+59）；seed11 u200 clean500（−59，fixed/broken=40/99）；u50 全臂（−6/−24/−20/−26/−64/−6、seed11 −16）；u200 复检（+32/+30/−29/+3）；E-β″ E3 零点 0.440/0.440、E-β′ 零点 0.446/0.436 逐位复核。
  - collΔ/offΔ 独立复算与驱动 JSON 一致（舍入差 ≤0.0001）。
- **复算入口**：`runs/*/episodes.csv`（逐 (id,seed) 配对）；`runs/*/metrics.json`（overall）；`sha256sum` 核对 §5.1；seed11 归因复算命令见 `docs/v6_reports/v6_seed11_attribution.md` §10。
- **未跑/未决（如实）**：arm6/7 未跑（预算门）；arm8 碰撞抑制臂未跑（已立项 + 预注册）；seed11 u175 clean500 post-hoc 写作时进行中；P5 阻断；跨代比较不可消（E-β′ 无法在新代码复评，missing=20）；rc 默认档位仍为开放项（§10）。
- **tag**：`v6-p4-closed-20261002`（指向本报告入库 commit；tag 说明含 E-β″ 基线 + P4 臂批结论 + 单 seed 限定）。

## 7. 版本与 tag

- 本报告随 v6 收尾 commit 入库；`git tag -a v6-p4-closed-20261002` 指向该 commit（`git show v6-p4-closed-20261002` 查看 tag 说明）。
- 前置 commit 链（收尾时 HEAD）：`5a26411`（§11 碰撞臂锚）← `8b95be5`（§7.5 + arm8）← `9b7848f`/`c5c3654`（Gate4 终审补录）← `b657fea`/`0c725db`（Gate4 修正集）← `920395e`/`02fe171`（P4 收尾/臂配置）← `abffdf6`/`f7ce361`（E-β″ 审计）← `fb4f8c1`（max_step 接线）← `0120ac0`/`b0fa3b1`/`0b67f3d`（P4 前置-A）。
