# deepseek-v4.1 argue / 主文档（**R39 当期快照**）

> **用法（用户口径 2026-10-16）**：本文件按**当期快照**维护——**每开新一轮可直接整篇重写**，不必保留旧轮信息。旧轮一律走 git：`git log --oneline -- docs/deepseekv4p1_argue.md` ／ `git show <commit>:docs/deepseekv4p1_argue.md`。
> **配套**：`chatgpt5p6sol_review.md`（GPT 方案/裁决往来原文）、`repo_handbook.md`（架构/训练/门/**附录 D 归档指针**）、`README.md`（索引+卫生规则）、`architecture_contract.md`（架构契约）。
> **GPT 规格（本阶段）**：**主方案 `647f909`**（R39：统一 imagined world state；§一 第 2–6 步接口 / §二 双 Memory / §三 current+aligned delta / §四 detach 账本 / §五 参数量 / §六 learner-state transition / §七 正式方案 A–D / §八 Probes / §九 四工程缺口 / §十 验收准则）+ **补充裁决 `fe5de55`**（双账本预算、HEAD 重登记、双 Memory 仅隔离、delta 离线先行、D0–D3 与 learner-state 后置）。两者共同构成执行规格。

## 0. 目标与基线

- **总目标**：闭环 success **超越 IDM+Pure Pursuit**；最终门 = Eval500 3-seeds > IDM 且 collision/off-road 不劣。
- **三级集（禁混比）**：
  - **Calib50**（开发；train split；历史锚 legacy 0.40 / Q1 0.50 / K1 0.52 / T2 0.58 / 统一基线 0.56 / **IDM 0.756** 均 calib50 口径）；
  - **Clean500**（晋级级）= `scenarios_val.json − scenarios_eval500.json`，n=500，canonical 副本 `env/specs/scenarios_clean500.json`（sha256 `087db3f5…`）；**@HEAD 重登记（R39-A3，pin `8cc6862`，plan 协议 `c082b57d7063be20`／IDM 入口 `3b76e52f…`）**：**IDM+PP 0.742 ／ C0_HEAD 0.588 ／ C1_HEAD 0.390**（C1_HEAD ≡ pin 的 c1_true，逐位同 ⇒ Q4-A 原生 obs 语义验证；旧 pin 表仅历史）；
  - **Eval500**（正式：3 seeds + paired bootstrap/McNemar）。
- **协议与 hash**：`evaluation_protocol_hash`（v1；工具 `.slim/deepwork/p1_v2/round38/c6_protocol.py`）+ **model/schema/ckpt 四 hash**；正式重测必带四 hash + git commit。
- **口径更新（Q4-A 起）**：env builder **原生供真值**（dynamic_ttc/dyn/static + present）⇒ **`C1_HEAD`（结构化+旧 WM+原生 obs）≠ 旧 C1（pin，三键=0）**；旧 C1=0.304 仅历史；`c1_fill` 降缺键兼容诊断。
- **参数基线（冻结）**：structured HEAD **total=1,510,570** / 激活 ≈1.0–1.1M；R39 期间 **total ≤+2%、active ≤+5%**；**取消 1.25× 扩容臂**。

## 1. 根因裁决（R38 结论；仍为当前最优解释）

1. **主瓶颈 = 第 2–6 步 imagined state 接口**（speed/dyn/static/TTC/nav 被 hold 或简化更新）；链上=LQR 参考速度（=plan 弦长/dt）⇒ 保速自锁、爬行（slow+accel Σds −24.8%）。
2. 归因层级：**接口 ＞ WM latent 漂移（真值替换零恢复）＞ 头部/监督语义（三阶）**；解析式修复已穷尽（A3 11 臂 FAIL：`S_self` 正反馈发散 5.77→6.33）。
3. 零训练通道总账：仅"接口修复（`--long-exog-fill`）"与"**Q4-A 原生生产者**"有效；F3/A 族/`path_*` 真值在爬行下/C2 warp（~中性）/Memory 内容（≈0）/MoE（增量显著为负）均无增益。
4. ⇒ **R39 = 统一 WorldState→PolicyStateView + 解析 anchor 更新 + 学习残差（`A+ΔA`）**，B2 必须优于 analytic-only（B1）。

## 2. R39 计划与执行状态（A→B→C→D）

- **R39-A 工程闭合（零训练；✅ 全绿）**
  - **A1 ✅ `ed83e57`**：正式入口 `--path-head {off,structured}`（结构化 ckpt 严格装载 186/186、18 键迁移）；ckpt migration 接入正式 loader + **LoadReport**（filled 186 / renamed 18 / 0 缺漏；mapper `r38p2b`；CI 守卫例外科化白名单）；`forward_mode` 全迁（生产布尔调用 **0**，shim 删除→fail-fast）；**四 hash 原生**（protocol `f156c2a1…`/`950e9ad2…`、model_tree `db60ba5b…`、schema `v7-6c3a7ffb…`、ckpt `5a7c6f9e…`）；入口 vs 测试入口逐位；pytest 1095。
  - **A2 ✅ `f1f52f5`**：双账本——参数四分类（default：total 1,278,879 / dormant 407,401（MoE）/ 可达 871,478；structured：total 1,510,570 / dormant 515,853 / 可达 994,717），**上限断言通过**（总 ≤+2%/激活 ≤+5%）；detach **31 项登记** + 实测可达范围（`z_next.detach` 阻断 step2→step1；`action.detach` 当前装配惰性）+ **CI 禁未登记新增 detach**。
  - **A3 ✅（读数见 §0）**：重登记三臂 + 四 hash；报告 `.slim/deepwork/p1_v2/round39/a3_report.md`。
  - **B1 ✅ `c78aee0`+`7443ca7`（实现+对照）**：`net/world_state.py`（877 行；WorldState + `analytic_update`〔阻尼 speed α〕+ `readout` 七量带 `source∈{analytic,learned,held,missing}`/present/**唯一 producer 断言**（held 白名单仅 nav_progress）+ 残差接口〔复用 `st_gnn` 解码头，**零新增参数**〕+ `residual_loss`〔one-step、真实下一状态、结构性排除未来帧〕+ 单 detach 写点 `D-WS01`）；开关 `set_world_state_api(off|analytic|analytic_residual)` 默认 off ⇒ 逐位（plain 25/structured 73 max|Δ|=0，键面不变）；probes 增 `WorldStateHooks`（六断言，含 writeback=0）。**离线对照**：**B0 9.826/2.309（复现逐位）｜B1 12.265/2.782 ❌ 门**（slow+accel 欠动 −3.246→−0.806〔**补上 B0 缺口 75%**〕、恢复率 0.785→0.824；中高速过发 [4,6) 19.14 vs 17.63 ⇒ 全样本 +0.47）｜B1_damp(α=0.5) 11.232/2.490（单调无解）｜B1_residual 11.797/2.951（**帧错配 ⇒ B2 必须重训**）；A3 诊断复现（5.57→6.32 vs 真实 5.11→4.64）。pytest **1119**（编排复跑）。
  - **B2 ✅ `23bee92`（学习残差；已过"B2>B1"正式门）**：`net/world_residual.py`（ReadoutResidualHead 26→64→64→4，**6,148 参数**，末层零初始化+caps·tanh 有界）+ 新档 `analytic_learned`（默认关/未挂头逐位）。训练：one-step 真值监督（analytic→实际下一状态；**kin_b+阻尼同基**）、250,622 对（**按 episode 切分**无泄漏）、optimizer==头参数集、挂头预算 1,516,718 ≤ +2%（1,540,781）、冷跑 160 s、MoE/PPO/BPTT 未启用。**门读数**：c1 slow+accel Σds **13.909 ✓（首次过线）** / c2 全样本 **2.648 ✗（+0.338）**；**严格优于 B1**（ΔΣds **+1.6434**、Δ|err| **−0.1337**、恢复率 0.509→0.598）；α×缩放 Pareto 不可救门（最好 |err|≈2.385，差 +0.075）。**分层诊断**：缺口本质 = **列 vs 步距口径**（真实步距隐含 ~5.78 m/s vs speed 列 4.94、趋势反向）+ one-step→自回归链偏移（[2,4)/[4,6) 过发）；分列 val MAE：speed 0.317→0.253、static 0.183→0.105、dyn 0.161→0.144、**ttc 3.61→3.39（近不可学）**。pytest **1129**；逐位 all_pass。
  - **门读法（记录）**：GPT §四"状态接口门"= 无 t0 hold ∧ 无静默 0 ∧ 唯一 producer ∧ h1–h6 不指数增长 ∧ **B2 优于 B1** ⇒ **B2 已满足**；两条阈值（Σds≥12.8 ∧ |err|≤2.31）为"须写明口径的既有门"（现 c1 ✓ / c2 ✗）——**属性能目标，非接口门**。
  - **B2b ✅ `74b3c06`（口径对齐重训；预注册先写后跑）**：`column_residual_loss_aligned`（逐列权重 + speed 目标 ds/dt 口径 + TTC 事件式；旧函数不动）。**门仍 FAIL**：A(ds/dt) **12.3194 ✗/2.6682 ✗**；B(列口径) 14.0940/2.9249；C(h112) 12.0560/2.5753。**两项可证伪发现（重要）**：①**ds/dt 口径在"学"层成立**（val speed one-step MAE 0.1834→**0.1338**，优于列口径 0.1713）**但链上推力 ∝ 目标噪声**（Δ* σ **0.696→0.377**）⇒ **B2 的 c1=13.91 系"目标噪声×暴露偏移"副产物，不是列语义**；②B2 的"列 vs 步距趋势反向"是 **present 掩码伪影**（掩码后行级 MAE 0.092 同向；执行 ds ≡ 步距 0.0009）。**全家族 22 点无交点**（c1≥12.8 最好 c2=2.3992〔B2_a0.5_s1.25〕；c2≤2.31 的学习臂 **0 个**；仅 B0 hold 天然 2.3086）。**门复核证据**：c2 阈 2.31 ≈ B0 零学习读数 +0.0014；GBM OOF oracle MAE 1.548、对 slow_accel 加 δ=+0.540 条件平移后可 c1=12.8 ∧ 1.528 ≤ 2.31 ⇒ **门原则不矛盾，但要求强条件性，当前途径（冻结 head + 4 列 ΔA + detach）做不到**。pytest **1136**；逐位 all_pass。
  - **⏳ 待 GPT 裁决（本项即讨论点）**：①**迁移破门机制**（直接 **ds 空间**监督 / **链级 teacher-forced**，新预注册）还是 ②**修订 c2 口径**（`|Σds−真|` 均值对"回归均值"行为天然有利 ⇒ 建议改为：按加速需求加权 / 令牌化分桶 / 或与 c1 绑定的联合门）。我方倾向 **①为主 + ②的建议一并提交**（c2 单独成门对 B0 天然有利，属口径缺陷）。
  - **B2c ✅ `1db2aea`（闭环分诊，calib50 三臂）——R39 目前最重要闭环证据**：`ws_off`（C1_HEAD）0.40/0.18/**0.08**/0.34（rc 0.776、crawl 17.3s）｜`ws_analytic` 0.44/0.14/0.22/0.20｜**`ws_learned`（B2 头）0.56/0.20/0.18/0.06（rc 0.811、crawl 0.7s）**。配对（vs ws_off）：learned **Δsucc +0.16（9/1, p=0.0215）**、**Δtmo −0.28（14/0, p=1.2e-4）**、**Δcoll +0.10（7/2）**、Δoff +0.02、Δmin_ttc −0.50 ⇒ **Hard 劣化不晋级**；机制**剂量单调**（plan 剖面斜率 −0.06→+1.02→+2.05；ref−v 0.37→1.57→3.52；throttle 0.11→0.42→0.85；|err_y| 0.095→0.216→0.423；steer 饱和 3.7%→15%→44.8%；爬行 17.3→8.4→0.67 s）⇒ **WorldState API 的纵向修复在闭环兑现（爬行/超时被解决），绑定约束显式转移到"横向 @speed 质量"**（与 R36/R37 横向层一致）。四 hash：protocol `4df9573d…`/`2c2ed5ff…`/`064b8295…`；net 树 `527297c4…`（未变）；off 逐位（npz 0/78、episode 0/2、scalar 0）。
  - **B2d ✅ `2259c24`（剂量-反应扫描；决定性负结论）**：六点（0/0.25/0.5/0.75/1.0/1.25）——**不存在 `Δcoll≤0 ∧ Δsucc>0` 的区间**（碰撞配对 7/0、6/0、6/0、8/1、7/2、8/0；**碰撞代价对剂量不平坦不单调**〔ρ≈+0.15，机制"开"即 +10..16pp，scale=0 的 analytic 亦 +14pp〕）；而 **|err_y|/|err_ψ|/steer·lon 饱和/v/throttle 完美单调（ρ=+1.0）**、爬行/超时单调释放（ρ=−1.0/−0.83）⇒ **闭环定论：lateral@speed 为绑定约束**（剂量不可解；1.0 最优，>1.0 无增益）。**碰撞定靶（新增 7 例剖析）**：**tollgate ×3** + ramp_in/uturn/split/roundabout——**几何约束场景**，且 ws_learned 侧速度仅变 ±0.04~0.9 m/s（含一例**降速仍撞**）⇒ 系**横向精度**问题（非单纯速度）。id 级：{7718,7784,7795} 全剂量新增。
  - **R39-B 收口判定**：实现与纵向机制 ✅（接口门"B2>B1"过、闭环 +16pp succ/−28pp tmo/爬行归零）；**性能门未过**（Hard 劣化）⇒ **R39-B 以"机制成立 + 约束定位完成"结案**，不晋级；B3/B4（BPTT）与 R39-C 暂缓（见下）。
  - **⏳ 待 GPT 裁决（最终版，4 项）**：①**横向@speed 专项训练放行**（推荐首要；保留 WorldState 纵向机制；靶=tollgate/ramp/roundabout/split/uturn 几何约束场景的横向精度；验收**必须含闭环安全轴**）；②机制迁移（ds 空间/链级 teacher-forced）是否仍需（若走①可暂缓）；③c2 口径修订（B0 天然过线=口径缺陷）；④B3/B4（BPTT detach 门）与 R39-C（Memory 表示）在①之后排序。
  - **验收对照（§四工程门）**：正式 loader 可载结构化 ✅；forward_mode 全迁 ✅；四 hash 完整 ✅；HEAD Clean500 可复现（A3）✅；无旧臂语义混用 ✅。
  - **验收**：入口与测试入口逐位一致；严格加载；迁移报告完整；shim 无生产调用；四 hash 齐全；参数分类无重叠/遗漏；最新基线可复现。
- **R39-B 统一 imagined WorldState**（A 过门后）：九步循环（Policy 读 `PolicyStateView_k` → 动作 → 解析运动学 → LD ego-motion → OD 对象先验+ego-motion → WM 学习残差 → **统一 readout 重算七量** → Imagined Memory → 同一 Policy）；`A_{k+1}=AnalyticUpdate(A_k,a_k)+ΔA_θ`；**禁**脚本注入未来真值 / t0 静默 hold / 缺失填 0 / 外部未来态。臂：**B0 现状 ／ B1 analytic-only（阴性对照预期）／ B2 +学习残差 ／ B3 B2+2-step BPTT ／ B4 B2+3-step BPTT**。
  - 执行注记：B2 学习残差用 **one-step 真值监督**（analytic→**实际下一状态**；禁自产状态自洽回灌），链上配阻尼/一致性；**复用 `st_gnn` 存物理解码头**（od_pred/ld_pred/ego_next）以满足"不新增参数"。
  - 门口径：slow+accel 子集 = `v0<4 ∧ 真实 3 s 提速 >5.69 m（超 clamp 保速先验）`，n=1261，Σds ≥12.8；全样本 n=7133（训练行 ∧ 同 episode 连续 7 决策步，cap 300/stratum）均值 ≤2.31；h1–h6 不指数增长；learned>analytic-only。
- **R39-C Memory 表示**（B 过门后）：**M0** 当前状态 ／ **M1** +aligned deltas（OD：slot identity+ego-motion 对齐后 Δposition/Δvelocity/Δheading/presence；LD：投影当前 ego 帧后 Δgeometry/Δheading/Δcurvature/validity；**类别字段禁减**；**先离线零训练证**）／ **M2** M1+轻 attention（仅当 M1 显著）。隔离：Real（只读、episode/scenario reset）vs Imagined（从真实编码复制、结束销毁、**写回=0**）。
- **R39-D 分布适配**（后置）：expert transition + learner-state real transition；WM：real state + **executed action** → real next；Policy：real→expert + imagined→与 real future **决策一致性**；交替冻结，**禁单一 loss**。**门未过不启动 DAgger/MoE/RL**。

## 3. 挂账（逐项明确）

1. **R39-A ✅**（A1/A2/A3 三线全绿；A1b=semantic 装配接入正式入口 🕐）；**R39-B/C/D 未开始**（B 门：B2 必须优于 B1）；
2. **B 类**（掩码进头 + `static_distance` 下沉）与 B 阶段同批；
3. **Eval500 3-seeds** 与 clean150 层级去留（CG）；
4. protocol hash 原生接入（含在 A1）；
5. **大规模瘦身**（关闭 MoE 407k、旧 Policy 108k、兼容分支）待基本闭环门后执行（GPT 裁决）；
6. `runs/` 归域规则执行中（新 run：official=`runs/BTC<ts>_<name>`、dev=`runs/dev/roundNN/<arm>`；含 `.pt` 目录删除须批准）。

## 4. 工程纪律（不变项）

- 默认关 + 逐位验证；生产改动随 pytest（当前 **1068**）与契约门；**无 CG 放行不训练**；
- **清理六规则**：docs≤5 当期快照制（**台账每轮可整篇重写**）／历史走 git／runs 域与命名／数据-spec 登记制（`datasets/README.md`、`env/specs/README.md` 唯一清单）／重资产保留制／doc_stats 节拍；
- 口径登记：Calib50 / Clean500 / Eval500 禁直比。

## 5. 索引（指针）

- **手册**：`docs/repo_handbook.md`（**附录 D = 归档指针**：R30–R38 报告、r36_*/r38_*、manifest/reachability/ckpt 契约等，均含 `git log --diff-filter=D -- <path>` 取档命令）；
- **台账历史**：`git log --oneline -- docs/deepseekv4p1_argue.md`；**R38 波次 commit**：A4 `b9d1721`｜Q4-A `9577078`｜P1 `596747c`+`48cb5eb`｜P2a `a226285`+`dd0a8c2`｜P2b `939d1a2`｜P3 `f81da17`｜清理 `aecf44f`+`e4d85e0`｜R39 勘误/受理 `9a400ae`；
- **评估链**：`.slim/deepwork/p1_v2/a3/eval_a3.py`、`round32/canary_eval.py`、`round38/c6_*`（protocol/run/benchmark）、`runs/README.md`；
- **数据/spec**：`datasets/README.md`、`env/specs/README.md`（Clean500 canonical 副本已登记）。
