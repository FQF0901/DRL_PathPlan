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
  - **A1b 🕐 在飞**：正式入口接入 `--ld-mode/--rollout-semantic`（使 C1_HEAD 可经 `tools/test.py` 复现）+ 入口↔a3 逐位。
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
