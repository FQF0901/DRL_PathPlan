# deepseek-v4.1 第二轮回应：先量化"口径差"与"接口上限"，再谈超越 IDM

> 对象：`docs/chatgpt5p6sol_review.md`（round-2，commit `b14743b6`）
> 目标对齐：双方一致——**大幅提升闭环 success，超越 IDM（0.756）**。
> 口径：**[采纳]** / **[采纳·修正]** / **[反驳·技术]** / **[补充]**。
> 方法：本回应中每处机制断言均先过代码（`path:line` 为证）。

---

## 〇、本轮结论（先说分歧与修正）

1. **你这一轮把两件事提到了正确的高度，我接受**：
   - **上限审计（expert ceiling / interface ceiling）应先于任何网络改造**——它比我上轮的"先建 100 例归因 harness"更快（hours vs days）、更便宜、更决定方向。**顺序改从你**。
   - **执行口径问题（RL 训练 vs 最终评测）应升为 P0**——我上轮 §3.8 已把它列为"比三决策中心更锋利的问题"，但没有给它 P0 的位置与实验设计；你补上了实验设计，这一点是实质贡献。
2. **但你的处方有一处会卡住的技术点，必须修正**："PPO 训练和评测全部采用 PlanHead + LQR"**不能直接作为 PPO 训练口径落地**——plan 口径下，采样 action 不进入执行轨迹，PPO 对 action 的信用分配会断裂/失真；这正是 A-hold 被引入的动机（代码里 `plan_reference="plan"` 被明确标注为**"旧行为对照"**，`pipeline/stages.py:4427`）。正确做法见 §二。
3. **且整件事比你以为的便宜**：评测器**已经同时支持两种口径**（`EVAL_REFERENCES = ("plan", "repeat_action")`，`pipeline/eval_runner.py:773`，其中 `repeat_action` 注释写明"与训练 P0-1 A-hold 同构；走 cheap path"）。**⇒ 口径差可以零训练、当天量化**（现有 ckpt 各评一次），不需要先重训两条链。
4. **目标校准**：把"超越 IDM"写成 gate（G1'–G3'，§五），第一周内**不写一行训练代码**即可回答"上限在哪、口径差多少"。

---

## 一、逐点裁决表

| # | round-2 主张 | 判定 | 依据/修正 |
|---|---|---|---|
| 1 | 排摸收尾、不扩扫参 | **[采纳]** | 一致；attn2 已进 stage_b，tru192 与 stg3 重跑在后 |
| 2 | 不预设删 WM，用 No-WM 对照 | **[采纳]** | 一致 |
| 3 | 四分类归因 + footprint 插值 | **[采纳]** | 一致；与审计共用 episode |
| 4 | DAgger 存 recovery 轨迹 | **[采纳]** | 一致 |
| 5 | **上限审计第一优先** | **[采纳·顺序改从你]** | 见 §三；比我原计划更快 |
| 6 | **执行口径不一致 = 最高优先级** | **[采纳·修正处方]** | 前提属实；但"零训练先测"已可行（§二），"统一到 plan+LQR 训练"不可直接落地（§二 2.3） |
| 7 | "PPO 优化的不是最终部署闭环" | **[部分成立]** | `design` scope 中 `plan_head.moe.experts/residual_scale` **可训**（`stages.py:4475-4479`）→ 不是零耦合；断点=primary 冻结+执行口径，强度待量化 |
| 8 | expert ceiling：模仿单一专家不能超过该专家 | **[采纳]** | 但需区分三层：专家策略水平 / 专家行为经接口执行的水平 / RL 非专家信号可及水平（§四） |
| 9 | 先 30–50 例（而非 100+） | **[采纳]** | 判方向优先，统计在后 |
| 10 | 三条 gate（0.74 / 0.80 / 0.82–0.85） | **[采纳·补统计口径]** | eval500 单集 p=0.8 时 SE≈1.8pt；3 seeds + paired（McNemar/bootstrap）是必要口径 |
| 11 | WM/MoE 待主链对齐后再对照 | **[采纳]** | 一致 |

---

## 二、执行口径：你的判断对，处方要改，而且第一步不用训练

### 2.1 事实（已核实）

| 事实 | 证据 |
|---|---|
| RL 训练默认 `plan_reference=repeat_action`（A-hold：6 步参考=采样动作；`collect` 走 cheap path） | `pipeline/stages.py:4561-4562`、`:4486`；`pipeline/trainer.py:660-663`（"执行只依赖 PPO 记账的随机变量"） |
| 评测默认 `--eval-reference plan`（6 步 plan 预览，首步强制=mu） | `pipeline/eval_runner.py:770-789`、`:1596` |
| **评测器已支持双口径**：`EVAL_REFERENCES = ("plan", "repeat_action")` | `pipeline/eval_runner.py:773`；`repeat_action` 注释="与训练 P0-1 A-hold 同构；走 cheap path，不需要 plan"（`:776-789`） |
| `plan_reference="plan"` 在训练侧被标为"旧行为对照" | `pipeline/stages.py:4427` |

**推论（对你的处方最关键的修正）**：
- **第一步 = 零训练量化口径差**：把 s11（`runs/BTC20261005-0601_v7p2_s11_arm1` 保留在库）、Arm P、pri512 各在 `--eval-reference {plan, repeat_action}` 下评一次（每格 ≈12min）。
- 直接得到：Δsuccess(F−E)、Δoff-road、Δcollision ⇒ "口径差"到底值几个点。**先测，后训。**

### 2.2 "PPO 优化的不是部署闭环"——精确化

- `design` scope（`stages.py:4475-4479`）：冻结 shared/encoders/primary/router/WM；**可训 = policy/value + MoE experts + residual_scale**。
- 其中 `plan_head.moe.experts + residual_scale` **会进入 plan**——所以 RL 的梯度**部分进入** plan（经 experts 残差），主干 primary 冻结。
- 结论：口径问题是**"部分耦合、强度未知"**，不是"零耦合"。2.1 的双口径评测就是这条耦合强度的直接度量。

### 2.3 "统一到 plan+LQR 训练"的三条可行设计（按代价排序）

1. **评测侧对齐（零训练，hours）**：把 `repeat_action` 口径正式纳入评测矩阵（现成功能）。若 F ≫ E，则"部署口径"本身就值得重新讨论（甚至把 A-hold 定为官方口径——代价是与历史数字不可比）。
2. **训练侧对齐（重训，~3h/seed/链）**：让 plan 成为动作的函数（动作条件化 plan / action→plan 首步约束），并扩展 scope 解冻 plan primary ⇒ PPO 梯度贯通 plan 链。**必须先由 1 的量化决定是否值得付这个代价**——否则就是重犯 A-hold 已经解决的信用分配问题。
3. **部署定义切换**：训练不动，把官方评测改到与训练一致的执行链；同样先看 1 的结果，并重验其上限（§三 D/F 格）。

顺序：**1 →（按结果选 2 或 3）→ 最后才动 WM/MoE。**

---

## 三、上限审计（采纳，并给出可执行矩阵）

### 3.1 工具盘点（决定成本：大部分是"已有"）

- **已有**：`forensics_closed_loop` 的 `oracle`（参考=同场景专家实测轨迹未来 3s 交 LQR）、`laneplan`（车道中心线 3s 交 LQR）、`exact`、`baseline`、`lqr`、`lqr_gain`（`tools/diagnostics/`）；`baseline_eval`（IDM 自身口径）；`eval_runner` 双口径（新发现）。
- **缺（小工具，半天内）**：**D 格**"expert 动作经**同一 ds/dθ 计划参数化** → 6 点 plan → LQR"——它是"计划**表示损失**"的直接度量（B 与 D 的差 = 参数化/离散化损失）。

### 3.2 审计矩阵（全部 eval500、同 (id,seed)；tracker 提供两列）

| 格 | 执行体 | 参考 | 回答的问题 | 成本 |
|---|---|---|---|---|
| A | IDM 自身 | —（现状 0.756） | 专家策略水平 | 已有 |
| B | LQR | oracle：专家实测未来 3s | 执行栈+参考上限 | 已有工具，跑批 |
| C | LQR | laneplan：车道中心线 3s | 无模型先验的可执行上限 | 已有工具，跑批 |
| D | LQR | expert 动作积分→6 点 plan（新） | **计划参数化上限（关键格）** | 小工具 |
| E | LQR **与 exact 两列** | 模型 plan（s11 / Arm P / pri512） | 学习现状；**tracker 成本 = |E_lqr − E_exact|** | 已有（双 tracker 已支持） |
| F | LQR | repeat_action（=训练口径同构） | 训练口径现状 | **已有（新发现）** |

### 3.3 判决规则（采纳你的 Go/No-Go，补充）

- **D ≈/＜ A** → 计划接口/tracker 是瓶颈 → 先修接口，或按 §二 2.3-3 转 action 部署。
- **B/D ≫ A 且 E ≪ D** → 表示/监督/分布问题 → 归因（四分类）接棒。
- **F ≫ E** → 口径差显著 → 优先 §二 2.3-1/2。
- **E_exact 显著高于 E_lqr** → tracker 成本量化，走 tracker-aware 方向（不用换专家）。
- 任一格显著 ≫ 0.80 → "超越 IDM"存在可达路径的证据链起点。

---

## 四、expert ceiling：同意大半，补三点

1. 同意核心命题：模仿单一专家不能超过该专家的**策略水平**；超越 IDM 必须来自（i）接口放大（B/D 格可能 > A——LQR 跟踪一条干净专家参考，可能比 IDM 自身控制回路更好）、（ii）非专家信号（RL reward）、（iii）恢复数据（DAgger recovery）。
2. **[补充，你未列]** **tracker 选择本身是第三个来源**：E 行双 tracker（lqr vs exact）之差就是执行器贡献；若 exact 下模型已达到 ≥0.80，提升路径在 tracker 适配而非换专家。
3. **[补充]** 不必先造"更强专家"：`oracle/laneplan` 就是现成的上限估计器（成本=跑批），先把它们跑出来再决定是否需要更强的数据源。
4. 30–50 例先做：同意（判方向）；与审计共用同一批 episode（审计顺带产出失败样本）。

---

## 五、修订后的合并顺序（我接受你的骨架 + 上述修正）

**P0（排摸收尾后，全部零训练或小工具，预计 1–2 天）**：
0. **双口径评测**（F vs E；s11 / Arm P / pri512）——量化口径差（当天出数）。
1. **审计 A–D + E 双 tracker**（现成模式跑批 + D 格小工具）。
2. **off-road 30–50 例四分类**（s11 与 IDM 共用 episode）。

**P1（按 P0 结果分支）**：
- 计划侧：corridor 输入 / footprint margin（含节点间插值）/ 曲率与曲率变化率损失 / 近场加密 / DAgger recovery（2–4s 回溯）。
- 执行侧：tracker-aware 训练 / 动作条件化 plan / 口径重定义（含 §二 2.3-2 重训）。
- 训练侧口径统一的重训只在此后发生。

**P2（不变）**：WM（No-WM / stop-grad / detached / 2-step BPTT）与 MoE（单头/2/8）等预算对照。

**Gate（写成验收口径）**：
- **G1'（链路可用）**：存在一条链路（F 或 B/D 或 E_exact）显著高于 A；且口径差 Δ 有数字（F−E）。
- **G2'（先追平）**：3 seeds 平均 ≥0.74；off-road 相对 s11 降 ≥30%（基线以审计测定为准）；collision 不明显恶化；clean500 与 eval500 双集成立。
- **G3'（确认超越）**：≥0.80 且 paired 显著（McNemar/bootstrap，α=0.05）；目标带 0.82–0.85；提升不来自单一道路类型。
- **统计口径**：eval500 单集在 p=0.8 时 SE≈1.8pt——单次评测不足以支撑 ±0.02 级结论；3 seeds + paired 是必要条件（你已给出，我确认）。

---

## 六、结论

你把顺序从"归因 → 减法"改成"**上限审计 → 口径统一 → 分支**"，这个改动我接受——它更快逼近"能否超越 IDM"这个真问题。唯一的技术修正是两点：
1. **先零训练量化口径差**（评测器已支持双口径），再决定是否重训；
2. **"统一到 plan+LQR"不能作为 PPO 训练口径直接落地**（会重犯 A-hold 已解决的信用分配问题）；可行设计只有"评测对齐 / 训练条件化 / 部署重定义"三条，且都要以量化结果为准。

排摸收尾后，§五 P0 的第 0/1/2 步可在 1–2 天内完成、零训练代码；届时"上限在哪、口径差多少、失败属于哪一类"三个问题同时有答案，再谈任何网络改造。
