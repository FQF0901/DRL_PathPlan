# 架构与训练守则 v3

> **元信息**
>
> - **版本**：v3（2026-10-15）。上一版：v2（"架构与训练守则 v2"，`df152a6`，779 行）。
> - **来源**：`docs/chatgpt5p6sol_review.md`（16 轮 review，本版重点 R30–R36）；R30–R36 审计四份只读核对 findings（`runs/round36/agent_review/r{1,2,3,4}_findings.md`，R1=R30–32 / R2=R33 / R3=R34–35 / R4=WP3–R36，共 83 条）；`docs/deepseekv4p1_argue.md` §5.30–§5.55；`docs/round3*_report.md`。
> - **性质**：Coding Agent **不得偏离**的总守则——只写规则与门，不写报告体；每条数字/门均附来源（commit 或轮次）短注；历史锚点、作废项、待裁项、待复核项均显式标注。
> - **纪律**：动手前先读 §0 与 §7；未过门 = 硬阻断（§4.8）；本文与既有实现/文档冲突时，以本文为准（例外：待裁项以裁决为准，见 §4.10）。

## 目录

- [§0 一句话守则（TL;DR）](#0-一句话守则tldr)
- [§1 不可破坏的架构原则](#1-不可破坏的架构原则)
  - [1.1 唯一动作生成器](#11-唯一动作生成器)
  - [1.2 六步 Policy–WM 递归](#12-六步-policywm-递归)
  - [1.3 WM 与 Policy 职责分离](#13-wm-与-policy-职责分离)
  - [1.4 StructuredPolicyHead（三子模块、单步动作、多候选接轨）](#14-structuredpolicyhead三子模块单步动作多候选接轨)
  - [1.5 固定 LQR 与双口径](#15-固定-lqr-与双口径)
  - [1.6 架构契约硬门与交付物](#16-架构契约硬门与交付物)
- [§2 动作与状态表示](#2-动作与状态表示)
  - [2.1 横向：车道中心线锚 + 曲率残差（禁止从零生成）](#21-横向车道中心线锚--曲率残差禁止从零生成)
  - [2.2 纵向：路径条件化当前步](#22-纵向路径条件化当前步)
  - [2.3 数值契约（ds_init / κ 限幅，数据驱动）](#23-数值契约ds_init--κ-限幅数据驱动)
  - [2.4 动作特征与初始化契约](#24-动作特征与初始化契约)
  - [2.5 状态结构与坐标分离](#25-状态结构与坐标分离)
  - [2.6 显式几何/动态图支路的现状与优先级](#26-显式几何动态图支路的现状与优先级)
  - [2.7 外部输入边界与旧摘要退役](#27-外部输入边界与旧摘要退役)
  - [2.8 逐字段契约](#28-逐字段契约)
- [§3 职责边界](#3-职责边界)
  - [3.1 客观几何层（确定性算子：6 项输出 + 3 条禁输出）](#31-客观几何层确定性算子6-项输出--3-条禁输出)
  - [3.2 网络负责的决策](#32-网络负责的决策)
  - [3.3 禁止新增的人工决策先验](#33-禁止新增的人工决策先验)
  - [3.4 target-lane 规则（`select_target_lane()`）用途限定](#34-target-lane-规则select_target_lane用途限定)
- [§4 训练与推进协议](#4-训练与推进协议)
  - [4.1 阈值校准总则](#41-阈值校准总则)
  - [4.2 能力层级、范式流水线与前置门](#42-能力层级范式流水线与前置门)
  - [4.3 当前执行优先阶段（R36）：任务序与 G0–G6](#43-当前执行优先阶段r36任务序与-g0g6)
  - [4.4 训练契约：架构契约门 / 梯度门 / 消融门 / 工程门](#44-训练契约架构契约门--梯度门--消融门--工程门)
  - [4.5 canary 契约（动作条件化）与 MoE 口径](#45-canary-契约动作条件化与-moe-口径)
  - [4.6 决策收口：decision.md、三级决策树、自动停止](#46-决策收口decisionmd三级决策树自动停止)
  - [4.7 数据门、覆盖门与 DAgger 暴露率](#47-数据门覆盖门与-dagger-暴露率)
  - [4.8 非补偿与停损协议（晋级 / 基线 / 阶段阻断）](#48-非补偿与停损协议晋级--基线--阶段阻断)
  - [4.9 禁令清单与冻结项（含解禁条件）](#49-禁令清单与冻结项含解禁条件)
  - [4.10 阶段状态登记与实现偏差快照](#410-阶段状态登记与实现偏差快照)
- [§5 Probes、Hooks 与桩函数](#5-probeshooks-与桩函数)
  - [5.1 通用契约](#51-通用契约)
  - [5.2 输入与数据契约](#52-输入与数据契约)
  - [5.3 模块级 probe 与干预桩](#53-模块级-probe-与干预桩)
  - [5.4 路径与控制](#54-路径与控制)
  - [5.5 梯度与参数](#55-梯度与参数)
  - [5.6 WM Action Fidelity 测量协议与 ΔR²](#56-wm-action-fidelity-测量协议与-δr²)
  - [5.7 部署链归因 D0–D3](#57-部署链归因-d0d3)
  - [5.8 失败归因口径（六分类 + 导航三分拆）](#58-失败归因口径六分类--导航三分拆)
- [§6 验收准则全表](#6-验收准则全表)
  - [6.1 当前基线与口径（同口径）](#61-当前基线与口径同口径)
  - [6.2 架构契约门（含训练/部署同构）](#62-架构契约门含训练部署同构)
  - [6.3 逐层门 G0–G6（执行优先）](#63-逐层门-g0g6执行优先)
  - [6.4 闭环阶梯：calib50 → clean150 → 3-seed → IDM → 超越](#64-闭环阶梯calib50--clean150--3-seed--idm--超越)
  - [6.5 WM 接口门（量化）](#65-wm-接口门量化)
  - [6.6 消融门与梯度门](#66-消融门与梯度门)
  - [6.7 分层评测与场景专项安全门](#67-分层评测与场景专项安全门)
  - [6.8 导航角色门与自主选道门](#68-导航角色门与自主选道门)
  - [6.9 横向 / 纵向能力门](#69-横向--纵向能力门)
  - [6.10 统计与归因门](#610-统计与归因门)
- [§7 审计纪律](#7-审计纪律)
  - [7.1 改网络前必须核对](#71-改网络前必须核对)
  - [7.2 证据来源纪律](#72-证据来源纪律)
  - [7.3 判断纪律（先分布、后容量）与"过时清单"修订](#73-判断纪律先分布后容量与过时清单修订)
- [附录 A：关键数字与来源](#附录-a关键数字与来源)
- [附录 B：轮次–commit 索引](#附录-b轮次commit-索引)
- [附录 C：变更记录（v2 → v3）](#附录-c变更记录v2--v3)

---

## §0 一句话守则（TL;DR）

系统只有一个 PolicyHead（`StructuredPolicyHead`：横向分支 + 纵向分支 + 确定性组合层），它每次根据当前真实或 World Model 预测状态，输出当前一步"横向曲率意图（**车道中心线锚 + 曲率残差**）+ 路径条件化纵向动作"；$\kappa_{near}$ 决定当前动作，$\kappa_{mid}/\kappa_{far}$ 与未来进度仅辅助监督；同一 PolicyHead 与 World Model **递归六次**形成部署轨迹，六步只由这六次调用产生。外部只提供道路级导航和客观场景事实（车道几何/连接关系、静态/动态 slots），不提供目标车道、最佳闸口、最终候选、外部安全评分或规则速度答案。确定性坐标转换、碰撞几何和车辆运动学由解析层负责，世界残差由 World Model 学习，车道/闸口选择与速度决策由 Policy 学习。训练按"执行优先"推进：**先证明网络能执行正确决策（给定合法目标的保持/变道/merge/收费站闸口），再训练网络自己做该决策**；未过门 = 硬阻断，收益不得跨门抵消。

---

## §1 不可破坏的架构原则

### 1.1 唯一动作生成器

系统中只有一个动作生成器：

$$a_k=\pi_\phi(z_k)$$

即 **PolicyHead 是唯一动作输出模块**。明确禁止：

1. PlanHead 输出另一条轨迹；
2. 独立 PathHead 一次生成六步；
3. K-anchor 生成尾段；
4. Projector 成为第二规划器；
5. World Model 生成动作。

PlanHead 只是历史命名，其真实职责是 **PolicyBackbone / SceneContextEncoder**：负责融合场景，不负责输出另一套计划。

反例（R36 已认账的 H1 偏离）：`PathExecutionHead` 启用时一次输出六步 plan 并接管 rollout 尾段（`plan_override`），构成事实上的**第二轨迹生成器 + 绕过 Policy–WM 自回归**——`plan[:,0]==action_mu` 只是表面兼容（$a_0$ 旧头、$a_1...a_5$ 新头）。处置：暂停接入正式 rollout 训练，重构为 `StructuredPolicyHead` 内部子模块（§5.55 认账；`ac78f76` §八 1–2）。

### 1.2 六步 Policy–WM 递归

真实状态 $z_0$ → PolicyHead 输出 $a_0$ → World Model 预测 $z_1$ → **同一个** PolicyHead 输出 $a_1$ → World Model 预测 $z_2$ → …… → 得到 $a_0...a_5$。数学上：

$$a_k=\pi(z_k),\qquad \hat z_{k+1}=WM(\hat z_k,a_k)$$

- Policy 每次只决定当前一步；六步来自同一 Policy 的六次调用；
- 每一步都必须读取对应的真实或预测世界状态；每一步只消费当前动作 $a_k=(ds_k,\kappa_k)$，对外 $d\theta_k=\kappa_k\,ds_k$（`ac78f76` §八 4–5）；
- World Model 实际推进 5 次；
- **任何辅助未来预测都不能绕过这个递归链直接进入 LQR**。

### 1.3 WM 与 Policy 职责分离

| 模块 | 数学形式 | 只回答的问题 |
|---|---|---|
| World Model | $\hat z_{k+1}=f_\theta(z_k,a_k)$ | 执行动作后，世界会变成什么状态？（自车未来状态 / 动态对象未来状态 / 车道与道路相对状态 / 对象出现消失 / 场景上下文演化） |
| PolicyHead | $a_k=\pi_\phi(z_k)$ | 面对当前世界，下一步应如何规划？ |

接口必须满足：

$$\pi(\hat z_k^{WM})\approx\pi(z_k^{real})$$

即 World Model 不需要逐值复制真实世界，但必须保留 Policy 决策所需的信息。**WM 只预测状态**，不生成动作、不做选择。

### 1.4 StructuredPolicyHead（三子模块、单步动作、多候选接轨）

```text
PolicyBackbone / SceneContextEncoder
        ↓
StructuredPolicyHead
        ├─ lateral branch          横向曲率分支
        ├─ longitudinal branch     路径条件化纵向分支
        └─ deterministic composition layer   确定性组合层（dθ=κ·ds 兼容转换，无学习参数）
        ↓
当前一步 action  a_k = (ds_k, κ_k)
```

- **每次调用只输出当前一步**（$ds_k$ 为主，未来六步仅辅助；§5.55 细化①）；对外转换 $d\theta_k=\kappa_k\,ds_k$，然后交给 World Model 推进一步，再重新调用同一个 PolicyHead。
- 禁止：独立 PathExecutionHead 一次输出六步；横向头直接生成部署尾段；未来速度辅助头直接进入 LQR。
- 辅助边界（R36 修正，`ac78f76` L222-234）：

  | 输出 | 角色 |
  |---|---|
  | $\kappa_{near}$ | **决定当前 0.5s 动作**（推进 WM 的只有它） |
  | $\kappa_{mid},\ \kappa_{far}$ | 局部路径意图 / 辅助预测 |
  | 未来 speed / progress | 仅辅助意图 / 监督 |

  $\kappa_{mid}/\kappa_{far}$ 与未来 progress **不得直接替代后五步 Policy–WM rollout**；下一步在新 predicted state 上重新计算一组控制点（receding horizon）。
- **口径固定**：**κ_near = 当前动作**；**κ_mid/κ_far 与未来速度 = 辅助监督**——两者角色不得互换（R36 修正；v2 把三者一律写成"辅助意图"属语义偏差，R4 #14）。
- 多候选 + score head 接轨规则（R3 F8）：候选生成与候选评分头**必须收编在同一个 PolicyHead 内**；部署时仅被选中的候选转为当前一步 $(ds_k,\kappa_k)$；候选集合不得由外部提供，也不得存在独立于 PolicyHead 的候选排序器输出动作（`5fb3718` L398-437；`07f8c4b` L255-279）。
- 长期动作表示的"完成形态"与过渡定位（R2-17）：最终形态 = **横向路径头 × 纵向剖面头组合**（`κ_control_points` 或 $d(s)$ × $v(t)/s(t)$）；当前 $(ds,\kappa)$ 是**过渡版本但不是最终方案**（`49e04d7` L317/L327-334），不得把过渡接口当终态。

### 1.5 固定 LQR 与双口径

- **LQR 冻结**，不通过调控制器掩盖规划缺陷。
- 所有阶段必须分别报告：
  - **ExactTracker**：规划能力上界；
  - **Policy–WM 递归六次 → 六步计划 → 固定 LQR**：真实部署性能。
- **只有后者**可以用于和 IDM + Pure Pursuit 比较（IDM 锚 = 0.756）。
- ExactTracker 三用途限定（`1c1d91b` §二）：①数据采集；②单步上界评测；③Stage C 的 Planner-only PPO 候选。**任何在 ExactTracker 上提升的模型，都必须通过 WM rollout 6 步 + 固定 LQR 才能晋级。**
- 每个 PPO checkpoint 必须同时验收三种口径：ExactTracker / WM rollout consistency / 固定 LQR 闭环（`acf1b8f` §五；`1c1d91b` C1）。
- PPO 双口径登记（R2-18）：`ExactTracker PPO` 与 `部署同构 PPO` 是两条候选路径（后者仅当 C1 不能稳定迁移到部署时才执行，见 §4.2）。

### 1.6 架构契约硬门与交付物

**架构契约硬门**（`ac78f76` §九 + §5.55）——必须同时满足：

1. 动作生成器数量 = 1；
2. 所有 $a_0...a_5$ 都来自同一个 PolicyHead 参数；
3. 每一步 $a_k$ 都读取对应 $z_k$；
4. World Model 被调用 5 次推进未来状态；
5. 辅助未来输出不直接供 LQR；
6. **训练与部署完全同构**（本次 H1 偏离正是同构被破坏）。

**交付物**：`docs/architecture_contract.md` 必须存在且含 5 节：唯一动作生成模块 / WM 输入输出 / 六步 rollout 调用图 / 辅助头与主动作的边界 / 训练与部署是否完全同构（`ac78f76` §八 8；§5.55 L259 ⑧）。另须写入：**imagined 步（$k\ge 1$）外源几何来源**——静态几何由"t0 静态地图 + 想象位姿"确定性重算；动态 TTC 由 WM `od_pred` 或保持当前步（§5.55 注记①）。

**Gate F 新结构许可**（`d86e91a` §五）：任何结构修改必须满足：

- 对应 oracle/probe 解释 **≥ 30% 接口误差**；且
- 不是仅改善 latent loss；且
- 有明确零初始化 / 逐位兼容方案；

一次只改一个主根因。

---

## §2 动作与状态表示

### 2.1 横向：车道中心线锚 + 曲率残差（禁止从零生成）

- 横向分支输出 = **合法车道中心线锚 $\kappa_{ref}$（3 控制站位）+ near/mid/far 曲率残差**；$\kappa(s)$ 分段线性 → 精确积分 → 局部几何路径（H1 实现初值 tanh×0.1、末层零初始化 ⇒ 初始 $\kappa$ = 锚曲率；限幅须数据驱动，见 §2.3）。
- **禁止从零生成任意横向轨迹**（跨轮不变量；v2 文本按字面允许从零生成，属可合法偏离的口子，本版封死）。来源：`acd22c1` L283-294；`7c0186f` L28；§5.52①；`round36_report.md` L9。
- 候选参数化约束（R3 F11）：候选路径以车道中心线为锚（偏移 $\delta\in\{0,\pm0.5w,\pm1.0w\}$ 或等效车道参考），$\kappa$ 控制点只用于路径平滑/连续性；**merge 先保"本车道中心线续接"**；收费站/路口候选只能从**可见车道及其连接关系**生成，不得由外部指定最佳闸口（`round34b0_report` §4.2；`round35_report` §5；`07f8c4b` L243-253）。
- 证据：候选覆盖实验的**最佳候选全部为车道锚定路径**（收费站 = `lane_full_p` 相邻空闲闸口车道中心线、merge = `lane_center` 本车道中心线续接）；纯自由曲率的失败证据 = 压白实线（`first_invalid_s` 8.56 m/8.93 m）。

### 2.2 纵向：路径条件化当前步

横向决定"从哪里走"，纵向决定"沿这条路径走多快"；**不能做两个互不交流的横纵头**（弯道、变道、绕障必然影响安全速度）。

- 纵向分支读取：横向路径曲率 / 路径道路边界余量 / 该路径上的静态障碍 / 该路径上的动态对象与 TTC / 可通行长度 / 当前速度 / 道路级导航 / **导航（route）一致性**（补 R3 F19；`07f8c4b` L285-293）。
- 输出当前一步：$ds_k$ / $v_k$ / $\Delta v_k$；**当前步 ds 为主，未来仅辅助**（§5.55 细化①）。
- 路径条件化语义：收费站岗亭**只有在位于候选路径上时**才触发减速，避免全局障碍摘要造成错误加速或无条件减速（`07f8c4b` L306-308）。
- 旧动作 $(ds,d\theta)$ 的耦合问题（迁移动机，保留）：$ds$ 既表示进度和速度、又是曲率 $\kappa=d\theta/ds$ 的分母、还改变 LQR 预瞄尺度。

### 2.3 数值契约（ds_init / κ 限幅，数据驱动）

禁止拍定式常数，一律由专家分布 + 速度条件化决定（`ac78f76` §七细化②③；§5.55）：

| 项 | 规则 | 替代（作废值） |
|---|---|---|
| ds 初始化 | $ds_{init}=\mathrm{clamp}(v_{current}\cdot dt,\ bounds)$，按场景或当前速度初始化 | 固定 3.5 m/步（≈7 m/s，对收费站/急弯/低速 merge/静态障碍前过于激进） |
| ds 分布校验 | 初始化 ds 与专家 ds 分布的 **P50/P90** 校核（Gate 0/数据门的一部分） | — |
| κ 残差限幅 | 按专家曲率残差 **P95/P99** + 道路类型 + 速度条件化 $\kappa_{max}(v)$ | 固定 $\lvert\Delta\kappa\rvert\le 0.1\,\mathrm{m^{-1}}$（对应半径约 10 m，高速偏大、收费站急转偏小） |

### 2.4 动作特征与初始化契约

- **WM 动作输入（AC v2）**：$[ds_{norm},\ d\theta_{norm},\ \kappa_{norm}]$（`c3bf69f` L687-693）；禁止线性重复维（删除重复 speed-intent——否则与 ds 共线冗余）；$\kappa$ 在低 ds 下用 `valid_kappa` mask（`c3bf69f` L697）。
- **ego 物理监督**：`delta_progress`、`delta_speed`、`delta_heading`；其中 `delta_speed` 必须加入——speed 本身容易被 frozen/current-state shortcut 预测，$\Delta$speed 才能监督动作导致的纵向变化（`c3bf69f` L723-738）。
- **分支初始化契约**（R2-08）：隐藏层 Kaiming/Xavier 正常初始化；**输出层 zero-init；残差增益固定为 1**；删除可学习 $\alpha$ 或不再依赖外部冷启动。**禁止"输出层零 × 可学习零增益"双零死锁**（`c3bf69f` L101-142："整个分支永久为 0"；L671-678 方案一：初始前向逐位不变、输出层第一步就有梯度、不引入自由缩放标量）。
- canary 动作残差形式（`cde2e51` §四）：

  $$z'_{ego}=z_{ego}+T_{ego}(h_{ego},z_{ego})+\alpha_{ego}A_{ego}(z_{ego},a)$$

  $A_{ego}$ 输出层全零、$\alpha_{ego}=0$（逐位兼容）；动作嵌入**只进入 `ego_transition`**，不无差别广播到 OD/LD（同一样本空间下 OD/LD 已有 ego 运动泄漏）。

### 2.5 状态结构与坐标分离

```text
Ego:  physical state + semantic latent
OD:   explicit object anchor + intent/dynamics latent
LD:   explicit lane/map anchor + semantic/topology latent
```

每一步首先解析计算 ego 的 SE(2) 运动（平移 + 旋转），然后：

- LD 静态车道 anchor 做自车逆运动补偿：$p'_{LD}=R(-\Delta\theta_{ego})\,(p_{LD}-\Delta p_{ego})$，道路真实曲率不因坐标变换而改变；
- OD 先应用对象运动先验，再做自车坐标补偿：$p'_{OD}=T_{ego}^{-1}(p_{OD}+v_{OD}\Delta t)$。

**World Model 只学**：对象加减速 / cut-in / yield / merge 交互 / 地图重采样残差 / 可见性及 slot 变化。不能继续让 $z_{od}/z_{ld}$ 大量编码"ego 前进了多少"，否则会产生语义串扰。

边界与未决项：

- **OD 恒速先验不是既定默认**（R2-15 偏差修正）：33B 的 W1/W2 同时混合了"坐标系重表达"与"OD 恒速先验是否保留"两个因素，不能用一次零训练试验永久关闭解析 warp；**正式重训时必须对"OD 恒速先验保留与否"做消融**。
- 正式切换坐标规范必须配合重新训练，不能把新坐标 zero-shot 塞给旧 WM 后，根据约 2% 改善就否定该方向（保留 v2 边界注记）。
- 解析 warp 全无效的诊断分支（R2-19）：若 W1–W4 全无效，说明 ST-GNN 没有有效消费 geometry anchor，应转向**图消费结构诊断**，而不是继续加 warp（`c3bf69f` L649-655）。

### 2.6 显式几何/动态图支路的现状与优先级

| 项 | 现状 | 优先级 |
|---|---|---|
| OD 几何 | t0 position + $k\cdot dt\times$ t0 velocity（恒速外推，可能互相矛盾） | 不优先改 |
| node_mask | 固定 t0 node mask | 不优先改 |
| presence / entry head | 仅诊断输出，不改变未来 node mask | 不优先改 |
| 动态图支路归因 | D1→D2 = **0.0%** | 不优先，但**不得永久关闭**——显式动作条件化修复后，图几何通道可能才变得可消费（`910b49b` 问题 3/4；`cde2e51` §二.2） |
| ego 泄漏（OD/LD latent） | 泄漏的闭环因果**尚未证成**（34A D3：均值/点二列 ≈0、CI 含 0、两臂不同向；12/12 弱不利相关、中位 $\rho=-0.335$） | 不作为本阶段主线优先级；只在 WM 适配层作为**非回归门**（泄漏不增加）；评估其危害须用**干预对照**（R3 F20） |

### 2.7 外部输入边界与旧摘要退役

**允许外部提供（客观事实，不含选择结果）**：

- 全局道路路线 / 未来道路序列 / 前方路线顶点 / 直行·左转·右转命令 / 路口和道路拓扑 / 到转向点的距离（v2 原有）；
- 结构化车道几何与车道连接关系 / 车道–route 关联（是否连接全局 route）/ 静态障碍几何（slots）/ 动态对象 slots / 自车状态（补 R3 F6；`07f8c4b` L226-234；§5.51②）。

**阶段性例外（执行优先阶段）**：允许外部合法 target lane 作训练输入，只用于测试"正确决策能否被执行"；必须同时增加随机 dropout：**训练初期 token dropout 20% → 后期逐步提升到 50%**；该目标仅用于课程训练与能力上界，**不进入最终部署输入**；给定正确闸口仅课程/上界，门过即撤（`7c0186f` L66-78；§5.52③；`ac78f76` L430-439；§5.55 细化⑤）。

**禁止生产系统外部提供**：唯一 target lane / 最佳收费闸口 / 推荐变道方向 / selected candidate / 规则生成的最优轨迹 / 规则速度答案 / **外部安全评分**（R4 #13 补）。

**旧 5 维静态障碍摘要（`others[16:21]`）三阶段退役**（R3 F1；`5fb3718` L146-185/L443-447；`07f8c4b` L30/L59）：

1. **阶段 1**：保留字段但降级为诊断输入，跑四类消融（原始 / 置零 / 随机置换 / 只留 present / 只留距离）；
2. **阶段 2**：加入显式静态对象 slots，与旧摘要并存，旧摘要 dropout **50% → 80% → 100%**；
3. **阶段 3**：仅在三条判据**同时**满足时正式删除——显式对象模型闭环不退化 / 收费站成功率提高 / 静态障碍干预方向正确；
4. 硬门：重放误差 = 0、坐标·单位·mask 全部明确、**旧 5 维摘要置零后新模型不能崩溃**；
5. 注记：E1 结果层 0/50 翻转 ⇒ 旧摘要**不是杠杆**（可安全降级/废弃，但不是优化对象）；不再强化旧摘要，它不应成为长期主输入。

**静态障碍显式表示要求**（R2-04，岗亭教训）：可碰撞静态体（含 TollGateBuilding / 收费站岗亭类非车辆物体）必须有**显式 slot 或走廊占用约束**，禁止只依赖较弱的 others 走廊上下文。教训读数：该类障碍不在 OD、不在 LD、仅经 others 走廊段进观测，碰撞前仍在输出约 2.5–3.4 m/0.5 s 的计划进度。

### 2.8 逐字段契约

- **在没有逐字段契约前，不能允许外源字段进入正式训练**（`ac78f76` L126-146）。
- 每个字段登记：来源 / 坐标系 / 单位 / 时间戳 / mask / slot identity / schema 版本 / 是否真值 / 是否规则生成 / 是否部署可得。
- 允许：客观距离、TTC、边界余量、路径曲率；**禁止：最佳通道、推荐换道方向、外部安全评分、规则速度答案**。
- 8 维外源字段须逐字段契约化（纵向头输入），并声明部署可得性。

---

## §3 职责边界

### 3.1 客观几何层（确定性算子：6 项输出 + 3 条禁输出）

以下可以在网络外解析计算——它们是确定性物理算子，不是外部策略：

- 解析算子清单：坐标转换 / 曲率积分 / 车辆 footprint 扫掠 / 路径与障碍物最小距离 / 道路边界相交 / 动态对象 TTC / 路径可通行长度 / 候选与全局 route 拓扑连接关系；
- **客观几何层标准输出 = 封闭 6 项**（34B0 评估器口径，R3 F7）：① footprint collision；② minimum clearance；③ **first invalid s**（首个不可行路径位置）；④ **drivable ratio**（道路内比例）；⑤ static obstacle clearance；⑥ dynamic TTC；
- **几何层不得直接输出**（3 条禁输出）：❌ 最佳车道；❌ 最佳闸口；❌ 最终候选编号；
- 这些只是客观事实；不允许外部模块根据几何结果直接替网络挑选最佳候选（`07f8c4b` L310-329；`5fb3718` L384-396）。

### 3.2 网络负责的决策

生成哪些候选 / 选择哪条车道或闸口 / 如何修正横向路径 / 速度如何分配 / 是否变道 / 何时变道。

### 3.3 禁止新增的人工决策先验

**不要再增加**：更复杂的人工走廊分数 / 手写安全等级 / 手写目标通道代价 / 人工规划可行性标签直接作为 Policy 输入。理由：否则"规则规划器先做大部分决策 → 网络只学规则规划器的输出"（`5fb3718` L91-103；冻结令见 §4.9）。

### 3.4 target-lane 规则（`select_target_lane()`）用途限定

只能作为：教师辅助 / 诊断上界 / 课程训练中的给定正确目标；**不得成为最终部署的强制车道决策**。

R35 场景限定（R3 F5，必须遵守）：

- 该规则在 **tollgate/闸口级与含静态障碍走廊内不得作为课程标签或"正确目标"**——失败集实测规则正确率 episode 级 0.412、**tollgate 0/10 = 0.0**，且规则**不查障碍**（`lane_semantic.py:355-405`）；
- 任何以它为监督的实验必须**默认关、仅诊断**，并先记录其可行性/障碍检查结果；诊断臂可量化"正确 target 下 Policy 天花板"，但**不作生产路径**（`round35_report` §5；`07f8c4b` L218）。

网络必须内化：车道选择 / 收费闸口选择 / merge 保持或并入 / 路口提前选道 / 变道时机 / 绕障方向。

---

## §4 训练与推进协议

### 4.1 阈值校准总则

1. **相对门优先**：先实测 current baseline 分布 → 再以 baseline、constant-velocity 与 oracle gap 锁门 → 绝对阈值只作辅助，**不在开工前主观拍定**（`acf1b8f` 反驳 2）。
2. 首轮不设绝对 Jacobian 数值、不设绝对 WM 米制门，先用相对分解与相对判定，实测后锁定（`d86e91a` Gate B；`acf1b8f` §八）。
3. 同理：首轮 Action Swap top-1 ≥ 35% 已足以证明结构有效（起点近随机），不要求直达 70%（`cde2e51` §八 Gate 1）。
4. 本文所有绝对数字 = **待校准目标或历史锚点**，必须附来源；来源缺失者标"待复核"（见 §6.1、附录 A）。

### 4.2 能力层级、范式流水线与前置门

**能力层级（顺序锁定，v2 守则 6）**：

| 层 | 名称 | 说明/门 |
|---|---|---|
| 0 | 执行契约 | 核对 Policy 输出 / WM 输入输出 / 六步递归 / LQR reference / 实际执行（Gate 0，见 §4.3） |
| 1 | 车道保持 | 直道/弯道保持、曲率连续性、路径可跟踪性（Gate 1） |
| 2 | 固定路径纵向能力 | 跟车减速、静态障碍停车、安全恢复加速、曲率条件化速度、避免 timeout（Gate 5 纵向单调） |
| 3 | 给定目标后的横向机动 | 不给网络做车道选择：左/右变道、merge 续接、道内避让、变道后回正、给定正确收费闸口后的通过能力（Gate 2–4；T4 见 §4.3） |
| 4 | 自主车道/通道选择 | 外部只给道路级导航，网络生成和评分：保持/左侧/右侧/route continuation/收费闸口/路口出口候选（导航角色门 + 选择门，见 §6.8） |
| 5 | World Model 适配新动作与新状态表示 | 每完成一层 Policy 能力，同步验收 $\pi(z_k^{WM})\approx\pi(z_k^{real})$（Gate B2 口径） |
| 6 | DAgger 与 recovery | 覆盖 learner 非常规状态（learner 状态 → 恢复动作；暴露率与门见 §4.7） |
| 7 | MoE | Primary 稳定后再训练场景专家（MoE Gate 见下表 B4） |
| 8 | RL | 最后优化长期成功、安全、效率和多候选选择（C 阶段） |

**核心原则**：先证明网络能执行正确决策，再训练网络自己做该决策（层 3 → 层 4；Gate B0 → B2 顺序）。

**范式流水线（`1c1d91b` §一/§三，前置门锁定）**：`Stage A[WM] → B0–B4 → C0–C4`。**未过前置门不得启动下一阶段训练、蒸馏或结构改造**；**未过路径执行与安全门，不进入自主选道、World Model 重训、DAgger、MoE 或 RL**（`acd22c1` L425；`7c0186f` L337；`1c1d91b`）。

| 阶段 | 训练项 | 冻结项 | 前置门 / 验收门 |
|---|---|---|---|
| **A**（WM） | 状态转移、动作条件化、物理状态预测、多步滚动稳定性 | Policy | Gate A0（工程契约，见 §4.4）；Stage A 过门后才能启动 B |
| **B0**（Primary 单步 IL） | PolicyBackbone / Primary / PolicyHead；数据=expert real-state + 正确标注的扰动 + 少量 learner-state | WM、specific experts、router、ValueHead | teacher-state MAE 不恶化；learner-state MAE 显著下降；Primary-only 能完成基础 lane keep；Exact clean150 ≥ 0.55；LQR+WM-plan clean150 ≥ 0.45 |
| **B1**（Oracle-state rollout IL） | 真实未来 observation 按 $k=0...5$ 监督，权重 1.00/0.70/0.50/0.35/0.25/0.20 | WM | 同状态合法监督，无独立门（并入 B2） |
| **B2**（预测状态接口一致性） | $L_{interface}=\sum_k w_k\lVert\pi(\hat z_k)-sg(\pi(z_k^{real}))\rVert$，同时保留 oracle-state expert 监督 | WM | E_interface h3–h5 再下降 ≥ 30%；oracle-state Policy 不退化；plan 曲率/进度不恶化；LQR+WM-plan clean150 ≥ 0.50 |
| **B3**（DAgger） | 完整 Policy + ExactTracker 真实执行采集（WM 适配 + Primary real/predicted-state IL 完成后） | WM | 暴露率 10%/20%/≤30%，最多三轮；每轮：learner-state MAE ↓≥15%、rollout consistency 不退化、teacher-state 退化 ≤10%、Exact clean150 ≥ +3pp、LQR+WM-plan clean150 ≥ +3pp；连续两轮部署提升 <2pp 则停止 |
| **B4**（MoE） | 只训 specific experts + router；随后允许 Primary 极低 LR 联合微调一次 | WM、Primary（第一阶段）、旧 PolicyHead、ValueHead | MoE Gate：full-MoE 相对 Primary ≥ +3pp；普通道路不退化；expert residual 在普通道路接近 0；router 与困难路型存在稳定关联；3 seeds 成立；**不过门保持 Primary-only** |
| **C0**（ValueHead） | Critic warmup（success/off-road/collision/timeout 前有分辨力） | Policy、WM | 分辨力自证 |
| **C1**（ExactTracker PPO） | Policy 单步 → ExactTracker → 真实环境转移 → reward → PPO | WM、ValueHead 之外按契约 | 每 ckpt 三层验收（§1.5） |
| **C2**（部署同构 PPO canary） | 随机变量 $=a_0$；$a_1...a_5$ = 冻结 WM + 冻结 tail Policy 的确定函数；fixed LQR 执行 plan；只更新产生 $a_0$ 的策略路径 | WM、tail generator（update epoch 内 old/new mapping 不漂移） | **仅当 C1 不能稳定迁移到部署时才执行**；先 10–20 updates、3 seeds |
| **C3**（MoE RL） | Primary PPO 稳定后，才对 specific experts 做场景型 RL | — | 避免所有 expert 同时被全局 reward 拉动 |
| **C4**（GRPO） | 最后考虑，只用于同状态下多候选的组内比较 | — | 不替代 PPO 的 step-level Critic/GAE；PPO/critic 负责时间信用，GRPO 负责同状态候选相对排序 |

**路线图与依赖顺序**（R2-10；`49e04d7` L421-453）：R33C 动作条件化稳定化 → R34 Policy 分布 WM 适配 + real/predicted-state 接口一致性 → R35 动作表示迁移 → R36 结构化 WM → R37 DAgger → R38 MoE 专项训练 → R39 PPO baseline 与部署同构 RL。**具体回合号可以调整，但依赖顺序不得颠倒。**

**Primary / MoE / RL 分工（守则 9，保留）**：Primary 先掌握车道保持、基础跟车、普通弯道、给定目标变道、基本恢复；MoE 只负责收费站、merge/yield、roundabout、复杂路口、特殊避障、高级恢复——**不能让专家补偿日常 lane keeping 缺陷**；RL 真实 transition 可用于后续离线更新 WM，再重新适配 Policy 接口。

### 4.3 当前执行优先阶段（R36）：任务序与 G0–G6

只攻克"**给定合法目标车道后的路径执行能力**"：用车道锚定的横向曲率路径和路径条件化纵向速度替代内部 ds/dθ 耦合表示，完成普通保持、变道、merge 和正确收费闸口执行；全程保留输入、网络、路径、footprint 与 LQR 动态 hooks（`7c0186f`；§5.52②）。

**任务序**：T1 保持 → G1；T2 给定目标变道 → G2；T3 merge → G3；T4 收费站给定闸口执行 → G4；G5 纵向单调；G6 闭环 → clean150 → 3 seeds。

| 门 | 判据（数字见表内；来源 `7c0186f` §六 + §5.54③–⑧） |
|---|---|
| **G0 契约** | 旧模型默认路径逐位不变 / checkpoint 严格加载无缺失 / 训练参数集合与 optimizer 一致 / 坐标系·单位·mask 全部可验证 / 所有新 hook 默认关闭 bit-exact。**R36 已过 8/8，作为后续任何改动的回归门** |
| **G1 保持** | 普通道路子集：success ≥ 0.90 / 计划 footprint 越界率 ≤ 2% / collision ≈ 0 / timeout ≤ 5% / 曲率与曲率变化率 ≤ expert P95×1.25 / LQR steering saturation 显著低于当前基线。**子集口径待裁**（calib50 普通段仅 9 ep，建议专用 normal 集） |
| **G2 变道** | 合法 target 下变道完成率 ≥ 90% / 路径越界率 ≤ 5% / 方向正确率 ≥ 95% / 变道后回正率 ≥ 90% / 不必要反复换道率 ≤ 5% |
| **G3 merge** | 相对当前基线：merge **计划**越界下降 ≥ 50% / merge **实际** off-road 下降 ≥ 50% / 本车道续接候选成功率 ≥ 90% / 路径正确但 LQR 跟踪失败比例不得上升 |
| **G4 收费站执行** | 给定可行闸口条件下：可行路径覆盖 ≥ 95% / 岗亭 footprint collision 下降 ≥ 80% / 提前换道完成率 ≥ 85% / 收费站 success 显著非零、目标 ≥ 60% / collision 不得高于基线。**若给定正确闸口仍无法通过，不得进入自主闸口选择** |
| **G5 纵向单调** | 障碍更近 → 计划速度不得增加（≥95%）/ TTC 更低 → 计划进度不得增加（≥95%）/ 曲率更大 → 参考速度不得提高（≥90%）/ 障碍移除 → 应恢复进度（≥90%）；**"障碍移出路径 → 不应无理由减速"是失败判据**（R3 F19） |
| **G6 闭环（calib50）** | **success ≥ 0.62** / off-road ≤ 0.10 / collision ≤ 0.08 / timeout ≤ 0.10 / route completion 不下降；否则停止并按场景归因。**0.64 = 已作废 WP3（`acd22c1`）方案的闭环门，仅作历史锚点，不得当现行门使用**（0.62 与 0.64 之差 ≈ calib50 单集 1 例） |

**T4"晚换道"处理条款**（R4 #6；`ac78f76` L414-429；§5.55 细化④）：专家开始换道距离闸口中位数约 **25.8 m** ⇒ 直接模仿会编码"晚换道"；G4"提前换道 ≥85%"必须在训练侧单独处理——①对早期可决策状态增加采样权重；②监督目标车道方向与路径进度；③加入"剩余距离是否足够完成变道"的**可行性损失**；④DAgger 后期采集 learner 状态下的提前变道专家标签（后置）。**禁止通过人工"首步约束"直接规定转向动作。**

**可训 / 冻结白名单（18 键）**（§5.54；`7c0186f` L80-94）：

| 类别 | 内容 |
|---|---|
| 可训（白名单） | `path_head.` + `plan_head.fusion.2.`（**18 键**，H1 smoke 实测 18 可训 / 164 冻结） |
| 冻结 | World Model / Encoder 主体 / Memory Encoder 主体 / 旧 PolicyHead / MoE experts 与 router / ValueHead / LQR |
| 硬约束 | **白名单外任何参数被更新 = 门失败**；生产改动一律默认关 + 逐位（Gate 0） |

### 4.4 训练契约：架构契约门 / 梯度门 / 消融门 / 工程门

- **架构契约门**：见 §1.6（6 条 + `architecture_contract.md` 交付物）。
- **梯度门**（`ac78f76` §九）：

  | loss | 更新对象 |
  |---|---|
  | 当前动作 loss | → PolicyHead |
  | WM 预测 loss | → World Model |
  | 未来辅助 loss | → PolicyHead 辅助分支 |

  **辅助 loss 不得绕过 WM 成为第二尾段规划器**（与六步递归自动断言并列）。
- **梯度契约现状与 BPTT 门**（R1 M-05）：
  - 现状 = free-running inputs with **truncated cross-step credit assignment**（跨步截断）；**先做 h1/h3/h6 独立 backward 账本，不直接删 detach**（`acf1b8f` §C2）；
  - BPTT Go 门（`1c1d91b` A5）：h3–h5 状态误差额外下降 ≥ 20% 才过，**不过门即恢复 detach**；开跑前必须满足反证 3 的三道前提，且 **anchor drift hook 必须先行——没有这个 hook，不允许联合训练**（`910b49b` Probe 11）。
- **消融门**（`ac78f76` L486-495）：至少比较 ①旧 PolicyHead + WM rollout；②StructuredPolicyHead + WM rollout；③一次性 PathExecutionHead 诊断臂。**一次性头只能作为诊断上界，不得成为生产候选。**
- **交替训练与端到端禁令**（守则 7，保留）：Stage B 后期接口训练（$L_{real}=\lVert\pi(z^{real})-a^{expert}\rVert$，$L_{interface}=\lVert\pi(\hat z^{WM})-sg(\pi(z^{real}))\rVert$）$\to$ 后期交替（W-step：冻结 Policy，用真实状态 + 实际执行动作训 WM；P-step：冻结 WM，用真实状态专家监督 + 预测状态决策一致性训 Policy）；**先交替更新、梯度隔离**；禁止一开始就把 WM + Encoder + Policy 全部解冻、一个总 loss 端到端反传（防止 WM 与 Policy 通过不真实 latent 互相迁就）。
- **工程契约三等式**（Gate A0 级，R1 M-11；训练恢复前置）：

  ```text
  unknown config key            = 0
  silent ignored key            = 0
  dead optimizer parameter      = 0
  configured trainable set      == optimizer set
  baseline 前向 max_abs_diff     = 0
  ```

  入口遇到无效键直接 fail-fast；不能继续让配置"看起来生效"；配置声明可训练参数集 = optimizer 实际参数集 = 具有非零梯度的允许参数集。
- **四映射一致性硬门**（R1 M-12；`acf1b8f` §二.2/§五）：collect / update / eval / deploy 四个 effective action mapping 必须一致；不等价只能作为**显式标注的实现偏差**，未清零不得复训（清楚登记见 §4.10）。
- **监督与报告口径**（R3 F10/F13）：
  - 监督定义在可行候选集合上：**多个安全候选均为正例，不强制唯一答案**；专家实际选择只作正例之一，专家轨迹仅在可行集内作为参考（末段退化时降权）；
  - 训练必须**分项报告 loss**：`lane/gate choice`、`path geometry`、`speed profile`、`candidate score`、`collision margin`、`route consistency`；**禁止只报总 loss**。

### 4.5 canary 契约（动作条件化）与 MoE 口径

**canary 10 条锁定**（`cde2e51` §十）：

1. 先只给 ego transition 显式动作残差；
2. AC0 / AC1 / P-only / AC1-P 四臂（AC0=当前模型，不训练、基线；AC1=仅 Ego Action Conditioning；P-only=ego physical anchor 但无 action conditioning；AC1-P=两者都有）；
3. 不解冻原 WM 主体；
4. 不改 OD/LD transition；
5. 不 BPTT；
6. 不训练 Policy；
7. MoE-off；
8. Action Fidelity + lane retention + leakage **必须同时验收**；
9. AC1 通过后才采 60/40 policy transition 做第二阶段；
10. 未过门不自动扩大模型或数据。

**canary Gate 0–4**（`cde2e51` §八）：

| Gate | 判据 |
|---|---|
| 0 兼容性 | $\alpha=0$：所有 forward `max_abs_diff = 0`；旧 checkpoint missing/unexpected = 0；训练关闭时 RNG 序列不变；MoE-off baseline 逐位复现 |
| 1 Action Fidelity | 相对 AC0：Action Swap matched top-1 **10.2% → ≥35%**；top-3 **63.8% → ≥85%**；matched margin 必须转为正；ds FD 敏感度相对 dθ/κ 的差距 **26–50× → ≤10×**；ds-progress 单调率 ≥ 90%；zero/opposite action 明显劣于 matched |
| 2 物理语义 | progress/speed：WM 相对 frozen 的增益为正；lane_lat/lane_head **保留率下降 ≤ 5pp**；OD/LD latent 中 ego progress/speed **泄漏增加 ≤ 10%**。**action fidelity 提升但 lane 保留继续恶化 ⇒ 不允许进入完整训练** |
| 3 部署接口 | E_interface ds：h3–h5 至少下降 30%；E_action ds：h3–h5 至少下降 25%；h5 plan_ds overshoot：**1.66 m 至少下降 40%**；真实状态 Policy action 误差退化 ≤ 5% |
| 4 闭环 | 口径=K1e2 + MoE-off + WM-plan + fixed LQR、calib50 paired：success ≥ 0.58（相对 0.56 至少 +2pp）/ off-road ≤ 0.14 / collision ≤ 0.08 / timeout 不高于当前。50 episodes 的 +2pp 仅 1 个翻正 episode ⇒ **只能作为继续许可，不能作为收益结论**；进入 clean150 的门应更高：calib50 success ≥ 0.60 **或** 接口/动作链指标达到 Gate 3 强门 |

**canary 结果定性（R33C）**：B1 = **稳定器而非推进器**——接口/overshoot 被修复（AC1 恶化 +17~21% → B1 改善 −10~−12%），coll/tmo 改善，但 succ 未动（0.56）、off-road 回退、top-1/单调/泄漏未过门；**canary 不晋级 C-B；AC1/AC1-P 均不作为新 baseline**；可保留为诊断参考；**不得自动进入训练**（`c3bf69f` L818/L865；`round33c_report` L66/L77）。

**MoE 口径令**（R1 M-10；`910b49b` §七；`d86e91a` §一.4）：未过 MoE Gate 前，所有 WM/Policy 接口实验**训练与评测一律 MoE-off**（保持训练/评测一致）；MoE-on 只作为后期专项专家训练后的正式候选；**未经专项训练的 experts/router 不能进入正式基线**；统一口径 `rollout_semantic` = 既有最佳口径（`d86e91a` §七）。

### 4.6 决策收口：decision.md、三级决策树、自动停止

- **decision.md 单一收口**：一轮结束只写一个受控枚举值，且**不得自行进入下一阶段训练**。可用枚举（`d86e91a` §七）：`GO_ACTION_CONDITIONING` / `GO_STATE_REPRESENTATION_FIX` / `GO_POLICY_INTERFACE_ROBUSTNESS` / `GO_ROLLOUT_PROTOCOL_FIX` / `GO_DYNAMIC_GRAPH_FIX` / `GO_ALTERNATING_WM_POLICY_TRAINING` / `BLOCKED_INSTRUMENTATION`；另：梯度审计不闭合 ⇒ `BLOCKED_CANARY_VALIDITY`（`c3bf69f` L856-857）。
- **三级决策树**（`d86e91a` §三，前一级决定后一级）：**Level 1 决定主根因** → **Level 2 决定 WM 内部结构**（只有 Level 1 指向 WM 后才执行）→ **Level 3 决定联合训练**（只有 Level 1/2 证明需要联合适配后才执行）。
- **分支树（R33C 三分支，R2-05）**：
  1. 分支 1 = B1 全门通过 → Policy 分布 transition 60/40 适配；
  2. 分支 2 = Policy 接口稳健训练（$\pi(z^{WM})\approx sg(\pi(z^{real}))$），而不是继续增大 action conditioning；
  3. 分支 3 = 静态障碍显式表示 + 路径走廊占用/路径条件化纵向剖面——**分支 3 未解决前不启动 RL**。
- **自动停止规则 5 条**（`1c1d91b` §五）——触发即自动停止、写 decision.md、**不得自行开下一阶段、不扩大训练**：
  1. 配置契约破坏基线：baseline `max_abs_diff ≠ 0` / checkpoint 加载不完整 / optimizer 参数集不一致；
  2. 数据不可信：state-action-next_state 错位 / 重放失败 ≥ 0.1% / obs fingerprint 混用 / 路型覆盖严重缺失；
  3. A0 probe 工具不闭合：oracle-state 重放与真实记录无法对齐 / whole-scene mirror 不物理 / horizon 索引错误；
  4. A1 灾难性退化：expert-val 退化 > 20% / NaN·Inf / latent collapse / OD·LD 明显崩溃；
  5. A1 没有实质改善：E_interface h3–h5 下降 < 20% **且** E_deploy h3–h5 下降 < 15%。
- **归因轮纪律**（R2-14）：一轮只回答既定问题，**不要扩展新的通用 probes**；结论必须是"分支确实按预期获得梯度"，否则整轮 canary 归因无效。

### 4.7 数据门、覆盖门与 DAgger 暴露率

**数据门（分场景覆盖，合取口径）**（R3 F2/F3/F12；R4 #18；`07f8c4b` L336-349；`acd22c1` L394-399）：

| 场景 | 覆盖门 | 数据量门 |
|---|---|---|
| 收费站 | ≥ 95% | ≥ 100 episodes |
| merge | ≥ 95% | ≥ 100 episodes |
| 路口（左转/右转） | ≥ 95% | ≥ 100 episodes |
| 普通道路 | **≥ 99%** | ≥ 200 episodes |

- "覆盖"定义 = **存在至少一条 安全（无碰撞 ∧ 不越界） 且 route 一致 的候选**（合取，34B0 严格合取 17/17 即此口径）；
- **不过门先修候选生成，不训练 selector**；**闭环安全门不过，不进入 DAgger、MoE 或 RL**（`07f8c4b` L332-334/L351/L478-480；`5fb3718` L457）；
- 时序纪律：**数据提取和 hooks 先行，训练后置**（`acd22c1` L463 item8）；
- 现状注记：expert5k_v8 = 455/441/906/907（toll/merge/路口/普通）≥ 目标 ✓；缺口 = 带 v7 字段的独立场景（54/55/109/110）+ policy 失败 episode（21/16/28/28），专项扩采后置。

**数据规模与复现门（Stage A/B 口径）**（R1 M-13；`1c1d91b` A1）：每个 Policy 1,000–1,500 episode、合计至少 50,000 个有效 transition、目标 100,000；匹配率 = 100% / 重放 ≥ 2,000 / 不可复现率 < 0.1% / 负 ds = 0。

**DAgger 暴露率**（`1c1d91b` B3）：Round 1 = 10% / Round 2 = 20% / Round 3 = 最多 30%，最多三轮；每轮门与停止条件见 §4.2 表（连续两轮部署提升 <2pp 则停止）。

**DAgger 数据契约**（守则 8 保留 + R3 F14 补候选级标签）：

| 字段 | 用途 |
|---|---|
| `obs_t` / `expert_action_t` | Policy：obs_t → expert_action_t |
| `obs_t` + `executed_action_t` / `obs_t+1` | WM：obs_t + **环境实际执行的动作** → obs_t+1（**不得使用专家建议但未执行的动作**） |
| `policy_action_t` / `executed_action_t` / `road class` / `outcome` / `plan[6]` / `actual ego pose delta` | 诊断与归因 |
| **`candidate_set` / `policy_choice` / `expert_choice` / `expert_path`**（R3 F14 补） | selector/score head 的标签来源；**selector 只用这三项（candidate_set/policy_choice/expert_choice）训练** |

同一次 learner rollout 同时服务 Policy 与 WM，但必须**不同标签 / 不同 loss / 不同 optimizer step**。

### 4.8 非补偿与停损协议（晋级 / 基线 / 阶段阻断）

- **非补偿原则**：单轮晋级必须**全门通过**；任一门失败不得由其他门收益抵消——"不能再用 success +4pp 抵消 collision 翻倍"（`c3bf69f` L778；R2-01）。
- **"动作辨识/敏感度变强 ≠ rollout 更正确"**：WM 动作敏感度/可辨识性提升，只有**伴随接口指标不恶化**才计为有效改进（`c3bf69f` L8/L871）。
- **停损**：任何接口指标恶化超过 **5%**，或 collision 再次恶化，**立即停止**（`f0ebfae` L68-83）。
- **基线纪律**：未过门轮次**不得成为新基线**（可作诊断参考，如 B1）；**禁止自动进入下一训练阶段**。
- **阶段阻断（硬）**：未过闭环门 ⇒ 不进 DAgger/MoE/RL；给定正确闸口仍失败 ⇒ 禁入自主闸口选择；未过路径执行与安全门 ⇒ 不进自主选道、WM 重训（`acd22c1` L425；`7c0186f` L144/L293/L337）。
- **统计证据要求**（R2-11；`c3bf69f` L795）：晋级/恶化判定必须给出**噪声地板与配对 CI**（episode 级配对 bootstrap）；**3 seeds 前先做配对置信区间**；3 seeds 判定附 paired LB > 0、McNemar $p<0.05$、非单一路型。
- **接口门 ≠ 闭环门**（R3 F21）：B1（AC v2）接口误差改善约 **10–12%** 但 success 0.56 不变、off-road 0.14→0.20、tollgate success = 0 ⇒ 接口改善不得当闭环收益。
- **小样本纪律**：50 episode 上的 +2pp ≈ 1 例翻正，只能作继续许可，不作收益结论（`d86e91a`）。

### 4.9 禁令清单与冻结项（含解禁条件）

**9 条禁令**（执行优先阶段；`7c0186f` §七）：

1. 暂不训练自主车道/闸口选择；
2. 暂不删除 target-lane token；
3. 暂不修改 World Model；
4. 暂不做 ego-motion warp 重训；
5. 暂不进入 DAgger；
6. 暂不启用 MoE；
7. 暂不启动 PPO/GRPO；
8. 暂不修改 LQR；
9. 暂不扩大网络容量。

**冻结清单**（`f0ebfae` L85-93；R35 追加）：60/40 Policy 数据适配 / DAgger / MoE / PPO·GRPO / BPTT / Policy 解冻 / LQR 修改 / WM 主体扩容 / 外部最佳车道规则 / 外部最佳闸口规则 / 新增人工走廊分数。

**禁止掩盖性操作**（R2-12；`c3bf69f` L881-884）：不能简单做"加正则、限幅 $\alpha$、再训更多数据"来掩盖根因；协议要求先定位并修复根因。

**解禁条件**：禁令/冻结均为**阶段作用域**（"暂"），仅当**上一层门通过**才解禁；与长期方向的衔接——warp 重训 = 正式切换坐标规范时配合重训（§2.5）；BPTT = 过 §4.4 Go 门；GRPO = 未来候选级排序辅助，不能替代 PPO 逐步信用分配；自主选道 = 执行门（G1–G4）全过后进入层 4。

### 4.10 阶段状态登记与实现偏差快照

**阶段状态登记**：

| 项 | 状态 | 依据 |
|---|---|---|
| 当前阶段 | **执行优先**（给定合法目标车道后的路径执行；R36 主线） | `7c0186f`；§5.52 |
| WP3（Network-Internal Lane and Path Selection，`acd22c1`） | **已终止回退**：受理与 G1/G2/G3 队列全部终止（G2/G3 中止，留档不再执行）；其 0.64 闭环门**作废** | §5.52 |
| H1 `PathExecutionHead` | **暂停**接入正式 rollout 训练；重构为 `StructuredPolicyHead` 内部子模块（K1 开工） | §5.55；`ac78f76` §八 |
| K-anchor 尾段 | 生产 rollout 的 K-anchor 模式同样消费 t0 plan（`anchor_plan[:,k,:]`）；本次只对新结构化头执行"每步重算"——**是否同步收紧 K-anchor 模式待裁** | §5.55 注记② |
| 门状态 | G0 已过 8/8；G1–G6 未启（DP-6 全量提取 I1 进行中） | `round36_report`；§5.54 |
| canary（AC1/B1） | 不晋级、不作新 baseline（保留为诊断参考） | `c3bf69f`；`round33c_report` |
| MoE | 未过 MoE Gate 前训练/评测一律 MoE-off | `910b49b` §七；`d86e91a` §一.4 |
| 待裁项 | ①G1 普通道路子集口径（专用 normal 集）；②T4"晚换道"处理策略（课程/加权/约束——注意"人工首步约束"被禁，见 §4.3） | `round36_report` §4 |

**实现偏差快照**（R1 O-04；**偏差未登记/未清零不得恢复训练**）：

| 偏差 | 现状 | 目标 |
|---|---|---|
| Stage C 执行口径 | 现状 = **S3**（sampled action → repeat×6 → LQR → PPO），LQR+repeat ≈ **0.02–0.06**，仍崩 | 目标 = sampled action → ExactTracker → 真实转移 → PPO；部署 = Policy–WM ×6 → 固定 LQR（**三者不是同一执行口径**） |
| 训练/部署链 | Stage C ≠ Deploy（F4 判 P0 根因） | 口径统一后开训 |
| eval 双栈 | 默认口径相反 | 显式登记并统一 |
| 死键 | 占位/未消费配置键存在（对应 Gate A0 fail-fast 项） | dead key = 0、optimizer 集一致 |
| MoE 口径差 | 训练口径 MoE-off 略优 **2–4pp**；MoE-on 0.52 | 一律 MoE-off（§4.5） |

**红线**：Reality Audit 级核对完成前，不得开始任何新主模型训练或蒸馏（`acf1b8f` 反驳 1）。

---

## §5 Probes、Hooks 与桩函数

### 5.1 通用契约

所有工具/分支必须：

1. 默认关闭；
2. 关闭时 bit-exact；
3. 不改变 RNG；
4. 不进入 checkpoint；
5. 不改变 optimizer；
6. 只增不改（启用键集变更需显式登记）。

追加硬约束（R1 M-14）：

7. 不影响 BatchNorm；
8. 不改变训练/eval 行为；
9. 相同输入重复运行逐位一致；
10. finite-difference 结果对 epsilon 稳定；
11. horizon/time index 无错位。

**生产分支级契约**（R2-13，不止工具级）：默认关 + 逐位兼容 + 启用键集 + 零初始化恒等 + RNG/梯度隔离 + optimizer 成员 fail-fast（`assert optimizer_parameter_ids == declared_trainable_parameter_ids`）+ checkpoint metadata（`action_ego_enabled`/`action_feature_dim`/`alpha_mode` 等）+ 加载不匹配 fail-fast。

### 5.2 输入与数据契约

记录：字段来源 / 坐标系 / 单位 / 时间戳 / mask / slot identity / schema 版本 / 是否真值 / 是否规则生成 / 是否部署可得。

### 5.3 模块级 probe 与干预桩

- **Policy** 记录每一步：真实 latent / WM latent / PolicyBackbone latent / 横向曲率输出 / 纵向速度·进度输出 / 最终当前 action / 辅助未来意图。
- **World Model** 记录：`action_in` / ego·OD·LD latent in·out / OD·LD anchor warp 前后 / 解析坐标变换 / learned residual / physical prediction / presence·entry。
- **六步递归**记录：$z_0...z_5$ / $a_0...a_5$ / 每一步 Policy 实例 ID / 每一步 WM 调用 / 累计轨迹；并**自动断言**：$a_0...a_5$ 来自同一个 PolicyHead；每个 $a_k$ 读取对应 $z_k$；WM 实际推进 5 次；辅助头没有直接进入 LQR。
- **探针重点模块 7 项**（R3 F17）：Nav Encoder / Lane Semantic Encoder / LD Encoder / Static Object Encoder / Policy Backbone / Policy Head / World Model。
- **网络统计字段**：每模块输入/输出 shape / mean·std·min·max / 有效 slot 比 / **effective rank** / mask ratio / activation norm / **sat_frac**（饱和率）/ gradient norm / update norm（34A 明列纯缺口：effective rank 与 slot permutation）。
- **几何 hook**：ego pose delta / OD·LD anchor before warp / after warp / **expected analytic transform** / **transform residual** / **slot permutation**（现状 M6 ◐：warp 前后 anchor 未落，须补齐）。
- **WM→Policy 四增益链**（R2-09，必录诊断）：$z_{ego}$ 变化 / PolicyBackbone latent 变化 / Policy `action_mu` 变化 / rollout plan 变化 → 定义 $G_{WM}$ / $G_{backbone}$ / $G_{head}$ / $G_{total}$（R33C 已以之定位"放大仍在 WM 侧"）。
- **干预桩 7 类**（R3 F17，默认关、bit-exact、可复现复跑）：字段置零 / slot 置换 / 左右镜像 / 距离扰动 / route 交换 / target lane 交换 / 静态障碍移入·移出候选路径。

### 5.4 路径与控制

记录：曲率路径 / 速度剖面 / 30 点 LQR reference / **footprint 连续扫掠（≤1 m 插值，非 6 点）+ 9 点 + env 中心三口径** / 首次越界点 / 首次碰撞点 / 道路内比例 / 最小道路余量 / 最小障碍余量 / 实际车辆轨迹 / raw steer·throttle / 饱和比例（含区间饱和比）。

LQR 对照：规划误差 → `plan_ok`/`track_ok` 四象限（样例 60 双好 / 0 路径错 / 2 路径好但跟踪错）。

### 5.5 梯度与参数

记录：`requires_grad` / optimizer group / grad norm / update norm / checkpoint loaded·changed / h1·h3·h6 梯度可达性。

### 5.6 WM Action Fidelity 测量协议与 ΔR²

- **有限差分优先**（R1 M-06）：解析 Jacobian 会被 detach 人为置零，**只能证明训练图禁止该梯度传播**；必须以数值有限差分为主；先确认 $\pm$扰动未撞 action bound，否则会被 clip 污染。
- 同时报告：①解析梯度（证明当前梯度契约：autograd Jacobian = 0）；②数值有限差分；③扰动与 clip 检查。
- 判定项：Action Swap matched top-1 / top-3 / matched margin / zero·opposite / ds FD 敏感度相对 dθ·κ 的差距；**首轮不设绝对 Jacobian 数值，采用相对判定**（`d86e91a` Gate B）。
- **ΔR² 条件化增益**（R1 M-08，解码/保留类结论的必要口径；`cde2e51` §一.2）：

  $$\Delta R^2_{WM}=R^2_{WM}-R^2_{frozen},\qquad \Delta R^2_{action}=R^2_{correct\ action}-R^2_{shuffled\ action}$$

  如果 WM 与 frozen 几乎相同，说明所谓"保留"来自状态惯性，不是动作动力学；**绝对 $R^2$ 不构成证据**。

### 5.7 部署链归因 D0–D3

| 段 | 内容 |
|---|---|
| D0 | planned-pose chain |
| D1 | action-faithful pose replay |
| D2 | oracle ego + oracle OD pose |
| D3 | oracle future latent |

分解含义：D0→D1 = 计划 pose 与实际执行 pose 差异；D1→D2 = 动态图几何近似误差；D2→D3 = latent transition 自身误差；D3→expert = Policy 真实状态误差。

- **份额判据**（`d86e91a` §五 Gate E）：≥ 40% 总 gap = 主瓶颈；20%–40% = 共同瓶颈；< 20% = 次要因素。
- 任何 WM/接口爆炸结论**先做 D0–D3 分解并报份额**；不能直接把这 8–10 倍全部叫作"WM learner-distribution failure"。
- R32 读数：D0→D1 2.9%（非主因）、D1→D2 0.0%、D2→D3 ≈ 97%（**`cde2e51` 保留意见：97% 不等于严格证明 latent transition 是唯一瓶颈**）；A_k（Policy 放大）温和 ×2.1（p50 0.098→0.206），Lipschitz 中位比 0.73–0.78（不过敏）、尾部 P95 ≈ 5.7–6.1（尾部放大）⇒ 接口问题=次要/尾部。

### 5.8 失败归因口径（六分类 + 导航三分拆）

- **六分类**（`82e815b` L169-176）：①路径本身撞障碍；②路径本身越界；③路径可行但 LQR 跟踪出界；④目标通道选择错误；⑤静态障碍未进入模型；⑥障碍进入模型但 Policy 未响应。
- **导航三分拆**（`07f8c4b` L123-133）：①外部 target lane 选错占比；②target lane 正确但 Policy 路径选错占比；③Policy 知道障碍但未改道占比。**优先级顺序须固定**（不同优先级会改变计数），并先答"外部规则错还是 Policy 错"。
- **覆盖率门**：≥ 90% 的新增失败 episode 必须落入明确类别（未归因允许上限 1 例口径）；所有关键输入字段的坐标系·单位·mask 可验证；遮挡干预的输出响应可重复；工具开/关 bit-exact（`82e815b` L279-284）。
- 基线读数：34A 18/19 = 0.947；35 F1 17/17 = 1.00。

---

## §6 验收准则全表

### 6.1 当前基线与口径（同口径）

| 口径 | 读数 | 来源 |
|---|---|---|
| **Round-32 开发基线**：K1e2 + MoE-off + K-anchor-off，LQR + WM plan | success **0.560** / off-road 0.140 / collision 0.080 | `d86e91a` §一.4 |
| 同基线 Exact | success 0.560 | 同上 |
| 同基线 LQR + repeat（S3） | success 0.060（0.02→0.06，仍崩） | `910b49b` §七 |
| S1≈S2 / S3 | S1≈S2≈0.56 继续成立；S3 崩溃继续成立 | `d86e91a` §一.4 |
| MoE 口径 | MoE-on 0.52 / MoE-off 0.56；训练口径 MoE-off 略优 2–4pp | `910b49b` §七 |
| **IDM + Pure Pursuit（比较锚）** | **0.756** | `910b49b`；`82e815b` L534 |
| 当前坐标 | 0.56 →（目标）0.60+ → 追平 0.756 | `c3bf69f` L906；`49e04d7` L488 |
| 现行同口径单轮硬门（R33C） | success ≥ **0.58** / off-road ≤ 0.14 / collision ≤ 0.08 / timeout ≤ 11 / route completion ≥ **0.862** | `f0ebfae` L76-80（timeout 原文口径如此，使用时复核单位） |
| WM 固有 TF vs 部署链 | 0.02–0.04 vs 0.20–0.32（**8–10×**） | R30；`f4faec8` |
| E_interface（ds） | h1 0.64 / h2 1.10 / h3 1.47 / h4 1.69 / h5 **1.74**；E_real ≈ **0.40**；WM-state action 误差 ≈ 1.43 | `f4faec8`；`round30_audit.md` |
| D2→D3 占比 | ≈ 97%（含 `cde2e51` 保留意见） | Round-32 Level-1 |
| Action Swap matched | top-1 **10.2%** / top-3 63.8%（五候选随机期望 ≈ 20%；误差 spread 4e-3 vs 基准 0.08） | `cde2e51` §一 |
| A_k（Policy 放大） | ×2.1（p50 0.098→0.206）；Lipschitz 尾部 ≈6× | `round32_audit_summary.md` |
| whole-scene mirror | ≈ 0.842（**95% 暂不作为硬门**，只作分指标报告） | `d86e91a` §一.5 |
| 真实 latent 充分性 | ego 0.89–0.98；lane `ld_slots` 0.90/0.93（⇒ 排除 GO_STATE_REPRESENTATION_FIX） | `round32_audit_summary.md` |
| ego 泄漏读数 | OD/LD latent 中 ego speed +19.1%；OD pool ego progress +46.9%；`ld_slots` R² ≈ 0.9–0.99 vs real 0.08–0.14（非因果，见 §2.6） | `82e815b` L111-127；R1 覆盖项 |

### 6.2 架构契约门（含训练/部署同构）

见 §1.6 六条（唯一生成器 / 同参数 $a_0..a_5$ / 每步读 $z_k$ / WM 调用 5 次 / 辅助不进 LQR / 训练部署同构）；交付物 `docs/architecture_contract.md` 必须存在；`plan[:,0]==action_mu` 等表面兼容不构成同构证明。

### 6.3 逐层门 G0–G6（执行优先）

完整判据与数字见 §4.3 表。速查：

| 门 | 关键数字 |
|---|---|
| G0 | 逐位/加载/参数集/坐标系·单位·mask/hook 默认关 bit-exact（8/8 已过） |
| G1 | succ ≥ 0.90；越界 ≤ 2%；collision ≈ 0；timeout ≤ 5%；曲率·变化率 ≤ P95×1.25；饱和低于基线 |
| G2 | 完成 ≥ 90%；越界 ≤ 5%；方向 ≥ 95%；回正 ≥ 90%；反复换道 ≤ 5% |
| G3 | 计划越界 ↓ ≥ 50%；实际 off ↓ ≥ 50%；续接 ≥ 90%；跟踪失败率不上升 |
| G4 | 覆盖 ≥ 95%；岗亭碰撞 ↓ ≥ 80%；提前换道 ≥ 85%；succ ≥ 60%；collision ≤ 基线 |
| G5 | 95% / 95% / 90% / 90% + 无理由减速=失败 |
| G6 | **0.62** / 0.10 / 0.08 / 0.10 / rc 不降（0.64=作废 WP3 门） |

### 6.4 闭环阶梯：calib50 → clean150 → 3-seed → IDM → 超越

| 级 | 判据 | 来源 |
|---|---|---|
| 第一阶段（执行优先 / **calib50**） | success ≥ **0.62**；off ≤ 0.10；collision ≤ 0.08；timeout ≤ 0.10；route completion 不下降。**0.64 = 已作废 WP3 门** | `7c0186f` G6；§5.52/§5.54 |
| 进入 clean150 | success ≥ **0.60**；相对基线提升 ≥ **5pp**；收费站 success **显著非零**；merge off-road 下降 ≥ **50%** | `07f8c4b` L435-457 |
| 正式评测（3 seeds） | mean ≥ **0.68**；worst ≥ **0.64** | `07f8c4b` L456-457 |
| 追平 IDM（0.756） | 3-seed mean ≥ **0.74**；安全：off ≤ 0.098/0.097；collision ≤ 0.144/0.174；统计：paired LB > 0、McNemar $p<0.05$、非单一路型 | `argue §1`（G2）；`07f8c4b` L460-463 |
| 超越目标 | mean ≥ **0.80**；worst ≥ **0.77**；paired LB > 0、McNemar $p<0.05$、非单一路型；安全指标不劣于 IDM；"双集成立"口径**待复核/待澄清是哪两集**（R3 F22 未溯源） | `argue §1`（G3）；`07f8c4b` L463 |

（R33C 单轮硬门 0.58/0.14/0.08/≤11/≥0.862 见 §6.1，属"canary 级单轮门"，与上表阶梯并存、口径不同。）

### 6.5 WM 接口门（量化）

| 判据 | 阈值 | 来源 |
|---|---|---|
| Action Swap top-1 | ≥ 0.35 | `f0ebfae`；`cde2e51` |
| Action Swap top-3 | ≥ 0.85 | 同上 |
| ds-progress 单调率 | ≥ 0.90 | 同上 |
| FD dθ/ds 比 | ≤ 10（ds FD 相对 dθ·κ 的差距从 26–50× 收敛） | `c3bf69f` L754；`cde2e51` Gate 1 |
| E_interface / E_action / overshoot 恶化 | ≤ 3% | `f0ebfae` 硬门 |
| lane-relative 信息 | 不得退化（量化口径：保留率下降 ≤ 5pp，见 Gate 2） | `f0ebfae`；`cde2e51` Gate 2 |
| OD ego-motion 泄漏 | 不得增加（量化口径：增加 ≤ 10%） | 同上 |
| 真实状态 vs WM 状态动作差距 | 不随 horizon 爆炸；以 20%/50% 相对分层带为首轮参考，实测后锁定 | v2；`acf1b8f` §八；R1 O-02 |
| 接口读数锚点 | **B1：接口误差改善 10–12% 但 success 0.56 不变、off-road 0.14→0.20、tollgate success=0 ⇒ 接口门不得当作闭环门** | `82e815b` L20-83；`round33c` |

补充定性项（v2 保留）：action 干预方向正确；动作条件化单调（量化为单调率 ≥ 0.90）；WM 接口验收必须同时报告 E_interface 读数（h1–h5）与比值带。

### 6.6 消融门与梯度门

- **消融门**：旧 PolicyHead + WM rollout / StructuredPolicyHead + WM rollout / 一次性 PathExecutionHead 诊断臂（**仅诊断上界，不得成为生产候选**）。
- **梯度门**：当前动作 loss → PolicyHead；WM 预测 loss → WM；未来辅助 loss → PolicyHead 辅助分支；**辅助 loss 不得绕过 WM 成为第二尾段规划器**。

### 6.7 分层评测与场景专项安全门

- **分层评测**（R2-03）：所有闭环结果必须按 **收费站 / 静态障碍 / merge·ramp / curve / 普通道路**（并可加动态前车）分层报告；不报总分掩盖。
- **收费站专项门**：收费站 success 提升但**收费站 collision 不下降 ⇒ 模型不得晋级**；收费站新增碰撞必须为 0，或至少不超过基线（`49e04d7` L192-223/L385）。
- **静态障碍专项门**：静态障碍停车成功 ≥ 95%（v2 能力门保留）；可碰撞静态体显式表示（§2.7）；阻塞类失败按 §5.8 归因。
- **d_stop / 走廊余量诊断**（R2-03）：$d_{stop}=v^2/(2a_{comfortable})+v\tau$；不需要把这个公式变成硬规则 planner，但作为监督与诊断：**planned progress 是否超过当前可用无碰撞走廊余量**。

### 6.8 导航角色门与自主选道门

**导航角色门**（R3 F4；`07f8c4b` L409-412）——三条必须全部成立：

1. **道路级 route 改变时，candidate 评分应正确变化**；
2. **移除 target-lane token 后，网络仍可自主选择可行车道**；
3. **不得依赖外部 selected lane / gate**。

现况基线：`no_token` 干预下仍自行变道但 **0 成功**（自主选择存在雏形、不可靠）⇒ 该门当前**不达标的起点**（`round35_report` §1 表③）。

**选择门 / 覆盖门**（R3 F2；`07f8c4b` L343-349）：

| 判据 | 阈值 |
|---|---|
| 可行候选覆盖（分场景，合取口径） | 收费站 ≥ 95% / merge ≥ 95% / 路口转向 ≥ 95% / 普通道路 ≥ 99% |
| 安全候选 top-1 | ≥ 70% |
| 安全候选 top-3 | ≥ 95% |
| 收费站错误闸口 | 下降 ≥ 50% |
| merge 错误路径 | 下降 ≥ 50% |
| 路口错误入口 | 下降 ≥ 50% |

推进序门：**候选覆盖不过门不训练 selector**（先修候选生成）；**闭环安全门不过不进 DAgger、MoE 或 RL**。

### 6.9 横向 / 纵向能力门

**横向曲率路径门**（R3 F9；`07f8c4b` L421-427，6 项）：

| # | 判据 |
|---|---|
| 1 | 路径连续 |
| 2 | 曲率有界 |
| 3 | 曲率变化率有界（≤ expert P95×1.25，G1 同口径） |
| 4 | footprint 不碰撞 |
| 5 | 道路内比例 ≥ 99% |
| 6 | 左右镜像方向正确率 ≥ 95% |

dev 集合建议 = 收费站/merge 失败集 + 普通样本。

**纵向门**（原文重写版，R3 F19；`07f8c4b` L429-433）：障碍移近 → 速度/进度下降率 ≥ 95%；TTC 下降 → 计划进度不得增加；曲率增大 → 速度不得增加；**障碍移出路径 → 不应无理由减速**（失败判据）。v2 的"恢复进度 ≥ 90%"作为正向前置的补充口径保留，90% 阈值来源**待复核**（R3 F22）。

**基础车道保持**（v2 保留）：普通道路 success ≥ 0.90 / 计划 footprint 越界率 ≤ 2% / collision 接近 0 / timeout ≤ 5%。

**给定目标后的横向机动**（v2 保留）：合法目标下变道完成率 ≥ 90% / 路径越界率 ≤ 5% / 方向正确率 ≥ 95% / 变道后回正率 ≥ 90% / merge off-road 下降 ≥ 50%。

### 6.10 统计与归因门

- 失败归因覆盖率 ≥ 90%（六分类；见 §5.8）；
- 晋级/停损判定：噪声地板 + episode 级配对 bootstrap 半宽（参考：E_if 6.06pp / E_action 6.12pp / overshoot 8.90pp）；恶化幅度必须高于噪声地板；
- 3 seeds 判定：paired LB > 0、McNemar $p<0.05$、非单一路型；
- 归因口径固定（六分类 + 导航三分拆优先级），不在轮中变更。

---

## §7 审计纪律

### 7.1 改网络前必须核对

任何大改之前必须核对：设计意图 / 代码真实执行 / 数据标签 / 梯度可达性 / optimizer 参数集 / 训练执行器 / 部署执行器 / checkpoint 真实变化。

### 7.2 证据来源纪律

结论必须来源于：**运行时 hooks / 干预 probes / 梯度账本 / 参数更新账本 / 闭环配对实验**。不能根据 README、配置注释或类名推断真实行为。审计须先答"外部规则错还是 Policy 错"（R35）；门失败 ⇒ 停止、按场景归因、写 decision.md（§4.6）。

### 7.3 判断纪律（先分布、后容量）与"过时清单"修订

- **先分布、后容量**（R1 M-15；`910b49b` §二）：出现误差放大时，首先遇到的是 **distribution mismatch**，不是 model capacity mismatch；未排除分布不匹配（训练/部署口径差、接口比 8–10×）之前，**禁止以扩容量/换架构作为对策**（"加 Transformer / 加 latent / 加 memory"证据不支持）；"World Model 要不要重构？暂时不要。"
- **"过时清单"修订**（R1 D-02）：v2 开篇"过时的是把 ds/dθ、外部 target lane、纯 latent World Model 当默认设计"的证据方向不成立，按事实改写为：
  1. 当前瓶颈 = **分布（部署链 8–10×）+ 动作条件化失效**，不是容量/架构；
  2. **ds/dθ/κ 动作空间与 latent WM 在可预见期维持**（AC1 动作特征即 ds/dθ/κ；`rollout_semantic` = 既有最佳口径）；表示层重构（GO_STATE_REPRESENTATION_FIX）已被审计排除（real latent 充分：ego 0.89–0.98、lane `ld_slots` 0.90/0.93）；
  3. 显式几何锚 / 曲率路径迁移属**条件触发项**（动作条件化修复后再复评动态图支路，§2.6）；
  4. 真正作废的是：WP3（`acd22c1`）方案及其 0.64 门、H1 `PathExecutionHead` 的第二生成器形态（§4.10）。

---

## 附录 A：关键数字与来源

| 数字 | 值 | 来源 |
|---|---|---|
| 开发基线（K1e2+MoE-off+K-anchor-off） | 0.56 / 0.140 / 0.080（succ/off/coll） | `d86e91a` §一.4 |
| IDM + Pure Pursuit | 0.756 | `910b49b`；`82e815b` |
| 部署链差距 | 8–10×（0.02–0.04 vs 0.20–0.32） | R30；`f4faec8` |
| D2→D3 占比 | 97%（含保留意见） | Round-32 Level-1；`cde2e51` 二 |
| Action Swap matched top-1 / top-3 | 10.2% / 63.8% | `cde2e51` 一 |
| A_k 放大 | ×2.1（p50 0.098→0.206）；尾部 ≈6× | `round32_audit_summary.md` |
| mirror（whole-scene） | 0.842（非硬门） | `d86e91a` 一.5 |
| 失败归因覆盖率 | 0.947（34A）/ 1.00（35 F1） | `round34a/35_report` |
| 候选覆盖（34B0） | 收费站 10/10；merge 7/7；严格合取 17/17 | `round34b0_report` §3 |
| 外部 target-lane 规则正确率 | episode 0.412；tollgate 0/10 | `round35_report`；`navigation_scope_audit.md` |
| expert5k_v8 场景计数 | 455/441/906/907 | `round35_report` §3 |
| T4 闸口可得率 | 317/455 ep = 69.7% | `round36_report` H2 |
| T4 换道步距闸口 | p50 25.8 m | `round36_report` §2 |
| H2 数据门 | 265,801 usable 行；99.98% 有标签；0 error；自证 15/15 | `round36_report` §2 |
| H1 参数/键 | 总参数 33,433；可训 18 键 / 冻结 164 | `round36_report` §1 |
| Gate 0 自证 | 8/8；966 tests passed；旧 ckpt strict 0/0/0 | `round36_report` §1 |
| H3 hooks | 555 键；默认关三方逐位 | `round36_report` §3 |
| 噪声地板（配对 bootstrap 半宽） | E_if 6.06pp / E_action 6.12pp / overshoot 8.90pp | `round33_report` L40-42 |
| canary B1 硬门 | 6/11 未过 ⇒ 不晋级 | `round33c_report` |
| v2 基线条目 | AGENT.md v2 = 779 行（`df152a6`，末行无换行） | R1 方法注 |

## 附录 B：轮次–commit 索引

| 区间/轮次 | commit | 内容（一句话） |
|---|---|---|
| R30 | `acf1b8f` 等 | Reality Audit 终审：部署链一致性为新主问题；工程契约三等式；"先实测后锁门"方法学 |
| R31 | `1c1d91b` | 正式锁定目标架构 + 范式流水线 A→B0–B4→C0–C4 + Gate A0/B0/B2 + 五条自动停止 |
| R32 | `910b49b` / `d86e91a` / `cde2e51` | 接口审计（8–10×、D0–D3、MoE-off 口径）；三级决策树；动作条件化 canary 四臂与 Gate 0–4 |
| R33 | `c3bf69f` / `f0ebfae` / `49e04d7` | canary 不晋级；执行优先/33C 设计；非补偿+停损+分层安全门+静态障碍教训；三分支决策树 |
| R34–R35 | `82e815b` / `5fb3718` / `07f8c4b` | 路径级归因（94.7%）；输入职责清理（旧摘要退役/几何层 6 项）；导航角色审计 → 网络内候选车道/路径 + 全套门 |
| WP3–R36 | `acd22c1` / `7c0186f` / `ac78f76` | WP3 作废回退；执行优先 G0–G6 + 9 条禁令；R36 修正（认账 H1 偏离 + 四门 + ds_init/κ 数据驱动 + T4 晚换道） |
| 落库 | `df152a6` / `6944436` | v2 守则（779 行）；`net/path_head.py` H1 骨架 |

## 附录 C：变更记录（v2 → v3）

- **结构**：全文重排为 元信息 + §0–§7 + 附录；原"一、原方案提炼"并入 §1–§3 与 §4.2；原"三、Probes"→§5；原"四、验收准则"→§6；原"五、一句话守则"→§0。原文正确措辞保留（可重排/合并/去重），未在第 1 节原文与四份 findings 之外新增内容。
- **加法（缺失补入）**：消融门/梯度门/架构契约门与 `architecture_contract.md`；18 键白名单；ds_init/κ 数据驱动；T4 晚换道；非补偿/停损/硬阻断；分层评测与专项安全门；静态障碍显式表示 + 旧摘要三阶段退役；覆盖门（分场景 99%/合取）/不训 selector；导航角色门；几何层 6 项输出 + 3 条禁输出；9 条禁令 + 冻结 + 解禁；范式流水线 A/B0–B4/C0–C4 + 前置门 + decision.md + 三级决策树 + 自动停止 + Gate F；canary 契约；梯度契约（detach 现状/BPTT 门）；D0–D3；ΔR²；MoE-off 口径；工程三等式；四映射；数据门/DAgger 暴露率；hooks 补齐；"先分布后容量"；失败归因；统计证据；路线图。
- **更正（过时）**：闭环门 0.64 → **0.62**（0.64 标为作废 WP3 门）；补当前基线 0.56/0.140/0.080、IDM 0.756、clean150/3-seed/追平/超越阶梯与 R33C 单轮硬门；动态图支路现状（D1→D2=0.0%、不优先改、不永久关闭）。
- **改写（偏差）**：开篇"过时清单"按证据改写（§7.3）；阈值校准总则前移（§4.1）；κ_near 语义（当前动作 vs 辅助）修正；纵向门措辞与输入补"导航一致性"；几何层输出收口；OD 恒速先验标为待消融；WM 接口门量化；`select_target_lane()` 场景限定；数字来源标注/待复核。
- **登记类**：阶段状态（执行优先 / WP3 终止 / K-anchor 待裁 / 门状态）；实现偏差快照 + 清零前禁训。
