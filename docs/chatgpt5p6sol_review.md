我审阅了 deepseekv4p1_argue.md。总体上，P0 结论和 P1 v0.4 已经很扎实，但文档对上一轮“Action-conditioned plan”的收拢还没有真正完成：它仍锁定为“2D (ds,dθ) 强制作为 plan 首步，再生成后五步”，这与我们后来讨论的“PPO action 应能控制完整轨迹”并不等价。

一、可以直接锁定的部分
1. P0 结论可靠

现有实验证据足以支持：

当前模型的主要短板是 plan/action 质量，而非 LQR；
repeat_action 只是 PPO 信用链工具，不适合部署；
闭环重规划明显优于开环专家动作链；
模型主要死于 off-road/压线，D2 主要死于 timeout，IDM 主要死于 collision；
exact 不优于 LQR，说明“更准确执行坏计划”不会解决问题。

这些结论可以停止反复讨论。

2. P1-0 至 P1-3 基本可执行

以下路线合理：

补测 rou64@e015 的 eval500；
做 D2 配速 V1/V2/V3；
上 A 批 probes；
从 student 真实状态采 recovery；
训练真实整段 recovery trajectory；
加 LD 几何代理、引擎真值审计、曲率和进度约束；
使用 R1/R2 检验 experts 是否应参与 recovery 学习。
二、必须重新打开的分歧：P1-4 不能照当前 v0.4 直接实施

文档当前写的是：

2D 采样动作强制为 plan 首步，PlanHead 根据 latent + z 生成其余五步。

这只能算弱 Action-conditioned plan。

它虽然比当前 A-hold 更接近部署链，但仍有两个根本问题。

1. 一个首步 (ds,dθ) 不能充分表达三秒规划意图

相同首步动作可以对应：

保持车道；
一秒后开始换道；
缓慢减速后停车；
穿过收费站后向中心收敛；
先让行，再恢复速度。

如果后五步主要由 PlanHead 自己决定，PPO 的 sampled 2D action 对整条计划的控制权很弱。

形式上：

Pt=fϕ(st,at)P_t=f_\phi(s_t,a_t)

但如果：

∂Pt,2:6∂at≈0\frac{\partial P_{t,2:6}}{\partial a_t}\approx 0

那么 PPO 虽然能计算正确的 log π(a|s)，却很难通过 ata_t 有效改善远期计划。

2. “plan[0] 等于 sampled action”不等于 PPO 控制完整 plan

当前网络已有 plan[0]=mu 类契约，但这只说明首点一致，不能证明：

sampled action 显著改变后续五步；
action 能控制速度、横向偏移、制动时机和轨迹形状；
PPO 能修复中后期逐渐恶化的 plan。

而 P0 显示模型的 T_plan 通常不是第一个策略步立即出错，而是在闭环推进一段时间后逐渐恶化。只强化首点可能继续修不到核心问题。

三、建议把 P1-4 改成两级实施
P1-4A：2D action-conditioned plan，仅作为因果烟测

可以保留当前方案，但降低定位：

它是验证 PPO→PlanHead→LQR 信用链的最小实验，不是主性能方案。

必须增加四个 probe：

后五步敏感性

对同一状态改变 sampled action，记录：

delta_plan_step_1
delta_plan_steps_2_to_6
delta_terminal_pose
delta_curvature_profile
delta_target_speed_profile


要求后五步不能几乎不变。

有效控制维数

在固定状态上采样不少于 256 个 2D action，计算完整计划的 PCA 或 Jacobian 敏感性。

如果计划变化只集中在：

首点；
单一角度缩放；
单一速度缩放；

则 2D action 无法覆盖有意义的计划空间。

反事实单调性

例如：

增大 ds → 终点纵向距离应总体增加
增大 dtheta → 终点横向偏移方向应正确


不能出现符号反转或后五步抵消首步。

执行一致性

必须记录：

sampled_action
decoded_plan
actual_LQR_reference
actual_motion


确认 PPO 采样变量没有在后续被 mu、clipping 或 deterministic path 覆盖。

若这些 probe 不通过，P1-4A 应立即停止，不进入完整三 seed 训练。

P1-4B：Mode + low-dimensional plan latent，作为主性能方案

建议 PPO action 定义为：

at=(mt,zt)a_t=(m_t,z_t)

其中：

离散 mode

首版不宜过多，建议 3–5 类：

follow_route
shift_left
shift_right
yield_or_decelerate
stop


使用道路几何和 route legality mask，避免在不允许换道时采样换道模式。

连续 latent

建议 4–6 维，表达：

target speed / speed delta
terminal lateral offset
braking timing
longitudinal aggressiveness
trajectory shape factors


Conditional PlanHead 输出完整：

6 x (ds, dtheta)


PPO 记录完整 joint log-prob：

log⁡π(m,z∣s)=log⁡π(m∣s)+log⁡π(z∣s,m)\log\pi(m,z\mid s) = \log\pi(m\mid s)+\log\pi(z\mid s,m)

这才是一条真正的 trajectory macro-action。

四、当前阶段不要直接做 GRPO

你的多候选、反事实 rollout、组内比较方向适合作为长期架构，但当前 repo 还不具备直接上 GRPO 的基础。

原因一：当前 WM 尚未证明能做反事实排序

当前 WM：

逐步 detach；
主要训练 latent/ego/OD/LD 预测；
没有验证周车对不同 ego trajectory 的响应；
没有候选轨迹 pairwise ranking 精度；
没有 unsafe false-negative 审计。

如果直接用 WM 对 G 条轨迹排序，policy 很可能学会利用 WM 偏差。

原因二：Best-of-G 选择会改变行为策略概率

如果：

从 policy 采 G 条
→ scorer 选一条
→ 执行选中轨迹


实际行为策略已不是单条原始 policy。不能直接拿选中候选原本的 log π 做标准 PPO，否则有 selection bias。

原因三：当前首先要证明“完整 plan 可被 PPO 改善”

在多候选搜索之前，应先证明：

单条 sampled plan latent
→ 完整计划
→ LQR
→ 闭环 reward
→ PPO 更新


确实能稳定提升性能。

五、GRPO/AlphaGo 路线应作为 P2/P3

推荐渐进顺序：

第一步：Best-of-G oracle gap

固定 policy，对同一状态采 G 条计划，用真实 MetaDrive 分支 rollout 离线评价。

比较：

single-sample performance
best-of-4 oracle
best-of-8 oracle


如果 best-of-8 仅比单样本好 1–2pt，说明候选缺乏多样性，没必要开发复杂 scorer。

如果 oracle gap 达到 10–15pt，才说明搜索值得投入。

第二步：训练 candidate scorer

用真实 simulator rollout 监督：

candidate return
collision
off-road
route progress
comfort


评价：

pairwise ranking accuracy；
Spearman correlation；
unsafe false-negative rate；
按 tollgate、roundabout 等 road class 分层。
第三步：蒸馏搜索结果

先采用监督蒸馏：

policy 生成 G 条
→ simulator/scorer 选优
→ 将优选 mode/latent 蒸馏回 policy


这比直接把 LLM GRPO 搬进驾驶更容易验证。

第四步：Group-ranked policy improvement

最后才考虑：

候选级 group advantage；
真实策略 step 的 Critic + GAE；
absolute safety gate；
group-relative objective。

候选级 group advantage 不能代替真实时间步上的 GAE。

六、对当前文档另外三点修订
1. “P1 冻结 PPO scope”与 P1-4 冲突

文档一方面写 P1 不动 PPO scope，另一方面 P1-4 又需要：

新 action head；
PlanHead 条件化；
放开相关 PlanHead 参数；
改 rollout/buffer/log-prob 契约。

建议改成：

P1-1 到 P1-3 不改 PPO scope；P1-4 建立独立 experimental scope，不修改历史 design scope 的语义。

避免新方案悄悄污染旧配方。

2. P1-4 的启动门槛不能只看 success 0.60–0.65

还应要求：

action-to-plan sensitivity PASS
plan diversity PASS
decoder reconstruction PASS
executed-plan identity PASS
safety projection rate < 5%
clipping rate < 5%


否则 PPO 可能在退化 action space 中训练。

3. Recovery decoder 必须为后续 latent plan 留出覆盖空间

如果 P1-2 使用强单模态均值监督，PlanHead 可能把相似场景的多个有效恢复方案平均成一条差轨迹。

建议训练时至少保留：

多模态 recovery 标签；
Best-of-N 或 Winner-Takes-All 头；
或按 coarse behavior mode 分组。

否则 P1-4B 加 latent 时，decoder 本身可能已经 mode collapse。

七、建议更新后的正式计划
P1-0 至 P1-3

按 v0.4 执行，仅增加：

P1-2 数据中保留恢复行为 mode；
PlanHead 预留 conditional interface；
不要把 recovery 多解全部平均。
P1-4A：2D 因果烟测

目标不是超 IDM，而是验证：

PPO sampled action
→ 后五步计划明显改变
→ LQR 实际执行
→ reward 可归因


只跑小规模 clean150、1 seed。

通过才继续。

P1-4B：Mode + latent full-plan PPO

作为实际性能 candidate：

mode K=3–5
latent D=4–6
decoder -> 6x2 action chain
explicit arc integration
LQR executes first 0.5s
replan every 0.5s


先冻结 decoder，只训练 mode/latent policy 和 value；随后根据 coverage 限制渐进解冻 decoder 尾层。

P2：Best-of-G 与 scorer

先做真实 simulator 的 oracle gap，再决定是否投入 learned WM ranking。

P3：Group-ranked policy improvement

在 scorer 可信、generator 有多样性后，再探索 group advantage + GAE，而不是现在直接做 GRPO。

最终裁决

Coding agent 的最新文档整体可以批准进入 P1-0，但 P1-4 尚未完成收拢。

当前写法中的“2D action-conditioned plan”应该：

保留为最小因果正确性实验；
不应直接作为主性能方案；
更不能视为最终轨迹级 PPO。

主性能方向应调整为：

PPO 采样 behavior mode + 低维 plan latent，条件解码完整 3 秒计划，LQR 只执行首 0.5 秒并持续重规划。

而你提出的：

多模态候选 → 反事实 rollout → 安全/价值评估 → 组内比较 + 时间级 GAE → 蒸馏

应当成为明确的 P2/P3 目标。但在当前 WM 还没有候选排序证据、基础 PPO 轨迹信用链还未打通之前，不应直接跳过去。

建议给 coding agent 的定稿指令是：

批准 P1-0 至 P1-3 按 v0.4 实施。

P1-4 重新拆分：

P1-4A：
2D action-conditioned plan 仅作因果烟测。
必须验证 sampled action 对后五步计划、终点 pose、
曲率和速度曲线具有足够敏感性。
不满足则停止，不做完整训练。

P1-4B：
主性能方案改为 behavior mode + 4–6D plan latent。
PPO 对 mode+latent 的 joint logprob 记账，
conditional decoder 输出完整 6x2 计划，
LQR 执行该计划首 0.5s，随后重规划。

P2：
先用 MetaDrive 真实分支 rollout 测 best-of-G oracle gap；
只有候选多样性和 oracle gap 足够，才训练 candidate scorer。

P3：
scorer 的 pairwise ranking、unsafe false-negative 和分层指标
通过后，再做 group-ranked policy improvement + GAE，
不要直接照搬 LLM 原生 GRPO。


这既保留了 coding agent 倾向的低风险渐进路线，也真正吸收了你提出的“轨迹是 macro-action、多模态反事实搜索、候选级和时间级双层信用”框架。