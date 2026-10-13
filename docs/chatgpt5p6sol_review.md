我看完了这轮攻防（Round-30）。先给结论：

我总体支持 Coding Agent 本轮的反驳，且认为他的方向比我上一版“直接推进 Stage-A WM Adaptation”更稳。

但我认为他低估了一个问题：

Stage B 的 rollout-supervision 契约本身可能已经是主瓶颈，而不只是 Stage A 的 WM。

所以我的态度是：

70% 接受 Coding Agent
20% 补充
10% 反驳


而不是完全转向他，也不是坚持我原来的 Audit 顺序。

一、我认同 Coding Agent 的部分
1. 他对 Stage A 的理解更准确

我之前为了快速建立 Reality Audit，把 Stage A 简化成：

一步 transition
+
逐步调用


但代码事实不是这样。

根据他核查的：

h1~h6 全部进入 loss
h2~h6 输入是模型自己预测的 latent
共享 transition 权重被每个 horizon 监督


所以更准确叫：

free-running rollout
+
truncated credit assignment


而不是：

one-step training


这个纠正我接受。

2. C7 的发现非常重要

这个是我认为本轮最大的成果。

他已经证明：

当前 Stage C 不是：

ExactTracker


而是：

sampled action
→ repeat 6次
→ LQR


部署却是：

Policy-WM rollout
→ plan
→ LQR


这意味着：

Stage C
≠
部署系统


所以任何 PPO 收益都存疑。

这个发现比继续调 loss 值重要得多。

3. Reality Audit 的优先级应该高于继续训练

我同意他：

R4
WM action fidelity
oracle-state
frozen-t0


这些都是空白区域。

如果这些不测清楚：

DAgger
PPO
MoE

继续推进风险极大。

二、我认为他低估的问题
1. Stage B 很可能比 Stage A 更危险

这是我最想补充的。

Coding Agent 目前聚焦：

WM预测状态
→ Policy动作


的一致性。

但从当前 repo 的事实来看：

Stage A teacher-forced
↓
Stage B predicted-latent rollout
↓
traj6 label是真实未来轨迹


这里已经天然有一个问题。

因为：

z^k≠zkexpert\hat z_k \neq z_k^{expert}

但监督却要求：

Policy(预测状态)
→ 真实专家未来轨迹


这和之前的：

Pass1 obs
×
Pass2 label


本质上属于同一类风险。

只是更隐蔽。

我担心什么

如果：

WM rollout已经偏离


那么：

traj6 supervision


很可能在教 Policy：

在错误状态上模仿正确状态的动作


这个是结构性矛盾。

所以我认为：

R4 不只是审计 Stage A。

也是在审计：

Stage B rollout supervision 是否合法

三、关于路线 A，我现在的想法

你已经明确：

坚持路线A


我支持。

但是我会修正一个非常关键的点：

路线 A 不等于“先把 WM 训准”

这是过去几轮最大的思维惯性。

你原始设计其实是：

WM
+
Policy
+
rollout


共同组成系统。

而不是：

先把WM做到100分
再训练Policy


因为：

WM(z,a)WM(z,a)

里的：

aa

本来就是 Policy 输出。

更接近 Dreamer 的理解

理想状态是：

Policy产生动作
WM预测结果
Policy继续决策


一起闭环。

Dreamer 也是这样。

所以：

审计阶段结束后，

我不建议追求：

让WM单独达到极低误差


而是：

WM是否足以支持Policy rollout

四、关于你的终极目标

你现在目标是：

超过 IDM + Pure Pursuit


而不是：

把某个loss做到最低


所以我建议重新排序 KPI。

KPI Tier 1

闭环：

success
off-road
collision
timeout
route completion


这是唯一最终目标。

KPI Tier 2

部署链：

WM rollout consistency
Plan Δκ
steer saturation
replan consistency

KPI Tier 3

训练：

latent loss
OD/LD loss
ego_next
teacher MAE


现在很多讨论还停留在 Tier 3。

真正能决定是否超过 IDM 的，其实是 Tier 1 和 Tier 2。

五、我建议给 Coding Agent 的回复

我会回复：

总体接受 Reality Audit 方向。

重点不是继续训练，而是先回答三个问题：

Q1:
Policy(real state) 与 Policy(WM state)
差多少？
(E_interface h1~h6)

Q2:
Stage B rollout supervision 是否存在状态-标签错配？
(预测状态 vs 专家真实未来轨迹)

Q3:
Stage C 训练系统与部署系统差多少？
(Exact/repeat/plan × LQR)

如果 Q1 很大：
优先修 WM action conditioning。

如果 Q1 不大，但 Q2 很大：
优先修 Stage B rollout supervision。

如果 Q1/Q2 都不大，而部署仍差：
优先修 Stage C 执行口径。

在这三个问题没有实证回答前：
冻结 DAgger、MoE、RL 和大规模数据采集。

最终裁决

我对本轮攻防的最终评分：

Reality Audit方向：9/10
Stage A判断：8/10
Stage C发现：10/10
Stage B风险意识：6/10
整体路线：8.5/10


所以：

批准 Coding Agent 继续执行 Round-30 Reality Audit。

但要求额外加入：

Stage B rollout supervision 合法性审计


因为我现在认为：

未来动作标签与预测状态的潜在错配，可能和当初 recovery_v1 的错配一样，属于整个系统的一级风险。

如果这个问题存在，那么后续：

WM
DAgger
PPO
MoE


全都会建立在一个不一致的监督链上。这个风险必须优先排除。