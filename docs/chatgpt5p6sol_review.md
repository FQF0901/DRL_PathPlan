这轮我认真看完了。

先给结论：

我基本站 Coding Agent 这边。

而且我觉得这轮讨论的质量明显高于前面几轮。

因为双方终于开始围绕“真实代码行为”而不是“理想架构”辩论。

事实上，经过 R25→R30 这么多轮后，我认为已经出现了一个很大的认知转折：

我的观点发生了什么变化

在 Round-18 以前，我认为：

root cause
≈
lane semantic缺失
+
covariate shift


到 Round-25 左右，我认为：

root cause
≈
lane semantic没被action采用


但看到 Round-30 审计后，我认为最大的问题已经变成：

训练链
≠
部署链


而且不止一处不一致。

我认可 Coding Agent 的三条核心反驳
反驳1：Q1 已经回答了

他指的是：

π(z^kWM)≈?π(zkreal)\pi(\hat z_k^{WM}) \stackrel{?}{\approx} \pi(z_k^{real})

R30已经测出来：

E_real ≈ 0.40
E_interface:
h1 0.64
h2 1.10
h3 1.47
h4 1.69
h5 1.74


这是非常糟糕的。

而且：

oracle-state action误差
≈ 0.40

WM-state action误差
≈ 1.43


所以：

Policy 在真实状态上没那么差。

真正炸的是：

Policy(预测状态)


这一层。

这点我同意 Coding Agent。

这已经不是假设。

这是事实。

反驳2：Q3 已经回答了

这是更重要的。

他把：

Stage C


拆成三套系统：

S1
Exact

S2
WM plan + LQR

S3
repeat(a_t) + LQR


结果：

S3
≈ 0


而：

S2
≈ 0.52


差距巨大。

所以：

Stage C 当前训练系统

根本不是部署系统。

这个结论已经成立。

无需继续空谈。

反驳3：Reality Audit 大于继续调Loss

我也赞同。

因为目前你已经有：

lane semantic
residual
mirror
amplitude anchor
kappa residual
tail projection


但核心问题仍然没解决：

World Model
+
Policy
+
Rollout


三者怎么对接。

所以继续加：

loss
loss
loss


边际收益一定越来越低。

我最认可的一段

就是 Coding Agent 最后那句话：

Stage B 的风险
不等于 recovery_v1 的错配

而是

预测状态链上的间接监督


这个判断非常重要。

这其实是什么问题？

它非常像：

Scheduled Sampling 所解决的问题。

训练时：

真实状态
→ expert action


部署时：

预测状态
→ policy action


状态分布变了。

而你现在 Stage B 有点像：

Policy(predicted_state)
被要求
接近
expert(real_state)


这里其实天然有 gap。

但又不像 recovery_v1 那么离谱：

obs_t
配
label_t+30


所以我同意 Coding Agent：

不能直接把它定性为 recovery_v1 同级 bug。

但：

它绝对值得成为一级审计对象。

这部分我觉得双方已经趋于一致。

我最不同意 Coding Agent 的地方

只有一个。

他还是低估了 Stage B

我继续坚持：

Stage A
问题已经比较清楚

Stage B
可能比想象中更危险


为什么？

因为现在：

WM teacher-forced
表现不错

但 rollout
表现很差


有两种解释：

解释A

World Model 不够好

解释B

Policy 在 rollout state 上从来没被正确训练过

R30 现在只证明：

A 一定存在


但没证明：

B 不存在


所以：

Q2

我认为必须做。

而且优先级不低。

关于你的路线A

我支持继续坚持。

甚至我觉得现在比一个星期前更应该坚持。

为什么？

因为现在已经证明：

单步Policy
不够


以及：

后5步
会显著影响当前控制


这是 R28 F4 和 R30 都证明的。

所以：

路线B
=
局部tail generator


虽然短期可能更稳定。

但会直接放弃：

Stage A 世界模型
Stage C model based RL


这一整套路线。

我更推荐

把路线A修正为：

A-R
World Model真实可用

B-R
真实状态
+
预测状态
联合IL

C-R
单步Policy RL
+
部署迁移验证


而不是推翻路线A。

如果我是 Coding Agent

我今晚不会做任何训练。

我会做：

R31-1

Q2审计

核心：

pi(real_state)
vs
pi(predicted_state)
vs
expert


按 horizon 1~5。

R31-2

World Model Action Fidelity

重点：

不同action
→
未来latent是否明显变化


因为现在：

1e-4


级别的响应太弱了。

这个必须厘清。

R31-3

Stage C Contract

确认：

collect
update
eval
deploy


四个 action mapping 是否一致。

我对下一轮的裁决
批准
Reality Audit继续
Q2审计
WM action fidelity
Stage C contract审计

暂停
A4 adaptation
action token
2-step BPTT
DAgger
MoE
PPO
GRPO

我现在对 root cause 的最新排序
P0

训练链和部署链不一致

Stage C
≠
Deploy

P1

WM rollout state

和

real state

动作分布严重分叉

E_interface
远大于
E_real

P2

Stage B 在预测状态上的监督是否合法

目前未知

P3

lane semantic

已经不再是主要矛盾

R21-3 以后，它已经从：

根因


降级成：

已解决事项


所以如果让我给 Coding Agent 一个最终攻防回复：

我接受你对 Q1/Q3 已经基本回答的判断。

我坚持 Q2 必须提升到一级审计对象。

在 Q2 完成前，不启动任何新的主训练。

因为现在最大的风险已经不是模型不会看车道，而是：

Policy究竟在什么状态分布上被监督


这件事如果没搞清楚，后面无论 DAgger、WM 适配、PPO，都会建立在可疑监督链之上。