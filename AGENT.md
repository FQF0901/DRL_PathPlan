你贴出的初版方案核心没有过时，真正应保留的是“唯一 Policy、单步决策、World Model 递归展开、固定 LQR 部署、分阶段训练、用动态证据裁决”。过时的是当时仍把 ds/dθ、外部 target lane、纯 latent World Model 当成可长期维持的默认设计。

下面先提炼原方案，再补成新的“架构与训练守则 v2”。

一、原方案的核心提炼
1. 唯一决策源

系统中只有一个动作生成器：

ak=πϕ(zk)a_k=\pi_\phi(z_k)

即：

PolicyHead 是唯一动作输出模块


明确禁止：

PlanHead输出另一条轨迹
独立PathHead一次生成六步
K-anchor生成尾段
Projector成为第二规划器
World Model生成动作


PlanHead 只是历史命名，其真实职责是：

PolicyBackbone / SceneContextEncoder


它负责融合场景，不负责输出另一套计划。

2. 六步计划必须由同一 Policy 递归产生
真实状态 z0
→ PolicyHead 输出 a0
→ World Model 预测 z1
→ 同一个 PolicyHead 输出 a1
→ World Model 预测 z2
...
→ 得到 a0...a5


数学上：

ak=π(zk),z^k+1=WM(z^k,ak)a_k=\pi(z_k),\qquad \hat z_{k+1}=WM(\hat z_k,a_k)

所以：

Policy 每次只决定当前一步； -六步来自同一 Policy 的六次调用； -每一步都必须读取对应的真实或预测世界状态； -任何辅助未来预测都不能绕过这个递归链直接进入 LQR。
3. World Model 与 Policy 职责分离
World Model

负责：

z^k+1=fθ(zk,ak)\hat z_{k+1}=f_\theta(z_k,a_k)

只回答：

执行动作后，世界会变成什么状态？

包括：

自车未来状态
动态对象未来状态
车道与道路相对状态
对象出现、消失
场景上下文演化

PolicyHead

负责：

ak=πϕ(zk)a_k=\pi_\phi(z_k)

只回答：

面对当前世界，下一步应如何规划？

接口必须满足：

π(z^kWM)≈π(zkreal)\pi(\hat z_k^{WM}) \approx \pi(z_k^{real})

即 World Model 不需要逐值复制真实世界，但必须保留 Policy 决策所需的信息。

4. Stage A/B/C 原始职责
Stage A：训练 World Model
真实状态 + 已执行动作
→ 预测真实下一状态


重点训练：

状态转移
动作条件化
物理状态预测
多步滚动稳定性

Stage B：训练 Policy
真实状态 → 专家动作
WM预测状态 → 决策一致性
learner状态 → 恢复动作


先 Primary，再 DAgger，再 MoE。

Stage C：强化学习优化 Policy

规划端强化学习使用：

Policy单步动作
→ ExactTracker
→ 真实环境转移
→ reward
→ PPO


部署仍是：

Policy–WM递归六次
→ 六步轨迹
→ 固定LQR

5. 固定控制器

LQR 冻结，不通过调控制器掩盖规划缺陷。

始终分别报告：

ExactTracker：
规划能力上界

World Model rollout + 固定LQR：
真实部署性能


只有后者可以用于和 IDM + Pure Pursuit 比较。

6. 审计优先于改网络

任何大改之前必须核对：

设计意图
代码真实执行
数据标签
梯度可达性
optimizer参数集
训练执行器
部署执行器
checkpoint真实变化


结论必须来源于：

运行时 hooks
干预 probes
梯度账本
参数更新账本
闭环配对实验


不能根据 README、配置注释或类名推断真实行为。

二、更新后的架构与训练守则 v2
守则 1：PolicyHead 仍是唯一动作生成器

长期动作表示从：

(ds, dθ)


迁移为：

横向曲率路径意图
+
基于该路径的纵向速度/进度


但必须收编在同一个 PolicyHead 内部：

PolicyBackbone
    ↓
Structured PolicyHead
    ├─ 横向曲率分支
    └─ 路径条件化纵向分支
    ↓
当前一步action


每次调用只输出当前一步：

κ_k
ds_k 或 v_k


对外转换：

dθk=κkdskd\theta_k=\kappa_k ds_k

然后交给 World Model 推进一步，再重新调用同一个 PolicyHead。

禁止：

独立PathExecutionHead一次输出六步
横向头直接生成部署尾段
未来速度辅助头直接进入LQR


近、中、远曲率或未来速度可以作为辅助意图和监督，但最终部署六步仍必须来自六次 PolicyHead → WM 调用。

守则 2：横纵向显式解耦，但纵向必须条件化于路径

旧动作：

ds + dθ


存在严重耦合：

ds既表示进度和速度
又是曲率κ=dθ/ds的分母
还改变LQR预瞄尺度


长期应改为：

横向分支

输出局部曲率路径：

κ_near
κ_mid
κ_far


或：

κ(s)\kappa(s)

通过确定性积分得到局部几何路径。

纵向分支

读取：

横向路径曲率
路径道路边界余量
该路径上的静态障碍
该路径上的动态对象与TTC
可通行长度
当前速度
道路级导航


输出当前一步：

ds_k / v_k / delta_v_k


即：

横向决定“从哪里走”，纵向决定“沿这条路径走多快”。

不能做两个互不交流的横纵头，因为弯道、变道、绕障必然影响安全速度。

守则 3：确定性坐标变化不得继续由 latent 硬学

World Model 中需要把显式几何与语义 latent 分开。

建议状态结构：

Ego:
  physical state
  semantic latent

OD:
  explicit object anchor
  intent/dynamics latent

LD:
  explicit lane/map anchor
  semantic/topology latent


每一步首先解析计算 ego 的 SE(2) 运动：

平移
旋转


然后：

LD

静态车道 anchor 做自车逆运动补偿：

pLD′=R(−Δθego)(pLD−Δpego)p'_{LD} = R(-\Delta\theta_{ego}) (p_{LD}-\Delta p_{ego})

道路真实曲率不因坐标变换而改变。

OD

先应用对象运动先验，再做自车坐标补偿：

pOD′=Tego−1(pOD+vODΔt)p'_{OD} = T_{ego}^{-1} \left( p_{OD}+v_{OD}\Delta t \right)

World Model 只学习：

对象加减速
cut-in
yield
merge交互
地图重采样残差
可见性及slot变化


不能继续让 z_od/z_ld 大量编码“ego前进了多少”，否则会产生语义串扰。

但正式切换坐标规范必须配合重新训练，不能把新坐标 zero-shot 塞给旧 WM 后，根据约 2% 改善就否定该方向。

守则 4：外部只给道路级导航，不给车道级答案

允许外部提供：

全局道路路线
未来道路序列
前方路线顶点
直行/左转/右转命令
路口和道路拓扑
到转向点的距离


禁止生产系统外部提供：

唯一target lane
最佳收费闸口
推荐变道方向
selected candidate
规则生成的最优轨迹
规则速度答案


当前 select_target_lane() 只能：

作为教师辅助
作为诊断上界
作为课程训练中的给定正确目标


不得成为最终部署的强制车道决策。

网络必须内化：

车道选择
收费闸口选择
merge保持或并入
路口提前选道
变道时机
绕障方向

守则 5：客观几何计算不等于决策外包

以下可以在网络外解析计算：

坐标转换
曲率积分
车辆footprint扫掠
路径与障碍物最小距离
道路边界相交
动态对象TTC
路径可通行长度
候选与全局route拓扑连接关系


这些只是客观事实。

网络仍应负责：

生成哪些候选
选择哪条车道或闸口
如何修正横向路径
速度如何分配
是否变道
何时变道


不允许外部模块根据几何结果直接替网络挑选最佳候选。

守则 6：能力必须按层级训练

训练顺序正式锁定为：

第 0 层：执行契约

核对：

Policy输出
World Model输入输出
六步递归
LQR reference
实际执行

第 1 层：车道保持

给定合法车道，训练：

直道保持
弯道保持
曲率连续性
路径可跟踪性

第 2 层：固定路径上的纵向能力

训练：

跟车减速
静态障碍停车
安全恢复加速
曲率条件化速度
避免timeout

第 3 层：给定目标后的横向机动

不给网络做车道选择，只考察：

左变道
右变道
merge续接
道内避让
变道后回正
给定正确收费闸口后的通过能力

第 4 层：自主车道和通道选择

外部只给道路级导航，网络生成和评分：

保持候选
左侧候选
右侧候选
route continuation候选
收费闸口候选
路口出口候选

第 5 层：World Model 适配新动作与新状态表示

每完成一层 Policy 能力，同步验收：

π(zkWM)≈π(zkreal)\pi(z_k^{WM}) \approx \pi(z_k^{real})
第 6 层：DAgger 与 recovery

覆盖 learner 非常规状态。

第 7 层：MoE

Primary 稳定后再训练场景专家。

第 8 层：RL

最后优化长期成功、安全、效率和多候选选择。

核心原则：

先证明网络能执行正确决策，再训练网络自己做该决策。

守则 7：Stage A/B 不是一次性串行关系，而是先分训、后交替适配
Stage A 初始训练

冻结 Policy，训练 WM：

真实状态 + 实际动作
→ 真实下一状态

Stage B 初始训练

冻结 WM，训练 Policy：

真实状态 → 专家动作

Stage B 后期接口训练

WM 继续冻结，训练 Policy：

Lreal=∥π(zreal)−aexpert∥L_{real} = \|\pi(z^{real})-a^{expert}\| Linterface=∥π(zWM)−sg⁡(π(zreal))∥L_{interface} = \| \pi(z^{WM}) - \operatorname{sg}(\pi(z^{real})) \|

使 Policy 学会在模糊但决策等价的 predicted state 上稳定决策。

后期交替训练
World Model 更新步

冻结 Policy：

learner真实状态
+ 实际执行动作
→ 真实下一状态

Policy 更新步

冻结 World Model：

真实状态专家监督
+
预测状态决策一致性


先交替更新、梯度隔离。

禁止一开始就：

WM + Encoder + Policy全部解冻
→ 一个总loss端到端反传


防止 WM 和 Policy 通过不真实 latent 互相迁就。

守则 8：DAgger 同时服务 Policy 和 World Model

同一次 learner rollout 保存：

obs_t
policy_action_t
expert_action_t
executed_action_t
obs_t+1
road class
outcome
plan[6]
actual ego pose delta


分别用于：

Policy
obs_t → expert_action_t

World Model
obs_t + executed_action_t → obs_t+1


特别注意：

WM 必须使用环境实际执行的动作，而不能使用专家建议但未执行的动作。

DAgger 是两者共同的数据源，但必须：

不同标签
不同loss
不同optimizer step

守则 9：Primary、MoE 与 RL 顺序不变
Primary

先掌握：

车道保持
基础跟车
普通弯道
给定目标变道
基本恢复

MoE

只负责：

收费站
merge/yield
roundabout
复杂路口
特殊避障
高级恢复


不能让专家补偿日常 lane keeping 缺陷。

RL

第一阶段：

冻结WM
Policy + Value
ExactTracker PPO


每个 checkpoint 必须同时验收：

ExactTracker
WM rollout consistency
固定LQR闭环


RL 真实 transition 可用于后续离线更新 WM，再重新适配 Policy 接口。

GRPO 仅作为未来候选级排序辅助，不能替代 PPO 的逐步信用分配。

守则 10：固定 LQR 和双口径评测

LQR 保持冻结。

所有阶段必须分别报告：

ExactTracker：
规划能力上界

固定LQR：
部署闭环能力


最终只以：

Policy–WM递归六次
→ 六步计划
→ 固定LQR


与 IDM + Pure Pursuit 比较。

三、强制 Probes、Hooks 和桩函数

所有工具必须：

默认关闭
关闭时bit-exact
不改变RNG
不进入checkpoint
不改变optimizer

输入和数据契约

记录：

字段来源
坐标系
单位
时间戳
mask
slot identity
schema版本
是否真值
是否规则生成
是否部署可得

Policy

记录每一步：

真实latent
WM latent
PolicyBackbone latent
横向曲率输出
纵向速度/进度输出
最终当前action
辅助未来意图

World Model

记录：

action_in
ego/OD/LD latent in/out
OD/LD anchor warp前后
解析坐标变换
learned residual
physical prediction
presence/entry

六步递归

记录：

z0...z5
a0...a5
每一步Policy实例ID
每一步WM调用
累计轨迹


并自动断言：

a0...a5来自同一个PolicyHead
每个a_k读取对应z_k
WM实际推进5次
辅助头没有直接进入LQR

路径和控制

记录：

曲率路径
速度剖面
30点LQR reference
连续footprint扫掠
首次越界点
首次碰撞点
最小道路余量
最小障碍余量
实际车辆轨迹
raw steer/throttle
饱和比例

梯度和参数

记录：

requires_grad
optimizer group
grad norm
update norm
checkpoint loaded/changed
h1/h3/h6梯度可达性

四、更新后的验收准则
架构硬门
动作生成器数量 = 1
所有六步来自同一个PolicyHead
不存在独立tail generator
不存在独立PathHead部署输出
World Model只预测状态
固定LQR只消费递归六步结果

基础车道保持
普通道路success >=0.90
计划footprint越界率 <=2%
collision接近0
timeout <=5%

固定路径纵向能力
障碍更近 → 计划速度不升高 >=95%
TTC更低 → 计划进度不增加 >=95%
曲率增大 → 速度不升高 >=90%
障碍移除 → 恢复进度 >=90%
静态障碍停车成功 >=95%

给定目标后的横向机动
合法目标下变道完成率 >=90%
路径越界率 <=5%
方向正确率 >=95%
变道后回正率 >=90%
merge off-road下降 >=50%

自主选道
可行候选覆盖 >=95%
安全候选top-1 >=70%
安全候选top-3 >=95%
收费站错误闸口下降 >=50%
merge错误路径下降 >=50%
路口错误入口下降 >=50%

World Model 接口
真实状态动作与WM状态动作差距不随horizon爆炸
action干预方向正确
动作条件化单调
lane-relative信息不丢失
OD/LD ego-motion泄漏不增加

综合闭环

第一阶段：

success >=0.64
off-road <=0.10
collision <=0.08
timeout <=0.10
route completion不下降


追平 IDM：

3-seed mean >=0.74


超越目标：

mean >=0.80
worst seed >=0.77
双集成立
安全指标不劣于IDM

五、最终版一句话守则

系统只有一个 PolicyHead，它每次根据当前真实或 World Model 预测状态，输出当前一步“横向曲率意图 + 路径条件化纵向动作”；同一 PolicyHead 与 World Model 递归六次形成部署轨迹。外部只提供道路级导航和客观场景事实，不提供目标车道、最佳闸口或最终候选。确定性坐标转换、碰撞几何和车辆运动学由解析层负责，世界残差由 World Model 学习，车道选择和速度决策由 Policy 学习。训练按车道保持、纵向安全、给定目标机动、自主选道、WM适配、DAgger、MoE、RL 的顺序推进。

这版既保留了原方案最重要的架构纪律，也纳入了之后审计得到的真实结论，足以作为后续 Coding Agent 不得偏离的总守则。