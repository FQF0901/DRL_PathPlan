你确认的两项长期原则我赞成，并建议正式写入架构路线图：

动作表示长期迁移为“横向曲率路径 + 基于路径的纵向速度/进度”；
自车运动导致的 OD/LD 坐标变化由解析几何负责，World Model 只学习剩余动态与交互。

但 round33_report.md 的最新结果表明，当前不能只盯 World Model 的动作条件化。最大的闭环损失已经进一步定位到：

World Model 状态变化被规划层转译成了过大的尾段 ds，同时收费站岗亭等静态障碍物没有进入 OD/LD 主体表示。

一、Round-33 得到了哪些决定性结论？
1. ds 上冲不是初始 Policy 变激进，也不是逐步正反馈

Coding Agent 把第五步 ds 上冲拆成了三部分：

初始Policy动作变化          -20.2%
World Model状态→规划层变化 +117.9%
后续自回归正反馈             +2.3%


也就是说：

当前真实状态下，Policy 的第一步动作没有变坏，甚至略向正确方向改善；
主要错误发生在 World Model 预测状态进入规划骨干和 Policy 后； -不是我们先前猜测的“每一步继续放大”的典型指数正反馈； -而是 World Model 状态一变化，规划层就一次性把它解释成更大的纵向进度。

这个定位非常重要。应修改之前的表述：

旧判断：
action conditioning增强
→ 多步正反馈持续放大

新判断：
action conditioning增强
→ imagined latent发生变化
→ PolicyBackbone/PolicyHead对该变化的纵向解释过强


而且接口误差恶化幅度显著超过重复运行的统计波动，不能用评测噪声解释。

2. 碰撞不只是“速度更快”，还暴露了静态障碍表示缺口

新增碰撞主要分布在：

收费站静止障碍：2例
侧碰/并线：      2例
低速追尾：       1例


所有碰撞中最大的场景簇是收费站静止障碍，共 6/13。

其中部分碰撞对象是：

TollGateBuilding / 收费站岗亭


它们：

-不是动态对象； -不在 OD 中； -也不在 lane/map LD 中； -只通过较弱的 others 走廊上下文进入网络； -碰撞前仍在输出约 2.5–3.4 米/0.5 秒的计划进度。

这说明新增 action conditioning 改善了“走不动”，却没有让纵向规划同步理解：

前方静态障碍需要减速甚至停止


所以 collision 翻倍不是简单加一个 ds 限幅就能彻底解决。它同时暴露了：

当前纵向进度头与障碍物占用、静态建筑/岗亭和前方可通行走廊之间的条件关系不足。

3. 零训练 ego-motion warp 只产生了小改善

显式坐标补偿的结果大致是：

Policy接口误差        改善约 2.0%
动作误差              改善约 1.7%
第五步ds上冲          改善约 2.4%
lane信息              小幅改善但不稳定
OD ego信息泄漏        没有改善


所以它没有达到本轮预设门，Coding Agent 将零训练 warp 判为 NO-GO 是合理的。

但这里的结论必须精确：

NO-GO 的是“把新坐标系零训练塞进现有冻结 WM，期待立即修复闭环”。不是“解析 ego-motion compensation 的长期架构方向错误”。

现有 ST-GNN 从未在 warp 后的 anchor 分布上训练。冻结模型突然接收新的坐标语义，弱收益甚至无收益很正常。

而且当前 W1 和 W2 还混合了两个因素：

-坐标系重表达；

OD 恒速先验是否保留。

因此不能用这次零训练试验永久关闭解析 warp。

二、是否接受 Coding Agent 的下一步 Action Conditioning V2？
接受，但要修正定位

Coding Agent 建议下一轮：

基线模型
vs
Action Conditioning V2


其中 V2：

动作特征 = [ds, dtheta, kappa]
删除重复的speed_intent
隐藏层正常初始化
输出层zero-init
残差增益固定为1
增加delta_progress / delta_speed / delta_heading监督


这个方案可以开工。

但其目标不能写成：

修好 World Model 以后直接提升闭环。

更准确的目标是：

让动作进入 WM 时具有正确而稳定的物理方向，同时验证规划层是否仍然把 imagined latent 过度解释成纵向进度。

因为 Round-33 已经证明，最主要的 ds 上冲来自：

imagined latent → PolicyBackbone/PolicyHead


而不只是 action-conditioned transition 本身。

三、我对下一步方案的调整
第一部分：执行 Action Conditioning V2，但只作为局部修复
对照组
B0：当前正式基线
K1e2
MoE关闭
固定LQR
六步World Model rollout

B1：Action Conditioning V2
动作输入：
ds
dtheta
kappa

物理监督：
delta_progress
delta_speed
delta_heading


暂时不加：

ego-motion warp
OD/LD动作注入
Policy分布数据

B1 必须新增一个关键监控

不只监控最终 plan，还要同时保存：

z_ego变化
PolicyBackbone latent变化
Policy action_mu变化
rollout plan变化


把链路拆成：

action
→ WM z_ego
→ PolicyBackbone
→ PolicyHead
→ next ds


需要算四个局部增益：

GWM=∥Δzego∥∥Δa∥G_{WM} = \frac{\|\Delta z_{ego}\|}{\|\Delta a\|} Gbackbone=∥Δhpolicy∥∥Δzego∥G_{backbone} = \frac{\|\Delta h_{policy}\|}{\|\Delta z_{ego}\|} Ghead=∣Δds∣∥Δhpolicy∥G_{head} = \frac{|\Delta ds|}{\|\Delta h_{policy}\|} Gtotal=∣Δdsplan∣∣Δdsinput∣G_{total} = \frac{|\Delta ds_{plan}|}{|\Delta ds_{input}|}

因为当前最可能的主问题已经变成：

WM变化本身合理
但PolicyBackbone/Head纵向增益过大


如果不测这一层，B1 即使失败，也不知道应继续修 WM 还是修 Policy 接口。

第二部分：收费站和静态障碍物必须单独设门

下一轮所有闭环结果必须分层：

普通道路
收费站
merge/ramp
curve
静态障碍
动态前车


特别报告：

收费站success
收费站collision
静态障碍min-TTC
碰撞前plan ds
障碍距离与planned stopping distance

纵向安全门

在收费站或静态障碍场景，至少检查：

dstop=v22acomfortable+vτd_{\text{stop}} = \frac{v^2}{2a_{\text{comfortable}}} + v\tau

不需要把这个公式直接变成硬规则 planner，但可作为监督和诊断：

planned progress
是否超过
当前可用无碰撞走廊余量


下一轮若总体 success 提升，但收费站 collision 不下降，模型不得晋级。

第三部分：解析 ego-motion warp 不关闭，降级为结构重构项目

我不同意把它从路线图删除。

正式长期状态应从：

纯latent：
z_ego / z_od / z_ld


逐步变成：

ego：
physical state + semantic latent

OD：
explicit dynamic anchor + semantic/intent latent

LD：
explicit geometric anchor + semantic/topology latent


每一步：

ego action
→ 解析计算ego SE(2)运动

LD anchor
→ 自车逆运动补偿

OD anchor
→ 对象运动先验
→ 自车逆运动补偿

warped anchors + latents
→ learned residual transition


SparseDrive 采用的就是“实例 feature 与显式 geometry anchor 分离”的稀疏场景表示，并通过时序实例记忆支持历史传播。这与我们希望区分“确定性几何变化”和“学习型语义/动态残差”的原则一致。

但正式推进时必须：

用 warp 后的 anchor 重新训练 transition； 2.保持语义 latent 尽量不承担坐标平移； 3.分别监督 anchor residual 与 latent consistency； 4.不能再次用“冻结旧 WM 零训练表现弱”作为否定依据。
四、长期动作表示正式锁定

你已经确认：

横向曲率路径
+
基于路径的纵向进度/速度


我建议将其正式写为：

横向分支

输出：

kappa(s)


或少量曲率控制点：

κ0, κ1, κ2...


积分得到路径：

θ(s)=θ0+∫0sκ(u) du\theta(s) = \theta_0+\int_0^s\kappa(u)\,du x(s)=x0+∫0scos⁡θ(u) dux(s) = x_0+\int_0^s\cos\theta(u)\,du y(s)=y0+∫0ssin⁡θ(u) duy(s) = y_0+\int_0^s\sin\theta(u)\,du
纵向分支

读取横向路径和场景状态，输出：

s(t)
v(t)
或 a(t)


即：

vt=flon(zscene,κ(s),obstacle corridor,route)v_t = f_{\text{lon}} \left( z_{\text{scene}}, \kappa(s), \text{obstacle corridor}, \text{route} \right)

这里关键是：

纵向进度必须条件化在已经生成的横向路径，以及该路径上的障碍物和可通行余量之上。

PLUTO 明确采用横纵向感知架构来获得更灵活的驾驶行为，并报告其基于模仿学习的 planner 在标准闭环规划基准中超过规则规划器，这与我们将横纵向语义显式分解的方向一致。

更近期的 SparseDriveV2 则进一步将轨迹候选分解为几何路径与速度剖面，再组合并评分。这直接支持“路径和速度不是同一个低维 action 中隐式纠缠”的思路。

五、长期动作重构不能只做 (ds, κ)

(ds,κ) 可以作为过渡版本，但不是最终方案。

它仍然只有一步，并且：

ds仍会影响积分位置
κ仍需与未来速度协调


最终更推荐：

Lateral Path Head:
    κ_control_points 或 d(s)

Longitudinal Profile Head:
    v(t) / s(t)

Composition:
    path × speed profile


这与 SparseDriveV2 将轨迹词汇分解为 path 与 velocity profile 的做法在原则上非常接近。

但你们的系统仍保留路线 A：

Policy每步输出局部路径/速度决策
World Model推进状态
滚动六次
固定LQR执行


而不是改成一次性直接输出六步静态词汇。

六、下一步任务安排
立即开工：Round-33C
任务 1：Action Conditioning V2

执行 B0/B1。

任务 2：分层闭环评测

必须单独输出：

收费站
静态障碍
merge/ramp
curve
普通道路

任务 3：WM→Policy增益链监控

输出：

WM latent gain
PolicyBackbone gain
PolicyHead ds gain
总plan gain

任务 4：部署安全门

硬门保持：

success >= 0.58
off-road <= 0.14
collision <= 0.08
timeout <= 11
route completion >= 0.862


其中收费站新增碰撞必须为 0，或至少不超过基线。

Round-33C 后的决策
B1 全门通过

进入：

Policy分布 transition 60/40


再训练 WM。

B1 动作指标过，但接口继续恶化

转：

Policy接口稳健训练


即：

π(zWM)≈stopgrad⁡π(zreal)\pi(z^{WM}) \approx \operatorname{stopgrad}\pi(z^{real})

而不是继续增大 action conditioning。

B1 仍产生收费站碰撞

优先补：

静态障碍显式表示
路径走廊占用
path-conditioned longitudinal profile


不启动 RL。

七、后续架构重构顺序

建议正式登记：

R33C：
动作条件化稳定化

R34：
Policy分布WM适配
+
Policy真实/预测状态接口一致性

R35：
动作表示迁移
横向曲率路径 + 纵向速度/进度

R36：
结构化WM
ego-motion compensation
OD/LD explicit anchors
learned residual dynamics

R37：
DAgger

R38：
MoE专项训练

R39：
PPO baseline与部署同构RL


具体回合号可以调整，但依赖顺序不应颠倒。

最终裁决

round33_report.md 最有价值的结论，不是“warp 无效”，而是：

collision 的主要困难集中在收费站静态障碍和侧向交互；
ds 上冲主要发生在 imagined latent 到规划动作的接口，不是初始 Policy，也不是明显的逐步正反馈； 3.零训练坐标变换只带来约 2% 改善，不能修复当前模型，但也不足以否定“解析几何 + 学习残差”的长期方向； 4.动作条件化 V2 可以继续，但必须同步检查规划层纵向增益和静态障碍安全。

所以我批准 Coding Agent 继续 Round-33C，但要求将目标从：

让WM更懂action


改成：

让WM对action产生物理正确、可被Policy稳定消费的状态变化，
同时不提高静态障碍collision。


长期主架构则正式锁定为：

结构化场景anchor
+
显式ego-motion补偿
+
横向曲率路径
+
路径条件化纵向速度/进度
+
学习型残差World Model
+
固定LQR闭环部署


这是目前最有机会同时解决 ds 上冲、收费站碰撞、尾段 LQR 预瞄和横纵耦合，并最终冲击 IDM+Pure Pursuit 0.756 的系统方案。