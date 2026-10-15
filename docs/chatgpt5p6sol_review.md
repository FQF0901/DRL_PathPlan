1. 代码中仍然物理存在旧 PolicyHead 和 path_head

概念上只有一个有效动作生成器，但实现仍是：

Plain Text
1
self.policy
2
self.path_head
显示更多行

启用结构化模式时，旧 PolicyHead 仍会前向计算，只是其 action_mu 被覆盖。

当前模仿学习可暂时接受，但正式命名和长期归属应改成：

Plain Text
1
policy.structured_action.*
显示更多行

至少必须保证：

Plain Text
1
同一forward中只有一个action_mu进入环境、WM和LQR
显示更多行

进入 RL 前必须彻底统一均值和方差来源。

2. oneshot 诊断臂仍保留生产代码路径

保留诊断上界可以，但必须增加 fail-fast：

Plain Text
1
eval/deploy模式如果mode=oneshot
2
→ 直接报错
3
 
4
只有带显式：
5
--diagnostic-oneshot
6
才允许运行
显示更多行

否则未来配置误写时，仍可能复活被否决的一次性六步方案。

3. imagined step 的纵向几何仍是近似值

当前实现中：

Plain Text
1
static_distance：
2
t0距离 - 累计进度
3
 
4
dynamic_distance / TTC：
5
保持t0值
6
 
7
speed：
8
保持t0值
9
 
10
nav_progress：
11
保持t0值
显示更多行

这还不是真正的“路径条件化纵向规划”。

对于 G1 车道保持暂时可以接受；但不得直接用于证明 G5 纵向能力，因为：

-障碍可能不在当前路径上； -动态对象会移动； -自车未来速度会变化； -单纯减累计进度不能表达横向换道后的障碍关系。

4. 首轮训练不应同时开放 Backbone 最后一层

当前白名单是：

Plain Text
1
path_head.*
2
plan_head.fusion.2.*
显示更多行

这会同时改变新动作头和其输入 latent，削弱归因。

建议拆成：

Plain Text
1
J1a：
2
只训练StructuredPolicyHead参数
3
 
4
J1b：
5
只有J1a表现为明显欠拟合，
6
才以0.1倍学习率开放plan_head.fusion.2
显示更多行

不能一开始两者同时训练。

5. 当前横向只是“曲率锚定”，不保证位置收敛

路径从自车原点开始，目标车道的横向误差和航向误差只是输入特征。

必须在训练中显式加入：

Plain Text
1
沿程车道横向误差
2
终点横向误差
3
终点航向误差
4
变道完成度
5
变道后回正
显示更多行

否则可能出现：

Plain Text
1
曲率与目标车道相似
2
但路径始终平行偏离目标车道
显示更多行


下面这版可直接发给 Coding Agent 讨论执行。核心纠偏是：车道保持不是只训练 Policy，而是同时建立 Structured Policy、结构化 World Model、OD/LD 自车运动坐标补偿和六步递归部署链。

下一阶段：车道保持闭环基础重构
一、目标与架构硬约束

本阶段只解决：

给定合法当前车道
→ 六步规划始终保持车道内
→ 路径连续、速度合理
→ 固定LQR可稳定跟踪


必须保持的大框架：

真实状态 z0
→ 唯一 PolicyHead 输出当前一步 a0
→ World Model 预测 z1
→ 同一个 PolicyHead 输出 a1
→ 重复至 a5
→ 固定LQR执行六步计划


硬约束：

PolicyHead 是唯一动作生成器； -不得新增独立 PathHead、尾段生成器或一次性六步规划头；
PlanHead 仅作为 PolicyBackbone/场景融合模块；
World Model 只预测下一状态，不产生动作； -六步动作全部来自同一 PolicyHead 实例的六次调用； -固定 LQR 不修改。
二、方案结构
1. Structured PolicyHead

在唯一 PolicyHead 内部增加两个子分支：

横向分支：
输出当前状态下的局部曲率意图
κ_near / κ_mid / κ_far

纵向分支：
读取局部曲率路径、场景和当前速度
输出当前一步 ds / v / Δv


只有当前一步真正进入部署链：

κ_current
ds_current
dtheta_current = κ_current × ds_current


κ_mid/κ_far 和未来速度只能作为辅助意图或辅助监督，不得直接生成后五步并送入 LQR。

2. World Model 结构化状态

状态由纯 latent 调整为：

EgoState:
  ego physical state
  ego latent

ODState:
  dynamic-object physical anchors
  object intent/dynamics latent

LDState:
  lane/map geometric anchors
  lane semantic/topology latent


每一步先进行确定性几何更新，再学习未知残差。

LD 坐标补偿

根据当前动作解析 ego 的 SE(2) 位姿变化，对静态道路 anchor 做逆运动补偿：

pLD′=R(−Δθego)(pLD−Δpego)p'_{LD} = R(-\Delta\theta_{ego}) (p_{LD}-\Delta p_{ego})
OD 坐标补偿

先按对象运动先验推进，再转换到新的 ego 坐标系：

pOD′=Tego−1(pOD+vODΔt)p'_{OD} = T_{ego}^{-1} (p_{OD}+v_{OD}\Delta t)

World Model 只学习：

对象加减速与交互
lane/map重采样残差
presence/entry
语义与意图变化
解析模型无法解释的剩余变化


禁止把 action embedding 无差别广播到 OD/LD latent。

三、训练流程
Step 1：解析坐标层与状态契约

先不训练。

完成：

ego 运动学积分；
LD 的 ego-motion warp；
OD 的恒速先验与 ego-motion warp； -结构化 anchor schema； -旧状态链作为对照； -所有坐标、单位、mask、slot identity 的契约测试。
Step 2：World Model 车道保持训练

冻结 Policy，训练：

ego transition
OD residual transition
LD residual transition
physical prediction heads
presence/entry


输入：

真实状态
实际执行动作


监督：

下一帧ego状态
下一帧OD anchors与latent
下一帧LD anchors与latent
lane-relative lateral error
lane-relative heading error
curvature


保持当前跨步 detach，暂不启用多步 BPTT。

Step 3：Policy 单步车道保持训练

冻结 World Model，训练唯一 PolicyHead 内部的：

横向曲率分支
路径条件化纵向分支


第一臂不解冻 PolicyBackbone。只有明确欠拟合且接口不过度漂移时，才低学习率开放最后一层。

监督：

真实状态下的当前专家动作
当前局部曲率路径
当前安全进度/速度

Step 4：Policy 预测状态鲁棒训练

继续冻结 World Model。

使用：

Lreal=∥π(zkreal)−akexpert∥L_{\text{real}} = \|\pi(z_k^{real})-a_k^{expert}\|

以及：

Linterface=∥π(z^kWM)−sg⁡(π(zkreal))∥L_{\text{interface}} = \| \pi(\hat z_k^{WM}) - \operatorname{sg}(\pi(z_k^{real})) \|

目标是让 Policy 在略有误差的预测世界中，仍能给出与真实世界一致的车道保持动作。

Step 5：World Model / Policy 交替适配

采用交替更新、梯度隔离：

WM-step：
冻结Policy，用真实transition更新WM

Policy-step：
冻结WM，用真实态监督和预测态一致性更新Policy


禁止：

WM + Encoder + Policy全部解冻
→ 单一总loss共同反传

四、必须部署的 Probes / Hooks

所有工具默认关闭；关闭时必须 bit-exact、不改变 RNG、checkpoint 和 optimizer。

1. 架构调用 Hook

每个 rollout 记录：

PolicyHead实例ID
World Model实例ID
z0...z5
a0...a5
每次Policy调用次数
每次WM调用次数
辅助输出去向
LQR实际输入


自动断言：

动作生成器数量 = 1
a0...a5来自同一个PolicyHead
每个a_k读取对应z_k
World Model推进5次
辅助未来输出未直接进入LQR

2. 坐标解耦 Hook

每一步记录：

ego delta pose
OD anchor warp前/后
LD anchor warp前/后
解析预测
learned residual
真实下一帧anchor
slot ID与mask


并分解误差：

解析坐标变换误差
对象运动先验误差
learned residual误差
最终状态误差

3. Policy Hook

记录：

真实latent
WM预测latent
PolicyBackbone输出
κ_near/mid/far
当前κ
当前ds/v/Δv
最终ds/dtheta
真实态和预测态的动作差

4. World Model Hook

按 horizon 1–6 保存：

z_ego / z_od / z_ld in/out
physical predictions
lane lateral/heading
OD relative position/velocity
presence/entry
latent rank
ego信息在OD/LD中的泄漏

5. 路径与 Footprint Hook

将六步轨迹连续插值到约 0.5–1 米间隔，记录：

x/y/theta
κ与Δκ
道路内比例
首次越界位置
最小道路余量
首次碰撞位置
最小障碍余量


不能只检查六个离散点。

6. LQR 对照 Hook

保存：

六步规划
插值后的LQR reference
实际车辆轨迹
横向/航向误差
raw steer/throttle
控制饱和
计划错误与跟踪错误分类

7. 梯度与更新 Hook

自动输出：

参数名
requires_grad
optimizer group
grad norm
update norm
checkpoint load/change
h1/h3/h6梯度可达性

五、关键干预 Probes

必须包含：

ds增减
κ正负扰动
左右完整场景镜像
LD warp关闭/开启
OD warp关闭/开启
真实future anchors替换
target-lane token置零
Policy读取real latent / WM latent


判断：

ds增加是否使ego progress单调增加
κ左/右是否产生正确方向响应
静态LD在解析warp后是否接近真实下一帧
OD residual是否主要表达对象运动而非ego位移
WM latent是否保留车道保持所需信息

六、验收准则
Gate 0：架构与代码契约

必须全部通过：

唯一动作生成器
同一PolicyHead六次调用
World Model推进五次
无独立尾段生成器
默认关闭新结构时旧模型逐位一致
checkpoint严格加载
optimizer参数集与声明一致
hooks关闭时bit-exact

Gate 1：坐标解耦
静态LD解析warp误差 P90 <=0.10m
heading误差 P90 <=1°
同速前车相对距离误差 P90 <=0.20m
ego progress在OD/LD latent中的泄漏较旧模型下降 >=30%
左右镜像方向正确率 >=95%


阈值若受 simulator schema 限制，可用当前基线分布校准一次，之后锁定。

Gate 2：World Model 六步车道状态
lane lateral误差不随horizon指数增长
lane heading误差不随horizon指数增长
WM tail优于frozen-t0 tail
h3–h6 Policy接口误差较当前基线下降 >=20%
OD/LD物理预测优于恒速/静态基线

Gate 3：Policy 单步与预测态鲁棒性
真实状态当前动作误差不退化超过5%
WM状态与真实状态动作差 h3–h6下降 >=20%
左右曲率方向正确率 >=95%
ds-progress单调率 >=90%
κ变化不超过专家P95×1.25

Gate 4：六步计划质量
连续footprint越界率 <=2%
无障碍碰撞接近0
计划Δκ不超过专家P95×1.25
重规划方向翻转率 <=5%
后五步不存在系统性ds上冲

Gate 5：普通道路闭环

使用固定 LQR：

普通道路success >=0.90
off-road <=0.02
collision接近0
timeout <=0.05
route completion不下降
steering saturation显著低于当前基线


必须同时报告 ExactTracker 上界，不得用 Exact 指标替代部署闭环。

七、本阶段禁止事项
不做自主车道/闸口选择
不新增独立PathHead
不一次输出六步部署轨迹
不修改LQR
不启动DAgger
不启用MoE
不启动PPO/GRPO
不扩World Model容量
不启用完整六步BPTT
不全量端到端解冻

八、通过后的下一顺序
1. 固定路径纵向跟车、停车和恢复加速
2. 给定目标车道后的变道、merge和道内避让
3. 网络内部自主车道、收费闸口和出口选择
4. DAgger同时适配Policy与World Model
5. MoE专项专家
6. PPO及部署同构评测

给 Coding Agent 的开工结论

下一阶段只攻克完整部署链上的基础车道保持：唯一 Structured PolicyHead 每次输出当前一步曲率与路径条件化进度，结构化 World Model 使用解析 ego-motion 对 OD/LD anchors 做坐标补偿并学习残差，同一 Policy 与 World Model 递归六次，最后由固定 LQR 执行。先分训，再交替适配，禁止独立轨迹头和一次性六步尾段。