下面可直接发给 Coding Agent。核心调整是：不再单线优化 World Model，下一阶段优先建立网络内部的车道/闸口选择与横纵向解耦规划；World Model 随后适配新的动作与状态表示。

下一阶段：自主车道选择与路径级规划重构
一、目标与原则
最终目标

将当前：

道路级导航
→ 外部规则选择target lane
→ Policy输出(ds, dtheta)
→ World Model滚动6步
→ LQR


逐步改为：

道路级导航
+ 结构化车道/静态障碍/动态对象
→ 网络内部生成并选择车道/闸口候选
→ 横向曲率路径
→ 路径条件化纵向速度/进度
→ World Model滚动6步
→ 固定LQR

锁定原则
外部只提供道路级路线与客观环境信息；
车道、闸口、变道时机和候选路径选择必须由网络完成；
几何变换、碰撞检测、车辆 footprint 扫掠属于确定性物理算子，不属于外部决策；
不再强化旧的 5 维静态障碍摘要；
本阶段先重构 Policy 规划能力，不继续单线扩大 World Model；
固定 LQR，不通过调控制器掩盖规划问题。
二、为什么这样调整

已有证据表明：

收费站失败：10/10存在可行候选路径
merge失败：7/7存在可行候选路径


但现有 Policy 没有生成或选中这些路径。

同时，当前代码存在：

道路级nav
+
外部规则select_target_lane
+
target-lane相对语义


因此网络目前更接近“跟踪外部指定车道”，并未完整学会：

根据道路导航自主选车道
自主选收费站闸口
决定merge保持还是并线
决定路口前提前进入哪条车道


World Model 动作条件化已经改善约 10% 的接口误差，却没有提高总体成功率，说明当前最大闭环杠杆已转移到路径生成与选择。

三、工作包 1：导航与 target-lane 责任审计

先只读排摸，不训练。

需要输出
docs/navigation_scope_audit.md
runs/round35/navigation_scope/

审计内容

列出所有相关输入：

nav
route_world
current_ref_lanes
next_ref_lanes
target_lane
target_lane_token
LD route association
lane semantic
route command


每个字段记录：

来源
道路级/车道级
是否由规则生成
是否使用模拟器真值
最终部署是否可获得
网络中由哪个模块消费

三组干预
干预一
道路级route固定
交换或修改target lane


观察：

Policy路径
首步动作
六步轨迹
候选车道选择

干预二
target lane固定
改变道路级route或转向命令


确认 Policy 是否真正消费道路级导航。

干预三
移除target-lane token
保留道路级route和全部车道几何


确认当前模型是否具备任何自主车道选择能力。

验证标准

必须能拆分：

收费站失败：
外部target lane选错占比
target lane正确但Policy路径选错占比
Policy知道障碍但未改道占比

merge和路口：
同样拆分


至少 90% 的失败 episode 必须落入明确类别。

四、工作包 2：输入与数据动态桩函数

所有 probe 默认关闭，开启后：

前向结果bit-exact
不改变RNG
不写入checkpoint
不改变训练行为

输入桩

每步记录：

字段名
原始值
归一化值
shape
mask
单位
坐标系
时间戳
schema版本


重点：

nav
target-lane语义
LD车道slots
OD动态对象slots
静态对象slots

网络消费桩

每层记录：

输入/输出mean、std、min、max
有效slot比例
activation norm
effective rank
gradient norm
parameter update norm


重点模块：

Nav Encoder
Lane Semantic Encoder
LD Encoder
Static Object Encoder
Policy Backbone
Policy Head
World Model

Rollout 桩

记录：

真实latent
World Model latent
Policy backbone latent
候选路径分数
横向曲率
纵向速度/进度
六步plan
LQR reference
实际执行轨迹

干预桩

支持：

字段置零
slot置换
左右镜像
距离扰动
route交换
target lane交换
静态障碍移入/移出候选路径

五、工作包 3：网络内部候选车道与路径生成

不再让外部规则输出最终目标车道。

输入
道路级nav与route
全部可见车道/连接车道slots
显式静态对象slots
动态对象slots
自车状态


外部可以提供：

车道几何
车道连接关系
是否连接全局route
静态障碍几何


但不得提供：

selected_lane
selected_gate
best_candidate
recommended_lane_change

第一版候选集合

至少包含：

保持当前车道
左侧合法连接车道
右侧合法连接车道
下一路段route continuation


收费站根据实际可见 lane/闸口生成候选，不由外部选择最佳闸口。

横向表示

每条候选输出三个曲率控制点：

near curvature
mid curvature
far curvature


插值为：

κ(s)\kappa(s)

再积分得到：

x(s)
y(s)
heading(s)


第一版外部仍转换成兼容格式：

dθt=κtdstd\theta_t=\kappa_t ds_t

固定 LQR 和环境接口不变。

六、工作包 4：路径条件化纵向规划

纵向模块不能再只根据全局 latent 输出一个 ds。

每条候选路径应读取：

路径曲率
静态障碍最小距离
动态对象TTC
可通行路径长度
道路边界余量
导航一致性
车辆当前速度


输出：

未来6步s(t)
或未来6步v(t)


原则：

先确定横向路径
再基于该路径决定速度和进度


这样收费站岗亭只有在位于候选路径上时才触发减速，避免全局障碍摘要造成错误加速或无条件减速。

七、确定性几何评估器的边界

几何层只计算客观属性：

footprint collision
minimum clearance
first invalid path position
drivable ratio
static obstacle clearance
dynamic TTC


不得直接输出：

最佳车道
最佳闸口
最终候选编号


由网络根据几何属性和道路级导航选择最终候选。

八、训练流程
第一步：候选覆盖验证

先不训练 selector，验证候选集合是否包含正确解。

数据至少覆盖：

收费站 >=100 episodes
merge >=100
路口左转/右转 >=100
普通道路 >=200

门
存在至少一条安全且route一致候选：
收费站 >=95%
merge >=95%
路口转向 >=95%
普通道路 >=99%


不过门先修候选生成，不训练 selector。

第二步：模仿学习

监督：

可行候选集合
专家采用候选
横向曲率路径
纵向速度/进度
候选评分


多个安全闸口都可作为正例，不强制唯一答案。

Loss 分开报告：

lane/gate choice
path geometry
speed profile
candidate score
collision margin
route consistency

第三步：World Model 适配

新 Policy 动作表示稳定后，再更新 World Model：

横向曲率路径状态
纵向速度/进度状态
结构化OD/LD anchors


同时实施：

LD anchor的ego-motion坐标补偿
OD运动先验 + ego-motion坐标补偿
World Model只学习残差动态

第四步：DAgger

每步同时保存：

道路级route
候选集合
Policy选择
专家选择
专家路径
实际执行动作
真实下一状态


分别用于：

Policy：learner状态下的候选与路径监督
World Model：learner状态下的真实transition监督

九、验证标准
导航角色门
道路级route改变时，candidate评分应正确变化
target-lane移除后，网络仍可自主选择可行车道
不得依赖外部selected lane/gate

候选选择门
可行候选top-1 >=70%
top-3 >=95%
收费站错误闸口下降 >=50%
merge错误通道下降 >=50%
路口错误入口下降 >=50%

横向路径门
路径连续
曲率有界
曲率变化率有界
footprint不碰撞
道路内比例 >=99%
左右镜像方向正确率 >=95%

纵向门
障碍移近 → 速度/进度下降率 >=95%
TTC下降 → 计划进度不得增加
曲率增大 → 速度不得增加
障碍移出路径 → 不应无理由减速

闭环开发门

相对当前约 0.56：

calib50 success >=0.64
off-road <=0.10
collision <=0.08
timeout <=0.10
route completion不下降


进入 clean150：

success >=0.60
相对基线提升 >=5pp
收费站success显著非零
merge off-road下降 >=50%


正式评测：

3 seeds mean >=0.68
worst seed >=0.64


之后再通过 DAgger 和 RL 冲击：

追平IDM >=0.74
超越IDM >=0.80 mean

十、当前冻结项

本阶段不得同时执行：

PPO/GRPO
MoE重训
LQR修改
World Model扩容
全模型端到端解冻
外部最佳车道规则
外部最佳闸口规则
新增人工走廊分数

给 Coding Agent 的一句话指令

下一阶段先完成导航职责审计和动态 hooks，再构建“道路级导航输入、网络内部生成并选择车道/闸口候选、横向曲率路径、路径条件化纵向速度”的最小原型；外部只提供客观几何和道路级路线，不提供最终车道答案。候选覆盖不过门不训练 selector，闭环安全门不过不进入 DAgger、MoE 或 RL。