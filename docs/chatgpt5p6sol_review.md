你的担心成立。根据 round36_report.md，Coding Agent 当前实现的 PathExecutionHead 已经出现了两处明显偏离原始大框架的地方：

新增了一个独立的轨迹执行头，命名和职责都容易与唯一 PolicyHead 冲突；
启用后一次性生成后续六步尾段，不再采用“同一个 PolicyHead + World Model”逐步自回归。

报告自己也明确写了：

启用后 rollout 尾段由组合层一次给出
不再逐步策略自回归


这已经不是小的实现细节，而是从路线 A 向“一次性轨迹头”路线发生了结构偏移。

一、先重申不可破坏的架构原则

我们之前约定的是：

PolicyHead：唯一动作生成器

a0 = PolicyHead(z0)
z1 = WorldModel(z0, a0)

a1 = 同一个 PolicyHead(z1)
z2 = WorldModel(z1, a1)

...

a5 = 同一个 PolicyHead(z5)


最终：

[a0...a5]
→ 连续轨迹
→ 固定LQR


其中：

World Model 只预测世界；
PolicyHead 只根据当前或预测世界决定下一步； -六步轨迹由同一个 PolicyHead 被调用六次形成； -不得再有第二个模块直接生成完整尾段。

所以“横向曲率路径 + 路径条件化纵向速度”应该是：

PolicyHead 的内部动作参数化方式改变了。

而不是：

在 PolicyHead 旁边新增一个 PathExecutionHead，一次输出完整六步。

二、PathExecutionHead 当前具体有什么问题？
1. 它事实上是第二个轨迹生成者

报告描述该模块负责：

3个曲率控制点
6步纵向进度
曲率积分
组合为ds/dtheta六步plan


这就是完整轨迹生成，而不是单纯的几何转换层。

因此即使它名字叫“执行头”，实际上职责是：

trajectory generator


这与“PolicyHead 是唯一轨迹决策源”的原则冲突。

2. 它绕过了 World Model rollout

路线 A 的价值在于：

每一步动作
→ 改变预测世界
→ Policy重新观察预测世界
→ 决定下一步


当前新头一次性输出后续六步，意味着：

后五步不再读取逐步更新的：
ego状态
周车状态
车道状态
未来交互状态


这样 World Model 对尾段决策失去作用。

相当于从：

Policy–WM–Policy–WM


变成：

Policy/PathHead一次输出整条轨迹


这会破坏我们坚持的主范式。

3. plan[:,0] == action_mu 只是表面兼容

报告强调：

plan[:,0] == action_mu


但这不能解决职责冲突。

真正的问题不是第一步是否一致，而是：

a1...a5由谁生成？


当前实现中：

a0来自旧PolicyHead
a1...a5来自PathExecutionHead


仍然是两个生成口。

4. 纵向头使用“外源8维”需要警惕

报告写到纵向模块使用：

κ控制点 + 外源8维


但没有在总报告中清楚说明这 8 维分别是什么、来自哪里、是否包括：

-人工走廊摘要； -外部目标车道； -障碍感知规则； -候选路径可行性结论； -外部推荐速度。

在没有逐字段契约前，不能允许其进入正式训练。

必须清楚区分：

允许：
客观距离、TTC、边界余量、路径曲率

禁止：
最佳通道、推荐换道方向、外部安全评分、规则速度答案

三、正确的架构改法
PolicyHead 仍然唯一，但内部改成结构化动作头

建议将当前实现改名并重构为：

StructuredPolicyHead


内部可以有两个子分支：

LateralActionBranch
LongitudinalActionBranch


但它们共同构成一个 PolicyHead，不是独立规划器。

每一步的输出可以改为：

横向：
局部曲率参数 κ_t
或短局部路径参数

纵向：
基于该局部路径的 ds_t / v_t


然后得到：

at=(dst,κt)a_t=(ds_t,\kappa_t)

对外兼容转换：

dθt=κtdstd\theta_t=\kappa_t ds_t

World Model 再根据这个单步 action 推进：

z_(t+1) = WM(z_t, a_t)


下一步继续调用同一个结构化 PolicyHead。

六步仍然必须递归生成

正确链路：

z0
→ StructuredPolicyHead
→ κ0, ds0
→ 转换成ds0,dθ0
→ WM
→ z1

z1
→ 同一个StructuredPolicyHead
→ κ1, ds1
→ WM
→ z2
...


这样既实现：

横向曲率
+
路径条件化纵向进度


又不破坏：

Policy–WM 六步自回归

四、三个曲率控制点还能不能保留？

可以，但语义必须收缩。

可以作为“单步局部路径意图”

Policy 在每个 imagined state 上输出：

κ_near
κ_mid
κ_far


其中：

κ_near 决定当前 0.5 秒动作；
κ_mid/far 是局部路径意图或辅助预测； -真正执行并推进 World Model 的仍然只有当前动作； -下一步重新在新 predicted state 上计算一组控制点。

这种形式近似 receding-horizon planning：

每一步都预测一个短局部路径
只执行第一段
状态更新后重新规划


这是合理的。

不允许作为一次性六步尾段生成器

不能：

t0预测κ_near/mid/far
→ 一次性积分生成a0...a5
→ 后续不再经过WM和Policy


否则就是第二个轨迹头。

五、纵向分支应该如何读取横向路径？

在每一步 tt，PolicyHead 内部先产生短期横向意图：

κ_near/mid/far


基于这个局部路径，确定性几何层计算：

局部路径曲率
静态障碍余量
动态对象TTC
道路边界余量
可通行距离


纵向分支再输出当前：

ds_t 或 v_t


然后只执行当前动作：

(ds_t, κ_t)


所以路径条件化发生在 PolicyHead 内部，但每次仍只产生一个环境 action。

六、哪些现有成果可以保留？

Round-36 并不是全部推倒。

以下内容可以保留。

1. 曲率路径数学层
曲率插值
曲率积分
x/y/theta生成
ds/dtheta兼容转换


这些是确定性几何，不是第二规划器。

2. 数据标签

已经提取的：

车道保持
变道
merge
收费站正确闸口
路口转向
曲率残差
纵向进度


仍可用于训练结构化 PolicyHead。

3. Hooks

新增的 555 项 hooks，包括：

曲率
轨迹
footprint
纵向速度
TTC
边界余量
LQR跟踪
网络激活与梯度


都值得保留。

4. 给定目标车道的课程设计

先练：

车道保持
→ 给定目标变道
→ merge
→ 给定正确闸口


顺序仍然正确。

七、我与当前实施的其他分歧点
1. 不应默认一次输出六步 progress_s

当前纵向头直接产生六步进度，相当于一次性定义完整速度剖面。

长期可以将六步速度作为辅助监督，但环境主动作仍应只取当前步：

ds0


其余：

ds1...ds5


只能作为：

-辅助预测；

consistency loss； -短期意图； -不能直接替代 WM rollout 尾段。
2. 初始 ds=3.5m/步 太强，需有专家分布依据

报告中纵向头初始化为：

3.5m / 0.5s
≈ 7m/s


这可能对普通道路合理，但对：

收费站
急弯
低速merge
静态障碍前


可能过于激进。

应按场景或当前速度初始化：

ds_init = clamp(v_current × dt, bounds)


而不是统一 3.5m。

至少需验证：

初始化ds与专家ds分布的P50/P90

3. tanh × 0.1 曲率残差可能过强或过弱
∣Δκ∣≤0.1 m−1|\Delta\kappa|\leq0.1\,m^{-1}

对应半径约：

10m


对高速普通道路可能偏大，对收费站急转或低速绕行又可能不足。

建议根据：

专家曲率残差P95/P99
道路类型
当前速度


确定范围，或做速度条件化限制：

κmax⁡(v)\kappa_{\max}(v)

不能只固定 0.1。

4. 收费站标签存在“晚变道模仿”风险

报告已经发现：

专家开始换道距离闸口中位数约25.8m


直接模仿可能学会晚换道。

不建议通过人工“首步约束”直接规定转向动作。

更合理的是：

-对早期可决策状态增加采样权重； -监督目标车道方向与路径进度； -加入“剩余距离是否足够完成变道”的可行性损失；

DAgger 后期采集 learner 状态下的提前变道专家标签。
5. 当前阶段不应训练收费站自主闸口选择

Round-36 给定的是正确闸口，现阶段先验证：

给对目标后能不能执行


这是对的。

但必须确保该目标仅用于课程训练和能力上界，不进入最终部署输入。

八、立即要求 Coding Agent 修改
1. 暂停将PathExecutionHead接入正式rollout训练。

2. 将其重构/重命名为StructuredPolicyHead的内部子模块：
   - lateral branch
   - longitudinal branch
   - deterministic composition layer

3. PolicyHead保持唯一动作生成者。

4. 每个World Model step只消费当前动作：
   a_k = (ds_k, kappa_k)
   dtheta_k = kappa_k * ds_k

5. a_(k+1)必须由同一个PolicyHead读取z_(k+1)后重新产生。

6. κ_mid/κ_far和未来progress只能作为辅助意图/监督，
   不得直接替代后五步Policy-WM rollout。

7. 保留所有曲率积分、footprint、LQR、数据标签和hooks。

8. 输出一份architecture_contract.md，明确：
   - 唯一动作生成模块
   - WM输入输出
   - 六步rollout调用图
   - 辅助头与主动作的边界
   - 训练与部署是否完全同构

九、验证准则
架构契约门

必须满足：

动作生成器数量 = 1
所有a0...a5都来自同一个PolicyHead参数
每一步a_k都读取对应z_k
World Model被调用5次推进未来状态
辅助未来输出不直接供LQR

梯度门
当前动作loss → PolicyHead
WM预测loss → World Model
未来辅助loss → PolicyHead辅助分支
辅助loss不得绕过WM成为第二尾段规划器

消融门

至少比较：

旧PolicyHead + WM rollout
StructuredPolicyHead + WM rollout
一次性PathExecutionHead诊断臂


一次性头只能作为诊断上界，不得成为生产候选。

闭环门

仍按：

普通保持
给定目标变道
merge续接
给定正确闸口


逐层过门。

最终判断

你的担心是正确的。

Round-36 总报告 中的 PathExecutionHead 当前实现已经超出了“PolicyHead 内部动作重参数化”，变成第二个六步轨迹生成器，并绕过了后五步 Policy–World Model 自回归。

应立即把方向拉回大框架：

唯一Structured PolicyHead
→ 当前局部曲率路径意图
→ 路径条件化当前纵向动作
→ World Model推进一步
→ 同一PolicyHead重新决策
→ 重复六次
→ 固定LQR


这样才能同时保留：

-唯一 Policy 决策源；

World Model 的真实作用； -横纵向解耦； -六步滚动； -固定 LQR 部署； -后续 IL、DAgger、MoE 和 RL 的一致训练契约。