审阅结论

deepseekv4p1_argue.md 对上一轮方案的受理总体合理，能够达成约 85% 共识。它正确接受了“先补可复现证据，再训练横向能力，同时并行闭合 WorldState”的排序，也主动指出了跨帧平滑、teacher-forcing 反事实合法性和底层状态 residual 预算等风险。

但有三点需要反驳或收紧：

不能继续把 R40 的 V4B 结果写在“已执行结论”中。
 既然正式注入路径未入库，这些结果只能登记为“未入库研究脚手架的暂定发现”，不能再与可复现结果并列。
不能把横向训练简单前置成唯一性能主线。
 当前 LD/OD/ego 状态消费者尚未闭合，Policy 可能继续依赖外部 lane anchor。横向训练和 WorldState 消费链审计必须并行。
不建议通过重放旧动作序列来解决 teacher-forcing 的动作条件一致性。
 在存在环境随机性、其他交通参与者和闭环状态偏差时，第二遍重放未必得到相同世界。更可靠的是 simulator snapshot/clone 分支，或者只在真实记录 transition 上做一步替换。
一、Coding Agent 正确理解并接受了什么？
1. 正确接受了阶段排序

它将下一轮拆成：

证据链正式化
→ 横向瓶颈训练
→ 统一 WorldState
→ Memory / detach
→ learner-state 数据后置


这与我们的方案一致。尤其是明确规定：

未过 Clean500 安全门
不得将 analytic_learned 设为默认
不得启动 MoE、DAgger、PPO 或 GRPO


这是正确的工程纪律。

2. 正确接受了 V4B 证据不可直接采信

它明确承认：

eval_runner.py 只有 protocol hash 参数； -没有真实模型注入路径； -实际闭环注入依赖未入库的 .slim 脚手架； -正式复现前不能接受“决定性否证”。

这和我们的审阅一致。

3. 正确接受了训练工具仍不自包含

它已经把以下内容列为 R41 第一批硬任务：

pipeline/residual_dataset.py
pipeline/residual_features.py
正式runner
正式config
入库小fixture


并要求移除 .slim 硬依赖，核心 CI 不得再因资源缺失而跳过。

这是必须完成的。

4. 正确接受了几何训练 A/B

它没有再把“legacy 几何碰撞更低”解释成 legacy 更正确，而是接受：

Policy-L：
legacy几何训练 + legacy评测

Policy-C：
corrected几何训练 + corrected评测

然后交叉测试


这才能区别：

几何定义差异
Policy适应差异
zero-shot分布迁移

5. 对跨帧曲率一致性提出了合理补充

Coding Agent 指出普通的：

∥κt−Warp⁡(κt−1)∥\|\kappa_t-\operatorname{Warp}(\kappa_{t-1})\|

可能惩罚合理避障与重新规划，因此建议根据状态变化门控。这个反驳成立。

更准确的监督应是：

Lreplan=w(ΔS)⋅∥κt−Warp⁡(κt−1)∥L_{\text{replan}} = w(\Delta S) \cdot \| \kappa_t- \operatorname{Warp}(\kappa_{t-1}) \|

其中：

道路、障碍和目标基本不变
→ 较强一致性约束

cut-in、障碍变化、路线变化
→ 降低一致性权重

安全边界突然收缩
→ 允许快速重规划


不能为了降低变号率，把 Policy 训练成僵直跟踪器。

6. 对可用标签的判断合理

它确认现有数据包含：

专家六步动作计划
路径锚点
横向误差
航向误差
有效性mask


因此可以派生：

κ_near / κ_mid / κ_far
横向偏移
航向误差
计划内部平滑


无需为了第一轮横向训练立即重采数据。

但“跨重规划帧一致性”仍需连续相邻决策窗口和准确 ego pose 对齐，不能只靠单条六步计划内部标签完成。

二、需要反驳或修正的内容
1. R40 V4B 不应继续出现在“已执行结论”区

文档一方面承认 V4B 没有正式接线，另一方面仍在“R40 验证批结论”中写：

oracle替换不能改善Hard
LD真·零消费者
根因最终转向Policy横向


这会让后续开发者继续把暂定发现当作正式裁决。

建议文档将其拆成：

正式可复现结论：
- 第一阶段训练/评测特征同源
- 双运动学链数值近似一致
- zero-shot fixed几何对冻结Policy不利
- ws_learned改善timeout但恶化安全

未入库研究观察：
- oracle OD/LD替换无收益
- ego oracle撤销纵向收益
- 横向命令5倍过量
- 75%方向翻转
- 99.1%全油门窗口高转向


第二组必须加醒目标记：

UNVERIFIED_IN_REPO


在正式 V4B 完成前不能用于终止 WorldState 方向。

2. Teacher-forcing 的“第二遍重放动作”仍不够严谨

Coding Agent 建议：

第一遍记录学生动作和状态
第二遍重放记录动作
并注入记录状态


这比“学生动作 + 专家 future state”好，但仍有问题：

-交通参与者可能有随机性； -仿真内部状态未完全由动作和随机种子决定； -浮点和控制周期偏差会逐步累积； -注入记录状态本身改变闭环； -第二遍状态未必真是当前当遍动作产生的状态。

优先级应是：

最优方案
simulator snapshot / clone
→ 从同一完整环境状态分支
→ 分别执行不同动作
→ 获取合法真实next state

可接受方案

只做 one-step：

真实obs_t
+ 真实executed action_t
→ 真实obs_(t+1)


不要把“重放未达到不动点的六步链”称为动作条件一致的 oracle。

诊断方案

如果只能两遍重放，必须报告：

replay pose divergence
OD state divergence
traffic RNG一致性
termination一致性
每步状态误差


超过阈值的 episode 不得纳入 oracle 裁决。

3. 横向 Policy 训练不能脱离消费者图审计

Coding Agent 倾向先跑最小横向训练臂，这可以接受，但必须先回答：

横向分支究竟读取了哪些状态？


否则很可能继续训练一条依赖：

外部 lane_anchor
旧 target lane
t0 route context


的路径，而 World Model 的 LD latent 和 anchors仍无作用。

横向训练前至少要完成一个轻量 State Consumer Graph：

对κ_near/mid/far分别计算：
- ego latent敏感度
- OD latent敏感度
- LD latent敏感度
- lane_anchor敏感度
- target lane token敏感度
- Memory敏感度


如果结果确认：

LD latent sensitivity ≈ 0
lane_anchor sensitivity很高


那么 Policy 横向训练应同步修正输入结构，否则只是强化外部 lane-anchor shortcut。

4. “底层状态 residual”还需要更小的第一版

Coding Agent 接受了完整 ego/OD/LD residual，但目前一下覆盖：

ego speed / acceleration / yaw rate
OD position / velocity / heading / presence
LD geometry / validity


风险较高。

建议第一版只做最有可识别性的：

ego:
  delta_speed
  delta_yaw_rate

OD:
  delta_position
  delta_velocity
  presence

LD:
  暂不学习运动残差
  只做由ego-motion得到的解析重表达
  validity作为独立分类监督


Lane geometry 本身是静态地图，不应让小网络自由修改它来补偿 Policy 缺陷。只有证明确有重采样误差，才增加 LD geometry residual。

5. LQR 可执行性先作为重加权指标是合理的，但必须避免标签循环

Coding Agent 建议先不做可微 LQR loss，而是用：

预计steer
饱和率
跟踪误差
速度损失


进行 batch 重加权。这个顺序合理。

但不能把所有高 steer 样本都降权，否则最需要学习的：

急弯
掉头
收费站
匝道


反而会被忽略。

正确做法是区分：

场景本身需要高转向
vs
计划抖动导致的不必要高转向


建议重加权依据是：

相对专家LQR执行差距


例如：

w∝max⁡(0,satstudent−satexpert)w \propto \max \left( 0, sat_{\text{student}} - sat_{\text{expert}} \right)

而不是单独根据 student 的 steer saturation。

三、对执行排序的裁决

我同意 Coding Agent 提议的总体顺序，但需加一个前置小门。

第一批：证据链正式化

必须先完成：

1.正式 oracle/teacher-force/speed 注入； 2.移除 residual runner 对 .slim 的硬依赖； 3.几何模式改成模型实例属性； 4.几何模式进入 hash、缓存键和 checkpoint metadata； 5.参数旧常量清理； 6.第一步 parity 改称 one-step parity； 7.增加 h2–h6 分布偏移报告。

第一批半：横向消费者图

在正式训练前完成：

κ输出对ego/OD/LD/lane-anchor/target-lane/Memory的敏感度


这项成本低，但能防止横向训练继续强化错误旁路。

第二批：横向训练

先跑最小臂：

P0：
现训练基线

P1：
+ 当前帧κ监督

P2：
+ 速度分桶的横向监督

P3：
P2 + 跨帧重规划一致性

P4：
P3 + LQR执行差距重加权


几何训练必须分成：

legacy训练/legacy评测
corrected训练/corrected评测


不使用 zero-shot 几何切换作主结论。

第三批：WorldState 一致性

并行实现：

统一Kinematics
完整ego latent纳入WorldState
底层ego/OD residual
readout完全由final state产生


但其第一阶段验收目标是：

状态一致
消费者真实存在
不污染安全指标


不预设它立刻比横向训练有更大闭环收益。

四、本轮可以达成的最终共识
完全接受

-生产继续 world_state_api=off； -研发继续保留 analytic_learned；

analytic_learned 未通过安全门不能设默认； -先补正式证据链； -横向动作幅值、变号和转向饱和是当前最高优先级性能问题； -跨帧一致性必须状态门控；
fixed 几何需要在训练侧 A/B；
WorldState 重构与 Policy 训练并行；
Memory、BPTT、DAgger、MoE、RL 后置。
需要修改后接受
V4B 只能是“未入库研究观察”，不是已执行正式结论；
teacher-forcing 两遍重放不能自动视为合法反事实； -正式 residual runner 仍未真正自包含； -横向训练前必须补消费者图； -底层 residual 第一版应收缩； -LQR重加权要基于 student-expert 执行差距，而非绝对 steer 大小。
给 Coding Agent 的简短回复
总体接受R41四阶段方案和执行排序。

但需修改六点：

1. R40 V4B结果移出“正式已执行结论”，标为
   UNVERIFIED_IN_REPO，正式接线与复现后再裁决。

2. teacher-force优先采用simulator snapshot/clone的一步合法transition；
   两遍动作重放必须报告环境状态漂移，不能自动视为动作条件一致。

3. residual训练runner必须移除.slim硬依赖；
   核心CI不得因harness缺失而skip。

4. 横向训练前增加State Consumer Graph，
   确认κ实际读取ego/OD/LD/lane-anchor/Memory中的哪些信息，
   防止继续强化外部lane-anchor shortcut。

5. 底层状态residual第一版收缩为：
   ego速度/yaw-rate + OD位置/速度/presence；
   LD先保持解析静态几何，只学习validity，不自由修改地图几何。

6. LQR可执行性重加权使用student-vs-expert执行差距，
   不能简单惩罚所有高steer样本。

完成证据链和消费者图后，
优先跑横向@速度、跨帧曲率一致性和几何训练A/B；
WorldState一致性重构并行，但先以状态闭合与不污染安全为门。


最终判断：可以达成共识并开工，但必须先把 V4B 从“结论”降回“待正式复现假设”。 当前最有可能带来近期闭环收益的是横向 Policy 的幅值、帧间一致性和固定 LQR 可执行性；而统一 WorldState 是避免未来继续依赖 shortcut、最终支持自主选道和复杂交互的必要架构工程。