下面的评审不是沿着 docs\v8_net_review.md 的问题清单做复述，而是先从“闭环为什么失败”重新审视输入、控制接口、网络、监督和验证闭环，再回头判断文档中的问题是否抓到了主因。结论会比较直接：我认为当前 repo 的主要矛盾不是模型容量不足，也不是 MoE 专家数不够，而是闭环目标、可观测信息、动作接口、世界模型监督和训练分布之间没有形成一致的因果链。

一、先给结论
总体评价

这个 repo 不是“网络没搭起来”，相反，它已经有相当完整的研究型框架：

观测有历史对齐、对象级表示、车道线、导航和全局路线；
网络包含 encoder、temporal memory、MoE、latent world model、policy/value；
训练链覆盖 Stage A 世界模型、Stage B BC、DAgger-lite、Stage C PPO；
有配置守卫、训练范围测试、闭环评测和多种诊断工具；
文档对代码偏差和证据边界总体诚实。

但从闭环结果看，当前系统更像是：

多个合理模块串联成了一条复杂训练链，但还没有证明这些模块围绕同一个闭环瓶颈协同工作。

最重要的实证是：

IDM 闭环成功率约 0.756；
网络最好的 s11 约 0.646，但多数架构臂只有 0.088 到 0.312；
失败中 off-road 占比达到 68% 到 97%；
primary 容量从 256 增至 768，闭环表现反而从约 0.224 降至 0.154；
离线拟合与闭环性能出现反转。

这些现象共同反驳了一个隐含假设：

“只要 representation、world model、MoE 和参数量继续增强，闭环自然会改善。”

目前没有证据支持这条路径。相反，证据更支持：

主要失败来自几何与控制可行性，而非高层交互策略能力不足。
现有监督对 off-road 主因不敏感。
世界模型预测的对象，并不是当前闭环最缺的信息。
MoE 增加了容量与优化自由度，却没有得到可辨识的专家分工。
PPO 可训练范围太窄，难以修正 BC 和 WM 形成的结构性偏差。
二、当前审核输入是否足够
2.1 足够进行架构级审核，但不足以得出最终模型选择结论

现有输入已经足够判断以下问题：

网络数据流和主要模块职责；
参数量分布；
Stage A/B/C 的梯度路径；
MoE 训练方式；
world model 的 target 构造；
freeze scope；
closed-loop 主要失败类型；
一部分架构消融和容量实验。

但是，还不足以可靠回答：

s11 为什么能达到 0.646；
它相对其他 arm 的提升究竟来自 DAgger、RL、初始化、数据量还是随机波动；
当前最优 checkpoint 是否可复现；
网络是否真的利用了 world-model rollout；
MoE 是否形成了有意义的 specialization；
off-road 是规划几何错误、LQR 跟踪误差、坐标变换误差，还是环境判定边界问题；
不同 map 类型、道路类别、曲率、车速、目标密度上的分层表现。

文档明确说明多数结果只是 A→B，只有 s11 走完完整 DAgger+RL 链；部分历史产物已经被清理，E0 等结果无法独立复算；clean150 误差约为 3.5 个百分点且多为单 seed。

因此，目前最危险的事情是继续做“大架构决策”，例如：

primary 选 512 还是 768；
experts 用 8 个还是 12 个；
hidden dim 从 128 放大到 192； -继续增加 world-model head。

这些结论在现有证据下都太早。

2.2 当前最缺的不是更多代码，而是闭环归因数据

建议 coding agent 首先补齐每个 episode、每个策略 step 的失败归因记录：

map_id / seed / road_class
route curvature and width
ego speed / yaw rate / lateral error
planned point sequence
LQR commanded steering and acceleration
actual executed trajectory
distance to drivable boundary
distance to lane center or valid corridor
nearest-object clearance
world-model prediction error
policy mean/std and sampled action
MoE router logits/top-k experts
termination cause and first irreversible-error time


尤其要记录三个不同时间：

首次计划不可行时间
首次车辆压线或越界时间
环境正式判定 off-road 时间

没有这个时间分解，很容易把 LQR 的执行失败误认为 planner 失败，也可能把 planner 的早期错误归结为终局附近的 policy action。

三、输入与显式建模是否足够
3.1 当前输入对一般道路规划基本够用，对“复杂边界约束”不够可靠

现有观测包含：

6 帧历史，约覆盖 2.5 秒；
OD 16×9；
LD 16×7；
ego；
nav；
signal；
others；
64×2 route_world；
历史观测经过 SE(2) 对齐。

这套输入对直道、缓弯、普通跟车和常规车道导航是合理的。但 off-road 集中在 tollgate 和复杂几何时，目前表示存在几个结构性缺口。

缺口一：没有直接建模可行驶区域边界

LD 是稀疏车道线采样，offset 为 5、10、15、20、30 米。它更像 lane reference，不等同于 drivable-area boundary。复杂收费站、道路分叉、汇入区、局部宽度快速变化时：

道路可行域不一定可以由当前车道左右线稳定描述；
“最近车道线”不一定代表不能跨越的边界；
车道槽位排序变化会造成拓扑不连续；
稀疏的前向点无法表达近车身 footprint 与道路边界的关系。

如果 off-road 是 68% 到 97% 的失败主因，那么网络应当显式得到并预测：

左右可行驶边界；
局部道路宽度；
route corridor；
ego footprint 到边界的最小有符号距离；
未来 plan 点的 corridor violation。

这是比扩展 primary hidden dim 更高优先级的输入。

缺口二：规划输入没有把执行器约束表达完整

网络输出 0.5 秒尺度的 (ds, dθ)，连续 6 步形成 3 秒 plan，然后由 LQR 在 10Hz 下跟踪。

这一接口非常紧凑，但存在明显风险：

(ds, dθ) 本身不能保证连续曲率；
相邻 step 的 dθ/ds 可能突变；
低速下 ds 很小时，等效曲率可能异常；
计划的几何可达性没有自然保证；
LQR 跟踪误差没有反馈进入 planner 的训练目标；
0.5 秒一个规划节点，对 tollgate 近场边界可能过粗。

因此要明确回答：

当前 off-road 是“计划点已经出界”，还是“计划点在界内，但 LQR 执行出界”？

如果是前者，问题在 planner representation 和 loss；如果是后者，优先修 action parameterization、轨迹平滑和 tracker-aware training，而不是改 MoE。

3.2 signal 当前恒为 unknown，应从主干中移除或明确标记为占位

代码和文档表明 signal 当前恒为 [0,0,0,1]。这意味着它没有任何样本间信息量。

这本身不会造成大量参数浪费，但会带来两个问题：

架构图上看似支持信号灯，实际上没有信号决策能力；
world model 或 policy 可能把一个恒定 token 当成偏置通道。

建议：

在 signal 真正接线前，从 active token 集合中关闭；
或加入 signal_available，不要把 unknown 伪装成有效感知；
测试与文档明确“当前 benchmark 不评价红绿灯行为”。
3.3 对象固定槽位与 matching 机制存在潜在监督噪声

OD 使用固定槽位和 presence/entry；未来监督在 identity 缺失时使用最近邻匹配，gate 可达到 8 米。

8 米对于邻车换道、交叉流和收费站并行车辆来说可能过宽，容易出现：

ID switch；
遮挡后重现匹配到错误对象；
一辆车的未来 target 被另一辆车占用；
presence 与 latent target 语义不一致。

但这些对象预测又不是当前 off-road 主因。我的建议不是立即重写 tracker，而是先做分流：

collision subset 上审查 OD matching；
off-road-only subset 上暂时关掉动态对象，观察性能变化；
如果 off-road 几乎不受 OD 影响，则不要继续把大量研发资源投入对象 latent rollout。
四、网络架构是否合理
4.1 Encoder

当前 encoder 参数约 19k，只占总参数约 1.5%，而整个模型约 1.28M。plan head 与 ST-GNN 合计超过 70%。

这并不能单独证明 encoder 太小，因为输入大多是结构化低维特征。但它暴露了一个不平衡：

大量容量放在“处理和外推 latent”，很少容量用于建立稳定、物理可辨识的初始表示。

特别是 latent target 由同一套 encoder 编码未来特征再 detach 产生。encoder 本身既定义 target 空间，又被其他任务更新，容易出现：

target 坐标系缓慢漂移；
latent loss 很低但物理信息损失；
encoder 与 transition 共适应；
不同 checkpoint 的 latent metric 不可比较。
建议

不要先简单扩大 encoder。先做三个 probe：

从 latent 线性解码道路边界距离、横向偏差、曲率、速度和对象相对运动；
比较 Stage A 前后 probe 指标是否改善；
看 latent consistency 降低时，物理解码误差是否同步降低。

如果 latent loss 下降而这些物理 probe 不改善，说明 Stage A 主要在做表示共适应。

4.2 Memory bank / temporal attention

6 帧、2.5 秒历史对 2Hz planner 是合理的，尤其适合估计动态对象和自车运动趋势。但当前 memory 只对 ego、OD、others 做三组时间注意力，LD 等相对静态结构没有同样的时序融合。

更值得注意的是已经发现：

全无效历史列在 TemporalAttention 中会输出 out layer bias，而不是严格零。

这不只是小工程 bug。warmup、reset 和短 episode 开头都可能引入系统性伪 token。建议立刻修复，并增加：

assert output[all_invalid_mask].abs().max() == 0


同时做一个低成本消融：

current-only；
2 帧；
6 帧；
6 帧但 shuffle history；
6 帧但全部复制 current frame。

如果 6 帧与复制 current 接近，说明 temporal memory 没有提供有效动态信息。

4.3 MoE

当前 MoE 是 primary 常开、8 个 residual experts、top-2 softmax、zero-residual 初始化，router 只接受 Switch 风格负载均衡，没有路由语义监督。

这个设计在工程上稳定，但从识别性看不充分：

primary 可以解释大部分输出；
experts 从零残差开始；
top-2 强制每个样本用两个专家；
load balance 迫使样本均匀分配；
没有证据证明数据中存在 8 个可分离的控制模态；
specific 阶段冻结共享 trunk 和 primary，只允许 experts/router 修补残差。

因此专家完全可能学成：

随机分片；
根据数据顺序或噪声分流；
多个高度相似的 residual；
一个困难场景被平均分给不同专家；
为满足 balance 而牺牲实际路由置信度。
我的判断

当前 MoE 不是第一优先级收益模块，更可能是实验复杂度放大器。

primary 256→768 变差也说明系统不是简单的欠拟合。参数越多，反而可能更容易拟合 expert action 中的局部噪声与开环偏差。

建议

先建立三条等预算 baseline：

单一 MLP plan head；
primary + 2 experts；
primary + 8 experts。

要求总参数量近似相同，训练数据、epoch、seed、DAgger 和 PPO 链完全相同。

MoE 必须满足以下条件才保留：

至少三 seed 提升；
提升主要出现在困难场景而非均值偶然波动；
专家路由对 road geometry / maneuver 稳定；
同一场景小扰动不会导致专家频繁切换；
expert residual norm 非零且差异显著；
关闭 expert 后性能明显下降。

否则建议删除 MoE，先用单头解决闭环基本盘。

4.4 World model / ST-GNN

这是当前架构中最需要质疑的部分。

Stage A 使用：

GT action chain；
GT pose accumulation；
OD 匀速外推；
LD 静止；
每一步 latent state detach；
未来 latent target 来自同一 encoder 并 detach；
latent consistency 权重 1.0；
OD 物理解码 0.1；
LD 物理解码仅 0.02。

这使系统被称为“自回归 world model”，但训练意义上更接近：

六个共享参数的一步 latent transition 监督，在 GT 条件下逐步采样。

因为 state 每步 detach，后续误差不会对更早 transition 提供 credit。它没有真正训练：

长期误差传播；
model rollout stability；
policy-induced distribution shift；
tracker execution error；
错误 action 后如何恢复；
off-road 边界风险。

更关键的是，当前最主要失败是 off-road，但 world model 的主要结构容量用于 ego/OD/LD latent transition 和对象预测，没有直接预测：

drivable probability；
signed boundary distance；
route corridor violation；
future footprint overlap；
plan feasibility；
tracker-induced deviation。

这就是任务目标与建模对象错位。

结论

world model 的存在有理论合理性，但当前监督不能证明它学到了对闭环有用的可控动力学。

latent WM 比 feature WM 高约 11 个百分点是有价值的信号，但因为历史产物不可复算、完整链路和 seed 不充分，不能据此认定当前 WM 机制正确。

两条可选路线
路线 A：近期务实路线

先删除或 bypass 复杂 latent rollout，使用：

当前表示；
历史编码；
直接 6 点轨迹 head；
显式 corridor / curvature / collision cost；
闭环 DAgger。

如果性能不降，说明 WM 当前主要是冗余。

路线 B：保留 WM，但改造成 control-relevant model

WM 不只预测 latent，而预测可验证量：

ego future pose distribution
ego footprint to boundary distance
route progress
road corridor occupancy
object occupancy / collision probability
tracking-error envelope
termination risk


并至少做 2 到 3 步的 truncated BPTT，不要每步全部 detach。可以先从 horizon=2 开始控制显存和稳定性。

4.5 Policy / value head

policy 与 value 分别约 108k 和 100k，使用相似的 cross-attention 结构。

问题不在参数量，而在职责重叠：

plan head 已经生成动作链或计划；
WM 又对 plan 进行 latent rollout；
policy 再 cross-attend rollout tokens 输出最终动作；
LQR 又依据 plan 执行。

这里至少存在三个“动作决策中心”：

PlanHead；
PolicyHead；
LQR tracker。

必须明确各自的因果职责。否则会出现：

planner 计划一条轨迹；
policy 输出与计划不一致的首步动作；
tracker 执行的又是离散 plan 的几何结果；
训练 loss 对不同输出施加不一致约束。

建议加入一致性指标：

policy_action vs plan_first_action
policy_action vs executed_action
plan rollout vs actual ego trajectory
LQR tracking residual
critic value vs empirical route completion


如果 policy 只是读取 plan 然后重建首步动作，可以把它简化成 plan confidence / residual correction head。如果 policy 是最终决策器，则 plan 应被定义为 policy 的显式候选或中间变量，并对最终执行动作建立一致的监督。

五、参数量分布是否合理

当前约 1.28M 参数中：

PlanHead 约 41.5%；
ST-GNN 约 30%；
memory 约 10.6%；
policy 约 8.5%；
value 约 7.8%；
encoder 约 1.5%。

这说明约 71.5% 的参数放在 plan head 与 world model。

这个分布只有在下面两件事成立时才合理：

plan head 是主要闭环性能来源；
world-model rollout 是 policy 不可替代的信息来源。

目前两者都没有被充分证明。

primary 容量增加没有带来单调增益，是参数分布可能错位的直接信号。

推荐的重新分配原则

不是简单把 1.28M 降到多少，而是先建立 400k 到 700k 的强 baseline：

structured encoder          60k–100k
temporal encoder            80k–120k
geometry/interaction trunk 100k–160k
single trajectory head      80k–150k
value/risk head             50k–100k


然后逐一加入：

latent WM；
MoE；
cross-attention policy； -更大 primary。

每个模块都必须给出增量闭环收益。否则当前参数统计只是“容量账本”，不是“收益账本”。

六、监督是否合理
6.1 Stage A

Stage A 的核心问题不是 loss 数量少，而是监督语义不闭环：

GT action 与 GT 位姿造成 teacher-forcing distribution；
state detach 取消长期 credit；
OD 匀速、LD 静止是强先验；
latent target 由同 encoder 产生； -物理解码权重明显低于 latent consistency。

建议至少加入：

scheduled sampling 或 action perturbation；
rollout horizon 2 到 3 的非 detach loss；
future ego pose / curvature；
boundary distance；
plan footprint feasibility；
action-conditioned contrastive loss；
EMA target encoder，避免 target representation 快速漂移。
6.2 Stage B

BC 可以建立基本驾驶能力，但 expert 的控制风格、网络的动作接口和 LQR 最终执行行为不一定一致。

worst-50% hard mining 当前主要依据 BC imitation error。

这是一个很弱的困难样本定义，因为：

高 BC error 可能对应一对多动作；
专家动作可能有噪声；
高 IL loss 不一定造成 off-road；
真正危险样本可能动作误差很小，但接近边界。

hard mining 应该改为闭环风险优先：

boundary margin low
planned footprint outside corridor
high curvature mismatch
large LQR tracking residual
collision TTC low
policy uncertainty high
failure precursor window


BC error 只能作为辅助特征。

6.3 DAgger-lite

当前 phase3 的 traj_aux=0，失败窗口的轨迹还是常量外推合成值，因此闭环修正主要依赖首步或动作链监督。

这与 off-road 根因不匹配。出界通常是数步累计后发生，只修首步 action 很容易：

在单步 loss 上看起来变好；
但计划曲率和边界 margin 没有改善；
下一次 rollout 又回到错误分布。

建议 DAgger 采集真实 expert recovery trajectory，而不是只存失败点附近的 action：

回溯失败前 2 到 4 秒；
标记 earliest deviation；
存 expert correction 和 student plan；
监督整段 plan；
加入 plan feasibility loss；
对 recovery 样本单独计权。
6.4 Stage C PPO

Stage C 默认只训练：

policy/value；
MoE experts；
residual scale。

而 encoder、memory、primary、router、ST-GNN 都冻结；world model 全期冻结，自监督 W2 尚未接入。

这意味着 PPO 即使发现 closed-loop representation 有问题，也没有能力修复：

road geometry 表示；
temporal feature；
router；
world-model transition；
primary plan prior。

它只能在一个固定表示和固定 WM 上做末端补偿。这非常可能解释“RL 看似接入，但只能提供有限改善”。

建议的解冻顺序

不要一次全部解冻。使用渐进式 scope：

policy/value only；
加 plan head 最后一层；
加 temporal/world trunk 的 LayerNorm 与 adapter；
加 router；
最后才考虑完整 encoder 或 WM。

每一级都监控：

KL；
BC holdout；
closed-loop success；
off-road；
collision；
representation drift；
planner action variance。
七、对 docs\v8_net_review.md 问题点的评价

文档列出的大多数问题是准确的，尤其包括：

Stage A detach 导致长期信用缺失；
teacher forcing 与闭环分布不一致；
latent target 共适应风险；
object matching 噪声；
specific-only 冻结过强；
DAgger trajectory 监督实际未生效；
PPO freeze scope 受限；
TemporalAttention 全无效列问题；
配置死键和文档漂移；
评测单 seed、产物不可复算。

但我认为文档还没有把问题排序到真正的 root cause 层级。

它更偏向“每个复杂模块分别有什么风险”，而当前更应追问：

为什么一个主要失败是越界的系统，把大部分研发复杂度放在对象 latent rollout、MoE 路由和 frozen RL 上，而不是可行驶边界、轨迹可达性和 tracker-aware loss 上？

也就是说，文档的问题点基本对，但仍有“在既定架构框架内优化”的倾向。下一步应该允许反驳整个架构组合，而不是默认每个模块都值得保留。

八、我对当前 repo 工作的根因判断
Root cause 1：主要闭环失败目标没有被直接建模

off-road 是 dominant failure，但网络没有把 boundary clearance、corridor feasibility、footprint violation 作为第一等输出和 loss。

Root cause 2：动作表示与 LQR 执行之间缺少可行性约束

(ds,dθ) 六步计划简洁，但没有天然保证曲率连续、车辆动力学可达和 tracker 可追踪。

Root cause 3：训练链每一阶段优化的对象不一致
A 优化 latent prediction；
B 优化 expert imitation；
DAgger 主要修首步 action；
C 在冻结 backbone 下优化 reward；
最终评价是 LQR 执行后的 closed-loop success。

这五个目标之间缺少可度量的一致性桥梁。

Root cause 4：架构复杂度超过证据能力

只有 s11 走完完整链，多数对照不是同训练链、同 seed、同产物可复算。

因此暂时无法区分：

架构收益；
数据收益；
DAgger 收益；
PPO 收益；
seed；
checkpoint selection；
评测噪声。
Root cause 5：容量不是当前瓶颈

primary 扩大后退化，离线拟合与闭环反转，说明更可能是 objective mismatch、distribution shift 和执行约束，而不是 under-capacity。

九、交给 coding agent 的高优先级事项
P0：一周内完成，先停止盲目扩模型
P0-1 闭环失败时间线与归因

实现每个失败 episode 的 structured trace：

first_bad_plan_step
first_boundary_cross_step
termination_step
plan_inside_corridor
executed_inside_corridor
tracking_error
min_boundary_margin
curvature_peak
road_class
map_id
router_topk
policy_std

验收标准：

随机抽取至少 100 个 off-road episode，能够自动分为：

计划本身出界；
计划在界内但 tracker 出界；
坐标、观测或判定异常；
recovery 失败。

无法完成这个分类前，暂停 primary/MoE 参数扫描。

P0-2 建立最小强 baseline

新增一个配置：

model:
  world_model: false
  moe: false
  plan_head: direct_trajectory
  history: true
  trajectory_points: 6


保留相同 observation、相同数据、相同 evaluator、相同 LQR。

验收标准：

三个固定 seed；
eval 至少 500 episodes；
输出 success、off-road、collision、route completion；
与 s11 使用相同训练链比较。

如果简化 baseline 接近或超过复杂模型，暂停 WM/MoE 开发。

P0-3 添加显式几何安全监督

至少添加：

plan_boundary_loss
plan_corridor_violation_rate
footprint_boundary_margin
curvature_smoothness_loss
delta_curvature_loss


不要只检测 plan 点。使用车辆 footprint 在中间插值位置检查，避免两个 0.5 秒节点都在界内但中间轨迹切出边界。

P0-4 修复 TemporalAttention 无效历史输出

全 invalid 输入必须严格输出零，并增加 reset/warmup 单测。该问题已经是已知实现偏差，不需要继续讨论，应直接修。

P0-5 锁定可复现实验协议

所有正式 run 必须保留：

git commit；
dirty diff；
完整 resolved config；
obs fingerprint；
dataset manifest；
seed；
checkpoint hash；
per-episode result；
tensorboard/metrics；
evaluator version。

禁止只留 summary JSON 后清理原始产物。

P1：P0 得到归因后推进
P1-1 若“计划出界”为主

优先：

drivable corridor 输入；
boundary-aware trajectory loss；
更密近场 plan；
trajectory spline / curvature parameterization；
DAgger recovery trajectory。
P1-2 若“tracker 出界”为主

优先：

plan tracking feasibility loss； -训练时引入近似 vehicle rollout；
将 LQR residual 作为输入或监督；
限制曲率与曲率变化率；
提高近场节点密度，例如前 1 秒使用 0.2 到 0.25 秒间隔。
P1-3 World model 价值验证

做四个严格对照：

No WM；
WM latent 输入 policy，但 stop-gradient；
当前 detached rollout；
2-step truncated BPTT + physical targets。

相同参数预算、相同训练链、三个 seed。

保留 WM 的条件：

success 均值稳定提升；
off-road 显著下降； -提升不只来自参数量；
WM 误差与闭环失败有可测相关性；
action perturbation 下 rollout 仍有校准能力。
P1-4 MoE 价值验证

比较：

0 experts；
2 experts；
4 experts；
8 experts。

同时记录：

router entropy；
expert occupancy；
expert residual cosine similarity；
expert switching rate；
按道路类型的 routing mutual information；
关闭单个 expert 的性能变化。

如果 8 experts 无显著优势，降为 0 或 2。

P2：只有 P0/P1 证明有效后再做
更大 encoder；
更复杂图网络； -更多专家；
learned object association； -长 horizon WM； -联合 actor-world-model RL； -更大 hidden dim； -多模态 plan anchor 扩容。

这些都不是当前最高优先级。

十、建议 coding agent 直接采用的任务描述
目标：
优先定位并降低 closed-loop off-road，而不是继续扩大模型。

任务 1：
增加 closed-loop failure trace。
区分 planned trajectory violation、LQR tracking violation、
observation/coordinate anomaly 和 recovery failure。
记录 first_bad_plan_step、first_boundary_cross_step、
termination_step、boundary margin、tracking error、curvature、
road class、map id 和 router statistics。

任务 2：
实现 no-WM/no-MoE direct trajectory baseline。
保持 observation、dataset、DAgger、PPO、evaluator 和 LQR 不变。
运行 3 seeds × eval500。

任务 3：
为整条计划增加 drivable-corridor 和 vehicle-footprint loss。
检查节点之间的插值轨迹，不只检查离散 plan points。
增加 curvature 和 delta-curvature penalty。

任务 4：
修复 TemporalAttention 在 all-invalid history 时输出 bias 的问题。
增加 reset、warmup、all-invalid 的单元测试。

任务 5：
将 DAgger hard mining 从 BC loss 优先改为 closed-loop risk 优先。
回溯失败前 2 到 4 秒，收集 expert recovery trajectory，
而不只收集首步 action。

任务 6：
完成严格 WM ablation：
No-WM、detached-WM、2-step-BPTT-WM。
所有配置必须同预算、同数据、同训练链、3 seeds。

停止条件：
若简化 baseline 达到复杂模型 success 的 95% 以上，
暂停 MoE 和大 WM 扩展，优先完成 geometry/control 对齐。

最终判断

我不会建议下一步继续调 primary=512/768，或者继续强化 8-expert MoE。现有结果已经显示容量与闭环表现不单调，而且 dominant failure 是 off-road。

更合理的推进顺序是：

先证明失败来自 plan 还是 tracker；
把 drivable geometry 和可执行性变成显式输入、输出和 loss；
建立无 WM、无 MoE 的强 baseline；
用严格完整训练链验证每个复杂模块的增量价值；
最后再决定保留 MoE、latent WM 和 cross-attention policy 中的哪些部分。

一句话概括：

当前 repo 的短板不是“模型不够强”，而是“复杂模型没有围绕闭环主失败建立可验证的目标一致性”。下一步需要做减法、补闭环因果证据、强化几何与执行约束，而不是继续堆容量。