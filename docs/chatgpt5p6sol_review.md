总体裁决

Coding Agent 这轮攻防质量较高，纠正了我上一版审计中的多处不精确表述。尤其是：

Stage A 不是“只训练一步”，而是六步 free-running latent 输入 + 每步局部监督 + 跨步梯度截断；
Stage A 的实际优化范围不是 YAML 声明的 trainable，而是代码中的排除式过滤；
Stage B 的未来五步确实在预测 latent 上产生，但没有逐步动作硬标签；
Stage C 当前并没有使用 ExactTracker，而是 LQR + repeat sampled action； -部署则是 LQR + WM rollout plan。

这些代码事实基本可信。

但我不同意 Coding Agent 将处置收缩为“已有材料很多，只补 R4、动作干预和梯度账本，同时 Round-30 原议题继续并行”。目前已经发现两个可能改变整个新一轮优化方向的重大结构事实：

Stage B 的未来动作监督与预测状态存在结构性错配风险；
Stage C 的训练执行口径与部署执行口径明显不同。

所以我的最终裁决是：

批准缩减版 Reality Audit 开工，但暂停 T2 蒸馏、首尾 Δκ 等 Round-30 训练议题。先完成四个决定性证据包，再决定 Stage A/B/C 的真正改造方向。

一、对 Coding Agent 逐项攻防
1. C1：Stage A 配置与真实训练目标漂移
Coding Agent 的纠正

config/train.yaml 中仍写着：

loss: direct_multi_step
ego_condition: expert_gt_actions
targets: future_od_ld_lookup
slot_matching: nearest_neighbor_t0


但真实代码中：

-主监督是 latent consistency，权重 1.0；

OD 物理解码权重 0.1；
LD 物理解码权重 0.02； -若干 YAML 键实际未被 run_stage_a 消费。
裁决

完全同意，并将其升级为配置契约 bug。

这不是单纯注释过时。它会让后续使用者误以为：

配置改变了 Stage A 的训练行为


但实际上改动无效。

必须处理

审计阶段不立刻改训练行为，但必须生成：

docs/stage_a_config_consumption.md


每个配置键标记为：

CONSUMED
IGNORED
LEGACY
DERIVED


审计结束后：

-删除无效键，或 -在入口遇到无效键直接 fail-fast。

不能继续让配置“看起来生效”。

2. C2：Detach 的真实作用
Coding Agent 的纠正

当前 Stage A 的状态链是：

预测 latent 作为下一步输入
但进入下一步前 detach


因此：

h2–h6 的输入处于 free-running 模型分布； -共享 transition 参数仍会从每个 horizon 的局部 loss 获得梯度； -但 h6 loss 无法通过状态链修正 h1–h5 的 transition 决策。
裁决

同意。这比我此前“一步 transition 训练”的表述更准确。

准确名称应是：

Free-running inputs with truncated cross-step credit assignment

即：

前向：六步自由滚动
反向：每步局部监督，跨状态链截断

这意味着什么？

不能简单推导：

detach一定错误


因为共享权重在每个 horizon 都受到监督，可能已经足够。

也不能简单推导：

所有horizon有loss，所以多步信用已经闭合


因为共享参数的局部梯度，与通过状态链链式传播的多步梯度并不等价。

裁决

先做 h1/h3/h6 独立 backward 账本，不直接删 detach。

3. C3：WM 动作条件化方式
当前事实

动作进入 WM 的路径主要是：

action
→ ego累积pose
→ 图边几何
→ ST-GNN transition


另外 nav 会随预计 ego pose 重建。

但 transition 没有显式 action embedding。

Coding Agent 的态度

“是否足够尚未证明，接受动作干预 probe。”

裁决

完全同意。

这里不应凭架构审美直接增加 action token。Pose-mediated conditioning 可能已经足够表达当前任务。

但 probe 必须覆盖的不只是 ego pose，还包括：

future latent
future physical decoder
next Policy action


否则只能证明运动学积分变了，不能证明 WM transition 真正理解了动作。

4. C4：Stage A optimizer 与 YAML trainable 不一致
当前事实

实际 optimizer 是：

除 policy.* / value.* 外的全部参数


而不是严格读取 YAML 的：

encoders
mem_encoder
plan_head
st_gnn


同时 MoE 被关闭，所以部分参数虽然进 optimizer，却没有梯度。

裁决

这是必须修复的工程契约问题，不只是文档问题。

风险包括：

-未来某模块意外获得非零梯度； -以为冻结但其实进入 optimizer； -恢复 optimizer state 时发生隐性状态变化； -配置和代码长期继续分叉。

建议

审计期间只记录，不改行为。审计后采用白名单式建组：

trainable_prefixes = resolved_stage_config["trainable"]


并强制：

配置声明可训练参数集
=
optimizer实际参数集
=
具有非零梯度的允许参数集


死参数不应进入 optimizer。

5. C5：PolicyHead 是唯一动作生成器
当前事实
PlanHead 输出融合 latent 和 ego_next；
PolicyHead 输出动作；
K-anchor 当前关闭。
裁决

完全同意。

正式架构文档应写：

PlanHead：历史命名，实际职责为 PolicyBackbone / ContextFusion
PolicyHead：唯一动作输出器
World Model：唯一状态预测器
Rollout Orchestrator：重复调用 Policy 与 WM


暂不改 class 名和 checkpoint key。

二、最大的两个结构性问题
1. Stage B 的未来动作监督存在真实的状态标签错配风险

Coding Agent 已确认：

a1...a5 是 Policy 在预测 latent 上输出的；
ST-GNN 在 Stage B 冻结； -状态链 detach； -尾五步没有直接 action label； -只有权重 0.1 的 traj6 位置监督； -phase3 才有 action_chain: 0.2。

这比原先想象更严重。

为什么是结构错配？

部署链是：

z^k+1=WM(z^k,a^k)\hat z_{k+1}=WM(\hat z_k,\hat a_k) a^k+1=π(z^k+1)\hat a_{k+1}=\pi(\hat z_{k+1})

但 traj6 标签来自真实专家轨迹：

PkexpertP_{k}^{expert}

当预测 latent 已偏离真实专家状态时，要求：

Policy(预测状态)
→ 仍然生成专家真实状态下的后续累计位置


不一定是合法监督。

这和此前 B-label v1 的 bug 并不完全相同，因为这里不是简单拿错时间戳，而是：

预测状态与专家轨迹目标所在状态不一致。

轨迹级监督有可能绕过这一点，因为累计位置仍提供弱终点方向，但不能将其默认视为同状态 action supervision。

必须通过 R4 分解

对同一真实序列，逐 horizon 计算：

a_real(k) = Policy(Encoder(real_obs_k))

a_wm(k) = Policy(WM_rollout_latent_k)

a_expert(k) = expert current action at real_obs_k


拆成：

Policy真实状态误差：
|a_real - a_expert|

接口误差：
|a_wm - a_real|

总部署动作误差：
|a_wm - a_expert|


如果：

a_real≈expert
a_wm差


问题在 WM/latent interface。

如果：

a_real也差


问题在 Policy/Stage B。

如果：

单步都不错，但累计traj差


问题在 action integration、chain consistency 或轨迹监督。

这是下一阶段的第一决定性证据。

2. Stage C 当前训练口径与原设计不一致
当前代码事实

Coding Agent 核验：

collect/update：cheap path
sample：单步 a_t
执行参考：repeat(a_t, 6)
执行器：LqrTracker
buffer：a_t / logprob / value / reward
部署：WM rollout六步plan + LQR


所以 Stage C 当前训练的是：

Policy当前动作
→ 重复六次
→ LQR
→ 环境reward


而部署是：

Policy–WM rollout六步不同动作
→ LQR


这不是小差异，而是 effective action mapping 不一致。

与你的原始设计差在哪里？

你定义的 Stage C 应该是：

Policy单步action
→ ExactTracker精确执行
→ 环境真实next state
→ PPO训练单步Policy


当前实现却是：

单步action重复六次
→ LQR预瞄整条恒动作plan
→ 控制跟踪误差进入reward


因此当前 PPO 实际同时在适配：

Policy 单步决策；
repeat-tail 几何； -LQR preview； -控制误差； -横纵耦合。

这与“规划端 RL 与控制端解耦”的设计目标不一致。

三、我对 Coding Agent 反驳的回应
反驳 1：“无需暂停新主模型训练”

Coding Agent 说最近主要是零训练和小 canary，因此无需暂停。

我的回应

字面上可以接受，但工程上仍需锁定：

Reality Audit完成前
不得开始任何新主模型训练或蒸馏


这不影响已经停止的 Round-29，也不应继续并行 T2 蒸馏。

原因是：

R4 可能证明主瓶颈在 WM；
Stage C 可能需要先修执行契约；
T2 的正向也可能只是补偿当前 WM tail 或 LQR preview，而非真正改善路线 A。

所以“没有正在训练”不等于“无需冻结令”。

反驳 2：“绝对物理门与 latent WM 目标不符”

Coding Agent 认为 ego P90 等绝对门暂不应直接锁定。

我的回应

同意阈值不应提前锁死，但不同意退化成只看 latent consistency。

原因是 latent consistency 可能出现共适应：

Encoder(real future)
与
WM predicted latent
很接近

但二者都丢了Policy真正需要的物理信息


Stage A 必须同时有三类指标：

表示指标
latent cosine
latent L2

物理指标
ego / OD / lane-relative / presence

决策充分性指标
π(z_hat) vs π(z_real)


正确做法是：

先实测 current baseline 分布； 2.再以 baseline、constant-velocity 和 oracle gap 锁门； 3.绝对阈值只作为辅助，不在开工前主观拍定。
反驳 3：“BPTT curriculum 与固定 detach 规格冲突”
我的回应

同意不能在只读审计前预注册完整 BPTT 课程。

正确顺序应为：

逐horizon梯度账本
→ WM多步误差增长
→ R4真实/预测动作一致性
→ 一次最小2-step no-detach canary


只有同时满足：

horizon误差明显累积
接口误差随horizon增长
当前detach无法校正前段transition


才批准 2-step truncated BPTT。

不直接规划 1→2→3→6 全课程。

反驳 4：“先量化，再定部署迁移阈值”

同意。

不能先拍 +5pp 或固定误差门。应先获得：

Exact / LQR
×
plan / repeat_action


的配对矩阵，再锁门。

但这里有一个修正：

由于你的正式设计明确要求 Stage C 使用 ExactTracker，当前 LQR+repeat 不能只是“训练口径之一”，而应标记为实现偏差。

审计后若确认无隐藏依赖，应修成 ExactTracker 主训练。LQR+repeat 保留为历史对照。

四、对处置建议的最终修订

Coding Agent 给出的优先级是：

R4
> 部署迁移矩阵
> WM动作条件化
> 梯度账本


我建议调整为：

Priority 0：Stage C 执行契约复现

首先建立当前四格矩阵：

Exact + repeat_action
Exact + WM plan
LQR + repeat_action
LQR + WM plan


使用相同 checkpoint、相同 episode、相同起始状态。

目的不是调整 LQR，而是确认：

当前 Stage C 实际优化的系统； -原设计 Exact 单步执行的系统； -正式部署系统； -三者差距。

这是架构契约问题，应与 R4 同优先级。

Priority 1：R4 Rollout Consistency

输出：

E_z(h1...h6)
E_policy(h1...h6)
E_action_total(h1...h6)


按路型和结局分层。

这直接回答路线 A 是否闭合。

Priority 2：WM Action Fidelity

测试：

ds × {0.8,1.0,1.2}
dθ / κ 左右扰动
zero action
mirror action


必须对比：

predicted next physical state
predicted latent
next Policy action

Priority 3：h1/h3/h6 梯度账本

分别构造单 horizon loss，独立 backward。

需要验证：

共享transition是否在每个horizon获得有效梯度
早期transition是否无法获得后段credit
Policy action loss能否更新WM
Stage B是否完全冻结WM

Priority 4：Oracle-state / frozen-t0 对照

不能无限 backlog。R4 发现明显接口误差后，必须立即用：

oracle real future state
frozen-t0 latent
current WM latent


做三臂。

否则无法判断 WM 到底是有益、无益还是有害。

五、Stage C 的正式裁决

针对你的设计意图，我建议审计文档明确区分：

现状
Stage C current:
single sampled action
→ repeated six times
→ LQR
→ PPO

目标设计
Stage C intended:
single sampled action
→ ExactTracker
→ real next state
→ PPO

部署
Policy–WM ×6
→ six-step plan
→ fixed LQR


三者不是同一执行口径。

审计后建议

若 ExactTracker 环境步骤和奖励口径没有隐藏问题：

将 Stage C 主训练修正为 ExactTracker。

然后每个 PPO checkpoint 强制做部署迁移 Gate：

Exact PPO KPI
+
R4 rollout consistency
+
WM-plan + fixed-LQR KPI


这样符合你的设计：

RL 训练单步 Policy；
WM 将单步能力展开六步； -LQR只在部署评测中消费六步计划。
六、Round-29 的 T2 怎么处理？

T2 是目前唯一正向 projector：

success 0.52 → 0.58
方向 0.737 → 0.967
saturation下降
timeout下降


但它仍未过 collision、off-road 和 timeout 门。

我的裁决：

保留为诊断 oracle，不进入蒸馏，不成为正式规划器，不与 Reality Audit 并行训练。

它能说明：

当前尾段若向target-lane curvature收敛，
比简单Δκ平滑更合理


但在路线 A 中，正确结论应该是：

WM+Policy rollout 应自己学会产生类似的尾段，而不是永久外挂 T2 生成器。

等 R4 得到结果后：

-若 oracle-state Policy 尾段接近 T2，而 WM 尾段差，修 WM； -若 oracle-state Policy 尾段也不如 T2，修 Stage B/Policy； -若二者都好但 LQR差，才追 plan feasibility。

七、批准的 Round-30 工作范围
批准立即执行
R30-0：现状冻结与产物复用索引
R30-1：Stage C四格执行口径矩阵
R30-2：R4 real-state vs WM-state rollout consistency
R30-3：WM action-conditioning intervention
R30-4：h1/h3/h6 gradient reachability
R30-5：训练/部署执行契约文档

条件触发

若 R4 显示 WM 接口误差显著：

R30-6：
oracle-state / current-WM / frozen-t0 三臂

本轮不执行
T2蒸馏
2-step BPTT
显式action token
Stage C重写
DAgger
MoE重训
PPO/GRPO
新增loss
扩大数据
修改LQR

八、建议验收方式

不先设绝对 WM 米制门，先用相对分解。

R4 接口门

对 horizon 1–6，定义：

Ereal(k)=∥π(zkreal)−akexpert∥E_{real}(k)= \|\pi(z_k^{real})-a_k^{expert}\| Einterface(k)=∥π(z^kWM)−π(zkreal)∥E_{interface}(k)= \|\pi(\hat z_k^{WM})-\pi(z_k^{real})\| Edeploy(k)=∥π(z^kWM)−akexpert∥E_{deploy}(k)= \|\pi(\hat z_k^{WM})-a_k^{expert}\|

重点报告：

interface error / deploy error


解释：

-小于 20%：WM接口不是主瓶颈； -20%–50%：共同瓶颈； -超过 50%：WM/预测latent是主瓶颈。

这些只作为首轮分层参考，实测后锁定。

WM action-fidelity 门

至少要求：

方向正确率 >=95%
mirror consistency >=95%
progress对ds扰动单调
yaw/lat对dθ或κ扰动单调
future latent可区分
Policy next-action响应不退化为常数

Stage C 四格矩阵

必须报告：

success
off-road
collision
timeout
route completion
reward


并做 paired episode 转换：

翻正
翻负
双方成功
双方失败


不能只比较总体均值。

九、给 Coding Agent 的攻防答复
总体接受 §5.29 的代码核验与四项措辞修正。

确认：
1. Stage A 是六步 free-running输入、跨步梯度截断，而非单步前向；
2. 共享transition受到六步局部监督；
3. PlanHead非动作输出器，PolicyHead唯一生成action；
4. Stage B尾五步位于预测latent，只有弱traj监督；
5. Stage C现状为LQR+repeat sampled action，不是设计意图中的ExactTracker；
6. WM动作敏感性、R4一致性、逐horizon梯度、oracle/frozen臂仍为空白。

不同意：
1. Reality Audit与T2蒸馏/首尾Δκ议题并行；
2. 仅以latent consistency作为WM主门；
3. 将Stage C执行差异降级为普通部署gap；
4. 将oracle-state/frozen-t0长期留作backlog。

Round-30批准范围：
A. Stage C四格口径：
   Exact/LQR × repeat_action/WM_plan；
B. R4：
   E_z/E_policy/E_action × h1...h6；
C. WM action intervention：
   ds scale、dθ/κ perturb、zero、mirror；
D. h1/h3/h6独立backward梯度账本；
E. 更新execution contract。

若R4接口误差显著，自动追加：
oracle-state / WM-state / frozen-t0三臂。

本轮禁止：
训练新主模型、T2蒸馏、删detach、BPTT、action token、
DAgger、MoE、PPO/GRPO、LQR修改、扩大数据。

最终判断

Coding Agent 的攻防没有推翻 Reality Audit，反而证明审计需要更聚焦。

目前最关键的新事实是：

Stage C 现状与用户定义的 Stage C 不一致；Stage B 未来五步缺少同状态动作监督；而连接 Stage C 单步 Policy 与部署六步计划的 R4 一致性至今完全没有实证。

因此，本轮不要再围绕 tail loss、投影或曲率平滑打转。先回答唯一核心问题：

π(z^kWM)≈?π(zkreal),k=1,…,5\boxed{ \pi(\hat z_k^{WM}) \stackrel{?}{\approx} \pi(z_k^{real}), \quad k=1,\ldots,5 }

如果成立，路线 A 的主链有坚实基础，可以继续优化 Stage B/C。

如果不成立，再根据分解结果决定：

-修 World Model； -修预测 latent 接口； -修 Stage B 的 predicted-state Policy 训练； -或修 Stage C 更新后的部署迁移。

这才是新一大轮优化的正确起点。