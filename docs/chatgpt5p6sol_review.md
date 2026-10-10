我读完了新 repo 和 docs\deepseekv4p1_argue.md。P0 的结果很有价值，因为它已经推翻了几个之前只能靠推理判断的假设：接口和 LQR 不是当前第一瓶颈，模型生成的单步动作与 3 秒计划质量才是核心瓶颈；同时，PPO 的 action 记账与最终 plan 执行链确实没有完全对齐。

下面我先解释你的五个疑问，再对 P1 路线给出我的判断。第 4 点我会明确反驳一部分，因为 P0 的 D1/D2 结果对“autoregressive action chain 就是正确规划接口”提出了很强的反证。

一、先重读 P0：它真正证明了什么

P0 的关键数字是：

执行方式	Success	主要含义PurePursuit + IDM 自身	0.756	当前规则专家基线
D2：当前专家动作重复 6 步 + LQR	0.706	(ds,dθ) + LQR 接口基本可用
Oracle future pose + LQR	0.608	开环未来轨迹并不天然优于反馈控制
Lane center plan + LQR	0.442	简单几何计划已超过多数 v8 模型
pri512 model plan + LQR	0.332	当前较好的 v8 BC 模型
Arm P model plan + LQR	0.224	当前基线
D1：未来专家 action chain + LQR	0.000	开环动作链严重失效
模型 action repeat + LQR	0.000	模型单步动作质量也严重不足

此外，pri512 的失败中有 254/334 是 off-road，至少 173 个失败带 plan infeasibility 因子，T_plan 中位约 120 个策略步；exact 执行模型计划不仅没有比 LQR 好，反而更差。这说明问题不是“LQR 没跟上一个好计划”，而是计划本身在闭环推进中逐渐失真。

所以目前最稳妥的根因排序是：

模型动作和计划质量不足；
长期推进后几何偏差累积，缺少边界和恢复监督；
PPO action 与最终 plan 执行链信用分配不一致；
LQR/tracker 不是当前主要瓶颈；
(ds,dθ) 接口不是绝对不可用，但“如何把它组织成计划”仍然有严重问题。
二、问题 1：expert5k 的专家不就是 IDM 吗？BC 接近，再由 RL 超越，不正是原设想吗？
2.1 更准确地说，它不是“纯 IDM”，而是 PurePursuit + IDM

repo 使用的是 PurePursuitIDMPolicy：

IDM 主要处理纵向跟车和速度；
Pure Pursuit 处理横向路径跟踪；
最终 expert5k 采的是这套组合规则策略。

评测里称为 IDM 基线是一种简称，但严格说它是 PurePursuit-IDM 规则专家。当前冻结评测上的 success 是 0.756。

2.2 你的原始设想在逻辑上成立

正确链路确实可以是：

Expert demonstrations
    ↓
BC 获得稳定初始策略
    ↓
DAgger 修复 student-induced distribution shift
    ↓
RL 利用环境奖励超越 expert


所以“BC 难以超过专家”不等于“整个 BC→DAgger→RL 系统不能超过专家”。

但要区分三个阶段的能力边界。

BC

纯 BC 只在拟合专家条件动作分布。通常不能系统性超越专家，但有可能在有限评测集上略高于专家，原因包括：

网络对专家噪声做平滑；
模型学到跨场景共享规律；
LQR 执行模型计划与专家自身控制回路不同；
expert 数据带有局部次优动作，但监督平均后反而更平滑。

这种“偶尔超过”不能当作稳定超越机制。

DAgger

DAgger 的主要功能是纠正 covariate shift：

学生偏离专家分布
→ 在学生真实访问的状态上重新请求专家
→ 学会恢复到专家行为区域


如果所有 DAgger 标签仍来自同一个 PurePursuitIDMPolicy，它主要让学生更可靠地逼近该专家在学生分布上的行为，而不是提供超过专家的新决策知识。

不过，DAgger 仍可能帮助最终系统超过专家的一些脆弱实现细节，因为：

student+LQR 的执行器和 expert 自身执行器不同；
学到的 recovery 可能比 expert 原始轨迹更平滑；
网络会在多个相似专家状态之间泛化。

但本质上，它不是主要的“超专家信号源”。

RL

真正可以系统性超过 expert 的应是 RL，因为奖励并不要求复制 expert：

expert 撞车时可以学会更保守；
expert 超时时可以学会更高效；
expert 在收费站选择不好时，可以探索更优 route behavior； -采用 BC/KL 只作为先验，环境 return 才是最终目标。
2.3 但当前 repo 的 RL 还没有形成这条闭环

这里是关键。当前 PPO 训练时：

用采样的 2D action 做 log-prob 和记账；
执行参考采用 repeat_action，确保环境结果依赖被 PPO 记账的随机变量； -最终评测却使用模型生成的完整 plan；
primary、router、encoder、memory 和 ST-GNN 大部分被冻结；
PPO 只能通过 policy/value、experts 和 residual scale 部分影响计划。

因此现在不能简单说：

BC 到 0.65，再让 PPO 从 0.65 优化到 0.80。

因为 PPO 训练的执行对象和最终部署执行对象并不完全相同，而且它能修改的模块范围有限。

2.4 我的结论

你的总体战略没错：

BC 接近 expert，DAgger 修复分布偏移，RL 超越 expert。

但必须补一句：

RL 所采样并计算 log-prob 的 action，必须真正决定最终被 LQR 执行的轨迹。

否则“RL 超越 expert”在数学上有目标，在实现上却没有完整信用路径。

另外，P0 已显示 D2=0.706，仍低于 expert=0.756。这说明现有 (ds,dθ)+LQR 链即使输入专家动作，也可能有约 5pt 的接口/离散化/速度效率损失。若目标是 0.82–0.85，RL 不仅要补模型相对 D2 的约 37pt，还要进一步越过 expert 本身。这不是不可能，但不能只靠现有 Stage C 小范围微调。

三、问题 2：为什么 PPO 训练用 plan 会发生信用分配断裂？A-hold 是什么？

这是整个系统里最需要讲清楚的地方。

3.1 PPO 实际优化的是什么

PPO 每一步需要：

策略根据状态 sts_t 定义分布；
从这个分布采样动作 ata_t；
环境执行 ata_t；
得到奖励 rtr_t；
用同一个 ata_t 的概率：
log⁡πθ(at∣st)\log \pi_\theta(a_t \mid s_t)

计算 PPO ratio：

rt(θ)=πθ(at∣st)πθold(at∣st)r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)} {\pi_{\theta_{\text{old}}}(a_t \mid s_t)}

基本因果要求是：

reward 必须由 PPO 记账的 sampled action 导致。

3.2 当前系统里存在两种候选执行参考

假设 policy 在状态 sts_t 采样：

a_t = (ds_t, dtheta_t)


同时 PlanHead+WM 还会输出：

P_t = [p_t^1, p_t^2, ..., p_t^6]

repeat_action

构造：

[a_t, a_t, a_t, a_t, a_t, a_t]
→ arc_step
→ 3 秒 LQR reference


此时车辆接下来怎么走，明确依赖 sampled ata_t。

因此：

sampled a_t
→ LQR reference
→ physical trajectory
→ reward


PPO 对 ata_t 的 log-prob 有效。

这就是 A-hold。更准确说，它是 sampled action 的零阶保持：

A-hold = hold/repeat current sampled action across the preview horizon

3.3 plan 口径为什么可能断裂

如果执行的是：

PlanHead/WM 生成的完整 P_t
→ LQR
→ reward


但 PPO buffer 里记录的仍只是：

sampled a_t
log pi(a_t | s_t)


就要问：

完整 plan 的第 2 到 6 步究竟是否由 sampled ata_t 唯一决定？

如果不是，例如：

第一步被 action_mu 覆盖，而不是 sampled action；
后五步由 deterministic PlanHead 和 latent WM 生成；
policy sampled action 只是一个旁路输出；
PPO noise 没进入后五步；
LQR 主要跟随后五步计划；

那么同样一个 sampled ata_t，可能对应基本相同的执行 plan；或者不同 sampled ata_t，执行 plan 差别很小。

因果链变成：

sampled action a_t ────────┐
                           │  PPO 对它计算 log-prob
PlanHead deterministic P_t ├→ LQR → reward
                           │
reward 主要由 P_t 决定 ────┘


PPO 却把 reward 归因给 ata_t。

这就是信用分配断裂或失真。

3.4 一个具体例子

假设：

sampled action A: dtheta = +0.08
sampled action B: dtheta = -0.06


但 PlanHead 在两个情况下都输出同样的右弯 3 秒计划，因为后五步主要由 primary+WM 决定。

两个 episode 的实际轨迹几乎一样，奖励也一样，但 PPO 会分别提升或降低：

log pi(+0.08 | s)
log pi(-0.06 | s)


这不是有效的策略梯度，因为 reward 并没有真正区分这两个 sampled action 的后果。

反过来也可能发生：

sampled action 几乎相同；
PlanHead 因 router 或其他隐藏确定性输出变化产生不同计划；
reward 差异很大；
PPO 仍把差异归结给相同的 2D action。
3.5 A-hold 为什么被引入

A-hold 不是为了得到更好的最终驾驶计划，而是为了满足 PPO 的因果账本：

PPO 采样什么，就让环境执行什么。


所以 Stage C 采用 repeat_action，把 sampled 2D action 重复 6 次，再交给 LQR。这样虽然计划非常原始，但 PPO 的数学语义成立。docs\deepseekv4p1_argue.md 也明确记录，plan_reference="plan" 被定位为旧行为对照，repeat_action 才是保证执行依赖 PPO 记账变量的训练口径。

3.6 但 P0 证明 A-hold 不适合作为最终部署计划

双口径评测结果：

Arm P：plan 0.224，repeat 0.000；
pri512：plan 0.332，repeat 0.000；
s11：plan 0.646，repeat 0.418。

这说明：

A-hold 可以解决 PPO 记账问题；
但重复模型单步动作不构成高质量部署轨迹；
当前模型 action 的横向偏差被重复 6 次后会迅速放大；
所以不能把 repeat_action 直接当最终规划接口。

因此当前真正需要的不是在 plan 和 repeat_action 中二选一，而是：

让 PPO 直接采样一个会被完整执行的 trajectory action，并对这个 trajectory 的概率正确记账。

后面第 4 点详述。

四、问题 3：3 seeds 是什么？为什么这不是“依赖随机种子”？McNemar/bootstrap 又是什么？
4.1 3 seeds 是训练随机种子，不是拿 seed 调参

seed 会影响：

网络参数初始化；
minibatch 顺序；
dropout 或随机采样；
PPO action exploration；
rollout 场景顺序； -并行环境时序；
DAgger 数据采样； -优化器早期轨迹。

同一个架构、相同超参数，因为非凸优化，可能得到不同策略。

所谓 3 seeds 是：

配置完全相同
seed = 1001
seed = 2001
seed = 3001


分别独立训练，然后都在相同冻结 eval500 上评测。

它不是：

试 20 个 seed
挑表现最好的一个发布


后者确实是 seed overfitting，应当禁止。

4.2 为什么 deterministic eval 仍需要多 seed

deterministic=true 只说明：

-评测时不用随机 action； -场景 spec 固定； -同一个 checkpoint 在相同环境条件下大体可复现。

它不能消除训练随机性。

例如一个方案三次训练得到：

0.62, 0.79, 0.66


单拿 0.79 会让人误判方案有效。

另一个方案得到：

0.75, 0.76, 0.75


后者才是可靠改善。

所以 seeds 不是优化手段，而是压力测试：

证明改进不是偶然落入一个好局部最优。

我同意你的原则：“不要依赖 seed”。而确保不依赖 seed 的方法，恰好就是要求多 seed 全部稳定。

4.3 3 个 seed 够吗？

3 个是工程最低线，不是统计上很充分。

建议分阶段：

开发筛选阶段
1 seed；
clean150； -只淘汰明显无效设计。
候选确认阶段
3 training seeds；
clean500 + eval500； -固定 checkpoint-selection 规则； -报告均值、最差 seed 和范围。
最终声称“超过 IDM”

如果预算允许，最好 5 seeds。至少要求：

3-seed mean ≥ 0.80；
每个 seed 不低于某个下限，例如 0.77； -不是一个 seed 0.90、两个 seed 0.75 拼出平均值； -每个 seed 对 IDM 做相同 episode 的 paired comparison。
4.4 McNemar 是什么

你的 eval500 中，两个策略跑的是同一批 (scenario_id, seed)。

假设比较新模型 A 和 IDM B。对每个 episode 记录 success/failure：

	IDM 成功	IDM 失败模型成功	两者都成功	模型独赢
模型失败	IDM 独赢	两者都失败

McNemar 只关注两个不一致格：

模型独赢数量 bb；
IDM 独赢数量 cc。

例如：

模型独赢 = 70
IDM独赢 = 35


虽然总体 success 差可能只是几个点，但同一 episode 上模型赢的次数明显多于输的次数，可以检验这种非对称是否可能只是随机波动。

P0 已经列出了这种原始计数，例如 pri512 与 IDM 的 paired 比较中，IDM 独赢远多于 pri512，说明差距不是普通评测波动。

McNemar 特别适合：

-同一批 episode； -结果是 binary success/failure； -两个策略配对比较。

4.5 Bootstrap 是什么

Bootstrap 是对 episode 配对样本反复有放回重采样。

例如有 500 个 episode，每个 episode 计算：

delta_i = success_new_i - success_baseline_i


然后：

从 500 个 paired episode 中有放回抽 500 个；
计算 success difference； 3.重复 5,000 或 10,000 次； 4.得到 95% confidence interval。

可能得到：

新模型 - IDM = +0.048
95% paired bootstrap CI = [+0.016, +0.080]


这种情况下，提升不太可能只是场景抽样碰巧。

Bootstrap 还可以用于非二元指标：

route completion；
min TTC；
boundary margin；
jerk； -每 episode off-road steps。
4.6 推荐最终统计口径

不要只写“3-seed 平均 0.81”。建议写：

每个 training seed:
  success
  off-road
  collision
  route completion
  paired wins/losses vs IDM
  McNemar p-value
  paired bootstrap 95% CI

跨 seed:
  mean
  min
  max
  standard deviation


最终 gate 可设为：

3-seed mean success ≥ 0.80；
worst seed ≥ 0.77；
paired bootstrap lower bound > 0；
McNemar p < 0.05；
off-road 相对当前模型下降至少 30%；
collision 不高于 IDM，或至少没有统计显著恶化； -提升不只来自一种 road class。
五、问题 4：World model 为什么必要？规划层和控制层的正确接口是什么？

你的思路中有一半是对的：

实车低层控制需要预瞄轨迹，单个 0.5 秒动作不足以让 LQR 发挥作用；因此策略应输出一段未来计划，而不是只给一个瞬时动作。

我同意。

但我不同意这句话的强版本：

“world model 辅助 policy head 逐步出 action，action 连起来的轨迹就是唯一逻辑通畅的正确路线。”

P0 的 D1=0.000 对这种“开环未来 action chain”提出了直接反证。

5.1 LQR 本身没有规划决策

你的理解正确。LQR 是跟踪控制器，不是决策规划器。

它解决的是：

已有参考轨迹
+ 当前车辆状态
→ 计算 steering / acceleration 等低层控制
→ 减小 tracking error


LQR 不会决定：

-走左岔还是右岔； -是否换道； -是否为 cut-in 减速； -收费站选哪个通道； -是否绕开障碍； -未来轨迹应该是什么形状。

这些必须由上游 planner/policy 决定。

因此合理分层是：

Prediction / representation
        ↓
Planner or trajectory policy
        ↓ trajectory reference
LQR tracker
        ↓ low-level controls
Vehicle dynamics

5.2 但 trajectory policy 不必等于 world model

World model 的职责应该是：

给定其余交通参与者状态、自车状态和候选 ego action
→ 预测未来环境状态或风险


例如：

-周围车辆未来 occupancy； -道路边界相对关系； -碰撞概率； -自车未来 pose； -tracker execution envelope； -route progress； -plan feasibility。

Planner 的职责是：

利用这些预测，产生最优 ego trajectory


所以 world model 是“候选轨迹评价和状态演化模型”，并不必然是“轨迹生成器本身”。

可以有三种合理架构：

架构 A：直接 trajectory actor
obs/history
→ trajectory distribution
→ sample 6-step trajectory
→ LQR


world model 可作为辅助训练任务或 critic 输入。

架构 B：候选轨迹 + world-model scoring
trajectory proposals
→ world model rollout
→ cost/value scoring
→ select trajectory
→ LQR


更接近 learned MPC。

架构 C：自回归 trajectory policy
z0 → sample a0
world model(z0, a0) → z1
z1 → sample a1
...
→ [a0...a5]
→ trajectory
→ LQR


这是你提出的路线。它是合理候选，但不是天然正确，更不是当前 P0 证据下已经成立的路线。

5.3 P0 对自回归 action chain 的警告

D1 使用未来专家 action chain，按 (ds,dθ) 积分成 6 步计划，交给 LQR，结果 success=0.000、off-road=0.95。

D2 每次只取当前专家 action，并在当前 3 秒参考中重复，却达到 0.706。

差别不是 D1 的未来 action 更差，而是：

D1 future action chain 是在专家自己的未来状态上生成的；
当前 ego 被 LQR 执行后，不会精确处于专家影子轨迹的未来状态； -后续 action 不再适合当前实际状态； -错误在 6 步链中无法反馈校正； -D2 每 0.5 秒基于真实当前状态重新查询专家，天然重新锚定。

这给你的方案一个非常明确的设计约束：

自回归生成 action chain 时，第 k 步 action 必须基于“该候选 action 导致的预测状态”，而不能基于 GT/expert 的未来状态，也不能把独立的一步策略输出简单串起来。

即：

z_0 = encode(current real state)
a_0 ~ pi(. | z_0)
z_1 = WM(z_0, a_0)
a_1 ~ pi(. | z_1)
z_2 = WM(z_1, a_1)
...


这在概念上是通的。

但还缺三个关键条件。

条件一：PPO 必须对整条 trajectory 的概率记账

如果 trajectory action 是：

At=(at0,at1,…,at5)A_t=(a_t^0,a_t^1,\dots,a_t^5)

PPO 应记录：

log⁡π(At∣st)=∑k=05log⁡π(atk∣ztk)\log \pi(A_t|s_t) = \sum_{k=0}^{5} \log \pi(a_t^k | z_t^k)

而不是只记录第一步 at0a_t^0 的 log-prob。

否则后五步仍是未记账的决策变量。

更简单的实现是一次输出 12 维 trajectory distribution：

[ds0, dtheta0, ..., ds5, dtheta5]


并计算完整 12D action 的 joint log-prob。第一版不一定要马上做复杂 autoregressive PPO。

条件二：被采样的整条 trajectory 必须真正交给 LQR

不能：

PPO sample trajectory A
但 LQR 执行 deterministic mean plan P


必须是：

sampled trajectory A
→ arc_step / trajectory construction
→ feasibility projection if any
→ LQR
→ reward


否则信用链又断。

条件三：每 0.5 秒必须 receding-horizon replan

LQR 可以预瞄 3 秒计划，但不是说一条 3 秒计划要开环执行完整 3 秒。

正确方式是：

t = 0.0s:
  生成未来3s计划
  只执行前0.5s

t = 0.5s:
  读取真实新状态
  再生成未来3s计划
  只执行前0.5s


这就是 receding horizon。

所以你说“rollout 后 policy action 连成轨迹”是合理的，但必须是：

3 秒预瞄、0.5 秒重规划、只执行首段。

不能是 D1 那种未来 action chain 的开环信任。

5.4 当前 PlanHead、PolicyHead、WM 应如何重新定义职责

建议收敛成以下结构：

Encoder + Memory
    ↓
current latent z0

World Model
    输入: zk, candidate ak
    输出: predicted zk+1
          ego pose delta
          occupancy/risk
          road-boundary features

Trajectory Actor / Policy
    基于 zk 输出 action distribution
    rollout 6 steps
    得到完整 sampled action chain A_t

Trajectory Decoder
    用统一 arc_step 将 A_t 转为 pose trajectory
    输出 vehicle-footprint trajectory

Safety / Feasibility Head
    输出 boundary margin
    corridor violation
    collision risk
    curvature feasibility

LQR
    跟踪 sampled trajectory
    只执行首0.5s

下一策略步重新规划


这样只剩两个层级：

trajectory policy 决策；
LQR 执行。

PlanHead 和 PolicyHead 不应继续作为两个模糊的动作中心。可以：

-把现有 PlanHead 重构为 trajectory actor； -将现有 PolicyHead 变成每个 rollout step 的 action distribution head； -或者反过来，保留 PolicyHead 作为 trajectory actor，PlanHead 只负责 deterministic proposal/context。

但最终只能有一个明确的 stochastic policy whose log-prob is used by PPO。

5.5 World model 是否“必要”

我不会说它绝对必要。

P0 只证明：

-当前 plan 质量差； -开环 expert chain 也会崩； -反馈重锚很重要； -接口/LQR 不是第一瓶颈。

它没有证明：

-必须使用 learned world model； -当前 detached latent WM 已经有用； -没有 WM 就无法输出好轨迹。

因此正确结论是：

自回归 trajectory policy 需要一个状态推进器，但这个推进器可以是 learned WM、显式运动学模型，或两者结合。

对 ego 的短时 3 秒传播，完全可以先使用显式车辆运动学：

pose_{k+1} = arc_step(pose_k, ds_k, dtheta_k)


这部分没有必要交给神经网络猜。

learned WM 更应该预测：

-动态对象如何响应； -未来 occupancy； -交互风险； -道路语义和不可见状态； -tracker error envelope。

也就是说：

Ego kinematics 用显式模型，环境 interaction 用 learned world model。

这比让 latent WM 同时承担 ego pose rollout、对象、车道和计划生成更可辨识。

5.6 对你方案的最终裁决

我赞同：

-最终接口应是 trajectory，而不是裸单步动作； -LQR 是 tracker，不是 planner； -policy 应对将被执行的 trajectory 负责； -WM 可以帮助生成和评价未来轨迹； -3 秒 plan 应以 0.5 秒 receding horizon 重规划。

我反对：

-认为只要“rollout policy action 连成轨迹”就天然正确； -认为 D1/repeat-action/PlanHead 都可以简单排除； -认为 policy 只给第一步 action 的 PPO log-prob，却让后五步影响执行也没有问题； -把所有未来 ego 状态都交给 learned latent WM，而不使用明确运动学。

P0 的 D1=0 已经说明，没有闭环重锚和状态一致性的 action chain，比重复当前反馈动作还危险。

六、问题 5：footprint 插值到底要拿来做什么？

不是为了把评测做得更漂亮，而是解决当前最主要的失败：计划看似“中心点没出界”，车辆实体却已经压线或切出可行域。

6.1 为什么只检查 6 个 plan 点不够

当前计划间隔是 0.5 秒。假设车速 7 m/s，相邻点距离约 3.5m；高速时可能更远。

可能出现：

plan point k：在可行域内
plan point k+1：也在可行域内
二者之间的弧线：切过实线、路肩或收费岛


另外，plan point 通常表示车辆中心，而车辆不是一个点。

即使中心在车道内：

-车头外侧角可能压线； -车尾在急弯中扫出边界； -LQR tracking error 可能让车身外缘越界； -收费站狭窄通道中，中心线可行不代表 2m 宽车身可行。

6.2 footprint 是什么

在每个计划 pose：

(x, y, heading)


根据车辆长度和宽度生成矩形车身。

实际检查至少包括：

-四个角； -四条边中点； -车身中心。

然后在相邻计划 pose 间每 0.5 到 1.0 米插值一次，再检查完整车身是否落在 drivable region。

P0 已按这个思路加入 footprint/时间戳归因，发现模型超过一半失败带 plan infeasibility 因子。

6.3 它有四个用途
用途一：失败归因

区分：

Plan 本来就不可行
vs
Plan 可行但 LQR 执行偏离


这是 P0 的首要用途。

用途二：训练监督

为每条候选轨迹计算：

min boundary margin
fraction of footprint outside
first violating rollout step
corridor violation severity


使 planner 在越界发生前就得到稠密信号，而不是等 episode 最终 off-road 才收到惩罚。

用途三：候选计划筛选

即使策略生成轨迹，也可以在部署前：

-拒绝明显越界的 candidate； -对多个 candidate 选择 margin 更大的； -或通过小幅 projection 修正到可行域。

用途四：hard mining

将高风险样本定义为：

min_margin < threshold
future footprint violation
rapid margin collapse
large tracking residual


比当前按 BC action error 选 worst-50% 更对应闭环失败。

6.4 但要修正一个重要口径问题

P0 发现很多“off-road”其实来自：

-连续黄/白实线； -车道 flag； -环境 termination 的 line-crossing 语义；

而 footprint 几何初版只检查 lane surface，因此出现大量 anomaly。

所以不能只维护一个 drivable_inside。

建议显式拆成三种约束：

surface_valid:
    完整车身是否仍在道路可行表面

legal_corridor_valid:
    是否跨越禁止跨越的实线/隔离线

route_corridor_valid:
    是否仍属于导航允许的道路分支/通道


相应输出：

surface_margin
solid_line_margin
route_corridor_margin


否则模型可能学会“不掉出道路”，却仍然横跨实线，被环境判 off-road。

6.5 footprint 不应成为硬保守约束

如果一味最大化 margin，模型可能：

-在狭窄收费口停车； -过于贴车道中心； -不敢换道； -通过低速规避 off-road； -导致 timeout 上升。

D2 当前主要失败就是 timeout，而不是 off-road。

因此 geometrical loss 应有死区和任务权衡，例如：

margin >= 0.5m:
    no penalty

0 < margin < 0.5m:
    smooth warning penalty

margin <= 0:
    steep violation penalty


具体阈值要根据车辆宽度、道路类型和 lane width 归一化，而不是全局固定 0.5m。

更合理的是：

mnorm=available lateral clearancelocal corridor half-widthm_\text{norm} = \frac{\text{available lateral clearance}} {\text{local corridor half-width}}

狭窄收费口和高速宽车道不能用同一个绝对 margin 评价。

七、对你列出的 13 点，我的更新判断
立即做，且不需要再争论
episode/step 级归因；
drivable boundary、local width、route corridor 与 footprint；
signal placeholder 显式化；
history 消融；
明确 PlanHead、PolicyHead、LQR 的职责；
hard mining 改为闭环风险；
DAgger 存真实 recovery trajectory；
PPO 渐进解冻，但要在 trajectory action 信用链打通后做。
方向正确，但需要限定
OD 8m gate

repo 复核表明：

-主路径是 track-id 精确匹配； -8m nearest-neighbor 是缺 identity 时的 v1 fallback。

所以不应把它列为当前 off-road 主根因。应做 audit，统计 fallback 实际占比以及错误匹配率。如果 fallback 占比低于 1% 到 2%，降级处理。

Same encoder + detach latent target

你说“这貌似是密集监督通用做法”，部分正确。

同 encoder、target detach 常见于自监督方法，但通常需要额外稳定机制：

EMA target encoder；
predictor asymmetry；
stop-gradient；
variance/covariance regularization；
reconstruction target；
contrastive negatives。

这里的风险不在 “detach 本身违规”，而在：

online encoder 定义 target
transition 学这个 target
encoder 又同时被其他 loss 更新
缺少固定物理坐标约束
物理解码权重较弱


因此不是一定会坍缩，而是 latent loss 很难解释。

最便宜的验证不是马上加 EMA，而是做 probe：

latent 能否线性读出 ego pose、speed、boundary margin、road curvature；
Stage A 过程中，latent loss 改善是否对应 probe 改善； -未来 1/3/6 步 probe error 是否随 horizon 合理上升。

若 probe 健康，再决定是否需要 EMA。

Memory 的 LD 时序融合

LD 静态结构不一定需要 temporal attention。对齐误差、lane slot 抖动和局部遮挡可能使它有价值，但不能预设。

你的五组消融是合理的：

current only；
2 frames；
6 frames；
shuffled history；
repeated current。

最关键比较是：

6-frame real history
vs
6-frame repeated current


如果接近，说明历史动态价值低。

MoE

风险判断仍然成立，但 P1 当前先不要动。P0 已锁定 plan quality 是近期问题，而 MoE 去留要等计划监督对齐后做等预算实验。否则 MoE 可能只是替现有错误目标背锅。

World model

当前 detached WM 的长期信用问题仍成立。但根据 P0，我会调整措辞：

-不是“WM 与任务无关”； -而是“当前 WM 没有被证明能产生状态一致、可闭环重锚的 trajectory rollout”。

未来应把 ego 运动学和 learned environment prediction 分开：

ego pose: explicit arc/kinematic update
other actors/risk: learned WM

八、我对当前 P1 计划的反驳与重排

docs\deepseekv4p1_argue.md 当前建议：

P1-3 曲率项先行；
P1-1 recovery 数据；
P1-2 footprint/corridor； 4.组合 candidate。

我认为这个顺序偏向“先做最便宜的”，但不完全对应根因。

我建议改成：
P1-0：先冻结真正的 trajectory action 契约

在写 recovery schema 前先定义：

本策略的 stochastic action 到底是什么？


短期不能重构 PPO 时，可以保持现有 plan 部署，但必须明确：

Stage B 学什么；
Stage C 哪些参数能改变 plan；
PPO sampled action 如何进入 plan；
buffer 中存哪些 log-prob；
LQR 最终执行哪条 sampled/deterministic trajectory。

产出一页契约：

policy output
sampling point
action dimensionality
plan construction
logprob definition
executed reference
replan interval
gradient/trainable path

P1-1：真实 recovery trajectory 数据

这是主线。必须存：

student observation
student plan
student actual executed trajectory
expert action at current real state
expert 6-step recovery trajectory
surface/line/route corridor labels
T_plan/T_cross/T_term
road class


重点是 expert trajectory 要从当前 student state重规划，不是从 expert shadow future state拷贝 action chain。D1 已经证明后者危险。

P1-2：先启用离线可学的整段轨迹监督

把当前 traj_aux=0 改为只对真实 recovery trajectory 行生效。

必须监督：

6 步 pose/action chain； -横向累计位置； -航向； -速度或 ds； -有效 mask； -整段曲率。

不要继续用常量外推的 traj6。

P1-3：加入几何与曲率监督

曲率项可以先实现，但不应单独作为主要实验结论。

建议联合最小 recipe：

L=Laction+λtrajLtraj+λκLκ+λΔκLΔκ+λbLboundaryL = L_{\text{action}} + \lambda_{\text{traj}} L_{\text{traj}} + \lambda_\kappa L_\kappa + \lambda_{\Delta\kappa}L_{\Delta\kappa} + \lambda_bL_{\text{boundary}}

初始量级应通过 gradient norm 校准，而不是直接猜系数。让每项在训练早期对 plan-head 的梯度范数处于同一数量级，例如相差不超过约 3 到 5 倍。

P1-4：再打通 PPO trajectory credit

这是超越 IDM 的必要步骤，不应无限后推。

两种实现路线：

保守版：12D 一次性 trajectory action
policy outputs mean/logstd for 6×2
sample complete chain
joint logprob = sum over 12 dims
sampled chain → arc_step → LQR


优点：

PPO 记账清楚； -容易实现； -可以直接监督 expert chain； -不会存在“后五步没有 log-prob”。

缺点：

action dimensionality 增大； -独立 Gaussian 不自然； -需要 smoothness/correlation prior。
研究版：autoregressive trajectory PPO
a0 ~ pi(.|z0)
z1 = transition(z0,a0)
a1 ~ pi(.|z1)
...
joint logprob = sum_k log pi(ak|zk)


优点：

-符合你的 WM 设想； -能表示条件依赖； -未来 action 基于预测状态。

缺点：

-更复杂； -WM 误差会直接污染 action distribution； -PPO ratio、entropy、KL 都必须按整链处理； -detached state 会影响梯度路径； -训练稳定性风险大。

建议先做 12D trajectory action 作为因果正确 baseline。它跑通后，再判断 autoregressive WM 是否值得。

九、最终结论
关于 expert 与超越 IDM

你的原设想正确：

BC 接近 PurePursuit-IDM，DAgger 修复 student distribution，RL 用非专家奖励超越。

但当前实现缺的是：RL 所记账的 action 没有完全等价于最终部署的 trajectory action。

关于 PPO 与 A-hold

A-hold 是为保证：

PPO sampled action
→ executed LQR reference
→ reward


它解决信用分配，不解决计划质量。P0 repeat-action 归零证明它不适合作为最终部署接口。

关于 three seeds

不是靠 seed 优化，而是用不同 seed 证明方案不依赖偶然性。禁止挑最好 seed，报告 mean、worst、paired confidence interval。

关于 world model 和轨迹接口

最终接口应该是：

trajectory policy → LQR


而不是裸单步 action。但 world model 不是逻辑上唯一的轨迹生成方式。

你提出的 autoregressive action-chain 路线可以成立，前提是：

-每一步基于候选 action 产生的预测状态； -整条 chain 都有正确 log-prob； -整条 sampled chain 真正交给 LQR； -每 0.5 秒 receding-horizon 重规划； -ego 运动学优先使用显式模型； -learned WM 主要负责环境交互预测。

关于 footprint

它不是辅助指标，而是把 dominant failure 变成可学习信号：

-完整车身而非中心点； -节点间插值而非只查 6 点； -区分 surface、solid line、route corridor； -用于归因、loss、候选筛选和 hard mining。

不过不能单纯最大化 margin，否则会把 off-road 转成 timeout。

下一步最优先事项

我建议 coding agent 不要直接按现有 P1-3→P1-1 顺序推进，而是采用：

P1-0：冻结 trajectory-action 与 PPO log-prob 契约；
P1-1：从 student 真实状态采 expert recovery trajectory；
P1-2：启用真实整段 trajectory supervision；
P1-3：加入 footprint boundary + curvature/Δcurvature；
P1-4：实现完整 6×2 trajectory action PPO baseline；
达到约 0.70 后再做 WM autoregressive rollout、MoE、渐进解冻和超 IDM。

目前最需要避免的误区是：

把 D2=0.706 解读为“接口已经完全解决”，或者把 D1=0 解读为“未来 trajectory 没价值”。

真正结论是：

闭环状态反馈与每 0.5 秒重锚至关重要；未来计划只有在状态一致、整链记账、完整执行且持续重规划时才有价值。