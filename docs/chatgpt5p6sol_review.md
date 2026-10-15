你提出的方向大部分成立，但有两点需要修正：

真实 Memory Bank 与想象 Memory Bank 分离是正确的，但想象侧不一定要复制并反复编码完整 raw feature FIFO。更稳的是复制真实记忆的编码状态，再维护本次 rollout 专用的结构化 anchors、latent 和 validity FIFO。
不能先凭直觉扩大 World Model 或 Policy 参数量。当前最大问题已有证据指向“第 2–6 步状态接口错误”，先修状态语义和梯度链，再通过等预算缩放实验决定容量。
一、当前第 2–6 步接口确实不合理

当前 Structured Policy 同时读取：

latent
+
当前速度
道路余量
静态距离
动态距离
TTC
路线进度
目标车道末端距离


问题不在“显式信息一定不能读”，而在于它们必须是当前 World Model 状态的统一读出，不能是另一条与 World Model 平行的外部规划旁路。

当前第 2–6 步大致这样得到：

速度：保持上一值
动态距离：保持t0值
TTC：保持t0值
路线进度：保持t0值
左右道路余量：根据想象ego位置重新计算
静态距离：t0距离减累计进度
目标车道末端距离：t0距离减累计进度


代码中明确写着：

speed / dynamic_distance / dynamic_ttc / nav_progress：hold


这正是 Round-38 识别出的主要问题。第 2–6 步虽然 World Model latent 在变化，但纵向 Policy 读取的一部分关键状态仍然被冻结或以简化公式更新。

正确边界

允许 Policy 读取：

CurrentWorldStateView = Readout(WMState)


其中包括：

-当前 ego 速度； -车道边界余量； -动态对象距离和相对速度；

TTC； -道路级路线进度； -到候选路径末端距离。

但这些量必须由：

解析anchors
+
World Model预测残差
+
统一确定性读出


在每个 imagined step 重新计算。

因此问题不是“Policy 只能读取 latent 向量”，而是：

显式物理读出必须属于 World Model state API，不能由评测脚本、t0 常量或独立旁路临时供应。

像 DreamerV3 这样的世界模型也不是要求 Actor 读取一个孤立 latent，而是使用包含递归信念状态和随机状态的完整模型状态；训练阶段用真实 observation 修正 posterior，想象阶段依赖 prior dynamics 推进。

二、双 Memory Bank 思路基本正确
1. Real Memory Bank

保存真实环境历史：

真实ego feature
真实OD feature和slot identity
真实LD feature和anchor
真实others/nav上下文
valid mask


要求：

FIFO
episode reset
场景切换reset
只压入真实观测
绝不写入imagined状态

2. Imagination Memory

每次六步 rollout：

从Real Memory的当前快照初始化
→ 压入World Model预测状态
→ FIFO推进
→ rollout结束立即销毁


绝不能写回 Real Memory。

这个理念与稀疏自动驾驶系统中的 temporal instance memory 一致。SparseDrive 使用固定长度实例记忆队列保存对象实例，并通过 ego pose 投影历史 anchors 以支持跟踪、运动预测和规划。

但不建议机械复制完整 raw FIFO

更稳的结构是：

RealMemory：
完整真实features FIFO

ImaginationMemory：
初始真实memory encoding
+ predicted ego/OD/LD anchors FIFO
+ predicted latent FIFO
+ validity / identity FIFO


理由：

-避免每一步重新编码大量 synthetic raw features； -防止预测误差被“解码成raw feature再重新编码”二次放大； -保持 OD slot identity 和坐标语义； -减小计算量； -更接近 recurrent world-state，而不是模拟一套伪 observation history。

当前 repo 不完全符合你的期望：

MemBank 本身确实有 clone() 和 shift_*()； -注释也写明 clone 是 rollout 隔离副本； -但当前正式 rollout 直接迭代 z_ego/z_od/z_ld； -每步预测 latent 后 detach()，并没有维护一套真正被下游消费的 imagined Memory FIFO； -历史实验还证明，真实历史顺序和内容几乎没有被 Policy 利用，复制当前帧六次与完整历史近似。

所以现状更像：

Real Memory负责初始编码
→ 后续直接latent递归


而不是你描述的双层 Memory 设计。

三、完整当前帧 + 历史相对增量是值得验证的

你的建议可行，但不能对所有 feature 简单做减法。

建议下游输入采用：

current_full_feature
+
history_delta_to_current
+
history_valid
+
frame_age


例如：

Ego
当前完整状态
Δspeed
Δacceleration
Δyaw
Δcurvature
Δprevious_action

OD

必须先按：

slot identity对齐
+
ego-motion坐标补偿


再计算：

Δposition
Δvelocity
Δheading
Δsize
presence transition

LD

先将历史 lane anchors 投影到当前 ego frame，再计算：

Δgeometry
Δheading
Δcurvature
Δvisibility


不应做 delta 的字段：

object type
lane type
raw ID
mask
presence category
route role


这些使用：

当前值
+
transition indicator


而非数值相减。

推荐先做三臂对照
M0：当前TemporalAttention
M1：current full + aligned deltas
M2：current full + aligned deltas +轻量attention


由于 Round-38 已证明真实历史内容净贡献接近零，不应直接增加更复杂的 Transformer。先证明 aligned delta 能带来：

低速恢复
cut-in
merge
WM-state Policy一致性


的增益。

四、所有 detach 必须建立账本

当前值得重点检查的不是 hook 中的 detach()，而是主计算链中的四类。

合理的 detach
debug hooks保存
日志统计
checkpoint比较
route/nav作为不可学习环境输入
Stage A中阻止WM loss更新Policy

高风险 detach
1. 每步 latent detach
z_ego, z_od, z_ld = z_next.detach()


结果是：

h6误差不能回传修正h1 transition


当前 Stage A 实质上不是多步反向传播。

2. action detach 后送入 World Model
wm_action = action.detach()


Stage A 中合理，但在 Policy imagined-state consistency 训练中可能阻断预期梯度。

3. pose detach
ego_pose_det = pose.detach()


会阻止后续状态误差回到早期动作和曲率。

4. progress_s += ds.detach()

会切断 imagined geometry 对纵向动作的跨步梯度。

不能简单删除这些 detach。成熟 latent-world-model 方法也大量使用 stop-gradient 来分别训练 dynamics 和 representation，但其位置必须由明确损失职责决定。DreamerV3 的 dynamics loss 与 representation loss就在 prior/posterior 两侧有不同 stop-gradient，以避免两个网络无约束地互相追逐。

必做 Probe

对下列损失分别单独 backward：

h1 state loss
h3 state loss
h6 state loss
h1 Policy loss
h3 Policy consistency loss
h6 Policy consistency loss


记录到：

Encoder
Real Memory Encoder
Imagination Memory
ego transition
OD transition
LD transition
PolicyBackbone
Structured Policy lateral
Structured Policy longitudinal


输出：

梯度是否到达
grad norm
update norm
梯度余弦


然后做四臂：

D0：当前detach
D1：2-step truncated BPTT
D2：3-step truncated BPTT
D3：只开放状态接口residual的跨步梯度


不直接开启六步全 BPTT。

五、参数量现在不能判“太小”，需要先拿实际账本

仓库已经有：

net/param_probe.py
总预算 <= 1,500,000


并会统计：

encoders
mem_encoder
plan_head / PolicyBackbone
MoE
st_gnn / World Model
policy
value


但本环境缺少 PyTorch，无法代 Coding Agent 运行实际训练配置。因此下一轮必须提交真实配置下的参数账本，而不是估算。

建议容量原则

在 MoE 关闭的当前阶段，优先确保：

World Model + structured state interface：
约35%–45%

Encoder + Memory：
约20%–30%

PolicyBackbone + Structured Policy：
约25%–35%

Value + disabled MoE：
尽量低于10%


这不是硬比例，只是异常检测范围。

真正的容量判据是等预算缩放：

0.75× WM / 1.0× Policy
1.0× WM / 1.0× Policy
1.25× WM / 0.85× Policy
0.85× WM / 1.25× Policy


总参数均不超过 1.5M，然后比较：

训练误差
验证误差
h1–h6 rollout误差
真实态/预测态动作差
Clean500闭环


如果扩大 WM 明显降低验证和闭环误差，才说明 WM 欠容量；如果只降低训练误差，则是数据、状态语义或过拟合问题。

特别注意：当前关闭的 MoE 若仍占用大量参数，应从正式 Primary 参数预算中移出，不能让“闲置专家”挤占 World Model 和 Policy 容量。

六、World Model 应使用 learner-state transition，原则上同意

但准确说，这不是“用专家标签训练 WM”，而是：

learner真实状态
+
环境实际执行动作
→
真实下一状态


Policy 的 DAgger 标签是：

learner state → expert action


World Model 的标签则来自环境真实转移：

learner state + executed learner action → next real state


两者共用采集，但必须分开训练。

这正好缓解：

专家动作分布
vs
学生错误动作分布


导致的 World Model covariate shift。DAgger 的核心价值本来就是在学习器实际访问状态上查询专家并聚合数据，而不是重复采集专家自己的正常状态。

执行纪律：

先建立稳定基础WM和Primary
→ learner roll-in采集
→ 冻结Policy更新WM
→ 冻结WM更新Policy
→ 不允许单一loss联合反传

七、下一阶段正式方案
阶段目标

建立一个统一、递归、可训练的 imagined world state：

Real Memory
→ 初始化Imagination Memory
→ analytic anchor update
→ learned residual
→ unified state readout
→ 同一Policy决策
→ 推进六步


主要解决：

第2–6步speed / margin / distance / TTC / route progress冻结或错位
真实历史未有效利用
多处detach阻断多步学习
官方入口和checkpoint仍未闭合

1. 状态架构
WorldState_k:
  ego_anchor + ego_latent
  od_anchors + od_latents + identities + masks
  ld_anchors + ld_latents + masks
  route/nav state
  memory_state


统一读出：

PolicyStateView_k = readout(WorldState_k)


PolicyStateView 可以包含显式物理量，但只能从同一 WorldState_k 派生。

禁止：

评测脚本单独注入
t0动态量静默hold
缺失值填0
外部脚本替World Model提供未来状态

2. 双 Memory
Real Memory
只存真实完整features
FIFO
episode/scenario reset

Imagination Memory
rollout开始复制Real Memory的编码状态和最新结构化state
每步压入predicted anchors/latents
FIFO
rollout结束销毁
绝不写回Real Memory


三种表示做等预算消融：

full-frame memory
current full + aligned deltas
current full + aligned deltas + light attention

3. 想象步动力学

每步：

Ak+1base=AnalyticUpdate⁡(Ak,ak)A_{k+1}^{base} = \operatorname{AnalyticUpdate}(A_k,a_k) ΔAk+1,zk+1=WMθ(Sk,ak,Mkimag)\Delta A_{k+1},z_{k+1} = WM_\theta(S_k,a_k,M_k^{imag}) Ak+1=Ak+1base+ΔAk+1A_{k+1} = A_{k+1}^{base} + \Delta A_{k+1}

然后重新计算：

speed
road margins
dynamic distance
TTC
route progress
distance to lane/path end


再送入同一个 Policy。

不新建第二 World Model，不添加第二轨迹头。

4. 训练顺序
Phase A：状态接口监督

使用真实连续 transition，训练：

analytic state → learned residual → actual next state


先 1-step，再 2-step、3-step truncated BPTT。

Phase B：Structured Policy

冻结 WM，训练：

real-state action
WM-state decision consistency

Phase C：交替适配
WM-step：
冻结Policy

Policy-step：
冻结WM

Phase D：learner-state aggregation

Policy learner-state 查询专家动作；WM 使用实际执行动作后的真实 transition。

暂不启用 MoE、PPO 或 GRPO。

八、必须新增的 Probes / Hooks
状态接口 Hook

每步记录：

WorldState来源
analytic anchors
learned residual
final anchors
PolicyStateView所有字段
字段valid mask

双 Memory Hook

记录：

Real Memory reset/shift次数
Imagination Memory create/shift/destroy次数
每帧来源real/predicted
slot identities
frame ages
delta norm


自动断言：

imagined数据不写回Real Memory
episode切换后Real Memory全reset
rollout结束Imagination Memory销毁

Detach Ledger

每个 detach 记录：

tensor
调用位置
所属stage
预期阻断对象
实际梯度可达范围


CI 禁止未经登记新增生产 detach。

Parameter Ledger

输出：

总参数
激活参数
冻结参数
每模块参数
每模块grad/update norm
MoE关闭时浪费参数


总量硬门：

<=1,500,000

Oracle/替换 Probe

分别替换：

ego physical state
OD state
LD state
memory state
PolicyStateView


用于验证新接口后，latent 替换是否仍然“无恢复”。

九、必须在本轮修掉的四个工程缺口
官方入口装载 Structured Policy，不能继续只靠实验脚本；
checkpoint migration 接入正式 loader，并落盘 LoadReport；
所有生产调用迁移到forward_mode，删除旧布尔 shim；
在最新 HEAD 重跑统一 Clean500，旧冻结树成绩只保留为历史登记。

同时原生接入：

evaluation_protocol_hash
model tree hash
schema hash
checkpoint hash

十、验收准则
架构门
唯一Policy
唯一World Model
Policy调用6次
World Model决策推进5次
双Memory严格隔离
第2–6步无动态字段静默hold
无缺失值静默填0

状态接口门

在 Round-38 同一 A1 数据上：

低速恢复子集 Σds >= 12.8m
全样本累计误差 <= 2.31m
h1–h6动作误差不指数增长
learned residual显著优于analytic-only

Memory 门
真实历史必须显著优于repeat-nearest
打乱历史必须产生可测退化
否则不增加Memory容量


建议最低效应：

累计进度误差改善 >=10%
或困难场景闭环提升 >=3pp

Detach 门
h3/h6 loss按设计触达允许模块
冻结模块grad=0
2/3-step BPTT优于1-step
不出现梯度爆炸

参数门
总参数 <=1.5M
无关闭MoE挤占主模型预算
缩放收益以验证/闭环为准

统一闭环门

最新 HEAD 同一 protocol：

先重跑C0和IDM+Pure Pursuit
再跑新候选


晋级要求分两层：

相对晋级门
success至少比最新C0高5pp
collision/off-road不劣于最新C0
timeout下降

最终目标门
3-seed Eval500 success > IDM+Pure Pursuit
同时collision/off-road不劣于IDM

给 Coding Agent 的执行摘要
Round-39 聚焦统一 imagined world state，不再给Policy扩展独立path_*旁路。

1. 实现Real Memory与Imagination Memory隔离；
2. imagined memory保存预测anchors/latents，rollout结束销毁；
3. 第2–6步全部Policy物理读出由WorldState_k统一派生；
4. analytic anchor update + learned residual；
5. 建立detach ledger，测试1/2/3-step BPTT；
6. 做full-history vs current+aligned-delta对照；
7. 提交真实参数分布与四臂等预算缩放，总量<=1.5M；
8. learner rollout同时生成：
   Policy专家标签；
   WM真实executed transition；
9. 修复官方入口、checkpoint loader、forward_mode调用点和最新HEAD Clean500；
10. Primary和WM过门前保持MoE/DAgger正式训练/RL关闭。


最核心的判断是：

你的双 Memory 与 delta 表示方向值得验证；World Model 也必须学习学生错误动作导致的真实后果。但当前第一优先级仍是把第 2–6 步的物理状态、latent、memory 和 Policy 输入统一为同一个可递归 World State，并借助梯度与替换探针证明每个模块真实贡献。