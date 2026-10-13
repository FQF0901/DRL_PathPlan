这轮我看完后，结论其实比前面任何一次都清晰。

我先给最终判断：

Coding Agent 这轮基本是对的。

而且我认为 R31 已经把未来 2~3 周的大方向决定了。

目前最大根因已经不再是：

lane semantic
MoE
DAgger
PPO
RL

而是：

Stage A/B/C 与 Deploy
不是同一个系统


以及：

WM rollout分布
与
teacher-forced分布
差异极大

一、R31最重要的发现

其实只有两个。

发现1：Stage C ≠ Deploy

这是已经被彻底坐实的。

来自 round31_stagec_contract_audit.md：

当前训练

collect / update 实际是：

sampled action a_t
→ repeat(a_t, 6)
→ LQR
→ reward
→ PPO


即：

S3

当前部署

部署是：

a0 = Policy(z0)

z1 = WM(z0,a0)
a1 = Policy(z1)

...

plan = [a0...a5]

→ LQR


即：

S2

差别有多大？

K1e2：

S2:
success = 0.52

S3:
success = 0.02


pri512：

S2:
success = 0.36

S3:
success = 0


这不是：

2%
3%


的差异。

而是：

完全不同系统


Coding Agent 这里是完全正确的。turn45search3

发现2：部署链误差不是 WM teacher-forced 能解释的

R31 的结果很漂亮。

teacher-forced

WM 固有误：

ego latent

0.02 → 0.04

实际部署链

R30：

0.20 → 0.32


扩大：

8倍~10倍


Coding Agent 的结论：

WM固有误差 OK

问题出在部署链


我是认可的。

因为：

WM expert-state
并没有明显崩


真正崩的是：

WM rollout state
→ Policy
→ WM rollout state
→ Policy


的循环。

这就是经典的：

closed-loop distribution drift

二、我最认同的新结论

来自：

round31_audit_summary.md

这里其实已经回答了：

World Model 要不要重构？

我的答案：

暂时不要。

为什么？

因为：

TF:
0.02~0.04

Deploy:
0.20~0.32


这两者之间的差距：

远大于
WM容量提升
能带来的收益


说明：

你首先遇到的是 distribution mismatch，

不是 model capacity mismatch。

这非常重要。

很多团队会本能地：

加Transformer
加latent
加memory


但现在证据不支持。

三、我最认同 Coding Agent 的地方

是他反驳我：

Stage B 不应直接定性为 recovery_v1 那种一级 bug

这一点他是对的。

recovery_v1

是：

obs_t

配

label_t+N


直接错。

Stage B rollout

是：

预测状态
→ traj6真值


它的问题是：

分布不一致


而不是：

标签错时间


这是两种不同级别问题。

所以：

危险


但不属于：

犯规

四、但我依然坚持 Q2 必须做

这里我仍然和 Coding Agent 有一点不同。

他认为：

Q1:
已经回答

Q3:
已经回答

Q2:
新增审计项


这我同意。

但我认为：

Q2优先级没那么低

因为：

目前：

Stage B


实际上是：

预测状态 rollout

↓

轨迹监督


而：

未来动作


没有强监督。

round31_audit_summary.md 已记录：

尾5步没有直接动作标签

只有 traj6 监督


这很关键。turn45search4

换句话说：

未来动作链学成什么样：

只要最终traj6还能过


就可能被接受。

而：

Deploy
``

恰恰依赖：

```text
尾5步动作


去影响：

LQR preview


这就是：

Stage B风险


所在。

所以：

我支持 Coding Agent 继续做：

Q2a
Q2b
Q2c


这部分。

五、对你路线A的最新判断

这是我最想说的。

之前我还怀疑：

路线A
是不是太理想化


现在我基本不怀疑了。

因为：

R31 已经说明：

S2

是好的：

Policy
+
WM rollout
+
LQR


能达到：

0.52

S3

是坏的：

repeat(a_t)
+
LQR


直接：

0.02


说明：

未来5步
确实有价值


而不是装饰品。

所以：

路线B：

tail generator


不一定是正解。

我现在反而更支持：

路线A


但条件是：

先把 rollout consistency 搞明白。

六、我对下一步的建议

我会做一个优先级排序。

Priority 0

Q2

直接开工。

Q2-a

Q2-b

Q2-c


不训练。

Priority 1

WM action fidelity

这个还没答完。

现在证据：

动作影响弱


但还没回答：

为什么弱


必须区分：

pose-mediated conditioning

vs

显式action conditioning


哪个不足。

Priority 2

Stage C contract

已经证明：

S3 ≠ S2


下一步要决定：

Stage C
走Exact

还是

走Deploy-isomorphic

Priority 3

A4

Policy-conditioned WM adaptation

这个我会放后面。

因为：

Q1

Q2

Q3

还没全部闭合。

七、我最反对的事

现在不要做：

DAgger


原因非常简单。

目前：

Policy(real state)
↓

World Model
↓

Policy(predicted state)


链路都没闭合。

如果直接：

DAgger


只会让：

Policy(real state)


更强。

但：

Policy(predicted state)


可能继续漂。

这治标不治本。

最终裁决

我给 Coding Agent 的回复会是：

总体接受 R31。

尤其接受：

1. Stage C ≠ Deploy 已被坐实；
2. WM固有teacher-forced能力并非主问题；
3. Q1 和 Q3 已有一轮定量答案。

继续推进：

Q2a
Q2b
Q2c

作为最高优先级。

我坚持：

Stage B rollout-supervision 风险
仍需成为一级审计对象。

在Q2出结果前：

冻结
A4
DAgger
MoE
PPO
GRPO
大规模采集

不再新增训练。


如果让我给当前 root cause 排序：

P0：
Deploy chain consistency

P1：
WM rollout state
→ Policy
接口漂移

P2：
Stage B rollout supervision

P3：
WM action conditioning

P4：
lane semantic（已进入次要问题）


这是我目前最接近真实代码行为的判断。

===============================================================

总体结论

我仔细排摸了附件中的 World Model 相关代码，重点看了：

net/st_gnn.py
net/model.py
net/mem.py
net/encoders.py
net/plan_head.py
net/policy.py
pipeline/stages.py
pipeline/trainer.py
config/model.yaml
config/train.yaml

我的最新判断是：

当前最大问题不只是“先训练 WM、后训练 Policy 导致分布不匹配”，而是当前 WM 的状态定义、动作条件化、物理监督和动态图更新都存在结构性弱点。

其中最重要的不是网络容量，而是以下五个问题：

动作没有显式进入 latent transition，只通过预计 ego pose 间接进入；
ego latent 没有真正的下一时刻物理状态解码监督；
OD 图位置采用 t0 恒速外推，预测 latent 并不反过来更新图几何；
node mask 固定为 t0，entry/presence 只是诊断头，无法真正改变未来图状态；
latent target encoder 与 Transition 同时训练，且 target 只 stop-gradient，不是慢速目标网络，存在表示共适应和漂移风险。

因此：

我不批准把 WM 和 Policy 全部解冻后，直接用一个总 loss 端到端联合训练。

更合理的方式是：

先诊断 WM 性状
→ 修动作条件化和 ego 动力学锚点
→ 在 Policy 分布上适配 WM
→ 冻结 WM 训练 Policy
→ 再做“同时迭代、梯度隔离”的联合适配
→ 最后才考虑有限的端到端联合梯度

一、当前 WM 真实架构
1. 当前状态和转移

WM 的状态是：

z_ego
z_od[slots]
z_ld[slots]


每一步通过 net/st_gnn.py：

nodes = message_pass(z_ego, z_od, z_ld, node_mask, pose, step_index)

z_ego_next = z_ego + ego_transition([h_ego, z_ego])
z_od_next  = z_od  + od_transition([h_od, z_od])
z_ld_next  = z_ld  + ld_transition([h_ld, z_ld])


也就是说它是一个：

图消息传递
+
残差 latent transition


转移头采用零初始化，所以初始状态近似恒等映射。

2. 当前动作条件化

当前不是：

zk+1=WM(zk,ak)z_{k+1}=WM(z_k,a_k)

而是：

action(ds,dθ)
→ 运动学积分得到预计ego pose
→ 将ego pose放入图节点位置
→ pose影响图边特征
→ ST-GNN间接感受到action


ds/dθ 没有作为显式向量进入：

ego_transition
od_transition
ld_transition


而且进入图的累计 pose 被 detach。

因此 WM 能看到：

“预计ego在哪里”


但不直接知道：

这一步ds多大
dθ多大
曲率κ是多少
速度意图是多少
这一步是加速还是保持速度
动作接近哪个执行边界


这与 R30 的实测一致：动作引起的 latent 变化比自然状态差异小 1–3 个数量级，尤其 ds 通道最弱。

二、五个潜在架构问题
问题 1：动作条件化过于间接
当前链路
action
→ pose
→ edge geometry
→ message passing
→ latent transition


问题是不同动作可能具有相近的短期 pose，但动力学语义不同。

例如：

较大ds + 较大dθ
较小ds + 较小dθ


可能得到相似曲率路径局部形状，但：

-速度意图不同； -进度不同； -后续 LQR preview 不同； -与周车的相对速度变化不同。

建议

先做 Action Fidelity probe。如果当前间接条件化未过门，再增加零初始化的显式动作残差：

action_features = [
    ds_normalized,
    dtheta_normalized,
    kappa_normalized,
    speed_intent,
]


然后：

legacy transition
+
alpha_action × action-conditioned delta


初始化：

alpha_action = 0


保证旧 checkpoint 前向逐位不变。

这更接近标准 action-conditioned world model。DreamerV3 的世界模型也是根据状态和动作预测未来表示，Actor/Critic 再在 imagined trajectory 上学习，而不是只通过预计位置间接推断动作。

问题 2：ego latent 缺乏明确物理锚点

当前：

z_ego_next 只有 latent consistency；
ego_next_pred 来自 PlanHead； -并没有一个由 z_ego_next 直接解码得到的 ego physical state head。

这意味着 WM 的 ego latent 可能在 latent 空间里接近目标 embedding，却未必精确保留 Policy 真正需要的：

速度
进度
航向
yaw rate
目标车道横向误差
目标车道航向误差


尤其当前部署问题集中于：

ds尾段上冲
速度/进度漂移
横向修正与速度耦合


但这些量没有被一个明确的 ego_state_decoder(z_ego_next) 强监督。

建议

增加统一的 ego dynamics decoder：

z_ego_next
→ [
    delta_progress,
    speed,
    delta_heading,
    lane_lateral_error,
    lane_heading_error
]


不要给每个变量单独造一个复杂 head，用一个小 MLP 统一输出即可。

监督优先级：

progress / speed
heading
lane-relative errors


其中 lane-relative target 必须依据未来真实 ego pose 和同一 target-lane 契约重算，不能重复使用 t0 semantic。

问题 3：图几何没有由 WM 预测结果闭环更新

当前 OD 图位置使用：

t0 OD position + k × dt × t0 velocity


也就是恒速外推。

WM 即使预测了：

z_od_next
od_pred


这些预测也没有用于下一步图节点 pose。

因此：

WM预测OD减速、变道或交互


和：

下一步消息传递所看到的OD位置


可能互相矛盾。

在 merge、cut-in、roundabout、tollgate 等场景中尤其严重。

建议先探测，不要直接改

做三个图几何臂：

G0：当前恒速图
OD pose = t0 + k·v0

G1：Oracle future pose

下一步图位置使用真实 future OD pose。

G2：Predicted pose

使用 WM 物理解码的 predicted OD pose。

比较：

latent error
Policy interface error
action-chain error
部署闭环


如果：

G1明显优于G0


说明恒速图是结构瓶颈。

如果：

G2接近G1


才允许把 predicted pose 接回图。

如果 G2 很差，先修 OD physical prediction，不要闭环接回。

问题 4：未来 mask 不会变化

当前：

node_mask = t0 node mask


整个 rollout 都复用。

虽然存在：

presence head
entry head


但它们只是诊断输出，不改变：

未来node mask
未来slot latent
未来消息传递


于是：

-离开视野的对象仍继续进入图； -新进入对象只有 entry logit，没有真实 latent slot； -未来图拓扑无法表达目标出现或消失。

在 3 秒 horizon 内，这可能不是普通直道的第一瓶颈，但对：

cut-in
merge
intersection
roundabout


可能很重要。

建议

先做 mask oracle probe：

M0：固定t0 mask
M1：真实future mask，但latent保持当前预测


如果 M1 对困难路型有明显收益，再考虑：

soft presence mask


不建议立即做 hard slot creation，因为 entry object 没有可用的初始 latent。

第一版最多允许：

existing slot × sigmoid(presence)


来软衰减离场对象。

新对象 entry 的建模可以后置。

问题 5：latent target 是移动目标

Stage A 中 future latent target 来自：

model.encoders.embed_ego(future).detach()
model.encoders.embed_od(future).detach()
model.encoders.embed_ld(future).detach()


detach() 只阻止 target 分支接收这一次 backward，但 encoder 本身仍通过当前状态分支和其他 loss 继续更新。

因此训练过程中 target representation 也在移动。

可能出现：

WM与Encoder共同移动
→ latent loss下降
→ 物理预测和Policy决策充分性没有同步改善


这就是典型的 latent 共适应风险。

DreamerV3 会区分 dynamics loss 与 representation loss，并使用不同方向的 stop-gradient，以平衡“让预测追上表示”和“让表示保持可预测”，而不是只使用一个对称 latent 拟合目标。

建议

增加 EMA target encoder 对照：

online encoder：
生成当前z，参与训练

EMA encoder：
生成future latent target，不接收普通梯度


更新：

theta_target =
0.995~0.999 × theta_target
+
(1-tau) × theta_online


但不要和 action embedding 同一轮启动。

推荐顺序：

先做表示稳定性 probe； 2.若 target drift 或 collapse 成立； 3.再做 EMA target encoder canary。
三、必须新增的 probes / hooks / 桩函数

所有 hook 必须：

默认关闭
零行为变化
不写入checkpoint
不改变随机数序列
不影响batch normalization

Probe 1：Action Jacobian

计算：

Ja(k)=∂zk∂a0J_a(k)= \frac{\partial z_k}{\partial a_0}

以及：

Jπ(k)=∂π(zk)∂a0J_\pi(k)= \frac{\partial \pi(z_k)}{\partial a_0}

输出：

||∂z_ego/∂ds||
||∂z_ego/∂dθ||
||∂z_od/∂ds||
||∂z_od/∂dθ||
||∂a_k/∂a_0||


按：

h1...h6
road class
speed
curve
outcome


分层。

重点检查
梯度接近0：
action conditioning太弱

随horizon指数增长：
rollout不稳定

左右极不对称：
动作或几何编码不对称

Probe 2：Action Swap

同一个真实状态，交换另一个样本的 action：

真实匹配action
随机action
同路型action
相反转向action
zero action


检查 WM 能否区分。

指标：

next-state ranking accuracy
correct action产生的预测是否最接近真实next state


验收建议：

真实action top-1匹配率 >=70%
真实action top-3匹配率 >=90%


初次仅作为相对门，实测后锁定。

Probe 3：Pose-only Shortcut

比较四种条件：

P0：正常action→pose
P1：保持pose，shuffle action token
P2：保持action，shuffle pose
P3：全部shuffle


在尚无动作 token 时，可对未来显式 action embedding canary 使用。

它能回答：

WM到底依赖action
还是只依赖pose geometry？

Probe 4：Ego Dynamics Sufficiency

从 z_ego_k 线性或小 MLP 解码：

progress
speed
heading
lateral error
heading error


分别比较：

real future latent
WM predicted latent
frozen-t0 latent


如果真实 future latent 可解码、WM latent 不可解码：

WM丢失物理语义


如果两者都不可解码：

Encoder state定义本身不充分

Probe 5：Latent Rank / Collapse

每个 horizon 输出：

per-dimension variance
effective rank
pairwise cosine
sample-to-sample distance
temporal distance


建议至少监控：

effective_rank(z_ego)
effective_rank(z_od pooled)
effective_rank(z_ld pooled)


若随 horizon 快速降秩：

latent collapse


若所有状态越来越相似，Policy 必然对尾段状态失去区分力。

Probe 6：Prior 与 learned residual 的贡献

物理解码当前是：

constant-velocity prior
+
learned residual


输出：

prior error
residual error
final error
residual/prior norm ratio


按 horizon 分层。

如果 final 的优势几乎全部来自 prior：

learned WM贡献有限


如果 residual 很大但 final 不改善：

残差头在制造噪声

Probe 7：动态图 Oracle

前述三臂：

current CV graph
oracle future graph
predicted future graph


输出：

state prediction
Policy interface
action chain


这是决定是否重构 ST-GNN transition 的关键 probe。

Probe 8：Mask Turnover

统计真实 horizon 1–6：

object disappear rate
new object entry rate
slot identity turnover


如果三秒内 turnover 很低，固定 mask 暂时可保留。

如果复杂路型 turnover 明显，则 presence/entry 必须从诊断头升级为状态机制。

Probe 9：Policy Latent Lipschitz

在真实 latent 周围和 WM latent 周围分别测：

Lπ=∥π(z+ϵ)−π(z)∥∥ϵ∥L_\pi = \frac{\|\pi(z+\epsilon)-\pi(z)\|} {\|\epsilon\|}

如果：

WM状态误差不算大
但Policy动作误差很大


可能是 Policy 在 WM 分布附近梯度过大，而不是 WM 单独很差。

这将决定修：

WM


还是修：

Policy robustness / interface consistency

Probe 10：Gradient Conflict

分别对以下 loss 单独 backward：

L_wm_latent
L_wm_physical
L_ego_dynamics
L_action
L_traj
L_interface


在以下模块报告梯度余弦：

Encoder
ST-GNN
PolicyBackbone
Primary
PolicyHead


输出：

grad_cos_matrix.json
grad_norm_by_module.json


尤其检查：

WM latent vs ego physical
WM physical vs Policy interface
Action vs trajectory
Trajectory vs interface

Probe 11：Joint-training Drift

若进行联合训练，每 N updates 固定同一 anchor set，保存：

Encoder(real obs) drift
WM(real state, action) drift
Policy(real latent) drift
Policy(WM latent) drift
physical decoder drift


没有这个 hook，不允许联合训练。

四、WM 与 Policy 是否应该同时解冻？
我的明确答案

最终可能需要共同适配，但不能“全部打开、一个总 loss、直接联合反传”。

原因是 WM 与 Policy 可以“串通”降低接口 loss：

WM产生一个不真实但Policy喜欢的latent
Policy适应该latent
interface loss下降
但世界预测失真


这种模型利用问题在世界模型 RL 中很常见。TD-MPC2 也强调学习的是控制有用的 latent dynamics，并通过状态转移、奖励和值目标共同约束，而不是只靠策略梯度定义世界表示。

五、推荐的联合训练策略
Phase 0：当前 WM 性状审计

不训练。

完成上述 Probe 1–11 的必要子集：

Action Jacobian
Action Swap
Ego Dynamics Sufficiency
Latent Rank
Dynamic Graph Oracle
Policy Lipschitz
Gradient Conflict

Phase 1：WM 独立预训练/适配

训练：

Encoder
ST-GNN transition
WM physical heads


冻结：

PolicyBackbone
Primary
MoE
PolicyHead
ValueHead


数据：

expert transition 60%
policy Exact transition 40%


目标：

latent prediction
ego dynamics
OD/LD physical
presence/entry


不使用 Policy loss 更新 WM。

Phase 2：Policy 独立训练

冻结：

Encoder
WM


训练：

PolicyBackbone
Primary
PolicyHead


监督两类输入：

真实状态
Policy(real latent)
→ expert action

WM 预测状态
Policy(predicted latent)
→ stopgrad Policy(real future latent)


也就是：

Linterface=∥π(z^k)−sg⁡(π(zkreal))∥L_{\text{interface}} = \|\pi(\hat z_k)-\operatorname{sg}(\pi(z_k^{real}))\|

并保留真实状态 expert action loss。

Phase 3：交替联合适配，梯度隔离

这是我最推荐的“联合训练”。

每个 cycle：

Step W：更新 WM
冻结Policy
用真实transition更新WM


损失：

WM latent
ego physical
OD/LD physical
presence/entry

Step P：更新 Policy
冻结WM
用real/predicted latent更新Policy


损失：

expert action
interface consistency
trajectory


也就是说：

同一阶段交替训练，但 Policy loss 不反传进 WM，WM loss 也不反传进 Policy。

这与“完全冻结后永不适配”相比，可以持续追随新 Policy 分布；同时避免两者通过虚假 latent 共谋。

Phase 4：有限联合梯度 Canary

只有 Phase 3 仍出现：

WM物理预测很好
但E_interface仍明显过大


才允许有限的端到端梯度。

开放：

ST-GNN transition最后一层
PolicyBackbone最后一层
PolicyHead


保持冻结：

Encoder
前层ST-GNN
Primary主体
MoE


学习率：

WM transition final：0.01×
PolicyBackbone final：0.05×
PolicyHead：0.1×


关键限制：

L_interface 对WM的梯度权重很小
WM physical loss仍是WM主梯度


可使用梯度投影：

若 interface gradient 与 physical gradient 余弦 <0
则去除冲突分量


不一定立即引入完整 PCGrad 库，第一版可只监控并拒绝冲突 batch。

六、阶段验收标准
Gate W0：状态定义充分
真实future latent对progress/speed/heading线性解码R² >=0.8
lane-relative lateral/heading R² >=0.9
effective rank不低于t0的70%


如果真实 latent 本身不可解码，先修改 state representation，不训练 WM。

Gate W1：Action Fidelity
ds-progress单调率 >=90%
dθ/κ方向正确率 >=95%
whole-scene mirror >=95%
真实action ranking top-1 >=70%
zero-action与normal-action可分

Gate W2：World Prediction

相对当前 A0：

policy-distribution h1 state error下降 >=30%
h3–h5 state error下降 >=25%
teacher/expert val退化 <=10%
OD prediction优于CV baseline
ego-dynamics显著优于frozen-state baseline


不能只看 latent cosine。

Gate P1：Policy Interface
E_interface h1–h2下降 >=30%
E_interface h3–h5下降 >=40%
E_deploy h3–h5下降 >=35%
真实状态action MAE退化 <=5%
Policy latent Lipschitz P95不增加

Gate J1：交替联合适配
连续两个cycle：
WM物理指标不退化
Policy真实态不退化
E_interface持续下降
部署plan无尾段ds overshoot
LQR calib50 success >=当前0.56
off-road/collision/timeout不恶化


注意 MoE-off 口径当前约 0.56，应作为同口径基线，而不是继续用 MoE-on 0.52。

Gate Deploy

进入 DAgger/MoE/RL 前，至少：

clean150 LQR+WM-plan success >=0.55
3 seeds mean >=0.53
off-road <=0.25
collision <=legacy
timeout <=legacy+3pp
WM plan明显优于repeat
Exact与LQR+plan差距可解释

七、对 MoE 最新结果的判断

你补充的 MoE-off 结果：

K1e2 LQR+plan：
MoE-on  0.52
MoE-off 0.56

Exact：
MoE-off 0.56

Repeat：
0.02 → 0.06，仍崩


结论：

训练口径 MoE-off 略优 2–4pp；
S1≈S2 仍成立；
S3 严重失败仍成立；
当前未经训练的 MoE-on 不仅没帮助，还有轻微负贡献； 5.之后所有 WM/Policy 接口实验应默认使用 MoE-off，保持训练与评测一致；
MoE-on 只作为后期专项专家训练后的正式候选。

当前未训练或分布不匹配的 experts/router 会给接口审计加入额外噪声，因此现在开着它们没有价值。

八、建议 Coding Agent 下一步任务
启动 Round-32：WM Architecture & Joint-Training Audit。

第一阶段只读：

1. Action Jacobian：
   ∂z_k/∂a_0、∂a_k/∂a_0，h1–h6。

2. Action Swap：
   正确action、同路型action、随机action、反向action、zero action。

3. Ego Dynamics Sufficiency：
   real latent / WM latent / frozen latent
   解码progress/speed/heading/lane error。

4. Latent Rank：
   variance/effective-rank/cosine随horizon变化。

5. Prior-vs-Residual：
   OD/LD decoder先验、残差、最终误差分解。

6. Graph Oracle：
   CV pose / oracle future pose / predicted pose。

7. Mask Turnover：
   horizon 1–6 presence/entry/slot turnover。

8. Policy Lipschitz：
   real latent与WM latent邻域动作敏感性。

9. Gradient Conflict：
   WM latent/physical/ego/action/traj/interface
   在Encoder/ST-GNN/Backbone/Policy上的grad norm与cosine。

10. Joint Drift Hook：
    为后续联合训练准备固定anchor集。

默认：
MoE-off；
K-anchor-off；
LQR不改；
不训练；
不加action token；
不删detach。


完成后只允许三种决策：

GO_ACTION_CONDITIONING
GO_STATE_REPRESENTATION_FIX
GO_ALTERNATING_WM_POLICY_TRAINING


如果 probes 不能定位，则：

BLOCKED_INSTRUMENTATION

最终判断

当前不应直接“WM 和 Policy 全部解冻一起训”。

正确的演进顺序应是：

WM结构和状态定义审计
→ WM在Policy分布上独立适配
→ Policy在real/predicted latent上独立训练
→ WM/Policy交替更新、梯度隔离
→ 极小范围联合梯度canary
→ DAgger
→ MoE
→ PPO


这既保留你的总范式：

IL + RL + MoE + World Model + 6-step rollout


也避免 WM 和 Policy 通过一个不真实但容易控制的 latent 互相迁就。

真正要优化的不是：

WM loss最低


而是同时满足：

WM(z,a)≈zreal,nextWM(z,a)\approx z_{\text{real,next}}

以及：

π(WM(z,a))≈π(zreal,next)\pi(WM(z,a)) \approx \pi(z_{\text{real,next}})

并最终让：

Policy–WM rollout 6步
→ fixed LQR


在闭环中稳定超过当前 0.56，逐步逼近并最终超过 IDM+Pure Pursuit 的 0.756。