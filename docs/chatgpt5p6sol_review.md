你说得对。之前的优化和负结果都很宝贵，但现在到了必须重新核对“设计意图、代码真实行为、训练梯度、部署链路”是否一致的节点。不能一边假设 Stage A/B/C 按理想方式工作，一边继续追加 loss 或结构。

我已结合最新 DRL_PathPlan_Code.txt 和 PPOGRPO.docx 做了第一轮代码级排摸。下面给出的是“先审计、再定架构、最后分阶段优化”的完整开工方案。

一、先锁定本轮的架构原则
1. 唯一动作生成者是 PolicyHead

正式定义：

ak=πϕ(zk)a_k=\pi_\phi(z_k)

其中：

a_k = 当前0.5秒的单步规划动作


当前动作可继续使用：

(ds, dθ)


或者内部重参数化为：

(ds, κ)，再转换 dθ=κ·ds


但无论如何：

World Model 不产生动作；
PlanHead 不产生另一条独立轨迹；
K-anchor 不产生独立尾部；
projector 不成为第二规划器； -六步计划全部由同一个 PolicyHead 被调用六次产生。

六步链是：

a0=π(z0)a_0=\pi(z_0) z^1=WM(z0,a0),a1=π(z^1)\hat z_1=WM(z_0,a_0), \quad a_1=\pi(\hat z_1)

一直到：

z^5=WM(z^4,a4),a5=π(z^5)\hat z_5=WM(\hat z_4,a_4), \quad a_5=\pi(\hat z_5)
2. World Model 只负责预测世界状态

World Model 的职责：

z^k+1=fθ(zk,ak)\hat z_{k+1}=f_\theta(z_k,a_k)

它负责：

ego future state； -周车 future state； -lane-relative state； -presence/entry； -上下文随动作演化后的状态。

它不负责判断 action 好不好。

PolicyHead 才负责：

π(z)→a\pi(z)\rightarrow a

但二者必须通过接口验收：

π(z^kWM)≈π(zkreal)\pi(\hat z_k^{WM}) \approx \pi(z_k^{real})

这不是让 WM 为 Policy 的准确性负责，而是检验预测状态是否保留了 Policy 决策所需的信息。

3. Stage C 使用 ExactTracker 训练单步 Policy

你对 Stage C 的定义是合理的：

真实环境状态
→ PolicyHead输出单步action
→ ExactTracker精确执行
→ 环境真实转移
→ reward
→ PPO更新PolicyHead


这样能够：

-隔离规划与控制； -不让 Policy 为 LQR 跟踪误差负责； -建立清晰的 action-to-reward 因果链； -训练同一个单步 PolicyHead。

部署时再通过 World Model 把同一个 PolicyHead 展开六次：

Policy–WM–Policy–WM × 6
→ 六步规划
→ 固定LQR


因此 Stage C 的 PPO 不需要把 action 扩成 12 维。RL 直接训练一步 action，六步能力来自参数共享和准确的 WM rollout。

4. LQR 完全冻结

本轮不再修改：

LQR gain
lookahead
Riccati参数
steer gain
speed gain
controller frequency
curvature feedforward


规划端必须适配固定 LQR 的输入契约。

最终应分别报告：

ExactTracker：
Planner自身能力上界

Fixed LQR：
可部署闭环性能


不能用 ExactTracker 指标直接宣称系统超过 IDM+Pure Pursuit。

二、当前 Repo 已确认的代码事实

这些是初步排摸后的事实，不是最终推断。

1. Stage A 当前是专家动作 teacher forcing

config/train.yaml 中 Stage A 的实际配置为：

name = world_model_teacher_forcing
ego_condition = expert_gt_actions
targets = future_od_ld_lookup
loss = direct_multi_step
rollout_steps = 6


真实流程是：

t0 observation
+ 专家六步 action chain
→ 逐步推进 latent
→ 与真实 future observation latent 对齐


这符合你的 Stage A 原始意图。

2. Stage A 当前每步会 detach latent

当前实现中：

z_ego, z_od, z_ld = (
    z_ego_next.detach(),
    z_od_next.detach(),
    z_ld_next.detach(),
)


因此当前 Stage A 虽然连续预测六步，但训练图不是完整六步 BPTT。

实际更接近：

一步transition训练
→ 把预测latent作为下一步输入
→ 但切断后续误差对前一步transition的梯度


后果：

h5/h6 错误不能修正 h1/h2 transition； -一步准确不代表六步稳定； -误差可能在 free rollout 中累积； -配置中的 direct_multi_step 不能自动等价为端到端六步优化。

这必须先形成实证，而不是立即删 detach。

3. 当前 Action 对 WM 的条件化主要是间接的

当前链路是：

(ds,dθ)
→ arc_step
→ 累积ego pose
→ pose进入ST-GNN图关系
→ latent transition


ST-GNN transition 没有显式接收 action token。

因此需要验证：

这种pose-mediated action conditioning是否足够？


不能直接假定一定不足，也不能直接假定配置里的 conditioned_on: ego 已完整实现：

WM(z,a)WM(z,a)

自动驾驶 World Model 尤其需要验证 action fidelity，仅有视觉、latent 或平均预测指标不足以证明模型遵循动作条件。ACT-Bench 也专门将 action controllability 与一般生成质量分开评估。

4. Stage A 不训练 PolicyHead

Stage A 当前可训练模块包括：

encoders
mem_encoder
plan_head
st_gnn


不包括：

policy
value


Stage A 的 PlanHead 主要通过 ego_next 辅助目标获得梯度。

这符合“WM 与 Policy 分阶段训练”的思想，但必须核查：

Stage A 预测 latent 的坐标系和统计尺度；
Stage B Policy 消费该 latent 时是否分布一致；
Stage B 是否在 predicted latent 上获得过监督。
5. 当前 PlanHead 实际不是动作输出头

当前代码职责大致是：

PlanHead：
ego/OD/LD/semantic/nav/signal融合
+ Primary/MoE
→ policy context latent
→ ego_next辅助预测

PolicyHead：
context tokens
→ action_mu / action_logstd

ValueHead：
context tokens
→ state value


因此 PlanHead 更准确的概念名称应是：

PolicyBackbone
PolicyContextEncoder
SceneDecisionBackbone


先只修文档，不立刻改 class/checkpoint key。

正式契约锁定：

PolicyHead 是唯一动作输出器。

K-anchor 继续保持关闭。

6. Stage B 虽然传 world_model=False，仍可能运行 latent rollout

当前 model(..., rollout=True, world_model=False) 的含义不是关闭 WM transition，而是：

-仍运行六步 latent rollout； -仍生成六步 action plan； -只是不返回 OD/LD/latent 预测诊断输出。

所以 Stage B 的未来五步仍依赖 ST-GNN transition。

目前必须继续核查：

Stage B 中 ST-GNN 是否冻结；
step 1–5 的 action loss 能否回到 ST-GNN；
action chain label 属于哪个真实状态； -真实 latent 与 predicted latent 的混用方式； -是否存在新的 state-label mismatch 风险。
7. Stage C 文档显示走 cheap path

pipeline\README.md 说明 PPO collect/update 使用 cheap path。

当前 cheap path 通常是：

rollout=False
world_model=False


这很可能意味着 Stage C：

真实状态
→ PolicyHead当前action
→ ExactTracker
→ PPO


这本身符合你的设计。

但必须核实：

-环境实际执行的是 action0 还是 plan； -PPO buffer 保存的是哪个 action； -Stage C update 是否只更新 Policy； -MoE、WM、Encoder、Value 的实际梯度； -Stage C checkpoint 部署时，WM rollout+LQR 是否与训练时动作口径一致。

三、新一轮不能先改网络，先做 Stage A/B/C Reality Audit

Coding Agent 第一任务不是训练，而是提交两份真实契约：

docs/stage_abc_reality_audit.md
docs/stage_abc_execution_contract.md


所有结论必须来源于：

实际CLI入口
实际forward调用
运行时hook
gradient/update ledger


不能用 README 或配置注释代替运行事实。

四、审计工作包 1：Stage A/B/C 真正的数据流
Stage A 要回答
1. 输入的6步动作来自哪里？
2. 每一步future target来自哪里？
3. future slot如何匹配？
4. ego/OD/LD分别监督什么？
5. latent target如何生成？
6. 哪一步detach？
7. 哪些loss能更新哪些模块？
8. action噪声如何注入？
9. teacher-forced与free rollout是否同口径？
10. validation是否只测teacher-forced？


输出：

runs/reality_audit/stage_a_execution.json
runs/reality_audit/stage_a_gradient_ledger.csv

Stage B 要回答
1. a0...a5是否全部来自同一个PolicyHead？
2. predicted latent从哪一步开始使用？
3. 每一步action label属于哪个物理状态？
4. 六步action-chain loss如何定义？
5. trajectory loss监督位置还是action？
6. ST-GNN是否冻结？
7. Stage B loss能否更新WM？
8. Primary和specific experts何时开关？
9. DAgger数据如何与普通BC数据并入？
10. predicted-state训练是否存在标签错配？


重点是：

z^kWM\hat z_k^{WM}

对应的监督 action 必须属于其代表的同一个物理状态。

不能再次发生：

预测状态 / 扰动状态
×
另一个时间或恢复后的expert label

Stage C 要回答
1. PPO sampled action是什么？
2. ExactTracker实际执行什么？
3. action/log_prob/value存入buffer的是什么？
4. Stage C更新哪些参数？
5. WM是否完全冻结？
6. Policy更新后部署tail如何变化？
7. collect/update/eval是否同一Policy口径？
8. MoE在Stage C是on还是off？
9. PPO entropy/clip/GAE的实际配置？
10. Stage C checkpoint如何进入WM rollout+LQR部署评测？


输出：

runs/reality_audit/stage_c_effective_action.jsonl
runs/reality_audit/stage_c_gradient_ledger.csv
runs/reality_audit/stage_c_deployment_gap.json

五、审计工作包 2：参数与梯度账本

对每个 Stage 自动生成：

parameter_name
module
requires_grad
optimizer_group
learning_rate
weight_decay
grad_norm
update_norm
checkpoint_loaded
checkpoint_changed


必须能直接回答：

Stage A：
Policy是否真的0更新？

Stage B：
WM是否真的冻结？
step5 action loss能否触达step1 transition？

Stage C：
只更新Policy/Value还是也更新Backbone/MoE？
WM是否有漂移？

必须增加的梯度探针

分别对：

h1 loss
h3 loss
h6 loss


单独 backward，记录到：

encoder
mem_encoder
plan_head / policy backbone
st_gnn transition
policy head
value head
primary
specific experts
router


这会证明 detach 后的真实梯度可达范围。

六、审计工作包 3：World Model 能力四分法

Stage A 不能只看一个总 loss。

A. Teacher-forced one-step

真实状态 + 专家 action：

z_t, a_t → z_hat_(t+1)


报告：

ego pose/speed/heading/yaw
OD position/velocity
lane-relative state
presence/entry
latent error

B. Teacher-action free latent rollout

动作仍使用真实 expert chain，但状态来自前一步预测：

z_hat_(k+1)=WM(z_hat_k,a_expert_k)


这隔离：

WM 多步累计误差。

C. Policy-action free rollout

动作也来自 Policy：

a_hat_k=Policy(z_hat_k)
z_hat_(k+1)=WM(z_hat_k,a_hat_k)


这是部署链的真实近似。

D. Oracle-state Policy

每步使用真实未来 observation 编码出的 latent：

a_oracle_k=Policy(Encoder(obs_real_k))


这隔离：

WM问题
vs
Policy问题

七、审计工作包 4：核心 Rollout Consistency Probe

这是连接 Stage C 单步 RL 与部署六步计划的核心。

真实环境链

Stage C使用ExactTracker运行，记录：

z_t, a_t
z_(t+1), a_(t+1)
...
z_(t+5), a_(t+5)


对应：

at+kreal=π(zt+kreal)a_{t+k}^{real}=\pi(z_{t+k}^{real})
WM链

从相同 z_t 开始：

a^0=π(zt)\hat a_0=\pi(z_t) z^1=WM(zt,a^0)\hat z_1=WM(z_t,\hat a_0) a^1=π(z^1)\hat a_1=\pi(\hat z_1)

直到第六步。

每个 horizon 报告三类误差
状态预测误差
Ez(k)=d(z^k,zkreal)E_z(k)=d(\hat z_k,z_k^{real})
Policy 接口误差
Eπ(k)=∥π(z^k)−π(zkreal)∥E_\pi(k)= \|\pi(\hat z_k)-\pi(z_k^{real})\|
动作链误差
Ea(k)=∥a^k−akreal∥E_a(k)= \|\hat a_k-a_k^{real}\|

输出：

runs/reality_audit/rollout_consistency_h1_h6.json


分层：

普通直道
弯道
tollgate
merge
roundabout
成功
off-road
collision
timeout

八、审计工作包 5：Action Conditioning Fidelity

同一真实状态输入动作反事实：

ds × {0.8, 1.0, 1.2}
dθ ± perturbation
κ ± perturbation
zero action
左右镜像 action


对 horizon 1–6 观测：

future ego pose
future heading
future speed
OD relative state
lane-relative state
next policy action
latent distance


通过条件：

ds增加 → future progress单调增大
左转 → yaw/lateral响应方向正确
镜像action → future状态近似镜像
不同动作 → future latent可区分
action影响不能只停留在pose字段


如果 pose-conditioned WM 已通过，就不加 action token。

如果失败，再进入最小改造：

action embedding
=
MLP(normalized ds, dθ, κ, speed intent)


并将 action embedding 接到：

ego transition
OD transition
可选graph message conditioning


先不加容量。

九、审计工作包 6：WM/no-WM/oracle 对照

同一 a0，比较未来尾段：

W0：当前WM rollout
W1：frozen-t0 latent + step embedding
W2：oracle真实未来state + Policy
W3：expert action chain
W4：几何可行tail，仅作诊断下界


固定 LQR，不修改控制器。

比较：

六步action error
plan κ / Δκ
重规划一致性
LQR preview step
raw steer
steer saturation
timeout
off-road
success


判定：

W2明显好于W0

World Model 是主要瓶颈。

W0≈W2，但都不好

Policy/Stage B 是主要瓶颈。

W1≥W0

当前 WM transition 在伤害 rollout。

W4明显好于W0

当前 WM/Policy tail 未达到基本规划可行性，但 W4 不能直接替代正式路线 A。

十、Stage A 新训练方案只能在审计后按分支启动
情况 A：一步准，多步漂移

不改架构，优化训练：

A0：1-step teacher forcing
A1：2-step truncated BPTT
A2：3-step truncated BPTT
A3：6-step free rollout validation


不直接六步全 BPTT，避免训练不稳定。

推荐 curriculum：

Epoch 1–3：1-step
Epoch 4–6：2-step
Epoch 7–10：3-step
最终：6-step只作验证


只有 3-step 稳定后才尝试 6-step 梯度。

情况 B：Action fidelity 不足

增加显式 action token：

action_embed = MLP(ds, dθ, κ, ds/dt)


但保留：

-原 pose conditioning； -原 ST-GNN； -原参数规模； -现有 checkpoint 可残差初始化。

结构：

transition_input
=
[z, graph_context, action_embed]

情况 C：WM 物理预测准，但 Policy action-chain 差

不改 WM，修 Stage B：

B0：real-state BC
B1：oracle-future-state BC
B2：WM-predicted-state distillation
B3：真实环境DAgger


关键：

预测latent上的动作监督
必须与其对应的真实future state action对齐

情况 D：WM 与 Policy 都好，但六步 plan 不可跟踪

问题在：

动作参数化
六步chain平滑性
可行性监督
固定LQR输入契约


此时才考虑内部：

(ds,κ)


以及：

κ continuity
speed profile
replanning consistency


外部仍转换成：

(ds,dθ)

十一、Stage B 新训练方案

在 Stage A 过门后执行。

B1：Primary 单步 IL
真实状态
→ PolicyHead
→ expert current action


Primary 先掌握：

lane keeping
route following
普通曲率跟随
普通前车跟随
轻微纠偏


MoE 暂时关闭。

B2：六步 rollout IL
real z0
→ Policy-WM ×6
→ action chain


监督：

h1当前动作高权重
h2–h6逐步衰减


建议第一版权重：

1.00, 0.50, 0.25, 0.15, 0.10, 0.05


但不能只看 chain action MAE，还要看：

WM-state vs real-state action consistency
plan κ
plan Δκ
重规划一致性
固定LQR闭环

B3：Mixed latent

逐步扩大 predicted latent 比例：

20% → 50% → 80%


不允许一步切成100% imagined latent。

B4：DAgger

只有满足：

WM rollout consistency过门
primary clean150 >=0.50
teacher-state不显著退化


才启动真实 learner-state DAgger。

B5：MoE

Primary 达到稳定 lanekeep 后再启用。

MoE 只负责：

tollgate
merge/yield
roundabout
复杂交互
避障
特殊恢复


不允许 experts 补偿日常 lanekeep。

十二、Stage C 新训练方案
C0：ExactTracker PPO baseline

训练：

单步 PolicyHead
ValueHead
可选冻结 PolicyBackbone


World Model 冻结。

PPO 使用：

-步级 reward； -Critic； -GAE； -clipped objective； -KL gate； -entropy regularization。

这符合 PPOGRPO.docx 对 PPO 在连续、逐时间步控制和细粒度信用分配上的定位。

C1：每个 PPO checkpoint 的部署迁移 Gate

每次 PPO 更新后必须同时通过：

ExactTracker真实闭环
WM rollout consistency
固定LQR闭环


如果：

Exact提升
但WM+LQR退化


该 checkpoint 不得进入下一轮。

C2：GRPO 暂不实施

PPOGRPO.docx 中的结论仍然成立：

-候选组内比较适合评估“哪条完整轨迹更好”； -但原生 GRPO 对轨迹内部时间信用分配较粗； -自动驾驶仍需要 Critic+GAE 的 step-level advantage。

因此本轮：

PPO主线
GRPO只做未来候选级辅助


等单策略稳定后，才考虑：

候选级Group Advantage
+
时间步Critic/GAE

十三、必须新增的只读 Hooks

所有 hook：

默认关闭
零行为变化
不进入checkpoint
不改变模型输出

World Model Hook

每个 horizon 保存：

action_in
z_ego_in/out
z_od_in/out
z_ld_in/out
ego_pose
nav_token
step_embedding
od_pred
ld_pred
presence
entry

Policy Hook
policy input tokens
PolicyBackbone latent
Primary output
MoE expert outputs
router weights
action_mu/logstd
value
semantic contribution

Gradient Hook
h1/h3/h6 loss到各模块的grad norm
参数update norm
optimizer state

Rollout Consistency Hook
real_state_latent
predicted_state_latent
real_state_action
predicted_state_action
horizon
episode/step
road class
outcome

Effective Action Hook

Stage C 保存：

sampled_action
executed_exact_action
old_log_prob
value
reward
next_real_state
deployment_rollout_plan[6]
fixed_LQR_reference[30]


此 hook 用于分析，不要求 Stage C 实际执行 LQR。

十四、阶段性验收门
Stage A-R Gate
action intervention方向正确率 >=95%
左右镜像一致率 >=95%
horizon误差不指数爆炸
WM tail优于frozen-t0 tail
困难路型无系统性坍塌


绝对门建议作为初始开发门：

ego position P90：
h1 <=0.20m
h3 <=0.60m
h6 <=1.20m

heading P90：
h1 <=2°
h3 <=5°
h6 <=10°


这些门需要结合 simulator 尺度复核，可在审计后调整一次，之后锁定。

Stage B-R Gate
real-state current-action误差过门
WM-state action误差接近real-state
六步action chain无指数漂移
plan Δκ <= expert P95×1.25
clean150 >=0.50
off-road显著下降

Stage C-R Gate
ExactTracker PPO显著正向
rollout consistency不退化
固定LQR clean150同步正向
teacher-state能力不显著退化


达到后才跑 eval500。

十五、本轮明确禁止事项

审计结束前不得：

修改LQR
新增tail生成器
一次输出6个action
启用K-anchor
增加World Model容量
直接删除detach
直接开启6步BPTT
启动DAgger
启动MoE重训
启动PPO/GRPO
扩大数据
新增大量loss

十六、给 Coding Agent 的正式开工指令
启动“Stage A/B/C Repo Reality Audit”，暂停新主模型训练。

目标：
确认当前repo是否真实实现：
PolicyHead单步action
→ World Model预测下一状态
→ 同一PolicyHead继续决策
→ rollout六步
→ 固定LQR部署；
以及Stage C ExactTracker PPO是否能通过该接口迁移到部署六步计划。

第一批交付：

1. docs/stage_abc_reality_audit.md
2. docs/stage_abc_execution_contract.md
3. Stage A/B/C真实调用图
4. Stage A/B/C参数、optimizer、gradient ledger
5. h1/h3/h6梯度可达性报告
6. teacher/free/policy/oracle/frozen-t0 rollout报告
7. action-conditioning intervention probe
8. real-state vs WM-state Policy action consistency
9. Stage C sampled-action / executed-action / deployment-plan契约
10. 当前代码与原始Stage A/B/C设计逐项差异
11. 唯一推荐的下一步和Go/No-Go依据

必须核实：

- PolicyHead是否唯一动作输出器；
- PlanHead实际职责；
- K-anchor是否全程关闭；
- Stage A每步detach的真实影响；
- WM是否对action具备足够敏感性；
- Stage B predicted latent对应的动作标签是否同状态；
- Stage C cheap path与部署rollout的迁移关系；
- PPO checkpoint更新后WM rollout consistency是否退化。

纪律：

- 所有probe/hook默认关闭、零行为变化；
- 不修改LQR；
- 不新增网络容量；
- 不训练新主模型；
- 不启动DAgger/MoE/RL；
- 不扩大数据；
- 结论以运行时hook和actual gradient为准，不以README注释代替。

最终判断

新一大轮优化的目标不是再寻找一个局部技巧，而是重新闭合整条因果链：

Stage A：
World Model真正预测准确、action-sensitive、可滚动

Stage B：
唯一PolicyHead在真实/预测/learner状态上都能做正确单步决策

Stage C：
ExactTracker真实环境PPO提高单步Policy

部署：
同一Policy经准确WM展开六步，
固定LQR闭环性能同步提升


等 Repo Reality Audit 完成后，再根据唯一根因选择：

显式action conditioning
多步BPTT
mixed latent IL
DAgger
MoE
PPO
或之后的候选级GRPO


这能最大限度避免再次出现“方案理论正确，但代码实际上没有按该方案运行”的方向性偏差。