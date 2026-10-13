你先休息。这一轮不再追逐单点指标，而是把完整范式重新排成一条有严格前置门的流水线：

Stage A：World Model
→ Stage B0：Primary 单步 IL
→ Stage B1：WM rollout IL
→ Stage B2：DAgger
→ Stage B3：MoE 专项训练
→ Stage C0：ExactTracker PPO
→ Stage C1：部署同构 RL 验证
→ 最终：WM rollout 6步 + 固定 LQR 闭环评测


总范式保持不变：

IL + RL + MoE + World Model + 6-step rollout


但执行顺序、冻结关系、训练执行器与验收门重新锁定。今晚让 Coding Agent 先连续完成“审计修复、数据 pilot、World Model 分布适配和关键 probes”，不得越门启动后续高风险训练。

一、正式锁定目标架构
1. 唯一动作生成链
真实 observation
      ↓
Encoder + Memory
      ↓
真实状态 latent z0
      ↓
PolicyBackbone
      ↓
Primary + MoE
      ↓
PolicyHead
      ↓
单步 action a0
      ↓
World Model(z0, a0)
      ↓
预测状态 z1
      ↓
同一个 PolicyHead
      ↓
单步 action a1
      ...
      ↓
a0...a5
      ↓
固定 LQR


职责锁定：

PolicyHead：唯一动作输出器；
World Model：只预测下一状态；
PolicyBackbone：当前代码中的 PlanHead，只负责融合上下文；
MoE：Policy 内部能力，不是第二规划器；
Rollout orchestrator：循环调用 Policy 和 WM；
LQR：部署控制器，固定不训练；
K-anchor：继续关闭。
二、训练与执行器总规则
ExactTracker 的用途

ExactTracker 只在以下场合使用：

1. World Model policy-distribution 数据采集
Policy action
→ ExactTracker 精确执行
→ 得到真实 next observation


目的：得到干净的 (state, action, next_state) 标签，不混入 LQR 跟踪误差。

2. Primary Policy 单步能力训练后的上界评测

ExactTracker 回答：

如果单步规划动作被准确执行，Policy 自身有多强？

3. Stage C 的 Planner-only PPO 候选

ExactTracker 可以作为 Stage C 的 C0 训练执行器，使 PPO 只优化规划动作，不适配控制误差。

固定 LQR 的用途

LQR 只在以下场合使用：

1. 正式部署评测
Policy–WM rollout 6步
→ 固定 LQR
→ 闭环 KPI

2. 每个阶段的部署迁移门

任何在 ExactTracker 上提升的模型，都必须通过：

WM rollout 6步 + 固定 LQR


才能晋级。

3. Stage C 的部署同构 RL 候选

后续可比较：

C0：ExactTracker PPO
C1：冻结WM尾段的部署同构 PPO


但今晚禁止开启。

三、长期阶段路线
Stage A：World Model
A0：配置与训练契约修复
任务

生成并固化 stage_a_config_consumption.md：

CONSUMED
IGNORED
LEGACY
DERIVED

接线 Stage A trainable 白名单：

optimizer 中不得有 dead params； -实际参数集合必须等于配置声明集合； -禁止当前“除 policy/value 外全部放入 optimizer”的排除式逻辑。

所有运行写入：

resolved config；
consumed-key map；
optimizer parameter hash；
checkpoint load coverage；
obs fingerprint；
expert/policy 数据来源；
tracker；
reference mode；
git state。

清理或拒绝无效键： -旧 direct_multi_step 文案； -未消费的 ego_condition/targets/slot_matching； -未消费的 Stage C tracker 值； -未消费的旧 KL 或 freeze 配置。

Gate A0
unknown config key = 0
silent ignored key = 0
dead optimizer parameter = 0
configured trainable set == optimizer set
baseline checkpoint完整加载
baseline前向 max_abs_diff = 0
全测试通过

A1：Policy-conditioned WM 数据 Pilot
固定策略

使用两个 checkpoint 采集：

P-policy：
K1 S1 e1/e2 当前最佳语义/曲率候选

B-policy：
pri512 或 M0 legacy 基线


避免 WM 只适配单一 Policy。

执行器
ExactTracker

每条序列保存
obs_t
policy_action_t
real_obs_t+1
...
real_obs_t+6

encoder(real_obs_t)
route / road class
episode outcome
teacher/current expert action，可选诊断
checkpoint id
policy hash

Pilot 规模

今晚先采：

每个Policy 1,000–1,500 episode
合计至少50,000个有效transition
目标100,000 transition


分布建议：

普通直道/缓弯        35%
急弯                  15%
tollgate              15%
merge / intersection   15%
roundabout             10%
失败前窗口              10%

数据门
state/action/next_state同一步匹配率 = 100%
随机重放 >=2,000 transition
不可复现率 <0.1%
负ds = 0
NaN/Inf = 0
coverage按路型输出
Policy checkpoint和obs fingerprint完整记录

A2：World Model Baseline Reality Probe

在不训练的情况下，对当前 WM 做四种前向。

A2.1 Expert-state + expert-action

回答：

WM 在原 Stage A 分布上还有多准？

A2.2 Policy-state + recorded policy-action

使用 ExactTracker 真实序列：

real z_t
+ recorded a_t
→ WM predicted z_t+1


回答：

只把状态和动作换成 learner 分布，WM 一步预测退化多少？

A2.3 Policy-action free rollout
z_hat_k
→ Policy(z_hat_k)
→ WM(z_hat_k, action_hat_k)


回答部署链累计漂移。

A2.4 Oracle-state Policy
Policy(Encoder(real_obs_k))


回答：

不受 WM 污染时，同一个 Policy 在未来真实状态上是否正确？

强制输出

按 horizon 1–5：

E_state
E_policy_interface
E_deploy_action
ego physical error
OD physical error
lane-relative error
presence / entry


定义：

Einterface(k)=∥π(z^k)−π(zkreal)∥E_{\text{interface}}(k) = \|\pi(\hat z_k)-\pi(z_k^{real})\| Edeploy(k)=∥π(z^k)−akreference∥E_{\text{deploy}}(k) = \|\pi(\hat z_k)-a_k^{reference}\|
A3：WM Action Fidelity Probe

同一真实状态进行：

ds × 0.8 / 1.0 / 1.2
dθ ±0.05
κ ±指定扰动
zero action
whole-scene mirror


逐 horizon 报告：

future ego progress
heading
lane-relative lateral error
speed
future latent distance
next Policy action

Gate A3
ds-progress单调率 >=90%
转向方向正确率 >=95%
whole-scene mirror一致率 >=95%
不同action的latent可辨识
zero action不得与normal action近似同预测
next-policy-action不得退化为近常数

A4：Policy-conditioned WM Adaptation

仅在 A0–A3 完成后执行。

训练数据 exposure
原 expert transition       60%
Policy Exact transition    40%

第一候选 A1-WM

保持：

-当前网络容量； -当前 detach； -无 BPTT； -无显式 action token； -无新大量 loss。

训练：

encoders
mem_encoder
st_gnn
必要的WM物理解码头


冻结：

PolicyHead
ValueHead
Primary
MoE experts
router


PolicyBackbone/PlanHead 是否参与 Stage A，要以 ego_next 对 WM 是否必要为准。建议第一臂冻结，第二臂才允许低 LR 解冻，不同时改。

损失

保留：

latent consistency
OD physical
LD physical
presence / entry


允许新增一个统一的 ego-dynamics loss：

progress
heading
speed
lane-relative lateral / heading


不得再加多套零散 loss。

Gate A4

相对当前 WM：

expert分布退化 <=10%
policy分布one-step error下降 >=30%
E_interface h3–h5下降 >=40%
E_deploy h3–h5下降 >=35%
尾段ds overshoot下降 >=50%
action fidelity通过A3

A5：条件分支
如果 A4 改善不足，且 Action Fidelity 不过

启动：

显式 action residual embedding


输入：

ds
dtheta
kappa
ds/dt


残差初始化：

alpha_action = 0


必须保证：

alpha=0前向逐位等于A4初始模型

如果 Action Fidelity 已过，但 h3–h5 仍累积漂移

只做一次：

2-step truncated BPTT canary


不直接做 3/6-step。

BPTT Go 门
h3–h5状态误差额外下降 >=20%
显存/吞吐可接受
梯度无爆炸
expert保持不退化


不过门即恢复 detach。

Stage B：IL

Stage A 过门后才能启动。

B0：冻结 WM，Primary 单步 IL

训练：

PolicyBackbone
Primary
PolicyHead


冻结：

World Model
specific experts
router
ValueHead


数据：

expert real-state
properly labeled perturbation
少量 learner-state


主目标：

当前状态 → 当前 expert action

Gate B0
teacher-state MAE不恶化
learner-state MAE显著下降
Primary-only能完成基础lanekeep
Exact clean150 >=0.55
LQR+WM-plan clean150 >=0.45

B1：Oracle-state rollout Policy IL

真实未来 observation：

Policy(Encoder(real_obs_k))
→ expert_action_k


按 k=0...5 监督。

权重建议：

1.00, 0.70, 0.50, 0.35, 0.25, 0.20


这里是同状态合法监督。

B2：Predicted-state Policy Consistency

WM predicted latent：

Policy(z_hat_k)


对齐：

stopgrad Policy(Encoder(real_obs_k))


损失：

Linterface=∑kwk∥π(z^k)−sg⁡(π(zkreal))∥L_{\text{interface}} = \sum_k w_k \| \pi(\hat z_k) - \operatorname{sg}(\pi(z_k^{real})) \|

同时保留 oracle-state expert supervision。

冻结：

World Model


避免 Policy consistency 反向污染刚验收的 WM。

Gate B2
E_interface h3–h5再下降 >=30%
oracle-state Policy不退化
plan曲率/进度不恶化
LQR+WM-plan clean150 >=0.50

B3：DAgger

DAgger 放在：

WM适配完成
+
Primary real/predicted-state IL完成


之后。

DAgger 采集

使用：

当前完整Policy
ExactTracker真实执行


保存：

learner real obs
Policy current action
expert current action
next real obs
WM prediction
rollout consistency
outcome


为什么 Exact？

为了给单步 Policy 和 WM 提供干净真实转移。

DAgger 数据 exposure

逐轮：

Round 1：10%
Round 2：20%
Round 3：最多30%


最多三轮。

每轮门
learner-state action MAE下降 >=15%
rollout consistency不退化
teacher-state退化 <=10%
Exact clean150 >=+3pp
LQR+WM-plan clean150 >=+3pp


连续两轮部署提升小于2pp则停止。

B4：MoE

只有 Primary 达标后启动。

Primary 负责
lane keeping
route following
普通曲率
基础前车跟随
普通纠偏

Specific experts 负责
tollgate
merge / yield
roundabout
复杂交叉口
避障
特殊恢复


训练：

冻结WM
冻结Primary第一阶段
只训specific experts + router


随后允许 Primary 极低 LR 联合微调一次。

MoE Gate
full-MoE相对Primary >=+3pp
普通道路不退化
expert residual在普通道路接近0
router与困难路型存在稳定关联
3 seeds成立


不过门保持 Primary-only。

Stage C：RL
C0：先固定 ValueHead

在当前 Policy / WM 上先做 Critic warmup：

Policy冻结
WM冻结
ValueHead训练


价值函数至少要在：

成功
off-road
collision
timeout


前有分辨力。

C1：ExactTracker PPO

训练目标：

Policy单步action能力


执行：

Policy action
→ ExactTracker
→ real environment


冻结：

WM
specific experts第一轮冻结
Encoder/Memory第一轮冻结


训练：

PolicyHead
ValueHead
PolicyBackbone最后层，条件开放
Primary最后层，条件开放


先只训 PolicyHead + Value。

每个 PPO checkpoint 必须做三层评测
ExactTracker闭环
R4 rollout consistency
WM-plan + fixed-LQR闭环


如果 Exact 提升但部署下降，该 checkpoint 不晋级。

C2：部署同构 PPO Canary

只有 C1 不能稳定迁移到部署时才执行。

流程：

随机变量 = a0
a1...a5 = 冻结WM + 冻结tail Policy的确定函数
fixed LQR执行plan


PPO update epoch 内：

WM冻结
tail generator冻结
old/new mapping不漂移


只更新产生 a0 的策略路径。

这是较高风险候选，先做：

10–20 updates
3 seeds

C3：MoE RL

Primary PPO 稳定后，才允许对 specific experts 做场景型 RL。

避免所有 expert 同时被全局 reward 拉动。

C4：GRPO

GRPO 最后考虑，只用于：

同状态下多个候选动作/轨迹的组内比较


不替代 PPO 的 step-level Critic/GAE。

候选可以来自：

Policy stochastic samples
MoE route samples
不同短期行为模态


但要保留：

PPO/critic负责时间信用
GRPO负责同状态候选相对排序

四、今晚 Coding Agent 的长任务队列

今晚只执行至 Stage A，不越门进入 IL/RL。

Task 1：工程契约

交付：

docs/stage_a_config_consumption.md
docs/stage_abc_execution_contract.md
runs/round31/config_contract/


完成：

-配置消费 fail-fast；

Stage A optimizer 白名单； -dead params 清零； -manifest 完整； -历史配置迁移说明； -测试。
Task 2：现有证据复核

对 round30_audit.md 的核心数字做一致性复算：

S1/S2/S3矩阵
R4 h1–h5
WM teacher/free/policy分布
gradient ledger


不需要重跑全部 episode，但要：

随机抽取20%重新运行
数值误差在预设容差内

Task 3：Policy-conditioned transition pilot

使用：

K1e2
+
pri512/M0


ExactTracker 采集。

第一目标：

至少50k transition


资源允许则达到100k。

完成：

-重放； -数据卡； -分布； -路型覆盖； -失败窗口； -指纹； -数据源 hash。

Task 4：A0 完整 probes

执行：

expert/expert
policy state/recorded action
policy free rollout
oracle-state policy
frozen-t0


输出：

E_state
E_interface
E_deploy
h1...h5
按路型和结局分层

Task 5：Action Fidelity

执行：

ds scale
dtheta perturb
kappa perturb
zero action
whole-scene mirror


输出：

ego physical
latent
next policy action
单调率
镜像一致率

Task 6：A1 WM Adaptation Canary

前提：

Tasks 1–5成功
数据门全部通过


训练一个最小 A1：

expert 60%
policy transition 40%
保持detach
不加action token
不BPTT
不扩容量


最多：

3–5 epochs
1 seed


按 validation 和 R4 选 checkpoint，不按训练 loss 单独选择。

五、今晚的自动停止规则

Coding Agent 必须自动停止并写 decision.md，不得自行开下一阶段。

停止条件 1：配置契约破坏基线
baseline max_abs_diff != 0
checkpoint加载不完整
optimizer参数集不一致

停止条件 2：数据不可信
state-action-next_state错位
重放失败 >=0.1%
obs fingerprint混用
路型覆盖严重缺失

停止条件 3：A0 probe 工具不闭合
oracle-state重放与真实记录无法对齐
whole-scene mirror不物理
horizon索引错误

停止条件 4：A1 灾难性退化
expert-val退化 >20%
NaN/Inf
latent collapse
OD/LD明显崩溃

停止条件 5：A1 没有实质改善
E_interface h3–h5下降 <20%
且
E_deploy h3–h5下降 <15%


此时不扩大训练，不加 epoch，提交结果等待裁决。

六、明早应看到的交付物
docs/stage_a_config_consumption.md
docs/stage_abc_execution_contract.md
docs/round31_policy_transition_dataset_card.md
docs/round31_wm_adaptation_report.md

runs/round31/
  contract/
    resolved_config.yaml
    consumed_keys.json
    optimizer_ledger.csv
    checkpoint_load_report.json
    tests.txt

  data/
    manifest.json
    schema.json
    replay_report.json
    coverage_report.json

  probes/
    rollout_consistency.json
    action_fidelity.json
    whole_scene_mirror.json
    horizon_metrics.csv

  a1/
    train_metrics.json
    val_metrics.json
    checkpoint_selection.json
    comparison_a0_a1.json
    decision.md


decision.md 只能给出以下之一：

GO_ACTION_EMBEDDING
GO_2STEP_BPTT_CANARY
GO_STAGE_B_REDESIGN
NO_GO_WM_ADAPTATION
BLOCKED_DATA_OR_TOOLING


不得自行进入 DAgger、MoE 或 RL。

七、明早我们的决策树
A1 显著有效
锁定policy-conditioned WM adaptation
→ 扩到三seed
→ 进入Stage B oracle/predicted-state IL

A1 改善一步但 action fidelity 仍弱
显式action residual embedding

A1 action fidelity过，但horizon继续漂
2-step truncated BPTT canary

A1完全无效，oracle-state Policy好
重审WM transition/latent接口

Oracle-state Policy也差
暂停WM改造
转Stage B Policy监督设计

Exact显著好、部署仍差
继续修rollout consistency
不启动RL

最终给 Coding Agent 的开工指令
启动 Round-31 长任务，仅执行 Stage-A Reality-to-Adaptation 流水。

总范式保持：
IL + RL + MoE + World Model + 6-step rollout。

今晚边界：
只做到配置契约、policy-conditioned数据、WM probes和A1 adaptation canary。
不得进入Stage B/C正式训练。

强制顺序：

1. 修复配置消费和optimizer白名单，baseline逐位不变。
2. 复核Round-30核心证据。
3. 用K1e2和legacy policy，经ExactTracker采集policy-conditioned transition pilot。
4. 完成expert/policy/oracle/frozen rollout consistency。
5. 完成WM action fidelity与whole-scene mirror。
6. 训练A1：
   expert 60% + policy transition 40%，
   保持detach，无action token，无BPTT，无扩容。
7. 用R4而非train loss选择checkpoint。
8. 输出唯一decision，停止等待用户裁决。

A1目标：
expert-val退化<=10%；
E_interface h3–h5下降>=40%；
E_deploy h3–h5下降>=35%；
尾段ds overshoot下降>=50%；
ds单调率>=90%；
whole-scene mirror>=95%。

若改善不足：
不自动加epoch；
不自动启action embedding；
不自动删detach；
不自动启BPTT；
等待下一轮裁决。


这样安排后，Coding Agent 今晚有足够长、足够具体的工作队列，同时不会越过证据门擅自启动下一阶段。明早我们只需要根据 decision.md 和 A0/A1 对照，决定下一步是显式 action conditioning、短 BPTT，还是转向 Stage B 的真实态/预测态接口训练。