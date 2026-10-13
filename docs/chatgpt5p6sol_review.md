审阅结论

我接受 deepseekv4p1_argue.md §5.36 的五项修正，并批准启动 Round-32 只读审计包。

但我建议把“10+1 个 probes”收敛成三层决策树，而不是平铺执行。原因是很多 probe 之间存在依赖关系。如果第一层已经证明动作根本没有有效进入 WM，后续动态图、mask turnover 和联合梯度审计就不应抢占 GPU 时间。

总体裁决：

批准：
Round-32 只读诊断
部署状态归因
有限差分 Action Fidelity
状态表示充分性
Policy 接口放大
动态图 Oracle
梯度冲突与联合训练可行性

不批准：
立即联合训练
显式 action token
EMA encoder
ego physical head
predicted pose 回灌
presence mask 回灌
BPTT
DAgger / MoE / PPO / GRPO

一、对 Coding Agent 五项修正的裁决
1. Q2 首轮已经回答，接受

R31 已经完成：

Q2-a：真实态与预测态 Policy 接口差
Q2-b：动作、轨迹等损失的梯度冲突
Q2-c：Oracle latent 反事实


现有结果支持：

Stage B 不是类似 recovery_v1 的数据错位 bug；
L_action 与 L_traj 未出现决定性负梯度冲突； -风险来自 imagined state 漂移和尾五步直接监督不足； -应定性为“中等风险的近似监督”，而不是“非法监督”。

所以 Round-32 不再重复泛化 Q2，而应深化：

尾段动作误差分别由状态预测误差、Policy 敏感性和轨迹监督近似造成多少？

2. 必须补“部署状态归因”，接受并升为最高优先级

当前最重要的未决问题是：

Teacher-forced WM误差很小
部署链误差大8–10倍


但“部署链”同时包含：

Policy action 分布变化；
LQR 实际执行残差；
每 0.5 秒重新规划，旧计划尾部被替换；
WM 按旧计划 rollout，而真实车没有逐位执行整条旧计划；
OD 使用恒速图几何； 6.当前动作定义与真实执行位姿之间的差异。

因此，现在不能直接把这 8–10 倍全部叫作“WM learner-distribution failure”。

推荐的归因分解

对同一真实 episode、同一起点 tt，计算四条链。

D0：当前部署 imagined chain
Policy action
→ 规划运动学pose
→ ST-GNN
→ next action

D1：Action-faithful pose replay

保持 WM latent 转移逻辑不变，但 ego pose 使用环境实际执行后的：

env x/y/theta


OD、LD、mask 暂时保持当前实现。

D2：Oracle ego + oracle OD pose

图几何使用真实未来 ego 和 OD pose，latent transition 仍使用当前 WM。

D3：Oracle future latent

直接编码真实未来 observation，作为理论接口上界。

分解：

D0 → D1：
计划pose与实际执行pose差异

D1 → D2：
动态图几何近似误差

D2 → D3：
latent transition自身误差

D3 → expert：
Policy真实状态误差


这是 Round-32 最关键的实验。

3. Action Jacobian 必须使用有限差分，接受

由于当前存在：

pose.detach()


和：

z_next.detach()


所以解析梯度：

∂zk∂a0\frac{\partial z_k}{\partial a_0}

沿跨步链路会被明确截断。解析 Jacobian 为零并不能证明模型“物理上不响应 action”，只能证明训练图禁止该梯度传播。

因此，应同时报告：

解析梯度

证明当前梯度契约：

autograd Jacobian = 0

数值有限差分

证明前向函数性状：

JaFD(k)=zk(a+ϵ)−zk(a−ϵ)2ϵJ^{FD}_{a}(k) = \frac{z_k(a+\epsilon)-z_k(a-\epsilon)} {2\epsilon}

推荐扰动：

ds：
±0.10m、±0.25m、±10%

dθ：
±0.01、±0.03、±0.05rad

κ：
在保持ds不变时构造±扰动


必须先确认扰动未撞 action bound，否则有限差分会被 clip 污染。

4. MoE-off 基线 0.56 应作为新基线，接受

新增结果：

K1e2，MoE-off：

LQR + WM plan：
success 0.560
off-road 0.140
collision 0.080

Exact：
success 0.560

LQR + repeat：
success 0.060


这说明：

S1≈S2 继续成立；
S3 崩溃继续成立；
MoE-on 没有制造主结论；
MoE-on 对当前候选有轻微负贡献； 5.后续接口审计应默认 MoE-off； 6.未经专项训练的 experts/router 不能进入正式基线。

应锁定：

Round-32开发基线：
K1e2 + MoE-off + K-anchor-off
LQR + WM plan = 0.56


但需要注意，50 episode 上的 +4pp 只有 2 个翻正 episode，不能把 MoE-off 的绝对提升解释成稳定收益。这里能稳健确认的是“主结论不依赖 MoE 口径”，不是“关闭 MoE 必然提升 4pp”。

5. Mirror 95% 暂不作为硬门，接受

当前 whole-scene mirror 约 0.842，不能马上判定 WM 架构失败，因为：

-网络并非显式 SE(2)/reflection-equivariant； -lane slot 排序和地图 topology 未必严格镜像； -route、signal、OD identity 的镜像契约可能仍有细节偏差； -部分基准动作接近零，比例误差不稳定。

第一轮应分指标报告：

动作方向一致性
动作幅值残差
latent镜像距离
物理预测镜像距离
slot permutation后的一致性


仅对以下样本统计方向准确率：

|base dθ| > deadband
|lateral error| > deadband
有效左右对应车道存在
route command可镜像


不先锁 95% 总门。

二、对五个 WM 结构隐患的再判断

Coding Agent 核验五项均成立，但“代码存在”不等于“已证明是主瓶颈”。必须按证据排序。

W1：动作只通过 pose 间接进入

高优先级嫌疑。

原因：

R31 已测得动作引起的 latent 改变量远小于自然状态变化； -特别是 ds 通道非常弱； -部署尾段的主要异常又正是 ds overshoot。

但先做有限差分和 Action Swap，再决定是否加显式 action embedding。

W2：缺少直接由 z_ego_next 解码的 ego physical head

高优先级嫌疑。

当前 ego_next_pred 来自 PolicyBackbone/PlanHead，而不是：

z_ego_next → ego physical state


这意味着：

ego_next辅助头预测准确


不能证明 WM 的 z_ego_next 自身包含未来：

progress； -speed； -heading； -lane-relative error。

应先做 probe：

real future z_ego
predicted z_ego
frozen z_ego


分别解码这些物理量。

如果真实 future latent 可解码，而 predicted latent 不可解码，才批准增加 ego physical head。

W3：OD 图位置恒速外推，预测结果不回灌

中高优先级，但必须 Oracle 验证。

这可能明显影响：

merge； -cutin； -roundabout； -intersection。

但普通 lane keeping 上未必是主因。

先做：

CV graph
Oracle future graph
Predicted graph


如果 Oracle graph 对接口误差和尾段动作有显著改善，再考虑将 predicted OD pose 回灌。

W4：mask 固定 t0，presence/entry不参与未来状态

中低优先级。

三秒内对象进出频率如果很低，固定 mask 可暂时接受。先统计真实 turnover：

h1...h6 disappear rate
entry rate
identity switch rate


没有明显 turnover 就不要改。

W5：latent target encoder 同步移动

真实风险，但当前优先级低于 W1/W2/W3。

detach() 不能阻止 encoder 在其他路径中更新，因此 future latent target 是移动目标。

但是否已经造成：

collapse； -共适应； -下游物理信息丢失；

尚未有证据。

先测：

effective rank
target drift
fixed anchor reconstruction
physical linear probe


只有成立后才做 EMA target encoder canary。

三、Round-32 不应平铺执行 11 个 probe

建议三级执行，前一级决定后一级。

Level 1：决定主根因

必须先完成。

P1：部署链四段归因
D0 当前planned-pose chain
D1 actual ego pose replay
D2 actual ego+OD geometry
D3 oracle future latent


输出：

E_state
E_interface
E_action
ds overshoot
dtheta error
kappa error


按 horizon 和路型分层。

P2：Action Fidelity

包括：

有限差分 Jacobian
Action Swap
zero action
ds/dtheta/kappa intervention

P3：Ego State Sufficiency

解码：

progress
speed
heading
lateral error
heading error

P4：Policy Latent Lipschitz

比较真实 latent 与 WM latent 邻域。

Level 1 决策
若动作忠实度差
GO_ACTION_CONDITIONING

若 real latent 本身不包含必要物理量
GO_STATE_REPRESENTATION_FIX

若 WM latent误差不大但Policy放大严重
GO_POLICY_INTERFACE_ROBUSTNESS

若实际pose/动态图解释大部分误差
GO_ROLLOUT_PROTOCOL_FIX


Coding Agent 当前给出的三种决策选项里缺少后两项，必须补上。

Level 2：决定 WM 内部结构

只有 Level 1 指向 WM 后执行。

P5：Latent Rank/Collapse
P6：Prior vs Learned Residual
P7：Graph Oracle
P8：Mask Turnover
P9：Encoder Target Drift

这些决定：

ego physical head
predicted geometry回灌
presence soft-mask
EMA encoder


各自是否有资格进入 canary。

Level 3：决定联合训练

仅当状态、动作和图结构都通过后执行。

P10：Gradient Conflict

对以下损失分别 backward：

WM latent
WM physical
ego physical
real-state action
interface consistency
trajectory

P11：Joint Drift Hook

固定 anchor set，记录：

encoder drift
WM prediction drift
Policy real-state drift
Policy WM-state drift
physical decoder drift

四、关于 WM 与 Policy 是否联合训练
明确裁决

不批准：

WM + Encoder + PolicyBackbone + PolicyHead全部解冻
→ 总loss一次backward


批准未来采用以下顺序。

Phase W：WM 适配

冻结：

PolicyBackbone
Primary
PolicyHead
ValueHead
MoE


训练：

Encoder/WM，具体范围由probe决定


但若 target drift 严重，应：

-冻结主 Encoder，仅训 transition； -或使用 EMA target encoder； -不能让 online encoder 和 target 表示任意共漂。

Phase P：Policy 接口训练

冻结：

Encoder
WM


训练：

PolicyBackbone
Primary
PolicyHead


监督：

真实状态 → expert action
WM状态 → stopgrad(真实未来状态下的Policy动作)

Phase WP：交替训练

一个 cycle 中：

W-step：
冻结Policy，更新WM

P-step：
冻结WM，更新Policy


两边使用更新后的分布，但梯度不跨模块。

这是第一种正式“联合训练”，也是当前最合理方案。

Phase E2E：有限跨模块反传

仅当：

WM物理预测已过门
Policy real-state已过门
E_interface仍显著过大


才允许：

interface loss
→ ST-GNN最后一层


而且：

Encoder 继续冻结； -WM physical loss 权重必须主导； -只开放 transition 最后一层； -低 LR； -必须有 anchor drift hook； -出现物理退化立即回滚。
五、Round-32 验收准则
Gate A：工具可信
所有hook默认关闭时前向逐位一致
不改变RNG序列
不写checkpoint
相同输入重复运行逐位一致
finite-difference结果对epsilon稳定
horizon/time index无错位

Gate B：Action Fidelity

首轮不要先设绝对 Jacobian 数值，采用相对判定：

matched action预测误差
<
random/swap/opposite action预测误差

ds perturbation与progress单调
dtheta perturbation与heading方向一致
action差异在h1–h5不完全消失

Gate C：State Sufficiency

推荐门：

Real latent：
progress/speed/heading probe R² >=0.80
lateral/heading error R² >=0.90

WM latent：
相对Real latent R²保留率 >=80%


若 real latent 自己不过，直接进入 state representation fix。

Gate D：Policy amplification

定义：

Ak=∥π(z^k)−π(zkreal)∥∥z^k−zkreal∥+ϵA_k = \frac{\|\pi(\hat z_k)-\pi(z_k^{real})\|} {\|\hat z_k-z_k^{real}\|+\epsilon}

要求输出 P50/P90/P95。

若 WM state distance 仅温和增长，但 AkA_k 急剧增长，主因在 Policy interface，而不是继续堆 WM。

Gate E：部署归因

分别统计 D0→D1→D2→D3 的误差下降比例。

某一段解释 >=40%总gap：
判定该段为主瓶颈

20%–40%：
共同瓶颈

<20%：
次要因素

Gate F：新结构许可

任何结构修改必须满足：

对应oracle/probe解释 >=30%接口误差
且
不是仅改善latent loss
且
有明确零初始化/逐位兼容方案

六、对最新建议的反驳

Coding Agent 提议 Round-32 完成后只允许：

GO_ACTION_CONDITIONING
GO_STATE_REPRESENTATION_FIX
GO_ALTERNATING_WM_POLICY_TRAINING
BLOCKED_INSTRUMENTATION


这个集合不完整，必须增加：

GO_POLICY_INTERFACE_ROBUSTNESS
GO_ROLLOUT_PROTOCOL_FIX
GO_DYNAMIC_GRAPH_FIX


因为当前 8–10 倍差距未必主要来自 WM transition：

-可能是 Policy 对 latent 小误差过敏； -可能是计划 pose 与实际执行 pose 不同； -可能是 0.5 秒重规划替换三秒计划； -可能是 OD 图几何恒速近似； -可能是多个因素共同作用。

在归因完成前，把决策限定为“改 WM 或联合训练”仍然过早。

七、给 Coding Agent 的正式推进指令
批准Round-32只读审计，但调整为三级决策流程。

统一口径：
- K1e2
- MoE-off
- K-anchor-off
- rollout_semantic=既有最佳口径
- LQR参数不变
- baseline success=0.56，仅作开发参考

Level 1必须先完成：
1. D0/D1/D2/D3部署链四段归因；
2. Action有限差分Jacobian；
3. Action Swap/zero/opposite干预；
4. Ego状态充分性probe；
5. Policy latent Lipschitz。

只有Level 1指向WM内部后，执行Level 2：
6. latent rank/collapse；
7. prior-vs-residual；
8. CV/oracle/predicted graph；
9. mask turnover；
10. target encoder drift。

只有Level 1/2证明需要联合适配后，执行Level 3：
11. loss梯度冲突；
12. joint drift anchor hook。

所有hook必须：
- 默认关闭；
- max_abs_diff=0；
- RNG不变；
- 不写checkpoint；
- 不改变训练/eval行为。

本轮禁止：
- 训练；
- action embedding；
- ego physical head；
- predicted pose回灌；
- EMA encoder；
- BPTT；
- 联合解冻；
- DAgger/MoE/PPO/GRPO；
- LQR修改。

decision.md允许：
GO_ACTION_CONDITIONING
GO_STATE_REPRESENTATION_FIX
GO_POLICY_INTERFACE_ROBUSTNESS
GO_ROLLOUT_PROTOCOL_FIX
GO_DYNAMIC_GRAPH_FIX
GO_ALTERNATING_WM_POLICY_TRAINING
BLOCKED_INSTRUMENTATION

不得自行进入下一阶段训练。

最终判断

Coding Agent 对我上一轮建议的五项修正都成立，尤其是：

Q2 已完成首轮审计； -解析 Action Jacobian 会被 detach 人为置零，必须以有限差分为主； -MoE-off 0.56 应成为一致口径； -mirror 暂时只能作为分解诊断； -必须增加部署状态分布归因。

但当前依然不能直接进入“WM/Policy交替训练”。

Round-32 的真正目标是把：

部署链8–10倍误差放大


精确分配给：

动作条件化
状态表示
动态图近似
执行pose差异
重规划协议
Policy敏感性


只有完成这一步，下一轮网络改造才有资格做到“一次只修一个主根因”，而不是再次同时改 WM、Policy、loss 和数据，最后无法判断成功或失败来自哪里。