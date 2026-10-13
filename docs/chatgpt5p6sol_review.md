总体裁决

最新结果把问题进一步收敛了：

车道语义输入、target-lane 选择和表示接线已经基本成立。
闭环失败不再主要归因于“网络看不到车道”。
当前瓶颈转变为：

语义信息已经存在于 latent，但训练过程中没有被稳定地转化为动作。全量 BC 又重新选择了 history/dynamics shortcut，并且替换式 semantic 接线破坏了旧策略的初始化基本盘。

因此：

R21-3D 的停止决定正确。
不批准正式大规模采集、DAgger、MoE 或 RL。
批准一个严格受限的 Round-25：平滑接入 + 动作级语义采用 + 分阶段解冻。
Round-25 若仍失败，应停止当前 semantic-BC 配方，不继续堆 loss 或扩大数据。
一、对最新结果的判断
1. R21-3D-1/2 已经排除了三个主要嫌疑

当前证据支持：

semantic_chain 明显优于 semantic_no_chain；
rollout 使用 static_role_only 优于全量重复注入 t0 动态语义；
target-lane correctness 在 983 个状态上达到 100%； -独立集上方向正确率约 99.6%–99.8%； -曲率响应达到 25%； -history/lane 敏感度比下降到 0.227。

所以目前不能再把失败简单归因于：

target lane 选错；
LD-LD chain； -rollout 重复使用动态 semantic； -输入不充分； -表示不可解码。

车道语义已经成为可用表示。

2. 3D-3 失败定位基本合理

小规模闭环 canary：

S0 success = 0.240
S1 success = 0.247
legacy baseline = 0.327 左右


同时出现：

steer saturation：0.80 → 0.21
lane sign：e1 约0.93 → e3 约0.60
独立集 sign：约0.56–0.60


这说明模型并非完全没学到 semantic，而是逐 epoch 把语义的动作影响削弱了。

最可能的机制是：

semantic 信息进入表示
→ 初期能够影响动作
→ BC 总体 loss 更容易通过原历史/动力学模式下降
→ 优化器逐步绕开 semantic
→ 动作变得更保守、转向变弱
→ steer saturation 降低，但 lanekeep 与闭环 success 反而下降


因此，当前 root cause 是：

Semantic representation 已建立，但 action adoption 没有被训练目标和初始化方式保护。

二、代码中最关键的问题：当前 semantic 是“替换式接入”

model.py 当前 semantic 模式把原来的 LD summary 替换为新构造的 geometry_context，并额外向 policy/value token 集合追加 target_lane_token。新 SemanticWiring 是惰性随机构造，因此即便从旧 checkpoint 加载，semantic 模式一开启，PlanHead 和 Policy 的输入分布就立即改变。

这很好地解释了：

S0 e001 success 仅0.093
随后 e003 恢复到0.240


这不只是“训练预算不足”，而是旧策略接入新语义时遭遇了明显的 feature-distribution shock。

正确修复方向

Semantic 不应替换旧 LD context，而应残差式加入：

geometry_context
    = legacy_ld_context
    + beta_context * semantic_delta


Policy 的 target token 也应受同一个门控约束：

target_token_effective
    = beta_token * target_lane_token


初始化：

beta_context = 0
beta_token = 0


从而保证：

semantic 模式初始 action_mu / value / plan / rollout
与 legacy checkpoint 逐位一致


否则再增加 epoch，只是在等待模型重新找回旧能力，不是高效地学习 semantic。

三、需要修复的一个隐藏 Mask 问题

目前 _head_tokens() 在存在 target_token 时，为它追加恒为 1 的 mask。model.py 中 target token 被直接加入 Policy/Value token 集，而没有传入 lane_semantic_mask。

虽然当前数据集的 target valid 几乎是 100%，正式部署仍可能遇到：

无 route；
lane 投影失败； -地图异常；
reset/warmup； -target-lane fallback 无效。

即使 target token 数值被置零，CrossAttention 的 key/value projection 如果带 bias，零 token 仍可能产生非零影响。

必须修正为
target_token_mask = lane_semantic_mask * target_lane_valid


并将它传递给 _head_tokens()，而不是恒为 1。

验收：

semantic invalid 时：
target token mask = 0
action/value 与无 target token 路径逐位一致

四、对 static_role_only 的判断

当前实现将以下字段视为 dynamic，并在 rollout 后段清零：

lateral offset
heading error
左右 margin
distance_to_lane_end
distance_to_target_lane_end


保留：

near/far curvature
target valid
route confidence
lane-change direction
role/group/route association


这个选择目前有实证支持，可以作为 Round-25 默认口径。

但要注意：near/far curvature 相对车辆前进后也不是严格静态。当前 3 秒 rollout 内保持 t0 curvature 只是近似，并不代表语义上完全正确。

因此：

Round-25 暂不修改； -仍需记录后段计划曲率和实际道路曲率偏差； -不要把 static_role_only 宣称为最终物理模型。
五、Round-25 实施方案

这次只解决一个问题：

如何让新 semantic 以不破坏旧策略的方式进入动作决策，并保持其动作因果影响。

不采新数据，继续使用 semantic_v1_3d3 的 53,765 行数据。

Phase 25A：平滑接入

新增 residual semantic adapter：

legacy_ld_context = 原 masked mean / 原路径
semantic_delta = SemanticWiring(...)

geometry_context =
    legacy_ld_context + beta_context * semantic_delta


Target token：

target_token_effective =
    beta_token * target_lane_token


初始化：

beta_context = 0
beta_token = 0

强制契约

加载旧 checkpoint 后，在任何训练之前：

action_mu max_abs_diff < 1e-6
value max_abs_diff < 1e-6
plan max_abs_diff < 1e-6
traj_xy max_abs_diff < 1e-6
router_logits max_abs_diff < 1e-6


如果做不到，不开始训练。

Phase 25B：分阶段解冻
Stage 1：只训练 semantic adoption

训练：

SemanticWiring；
beta_context；
beta_token；
lane auxiliary heads； -必要的 target-token projection。

冻结：

encoder；
MemBank；
ST-GNN；
PlanHead legacy fusion；
primary；
experts/router；
policy/value；
WM。

目的：

-先让 semantic feature 具有正确表示； -不允许旧动作决策器为了降低 BC loss重新改写全部表示。

建议：

1–2 epochs

Stage 2：动作接口适配

解冻：

PlanHead fusion 最后一层；
Policy trunk/输出前最后一层；
Primary 最后一层，可选，低 LR。

继续冻结：

encoder；
ST-GNN；
WM；
MoE experts/router；
Policy CrossAttention 主体先不全开。

建议学习率：

semantic path：1.0×
PlanHead final：0.1×
Policy final：0.1×
Primary final：0.03–0.05×

Stage 3：仅在 Stage 2 通过后

才允许扩大到：

-完整 PlanHead fusion； -完整 Policy trunk。

不允许一次性全模型解冻。

六、动作级监督应该怎么做

Coding Agent 提议“action 级方向监督”是对的，但不能只加一个不连续的 sign classification。

现有 BC 已经监督动作，但整体数据多数位于 lane center，模型仍可通过 dynamics shortcut 获得低平均 loss。新监督必须强调：

lane perturbation
→ 动作变化方向

1. 当前动作回归

对 controlled perturbation 和 learner-state 行，提高当前动作权重：

L_action =
    SmoothL1(mu_ds, expert_ds)
  + lambda_theta * SmoothL1(mu_dtheta, expert_dtheta)


B/C 数据的权重高于普通 expert-center 数据，但不做 40 倍过采样。

建议 exposure：

normal expert：50%
controlled perturbation：25%
learner-state：20%
complex topology：5%

2. 方向 Margin Loss

定义 expert correction direction：

y = sign(expert_dtheta - baseline_dtheta)


仅在 correction magnitude 超过死区时启用：

|expert_dtheta - baseline_dtheta| > theta_deadband


损失可使用：

L_direction =
    softplus(-y * (mu_dtheta - baseline_dtheta) / temperature)


避免在“专家几乎不需要修正”时强行产生方向标签。

3. Paired Counterfactual Loss

对左右对称扰动对：

state_left
state_right


要求：

mu_dtheta(left) > mu_dtheta(center)
mu_dtheta(right) < mu_dtheta(center)


具体符号按 repo 坐标定义校验，不能凭自然语言假设。

这比单样本 sign loss 更难被 history shortcut 绕过。

4. Semantic adoption loss

监控但不宜过强：

L_adoption =
    max(0, margin - |∂mu_dtheta / ∂lane_lateral_error|)


实际训练不一定需要直接求 Jacobian，可以用左右 paired 输出差近似。

重点是保证：

lane semantic 的变化确实改变动作


而不仅是能从 latent 中线性解码出来。

七、Round-25 只允许三个正式实验

避免再次膨胀。

P0：平滑接入，不加新动作监督
Residual semantic initialization
+ staged unfreeze
+ 原 BC loss


作用：分辨此前失败是否主要来自替换式初始化。

P1：P0 + action direction supervision

加入：

perturbation/learner-state 当前动作加权； -direction margin； -paired counterfactual loss。

这是主要候选。

P2：P1 + Primary final low-LR

只额外解冻 Primary 最后一层。

若 P2 不优于 P1，则 Primary 保持冻结，不再扩大解冻范围。

不做：

-不同 loss weight 大扫描； -不同网络容量； -不同 optimizer 大扫参；

MoE； -RL； -DAgger； -更多数据。
八、checkpoint selection 不能再只看训练 loss

当前已经观察到：

e1 semantic sign ≈0.93
e3 semantic sign ≈0.60


训练 loss 下降但动作语义变差。

因此每个 epoch 必须计算联合评分：

Score =
  clean150 success proxy/canary
  + lane sign score
  + curvature response
  - history shortcut penalty
  - off-road proxy


不应该在同一 clean150 上频繁进行真正闭环选择。建议：

semantic_val 用于 epoch 选择；
calibration50 用于短闭环选择；
clean150 只用于最终开发验证。
必须记录
sign accuracy
mirror consistency
lane response ratio
curvature response ratio
history/lane ratio
teacher MAE
perturbed-state MAE
learner-state MAE
beta_context
beta_token
semantic gradient norm
policy final-layer gradient norm

九、Round-25 验收标准
Gate A：初始化与代码正确性

全部通过：

beta=0 时完整输出逐位等价
invalid semantic token mask 正确
semantic_no_chain 开关隔离
static_role_only 输出确定
checkpoint 完整加载
训练参数/optimizer/update 账本一致

Gate B：动作语义保持

在独立 semantic 集：

sign accuracy >=95%
mirror consistency >=95%
lane response >= legacy 3×
curvature response >=25%
history/lane ratio <=1.0


并要求：

连续两个 epoch 不低于门槛


不能只挑一个瞬时峰值。

Gate C：动作质量

相对 legacy：

perturbed-state dtheta MAE 至少下降20%
learner-state dtheta MAE 至少下降15%
dtheta sign error 至少下降30%相对值
teacher-state MAE 恶化不超过10%

Gate D：clean150

Round-25 的正式候选至少达到：

success >=0.38
且相对 legacy >=+5pp

off-road 相对 legacy下降至少5pp
collision 不增加超过2pp
timeout 不增加超过3pp
steer saturation：
  不得再次接近0.80
  也不得低于合理响应区间导致欠转


建议 steer saturation 不设简单“越低越好”，而看：

off-road episode 中 saturation 是否下降
成功弯道路段是否仍有足够 steering response


3D-3 将 saturation 从 0.80 降至 0.21，却性能变差，说明低 saturation 也可能是欠响应。

Gate E：扩大数据许可

只有 Gate A–D 全过，才允许正式采集 20 万级数据。

否则不采。

十、对 DAgger、MoE、RL 的状态
DAgger

继续冻结。

虽然 covariate shift 已被证明存在，但当前模型对已有 learner-state 和 perturbation 数据仍未稳定采用 semantic。现在做 DAgger只会增加模型继续忽略的数据。

启动门：

Round-25 Gate A–D 全过
clean150 >=0.45
learner-state error仍显著高于teacher-state

MoE

继续关闭。

Primary 尚未证明已经恢复基础 lane keeping。此时启用 experts 会再次让 specific 补偿 Primary 的基本能力。

RL

继续冻结。

当前闭环约 0.24–0.33，远不到合法 RL 启动门。

十一、我对正式目标的现实判断

R21 semantic 路线解决的是一个真实且重要的问题，但它不会自动把 0.33 推到 0.80。

当前合理里程碑：

Round-25：恢复并超过 legacy，目标 clean150 0.38–0.45
正式 semantic BC：eval500 0.50–0.60
semantic + current-action DAgger：0.60–0.70
后续场景型 MoE / PPO：尝试追平并超过0.756


如果 Round-25 连 legacy 都不能稳定超过，应停止继续完善该 semantic adapter，重新评估：

BC 数据和 expert 行为的一致性；
chain/action 监督； -模型执行接口； -IDM+Pure Pursuit hybrid 路线。
十二、最终批准
批准
Round-25：
P0 平滑初始化
P1 动作级 semantic adoption
P2 Primary final low-LR
分阶段解冻
联合 checkpoint selection

暂不批准
正式20万级采集
DAgger
MoE experts
RL
WM修改
K-anchor
容量扩展
geometry loss

给 Coding Agent 的执行指令
批准 Round-25，仅使用现有 semantic_v1_3d3 数据。

先将 semantic 接线改为 residualized adoption：

geometry = legacy_geometry + beta_context * semantic_delta
target_token = beta_token * semantic_target
beta 初始 0，确保旧 checkpoint 完整输出逐位一致。

修复 target token mask：
mask = lane_semantic_mask * target_lane_valid，
不得恒为1。

训练采用三阶段：
1. semantic path only；
2. semantic + PlanHead/Policy final low-LR；
3. 仅在前两阶段达门后，允许 Primary final low-LR。

正式三臂：
P0 平滑接入；
P1 P0 + action direction/paired counterfactual supervision；
P2 P1 + Primary final low-LR。

每 epoch 强制测：
sign/mirror/lane-response/curvature/history-ratio，
不得只按 BC loss 选择 checkpoint。

clean150 达到 success>=0.38 且相对 legacy +5pp，
off-road下降、collision/timeout不恶化，
才允许扩大数据。

Round-25 失败：
停止 semantic adapter 训练扩展，
不得转向 DAgger/MoE/RL 掩盖问题。


这个阶段的中心任务不是“让 latent 包含 lane semantics”，这一点已经完成；现在必须证明：

Semantic 能在不破坏旧策略的前提下，持续、稳定、因果地影响最终动作。