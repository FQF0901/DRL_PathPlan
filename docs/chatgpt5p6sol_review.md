总体裁决

这轮 canary 的结果有价值，但暂时不能按报告原结论直接进入 C-B 或进一步放大 action conditioning。

最关键的新事实有三条：

显式动作条件化确实能改变 WM 性状，Action Swap、有限差分敏感度、单调性等指标明显变化；
动作辨识变强，没有转化成 rollout 接口改善，反而让 ds 尾段漂移和 overshoot 加重；
AC1 闭环 success 从 0.56 到 0.60，但 collision 从 0.08 跳到 0.18，说明当前改动更像“增强进度与激进性”，还不是可靠 World Model 改进。

此外，我在代码里发现一个必须优先核验的严重问题：

action_ego 输出层和 alpha_ego 同时初始化为零。按当前公式，两者理论上会形成零梯度死锁。

所以第一步不是继续训练，而是先证明这次 canary 的新分支究竟如何学起来，以及 checkpoint 中实际的 alpha_ego、输出层参数是否确实发生变化。

一、先审阅代码具体修改
1. 动作特征设计存在冗余

net/st_gnn.py 当前动作特征为：

[
    ds / 10.0,
    dtheta / 1.2,
    kappa / 1.2,
    (ds / dt) / 20.0,
]


但当前固定：

dt = 0.5s


所以：

ds/dt20=ds/0.520=ds10\frac{ds/dt}{20} = \frac{ds/0.5}{20} = \frac{ds}{10}

即：

ds_norm == speed_intent_norm


代码注释也明确承认两维完全等价。

问题

这不是致命 bug，但会：

-重复强化 ds； -让 MLP 两个输入维实际表达同一信息； -掩盖“速度意图”和“位移”本应独立的物理语义； -可能加剧本轮看到的纵向激进性和 overshoot。

建议

当前 action 特征应简化为：

ds_norm
dtheta_norm
kappa_norm


如果未来要加入速度意图，应使用不同于 ds/dt 的量，例如：

target_speed - current_speed
delta_speed
longitudinal_acceleration_intent


这些需要当前真实 ego speed，不能用 ds 的线性重复项冒充。

2. kappa 特征在低 ds 区间存在人为形状

当前：

kappa = dtheta / ds.clamp_min(0.5)


这意味着：

ds < 0.5m


时，所有样本都使用固定分母 0.5。

结果是：

-低速曲率语义被人为压缩；

ds=0.05 与 ds=0.49 的 κ 都使用同一分母； -停车、低速恢复、tollgate 区域可能获得不自然的动作 embedding； -动作干预 probe 的低速结果可能受 clamp 规则主导。
建议

首轮不要改公式，但必须按速度区间分层报告：

ds < 0.5m
0.5 <= ds < 1.5m
1.5 <= ds < 3m
ds >= 3m


如果 collision 或 overshoot 主要集中在某个速度段，再修改低速表示。

3. 最严重问题：双零初始化可能导致分支完全无法学习

当前初始化：

self.action_ego = _latent_head(...)
self.alpha_ego = nn.Parameter(torch.zeros(()))


而 _latent_head() 的最后一层也是：

weight = 0
bias = 0


前向：

z_ego_next += alpha_ego * action_ego(...)


初始时：

alpha_ego = 0
action_ego(x) = 0


因此：

对 alpha_ego 的梯度
∂L∂α=∂L∂z′Aego(x)=0\frac{\partial L}{\partial \alpha} = \frac{\partial L}{\partial z'} A_{\text{ego}}(x) = 0

因为 action_ego(x)=0。

对 action_ego 参数的梯度
∂L∂θA=∂L∂z′α∂A∂θA=0\frac{\partial L}{\partial \theta_A} = \frac{\partial L}{\partial z'} \alpha \frac{\partial A}{\partial\theta_A} = 0

因为 alpha=0。

也就是说，按附件中的生产代码，理论上：

alpha_ego没有梯度
action_ego也没有梯度
整个分支永久为0


但报告中 AC1 与 AC1-P 明显改变了结果。这只有几种可能：

canary 脚本训练前手工修改了 alpha_ego； 2.脚本手工重新初始化了最后一层； 3.实际训练 checkpoint 不对应附件中的生产初始化逻辑；
optimizer 或加载逻辑有额外操作； 5.所谓 AC1 的变化来自别的参数被意外解冻； 6.结果产物和最新源代码不在同一 commit。
这是当前第一优先级

必须立即输出：

训练前：
alpha_ego
action_ego最后层weight norm
action_ego最后层bias norm

第一次backward后：
grad(alpha_ego)
grad(last_weight)
grad(last_bias)
各模块grad norm

第一次optimizer.step后：
参数变化量

最终checkpoint：
alpha_ego
各层参数norm


并进行单 batch smoke test。

预期

如果附件代码原样执行：

所有新分支梯度和更新都应为0


若实测不为零，应定位外部初始化逻辑。

在这个问题闭合前，不能把本轮变化归因于 action-conditioning 架构。

4. 惰性构造方式需要防止 optimizer 漏参

set_action_ego(True) 调用时才创建：

action_ego
alpha_ego


风险在于：

optimizer创建在前
set_action_ego调用在后


那新参数不会进入 optimizer。

虽然本轮训练显然做了某种专门处理，但正式实现必须 fail-fast：

assert optimizer_parameter_ids == declared_trainable_parameter_ids


并在 manifest 中写：

action_ego parameter count
alpha_ego parameter included
ego_phys parameter included
trainable parameter hash


不能靠调用顺序隐式保证。

5. 将新参数完全从默认 state dict 中移除，有兼容优势，但增加实验复现风险

惰性构造的好处：

默认关闭时旧模型键集逐位一致


这是合理的 canary 设计。

但正式进入主线后，如果 enable_action_ego 只是普通 Python 属性、不进 checkpoint，那么加载模型时可能发生：

checkpoint里有action_ego参数
但运行时忘记set_action_ego(True)


从而静默关闭新结构。

正式化前必须做到

checkpoint metadata 写入：

{
  "action_ego_enabled": true,
  "ego_phys_enabled": false,
  "action_feature_version": "v1",
  "action_feature_dim": 3,
  "alpha_mode": "fixed_or_bounded"
}


加载时：

checkpoint启用了action_ego
但模型未构造
→ fail-fast


不能静默兼容。

二、如何解释 Gate 1 和 Gate 2
1. 结构确实获得了动作辨识能力，但还不够稳

AC1：

Action Swap top-1：
0.101 → 0.284

top-3：
0.635 → 0.917

zero−matched：
负值 → +0.186

FD dθ/ds：
38.74 → 7.34


说明新结构至少改变了 action-conditioned discrimination。

但 top-1 仍低于 0.35 的门，而且 0.284 也只是从“明显劣于随机”上升到“略高于五候选随机的 0.20”。

所以正确措辞是：

动作信号开始可用，但仍不足以稳定选择真实 action。

不能写成“动作条件化问题已经解决”。

2. AC1 单调率反而严重恶化

AC1：

ds-progress 单调率：
0.743 → 0.202


AC1-P：

0.946


这很关键。

它说明：

AC1虽然更能区分action，
但学到的动作响应不符合正确的纵向物理单调性。


也就是动作辨识变强，但因果方向错乱。

AC1-P 加入 ego physical anchor 后才恢复单调性，说明：

显式动作输入本身不够，必须有物理锚约束它“如何响应动作”。

这与成熟 World Model 的做法一致。动作条件化动力学通常不能仅靠 latent matching，需要由状态、奖励、物理量或控制相关目标共同约束。DreamerV3 通过动作条件化的世界模型预测未来表示并训练行为；TD-MPC2 则以控制相关的 latent dynamics 进行局部规划，而不是只要求不同动作产生不同 latent。

3. Gate 2 的 lane 保留改善值得肯定，但不能过度解释

AC1 的 lane 保留：

lat  0.456 → 0.521
head 0.415 → 0.601


这说明 action conditioning 并没有简单破坏 z_ld，甚至可能通过更准确的 ego transition 改善了图消息。

但同时：

od_pool中的ego progress泄漏：
AC1 +66%
AC1-P +41%


这支持了我们前一轮的担忧：

ego 动作信息虽然只显式进入 z_ego，但仍会通过 ST-GNN 消息传递泄漏进 OD latent。

因此“只给 ego transition 加 action”并不等于 OD/LD 完全不受影响。

下一轮必须进一步检查：

OD泄漏发生在哪个horizon
哪些OD slot最严重
是否集中于同车道前车
是否与collision episode相关

三、Gate 3 比 Gate 4 更能说明根因
1. Gate 3 明确失败

AC1：

E_interface ds：
+20.9%

E_action ds：
+17.2%

h5 overshoot：
+16.5%


AC1-P：

接口与overshoot近似中性，但没有改善


这说明动作条件化让 WM 更敏感，却没有让 rollout 更准确。

最合理的解释是：

Policy输出略偏大的ds
→ WM更敏感地响应
→ predicted ego latent推进更激进
→ Policy在更激进latent上继续输出
→ 尾段形成正反馈


也就是：

原来 WM 对 action 过于迟钝；现在增加灵敏度后，暴露出 rollout 缺少稳定反馈与物理约束，导致误差被更快放大。

所以 AC1 不是“修复后只剩 collision 小问题”，而是：

修复了动作不可辨识
却加剧了闭环 imagined rollout instability

2. Gate 4 的 success +4pp 不能覆盖 Gate 3 失败

AC1：

success 0.56 → 0.60
timeout 11 → 5
off-road 0.14 → 0.12
collision 0.08 → 0.18


这很像：

车辆更敢走、更少超时
但安全性显著下降


其机制可能只是纵向更激进，而不是规划理解更准确。

同时 50 episode 中：

3翻正 / 1翻负


样本太少，不足以证明稳定收益。

当前正确结论
AC1提高了进度和成功率倾向，
但以安全退化和rollout误差增加为代价。


它没有过最终门，也不应作为新 baseline。

四、AC1-P 为什么更物理，却闭环更差？

AC1-P：

-动作单调性最好； -真实状态 Policy action 误差更低； -collision 最低； -但 success、timeout、route completion 更差。

这可能说明 ego physical anchor 将行动约束得更保守或更接近专家数据，但当前 expert 数据本身在：

配速
进度
低速恢复


方面并不足以达到部署上的高成功率。

也可能是 physical head 的目标尺度和权重不合适，让 latent 优先满足：

局部物理预测


却损害：

Policy可消费性
长期route progress


因此 P-only 没有作用，而 AC1-P 有明显作用，实际上证明：

physical loss 通过 action branch 改变了 latent dynamics，但当前物理目标仍不足以定义“对闭环最有用的 latent”。

下一轮不能简单增加 physical loss 权重。

五、代码更改的正面评价

尽管结果未过门，工程实现里有几个地方做得不错：

只将 action residual 接入 ego transition，没有无差别注入 OD/LD； 2.默认关闭时保持旧 checkpoint 键集； 3.在 net/model.py 部署 rollout 和 pipeline/stages.py Stage A teacher rollout 中都传入了对应 action； 4.将 ego_phys 加入 Stage B 各冻结表，避免后续 BC 意外训练；
Gate 0 覆盖了 strict loading、前向逐位、RNG、梯度范围和完整测试； 6.P-only 逐位等于 base，为机制消融提供了可靠对照。

这些值得保留。

六、不要马上进入 60/40 Policy transition 训练

报告遵守预注册，没有自动进入 C-B，这是正确的。

原因：

expert-only下rollout接口已经恶化


此时引入 Policy transitions 可能：

-帮助适配 learner distribution； -也可能进一步强化激进行为； -或让 WM 学会补偿某个固定 Policy 的错误，而不是学习真实动力学。

在搞清 AC1 collision 和 overshoot 机制前，不能扩大数据。

七、必须先做的 Round-33A：只读归因
1. 双零初始化和真实梯度审计

这是 P0。

输出：

alpha_ego init/final
alpha_ego grad at step0/1/10
action_ego layer norm init/final
new branch update norm
optimizer inclusion
checkpoint keys


结论必须是：

分支确实按预期获得梯度


否则整轮 canary 归因无效。

2. Collision episode 配对归因

对：

AC1新增collision
AC1翻正episode
AC1公共成功
base成功但AC1失败


逐 episode 输出：

road class
collision object type
collision time
route completion
actual speed
planned ds[0:6]
planned dtheta[0:6]
planned kappa[0:6]
TTC
front gap
LQR throttle/steer
horizon-wise z/action error
OD leakage


重点判断 collision 来自：

纵向追尾
交叉口冲突
merge/cutin
弯道侧碰
静态障碍物


如果主要是追尾，问题是纵向动作条件化缺少对象交互，而不只是残差幅度过大。

3. alpha_ego 和 residual 贡献分解

每个 horizon 保存：

legacy ego-transition delta norm
action residual norm
action residual / legacy delta
alpha_ego


分层：

normal success
new collision
timeout fixed
off-road


目标是确认是否存在：

action residual逐horizon累积膨胀


而不是盲目加 alpha 上限。

4. Action feature 消融，只读或极小 probe

比较：

F0 = [ds, dtheta, kappa, duplicate speed]
F1 = [ds, dtheta, kappa]
F2 = [ds, dtheta]
F3 = [ds, kappa]


不需要四臂完整闭环训练。先在已训练 branch 上做输入归因或通过小 probe 评估。

重点检查：

重复ds维是否放大纵向激进性
kappa在低ds时是否主导异常

八、把上轮讨论的 ego-motion 解耦正式并入下一步

当前结果进一步支持显式 ego-motion 解耦，但仍然要先做零训练验证。

为什么？

AC1 将 action 信息写入 z_ego 后：

OD ego-progress泄漏增加66%


这正是“ego action 通过消息传递污染 OD latent”的证据之一。

更物理的结构应该是：

ego action
→ 解析计算ego SE(2)位姿变化
→ 显式warp OD/LD physical anchors
→ latent transition只学习剩余语义和动态


这与 SparseDrive / Sparse4D 使用“实例 feature + 显式 geometry anchor + temporal instance propagation”的原则相近。SparseDrive 将动态对象表示为实例特征与包含位置、航向、速度的 anchor，将地图表示为实例特征与 polyline anchor；历史实例依赖几何 anchor 进行时间传播，而不是让 latent 独自承担全部坐标变化。

但 SparseDrive 是并行多模态轨迹规划，不是与你项目完全相同的 action-conditioned rollout WM，所以只能借鉴“几何与语义分离”的原则，不能直接照搬。

九、Round-33B：Analytic Ego-Warp Oracle Probe

不训练，不改 LQR。

W0：当前实现
ego累计pose进入图
OD使用t0恒速外推
LD保持t0 anchor

W1：显式 LD Ego Warp

对 LD 物理 anchor 做：

pld′=R(−Δθego)(pld−Δpego)p'_{ld} = R(-\Delta\theta_{ego}) (p_{ld}-\Delta p_{ego})

方向：

ψld′=ψld−Δθego\psi'_{ld} = \psi_{ld}-\Delta\theta_{ego}

曲率保持不变。

W2：显式 OD Ego Warp + 对象运动先验
pod′=Tego−1(pod+vodΔt)p'_{od} = T^{-1}_{ego} (p_{od}+v_{od}\Delta t)
W3：W1 + W2
W4：Oracle future anchors

使用真实未来 ego、OD 和 lane geometry。

比较：

lane信息保留
OD ego-motion泄漏
E_interface h1–h5
E_action h1–h5
ds overshoot
action swap

Ego-Warp Gate

如果 W3 相对 W0：

OD progress泄漏下降 >=30%
lane-relative保留提升 >=15%
E_interface h3–h5下降 >=15%
overshoot下降 >=20%


则批准：

GO_ANALYTIC_EGO_WARP


如果仅 W4 有效，说明：

显式坐标变换方向正确
但对象自身运动、slot或地图更新仍不足


如果 W1–W4 都无效，则 ST-GNN 没有有效消费 geometry anchor，应转向图消费结构，而不是继续加 warp。

十、Round-33C：修正版动作条件化 canary

只有完成归因和 warp probe 后执行。

首先修初始化

不要同时：

branch最后层=0
alpha=0


推荐二选一。

方案一，优先推荐
branch最后层=0
alpha固定为1


这样：

-初始前向逐位不变； -输出层第一步就有梯度； -不引入自由缩放标量； -结构最简单。

方案二
branch小随机初始化
alpha=0


这样 alpha 第一轮有梯度，但 branch 初始方向随机，不如方案一稳。

动作特征改为
ds_norm
dtheta_norm
kappa_norm


删除重复的 speed-intent。

低 ds 通过：

valid_kappa mask


而不是让 clamp 结果无条件主导。

三臂即可
B0：base

B1：ego action conditioning v2
    无physical head

B2：ego action conditioning v2
    + ego physical anchor


P-only 已经证明不会影响主链，不必重复完整闭环。

Ego physical target 应增加一个关键量

现有：

delta_progress
speed
delta_heading


建议加入：

delta_speed


因为：

speed本身


容易被 frozen/current-state shortcut 预测；

delta_speed


更能监督动作导致的纵向变化。

第一版仍不加入 lane-relative target，避免重新混淆 ego 和 LD 职责。

十一、修正版训练门
Gate A：训练可达性
新分支step0梯度非零
首个optimizer step参数变化非零
仅声明参数更新
alpha/残差norm可追踪

Gate B：动作因果
top-1 >=0.35
top-3 >=0.85
ds-progress单调 >=0.90
zero/opposite显著差于matched
FD dtheta/ds比 <=10

Gate C：接口稳定

这次应比上一轮更严格：

E_interface h3–h5至少不恶化
E_action h3–h5至少不恶化
overshoot至少不恶化

任何一项恶化 >5%
→ 停止


只有先做到“不放大”，才谈 30% 改善。

Gate D：安全闭环
success >=0.58
off-road <=0.14
collision <=0.08
timeout <=11
route completion >=0.862


必须全过，不能再用 success +4pp 抵消 collision 翻倍。

Gate E：clean150

进入条件：

calib50全门通过
+
接口指标改善


clean150：

success >=0.58
至少相对base +5pp
collision不劣
timeout不劣
3 seeds前先做paired置信区间

十二、后续训练顺序

如果修正版 action conditioning 与 analytic warp 均有效：

1. Ego action conditioning v2
2. Analytic ego-motion warp
3. Expert-only WM训练
4. Policy-transition 60/40 WM适配
5. 冻结WM，训练Policy real/predicted latent一致性
6. WM-step / Policy-step交替训练，梯度隔离
7. DAgger
8. MoE
9. Exact PPO vs deployment-isomorphic PPO


仍然不允许：

WM/Policy全量解冻
单一总loss端到端训练

十三、给 Coding Agent 的正式下一步指令
Round-32 canary不晋级C-B；AC1/AC1-P均不作为新baseline。

先执行Round-33A/B，只读归因：

A. 梯度与初始化审计
- alpha_ego及action_ego各层init/grad/update/final；
- optimizer membership；
- checkpoint/source hash；
- 解释双零初始化为何仍出现训练效果。

B. AC1 collision/overshoot配对归因
- episode级road class、对象类型、速度、TTC；
- plan ds/dtheta/kappa h0-h5；
- residual norm/legacy transition norm；
- OD leakage随horizon；
- 翻正/翻负/新增collision分类。

C. Action feature审计
- 删除重复speed-intent前先量化贡献；
- 按ds速度带报告kappa clamp效应。

D. Analytic ego-warp oracle probe
- W0 current；
- W1 LD ego warp；
- W2 OD ego warp+CV；
- W3 combined；
- W4 oracle future anchors；
- 比较lane retention、OD leakage、E_interface、overshoot。

本阶段禁止：
- C-B 60/40；
- 扩大数据；
- 改LQR；
- BPTT；
- 解冻Policy；
- DAgger/MoE/PPO/GRPO；
- 将AC1设为默认。

若梯度审计不闭合：
decision=BLOCKED_CANARY_VALIDITY。

若warp通过门：
decision=GO_ANALYTIC_EGO_WARP。

若warp无效但action结构仍成立：
decision=GO_EGO_ACTION_CONDITIONING_V2。

不得自动进入训练。

最终判断

这轮不是失败，反而回答了一个非常关键的问题：

让 WM“更能感知 action”并不等于让 rollout 更正确。

AC1 证明动作条件化方向有真实作用，但同时暴露了更深层问题：

动作敏感度增强
→ imagined latent推进更激进
→ Policy继续放大
→ ds overshoot与collision增加


因此下一步不能简单做：

加正则、限幅alpha、再训更多数据


而应先回答两个 root cause：

这次分支在双零初始化下究竟如何获得梯度；
ego action 是否应该主要通过显式 SE(2) warp 改变 OD/LD 几何，而不是依赖 latent 消息传播吸收坐标变化。

从长期架构看，最有希望的目标仍是：

显式ego运动学
+
OD/LD anchor坐标补偿
+
ego-only action-conditioned latent residual
+
Policy分布WM适配
+
real/predicted latent Policy一致性


这比把更多动作信息直接塞入所有 latent 更物理，也更接近 SparseDrive 的“几何 anchor 与语义 feature 解耦”原则。

只有这条链在接口、安全和闭环三个层面同时过门，才有资格从当前 0.56 稳定推到 0.60 以上，并继续逼近最终超过 IDM+Pure Pursuit 的 0.756 目标。