总体裁决

Round-32 Level-1 审计很有价值，GO_ACTION_CONDITIONING 是目前最合理的主方向。但我不完全接受“D2→D3 占 97%，因此主瓶颈已经严格证明是 latent transition”这一表述。

更严谨的结论应是：

当前 imagined latent 与真实未来 latent 之间的差异，几乎不能由执行 pose 偏差或当前使用的动态图几何替换解释；与此同时，WM 对动作，尤其 ds，缺乏足够辨识力。因此，显式动作条件化是首个有资格进入训练 canary 的结构修改。

round32_audit_summary.md 显示：

D0→D1 只解释约 2.9%；
D1→D2 约 0%；
D2→D3 约 97%；
ds 的有限差分响应比 dθ/κ 弱约 26–50 倍； -正确 action 在五个 candidate 中 top-1 只有 10.2%，top-3 为 63.8%，margin 约为零； -真实 latent 足以表达关键 ego/lane 状态，但 WM 对 lane-relative 信息的保留明显不足。

这些证据足以批准动作条件化 canary，但还不足以批准大规模 WM 重训、BPTT 或 WM–Policy 全解冻联合训练。

一、Round-32 真正证明了什么
1. 动作条件化确实存在严重不足

这是本轮最强证据。

有限差分响应

ds 对 z_ego 和后续 action 的影响远弱于 dθ/κ。例如 h1 的 z_ego 响应：

ds ±0.10      0.0175
dθ ±0.01      0.8705
κ  ±0.01      0.8722


即使考虑不同变量单位，这个差距仍然过大，且与此前部署链的 ds overshoot 相互印证。

Action Swap

若 WM 真正理解：

zt+1=f(zt,at)z_{t+1}=f(z_t,a_t)

那么匹配的真实 action 应比随机、反向或 zero action 更容易预测真实 next state。

目前：

matched top-1 = 10.2%
matched top-3 = 63.8%
margin ≈ 0


五候选随机 top-1 期望约 20%，当前 matched 甚至没有表现出可靠优势。不同 action 下预测误差 spread 约 4e-3，远小于基准预测误差约 0.08。这说明预测函数几乎无法利用 action 区分 next state。

因此：

动作条件化不足不是推测，而是 Level-1 已经获得的最明确机制证据。

这也符合成熟 World Model 的一般设计。DreamerV3 通过状态和动作预测未来表示，并利用 imagined trajectory 改善行为；TD-MPC2 同样把潜在动力学明确建模为 action-conditioned dynamics，而不是仅依赖预计位置间接编码动作。

2. 不能简单说“ego dynamics 已经保留得很好”

round32_audit_summary.md 将 progress、speed、heading 的 WM 保留率列为 0.91–1.10，但 frozen baseline 同样很高：

progress frozen ≈1.00
speed frozen ≈0.87
heading frozen ≈1.08


这说明这些量可能主要来自：

-静态场景相关性； -当前状态与短时间未来的强自相关； -数据集中速度变化较小； -线性 probe 可以依靠非因果 shortcut。

所以：

WM保留率高


不等同于：

WM正确预测了动作导致的ego dynamics


尤其 progress 的 frozen 对照已经约为 1.0，这个 probe 无法证明 WM 对 progress 有动态预测增益。

必须补一个条件化增益指标

后续应报告：

ΔRWM2=RWM2−Rfrozen2\Delta R^2_{\text{WM}} = R^2_{\text{WM}} - R^2_{\text{frozen}}

以及：

ΔRaction2=Rcorrect action2−Rshuffled action2\Delta R^2_{\text{action}} = R^2_{\text{correct action}} - R^2_{\text{shuffled action}}

如果 WM 与 frozen 几乎相同，说明所谓“保留”来自状态惯性，不是动作动力学。

3. Lane-relative 信息损失是第二个真实问题

真实 z_ld 逐槽表示可以很好解码：

lane_lat  R² ≈0.90
lane_head R² ≈0.93


但 WM latent 的保留率仅：

lane_lat  0.61
lane_head 0.77


且逐槽展开明显优于 mean pooling，说明 lane slot 结构本身很重要。

更值得警惕的是：

WM 的 OD/LD latent 可以高精度解码 ego progress/speed，而真实 OD/LD latent 对这些量的可解码性很低。

这说明 imagined latent 可能出现语义串扰：

z_ld / z_od
被ego运动信息污染


而不是分别保持：

ego状态
对象状态
车道状态


这可能解释为什么 latent cosine 尚可，但 Policy 在 imagined latent 上动作明显漂移。

因此，动作条件化 canary 不能只看：

E_interface有没有下降


还必须检查：

lane信息是否恢复
OD/LD中的ego泄漏是否减少


否则显式 action embedding 可能进一步把 ego action 广播到所有 slot，反而强化语义污染。

二、对“D2→D3 97%”的保留意见
1. D3 本身被定义为 oracle latent

D3 直接使用真实未来 observation 编码出的 latent，误差被定义为零。因此：

D2 → D3


天然包含所有尚未解释的差异。

所以“97%”准确表达的是：

pose replay 和当前 oracle geometry 替换只解释了约 3%，其余 gap 留在 latent transition/状态更新体系中。

但它不能进一步区分：

transition 网络本身；
action conditioning； -latent target 漂移； -slot identity； -mask； -step embedding； -编码器 target 共适应； -状态语义串扰； -未记录的上下文差异。

所以我建议将报告措辞从：

主瓶颈 = latent transition


修正为：

主瓶颈定位在 latent-state evolution block，
其中 action conditioning 是首个获得直接失败证据的子机制。

2. D1→D2 为零的证据仍有限

本轮 D1/D2 无法计算完整的动作级：

π(z_hat_k)


因为录制缺失部分 t0 context 键；同时位姿置零自检表明模型本身对 OD/LD pose 通道就很不敏感。

因此：

Oracle geometry没有改善


可能表示两件事：

动态图确实不重要； 2.模型没有有效消费图 pose，所以换 oracle pose 也没有作用。

报告已经意识到这一 caveat。因此当前可以说：

不优先改动态图。

但不应该永久关闭动态图支路。显式动作条件化修复后，图几何通道可能才变得可消费。

三、是否批准 GO_ACTION_CONDITIONING
批准，但只批准最小、可逆 canary

round32_decision.md 提议：

ds/dθ/κ归一化特征
+ 零初始化动作残差
+ alpha=0逐位兼容
+ 60/40 expert/policy数据
+ 保持detach
+ 不BPTT
+ 不扩容


整体正确。

但我建议不要一步把同一个 action embedding 无差别加入 ego、OD、LD 三类 transition。因为 P3 已揭示 OD/LD latent 出现 ego 运动泄漏，无差别广播 action 可能加重这个问题。

四、建议的动作条件化三臂
AC0：当前模型

不训练，基线。

AC1：仅 Ego Action Conditioning

归一化动作：

ds_norm
dtheta_norm
kappa_norm
speed_intent = ds / dt


生成：

a_emb = MLP(action_features)


只进入：

ego_transition


形式：

zego′=zego+Tego(hego,zego)+αegoAego(zego,a)z'_{ego} = z_{ego} + T_{ego}(h_{ego},z_{ego}) + \alpha_{ego}A_{ego}(z_{ego},a)

初始化：

A_ego输出层全零
alpha_ego = 0


优点：

-直接修最弱的 ds 通道； -不会立刻把 action 广播到 lane/object latent； -便于判断 ego 动作条件化是否已经足够改善整条链。

AC2：Ego + Relative Scene Conditioning

只有 AC1 的 ego action fidelity 有效，但 OD/LD 相对状态仍明显错误时才运行。

不要简单将同一个 a_emb 加到每个 OD/LD slot。更合理的是使用 ego motion compensation：

relative_action_features:
    delta_x_ego
    delta_y_ego
    delta_heading_ego
    delta_speed_ego


分别作用于：

od_transition
ld_transition


这表达的是：

自车动作如何改变相对对象和相对车道状态。

而不是暗示道路或其他车辆本身由 ego action 驱动。

五、训练范围

第一轮 AC1：

训练
ego action embedding
ego transition residual支路
必要的ego latent physical decoder

冻结
主Encoder
原ST-GNN主体
OD transition
LD transition
PolicyBackbone
Primary
PolicyHead
ValueHead
MoE


为什么第一轮不训练整个 WM？

因为我们需要回答单一因果问题：

显式 action conditioning 能否修复 ds 不敏感和 rollout overshoot？

如果同时解冻所有 transition，结果无法归因。

六、数据建议
第一阶段不必立刻 60/40 混合大训练

先做两个小 canary。

C-A：Expert transitions only

验证结构本身是否能学习 action discrimination。

目标：

Action Swap top-1
ds-progress monotonicity
teacher-forced prediction

C-B：Expert 60% + Policy 40%

验证是否能迁移到部署分布。

如果 C-A 都无法提升 Action Swap，问题在结构或监督，不是 Policy 数据不足。

如果 C-A 可改善但 C-B 部署链仍无改善，问题可能在 Policy-distribution adaptation 或 Stage B 接口。

七、需要新增的 ego physical anchor

我建议 AC1 同时增加一个很小的 z_ego_next 物理解码头，但必须作为独立开关：

z_ego_next
→ [
    delta_progress,
    speed,
    delta_heading
]


第一轮先不加入 lane-relative 目标，因为 lane-relative 状态主要在 z_ld 中，且当前正存在 slot 语义和 ego 泄漏问题。

三个候选应分开：

AC1：
显式action conditioning

AC1-P：
action conditioning + ego physical anchor

P-only：
ego physical anchor但无action conditioning


否则若结果改善，将无法判断来自 action 输入还是物理监督。

八、canary 验收标准
Gate 0：兼容性
alpha=0：
所有forward max_abs_diff = 0
旧checkpoint missing/unexpected = 0
训练关闭时RNG序列不变
MoE-off baseline逐位复现

Gate 1：Action Fidelity

相对 AC0：

Action Swap matched top-1：
10.2% → 至少35%

top-3：
63.8% → 至少85%

matched margin：
必须转为正

ds FD sensitivity：
相对 dθ/κ 的差距
从26–50×缩小到 <=10×

ds-progress单调率 >=90%
zero/opposite action明显劣于matched


第一版不要求 top-1 直接达到 70%，因为当前起点几乎随机，35% 已足以证明结构有效。

Gate 2：物理语义
progress/speed：
WM相对frozen的增益为正

lane_lat/lane_head保留率不得下降超过5pp

OD/LD latent中的ego progress/speed泄漏
不得增加超过10%


如果 action fidelity 提升但 lane 保留继续恶化，不允许进入完整训练。

Gate 3：部署接口
E_interface ds：
h3–h5至少下降30%

E_action ds：
h3–h5至少下降25%

h5 plan_ds overshoot：
1.66m至少下降40%

真实状态Policy action误差：
退化 <=5%

Gate 4：闭环

使用：

K1e2
MoE-off
WM-plan
fixed LQR
calib50 paired


最低：

success >=0.58
相对0.56至少+2pp
off-road <=0.14
collision <=0.08
timeout不高于当前


由于 50 episodes 的 +2pp 只是一个 episode，只能作为继续许可，不能作为收益结论。

进入 clean150 的门应更高：

calib50 success >=0.60
或
接口/动作链指标达到Gate 3强门

九、什么时候允许 Policy–WM 交替训练？

动作条件化 canary 通过之后，才进入：

W-step
Policy冻结
WM在expert + policy transitions上更新

P-step
WM冻结
Policy接受real latent和WM latent训练


损失：

Lreal-action=∥π(zreal)−aexpert∥L_{\text{real-action}} = \|\pi(z^{real})-a^{expert}\| Linterface=∥π(z^WM)−sg⁡(π(zreal))∥L_{\text{interface}} = \|\pi(\hat z^{WM})-\operatorname{sg}(\pi(z^{real}))\|

先交替，不跨模块反传。

成熟的 World Model 方法也通常保持“模型学习”和“行为学习”的职责边界。DreamerV3 从真实经验学习世界模型，再利用 imagined trajectories 改善行为；TD-MPC2 则使用明确 action-conditioned latent dynamics，并通过控制相关目标约束模型。二者都不是把全部模块无约束地用单一损失一起拖动。

十、对 round32_decision.md 的最终修改建议

建议把 decision 从：

GO_ACTION_CONDITIONING


细化为：

GO_EGO_ACTION_CONDITIONING_CANARY


并锁定：

1. 先只给ego transition显式动作残差；
2. AC0 / AC1 / P-only / AC1-P四臂；
3. 不解冻原WM主体；
4. 不改OD/LD transition；
5. 不BPTT；
6. 不训练Policy；
7. MoE-off；
8. Action Fidelity + lane retention + leakage必须同时验收；
9. AC1通过后才采60/40 policy transition做第二阶段；
10. 未过门不自动扩大模型或数据。

最终判断

Round-32 已经把最大根因缩小到足够动刀的程度：

真实状态表示基本充分
执行pose差异不是主因
当前动态图不是首要解释
Policy中位敏感性不是主因
latent evolution block解释绝大多数gap
其中动作条件化几乎失效，尤其ds通道


因此动作条件化 canary 是对的。

但要防止从一个错误走向另一个错误：

不能因为 action conditioning 弱，就把 ego action embedding 广播给所有 ego/OD/LD latent，再全量解冻，以至于加强当前已经存在的语义串扰。

正确的第一刀应非常小：

只修 ego transition 的显式动作条件化
+
独立验证 ego physical anchor
+
严格监控 lane 保留与 OD/LD ego 泄漏


如果它能让 Action Swap 从近随机变为明确可辨，同时降低 h3–h5 的 ds overshoot，再进入 Policy 分布适配和 WM–Policy 交替训练。

这条路径比继续调 lane loss、tail projection、DAgger 或 PPO 更接近真正的 root cause，也最有希望把当前同口径 0.56 推向 0.60 以上，再逐步逼近 IDM+Pure Pursuit 的 0.756。