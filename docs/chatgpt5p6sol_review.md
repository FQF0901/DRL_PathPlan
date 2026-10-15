总体裁决

Coding Agent 的反驳和细化大部分成立。可以按 R39-A → R39-B → R39-C → R39-D 推进，但需修正参数预算和 Memory 定位。

核心共识：

当前主问题：
第2–6步 imagined state 不完整、不真实，
导致同一个 Policy 在想象状态上失去正确决策依据。

下一步不是增加第二动作头或扩大网络，
而是统一：
WorldState
→ PolicyStateView
→ 同一个 StructuredPolicyHead

一、对 Coding Agent 分歧的逐项裁决
1. 参数预算：不接受只看“激活参数”

Coding Agent 提议排除关闭的 MoE 和死路径，只按激活参数执行 ≤1.5M。

我的裁决：

同时记录总参数和激活参数，不能只看激活参数。

原因：

-闲置参数仍占 checkpoint、加载复杂度和维护成本； -未来误开 MoE 或旧 Policy 会突然改变实际计算； -只看激活参数容易继续叠模块； -但现在立刻大规模删除旧模块会干扰主问题验证。

执行口径：

当前总参数：
以最新HEAD实际值为冻结基线，约1.51M

当前激活参数：
约0.995M–1.103M

R39期间：
总参数原则上不得增加超过2%
激活参数原则上不得增加超过5%

禁止1.25×扩容臂
允许1.0×与0.75×等预算对照


新的状态 residual、readout 或 anchor 更新模块，应优先：

复用现有层
替换闲置路径
减少hidden后腾出预算


不是继续叠加。

达到基本闭环性能后，再统一清理：

关闭的MoE
旧Policy死路径
旧兼容分支
重复readout

2. 最新 HEAD 基线必须重登记，接受

Coding Agent 指出：

最新版环境builder已经原生提供真实动态距离和TTC


因此最新版 C1 已不再等同历史 C1，而更接近过去的 C1_true。

这项反驳完全成立。

必须重新登记：

C0_HEAD：
最新版正式基线

C1_HEAD：
最新版Structured Policy + 当前WM + 原生字段

旧C1：
只保留历史记录

c1_fill：
降级为缺键兼容诊断


禁止把旧 0.304 与新版 0.390 直接比较并宣称模型提升，因为其中包含 observation contract 变化。

正式重测必须携带：

protocol hash
model hash
schema hash
checkpoint hash
git commit

3. 双 Memory：接受“隔离”，反对先扩容量

Coding Agent 的修正正确：

现有实验已经显示完整历史与重复最近帧近似，因此不能以“历史很重要”为由先增加 Transformer 或更大 Memory。

双 Memory 当前应只解决生命周期与污染问题：

Real Memory：
保存真实环境历史
episode/scenario reset
只读

Imagined Memory：
从真实记忆的编码状态复制
只保存预测anchor、latent、validity
每次rollout结束销毁
绝不写回Real Memory


它是：

状态API与隔离机制


不是扩容理由。

Memory 表示实验只做三臂：

M0：当前最近状态

M1：当前状态 + 对齐后的历史delta

M2：M1 + 轻量attention


顺序必须是：

M1显著优于M0
→ 才允许M2

M1无收益
→ 停止，不加容量

4. 历史 delta 方案可做，但类别字段不得相减

接受 Coding Agent 的提议，先做离线零训练验证，再接入网络。

正确 delta：

OD：
位置差
速度差
relative heading差
presence变化

LD：
ego-motion对齐后的几何差
heading差
curvature差
validity变化


禁止：

type id做减法
lane role做减法
object class做减法
raw slot index做差


前提是：

OD完成slot identity对齐
LD完成ego-motion坐标对齐


未对齐的 delta 没有物理意义。

5. 多步梯度实验接受，但放在统一 State 接口之后

现有事实已经明确：

每步latent detach
→ 后续误差不能修正前一步transition


但不能直接删 detach。

实验顺序：

D0：当前1-step detach基线

D1：2-step truncated BPTT

D2：3-step truncated BPTT

D3：
主状态链保持detach，
只允许小型interface residual跨步反传


每臂记录：

h1/h3/h6 state loss
h1/h3/h6 Policy action consistency
各模块grad norm
显存峰值
梯度爆炸
训练耗时
闭环指标


先统一 WorldState → PolicyStateView，再做 D0–D3。否则是在不完整状态接口上延长梯度链，意义不足。

6. Learner-state 数据同时训练 WM，接受但后置

原则成立：

Policy标签：
learner state → expert action

WM标签：
learner state + executed action → real next state


必须使用：

executed_action


不能使用未执行的 expert action 训练真实 transition。

但当前不立即启动 DAgger。执行位置：

R39-A 工程闭合
→ R39-B WorldState接口过门
→ R39-C Memory表示裁决
→ R39-D learner-state transition聚合


训练仍采用交替冻结：

WM-step：
冻结Policy，更新WM

Policy-step：
冻结WM，更新Policy


禁止一个总损失同时修改两者。

二、R39 的正式方案
R39-A：工程闭合，零训练

立即开工。

任务
正式 CLI/评测入口可加载 StructuredPolicyHead；
checkpoint migration 接入正式 loader； 3.输出完整 LoadReport； 4.生产调用统一迁移到 forward_mode； 5.删除易误用的布尔 shim，保留 fail-fast； 6.原生记录四个 hash； 7.升级参数账本：
total；
active；
frozen；
dormant； 8.升级 detach/梯度账本； 9.重跑最新版：
C0_HEAD Clean500；
C1_HEAD Clean500；
IDM + Pure Pursuit。
验收
正式入口与测试入口逐位一致
checkpoint严格加载
旧模型迁移报告完整
布尔shim无生产调用
四hash齐全
参数分类无重叠、无遗漏
最新基线可复现

R39-B：统一 imagined WorldState
唯一状态契约
WorldState_k：
ego physical + ego latent
OD anchors + OD latent + validity
LD anchors + LD latent + validity
nav/route state
memory state


每一步：

1. Policy读取PolicyStateView_k
2. 输出当前action a_k
3. 解析运动学更新ego
4. LD anchor做ego-motion补偿
5. OD anchor做对象先验+ego-motion补偿
6. WM预测剩余残差
7. 统一readout重算Policy需要的全部物理量
8. 更新Imagined Memory
9. 同一个Policy决策下一步


公式：

Ak+1=AnalyticUpdate⁡(Ak,ak)+ΔAθ(Sk,ak)A_{k+1} = \operatorname{AnalyticUpdate}(A_k,a_k) + \Delta A_\theta(S_k,a_k)

然后统一计算：

speed
left/right margin
static clearance
dynamic distance
TTC
route progress
distance-to-end


禁止：

评测脚本注入未来真值
t0字段静默hold
缺失字段静默填0
外部提供未来状态

对照臂
B0：当前实现

B1：只使用analytic update

B2：analytic update + learned residual

B3：B2 + 2-step BPTT

B4：B2 + 3-step BPTT


B2 必须优于 B1，才能证明学习残差有贡献。

R39-C：Memory 表示

只有 R39-B 过门后执行。

M0：current state only

M1：current + aligned historical deltas

M2：M1 + lightweight attention


M1 无显著收益则停止，不运行 M2。

R39-D：Policy 分布适配

执行：

expert transition
+
learner-state real transition


World Model：

real state + executed action → real next state


Policy：

real state → expert action

imagined state
→ 与real future state具有决策一致性


仍然交替冻结，不联合作弊。

三、必须新增的 Probes / Hooks
WorldState Hook

每个 horizon 保存：

ego pose/speed
OD anchor before/after analytic update
LD anchor before/after ego-motion compensation
learned residual
validity/presence
PolicyStateView所有字段
字段来源：analytic / learned / held / missing


其中：

held
missing
external_future


在正式模式必须为零。

State Readout Hook

逐字段记录：

value
unit
coordinate frame
producer
presence mask
horizon


并断言：

同一字段只有一个正式producer
训练和部署producer一致

Memory Hook
Real Memory写入次数
Imagined Memory写入次数
Real→Imagined复制时刻
rollout结束销毁
是否发生Imagined→Real写回
episode reset是否生效


硬门：

Imagined→Real写回次数 = 0

梯度 Hook

分别对：

state prediction loss
physical anchor loss
Policy interface loss


在：

h1
h3
h6


单独 backward，记录：

WM transition
anchor residual
readout
PolicyBackbone
StructuredPolicyHead
Encoder

参数 Hook

每次运行输出：

total parameters
active parameters
frozen parameters
dormant parameters
optimizer parameters
per-module update norm


并断言：

总参数增量 <=2%
激活参数增量 <=5%
optimizer只包含声明模块

四、验收准则
工程门
正式loader可加载Structured Policy
forward_mode全迁
四hash完整
HEAD Clean500可复现
无旧臂语义混用

状态接口门
第2–6步无t0静默hold
无缺失字段静默填0
所有Policy输入具有明确producer
h1–h6误差不指数增长
B2优于analytic-only B1


当前既有门可以保留，但必须在文档中写明单位和统计口径：

slow+accel Σds >=12.8
全样本接口误差 <=2.31


否则不能作为硬门。

Memory 门
M1显著优于M0
才允许M2

若M1无收益：
Memory不扩容

Detach 门
2-step或3-step至少有一个显著优于1-step
h1不得恶化
梯度无爆炸
显存适配12GB

参数门
总参数不超过当前HEAD基线+2%
激活参数不超过当前激活基线+5%
禁止1.25×扩容臂

闭环门

先建立最新基线后，新候选必须：

相对C1_HEAD success提升 >=5pp
off-road不恶化
collision不恶化
timeout不恶化
route completion不下降


并要求：

Clean500
至少3 seeds


单个 50-episode 结果只能作为继续许可，不能作为收益结论。

给 Coding Agent 的新答复

接受 R39 的总体方向和 A→B→C→D 顺序。参数预算不采用单一“激活参数≤1.5M”口径，而采用总参数与激活参数双账本：R39 期间总参数不得超过当前 HEAD 基线 2%，激活参数不得超过当前激活基线 5%，取消 1.25×扩容臂，暂不做大规模瘦身。首先完成 R39-A 的正式装载、checkpoint migration、forward_mode 全迁、四 hash、参数/梯度账本和 HEAD Clean500 重登记。随后 R39-B 统一 WorldState → PolicyStateView，每个 imagined step 必须由解析 anchor 更新、WM 学习残差和统一 readout 重算全部 Policy 输入，禁止脚本注入、t0 hold 和静默补零。R39-C 的双 Memory 仅作真实/想象隔离，历史 delta 先做等预算验证，无收益不加 attention。R39-D 才引入 learner-state transition，并与 Policy 的专家标签分开、交替冻结训练。上述状态接口和最新基线门未过，不启动 DAgger、MoE 或 RL。