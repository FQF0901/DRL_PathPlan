我从“如果我是 reviewer，要把这个项目搞死” 的角度来审。

先说结论：

总体方向：正确
主要风险：不是RL，不是GRPO

而是：

1. Policy/WM状态定义不统一
2. Action语义与轨迹语义混杂
3. LD未来表示不闭合
4. DAgger成功率可能被评估口径误导
5. 未来WM上线时会出现representation drift


其中我认为真正的P0问题是前3个。

P0. 当前最大的疑问

你现在说：

IL = 2%
DAgger r3 = 61%


但我第一反应不是高兴。

而是：

61% 是谁开的车？


因为我看到当前配置：

expert_ratio = 0.65
shield_ttc = 3.5
stuck shield = 20


来自 DRL_PathPlan_Code.txt。

我会怀疑：

61%
=
Student能力
+
Shield能力
+
Expert介入能力


而不是：

纯Student


能力。

我建议马上做：

expert_ratio:
0.65
0.4
0.2
0


以及：

shield:
ON
OFF


矩阵。

如果：

61%
↓
58%
↓
54%
↓
49%


很好。

如果：

61%
↓
35%
↓
18%
↓
8%


那其实还没到RL阶段。

P1. 当前Net最大的结构问题

我认为是：

Action
=
(dx,dy,dpsi,v_next)


和：

Trajectory
=
6-step rollout


的语义混在一起。

你现在实际上在学：

a_t
→ 0.5s future endpoint


但是评估的是：

3s trajectory


这里有个隐患：

step local optimum
≠
trajectory optimum


例如：

每个0.5s都合法

但连续拼接

最后蛇形振荡


所以我建议增加一个loss：

segment consistency


检查：

step_i endpoint
vs
step_i+1 startpoint


一致性。

否则：

DAgger
很好

RL
突然开始抖方向


很常见。

P2. 文件架构有一个问题

我看你之前几版合同。

实际上有：

dataset/
net/
pipeline/
env/


这很好。

但有一个缺失层：

representation/


或者：

schema/


因为现在：

OD token
LD token
Nav
Action
Trajectory
Memory


定义散落在：

dataset
net
pipeline


里面。

后面改WM的时候一定出事。

我会单独搞：

schema/
  observation.py
  action.py
  trajectory.py
  memory.py


整个工程所有地方：

import schema


而不是：

每个模块自己定义

P3. WM上线时最大的炸点

这个不是现在。

但要提前设计。

你现在：

Policy


看：

OD
LD
Nav
Ego


未来WM：

OD'
LD'


提供：

imagined state


问题：

WM loss很好


不等于：

Policy还能决策


因为：

latent fidelity
≠
decision fidelity


这是目前方案里最大的潜在炸点。

我建议提前定义：

Action Consistency


指标。

同一个真实状态：

Real
→ Policy
→ a_real


和：

WM one-step
→ Policy
→ a_wm


比较：

KL
MSE


未来：

WM上线条件


不要写：

latent mse


而写：

action consistency

P4. 关于你想做GRPO

我其实支持。

但是有个前提。

你现在：

action
=
(dx,dy,dpsi,v_next)


GRPO天然适合：

一次采样G条


例如：

G=8


然后：

rollout


比较：

collision
progress
comfort


但是这里有个问题。

如果你现在：

single-step action


打分。

GRPO意义不大。

因为：

轨迹质量


通常需要：

1~3s horizon


才能看出来。

所以你的GRPO应该长这样：

Policy
↓
sample 8 trajectories
↓
rollout 3s
↓
group ranking
↓
GRPO


而不是：

sample 8 actions
↓
0.5s reward
↓
GRPO


后者收益有限。

P5. 关于RealBank

这里我和你站一边。

我不赞成：

Bank:
OD
LD
NAV
EGO


全塞进去。

我更喜欢：

Bank:
OD
LD


然后：

Nav
Ego


是条件输入。

原因：

WM


学的是：

世界变化


而：

Nav


不是世界状态。

是任务条件。

否则：

换个导航


WM状态就变了。

逻辑很怪。

如果我是现在的Adversarial Reviewer

我会给：

✅ 通过
History memory
DAgger
4D action
GRPO方向
OD/LD中心表示

⚠️ 必须补
Pure student success率
Shield影响量化
Action consistency gate
Schema统一
Trajectory consistency loss

❌ 不建议现在做
WM
Dreamer
Imagined RL
复杂GRPO reward


一句话总结：

当前架构最大的风险已经不是表达能力，而是“你以为61%是学生学会了，但实际上可能有大量 expert/shield 成分”。在确认 pure-student 能稳定达到 50%+ 之前，我不会把主要资源投入 WM，而会先把评估口径、trajectory consistency 和 GRPO 基础打扎实。

=====================================================
我觉得你已经碰到自动驾驶 WM 最核心的问题了，而且这个问题实际上比：

WM MSE
WM rollout error


更重要。

你说的本质是：

Policy 在真实ENV上训练

WM 在拟合ENV

二者永远不可能完全一致


所以目标不应该是：

WM = ENV


而应该是：

WM 对 Policy 来说是有用的


这是两种完全不同的思路。

1. 你说的其实是“Decision Fidelity”而不是“State Fidelity”

很多项目死在这里：

WM latent loss ↓↓↓

Policy性能 ↓


因为他们优化的是：

State Fidelity


例如：

L_wm
=
MSE(z_hat,z_gt)


但是自动驾驶真正需要的是：

Decision Fidelity


即：

在同一个状态下

WM给Policy的决策

≈

真实ENV给Policy的决策


举例：

真实世界：

前车距离 12m


WM：

前车距离 11m


MSE不小。

但是：

Policy

都选择刹车


那么：

Decision Error ≈ 0


反过来：

真实世界：

前车距离 10m


WM：

前车距离 10.2m


MSE很小。

但：

真实 -> 刹车

WM -> 加速


那么：

Decision Error 很大


所以我越来越觉得：

Policy Consistency


应该是主指标之一。

不是Probe。

而是Loss。

2. 我会怎么解决WM/Policy漂移

我不建议：

先训练WM

训练完

再接Policy


因为一定漂。

我的建议：

Phase D1

只训练：

L_wm


得到：

one-step WM

Phase D2

冻结Policy。

增加：

L_consistency
=
KL(
 Policy(z_real),
 Policy(z_wm)
)


即：

真实状态
↓
Policy
↓
a_real


和：

WM one-step
↓
Policy
↓
a_wm


对齐。

此时：

WM


必须学会：

给Policy有意义的latent


而不只是：

MSE最小

3. 为什么我觉得这个比Joint更稳

很多人第一反应：

Policy
+
WM

一起训练


实际上容易变成：

Policy适应WM错误

WM适应Policy错误


然后：

一起漂


非常像：

GAN collapse


所以我更喜欢：

Step1

固定Policy

训练WM

Step2

对齐：

Policy(z_real)

≈

Policy(z_wm)

Step3

最后再Joint。

这样：

Policy


相当于：

Teacher


WM：

Student

4. 你现在其实有一个天然优势

你和很多Dreamer项目不一样。

因为：

Policy


不是从WM里长出来的。

而是：

IL
↓
DAgger
↓
GRPO


在真实ENV里长出来的。

这意味着：

Policy


是可信的。

所以：

Policy


可以作为：

WM的语义教师


而不是：

WM当教师
Policy当学生


这点非常重要。

5. 我会把WM目标改成三层

现在很多人只有：

L_wm
=
latent mse


我建议：

Layer1
L_transition


普通：

latent loss

Layer2
L_decision

=
KL(
 Policy(z_real),
 Policy(z_wm)
)

Layer3
L_value

=
MSE(
 Value(z_real),
 Value(z_wm)
)


这样：

WM必须同时保持：

状态相似

决策相似

价值相似


这比只追：

latent mse


靠谱得多。

6. 你前面说 a_t rollout 6次得到3s trajectory

这个我同意。

所以我认为现在最大的好消息是：

你的Policy

本身已经是

latent planner


而不是：

waypoint predictor


因此：

RealBank
↓
Policy
↓
a1..a6


和：

RealBank
↓
WM
↓
Policy
↓
a1..a6


天然就能比较。

所以未来我不会只比较：

z_real

vs

z_wm


我会比较：

trajectory_real

vs

trajectory_wm


甚至：

GRPO reward_real

vs

GRPO reward_wm

最后的结论

你现在最值得关注的已经不是：

WM怎么更准


而是：

WM怎么对Policy更有用


我建议把未来的WM验收顺序改成：

Latent Fidelity
      ↓
Action Consistency
      ↓
Value Consistency
      ↓
3s Trajectory Consistency
      ↓
Joint Training


而不是：

Latent Fidelity
      ↓
Joint Training


因为对于你这种：

IL
→ DAgger
→ GRPO
→ WM


路线，

Policy 是老师，WM 是学生。

只要坚持这个原则，WM/Policy 漂移问题会比 Dreamer 系列容易控制很多。