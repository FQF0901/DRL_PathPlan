下面这版可直接交给 Coding Agent。核心原则是：

先对尚未证实的猜想做全链路归因，不再凭局部离线指标决定方向；同时以 analytic_learned / ws_learned 为研发主线，保留其纵向收益，但不能继续停留在“四个读出标量残差”，必须逐步闭合 latent、physical anchors、Memory 和多步训练。

Round-40：ws_learned 全链路归因与一致状态模型重构
一、不可变架构原则

正式链路保持：

真实状态 z0
→ 唯一 Structured PolicyHead 输出 a0
→ 唯一 World Model 推进 WorldState1
→ 同一 Structured PolicyHead 输出 a1
→ 递归得到 a0...a5
→ 固定 LQR


硬约束：

一个 Policy 动作出口； -一个 World Model 状态推进源； -六步只由同一 Policy 六次调用生成； -不得恢复 oneshot、plan_override、第二轨迹头； -显式几何读出必须属于 WorldState，不能由评测脚本旁路注入； -研发主线使用 analytic_learned，生产基线继续使用 off，直到安全门通过； -模型总参数暂放宽至 2,000,000； -暂不启用 MoE、PPO/GRPO；
DAgger/learner-state transition 待本轮状态链过门后启动。
二、先验证的关键猜想

以下结论目前尚未完全证实，必须先用 probes 和配对闭环验证。

猜想 1：ws_learned 的 collision 上升来自横纵状态不一致

当前 learned residual 只修正：

speed
static_distance
dynamic_distance
dynamic_ttc


但没有同步修正：

ego latent
OD anchors和OD latent
LD anchors和LD latent
presence/entry
横向Policy所消费的场景表示


可能形成：

纵向读出认为可以加速
OD/LD latent仍表示旧世界
横向路径未适配新速度
→ collision / off-road

验证臂
H0：off
H1：analytic_learned，仅四列残差
H2：H1 + oracle ego physical state
H3：H1 + oracle OD anchors
H4：H1 + oracle LD anchors
H5：H1 + oracle ego/OD/LD anchors
H6：H1 + oracle横向路径


分场景报告：

tollgate
ramp_in
uturn
split
roundabout
merge
普通道路


若 H3/H4/H5 明显降低 collision/off-road，则证明当前问题来自底层世界状态不一致，不是简单速度过高。

猜想 2：两套 ego-motion 推进发生漂移

当前同一 forward 中存在：

assemble_pose_k / ST-GNN图位姿链
analytic_update / WorldState物理链

必须逐步对拍

每个 horizon 保存并比较：

ego pose
OD anchor in ego frame
LD anchor in ego frame
route progress
lane lateral offset
lane heading error


统计：

max / mean / P90差异
首次发生差异的horizon
差异到Policy动作的局部增益

判定

如果两条链并非逐步一致：

将解析运动学、坐标变换和 anchor 更新下沉至唯一 net/geometry.py / net/kinematics.py，ST-GNN 与 WorldState 共用同一结果，禁止平行推进。

猜想 3：fixed 几何口径混入了 B0/B1/B2 差异

当前：

off：
旧生产几何

analytic / analytic_learned：
readout强制fixed几何


因此离线 B0、B1、B2 不是严格单变量。

四臂对照
G0：off + legacy geometry
G1：off + fixed geometry
G2：analytic + legacy-aligned geometry
G3：analytic + fixed geometry


必须分别报告：

离线h1–h6
Calib50
计划footprint
collision
off-road


后续所有 B 阶段比较必须统一几何口径。文档中“HEAD 逐位一致”只能描述 world_state_api=off 默认路径，禁止暗示 analytic 与 off 逐位一致。

猜想 4：训练与评测的 residual 特征构造可能不一致

虽然仓库已有：

readout_residual_features()


但实际训练 runner 未正式入库，不能证明训练与评测调用相同代码。

强制对拍

使用同一批 transition：

train runner features
eval runner features


逐项比较：

prev_exog
prev_present
action
base_exog
base_present
delta output


硬门：

max_abs_diff = 0
key set完全相同
normalization完全相同
mask完全相同


训练 runner、配置、数据 split 和 checkpoint 生成命令必须正式入库，不能只保存在 .slim/ 或临时脚本中。

猜想 5：当前 collision 不只是速度过高，而是高速横向精度不足
反事实速度配对

对相同 episode、相同状态和相同横向输出，运行：

实际learned速度
0.75×速度
0.50×速度
oracle横向路径 + learned速度
learned横向路径 + oracle安全速度


比较：

collision
off-road
曲率误差
曲率抖动
航向误差
横向误差
steer saturation
footprint clearance


判定：

-降速即可消除：速度规划为主因； -oracle横向路径才能消除：横向路径为主因； -二者都需要：横纵协同是主因； -仍无法消除：候选车道、静态障碍或控制接口问题。

三、优化方案
1. 将 world_residual 升级为一致状态 residual

当前四列 residual 暂时保留为对照，但新的正式候选应预测底层状态残差。

Ego residual
delta_speed
delta_acceleration
delta_yaw_rate

OD residual

至少预测：

delta_position
delta_velocity
delta_heading
presence / entry

LD residual

静态道路通常不预测运动，但可预测：

anchor重采样残差
可见性 / validity
lane-relative几何残差

统一流程
WorldState_k
+ action_k
→ analytic physical update
→ learned ego/OD/LD residual
→ WorldState_(k+1)
→ 唯一readout
→ PolicyStateView_(k+1)
→ 同一Structured Policy


dynamic_distance、TTC 和道路余量必须从最终 anchors 重新计算，不再直接预测与底层状态可能矛盾的最终标量。

四列 residual 只保留为诊断上界，用于回答“直接修读出能提升多少”。

2. 补齐完整 WorldState

WorldState 必须包含：

ego physical state
ego latent

OD anchors
OD latent
OD identity / presence / masks

LD anchors
LD latent
LD masks

route/nav state
Real Memory snapshot reference
Imagined Memory


禁止继续保留：

ego latent在WorldState之外单独递归


PolicyBackbone 和显式 readout必须消费同一 WorldState_k 的不同视图，而不是两个独立状态系统。

3. 将 transition 改为非原地 API

替换：

analytic_update(state, action)  # 原地修改


为：

next_state = transition(state, action)


要求：

-输入 state 不被修改；

next state 拥有明确 step/version； -hooks 可同时保存 before/after； -便于 truncated BPTT； -避免不同消融臂共享对象时串扰。

解析几何移动到公共基础模块：

net/geometry.py
或
net/kinematics.py


禁止 world_state.py 反向 lazy import net/model.py。

4. 完整实现双 Memory，但先不扩复杂度
Real Memory
只保存真实完整feature
FIFO
episode/scenario切换reset
禁止imagined写回

Imagined Memory
rollout开始由Real Memory编码快照初始化
每步压入predicted WorldState
FIFO
rollout结束destroy
writeback恒0


当前 ImaginedMemory 只完成了记录和隔离。下一步必须明确谁消费它：

World Model transition
或
PolicyBackbone


先做三臂：

M0：不消费Imagined Memory
M1：current full + aligned historical delta
M2：M1 + 轻量attention


M2 只有在 M1 显著优于 M0 时才允许。

Delta 规则：

ego：
当前完整feature + 历史速度/航向/动作delta

OD：
先按slot identity和ego-motion对齐
再计算position/velocity/heading delta

LD：
投影到当前ego frame
再计算geometry/heading/curvature delta

类别、ID、mask：
不做数值相减，使用当前值和transition indicator

5. Detach 分阶段消融

不直接全删。

D0

当前全部 detach。

D1

仅允许状态 residual 做 2-step truncated BPTT。

D2

状态 residual 做 3-step truncated BPTT。

D3

在 D2 基础上，允许 predicted-state Policy consistency loss 回到状态 residual，但不能回到 Policy 动作生成器。

每臂记录：

h1/h3/h6 loss
每模块grad norm
gradient cosine
update norm
显存
训练稳定性
离线多步误差
Calib50安全指标


任何 detach 必须在 Detach Ledger 中包含：

位置
作用tensor
所属stage
阻断的梯度路径
预期理由
实测梯度范围


CI 禁止未经登记新增 detach。

6. nav_progress 不再允许有效值 held

当 route length 或 projection 不可推断时：

value = 0
present = 0
source = missing


不能：

保持上一值
present仍为1


Structured Policy 必须真正消费 present mask。

四、必须新增的 Probes / Hooks
1. Unified State Hook

每步记录：

ego physical + latent
OD anchors + latent
LD anchors + latent
route state
PolicyStateView
每字段producer/source/present

2. Dual Ego-Motion Hook

同时记录：

assemble_pose_k结果
WorldState transition结果
差值
首次分歧step
动作敏感度


目标是最终删除双源，只保留公共几何实现。

3. Residual Consistency Hook

验证：

final anchors
→ deterministic readout
→ distance/TTC/margins


与 Policy 实际消费值逐位一致。

禁止：

anchor不变
但标量读出被residual独立改写


正式状态 residual 模式下应满足：

scalar residual direct-apply count = 0

4. Memory Lifecycle Hook

记录：

Real Memory reset/shift
Imagined Memory create/push/read/destroy
frame source
slot identity
writeback count
consumer count


硬断言：

writeback = 0
rollout结束destroyed = true
consumer count > 0，才可称为已启用Memory

5. Train/Eval Feature Parity Hook

在同一 transition 上保存 feature hash：

train_features_hash
eval_features_hash


要求完全一致。

6. Closed-loop Failure Attribution

对每个失败自动分类：

目标车道/通道选择错误
路径本身越界
路径本身碰撞
纵向速度错误
WorldState/PolicyStateView不一致
LQR跟踪错误
未知


重点单独统计：

新增collision
新增off-road
成功翻正
成功翻负

五、评测与验收
1. 离线单变量门

所有 B0/B1/B2 对照必须统一：

相同fixed/legacy几何口径
相同数据split
相同feature builder
相同checkpoint装配


要求：

train/eval features max_abs_diff = 0
B2优于B1
slow+accel Σds >= 12.8m
全样本累计误差 <= 2.31m
h1–h6不指数增长


若原门经审计被证明口径错误，可以修订，但必须预注册后再跑，禁止看结果后改门。

2. 状态一致性门
双ego-motion链 max error接近数值精度
Policy消费的distance/TTC可由final anchors逐位复算
WorldState包含ego/OD/LD完整latent和anchors
第2–6步无静默hold
missing字段present必须为0

3. Memory 门
Real/Imagined零写回
Imagined Memory有真实consumer
M1必须显著优于M0
真实历史必须优于repeat-nearest
打乱历史必须产生退化


否则保持 M0，不启用 attention。

4. 梯度门
D1或D2必须优于D0
h3/h6梯度到达声明的transition/residual模块
冻结模块grad=0
无梯度爆炸
12GB显存可运行

5. 参数门
总参数 <=2,000,000


但每次报告：

total
reachable
trainable
frozen
dormant


不能只报总参数。

6. 闭环门
Calib50

只作快速机制验证，不能声明收益。

研发候选必须至少满足：

success >= ws_learned当前0.56
timeout <=0.06
collision显著低于0.18
off-road显著低于0.20

Clean500

所有正式候选与以下版本配对：

C0_HEAD
C1_HEAD
ws_learned旧版
IDM + Pure Pursuit


晋级要求：

success相对C1_HEAD提升 >=5pp
collision/off-road不劣于C1_HEAD
timeout下降
route completion不下降

最终 Eval500
3 seeds
paired bootstrap
McNemar
success > IDM + Pure Pursuit
collision/off-road不劣于IDM

六、给 Coding Agent 的执行摘要
Round-40 分为“猜想验证”和“状态模型重构”两部分。

第一批先验证：
1. ws_learned collision是否由横纵状态不一致导致；
2. assemble_pose_k与analytic_update是否逐步漂移；
3. fixed几何是否污染B0/B1/B2单变量对照；
4. train/eval residual features是否逐位同源；
5. collision究竟由速度、横向路径还是二者协同导致。

验证后实施：
1. 将WorldState补齐ego/OD/LD physical+latent；
2. residual从四个最终标量升级为底层ego/OD/LD状态残差；
3. 所有距离/TTC/margin由最终anchors统一readout；
4. 删除双ego-motion推进，只保留公共geometry/kinematics实现；
5. transition返回新WorldState，禁止原地修改；
6. 完整接入Imagined Memory consumer；
7. 做M0/M1/M2 memory对照；
8. 做D0/D1/D2/D3 detach与BPTT对照；
9. nav_progress不可推断时missing，不得held；
10. 正式入库训练runner、配置、split与train/eval feature parity测试。

生产继续使用world_state_api=off；
研发基线使用analytic_learned；
未通过Clean500安全门前不得设为默认，也不得启动MoE、DAgger或RL。


最核心目标是：

保留 ws_learned 已证明的纵向闭环收益，同时消除它造成的横向状态失配和 collision，把“显式四列读出补丁”升级为真正统一、递归、可训练的 latent + physical WorldState。