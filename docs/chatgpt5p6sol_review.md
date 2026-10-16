你们现在已经把问题从“低速爬行”推进到了更本质的层面：状态链尚未完全闭合，同时 Structured Policy 的横向动作幅值、帧间一致性和固定 LQR 可执行性成为当前直接性能瓶颈。 下一轮不能只训练横向头，也不能立即重构整个 World Model，应先补齐可复现证据，再用严格单变量实验决定训练范围。

Round-41：闭环可复现归因 + 横向时序训练 + 一致 WorldState
一、阶段目标

保留当前最有价值的研发候选：

world_state_api = analytic_learned


但生产默认仍为：

world_state_api = off


本轮目标：

1.正式复现并验证当前未经入库的闭环归因实验； 2.保留 analytic_learned 降低 timeout 的能力； 3.显著降低 collision 和 off-road； 4.闭合 ego、OD、LD、lane anchor、latent 和 Policy 的真实因果链； 5.使所有训练、评测和协议在干净 clone 中可复现。

最终架构仍是：

真实WorldState0
→ 唯一Structured Policy输出a0
→ 唯一World Model预测WorldState1
→ 同一Policy输出a1
→ 递归六次
→ 固定LQR

二、第一阶段：先修证据链，不训练新主模型
1. 正式实现闭环归因实验

将目前只存在于未入库脚手架中的功能正式接入：

--teacher-force
--inject-ego-truth
--inject-od-truth
--inject-ld-truth
--inject-lateral-truth
--speed-scale
--speed-oracle
--record-truth


必须修改并入库：

pipeline/eval_runner.py
net/model.py
tools/test.py
对应配置和测试


每个开关必须：

-有正式 CLI； -有模型实际执行路径； -进入 evaluation protocol hash； -进入模型缓存 key； -写入 metrics 和 manifest； -默认关闭时逐位一致。

当前仅有 hash 参数占位的代码不得称为“已接线”。

2. 将 residual 训练链完全入库

移除正式 runner 对未跟踪 .slim/ 脚本的硬依赖。

正式模块至少拆成：

pipeline/residual_dataset.py
pipeline/residual_features.py
tools/train_residual.py
config/residual_train.yaml


必须涵盖：

数据加载
连续transition配对
episode级split
obs构建
模型装载
feature构建
训练
validation
checkpoint保存
账本输出


使用一个小型入库 fixture 做 CI，禁止关键一致性测试因 .slim 缺失而 skip。

3. 清理实验协议

将几何模式改为模型实例配置：

readout_lane_geometry:
  legacy_aligned
  corrected


删除或废弃进程级全局模式与环境变量作为正式入口的做法。

该字段必须进入：

model config
checkpoint metadata
protocol hash
model cache key
metrics


参数预算只保留：

total_parameters <= 2,000,000


同时报告：

reachable
trainable
frozen
dormant


删除旧 TOTAL_LIMIT=1.02 正式语义；若保留兼容，迁入 tools/legacy/。active +5% 若继续作为硬门，必须单独获得批准，否则降级成警告。

三、第二阶段：严格定位横向瓶颈
1. 几何训练 A/B

不能继续 zero-shot 比较冻结 Policy。

训练两套等预算模型：

Policy-L：
legacy_aligned几何训练
legacy_aligned几何评测

Policy-C：
corrected几何训练
corrected几何评测


再做交叉测试：

Policy-L → corrected
Policy-C → legacy_aligned


由此拆分：

几何定义收益
Policy适应收益
几何切换分布偏移

2. 横向动作与时序监督

在同一个 Structured Policy 中训练：

当前κ_near
κ_mid / κ_far辅助意图
横向偏移
航向误差
连续footprint道路余量
固定LQR跟踪可执行性


增加两类一致性监督。

计划内部平滑
LΔκ=∑k∣κk+1−κk∣L_{\Delta \kappa} = \sum_k \left| \kappa_{k+1}-\kappa_k \right|
跨重规划帧一致性

将上一帧计划按真实 ego 位姿变化重投影到当前帧：

Lreplan=∥κt(s)−Warp⁡(κt−1(s))∥L_{\text{replan}} = \left\| \kappa_t(s) - \operatorname{Warp} \left( \kappa_{t-1}(s) \right) \right\|

禁止未经坐标对齐直接比较相邻帧输出。

3. 速度条件化横向能力

横向分支必须读取：

当前真实或imagined速度
当前计划进度
道路曲率
道路边界


加入物理约束：

ay=v2κa_y=v^2\kappa

重点约束：

高速曲率幅值
高速曲率变化率
方向翻转率
预测steer saturation
连续footprint clearance


当前证据显示横向命令过量会让固定 LQR 长期处于高转向状态，车辆纵向能力被横向执行显著消耗。横向和纵向必须联合验收，不能分别把离线 loss 降低视为成功。

4. 固定 LQR 可执行性辅助监督

训练时冻结 LQR，但离线展开固定控制器，计算：

预计steer
steer saturation
横向跟踪误差
航向跟踪误差
路径曲率抖动
速度损失


作为：

辅助loss
或batch重加权指标


不允许 Policy 输出几何上正确、但固定 LQR 无法执行的计划。

行业方案中，SparseDrive 通过稀疏实例表示联合建模对象、地图与自车，并引入碰撞感知的规划重评分，说明安全规划需要将对象状态、地图状态与规划评价置于一致场景表示中，而不是仅优化独立轨迹误差。

四、第三阶段：闭合统一 WorldState

此阶段与横向训练并行推进，但验收重点是状态一致性，不预设其一定直接改善安全指标。

1. 单一运动学来源

新增公共基础模块：

net/kinematics.py
net/geometry.py


统一提供：

ego motion
OD ego-frame transform
LD ego-frame transform
lane-anchor transform
footprint geometry


以下模块共同调用：

ST-GNN
WorldState
Policy readout
diagnostics


删除两套独立 ego-motion 实现，但保留旧行为 fixture，确保迁移前后数值一致。

2. 完整 WorldState

统一状态必须包含：

ego physical state + ego latent

OD anchors + OD latent
OD identity + presence + validity

LD anchors + LD latent
LD validity

lane/path anchors
route/navigation state

Real Memory snapshot
Imagined Memory


PolicyBackbone、ST-GNN 和显式 readout均从同一个 WorldState_k 取得不同视图。

3. 底层状态 residual

当前四列 residual 保留为诊断对照，正式候选改为预测：

ego speed / acceleration / yaw-rate residual

OD position / velocity / heading residual
OD presence / entry residual

LD geometry / validity residual


流程：

analytic state update
→ learned physical/latent residual
→ final WorldState
→ deterministic readout
→ distance/TTC/margins


正式候选禁止：

anchors保持不变
但dynamic distance / TTC被独立改写


DreamerV3 的关键不是预测若干策略输入标量，而是维护一个动作条件化的递归模型状态，并让策略在想象状态轨迹上学习。它也将动力学学习和表征学习通过不同方向的 stop-gradient 分离，而不是让读出标量单独形成与底层状态不一致的递归回路。

4. 非原地 transition

将：

analytic_update(state, action)


改成：

next_state = transition(state, action)


要求：

-不修改输入状态；

next state具有明确 step、source 和 version； -hooks能同时保存 before/after； -支持后续有限步反向传播； -不同实验臂不得共享可变状态对象。
五、第四阶段：Memory 与 detach 验证
1. Memory 消费验证

当前 ImaginedMemory 仅完成生命周期隔离，还不算真正生效。

做三臂：

M0：
不消费Imagined Memory

M1：
current full feature
+ aligned historical deltas

M2：
M1
+ 轻量Temporal Attention


只有 M1 显著优于 M0，才允许 M2。

对齐规则：

OD：
slot identity对齐
+ ego-motion对齐
+ position/velocity/heading delta

LD：
投影到当前ego frame
+ geometry/heading/curvature delta

类别、ID、mask：
当前完整值
+ transition indicator
禁止数值相减


类似 SparseDrive 的实例记忆队列强调实例身份、锚点与时间传播的一致性，而不是简单堆叠未对齐历史特征。

2. Detach 消融
D0：当前one-step detach

D1：
状态residual 2-step truncated BPTT

D2：
状态residual 3-step truncated BPTT

D3：
D2 + WM-state Policy consistency
梯度只回状态模型，不回动作生成器


记录：

h1/h3/h6 loss
每模块grad norm
gradient cosine
update norm
显存
训练时间
多步误差
闭环安全指标


不直接开启六步全反向传播。

六、后置 learner-state 数据

本轮状态和 Policy 基础门通过后，再启动新 DAgger 接口：

Policy样本：
learner state → expert structured action

World Model样本：
learner state + executed learner action
→ real next state


两者共用采集，但不同标签、不同 optimizer、交替冻结。

DAgger 的核心价值正是在学习器实际访问的状态上查询专家并聚合数据，以缓解行为克隆的状态分布偏移；它不应被用来替代当前状态接口和横向执行契约的修复。

七、必须新增的 Probes / Hooks
1. 正式 Oracle Injection Hook

记录：

注入模块
注入步数
真实来源
动作条件是否一致
替换前后state hash
替换前后Policy输出


自动拒绝：

学生action
+
专家action产生的future state


这种非法反事实组合。

2. State Consumer Graph

自动记录每个动作输出对以下输入的局部敏感度：

ego physical / latent
OD anchors / latent
LD anchors / latent
lane anchor
route state
Memory
readout fields


输出：

gradient norm
finite-difference sensitivity
遮挡后的动作变化


用于确认 LD、OD 和 Memory 是否有真实消费者。

3. Residual Consistency Hook

检查：

final WorldState
→ deterministic readout
→ Policy实际消费值


要求逐位一致。

正式底层 residual 模式：

direct scalar residual count = 0

4. Curvature Stability Hook

每 episode 记录：

κ幅值
Δκ
帧间符号翻转率
速度分桶下κ分布
预计横向加速度
steer saturation
计划与实际航向误差

5. Train/Eval/Horizon Parity

不能只检查第一步。

逐 horizon 记录：

训练输入分布
评测h1...h6输入分布
residual输出分布
Mahalanobis distance
P50/P90/max偏移


重点验证第 2–5 步 imagined exposure shift。

6. Memory Lifecycle Hook

断言：

Real Memory只接收真实帧
Imagined Memory每次rollout新建
writeback = 0
rollout结束destroy
consumer count > 0


consumer count=0 时不得宣称 Memory 已实现。

八、验收准则
工程可复现门
干净clone可运行residual训练
核心测试不得因.slim缺失而skip
所有oracle/速度干预有正式CLI与模型接线
几何模式进入protocol hash
训练/评测第一步同 batching max diff=0
在线batch=1差异有明确上界

状态一致性门
唯一运动学来源
WorldState包含ego/OD/LD physical+latent
Policy消费值可由final WorldState逐位复算
LD/OD/Memory均有可测动作敏感度
第2–6步无静默hold
nav不可推断时present=0

横向门

相对当前 ws_learned：

帧间曲率符号翻转率下降 >=50%
steer saturation下降 >=40%
高速横向误差不恶化
连续footprint越界下降
collision显著低于0.18
off-road显著低于0.20

Calib50 快筛

候选至少：

success >=0.56
timeout <=0.06
collision <=0.08
off-road <=0.18
route completion >=0.815


Calib50 仅用于筛选，不得作为最终收益声明。

Clean500 晋级

与同协议版本配对：

C0_HEAD
C1_HEAD
旧ws_learned
IDM + Pure Pursuit


要求：

success相对C1_HEAD提升 >=5pp
collision不劣于C1_HEAD
off-road不劣于C1_HEAD
timeout下降
route completion不下降

最终目标
Eval500
3 seeds
paired bootstrap
McNemar

success > IDM + Pure Pursuit
collision/off-road不劣于IDM + Pure Pursuit

给 Coding Agent 的开工摘要
Round-41 不接受当前“V4B决定性否证”结论，先补正式实现和可复现证据。

第一批：
1. 正式入库并接线oracle/teacher-force/speed反事实；
2. residual训练链移除.slim硬依赖；
3. 几何模式改为模型实例配置并进入hash；
4. 清理旧参数预算常量和测试语义；
5. parity扩展到h1...h6分布，不只检查第一步。

性能主线：
6. 做legacy训练/评测与corrected训练/评测的几何A/B；
7. 训练横向@速度能力、帧间曲率一致性和固定LQR可执行性；
8. 保留analytic_learned纵向收益，但禁止继续扩大scalar residual剂量。

架构并行线：
9. 统一ST-GNN与WorldState运动学来源；
10. 补齐ego/OD/LD physical+latent WorldState；
11. residual作用于底层状态，距离/TTC由最终anchors统一readout；
12. transition改为非原地；
13. 验证Imagined Memory真实消费者；
14. 再做2/3-step BPTT。

未通过Clean500安全门前：
不得设analytic_learned为默认，
不得启动MoE、DAgger、PPO或GRPO。


核心判断是：

当前最直接的提分方向是修复 Structured Policy 的横向动作幅值、帧间一致性和固定 LQR 可执行性；但只有同步闭合统一 WorldState、底层 residual 和真实状态消费者，才能避免再次用新的 shortcut 换取局部 success，最终才有可能稳定超过 IDM + Pure Pursuit。