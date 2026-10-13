总体裁决

我同意 docs\deepseekv4p1_argue.md 对 Round-27 的 停止判定，但不同意继续沿“加 preservation 权重、加更多 trackability loss”这条路细调。

最新结果很明确：

R1 把 steer saturation 从约 0.95 降至 0.71，说明正则确实约束了横向控制；
learner-state 误差继续改善约 72%，方向错误下降约 80%； -但 timeout 从约 0.12 升至 0.34； -teacher-state MAE 恶化约 60%； -plan Δκ 仍为 expert 的 2.1倍。

这不是单纯“loss 权重没调好”，而是在暴露更底层的问题：

当前 (ds,dθ) 动作参数化、联合动作头、监督目标和 LQR 执行器，把纵向进度与横向曲率过度耦合了。

你的怀疑是成立的，而且很可能已经成为现阶段的主 root cause。

一、横纵向在哪里发生了耦合？
1. 动作参数化本身就是耦合的

net\model.py 中，轨迹按恒曲率圆弧生成：

κ=dθds\kappa = \frac{d\theta}{ds} R=dsdθR=\frac{ds}{d\theta}

也就是说：

改 dθ 会改变路径曲率； -改 ds 也会改变路径曲率； -缩短 ds 而保持 dθ 不变，会显著增大曲率。

因此当前动作并不是：

ds = 纯纵向控制
dθ = 纯横向控制


而是：

ds 与 dθ 共同决定横向曲率；
ds 还决定纵向速度。


Round-27 中为了压低转向饱和而改变动作或计划平滑性，很容易同时损伤 ds 和 route progress，最终把 off-road 转化为 timeout。

2. LQR 执行层也把两者继续耦合

env\tracking.py 中：

reference speed 由轨迹相邻点的 chord distance 计算； -这些轨迹点同时由 ds 和 dθ 生成； -LQR lookahead 又依赖车辆当前速度； -横向控制使用预瞄位置和航向误差； -纵向控制使用同一个 preview 点的参考速度。

因此链路是：

ds,dθ
→ 圆弧几何
→ chord speed
→ longitudinal throttle
→ actual speed
→ lookahead distance
→ lateral steer


也就是说，纵向变化会改变横向预瞄，横向曲率又会轻微改变参考速度。

这是物理上合理的一定程度耦合，但当前模型与监督没有将：

必要的物理协调


和：

不必要的表示/优化纠缠


区分开。

传统 Frenet 规划通常先在路径坐标系中分解横向路径和纵向速度，再在可行性层重新联合检查；FRENETIX 就采用轨迹采样、可行性检查和多目标代价来协调安全、舒适与路径精度。

3. 网络输出头也是联合的

当前 Policy 基于同一 latent、同一 trunk 输出 2D action：

(ds, dθ)


因此：

lane semantic 进入 shared latent； -动作级 semantic loss 同时影响两维； -即使主要目标是纠偏，梯度也可能改变 ds； -正常状态 preservation 和 trackability loss 也可能通过 shared trunk 改变两维。

已有 action_dim_weights 还曾因为动作尺度不同，对 dθ 使用很大的相对权重。这有其历史原因，但也会使横向监督在共享网络中对纵向输出产生间接影响。

4. 当前监督仍在分别拟合 ds/dθ，却没有直接监督真正的物理变量

现在重点监督：

Δds
Δdθ


但车辆真正执行关心的是：

速度 / 进度
曲率
曲率变化率
轨迹可跟踪性


两个动作在 dθ 上误差相同，如果 ds 不同，其物理曲率误差可能完全不同。

例如：

动作A：(ds=5m, dθ=0.1rad) → κ=0.02
动作B：(ds=1m, dθ=0.1rad) → κ=0.10


两者 dθ 相同，但第二个是第一条路径曲率的 5 倍。

所以用 dθ residual 的 P95/P99 做软锚仍不充分。应该锚定的是：

Δκ
以及 path-conditioned progress


而非只锚定 Δdθ。

二、对 Round-27 代码修改的审阅
1. env\tracking.py 的遥测改动正确

新增 steer_raw，保留 clip 前转向指令，是正确做法。它能区分：

模型计划只是接近饱和
vs
模型计划实际远超可执行边界


但当前遥测仍缺少：

reference curvature at preview
actual curvature
ref_speed 与 ds/dθ 的分解来源
preview point 属于哪个 plan step
throttle saturation


Round-28 应补上这些字段，否则“首步、尾部、trackability”分类仍然偏经验化。

2. pipeline\residual_classes.py 的 N0/N1/C 分类方向合理

当前分类大致是：

N0：legacy 已安全且 expert residual 很小；
N1：legacy 有稳定小偏差，但未来仍安全；
C：明显 correction 状态。

这个逻辑优于将所有 teacher-state 强制 residual=0。

但有两个问题。

问题 A：分类仍在 action space 判断

N0/N1 主要根据：

|expert_ds - legacy_ds|
|expert_dθ - legacy_dθ|


而不是：

speed residual
curvature residual
path-relative progress residual


所以一个 dθ 差值很小、但 ds 很小导致曲率差很大的状态，可能被错误归类为 N0/N1。

问题 B：safe_future 还不等于 legacy policy 本身安全

未来 lat/head/speed 指标来自哪条 rollout 必须明确。如果是 expert future 或数据记录，而不是 legacy 从当前状态闭环执行的结果，那么：

状态未来安全


不等于：

legacy action 在此状态是安全的


N0 的 preservation 应依赖 legacy 行为的同状态短 horizon rollout，而不是只依赖 expert 轨迹未来安全。

3. pipeline\semantic_labels.py 的 v2 fail-fast 正确

它已经：

-加载同状态 v2 expert label； -剔除不可复现行； -发现任何旧 B-label 暴露就直接退出； -覆盖率进入元数据。

这部分可以保留。

建议再将：

label source hash
expert implementation hash
backend
query timing


写入每个正式训练 manifest，防止后续数据版本再次混淆。

三、Round-27 结果应该如何解释
“73.7% 为场景必要饱和”这个结论需要谨慎

如果“场景必要”定义为：

expert 或 legacy 在该场景也出现较大 steering


仍不代表 Q1/R1 当前的饱和是合理的。

应区分：

场景确实需要较强曲率； 2.模型走晚了，导致必须急转； 3.路径方向正确但 ds 太低，造成 κ=dθ/ds 偏大； 4.连续重规划令纠偏反复累积； 5.LQR lookahead 与速度不匹配。

因此“场景必要”最好改名为：

场景具有高横向控制需求。

不能直接从中推导模型饱和合理。

R1 修好 steer、却让 timeout 爆发，是横纵耦合的强证据

R1：

steer saturation 低于 legacy；
learner-state 误差改善更大； -却 timeout 增加 26pp； -plan Δκ 仍是 expert 两倍以上。

这表明：

通过动作/计划平滑令横向执行更温和时，纵向进度也被同步削弱了；同时真正路径曲率变化并未被解决。

所以继续加 preservation 或 throttle reward，只会在耦合空间中拔河。

四、行业方案给出的启发

完全独立的横纵规划也不是答案。并行、互不条件化的横纵头容易产生“路径想左转、速度却仍按直行方案”的协调失败。

最新的 AlignDrive 正是针对这一点：先产生 lateral drive path，再让 longitudinal prediction 条件化于该路径，并在路径上预测一维纵向位移，而不是让横纵两个头完全独立。

PLUTO 同样强调 longitudinal-lateral aware architecture，并通过显式横纵行为建模和数据增强改善闭环，且在 nuPlan 上报告超过规则规划器。

因此我们不应选择两个极端：

当前：ds/dθ 完全纠缠
或
错误方案：两者完全独立、互不通信


更合理的是：

先规划横向路径，再基于该路径规划纵向速度。

五、建议 Round-28：先证实耦合，再做最小重参数化

这轮不要继续添加 loss。分两步。

Round-28A：零训练横纵耦合审计
1. 将动作转换为物理变量

逐 plan step 计算：

κt=dθtmax⁡(dst,ϵ)\kappa_t = \frac{d\theta_t}{\max(ds_t,\epsilon)} vtarc=dst0.5sv_t^{arc} = \frac{ds_t}{0.5s}

同时计算 tracker 实际 chord speed：

vtchord=2Rtsin⁡(∣dθt∣/2)0.5sv_t^{chord} = \frac{2R_t\sin(|d\theta_t|/2)}{0.5s}

输出：

ds
dtheta
kappa
arc-speed
chord-speed
delta-kappa
actual speed
route progress
raw steer
throttle

2. 做四类反事实
F1：固定路径曲率，只改速度
ds *= scale
dtheta *= scale


使 κ=dθ/ds 近似不变。

F2：固定速度，只改曲率

保持 ds 不变，只修改 dθ。

F3：固定几何 plan，只独立修改 tracker v_ref

几何不变，用独立 speed override。

F4：保持第一步不变，只修改尾段速度/曲率

用于区分：

首步动作影响
vs
LQR preview 尾段影响

3. 关键输出
success / off-road / timeout
raw steer / steer saturation
throttle saturation
route completion
tracking error


按以下分层：

直道
弯道
tollgate
merge
成功/off-road/timeout

通过判定

若 F3 能减少 timeout 且不恶化 off-road：

主要问题在 speed reference，而非 path geometry。

若 F1 明显优于仅调 ds：

当前失败来自曲率随 ds 变化的耦合。

若 F2 改善 off-road但 timeout基本不变：

横向曲率头应独立重构。

若 F4 尾段变化显著改变当前 steer：

LQR preview/chain 是主要耦合源。

Round-28B：最小动作重参数化 canary

仅在 28A 后实施，建议改为内部输出：

longitudinal：ds 或 v
lateral：kappa


执行前转换：

dθ=κ⋅dsd\theta = \kappa \cdot ds

接口对外仍保持：

(ds,dtheta)


因此无需重写环境和 LQR，只修改 policy head 和监督口径。

六、推荐模型结构：解耦输出，条件耦合决策
1. Lateral branch

输入重点：

lane semantic
target lane token
ego lateral/heading error
road curvature
OD lateral constraints


输出：

kappa_0...kappa_5


或先只输出第一步 κ residual。

主监督：

expert kappa
delta-kappa
lane-direction consistency
mirror consistency

2. Longitudinal branch

输入：

最终 lateral path / kappa sequence
ego speed
speed limit
route progress
OD/TTC
signal


输出：

ds_0...ds_5 或 speed profile


关键是：

longitudinal branch 必须 condition on lateral path。

这样急弯时可自然减速，直道时保持进度，而不是由一个 shared residual 同时修改 ds/dθ。

3. 最终转换
dtheta_t = kappa_t * ds_t


并通过现有 arc_step 和 LQR 执行。

这比直接输出独立 ds/dθ 更符合物理语义，也更容易约束。

七、Round-28 只批准两个训练臂
K0：当前 Q1/R0 基线

同批复现。

K1：纵横重参数化 canary
legacy action → (ds_legacy, kappa_legacy)

semantic lateral residual：
delta_kappa

longitudinal residual：
第一版固定为0，即 ds_final = ds_legacy

kappa_final = kappa_legacy + delta_kappa
dtheta_final = kappa_final * ds_legacy


第一版故意只让 semantic 修正横向曲率，完全不碰 ds。

这样能直接回答：

Timeout 是否主要由 semantic 横向修正意外修改纵向进度导致？

若 K1 保留 off-road 改善并消除 timeout 恶化，横纵耦合 root cause 得到确认。

只有 K1 通过后，下一轮才增加独立 longitudinal residual。

八、K1 的监督

目标：

κexpert=dθexpertmax⁡(dsexpert,ϵ)\kappa_{expert} = \frac{d\theta_{expert}} {\max(ds_{expert},\epsilon)} Δκtarget=κexpert−κlegacy\Delta\kappa_{target} = \kappa_{expert}-\kappa_{legacy}

模型：

κfinal=κlegacy+Δκsemantic\kappa_{final} = \kappa_{legacy}+\Delta\kappa_{semantic}

损失：

L_kappa_residual
L_mirror_kappa
L_delta_kappa_smooth
L_normal-preservation，仅N0


低速和极小 ds 时必须 mask，建议：

ds < 0.5m/0.5s 不启用kappa监督


否则 κ 会数值爆炸。

不要在同一轮加入 longitudinal residual、progress loss、MoE 或 DAgger。

九、Round-28 验收标准
Gate A：零训练耦合审计
至少 90% timeout/off-road 差异可归因到 path、speed 或 preview； -明确 F1/F2/F3/F4 哪个轴主导； -输出结果可逐 episode paired。
Gate B：横向能力

K1 相对 K0：

learner-state kappa MAE下降 >=30%
mirror >=95%
lane correction sign >=95%
plan delta-kappa <=expert P95×1.25
steer saturation <=legacy+5pp

Gate C：纵向保持

由于 K1 不修改 ds：

ds输出逐位等于legacy
reference speed差仅来自几何chord效应
timeout <=legacy+3pp
route completion不下降超过2pp
teacher ds MAE逐位不变

Gate D：clean150
success >=0.43
相对legacy >=+8pp
off-road <=0.35
collision <=legacy+2pp
timeout <=legacy+3pp

Gate E：下一阶段许可

K1 全过后才允许 K2：

path-conditioned longitudinal residual / speed profile


K1 失败则不应继续加纵向头，应回到 chain/LQR preview 或 expert label 可持性问题。

十、代码具体建议
net/model.py

新增明确诊断输出：

legacy_ds
legacy_kappa
semantic_delta_kappa
final_ds
final_kappa
final_dtheta


不要在 forward() 内塞训练分类逻辑。

需要严格处理：

valid_kappa = ds.abs() >= epsilon_ds


所有 curvature loss 和指标必须带 mask。

env/tracking.py

新增：

lqr_ref_curvature
lqr_actual_curvature
lqr_preview_plan_step
throttle_raw
throttle_clipped
ref_chord_speed


目前只有 steer_raw，不足以分析完整横纵耦合。

pipeline/residual_classes.py

N0/N1/C 判断增加：

res_kappa
legacy_rollout_safe


不要继续只按 res_ds/res_dtheta 分类。

pipeline/semantic_labels.py

继续保持 v2 fail-fast，并在 label metadata 中增加：

expert_ds
expert_dtheta
expert_kappa
kappa_valid
legacy_ds
legacy_dtheta
legacy_kappa


避免训练阶段反复、可能不一致地重算。

十一、批准与冻结
批准
Round-28A 零训练横纵耦合审计
tracking遥测补齐
N0/N1/C curvature-aware审计
Round-28B K0/K1 两臂
仅 lateral kappa residual

暂不批准
继续调R1 preservation权重
全局progress loss
独立longitudinal head训练
正式DAgger
MoE
PPO/GRPO
大规模采集
WM修改
K-anchor
容量扫描

最终判断

你的疑问是关键的：

当前 root cause 很可能包含显著的横纵向耦合，而且不仅在网络层，也同时存在于动作定义、监督空间、chain rollout 和 LQR preview 中。

但不应简单改成两个完全独立的 head。行业趋势更加支持：

先确定 lateral path / curvature
→ 再让 longitudinal speed 条件化于该 path


AlignDrive 明确指出，平行且互不条件化的横纵预测会产生协调失败，因此采用 path-conditioned longitudinal planning；PLUTO 也通过显式 longitudinal-lateral aware 架构提升了闭环规划性能。

下一步最干净的实验是：

固定 legacy ds，让 semantic 只学习 Δκ。

如果这样既保留 off-road 改善，又消除 timeout 恶化，就基本确认当前 (ds,dθ) residual 的横纵纠缠是核心瓶颈。