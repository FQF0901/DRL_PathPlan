我审阅了附件中的 P0 结果和 P1 v0.2。总体判断：P0 做得扎实，P1 方向基本正确，但当前方案仍不足以支撑“达到 0.80–0.85”。最重要的问题不是步骤顺序，而是目前尚未证明存在一条超过 IDM 的可执行教师或控制链。

一、P0 结论基本认可

以下判断有充分数据支持：

repeat_action 不适合作为部署口径。Arm P、pri512 均为 0，s11 也是 plan 0.646 高于 repeat 0.418，因此继续保持 PlanHead + LQR 评测协议是正确的。
tracker 暂时不是第一瓶颈。模型 plan 在 exact 下不优于 LQR，甚至碰撞更多，说明精准执行坏 plan 不会改善结果。
当前主要问题确实在 plan/action 质量。pri512 只有 0.332，而相同执行栈下 D2 达到 0.706。
off-road/压线是模型的 dominant failure，且 plan 不可行通常在中后期出现，支持“随闭环推进逐渐漂移”的判断。
fail-fast、laneplan bug 修复、s11 兼容评测都是必要且有效的工程改进。
二、必须修正的核心结论
1. G1 实际上没有满足

当前最高可执行链：

IDM：0.756
D2：0.706
Oracle：0.608
laneplan：0.442

没有任何已验证的 plan/control 链超过 IDM，更没有达到 0.80。附件将“主要损失层已明确”视为 G1 满足，可以作为诊断 Gate，但不能作为“超越 IDM 的路径已证明”。

应该拆成两个 Gate：

G1-diagnosis：主要损失定位完成，已满足。
G1-ceiling：存在 success ≥0.80 的可执行 teacher/control 链，目前未满足。

这意味着即使模型完美模仿 D2，理论上目前也只有约 0.706。想达到 0.80–0.85，必须额外解决：

D2 的 timeout/配速；
专家 recovery 和交互能力；
或依靠闭环 RL 超过教师。

因此，“超越 IDM”不能只依赖 recovery BC 和几何监督。

2. D2 不能同时作为训练目标和最终上限

D2 的优势来自每 0.5 秒重新查询当前状态下的专家动作，具备闭环纠偏；其失败又主要是 timeout。

D2 更适合作为：

recovery teacher；
在线行为上限；
闭环重规划参考。

但如果最终模型仍输出固定 3 秒 plan，而专家每 0.5 秒重查询，两者并不等价。训练时必须强调 receding-horizon 的首段正确性，而不能把整条未来动作链当成绝对真值。D1 全崩已经很好地证明了开环专家链不可靠。

三、P1 v0.2 中需要调整的地方
1. 起点模型不能立即在 pri512 和 rou64 中拍脑袋选择

rou64 目前只有 clean500 0.370，pri512 有 eval500 0.332 和 clean500 0.334。两者还缺同一 eval500、同一逐 episode paired 口径。

应先补：

rou64@e015 on eval500
rou64@e015 vs pri512@e020 paired comparison


若 rou64 在 eval500 也稳定领先，P1 用 rou64；否则用证据更完整的 pri512。

不要再训练组合臂，pri512+rou64 已证明存在负交互。

2. Recovery 数据优先是对的，但训练范围不能继续 specific-only

现有 phase3 的问题不只是缺真实 traj6，还包括 specific-only 主要修 experts，而 dominant error 是共享 plan 逐渐漂移。仅训练 MoE residual 很可能继续让 primary 保持错误几何趋势。

建议 recovery BC 的可训练范围为：

训练 PlanHead fusion/trunk；
训练 primary 最后一层或整个 primary，低学习率；
训练 experts/residual；
暂时冻结 encoder、memory、WM、router。

同时保留原 BC 数据混合和 anchor，防止 recovery 数据把正常路况能力冲掉。初始 recovery 样本比例建议从 batch 的 20%–30% 开始，而不是让失败数据主导全部训练。

3. 整段监督要强调“首段 + 可重规划”，不能平权六步

D1 表明远期专家 action chain 对实际闭环误差非常脆弱。因此六步 loss 不应等权。

建议时间权重类似：

step 1: 1.00
step 2: 0.80
step 3: 0.60
step 4: 0.40
step 5: 0.25
step 6: 0.15


重点监督：

第一步 (ds,dθ)；
前 1–1.5 秒 pose；
横向位置与航向；
曲率连续性；
后段作为 soft target，而不是硬性复制专家链。
4. 几何监督必须区分“指标/奖励”和“可微 loss”

MetaDrive 的地图查询、footprint inside/outside 和实线判定通常是离散几何操作，不能直接对网络 plan 反向传播。

因此 P1-3 必须明确分为：

可立即使用
corridor violation 作为 hard mining 指标；
rollout reward；
dataset sample weight；
验收指标。
要作为可微训练 loss

必须先生成可微 target，例如：

每个 plan step 的左右 corridor margin 标签；
最近边界点及法向； -局部 corridor polyline；
signed-distance 近似标签。

否则“footprint loss”只是名字，实际梯度无法进入 PlanHead。

第一版建议采用离线标签：

left_margin[t]
right_margin[t]
legal_line_cross[t]
route_deviation[t]
valid_mask[t]


训练时用预测 pose 与局部边界法向构造可微 hinge loss。不要在训练 forward 中直接调用引擎地图查询。

5. 曲率 loss 不能惩罚正确急弯

建议对：

κt≈Δθtmax⁡(Δst,ϵ)\kappa_t \approx \frac{\Delta\theta_t}{\max(\Delta s_t,\epsilon)}

做以下处理：

低速、极小 ds 使用 mask；
惩罚超过车辆可达阈值的曲率；
主要约束 Δκ，而非一味压低 κ；
tollgate、roundabout 使用道路曲率条件化阈值。

否则很容易把 off-road 降下来，却制造 timeout 和转弯不足。附件已经发现多个链路存在明显配速问题，这个风险是真实的。

四、P1-4 选型建议

我不建议第一版采用 12D 独立高斯，coding agent 的反驳成立。

在现有代码中，PPO 的随机变量和 log-prob 基于 2D action，直接扩成 12D 会同时改变：

PolicyHead 输出；
buffer action shape；
old/new log-prob；
entropy；
KL anchor；
checkpoint；
plan first-step contract；
PPO 测试体系。

风险太高。

推荐第一版：2D action-conditioned plan

保持现有 PPO action 和 log-prob 语义：

sampled action z ∈ R²
        ↓
强制成为 plan 第一步
        ↓
PlanHead 根据 latent + z 生成后续 5 步
        ↓
整个 6 步 plan 交给 LQR


优势：

与当前 2D PPO buffer、KL、entropy 最兼容；
被采样变量确实影响最终执行 plan；
信用链明确；
可以从 BC 首步 action 初始化；
比新增 2–4D latent decoder 少一层不可辨识性。

低维噪声解码器可以作为第二个候选，但必须证明噪声的每个维度稳定影响计划，否则 PPO 会在一个退化 latent action space 中优化。

五、建议锁定的 P1 顺序
P1-0：补两个关键基线
rou64@e015 跑 eval500，与 pri512 paired；
D2 做配速 oracle： -适度提高 ds 或参考速度； -保持 dθ/ds 和重规划逻辑； -观察 timeout 能否转 success，且 off-road/collision 不恶化。

如果 D2 经配速后仍不能超过 0.80，应承认当前 teacher ceiling 不足，不能仅靠监督学习完成最终目标。

P1-1：真实 recovery 数据

保存：

student observation/history；
student plan； -实际执行轨迹； -专家从 student 当前真实状态重规划的动作； -前 1–1.5 秒 recovery pose； -边界和实线标签； -速度、route progress、road class； -T_plan/T_cross。
P1-2：Recovery trajectory BC
不再使用合成 traj6；
时间衰减监督； -训练 PlanHead shared/fusion + primary 后段 + experts；
encoder/memory/WM/router 暂冻；
recovery 与普通 BC 混合； -监控正常样本退化。
P1-3：几何、曲率与配速联合 recipe

加入：

legal-line crossing；
footprint margin；
route deviation；
Δκ； -速度/进度下限；
timeout-aware progress loss。

必须同时监控 off-road、collision、timeout，防止失败类型互换。

P1-4：action-conditioned plan PPO

只有在监督阶段显著超过当前 0.37，最好达到 0.60–0.65 后再做。否则 PPO 很可能在质量过低的 plan manifold 上放大错误。

P1-5：验收

开发期：

clean150，单 seed；
同时看 success/off-road/collision/timeout。

候选期：

3 seeds；
eval500 + clean500；
固定 checkpoint selection；
paired McNemar/bootstrap。

最终声明 0.80–0.85 时，建议补 5 seeds。

六、我建议记录的当前分歧

目前只剩三个实质待决：

教师上限是否足够
 我认为尚未证明。D2=0.706 仍低于 IDM，必须先做配速修复或承认需要 RL 超教师。

Recovery BC 可训练范围
 我不支持继续 specific-only。至少应开放 PlanHead shared/fusion 与 primary 后段，否则无法修正共享几何漂移。

PPO 轨迹动作参数化
 我建议先用 2D action-conditioned plan；coding agent 推荐的 2–4D低维噪声解码器作为第二候选；12D 暂不做。

最终审阅结论

P0 可以定稿，主根因已经从“容量问题”明确收敛到“计划质量、闭环漂移、几何余量和配速”。但 P1 要避免一个新的误区：

不能因为已经定位 plan-side，就默认 recovery 数据加几何 loss 足以达到 0.80。

目前已验证的最高可执行链只有 0.706。正确策略是：

先把 D2 配速后的真正上限跑出来；
用真实 recovery 修正 plan 漂移； 3.让几何标签真正可微或用于风险采样；
用 action-conditioned plan 建立合法 PPO 信用链；
再依靠 RL 尝试超过教师和 IDM。

这条路线比直接投入 12D trajectory PPO 或继续调 WM/MoE，更符合当前代码结构和 P0 实证。

===============================================================================
基于 DRL_PathPlan_code.txt 中的网络、训练、评测、环境和测试实现，以及已经暴露过的 checkpoint 静默不匹配、训练/评测执行口径差、合成 traj6、TemporalAttention 全无效输入等问题，我建议增加一套**“契约探针 + 在线打桩 + 反事实探针”**。目标不是堆日志，而是尽早发现“代码正常运行，但语义已经错了”的隐性问题。

一、P0：必须先加的正确性契约
1. Checkpoint 完整加载探针

位置： pipeline/eval_runner.py、训练 resume 路径。

正式评测必须 fail-fast：

assert not missing_keys
assert not unexpected_keys
assert not shape_mismatch
assert obs_fingerprint == checkpoint_obs_fingerprint
assert model_snapshot_hash == resolved_model_hash


另外记录：

实际加载参数数 / checkpoint 参数数；
加载参数字节数比例；
每个顶层模块加载比例；
checkpoint SHA256；
模型配置与观测 schema diff。

仅判断 missing_keys == 0 仍不够，因为错误的过滤、重命名或遗漏模块也可能“看起来加载成功”。建议要求：

loaded_parameter_ratio == 1.0
loaded_parameter_bytes_ratio == 1.0


这要覆盖 s11 的 v7/v5 兼容路径，以及当前 v8 路径。仓库已有 checkpoint resume、dataset contract 和 obs consistency 测试，可在这些测试上扩展。

2. Plan/action 语义契约

位置： net/model.py、pipeline/eval_runner.py、pipeline/trainer.py。

每次正式评测随机抽样 1% step，断言：

plan[0] == policy_mu 或当前协议规定的首步动作
arc_step(plan actions) == exported plan poses
repeat_action 每步确实重复同一 action
单位、裁剪、dt 在 train/eval/forensics 完全一致


重点记录四组差异：

policy_mu - plan_action[0]
sampled_action - plan_action[0]
plan_action[0] - executed_reference[0]
executed_reference[0] - actual_motion


推荐字段：

action_plan_l2
action_exec_l2
expected_ds_vs_actual_distance
expected_dtheta_vs_actual_yaw_delta


阈值建议：

plan[0] 契约误差 <1e-6；
arc_step 单元测试误差 <1e-5；
如发生 clipping，必须显式记录，不能静默修改。

这个 probe 能直接防止 action-conditioned plan 改造后出现“PPO action 看似进入 plan，实际又被后续逻辑覆盖”的隐患。

3. 时间尺度和索引契约

当前 planner 是 0.5 秒、物理执行 0.1 秒、六步 3 秒，非常容易出现一帧偏移或 stride 错误。

位置： 数据采集、pipeline/stages.py、pipeline/trainer.py、环境 step。

为每个训练样本记录：

obs_timestamp
action_timestamp[0:6]
future_frame_timestamp[0:6]
physics_step_index
policy_step_index
wm_target_step_index


自动断言：

future[t] - obs_time == (t + 1) * 0.5s
每个 policy action 实际执行 5 个 physics steps
WM stride 与 BC action chain 的时间点一致


必须增加一个“恒速恒航向合成 episode”测试：

设置 ds=5m, dθ=0；
六步后应前进约 30m；
future target、plan pose、实际 reference 三者索引必须一致。

很多 world-model 看似预测不准，实际可能是 future target 错一帧。

二、P0：闭环执行链探针
4. Plan、reference 和实际轨迹三轨并记

位置： tools/diagnostics/forensics_closed_loop.py。

每个策略步同时保存：

plan_pose[6]
tracker_reference_10hz
actual_pose_10hz
policy_action
plan_action[6]
LQR control command
applied simulator control


派生三种误差：

plan prediction error
tracker reference interpolation error
vehicle tracking error


特别要区分：

网络 plan 已错误；
plan 正确但轨迹插值错误；
reference 正确但 LQR 错；
LQR 输出正确但 simulator 实际控制被限幅。

后两项目前很容易一起被归为 tracker-out。

5. LQR 饱和和不可达性 hook

位置： LQR 控制器调用点和环境 action adapter。

每个 0.1 秒物理 step 记录：

steer_raw / steer_clipped
throttle_raw / throttle_clipped
brake_raw / brake_clipped
lateral_error
heading_error
speed_error
curvature_reference
curvature_actual


关键指标：

steer_saturation_rate
longitudinal_saturation_rate
max_lateral_error
tracking_error_auc
delay_to_divergence


建议告警阈值：

连续 3 个物理 step 转向饱和；
单策略周期内横向误差增加超过 0.5m；
reference 曲率超过车辆实测可达曲率；
plan 节点在路内，但插值 reference 越界。

这能识别“不是 LQR 参数不佳，而是计划根本不可跟踪”的情况。

6. Progress/timeout probe

P0 已显示 D2 有明显 timeout 风险，因此要把闭环失败分解为安全和效率两个轴。

每个策略 step 记录：

route_progress_delta
distance_travelled
ds_command
actual_speed
speed_limit
remaining_route_distance
stationary_steps


告警场景：

ds_command 长期偏小
有安全空间但速度持续低于 speed_limit 的 40%
route_progress 连续 10 个策略步接近 0
临近终局时剩余路线很短但仍 timeout


必须区分：

真正停滞；
安全减速；
route completion 计算异常；
计划 ds 系统性低估；
LQR 纵向跟踪不足。

否则 geometry loss 可能以降低越界为代价，让 timeout 大幅增加。

三、P0/P1：几何边界探针
7. 完整 footprint 插值检查

位置： forensics、DAgger collector，之后扩展至标签生成。

不能只检查 6 个 plan pose。对每两个节点之间：

空间间隔建议不超过 0.5m；
航向变化建议不超过 2°–3°；
为每个插值 pose 生成车辆 footprint；
检查四角、边中点和车体中心；
分别记录 drivable-area、实线、route corridor 判定。

输出：

min_boundary_margin
first_invalid_segment
invalid_footprint_ratio
solid_line_cross_count
plan_inside_but_interpolation_outside


需要特别检查：

road_edge_distance_from_ctx 是否只适用于当前 ego pose。

如果它内部读取当前车辆或 reward context，而不是接受任意查询 pose，就不能直接用于未来 plan。应抽取纯函数式 geometry query，避免测试未来 pose 时实际上仍在查询当前 ego。

8. 坐标系 round-trip probe

位置： env/obs/memory.py、pipeline/frames.py、nav/world-route 重建、plan 转换。

建立 round-trip 测试：

world → ego frame → world
history frame → current frame → original history frame
route_world → nav checkpoint → reconstructed world checkpoint


测试对象：

点；
航向； -速度向量； -完整六点 plan； -车辆 footprint。

误差阈值建议：

position < 1e-4 m
angle < 1e-5 rad


同时覆盖：

yaw 接近 ±π；
reset；
原地低速；
倒车或负 ds，如果接口允许；
route 末端重复点；
history 部分无效。

复杂几何上的 off-road 很可能由微小坐标语义错误放大，不能只假设 SE(2) 实现正确。

9. 道路语义和实线 crossing 一致性检查

同一 pose 分别调用：

reward 判定；
termination 判定； -新 footprint inspector；
route/lane context 判定。

建立一致性矩阵：

reward says off-road
termination says off-road
geometry probe says outside
solid-line probe says crossed


如果这些组件对同一状态判断不同，应输出 anomaly，而不能训练网络去拟合冲突标签。

建议随机抽取至少 10,000 个实际 pose 和 near-boundary perturbation pose，统计 disagreement rate。正式使用 geometry label 前，目标应低于 0.5%；若更高，先解决定义差异。

四、网络内部 probes
10. Encoder 可辨识性 probe

位置： encoder 输出、当前帧 latent、memory latent。

冻结模型后训练小型线性 probe，预测：

ego speed
lateral offset
heading error
road curvature
left/right boundary margin
route progress
next-step ds/dtheta
OD relative position/velocity


用途不是提升模型，而是判断表示中是否存在规划所需信息。

判定方式：

当前帧 latent；
temporal memory 后 latent；
WM rollout latent； -不同 Stage A/B/C checkpoint。

如果 Stage A latent loss 下降，但 boundary margin、curvature、future pose 的 probe 不改善，说明 latent consistency 可能发生共适应，而非学到闭环物理。

11. 历史信息因果 probe

利用现有模型做不重训反事实：

normal history
current frame repeated 6 times
history reversed
history shuffled
history all-zero but valid
history valid values but hist_valid=0


比较：

policy_mu delta
plan delta
value delta
router delta
world rollout delta
closed-loop KPI


预期：

正常与复制当前帧若几乎一样，mem bank 没有使用动态历史；
hist_valid=0 后仍显著影响输出，说明 mask 泄漏；
reversed/shuffled 几乎无变化，说明模型没有利用时序顺序； -空历史导致非零变化时，重点检查 TemporalAttention bias。

代码已有 test_mem_rollout.py、test_v6_attn_heads.py 等入口，适合扩展为 contract test。

12. World Model action-sensitivity probe

当前 WM 每步 latent detach，因此必须确认它至少对 action 有敏感且合理的响应。

对同一 observation 输入不同动作：

nominal
ds +10%
ds -10%
dtheta +小扰动
dtheta -小扰动
左右镜像动作
zero action


记录：

ego latent delta
predicted ego pose delta
OD/LD latent delta
policy output delta
value delta


关键异常：

不同 action 得到几乎相同 rollout：WM 忽略 action；
左右转扰动产生同方向响应：符号或坐标错误；
第一步敏感、后续迅速完全相同：detach 或 transition 信息丢失；
微小 action 引发巨大 latent jump：transition 不稳定。

建议计算有限差分 Jacobian：

∂future_pose / ∂ds
∂future_pose / ∂dtheta
∂policy / ∂WM_latent


无需反向传播，有限差分即可诊断。

13. World Model rollout 漂移 probe

分别计算 horizon 1–6 的：

ego pose error
ego latent cosine/L2
OD position error
LD geometry error
presence calibration


不要只报告六步平均值。重点看误差增长规律：

近似线性；
指数增长； -第 2 步突然跳变； -只有 latent 误差低但物理误差高。

另外按以下维度分组：

直道/弯道/tollgate； -速度区间； -边界 margin； -对象密度；
action curvature。

如果 WM 在简单场景很好、收费站完全失效，平均 loss 会掩盖真正问题。

14. MoE 路由有效性 probe

位置： net/moe.py、PlanHead 输出。

每个 step 记录：

router logits
top-2 experts
top-2 weights
router entropy
primary norm
per-expert residual norm
mixed residual norm
residual / primary norm
expert switching rate


重点检查：

专家是否真的有影响

暂时屏蔽每个 expert，测 plan 变化和闭环 KPI。

专家是否只是重复

计算 expert residual 的 pairwise cosine similarity。

路由是否稳定

同一 episode 相邻策略步的 top-1 切换率；对输入加微小噪声后的切换率。

负载均衡是否伪装成 specialization

比较 expert ID 与 road class、曲率、速度、边界风险之间的互信息。

建议警戒值：

residual/primary norm 长期 <1%：MoE 几乎没作用；
多数专家 cosine >0.9：高度重复；
微小扰动 top-1 切换率 >30%：路由不稳定；
occupancy 均衡但专家消融无影响：只是满足 balance loss。
15. Policy std、clipping 与有效探索 probe

PPO 不能只看配置中的 log_std。应记录：

raw sampled action
squashed action
clipped action
executed action
policy_mu
policy_std
clip fraction
effective action variance


特别比较：

网络采样方差；
clipping 后方差；
LQR 执行后的实际轨迹方差。

如果大量 action 被 clipping，PPO 的 log-prob 对应的是 raw action，而环境实际接收的是另一个分布，可能形成隐蔽的 policy-gradient 偏差。

建议告警：

任一 action 维 clipping >5%
实际执行方差 < raw 方差的 20%
std 很大但轨迹几乎不变


仓库已有 test_policy_logstd_max.py、test_stage_c_effective_action.py，应扩展为分布级测试，而非只验证单个上限值。

16. Value head 校准 probe

按 value prediction 分桶，比较：

predicted value
empirical discounted return
success rate
off-road rate
timeout rate


额外输出：

explained variance；
calibration curve；
success/off-road 二分类 AUC；
按 road class 的 bias；
terminal 前 1–5 步 value drop。

若 off-road 前 value 仍很高，critic 没有识别 dominant risk，PPO advantage 很难驱动正确修复。

还要检查 timeout 与 off-road 是否在 return 中得到合理区分，避免 critic 学成“慢行最安全”。

五、监督与数据 probes
17. BC 标签自一致性

对每个数据样本验证：

action chain 经 arc_step 积分得到的 pose
vs future ego world pose


按 horizon 1–6 报告误差。

如果 action label 与 future ego trajectory 不一致，应按来源分类：

expert 执行受 tracker/动力学影响；
action clipping； -时间索引偏移； -坐标变换；
episode boundary 窗口污染。

这是非常关键的盲点。否则模型同时被要求拟合 action chain 和 ego future，而二者可能互相矛盾。

18. Episode boundary leakage

检查训练窗口是否跨越：

reset；
termination；
timeout； -地图切换；
object track ID 重置。

任何跨 episode 的 history/future/action chain 必须被 mask，不得靠零填充后继续作为有效监督。

建议测试：

最后 5 个样本 + 下一个 episode 前 5 个样本


逐字段检查时间戳、world pose、route ID、track ID 和 valid mask。

19. Recovery 标签质量 hook

真实 recovery 采集后，每条标签必须附：

expert_recovered
recovery_time
minimum_margin
collision_after_recovery
route_progress_after_recovery
expert_vs_student_first_action_delta


低质量标签应过滤：

专家自己也失败； -专家首次动作与 student 几乎相同，但结果仍失败；
recovery 后长期 timeout； -动作链与 future pose 不一致； -边界标签不可用。

不要把“专家接管”自动等价成“有效恢复示范”。

20. Loss 梯度贡献 probe

每隔固定 update，例如 200–500 step，分别对每项 loss 单独计算目标模块的梯度范数：

BC action loss
trajectory loss
WM latent loss
ego-next loss
OD/LD loss
load-balance loss
anchor/KL loss
geometry loss
curvature loss


按模块记录：

encoder
memory
plan fusion
primary
experts
router
ST-GNN
policy
value


重点查：

某项配置了非零权重但梯度始终为零；
loss 数值很小但梯度极大；
geometry loss 被其他 loss 完全淹没；
specific-only 下本应冻结模块出现梯度； -可训练模块有梯度但 optimizer 没有包含；
optimizer 包含冻结参数。

仓库已有 test_grad_group_probe.py 和 test_anchor_grad_probe.py，应将其扩展成运行时梯度账本。

21. Optimizer 真正更新 probe

仅检查 requires_grad 不够。每次训练阶段开始后，抽取一次 update，记录：

parameter_before
gradient_norm
optimizer_state_exists
parameter_after
delta_norm


输出每个模块：

trainable parameter count
nonzero-gradient count
updated parameter count
mean/max update ratio


必须检测两类静默 bug：

参数有梯度，但没有加入 optimizer；
参数在 optimizer 中，但被意外冻结或梯度覆盖为零。
六、建议的落地优先级
立即和 P1 一起做
checkpoint fail-fast；
plan/action/arc-step 契约；
时间索引契约；
plan/reference/actual 三轨记录；
LQR 饱和与 progress probe；
footprint 插值；
recovery 标签质量；
loss-gradient 和 optimizer-update 账本。
首个 P1 candidate 前完成
BC action 与 future pose 自一致性；
坐标系 round-trip；
history 反事实；
WM action sensitivity；
MoE residual/router probe；
policy clipping/effective action probe；
value calibration。
P2 模块裁决前完成
WM horizon 分层误差；
encoder 线性 probe；
expert ablation；
No-WM/MoE 等预算对照；
按 road class、边界风险、速度分层的 paired KPI。
七、建议统一产出一个 health_report.json

每次正式训练或评测生成：

{
  "checkpoint_contract": {},
  "obs_contract": {},
  "time_alignment": {},
  "action_plan_contract": {},
  "geometry": {},
  "tracker": {},
  "progress": {},
  "history_sensitivity": {},
  "wm_sensitivity": {},
  "moe_health": {},
  "policy_distribution": {},
  "value_calibration": {},
  "gradient_accounting": {},
  "optimizer_accounting": {}
}


并给每项状态：

PASS
WARN
FAIL
NOT_AVAILABLE


正式 checkpoint 进入三 seed eval500 前，至少要求所有 correctness 类项目 PASS。性能类 probe 可以 WARN，但不能缺失。

最关键的五个隐藏风险

如果只能先做五项，我会选：

BC action chain 与 future ego pose 是否真正一致；
PPO sampled action 是否实际决定最终执行 plan；
训练、评测、forensics 的时间索引和 arc_step 是否完全一致；
geometry probe 查询的是否真是未来 footprint，而不是当前 ego context；
配置中的每项 loss 和 trainable scope 是否实际产生参数更新。

这五项最容易出现“所有测试都能跑、loss 也下降，但闭环永远上不去”的语义级 bug。