我审阅了 Coding Agent 最新的 deepseekv4p1_argue.md。这轮大部分是合理的工程收口，但有一个重要错误：不能把 v_next 与平均速度高度相关，视为删除 v_next、自动退化为 3D 动作的依据。

一、可以直接接受
1. EMA 与 detach 按阶段解释

接受：

D1：冻结 Encoder，Target Encoder 固定，EMA 实际为 no-op
D2：Encoder 继续冻结，固定 Target + 防坍缩监控
E：解冻 Encoder adapter 后，才真正启用 EMA 更新


并监控：

||theta_target - theta_online||
latent norm
per-dimension std
effective rank


Coding Agent 说得对，EMA 是 moving-target 稳定器，不是单独就能防止 latent collapse 的万能方案。

2. 验证集分层

原则接受，建议最终定为：

Train：10000 scenes
  └── Calib50：从 Train 中固定抽取，只作快速调试

Validation：500 scenes
  └── checkpoint selection、阈值选择、消融比较

Test：500 scenes
  └── 最终里程碑评测，不调参、不删失败样本


这样仍满足原始的“10000 个训练场景、1000 个冻结评测场景”，只是把原来的 1000 拆成 Val500/Test500。

需要注意：

Calib50 属于训练域，不能用于收益声明；
Test500 在项目开始时固定；
failure case 不能因为“仿真困难”就在看到结果后删除；
全量 Test500 只在阶段里程碑运行。
3. 先用小规模打通工程

接受：

P0/P1 smoke：
2000 train / 200 evaluation-scale scenes

最终：
10000 train / 500 val / 500 test


但 2k/200 只能验证：

Schema；
Expert 数据采集；
action round-trip；
train/eval assembly；
BC 是否能收敛；
内存、FPS 和 worker 稳定性。

不能用 2k/200 得出最终闭环性能结论。4 worker 和内存水位应由实测确定，不必把 recycle=100–150 写成不可修改的算法合同。

4. v_next 跨步速度链

完全接受：

第 n 步：
(v_n, action_n) → v_{n+1}

第 n+1 步：
其起始速度必须来自 imagined ego state 中的 v_{n+1}


不能让下一步 Plan Head 重新生成一个与上一段终点无关的起始速度。

应增加：

test_action_kinematics
test_velocity_chain_consistency


检查：

段间速度连续；
heading 连续；
curvature/curvature-rate 可接受；
acceleration/jerk 不越界；
stop-to-go 不发生速度跳变。
5. 动力学边界和低速 Controller

接受以下约束：

v_next >= 0
a_min <= (v_next - v_n) / 0.5 <= a_max
|kappa| <= kappa_max
|v^2 * kappa| <= a_lat_max


低速时单独使用 stop/hold controller 是合理的，因为这是执行控制器职责，不是向 Planner 注入 TTC、target_lane 等强策略逻辑。

同时必须记录：

action_clip_rate
action_reject_rate
post_process_rate
throttle/brake/steer saturation


特别是 PPO 中不能让 feasibility layer 大幅修改采样动作，却仍把修改前的 log_prob 当成已执行动作概率。

6. 静态 LD/Nav 监督

接受，建议定案为：

Static LD:
  geometry reprojection loss
  + re-encode consistency
  + optional low-weight residual loss

Road-level Nav:
  route-progress consistency
  + geometry/topology consistency
  不做普通动力学预测


Target 侧 stop-gradient。静态 LD/Nav 不进入动态 OD 的普通 latent transition loss。

7. 测试和失败分类补充

Coding Agent 建议新增的测试基本都应接受，特别是：

test_action_kinematics
test_plan_head_single_instance
test_param_budget
test_tensor_budget
test_lqr_gt_tracking
test_lqr_plan_tracking
test_train_eval_assembly_parity
test_bank_entity_schema
test_validation_isolation


新增失败分类也合理：

PLAN_INTERFACE_SHIFT
ACTION_V_NEXT_INCONSISTENT
STATIC_REPROJECTION_ERROR
MEMORY_WARMUP_ERROR
TRAIN_EVAL_ASSEMBLY_MISMATCH


这些都对应此前真实暴露过或当前架构高度可能出现的问题。

二、部分接受，但要修改
1. P0 应是“校准验收门”，但不能完全取消否决能力

建议正式定义为：

4D 动作 (dx,dy,dpsi,v_next) 已是基线合同。P0 默认只校准标签、bounds、归一化、插值和动力学可行性，不允许脚本静默改变动作维度。

但如果发现以下结构性失败：

Expert 轨迹无法稳定转成 4D 标签；
Hermite 无法满足端点、终端速度和曲率边界；
full-stop/stop-to-go 无法重建；
Expert GT 经该动作重建后也无法被 LQR 稳定跟踪；

则必须暂停 P1，提交 decision log，人工重新审阅 action schema。

所以它不是日常意义上的“重新选型门”，但仍须保留灾难性结构缺陷的人工否决权。

2. v_next 跨步一致性应采用“状态硬继承 + 轨迹软约束”

这里不能只选硬约束或 soft loss，二者用途不同：

硬约束
next_step.v_start = ego_dynamics(previous_step.v_next)


下一步的起始速度来自上一段终态，这是状态递推合同，必须硬保证。

Soft loss
acceleration consistency
jerk consistency
segment-boundary velocity/curvature continuity


Plan Head 输出和插值轨迹的平顺性采用 soft loss，并由 feasibility gate 做最终限制。

3. Taxonomy 应按“语义类型”和“动力学类型”双轴划分

Coding Agent 希望直接分成 OD-dynamic、OD-static、LD、Nav、RoadState，方向正确，但不能把类别互斥得过于僵硬。

建议 token schema 至少包含：

semantic_type
mobility_type: dynamic / static
geometry_type
state_type


具体处理：

车辆、行人、骑行者：OD + dynamic；
锥桶、护栏、施工障碍：OD + static；
收费站建筑及固定通道几何：静态基础设施，可由 LD/infrastructure encoder 表达；
闸机开闭、交通灯相位：RoadState + dynamic；
车道、边界、停止线、路口拓扑：LD；
道路级路线与 route progress：Nav。

关键不在于收费亭究竟必须叫 LD 还是 Static OD，而是静态设施不能被送入动态 WM transition 分支。

4. Traffic light 保留为配置能力，不强制首版场景必须启用

建议：

use_traffic_light: false  # P0/P1 默认
use_dynamic_gate_state: true/false


即使首批 MetaDrive 场景不含交通灯，也保留 RoadState schema 和 mask，避免以后加路口信号时修改网络接口。

三、需要反驳
1. 反驳自动回退为三维动作

Coding Agent 建议：

如果 v_next 不可辨识或与 ||(dx,dy)||/0.5 相关性过高，就退化为
 v_next = ||(dx,dy)||/0.5。

这个建议不能接受。

高相关性只说明多数普通驾驶片段接近匀速，不代表 v_next 冗余。最重要的样本恰恰是低占比但关键的：

障碍前停车；
红灯或闸口停车；
跟随慢车减速；
停车后重新起步；
同一位移终点下不同终端速度。

例如，同样在 0.5 秒内前进 2 米：

轨迹 A：v0=8 m/s，强制动后 v_end≈0
轨迹 B：v0=4 m/s，近匀速后 v_end≈4


两者 (dx,dy,dpsi) 可以接近，但下一时刻状态完全不同。

因此：

corr(v_next, distance / 0.5) 高


不是删除 v_next 的有效判据。

真正需要评估的是：

conditional residual of v_next
stop/full-stop reconstruction error
stop-to-go reconstruction error
terminal-speed prediction error
下一步可达域误差


定案：不预定义自动 3D 回退。 如果 4D 标签存在问题，优先修复 Expert state extraction、时间对齐和速度标签，而不是把平均速度冒充终端速度。

2. 不接受把具体机器参数写成长期算法硬合同

例如：

4 worker
recycle 100–150
mem_available_floor_mb=3000


可以作为当前机器的 P0 默认配置，但应放在运行配置或环境报告，不应写进不可违反的架构合同。

因为后续更换机器、并行环境或 observation token 数后，这些值必然需要重标定。

四、需要 Coding Agent 说明或修正
1. P0 100–500 个 episode 是否覆盖关键动作分布

不能随机收集 100–500 个普通道路 episode 就判断动作参数化成立。必须按场景分桶采集：

lane keeping
curve
full stop
stop-to-go
slow following
obstacle braking
lane change
junction
fork/merge
toll-gate approach


并对低速、急减速和大曲率样本做最低配额要求。否则相关性统计会被大量匀速直行样本支配。

2. LQR shadow 通过阈值不要现在拍死

Coding Agent 要求明确 seeds 和阈值，方向正确，但当前还没有 MetaDrive/LQR 基线数据，不应编造固定数值。

建议先规定相对门槛：

Expert GT + LQR 必须先通过
Plan + LQR 相对上一阶段不得显著退化
Exact 提升但 LQR deterioration 超出统计波动时禁止推进
控制饱和率和 infeasible rate 必须持续报告


P0 获取分布后，再把绝对阈值写入 config/eval_thresholds.yaml 和 decision log。

3. DAgger 混合系数不应预先固定

1.0→0.7→0.4→0.2→0.0 可以作为默认候选，但实际下降应由以下指标触发：

Student success；
collision/off-road；
Expert intervention rate；
recovery success；
LQR shadow 可跟踪率；
failure-buffer 覆盖率。
五、可直接回复 Coding Agent

本轮关于 EMA/detach 分阶段、Val/Test 隔离、2k/200 工程 smoke、v_next 速度链、动力学边界、低速 stop controller、静态 LD/Nav 监督、taxonomy 双轴拆分、测试与失败归因补充，原则接受。

P0 定义为 4D action 的校准/验收门，schema 默认固定，不允许脚本静默改维度；但若 Expert action round-trip、full-stop/stop-to-go 或 Expert-GT LQR 根本无法成立，必须暂停并提交 decision log 人工复审。

不接受以 v_next 与 distance/0.5 相关性高为由自动退化成 3D。平均速度不等于终端速度，停车、起步和制动场景正是该维度的必要性所在。应检查 terminal-speed residual、full-stop、stop-to-go 和下一步状态误差，优先修复标签与时间对齐。

v_next 一致性采用“状态硬继承 + 平顺性 soft loss”：下一步 v_start 必须来自上一段 Ego Dynamics 的终态；acceleration、jerk 和曲率连续性用 loss 和 feasibility gate 约束。

10000 Train + 500 Val + 500 Test 保留；Calib50 从 Train 抽取。2k/200 只用于 P0/P1 smoke，不替代最终数据规模。4 worker、recycle 和内存水位作为当前机器运行配置，不写成长期架构硬约束。

请据此对终板做最后一轮局部合同补充，不再重开动作、时间、Bank/Fusion、Hybrid WM、主训练顺序及 B1/B3。

总体判断：约 85% 可接受；必须反驳的是“按相关性自动删除 v_next”，其余主要是把合理建议从模糊原则改成明确的分阶段合同。