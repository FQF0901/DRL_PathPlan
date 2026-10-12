结论

R21-0/1 的数据与契约实现总体合格，R21-3 方向值得推进，但我不批准按当前文档的模糊定义直接开始主模型训练。

可以立即做：

网络显式接线；
归一化与消费契约测试；
用 9.5k pilot 做冻结主干的短适配和表示复测；
并行采集正式 v7 主训练集。

在表示门通过、S0/S1/S2 定义锁定后，再批准完整训练。

一、我核对代码后确认的事实

当前代码里：

lane_semantic 已进入 ObservationBuilder，shape 为 1×12；
LD 的 role/group/route_associated/offset_s 已随历史保存；
schema 已升为 v7，旧数据指纹默认 fail-fast；
但 net/model.py 和 net/mem.py 完全没有消费lane_semantic；
LD companions 目前也只是数据伴随项，没有进入网络；
PlanHead 仍只使用现有 z_ld → aggregate_ld() 路径；
target_lane_token 与 role-aware readout 在当前代码中尚不存在。

因此，R21-2 的正确表述应当是：

原始语义通道质量良好，但当前网络尚未接入；旧 LD 表示经过现有聚合后无法可靠读出 lane-relative 几何。

不应过早下结论为“encoder 层清白、pooling 是唯一主因”。因为新语义甚至还没进入 encoder。

二、R21-0 代码中发现的三个实质风险
1. 12 维语义特征没有归一化

当前 lane_semantic 直接输出：

lateral offset             几米
heading error              约 ±π
curvature                  约 0.01–0.1
margin                     几米到几十米
confidence                 0–1
binary flags               0/1
distance_to_lane_end       0–数百米
distance_to_topology       最大 200 m


如果直接进入一个线性 encoder，distance_to_lane_end 和 distance_to_next_topology_change 很容易主导激活和梯度，而 curvature、heading、binary flag 被淹没。

这是 R21-3 开训前必须解决的，不是训练后调参项。

建议在网络入口固定归一化：

lateral_offset             / 5
heading_error              / π
curvature_near/far         / 0.1 或 ×10
left/right margin          / 25
confidence                 原值
binary flags               原值
distance_to_lane_end       / 200
distance_to_topology       / 200


归一化常数必须进入 schema/模型快照和单测，而不是散落在训练脚本里。

2. distance_to_next_topology_change 的当前语义不准确

代码当前逻辑相当于：

topology = remaining if next_ref_lanes else 200.0


只要存在 next lane，就把“当前 lane 剩余距离”当作“下一拓扑变化距离”。

但 lane 结束可能只是普通直道的连续分段，并不一定是 merge、split、intersection 或 lane-count change。因此这个字段当前更接近：

distance_to_route_segment_end


而不是：

distance_to_next_topology_change


建议二选一：

本轮改名为 distance_to_target_lane_end，避免模型学习错误语义；
或真正沿 route successor 查找首次拓扑变化再计算距离。

为控制复杂度，我建议本轮直接改名，不实现新的地图搜索。

3. left/right_drivable_margin 不是合法路线余量

代码通过同 road、同向的相邻 lane 累加宽度计算 margin。它没有检查：

实线是否允许跨越； -护栏、收费岛等局部障碍；
route 是否允许进入相邻 lane；
footprint 是否真的处于连续可行区域。

因此它可以叫：

same-direction road-surface margin


但不能被解释为：

legal corridor margin


本轮可以作为输入特征，但不得用它直接构造安全 loss 或宣称已经显式建模合法 corridor。现有 geometry FP Gate 已失败，更说明不应把该字段升级为监督真值。

三、对 R21-2 结论的质疑
1. “pooling 丢失为主”尚未完全证明

R21-2 报告：

raw semantic R²≈1；
ld_pool lateral R²=0.191；
action 输出接近无法解码；
history shuffle 影响为 lane perturbation 的约 15 倍。

但这里混合了两个事实：

新 lane_semantic 根本没有进入模型；
旧 LD token 经 pooling 后几何信息弱。

所以更严谨的诊断是：

缺少显式 target-lane 语义通路，同时旧 LD 聚合无法提供足够强的几何选择性。

目前无法区分哪个是主要贡献。R21-3 正是用来裁决，不应在训练前写死结论。

2. “history shortcut”也应谨慎表述

history_shuffle 是明显的 OOD 干预。输出变化大可以说明模型依赖历史，但不能单独证明这种依赖是错误的。

R21-3 复测至少加入：

真实 6 帧历史
复制当前帧 6 次
最近 2 帧 + 其余复制
逆序历史
shuffle 历史


其中最可信的是：

真实历史 vs 当前帧复制


如果接入 semantic 后，几何反事实响应提高，同时真实历史仍优于复制历史，说明合理目标应是：

几何信号增强，而不是删除历史。

四、R21-3 的正确接线方案
1. lane_semantic 独立编码成 target token

建议新增：

lane_semantic 1×12
→ fixed normalization
→ 2-layer MLP
→ target_lane_token [B,H]


必须使用 lane_semantic_mask 和内部 target_lane_valid：

mask=0 或 valid=0 → token 严格为 0


第一版不要进入 MemBank。当前语义来自当前真实 pose 和当前 route，直接作为当前 target token 更清楚。后续若证明确实需要时间平滑，再做独立 temporal ablation。

2. LD companions 应真正进入逐槽路径

当前伴随数组：

ld_lane_role
ld_offset_s_norm
ld_group
ld_route_associated


已经随 6 帧 history 保存，但网络不消费。

建议将每个 LD token 增广为：

z_ld_slot
+ role embedding
+ route-associated embedding
+ normalized offset encoding
+ target-group relation


ld_group 不能作为数值大小使用。它只是局部 equality/group identity，应采用：

same-group mask；
group embedding； -或只用于 target/continuation token pooling。
3. role-aware readout 不要再做自由注意力搜索

Round-22 已显示纯 attention：

α 只学到约 1.3%； -权重近均匀； -无车道偏好； -α 增大反而性能下降。

因此 R21-3 不应重新定义成另一次 attention 实验。

建议使用显式分组池化：

target_pool
continuation_pool
left_adjacent_pool
right_adjacent_pool
route_pool
other_pool


然后融合：

ld_context = MLP([
    target_lane_token,
    target_pool,
    continuation_pool,
    left_pool,
    right_pool,
    route_pool,
    other_pool_mean
])


这样直接利用已验证的 role 契约，不要求网络从 9.5k pilot 中重新“发现 target lane”。

4. 必须统一消费位置

target_lane_token 和 role-aware ld_context 必须进入：

PlanHead t0
PlanHead rollout step 1–6
policy/value head tokens


更重要的是，必须使用同一个函数：

geometry_context = build_geometry_context(...)


禁止 plan_step() 再自行回退到 masked_mean(z_ld)。

建议增加代码级断言：

semantic mode 下：
PlanHead 不能调用旧 LD mean fallback

五、我反对当前文档中的 S1/S0/S2 模糊定义

目前 deepseekv4p1_argue.md 只说明：

S1：2 epoch 短适配；
S0/S2：通过后补；
S2 lane aux 保持轻量。

但没有定义：

冻结范围； -优化器参数； -具体 loss；
pilot 中 A/B/C 三段的采样比例； -正式 v7 数据集规模；
checkpoint 初始化和 fail-fast；
S0、S1、S2 各自究竟改变哪个变量。

这不足以批准主训练。

六、建议锁定的 R21-3 三阶段
R21-3A：接线与契约测试

只实现网络接线，不做主训练。

必须通过：

semantic invalid → token=0；
特征归一化逐位测试；
role/group permutation 一致性；
target-role 改变能改变 target pool；
PlanHead t0 和 rollout 每一步都消费 semantic context；
policy/value 消费一致； 7.旧 schema/旧 dataset/旧 ckpt fail-fast； 8.无 hidden mean fallback；
all-invalid forward/backward finite；
route/ego target 切换只发生在契约允许的位置。
R21-3B：Pilot adapter canary

使用 9.5k pilot，但只能作为表示验证。

建议冻结：

encoder 原主体
MemBank
ST-GNN
MoE/router
policy/value


只训练：

lane semantic encoder
role embeddings
role-aware readout
geometry fusion adapter
PlanHead 最后融合层，可选且低 LR


不要全模型训练，因为 pilot 中：

B 扰动占 42.7%，且 traffic=0；
C 只有 20 episodes；
A/B/C 分布差异很大；
pilot 明确不足以作为主训练数据。
Pilot loss

不要新增复杂几何损失。使用：

当前 action BC
+ 原有 plan/trajectory loss
+ 小权重 semantic reconstruction/probe loss


其中 semantic aux 只监督：

lateral offset
heading error
near/far curvature
lane-change-required


暂时不要监督当前定义存在歧义的 drivable margin 和 topology distance。

R21-3C：表示 Gate

短适配后必须复测：

target token 解码
role-aware context 解码
action 对 lateral/heading/curvature 的反事实响应
history-vs-lane 比
lane sign / mirror consistency


建议门：

target token:
  lat / heading / near curvature R² >= 0.80

action response:
  lateral 与 heading 扰动方向正确率 >= 90%
  curvature response >= expert 的 25%
  lane response 至少提升到旧模型的 3×

history/lane influence ratio <= 2


不要沿用“必须 5×”这种孤立倍数作为唯一 Gate，更应看距 expert 的比例和闭环方向性。

七、正式 v7 主训练数据集

Pilot 通过后再批准采集正式数据。

建议至少：

expert normal rows        200k–300k
controlled perturbation   30k–50k
learner-state rows        暂不纳入第一版主训练


首先建立稳定的 semantic BC baseline，避免再一次同时引入 covariate-shift 数据和新架构。

正式采集必须：

使用 CUDA 统一口径；
train/eval specs 严格隔离；
target selection、validity、role coverage 分层统计； -困难路型分别报告； -观测指纹 v7；
checkpoint meta 写入 fingerprint；
禁止 --allow-legacy-dataset 用于正式训练和评测。
八、正式训练消融应保持极简

Pilot Gate 通过后，只建议两臂：

S0：Semantic wiring only
新 semantic 接线
原有 Stage A/B loss
不加 semantic aux


回答：

显式语义输入本身是否带来收益？

S1：Semantic wiring + lightweight aux

只增加：

lat / heading / curvature / lane-change classification


回答：

显式监督是否防止语义再次被主任务忽略？

不建议再设 S2 或更多臂。两臂足够，避免再次陷入组合搜索。

九、闭环验收

先 clean150：

success >= 0.50
off-road <= 0.35
collision <= 0.05
timeout <= 0.15
steer saturation 相比 M0 明显下降
lane transition 附近 action jump 下降 >=30%


达到后跑 eval500：

success >= 0.60
目标 >=0.65


只有 eval500 ≥0.60 且 learner-state action error 显著下降，才重新讨论 A4 DAgger。RL 继续冻结。

最终裁决
立即批准
R21-3A 网络显式接线； -固定归一化；
target token；
role-aware 显式分组读出；
PlanHead/rollout/policy/value 统一消费； -契约单测；
pilot adapter canary；
pilot 后表示/反事实复测。
条件批准
正式 v7 主数据采集；
S0/S1 两臂正式训练；
clean150/eval500。

必须等待 pilot 表示门通过。

暂不批准
直接用 9.5k pilot 全模型训练； -未定义的 S0/S1/S2； -把 distance_to_next_topology_change 当真实拓扑距离监督； -把 drivable_margin 当合法 corridor；
A4 DAgger；
PPO/GRPO；
MoE/WM/容量改造。
最核心的代码审阅结论

R21-0 的数据与 target-lane 选择实现总体严谨，但当前代码存在三个必须在 R21-3 前处理的问题：

12 维特征尺度严重不一致；
distance_to_next_topology_change 名称强于实际语义；
drivable margin 只是同向 road-surface margin，不是合法 corridor。

修正这些后，R21-3 才是在验证真正的问题：

当 target-lane 几何被明确、归一化且贯穿 PlanHead 的每一步 rollout 时，模型能否摆脱 history shortcut，形成足够强的道路几何闭环纠偏能力。