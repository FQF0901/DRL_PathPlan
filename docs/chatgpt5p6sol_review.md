总体判断

不批准当前“Round-20 已穷举模型侧，立即转数据侧重采”的结论。

实验结果本身可信，但我审查 DRL_PathPlan_Code.txt 后发现：M2 实际没有完整测试所宣称的“用 set-attention 替换 PlanHead 的 LD 均值池化”。因此，M2=0 不能用于否定 LD 架构侧路线，更不能作为全量重采数据的充分依据。

一、最关键的代码问题：M2 没有改到主要决策路径

当前 plan_step() 会重新计算：

od_pool = masked_mean(z_od, od_live)
ld_pool = masked_mean(z_ld, ld_live)


随后 PlanHead 使用这个 ld_pool。而 set_ld_mode("set_attn") 产生的 set-attention ld_pool 位于 encode() 路径，主要进入 policy/value token；它没有替换 rollout 每一步 plan_step() 内重新计算的 LD 均值池化。

也就是说，M2 实际更接近：

关闭 LD 的 spatial message passing
+ 新增 set-attention 给 policy/value 部分使用
+ PlanHead / rollout 仍然使用 mean(z_ld)


而不是预期的：

PlanHead 每一步都使用 ego/nav-conditioned LD set attention


这是决定性差异。

影响

M2 的闭环崩溃不能证明 set-attention 无效。它可能只是因为：

LD 被从 ST-GNN message passing 中完全排除；
主 planner 仍使用无选择性的均值池化；
policy/value 路径突然换成随机初始化的 attention context；
rollout 与首帧使用了不一致的 LD 聚合方式。

因此“模型侧 LD 修复已经穷举”这个结论不成立。

二、M2 还同时改了多个变量

set_attn 模式会设置：

exclude_ld_nodes = True


这会把 LD 从 spatial message passing 中全部屏蔽，再把原始编码 token 原样恢复。它不只是“换池化”，还改变了：

ego↔LD message passing；
ST-GNN rollout 中的空间交互；
policy/value 的 LD context； -新增随机初始化 attention 模块。

所以 M2 不是一个干净的单变量消融。

正确消融应拆成
M2a：
保留现有 spatial MP
仅将 PlanHead 的 LD mean 改为 conditioned set-attention

M2b：
在 M2a 基础上关闭 LD-LD chain
但保留 ego↔LD

M2c：
只有得到正向证据后，才考虑让 LD 完全绕过 spatial MP

三、LdSetAttention 本身还有两个实现风险
1. 没有平滑初始化

当前模块是：

query = Linear(...)
attn = MultiheadAttention(...)
out = Linear(...)


没有 residual、gate 或与旧 mean pooling 对齐的初始化。新模块直接替换旧 context，Stage B 需要从随机 attention 恢复完整闭环语义。

建议使用渐进形式：

ld_mean = masked_mean(ld_tokens, ld_mask)
ld_attn = set_attention(ld_tokens, ld_mask, context)

ld_context = ld_mean + alpha * ld_attn


其中：

alpha 初始为 0


这样新模型初始行为与 baseline 一致，再让训练学习 attention 增量。否则 M2 的 0.0 很可能包含初始化冲击。

2. 全 invalid 行存在潜在 NaN

当前先把所有 invalid token 交给 MultiheadAttention，再用：

torch.where(valid_any, out, zeros)


当一整行 key_padding_mask=True 时，softmax 可能先产生 NaN。后面的 torch.where 不保证梯度路径完全无 NaN。

应在 attention 前处理：

对 all-invalid 行插入一个零 dummy token； -或仅对 valid_any=True 的 batch 子集执行 attention； -并增加 forward/backward finite 单测。
四、对 M1 的判断也不应过早关闭

M1 只去除了 LD-LD 槽位链，保留 ego↔LD、ego↔OD 和 OD-KNN，这是一个干净消融。结果：

success:   0.3267 → 0.3533
off-road:  0.5533 → 0.5200
collision: 0.0333 → 0.0067
timeout:   0.0933 → 0.1200


它没有达到预注册的 +5pp，但方向并非“无效”，而是：

安全性改善，部分 off-road/collision 转化为 timeout，净成功率小幅提高。

clean150 上 +2.67pp 只相当于 4 个 episode，不能确认有效，也不能确认无效。建议不要马上升为主配方，但应保留为下一轮的低成本可选项。

五、置换不敏感不能证明伪边无害

当前模型对 LD slot permutation 近似不变，可能有两种解释：

LD-LD 伪边确实被网络学会忽略；
模型整体几乎没有有效利用 LD 几何，均值池化进一步抹平差异。

结合：

LD 平移/旋转响应极弱； -PlanHead 最终使用 LD 均值；
M1 小幅改善； -清空 LD 会有明显影响；

第二种解释不能排除。

因此正确结论应是：

槽位伪边不是当前输出置换敏感性的主要来源，但网络对 LD 的有效选择性和几何增益仍然不足。

而不是“伪边无害，模型侧已经穷举”。

六、暂不批准直接重采的“最小语义特征”

方向上，新增这些字段是合理的：

target_lane_flag
offset_index
route_association
lateral_offset
heading_error
curvature


但有两个设计问题。

1. 不应把 lane_id 当连续数值输入

MetaDrive 的 lane ID 通常是标识符，不具备数值距离语义。lane_id=3 与 lane_id=4 不代表几何更近。

正确表达应是：

same_lane_as_ego
same_lane_as_route
same_road
left/right relation
lane-group equality
lane id embedding 或 equality mask


不要直接归一化 raw lane_id 输入 MLP。

2. target_lane_flag 必须定义清楚

当前 route/nav 与 ego lane 并非简单一一对应。数据显示 ego.lane != nav current[0] 达 99.3%，原因是 nav candidate 的排序语义不同，而不是 ego 一直走错 lane。

因此需要分别定义：

ego_lane_flag
route_current_candidate_flag
route_next_candidate_flag
same_road_as_route
legal_successor_flag


不能只使用一个含义模糊的 target_lane_flag。

七、建议修正后的 Round-21
Round-21A：先修正确的 M2，不重采数据
M2a：PlanHead conditioned pooling

确保以下所有位置统一使用同一个 LD context：

t0 PlanHead
rollout step 1–6 PlanHead
policy/value token


建议接口：

ld_context = aggregate_ld(
    z_ld=z_ld,
    ld_mask=ld_live,
    ego=z_ego,
    nav=nav_token,
    others=others_ctx,
)


plan_step() 不得再私自重新 masked_mean(z_ld)。

初始化
baseline mean + zero-init gated attention residual


以保证初始化时逐位接近 baseline。

Spatial 口径

第一版保留 ego↔LD message passing，只关闭 LD-LD slot chain。不要 exclude_ld_nodes=True。

Round-21B：增加四个代码契约测试

PlanHead consumption test

修改 attention 输出时，plan step 1–6 必须发生变化。

No hidden mean fallback

set-attention 模式下，禁止 plan_step() 再调用 LD masked mean。

All-invalid finite

LD 全无效时，forward、backward 均 finite，输出为零或 baseline fallback。

Zero-init equivalence

alpha=0 时，set-attention 模式与 baseline 前向逐位一致或误差低于 1e-6。

Round-21C：小规模闭环 Gate

先只跑 clean150：

M0 baseline
M1 no_chain
M2a no_chain + correctly wired gated attention


验收：

success >= M0 +5pp
off-road 至少下降 7–10pp
collision 不恶化
timeout 不增加超过3pp
steer saturation 下降
几何扰动响应显著高于 M0


如果正确接线后的 M2a 仍低于 0.45，再批准数据侧重采。

八、数据侧重采的批准条件

只有在 M2a 失败后，才进入 schema v7：

LD geometry:
  dx, dy
  heading_rel
  curvature
  speed_limit
  line types

LD semantics:
  ego_lane_flag
  route_current_flag
  route_next_flag
  legal_successor_flag
  offset_index one-hot/embedding
  same-road / left-right relation


同时：

signal 恒 unknown 继续保留 availability flag，或从 active trunk 移除；
obs fingerprint 必须升级； -旧 checkpoint 必须 fail-fast； -先采 200–500 episode pilot，不直接全量 5k； -确认 feature coverage、label consistency 和几何响应后再采全量。
九、最终裁决
批准
保留 Round-20 结果与 M1 证据； -修复 M2 的实际接线； -实现统一 aggregate_ld()； -使用 gated residual attention 初始化； -补四个契约单测； -重跑一次 M0/M1/M2a clean150。
暂不批准
立即全量重采； -直接增加模糊 target_lane_flag/lane_id；
M3 几何辅助 loss；
A4 DAgger；
PPO/GRPO； -宣称“模型侧 LD 路线已经穷举”。
核心判断

Coding agent 的实验纪律很好，但这次代码审阅发现了一个关键问题：

M2 改到的主要是 policy/value 和 spatial rollout，而 PlanHead 每一步仍重新使用 LD 均值。

因此 M2=0 并没有测试我们真正想测试的架构。

下一步不应昂贵地全量重采，而应先用一轮很小、接线明确的 M2a 验证：

当 PlanHead 在每个 rollout step 真正获得 ego/nav-conditioned、排列不变的 LD context 后，闭环性能是否改善。

只有这个实验仍失败，转数据侧才是证据充分的决定。

==============================================================
我希望chatgpt5p6sol给出充分的下一步计划、可能的情况和通过准则，让coding agent充分排摸并验证模型侧，避免来回拉扯很耗费时间
==============================================================
结论先行

可以停止模型侧的来回试探，正式锁定 Round-21：车道语义重建 → Primary 基础能力 → learner-state DAgger。

最新结果已经证明：

当前 LD 的相邻槽位中，仅约 23.5% 属于同一 lane，伪边约 76.5%；
lane assignment 断点在失败集中的发生率约为成功集的 14 倍；
模型对 LD 几何扰动的响应只有 oracle 的 2%–4%；
zero_history 干预远强于 LD 几何干预，说明动作主要受历史/动力学 shortcut 支配；
M1 仅从 0.3267 提升到 0.3533，M2 set-attention 直接崩到 0。

但必须精确表述：

M2 失败证明的是“当前 LD 语义与槽位组织下，继续修图结构无效”，不证明 set-attention 或结构化道路建模本身无效。

下一步不再改 LD pooling、邻接、注意力或网络容量。先重建输入语义，并让 Primary 单独掌握 lanekeep。

一、Round-21 总体目标

回答四个问题：

输入是否稳定表达了 应该跟随哪条 route lane？
网络是否理解 自车相对目标 lane 的位置和方向？
Primary 是否独立具备基础 lane keeping 和轻度纠偏能力？
完成上述三点后，剩余差距是否主要来自 learner-state covariate shift？

执行顺序锁定为：

R21-0 语义契约与标签审计
→ R21-1 semantic_v1 pilot 数据
→ R21-2 表示与反事实 Gate
→ R21-3 Primary-only 基础训练
→ R21-4 learner-state current-action DAgger
→ R21-5 恢复 MoE / RL 的条件裁决


每一阶段未通过即停止，不自动进入下一阶段。

二、R21-0：定义最小车道语义契约
2.1 新增输入

不要把 MetaDrive 的原始全局 lane_id 直接作为模型特征。它是地图内部标识，容易导致记忆地图或产生不可泛化语义。

建议增加一个独立的 lane_semantic 通道：

ego_to_target_lane_lateral_offset
ego_heading_error_to_target_tangent
target_lane_curvature_near
target_lane_curvature_far
left_drivable_margin
right_drivable_margin
target_lane_valid
route_association_confidence
lane_change_required_left
lane_change_required_right
distance_to_lane_end
distance_to_next_topology_change


现有 LD 点额外补充：

local_lane_role:
  target
  target_continuation
  left_adjacent
  right_adjacent
  other

offset_s_normalized
same_lane_group
route_associated


其中 same_lane_group 只能是当前 observation 内的局部 group，不是地图全局 lane ID。

2.2 Target lane 选择规则

必须写成唯一、可测试的纯函数：

优先 route current_ref_lane
→ route continuation
→ next_ref_lane
→ 最后才允许 ego.lane fallback


不能继续以 ego.lane 为最高优先级。车辆已经偏离时，ego.lane 很可能正是错误 lane。

2.3 产物
docs/r21_lane_semantic_contract.md
docs/r21_target_lane_selection.md
tests/test_lane_semantic_contract.py
tests/test_target_lane_continuity.py
tests/test_ld_group_topology.py

2.4 通过标准
target lane 有效率 ≥99%； -普通道路 ≥99.5%； -在非真实 route transition 时，target lane identity 不得因 ego 横向偏移而切换； -同一 lane 的 offset_s 严格递增； -同 lane group 内不能混入其他 lane；
lane-relative 特征与 MetaDrive 几何查询的误差：
lateral offset <0.05m；
heading error <0.5°；
curvature 相对误差 <10%，极小曲率除外； -所有坐标 round-trip 测试通过；
CUDA 作为采集、训练诊断和正式评测的统一后端。

任一失败，先修 observation builder，不采正式数据、不训练。

三、R21-1：采集 semantic_v1 pilot

不要先采大数据。先验证语义。

3.1 数据组成

建议首批 8k–12k 行：

A. 正常 expert 状态，约 40%

覆盖：

-直道； -缓弯； -急弯； -tollgate； -merge； -roundabout；

route transition。
B. 受控 lane-relative perturbation，约 40%

从合法 expert 状态构造：

lateral offset:
-1.0, -0.5, -0.25, +0.25, +0.5, +1.0m

heading error:
-10°, -5°, -2.5°, +2.5°, +5°, +10°


仅保留：

footprint 仍在有效道路内； -没有即时碰撞； -target lane 定义明确； -expert 能在指定 horizon 内恢复。
C. learner-state，约 20%

来自当前 pri512/M1 的真实闭环状态，保存：

obs_t
history_t
target-lane semantics_t
model plan_t
expert current action_t
model current action_t
episode outcome


只查询当前 expert action，不要求未来六步 expert chain。

3.2 数据质量 Gate
同状态 observation/label 匹配率 = 100%
target lane 有效率 >=99%
随机重放 >=1,000 行
episode split 无泄漏
专家恢复成功率 >=90%
异常 lane assignment 跳变率 <1%


还必须输出：

dataset road-class distribution
lane-relative error distribution
动作修正方向分布
各速度带覆盖
target-lane fallback 使用率

3.3 产物
datasets/semantic_v1_pilot/
  samples.*
  manifest.json
  schema.json
  split_manifest.json
  replay_report.json
  coverage_report.json

docs/semantic_v1_dataset_card.md

四、R21-2：不开训练主模型，先做表示 Gate

这一阶段决定模型是否“看得懂 lane”。

4.1 输入级正确性

直接从 lane_semantic 验证：

lateral offset
heading error
curvature
boundary margin
target lane role


应无歧义、无严重不连续。

4.2 表示可解码性

分别从以下位置训练轻量线性 probe：

raw lane_semantic
LD embedding
target-lane token
PlanHead fused latent
primary hidden
最终 action latent


预测：

signed lateral offset
heading error
local curvature
左右 margin
是否需要 lane change

通过标准
lateral offset R² ≥0.90；
heading error R² ≥0.90；
curvature R² ≥0.75；
lane-change direction balanced accuracy ≥0.90； -若 raw 输入通过，但融合 latent 不通过，说明信息在 encoder/pooling 过程中损失。
4.3 Counterfactual lane-grounding

在保持动态对象和速度一致时，改变 lane-relative error：

横向偏移
heading error
lane curvature
target lane role


检查 Primary 输出的修正方向。

通过标准
correction sign accuracy ≥95%； -左右镜像一致率 ≥95%； -误差增大时，修正幅度总体单调，Spearman ρ ≥0.8； -零误差直道时，dθ 分布接近零； -曲率改变时，输出变化达到 expert/oracle 响应幅度的至少 25%； -清除 prev_action 后仍保留正确 lane correction； -历史动作反号不能把 lane correction 符号整体带反。

目前只有 oracle 的 2%–4%，新输入至少应提升到 25%，否则不值得进入完整训练。

4.4 Shortcut 对照

比较：

正常输入
去掉 lane_semantic
去掉 LD
prev_action 清零
prev_action 反号
history 重复 current
history shuffle


判定：

lane 特征变化对 dθ 的影响必须显著高于 prev_action 微扰；
lane correction 方向不能依赖历史动作符号； -若仍由 history 主导，停止并检查 fusion、归一化和监督。
五、R21-3：Primary-only 基础能力训练

只有 R21-2 通过后才训练。

5.1 模型原则

保留 learned chain，因为它已被证明能缓冲不稳定首步动作。不要继续使用 fixed repeat tail。

但重新定义职责：

Primary:
  lane keeping
  route following
  normal curvature tracking
  ordinary recovery
  plan smoothness and trackability

Chain:
  可跟踪的短期 reference generator
  不是精确复现未来 expert chain

Experts:
  本阶段完全关闭

5.2 最小网络改动

新增：

target_lane_token
lane_semantic_encoder
role-aware target-lane pooling


不要重新启用错误的 slot-chain adjacency，也不要继续试 set-attention。

融合结构保持简单：

ego token
target-lane token
adjacent-lane summary
OD summary
nav token


Primary 必须直接看到 target-lane token，不能再只看到所有 LD 点的均值。

5.3 监督

主要监督：

当前 expert action
第一步 plan/action consistency
lane-relative correction direction
lane tangent alignment
curvature consistency
chain smoothness
LQR trackability


未来链可以保留弱监督，但不能主导：

step weights:
1.00, 0.40, 0.20, 0.10, 0.05, 0.05


不加入：

gated residual；
MoE experts； -未经验证的 LD margin loss； -新的 WM loss； -复杂 anchor； -RL。
5.4 三组必要对照

必须等参数、同数据、同 seed：

S0：旧输入 + 旧模型
S1：新语义输入 + Primary-only
S2：新语义输入 + Primary-only + lane auxiliary


不允许同时改变 hidden size、expert 数或 WM。

5.5 基础能力 Gate
合成/受控 lane grid

-直道/缓弯 lane keeping success ≥95%； -横向/航向 perturbation 3 秒恢复率 ≥90%； -修正方向正确率 ≥98%；

steer saturation <5%； -无明显左右不对称。
clean150

首个候选必须达到：

success >=0.50
off-road <=0.35
collision <=0.05
timeout 不高于旧基线 +3pp


未达到则不进入 eval500，回到 representation/监督定位。

eval500
success >=0.60
off-road <=0.25
collision <=IDM


达到才允许 learner-state DAgger。

六、R21-4：Current-action DAgger

这一步才针对你指出的 covariate shift。

6.1 采集内容

在 learner 实际状态查询 expert：

learner obs/history
lane_semantic
learner current action/plan
expert current action
target lane
road class
termination precursor


明确禁止：

-未来六步 expert chain 作为主要标签； -recovery gate； -Pass-1/Pass-2 状态错配； -极端失败窗口过采样。

6.2 数据并入

逐轮：

Round 1：learner-state 10%
Round 2：learner-state 20%
Round 3：最多 30%


每轮必须保留原 expert 数据，避免基本盘遗忘。

建议最多三轮，每轮只训练 1–2 epochs，并固定验证集与 checkpoint selection。

6.3 每轮通过标准

相对上一轮：

learner-state MAE_ds 至少下降15%
learner-state MAE_dθ 至少下降20%
dθ sign error 明显下降
clean150 success 至少 +3pp
off-road 至少下降10%相对值
teacher-state MAE 恶化不超过10%


连续两轮闭环提升 <2pp，停止 DAgger，不再扩数据。

七、R21-5：MoE、RL 和最终目标
7.1 MoE 只在 Primary 通过后恢复

重新启用前必须证明：

primary-only 已具备稳定 lanekeep


Experts 只允许处理：

tollgate；
merge/yield； -复杂交叉口； -避障； -特殊恢复。

必须有场景标签或可验证 router specialization。不能只靠负载均衡。

MoE 保留 Gate
full-MoE 相对 primary-only success >=+3pp
普通 lane keeping 不退化
expert residual 对普通道路接近0
路由与困难路型存在稳定关联
3 seeds 成立


不满足则保持 Primary-only。

7.2 RL 启动门

同时满足：

eval500 >=0.60–0.65
learner-state action error已经明显下降
sampled action确实影响执行plan
训练和评测同为PlanHead+LQR
action clipping <5%
LQR saturation可控
value能在失败前识别风险


只从 2D action-conditioned plan PPO 开始，不启动 GRPO、12D trajectory 或 mode+latent。

7.3 最终验收
G2：追平
3 seeds mean success >=0.74
off-road <=0.098
collision <=IDM
eval500 + clean500均成立

G3：正确超过 IDM+Pure Pursuit
3 seeds mean success >=0.80
worst seed >=0.77
paired bootstrap lower bound >0
McNemar p<0.05
collision <=IDM
timeout无明显恶化
至少两个困难路型显著提升


目标 0.82–0.85 时补 5 seeds。

八、完整决策树
情况 A：target lane 覆盖低或语义跳变高

根因在 observation builder。

行动：

修 target-lane selection
修 route continuation
重采 semantic_v1
禁止训练

情况 B：raw semantic 正确，但 latent 不可解码

根因在 encoder/pooling。

行动：

检查归一化
增加 target-lane token
取消全 LD masked mean 对 target lane 的替代
不加容量

情况 C：latent 可解码，但 Primary 动作不响应

根因在 head 或监督。

行动：

primary-only lane auxiliary
lane tangent / correction direction监督
保持 experts关闭

情况 D：expert-state 正确，learner-state 错

根因以 covariate shift 为主。

行动：

current-action DAgger
每轮严格闭环验收

情况 E：Primary 基础能力仍失败

停止 MoE 和 RL。检查：

target lane标签
动作定义
归一化
LQR可跟踪性
expert标签连续性

情况 F：Primary 达到0.60–0.65，但低于IDM

继续：

DAgger
稳定bias calibration
场景型MoE
最后才是PPO

情况 G：语义+DAgger 后仍长期低于0.60

承认纯模型路线暂时失败，转向：

IDM+Pure Pursuit
+ bounded learned residual
+ safety arbitration


但只有合法在线信息 oracle 能达到 ≥0.82，才值得实现。

九、统一产物

每阶段必须生成：

manifest.json
resolved_config.yaml
git_state.json
checkpoint_load_report.json
health_report.json
probe_results.json
episodes.csv
metrics.json
decision.md


Round-21 专项文档：

docs/r21_lane_semantic_contract.md
docs/r21_target_lane_selection.md
docs/r21_model_plan.md
docs/semantic_v1_dataset_card.md
docs/r21_dagger_contract.md


decision.md 只回答：

假设是什么
改变了什么
证据是什么
通过/失败哪个Gate
下一步唯一允许动作

十、Coding Agent 开工指令
批准 Round-21，但仅先执行 R21-0 至 R21-2。

1. 定义 target-lane 与 lane-semantic 契约。
2. 修复并测试 lane group、offset、route association。
3. 采 semantic_v1 pilot，包含 expert、受控扰动和少量 learner-state。
4. 完成 representation decoding、lane counterfactual、shortcut probes。
5. 输出明确的 Gate 结论。

R21-2 通过前：
- 不训练完整主模型；
- 不启动 DAgger；
- 不启用 MoE experts；
- 不改 WM；
- 不启用 RL；
- 不扩大容量；
- 不再试 LD set-attention 或 slot-chain adjacency。

R21-2 报告必须明确问题位于：
输入缺失 / target-lane不稳定 / encoder丢失 / pooling丢失 /
primary监督不足 / history shortcut 中的哪一层。


这版计划已经覆盖输入、表示、Primary、DAgger、MoE 和 RL 的完整决策链。后续 Coding Agent 不应再次就大方向来回讨论，只需按 Gate 提交证据并执行唯一允许的下一步。