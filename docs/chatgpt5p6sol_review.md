你说得对。下一步不应先围绕 DAgger、纵向头或某个局部 loss 开工，而应先完成一次全链路瓶颈裁决与代码减枝。目标是确认问题究竟主要在 World Model、Structured Policy、Memory Bank、输入旁路还是部署接口，然后只训练被证据指向的模块。

以下内容可直接给 Coding Agent 讨论执行。

下一阶段：全链路瓶颈裁决与代码减枝
一、架构守则

正式部署链只能是：

真实状态 z0
→ 唯一 PolicyHead 输出当前动作 a0
→ latent World Model 预测 z1
→ 同一个 PolicyHead 输出 a1
→ 重复至 a5
→ 固定 LQR 执行六步计划


必须满足：

只有一个动作生成器； -每次 Policy 只生成当前一步； -六步来自同一 Policy 的六次调用；
World Model 只推进世界状态，不生成动作；
PlanHead 只负责场景融合，相当于 Policy Backbone； -解析坐标变换可在网络外完成，但未来决策信息必须进入统一 World Model 状态； -不得依赖只有训练脚本或评测脚本才能生成的 path_* 旁路输入；
DAgger、MoE、RL 暂不启动； -固定 LQR 不修改。
二、本阶段不先训练，先回答五个问题
1. Structured Policy 在真实未来状态上是否正确？

比较：

真实未来 observation → Encoder/Memory → Policy

World Model预测未来latent → Policy


分别报告第 1 到第 6 步：

曲率动作
纵向动作
最终 ds/dtheta
车道保持误差
速度恢复能力


判定：

-真实未来状态下表现好，预测状态下差：World Model 是主瓶颈； -真实未来状态下也差：Policy、监督或状态表示是主瓶颈； -两者都好，但固定 LQR 差：轨迹可跟踪性或执行契约有问题。

2. World Model 哪一部分导致错误？

进行“真值替换”实验：

E0：全真实未来状态
E1：只预测 ego，OD/LD用真实值
E2：只预测 OD，ego/LD用真实值
E3：只预测 LD，ego/OD用真实值
E4：ego+OD预测，LD真实
E5：ego+LD预测，OD真实
E6：全预测


观察：

Policy输出变化
六步路径变化
横向误差
纵向速度
固定LQR结果


从而定位：

ego dynamics
OD动态对象
LD车道状态
还是多模态联合漂移

3. Memory Bank 是否有真实贡献？

比较：

完整6帧历史
仅当前帧
仅ego历史
仅OD历史
仅others历史
历史顺序打乱
最近帧重复6次


分场景报告：

普通保持
弯道
跟车
cut-in
merge
低速恢复
收费站


若完整历史没有显著优于当前帧，甚至恶化，不得继续扩展 Memory Bank。

还要专项检查：

-无效历史 token 是否严格清零； -全 mask 时 TemporalAttention 是否因输出层 bias 产生非零输出；

OD 历史 slot identity 是否稳定； -历史 OD 坐标是否需要 ego-motion 对齐；
LD 当前帧-only 是否符合预期。
4. 当前 path_* 旁路输入是否必要？

逐字段测试：

path_left_margin
path_right_margin
path_static_distance
path_dynamic_distance
path_dynamic_ttc
path_nav_progress
path_distance_to_target_lane_end


每个字段比较：

缺失
零值
中性值
真实值
由World Model anchor确定性计算


记录：

单步动作变化
h1到h6变化
闭环episode翻转
训练/评测分布差异


最终每个字段只能被归入四类之一：

删除； 2.并入标准 observation； 3.由 World Model 状态中的 anchors 确定性派生； 4.仅保留为诊断，不进入 Policy。

禁止继续保留“缺失时静默填 0”的模糊语义。

5. Primary、MoE 是否值得继续？

当前先做只读对照，不训练 MoE：

Primary-only
Primary + learned router
固定单个expert
随机router
Oracle router


但仅作为审计。

只要 Primary 在普通车道保持、基础跟车、低速恢复上不过门，MoE 就保持关闭。专家不能用于掩盖基础能力缺陷。

三、坐标解耦与 latent World Model

继续保留已实现的解析坐标层：

LD：
静态车道做ego SE(2)逆变换

OD：
对象场景速度推进
+
ego SE(2)逆变换


对象场景速度必须使用：

v_object = v_relative + v_ego


必须明确拆分：

Ak+1=Aanalytic+ΔAlearnedA_{k+1} = A_{analytic} + \Delta A_{learned}

即：

解析anchor
+ World Model学习残差
= 最终anchor


但是否立即重训 World Model，由真值替换实验裁决。不能因为零训练 warp 闭环无提升就否定 C3，也不能未经证据直接全量重训。

四、强制 Probes、Hooks 和桩函数

全部默认关闭，关闭时必须：

前向逐位一致
不消耗额外随机数
不改变optimizer
不改变checkpoint

1. 状态来源 Hook

每一步记录：

real / predicted / substituted
ego latent
OD latent
LD latent
nav/context latent
Memory latent

2. Policy Hook

记录：

PolicyBackbone latent
当前曲率输出
当前纵向输出
最终 ds/dtheta
辅助输出
输入字段 valid mask

3. World Model Hook

记录：

输入 action
warp前anchor
解析warp后anchor
learned residual
最终anchor
真实下一帧anchor
ego/OD/LD latent in/out
presence/entry

4. Memory Hook

记录：

每帧mask
slot identity
attention权重
历史帧贡献
全mask输出
历史顺序

5. 六步调用链 Hook

自动断言：

Policy调用6次
同一Policy实例
每个a_k读取对应z_k
用于决策的WM推进5次
ST-GNN总调用数单独记录
辅助输出不进入LQR
正式路径无oneshot和plan_override

6. 连续路径与控制 Hook

记录：

六步plan
插值LQR reference
连续footprint
首次越界/碰撞点
实际轨迹
steer/throttle
控制饱和

7. 参数与梯度账本

输出：

参数名
模块
requires_grad
optimizer group
grad norm
update norm
checkpoint是否变化


分别对第 1、3、6 步 loss 单独反向，验证真实梯度可达性。

五、验收准则
Gate A：架构契约
唯一动作源
Policy调用6次
WM决策推进5次
没有生产oneshot
没有生产plan_override
辅助输出不进入LQR
训练和部署调用链一致

Gate B：模块定位必须闭合

必须明确回答：

真实未来状态下Policy是否过门
哪种预测模态替换导致最大性能恢复
Memory完整历史的净贡献
每个path_*字段的最终归属
Primary和MoE的真实增量


不允许输出“可能多方面共同影响”而无量化比例。

Gate C：World Model
h1至h6误差不指数增长
解析anchor优于旧运动先验
learned residual进一步改善解析anchor
真实态与WM态Policy动作差距可量化
动态距离/TTC随rollout更新，不静默hold

Gate D：统一闭环

使用相同 evaluation_protocol_hash：

Calib50：
只做快速淘汰和机制分析

Clean500：
历史统一晋级集

Eval500：
正式历史比较和IDM+Pure Pursuit比较


所有关键版本必须重跑：

旧正式基线
当前候选
IDM + Pure Pursuit


主排序：

安全硬门：off-road、collision
主指标：success
次指标：timeout、route completion
诊断：离线/WM/路径指标

六、废旧代码清理原则

废旧代码不能靠“看起来没用”直接删除。每个候选必须经过四项判定：

1. 运行可达性

搜索所有：

import
构造
配置键
CLI参数
checkpoint键
反射调用
字符串模式
测试调用


若生产入口、训练入口、评测入口均不可达，标记：

UNREACHABLE

2. 契约合法性

即使可达，只要违反当前守则，例如：

第二动作生成器
一次输出六步
绕过World Model
外部替网络选择车道
缺失值静默填充
训练部署不同构


标记：

ARCHITECTURE_FORBIDDEN

3. 实验价值

若只用于历史消融或复现实验，迁移到：

tools/legacy/
experiments/archive/


不得继续出现在：

net/
pipeline/
config/default.yaml
正式CLI

4. Checkpoint 兼容性

仅为旧 checkpoint 加载保留的代码，应隔离为：

checkpoint_migration.py
legacy_key_mapper.py


不能继续参与正式 forward。

七、根据当前代码，优先审计的清理候选
A. 一次性六步路径代码

优先级最高：

oneshot模式
plan_override
compose_path_plan
one_shot_plan
PathExecutionHead别名
allow_oneshot
--diagnostic-oneshot


当前它们虽标记为诊断，但仍位于正式 net/model.py、net/path_head.py 主代码中。

处理原则：

-若没有必须长期保留的复现实验，全部迁到 tools/legacy/oneshot_path.py； -生产 DrivingModel 删除 oneshot 分支； -正式配置和 CLI 删除 oneshot 参数；

PathExecutionHead = StructuredPolicyHead 这种兼容别名应删除，避免误导；
plan_override 从正式 _rollout() 参数和分支中移除。

这是最容易再次造成架构倒退的一组代码。

B. K-anchor 旧计划头

当前仍存在：

num_anchors
plan_anchors
anchor dictionary
K-anchor selection head
相关冻结前缀
配置 plan_anchors_k6.json


长期保持关闭，而且设计上存在产生独立计划候选、偏离唯一 Policy 的风险。

处理方法：

1.确认所有正式配置 num_anchors=0； 2.重跑全仓 reachability； 3.如仅供旧实验复现，整体迁入 experiments/archive/k_anchor/； 4.从默认模型构造、训练冻结表、checkpoint主契约中移除； 5.旧 checkpoint 通过 key mapper 忽略这些键。

C. 遗留 set_ego_warp() 实验接口

当前同时存在：

set_ego_warp()
set_ego_warp_mode(off|analytic)


前者对应历史 w1/w2 实验，后者才是正式接口。

应：

-保留统一的 set_ego_warp_mode()； -将旧 set_ego_warp() 迁移到 legacy probe； -禁止生产代码同时出现两套互斥 warp 状态机； -配置只保留 off|analytic。

D. 旧 Policy 与新 Structured Policy 的双实体

当前运行时已做到单一动作源，但代码仍有：

model.policy
model.path_head
enable_path_head
path_head_ablation
policy_diag


长期应统一为：

model.policy
  structured_action
  action_distribution
  auxiliary_intent


处理顺序：

1.先完成 checkpoint key 迁移器； 2.将 Structured Policy 收编到 model.policy； 3.删除 enable_path_head 和 path_head_ablation； 4.旧 Policy 迁到 legacy 模型加载器； 5.正式 forward 只保留一个动作出口； 6.旧头诊断改为离线工具，不在主 forward 中并行计算。

E. path_* 旁路字段

重点候选：

path_left_margin
path_right_margin
path_static_distance
path_dynamic_distance
path_dynamic_ttc
path_nav_progress
path_distance_to_target_lane_end


这些字段不是统一 latent 状态天然产物，且出现过训练/评测缺键。

不要立即统一删除，先通过旁路审计分类：

删除
标准observation
从WM anchors确定性派生
仅诊断


分类完成后：

-正式 Policy 输入只允许统一 schema； -删除 None → 0 静默回退； -每个 optional 字段必须伴随 valid mask； -训练器、DAgger、评测共享同一 builder。

F. 外部 target-lane 强逻辑

当前：

select_target_lane()
target_lane token
target identity
lane semantic


不能全部删除，因为当前阶段仍用于“给定正确目标后的执行能力”。

但必须区分：

客观lane topology：保留
训练教师target：teacher_only
部署外部唯一target：废弃目标


应在 schema 中增加来源标签：

road_level_navigation
teacher_target_lane
deployment_available


长期删除生产部署对唯一 target lane 的依赖，但暂不删除课程训练和诊断路径。

G. Monitoring legacy tags

pipeline/monitoring.py 保留了大量：

_LEGACY_* regex
legacy_tags
旧标量映射


如果仅用于读取旧日志，应迁到：

tools/legacy/monitoring_reader.py


正式训练只写新 schema。

不要让每次新增 metric 都继续兼容多代 tag。

H. Cheap path 与隐式布尔开关

当前类似：

rollout=False
world_model=False


曾出现“看似关闭 World Model，实际仍执行 latent rollout”的语义混乱。

应替换为显式枚举：

forward_mode:
  policy_single_step
  policy_wm_rollout
  world_model_teacher_forcing
  diagnostics


避免多个布尔组合产生隐式行为。

八、代码清理交付物

Coding Agent 应提交：

docs/code_reachability_audit.md
docs/deprecation_manifest.yaml
docs/runtime_entrypoint_matrix.md
docs/checkpoint_migration_contract.md


deprecation_manifest.yaml 每项至少包含：

symbol:
file:
category: UNREACHABLE | ARCHITECTURE_FORBIDDEN | LEGACY_REPRO | CHECKPOINT_ONLY
production_references:
training_references:
evaluation_references:
test_references:
checkpoint_keys:
replacement:
removal_commit:


同时增加 CI：

禁止生产目录重新引用legacy模块
禁止default config出现deprecated键
禁止正式forward包含oneshot/plan_override
禁止两个动作生成器同时活跃
禁止缺失关键字段静默填0

九、给 Coding Agent 的执行摘要
本轮先做Root-Cause Matrix与代码减枝，不启动新训练主线。

1. 用真实未来状态、WM预测状态和模态替换，
   裁决瓶颈在Stage A还是Stage B。

2. 做Memory历史消融，
   证明时间信息是否有净贡献。

3. 审计全部path_*旁路字段，
   决定删除、标准化、由WM anchors派生或仅诊断。

4. MoE只读消融，Primary不过门不得训练专家。

5. 保留analytic OD/LD warp，
   但是否重训residual WM由oracle-state和模态替换证据裁决。

6. 建立统一Clean500/Eval500闭环基准，
   所有关键版本及IDM+Pure Pursuit重跑同一protocol hash。

7. 清理优先级：
   oneshot/plan_override/PathExecutionHead别名
   → K-anchor
   → 旧set_ego_warp接口
   → Policy/path_head双实体
   → path_*旁路
   → target-lane生产强依赖
   → legacy monitoring
   → cheap-path布尔组合。

8. 所有废旧功能迁出net/pipeline/config主路径；
   旧checkpoint只允许通过独立migration layer兼容。


最终原则：

先用动态证据确认哪个模块导致闭环失败，再训练它；同时将所有违反唯一 Policy、latent World Model、逐步六步递归和训练部署同构的历史代码从生产主路径彻底移走。