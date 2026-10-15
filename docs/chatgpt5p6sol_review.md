你说得对。Round-36 的离线结果只能证明新结构“值得进入闭环验证”，不能替代闭环性能。下一阶段必须围绕统一闭环基准推进，同时修复当前代码中的动作头归属、World Model 坐标语义和动态场景更新问题。

下面这版可直接交给 Coding Agent。

Round-37：结构化 Policy + 坐标解耦 World Model 闭环验证
一、总目标

保持唯一部署链：

真实状态
→ 唯一 Structured PolicyHead 输出当前一步动作
→ World Model 预测下一状态
→ 同一个 PolicyHead 重新决策
→ 递归六步
→ 固定 LQR
→ 闭环评测


本轮目标不是继续优化单个离线指标，而是回答：

新的曲率和纵向动作表示，配合 OD/LD 自车运动坐标补偿，能否真正提升固定 LQR 下的闭环 success，并降低 off-road、collision 和 timeout？

二、统一评价体系

所有后续版本必须使用同一基准、同一随机种子和同一统计字段。阶段性专项指标只能解释结果，不能代替统一指标。

1. 固定评测层级
开发集：Calib50

用途：

快速淘汰明显失败版本
paired episode逐例比较


不能用于正式收益声明。

晋级集：Clean150

用途：

确定模型是否具备继续扩评资格

正式集：Eval500

用途：

与历史模型及IDM + Pure Pursuit比较

最终声明
至少3 seeds
paired bootstrap
McNemar成功/失败翻转检验


最终超越目标继续保持：

3-seed mean success >= 0.80
worst seed >= 0.77
collision和off-road不劣于IDM

2. 所有版本统一输出
主指标
success
off-road
collision
timeout
route completion

安全和执行指标
min-TTC
首次计划越界距离
首次计划碰撞距离
steer saturation
throttle saturation
LQR横向误差
LQR航向误差

World Model 接口指标
真实状态 vs WM状态的Policy动作差
h1...h6的ego/OD/LD误差
lane-relative误差
ds overshoot
OD/LD中的ego-motion泄漏

3. 不使用模糊加权总分

不同版本的主排名建议使用带安全约束的字典序：

第一：必须满足collision/off-road硬门
第二：在安全门内比较success
第三：比较timeout和route completion
第四：用离线指标解释差异


避免出现：

success提高
但collision翻倍
却因加权总分仍然晋级


统一版本摘要可明确写成：

Primary = success
Hard constraints = collision, off-road
Secondary = timeout, route completion
Diagnostic = offline/path/WM metrics

三、当前代码必须修正的问题
1. 将结构化动作分支真正收编进唯一 Policy

当前运行时已经做到单一动作源，但代码仍有：

model.policy
model.path_head


两个实体，容易再次误接线。

修改要求

正式结构应统一为：

model.policy
  ├─ backbone adapter
  ├─ lateral branch
  ├─ longitudinal branch
  ├─ action distribution
  └─ auxiliary intent outputs


过渡期若不能改 checkpoint key，也至少做到：

path_head改为policy内部子模块
外部不得直接调用path_head
model.forward只暴露policy action


保留旧 Policy 仅作版本兼容和离线消融，不进入正式 rollout。

2. 统一 Policy 均值与方差来源

当前结构化模式：

action mean来自StructuredPolicyHead
action logstd来自旧PolicyHead


因此 Coding Agent 暂时移除了 action_logstd，这是正确的防错，但强化学习尚不可用。

修改要求

在同一个 Structured Policy 内定义：

action_mu
action_logstd


两者必须共享同一状态表示和同一动作参数化：

ds / κ


在本轮闭环 IL 验证中可以暂不启用采样，但应补齐接口和 fail-fast：

若RL请求logstd但结构化分支未实现
→ 直接报错

3. 接入 OD/LD 坐标补偿主路径

net/ego_warp.py 当前只是独立数学模块，尚未进入 World Model。

必须增加配置：

ego_warp_mode:
  off
  analytic


默认 off 保持旧模型逐位兼容。

analytic 模式下：

LD：
静态anchor做ego SE(2)逆变换

OD：
使用对象场景速度推进
再做ego SE(2)逆变换


关键修复：

OD场景速度
=
OD相对速度 + ego当前速度


禁止使用相对速度推进后再扣一次自车运动。

4. World Model 学习解析补偿后的 residual

主路径不能只改变图 pose，还应明确区分：

解析anchor
学习residual
最终anchor


形式：

Ak+1=Warp⁡(Ak,ak)+ΔAθA_{k+1} = \operatorname{Warp}(A_k,a_k) + \Delta A_\theta

训练分别报告：

解析先验误差
学习残差贡献
最终anchor误差


防止 latent 再次把确定性自车运动复制到 OD/LD 中。

5. 更新 imagined step 的动态场景信息

当前 step_geometry(dynamic_mode="hold") 会固定：

动态对象距离
TTC
其他纵向上下文


这使后续 imagined step 的纵向动作依赖过期动态信息。

修改要求

优先使用 World Model 的物理预测更新：

OD relative position
OD relative velocity
nearest dynamic distance
TTC


若对应预测不可用：

显式标记dynamic_valid=0


不得静默使用 t0 值并当成已更新数据。

第一版只读对照：

hold
vs
WM-updated
vs
oracle future


实测后再决定生产模式。

6. 精确区分 World Model 调用口径

统一记录：

Policy调用：6次
用于下一动作的WM推进：5次
ST-GNN总调用：6次


第六次状态预测只用于：

终端World Model监督或诊断


不能在报告中混写为“WM只调用5次”。

四、实验架构

本轮至少保留四个严格对照。

C0：旧正式基线
旧Policy
旧World Model坐标口径
固定LQR


当前可靠基线约为 success 0.56。

C1：Structured Policy + 旧 World Model

验证：

新动作表示单独贡献

C2：Structured Policy + analytic ego warp

使用：

新动作表示
OD/LD解析坐标补偿
旧WM residual能力


用于判断坐标补偿直接贡献。

C3：Structured Policy + 重训后的 residual World Model

使用：

新动作表示
解析warp
在新坐标规范下训练的World Model residual


这是本轮正式候选。

所有臂：

同一数据
同一Policy调用次数
同一World Model horizon
同一固定LQR
同一评测episodes

五、训练顺序
1. 坐标层验证

不训练，完成：

OD/LD warp接入
速度语义修复
bit-exact off模式
解析几何单元测试

2. World Model residual 训练

冻结 Policy，只训练：

ego transition
OD residual transition
LD residual transition
physical prediction heads


暂时保留跨步 detach，不启用完整六步反向传播。

3. Structured Policy 训练

冻结 World Model，训练：

Policy内部横向分支
Policy内部纵向分支
PolicyBackbone最后融合层


PolicyBackbone 最后一层已由实验确认需要适配，但使用独立低学习率，例如：

Structured Policy分支：1.0× LR
PolicyBackbone最后一层：0.1× LR


不得扩大到整个 Backbone。

4. 预测状态一致性训练

冻结 World Model：

π(zkWM)≈stopgrad⁡π(zkreal)\pi(z_k^{WM}) \approx \operatorname{stopgrad}\pi(z_k^{real})
5. 交替适配
WM-step：冻结Policy
Policy-step：冻结WM


禁止全部解冻后共同反传。

六、强制 Probes / Hooks
1. 唯一动作链

记录并断言：

Policy实例ID
Policy调用6次
WM决策推进5次
ST-GNN总调用6次
a0...a5及其输入z0...z5
辅助输出没有进入LQR

2. Warp 五项

每一步记录：

anchor_pre
anchor_analytic
anchor_actual_next
residual_learned
anchor_final

3. 动态场景更新

记录：

t0 dynamic distance/TTC
WM-updated distance/TTC
oracle distance/TTC
longitudinal branch实际消费值
valid mask

4. Policy 增益链

记录：

WM latent变化
PolicyBackbone变化
横向κ变化
纵向ds/v变化
最终六步plan变化

5. 连续路径与 LQR

记录：

六步plan
30点reference
连续footprint扫掠
首次越界/碰撞点
实际轨迹
raw steer/throttle
跟踪误差

6. 梯度和参数账本

输出：

trainable参数
optimizer分组
学习率
grad norm
update norm
冻结参数grad=0
checkpoint hash

七、验收准则
Gate A：代码契约
off模式旧模型逐位一致
动作生成器数量=1
正式路径无oneshot/plan_override
Policy调用6次
WM决策推进5次
辅助输出不能进入LQR

Gate B：坐标补偿
静态LD warp误差接近数值精度或显著优于旧先验
同速前车相对距离误差显著下降
OD/LD ego-motion泄漏下降 >=30%
左右镜像正确率 >=95%


不能只用全 OD 集平均，必须单独报告：

恒速对象
同速前车
加减速对象
cut-in/merge对象

Gate C：World Model
h1...h6 lane-relative误差不指数增长
C3优于C2和C1
WM tail优于frozen-state tail
h3...h6 Policy接口误差下降 >=20%

Gate D：离线路径指标

仅作晋级前提：

车道保持路径越界不劣于专家同口径地板
给定目标变道收敛显著优于旧基线
重复换道不劣于专家同口径
曲率与曲率变化率可执行


所有“投影边缘行”必须预先定义过滤规则，不能在看到结果后剔除。

Gate E：统一闭环指标
Calib50

候选必须同时满足：

success >=0.62
off-road <=0.10
collision <=0.08
timeout <=0.10
route completion不下降

Clean150
success >=0.60
相对C0提升 >=5pp
off-road和collision不恶化
普通道路、弯道、merge至少各有正向

Eval500

只有 Clean150 过门才运行：

3 seeds
paired bootstrap
McNemar翻转检验
与IDM + Pure Pursuit统一口径

八、给 Coding Agent 的正式开工指令
Round-36离线结果只作为闭环候选筛选，不作性能结论。

立即进入Round-37：
Structured Policy + analytic OD/LD ego warp
+ residual World Model + fixed LQR闭环验证。

必须完成：

1. 将StructuredPolicyHead收编为唯一Policy内部模块；
2. RL方差接口同源化或fail-fast；
3. ego_warp以off/analytic配置接入model主路径；
4. 修复OD速度口径：v_world = v_rel + v_ego；
5. World Model预测并更新imagined step动态距离/TTC；
6. 在新坐标规范下训练World Model residual；
7. 冻结WM训练Structured Policy；
8. real/WM latent决策一致性训练；
9. WM-step与Policy-step交替适配；
10. 跑C0/C1/C2/C3四臂的Calib50闭环配对评测。

统一主指标：
success；
硬约束：
off-road、collision；
次指标：
timeout、route completion；
离线路径与WM指标仅用于归因和晋级解释。

在Calib50达到：
success>=0.62、
off-road<=0.10、
collision<=0.08、
timeout<=0.10
之前，不进入Clean150、DAgger、MoE或RL。

最终判断

Round-36 的离线结果和代码契约是有价值的，但现在必须把项目重新拉回终极目标：

任何结构改动都必须最终证明它提高了 Policy–World Model × 6 → 固定 LQR 的闭环成功率，而不是只改善离线曲率、越界或收敛指标。

下一步要同时完成坐标层、World Model residual 和 Structured Policy 的系统集成，并用统一的 C0/C1/C2/C3 闭环矩阵裁决。