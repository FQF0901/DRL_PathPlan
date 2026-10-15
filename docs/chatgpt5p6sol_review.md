审阅结论

这次提交主要是文档纠偏，不是功能实现进展。积极的一面是 Coding Agent 已正式承认 PathExecutionHead 偏离路线 A，并接受“唯一 PolicyHead、逐步 Policy–WM 自回归”的架构契约；但附件里没有 net/model.py、net/policy.py、World Model 或训练代码的实际 diff，因此目前只能批准“方向纠正”，不能认定重构已经完成或可以开始正式训练。

一、这次正确纠正了什么？

Coding Agent 已认账原 PathExecutionHead 的三个结构性问题：

1. 它事实上是第二个轨迹生成器；
2. 它通过 plan_override 一次产生六步计划；
3. 后五步绕过了 Policy → World Model → Policy 的逐步自回归。


同时接受改为：

唯一 StructuredPolicyHead
  ├─ lateral branch
  ├─ longitudinal branch
  └─ deterministic composition

每一步：
z_k → 同一个PolicyHead → (ds_k, κ_k)
dθ_k = κ_k × ds_k
→ World Model → z_(k+1)


并明确 κ_mid/κ_far 和未来 progress 只能作为辅助监督，不能直接接入 LQR。这个方向完全符合我们的总守则。

二、当前进展不能被高估

附件只修改了：

docs/deepseekv4p1_argue.md


没有看到以下关键文件的代码变更：

net/model.py
net/policy.py
net/st_gnn.py
pipeline/stages.py
pipeline/trainer.py
docs/architecture_contract.md


因此当前尚未证明：

plan_override 已从正式 rollout 删除； -旧 PolicyHead 是否被真正替换或收编； -六步动作是否全部来自同一 PolicyHead； -每一步是否读取对应的 World Model 状态； -辅助未来输出是否无法进入 LQR； -训练器是否只优化唯一 Policy 内部的新分支；
World Model 和 OD/LD 坐标解耦是否已经接线。

所以当前状态应标记为：

ARCHITECTURE_DIRECTION_ACCEPTED
IMPLEMENTATION_NOT_YET_VERIFIED


而不是“重构完成”。

三、我与 Coding Agent 仍有一个关键分歧

旧版 §5.56 曾主张：

先用冻结旧World Model闭合Policy门，
结构化World Model只做零训练契约层，
是否重训等Gate 2实测触发。


这部分虽然在本次 diff 中被删除，但必须防止它以实施策略继续存在。

你已经明确，车道保持是完整系统能力：

StructuredPolicyHead
+
World Model
+
OD/LD ego-motion坐标解耦
+
六步递归
+
固定LQR


因此不能用：

T1在冻结旧WM下接近某个专家越界地板


就推断 World Model 暂时不需要参与。

理由很简单：

T1 只能证明某个离线或短期路径指标有所改善； -不能证明后五步 predicted-state 仍保持 lane-relative 信息； -不能证明新 Structured Policy 在 predicted latent 上与真实 latent 决策一致； -不能证明 OD/LD 的坐标变化已经从 latent 中解耦； -更不能证明固定 LQR 下的完整部署闭环已过门。

结论：结构化 World Model 不应被推迟到所有 Policy 能力之后。

四、下一步必须先交付的代码，而不是继续写原则文档
1. 唯一 Policy 契约落地

必须提交实际代码 diff，证明：

旧PathExecutionHead不再接管正式rollout
plan_override不能出现在训练或部署主路径
StructuredPolicyHead每次只输出当前一步
a0...a5来自同一个PolicyHead实例
World Model在六步链中实际执行5次


建议加入运行时断言：

assert policy_instance_ids_are_identical
assert world_model_call_count == 5
assert policy_call_count == 6
assert auxiliary_future_outputs_not_consumed_by_lqr
assert not deployment_uses_plan_override

2. architecture_contract.md

必须明确写出：

唯一动作生成模块是谁
PolicyBackbone职责是什么
StructuredPolicyHead内部子分支是什么
每次forward输出什么
辅助输出能进入哪些loss
辅助输出不得进入哪些部署路径
World Model输入输出是什么
六步rollout调用图
LQR最终消费什么
训练与部署是否同构

3. OD/LD 坐标解耦最小实现

不能仅停留在文档里。至少实现并保留旧路径对照：

LD：
静态anchor按ego SE(2)逆变换

OD：
对象恒速先验
→ ego SE(2)逆变换

World Model：
只学习warp后的残差


第一版暂不扩容，也不做六步 BPTT，但必须有：

warp前anchor
解析warp后anchor
真实下一帧anchor
learned residual
最终预测anchor


五项 hook。

五、训练顺序
阶段 1：坐标和架构契约

不训练，完成：

-唯一 Policy 调用链；

OD/LD anchor schema；
ego-motion warp； -旧路径与新路径的等价性、镜像和物理单测； -hooks 默认关闭 bit-exact。
阶段 2：World Model 车道保持状态转移

冻结 Policy，训练：

ego transition
OD residual transition
LD residual transition
physical heads


先保持跨步 detach，不启用 BPTT。

阶段 3：Structured Policy 当前动作

冻结 WM，只训练唯一 PolicyHead 内部：

lateral branch
longitudinal branch
deterministic composition无参数


不预先开放 PolicyBackbone。

阶段 4：预测状态鲁棒训练

冻结 WM，训练：

π(zWM)≈sg⁡π(zreal)\pi(z^{WM})\approx \operatorname{sg}\pi(z^{real})
阶段 5：交替适配
WM-step：冻结Policy
Policy-step：冻结WM


不得全解冻、单一总损失反传。

六、强制验证准则
架构门
动作生成器数量 = 1
Policy调用次数 = 6
World Model调用次数 = 5
每个a_k读取对应z_k
没有plan_override正式路径
辅助未来预测不能进入LQR

坐标门
静态LD warp误差 P90 <= 0.10m
LD heading误差 P90 <= 1°
同速前车相对距离误差 P90 <= 0.20m
左右镜像正确率 >= 95%
OD/LD中的ego-motion泄漏下降 >= 30%


阈值可用基线样本校准一次，之后锁定。

World Model 门
h1...h6 lane lateral/heading误差不指数增长
WM tail优于frozen-state tail
h3...h6 Policy接口误差下降 >=20%
物理预测优于静态/恒速基线

Policy 门
真实状态动作误差退化 <=5%
预测状态与真实状态动作差下降 >=20%
曲率方向正确率 >=95%
纵向进度单调率 >=90%

六步和闭环门

原先“连续越界率 ≤2%”若低于同口径专家数据地板，应改为相对门：

显著优于旧基线
且不劣于专家同口径分布 + 统计余量


最终普通道路固定 LQR：

success >=0.90
collision接近0
timeout <=0.05
off-road显著低于现有基线
计划错误与LQR跟踪错误分别报告

七、给 Coding Agent 的正式回复
接受对PathExecutionHead结构性偏离的认账，
以及重构为唯一StructuredPolicyHead、逐步Policy–WM自回归的方向。

但当前提交仅为文档更新，不代表实现完成。
正式训练继续暂停，先提交代码与运行证据：

1. 删除正式路径中的plan_override；
2. StructuredPolicyHead每次只生成当前一步(ds, kappa)；
3. a0...a5由同一Policy实例读取z0...z5逐步生成；
4. WM实际推进5次；
5. 辅助kappa/progress不得直接进入LQR；
6. 完成architecture_contract.md；
7. 接入OD/LD显式anchors和ego-motion解析warp最小路径；
8. 添加Policy调用、WM调用、anchor warp、residual和LQR输入断言；
9. hooks关闭时bit-exact；
10. 提交旧链/新链的调用图、参数账本和梯度账本。

结构化WM不能整体后置。
车道保持必须以：
StructuredPolicyHead + structured WM + ego-motion compensation
+ 六步递归 + fixed LQR
作为完整系统验收。

在架构门、坐标门和调用链门通过前，
不得开始正式主训练。

最终判断

方向已经纠正，可以继续开工；但下一回合必须看到代码、运行时调用链和断言结果，不能再以文档“接受原则”代替真实实施。当前最有价值的动作是把唯一 Policy 与结构化 World Model 的契约真正落到代码主路径中。