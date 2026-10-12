总体判断

这轮结果非常重要。deepseekv4p1_argue.md 已经把问题从“训练配方不对”进一步推进到“道路几何表示和网络消费方式可能存在结构缺陷”。

我认可这些事实：

A3 证明，简单重复模型当前动作会导致转向抖动被 LQR 预瞄放大，迅速出界；六步 chain 的现实作用更像执行噪声缓冲，而非纯远期预测。
learner-state 当前动作误差仍然很大，纯 expert-state BC 没有修复 covariate shift。
LD 槽位链中约 76.5% 的相邻边跨 lane，当前 LD-LD message passing 的拓扑语义高度可疑。
PlanHead 只消费 LD 全槽均值，可能把 target lane、相邻 lane 和复杂分叉信息混在一起。
模型对 LD 平移、旋转的动作响应非常弱，而清空 history 的影响明显更大，说明策略更依赖历史动力学惯性，几何纠偏能力不足。

但是，我不同意现在就把根因定案为：

“LD 伪边导致闭环失败，关闭邻接并加入显式池化即可解决。”

当前证据支持“高风险嫌疑”，还没有完成因果证明。

一、需要收紧的三个结论
1. “76.5% 伪边”不等于“这些边一定有害”

槽位相邻不等于 lane 相邻，这确实是结构错误；但 message passing 也可能把它当普通 token mixing 使用，而不是严格拓扑推理。

必须通过同预算重训消融证明：

现有槽位链
vs
完全关闭 LD-LD 边
vs
排列不变的 LD set attention


如果关闭槽位链以后闭环不改善，说明问题不只在边；如果 set attention 明显提升，才能确认“错误拓扑 + 均值池化”是主要原因。

因此，“伪边率 76.5%”是架构缺陷证据，不是闭环因果证据。

2. “几何响应弱约 50 倍”目前不是可靠量纲

模型对 LD 平移 ±1m 只改变约 0.002 rad 的 μ_dθ，确实偏弱。但“合理响应应该大 50 倍”缺少明确基准，因为 policy action 的 0.5 秒语义、LQR preview 和速度共同决定最终横向运动。

更严谨的口径应是：

与 expert 对同样几何扰动的动作响应比较；
与 oracle/current-action controller 比较；
检查响应方向、单调性、局部增益和闭环恢复能力；
不直接给出“应为多少倍”的绝对判断。

建议将结论改成：

模型对局部道路几何扰动的动作敏感度，显著低于 expert/oracle 所需的闭环纠偏敏感度。

3. zero_history 影响大，不足以证明“动力学 shortcut”

将所有历史清零是强 OOD 干预，既删除动力学信息，也改变输入分布。输出变化大不一定说明模型错误依赖历史。

需要增加更温和的对照：

真实历史
复制当前帧六次
历史逆序
历史 shuffle
只保留最近 2 帧
LD 保持不变但 ego history 改变
ego history 保持不变但 LD 当前帧改变


最关键的是：

真实六帧 vs 当前帧复制六次


如果两者输出和闭环接近，memory 没利用动态信息；如果差异明显且方向合理，history 依赖本身不是问题。真正的问题可能是几何分支影响不足以覆盖动力学先验。

二、同意下一步先做“模型侧修复”，但不要一次改三件事

Coding agent 建议：

关闭或重建 LD-LD 邻接；
显式池化替换全槽均值；
可选几何响应辅助监督。

方向正确，但不能组合成一个实验，否则即使提升也无法知道原因。

建议严格拆成四个等预算臂。

M0：基线复训

使用当前代码、相同数据、相同训练预算重新训练一次。

目的：

控制代码漂移； -得到同批环境下的训练随机性参考； -不能只拿历史 pri512 checkpoint 与新架构横比。
M1：只关闭 LD-LD 槽位链

保持：

encoder； -所有池化；
PlanHead；
loss； -参数预算；

尽量不变。

只移除错误的 LD-LD slot adjacency，保留 ego↔LD 或其他合法交互。

这回答：

槽位伪边本身是否有害？

M2：LD 改成 permutation-invariant set attention

不要立即做“target/adjacent/boundary 显式池化”，因为当前数据缺少可靠 lane identity、route association 和 target-lane flag。

第一版更安全的是：

query = ego + nav + route context
keys/values = all valid LD tokens
output = masked cross-attention / set pooling


特点：

不依赖槽位顺序； -不用假定相邻 slot 在同一 lane； -可根据 ego/nav 查询相关 lane geometry； -无需重新采集数据； -比全槽均值有更强选择性。

M2 同时关闭 LD-LD 槽位链，避免 set attention 之前 token 已被错误拓扑污染。

M3：M2 + 几何响应辅助监督

这一臂暂缓，除非先证明监督标签可靠。

现有 ld_line_margin_loss 已经失败，不能换个名称重新加入。若做辅助监督，只允许使用经契约验证的物理量，例如：

当前 ego lane-relative lateral offset；
heading error； -局部道路曲率；
target-lane existence；
LD token 与 ego/nav 的关联分类。

如果这些标签仍来自不稳定的 lane assignment，则 M3 不应实施。

三、显式池化方案应避免新的错误语义

Coding agent 提议 target/adjacent/boundary pooling，但当前最小语义特征尚未提供：

target_lane flag
lane_id
offset index
route association
lane-relative lateral offset
heading difference


所以当前数据上不能可靠地区分：

target lane； -相邻 lane； -道路边界； -收费站不同通道。

不要在缺少标签时用几何启发式强行分组，这可能制造新的隐式错配。

推荐当前数据可直接支持的版本
LD tokens
→ independent token encoder
→ ego/nav-conditioned cross-attention
→ attended geometry context
→ PlanHead


保留每个 token 的原始几何，不做 slot-chain message passing，也不做简单 mean。

数据侧 v7 特征何时启动

只有满足以下任一条件才重采：

M2 明显提升，但仍在 lane transition / tollgate 处失败；
attention 权重无法稳定选中 ego/route 相关 LD；
target-lane 缺失的 4.9% 样本对失败贡献明显高于其频率； 4.同路型成功/失败匹配分析确认 lane assignment discontinuity 是领先指标，而不只是共现。

否则先不要承担全量重采和 fingerprint 迁移成本。

四、需要补一个真正的因果分析

当前 lane-assignment 断点：

失败集 0.5146
成功集 0.0363
失败前 20 步 127/127 出现


相关性很强，但复杂路口本来就同时更容易：

lane slot 重排；
route topology 变化；
off-road。

建议做 matched-case 分析：

相同 road class
相同速度带
相近 route progress
相近曲率
相同 map / 相邻 episode spec


比较成功和失败样本的：

lane assignment discontinuity；
LD token jump；
attention/pooling 输出 jump； -μ_dθ jump； -首次不可恢复时间。

如果匹配后差异仍显著，才更接近因果证据。

此外，对相同 observation 做反事实：

只重排 LD slots，不改变 token 内容


理想架构的输出应近似不变。当前架构如果对合法 permutation 极敏感，这是直接、强有力的结构缺陷证据。

这个 slot-permutation invariance probe 应在重训前加入。

五、下一轮训练的验收不能只看 success
结构性 probe Gate

M1/M2 至少应做到：

LD slot permutation 后输出变化显著下降
LD 平移/旋转响应方向正确且单调
几何响应强于当前基线
历史轻扰动不会造成异常跳变
lane transition 处 plan jump 降低

clean150 Gate

相对同批 M0：

success 至少 +5pp
off-road 至少下降 10pp
collision 不增加超过 2pp
timeout 不增加超过 3pp
steer saturation 明显下降


只提升 1–2pp 不足以支撑全量数据重采。

eval500 Gate

只有 clean150 通过，才跑 eval500。

建议：

success >= 0.45：证明结构修复有价值
success >= 0.55：继续进入 learner-state DAgger
success >= 0.60：重新评估 RL 启动路径


当前基线约 0.33，因此这轮期待的是结构性提升，不是噪声级领先。

六、A4 DAgger 暂时继续冻结

虽然 learner-state covariate shift 已被证明，但现在发现 observation→action 的几何通路本身可能不健全。

如果先做 DAgger，模型可能：

-用更多 learner-state 数据强化历史动力学 shortcut； -继续无法理解 lane-relative correction； -在训练集上拟合 learner action，闭环仍不稳定。

因此顺序应为：

修复 geometry consumption
→ 验证几何响应
→ 再做 learner-state 当前动作 DAgger


A4 的 collector 和数据契约可以保留，但现在不要采集和训练。

七、MoE 与 World Model 的新判断
MoE

primary_only 与 full 结果基本相同，说明 experts 当前没有贡献明显的几何纠偏。

继续冻结正确。不要在这轮重训 experts 或 router。

World Model

这轮问题主要发生在当前几何消费和当前动作响应上，暂时不需要改 WM。

但新架构必须验证：

PlanHead 是否真正使用新的 geometry context；
WM rollout 是否又把新几何信号稀释掉；
chain tail 是否继续提供执行平滑。

因此建议保留现有 chain 执行形式，仅替换其几何 context 输入，不重新设计 WM。

八、批准范围
立即批准
Round-20A：slot permutation invariance probe
Round-20B：成功/失败匹配的 lane-transition 分析
M0：当前模型同预算复训
M1：仅关闭 LD-LD 槽位链
M2：关闭槽位链 + ego/nav-conditioned LD set attention


第一阶段可先训练 M0/M1；M2 实现后并行，但每个变更必须独立记录。

条件批准
M3：几何辅助监督
数据侧语义特征重采
A4 learner-state DAgger


这些需要 M1/M2 结果支持。

继续禁止
PPO / GRPO
gated residual 复活
geometry proxy 反向传播
MoE/router 改造
WM BPTT
容量扫描
一次性组合“新拓扑+新池化+新特征+新 loss”

最终判断

Coding agent 已经找到当前最有价值的突破口：

模型不是缺少 LD，而是用错误拓扑聚合 LD，并把几何压缩成无选择性的均值，导致当前动作主要依赖历史动态而非道路几何。

这个判断值得推进，但仍需通过 slot permutation probe、matched-case 分析和 M0/M1/M2 独立消融完成因果闭环。

因此我的裁决是：

批准先做模型侧修复，但不批准直接加入语义特征和几何辅助监督。先证明移除伪拓扑、改用排列不变的条件池化能带来显著闭环收益。

若 M2 连 clean150 0.45 都达不到，应停止继续精修 LD 架构，重新评估 observation 本身是否缺少可恢复的道路语义，并转向数据侧重采。