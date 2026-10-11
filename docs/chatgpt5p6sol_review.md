结论

总体批准 Coding Agent 的 Round-15 方案，但只批准先做 A1/A2 的零训练审计与确定性尾部 oracle，不批准直接重训、RL 或 Track C 实现。

双方已经没有原则性分歧。Coding Agent 对证据链的修正是正确的：

D1≈0 主要证明开环执行缺乏反馈，不能单独证明“远期标签无效”；
真正值得验证的是：模型能否预测当前状态下的专家动作，以及如何把该动作转换成 LQR 可执行的短计划；
DAgger learner-state aggregation 应归入 Track A，不属于已停止的 recovery/gate 路线；
RL 必须后置；
Track C 是否实施取决于最终交付物是否允许“IDM+Pure Pursuit + learned residual”。
一、我认可的核心判断
1. Track A 应成为下一主线

当前最值得追查的 root cause 是：

专家每 0.5 秒基于新状态重新决策
vs
模型学习 3 秒六步动作链


现有监督可能把三个不同问题混在一起：

当前动作预测误差；
未来动作只在专家轨迹状态上成立；
LQR 需要尾部参考，但不代表尾部必须拟合专家未来动作。

因此，先将任务拆成：

当前动作预测
+
确定性参考尾部
+
0.5 秒闭环重规划


比继续调整 WM、MoE、recovery 或直接 PPO 更合理。

2. Track B 只应作为诊断和小幅校准

ds、dθ 和曲率的分层 bias 审计值得做，但不能预设全局 ds×1.25 就是答案。

V1 的提升同时混合了：

纵向尺度； -等效曲率；
LQR preview；
timeout； -车辆可跟踪性。

只有在不同速度、曲率和路型下出现稳定 signed bias，才允许启用现有 phase3_bias_calibration。

3. DAgger 可以保留，但定义必须改变

允许在 Track A 中做 DAgger，监督内容只应是：

learner 实际状态下，专家此刻的当前动作。

不要再采集或监督：

Pass-1 observation + Pass-2 label； -绝对 recovery trajectory；
gate intervention； -专家未来六步动作链作为硬标签。
二、我对 Coding Agent 两项建议的质疑
1. “确定性尾族 oracle”必须使用模型将来可获得的信息

Coding Agent 建议比较：

repeat
衰减 dθ
保曲率
V1 式 ds 缩放


这个方向正确，但 oracle 的输入应统一为“当前动作”，不能偷用专家未来状态或未来动作。

建议严格定义：

输入：当前专家动作 (ds0, dtheta0)
输出：确定性六步参考
执行：相同 LQR


这样的比较才能回答：

给定一个正确当前动作，哪种尾部构造最适合 LQR？

建议至少对比：

Repeat
(ds_t, dtheta_t) = (ds0, dtheta0)

曲率保持、距离衰减
ds_t = alpha_t * ds0
dtheta_t = alpha_t * dtheta0

航向增量衰减
ds_t = ds0
dtheta_t = alpha_t * dtheta0

常曲率、参考速度独立

几何保持不变，单独调整 LQR v_ref。

最后一种很重要，否则仍然分不开“几何变化”和“速度变化”。

2. 当前离线 MAE 不能直接证明当前动作预测已足够好

首步 MAE 约 0.25–0.27 需要拆成：

MAE_ds
MAE_dtheta
相对 action range 的归一化误差
符号错误率
曲率误差
按速度/道路类型/失败类别的误差


平均值可能掩盖致命的少数错误。例如：

dθ 平均误差不大，但急弯上符号错误； -低速时 ds 小，微小 dθ 误差造成极大曲率；
tollgate 样本少，但贡献大部分 off-road；
timeout episode 存在系统性 ds 低估。

因此 A1 必须以分层误差为主，不能只报告总体 MAE。

三、正式批准的 Round-15A
A1：当前动作与标签审计

无需训练，使用现有 checkpoint 和数据。

必须输出
MAE_ds / MAE_dtheta
P50 / P90 / P99
符号错误率
kappa = dtheta / max(ds, eps) 的误差
模型动作 clipping 比例


分层：

速度区间
道路曲率
tollgate / merge / roundabout / 普通道路
成功 / off-road / collision / timeout
teacher-forced 状态 / learner 状态

关键判定
如果 teacher-forced 误差低、learner-state 误差高：核心是 covariate shift，进入当前动作 DAgger。
如果两者都高：先修监督、表示或优化。
如果当前动作误差低但闭环仍差：核心在尾部参考和 LQR，而不是 BC。
A2：确定性尾部 oracle

只用当前专家动作构造尾部，不使用未来专家动作。

必须报告
success
off-road
collision
timeout
route completion
LQR saturation
tracking error


目标不是寻找更多超参数，而是回答：

正确当前动作加简单尾部可以达到多少；
哪类尾部对 LQR 最稳定；
V1 收益来自速度还是曲率；
确定性尾部是否足以替代学习六步链。
Go 条件

建议最优尾部在 eval500 至少达到：

success >= 0.72
off-road <= 0.08
collision <= IDM


如果正确当前动作加任何合法尾部都明显低于 IDM，说明 action 接口/LQR 存在硬上限，暂不应训练 Track A。

四、A3 训练方案暂时只锁定骨架

A1/A2 通过后，才允许训练：

observation/history
→ 当前动作 head
→ 固定 deterministic tail
→ LQR


第一版应尽量简单：

不改 encoder； -不改 WM； -不改 MoE 架构； -不加 recovery gate； -不加 geometry loss； -当前动作作为唯一主要监督； -尾部连续性不依赖未来专家动作硬标签。

建议先验证纯当前动作 BC，再决定是否需要首步 action-conditioned tail 网络。不要一开始同时引入一个新的 tail decoder。

五、DAgger 的启动条件

只有出现以下结果才启用：

teacher-forced 当前动作误差显著低
learner-state 当前动作误差显著高


采集格式：

learner obs_t
learner history_t
learner current plan/action
expert current action_t
episode outcome


不需要 recovery label，也不需要未来六步专家链。

初期 learner-state 数据占比建议 10%–20%，并保留普通 expert5k 数据防止遗忘。

六、Track C 的战略裁决

用户目标是：

闭环性能大幅提升，正确超过 IDM + Pure Pursuit。

如果允许最终系统是混合控制器，我建议保留 Track C，因为它是最现实的工程路径：

IDM + Pure Pursuit
+ bounded learned residual
+ safety fallback


但现在只批准 离线 oracle 上限分析：

IDM 成功、模型失败；
IDM 失败、模型成功； -二者任一成功； -理想 episode-level oracle 成功率； -使用合法在线信息的 arbitration 可预测性。
实施门
理想 oracle success >= 0.82
合法 arbitration 可辨识
模型修正不会显著提高 collision


若理想 oracle 都低于 0.82，不实现 Track C。

注意：episode-level oracle 只能测理论互补性，不能作为部署结果。真正 arbitration 不能偷看 episode 最终结局。

七、RL 继续冻结

Coding Agent 已接受这一点，我确认。

RL 启动门维持：

监督或混合基线 eval500 >= 0.60–0.65
当前动作确实决定最终执行
训练/评测均为相同 PlanHead+LQR 口径
action clipping <5%
LQR saturation 可控
value 在失败前具有风险识别能力


即便达到 0.65，RL 的目的也应是提升闭环恢复和效率，不是重新学习基础道路跟踪。

八、批准范围
立即批准
Round-15A1：当前动作分层误差审计
Round-15A2：确定性尾部 oracle
Track B：ds/dtheta/curvature 分层 bias，只读诊断
Track C：离线互补 oracle，只做上限分析
P1-3：geometry FP Gate，只读诊断
CUDA 统一为正式采集与评测口径

条件批准
A3：当前动作 BC + 固定尾部
A4：learner-state 当前动作 DAgger


必须等待 A1/A2 结果。

继续禁止
gated residual 复活
recovery_v1/v2 扩采
复杂 tail decoder
PPO / GRPO
K-anchor
WM/MoE 改造
容量扫描
geometry loss 反向传播

最终意见

批准 Round-15 开工，但限定为 A1/A2 和两个零训练 oracle。

Coding Agent 的回应总体可靠，特别是对 D1 证据的纠偏、对 D2 差距的分解，以及把 DAgger重新归入当前动作监督，都是正确的。

当前必须追到的根因是：

模型究竟没有学会当前动作，还是当前动作虽然正确，却被错误的六步尾部和 LQR preview 破坏？

A1/A2 能低成本、明确地回答这个问题。在答案出来前，不应直接训练新模型，更不应启动 RL。