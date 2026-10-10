我审阅了 coding agent 最新答复。整体判断是：**P1 v0.3 已经接近可执行，可以接受“三批 probes”裁剪，也可以接受 recovery 配方和 action-conditioned plan 方向。但正式开工前仍需修正 5 个实现细节，否则可能产生新的伪结论。**以下完全基于 deepseekv4p1_argue.md 中记录的代码核查和 P0 实测结果。

一、同意并锁定的事项
1. Probes 分三批是合理的

21 项全部前置会拖慢主线。coding agent 选择以下 A 批立即实施，覆盖了最危险的语义错误：

checkpoint 加载比例；
plan/action 契约；
时间索引；
plan/reference/actual 三轨；
LQR 饱和；
progress/timeout；
recovery 标签质量；
loss 梯度与 optimizer 更新账本。

这批足以保护 P1，不必等 B/C 批全部完成。

2. Recovery 不能继续 specific_only

新增 recovery freeze recipe，训练 PlanHead fusion/trunk、primary 和有限的 experts/residual，同时冻结 encoder、mem、WM、router，方向正确。现有代码已经有 specific_only / trunk_only / all 三种范围，因此新增第四档工程风险可控。

3. Action-conditioned plan 作为首个 PPO 方案

保持 2D stochastic action 与现有 PPO log-prob、buffer、KL 兼容，并让采样 action 强制成为 plan 首步，再条件生成后五步，这是当前代码下最稳妥的信用链。12D 独立高斯暂不做。

4. 起点模型补测后再定

先跑 rou64@e015 的 eval500，并与 pri512@e020 做逐 episode paired，再选择 P1 起点。不能只凭 clean500 的 0.370 直接决定。

二、开工前必须修正的 5 点
1. “从 LD 直接构造几何 loss”只能视为近似，不是真实 corridor loss

当前 LD 是稀疏车道线点集，并不能完整表达：

drivable surface；
复杂收费站和分叉的可行域； -完整道路边界；
route corridor 宽度； -任意未来 pose 下的合法区域。

因此 P1-3 应明确拆成：

可微代理 loss

由 obs 中 LD 构造：

到可见实线的最小距离； -跨线侧符号变化；
plan 与局部车道方向夹角； -近场线段 margin。
引擎真值

通过 rollout/forensics 计算：

footprint 是否在 drivable surface；
termination 同口径的连续线判定； -真实 route corridor violation；
hard mining 权重和验收指标。

**不能把 LD-based proxy 命名为 drivable_footprint_loss，建议命名为 ld_line_margin_loss。**否则后续容易误以为网络已经接受完整道路边界监督。

另外，必须加入 LD 可用性 mask。收费站或拓扑变化处 LD 稀疏、槽位突变时，不能强行产生错误梯度。

2. D2 配速 oracle 不能只缩放 ds

coding agent proposes --d2-ds-scale。如果只放大 ds 而保持 dθ 不变，则：

κ=dθds\kappa = \frac{d\theta}{ds}

会下降，相当于同时改变速度和路径曲率。这测到的不是纯“配速上限”，而是“速度与几何共同变化”。

建议至少测试三种版本：

V1: ds *= s, dtheta 不变
V2: ds *= s, dtheta *= s，近似保持曲率
V3: 只修改 LQR reference speed，不修改几何 plan


其中 V3 才最接近纯配速实验。若当前 LQR reference speed 直接由 ds/0.5s 推导，则需要增加独立 speed override，而不是只改 plan 几何。P0 已证明 d2 的 118 个 timeout 多数 route completion 很高，因此这个实验值得做，但必须避免速度和曲率混杂。

3. “P1 冻结 MoE”与 recovery 配方中训练 experts 存在表述冲突

文档一处写 P1 不动 MoE，另一处建议 recovery recipe 训练 experts/residual。

建议明确定义：

不改 MoE 架构、专家数、router 和负载均衡机制；
recovery 阶段可以训练现有 experts/residual；
router 保持冻结；
primary 与 experts 使用不同 LR。

建议首个消融只做两档：

R1: fusion + primary
R2: fusion + primary + experts/residual


如果 R2 没有显著优于 R1，则 recovery candidate 不训练 experts，避免专家吸收共享几何问题。

4. Recovery 样本不能按“失败前固定 2–4 秒”机械截取

P0 显示 T_plan 可能在第 120 或 295 个策略步才出现，且失败形成是渐进漂移。

更合理的采样锚点是：

start = max(
    T_plan - context_window,
    T_cross - recovery_window,
    first_low_margin_step
)


建议 recovery 数据分三类：

prevention：首次 margin 下降但仍完全可恢复；
correction：已偏离中心但仍在合法区域；
late recovery：接近或已经压线。

训练配比建议优先：

prevention 50%
correction 35%
late recovery 15%


因为 late recovery 常存在多解、专家急动作和不可恢复状态，比例过高可能污染正常驾驶。

每条 recovery 必须带：

expert_recovered
recovery_success_horizon
min_margin_after_takeover
route_progress_after_takeover


专家接管后仍失败的样本不能作为正监督。

5. 几何 loss 的梯度对齐不能只看 norm

coding agent 建议各项 gradient norm 控制在约 3–5 倍内，这可以作为起点，但不够。不同 loss 的梯度可能大小相近、方向却互相冲突。

应额外记录至少三组 cosine：

cos(grad_traj, grad_geometry)
cos(grad_traj, grad_curvature)
cos(grad_progress, grad_geometry)


若长期 < -0.3，说明目标存在明显冲突。例如 geometry loss 可能通过减小 ds 避免压线，而 progress loss 要求加速，从而产生 off-road→timeout 的失败置换。

同时在固定 probe batch 上记录：

plan margin；
progress；
curvature； -第一步 action error； -六步 pose error。

不能仅根据梯度范数调权重。

三、对 probes 计划的具体裁决
A 批：全部批准，但补两个项目

除 coding agent 已列的 8 项外，建议加入：

A9：freeze recipe 审计

每个阶段输出：

requires_grad parameter count
optimizer parameter count
nonzero-gradient parameter count
actually-updated parameter count


尤其验证新增 recovery recipe：

encoder/mem/WM/router 更新量严格为零；
fusion/primary 确实更新；
experts 是否按 R1/R2 配方变化。
A10：geometry proxy coverage

记录每批次：

LD geometry valid ratio
solid-line available ratio
plan points covered by LD ratio
loss-active sample ratio


否则 ld_line_margin_loss 可能只在少数简单路段生效，但训练日志看起来正常。

B 批：同意排在首候选之前

尤其重要的是：

BC action chain 与跨行 ego_world 对齐；
history 反事实； -坐标 round-trip；
WM action sensitivity；
value calibration。

其中我建议将 跨行 ego_world 对齐提前到 P1-1 数据采集前。因为 recovery 轨迹将依赖跨时间 pose，如果这里仍有时间索引或坐标误差，新数据会整体被污染。

C 批：同意推迟到 P2

Encoder probe、WM horizon、MoE 路由等不会阻塞当前 plan-side 修复，可以后置。

四、收敛后的 P1 方案
P1-0：证据和契约
rou64@e015 跑 eval500，与 pri512 paired；
D2 配速执行 V1/V2/V3，优先看 V3；
完成 A 批 probes，加 freeze audit 和 geometry coverage；
完成跨行 action/ego pose 对齐；
固化 recovery schema 和质量门。
退出条件
起点模型明确；
配速后的 teacher ceiling 有数字；
action/plan/time/optimizer 契约全部 PASS；
recovery 标签能验证专家确实恢复。
P1-1：Recovery 数据

存储：

student obs/history
student plan
actual trajectory
expert recovery action/plan
surface/legal-line/route labels
T_plan/T_cross
recovery outcome
road class


按 prevention/correction/late 分类采样。

P1-2：Recovery BC

首轮两档：

R1 = fusion + primary
R2 = fusion + primary + existing experts/residual


共同设置：

router、encoder、mem、WM 冻结；
recovery batch 占 20%–30%；
时间衰减六步监督； -普通 BC anchor 防遗忘； -拒绝合成 traj6。

先用 clean150 做 paired 筛选，再将胜者跑 eval500。

P1-3：几何/曲率/进度联合 recipe

使用：

ld_line_margin_loss，而非声称完整 drivable loss； -引擎真值用于 hard mining 和审计； -低速 curvature mask； -主罚 Δκ 和不可达曲率；
progress/timeout loss；
gradient norm 与 gradient cosine 联合校准。

必须报告失败类型是否从 off-road 转成 timeout 或 collision。

P1-4：Action-conditioned plan PPO

启动门槛保持：

监督模型至少达到 0.60； -最好达到 0.65；
action-conditioned plan 的因果探针通过； -采样 action 确实改变最终执行 plan；
PPO log-prob 只对应实际进入执行链的 2D action。

不满足门槛则不进入 PPO。

五、教师上限的最终判决规则

当前 D2=0.706 不足以证明可以靠监督超过 IDM。

建议明确：

配速 V3 后 D2 ≥ 0.80：监督路径存在足够上限；
0.75 ≤ D2 < 0.80：可追平 IDM，但最终超越需要 RL 或更强 recovery teacher；
D2 < 0.75：当前教师/执行链不足，不能把 0.80 目标押在 BC 上；
如果配速只提升 success、同时 collision/off-road 显著上升，则不算 ceiling 提升。
六、最终意见

coding agent 的 P1 v0.3 可以批准，但应附加以下明确修改：

LD loss 定位为几何代理，不冒充完整道路可行域；
D2 配速拆成几何缩放和独立速度 override，避免混淆；
明确 P1“不改 MoE 架构”不等于“experts 永久冻结”；
recovery 以 T_plan/低 margin 为锚，不机械截取终局前窗口；
梯度校准同时看 norm 和 cosine；
跨行 action/ego pose 对齐提前到采集前；
新增 freeze audit 与 geometry coverage probe。

**剩余真正开放的分歧只有一个：配速修复后 D2 的实际 ceiling 是否足够。**其余实现方向已经基本收敛，可以在上述修改后进入 P1-0。