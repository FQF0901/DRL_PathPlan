结论：批准 A3，A4 条件批准

这轮结果很关键，根因已经明显收敛：

模型在 expert-state 上当前动作预测很好，但在 learner-state 上误差扩大 6–16 倍，说明主要问题是 covariate shift，而不是模型容量不足。
正确当前动作加简单 repeat 尾，闭环可达 0.72–0.78，说明无需学习脆弱的未来六步专家链，也暂未触及接口硬上限。
Track C 的理想 union 只有 0.790，未达到预设 0.82，因此不值得转向 IDM+Pure Pursuit 混合残差系统。
LD 几何代理五项 Gate 失败，ld_margin_weight 继续为 0 是正确决定。
一、批准 A3，但要纠正其定义

A3 不应再称为“当前动作 BC + 尾部轻监督”，而应严格定义为：

obs/history
→ 预测当前 expert action (ds0, dtheta0)
→ 固定确定性尾部
→ arc_step
→ LQR
→ 每 0.5 秒重新规划

第一版只保留两个尾部
A3-R：原始 repeat
(ds_k, dtheta_k) = (ds0, dtheta0)

A3-V：V1 风格纵向尺度
(ds_k, dtheta_k) = (s * ds0, dtheta0)


但 s 不能直接锁死为 1.30。1.25–1.30 是特权专家动作上的最佳值，不保证适用于有预测误差的模型动作。模型动作存在噪声时，放大 ds 可能同步放大执行风险。

建议只测试：

s = 1.00, 1.15, 1.25


先用 clean150 选择方向，禁止继续扩大扫描。

二、A3 的训练目标必须真正删除远期硬标签

这是本轮最重要的实现约束。

Coding agent 写“步 0 主导、尾只担连续性/平滑”，我赞同，但必须落实为：

L = L_current_action
  + λ_smooth * L_tail_smooth
  + λ_curv * L_delta_curvature


禁止继续保留：

未来 5 步 expert action regression
未来专家状态上的绝对 trajectory target
合成 traj6
recovery trajectory loss


否则只是调整权重，没有真正验证“当前动作 + 确定性尾”假设。

此外，L_delta_curvature 只能约束平滑，不能重新启用已经失败的 ld_line_margin_loss。

三、A3 必须从原模型结构中隔离干扰源

第一版建议：

复用现有 encoder/memory；
当前动作只由一个明确 head 输出；
不使用 WM rollout 生成尾部；
不使用 MoE experts 生成尾部；
不让 PolicyHead 和 PlanHead 分别产生两个动作；
deterministic tail 必须是纯函数；
LQR 实际 reference 必须与导出的 plan 逐位一致。

这样才能回答：

当前动作 BC + 简单尾部本身能达到多少？

若继续让 PlanHead、MoE、WM 修改后五步，A3 就失去实验辨识性。

四、A4 条件批准，不应与 A3 同时开训

A1 已满足 DAgger 的触发条件：

expert-state 当前动作误差低；
learner-state 当前动作误差显著高； -错误在多个道路类型上存在。

但正确顺序必须是：

先训练并评测 A3
→ 再从 A3 自己访问的状态采集 A4 数据
→ 再训练 A4


不能直接复用 pri512 或 recovery_v2 的 learner-state 数据作为 A4 主数据，因为 A3 的状态访问分布可能完全不同。

A4 数据契约

每行只需：

learner obs/history
learner current predicted action
expert current action
road class
speed / curvature
episode outcome


不采：

六步专家链；
recovery gate 标签； -未来绝对轨迹；
Pass-2 takeover plan。
混合比例

先从：

expert5k : A4 = 90 : 10


开始，最多升到 80:20。禁止高倍过采样。

五、对现有结论的一个重要反驳

文档称：

“历史主要损失之一 = 让模型承担远期六步链拟合。”

这个结论目前高度可信，但尚未最终证实。

A1 证明 learner-state 当前动作已经严重退化；因此模型闭环差可能主要由首步 covariate shift 导致，而不是远期链本身。A2 只证明正确当前动作配简单尾可以工作。

真正证实该结论需要 A3 对照：

相同 encoder、相同数据、相同训练预算

旧六步链监督
vs
仅当前动作监督 + fixed repeat tail


若后者闭环显著更好，才能裁决未来链监督确实有害或无益。

六、A3/A4 验收标准
A3 clean150 Gate

相对 pri512 0.340：

success >= 0.50
off-road <= 0.35
collision <= 0.05
timeout <= 0.15
action clipping < 5%
LQR steer saturation 不高于 A2 oracle 同尾族明显水平


同时要求：

teacher-state MAE 不恶化
learner-state action error 相对 pri512 下降 >= 25%
模型当前动作 + tail 的计划首段与 LQR reference 一致


若 A3 未达到 0.50，先不做 A4 全量训练。应判断：

当前动作 head 没学好；
fixed tail 对模型误差过敏； -或 observation 在 learner-state 上信息不足。
A4 clean150 Gate

相对 A3：

success +5pp 以上
off-road 相对下降 >= 15%
collision 不增加超过 2pp
timeout 不增加超过 3pp
teacher-state MAE 恶化不超过 10%


通过后再跑 eval500。

eval500 Gate
success >= 0.60
目标 >= 0.65


低于 0.60，则不进入 RL。

七、其他线路裁决
Track B：保留但不独立开训练

learner-state 的 ds 存在稳定低估，但这也可能是 covariate shift 的表现。

因此先让 A4 修 learner-state 分布，再重新测 bias。只有 A4 后仍存在跨路型稳定 ds 低估，才启用 phase3_bias_calibration。

Track C：关闭实现，仅保留报告

理想 union 0.790 未达到 0.82，而且仲裁信号主要由静态信息解释，继续做混合 controller 的收益不足。

Geometry：关闭当前定义

五项 FP Gate 失败意味着问题在 LD 语义和 loss 定义，不是调阈值可解决。保持：

ld_margin_weight = 0


如未来重启，应从 drivable corridor 表示重新设计，而不是继续修现有代理。

RL：继续冻结

A3/A4 的目标只是建立可靠地基。即使达到 0.65，与 IDM 仍有约 10pt 差距。只有满足原定六项门槛后，才可进入合法信用链 PPO。

最终批准意见
立即批准
A3-R：当前动作 BC + repeat tail
A3-V：当前动作 BC + 有限 ds-scale tail
旧六步链监督对照
完整 probes 与 clean150

条件批准
A4：从 A3 自身闭环状态重新采 learner-state 当前动作标签

明确禁止
复用 recovery_v2 代替 A4 新采集
继续监督未来五步专家动作
启动 geometry loss
全局固定 ds × 1.30
PPO / GRPO
WM/MoE/容量改造


因此可以批准 Round-17 开工。

不过请把实验核心写清楚：

这一轮不是“再训练一个更简单模型”，而是严格验证：闭环规划是否应从“预测三秒专家动作链”改为“预测当前专家动作，加确定性 LQR 预瞄尾部，并持续重规划”。