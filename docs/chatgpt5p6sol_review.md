我已阅读第三轮收拢文档 deepseekv4p1_argue.md。整体上，方案已经可以从“争论”进入“执行设计”，我接受 coding agent 对 PPO 口径问题的技术修正，并建议按下面版本锁定。

一、确认三项修正
D-A：接受

不能直接把 PPO 训练改成 PlanHead+LQR。若 PPO 采样 action 不实际决定执行轨迹，credit assignment 会失真。正确顺序是：

先用现有 checkpoint 做双口径零训练评测；
如果口径差显著，再选择：
action-conditioned plan；
或将 repeat-action 定义为部署接口；
不直接恢复旧的 plan-reference PPO。
D-B：接受

第一步无需重训。评测器已经支持 plan 和 repeat_action 两种 reference，因此先量化同一 checkpoint 的执行口径差，是成本最低且信息量最高的实验。

D-C：接受

“PPO 完全没有优化部署 plan”这个表述不准确。当前 PPO 可以训练 MoE experts 和 residual scale，因此与 plan 部分耦合；真正的断点是 primary、router、WM 等被冻结，以及训练执行口径与默认评测不同。

二、关闭 D1–D5
D1：审计对象

锁定以下对象：

IDM；
s11；
Arm P；
pri512；
oracle；
laneplan。

主集使用 eval500，最终候选再用 clean500 复核。P0 阶段不必让所有对象都做三 seed，先用相同 episode ID 做 paired audit。

D2：D 格工具口径

同意 coding agent 的方案：

expert action 按模型相同的 (ds,dθ) 规则积分为 6 点、3 秒 plan，再交给同一个 LQR。

D 格必须复用模型完全相同的：

单位和归一化；
action clipping；
0.5 秒时间间隔；
SE(2) 累积；
首点处理；
LQR 参数。

否则测到的不是 plan parameterization ceiling，而是工具实现差异。

D3：口径统一路线

现在不预选 ①②③，由 P0 双口径结果决定：

若 repeat_action 仅高 0–3pt：暂不改执行口径，继续查计划和 tracker；
若高 3–8pt：优先研究 action-conditioned plan；
若高于 8pt，且 collision/off-road 同时改善：部署口径本身必须重新评估。

不建议仅因为 success 高就直接把 repeat-action 设为正式口径，还要检查平顺性、碰撞率和与历史协议的可比性。

D4：验收门槛

同意：

G2：3 seeds 平均 success ≥ 0.74；
G3：success ≥ 0.80 且 paired 显著；
stretch target：0.82–0.85。

最终验收同时要求 eval500 + clean500。由于目标是“大幅超过 IDM”，0.77 左右只能算追平或弱领先，不能算完成。eval500 在成功率约 0.8 时单集标准误约 1.8pt，因此三 seed 和 paired 检验是必要的。

D5：WM/MoE 对照时点

锁定在 P2，不提前开发 0/2/8 experts。

原因不是 MoE 一定有效，而是当前最需要先解决接口上限、执行口径和主要失败类型。提前改 MoE 会改变失败分布，降低归因价值。

例外：如果 P0 显示 s11 的 router/expert 输出异常，例如频繁切换、残差爆炸或直接导致 plan-out，可以提前做单头对照。

三、锁定后的总体方案
S0：排摸收尾

允许现有 attn2、tru192、stg3 等已排队实验结束，但：

不再增加新容量臂；
不用这些结果直接裁决 WM/MoE 去留；
正式保留 manifest、配置、checkpoint hash 和逐 episode 结果。
S1：P0 审计，先不训练
1. 双口径评测

对 s11、Arm P、pri512 分别运行：

plan reference
repeat_action reference


输出：

success；
off-road；
collision；
route completion；
paired episode 差异。
2. 上限矩阵

运行 A–F：

A：IDM；
B：oracle trajectory + LQR；
C：laneplan + LQR；
D：expert action → 模型格式 plan → LQR；
E：模型 plan，分别用 LQR 与 exact；
F：模型 repeat_action。
3. 初步失败归因

先对 s11 和 IDM 各取 30–50 个 off-road episode，分类：

plan-out；
tracker-out；
geometry/termination anomaly；
recovery failure。

必须加入节点之间的 ego footprint 插值检查，不能只检查 6 个 plan 点。

四、P0 结果的决策树
情况 A：D 不高于 IDM

说明 expert action 经过当前 (ds,dθ) plan 和 LQR 后已损失过大。

下一步：修 action/plan 接口，不改网络。

重点检查：

曲率连续性；
0.5 秒节点是否过疏；
action clipping；
低速时 dθ/ds 异常；
LQR 对急弯和收费站的跟踪能力。
情况 B：D、B 明显高于 IDM，但 E 显著较低

说明接口存在高上限，主要问题在学习。

下一步：走计划侧分支。

首选实现：

DAgger 真实 recovery trajectory；
footprint boundary margin；
节点间 corridor violation；
curvature 和 delta-curvature loss；
hard mining 攓为闭环风险。

这里我建议把 DAgger recovery 放在新增 corridor observation 之前，因为它更直接利用现有失败分布，也更容易验证收益。

情况 C：Exact 明显高于 LQR

说明模型 plan 本身可用，主要损失来自 tracker。

下一步：走执行侧分支。

优先顺序：

LQR 参数与速度调度；
轨迹曲率和连续性约束；
tracker-aware plan loss；
必要时增加近场计划点密度。
情况 D：repeat_action 明显高于 plan

说明训练和部署口径错位是主要损失源。

不要简单把部署改成 repeat-action。先实现：

PPO action 条件化 PlanHead，并保证 plan 第一段严格由 PPO action 决定。

随后逐步解冻：

plan final/residual；
primary 最后一层；
router；
必要时再解冻更深层。

每步监控 KL、BC holdout、plan-action consistency 和 closed-loop KPI。

情况 E：主要是 recovery failure

优先修改 DAgger：

回溯失败前 2–4 秒；
保存 student plan、实际轨迹、expert recovery plan；
训练完整 6 点 recovery trajectory；
不再使用常量外推的 traj6；
hard mining 使用 boundary margin、tracking residual、collision risk，而非单纯 BC error。
五、对当前草案的两点调整
1. G1 不必须“显著高于 IDM”

G1 的作用应是确定可达路径，而不是提前完成统计显著性验证。建议改为：

B、D、E_exact 或 F 中至少一条在 paired eval 上达到 ≥0.80，或者相对 IDM 提升 ≥5pt，并明确主要损失发生在哪一层。

正式统计显著性留给 G3。

2. P1 不要机械地只做一个代码点

应只选择一个根因分支，但一个分支内部通常需要组合修复。例如 plan-out 分支可能同时需要：

recovery data；
footprint loss；
curvature regularization。

不应为了“单变量”而只改其中一项。正确方法是：

先用小消融确认各项方向；
再形成一个完整 candidate recipe；
candidate recipe 补三 seed。
六、最终收拢版本
已锁定共识
排摸收尾，不扩扫参；
不预设删除 WM/MoE；
上限审计先于网络改造；
双口径先零训练量化；
失败分类从 30–50 例开始；
加入 footprint 插值检查；
DAgger 使用真实 recovery trajectory；
正式结论使用三 seed、paired test、eval500 和 clean500。
当前无重大原则分歧

剩余问题已经从架构争论转为数据判决：

超过 IDM 的可达链路在哪里；
损失主要发生在 plan representation、学习、tracker 还是执行口径；
P1 应进入计划侧、执行侧还是 recovery 分支。
建议下一条给 coding agent 的指令
方案原则已基本收拢。

按 S0→S1 推进：

1. 允许当前容量排摸收尾，但禁止新增扫参臂。
2. 完成 s11、Arm P、pri512 的 plan/repeat_action 双口径评测。
3. 完成 A–F 上限矩阵，优先实现 D 格：
   expert action → 同模型 ds/dtheta 6点计划 → 同一 LQR。
4. 对 s11 和 IDM 各抽取 30–50 个 off-road episode，
   完成 plan-out / tracker-out / anomaly / recovery 分类，
   包括节点间 footprint 插值校验。
5. 输出一份统一审计报告，包含逐 episode paired 数据、
   manifest、checkpoint hash、运行口径和推荐 P1 分支。

S1 完成前不修改 WM、MoE、encoder 或 PPO 冻结范围，
也不开始新的容量实验。


这版可以作为大致方案定稿。下一次讨论应只围绕 P0 数据和它触发的分支，不再重复讨论“是否先删 WM”“是否继续扩容量”等已关闭问题。