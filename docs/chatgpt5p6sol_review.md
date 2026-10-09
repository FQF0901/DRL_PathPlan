我继续基于 DRL_PathPlan_code.txt 中已核查的实现和 coding agent 最新排摸收拢。总体方案可以开工，但我建议在正式执行前补上 4 个技术约束，否则 P0 的数字可能仍不可比较。

一、同意 coding agent 的新增结论
1. s11 必须走兼容代码

s11 属于 obs v5、v7 网络，当前代码是 obs v6、v8 网络。若当前 eval_runner.py 对 shape mismatch 只记录并继续运行，确实可能出现部分权重随机初始化，结果完全无效。

我同意：

使用 v7-lock-20261006 或对应 commit 重建 s11 runner；
s11 做完整的 plan/repeat_action 双口径评测；
每次评测强制验证 checkpoint 加载完整性。
2. checkpoint 加载校验必须升级为 fail-fast

不能只是报告里附日志，应直接让正式评测退出：

missing_keys == 0
unexpected_keys == 0
shape_mismatch == 0
obs_schema/fingerprint compatible


否则以后仍可能制造“模型性能下降”的假结论。当前仓库历史上已经存在 shape mismatch 后继续执行的风险，因此这是 P0 的前置修复，不是一般记录事项。

3. D 格首点无需额外覆盖

专家 6 步动作链的第一个 (ds,dθ) 本身就是下一规划步，确实无需再使用模型 mu 覆盖。但必须复用 net/model.py 的 arc_step 和模型相同的 clipping、单位、SE(2) 累积及 LQR 参数，不能重新写近似转换。

二、需要补充的 4 个关键约束
约束 1：s11 不能简单使用“完整旧评测栈”和 v8 横向比较

如果直接用整个 v7 archive 跑 s11，而 Arm P/pri512 用当前 v8 evaluator，差异可能同时来自：

模型和 obs；
MetaDrive 环境包装；
termination 判定；
reward context；
LQR 或 action clipping；
episode spec 解析；
KPI 聚合代码。

这样不再是严格 paired comparison。

建议实现

建立两个版本层：

模型兼容层：分别使用 v7 model + obs v5、v8 model + obs v6
评测语义层：尽量统一使用当前 scenario IDs、termination、tracker 参数和 KPI 聚合


若无法统一，报告必须明确分开：

同版本内部比较：s11 plan vs s11 repeat_action；
跨版本参考比较：s11 vs v8，只用于方向判断，不做百分点精确归因。

不要把跨 evaluator 的 2–3pt 差异解释成网络差异。

约束 2：D 格是 privileged ceiling，不是可部署 expert ceiling

D 格使用 live expert 获取未来 6 步动作。如果这些动作来自专家未来状态轨迹，它实际上利用了未来闭环信息：

expert future states
→ future actions
→ 6-step plan
→ LQR


模型在线时并没有这些未来 expert states，所以 D 应标记为：

expert-oracle parameterization ceiling

它可以回答“(ds,dθ) + LQR 是否有能力超过 IDM”，但不能证明模型仅靠当前观测能学到这个水平。

建议同时保留两项指标：

D-oracle：真实未来 expert action chain；
D-hold：只使用当前 expert action 重复 6 步。

D-oracle 与 D-hold 的差距，可以测量“未来动作链信息”的价值，并与模型 plan、repeat_action 直接对应。

约束 3：exact 只能诊断，不能作为可部署性能上限

需要先确认 --tracker exact 的实际语义。如果 exact 直接把车辆状态推进到参考轨迹，或绕过真实控制/车辆动力学，它只能回答：

计划几何本身是否合理。

不能把 E_exact ≥ 0.80 直接解释为“只要调好 LQR 就能达到 0.80”。两者之间还可能存在：

车辆动力学不可达；
转向和加速度饱和；
10Hz 控制误差；
轨迹曲率不连续。

因此推荐解释：

E_exact - E_lqr = 执行栈相关损失的上界


而不是纯粹的“LQR 参数损失”。

约束 4：off-road 四分类必须按最早根因分层，不能简单互斥打标签

plan-out、tracker-out 和 recovery failure 可能同时发生。例如：

计划先产生低 margin；
LQR 跟踪误差进一步放大；
最后 recovery 失败并越界。

建议记录三时间戳后采用层级判定：

T_plan   = 首次计划 footprint 不可行
T_track  = 首次实际轨迹明显偏离仍可行的计划
T_cross  = 首次实际 footprint 越界
T_term   = 环境正式终止


分类规则：

T_plan < T_cross 且计划本身越界：plan-originated；
计划始终可行，但 tracking residual 先超阈值并越界：tracker-originated；
进入危险状态后仍存在可行 recovery，但模型持续失败：recovery failure；
几何判定和环境 termination 不一致：anomaly。

报告中同时保留多个 contributing factors，不要只输出一个标签。

三、对 corridor / boundary 实现的具体要求

coding agent 提议复用 road_edge_distance_from_ctx，方向正确，但要注意：

reward 使用的通常是当前实际车辆状态；
P0 需要检查的是任意未来计划 pose 的完整车辆 footprint；
不能直接把当前 ego 的 road-edge distance 套到未来 plan 点。

正确做法是：

用 arc_step 得到每个计划 pose；
在相邻规划节点间按固定空间间隔插值，建议 0.5–1.0m；
在每个 pose 上生成车辆矩形 footprint；
至少检查四角、四边中点和中心；
使用 MetaDrive 地图几何判断采样点是否属于 drivable region；
记录最小 signed margin。

如果引擎不能稳定提供 signed distance，第一版可记录：

inside_ratio
first_invalid_pose
invalid_footprint_point_count


不要伪造不可靠的连续距离。

四、最终 P0 执行版本
P0-A：评测安全前置
正式评测 checkpoint 加载改为 fail-fast；
保存 missing/unexpected/mismatch 摘要；
保存 obs schema、fingerprint、commit 和 evaluator version；
s11 建立 v7 兼容 runner；
明确跨版本结果只能方向比较。
P0-B：双口径与上限矩阵

运行：

A       IDM
B       oracle trajectory + LQR
C       laneplan + LQR
D1      future expert action chain + arc_step + LQR
D2      current expert action repeated + arc_step + LQR
E1      model plan + LQR
E2      model plan + exact
F       model repeat_action + LQR


D1 是 privileged ceiling，D2 与 repeat_action 更可直接比较。

P0-C：失败归因

对 s11 与 IDM 各先做 30–50 个 off-road episode，记录：

T_plan / T_track / T_cross / T_term
plan footprint validity
executed footprint validity
tracking residual
road class / map id
policy action / plan first action
router top-k / policy std


初始分类用于决定方向，不作为最终统计结论。

五、收拢后的唯一剩余风险

现在已经没有架构层面的原则分歧。剩余最大风险是：

为了兼容 s11 使用旧代码，导致跨版本评测语义变化，最终把 evaluator 差异误判成模型性能差异。

因此建议 coding agent 在实施前先输出一份很短的“评测兼容设计”：

哪些模块必须使用 v7
哪些环境、tracker、termination、spec、KPI 使用统一实现
哪些指标允许跨版本比较
哪些指标仅允许版本内比较

可以给 coding agent 的下一条回复
认可第四轮排摸，按完整版 P0 推进，并增加以下硬约束：

1. 正式评测 checkpoint 加载必须 fail-fast：
   missing/unexpected/shape_mismatch 均为 0，
   obs schema/fingerprint 必须兼容。

2. s11 使用 v7 模型与 obs 兼容代码，但尽量统一当前场景 spec、
   tracker 参数、termination 与 KPI 聚合。
   若无法统一，s11 只做版本内 plan vs repeat_action 的严格比较；
   与 v8 的绝对百分点比较标记为跨版本参考。

3. D 格拆为：
   D1 = 未来 expert action chain -> arc_step -> LQR，
   标记为 privileged/oracle ceiling；
   D2 = 当前 expert action repeat 6步 -> arc_step -> LQR，
   用于和 repeat_action 直接比较。

4. exact 只作为计划几何诊断，不视为可部署 tracker 上限。
   E_exact-E_lqr 解释为执行栈损失上界。

5. off-road 分类采用 T_plan/T_track/T_cross/T_term 的最早根因层级，
   同时保留 contributing factors。

6. future plan boundary 检查必须对插值后的完整车辆 footprint
   查询 drivable geometry，不能直接复用当前 ego 的 reward distance。

先输出评测兼容设计与字段定义，再实现并跑 P0。
P0 完成前继续禁止修改 WM、MoE、encoder 和 PPO scope。


这个版本已经足够开工，同时最大程度避免“评测看似严谨，但比较对象实际不一致”。