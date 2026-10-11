总体结论

这轮执行可信，关闭 gated residual 主线是正确决定。15 组零训练配置没有找到稳定有效区域，gate 对翻正、翻负和其他 episode 的激活率几乎没有选择性；最优配置虽在 clean150 上净增 1.33pp，但只是单 seed、净多 2 个成功 episode，且 CPU/CUDA 仍有 6/40 个结局翻转，不能据此继续投入。

但我不同意直接把“RL 合法信用链”升为下一主线。当前学习模型约 0.33，离 IDM+Pure Pursuit 的 0.756 太远。在这个质量级别上进行复杂 RL，大概率只会优化一个错误的计划流形。

一、这轮真正证明了什么
1. Gated residual 失败不是简单阈值问题

证据已经足够：

激活对历史翻正、翻负和其他 episode 几乎无差别；
全扫描无法同时保留主要翻正并消除主要翻负；
threshold、scale、debounce 存在局部净收益，却没有稳定机制；
原始 residual 的收益主要体现为降低 off-road/collision，同时提高 timeout。

因此根因不是单纯的：

阈值太低
负样本太少
残差太大


更可能是：

当前状态表示和 residual 目标不足以稳定判断“何时、往哪个方向、修正多少”。

继续补 hard negatives、扩大 gate 或调整采样比例，预期收益很低。

2. CPU/CUDA 不是 gate 失败的首要根因，但必须治理

单步 gate 跨阈值率不超过 0.12%，说明 gate 本身对数值差异并不特别敏感；但闭环长时程把微小差异放大成 6/40 个 episode 的结局翻转。

所以应将其定义为：

闭环评测方差与复现风险，而不是 gated residual 失效的主要解释。

后续统一 CUDA 采集、训练校准和评测即可，不值得单独开启“后端鲁棒训练”项目。

3. 当前最强证据仍是 V1，但不能过度解释

V1 达到约 0.778，说明：

当前 (ds,dθ) 接口和 LQR 并非绝对无法超过 IDM； -主要问题之一确实与 plan 的纵向尺度、曲率和执行耦合有关。

但 V1 使用在线专家动作并修改 ds，不是可部署学习模型；它同时改变速度、等效曲率和 tracker 难度。

所以 V1 是机制线索，不是模型上限证明。

二、需要追到更深的 root cause

当前 repo 的核心矛盾很可能不是“缺 recovery”，而是：

教师行为与模型训练目标不等价

已有证据非常强：

D2：每 0.5 秒重新查询当前专家动作，成功率 0.706；
D1：提前生成未来专家动作链并开环执行，成功率接近 0； -模型却被要求一次输出六步、三秒计划； -模型 plan 约 0.33，exact 执行也没有改善。

这表明系统本质依赖：

观察新状态
→ 重新规划
→ 再观察
→ 再修正


但训练却较重地要求模型拟合远期六步动作链。远期 target 在 learner 状态下本身不稳定，甚至未必有唯一正确解。

这比 gate、MoE、WM 或参数量更接近根因。

三、下一主线建议
优先级 1：重新验证基础规划监督，而不是立即 RL

下一阶段应回答：

模型能否先稳定学会“当前状态下正确的下一步动作”，并由确定性几何 rollout 形成可执行短计划？

建议做一个最小、强约束的 baseline：

当前 observation/history
→ 直接预测 expert 当前动作 (ds,dθ)
→ action-conditioned deterministic tail
→ LQR


关键点：

第一步只监督当前状态下专家动作； -后五步不再平权拟合未来专家动作链； -后五步主要承担连续性、曲率和平滑约束； -每 0.5 秒重新规划； -第一步权重远高于远期点； -取消 recovery gate、WM rollout 和新的复杂路由变量。
验收顺序
1. 当前 expert action 的离线 MAE
2. 第一步方向和尺度 bias
3. closed-loop repeat/current-action baseline
4. action-conditioned tail 相比纯 repeat 的增益


如果连当前专家动作都无法稳定拟合，则问题在 observation、标签、encoder 或优化，不应进入 RL。

优先级 2：做 V1 的机制分解，但只做少量关键实验

不再继续大范围扫描，只检查三件事：

模型是否系统性低估 ds；
低估是否集中在特定速度、曲率和道路类型；
修正 ds 后，成功提升来自减少 timeout，还是降低等效曲率。

建议输出：

expert_ds - model_ds
expert_dtheta - model_dtheta
expert_curvature - model_curvature


按以下维度分层：

速度区间
道路曲率
tollgate / merge / roundabout
成功 / off-road / timeout


只有发现跨路型稳定的 signed bias，才启用现有 phase3_bias_calibration。不要直接把全局 ds×1.25 写入模型。

优先级 3：建立 IDM+Pure Pursuit 的安全残差基线

若最终目标是可靠超过 IDM+Pure Pursuit，最务实的路线不是让当前 0.33 模型完全替代它，而是：

IDM+Pure Pursuit base action
+ learned bounded residual
+ safety arbitration


与已失败的 plan residual 不同，这里的 base 是稳定、闭环、可执行的强基线，不是当前模型的弱 plan。

建议约束：

|residual_ds| <= 10%–15% base_ds
|residual_dtheta| <= 小范围


只在高置信度且安全检查通过时应用；否则回退到 IDM+Pure Pursuit。

训练目标聚焦它的弱点：

tollgate； -复杂几何；
timeout； -过度保守； -局部路线选择。

这条路线更可能实现“在不丢掉 0.756 基本盘的前提下获得增益”。

但必须防止评测作弊或循环定义。模型可使用的地图、路线和对象信息必须与正式部署允许的信息一致。

四、RL 何时才能启动

RL 不是不能做，而是现在不应作为第一步。

建议启动门：

监督或混合基线，三 seed success >= 0.65
计划首步确实决定最终执行
训练和评测都使用相同的 PlanHead+LQR 口径
action clipping <5%
LQR 饱和可控
value 能在失败前识别风险


达到门槛后，使用 2D action-conditioned plan PPO，而不是立即启用 mode+latent、12D trajectory 或 GRPO。

如果监督模型仍停留在 0.4 左右，RL 更可能利用 reward 漏洞或退化成慢行，而不是补齐 0.35 以上的成功率差距。

五、P1-3 geometry FP Gate

建议继续做完，只作为工具资产，不作为下一训练主线。

如果 LD proxy 对成功专家轨迹误报明显，直接将其降级为监控和 hard mining 指标；不要再投入时间调 loss 权重。

六、建议锁定的 Round-15
Track A：基础监督重构
A1. 当前专家动作标签一致性审计
A2. 单步 expert-action BC baseline
A3. action-conditioned deterministic tail
A4. 三 seed clean150 → eval500

Gate
单步动作 bias 明显下降；
clean150 至少达到 0.50，才继续完整 eval500；
eval500 达到 0.60–0.65，才考虑 RL。
Track B：V1 机制诊断
B1. ds/dtheta/curvature 分层 bias
B2. 仅在稳定 bias 存在时启用 bias calibration
B3. 一次独立消融

Track C：强基线安全残差可行性设计

先只做设计和离线 oracle：

IDM+Pure Pursuit 在哪些 episode 失败
网络是否在这些 episode 提供正确修正
理想 arbitration 的理论上限


如果离线 oracle 无法把 0.756 提高到至少 0.82，就不值得实现混合控制器。

七、停止事项

继续冻结：

gated residual；
recovery_v1/v2 扩采； -更多 gate 特征；
P1-4B；
K-anchor；
GRPO；
WM/MoE 重构； -容量扫描； -未验证的 geometry loss。
最终判断

Coding agent 对 Round-12A 的执行和 NO-GO 判断是严谨的，我同意主线关闭。

但我反对直接把复杂 RL 作为 fallback 第一优先级。现有证据指向更基础的根因：

当前系统依赖频繁闭环重规划，但模型训练仍过度承担三秒远期动作链预测；同时模型尚未稳定复现当前状态下的专家动作尺度与曲率。

下一步应先重建强而简单的闭环监督基线，并评估 IDM+Pure Pursuit 上的安全残差上限。只有把基础闭环能力从 0.33 推到至少 0.60–0.65，RL 才有现实机会把它推向并超过 0.756。