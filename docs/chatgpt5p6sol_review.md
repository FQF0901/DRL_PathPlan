总体结论

Round-28 给出了一个非常关键的否定结果：

固定 ds 并只学习横向曲率 κ，仍然没有消除 timeout。

所以此前“timeout 主要由 semantic 意外修改纵向动作造成”的假设不成立。横纵耦合确实存在，但当前主要矛盾已经进一步定位到：

横向修正改变完整六步路径
→ LQR 从尾段预瞄
→ 横向误差、实际车速、油门需求相互影响
→ 即使 ds 数值保持 legacy，一样可能车辆掉速并 timeout


目前最强候选不是失败品。K1 的第一阶段已经取得：

teacher MAE 只恶化约 1.1%；
learner-state κ MAE 下降约 71.6%； -纠偏方向正确率 100%； -steer saturation 比 legacy 低 12.5pp； -calibration50 success 达 0.52； -off-road 约 0.12。

这说明“固定纵向、只学曲率 residual”的方向基本正确。真正失败的是：

timeout 仍增加约 20pp； -plan Δκ 仍为 expert 的 2.78 倍； -镜像曲率一致性只有 0.70； -S2 一旦解冻末层就彻底崩溃。

因此我不赞成立刻推翻 K1，更不赞成现在增加纵向头。下一步应把 K1 的尾段路径与 LQR 预瞄链闭合。

一、Round-28 真正证明了什么
1. 横纵耦合是真实的，但不只来自动作头

四组反事实已经给出较强证据：

F1 保持曲率、联合缩放 ds/dθ，timeout 消失但 off-road 显著恶化，说明速度、路径尺度和跟踪能力强耦合；
F3 不改路径，只提升独立 v_ref，timeout 从 12% 降至 8%，且 off-road 不变，说明 tracker 速度参考是一个独立有效轴；
F4 只改尾段就显著影响当前 steer 和 timeout，说明 LQR preview 让未来计划点直接作用于当前控制；
K1 保证 ds 与 legacy 逐位相同，timeout 仍明显增加，说明 timeout 不是简单的 ds residual 造成。

所以现在不能再用：

ds = 纵向
dθ = 横向


这种简单理解。

真正的控制链是：

六步 ds / dθ
→ 六段曲率与路径形状
→ 10Hz reference
→ LQR 预瞄尾段
→ steering 与车辆真实掉速
→ longitudinal controller 追赶参考速度
→ throttle 饱和 / 横向误差 / route progress

二、对具体代码更改的审阅
1. env/tracking.py 的新增遥测是正确的

当前已经增加：

steer_raw
throttle_raw
lqr_ref_curvature
lqr_actual_curvature
lqr_preview_plan_step
ref_chord_speed


这些字段足以区分：

参考路径本身过弯；
LQR 增益导致过度控制； -车辆真实曲率达不到参考曲率； -当前控制主要由第几个 plan step 决定；
timeout 是否来自 throttle 长期饱和。

建议保留。

但还缺两个关键指标

建议补充：

lqr_curvature_error =
    lqr_ref_curvature - lqr_actual_curvature

lqr_speed_error =
    lqr_ref_speed_mps - actual_speed


以及：

preview_heading_error
preview_lateral_error


虽然已有 lqr_error_y/psi，最好在归因报告中明确形成组合字段。

还建议记录：

preview_plan_step 切换频率


如果 preview 在 plan step 边界频繁跳跃，控制抖动可能来自离散预瞄索引，而不是模型轨迹本身。

2. net/model.py 的曲率诊断基本正确，但低速处理仍不严谨

当前：

kappa = dtheta / ds.clamp_min(1e-6)


并通过：

valid_kappa = abs(ds) >= 0.5


标记有效性。

在本项目 ds 预计非负，因此通常不会发生负值分母问题。但这里仍有两个风险：

风险 A：无效曲率仍被数值计算

当 ds 很小时，kappa 会产生巨大值，只是额外返回一张 valid mask。

必须确保所有：

loss； -分位数； -均值； -镜像指标； -plan Δκ；

都真正应用 mask，而不是先聚合再看 valid ratio。

风险 B：plan_delta_kappa 没有 pairwise validity

当前：

plan_delta_kappa = kappa[..., 1:] - kappa[..., :-1]


正确的有效 mask 应是：

valid_delta_kappa[t]
=
valid_kappa[t] & valid_kappa[t+1]


否则一个无效低速 step 会污染相邻两个 Δκ。

建议模型明确输出：

plan_valid_delta_kappa

3. pipeline/semantic_labels.py 的曲率标签方向正确

现在保存：

expert_ds / expert_dtheta / expert_kappa
legacy_ds / legacy_dtheta / legacy_kappa
kappa_valid


并统一使用 ds >= 0.5m/0.5s 的有效门，方向是对的。

小问题

代码中 validity 使用 abs(ds) >= 0.5，但代码库的物理动作语义是前进行驶距离，正常情况下 ds 应非负。

建议增加 fail-fast：

ds < -epsilon 的行数必须为0


而不是默默使用绝对值接受负 ds。否则未来若出现动作解码 bug，曲率标签系统会把倒车动作视为有效正常样本。

4. pipeline/residual_classes.py 对曲率增加审计是进步，但 invalid-kappa 策略不合理

当前逻辑相当于：

kappa_ok = (~valid_kappa) | (res_kappa <= threshold)


也就是：

曲率无效时自动通过曲率门。

这对一般数据审计方便，但对本项目当前的 timeout/低速问题很危险。

因为：

-低 ds 正是最容易产生高曲率和 timeout 的区域； -曲率无效不代表状态安全； -自动通过可能把低速、高 dθ 状态归为 N0/N1； -随后 preservation 会保护一个实际上不可持或者不可跟踪的动作。

建议改成三种状态
valid_curvature
low_progress_yaw_motion
invalid_or_unknown


对于 ds < 0.5m：

-如果 |dθ| 也小，可以作为低速近静止状态； -如果 |dθ| 大，应归入 correction/unsafe； -不能自动视为 curvature safe。

例如：

low_ds_safe = (ds < 0.5) & (abs(dtheta) < theta_low_speed_limit)
kappa_ok = (valid_kappa & residual_kappa_ok) | low_ds_safe


阈值应来自 expert 低速动作分布，而不是手工猜测。

5. S2 解冻崩溃不是偶然，应正式关闭这条支路

S2 解冻末层后：

teacher error ≈ 0.28
calibration success = 0
off-road = 1


类似崩溃已经重复出现多次。

这说明当前末层一旦接受新的 semantic/κ 监督，就会破坏 legacy 的动作流形。可能原因包括：

-同一个输出层同时控制 ds/dθ； -动作量纲和梯度尺度不同； -旧 chain 与新第一步动作不再匹配； -优化器新建状态导致旧层更新过猛； -辅助 loss 与 BC 对输出层发生方向冲突。

因此：

Round-29 不允许再次解冻 Policy/Primary/PlanHead legacy 层。

只训练 semantic residual path。S2 不再当作普通候选保留。

三、K1 为什么 ds 不变仍然 timeout？

这是当前最关键的问题。

1. 实际车速不是由 ds 单独决定

虽然 ds_final == ds_legacy，但模型修改了 κ，于是参考路径更弯。

车辆实际执行时：

-需要更大横向力； -LQR steering 增大； -车辆动力学导致实际速度下降； -纵向控制器继续追踪 legacy 的参考速度； -因而出现高 throttle、低实际 progress； -最终 timeout。

Round-28 中：

throttle saturation 0.50 vs 0.19
|e_y| 0.99 vs 0.51
raw steer 2.88 vs 1.40


正好支持这个解释。

2. 六步尾段直接影响当前控制

LQR 使用速度相关 lookahead，从轨迹中选择未来 preview point。

所以第一步曲率合理，不代表当前 steer 合理。后续某个 plan step 曲率或航向突变，也会通过 preview 提前影响当前控制。

F4 只修改尾段就显著改变 timeout，说明这是当前主因之一。

3. Δκ 仍然远高于 expert

K1 的 plan Δκ 仍约为 expert 的 2.78 倍。

这说明 K1 已经学会“往哪转”，但六步路径仍然：

转向变化过快
弯曲方式不平滑
与跟踪器预瞄不匹配


因此现在的主要问题已经从：

lane semantic 是否正确


转为：

semantic lateral correction 如何形成可执行的六步路径

四、对 Round-28 结论的反驳与修正

Coding Agent 写道：

K1 核心假设被否定。

这个表述过强。

被否定的是：

“只要固定 legacy ds、只学习 κ residual，就能自动消除 timeout。”

但 K1 的核心价值并没有被否定。相反，K1 S1 在多个关键指标上是目前最好结果：

-正常 teacher action 基本不退化；

learner κ 大幅改善； -转向饱和下降； -calibration success 达到 0.52； -off-road 只有 0.12。

所以更准确的结论是：

横纵动作解耦有效，但仅解耦第一步动作不够；还需要同时解决六步 lateral path shaping 与 path-conditioned speed tracking。

这与行业方向一致。PLUTO 显式建模横纵行为，AlignDrive 则先规划 lateral path，再让纵向决策条件化于路径，而不是让两个头完全独立。

五、Round-29：先修尾段，不训练纵向头
目标

保留 K1 S1 的优势：

teacher稳定
learner κ改善
off-road低
semantic方向正确


只解决：

plan Δκ过大
LQR preview受尾段干扰
实际速度下降导致timeout

29A：零训练尾段投影实验

以 K1 S1 e1/e2 checkpoint 为唯一基础。

保持第一步：

ds0, kappa0 不变


只处理 plan step 1–5。

T0：原始 K1 plan

作为基线。

T1：曲率变化率投影

求一个最接近原计划的平滑曲率序列：

min⁡κ1′...κ5′∑twt(κt′−κt)2\min_{\kappa'_1...\kappa'_5} \sum_t w_t(\kappa'_t-\kappa_t)^2

约束：

∣κt+1′−κt′∣≤Δκmax|\kappa'_{t+1}-\kappa'_t| \leq \Delta\kappa_{max}

并固定：

κ0′=κ0\kappa'_0=\kappa_0

其中 Δκ_max 使用 expert plan 的 P95 或 P99，按速度和道路曲率分组。

最终：

dθt′=κt′dstd\theta'_t=\kappa'_t ds_t
T2：尾段指数衰减到道路曲率

令：

κt′=ρtκt+(1−ρt)κlane,t\kappa'_t = \rho_t\kappa_t+ (1-\rho_t)\kappa_{lane,t}

其中前段更信任模型，后段逐步靠近 target-lane curvature。

第一版可以使用：

rho = [1.0, 0.8, 0.6, 0.4, 0.25, 0.15]


但只作为一次固定对照，不扫参数。

T3：T1 + 独立 v_ref ×1.15

几何用 T1，tracker 参考速度使用 F3 已验证的 1.15。

这是当前最关键组合，因为：

T1 解决尾部 preview 的 lateral trackability； -独立 v_ref 解决实际掉速后的 longitudinal progress； -不会重新引入 learned longitudinal residual。
六、Round-29 不应立刻训练 tail decoder

先用确定性投影回答：

一个可跟踪的尾部参考，能否让 K1 的 lateral correction 变成更好的闭环？

如果 T1/T3 有效，再选择：

部署时保留确定性 projector； 2.用 projector 输出作为 tail distillation target； 3.将约束做成可微投影层。

目前不要训练新的 tail network，否则又会重新引入数据和归因变量。

七、Round-29 评测标准
Gate A：路径可跟踪性

相对 K1 T0：

plan Δκ P95 <= expert P95 ×1.25
raw steer P95 至少下降20%
steer saturation <= legacy +5pp
|e_y| P95 至少下降15%
曲率方向正确率 >=95%

Gate B：纵向进度
timeout <= legacy +3pp
route completion不下降
throttle saturation <= legacy +10pp
实际平均速度不低于legacy 5%以上

Gate C：闭环开发门

在 calibration50：

success >=0.54
off-road <=0.12
collision不恶化
timeout <=0.12


达到后才跑 clean150。

clean150
success >=0.43
相对legacy >=+8pp
off-road <=0.35
collision <=legacy+2pp
timeout <=legacy+3pp

eval500

只有 clean150 通过才执行：

success >=0.45
relative legacy >=+7pp
off-road相对下降至少15%
collision <=IDM
timeout无显著恶化

八、若 Round-29 成功，下一步是什么？
方案 A：确定性 projector 保留

如果 T3 明显有效且计算开销很低，可以先作为正式 planning layer：

model lateral intent
→ curvature projection
→ path-conditioned speed reference
→ LQR


这不是不纯粹的“后处理补丁”，而是规划可行性层。传统规划和 FRENETIX 一样，也会在候选轨迹后进行可行性检查与多目标约束。

方案 B：蒸馏 projector

网络训练时监督：

raw tail → projected feasible tail


逐渐让模型直接产生平滑可执行路径。

只有 projector 上限明确后才值得做。

九、如果 Round-29 失败

若 T1/T3 仍不能同时解决 off-road 与 timeout，则问题不再适合继续通过 plan loss 修补。

应转向两种选择：

路径 1：重新设计 tracker

检查：

-速度相关 lookahead； -preview index 离散跳变； -steer gain； -LQR 对曲率前馈缺失。

当前 LQR 只有 error feedback，没有显式道路曲率 feedforward。对于弯道，全部 steering 依赖 lateral/heading error 累积产生，这可能是 raw steer 很高和跟踪滞后的原因。

可考虑：

δ=Lκref+K[ey,eψ]\delta = L\kappa_{ref} + K[e_y,e_\psi]

其中第一项是曲率前馈。

路径 2：Frenet/path-conditioned planner

若现有六步 (ds,dθ) 很难稳定协调，则进一步改为：

lateral path / curvature profile
→ longitudinal progress along path


但这应是更后续的结构升级，不能在 Round-29 一起做。

十、具体代码建议
env/tracking.py

新增：

lqr_speed_error
lqr_curvature_error
preview_index_delta
preview_plan_step_transition


并增加一个实验性、默认关闭的 curvature feedforward：

delta_ff = atan(wheelbase * ref_curvature)
delta = delta_ff + gain @ error


但本轮只做零训练对照，不进入默认配置。

net/model.py

修复或补齐：

plan_valid_delta_kappa


定义为相邻两步都满足 ds>=0.5。

不要让无效低速点污染 Δκ 指标与投影。

pipeline/residual_classes.py

低 ds 时不要自动通过 curvature gate。

增加：

low_ds_small_dtheta
low_ds_large_dtheta


其中后者直接进入 correction/unsafe，而不是 N0/N1。

pipeline/semantic_labels.py

增加 fail-fast：

expert_ds < -epsilon 或 legacy_ds < -epsilon → 失败


并保存：

expert_delta_kappa
legacy_delta_kappa
valid_delta_kappa

十一、批准范围
批准
Round-29A：
K1 S1 e1/e2 固定checkpoint
T0/T1/T2/T3零训练尾段投影
独立v_ref组合
补齐trackability遥测
条件性clean150/eval500

条件批准
LQR curvature feedforward只读/零训练对照

暂不批准
新纵向head
新tail decoder
再次解冻legacy层
DAgger
MoE
PPO/GRPO
大规模采集
WM/K-anchor
更多semantic loss

最终判断

横纵耦合已被证实，但 Round-28 也证明：

问题不在第一步 ds residual，而在“横向 correction 如何形成六步路径，以及这条路径如何被 LQR 预瞄和纵向执行”。

目前最合理的推进不是继续训练，而是利用已经很强的 K1 checkpoint，增加一个严格受限的轨迹可行性层：

semantic curvature intent
→ 平滑、受约束的 lateral tail
→ 独立 speed reference
→ LQR


如果这能把 calibration50 从 0.52 进一步提升，并同时压低 timeout，才真正形成一条有希望向 0.60、0.70，最终超过 IDM+Pure Pursuit 的可扩展路径。