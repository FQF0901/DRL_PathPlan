结论

可以达成共识，且已基本达成。
 Coding agent 已接受我上一轮的核心裁决，并将路线收敛为：

可辨识性探针
→ recovery_v2 同状态采集
→ gated residual pilot
→ 几何代理有效性 Gate
→ 独立小消融
→ candidate recipe


双方也一致认定：

P1-0 可以关闭；
recovery_v1 仅用于诊断，不再用于正式监督；
停止 recovery 比例、freeze scope、数据子集等剂量搜索；
P1-3 暂不训练，先做几何代理误报验证；
P1-4A/4B 继续冻结；
V1 的 ds 拉伸只是 calibration oracle，不能解除 4B 门槛。
仍有一项实质分歧
Coding agent 建议并行推进“无条件校准轨”

它提出：

条件轨：
可辨识性 → recovery_v2 → gated residual

无条件轨：
V1 同构的目标重标定 + Δκ / progress


理由是 V1 在 ds scale=1.20–1.30 时达到 0.762–0.778，且 timeout 显著下降，是目前最强正向证据。

我的意见：可以保留，但必须降级为“诊断轨”，不能直接训练推广

V1 同时改变：

行驶距离；
等效曲率 κ=dθ/ds；
LQR 前视参考； -速度和横向控制难度。

因此不能直接把 “expert ds × 1.25” 写进训练目标。否则可能只是掩盖系统性欠转、动作标定或 tracker-reference 耦合问题。

建议无条件轨先只做三个反事实：

ds × s, dθ 不变；
ds × s, dθ × s，近似保持曲率；
仅修正模型动作中实测的 ds/dθ bias。

只有第 3 项在不同路型、速度区间都稳定改善，才允许进入训练 target calibration。

所以这是执行顺序分歧，不是方向分歧。

对 coding agent 三项补充的裁决
1. 三特征集可辨识性探针：同意

一次测试三组输入非常有价值：

现有正式 observation；
observation + margin/history；
observation + plan geometry。

这能直接回答：

是现有观测足够，只需 gate；还是必须增加几何状态；还是 plan 自身已经包含充分的危险信号。

但标签必须严格定义为：

在当前同一状态下，
baseline plan 将失败，
且 expert intervention 能在规定 horizon 内恢复


不能简单用“位于失败窗口”作为正标签，否则 probe 可能只学到场景难度或 episode 时间。

2. Recovery 自然占比 2%–5%：同意

同意使用自然并入和显式 loss weight，禁止不留痕迹的约 40 倍 batch 过采样。

必须记录：

raw sample ratio
sampler exposure ratio
loss weight
effective importance ratio


这能避免再次出现少量 recovery 样本重写正常驾驶分布的问题。

3. Recovery_v2 增加 mode 标签：有条件同意

可以存粗标签，但只能用于：

数据分层； -误差分析； -未来 4B 的研究输入。

暂时不能作为训练目标或 4B 启动依据。
 必须先验证 mode 的稳定性、类内方差和相邻 step 切换率。

建议锁定的实施方案
Step 1：可辨识性 Gate

使用同状态数据构建正负样本，跑三组输入。

产物
docs/p1_v2_decision.md
runs/p1_v2/intervention_probe.json
runs/p1_v2/intervention_probe_by_slice.json

验收
ROC-AUC ≥ 0.80：允许 gated residual；
0.65–0.80：只允许增加明确的 margin/plan 特征后重测；
<0.65：停止条件 recovery 监督，转向 observation 改造或全局策略优化；
2–4 m/s、tollgate、merge、roundabout 必须分别报告；
precision@10% 必须足够高，建议至少 0.60，避免 gate 在正常状态频繁误触发。
Step 2：Recovery_v2 pilot

先采 2k–3k 行，不立即追求 10k。

每行必须来自同一状态：

obs_t
history_t
student plan_t
student action_t
expert action_t
expert recovery plan_t
actual state_t
margin / T_plan / T_cross
intervention label
recovery outcome

验收
observation/label 同状态匹配率 100%； -随机重放至少 1,000 行；
episode split 无泄漏；
expert recovery 成功率 ≥80%；
action-chain 与实际 recovery trajectory 一致； -禁止 Pass-1 observation 配 Pass-2 label。
Step 3：Gated residual pilot

结构保持简单：

final_plan = base_plan + gate(obs, base_plan) * residual(obs, base_plan)

训练纪律
正常样本监督 gate=0；
recovery 样本监督 residual=expert_plan-base_plan；
residual 仅在 intervention 有效时训练；
recovery 自然占比 2%–5%；
gate 与 residual 分开记录梯度；
base plan 保持 anchor，避免正常驾驶能力漂移。
clean150 验收
success 不低于起点 2pp；
off-road 不增加；
collision 不增加超过 2pp； -第二个 epoch 不崩塌； -正常状态 gate 激活率 <5%； -需要 intervention 的样本 gate recall ≥60%；
BC 首步 MAE、6 点 ADE 恶化不超过 10%。
Step 4：几何代理 Gate

在固定、不训练的数据上评估 ld_line_margin_loss：

成功 expert active-loss <5%； -普通道路 <2%； -模型失败 plan 的激活率显著高于成功 plan；
2–4 m/s 单独统计； -合法变道、分叉、收费站不得被系统性误罚。

不通过时：

该 loss 只允许用于监控和 hard mining，不进入反向传播。

当前 grad cosine=-0.73 已经说明不能靠减小权重硬塞进训练。

建议关闭的事项

以下内容本轮不再讨论或尝试：

recovery_v1 更多训练； -新的 mix ratio； -新的 freeze recipe；
correction/late 子集组合； -直接 residual、无 gate； -直接开启联合 geometry recipe；
P1-4A/4B；
K-anchor、GRPO、WM/MoE 重构； -新的容量实验。
最终共识状态
已达成共识
Recovery_v1 监督路线失败的根因； -可辨识性优先；
Recovery_v2 必须同状态；
gated residual 优于直接绝对目标； -几何 loss 必须先通过误报 Gate； -复杂 PPO 暂停； -后续以小 pilot 和严格停止条件推进。
唯一保留分歧

Coding agent 希望并行推进 V1 风格的无条件目标重标定；我认为它可以并行诊断，但在分离 ds / curvature / tracker 三种作用前，不应直接进入正式训练。

建议最终落板
P1-v2：LOCKED

主线：
可辨识性 probe
→ recovery_v2 pilot
→ gated residual pilot
→ geometry FP gate
→ 独立小消融
→ candidate recipe

旁线：
V1 机制分解，仅诊断；
未经 bias/曲率/路型验证不得改训练 target。

冻结：
recovery_v1 训练、P1-4、K-anchor、GRPO、
WM/MoE 重构及新容量扫描。


因此，双方已经能够锁定 P1-v2。剩余分歧不会阻塞主线，只需将“无条件校准轨”明确标注为诊断性旁线，而不是新的训练主方案。