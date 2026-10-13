总体裁决

这轮是截至目前最接近可用方案的一次正向实验：

Q1 相对 Q0，在扰动状态、learner-state、方向错误、修正幅度和镜像一致性上全面改善；
calibration50 达到 0.50，相对 legacy 的 0.40 提升 10pp；
MoE-off 同口径下，Q1 达到 0.52，对照约 0.26；
语义动作修正的 P99 已基本贴近专家，说明 residual target、幅度软锚和完整镜像增强确实生效。

但仍然没有达到正式候选门：

正常 expert-state MAE 恶化约 40%–44%； -转向饱和增加约 14.5pp； -没有连续两个 epoch 同时满足全部机制门； -尚未完成 clean150 独立验证，更谈不上 eval500 和超越 IDM。

因此我的结论是：

Round-26 方向批准，Q1 保留为当前最佳候选；但不批准直接进入 DAgger、大数据、MoE 或 RL。下一阶段只关闭“正常态保持”和“转向饱和”两个问题。

一、代码实现审阅
1. Residual target 实现正确

当前监督是：

模型语义增量：
mu_semantic_on - mu_semantic_off.detach()

目标增量：
expert_action - legacy_action


这里 semantic_off.detach() 与冻结 legacy base 非常重要：

梯度不能通过修改 base policy 来“伪造 residual”； -语义路径必须解释 expert 与 legacy 的动作差； -正常状态 expert≈legacy 时，目标 residual 自动接近零。

这比 Round-25 直接让 semantic 强力改写完整动作合理得多。

2. 幅度锚定已发挥作用

Q1 的模型 Δdθ P99≈0.231，专家约 0.227，P99 比率约 1.017；不同分组的模型修正量也已接近 expert，而 Q0 超过 expert 两倍。

这证明软锚确实解决了“semantic 修正量爆炸”的主要问题。

但要注意：

输出 residual 分位数正确，不等于最终 LQR 控制强度正确。

因为 LQR 消费的是完整六步 plan。即使第一步 residual 合理，后续 chain 的累计曲率、连续触发和速度耦合仍可能造成转向饱和。

所以下一轮不能只继续收紧 Δdθ 的 P95/P99。

3. Full-mirror 配对基本有效

Q1 镜像残差约 0.014–0.040，明显优于 Q0 的 0.042–0.059，说明完整状态镜像的设计有效。

这一方向与行业经验一致：ChauffeurNet 通过合成扰动、环境约束和运动历史 dropout，迫使模型依据道路和环境因果线索，而非只复制专家历史。

但目前镜像增强的训练分布仍主要来自合成 B 行：

traffic=0； -置位后 history 与真实连续 learner history 不完全同分布； -高速覆盖不足； -复杂路型扰动较少。

所以不能把 synthetic mirror 的通过等价为真实闭环对称性已经解决。

4. Label fail-fast 修复正确，但历史报告必须标记失效

旧 B-label-v1 存在 observation 与 expert label 状态错配，修复后相关性从负值变成明确正值。旧标签现在已被 fail-fast 禁止，且不可复现的 4 行被剔除，这是正确做法。

必须进一步完成两件事：

所有历史报告加统一勘误标签；
任何 checkpoint selection、probe 或统计若曾读取 v1，都应标记为不再具有动作监督证据。
二、Q1 没过门的真正原因

Coding Agent 将剩余问题概括为：

teacher-state 保持性
+
steer saturation


这基本正确，但还需继续追一层。

1. Teacher MAE 恶化不一定只是“零修正约束不足”

它可能来自四种机制：

A. 正常态 residual target 并非真正为零

Legacy action 与 expert action 本来就有差值，即使 expert-state 行驶正常：

expert_action - legacy_action != 0


如果在全部正常样本上学习这个 residual，semantic 仍会系统性重写 legacy policy。

B. 新语义路径在 normal expert 数据上过度激活

虽然 residual target 较小，但大量 normal expert 样本的累计梯度，可能推动 beta_context 形成持续非零修正。

C. 训练正常态的 expert residual 会覆盖 legacy 的闭环补偿习惯

Legacy 与 learned chain 是联合工作的。单步 expert action 更准，不代表替换 legacy action 后的完整 plan 更稳定。

D. 数据分布不平衡

Normal expert 行数较多，但真正决定 lane correction 的大误差状态较少。若 loss 权重不合理，模型可能同时：

-在 normal 上学不必要的细微修正； -在 learner-state 上学过强修正。

因此不建议简单把 normal residual target 全部强制为严格零，这可能阻止修复 legacy 的系统性 bias。

2. Steer saturation 来源尚未定位

当前只知道：

最终 steer saturation 增加14.5pp


还不知道来自：

第一步 dθ 过大； 2.六步 chain 累计曲率过大； 3.相邻计划点 Δκ 不连续； 4.每次重规划 residual 符号来回切换； 5.速度 ds 与 dθ 比例改变，导致等效曲率增大； 6.LQR preview 对尾段过度响应。

行业中的 VAD 不只监督动作或轨迹误差，还显式加入 lane-direction、road-boundary 和 collision constraints，因为几何方向正确并不能自动保证规划轨迹安全、连续且可执行。

所以 Round-27 前必须先完成 saturation attribution，而不是直接增加“饱和惩罚”。

三、建议的 Round-27

目标只剩两个：

保留 Q1 的 learner-state 修复收益
同时恢复正常驾驶基本盘并降低 LQR 饱和


不改模型容量，不采新大数据。

27A：先做零训练饱和归因

对 Q1 与 legacy，在相同 calibration50 + clean150 episode 上逐策略步记录：

mu_ds / mu_dtheta
semantic Δds / Δdtheta
plan[0:6]
每步 kappa = dtheta / max(ds, eps)
delta-kappa
LQR reference curvature
raw steer / clipped steer
heading error
lateral error
speed


将饱和分为：

首步动作型
尾段chain型
曲率不连续型
重复重规划累积型
低ds高dtheta型
场景必要饱和

必须输出
首步导致的饱和比例
尾部导致的饱和比例
成功episode中的饱和率
off-road前饱和率
timeout前饱和率
按road class分组


没有这个归因，不允许设计新的饱和 loss。

27B：正常状态 residual 分解

将 expert-state 分成三类：

N0：Legacy 已足够正确

定义例如：

|expert - legacy| 小
legacy计划可行
未来H步无失败前兆


目标：

semantic residual ≈ 0

N1：Legacy 有稳定小偏差

例如跨路型一致的 ds 低估或轻微曲率 bias。

目标：

允许小 residual

C：Perturbation / learner correction

目标：

学习完整 expert residual


这样比“所有 teacher-state 一律零修正”更合理。

四、Round-27 仅允许两个候选
R0：Q1 重跑基线

固定 Q1 代码与数据，用于同批比较。

R1：Q1 + normal-state preservation + saturation-aware regularization
Normal-state preservation

对 N0 样本：

Lpreserve=∥Δasemantic∥1L_{preserve} = \|\Delta a_{semantic}\|_1

但只对“legacy 已足够好”的样本启用，不能覆盖所有 teacher-state。

Saturation-aware regularization

根据 27A 归因选择：

若第一步曲率过大

约束：

Δκ0=dθfinaldsfinal−dθlegacydslegacy\Delta \kappa_0 = \frac{d\theta_{final}}{ds_{final}} - \frac{d\theta_{legacy}}{ds_{legacy}}

相对 expert residual 做软锚。

若尾段累计曲率导致饱和

约束：

LΔκ=∑t∣κt+1−κt∣L_{\Delta\kappa} = \sum_t |\kappa_{t+1}-\kappa_t|

但只惩罚超过 expert/legacy 正常分布的部分，不能全面追求小曲率。

若连续重规划抖动导致饱和

对相邻 decision step 的 semantic residual 加时序平滑：

Ltemporal=∥Δat−Δat−1∥L_{temporal} = \|\Delta a_t-\Delta a_{t-1}\|

仅在 target lane 和 route context 未发生变化时启用。

五、一个关键反驳：暂时不要增加更大“teacher 零修正权重”

如果直接把正常 expert residual 压到零，可能保住 teacher MAE，却失去 Q1 改善 off-road 的核心能力。

ChauffeurNet 的经验不是简单增加 imitation 权重，而是同时使用合成扰动、环境约束、progress 等任务，并通过 past-motion dropout 限制历史 shortcut。

当前 repo 也应避免再走向：

为了保住正常态
→ semantic大部分时间被压成0
→ learner-state收益消失


所以 preservation 必须基于 N0 条件，而不是面向全部 expert 行。

六、是否需要 history dropout？

值得加入诊断，但暂不进入 Round-27 正式候选。

当前已经证明 history shortcut 长期存在。ChauffeurNet 在一部分训练样本中去掉过去运动历史，迫使模型查看道路和环境线索，这是高度相关的行业方案。

建议先做一个小离线实验：

正常history
history dropout 25%
history dropout 50%


仅观察：

teacher MAE
learner MAE
lane response
history/lane ratio


若 25% dropout 能降低 learner error 且 teacher degradation 可控，再进入后续正式训练。不要与 Round-27 R1 同时加入，否则无法归因。

七、DAgger 是否即将具备条件？

还没有。

DAgger 用 learner 实际访问状态向 expert 查询动作，正是解决 covariate shift 和误差累积的标准路径。

但当前 Q1 已经在 learner-state 上减错 51.5%，却仍存在：

teacher MAE +40%； -饱和 +14.5pp； -没有连续两个 epoch 稳定通过。

这意味着现在加入更多 learner-state，可能放大“过强修正”。

DAgger 启动门仍应是：

teacher degradation <=10%
saturation <=legacy+5pp
clean150 >=0.42
连续两个epoch机制门通过

八、Round-27 验收标准
Gate A：饱和归因完成
>=90%的新增饱和可归入明确机制
首步/尾段/抖动/低ds高dtheta占比明确
按路型、速度、结果分层

Gate B：Normal-state preservation
teacher-state MAE恶化 <=10%
N0语义residual P95 接近0
N1小偏差修正不被完全抹除
learner-state dtheta MAE仍下降>=30%


Round-26 已下降 51.5%，允许部分回退，但不能低于 30%。

Gate C：控制可执行性
steer saturation <=legacy+5pp
semantic delta P99 <=expert P99×1.25
plan delta-kappa不高于expert P95×1.25
timeout <=legacy+3pp

Gate D：机制稳定性

连续两个 epoch：

sign >=95%
mirror >=95%
curvature >=25%
history/lane <=1
teacher degradation <=10%

Gate E：clean150
success >=0.43
relative legacy >=+8pp
off-road <=0.35
collision <=legacy+2pp
timeout <=legacy+3pp


达到后才跑 eval500。

eval500 开发门
success >=0.45
relative legacy >=+7pp
off-road相对下降>=15%
collision <=IDM
timeout无明显恶化


这仍只是确认 semantic 路线可扩展，不代表接近最终目标。

九、代码与文档需补的内容
model.py

保留：

residual beta； -target token 折叠； -static_role_only； -semantic-off detach 作为 base。

新增监控输出：

semantic_delta_action
legacy_action
final_action
plan_kappa[6]
plan_delta_kappa[5]
semantic_beta_context
semantic_beta_token


不要把训练专用标签逻辑写进模型 forward。N0/N1/C 分类应放在 trainer/dataset 层。

semantic_v1_dataset_card.md

Errata 已写得很好，但应再明确：

semantic_v1_pilot/samples.npz 中的 B expert_action 不得使用
有效动作标签必须来自 round26/baseline_labels.npz


最好在原 NPZ 消费入口直接拒绝读取 B 行的旧 expert_action，不能依赖使用者记得覆盖。

deepseekv4p1_argue.md

建议将 Round-25/26 的结论收敛成一句：

lane semantic 的表示、采用、幅度和对称性四层中，前三层已得到正证；当前仅剩 normal-state preservation 与 plan-level trackability 未闭合。

十、距离终极目标的现实判断

Q1 在 calibration50 达到 0.50 是重要进展，但样本只有 50 个，不能与 IDM 的 0.756 直接比较。

当前应按以下阶梯：

Round-27 clean150 >=0.43
→ eval500 >=0.45
→ 正式 semantic BC >=0.55
→ current-action DAgger >=0.65
→ 再考虑 MoE/PPO追平或超过0.756


行业方案同样表明，显式 lane-direction、boundary、collision 约束能提高规划安全性，但它们不是单独解决闭环分布偏移的万能药。VAD 通过向量场景交互与显式规划约束共同降低碰撞，而 imitation 系统仍需关注闭环分布和后续训练。

最终审批
批准
Round-27A：零训练 saturation attribution
Round-27B：N0/N1/C normal-state residual 分解
Round-27 R0/R1 两臂
条件 preservation
归因驱动的 trackability regularization

暂不批准
盲目提高teacher零修正权重
全局更严dtheta裁剪
正式大数据扩采
DAgger
MoE
RL
history dropout进入正式候选
新网络容量

给 Coding Agent 的推进指令
Round-26 方向确认，Q1 为当前最佳候选。

先对新增 steer saturation 做逐步归因：
首步幅度、尾段累计曲率、delta-kappa、连续重规划抖动、
低ds高dtheta、路型和结果分层。

然后仅做两臂：
R0 = Q1复现；
R1 = Q1 + 条件化normal-state preservation
     + 由归因决定的plan-level trackability regularization。

Normal preservation仅作用于legacy已安全正确的N0状态，
不得把全部expert-state residual强压为0。

必须达到：
teacher MAE degradation <=10%
learner MAE improvement >=30%
steer saturation <=legacy+5pp
timeout <=legacy+3pp
sign/mirror >=95%
curvature >=25%
clean150 success >=0.43且相对legacy +8pp

未达到：
停止继续增加semantic loss与数据；
不启动DAgger/MoE/RL。


这轮最重要的转变是：

问题已经不再是“网络是否理解车道”，而是“如何让正确的车道修正转化为平滑、可执行、且不破坏正常驾驶的完整计划”。