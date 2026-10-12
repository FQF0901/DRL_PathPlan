1. 聊天记录是否丢失

是的，界面上下文曾被压缩，所以我上一轮只看到 Round-15 附近的摘要，未完整呈现 Round-16/17 的执行过程。你这次补充的 chatgpt5p6sol_review.md 和 deepseekv4p1_argue.md 已补齐关键链路：

Round-15A
→ A1 当前动作审计
→ A2 确定性尾部 oracle
→ Round-17 A3 训练
→ A3 决定性失败
→ 转 Round-18


最新关键结果是：

expert-state：MAE_ds=0.064m，MAE_dθ=0.006rad；
learner-state：误差扩大约 6.4 倍和 16.5 倍，dθ 符号错误率 21.7%；
正确专家当前动作加确定性尾，成功率可到 0.687–0.780；
模型动作加同样尾部，成功率为 0，off-road 为 1；
同一个 A3 checkpoint 使用原 chain 模式仍有 0.353；
因此 chain 尾部并非纯粹“脆弱远期预测”，它还在为模型首步动作噪声提供执行缓冲。

这确实改变了此前“删除未来链监督即可”的判断。Coding agent 对 A3 的 NO-GO 是正确的。

2. Covariate shift 是根因之一，但不能解释全部问题

你的判断成立。DAgger 正是通过在 learner 实际访问状态上查询 expert，并把这些 observation-action pair 聚合回训练集，处理行为克隆的状态分布偏移。行业实践也把 compounding error 和 learner-state 数据缺失视为 IL 的核心问题。

但目前不能把一切都归于 covariate shift，因为现有证据同时支持另一个问题：

网络可能没有建立稳定的“自车相对道路几何”表示。

如果只是普通 covariate shift，理论上同一类型的小幅 lateral/yaw 偏移仍应保持动作连续性。但当前 learner-state 的 dθ MAE 从 0.006rad 增至 0.099rad，且直道状态出现 ±0.09–0.23rad 抖动。这不仅是“没见过该状态”，也可能说明模型没有真正理解：

自车位于哪条 lane/corridor；
自车相对 lane center 的横向偏移；
自车 heading 与 lane tangent 的夹角；
应该朝哪个方向回正；
lane boundary 和 route centerline 的区别。

因此 Round-18 不应立即在原输入上做 A4 DAgger。应先判断：

是看过的状态不够
还是道路几何本身不可辨识
还是有信息但网络没用

3. 你对车道线输入的 concern 很合理

当前 repo 的 LD 表示存在结构性不足：

LD 是 16×7 的稀疏采样点； -前向 offset 主要为 5、10、15、20、30m；
lane 槽位采用当前车道优先及其他 lane 排序；
LD 没有像 ego/OD/others 一样进入主要 temporal memory；
plan head 没有被可靠的 lane-direction 或 centerline-alignment 目标约束； -现有 ld_line_margin_loss 的 FP Gate 已失败，说明当前 LD 语义不能直接充当可靠可行域监督。

行业中的向量化规划方法通常不会只把 lane points 扔给网络并期待其隐式学会驾驶规则。以 VAD 为例，它显式使用：

road-boundary overstepping constraint；
ego-lane direction constraint；
ego-agent collision constraint。

其中 lane-direction constraint 直接约束自车规划方向与最近 lane vector 的方向一致。

车道图研究也强调，规划需要的不只是独立车道点，而是 lane connectivity、方向和拓扑关系；LaneGAP 特别强调 path-wise representation 保留车道连续性，TopoNet 则显式建模 lane-lane 和 lane-traffic-element 拓扑。

所以当前 repo 的风险不是简单的“LD 维度少”，而是：

LD 表达了若干线段点，却没有明确表达“哪条是自车应跟随的中心路径、自车相对它在哪里、它的方向和曲率是什么”。

4. 也确实可能存在动力学外推 shortcut

模型在 expert-state 上当前动作很准，并不能证明它理解 lane。它可能只学到：

上一动作
+ 当前速度
+ yaw rate / heading trend
→ 下一动作


当前 ego 的末两维还包含历史动作，memory 又能读 ego 历史。这为 shortcut 提供了条件。

在 expert 轨迹上：

-车辆本来就沿 lane； -历史动作本来就平滑； -仅靠动力学延续也能得到低 MAE。

进入 learner-state 后：

-一旦横向偏离； -历史动作继续暗示旧运动趋势； -网络看不懂 lane-relative error； -便无法产生方向正确的 correction。

这与“expert-state 很准、learner-state 崩溃”的结果完全兼容。

所以 A1 的低 teacher-state MAE 不能被理解为 primary 已经掌握 lanekeep。

5. Round-18 应先做四个 lane-grounding probe

这些 probe 成本比重新训练低，而且可直接判断你的 concern。

Probe A：LD 消融与反事实

同一 observation 分别运行：

正常 LD
LD 全零且 mask=0
LD 复制为直道
LD 左右镜像
LD 沿横向平移 ±0.5m / ±1.0m
LD 旋转 ±5° / ±10°
仅保留 ego/history，去掉 LD
仅保留 LD/nav，清除 prev_action


检查：

mu_dθ 的方向和幅度
plan curvature
policy/value
MoE expert/router


预期基本关系：

将 lane 向左平移，模型应适当向左修正；
lane tangent 左旋，模型的 dθ 应同方向变化； -左右镜像后输出应近似镜像； -清除历史动作后，输出不应完全失去 lane-following 能力。

如果这些关系不成立，说明模型没有 grounded lane semantics。

Probe B：Lane-relative 线性 probe

从不同中间表示预测：

signed lateral offset
heading error to lane tangent
local lane curvature
left/right boundary distance
target-lane identity


分别测：

encoder LD token
fusion latent
memory latent
primary hidden
MoE residual input


若原始 LD 可以计算这些标签，但 latent 无法线性解码，问题在 encoder/fusion；如果 latent 可解码但 action 不响应，问题在 planner head 或监督。

Probe C：动力学 shortcut probe

对相同道路几何，交换或扰动：

previous action
ego history
current speed/yaw history


同时保持当前 ego pose 与 lane geometry 不变。

比较：

正常输出
历史动作清零
历史动作反号
历史帧重复 current
历史顺序打乱


若输出对 prev_action/history 极敏感、对 LD 反事实不敏感，基本可确认 shortcut。

Probe D：基础 lane-following counterfactual grid

构造简单直道和定曲率道路网格：

lateral offset: -1.0, -0.5, 0, +0.5, +1.0m
heading error: -10°, -5°, 0°, +5°, +10°
speed: 2, 5, 10m/s


测模型 dθ 是否具备正确恢复符号：

offset 左 → 合理向右回正
heading 偏左 → 合理向右修正
误差增大 → 修正幅度大体单调
零误差 → dθ 接近 lane curvature 所需值


这是对“primary 是否具备基本 lanekeep”的最直接单元测试。

6. 对 MoE scope 的判断：你的定义是对的

我同意：

Lanekeep 必须属于 primary 的基本能力，不应由 specific experts 补偿。

合理职责应是：

Primary

负责高覆盖、低熵、连续的基础驾驶：

lane center / route corridor following； -与 lane tangent 对齐； -速度与曲率匹配； -基础前车跟随； -平滑、可跟踪轨迹； -轻微偏离后的常规纠正。
Specific experts

只负责明确的条件残差：

tollgate 多通道选择；
merge/yield； -复杂交叉口； -避障绕行； -极端恢复； -特殊道路拓扑。

如果正常直道 lanekeep 需要 expert residual 才成立，架构职责已经倒置：

primary 不是基础策略
experts 也就不是残差专家


更严重的是，当前 router 缺少明确语义监督，仅用负载均衡，无法保证 expert 学的是“复杂场景补偿”，而不是随机分摊基础驾驶。

因此在 primary lane-following probe 通过前，不应解冻 experts，也不应启动 A4 全量 DAgger。

7. 我对 Coding Agent Round-18 候选的裁决
a. “增加动作可持性约束后重跑 A3”

暂不批准训练，先做 probe。

它可能降低 dθ 抖动，但也可能只是把输出压向 0，掩盖网络不理解 lane。

若要使用，只能定义为：

预测动作相对 lane/reference curvature 的一致性


不能简单惩罚：

dθ 小
dθ 时间变化小


否则急弯时会欠转。

b. “重议 A4 DAgger”

条件批准，但必须晚于 lane-grounding probe。

若 probe 证明：

LD 反事实响应正确
lane-relative 信息可解码
主要缺口仅在 learner-state 覆盖


则 A4 是正确下一步。

若 probe 证明网络基本不使用 LD，直接加 DAgger 可能产生两个结果：

-靠更多状态隐式记忆修正； -继续依赖 ego dynamics shortcut。

它可能短期提升，却没有建立可泛化 lanekeep。

c. “保留 chain 作为执行噪声缓冲，集中做 learner-state 数据”

原则同意。

A3 已证明 fixed repeat tail 对有噪模型动作过于敏感，而 learned chain 能缓冲首步抖动。不要为了架构简洁删除一个已被证明确有执行价值的模块。

但 chain 的目标应重新定义为：

可跟踪的平滑 reference generator，而不是精确预测未来专家动作。

可以保留现有 chain inference，后续逐步将其监督改为：

-第一步 action consistency； -曲率连续； -可跟踪性； -lane direction； -boundary/corridor safety。

d. “obs 表示改造”

暂不直接批准改造，但批准优先诊断。

先用上述 probes 确认缺什么。如果确实缺 lane grounding，建议最小新增显式特征：

ego-to-target-centerline signed lateral offset
ego heading error to target-lane tangent
target-lane local curvature
left/right drivable margin
target-lane validity / confidence
route-lane association


这比继续增加 raw LD 点数更有效。

行业方案的共同点也是显式保留 geometry、direction 和 topology，而不是期待一个无结构 point set 自动形成道路语义。

8. 建议锁定的新顺序
Round-18A：Lane-grounding 与 shortcut probes
    ↓
若 primary 已理解 lane：
    A4 current-action DAgger
    保留 learned chain 作为执行缓冲

若 primary 不理解 lane：
    增加最小 lane-relative 显式特征
    primary-only lane-following 监督
    通过 counterfactual grid 后再做 DAgger

Round-18A 验收

至少要求：

1. lane 左右镜像后 dθ 基本镜像；
2. lateral/yaw error 的修正符号正确率 ≥95%；
3. 修正幅度随误差总体单调；
4. 清除 prev_action 后仍保留基本 lane-following；
5. LD 移除后性能或输出显著变化，证明网络实际使用 LD；
6. primary-only 即可完成直道/缓弯 lane keeping；
7. experts 关闭时基础场景性能不应明显下降。


若第 5 条不成立，就是显著 shortcut 证据。

9. 关于网上行业经验的落点

外部研究支持三点：

DAgger 解决 learner-state covariate shift，但需要在 learner 真实访问状态查询 expert，而不是混入错位 recovery target。
道路结构应该以 centerline、boundary、direction、connectivity 等结构化关系进入规划，而不能只依赖隐式特征。
规划最好同时具备隐式 map interaction 和显式 lane-direction/boundary constraint。 VAD 的 lane-direction 和 boundary constraints 正好对应你担心的“网络是否理解应该沿车道方向走”。

这并不意味着照搬 VAD。当前 repo 已经有结构化 OD/LD/nav，最合理的动作是补足 lane-relative semantics 和 probes，而不是引入大型 BEV 或重写全网。

最终判断

目前最可能的 root cause 不是单一问题，而是两层叠加：

第一层：primary 可能主要学到 expert-state 动力学 shortcut，
       没有稳定掌握 ego-lane relation。

第二层：部署后进入 learner-state，covariate shift 放大该缺陷，
       dθ 符号和幅度迅速失控。


因此：

Lanekeep 必须回归 primary 基础能力。
specific MoE 不应承担日常 lane correction。
A4 DAgger 很可能需要，但必须先证明模型看得懂 lane。
learned chain 暂时保留，作为执行噪声缓冲。
不要立即用“可持性正则”把 dθ 压小，那可能掩盖 lane grounding 缺失。

下一步最优先不是再训练，而是用一组强反事实 probe 裁决：

网络究竟在看车道，还是只在延续自己的运动历史。