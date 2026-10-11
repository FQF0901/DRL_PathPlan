结论

附件中的 deepseekv4p1_argue.md 仍停留在 §7，内容基本是上一轮方案，没有看到针对我最新意见的新增回应。就当前文本而言：

Round-12A 方向正确，但方案尚未完全闭合。可以开始实现遥测，暂不应直接跑完整筛选矩阵，更不能批准 12B/C。

主要缺口不是小细节，而是三个可能改变根因判断的问题。

一、认可的部分

Coding agent 做对了四件事：

承认现有产物没有逐步 gate、residual、base/final plan 数据，不能假装可以离线复盘。
改为零训练闭环重跑，而不是立即补数据、重新训练。
threshold、residual scale、debounce 顺序筛查，而非 60 个全网格盲扫。
预先接受“有效操作区域为空”这一结果，避免为了保住方案而继续调参。

因此，逐步遥测和 gate-off/gated 重跑可以立即开工。

二、尚未回应的三个关键问题
1. 缺少 CPU/CUDA 后端一致性检查

文档已记录 60 个 episode 有 10 个 CPU/CUDA 结局翻转，但 §7.4 没有把它纳入 Round-12A。

这是严重缺口。当前可能存在：

CPU collector 状态与标签
≠
CUDA 部署时 gate 实际看到的状态


如果 gate probability 或 residual 对后端微小差异敏感，补 hard negatives 只是拟合错误分布，无法解决根因。

Round-12A 必须补同输入对照：

base plan 差异
gate probability 差异
residual 差异
首步动作差异
首次轨迹分叉时间
最终 episode 结果


若差异明显，优先统一 collector 和部署后端，不进入 12B。

2. “逐步 diff”必须限定在首次轨迹分叉前

gate-off 与 gated 一旦执行不同 action，下一时刻 observation 就不同。此后逐步比较不再是同状态反事实。

报告必须明确分为：

Common prefix：
首次轨迹分叉前，可用于机制归因

Post divergence：
分叉后的闭环结果，只能用于效果描述


否则很容易看到 gated 后续 margin、速度、gate probability 不同，就错误归因于 gate 本身。那些差异也可能只是早期动作分叉后的连锁结果。

必须记录：

first_action_divergence_step
first_state_divergence_step

3. “正常 episode 激活率低于 5%”定义仍不正确

平均 step 激活率不能代表安全性。一次错误激活就可能把成功 episode 变成 timeout，而一个真正危险 episode 连续激活多步可能完全合理。

应改成 episode 级指标：

破坏性误触发 episode 比例
首次误触发时间
每 episode 激活段数
最长连续激活长度
累计 residual_ds / residual_dtheta
激活后的 progress 损失


“破坏性误触发”建议定义为：

gate-off 成功，但 gated 在首次激活后没有改善几何余量，并最终变成 timeout、off-road 或 collision。

Go 条件应是：

破坏性误触发 episode 比例 <5%


而不是简单的正常 episode 平均激活率低于 5%。

三、对 Round-12A 扫描方案的进一步质疑

当前提议是：

先扫 threshold；
在最优 threshold 上扫 residual scale；
再扫 debounce。

这是一种贪心搜索，隐含假设三个变量近似独立。但它们可能强耦合：

低 threshold + 小 residual 可能有效；
高 threshold + 大 residual 可能有效；
单独扫 threshold 时，两者都可能被错过。

不需要跑完整 60 格，但建议在顺序扫描后补一个小型交叉验证：

选择最优的 2 个 threshold
× 最优的 2 个 residual scale
× debounce {1, 2}
= 8 个组合


这样可以验证顺序搜索是否遗漏明显交互。

仍然属于零训练诊断，不算扩大模型搜索。

四、对“残差有效、只是 gate 拖累”的反驳

当前证据仅能说明：

8 个失败 episode 被修复；
13 个成功 episode 被破坏；
off-road/collision 减少；
timeout 增加； -整体 success 减少 3.33pp。

因此最准确的结论应该是：

residual 对部分状态有局部收益，但适用条件、幅度安全性和长期 progress 影响尚不明确。

尚不能断定瓶颈只在 gate。还可能是：

residual 主要通过减小 ds 避免越界，却造成 timeout；
residual 只适合某些 recovery 类型；
repeated activation 导致修正累积过量；
gate 和 residual 联合训练后存在共适应；
CPU/CUDA 口径差放大 gate 或 residual 输出。

Round-12A 必须同时分析：

gate precision
residual magnitude
residual direction
累计纵向减速
连续触发
backend sensitivity

五、对 12B hard negatives 的限制

即使 12A 证明 gate 误报，也不能直接把 gated 模型遇到的状态全部标成负例。

Hard negative 必须经过同状态双 rollout：

baseline rollout：
安全、无碰撞、进度正常

expert rollout：
相比 baseline 没有显著提高成功率、margin 或 progress


只看当前六点 base plan 在路内不够。当前系统的问题本来就可能在后续重规划逐渐漂移。

如果无法做同状态双 rollout，12B 数据的标签仍不可靠。

六、建议正式锁定的 Round-12A
第一阶段：基础遥测

先只实现并验证：

gate_prob
gate_activation
residual action/plan
base plan
final plan
tracker reference
actual trajectory
speed
route progress
footprint validity
termination
CPU/CUDA backend


原始数组保存 NPZ 或 Parquet，JSONL 仅保留标量和索引。

第二阶段：基准重跑

同一预注册 episode 集：

gate-off
原 gated 配置
CPU/CUDA 对照子集


先确认历史结果可复现，再开始扫描。

第三阶段：机制筛查

执行：

threshold sweep
residual scale sweep
debounce sweep
小型 2×2×2 交叉验证


分析严格区分：

common-prefix causal analysis
post-divergence outcome analysis

第四阶段：Go/No-Go

进入 12B 至少需要：

存在非空有效操作区域
保留至少 6/8 个历史翻正
消除至少 7/13 个历史翻负
timeout <= baseline +3pp
破坏性误触发 episode 比例 <5%
离线 intervention recall >=60%
CPU/CUDA 一致，或 collector 已统一部署后端

七、最终裁决
可以立即做
实现逐步遥测；
gate-off/gated 基准重跑；
CPU/CUDA 一致性探针；
common-prefix 分析；
threshold/scale/debounce 零训练筛查；
geometry FP 只读诊断。
暂不批准
hard-negative 数据采集；
gated residual 再训练； -几何 loss 反向传播； -修改 gate/residual 架构； -进入 4A/4B。
总体判断

Coding agent 的方向是理性的，但当前文本仍略早地把问题定性为“数据覆盖/标定级，而非架构级”。现有证据还不足以排除：

后端不一致；
residual 纵向副作用； -连续激活累积；
gate-residual 共适应； -闭环不可辨识。

所以建议：

批准 Round-12A 的遥测与诊断实现，但补齐 CPU/CUDA、common-prefix 和 episode-level 破坏性误触发三个约束后再运行。

若 Round-12A 找不到非空有效操作区域，应直接关闭 gated residual，不再通过补数据或加模型复杂度挽救它。