结论：已达成共识，可以开工

Coding agent 已完整接受上一轮核心约束，并进一步补齐了后端一致性、common-prefix 归因和 episode 级破坏性误触发分析。当前没有阻塞性分歧。

建议将本阶段正式锁定为：

Round-12A：零训练诊断，LOCKED

立即执行范围

实现逐 step 遥测：

gate probability
residual (ds,dθ)
base/final plan
footprint、lane offset、progress、speed
first action/state divergence

重跑 gate-off 与 gated 基线，确认复现。

做 CPU/CUDA 一致性探针：

单状态前向差异；
threshold 跨界率；
30–50 episodes 端到端对照。

做轻量闭环扫描：

threshold；
residual scale；
debounce；
最后补 2×2×2 交叉验证。

输出 common-prefix 与 post-divergence 分离的归因报告。

做 Go/No-Go，允许“无有效区域”这一结果。

两点最终确认
1. CPU/CUDA 探针范围合理

不仅比较 gate 输出，还要比较完整链：

F2m features
→ base plan
→ gate
→ residual
→ final plan
→ threshold crossing


长时闭环结果可能被很小的门槛差异放大，因此必须包含端到端 episode 子集，不能只检查 tensor 误差。

2. 后端差异的处理不能临时决定

如果 CPU/CUDA 导致明显 threshold 跨界或闭环结局变化，优先选择：

训练、采集、校准和正式评测统一到 CUDA 口径。

不建议为了兼容已采 CPU 标签而把正式评测迁移到 CPU，因为当前基线 0.3400 是 CUDA 口径，改变评测后端会破坏历史可比性。若重抽 CUDA 特征和标签，必须保留原数据版本及差异报告。

Go/No-Go 建议锁定

只有扫描后存在一个非孤立有效区域，同时满足以下条件，才允许进入 12B/12C：

success >= baseline - 2pp
off-road 不增加
collision <= baseline + 2pp
timeout <= baseline + 3pp
破坏性误触发 episode 比例 < 5%
保留至少 6/8 个历史翻正
消除至少一半 13 个历史翻负
结果对相邻 threshold/scale 稳健
CPU/CUDA 不造成实质性策略分叉


这里“非孤立有效区域”很重要。不能只有某一个精确 threshold 偶然通过，否则属于对 clean150 调参，不具备部署稳定性。

若没有满足条件的区域：

直接关闭 gated residual 主线


不再补 hard negatives、不扩大 gate、不调整采样比例。

暂不批准

继续冻结：

hard-negative 采集；
gated residual 重训；
geometry loss 反传；
gate/residual 架构修改；
P1-4A/4B；
K-anchor、GRPO、WM/MoE； -任何新容量扫描。
最终意见

可以让 coding agent 开工。

本轮只做零训练诊断，目标不是寻找最好超参数，而是回答三个问题：

gate 是否误报；
residual 是否过强或方向错误；
连续触发和 CPU/CUDA 差异是否放大了闭环损失。

这一轮结束后必须作出明确裁决：进入一次性修复实验，或者关闭 gated residual 路线。