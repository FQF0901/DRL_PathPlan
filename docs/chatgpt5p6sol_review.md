总体判断

这轮取得了第一个可信的闭环正向结果：

success 从约 0.327 提升到 0.400，+7.3pp；
off-road 从约 0.553 降到 0.347，下降 20.7pp；
learner-state 的 dθ 误差下降约 24%； -扰动状态误差下降约 31%； -方向错误率下降约 58%。

这证明两件事：

显式 lane semantic 不是无效设计；
只有加入动作级语义监督，网络才会真正采用 lane semantic，普通 BC 会重新退回 history/dynamics shortcut。

但当前还远不能进入大规模数据、DAgger、MoE 或 RL，因为提升伴随着：

timeout 增加约 14pp； -转向饱和升至约 0.89–0.94； -teacher-state MAE 恶化约 57%； -镜像一致性未过门； -整个实验还发现了一个严重的扰动标签状态错配 bug。

所以这轮应定性为：

机制和方向获得关键正证，但控制幅度、对称性和正常驾驶保持性尚未闭合。

一、代码修改审阅
1. 残差式 semantic 接入正确

model.py 现在采用：

geometry_context
= legacy_context + beta_context × semantic_delta


以及：

effective_target_token
= beta_token × target_lane_token


beta_context 和 beta_token 都初始化为 0，因此载入旧 checkpoint 后：

action_mu
plan
rollout
value
router logits


与旧模型逐位相同。

实际验证 max|Δ|=0，说明此前的 feature-distribution shock 已被解决。

这是这轮最重要、最正确的工程修改。

2. Target token 的 mask 问题已合理规避

之前 target token 被追加为一个始终有效的独立 token，存在无效语义仍通过 projection bias 影响输出的风险。

现在 target token 不再增加新 token，而是：

ld_slot = ld_pool + target_proj(target_token)


并且 target_token 已经过：

beta_token * raw_target


无效 semantic 时 raw target 为零，初始 beta 也为零，因此不会影响输出。

这比仅仅修一个 mask 更稳健，还保持 token 数量恒为 39。

不过必须保留以下测试：

semantic invalid
→ target contribution 严格为0
→ action/value/plan 与 legacy 路径逐位一致

3. 单一标量 beta 简洁，但不是最终形态

当前：

beta_context: scalar
beta_token: scalar


优点是：

参数少； -解释性强； -能判断 semantic 是否被采用； -不容易过拟合。

但缺点也很明显：

-所有道路类型、速度和语义维度共享同一个强度； -无法区分 lateral、heading、curvature； -无法区分正常驾驶与恢复状态； -一个标量只能整体放大或缩小 semantic。

本轮不应立即复杂化，但 Round-26 应记录分组效果：

beta × road class
beta × speed
beta × lateral-error bucket
beta × heading-error bucket


如果同一 beta 在某些组明显过强、某些组明显不足，后续才考虑弱条件化 gate。现在不要提前引入。

4. P1 证明“动作级监督必需”，但监督幅度过强

三臂结果非常有信息量：

P0：只有普通 BC
beta_context ≈ -0.06
lane response 只有 0.13–0.6×
history/lane ratio 2.3–9.4


网络不仅不采用 semantic，甚至倾向于反向抵消它。

P1：动作级 semantic 监督
beta_context ≈ +0.10～0.12
sign ≈ 1.0
lane response 约 98×
history/lane ≈ 0.023


semantic 被强力采用，闭环也首次超过 legacy。

P2：进一步解冻 Primary

没有优于 P1。

因此可明确裁决：

Primary 本体暂不需要继续解冻。当前问题在动作监督尺度，不在 Primary 容量。

P1 的问题是从“完全不用 semantic”跳到了“过度使用 semantic”。

二、重大数据 bug 的影响
Bug 内容

受控扰动 B 行原先是：

obs = 人工置位后的扰动状态
label = 运行恢复窗口之后才查询的 expert action


因此 observation 与 label 不属于同一状态。

表现为：

corr(perturb_lat, expert_dtheta) = -0.156


修复为置位后立即查询 expert，重采 10,803 行后：

corr = +0.606


这不是小噪声，而是监督方向错误。

影响范围

以下结论需要标记为受污染：

R21-3B/3C 的扰动动作结果； -R21-3D-3 的 S0/S1； -Round-25 最初使用旧 B 标签的首跑； -任何基于旧 B 行得出的 mirror、direction、amplitude 结论。

表示层结果不一定全部失效，因为 lane semantic 本身仍正确，但动作采用相关结论必须以修复后的 10,803 行数据为准。

建议在所有相关报告中加入统一勘误：

B-label-v1: invalid for action supervision
B-label-v2: same-state expert label, valid


正式训练入口应永久拒绝旧版本 B 数据。

三、为什么闭环提高但仍不过门
1. Off-road 大幅下降说明修正方向有效

P1 将 off-road 减少约 20.7pp，这不是随机小波动。模型确实更能根据 lane geometry 纠正横向偏离。

2. Timeout 大幅增加说明修正幅度和纵横耦合错误

timeout 增加约 14pp，同时转向饱和约 0.89–0.94，说明模型可能在做：

过早、过强、持续的横向修正
→ LQR 大量转向饱和
→ 速度下降或路线进度不足
→ off-road 被部分转化为 timeout


因此不能只看 success 和 off-road 就宣布成功。

3. Teacher MAE 恶化 57% 表明基础驾驶被扰动

动作级语义监督在受控偏离状态有效，但也改变了正常 expert-state 上的动作。

这说明监督还没有学会：

正常状态：
semantic correction 应接近0

偏离状态：
semantic correction 应按需要增加


而是学成：

所有状态都对 lane semantic 高增益响应

4. 镜像一致性失败指向状态交互未覆盖

mirror-lateral 只有约 0.891，在大 heading error 样本上甚至只有 0.51–0.73。

当前 paired loss 主要覆盖 lateral-only 扰动，却没有完整覆盖：

lateral offset × heading error


例如：

-自车位于车道左侧，但车头已经朝右； -自车位于车道右侧，但车头继续朝右； -位置误差和航向误差要求的瞬时动作可能相互竞争。

仅根据 lateral sign 监督动作会产生错误标签。

四、Round-26 推荐方案

本轮不要再改网络结构，也不要扩大数据。只修两个已明确的问题：

动作修正幅度过强；
lateral 与 heading 的联合对称性不足。
26A：修复后的数据契约

只允许：

B-label-v2
同状态 observation / expert action
CUDA 同口径


永久禁止旧 B 行进入：

-训练； -验证； -checkpoint selection； -probe； -mirror 统计。

新增 fail-fast：

dataset.label_state_version == "same_state_v2"

26B：重新定义动作修正目标

不要直接让 semantic 输出拟合完整 expert action。应监督相对于 legacy policy 的修正量：

delta_expert
= expert_action - legacy_action


模型语义修正：

delta_model
= semantic_action - legacy_action


训练：

L_delta = SmoothL1(delta_model, delta_expert)


这样：

-正常状态 expert 与 legacy 接近，修正自然接近 0； -偏离状态才产生较大修正； -旧驾驶基本盘不会被整体重写。

26C：幅度锚定
推荐软锚，而非硬裁剪

定义：

delta_theta_limit
= 分组 expert correction 的 P95 或 P99


按以下分组计算：

速度带
道路曲率
lateral-error bucket
heading-error bucket


损失：

L_amp = relu(
    abs(delta_model_theta) - delta_theta_limit
)^2


不建议直接设置一个全局手工阈值，因为：

-低速和高速可接受的 dθ 不同； -直道和急弯不同； -横向误差和航向误差组合不同。

运行时仍保留安全限幅

建议 semantic correction 的部署上限先设为 expert correction P99，而不是 action 全范围。

26D：完整镜像 paired supervision

镜像必须同时变换：

lateral_offset → -lateral_offset
heading_error → -heading_error
curvature → -curvature
lane_change_left ↔ lane_change_right
left_margin ↔ right_margin
LD dy / heading / curvature 同步变换
左右 lane role 同步交换
OD 横向坐标、横向速度同步镜像


监督：

delta_ds(mirror) ≈ delta_ds(original)
delta_dtheta(mirror) ≈ -delta_dtheta(original)


必须包含联合网格：

lateral:
-1.0, -0.5, 0, +0.5, +1.0m

heading:
-10°, -5°, 0°, +5°, +10°


重点报告四象限：

lat+, heading+
lat+, heading-
lat-, heading+
lat-, heading-


这比单独 lateral paired 更能识别位置误差与航向误差的交互。

五、Round-26 只允许两个正式候选

不要再做三四个结构臂。

Q0：修复数据后的 P1 重跑

目的：

确认旧正向结果在正确标签下是否仍成立。

配置保持 Round-25 P1 不变，只替换：

B-label-v1 → B-label-v2


这是必要的基准，不可跳过。

Q1：Q0 + delta supervision + amplitude anchor + full mirror pair

这是唯一改进候选。

Primary 保持冻结，不运行类似 P2。

如果 Q0 已经明显改善，可以判断旧污染标签是主要问题；如果 Q1 再改善，则幅度和对称性修正有效。

六、训练和 checkpoint 选择
Stage 1

只训练：

SemanticWiring；
beta_context；
beta_token；
target projection； -语义动作辅助头。
Stage 2

只解冻：

PlanHead fusion 最后一层；
Policy 输出前最后一层。

Primary 保持冻结。

Epoch 选择

不能只看 loss，也不能只看 sign accuracy。

建议硬门筛选：

sign >=95%
mirror >=95%
curvature >=25%
history/lane <=1
teacher MAE degradation <=10%
semantic correction P99 不超过 expert P99 的1.25倍


满足硬门后，再按 calibration50 闭环 success 选择 epoch。

clean150 只用于最后开发验证。

七、Round-26 验收标准
Gate A：数据正确性
旧 B-label-v1 暴露数 = 0
同状态匹配率 =100%
随机重放 >=2,000 行
perturb_lat 与 expert_dtheta 的方向关系合理
左右/航向四象限覆盖完整

Gate B：语义采用

连续两个 epoch：

sign >=95%
mirror >=95%
lane response >=3× legacy
curvature response >=25%
history/lane <=1

Gate C：幅度与基础能力
perturbed-state dtheta MAE 至少下降20%
learner-state dtheta MAE 至少下降15%
teacher-state MAE 恶化 <=10%
semantic delta P95/P99 不超过 expert对应分位数的1.25倍
steer saturation 不高于 legacy +5pp

Gate D：clean150

最低要求：

success >=0.42
且相对 legacy >=+7pp

off-road <=0.35
collision <= legacy +2pp
timeout <= legacy +3pp


这里将 success 门从 0.38 提高到 0.42，因为 Round-25 已经达到 0.40，修复后必须继续进步，而不是仅复现。

Gate E：进入正式采集许可

只有 Q1 的 A–D 全过，才允许：

-扩充正确标注的扰动数据； -正式 20 万级 expert 数据； -下一阶段 learner-state DAgger 设计。

八、暂不批准的事项

继续冻结：

Primary 全量解冻
MoE experts
DAgger 实施
PPO / GRPO
WM 修改
K-anchor
新容量扫描
新的 semantic 网络层
geometry loss


其中 DAgger 虽然最终很可能需要，但当前模型还无法做到：

在不破坏正常驾驶的前提下，以正确幅度使用 lane semantics。

现在加入更多 learner-state，只会扩大错误修正。

九、对终极目标的判断

Round-25 是关键突破，但距离超过 IDM+Pure Pursuit 的 0.756 仍有很大差距：

legacy约0.33
Round-25 P1约0.40
IDM+Pure Pursuit约0.756


接下来合理的里程碑是：

Round-26：
语义修正稳定化，clean150 达0.42–0.48

正式 semantic BC：
eval500 达0.50–0.60

current-action DAgger：
推进到0.60–0.70

之后才考虑：
场景型 MoE / 合法信用链 PPO


如果 Round-26 在正确标签和幅度控制后仍无法超过 0.42，应停止继续调 semantic loss，重新审视：

chain 与动作的联合监督； -专家标签和 LQR 执行兼容性； -训练数据中普通驾驶与偏离状态比例； -混合 IDM+Pure Pursuit + learned residual 的工程路线。
最终批准
批准
Round-26：
Q0 正确标签重跑
Q1 delta supervision + amplitude anchor + full mirrored state pairs
Primary冻结
分阶段小范围解冻
严格checkpoint选择

关键反驳

我不同意简单按 Coding Agent 的说法：

“P1 正向，但量级过强，因此加一个动作上限即可。”

根因不只是幅度过强，而是：

旧扰动标签存在状态错配
+ 监督目标重写完整动作而非修正量
+ lateral-only paired 未覆盖 lateral×heading 交互
+ 正常 expert 状态缺少零修正约束


Round-26 必须四项一起纠正，否则硬限幅只能掩盖监督语义错误。

给 Coding Agent 的推进指令
批准 Round-26，不改网络容量与主架构。

先做 Q0：
仅用 same-state B-label-v2 重跑 Round-25 P1，
确认原正向在正确标签下是否成立。

再做 Q1：
学习 expert_action - legacy_action 的 semantic delta，
加入按速度/曲率/lat/heading 分组的幅度软锚，
并使用完整 lat+heading 镜像状态 paired supervision。

Primary保持冻结；
只解冻 semantic path + PlanHead/Policy末层低LR。

必须满足：
- teacher MAE退化<=10%
- mirror>=95%
- semantic delta分位不超过expert 1.25倍
- steer saturation<=legacy+5pp
- clean150 success>=0.42
- off-road<=0.35
- timeout<=legacy+3pp

未过门：
停止增加semantic loss、数据规模或模型复杂度；
不转DAgger/MoE/RL。


这一步的目标非常明确：

把已经证明“方向正确”的车道语义修正，校准成强度正确、左右对称、不会破坏正常驾驶的动作增量。