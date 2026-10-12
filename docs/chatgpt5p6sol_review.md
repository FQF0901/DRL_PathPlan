总体结论

我认可 Coding Agent 对上一轮两处事实的纠正：

LD 并非只有“车道点”，每点已有 dx/dy/heading_rel/curvature/speed_limit/左右线型；
LD 历史被存入 memory，但 MemEncoder 最终只使用当前帧，不做 LD temporal aggregation。

Round-18A 可以开工。
 不过我亲自核查 DRL_PathPlan_Code.txt 后，认为问题比“网络是否使用 LD”更具体：

当前 LD 有局部几何，但缺少稳定的车道身份、目标车道关联和正确拓扑；网络又在关键位置对 LD 做无结构池化或错误邻接。因此它可能“看到车道点”，却无法稳健理解“自车应该沿哪条连续中心线行驶”。

一、代码排摸发现
1. LD 输入信息不算少，但缺失关键语义

每个 LD 点包含：

dx, dy, heading_rel, curvature,
speed_limit, left_line_type, right_line_type


当前车道优先占 5 个点，实际 offsets 是 5/10/15/20/30m。其他候选包括下一参考车道和同路相邻车道。

但每个槽位没有显式提供：

lane_id
是否当前车道
是否 route target lane
是否 continuation
相对 lane index
offset/s 坐标
该点与哪个其他点属于同一条 lane


因此，网络只能从点的坐标猜测这些关系。复杂分叉、收费站、相邻车道并行时，这种猜测很容易失败。

一个文档问题

env/obs/ld.py 顶部注释仍多处写“0/20/40/60/80m”和“0m 投影点”，但实际常量是：

LD_OFFSETS_M = (5, 10, 15, 20, 30)


这是注释漂移，虽然不直接影响运行，但会误导 probe 和监督设计。

2. 最危险的问题：LD 图邻接与槽位排序不一致

LD 槽位排序是：

当前车道五个点
+
其他车道按 offset 环优先排列


也就是说，secondary 部分很可能是：

lane A @5m
lane B @5m
lane C @5m
lane A @10m
lane B @10m
...


但 SpatialEncoder 对 LD 建图时，直接连接：

slot[i] ↔ slot[i+1]


这意味着很多“相邻边”并不是同一条车道上的连续点，而是：

不同车道同一 offset； -当前车道末端点与另一车道近端点；
continuation 与 sibling lane 之间的伪连接。

这不是纯粹表示能力不足，而是图拓扑可能明确建错了。

在收费站、merge、split 等多车道结构中，这尤其危险。ST-GNN 可能把并行车道误当成一条连续路径传播消息。

这是 Round-18A 必须新增的最高优先级静态审计。

3. PlanHead 把 LD 压成 masked mean

PlanHead 的融合输入为：

ego_ctx, od_pool, ld_pool, others_ctx, nav_token, signal_token


其中：

ld_pool = masked_mean(z_ld, ld_live)


masked mean 会抹掉：

哪个点属于当前车道； -前后顺序； -哪条 lane 是 route continuation； -左右车道区别；
offset 位置。

虽然后面的 policy/value cross-attention 仍能看到 16 个 LD token，但主 PlanHead latent 和 MoE router 接收到的是高度压缩后的 LD 均值。

这会直接影响你的架构初衷：

primary 无法明确获得 target-lane path；
router 更难识别“普通 lanekeep vs 特殊拓扑”；
experts 可能被迫补偿 primary 丢失的道路结构。

所以当前职责倒置未必只是训练问题，也可能来自 PlanHead 的信息瓶颈。

4. ego.lane 优先可能在偏离状态造成语义跳变

LD 的第一候选是：

ego.lane


其次才是导航参考 lane。

当 learner 已经横向偏离、接近另一条 lane 或错误通道时，MetaDrive 的 ego.lane 可能切换。此时 LD 的“primary lane”会突然变成当前误入的 lane，而不是应该返回的 route lane。

这会产生危险反馈：

车辆偏离
→ ego.lane 切换
→ LD 当前车道语义切换
→ 网络认为新车道才是应跟随对象
→ 不再回正


这可能同时解释：

learner-state 的 dθ 误差骤增；
gate 无法识别 recovery；
tollgate/复杂几何出界集中； -模型在 expert-state 表现很好，因为 expert 几乎不会触发 lane identity 跳变。

Round-18A 应专门统计：

ego.lane != navigation target lane


出现的频率，以及它与 off-road、dθ 符号错误的相关性。

5. LD temporal 不使用本身未必错误，但其前提不成立

代码明确只取：

ld_ctx = ld_h[:, -1]


理由是道路静态，历史对齐后信息近似重复。

这个选择只有在以下条件成立时合理：

同一 slot 跨帧始终代表同一条 lane、同一语义位置


但当前 slot 由 ego.lane、导航 lane、siblings 和有效 offset 动态生成，车道变化、短 lane、分叉时很可能发生 slot churn。

因此问题不一定是“应增加 temporal attention”，而是应先测：

相邻帧同 slot 的 lane identity 保持率
相邻帧排序置换率
短 lane 造成的 mask 跳变率
ego.lane 切换时 LD 全局突变量


若槽位不稳定，直接做 temporal attention 反而会融合不同 lane 的语义。

6. K-anchor 的 lane context 实际是零

代码中明确写明，v8 删除旧 lane 通道后：

lane_ctx = zeros(B, 3)


所以即使未来重新开启 K-anchor，它也没有真实 lane context。

目前 anchor 默认关闭，因此不是当前主因。但它说明 repo 曾经依赖的“lane-frame plan”语义已经被削弱，不能轻易重启 K-anchor。

二、对 Coding Agent Round-18A 的裁决
批准，但增加三个代码级探针
Probe 0A：LD slot/topology 审计

为每个 LD slot 临时记录：

lane_index
lane_priority
lane_order
offset_idx
is_ego_lane
is_nav_current
is_nav_next


统计：

相邻 slot 属于同一 lane 的比例
SpatialEncoder LD-LD 边中错误跨 lane 比例
跨帧 slot identity 保持率
ego.lane / route lane 不一致率


如果 LD-LD 边的大部分不是同 lane 连续点，应先关闭 LD-LD slot adjacency，或按 lane_id + offset 重建图，而不是继续训练。

Probe 0B：target-lane coverage

统计当前 route target lane 在 LD 中：

是否出现
槽位数
最近点距离
5/10/15/20/30m 覆盖率
heading/curvature 有效率


按 tollgate、merge、roundabout、普通道路分层。

若 target lane 经常不在 LD 中，网络“不沿正确车道”首先是输入缺失，不是学习失败。

Probe 0C：lane assignment discontinuity

围绕 learner 轨迹记录：

ego.lane identity
navigation.current_ref_lanes
navigation.next_ref_lanes
LD slot fingerprint
action_mu


检查 lane identity 切换前后：

LD token 跳变量
mu_dtheta 跳变量
plan curvature 跳变量


这可能直接找到 off-road 前的首个语义断点。

三、原四个反事实 probe 的修正
1. 不要只在 tensor 上粗暴镜像 LD

镜像必须同时处理：

dy → -dy
heading_rel → -heading_rel
curvature → -curvature
left_line_type ↔ right_line_type
左右 lane 顺序交换
nav checkpoint 同步镜像
OD 若参与控制则同步镜像
历史帧同步镜像


否则得到的是不物理的 observation，不能据此判断模型是否理解车道。

核心测试最好在环境层构造镜像状态，tensor-level 只作单元测试。

2. “去掉 LD 后输出显著变化”不是充分条件

模型可能依靠 nav、others 或 route_world 获得冗余道路信息。因此：

无变化可能表示 LD 未使用； -也可能表示道路信息被其他通道替代； -有变化也不代表响应方向正确。

真正重要的是：

对 lane lateral/heading perturbation 的响应方向正确
且该响应主要由 route-consistent lane 引起


所以验收应优先看符号、单调性和 target-lane selectivity，而不是单纯看输出变化幅度。

3. Probe B 应增加分层定位

依次从以下位置解码：

原始 LD
LD embedding
按槽位 LD tokens
masked ld_pool
PlanHead fused latent
primary output
最终 action_mu


这样可以明确损失发生在哪一步：

原始 LD 不可计算：输入缺失；
embedding 可解码、pool 不可解码：pooling 问题；
fused latent 可解码、primary 不响应：监督/head 问题；
primary 正确、最终 action 错：policy/chain/执行问题。
四、我对当前 LD 设计的最终判断
做得合理的部分
使用自车坐标系； -包含点位置、方向和曲率； -当前车道有多距离前视； -有 line type 和 speed limit； -有 route continuation 候选； -保留 per-slot tokens 给 policy/value； -作为结构化输入，比纯 raster 更适合当前小模型。
不合理或高风险的部分
缺少 lane identity 和 offset identity；
缺少明确 target-route association；
LD-LD adjacency 与实际槽位拓扑不一致；
PlanHead/MoE 只使用无结构 mean pool；
优先使用 ego.lane，偏离后可能目标语义跳变；
跨帧 slot consistency 没有被验证；
raw feature 尺度差异很大，却只经单一线性层编码；
没有可靠 lane-direction / centerline-alignment 监督；
旧 lane context 被删除后，K-anchor lane context 退化为零。

因此，Coding Agent 所说“几何并非不存在，缺的是显式关联、选择与监督”是正确的，但还不够强：

不仅缺显式关联，当前 SpatialEncoder 甚至可能把错误的 lane-slot 邻接当成真实拓扑。

五、关于 primary 与 MoE

你的原则应正式锁定：

Primary 必须负责
route-consistent lane keeping
lateral/heading error correction
基础曲率跟随
平滑、可持、可跟踪 plan
普通前车跟随

Specific experts 才负责
tollgate 通道选择
merge/yield
复杂交叉口
临时避障
极端恢复
特殊拓扑


Round-18A 必须分别测试：

primary-only
primary + MoE


但“关闭 experts”不能只让 router 权重归零后改变 LayerNorm 或 residual scale，应增加严格的 primary_only 推理模式，并验证其他路径逐位一致。

如果 primary-only 在直道和缓弯都无法通过 counterfactual grid，禁止 A4 specific-only DAgger。否则 experts 会继续承担基础 lanekeep，进一步损害模块职责。

六、建议的实施顺序
立即批准
1. Probe 0A：slot/topology 审计
2. Probe 0B：target-lane coverage
3. Probe 0C：lane assignment discontinuity
4. 原 Round-18A lane-grounding/shortcut probes
5. pri512 与 A3 checkpoint 双模型、primary-only 与 full-MoE 双模式

根据结果分支
若输入覆盖和拓扑正确，且 primary 对 lane perturbation 响应正确

进入：

A4 learner-state current-action DAgger
保留 learned chain 作为执行缓冲

若 target lane 存在，但 pooling/head 丢失语义

最小改造：

current/target lane 显式池化
而不是所有 LD slots masked mean


例如分别输出：

target_lane_pool
adjacent_lane_pool
boundary_pool


不要一开始扩大网络。

若输入本身缺失或 lane assignment 跳变

最小新增：

target_lane flag
lane_id/group id
offset index
route association
ego-to-target-lane lateral error
heading error
local curvature


这会改变 obs fingerprint，必须重采数据。

若输入和 latent 都正确，但 action 不响应

优先增加 primary-only 辅助监督：

signed lateral-error correction
lane-tangent direction consistency
curvature consistency


暂不改 MoE。

七、最终审批

批准 Round-18A 开工，但应将三项静态代码审计 Probe 0A/0B/0C 加到最前面。

暂不批准：

A4 DAgger； -可持性正则训练； -增加 raw LD 点数； -解冻 specific experts； -重启 K-anchor； -加入新的 geometry loss。

最需要优先验证的 root cause 是：

当前 LD 是否稳定表达了“route target lane 的连续路径”，以及 SpatialEncoder 是否正在用错误槽位邻接破坏这份信息。

如果错误邻接率或 target-lane 缺失率高，那么此前的 covariate shift、gate 无选择性、learner-state dθ 抖动都可能只是下游症状。先修语义和图拓扑，再谈 DAgger，才不会继续用更多数据补偿错误表示。