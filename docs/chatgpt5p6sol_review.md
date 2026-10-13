总结判断

最新进展不是普通的小幅正向，而是第一次出现较完整的因果闭环：

原始语义可计算
→ 网络原先完全没消费
→ 显式接线后 latent 可解码
→ 动作方向响应从 73% 提升到约 99%
→ history shortcut 相对影响显著下降


因此，docs\r21_3a_semantic_wiring.md、env\obs\lane_semantic.py、net\semantic.py 这条路线是目前最有实证支持的方向。

但我不批准立即采集 20–30 万 expert + 3–5 万扰动并直接进行完整 S0/S1 训练。现有结果证明了 semantic wiring 有效，却还没有证明当前代码语义在 rollout 中完全一致，也没有证明它能转化为闭环增益。

我的裁决是：

批准进入 R21-3D“代码语义闭环 + 小规模闭环 canary”，通过后再进行正式采集。

一、恢复的上下文

这次提供的 docs\chatgpt5p6sol_review.md 和 docs\deepseekv4p1_argue.md 已经补齐此前缺失的讨论。当前完整链路是：

Round-18A：
发现 LD 拓扑和几何响应问题

Round-20 / Round-22：
M1/M2/M2a/M2b 未过闭环门
确认不是简单 pooling/attention 可以解决

R21-0：
增加 target-lane 与 lane-semantic 契约

R21-1：
完成 semantic_v1 pilot

R21-2：
发现 raw semantic 信息正确，但原模型完全不消费

R21-3A：
将 semantic 显式接入模型

R21-3B：
短 canary 训练

R21-3C：
表示 3/3 通过，动作反事实 4/5 通过


因此，现在不是重新讨论“是否需要lane semantics”，而是审核其实现是否足以进入正式训练。

二、值得肯定的实现
1. lane_semantic 契约整体合理

env\obs\lane_semantic.py 新增的信息抓住了此前缺失的核心：

自车相对目标车道的 lateral offset；
自车相对车道切向的 heading error；
近场和远场 curvature；
左右同向道路表面余量；
target lane validity；
route association confidence；
lane-change direction；
target lane end distance。

这比单纯增加更多 LD 点更合理。它直接表达了规划真正需要的：

我在哪
目标车道在哪
目标车道朝哪里
应该往哪个方向修正
前方道路如何弯曲


禁止把 raw map lane ID 直接输入网络，也是正确的。否则容易记忆地图内部编号，泛化性差。

2. target lane 从“ego lane 优先”改为“route 优先”

这是关键修复。

旧逻辑中，车辆偏离后 ego.lane 可能切到误入车道，网络随之把错误车道当成目标。现在：

route current
→ route continuation
→ route next
→ ego lane fallback


更符合规划语义。

3. 语义终于被模型真正消费

此前最大的代码问题是：

obs 中有 lane information
≠
PlanHead 真正使用 lane information


现在 net\semantic.py 创建：

target_lane_token；
role-aware LD augmentation；
target/left/right/route/other 分组池化； -统一的 geometry_context。

并在 net\model.py 中统一接入：

t0 PlanHead；
rollout step 1–6；
policy/value tokens。

这修复了过去“数据存在但网络不消费”的根本性问题。

4. fail-fast 与默认关闭做得很好
旧 schema 缺键直接失败；
semantic 模式默认关闭；
baseline 逐位不变； -新参数支持严格白名单；
mask 无效时 token 严格为零； -全部测试 925 个通过。

这些工程契约可以保留。

5. 3C 结果是强机制证据

关键变化：

lateral/heading/curvature 解码接近满分； -校正方向从约 73% 提升到约 99%；
lane 几何响应是旧模型的 4.31 倍；
history/lane 影响比从明显 history 主导下降到 0.243； -镜像一致率超过 96%。

这证明此前的核心问题之一确实是：

车道语义没有进入规划决策主干，而不是单纯数据量不足。

三、我发现的三个代码级风险
风险 1：semantic 模式仍保留错误的 LD-LD 槽位链

net\model.py 中：

spatial.include_ld_chain = mode in ("baseline", "attn", "semantic")


也就是说，semantic 模式仍保留 LD slot-chain adjacency。

但前面的审计已经证明：

相邻槽位同 lane 的比例只有约 23.5%； -约 76.5% 的 LD-LD 边连接的不是同一条连续车道。

即使 net\semantic.py 后续做了 role/group pooling，输入给 semantic wiring 的 z_ld 已经经过 ST-GNN，可能提前混入跨 lane 伪消息。

建议

正式训练前必须做一个极小对照：

semantic_chain
semantic_no_chain


这不是重新开启结构扫参，而是在消除已知错误拓扑。

我倾向于正式 semantic 模式默认：

include_ld_chain = False


除非 semantic_chain 在同协议下明确优于 semantic_no_chain。

风险 2：rollout 使用动态 latent，但 lane semantic 仍是 t0 静态值

net\model.py rollout 中，每一步重新构造：

step_context = build_geometry_context(
    z_ld_k,
    z_ego_k,
    semantic=t0_semantic
)


这里：

z_ego 和 z_ld 在 rollout 中变化； -但 lateral offset、heading error、margin、distance-to-lane-end 等 lane_semantic 仍来自 t0。

这存在语义不一致。

例如：

t0 lateral offset = +0.8m
模型第一步已经向中心修正
t1 仍注入 +0.8m 的静态 offset


网络可能在尾段反复过度纠偏。

这不是小问题

当前 chain 已被证明承担执行噪声缓冲。如果 rollout 中不断复用陈旧的 lateral/heading error，可能破坏刚恢复的 chain 稳定性。

正式训练前必须比较三种口径
C0：semantic 仅用于 t0，rollout 后续不重复注入动态 semantic
C1：当前实现，t0 semantic 在后续重复使用
C2：将 semantic 拆成 static / dynamic


其中：

静态部分可复用
target-lane role
route confidence
lane-change requirement
道路拓扑身份

动态部分不应原样复用
lateral offset
heading error
distance to lane end
左右 margin


第一版无需实现完美的 C2。若不能从 latent/pose 可靠更新动态语义，我更倾向：

t0 使用完整 semantic，rollout 后续只保留静态 lane-role context。

风险 3：target lane 稳定性已证明，但正确性还没有完全证明

现有报告证明：

-横向扰动时 target lane 不切换； -target lane valid 达到 100%； -几何计算准确。

但：

稳定选择同一条 lane


不等于：

选择的是正确 route lane


尤其在：

多车道并行； -收费站；
merge/split； -roundabout；
current_ref_lanes 包含多条 lane；

route current 的候选可能不止一个。

正式采集前增加 target-lane correctness 审计

至少对 500–1,000 个 route transition 状态，验证：

target lane 是否与 route waypoint/checkpoint 连通
target lane tangent 是否指向下一导航点
专家未来 1–2 秒是否主要沿该 lane 前进
同一 road 多 lane 时是否选到正确 lane


建议门槛：

普通道路正确率 >=99%
复杂拓扑 >=95%
严重错误率 <1%


不能只看 valid=100%。

四、对 3C 结果的保守解释

Coding Agent 倾向：

3C 基本过门，唯一未过的曲率 20.1% 是冻结栈和训练预算造成。

这个解释合理，但还没有被证明。

曲率响应不足也可能来自：

target_lane_curvature 被辅助任务解码出来，但 policy 没真正使用；
action label 在弯道样本不足；
frozen policy/value 限制； 4.从 curvature 到 (ds,dθ) 的尺度映射不正确；
rollout 静态 semantic 抵消了曲率响应；
chain 为抑制噪声主动压低响应。

所以我不同意直接将 20.1% 判定为“冻结栈伪影”。

更准确的表述是：

lane direction grounding 已得到强证据，曲率幅值建模仍未完全闭合。

五、对 semantic_v1 pilot 的审阅

docs\semantic_v1_dataset_card.md 的 pilot 适合作 probe，但不适合直接支撑正式训练，文档对此已经诚实说明。

主要局限：

B 段交通为零；
B 段 history 是 reset 后 warmup；
learner-state C 段只来自单一 checkpoint； -高速样本无覆盖； -困难路型扰动覆盖有限； -pilot 只有约 9.5k 行。

尤其需要警惕：

3C 的强正向可能部分来自“显式 semantic + 干净受控扰动”的容易分布。

所以正式采集不能简单放大 pilot 配方。

六、正式数据不建议按“30万 expert + 5万扰动”直接采

若原始 expert 数据远多于扰动和 learner-state，模型可能再次选择最便宜的 shortcut：

绝大多数样本都位于车道中心
→ lane_semantic 接近零
→ 使用历史动作即可降低总体 loss


建议正式训练 exposure，而不是仅看原始行数：

正常 expert：50%–60%
受控 lane perturbation：20%–25%
learner-state：15%–20%
复杂拓扑定向样本：5%–10%


原始数据可以采更多，但 batch sampler 必须维持上述训练暴露比例。

同时，扰动样本必须包含：

连续真实 history
有交通扰动子集
不同速度带
普通道路和复杂路型
左右对称扰动


不应全部是 traffic=0 + reset warmup。

七、建议锁定 R21-3D
3D-1：代码语义一致性

完成三项：

semantic_chain vs semantic_no_chain；
rollout static/dynamic semantic 对照；
target-lane correctness 审计。
通过条件
不再依赖已知错误的 slot-chain；
rollout 后段不会因重复 t0 lateral/heading error 而过度纠偏； -target lane 普通道路正确率 ≥99%，复杂拓扑 ≥95%。
3D-2：独立数据上的表示验证

不能继续只在 pilot train-like split 上验证。

使用：

全新 episode
连续 history
部分真实 traffic
未参与 canary 的 map/spec


重复 3C probes。

通过条件
sign accuracy >=95%
mirror consistency >=95%
lane response >=原模型3倍
history/lane ratio <=1
curvature response >=20%


这里我接受 curvature 20% 作为进入小闭环 canary 的最低门，但正式训练后仍应达到 25%。

3D-3：小规模闭环 canary

先采：

30k–50k normal expert
10k–15k controlled perturbation
5k–10k learner-state


只训练一组 S0/S1 小 canary，不立即采 30 万。

建议两臂
S0：
semantic wiring
Primary-only
无 lane auxiliary

S1：
semantic wiring
Primary-only
lane direction / lateral correction / curvature auxiliary


保持：

learned chain； -MoE experts 关闭； -WM 不改； -同一数据、epoch、seed； -相同参数预算。
clean150 开发门

相对 legacy baseline：

success 至少 +5pp
off-road 至少 -5pp
collision 不增超过2pp
timeout 不增超过3pp
steer saturation 至少下降20%
lane counterfactual 继续通过


同时要求：

success >=0.40


这里仍低于之前计划的 0.50，但作为第一轮 semantic closed-loop canary 更现实。

3D-4：只有小 canary 通过才正式采集

正式采集建议：

expert-state：150k–250k
controlled perturbation：50k–80k
learner-state：30k–50k
targeted complex topology：20k–30k


并通过 sampler 控制 exposure，不能让 expert centerline 样本完全淹没偏离状态。

八、S0/S1 训练时必须监控

每个 epoch 输出：

lane semantic auxiliary loss
target token norm
geometry context norm
Primary action sensitivity
curvature response ratio
history/lane sensitivity ratio
teacher-state action MAE
perturbed-state action MAE
learner-state action MAE
chain curvature / smoothness
LQR steer saturation


还要分组闭环：

普通直道
普通弯道
tollgate
merge
roundabout
lane-transition


若总体提升只来自普通道路、复杂路型反而恶化，不得推进正式训练。

九、Primary 与 MoE 的职责继续锁定

当前阶段：

Primary：
lanekeep
route following
lane-relative correction
正常曲率跟随
平滑可执行 chain

MoE：
全部关闭


直到 Primary 达到：

clean150 >=0.50
eval500 >=0.60


才讨论 experts。

否则 experts 很可能再次承担基础 lane correction，违背你的架构初衷。

十、对最新进展的正式审批
批准
env\obs\lane_semantic.py 当前方向；
net\semantic.py 的 target token 和分组池化；
schema v7 和 fail-fast；
R21-3A/3B/3C 作为机制验证成功； -进入 R21-3D； -小规模 semantic closed-loop canary。
需要修正后再批准正式训练
semantic 模式继续保留伪 LD-LD chain；
rollout 重复使用 t0 动态 semantic；
target-lane correctness 还缺复杂拓扑真值审计； 4.正式数据 exposure 尚未防止动力学 shortcut 复发； 5.曲率响应只达到 20.1%，不能直接当作已闭合。
暂不批准
直接采集 20–30 万 expert 后全量训练；
A4 DAgger； -MoE experts； -RL； -WM 调整； -K-anchor； -新的 capacity sweep。
给 Coding Agent 的下一条指令
R21-3A/3B/3C 机制方向批准，但不直接进入大规模正式采集。

先执行 R21-3D：

1. 审计 semantic 模式的 LD-LD slot-chain。
   做 semantic_chain / semantic_no_chain 单变量对照；
   已知伪边不得未经验证进入正式配置。

2. 审计 rollout semantic：
   t0 的 lateral/heading/margin 不得在 6 步中无条件作为动态真值重复使用。
   对比 full-static / t0-only / static-role-only 三种口径。

3. 在 500–1000 个复杂 topology 状态上验证 target-lane correctness，
   不只验证 valid 和 stability。

4. 使用全新 episode、连续 history、含 traffic 的独立集复跑 3C probes。

5. 采 45k–75k 行小训练集，
   做 S0/S1 Primary-only semantic closed-loop canary。

小 canary 达到：
- success >= legacy +5pp 且 >=0.40
- off-road >=下降5pp
- collision <=+2pp
- timeout <=+3pp
- steer saturation下降>=20%
- 反事实 probe 继续通过

才允许采正式 20万级数据。

未达门：
先定位 chain / rollout semantic / target selection，
不得直接转 DAgger、MoE 或 RL。

最终判断

这是目前最有希望的一条路线，但仍处于：

“证明网络终于看见并使用了车道语义”


还没有进入：

“证明闭环性能因此提升”


下一步最重要的不是继续扩大数据，而是确保语义在 图消息传递、rollout 和复杂拓扑 target selection 三处没有自相矛盾，再用一个小规模闭环 canary 完成从机制正确到性能改善的最后证据链。