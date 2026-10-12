结论

Coding Agent 的反驳成立：此前 M2 不是一个干净的 set-attention 单变量实验，不能据此宣布“模型侧 LD 建模已穷举”。

我批准在重采数据前做最后一次、严格受限的模型侧决定性实验：

M0：原始基线
M1：仅关闭错误的 LD-LD 槽位链
M2a：M0 + 条件化 LD attention pooling
M2b：M2a + 关闭 LD-LD 槽位链


但这是最后一次模型侧结构排摸。若失败，立即转入已预备案的车道语义重建，不再继续发明 M2c/M2d、换 attention、加层数或扫超参数。

一、为什么批准 M2a/M2b

当前已知：

LD 槽位链伪边率约 76.5%；
PlanHead 使用全槽 masked mean；
模型对 lane translate/rotate 的响应比合理量级弱约 50 倍； -历史信息对输出的影响远强于车道几何；
M1 有温和正向迹象； -此前 M2 同时改变多处消费路径且接线不干净，不能单独归因 set-attention。

因此还有一个未被正确回答的问题：

现有 LD 原始特征是否已经包含足够几何，只是 PlanHead 的无条件均值无法选择相关 lane？

M2a/M2b 正好回答这个问题，成本低于修改 observation schema 和全量重采。

二、必须收紧 M2a/M2b 的实现范围
1. aggregate_ld() 只替换“池化摘要”，不得替换 per-slot tokens

建议契约：

ld_summary = aggregate_ld(
    ld_tokens=z_ld,
    ld_mask=ld_live,
    ego_token=z_ego,
    nav_token=nav_token,
)


替换原来的：

ld_pool = masked_mean(z_ld, ld_live)


但以下内容保持不变：

Policy/Value 的 16 个 LD per-slot tokens；
WM 的 LD slot state；
OD 路径；
action/plan 接口； -损失函数； -训练数据； -参数预算主体。

Coding Agent 所说“统一供 t0/rollout/policy 消费”需要谨慎。正确含义应该是：

所有需要 LD pooled summary 的位置统一调用 aggregate_ld()；不得把原本的 per-slot token 全部替换成一个 summary。

否则又成为多变量架构实验。

2. M2a 不能声称已经解决 target-lane selection

当前 LD 没有明确的：

target_lane flag
lane role
route association
same-lane group
offset identity


所以 ego/nav-conditioned attention 只能根据几何和导航 token推测哪条 lane 相关。

M2a 检验的是：

条件化读出是否比 masked mean 更有效。

它不能证明：

网络已经稳定识别 route target lane。

即使 M2a 阳性，后续显式 lane semantics 仍可能有价值。

三、M2a 的强制实现契约
1. 残差式初始化

建议：

ld_summary = ld_mean + alpha * ld_attention
alpha 初始化为 0


必须满足：

alpha=0 时，M2a 与 M0 前向逐位一致


训练后必须报告：

alpha
attention_output_norm
mean_output_norm
attention / mean norm ratio
attention entropy


避免两类假结论：

alpha 没打开，误判 attention 无效；
attention 爆炸，提升来自尺度变化而非 lane selection。
2. Query 必须受 ego 与 nav 条件化

不能只用一个全局 learned query。建议 query 至少依赖：

ego token
nav token
可选 others context


否则仍然只是另一种无条件 pooling。

第一版不引入多层 Transformer，单层 attention 足够。

3. Mask 与全空输入

必须验证：

无效 LD 权重严格为 0
全无效 LD 时输出有限且退化为 0 summary
无 NaN / Inf

4. 梯度与优化器

运行时记录：

attention parameter grad norm
alpha grad / update
query projection update
key/value projection update


防止“实现存在但未真正训练”。

四、四臂实验必须严格同口径
M0：baseline
M1：baseline，关闭 LD-LD 槽位链
M2a：baseline，条件化 attention pooling
M2b：attention pooling，关闭 LD-LD 槽位链


必须保持一致：

同一初始 checkpoint； -同一训练数据； -同一 epoch； -同一 seed； -同一 optimizer 和 LR； -同一 checkpoint selection； -同一 CUDA 评测后端； -除目标改动外 resolved config 无差异。

不增加：

hidden size； -attention 层数搜索； -新的 auxiliary loss； -lane semantics； -MoE 改动； -WM 改动； -DAgger； -recovery 数据； -geometry loss。
五、先过机制 Gate，再看闭环
Gate A：实现正确性

全部必须通过：

alpha=0 与 M0 逐位一致；
所有 pooled LD summary 经统一入口；
per-slot LD tokens 保持原路径；
无隐藏的旧 masked mean 替代 attention summary；
全 mask forward/backward 有限；
attention 参数获得非零梯度并实际更新；
M1 与 M2b 只修改 LD-LD adjacency；
checkpoint 严格完整加载。

任何一项失败，不允许解释性能。

Gate B：机制有效性

训练后至少满足：

alpha 明显离开 0
attention output 非退化
attention 不是近似均匀分布
target/current/continuation 附近槽位获得更高权重


同时重跑 lane counterfactual：

translate ±1m
rotate ±10°
straighten
zero_ld
zero_history


建议通过标准：

lane translate/rotate 的 Δmu_dθ 至少达到 M0 的 5 倍； -响应符号正确； -左右扰动近似对称；
attention 权重随 lane 扰动发生合理变化；
zero_history / lane-perturbation 敏感度比相对 M0 至少下降 50%； -不能通过简单放大所有输出获得上述结果。

如果 attention 学会了非均匀权重，但几何响应仍然极弱，说明它选择的是其他相关性，不是 lane grounding。

六、闭环验收标准需要稍作调整

Coding Agent 建议 clean150 相对 M0：

success +5pp
off-road -7～10pp


方向合理，但 clean150 单 seed 每个 episode 约占 0.67pp，+5pp 约为 7–8 个 episode，适合作为开发门。

正式开发门

M2a 或 M2b 至少满足：

success >= M0 +5pp
off-road <= M0 -5pp
collision 不增加超过2pp
timeout 不增加超过3pp
steer saturation 相对下降至少20%
lane perturbation sensitivity 至少提升5倍


同时必须是非孤立改进：

M2a/M2b 至少有一个明显胜出； -相邻 epoch 不应立即回落； -改善不能只集中于单一道路类型； -不能只是 off-road 转 timeout。
进入 eval500 的条件

只有 clean150 开发门通过才跑 eval500。

建议 eval500 Gate：

success >=0.45
相对同批 M0 >=+5pp
off-road 相对下降至少10%
collision <=0.05
timeout不显著恶化


0.45 仍离最终目标很远，只代表模型侧机制值得保留，不代表问题解决。

七、四种结果对应的唯一行动
情况 1：M1 胜出，M2a/M2b 无增益

说明：

-错误 adjacency 确实有害； -现有 LD 缺少足够语义，attention 不能可靠选择 target lane。

行动：

保留 M1 修复
立即进入 lane semantic 重建
不再试其他 attention

情况 2：M2a 明显胜出，M2b 不再提升

说明：

-主要瓶颈是 mean pooling； -错误 adjacency 影响有限，或被条件化 pooling 绕过。

行动：

保留 M2a
仍进入轻量 lane semantics pilot
但可复用 attention pooling


不能因此取消显式 target-lane 契约，因为 attention 仍在猜 lane role。

情况 3：M2b 最优

说明：

-错误图拓扑和无条件 pooling 都是实质问题。

行动：

保留 M2b
进入 lane semantics pilot
不做 M2c


因为下一瓶颈仍是 lane identity、route association 和 ego-lane relation。

情况 4：四臂均无实质提升

说明：

现有 LD 特征即使改善消费机制，也不足以让 Primary 建立 lane grounding。

行动：

模型侧结构排摸正式结束
启动 R21-0 ～ R21-2 语义重建


禁止：

-再换 attention； -加更多 heads； -加 Transformer 层； -恢复 slot adjacency； -扫 pooling 超参数； -继续增加 hidden size。

八、对数据侧 R21 方案的确认

Coding Agent 要求将 R21 语义重建作为 M2a/M2b fail 分支，我有一个小修正：

它不应只是“失败分支”，而应是已经预注册的下一阶段。M2a/M2b 阳性时，只是复用其 pooling；阴性时，则更明确地直接实施。

因为当前输入缺失以下语义是代码事实，不会因 attention 改善而消失：

target-lane role
route association
same-lane grouping
offset identity
ego-to-target-lane lateral error
heading error
boundary margins


所以模型侧正结果只能改变语义重建的优先级和复用机制，不能永久取消语义重建。

九、对 A4 DAgger 的状态

继续冻结。

只有满足下列条件才启动：

Primary 对 lane perturbation 响应方向正确
target-lane 表示稳定
clean150 >=0.50
teacher-state 当前动作误差仍低
learner-state 当前动作误差仍高


否则 DAgger 只会把更多 learner 状态喂给一个仍不理解 lane 的模型，让它继续学习动力学 shortcut。

十、最终审批
批准
Round-22：
M0 / M1 / M2a / M2b 四臂严格对照
机制 probes
clean150开发门
条件性eval500

条件
限一个实验 session； -不增加第五臂； -不扫 attention 超参数； -不加入新监督； -不更改数据； -不更改 MoE/WM； -失败后自动进入 R21 语义重建，不再开会重新争论模型结构。
暂不批准
M2c
全量数据重采
A4 DAgger
geometry loss
MoE experts
RL
K-anchor
容量扩展

给 Coding Agent 的最终指令
批准 Round-22。

先完成 M0/M1/M2a/M2b 四臂同批实验。
M2a 仅替换 pooled LD summary，保留 per-slot token 路径；
M2b 仅在 M2a 基础上关闭 LD-LD slot-chain adjacency。

必须先通过：
- alpha=0 逐位等价；
- masked/all-empty 安全；
- attention 梯度和实际更新；
- 无隐藏旧 mean 路径；
- lane counterfactual sensitivity；
再解释闭环结果。

clean150 未达到预注册门：
立即结束 LD 架构排摸，启动 lane-semantic R21-0～R21-2。
不得新增 M2c/M2d、attention 层数或 pooling 超参实验。

clean150 达门：
只跑 eval500 验证；
无论最终阳性或阴性，都保留 lane-semantic 契约作为下一阶段，
阳性时复用 M2 pooling，阴性时直接重建输入语义。


这版可以批准执行。它既修正了此前 M2 无法归因的问题，也设置了明确终点，避免模型结构和数据语义之间继续来回拉扯。