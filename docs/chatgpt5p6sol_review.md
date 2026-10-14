审阅结论

我接受 round33c_report.md 的实验事实，但不完全接受“下一步直接进入静态障碍显式表示 + 走廊占用 + 路径条件化纵向规划”作为唯一主线。

更准确的结论是：

动作条件化 V2 已经把 World Model 与 Policy 的接口误差改善约 10% 到 12%，证明方向有效；但它没有提升总成功率，并把失败从碰撞、超时部分转移成了 merge 和收费站 off-road。当前瓶颈已经从“WM 不理解 action”推进到了“缺少路径级横纵解耦、静态障碍表达不足，以及 ego 信息向 OD/LD latent 串扰”。

因此：

B1 不晋级为新 baseline，正确； -不进入 60/40 Policy 分布适配，正确； -不启动 DAgger、MoE、RL，正确； -下一阶段应正式启动你刚锁定的两条架构主线：
横向曲率路径 + 路径条件化纵向速度/进度
显式 ego-motion 坐标补偿 + 结构化 OD/LD anchor

但要用小规模、可回滚、具备完整 hooks 的方式推进。

一、本轮到底取得了什么进展？
1. World Model 与 Policy 的接口确实改善了

相对基线，B1 得到：

Policy在WM状态与真实状态上的ds差异：下降10.1%
最终动作误差：下降11.9%
第5步ds上冲：下降12.3%
真实状态上的Policy误差：下降4.3%


这与上一版 AC1 的全面恶化形成鲜明对照，说明以下修改是真正有效的：

删除重复的speed-intent输入
取消可学习alpha
固定残差增益
加入delta_progress / delta_speed / delta_heading监督


所以 Action Conditioning V2 不是失败，而是：

成功地把一个不稳定的动作条件分支改造成了接口稳定器。

2. 但动作因果仍没有学扎实

B1 的结果：

Action Swap top-1：0.192，未达到0.35
top-3：0.914，通过
ds-progress单调率：0.60，未达到0.90


这说明模型大体能排除明显不合理的动作，所以 top-3 很高；但要在多个相近动作中找出真正正确的一个，能力仍然较弱。

更关键的是：

对ds的敏感度已经增强
但响应方向并不稳定


也就是：

WM 已经“听得到 action”，但还没有稳定理解 action 导致的物理后果。

3. 成功率没有提高，失败模式发生了转移

闭环结果：

success：0.56 → 0.56
collision：0.08 → 0.06
timeout：11 → 9
off-road：0.14 → 0.20
route completion：0.862 → 0.855


它减少了一部分碰撞和超时，但新增了：

merge off-road ×2
tollgate off-road ×1


而收费站仍然：

success = 0


这说明 B1 改善了纵向进度和部分安全响应，却没有产生更合理的路径级规划。

4. 收费站问题已证明不仅是纵向制动

收费站碰撞案例中，B1 已经做到：

碰撞前计划停车余量约 +8.84m
min-TTC 从1.43提升到2.28


却仍然碰撞。

这说明至少这一案例不是简单的：

ds太大
刹车太晚


而更可能是：

横向路径经过岗亭
车体footprint与静态障碍冲突
目标通道选择错误
路径几何不合理


所以 Coding Agent 提议做收费站/merge 的路径级归因是必要的。

5. latent 语义串扰仍然严重

尽管 action 只显式进入 ego transition，B1 中仍出现：

聚合OD/LD中的ego speed信息增加19.1%
OD pool中的ego progress信息增加46.9%


这说明：

ego action信息
→ z_ego
→ 图消息传递
→ z_od / z_ld


仍然发生明显泄漏。

因此，我们之前讨论的长期方案是必要的：

几何坐标变化由解析层负责
ego/OD/LD latent分别承担语义和残差


而不能继续让 ST-GNN 自由地把 ego motion 写进所有节点。

二、对 Coding Agent 下一步建议的修正

Coding Agent 推荐：

静态障碍显式表示
+ 走廊占用
+ path-conditioned纵向剖面


方向总体正确，但三项不能一次性全部实现，否则又会失去归因能力。

下一阶段应拆成两个连续工作包。

Round-34A：路径级事实审计和静态障碍契约

先不训练。

任务 1：收费站和 merge 路径级归因

每个失败 episode 保存：

六步plan的x/y/theta
六步ds/dtheta/kappa
插值后的30个reference points
车辆footprint沿轨迹扫掠区域
目标车道中心线和边界
岗亭、建筑、静态障碍物轮廓
最近碰撞距离
首次越界点
LQR preview命中哪一段


分类：

路径本身撞障碍
路径本身越界
路径可行但LQR跟踪出界
目标通道选择错误
静态障碍未进入模型
障碍进入模型但Policy未响应


这一步非常关键，因为：

收费站碰撞和 merge off-road 不能混成一个“静态障碍问题”。

任务 2：静态障碍输入契约审计

目前收费站岗亭可能只存在于 others 的弱走廊摘要里，而不在 OD/LD 主体表示中。

必须动态记录：

岗亭是否进入obs
进入哪个字段
字段值是否随距离变化
Encoder输出是否变化
Policy latent是否变化
最终ds/kappa是否变化


做遮挡和干预：

原始静态障碍输入
静态障碍字段置零
岗亭距离增减
左右位置镜像
障碍移入/移出规划走廊


如果输入字段变化明显，但 Policy 不响应，问题在网络消费。

如果输入里根本没有岗亭几何，才批准新增静态障碍表示。

任务 3：完善 probes/hooks

我同意你强调动态 probe 优先于我单独做静态代码审查。以后所有新模块强制具备以下 hooks。

输入 hook
raw obs
normalized obs
mask
slot identity
coordinate frame
timestamp
units
valid range

几何 hook
ego pose delta
OD/LD anchor before warp
anchor after warp
expected analytic transform
transform residual
slot permutation

网络 hook
每个模块输入/输出shape
mean/std/min/max
effective rank
mask ratio
gradient norm
update norm
activation saturation

Policy rollout hook
真实latent
WM latent
PolicyBackbone latent
action_mu
plan 6步
累计pose
LQR reference
actual executed pose

数据契约 hook

每一行保存：

obs_t
policy_action
expert_action
executed_action
obs_t+1
tracker mode
coordinate frame
checkpoint hash
schema version


这些 hooks 默认关闭，开启时不得改变随机数序列和模型输出。

Round-34A 验收

只有回答以下问题后才进入结构修改：

1. 收费站失败中，路径本身错误占比是多少？
2. merge off-road中，横向路径与LQR跟踪各占多少？
3. 静态障碍是否存在于输入？
4. 存在于输入后，Encoder/Policy是否真正消费？
5. footprint连续扫掠是否在碰撞前已判定不可行？


门建议：

至少90%的新增失败可归因到明确类别
所有关键输入字段的坐标系、单位、mask可验证
遮挡干预的输出响应可重复
工具开启/关闭bit-exact

三、Round-34B：正式启动新动作表示

你已经确认长期使用：

横向曲率路径
+
基于路径的纵向进度/速度


我认为不应再无限后置，Round-34A 审计完成后就应开始最小原型。

横向路径分支

第一版不要太复杂，输出 3 个曲率控制点：

kappa_near
kappa_mid
kappa_far


通过平滑插值形成：

κ(s)\kappa(s)

然后积分成路径：

θ(s)=θ0+∫κ(s)ds\theta(s)=\theta_0+\int\kappa(s)ds x(s)=x0+∫cos⁡θ(s)dsx(s)=x_0+\int\cos\theta(s)ds y(s)=y0+∫sin⁡θ(s)dsy(s)=y_0+\int\sin\theta(s)ds

这样显式控制：

最大曲率
曲率变化率
左右镜像
路径连续性

纵向分支

不是单独输出 ds，而是读取：

横向候选路径
路径曲率
路径上的静态障碍距离
路径上的动态对象距离和相对速度
可通行走廊长度
导航意图


输出：

v_ref[6]
或
s(t)[6]


即真正的：

path-conditioned longitudinal planning


收费站岗亭若位于当前横向路径上，纵向分支必须自动减速；如果岗亭不在候选路径上，则不应无条件减速。

这正是当前 ds/dtheta 无法清楚表达的因果关系。

保留现有外部接口

第一阶段外部仍转换为：

ds_t
dtheta_t = kappa_t × ds_t


这样：

-固定 LQR 不改； -环境接口不改； -历史评测仍可比较； -只重构 Policy 内部表示。

四、Round-35：结构化 World Model 与 ego-motion 解耦

解析坐标补偿长期方向正式保留。

零训练 warp 只有约 2% 改善，是因为冻结旧 WM 从未在新坐标语义上训练。它只能证明：

zero-shot warp不能直接替换旧坐标口径


不能否定重训后的结构化模型。

新 WM 状态建议：

Ego:
  physical state
  semantic latent

OD:
  explicit anchor
  intent/dynamics latent

LD:
  explicit polyline/geometry anchor
  semantic/topology latent


每一步：

action
→ ego运动学积分

LD anchor
→ SE(2)自车逆运动补偿

OD anchor
→ 对象运动先验
→ SE(2)自车逆运动补偿

warped anchors
+ latents
→ learned residual dynamics


SparseDrive 将动态对象和地图元素都拆成显式几何 anchor 与实例 feature，并利用时间实例记忆进行传播，这支持“几何与语义分离”的原则。

五、Stage A/B 联合适配也应写入实施计划

你上轮认可：

WM与Policy需要磨合
DAgger数据同时服务WM和Policy


建议 Round-35 后按以下流程执行。

World Model 更新步

冻结 Policy，使用：

真实状态
实际执行动作
真实下一状态


训练 WM。

Policy 更新步

冻结 WM，使用：

真实状态 → expert action
WM状态 → 真实未来状态下的Policy动作


训练 Policy。

DAgger 采集

同时保存：

Policy监督：
obs → expert action

WM监督：
obs + executed action → next obs


但两种损失分别训练，不能混标签。

六、是否让 B1 晋级？

不晋级。

原因不是它没有价值，而是：

接口改善
collision改善
timeout改善


仍不足以抵消：

success无提升
off-road显著增加
route completion下降
top-1和单调性失败
OD语义泄漏增加


round33c_report.md 将它定义为“稳定器而非推进器”是准确的。应保留 checkpoint 和代码作为诊断分支，不作为后续默认起点。

七、给 Coding Agent 的下一步指令
Round-33C结案：
B1不晋级，不进入60/40，不启动DAgger/MoE/RL。

启动Round-34A，只读路径与输入契约审计：

1. 收费站/merge所有失败做连续footprint路径归因；
2. 保存plan几何、LQR reference、实际轨迹与首次无效点；
3. 审计静态障碍是否进入obs、哪个字段、坐标系和mask；
4. 对静态障碍输入做zero/mirror/distance/corridor干预；
5. 逐层记录Encoder、PolicyBackbone、PolicyHead响应；
6. 区分：
   - 表示缺失
   - 网络不消费
   - 横向路径错误
   - 纵向速度错误
   - LQR跟踪错误。

所有新模块和数据处理必须提供：
- 输入输出schema hook；
- 坐标系/单位/mask hook；
- activation/gradient/update hook；
- rollout真实态/预测态/action/plan hook；
- checkpoint/config/schema hash；
- 默认关闭bit-exact。

Round-34A完成后，启动Round-34B设计：
横向κ路径
+
路径条件化纵向v/s profile
+
外部兼容(ds,dtheta)
+
固定LQR。

解析ego-motion warp保留为Round-35结构化WM重训项，
不得用zero-shot NO-GO永久关闭。

最终判断

当前已经明确：

Action Conditioning V2
修复了约10%到12%的WM–Policy接口误差，
但无法解决路径选择、静态障碍和merge几何问题。


所以继续在原 ds/dtheta 输出和纯 latent WM 上微调，收益会越来越有限。

下一大步应该是：

路径级失败审计
→ 横向曲率路径
→ 路径条件化纵向速度
→ 显式静态障碍/走廊
→ 结构化OD/LD anchor与ego-motion补偿
→ WM/Policy交替适配
→ DAgger
→ MoE
→ RL


这比继续围绕单个 latent loss 或 action branch 调参，更有机会把当前 0.56 的闭环成功率推过 0.60、0.70，并最终挑战 IDM+Pure Pursuit 的 0.756。