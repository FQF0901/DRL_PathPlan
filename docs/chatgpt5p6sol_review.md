我看完了 deepseekv4p1_argue.md。Coding agent 的反驳大部分成立，但它的推进方案仍偏“先完善诊断、再逐步实验”，对于“显著超过 IDM=0.756”这个目标还不够聚焦。

一、我与 coding agent 的共识

以下判断可以直接接受：

让已经接近完成的容量排摸收尾，但不再扩大扫参范围。
不应预设删除 World Model，而应通过 No-WM 对照决定去留。
先区分 plan-out、tracker-out、异常判定和 recovery failure。
增加 footprint 插值检查，而不只检查六个离散计划点。
DAgger 应采集真实 recovery trajectory，不能继续依赖首步动作和合成 traj6。
MoE、WM、history 必须通过严格对照证明价值。
二、我认为 coding agent 仍低估了两个关键问题
1. RL 训练与最终评测的执行口径不一致，应升为最高优先级

目前：

RL 训练使用 repeat_action / A-hold；
最终评测使用 PlanHead 生成计划，再由 LQR 跟踪。

这意味着 PPO 优化的不是最终部署闭环。即使 PPO 在训练环境中学得很好，也不能保证 PlanHead + LQR 的评测性能同步改善。

这比 TemporalAttention、MoE 参数分配甚至部分 off-road 分类都更接近根因。

建议立即增加两条执行链对照：

PPO 训练和评测全部采用现有 A-hold；
PPO 训练和评测全部采用 PlanHead + LQR。

如果只有同口径链路明显改善，就能直接确认 train-eval execution mismatch。

2. 没有先确认“IDM 是否构成可学习上限”

当前训练数据主要来自 expert5k。如果专家本身接近 IDM，BC 很难显著超过 IDM；DAgger 再持续查询同一个专家，也只是在逼近该专家，而不是突破它。

要超过 IDM，必须至少有一种额外信号：

优于 IDM 的 oracle 或搜索型专家；
基于闭环 reward 的有效 RL；
failure recovery 标签；
对 IDM 弱点进行针对性优化，例如复杂几何、收费站、边界安全和路线完成率。

因此先做 expert ceiling audit：

IDM 在 expert5k 对应场景上的成功率；
expert action 经“网络 plan 接口 + LQR”回放后的成功率；
oracle/laneplan/精确控制模式的成功率；
IDM 失败而 oracle 成功的场景数量和类型。

如果 expert + 当前动作接口 + LQR 都无法明显超过 0.756，那么先改网络没有意义。

三、建议的实际推进顺序
P0：先回答三个决定性问题
P0-1：上限审计

用相同 eval500 测：

IDM；
expert action；
expert plan + LQR；
oracle/laneplan；
s11。

目标是得到：

环境理论上限
专家上限
计划接口上限
LQR 执行上限
学习模型当前水平


Go/No-Go：

若 expert plan + LQR ≤ IDM：先修计划接口或 tracker。
若 oracle ≫ IDM，但 expert ≈ IDM：需要更强专家或 RL。
若 expert plan + LQR ≫ IDM，但模型低：才是表示、监督或分布偏移问题。
P0-2：统一 RL 训练和评测执行口径

优先测试 PlanHead + LQR 闭环进入 PPO rollout。至少保证 PPO 的 action、plan 和环境实际执行之间有明确梯度或可归因关系。

短期无法完全统一时，也应让 PPO reward 针对实际 plan rollout 计算，而不是只针对重复首步动作。

P0-3：完成 off-road 归因

Coding agent 提出的四分类可以保留：

plan-out；
tracker-out；
判定或坐标异常；
recovery failure。

但不要等“100 个失败全部人工完善”才推进。先用 s11 与 IDM 各 30 到 50 个失败得到方向，随后再扩大到 100 个以上。

四、归因后的实现分支
如果主要是 plan-out

优先实现：

drivable corridor 输入；
footprint boundary margin；
节点间插值越界损失；
曲率和曲率变化率损失；
近场更密的轨迹点；
失败前 2–4 秒的 expert recovery trajectory。
如果主要是 tracker-out

优先实现：

tracker-aware trajectory loss；
bicycle-model 或实际 MetaDrive rollout 可达性检查；
限制最大曲率和曲率变化率；
把 LQR tracking residual 纳入 hard mining；
必要时重新调整 0.5 秒计划采样粒度。
如果主要是 recovery failure

这时单纯 BC 不够，需要：

DAgger 回溯失败前窗口；
专门采集偏离分布下的恢复轨迹；
对恢复样本提高权重；
PPO 在与评测一致的 PlanHead + LQR 链路上训练。
五、WM 与 MoE 怎么处理
World Model

不删除，但暂时不做大改。先比较：

当前 detached WM；
WM rollout 不输入 policy；
2-step BPTT WM；
No-WM 等预算模型。

只有 WM 能稳定降低 plan-out、collision 或 recovery failure，才继续投入 EMA target、长 horizon 等改造。

MoE

先继续使用现有 s11，不立即重构。待主链闭环对齐后，再比较：

单头；
2 experts；
当前 8 experts。

如果 8 experts 没有稳定优于单头，或者路由与道路类型、几何难度无稳定关系，应删除或缩减。现在先做 MoE 重构会混淆主要问题。

六、超过 IDM 的阶段性门槛

我建议不要直接把“超过 0.756”设为唯一终点，而是设三个 gate：

Gate 1：链路正确
expert plan + LQR 明显高于 IDM；
PPO 训练和评测执行口径一致；
95% 以上 off-road 能自动归因。
Gate 2：先追平
三 seed 平均 success ≥ 0.74；
off-road 相比 s11 至少下降 30%；
collision 不明显恶化。
Gate 3：确认超过 IDM
三 seed 平均 success ≥ 0.80；
相对 IDM=0.756 的 paired eval 有统计显著性；
bootstrap 95% CI 或 McNemar 检验支持提升；
提升不能只来自单一道路类型；
clean500 和 eval500 均成立。

所谓“大幅超过”，我建议目标至少设为 0.82–0.85，而不是 0.77 左右的噪声级领先。

最终执行建议

给 coding agent 的下一条指令应是：

允许当前 v8b 排摸收尾，但禁止扩大扫参。

最高优先级不是继续改 WM/MoE，而是：

1. 在同一 eval500 上完成 IDM、expert action、expert plan+LQR、
   oracle/laneplan、s11 的上限审计。
2. 增加 PPO A-hold 与 PlanHead+LQR 两条同口径训练/评测对照，
   确认 train-eval execution mismatch 的影响。
3. 用 s11 和 IDM 各抽取 30–50 个失败，先完成
   plan-out / tracker-out / anomaly / recovery-failure 分类。
4. 根据主要类别只实现一条优化分支：
   plan-out -> corridor/footprint/curvature；
   tracker-out -> tracker-aware rollout；
   recovery -> DAgger recovery trajectory。
5. 主链达到 0.74 以上后，再做 No-WM 和 MoE 等预算对照。
6. 最终以 3 seeds、eval500 paired test、success>=0.80
   作为超过 IDM 的第一阶段验收标准。


核心判断是：**先证明系统具备超过 IDM 的专家与执行上限，再统一 RL 与最终部署链路，最后才优化 WM/MoE。**否则很可能花两周把网络做得更复杂，却仍被 expert ceiling 或 PlanHead-LQR 接口上限卡住。