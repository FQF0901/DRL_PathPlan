结论：基本达成共识，可以锁板

建议正式锁定 deepseekv4p1_argue.md §7.10 为 P1 执行手册 v1，实施范围明确为：

P1-0 → P1-1 → P1-2 → P1-3


同时明确：

4A：只锁定契约与最小实现，实际实验等 P1-2/3 达门槛后执行；
4B、K-anchor 重启、GRPO、Best-of-G、WM/MoE 对照：不属于本轮承诺；
不扩容量、不改评测协议、不改 WM/MoE 架构；
后续只允许依据 Gate 和实验数据调整，不再重新展开架构争论。
对 §7.10 的确认意见
A1：4A 排期，确认

采用文档中的最终解释：

P1-0 可先完成 docs/p1_4a_contract.md 和最小 action-conditioned plan 实现；
不立即跑性能实验；
等 P1-2/3 的监督模型达到 eval500 ≥0.60，最好 ≥0.65，再执行四类因果探针和 clean150 canary；
4A 不通过，直接停止，不进入 PPO。

这里建议再写清楚：

4A 的默认 base 是 P1-3 胜出 checkpoint；如果 P1-3 不优于 P1-2，则使用 P1-2 胜出 checkpoint。

A2–A4、A6：全部确认

包括：

rou64@e015 与 pri512@e020 必须同集重跑，不能拿 final.pt 的 0.332 与 epoch checkpoint 混比；
rou64 选择条件为 Δsuccess ≥2pp + McNemar p<0.05 + off-road 不恶化；
D2 V1/V2/V3 共 15 行，V3 优先，ref_speed_scale=1.0 必须与历史 D2 逐 episode 等价；
checkpoint、freeze、gradient、optimizer、geometry coverage 等 probes 按规定落盘；
P1-2 的 BC 防遗忘以首步 MAE、6 点 ADE 和 clean150 paired 共同判断；
R1/R2 消融、真实 recovery traj、时间衰减权重等均可锁定。
待确认量①：Recovery 样本量

§7.10 建议的“有效行 ≥10k、失败 episode ≥800、三档各 ≥1.5k、困难路型各 ≥10%”方向正确，但建议稍微调整，避免为了凑比例进行重复采样或扭曲自然分布。

建议锁定为两级门
采集可用门
有效 recovery 行数 ≥ 10,000
独立 episode 数 ≥ 500
单 episode 最多贡献 20 行
质量门通过率 ≥ 80%


限制单 episode 最多 20 行非常重要，否则 10k 行可能只是少数长 episode 的高度相关窗口。

三类最低覆盖
prevention ≥ 4,000
correction ≥ 2,500
late recovery ≥ 1,000


剩余样本允许按自然分布进入，不必强制精确达到 50/35/15。

困难路型最低覆盖

不要硬性要求 tollgate、merge、roundabout 各占全部数据的 10%。这可能严重改变真实训练分布。建议改为：

每类困难路型：
有效行 ≥ 300
独立 episode ≥ 30


如果自然采集达不到：

单独建立 targeted_hard_split；
训练时通过 sampler 权重使用；
不复制样本凑覆盖率；
在 manifest 中分别报告自然池和定向池。
最终确认值

我建议锁定：

有效行 ≥10k、独立 episode ≥500、每 episode 最多 20 行；prevention/correction/late 最少 4000/2500/1000；关键困难路型各 ≥300 行且 ≥30 episodes。

若达不到，不直接取消 P1-2，可以先做 pilot，但不能进入正式候选验收。

待确认量②：Anchor canary 判据

我不建议把 clean150 的 −2pp 设为单独硬门。

原因是 clean150 中：

1 个 episode 对应约 0.67pp；
−2pp 只有约 3 个 episode；
单 seed clean150 的统计噪声明显大于这个阈值。

“Δ≥−2pp 且不显著劣化”在 clean150 上既容易假阴性，又没有足够统计功效。

建议改成两段式 canary
第一段：clean150 灾难性回归门
Δsuccess ≥ -5pp
off-road 增加 ≤5pp
collision 增加 ≤3pp
无 NaN / collapse / expert occupancy 异常
重建 ADE、WTA、balanced accuracy 完整报告


这一步只负责排除明显崩溃，不证明 anchor 有效。

第二段：clean500 非劣门

只有第一段通过才跑 clean500：

paired Δsuccess lower bound > -2pp
off-road 不显著恶化
collision 不显著恶化
表示指标不得明显低于无 anchor 起点


建议使用 episode-level paired bootstrap，非劣 margin 设为 −0.02。

但有一个更重要的顺序问题

由于 4B 不属于本轮承诺，anchor canary 不应阻塞 P1-0→P1-1 主线。建议将它从 P1-0 必做项改为：

P1-3 后、准备评估 4B 时执行的前置 Gate。

这样不会为了当前不实施的方案占用主线 GPU 与工程时间。

§7.10 建议补充的三个文字修订
1. G2 的 off-road 基线注明跨版本属性

G2 使用 s11 的 off-road 率：

eval500 ≤0.098
clean500 ≤0.097


s11 来自 v7/obs v4 兼容栈，应注明这是目标锚点，不是严格同版本 paired baseline。正式 paired 主比较仍应针对 P1 起点模型。

2. Collision 门区分 Gate 用途

建议：

G2：各数据集分别不高于对应 IDM；
G3：统一要求 eval500 ≤0.144，clean500 ≤0.174；
若最终宣称模型整体优于 IDM，不能只靠 success 更高而 collision 明显更差。
3. P1-3 的几何监督名称保持准确

正式产物和配置统一使用：

ld_line_margin_loss


不要写成完整 drivable_footprint_loss。引擎 footprint 真值用于 hard mining、审计和验收，不假装是直接可微 loss。

锁板后的范围
纳入本轮
P1-0A  起点 checkpoint paired
P1-0B  D2 V1/V2/V3 配速矩阵
P1-0C  correctness probes + health report
P1-1   recovery_v1 数据集
P1-2   R1/R2 recovery trajectory BC
P1-3   几何代理 + 曲率 + progress 联合 recipe

条件执行
P1-4A  action-conditioned plan 因果验证


启动条件：

P1-2/3 胜者达到 eval500 ≥0.60；
correctness probes 全部 PASS；
sampled action 确实进入最终计划与 tracker reference。
不纳入本轮
P1-4B mode+latent PPO
K-anchor 重启
GRPO / Best-of-G
WM BPTT / No-WM 对照
MoE 0/2/8 对照
新容量扫描

最终裁定

可以锁板。

建议将 §7.10 状态更新为：

P1 执行手册 v1：LOCKED

锁定范围：P1-0 至 P1-3
条件范围：P1-4A
不承诺范围：P1-4B / K-anchor / GRPO / WM-MoE 对照

Recovery 数据门：
有效行 ≥10k、独立 episode ≥500、每 episode ≤20 行；
prevention/correction/late ≥4000/2500/1000；
关键困难路型各 ≥300 行且 ≥30 episodes。

Anchor canary：
不阻塞 P1 主线；
clean150 仅作灾难性回归门（Δsuccess ≥−5pp）；
进入 4B 前补 clean500 paired 非劣验证，margin=−2pp。


除了 D2 V3 ceiling、rou64 vs pri512、Recovery+几何能否达到 0.60 这三个必须由实验回答的问题外，已不存在需要继续讨论的设计分歧。下一步可以直接按锁板版本开工。