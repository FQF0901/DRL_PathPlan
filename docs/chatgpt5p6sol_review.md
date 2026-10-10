我审阅了最新 deepseekv4p1_argue.md。当前已经足够锁定 P1-0 至 P1-3，但不应把 P1-4B 的 mode+latent PPO 一并视为已批准实施。它应保留为条件分支，等待监督阶段结果和因果探针决定。

一、总体结论
可以锁定
先跑 rou64 eval500，确定 P1 起点。
完成 D2 配速 V1/V2/V3，裁决监督上限。
完成 A 批正确性 probes。
基于真实 student state 采 recovery 数据。
使用真实六步 recovery 监督，停止使用合成 traj6。
Recovery 训练开放 PlanHead fusion/primary，并对 experts 做 R1/R2 消融。
使用 LD 几何代理、曲率和 progress 联合训练，同时用引擎几何做审计与 hard mining。
所有候选先 clean150 筛选，再 eval500，最终补三 seed。
暂不锁定实施
P1-4B 的 categorical mode + 4–6D latent PPO；
重新启用 K-anchor；
GRPO、Best-of-G；
WM/MoE 架构对照。

原因是当前监督模型只有约 0.33–0.37，上述模块会同时改变表示、动作分布、buffer、log-prob、KL 和执行链，归因风险过高。仓库已有 K-anchor 失败史，虽然存在 obs、冻结范围等混杂，但目前也没有正向证据。

二、对 v0.5 的关键修订
1. P1-4 拆成“方案设计”和“实施授权”
现在允许实施：4A 因果烟测

4A 只做最小 action-conditioned plan：

sampled 2D action
→ 强制成为 plan[0]
→ 作为显式条件输入后五步生成
→ 六步计划进入 LQR


目的不是提升性能，而是验证：

后五步是否真的受 action 控制；
尾段是否对横向误差和航向误差有反馈；
sampled action 是否贯穿 plan、tracker reference 和实际运动；
条件化后是否仍保持 BC 基本性能。
暂不实施：4B mode+latent PPO

只有同时满足以下条件才授权开发：

P1-2/3 监督模型 eval500 ≥ 0.60，最好 ≥ 0.65；
4A 四类因果探针全部通过；
D2 V3 或其他可执行教师链证明存在 ≥0.75 的上限；
K-anchor v8 重拟合 canary 不低于相同起点模型；
action saturation、LQR saturation 均低于 5%；
recovery mode 标签具有稳定可分性。

任何一项不满足，4B 停止，不因为“代码已有基础”而继续。

2. Mode 语义暂不拍板

coding agent 建议“横向离散形状 + 纵向连续速度”，方向合理，但当前不应立刻固化为 PPO action space。

P1-1/2 只保留用于分析的标签，例如：

keep_lane
left_shift
right_shift
left_curve
right_curve
merge
turnaround_or_complex


先检查：

类别占比；
类内轨迹方差；
类间 ADE；
balanced accuracy；
tollgate、merge、roundabout 的覆盖率；
mode 在相邻策略步的切换率。

如果 balanced accuracy 只在 0.47–0.53 左右，不能直接推断它适合做 RL 离散动作。必须同时证明类别对闭环行为具有因果意义，而非仅是轨迹聚类。

三、锁定后的实施方案
P1-0：基线、上限与正确性契约
工作项
P1-0A：确定模型起点

运行：

rou64@e015 on eval500
pri512@e020 on the same eval500
paired McNemar + episode-level delta


选择规则：

rou64 success 高至少 2pt，且 paired 胜出、off-road 不恶化，选择 rou64；
差异小于 2pt 或结论不稳定，选择 pri512，因为其双集和审计证据更完整；
不再训练 pri512+rou64 组合，组合负交互已经证实。
P1-0B：D2 配速 V1/V2/V3

必须同时运行：

V1：ds *= scale，dtheta 不变；
V2：ds *= scale，dtheta *= scale，近似保持曲率；
V3：几何 plan 不变，仅缩放 LQR v_ref。

建议 scale：

0.90, 1.00, 1.05, 1.10, 1.15


V3 是主要裁决项；V1/V2 仅帮助判断几何与配速耦合。

验收不只看 success，同时看：

off-road
collision
timeout
route completion
steer saturation
longitudinal saturation
tracking error AUC


D2 的主要失败为 timeout，且 timeout 中 route completion 中位约 0.849，所以独立速度实验确有必要。

P1-0C：A 批 probes

必须包括：

checkpoint 加载比例与模块级覆盖；
plan/action/arc_step 契约；
0.5s/0.1s 时间索引；
plan/reference/actual 三轨；
LQR 饱和；
progress/timeout；
recovery 标签质量；
loss gradient 与 optimizer update；
freeze recipe 审计；
geometry proxy coverage；
action chain 与跨行 ego_world 对齐。
P1-0 产物
docs/p1_implementation_contract.md
docs/p1_probe_spec.md
runs/p1_0/model_start_paired.json
runs/p1_0/d2_pacing_matrix.json
runs/p1_0/health_report.json
runs/p1_0/action_ego_alignment.json

P1-0 验收
所有 correctness probes 为 PASS；
V3 默认 scale=1.0 与历史 D2 逐 episode 等价；
rou64/pri512 同集起点选择完成；
D2 ceiling 被归入以下之一：
≥0.80：监督上限充足
0.75–0.80：监督可追平，超越需 RL 或更强教师
<0.75：当前教师链不足


若配速提升伴随 off-road 或 collision 明显恶化，不认定为上限提升。

P1-1：Recovery 数据采集
采集原则

不能机械截取失败前固定窗口。起点由以下信号决定：

first_low_margin
T_plan
T_cross


分为：

prevention：仍合法，但 margin 开始下降；
correction：已经明显偏离，但尚可恢复；
late recovery：接近压线或已进入危险状态。

目标比例：

50% prevention
35% correction
15% late recovery

每条 recovery 样本必须保存
episode_id / map_id / road_class
student obs + history
student plan
student actual trajectory
expert current-state action
expert recovery 6-step plan
expert actual recovery trajectory
T_plan / T_cross / first_low_margin
surface / legal-line / route labels
expert_recovered
recovery_success_horizon
min_margin_after_takeover
route_progress_after_takeover
mode_label + mode_confidence
all validity masks

质量门

以下样本不得作为正监督：

专家接管后仍 off-road 或 collision；
recovery horizon 内没有恢复 margin；
route progress 没有改善；
action chain 与 recovery trajectory 不一致；
坐标或时间契约失败；
几何标签不可用。
P1-1 产物
data/recovery_v1/
  manifest.json
  samples.*
  schema.json
  quality_report.json
  split_manifest.json

docs/recovery_v1_dataset_card.md


必须按 episode 划分 train/validation，禁止同一 episode 的相邻窗口跨集合泄漏。

P1-1 验收
有效 recovery 成功率 ≥80%；
prevention/correction/late 分布接近预注册比例；
action/trajectory 对齐误差通过契约；
train/validation 无 episode 泄漏；
tollgate、merge、roundabout 等困难类型有明确覆盖统计。
P1-2：Recovery trajectory BC
两个正式消融
R1
train:
  plan fusion/trunk
  primary

freeze:
  encoder
  mem
  WM/ST-GNN
  router
  experts
  anchor heads

R2
train:
  plan fusion/trunk
  primary
  existing experts/residual

freeze:
  encoder
  mem
  WM/ST-GNN
  router
  anchor heads


不改变专家数、router 或负载均衡机制。

训练配方
recovery 样本占 batch 的 20%–30%； -其余保留原 BC 数据； -使用普通数据的 anchor/BC loss 防遗忘；
primary 使用较低 LR；
experts 使用独立 LR； -仅真实 recovery 行启用 traj auxiliary； -完全禁止合成 traj6 进入 loss。

六步监督建议：

1.00, 0.80, 0.60, 0.40, 0.25, 0.15


同时监督：

action chain； -相对 pose； -横向位置； -航向；
ds；
curvature/valid mask。
P1-2 验收

先 clean150：

success 相对起点至少 +5pt；
off-road 相对下降至少 15%；
collision 不增加超过 2pt；
timeout 不增加超过 5pt； -正常 BC validation 不退化超过 10%。

胜者再跑 eval500。若 R2 未显著优于 R1，则后续 recipe 使用 R1，不训练 experts。

P1-3：几何、曲率和进度联合 recipe
明确两个轨道
可微代理

命名为：

ld_line_margin_loss


基于 LD 点、line type 和 mask 构造，不能称为完整 drivable-area loss。

记录：

LD valid ratio；
solid-line available ratio；
plan coverage ratio；
loss-active sample ratio。
引擎几何真值

用于：

hard mining；
rollout KPI； -候选验收； -离线标签校准。

分解为：

surface validity
legal-line crossing
route-corridor consistency

曲率约束

使用：

κt=Δθtmax⁡(Δst,ϵ)\kappa_t = \frac{\Delta\theta_t}{\max(\Delta s_t,\epsilon)}

但必须：

mask 极小 ds； -主要惩罚 Δκ； -只对超出车辆可达阈值的 κ 强罚； -道路类型条件化； -联合 progress loss，防止降低速度规避所有风险。
梯度校准

既记录 norm，也记录：

cos(traj, geometry)
cos(traj, curvature)
cos(progress, geometry)


长期低于 -0.3 视为目标冲突，必须重新调权重或死区。

P1-3 验收

相对 P1-2 胜者：

eval500 success 至少 +3pt；
off-road 再下降至少 15%；
timeout 不恶化超过 3pt；
collision 不高于 IDM 的 0.144； -至少两个主要 road-class 分组改善； -无异常的 action/LQR 饱和增长。
四、P1-4 的正式状态
4A：批准为实验性因果门

产物：

docs/p1_4a_contract.md
runs/p1_4a/causal_probe.json
runs/p1_4a/clean150_canary.json


四类探针：

后五步对 2D action 的敏感性； 2.有效控制维数； 3.反事实单调性； 4.执行一致性。

还必须加入：

尾段对横向误差与航向误差反馈；
tracker 实际消费 reference；
LQR 饱和与实际运动响应。

如果不过，4A 停止，不进入 PPO。

4B：仅保留设计，不进入当前开工范围

在满足前述启动条件后，另行出：

docs/p1_4b_design.md
docs/p1_4b_distribution_contract.md
docs/p1_4b_buffer_migration.md


再决定是：

shape mode + speed latent； -纯连续低维 latent； -或继续 2D conditioned plan。

不能现在提前把 K-anchor 或 categorical mode 写成既定实现。

五、统一产物定义

每个正式实验目录必须包含：

manifest.json
resolved_config.yaml
git_state.json
checkpoint.sha256
checkpoint_load_report.json
health_report.json
episodes.csv
metrics.json
probe_results.json
stdout.log
decision.md


decision.md 必须只给出：

hypothesis
change
result
paired comparison
failure composition
probe status
go / no-go
next allowed action


正式实验不得被自动清理。

六、最终验收标准
开发门

clean150，单 seed：

用于方向筛选； -不作最终结论； -必须同时报告 success/off-road/collision/timeout。
候选门

3 seeds，eval500 + clean500：

固定 checkpoint selection； -逐 seed paired； -报告 mean、worst、range； -不得从多个 epoch 中事后挑最好结果而不记录 selection rule。
追平门 G2
3-seed mean success ≥ 0.74
off-road 相对 s11 下降 ≥30%
collision 不高于 IDM
clean500 与 eval500 均成立

超越门 G3
3-seed mean success ≥ 0.80
worst seed ≥ 0.77
paired bootstrap lower bound > 0
McNemar p < 0.05
collision ≤ 0.144
提升不集中于单一路型


正式声明达到 0.82–0.85 时建议补 5 seeds。

七、当前需要记录的开放项

只保留三个数据驱动问题：

D2 V3 配速后是否能达到 0.80。
rou64 是否在 eval500 上稳定优于 pri512。
Recovery + 几何监督能否把模型从约 0.37 推到至少 0.60。

其余争论暂时关闭：

不扩容量； -不改 WM/MoE 架构； -不重定义评测协议； -不启动 GRPO； -不默认重启 K-anchor； -不在监督基础不足时启动复杂 PPO。
最终建议

可以结束方案争论并执行 P1-0 → P1-1 → P1-2 → P1-3。P1-4A 只作为条件性因果烟测，P1-4B 不纳入本轮实施承诺。

建议给 coding agent 的开工指令是：

批准 P1-0 至 P1-3，按锁定产物和验收标准执行。

P1-0 首先完成：
1. rou64@e015 eval500 与 pri512 paired；
2. D2 V1/V2/V3 配速矩阵；
3. A 批 probes、freeze audit、geometry coverage；
4. action-chain 与跨行 ego_world 对齐；
5. recovery schema 与质量门。

P1-0 报告提交后，再启动 recovery 数据采集。
P1-4A 仅在 P1-2/3 监督模型达到门槛后实施。
P1-4B、K-anchor、mode+latent PPO、GRPO 均暂不授权。


这一版把主线、产物、退出条件和停止条件都固定了，后续应以实验数据推进，不再继续扩大设计讨论。