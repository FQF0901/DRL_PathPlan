总体判断

根据附件中对 docs\p1_0_report.md、docs\p1_2_diagnosis.md、docs\p1_implementation_contract.md、docs\p1_probe_spec.md 等产物的汇总，我认为：

P1-0 结论可信，可以关闭。
P1-1 数据集“格式验收通过”，但训练语义验收失败。
P1-2 路线已经被证伪，应停止追加剂量类消融。
P1-3 不应直接开训，必须先解释 grad cosine=-0.73。
原 §7.10 已完成历史使命，但不能继续原样锁板执行，应升级为 P1 v2。

一句话概括：

当前不是 recovery 数据不够，而是 recovery 监督的条件语义不成立。

一、对当前进展的审阅
1. P1-0：成果有效

三个重要结论已经清楚：

pri512@e020 作为起点合理，rou64 的 +1.4pp 不显著。
D2 的纯速度 V3 无明显收益；V1 在 1.20–1.30 达到约 0.76–0.778，但这不是单纯配速提升。
探针和 fail-fast 体系多次发现真实问题，证明这部分投入有价值。

但应修正一个表述：

V1 不是“纯教师上限”，因为只放大 ds 会降低等效曲率 dθ/ds。

它证明的是专家动作经过纵向尺度校准后能达到 IDM 水平，不能证明原教师或当前模型模仿上限已经达到 0.778。

因此，V1 可以作为 calibration oracle，但不应据此解除 4B/PPO 的教师上限门。

2. P1-1：数据门设计有盲区

数据集形式上达到：

17,256 行；
1,493 episodes；
13/13 数据门通过。

但训练后才发现 80% 左右的 k≥1 样本是：

Pass-1 student observation
+
Pass-2 expert-state label


这是反事实错位，不是一般噪声。

这说明原质量门验证了：

数量； -覆盖； -恢复结果； -文件完整性；

却没有验证最重要的契约：

每一行 observation 和 teacher target 是否来自同一个决策状态。

建议把本次 recovery_v1 明确定义为：

可用于诊断，不可作为正式监督数据集


不要继续在 recovery_v1 上做更多训练配方搜索。

3. P1-2：诊断基本可信，但“残差监督优先”仍不够

附件认定的核心问题是：

绝对 recovery target； -正常态与失败态在输入上不可分； -少量 recovery 行被高倍过采样； -导致模型在常见低速状态形成错误条件反射。

这个判断与以下现象一致：

离线 validation 很好，闭环迅速崩溃； -无 recovery 的 A2 基本稳定； -恢复比例从 25% 降至 2% 仍可能在第二 epoch 崩； -更换 freeze 范围不能解决； -对齐行、correction/late 子集也不能解决。

但我不同意简单把下一步定为“残差监督优先”。

原因

如果同一个有效 observation 对应：

正常状态：残差接近 0
失败恢复状态：残差很大


那么把绝对目标改成残差目标，仍然是一对多映射。残差会减轻破坏范围，但不会解决条件不可辨识。

因此正确顺序应是：

先证明 recovery need 在现有输入中可辨识；
再决定 residual、gate 或增加显式状态；
最后才重训。
二、P1-3 暂不能直接先行

附件提出 P1-3 可以与 recovery 正交推进，但实测：

cos(grad_traj, grad_geometry) = -0.73
19/20 samples < -0.3


这不是一般的权重问题，而是两个目标在持续互相对抗。

可能有三种根因：

expert trajectory 本身会触发 LD proxy；
LD proxy 把合法变道、分叉或收费站路径误判为危险；
recovery/BC target 与当前几何定义确实矛盾。

在确定是哪一种以前，不应通过减小 loss weight 掩盖冲突。

P1-3 开训前的必要验证

用不训练的固定数据计算：

成功 expert trajectory 的几何误报率
IDM 成功轨迹的几何误报率
模型失败轨迹的几何检出率
按 tollgate / merge / roundabout 分层的误报率


建议门槛：

成功 expert 的 active-loss 比例 <5%； -普通道路 <2%； -失败 plan 的检出率明显高于成功 plan； -合法变道不得被统一惩罚。

如果不能满足，ld_line_margin_loss 只能作为监控指标，不能进入训练。

三、建议锁定 P1 v2
Step 1：立即停止三类工作

停止：

在 recovery_v1 上继续搜索 mix ratio； -继续尝试不同 freeze scope； -直接启用 P1-3 联合 loss。

现有四轴消融已经足够说明，这三条路边际信息很低。

Step 2：先做“可辨识性 Gate”

在现有 observation 上训练一个轻量 probe，预测：

该状态是否需要 recovery intervention


输入只允许使用正式模型可见的信息，不能偷看未来 termination 或引擎私有状态。

报告：

ROC-AUC；
PR-AUC；
precision@10%；
recall； -按速度、road class、margin 分组结果。
决策
ROC-AUC ≥0.80：现有输入足够，采用 gated residual。
0.65–0.80：可尝试 gate，但必须增加 margin/history 特征审计。
<0.65：现有输入不可辨识，先修改 observation 或 recovery 定义，禁止继续监督学习。

这是下一步最关键、成本最低的实验。

Step 3：采集 recovery_v2

每一行必须严格满足：

obs_t
baseline_plan_t
expert_action_t
expert_recovery_plan_t
actual student state_t


全部来自同一个状态、同一个时刻。

核心改动：

不再用 Pass-1 observation 配 Pass-2 label；
k≥1 必须保存 Pass-2 对应 observation； -每一步都从当前真实状态重新查询 expert； -保存 baseline plan，便于构造 residual target； -按 episode 划分训练和验证； -采集后随机重放至少 1,000 行验证状态等价。

第一版无需再次追求 10k 行。先采一个 2k–3k 行 pilot，通过语义门后再扩容。

Step 4：训练 gated residual，而非直接绝对恢复

建议结构：

final_plan = base_plan + gate(obs) * residual(obs)


训练目标：

-正常 BC 行：gate=0，保持 base plan；

recovery 行：监督 expert_plan - base_plan；
gate 使用 intervention label；
residual 只在 gate 激活时产生梯度； -对 residual norm 加上限；
primary/base 分支尽量保持稳定。

初始 recovery 自然占比建议 2%–5%，不要再通过 40 倍过采样模拟 25% batch 占比。

如果需要平衡，使用 loss weight 或 episode sampler，但必须记录有效 importance ratio。

Step 5：P1-3 只保留已验证的项

先保留：

Δκ 平滑； -不可达曲率约束；
progress/timeout 监控。

ld_line_margin_loss 只有在 expert false-positive Gate 通过后才进入训练。

不要把 geometry、recovery、progress 一次性组合，否则再次失败时无法归因。

四、4A/4B 的裁决
4A

继续保持条件状态，不启动。

启动门仍应是：

监督模型 eval500 ≥0.60
correctness probes 全 PASS
sampled action 真正进入 plan 和 tracker reference


目前 P1-2 已崩，显然未满足。

4B

V1 达到 0.778 不能单独视为 4B 教师上限条件通过，因为 V1 同时改变了距离和曲率。

因此 4B 继续冻结，不应因为“有一条手工缩放链达到 0.75”而提前启动复杂 PPO。

五、建议产物调整
保留并定稿
docs\p1_0_report.md
docs\p1_2_diagnosis.md
docs\p1_implementation_contract.md
docs\p1_probe_spec.md
新增两个短文档即可
docs/p1_v2_decision.md

只记录：

被证伪的假设
保留的证据
停止项
下一假设
Gate
Go/No-Go

docs/recovery_v2_contract.md

只定义：

同状态 observation/label 契约
baseline plan
residual target
intervention label
重放验证
episode split


不要继续扩大 docs\deepseekv4p1_argue.md。它现在更适合作为历史审计记录，不适合作为日常执行手册。

六、最终判断

当前最优路线不是附件建议的简单顺序：

残差监督 → collector v2 → DAgger 复盘 → 门控


我建议改为：

可辨识性 probe
→ collector v2 同状态数据
→ gated residual pilot
→ 几何代理有效性 Gate
→ 独立小消融
→ candidate recipe

可以锁定的下一阶段验收
数据 Gate
observation/label 同状态匹配率 100%； -随机重放 1,000 行通过；
episode 无泄漏；
recovery expert 成功率 ≥80%。
可辨识性 Gate
intervention ROC-AUC ≥0.80； -低速 2–4m/s 子集不能明显失效； -困难路型分别报告。
Pilot Gate

相对 pri512 clean150：

success 不低于基线 2pp；
off-road 不增加； -首步 MAE/ADE 不恶化超过 10%； -第二 epoch 不出现崩塌； -正常状态 gate 激活率 <5%。

满足后再扩 recovery_v2 和跑 eval500。

一句话结论

P1-0 成功，P1-2 被可靠地证伪，P1-3 尚未获得训练授权。

下一步不要继续调剂量或组合 loss。应先回答：

模型从当前 observation 能否识别“此刻需要恢复”？

如果不能，残差、更多数据和更复杂 PPO 都只会把同一个语义冲突包装得更复杂。