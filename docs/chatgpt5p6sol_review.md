总体判断

进展是可信的，执行纪律也很好：

recovery_v2_contract.md 的同状态契约已经真正落地，2,693 行、110 episodes，六项数据硬门全部通过，说明 recovery_v1 的错位问题已解决。
可辨识性在干净的 F2m 特征上达到 AUC 约 0.87–0.92，说明“是否需要干预”并非完全不可观测。
gated residual 的离线指标全部过门，但 clean150 从 0.3400 降至 0.3067，按照预注册规则停止，没有继续调参掩盖失败。这一决策正确。
关键判断：还不能直接认定只是“标定问题”

Coding agent 将失败定性为：

gate 分布迁移过触发，属于数据覆盖/标定问题，非架构问题。

这个结论方向大概率正确，但证据还差一步。

现有证据是：

离线正常样本激活率 3.99%；
闭环激活率 16.1%；
13 个成功变失败，8 个失败变成功；
off-road 和 collision 下降，但 timeout 增加，均速下降。

这同时可能由三种机制导致：

gate 误报；
residual 方向基本正确，但幅度过大；
gate 正确识别风险，但 recovery residual 过度减速，最终把 off-road 转成 timeout。

因此，不能只通过调高 gate threshold 得出结论。必须先做一次零训练反事实复盘。

对 Round-12 的裁决
批准 ④：先做反事实复盘

这是成本最低、信息价值最高的一步。

对 clean150 已保存轨迹，分别检查：

8 个失败转成功 episode；
13 个成功转失败 episode；
其余未改变 episode。

记录：

首次 gate 激活时间
激活持续长度
gate probability
residual ds / dtheta
base plan 与 final plan 差异
激活时 plan 是否已呈危险趋势
route progress
后续 termination 类型


并离线回放三个轴：

threshold：0.05 / 0.10 / 0.20 / 0.30 / 0.50
residual scale：0.25 / 0.50 / 0.75 / 1.00
temporal debounce：连续 1 / 2 / 3 步超过阈值才激活


这不是扫参训练，而是确定问题究竟在 gate precision、residual magnitude，还是缺少时间稳定性。

Go 条件

只有存在一个区域同时满足：

正常 episode 闭环激活率 <5%
intervention recall >=60%
保留至少 6/8 个翻正收益
消除至少一半 13 个翻负
timeout 不继续增长


才批准下一轮同架构 pilot。

有条件批准 ①：补部署分布控制行

如果反事实确认主要问题是 gate 误报，可以新增部署同源控制数据，但不建议重新采一整套 recovery_v3。

只补：

gated 模型闭环运行中的高分误触发状态；
pri512 正常闭环中的晚段、高 route-completion 状态； -低速 2–4m/s 正常状态；
tollgate、merge、roundabout 正常通过状态。

这些应作为 hard negatives，每行仍满足 recovery_v2_contract.md 的同状态契约。

建议新增约 1,000–2,000 行，而非扩大所有类别。

一个重要修正

不要把所有 gated 模型遇到的状态都标为 gate=0。必须证明：

base plan 在验证 horizon 内安全有效
且 expert intervention 没有明显优于 baseline


否则会把真正需要 recovery 的状态错误标成负例。

② 闭环阈值标定：可以做，但不能成为最终修复

阈值应按闭环分布确定，而不是沿用离线 0.05。这个修正合理。

但如果必须把 threshold 提得很高才能避免过触发，同时 intervention recall 大幅下降，说明 gate 的排序能力不足，不是标定可以解决的。

因此：

threshold 是部署参数； -不能用 clean150 直接选择并在同一 clean150 上报告最终收益； -应用一个校准 split 选择 threshold，再用独立 clean150/eval500 验证。

否则会产生评测集过拟合。

③ 特征离线化：暂缓

将 footprint/margin 从地图和 plan 离线重建，工程价值存在，但当前不是最短路径。

先完成反事实复盘。如果结果证明 F2m 特征本身稳定有效，只是负样本覆盖不足，再做离线化以扩大数据源。

如果最终发现主要问题是 residual magnitude 或时间抖动，特征离线化不会解决核心问题。

P1-3 geometry FP Gate：可以平行做，只诊断

批准并行运行固定数据上的误报 Gate，但仍不得进入反向传播。

核心判断：

成功 expert active-loss <5%
普通道路 <2%
失败 plan 检出率显著更高
合法变道/分叉无系统性误罚


当前 traj 与 geometry 梯度 cosine 为 −0.73，在误报原因解释清楚前，不能启动联合训练。

下一轮仅允许一个训练实验

建议顺序：

1. 翻正/翻负反事实复盘
2. 判断 gate、residual magnitude、temporal debounce 的贡献
3. 有必要时补 1k–2k hard negatives
4. 在校准集确定 operating point
5. 同架构重训一次
6. 独立 clean150 验证


不要同时：

改 gate 架构； -改 residual 架构； -加入 geometry loss； -重新调 recovery 占比； -解冻更多 base 参数。

否则又会失去归因。

下一轮验收标准

建议保持原 clean150 主门，并补两个闭环指标：

success >= baseline - 2pp
off-road 不增加
collision <= baseline +2pp
timeout <= baseline +3pp
正常 episode 闭环 gate 激活率 <5%
intervention episode recall >=60%
13 个历史翻负中至少消除一半
8 个历史翻正中至少保留 6 个
第二 epoch 不坍塌


如果修复后仍不过门：

关闭 gated residual 主线，不再继续追加 gate 特征、模型容量或采样技巧。

然后把资源转向：

V1 calibration mechanism 的可部署分解；
action/plan 纵向尺度校准； -后续具备合法信用链的 RL。
最终建议

批准 Round-12，但不是直接批准“补数据后重训”。

应锁定为：

Round-12A：零训练反事实复盘
→ Round-12B：必要时补 hard negatives
→ Round-12C：同架构仅重跑一次 pilot


冻结清单继续维持。recovery_v2_contract.md 不需要改原则，只需补充“部署 hard-negative”的质量定义。

一句话判断：

条件恢复路线还有一次值得验证的机会，但必须先分清是 gate 误报、残差过强还是时序抖动。若同架构修复仍失败，就应果断退出，不再把复杂度投入 gate。