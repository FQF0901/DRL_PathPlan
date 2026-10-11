结论：已达成共识，可以开工

我同意 Coding Agent 在 §11.4 给出的锁板方案。主线、旁线、冻结范围和验收条件都已清晰，没有阻塞性分歧。

唯一需要措辞修正

用户问“达成公式”，这里应为达成共识。方案正式状态建议标记为：

P1-v2：LOCKED

锁定的主线
1. 可辨识性 probe
2. recovery_v2 同状态数据 pilot
3. gated residual pilot
4. geometry FP Gate
5. 独立小消融
6. candidate recipe


关键原则：

intervention 标签必须表示：同一状态下，baseline 将失败且 expert 能恢复；
recovery_v2 的 observation、baseline plan 和 expert label 必须来自同一时刻、同一状态；
gated residual 使用 2%–5% 自然 recovery 占比；
geometry proxy 未通过误报门前，不允许反向传播； -每一步都有明确的 Go/No-Go，不自动进入下一阶段。
同意并行的诊断旁线

可以在主线准备数据期间并行执行，但不得直接修改训练 target：

新增 dθ-only 反事实；
使用现有 phase3_bias_calibration 检查模型的 ds/dθ 系统偏差；
监控 Δκ 和 progress。

只有 bias 修正在不同速度带、道路类型和困难场景均稳定改善，才可升级为训练候选。

开工前再写清楚三个边界
1. 可辨识性 probe 结果不能只看总体 AUC

必须同时报告：

ROC-AUC
PR-AUC
precision@10%
2–4 m/s 子集
tollgate / merge / roundabout
matched-negative 结果


Gate：

ROC-AUC >= 0.80
precision@10% >= 0.60


如果总体通过、低速或关键路型严重失败，只能有条件通过，不能直接进入全量 gated residual。

2. recovery_v2 先做 2k–3k pilot

无需先采 10k。进入训练前必须：

同状态匹配率 = 100%
随机重放 >= 1,000 行
episode split 无泄漏
expert recovery 成功率 >= 80%


任何一项失败，先修 collector，不训练。

3. gated residual 不通过就停止，不继续加复杂度

clean150 的 pilot 门：

success >= baseline - 2pp
off-road 不增加
collision 增加 <= 2pp
正常状态 gate 激活率 < 5%
intervention recall >= 60%
首步 MAE / 6点 ADE 恶化 <= 10%
第二 epoch 不坍塌


不通过时停止，不转向更大 gate、更复杂 MoE 或更多 recovery 过采样。

冻结事项确认

本轮不做：

recovery_v1 继续训练；
mix ratio 和 freeze scope 搜索；
P1-4A/4B；
K-anchor；
GRPO / Best-of-G；
WM/MoE 重构； -新增容量扫描； -未经验证的 geometry 联合训练。
允许 Coding Agent 立即执行的第一批任务
A. 建立 docs/p1_v2_decision.md
B. 建立 docs/recovery_v2_contract.md
C. 实现三特征集 intervention probe 和 matched-negative 设计
D. 新增 dtheta-only forensics flag
E. 运行 phase3_bias_calibration 离线诊断
F. 完成 intervention probe 报告


只有 intervention probe 给出结论后，才进入 recovery_v2 collector 改造。

最终裁决

可以让 Coding Agent 开工。

没有剩余的原则性分歧。仅需确保：

诊断旁线不能越级变成训练方案，任何阶段失败都按 Gate 停止，而不是继续追加复杂度。