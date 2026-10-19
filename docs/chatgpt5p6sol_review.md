已审阅 deepseekv4p1_argue.md，并在 arch_lock.md 原有章节内完成局部增删，没有重构文档。

下载更新后的 arch_lock.md

一、对 Coding Agent 疑问的答复

这轮已经没有实质架构分歧，Coding Agent 撤回“根据相关性自动删除 v_next”是正确的，可以正式收口。

可以采纳
4D 动作 (dx,dy,dpsi,v_next) 固定，P0 只做校准与验收；
只有出现 action round-trip、停车起步或 Expert-GT LQR 的结构性失败，才暂停 P1 并人工复审；
v_next 使用“下一段起始速度硬继承 + acceleration/jerk 等 soft loss”；
D1/D2 冻结 Encoder 时 Target Encoder 固定，EMA 实际不更新；
Stage E 解冻 adapter 后才真正启用 EMA；
采用 10k Train + 500 Val + 500 Test，Calib50 从训练域抽取；
2k/200 只用于工程 smoke；
静态 LD/Nav 使用重投影与 re-encode consistency；
taxonomy 使用语义类型、动态属性、几何类型和状态类型等字段；
LQR 门槛先采用相对门槛，绝对值等 P0 获得真实分布后写入配置；
DAgger 混合比例由闭环指标触发，而不是机械按轮次下降；
worker 数、环境回收周期和内存水位属于机器运行配置，不属于长期架构合同。
明确不接受

不允许任何代码根据：

corr(v_next, distance / 0.5)


自动将动作从 4D 改成 3D。

相关性高只代表普通匀速样本占多数，不能证明终端速度冗余。停车、起步、障碍制动和慢车跟随仍依赖独立的 v_next。

P0 数值门槛如何处理

Coding Agent 提出目前缺少具体阈值，这不是技术分歧。当前合同只冻结：

必测场景桶；
必测指标；
相对推进条件；
结构性失败的人工复审机制。

最低样本配额、LQR 绝对误差、统计波动区间等，应在 P0 实测后写入：

config/eval_thresholds.yaml
docs/decision_log.md


不应现在凭空指定。

二、已更新到 arch_lock.md 的内容

本次在原章节内补充了：

4D action 固定及禁止自动降为 3D；
v_next 跨段硬继承；
acceleration、jerk、curvature 连续性 soft loss；
动力学边界和低速 stop/hold Controller；
OD/LD/RoadState 双轴 taxonomy；
静态设施不得进入动态 WM transition；
静态 LD/Nav 的 reprojection 与 re-encode supervision；
D1/D2 固定 Target，Stage E 才启用 EMA；
P0 按停车、起步、制动、变道、分岔、收费站等场景分桶；
10k Train + Val500 + Test500 + Calib50；
2k/200 工程 smoke 口径；
DAgger 自适应混合条件；
runtime_local.yaml 与 eval_thresholds.yaml；
新增动作、速度链、参数预算、LQR、Bank schema、评测隔离等测试；
新增失败类型：
PLAN_INTERFACE_SHIFT
ACTION_V_NEXT_INCONSISTENT
STATIC_REPROJECTION_ERROR
MEMORY_WARMUP_ERROR
TRAIN_EVAL_ASSEMBLY_MISMATCH
修正 P3 为短 PPO canary；
修正 P5 为完整 3 秒 LQR 正式闭环和正式 Real-ENV PPO；
删除了原文中的异常标题 ### 6.---。

现在可以将更新后的 arch_lock_updated.md 作为新的合同真源，并让 Coding Agent 开始 P0，不需要继续进行架构层面的攻防。