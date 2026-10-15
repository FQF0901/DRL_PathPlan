# AGENT 守则与阶段对齐

## 系统守则

1. 系统只有一个 PolicyHead（StructuredPolicyHead：面对当前世界，下一步应如何规划？。横向分支 + 纵向分支 + 确定性组合层）。横向决定"从哪里走"，纵向决定"沿这条路径走多快"；不能做两个互不交流的横纵头
2. Latent World Model：执行动作后，世界会变成什么状态？（可以解码出：自车未来状态 / 动态对象未来状态 / 车道与道路相对状态 / 对象出现消失 / 场景上下文演化）
3. 同一 PolicyHead 与 Latent World Model 递归六次形成部署轨迹，六步只由这六次调用产生：真实状态 $z_0$ → PolicyHead 输出 $a_0$ → World Model 预测 $z_1$ → 同一个 PolicyHead 输出 $a_1$ → World Model 预测 $z_2$ → …… → 得到 $a_0 \ldots a_5$
4. 确定性坐标转换、碰撞几何和车辆运动学由解析层负责，世界残差由 World Model 学习，即World Model 只学：对象加减速 / cut-in / yield / merge 交互 / 地图重采样残差 / 可见性及 slot 变化。不能继续让 zod/zld 大量编码"ego 前进了多少"，否则会产生语义串扰
5. 外部只提供道路级导航和客观场景事实（车道几何/连接关系、静态/动态 slots），不提供唯一 target lane / 最佳收费闸口 / 推荐变道方向 / selected candidate / 规则生成的最优轨迹 / 规则速度答案 / 外部安全评分。阶段性例外（执行优先阶段）：允许外部合法 target lane 作训练输入，只用于测试"正确决策能否被执行"。不要再增加：更复杂的人工走廊分数 / 手写安全等级 / 手写目标通道代价 / 人工规划可行性标签直接作为 Policy 输入。理由：否则"规则规划器先做大部分决策 → 网络只学规则规划器的输出"
6. 车道/闸口选择与速度决策由 Policy 学习：生成哪些候选 / 选择哪条车道或闸口 / 如何修正横向路径 / 速度如何分配 / 是否变道 / 何时变道
7. 先证明网络能执行正确决策（给定合法目标的保持/变道/merge/收费站闸口），再训练网络自己做该决策
8. 能力层级：
   - 车道保持（直道/弯道保持、曲率连续性、路径可跟踪性）
   - 固定路径纵向能力（跟车减速、静态障碍停车、安全恢复加速、曲率条件化速度、避免 timeout）
   - 给定目标后的横向机动（不给网络做车道选择：左/右变道、merge 续接、道内避让、变道后回正、给定正确收费闸口后的通过能力）
   - 自主车道/通道选择（外部只给道路级导航，网络生成和评分：保持/左侧/右侧/route continuation/收费闸口/路口出口候选（导航角色门 + 选择门））
9. Primary / MoE / RL 分工（守则 9，保留）：Primary 先掌握车道保持、基础跟车、普通弯道、给定目标变道、基本恢复；MoE 只负责收费站、merge/yield、roundabout、复杂路口、特殊避障、高级恢复——不能让专家补偿日常 lane keeping 缺陷；RL 真实 transition 可用于后续离线更新 WM

## 阶段门控

| 阶段 | 训练项 | 冻结项 | 前置门 / 验收门 |
| --- | --- | --- | --- |
| A（WM） | 状态转移、动作条件化、物理状态预测、多步滚动稳定性 | Policy | Gate A0（工程契约，见 §4.4）；Stage A 过门后才能启动 B |
| B0（Primary 单步 IL） | PolicyBackbone / Primary / PolicyHead；数据 = expert real-state + 正确标注的扰动 + 少量 learner-state | WM、specific experts、router、ValueHead | teacher-state MAE 不恶化；learner-state MAE 显著下降；Primary-only 能完成基础 lane keep；Exact clean150 ≥ 0.55；LQR+WM-plan clean150 ≥ 0.45 |
| B1（Oracle-state rollout IL） | 真实未来 observation 按 $k = 0 \ldots 5$ 监督，权重 1.00/0.70/0.50/0.35/0.25/0.20 | WM | 同状态合法监督，无独立门（并入 B2） |
| B2（预测状态接口一致性） | $L_\text{interface} = \sum_k w_k \lVert \pi(\hat{z}_k) - \mathrm{sg}(\pi(z_k^\text{real})) \rVert$，同时保留 oracle-state expert 监督 | WM | E_interface h3–h5 再下降 ≥ 30%；oracle-state Policy 不退化；plan 曲率/进度不恶化；LQR+WM-plan clean150 ≥ 0.50 |
| B3（DAgger） | 完整 Policy + ExactTracker 真实执行采集（WM 适配 + Primary real/predicted-state IL 完成后） | WM | 暴露率 10%/20%/≤30%，最多三轮；每轮：learner-state MAE ↓≥15%、rollout consistency 不退化、teacher-state 退化 ≤10%、Exact clean150 ≥ +3pp、LQR+WM-plan clean150 ≥ +3pp；连续两轮部署提升 <2pp 则停止 |
| B4（MoE） | 只训 specific experts + router；随后允许 Primary 极低 LR 联合微调一次 | WM、Primary（第一阶段）、旧 PolicyHead、ValueHead | MoE Gate：full-MoE 相对 Primary ≥ +3pp；普通道路不退化；expert residual 在普通道路接近 0；router 与困难路型存在稳定关联；3 seeds 成立；不过门保持 Primary-only |
| C0（ValueHead） | Critic warmup（success/off-road/collision/timeout 前有分辨力） | Policy、WM | 分辨力自证 |
| C1（ExactTracker PPO） | Policy 单步 → ExactTracker → 真实环境转移 → reward → PPO | WM、ValueHead 之外按契约 | 每 ckpt 三层验收（§1.5） |
| C2（部署同构 PPO canary） | 随机变量 $= a_0$；$a_1 \ldots a_5$ = 冻结 WM + 冻结 tail Policy 的确定函数；fixed LQR 执行 plan；只更新产生 $a_0$ 的策略路径 | WM、tail generator（update epoch 内 old/new mapping 不漂移） | 仅当 C1 不能稳定迁移到部署时才执行；先 10–20 updates、3 seeds |
| C3（MoE RL） | Primary PPO 稳定后，才对 specific experts 做场景型 RL | — | 避免所有 expert 同时被全局 reward 拉动 |
| C4（GRPO） | 最后考虑，只用于同状态下多候选的组内比较 | — | 不替代 PPO 的 step-level Critic/GAE；PPO/critic 负责时间信用，GRPO 负责同状态候选相对排序 |

10. 所有阶段必须分别报告：ExactTracker：规划能力上界；Policy–WM 递归六次 → 六步计划 → 固定 LQR：真实部署性能。只有后者可以用于和 IDM + Pure Pursuit 比较（IDM 锚 = 0.756）。ExactTracker 三用途限定（`1c1d91b` §二）：①数据采集；②单步上界评测；③Stage C 的 Planner-only PPO 候选。任何在 ExactTracker 上提升的模型，都必须通过 WM rollout 6 步 + 固定 LQR 才能晋级。

## 当前基线

| 口径 | 读数 | 来源 |
| --- | --- | --- |
| Round-32 开发基线：K1e2 + MoE-off + K-anchor-off，LQR + WM plan | success 0.560 / off-road 0.140 / collision 0.080 | `d86e91a` §一.4 |
| 同基线 Exact | success 0.560 | 同上 |
| 同基线 LQR + repeat（S3） | success 0.060（0.02→0.06，仍崩） | `910b49b` §七 |
| S1≈S2 / S3 | S1≈S2≈0.56 继续成立；S3 崩溃继续成立 | `d86e91a` §一.4 |
| MoE 口径 | MoE-on 0.52 / MoE-off 0.56；训练口径 MoE-off 略优 2–4pp | `910b49b` §七 |
| IDM + Pure Pursuit（比较锚） | 0.756 | `910b49b`；`82e815b` L534 |
| 当前坐标 | 0.56 →（目标）0.60+ → 追平 0.756 | `c3bf69f` L906；`49e04d7` L488 |
| 现行同口径单轮硬门（R33C） | success ≥ 0.58 / off-road ≤ 0.14 / collision ≤ 0.08 / timeout ≤ 11 / route completion ≥ 0.862 | `f0ebfae` L76-80（timeout 原文口径如此，使用时复核单位） |
| WM 固有 TF vs 部署链 | 0.02–0.04 vs 0.20–0.32（8–10×） | R30；`f4faec8` |
| E_interface（ds） | h1 0.64 / h2 1.10 / h3 1.47 / h4 1.69 / h5 1.74；E_real ≈ 0.40；WM-state action 误差 ≈ 1.43 | `f4faec8`；round30_audit.md |
| D2→D3 占比 | ≈ 97%（含 `cde2e51` 保留意见） | Round-32 Level-1 |
| Action Swap matched | top-1 10.2% / top-3 63.8%（五候选随机期望 ≈ 20%；误差 spread 4e-3 vs 基准 0.08） | `cde2e51` §一 |
| A_k（Policy 放大） | ×2.1（p50 0.098→0.206）；Lipschitz 尾部 ≈6× | round32_audit_summary.md |
| whole-scene mirror | ≈ 0.842（95% 暂不作为硬门，只作分指标报告） | `d86e91a` §一.5 |
| 真实 latent 充分性 | ego 0.89–0.98；lane ld_slots 0.90/0.93（⇒ 排除 GO_STATE_REPRESENTATION_FIX） | round32_audit_summary.md |
| ego 泄漏读数 | OD/LD latent 中 ego speed +19.1%；OD pool ego progress +46.9%；ld_slots R² ≈ 0.9–0.99 vs real 0.08–0.14（非因果，见 §2.6） | `82e815b` L111-127；R1 覆盖项 |
