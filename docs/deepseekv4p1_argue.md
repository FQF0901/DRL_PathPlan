# deepseek-v4.1 argue：arch_lock staged review 的回应、反驳与修复顺序

> 本文件只承载针对 ChatGPT 5.6 sol / code review 的争论、辩驳、解释和修复顺序。
>
> 实现状态、代码证据、运行结果由 `docs/implementation_status.md` 独立维护；合同真源是 `docs/arch_lock.md`。
>
> 本轮审阅对象：`docs/arch_lock.md` staged review（2026-10-20，提交 `437b076` 相对 `af73a99`）。

---

## arch_lock staged review：回应、反驳与修复顺序（2026-10-20）

### 0. 总体判断

这份 review 的**事实层面大部分是对的**：它逐条对照 `arch_lock.md`，把“代码里确实没有”的东西标成“未实现”，把“实现了一部分但口径不同”的东西标成“有偏差”。我接受其中约 80% 的标记，尤其是 P1 的真实缺陷：

- Nav 来自 lane 端点/`current_ref_lanes`，不是纯道路级；
- OD/LD 没有 stable ID，6 帧 history 按槽位而不是按实体对齐；
- `v_next` 只作为标签，Exact/LQR 执行链只吃几何；
- 6 步 preview 是同一 batch 重复查询，不是真实 3 s rollout；
- PPO canary 的 log-prob 与 clamp 后执行动作不一致，不能称为合法 PPO；
- BC 失败 expert episode 未过滤；
- expert 特权审计未落盘；
- plan loss 只有 4D MSE，没有 heading/curvature/jerk/feasibility 分项；
- value/risk heads 基本未训练；
- 缺 tracking/saturation/wrong-route/stuck/recovery/场景桶指标；
- 10k/500/500/Test500 尚未冻结；
- 参数/延迟/初始化/token 数 probes 缺失。

这些不是“文档没写清楚”，而是 P1 实现确实没做到。我会按 §4 的顺序修。

但 review 里有 **3 类判断需要澄清或反驳**，否则会把“阶段未到”误判成“实现失败”，或者把“参考预算”当成硬门。

### 1. 需要澄清：P2/final 的“未实现”不是当前 commit 的 P1 偏差

Review 对 §1.3、§2.2、§5.2、§7、§13 D–G、§14 Test500 大量标“未实现”。这些标记在字面上正确，但当前 commit `437b076` 是一个 **P1/P2/P3 的提前实现、尚未过门**，不是 final system：

- 项目合同 §20 的编号是：P0 合同/测试/环境 → P1 单帧 BC → P2 真实历史 + DAgger → P3 短 PPO canary → P4 WM → P5 完整 3 s LQR + joint + 正式 PPO → P6 imagined RL。**“P2 = WM”是错误编号，必须全文改掉。**
- 当前故意没有实现 Real/Imagine Bank、Latent WM、Ego Dynamics、3 s Plan+WM+LQR、imagined RL、Test500。它们在 P1 阶段没有可评测对象；提前实现会重演旧分支“WM 先上、TF/部署 gap 8–10×”的问题。
- 因此这些项应该标成 **“未到阶段 / not yet in scope”**，而不是“P1 偏差”。否则任何 P1/P2/P3 的 commit 都会因为 P4/P5 没做而被判失败，review 失去阶段判决力。
- 但有三项是**已提前实现、未达门槛**，不能归为“未到阶段”：6 帧 per-slot temporal（A2 特性，但无 stable ID）、DAgger（B 阶段，但无 A2 checkpoint/failure-priority buffer）、PPO canary（P3，但 log-prob 合同不成立）。建议 review 增加第四类：**已提前实现 / 未达门槛**。

**建议**：review 增加一列 `阶段`（P0/P1/P2/final），并按阶段给通过/不通过；P2 项在 P1 review 里只记“未到阶段”，不计入 P1 gate。

### 2. 反驳：per-module 参数范围是参考，不是硬合同

Review 逐项对照 §8.2 的 per-module 范围，标出：

- OD 34.5k < 60k；
- LD 17.5k < 100k；
- Temporal 268k > 200k；
- Spatial 396k > 220k；
- Plan 132k > 120k；
- Value/Risk 258 << 30k。

这些数字是事实，但结论需要修正：

1. 合同原文写的是 **“目标约 100 万，允许 0.8M–1.2M 的可解释浮动”**，不是 ≤1M 硬门；当前 deployable 921,482，在容差内。per-module 范围是“参考预算/推荐起点”，上限相加约 1.19M，仍在 1.2M 容差内，不能同时当 1M 硬门。
2. 当前分配是**有意重分配**：把容量放在 temporal/spatial 交互上，把 encoder 做小；这是 P1 曲线/交互表现更好的直接原因。强行把 LD 拉到 100k、OD 拉到 60k，会挤压交互层，不一定提升闭环。
3. **Value/Risk 258 确实太小**，这是真实缺陷；我会把它扩到约 30–50k（`H→H→1` 两头），并从 spatial/temporal 里回收参数。LD token 上限 16→24 也可以单独做 ablation，但不作为 P1 硬门。
4. 当前没有 EMA target，因此 §8.1 的 “training-only target params” 为 0；这不是超预算，而是 P2 还没到。

**结论**：我接受“per-module 分配偏离参考”，不接受“因此违反 1M 参数合同”。我会修 value/risk 和 token 上限，但不会为每个参考区间牺牲总预算和闭环。

### 3. 修正：可观测量 / 派生量 / 选择量三分法（撤回上一版“都是普通特征”的表述）

上一版把 `nearest-OD / TTC / lat_offset` 都称为“特征”，这个表述过宽。按你的三分法，应拆成：

| 特征 | 性质 | 当前用法 | 处理 |
|---|---|---|---|
| `lat_offset` | 可直接测量几何量 | LD 每点特征；另经 role0 选 `lat0` 进 plan head | **保留**每 lane 的 `lat_offset`；移除 role0 硬选的 `lat0` 标量 |
| `heading_rel`、`curvature` | 几何量 | LD token 特征；另经 role0 选 `heading0/kappa0` | **保留**每 lane 特征；移除 role0 硬选标量 |
| `speed_limit`、`line_type` | 环境语义 | LD token 特征 | **保留** |
| `lane_role` | 车道身份/相对排序；合同 §3.2 允许 left/right topology，属于环境事实，不是 policy oracle | LD token 特征 + `role_embed`；role0 硬选 | **不与 role0 捆绑删除**；保留为候选输入，A3 单独 ablation。真正违规的是 role0 硬选+派生标量 |
| `TTC / od_urgent / ttc_min` | 由 OD 运动学派生的风险量 | `net/model.py:215-222` 选 token + scalar 拼 plan head | **从 plan head 移除**；raw OD 保留；TTC 作为 risk/TTC decoder 的辅助标签 |
| `od_near / od_min` | 手工最近邻选择 | `net/model.py:209-214` 硬选最近 OD + 距离 scalar | **移除硬选**；改 learned OD query attention；gap/TTC 只做辅助 loss |
| `role0` 硬选择 | 当前车道硬编码 | `net/model.py:203-205` 决定 `ld_role0/lat0/heading0/kappa0` | **移除硬选**；改 learned lane query + 辅助 current-lane supervision |
| `type_embed`、`static_embed` | 类型/静态属性，环境事实 | `net/model.py:89-90,136-137` | **保留** |

关键区分：`lat_offset/heading_rel/curvature` 是每 lane 的可观测量；`lat0/heading0/kappa0` 是“用 role0 硬选当前 lane 后再喂 plan head”的派生策略量。前者保留，后者移除。

**建议的 clean 结构**（先 A/B，不直接替换 A0）：

1. `role0` 硬选和 `lat0/heading0/kappa0` 派生标量从 plan path 移除；`lane_role` 是否保留单独做 A3（它本身是环境事实，不能和 role0 一起判违规）。
2. 在 spatial transformer 的 token 集合里增加 `q_lane`、`q_od` 两个 learned query，输出 `lane_ctx/od_ctx`；raw OD/LD token 仍全部参与 attention。
3. 辅助头：`lane_aux` 用 `lane_ctx` 预测 role0（及可选 lat/heading/kappa），`od_aux` 用 `od_ctx` 预测 gap/TTC，`risk_head` 预测 collision/off-road；这些量只做监督，不拼进 plan head。
4. Plan 输入改为 `[ctx_ego, nav_ctx, od_pool, ld_pool, lane_ctx, od_ctx]`；移除 `ld_role0, od_near, od_urgent, lat0, od_min, ttc_min, heading0, kappa0` 这些 raw/derived scalar。
5. 参数量：两个 query embedding + 辅助头约 20–40k；移除 `role_embed` 和标量拼接后，deployable 仍约 0.93–0.97M。
6. `train_bc.py` 目前用 TTC 给 braking/near 行做 sample weighting；这是 training-only curriculum，不是 plan input。保留但做 ablation；严格版可改用 expert braking 标签或均匀采样。

**A/B 计划**：A0 当前；A1 只移除 role0 硬选和派生标量（保留 lane_role）；A2 只移除 nearest/TTC 硬选；A3 单独测试 lane_role 保留/移除；A4 全 clean（role0/nearest/TTC 必去，lane_role 由 A3 决定）。  
统计口径：50 val 单 seed 的 2–3pp 完全在噪声内（二项 95% 半宽约 ±13pp），不能作为晋级 gate。改为 **paired per-scene 比较 + bootstrap CI**，或 ≥500 val/3 seeds；A0 只能作为 diagnostic baseline，若 A4 失败应查容量/优化/aux 权重，**不能把 A0 重新合法化为目标架构**。

**合同措辞建议**：在 `arch_lock.md` 增加 Feature Governance：可观测量（含 lane_role 这类环境拓扑事实）可作 input 或单独 ablation；TTC/gap 只作 aux supervision；最近 OD/当前 lane 必须 learned query，不得 hard-code；target lane/lane-level route/expert future 仍禁止。


### 4. 我接受的 P1 真实缺陷与修复顺序

按“先修根因，再谈 P2”的顺序。数据面顺序按 code_reviewer 修正，避免两次全量采数：

- **先做 4D 执行链 + `net/action_space.py`**：不依赖新观测 schema，立即止损、可独立验证；
- **一次 schema v2 迁移**：road-level Nav + stable ID/taxonomy + anchor + source/provenance + map-seed split 一起改；
- 用 **200–500 场景 schema smoke** 验证新 schema，不重复采 2k；
- 冻结 **10k/500/500 + Test500**，只做一次最终采集；
- 然后再做 plan loss/PPO/model split。

1. **Nav 去 lane 化**：先做 P0 API spike，确认 MetaDrive `road_network` 能给出稳定 road ID/link；若只能拿 lane route，就做 lane→road 映射，但绝不再用 lane endpoint/later_middle。`current_ref_lanes` 只允许作为 expert 标签，不得进 Student。
2. **Stable ID + taxonomy + same-ID temporal（与 Nav 同一次 schema v2）**：
   - OD 存 `track_id`、`semantic_type/mobility_type/geometry_type/state_type`；
   - LD 存可复现的 lane/polyline ID、固定弧长 chunk；
   - `stack_history` 按 ID 对齐，不再按距离排序槽位；
   - birth/death/presence mask 分开；source/provenance 字段落盘。
3. **4D 执行链 + action space**：先独立实现 `net/action_space.py`（bounds/normalization/feasibility），再让 `v_next` 进入 speed profile（不是只做几何 Hermite）；加 `a_long` 边界、`κ`/`a_lat` 限幅、feasibility gate、clip/reject/post-process 计数；LQR 参考速度用 `v_n→v_{n+1}`，不是 chord/0.1。
4. **6 步 preview**：P1 只允许 first-segment 执行，`plan_actions` 的 same-batch 重复 6 次标记为 **placeholder**，不得作为 3 s 规划证据；P4 WM 上线后替换为真实 imagine rollout。
5. **PPO log-prob**：当前 `sample→clamp→execute` 的 log-prob 不合法。要么实现 squashed Gaussian + logdet + executed-action 记录，要么把 `train_ppo.py` 降级为“wiring probe”，不称为 PPO canary。我倾向先降级命名，P2 前修分布。PPO canary 还没有验证 GAE 的 episode 边界、buffer 的 executed feasible action/clip mask、reward/cost 分离；这些都属于同一项缺陷。
6. **BC 数据质量**：过滤/降权失败 expert episode；加 `privileged_expert_sample` 与 DAgger source 标签；expert audit 字段表至少包含 target lane、lane-level route、centerline、static scan、future state、lane-change decision，并给出 privileged 标记规则；DAgger 至少两轮 + failure-prioritized。DAgger 标签同时保存 kinematic-bicycle lift 与 expert 0.5 s replay，报告 0.5 s 终点残差；超过 0.5 m / 0.1 rad 的行降权或丢弃。
7. **Plan loss 分项**：heading/curvature/acceleration/jerk/curvature-rate/feasibility + 6 步一致性，不再只有 4D MSE。
8. **Value/Risk**：扩头到 30–50k，训练 cost/value 分离；至少 collision/off-road/stuck/comfort 分项标签。
9. **评测**：10k/500/500 + Test500 按 **map seed** 冻结（不是 episode 随机切）；tracking/saturation/wrong-route/stuck/timeout/comfort/recovery/场景桶；checkpoint 选择按安全门槛→success→completion，不再按 val MSE；failure taxonomy 要落代码/统计，而不是只存在于文档。
10. **工程 probes**：params/latency/peak memory/init stats/token count；`test_detach_contract` 等。
11. **P4 WM 前置接口（P2/P4 共用）**：`encode_entities(batch) -> od/ld/nav/ego per-entity latents + ids/anchors/masks`；当前 `StructuredPlanner.forward` 只返回 fused `ctx`，P4 WM 无法构造 per-entity GT。没有这个接口，WM target 仍会错用 fused context。

### 4.1 P1/P2 重新评审 gate（未通过前不宣布 P1 通过，不启动 P4）

1. Val500 按 **map seed** 冻结（不是 episode 随机切），3 seeds 或 paired 统计；
2. road-level Nav（无 lane endpoint/later_middle），加 `test_nav_no_lane_leak`；
3. stable ID + taxonomy + same-ID temporal，birth/death/presence mask；
4. `v_next` 进入 speed profile，feasibility gate + clip/reject/post-process 计数；
5. expert audit + `privileged_expert_sample` + 失败 episode 过滤/降权；
6. exact + LQR 双口径：success/collision/off-road/wrong-route/stuck/comfort、tracking error、saturation、feasibility、段连接连续性；
7. Plan loss 分项（heading/curvature/accel/jerk/curvature-rate/6 步一致性）；
8. value/risk 扩到 30–50k，cost/value 分离；
9. checkpoint 按安全门槛→success→completion 选择，不再按 val MSE；
10. `encode_entities()` + `net/action_space.py` 就位（P4 前置）。

PPO 在 log-prob/GAE/buffer 修好前只保留为 wiring probe，**不计入 P3 gate**。

### 5. 我不准备做的“对齐”

- 不会为了把 checklist 全变绿而伪造 P2/WM/Test500 结果；
- 不会在 P1 根因（Nav/ID/4D 执行/PPO log-prob）没修前启动 P2；
- 不会强行把每个 per-module 参数范围调成参考值而破坏总预算；
- 不会把可观测量（`lat_offset/heading_rel/curvature/speed_limit/line_type`）删掉；也不会继续把派生量（`TTC/nearest/role0`）作为 plan head raw input——会改为 aux supervision + learned query，而不是让 1M 模型从 16 个槽位里重新发现 TTC。

### 6. 当前证据（供 review 对照）

- P1 best exact：`runs/p1_weight_hi`，50 val，success 0.70 / collision 0.16 / off-road 0.10 / RC 0.86。
- P1 LQR：`runs/p1_weight_hi_lqr50`，20 val，success 0.55 / collision 0.10 / off-road 0.15；`..._ff50` 50 val，0.54 / 0.06 / 0.22。
- Expert PurePursuitIDM（20 val）：0.80 / 0.05 / 0.05 / RC 0.95。
- Params 921,482；tests 7 passed；PPO canary only wiring, not valid PPO.
- 结论：P1 可用但未达 Expert；最大缺口是 collision（16% vs 5%）和 Exact→LQR 曲线 off-road（10% vs 22%）。

---

## 2026-10-21 — code_review @ 6de65f4：排模结论、接受、反驳与解释

**结论先行：我接受“P1/P2 gate 不予放行”。** 这次 review 的四个 P0 级问题我逐条复核后确认存在，且足以使当前 Val500/LQR/PPO 数字不能作为 gate 证据。下面区分“确认的缺陷”“影响范围的限定”“对合同细节的反驳/保留意见”，并给出修复顺序。

本轮只做排模和分析，没有修改实现代码；不会在 P0 修复前继续用这些数字支撑 P1/P2 结论。

### 1. 四个 P0：确认，且都进入训练/评测主链

| 项 | 复核结果 | 证据/影响 |
|---|---|---|
| P0-1 Nav 历史 100× | **确认** | Nav 的 entry/mid/exit/dest 在 `env/obs.py` 中按 `/100` 存储，但 `_compensate_nav_road`/`_compensate_nav_global` 用米制 pose 平移。直接探针：归一化 `entry_dx=0.07425`，ego 前进 7.425 m 后补偿为 **-7.35075**，正确值应为 0。当前 clean 数据的 6 帧 Nav 历史被污染；legacy 数据不走这条路径，所以 clean/legacy 的 A/B 存在不对称污染。 |
| P0-2 eval/PPO ego 历史语义错配 | **确认** | 采集侧 `prev_speed/prev_heading` 取段首值（`dataset/collect.py`、`collect_dagger.py`）；eval/PPO 在 5 个 0.1 s 子步后取段末值（`pipeline/eval.py`、`train_ppo.py`），因此下一帧 `a_long/yaw_rate/a_lat/curvature` 恒为 0。训练数据中 a_long 非零约 98.1%、yaw 非零约 86.7%，eval/PPO 中结构性为零。eval 还把 `prev_action` 设为 requested 而非 executed，fallback 时也不一致。 |
| P0-3 exact `ego.steering` 恒为 0 | **确认** | `ExactTracker.set_pose` 不写 steering；训练数据中 `ego[:, -1, 4]` 非零约 **97.2%**，exact 模拟中 reset 后和 5 子步后均为 0。即使修 P0-2，exact 的 ego 输入仍有分布偏移。 |
| P0-4 PSTL 失败标签未进入训练 | **确认** | 采集 npz 中有 `feasible/label_pass/action_in_bounds/label_source`，但 `pipeline/train_bc.py` 的 `KEEP_KEYS` 不加载这些列，`_plan_weights` 也不使用。DAgger r2 整体 feasible 约 83.4%，其中 measured replay 仅 **46.4%**；这些行仍以 plan_weight=1 进入 point/heading/velocity loss，NLL 只对 clipped target 计算。status 的“marked, not silently clipped”只在采集侧成立，训练侧确实没有消费。 |

补充确认的数字：

- clean checkpoint `runs/bc_v2_2k_q_dagger_r2` 的 `action_logstd` 均值约 **-4.64**，最小 -5.0；PPO 起点探索接近零。
- `gate([0.24,0,0.8,0.5], v0=0.5)` 当前被判 accepted，但 κ≈**11.96**、路径长度约 **0.249 m**；`_MIN_CURVATURE_PATH=0.25` 使短路径绕过 κ/a_lat 检查。这是真实的 gate 漏洞。
- `SPLITS` 仍只有 `("train","val")`；Test500 不存在。
- 全仓库没有 `import yaml`，`config/*.yaml` 不进入执行路径。

### 2. 我要限定/反驳的判断

1. **“所有数字完全不可用”需要拆开说。**  
   绝对闭环数字和训练分布对比确实不可用；但 clean 与 legacy 的 exact 评测都在各自旧 harness 下运行，错误方向并不完全相同（尤其 P0-1 的 Nav 历史污染只影响 clean）。因此正确结论不是“clean 架构被证明更差”，而是**当前 A/B 不可判决**。这与 review 的最终判决一致，但我不接受把 clean 的 -4.4pp 直接解释为架构失败；它至少混入了 P0-1/2/3 和 DAgger 数据质量差异。

2. **P0-4 不是“数学上污染评测数字”的同类 bug。**  
   它不会改变已跑 checkpoint 的 eval 轨迹，但会污染训练样本权重和 clean 模型质量。把它列为 P0 训练缺陷、阻止 gate 是合理的；把它说成“当前 eval 数字全错”的直接原因则不准确。两者应分开修复和归因。

3. **`§4.4` 的“联合 Transformer 一定违规”是合同细节层面的判断，不是行为级定论。**  
   当前 `net/model.py` 是 ego/nav/od/ld 单 joint TransformerEncoder，不是合同写的 OD/LD self + OD-LD cross + Nav/Ego cross。我接受“与合同文字不一致”这一事实，但不接受“joint masked attention 必然不能实现同样条件融合”的绝对说法。它可以作为待 ablation 的替代结构；在证明闭环等价前，不能声称满足 §4.4，也不能把它当成架构已解耦的证据。

4. **god module / 配置 / checkpoint 校验是工程质量债，不必全部阻塞 P0 修复。**  
   `env/obs.py`、`collect*.py`、`train_bc.py`、`train_ppo.py`、`tracking.py` 的体量和重复代码确实存在；但当前最优先的是 P0 数据/评测正确性。先修 P0，再做 schema/execution 唯一真源和模块拆分，避免在大重构中同时改变行为。

5. **`expert_plan_valid=success` 是过度保守，不是“失败 expert 仍被当正样本”。**  
   review 对 29.3% 行 plan_weight=0 的判断成立；但这也把 timeout episode 的早段有效专家行为整段清零。正确方向是按 label/segment 质量降权，而不是整条 episode 全零，尤其对 timeout 早段。

6. **Val500 的固定 seed 使用是有的，但协议冻结没有完成。**  
   我确实用同一批前 500 个 val seeds 跑了 clean/legacy；但没有 manifest/hash、spec 未入库，且 Test500 未建立。因此“使用过同一批 seed”不等于“Val500 已冻结”；review 的 gate 判断成立。

7. **LQR gap 不能直接归因于 Planner。**  
   LQR 只跑 0.5 s canary，preview 被 clamp 到参考末端；加上 P0-2/P0-3 的 ego 特征错配，当前 26% vs 64% 的 exact→LQR gap 不能作为架构结论。修复 P0 后必须重跑；这一点我同意 review 的“先修再判断”。

### 3. 我接受的其他关键点

- OD 只收 vehicle/cone/building，行人/barrier 缺失；`acc_x/acc_y` 用 0.5 s 差分而 `last_velocity` 按 0.1 s physics 更新，系统性缩小约 5×；age 只是窗口内停留时间。接受。
- LD 无 boundary polyline、lane type、稳定拓扑 ID；`LD_CHUNK_METERS=20` 被写进 stable ID，速度 >8 m/s 时当前车道的 6 帧历史基本退化为当前帧。接受。
- stable-ID 对齐和 SE(2) 补偿存在，但没有 content latent 拆分、显式 geometry delta、Δtimestamp、death/source mask。接受。
- Plan loss 的 curvature/jerk/κ-rate 是有限差分代理，不是真实 PSTL 轨迹量；无 3 s 一致性、无 heading wrap/跨段连续性。接受。
- Value/Risk 的 episode outcome 广播、value 只用 ±γ^remaining、无 route completion/time-to-go；接受。
- `privileged_expert_sample` 全 1、`w_privileged=1`，且 `_weighted_mean` 按权重和归一化，统一乘常数会抵消；该旋钮当前结构性无效。接受。
- DAgger r2 measured replay 仅 30.6%，69.4% 是 lift；`label_source` 未加载，lift 行 residual 为 NaN，`w_residual` 不生效；failure buffer、跨轮混合系数、逐轮 LQR shadow 未实现。接受。
- PPO cost_adv 已计算但没有进入 policy loss；只有 cost critic 在学；cost 不进 policy。接受。BC logstd 接近 -5；PPO 探索不足。接受。
- checkpoint 只按 open-loop val objective 选；Val500 未冻结；Test500 缺失；checkpoint 不校验 schema/bounds；config 不加载；2D `interpolate`/points-only API 仍活着；测试缺 eval/collect/PPO parity。接受。

### 4. 修复顺序（在继续训练前）

1. **P0-1 Nav 单位**：改 `_compensate_nav_road/global` 先乘 100 或直接存米；补 Nav pure-translation 测试；重算/重采当前 clean 2k/DAgger，旧 clean 数据不再用于 gate。
2. **P0-2/P0-3 共享 decision state**：抽 `prev_speed/prev_heading/prev_action` 的唯一实现，统一取段首 \(v_0/h_0\)、使用 executed action；exact tracker 要么写 steering，要么从 ego 输入移除 steering/改用 PSTL 曲率合成；加 collect/eval/PPO parity 测试（a_long/yaw/a_lat/curvature/steer 分位一致）。
3. **P0-4 数据质量消费**：train loader 加载 `feasible/label_pass/action_in_bounds/label_source`；PSTL 失败行丢弃/重降权；measured replay 与 lift 分开权重；`residual` 仅对 measured replay 生效；`w_privileged` 改为 per-sample/相对权重；补 multi_solution 标记。
4. **LD stable ID**：road pair + lane_id 作为稳定 ID，chunk_s 只作 anchor 特征；补 1/2/4/6 帧 ablation，确认 history 真的超过单帧后再声称 P2。
5. **Val/Test/checkpoint**：为 Val500 生成 manifest/hash 或入库；建立 Test500；best.pt 改为 collision/off-road 硬门槛→success→RC，不用 open-loop val objective。
6. **PPO/LQR**：PPO 只有 wiring probe 地位；cost 用 Lagrangian/硬约束进入 policy，BC 加 logstd floor/entropy；LQR 在 P0 修复后重跑 0.5 s canary，再决定 3 s preview。
7. **工程债**：YAML 真正加载并保存 resolved config；checkpoint 校验 schema/bounds/encoder version；抽 `dataset/schema.py`、`execution.py`、`checkpointing.py`；删除/标记 2D 旧路径。此步不应改变 P0 的语义。

**在 1–3 完成并重跑之前，我不会用当前 Val500/50-scene/LQR/PPO 数字支撑 P1/P2；P4 继续不动。**
