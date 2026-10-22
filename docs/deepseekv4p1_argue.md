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
   LQR 只跑 0.5 s canary，preview 被 clamp 到参考末端；加上 P0-2/P0-3 的 ego 特征错配，当时的 26% vs 64% 不能作为架构结论（该旧数已被下一节的 0.5 s canary 修正）。修复 P0 后必须重跑；这一点我同意 review 的“先修再判断”。

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

## 2026-10-21 更新：P0 修复后的实测结果

四个 P0 已落地并提交 `a6b2628`：

- Nav 历史归一化坐标先乘 100 再补偿；补了 Nav pure-translation 测试（当前 `entry_dx=0.07425`、前进 7.425 m 后为 -2.3e-9）。
- eval/PPO 统一使用段首 `v0/h0` 和 executed action；exact tracker 从 PSTL κ 合成 normalized steering。
- 训练加载并消费 `feasible/label_pass/action_in_bounds/label_source`；infeasible 行降权而不是全删；lift/replay 分开权重；failed expert episode 用 0.2 权重。
- 另修：LD stable ID 改为 road pair+lane_id（chunk_s 只作 anchor 特征）；OD acceleration 使用 physics 0.1 s；Val500/Test500 manifest 冻结（`docs/eval_split_manifest.json`）。

### 有效 exact 结果

2k canonical 数据重采：`datasets/bc_2k_s0..s3`，137,764 rows，expert success 70.6%，PSTL pass 95.6–96.1%。从零训练 BC/DAgger，50 val：

| 模型 | success | collision | off-road | RC |
|---|---:|---:|---:|---:|
| BC 2k | 8% | 22% | 70% | 0.42 |
| DAgger r1b | 44% | 24% | 32% | 0.733 |
| DAgger r2 | 50% | 32% | 12% | 0.774 |
| DAgger r3 + hard500 + safety weights | 58% | 12% | 16% | 0.844 |

Val500 exact（冻结前 500 seeds，paired）：

- legacy fixed：**65.0 / 18.2 / 12.0 / 0.854**
- clean r3：**66.2 / 12.2 / 8.6 / 0.859**
- paired diff：success **+1.2 pp**（CI −2.2..+4.6），collision **−6.0 pp**（−9.4..−2.6），off-road **−3.4 pp**（−6.0..−0.8）。

**历史结论（已被下一节 `code_review @ 639e958` 撤回）**：当时写“exact A/B 通过 success/no-worse 与安全指标”；下一节确认 success 的 CI 跨 0，不能称 formal pass。之前 clean 的 -4.4 pp 主要是 P0/数据污染，不是 clean 架构结论。

### LQR 仍阻塞

50 val LQR：

- legacy fixed：**44% success / 8% collision / 26% off-road / RC 0.776**
- clean r3：**34% / 2% / 12% / 52% timeout / RC 0.740**

单场景诊断：LQR 下模型在低速状态持续预测低 `v_next`（约 2.4 m/s），控制器只是跟随；这是 **closed-loop speed/distribution trap**，不是简单 speed-gain 不够（提速试验 20 val 反而 offroad 40%）。在 LQR 闭环速度参考、LQR shadow DAgger 或可行的 speed floor 解决前，clean 不能作为完整 controller-realized P1 目标；exact 结果可以保留为 planner-only 证据。

### 当前判断

- P0-1/2/3/4 已修并重采/重训；当时判断 exact A/B 有效且 clean 通过（下一节已把 A/B 降级为 safety 改善、success CI 跨 0）。
- LQR controller-realized gate 仍未通过；Test500 未运行；PPO 仍是 contract canary；P4 未启动。
- 因此不宣布完整 P1 通过，也不启动 P4。

---

## 2026-10-21 — code_review @ 639e958：逐条答复、LQR 优先级修正、数据/验证集方案

审阅范围：`6de65f4..639e958`；合同逐条批注在 staged `docs/arch_lock.md`（本轮不修改该文件）。下面是接受、纠正、反驳和实施顺序。

**先修正我上一节的两处过度结论：**

1. 撤回“exact A/B 现在通过”。success 的 +1.2 pp 的 95% CI 是 −2.2..+4.6，跨 0；只能主张 collision/off-road 显著改善，不能主张 success non-inferiority。legacy artifact 也没有等价 harness 的证明。
2. **更正（不是撤回）：0.5 s LQR canary 仍是 `arch_lock` §13 A2/B/C 的 gate，只是它不是最终 3 s controller-realized 成绩。** 我之前把“WM 后提高关注”误写成“当前不是 gate/可以整体推迟”，这是错误的；现在按合同恢复：0.5 s canary 指标现在必须报告，3 s/6 点完整接口在 WM/imagined rollout 后才成为重点。

### 1. 接受的部分

- **P0-4 数据语义仍不完整**：`expert_plan_valid=success` 是整段 episode 广播；失败 episode 统一乘 0.2，timeout 早段与 collision/off-road 失败段同权；`privileged_expert_sample` 全 1，`w_privileged` 在当前全 privileged 数据上结构性无效；`multi_solution` 不存在。
- **DAgger `intervened` 语义**：`dataset/collect_dagger.py:949` 写的是 `intervened = 1 if replay else 0`，`shield` 另有字段；`train_bc` 的 `w_intervened=3` 实际作用于全部 measured replay，不是只作用于 shield。接受。
- **Test500 不是真正的 test split**：`env/specs/scenarios_test500.json` 的 500 条 spec 仍是 `split="val"`；`SPLITS` 仍只有 `("train","val")`；没有 loader/CI 校验 manifest；Test500 未运行。接受。
- **checkpoint 选择仍是 open-loop**：`best.pt` 按内部 val total loss 选；r3 的 best 是 epoch 2/15；`load_model` 不校验 schema/action bounds/encoder version；`config/*.yaml` 没有被加载。接受。
- **PPO 仍是 wiring probe**：`runs/ppo_v2_canary/contract.json` 的 checkpoint 仍是 pre-P0 的 `runs/bc_v2_2k_q_dagger_r3/best.pt`，只有 4 transitions；`cost_adv` 在 `pipeline/train_ppo.py:666` 计算，但 policy loss 在 1120 行只使用 surrogate，没有 cost 项。接受。
- **PSTL short-path 漏洞**：`net/action_space.py:847` 的 `curvature_checked = length >= _MIN_CURVATURE_PATH`（0.25 m），短路径可绕过 κ/a_lat；review 的 `[0.24, 0, 0.8, 0.5]` 例子是真实漏洞。接受。
- **状态文档矛盾**：`docs/implementation_status.md` 的 exact A/B 与 LQR 旧数字确实自相矛盾；本轮已改写为“safety 改善但 success CI 跨 0 / LQR 只是 0.5 s interface smoke”，并把 LQR 复现命令改为 `scenarios_val500.json --limit 50`。接受。
- **工程质量 partial**：`env/obs.py` 1,430 行、`pipeline/train_bc.py` 1,199 行、`pipeline/train_ppo.py` 1,621 行、`net/action_space.py` 1,365 行、`env/tracking.py` 838 行；`net/model.py` 反向 import `env.obs`；`train_ppo` import `pipeline.eval`；`resolve_action`/fallback 在 eval、PPO、DAgger 有多个 mirror；`prev_action/prev_speed/prev_heading` 在 collect/eval/PPO 有多份拷贝。接受。P1/P2 gate 前只做最小抽取，不做“大重构”。

### 2. P0-2：review 的表述需要缩小，但残留缺陷成立

Review 写“eval/PPO 的 `prev_action` 仍是 resolved requested”。这句话对当前 `639e958` 的 exact 路径不准确：

- `pipeline/eval.py:298/321/327` 分别把 executed 设为 requested、v1 projection、safe brake；`pipeline/eval.py:446` 明确使用 `executed_action`。
- `pipeline/train_ppo.py:836` 使用 `resolved["executed_action"]`，`:954` 写回 `executed`。
- `dataset/collect.py:542` 使用 measured action；`dataset/collect_dagger.py:973` 使用实际执行的 `executed_action`。

所以“exact 下 requested vs executed 不一致”这一点不成立；exact 的 requested 被执行器精确执行，二者数值等价。

但接受更窄、真实的缺陷：

- **commanded 与 realized 没有被区分**。LQR 的 `executed_action` 仍是计划 4D，而不是由真实段末 pose/speed 反算出的 realized 4D；因此 LQR 下的 `prev_action` 不是 collector 的 measured 语义。P0-2 只在 exact 路径等价。
- **没有 collect/eval/PPO parity test**；82 个测试中没有 `test_train_eval_assembly_parity`。
- **PPO truncation 的 dt 边缘**：`pipeline/train_ppo.py:971-977` 的 bootstrap observation 固定 `dt=DECISION_DT`；若 max-step 在 5 子步中途截断，实际经过时间小于 0.5 s，`a_long/yaw` 的分母应使用实际执行时间；这一点接受。
- **PPO 的 `prev_speed/prev_heading` 是段首，已修；但真实 rollout 的 time bookkeeping 仍不是一个统一对象**。

修复方向：抽 `pipeline/execution.py`，明确 `commanded_action`、`realized_action`、`executed_steps`、`segment_dt`；collector 用 realized，exact 直接用 commanded（等价），LQR 由段首/段末 pose 计算 realized；`prev_state` 只从这里产生。先补 parity test，再谈 LQR。

### 3. exact A/B：不成立为 gate，但 safety 改善是真实证据

- success +1.2 pp，CI −2.2..+4.6，跨 0：**不能称为 success/no-worse gate 通过**。
- collision −6.0 pp（−9.4..−2.6）、off-road −3.4 pp（−6.0..−0.8）显著；这是当前 clean 的真实 safety 收益。
- legacy `runs/val500_legacy_p0fixed/metrics.json` 的 `config: null`，没有 commit/patch/dirty 信息；它是在临时 worktree 里对 `pre-consolidation` 代码打补丁后跑出来的。即使数值可复算，也不能证明它与 canonical evaluator 是同一个 harness；因此 A/B 是不对称的。
- 以后所有 **canonical clean 版本** 必须使用同一 `val500` manifest、同一 evaluator、同一 seed 集；legacy 只能作为历史参考，不能进入 gate 表。

我接受 review 的结论：“exact A/B 不能写成 passes”。上一节和 status 中的相关措辞已经撤回/改写。

### 4. LQR：0.5 s canary 仍是合同 gate，3 s 才是 WM/rollout 后的重点

我先纠正上一版本节的错误：我曾写“LQR 不是当前 gate、可以整体推迟到 WM 之后”，并把当前 0.5 s 结果称为“smoke only”。这个判断**不接受**。`docs/arch_lock.md` §13 A2/B/C 把 Plan Head 0.5 s segment + LQR canary 写成 checkpoint/阶段门槛；§6.3/§6.4 又要求最终 3 s / 6 点预瞄。两者是**当前 canary gate**和**后续正式 3 s 口径**的关系，不是“当前可以不做”。

正确的表述：

- **0.5 s canary 是现在的 gate**：必须报告 fixed seeds 上的 tracking error、可跟踪率、steering/brake/throttle saturation，以及 Exact 成功但 LQR canary 失败的比例；不能只用“无 NaN/无崩溃/无异常饱和”替代。
- **3 s / 6 点完整预瞄是 WM/imagined rollout 后的重点**：当前 evaluator 只执行 first 0.5 s（5×0.1 s），`preview=NOT_IMPLEMENTED`；所以当前 LQR 数字不是最终 controller-realized 成绩，不能作为“clean 完整闭环”结论。这个限定不等于取消当前 canary gate。
- 当前 clean r3 50-val LQR：34% success / 2% collision / 12% off-road / 52% timeout / RC 0.740；tracking xy 0.016 m、heading 0.008 rad、speed 0.210 m/s、steer saturation 0.0007、throttle saturation 0.0。它说明当前 canary 不通过/未闭环，而不是“可以不管”。
- 之前“低 `v_next` 速度陷阱”只有一个场景观察，没有 per-step artifact；调参同时改了 speed/accel/brake/lookahead，不能证明单一根因。降级为 hypothesis，不再作为 root cause。
- 不再为了 0.5 s success 盲调 gain；但必须先把 canary 指标表补齐、固定 seeds、逐 checkpoint 记录。3 s/6 点接口在 WM/rollout 之后接，再提高 LQR 权重。

**合同权威**：`arch_lock.md` 是合同真源；`implementation_status.md` 和本文件只能记录实现/争论，不能反向覆盖 §13/§20 的 gate。若“最新澄清”要改变 gate，必须先由合同 owner 修改 `arch_lock.md` 对应正文，或在 `docs/decision_log.md` 记录变更并复审；在此之前按合同执行。此前我在 status/argue 里写“LQR 不是 gate”，已撤回并改为“0.5 s canary gate，非最终 3 s 成绩”。
### 5. P0-4：接受“segment-level/语义”批评，不再只靠 0.2 权重

当前问题不是“没消费”，而是“消费的粒度和字段语义不对”：

- `expert_plan_valid=success` 在 `dataset/collect.py` 的 postprocess 中整段广播；失败段统一 0.2；timeout 早段也被同样处理。
- `privileged_expert_sample` 全 1；`w_privileged` 只在存在非 privileged 行时才有相对作用，当前结构性无效。
- `multi_solution` 不存在；多解场景仍用单轨迹 L2。
- DAgger 的 `intervened` 实际是 replay（expert override），`shield` 是另一个字段；`w_intervened=3` 没有只作用于 shield。
- 因而“P0-4 已修”只能表述为“列已加载、infeasible 已降权”，不能表述为“数据质量闭环完成”。

拟定的最小 schema：

| 字段 | 语义 | 训练用途 |
|---|---|---|
| `label_valid` | 该 decision 的标签可执行/可测 | 无效段归零 |
| `action_in_bounds` | 原始 4D 在 bounds 内 | 越界归零/修正标记 |
| `feasible` / `label_pass` | PSTL hard gate | 降权，不无条件全删 |
| `source` | expert / dagger | 分层权重 |
| `label_source` | measured / lift | measured 与 lift 分开 |
| `expert_replay` | 本段确实由专家接管 | replay 权重 |
| `shield` | 安全 shield 触发 | shield 权重；与 replay 分开 |
| `privileged_expert_sample` | 真的使用 lane oracle | per-sample/分桶 |
| `multi_solution` | 场景存在多条合法车道 | L2 降权/合法区域监督 |
| `segment_index` / `episode_outcome` | 段与 episode 结局分开 | timeout 早段不被 collision/off-road 晚段连带 |

权重规则改为：`label_valid` 硬门；`action_in_bounds` 硬门；`feasible/label_pass` 软降权；`expert_replay`、`shield` 分开；timeout 早段与 collision/off-road 失败段分开；`w_privileged` 要么改成 per-sample 相对权重，要么删除。P1 gate 前先把这个 schema 和统计写清楚，再谈重训。

### 6. Test500、manifest、路径和“所有版本同一验证集”

接受 review 的三个事实：

1. Test500 的 500 条 spec 仍是 `split="val"`；
2. `SPLITS` 没有 `test`；
3. manifest 没有 enforce，且放在 `docs/` 不合适。

**manifest 是什么**：它记录验证集的 seed 列表、spec 文件路径、数量、SHA-256 和 split rule，使“同一个验证集”可核验，而不是靠某次 run 的记忆。

**为什么 docs 错**：`docs/` 是解释性文档；manifest 是机器可读的评测契约/配置，应放在 `config/splits/val500.json`（并同时有 test 的 manifest）。`tools/freeze_splits.py` 是一次性生成器，放在 `tools/` 本身不算错，但 split 是数据契约的一部分，应该归到 `dataset/freeze_splits.py`（或 `dataset/splits.py` 的 CLI），`tools/` 只保留与环境/运行相关的 wrapper。

**为什么 Test500 不是 test**：spec 的 `split` 字段决定语义；把 500 条 val spec 改名成 test500 不会自动变成 test。需要在生成时写 `split="test"`，扩展 `SPLITS`，并在 loader 中禁止 train 读取 test。

拟定结构：

```text
config/splits/val500.json          # tracked: 500 seeds + SHA + split_rule
config/splits/test500.json         # tracked: 500 disjoint test seeds + SHA
env/specs/scenarios_val500.json    # generated, gitignored
env/specs/scenarios_test500.json   # generated with split="test"
dataset/freeze_splits.py           # generate + verify manifest
```

`pipeline/eval.py` 增加 `--split-manifest`，启动时校验 seed 集、数量、SHA；输出目录写入 manifest digest。所有 clean checkpoint 的 exact 评估都走同一入口。Test500 只在 P1/P2 gate 通过后运行，且只读。

### 7. 数据集目录：可以只保留四个逻辑数据集

可以，而且应该。当前 `datasets/` 有 **40 个 top-level 目录、约 4.6 GiB**，其中大量是历史版本，不能支撑“同一验证集、所有版本可比较”的要求。

建议：

```text
datasets/train_bc_2k/shard_0000.npz  ... shard_0003.npz
datasets/dagger_r1/shard_0000.npz    ... shard_0003.npz
datasets/dagger_r2/shard_0000.npz    ... shard_0003.npz
datasets/dagger_r3_hard500/shard_0000.npz
```

不再使用 `bc_2k_s0..s3` / `dagger_2k_r1_s0..s3` 这种“版本名+shard”并列的 40 个目录；顶层只保留四个逻辑数据集，shard 是内部实现。`dataset.load_data` 改为读取目录下 `*.npz`，训练命令只引用四个顶层路径。

清理顺序：先在 canonical 目录完成行数/seed/列/SHA 校验，并跑一个 BC smoke；确认 loader 和 checkpoint 都能读取后，再删除旧的 `bc_v2_*`、`dagger_v2_*`、`*_c*`、`bc_200` 等目录。不会在未验证 loader 前直接 `rm`。旧数据不是 git 资产，删除前会记录清单和大小；需要回滚时用采集脚本/commit 重建。

**所有版本同一验证集**：是。以后只允许用 `config/splits/val500.json` 的 500 个 seed 评测 clean 版本；如果某版本只能在旧 harness 下跑，它只能标为“历史参考”，不能进入比较表。当前 legacy exact 65.0/18.2/12.0 已经是 ad-hoc 结果，不再作为正式 A/B。

### 8. checkpoint、PPO、short-path、代码质量

- **checkpoint**：`pipeline/train_bc.py:873` 的 `selection` 明确是 open-loop val objective；`best.pt` 不是闭环选出的最终模型。接受。后续用 `pipeline/checkpointing.py` 校验 schema/action bounds/encoder version，用 `pipeline/gate.py` 按 collision/off-road 硬门槛 → success → RC 选；`best_loss.pt` 只作为训练诊断，不再叫 final。
- **PPO**：4 transitions / 25% mask / pre-P0 checkpoint，接受为无效 P3 证据。P3 不能在这个 artifact 上继续；要重跑必须用 P0 后 checkpoint、冻结 val、至少覆盖多个 episode，并明确 cost 是否进 policy。`cost_adv` 不进 policy 是事实，不是“reward 已包含 cost”。
- **short-path**：接受。`<0.25 m` 路径不应绕过 κ/a_lat；要么直接 reject，要么用端点解析曲率计算，不能把 `curvature_checked=False` 当作可行。补对应测试。
- **代码质量**：接受 partial。P1/P2 gate 前只做两件最小工程事：
  1. `pipeline/execution.py`：decision state、commanded/realized action、resolve/fallback、safe brake、termination、segment dt；
  2. `pipeline/checkpointing.py` + `pipeline/gate.py` + split manifest enforcement + parity test。
  `env/obs.py`/`action_space.py` 的物理拆分、YAML 全量接入、`net/model.py` 反向依赖、`train_ppo` 对 `eval` 的依赖后置到 gate 后；不为了结构好看改行为。
- **状态文档/合同权威**：本轮修正 `docs/implementation_status.md` 的 exact/LQR 矛盾，但不把 status/argue 当作 contract 真源。`arch_lock` §13 A2/B/C 仍是当前 gate；如果合同 owner 要改变 gate，应先修改 `arch_lock.md` 正文或在 `docs/decision_log.md` 记录并复审。当前不在 staged 合同上做未授权修改。

### 9. 修正后的推进顺序

1. **文档/契约层**：把 split manifest 移到 `config/splits/`，生成 `test` split，加 evaluator/CI 校验；数据集顶层收敛为四个逻辑目录；所有 run 记录 manifest digest 和 commit/dirty。
2. **P0-2/P0-4/short-path**：`execution.py` 唯一 decision state；realized action；PPO segment dt；segment-level 权重/字段；short-path gate；补 parity、weight、short-path 测试。
3. **checkpoint/gate**：`checkpointing.py` 校验 schema/bounds/encoder；closed-loop 选择顺序；统一 Val500。
4. **闭环提升**：先分析 66.2% 的 169 个失败场景，按 failure bucket 做 DAgger；再考虑 10k/500/500；Test500 只在 P1/P2 gate 后跑。
5. **LQR**：当前先按合同跑 0.5 s canary 指标（tracking/可跟踪率/saturation/Exact→LQR 失败比例，fixed seeds、逐 checkpoint）；在 ego-conditioned latent WM + imagined rollout 后，再接 3 s / 6 点 preview 并提高 LQR 权重。不能把当前 canary 说成“不是 gate”。
6. **P4/WM**：继续不动，直到 P1/P2 数据/验证/闭环证据成立。

### 10. Review 指出、我接受但上一版未覆盖的 5 个残留

这些不是反驳项，直接进入修复清单：

1. **exact steering 测试太弱**：`test/test_action_kinematics.py:133-142` 只断言 `abs(steering)>1e-6` 和 `[-1,1]` 有界；没有验证符号、`wheelbase` 来源、degrees/radians 换算，也没有 collect/eval 的 steering 分布 parity。功能修复保留，但证据等级只能算 P2；补精确解析断言和真实 rollout 分位数对比。
2. **`train_bc._plan_weights` 与日志漂移**：`pipeline/train_bc.py:1040-1048` 的 `residual_downweight_rows` 仍按原始 residual 计算，没有使用新的 `measured & finite` mask；checkpoint 的 `assumptions` 还写着 “failed expert episodes ... zero plan weight”，但实际已改为 0.2。补诊断字段和测试，不能只改权重不改报告。
3. **测试不 hermetic**：`test/test_obs_schema.py` / `test/test_eval.py` 依赖 gitignored `env/specs`；`test/test_train_bc.py` 的 fixture 没有 `feasible/label_pass/action_in_bounds/label_source`，`_plan_weights` 的新分支实际上没有被测到。改为 tmp 生成 spec 或带固定 fixture，并直接覆盖新权重分支。
4. **`OD_ACC_DT` 硬编码 0.1**：`env/obs.py:233,772` 修对了单位，但没有从 `decision_repeat × physics_step` 或配置推导；配置变化会静默回退。应集中到环境常量/配置并加测试。
5. **LD stable ID 缺跨 chunk 测试/ablation**：`ld_ids=hash(scene_seed, road_pair, lane_id)` 已修，但无跨 20 m chunk 的 history 测试，也无 1/2/4/6 帧 ablation；代码接受，“history 真的带来收益”仍未证明。

这些项不改变当前“P1/P2 未过”的结论；但如果不修，canary/权重/冻结验证集的证据链仍然不闭合。

### 11. 对本次复核的最终答复

- **接受本次复核的 80–85% 判断，并接受两个拒绝项。** 我撤回“0.5 s LQR 不是 gate、可以整体推迟到 WM 后”的说法；撤回“status/argue 可以覆盖 arch_lock”的做法。`arch_lock` §13/§20 是 gate 真源；如果要改变 LQR gate，必须由合同 owner 先修改 contract 正文或在 decision log 记录变更，不能由实现文档反向改写。
- **保留的只是范围限定，不是 gate 豁免**：0.5 s LQR 是当前 canary gate，3 s/6 点是 WM/rollout 后才成为重点的正式 controller-realized 口径。当前 must-report：fixed-seed tracking error、可跟踪率、steering/brake/throttle saturation、Exact 成功但 LQR 失败的比例。不能用“无 NaN/无崩溃/无饱和”替代。
- **状态文档已按新口径改**：`implementation_status.md:115` 删除/限定旧的 “50% then 64%”，改为当前 r1b/r2/r3 44/50/58；`:168` 改为“0.5 s canary gate，不是最终 3 s 成绩”；`:32` 改为 P0-1/P0-3 fixed、P0-2/P0-4 partial，不再写成全部 closed。
- **5 个残留**（steering 测试、plan-weight 诊断、非 hermetic 测试、OD_ACC_DT、LD chunk 测试/ablation）已进入修复清单，不能作为“已证明”计入 gate。
- **本轮不做代码语义修改**；先把 contract/状态/证据口径修齐，再按 execution.py → checkpointing.py/gate.py → LQR canary 指标 → DAgger 的顺序推进。

**结论：** 接受 review 的“P1/P2 未过、Test500/checkpoint/PPO/short-path/data semantics 仍需修”；撤回“exact A/B passes”；接受“0.5 s LQR canary 仍是合同 gate”，只把 3 s/6 点完整口径放到 WM/rollout 后。当前最重要的是把验证集和数据目录冻结成唯一口径，补齐 canary 指标和 P0-2/P0-4 证据，再把 66.2% 的失败场景变成下一轮 DAgger 的目标。

---

## 2026-10-22 — bugfix 落地、专家上限与实测结果

本轮按“先修正确性，再按合同 gate 推进”执行，代码提交 `a639959`。没有修改 `arch_lock.md`。

### 已落地

- `pipeline/execution.py`：统一 segment-start `prev_speed/prev_heading`、measured/realized 4D action、`segment_dt`；eval/PPO 使用 realized action，PPO truncation 使用实际执行时间。
- `pipeline/checkpointing.py`：checkpoint schema/action bounds/decision 校验；eval `--strict-checkpoint` 会拒绝不匹配的 checkpoint。
- PSTL short-path：`<0.25 m` 且运动/转向时仍检查 κ/a_lat，旧的 `[0.24,0,0.8,0.5]` 现在 reject。
- OD acceleration：从 `physics_world_step_size × decision_repeat` 推导，保留 0.1 s fallback，并加测试。
- train_bc：failed expert 只降权 terminal segment；`shield` 与 replay 分开；residual 诊断使用 measured/finite mask；fixture 覆盖 feasible/label_pass/action_in_bounds/label_source；canonical shard loader。
- Val/Test：迁移到 `config/splits/val500.json` / `test500.json`；Test spec 写 `split="test"`；evaluator 校验 seed/SHA；删除 docs 下的 manifest 和 tools 生成器。
- 数据/运行清理：`datasets/` 只保留 `train_bc_2k/`、`dagger_r1/`、`dagger_r2/`、`dagger_r3_hard500/`；其他目录移到 `/tmp/opencode/datasets_archive_20261022/`（未删除）。`runs/` 只保留当前 lineage/evidence（69 MiB），旧目录移到 `/tmp/opencode/runs_archive_20261022/`。
- Tests：95 passed, 1 skipped。

### 实测：先看专家上限

在 500 个冻结 val seed 上采集当前 Expert：

- success **73.2%**, collision **8.7%**, off-road **7.5%**, timeout **10.6%**, stuck 0%。

因此“用当前 Expert 示范把 Val500 做到 100%”在数据上是不可能的；clean 的当前上限接近 Expert，而不是 100%。这是必须报告的结构性 finding，不能用更多 epoch 或调参掩盖。

### exact Val500

| model | success | collision | off-road | stuck | RC |
|---|---:|---:|---:|---:|---:|
| legacy ad-hoc | 65.0% | 18.2% | 12.0% | 0% | 0.854 |
| clean r3 | 66.2% | 12.2% | 8.6% | 0.4% | 0.859 |
| clean r3b | **67.6%** | 15.6% | **7.4%** | 4.6% | 0.857 |

r3b − legacy：success +2.6 pp（95% CI −1.2..+5.8），collision −2.6（−6.0..+0.6），off-road −4.6（−7.0..−2.2），stuck **+4.6**（+2.8..+6.4）。success/collision 不显著，stuck 显著更差；**不是 gate pass**。r3 更安全，r3b 更高 success，两者都不 promotion。

r4 / r5 也未超过：
- r4: 50 val 62/16/12/10；
- r5: 50 val 66/8/14/12，但 Val500 只有 64.8/13.8/11.8，stuck 5.0。

### LQR 0.5 s canary

- r3b: 48% success / 6% collision / 10% off-road / 36% timeout / RC 0.797；tracking xy 0.024 m、heading 0.009 rad、speed 0.247 m/s。
- r3: 34% / 2% / 12% / 52%。
- 同一 50 seeds：exact r3b 成功 34 个，其中 **11 个 LQR 失败（32.4%）**。
- 0.5 s canary 仍未闭合；它不是最终 3 s controller-realized 成绩，但按合同必须报告，不能以 NaN/crash-only 替代。
- 本轮还发现并修复一个 LQR 数值 bug：时间律速度出现 −1.2e-9 量级负值导致 `set_reference` 抛错；现在对微小负值 clamp，大负值 fallback，而不是让评测崩掉。

### 结论

- P0-1/P0-3 已修；P0-2/P0-4 仍是 partial，但共享执行/权重语义已明显收敛。
- P1/P2 仍未通过：exact success 不显著、stuck 退化；LQR canary 未闭合；Test500 未运行。
- Expert 上限 73.2%，100% 目标在当前数据/专家下不可实现；下一步应优先 **强 expert / RL / 课程与 failure bucket**，而不是继续在 67% 附近堆 epoch。
- P4 WM 未启动；继续遵守 arch_lock §13/§20。

### 追加（2026-10-22 02:00）：按 §13/§20 已推进到的位置

在修正 P0/接口后，本轮实际推进了：

- **Stage A2 证据**：用同一 r3b checkpoint/seed 做 `hist_valid` 推理 ablation，1/2/4/6 帧 success = **16/38/52/68%**；证明 history 不是装饰。但这不是重新训练的 1/2/4/6 模型，合同要求的 training ablation 仍 open。
- **Stage B**：加入 `student_failure` failure-prioritized plan weight；r6（r5 data + `w_student_failure=2`）50 val 只有 60/14/12/14，没有超过 r3b；失败优先权重本身没有解决问题。
- **Stage C wiring canary**：`runs/ppo_v3_canary` 20 episodes / 5 updates / 399 transitions，boundary 0、logprob diff 0、ratio-at-old 1.0；`cost_adv` 已作为 Lagrangian 项进入 policy loss。仍是短 canary，不是 P3 性能 gate。
- **P0 仍缺**：Expert-GT 3 s + LQR controller smoke、action-parameterization veto 的完整报告；这两项没有假装完成。
- **未推进 Stage D/P4**：arch_lock §13 要求 P1/P2 通过后才进 WM；当前 exact success 67.6 vs Expert 73.2，stuck 4.6 vs legacy 0，LQR canary 未闭合，因此不启动 WM。这不是“跳过”，而是 gate 未过。

**为什么没有继续堆 Stage D/P4**：不是不知道 §13/§20，而是继续堆模型会违反“先过 gate 再上 WM”的硬顺序；在本轮 P1/P2 仍未通过、Expert 上限只有 73.2% 的情况下，强行上 WM 只会得到不可解释的数字。下一步应补 Expert-GT 3 s LQR、真正的 1/2/4/6 训练 ablation、P3 hacking audit，或引入更强 Expert/RL；而不是在 67% 附近继续堆 epoch。

---

## 2026-10-22 — oracle review 回复与修正

Oracle 对 `ac8beb7..5932122` 的 review 结论我基本接受：**D1/D2 不能作为 gate 证据**，ledger 当时确实 overclaim。逐项处理：

### 接受并已修

- **R3 persistence 广播 bug**：`_persistence` 用 `[N,1]` mask 广播成 N×N；已改为显式 reshape/broadcast，并加 2-D mask 单测。修正后 D1 WM **83.39** vs persistence **~1.0**，WM **失败**；不再声称“beats persistence”。
- **R1 coverage/D2 leakage**：`_build_pairs` 不再取第一个目录前缀；改为全量 pair 后按 seed 抽样；`probe_wm` 只评估 held-out scene，并记录 data/ckpt/split。D1/D2 仍不是 gate。
- **R2 action condition**：`executed_action` 已进入 loader；`train_wm`/`probe_wm` 用 executed action，缺失时回退 action。
- **R4 terminal row**：BC collector 之前仍在保存前 break；已删除该 break，实际 terminal segment 会写入，`terminal_failed` 由 `_backfill` 标记。旧 r3b/r4/r5/r6 数据仍不含此语义，已标 debt；terminalfix2 重采已启动。
- **R5 PPO artifact**：去掉了 contract 里重复/覆盖的 `schema_version`，分别记录 `contract_version`/`obs_schema`；per-transition 容差 1e-5、batch 诊断容差 1e-4；PPO warm-start 改为 strict loader。
- **R6 WM 参数**：smoke 模型改为 hidden=80/latent=128/layer=1，约 **245,792** params（记录在 config）。
- **R7 static targets**：还没有按 §9.2 的 static-LD reprojection/Nav re-encode/Ego dynamics 重写；这是未关闭项，不能把 D1 当 gate。

### 新增实测

- `runs/ppo_v7_canary`：20 episodes / 398 transitions；per-transition logp diff 0，TF32 关闭；batch 诊断 1.1–2.6e-5；LQR shadow 50 = 40/2/14/44。仍是 wiring canary，不是 P3 gate。
- `runs/ppo_v5_tf32off` 证明 **GPU TF32** 是 batch old-logp 0.01 级别偏差的根因；关闭 TF32 后降到 1e-5 量级。
- `runs/wm_d1_fixed2`：旧指标；已被下文明细修正为 D1 .616 vs 1.051（旧 83.39 是 `masked_mse` 的 128× 放大）。
- `runs/wm_d2_probe_fixed`：旧、非 held-out probe，已被 `wm_d2_probe_final2` 取代。

### 判断

- Oracle 的 R1–R7 大部分接受；R7 和训练版 A2 ablation 仍未解决。
- D1/D2 不能进入 Stage E/F；Stage C 仍是 wiring canary。
- terminalfix2 重采完成后，再决定是否重训 BC/DAgger；旧数据不作为 B-gate 证据。

### 2026-10-22 Oracle 第二轮复核后的修正

接受第二轮 review 的 R1–R6；R7 仍未解决。关键修正是：

- **R3 指标**：`masked_mse` 的分母也要扩展到 latent 维度；之前 D1 的 83.39 是 128× 放大值。修正后 `runs/wm_d1_final` 为 **0.616 vs persistence 1.051**（od .162/.185、ld .401/.648、nav .007/.021、ego .046/.198），D1 one-step smoke 通过 baseline；不再写“D1 fails”。
- **R1/R2**：D1 pair sampling 覆盖 all data，D2 只在 WM checkpoint 的 `val_scene_seeds` 上评测；action 用 `executed_action`；`runs/wm_d2_probe_final2` 是 held-out，depth 6 WM od .522 vs persistence .673、ld .990 vs 1.182、ego .133 vs .617。
- **R4**：BC terminal partial segment 保存但 `label_valid=0`（不再当作完整 0.5 s 标签）；DAgger `_backfill` 写 `terminal_failed`。旧数据仍不合格。
- **R5**：PPO contract schema 分为 `contract_version`/`obs_schema`；TF32 关闭；update 走 per-transition logp；`runs/ppo_v7_canary` 20 episodes/398 transitions，per-transition diff 0，batch 诊断 1.1–2.6e-5（独立 1e-4 诊断阈值）。
- **R6**：WM smoke 实际 **167,952** params（hidden 80/latent 128/1 layer），不是 245,792；planner+WM=1,221,094，略高于 1.2M，作为部署预算偏差记录。
- **R7**：static LD/Nav/ego target 仍是普通 MSE，没有 reprojection/re-encode/ego-dynamics；Stage E/F 不能开始。

**仍未关闭**：D1/D2 不是训练课程 gate；R7 static-target；`select_checkpoint.py` 的 LQR 分支和测试；Test500 未跑；terminalfix2 的 BC/DAgger 重训结果尚未作为 gate。
