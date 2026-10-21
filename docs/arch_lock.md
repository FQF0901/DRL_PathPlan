# MetaDrive 端到端自动驾驶 IL + DAgger + RL 实施设计合同

> **文档用途**：供 Coding Agent 直接实施。除非实验结果明确推翻，本文中的模块边界、数据合同、因果顺序、训练阶段、冻结策略、评测口径和工程约定均视为第一版基线约束。
>
> **核心目标**：基于 MetaDrive 构建约 100 万参数的结构化端到端自动驾驶系统，使用 OD、LD、道路级导航和自车动力学输入，依次完成 IL、DAgger、真实环境 RL，再引入 ego-conditioned latent world model、imagined rollout 与 LQR，实现车道保持、障碍物响应、自主选道、路口和收费站通行，并在冻结验证集闭环指标上争取超过 IDM + Pure Pursuit。

> **核对说明（2026-10-21，复核 `639e958`）**：本轮代码 delta 为 `6de65f4..639e958`（6 个提交，13 个文件：代码/测试/工具 9 个、文档 4 个；约 +1766/−391 行）。`docs/arch_lock.md` 的合同正文未改，只更新批注。状态仍取三种：**已实现无偏差 / 已实现有偏差 / 未实现**；“有偏差”包含部分实现、合同降级或证据不足。对“非目标/禁止”类条款，`已实现无偏差` 表示**禁止项未被违反**（功能本身可以不存在），`已实现有偏差` 表示存在部分违反，`未实现` 仅用于条款要求的机制/产物完全不存在。逐条批复见每条后的【】标记；代码块、公式、目录树等无法内联的位置，在块后以 `> 批复` 汇总。
>
> **本轮复核摘要（`639e958` vs `6de65f4`）**：
> - **已修复/已落地**：Nav 历史 xy 的 ×100→SE(2)→÷100 补偿；eval/PPO 改用段首 `v0/h0`；exact 由 PSTL κ 合成 normalized steering；`train_bc` 加载 `feasible/label_pass/action_in_bounds/label_source`；LD stable ID 改为 `road pair+lane_id`（`chunk_s` 仅 anchor）；`OD_ACC_DT=0.1`；Val500/Test500 spec + manifest；`82 passed, 1 skipped`；参数 `1,053,142`。
> - **仍存 P0/P1 偏差**：eval/PPO 的 `prev_action` 仍是 resolved requested，不是 measured realized（exact 数值等价，LQR 不等）；无 collect/eval/PPO parity test；PSTL short-path κ 门漏洞仍在；failed expert 仍整段 0.2 权重，非 segment-level；`privileged_expert_sample` 全 1、`multi_solution` 无；PPO `cost_adv` 不进 policy、canary 仅 4 transitions 且用旧 checkpoint；exact A/B 的 success +1.2 pp（95% CI −2.2..+4.6）跨 0，legacy harness/provenance 不对称；LQR clean 34/2/12/52 仍阻塞；Test500 只是 `split="val"` 的 spec，未接入、未运行；checkpoint 仍按 open-loop val loss；YAML/checkpoint schema 未校验。
> - **文档一致性**：`docs/implementation_status.md` 仍自相矛盾（:63 说 exact A/B passes，:166 说 fails；:116 仍是 26%/64% 旧数；:154 的 LQR 复现用 `scenarios_val_slice50.json`，但 artifact 用 `scenarios_val500.json`；PPO canary artifact 用 pre-P0 `bc_v2` checkpoint）。
> - **结论**：P0-1/P0-3 已修；P0-2 仅 exact 路径等价，LQR/parity test 残留；P0-4 消费已接，数据语义仍不完整。P1/P2 gate 仍未通过，LQR/Test500 不放行，P4 不动。

## 1. 目标、范围与非目标

### 1.1 最终目标

系统需要在 MetaDrive 闭环环境中完成：

- 官方案例和自定义场景中的稳定车道保持；　——【已实现有偏差：Val500 exact 66.2%；LQR 34%/52% timeout；exact A/B 仅 collision/off-road 显著，success CI 跨 0；Test500 未跑】
- 根据车辆、锥桶、收费站等 OD 做减速、停车、绕行；　——【已实现有偏差：OD/锥桶/收费站字段有；锥桶/收费站闭环未专项验证】
- 在仅有道路级导航、没有车道级 oracle 的情况下自主选道；　——【已实现有偏差：current-frame Nav 已去 lane-level；Nav 历史单位已修；无选道专项评测】
- 通过分岔、匝道、路口、收费站等拓扑复杂区域；　——【未实现：无分场景闭环证据】
- 保持安全性、平顺性、效率和路线达成率；　——【已实现有偏差：success/collision/off-road/RC/tracking/saturation 有；wrong-route 0；stuck 2/500；comfort 100% 违反】
- 最终在冻结验证集的 controller-realized closed-loop 指标上超过 IDM + Pure Pursuit 基线。　——【未实现：Test500 未跑、无多 seed】

### 1.2 第一阶段目标

第一阶段暂不上 latent world model，以 exact tracking 为 Planner 主诊断，同时从早期并行运行 LQR canary，重点证明：

- 输入 schema 正确；　——【已实现有偏差：schema v2 + stable ID/taxonomy/anchor；LD stable ID/OD acc 已修；仍缺 content latent、boundary polyline、lane type、traffic light】
- 道路级导航真的足以支持自主选道；　——【已实现有偏差：current-frame 无 lane API；无选道专项评测；Nav 历史单位已修】
- Plan Head 能生成闭环可用轨迹；　——【已实现有偏差：v_next 由 PSTL 精确执行；exact 66.2%；LQR 34%/52% timeout】
- BC、DAgger 以及短 Real-ENV PPO canary 的训练链路可运行；　——【已实现有偏差：三条链路跑通；PPO sampled log-prob 正确，但 cost 不进 policy、mask 25%、仅 4 transitions】
- 模型可以闭环恢复，而不只是 open-loop 拟合专家。　——【未实现：无 recovery 指标】

### 1.3 第二阶段目标

在第一阶段 Plan Head 闭环能力达标后：

- 加入真实历史 Memory Bank；　——【未实现】
- 加入 ego-conditioned deterministic latent world model；　——【未实现】
- 建立 Imagine Memory Bank；　——【未实现】
- 自回归 rollout 6 次，每次 0.5 s，形成未来 3 s 轨迹；　——【未实现】
- 将 6 个轨迹点插值至 0.1 s，交给 LQR 预瞄跟踪；　——【未实现】
- 联合优化 World Model、Plan Head、Value/Risk Head；　——【未实现】
- 在 Plan+WM+LQR 闭环成立后进行正式 Real-ENV PPO，争取超过 Expert；　——【未实现】
- 最后再评估 imagined RL 或 real-env-to-WM distillation。　——【未实现】

### 1.4 明确非目标

第一版不做：

- 图像输入或视觉 Backbone；　——【已实现无偏差：非目标，未引入视觉输入】
- 多候选轨迹 K-sampling；　——【已实现无偏差：非目标，未引入 K-sampling】
- 多模态生成式 World Model；　——【已实现无偏差：非目标，未引入生成式 WM】
- Diffusion World Model；　——【已实现无偏差：非目标，未引入 Diffusion WM】
- 车道级导航或外置推荐车道；　——【已实现无偏差：current-frame Nav 已去 lane-level；无推荐车道】
- 由规则模块替网络完成收费站选道、路口选道或绕障；　——【已实现有偏差：nearest-OD/min-TTC 已从 plan 路径移除；q_od 仍用规则 TTC 做 aux 监督】
- 用 open-loop ADE/FDE 替代闭环性能判断；　——【已实现无偏差：非目标，以闭环为主】
- 一次性将全部模块同时上线。　——【已实现有偏差：6 帧时序+DAgger+PPO 已提前；P4 未启动】

### 1.5 统一时间合同

```text
env/control_dt = 0.1 s
policy_dt = 0.5 s
memory_dt = 0.5 s
wm_dt = 0.5 s
rollout_steps = 6
rollout_horizon = 3.0 s
```

> **批复：已实现有偏差** —— control_dt=0.1、policy_dt=0.5、history=6 已落实；`memory_dt/wm_dt` 无实现；六步 preview 仍是同一 batch 重复，但已明确标为 diagnostic/placeholder，不能当 3 s WM 证据（`pipeline/rollout.py:78-96`）。

每个真实 policy decision 产生一个 0.5 s 轨迹段，由 Exact Tracker 或 LQR 在内部执行 5 个 0.1 s 控制步；执行完该段后再读取真实 observation 并重新决策。禁止训练阶段使用 0.1 s 重规划、部署阶段改成 0.5 s 重规划。六步 imagined rollout 仅用于 3 s preview，不代表真实环境开环执行 3 s。　——【已实现有偏差：段内 5 步已实现；六步 imagined 未实现，plan_actions 重复同 batch】

---

## 2. 总体架构

### 2.1 阶段一：无 World Model 的闭环 Planner

```text
MetaDrive structured observation
    ├── OD: vehicles / cones / toll facilities / obstacles
    ├── LD: lanes / boundaries / junctions / ramps / toll geometry / speed limits
    ├── Road-level Nav: road graph route, no target lane
    └── Ego: kinematics and previous executed control
            ↓
Coordinate normalization + ego-motion compensation where applicable
            ↓
OD / LD / Nav / Ego Encoders
            ↓
Current-frame spatial fusion
or real-history temporal fusion when enabled
            ↓
Plan Head: future ego trajectory representation
Value/Risk Head: progress, safety and return estimates
            ↓
Exact Tracking in early phase
            ↓
MetaDrive closed loop
```

> **批复：已实现有偏差** —— Encoders/Spatial/Plan/Value-Risk 头均有；value/risk/cost 头已训练但目标粗；6 帧时序已提前加入（A2 特性），P1 不再是单帧。

### 2.2 阶段二：带 ego-conditioned latent World Model 的自回归规划

```text
Real Environment
    ↓
Encoders → per-entity token records → Real Memory Bank, six frames including current
    ↓ StateFusion reads Bank and computes fused context on demand
    ↓ copy token records at rollout t0
Imagine Memory Bank M0

M0 → Plan Head → ego motion a0 → Ego Dynamics Transform → Latent WM → predicted z1 → M1
M1 → Plan Head → ego motion a1 → Ego Dynamics Transform → Latent WM → predicted z2 → M2
M2 → Plan Head → ego motion a2 → Ego Dynamics Transform → Latent WM → predicted z3 → M3
M3 → Plan Head → ego motion a3 → Ego Dynamics Transform → Latent WM → predicted z4 → M4
M4 → Plan Head → ego motion a4 → Ego Dynamics Transform → Latent WM → predicted z5 → M5
M5 → Plan Head → ego motion a5 → Ego Dynamics Transform → Latent WM → predicted z6

Collect a0...a5
    ↓
Transform all points to rollout-t0 ego frame
    ↓
3 s trajectory with 0.5 s spacing
    ↓
Interpolation and feasibility processing to 0.1 s
    ↓
LQR preview tracking
    ↓
MetaDrive
```

> **批复：未实现** —— 无 Real/Imagine Memory Bank、无 Latent WM、无自回归 rollout；`plan_actions` 用同一 context 重复 6 次。

### 2.3 因果约定

每个 imagined step 的因果关系必须是：　——【未实现：无 WM/Imagine Bank】

\[
a_n = \pi(M_n)
\]

\[
\hat z_{n+1}=W(M_n,a_n)
\]

\[
M_{n+1}=\operatorname{FIFO}(M_n,\hat z_{n+1})
\]

即：

```text
当前 imagined environment
→ 当前 ego action/motion
→ 下一时刻 environment
→ 下一步 ego action/motion
```

> **批复：未实现** —— 无 WM，因果链无法成立；当前 6 步查询不改变 context。

禁止实现成“先无条件预测下一环境，再决定同一步动作”。World Model 必须显式接受 ego motion condition。　——【未实现：无 WM；当前 6 步为同一 batch 重复查询】

---

## 3. 输入数据合同

## 3.1 Object Data，OD

每个 OD token 至少包含编码前的结构化字段：

- stable object/track ID，仅用于对齐、监督和诊断；　——【已实现无偏差：od_ids 稳定 ID 存在，按 ID 对齐】
- object type：车辆、行人、锥桶、障碍物、收费站动态设施等；　——【已实现有偏差：车辆/锥桶/建筑；行人/barrier 被过滤】
- taxonomy fields：`semantic_type / mobility_type(dynamic|static) / geometry_type / state_type`；　——【已实现无偏差：semantic/mobility/geometry/state 四元组存在】
- 相对 ego 的位置、航向、速度、加速度；　——【已实现无偏差：OD_ACC_DT=0.1 修正 5×；acc_x/acc_y 为 raw m/s²，无归一化/测试】
- object size；　——【已实现无偏差：L/W 入特征】
- visibility/validity；　——【已实现有偏差：presence flag；age 只是 6 帧窗口停留时间】
- observation age；　——【已实现有偏差：od_age 存在，但只在本窗口累加，非跨窗口 observation age】
- 可选的 turn signal、brake state 等 MetaDrive 可稳定提供的信息。　——【未实现】

原则：

- Planner 不直接读取裸 ID 做策略逻辑；　——【已实现无偏差：ID 只做对齐/监督，不进 plan】
- ID 用于跨帧对齐和监督；　——【已实现无偏差：same-ID 对齐成立】
- 所有环境语义字段先进入 OD Encoder；　——【已实现有偏差：OD/LD 走 encoder；q_od 的 TTC aux 仍由规则生成】
- 不允许以规则形式直接告诉 Planner “哪辆车需要避让”。　——【已实现有偏差：nearest-OD/min-TTC 已移除；q_od 仍受 TTC aux 监督】
- 静态设施不得进入动态 WM transition；动态闸机/交通灯状态进入 RoadState token。　——【未实现：无 WM；RoadState token 未实现】

## 3.2 Lane/Road Data，LD

LD 包括：

- lane centerline polyline；　——【已实现无偏差：5 点 polyline】
- lane boundary polyline 和 boundary type；　——【已实现有偏差：line type id 有；boundary polyline 无】
- predecessor、successor、left、right topology；　——【已实现有偏差：has_left/right/prev/next flag 有；ID 拓扑无】
- road segment ID；　——【已实现无偏差：ld_road_ids/nav_road_ids 有】
- lane ID、boundary ID 或稳定 polyline ID；　——【已实现有偏差：stable ld_ids 有；boundary ID 无】
- lane type、road type；　——【已实现有偏差：road type 有；lane type 无】
- junction、ramp、fork、merge；　——【已实现有偏差：is_junction/road type 有；ramp/fork/merge 显式标签无】
- toll lane/toll booth static geometry；　——【已实现有偏差：toll 作为 OD/road type；LD 几何未做】
- speed limit；　——【已实现无偏差】
- traffic light 等动态道路状态应单列为 dynamic road-state token。　——【未实现】

LD 对齐不能只靠 lane ID。如果同一 lane 每帧被局部裁剪，必须增加：

- world-coordinate anchor；或　——【已实现有偏差：ld_chunk_s/anchor 有；chunk_s 不再写入 stable ID；history 可跨 20 m】
- 固定弧长 segment index；或　——【已实现无偏差：chunk_s 有；仅作 anchor，不再参与 stable ID】
- 可复现的 polyline chunk ID。　——【已实现有偏差：ld_ids=hash(scene_seed,road_pair,lane_id)，不是 chunk ID；chunk_s 仅 anchor；history 可跨 20 m chunk；无跨 chunk 测试/1-2-4-6 ablation】

否则同一 lane ID 的 latent delta 会混入裁剪窗口变化。　——【已实现有偏差：ID 对齐有；LD 改用 lane 级 stable ID；chunk_s 仅 anchor；仍无 content delta/Δt】

## 3.3 Road-level Navigation

允许输入：

- 当前 road segment；　——【已实现无偏差】
- 后续若干 road segments；　——【已实现无偏差：NAV_ROADS=4】
- road graph topology；　——【已实现有偏差：road graph degree/type 有；显式 lane-to-road 拓扑只有 ld_road_ids】
- destination relation；　——【已实现无偏差】
- route progress；　——【已实现无偏差】
- 下一道路在拓扑上的方向关系。　——【已实现有偏差：road entry/exit heading 有；不是 lane heading】

严禁输入：

- target lane ID；　——【已实现无偏差：未提供】
- recommended lane；　——【已实现无偏差：later_middle/get_checkpoints 已移除】
- lane-change command；　——【已实现无偏差：未提供】
- 专家使用的 lane-level route；　——【已实现无偏差：current_ref_lanes/next_ref_lanes 未读】
- 专家参考中心线；　——【已实现有偏差：未读专家 centerline；但 Nav 的 entry/mid/exit 由道路所有 lane 端点聚合，奇数车道时数值上等于中间车道中心线】
- 由规则模块提前算出的收费站入口选择。　——【已实现无偏差：未提供】

允许网络看到 lane-to-road topology，因为这是环境事实；不允许直接告诉网络应选择哪条 lane。　——【已实现有偏差：ld_road_ids 有；Nav 不含 target lane】

## 3.4 Ego Kinematics

至少包括：

- longitudinal/lateral velocity；　——【已实现无偏差：v_long/v_lat 有】
- longitudinal/lateral acceleration；　——【已实现有偏差：a_lat=v·yaw_rate 近似；eval/PPO 已改段首 v0/h0，a_long/yaw 非零；LQR 的 prev_action 仍非实测】
- yaw、yaw rate；　——【已实现无偏差】
- steering angle 或等价状态；　——【已实现有偏差：exact 写 PSTL κ 合成 normalized steering；测试只验非零/有界，未验分布；LQR 不用此路径】
- vehicle length、width、wheelbase；　——【已实现有偏差：L/W 有；wheelbase 无】
- previous executed action/control；　——【已实现有偏差：eval/PPO 用 executed_action；collect.py 用 measured action；exact 下数值等价，LQR 下不等】
- 当前 pose 只用于坐标变换，网络输入使用规范化相对表示。　——【已实现无偏差】

## 3.5 控制元数据与语义信息边界

以下信息不属于驾驶策略，不应强制编码后再使用：

- scene reset signal；　——【已实现有偏差：env.reset 有；无显式 signal 入 obs】
- valid/padding mask；　——【已实现有偏差：mask 有；语义只在局部 docstring】
- timestamp 和 delta_t；　——【已实现有偏差：timestamp/dt 在 meta；未入模型】
- REAL/IMAGINED source flag；　——【已实现有偏差：meta.source 有；无 IMAGINED 张量】
- rollout depth；　——【未实现】
- batch/sample index。　——【已实现无偏差】

这些是张量管理、因果和安全正确性所需的控制元数据，可作为 mask、索引或控制信号直接使用。　——【已实现有偏差：mask 直接用；timestamp/source 有，depth 无】

所有环境语义信息必须经过 Encoder 后进入下游；不得以显式规则旁路计划网络。　——【已实现有偏差：OD/LD 走 encoder；TTC/gap aux 仍是规则】

---

## 4. 坐标系、对齐与时序表示

## 4.1 Ego-motion compensation

所有历史 OD/LD 在时序融合前必须对齐到当前时刻 ego frame，但**坐标补偿作用于 transform metadata / geometry anchor，不直接作用于普通 learned content latent**。普通 MLP、GNN 或 Attention 产生的 content latent 默认不具备可解释的 SE(2) 变换规则，禁止把 latent 向量当作二维坐标直接旋转或平移。　——【已实现有偏差：OD/LD 几何补偿正确；Nav 历史 xy 单位已修；无 content latent 拆分】

每个 OD/LD token 明确拆为：

```text
content latent
+ transform metadata / geometry anchor
+ stable ID / timestamp / valid mask
```

> **批复：已实现有偏差** —— 有 per-entity stable ID/geometry anchor；timestamp 在 meta，非 per-entity；无 content latent 拆分/transform metadata。

其中：

- `content latent`：类别、尺寸、限速语义、拓扑语义、局部运动语义等 Encoder 表征；　——【未实现】
- `geometry anchor`：OD 的中心点/朝向/包围盒锚点，LD 的 polyline control points/切向/边界锚点；　——【已实现有偏差：OD/LD anchor 有；无 transform metadata】
- `transform metadata`：写入 Bank 时的 ego pose、坐标系标识、时间戳，以及完成重投影所需的刚体变换信息；　——【未实现】
- Nav 不做普通动力学预测；道路级路由作为静态或拓扑条件保留，仅在路段切换、路由进度变化或仿真器真实更新时刷新。　——【已实现无偏差】

几何补偿原则公式：

\[
p_{i,t-k}^{ego_t}=T_{world\rightarrow ego_t}T_{ego_{t-k}\rightarrow world}p_{i,t-k}^{ego_{t-k}}
\]

读取历史槽位时执行：

```text
historical geometry anchor
    ↓ SE(2) compensate to current ego frame
shared geometry/pose adapter
    ↓
compensated geometry feature
    + detached historical content latent
    + current content latent
    ↓
same-ID temporal fusion
```

> **批复：已实现有偏差** —— OD/LD 已按 stable ID 对齐；Nav 历史 xy 单位已修；LD stable ID 已改 road pair+lane_id；但无 content latent/delta/Δt。

这样将自车平移和旋转从环境变化中解耦，同时避免假设任意 latent 天然可变换。允许计算 content latent residual，但它只是学习特征差，不承担坐标补偿职责。　——【已实现有偏差：OD stable ID/ID 对齐有；Nav 历史单位已修；无 latent delta/Δt】

必须写独立单元测试，覆盖：

- 纯直行、纯旋转、同时平移旋转；　——【已实现有偏差：直线/旋转有；组合平移+旋转无】
- 静态锥桶、收费站或车道点经补偿后在当前 ego frame 的位置正确；　——【未实现】
- 动态车辆的补偿后位移与相对速度方向一致；　——【未实现】
- 同一实体改变全局原点或全局朝向后，Planner 输入的相对几何保持一致；　——【未实现】
- content latent 不进入手写 SE(2) 旋转/平移代码；　——【已实现无偏差：禁止项满足（无 content latent）】
- Nav 未被当作普通动态实体进行未来状态外推。　——【已实现无偏差】


## 4.2 按稳定 ID 对齐后构造时序特征

不使用“所有 OD/LD 压成单个全局 latent 后相减”的方案。采用实体级稳定 ID 对齐，并把内容变化与几何变化分开表示：　——【已实现有偏差：same-ID 对齐有；LD stable ID 已修；无 content residual/显式 geometry delta/Δt】

```text
current content latent z_content(i,t)
historical content latent z_content(i,t-k)
compensated historical geometry g(i,t-k→t)
current geometry g(i,t)
    ↓ stable-ID alignment
current absolute state
+ content latent residual
+ explicit geometry delta
+ delta_t / masks
```

> **批复：已实现有偏差** —— same-ID 对齐和 SE(2) 补偿有；Nav/LD ID 已修；但无 content delta/Δt，且无 1/2/4/6 帧 ablation。

建议实体时序输入：

\[
h_{i,t-k}=\left[z^c_{i,t},\ z^c_{i,t-k},\ z^c_{i,t}-z^c_{i,t-k},\ g_{i,t},\ g_{i,t-k\rightarrow t},\ g_{i,t}-g_{i,t-k\rightarrow t},\ \Delta t,\ m_{i,t-k}\right]
\]

> **批复：已实现有偏差** —— 有 same-ID 对齐与显式几何；但无 content latent residual/Δt；Nav 历史单位已修。

其中：

- \(z^c\)：content latent，不接受手写坐标变换；　——【未实现】
- \(g\)：由 geometry anchor 和 transform metadata 构造的当前 ego frame 几何表征；　——【已实现有偏差：几何 anchor 有；无 transform metadata】
- \(z^c_{i,t}-z^c_{i,t-k}\)：可选 content residual，用于表达语义或运动状态变化；　——【未实现】
- \(g_{i,t}-g_{i,t-k\rightarrow t}\)：显式几何变化，负责表达补偿后的实体相对运动；　——【已实现有偏差：有补偿后几何，无显式 delta 特征】
- \(m\)：valid/birth/death/source mask。　——【已实现有偏差：presence mask 有；birth/death/source 无】

基线可先关闭 content residual，仅使用“当前 content latent + 历史 content latent + 显式几何变化”，再通过 ablation 判断 latent residual 是否提供稳定闭环收益。不得只输入 delta，否则会丢失绝对距离、类别、尺寸、基准速度和道路位置。　——【未实现】


## 4.3 实体 birth/death

必须区分：

- 当前存在、历史存在：正常 delta；　——【已实现有偏差：presence mask 有；无显式 delta】
- 当前存在、历史不存在：birth mask，不得用零 latent 假装历史实体；　——【已实现有偏差：mask 可隐式推出；无 birth mask】
- 当前不存在、历史存在：作为 death/exit 状态供 WM 监督或诊断，是否保留给 Planner 由配置决定；　——【已实现有偏差：death 实体丢弃；无 death/source mask】
- 当前和历史都不存在：padding mask。　——【已实现有偏差：presence/warm-up mask 有；实体身份靠 ID】

禁止把缺失实体直接填零后参与普通减法并当作真实运动。　——【已实现有偏差：缺失填零+mask；ID 对齐后不会把换人当运动】

## 4.4 时空融合顺序

基线采用：

```text
Same-ID entity temporal fusion
    ↓
Current-time entity representations
    ↓
OD↔OD self-attention
LD↔LD self-attention
OD↔LD cross-attention
Nav/Ego-conditioned cross-attention
    ↓
Plan / Value / World Model
```

> **批复：已实现有偏差** —— 先时序后空间成立；空间仍是单个 joint Transformer，不是合同结构；LD stable ID 已修。

即先做实体内时序，再做当前时刻实体间空间交互。　——【已实现有偏差：先时序后空间；空间不是规定结构】

不采用所有 6 帧全部 OD/LD token 的全量时空 self-attention，以控制显存、参数和调试难度。　——【已实现无偏差】

---

## 5. Memory Bank 设计

## 5.1 Real Memory Bank

- 容量：6 帧；　——【已实现有偏差：6 帧 list；无 Bank 类】
- 帧间隔：0.5 s；　——【已实现无偏差】
- 覆盖历史：固定保存 \(t-2.5\) s 至 \(t\) 共 6 帧，**包含当前帧**；dataset、online rollout 和 test 必须使用完全相同的索引语义；　——【已实现有偏差：在线 episode 内成立；dataset/online/test 语义未完全统一】
- episode 初始不足 6 帧时可复制最早真实帧完成定形，但复制槽位必须 `valid=0`，不得伪装成真实历史；　——【已实现有偏差：hist_valid=0；per-entity mask=0】
- 场景切换必须 hard reset；　——【已实现有偏差：每 episode 新建 env；无 Bank reset】
- Bank 只接收真实 MetaDrive ENV 观测经 Encoder 得到的 per-entity token record，不写入 imagined state，也不存储 StateFusion 之后的全局 fused context；　——【未实现：保存原始 obs，不是 per-entity token record】
- 写入 content latent 前默认 detach，避免计算图跨真实环境 step 无限增长；　——【未实现：无 Bank】
- 同一训练 batch 若需要 BPTT，使用显式 sequence tensor/unroll，不依赖长期在线 Bank 的历史计算图。　——【未实现】

每帧必须存储：

```text
OD:
  content_latent
  geometry_anchor
  transform_metadata
  stable_id / semantic_type / valid_mask

LD:
  content_latent
  polyline_or_boundary_anchor
  transform_metadata
  stable_id / topology_id / valid_mask

Nav:
  content_latent
  route_segment_id / route_progress / valid_mask

Ego:
  content_latent
  explicit_kinematics
  ego_pose

Common:
  timestamp / scene_id / encoder_version / source=REAL
```

> **批复：未实现** —— 无 Bank/per-entity record；npz 存原始张量。

约束：

- `content_latent` 是主要学习表示；`geometry_anchor + transform_metadata` 是坐标补偿和 ID 对齐所需的最小结构化元数据，不属于外部规划规则或车道级导航特权；　——【未实现：模型只有 fused/entity latent，无 Bank】
- OD anchor 建议保留中心点、朝向、长宽和速度锚点；LD anchor 建议保留固定数量 polyline control points、切向和拓扑 ID；　——【已实现有偏差：OD/LD anchor 有；topology ID 有 ld_road_ids；无 content latent】
- Road-level Nav 只给道路/路由级意图，不给目标车道 ID、推荐变道方向或专家未来轨迹；　——【已实现有偏差：current-frame road-level；Nav 历史单位已修；无 lane-choice 专项】
- Ego 显式动力学用于 causal conditioning、几何补偿和诊断，不要求 WM 从 latent 中反推 ego pose；　——【已实现有偏差：12 维 ego；无 pose/尺寸完整链】
- 必须为 Bank 定义固定 shape、padding 和 mask，避免 Python list 导致训练/部署语义不一致。　——【已实现有偏差：历史张量固定 shape；无 Bank】


## 5.2 Imagine Memory Bank

- rollout t0 从 Real Memory Bank 复制；　——【未实现】
- 复制后不与 Real Memory Bank 共享可变写入；　——【未实现】
- 每次 World Model 预测下一帧 `content latent + predicted/propagated geometry anchor + transform metadata` 后执行 FIFO append；　——【未实现】
- rollout 结束后立即 reset；　——【未实现】
- 每个元素带 `source=REAL/IMAGINED` 和 `rollout_depth=0..6`；　——【未实现】
- 静态 LD/Nav 优先由 t0 真实结构和 imagined ego pose 做解析重投影/传播，不要求 WM 从零生成静态地图；动态 OD 与必要 road-state 由 WM 预测；　——【未实现】
- forbidden：imagined content latent、geometry anchor 或 metadata 回写 Real Memory Bank。　——【未实现】

## 5.3 缓存与 Encoder 版本

训练集、DAgger Buffer 和 PPO Replay/Rollout Buffer 优先保存**原始结构化 observation 与必要的 action/reward/mask**，训练取样时再用当前 Encoder 计算 latent，不把 latent 作为可跨多轮训练长期复用的唯一数据。原因是 Encoder 会随训练更新：同一帧 observation 经旧 Encoder 得到的 `z_old` 与经新 Encoder 得到的 `z_new` 可能处于不同表征基底；若把旧缓存 latent 与当前新算 latent 放入同一个 6 帧窗口，时序差分可能主要反映 Encoder 参数变化，而不是真实环境变化。　——【已实现无偏差：npz 存原始 obs+action】

这不影响 Real Memory Bank 的在线运行：单个 episode 内，Bank 保存最近 6 帧且写入时 detach；其生命周期短，允许使用当前参数快照的 Encoder 输出。训练参数更新发生在 batch/rollout 边界，不得在同一个 6 帧窗口构建过程中更新 Encoder。　——【已实现有偏差：采集时 encoder 冻结；无 Bank】

若为吞吐性能必须离线缓存 latent：

- 每条缓存记录 `encoder_version/hash`，对应生成 latent 的 Encoder checkpoint 或权重哈希；　——【未实现】
- 训练加载时只允许使用与当前 Encoder 版本一致的缓存，否则重新编码原始 observation 或使缓存失效；　——【未实现】
- 禁止在同一时序窗口、同一 Real/Imagine Bank 中混用不同版本 latent；　——【未实现】
- 若 Encoder 在某阶段完全冻结，可为该冻结阶段一次性重建 latent cache；Encoder 再次解冻后，旧 cache 自动失效；　——【未实现】
- 部署推理时 Encoder 权重固定，因此不存在跨训练版本漂移，但场景 reset 和 Bank 隔离规则仍须保持。　——【已实现有偏差：无 cache/Bank 可谈】

---

## 6. Plan Head 与轨迹定义

## 6.1 每次 policy query / imagined step 的输出

每次 Plan Head query 只输出下一 0.5 s ego motion；同一个 Plan Head 参数实例贯穿 BC、DAgger、PPO 和 WM rollout。真实环境每 0.5 s 决策一次，Exact Tracker 或 LQR 在段内执行 5 个 0.1 s 控制步；下一决策时刻使用新 observation 重规划。　——【已实现有偏差：同一 Plan Head 贯穿；真实环境只执行 a0，6 步 preview 是 placeholder】

第一版主动作定义为带终端速度边界的 SE(2) endpoint：

\[
a_n=(\Delta x_n,\Delta y_n,\Delta\psi_n,v_{n+1})
\]

> **批复：已实现有偏差：输出 4D，执行链已改为 4D PSTL；端点/终端速度精确；但 short-path κ 检查有洞。**

其中：
- `(Δx, Δy, Δψ)` 位于当前 real/imagined ego frame；　——【已实现无偏差】
- `v_{n+1}` 是 0.5 s 后目标速度，不得用 `distance/0.5` 的平均速度替代；　——【已实现无偏差】
- pose endpoint 决定空间边界，`v_{n+1}` 决定纵向时间边界，二者不是冗余量；　——【已实现无偏差：PSTL 精确执行 v_next】
- 可等价改用 `(Δx, Δy, Δψ, Δv)`，但必须在 P0 写入 decision log 后全流程统一；　——【已实现无偏差】
- `(ds,dtheta)` 仅保留为对照基线，不作为长期主动作。　——【已实现有偏差：生产路径 4D；tracking.py 仍保留 2D legacy 说明/入口】

P0 将 4D `(dx,dy,dpsi,v_next)` 作为固定基线，只校准标签、bounds、归一化、Hermite 插值和动力学可行性；不得根据 `v_next` 与平均速度的相关性自动退化为 3D。若 action round-trip、full-stop/stop-to-go 或 Expert-GT LQR 出现结构性失败，则暂停 P1，提交 decision log 人工复审，不允许脚本静默改动作维度。至少报告 endpoint/heading/terminal-speed、conditional terminal-speed residual、acceleration、jerk、curvature/curvature-rate、full-stop、stop-to-go、LQR tracking error 和控制饱和率。　——【已实现有偏差：PSTL bounds/feasibility 有；P0 否决门（Expert-GT LQR/roundtrip/full-stop）未完整】

必须从相邻点和连续 0.5 s 决策段派生并约束：
- heading / `Δψ`；　——【已实现有偏差：heading loss 有；无跨段 consistency】
- curvature / `κ`；　——【已实现有偏差：κ hinge 有；代理量】
- longitudinal/lateral acceleration；　——【已实现有偏差：a_long 有；无真实轨迹约束】
- jerk；　——【已实现有偏差：jerk proxy 有；非真实 PSTL jerk】
- curvature rate；　——【已实现有偏差：proxy 有】
- 相邻真实决策段连接处的位姿、速度、航向和曲率连续性。　——【未实现】
- 第 `n+1` 段起始速度硬继承 Ego Dynamics 推进后的第 `n` 段终态 `v_{n+1}`；平顺性使用 soft loss + feasibility gate。　——【已实现有偏差：PSTL 终端速度硬继承；跨段 soft loss 无】

Exact Tracking 只替代早期执行器，不豁免轨迹动力学约束。即使仿真器能够瞬时落到目标点，也不得把横移过大、航向突变、曲率不连续或速度跳变的轨迹判为可用。　——【已实现有偏差：feasibility gate 有；short-path <0.25 m 跳过 κ/a_lat】

## 6.2 坐标累计

每个点来自不同 imagined ego frame，不能直接拼接。必须把每一步相对运动累计到 rollout-t0 ego frame：　——【已实现有偏差：PSTL 段内 5 点；无 6 步 t0 compose】

\[
P_{n+1}=P_n\oplus\Delta P_n
\]

最终生成：

\[
\tau=\{(x_i^{ego@t0},y_i^{ego@t0},v_i)\}_{i=1}^{6}
\]

> **批复：未实现：无 6 点 t0 带速度轨迹。**

单元测试必须验证：

- 六次直行累计后位置正确；　——【已实现有偏差：单段精确；6 步累计无】
- 连续转弯时坐标旋转正确；　——【未实现】
- 输出点都在同一个 t0 坐标系；　——【未实现】
- 轨迹转换到 world frame 后和 ego dynamics rollout 一致。　——【未实现】

## 6.3 插值与 LQR 输入

6 个点对应未来 0.5、1.0、1.5、2.0、2.5、3.0 s。进入 LQR 前：

- 插值为 0.1 s；　——【已实现有偏差：PSTL 5×0.1 s；只执行第一段】
- 推荐使用带切线约束的 cubic Hermite 或其他不会明显过冲的插值；切向方向由 heading 决定，切向大小由速度边界决定，禁止仅凭 `Δψ` 任意设定；　——【已实现有偏差：PSTL 几何切线用 chord，速度边界在时间律；已在 status 记录，待合同 owner 确认】
- 速度使用 `v_n → v_{n+1}` 做单调/限幅插值；　——【已实现无偏差：cubic time law 端点速度精确；单调性有检查】
- 计算 heading、curvature、acceleration、jerk；　——【已实现有偏差：PSTL 计算 heading/κ/a/jerk；训练 loss 仍用代理】
- 对不连续曲率、逆行、速度负值、超动力学限制做 reject 或修正；　——【已实现有偏差：gate/project 有；short-path 洞】
- 保留完整 2 至 3 s 参考轨迹供 LQR 预瞄，不得只传第一个 0.5 s。　——【已实现有偏差：LQR 只吃 0.5 s；3 s 未接】

## 6.4 Action semantics 一致性

BC、DAgger、RL、World Model rollout 必须使用同一 Plan Head 输出语义。禁止出现：　——【已实现有偏差：BC/DAgger/PPO 4D；tracking.py 2D legacy 仍活】

- BC 输出轨迹点，RL 改为 steering/throttle；　——【已实现无偏差：全部 4D】
- DAgger 用 \(x,y,v\)，WM 阶段改成 \(ds,d\theta\)；　——【已实现无偏差：禁止项未被违反（DAgger 用 4D；WM 不存在）】
- train 和 inference 使用不同坐标系。　——【已实现有偏差：4D 语义一致；eval ego 历史 parity bug】

MetaDrive 原生 steering/throttle 仅作为下游控制执行接口，不作为 Planner 的主要动作定义。　——【已实现无偏差】

### 6.5 PPO 概率动作合同

PPO 要求可计算 `log_prob(action | state)`。因此 Plan Head 不是只输出确定性三元组，而是输出受约束轨迹动作分布：　——【已实现有偏差：TruncatedNormal 有；但分布只受静态 box 约束，状态相关可行性靠 mask】
- 对归一化 `(dx, dy, dpsi, v_next)` 输出 `mu, log_std`，或使用 squashed Gaussian 后映射到动力学边界；　——【已实现有偏差：TruncatedNormal 有；但分布只受静态 box 约束，状态相关可行性靠 mask】
- BC/DAgger 使用 expert action 的 NLL，并保留 point/velocity/feasibility auxiliary loss；　——【已实现无偏差：NLL + point/velocity/feasibility aux 有】
- PPO buffer 保存 sampled pre-squash action、executed feasible action、old log-prob、old value 与修正/裁剪标记；　——【已实现有偏差：buffer 有 sampled/executed/fallback/mask；无 clip mask 与 executed log-prob 绑定】
- 若 feasibility layer 改写了动作，必须避免把“采样动作的 log-prob”错误地对应到“被大幅改写后的执行动作”。基线应让分布本身落在可行域，并统计 post-process rate；　——【已实现有偏差：fallback 被 mask 出 policy loss；post-process rate 有统计】
- Value Head 输入 state/context，不输入未来 GT；Risk/Cost Value 与 reward Value 分头训练。　——【已实现有偏差：value/risk/cost 分头；risk 5 类；cost 不进 policy】
正式 PPO 中：当前 `a0` 从策略分布采样并记录 `log_prob`，执行完整 0.5 s 后形成一个 transition；`a1...a5` 使用各 imagined step 分布均值，仅作为非执行 preview，不计入当前 `old_log_prob`，下一真实决策时刻重新采样新的 `a0`。不得把 preview tail 对真实控制造成的影响算入一个未覆盖它的 log-prob。　——【已实现有偏差：a0 采样执行；a1..a5 不计算；fallback 被 mask】


---

## 7. Ego Dynamics 与 Latent World Model

## 7.1 模块职责分离

Ego Dynamics：

- 判断或推进“自车能否以及如何从当前状态走到该轨迹段”；　——【已实现有偏差：PSTL time law 推进 ego；无独立 Ego Dynamics 模块】
- 计算 ego pose、heading、velocity、curvature 等下一状态；　——【已实现有偏差：PSTL/ExactTracker 给 pose/heading/v；无 WM condition】
- 为坐标变换和 World Model 提供 ego motion condition。　——【未实现：无 WM condition】

Latent World Model：

- 在给定当前 imagined context 和 ego motion 的情况下，预测周围环境下一帧 latent；　——【未实现】
- 不负责替代完整车辆动力学；　——【未实现】
- 不负责生成导航目标；　——【未实现】
- 不负责保证 Plan Head 轨迹可跟踪。　——【未实现】

LQR：

- 使用完整 3 s 插值轨迹进行预瞄跟踪；　——【已实现有偏差：LQR 只跟 0.5 s canary】
- 将参考轨迹转换为 MetaDrive 可执行控制。　——【已实现无偏差】

## 7.2 World Model 条件

WM 不应只看到 0.5 s 终点。条件至少包括从轨迹段确定性派生的：　——【已实现有偏差：PSTL 段内有 v_next；无 WM condition】

\[
c_n=(\Delta x,\Delta y,v_n,v_{n+1},\Delta\psi,\kappa,a_{lon})
\]

其中：

\[
a_{lon}=\frac{v_{n+1}-v_n}{0.5}
\]

> **批复：已实现有偏差** —— PSTL 段内用 v_next 构造时间律；未用于 WM condition。

这些量由 Plan Head 输出和当前 ego state 计算，不需要另开互相矛盾的动作头。　——【未实现：无独立 WM 动作头；PSTL 有 v_next】
每段至少 enforce：`v_next>=0`、`a_min <= (v_next-v_n)/0.5 <= a_max`、`|kappa|<=kappa_max`、`|v^2*kappa|<=a_lat_max`；低速 stop/hold 属于 Controller；记录 clip/reject/post-process 和控制饱和率。　——【已实现有偏差：PSTL gate/project 有 enforce/统计；无 WM】

## 7.3 World Model 输出拆分
分类合同：仅 dynamic OD/RoadState 进入 learned transition；Static OD/infrastructure/LD 做 geometry reprojection + shared Encoder re-encode；Nav 按 route progress 更新。交通灯 schema 保留，首版可默认关闭。　——【未实现】

动态预测主体：

- OD latent；　——【未实现】
- dynamic road-state latent，例如 traffic light；　——【未实现】
- 可选的 occupancy/risk latent；　——【未实现】
- uncertainty scalar/head。　——【未实现】

显式推进或重新编码：

- Ego latent：由 ego dynamics 推进后编码；　——【未实现】
- Static LD：依据 predicted ego pose 做坐标变换，可预测小 residual，但不应从零重建地图；　——【未实现】
- Road-level Nav：依据 route graph 和 predicted route progress 更新/重新编码，不做普通动力学幻想。　——【未实现】

## 7.4 确定性模型与不确定性

第一版采用 deterministic World Model，不做多模态未来或 K 候选轨迹。　——【未实现】

建议保留轻量 uncertainty head，用于：

- 识别 out-of-distribution imagined state；　——【未实现】
- 统计 rollout depth 1 至 6 的误差增长；　——【未实现】
- 在 imagined RL 中衰减或截断高不确定性样本；　——【未实现】
- 防止 Policy 利用 WM 明显错误。　——【未实现】

uncertainty head 不改变单轨迹主方案。　——【未实现】

---

## 8. Encoder 与网络规模建议

### 8.1 总参数约束

部署推理主网络目标约 100 万参数，允许工程上约 0.8M 至 1.2M 的可解释浮动。EMA Target Encoder 属于 training-only teacher copy，必须单独统计，不计入 deployable parameter budget，但其显存和 optimizer/activation 开销仍需报告。必须在每次结构修改后记录：　——【已实现有偏差：1,053,142 deployable；无 EMA target；无 latency/memory】

- deployable total params 与 training-only target params；　——【已实现有偏差：deployable total 有；target 无】
- trainable params；　——【已实现无偏差】
- peak allocated memory；　——【未实现】
- single-step latency；　——【未实现】
- six-step rollout latency。　——【未实现】

### 8.2 参考预算

- OD Encoder：约 60k 至 100k；　——【已实现有偏差：36.7k < 60k】
- LD/polyline Encoder：约 100k 至 150k；　——【已实现有偏差：18.5k < 100k】
- Nav Encoder：约 20k 至 40k；　——【已实现有偏差：47.0k > 40k】
- Ego Encoder：约 10k 至 30k；　——【已实现有偏差：34.7k > 30k】
- Temporal Fusion：约 120k 至 200k；　——【已实现有偏差：268k > 200k】
- Spatial/Cross Attention：约 150k 至 220k；　——【已实现有偏差：396.5k > 220k】
- Plan Head：约 80k 至 120k；　——【已实现无偏差：114.9k 在范围内】
- Value/Risk Head：约 30k 至 70k；　——【已实现有偏差：约 1.2k << 30k】
- Latent World Model：约 180k 至 260k。　——【未实现】

推荐起点：

- hidden dimension：96；必要时 128；　——【已实现无偏差：128】
- attention heads：4；　——【已实现无偏差】
- temporal layers：1 至 2；　——【已实现无偏差：每模态 1 层】
- spatial layers：1 至 2；　——【已实现无偏差：2】
- max OD tokens：16 至 32；　——【已实现无偏差：16】
- max LD/polyline tokens：24 至 48；　——【已实现有偏差：16 < 24】
- polyline 内先做 point MLP/PointNet pooling，再做实体间 attention。　——【已实现无偏差：point MLP + mean pooling】

### 8.3 初始化

- ReLU 类 MLP：Kaiming；　——【已实现无偏差：全 GELU，无 ReLU 分支】
- GELU/attention linear projection：Xavier；　——【已实现无偏差】
- LayerNorm：weight=1，bias=0；　——【已实现无偏差】
- 最终 trajectory residual/output head 可使用小方差初始化，但禁止整个分支双零初始化；　——【已实现无偏差：xavier gain 0.1；logstd bias -1】
- 每个模块必须提供初始化统计 probe，报告 weight mean/std 和输出分布。　——【未实现：param_probe 只计数，无初始化统计】

---

## 9. 监督目标与 Loss

## 9.1 Plan Loss

每个 rollout step 对 0.5 s motion 监督；六步后再对完整 3 s 轨迹做一致性监督。　——【已实现有偏差：0.5 s 多分项；无 3 s 一致性】

建议组成：

\[
L_{plan}=L_{point}+\lambda_vL_v+\lambda_hL_{heading}+\lambda_cL_{curvature}+\lambda_sL_{smooth}
\]

> **批复：已实现有偏差：有 NLL/point/heading/velocity/curvature/accel 分项；κ/jerk 是代理；无 3 s。**

其中：

- point loss：x/y 或 delta x/delta y；　——【已实现无偏差】
- velocity loss；　——【已实现无偏差】
- heading consistency；　——【已实现有偏差：heading MSE；无 consistency/wrap】
- curvature/radius feasibility；　——【已实现有偏差：κ hinge 代理】
- acceleration、jerk、curvature-rate smoothness。　——【已实现有偏差：a/jerk proxy】

不允许只用单一 XY MSE 作为全部轨迹监督。　——【已实现无偏差：多分项】

## 9.2 World Model Loss

真实下一帧 observation 经 Target Encoder 得到 latent GT：　——【未实现】

\[
z_{t+1}^{GT}=E_{target}(o_{t+1})
\]

\[
L_{wm}=D(\hat z_{t+1},\operatorname{sg}(z_{t+1}^{GT}))
\]

建议分模态报告：

\[
L_{wm}=\lambda_{od}L_{od}+\lambda_{roadstate}L_{roadstate}+\lambda_{ego}L_{ego}+\lambda_{aux}L_{aux}
\]

Static LD 和 Nav 不作为普通无条件预测主项。
Static LD 使用 geometry reprojection + stop-gradient re-encode consistency；Nav 使用 route-progress + geometry/topology consistency。

不能只报告 aggregate latent MSE。至少按：　——【未实现】

- modality；　——【未实现】
- entity type；　——【未实现】
- rollout depth；　——【未实现】
- distance bin；　——【未实现】
- static/dynamic；　——【未实现】
- seen/unseen map；　——【未实现】

拆分统计。

## 9.3 Value/Risk Supervision

Value Head 预测：

- expected discounted return；　——【已实现有偏差：value_coef=0.5；目标仅 ±γ^remaining】
- route completion probability；　——【未实现】
- time-to-goal 或 progress；　——【未实现】
- optional cost value。　——【已实现有偏差：cost_value 头有；目标事件计数】

Risk Head 至少预测：

- collision risk；　——【已实现无偏差：5 类 risk logit】
- off-road risk；　——【已实现无偏差：risk[1]】
- wrong-way/wrong-road risk；　——【已实现有偏差：risk[2]；标签 episode 广播，Val500 0 事件】
- stuck risk；　——【已实现有偏差：risk[3]；标签 episode 广播，Val500 0 事件】
- comfort violation risk。　——【已实现有偏差：risk[4]；plan-side 100% 违反】

建议 reward value 和 safety cost value 分开，避免成功奖励抵消安全成本。　——【已实现有偏差：分头；cost 不进 policy】

## 9.4 Representation 防坍缩

只做 latent regression 可能出现 Encoder 与 WM 共同缩小特征幅度。必须至少具备：

- EMA Target Encoder；　——【未实现】
- stop-gradient target；　——【未实现】
- latent variance/statistics monitor；　——【未实现】
- 每模态 decoder 或 auxiliary task 中至少一种；　——【未实现】
- planner task loss 持续约束 latent 的任务相关性。　——【未实现】

可选辅助任务：

- OD type/relative state reconstruction；　——【未实现】
- LD topology classification；　——【未实现】
- route-progress prediction；　——【未实现】
- ego-motion prediction；　——【未实现】
- next-state contrastive discrimination。　——【未实现】

---

## 10. Detach 与梯度合同

## 10.1 默认 detach 规则

应 detach：

- Future GT latent；　——【未实现】
- EMA Target Encoder 输出；　——【未实现】
- Real Memory Bank 跨真实 env step 的写入；　——【未实现】
- Replay/DAgger/PPO Buffer 中保存的数据；　——【已实现有偏差：npz/PPO detach；无显式合同测试】
- PPO old log-prob 和 old value；　——【已实现无偏差：detach；PPO audit 有】
- 场景采集阶段的 observation/action；　——【已实现无偏差：np 数组无图】
- 默认情况下，早期联合训练时 Plan action 进入 WM 前可 detach，防止 Plan Head 利用 WM 漏洞。　——【未实现】

不应 detach：

- 当前 batch 内 Encoder → Planner → Plan Loss；　——【已实现无偏差】
- 当前显式 WM unroll 内 WM recurrent path；　——【未实现】
- WM output → Value/Risk Head；　——【未实现】
- RL Policy Head → policy loss；　——【已实现无偏差】
- 允许联合调整时的 Encoder top layers/adapters。　——【未实现】

## 10.2 必须配置化的梯度开关

```yaml
detach_real_mem_write: true
detach_future_target: true
detach_wm_input: false
detach_wm_recurrent_path: false
detach_wm_to_value: false
detach_plan_action_to_wm: true
detach_encoder_from_wm_loss: true # D1/D2 固定 target、Stage E 才启用 EMA
use_ema_target_encoder: true  # D1/D2 固定 target、Stage E 才启用 EMA
```

> **批复：未实现** —— 无 YAML 梯度开关；`config/*.yaml` 不加载；PPO/BC 有若干硬编码 detach。

训练后期只有在 WM 通过六步 fidelity 门槛后，才允许对照实验：

```yaml
detach_plan_action_to_wm: false
detach_encoder_from_wm_loss: false
```

> **批复：未实现** —— 无 joint training/WM，开关无对应实现。

开放后必须监控：

- Policy 是否找到 WM 幻觉漏洞；
- imagined return 上升但 real-env return 下降；
- action 分布是否越过训练数据范围；
- latent norm 是否异常漂移。

> **批复：未实现** —— 无相关监控。

## 10.3 EMA Target Encoder

Target Encoder 不参与反向传播。采用：　——【未实现】
D1/D2 冻结 Encoder 时 target 固定，EMA 为 no-op；D2 增加 latent variance/effective-rank gate；Stage E 解冻 adapter 后才启用 EMA，并监控 target-online drift。

\[
\theta_{target}\leftarrow m\theta_{target}+(1-m)\theta_{online}
\]

> **批复：未实现** —— 无 target encoder/EMA。

建议起点：

- 初期 \(m=0.99\)；　——【未实现】
- 稳定后可增加至 0.995 或 0.999；　——【未实现】
- 具体通过 target drift 与 WM 收敛速度决定，不能只看 loss 更低。　——【未实现】

---

## 11. Expert、BC 与 DAgger

## 11.1 Expert 定义

第一版唯一可用 Expert 为 IDM + Pure Pursuit 或可复现的等价组合。接受其不是所有场景的强专家，也不把“缺少更强专家”作为阻塞项：IL/DAgger 的目标是让 Student 接近并吸收可用示范，最终依靠真实环境 RL 和闭环筛选争取超过 Expert。但必须：

- 记录 expert success、collision、off-road、wrong-route、stuck；　——【已实现有偏差：success/collision/off-road 有；wrong-route/stuck 在 expert 数据恒 0】
- 失败 expert episode 不得无条件作为正样本；　——【已实现有偏差：expert_plan_valid=success 整段广播；失败段乘 0.2（非 0）；timeout 早段同权；无 segment-level 规则】
- 保存 expert 是否使用 lane-level 特权信息的审计结果；　——【已实现有偏差：字段落盘；privileged 仍全 1；无 multi_solution】
- 将 expert 轨迹统一转换为 Student 的轨迹表示。　——【已实现有偏差：BC measured 4D；DAgger 30% measured + 69% lift】

## 11.2 特权信息审计

必须排查 MetaDrive Expert/IDM 路径：

- 是否读取 target lane；　——【已实现有偏差：字段有；审计是常量声明，不是逐样本证据】
- 是否读取 lane-level route；　——【已实现有偏差：同上】
- 是否读取学生不可见的未来交通状态；　——【已实现有偏差：used_future_state=0；无逐样本审计】
- Pure Pursuit 参考点是否来自 lane-level centerline；　——【已实现有偏差：used_centerline=1；字段是常量】
- expert lane-change decision 的信息来源。　——【已实现有偏差：target!=anchor 标记有区分度（0.7-1.1%）】

若 Expert 使用 lane-level oracle：

- 允许其继续作为数据生成器，不要求先替换 Expert；　——【已实现无偏差】
- 禁止将该 oracle、target lane 或由其派生的推荐方向输入 Student；　——【已实现无偏差：current-frame Nav 去 lane-level；无 target lane】
- 数据中标记 `privileged_expert_sample`，用于分桶评测和 loss weighting，不作为 Planner 语义输入；　——【已实现有偏差：字段有；每行仍 1；w_privileged 仅在混合 batch 生效，当前结构性无效】
- 对分岔、路口、收费站和多条合法车道场景，禁止仅用单一轨迹 XY-MSE 强迫复制专家所选车道；应降低单轨迹 L2 权重，并辅以合法道路区域、路线进度、轨迹可行性和安全性监督；　——【未实现：无 multi_solution flag/L2 降权】
- 不要求 BC 在不可辨识的多解场景逐点复现 Expert，重点检查是否选择了合法、可达且安全的解；　——【未实现：仍是逐点 MSE】
- 由 DAgger 补充 Student 访问状态下的恢复示范，由真实环境 RL 最终解除对专家车道选择和次优行为的绑定，并以冻结验证集闭环结果判断是否超越。　——【已实现有偏差：DAgger 3 轮；无 failure buffer/无解绑证据】

## 11.3 DAgger

每轮在 Student 访问到的状态上查询 Expert，并把恢复样本加入数据集。　——【已实现有偏差：3 个独立 round；无跨轮聚合】

参考混合系数：

```text
1.0 → 0.7 → 0.4 → 0.2 → 0.0
```

> **批复：已实现有偏差** —— 有 3 个独立 round；无逐轮混合系数/跨轮 buffer；train_bc 不加载 round。

但实际以闭环安全和数据覆盖决定。下降由 Student success、collision/off-road、Expert intervention、recovery、LQR shadow 和 failure-buffer 覆盖率触发，不按固定轮次机械下降。　——【未实现：无混合系数/触发条件】

重点采集：

- 偏离车道后的恢复；　——【已实现有偏差：有 DAgger 数据；无 bucket 权重】
- 错过换道窗口；　——【未实现】
- 锥桶邻近；　——【未实现】
- 慢车跟驰/绕行；　——【已实现有偏差：数据有；无 bucket】
- 路口犹豫；　——【未实现】
- 收费站入口选择；　——【未实现】
- 出路边缘状态；　——【未实现】
- 低速大曲率；　——【未实现：无最低配额】
- expert 与 student 决策分歧大但两者均未立即失败的状态。　——【已实现有偏差：intervened 标记约 0.5-0.7%】

必须使用 failure-prioritized buffer，不能只做均匀追加。　——【未实现：均匀拼接】

---

## 12. RL 设计

## 12.1 训练顺序

真实环境 RL 必须先于 imagined RL。MetaDrive 已提供真实 simulator，不应在 WM 尚未可靠时引入 model bias。　——【已实现有偏差：无 imagined RL；但 PPO 早于 A2/B 门禁】

第一版 RL 推荐 PPO，保持 Plan Head action semantics 不变。　——【已实现有偏差：采样/执行 4D；fallback 改写并 mask】

## 12.2 Reward 与 Cost

Reward 至少包含：

- route progress；　——【已实现有偏差：10×Δroute_completion】
- success；　——【已实现有偏差：+20】
- efficiency；　——【未实现】
- comfort；　——【已实现有偏差：−0.1 reward + cost；Val500 100% 违反成常数】
- lane/route compliance。　——【未实现】

Cost 至少包含：

- vehicle/object collision；　——【已实现有偏差：cost 向量 collision=1.0；不分 vehicle/object】
- off-road；　——【已实现有偏差：cost 向量 off_road=1.0】
- wrong-way/wrong-road；　——【已实现有偏差：cost 向量 wrong_route=1.0】
- severe overspeed；　——【已实现有偏差：0.2×exceedance】
- unsafe proximity；　——【未实现】
- stuck；　——【已实现有偏差：cost 向量 stuck=1.0】
- excessive lateral acceleration/jerk。　——【未实现】

参考初始量级仅作为调试起点：

```text
route progress              +1.0 * delta_s_route
success                     +20
vehicle collision           -20
object collision            -15
off-road                    -20
wrong road / wrong way      -10
overspeed                   -0.2 * exceedance
lateral jerk                -0.05 * abs(j_lat)
longitudinal jerk           -0.03 * abs(j_lon)
steering/curvature rate     -0.02 * abs(delta)
stuck                       -0.1 per step after grace period
```

> **批复：已实现有偏差** —— reward 为 progress 10 + success 20 + comfort −0.1；cost 有 6 项并单独记录，但 cost_adv 不进 policy，collision/off-road 不直接惩罚 policy；comfort 100% 违反成常数。

同时分开记录 reward 和 safety cost，不能依靠一个加权总分掩盖碰撞。　——【已实现有偏差：reward/cost 分列；cost 不进 policy】

## 12.3 防 Reward Hacking

必须排查：

- 靠高速冲线抵消碰撞；　——【未实现：无 hacking audit】
- 原地左右摆动刷 progress；　——【未实现】
- 切角获得 route progress；　——【未实现】
- 停车规避风险但无法达成；　——【未实现】
- 借 off-road 缩短路线；　——【未实现】
- 沿错误道路获得局部前进奖励。　——【未实现】

所有 progress 应基于合法 route arc-length，而不是 world displacement。　——【已实现有偏差：用 route_completion 比例，未核 arc-length】

---

## 13. 分阶段训练流程与门槛

## Stage 0：环境、Schema 与 Expert 基线

实施：

- 固定结构化 observation schema；　——【已实现无偏差：schema v2】
- 固定 train/validation/test 场景划分；　——【已实现有偏差：train/val scene split；Test500 spec 存在但 split=val，未接入】
- 建立 Expert 收集、回放和有效性过滤；　——【已实现有偏差：measured replay + 失败段 0.2 权重；无 expert audit/多解降权】
- 完成坐标变换、reset、nav 泄漏、轨迹转换测试。　——【已实现有偏差：ID/补偿/nav leak 测试有；reset 无】

通过条件：

- 所有 schema/unit tests 通过；　——【已实现有偏差：82 passed/1 skipped；闭环 smoke 默认 skip；依赖 gitignored specs；无 parity test】
- Expert 闭环指标可重复；　——【已实现有偏差：单次确定 run；无多 seed】
- 验证集完全冻结；　——【已实现有偏差：manifest+SHA 有；specs gitignored、无 loader/CI 校验；Test500 split=val】
- 没有 train/validation map seed 或 episode 泄漏；　——【已实现无偏差：scene_seed split】
- Expert 特权信息审计报告完成，并明确哪些样本属于 privileged/multi-solution 场景及其监督降权方式。　——【未实现：无 docs/expert_audit.md；multi_solution 无】
- P0 动作数据按 lane keeping、curve、full stop、stop-to-go、slow following、obstacle braking、lane change、junction、fork/merge、toll-gate 分桶；低速、急减速、大曲率设最低配额，数值写 decision log。　——【未实现：无 action 类型配额】

## Stage A1：单帧 BC + Exact Tracking

网络：

- Encoders；　——【已实现无偏差】
- Spatial Fusion；　——【已实现无偏差】
- Plan Head；　——【已实现有偏差：4D PSTL；exact 66.2%；LQR 34%】
- 可选 Value/Risk auxiliary heads；　——【已实现有偏差：heads 有且训练；目标粗】
- 不使用 Memory Bank 和 WM。　——【已实现有偏差：无 Bank/WM；但 6 帧时序已提前】

目的：

- 验证 action/trajectory 定义；　——【已实现无偏差：PSTL 精确执行 v_next】
- 验证 road-level nav 能被网络正确使用；　——【已实现有偏差：无 lane API；无选道专项；Nav 历史单位已修】
- 验证 basic closed-loop；　——【已实现有偏差：Val500 66.2%；exact parity 已修；LQR 34%】
- 提前验证轨迹插值、坐标转换、车辆模型与 LQR 接口，避免到 Stage F 才首次暴露控制不可跟踪问题。　——【已实现有偏差：0.5 s LQR canary；无 Expert-GT 3s】

本阶段以 Exact Tracking 为 Planner 主评测，同时并行运行两类 LQR canary：

1. **Expert GT 3 s trajectory + LQR**：验证插值、坐标系、速度边界、车辆模型和控制接口，只评价 Controller，不评价 Planner；　——【未实现】
2. **Plan Head 0.5 s segment + LQR**：将当前单段插值为 5 个 0.1 s reference points，只做接口、可跟踪性和控制饱和率 canary，不作为正式闭环成绩。　——【已实现有偏差：canary 有；tracking/saturation 有；无固定 seed】

无 WM 时不得声称 Plan Head 能因果生成完整 3 s LQR preview。　——【已实现无偏差：只执行 0.5 s；status 明确 placeholder】

必须报告：

- open-loop point/velocity errors；　——【已实现无偏差：MAE 有】
- exact-tracking closed-loop success；　——【已实现有偏差：Val500 66.2%；exact A/B success CI 跨 0；LQR 34%】
- collision/off-road/wrong-route/stuck；　——【已实现有偏差：wrong-route 0；stuck 2/500；legacy 无 stuck 字段】
- heading、curvature、acceleration、jerk、curvature-rate 与动力学可行率；　——【已实现有偏差：计划侧 max κ/a/jerk；无 realized jerk/steer rate】
- 连续 0.5 s policy decision 段连接处的位姿/速度/航向/曲率跳变量；　——【未实现】
- Expert GT + LQR 的 tracking error、steering/brake/throttle saturation 与失败类型；　——【未实现】
- Plan Head 单段 + LQR 的 tracking error、可跟踪率、控制饱和率，以及 Exact 成功但 LQR canary 失败的比例；　——【已实现有偏差：tracking/saturation 有；无固定 seed】
- 按场景类型拆分结果。　——【已实现有偏差：只按 geometry[0] straight/curve】

若 Exact Tracking 闭环成功但轨迹可行率、连续帧平顺性或 LQR canary 不达标，应先修正 Plan Head/Plan Loss、插值或 Controller，不得等到 Stage F 才处理。Stage F 是 LQR 首次进入完整 3 s 正式执行链路，不是第一次测试 LQR。

## Stage A2：真实历史 Memory Bank

网络增加：

- 6-frame Real Memory Bank；　——【未实现】
- ego-motion compensation；　——【已实现有偏差：补偿有；Nav 历史单位已修；无 Bank】
- stable-ID alignment；　——【已实现有偏差：same-ID 有；LD stable ID 已修；无 ablation】
- entity temporal fusion。　——【已实现有偏差：per-entity temporal pool；LD stable ID 已修，history 可跨 chunk；无 1/2/4/6 ablation】

必须做同 seed ablation：

```text
single frame
2 frames
4 frames
6 frames
```

> **批复：未实现** —— 无 1/2/4/6 帧 ablation；当前 6 帧时序为固定配置。

每个候选 checkpoint 除 Exact Tracking 外，还必须运行固定 seeds 的 **Plan Head 0.5 s segment + LQR canary**，比较：　——【已实现有偏差：r2 一个 LQR50；无逐 ckpt】

- Memory-on/off tracking error 与可跟踪率；　——【未实现】
- steering/brake/throttle saturation；　——【未实现】
- 曲率、速度和航向在连续决策段连接处的跳变量；　——【未实现】
- Exact 指标提升但 LQR canary 退化的比例。　——【未实现】

若历史模型没有稳定超过单帧，或 Exact 提升但 LQR 可跟踪性显著退化，禁止进入 WM 阶段，优先查：　——【未实现：PPO 已跑；无 A2 门禁】

- frame timestamp；　——【未实现】
- reset；　——【未实现】
- ID churn；　——【已实现有偏差：LD stable ID 已修；仍无跨 chunk 测试/ablation】
- coordinate transform；　——【已实现有偏差：OD/LD 有；Nav 历史单位已修；无 parity test】
- birth/death mask；　——【未实现】
- LD segment alignment；　——【已实现有偏差：LD stable ID 已修；仍无 1/2/4/6 ablation】
- attention mask。　——【未实现】

## Stage B：DAgger + Exact Tracking

- 以 A2 最佳 checkpoint 初始化；　——【未实现：A2 不存在】
- 逐轮降低 Expert 混合比例；　——【未实现】
- failure-prioritized aggregation；　——【未实现】
- 每轮固定验证集闭环评测；　——【已实现有偏差：r2/r3 Val500；r1 50】
- 保持 Expert 原始数据与 DAgger recovery 数据的来源标签；　——【已实现有偏差：source/intervened；label_source 已加载（replay=1/lift=0.5）；intervened=replay 仍 3×】
- DAgger 主数据采集仍使用 Exact Tracking，以隔离 Controller 误差；　——【已实现无偏差】
- 每轮 checkpoint 必须在固定 seeds 上运行 Plan Head 单段 LQR shadow evaluation，不参与采集但作为推进门槛。　——【未实现：只有 r2 一个 canary】

进入下一阶段的最低要求：

- Student exact-tracking success 接近 Expert；　——【未实现：66.2% vs 70.15%】
- collision/off-road 明显下降并稳定；　——【已实现有偏差：50 val r3 12/16；Val500 clean 12.2/8.6 vs legacy 18.2/12.0；无多 seed/Test500】
- 复杂场景不因普通道路提升而退化；　——【未实现】
- 冻结验证集连续多个 seed 改善，而非单次偶然值；　——【未实现】
- LQR shadow 的 tracking error、可跟踪率和控制饱和率不得相对 A2 最佳 checkpoint 显著退化；若 Exact 提升但 LQR 持续恶化，不得进入 Stage C。　——【未实现】

## Stage C：短 Real-ENV PPO Canary，无 WM
- 从最佳 DAgger checkpoint 初始化；　——【已实现无偏差：r3 ckpt】
- 保持同一 `(dx,dy,dpsi,v_next)` 概率动作；　——【已实现有偏差：采样/执行 4D；fallback mask】
- 每 0.5 s 采样并执行一个 `a0`，段内运行 5 个 0.1 s 控制步；　——【已实现无偏差】
- PPO 的 value/GAE/return 均来自真实 MetaDrive transition；　——【已实现无偏差：per-episode GAE/truncation bootstrap】
- PPO transition 基线继续使用 Exact Tracking，以减少 WM 与 Controller 变量干扰；　——【已实现无偏差】
- 只进行足以验证 PPO、GAE、reward/cost、buffer、checkpoint 和遗忘控制的短 canary；　——【已实现无偏差：1 update/4 transitions；过短】
- 保留 BC/DAgger auxiliary loss 或 replay regularization；　——【未实现】
- 每个主要 checkpoint 必须在同一组固定 seeds 上运行 Plan Head 单段 LQR shadow evaluation；　——【未实现】
- 不以本阶段是否超过 Expert 作为推进 WM 的硬条件。　——【已实现无偏差】

通过条件：训练无 NaN、log-prob 与 executed action 合同正确、reward/cost 无明显 hacking、冻结验证集安全指标不灾难性退化；同时 LQR shadow 的可跟踪率、控制饱和率和舒适性不得显著恶化。禁止依靠 Exact 的瞬时执行能力获得不可由 LQR 实现的虚假 reward。　——【已实现有偏差：sampled log-prob 正确；cost 不进 policy；mask 25%；无冻结 val/LQR shadow】

## Stage D1：单步 World Model

输入：真实历史 + Expert/Student/PPO canary 实际执行 ego motion。WM target 是真实下一帧 observation 经 EMA Target Encoder 得到的 per-entity latent token，不是融合后的 scene context，也不是 WM 自身输出。　——【未实现】

数据覆盖：必须混合 Expert、DAgger Student、恢复扰动和真实 PPO policy 的已执行 motion；按 `(dx, dy, delta_yaw, acceleration, curvature)` 做覆盖率与 OOD 报告。只用 Expert action 达标不算 WM 可用于 Policy rollout。　——【未实现】

训练策略：

- 冻结或基本冻结已验证好的 Planner/Encoder；　——【未实现】
- Encoder 冻结时 Target Encoder 固定，记录 target drift=0；　——【未实现】
- 先训练 one-step latent prediction；　——【未实现】
- 对动态 OD、ego、road-state 分模态监督。　——【未实现】

必须建立基线：

- persistence predictor；　——【未实现】
- constant-velocity predictor；　——【未实现】
- optional simple MLP transition。　——【未实现】

WM 必须明显超过简单基线，才能进入多步阶段。　——【未实现】

## Stage D2：多步 World Model

课程：

```text
1-step → 2-step → 3-step → 6-step
```

> **批复：未实现** —— 无 WM 课程。

区分三种实验：

1. GT state + GT action teacher forcing；　——【未实现】
2. predicted state + GT action；　——【未实现】
3. predicted state + Plan Head action。　——【未实现】

这样可以分别定位：

- WM transition error；　——【未实现】
- state autoregression error；　——【未实现】
- policy-WM coupling error。　——【未实现】

必须按 rollout depth 画误差曲线，并监控 latent norm/std/effective rank；失真或 rank gate 不通过不得进入 imagined RL。　——【未实现】

当 WM 已能生成 1/2/3/6 步 preview 时，同步进行递增时域 LQR shadow evaluation：先验证 0.5 s，再验证 1.0/1.5/3.0 s。该评测只诊断轨迹拼接、插值和 Controller 可实现性，不参与 WM latent loss。　——【未实现】

## Stage E：Planner + WM 联合磨合

初期：

- 冻结 Encoder 主体；　——【未实现】
- 训练 WM、Plan Head、Value/Risk Head；　——【未实现】
- Plan action 进入 WM 前 detach；　——【未实现】
- Target Encoder 始终 stop-gradient；　——【未实现】
- 此阶段允许 `WM output → Plan/Value` 的梯度更新下游 Head，但默认不允许 PPO/Plan loss 反向改写 WM；只有 real-env 闭环 A/B 证明收益后，才逐层开放 adapter。　——【未实现】

后期可选：

- 只解冻 Encoder 顶层 adapter；此时才启用 EMA 更新并监控 drift；　——【未实现】
- 降低 WM loss 权重；　——【未实现】
- 通过 A/B 决定是否开放 action-to-WM 梯度。　——【未实现】

本阶段必须把 Plan Head + WM 生成的完整 3 s preview 接入 LQR shadow evaluation，并与 Exact/kinematic rollout 同 seed 对照。只有 3 s 轨迹拼接、插值、tracking error、控制饱和率和安全指标达到门槛，才进入 Stage F 的正式 Controller-realized 执行。　——【未实现】

联合损失参考：

\[
L=L_{plan}+\lambda_{wm}L_{wm}+\lambda_vL_{value}+\lambda_rL_{risk}+\lambda_{aux}L_{aux}
\]

> **批复：未实现** —— 无联合损失。

\(\lambda_{wm}\) 初始可从 0.05 至 0.2 搜索，不得让 latent loss 主导 Planner 闭环目标。　——【未实现】

## Stage F：完整 3 s Plan+WM+LQR 正式闭环 + Real-ENV PPO

Stage F 不是第一次测试 LQR，而是 LQR 首次成为完整 3 s Plan+WM 轨迹的**正式执行器和最终成绩口径**。

- 6 个 0.5 s 点插值至 0.1 s；　——【未实现】
- 使用完整 3 s 预瞄参考；　——【未实现】
- 评估 tracking error、curvature、steering、acceleration、jerk；　——【未实现】
- 与 exact tracking 结果并列报告；　——【未实现】
- 在 Plan+WM+LQR 推理链稳定后，从最佳联合 checkpoint 进行正式 Real-ENV PPO + LQR；　——【未实现】
- 正式 PPO 才承担在安全门槛内超过 IDM + Pure Pursuit 的目标，value/GAE/return 仍全部来自真实 MetaDrive transition。　——【未实现】

必须区分：
- planner-oracle closed loop；　——【未实现】
- controller-realized closed loop。　——【未实现】

如果 exact tracking 好而 LQR 差，优先查轨迹可行性和控制器，不要把问题直接归因于 Planner。　——【已实现有偏差：本次 review 已按此诊断，代码无自动归因】

## Stage G：Imagined RL / WM Distillation，可选最后阶段

只有满足以下条件才开放：

- 6-step WM fidelity 达标；　——【未实现】
- imagined risk 与 real-env failure 有稳定相关性；　——【未实现】
- imagined action distribution 未明显 OOD；　——【未实现】
- real-env reward 没有因 joint training 下降。　——【未实现】

策略：

- imagined loss 按 rollout depth 衰减；　——【未实现】
- high uncertainty 时截断；　——【未实现】
- REAL 和 IMAGINED 数据严格标记；　——【未实现】
- 最终模型始终用 real-env closed-loop 决策，不以 imagined score 单独选 checkpoint。　——【未实现】

---

## 14. 评测协议

## 14.1 数据划分
- Train：10,000；其中 `Calib50` 属于训练域，只作快速调试；　——【已实现有偏差：2k 数据/10k specs；Calib50 未按训练域单独定义】
- Validation：500，用于 checkpoint、阈值和消融，永久冻结；　——【已实现有偏差：manifest+SHA 有；未 enforced】
- Test：500，仅用于里程碑和最终结论，不调参、不删失败样本；　——【已实现有偏差：Test500 spec 存在但 split=val；未接入/未跑】
- P0/P1 可先用 2,000 train / 200 evaluation-scale scenes 做工程 smoke，不替代最终规模；　——【已实现无偏差：2k train/Val500】
- 按 scenario/map seed 划分，禁止 train/val/test 泄漏。　——【已实现无偏差：scene_seed split】

## 14.2 核心闭环指标

主指标：

- route success rate；　——【已实现无偏差】
- route completion；　——【已实现无偏差】
- collision rate；　——【已实现无偏差】
- off-road rate；　——【已实现无偏差】
- wrong-route/wrong-way rate；　——【已实现有偏差：代码有；Val500 0 事件，未验证】
- stuck rate；　——【已实现有偏差：代码有；Val500 2/500；legacy 无 stuck 字段】
- timeout；　——【已实现有偏差：termination 有；独立率无】
- comfort violations；　——【已实现有偏差：plan-side 有；100% 违反】
- mean time/distance to failure。　——【未实现】

控制指标：

- lateral tracking error；　——【已实现有偏差：tracking xy 有；realized 指标部分】
- heading error；　——【已实现有偏差：tracking heading 有】
- speed tracking error；　——【已实现有偏差：tracking speed 有】
- peak/mean curvature；　——【已实现有偏差：计划侧 max κ；realized 无】
- steering rate；　——【未实现】
- longitudinal/lateral acceleration；　——【已实现有偏差：计划侧 max a_lat；realized 无】
- jerk。　——【已实现有偏差：计划侧 max jerk_long；realized 无】

WM 指标：

- one-step and 1..6-step latent error；　——【未实现】
- entity decode error；　——【未实现】
- rollout uncertainty calibration；　——【未实现】
- static LD invariance after ego compensation；　——【未实现】
- dynamic OD prediction by distance/type；　——【未实现】
- imagined-vs-real risk correlation。　——【未实现】

## 14.3 场景分桶

至少分为：

- straight lane keeping；　——【已实现有偏差：只 geometry[0] straight/curve】
- following and braking；　——【未实现：无 bucket 指标】
- obstacle/cone avoidance；　——【未实现】
- lane choice before fork；　——【未实现】
- merge；　——【未实现】
- ramp；　——【未实现】
- intersection；　——【未实现】
- toll station；　——【未实现】
- low-speed sharp turn；　——【未实现】
- dense traffic；　——【未实现】
- recovery from disturbed state。　——【未实现】

总体指标不能掩盖关键场景退化。　——【未实现：无分桶表】

## 14.4 Checkpoint 选择

不能只按 training loss 或 open-loop ADE 选择。建议使用带安全门槛的闭环优先级：　——【已实现有偏差：best.pt 按内部 val total loss（r3 epoch 2/15）；无闭环门槛；YAML 不读】

1. collision/off-road 不超过硬门槛；　——【未实现】
2. route success 最大；　——【未实现】
3. route completion 最大；　——【未实现】
4. comfort 和效率作为次级排序；　——【未实现】
5. open-loop loss 仅作为诊断。　——【未实现】

---

## 15. 重要排摸点与 Probes/Hooks

## 15.1 数据与坐标

必须记录/可视化：

- 每帧 ego pose；　——【已实现有偏差：obs 内有 pose；未落盘/可视化】
- compensation 前后 OD/LD；　——【未实现】
- 同 ID 轨迹；　——【未实现】
- lane/polyline anchor；　——【未实现】
- route graph 与 student-visible nav；　——【未实现】
- Expert-visible privileged info；　——【未实现】
- frame timestamp 和 scene reset。　——【未实现】

关键断言：

- 场景切换后 bank 全空；　——【未实现：Bank 机制不存在，禁止项暂无适用对象】
- 静态 LD 经 compensation 后跨帧基本一致；　——【未实现】
- 同 ID 物体不会跳槽；　——【未实现】
- 不同 ID 不会误作同一实体；　——【未实现】
- 六个 rollout 点位于同一 t0 坐标系。　——【未实现】

## 15.2 Latent probes

每模态记录：

- mean/std/min/max/norm；　——【未实现】
- temporal cosine similarity；　——【未实现】
- same-ID delta 与 different-ID delta 分布；　——【未实现】
- padding token 激活；　——【未实现】
- static LD delta；　——【未实现】
- ego information leakage 到 OD/LD latent 的线性 probe；　——【未实现】
- gradient norm；　——【未实现】
- attention entropy；　——【未实现】
- top attended entities。　——【未实现】

重点判断：

- latent collapse；　——【未实现】
- OD/LD latent 仅复制 ego motion；　——【未实现】
- Planner 忽视 Nav；　——【未实现】
- Planner 只看最近障碍物而不看长程路线；　——【未实现】
- World Model 无视 ego condition；　——【未实现】
- Token mask 泄漏。　——【未实现】

## 15.3 Causality probes

对同一个 Memory Bank：

- 固定环境，替换不同 ego action，WM 输出应显著不同；　——【未实现】
- 固定 ego action，扰动关键 OD，Planner action 应合理变化；　——【未实现】
- shuffle ego condition 后，WM loss 应恶化；　——【未实现】
- 将 ego condition 置零，WM 不应仍得到近乎相同性能；　——【未实现】
- 替换 road-level Nav 后，选道应变化，但近距离碰撞响应不应失效。　——【未实现】

## 15.4 World Model probes

必须实现：

- one-step teacher-forced decode visualization；　——【未实现】
- six-step autoregressive visualization；　——【未实现】
- per-depth error；　——【未实现】
- GT action vs Plan action rollout 对照；　——【未实现】
- real-vs-imagined entity count、position、velocity 对照；　——【未实现】
- uncertainty-error reliability curve；　——【未实现】
- simple baseline comparison。　——【未实现】

## 15.5 Planner probes

必须实现：

- 输出六步轨迹叠加到场景；　——【未实现】
- 每一步局部坐标和 t0 累计坐标同时显示；　——【未实现】
- route topology 和 chosen lane 可视化；　——【未实现】
- 轨迹 feasibility report；　——【未实现】
- exact tracking 和 LQR tracking 对照；　——【未实现】
- action distribution by scenario；　——【未实现】
- Expert/Student disagreement case replay。　——【未实现】

## 15.6 梯度 probes

每个训练阶段报告：

- 模块级 gradient norm；　——【未实现】
- 哪些模块 requires_grad；　——【未实现】
- 哪些 tensor detached；　——【未实现】
- EMA update 正常性；　——【未实现】
- recurrent rollout graph depth；　——【未实现】
- 是否出现跨 episode graph 泄漏；　——【未实现】
- plan loss、WM loss、value loss 对 Encoder 梯度贡献比例。　——【未实现】

提供自动化 `test_detach_contract.py`，不能只靠代码阅读。　——【未实现】

---

## 16. 关键 Failure Taxonomy

每次闭环失败必须归入至少一个主因：

- `DATA_SCHEMA_ERROR`
- `COORDINATE_TRANSFORM_ERROR`
- `ID_ALIGNMENT_ERROR`
- `SCENE_RESET_LEAK`
- `NAV_LEAK_OR_INSUFFICIENT_NAV`
- `EXPERT_PRIVILEGE_MISMATCH`
- `PLANNER_LANE_SELECTION_ERROR`
- `PLANNER_SPEED_ERROR`
- `PLAN_INTERFACE_SHIFT`
- `ACTION_V_NEXT_INCONSISTENT`
- `PLANNER_OBSTACLE_RESPONSE_ERROR`
- `TRAJECTORY_INFEASIBLE`
- `INTERPOLATION_OVERSHOOT`
- `LQR_TRACKING_ERROR`
- `WM_ACTION_INSENSITIVE`
- `WM_ROLLOUT_DRIFT`
- `WM_LATENT_COLLAPSE`
- `STATIC_REPROJECTION_ERROR`
- `MEMORY_WARMUP_ERROR`
- `TRAIN_EVAL_ASSEMBLY_MISMATCH`
- `POLICY_EXPLOITS_WM`
- `REWARD_HACKING`
- `OOD_SCENARIO`
- `UNKNOWN`

> **批复：未实现** —— 无 failure taxonomy 代码/统计；本次 review 已实际命中 `DATA_SCHEMA_ERROR`、`ID_ALIGNMENT_ERROR`、`NAV_LEAK_OR_INSUFFICIENT_NAV`、`EXPERT_PRIVILEGE_MISMATCH`、`PLAN_INTERFACE_SHIFT`、`ACTION_V_NEXT_INCONSISTENT`、`TRAJECTORY_INFEASIBLE`、`LQR_TRACKING_ERROR` 等症状，但未被系统记录。

`UNKNOWN` 比例必须持续下降。每个关键失败要能用 scene seed 一键复现。　——【未实现：无 taxonomy/统计】

---

## 17. 文件夹结构

```text
env/
  metadrive_env.py
  observation_adapter.py
  scenario_manager.py
  expert_policy.py
  exact_tracker.py
  controller/
    lqr.py
    controller_adapter.py

dataset/
  schema.py
  metadrive_adapter.py
  expert_collector.py
  scenario_split.py
  sequence_builder.py
  replay_buffer.py
  dagger_buffer.py
  failure_buffer.py
  id_alignment.py
  coordinate_transform.py

net/
  encoders/
    od_encoder.py
    ld_encoder.py
    nav_encoder.py
    ego_encoder.py
    target_encoder.py
  fusion/
    entity_temporal_fusion.py
    od_self_attention.py
    ld_self_attention.py
    od_ld_cross_attention.py
    context_fusion.py
  planner/
    plan_head.py
    trajectory_assembler.py
    trajectory_interpolator.py
    feasibility.py
  value/
    value_head.py
    risk_head.py
  world_model/
    ego_condition.py
    ego_dynamics.py
    latent_world_model.py
    uncertainty_head.py
    imagine_memory_bank.py
  memory/
    real_memory_bank.py
    memory_schema.py
  model.py

config/
  env.yaml
  data.yaml
  model.yaml
  memory.yaml
  train_bc.yaml
  train_dagger.yaml
  train_rl.yaml
  train_wm.yaml
  train_joint.yaml
  evaluate.yaml
  eval_thresholds.yaml
  runtime_local.yaml

pipeline/
  collect_expert.py
  build_dataset.py
  train_bc.py
  train_dagger.py
  train_rl.py
  train_wm_one_step.py
  train_wm_multistep.py
  train_joint.py
  evaluate_open_loop.py
  evaluate_closed_loop_exact.py
  evaluate_closed_loop_lqr.py

  stages.py
  checkpointing.py
  losses.py
  metrics.py

  rollout/
    real_rollout.py
    imagined_rollout.py
    dagger_rollout.py

  probes/
    activation_probe.py
    attention_probe.py
    gradient_probe.py
    latent_probe.py
    causality_probe.py

  hooks/
    tensor_hook.py
    module_hook.py
    failure_hook.py

tools/
  visualize_scene.py
  visualize_tokens.py
  visualize_trajectory.py
  inspect_memory_bank.py
  inspect_latent.py
  inspect_gradients.py
  reproduce_failure.py
  audit_expert_privilege.py
  audit_nav_leak.py
  profile_model.py
  count_parameters.py
  export_model.py

test/
  test_schema.py
  test_coordinate_transform.py
  test_id_alignment.py
  test_ld_segment_alignment.py
  test_scene_reset.py
  test_memory_fifo.py
  test_real_imagined_isolation.py
  test_nav_no_lane_leak.py
  test_trajectory_assembly.py
  test_trajectory_interpolation.py
  test_trajectory_temporal_continuity.py
  test_encoder_cache_version.py
  test_ego_condition.py
  test_wm_action_sensitivity.py
  test_rollout_consistency.py
  test_detach_contract.py
  test_checkpoint_reproducibility.py
  test_action_kinematics.py
  test_velocity_chain_consistency.py
  test_plan_head_single_instance.py
  test_param_budget.py
  test_tensor_budget.py
  test_lqr_gt_tracking.py
  test_lqr_plan_tracking.py
  test_train_eval_assembly_parity.py
  test_bank_entity_schema.py
  test_validation_isolation.py

docs/
  architecture_contract.md
  data_contract.md
  trajectory_contract.md
  detach_contract.md
  training_stages.md
  evaluation_protocol.md
  expert_audit.md
  failure_taxonomy.md
  experiment_log.md
  decision_log.md
```

> **批复：已实现有偏差** —— 目录树逐项对照（实际仓库 58 个 tracked 文件；`git ls-files` 2026-10-20）：
>
> | 合同目录 | 实际 | 判定 |
> |---|---|---|
> | `env/metadrive_env.py` | 存在 | 已实现有偏差 |
> | `env/observation_adapter.py` | 无；功能在 `env/obs.py`（1430 行，schema/几何/ID/Nav/history 混合） | 已实现有偏差 |
> | `env/scenario_manager.py` | 无；改为 `env/scenario/`（7 文件） | 已实现有偏差 |
> | `env/expert_policy.py` | 无；改为 `env/expert/pure_pursuit_idm.py` | 已实现有偏差 |
> | `env/exact_tracker.py` | 无；ExactTracker/LQR/DARE/2D 插值同在 `env/tracking.py`（838 行） | 已实现有偏差 |
> | `env/controller/lqr.py` | 无；LQR 同在 `env/tracking.py` | 已实现有偏差 |
> | `dataset/schema.py`、`id_alignment.py`、`coordinate_transform.py`、`sequence_builder.py`、`scenario_split.py` | 无；常量/ID/补偿/history 在 `env/obs.py`，split 在 `dataset/splits.py` | 未实现（结构） |
> | `dataset/replay_buffer.py`、`dagger_buffer.py`、`failure_buffer.py` | 无；PPO buffer 在 `pipeline/train_ppo.py` | 未实现（结构） |
> | `dataset/expert_collector.py` | 无；改为 `dataset/collect.py`（902 行） | 已实现有偏差 |
> | `net/encoders/*`、`fusion/*`、`planner/*`、`value/*`、`memory/*`、`world_model/*` | 无；主体在 `net/model.py`（403 行），执行层在 `net/action_space.py` | 未实现（结构） |
> | `config/*` | 仅 5/12；且无 `yaml` 加载 | 已实现有偏差 |
> | `pipeline/*` | 有 train_bc/train_ppo/eval/eval_metrics/distributions/rollout；无 `stages/losses/metrics/checkpointing`、`rollout/`、`probes/`、`hooks/` | 已实现有偏差 |
> | `tools/*` | monitor/status/venv-python + `freeze_splits`；仍缺 11 个规定工具/CI | 未实现 |
> | `test/*` | 13 个测试文件，82 passed/1 skipped；缺 `test_schema`、`test_id_alignment`、`test_nav_no_lane_leak`、`test_train_eval_assembly_parity` 等合同命名 | 已实现有偏差 |
> | `docs/*` | 5 个（新增 eval_split_manifest.json）；缺 `decision_log.md`、`expert_audit.md` 等 7 个合同/日志文档 | 未实现 |

> **代码质量总评（2026-10-21）**：规范/解耦/内聚 = **partial**。delta 本身是外科手术式的（Nav/steering/LD/OD/weights 均有明确修复和测试）；但 `prev_state/executed_action` 在 collect/eval/PPO 有 3–4 份拷贝，`resolve_action` 在 eval/PPO/DAgger 有 3 份“mirror”实现且已出现结构差异；`pipeline/eval.py` 的 `run_episode`（约 310 行）同时承担 rollout/fallback/metrics/CLI；`train_bc._plan_weights` 与日志 diagnostic 已漂移；`env/obs.py` 1,430 行、`pipeline/train_bc.py` 1,199 行、`pipeline/train_ppo.py` 1,621 行、`net/action_space.py` 1,365 行，`env/tracking.py` 838 行仍含 2D legacy；`config/*.yaml` 无 `yaml` import；测试依赖 gitignored `env/specs`，非 hermetic；无 pyproject/lint/CI。结论：P0 修复可保留，但 P1/P2 gate 前应先抽 `execution.py` + `checkpointing.py`、补 parity test 和 checkpoint schema/bounds 校验；`env/obs.py`/`action_space.py` 的物理拆分后置。

---

## 18. 工程约定

## 18.1 配置与可复现性

- 禁止在训练代码散落 magic numbers；　——【已实现有偏差：大量硬编码；YAML 不加载】
- 超参数统一进入 YAML；　——【未实现：无 yaml import；config 不加载】
- 每次 run 保存 resolved config；　——【已实现有偏差：checkpoint 内 config 有；无 git/dirty/deps/RNG】
- 固定 Python、NumPy、PyTorch、MetaDrive 和 environment seed；　——【已实现有偏差：torch/np/spec seed；PYTHONHASHSEED 无】
- 保存 git commit、dirty status、依赖版本和设备信息；　——【未实现：无 dirty/依赖/设备】
- 每个实验生成唯一 run ID；　——【已实现有偏差：输出目录名；无唯一 ID】
- validation set 配置和 seed 只读，不允许训练脚本改写。　——【已实现有偏差：manifest+SHA 有；未 enforced】
- validation/test 均只读；worker/recycle/memory floor 写 `runtime_local.yaml`，按机器标定，不是架构硬约束。　——【已实现有偏差：runtime_local 不读】

## 18.2 Shape 与 Schema 约定

每个模块 docstring 必须写：

- 输入/输出 tensor shape；　——【已实现有偏差：部分模块有；net/model.py 仍有 shape 错】
- 坐标系；　——【已实现有偏差：部分】
- 单位；　——【已实现有偏差：部分】
- mask 语义；　——【未实现：无统一约定】
- ID 对齐语义；　——【已实现有偏差：代码有；docstring 未统一】
- 是否 REAL/IMAGINED；　——【未实现】
- 是否 detach；　——【未实现】
- 时间索引。　——【已实现有偏差：部分模块有帧索引】

对关键 tensor 使用统一命名：　——【已实现有偏差：B/T 有；R/No/Nl/P/D 未统一】

```text
B: batch
T: history length, 6
R: rollout depth, 6
No: max OD entities
Nl: max LD entities
P: polyline points
D: latent dimension
```

> **批复：已实现有偏差** —— `net/model.py` 的 `od_ids`/`od_anchor` 注释形状写成 `[B,T,...]`，实际 `encode_entities` 返回 `[B,...]`；R/No/Nl/P/D 未统一；LD shape 已修正为 16。

## 18.3 单位

全工程统一 SI：　——【已实现无偏差】

- position：m；
- velocity：m/s；
- acceleration：m/s^2；
- yaw/steering：rad；
- yaw rate：rad/s；
- time：s；
- curvature：1/m。

严禁角度和弧度混用。所有归一化必须可逆并有单元测试。　——【已实现有偏差：归一化 roundtrip 有测试；标签 OOB 0.07–0.3%，未用于权重】

## 18.4 Mask 约定

必须明确 attention API 中：

- `True` 表示 valid 还是 masked；　——【已实现有偏差：代码一致但未文档化】
- padding mask 与 causal mask 的 shape；　——【已实现有偏差：padding 有；causal 无】
- birth/death mask 的独立语义；　——【未实现】
- imagined depth mask。　——【未实现】

严禁依赖不同 PyTorch API 的默认 mask 语义。　——【已实现有偏差：显式 key_padding_mask，但无约定文档】

## 18.5 Logging

每个 run 至少保存：

- losses by component；　——【已实现无偏差：分量 loss 有】
- closed-loop metrics；　——【已实现有偏差：指标有；exact parity 已修；LQR 34%】
- scenario bucket metrics；　——【已实现有偏差：仅 geometry/difficulty】
- model/optimizer LR；　——【未实现】
- gradient norms；　——【未实现：仅 clip】
- latent statistics；　——【未实现】
- rollout depth errors；　——【未实现】
- sample videos/plots；　——【未实现】
- failure seeds；　——【已实现无偏差：eval 记录 scene_seed】
- checkpoint selection reason。　——【已实现有偏差：best.note；非闭环】

## 18.6 Checkpoint

Checkpoint 必须包含：

- online model；　——【已实现无偏差】
- EMA target encoder；　——【未实现】
- optimizer/scheduler；　——【未实现】
- scaler；　——【未实现】
- training stage；　——【未实现】
- global step/epoch；　——【已实现有偏差：仅 epoch】
- RNG states；　——【未实现】
- resolved config；　——【已实现有偏差：config dict 有；非 YAML resolved】
- dataset split hash；　——【未实现】
- encoder version。　——【未实现】

禁止加载不兼容 checkpoint 后 `strict=False` 静默跳过关键参数。任何 missing/unexpected keys 必须显式记录并审计。　——【已实现无偏差：strict load；但不校验 schema/bounds】

## 18.7 测试门槛

任何进入主分支的改动必须至少通过：

```text
unit tests
small deterministic smoke train
one-episode closed-loop smoke test
shape/mask assertions
NaN/Inf check
parameter-count check
```

> **批复：已实现有偏差** —— unit tests 82/1；small deterministic smoke train 有；one-episode closed-loop smoke 默认 skip；shape/mask 部分；NaN/Inf 数据层；parameter-count 有；causality/reset 无。

影响 Memory/WM/Trajectory 的改动还必须通过对应 causality 和 reset tests。　——【未实现】

## 18.8 性能和大张量约束

- 不做 6 帧所有实体点的全连接全量 attention；　——【已实现无偏差】
- polyline 先局部聚合；　——【已实现无偏差】
- 记录每层 token 数；　——【未实现】
- 禁止隐式 broadcast 产生巨大中间 tensor；　——【已实现无偏差】
- profiler 中出现异常大张量必须解决后再推进；　——【未实现】
- rollout 6 步时严查计算图和显存线性/超线性增长。　——【未实现：无 WM/6 步图】

## 18.9 代码质量

- 不以临时 if/else 规则修复具体场景；　——【已实现无偏差】
- 不把场景 seed 写进策略；　——【已实现无偏差】
- 不隐藏 fallback；　——【已实现有偏差：env/obs.py 30 处 broad except + 零填充】
- 所有 fallback 要可计数并记录；　——【未实现】
- 核心算法改动必须配设计说明和 ablation；　——【未实现：PPO/DAgger 有 run；无同 seed 关闭模块 ablation】
- 文档结论必须能追溯到代码、配置、run 和 checkpoint。　——【未实现：runs/datasets/specs gitignore；meta git_commit 不 dirty】

---

## 19. 必做 Ablation

至少完成以下对照：

1. single frame vs 2/4/6-frame memory；　——【未实现】
2. no ego-motion compensation vs compensation；　——【未实现】
3. no latent delta vs absolute+delta；　——【未实现】
4. global scene temporal fusion vs same-ID entity temporal fusion，如实现过前者；　——【未实现】
5. no Nav vs road-level Nav；　——【未实现】
6. exact tracking vs LQR；　——【已实现有偏差：Val500 + LQR50 canary；非正式 ablation】
7. Plan without WM vs Plan with WM；　——【未实现】
8. one-step WM vs six-step WM；　——【未实现】
9. teacher-forced action vs predicted Plan action；　——【未实现】
10. action condition enabled vs shuffled/zero action condition；　——【未实现】
11. frozen Encoder vs adapter joint tuning；　——【未实现】
12. plan-action-to-WM gradient detached vs open；　——【未实现】
13. BC vs DAgger vs real-env RL；　——【已实现有偏差：三种 run 有；非同口径】
14. deterministic WM with vs without uncertainty head。　——【未实现】

每个 ablation 使用相同 split、seed 组、评测协议和 checkpoint 规则。　——【未实现】

---

## 20. Coding Agent 实施优先级

按以下顺序开工，不得跨阶段一次性堆满：　——【已实现有偏差：DAgger/PPO 提前；6 帧 A2 特性提前】

### P0：合同与测试基础

- schema；　——【已实现无偏差：v2】
- coordinate transform；　——【已实现有偏差：OD/LD 有；Nav 历史补偿已修复；无 parity test】
- scenario split；　——【已实现无偏差：scene_seed】
- Expert collector/audit；　——【已实现有偏差：collector 有；audit 字段常量】
- trajectory representation；　——【已实现无偏差：4D PSTL】
- exact tracking；　——【已实现无偏差】
- 动作参数化 P0 否决门；　——【未实现】
- LQR 跟踪 Expert GT 的 controller smoke；　——【未实现】
- MetaDrive headless smoke 和 worker 内存标定；　——【已实现无偏差】
- all critical unit tests。　——【已实现有偏差：82/1；缺关键 parity/causality】

### P1：单帧 BC

- Encoders；　——【已实现无偏差】
- Spatial Fusion；　——【已实现无偏差】
- Plan Head；　——【已实现有偏差：4D PSTL；exact 66.2%；LQR 34%】
- open/closed-loop evaluation；　——【已实现有偏差：Val500 66.2%；LQR 34%】
- failure reproduction。　——【未实现】

### P2：真实历史 + DAgger

- Memory Bank；　——【未实现】
- ID alignment；　——【已实现有偏差：same-ID 有；LD stable ID 已修；无 1/2/4/6 ablation】
- temporal fusion；　——【已实现有偏差：per-entity temporal；LD stable ID 已修，history 可跨 chunk；无 1/2/4/6 ablation】
- DAgger/failure buffer；　——【已实现有偏差：DAgger 3 轮；failure buffer 无】
- nav leak and reset audit。　——【已实现有偏差：nav leak test 有；reset 无】

### P3：短 Real-ENV PPO Canary
- PPO/action-log-prob/GAE/buffer；　——【已实现有偏差：sampled log-prob 正确；cost 不进 policy】
- reward/cost 与 hacking audit；　——【未实现】
- 只验证训练管线与无灾难性退化，不承担 exceed-IDM 结论。　——【已实现有偏差：1 update/4 transitions】

### P4：World Model

- target encoder；　——【未实现】
- ego condition；　——【未实现】
- one-step WM；　——【未实现】
- multi-step WM；　——【未实现】
- Imagine Bank；　——【未实现】
- causality/rollout probes。　——【未实现】

### P5：完整 3 s LQR 正式闭环、联合磨合与 Real-ENV PPO
- Plan+WM 完整 3 s preview；　——【未实现】
- trajectory assembly/interpolation 与 dynamic feasibility；　——【未实现】
- LQR controller-realized closed loop；　——【未实现】
- joint training；　——【未实现】
- 正式 Real-ENV PPO + LQR；　——【未实现】
- exceed-IDM 与 exact-vs-LQR diagnosis。　——【未实现】

### P6：可选 imagined RL

仅在所有 WM 门槛通过后进行。　——【未实现】

---

## 21. Coding Agent 每轮交付要求

每一轮必须提交：

1. 修改文件清单；　——【已实现无偏差：本轮 delta 13 文件（代码/测试/工具 9、文档 4）；仓库 tracked 58】
2. 设计意图与本文合同的对应条款；　——【已实现有偏差：status 未逐条映射】
3. 关键代码 diff 说明；　——【未实现】
4. 新增/修改测试；　——【已实现有偏差：82/1；覆盖不足；无 parity/OD acc/weight 分支测试】
5. 实际运行命令；　——【已实现无偏差】
6. 配置文件；　——【已实现有偏差：文件存在但不读】
7. 运行结果与原始日志路径；　——【未实现：runs/datasets ignore】
8. 与上一基线的同口径对比；　——【已实现有偏差：Val500/50 混用；slice50 与 Val500 43/50 scene 不同】
9. 已知问题和未验证假设；　——【已实现无偏差：status 诚实列出】
10. 下一轮只解决的一个主要问题。　——【已实现有偏差：多个方向】

禁止只提交总结文档而不提供代码、日志和测试证据。禁止用单次成功 episode 代表总体性能。　——【已实现有偏差：小样本结论强于证据】

---

## 22. 最终不可违反的硬约束

1. World Model 必须显式 ego-action/ego-trajectory conditioned。　——【未实现：无 WM】
2. Plan Head 每次输出下一 0.5 s `(dx,dy,dpsi,v_next)`，rollout 6 次形成 3 s；不得自动删除 `v_next` 或退化为 3D。　——【已实现有偏差：4D/PSTL/v_next 已执行；6 步自回归无】
3. 六步轨迹点必须转换到同一个 rollout-t0 ego frame 后才能拼接。　——【未实现】
4. Stage F 正式 LQR 使用完整 3 s；A1-C 的 0.5 s LQR 仅是 canary/shadow。　——【已实现有偏差：0.5s canary；3s 无】
5. BC、DAgger、RL 使用同一种轨迹动作语义。　——【已实现有偏差：生产 4D；tracking 2D legacy】
6. OD/LD 先 ego-motion compensation，再按稳定 ID 对齐并计算 latent delta。　——【已实现有偏差：stable ID 有；Nav 历史单位已修；latent delta 无】
7. latent delta 必须与绝对 latent、时间和 mask 一起使用。　——【未实现】
8. Road-level Nav 严禁包含 lane-level oracle 或推荐车道。　——【已实现无偏差：current-frame road-level；无 target lane】
9. 环境语义全部经 Encoder；控制 mask/reset 可直接使用，不属于策略旁路。　——【已实现有偏差：OD/LD encoder；TTC aux 规则】
10. Real Memory Bank 与 Imagine Memory Bank 禁止混写。　——【未实现：Bank 不存在】
11. Future latent GT 必须来自真实下一帧 observation 的 per-entity EMA Target Encoder 输出并 stop-gradient；不得用 fused context 或 WM 自输出作 GT。　——【未实现】
12. WM 先单步、再多步，未通过六步 fidelity 不得进入 imagined RL。　——【未实现】
13. Exact tracking 只证明 Planner 闭环能力；最终成绩必须报告 LQR controller-realized closed loop。　——【已实现有偏差：exact/LQR canary；正式 LQR 口径无】
14. IDM + Pure Pursuit 可作为唯一 Expert，但必须审计特权信息，并以真实闭环指标而非 imitation distance 判断是否超越。　——【已实现有偏差：字段有；无报告/多解降权】
15. 每新增模块必须有关闭该模块的同 seed ablation。　——【未实现】
16. 调参与 checkpoint 依据冻结 Val500；最终结论必须由只读 Test500、多 seed、分场景复核。　——【已实现有偏差：Val500 manifest 有但未 enforced；Test500 spec=val、未跑】

---

## 23. 当前待实现时允许搜索、但不允许静默自行改动的开放项

以下参数允许 Coding Agent 通过小规模实验选择，但必须记录对照，不得默认拍脑袋：

- latent dimension 96 或 128；　——【已实现有偏差：选 128；无 decision log】
- temporal/spatial attention 层数；　——【已实现有偏差：2 spatial/1 temporal；无 log】
- OD/LD token 上限；　——【已实现有偏差：16/16；LD 低于建议；无 log】
- polyline anchor/chunk 方法；　——【已实现有偏差：5 点/chunk 20m；无 log】
- exact tracking 的实现细节；　——【已实现无偏差：有；无 log】
- Plan Loss 各项权重；　——【已实现有偏差：3/2→8/4→12/6；无 log】
- EMA momentum；　——【未实现】
- WM loss 权重；　——【未实现】
- PPO 具体超参数；　——【已实现有偏差：硬编码；无 log】
- LQR state、Q/R 和 preview 实现；　——【已实现有偏差：有；无 log】
- trajectory interpolation 方法；　——【已实现有偏差：PSTL；无 log】
- uncertainty head 的监督定义；　——【未实现】
- `(dx,dy,dpsi,v_next)` 的最终 bounds 与归一化范围，须经 P0 Expert 分位数、动力学边界和 LQR 可跟踪性联合标定。　——【已实现有偏差：CLEAN_BOUNDS 标定；LQR 未闭环联合校准】

任何开放项一旦通过实验确定，应写入 `docs/decision_log.md`，包含：候选方案、实验运行、同口径结果、选择理由和回退条件。　——【未实现：无 docs/decision_log.md；设计决策在 status】

---

## 24. 最终验收标准

项目不以“代码能跑”验收，而以以下证据链验收：

```text
Schema/坐标/Reset/ID tests 全通过
    ↓
单帧 BC exact-tracking 闭环可用
    ↓
历史 Memory 带来可重复闭环收益
    ↓
DAgger 改善 Student 访问状态下的恢复能力
    ↓
短 Real-ENV PPO canary 证明训练管线正确且无灾难性退化
    ↓
Action-conditioned WM 明显超过简单预测基线
    ↓
6-step imagined rollout 误差和风险可控
    ↓
完整 3 s 轨迹经插值和 LQR 后闭环性能达标
    ↓
正式 Real-ENV PPO + LQR 在安全门槛内接近或超过 IDM + Pure Pursuit
    ↓
联合训练带来真实环境收益，而非仅 imagined score 提升
```

> **批复：未实现** —— 证据链逐项：
>
> | 环节 | 状态 |
> |---|---|
> | Schema/坐标/Reset/ID tests 全通过 | 已实现有偏差（schema/ID/补偿/nav leak 有；reset 无；Nav 历史补偿已修复） |
> | 单帧 BC exact-tracking 闭环可用 | 已实现有偏差（Val500 66.2%；exact parity 已修；LQR 34%；P1 实际 6 帧） |
> | 历史 Memory 带来可重复闭环收益 | 已实现有偏差（stable-ID history；LD stable ID 已修；无 Bank/1-2-4-6 ablation） |
> | DAgger 改善恢复能力 | 已实现有偏差（3 轮；69% lift；无 failure buffer） |
> | 短 Real-ENV PPO canary 证明管线正确 | 已实现有偏差（sampled log-prob 正确；cost 不进 policy；1 update/4 transitions） |
> | Action-conditioned WM 超过基线 | 未实现 |
> | 6-step imagined rollout 误差/风险可控 | 未实现 |
> | 完整 3 s 插值 + LQR 闭环达标 | 已实现有偏差（0.5s canary；3s 无） |
> | 正式 Real-ENV PPO + LQR 接近/超过 IDM | 未实现 |
> | 联合训练带来真实环境收益 | 未实现 |

最终模型的核心判据是：

> 在冻结验证集、多 seed、controller-realized closed loop 下，route success 和 completion 提升，collision/off-road 不恶化或下降，并且提升能够在车道保持、障碍物响应、自主选道、路口与收费站等关键场景中解释和复现。　——【未实现：Val500 manifest 有但未 enforced、Test500 spec=val、多 seed 无】


