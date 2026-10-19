# MetaDrive 端到端自动驾驶 IL + DAgger + RL 实施设计合同

> **文档用途**：供 Coding Agent 直接实施。除非实验结果明确推翻，本文中的模块边界、数据合同、因果顺序、训练阶段、冻结策略、评测口径和工程约定均视为第一版基线约束。
>
> **核心目标**：基于 MetaDrive 构建约 100 万参数的结构化端到端自动驾驶系统，使用 OD、LD、道路级导航和自车动力学输入，依次完成 IL、DAgger、真实环境 RL，再引入 ego-conditioned latent world model、imagined rollout 与 LQR，实现车道保持、障碍物响应、自主选道、路口和收费站通行，并在冻结验证集闭环指标上争取超过 IDM + Pure Pursuit。


## 1. 目标、范围与非目标

### 1.1 最终目标

系统需要在 MetaDrive 闭环环境中完成：

- 官方案例和自定义场景中的稳定车道保持；
- 根据车辆、锥桶、收费站等 OD 做减速、停车、绕行；
- 在仅有道路级导航、没有车道级 oracle 的情况下自主选道；
- 通过分岔、匝道、路口、收费站等拓扑复杂区域；
- 保持安全性、平顺性、效率和路线达成率；
- 最终在冻结验证集的 controller-realized closed-loop 指标上超过 IDM + Pure Pursuit 基线。

### 1.2 第一阶段目标

第一阶段暂不上 latent world model，以 exact tracking 为 Planner 主诊断，同时从早期并行运行 LQR canary，重点证明：

- 输入 schema 正确；
- 道路级导航真的足以支持自主选道；
- Plan Head 能生成闭环可用轨迹；
- BC、DAgger 以及短 Real-ENV PPO canary 的训练链路可运行；
- 模型可以闭环恢复，而不只是 open-loop 拟合专家。

### 1.3 第二阶段目标

在第一阶段 Plan Head 闭环能力达标后：

- 加入真实历史 Memory Bank；
- 加入 ego-conditioned deterministic latent world model；
- 建立 Imagine Memory Bank；
- 自回归 rollout 6 次，每次 0.5 s，形成未来 3 s 轨迹；
- 将 6 个轨迹点插值至 0.1 s，交给 LQR 预瞄跟踪；
- 联合优化 World Model、Plan Head、Value/Risk Head；
- 在 Plan+WM+LQR 闭环成立后进行正式 Real-ENV PPO，争取超过 Expert；
- 最后再评估 imagined RL 或 real-env-to-WM distillation。

### 1.4 明确非目标

第一版不做：

- 图像输入或视觉 Backbone；
- 多候选轨迹 K-sampling；
- 多模态生成式 World Model；
- Diffusion World Model；
- 车道级导航或外置推荐车道；
- 由规则模块替网络完成收费站选道、路口选道或绕障；
- 用 open-loop ADE/FDE 替代闭环性能判断；
- 一次性将全部模块同时上线。

### 1.5 统一时间合同

```text
env/control_dt = 0.1 s
policy_dt = 0.5 s
memory_dt = 0.5 s
wm_dt = 0.5 s
rollout_steps = 6
rollout_horizon = 3.0 s
```

每个真实 policy decision 产生一个 0.5 s 轨迹段，由 Exact Tracker 或 LQR 在内部执行 5 个 0.1 s 控制步；执行完该段后再读取真实 observation 并重新决策。禁止训练阶段使用 0.1 s 重规划、部署阶段改成 0.5 s 重规划。六步 imagined rollout 仅用于 3 s preview，不代表真实环境开环执行 3 s。

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

### 2.3 因果约定

每个 imagined step 的因果关系必须是：

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

禁止实现成“先无条件预测下一环境，再决定同一步动作”。World Model 必须显式接受 ego motion condition。

---

## 3. 输入数据合同

## 3.1 Object Data，OD

每个 OD token 至少包含编码前的结构化字段：

- stable object/track ID，仅用于对齐、监督和诊断；
- object type：车辆、行人、锥桶、障碍物、收费站动态设施等；
- taxonomy fields：`semantic_type / mobility_type(dynamic|static) / geometry_type / state_type`；
- 相对 ego 的位置、航向、速度、加速度；
- object size；
- visibility/validity；
- observation age；
- 可选的 turn signal、brake state 等 MetaDrive 可稳定提供的信息。

原则：

- Planner 不直接读取裸 ID 做策略逻辑；
- ID 用于跨帧对齐和监督；
- 所有环境语义字段先进入 OD Encoder；
- 不允许以规则形式直接告诉 Planner “哪辆车需要避让”。
- 静态设施不得进入动态 WM transition；动态闸机/交通灯状态进入 RoadState token。

## 3.2 Lane/Road Data，LD

LD 包括：

- lane centerline polyline；
- lane boundary polyline 和 boundary type；
- predecessor、successor、left、right topology；
- road segment ID；
- lane ID、boundary ID 或稳定 polyline ID；
- lane type、road type；
- junction、ramp、fork、merge；
- toll lane/toll booth static geometry；
- speed limit；
- traffic light 等动态道路状态应单列为 dynamic road-state token。

LD 对齐不能只靠 lane ID。如果同一 lane 每帧被局部裁剪，必须增加：

- world-coordinate anchor；或
- 固定弧长 segment index；或
- 可复现的 polyline chunk ID。

否则同一 lane ID 的 latent delta 会混入裁剪窗口变化。

## 3.3 Road-level Navigation

允许输入：

- 当前 road segment；
- 后续若干 road segments；
- road graph topology；
- destination relation；
- route progress；
- 下一道路在拓扑上的方向关系。

严禁输入：

- target lane ID；
- recommended lane；
- lane-change command；
- 专家使用的 lane-level route；
- 专家参考中心线；
- 由规则模块提前算出的收费站入口选择。

允许网络看到 lane-to-road topology，因为这是环境事实；不允许直接告诉网络应选择哪条 lane。

## 3.4 Ego Kinematics

至少包括：

- longitudinal/lateral velocity；
- longitudinal/lateral acceleration；
- yaw、yaw rate；
- steering angle 或等价状态；
- vehicle length、width、wheelbase；
- previous executed action/control；
- 当前 pose 只用于坐标变换，网络输入使用规范化相对表示。

## 3.5 控制元数据与语义信息边界

以下信息不属于驾驶策略，不应强制编码后再使用：

- scene reset signal；
- valid/padding mask；
- timestamp 和 delta_t；
- REAL/IMAGINED source flag；
- rollout depth；
- batch/sample index。

这些是张量管理、因果和安全正确性所需的控制元数据，可作为 mask、索引或控制信号直接使用。

所有环境语义信息必须经过 Encoder 后进入下游；不得以显式规则旁路计划网络。

---

## 4. 坐标系、对齐与时序表示

## 4.1 Ego-motion compensation

所有历史 OD/LD 在时序融合前必须对齐到当前时刻 ego frame，但**坐标补偿作用于 transform metadata / geometry anchor，不直接作用于普通 learned content latent**。普通 MLP、GNN 或 Attention 产生的 content latent 默认不具备可解释的 SE(2) 变换规则，禁止把 latent 向量当作二维坐标直接旋转或平移。

每个 OD/LD token 明确拆为：

```text
content latent
+ transform metadata / geometry anchor
+ stable ID / timestamp / valid mask
```

其中：

- `content latent`：类别、尺寸、限速语义、拓扑语义、局部运动语义等 Encoder 表征；
- `geometry anchor`：OD 的中心点/朝向/包围盒锚点，LD 的 polyline control points/切向/边界锚点；
- `transform metadata`：写入 Bank 时的 ego pose、坐标系标识、时间戳，以及完成重投影所需的刚体变换信息；
- Nav 不做普通动力学预测；道路级路由作为静态或拓扑条件保留，仅在路段切换、路由进度变化或仿真器真实更新时刷新。

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

这样将自车平移和旋转从环境变化中解耦，同时避免假设任意 latent 天然可变换。允许计算 content latent residual，但它只是学习特征差，不承担坐标补偿职责。

必须写独立单元测试，覆盖：

- 纯直行、纯旋转、同时平移旋转；
- 静态锥桶、收费站或车道点经补偿后在当前 ego frame 的位置正确；
- 动态车辆的补偿后位移与相对速度方向一致；
- 同一实体改变全局原点或全局朝向后，Planner 输入的相对几何保持一致；
- content latent 不进入手写 SE(2) 旋转/平移代码；
- Nav 未被当作普通动态实体进行未来状态外推。


## 4.2 按稳定 ID 对齐后构造时序特征

不使用“所有 OD/LD 压成单个全局 latent 后相减”的方案。采用实体级稳定 ID 对齐，并把内容变化与几何变化分开表示：

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

建议实体时序输入：

\[
h_{i,t-k}=\left[z^c_{i,t},\ z^c_{i,t-k},\ z^c_{i,t}-z^c_{i,t-k},\ g_{i,t},\ g_{i,t-k\rightarrow t},\ g_{i,t}-g_{i,t-k\rightarrow t},\ \Delta t,\ m_{i,t-k}\right]
\]

其中：

- \(z^c\)：content latent，不接受手写坐标变换；
- \(g\)：由 geometry anchor 和 transform metadata 构造的当前 ego frame 几何表征；
- \(z^c_{i,t}-z^c_{i,t-k}\)：可选 content residual，用于表达语义或运动状态变化；
- \(g_{i,t}-g_{i,t-k\rightarrow t}\)：显式几何变化，负责表达补偿后的实体相对运动；
- \(m\)：valid/birth/death/source mask。

基线可先关闭 content residual，仅使用“当前 content latent + 历史 content latent + 显式几何变化”，再通过 ablation 判断 latent residual 是否提供稳定闭环收益。不得只输入 delta，否则会丢失绝对距离、类别、尺寸、基准速度和道路位置。


## 4.3 实体 birth/death

必须区分：

- 当前存在、历史存在：正常 delta；
- 当前存在、历史不存在：birth mask，不得用零 latent 假装历史实体；
- 当前不存在、历史存在：作为 death/exit 状态供 WM 监督或诊断，是否保留给 Planner 由配置决定；
- 当前和历史都不存在：padding mask。

禁止把缺失实体直接填零后参与普通减法并当作真实运动。

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

即先做实体内时序，再做当前时刻实体间空间交互。

不采用所有 6 帧全部 OD/LD token 的全量时空 self-attention，以控制显存、参数和调试难度。

---

## 5. Memory Bank 设计

## 5.1 Real Memory Bank

- 容量：6 帧；
- 帧间隔：0.5 s；
- 覆盖历史：固定保存 \(t-2.5\) s 至 \(t\) 共 6 帧，**包含当前帧**；dataset、online rollout 和 test 必须使用完全相同的索引语义；
- episode 初始不足 6 帧时可复制最早真实帧完成定形，但复制槽位必须 `valid=0`，不得伪装成真实历史；
- 场景切换必须 hard reset；
- Bank 只接收真实 MetaDrive ENV 观测经 Encoder 得到的 per-entity token record，不写入 imagined state，也不存储 StateFusion 之后的全局 fused context；
- 写入 content latent 前默认 detach，避免计算图跨真实环境 step 无限增长；
- 同一训练 batch 若需要 BPTT，使用显式 sequence tensor/unroll，不依赖长期在线 Bank 的历史计算图。

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

约束：

- `content_latent` 是主要学习表示；`geometry_anchor + transform_metadata` 是坐标补偿和 ID 对齐所需的最小结构化元数据，不属于外部规划规则或车道级导航特权；
- OD anchor 建议保留中心点、朝向、长宽和速度锚点；LD anchor 建议保留固定数量 polyline control points、切向和拓扑 ID；
- Road-level Nav 只给道路/路由级意图，不给目标车道 ID、推荐变道方向或专家未来轨迹；
- Ego 显式动力学用于 causal conditioning、几何补偿和诊断，不要求 WM 从 latent 中反推 ego pose；
- 必须为 Bank 定义固定 shape、padding 和 mask，避免 Python list 导致训练/部署语义不一致。


## 5.2 Imagine Memory Bank

- rollout t0 从 Real Memory Bank 复制；
- 复制后不与 Real Memory Bank 共享可变写入；
- 每次 World Model 预测下一帧 `content latent + predicted/propagated geometry anchor + transform metadata` 后执行 FIFO append；
- rollout 结束后立即 reset；
- 每个元素带 `source=REAL/IMAGINED` 和 `rollout_depth=0..6`；
- 静态 LD/Nav 优先由 t0 真实结构和 imagined ego pose 做解析重投影/传播，不要求 WM 从零生成静态地图；动态 OD 与必要 road-state 由 WM 预测；
- forbidden：imagined content latent、geometry anchor 或 metadata 回写 Real Memory Bank。

## 5.3 缓存与 Encoder 版本

训练集、DAgger Buffer 和 PPO Replay/Rollout Buffer 优先保存**原始结构化 observation 与必要的 action/reward/mask**，训练取样时再用当前 Encoder 计算 latent，不把 latent 作为可跨多轮训练长期复用的唯一数据。原因是 Encoder 会随训练更新：同一帧 observation 经旧 Encoder 得到的 `z_old` 与经新 Encoder 得到的 `z_new` 可能处于不同表征基底；若把旧缓存 latent 与当前新算 latent 放入同一个 6 帧窗口，时序差分可能主要反映 Encoder 参数变化，而不是真实环境变化。

这不影响 Real Memory Bank 的在线运行：单个 episode 内，Bank 保存最近 6 帧且写入时 detach；其生命周期短，允许使用当前参数快照的 Encoder 输出。训练参数更新发生在 batch/rollout 边界，不得在同一个 6 帧窗口构建过程中更新 Encoder。

若为吞吐性能必须离线缓存 latent：

- 每条缓存记录 `encoder_version/hash`，对应生成 latent 的 Encoder checkpoint 或权重哈希；
- 训练加载时只允许使用与当前 Encoder 版本一致的缓存，否则重新编码原始 observation 或使缓存失效；
- 禁止在同一时序窗口、同一 Real/Imagine Bank 中混用不同版本 latent；
- 若 Encoder 在某阶段完全冻结，可为该冻结阶段一次性重建 latent cache；Encoder 再次解冻后，旧 cache 自动失效；
- 部署推理时 Encoder 权重固定，因此不存在跨训练版本漂移，但场景 reset 和 Bank 隔离规则仍须保持。

---

## 6.---

## 6. Plan Head 与轨迹定义

## 6.1 每次 policy query / imagined step 的输出

每次 Plan Head query 只输出下一 0.5 s ego motion；同一个 Plan Head 参数实例贯穿 BC、DAgger、PPO 和 WM rollout。真实环境每 0.5 s 决策一次，Exact Tracker 或 LQR 在段内执行 5 个 0.1 s 控制步；下一决策时刻使用新 observation 重规划。

第一版主动作定义为带终端速度边界的 SE(2) endpoint：

\[
a_n=(\Delta x_n,\Delta y_n,\Delta\psi_n,v_{n+1})
\]

其中：
- `(Δx, Δy, Δψ)` 位于当前 real/imagined ego frame；
- `v_{n+1}` 是 0.5 s 后目标速度，不得用 `distance/0.5` 的平均速度替代；
- pose endpoint 决定空间边界，`v_{n+1}` 决定纵向时间边界，二者不是冗余量；
- 可等价改用 `(Δx, Δy, Δψ, Δv)`，但必须在 P0 写入 decision log 后全流程统一；
- `(ds,dtheta)` 仅保留为对照基线，不作为长期主动作。

P0 将 4D `(dx,dy,dpsi,v_next)` 作为固定基线，只校准标签、bounds、归一化、Hermite 插值和动力学可行性；不得根据 `v_next` 与平均速度的相关性自动退化为 3D。若 action round-trip、full-stop/stop-to-go 或 Expert-GT LQR 出现结构性失败，则暂停 P1，提交 decision log 人工复审，不允许脚本静默改动作维度。至少报告 endpoint/heading/terminal-speed、conditional terminal-speed residual、acceleration、jerk、curvature/curvature-rate、full-stop、stop-to-go、LQR tracking error 和控制饱和率。

必须从相邻点和连续 0.5 s 决策段派生并约束：
- heading / `Δψ`；
- curvature / `κ`；
- longitudinal/lateral acceleration；
- jerk；
- curvature rate；
- 相邻真实决策段连接处的位姿、速度、航向和曲率连续性。
- 第 `n+1` 段起始速度硬继承 Ego Dynamics 推进后的第 `n` 段终态 `v_{n+1}`；平顺性使用 soft loss + feasibility gate。

Exact Tracking 只替代早期执行器，不豁免轨迹动力学约束。即使仿真器能够瞬时落到目标点，也不得把横移过大、航向突变、曲率不连续或速度跳变的轨迹判为可用。

## 6.2 坐标累计

每个点来自不同 imagined ego frame，不能直接拼接。必须把每一步相对运动累计到 rollout-t0 ego frame：

\[
P_{n+1}=P_n\oplus\Delta P_n
\]

最终生成：

\[
\tau=\{(x_i^{ego@t0},y_i^{ego@t0},v_i)\}_{i=1}^{6}
\]

单元测试必须验证：

- 六次直行累计后位置正确；
- 连续转弯时坐标旋转正确；
- 输出点都在同一个 t0 坐标系；
- 轨迹转换到 world frame 后和 ego dynamics rollout 一致。

## 6.3 插值与 LQR 输入

6 个点对应未来 0.5、1.0、1.5、2.0、2.5、3.0 s。进入 LQR 前：

- 插值为 0.1 s；
- 推荐使用带切线约束的 cubic Hermite 或其他不会明显过冲的插值；切向方向由 heading 决定，切向大小由速度边界决定，禁止仅凭 `Δψ` 任意设定；
- 速度使用 `v_n → v_{n+1}` 做单调/限幅插值；
- 计算 heading、curvature、acceleration、jerk；
- 对不连续曲率、逆行、速度负值、超动力学限制做 reject 或修正；
- 保留完整 2 至 3 s 参考轨迹供 LQR 预瞄，不得只传第一个 0.5 s。

## 6.4 Action semantics 一致性

BC、DAgger、RL、World Model rollout 必须使用同一 Plan Head 输出语义。禁止出现：

- BC 输出轨迹点，RL 改为 steering/throttle；
- DAgger 用 \(x,y,v\)，WM 阶段改成 \(ds,d\theta\)；
- train 和 inference 使用不同坐标系。

MetaDrive 原生 steering/throttle 仅作为下游控制执行接口，不作为 Planner 的主要动作定义。

### 6.5 PPO 概率动作合同

PPO 要求可计算 `log_prob(action | state)`。因此 Plan Head 不是只输出确定性三元组，而是输出受约束轨迹动作分布：
- 对归一化 `(dx, dy, dpsi, v_next)` 输出 `mu, log_std`，或使用 squashed Gaussian 后映射到动力学边界；
- BC/DAgger 使用 expert action 的 NLL，并保留 point/velocity/feasibility auxiliary loss；
- PPO buffer 保存 sampled pre-squash action、executed feasible action、old log-prob、old value 与修正/裁剪标记；
- 若 feasibility layer 改写了动作，必须避免把“采样动作的 log-prob”错误地对应到“被大幅改写后的执行动作”。基线应让分布本身落在可行域，并统计 post-process rate；
- Value Head 输入 state/context，不输入未来 GT；Risk/Cost Value 与 reward Value 分头训练。
正式 PPO 中：当前 `a0` 从策略分布采样并记录 `log_prob`，执行完整 0.5 s 后形成一个 transition；`a1...a5` 使用各 imagined step 分布均值，仅作为非执行 preview，不计入当前 `old_log_prob`，下一真实决策时刻重新采样新的 `a0`。不得把 preview tail 对真实控制造成的影响算入一个未覆盖它的 log-prob。


---

## 7. Ego Dynamics 与 Latent World Model

## 7.1 模块职责分离

Ego Dynamics：

- 判断或推进“自车能否以及如何从当前状态走到该轨迹段”；
- 计算 ego pose、heading、velocity、curvature 等下一状态；
- 为坐标变换和 World Model 提供 ego motion condition。

Latent World Model：

- 在给定当前 imagined context 和 ego motion 的情况下，预测周围环境下一帧 latent；
- 不负责替代完整车辆动力学；
- 不负责生成导航目标；
- 不负责保证 Plan Head 轨迹可跟踪。

LQR：

- 使用完整 3 s 插值轨迹进行预瞄跟踪；
- 将参考轨迹转换为 MetaDrive 可执行控制。

## 7.2 World Model 条件

WM 不应只看到 0.5 s 终点。条件至少包括从轨迹段确定性派生的：

\[
c_n=(\Delta x,\Delta y,v_n,v_{n+1},\Delta\psi,\kappa,a_{lon})
\]

其中：

\[
a_{lon}=\frac{v_{n+1}-v_n}{0.5}
\]

这些量由 Plan Head 输出和当前 ego state 计算，不需要另开互相矛盾的动作头。
每段至少 enforce：`v_next>=0`、`a_min <= (v_next-v_n)/0.5 <= a_max`、`|kappa|<=kappa_max`、`|v^2*kappa|<=a_lat_max`；低速 stop/hold 属于 Controller；记录 clip/reject/post-process 和控制饱和率。

## 7.3 World Model 输出拆分
分类合同：仅 dynamic OD/RoadState 进入 learned transition；Static OD/infrastructure/LD 做 geometry reprojection + shared Encoder re-encode；Nav 按 route progress 更新。交通灯 schema 保留，首版可默认关闭。

动态预测主体：

- OD latent；
- dynamic road-state latent，例如 traffic light；
- 可选的 occupancy/risk latent；
- uncertainty scalar/head。

显式推进或重新编码：

- Ego latent：由 ego dynamics 推进后编码；
- Static LD：依据 predicted ego pose 做坐标变换，可预测小 residual，但不应从零重建地图；
- Road-level Nav：依据 route graph 和 predicted route progress 更新/重新编码，不做普通动力学幻想。

## 7.4 确定性模型与不确定性

第一版采用 deterministic World Model，不做多模态未来或 K 候选轨迹。

建议保留轻量 uncertainty head，用于：

- 识别 out-of-distribution imagined state；
- 统计 rollout depth 1 至 6 的误差增长；
- 在 imagined RL 中衰减或截断高不确定性样本；
- 防止 Policy 利用 WM 明显错误。

uncertainty head 不改变单轨迹主方案。

---

## 8. Encoder 与网络规模建议

### 8.1 总参数约束

部署推理主网络目标约 100 万参数，允许工程上约 0.8M 至 1.2M 的可解释浮动。EMA Target Encoder 属于 training-only teacher copy，必须单独统计，不计入 deployable parameter budget，但其显存和 optimizer/activation 开销仍需报告。必须在每次结构修改后记录：

- deployable total params 与 training-only target params；
- trainable params；
- peak allocated memory；
- single-step latency；
- six-step rollout latency。

### 8.2 参考预算

- OD Encoder：约 60k 至 100k；
- LD/polyline Encoder：约 100k 至 150k；
- Nav Encoder：约 20k 至 40k；
- Ego Encoder：约 10k 至 30k；
- Temporal Fusion：约 120k 至 200k；
- Spatial/Cross Attention：约 150k 至 220k；
- Plan Head：约 80k 至 120k；
- Value/Risk Head：约 30k 至 70k；
- Latent World Model：约 180k 至 260k。

推荐起点：

- hidden dimension：96；必要时 128；
- attention heads：4；
- temporal layers：1 至 2；
- spatial layers：1 至 2；
- max OD tokens：16 至 32；
- max LD/polyline tokens：24 至 48；
- polyline 内先做 point MLP/PointNet pooling，再做实体间 attention。

### 8.3 初始化

- ReLU 类 MLP：Kaiming；
- GELU/attention linear projection：Xavier；
- LayerNorm：weight=1，bias=0；
- 最终 trajectory residual/output head 可使用小方差初始化，但禁止整个分支双零初始化；
- 每个模块必须提供初始化统计 probe，报告 weight mean/std 和输出分布。

---

## 9. 监督目标与 Loss

## 9.1 Plan Loss

每个 rollout step 对 0.5 s motion 监督；六步后再对完整 3 s 轨迹做一致性监督。

建议组成：

\[
L_{plan}=L_{point}+\lambda_vL_v+\lambda_hL_{heading}+\lambda_cL_{curvature}+\lambda_sL_{smooth}
\]

其中：

- point loss：x/y 或 delta x/delta y；
- velocity loss；
- heading consistency；
- curvature/radius feasibility；
- acceleration、jerk、curvature-rate smoothness。

不允许只用单一 XY MSE 作为全部轨迹监督。

## 9.2 World Model Loss

真实下一帧 observation 经 Target Encoder 得到 latent GT：

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

不能只报告 aggregate latent MSE。至少按：

- modality；
- entity type；
- rollout depth；
- distance bin；
- static/dynamic；
- seen/unseen map；

拆分统计。

## 9.3 Value/Risk Supervision

Value Head 预测：

- expected discounted return；
- route completion probability；
- time-to-goal 或 progress；
- optional cost value。

Risk Head 至少预测：

- collision risk；
- off-road risk；
- wrong-way/wrong-road risk；
- stuck risk；
- comfort violation risk。

建议 reward value 和 safety cost value 分开，避免成功奖励抵消安全成本。

## 9.4 Representation 防坍缩

只做 latent regression 可能出现 Encoder 与 WM 共同缩小特征幅度。必须至少具备：

- EMA Target Encoder；
- stop-gradient target；
- latent variance/statistics monitor；
- 每模态 decoder 或 auxiliary task 中至少一种；
- planner task loss 持续约束 latent 的任务相关性。

可选辅助任务：

- OD type/relative state reconstruction；
- LD topology classification；
- route-progress prediction；
- ego-motion prediction；
- next-state contrastive discrimination。

---

## 10. Detach 与梯度合同

## 10.1 默认 detach 规则

应 detach：

- Future GT latent；
- EMA Target Encoder 输出；
- Real Memory Bank 跨真实 env step 的写入；
- Replay/DAgger/PPO Buffer 中保存的数据；
- PPO old log-prob 和 old value；
- 场景采集阶段的 observation/action；
- 默认情况下，早期联合训练时 Plan action 进入 WM 前可 detach，防止 Plan Head 利用 WM 漏洞。

不应 detach：

- 当前 batch 内 Encoder → Planner → Plan Loss；
- 当前显式 WM unroll 内 WM recurrent path；
- WM output → Value/Risk Head；
- RL Policy Head → policy loss；
- 允许联合调整时的 Encoder top layers/adapters。

## 10.2 必须配置化的梯度开关

```yaml
detach_real_mem_write: true
detach_future_target: true
detach_wm_input: false
detach_wm_recurrent_path: false
detach_wm_to_value: false
detach_plan_action_to_wm: true
detach_encoder_from_wm_loss: true
use_ema_target_encoder: true
```

训练后期只有在 WM 通过六步 fidelity 门槛后，才允许对照实验：

```yaml
detach_plan_action_to_wm: false
detach_encoder_from_wm_loss: false
```

开放后必须监控：

- Policy 是否找到 WM 幻觉漏洞；
- imagined return 上升但 real-env return 下降；
- action 分布是否越过训练数据范围；
- latent norm 是否异常漂移。

## 10.3 EMA Target Encoder

Target Encoder 不参与反向传播。采用：
D1/D2 冻结 Encoder 时 target 固定，EMA 为 no-op；D2 增加 latent variance/effective-rank gate；Stage E 解冻 adapter 后才启用 EMA，并监控 target-online drift。

\[
\theta_{target}\leftarrow m\theta_{target}+(1-m)\theta_{online}
\]

建议起点：

- 初期 \(m=0.99\)；
- 稳定后可增加至 0.995 或 0.999；
- 具体通过 target drift 与 WM 收敛速度决定，不能只看 loss 更低。

---

## 11. Expert、BC 与 DAgger

## 11.1 Expert 定义

第一版唯一可用 Expert 为 IDM + Pure Pursuit 或可复现的等价组合。接受其不是所有场景的强专家，也不把“缺少更强专家”作为阻塞项：IL/DAgger 的目标是让 Student 接近并吸收可用示范，最终依靠真实环境 RL 和闭环筛选争取超过 Expert。但必须：

- 记录 expert success、collision、off-road、wrong-route、stuck；
- 失败 expert episode 不得无条件作为正样本；
- 保存 expert 是否使用 lane-level 特权信息的审计结果；
- 将 expert 轨迹统一转换为 Student 的轨迹表示。

## 11.2 特权信息审计

必须排查 MetaDrive Expert/IDM 路径：

- 是否读取 target lane；
- 是否读取 lane-level route；
- 是否读取学生不可见的未来交通状态；
- Pure Pursuit 参考点是否来自 lane-level centerline；
- expert lane-change decision 的信息来源。

若 Expert 使用 lane-level oracle：

- 允许其继续作为数据生成器，不要求先替换 Expert；
- 禁止将该 oracle、target lane 或由其派生的推荐方向输入 Student；
- 数据中标记 `privileged_expert_sample`，用于分桶评测和 loss weighting，不作为 Planner 语义输入；
- 对分岔、路口、收费站和多条合法车道场景，禁止仅用单一轨迹 XY-MSE 强迫复制专家所选车道；应降低单轨迹 L2 权重，并辅以合法道路区域、路线进度、轨迹可行性和安全性监督；
- 不要求 BC 在不可辨识的多解场景逐点复现 Expert，重点检查是否选择了合法、可达且安全的解；
- 由 DAgger 补充 Student 访问状态下的恢复示范，由真实环境 RL 最终解除对专家车道选择和次优行为的绑定，并以冻结验证集闭环结果判断是否超越。

## 11.3 DAgger

每轮在 Student 访问到的状态上查询 Expert，并把恢复样本加入数据集。

参考混合系数：

```text
1.0 → 0.7 → 0.4 → 0.2 → 0.0
```

但实际以闭环安全和数据覆盖决定。下降由 Student success、collision/off-road、Expert intervention、recovery、LQR shadow 和 failure-buffer 覆盖率触发，不按固定轮次机械下降。

重点采集：

- 偏离车道后的恢复；
- 错过换道窗口；
- 锥桶邻近；
- 慢车跟驰/绕行；
- 路口犹豫；
- 收费站入口选择；
- 出路边缘状态；
- 低速大曲率；
- expert 与 student 决策分歧大但两者均未立即失败的状态。

必须使用 failure-prioritized buffer，不能只做均匀追加。

---

## 12. RL 设计

## 12.1 训练顺序

真实环境 RL 必须先于 imagined RL。MetaDrive 已提供真实 simulator，不应在 WM 尚未可靠时引入 model bias。

第一版 RL 推荐 PPO，保持 Plan Head action semantics 不变。

## 12.2 Reward 与 Cost

Reward 至少包含：

- route progress；
- success；
- efficiency；
- comfort；
- lane/route compliance。

Cost 至少包含：

- vehicle/object collision；
- off-road；
- wrong-way/wrong-road；
- severe overspeed；
- unsafe proximity；
- stuck；
- excessive lateral acceleration/jerk。

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

同时分开记录 reward 和 safety cost，不能依靠一个加权总分掩盖碰撞。

## 12.3 防 Reward Hacking

必须排查：

- 靠高速冲线抵消碰撞；
- 原地左右摆动刷 progress；
- 切角获得 route progress；
- 停车规避风险但无法达成；
- 借 off-road 缩短路线；
- 沿错误道路获得局部前进奖励。

所有 progress 应基于合法 route arc-length，而不是 world displacement。

---

## 13. 分阶段训练流程与门槛

## Stage 0：环境、Schema 与 Expert 基线

实施：

- 固定结构化 observation schema；
- 固定 train/validation/test 场景划分；
- 建立 Expert 收集、回放和有效性过滤；
- 完成坐标变换、reset、nav 泄漏、轨迹转换测试。

通过条件：

- 所有 schema/unit tests 通过；
- Expert 闭环指标可重复；
- 验证集完全冻结；
- 没有 train/validation map seed 或 episode 泄漏；
- Expert 特权信息审计报告完成，并明确哪些样本属于 privileged/multi-solution 场景及其监督降权方式。
- P0 动作数据按 lane keeping、curve、full stop、stop-to-go、slow following、obstacle braking、lane change、junction、fork/merge、toll-gate 分桶；低速、急减速、大曲率设最低配额，数值写 decision log。

## Stage A1：单帧 BC + Exact Tracking

网络：

- Encoders；
- Spatial Fusion；
- Plan Head；
- 可选 Value/Risk auxiliary heads；
- 不使用 Memory Bank 和 WM。

目的：

- 验证 action/trajectory 定义；
- 验证 road-level nav 能被网络正确使用；
- 验证 basic closed-loop；
- 提前验证轨迹插值、坐标转换、车辆模型与 LQR 接口，避免到 Stage F 才首次暴露控制不可跟踪问题。

本阶段以 Exact Tracking 为 Planner 主评测，同时并行运行两类 LQR canary：

1. **Expert GT 3 s trajectory + LQR**：验证插值、坐标系、速度边界、车辆模型和控制接口，只评价 Controller，不评价 Planner；
2. **Plan Head 0.5 s segment + LQR**：将当前单段插值为 5 个 0.1 s reference points，只做接口、可跟踪性和控制饱和率 canary，不作为正式闭环成绩。

无 WM 时不得声称 Plan Head 能因果生成完整 3 s LQR preview。

必须报告：

- open-loop point/velocity errors；
- exact-tracking closed-loop success；
- collision/off-road/wrong-route/stuck；
- heading、curvature、acceleration、jerk、curvature-rate 与动力学可行率；
- 连续 0.5 s policy decision 段连接处的位姿/速度/航向/曲率跳变量；
- Expert GT + LQR 的 tracking error、steering/brake/throttle saturation 与失败类型；
- Plan Head 单段 + LQR 的 tracking error、可跟踪率、控制饱和率，以及 Exact 成功但 LQR canary 失败的比例；
- 按场景类型拆分结果。

若 Exact Tracking 闭环成功但轨迹可行率、连续帧平顺性或 LQR canary 不达标，应先修正 Plan Head/Plan Loss、插值或 Controller，不得等到 Stage F 才处理。Stage F 是 LQR 首次进入完整 3 s 正式执行链路，不是第一次测试 LQR。

## Stage A2：真实历史 Memory Bank

网络增加：

- 6-frame Real Memory Bank；
- ego-motion compensation；
- stable-ID alignment；
- entity temporal fusion。

必须做同 seed ablation：

```text
single frame
2 frames
4 frames
6 frames
```

每个候选 checkpoint 除 Exact Tracking 外，还必须运行固定 seeds 的 **Plan Head 0.5 s segment + LQR canary**，比较：

- Memory-on/off tracking error 与可跟踪率；
- steering/brake/throttle saturation；
- 曲率、速度和航向在连续决策段连接处的跳变量；
- Exact 指标提升但 LQR canary 退化的比例。

若历史模型没有稳定超过单帧，或 Exact 提升但 LQR 可跟踪性显著退化，禁止进入 WM 阶段，优先查：

- frame timestamp；
- reset；
- ID churn；
- coordinate transform；
- birth/death mask；
- LD segment alignment；
- attention mask。

## Stage B：DAgger + Exact Tracking

- 以 A2 最佳 checkpoint 初始化；
- 逐轮降低 Expert 混合比例；
- failure-prioritized aggregation；
- 每轮固定验证集闭环评测；
- 保持 Expert 原始数据与 DAgger recovery 数据的来源标签；
- DAgger 主数据采集仍使用 Exact Tracking，以隔离 Controller 误差；
- 每轮 checkpoint 必须在固定 seeds 上运行 Plan Head 单段 LQR shadow evaluation，不参与采集但作为推进门槛。

进入下一阶段的最低要求：

- Student exact-tracking success 接近 Expert；
- collision/off-road 明显下降并稳定；
- 复杂场景不因普通道路提升而退化；
- 冻结验证集连续多个 seed 改善，而非单次偶然值；
- LQR shadow 的 tracking error、可跟踪率和控制饱和率不得相对 A2 最佳 checkpoint 显著退化；若 Exact 提升但 LQR 持续恶化，不得进入 Stage C。

## Stage C：短 Real-ENV PPO Canary，无 WM
- 从最佳 DAgger checkpoint 初始化；
- 保持同一 `(dx,dy,dpsi,v_next)` 概率动作；
- 每 0.5 s 采样并执行一个 `a0`，段内运行 5 个 0.1 s 控制步；
- PPO 的 value/GAE/return 均来自真实 MetaDrive transition；
- PPO transition 基线继续使用 Exact Tracking，以减少 WM 与 Controller 变量干扰；
- 只进行足以验证 PPO、GAE、reward/cost、buffer、checkpoint 和遗忘控制的短 canary；
- 保留 BC/DAgger auxiliary loss 或 replay regularization；
- 每个主要 checkpoint 必须在同一组固定 seeds 上运行 Plan Head 单段 LQR shadow evaluation；
- 不以本阶段是否超过 Expert 作为推进 WM 的硬条件。

通过条件：训练无 NaN、log-prob 与 executed action 合同正确、reward/cost 无明显 hacking、冻结验证集安全指标不灾难性退化；同时 LQR shadow 的可跟踪率、控制饱和率和舒适性不得显著恶化。禁止依靠 Exact 的瞬时执行能力获得不可由 LQR 实现的虚假 reward。

## Stage D1：单步 World Model

输入：真实历史 + Expert/Student/PPO canary 实际执行 ego motion。WM target 是真实下一帧 observation 经 EMA Target Encoder 得到的 per-entity latent token，不是融合后的 scene context，也不是 WM 自身输出。

数据覆盖：必须混合 Expert、DAgger Student、恢复扰动和真实 PPO policy 的已执行 motion；按 `(dx, dy, delta_yaw, acceleration, curvature)` 做覆盖率与 OOD 报告。只用 Expert action 达标不算 WM 可用于 Policy rollout。

训练策略：

- 冻结或基本冻结已验证好的 Planner/Encoder；
- Encoder 冻结时 Target Encoder 固定，记录 target drift=0；
- 先训练 one-step latent prediction；
- 对动态 OD、ego、road-state 分模态监督。

必须建立基线：

- persistence predictor；
- constant-velocity predictor；
- optional simple MLP transition。

WM 必须明显超过简单基线，才能进入多步阶段。

## Stage D2：多步 World Model

课程：

```text
1-step → 2-step → 3-step → 6-step
```

区分三种实验：

1. GT state + GT action teacher forcing；
2. predicted state + GT action；
3. predicted state + Plan Head action。

这样可以分别定位：

- WM transition error；
- state autoregression error；
- policy-WM coupling error。

必须按 rollout depth 画误差曲线，并监控 latent norm/std/effective rank；失真或 rank gate 不通过不得进入 imagined RL。

当 WM 已能生成 1/2/3/6 步 preview 时，同步进行递增时域 LQR shadow evaluation：先验证 0.5 s，再验证 1.0/1.5/3.0 s。该评测只诊断轨迹拼接、插值和 Controller 可实现性，不参与 WM latent loss。

## Stage E：Planner + WM 联合磨合

初期：

- 冻结 Encoder 主体；
- 训练 WM、Plan Head、Value/Risk Head；
- Plan action 进入 WM 前 detach；
- Target Encoder 始终 stop-gradient；
- 此阶段允许 `WM output → Plan/Value` 的梯度更新下游 Head，但默认不允许 PPO/Plan loss 反向改写 WM；只有 real-env 闭环 A/B 证明收益后，才逐层开放 adapter。

后期可选：

- 只解冻 Encoder 顶层 adapter；此时才启用 EMA 更新并监控 drift；
- 降低 WM loss 权重；
- 通过 A/B 决定是否开放 action-to-WM 梯度。

本阶段必须把 Plan Head + WM 生成的完整 3 s preview 接入 LQR shadow evaluation，并与 Exact/kinematic rollout 同 seed 对照。只有 3 s 轨迹拼接、插值、tracking error、控制饱和率和安全指标达到门槛，才进入 Stage F 的正式 Controller-realized 执行。

联合损失参考：

\[
L=L_{plan}+\lambda_{wm}L_{wm}+\lambda_vL_{value}+\lambda_rL_{risk}+\lambda_{aux}L_{aux}
\]

\(\lambda_{wm}\) 初始可从 0.05 至 0.2 搜索，不得让 latent loss 主导 Planner 闭环目标。

## Stage F：完整 3 s Plan+WM+LQR 正式闭环 + Real-ENV PPO

Stage F 不是第一次测试 LQR，而是 LQR 首次成为完整 3 s Plan+WM 轨迹的**正式执行器和最终成绩口径**。

- 6 个 0.5 s 点插值至 0.1 s；
- 使用完整 3 s 预瞄参考；
- 评估 tracking error、curvature、steering、acceleration、jerk；
- 与 exact tracking 结果并列报告；
- 在 Plan+WM+LQR 推理链稳定后，从最佳联合 checkpoint 进行正式 Real-ENV PPO + LQR；
- 正式 PPO 才承担在安全门槛内超过 IDM + Pure Pursuit 的目标，value/GAE/return 仍全部来自真实 MetaDrive transition。

必须区分：
- planner-oracle closed loop；
- controller-realized closed loop。

如果 exact tracking 好而 LQR 差，优先查轨迹可行性和控制器，不要把问题直接归因于 Planner。

## Stage G：Imagined RL / WM Distillation，可选最后阶段

只有满足以下条件才开放：

- 6-step WM fidelity 达标；
- imagined risk 与 real-env failure 有稳定相关性；
- imagined action distribution 未明显 OOD；
- real-env reward 没有因 joint training 下降。

策略：

- imagined loss 按 rollout depth 衰减；
- high uncertainty 时截断；
- REAL 和 IMAGINED 数据严格标记；
- 最终模型始终用 real-env closed-loop 决策，不以 imagined score 单独选 checkpoint。

---

## 14. 评测协议

## 14.1 数据划分
- Train：10,000；其中 `Calib50` 属于训练域，只作快速调试；
- Validation：500，用于 checkpoint、阈值和消融，永久冻结；
- Test：500，仅用于里程碑和最终结论，不调参、不删失败样本；
- P0/P1 可先用 2,000 train / 200 evaluation-scale scenes 做工程 smoke，不替代最终规模；
- 按 scenario/map seed 划分，禁止 train/val/test 泄漏。

## 14.2 核心闭环指标

主指标：

- route success rate；
- route completion；
- collision rate；
- off-road rate；
- wrong-route/wrong-way rate；
- stuck rate；
- timeout；
- comfort violations；
- mean time/distance to failure。

控制指标：

- lateral tracking error；
- heading error；
- speed tracking error；
- peak/mean curvature；
- steering rate；
- longitudinal/lateral acceleration；
- jerk。

WM 指标：

- one-step and 1..6-step latent error；
- entity decode error；
- rollout uncertainty calibration；
- static LD invariance after ego compensation；
- dynamic OD prediction by distance/type；
- imagined-vs-real risk correlation。

## 14.3 场景分桶

至少分为：

- straight lane keeping；
- following and braking；
- obstacle/cone avoidance；
- lane choice before fork；
- merge；
- ramp；
- intersection；
- toll station；
- low-speed sharp turn；
- dense traffic；
- recovery from disturbed state。

总体指标不能掩盖关键场景退化。

## 14.4 Checkpoint 选择

不能只按 training loss 或 open-loop ADE 选择。建议使用带安全门槛的闭环优先级：

1. collision/off-road 不超过硬门槛；
2. route success 最大；
3. route completion 最大；
4. comfort 和效率作为次级排序；
5. open-loop loss 仅作为诊断。

---

## 15. 重要排摸点与 Probes/Hooks

## 15.1 数据与坐标

必须记录/可视化：

- 每帧 ego pose；
- compensation 前后 OD/LD；
- 同 ID 轨迹；
- lane/polyline anchor；
- route graph 与 student-visible nav；
- Expert-visible privileged info；
- frame timestamp 和 scene reset。

关键断言：

- 场景切换后 bank 全空；
- 静态 LD 经 compensation 后跨帧基本一致；
- 同 ID 物体不会跳槽；
- 不同 ID 不会误作同一实体；
- 六个 rollout 点位于同一 t0 坐标系。

## 15.2 Latent probes

每模态记录：

- mean/std/min/max/norm；
- temporal cosine similarity；
- same-ID delta 与 different-ID delta 分布；
- padding token 激活；
- static LD delta；
- ego information leakage 到 OD/LD latent 的线性 probe；
- gradient norm；
- attention entropy；
- top attended entities。

重点判断：

- latent collapse；
- OD/LD latent 仅复制 ego motion；
- Planner 忽视 Nav；
- Planner 只看最近障碍物而不看长程路线；
- World Model 无视 ego condition；
- Token mask 泄漏。

## 15.3 Causality probes

对同一个 Memory Bank：

- 固定环境，替换不同 ego action，WM 输出应显著不同；
- 固定 ego action，扰动关键 OD，Planner action 应合理变化；
- shuffle ego condition 后，WM loss 应恶化；
- 将 ego condition 置零，WM 不应仍得到近乎相同性能；
- 替换 road-level Nav 后，选道应变化，但近距离碰撞响应不应失效。

## 15.4 World Model probes

必须实现：

- one-step teacher-forced decode visualization；
- six-step autoregressive visualization；
- per-depth error；
- GT action vs Plan action rollout 对照；
- real-vs-imagined entity count、position、velocity 对照；
- uncertainty-error reliability curve；
- simple baseline comparison。

## 15.5 Planner probes

必须实现：

- 输出六步轨迹叠加到场景；
- 每一步局部坐标和 t0 累计坐标同时显示；
- route topology 和 chosen lane 可视化；
- 轨迹 feasibility report；
- exact tracking 和 LQR tracking 对照；
- action distribution by scenario；
- Expert/Student disagreement case replay。

## 15.6 梯度 probes

每个训练阶段报告：

- 模块级 gradient norm；
- 哪些模块 requires_grad；
- 哪些 tensor detached；
- EMA update 正常性；
- recurrent rollout graph depth；
- 是否出现跨 episode graph 泄漏；
- plan loss、WM loss、value loss 对 Encoder 梯度贡献比例。

提供自动化 `test_detach_contract.py`，不能只靠代码阅读。

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

`UNKNOWN` 比例必须持续下降。每个关键失败要能用 scene seed 一键复现。

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

---

## 18. 工程约定

## 18.1 配置与可复现性

- 禁止在训练代码散落 magic numbers；
- 超参数统一进入 YAML；
- 每次 run 保存 resolved config；
- 固定 Python、NumPy、PyTorch、MetaDrive 和 environment seed；
- 保存 git commit、dirty status、依赖版本和设备信息；
- 每个实验生成唯一 run ID；
- validation set 配置和 seed 只读，不允许训练脚本改写。
- validation/test 均只读；worker/recycle/memory floor 写 `runtime_local.yaml`，按机器标定，不是架构硬约束。

## 18.2 Shape 与 Schema 约定

每个模块 docstring 必须写：

- 输入/输出 tensor shape；
- 坐标系；
- 单位；
- mask 语义；
- ID 对齐语义；
- 是否 REAL/IMAGINED；
- 是否 detach；
- 时间索引。

对关键 tensor 使用统一命名：

```text
B: batch
T: history length, 6
R: rollout depth, 6
No: max OD entities
Nl: max LD entities
P: polyline points
D: latent dimension
```

## 18.3 单位

全工程统一 SI：

- position：m；
- velocity：m/s；
- acceleration：m/s^2；
- yaw/steering：rad；
- yaw rate：rad/s；
- time：s；
- curvature：1/m。

严禁角度和弧度混用。所有归一化必须可逆并有单元测试。

## 18.4 Mask 约定

必须明确 attention API 中：

- `True` 表示 valid 还是 masked；
- padding mask 与 causal mask 的 shape；
- birth/death mask 的独立语义；
- imagined depth mask。

严禁依赖不同 PyTorch API 的默认 mask 语义。

## 18.5 Logging

每个 run 至少保存：

- losses by component；
- closed-loop metrics；
- scenario bucket metrics；
- model/optimizer LR；
- gradient norms；
- latent statistics；
- rollout depth errors；
- sample videos/plots；
- failure seeds；
- checkpoint selection reason。

## 18.6 Checkpoint

Checkpoint 必须包含：

- online model；
- EMA target encoder；
- optimizer/scheduler；
- scaler；
- training stage；
- global step/epoch；
- RNG states；
- resolved config；
- dataset split hash；
- encoder version。

禁止加载不兼容 checkpoint 后 `strict=False` 静默跳过关键参数。任何 missing/unexpected keys 必须显式记录并审计。

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

影响 Memory/WM/Trajectory 的改动还必须通过对应 causality 和 reset tests。

## 18.8 性能和大张量约束

- 不做 6 帧所有实体点的全连接全量 attention；
- polyline 先局部聚合；
- 记录每层 token 数；
- 禁止隐式 broadcast 产生巨大中间 tensor；
- profiler 中出现异常大张量必须解决后再推进；
- rollout 6 步时严查计算图和显存线性/超线性增长。

## 18.9 代码质量

- 不以临时 if/else 规则修复具体场景；
- 不把场景 seed 写进策略；
- 不隐藏 fallback；
- 所有 fallback 要可计数并记录；
- 核心算法改动必须配设计说明和 ablation；
- 文档结论必须能追溯到代码、配置、run 和 checkpoint。

---

## 19. 必做 Ablation

至少完成以下对照：

1. single frame vs 2/4/6-frame memory；
2. no ego-motion compensation vs compensation；
3. no latent delta vs absolute+delta；
4. global scene temporal fusion vs same-ID entity temporal fusion，如实现过前者；
5. no Nav vs road-level Nav；
6. exact tracking vs LQR；
7. Plan without WM vs Plan with WM；
8. one-step WM vs six-step WM；
9. teacher-forced action vs predicted Plan action；
10. action condition enabled vs shuffled/zero action condition；
11. frozen Encoder vs adapter joint tuning；
12. plan-action-to-WM gradient detached vs open；
13. BC vs DAgger vs real-env RL；
14. deterministic WM with vs without uncertainty head。

每个 ablation 使用相同 split、seed 组、评测协议和 checkpoint 规则。

---

## 20. Coding Agent 实施优先级

按以下顺序开工，不得跨阶段一次性堆满：

### P0：合同与测试基础

- schema；
- coordinate transform；
- scenario split；
- Expert collector/audit；
- trajectory representation；
- exact tracking；
- 动作参数化 P0 否决门；
- LQR 跟踪 Expert GT 的 controller smoke；
- MetaDrive headless smoke 和 worker 内存标定；
- all critical unit tests。

### P1：单帧 BC

- Encoders；
- Spatial Fusion；
- Plan Head；
- open/closed-loop evaluation；
- failure reproduction。

### P2：真实历史 + DAgger

- Memory Bank；
- ID alignment；
- temporal fusion；
- DAgger/failure buffer；
- nav leak and reset audit。

### P3：短 Real-ENV PPO Canary
- PPO/action-log-prob/GAE/buffer；
- reward/cost 与 hacking audit；
- 只验证训练管线与无灾难性退化，不承担 exceed-IDM 结论。

### P4：World Model

- target encoder；
- ego condition；
- one-step WM；
- multi-step WM；
- Imagine Bank；
- causality/rollout probes。

### P5：完整 3 s LQR 正式闭环、联合磨合与 Real-ENV PPO
- Plan+WM 完整 3 s preview；
- trajectory assembly/interpolation 与 dynamic feasibility；
- LQR controller-realized closed loop；
- joint training；
- 正式 Real-ENV PPO + LQR；
- exceed-IDM 与 exact-vs-LQR diagnosis。

### P6：可选 imagined RL

仅在所有 WM 门槛通过后进行。

---

## 21. Coding Agent 每轮交付要求

每一轮必须提交：

1. 修改文件清单；
2. 设计意图与本文合同的对应条款；
3. 关键代码 diff 说明；
4. 新增/修改测试；
5. 实际运行命令；
6. 配置文件；
7. 运行结果与原始日志路径；
8. 与上一基线的同口径对比；
9. 已知问题和未验证假设；
10. 下一轮只解决的一个主要问题。

禁止只提交总结文档而不提供代码、日志和测试证据。禁止用单次成功 episode 代表总体性能。

---

## 22. 最终不可违反的硬约束

1. World Model 必须显式 ego-action/ego-trajectory conditioned。
2. Plan Head 每次输出下一 0.5 s `(dx,dy,dpsi,v_next)`，rollout 6 次形成 3 s；不得自动删除 `v_next` 或退化为 3D。
3. 六步轨迹点必须转换到同一个 rollout-t0 ego frame 后才能拼接。
4. Stage F 正式 LQR 使用完整 3 s；A1-C 的 0.5 s LQR 仅是 canary/shadow。
5. BC、DAgger、RL 使用同一种轨迹动作语义。
6. OD/LD 先 ego-motion compensation，再按稳定 ID 对齐并计算 latent delta。
7. latent delta 必须与绝对 latent、时间和 mask 一起使用。
8. Road-level Nav 严禁包含 lane-level oracle 或推荐车道。
9. 环境语义全部经 Encoder；控制 mask/reset 可直接使用，不属于策略旁路。
10. Real Memory Bank 与 Imagine Memory Bank 禁止混写。
11. Future latent GT 必须来自真实下一帧 observation 的 per-entity EMA Target Encoder 输出并 stop-gradient；不得用 fused context 或 WM 自输出作 GT。
12. WM 先单步、再多步，未通过六步 fidelity 不得进入 imagined RL。
13. Exact tracking 只证明 Planner 闭环能力；最终成绩必须报告 LQR controller-realized closed loop。
14. IDM + Pure Pursuit 可作为唯一 Expert，但必须审计特权信息，并以真实闭环指标而非 imitation distance 判断是否超越。
15. 每新增模块必须有关闭该模块的同 seed ablation。
16. 调参与 checkpoint 依据冻结 Val500；最终结论必须由只读 Test500、多 seed、分场景复核。

---

## 23. 当前待实现时允许搜索、但不允许静默自行改动的开放项

以下参数允许 Coding Agent 通过小规模实验选择，但必须记录对照，不得默认拍脑袋：

- latent dimension 96 或 128；
- temporal/spatial attention 层数；
- OD/LD token 上限；
- polyline anchor/chunk 方法；
- exact tracking 的实现细节；
- Plan Loss 各项权重；
- EMA momentum；
- WM loss 权重；
- PPO 具体超参数；
- LQR state、Q/R 和 preview 实现；
- trajectory interpolation 方法；
- uncertainty head 的监督定义；
- `(dx,dy,dpsi,v_next)` 的最终 bounds 与归一化范围，须经 P0 Expert 分位数、动力学边界和 LQR 可跟踪性联合标定。

任何开放项一旦通过实验确定，应写入 `docs/decision_log.md`，包含：候选方案、实验运行、同口径结果、选择理由和回退条件。

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

最终模型的核心判据是：

> 在冻结验证集、多 seed、controller-realized closed loop 下，route success 和 completion 提升，collision/off-road 不恶化或下降，并且提升能够在车道保持、障碍物响应、自主选道、路口与收费站等关键场景中解释和复现。


