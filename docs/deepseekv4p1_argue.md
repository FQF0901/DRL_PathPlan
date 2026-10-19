# deepseek-v4.1 argue（第三轮）：对 c2635ac 终板合同的 delta 审阅

> 审阅对象：`docs/chatgpt5p6sol_review.md` @ `c2635ac`（1779 行，终板合同）
>
> 对照：上一轮 `docs/deepseekv4p1_argue.md` @ `bd675cf`；本地 9 个规划 repo 的 head 实测；旧分支 `origin/Planner_RL` 只作风险证据。
>
> 结论先行：**终板合同可以作为基线，不再重开动作方案、时间合同、Bank/Fusion、Hybrid WM 和训练主顺序。** 上一轮我推荐的 3D `(dx,dy,dtheta)` 撤回；终板的 4D `(dx,dy,dpsi,v_next)` 是它的超集，`v_next` 是终端速度而不是平均速度，是 cubic Hermite 的合法纵向边界条件。
> 但终板仍有 **3 个 blocking 缺口**（B1–B3）和 **1 个会重开动作方案的歧义**（B4）；其余是合同级细节/测试/数据规模问题。

---

## 0. 接受为基线

| 项 | 终板决定 | 本轮态度 |
|---|---|---|
| 动作 | `(dx,dy,dpsi,v_next)`，当前 ego frame，0.5s | **接受**；`v_next` 是终端速度，不是 `distance/0.5`；4D 比 3D 多一个真实纵向边界 |
| 参数 | deployable 约 1M；EMA Target Encoder training-only 单独统计 | **接受口径**；但 0.8–1.2M 的容差需要收紧，见 B3 |
| 时间 | 0.1s 控制、0.5s 决策、6 帧含当前、6 步 3s | **接受**，无异议 |
| Bank/Fusion | Bank 存 per-entity token；StateFusion 现算；不存 fused context | **接受** |
| WM | 动态 OD 学习；静态 OD/LD/Nav/Ego 重投影+同 Encoder 重编码；RoadState 独立 | **接受** |
| 训练主顺序 | P0 → A1/A2 BC → B DAgger → C 短 PPO canary → D1/D2 WM → E joint → F LQR+正式 PPO → G imagined RL | **接受**；但 A/B 缺 oracle rollout，C/F 的 tracker 需要写清 |
| Expert/评测 | IDM+PP 唯一 Expert；特权审计；冻结验证集；闭环优先 | **接受** |

**为什么撤回 3D 提案**：终板的 4D 动作不是原文档的 `(dx,dy,v_next)` 错误。原错误是用 `v_next` 替代了平均速度且没有 `dpsi`；终板明确 `v_next` 是终端速度，`(dx,dy,dpsi)` 是 pose endpoint，二者共同定义 Hermite 的端点位置/切向/切向大小。它是 pose+terminal-speed 的合理扩展；9 个本地 repo 的 pose/increment 证据也支持“动作输出 pose 边界，而不是 curvature”。

---

## 1. 现状排模

- 最新 `origin/pnp_ilrl` tree 仍然只有两份文档：`docs/chatgpt5p6sol_review.md`、`docs/deepseekv4p1_argue.md`；`c2635ac` 只改前者（E1）。
- 当前分支没有 `env/`、`net/`、`pipeline/`、`dataset/` 代码；没有可跑的网络、数据集、训练脚本。所谓“代码排模”仍只能到合同级。
- 当前环境 Python 3.10.12、torch 2.3.0、numpy 1.26.4；MetaDrive/gymnasium/pygame/panda3d 未安装（E1）。P0 第一步仍是 MetaDrive headless smoke。
- 终板合同 1779 行，已经消除了上一版的 10Hz 重规划、Stage F 首次 LQR、Bank 存 fused context、6 帧含当前待定等主要矛盾；剩余问题不是“大方向”，而是**阶段接口和门禁**。

---

## 2. 关键缺口（B1–B3 blocking；B4 schema 歧义；B5–B8 合同级）

### B1（blocking）. Stage A1/A2/B 没有 6 步 oracle-state 训练；Plan Head 第一次见 6 步接口在 WM 阶段

终板 Stage A1 是“单帧 BC + Exact”，A2 加 memory，B 是 DAgger；整个 A/B 没有“同一个 Plan Head 在真实未来状态上连续调用 6 次”的训练。Stage D2 的“predicted state + Plan Head action”是 WM 实验，不是 Plan Head 的 6 步监督；Stage E 才第一次把 Plan Head 放到 imagined state 上。

这是终板最大的继承性缺口：

- Plan Head 在 A/B 只学过“当前真实状态 → 下一段 0.5s”；
- D/E 第一次要求它在 WM 预测状态上连续 6 次；
- 失败时无法区分“WM transition error”“Plan Head 在 imagined state 上的 exposure bias”“controller error”。

旧分支的证据（E2）：`E_interface` 从 h1 0.64 增长到 h5 1.74；WM-state action error≈1.43 vs real≈0.40。这不是旧架构的特例，而是“同一头在预测状态上调用”的固有误差。

**建议**：在 Stage A1/A2/B 增加一个 **training-only oracle-state 6-step rollout**：

```
真实 s_t     → 同一 Plan Head → a_t
真实 s_{t+0.5} → 同一 Plan Head → a_{t+0.5}
...
真实 s_{t+2.5} → 同一 Plan Head → a_{t+2.5}
```

- 用数据集里的真实未来帧 latent 作为每步输入；不是让环境开环执行 3s；
- 目标仍是专家 6 段动作；权重按 depth 衰减；
- 不新增第二个 Plan Head，不改变 action schema；
- D/E 阶段再加 **interface loss**：`KL(π(ŝ_k)||sg(π(s_k_real)))` 或 trajectory-space endpoint/heading/speed consistency；权重小于 BC/DAgger，WM fidelity 达标后启用。
- 通过条件：`a0..a5` 同一实例；plan loss 在 real-state 和 imagined-state 两条路径都计算；`E_interface` 随 depth 的曲线可解释。

**这是 blocking，不是可选优化。** 没有它，终板的“同一个 Plan Head 贯穿”只在参数层面成立，在输入分布层面不成立。

### B2（blocking）. LQR canary 没有进入阶段表；Stage F 才第一次真正接 LQR

终板 §1.2 说“从早期并行运行 LQR canary”，但 Stage A1/A2/B/C 都没有 LQR 评测；Stage F 才第一次“LQR Controller-realized Closed Loop + 正式 PPO”。这和上一轮已经接受的“LQR 提前 canary”不一致。

**建议**：

- Stage A1：Exact 主诊断 + 一个很小的 LQR tracking canary（例如 10 个场景），跟踪同一 checkpoint 的 3s plan；
- Stage B：DAgger 数据用 Exact 采集，但每轮用 LQR 跑固定小集，报告 tracking error/饱和；
- Stage C：短 PPO canary 同时做 Exact plumbing canary 和 LQR tracking canary；
- Stage F 才是正式 LQR 训练/评测，不是第一次接 LQR。

**通过条件**：先证明 LQR 能跟踪 Expert GT 3s 轨迹，再证明能跟踪 Plan Head 轨迹；否则不要用 LQR 的失败反推 Planner。

### B3（blocking）. deployable 1M 的硬门与终板 0.8–1.2M 容差冲突

终板 §8.1 写“目标约 100 万，允许 0.8M–1.2M 浮动”；你已裁决 **1M = deployable net**。这两个不能同时成立。

**建议**：

- 硬门：`deployable_total <= 1,000,000`；
- 工程目标：0.8M–1.0M，留余量；
- 1.2M 只能作为**显式超预算实验**，需要单独批准，不能默认允许；
- `training_only_target`（EMA）、`trainable`、`optimizer_states` 分开报告；EMA 不计入 deployable，但显存/activation 要报告；
- P0 用实例化模型做四栏账本，未知前缀 fail-fast。

### B4（schema 歧义）. P0 的“动作参数化否决门”语义不清：是校准，还是可以推翻终板动作

终板 §6.1 说 P0 必须完成“动作参数化否决门”，§23 又把 `(dx,dy,dpsi,v_next)` 的 bounds 列为开放项。问题是：

- 如果 P0 只能校准 bounds/标签，那它是**验证门**，动作方案不变；
- 如果 P0 可以因为重建误差/LQR 误差否决 4D 动作，那终板就不是终板，P1 不能先开工；
- 终板没有写“否决后回退到哪个方案、由谁批准、decision log 在哪里”。

**建议**：把 P0 定义为**校准/验收门**，不是重新选型：

- 动作 schema 固定为 `(dx,dy,dpsi,v_next)`；
- P0 只校准 `dx/dy/dpsi/v_next` 的 bounds、归一化、`v_next` 标签、Hermite 切向；
- 预定义否决条件：如果 `v_next` 在专家数据中不可辨识（例如与 `||(dx,dy)||/0.5` 的相关性过高、低速标签噪声过大），回退到 `v_next = ||(dx,dy)||/0.5` 的 3D 退化档，并写 decision log；
- 任何 schema 变更必须单独批准，不能由 P0 脚本静默改。

### B5（合同级）. `detach_encoder_from_wm_loss=true` 与 EMA Target Encoder 需要分阶段，不应作为全局开关

终板 §10.2 同时设：

```yaml
detach_encoder_from_wm_loss: true
use_ema_target_encoder: true
```

这两个一起用需要解释清楚：

- 如果 Encoder 冻结（D1），EMA target 是 no-op，直接用 online encoder stop-grad 即可；
- 如果 Encoder 不接收 WM loss，EMA target 只跟踪一个为 BC/planning 优化的表征，不能把 EMA 当作“防坍缩”的主要机制；
- 只有在解冻 Encoder adapter、WM loss 能改写表征时，EMA 才承担 moving-target 稳定作用，并需要监控 `||θ_target−θ_online||`。

**建议**：按阶段写死：

- D1：Encoder 冻结 → 固定 target（online encoder stop-grad），不需要 EMA；
- D2：Encoder 仍冻结 → 固定 target + variance/rank 正则；
- E：只解冻 adapter 时才启用 EMA，并报告 target drift；
- 不要把 `detach_encoder_from_wm_loss=true` 和“EMA 防坍缩”同时作为全局配置宣称。

### B6（合同级）. 验证集同时用于 checkpoint selection 和最终结论

终板 §14.1 是 10k/1k，§14.4 用验证集做 checkpoint 选择；没有 Calib/Val/Test 分层。这样 1000 验证集会被反复用于选点，不再是“冻结”。

**建议**：

- Calib50（从 train 抽，快速迭代，不作收益声明）；
- Val500（checkpoint selection，预注册，限制次数）；
- Test500（最终，不调参；或 1000 全作 Test + Calib200）；
- 禁止在 test 上调场景/阈值/失败样本。

### B7（合同级）. 数据规模：10k/1k 从第一天开始，当前机器会卡死

终板仍写 10k/1k；当前 15GB RAM、4 worker 的采集/评测成本没有预算。全量 1k × 60s × 10Hz = 600k env steps，每个 checkpoint 都跑不现实。

**建议**：

- P0/P1 先用 2k/200 验证 schema、专家采集、round-trip、训练/评测装配；
- 最终再扩到 10k/1k；
- 4 worker、recycle 100–150、`mem_available_floor_mb=3000`；
- Calib50/Clean500/Test500 分层，全量只在里程碑。

### B8（合同级）. Stage C 短 PPO canary 的 tracker 没定义

终板 §13 Stage C 只说“短 Real-ENV PPO Canary，无 WM”，没有说 Exact 还是 LQR。若用 Exact，只验证 PPO 管线，不验证部署链；若用 LQR，policy/WM 未磨合，可能早期崩。

**建议**：

- Stage C 做两个 canary：Exact PPO plumbing canary（10–20 updates）+ 同一 checkpoint 的 LQR tracking canary（10–20 场景）；
- 两者都不作性能结论；
- Stage F 才是正式 LQR PPO。

---

## 3. 非 blocking，但必须在合同里写清

### 3.1 `v_next` 的跨步一致性

终板 §6.1 说 `v_next` 是 0.5s 后目标速度；§6.3 用 `v_n → v_{n+1}` 插值。需要明确：

- 第 n 步输出的 `v_{n+1}` 必须等于第 n+1 步的起始速度 `v_n`（或加一个小的连续性惩罚）；
- 6 步轨迹的 `v_i` 是同一速度链，不是每步独立的目标速度；
- 在 WM rollout 里，ego dynamics 用 `v_{n+1}` 推进；下一步的 `v_n` 必须从同一状态取，不能重新从 Plan Head 随便采一个。
- 加 `test_action_kinematics`：4D 动作 → Hermite → 速度/heading/curvature 连续，无跳变。

### 3.2 动作 bounds 与 LQR 低速保护

终板 §6.1 的 bounds 由 P0 标定；还需要写：

- `v_next` 非负；`|v_next−v_n|/0.5` 受 `a_long` 限制；
- `kappa`/`a_lat` 限幅；`v<0.2 m/s` 时 LQR 用 stop controller，curvature fit 加正则（NAVSIM 的 `initial_curvature_penalty`）；
- 4D 动作的可行性层不能强投影 PPO 采样；post-process rate 要计数。

### 3.3 静态 LD/Nav 监督

终板 §9.2 说静态 LD/Nav 不作为普通预测主项；需要写清监督：

- 静态 LD/Nav：重投影 geometry error + 同一 Encoder 重编码一致性（stop-grad target）；
- 不做 latent MSE 主项；如果做，必须 mask/低权重；
- Nav 只监督 route progress/几何一致性，不做动力学外推。

### 3.4 Taxonomy：toll booth/cone/traffic light

终板 §2.1/§3.1 仍把 toll facilities、cones 放在 OD；但：

- toll booth 是 `TollGateBuilding(BaseBuilding)`，静态；
- cone 是 `TrafficCone`，静态；
- traffic light/闸口状态应归 RoadState；
- OD-dynamic 只放车辆/行人/骑行者；OD-static 放锥桶/护栏/施工障碍；LD 放车道/收费亭几何；Nav 放道路级 route。

不澄清会导致 OD encoder 同时处理动态/静态实体，WM 被要求预测静态建筑。

### 3.5 测试清单缺项

终板 `test/` 已有 schema、坐标、ID、reset、FIFO、real/imagined isolation、nav leak、trajectory assembly/interpolation/continuity、encoder cache、ego condition、WM action sensitivity、rollout consistency、detach、checkpoint。建议补：

- `test_action_kinematics`（4D 动作 → Hermite → 速度/heading/curvature 连续）；
- `test_plan_head_single_instance`（6 次调用同一实例）；
- `test_oracle_rollout`（真实未来状态 6 步监督）；
- `test_interface_loss`（predicted vs real state 的 KL/一致性）；
- `test_param_budget` / `test_tensor_budget`（deployable≤1M；6 步峰值）；
- `test_lqr_gt_tracking` / `test_lqr_plan_tracking`；
- `test_train_eval_assembly_parity`（旧分支 50/50 out_of_road 的教训）；
- `test_bank_entity_schema`（Bank 没有 fused context）；
- `test_validation_isolation`（train/val/test 无 seed 泄漏）。

### 3.6 评测/失败归因

终板 failure taxonomy 已有 `LQR_TRACKING_ERROR`、`WM_ROLLOUT_DRIFT` 等；建议再加：

- `PLAN_INTERFACE_SHIFT`（同一头在 real vs imagined state 的动作分布偏移）；
- `ACTION_V_NEXT_INCONSISTENT`（v 链断裂）；
- `STATIC_REPROJECTION_ERROR`；
- `MEMORY_WARMUP_ERROR`；
- `TRAIN_EVAL_ASSEMBLY_MISMATCH`。

---

## 4. 对终板合同的 contract diff（按优先级）

1. **Stage A1/A2/B 增加 oracle-state 6-step rollout**：同一 Plan Head，真实未来状态，6 段专家动作；不改变 action schema。
2. **Stage D/E 增加 interface loss**：KL 或 trajectory consistency，分 depth，权重 < BC/DAgger，WM fidelity 达标后启用。
3. **LQR canary 写进阶段表**：A1/B/C 各一个小 canary；F 才是正式 LQR。
4. **1M 硬门**：`deployable_total <= 1,000,000`；0.8–1.0M 目标；1.2M 仅显式批准。
5. **P0 动作门语义**：schema 固定；P0 只校准 bounds/`v_next` 标签/Hermite；预定义 3D 退化回退与 decision log。
6. **EMA/detach 分阶段**：冻结 Encoder → 固定 target；解冻 adapter → EMA；不要同时宣称 `detach_encoder_from_wm_loss=true` + EMA 防坍缩。
7. **评测分层**：Calib50/Val500/Test500；Test 不调参；预注册 selection。
8. **数据规模**：2k/200 起步，最终 10k/1k；4 worker。
9. **Stage C tracker**：Exact plumbing canary + LQR tracking canary。
10. **`v_next` 跨步一致性**：`v_{n+1}` 链、速度/加速度限幅、低速 stop controller。
11. **静态 LD/Nav 监督**：geometry + reencode consistency，不做普通 latent MSE。
12. **Taxonomy**：OD-dynamic / OD-static / LD / Nav / RoadState。
13. **测试清单**：补 action kinematics、oracle rollout、interface、param/tensor、LQR GT、assembly parity、bank entity、validation isolation。

---

## 5. P0 spike 与门

### 5.1 Environment smoke

- 建 `.venv`，安装 `metadrive-simulator 0.4.3`、`numpy==1.26.4`、`opencv-python-headless`；
- headless `MetaDriveEnv` reset/step；测 reset/FPS/RSS；
- 4 worker 压测；`mem_available_floor_mb=3000`；
- 通过：10 个场景无异常，4 worker 不 OOM，FPS 可支撑 2k/200。

### 5.2 Action calibration（不是重新选型）

- 用 100–500 个专家 episode 转 `(dx,dy,dpsi,v_next)`；
- 统计 endpoint/heading/terminal-speed、acceleration、jerk、curvature/curvature-rate、full-stop、stop-to-go 重建误差；
- 校准 bounds/归一化；`v_next` 用专家终端速度，不用 `distance/0.5`；
- 通过：重建 ADE/FDE 达标；LQR tracking error/饱和率可接受；`v_next` 与平均速度的相关性/残差可解释。
- 如果 `v_next` 不可辨识，按 B4 预定义回退到 3D 退化档，写 decision log；不静默改 schema。

### 5.3 Model/param/tensor

- 实例化 deployable 模型；四栏账本；`deployable<=1M`；
- profile 单步/6 步 rollout；batch 1/8/64；最大 token/attention/activation；
- 通过：6 步不 OOM；峰值显存/延迟可接受。

### 5.4 LQR/assembly

- `test_lqr_gt_tracking`：LQR 跟踪 Expert 3s 轨迹；
- `test_lqr_plan_tracking`：LQR 跟踪 Plan Head 轨迹；
- `test_train_eval_assembly_parity`：同一 checkpoint/同一帧，训练 vs 评测 obs/feature/action 逐位一致；
- 通过：GT 可跟踪；Plan 失败可分解为 planner vs controller；assembly 差异 <= 数值容差。

---

## 6. 仍待确认的细节

1. 1M 硬门是否接受 0.8–1.0M、1.2M 仅显式批准？
2. P0 的 4D 动作门是“校准”还是“可否决”？如果可否决，回退方案是否 3D `v_next=||(dx,dy)||/0.5`？
3. Stage A1/A2/B 的 oracle rollout 是主损失还是辅助损失？权重/深度衰减？
4. Stage C 的 Exact/LQR canary 场景数/update 数/门限？
5. 交通灯是否在范围内？若不在，是否删除 signal 通道？
6. 2k/200 起步是否接受？若不接受，采集/评测预算如何解决 15GB RAM/4 worker？
7. DAgger 轮数/专家混合系数（旧分支 1.0→0.7→0.4→0.2→0.0 只作参考）。
8. `v_next` 的跨步一致性是硬约束还是 soft loss？

---

## 7. 证据附录

### 7.1 当前仓库/环境

- `c2635ac` 只改 `docs/chatgpt5p6sol_review.md`（1631 insertions / 233 deletions）；tree 仍只有两份文档（E1）。
- Python 3.10.12、torch 2.3.0、numpy 1.26.4；MetaDrive/gymnasium/pygame/panda3d 未安装（E1）。
- RTX 4070 12GB、20 CPU、15GiB RAM、646GiB free；4 worker 建议（E1/E2）。

### 7.2 终板合同关键位置

- 时间合同：§1.5。
- 动作 4D `(dx,dy,dpsi,v_next)`：§6.1。
- 插值/LQR：§6.3。
- PPO 概率动作：§6.5（当前 `a0` 采样，`a1..a5` preview）。
- Bank per-entity / StateFusion：§2.2、§5.1。
- WM target per-entity EMA：§9.2、§13 D1。
- 阶段表：§13 A1/A2/B/C/D1/D2/E/F/G。
- 参数：§8.1（0.8–1.2M 容差）。
- detach/EMA：§10.1–10.3。
- 评测：§14.1/§14.4（1000 val 用于 selection）。
- 测试：§17。

### 7.3 旧分支证据（E2，仅风险）

- `origin/Planner_RL:AGENT.md`：IDM+PP 0.756；模型 0.56；LQR+repeat 0.06；LQR+WM plan 0.56；`E_interface` h1 0.64→h5 1.74；WM-state action error≈1.43 vs real≈0.40。
- `origin/Planner_RL:docs/deepseekv4p1_argue.md`：训练/评测装配不一致导致 Calib50 50/50 out_of_road；训练域 v≈8.3 vs 评测域 v≈3.0 的策略自诱发域差；LQR 预瞄只是缓解不是修复。
- `origin/Planner_RL:net/param_probe.py`：H=128 default total 1,278,879 / active 871,478；structured total 1,510,570 / active 994,717。

### 7.4 本地规划 head 证据（E2）

- SparseDrive：per-step `(dx,dy)`，cumsum；loss on deltas；无 LQR。
- VAD/LAW：per-step `(dx,dy)`，cumsum；无 LQR。
- Plan-R1：离散 `(dx,dy,dtheta)` token，0.5s。
- SparseDriveV2：path×speed vocabulary；最终 pose；LQR 下游。
- WoTE/DriveSuprim/Epona/drivelaw：absolute local `(x,y,heading)`；LQR/无控制器。
- 结论：主流 head 输出 pose/pose increment；terminal speed 是合理的额外纵向边界，终板 4D 与这一证据不冲突。

### 7.5 外部参考

- SparseDrive: arXiv:2405.19620
- VAD: arXiv:2303.12077
- UniAD: arXiv:2306.16927
- SparseDriveV2: arXiv:2603.29163
- DriveSuprim: arXiv:2506.06659
- TCP: arXiv:2206.08129
- ADT: arXiv:2606.02105
- Waypoint–action lifting / CCPP: arXiv:2602.06214
- Apollo EM planning: arXiv:1807.08048
- Epona: arXiv:2506.24113
- DriveLaW: arXiv:2512.23421

---

*本轮只改本文档；不修改终板合同或任何代码。*
