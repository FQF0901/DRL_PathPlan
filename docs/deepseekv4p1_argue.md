# deepseek-v4.1 argue（第四轮）：终板合同 b05291e 的收口审阅

> 审阅对象：`docs/chatgpt5p6sol_review.md` @ `b05291e`（终板合同 + Stage A1/A2/B/C 的 LQR canary 修订）
>
> 本轮裁决：
> - **B1 驳回**：维持终板现状，不在 Stage A1/A2/B 增加 oracle-state 6-step rollout；
> - **B3 驳回**：维持终板现状，deployable 参数允许 0.8–1.2M，1M 是目标值；
> - **B2 已由 `b05291e` 解决**：Stage A1/A2/B/C 都加入了 LQR canary/shadow eval，Stage F 不再是第一次接 LQR。
>
> 结论先行：**终板合同作为实施基线，动作、时间、Bank/Fusion、Hybrid WM、训练主顺序不再重开。** 本轮只收口剩余合同细节：P0 动作门语义、EMA/detach 分阶段、验证集分层、数据规模、`v_next` 一致性、静态 LD/Nav 监督、taxonomy 和测试清单。B1/B3 的残余风险作为“已接受风险”记录，不再作为改动要求。

---

## 0. 已关闭 / 已接受

| 项 | 裁决 | 本轮处理 |
|---|---|---|
| B1：Stage A1/A2/B 无 oracle-state 6-step rollout | **驳回，维持现状** | 不再要求新增 oracle rollout；接受“Plan Head 在 WM 阶段第一次进入 6 步/imagined state”的残余 exposure-bias 风险，记录为已接受风险 |
| B2：LQR canary 未进入阶段表 | **已由 `b05291e` 解决** | A1 有 Expert GT + LQR 和 Plan Head 单段 + LQR 两类 canary；A2/B 有固定 seeds 的 LQR shadow；C 有 LQR shadow；Stage F 是正式 3s Plan+WM+LQR，不是首次测试 |
| B3：1M 硬门 vs 0.8–1.2M 容差 | **驳回，维持现状** | 接受 0.8–1.2M 的 deployable 容差；1M 是目标值，不是硬上限；EMA training-only 单独统计 |
| 动作 `(dx,dy,dpsi,v_next)` | 终板决定 | 接受；`v_next` 是终端速度，不是平均速度；P0 只校准 bounds/标签，不重开 schema |
| 时间/Bank/Fusion/Hybrid WM/训练主顺序 | 终板决定 | 接受，不再重开 |

**B1 的残余风险（仅记录，不要求改动）**：Plan Head 在 A/B 只学单步；第一次在 6 步/imagined state 上使用是在 WM 阶段。失败时需要用 `E_interface`/分 depth 误差区分 WM transition error、Plan Head 在 imagined state 上的误差、controller error。旧分支 `E_interface` h1 0.64→h5 1.74、WM-state action error≈1.43 vs real≈0.40（E2），可作为调试先验，不作为合同改动理由。

**B3 的残余风险（仅记录）**：deployable 允许到 1.2M，实际结构应在 P0 用参数账本校准；如果 1.2M 导致显存/延迟/训练不稳定，再按实测回退，不预先假设。

---

## 1. 现状排模

- 最新 `origin/pnp_ilrl` HEAD = `b05291e`；tree 仍只有两份文档（E1）。
- `b05291e` 只改 `docs/chatgpt5p6sol_review.md`（35 insertions / 7 deletions），把 LQR canary 写进 Stage A1/A2/B/C，并明确 Stage F 是正式执行口径（E1）。
- 当前分支没有 `env/`、`net/`、`pipeline/`、`dataset/` 代码；没有网络、数据、训练脚本。所谓“代码排模”仍只能到合同级。
- 当前环境 Python 3.10.12、torch 2.3.0、numpy 1.26.4；MetaDrive/gymnasium/pygame/panda3d 未安装（E1）。P0 第一步仍是 MetaDrive headless smoke。
- 终板合同 1807 行（`b05291e` 后）；主要矛盾已消除：10Hz 重规划、Bank fused context、6 帧含当前、Stage F 首次 LQR 都已关闭。

---

## 2. 仍需收口的合同细节

### 2.1 P0 动作门的语义：校准还是可否决 4D 动作

终板 §6.1 要求 P0 完成“动作参数化否决门”，§23 把 `(dx,dy,dpsi,v_next)` 的 bounds/归一化列为开放项。需要明确：

- 如果 P0 只能校准 bounds/标签/Hermite 切向，那它是**校准门**，动作 schema 不变；
- 如果 P0 可以否决 4D 动作，那终板就不是终板，P1 不能先开工；
- 终板没有写“否决后回退到哪个方案、由谁批准、decision log 在哪里”。

**建议**：把 P0 定义为**校准/验收门**：

- 动作 schema 固定为 `(dx,dy,dpsi,v_next)`；
- P0 只校准 `dx/dy/dpsi/v_next` 的 bounds、归一化、`v_next` 标签、Hermite 切向；
- 预定义否决条件：如果 `v_next` 在专家数据中不可辨识（例如与 `||(dx,dy)||/0.5` 的相关性过高、低速标签噪声过大），回退到 `v_next = ||(dx,dy)||/0.5` 的 3D 退化档，并写 decision log；
- 任何 schema 变更必须单独批准，不能由 P0 脚本静默改。

### 2.2 EMA 与 `detach_encoder_from_wm_loss` 需要分阶段

终板 §10.2 同时设：

```yaml
detach_encoder_from_wm_loss: true
use_ema_target_encoder: true
```

需要写清阶段语义：

- D1：Encoder 冻结 → 固定 target（online encoder stop-grad），EMA 是 no-op；
- D2：Encoder 仍冻结 → 固定 target + variance/rank 正则；
- E：只解冻 adapter 时才启用 EMA，并监控 `||θ_target−θ_online||`；
- 不要把 `detach_encoder_from_wm_loss=true` 和“EMA 防坍缩”同时作为全局配置宣称。EMA 是 moving-target 稳定器，不是万能防坍缩项。

### 2.3 验证集分层

终板 §14.1/§14.4：10k train / 1k validation；1000 validation 同时用于 checkpoint selection 和最终结论。这样验证集会过拟合。

**建议**：

- Calib50（从 train 抽，快速迭代，不作收益声明）；
- Val500（checkpoint selection，预注册，限制次数）；
- Test500（最终，不调参；或 1000 全作 Test + Calib200）；
- 禁止在 test 上调场景/阈值/删除失败样本。

### 2.4 数据规模

终板仍写 10k/1k；当前 15GB RAM、4 worker。全量 1k × 60s × 10Hz = 600k env steps，每个 checkpoint 都跑不现实。

**建议**：

- P0/P1 先用 2k/200 验证 schema、专家采集、round-trip、训练/评测装配；
- 最终再扩到 10k/1k；
- 4 worker、recycle 100–150、`mem_available_floor_mb=3000`；
- Calib50/Clean500/Test500 分层，全量只在里程碑跑。

### 2.5 `v_next` 的跨步一致性

终板 §6.1 说 `v_next` 是 0.5s 后目标速度；§6.3 用 `v_n → v_{n+1}` 插值。需要明确：

- 第 n 步输出的 `v_{n+1}` 必须等于第 n+1 步的起始速度 `v_n`（或加一个小的连续性惩罚）；
- 6 步轨迹的 `v_i` 是同一速度链，不是每步独立的目标速度；
- 在 WM rollout 里，ego dynamics 用 `v_{n+1}` 推进；下一步的 `v_n` 必须从同一状态取，不能重新从 Plan Head 随便采一个；
- 加 `test_action_kinematics`：4D 动作 → Hermite → 速度/heading/curvature 连续，无跳变。

### 2.6 动作 bounds 与 LQR 低速保护

终板 §6.1 的 bounds 由 P0 标定；还需要写：

- `v_next` 非负；`|v_next−v_n|/0.5` 受 `a_long` 限制；
- `kappa`/`a_lat` 限幅；`v<0.2 m/s` 时 LQR 用 stop controller，curvature fit 加正则（NAVSIM 的 `initial_curvature_penalty`）；
- 4D 动作的可行性层不能强投影 PPO 采样；post-process rate 要计数。

### 2.7 静态 LD/Nav 监督

终板 §9.2 说静态 LD/Nav 不作为普通预测主项；需要写清监督：

- 静态 LD/Nav：重投影 geometry error + 同一 Encoder 重编码一致性（stop-grad target）；
- 不做 latent MSE 主项；如果做，必须 mask/低权重；
- Nav 只监督 route progress/几何一致性，不做动力学外推。

### 2.8 Taxonomy：toll booth/cone/traffic light

终板 §2.1/§3.1 仍把 toll facilities、cones 放在 OD；但：

- toll booth 是 `TollGateBuilding(BaseBuilding)`，静态；
- cone 是 `TrafficCone`，静态；
- traffic light/闸口状态应归 RoadState；
- OD-dynamic 只放车辆/行人/骑行者；OD-static 放锥桶/护栏/施工障碍；LD 放车道/收费亭几何；Nav 放道路级 route。

不澄清会导致 OD encoder 同时处理动态/静态实体，WM 被要求预测静态建筑。

### 2.9 测试清单缺项

终板 `test/` 已有 schema、坐标、ID、reset、FIFO、real/imagined isolation、nav leak、trajectory assembly/interpolation/continuity、encoder cache、ego condition、WM action sensitivity、rollout consistency、detach、checkpoint。建议补：

- `test_action_kinematics`（4D 动作 → Hermite → 速度/heading/curvature 连续）；
- `test_plan_head_single_instance`（6 次调用同一实例）；
- `test_param_budget` / `test_tensor_budget`（deployable 0.8–1.2M；6 步峰值）；
- `test_lqr_gt_tracking` / `test_lqr_plan_tracking`（`b05291e` 的 canary 需要自动化）；
- `test_train_eval_assembly_parity`（旧分支 50/50 out_of_road 的教训）；
- `test_bank_entity_schema`（Bank 没有 fused context）；
- `test_validation_isolation`（train/val/test 无 seed 泄漏）。

### 2.10 评测/失败归因

终板 failure taxonomy 已有 `LQR_TRACKING_ERROR`、`WM_ROLLOUT_DRIFT` 等；建议再加：

- `PLAN_INTERFACE_SHIFT`（同一头在 real vs imagined state 的动作分布偏移）；
- `ACTION_V_NEXT_INCONSISTENT`（v 链断裂）；
- `STATIC_REPROJECTION_ERROR`；
- `MEMORY_WARMUP_ERROR`；
- `TRAIN_EVAL_ASSEMBLY_MISMATCH`。

---

## 3. 对终板合同的 contract diff（仅剩余项）

1. **P0 动作门语义**：明确“校准/验收门”，schema 固定；预定义 3D 退化回退与 decision log。
2. **EMA/detach 分阶段**：D1 固定 target；D2 固定 target + rank 正则；E 解冻 adapter 才 EMA。
3. **评测分层**：Calib50/Val500/Test500；Test 不调参；预注册 selection。
4. **数据规模**：2k/200 起步，最终 10k/1k；4 worker。
5. **`v_next` 跨步一致性**：速度链、加速度限幅、低速 stop controller。
6. **静态 LD/Nav 监督**：geometry + reencode consistency，不做普通 latent MSE。
7. **Taxonomy**：OD-dynamic / OD-static / LD / Nav / RoadState。
8. **测试清单**：补 action kinematics、plan single instance、param/tensor、LQR GT/plan、assembly parity、bank entity、validation isolation。
9. **失败归因**：补 `PLAN_INTERFACE_SHIFT`、`ACTION_V_NEXT_INCONSISTENT`、`STATIC_REPROJECTION_ERROR`、`MEMORY_WARMUP_ERROR`、`TRAIN_EVAL_ASSEMBLY_MISMATCH`。

---

## 4. P0 spike 与门

### 4.1 Environment smoke

- 建 `.venv`，安装 `metadrive-simulator 0.4.3`、`numpy==1.26.4`、`opencv-python-headless`；
- headless `MetaDriveEnv` reset/step；测 reset/FPS/RSS；
- 4 worker 压测；`mem_available_floor_mb=3000`；
- 通过：10 个场景无异常，4 worker 不 OOM，FPS 可支撑 2k/200。

### 4.2 Action calibration（不是重新选型）

- 用 100–500 个专家 episode 转 `(dx,dy,dpsi,v_next)`；
- 统计 endpoint/heading/terminal-speed、acceleration、jerk、curvature/curvature-rate、full-stop、stop-to-go 重建误差；
- 校准 bounds/归一化；`v_next` 用专家终端速度，不用 `distance/0.5`；
- 通过：重建 ADE/FDE 达标；LQR tracking error/饱和率可接受；`v_next` 与平均速度的相关性/残差可解释；
- 如果 `v_next` 不可辨识，按 §2.1 预定义回退，写 decision log；不静默改 schema。

### 4.3 Model/param/tensor

- 实例化 deployable 模型；四栏账本；deployable 0.8–1.2M 目标区间；
- profile 单步/6 步 rollout；batch 1/8/64；最大 token/attention/activation；
- 通过：6 步不 OOM；峰值显存/延迟可接受。

### 4.4 LQR/assembly

- `test_lqr_gt_tracking`：LQR 跟踪 Expert 3s 轨迹；
- `test_lqr_plan_tracking`：LQR 跟踪 Plan Head 轨迹；
- `test_train_eval_assembly_parity`：同一 checkpoint/同一帧，训练 vs 评测 obs/feature/action 逐位一致；
- 通过：GT 可跟踪；Plan 失败可分解为 planner vs controller；assembly 差异 <= 数值容差。

---

## 5. 仍待确认的细节

1. P0 动作门是“校准”还是“可否决”？回退是否 3D `v_next=||(dx,dy)||/0.5`？
2. 1.2M 上限是否需要按实测回退？当前接受终板 0.8–1.2M。
3. 交通灯是否在范围内？若不在，是否删除 signal 通道？
4. 2k/200 起步是否接受？若不接受，采集/评测预算如何解决 15GB RAM/4 worker？
5. DAgger 轮数/专家混合系数（旧分支 1.0→0.7→0.4→0.2→0.0 只作参考）。
6. `v_next` 的跨步一致性是硬约束还是 soft loss？
7. `b05291e` 的 LQR shadow 具体场景数/seed 集/通过阈值。
8. 静态 LD/Nav 的 reencode consistency 权重与 stop-grad 方式。

---

## 6. 证据附录

### 6.1 当前仓库/环境

- `b05291e` 只改 `docs/chatgpt5p6sol_review.md`（35 insertions / 7 deletions）；tree 仍只有两份文档（E1）。
- Python 3.10.12、torch 2.3.0、numpy 1.26.4；MetaDrive/gymnasium/pygame/panda3d 未安装（E1）。
- RTX 4070 12GB、20 CPU、15GiB RAM、646GiB free；4 worker 建议（E1/E2）。

### 6.2 终板合同关键位置

- 时间合同：§1.5。
- 动作 4D `(dx,dy,dpsi,v_next)`：§6.1。
- 插值/LQR：§6.3。
- PPO 概率动作：§6.5（当前 `a0` 采样，`a1..a5` preview）。
- Bank per-entity / StateFusion：§2.2、§5.1。
- WM target per-entity EMA：§9.2、§13 D1。
- 阶段表：§13 A1/A2/B/C/D1/D2/E/F/G；`b05291e` 在 A1/A2/B/C 增加 LQR canary/shadow。
- 参数：§8.1（0.8–1.2M 容差）。
- detach/EMA：§10.1–10.3。
- 评测：§14.1/§14.4（1000 val 用于 selection）。
- 测试：§17。

### 6.3 旧分支证据（E2，仅风险）

- `origin/Planner_RL:AGENT.md`：IDM+PP 0.756；模型 0.56；LQR+repeat 0.06；LQR+WM plan 0.56；`E_interface` h1 0.64→h5 1.74；WM-state action error≈1.43 vs real≈0.40。
- `origin/Planner_RL:docs/deepseekv4p1_argue.md`：训练/评测装配不一致导致 Calib50 50/50 out_of_road；训练域 v≈8.3 vs 评测域 v≈3.0 的策略自诱发域差；LQR 预瞄只是缓解不是修复。
- `origin/Planner_RL:net/param_probe.py`：H=128 default total 1,278,879 / active 871,478；structured total 1,510,570 / active 994,717。

### 6.4 本地规划 head 证据（E2）

- SparseDrive：per-step `(dx,dy)`，cumsum；loss on deltas；无 LQR。
- VAD/LAW：per-step `(dx,dy)`，cumsum；无 LQR。
- Plan-R1：离散 `(dx,dy,dtheta)` token，0.5s。
- SparseDriveV2：path×speed vocabulary；最终 pose；LQR 下游。
- WoTE/DriveSuprim/Epona/drivelaw：absolute local `(x,y,heading)`；LQR/无控制器。
- 结论：主流 head 输出 pose/pose increment；终板 4D 动作是 pose endpoint + terminal speed，与这一证据不冲突。

### 6.5 外部参考

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
