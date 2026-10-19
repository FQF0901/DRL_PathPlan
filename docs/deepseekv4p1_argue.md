# deepseek-v4.1 argue（第二轮）：从 65244be 裁决到可执行合同

> 审阅对象：`docs/chatgpt5p6sol_review.md` @ `65244be`（380 行，ChatGPT 5.6 sol 的裁决）
>
> 上一轮：`docs/deepseekv4p1_argue.md` @ `2e8424b`（841 行）
>
> 本轮输入：
> - 你已裁决 **1M = deployable net**；
> - 你要求动作参数化**现在选定优先方案**，不能留给 P0 待定；
> - 已排模 9 个本地规划 repo 的实际 planning head 输出；
> - ChatGPT 5.6 sol 对上一轮 841 行文档的逐条裁决。
>
> 证据分级：**E1** = 当前仓库/依赖源码/命令输出；**E2** = 本地 repo 代码或旧分支实测；**E3** = 估计/待验证。
>
> 结论先行：**采纳 ChatGPT 的大部分裁决；动作参数化不再留成 P0 决策，直接选定 `(dx, dy, dtheta)` 的 SE(2) 增量。** 理由不是“旧分支跑通过”，而是 9 个本地规划 repo 的实测：**没有一个 learned planning head 把连续 curvature / speed / acceleration 作为主 action；它们全部输出 pose 或 pose increment，curvature/speed 是下游 LQR/控制器的派生量。** 这直接否定了把 `(kappa, a)` 或 `(ds, dtheta)` 锁成主方案的必要性。

---

## 0. 本轮定案

| 项 | 决定 | 说明 |
|---|---|---|
| 参数口径 | **deployable net ≤ 1M** | EMA Target Encoder、optimizer states 是 training-only，单独报告，不计入 deployable；上一轮“total≤1M 所以不能有 EMA”结论撤回 |
| 时间 | `control_dt=0.1s`、`policy_dt=0.5s`、`memory_dt=0.5s`、`wm_dt=0.5s`、`rollout=6`、horizon=3s | 2Hz 决策 + 10Hz 执行；每 0.5s 决策一次，执行 5 个 0.1s 控制步 |
| 动作 | **`a=(dx,dy,dtheta)`，当前 ego frame 下的 0.5s SE(2) 增量** | 3D；`dx,dy` 是位移，`dtheta` 是航向增量；速度/曲率由几何派生，不单独输出 |
| Plan Head | 单实例、每步调用 6 次 | BC 阶段用真实未来状态 oracle rollout；WM 后输入替换为 imagined state；action schema 不变 |
| Bank | 存 per-entity token，不存 fused context | `M_t → StateFusion → c_t → Plan/Value`；`M_t + a_t → WM → z_entity,t+1` |
| WM | Hybrid：动态 OD 学习预测；静态 OD/LD/Nav/Ego 确定性重投影+同一 Encoder 重编码；RoadState 独立 | 不再争论“是否纯 latent WM” |
| 训练顺序 | P0 → P1 Plan BC → P2 DAgger → P2.5 Real-ENV PPO canary → P3 WM → P4 Joint+LQR → P5 正式 PPO+LQR → P6 Imagined RL | R16 为准；R9 是最初设想，不再作为执行顺序 |
| LQR | Exact 只作 planner 上界；LQR 从 P1 起并行 canary；正式 PPO 用 LQR | P2.5 同时做 Exact plumbing canary 和 LQR tracking canary |
| 数据/评测 | 最终 10k/1k；P0/P1 先 2k/200；4 个 env worker；Calib/Val/Test 分层 | 当前 15GB RAM，不能用 16/20 worker；全量 1k 只在里程碑跑 |

---

## 1. 现状排模

### 1.1 仓库与代码

- 最新 `origin/pnp_ilrl` 的 tree 只有两份文档：`docs/chatgpt5p6sol_review.md`、`docs/deepseekv4p1_argue.md`。`65244be` 只改了前者；`2e8424b` 只加了后者（E1）。
- 当前分支没有 `env/`、`net/`、`pipeline/`、`dataset/` 代码；没有可跑的网络、没有数据集、没有训练脚本。所谓“代码排模”只能到**合同级**，不能到实现级。
- 当前 Python 3.10.12、torch 2.3.0、numpy 1.26.4；MetaDrive/gymnasium/pygame/panda3d 仍未安装（E1）。P0 第一步仍是 MetaDrive headless smoke，而不是写网络。

### 1.2 本地规划 repo 的实际 head 输出

这一轮排模了 9 个本地 repo，重点看 planning head 到底输出什么：

| Repo | Head 输出 | 语义 | 轨迹转换 | 控制器 |
|---|---|---|---|---|
| SparseDrive | `(B,3,6,6,2)` plan regression；class logits `(B,3,6)` | 每步 `(dx,dy)`，0.5s，3 command × 6 mode | `.cumsum(dim=-2)` 成绝对 `(x,y)` | 无 LQR；open-loop L2/collision |
| VAD / LAW | `(B,3,6,2)` / `(B,6,2)` | 每步 `(dx,dy)`，0.5s | `.cumsum(dim=-2)` | 无 LQR；open-loop ST-P3 |
| Plan-R1 | 1024 类 token；ego token `(dx,dy,dtheta)` | 每步 SE(2) 增量，0.5s | `p'=R(theta)[dx,dy]+p`，`theta'=wrap(theta+dtheta)` | nuPlan InterpolatedTrajectory；可选 PDM LQR |
| SparseDriveV2 | `(B,8,3)` absolute local `(x,y,heading)` | 0.5s，4s horizon；offline path×speed vocab | 直接 pose lookup；path 按 `cumsum(speed*dt)` 重采样 | NavSim PDM LQR |
| WoTE | `(B,8,3)` absolute local `(x,y,heading)` | 0.5s，4s horizon；anchor + offset | 直接 pose；heading `tanh*π` | NavSim PDM LQR |
| DriveSuprim | `(B,40,3)` / `(B,8,3)` absolute local pose | 0.1s/0.5s horizon；vocab lookup | 直接 pose | NavSim PDM LQR |
| Epona | `(B,15,3)` `(x,y,yaw)` | 5Hz，3s horizon，当前 ego frame | 直接 pose | 无控制器 |
| drivelaw / DriveLaW-Act | `(B,8,3)` absolute local `(x,y,heading)` | 0.5s，4s horizon | 直接 pose | NavSim PDM LQR |
| HorizonDrive | 无 planning head；action `(dx,dy,dyaw)` | 0.5s 动作条件 | 不是 planner | 无 |

关键事实（E2）：

1. **没有一个 learned head 把连续 curvature、speed、acceleration 作为主 action**。它们输出 pose 或 pose increment；curvature/speed 由下游 LQR/控制器或评测指标派生。SparseDriveV2 有离散 speed vocabulary，但最终输出仍是 pose 轨迹，不是连续 speed action。
2. SparseDrive / VAD / LAW 训练时回归 per-step `(dx,dy)`，推理时 cumsum；SparseDrive 的 plan loss 甚至直接作用在 delta 上（`motion_planning_head.py:441-448`），decode 才 cumsum（`decoder.py:130-135`）。
3. Plan-R1 是唯一使用**离散 SE(2) token `(dx,dy,dtheta)`** 的；它的 token 里甚至有 `|dtheta|≈π` 的极端样本，没有运动学过滤。这说明“pose increment”是行业常用接口，但**可行性必须由下游/损失约束**。
4. SparseDriveV2 是唯一显式做 path×speed 分解的，但它是**离线词表**：`target_distance=cumsum(velocity*DT)` 后按弧长重采样 path（`scripts/cluster/cluster_anchor.py:59-75`），不是 learned head 直接输出 curvature/speed。
5. NavSim 的 LQR 是 pose → 轨迹 → 控制；低速度曲率问题是在 LQR 的 curvature fit 里处理的（`batch_lqr_utils.py:149,170-174`；`batch_lqr.py:71,84,161-170`），不是在 learned head 里输出 curvature。

**结论**：如果目标是“行业通用 + 与 LQR 对接”，动作应该是 **pose/SE(2) increment**，不是 curvature/acceleration。`(ds,dtheta)` 是一个特殊的弧长参数化；`(kappa,a)` 是经典规划/控制参数化，但不是这 9 个 end-to-end repo 的 planning head 输出。上一轮我把 `(ds,dtheta)` 提升为“唯一答案”是错的；ChatGPT 把它拉回“不能锁死”是对的。但这轮不应再走到另一个极端去锁 `(kappa,v_end)`；证据支持的是 **`(dx,dy,dtheta)`**。

---

## 2. 动作参数化：选定 `(dx,dy,dtheta)`

### 2.1 定义

在每个 0.5s 决策步 `n`，Plan Head 输出：

\[
a_n = (dx_n,\ dy_n,\ dtheta_n)
\]

- 位于当前 imagined/real ego frame（x 前、y 左、theta 为航向）；
- `dx,dy` 是 0.5s 末的位移，`dtheta` 是航向增量；
- 6 次调用得到 6 个 SE(2) 增量，累计成 3s 轨迹：

\[
P_{n+1}=P_n\oplus\left(R(\theta_n)\begin{bmatrix}dx_n\\dy_n\end{bmatrix},\ \theta_n+dtheta_n\right)
\]

- 速度/加速度/曲率不作为动作输出；由轨迹派生：
  - `v_avg = sqrt(dx^2+dy^2)/0.5`；
  - `a_long` 由相邻段速度差/0.5 派生；
  - `kappa` 由 10Hz 插值轨迹的 heading 变化/弧长派生；
  - LQR 使用完整 3s 参考（7 个 pose → 30 个 0.1s 点）。

### 2.2 为什么不是 `(dx,dy)` 两维

- SparseDrive/VAD/LAW 只输出 `(dx,dy)`，因为它们的评测是 open-loop L2，不需要 heading；decode 时 cumsum 成 `(x,y)`。
- 我们的系统需要 **closed-loop ego frame 推进**：WM 要把 OD/LD/Nav anchor 旋转/平移到下一帧，LQR 需要参考 heading。没有 `dtheta` 就只能从弦方向反推 heading，会引入累计误差，且低速/小位移时不稳定。
- Plan-R1 和 HorizonDrive 都把 `dtheta` 作为动作的一部分；这是有先例的。
- 代价只是一个额外维度（3D vs 2D），对 1M 模型和 PPO 都可接受。

### 2.3 为什么不是 `(dx,dy,v_next)`

- `v_next` 与 `(dx,dy)` 的位移长度冗余：`v_avg=||(dx,dy)||/0.5`；两者可以自相矛盾。
- 初始方案 `(dx,dy,v_next)` 的问题不是“waypoint 方向错了”，而是多输出了一个不必要的速度维度。
- 速度/加速度由轨迹派生即可；如果 LQR 需要速度参考，从插值轨迹取。

### 2.4 为什么不是 `(ds,dtheta)`

- 它是 `(dx,dy,dtheta)` 在常曲率圆弧假设下的特殊参数化：`kappa=dtheta/ds`，`v_avg=ds/0.5`。
- 它把横向路径和纵向速度耦合在一起：同一路径减速必须同时缩 `ds` 和 `dtheta`；低速时 `kappa=dtheta/ds` 病态，容易产生极大曲率。
- 旧分支跑通过 `(ds,dtheta)`，但这只证明“可实现”，不证明“最优/行业通用”。这轮 9 个 repo 里没有 learned head 用它作为主输出。
- 保留为 baseline/ablation，不作为主合同。

### 2.5 为什么不是 `(kappa,a_long)` 或 `(kappa,v_end)`

- 它是经典 path-speed 分解，适合 Apollo/EM/Frenet/lattice/MPC；但在这 9 个 end-to-end repo 里，**没有把连续 curvature/acceleration 作为主 action 的 learned head**（SparseDriveV2 的 speed 是离散词表，最终仍是 pose）。
- `kappa` 的可行域依赖速度：`|kappa| <= a_lat_max/v^2`；高速时很小，低速时很大；固定 bound 不对。
- 从专家 10Hz 轨迹反推 `kappa` 在低速时噪声大（`dtheta/ds`，`ds→0`）；`(kappa,a)` 不会自动解决这个标签问题，只是把问题从动作空间移到数据空间。
- `v_end` 有非负约束且依赖 `v0`；`a_long` 更自然，但它仍然不是行业 head 的常见输出。
- 固定 0.5s 时间窗内，空间位移与速度天然耦合；`(kappa,a)` 只是把耦合从动作空间移到积分器，并不会让同一 0.5s 段同时“更慢但走同样远”。pose increment 更直接。
- 保留为 ablation/经典规划对照；不作为主合同。

### 2.6 动作 bounds 与可行性

`(dx,dy,dtheta)` 是 pose increment，不是任意曲线；必须有可行性约束，否则会输出不可跟踪的“横向跳变”。建议：

```
dx ∈ [0, 10] m            # 0.5s 前向位移；倒车另开 reverse 模式
dy ∈ [-2.0, 2.0] m        # 0.5s 横向位移
dtheta ∈ [-0.6, 0.6] rad  # 0.5s 航向变化
v_avg = sqrt(dx^2+dy^2)/0.5
kappa_est = 2*sin(dtheta/2)/max(sqrt(dx^2+dy^2), eps)   # 或从 Hermite 轨迹派生
|kappa_est| <= kappa_max(v_avg)
a_lat = v_avg^2 * |kappa_est| <= a_lat_max
```

- `kappa_max(v)` = `min(kappa_vehicle, a_lat_max/max(v^2, v_floor^2))`；`v_floor` 取 2 m/s 左右，避免静止时除零。
- 训练时用 **feasibility loss / penalty**，不要对 PPO 采样动作做强投影；强投影会让 log-prob 与 executed action 不对应。
- 推理/评测时可以 reject 或平滑，但必须计数 `post_process_rate`，不能静默改写。
- `dy`/`dtheta` 的 bounds 是起点，P0 用专家分布校准；这不是“选择动作方案”，是“校准同一个方案的边界”。

### 2.7 轨迹插值与 LQR

- 6 个 SE(2) 增量 → 7 个 pose（含 t=0）。
- 用带切线约束的 cubic Hermite 插值到 30 个 0.1s 点；不要用无约束 Catmull-Rom（会过冲）。heading 由端点切线给出，速度由弧长/时间派生，曲率由 heading 差分派生。
- LQR 使用完整 3s 参考；不是只传第一段。
- 插值后做曲率/横向加速度/速度限幅；如果超限，标记并回退到平滑/减速，而不是静默裁剪。
- 这与 NavSim/WoTE/Epona/drivelaw 的“pose → trajectory → LQR”链路一致；区别只是我们在模型内自回归生成 pose。

### 2.8 Plan Head / PPO / WM 的接口

- Plan Head 输出 3D `(dx,dy,dtheta)` 的 `mu/log_std`；用 sigmoid/tanh 压缩到 bounds，并做 log-prob 的 change-of-variables 修正。
- PPO 当前 transition 只包含 `a0`：`s_t → a0 → 执行 0.5s（5 个 0.1s 控制步）→ s_{t+0.5}`；`a1..a5` 只是 preview/value/LQR 预瞄，不进入当前 log-prob。它们可以是均值或采样，但不是当前 PPO 的随机动作。
- WM 的 action condition 就是 `(dx,dy,dtheta)`；ego dynamics 用同一积分；OD 预测看到的是 action embedding，不是 speed/curvature。
- 这与 R10 的“6 个动作连成轨迹给 LQR”一致；也与 R13/R14 的“同一 Plan Head/同一 action schema”一致。

---

## 3. 采纳 ChatGPT 的裁决

| ChatGPT 裁决 | 本轮处理 |
|---|---|
| 统一时间合同：0.1/0.5/0.5/0.5/6/3s | **采纳**，写死 |
| Plan Head 早期保留 6 步调用；BC 用真实未来状态 oracle rollout | **采纳**；不伪造无 WM 的 imagined rollout |
| Bank 存 per-entity；StateFusion 现算；Plan/Value 读 fused context | **采纳**；禁止把 `c_t` 写回 Bank |
| Hybrid WM：动态 OD 学习；静态 OD/LD/Nav/Ego 重投影+重编码；RoadState 独立 | **采纳** |
| Memory：6 帧含当前；warmup 用最早真实帧 + valid=0；runtime detach；offline/runtime 分开；encoder hash | **采纳** |
| LQR 提前 canary；Exact 只作上界；最终成绩必须 LQR | **采纳** |
| `env/`、LQR 不放 `net/`、headless smoke、4 worker、assembly parity、Calib/Val/Test、taxonomy、测试 | **采纳** |
| R9/R16：R16 为准；短 Real-ENV PPO canary；WM 在 canary 与正式 RL 之间 | **采纳**；R9 只作为最初设想 |
| “同一个 Plan Head”不要求输入分布相同；要求同实例/同 action schema/同单位/同积分/checkpoint 连续 | **采纳**；上一轮 B3 措辞过强，已修正 |
| Future latent GT = 下一帧真实 observation 经 Target Encoder 的 per-entity token | **采纳** |
| Interface loss 不要强求 `π(ŝ)=π(s_real)`；用 KL/一致性、分 depth、WM fidelity 达标后启用、权重小于 BC | **采纳**；见 §4.3 |
| Anti-collapse 不要一次堆满；先 MSE+cosine+monitor+OD decoder，检测到 rank/std 下降再加 VICReg | **采纳**；见 §4.4 |
| PPO 只对当前 `a0` 采样/记 log-prob；未来 5 段只是 preview | **采纳**；见 §2.8 |
| Action conditioning 分实体：ego/geometry 必须敏感；交互 OD 应敏感；远 OD 可近似不敏感；静态 LD/Nav 物理内容不变 | **采纳**；见 §4.5 |
| EMA/1M 要分 deployable/training-only/trainable/optimizer states | **采纳**；你已裁决 deployable≤1M，见 §4.6 |

---

## 4. 仍需修正/明确的细节

### 4.1 Temporal 输入：不要 delta-only，但也不要 6 帧全量 token attention

ChatGPT 的“`z_current + (z_current-z_history)` 可以恢复历史”在数学上只对线性/可逆编码成立；实际 Encoder 非线性，且有 birth/death、mask、类别变化。所以不要把 delta-only 写成硬合同。

**默认输入**（每实体、每历史帧）：

```
current absolute content
+ compensated historical content
+ content residual (current - historical)
+ explicit geometry delta
+ age / mask
```

**硬规则**：不对 6 帧所有 token 做全量时空 self-attention；而是 per-entity temporal encoder（小 GRU/attention/MLP）→ 当前实体表示 → spatial attention。这样既保留“变化”信号，又不丢绝对内容/birth-death。

**消融**：

1. full record（6 帧绝对 + 几何）；
2. current + delta；
3. current + explicit geometry delta（不用 latent delta）。

R4 的“不要对完整历史 6 帧做时序”我理解为“不要做 6 帧全量 token 时空 attention”，不是“删除 absolute content”。如果用户坚持字面 delta-only，作为消融，不作为默认。

### 4.2 Interface loss：KL/一致性，不是 L2 equality

- 有分布时：`KL(π(ŝ_k) || sg(π(s_k_real)))` 或 symmetric KL；
- 确定性 BC：trajectory-space consistency（endpoint、heading、curvature、speed）；
- 分 rollout depth 统计；权重小于 BC/DAgger；只在 WM fidelity 达标后启用；
- 不要求 predicted state 与 real state 的动作逐点相等；预测状态的细小差异可能有真实驾驶意义。

### 4.3 Anti-collapse：分阶段，不一次堆满

第一版：

```
normalized latent MSE
+ cosine loss
+ per-dim std / effective-rank monitor
+ OD physical auxiliary decoder
+ ego physical auxiliary (optional)
+ planner task supervision
```

只有检测到 variance/effective rank 下降，才加 VICReg covariance（小权重）。但 monitor 必须是 **gate**：rank/std 低于阈值就停止 WM 训练或加正则，不能只写日志。

### 4.4 Action sensitivity：分实体测试

| 对象 | 要求 |
|---|---|
| Ego pose / geometry | action 改变 → 必须改变；这是 WM 的因果条件 |
| 同车道/近距/TTC<3s 的交互 OD | 应改变；用相关性/方向合理性衡量，不要求所有维度显著 |
| 远处非交互 OD | 允许近似不敏感；不要人为制造虚假交互 |
| 静态 LD/Nav | 物理内容不应因 action 改变；只改变相对几何/route progress |

加 `test_action_conditioning.py` 的分组指标；不要用一个全局“OD 必须变化”的断言。

### 4.5 P2.5 tracker：Exact + LQR 两个 canary

ChatGPT 的 P2.5 只写“Real-ENV PPO canary”，没定义 tracker。建议：

- **P2.5a**：Exact PPO canary，10–20 updates，只验证 PPO/GAE/reward/buffer/日志/恢复；
- **P2.5b**：同一个 checkpoint 的 LQR tracking canary，不训练，只跑 10–20 个场景，验证 controller-realized 链；
- 两者都不作性能结论；正式 P5 才用 LQR 训练/评测。

### 4.6 1M 口径：deployable vs training-only

你已裁决 **deployable net ≤ 1M**。因此：

- `deployable`：推理时实际加载的主网络（Encoder + Fusion + Plan/Value + WM），必须 ≤1M；
- `training_only_target`：EMA Target Encoder 可以是完整副本，单独报告，不计入 deployable；
- `trainable`：当前阶段可训练参数（可能包含 target 之外的子集）；
- `optimizer_states`：Adam 的 m/v，不是网络参数，不计入 1M。

上一轮“total≤1M 所以不能有完整 EMA”结论撤回。四栏账本必须都报告；硬门只对 deployable。

### 4.7 静态 LD/Nav 监督

- 静态 LD/Nav 不作为普通 learned dynamics 的主 loss；
- 监督 = 重投影后的 geometry error + 同一 Encoder 重编码的一致性（stop-grad target）；
- Nav 只监督 route progress/几何一致性，不做动力学外推；
- 静态 LD/Nav 不进 WM 的 latent MSE 主项；如果进，必须用 mask/低权重，并报告独立指标。

### 4.8 数据规模与评测

- 最终目标仍是 10k/1k；
- P0/P1 先用 **2k/200** 验证 schema、专家采集、round-trip、训练/评测装配；
- 当前 15GB RAM，4 个 env worker；recycle 100–150；`mem_available_floor_mb=3000`；
- Calib50 迭代、Clean500 晋级、Test500/1000 只在里程碑；
- 全量 1k × 60s × 10Hz = 600k env steps，不能每个 checkpoint 都跑。

### 4.9 交通灯/收费站/锥桶 taxonomy

- OD-dynamic：车辆、行人、骑行者；
- OD-static：锥桶、护栏、施工障碍；
- LD：车道、边界、拓扑、匝道、分岔、路口、限速、收费站几何；
- Nav：道路级 route/progress；
- RoadState：交通灯、闸口状态（若做）；
- toll booth 是 `BaseBuilding`，不是车辆；cone 是 `TrafficCone`，不是动态车辆。
- 交通灯如果不在本轮范围，就不要保留假 signal 通道；如果在，定义相位/状态和是否由 WM 预测。

---

## 5. 最终合同（可直接写进新 chatgpt5p6sol_review.md）

| 维度 | 合同 |
|---|---|
| 参数 | deployable ≤1M；H=96；无 MoE；默认 1 层；EMA target training-only 单独报告 |
| 时间 | env/control 0.1s；policy/memory/WM 0.5s；6 帧含当前；6 步=3s；30 个 0.1s 参考点 |
| 动作 | `(dx,dy,dtheta)` per 0.5s，当前 ego frame；速度/曲率派生；bounds + feasibility loss |
| Plan Head | 单实例 6 次；BC oracle-state 6 步；WM 后 imagined state；同一 action schema |
| Bank | per-entity token；6 帧含当前；warmup 重复最早真实帧 + valid=0；runtime detach；encoder hash |
| Fusion | temporal per-entity（current+compensated history+delta+geometry+age/mask）→ spatial；fused `c_t` 不落 Bank |
| WM | 动态 OD 学习；静态 OD/LD/Nav/Ego 重投影+同 Encoder 重编码；RoadState 独立；action-conditioned |
| Plan loss | 6 步 action/pose/heading/curvature/smoothness/feasibility；不是单一 XY MSE |
| WM loss | normalized MSE + cosine + OD/ego physical decoder + per-modality mask；static LD/Nav 用 geometry+reencode consistency；anti-collapse 分阶段 |
| Interface | KL/一致性，分 depth，权重 < BC，WM fidelity 达标后启用 |
| Detach | future target stop-grad；runtime Bank detach；training sequence 保留图到 batch；episode 边界 detach；PPO old log-prob/value detach |
| 训练 | P0 → P1 BC → P2 DAgger → P2.5 Exact PPO + LQR canary → P3 WM → P4 Joint+LQR → P5 正式 PPO+LQR → P6 Imagined RL |
| LQR | Exact 上界；LQR canary 从 P1；正式成绩 LQR；GT 轨迹 sanity；exact/LQR 分解 |
| 数据/评测 | 最终 10k/1k；P0/P1 2k/200；4 worker；Calib/Val/Test；assembly parity；多 seed/CI |
| 测试 | timing/action kinematics/plan single instance/oracle rollout/interface/WM static reprojection/memory warmup/latent collapse/action conditioning/param/tensor/LQR GT/validation isolation/assembly parity |

---

## 6. 对 `chatgpt5p6sol_review.md` 的 contract diff

1. 删除 Stage A-C 的 10Hz 重规划；统一 2Hz 决策 + 10Hz 执行。
2. 删除“Stage F 才首次接 LQR”；LQR 从 P1 起 canary，正式 PPO 用 LQR。
3. Bank 改为 per-entity token；禁止存 fused `c_t`。
4. 6 帧含当前；删除“是否含当前”的开放项。
5. 动作从 `(dx,dy,v_next)`/`(ds,dtheta)` 改为 **`(dx,dy,dtheta)`**；速度/曲率派生。
6. Temporal 默认改为 current absolute + compensated historical + residual + geometry delta + age/mask；不做 6 帧全量 token attention；delta-only 只作消融。
7. Interface loss 从 L2 equality 改为 KL/一致性，分 depth、小权重、WM fidelity 门控。
8. Action sensitivity 从“所有 OD 必须变化”改为分实体矩阵。
9. 参数账本改为 deployable/training-only/trainable/optimizer states；deployable≤1M。
10. P2.5 明确 Exact PPO canary + LQR tracking canary。
11. 数据规模从“一上来 10k/1k”改为“2k/200 起步，最终 10k/1k”。
12. 静态 LD/Nav 监督改为 geometry + reencode consistency，不做普通 latent MSE 主项。

---

## 7. P0 spike 与门

### 7.1 Environment smoke

- 建 `.venv`，安装 `metadrive-simulator 0.4.3`、`numpy==1.26.4`、`opencv-python-headless`，跑 headless `MetaDriveEnv` reset/step；
- 测 reset 时间、FPS、RSS；
- 4 worker 压测；`mem_available_floor_mb=3000`；
- 通过条件：10 个场景 reset/step 无异常，4 worker 不 OOM，FPS 可支撑 2k/200 采集。

### 7.2 Action calibration（不是选择方案，是校准边界/标签）

- 用 100–500 个专家 episode，把 10Hz pose 转成 `(dx,dy,dtheta)`：
  - `dx,dy` 由当前帧到 t+0.5s 的 ego frame 位移；
  - `dtheta` 由 heading 差；
  - 不需要 `dtheta/ds`，避免低速除零。
- 统计 P1/P50/P99、低速/高速分布、`dy/dtheta` 相关、曲率/横向加速度超限率；
- 用 cubic Hermite 重建 30 点，测 ADE/FDE、heading/curvature 误差；
- 校准 `dx/dy/dtheta` bounds 和 `kappa_max(v)`；
- 如果校准发现 `dtheta` 噪声过大，先用低通/专家平滑，不改变动作方案；如果 `dy/dtheta` 的耦合导致大量不可行，再考虑用 `(dx,dy)` + 派生 heading 的 ablation。
- 通过条件：专家轨迹重建 ADE 小于 0.2m、P99 小于 0.5m；低速曲率超限率 <5%；LQR 跟踪专家 GT 的横向误差可接受。

### 7.3 Model/param/tensor

- 实例化 deployable 模型，四栏账本；deployable ≤1M；
- profile 单步/6 步 rollout；batch 1/8/64；最大 token/attention/activation；
- 通过条件：deployable ≤1M；6 步不 OOM；峰值显存/延迟可接受。

### 7.4 LQR

- `test_lqr_gt_tracking`：LQR 跟踪专家 3s 轨迹；
- `test_lqr_plan_tracking`：LQR 跟踪 Plan Head 轨迹；
- 报告横向/航向/速度误差、steer/throttle 饱和、jerk；
- 通过条件：GT 轨迹可跟踪；Plan Head 轨迹的失败可分解为 planner 几何 vs controller。

### 7.5 Assembly parity

- 同一 checkpoint、同一帧，训练装配 vs 评测装配的 obs/feature/action 逐位一致；
- 旧分支 50/50 out_of_road 的教训（E2）；
- 通过条件：关键张量 `max_abs_diff=0` 或低于数值容差。

---

## 8. 仍待确认的细节（不是动作方案）

1. `dx/dy/dtheta` 的具体 bounds 和 `kappa_max(v)`：P0 校准。
2. 动作分布：squashed Gaussian + logdet，还是 Beta/截断分布；P0 选实现。
3. 交通灯是否在本轮范围；若不在，删除 signal 通道。
4. 数据规模是否接受 2k/200 起步；若不接受，需要先解决 15GB RAM/4 worker 的采集成本。
5. P2.5 的 update 数/场景数/门限。
6. `v_end` 是否需要作为辅助输出（仅 LQR 诊断），还是完全派生。
7. DAgger 轮数和 expert 混合系数；旧分支的 1.0→0.7→0.4→0.2→0.0 只作参考。

---

## 9. 证据附录

### 9.1 当前仓库/环境

- 最新 `origin/pnp_ilrl` tree 只有 `docs/chatgpt5p6sol_review.md`、`docs/deepseekv4p1_argue.md`（E1）。
- `65244be` 只改 `chatgpt5p6sol_review.md`（235 insertions / 1605 deletions）；`2e8424b` 只加 `deepseekv4p1_argue.md`（841 lines）（E1）。
- Python 3.10.12、torch 2.3.0、numpy 1.26.4；MetaDrive/gymnasium/pygame/panda3d 未安装（E1）。
- RTX 4070 12GB、20 CPU、15GiB RAM、646GiB free；4 worker 建议（E1/E2）。

### 9.2 本地规划 head 证据（E2）

- SparseDrive：`projects/mmdet3d_plugin/models/motion/motion_blocks.py:48-54,79`（plan regression `(B,3*M,6,2)`）；`decoder.py:130-135`（cumsum）；`motion_planning_head.py:441-448`（loss on deltas）；config `ego_fut_ts=6`。
- VAD/LAW：`VAD_head.py:417-424,792-794`（`(B,3,6,2)`）；`VAD_head.py:1138-1147`（L1 on deltas）；`VAD.py:423-432`（cumsum）；`waypoint_query_decoder.py:114-119,252-255`（`(B,6,2)` L1）。
- Plan-R1：`model/PlanR1.py:293-303`（SE(2) transition）；`transforms/token_builder.py:55-58`（`(dx,dy,dtheta)`）；`config/train/plan.yaml:5-7`（5 steps @10Hz = 0.5s）。
- SparseDriveV2：`sparsedrive_config.py:31-40`（path/vel/traj vocab）；`custom_decoder.py:238-267`（path/velocity CE, traj loss on x,y）；`scripts/cluster/cluster_anchor.py:59-75`（cumsum speed → path arc length）；`batch_lqr.py:28-57`（LQR）。
- WoTE：`WoTE_model.py:873-891`（pose offset, heading `tanh*π`）；`WoTE_model.py:652-681`（abs pose output）；NavSim LQR。
- Epona：`models/model.py:239-249`（`(B,15,3)` x,y,yaw）；`configs/dit_config_dcae_nuplan.py:16,63`（15 tokens, 5Hz）；无控制器。
- drivelaw：`diffusion_planner.py:96-110,890-901`（`(B,8,3)` pose, norm/denorm）；`videodrive_agent.py:395-439`（Trajectory）；NavSim LQR。
- DriveSuprim：`drivesuprim_model.py:196-247`（vocab `(8192,40,3)`, argmax/refine）；`drivesuprim_loss_fn.py:86-99`（IMI CE）；NavSim LQR.
- HorizonDrive：`datasets/nuscenes_dataset.py:729-751`（action `(dx,dy,dyaw)`）；无 planning head。

### 9.3 旧分支证据（E2，仅风险）

- `origin/Planner_RL:config/env.yaml`：`policy_dt=0.5`、`mpc_dt=0.1`、`history_frames=6`。
- `origin/Planner_RL:env/tracking.py`：`(ds,dtheta)` → 常曲率 30 点；ExactTracker 在 `env.step` 后写位姿；LqrTracker。
- `origin/Planner_RL:net/param_probe.py`：H=128 default total 1,278,879 / active 871,478；structured total 1,510,570 / active 994,717。
- `origin/Planner_RL:AGENT.md`：IDM+PP 0.756；模型 0.56；LQR+repeat 0.06；LQR+WM plan 0.56；E_interface h1 0.64→h5 1.74。
- `origin/Planner_RL:docs/deepseekv4p1_argue.md`：训练/评测装配不一致导致 Calib50 50/50 out_of_road；训练域 v≈8.3 vs 评测域 v≈3.0 的策略自诱发域差；LQR 预瞄只是缓解不是修复。

### 9.4 外部参考

- SparseDrive: arXiv:2405.19620（end-to-end sparse; planning head outputs per-step `(dx,dy)` deltas）。
- VAD: arXiv:2305.13213（vectorized scene; waypoint deltas）。
- UniAD: arXiv:2306.16927（planning-oriented; waypoint trajectory）。
- Epona: arXiv:2506.24113（video diffusion world model; trajectory head `(x,y,yaw)`）。
- DriveLaW: arXiv:2512.23421（latent world model + diffusion planner; local pose actions）。
- 其他本地 repo（SparseDriveV2、WoTE、DriveSuprim、Plan-R1）以其代码路径为准；未逐一核实 arXiv 编号，避免猜测。

---

*本轮只改本文档；不修改 `chatgpt5p6sol_review.md` 或任何代码。*
