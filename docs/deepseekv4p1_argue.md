# deepseek-v4.1 argue：对 a93c60b 初版方案的审阅、分歧与可行性排模

> 审阅对象：`docs/chatgpt5p6sol_review.md` @ `a93c60b04629bc1d96692032bfeb90f48e9393b1`
>
> 审阅范围：需求一致性、方案结构、MetaDrive 0.4.3 可行性、参数/数据/算力、训练继承性、评测口径。
>
> 证据分级：
> - **E1**：当前仓库、依赖源码、命令输出可直接复核的事实；
> - **E2**：`origin/Planner_RL` 旧分支的实测记录（不同 schema/不同架构，只作风险证据与工程参照，**不作新方案的架构权威**）；
> - **E3**：估计或待验证项，已在文中标注。
>
> 编号约定：**R# = 你列出的 16 条需求编号**（例如 R9 = “IL 先训练 latent WM…”，R16 = “一开始不要上 latent WM…”）；**K# = 本文风险登记表编号**，与需求编号区分。
>
> 结论先行：**初版是一份高质量的“约束清单”，但不是一份可以直接交给 Coding Agent 的单一契约。** 它至少有 6 个 blocking 级问题，其中 3 个来自需求本身的冲突（R3/R9/R10/R16），必须由你裁决；另外 3 个是方案内部矛盾（时间尺度、动作语义、Plan Head 继承）。在这些问题解决前直接开工，最可能得到的是“Stage A 能跑、Stage D 换了一个任务”的系统。

---

## 0. 结论摘要

### 0.1 可以接受的部分

- 输入模态划分（OD / LD / 道路级 Nav / Ego）与“环境语义必须过 Encoder、控制元数据可直接用”的边界，方向正确。
- 真实 Memory Bank 与 Imagine Memory Bank 隔离、场景 reset、未来 latent stop-gradient、EMA target 的基本方向正确。
- 先 Exact Tracking 证明 Plan Head、再引入 WM/LQR 的分阶段思想，方向正确，且与旧分支“Exact 0.56、LQR repeat 0.06”的教训一致（E2）。
- 评测强调冻结集、闭环、多 seed、按场景分桶，而不是 open-loop ADE/FDE，方向正确。
- 对 Expert 特权信息、Nav 泄漏、坐标变换、ID 对齐、detach 合同、failure taxonomy 的重视，是正确的工程直觉。

### 0.2 必须修改的 blocking 问题

| # | 问题 | 文档位置 | 为什么 blocking | 建议 |
|---|---|---|---|---|
| B1 | **0.5 s 规划间隔 / 0.1 s 控制间隔 / 10 Hz 重规划 / PPO 动作 horizon 四者不一致** | §6.1、§6.5、§12、§13 | Stage A-C 每 0.1 s 重规划一次 0.5 s 动作但只执行 0.1 s；PPO 的 transition 是 0.1 s，动作却按 0.5 s 定义，log-prob 与 executed action 不对应。且与文档自己的硬约束 §22.2“6 步 rollout”矛盾 | 固定为 **2 Hz 决策 + 10 Hz 控制**：每 0.5 s 产生 6 步 3 s 轨迹，LQR/Exact 执行第一段 5 个 0.1 s 子步；PPO 只对 `a0` 采样，`a1..a5` 为同策略+WM 的确定性尾 |
| B2 | **动作参数化 `(dx,dy,v_next)` 欠约束，与 6 步拼接/LQR/PPO 不匹配** | §6.1、§6.2、§6.3、§7.2 | `dx,dy` 已经隐含了 0.5 s 内的平均速度，再输出 `v_next` 会冗余且可自相矛盾；heading/curvature 只能事后差分，容易出现不可行轨迹和插值过冲 | 优先采用 **`(ds,dtheta)`（0.5 s 弧长+航向变化）**，可选 `v_end/a_long`；用常曲率圆弧做唯一积分与插值。若坚持 `(dx,dy,v)`，必须加可微运动学一致性层与损失 |
| B3 | **Plan Head 在 Stage A-C 与 Stage D-F 做的是两个不同任务** | §6.1、§13 A1/A2/B/C/D/E | Stage A-C 训练的是“每 0.1 s 重规划、0.5 s horizon”的头；Stage D-F 才要求“同一头自回归 6 次、3 s”。输入分布、动作 horizon、状态来源全变。这直接违反 R13/R14“同一规划头贯穿、有继承性” | 从 Stage A 就固定 6 步接口：用 **oracle-state rollout**（真实未来 latent 作为每步输入）训练 6 步动作；WM 可用后加 **B2 interface loss**。不要先训 10 Hz 头再换任务 |
| B4 | **R9 与 R16 直接冲突，文档静默选择了 R16** | §13、§20 | R9 要求“先 WM，再 policy/value，再 DAgger，再 RL”；R16 要求“先 plan head IL+DAgger+RL，再 WM”。两者不能同时成立。文档按 R16 排了 A-C-D，但没有声明取舍 | 已解释两者原文；推荐 §4.2 的折中路线：plan head BC → WM → joint/LQR → DAgger → 正式 RL，P2.5 保留很短的 Exact PPO canary。待 chatgpt5p6sol 审阅 |
| B5 | **WM 的 temporal→spatial→latent 输出顺序与文档 §4.4 的“融合在 Plan/Value/WM 之前”不一致；静态 LD/Nav 到底是预测还是重投影也没定死** | §4.4、§5.2、§7.3、§9.2 | R4 明确说 WM 先时序（当前帧+历史 delta）、再空间、再输出与单帧同构的 OD/LD/Nav/Ego latent，压入 imagine bank 并给 policy/value。文档把它拆成“encoder 融合 + WM 残差 + 静态解析重投影”，但没有给出唯一数据流，也没有声明这是 hybrid WM | 定义一个共享的 `StateFusion`（temporal→spatial）作为 WM 的状态更新；bank 只存 per-entity `z_entity`，fusion 现算；静态 LD/Nav 明确采用“确定性重投影+同一 Encoder 重编码”，并加同构/静态漂移测试。不要把它说成纯 learned latent WM |
| B6 | **1M 参数预算没有区分 total / active / EMA copy；旧分支证据显示“1M active”与“1M total”是两回事** | §8.1、§8.2、§10.3 | 文档的预算表相加 0.75–1.19M，但没算 LayerNorm/embedding，也没算 EMA target 的整份 encoder 副本。旧分支 H=128 时 active≈0.995M，total≈1.28–1.51M（E2）。**已裁决 total≤1M**，因此不能保留整份 EMA copy | 采用五栏账本（total / reachable / trainable / frozen / dormant）；H=96、无 MoE、多数 block 1 层、EMA 只保留小型 target projector 或直接用 online encoder stop-grad；总预算 envelope 见 §3.12 |

### 0.3 高风险但非 blocking 的问题

- 旧分支的 Exact/LQR 差距与“训练/评测装配不一致”是真实发生过的灾难（E2）；新方案必须把 **train/eval assembly parity** 作为 P0 硬门，而不是只写“保存 resolved config”。
- MetaDrive 0.4.3 在当前环境**尚未安装**；`/tmp/opencode/metadrive043` 只有源码，缺 `pygame/gymnasium/panda3d`。当前机器 15 GB RAM、约 8 GB available，旧配置的 16 workers×0.65 GB 很可能 OOM。
- 10k 场景、1k 验证集的采集/评测成本、DAgger 多轮成本、以及 1000 场景全量评测的频率，文档没有给出可执行的预算。
- 1000 验证集同时用于 checkpoint selection 和最终结论，会破坏“冻结”语义；需要 Calib/Val/Test 分层。
- `(dx,dy,v)` 与 `(ds,dtheta)` 的选择会级联影响 WM 条件、LQR、PPO、插值、loss、probe；这不是一个“实现细节”，而是架构决策。

### 0.4 推荐方案摘要（给 chatgpt5p6sol 的审阅基线）

你已经裁决 **total ≤ 1M**；下面是我建议初版收敛到的单一契约。理由和细节见 §2–§4，审阅时可直接按这 9 条逐条 challenge。

1. **参数口径：total ≤ 1M。** H=96；无 MoE；Temporal/Spatial/WM 默认 1 层（Spatial 可 2 层但需账本余量）；**不做整份 EMA encoder 副本**。WM 阶段 encoder 冻结时直接用 online encoder 作为 target（stop-grad）；若必须 EMA，只 EMA 一个小 target projector（约 20–30k）并计入 total。用五栏账本（total/reachable/trainable/frozen/dormant）做 CI 硬门。
2. **时间契约：** `policy_dt=0.5 s`（2 Hz 决策）、`control_dt=0.1 s`（10 Hz 执行）、`history=6` 帧含当前、`rollout=6` 步 = 3 s、轨迹 30 个 0.1 s 参考点。每 0.5 s 重规划一次，LQR/Exact 执行第一段 5 个 0.1 s 子步；剩余 5 个动作只作 preview。**PPO 只对 `a0` 采样**，`a1..a5` 用同一 policy 的均值 + WM 生成；reward 在 0.5 s 窗口上聚合。
3. **动作：** `(ds, dtheta)`（0.5 s 弧长 + 航向变化），用常曲率圆弧作为唯一积分/插值实现；可选 `v_end/a_long` 辅助，但必须加 `v_end ≈ ds/0.5` 的一致性损失。**不把 `(dx,dy,v)` 作为主动作**；它需要额外的可微运动学层，且 6 步拼接/PPO/LQR 都会更复杂。
4. **Plan Head：** 只有一个实例，每步调用 6 次；**从 P1 就训练 6 步接口**：B1 oracle-state rollout（真实未来 latent 作为每步输入）+ policy-only rollout；WM 可用后加 B2 interface loss `Σ w_k ||π(ŝ_k) − sg(π(s_k_real))||`。不允许 Stage A-C 用 10 Hz 重规划、Stage D-F 再换成 6 步。
5. **WM 数据流（two-tier state）：** bank 只存 **per-entity、per-frame 的 `z_entity`**（OD/LD/Nav/Ego + anchor/mask/id），**不存 fused context**；每次 plan query/imagined step 重新计算 `c_t = StateFusion(temporal(current + Δhistory) → spatial)`；Plan/Value 读 `c_t`；WM 以 `z_entity` 为单位做 transition，输出下一帧 `z_entity`。目标 encoder 必须编码 `z_entity`，不是 `c_t`。
6. **静态 LD/Nav：** 明确采用 hybrid WM：动态 OD（和 road-state，若做）学习预测；静态 LD/Nav 由 `StaticReprojector(predicted ego pose)` + **同一个 encoder** 重编码；Ego 由确定性 ego dynamics 推进 + 重编码。不要声称这是纯 learned latent WM；加 `test_wm_static_reprojection` 和静态漂移测试。
7. **WM 条件化与监督：** action embedding 进入**每个动态预测层**（不只 ego latent）；加 action-shuffle/反事实损失；latent loss 用归一化 MSE + cosine + VICReg/BYOL variance/covariance + OD/ego 物理 decoder + per-modality mask；静态 LD/Nav 不进入主 latent loss；EMA 只在 encoder 解冻时使用，且不与 `detach_encoder_from_wm_loss=true` 同时宣称“防坍缩”。
8. **训练顺序（推荐折中）：** P0 环境/数据/schema → P1 Plan Head BC（6 步 oracle-state，Exact）→ P2 DAgger（Exact + LQR canary）→ **P2.5 Exact PPO canary（10–20 updates，只证明 RL 管线，不作最终结论）** → P3 WM 1/2/3/6 步（encoder/plan 冻结）→ P4 joint Plan+WM（LQR）→ P5 正式 PPO（LQR，KL/BC 正则）→ P6 imagined RL。这样保留 R16 的“plan head 先跑通”，同时避免 R16 字面顺序里“先 Exact RL、后 LQR”造成的部署断裂；也部分满足 R9 的“WM 在 policy 强化之前”。
9. **数据/评测：** 最终目标仍是 10k/1k，但 **P0/P1 先用 2k/200 子集**验证管线与预算；4 个 env worker（不是 16/20），`use_render=False`、recycle、`mem_available_floor_mb=3000`；2 Hz 决策窗口存储，10 Hz 帧用于 lookup；Calib50 / Val500 / Test500 分层，Test 不调参；LQR 从 P1 就作为并行诊断，Exact 只作上界。

这 9 条如果被接受，我建议把初版 `docs/chatgpt5p6sol_review.md` 重写成一份“单一契约 + 分阶段门”的短文档，而不是在 1751 行里保留多种可选解释。

---

## 1. 现状可行性排模

### 1.1 仓库与分支现状

- 当前分支 `pnp_ilrl` 只有一份 tracked 文件：`docs/chatgpt5p6sol_review.md`（1751 行）。`git show --stat a93c60b` 显示该 commit 只新增了这份文档，没有代码（E1）。
- `origin/pnp_ilrl` 同样只有这份文档；`origin/Planner_RL` 有 318 个文件，是上一代完整实现（E1）。
- 因此当前是 **greenfield**：没有可复用的新分支代码。旧分支可以作为 MetaDrive API、场景生成、Exact/LQR、数据采集的“事实参照”，但它的 schema、网络、训练顺序与新方案不同，不能直接移植。
- 目录里没有 `.codegraph/`，也没有 `env/`、`net/`、`pipeline/` 等目录；当前只能做文档级审阅（E1）。

### 1.2 运行环境现状

命令输出（E1）：

```text
Python 3.10.12
torch 2.3.0a0
numpy 1.26.4
metadrive: MISSING
gymnasium: MISSING
gym: MISSING
pygame: MISSING
panda3d: MISSING
```

- `/tmp/opencode/metadrive043` 有 `metadrive_simulator-0.4.3` 源码与 dist-info，但 `import metadrive` 在导入 `pygame` 时失败（E1）。
- 旧分支的 `tools/venv-python` 依赖仓库根目录 `.venv`；当前 checkout 没有 `.venv`（E1）。
- MetaDrive 0.4.3 的依赖声明要求 `gymnasium>=0.28`、`pygame`、`panda3d==1.10.13`、`panda3d-gltf==0.13`、`numpy>=1.21.6`，Python `<3.12`（E1，`/tmp/opencode/metadrive043/metadrive_simulator-0.4.3.dist-info/METADATA`）。当前 Python 3.10 满足。
- 旧 README 的已知可用安装方式是：`python3 -m venv --without-pip --system-site-packages .venv`，安装 `metadrive-simulator 0.4.3`、`numpy==1.26.4`、`opencv-python-headless==4.10.0.84`，再跑 `tools/setup_gl_libs.sh` 注入 glvnd（E2）。新分支应把它作为 P0 的第一步，而不是假设环境已就绪。

**结论：当前不能直接跑 MetaDrive 闭环。** 在写任何网络代码前，必须先做一个 P0 environment spike：建 venv、跑一个 headless `MetaDriveEnv` reset/step smoke、测 reset 时间/FPS/RSS。否则后面所有“可行性”都是纸面推断。

### 1.3 算力/内存/存储

命令输出（E1）：

```text
GPU: NVIDIA GeForce RTX 4070, 12282 MiB
CPU: 20
RAM: 15 GiB total, ~8 GiB available (buff/cache 5.2 GiB)
Disk: 646 GiB free on /workspace
```

- **GPU 12 GB**：1M 参数、6 帧 history、6 步 rollout 的模型本身可以放下；但旧分支的记录是 6 步 ST-GNN 展开约 32 MB/样本，micro batch 256 是 12 GB 卡的上限，384 会 OOM（E2，`config/train.yaml`）。新模型必须重新标定 micro batch，不能照搬 256。
- **RAM 15 GB / 8 GB available**：旧分支的 `config/env.yaml` 写 16 workers × ~0.65 GB ≈ 10.4 GB（E2）；在当前机器上很可能 OOM。旧评测配置还写了 `mem_available_floor_mb: 3000`、`recycle_every_specs: 150`、`train_pool_policy: pause`（E2）。新方案必须把 worker 数降到 **4**，并保留回收/暂停策略。MetaDrive 一个进程一个 engine（E2，`pipeline/trainer.py:6573`）；旧分支 6 个 eval worker≈9.5 GB（E2，`config/train.yaml:152`）——20 CPU 不等于 20 env。
- **Disk 646 GB**：足够 10k 场景的原始/处理后数据；但需要明确存储格式与压缩，否则 DAgger 多轮会迅速膨胀。

### 1.4 MetaDrive 0.4.3 关键事实（E1，依赖源码）

| 事实 | 证据 | 对方案的影响 |
|---|---|---|
| `env.step` 默认 0.1 s | `metadrive/envs/base_env.py:188-190`：`physics_world_step_size=2e-2`，`decision_repeat=5` | 控制帧 0.1 s 可行；规划 0.5 s = 5 个 env step |
| 动作是连续 `[steering, throttle]` | `metadrive/policy/base_policy.py`；旧专家注释引用 `component/vehicle/base_vehicle.py:473-481` 与 `max_steering=40` | Plan Head 输出轨迹后必须经 LQR/Exact 转成 `[steer, throttle]`，不能直接喂 `(ds,dtheta)` |
| `TollGate` block 存在 | `metadrive/component/pgblock/tollgate.py:11`；`component/buildings/tollgate_building.py:7` | 收费站场景可生成；但 toll booth 是 `BaseBuilding`（静态），不是车辆 |
| `TrafficCone` 存在 | `metadrive/component/static_object/traffic_object.py:44`；`type.py:72` | 锥桶可作为静态对象；不应与动态车辆共用“动力学预测”语义 |
| 官方 `IDMPolicy` 使用 lane-level routing | `metadrive/policy/idm_policy.py:177` 起：`navigation.current_ref_lanes`、`routing_target_lane`、`lane_change_policy` | IDM/Pure Pursuit 作为 Expert 必然带 lane-level 特权；必须审计并禁止进入 Student 输入 |
| 交通灯可开关 | 旧分支 `config/env.yaml` 明确 `traffic_lights: false`（E2） | 新方案若不做交通灯，就不要保留“假 signal 通道”；若做，需定义相位时钟/WM 责任 |

### 1.5 旧分支可复用的“事实”与不可复用的“经验”

旧分支 `origin/Planner_RL` 的实测记录（E2，**不同 schema/不同架构，不能当新方案目标**）：

- `config/env.yaml`：`policy_dt=0.5`、`mpc_dt=0.1`、`history_frames=6`、`topk_objects=16`、`topk_lanes=16`；这证明 2 Hz 策略 + 10 Hz 控制在 MetaDrive 0.4.3 上是跑通的。
- `env/tracking.py`：`interpolate` 把 6 个 `(ds,dtheta)` 动作按常曲率圆弧展开为 30 个 10 Hz 位姿；`ExactTracker` 在 `env.step` 之后直接把 ego 写到位姿；`LqrTracker` 通过 `engine.add_policy` 输出 `[steer, throttle]`。这说明 Exact 是“运动学播放”，不是物理闭环。
- `net/param_probe.py`：H=128 时 default total=1,278,879、structured total=1,510,570；active=871,478/994,717。也就是说 **1M active 是已证明可行的，但 1M total 不是**。
- `AGENT.md` 基线：IDM+PP success=0.756，当前模型=0.56；LQR+repeat=0.06，而 LQR+WM plan=0.56。这说明 LQR 不是“接上就行”，而是一个独立的失败源。
- `docs/deepseekv4p1_argue.md`：旧分支曾出现训练/评测装配不一致导致 Calib50 50/50 out_of_road；也出现训练域 v≈8.3 m/s、评测域 v≈3.0 m/s 的策略自诱发域差。这些是**风险证据**：新方案必须把 train/eval assembly parity 和速度/横向分布审计作为硬门。

**不要从旧分支继承的东西**：它的 OD/LD schema、`lane_semantic` target-lane 输入、MoE 结构、Stage A WM-first 顺序、`(ds,dtheta)` 的具体实现细节。**可以继承的只是事实**：MetaDrive 的时序、动作接口、场景 block、Exact/LQR 的工程陷阱、参数/内存标定方法。

### 1.6 可行性结论表

| 维度 | 结论 | 依据 |
|---|---|---|
| MetaDrive 闭环 | **可行，但当前未安装**；P0 必须先建 venv + smoke | E1 |
| 2 Hz 规划 + 10 Hz 控制 | **可行**，MetaDrive 默认 0.1 s；旧分支已跑通 | E1/E2 |
| 1M 参数 | **可行但紧**：1M active 有旧证据；1M total 需要 H=96/无 MoE/无完整 EMA copy | E2 + 估算 |
| 6 帧 history + 6 步 rollout | **可行**；显存需按 micro batch 标定 | E2 |
| 10k/1k 场景 | **可行但采集/评测成本高**；当前 RAM 限制 worker 数 | E1/E2 |
| LQR 闭环 | **高风险**；旧分支有 LQR 崩溃与域差证据 | E2 |
| 直接按当前文档开工 | **不建议**；B1–B6 未解决 | 本审阅 |

---

## 2. 需求内部冲突：必须先裁决

### 2.1 R9 vs R16

- **R9（需求第 9 条，原文）**：“IL先训练latent WM，再训练policy/value head，然后dagger，最后RL。”顺序是 **WM → policy → DAgger → RL**。
- **R16（需求第 16 条，原文）**：“一开始不要上latent WM，而是让剩余net特别是plan head先跑通IL+Dagger+RL（这时候用exact tracking而非LQR）。确保plan head在真实ENV下有好的闭环性能，然后再让latent WM逼近真实ENV，并与plan head相互磨合（这时候需要rollout和LQR）。”顺序是 **plan head IL+DAgger+RL → WM → joint/LQR**。
- 这两条不能同时成立。初版文档在 §13/§20 静默选择了 R16，但没有声明“R9 被降级为最终目标而非执行顺序”。

**我的判断**：R16 的出发点（可调试、先证明 plan head）是对的；R9 的出发点（WM 先学环境、policy 在 WM 里学）是 model-based RL 的经典顺序，但在这个 1M 模型 + 新 schema + 旧分支 TF/部署 gap 8–10× 的背景下，风险更高。建议不要二选一，而是采用折中：

> **plan head BC（6 步 oracle-state） → WM（1/2/3/6 步） → joint + LQR → DAgger（LQR/Exact） → RL（LQR） → imagined RL**

这样既保留“先证明 plan head”，又让 WM 在 RL 之前与 LQR 磨合，避免 PPO 在 Exact 下过拟合后再被 LQR 打破。它偏离 R16 的“先 RL 再 WM”，但更接近 R9/R10 的“WM 与 LQR 在 IL/DAgger 阶段就介入”。**这是建议，不是擅自改需求；需要你确认。**

### 2.2 R10 vs R16

- **R10**：IL/DAgger 阶段 policy/value head 在 rollout 6 后得到 3 s 的 6 个动作，连成轨迹给 LQR。
- **R16**：第一阶段用 Exact Tracking 而非 LQR，先确保 plan head 在真实 ENV 下有好的闭环性能。
- 这两条也不完全兼容：R10 要求 IL/DAgger 就用 LQR；R16 要求第一阶段 Exact。初版文档把 LQR 推迟到 Stage F，等于选择了 R16，但 R10 的“IL/DAgger + LQR”没有实现。

**建议**：把“Exact”和“LQR”定义为两种评测/执行模式，而不是训练阶段的硬分界：
- Exact 用于 **plan head 上界**、数据采集、快速迭代；
- LQR 用于 **controller-realized** 评测和最终 RL；
- 在 DAgger 阶段至少做一次 **LQR canary**，不要等到 Stage F 才第一次接 LQR。
- 如果 R10 是硬需求，则 IL/DAgger 的正式评测必须报告 LQR；Exact 只作为诊断。

### 2.3 R3/R10/R13/R14 vs 文档 §6.1

文档 §6.1 原文：

> “Stage A-C 在真实 ENV 中按固定控制周期滚动重规划，建议基线为 10 Hz（0.1 s），Exact Tracker 每次只执行首个 0.1 s；Stage D-F 才在冻结的 rollout-t0 上复用同一个 Step Plan Head 自回归 6 次，形成 3 s 轨迹。”

问题：
- R3 说“规划帧间隔 0.5 s，控制帧间隔 0.1 s”；§6.1 的 Stage A-C 是 **0.1 s 规划**，不是 0.5 s。
- R10 说 IL/DAgger 就要 6 步 → 3 s → LQR；§6.1 的 Stage A-C 没有 6 步。
- R13/R14 说 Plan Head 贯穿、有继承性；§6.1 让同一个头在 Stage A-C 做 0.5 s 重规划、在 Stage D-F 做 6 步自回归，任务语义变了。
- 文档自己的硬约束 §22.2 说“Plan Head 每次输出下一 0.5 s ego motion，rollout 6 次形成完整 3 s 轨迹”；§6.1 的 Stage A-C 不满足。

**这是文档最严重的内部矛盾。** 必须二选一：
- **A. 目标接口优先**：所有阶段都是 2 Hz 决策、6 步 3 s 轨迹、10 Hz 执行；Stage A-C 用 oracle-state/policy-only rollout 代替 WM。这样 R3/R10/R13/R14 都满足，但 Stage A-C 的实现比“单帧 10 Hz 重规划”复杂。
- **B. 可调试优先**：Stage A-C 用 10 Hz 重规划、0.5 s horizon；那么它就是一个 **不同的控制器/任务**，不能声称是同一个 Plan Head 的继承；R10/R13/R14 需要改写。

我建议 A。理由不是“旧分支这么做”，而是：只要决策 epoch 与动作 horizon 不一致，PPO 的 log-prob、reward 聚合、LQR 参考、memory 更新全都会出现二义性；后期再改会推翻已经训好的头和缓冲区。

### 2.4 建议的裁决表

| 冲突 | 选项 A | 选项 B | 建议 |
|---|---|---|---|
| R9 vs R16 | WM-first | plan-head-first | **折中：plan head BC → WM → joint/LQR → DAgger → RL**；若你坚持 R16，则至少在 RL 前做 LQR canary |
| R10 vs R16 | IL/DAgger 就用 LQR | 先 Exact 后 LQR | **Exact 做上界/采集，LQR 做正式评测**；DAgger 至少一次 LQR canary |
| 规划 epoch | 2 Hz + 6 步 | 10 Hz + 0.5 s | **2 Hz + 6 步**；10 Hz 只作为消融 |
| 动作参数 | `(ds,dtheta)` | `(dx,dy,v)` | **`(ds,dtheta)` 主方案**；后者需要额外运动学一致性层 |
| 静态 LD/Nav | 解析重投影+重编码 | WM 学习预测 | **解析重投影+重编码**；声明为 hybrid，不冒充纯 latent WM |

---

## 3. 逐项审阅

### 3.1 时间尺度与决策 epoch（§6.1、§6.5、§12、§13）

**文档写法**：Plan Head 每次输出 0.5 s motion；Stage A-C 10 Hz 重规划，只执行首个 0.1 s；Stage D-F 6 步自回归；PPO 输出 0.5 s 轨迹动作分布。

**问题**：
1. 如果每 0.1 s 重新查询一次 0.5 s 动作，那么策略的实际决策频率是 10 Hz，动作 horizon 是 0.5 s，执行 horizon 是 0.1 s。PPO 的 transition 是 0.1 s，但 `log_prob` 对应的是 0.5 s 动作；这个动作的 0.4 s 尾部被丢弃，下一帧又采样一个新动作。这不是标准的 MDP，GAE 的 credit assignment 会错位。
2. Memory bank 的帧间隔是 0.5 s；如果 plan 每 0.1 s 查询，bank 在 5 次查询中不变，plan head 看到的是同一个历史但不同的当前帧（如果当前帧单独编码）。文档没有定义当前帧与 bank 的关系。
3. Stage A-C 不跑 6 步，Stage D-F 才跑 6 步；同一 Plan Head 的输入分布和动作语义变了。
4. 文档 §22.2 的硬约束与 §6.1 直接矛盾。

**证据**：
- MetaDrive `env.step=0.1 s`（E1）。
- 旧分支 `policy_dt=0.5`、`mpc_dt=0.1`、6 个动作 → 30 个 10 Hz 点（E2，`config/env.yaml`、`env/tracking.py`）。
- 旧分支 `pipeline/eval_runner.py:2571` 的 `decision_interval=5`、`env/obs/memory.py:1-5` 每 5 个 env step 存一帧、`env/tracking.py:72-84` 把每个 0.5 s 动作展开成 5 个 0.1 s 位姿；它执行整段，不是只执行首个 0.1 s（E2）。这与文档 Stage A-C 的“只执行首个 0.1 s”相反。
- 旧分支 AGENT.md 的 C2：PPO 只把 `a0` 当随机变量，`a1..a5` 是冻结 WM + 冻结 tail policy 的确定函数（E2）。这是处理“6 步轨迹但只执行首段”的正确方式。

**建议**：
- 固定 `policy_dt=0.5`、`control_dt=0.1`、`rollout_steps=6`、`history_frames=6`（含当前帧）。
- 每 0.5 s 产生 6 个动作（3 s 轨迹），LQR 执行第一段 0.5 s（5 个 0.1 s 子步），然后重规划；剩余 5 个动作仅作为 preview/诊断。
- PPO 的随机变量只有 `a0`；`a1..a5` 用同一 policy 的均值（或冻结 tail）与 WM 生成，不参与 log-prob。Reward 在 0.5 s 的 5 个控制步上聚合；任一步终止则 done。
- 如果要做 10 Hz 重规划，定义成独立消融 `policy_dt=0.1, action=(ds,dtheta) for 0.1 s`，不要和 0.5 s 动作混用。

### 3.2 动作参数化（§6.1、§6.2、§6.3、§7.2）

**文档写法**：`a_n = (dx_n, dy_n, v_{n+1})`，位于当前 imagined ego frame；6 次输出经坐标累计形成 3 s 轨迹；用 cubic Hermite 插值到 0.1 s；从相邻点派生 heading/curvature。

**问题**：
1. `dx,dy` 已经决定了 0.5 s 内的位移和平均速度；再输出 `v_next` 是冗余的。若 `||(dx,dy)||/0.5` 与 `v_next` 不一致，轨迹的“点”和“速度”会矛盾；LQR 该信哪个？
2. `dx,dy` 没有直接给出 heading 变化；heading/curvature 只能从相邻点差分。对于小位移、原地转向、低速大曲率，差分噪声很大。
3. 6 步拼接时，每一步的 `dx,dy` 在各自的 ego frame；累计需要先知道每步的 heading 变化。文档说从点派生 heading，但这是一个后处理闭环，容易出现不一致。
4. PPO 在 3D 动作上采样；`v_next` 的可行域依赖当前速度，固定 bounds 不合适。
5. cubic Hermite 可能过冲；文档自己也承认要“不会明显过冲”。这说明参数化本身不够安全。

**旧分支的证据**：旧分支用 `(ds,dtheta)`，`ds ∈ [0,10]`、`dtheta ∈ [-0.6,0.6]`，通过常曲率圆弧 `interpolate` 精确展开为 10 Hz 位姿；`arc_step/compose_pose` 是唯一积分实现（E2）。具体代码：`net/model.py:141-143` 定义动作，`:371-383` 转 `(dx,dy)`/heading，`:395-415` 插值到 10 Hz；`net/path_head.py:12-17` 做 6 步组装。这套参数化在旧分支上跑通了 Exact 0.56、LQR+WM plan 0.56——不是说新方案必须照抄，而是证明它至少是自洽的。

**建议**：
- 主方案：`a = (ds, dtheta)`，0.5 s 弧长 + 航向变化；`kappa = dtheta / max(ds, eps)`；用常曲率圆弧积分。可选输出 `v_end` 或 `a_long` 作为辅助速度目标，但轨迹仍由 `(ds,dtheta)` 决定。
- 如果必须保留 `(dx,dy,v)`：增加一个可微运动学层，把 `(dx,dy)` 转成 `(ds,dtheta)`，并加 `L_kin = ||(dx,dy) - arc(ds,dtheta)||` 与 `L_speed = ||v_next - ds/0.5||`；否则不要用它做 6 步拼接。
- 插值：如果主方案是 `(ds,dtheta)`，直接用常曲率圆弧，不需要 cubic Hermite；如果坚持位置点，Hermite 必须做曲率/速度限幅，并加 overshoot 测试。
- 在文档中把“动作语义”写成不可变的 hard contract，列出 `ds/dtheta` 的 bounds、单位、积分公式、LQR 参考点数量。

### 3.3 Plan Head 继承与 6 步 rollout（§13）

**文档写法**：
- Stage A1：单帧 BC + Exact，无 memory/WM。
- Stage A2：加 6 帧 memory + temporal fusion。
- Stage B：DAgger + Exact。
- Stage C：真实 MetaDrive PPO，无 WM。
- Stage D1/D2：单步/多步 WM。
- Stage E：Planner + WM 联合。
- Stage F：LQR。
- Stage G：imagined RL。

**问题**：
1. Stage A1/A2/B/C 的 Plan Head 只做“单步 0.5 s 动作 / 10 Hz 重规划”；Stage D-F 才要求“同一头自回归 6 次”。这不是继承，而是换任务。Plan Head 在 Stage A-C 学到的输入分布（真实当前状态、0.5 s horizon）与 Stage D-F 的输入分布（WM 预测状态、6 步自回归）不同；Stage D 一上来会出现 exposure bias。
2. 文档没有 **oracle-state rollout**：即用真实未来帧的 latent 作为每一步的输入，监督 6 步动作。这是让 Plan Head 在 WM 不存在时也学会 6 步接口的标准做法。旧分支的 B1/B2 阶段有类似设计（E2，`AGENT.md`）。
3. 文档没有 **interface loss**：`L_if = Σ w_k ||π(ŝ_k) - sg(π(s_k_real))||`。旧分支的 `E_interface` 从 h1 0.64 增长到 h5 1.74，WM-state action error≈1.43 vs real≈0.40（E2）；说明“在预测状态上调用同一个头”是独立的误差源，必须显式训练。旧分支 `pipeline/stages.py:2061-2080` 的 WM 教师强制链会逐步调用 plan head 的 `ego_next`（E2），`:2462` 用 6 步 ego KPI 评测；可以作为“逐步接口”的工程参照。
4. Stage C 的 PPO 如果使用 Exact Tracking，学到的策略是 planner-oracle；Stage F 换成 LQR 后，动作分布/跟踪误差变了，可能出现旧分支的“Exact 0.56 → LQR repeat 0.06”式崩溃（E2）。文档没有规定 Stage C 的 tracker，也没有 deployment-isomorphic canary。

**建议**：
- 从 Stage A 就固定 6 步接口：
  - **B1 oracle-state rollout**：k=0..5，输入 `s_{t+k}`（真实未来帧编码），目标专家 `a_{t+k}`；权重按深度衰减（如 1.0/0.7/0.5/0.35/0.25/0.2）。
  - **B0 policy-only rollout**：没有 WM 时，用当前状态/解析 ego+静态地图推进生成 6 步；这不是为了性能，而是为了让头见过“非真实状态”。
  - **B2 interface consistency**：WM 可用后，冻结 WM，训练 `π(ŝ_k)` 逼近 `sg(π(s_k_real))`；这是 Plan Head 继承的关键。
- Stage C 明确 tracker：Exact PPO 只能叫“planner-only canary”；正式 RL 用 LQR，或至少做 C2 deployment-isomorphic PPO canary。
- 如果坚持文档的 Stage A-C 10 Hz，则必须在文档中声明“Stage A-C 的 Plan Head 与 Stage D-F 不是同一任务，继承只到 trunk/encoder，动作头需要重新适配”；这违反 R14，不能悄悄做。

### 3.4 WM 的 temporal→spatial→latent 数据流（§4.4、§7）

**文档写法**：
- §4.4：Same-ID entity temporal fusion → current-time entity representations → OD self-attn → LD self-attn → OD-LD cross-attn → Nav/Ego cross-attn → Plan/Value/WM。
- §7：WM 由 ego action/trajectory 条件化；输出动态 OD/road-state latent，静态 LD 重投影，Nav 更新，Ego 重编码。
- §2.2 循环：`M0 → Plan Head → a0 → Ego Dynamics → Latent WM → z1 → M1`。

**问题**：
1. 文档没有明确 WM 内部是否执行“temporal fusion → spatial attention → per-entity latent output”。§4.4 的融合发生在 Plan/Value/WM 之前，读起来像“encoder 先融合，WM 只做 transition”。这与 R4 的“WM 先时序、再空间、再输出 latent 给 policy/value”不一致。
2. 如果 WM 只预测一个全局 latent，而 Plan Head 需要 OD/LD/Nav/Ego 的 per-entity latent，形状/语义不匹配。
3. 如果 Plan Head 在 WM 之前读取的是 fusion 后的 state，而 WM 又输出下一帧的 per-entity latent，那么“同一个 fusion 模块”在 real 和 imagined 两条路径上是否共享权重、是否共享 mask/age embedding，文档没有说。
4. 文档没有定义 imagine bank 中每个元素的形状与 real bank 的对应关系；只说“同构同质”，没有 schema。

**建议**：
- 定义一个唯一的 `StateFusion` 模块：
  ```
  per-frame per-entity latents + anchors + masks
      → temporal fusion (current + Δ vs current)
      → spatial fusion (OD self / LD self / OD-LD cross / ego-nav cross)
      → per-entity state tokens s_od, s_ld, s_nav, s_ego
  ```
  该模块在 real t0 和每一个 imagined step 都使用**同一实例、同一权重**；Plan Head 读取 `s_*`；WM transition 输出下一帧 `z_*`，再进入 `StateFusion`。
- 把数据流画成：
  ```
  M_t (real/imagined bank)
    → StateFusion → s_t → PlanHead → a_t
    → EgoDynamics(a_t) → pose_{t+1}
    → DynamicWM(s_t, a_t, pose_{t+1}) → z_od_{t+1}, z_roadstate_{t+1}
    → StaticReproject(pose_{t+1}) → z_ld_{t+1}, z_nav_{t+1}
    → z_ego_{t+1} = EgoEncoder(pose_{t+1}, v_{t+1})
    → append to M_{t+1}
  ```
  这里 `StateFusion` 是“WM 的 inference step”，不是 encoder 的旁路。
- 对每个模态定义 `LatentSchema`（shape/dtype/semantics/mask），并加测试 `wm_output.shape == encoder_output.shape`、`same module called 6 times`、`real/imagined share weights`。
- **two-tier state 必须写死**：bank 只存 per-entity `z_entity`，不存 fused `c_t`。如果 bank 存的是 fused scene latent，那么“静态 LD/Nav 解析重投影”在信息上不成立——fused latent 已经把静态和动态内容纠缠在一起，无法只替换静态部分。正确关系是：
  ```
  M_k = FIFO(z_entity history)              # bank
  c_k = StateFusion(M_k)                    # fused context, 不落 bank
  a_k = Plan(c_k)
  z_entity_{k+1} = WM(M_k, a_k)
  M_{k+1} = FIFO(M_k, z_entity_{k+1})
  ```
  `E_target` 必须编码 `z_entity`，不是 `c_t`；否则 WM 的 target space 与 bank 不一致。加测试：fusion 每次 plan query 只调用一次；bank 永远不出现 `c_t`。
- 明确 temporal 的输入：`[z_i[t], Δz_i[t-1], ..., Δz_i[t-5], age, mask]`，默认不喂完整 6 帧绝对 latent；绝对历史只作为消融。这是 R4 的字面要求。
- **temporal 的输入再强调一次**：R4 要的是“当前帧 + 历史帧相对当前帧的 delta”，不是“6 帧绝对 latent 的时序 attention”。§4.2 的输入里同时放了 `z_hist` 和 `z_current - z_hist`，这会把 delta 信号稀释；默认只保留 current + delta，完整历史作为 ablation。

### 3.5 静态 LD/Nav：解析重投影还是 WM 预测（§5.2、§7.3、§9.2）

**文档写法**：
- §5.2：静态 LD/Nav 优先由 t0 真实结构和 imagined ego pose 做解析重投影/传播，不要求 WM 从零生成静态地图。
- §7.3：Static LD 可预测小 residual；Nav 不做普通动力学预测。
- §9.2：Static LD 和 Nav 不作为普通无条件预测主项。
- R4：WM 输出与单帧信息同构同质的 OD/LD/Nav/Ego latent。

**问题**：
1. “WM 输出 LD latent”与“LD 由解析重投影”不是同一件事。如果 LD latent 由 `Encoder(reprojected geometry)` 产生，那么它不是 WM 学出来的；这是一个 hybrid world model。文档没有明确声明，也没有测试“imagined LD latent 与真实 encoder 输出同分布/同语义”。
2. 如果 LD latent 由 WM 预测，静态地图会漂移，且 1M 参数要浪费大量容量在已知地图上；这与“不要大张量/不要预测静态地图”的工程直觉冲突。
3. Nav 同理：道路级 route 是已知图结构，ego pose 推进后 route progress 可解析更新；让 WM 学 Nav 动力学没有意义，文档也承认。但 R4 又要求 WM 输出 Nav latent；需要明确“输出 = 重编码，不是预测”。

**建议**：
- **明确采用 hybrid WM，并写进 contract**：
  - 动态 OD、road-state（若做）由 WM 学习预测；
  - 静态 LD、Nav 由 `StaticReprojector` 在 predicted ego frame 下重投影，再用**同一个 encoder**重编码；这不是 learned dynamics，但仍是 WM 输出的一部分。
  - Ego 由确定性 ego dynamics 推进 + encoder 重编码。
- 加测试：
  - `test_wm_static_reprojection.py`：给定同一 t0 静态地图和想象的 ego pose，`WM LD/Nav latent == Encoder(reprojected)`（数值容差内）。
  - `test_static_map_drift.py`：6 步 rollout 后，LD/Nav 的物理几何误差应只来自 ego pose 误差，不应有额外漂移。
- 如果用户坚持“LD/Nav 也必须由 WM 学习预测”，则需要证明它比解析重投影更好；在 1M 参数下我不建议。可以把 learned residual 作为可选消融，而不是主路径。

### 3.6 Ego-motion compensation 与 latent 纠缠（§4.1）

**文档写法**：历史 OD/LD 的 geometry anchor 做 SE(2) 补偿到当前 ego frame；content latent 不做手写旋转/平移；用 stable ID 对齐；输入 `z_current, z_hist, z_current-z_hist, g_current, g_hist→current, g_current-g_hist→current`。

**问题**：
1. 文档正确地区分了 content latent 和 geometry anchor，但没有保证 **content encoder 本身对 ego motion 不变**。如果 OD encoder 吃的是 `dx,dy,vx,vy`（相对 ego 的位置/速度），那么 content latent 已经混入了 ego motion；历史 content latent 的 delta 会同时包含“环境变化”和“ego 运动”。
2. 旧分支的证据：OD/LD latent 中 ego speed 泄漏 +19.1%，LD slot 的 R²≈0.9–0.99 vs real 0.08–0.14（E2，`AGENT.md`/`deepseekv4p1_argue.md`）。旧分支的 `zod/zld` 被指出编码了 ego 前进量，导致语义串扰。新方案如果不解决，WM 的 delta 主要在学 ego motion，而不是环境变化。
3. 文档的 §15.2 有 ego leakage probe，但只是诊断，没有训练时的约束。

**建议**：
- 把 encoder 输入拆成：
  - **content features**：类型、尺寸、turn signal/brake、车道类型、限速、拓扑语义等，理论上 ego-invariant；
  - **geometry features**：相对位置/速度/航向，明确允许随 ego frame 变换。
  content latent 不直接吃 raw `dx,dy`；如果吃，必须做 canonicalization 或数据增强。
- 训练时加 ego-frame 随机变换/equivariance loss：同一场景、不同 ego 起点/朝向，content latent 应接近，geometry anchor 应按 SE(2) 变换，plan 输出应等变。
- 加一个线性 probe：从 OD/LD content latent 预测 ego speed/yaw rate；如果 R² 高，说明泄漏，必须修。
- 对 LD：静态车道在补偿后，content latent 应几乎不变；如果随时间变化，说明 encoder 在编码 ego 运动，需检查输入。
- 这些应该进 P0/P1 的硬门，而不是等到 WM 训练失败再查。

### 3.7 Memory Bank（§5）

**文档写法**：
- Real bank 容量 6、帧间隔 0.5 s、保存 `t-2.5..t` 共 6 帧（含当前）；reset；只收真实 encoder 输出；写前 detach；固定 shape/mask。
- Imagine bank 从 real bank 拷贝；WM 追加；rollout 后 reset；source/rollout depth；禁止回写。
- §5.3 训练/DAgger/PPO buffer 优先保存原始结构化 observation，不把 latent 当长期唯一数据；encoder version 失效。
- §23 又把“6 帧是否含当前”列为开放项。

**问题**：
1. **6 帧含不含当前**：正文说含当前，§23 说开放。必须定死。否则 FIFO 语义、warmup、WM 的 t0 拷贝、delta 的 age 都会不一致。
2. **warmup**：episode 开始时没有 6 帧历史。文档没有定义。是重复第一帧、补零、还是等 2.5 s？不同选择会改变训练/评测分布。
3. **detach**：文档说 real bank 写入 detach，但又说“训练 batch 若需要 BPTT，用显式 sequence tensor”。这两条需要明确区分 runtime bank 和 training sequence；否则实现者可能把整个 6 帧历史 detach，导致 encoder 只从当前帧学习。
4. **latent vs raw**：R3 说 bank 只存 latent/必要显式信息；§5.3 说训练数据存 raw observation。这是合理的，但需要在文档中明确“runtime bank”和“offline dataset”是两个东西，不能混。
5. **encoder version**：文档说得好，但没有给出实现方式（hash/checkpoint id/invalidation）。需要具体字段。
6. **固定 shape**：文档说“必须固定 shape”，但没有给出每个模态的 shape/大小；这会影响参数/显存预算。

**建议**：
- 定死：**6 帧含当前帧**，索引 `t-5..t`，最老在前、当前在最后；`age=0..5`；warmup 用最早可用帧重复 + `valid=0`，delta 只在 `valid=1` 的帧上计算；或者显式等待 5 帧再规划（评测可用，训练用 mask）。旧分支 `env/obs/memory.py:1-12` 就是“6 帧含当前 + warmup 重复最早的真实帧 + `hist_valid=0`”，且明确不补零（E2）；这是可取的明确规则。
- Runtime bank 固定 shape：
  - `z_od: [6, N_od, H]`, `anchor_od: [6, N_od, A_od]`, `mask_od: [6, N_od]`, `id_od: [6, N_od]`, `presence_od: [6, N_od]`;
  - `z_ld: [6, N_ld, H]`, `anchor_ld: [6, N_ld, A_ld]`, `mask_ld`, `topo_id`;
  - `z_nav: [6, H]`, `route_progress: [6,1]`, `mask_nav`;
  - `z_ego: [6, H]`, `ego_state: [6, E]`, `mask_ego`;
  - `frame_index: [6]`, `source: [6]`, `valid: [6]`.
- Offline dataset：按 env step 存 raw/compact observation；用 `FrameLookup` 按 stride=5 组 6 帧；不要在每个 2 Hz row 里重复存 6 帧历史。旧分支的 `pipeline.frames` 是这种设计（E2）。
- Detach：runtime bank 写入 detach；training sequence 从 raw obs 一次性 encode 6 帧，保留图到当前 batch；episode 边界 detach；不要用 runtime bank 做 BPTT。
- Encoder version：每条 cache 记录 `encoder_hash/checkpoint_id`；加载时不一致就重新编码或失效；同一窗口禁止混版本。

### 3.8 Latent supervision（§9）

**文档写法**：
- `z_{t+1}^{GT}=E_target(o_{t+1})`，`L_wm=D(hat z, sg(z_GT))`；EMA target；分模态；latent variance/statistics monitor；辅助任务可选；planner task loss 持续约束。
- 多步 1→2→3→6；三种实验：GT state+GT action、pred state+GT action、pred state+Plan action。

**问题**：
1. 直接对 latent 做 MSE，在 1M 结构化模型上很容易坍缩（encoder 和 WM 一起把特征幅度缩小）。文档只写了“variance monitor”和“可选辅助任务”，没有具体的 anti-collapse loss。LAW 能用简单 MSE，是因为它有 rich image feature + 大 backbone；这里不是同一个 regime。
2. EMA target 的参数量没有计入 1M 预算；如果 EMA 是整份 encoder 的副本，total 可能超。
3. 静态 LD/Nav 的 latent 可能维度大、变化小，MSE 会被它们主导或忽略；需要 per-modality 权重/掩码。更具体地说，如果 LD 有 24–48 个 token、OD 只有 16–32 个，静态 LD 的 token 数会压过 OD；复制静态 latent 是最容易的“作弊解”。旧分支的 future encoder 输出 `.detach()`、loss 按 horizon mask（E2，`pipeline/stages.py:2124-2153`、`trainer.py:2377-2445`），说明 target construction 不是唯一问题，per-depth/mask 诊断同样重要。
4. 动作条件化只靠“输入 action”不够；WM 可能忽略 action。文档的 §15.3 只有诊断 probe，没有训练时的 action sensitivity loss。
5. LAW/EPONA 的引用不具体。LAW 的实际做法是：action-aware encoder 把当前 waypoint 拼到每个 spatial token，`wm_decoder` 预测下一帧 feature，loss 是 `MSE(prev_pred_feat, cur_feat.detach())`，one-step，没有 EMA（E1，`LAW/projects/mmdet3d_plugin/LAW/dense_heads/waypoint_query_decoder.py:239-246`、`LAW.py:225-231`）。EPONA 是 video diffusion + DCAE/DiT，参数量/计算量远大于 1M（E1，`Epona/README.md`）。文档应该引用具体机制，而不是泛泛说“借鉴”。
6. 多步训练没有定义 teacher forcing / free running / scheduled sampling 的比例，也没有定义 recurrent detach。

**建议**：
- `L_wm` 至少包含：
  - 归一化 latent MSE（LayerNorm 后）+ cosine；
  - VICReg/BYOL 风格的 variance/covariance 正则（或 InfoNCE）；
  - OD/ego 的物理 decoder 辅助（位置/速度/航向/presence；ego speed/yaw rate）；
  - 静态 LD/Nav 的权重降低，且用 mask；
  - 按 rollout depth 衰减权重；
  - action counterfactual/sensitivity loss：同一 state 下扰动 action，要求预测的 next latent 有可分辨差异，且与实际 next latent 的差异方向一致（没有 counterfactual GT 时，用 action-shuffle 负样本/对比损失）。
- 训练课程：
  - teacher forcing（GT state + GT action）→ pred state + GT action → pred state + plan action；
  - 1→2→3→6 步；
  - 每步报告 latent error、physical decode error、action sensitivity、effective rank。
- EMA：
  - 如果 encoder 冻结（WM 阶段），直接用 online encoder stop-grad，不需要 EMA 副本；
  - 如果 encoder 解冻，再用 EMA，且把 EMA 参数计入 total，或只 EMA 一个小的 target projector；
  - **不能同时设 `detach_encoder_from_wm_loss=true` 并宣称 EMA 防坍缩**：encoder 不接收 WM loss 时，EMA target 只跟踪一个为 BC/planning 优化的表征，不是为可预测性优化的表征；D1 冻结 encoder 时 EMA 更是 no-op。要么 encoder 解冻 + EMA + 小 LR adapter，要么冻结 + 固定 target + 明确的 variance/covariance 正则；
  - 单独监控 `||θ_target − θ_online||`（target drift），不要只按 loss 调 momentum。
- 监控：per-dim std、effective rank、same-ID vs different-ID cosine、static LD delta、action-shuffle loss gap。文档的 probes 可以保留，但要加 rank 和 action sensitivity 的训练信号。

### 3.9 Value/Risk 监督（§9.3）

**文档写法**：Value Head 预测 expected discounted return、route completion probability、time-to-goal/progress、optional cost value；Risk Head 预测 collision/off-road/wrong-way/stuck/comfort risk；reward value 与 safety cost value 分开。

**问题**：
1. 只定义了输出，没有定义**监督目标从哪来**。IL/DAgger 阶段没有 RL return；如果用 expert 数据做 Monte Carlo，需要 reward model 和 episode 终止；如果用 auxiliary labels，需要明确标签。
2. Risk Head 的标签窗口/阈值没有定义（未来 3 s？整个 episode？）。
3. Value/Risk 在 imagined state 上训练时，是否 stop-grad？如果让 value loss 反传 WM，可能让 WM 迎合 value，而不是物理预测。
4. “Reward model”是 handcrafted 还是 learned？文档的 §12.2 是 handcrafted terms；如果用户期望 learned reward model，需要另说。

**建议**：
- IL/DAgger：
  - Value target = 从当前步到 episode 结束的折扣回报（用 §12.2 的 reward/cost），或 route completion/time-to-goal 的辅助回归；mask 掉终止后。
  - Risk target = 未来 3 s 内 collision/off-road/unsafe proximity 的 binary；用 BCE/focal；按场景/速度分桶校准。
- RL：GAE + value loss；保留 BC/DAgger 的 value auxiliary 防遗忘。
- Imagined：value/risk 读取 imagined latent 时，默认 stop-grad WM（或只允许小 adapter），避免 WM 被 value loss 塑形；如果要做 model-based value gradient，单独 A/B。
- 术语：把 §12.2 明确称为“handcrafted, auditable reward function”，不要叫 learned reward model；如果要 learned reward，先证明 handcrafted 不够，并加 reward audit。

### 3.10 RL 设计（§12）

**文档写法**：PPO，Plan Head action semantics 不变；真实 ENV 先于 imagined RL；reward/cost 项；防 reward hacking；curriculum。

**问题**：
1. 没有明确 PPO 的决策 epoch 和 tracker（见 3.1/3.3）。如果 PPO 在 10 Hz 下跑 0.5 s 动作，或者用 Exact tracker 训练、LQR 部署，都会出问题。
2. Reward 的量级是按“每步”定义的，但步长没有定义；0.1 s 和 0.5 s 的 progress 差 5 倍。
3. 没有定义 6 步轨迹中哪些动作是随机变量；如果 6 个都采样，动作空间是 12 维（或 18 维），credit assignment 很难；如果只采样 `a0`，要明确 tail 是均值还是冻结策略。
4. 没有定义 cost 如何进入优化：reward penalty、Lagrangian、还是只在评测时监控。
5. 没有定义 RL 的初始化/KL anchor/BC regularization 的具体方式；文档只说“保留 BC/DAgger auxiliary loss 或 replay regularization”。

**建议**：
- PPO 决策 2 Hz；只对 `a0` 采样；`a1..a5` 用同一 policy 的均值 + WM 生成；log-prob 只对应 `a0`。
- Reward 在 0.5 s 窗口上聚合（5 个 0.1 s 控制步）；progress 用 route arc-length；碰撞/出界任一步终止。
- Tracker：正式 PPO 用 LQR；如果先用 Exact 做 canary，必须再跑 LQR canary，且不能把 Exact 的结果当最终。
- Cost：先做 reward penalty + 硬安全门；如果 collision/off-road 仍高，再考虑 Lagrangian/constrained PPO。
- 初始化：从 DAgger/IL checkpoint 初始化；KL anchor 到该 checkpoint；BC/DAgger auxiliary 或 replay regularization；value warmup；primary LR 缩小（旧分支 evidence: primary lr×0.1, KL coef 0.05→0）。
- 课程：低密度→目标密度；按场景桶采样；DAgger 失败优先。

### 3.11 Exact 与 LQR（§7、§13F）

**文档写法**：Exact Tracking 只替代早期执行器；LQR 使用完整 3 s 预瞄；Stage F 才评估 controller-realized closed loop；要区分 planner-oracle 与 controller-realized。

**问题**：
1. 没有定义 Exact Tracking 的实现。旧分支的 ExactTracker 是 `env.step` 之后把 ego **瞬移**到插值位姿（E2，`env/tracking.py`）。这不是物理闭环；碰撞/出界检查仍会发生，但车辆动力学/控制器被绕过。它只能衡量 planner 的几何可行性上界，不能代表部署性能。
2. LQR 被推迟到 Stage F；如果 Stage C 的 PPO 在 Exact 下训练，Stage F 的性能下降不能归因于 planner，可能是 controller。文档说“如果 exact 好而 LQR 差，先查轨迹可行性和控制器”，但缺少具体的隔离实验。
3. 没有 LQR 的“GT 轨迹跟踪” sanity check。如果 LQR 连专家轨迹都跟不好，就不应该用它评判 planner。
4. 没有定义 LQR 的状态、Q/R、preview、速度控制、转向/油门限幅；这些在 §23 被列为开放项。开放项太多，无法保证可复现。

**建议**：
- 明确 Exact = kinematic playback：只用于数据采集、plan head 上界、快速迭代；所有最终指标必须 LQR。
- LQR：
  - 先做 `test_lqr_gt_tracking.py`：用专家 3 s 轨迹作为参考，LQR 跟踪，报告横向/航向/速度误差；如果误差大，先调控制器。
  - 再做 `test_lqr_plan_tracking.py`：Plan Head 轨迹；报告 exact vs LQR 的分解（planner 几何误差、controller 跟踪误差、执行器饱和）。
  - 固定 Q/R/preview/速度控制参数到一个 config；不要每次实验改。
- 插值：如果主方案是 `(ds,dtheta)`，用常曲率圆弧；LQR 参考 30 点 10 Hz；不要用会过冲的 Hermite。
- 在评测中同时报告 `steer/throttle` 饱和率、横向误差、航向误差、jerk；不要只看 success。

### 3.12 参数预算（§8）

**文档写法**：
- 总参数约 1M，允许 0.8–1.2M；预算表：OD 60–100k、LD 100–150k、Nav 20–40k、Ego 10–30k、Temporal 120–200k、Spatial/Cross 150–220k、Plan 80–120k、Value/Risk 30–70k、WM 180–260k。
- H=96，4 heads，1–2 layers，OD 16–32，LD 24–48。

**问题**：
1. 预算表相加 750k–1.19M；没有算 LayerNorm、embedding、位置编码、输出头、EMA target、优化器状态。实际 total 可能超 1.2M。
2. 没有区分 **total** / **active/trainable** / **frozen** / **dormant**。旧分支的 1M active 是 995k，但 total 是 1.28–1.51M（E2）。用户说“net总参数 100W”，如果指 total，当前预算需要压缩；如果指 active，H=128 也可以。
3. EMA target 如果是一整份 encoder 副本，参数量可能 +200–300k；文档没有把它计入预算。
4. `nn.TransformerEncoderLayer(d_model=96, nhead=4, dim_feedforward=384)` 单层约 111k 参数；如果 Temporal 和 Spatial 各 2 层，就已经 440k，超出预算表的 Temporal+Spatial 上限 420k，还没算 encoder/plan/WM。所以“1–2 层”必须与预算表联动，不能自由选。
5. 没有 tensor budget：6 帧 × OD/LD token × H、attention 矩阵、6 步 rollout 的 activation；文档只说“不要大张量”，没有上限/测试。

**建议**：
- 五栏账本（total/reachable/trainable/frozen/dormant），每个参数恰好归一类；EMA copy 明确计入 total 或不计入但单独报告。
- 明确 100W 的定义：
  - 若 **total ≤ 1.0M**：H=96、无 MoE、Temporal 1 层、Spatial 1–2 层、WM 1 层、LD encoder 1 层、EMA 只做小 projector 或不做；预算表要重算。
  - 若 **active/trainable ≤ 1.0M**：H=128 可行（旧证据），total 可到 1.3–1.5M；需要用户接受 total 超 1M。
- 给出 per-module 参数预算与实例化测试；任何新模块必须进账本，未知前缀 fail-fast。
- Tensor budget：定义最大 token 数、最大 attention 矩阵、最大 activation；profile batch 1/8/64，6 步 rollout；超过阈值就改结构。
- 初始化：Kaiming（ReLU/GELU MLP）、Xavier（attention projection）、LayerNorm 1/0、残差输出小/零；加初始化统计 probe。
- **既然已裁决 total ≤ 1M，建议把预算表改成 upper-bound envelope（E3 估算，实例化后校准）**：
  - OD encoder 60k、LD encoder 100k、Nav 20k、Ego 15k、Temporal fusion 100k、Spatial fusion 120k、Plan Head 80k、Value/Risk 45k、WM 220k、embedding/LayerNorm/位置编码 30k，合计 **790k**；留 **210k** 给必要的第二层/输出头/消融。
  - **不做整份 EMA encoder 副本**：旧分支的 EMA/target 若复制 OD/LD/Nav/Ego encoder，会加 0.19–0.32M；复制 temporal/spatial 会到 1.21–1.61M，直接破坏 total≤1M。若必须 EMA，只做 20–30k 的小 target projector，并计入 790k 以内。
  - 这只是上限，不是实测；P0 必须用 `param_probe` 逐模块校准，并把“未分类前缀”fail-fast。

### 3.13 数据与评测（§14）

**文档写法**：
- 训练 10,000 场景，验证 1,000，验证集首次生成后永久冻结；建议额外 test；按 seed 划分；禁止帧随机切分。
- 指标：success/collision/off-road/wrong-route/stuck/timeout/comfort；控制指标；WM 指标；场景分桶；checkpoint 选择按安全门槛→success→completion→comfort。

**问题**：
1. **验证集同时用于 checkpoint selection 和最终结论**。如果每次实验都在 1000 验证集上选 checkpoint，它就不再是“冻结”的；这是常见的 validation overfitting。文档没有 Calib/Val/Test 分层。
2. **数据规模/存储/采集成本**没有预算。旧分支 `datasets/path_exec_v1`（由 5k expert 数据派生）有 360,497 rows（E2，`repo_handbook`）；10k 场景按同量级约 720k 2 Hz rows（E3 估算）；如果存 10 Hz 帧，约 3.6M 帧；DAgger 多轮再乘。文档没有说存哪一层、压缩/分片/memmap。
3. **评测成本**：1000 场景 × 60 s × 10 Hz = 600k env steps；如果每个 checkpoint 都跑全量，成本很高。文档没有 Calib50/Clean500/Test500 的节流。
4. **场景分桶与生成器不一致**：文档 §14.3 的分桶没有 roundabout、uturn、split、t_intersection、ramp_in/out；旧生成器有这些标签（E2）。需要对齐。
5. **验证集冻结**不等于“不能用于 early stopping”；需要明确使用协议，否则实现者会反复调参。

**建议**：
- 三层评测：
  - `Calib50`（从 train 分层抽，快速迭代；**不得作收益声明**）；
  - `Val500`（从 1000 val 中分层抽，用于 checkpoint selection；每个 stage 只允许有限次）；
  - `Test500`（剩余 500，或独立 seed 区间；最终只跑一次/少量 seed；**不得用于调参**）。
  如果用户坚持 1000 val 全冻结，则把 1000 当 Test，另建 200 Calib；Val 只用于预注册的 selection。
- 数据：
  - **分两档规模**：最终目标 10k/1k；P0/P1 先用 **2k/200** 子集验证 schema、专家采集、round-trip、训练/评测装配；不要一开始就全量 10k/1k，否则在 15 GB RAM 上迭代会被采集/评测卡死。
  - 按 env step 存 compact frame；2 Hz 训练样本用 `FrameLookup(stride=5)` 组 6 帧；不要在每个 row 重复 6 帧历史。
  - 分片 `npz`/memmap/parquet；float16；记录 schema/fingerprint/encoder version。
  - 预算：先跑 P0 smoke 测 FPS/RSS，再定 worker 数；当前 15 GB RAM 建议 **4 个 env worker**（不是 16/20），recycle 100–150，`mem_available_floor_mb=3000`；旧分支 6 个 eval worker≈9.5 GB 的记录（E2）说明 20 CPU 不等于 20 env。
  - 评测：Calib50 每次迭代；Clean500 过门才跑；Test500/1000 只在阶段里程碑跑；全量 1k × 60 s × 10 Hz = 600k env steps，不能每个 checkpoint 都跑。
- 场景生成：复用旧分支的分层几何分配思路（不是代码）：覆盖 11 类几何；排除 bidirection（旧分支实测 S→B 接缝不可通行）；roundabout 密度上限；tollgate；官方 map 作为单独 bucket。
- 评测协议：预注册指标、阈值、checkpoint 选择规则、seed 组；报告 Wilson CI / McNemar；不要用单次 episode。

### 3.14 Expert、场景与 OD/LD taxonomy（§3、§11）

**文档写法**：
- OD：车辆、行人、锥桶、障碍物、收费站动态设施；LD：车道、匝道、分岔、路口、收费站、限速；Nav：道路级。
- Expert：IDM + Pure Pursuit，唯一可用；审计 target lane/lane-level route/未来交通/参考中心线；privileged 样本降权。

**问题**：
1. **toll booth 同时出现在 OD 和 LD**。MetaDrive 的 toll booth 是 `TollGateBuilding`（`BaseBuilding`，静态），不是车辆；把它放 OD 会让 OD encoder 同时处理动态车辆和静态建筑，WM 也会被要求“预测”静态建筑。需要 taxonomy。
2. **锥桶是 `TrafficCone`（静态对象）**，不是动态车辆；如果放 OD，需要 type embedding 和 static/dynamic mask；WM 不应预测其运动。
3. **交通灯**：文档提到 dynamic road-state token，但旧分支关闭了交通灯（E2）。如果新方案不做交通灯，不要保留假 signal 通道；如果做，定义相位时钟/是否由 WM 预测。
4. **Expert 特权**：旧 `IDMPolicy` 使用 `navigation.current_ref_lanes`、`routing_target_lane`、`lane_change_policy`（E1）；旧 `PurePursuitIDMPolicy` 使用 lane centerline、lane chain、static building scan（E2）。它们都有 lane-level 特权。文档的审计方向正确，但需要具体的“允许/禁止”清单和测试。
5. **Expert 轨迹转学生动作**：文档说“统一转换为 Student 的轨迹表示”，但没有定义。需要 10 Hz 执行轨迹 → 0.5 s 窗口 `(ds,dtheta)` → 常曲率重建 → round-trip 误差门；失败窗口剔除。

**建议 taxonomy**：

| 通道 | 内容 | 是否动态 | WM 责任 |
|---|---|---|---|
| OD-dynamic | 车辆、行人、骑行者 | 是 | 学习预测位置/速度/heading/presence |
| OD-static | 锥桶、护栏、施工障碍 | 否 | 重投影/重编码；不学习动力学 |
| LD | 车道中心线/边界、拓扑、匝道、分岔、路口、限速、收费站几何 | 否 | 重投影/重编码 |
| Nav | 道路级 route、road graph、route progress、下一道路方向 | 否（随 ego pose 推进） | 解析更新/重编码 |
| RoadState | 交通灯、收费闸口状态（若做） | 是/半静态 | 若做，显式状态/相位；不要让 WM 从零学 |

- Expert 审计清单：target lane、lane-level route、推荐变道、未来交通、参考中心线、收费站闸口选择；每一项给出“允许/禁止/仅训练标签”的结论。
- Expert 数据过滤：无碰撞/出界、on-lane 容差、3 s 窗口干净、event fired/alive、round-trip 误差达标；失败 episode 不作为正样本。
- 加 `test_expert_privilege.py` 和 `test_expert_roundtrip.py`。

### 3.15 目录结构（§17）

**文档写法**：`dataset/ net/ config/ pipeline/ tools/ test/ docs/`；`net/controller/lqr.py`；`dataset/metadrive_adapter.py`；`pipeline/rollout/`。

**问题**：
1. 缺少 `env/`（或 `sim/`）目录：MetaDrive wrapper、observation builder、scenario generator、expert、tracking 是独立于 dataset/net/pipeline 的一层。把它们塞进 `dataset/` 或 `net/controller/` 会让职责混乱。
2. `net/controller/lqr.py` 不是网络模块；LQR 是执行/控制层，应放 `env/tracking.py` 或 `pipeline/control/`。
3. `reward_model/` 没有目录；如果 reward 是 handcrafted，可以放 `pipeline/reward.py` 或 `env/reward.py`；如果独立，放 `reward_model/`。
4. 用户要求的 `test/` 是 singular；文档用了 `test/`，可以；但要和旧分支 `tests/` 区分。

**建议**：
```
env/            # MetaDrive wrapper, obs, scenario, expert, tracking
dataset/        # schema, adapter, collector, split, sequence builder, buffers
net/            # encoders, fusion, planner, value, world_model, memory, model
config/
pipeline/       # stages, trainers, rollouts, losses, metrics, probes
tools/
test/
docs/
```
- 明确 ownership：`env` 产生 observation/transition；`dataset` 负责存储/采样；`net` 只做张量计算；`pipeline` 编排训练/评测。
- 这样也方便 P0 的 environment smoke 和 P1 的数据契约测试。

### 3.16 测试与 probes（§15、§18.7）

**文档已有**：schema、坐标、ID、LD segment、reset、FIFO、real/imagined isolation、nav no lane leak、trajectory assembly/interpolation/continuity、encoder cache、ego condition、WM action sensitivity、rollout consistency、detach、checkpoint reproducibility。方向很好。

**缺失的关键测试**：
1. `test_timing_contract.py`：2 Hz policy / 10 Hz control / 6 步 = 3 s / 30 个 0.1 s 参考点；PPO 只采样 `a0`。
2. `test_action_kinematics.py`：`(ds,dtheta)` → 常曲率圆弧 → 6 步累计位姿；与解析积分一致；无过冲/负速度。
3. `test_plan_head_single_instance.py`：同一 Plan Head 被调用 6 次；没有第二个头；real/imagined 状态共享权重。
4. `test_oracle_state_rollout.py`：真实未来 latent 的 6 步监督；mask/权重正确。
5. `test_interface_loss.py`：WM 预测状态 vs 真实状态的动作一致性；`E_interface` 随 depth 的曲线。
6. `test_wm_static_reprojection.py`：静态 LD/Nav 输出 == 同一 encoder 的重投影输出。
7. `test_memory_warmup.py`：episode 前 5 帧的 mask/delta；不把补零当真实运动。
8. `test_memory_6_including_current.py`：索引语义固定。
9. `test_latent_collapse.py`：effective rank、per-dim std、same-ID/different-ID cosine。
10. `test_action_conditioning.py`：同一 state 不同 action → OD/LD 预测显著不同；action shuffle → loss 变差。
11. `test_param_budget.py`：五栏账本；EMA copy 计入；未知前缀 fail-fast。
12. `test_tensor_budget.py`：最大 token/attention/activation；6 步 rollout 峰值显存。
13. `test_expert_roundtrip.py`：10 Hz 专家轨迹 → `(ds,dtheta)` → 重建误差。
14. `test_lqr_gt_tracking.py`：LQR 跟踪专家 3 s 轨迹的误差；控制器 sanity。
15. `test_validation_isolation.py`：train/val/test 的 `(id,seed)` 无交集；DAgger 不得用 eval/val。
16. `test_train_eval_assembly_parity.py`：同一 checkpoint、同一帧，训练装配与评测装配的 obs/feature/action 逐位一致；这是旧分支的惨痛教训（E2）。
17. `test_domain_distribution.py`：训练 vs 评测的速度/横向偏差分布；DAgger 后的覆盖。
18. `test_reward_audit.py`：reward/cost 项的量级、路线进度、安全门槛；防 reward hacking。

### 3.17 文档内部矛盾清单

| 矛盾 | 位置 | 说明 |
|---|---|---|
| 0.5 s 规划 vs 10 Hz 重规划 vs 6 步硬约束 | §6.1 vs §22.2 | Stage A-C 不跑 6 步，硬约束说必须 6 步 |
| 6 帧含当前 vs 开放项 | §5.1 vs §23 | 正文说 `t-2.5..t` 含当前，§23 又说待定 |
| 完整历史 6 帧 vs 当前+delta | §4.2 vs R4 | §4.2 输入包含 `z_hist` 绝对值和 delta；R4 明确不要对完整 6 帧做时序 |
| 静态 LD 重投影 vs 预测小 residual | §5.2 vs §7.3 | 一个说解析重投影，一个说可预测 residual |
| WM recurrent path 不 detach vs Plan action detach/frozen | §10.1 vs §13E | 需要分 stage/训练/推理定义 |
| 1M 预算 vs 预算表上限 | §8.1 vs §8.2 | 相加 1.19M，未算 EMA/norm/embedding |
| RL 无 WM vs Stage D 在 C 之后 | §13C vs §13D | 文档顺序是 RL 先、WM 后；R9 相反 |
| 验证集冻结 vs 用于 checkpoint selection | §14.1 vs §14.4 | 需要 Calib/Val/Test 分层 |
| 缺 `env/` 目录 | §17 vs 需求 | MetaDrive wrapper/obs/scenario/expert/tracking 无归属 |
| 依赖版本只写“固定” | §18.1 | 没有给出 MetaDrive/Python/torch/numpy 的具体 pin |

---

## 4. 更好的方案提议

### 4.1 单一接口契约（建议写进文档的 §0）

```
policy_dt        = 0.5 s          # 2 Hz decision
control_dt       = 0.1 s          # 10 Hz execution
history_frames   = 6              # includes current: t-2.5 ... t
rollout_steps    = 6              # 6 × 0.5 s = 3 s future
action           = (ds, dtheta)   # arc length + heading change per 0.5 s
trajectory       = 30 points @ 10 Hz (0.1 ... 3.0 s), plus current state at t=0
replan           = every 0.5 s; execute first 5 control steps; rest is preview
PPO action       = a0 only; a1..a5 = deterministic tail from same policy + WM
memory reset     = episode boundary; imagine bank reset after each rollout
```

- 这个契约同时满足 R3（0.5/0.1）、R10（6 步→3 s→LQR）、R13/R14（同一头）、R16（先 plan head 后 WM，若按 4.2 的折中）。
- 如果用户坚持 `(dx,dy,v)`，把它作为 `(ds,dtheta)` 的上层参数化，加运动学一致性层；不要让两个参数化并存。

### 4.2 推荐分阶段路线（与文档的差异）

**文档路线**：A1 单帧 → A2 memory → B DAgger → C RL（Exact）→ D1/D2 WM → E joint → F LQR → G imagined RL。

**建议路线**：

| 阶段 | 训练 | 执行/评测 | 目的 | 继承 |
|---|---|---|---|---|
| P0 | 环境/数据/schema/expert/exact/LQR smoke | — | 证明 MetaDrive 可跑、数据契约、评测链 | — |
| P1 | Plan Head BC：B1 oracle-state 6 步 + policy-only rollout | Exact + LQR 诊断 | 证明 6 步接口、车道保持/跟车/避障 | Plan Head 初始化 |
| P2 | DAgger（student states, expert labels） | Exact + LQR canary | 恢复能力、分布覆盖 | 同 Plan Head |
| P2.5 | Exact PPO canary（10–20 updates，可选） | Exact | 只证明 PPO/GAE/日志/恢复管线；**不作最终结论** | 同 Plan Head |
| P3 | WM 1→2→3→6 步（encoder/plan 冻结） | Imagine bank + LQR canary | 证明 WM fidelity、静态重投影、动作条件 | 同 Plan Head/Encoder |
| P4 | Joint Plan+WM（encoder frozen/adapters） | LQR | 让 Plan Head 适应 WM 状态；interface loss | 同 Plan Head |
| P5 | 正式 ENV PPO（LQR；BC/KL 正则） | LQR | 超越 IDM/PP 的闭环 | 同 Plan Head |
| P6 | Imagined RL / WM distillation（可选） | WM + LQR | 样本效率/长尾 | 同 Plan Head |

- 与文档的最大差异：**WM/LQR 在正式 RL 之前**，而不是 RL 之后；正式 PPO 用 LQR，而不是 Exact。
- P2.5 是对 R16 的让步：保留一个很短的 Exact PPO canary，证明 RL 管线能跑；但它的 checkpoint 不能作为最终模型，也不能用来宣称“超过 IDM”。正式 P5 必须在 LQR 下训练/评测。
- 与 R9/R16 的关系：R9 字面是 WM → policy → DAgger → RL；R16 字面是 plan head IL+DAgger+RL → WM → joint。推荐路线是 **plan-head-first + WM before full RL**：保留 R16 的“plan head 先跑通”，同时让 WM 在正式 policy 强化之前介入；它不逐字满足任何一条，但避免了两条路线各自的已知断裂点。若你要求逐字 R9 或 R16，需要单独接受对应的风险（R9：TF/部署 gap；R16：Exact→LQR 断裂）。

### 4.3 推荐 WM：canonical state + latent residual

把 WM 拆成“确定性物理 + 学习残差”，而不是让一个黑盒预测所有 latent：

```
EgoDynamics:   (pose_t, v_t, a_t) -> pose_{t+1}, v_{t+1}        # no/ tiny learned residual
StaticReproject: (map_t0, pose_{t+1}) -> LD/Nav geometry         # deterministic
DynamicPrior:  (OD_t, a_t) -> constant-velocity/accel prior      # physical prior
DynamicWM:     (state_t, a_t, prior) -> ΔOD canonical + presence # learned residual
Encoder:       (OD canonical, geometry) -> z_od_{t+1}            # same encoder
```

- 好处：
  - 静态 LD/Nav 不漂移；
  - OD 的物理先验/残差可解释，WM 不需要从零学 ego motion；
  - 输出与 encoder latent 同构（通过同一 encoder 重编码）；
  - 动作条件化可以直接加在 DynamicWM 的每一层，而不是只加在 ego latent 上。
- 代价：多一个物理 decoder/重编码步骤；需要保证可微/性能。可以用 stop-grad 的物理先验 + 学习残差。
- 这与旧分支的 `st_gnn`（先验+零初始化残差）方向一致，但不要照抄它的 5/4 维物理头和 action-only-ego 设计（E2）。

### 4.4 推荐 memory/latent schema

- Runtime bank：6 帧含当前；per-modality fixed shape；只存 latent + anchor + mask/id/presence；不存 raw obs。
- Offline dataset：按 env step 存 compact frame；2 Hz 样本用 lookup；记录 schema fingerprint/encoder version。
- Temporal input：`[z_t, Δz_{t-1}, ..., Δz_{t-5}, age, mask]`；默认不喂完整绝对历史；绝对历史作为消融。
- Static LD/Nav：单独 static path，mask 掉 delta；不要强迫静态实体参与动态时序。
- Warmup：repeat earliest + valid=0；或等待 5 帧；在 schema 中定死。
- Detach：runtime bank detach；training sequence 保留图到当前 batch；episode 边界 detach；WM rollout 的 recurrent detach 按 stage 配置。

### 4.5 推荐 loss/probe 组合

- Plan：`L_ds + L_dtheta + L_endpoint + L_heading + L_curvature + L_smooth + L_feasibility`；不要只 XY MSE。
- WM：normalized MSE + cosine + VICReg/BYOL variance/covariance + physical decoder + action counterfactual + per-depth weights + mask。
- Interface：`L_if = Σ w_k ||π(ŝ_k) - sg(π(s_k_real))||`。
- Value/Risk：MC/GAE + BCE/focal；risk 标签窗口 3 s；校准。
- Probes：effective rank、per-dim std、same-ID/different-ID cosine、static LD delta、action-shuffle gap、E_interface per depth、exact/LQR gap、domain distribution、train/eval assembly parity。
- 每个新模块必须有“关闭该模块的同 seed ablation”；文档 §19 已有，但需要加“delta-only vs full history”“static reproject vs learned LD”“action conditioning on/off”“oracle vs predicted state”。

### 4.6 推荐参数/张量账本

- 五栏：total/reachable/trainable/frozen/dormant；未知前缀 fail-fast。
- **已裁决 total≤1M**：按 H=96、1-layer、无 MoE、无整份 EMA copy 设计；active 只会更低。不要用 active≤1M 的口径为 H=128/EMA 副本留后门。
- 加 `test_param_budget.py` 和 `test_tensor_budget.py`；profile 1/8/64 batch、6 步 rollout；记录单步/6 步延迟。
- 初始化：Kaiming/Xavier/LayerNorm/零残差；加统计 probe。

### 4.7 推荐数据/评测协议

- 10k train；Calib50（train 子集，快速迭代）；Val500（checkpoint selection）；Test500（最终，不再调参）。如果用户坚持 1000 val，则 1000 全部作 Test，另建 Calib200；或者 500/500 但预注册。
- 场景覆盖：straight/curve/ramp_in/out/merge/split/intersection/t_intersection/roundabout/uturn/tollgate；排除 bidirection；roundabout 密度上限；official map 单独 bucket。
- 采集：**4 个 env worker**（当前 15 GB RAM；不要用 16/20）；recycle 100–150；`mem_available_floor_mb=3000`；P0 先测 FPS/RSS；数据分片/float16/memmap。
- 评测：每个 checkpoint 先 Calib50，过门再 Clean500，最终 Test500；多 seed；Wilson CI/McNemar；报告失败 seed。
- 训练/评测装配：同一 obs builder/net config/feature order；加逐位 parity 测试；旧分支的 50/50 out_of_road 是前车之鉴（E2）。

### 4.8 推荐 expert/taxonomy

- OD-dynamic / OD-static / LD / Nav / RoadState 五类；toll booth/cone 归 static；traffic light 若不做就不留通道。
- Expert 审计清单 + round-trip + 过滤；privileged 样本降权/分桶；禁止 lane-level 信息进入 Student。
- 专家轨迹转 `(ds,dtheta)` 的窗口重建误差门；失败窗口剔除。

### 4.9 开写代码前的最小契约修改（按优先级）

1. **冻结一张时间表**：env 0.1 s、decision 0.5 s、tracker 0.1 s、memory 0.5 s、WM 0.5 s、PPO 0.5 s；并加 CI 断言“executed action == policy action 的第一段”。
2. **冻结动作**：`(ds,dθ)` + 确定性 `arc_step` 得到 `(dx,dy,heading)`；速度只作为 `v_end=ds/0.5` 或带一致性损失的辅助；不要再用“从点差分 heading”作为主路径。
3. **补 oracle-state 6 步 Plan Head 训练 + interface loss**：在 WM 之前就让同一个头见过 6 步真实未来状态；D2 的 predicted-state 训练要成为正式 regime，而不是诊断实验。
4. **把 bank 的 `z_entity` 与 fused `c_t` 分开**：bank 只存 per-entity；fusion 每次 plan query/imagined step 重算；`E_target` 编码 `z_entity`。
5. **补 anti-collapse 与 action conditioning**：variance/covariance 正则、per-modality mask/denominator、静态 LD/Nav 不进主 latent loss、action-shuffle/反事实损失；EMA 只在 encoder 解冻时使用，不能与 `detach_encoder_from_wm_loss=true` 同时宣称防坍缩。
6. **补 LQR 并行诊断与 GT 轨迹 sanity**：Exact 只作上界；LQR 从 P1 起就作为并行诊断；正式 PPO 用 LQR；加 `test_lqr_gt_tracking` 与 exact/LQR 分解。
7. **补参数/数据/评测门**：五栏账本、tensor budget、2k/200 起步、4 worker、Calib/Val/Test、train/eval assembly parity。

如果这 7 条不先写死，直接实现 1751 行文档的下一版，最可能的结果是“每个模块都能单独跑，但 Stage 之间接口不闭合”。

---

## 5. 风险登记表

| ID | 风险 | 证据 | 影响 | 缓解 | 门 |
|---|---|---|---|---|---|
| K1 | 时间尺度/动作 horizon 不一致 | 文档 §6.1 vs §22.2；MetaDrive 0.1 s；旧分支 2 Hz/10 Hz | PPO/GAE/LQR 全部错位 | 固定 2 Hz/10 Hz/6 步；PPO 只采样 a0 | `test_timing_contract` |
| K2 | 动作参数化不可行/过冲 | 文档 `(dx,dy,v)`；旧分支 `(ds,dtheta)` | 轨迹不可跟踪、LQR 崩 | `(ds,dtheta)` + 常曲率；或加运动学一致性层 | `test_action_kinematics` |
| K3 | Plan Head 换任务 | 文档 Stage A-C vs D-F | 继承失败、exposure bias | oracle-state 6 步 + interface loss | `test_interface_loss` |
| K4 | WM 数据流/静态 LD 不明 | 文档 §4.4/§5.2/§7.3 | WM 不是 R4 描述的东西 | 共享 StateFusion + hybrid static path | `test_wm_static_reprojection` |
| K5 | latent collapse | 1M 模型 + MSE；LAW 依赖 rich feature | WM 学不到动力学 | VICReg/BYOL + physical decoder + rank probes | `test_latent_collapse` |
| K6 | action insensitive | 旧分支 action-only-ego canary | WM 忽略 ego action | 每层 action conditioning + shuffle/contrastive loss | `test_action_conditioning` |
| K7 | LQR 崩/域差 | 旧分支 LQR repeat 0.06；训练/评测装配不一致 | 闭环失败无法归因 | LQR GT sanity；Exact/LQR 分解；assembly parity | `test_lqr_gt_tracking` |
| K8 | 数据/内存不足 | 当前 15 GB RAM/8 GB avail；旧 16 workers×0.65 GB | 采集/评测 OOM | 4 workers；recycle；pause；memmap | P0 smoke |
| K9 | 参数超预算 | 文档预算表未算 EMA/norm；旧 total 1.28–1.51M | 1M 目标失败 | total≤1M：H=96/no MoE/no full EMA；五栏账本 | `test_param_budget` |
| K10 | 验证集过拟合 | 1000 val 用于 selection+结论 | 结论不可信 | Calib/Val/Test 分层；预注册 | `test_validation_isolation` |
| K11 | Expert 特权泄漏 | IDM/Pure Pursuit lane-level | Student 学到 oracle | 审计/禁用/降权/分桶 | `test_expert_privilege` |
| K12 | toll/cone/交通灯 taxonomy 错 | MetaDrive `TollGateBuilding`/`TrafficCone` | OD/WM 语义混乱 | static/dynamic 分层 | schema test |
| K13 | 静态地图漂移 | WM 预测 LD/Nav | 闭环出界 | 解析重投影+重编码 | `test_static_map_drift` |
| K14 | reward hacking | 旧分支 reward audit 教训 | 撞车/切角刷分 | 安全 cost 分离、route arc-length、audit | `test_reward_audit` |
| K15 | bank 存 fused `c_t` 导致静态 LD/Nav 无法重投影 | 文档 §2.2/§4.4/§5.2 | WM 输出空间与 target 不一致 | bank 只存 `z_entity`；fusion 现算 | `test_bank_entity_schema` |
| K16 | EMA 与 `detach_encoder_from_wm_loss` 逻辑冲突 | 文档 §9.4/§10.1 | 防坍缩无效、target 漂移 | 分阶段：冻结→固定 target；解冻→EMA+adapter | `test_ema_phase` |

---

## 6. 已裁决 / 待审阅

| 问题 | 状态 | 结论/建议 |
|---|---|---|
| 100W 参数 | **已裁决** | **total ≤ 1M**；H=96、无 MoE、默认 1 层、无整份 EMA copy；五栏账本 + tensor budget。 |
| R9 vs R16 | **已解释，推荐待审** | R9 = “WM→policy→DAgger→RL”；R16 = “plan head IL+DAgger+RL→WM”。推荐 §4.2 的折中：plan head BC → WM → joint/LQR → DAgger → 正式 RL；P2.5 保留一个很短的 Exact PPO canary。 |
| 动作参数化 | **推荐待审** | `(ds,dtheta)` 主方案；`(dx,dy,v)` 若保留，必须加运动学一致性层。 |
| 规划 epoch | **推荐待审** | 2 Hz 决策 + 10 Hz 控制 + 6 步 3 s；10 Hz 重规划只作消融。 |
| 静态 LD/Nav | **推荐待审** | hybrid：解析重投影 + 同一 encoder 重编码；不要求纯 learned latent。 |
| LQR 何时介入 | **推荐待审** | P1 起 LQR 并行诊断；正式 PPO 用 LQR；Exact 只作上界/采集。 |
| 验证集协议 | **推荐待审** | 1000 全作 Test + Calib200，或 500 Val/500 Test；预注册 selection。 |
| 交通灯/收费站/锥桶 taxonomy | **推荐待审** | OD-dynamic / OD-static / LD / Nav / RoadState 五类；交通灯若不做就不留通道。 |

这些是给 `chatgpt5p6sol` 审阅的“推荐基线”，不是已经锁死的最终决定；但每一条都建议在写代码前明确。

---

## 7. 证据附录

### 7.1 当前仓库/环境

```bash
git status --short --branch
# ## pnp_ilrl...origin/pnp_ilrl

git show --stat --oneline a93c60b
# docs/chatgpt5p6sol_review.md | 1751 ++++++++++++++++++++++++++++++++++++++++++
# 1 file changed, 1751 insertions(+)

git ls-tree -r --name-only HEAD
# docs/chatgpt5p6sol_review.md

python -c "import metadrive"
# ModuleNotFoundError: No module named 'metadrive'
```

- 当前 tracked 文件只有 `docs/chatgpt5p6sol_review.md`（E1）。
- `origin/Planner_RL` 有 318 个文件（E1）。
- 当前 Python 3.10.12、torch 2.3.0、numpy 1.26.4；MetaDrive/gym/gymnasium/pygame/panda3d 未安装（E1）。
- 硬件：RTX 4070 12 GB、20 CPU、15 GiB RAM、646 GiB free（E1）。

### 7.2 MetaDrive 0.4.3 源码

- `metadrive/envs/base_env.py:188-190`：`physics_world_step_size=2e-2`、`decision_repeat=5`。
- `metadrive/component/pgblock/tollgate.py:11`：`TollGate` block。
- `metadrive/component/buildings/tollgate_building.py:7`：`TollGateBuilding(BaseBuilding)`。
- `metadrive/component/static_object/traffic_object.py:44`：`TrafficCone`。
- `metadrive/policy/idm_policy.py:177`：`IDMPolicy`，使用 `navigation.current_ref_lanes`/`routing_target_lane`/`lane_change_policy`。
- `metadrive_simulator-0.4.3.dist-info/METADATA`：依赖 `gymnasium>=0.28`、`pygame`、`panda3d==1.10.13` 等。

### 7.3 旧分支证据（E2，仅风险证据）

- `origin/Planner_RL:config/env.yaml`：`policy_dt: 0.5`、`mpc_dt: 0.1`、`history_frames: 6`、`topk_objects: 16`、`topk_lanes: 16`、`traffic_lights: false`、blocks 含 `tollgate`。
- `origin/Planner_RL:env/tracking.py`：`interpolate` 6 个 `(ds,dtheta)` → 30 个 10 Hz 位姿；`ExactTracker` 在 `env.step` 后写位姿；`LqrTracker` 输出 `[steer, throttle]`。
- `origin/Planner_RL:net/param_probe.py`：default total 1,278,879 / structured total 1,510,570；active 871,478 / 994,717。
- `origin/Planner_RL:docs/repo_handbook.md`：网络/训练/评测/参数的详细代码口径；`datasets/path_exec_v1`（由 5k expert 数据派生）rows=360,497。
- `origin/Planner_RL:AGENT.md`：IDM+PP 0.756；模型 0.56；LQR+repeat 0.06；LQR+WM plan 0.56；`E_interface` h1 0.64→h5 1.74；WM-state action error≈1.43 vs real≈0.40。
- `origin/Planner_RL:docs/deepseekv4p1_argue.md`：训练/评测装配不一致导致 Calib50 50/50 out_of_road；训练域 v≈8.3 vs 评测域 v≈3.0 的策略自诱发域差；LQR 预瞄只是缓解不是修复。

### 7.4 LAW / EPONA

- `LAW/projects/mmdet3d_plugin/LAW/dense_heads/waypoint_query_decoder.py:239-246`：`wm_prediction` 把 current waypoint 拼到每个 spatial token，经 `action_aware_encoder` + `_wm_decoder` 预测下一 latent。
- `LAW/projects/mmdet3d_plugin/LAW/LAW.py:225-231`：`loss_rec = MSE(prev_pred_img_feat, cur_img_feat.detach()) * wm_loss_weight`；one-step，无 EMA。
- `Epona/README.md`：autoregressive diffusion video world model，DCAE/DiT，4090/DeepSpeed，参数量/计算量远大于 1M。适合借鉴“action-conditioned autoregressive rollout”的思想，不适合移植架构。

---

## 8. 最终判断

**可行性**：可行，但有条件。

- 如果先解决 B1–B6，并接受 §4.1 的单一接口契约与 §4.2 的折中路线，这个项目是可在当前 4070/20 CPU/15 GB RAM 上推进的。
- 如果不解决 B1–B6 就直接开工，最可能的结果是：Stage A 的单帧/10 Hz BC 能跑，Stage D 的 6 步 WM/Plan Head 需要重训；LQR 在 Stage F 才第一次接入，出现旧分支式的 Exact/LQR 断裂；参数/EMA 超预算；验证集被反复用于选 checkpoint，最终结论不可信。
- 当前最值得先做的不是写网络，而是三个 P0 spike：
  1. **Environment spike**：venv + MetaDrive headless reset/step + FPS/RSS + 4 worker 压测。
  2. **Data/Expert spike**：100 个场景的专家采集 + `(ds,dtheta)` round-trip + 过滤 + 存储格式/大小。
  3. **Interface spike**：一个最小 1M 模型 + 6 步 `(ds,dtheta)` rollout + 参数/张量账本 + Exact/LQR 各跑 10 个场景，证明接口和控制器链可行。
- 这三个 spike 通过后，再决定 WM 的结构和 Stage 顺序，比在 1751 行文档上直接堆代码更稳。

---

*本文件只做审阅与提议，未修改任何代码/配置。*
