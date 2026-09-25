# Planner_RL 城市/高速行车规划 —— 可行性分析（v0.9 草稿）

日期：2026-09-24 · 分支：`Planner_RL` · 状态：**调研阶段，未做任何实现改动**
依据：本机实测 + 5 条调研线（MetaDrive / HighwayEnv / RL 算法 / 小参数架构 / MetaDrive 追问）+ 参数预算实测探针。

> 本文档是"先排摸可行性"的交付物。文中【lib-4】【lib-5】标记处待两条调研线返回后补充。

---

> **已确认决策与范围变更见 §13（2026-09-24 起生效，v1.1）**；本文档主体为 v0.9 分析，§13 为准。

---

## 0. 结论摘要

**总体结论：技术路线可行，但按当前设计直接开工会在 4 个点上卡住；其中 3 个是仿真器能力缺口，1 个是算法选型问题。**

| # | 结论 | 依据 |
|---|---|---|
| 1 | **仿真器选 MetaDrive**，HighwayEnv 排除 | HighwayEnv 根本没有交通灯（#506 明确拒绝）、稳定版无程序化地图、无 per-lane 观测、不暴露限速；MetaDrive 覆盖 90% 场景块但需自建交通灯/限速/观测 |
| 2 | **网络 500K 参数预算可行**（实测 264,294 @ H=64） | 本地探针实例化统计，见 §5 |
| 3 | **GRPO 不建议作为主力算法**；用 PPO（备选 SAC），GRPO 作消融 | 连续控制上 GRPO 未经证实；vanilla GRPO 在稠密奖励长时程任务上失败；无成熟实现支持自定义连续策略，见 §6.2 |
| 4 | **"outcome reward → token reward" 不必绕道**：规则奖励模型天然可给每步分，建议稠密+终局混合 | 见 §7.1 |
| 5 | 内存（16GB）是硬约束，显存（12GB）不是 | 单实例单进程模型 + 实测口径，见 §3.4 |
| 6 | 真正的工程量在：**自定义观测类、场景生成+校验器、交通灯系统、MPC/LQR 跟踪器** | 见 §4/§9 |
| 7 | ~~PG 建图耗时~~ → **P0 实测退档**：3/5/8 blocks 建图 0.21/0.33/0.57s；10k 校验 ≈8min（8 workers） | 见 `docs/p0-measurements.md` |
| 8 | **交通行为工程被低估**：IDM 不看灯、无信号路口不让行 → 交通车行为需工程化，否则 reward/信用全是噪声 | 见 §9 R1/R16 |
| 9 | **评测协议必须冻结**（独立进程/固定种子/val-test 分离/worker 重启/数字阈值），否则 KPI 曲线不可比 | 见 §7.3/§9 R13–R14 |

---

## 1. 环境与硬约束（本机实测，2026-09-24）

| 项目 | 实测值 | 影响 |
|---|---|---|
| GPU | RTX 4070 **12GB**（驱动 580 / CUDA 13） | 网络 ≤500K 参数 → 显存完全够；显存不是瓶颈 |
| 内存 | **15Gi 总量，实测可用 ~9.2Gi**（另有 4Gi swap） | **硬约束**：决定并行 env 数量（建议 4–8） |
| CPU | i5-13600KF，20 线程 | 仿真吞吐的瓶颈资源；MetaDrive 单实例单进程 |
| 磁盘 | /workspace 剩余 769G | 充裕；但不要缓存 1 万张地图 |
| Python | 3.10.12 系统版，**无 conda** | `metadrive-simulator` 要求 `>=3.6,<3.12` → 3.10 可用 |
| torch | 2.3.0a0（NGC），`cuda.is_available()==True` | 可用；SB3 新版要求 torch≥2.8，**不要用 SB3**（见 §6.3） |
| numpy | 1.26.4 | 满足 MetaDrive 要求；**必须 pin numpy<2** |
| 仿真器 | **均未安装**（无 gym/gymnasium/highway-env/metadrive/panda3d） | 需安装；pip 可联网（DNS 偶发抖动但重试成功）。**实测 pip 解析通过**：`metadrive-simulator==0.4.3` 仅新增 11 个包，不动系统 numpy/torch（注意 PyPI 名是 `metadrive-simulator`，不是 `metadrive`） |
| 系统库约束 | conda/python 官方库**只读不改** | 方案：`python3 -m venv --system-site-packages .venv`（复用系统 torch/numpy，只往 venv 装新包，不改系统） |

---

## 2. 仿真器选型

### 2.1 MetaDrive（选定）

包名 **`metadrive-simulator` 0.4.3**（2024-12-06 发布；main 最后提交 2025-05，维护力度低）。
依赖：`panda3d==1.10.13`（有 cp310 manylinux wheel）、`gymnasium>=0.28`、`numpy>=1.21.6`（pin <2）、shapely、pygame、opencv-python、scipy、lxml、psutil、matplotlib、mediapy、pillow；资源包 assets.zip ≈134MB。

**能力（与我们需求的对应）：**

| 需求 | MetaDrive 现状 |
|---|---|
| 直道/弯道/匝道/掉头/路口/环岛 | ✅ 有对应 block：`S`直道 `C`弯道 `r/R`进出匝道 `f/F`进出分叉 `O`环岛 `X`十字 `T`丁字 `U`含掉头路口 `$`收费站 `B`双向 `y/Y`瓶颈 `P`停车场 + Merge/Split |
| 10k 场景随机生成 | ✅ 种子决定论生成（`num_scenarios`+`start_seed`）；可 dump/load |
| 交通灯 | ❌ **PG 地图不放置任何交通灯**（issue #773）；0.4.3 的 `IDMPolicy` 完全不检查 `red_light`（全库仅 `base_vehicle.py` 有该标志）；红灯检测靠**接触式"隐形墙"**（`PLACE_LONGITUDE=5`，撞上才触发）→ 正确停车需自写"停车线前减速"逻辑 + `IDMPolicy` 子类 + 每次 reset 重建灯的 manager；ScenarioNet 回放路径有灯，但回放/IDM 交通车都不读灯状态（lib-5） |
| 道路限速 | ❌ PG 车道默认 1000，仅收费站设置；需自建 `set_speed_limit` 管线 |
| OD（周车列表） | ❌ 无内置 per-object 向量观测；lidar 只能给 k 近邻×4 维；需自建 `BaseObservation` |
| LD（车道线/车道段） | ❌ 同上，需从 `map.road_network`/`navigation.current_ref_lanes` 自建 |
| 导航（dist+action） | ⚠️ 有 checkpoints（未来2个点×5维）、`route_completion`、`navigation_command∈{forward,left,right}`；**无匝道/掉头专用标签**，距离需自算 |
| cut-in/cut-out | ❌ 无内置；需脚本化（spawn + `WaypointPolicy`） |
| 拥挤交通 | ✅ `traffic_density`（0.1→8 车，0.5→43 车）；IDM 默认策略 |
| 多智能体（30% 车为 agent） | ⚠️ `MultiAgentMetaDrive`（默认 15 agents）/`MixedTrafficEnv.rl_agent_ratio`；**一个进程只能一个实例**；论文值 40 agents 时 60 FPS |
| 精确跟踪（阶段A） | ⚠️ `WaypointPolicy` 可直接 set_position/velocity/heading（运动学播放），但内置按 dt=0.1s、horizon=10 → 需自写 0.5s×6 点版本 |
| MPC/LQR（阶段C） | ✅ 连续动作 `Box(-1,1,(2,))=[steering,throttle]`；可 `BasePolicy` 子类接入外部控制器 |
| 奖励真值 | 碰撞标志/出界/`red_light`/线型（`CONTINUOUS/YELLOW` + `on_white/yellow_continuous_line`）/`lane.local_coordinates`/`route_completion`；**无碰撞对象ID、无双黄线、无 per-lane 限速** |

性能（公开值，需本机复测）：论文 300 FPS（单实例+10 交通车）、60 FPS（40 agents 多智能体）；README 宣称 1000 FPS；用户实测 1100 FPS（纯物理）；CI 断言 >200 FPS。渲染可完全关闭（`use_render=False`）。

已知坑：同进程 reset 非确定性（#758）、地图生成 MAX_TRIAL=5 回溯可能产生退化布局、每次 reset 约 1MB 泄漏（关渲染）、弯道半径不可配置（#671）、`preload_models` 启动开销（#469）。**建图耗时是最大未知量**：社区基准（未验证）显示 60 blocks 平均 ≈362s、最大 2789s（回溯爆炸）；issue #273 的 nuPlan 真实地图 1000 次 reset ≈300s（≈0.3s/次）说明"真实地图快、PG 生成慢"；且 `store_map=False` 时**每次 reset 都重新生成地图** → P0 必须实测并用 LRU 缓存工作集。另：**跨进程种子确定性、gymnasium 1.3.0 与 0.4.3 的兼容性、fork/spawn 启动方式均未验证**（P0 冒烟测试覆盖）。

### 2.2 HighwayEnv（排除）

稳定版 1.12.1（2026-08），维护活跃，但：
- **交通灯：全库不存在**，intersection 是"无信号路口"，#506 被明确关闭（"I do not plan on making any addition"）；
- 程序化地图只在 `main`（2026-08-31 合入，实验性、单智能体、默认无交通、无场景标签），稳定版全是硬编码地图；
- 观测类是 `KinematicObservation`（不是 `VectorObservation`，旧名早于 v1.0）：固定 (vehicles_count, F) 张量+presence mask；**无 per-lane 列表、不暴露限速、无向量历史**；
- 导航观测（仅 main）= 距离+航向，无 action 建议；
- 性能基准（2021 硬件）：highway-v0 ~1.1 policy steps/s、highway-fast ~14.5；
- 结论：**无法满足场景清单（交通灯/标注化 10k 场景/OD-LD 列表/限速），不作为训练环境**。

---

## 3. 场景生成方案（10,000 训练 + 1,000 验证）

### 3.1 分类体系（多标签）

| 组 | 标签 |
|---|---|
| 几何 | `straight`, `curve`, `ramp_in`, `ramp_out`, `fork_in`, `fork_out`, `merge`, `split`, `intersection_std`, `intersection_T`, `intersection_uturn`, `roundabout`, `bottleneck` |
| 交通形态 | `free_flow`, `car_following`, `cut_in`, `cut_out`, `crowded`, `lane_change`, `zipper_merge` |
| 控制要素 | `traffic_light`, `speed_limit_zone` |
| 导航动作 | `go_straight`, `turn_left`, `turn_right`, `exit_ramp`, `enter_ramp`, `uturn`, `roundabout_exit` |
| 难度 | `easy`/`medium`/`hard`（密度、间隙、速度差、事件距离） |

抽样：训练集分层抽样保证每个几何标签 ≥600、每个交通形态 ≥400，并覆盖多标签组合（如 `intersection_std ∧ traffic_light ∧ turn_left ∧ crowded`）；验证集 1000 个**冻结种子**（同分布分层），用于 KPI 可比性。

### 3.2 生成器设计

- 地图：默认 v2 分布里 U-turn/fork/merge/split/收费站/双向 = 0.0，**必须自定义 `block_dist_config` 或显式 `BLOCK_SEQUENCE`**（按分类体系反推块序列）。
- 交通灯：在信号化路口按车道 spawn `BaseTrafficLight` + 配时（周期/相位），并给交通车注入"红灯停车"逻辑（lib-5 返回后细化实现路径）。
- 限速：建图后按道路类型批量 `set_speed_limit`。
- 线型（实线合规用）：**无建图期 API**，且 `load_all_maps` 会忽略 `map_features` 只按 block 重建 → 任何线型后处理必须在**每次建图后重放**（lib-5）。
- cut-in/cut-out：脚本化（spawn 目标车 + `WaypointPolicy` 轨迹 / 强制换道），带事件触发条件（TTC/距离）。
- 输出：`scenarios_train.json` / `scenarios_val.json`，每条 ~200B（11k 条 ≈2.2MB）：

```json
{"id": 0, "seed": 1000, "split": "train", "blocks": "SCO",
 "geometry": ["curve","roundabout"], "traffic": {"density": 0.3, "patterns": ["cut_in"]},
 "signals": {"traffic_light": true, "cycle_s": 30}, "limits": {"speed_limit_mps": 13.9},
 "nav": {"turns": ["left"], "min_completion": 0.9}, "difficulty": "medium"}
```

### 3.3 校验流程（真正的成本所在）

`gene_env.sh`：生成 spec → **逐条实例化校验**（路线存在、地图非退化、分类标签自检、出生点无重叠、导航可达）→ 失败重采样 → 输出 JSON + 校验报告（各类别覆盖直方图）。

- 成本估算：**强依赖建图耗时（P0 必测）**。若 PG 建图在复杂地图上真达到数十~数百秒（社区基准，未验证），则 11k 次校验与"每 episode 重建地图"都不可行 → 必须限制 block 数（小地图）+ LRU 缓存工作集 + 必要时预生成子集 dump/load；若实测 ≈0.3s/次（issue #273 口径），11k 次校验 8–12 worker 并行 ≈5–15 分钟。
- **不要缓存 1 万张地图**（`store_map=True` 内存不可行、dump 文件 10–50GB 且启动加载不可行）；运行时按 seed 按需生成 + LRU 缓存（几百张）。

### 3.4 并行与内存口径

- MetaDrive **一个进程一个实例** → 训练用 `SubprocVecEnv` 风格 4–8 worker；每个 worker 关渲染。
- 16GB 下 4–8 进程可行（每进程 RAM 未公开，**P0 必须实测**），实测后定 worker 数。
- 若后续要 30% agents（P5）：单进程内多 agent 共享策略，成本随 agent 数上升（论文 40 agents→60 FPS）。

---

## 4. 观测与输入接口（OD / LD / Nav / 交通灯 / 限速 / 自车）

### 4.1 目标 schema（与用户设计一致）

| 通道 | 内容 | 维度（每帧） |
|---|---|---|
| OD top-16 | dx,dy,vx,vy,cosθ,sinθ,L,W,type,confidence（+mask） | 16×10 |
| LD top-16 | dx,dy,方向,曲率,限速,线型,到自车距离（+mask） | 16×7 |
| Nav | 距离序列 + 动作 one-hot（直行/左/右/出匝道/进匝道/掉头） | ~14 |
| 交通灯 | one-hot（红/黄/绿/无） | 4 |
| 限速 | 当前道路限速 | 1 |
| 自车 | v,a,yaw_rate,steer,heading,曲率等 | ~8 |
| 历史 | **6 帧 @0.5s（3s）** 环形缓冲（10Hz 采样每 5 步存 1 帧；首帧重复填充+mask） | ×6 |

### 4.2 在 MetaDrive 的落地方式

- **必须自写 `BaseObservation` 子类**（`agent_observation=MyObs`）——这是最大的单项工程量。
- OD：从 `env.engine.get_objects()` / `lidar.get_surrounding_vehicles_info` 取车与静态物，自行实现 top-16 选择（建议：距离+TTC 混合准则，按重要性排序，保证时序稳定）。
- LD：从 `navigation.current_ref_lanes`/`next_ref_lanes` 与 `map.road_network` 抽取车道中心线段（前向 N 米切分）+ 线型（`PGLane.line_types/line_colors`）+ 限速。
- Nav：`get_checkpoints()`（未来 2 点）+ 自算"到下一个转向点的距离" + 将 `navigation_command` 映射到 6 类动作（匝道/掉头需从 route/block 类型推断）。
- 交通灯：需自建信号系统（见 §3.2），观测里输出"当前车道相位 + 到停车线距离"；**闯红灯判定必须用自建"停车线越线检测"，不能用接触式 `vehicle.red_light`**（撞隐形墙才触发 → 漏判、噪声大；Oracle must-fix）。交通车看灯/无信号路口让行逻辑同属 P1 范围（Oracle 指出的最大 regret 来源）。
- **槽位一致性**：OD/LD 是变长集合，监督 world model 时需要固定槽位 → 采用"上一帧槽位优先匹配 + 匈牙利匹配"或 mask 机制（lib-4 有对应文献支撑）。

---

## 5. 网络架构与参数预算

### 5.1 结构（与用户设计对齐）

```
输入(6帧) → 每类输入投影(H) → 时间 GNN（逐节点 GRU/注意力，6 帧）
          → 空间 GNN（OD+LD+ego 节点消息传递 ×2 层）
          → MoE：primary（恒激活）+ 16×specific（router 门控，残差相加）
          → 输出头：① 自车 6 点轨迹(12) ② OD rollout(6步) ③ LD rollout(6步)
```

### 5.2 实测参数（本地探针 `/tmp/opencode/probes/param_probe.py`）

| 模块 | 参数 |
|---|---|
| 输入投影（ego/nav/OD/LD） | ~3.4K |
| 时间 GRU（共享，H=64） | 24,960 |
| 空间 GNN ×2 层 | 50,432 |
| primary expert | 8,320 |
| 16×specific expert（各 64→64→64） | 133,120 |
| router | 5,200 |
| world model（GRUCell+双解码器） | 33,930 |
| 策略头 | 4,940 |
| **合计** | **264,294（预算的 52.9%）** |

H=96 时 586,886（**超预算**）→ **H 保持 64–80**，剩余预算留给 embedding/辅助头。

### 5.3 设计要点与风险（lib-4 证据）

- **层序有先例**：HiVT（CVPR'22）正是"逐帧局部空间（agent–agent + agent–lane，r=50m）→ 时间层×4 → 全局交互层×3"；Wayformer 消融显示分解式时间/空间注意力的顺序不敏感。VectorNet 编码器仅 **72K** 参数（全 MLP hidden=64）即可支撑向量化表示。
- **primary + specific 的残差关系**与 DeepSeekMoE"共享专家 + 路由专家"一致（有文献支撑）；router 用场景标签监督属于任务条件路由（Task-MoE：+3.6 BLEU）。
- ⚠️ **16 个 routed expert 在 500K 预算下每个仅 8.3K 参数——证据不支持**：所有正向 MoE 证据都在 100M+ 规模（DeepSeekMoE/V3、Switch、ST-MoE）；token-choice MoE 已知欠训练 expert（Expert Choice 论文明确列出 load imbalance / under-specialized experts）；DeepSeekMoE 也强调路由坍塌与负载均衡。
  **Oracle 评审结论**：16×16.6K 本质是"16 个场景条件 residual adapter"，不是 MoE；**建议直接选 a) 4–8 个加宽 expert**，16×128（≈396K）仅作上界；**必须记录每 expert token 数、输出范数、primary 漂移**，否则 residual 会被 primary 吸收、specific 训不动。
- router 训练：label 监督必须配显式均衡机制（Expert-Choice/分配式，或 Switch aux loss α≈1e-2 + **router z-loss** 稳定训练）；**监控每个 expert 的 token 数与 loss**；路由抖动大时可用 StableMoE 的"蒸馏 + 冻结 router"。
- ⚠️ **标签语义（Oracle must-fix）**：spec 级场景标签（如 `cut_in`）在 t=0 不可观测 → router 的 label 必须来自**当前步可观测状态**（事件是否激活、局部密度/TTC、路口类型是否可见），否则 router 学的是 scenario-ID 泄漏而非场景语义。
- ⚠️ **动作参数化（已确认 2026-09-24）**：动作 = **`(Δs, Δθ)`**（下一步 0.5s 的弧长 + 航向变化，等价固定 dt 下的 `(v,κ)`）；有界、平滑、探索友好，且天然用圆弧插值。轨迹监督目标仍是 6 个点 `(x,y,θ)`。
- ⚠️ **推理时没有场景标签**：label 监督只能是训练期辅助损失，router 必须从观测推断场景；且场景是**多标签**（cut-in ∧ 路口 ∧ 拥挤）→ router 用 BCE（多标签）而非 softmax 单选；阶段 C 去掉强监督后建议保留小权重辅助 + 路由熵监控。
- ⚠️ **输出参数化**：学习式规划器普遍输出自车坐标系航点（TransFuser/ChauffeurNet/TNT）；Apollo 下游是 0.1s 稠密 `PathPoint(x,y,θ,κ,s)`。但 **0.5s 点之间直接线性插值会产生切线不连续、节点处曲率未定义 → LQR/MPC 方向盘抖动**；建议 6 点先拟合 **C2 样条（或 min-snap）**，航向取样条导数，再以 10Hz 采样给 MPC/LQR，并加 `(κ(s), v(s))` 辅助头。6 个点（0.5s ≈ 7–14m 间距）对路口几何偏粗，可能需要更多点或曲率辅助监督。
- **容量现实检查**：PilotNet 250K（车道保持，真车）、TinyLidarNet 11–22 万（F1TENTH）、VectorNet 编码器 72K、HiVT-64 662K；**没有 ≤500K 参数实现"无信号路口 + 匝道汇入"的公开先例** → 500K 对结构化向量输入 + 强辅助监督"合理但未证实"，建议预留 1–2M 参数的扩展路径（与硬性 ≤500K 存在冲突，需决策）。
- **集合处理**：固定 K=16 + padding/valid mask + 跟踪 ID 关联即可；不需要 slot attention（若确实要无身份分组，则加 SAVi 式时序槽位预测器保证槽位跨时间一致）。

---

## 6. 训练方案

### 6.1 用户三阶段计划的评估与修改建议

| 阶段 | 用户计划 | 评估/建议 |
|---|---|---|
| A | 不做 rollout，真实环境交互训 primary+specific，默认精确跟踪 | ✅ 合理（≈开环轨迹学习+RL）。**建议先用经典规划器/IDM 专家做 BC 预热**，再 PPO 微调，显著降低样本需求 |
| B | 全部锁住，只训 world model（OD/LD rollout） | ⚠️ **不必与环境交互**：直接用阶段 A 的 replay/离线数据训练即可（更便宜、可重复）；监督用"直接多步损失"（K 步联合），加 mask、噪声增强（p≈0.3）、自车运动条件；务必与**匀速基线**对比 ADE/FDE。⚠️ **"优于 CV" 必要非充分**（open-loop 赢 CV ≠ 规划有用）；**若 WM 要用于 learned MPC，必须 action-conditioned（ego 计划作为输入），该决策必须在 P2 结束前定**，不能拖到 P3（Oracle） |
| C | 降 LR，放开 specific+world model+rollout，env 用 MPC/LQR，router 弱监督 | ✅ 正确方向，但注意**分布漂移**（精确跟踪→跟踪误差）：建议 KL/BC 锚 + 小步放开 + 先固定 world model 权重观察 |
| 全局 | — | ⚠️ "epoch" 需定义（RL 里建议 = 固定 env steps，如 100k 步），否则"每 5 epoch 验证"无意义；验证种子必须冻结 |

### 6.2 算法选型（对"用 GRPO"的反驳）

**证据（lib-3）：**
- 系统研究（arXiv:2511.03527，CartPole→Humanoid）：**所有无 critic 基线在长时程任务上都不如 PPO**，只有 CartPole 例外；
- **GR2PO**（arXiv:2609.19850）：原始 GRPO 在稠密奖励 MuJoCo 上**完全失败**；改为"分组折扣回报 + 逐时间步组归一化"后才与 PPO/SAC 竞争（且只在单篇论文、无公开代码）；
- 连续动作 GRPO 的理论扩展（arXiv:2507.19555）自称"remains unexplored"，无实证；
- **没有任何成熟 GRPO 实现支持自定义连续策略**（TRL/verl/OpenRLHF 都是 LLM 专用）；自研约 200–400 行；
- 驾驶 RL 的实证规模：MetaDrive 基准 ~1M 步、基本能力 1–6M 步；CARLA PPO 10M 步/32h/1×A100。

**建议：**
- **主力：PPO + GAE**（CleanRL 风格自研，接受自定义 GNN/MoE 模块；稠密规则奖励天然给每步信用）；
- **备选：SAC**（MetaDrive 论文显示状态输入下样本效率更高：SAC-RS 0.80 vs PPO-RS 0.21），代价是需要 replay（内存注意）；
- **GRPO：只作消融实验**，且用 GR2PO 式"分组折扣回报 + 逐时间步归一化"，不用 vanilla 版本；
- 若有真实日志/规划器标签：**BC → IQL/CQL 离线 → PPO/AWAC 在线微调**（WaymoOfflineRL：CQL 54.4% vs BC 17.3%；MetaDrive：CQL 72% vs BC 13%）。

### 6.3 框架

- **CleanRL**（单文件 PPO/SAC，任意自定义模块）首选；**skrl**（自定义 Model + RNN memory）次选；Tianshou v2 beta 若需要离线算法；**SB3 当前要求 torch≥2.8，与本机 torch 2.3 不兼容，不用**。

---

## 7. 奖励模型与 KPI

### 7.1 对"outcome reward 转 token reward"的意见（反驳）

规则奖励模型在驾驶里**天然可以逐步计算**（每步的碰撞/压线/速度/加速度），把逐步信号聚合成 outcome 再转回 token 是**有损**的。证据（lib-3）：纯稀疏从零学习劣于稠密→稀疏退火；稠密特权奖励会过拟合/失配（arXiv:2512.04279）。

**建议：** 每步稠密项（进度/舒适/合规/效率）+ 终局项（到达/碰撞/超时）+ CaRL 式乘性/终止惩罚；训练后期再退火稠密项。若必须 outcome-only，则用 GR2PO 式分组折扣回报，而不是 episode 标量组基线。

**Oracle 补充（更强替代）**：进度项用 **potential-based shaping**（保策略排序不变，避免"绕圈刷进度"）；配 terminal bonus + 乘性惩罚，防止"龟速/自杀式刷安全分"的 reward hacking。

### 7.2 五项准则在 MetaDrive 的可测性

| 准则 | 可测性 | 备注 |
|---|---|---|
| 安全性（碰撞/出界） | ✅ | `crash_*` 标志 + `contact_results`；**无碰撞对象 ID**（只知类型） |
| 舒适性（横纵加速度/急动度） | ✅ | 从车辆状态差分 |
| 效率（车速） | ✅ | 速度/限速比、到达时间 |
| 合规性：闯红灯 | ⚠️ | 有 `vehicle.red_light`，但**交通灯需自建**（PG 无灯） |
| 合规性：越实线 | ⚠️ | 有 `CONTINUOUS` 线型 + `on_*_continuous_line` 标志；**无"双黄线"概念** → 降级为"连续实线不可跨越" |
| 合规性：超速 | ⚠️ | 需自建 per-lane 限速（PG 默认无） |
| 达成率（路口/匝道跟随 nav） | ⚠️ | `route_completion` 可用；**匝道/掉头无专用 nav 标签**，需自行推断 |

### 7.3 KPI 矩阵（每 5 epoch，冻结 1000 验证场景，逐类 + 总体）

| 维度 | 指标 |
|---|---|
| 安全 | 碰撞率（车/物/边界）、出界率、近失（min-TTC）率 |
| 舒适 | 平均/最大 |a_lon|、|a_lat|、jerk、方向盘速率 |
| 效率 | 平均车速/限速比、到达时间 |
| 合规 | 闯红灯率、压连续实线率、超速率 |
| 达成 | `route_completion≥0.95` 比率、各导航动作成功率 |
| 稳健 | 各难度成功率；**按场景类别（路口/cut-in/merge/匝道…）分别出分** |

**Eval 协议（冻结，Oracle must-fix）**：独立进程评测、固定种子、每场景 1 次 rollout、确定性交通；**val/test 分离**（避免用 val 做模型选择造成选择泄漏）；worker 定期重启（对抗 ~1MB/reset 泄漏）；每个 gate 事前写**数字阈值**；预算 eval wall-clock（1000 场景 × 每 5 epoch）。"epoch" = 固定 env steps（如 100k，config 中定义）。

---

## 8. 工程结构与配置（与用户要求一致）

```
config/         # yaml：env(场景/交通/信号)、model、train、eval、runner
env/            # 场景生成器 + 校验器 + MetaDrive 包装（自定义 obs/reward hooks）+ 调用接口
reward_model/   # 规则奖励（分项 + outcome 聚合 + 信用分配）
net/            # GNN + MoE + world model + 输出头（~264K 参数）
pipeline/       # 阶段A/B/C 训练管线、评测管线、rollout/PPO 循环
tools/          # gene_env.sh, train.py, test.py
```

- 依赖：`.venv`（`--system-site-packages`）+ `metadrive-simulator`（pin numpy<2）+ 现有系统 torch；
- 日志：tensorboard（已装 2.9.0）+ CSV；
- 注意：pip DNS 偶发抖动（重试可成功）；如需更稳可配置镜像源。

---

## 9. 风险清单（按影响排序）

> Oracle 评审：**R11 是头号 kill risk（优先级高于 R1）**；R1 是"成本已知的脏活"；下表已按此调整。

| # | 风险 | 根因 | 缓解 | 验证 |
|---|---|---|---|---|
| **R11** | ~~**PG 建图耗时**~~ **【P0 实测退档】** 建图 0.21–0.57s（3–8 blocks），无需 dump/load；保留显式序列仅为可复现 | 回溯在 60-block 级才成瓶颈（社区数字），小地图无碍 | P0 实测 | 已完成 |
| R1 | 交通灯合规训练不可用 | PG 无灯；`IDMPolicy` 完全不读 `red_light`；检测是"撞上隐形墙"才触发 | 自建：灯 manager（每 reset 重建）+ **停车线越线检测（奖励/KPI 判定口径，不用 contact 标志）** + `IDMPolicy` 子类；交通车看灯/让行同属工程范围 | 信号路口红灯停车率（越线检测口径） |
| R12 | 500K 参数对路口/汇入容量未证实 | 无 ≤500K 公开先例；MoE 正向证据在 100M+ 规模 | 预留 1–2M 扩展路径；强辅助监督；先窄后宽 | 各类别 KPI 是否达标 |
| R2 | 内存不足导致并行度低、训练慢 | 16GB RAM + 单实例单进程 + 潜在 replay | 实测每进程 RAM；4–8 worker；replay 存单帧、历史窗口在线拼 | P0 实测 FPS/RAM |
| R3 | 场景校验成本高/退化地图 | MAX_TRIAL=5 回溯、无导航路线、出生重叠 | 并行校验 + 重采样 + 覆盖直方图 | 校验报告 |
| R4 | world model 坍塌成匀速预测 | 多步回归的均值坍缩（TD-AE 常数解） | 直接多步损失 + 噪声增强 + 自车运动条件 + 与 CV 基线对比 | ADE/FDE vs CV |
| R5 | 阶段 C 分布漂移/性能回退 | 精确跟踪→MPC/LQR 跟踪误差 | KL/BC 锚、小 LR、分步放开、先冻结 WM | 阶段 C 前后 KPI 对比 |
| R6 | router 坍塌/推理期无标签失配 | 多标签场景 + 推理无 label | BCE 多标签 + load-balance + 保留小权重 label 辅助 + 熵监控 | 路由分布/熵 |
| R7 | 16 个 expert 容量不足 | 每个仅 8.3K 参数 | 优先加宽 expert 或低秩 adapter | 各 specific 场景 KPI |
| R8 | 精确跟踪（阶段A）需自写策略 | 内置 `WaypointPolicy` 假设 dt=0.1s/horizon=10 | 自写 6→30 点插值 + 0.5s 版 waypoint policy（与阶段C的 30 点接口复用） | 轨迹复现误差 |
| R9 | 多智能体（30% agents）成本高 | 单进程单实例 + agent 数线性开销 | 放 P5；参数共享；先 ego-only | 多智能体 FPS/KPI |
| R10 | MetaDrive 维护停滞/兼容性 | 0.4.3 已 21 个月；numpy2 不支持 | pin 版本（0.4.3 + numpy<2，gymnasium 兼容性 P0 冒烟，必要时 pin 0.29.x）；**子进程用 spawn 启动**（engine singleton 与 fork 不兼容）；封装隔离，保留换仿真器接口 | 冒烟测试 |
| R13 | 评测确定性与选择泄漏 | #758 同进程 reset 非确定；val 既做 KPI 又做模型选择 | 独立进程评测 + 固定种子 + **val/test 分离** + 协议写死 | KPI 复现性 |
| R14 | ~~**~1MB/reset 泄漏 × 长跑耗尽内存**~~ **【P0 实测退档】** store_map=False 实测 +11MB/1000 resets | 已知泄漏（关渲染仍存在） | worker 重启周期 5000（config 已冻结）；监控 RSS | 长跑 RSS 曲线 |
| R15 | reward hacking + 吞吐预算未算 | 龟速刷安全/舒适分、`route_completion` 被利用；训练步预算 × FPS、eval wall-clock 未估算 | potential-based shaping + 乘性/终止惩罚；开工前算吞吐预算表 | 奖励曲线/行为审计 + 吞吐表 |
| R16 | 交通行为工程被低估 | IDM 不让行、不看灯 → ego 被堵/被撞，reward 与信用分配全是噪声 | P1 明确 traffic 行为范围；必要时脚本化/自定义 policy；无信号路口让行规则一并定 | 场景内交互成功率 |

---

## 10. 分阶段实施计划（每阶段含 Gate）

| 阶段 | 内容 | 交付物 | Gate（数字阈值事前写死） |
|---|---|---|---|
| **P0 环境与四测** | venv + 安装 + headless/gymnasium/spawn 冒烟；仓库骨架；**四测**：① PG 建图耗时（随机生成 vs 显式 `BLOCK_SEQUENCE`，按 block 数分档）② 跨进程种子确定性 ③ ~1MB/reset 泄漏曲线 ④ IDM 专家 BC 数据 spike（交通灯 spike 已随交通灯移除） | 实测报告 + 最小可跑 env | **数字 go/no-go（暂定，P0 结束统一冻结）**：PG 建图 ≤2s/张（复杂图 ≤5s）、单实例 ≥150 FPS（10 交通车）/ ≥50 FPS（40 车）、RSS 增长 ≤1GB/千次 reset 且 worker 重启周期 ≤1000 reset、1000 场景 eval wall-clock ≤20min（8 worker）（Oracle） |
| **P1a 场景最小集** | ~200 场景覆盖全部类别（小地图优先）；自定义 obs；交通灯/限速/cut-in 最小实现 | 200 场景 + 覆盖直方图 | 类别覆盖 100% + 路线有效（Oracle） |
| **P2 细切片（thin slice）** | 直线+跟车端到端：net 最小版 + 6→30 插值 + 精确跟踪 + PPO；**提前验证 30 点 LQR 接口**；**定 WM 用途**（aux 监督 vs learned MPC；后者需 action-conditioned） | 端到端跑通 + 首批 KPI | 训练循环稳定 + 直线跟车 KPI ≥ 阈值（Oracle） |
| **P1b 场景全量** | 10k 训练 + 1k 冻结验证（分层抽样 + 校验 + 覆盖报告） | scenarios_*.json + 校验报告 | 校验通过率 ≥99% 且建图耗时在预算内（Oracle） |
| **P3 阶段B（world model）** | 用阶段 A/P2 的 replay 离线训练多步预测（mask/噪声/自车运动条件） | ADE/FDE 报告（vs 匀速/常数基线） | 优于 CV 且不坍塌；用于规划时需 action-conditioned 消融（Oracle） |
| **P4 阶段C（闭环）** | MPC/LQR 跟踪器；放开 specific+WM；router 弱监督；KL 锚；GRPO 消融 | 闭环 KPI 报告 | KPI ≥ P2 且安全不回退（数字阈值）（Oracle） |
| **P5 多智能体（可选）** | 30% agents、参数共享 | 多智能体 KPI | 成本/收益评估 |

> 所有 gate 阈值在 **P0 结束时统一冻结**（校准依据 = P0 实测吞吐/建图耗时/内存曲线）；P2 之后的阈值沿用 P0 口径。

---

## 11. 待确认问题（需要你决策）

1. **依赖安装**：接受 `.venv --system-site-packages`（复用系统 torch，不改系统库）吗？还是你有内网源/指定版本要求？
2. **world model 的用途**：rollout（OD/LD 推演）只作辅助监督（表示学习），还是推理时要用于规划（learned MPC）？这决定 P3/P4 的架构与评估方式（**建议 P2 结束前定**；若用于规划必须 action-conditioned）。
3. **交通灯范围（开工前必须先答）**：自建信号灯系统（含交通车看灯/让行）是 P1 必做，还是可以先只做"无信号路口 + 简化红灯"？直接决定 P1 工期。
4. **阶段A 的监督来源**：是否允许先用经典规划器/IDM 专家生成 BC 数据（推荐），还是坚持纯 RL 从零开始？
5. **合规项降级**：MetaDrive 无"双黄线"，是否接受降级为"连续实线不可跨越"？
6. **算法确认**：是否接受"PPO 为主 + GRPO 仅作消融"？若坚持 GRPO 为主，需要接受样本效率与调参风险。
7. **500K 硬上限**：若路口/汇入场景在 500K 下达不到 KPI，是否允许放宽到 1–2M？（lib-4 建议预留扩展路径）
8. **地图规模**：建图耗时直接决定吞吐；是否接受"小地图（少 block）+ 短路线"以换取建图速度？
9. **sim-to-real 预期**：纯仿真研究，还是要上车？若要上车，需要补车辆动力学标定/域随机化计划（当前报告未覆盖）。

---

## 12. Oracle 评审记录（Gate 1，2026-09-24）

**结论**：选型 sound；风险排序 needs-change（R11 应为头号）；架构/排序各 1 处 needs-change；训练方案 sound（1 处加强）。

**Must-fix（已全部落入本文档）**：
1. R11 升为头号风险，P0 扩为"四测 + 数字 go/no-go"（§9/§10）；
2. 闯红灯判定改用"停车线越线检测"（不用接触式 `red_light`）；交通车看灯/让行纳入 P1（§4.2/§9 R1/R16）；
3. Router label 改为"当前步可观测状态"；expert 定 4–8 个加宽 + utilization/primary-drift 监控（§5.3）；
4. 冻结 eval 协议（独立进程/固定种子/val-test 分离/worker 重启/数字阈值）（§7.3）；
5. 风险清单补 R13–R15；gymnasium pin 与 spawn 启动入 P0；动作参数化列为待决策（§5.3/§9/§11）。

**Remediation 说明**：本修复按 Oracle 自身处方执行（沿用仓库既有 GeoPlanV2 先例），不追加 re-review；剩余 2 次 re-review 额度保留。

---

## 13. 已确认决策与范围变更（2026-09-24，v1.1）

| # | 决策 |
|---|---|
| 1 | Router 监督 = 由 spec 派生的**逐步可观测标签**（cut-in 当前是否激活 / 是否在路口内 / 密度是否超阈值），多标签 BCE；spec 级标签只用于**构造**逐步标签 |
| 2 | 架构 **B**：动作 = 下一个 0.5s 点（`(Δs, Δθ)`）；3s 轨迹由 net 内部 rollout 产生（仍被监督） |
| 3 | World model **ego-conditioned**（以 ego 未来计划为条件输入） |
| 4 | Rollout = **B1**（策略自回归 ×6 + world model 推演 OD/LD） |
| 5 | 阶段 A：**BC 预热**（仿真内 IDM/经典规划器专家产标签）→ PPO 微调 |
| 6 | 阶段 A 顺序：**先训 primary → 锁住 → 训 specific**；primary 恒激活、不受 router 控制；router 只控 8 个 specific（残差关系，DeepSeekMoE 共享专家模式）；sigmoid 门控 + 零初始化 + 阶段 C primary lr×0.1 + 有效 N 监控 |
| 7 | MoE **软加权** |
| 8 | 地图：spec 固化 `BIG_BLOCK_SEQUENCE`（跳过随机搜索）+ 有界 LRU 工作集（几百张） |
| 9 | 范围：限速 ✅；导航用现有 `get_checkpoints()`+`navigation_command`（不算距离）✅；实线合规 ✅；cut-in/out 脚本化 ✅；掉头/匝道/分叉/merge ✅；**新增收费站/瓶颈/双向**；**交通灯全部移除**（含输入通道，保留占位通道）；**多智能体暂不做** |
| 10 | 小地图 + 短路线 ✅ |
| 11 | 解耦/可插拔结构；监控补**场景 label 统计 + MoE 路由统计**；hook/probe 非侵入；需要**可视化入口**（`tools/visualize.py`、`tools/inspect_dataset.py`） |
| 12 | 验收：① 数据集抽验无"与物理时间相悖"现象（帧间位移/航向一致、速度有界、无时间倒流）；② 各训练阶段 loss/KPI 符合预设期望区间 |

**默认决策（可否决）**：动作参数化 `(Δs, Δθ)`；BC 专家 = MetaDrive `IDMPolicy`；验收标准写入 `config/eval.yaml`。

**范围变更影响**：R1（交通灯）移出 P1，降为可选增强；R9（多智能体）P5 不做；§4.2 信号条目改为占位通道；§10 P0 第④测改为"IDM 专家 BC 数据 spike"。

---

## 14. 训练流程修订 v1.1（2026-09-25 用户确认并实施）

**修订动机（因果一致性）**：原实现里 Stage A 的 BC 直接监督 `traj_xy`，但 `traj_xy` 是内部 rollout 的产物，
而 rollout 依赖**尚未训练**的 world model → 监督"假观测下产出的轨迹"不合理（且梯度还会顺带污染 WM）。
用户提出并确认按"先模型、后策略、再 RL"重排：

| 阶段 | 内容 | 可训练 | 监督 GT | 验收证据 |
|---|---|---|---|---|
| **A：WM teacher forcing** | ego 条件=专家 GT 动作序列；目标=未来 OD/LD 帧（(episode, step+5k) 查表、t0 对齐、mask+valid）；直接多步损失（Huber+角度）；ego 计划噪声增强 | 编码器/时序/空间/**MoE（共享）**/WM | 未来 OD/LD 真值 | **ADE 1.647 / FDE 2.739 vs 匀速 3.269 / 4.483 ✓**（`runs/train/stage_a_matched`）|
| **B：Planner BC** | 先 primary(+主干) 后 specific（冻结 primary）；动作 BC（`action_mu` vs 专家首动作）+ **rollout 轨迹小权重辅助**（WM 冻结+detach）+ router BCE | 主干/MoE/策略头/specific | 专家动作 + 专家 traj6 + 逐步标签 | ds **3.319m**（专家 3.260）；aux=0 消融 off-road 1.0 → 0.5（辅助必需）；10 条评测 succ 0.4 / off-road 0.5 / speed 0.75 |
| **C：PPO RL** | rollout + 闭环（LQR 跟踪 6 点预瞄）；KL 锚到 **Stage B 快照**、系数 0.05→0 衰减；primary lr×0.1；WM 先冻后放 | specific/策略头/价值头（WM 后放） | 规则奖励（PPO/GAE） | 50 updates 跑通（44–49 steps/s，无坍塌）；`explained_var≈0` → 下一版加 critic 预热 |

**关键实现修正（本轮）**：轨迹监督目标对齐（`traj6`）；动作损失接线；策略头改 sigmoid 非饱和；WM 冻结+detach；
**跟踪器参考改为 6 点预瞄**（原单动作弧是"偏出车道+速度衰减"的元凶）；评测注入 `prev_action`（原 OOV）；
GL 运行时守卫（spawn worker）；观测 scope 改盒式（前100/后50/左25/右25）。

**当前差距（诚实记录）**：策略 vs 冻结基线（50 条 val）：exact 执行 0.40/0.50 vs 0.82/0.06；LQR 闭环 0.20–0.24/0.68–0.70。
主要缺口是**横向/车道保持**与**终点到达行为**（策略会冲过路线终点继续开，rc>1）；下一步：更大 BC 数据（2000 场景采集中）、
critic 预热、跟踪器调参（需干净指标）。

**已知风险**：MetaDrive 对少数脚本事件场景存在**运行间非确定性**（同种子下 cut-in 样本 11 vs 12）→ 事件类 KPI 有噪声，
Gate 4 需记录；候选修复：事件脚本改为纯步数驱动。

## 15. 现状与证据索引（2026-09-25）

- **管线状态**：全链路端到端跑通（场景生成 → 专家数据 → Stage A WM → Stage B BC → Stage C PPO → 评测/KPI/监控），
  88 项测试通过；所有 MetaDrive 运行经 `tools/venv-python`（glvnd 本地解包 + spawn 守卫）。
- **性能现状**：开环模仿已接近专家（动作 `mu_ds` 3.499 m vs 专家 3.500 m；轨迹 MAE 0.633 m；WM ADE 1.647 vs
  匀速 3.269），但闭环 KPI 落后规则基线（success 0.26–0.40 vs 0.82；off-road 0.60–0.72 vs 0.06）。
- **主要缺口**：横向车道保持（BC 复合误差，失败终止以 `out_of_road` 为主，发生在途中）；闭环执行链（LQR 跟踪）
  与 critic 解释力（`explained_var ≈ 0`）为次要缺口。
- **Stage C 观察（v2，300 updates）**：RL 修好了低速吸引子（探针 0–1 m/s 档 ds 0.19 → 5.7 m、告警解除；
  评测速度比 0.364 → 0.631），但**未带来 KPI 增益**：success 0.26 → 0.18、碰撞 0.00 → 0.14（exact 口径更极端：
  0.4 → 0.1 / 0.2 → 0.5 / 速度比 → 1.093）。训练期出现"加速+驶出道路"的奖励 hack 期（off-road/步 → −1.28、
  总奖励转负），根因：横向弱点 × 速度项主导的稠密奖励 × critic 解释力不足（`explained_var` ≈ 0.004）。
  修复方向：速度项按在道状态门控、提高 off-road/crash 权重、KL 系数下限、critic 强化、先修横向弱点。
- **详细数字与命令**：`README.md` §2–§3、`docs/experiments.md`（运行清单/数据集/各阶段/消融/口径）。
