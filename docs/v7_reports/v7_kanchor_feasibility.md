# K-anchor 可行性探针报告（v7 / Q3 轨迹聚类 anchor + 选择头）

- **日期**：2026-10-05（CPU，只读数据；未改 repo）
- **数据**：
  - 主：`datasets/BTC20261002-2329_v7p1dagger_w{1..4}`（DAgger v7p1，学生 roll-in + 专家空问标签，全部为失败窗口）
  - 参照：`datasets/BTC20261002-0941_expert5k_v41`（专家 5k 完整 rollout，schema v4）
- **脚本**（均在 `/tmp/opencode`）：
  - `kanchor_probe.py` → `kanchor_results.json` / `kanchor_probe.log`（全量聚类、几何分组、覆盖、稳定性、PCA）
  - `kanchor_shape.py` → `kanchor_shape_results.json` / `kanchor_shape.log`（形状归一化锚、覆盖、边界 margin）
  - `kanchor_lane.py` → `kanchor_lane_results.json`（车道系修正对比）
- **口径**：聚类只使用 `train_weight>0` 且 6 步链全有限的样本；物理重建误差（ADE/FDE，米）用"簇内物理均值原型"计算；silhouette 在 ≤6000 随机子样上算（全量 CH/DB）；分组统计（覆盖）按 episode 去重（每窗口取失败前最后一帧）以避免窗口内自相关。
- **重要数据事实**：DAgger 的 `action[0:6]` = 策略帧 t..t+5 的专家空问标签（counterfactual），w1 约 25% 行链尾 NaN（被本探针剔除）；expert5k 的 `action[0:6]` 是专家自身轨迹的 6 步链，全有限。

---

## 0. 结论（TL;DR）

| 问题 | 结论 |
|---|---|
| K-anchor 在我们数据上是否可行 | **可行（数据/栈层面）**，但必须**因子化**：形状锚 + 速度头。直接对 12 维 `(ds,dθ)` 链做 K-means **不可取**——expert 路径 95.4% 方差在第 1 主成分（速度），K 会被速度档位吃掉，簇退化为"快/中/慢"而非"直行/左转/右转/绕行"。 |
| K 取多少 | **形状 K=6（推荐）**，可选 8。K=6 的形状簇：直行 77.1% + 左弯、右弯、左转回正、S 弯、缓右 5 个模式（各 2.5–7.1%）；K=8 只是把弯道拆得更细，silhouette 不再提升（0.75→0.74）。若坚持联合 12 维锚，需 K≈12–16 且按速度分档。 |
| 簇是否可分（多模态证据） | **是，但仅对"形状"可分**：expert 形状锚 silhouette 0.72–0.77（强），且簇与 nav 命令强相关（Cramér's V=0.57–0.60，NMI=0.23）；dagger 失败窗口 silhouette 0.32–0.45（噪声大）。失败窗口在形状锚空间与同几何 expert 分布 JSD：**tollgate 0.32 最高**、merge 0.29、uturn 0.18、roundabout 0.18、**curve 0.10 最低**。 |
| 失败样本是否落入特定簇 | **tollgate/merge 是**：tollgate 失败窗口 64% 落在"左转回正"（end-heading +17°）簇，而 expert tollgate 只有 4% 用该模式（72% 直行）；merge 失败 43% 同簇。**curve 不是**（分布接近 expert，失败更像跟踪/速度问题而非模式选择）。 |
| 车道对齐是否可行 | **可行且几乎零成本**：`ld` 通道已含 `heading_rel`（车道航向−自车航向）+ `curvature`，`lane_lat` 给出 t0 横向偏差，`route_world` 可离线重建路线系。修正估计下对齐对 expert 有小幅稳定收益（K=8 ADE 0.532→0.507，sil +0.02~0.03），对失败窗口中性；**dθ-profile 锚本身帧不变**，车道对齐主要用于横向起点与锚→自车系回投。 |
| 选择头 PPO 兼容性 | **方案 A（soft-mixture 锚 + 连续残差）零改 PPO**：保持 2 维高斯动作空间，`plan = Σ softmax(z/τ)·A_k + Δ`；锚 logits 用 BC/CE 训练，τ 退火到硬选择。方案 B（硬 categorical + 残差混合分布）需改 `sample_action`/`logprob_from_action`/entropy/KL 与 rollout 存储，风险中等。跟踪器接口（`plan (6,2)`）两方案都不变。 |
| 最大风险 | ① 锚是**表示**不是修复：tollgate 0/45 可能同时是跟踪/地图几何问题，锚本身不保证解决；② 速度必须由连续头（现有 `action_mu` ds）或分档头承担；③ 选择头从 latent 的**可学性尚未实测**（建议下一步 latent 线性探针 / 先 BC anchor head 看 plan 尾链 ADE）；④ 现有专家锚来自 pure_pursuit，换教师（IDM/RL）需重拟合。 |

---

## 1. 数据与特征

### 1.1 样本量

| 数据集 | 可训行（tw>0 且链完整） | episode（round,spec） | 几何覆盖 | 终止 |
|---|---|---|---|---|
| DAgger w1..w4 | **60,005**（w1..w4 = 15,017/14,981/15,001/15,006） | **4,023** | 11 类，tollgate 8,158 / curve 7,748 / roundabout 6,845 / uturn 6,316 / merge 5,487 … | out_of_road 54,696（91.2%）/ collision 5,309（8.8%） |
| expert5k_v41 | **265,742** | 4,999 | 11 类，每类 20.6k–31.3k | 成功/失败混合（训练场景全量 rollout） |

- DAgger 四轮 spec 有重叠（两两 279–318 个 spec），但每轮是不同策略快照的 roll-in，按 `(round, spec)` 计 4,023 个独立失败窗口。
- w1 有 5,030/20,071 行 `action` 链含 NaN（episode 尾），本探针剔除后剩余 15,017 行；这是锚训练必须处理的 mask（或只取完整链行）。

### 1.2 特征表示

| 名称 | 构造 | 用途 |
|---|---|---|
| `chain_z` | 12 维 `(ds,dθ)×6`，逐维 z-score | 题面要求的"动作链 K-means"基线 |
| `path_ego` | 链积分成 6 个端点位姿（t0 自车系，前向 Euler） | 物理锚（米），LQR 可直接用 |
| `path_lane` | `R(−Δψ)·path_ego`，Δψ = 最近当前车道点 `heading_rel` | 车道对齐锚 |
| `cumdtheta` | 6 维累积航向 `Σdθ` | **形状锚（推荐）**，帧不变、与速度解耦 |
| `unit_path` | `path_lane / max(路径长,1m)` | 形状锚备选（含横向形状） |
| `path_frenet` | `path_lane + (0, −dy_slot0)` | 车道中心系（探针中与 lane 差异小，未作为主口径） |

Δψ 估计：`ld` 槽位 0（当前车道最近点，5 m）的 `heading_rel`；若需严格 t0 车道航向，应做 `heading_rel(slot0) − curvature(slot0)·5m` 的曲率修正（本探针未做，属已知小偏差，tight curve 上可达数度）。

### 1.3 指标口径

- **inertia / var_explained**：特征空间内簇内平方和；**silhouette**（子样）/**CH**/**DB**：常规内部指标。
- **ADE/FDE（米）**：最近簇原型的 6 点平均/末点欧氏误差，原型=簇内物理均值（避免 z-score 空间不可解释）。
- **JSD**：失败窗口 vs expert 在同一几何内、同一锚集合上的分配分布 Jensen-Shannon 散度（0=同分布；样本量 ≥20/≥100 才计）。
- **Cramér's V / NMI**：簇标签与几何/终止/nav 命令/多标签的关联强度（按 episode 去重后）。
- **margin**：样本到最近与次近锚的距离差（形状空间），度量"选择头目标是否模糊"。

---

## 2. 聚类结果

### 2.1 全量（题面口径）

**DAgger 失败窗口（60,005 行）**——`path_ego` 与 `chain_z`：

| 表示 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 |
|---|---|---|---|---|---|---|
| `chain_z` sil / ADE* | .210/.441 | .202/.371 | .190/.367 | .170/.358 | .177/.332 | .190/.324 |
| `path_ego` sil / ADE(m) | .320/1.746 | .338/1.503 | .309/1.387 | .306/1.276 | .325/1.170 | .318/1.096 |
| `path_ego` var_expl | .549 | .668 | .722 | .760 | .800 | .821 |

\* chain 空间 ADE 为 ds(m) 与 dθ(rad) 混合量纲，只用于横向比较。

**expert5k（120k 子样）**：

| 表示 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 |
|---|---|---|---|---|---|---|
| `chain_z` sil / ADE* | .533/.406 | .553/.397 | .528/.306 | .515/.294 | .523/.296 | .531/.297 |
| `path_ego` sil / ADE(m) | .686/.870 | .657/.732 | .619/.649 | .628/.599 | .619/.553 | .626/.532 |
| `path_lane`(修正) sil / ADE | .707/.842 | .679/.687 | .653/.614 | .636/.568 | .649/.541 | .623/.507 |

**基线对照（expert 子样，路径 ADE/FDE，米）**：

| 基线 | ADE | FDE |
|---|---|---|
| 全局均值路径（K=1） | 3.091 | 5.311 |
| 首步动作重复外推 | 0.726 | 1.795 |
| K=8 联合路径锚（`path_ego`） | **0.532** | **0.901** |

→ 联合路径锚比"重复首步"好 27%，但注意 **expert 路径 95.4% 方差在 PC1（速度）**，簇几乎按速度切分（见 §2.5）。DAgger 上 K=8 锚 ADE 1.096 反而差于"重复首步"0.969——失败窗口的路径方差太大且非高斯，联合锚不划算。

### 2.2 形状归一化锚（推荐表示）

| 表示 | 数据集 | K=3 | K=4 | K=5 | K=6 | K=7 | K=8 |
|---|---|---|---|---|---|---|---|
| `cumdtheta` sil | expert | **.771** | .725 | .730 | .746 | .753 | .744 |
| `cumdtheta` sil | dagger | .447 | .449 | .402 | .379 | .347 | .320 |
| `unit_path` sil | expert | .681 | .699 | **.706** | .661 | .656 | .660 |
| `unit_path` sil | dagger | .423 | .400 | .388 | .341 | .317 | .328 |

**expert `cumdtheta` K=6 簇语义**（120k 子样；与 §3 覆盖分析同一拟合；nav L/R/F = 左/右/直行命令占比）：

| 簇 | n (%) | ds0 中位 (m) | 末端航向 (°) | nav L/R/F | 主要几何 | 语义 |
|---|---|---|---|---|---|---|
| C0 | 92,519 (77.1%) | 4.16 | ~0 | .07/.09/.84 | ramp_in/out | 直行（多速度） |
| C1 | 7,823 (6.5%) | 3.63 | −19.5 | .02/**.65**/.33 | curve/t_intersection | 右弯 |
| C2 | 3,012 (2.5%) | 3.52 | +17.4 | .27/.12/.61 | **tollgate**/roundabout | 左转回正 |
| C3 | 8,461 (7.1%) | 3.68 | −9.3 | .04/.35/.61 | curve/ramp_in | 缓右弯 |
| C4 | 4,515 (3.8%) | 3.80 | +16.5 | **.45**/.07/.48 | curve/tollgate | 左弯 |
| C5 | 3,670 (3.1%) | 3.59 | +2.8（峰 +10.4） | .08/.16/.77 | **roundabout**/ramp_in | 先左后回正（环岛） |

- **簇与 nav 命令强相关**：`cumdtheta` K=6 的 Cramér's V=0.571、NMI=0.226；K=8 V=0.596。→ 选择头有**可观测信号**（nav 已在 obs）。
- K=8 时直行簇约 76.8%（扫描拟合 92,158/120k），7 个模式簇各 1.4–5.2%：把弯道/环岛拆得更细，silhouette 不升（0.746→0.744），**K=6 足够**。
- 边界模糊度：expert 只有 6.4% 样本 margin<0.05 rad（≈2.9°）；dagger 13.3%——失败窗口更常处于两模式边界（多模态决策点），但非主流。
- **形状可重建性**（给定真实 ds 剖面，用 K=8 `cumdtheta` 锚重建路径）：expert ADE **0.102 m**（直行簇 0.041；各弯道簇 0.20–0.48），dagger 0.449 m。→ **航向模式低维、可分**；剩余误差主要来自速度（ds）。

### 2.3 按几何分组（K=6，`path_lane`，均值槽位估计）

| 几何 | nE / nD | expert sil / ADE | dagger sil / ADE |
|---|---|---|---|
| curve | 27,352 / 7,748 | .591 / 0.750 | .333 / 1.298 |
| intersection | 21,094 / 6,535 | .664 / 0.492 | .359 / 1.370 |
| merge | 26,270 / 5,487 | .533 / 0.763 | .384 / 1.567 |
| roundabout | 21,136 / 6,845 | .626 / 0.504 | .323 / 1.322 |
| tollgate | 21,625 / 8,158 | .585 / 0.787 | .364 / 1.294 |
| uturn | 20,576 / 6,316 | .633 / 0.625 | .342 / 1.303 |
| straight | 26,244 / 4,153 | .635 / 0.653 | .314 / 1.241 |
| t_intersection | 20,928 / 5,183 | .626 / 0.620 | .346 / 1.401 |

- expert 每个几何内部 K=6 都稳定可分（sil 0.53–0.68）；dagger 每几何 sil 0.31–0.39（K=3..8 单调缓降，见 JSON）。
- 结论：**场景内锚可分，不需要跨场景强行共享**；但跨场景锚语义不同（同簇号在不同几何含义不同），选择头应吃场景/导航条件。

### 2.4 去重与稳定性

- 每 episode 取 1 行（失败前最后一帧，2,000 窗口）后：`chain_z` K=6 sil **0.265**/ADE 0.330；`path_lane` K=3 sil 0.489/ADE 1.865。→ 全量 silhouette 被窗口内自相关抬高/压低，但结论不变。
- 四轮 DAgger 间锚稳定：K=6 锚分配 JSD **0.012–0.034**，Hungarian 匹配后质心平均距离 **0.46–0.92 m**。
- expert 锚 vs DAgger 锚：分配 JSD 0.126，匹配质心距离 2.57 m。→ 失败窗口的专家标签分布与 expert 自身分布**有实质差异**（这正是选择头要判别的）。

### 2.5 为什么不能直接对 12 维链做 K-means

| 特征 | n90（90% 方差所需维数） | 说明 |
|---|---|---|
| expert `path_lane` | **1**（PC1=95.4%） | 路径≈"走了多远"，形状是残差 |
| dagger `path_lane` | 2（PC1=55.8%, PC2=41.6%） | 失败窗口更散但仍 2 维主导 |
| expert `chain_z` | 6 | ds/dθ 混合，6 维有效 |
| dagger `chain_z` | 5 | 同上 |

- 用首步动作（2 维）线性预测后续步：ds 的 R² = 0.91/0.78/0.68/0.60/0.54（可预测），**dθ 的 R² = 0.095/0.025/0.019/0.003/0.000（不可预测）**。
- → **速度剖面可由现有连续头外推；真正的多模态信息在后续步的航向（形状）**。这正是"形状锚 + 速度头"因子化设计的依据。

---

## 3. 覆盖性检查（失败样本 vs 场景/终止）

### 3.1 簇 × 几何 / 终止（联合路径锚，K=6，episode 去重）

- 簇 × 几何：Cramér's V=0.187（p≈2e-115）；簇 × 终止：**V=0.394**（p≈2e-115）→ 失败类型与锚相关，几何关联弱。
- 簇 × 多标签 NMI：car_following 0.107、near_intersection 0.075、crowded 0.070、on_curve 0.065，其余 ≤0.06。
- 联合路径锚按速度切分，因此"几何覆盖"被速度掩盖；形状锚下见 §3.2（Vnav 0.57）。

### 3.2 失败窗口 vs expert 同几何分布（形状锚 `cumdtheta` K=6）

| 几何 | JSD | 失败窗口分布 D（C0..C5） | expert 分布 E |
|---|---|---|---|
| **tollgate** | **0.324** | [.07,.14,**.64**,.06,.06,.02] | [.72,.08,.04,.07,.05,.04] |
| **merge** | **0.290** | [.11,.19,**.43**,.07,.15,.05] | [.78,.07,.03,.06,.03,.03] |
| uturn | 0.179 | [.18,.28,.10,.18,.11,.15] | [.75,.08,.02,.07,.05,.04] |
| roundabout | 0.178 | [.19,.23,.25,.12,.06,.16] | [.74,.07,.04,.07,.03,.04] |
| **curve** | **0.099** | [.25,.32,.09,.19,.09,.06] | [.67,.10,.03,.12,.06,.03] |

（K=8 同序：tollgate 0.349 > merge 0.322 > roundabout 0.228 > uturn 0.219 > curve 0.160。）

### 3.3 逐场景结论（对照 eval500：tollgate 0/45、curve 0.28、merge 0.52、roundabout 0.10）

- **tollgate**：失败窗口 64% 落在 C2"左转回正"（末端 +17°），expert tollgate 只有 4% 用该模式。**模式级偏移明确**——学生到闸口附近发生了专家不会做的转向/避让行为。锚+选择头**具备表达正确模式（直行）的能力**，但能否学会取决于 obs 中障碍/闸口上下文与训练信号；若 0/45 的根因是跟踪/地图几何（3 m/s 区 + 护栏），锚本身不能修复。
- **merge**：失败 43% 落在 C2、19% C1、15% C4——分布右偏且多模式；锚空间可分，适合选择头 + RL 学"让行/汇入"模式。
- **roundabout**：失败分布相对均匀（C2 25%/C1 23%/C0 19%/C5 16%），JSD 中等；C5（先左后回正）是环岛专家模式，失败窗口也有 16% 落入——可解释为"进环/出环选择错误"。
- **uturn**：JSD 0.18，失败分散；样本里 dagger 的 `cumdtheta` 簇末端航向出现 −54°/+62°/+124° 的极端模式（expert 无），说明失败窗口包含"乱打方向"状态。
- **curve**：JSD 最低（0.10），失败窗口形状分布接近 expert（C0 25% vs 67%，更多落在 C1 32%）。→ **curve 失败不是模式选择问题**（更可能是速度/跟踪/压线），K-anchor 对该项的预期收益低；README 中 curve 的 off-road 0.61 与压线 0.61 也支持这一判断。

### 3.4 多模态证据小结

1. 形状锚 expert silhouette 0.72–0.77、K=6 与 nav 命令 V=0.57：**形状模式真实存在且可从观测预测**。
2. 失败窗口与 expert 的锚分布 JSD 在 tollgate/merge 显著（0.29–0.32）：**失败对应特定模式的错选/错用**。
3. 13.3% 失败行处于锚边界（margin<0.05 rad）：存在真正的**多模态决策点**（让行/绕行/汇入），选择头有存在意义。
4. 速度维度上失败与 expert 也分离（路径锚 JSD 0.14–0.47，簇×终止 V=0.39），但速度更适合连续头处理。

---

## 4. 坐标系检查

### 4.1 现有坐标资产

| 资产 | 内容 | 结论 |
|---|---|---|
| 全部 obs 通道 | ego 系（SE(2) 对齐到 t0 自车，x 前向/y 左向） | 无车道系轨迹 |
| `ld (16,7)` | 车道线点 `[dx,dy,heading_rel,curvature,speed_limit,left_type,right_type]`；槽位 0–4 = 当前车道 5/10/15/20/30 m（优先填充） | **有 `heading_rel` 与 `curvature`** → 可构造 t0 车道系 |
| `lane_lat`（数据） | t0 相对车道中心横向偏差（m） | t0 横向起点可用 |
| `route_world (64,2)` + `ego_world` | 世界系路线折线（road 中心，block 级粗） + 世界位姿 | 可离线重建路线系（粗；转角处不可靠） |
| `nav (11)` | ego 系 checkpoint×2 + 命令 one-hot + route_completion | 命令是形状锚的强条件变量 |

### 4.2 车道对齐收益（修正估计：slot0 heading_rel）

- Δψ 可用率：**100%**（每行至少一个当前车道槽有效）；|Δψ|>5°：expert 13.6%（p90 5.9°），dagger 25.0%（p90 7.4°）。
- `d0`（由 slot0 的 dy 取负）与 `lane_lat` 相关 **−0.868**（dagger）→ 车道横向起点可靠（符号为左正 vs MetaDrive local lat 右正）。
- 聚类对比（K=3..8）：

| 数据 | 指标 K=3 | K=6 | K=8 |
|---|---|---|---|
| expert `path_ego` sil / ADE | .686/0.870 | .628/0.599 | .626/0.532 |
| expert `path_lane` sil / ADE | **.707**/0.842 | **.636**/**0.568** | .623/**0.507** |
| dagger `path_ego` sil / ADE | .320/1.746 | .306/1.276 | .318/1.096 |
| dagger `path_lane` sil / ADE | .306/1.760 | .315/1.282 | .311/1.129 |

- **expert 上车道对齐小幅稳定更优**（sil +0.02~0.03，ADE −3~−5%）；**dagger 上中性**（失败窗口噪声淹没收益）。

### 4.3 结论

1. **车道系可行、无需重采数据**：`ld.heading_rel/curvature` + `lane_lat` 已足够；建议严格实现时做 `heading_rel − κ·5m` 修正（或按 s=0 重采样）。
2. **对锚表示并非必需**：`cumdtheta`（dθ 剖面）本身**帧不变**，锚可以完全不依赖车道系；车道对齐主要服务：
   - 横向起点（`lane_lat`/d0）与"车道中心系"锚（可选）；
   - 推理时把锚从车道系回投自车系（`R(Δψ)`，Δψ 来自当帧 `ld`）——已有通道，零 schema 改动。
3. 若要做**严格 lane-relative anchor**（Frenet s/d），建议锚=(dθ 剖面, 归一化横向剖面, 起点 d0)，全部可由现有 obs 在训练/推理两端复算。

---

## 5. 与现有实现路径（改动点 + PPO 兼容性）

### 5.1 现状（关键链路）

- `net/plan_head.py::PlanHead.forward`：mem 聚合 → fusion+MoE → `latent`，输出 `ego_next`（下一 ego 特征 6 维）。**MoE 只在这里**。
- `net/model.py::DrivingModel._rollout`（L663–770）：`action0 = action_mu`，逐步用 `policy(tokens)` 迭代 6 步产出 `plan (6,2)`；`plan[:,0] == action_mu`。
- `net/policy.py::PolicyHead`：2 维 `(ds,dθ)` 高斯（sigmoid 压缩，界 `[0,10]×[−0.6,0.6]`）。
- `pipeline/trainer.py`：
  - `weighted_action_chain_loss`（L2127）已把 `plan[:,1:]` 监督到 DAgger 教师链（`action[k]`，NaN 尾 mask）；
  - PPO（`PPOTrainer` L5628）：`sample_action`（L413）/`logprob_from_action`（L466）/`gaussian_kl`/`gaussian_entropy` 均为**逐维通用**的 2 维高斯；
  - `update`（L6253）消费 `action_mu/action_logstd/value`；rollout buffer 只存连续动作。
- `pipeline/eval_runner.py::_CkptController.action`（L915–950）：`plan` → `build_eval_references`（`plan[0]` 强制=`action_mu`）→ `LqrTracker.set_reference` 插值 30 点，预瞄距离 `clip(0.8v+2.5, 3, 15) m`（≈第 2–3 步）→ **尾链直接影响转向**。
- `pipeline/stages.py`：DAgger 行按 `action_chain_source=auto` 判定逐行教师链，`weighted_action_chain_loss` 已就位。

### 5.2 三个集成方案

**方案 A（推荐）：soft-mixture 锚 + 连续残差，PPO 零改动**
- `plan = Σ_k softmax(z_k/τ)·A_k + Δ`，`A_k` 为可学习/冻结锚缓冲（K×6×2），`Δ` 为现有 policy 的残差动作链；`action_mu = plan[:,0]`（保持执行语义）。
- 锚 logits `z = anchor_head(latent)`；BC 阶段用 `anchor* = argmin_k ||chain − A_k||`（NaN 尾按有效步加权）做 CE；τ 从 1.0 退火到 ~0.1，推理可硬取 argmax。
- PPO 只对 `Δ`（或直接对 `plan`）算高斯 logprob，ratio/KL/entropy 全复用现有函数；**改动 = 新头 + 前向里一次混合**，PPO 代码不动。
- 风险：软混合下梯度会同时压所有锚（早期坍缩到少数锚）；用 CE 预训练 + 负载均衡正则 + τ 退火缓解。

**方案 B（硬选择，若需要 categorical）**：混合分布 `p(k)·N(Δ)`：
- `sample_action`/`logprob_from_action` 需返回 `logprob = log p(k) + log N(Δ)`；entropy/KL 需分解相加；
- rollout buffer 需存 `anchor_idx`（或存 `log p(k)`）；
- 兼容性可行但触及 4–5 个函数 + 缓冲 schema；建议先 A 后 B。

**方案 C（最保守，仅辅助）**：锚只作为 plan 的替代输出 + CE 辅助 loss（不训练 categorical 的 RL），执行仍走 `action_mu`。由于 LQR 跟踪整条 `plan`，尾链换锚仍会影响闭环，但 RL 不优化选择——只适合先验证表示收益。

### 5.3 具体改动点

| 文件 | 改动 |
|---|---|
| `net/plan_head.py` | `PlanHead.forward` 增加 `self.anchor_head = nn.Linear(H, K)`，把 `anchor_logits` 放进 `moe_aux`（或返回三元组）；注册 `anchors` buffer（K,6,2） |
| `net/model.py::_rollout` | 在 step (a) 用锚混合替换/叠加 `action0`：`action = mix(anchor_logits) + residual`；`plan[:,0]` 与 `action_mu` 的一致性保持（回投/裁剪） |
| `net/model.py::forward` | cheap path 也输出 `anchor_logits`（PPO/BC 都用） |
| `net/policy.py` | 方案 A 不需要改；方案 B 增加 `Categorical` 头与 `log_prob/entropy` 分解 |
| `pipeline/trainer.py` | ① 新增 `anchor_ce_loss(anchor_logits, chain, valid)`，与 `weighted_action_chain_loss` 同批；② 方案 A 不改 PPO；方案 B 扩 `sample_action/logprob_from_action/gaussian_*`；③ 监控 `plan` 尾链 ADE/FDE（已有 bc_traj_* 口径可复用） |
| `pipeline/eval_runner.py` | 无接口改动（`plan (6,2)` 不变）；建议加 `--eval-anchor` 诊断开关（记录每步选中锚 id） |
| 数据侧 | 用 `expert5k` 拟合锚（干净），DAgger 只做 CE 验证；锚目标可从现有 `action[0:6]` 零成本生成（NaN 尾 mask） |

### 5.4 与现有 loss 的关系

- `weighted_action_chain_loss` 已监督 `plan[1:]` 的多步链——**anchor CE 是它的离散对偶**（把"回归到某条链"变成"选择到某个模式"）；两者可并联（CE 1.0 + chain 0.3 起调）。
- KL 锚（Stage C `--kl-anchor-coef`）是参数空间锚，与 K-anchor 不冲突；K-anchor 是**动作表示层**锚。

---

## 6. 风险、限制与建议下一步

**限制（本探针）**
1. 只做离线聚类/统计，**未跑模型前向**：选择头从 `latent` 的可学性未实测。
2. 失败覆盖是"失败窗口分布 vs expert 分布"的几何级比较，非逐 episode 归因；DAgger 窗口本身全是失败样本。
3. `path_lane` 分组表用了均值槽位 Δψ（未做曲率修正），dagger 的 Δψ 会被高估；§4 已用修正估计复核，结论方向不变。
4. 锚来自 pure_pursuit 专家（IDM/RL 教师可能形状不同）。
5. 所有 K-means 是欧氏 + 一次性拟合，未做加权/平衡（straight 簇占 78%）。

**建议下一步（按性价比排序）**
1. **latent 线性探针**：取 Stage-B ckpt，对 expert5k 抽 `latent`，用线性/logistic 预测 `cumdtheta` K=6 锚标签 → 直接量化选择头上限（1 次前向，成本低）。
2. **BC anchor head 消融**：在现有 phase3 上加 CE 辅助（方案 C），只测 `plan` 尾链 ADE/FDE 与 eval500 是否改善（尤其 tollgate/merge），不动 PPO。
3. 若 ① 显示可学且 ② 有增益，再做方案 A 的 soft-mixture PPO（KL 锚照旧）。
4. 锚拟合规范：expert5k 全量、按几何平衡采样、K=6；速度维度交给连续头（或 ds 分 3 档 × 形状 6 = 18 个联合锚的对照臂）。
5. tollgate 专项：把"锚选择"与"跟踪/地图"分离验证（例如强制直行锚 vs 自由选择），确认 0/45 是否行为问题。

---

## 7. 复现

```bash
cd /workspace/01_Proj/DRL_PathPlan
.venv/bin/python /tmp/opencode/kanchor_probe.py   # ~12 min（全量/分组/覆盖/稳定性/PCA）
.venv/bin/python /tmp/opencode/kanchor_shape.py   # ~3 min（形状锚 + 覆盖 + margin）
.venv/bin/python /tmp/opencode/kanchor_lane.py    # ~2 min（修正车道系对比）
```

产物：`/tmp/opencode/kanchor_results.json`、`kanchor_shape_results.json`、`kanchor_lane_results.json` 与对应 `.log`。

---

## 附录 A：关键数字速查

- expert 路径 PC1 方差占比 **95.4%**；首步动作对后续 dθ 的 R² ≤ 0.095。
- expert `cumdtheta` K=6 簇大小（120k，覆盖分析同拟合）：[92,519 / 7,823 / 3,012 / 8,461 / 4,515 / 3,670]；K=8 直行约 76.8%。
- expert 形状锚 silhouette：`cumdtheta` 0.72–0.77；`unit_path` 0.66–0.71。
- 形状重建（真 ds）：expert ADE 0.102 m（直行 0.041）；dagger 0.449 m。
- 因子化 vs 联合：`unit_path`×恒速 K=6 expert ADE 0.423 m vs 联合路径锚 K=6 0.599 m（path_ego）。
- 覆盖 JSD（`cumdtheta` K=6）：tollgate 0.324 > merge 0.290 > uturn 0.179 ≈ roundabout 0.178 > curve 0.099。
- tollgate 失败形状分布：[.07,.14,.64,.06,.06,.02] vs expert [.72,.08,.04,.07,.05,.04]。
- 簇×终止 V=0.394（p≈2e-115）；簇×nav V=0.571（形状锚）；簇×几何 V≈0.19–0.23。
- 跨轮稳定性：分配 JSD ≤0.034；质心漂移 ≤0.92 m。
- Δψ：可用率 100%；|Δψ|>5° expert 13.6% / dagger 25.0%；`d0` vs `lane_lat` r=−0.868。
- 车道对齐收益（expert，修正 Δψ）：K=8 ADE 0.532→0.507（−4.7%），sil 0.626→0.623（K=8）/0.628→0.636（K=6）。
