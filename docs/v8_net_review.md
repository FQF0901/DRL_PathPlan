# v8 网络架构与问题综述（供外部研究/评审）

> 生成：2026-10-09 · 代码基线：`68115bc`（工作区含 v8b 参数排摸临时改动，排摸进行中）
> 用途：交给外部研究型模型（ChatGPT 等）做 research & review。**不是论文/设计愿景**，而是"代码现状 + 实测证据 + 已知问题"的忠实快照。
> 阅读约定：**[已核实]** = 有代码/实测依据（给 `path:line`）；**[假设]** = 我们的机制解释，可反驳；**[待验证]** = 进行中/证据不足；**[已知偏差]** = 代码与文档/约定的不一致点。
> 仓库内文档已于同日按代码现状重写（env/net/pipeline/reward_model/tools/tests 等 README），并经过独立核查子任务对账。

## 0.1 Abstract (EN)

This document describes a MetaDrive-based closed-loop planning system: a learned latent world model (ST-GNN with autoregressive latent rollouts) plus a Mixture-of-Experts (MoE) planner trained by imitation learning (IL), iterated DAgger-lite, and PPO fine-tuning. We summarize the observation design (obs v6), module-level parameter allocation (≈1.28M params), supervision/loss composition per stage, and the evaluation protocol, followed by empirical findings: (i) a reproducible **capacity–robustness inversion** — larger policy-side capacity improves offline fit but degrades closed-loop success; (ii) **off-road excursions as the dominant closed-loop failure mode** (episode-level off-road rates ≈0.49–0.96; 68–97% of failures across evaluated arms); (iii) **non-monotone capacity optima** (MoE primary width appears to peak near 512, with 128 and 768 worse); (iv) a parameter budget dominated by the plan head + MoE (~42%) and the ST-GNN (~30%). We list known implementation caveats and open research questions (mechanism explanations, experiment design, evaluation statistics) where external review is explicitly requested.

---

## 1. 任务与训练范式

- **任务**：MetaDrive 城市驾驶场景中的闭环路径规划。策略步长 0.5 s（2 Hz），物理步长 0.1 s（10 Hz），每策略步决策一次、下 5 个物理步执行（`config/env.yaml:16-17`）。
- **观测**：自车/对象/车道/导航/信号/静态几何的稠密表征（见 §2），无图像输入（纯状态观测）。
- **决策输出**：动作 `(a_lon, a_lat)`（归一化 [-1,1]），另有价值、世界模型（WM）预测头（见 §4）。
- **训练范式（四段）**：
  1. **Stage A**：WM 教师强制预训练（预测未来 6 帧各实体状态/存在性）；除 policy/value 外全部可训、MoE 关闭；
  2. **Stage B**：planner 行为克隆（BC）——先 primary 段、后 specific 段（MoE 开）；
  3. **phase3**：DAgger-lite 迭代（学生 roll-in + 规则专家标注），配冻结配方与守护；
  4. **Stage C**：PPO 微调（WM 全期冻结），KL 锚约束不偏离 BC 太远。
- **数据**：规则专家 `PurePursuitIDMPolicy`（确定性）采集 `expert5k`（当前数据集 `BTC20261007-2202_expert5k_v8`，训练 360,497 行 / 500 episodes 验证）。
- **评测**：闭环评测在冻结 val spec 集（eval500：500 场景）上，用 LQR tracker 跟踪模型 plan（`--eval-reference plan`）；KPI 12 项（§5.2）。离线指标 = val BC loss / action err / traj MAE。

**[假设]** 该系统的核心研究问题是：**在世界模型可学习的前提下，如何分配模型容量与控制模型行为，使闭环规划在分布外（OOD）场景下仍然稳健**。我们的经验是"离线拟合越好 ≠ 闭环越好"，且该反转在多个模块上重复出现（§6.1）。

---

## 2. 观测接口（obs v6）

**通道表 [已核实]**（`env/obs/builder.py:3-19`，实机 `feature_spec()` 输出）：

| 通道 | 形状 | 内容 |
|---|---|---|
| `ego` | (1, 8) | 速度/加速度/朝向误差等 + 末 2 维 reserved（采集侧写入历史动作，`net/encoders.py:8-10`） |
| `od` | (16, 9) | 周边对象：固定槽位（槽=track id，`env/obs/od.py:87-97`），含 id/presence/位置/速度等 |
| `ld` | (16, 7) | 车道线采样点：当前车道占满 5 个 offset，其余车道按"环优先"（`env/obs/ld.py:209-261`） |
| `nav` | (1, 11) | 导航（路口命令 + 检查点） |
| `signal` | (1, 4) | 信号灯（当前恒 [0,0,0,1]） |
| `others` | (1, 21+K) | 汇总向量（K=12 道路类别几何），实际 33 维 |
| `ego_world` | (1, 3) | 自车世界坐标 |
| `route_world` | (64, 2) | 全局路线折线（末点重复、mask=0） |

- **历史**：6 帧 @0.5 s（5 个间隔、≈2.5 s 跨度），SE(2) 对齐当前帧，`hist_valid` 标记有效性（`env/obs/memory.py`）。
- **scope**：前 150 / 后 50 / 左 25 / 右 25 m（`env/obs/builder.py:76`）。
- **LD offsets = (5, 10, 15, 20, 30) m**（`env/obs/ld.py:76`）。**[假设]** 近场 5/10 m 点提供横向锚定（v8 诊断臂 A1' 回退 v4 近场口径的产物）。
- **obs 指纹机制 [已核实]**：`env/obs/__init__.py:37-49` 对 `env/obs/*.py` **全部文件字节**做 md5 → `v6-<12hex>`；BC 数据 meta 记录该指纹，加载时比对（不一致 = `RuntimeWarning`；比对 `pipeline/trainer.py:2956-2964`、告警 `:2967-2972`）。**副作用：任何注释级改动都会改指纹**，修复注释须同步数据集 meta（§6.2-①）。
- obs v5→v6 变更：删除 `lane`/`ttc` 上下文 token（见 §6.2 与历史文档 `docs/v7_reports/`）。

---

## 3. 网络架构（`net/`）

### 3.1 数据流（前向）[已核实]

```
obs v6（8 通道 + 6 帧历史）
  → per-frame encoders（rerank/embed，19,072 params）
  → mem_encoder（6 帧时间注意力 ×3 组：ego / od / others；136,128 params）
  → t0 帧 latent（单帧编码）+ 计划头 plan_head（fusion + MoE；531,247 params）
  → st_gnn：空间消息传递（spatial，2 层、199,424 params）
      + 潜状态自回归 rollout（transitions ×3：ego/od/ld，各 49,408 params）
      + od/ld 预测头（17.8k/17.5k）+ presence/entry
  → policy（交叉注意力 1 层 layers 66,304 + trunk 2 层 41,248；108,452 params）
  → value（同构 + net 33,281；99,969 params）
```

- **令牌集合 T=39 [已核实]**（`net/model.py:529-572`）：当前帧 + 历史/实体摘要 + 计划 token 的拼接集合。
- **latent WM 口径 [已核实]**：rollout 为 latent 自回归（不滑 raw mem、不合成 ego 特征）；detached 状态链、action/pose 可微（`net/model.py:800-801`）；`wm_detach` 为 no-op（`:861`）；`forward(rollout=False, world_model=True)` 抛 `ValueError`（防误用）；**cheap path 与完整前向逐位一致 [已核实]**（我们实测 `torch.equal=True`）。
- **[假设]** latent WM 优于 feature WM（同参数对照 +11pt 量级），我们解释为"潜状态预测提供了与规划耦合更好的想象表征"；E0 消融（§5.4）支持该方向但未定论。

### 3.2 MoE / plan head [已核实]

- E=8 experts，**top-2 软混合**；`primary`（always-on）+ `experts`（稀疏）+ `router`（无监督门控）。plan head 输出 = `primary + top2 混合 + residual`。
- **router 无监督**：v6 曾有 supervised labels（BCE），v8 已删除；`moe.router.supervised_labels` 仅用于数据集标签顺序校验（`pipeline/trainer.py`）。
- phase1（BC primary 段）MoE 关闭、phase2（specific 段）打开（`pipeline/stages.py:3310` / `:3433`；Stage A 亦关闭，`:1744`）。
- 负载均衡 aux：**phase2（Stage B specific）起生效**，权重 0.01（`config/train.yaml:178`、`pipeline/stages.py:3262`、`pipeline/trainer.py:4042-4045`）。

### 3.3 K-anchor（可选计划头）[已核实]

- `plan_anchor.enabled=false`（默认关）→ `num_anchors=0`，模型与旧版逐位一致。开启时：K=6 形状锚字典（`net/anchor.py`，由 `tools/fit_plan_anchors.py` 拟合）、`lane_ctx=zeros`、初速先验 3.5 m/策略步（0.5 s，≈7 m/s；`net/plan_head.py:35-36`）、计划头零初始化。

### 3.4 参数量分布 [已核实]（本文件生成时以 `build_model` 逐模块聚合）

**HEAD 默认配置**（primary 768 / experts 76 / router 384 / trunk 160 / net 256）= **1,278,879**：

| 模块 | 参数量 | 占比 |
|---|---|---|
| `plan_head.moe.primary` | 197,504 | 15.4% |
| `plan_head.moe.experts`（8×19,660） | 157,280 | 12.3% |
| `plan_head.fusion` | 114,944 | 9.0% |
| `plan_head.moe.router` | 52,616 | 4.1% |
| `plan_head.*`（其余：ego_next/norm） | 8,903 | 0.7% |
| **plan_head 合计** | **531,247** | **41.5%** |
| `st_gnn.spatial`（2 层） | 199,424 | 15.6% |
| `st_gnn.{ego,od,ld}_transition`（各 49,408） | 148,224 | 11.6% |
| `st_gnn.{od,ld}_head` | 35,337 | 2.8% |
| `st_gnn` 其余（step_embed/presence/entry） | 1,026 | 0.1% |
| **st_gnn 合计** | **384,011** | **30.0%** |
| `mem_encoder`（3 组注意力） | 136,128 | 10.6% |
| `policy`（layers 66,304 + trunk 41,248 + 其余） | 108,452 | 8.5% |
| `value`（layers 66,304 + net 33,281 + 其余） | 99,969 | 7.8% |
| `encoders` | 19,072 | 1.5% |

**Arm P 对照**（primary 256）= **1,147,295**：primary 197,504→65,920（省 131,584）。其余模块同 HEAD。
**排摸臂单变量 Δ（vs Arm P）[已核实]**：primary 512 +65,792 / 128 −32,896；router 64 −43,840；experts 128 +106,912；spatial 3 层 +99,712；policy attn 2 层 +132,608（同时作用于 policy+value）；trunk 192 +8,224。

**[假设]** 参数预算的"合理性"本身是研究问题：目前 plan head+MoE 占比 42%，而闭环表现对该区的容量变化**非单调**（§5.5）；感知侧（st_gnn/mem）容量从未被单变量扫过（排摸进行中）。

---

## 4. 输出与监督

### 4.1 模型输出键 [已核实]（`net/model.py:843-910`）

`action_mu/logstd`（策略）、`value`、latent 预测（`z_*_pred`，诊断）、`od/ld/ego_next/entry/presence` 预测、`router_logits/expert_weights`、可选 `anchor_*`。

### 4.2 Stage A（WM 教师强制）[已核实]

- 目标：预测未来 6 帧（**stride=5，即 `step+5k`**，`pipeline/frames.py:344-345`；`pipeline/stages.py:827,1760`）。
- 损失权重：**latent 1.0 / od 0.1 / ld 0.02 / ego_next 0.1 / presence 0.1 / entry 0.1**（`pipeline/stages.py:1848-1875`）。
- 可训范围：除 `policy./value.` 外全部；MoE 关闭（`pipeline/stages.py:1742-1745,1794-1802`）。
- **[已知偏差]** `config/train.yaml:86 stages.A.world_model.trainable` 未被代码读取（可训范围硬编码）。

### 4.3 Stage B（planner BC）[已核实]

- 两段：`primary_phase_split=0.5`（前 10 / 后 10 epochs，`config/train.yaml:173`）；phase1 primary 可训（MoE 关）、phase2 specific 可训（MoE 开，含 experts/router）。
- 动作 BC 损失 + 轨迹辅助（`traj_aux_weight=0.1`，`config/train.yaml:92/175`，rollout traj vs 专家 traj6；WM 冻结 + detach）。
- 细粒度冻结清单：`pipeline/stages.py:2734-2765,3295-3343,3424-3443`。
- **[待验证]** phase2 在 closed-loop 上是否带来增益（v8 排摸中：在 primary 768 设置下 phase1 边界≈final，phase2 中性）。

### 4.4 phase3（DAgger-lite）[已核实]

- 迭代恢复（`--phase3 <dir>`），学生 roll-in + 专家标注；守护（safe-recipe guard）、keep-best、原子状态（`pipeline/phase3_loop.py`）。
- 损失 12 项生效默认：action 1.0 / chain 0.2 / bias 0 / ego_next 0.1 / od 0.005 / ld 0.002 / presence 0.1 / entry 0.1 / latent 0.01 / traj_aux 0 / anchor 0 / load_balance 0.01（生效值=config 覆盖后；`config/train.yaml:116-137`、`pipeline/stages.py:3957`；`pipeline/trainer.py:856-879` 为 dataclass 默认；anchor 列 = `anchor_ce`+`anchor_wta` 两项）。

### 4.5 Stage C（PPO RL）[已核实]

- PPO：clip 0.2 / γ 0.99 / λ 0.95；horizon = 200 策略步（=100 s）；`--pool local`、`envs=1`。
- **WM 全期冻结 + fail-fast**（`pipeline/stages.py:4639-4651`）；cheap path；trainable scope allowlist（`design`）。
- **KL 锚**：0.05 → 0.02 线性衰减（每 update 插值，`pipeline/stages.py:4808`）。
- 采用配方（v7 arm1，`config/arms/v7_arm1_offroad.yaml`）：奖励 = v5 bundle + `off_road_edge`（离路距离型稠密项，weight −0.5 / scale 1.0）。
- **[已核实]** 奖励系统：注册 **17 项**（comfort_jerk, comfort_jerk_win, comfort_lat, comfort_lon, crash, lane_boundary, lane_center, lead_gap, low_speed, off_road_edge, out_of_road, route_completion, solid_line, speed_deficit, speed_limit, speed_ratio, ttc）；默认启用 **10 项**；终局值机制 + CaRL 正负拆分聚合。
- **[已知偏差]** `reward_model.aggregation.default_terminal_values()` = (+30/−19/−15/−23/−5)（旧默认），与当前 arm 配方 (+29/−22/−14/−46/−5) 不一致——**未带 `stages.C.reward` 配置的 RL 运行会落旧默认值**。

---

## 5. 训练与评测协议

### 5.1 运行约定 [已核实]

- run 命名：训练 `runs/BTC<分钟戳>_<name>`（A/B 共用一个 run 根）；评测 `runs/BTC<秒级戳>_<spec kind>_<policy>`；ckpt：`ckpt_epoch{NNN}.pt`（A/B，每 5 epoch）/ `ckpt_u{NNN}.pt`（C，每 25 update）。
- `runs/`、`datasets/`、`env/specs/*.json` 均 gitignored。
- spec 隔离硬校验：DAgger 池 ∩ eval500/val = ∅（命中即 `SystemExit`）。

### 5.2 评测协议 [已核实]

- 闭环：500 场景（eval500），LQR tracker + `--eval-reference plan`；KPI 12 项：collision, offroad, min_ttc, a_lon, a_lat, jerk, speed_ratio, solid_line_crossing, speed_limit_violation, route_completion, per_category_success, overall_success。
- 统计工具：`tools/paired_eval.py`（配对翻牌/McNemar/bootstrap，主判据需 pin 单 baseline）。
- 选择集纪律：筛选用 clean150（与 eval500 不相交）→ 确认用 clean500/eval500。**[已知方法论问题]** clean150 的成功率 SE ≈ 3.5pt——单臂 ±5pt 差异在噪声内（§6.3）。

### 5.3 闭环结果矩阵（v8 诊断链，eval500 / LQR+plan）[已核实]

| 对象 | success | 说明 |
|---|---|---|
| 规则专家 IDM 基线 | **0.756** | 专家上限参照 |
| s11（v7 RL 链） | **0.646** | 唯一含 DAgger+RL 的完整链 |
| P1-B | 0.312 | 离线链（无 DAgger/RL） |
| A3（旧参数 + feature WM） | 0.258 | 旧参数分布对照 |
| **Arm P（primary 256 + 新分布其余 + latent WM）** | **0.224** | v8 参数再分配的基线 |
| 首链（v8 第一版） | 0.204 | |
| A1'（latent WM + primary 768） | 0.154 | 诊断：primary 768 回归主因 |
| Arm T（trunk 128 + primary 768） | 0.088 | trunk 回退的负收益 |
| A2（feature WM） | 0.044 | |

- **根因结论 [已核实]**：v8 闭环回归主因 = **primary 768（3×）**（同 latent WM 下 256 vs 768：0.224 vs 0.154，+7pt）；trunk 160 有益（128 → 0.088）；latent WM 有益；phase2 中性。
- **注意**：除 s11 外均为 A→B only（无 DAgger/RL），不可与 s11 直接对比；A2/A3 使用 feature WM（非当前架构）；IDM 0.756 与 P1-B 0.312 的原始 run 已随清理删除（数字来自 v7 报告记录，无法在本仓库独立复算）。

### 5.4 WM 特征消融（E0，s11 权重、eval500）[记录一致·原始产物已清理，无法独立复算]

| 消融 | 效果 |
|---|---|
| 关 `wm_obj`（对象预测） | collision −5.6pt / off_road +2.6pt |
| 关 `latent`（潜状态预测） | success −3.8pt / collision +5.8pt |
| 关 `wm_writeback`（写回） | collision −3.2pt / off_road +4.4pt |

- **[假设]** 失败在 tollgate（收费站）组高度集中，且"撞亭 ↔ 出界"互搬——提示 WM 特征主要影响**边界/几何的判读**而非全局策略；需进一步定位。

### 5.5 参数量 × epoch 排摸（v8b，进行中）[待验证]

- 协议：单变量臂 vs Arm P 基线；Stage A 20ep + Stage B 20ep（同数据）；筛选集 clean150；逐 ckpt（e005/e010/e015/e020）评测。
- 现状（success，clean150；**SE≈3.5pt，待 clean500 确认**）：

| 臂 | e005 | e010 | e015 | e020 | 结论倾向 |
|---|---|---|---|---|---|
| base（primary 256） | 0.280 | 0.147 | 0.253 | 0.193 | 基线 |
| **primary 512** | 0.180 | 0.320 | 0.307 | **0.340** | 甜点可能在 512 |
| **router 64** | 0.153 | 0.213 | **0.347** | 0.313 | 小 router 有利 |
| primary 128 | 0.260 | 0.140 | 0.187 | 0.093 | 太小不利 |
| experts 128 | 0.187 | 0.153 | 0.027 | 0.120 | 不利 |
| spatial 3 层 / attn 2 层 / trunk 192 | — | — | — | — | 运行中 |

- **初步观察**：① primary 容量对闭环**非单调**（128 差 / 256 中 / 512 好 / 768 已知差）；② 最优 epoch 依赖配置（base 峰在 e005；512/rou64 的峰在 e015–e020，其中 rou64 e020 回落 0.347→0.313）——**容量与训练步数存在交互**；③ 离线 val 指标无法区分（pri128 离线好、闭环最差）——offline↔online 反转再现。

---

## 6. 存在的问题与难点

### 6.1 机制层（最需要外部 review）

1. **容量–鲁棒性反转 [已核实现象 / 假设解释]**：多个模块上重复出现"离线拟合↑ → 闭环↓"（primary 768 / primary 128 的 val action err 优于基线而闭环显著更差；trunk 128 的 val bc_loss 更优但 val action err 也劣——指标依赖）。**[假设]** 大容量策略更锐利、更贴合专家分布，复合误差下鲁棒性差；也可能与优化/初始化尺度有关。**我们无法区分**：(a) 纯容量-鲁棒性权衡；(b) 训练动态（大模块收敛更快/更慢）；(c) 表征与控制的耦合问题。
2. **off-road 主导失败 [已核实]**：off-road 是 episode 级主失败（clean150 四臂 0.49–0.83、exp128 e015 达 0.96；v8 A→B 臂 0.54–0.85），占失败份额 68–97%；collision 仅 0.007–0.05（个别臂 0.06）。失败集中在 tollgate/几何复杂场景，且"撞亭↔出界"互搬（E0）。**[待验证]** 是观测分辨率/WM 几何判读问题、控制振荡问题、还是奖励塑形问题。
3. **容量分配缺乏原则 [假设]**：1.28M 参数中 plan head+MoE ~42%、st_gnn ~30%；但单变量结果显示"更多容量≠更好"，而且各轴最优值非单调。需要一个**容量分配的判据**（信息瓶颈？谱分析？副任务探针？）。
4. **WM 与闭环的因果链不清 [待验证]**：latent WM 有帮助（+11pt 量级）、no_latent −3.8pt；但 WM 的离线 ADE/FDE 与闭环表现的关系未被建立（"更好的预测"不等于"更好的规划"）。
5. **专家数据质量与上限 [已核实]**：专家（IDM）自身在评测集上 success≈0.756；BC 目标分布的上限与偏置未知（expert 在 tollgate 也常失败）。DAgger/RL 只在 v7 链完整跑过（s11=0.646）；v8 尚未跑 DAgger/RL。
6. **epoch/step 与容量的交互 [待验证]**：见 §5.5 ②——"训练多久"和"多大容量"在闭环上耦合，现有实验无法分离。
7. **单 seed 谨慎 [已核实]**：闭环评测为单次冻结协议（deterministic=true），且环境存在已记录的内禀非确定性（≈1/500，`docs/v7_reports/v7_p0_idm_baseline.md:48`）；训练 seed 间方差未见系统性测量。

### 6.2 工程实现层（影响实验可靠性的已知偏差/陷阱）

1. **obs 指纹的脆弱性 [已知偏差]**：`env/obs/*.py` 任一字节变化 → 指纹变化 → 所有数据集"失配"（软告警）。当前存在若干**注释级 stale 引用**（`env/obs/schema.py:239` 等），修注释需联动数据集 meta 或接受告警。**这是一个设计权衡（强一致性 vs 脆弱性）**。
2. **config 死键 [已知偏差]**：`build_model` 只消费部分 `config/model.yaml` 键（未消费：`temporal.*`、`world_model.enabled/conditioned_on/loss`、`moe.router.type/supervision/top_k`、`policy.rollout_steps`、`policy.action.*`、`param_budget_estimate`、`spatial.type/hidden_dim` 等）。**风险**：改配置以为改了模型，实际静默 no-op（本次排摸已逐臂验证无此问题）。同类：`config/env.yaml::obs`（及 `sim.*` 部分键）未被任何运行链路消费——采集/训练/评测全链路用 `ObservationBuilder` 内置默认（实测两侧 `config["env"]` 为空、`build_pool` 无 obs_config 参数）；`config/train.yaml:200-202` 的 `kl_anchor_coef/final_coef` 无读取点（KL 锚只认 CLI）。
3. **`TemporalAttention` 掩码偏差 [已知偏差，待修]**：全列无效（掩码全 0）时 pooled=0 → 输出为 `out` 层 bias（范数 ≈0.5–0.6，依初始化），**非约定的严格 0**（`net/temporal.py:85`）。未修的原因：改动会改变在跑实验的训练数值。
4. **默认终局值不一致**（§4.5）：无 reward 配置的 RL 落旧默认值 (+30/−19/−15/−23/−5)。
5. **基线参照文件缺失**：`runs/baseline_eval/val_reference.json` 不在仓库（gitignored 且已被清理）→ 缺省评测走"缺少可比数据"告警分支。
6. **LD 口径版本漂移**：`{0,20,40,60,80}`（v6）→ `{5,10,15,20,30}`（v8 A1'）——与数据集指纹绑定，历史数据不可直接复用（已核实 1838 数据与新代码指纹不一致）。
7. **`tools/train.py` argparse description 过期**（语义与 `pipeline/stages.py` 不符）；`collect_expert.py`/`gene_env.sh` 帮助文本默认值与常量不一致（已记录）。

### 6.3 评测方法论层

1. **筛选分辨率不足 [已核实]**：clean150 的 SE≈3.5pt；±1σ 级差异不可判。当前排摸的多项"+7~15pt"需 clean500 确认。
2. **选择集污染风险**：先看 eval500 再挑配置 = 污染；我们使用 clean150 筛选 → clean500/eval500 确认的纪律，但**clean500 与 eval500 同源（val spec 分层抽样）**，独立性有限。
3. **单次评测（冻结协议）**：无 seed 重复，方差估计缺失；环境存在已记录的内禀非确定性（≈1/500）；paired_eval 只在 v7 用过。
4. **offline 指标无区分力 [已核实]**：val loss/action err 在排序上失败（多臂离线优于基线、闭环各异）。

---

## 7. 希望获得的 review / 研究方向

1. **容量–鲁棒性反转的机制解释与可操作的判据**：是否有已知文献/理论（信息论、泛化界、控制中的鲁棒 MPC/IL 复合误差）能解释并指导容量分配？如何设计实验区分 §6.1-1 的三个假说？
2. **off-road 主导失败的诊断路径**：给定我们的观测/动作空间，推荐哪类实验（消融/探针/反事实）定位"几何判读 vs 控制振荡 vs 奖励塑形"？
3. **世界模型的学习目标**：latent 自回归 + 教师强制之外，是否有更适合"为规划服务"的 WM 监督（多步一致性、不确定性、值感知）？
4. **模仿数据的迭代策略**：在专家上限≈0.76 的前提下，DAgger/RL 的合理投入点与配方（我们观察到 +11pt 级提升空间）。
5. **评测协议**：如何在有限算力下获得可信的配置排序（统计设计：样本量/配对/复现的取舍）？
6. **对我们方法论的反驳**：哪些结论在你看来越界了证据？（请指名 §5/§6 的具体条目）

---

## 8. 附录

### 8.1 关键文件索引

| 区域 | 文件 | 说明 |
|---|---|---|
| 观测 | `env/obs/{builder,schema,ego,od,ld,nav,signal,others,static,world,memory}.py` | 通道实现 + schema manifest |
| 环境 | `env/metadrive_env.py`、`env/scenario/*`、`env/specs/*`（生成物） | 场景生成/校验、spec 契约 |
| 模型 | `net/{model,st_gnn,spatial,moe,mem,temporal,encoders,policy,plan_head,anchor}.py`（`ValueHead` 在 `policy.py:203`） | 见 §3 |
| 训练 | `pipeline/{stages,trainer,eval_runner,monitoring,frames,buffer,phase3_loop,run_paths}.py` | 四段训练 + 评测 |
| 奖励 | `reward_model/{terms,aggregation,kpi}.py`、`docs/rl_reward_v5.md` | 17 项 + 聚合 + KPI |
| 配置 | `config/{default,train,model,env,eval}.yaml`、`config/arms/v7_arm1_offroad.yaml` | 当前采用 RL 配方 |
| 证据 | `docs/v7_reports/`（锁定）、`docs/cleanup/`（清理记录）、`docs/rl_reward_v5.md` | 历史与审计 |
| 数据 | `datasets/BTC20261007-2202_expert5k_v8`（+`_expert500val_v8`） | 当前训练数据（360k 行） |

### 8.2 复现命令（示例）

```bash
# 参数计数（CPU）
tools/venv-python -m net.param_probe            # 或 build_model 逐模块聚合
# 数据采集（规则专家）
tools/venv-python tools/collect_expert.py --specs env/specs/scenarios_train.json --limit 5000 \
    --out datasets/BTC<ts>_expert5k --workers 6
# Stage A / B
tools/venv-python tools/train.py --stage A --bc-dir datasets/BTC<ts>_expert5k --wm-epochs 20 --out runs/train/stage_a
tools/venv-python tools/train.py --stage B --bc-dir datasets/BTC<ts>_expert5k --ckpt runs/train/stage_a/final.pt --bc-epochs 20 --out runs/train/stage_b
# 闭环评测（eval500 协议）
tools/venv-python tools/test.py --policy ckpt --ckpt runs/train/stage_b/final.pt \
    --spec env/specs/scenarios_eval500.json --tracker lqr --workers 6 --out runs/eval
# Stage C（RL，WM 冻结 + KL 锚）
tools/venv-python tools/train.py --stage C --config config/arms/v7_arm1_offroad.yaml \
    --ckpt runs/<base>/ckpt_epoch005.pt --spec env/specs/scenarios_train_dagger_r1.json \
    --updates 200 --rollout-steps 256 --seed 11 --kl-anchor-coef 0.05 --kl-anchor-final-coef 0.02
```

### 8.3 事实核验说明

- 本文件的架构/维度/损失/奖励数字均在 2026-10-09 由独立核查子任务逐条对照代码复核（含实机运行 `available_terms()`、`build_model` 参数计数、cheap-path 逐位比对）；§5 的闭环/消融/排摸数字来自实际评测产物。
- 文档与代码同步修复项（同日）：`env/scenario/generator.py` 两处"与 n 无关"错误 docstring（实测证伪：几何槽位依赖 split 总数 n）；各域 README 重写。
- **[仍需注意]** §5.5 排摸未完成；clean150 数字未经 clean500 确认前不应作为结论引用。
