# v7 P1-B 失败诊断：为什么 obs v4 的 static 段没有修好 tollgate？

- 日期：2026-10-02（只读诊断；不改 repo；脚本/证据写 `/tmp/opencode/p1b_diag/`）
- 对象 ckpt：`runs/BTC20261002-0941_v7p1b/stage_b/ckpt_epoch010.pt`（sha256 `d977dec4…`，P1-B 选定）
- 数据：`datasets/BTC20261002-0941_expert5k_v4`（train，5000 spec / 360,561 行）、
  `datasets/BTC20261002-0941_expert500val_v4`（val = `scenarios_eval500`，含 T3/tg45 全部场景）
- 采集教师：MetaDrive `IDMPolicy`（`report.json::expert=idm`）；评测基线：`PurePursuitIDMPolicy`

---

## 0. TL;DR

1. **数据侧假设被证伪**：v4 采集教师（IDMPolicy）的演示**包含"变道绕亭"行为** ——
   train 455 条 tollgate episode 中 **317 条（69.7%）** 在进 gate 前换到自由车道、通过净空
   median 1.81 m；评测 45 条 tollgate 场景的 val 演示 **31/45**；T3 9 条中 **7/9**。
   教师**没有**显式建筑扫描，但它的 lidar 能看见 `TollGateBuilding`（speed=0），
   IDM 把它当"静止前车"→ 减速 → overtake 变道；**无交通实验（density=0）仍绕亭 6/7**。
   "演示无变道 ⇒ BC 无从学会"**不成立**。
2. **static 特征不是"未被使用"**：同状态置零 static 5 维 → `action_mu` 变化
   （present=1：|Δds| 0.045 m、|Δdθ| 0.005 rad、plan 0.27 m；present=0：≈5e-5）；
   整段置零重评 tg45：碰撞数 3→11（行为确实变了）。**但使用方式几乎全是"纵向减速/避碰"，
   不产生变道**：45/45 仍失败，T3 9 条三种臂终止逐条完全一致（8 out_of_road + 1 collision）。
3. **模型横向欠表达（关键）**：岗亭接近态 ckpt 闭环 |dθ| 输出 mean 0.0072 / max 0.041 rad、
   0% 帧 >0.05；演示同条件 |dθ| mean 0.059（val T3）/0.043（train）、31%/26% 帧 >0.05
   （6–8×）；教师强制（喂专家自身状态）下 μ|dθ| 0.033、corr 0.50、MAE 0.041 ≈ 信号
   ⇒ **不是完全没学会，而是学得弱 + 闭环自漂移后横向塌缩**。
   变道"执行帧"（|lane_lat|>0.8）62% 被 `roundtrip_fail` 过滤（829/1333）；
   action head 界 [-0.6,0.6] 远未饱和 ⇒ 不是动作空间截断。
4. **BC 本体即失败**：Stage A（纯 BC epoch20）T3 也 0/9；ckpt 在 eval500 0.312 vs
   教师 IDMPolicy 同场景 **0.702**、PurePursuitIDM 0.756 ⇒ 基础模仿质量差距 ~39pp；
   tollgate 失败是这一欠拟合的症状之一。
5. 结论：**v4 没有修好 tollgate 的原因不在"观测看不见岗亭"，而在
   （a）演示的变道触发信号主要是"前方静止障碍（车/亭）→减速→overtake"这一动力学链条，
   BC 学到了减速、没学到 lateral 决策；（b）变道横向动作在数据过滤+损失平均下被压扁；
   （c）plan/策略没有显式的"自由车道目标"表征。** 修复路线见 §5。

---

## 1. 数据侧（关键）：演示里到底有没有"变道绕亭"？

### 1.1 逐 episode 几何统计（env 重建岗亭位置 + npz 专家轨迹投影）

方法：对每条 spec 重建 env（build+reset）取 `TollGateBuilding` 的 lane/位置；把 npz 记录的
专家轨迹（2 Hz `pose`）投影到岗亭车道，算 gate zone（亭半长+2.25 m）内 min|lat|（净空 =
min|lat| − lane_width/2）、进 gate 前是否已换到邻车道。脚本
`booth_traj.py`，结果 `booth_traj_*.json`。

| 数据集 | episode 数 | 变道绕亭（clearance≥1 m） | 未到 gate/失败 | clearance 中位数 |
|---|---|---|---|---|
| train tollgate（expert5k_v4） | 455 | **317（69.7%）** | 138（86 out_of_road + 40 collision + 12 max_step） | 1.81 m |
| val tollgate45（=评测场景） | 45 | **31（68.9%）** | 14（8 off + 3 coll + 3 max_step） | 1.81 m |
| T3 9 条 | 9 | **7/9**（clearance 1.75–1.87） | 2（id166/147 out_of_road，教师自己没过闸） | 1.79 m |

对照：P1-B 评测基线 PurePursuitIDM 同 T3 9 条 9/9 `arrive_dest`、净空 1.74 m；
tg45 35/45=0.778。**IDMPolicy 演示的绕亭净空与评测基线同级。**

补充（纯 npz static 段独立复核，`analyze_static_seqs.py`）：present 期间 rel 桶由
"same→left"（亭从本车道变到左邻车道）在 train 455 中 288 条、T3 6/9 —— 与几何结论一致。

### 1.2 教师机理：IDMPolicy 为什么/何时变道（`expert_probe.py` / `expert_notraffic.py`）

- **有交通**（id34 默认 density 0.13）：lane 1 前方 25–39 m 有**静止 MVehicle**（speed≈0.7 m/s，
  停在收费站），IDM 跟车减速到 1.7 m/s；`overtake_timer>50` 后 overtake 分支触发（左道空闲）
  → 在 static gap 15–27 m 处换到 lane 0（岗亭转左邻），随后通过（clearance 1.765 m）。
- **无交通**（`traffic_density=0.0`）：**同 6/7 spec 仍然绕亭**（clearance 1.77–1.86）；
  逐步探针显示 `front=TollGateBuilding, front_dist 48.8→25.2, front_speed=0.0` ——
  岗亭被 lidar 检出、被当作静止前车，走**同一减速+overtake 链路**。
  （id76 无交通时 out_of_road，说明该链路并非所有场景都可靠。）
- 也就是说：教师**没有显式建筑扫描，但行为上确实会绕亭**；触发信号是
  "前方静止物体 + 减速到阈值 + overtake timer"，**不是**"static 观测本身"。
  在 138/455 条 train tollgate 演示里教师自己失败（没绕成），这 30% 是"反面教材"。

### 1.3 与评测基线 PurePursuitIDM 的对比

| 策略 | T3 9 条 S1 解除 | tg45 成功率 | 变道触发 |
|---|---|---|---|
| IDMPolicy（采集教师，演示） | 7/9 绕亭（val 演示） | 30/45=0.667（8 off + 4 coll + 3 max_step） | lidar 静止前车 + IDM overtake |
| PurePursuitIDM（评测基线） | 9/9 | 35/45=0.778 | 显式 `_static_blocker_gap` 扫建筑 + `_free_alternate_lane` |
| ckpt epoch010 | **0/9** | **0/45** | —（不变道，plan 穿亭） |

结论：**演示不是"没有变道"，而是变道成功率略低于专用规则基线**；BC 应当能从演示中学到，
ckpt 0/9、0/45 属于模型侧失败。

### 1.4 演示帧的可训练性（`trainability.py`）

- tollgate approach 窗口（static present & rel_same）：train 4,172 行，**65.6% trainable**；
  近亭 gap∈[0.4,0.8] 段 62–65% trainable；wm_valid 6 个 horizon 0.98→0.83。
- **变道执行帧**（|lane_lat|>0.8）：1,333 行，只有 **33.3% trainable**，
  过滤原因 62% = `roundtrip_fail`（constant-(v,ω) 圆弧 6 点回代误差超阈值）。
- 变道"发起"帧的监督在（演示首步 dθ：train approach |dθ| mean 0.043 / 25.7% 帧 >0.05；
  val T3 |dθ| mean 0.059 / 31% 帧 >0.05），但执行中段的动作/轨迹目标被大量丢弃。

---

## 2. 模型侧：static 段消融探针（`ablate_static.py`）

协议：T3 9 条 + tollgate45 45 条，ckpt=epoch010，`tracker=lqr`、`eval_reference=plan`；
三臂：ctrl / zero5（static 5 维当前帧+历史帧置 0）/ zero_present（仅 present 置 0）。
结果 `ablation_static.json`、汇总 `ablation_summary.json`。

### 2.1 行为消融

| 臂 | 45 条成功率 | collision | off_road | T3 9 条终止 |
|---|---|---|---|---|
| ctrl | 0/45 | 3 | 42 | 8 off + 1 coll（复现 P1-B） |
| zero5 | 0/45 | **11** | 34 | 与 ctrl **逐条相同** |
| zero_present | 0/45 | 4 | 41 | 与 ctrl **逐条相同** |

- **static 段影响行为**（zero5 使碰撞 3→11、max_rc 普遍更高——没有 static 时模型更敢往前开、
  撞上亭；有 static 时减速、出界），**但任何臂都不能过闸**。
- T3 9 条终止逐条不变 ⇒ 对"变道/过闸"这一决定性问题，static 的有无不改变结果。

### 2.2 同状态灵敏度（ctrl 臂内，同一 obs 双前向）

| 分组 | n | \|Δds\| mean | \|Δdθ\| mean | \|Δdθ\| max | plan 位移 mean/max |
|---|---|---|---|---|---|
| present=1 | 677 | **0.0448 m** | 0.00533 rad | 0.065 rad | 0.272 / 1.625 m |
| present=0 | 1566 | 5.0e-5 | 1.4e-5 | 0.005 rad | 0.0016 m |

- static 被网络消费（present=1 时输出确实变化），但**横向灵敏度只有纵向的 ~1/8**；
  说明模型把 static 当作"减速/避碰"线索，而不是"换道选择"线索。
- ckpt 在岗亭接近态的 dθ 输出：|dθ| mean 0.0072、max 0.041 rad、**0% 帧 >0.05**；
  演示同条件 |dθ| mean 0.059（val T3）/0.043（train）、31%/26% 帧 >0.05
  ⇒ **闭环横向欠表达 6–8×**。
- action head 界 dθ∈[-0.6,0.6]，远未饱和 ⇒ 不是动作空间限制。

### 2.3 教师强制拟合（`tf_probe.py`，在 v4 训练集自身帧上，6 帧历史精确重建）

| 分组 | n | 专家 \|dθ\| | μ \|dθ\| | dθ MAE | dθ corr | 置零 static 的 \|Δds\|/\|Δdθ\| |
|---|---|---|---|---|---|---|
| tg approach（present&same，gap 0.2–1.0） | 2500 | 0.0437 | **0.0326** | **0.0413** | **0.50** | 0.0698 / **0.0195** |
| tg mid-lane-change（\|lane_lat\|>0.8） | 800 | 0.0841 | 0.0402 | 0.0591 | 0.71 | 0.0435 / 0.0052 |
| 非 tollgate 随机（对照） | 2500 | 0.0159 | 0.0112 | 0.0131 | 0.62 | 0.0000 / 0.0000 |

- 教师强制下模型**能**输出有意义的 dθ（approach：μ 0.033 vs 专家 0.044），
  但拟合只是中等（corr 0.50、MAE 0.041 ≈ 信号本身），且执行帧（mid-lc）欠表达 ~2×。
- 对比闭环（§2.2）：ckpt 在自己状态上 dθ 只剩 **0.0066**（教师强制的 1/5、演示的 1/6）
  ⇒ 横向动作在**闭环自漂移后塌缩**（compounding / 状态分布偏移 + 横向响应脆弱）。
- 置零 static 在训练分布上同时改变 ds（0.070）与 dθ（0.020）；在闭环状态上 dθ 效应降到
  0.005 ⇒ static→横向的映射在模型自己的状态上几乎消失。

### 2.4 Stage A（纯 BC，epoch20）T3 9 条

`stage_a/final.pt`：**0/9**（3 collision：id243/166/239；6 out_of_road）。
⇒ tollgate 失败**在 Stage B 之前就存在**（BC 本体），不是 Stage B PPO 坍缩造成的；
Stage B 只是让 clean500 整体再降。

---

## 3. 表征侧（`grad_probe.py` + 结构检查）

- `others`（33 维）在网络里只有**一个** `nn.Linear(33→H)` 投影 + 一个全局 context token
  （`net/encoders.py::embed_others`）；static 5 维与 nav/限速/信号/road_class 共享线性投影，
  没有逐维注意力/门控可读，故用"同状态置零灵敏度"（§2.2）+ 梯度幅值替代。
- **梯度探针**（tollgate approach 1024 行，∂action_mu/∂others_hist 的平均 |grad|）：

  | 段 | ∂ds | ∂dθ |
  |---|---|---|
  | static(16:21) | **0.00632** | **0.00190** |
  | nav(0:11) | 0.00202 | 0.00060 |
  | speed_limit(11) | 0.00230 | 0.00060 |
  | signal(12:16) | 0.00036 | 0.00012 |
  | road_class(21:33) | 0.00337 | 0.00082 |

  ⇒ static 的**逐维**梯度是 others 各段里最大的（约 nav 的 3×）——网络没有忽略 static；
  但 dθ 的梯度整体比 ds 小 3.3×，与"纵向主导"的使用方式一致。
- 评测路径下 `others` 当前帧张量不直接进入计算图（`mem_from_obs` 用 `others_hist`，
  梯度为 None）；static 信息经 6 帧历史（末帧=当前）被消费。
- 策略头 sigmoid 压缩 (ds,dθ) 到 [0,10]×[-0.6,0.6]，**未饱和**；ckpt 的 dθ 输出远小于演示。
- 变道"发起"需要同时满足：岗亭可见（static）、前方静止物（OD/静态）、减速到阈值、
  lateral 决策；BC 训练信号里这四者只在少数帧同时出现，且执行帧 62% 被过滤。

---

## 4. 结论：为什么 v4 没有修好 tollgate

按证据权重排序：

1. **不是"演示无行为"**（假设被证伪，§1）。教师演示 69.7%/7/9 有绕亭行为，净空与评测基线同级。
2. **static 可观测性本身生效但方向不对**：模型确实读了 static（输出灵敏度、zero5 碰撞 3→11），
   但只学会"岗亭在前 → 纵向减速/避碰"，没学会"→ 换到自由车道"（§2）。
3. **横向决策在数据与损失两端被削弱**：
   - 数据：变道执行帧 62% `roundtrip_fail` 被过滤（§1.4）；
   - 损失/表征：动作是连续 (ds,dθ) 回归，变道是少数模式，L2/轨迹损失把横向动作平均成小 dθ；
   - 教师强制下 dθ corr 仅 0.50、MAE≈信号；闭环后 dθ 再塌缩到 1/6（§2.3）——
     不是"完全没学会"，而是"学会了但脆弱、在自身状态上不触发"。
4. **教师的因果链是"静止障碍（车或亭）→减速→overtake"**：static 不是教师变道的唯一/首要触发；
   BC 学到减速链、没学到 timer/overtake 的横向链（§1.2）。
5. **BC 本体即失败（Stage A 0/9）**，且整体欠拟合：ckpt eval500 0.312 vs 教师 0.702 /
   规则基线 0.756。tollgate 不是独立 bug，而是整体模仿质量的症状。

与 P1-B posthoc 的对账：v4 vs v6 AB-only（clean500 Δ=-0.8pp，p=0.70 不可区分）与本节一致——
static 只把行为推向"更谨慎的纵向响应"（zero5 碰撞 3→11 说明有影响，但成功率 0/45 不变），
**没有产生"变道"这一质变**。

---

## 5. 对修复路线的含义（建议）

1. **教师换成 PurePursuitIDM 做闭环 DAgger（最高优先）**：它的 `_static_blocker_gap` +
   `_free_alternate_lane` 是显式、稳定的"建筑→自由车道"触发；DAgger 还能覆盖 IDMPolicy
   失败的 30% tollgate 场景（138/455）与 T3 的 id166/147。用规则教师提供"正确示范"，
   而不是继续从 IDMPolicy 的偶发 overtake 里学。
2. **监督口径修变道帧**：对 `|lane_lat|>0.8` 的变道执行帧放宽/修复 `roundtrip_fail`
   （或单独加权/单独 head）；把"变道"从连续回归里解耦出来（lane-choice 分类头 / 混合密度 /
   plan 条件于目标车道），直接治 dθ 均值回归。
3. **表征加强**：static 段单独 token 或与 OD 障碍 token 做显式 cross-attention；
   把"当前车道前方静态障碍（present/gap/rel）"直接拼进 ego/plan 输入；
   给 plan 增加显式"自由车道目标"（例如输出目标 lane 相对偏移，再生成轨迹）。
4. **评测加早期信号**：除闭环 T3/tg45 外，增加 teacher-forcing 指标
   （approach 窗口 dθ MAE、变道分类准确率、plan 是否指向自由车道），否则每次都要等
   A20+B20 才发现横向没学会。
5. **先修基础模仿质量**：clean500/eval500 0.31 vs 教师 0.70 是主要缺口；在它没解决前，
   tollgate 专项改动很难在总成功率上体现（P1-B posthoc Δ=-0.8pp 不可区分即此意）。

---

## 6. 证据与复现

脚本（全部只读 repo，输出 `/tmp/opencode/p1b_diag/`）：

| 脚本 | 产出 | 说明 |
|---|---|---|
| `analyze_static_seqs.py` | `static_seq_*.json` | npz static 段 + lane_lat 的纯数据行为分类 |
| `booth_traj.py` | `booth_traj_*.json` | env 重建岗亭几何 + 轨迹净空/换道判定 |
| `trainability.py` | `trainability.json` | approach/执行帧的 train_weight/过滤原因 |
| `expert_probe.py` | `expert_probe.json` | IDMPolicy 实时逐步：车道/路由/static/亭几何 |
| `expert_notraffic.py` | `expert_notraffic.json` | 教师 density=0 vs 默认的绕亭对比 |
| `ablate_static.py` | `ablation_static.json` | ctrl/zero5/zero_present 消融 + 同状态灵敏度 |
| `summarize_ablation.py` | `ablation_summary.json` | 消融汇总 |
| `tf_probe.py` | `tf_probe.json` | 教师强制拟合 |
| `grad_probe.py` | `grad_probe.json` | ∂action_mu/∂others 各段梯度幅值 |

复现命令（GPU 串行）：

```bash
cd /workspace/01_Proj/DRL_PathPlan
tools/venv-python /tmp/opencode/p1b_diag/booth_traj.py --which val_t3
tools/venv-python /tmp/opencode/p1b_diag/booth_traj.py --which val_tg45
tools/venv-python /tmp/opencode/p1b_diag/booth_traj.py --which train_tg
tools/venv-python /tmp/opencode/p1b_diag/expert_notraffic.py
tools/venv-python /tmp/opencode/p1b_diag/ablate_static.py --which t3,tg45 --arms ctrl,zero5,zero_present
tools/venv-python /tmp/opencode/p1b_diag/tf_probe.py
tools/venv-python /tmp/opencode/p1b_diag/grad_probe.py
```
