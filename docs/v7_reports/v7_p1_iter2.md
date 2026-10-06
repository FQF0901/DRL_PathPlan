# v7 P1 迭代 2：roundtrip 过滤修复 + 横向表达修复 + 教师强制指标

- 日期：2026-10-02
- 基线诊断：`/tmp/opencode/v7_p1b_failure_diag.md`
- 产物：`datasets/BTC20261002-0941_expert5k_v41`（train）、`datasets/BTC20261002-0941_expert500val_v41`（val）
- 代码：`tools/collect_expert.py`（过滤放宽，commit `386b091`）、`pipeline/trainer.py` + `pipeline/stages.py` + `config/train.yaml`（动作 dθ 损失权重，单变量，commit `2e13dfa`）
- 证据目录：`/tmp/opencode/v7_p1_iter2/`（分析脚本 + JSON + 日志）

## 0. TL;DR

1. **roundtrip 门根因**：不是标签错，而是**弧模型对低速大转角侧偏的本质失配**。逐 key 复算
   6 点误差随 |dθ| 单调（err0~|dθ0| corr **0.96**）；spec1294 step255 实测
   `vel_dir − heading = +0.41 rad`（2.35 m/s、steer 0.89 的绕亭变道）→ 常曲率 (ds,dθ) 弧
   无法表示侧偏轨迹。原门（3 s 均值 0.25 m / max 0.5 m）因此丢弃 **62%** 的
   `|lane_lat|>0.8` 执行帧（tollgate 829/1333），LC 可训练率仅 33.5%。
2. **数据修复（单条判据）**：LC 执行帧在首步误差 ≤0.5 m 且 6 点均值 ≤0.5 m 时保留。
   重采 v4.1：train **+6,191 行（全部 LC）**，LC 33.4%→**78.0%**（tollgate LC
   33.3%→**91.1%**、curve 44.8%→88.8%、uturn 31.6%→70.2%）；val +623（LC 34.7%→79.1%）。
   **教师成功率不变**（train 69.10%→69.08%、val 70.2%→70.0%，采集内禀噪声）。
3. **横向欠表达根因（损失尺度）**：动作 L2 = 两维等权平均，dθ 项只占 ds 项的
   **0.85%**（approach 帧 MAE 0.4475 m vs 0.0413 rad）；sigmoid 输出层 raw 梯度再乘
   span 比（10 / 1.2）→ 横向学习信号 ~1%。TF：μ dθ std 0.048 vs 专家 0.082、corr 0.50。
4. **最小修复**：`BCConfig.action_dim_weights=(w_ds, w_dθ)`（默认 (1,1) 逐位不变），
   train.yaml 取 **(1.0, 69.4=(10/1.2)²)**（相对误差等权）。2-epoch TF 探针
   （v4.1，非完整重训）：**train/留出 val 一致改善** —— approach dθ MAE
   0.0341→**0.0137**（train）/0.0426→**0.0254**（val）、corr 0.605→**0.926**/0.507→**0.775**；
   val T3 corr 0.616→**0.843**；mid-LC MAE −49%/−33%；ds/traj 不回退。对照臂
   （w=1 同 2 epoch）证明改善来自权重本身。commit `2e13dfa`。
5. **下一步**：用 v4.1 + `[1.0, 69.4]` 跑完整 A20+B20，闭环 T3/tg45/eval500 对照；
   若横向仍弱，再上 lane-choice 解耦头 / 显式自由车道目标（诊断 §5.2–5.3）。

## 1. roundtrip 过滤根因（读数）

方法：直接读 v4 npz 的 `traj6`（弧重建关键点）vs `traj30_measured`（实测局部轨迹）复算
逐 key 误差（与采集侧 `key_err` 同口径；复算均值 vs 存储值 maxdiff 5e-7）。脚本
`analyze_roundtrip.py`、`gate_simulation.py`、`dump_lc_frames.py`、`probe_sideslip2.py`。

### 1.1 误差画像（train v4；n = 窗口可算行）

| 分组 | n | key1 | key2 | key3 | key4 | key5 | key6 | 首步 err0 | err0~|dθ0| corr |
|---|---|---|---|---|---|---|---|---|---|
| 全部可算行 | 331,841 | 0.042 | 0.078 | 0.108 | 0.134 | 0.158 | 0.181 | 0.042 | **0.965** |
| tollgate | 25,307 | 0.036 | 0.066 | 0.091 | 0.110 | 0.127 | 0.142 | 0.036 | 0.958 |
| LC（\|lane_lat\|>0.8） | 13,310 | 0.155 | 0.282 | 0.369 | 0.413 | 0.433 | 0.448 | 0.155 | 0.965 |

- 误差随 horizon 累积、随转向量线性放大；`err0 ~ 速度` 相关 ≈0（−0.04）。
- 被 roundtrip 门过滤的 LC 帧（8,641 行）：首步 err mean 0.194 / p90 0.452，
  94% 首步 ≤0.5 m、71% ≤0.25 m —— **首步动作标签本身大多可用，是 3 s 窗口的
  累积发散把它们判死**。

### 1.2 物理根因：低速大转角侧偏

spec=1294（tollgate，CS$）step≈255 绕亭变道（IDM overtake，speed≈2.3–2.7 m/s）实测：

```
step=255 speed=2.35 steer=+0.891 head=-1.9560 vel_dir-head=+0.4083 disp_dir-head=+0.3469
```

- `vel_dir − heading = +0.41 rad`（≈23°）：MetaDrive 车辆在低速度+大转角下**真实侧偏**，
  位移方向与车头方向显著不一致；
- 常曲率弧模型假设速度沿车头方向（`interpolate`），因此 0.5 s 内即产生 ~0.15 m 横向
  偏差、3 s 累积 ~0.45 m；
- 同一窗口实测航向变化仅 0.242 rad —— 弧模型把"侧滑的平移"错误当成"转弯"。
- 结论：`roundtrip_fail` **不是 bug，也不是标签错误**，而是 (ds,dθ)@2 Hz 表示对高侧偏
  机动的固有近似误差；它对变道/绕行帧的过滤率因此系统性偏高（tollgate 62%）。

### 1.3 候选门量化（train v4，新救回行）

| 候选 | 新增 trainable | LC 可训练率 | tg 可训练率 | approach 可训练率 | 救回帧 err0 / 6key 均值 |
|---|---|---|---|---|---|
| G0 原门（mean≤.25 & max≤.5） | 0 | 33.5% | 74.6% | 65.7% | — |
| G1 首步≤0.5（全局） | +70,634 | 91.8% | 90.3% | 87.2% | 0.119 / 0.381 |
| G2 首2点 mean/max 原门 | +54,516 | 70.9% | 87.8% | 81.5% | 0.079 / 0.337 |
| G4 阈值×2（mean≤.5 & max≤1） | +57,185 | 78.0% | 89.7% | 86.1% | 0.104 / 0.318 |
| **G-LC（采纳）**：LC 且 first≤.5 且 mean≤.5 | **+6,191** | **78.0%** | 77.4% | 70.2% | 0.130 / 0.337 |

\* 表内为 train v4 上复算；v4.1 实际重采一致（§2）。采纳 G-LC 的理由：**只动问题帧**
（新增行 100% 为 `|lane_lat|>0.8`，其余 25.9 万可训练帧逐位不变），把"3 s 弧不可复现"与
"首步动作不可信"解耦，同时用 6 点均值 ≤0.5 m 限制全窗发散。

## 2. 数据修复与 v4.1 重采（对照）

重采命令与 v4 完全一致（`--specs env/specs/scenarios_train.json --limit 5000 --workers 6`；
val `--specs env/specs/scenarios_eval500.json --workers 6`）。脚本 `compare_v4_v41.py`。

### 2.1 filter 前后（同一 episode 精确对照 = 在 v4.1 帧上复算旧门）

| 指标 | v4（旧门） | v4.1（新门） |
|---|---|---|
| train 行数 / trainable | 360,561 / 259,647（72.0%） | 360,450 / **265,742（73.7%）** |
| train roundtrip_fail | 71,279 | 65,073 |
| train LC（\|lat\|>0.8）trainable | 4,653 / 13,905（33.5%） | **10,833 / 13,897（78.0%）** |
| — tollgate LC | 33.3% | **91.1%**（救回 771） |
| — curve / uturn / roundabout LC | 44.8% / 31.6% / 8.4% | 88.8% / 70.2% / 36.0% |
| — intersection / t_intersection LC | 24.7% / 32.5% | 63.2% / 72.7% |
| val LC trainable | 493 / 1,407（35.0%） | **1,111 / 1,405（79.1%）** |
| val tollgate LC | 30.7% | 86.6% |
| 救回帧画像（train） | — | err0 mean 0.130 / 6key 均值 0.337 |
| tollgate approach 可训练 | 65.7% | 70.2% |
| T3 9 条 LC 可训练帧 | 13 | 30（id166 4 帧仅 1 可训练，教师自身失败场景） |

### 2.2 教师成功率（filter 不影响 episode，仅核验重采一致）

| | v4 | v4.1 |
|---|---|---|
| train arrive_dest | 3,455/5,000 = **69.10%** | 3,454/5,000 = **69.08%** |
| val arrive_dest | 351/500 = **70.2%** | 350/500 = **70.0%** |

终止分布同样稳定（collision 730→731、out_of_road 455→455、max_step 360→360）。
**两次采集非严格确定**（共同帧 pose maxdiff 63 m：MetaDrive 交通/物理内禀噪声），
故跨数据集比较是分布性的；2.1 的精确对照用 v4.1 内复算旧门完成。

## 3. 横向表达分析与最小修复

### 3.1 读数：动作分布与损失尺度

- 专家动作（v4 trainable）：全体 |ds| 3.47 m、|dθ| 0.015 rad；tg approach |ds| 2.64、
  |dθ| 0.036（std 0.075）；LC |ds| 2.46、|dθ| 0.062（std 0.080）。
- ckpt TF（v4，`tf_probe.json`）：approach μ|dθ| 0.033 vs 专家 0.044、μ std 0.048 vs 0.082、
  corr 0.50、MAE 0.041 ≈ 信号；mid-LC μ std 0.046 vs 专家 0.103（欠表达 2.2×）。
- **损失量级**：`per_sample = mean_d(diff_d²)`；approach 帧 MAE 0.4475 m vs 0.0413 rad
  → dθ 项只占 ds 项的 **0.85%**；sigmoid 输出层 `d(action)/d(raw) = span·σ'`，
  span 比 10/1.2=8.33 → 每单位动作误差的 raw 梯度比 ≈ **1.1%**。dθ 在共享主干上的
  学习信号被 ds 淹没，这是"学得弱 + 闭环自漂移后塌缩"的直接机制。
- **mu/logstd 初始化**：mu 零初始化 → (5.0, 0.0)，dθ 无偏（sigmoid 中点 = 0 rad）；
  logstd init −1.0（std 0.37 rad）对确定性 BC 无影响，但对 Stage C PPO 而言
  dθ 探索 std 0.37 vs 专家 0.04–0.08 过宽（留待 RL 侧，非本轮）。
- **roundtrip "平滑"项**：全库无显式动作/轨迹平滑损失（grep 仅 WM 的 smooth_l1）；
  所谓"平滑"即 §1 的 3 s 弧聚合门——它的过过滤才是横向监督损失，已按 §2 修复。

### 3.2 最小修复（单变量）

`BCConfig.action_dim_weights: (w_ds, w_dθ)`（默认 `(1,1)` = 旧行为逐位一致）：

```python
per_dim = diff ** 2 if loss_type == "l2" else diff.abs()
per_sample = (per_dim * action_dim_weights).sum(dim=-1) / 2.0   # 旧: (diff**2).mean(dim=-1)
```

- 候选 A：`w_dθ = 10/1.2 ≈ 8.33` —— **尺度对齐**：让两维"每单位动作误差"的 raw 梯度相等；
- 候选 B：`w_dθ = (10/1.2)² ≈ 69.4` —— 相对误差等权（归一化到动作界比例）。
- 探针结果决定默认值（§4）；配置经 `stages.B.bc.action_dim_weights` 读取并写入 metrics。

## 4. 教师强制前后对照（探针）

协议：固定 ckpt `stage_b/ckpt_epoch010.pt`、固定数据 v4.1、复刻 Stage B primary
（MoE 关、冻结 st_gnn/value/experts/router、lr 3e-4、batch 1024/micro 512、action 1.0 +
traj 0.1、全 trainable 行、同 seed），每个 w 臂从头加载 ckpt 微调 **2 epoch**
（≈4 min/臂，非完整重训）。脚本 `tf_finetune_probe.py`（train 组）+ `tf_finetune_probe_val.py`
（留出 val 组，eval500 场景）。ckpt_before = 不微调直接 TF。

### 4.1 train 组（v4.1 trainable 行；n: approach 1500 / mid-LC 1215 / 非 tg 1500）

| 臂 | approach μ\|dθ\| | approach std（专家 0.0727） | approach MAE | approach corr | mid-LC MAE | mid-LC corr | MAE ds（approach） |
|---|---|---|---|---|---|---|---|
| ckpt_before | 0.0250 | 0.0409 | 0.0337 | 0.542 | 0.0557 | 0.727 | 0.1097 |
| w=1（对照） | 0.0372 | 0.0570 | 0.0341 | 0.605 | 0.0469 | 0.759 | 0.1091 |
| w=8.33 | 0.0359 | 0.0579 | **0.0251** | **0.799** | 0.0397 | 0.833 | 0.1006 |
| **w=69.4** | 0.0348 | **0.0657** | **0.0137** | **0.926** | **0.0241** | **0.933** | 0.1061 |

- **剂量响应单调**：dθ MAE 0.0341（对照）→ 0.0251（8.33）→ **0.0137**（69.4，−60%）；
  corr 0.605 → 0.799 → 0.926；μ std 0.0570 → 0.0579 → 0.0657（专家 0.0727）。
- **ds 无回退**：approach MAE ds 0.1091→0.1061、mid-LC 0.1585→0.1486（w69.4）；
  训练端 `bc_action_err_mean` 0.04357→0.03984、`bc_traj_mae_m` 0.3353→0.3207 均改善。
- 对照臂（w=1）证明：2 epoch 微调本身只把幅度从 0.0409 抬到 0.0570、MAE 不动（0.0341）；
  **MAE/corr 的改善来自 dθ 权重，不是微调本身**。
- 非 tg 随机组同向（w69.4：MAE 0.0085 vs 对照 0.0134、corr 0.871 vs 0.707）。

### 4.2 留出 val 组（eval500 场景，含 tollgate45/T3；ckpt 从未见过）

| 臂 | val approach μ\|dθ\| | std（专家 0.0819） | MAE | corr | val mid-LC MAE | mid-LC corr | val T3 MAE | T3 corr | T3 std（专家 0.0618） |
|---|---|---|---|---|---|---|---|---|---|
| ckpt_before | 0.0284 | 0.0454 | 0.0418 | 0.436 | 0.0553 | 0.738 | 0.0226 | 0.599 | 0.0332 |
| w=1（对照） | 0.0409 | 0.0616 | 0.0426 | 0.507 | 0.0475 | 0.747 | 0.0219 | 0.616 | 0.0439 |
| w=8.33 | 0.0422 | 0.0600 | 0.0359 | 0.662 | 0.0435 | 0.792 | 0.0193 | 0.709 | 0.0462 |
| **w=69.4** | 0.0398 | **0.0654** | **0.0254** | **0.775** | **0.0320** | **0.862** | **0.0138** | **0.843** | **0.0518** |

- 留出泛化与 train 组**同向且幅度可观**（T3：MAE −37%、corr 0.616→0.843），排除"训练集
  记忆"解释；`val_nontg`（n=1000）同样 w69.4 最优（MAE 0.0137 vs 对照 0.0211、corr 0.911）。
- ds MAE 在 val 上基本不变（approach 0.3595→0.3599、T3 0.4089→0.4105），确认
  dθ 权重不以纵向/整体拟合为代价。
- 结论：**取 w_dθ=69.4**（train.yaml 默认）；w=8.33 保留为保守备选（单变量消融时可切）。

### 4.3 与诊断基线对账

诊断 `tf_probe.json`（v4 数据、gap 0.2–1.0 含过滤行）approach：专家 |dθ| 0.0437、
μ 0.0326、MAE 0.0413、corr 0.50；本探针 ckpt_before 在 v4.1 trainable approach 上
专家 0.0358、μ 0.0250、MAE 0.0337、corr 0.542 —— 口径一致（样本构成不同）。
修复后 MAE/corr 均显著超过原 ckpt，且幅度 std 接近专家（0.0657/0.0727 train、
0.0654/0.0819 val），"欠表达"基本消除（教师强制口径）。


## 5. 下一步

1. **完整重训（单变量链）**：v4.1 + `action_dim_weights: [1.0, 69.4]` 跑 A20+B20
   （同一 run 根，记录 config hash），闭环评测 T3 9 / tollgate45 / eval500（`--tracker lqr`、
   `eval_reference=plan`，与 P1-B 同口径），对照 P1-B ckpt（0/9、0/45）与教师基线。
   预期：横向输出幅度/触发改善；但 tollgate 成功仍需"变道决策"而非仅动作幅度。
2. **早期信号**：把 §4 的 TF 探针（approach/mid-LC dθ MAE、corr、μ/专家 std 比）作为
   每次 BC 训练的验收项（不需等闭环）；若 corr 未到 ~0.8 或 std 比 <0.8，先查损失/数据。
3. **若闭环横向仍弱**（变道不触发）：按诊断 §5.2–5.3 升级——lane-choice 分类头 /
   plan 条件于目标车道 / static 与 OD 障碍显式 cross-attention；数据侧用
   PurePursuitIDM 规则教师做 DAgger（显式 `_static_blocker_gap` + `_free_alternate_lane`）。
4. **Stage C 探索宽度**：logstd init −1.0（std 0.37 rad）对 dθ（专家 std 0.06–0.08）
   过宽，进入 PPO 前考虑 `policy_logstd_max` 或分维探索 std。
5. **一致性（可选）**：`Phase3Config` 尚未接 `action_dim_weights`；若 phase 3 继续使用，
   需同步该键（本轮改动范围外，已在代码注释标注）。

## 6. 复现

```bash
cd /workspace/01_Proj/DRL_PathPlan
# 数据（CPU；v4.1 已产出，datasets/ gitignore）
tools/venv-python tools/collect_expert.py --specs env/specs/scenarios_train.json --limit 5000 \
    --out datasets/BTC20261002-0941_expert5k_v41 --workers 6
tools/venv-python tools/collect_expert.py --specs env/specs/scenarios_eval500.json \
    --out datasets/BTC20261002-0941_expert500val_v41 --workers 6
# 分析（CPU）
tools/venv-python /tmp/opencode/v7_p1_iter2/analyze_roundtrip.py
tools/venv-python /tmp/opencode/v7_p1_iter2/gate_simulation.py
tools/venv-python /tmp/opencode/v7_p1_iter2/compare_v4_v41.py
# TF 探针（GPU）
tools/venv-python /tmp/opencode/v7_p1_iter2/tf_finetune_probe.py \
    --dataset datasets/BTC20261002-0941_expert5k_v41 --weights 1.0,8.33,69.4 --epochs 2
tools/venv-python /tmp/opencode/v7_p1_iter2/tf_finetune_probe_val.py \
    --weights 1.0,8.33,69.4 --epochs 2   # 附留出 val 评估
# 测试
tools/venv-python -m pytest tests/test_collect_expert_v2.py tests/test_bc_pretrain.py -q
```
