# 设计决议 v1.2（2026-09-26，项目主批准后实施）

> 本文是四条实施 lane（obs/数据、net、trainer、聚类）共同遵循的**冻结契约**，也是 IL 评审的基准。
> 上游事实与缺陷证据见 `.slim/deepwork/planner-rl-feasibility.md`、`docs/experiments.md` 与 `docs/db44fefe-system-review.md`。
> **本轮不启动 RL**；Stage C 在 P0-1/P0-2 修复前标记 EXPERIMENTAL。

---

## 1. 动机：三处已量化缺陷（本轮必修）

| 编号 | 缺陷 | 实测 | 修法 |
| --- | --- | --- | --- |
| **D1** | OD/LD 历史槽位逐帧重排（无 id），而时序核心是 node-wise GRU | **12.5%/步** 的槽位对换物体 | 槽位 = track id（本文件 §3.1） |
| **D2** | 采集逐帧过滤 + 历史按"行位置"重建 → 窗口时间不均匀；episode 头部复制帧带 `valid=1` | **16.8%** 窗口非均匀（最大 3 s 洞）| 存全帧 + 精确查表 + 真实 `hist_valid` |
| **D3** | 未来目标缺失（过滤掉的帧直接 mask 且不补）→ 按有效项归一造成隐性 horizon 重加权，且无 per-horizon 日志 | k=6 时 **25%** 的行无目标 | `wm_valid[6]` + 逐 horizon 日志 |

---

## 2. 架构 v1.2 总览

```
ENV（唯一写入真 mem 的一方）
  ├─ ego / od / ld / others 编码输入（当前帧）+ 6 帧历史（每模态一条 mem，过去 3 s）
  ▼
net（只读真 mem）
  ├─ 分模态编码 → 编码 mem（6 帧）
  ├─ 时序聚合：OD / Ego / Others 用「对各自 mem 的注意力」；**LD 不做时序**（对齐正确时历史近似恒等）
  ├─ Plan head（ego net，**MoE 在此**）= primary(常开) + 8 specific（top-2 软混合）
  │     └─ 产出「下一时刻 ego 特征」
  ├─ Spatio-temporal GNN：基于「更新后的 ego mem + od/ld mem」→ 下一时刻 OD/LD 特征（+ presence/entry 头）
  └─ World model 监督头：od_pred(6,16,5)、ld_pred(6,16,4)、od_presence_pred(6,16)、od_entry_pred(6,16)
        （**未来 LD 监督已移除**；LD 只作输入）

rollout（6 步）
  1) 从真 mem 拷贝 4 份（拷贝隔离：绝不写回真 mem）
  2) 每步：plan head(mem 副本) → 下一 ego 特征 → 挤入 ego mem 副本（弹最老帧）
             → ST-GNN → 下一 OD/LD 特征 → 挤入各自副本（弹最老帧）
  3) 重复 6 次；下次 rollout 重新拷贝
```

### 2.1 detach 语义（项目主指定）

- **真实帧（step0，来自 obs）不 detach** —— 可看作"检测任务"。
- **rollout 合成的后续 5 帧 detach** —— "预测任务"，不回传。
- 实现要点：被切的是**帧（状态）**，**action/pose 链保持可微** → 6 点轨迹目标仍能训练 plan head 的动作链。
- 必须断言：① 每步 loss 对 plan head/encoder 权重有梯度；② 梯度不跨"合成帧"边界回传；③ 真 mem 未被写入。

### 2.2 坐标

保持已验证实现：WM 在 **t0 系**预测 → `od_pred_to_features/ld_pred_to_features` 做 **SE(2) 逆变换到当前累积位姿系** → 重编码。若改局部系递推，需提供等价性测试。

### 2.3 梯度表（现行阶段设置）

| 阶段 | 可训练 | 冻结 | 监督来源 |
| --- | --- | --- | --- |
| A（WM） | 编码器/时序/空间/plan head/MoE/WM | policy/value | GT 动作序列（teacher forcing）+ 未来 OD/LD/presence/entry（逐 horizon 掩码）|
| B（BC） | 主干 + plan head + MoE + policy | WM、value | 首步动作 + 6 点轨迹辅助 + router 软目标（聚类） |
| C（PPO） | **暂不启用**（P0-1/P0-2 未修） | — | — |

---

## 3. 数据契约 v2

### 3.1 OD 固定槽位（槽位 = track id）

- 每 episode 维护 `id → slot` 表，`reset()` 清空；**新对象 → 第一个空槽**（候选内按紧迫度/距离优先）。
- **槽满**：紧迫度最高的新对象驱逐"最长未出现**且已出盒**"的槽。
- 对象出盒 → `presence=0`（槽保留）；**出盒持续 > 1.0 s → 释放**。
- 每帧输出：`od(16,9)`、`od_mask(16)`、`od_id(16) int64`、`od_presence(16) float32`。

### 3.2 obs 键（v2）

| 键 | 形状 | 说明 |
| --- | --- | --- |
| `ego` / `ego_mask` | (1,8) / (1,) | 含末 2 维 `prev_action` |
| `od` / `od_mask` / `od_id` / `od_presence` | (16,9) / (16,) / (16,) / (16,) | 固定槽位语义 |
| `ld` / `ld_mask` | (16,7) / (16,) | 盒式 scope（**前 150 / 后 50 / 左右 25**）|
| `others` / `others_mask` | (1,F_o) / (1,) | **nav(11) + speed_limit(1) + signal(4) + road_class one-hot(K)** |
| `ego_hist` / `ego_hist_mask` | (6,8) / (6,1) | 新 |
| `od_hist` / `od_hist_mask` / `od_id_hist` / `od_presence_hist` | (6,16,9) / (6,16) / (6,16) / (6,16) | id 一致 |
| `ld_hist` / `ld_hist_mask` | (6,16,7) / (6,16) | 只作输入 |
| `others_hist` / `others_hist_mask` | (6,F_o) / (6,1) | 新 |
| `hist_valid` | (6,) | 按**真实缓冲长度**计算 |

> 旧 `nav`/`signal` 键保留兼容，但 `others` 是规范输入。

### 3.3 数据集 v2（`expert_bc.npz`）

- **存全部 policy 帧**（不再逐帧删行），0.5 s 步长（stride=5）；
- 每行：`train_weight`（float，过滤命中 → 0：terminal_window / 不在车道 / cut 标签未验证 / roundtrip 与密点失败）、`wm_valid[6]`（该 horizon 目标帧是否存在且仍在盒内）；
- 每帧：v2 obs 全键 + `od_id/od_presence`；
- 配平/统计 **权重感知**（"计数 = 行数" 与 "加权 = 权重和" 两个口径显式区分）；
- `obs_fingerprint` 升 **v2**；`history_storage: per_frame_v2`；完整 schema 清单写入 `expert_bc.meta.json`。

### 3.4 查表纪律

- 历史：`(episode_id, step − 5j)` 精确查表 + per-slot valid；
- 目标：`(episode_id, step + 5k)` 精确查表 + `wm_valid`；
- `(episode_id, step)` 唯一性 + stride=5 断言；**禁止**按行位置取窗口。

---

### 3.5 数据/产物目录纪律（2026-09-26 定稿，事故驱动）

**`runs/` 清理绝不应影响数据。** 2026-09-26 出现过"全清 `runs/` → 连数据集一起删掉 → 被迫重采 30 分钟"的事故，固化为：

| 目录 | 放什么 | 可清理？ | 命名 |
| --- | --- | --- | --- |
| `datasets/` | 专家 BC 数据集（`expert_bc.npz` + meta + report）| **不可**（仅 obs 版本变更或规模调整时重采）| `BTC<YYYYMMDD-HHMM（北京）>_expert<N>k` |
| `runs/` | 训练/评测/基线产物（run 根 + stage_a/stage_b + logs/manifest）| 可随时清 | `BTC<北京戳>_<name>/`（内含 `stage_a|b/`、`logs/`、`manifest.txt`、快照、`detach.pid`）、`BTC<北京戳>_eval_<tracker><N>` |
| `checkpoints/` | 需要长期复用的 ckpt（跨清理保留）| 谨慎 | `BTC<ts>_stageB.pt` |
| `tools/diagnostics/` | 一次性诊断/侦察脚本（非主流程）| 可清 | 原文件名 |
| `config/baselines/` | 冻结的 KPI 基线参考 JSON（评测协议依赖）| **不可** | `BTC<ts>_val_reference/` |

- **重采的唯一理由**：`obs_fingerprint` 变更（观测契约变了）或用户明确要求扩规模；同版本内一律 `BC_DIR` 复用。
- 数据集与 `runs/` 分离后，"清理 runs/" 是安全的日常操作。
- **后台长任务纪律（2026-09-27 定稿，harness 会话轮换事故驱动）**：训练/评测一律走 `tools/train.sh` / `tools/test.sh`（`setsid+nohup` 脱离 harness 进程组）；run 目录为 `runs/BTC<北京戳>_<name>/`，内含 `logs/stage_{a,b}.log`（唯一日志，含 `[heartbeat]`/`[exit] code|signal` 存活证据）、`manifest.txt`、`config/model.snapshot.yaml`、`detach.pid`、`stage_a|b/`（ckpt/monitor 由训练进程写）；被静默杀掉后用 `run.resume: auto` + `ckpt_every` 从周期 ckpt 原地续跑。

**采集内存模型（2026-09-26 实测 + 修复）**：旧实现的峰值 =
**父进程累积全部逐帧记录**（≈10–15 KB/行；5k specs ≈5 GB，随数据集线性增长，非泄漏）
+ workers（≈1.1 GB × N，由 `--recycle-every` 界住，4 MB/spec 残余）。
8 workers 时 ≈14 GB，逼近 16 GB 上限并引发抖动（实测单条 1 s → 2.7–5.1 s，速率掉到 0.5 条/s）。
**修复**：worker 各自把分片写到磁盘（`_shards/`），父进程只收标量摘要（RSS ≈O(1)），结束后再合并成最终 `expert_bc.npz`
（峰值 = 最终数据一份）。`--workers` 默认 auto（CPU 取半、上限 10），内存不足只警告不降级。

## 4. 路由与聚类（替代手工规则标签）

- **监督**：无监督聚类（8 簇）→ top-2 softmax（τ 默认 0.5）软目标（形如 0.65/0.35）+ 边界平滑；**不再用 BCE/手工标签**（规则标签仅用于体检）。
- **特征**：只用当前帧 `ego+od+ld+others`（禁学习表征）→ 去常数维 → 标准化 + PCA/白化到 ~48 维；与 router 输入口径一致且可复现。
- **防"容量被平淡那团吃掉"**：① 稀有度加权 `w_i = clip((ρ_median/ρ_i)^α, 0.1, 10)`（α=0.75，rule-free）；② 每簇容量上限 ≤25% + 迭代重分配；③ 体检后处理（>30% 二分、<2% 并入）。
- **冻结与版本**：`config/clusters/cluster_v1.npz`（质心/PCA 参数/数据指纹/git hash）+ `pipeline/clusters.py`（加载与软目标）；重聚类走 Hungarian 对齐。
- **体检门**：簇规模/熵/半径；每簇难度/几何/规则标签占比；稀有结构富集（cut-in 等）≥3× 全局，否则 FLAG 并启用两段式退路（先 2-means 分"活跃 vs 平稳"，7 簇只花在活跃子集）。

### 4.1 实测口径与冻结产物（fix-4，2026-09-26）

- **特征（292 维）**：`ego(8) + od(16×9) + ld(16×7) + others(28)` 展平；**od 有效性以 `od_presence` 优先（否则 `od_mask`），无效槽全部清零**（v2 固定槽位下 presence=0 的槽会保留陈旧特征，不清零则簇不是当前帧的函数）；槽位顺序原样保留（=track id 语义，禁止重排）。去常数/近零方差维 → 标准化 → **PCA 白化 48 维**。旧数据 fallback（v1 obs）已实现并校验 `raw_dim`。
- **两类 artifact**：`config/clusters/cluster_v1.npz` + `cluster_v1.report.json`（含 `flag/flag_reason`、簇构成、富集倍数、软目标样例），以及两段式对照 `cluster_v1_two_stage.*`。
- **实测体检（128-spec dev 切片，9509 行，`obs_fingerprint=v2-…`）**：**flag=false**；簇规模 2.4%–20.2%；`cutout_active` 富集 **4.49×** @ 簇 5（≥3× 门通过）、cutin 1.66×；软目标 top1 均值 0.578 / top2 质量 0.915。
- **调用契约**：`pipeline.clusters.load(path) → ClusterSpec`；`soft_targets_from_obs(obs_batch) → (B,8)`（行和=1，torch/numpy 同型）。
- ⚠️ 当前 artifact 是 **128-spec dev 切片占位**；全量 v2 重采后必须重拟合：
  `tools/venv-python tools/fit_clusters.py --bc-dir <V2_BC_DIR> --out config/clusters/cluster_v1.npz --report config/clusters/cluster_v1.report.json`（FLAG 则加 `--two-stage`；换簇编号用 `--align-to`）。

---

## 5. 训练与 IL 度量

- **Stage A**：多步直接监督（GT 动作）+ 逐 horizon loss/ADE/FDE + 每 horizon 有效样本数 + presence/entry BCE/AUC；损失按 `train_weight × wm_valid` 加权。
- **Stage B**：首步动作 + 6 点轨迹辅助 + router 软目标；日志含动作误差均值/中位数/**p95**/分切片、逐 horizon ego 误差、router 指标（top-1 vs 簇、NMI、gate 熵、专家利用率）。
- **会计**：配平、`_action_mu_stats`、PPO BC 锚全部权重感知。
- **`tools/train.sh` / `tools/test.sh`**：零参即可跑（路径/epoch/batch 来自 `config/train.yaml::run|stages`，目录由 `pipeline/run_paths.py` 统一生成）；`setsid+nohup` 后台运行，stdout/stderr 落 `<run>/logs/stage_<x>.log`。

### 5.1 IL 验收门（交项目主 review 的内容）

1. WM 逐 horizon ADE/FDE（仅 present 对象）**显著胜** 匀速 / copy-last / 静止三条基线；
2. presence/entry AUC；目标在盒内比例 vs horizon；
3. ego 逐 horizon ADE/FDE + 平滑度（jerk/曲率）；
4. 动作误差分布（含 p95 与急刹/急转/弯道切片）；
5. router：软目标 CE/KL、vs 簇 top-1、gate 熵、专家利用率、与规则标签 NMI；
6. 闭环 50 条 LQR slice（集成测试，单独列）。

所有曲线标注：`obs_fingerprint` + `cluster_version` + git hash + 种子。**IL 未过门不启动 RL。**

---

### 5.2 监控口径（2026-09-26 定稿，fix-1/fix-2）

**写盘节奏（逐 epoch）**：Stage A 每 epoch 一个 `step`（1..N）；Stage B `step = phase_offset + epoch`（primary 1..P、specific P+1..P+S，全局单调）；
`step 0` 仅写元数据（`cluster_version/k/soft_targets`）。

**tag 约定（CSV 长表 `step,tag,value` 保持不变）**
- 分组统计：`*/mean` 为均值，**`*/n_updates` = 该均值由几次 update 贡献合成**（与物理样本数无关）；
- 物理计数：`valid_samples`（有效样本数）、`valid_weight_sum`（有效权重和）、`slot_count`；
- **val 常量只记一次**（`cv_ade/cv_fde`、`valid_samples/valid_weight_sum`、`train/cv_ade|cv_fde` 仅在首个 epoch）；
- Stage B 同时写 `train/<phase>_*` 与 `val/<phase>_*`（val = 与 Stage A **同 seed/val_frac** 的 episode 留出，`--val-frac` 对 B 生效）；
- 已删除 `train/per_horizon/*`（与 `horizon/*` 逐位重复）。

**TensorBoard 分组多线（`add_scalars`）**：`flush()` 末尾把同族 canonical tag 归到一张图（≥2 sub 才写）。
torch 的 `add_scalars` 语义是**每个 sub 写成一个子 run**（`<logdir>/<main>_<sub>/`，文件内 tag = `main_tag`）→ 在 TB 里勾选同族子 run 即叠加为**一张多线图**；run 列表会多出 `main_sub` 条目。
受影响族：`horizon_loss/ade/cv_ade/ego_next`（A）、`horizon_traj_mae_m/traj_mse_m2`、`slice_action_err`、`label_action_err`、`expert_mix_weight_{primary,specific}`、`bc_terms_{primary,specific}`、`wm_terms`、`grad_norm`、`health`（`val_` 前缀同名分图）。
**离线替代**：`tools/plot_curves.py --stage-a <dir> --stage-b <dir> --out <dir>` 直接出多线 PNG（不依赖 TB）。

**度量 vs 监督的边界（review 必读）**
- 参与优化：A → 逐 horizon `loss`、`ego_next_loss`、`wm_loss_{od,presence,entry}`；B → `bc_loss = action + traj + router`（router 用**聚类软目标** CE/KL）。
- 影响损失但不被优化：`valid_samples/valid_weight_sum/slot_count/wm_valid/train_weight`（掩码与分母）。
- 纯诊断：`ade/fde/cv_ade/cv_fde`、`traj_mae_m/traj_mse_m2`、`action_err_*`（含 `label/`、`slice/` 切分）、`presence/entry AUC`、`top1/NMI/entropy/expert utilization`、`grad_norm`、计时/显存。
- **规则标签只用于体检**（`label/*`），不进 router 监督。

## 6. 实施顺序与合并验收

| 步骤 | 内容 | 验收 |
| --- | --- | --- |
| 1 | obs/数据 schema v2（fix-1） | schema 清单 + 20-spec 抽样 + 单测 |
| 2 | net 重构（fix-2） | 形状/拷贝隔离/detach 断言 + 参数 ≤1.5M |
| 3 | trainer 与仪表（fix-3） | 查表正确性、权重会计、逐 horizon 日志、`train.sh/test.sh` |
| 4 | 聚类与软目标（fix-4） | 体检表（含 FLAG 结论）+ 版本冻结 + 单测 |
| 5 | 小规模重采 → 全量重采 | 窗口均匀、目标完整、权重可审计 |
| 6 | Stage A → Stage B 重训 | **IL 报告（曲线 + 判定表）交项目主 review** |
| 7 | （过门后）RL 前置修复 | P0-1/P0-2/P0-3 修好才谈 Stage C |

---

## 7. 风险与回退

1. **评测噪声**：50 场景 success ±0.2 → 一切闭环比较必须**配对场景 + Wilson CI**，否则会追噪声。
2. **数据版本**：schema v2 重采使旧 ckpt 失效 → 旧数据集与 ckpt 保持可跑，指纹升版并在 meta 声明。
3. **范围蔓延**：mem-bank/聚类合计改动大 → 严格按本文件契约；任何超范围改动先记录再决定。
4. **RL 前置**：`docs/db44fefe-system-review.md` 的 P0-1（plan-as-action/PPO 记账）、P0-2（router 标签时序）、P0-3（rollout 不更新 nav）未修前，Stage C 结果不可作为放行证据。
