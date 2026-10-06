# v7 结构迭代 B：K-anchor 计划头（K=6 形状锚 + 连续速度 + WTA + 选择头；车道系对齐；方案 A）

- 日期：2026-10-05（实现 + CPU 全量测试 + GPU 冒烟 ≤15 min）
- 提交：内容 `f42d648`；修订锚 `489091b`（docs/v7_program_prereg.md §11 + §8 锚行）；
  测试断言收紧 follow-up `583efee`（plan[:,0]=arc_step 一致性）
- 范围：`net/anchor.py`（新）、`net/plan_head.py`、`net/model.py`（plan 路径）、
  `pipeline/trainer.py`、`pipeline/stages.py`、`config/`、`tests/`、`docs/`、`tools/`；
  **未动 `env/`**。
- 预注册：`docs/v7_program_prereg.md` §11（揭盲前冻结）。

---

## 1. 设计（方案 A：soft-mixture 锚 + 连续速度 + 连续残差；PPO 零改）

### 1.1 前向

```
latent ──► anchor_head  → logits (B,K)          # 选择头（零初始化 ⇒ 初始均匀）
       ──► speed_head   → speed_ds (B,6)        # 连续速度头（10·sigmoid，初始 ≈3.5 m/步）
       ──► residual_head(latent ⊕ anchor_embed_k) → residual (B,K,6,2)  # 逐锚残差（零初始化）

p = softmax(logits/τ)（推理可 hard=argmax）
plan_ds      = speed_ds + Σ_k p_k·res_ds_k
plan_dθ_lane = Σ_k p_k·(anchor_dθ_k + res_dθ_k)
plan_dθ_ego  = plan_dθ_lane + lane_follow(plan_ds, Δψ, κ)      # lane 帧 → ego 帧
plan[:,0] = action_mu（输出契约不变）；plan[:,1:] = 上式混合计划
```

- **因子化依据（fix-3）**：expert 路径 95.4% 方差在速度（PC1）；首步动作对后续 dθ 的
  R² ≤ 0.095 ⇒ 形状（dθ 剖面）用 K=6 锚、速度用连续头。
- **rollout 集成**：t0 一次算出的锚计划驱动 6 步 rollout 尾段（step0 仍 = `action_mu`），
  `traj_xy` 与 `plan` 一致；逐步 policy 调用在锚开启时跳过（WM 状态链仍由 plan head 的
  `ego_next` 推进）。
- **PPO 兼容**：policy/PPO 路径（`action_mu/action_logstd/value/sample/logprob/entropy/KL`）
  零改动；额外输出仅在 `num_anchors>0` 时出现（`anchor_logits/probs/plan/speed/residual/ctx`）。

### 1.2 锚字典（K=6，可复现）

- `tools/fit_plan_anchors.py`：expert5k_v41（`train_weight>0` 且 action 链有限；moving L≥1 m；
  120k 子样 seed=0；ego `cumdtheta` KMeans `n_init=10, random_state=0`）→ 每簇物理原型
  （`ds` 均值 + **lane 帧** dθ 均值）。
- 复现读数：簇大小 **`[92519, 7823, 3012, 8461, 4515, 3670]`**，与 fix-3 逐位一致；
  产物 `config/plan_anchors_k6.json`（sha256 `79829ef705285c79a60252e9db6ac58d1b1e48830a4fe4d32327a1fa0f3873bf`）。
- 内置默认与文件同源（截断 ≤5e-4）；文件缺失 → 回退内置默认；文件非法 → 报错（不静默）。
- 簇语义（fix-3）：C0 直行（77.1%）、C1 右弯、C2 左转回正（tollgate/merge 主导失败模式）、
  C3 缓右、C4 左弯、C5 环岛（先左后回正）。

### 1.3 车道系对齐（记录口径）

- 锚/残差 dθ 以 **lane 帧**表达；回投 = `dθ_0 = Δψ + κ·ds_0`、`dθ_i = κ·ds_i`（i≥1），
  其中 `Δψ = lane.heading_err`（v5 lane 块 dim 1，正 = 车道在左）、`κ = lane.curvature`（dim 3）。
- `valid = lane_mask × near_valid`；无效 → 恒等（旧 schema v4 数据缺 lane → 0 token/mask=0，
  锚退化为 ego 系原型，逐位兼容）。
- 拟合侧对 `ld` 槽位 0 做 `−κ·5 m` 曲率修正（fix-3 §4.2 口径）；与运行时 v5 `lane.heading_err`
  存在口径差（同点 vs 5 m 点），已由修正压缩到数度以内（记录，未单独消融）。
- 数学注：`(ds,dθ)` 是车体量，对整条路径旋转/平移不变——lane 帧的唯一作用就是上述航向剖面
  修正（fix-3 §4.3）；车道系变换往返测试锁定（误差 < 3e-8）。

### 1.4 WTA 与损失

- **分配**：形状空间 = lane 帧累积 dθ；逐 step 有效掩码加权 RMS 最近锚（`assign_anchors`）。
- **选择 CE**：`CE(logits/τ, index)`（行权重 × 行有效；NaN 尾行 mask）。
- **WTA 回归**：被分配锚的 `Σ(anchor+residual)` 链 vs 教师链（逐 step mask）；速度头与残差头
  同梯度。另保留软混合端到端链损失（`weighted_action_chain_loss` 作用在最终 `plan` 尾段）。
- **接线**：
  - Stage B（BC）：`stages.B.bc.anchor_ce_weight` / `anchor_wta_weight`（默认 0 = 旧行为）；
    arm 定义 = 1.0 / 1.0；
  - phase3：`losses.anchor_ce/anchor_wta`（默认 0）；`freeze=specific_only` 下锚头随 plan head
    主干冻结 → **自动降级为 0**（`phase3_effective_config`，清单仅在实际启用时列出）；
  - Stage C：`trainable_scope=design` allowlist 不含锚头 → plan head 全冻结；PPO 不改。

---

## 2. 改动清单（file-level）

| 文件 | 改动 |
|---|---|
| `net/anchor.py`（新） | 锚字典加载/校验、内置默认、lane 帧变换（`lane_follow_dtheta/to_lane/from_lane`）、WTA 分配、软混合/指派计划（纯函数） |
| `net/plan_head.py` | `num_anchors/anchor_embed_dim/temperature/hard` 参数；选择头/速度头/残差头/锚 buffer；`set_anchors/set_anchor_mode/plan_anchors`（关闭时零新键） |
| `net/model.py` | `DrivingModel(num_anchors, anchor_path, anchor_temperature, anchor_hard)`；`_struct_context_tokens` 返回 lane 原始块；`encode` 计算锚计划；`_rollout` 尾段锚驱动；`forward` 输出 `anchor_*` |
| `pipeline/trainer.py` | `BCConfig/Phase3Config.anchor_*`；`anchor_wta_loss`（CE+WTA+诊断）；`pretrain_bc`/`evaluate_bc`/`_phase3_loss_terms`/phase3 训练循环/`evaluate_bc_phase3` 接线与缩放；`phase3_effective_config` 降级 |
| `pipeline/stages.py` | `build_model` 读 `plan_anchor`；Stage B / phase3 权重透传；`frozen_loss_keys` 含实际启用的锚项 |
| `config/model.yaml` | `plan_anchor`（enabled=false 默认、K=6、path、τ、hard） |
| `config/train.yaml` | `stages.B.bc.anchor_ce_weight/anchor_wta_weight/anchor_temperature` + `phase3.losses.anchor_ce/anchor_wta`（默认 0） |
| `config/plan_anchors_k6.json`（新） | K=6 锚字典（拟合产物，含 provenance/sha） |
| `tools/fit_plan_anchors.py`（新） | 锚拟合脚本（可复现） |
| `tests/test_plan_anchor.py`（新） | 18 项测试 |
| `docs/v7_program_prereg.md` | §11 预注册 + §8 修订锚行 + 变更记录 |

---

## 3. 测试

### 3.1 新增 `tests/test_plan_anchor.py`（18 项）

- 锚加载/形状/内置回退/非法拒绝；WTA 最近锚 + mask + margin；软混合 sharp τ/argmax/等权均值/
  指派计划；lane 帧往返 + 无效恒等 + 航向/曲率语义；模型集成（输出形状、`plan[:,0]==action_mu`、
  尾段 = 锚计划、关闭时无新键、锚模块不影响 policy 输出、零残差 = 变换后锚混合）；
  损失（CE 下降 + 指派正确 + 梯度流、NaN mask、phase3 降级开关）；PPO 兼容
  （`sample/logprob/entropy/KL` 形状与重算一致）。

### 3.2 全量 pytest

```
tools/venv-python -m pytest tests/ -q  →  677 passed（659 → +18），~40 s
```

### 3.3 端到端接线复核（CPU，真实 v5 数据 shard）

- Stage B（`pretrain_bc`）开锚：`bc_anchor_loss/ce/wta/assign_frac_*` 指标出现且有限；
  权重 0 时**无 anchor 指标键**（旧 metrics 逐位兼容）。
- phase3（`pretrain_bc_phase3`）开锚：ce/wta 与逐簇占比出现；权重 0 时旧键集合不变；
  `specific_only` 降级清单 = WM 5 项 + anchor 2 项。

---

## 4. GPU 冒烟（RTX 4070，free 11.2 GiB）

脚本 `/tmp/opencode/v7_struct_b_smoke.py`（不入 repo），真实 v5 expert shard 32 行：

| 项 | 读数 |
|---|---|
| 前向（`rollout=True`，锚开） | 全部输出 finite；`plan[:,0]==action_mu`；`plan[:,1:]==anchor_plan[:,1:]` |
| 损失/反向 | `action=1.19 · chain=0.22 · ce=1.79 · wta=0.22`；梯度全 finite；锚头梯度 L1=62.2 |
| 锚诊断 | `assign_dist=0.023 · margin=0.021 · plan_ade=0.657 m · frac=[.66,.13,.06,.03,.06,.06]` |
| PPO cheap path | `sample/logprob/entropy/KL` finite；`logprob_from_action` 重算 max|Δ|=9.5e-7 |
| phase3 降级 | all=()；specific_only=('ego_next','od','ld','presence','entry','anchor_ce','anchor_wta') |
| 资源 | 参数 0.767M · fwd 237 ms · bwd 24 ms · VRAM peak 787 MB · 总耗时 2.2 s |

**结论：1 批前向/反向无 NaN，远低于 15 min 预算。**

---

## 5. 未决 / 边界（供编排决策）

1. **arm 冻结策略**：§11 冻结 Stage B 训练锚头（CE=WTA=1.0）+ phase3 `specific_only`（锚损失
   自动降级）+ Stage C design 冻结。phase3 `freeze=all` 下训练锚头（含 chain 损失进锚计划）
   未纳入本臂（历史 all 解冻两次崩解，需单独预注册）。
2. **step0 契约**：`plan[:,0]` 恒 = `action_mu`；锚混合的第 0 步输出仅作辅助训练目标、不写入
   plan（`build_eval_references` 亦强制 `plan[0]=mu`）。若要"整条 plan 都由锚混合给出"需改输出
   契约 + 评测口径，属后续迭代。
3. **零初始化上游梯度延迟**：选择头/速度头/残差头末层零初始化 ⇒ 首步上游（trunk/embed）
   无梯度（与 policy `mu` 同模式）；末层更新后恢复。BC 前 1–2 batch 属正常现象。
4. **口径差**：锚在 v41 + `ld` slot0（−κ·5m 修正）上拟合，运行时用 v5 `lane.heading_err`；
   已记录，未做消融。换教师（IDM/RL）需重拟合（fix-3 风险④）。
5. **选择头上限**：fix-5 探针（v4 时代 ckpt、latent-only）expert balanced acc 0.47–0.53、
   失败窗 0.31–0.35（乐观口径）；本实现的注意力 token（lane/ttc）增量未测。
6. **评测模型构建**：eval worker 经 `build_model(model_config)` 构造；run 的 `model_config`
   快照需含 `plan_anchor` 块，否则评测模型无锚头（`_filter_checkpoint_state` 会把锚权重记为
   unexpected 并丢弃）。arm 驱动需确保配置透传。
7. **resume 边界**：`load_checkpoint`（阶段 A→B 初始化）在 ckpt 缺锚键时自动 strict=False 回退；
   `load_training_checkpoint`（`--resume`）仍是严格加载 → 开锚后 resume 需用同结构 ckpt。
8. **tollgate 0/45 不承诺**：fix-3 §6 风险①（锚是表示不是修复；可能同时是跟踪/地图几何问题）；
   §11 只设表示层主判据 + 闭环描述性方向。
9. **`hard=true` 推理路径**只有单元测试覆盖（one-hot 几何），未做闭环消融。

---

## 6. 复现

```bash
# 锚拟合（~7 s CPU）
tools/venv-python tools/fit_plan_anchors.py \
    --dataset datasets/BTC20261002-0941_expert5k_v41/expert_bc.npz \
    --k 6 --out config/plan_anchors_k6.json

# 新测试 / 全量
tools/venv-python -m pytest tests/test_plan_anchor.py -q
tools/venv-python -m pytest tests/ -q

# GPU 冒烟（~2 s）
tools/venv-python /tmp/opencode/v7_struct_b_smoke.py
```
