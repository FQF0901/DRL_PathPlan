# v7 结构迭代 A2：v5 开训前置——pipeline 接线（lane/ttc）+ 选择头可学性探针

- 日期：2026-10-05（CPU；探针只读）
- 范围：`pipeline/trainer.py`、`pipeline/buffer.py`、`tests/test_v5_pipeline_wiring.py`（新增）；
  **未碰** `net/plan_head.py` 与 plan 生成路径。
- 报告路径：`/tmp/opencode/v7_struct_a2_wiring_probe.md`；探针脚本：`/tmp/opencode/v7_struct_a2_probe.py`；
  探针原始数据：`/tmp/opencode/v7_struct_a2_probe.json` / `.log`
- 全量测试：`tools/venv-python -m pytest tests/ -q` → **659 passed**（654 → +5 新接线测试；复跑 52.9 s）
- 提交：内容 commit `fcf047e`；锚 commit `326c260`（§8 修订锚行，不改动其余内容）

---

## 1. 接线（fix-4 未决项的 3 处）

### 1.1 改动点

| # | 位置 | 改动 | 效果 |
|---|---|---|---|
| 1 | `pipeline/trainer.py::BCDataset._obs_keys` | 硬编码元组加入 `lane`（`ld` 后）与 `ttc`（`others` 后） | v5 npz 的 `lane/ttc/lane_mask/ttc_mask` 不再被过滤；`build_obs_batch`/`MaterializedBCDataset` 透传到 net |
| 2 | `pipeline/trainer.py::SINGLE_SLOT_CHANNELS` | 加入 `lane`/`ttc` | `squeeze_single_slot` 把 `(B,1,17)/(B,1,12)` → `(B,17)/(B,12)`，匹配 net `_validate_obs` / `context_features_from_obs` |
| 3 | `pipeline/buffer.py::DEFAULT_CHANNELS` | 加入 `lane (1,17)` / `ttc (1,12)`（形状经 `_struct_dims()` 懒加载 `env.obs.lane.LANE_DIM`/`env.obs.ttc.TTC_DIM`，缺 env 时退回 17/12） | PPO `RolloutBuffer` 默认通道含新键；`add_step` 缺键填 0 + mask=0 |

顺序口径：与 `tools/collect_expert.py::CURRENT_CHANNELS` 和 `env/obs/builder.DEFAULT_CHANNELS` 一致
（`ego, od, ld, lane, nav, signal, others, ttc, ego_world, route_world`；lane 与 LD 主块并列、
ttc 与 nav/others 同组），即 net `_head_tokens` 令牌序 `[OD16, LD16, lane, others, ego, nav, ttc, signal, latent]`
中两个新 token 的语义位置。`lane/ttc` **不进 6 帧历史**（`DEFAULT_HISTORY_CHANNELS`/`pipeline/frames.HISTORY_CHANNELS`
保持 `ego/others/od/ld`，与 v5 设计一致）。

向后兼容：`_obs_keys`/`build_obs_batch` 均为「键存在才读取」；旧 schema（<v5）数据无新键 →
键集合与旧行为逐位相同，net 走 `context_features_from_obs` 缺键回退（0 token + mask=0 + 一次性警告），
回退路径**未改动**。

### 1.2 测试（`tests/test_v5_pipeline_wiring.py`，5 项）

1. `test_v5_dataset_roundtrip_reads_lane_ttc`：v5 npz（schema=5）→ `BCDataset.load` →
   `_obs_keys` 含 lane/ttc 且顺序正确 → `build_obs_batch` 形状 `(B,17)/(B,12)` + mask `(B,1)` + **逐值往返**；
   `MaterializedBCDataset`（训练实际消费的快路径）不丢新键；
2. `test_v4_dataset_without_struct_keys_still_loads`：无 lane/ttc 的旧数据仍可加载，
   batch 无新键、既有通道形状不变（回退路径不变）；
3. `test_single_slot_channels_include_lane_ttc`：`(B,1,F) → (B,F)` 挤压，多槽通道不动；
4. `test_rollout_buffer_default_channels_include_lane_ttc`：默认形状与 env `LANE_DIM/TTC_DIM` 一致；
   有键透传、缺键填 0 + mask=0；
5. `test_v5_batch_feeds_net_lane_ttc_tokens`：v5 batch → `DrivingModel.encode` → tokens `(B,39,H)`，
   `key_mask[:,32]`（lane）与 `key_mask[:,36]`（ttc）全 1 —— **证明接线生效而非缺键回退**。

### 1.3 真实 v5 数据复核（接线后，只读）

用 obs v5 冒烟采集产物 `/tmp/opencode/struct_a_collect`（84 行，fingerprint `v5-190fac36e6be`）：

```
obs_keys = [ego, od, ld, lane, nav, signal, others, ttc, ego_world, route_world]
batch: lane (84,17) / ttc (84,12) / lane_mask (84,1) / ttc_mask (84,1)，全有限，mask 全 1
DrivingModel.encode → tokens (84,39,32)，lane key_mask=1、ttc key_mask=1
```

### 1.4 风险 / 边界

- `SINGLE_SLOT_CHANNELS` 只影响挤压逻辑；`lane/ttc` 已是单槽（1×17/1×12），无槽位维歧义。
- PPO collect 的 `channels` 由实际 obs 模板生成（`_current_template`），接线后 v5 环境自动带上
  lane/ttc；`DEFAULT_CHANNELS` 只是无模板时的兜底形状。
- `pipeline/stages.py::_REPLAY_CHANNELS`（ego/od/ld/nav/signal）仅用于 `FrameWindows.build_future`
  的未来目标查表（OD/LD/presence/entry/ego_fut），不含 lane/ttc 属预期（lane/ttc 是当前帧上下文，
  无未来目标语义）；`FrameWindows.build_obs` 全仓无调用点。故本次未改 `stages.py`。
- 训练前仍需**重新采集 v5 数据**（旧 v4 数据缺 lane/ttc，接线只保证「新键不被丢弃」，不会凭空补值；
  `obs_fingerprint` 不匹配警告照旧）。

---

## 2. 选择头可学性探针（K=6 形状锚 × 现有模型 t0 融合 latent）

### 2.1 设置

- **锚字典（复现 fix-3）**：expert5k_v41，`tw>0` 且 action 链有限（265,742 行）；moving 过滤
  `L>=1m`；120k 子样；`cumdtheta = cumsum(dθ[0:6])`；`KMeans(K=6, n_init=10, random_state=0)`。
  复现簇大小 = **[92519, 7823, 3012, 8461, 4515, 3670]**，与 `kanchor_shape_results.json`
  逐位一致（锚字典与 fix-3 同源）。
- **latent**：`runs/BTC20261002-2329_v7p1dagger_w4/final.pt`（epoch 15，最新 BC 策略快照，
  schema v4）→ `DrivingModel.encode(obs)["latent"]`（plan head 融合输出，H=128，t0）。
  该 ckpt 早于 v5：`encoders.type_embed` 形状 5→7、`encoders.lane/ttc` 为新增头（探针加载时
  按形状过滤跳过，仅 `type_embed` 1 项；latent 路径不消费 lane/ttc，无影响）。
- **探针数据**（obs 侧，逐行取 t0）：
  1. `expert500val_v41`（36,148 行；与锚拟合集场景不重叠的 expert 留出集；合格 26,474 行）；
  2. `v7p1dagger_w4`（20,049 失败窗口；**该 ckpt 训练集含这些行** → 记为乐观上界）。
- **探针**：StandardScaler + LogisticRegression（线性）/ MLP(64, early stopping)；两口径切分：
  - `episode_split`：按 episode 70/30（跨场景泛化，**主口径**）；
  - `row_split`：行随机 70/30（同期自相关下乐观上界）。
- **基线**：chance = 1/6 ≈ 0.167；多数类（val 内 straight 簇）；nav 命令条件多数类
  （train 上按 nav 0/1/2 拟合 argmax 锚，val 评估 = 「只用 nav」参照）。

### 2.2 读数

**expert500val_v41（26,474 合格行；锚分布 [20585, 1587, 656, 1823, 973, 850]）**

| 口径 | 探针 | acc | balanced acc | macro-F1 | 备注 |
|---|---|---|---|---|---|
| episode_split（主） | linear | 0.841 | 0.473 | 0.535 | 多数类 0.747（= 全 C0）· chance 0.167 |
| episode_split（主） | MLP(64) | **0.855** | **0.528** | **0.592** | 多数类 macro-F1 ≈ 0.14 |
| row_split | linear | 0.866 | 0.499 | 0.560 | 多数类 0.782 |
| row_split | MLP(64) | 0.880 | 0.561 | 0.617 | — |

- 逐类召回（episode_split/MLP）：C0 直行 0.983 · C1 右弯 0.792 · C2 左转回正 0.456 ·
  C3 缓右 0.435 · C4 左弯 0.223 · C5 环岛（先左后回正）0.279。
- nav 条件多数类基线 = 0.747（= 多数类）：expert 上每个 nav 命令内 C0 都占优
  （nav 表三行 argmax 均 C0）→ **仅 nav 命令无法解释锚选择，latent 携带额外信号**。

**v7p1dagger_w4（15,006 合格行；锚分布 [1813, 5130, 3939, 2071, 1143, 910]）**

| 口径 | 探针 | acc | balanced acc | macro-F1 | 备注 |
|---|---|---|---|---|---|
| episode_split（主） | linear | 0.488 | 0.311 | 0.302 | 多数类 0.362（C1 右弯）· chance 0.167 |
| episode_split（主） | MLP(64) | 0.507 | 0.354 | 0.354 | nav-only 0.401 |
| row_split | linear | 0.497 | 0.314 | 0.300 | 多数类 0.349 |
| row_split | MLP(64) | **0.538** | **0.377** | **0.379** | nav-only 0.406 |

- 逐类召回（episode_split/MLP）：C1 0.720 · C2 0.693（**tollgate/merge 失败的主导模式**，可判）·
  C0 0.255 · C3 0.115 · C4 0.184 · C5 0.156。
- nav 条件基线 0.401 > 多数类 0.362（失败窗口里 nav 有信息：左/直行命令大量落 C2），
  但探针仍显著高于 nav-only。

### 2.3 结论（「从 obs 选锚」信号可学性）

1. **可学，且不是"直行先验"**：留出 expert 上线性探针 balanced acc 0.473 / macro-F1 0.535，
   MLP 0.528 / 0.592，均远高于 chance（0.167）与多数类基线（acc 0.747、macro-F1 ≈0.14）；
   即 **t0 融合 latent 已线性可解码形状锚**。MLP 稳定优于线性（macro-F1 +0.04~0.06）→
   选择头宜用轻量非线性（MLP/注意力），纯线性不是上限。
2. **失败窗口（选择头的目标分布）信号较弱但真实**：dagger 上 balanced acc 0.31→0.35、
   macro-F1 0.30→0.38，约为 chance 的 2 倍并高于多数类/nav-only；且这些行**在模型训练集内**
   （乐观口径）仍只有此水平 → 失败窗口的"模式选择"本身噪声大（与 fix-3 的 dagger silhouette
   0.32–0.45 一致），选择头收益应主要来自 tollgate/merge 的 C1/C2 判别（召回 0.69–0.75）。
3. **类别不均是主要瓶颈**：expert 上 C4/C5 召回仅 0.22/0.28、C2 0.46；训练选择头需要
   类平衡采样/权重（fix-3 已提示 straight 占 77%）。
4. **上界口径注意**：本探针用 v4 时代 ckpt，lane/ttc 缺键回退且 latent 路径不消费它们；
   v5 的 lane/ttc 进的是注意力令牌集合（`net.model._head_tokens`）。若选择头从 **token/注意力输出**
   读取（而非仅 `latent`），v5 新增的 lane（d_lat/航向误差）与 ttc 可能带来本探针未测到的增量；
   若选择头仅接 `latent`，本读数即为近似上界（线性 +0.04~0.06 的非线性余量）。

---

## 3. 后续 / 建议

1. **开训顺序不变**：接线已就位，重采 v5 数据后 Stage A/B 可直接开训（v5 npz 的 lane/ttc
   会进入 batch 与 PPO 缓冲；旧 v4 数据仍按回退路径工作）。
2. **选择头形态（fix-3 方案 A 的输入侧）**：探针支持「轻量 MLP/注意力头 + 类平衡」而非纯线性；
   锚 CE 预训练可行，但需监控 C2/C4/C5 的召回与负载均衡（straight 先验坍缩风险）。
3. **v5 增量验证**：重采数据后可在**同一探针**上换成 v5 模型 + 含 lane/ttc 的 batch，
   对照 latent 探针读数是否提升；若选择头计划读 token/注意力输出，建议另做 token 级探针。
4. **锚字典**：本探针复现 fix-3 的 expert `cumdtheta` K=6（簇大小逐位一致），可直接作为
   选择头的初始锚定义；速度维度继续由现有连续头承担（fix-3 因子化结论）。
