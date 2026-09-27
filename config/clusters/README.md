# config/clusters

场景原型聚类的**冻结 artifact 与规格**（lane D；契约 `docs/design-v1.2.md` §4）。
聚类替代手工规则标签监督 MoE router：8 个"场景原型"簇 → **硬标签**（top-1）CE；
**规则标签只用于体检，绝不进入监督。**

## 文件

| 文件 | 说明 |
| --- | --- |
| `default.yaml` | 加载入口：`cluster_version` + `spec` 路径 + 软目标参数（τ/边界平滑） |
| `cluster_v2.npz` | **当前冻结 spec（lane D）**：特征契约 v2（47 维）/ PCA 白化 32 / 质心 / 数据指纹（obs_fingerprint + 行数 + 拟合时间 + git hash + fit_params） |
| `cluster_v2.report.json` | 体检报告：簇规模/熵/半径、`capacity_ok`（cap 15% / floor 8%）、难度/几何/规则标签占比、稀有富集、soft gap，**含与 v1 的对照** |
| `cluster_v1.npz` / `cluster_v1.report.json` | **历史 v1**（292 维特征、cap 25%/floor 2%）：全量 c0=84.7%、`capacity_ok=False` → router 退化；保留仅供旧 run 复现，不再使用 |
| `cluster_v1_two_stage.*` | 历史两段式对照产物 |

## 特征契约 v2（`pipeline/clusters.py::FEATURE_CONTRACT_V2`，47 维）

- `ego(8)` 全保留；
- `od`：每帧 6 对象 × `(dx, dy, vx, vy)`：
  - 候选 = `od_presence == 1`；
  - 先取 **TTC 最小 3 个**：`urgency = min(TTC, 5 s)`，`TTC = dx/(-vx)`（`dx>0 且 vx<-1e-3`，否则 inf）
    —— 与 `env/obs/od.py` 槽位分配同公式、同参数（本模块内置向量化实现）；
  - 再从其余候选取距离（`√(dx²+dy²)`）最近 3 个；
  - 不足 6 个零填充；并列按槽位下标升序；顺序 = TTC-3（urgency 升序）→ 距离-3（dist 升序）；
- `nav`：命令 one-hot(3)（forward/left/right；checkpoints/reserved/route_completion 丢弃）；
- `road_class`：one-hot(12)；**丢弃** ld、od cos/sin/length/width/type_id、speed_limit、signal。

## 加载与推理（trainer / annotate）

```python
from pipeline.clusters import load, soft_targets_from_obs, od_top6_indices

spec = load()                                  # default.yaml → cluster_v2.npz（进程内缓存）
soft = soft_targets_from_obs(obs_batch)        # (B,8) float32，行和=1（annotate 取 argmax = 硬标签）
labels = spec.assign_hard(raw)                 # 原始特征 (N,47) → 硬标签（与 soft argmax 同族）
# obs_batch 与 BCDataset.build_obs_batch 同口径：
#   ego(B,8)/od(B,16,9)/od_presence(B,16)/nav(B,1,11)/others(B,1,28)（v2 契约要求全键）
```

## 拟合 / 重拟合（lane D 全量命令）

```bash
# 变体（a，默认）：flat + 严格容量（cap 15% / floor 8%）+ 8 restarts
tools/venv-python tools/fit_clusters.py --bc-dir datasets/BTC20260926-2343_expert5k \
    --cluster-version v2 --restarts 8 \
    --out config/clusters/cluster_v2.npz --report config/clusters/cluster_v2.report.json

# 变体（b）：活跃度分桶 + 两桶都做容量约束 k-means（仍是 flat spec；对照用）
tools/venv-python tools/fit_clusters.py --bc-dir datasets/BTC20260926-2343_expert5k \
    --cluster-version v2 --balanced-split --two-stage-max-bucket 0.5 \
    --out /tmp/cluster_v2_b.npz --report /tmp/cluster_v2_b.report.json

# sidecar（Stage B 只读；采集收尾会自动调用一次）
tools/venv-python tools/annotate_clusters.py --dataset datasets/BTC20260926-2343_expert5k \
    --cluster-config config/clusters/default.yaml

# 对照排查图（每簇 100 帧 + 占比 + summary）
tools/venv-python tools/diagnostics/cluster_survey.py --spec config/clusters/cluster_v2.npz \
    --out runs/BTC<北京戳>_cluster_survey_v2
```

数据指纹纪律：`ClusterSpec.load` 校验 `feature_contract`/`raw_dim`；数据集 `obs_fingerprint` 与 spec
拟合指纹不一致时 `transform` 直接报错，必须重拟合；sidecar 的 `spec_hash`/`rows`/`obs_fingerprint_dataset`
由 `pipeline.clusters.load_assignments` 严格校验（不符 → fail-fast 并打印生成命令）。

## 算法要点（无监督、rule-free）

1. 特征 = 契约 v2（见上）→ 去常数/近零方差维 → 标准化 → **PCA 白化 `min(32, 去常量后维数, N-1)`**；
2. 稀有度加权：白化空间 kNN 局部密度 ρ，`w = clip((ρ_med/ρ)^0.75, 0.1, 10)`——稀有/困难结构
   权重高，防"平淡驾驶吃掉多簇"；
3. 容量约束加权 k-means：k=8、**每簇 ≤15%（floor 8%）**，多次 restart 取满足约束且 objective 最低；
   变体（b）先按活跃度分桶（每桶 ≤50%）再在桶内做容量约束 k-means；
4. 软目标：top-2 距离 `softmax(-d/τ)`（τ=0.5）；边界样本向 top-3/均匀平滑（`boundary_margin=0.05`）；
   **训练侧只用其 argmax 硬标签**（lane B B3）；
5. 体检门：`capacity_ok=True`（cap/floor）；稀有标签（全局占比 ≤5%）富集 ≥3×，否则 FLAG 并给两段式命令。
