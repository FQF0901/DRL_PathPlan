# config/clusters

场景原型聚类的**冻结 artifact 与规格**（lane C / fix-4；契约 `docs/design-v1.2.md` §4）。
聚类替代手工规则标签监督 MoE router：8 个"场景原型"簇 → top-2 softmax 软目标。
**规则标签只用于体检，绝不进入监督。**

## 文件

| 文件 | 说明 |
| --- | --- |
| `default.yaml` | 加载入口：`cluster_version` + `spec` 路径 + 软目标参数（τ/边界平滑） |
| `cluster_v1.npz` | 冻结 spec：质心 / PCA 白化参数 / 特征口径 / 数据指纹（obs_fingerprint + 行数 + 拟合时间 + git hash） |
| `cluster_v1.report.json` | 体检报告：簇规模/熵/半径、难度/几何/规则标签占比、稀有结构富集、FLAG 结论 |
| `cluster_v1_two_stage.npz/.report.json` | 两段式退路版本（活跃度 2-means + 7 簇活跃子集）的对照产物 |

> **当前产物是 v2 dev 切片占位**：128 spec / 9,509 行（`/tmp/opencode/bc_v2_dev128`，
> `obs_fingerprint=v2-58f1257a6c05`，flat 体检通过：cutout_active 富集 4.49×）。
> **全量 v2 重采完成后必须按上面的命令重拟合**，否则源数据不在仓库内、不可复现。

## 加载与推理（trainer / 评估）

```python
from pipeline.clusters import load, soft_targets_from_obs

spec = load()                                  # config/clusters/default.yaml → cluster_v1.npz（进程内缓存）
soft = soft_targets_from_obs(obs_batch)        # (B,8) float32，行和=1
# obs_batch 与 BCDataset.build_obs_batch 同口径：
#   ego(B,8)/od(B,16,9)/od_mask(B,16)/ld(B,16,7)/ld_mask(B,16)/nav(B,11)/signal(B,4)
#   v2 追加 od_presence(B,16)/others(B,28)；od 有效性优先 presence（v2 固定槽位语义）
# torch 输入 → torch 输出（同 device）；numpy → numpy。
```

## 拟合 / 重拟合（v2 数据就绪后全量执行）

```bash
# 全量拟合（v2 数据集；`others` 为规范 28 维输入）
tools/venv-python tools/fit_clusters.py --bc-dir <V2_BC_DIR> \
    --out config/clusters/cluster_v1.npz \
    --report config/clusters/cluster_v1.report.json

# 若体检报告 FLAG（无簇对稀有结构富集 ≥3×）→ 两段式退路
tools/venv-python tools/fit_clusters.py --bc-dir <V2_BC_DIR> --two-stage \
    --out config/clusters/cluster_v1.npz \
    --report config/clusters/cluster_v1.report.json

# 重聚类（改 K/参数后）并把新簇编号 Hungarian 对齐旧 spec
tools/venv-python tools/fit_clusters.py --bc-dir <V2_BC_DIR> --align-to config/clusters/cluster_v1.npz \
    --out config/clusters/cluster_v2.npz

# 可选：把软目标离线物化（trainer 在线调用 soft_targets_from_obs 亦可）
tools/venv-python tools/fit_clusters.py --bc-dir <V2_BC_DIR> --dump-targets runs/<run>/router_targets.npz
```

数据指纹纪律：`ClusterSpec.load` 校验 `feature_contract.others_source`/`raw_dim`；`opt.obs` 的
`obs_fingerprint` 变更（v1→v2）后旧 spec 与新数据特征口径不一致时，`transform` 直接报错，
必须重拟合（报告 `data.obs_fingerprint` vs `data.obs_fingerprint_current` 会记录差异）。

## 算法要点（无监督、rule-free）

1. 特征 = 当前帧 `ego(8)+od(16×9)+ld(16×7)+others(28)` 展平（无效槽位按 mask/presence 清零，
   固定槽位顺序原样保留）→ 去常数/近零方差维 → 标准化 → PCA 白化到 48 维；
2. 稀有度加权：白化空间 kNN 局部密度 ρ，`w = clip((ρ_med/ρ)^0.75, 0.1, 10)`——稀有/困难结构
   权重高，防"平淡驾驶吃掉多簇"；
3. 容量约束加权 k-means：k=8、每簇 ≤25%（逐轮贪心 + 松弛填充 + 成对交换改进）；
   体检后处理：>30% 沿第一主方向二分、<2% 并入最近簇；
4. 软目标：top-2 距离 `softmax(-d/τ)`（τ=0.5，典型 0.65/0.35）；top-2 相对间隔 <0.05 的边界
   样本 → 向 top-3/均匀平滑（β 上限 0.5）；
5. FLAG 门：任一稀有标签（全局占比 ≤5%）在某簇富集 <3× → 报告 FLAG；`--two-stage` 退路先用
   连续活跃度（OD 在场数/交互强度 + 自车机动强度）2-means 分"活跃/平稳"，7 簇只花在活跃子集。
