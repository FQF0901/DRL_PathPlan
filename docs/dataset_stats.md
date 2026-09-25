# 数据集场景分布统计（2026-09-25）

## train (n=10000)

| 维度 | 分布 |
|---|---|
| 难度 | easy 3630 (36.3%), hard 2876 (28.8%), medium 3494 (34.9%) |
| 几何（spec 去重） | straight 10000 (100.0%), curve 5323 (53.2%), ramp_out 1212 (12.1%), intersection 1071 (10.7%), roundabout 1050 (10.5%), uturn 909 (9.1%), t_intersection 909 (9.1%), ramp_in 909 (9.1%), tollgate 909 (9.1%), split 909 (9.1%), merge 909 (9.1%) |
| 几何（block 计数） | straight 16991, curve 5924, ramp_out 1212, intersection 1071, roundabout 1050, uturn 909, t_intersection 909, ramp_in 909, tollgate 909, split 909, merge 909 |
| 交通形态 | cut_in 5048 (50.5%), crowded 3877 (38.8%), cut_out 3826 (38.3%), lane_change 3242 (32.4%), zipper_merge 630 (6.3%) |
| 脚本事件 | cut_in 5889, cut_out 3826 |
| 多几何场景 | 9552 (95.5%) |
| 地图块数 | 3块 8922 (89.2%), 4块 454 (4.5%), 5块 624 (6.2%) |

## val (n=1000)

| 维度 | 分布 |
|---|---|
| 难度 | easy 372 (37.2%), hard 290 (29.0%), medium 338 (33.8%) |
| 几何（spec 去重） | straight 1000 (100.0%), curve 522 (52.2%), ramp_out 120 (12.0%), intersection 107 (10.7%), roundabout 104 (10.4%), t_intersection 91 (9.1%), uturn 91 (9.1%), split 91 (9.1%), ramp_in 91 (9.1%), merge 91 (9.1%), tollgate 90 (9.0%) |
| 几何（block 计数） | straight 1699, curve 583, ramp_out 120, intersection 107, roundabout 104, t_intersection 91, uturn 91, split 91, ramp_in 91, merge 91, tollgate 90 |
| 交通形态 | cut_in 513 (51.3%), crowded 393 (39.3%), cut_out 378 (37.8%), lane_change 320 (32.0%), zipper_merge 56 (5.6%) |
| 脚本事件 | cut_in 581, cut_out 378 |
| 多几何场景 | 955 (95.5%) |
| 地图块数 | 3块 901 (90.1%), 4块 40 (4.0%), 5块 59 (5.9%) |

## BC 专家数据集（2026-09-25）

| 数据集 | 场景数 | 样本数 | 产出率 | 备注 |
| --- | --- | --- | --- | --- |
| `runs/bc_expert_full` | 200 | 10,777 | 0.727 | 早期版本（旧观测 scope 重采版）|
| `runs/bc_expert_2k` | 2,000 | **103,938** | 0.718 | 当前训练用；6 workers 并行采集（输出与单进程逐字节一致）|

- 过滤规则（2k 采集）：`roundtrip_fail` 28,982 + `terminal_window` 11,865；候选 144,785 → 保留 103,938。
- 标签分布（逐步标签，下限 50）：cutin_active 1,777 / cutout_active 3,212 / crowded 15,892 /
  car_following 9,961 / on_curve 8,453 / merging 19,388 / roundabout_near 2,794 / near_intersection 11,237。
- 配平：难度 × 主标签共 30 组，采样权重范围 0.62–3.69。
- **版本纪律**：`expert_bc.meta.json` 记录 `obs_fingerprint`；观测 scope 变更后必须重采，`BCDataset.load`
  在不匹配时告警。
- 采集命令：
  ```bash
  tools/venv-python tools/collect_expert.py --specs env/specs/scenarios_train.json \
      --limit 2000 --out runs/bc_expert_2k --workers 6
  ```
