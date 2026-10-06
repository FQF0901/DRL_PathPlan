# curve / roundabout / uturn / tollgate 四类 0% 的根因诊断（只读；未改任何行为代码）

输入（历史产物，已清理）：`runs/eval/il_v2_fixed_lqr50/`（50 条冻结 val slice，ckpt=`runs/train/il_v2_10x10_b_fixed/stage_b/final.pt`，tracker=lqr，H=128）
新增（只读脚本）：`tools/diagnostics/forensics_offline.py`、`tools/diagnostics/forensics_closed_loop.py`、`tools/diagnostics/forensics_report.py`
产物（历史产物，已随 `runs/` 清理；可由文末命令重生成）：`runs/forensics/{offline_counts,offline_clusters,offline_plan,closed_lqr,closed_exact,closed_baseline,closed_oracle,closed_gain2,closed_gain4}.json`
## 结论（按证据强度）

1. **[事实·强] 主瓶颈是"预瞄/plan 不是一条能回到车道的路径"，不是 tracker、不是数据量。**
   - 把策略自己的 plan 用 ExactTracker **完美执行**（0 跟踪误差），同 17 条 spec 仍 17/17 失败（10 out_of_road + 7 collision，rc_mean 0.443）。
   - 同一个 LqrTracker，把参考换成规则专家（PurePursuitIDM）实测轨迹 → 6/17 arrive_dest、7/17 到 rc 0.72–0.98（仅超 100 s 上限）、rc_mean 0.805。
   - plan 锚在**自车当前位姿**（`interpolate` 以 base_pose 为原点），闭环里**没有任何车道中心反馈**；LQR 只是忠实跟随 plan（plan→执行横向偏差 focus 0.081 m vs easy 0.110 m），所以累积横偏不会回收。
2. **[事实·强] 失败形态 = 横向漂移出界**：LQR 失败前 1 s 内 |lane_lat| 峰值：curve 0.92 / roundabout 0.58 / uturn 0.55 / tollgate 0.75 m（baseline 同场景 0.07 m）；漂移从**直道 crawl 段**（0.8–2 m/s 持续 10–20 s，专家 3.2–3.9 m/0.5 s）就开始。
3. **[事实·强] plan 的"转向通道"在大转角上塌缩**：离线（训练分布内、24k 行、按 train_weight 加权）`plan[:,k].dtheta` 与专家 dθ 相关性 0.31–0.51、过原点斜率 0.14–0.27；
   |专家 dθ|≥0.1 rad/0.5s（4.1–5.5% 行）plan/expert = 0.05–0.17；≥0.2 rad（0.3–1.4%，uturn/环岛级）≈0.05。
   同一份前向里 `ds`（速度）通道正常：corr 0.88–0.96、斜率≈1.01（不是"plan 全废"，是转向通道废）。
4. **[事实·中] 简单放大转向通道**（纯诊断：`plan[:,1]*=g` 后再给同一个 LQR，g=2/4）**不能修复**：仍 17/17 出界（rc_mean 0.35/0.33）→ 说明缺的是"绝对路径锚点 + 偏差回收"，不是纯幅度。
5. **[非因] tracker 增益/曲率能力不足**：跟踪 plan 的 e_y p95 = 0.46 m（focus）vs 0.57 m（easy）——失败类反而更小；合成圆弧实验（R=60/25 m，4–8 m/s）瞬态横向误差 0.21–0.44 / 0.63–0.87 m，而这些 block 实测曲率 0.0157–0.0176（R≈57–64 m）→ 够用。
6. **[非因] 数据量**：四类 trainable 行数 7534–11284（占 7.3–10.9%），与 straight（9723, 9.4%）同级；簇空间无类别专属簇（详情见 Q4）。
7. **[次因] 纵向**：LQR 实测 v/参考 v 中位数 0.807（p10 0.53）——P-only 稳态差（oracle 组 7/17 因此超时）；plan 的 ds 预测本身没问题。
8. **[次因·仅 tollgate]** 2/4 是**撞岗亭**（rc 0.93/0.55，'$' 内），且 oracle 回放同样在 rc≈0.53 撞 2 次 → 除 plan 外还缺"过闸"行为/避障（专家实现里有专门的 tollgate 扫描逻辑，`env/expert/pure_pursuit_idm.py:572,777`）。
## Q1 失败形态（`runs/forensics/closed_lqr.json`，历史产物已清理、可由文末命令重生成；20 条重跑与 `episodes.csv` 的 termination/rc/steps 逐条一致）

| class | n | 冻结 eval 终止 | rc@fail | 失败所在 block | 失败前1s均速(m/s) | max&#124;lane_lat&#124;(m) | tracker e_y p95(m) |
|---|---|---|---|---|---|---|---|
| curve | 5 | out_of_road×5（0/5 成功） | 0.151–0.419（均值 .241） | C×4, I×1 | 4.58 | 0.92 | 0.49 |
| roundabout | 4 | out_of_road×3 + max_step×1 | 0.316–0.967 | O×3, S×1 | 3.88 | 0.58 | 0.59 |
| uturn | 4 | out_of_road×4 | 0.194–0.710 | U×2, C×1, S×1 | 2.70 | 0.55 | 0.41 |
| tollgate | 4 | collision×2 + out_of_road×2 | 0.405–0.934 | $×2, S×2 | 3.38 | 0.75 | 0.36 |

对照组（同 17 条 spec）：baseline 13/17 arrive；ckpt+lqr 0/17；ckpt+exact 0/17；lqr+专家路径 **6 arrive + 7 rc≥0.72**。
KPI 口径提醒：`episodes.csv::mean_speed_mps` 实为**末步速度**（`eval_runner.py:966` 只取 `info['velocity']`）；ckpt 路径的 `steer_abs_mean/throttle_mean` 恒为 0（`_CkptController._action` 从不更新）。本报告的速度/横向量均来自重跑记录。
## Q2 预瞄保真（WM/plan 侧）

- 逐 horizon `traj_xy` vs `traj6` 加权 MAE（m，`tools/venv-python tools/diagnostics/forensics_offline.py --section plan`）：
  curve .19/.44/.75/1.09/1.46/1.89；roundabout .14/.34/.56/.77/.99/1.25；uturn .15/.38/.65/.95/1.28/1.67；tollgate .19/.47/.83/1.21/1.63/2.10；straight .18/.42/.74/1.07/1.42/1.83
  → **四类并不比 straight 差**（roundabout/uturn 更好）；问题不在"整体预测精度"，而在**条件于大转角时的转向响应**（见结论 3）。
- 闭环"plan vs 实际执行"偏差（每 0.5 s 窗口最大横向 / 航向）：focus 0.081 m / 0.041 rad，easy 0.110 m / 0.058 rad → 执行保真不是瓶颈。
## Q3 跟踪器（执行侧）

- 跟随误差（同 plan、同场景）：focus e_y p95 0.46 / e_ψ p95 0.069；easy 0.57 / 0.084 → **无类别差异**。
- 曲率可行域（合成圆弧 + 真实 LQR，`--mode arc`，id 0/34，v=4/6/8 m/s）：直线 0.001 m；R=60 m 瞬态 0.21–0.44 m；R=25 m 0.63–0.87 m → R≥60（本批 block 实际 R≈57–64 m）够用；R≤25 会额外贡献 ~0.5–0.9 m（次要）。
- 纵向：v/ref 中位 0.807（p10 0.53，n=6013 step）→ P-only 稳态差，独立于本次 0% 问题。
## Q4 数据覆盖（`runs/bc_expert_2k_v2/report.json` + `expert_bc.npz`，历史数据集已清理、可重采）

| geometry | trainable rows | 占比 | |dθ|≥0.1 rad 行 | ≥0.2 rad 行 | |dθ| 均值(rad/0.5s) | stage B 加权动作误差 |
|---|---|---|---|---|---|---|
| curve | 11284 | 10.9% | 4.3% | 0.5% | 0.020 | （on_curve）0.156 |
| roundabout | 8779 | 8.5% | 5.5% | 1.4% | 0.021 | （roundabout_near）0.099 |
| uturn | 7534 | 7.3% | 4.4% | 0.9% | 0.016 | 无对应标签 |
| tollgate | 8621 | 8.3% | 4.1% | 1.1% | 0.015 | 无对应标签 |
| straight | 9723 | 9.4% | 3.1% | 0.5% | 0.014 | （near_intersection）0.109 |

- **"样本不足"不成立**（行数/加权同级）；但**转弯关键帧是薄尾**（0.3–1.4%），且训练 loss 看不到该缺陷（总体加权动作误差 0.120，on_curve 0.156）。
- 簇空间（`soft_targets_from_obs`，8 簇）：无类别专属簇；roundabout 偏 cluster2 27%/cluster4 22%，uturn/tollgate/curve 最大簇占比仅 16–20%，top1 软权重 0.58–0.62（低置信）→ router 不区分类别（MoE 只在 plan head 内，因此对四类无专门专家）。
- 监督标签只有 `on_curve / roundabout_near / near_intersection`，**没有 uturn / tollgate / 环岛转弯**标签；`others` 的 road_class one-hot 虽含 uturn/tollgate，但只是输入。
## Q5 下一步（ROI 排序）与 ≤1 h 最小验证实验

1. **给 plan 一个"道路绝对锚点 + 横偏回收"**（主因；也可在评测/执行侧先验证）
   实验 E1（10 min，已跑完作为判据）：LQR 参考=专家实测路径 → 6 arrive + 7 rc≥0.72（对照 0/17）。成功判据：≥8/17 arrive 或 ≥12/17 rc≥0.9。
2. **训练侧让 plan 学会"从偏移位姿回到专家路径"**（与 1 同源）：在数据上做横向/航向扰动增广（当前位姿摄动、目标仍为原专家路径），或把 plan 目标改为"相对车道中心线的修正量"。
   实验 E2（≤30 min）：微调 stage_b/final.pt 1–2 epoch（stage B 全量 10 epoch 实测 ≤9 min）→ 判据：离线 `strong_turn plan/expert ≥0.6`（现 0.05–0.23）且 val traj6 MAE 不退步；闭环 17 条 ≥8 成功。
3. **DAgger-lite 修 crawl/OOD 行为**（结论 2 的补充，crawl 段是漂移起点）
   实验 E3（≤10 min）：用当前策略闭环跑 50 条 spec，把 (obs, 专家动作) 重新标注后微调 1 epoch → 判据：crawl 段 ds 均值回到 3.0–3.9 m/0.5s；失败前 max|lane_lat| <0.5 m。
4. **纵向 P-only 稳态差**（独立、低成本）：speed_gain 1→2 或加前馈 → 判据：v/ref 中位数 ≥0.95（现 0.81），oracle 组 max_step 从 7 降下来。
5. **不建议先动 tracker 横向增益**：e_y 已足够小，且完美执行同样失败；tollgate 的"过闸"需要单独行为（专家有专门逻辑），可用 oracle 组 rc≈0.53 碰撞作为回归锚点。
6. 评测口径：四类当前 n=4–5（Wilson CI 很宽，如 roundabout [0,0.49]），修完请用 1000 条 val 集里这四类的全部 spec 复评。

复现命令（`--out` 产物落在 `runs/forensics/`，可重生成）：
```
tools/venv-python tools/diagnostics/forensics_offline.py --section all --rows 24000
tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode lqr   --ids 0,5,11,14,18,19,22,23,28,29,30,32,34,40,43,44,47,3,6,25 --out runs/forensics/closed_lqr.json
tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode exact --ids ... --out runs/forensics/closed_exact.json
tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode baseline --ids ... --out runs/forensics/closed_baseline.json
tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode oracle --ids ... --out runs/forensics/closed_oracle.json
tools/venv-python tools/diagnostics/forensics_closed_loop.py --mode lqr_gain --dtheta-gain 4 --ids ... --out runs/forensics/closed_gain4.json
```
