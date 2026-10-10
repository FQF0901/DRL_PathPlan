# P1 探针规格（P1 Probe Spec）v1

> 依据：`docs/deepseekv4p1_argue.md` §7.10（P1 执行手册 v1，LOCKED）A 批探针 + round-4/5/6/7 追加项。
> 约定：全部经 `tools/venv-python` 运行；退出码 **PASS=0 / FAIL=1 / NOT_AVAILABLE=2**；
> 产物落 `runs/p1_0/*.json`（run 级产物落各 run 目录）；探针运行期间**不得修改**被执行实验正在依赖的源文件。

## 1. 探针清单与状态

| # | 探针 | 目的 | CLI / 入口 | 判据 | 产物 | 状态（实测） |
|---|---|---|---|---|---|---|
| 1 | checkpoint 加载契约 | ckpt↔模型一致性 | `tools/controls/artifacts.py --run-dir`（提取 `meta.ckpt_load`）+ `tools/controls/probe_load_contract.py` | counts 0/0/0 且 loaded_ratio=1.0 | `<run>/checkpoint_load_report.json` | ✅ PASS（p0_pri512_plan；两 r3 run） |
| 2 | 统一产物 11 项 | run 可复算 | `artifacts.py`（幂等、不覆盖；`--force` 仅刷新 health） | 11 项齐 + 二次运行零改动 | `<run>/{manifest.json,…}` | ✅ 实测（6 个 run 试点） |
| 3 | plan/action 契约 | `plan[0]==mu`、链有限 | `probe_plan_action_contract.py --forensics-json …` | violations=0 | `runs/p1_0/probe_plan_action_contract.json` | ✅ PASS（全量 76,999 决策步 0 违规；抽样 772 复核） |
| 4 | 时间索引 | step 单调/步长 5 | `probe_time_index.py`（+恒速合成单测 ds=5,dθ=0→(5..30,0)） | n_violations=0 | `runs/p1_0/probe_time_index.json` | ✅ PASS（5000 eps / 355,497 对 / 0） |
| 5 | 跨行 action↔ego_world | 标签 vs 真实运动 | `probe_action_ego_alignment.py` | median≤0.05m、p99≤1.0m、coverage≥99%、零链∩weight>0=0 | `runs/p1_0/action_ego_alignment.json` | ✅ PASS（median 1mm→1.5cm、p99≤0.472） |
| 6 | 几何代理覆盖 | LD/实线可用性 | `probe_geometry_coverage.py` | 诊断阈值（默认 0.5/0.5/0.8） | `runs/p1_0/probe_geometry_coverage.json` | ✅（LD 94.0%、实线 96.9%、plan 覆盖 90.2% 全帧/97.5% weight>0） |
| 7 | freeze 四层审计 | 配方→rg→optimizer→更新一致 | `probe_freeze_audit.py --recipe …` | freeze 组 a=b=c=d=0；train 组 a>0 且 d>0 | `runs/p1_0/probe_freeze_audit*.json` | ✅ specific_only/trunk_only/all PASS；⏳ recovery=NOT_IMPLEMENTED（待 P1-2） |
| 8 | 训练账本 | trainable⊆optimizer | `probe_train_account.py` | not_in_optimizer=0；frozen 不变量空 | `runs/p1_0/probe_train_account*.json` | ✅ PASS（coverage 1.0） |
| 9 | recovery 质量门 | 样本可用性六门 | `recovery_quality_gates.py`（纯函数 + CLI，schema 契约见模块 docstring） | 每门 False 即不可用 | summary JSON | ✅（fixture 全覆盖；真实数据待 P1-1 采集） |
| 10 | LQR/动作饱和 | 饱和率 <5% 证据 | forensics `records` 列 `steer_cmd/steer_sat/lon_cmd/lon_sat`（d2）；聚合见配速矩阵 | 饱和率报告 | `runs/p1_0/d2_pacing_matrix.json` | ✅（d2 列已入；lqr 行在 P1-4 前按需加） |
| 11 | progress/timeout | 配速/超时诊断 | 配速矩阵聚合（timeout 率、rc、`tracking_residual` 中位/均值） | 见 P1-0B 判据 | `runs/p1_0/d2_pacing_matrix.json` | ✅（聚合覆盖）；独立 JSON 待定 |
| 12 | 三轨 reference 中轨 | plan/reference/actual 全链 | forensics 增 `ref_world_head`（决策步：6 点×0.1 s 世界系参考，round 1e-4）+ `ref_arc_len`（全参考弧长；全 tracker 模式、additive） | 三轨并存；旧/新深比对除新列 0 差异 | forensics JSON | ✅ 完成（`3b39ee3`；8 测试 + d2/lqr 冒烟 72/72；内容强校验 max 5e-5；首点=首子步目标，偏移=ds0/5≤2.0 m） |

## 2. 回归基线（复跑对照）

- 探针 5：k1 median 0.000985 / k6 median 0.014761；p99 max 0.47191；k6 coverage 0.9985；零链 28,721（∩weight>0=0）；weight>0 行 265,801。
- 探针 6：LD 有效 0.9399；实线帧可用 0.9687（solid 槽比 0.3269）；plan 覆盖 0.9016（全帧）/0.9753（weight>0）；LD 点距 min 4.85 / p10 6.09 / median 18.08 / p90 43.62 / p99 72.75 / max 126.15 m。
- 探针 7：specific_only frozen=129（optimizer=[specific] n=37）；trunk_only frozen=95（[base] n=71）；all n=166。
- 探针 4：stride=5；n_episodes=5000（1 个单行 episode）；n_violations=0。

## 3. 变更控制

- 任何探针判据/阈值变更 = 显式提交 + 在本文档更新"状态/基线"两栏；禁止静默改判。
- 探针 7 的 `recovery` 配方在 P1-2 落地后必须补跑（A9），并把四计数表并入 P1-2 报告。
- 探针 12 完成后，`plan/reference/actual` 三轨成为 P1-4A"执行一致性"探针的输入之一。

**版本**：v1.1（2026-10-11）：条目 12 完成（`3b39ee3`）；覆盖 commit `3b39ee3` 时点的工具集。
