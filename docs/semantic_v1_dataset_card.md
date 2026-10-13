# semantic_v1_pilot 数据集卡（R21-1）

状态：R21-1 门过（2026-10-12）；**零主模型训练**——本数据集仅用于 R21-2 表示/反事实探针与
后续（需批准）的语义接入训练，不用于当前主模型训练。

## 1. 目的与依据

- 依据：`docs/chatgpt5p6sol_review.md` round-21 第二部分；`docs/deepseekv4p1_argue.md` §5.13
  （Round-22 Outcome D：条件化读出量级不足 → 按预注册转 R21 语义重建）。
- 目标：为 lane-relative 语义（target lane 选择、横向/航向误差、曲率、可行驶 margin、变道需求）
  提供**同状态**监督与反事实状态；契约见 `docs/r21_lane_semantic_contract.md` /
  `docs/r21_target_lane_selection.md`。

## 2. 产物

| 文件 | 说明 |
|---|---|
| `samples.npz` | 9515 行（逐行全量 obs + expert/模型动作 + 扰动/switch 元数据） |
| `manifest.json` | 行数/类型、obs 指纹、ckpt 溯源（`<legacy>` 显式标注）、设备 |
| `schema.json` | obs schema v7 清单 + 行键说明（round-23 字段名同步；见 `schema_sync.json`） |
| `schema_sync.json` | round-23 改名同步记录：旧/新指纹、改名字段、**数值逐位不变**证据与允许用途 |
| `split_manifest.json` | episode 级 train/val 划分（无泄漏断言） |
| `replay_report.json` | 随机重放核验（同状态匹配门） |
| `coverage_report.json` | 路型/速度带/扰动分布/target 使用率/switch 异常率/B 恢复率 |
| `episodes.json` | 每 episode 的 spec/kind/split/扰动日程（重放依据） |
| `_shards/` | 采集中间分片（可续跑/复验；不参与消费） |

## 3. 组成（实测）

| 段 | 行数 | 占比 | 内容 |
|---|---|---|---|
| A 正常 expert | 3844 | 40.4% | `PurePursuitIDMPolicy` 闭环，spec 交通；逐策略步一行；expert 当前动作 |
| B 受控 lane-relative 扰动 | 4063 | 42.7% | 合法 expert 状态置位（lat ±0.25/0.5/1.0 m × heading ±2.5/5/10°），traffic=0；仅留合法且可恢复 |
| C learner-state | 1608 | 16.9% | pri512@e020 真实闭环（`InstrumentedCkpt`）；model plan/mu + expert 当前动作 + outcome |

- episodes：1275（A 55 / B 1200 / C 20）；split：train 1020 / val 255（spec.id % 5 == 0 → val），
  **train/val 无重叠**（`split_manifest.no_leak=true`；train 7507 行 / val 2008 行）。
- 路型覆盖（11 类全有）：straight 648 / curve 1177 / ramp_in 969 / ramp_out 862 / merge 665 /
  split 643 / intersection 808 / t_intersection 887 / roundabout 1138 / uturn 880 / tollgate 838。
- 速度带：<5 m/s 2054 / 5–10 6186 / 10–15 1275 / ≥15 0（本批地图限速上限所致）。
- B 扰动分布（kept 4063）：|lat| 0.25/0.5/1.0 m ≈ 1403/1383/1277（左右均衡：+0.25 703 / −0.25 700 等）；
  |heading| 2.5/5/10° ≈ 1352/1391/1320（左右均衡）。
- target lane 使用：`target_lane_valid` 100%（mask 100%）；来源 route_current 9429 /
  route_continuation 86 / route_next 0 / ego_fallback 0（**无兜底依赖**）。
- target switch：正常 transition 2053、**异常跳变 0（0%）**。

## 4. 质量门（R21-1）

| 门 | 阈值 | 实测 | 结果 |
|---|---|---|---|
| 同状态匹配（随机重放） | 100% | 1013 行 exact 100%（obs/expert 逐位 max diff 0.0；C 模型输出容差内） | ✅ |
| 随机重放行数 | ≥1,000 | 1013 | ✅ |
| target lane 有效 | ≥99% | 100% | ✅ |
| episode split 无泄漏 | 必须 | no_leak=true（spec 集合不相交） | ✅ |
| expert 恢复 | ≥90% | 92.6%（4063/4388 feasible；扰动 4637，skipped 249） | ✅ |
| 异常 lane assignment 跳变 | <1% | 0%（2053 真实 transition / 0 异常） | ✅ |

## 5. 构造细节与口径

- **A**：`build_env(spec, traffic_density=None)` + `engine.add_policy(PurePursuitIDMPolicy)`；
  逐策略步（5 env step）一行；`expert_action` = 空问 labeler 的当前动作（ds/dθ），
  `expert_raw_action` = (steer, throttle)。
- **B**：专家 warmup（默认 30 策略步）→ 合法性检查（on-lane、|lat|≤0.5 m、v≥2 m/s）→
  `set_position/set_heading_theta/set_velocity` 置位（保持速度）→ **builder.reset()**
  （历史按新状态 honest warmup）→ 行 + 恢复窗口（默认 10 策略步，|lat|<0.5 且无
  crash/out_of_road 才算 recovered）→ 只保留 feasible ∧ recovered 的行。
  每 episode 日程 = warmup + k×(1+recovery) 策略步，网格按 `(episode_id*7+k) % 36` 轮转
  （确定性，可重放）。
- **C**：pri512@e020（`runs/BTC20261009-0824_sw_pri512/stage_b/ckpt_epoch020.pt`，legacy obs
  指纹显式标注）真实闭环；每决策步记录 model mu/plan/references + expert 当前动作 + outcome；
  device=cuda。
- **禁六步链**：B/C 均只查询 expert **当前**动作（`tools.dagger_collect._make_labeler` 空问；
  可行性审计确认查询前后 env/ego 状态不变且两次调用一致）。

## 6. 已知限制（必读）

1. **B 段 traffic=0**（受控扰动需要干净恢复环境）；A/C 用 spec 交通。B 行 OD 为空 →
   探针使用 B 行时须注意动态对象分布差异。
2. **B 行历史为 warmup**（置位后 builder.reset()）：这是"扰动 = 新状态"的诚实语义，但
   与 A/C 的连续历史分布不同；R21-2 的 history shortcut 对照须按段分别解释。
3. **C 段单一 ckpt（pri512@e020）**，非 M0/M1 四臂模型；R21-2 的模型侧探针另用 M0 ckpt
   对同一行 obs 前向（`<legacy>` 指纹，仅探针，不训练）。
4. pilot 规模（9.5k 行）与 2 个 split 分片用于探针，**不足以**作为主模型训练集；
   全量语义重采需 R21-3+ 批准。
5. 速度 ≥15 m/s 覆盖为 0（地图限速）；tollgate 等路型 B 扰动较少（受地图位置与恢复窗口限制）。

## 7. 复现命令

```bash
# 采集（6 worker；spawn；setsid+nohup 日志）
tools/venv-python .slim/deepwork/p1_v2/round21/collect_semantic_v1.py collect \
  --out datasets/semantic_v1_pilot --specs env/specs/scenarios_train_5k.json \
  --a-specs 55 --b-specs 1200 --c-specs 20 --workers 6 --device cuda
# 合并 + 报告
tools/venv-python .slim/deepwork/p1_v2/round21/collect_semantic_v1.py finalize \
  --out datasets/semantic_v1_pilot --device cuda
# 随机重放核验（≥1000 行）
tools/venv-python .slim/deepwork/p1_v2/round21/collect_semantic_v1.py verify \
  --out datasets/semantic_v1_pilot --specs env/specs/scenarios_train_5k.json \
  --workers 6 --device cuda --min-rows 1000
```

- obs 指纹：采集时 `v7-d93a5169c3f7`（schema v7；`lane_semantic` + LD 点补充）。
  **round-23 改名后**（`distance_to_next_topology_change` → `distance_to_target_lane_end`；
  `left/right_drivable_margin` → `left/right_same_direction_road_surface_margin`）当前指纹
  = `v7-f19d3b2d3d4e`；**数值逐位不变**（`runs/p1_v2/round23/r21_3a_audit.json`：同过程重跑
  71 行 × 39 obs 键 max abs diff = 0.0）。允许用途：R21-3B canary / R21-3C 探针（指纹白名单，
  见 `schema_sync.json`）；**正式训练/评测禁止**（round-23 §7）。
  旧指纹数据集在 stage 入口硬失败（`pipeline.stages.validate_bc_dataset`，逃生门
  `--allow-legacy-dataset`）；旧 ckpt 仅允许显式 legacy 标注下加载（见契约 §6）。
- 资源：采集 1275 task / 6 worker 176.8s（父进程峰值 RSS 77MB）；verify 120 episode 约 2 min。

## 8. Errata（R21-26A；B 行标签状态契约）

- **B-label-v1（无效）**：数据集内 `expert_action` 的 B 行旧标签在**恢复窗口之后**查询，
  查询状态 ≠ 存储 obs 状态（obs = 置位态、label = 恢复后状态）；`corr(perturb_lat,
  expert_dθ) = −0.156`。**v1 标签不得用于动作监督/探针/mirror 分母**。
- **B-label-v2（`label_state_version=same_state_v2`，有效）**：确定性重放（置位后**立即**查询）
  的 same-state 专家标签；`corr = +0.606`、`correct_sign_frac = 0.685`（repo 坐标：lat 正
  ⇒ 修正 dθ 为正）。产物：`runs/p1_v2/round26/baseline_labels.npz`（3d3，10,803 行；
  独立集 `indep_labels/baseline_labels.npz`，537 行），含 `label_pose` 供 same-state 审计。
- **fail-fast 契约**：训练/验证/选择/probe/mirror 入口必须校验 `label_state_version ==
  same_state_v2` 且 **v1 暴露 = 0**，否则拒绝运行（`runs/p1_v2/round26/data_contract.json`）。
- **排除行**：same-state 审计失败且不可复现轨迹的 B 行（ep1091 的 4 行；重放轨迹漂移 ~10–16 m）
  在训练/验证/选择/probe 全链剔除并记录（`excluded_b_rows`），不得以 v1 标签回填。
