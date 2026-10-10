# P0 审计报告：上限 / 口径 / 失败归因（v1.0-draft）

> 执行窗口：2026-10-10 02:47–07:22（P0-B 矩阵）+ 08:31–（laneplan 修复版重跑）
> 审计对象：Arm P（primary 256）与 pri512（primary 512）两个 v8 模型；s11（v7 完整链，跨版本参考）；IDM 规则专家。
> 纪律：评测安全前置（fail-fast 加载校验 + 版本戳）；跨版本只作方向参考；逐 episode paired 数据见 `runs/forensics/summaries/*_eps.json`。
> 状态：**laneplan 修复版全量重跑中**（该单元格待补）；其余数据定稿。

---

## 1. 结论摘要（决定性问题与答案）

**Q1 上限在哪？** 最高**可执行上限 = d2（重复"当前专家动作"）0.706** > oracle（开环重放专家路径）0.608 < IDM 自身 0.756。
⇒ **执行栈与 (ds,dθ) 接口可承载 ≈0.71–0.76**；v8 模型现处 0.22–0.33，**差距在"学到的 plan/动作质量"，不在接口可行性**。

**Q2 口径差多大？** v8 模型在 repeat_action 口径下**归零**（arm_p/pri512 均 0.000、off=1.000）；s11（A-hold 训练）plan 0.646 > repeat 0.418。
⇒ **保持 plan 执行协议**；A-hold 只是 RL 内部记账口径，不得作为部署/评测口径。

**Q3 失败属于哪一类？** 模型失败**以 off-road/压线为主（76–78%）**（IDM 以碰撞为主、d2 以超时为主）；其中 **>50% 的失败带 plan-infeasibility 因子**（T_plan 中位 120/295 步——不是瞬时错 plan，而是随驾驶漂移恶化）；exact ≤ lqr。
⇒ **P1 = 计划侧（plan-side）**：几何 margin / 曲率平滑 / recovery 数据。

---

## 2. 运行清单与可复算性

### 2.1 双口径评测（`tools/test.py`，eval500，LQR，plan/repeat_action）

| 运行 | 参考 | succ | ckpt sha256（前 10） | git | 加载校验 |
|---|---|---|---|---|---|
| `BTC20261010-022524_p0_arm_p_plan` | plan | 0.224 | `3d7d16a9c1` | `7c22939` | 0/0/0 |
| `BTC20261010-023500_p0_arm_p_repeat` | repeat_action | 0.000 | `3d7d16a9c1` | `7c22939` | 0/0/0 |
| `BTC20261010-023556_p0_pri512_plan` | plan | 0.332 | `f9c91d08b5` | `7c22939` | 0/0/0 |
| `BTC20261010-024520_p0_pri512_repeat` | repeat_action | 0.000 | `f9c91d08b5` | `7c22939` | 0/0/0 |
| `BTC20261010-024746_s11_v7_plan` | plan | 0.646 | `a7cc091f…`（runner pin） | v7 栈 `c37acbb` | 烟测 0/0/0 |
| `BTC20261010-024746_s11_v7_repeat_action` | repeat_action | 0.418 | 同上 | 同上 | 同上 |

完整 sha256：arm_p `3d7d16a9c10389663601d00c7c132b2f5e787fdf7b2132f6c5bd280c6b0a31ca`；pri512 `f9c91d08b56c24eb4ba999927d6856decf0a0e00cae5027e2b71326cfbf70161`。
s11 ckpt：`runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt` sha256 `a7cc091f…ba2e`（runner 内 fail-closed pin）。

### 2.2 P0-B 矩阵（`forensics_closed_loop`，eval500 全量 500 条）

产物：`runs/forensics/p0_{oracle,laneplan,d1,d2,baseline,e1_arm,e2_arm,f_arm,e1_pri,e2_pri,f_pri}.json`；md5 见 `.slim/deepwork/s1_audit/logs/matrix_summary.txt` 与会话记录；逐 episode：`runs/forensics/summaries/*_eps.json`。
统一口径：`--spec env/specs/scenarios_eval500.json`、显式 500 ids、`--tracker lqr`（e2=exact 除外）。

---

## 3. 上限矩阵结果与判读

| 行 | 参考构建 | succ | off | coll | rc |
|---|---|---|---|---|---|
| A/baseline | IDM 自身 | 0.756 | 0.068 | **0.144** | 0.882 |
| B/oracle | 专家实测未来 3s 位姿（cursor 重定位）+ LQR | 0.608 | 0.134 | 0.058 | 0.856 |
| **D2/d2** | **当前专家动作 repeat 6 步 + LQR** | **0.706** | 0.038 | 0.020 | 0.913 |
| C/laneplan | 车道中心线 + LQR（修复版，全量 500） | **0.442** | 0.232 | 0.192 | 0.794 |
| D1/d1 | 未来专家动作链 → 开环 (6,2) + LQR | 0.000 | 0.950 | 0.050 | 0.259 |
| E1/e1_arm | 模型 plan（Arm P）+ LQR | 0.224 | 0.604 | 0.026 | 0.695 |
| E2/e2_arm | 模型 plan（Arm P）+ exact | 0.210 | 0.580 | 0.210 | 0.494 |
| E1/e1_pri | 模型 plan（pri512）+ LQR | **0.332** | 0.508 | 0.022 | 0.689 |
| E2/e2_pri | 模型 plan（pri512）+ exact | 0.260 | 0.614 | 0.118 | 0.544 |
| F/f_arm | 模型动作 repeat + LQR | 0.000 | 0.990 | 0.010 | 0.098 |
| F/f_pri | 模型动作 repeat + LQR | 0.000 | 0.994 | 0.006 | 0.080 |

**判读**：
1. **d2=0.706 是唯一贴近 IDM 的执行栈证明**（且 > oracle 0.608：闭环重锚 > 开环重放）。
2. **exact ≤ lqr**（两臂一致）⇒ 照着模型 plan 精确执行不会更好 → plan 几何本身是瓶颈；exact 还显著抬高碰撞（0.21/0.118）。
3. **F 全崩 vs d2 正常**（同口径、只换动作来源）⇒ **模型单步动作质量是硬伤**（repeat 放大：dθ≈0.1–0.19 rad/步被复读 → 参考过弯/降速 → LQR 饱和 → 螺旋出界）。
4. d1（开环专家链）崩溃经复核为**真实结果**（非 bug）：(6,2) 开环链不随自车误差回正；d2 的现状态反馈动作天然含回正信号。
5. **laneplan（纯车道中心参考，无路线规划）= 0.442、碰撞率 0.192**：单靠"沿车道中心"可达 0.44，但仍显著低于 oracle/d2（0.61/0.71）且碰撞高发——路线选择与交互处理是必要成分；**同时它仍高于模型 e1_pri 0.332**——即"无脑跟车道中心"的执行栈都比当前模型学到的 plan 更可用，进一步坐实"计划/动作质量"是主差距。

---

## 4. 失败归因（失败条件化，500 eps/行）

| 行 | 失败数 | offroad | timeout | collision | plan 因子 | anomaly | T_plan 中位 |
|---|---|---|---|---|---|---|---|
| IDM | 122 | 34 | 16 | **72** | 0 | 34 | — |
| d2 | 147 | 19 | **118** | 10 | 52 | 19 | 60 |
| oracle | 196 | 67 | 100 | 29 | 0 | 62 | — |
| **e1_pri** | 334 | **254** | 69 | 11 | **173（52%）** | 241 | 120 |
| e1_arm | 388 | **302** | 73 | 13 | 176（45%） | 297 | 295 |
| f_pri | 500 | 497 | 0 | 3 | 433 | 407 | 15（≤5 步 89） |

**结构差异**：IDM 输在**碰撞**、d2 输在**超时**、模型输在**off-road/压线**——三系统失败画像截然不同。
**"off-road"的解释**：env `_is_out_of_road` 含黄/白实线与行道 flag（`out_of_road_done`+`on_continuous_line_done`），而 footprint 检查只查 lane 面 → 大多数"off-road"实为**连续线/几何余量**问题（~95% 落在 anomaly 桶 = 两套判据口径差）。
**时机**：e1 行 T_plan 中位 120/295 步、≤5 步=0 → **plan 恶化是随驾驶推进的漂移过程**（与 recovery/边界 margin 监督方向吻合）。

**关键配对（McNemar 原始计数）**：
- e1_pri vs e1_arm：+80 / −26（pri512 显著更优，净 +54）。
- **e1_pri vs d2：+3 / −190（天花板差距决定性）**。
- baseline vs e1_pri：+229 / −17。
- d2 vs baseline：+47 / −72（d2 是唯一贴近 IDM 的对照）。
- d2 vs oracle：+58 / −9。

---

## 5. 工具勘误与安全事件（本审计的"防线"记录）

1. **fail-fast 两次拦截静默错评**：
   a. 历史事故背景：A3 曾在"形状不匹配→随机初始化"下产出 0.000 假阴性；
   b. 本次：stg3 臂评测缺 `spatial` 透传（ckpt 3 层 vs 模型 2 层）被 fail-fast 拦为 rc=2——**若在 fail-fast 之前，会静默产出错数字**。修复：`eval_runner.task_model_config()` 补 `spatial`（commit `08582f2`）+ 守卫单测；stg3 四评测已排队补跑。
2. **laneplan 参考构造 bug（已修）**：丢失自车相对中心线的横向/航向偏差 → 参考退化为切线 → 自激振荡；修复为 (N,3) 自车位姿参考（commit `5e6e1b3`）。
3. **d1 非 bug**：链重建逐位复核正确；崩溃为"开环链不回正"的真实结果。
4. **anomaly 语义**：判定口径差（非 bug）；建议后续独立记录连续线/行道 flag。

---

## 6. 决策树逐条判读（锁定版）

| 情况 | 判定 | 依据 |
|---|---|---|
| A：D 不高于 IDM → 修接口 | **否** | d2=0.706 逼近 IDM；接口非瓶颈 |
| B：B/D ≫ A 且 E ≪ D → 学习/表示侧 | **是（主结论）** | d2 0.706 ≫ e1_pri 0.332；plan 因子 >50% |
| C：Exact ≫ LQR → tracker 侧 | **否** | e2 ≤ e1（两臂） |
| D：repeat_action ≫ plan → 口径错位 | **否（反转）** | plan ≥ repeat（模型 repeat 归零；s11 0.646>0.418） |
| E：主要是 recovery failure | **否（主标签层面）** | recovery 主标签仅 1–2 例；但 recovery 数据仍入 P1 |

---

## 7. P1 提案（单一分支：计划侧）

1. **DAgger 真实 recovery 轨迹**：失败前 2–4s 回溯，存 student plan + 实际轨迹 + expert recovery plan；训练端对**整段 6 点 recovery plan** 监督（替换常量外推 traj6）；hard-mining 换**闭环风险**（boundary margin / low TTC / tracking residual / plan footprint invalid）。
2. **几何监督**（rollout 期起）：plan 节点间 0.5–1.0m 插值 footprint → corridor violation / min margin（signed 不可得则 inside_ratio/first_invalid）。
3. **曲率平滑**：κ 与 Δκ 惩罚（针对 repeat 曲率放大与压线）。
4. （可选实验）近场节点加密。
5. **不重定义部署口径**；PPO 侧保持 A-hold 记账 + KL 锚设计。
6. **验收锚点**：d2=0.706 / IDM=0.756；G1'（链路可用）✅ 已满足（"明确主要损失层"）；G2 追平（3 seeds 平均 ≥0.74、off-road −30%）；G3 超越（≥0.80，目标带 0.82–0.85；paired 显著、clean500+eval500）。

---

## 8. 限制与未完成

- **laneplan 修复版已全量重跑完成**（0.442；旧 bug 版留档 `runs/forensics/p0_laneplan_bug.json`）。
- **s11 逐 step 分类未做**（v7 栈缺新 forensics 字段；成本/收益低）；s11 仅作跨版本方向参考。
- **timeout 占比高**（d2 80%、oracle 51%）值得后续单独分析（配速/卡滞）——可能压低上限估计。
- clean500 复核未做（G2 前补）；分类统计基于 `tie（id,seed）` 全量。
- 所有评测为单一冻结协议（deterministic=true；环境存在已记录的内禀非确定性 ≈1/500）。

---

## 附录 A：复现命令（节选）

```bash
# 双口径评测（示例：arm_p）
tools/venv-python tools/test.py --policy ckpt --ckpt runs/BTC20261007-2202_arm_p/stage_b/final.pt \
  --spec env/specs/scenarios_eval500.json --tracker lqr --eval-reference plan --workers 6 --out runs/eval
# 矩阵行（示例：d2；ids 显式全量）
tools/venv-python -m tools.diagnostics.forensics_closed_loop --mode d2 --spec env/specs/scenarios_eval500.json \
  --ids "$(tools/venv-python -c "import sys;sys.path.insert(0,'.');from env.scenario.spec import load_specs;print(','.join(str(s.id) for s in load_specs('env/specs/scenarios_eval500.json')))")" \
  --out runs/forensics/p0_d2.json
# s11（v7 栈）
bash .slim/deepwork/s1_audit/run_s11.sh
```