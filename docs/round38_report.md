# Round-38 报告：全链路瓶颈裁决与代码减枝（R38 收口）

- 日期：2026-10-15 ｜ 分支：`Planner_RL` ｜ HEAD：`75c9d64`（§5.80 收口；工作区仅 `AGENT.md`/`opencode_mobile.md` 外来改动）
- 范围：GPT R38 方案（`2d670cc` 版 `docs/chatgpt5p6sol_review.md`，§一–§九）全案执行——五问裁决 + 代码减枝八组 + 统一闭环基准
- 口径：**零训练**（未获 CG 放行前不训练）；离线裁决 = teacher-forcing 矩阵（A1/A3/C1/C2/C3）；闭环 = calib50（R37 协议）与 Clean500（C6 基准）；减枝 = 默认关 + 逐位（plain 25 / structured 73 键 `max|Δ|=0`）
- 约束：`.slim/`、`runs/` 不入库；生产改动默认关；本文件为 docs 单文件交付（不 commit）
- 依据（数字全部取自）：`docs/chatgpt5p6sol_review.md` §一–§九；`docs/deepseekv4p1_argue.md` §5.70–§5.80；`docs/r38_unified_benchmark.md`；`docs/r38_path_field_classification.md`；四交付物（审计/manifest/入口矩阵/ckpt 契约）；各 lane 报告（`.slim/deepwork/p1_v2/round38/{a1,a3,a4,c1,c2,c3,c6,d1,p1,p1b,p2a,p2b,p3,docs_sync}_report.md`）

---

## 1. TL;DR

1. **五问全闭合（§5.80）**：Q1/Q2=A1（主瓶颈=**想象步状态接口**，非 WM latent、非头/监督）｜Q3=C1（Memory 历史**内容**净贡献≈0）｜Q4=C3 分类 + Q4-A 落地（3 硬 0 + 4 回退；生产者化）｜Q5=C2（MoE 三档门全不过、维持关闭）。
2. **三层根因更新（§5.70/§5.71）**：**主项=想象步接口**（`step_geometry` speed/静态/动态列 hold + 重锚 warp；S_geo_E0 slow+accel −2.81 m）＞ **WM latent 漂移（二阶，模态真值替换零恢复，E1 +0.003）** ＞ **监督语义（三阶，h1 超发 +0.116 m/步）**；我方 §5.68"监督语义主瓶颈"判断降级（认账）。
3. **解析式修复穷尽 ⇒ C3 升 P0**：A3 11 臂无解（S_self 过发 +14.3%；最优带副作用组合全样本仍 +0.435 m）⇒ **裁决=必须"解析基+学习残差"想象步动力学**（D8 待 CG）；两条确定性几何缺陷落地（foot 伪影 65.5%→0 / margin 反号）。
4. **零训练通道总账**：R38 唯一改变闭环数字的零训练动作 = 接口生产者化（Q4-A，逐位复现 `C1_true`，非新增益）；warp 中性、MoE 负增量、Memory≈0、A4 几何修复闭环 KPI 未见变化（收益预期在训练侧）——**无任何零训练臂达成晋级**。
5. **减枝八组全落地**：A–D 四波（`596747c`+`48cb5eb` / `a226285`+`dd0a8c2` / `939d1a2` / `9577078`）+ E–H 尾波 `f81da17`；prod 命中收敛（A 89→24=仅护栏/契约键/文档串、B 115→0）、pytest **987→1068**、全程默认路径逐位。
6. **统一基准（Gate D）**：protocol hash v1 + Clean500（n=500）7 臂重跑；**IDM+PP 0.742 逐位复现 v7 锚**；候选 `c1_true` 0.390 > `c1_fill` 0.366 > `c1` 0.304 ≈ `c2` 0.302——**全部候选安全硬门不过（off 0.128–0.208 / coll 0.116–0.132）⇒ 均不晋级**。
7. **收口与待办**：§一–§九逐条落地（§5.80）；四项挂账 = B 类（掩码进头+static 下沉）/ D8-C3 训练 / Eval500 3-seed + clean150 / protocol hash 原生接入（pin 易失已记）。

---

## 2. 协议与口径

### 2.1 三级集
| 集 | 定义 | n | 用途 | 证据 |
|---|---|---|---|---|
| **calib50** | `runs/p1_v2/round25/specs_calibration50.json`（**train split**，id 7xxx） | 50 | 开发/快筛（R25 起主线开发集；±1 集≈±2pp） | `r38_unified_benchmark.md` §3 |
| **clean500** | `docs/v7_reports/specs/specs_val_only500.json`＝val1000 − eval500；sha256 `087db3f5…`（seeds `a2b66940…`） | 500 | **历史统一晋级集**（v7 pregreg §0 即此文件；核验 ∩eval500=∅、∪=val、easy189/med170/hard141） | 同上 §3 + `runs/round38/c6/clean500_recon.json` |
| **eval500** | `env/specs/scenarios_eval500.json`（sha256 `98856105…`） | 500 | 正式声明（**3 seeds + paired bootstrap/McNemar 待跑**） | `r38_unified_benchmark.md` §3/§6 |

- **calib50 与 clean500/eval500 零交集 ⇒ 跨集禁直比**（`r38_unified_benchmark.md` §3/§4；跨集登记仅作对照，见 §5.2）；clean150 = clean500 前 150 条（层级去留待 CG，§7 提案）。
- 闭环协议串（各臂逐字相同，仅加臂专属 flag）：`--ld-mode semantic --rollout-semantic static_role_only --moe-off --tracker lqr --tail chain`（R37 §1.3；C6 全臂沿用）。

### 2.2 evaluation_protocol_hash（v1）
- 工具 `.slim/deepwork/p1_v2/round38/c6_protocol.py`（纯 CPU 只读；plan 臂/IDM+PP 同一工具），逐 run 落 `<out>/protocol.json`+`run_meta.json`（`r38_unified_benchmark.md` §2）。
- **哈希域**：`spec_set`（name/sha256/n/`seeds_sha256`，seed 口径 `spec_native`）+ `harness`（workers6/max_steps1000/device）+ `control`（lqr/chain）+ `model`（semantic/static_role_only/off）+ `frozen.net_tree_sha16`（`net/**/*.py` 树指纹）+ `frozen.entry.sha16`（`eval_a3.py`）；canonical JSON sha256 取前 16 hex。**`spec_set.path` 不入 hash**（仅记录）。
- **protocol_common_hash = `f5d9286a1d83b7b5`**（全部臂一致）；plan 臂 `protocol_hash=a1802ac8bb657afc`；IDM+PP 双入口 `68eaeb8e3dc64de1`/`549f3a2c4c83c427`——差异**仅**为 `control/model` 的 `not_applicable:baseline_native_control`（规则策略原生控制，`pipeline/eval_runner.py:604`；c6_report §0）。
- **net/ 或入口脚本任何内容改动都会换 hash**；每臂 run 前后各记指纹（`net_changed_during_run`），见 §5.5 并发处置。

### 2.3 主排序与统计纪律
安全硬门（`off_road`≤0.10 / `collision`≤0.08，R37 Calib50 门）→ `success` → `timeout`/`rc` → 诊断量（禁模糊加权；GPT R38 §五 Gate D）。n=500 单臂 ≈ ±2.2pp；单 seed 只作方向/筛选，正式声明留 Eval500 3-seed（`r38_unified_benchmark.md` §4/§5）。

---

## 3. 裁决矩阵（GPT §二 五问 + §三 latent WM）

| 问 | 裁决 lane（证据） | 关键读数（全部取自 lane JSON/报告） | 结论 | 自证逐位 |
|---|---|---|---|---|
| **Q1 真实未来状态上 Policy 是否正确？** | A1：臂 T（真实未来 obs→encode→同一头）vs 臂 W（生产想象链）vs teacher；n=7133（train 行 ∧ 同 episode 连续 7 决策步，task×速度桶分层 cap300 seed0；主 ckpt `runs/round36/j2/t1_keep_res/ckpt_epoch001.pt`） | T ds bias ≤0.063、MAE 0.146→0.211、r 0.983→0.953（h1–h6），计划 \|Σds−真\| 0.791 m；W MAE **0.146→0.664（2.49×T）**、r→0.719、\|Σds−真\| **2.309 m**、末端位置 p50 1.95 m；slow+accel 子集：W Σds **9.826 vs 真 13.071（−24.8%）**、T 12.786≈teacher 13.035 | **头≈教师（不是瓶颈）；退化随 h 累积（h1 与 T 逐位同一）⇒ 纯链上状态传输误差** | S1 E6 链 vs 生产 forward `max\|Δ\|=0.0`；S2 臂 T h1≡臂 W h1=0.0；S3 `plan[:,0]==action_mu`=True；契约 runtime ok=1.000（a1_report §0/§1） |
| **Q2 WM 哪一部分导致错误？** | A1：E0–E6 模态真值替换 + S_speed/S_geo 附加臂 | **latent 替换零恢复**：Δ\|Σds−真\|（vs E6）E1 **+0.0030**、E0 +0.1443（更差）、E2 +0.1766、E3 +0.1973、E4 +0.0735、E5 +0.0644（全 ≥0）；**几何/接口真值→显著恢复**：S_geo_E6 −0.6315（slow+accel −1.3247）、S_geo_E0 −1.4989（−2.8116）、S_speed_E6（slow+accel −0.7296，恢复率 0.707→0.766/0.842/0.901）；WM 漂移存在但中等（ego cos 0.958→0.912、LD 0.988→0.935、OD 0.899→0.836） | **排序：想象步状态接口 ＞ WM latent 漂移（二阶）＞ 头/监督语义（三阶）**；监督语义=决定 h1 量级（teacher h1 超发 +0.1157，T 复现 0.1133），非自锁主因 | 同 A1；E* 为状态链级替换、S1 覆盖复刻正确性（a1_report §2） |
| Q1/Q2 后续：**解析式接口修复是否可行？** | A3：11 臂 harness-only（与 A1 同源同集；含标定块 n=256） | **无组合满足"slow+accel Σds≈13.0 ∧ 全样本 \|Σds−真\|≤2.309"**：S_self 过发 **14.935（+14.3%）**（h6 3.83 vs teacher 2.81；v_self 5.77→6.33 vs 真 5.23→4.87）；S_dyn 全样本 −0.016/−0.010 但 slow+accel 无恢复；S_gfix_self_dyn_B slow+accel −1.194（−9.1%）但全样本 **+0.435 恶化**；oracle（真实几何）−1.32 vs 最优可预测 −1.19 ⇒ **oracle→可预测差距不能在解析式内弥合**；两条确定性缺陷：①foot 伪影（重锚 foot 不看 `anchor_mask`，链上 **65.5%** 想象步锚几何零化；修复后 0%/valid_agree 0.302→0.881）②margin 反号（`left=W/2−lat` vs 数据集 `W/2+lat`；对齐 MAE 3.617/3.690→2.928/3.028） | **判定 FAIL ⇒ 裁决：C3 学习型想象步动力学升 P0（必要项）**；解析基+学习残差 `A_{k+1}=A_analytic+ΔA_learned`；唯二出路=学习型动力学或策略×动力学联合（J2，带阻尼）。A1-⑦ 预测（接口修成后仍爬行 ⇒ 转判执行契约）由 D8 门③预置 | S1/S4/S6 逐位=0.0；与 A1 共享臂逐项差=0（a3_report §0–§3） |
| **Q3 Memory Bank 是否有真实贡献？** | C1：8 变体（full/cur_only/ego_only/od_only/others_only/shuffled/shuffled_hist5/repeat_nearest）+ 5 专项；n=7133 同口径 | 字面判据（full vs cur_only）：**2.309 vs 3.493（Δ−1.184 m，t=−41.1）⇒ 未触发"不得扩展"**；但机制归因（逐位）：plan **只经 `others_ctx` 吃历史**（others_only≡full 逐位；`probe_od_hist` 改 od_ctx Δ4.42 而 plan Δ=0.0、仅 value Δ0.012；ego 历史无消费者、LD 历史零影响）；把 0..4 帧替换为"当前帧复制"（repeat_nearest=**2.304**，Δ−0.005 不显著）/打乱（shuffled_hist5 2.294）⇒ **历史内容与顺序净贡献≈0**；1.18 m 增益来自"6 帧聚合可见性"（cur_only 在训练分布仅 0.83% 边缘） | **不支持扩展 Memory Bank**（对象记忆若保留须先接进 plan 路径；现仅进 value）；修正 GPT 字面口径与分布内对照须同读 | 8/8 变体链≡生产 forward（plan/ds/κ/traj max\|Δ\|=0.0）；full 臂与 A1 臂 W 逐项相同（c1_report §0/§5） |
| **Q4 `path_*` 旁路是否必要？** | C3：7 列 × 5 档矩阵（38 臂；n=7133 同源）+ 评测侧只读探针（3ep×300step，n=180 头调用）；**Q4-A**：生产者补全落地 | 口径修正：**3 硬 0（static/dyn/ttc 静默 0）+ 4 确定性回退（margin×2/nav/d2end）**；`present` 掩码**不进头**（列 3/4/5/7 上"缺键"≡"静默 0"逐位 0.0）；TTC 唯一有量级：**Δds0 +0.123、\|Δds\|≤0.482/步、ΔΣds −1.816、Δ\|Σds−真\| +0.757**；分类：`dynamic_ttc`/`dynamic_distance`→**并入标准 obs**；`left/right_margin`/`nav_progress`/`d2end`→**WM anchors 确定性派生**；`static_distance`→**仅诊断**；节奏=**先接口后掩码**。Q4-A：builder==M7 真值 **334/334 逐位**、闭环逐 id **复现 `C1_true` 50/50（0.400/0.080/0.180/0.340）**、头入口雷达 7/7 在场率 **0.0→1.0**、dynamic 两列缺键 fail-fast、pytest 998→1005、K1 契约门 (a)–(d) PASS（(e) 因 P1 先删 oneshot 记 skipped） | **分类定稿 + 生产者侧落地（A 类）；"掩码进头 + static 下沉"=B 类，与下一次重训同批** | S1/S2/S5/S6/S7 逐位=0.0（含"零值≡缺失"与 nav 回退==数据集列逐位）；Q4-A 语义等价 max\|Δ\|=0（c3_report §0.2/§B.2；q4a_report §0–§2） |
| **Q5 Primary、MoE 是否值得继续？** | C2：只读消融（P='--moe-off'/L=learned/F0–F7 固定 expert/R 随机/O1–O2 oracle；n=7133 同口径） | **Primary 三档代理门全不过**（keep_normal 1/3、follow 0/3、slow_recover 1/3 判据）；**learned MoE 增量为负**：\|Σds−真\| Δ **+0.055/+0.030/+0.163 m**（95% CI 不含 0）、恢复率 −0.035；上界：Oracle（不可部署）只吸收缺陷 **24%/22%/11%**，best-fixed 仅 slow_recover −0.161 m（4.4%），learned↔oracle 选择一致率仅 0.128、oracle 后残差仍 0.353 m/步 | **MoE 保持关闭（`--moe-off`）；结构保留、接口修复前不投训练预算**；"路由不是出路"；**口径校正：A1/A3 的"生产口径"= MoE-off（P 臂 ≡ A1 臂 W 逐键 0 差异）** | S1/S2/S3/S4/S9 逐位=0.0（S9 对 A1 臂 W 8 键逐项同；组合律/随机 gates 合法；c2_report §0/§1.3） |

**结论链（一句）**：头不是瓶颈（Q1）→ 链上状态接口是主项、latent 非杠杆（Q2）→ 解析式修不动、学习型动力学必要（A3）→ 历史内容≈0、MoE 负、旁路字段归位（Q3/Q5/Q4）⇒ 训练预算指向 **C3（D8）+ 下次重训同批的 B 类 schema**。

---

## 4. 减枝八组（GPT §七 A–H + §八）

**总账**：审计 67 符号（`code_reachability_audit.md`，**零纯 UNREACHABLE**：65 REACHABLE + 2 TESTS_DOCS_ONLY）；manifest **46 项**（ARCHITECTURE_FORBIDDEN 19 / LEGACY_REPRO 24 / CHECKPOINT_ONLY 3；21 KEEP 不入册）；A/B/C/D 共 **27 项已回填 `removal_commit`**（`deprecation_manifest.yaml::landing`，`71b8153`）；审计 §8 落地状态同步。每波过"全量 pytest + 默认关逐位（plain 25 / structured 73 键 `max|Δ|=0`）+ 键集/闸门"。

| 组 | 对象 | 状态（commit） | 关键读数 | 残留（有意保留） |
|---|---|---|---|---|
| **A** | oneshot/`plan_override`/`PathExecutionHead` 别名/`compose_path_plan`/`allow_oneshot`/`--diagnostic-oneshot` | ✅ 入库 `596747c`（S1 迁）+ `48cb5eb`（S2/S3 删） | prod 行命中 **89→24**（残留=守卫/契约键/文档串）；`compose_path_plan` 等 4 符号 prod 命中=0；pytest 987→**998**；`.slim` 11/12 清扫（74→23 行、活调用点 0；`eval_a3` 冻结→P3 清零 15 处） | 契约⑤键（语义=结构派生"路径不存在"，可证伪测试）；`PATH_HEAD_MODES` 两档；`path_head_mode()` 派生读出；`net/probes.py` 守卫 flag；`.slim` 守卫 `allow=` tripwire（fail-fast 语义） |
| **B** | K-anchor 旧计划头（num_anchors/anchor_* 子模块/冻结表/`plan_anchors_k6.json`） | ✅ 入库 `a226285`（S1 迁 9 文件）+ `dd0a8c2`（S2/S3） | **12/12 符号 prod=0**（115 行/9 文件删除）；`net/anchor.py` −371；config 零活键；pytest 1005→**1022**；键集 168/186 sha 双侧一致；v7 K-anchor ckpt **20 键全登记 ignore、reject=0** | `legacy_key_mapper.IGNORE_PREFIXES`（20 键，CHECKPOINT_ONLY 隔离）；13 带锚/15 带 lane·ttc 旧 ckpt 键面按规忽略；`.slim/g3_wp3a_proof.py` 在新 net 下预期 fail-fast |
| **C** | 旧 `set_ego_warp()`（w1/w2 实验接口） | ✅ 入库 `939d1a2`（先冻结 oracle 再删） | fixture `tests/fixtures/r38_p2b_warp/`（4×.pt+manifest 逐键 sha256，~4.7MB）；**M1 Gate A PASS**（五臂逐位；w1/w2=fixture↔HEAD live）、**R33-B Gate 0 PASS**；`WARP_MODES=("off","analytic")` | 快照 `tools/legacy/legacy_ego_warp.py`；fixture 常驻测试物 |
| **D** | Policy/path_head 双实体（`enable_path_head`/`path_head_ablation`/旧头诊断） | ✅ 入库 `939d1a2`（六步含 key mapper 先行） | 结构化头收编 **`model.policy.structured_action`**（`path_head` 只读 alias）；forward 单动作出口；L-A 契约断言零弱化（`contract_*` 全 1.0）；键面 raw 差=**恰好 18 键**、mapper 归一后 sha 一致；pytest 1022→**1043** | `set_path_head_ablation` 两档 legacy 诊断入口（oneshot fail-fast）；`set_policy_diag` fail-fast 桩（指离线工具）；`path_head` 只读 alias；migration loader 未接线（见 §8） |
| **E** | `path_*` 旁路 7 字段 | ✅ Q4-A 生产者侧 `9577078`（A 类）；P3 残留语义文档化 | 见 §3-Q4；t0 段 8 列（+`path_speed`）由生产者给出 + 同源 `__present`；dynamic 两列缺键 fail-fast；雷达 7/7 在场 0.0→1.0；P3：逐字段禁静默 0 语义入文档 + 回退 present 对齐（**回退值逐位不变**） | B 类（掩码进头 + `static_distance` 下沉）挂账，与下次重训同批（分类文档 §6.3） |
| **F** | target-lane 强逻辑（`select_target_lane`/token/identity） | ✅ P3 `f81da17`（只加来源标签，不删） | `TARGET_LANE_SOURCE_TAGS`（road_level_navigation/teacher_target_lane/deployment_available）env 单一出处 + net 转出；**零行为** | 生产部署去依赖唯一 target lane = 长期项（现 `lane_anchor_xy` 为 teacher-only 注入） |
| **G** | legacy monitoring（`_LEGACY_*`/legacy_tags）+ others-28 维重排 | ✅ P3 `f81da17` | `pipeline/monitoring.py` −148/+45（`TrainingMonitor(legacy_tags=…)` 写入即 fail-fast；CLI `--monitor-legacy-tags` 退役）；`net/mem.py` 28 维输入 **fail-fast**；两 legacy 模块自检 OK | 旧数据集（schema v2/v3，others=28）直喂 fail-fast，迁移路径=重采或显式 `tools/legacy/others_layout` |
| **H** | cheap-path 隐式布尔（`rollout=`/`world_model=`/`wm_detach`） | ✅ P3 `f81da17` | `forward_mode` 四态枚举（policy_single_step/policy_wm_rollout/world_model_teacher_forcing/diagnostics）+ 映射表 + 混用/非法 fail-fast；**11 例布尔↔枚举逐位对照**；`wm_detach` 撤形参（经 `**legacy_signature` 恒 no-op） | 旧布尔入口=兼容 shim（~50 调用点迁移挂账）；`diagnostics` 独立语义未接线；`config/train.yaml::wm_detach` 注释键保留 |

**八组之外**：D1（§四 七类 Probes/Hooks，`248a3ed`，pytest 987，默认关逐位）与 A4（step_geometry fixed 默认关，`b9d1721`，973）为裁决/修复载体，计入 §7 舞台资产；CI 守卫五条随 P3 落 `tests/test_r38_ci_guards.py`（10 例）。

---

## 5. 统一基准（Gate D / C6）

### 5.1 七臂表（Clean500，n=500，单 seed=spec_native；**全部 = pin 内重跑**）
| 臂 | succ | coll | off | tmo | rc | 速度(m/s) | 步数 | wall_s | protocol_hash |
|---|---|---|---|---|---|---|---|---|---|
| IDM+PP（baseline_eval 入口） | **0.742** | 0.174 | 0.054 | 0.030 | 0.865 | 7.75 | 354.7 | 117.0 | `68eaeb8e…` |
| IDM+PP（eval_runner 交叉核验） | 0.744 | 0.172 | 0.054 | 0.030 | 0.866 | 7.75 | 354.9 | 120.4 | `549f3a2c…` |
| C0 旧正式基线（K1e2） | **0.588** | 0.120 | 0.142 | 0.152 | 0.878 | 3.63 | 725.1 | 1199.5 | `a1802ac8…` |
| C1 当前候选（结构化） | **0.304** | 0.124 | 0.130 | 0.442 | 0.751 | 2.51 | 852.6 | 1737.5 | `a1802ac8…` |
| C1_fill（+TTC 中性 17.7 s） | **0.366** | 0.116 | **0.208** | 0.310 | 0.776 | 2.85 | 786.2 | 1648.4 | `a1802ac8…` |
| C2（+analytic warp） | **0.302** | 0.122 | 0.128 | 0.448 | 0.749 | 2.51 | 850.8 | 1744.1 | `a1802ac8…` |
| C1_true（+真值回填） | **0.390** | 0.132 | 0.178 | 0.300 | 0.787 | 2.94 | 775.0 | 1595.6 | `a1802ac8…` |

- **IDM+PP 双入口一致性**：逐 episode 仅 id=593 一处 driver 口径差（1/500）⇒ 锚 0.742–0.744；**与 v7 clean500 历史锚 0.742（coll 0.174/off 0.054/rc 0.865）逐位复现**（c6_report §1/§5）。
- **配对翻转（episode 级，McNemar 精确）**：C0→C1 succ **+18/−160**（p<1e-4）、timeout +154/−9；C1→C1_fill succ **+41/−10**、timeout +7/−73、**off_road +44/−5**（p<1e-4）、collision +9/−13（ns）；C1→C1_true succ **+50/−7**、timeout +9/−80、**off_road +37/−13**（p=0.0009）；IDM→C1 +27/−246、IDM→C1_fill +25/−213（c6_report §2）。
- **候选排序（硬门→success）**：`c1_true` 0.390 > `c1_fill` 0.366 > `c1` 0.304 ≈ `c2` 0.302；**全部候选 off（0.128–0.208）超 0.10、coll（0.116–0.132）超 0.08 ⇒ 均不晋级**；`c2` 相对 `c1` 同量级（succ −0.2pp、crawl 33.4 s 同）⇒ warp 机制臂中性确认。
- **诊断量**（同批）：crawl_mean C1 **33.44 s** → C1_fill 21.44 → C1_true **15.32**；low-speed 步占比 0.372→0.256/0.179；a_lat p95 0.68→1.10/1.16；IDM crawl 2.73、speed_ratio 0.757 ⇒ 接口修复解除部分爬行但**横向跟不上**（off 恶化），与 R37 结论同型且更重。

### 5.2 跨集登记（禁直比，仅对照）
C0 0.580→0.588 ｜ C1 0.340→0.304 ｜ C1_fill 0.420→0.366（off 0.12→0.208）｜ C1_true 0.400→0.390（off 0.18→0.178）｜ IDM 0.756→0.742（calib50→clean500；`r38_unified_benchmark.md` §4；历史锚跨集登记表仍待回填，见 §8）。

### 5.3 并发处置：首轮作废 + 冻结重跑（pin）+ overlay
- **首轮（仓库直跑）**：`c0_k1e2` 期间 net/ 被 D1 改动（flagged）；`c1` 期间被 P1 减枝改动并触发 **108 契约断言失败 + 8 NameError（n_error=116）⇒ 作废**；`c1_fill` 被二次改动 ⇒ 作废（旧读数留档 `*_r1_oldtree`）。
- **冻结重跑**：`git archive 48cb5eb` → `/tmp/opencode/c6_pin`（runs/.venv/datasets 符号链接回仓库）；pin net 树 `0311046b…`、`eval_a3` sha16 `2ff136b84a40ad72`、spec sha256 `087db3f5…`；**本文全部臂读数 = pin 内重跑**（`pin_commit=48cb5eb`）。
- **`c1_true` overlay**：叠加 Q4A `env/obs/long_exog.py`（sha256 `2366a626…`；纯函数 numpy-only）——pin 内 `m7_unit_check --phase unit` **A–D 全过**（A row-wise bitwise / B 933·933 / C 1055·1055 / D 5.7e-14），已记入该臂 `pin_note`（c6_report §4b）。
- **跨树复现核验（r1 vs pin）**：`idm_pp`/`c0_k1e2` 逐 episode 结局字段 **500/500 全同**（各 1–2 条 steps 微差）⇒ D1（默认关）/P1（oneshot 删除）对该协议读数无结局级影响（跨树复现证据，非逐位断言）。

### 5.4 待补（C6 §6/§7）
① **Eval500 3-seed**（paired bootstrap/McNemar + `seed_mode` 变体）｜② Q4A+A4 全量落地后重入基准（新 `net_tree_sha16` ⇒ 新 hash；`c1_gfix`/`c1_fill_gfix` 已有 calib50 读数）｜③ P1 减枝后 oneshot 历史 ckpt 复现走 `tools/legacy/oneshot_path.py`｜④ calib50 历史锚跨集登记表｜⑤ protocol hash 原生接入评测入口 + clean500 是否同 sha 复制入 `env/specs/`｜⑥ **pin 易失**（复现命令+校验锚已入文档）。§7 提案（层级确认/复制/hash 接入/登记表）待 CG。

---

## 6. Gate A–D 对照（GPT §五）

| Gate | 判据 | 状态 | 依据 |
|---|---|---|---|
| **A 架构契约** | 唯一动作源；Policy×6；WM×5；无生产 oneshot；无生产 plan_override；辅输出不进 LQR；训练部署调用链一致 | ✅ | P2b-⑤ 单出口（`action_generator_names()` 单元素 = `("policy.structured_action",)`）；D1 ChainHooks **8/8 断言**（6 次调用/同实例/5 次推进/`a_k↔z_k` 逐位/aux 不进 LQR/无 oneshot·plan_override/ST-GNN 总调用=6 单列）；P1 删除 + CI 守卫③④；`contract_*` 全 1.0。**注**：结构化链仍仅在 `.slim`，官方入口缺口另记（§8-未决⑦） |
| **B 模块定位** | 五问必须量化闭合（禁"多方面共同影响"） | ✅ | §3 全表（A1 逐 h/E0–E6/S_* + A3 11 臂 + C1 8 变体 + C3 五档 + C2 oracle 上界） |
| **C World Model** | ①h1–h6 误差不指数增长 ②解析 anchor 优于旧运动先验 ③learned residual 进一步改善 ④真实态 vs WM 态差距可量化 ⑤动态距离/TTC 随 rollout 更新、不静默 hold | **4/5**（缺③） | ①W ds MAE 0.146→0.664 近线性、r 0.983→0.719（A1）；②生产已实现 margin/d2end 派生（与基线逐位同）+ A4 修两缺陷（横向 term_lat 0.895→0.562）；④A1 全表；⑤Q4-A t0 段生产者真值 + 缺键 fail-fast（k≥1 hold 为 C3 裁定的显式口径，kin_b 收益≈0）；**③=学习残差改善解析 anchor ⇒ 属 C3（D8 待 CG）**（§5.80） |
| **D 统一闭环** | 同 protocol hash；Calib50→Clean500→Eval500；关键版本（旧基线/候选/IDM+PP）重跑；主排序硬门→success | ✅（正式声明待 Eval500） | §5 全表；IDM 0.742 复现 v7 锚；protocol hash v1 + common hash 全臂一致；待= Eval500 3-seed + clean150 层级 CG 拍板（§8） |

---

## 7. 舞台资产（本轮沉淀）

| 资产 | 内容 | 状态/证据 |
|---|---|---|
| **Probes/Hooks（D1）** | `net/probes.py` ~1050 行：①状态来源②Policy③WM④Memory⑤六步调用链（8 断言；ST-GNN 总调用单列）⑥连续路径/控制（首次越界/OD 碰撞）⑦参数梯度账本（**step 1/3/6 loss 各自反向**；update/ckpt 读数）；统一出口 `ProbeRecorder`；**全部默认关**（关闭时前向逐位、无 RNG/optimizer/ckpt 影响） | 入库 `248a3ed`；14 测试；987 passed；`runs/round38/d1/{probe_tensors.pt,grad_ledger.json,breach_scan.json}`（d1_report §0/§4） |
| **path 生产者（Q4-A）** | `env/obs/long_exog.py`（公式**单一出处**，M7 注入器/I1 抽取改调同函数）+ `path_field.py`（t0 段 8 列 + 同源 `__present`）+ `builder` 挂载；`net/path_head` dynamic 两列缺键 **fail-fast**；"头入口雷达" `c3_eval_probe.py` 纳入常规工具 | 入库 `9577078`；1005 passed；契约门 (a)–(d) PASS；雷达 7/7 在场 0→1（q4a_report §0/§5.1） |
| **forward_mode（H 组）** | 四态枚举 + 映射表 + `resolve_forward_mode`（纯函数）；旧布尔=兼容 shim；混用/非法 **ValueError/TypeError**；`wm_detach` 撤形参 | 入库 `f81da17`；11 例布尔↔枚举逐位（p3_report §2-S4） |
| **legacy 载体** | `tools/legacy/`：`oneshot_path.py`、`k_anchor/`（9 文件快照）、`legacy_ego_warp.py`、`legacy_policy.py`、`legacy_key_mapper.py`（含 `PREFIX_RENAME_MAP`）、`checkpoint_migration.py`、`monitoring_reader.py`、`others_layout.py`；`tools/diagnostics/legacy_policy_diag.py`；warp oracle fixture `tests/fixtures/r38_p2b_warp/` | 各波入库；CI 守卫①禁 prod 再引用 |
| **CI 守卫** | `tests/test_r38_ci_guards.py` **5 条 10 例**（含负控制）：禁 prod 引用 legacy / 禁 default config deprecated 键 / 禁 forward oneshot·plan_override（AST+运行时）/ 禁双动作源 / 禁缺键静默 0 | 入库 `f81da17`（p3_report §3d） |
| **四交付物** | `code_reachability_audit.md`（67 符号；0 纯 UNREACHABLE；§8 落地状态）/ `deprecation_manifest.yaml`（46 项；27 回填 + `landing` block）/ `runtime_entrypoint_matrix.md`（结构化链仅 `.slim`；`tools/test.py` 结构化 ckpt **fail-fast**）/ `checkpoint_migration_contract.md`（522 ckpt 普查；168/186 键面；mapper 已实现、loader 未接线） | 入库 `802468b`/`71b8153`（+`c5_*` 机器产物 `.slim/.../round38/`） |

---

## 8. 问题与风险

### 8.1 未决（跨文档合并，10 项）
| # | 项 | 处置/出处 |
|---|---|---|
| 1 | **D8/C3 训练放行**（学习型想象步动力学：解析基+学习残差；门=①A1 复算 slow+accel Σds≥12.8 且全样本≤2.31 ②`c1_evo` 爬行解除且安全不劣化 ③否则转判执行契约） | **待 CG**（§5.71；§5.80 待办②；`r37_p0b_approval_packet.md` §6） |
| 2 | **B 类批次**：掩码进头（`long_exog(8)+long_present(8)`）+ `static_distance` 下沉 | 与下一次重训同批（分类文档 §6.3；§5.80 待办①） |
| 3 | **Eval500 3-seed 正式声明 + clean150 层级去留** | 待 CG（c6 §6-1/§7；§5.80 待办③） |
| 4 | `protocol hash` 原生接入 `eval_a3`/`eval_runner`（+`seed_mode` 扩展） | 工程待办（c6 §6-5/§7-3；§5.80 待办④） |
| 5 | **pin 易失**：`/tmp/opencode/c6_pin` | 复现命令+校验锚已入文档（c6 §6-6；§5.80 待办⑤） |
| 6 | **ckpt migration 未接线 loader**：官方 loader 对旧键面仍 fail-fast；`INIT_FROM_CALIBRATION` 空表；`LoadReport`→`metrics.json::meta.ckpt_load` 未接线 | 契约 §5.4/§7；p2a §④-3；p2b §⑦-1/5 |
| 7 | **正式入口装配缺口**：官方链不装载结构化头（168 vs 186 键；structured ckpt `SystemExit(2)`） | 建议显式装配开关（入口矩阵 §1/§5-1） |
| 8 | **训练侧接线缺口**：`BCDataset._obs_keys` 不含 `path_*`（structured 训练 fail-fast=预期）；`collect_expert` 白名单不含 `path_*`（schema v8 决策） | q4a_report §7 |
| 9 | **风险：接口修复后横向成主约束**（clean500 off 0.130→0.178/0.208、a_lat p95↑；@speed 横向质量）⇒ 横向修复须与 C3/J2 配套；κ 限幅表闭环重标定待数据 | R37 §8-3；c6_report §5 |
| 10 | **文档面残留**：manifest E/F/G/H `removal_commit` 仍 **null（19 项）** 待回填；`r38_unified_benchmark.md` 正文状态行"草稿"与 README"已定稿"不一致 | manifest::landing.pending；docs_sync_report §6-2 |

### 8.2 P3 遗留清单要点（执行向，详见 p3_report §④）
- `rollout=/world_model=` 调用点迁移（生产 20+、tools/tests ~50 处；先机械替换后删布尔入口）；`diagnostics` 独立语义未接线；`config/train.yaml::wm_detach` 键退役待下一笔。
- 旧数据集（schema v2/v3，others=28）直喂 fail-fast ⇒ 重采或显式 adapter；F 组生产去依赖唯一 target lane（长期）。
- 证据脚本质断言边角（`d1` 的 `head_has_no_probes_module`、keys 脚本空 diff 布尔口径）；`round36/c_old_eval_a3.py` 的 `--diagnostic-oneshot` 冻结残留（或同款清扫）。

---

## 9. 待办与放行项

**待 CG 放行/裁决（4 项）**
1. **D8/C3 训练放行**（学习型想象步动力学；门×3 见 §8.1-1）——§5.71/§5.80；
2. **B 类批次时点**（掩码进头 + `static_distance` 下沉，与下次重训同批）——分类文档 §6.3；
3. **Eval500 3-seed 正式声明 + clean150 层级去留**——c6 §6/§7；
4. **正式入口装配开关**（结构化头收编官方链，消除 168/186 键面缺口）——入口矩阵 §5-1（建议，未立项）。

**工程待办（可先行）**
1. protocol hash 原生接入评测入口（+seed_mode 扩展；Q4A+A4 落地后重入基准 ⇒ 新 hash）；
2. ckpt migration loader 接线（`migrate_state_dict` 前置 + `LoadReport` 落 `meta.ckpt_load`）；
3. `forward_mode` 调用点迁移（生产 20+、总 ~50 处）+ `diagnostics` 语义 + `wm_detach` 键退役；
4. pin 复现走查（易失快照；命令+校验锚已入 c6 §6-6）；
5. 文档回填：manifest E/F/G/H `removal_commit`（19 项）、benchmark 状态行收口。

---

## 10. 证据索引

- **方案/裁决**：`docs/chatgpt5p6sol_review.md`（R38 方案原文 = `2d670cc` 版，§一–§九）；`docs/deepseekv4p1_argue.md` §5.70–§5.80；本报告；`docs/r37_p0b_approval_packet.md`（D1–D7 + §6/D8）
- **R38 文档交付**：`docs/r38_unified_benchmark.md`（protocol hash v1 + Clean500 + 7 臂）、`docs/r38_path_field_classification.md`（分类定稿 + 编排裁定）、`docs/code_reachability_audit.md`（含 §8）、`docs/deprecation_manifest.yaml`（46 项 + `landing`）、`docs/checkpoint_migration_contract.md`、`docs/runtime_entrypoint_matrix.md`
- **lane 报告**：`.slim/deepwork/p1_v2/round38/{a1,a3,a4,c1,c2,c3,c6,d1,p1,p1b,p2a,p2b,p3,docs_sync}_report.md`
- **机器产物**：`runs/round38/{a1,a1_ckpt_t2term2,a3,a4,c1,c1_ckpt_t2term2,c2,c3,c6,d1,q4a}/`；`runs/round37/{m6,m7}/`；C6 pin = `/tmp/opencode/c6_pin`（`git archive 48cb5eb`）
- **波次 commit**（时间序）：
  | 波次 | commit | 内容（pytest 计数） |
  |---|---|---|
  | 方案输入 | `2d670cc` | GPT R38 方案（chatgpt5p6sol_review §一–§九，+582 行） |
  | A4 | `b9d1721` | step_geometry 默认关 fixed 模式（966→973） |
  | D1 | `248a3ed` | 七类 Probes/Hooks（→987） |
  | Q4-A | `9577078` | `path_*` 生产者补全 + 禁静默 0（→1005） |
  | P1 | `596747c` + `48cb5eb` | A 组减枝（→998） |
  | P2a | `a226285` + `dd0a8c2` | B 组 K-anchor（→1022） |
  | P2b | `939d1a2` | C 组旧 warp 退役 + D 组双实体收编（→1043） |
  | P3 | `f81da17` | F/G/H + E 残留 + CI 守卫 + `eval_a3` 清扫（→1068） |
  | docs 收口 | `802468b`/`99fcd2f`/`fe2f11f`/`74f0bd4`/`2cfd8e8`/`71b8153`/`75c9d64` | §5.72–§5.75 结案（§5.71=`f3c4179` 见 git log）、四波落地 + 基准终稿、生产文档同步批、§5.80（**HEAD**） |
