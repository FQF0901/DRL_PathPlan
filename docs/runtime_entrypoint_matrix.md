# R38-C5 运行时入口矩阵（GPT R38 §八 第三件交付物）

> **目的**：把"哪些入口会真正跑到哪些模块 / 哪些入口会踩到待清理的 legacy 面"变成一张可核对的表，
> 供 R38-C 减枝时判断"删这个符号会不会打断某个入口"，以及 Clean500/Eval500 统一 protocol hash 重跑前
> **固定装配口径**。
> **口径**：入口 = 有 `__main__`/CLI 或显式被 CLI 调用的脚本；`.slim` 侧按 §5.69-② 视为评估/训练链的一部分
> （它们才是当前结构化头的实际运行入口）。引用面数字来自 `docs/code_reachability_audit.md`（同源扫描）。
> **只读**：本文件不改任何入口代码。
> **配套交付物**：`docs/code_reachability_audit.md`（引用面与判定）、`docs/deprecation_manifest.yaml`（逐项弃用清单）、
> `docs/checkpoint_migration_contract.md`（ckpt 键面与迁移层）。

---

## 1. 结论（三条结构事实）

1. **结构化头链不在官方入口**：`tools/train.py`、`tools/test.py`、`pipeline/eval_runner.py`、
   `tools/baseline_eval.py`、`tools/dagger_collect.py` 对 `path_head` **零命中**
   （`pipeline/**` 与 `tools/**` 全目录 `path_head` 引用数 = 0/0/0/0/0）；
   `set_path_head(True)` 的调用只存在于 `.slim`（`a3/eval_a3.py`、`round32/canary_eval.py`、
   `round36/{j1_train,j1_eval,a_hardening_check,k1_contract_gate,h1_compat_check,...}.py`）。
   ⇒ **"正式评测口径"当前等于"旧 PolicyHead 口径"**；结构化头候选要出正式数字，只能走 `.slim` 链或先扩
   `pipeline/eval_runner.py`。
2. **装配口径缺口（可复现数字）**：按 `config/model.yaml`（`plan_anchor.enabled=false`、无 `path_head`）构造的
   `DrivingModel` = **168 键**；
   - 旧正式 ckpt（`runs/BTC20261009-0824_sw_pri512/stage_b/final.pt`，168 键）→ 干净加载（missing/unexpected = 0/0）；
   - K1 结构化 ckpt（`runs/round36/j1/t1_keep/ckpt_epoch001.pt`，186 键）→ **unexpected=18（`path_head.*`）**；
   - v7 K-anchor ckpt（`runs/BTC20261005-0856_v7struct_v5/stage_b/final.pt`，174 键）→ **missing=14 + unexpected=20**。
   `pipeline/eval_runner._load_ckpt_model` 默认 **fail-fast（`SystemExit(2)`）**，仅 `--allow-partial-load` 放行
   ⇒ **用 `tools/test.py` 评测结构化 ckpt 会直接 fail-fast**（不是"数字偏"，是"跑不起来"）。
   `set_path_head(True)` 后模型 = **186 键**，K1 ckpt 干净加载（0/0）、旧 ckpt missing=18（`path_head.*`，由
   `runs/round36/k1/calibration.json` 构造期标定值承接）——这正是 `checkpoint_migration_contract.md` 的迁移路径。
3. **`.slim` 是事实上的评估/部署链**：`a3/eval_a3.py`（calib50/闭环 LQR 评测）、`round32/canary_eval.py`
   （Gate 1/2 离线门）、`round36/{c_old_*}`（新旧工具逐位对照）、`round37/{m6_calib,m7_fill}`（`path_*` 注入实验）、
   `round38/{a1_matrix,a3_matrix,a4_run}`（A 系列裁决/修复）构成完整链路。**减枝若断掉这些引用而不迁移，
   等于删掉当前唯一的可达面**（§5.69-③）。

---

## 2. 官方入口（`tools/` + `pipeline/`）

| 入口 | 触发路径 | 触及模块（调用链） | 进入 legacy？ | 备注 |
|---|---|---|---|---|
| `tools/train.py` | `tools/venv-python tools/train.py --stage {A,B,C}`（`tools/train.sh`） | `pipeline.run_paths` → `pipeline.gl_runtime` → `pipeline.stages.main` → `pipeline.trainer` / `pipeline.phase3_loop` / `pipeline.buffer` / `pipeline.monitoring` → `net.model.DrivingModel`（经 `stages.build_model`）| **是**：B（K-anchor 透传，`build_model` 读 `plan_anchor`，默认关）、H（`rollout=/world_model=` 布尔组合；`wm_detach` 透传）、G（`legacy_tags` 仅在 `--monitor-legacy-tags`）| `path_head` 零引用 ⇒ 官方训练只训旧 PolicyHead 口径；结构化头训练在 `.slim`（见 §3） |
| `tools/test.py` | `tools/venv-python tools/test.py --policy ckpt --ckpt ...`（`tools/test.sh`） | `pipeline.run_paths` → `pipeline.eval_runner.main` → worker：`_load_ckpt_model`（fail-fast）→ `_CkptController` → `model(obs, rollout=need_plan, world_model=False)` → `env.tracking`（LQR/exact）| **是**：H（两布尔）、B（`plan_anchor` 透传，`eval_runner.py:880`）| `need_plan = (eval_reference == "plan")`；结构化 ckpt 默认致命（见 §1-2） |
| `pipeline/eval_runner.py`（模块 CLI） | `tools/venv-python -m pipeline.eval_runner --policy {baseline,ckpt}` | 同 `tools/test.py`（`tools/test.py` 只是参数规整 + run 目录 + 透传） | **是**：同 `tools/test.py` | 冻结验证集 1000 条、KPI 口径、Wilson CI、fail-fast 三计数（`metrics.json::meta.ckpt_load`） |
| `tools/baseline_eval.py` | `tools/venv-python tools/baseline_eval.py` | `env.expert.pure_pursuit_idm` + `env.metadrive_env` + `env.scenario.spec`（**无 net**） | **否**（规则策略；仅 env 侧 F 组 target-lane = KEEP 项） | 规则基线（IDM+PP 同源专家），是 Eval500 对照臂 |
| `tools/dagger_collect.py` | `tools/venv-python tools/dagger_collect.py --ckpt ... --specs ... --labeler {pure_pursuit,idm}` | `pipeline.eval_runner.load_config` → worker：`_CkptController`（评测同款驱动）+ 规则专家标签 | **是**：B（`plan_anchor` 透传，`dagger_collect.py:1294`）、H（经 `_CkptController` 的 `rollout/world_model` 布尔） | **P0-B 教师蒸馏的现成入口**；注意：它**不装载结构化头** ⇒ 蒸馏学生若是 structured 口径，须先扩装配（与 §1-1 同源问题） |
| `pipeline/phase3_loop.py`（`tools/train.py --phase3-loop`） | `tools/venv-python tools/train.py --phase3-loop ...` | `pipeline.trainer`（PPO collect/update，cheap path）+ `pipeline.vector_env` | **是**：H（`rollout=False, world_model=False` cheap path，`trainer.py:7422/7601/7677` 注释明示 V9 口径） | 执行契约/PPO 相关；与 P0 教学训练不同链 |
| `tools/collect_expert.py` | `tools/venv-python tools/collect_expert.py ...` | `env/**`（规则专家录制 BC 数据） | **否**（无 net） | BC 数据集生产；`datasets/BTC*_expert*` |
| `tools/mine_hard.py` / `tools/make_dagger_pools.py` / `tools/il_report.py` | 各自 CLI | `pipeline.hard_mining` / 数据集统计 / 报告 | 边缘：`il_report.py` 记录键含 `wm_detach`（H 组元数据面） | 数据/报告工具，不跑模型 forward |
| `tools/fit_plan_anchors.py` | `tools/venv-python tools/fit_plan_anchors.py` | `net.anchor`（K-anchor 拟合） | **是**：B（LEGACY_REPRO，随 K-anchor 归档） | 产出 `config/plan_anchors_k6.json` |
| `tools/paired_eval.py` / `tools/measure/*` / `tools/diagnostics/*` | 各自 CLI | `paired_eval` 读 `episodes.csv`（无 net）；`diagnostics/forensics_closed_loop.py` import `net.model.{arc_step,compose_pose}`；`controls/probe_*` 读 ckpt/张量 | 边缘：`probe_freeze_audit.py` 读 `net.encoders` 维度常量（G 组 others 面） | 分析与取证工具；**不含模型 forward 主链** |
| `tools/visualize.py` / `tools/debug_rollout_viz.py` | 各自 CLI | `pipeline.eval_runner` 复用 + 渲染 | **是**：H（`rollout=` 显式传参，`debug_rollout_viz.py:757`） | 可视化部署记录 |

## 3. `.slim` 评估 / 训练 / 部署链（结构化头的实际运行入口）

| 入口 | 触发路径 | 触及模块 | 进入 legacy？ | 备注 |
|---|---|---|---|---|
| `.slim/deepwork/p1_v2/a3/eval_a3.py` | `tools/venv-python .slim/.../a3/eval_a3.py --ckpt ... --path-head {off,structured} --lane-anchor {off,inject} --long-exog-fill {off,neutral,true} --step-geometry {production,fixed} --tail {repeat,scale,chain} --tracker {lqr,exact}` | `pipeline.eval_runner`（装配/装载/worker 复用）+ `round36/c_lane_anchor.py`（inject 注入口）+ `round37/m6_calib.py`/`m7_fill.py`（`long-exog-fill` 注入）+ `a3_tails.py`（尾段口径） | **是（全量）**：A（`--diagnostic-oneshot` 守卫）、D（`--path-head`/模式读出）、E（`--long-exog-fill` 注入 `path_*`）、H（`--step-geometry`/尾段/rollout 口径）、F（`--lane-anchor inject`） | calib50 闭环评测的事实标准入口（M2–M7、C1/C2/A1/F3 全部经它）；`--canary-ckpt` 装载路径 |
| `.slim/deepwork/p1_v2/round32/canary_eval.py` | `tools/venv-python .slim/.../round32/canary_eval.py --arm {base,...} --path-head ... --diagnostic-oneshot` | 自建装载（oneshot 守卫）+ Gate 1（锚点数据集）→ Gate 2（闭环/前向一致性）+ `--action-ego/--ego-phys`（R32 canary 惰性模块） | **是**：A（守卫）、D（`--path-head`）、B（`num_anchors` 读出 `:658`）、H（rollout/世界模型口径） | Gate 0–4 与 `canary_monitor.sh` 的后端；`--strict-load/--fill-from` 控制装载 |
| `.slim/deepwork/p1_v2/round36/j1_train.py`（+ `j2_train_stripped_check.py`） | `tools/venv-python .slim/.../round36/j1_train.py --task {t1_keep,...} --chain imagination` | `StructuredPolicyHead` 训练接线器（allowlist `path_head.` + `plan_head.fusion.2.`）+ `path_exec_v1` 数据 + 契约/梯度/smoke 三组门 | **是**：A（`path_head_mode` 声明与断言）、D（set_path_head 装配）、E（`long_exog` 列映射）、F（`--target-lane-dropout`） | **K1 唯一训练入口**；J2 = 其"剥掉 J2 新增"的逐位对照副本 |
| `.slim/deepwork/p1_v2/round32/canary_train.py` | `tools/venv-python .slim/.../round32/canary_train.py --arm {ac1,p_only,ac1p}` | Stage-A 最小子集复刻 + `set_action_ego(True)` / `set_ego_phys(True)`（惰性 canary 模块） | **是**：H（状态链 detach 语义与 stages 一致）、G（others/维度常量） | 产出 `st_gnn.action_ego.*`/`ego_phys.*` 键的 ckpt（CHECKPOINT_ONLY 族） |
| `.slim/deepwork/p1_v2/round36/c_old_{canary_eval,eval_a3,p1_deploy_record}.py` | 与新版对跑（`c_bitwise_check.py` 驱动） | 冻结的历史副本（bitwise 对照基线） | **是**：A/D/E/H 全量（副本口径） | **迁移时的"逐位对照基线"**；A 组改动必须重跑 c_bitwise |
| `.slim/deepwork/p1_v2/round36/{a_hardening_check,k1_contract_gate,h1_compat_check,h1_skeleton_smoke,c_bitwise_check,c_selfproof}.py` | 各自 CLI | 装载守卫 / L-A 契约门 / K1 契约 / 逐位自证 | **是**：A（oneshot 守卫）、D（契约断言）、E（`path_*` 供给） | **当前 A/D 组守卫的唯一载体**（`tests/` 零覆盖，见审计 §5） |
| `.slim/deepwork/p1_v2/round37/{m1_gate_a,b_warp_compat}.py` | 各自 CLI | `model.set_ego_warp(w1/w2)` 逐位 oracle + `assemble_pose_k` 等价门 | **是**：C（legacy warp 的唯一使用者） | **C 组删除的硬前置**：先把 w1/w2 期望冻结为 fixture |
| `.slim/deepwork/p1_v2/round37/{m2_run_c0c1,m4_run,m5_run,m6_run,a3 等}.py` + `m6_calib.py`/`m7_fill.py` | 驱动 `eval_a3.py`（C1 协议：`--ld-mode semantic --rollout-semantic static_role_only --moe-off --tracker lqr --tail chain --canary-ckpt ... --path-head structured --lane-anchor inject`） | 同上 + `path_*` 注入器 | **是**：E 核心（`long-exog-fill` 中性/真值）、D、H | M6/M7 即"缺键填 0 / 中性填充 / 真值回填"三形态实验 |
| `.slim/deepwork/p1_v2/round38/{a1_matrix,a3_matrix,a4_run,a4_gate}.py` | 各自 CLI（`a4_run.py <arm>` → `eval_a3`） | A1 裁决矩阵 / A3 解析修复臂 / A4 几何修复闭环 | **是**：E（几何/接口）、D、H | 在飞链；A4 落地后需重跑本审计（工作树 `net/path_head.py` 有未提交改动） |
| `.slim/deepwork/p1_v2/round26|27/r2*_train.py` 等历史训练器 | 各自 CLI | 历史结构训练（ld_mode/语义接线期） | 边缘：G（others 重排）、B（锚） | `net/anchor.py` 归档时须扫一遍这些引用面 |
| `runs/**/*.py`（`runs/reward_viz/scripts/*`、`runs/p1_v2/round15a/*`） | 各自 CLI | 奖励可视化 / 一次性取证 | 边缘：H（`rollout=` 参数） | **产物目录脚本**（不在 git 白名单内），减枝时只作参考不计入判定 |

## 4. Legacy 面 × 入口 触达矩阵（减枝影响面速查）

| legacy 面（组） | `tools/train.py` | `tools/test.py` / `eval_runner` | `tools/dagger_collect.py` | `.slim/a3/eval_a3.py` | `.slim/round32/canary_eval.py` | `.slim/round36/j1_train.py` |
|---|---|---|---|---|---|---|
| A oneshot / plan_override / 别名 | —（官方链零引用） | —（`eval_runner` 零引用） | — | ✅ `--diagnostic-oneshot` | ✅ `--diagnostic-oneshot` | ✅ 模式断言 |
| B K-anchor | ✅ 透传（默认关） | ✅ 透传（默认关） | ✅ 透传（默认关） | ✅ 透传 | ✅ `num_anchors` 读出 | — |
| C 旧 `set_ego_warp`（w1/w2） | — | — | — | —（用的是 `--ego-warp-mode analytic` = 主路径） | — | — |
| D 双实体 / ablation | —（仅默认关路径跑旧头） | —（同上） | — | ✅ `--path-head` | ✅ `--path-head` | ✅ 装配 |
| E `path_*` 旁路 | — | — | — | ✅ `--long-exog-fill` | — | ✅ 数据列映射 |
| F target-lane / 外部选道 | — | — | —（规则专家自带 target） | ✅ `--lane-anchor inject` | — | ✅ `--target-lane-dropout` |
| G legacy monitoring / others-28 | ✅ `--monitor-legacy-tags`（默认关）+ others 重排（旧数据自动触发） | —（旧数据经 `mem_from_obs` 重排） | —（同左） | —（同左） | —（同左） | —（同左） |
| H cheap-path 布尔 / `wm_detach` | ✅ 多处 | ✅ `rollout=need_plan, world_model=False` | ✅ 经 `_CkptController` | ✅ 尾段/几何口径 | ✅ | ✅ |

> 读法：`✅` = 该入口会实际触达该 legacy 面（删代码前必须处理）；`—` = 不触达（删代码时无需改该入口，
> 但仍需保 `tests/` 与 `.slim` 其它脚本的同步面）。
>
> 精确清单（A 组 eval 命中 8 文件）：`.slim/.../a3/eval_a3.py`、`round32/{canary_eval,p1_deploy_record}.py`、
> `round36/{a_hardening_check,c_old_canary_eval,c_old_eval_a3,c_old_p1_deploy_record,k1_contract_gate}.py`
> ——**`tools/` 与 `pipeline/eval_runner.py` 对 A 组零引用**（这也是审计 §0-3 的入口缺口证据之一）。

## 5. 缺口与建议（供 CG 裁决）

1. **装配口径缺口**（§1-1/1-2）：正式评测入口不装载结构化头，导致
   (a) 结构化候选无法用 `tools/test.py` 出正式数字；(b) `--allow-partial-load` 一旦被用来"绕过"，
   会产出**表观通过但实际未装载结构化头**的数字（这正是 P0-A fail-fast 要防的）。
   建议：在 `pipeline/eval_runner` 侧增加**显式装配开关**（`forward_mode`/`policy_head` 口径，与 H 组枚举同批），
   并把"ckpt 键面 ↔ 装配开关"一致性检查写进 `_load_ckpt_model`（缺 `path_head.*` 且声明 structured ⇒ fail-fast 文案直接给出修法）。
2. **P0-B 教师蒸馏入口缺口**：`tools/dagger_collect.py` 复用 `_CkptController`（旧头口径）⇒ 若学生/教师走 structured，
   需同批扩装配（否则 DAgger 的 roll-in 与最终部署不同构，违反"训练部署不同构"守则）。
3. **Clean500/Eval500 统一 protocol hash**：必须**先冻结装配口径**（哪一套入口/开关组合），否则同 hash 下
   `.slim` 链与 `tools/test.py` 跑的不是同一个模型口径；建议以 §3 的 C1 协议串为基线，并在 `metrics.json::meta`
   固化 `protocol_hash`（含装配开关快照）。
4. **减枝顺序对入口的影响**：C 组（先冻结 oracle）→ A（同步 8 个 `.slim` 工具 + 保留 L-A 断言）→ D（先做键迁移器）
   → B（归档）→ E（Q4 裁决后 schema 冻结）→ G → H（枚举化，附布尔→枚举映射与逐位对照）。详见
   `docs/code_reachability_audit.md` §3 与 `docs/deprecation_manifest.yaml::meta.migration_order`。

## 6. 复现与证据

```bash
# 引用面（本文件的数字来源）
tools/venv-python .slim/deepwork/p1_v2/round38/c5_reach_scan.py
# ckpt 键面（§1-2 的 168/186/174 与 missing/unexpected）
tools/venv-python .slim/deepwork/p1_v2/round38/c5_reach_scan.py --ckpt-keys
# 键族普查（522 ckpt）
tools/venv-python .slim/deepwork/p1_v2/round38/c5_reach_scan.py --ckpt-families
```

- §1-2 的模型键面核对（可复现，CPU）：
  `build_model(load_config("config/model.yaml")).state_dict()` = 168 键；
  `model.set_path_head(True)` 后 = 186 键；对三代 ckpt 做 `missing/unexpected` 计数（见 `docs/checkpoint_migration_contract.md` §2）。
- 入口触达明细：`c5_reach_scan.json` 的逐符号 `files[bin]` 清单（prod/train/eval 三面已按入口粒度收集）。
