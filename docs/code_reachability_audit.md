# R38-C5 代码可达性审计（GPT R38 §六 四判据 × §七 八组候选）

> **出处**：`docs/chatgpt5p6sol_review.md` §六（废旧代码清理四判据）/ §七（8 组优先候选）/ §八（交付物清单）；
> 执行口径沿用我方案 §5.69 三条修正（本文件 §4 逐条核对）。
> **性质**：只读审计 + 文档（不改任何被审计代码）。机器原始表 = `.slim/deepwork/p1_v2/round38/c5_reach_tables.md`
> （由 `c5_reach_scan.py` 生成，本文件直接内嵌，二者同源）。
> **生成命令（可复现）**：

```bash
tools/venv-python .slim/deepwork/p1_v2/round38/c5_reach_scan.py            # → c5_reach_scan.json + c5_reach_tables.md
tools/venv-python .slim/deepwork/p1_v2/round38/c5_reach_scan.py --ckpt-keys        # → c5_ckpt_keys.json（三代代表 ckpt 键面）
tools/venv-python .slim/deepwork/p1_v2/round38/c5_reach_scan.py --ckpt-families    # → c5_ckpt_families.json（522 ckpt 键族普查）
```

---

## 0. 摘要（结论先行）

1. **零纯 UNREACHABLE**：67 个符号里 **65 个 `REACHABLE`**，仅 2 项 `TESTS_DOCS_ONLY`
   （`encoders.lane/ttc` 旧 ckpt 键面——只存在于二进制 ckpt，文本零命中；`forward_mode`——方案提案、尚未实现）。
   ⇒ 本轮减枝**不能靠"看起来没用"**，必须走契约合法性（§六-2）+ 实验价值（§六-3）+ ckpt 兼容（§六-4）三条判据；
   这与 GPT 方案开篇"废旧代码不能靠'看起来没用'直接删除"一致。
2. **真正的删枝驱动 = 契约合法性**：判定分布 = `ARCHITECTURE_FORBIDDEN` **19**（A 8 + D 2 + E 9）、
   `LEGACY_REPRO` **24**（B 10 + C 3 + D 2 + F 1 + G 5 + H 3）、`CHECKPOINT_ONLY` **3**（B 2 + G 1）、
   `KEEP` **21**（不入 deprecation_manifest）。
3. **本轮最重要的结构性发现（影响减枝顺序）**：**结构化头链（`path_head`）不在任何官方入口**——
   `tools/train.py`、`tools/test.py`、`pipeline/eval_runner.py`、`tools/baseline_eval.py`、`tools/dagger_collect.py`
   对 `path_head` **零引用**（`pipeline/**` 与 `tools/**` 的 `path_head` 命中数为 0）；`path_head` 的训练/评测/部署
   全部在 `.slim`（`j1_train.py`/`j2_train_stripped_check.py`/`a3/eval_a3.py`/`round32/canary_eval.py`/`round37/m*.py`）。
   ⇒ **"迁 tools/legacy + 删主链"必须以 `.slim` 评估链同步为前提**（§5.69-③ 已预警），且 Clean500/Eval500 统一
   protocol hash 重跑前必须**先裁决模型装配口径**（否则正式评测跑的仍是旧 `PolicyHead` 口径）。
4. **D/E 组的 prod 爆炸半径极小**（D：`net/{model,path_head,probes}.py` 3 文件 79 行；E：仅 `net/path_head.py`
   1 文件 30 行），成本不在 net 侧，而在 `头输入 schema 冻结 → 重训 → .slim 注入口`。
5. **C 组顺序依旧是硬约束**：legacy `set_ego_warp(w1/w2)` 是 analytic warp 的逐位 oracle
   （`.slim/round37/m1_gate_a.py`、`.slim/round33/b_warp_compat.py`）⇒ **先冻结 fixture，再删实现**（§5.69-① 不变）。
6. **G 组范围按 §5.69-② 扩到 obs 层**：`net/mem.py`（`_remap_legacy_others`/`OTHERS_LEGACY_DIM`）+ `net/encoders.py`
   常量 + `pipeline/monitoring.py` 的 `_LEGACY_*` 族（20 处）同批；**另有 4 个 ckpt 键族**（见 `checkpoint_migration_contract.md`）。

---

## 1. 判定口径（与脚本同源）

### 1.1 引用面分类（bins）

| bin | 路径口径 | 说明 |
|---|---|---|
| **prod** | `net/**`、`env/**`、`reward_model/**`、`config/**` + `pipeline/{frames,gl_runtime,run_paths,monitoring}.py` | 生产运行可达面 |
| **train** | `tools/{train,collect_expert,dagger_collect,mine_hard,make_dagger_pools,fit_plan_anchors,il_report,run_config,reward_audit*}.py` + `pipeline/{trainer,stages,phase3_loop,buffer,hard_mining,vector_env,residual_classes,semantic_labels}.py` + **`.slim` 训练链脚本**（文件名含 `train`/`collect`/`extract`，如 `j1_train.py`/`i1_extract_dataset.py`） | 训练入口可达面 |
| **eval** | `pipeline/eval_runner.py`、`tools/{test,baseline_eval,paired_eval,make_eval_spec,debug_rollout_viz,visualize}.py`、`tools/{diagnostics,measure,controls}/**` + **`.slim` 评估/部署链脚本**（文件名匹配 `eval/canary/deploy_record/gate/equivalence/compat/hardening/bitwise/selfproof/matrix/paired/offline_check/calib/fill/smoke/report/run`，共 15 类模式） | 评测/部署入口可达面（§5.69-② 修正：评估链在 `.slim`） |
| **test** | `tests/**` | 不计入可达性判定；决定"删除时同步迁移测试"的同步面 |
| **slim** | `.slim/**` 其余（实验/分析/报告脚本） | 不计入判定；作为"证据链载体"参考 |
| **runs** | `runs/**.py|.sh|.yaml|.md` + `runs/**.json`（≤2 MB） | 产物/日志面 |
| **docs** | `*.md`（`docs/**`、各 `README.md`；`.slim`/`runs` 内 md 归各自 bin） | 文档面 |

### 1.2 四判据 → 五项判定

- **可达性（§六-1）**：`prod ∪ train ∪ eval` 任一命中 ⇒ `REACHABLE`；仅 `.slim`(其余)/`runs` ⇒ `SLIM_RUNS_ONLY`；
  仅 `test`/`docs`/零命中 ⇒ `TESTS_DOCS_ONLY`（**= §六-1 的 `UNREACHABLE`**）。
- **契约合法性（§六-2）** ⇒ `ARCHITECTURE_FORBIDDEN`：违反守则六条之一（第二动作生成器 / 一次输出六步 /
  绕过 WM / 外部替网络选道 / 缺失静默填充 / 训练部署不同构）。
- **实验价值（§六-3）** ⇒ `LEGACY_REPRO`：仅历史消融/复现价值 → 迁 `tools/legacy/` 或 `experiments/archive/`，
  不得留在 `net/`、`pipeline/`、`config/default.yaml`、正式 CLI。
- **ckpt 兼容（§六-4）** ⇒ `CHECKPOINT_ONLY`：仅为旧 ckpt 加载保留 → 隔离进 `checkpoint_migration.py`/`legacy_key_mapper.py`，
  不参与正式 forward。
- **`KEEP`**：四判据全过（可达 + 合法 + 非历史消融 + 非 ckpt 专用）⇒ 不在 `deprecation_manifest.yaml` 中立项。

### 1.3 三个附加维度（§六-1 要求的"字符串/配置键/ckpt 键"）

| 维度 | 检测式 | 用途 |
|---|---|---|
| `string` | 引号包裹且含 token（含 f-string / CLI 模式串） | 反射/字符串装配（`getattr`/`set_attr`/CLI 透传）风险面 |
| `config` | `config/**` 行命中 + 任意 yaml/json 的 `key:` 键行 | §八 CI"禁 default config 出现 deprecated 键"的扫描面 |
| `ckpt` | 引号包裹的 state_dict 键名（**去重计数**，键名清单落 JSON） | §六-4 ckpt 兼容 + `removal_commit` 前置的键面证据；二进制 ckpt 由 `--ckpt-keys`/`--ckpt-families` 单独普查（见 `docs/checkpoint_migration_contract.md`） |

### 1.4 扫描面与自证

- 扫描文本文件 **6294** 个；**跳过 119 个超大文件**（`runs/**.json` > 2 MB，占 runs 文本量约 39.6 GB/40.4 GB）。
  跳过项只影响 `runs` 计数，**不影响任何判定**（runs 不计入可达性）。
- 排除目录：`.git`/`.venv`/`datasets`/`checkpoints`/`__pycache__`/`.pytest_cache`；
  **未排除 `.slim` 与 `runs` 脚本**（§5.69-② 硬要求）。
- 自指排除：本文件的四份 C5 文档 + `.slim/.../c5_*` 产物不参与扫描（否则审计产物会被计入引用面）。
- **快照时点**：2026-10-15 18:33。扫描期间仓库有并发 lane 在改 `net/**`（R38-A4 已入库 `b9d1721`，
  其后 `probes.py`/`mem.py`/`model.py` 等仍在写）⇒ 本文件的行命中数**会随该 lane 落定而漂移**；
  判定口径不变，但减枝前应重跑脚本刷新数字（见 §7）。
- 计数口径：bin 列 = **行命中数**；`ckpt` 列 = **去重键名数**。注释/文档串同计（保守高估，不会造成"漏引用"）。
  **narrative（§0/§3/§4）的数字取生成时快照；以 §2 内嵌表为准**（并发 lane 改 `net/**` 会使行命中漂移，
  必要时重跑脚本刷新；判定与结论不受影响）。

---

## 2. 组级汇总（机器表）

> 格式 `行命中/各符号文件数之和`（同文件被多符号命中会重复计数）；`·` = 0。判定分布 = 该组符号的建议判定计数。
> §3 narrative 里引用的"文件数"用**去重后**的口径（`c5_reach_scan.json::file_counts`），两者不要混比。

<!-- 内嵌 c5_reach_tables.md（同源产物）：组级汇总表 + 8 张明细表 + 逐符号证据 -->

- root: `/workspace/01_Proj/DRL_PathPlan`；扫描文本文件 6311（跳过超大 119：`runs/**.json>2MB` 等）

## 组级汇总（行命中 / 各符号文件数之和）

| 组 | prod | train | eval | test | slim | runs | docs | 判定分布 |
|---|---|---|---|---|---|---|---|---|
| **A** | 89/16 | 2/2 | 181/31 | · | 37/24 | 55/21 | 64/30 | ARCHITECTURE_FORBIDDEN×8、KEEP×1 |
| **B** | 115/20 | 79/14 | 24/8 | 71/17 | 87/60 | 606/352 | 157/76 | CHECKPOINT_ONLY×2、LEGACY_REPRO×10 |
| **C** | 60/15 | · | 63/15 | · | 83/33 | 25/4 | 38/16 | KEEP×4、LEGACY_REPRO×3 |
| **D** | 79/15 | 10/7 | 128/44 | 11/5 | 43/30 | 151/142 | 37/24 | ARCHITECTURE_FORBIDDEN×2、KEEP×6、LEGACY_REPRO×2 |
| **E** | 30/9 | 8/6 | 163/84 | · | 132/59 | 330/72 | 38/19 | ARCHITECTURE_FORBIDDEN×9 |
| **F** | 138/24 | 69/19 | 225/65 | 50/12 | 386/94 | 619/326 | 79/38 | KEEP×7、LEGACY_REPRO×1 |
| **G** | 67/10 | 19/5 | · | 27/7 | 2/2 | · | 14/10 | CHECKPOINT_ONLY×1、KEEP×1、LEGACY_REPRO×5 |
| **H** | 29/9 | 161/31 | 187/75 | 142/62 | 206/85 | 1114/66 | 86/45 | KEEP×2、LEGACY_REPRO×3 |

> 口径：prod/train/eval = 生产·训练·评测三类**运行入口可达面**；test/slim/runs/docs 不计入可达性判定（但 `.slim` 评估链已按 §5.69-② 归入 eval，见各组明细的文件清单）。「文件数」列 = 组内各符号文件数**之和**（同一文件被多个符号命中会重复计数；去重后的清单见 JSON 的 `file_counts`）。

## A 组｜一次性六步路径代码（oneshot / plan_override / 别名）

| 符号 | kind | prod | train | eval | test | slim | runs | docs | 行数合计 | string | config | ckpt | 建议判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `oneshot 模式串（PATH_HEAD_MODES）` | string | 36 | · | 82 | · | 17 | 11 | 22 | 168 | 168 | · | 2 | ARCHITECTURE_FORBIDDEN |
| `allow_oneshot（守卫形参）` | ident | 6 | · | 9 | · | 1 | · | 4 | 20 | 3 | · | · | ARCHITECTURE_FORBIDDEN |
| `--diagnostic-oneshot（CLI 守卫）` | string | 3 | · | 65 | · | 1 | · | 6 | 75 | 75 | · | · | ARCHITECTURE_FORBIDDEN |
| `path_head_mode()（模式读出）` | ident | 8 | 2 | 18 | · | 6 | · | 3 | 37 | 25 | · | · | ARCHITECTURE_FORBIDDEN |
| `plan_override（钉住尾段）` | ident | 17 | · | · | · | 1 | · | 15 | 33 | 1 | · | · | ARCHITECTURE_FORBIDDEN |
| `one_shot 形参` | ident | 6 | · | · | · | 1 | · | · | 7 | 1 | · | · | ARCHITECTURE_FORBIDDEN |
| `one_shot_plan() 方法` | ident | 3 | · | · | · | · | · | 1 | 4 | · | · | · | ARCHITECTURE_FORBIDDEN |
| `compose_path_plan() 函数` | ident | 5 | · | 4 | · | 2 | · | 4 | 15 | 2 | · | · | KEEP |
| `PathExecutionHead 别名` | ident | 5 | · | 3 | · | 8 | 44 | 9 | 69 | · | · | · | ARCHITECTURE_FORBIDDEN |

> oneshot/plan_override/别名 = 唯一动作源守则的直接违反项（第二动作源+一次输出六步）；`compose_path_plan` 是 structured 模式 plan 组装的内部实现（非 oneshot 专属），单列。

### A 组逐符号证据（每项 1 行：入口侧命中 → 其它命中）

- **oneshot 模式串（PATH_HEAD_MODES）**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/model.py:63`、`net/model.py:69`、`net/model.py:71`；eval=`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:47`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:48`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:49`；slim=`.slim/deepwork/p1_v2/round36/c_old_p1_deploy_record.diff.txt:62`、`.slim/deepwork/p1_v2/round36/c_old_p1_deploy_record.diff.txt:148`、`.slim/deepwork/p1_v2/round36/k1_manifest.json:11`；runs=`runs/round36/a/hardening.md:1`、`runs/round36/a/hardening.md:12`、`runs/round36/a/hardening.md:14`；docs=`net/README.md:80`、`net/README.md:85`、`docs/repo_handbook.md:130`；ckpt 键面=2（path_head_ablation、path_head_mode）
- **allow_oneshot（守卫形参）**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/model.py:78`、`net/model.py:80`、`net/model.py:973`；eval=`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:50`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:441`、`.slim/deepwork/p1_v2/round36/a_hardening_check.py:19`；slim=`.slim/deepwork/p1_v2/round37/handbook_facts.md:44`；docs=`net/README.md:80`、`docs/deepseekv4p1_argue.md:285`、`docs/chatgpt5p6sol_review.md:415`
- **--diagnostic-oneshot（CLI 守卫）**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/model.py:80`、`net/model.py:983`、`net/model.py:992`；eval=`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:49`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:419`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:440`；slim=`.slim/deepwork/p1_v2/round37/handbook_facts.md:184`；docs=`docs/repo_handbook.md:197`、`docs/deepseekv4p1_argue.md:267`、`docs/deepseekv4p1_argue.md:285`
- **path_head_mode()（模式读出）**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/model.py:71`、`net/model.py:1012`、`net/model.py:1023`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:1201`、`.slim/deepwork/p1_v2/round36/j1_train.py:1456`；eval=`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:419`、`.slim/deepwork/p1_v2/round36/j1_eval.py:663`、`.slim/deepwork/p1_v2/round36/j1_eval.py:764`；slim=`.slim/deepwork/p1_v2/round36/k1_manifest.json:8`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.diff.txt:28`、`.slim/deepwork/p1_v2/round38/c1_membank.py:814`；docs=`docs/repo_handbook.md:31`、`docs/repo_handbook.md:142`、`docs/architecture_contract.md:21`
- **plan_override（钉住尾段）**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/model.py:85`、`net/model.py:979`、`net/model.py:1029`；slim=`.slim/deepwork/p1_v2/round36/k1_manifest.json:11`；docs=`docs/deepseekv4p1_argue.md:258`、`docs/deepseekv4p1_argue.md:281`、`docs/deepseekv4p1_argue.md:285`
- **one_shot 形参**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/model.py:1436`、`net/path_head.py:37`、`net/path_head.py:1169`；slim=`.slim/deepwork/p1_v2/round36/k1_manifest.json:11`
- **one_shot_plan() 方法**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:1028`、`net/path_head.py:1053`、`net/path_head.py:1289`；docs=`docs/chatgpt5p6sol_review.md:413`
- **compose_path_plan() 函数**（KEEP·REACHABLE）：prod=`net/__init__.py:22`、`net/__init__.py:84`、`net/path_head.py:130`；eval=`.slim/deepwork/p1_v2/round36/h1_compat_check.py:427`、`.slim/deepwork/p1_v2/round36/h1_compat_check.py:490`、`.slim/deepwork/p1_v2/round36/h1_compat_check.py:496`；slim=`.slim/deepwork/p1_v2/round36/journal.md:12`、`.slim/deepwork/p1_v2/round37/handbook_facts.md:23`；docs=`net/README.md:16`、`net/README.md:83`、`docs/repo_handbook.md:58`
- **PathExecutionHead 别名**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/__init__.py:20`、`net/__init__.py:74`、`net/path_head.py:8`；eval=`.slim/deepwork/p1_v2/round36/h1_compat_check.py:425`、`.slim/deepwork/p1_v2/round36/h1_compat_check.py:459`、`.slim/deepwork/p1_v2/round36/h1_skeleton_smoke.py:109`；slim=`.slim/deepwork/p1_v2/round36/k1_manifest.json:7`、`.slim/deepwork/p1_v2/round36/journal.md:12`、`.slim/deepwork/p1_v2/round36/h3_hooks_ext.py:684`；runs=`runs/round36/h3_hooks.json:1011`、`runs/round36/h3_hooks.json:2930`、`runs/round36/h3_hooks.json:3304`；docs=`net/README.md:16`、`docs/repo_handbook.md:58`、`docs/deepseekv4p1_argue.md:14`

## B 组｜K-anchor 旧计划头

| 符号 | kind | prod | train | eval | test | slim | runs | docs | 行数合计 | string | config | ckpt | 建议判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `num_anchors（构造/属性）` | ident | 31 | 10 | 5 | 5 | 37 | 147 | 26 | 261 | 112 | 60 | · | LEGACY_REPRO |
| `plan_anchors()（选择头前向）` | ident | 2 | · | · | · | · | · | · | 2 | · | · | · | LEGACY_REPRO |
| `plan_anchor（配置块）` | config | 2 | 7 | 7 | 4 | 24 | 84 | 36 | 164 | 16 | 74 | · | LEGACY_REPRO |
| `plan_anchors_k6.json（锚字典文件）` | config | 3 | 3 | · | 1 | 15 | 45 | 28 | 95 | 5 | 2 | · | LEGACY_REPRO |
| `set_anchors()/load_anchor_dictionary()` | ident | 5 | · | · | · | · | · | 1 | 6 | · | · | · | LEGACY_REPRO |
| `anchor_head/embed/residual/speed 子模块` | ident | 14 | 2 | 6 | 5 | · | 96 | 12 | 135 | 54 | · | 4 | CHECKPOINT_ONLY |
| `anchor 输出键（anchor_logits/probs/plan/…）` | ident | 23 | 9 | · | 26 | 4 | 6 | 10 | 78 | 11 | · | · | LEGACY_REPRO |
| `拟合工具 fit_plan_anchors.py` | ident | 3 | 1 | · | · | 5 | 12 | 20 | 41 | · | 2 | · | LEGACY_REPRO |
| `锚损失配置键（anchor_ce/anchor_wta/temperature/hard）` | config | 4 | 41 | · | 6 | 1 | 120 | 19 | 191 | 73 | 6 | · | LEGACY_REPRO |
| `plan_head.anchor* ckpt 键面` | ckpt | · | 2 | 6 | 2 | · | 96 | · | 106 | · | · | · | CHECKPOINT_ONLY |
| `net/anchor.py 模块（K-anchor 工具集，371 行）` | ident | 4 | 4 | · | 1 | · | · | 3 | 12 | · | 1 | · | LEGACY_REPRO |
| `anchor 纯函数（mixture/assign/lane 变换）` | ident | 24 | · | · | 21 | 1 | · | 2 | 48 | 1 | · | · | LEGACY_REPRO |

> K-anchor 长期关闭（config/model.yaml plan_anchor.enabled=false）且设计上产生独立计划候选；旧 ckpt 键面经 key mapper 忽略。`num_anchors=0` 时键集与旧模型逐位一致（net/plan_head.py:119）。

### B 组逐符号证据（每项 1 行：入口侧命中 → 其它命中）

- **num_anchors（构造/属性）**（LEGACY_REPRO·REACHABLE）：prod=`net/plan_head.py:54`、`net/plan_head.py:82`、`net/plan_head.py:83`；train=`tools/dagger_collect.py:1292`、`pipeline/stages.py:175`、`pipeline/stages.py:177`；eval=`.slim/deepwork/s1_audit/smoke_load.py:8`、`.slim/deepwork/s1_audit/smoke_load.py:97`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:644`；slim=`.slim/deepwork/s1_audit/README.md:52`、`.slim/deepwork/s1_audit/README.md:82`、`.slim/deepwork/v8_run/model_sw_pri128.yaml:56`；runs=`runs/BTC20261009-0824_sw_pri512/model.snapshot.yaml:56`、`runs/round36/c_smoke/a3_struct/manifest.json:51`、`runs/round36/c_smoke/a3_struct/summary.json:52`；test=`tests/test_recovery_recipe.py:55`、`tests/test_recovery_recipe.py:69`、`tests/test_plan_anchor.py:274`；docs=`net/README.md:51`、`net/README.md:73`、`net/README.md:102`
- **plan_anchors()（选择头前向）**（LEGACY_REPRO·REACHABLE）：prod=`net/plan_head.py:161`、`net/model.py:1416`
- **plan_anchor（配置块）**（LEGACY_REPRO·REACHABLE）：prod=`config/train.yaml:192`、`config/model.yaml:70`；train=`tools/dagger_collect.py:1294`、`pipeline/stages.py:174`、`pipeline/stages.py:176`；eval=`.slim/deepwork/s1_audit/smoke_load.py:8`、`.slim/deepwork/s1_audit/smoke_load.py:10`、`.slim/deepwork/s1_audit/smoke_load.py:78`；slim=`.slim/deepwork/v7-beat-idm.md:146`、`.slim/deepwork/v8_backlog.md:81`、`.slim/deepwork/s1_audit/run_s11.sh:15`；runs=`runs/BTC20261009-0824_sw_pri512/model.snapshot.yaml:54`、`runs/BTC20261005-134825_v7sb_p3_ckpt_epoch005_clean500/config.snapshot.yaml:2`、`runs/BTC20261005-134825_v7sb_p3_ckpt_epoch005_clean500/config.snapshot.yaml:6`；test=`tests/test_recovery_recipe.py:55`、`tests/test_phase3_safe_recipe.py:52`、`tests/test_eval_model_config_forwarding.py:4`；docs=`net/README.md:102`、`docs/repo_handbook.md:133`、`docs/repo_handbook.md:152`
- **plan_anchors_k6.json（锚字典文件）**（LEGACY_REPRO·REACHABLE）：prod=`net/anchor.py:34`、`net/anchor.py:37`、`config/model.yaml:73`；train=`tools/fit_plan_anchors.py:16`、`tools/fit_plan_anchors.py:23`、`tools/fit_plan_anchors.py:93`；slim=`.slim/deepwork/v8_run/model_sw_pri128.yaml:57`、`.slim/deepwork/v8_run/model_sw_attn2.yaml:58`、`.slim/deepwork/v8_run/model_arm_t.yaml:71`；runs=`runs/BTC20261009-0824_sw_pri512/model.snapshot.yaml:57`、`runs/BTC20261007-1838_v8chain/model.snapshot.yaml:71`、`runs/BTC20261007-2202_arm_t/model.snapshot.yaml:71`；test=`tests/test_plan_anchor.py:50`；docs=`README.md:66`、`tools/README.md:36`、`tools/README.md:70`
- **set_anchors()/load_anchor_dictionary()**（LEGACY_REPRO·REACHABLE）：prod=`net/plan_head.py:118`、`net/plan_head.py:131`、`net/model.py:190`；docs=`docs/v7_reports/v7_struct_b_kanchor.md:79`
- **anchor_head/embed/residual/speed 子模块**（CHECKPOINT_ONLY·REACHABLE）：prod=`net/plan_head.py:91`、`net/plan_head.py:96`、`net/plan_head.py:97`；train=`pipeline/stages.py:2765`、`pipeline/stages.py:2768`；eval=`tools/controls/probe_freeze_audit.py:79`、`tools/controls/probe_freeze_audit.py:82`、`tools/controls/probe_freeze_audit.py:127`；runs=`runs/round31/probes/q2_grad_conflict.md:62`、`runs/round31/probes/q2_grad_conflict_b256.json:1484`、`runs/round31/probes/q2_grad_conflict_b256.json:1487`；test=`tests/test_plan_anchor.py:316`、`tests/test_plan_anchor.py:318`、`tests/test_plan_anchor.py:332`；docs=`docs/v7_program_prereg.md:334`、`docs/v7_program_prereg.md:335`、`docs/v7_reports/v7_struct_fail_diag.md:74`；ckpt 键面=4（plan_head.anchor_embed、plan_head.anchor_head、plan_head.residual_head、plan_head.speed_head）
- **anchor 输出键（anchor_logits/probs/plan/…）**（LEGACY_REPRO·REACHABLE）：prod=`net/model.py:61`、`net/model.py:108`、`net/model.py:120`；train=`pipeline/trainer.py:2596`、`pipeline/trainer.py:2597`、`pipeline/trainer.py:2613`；slim=`.slim/deepwork/p1_v2/round35/f2_hooks.py:705`、`.slim/deepwork/p1_v2/round35/f2_hooks.py:706`、`.slim/deepwork/p1_v2/round35/f2_hooks.py:765`；runs=`runs/round35/f2_hooks.json:1217`、`runs/round35/f2_hooks.json:1221`、`runs/round35/f2_hooks.json:3673`；test=`tests/test_plan_anchor.py:69`、`tests/test_plan_anchor.py:70`、`tests/test_plan_anchor.py:71`；docs=`net/README.md:51`、`docs/v7_program_prereg.md:283`、`docs/v7_program_prereg.md:303`
- **拟合工具 fit_plan_anchors.py**（LEGACY_REPRO·REACHABLE）：prod=`net/anchor.py:6`、`net/anchor.py:39`、`config/model.yaml:73`；train=`tools/fit_plan_anchors.py:21`；slim=`.slim/deepwork/v8_run/model_arm_t.yaml:71`、`.slim/deepwork/v8_run/model_arm_p.yaml:71`、`.slim/deepwork/v8_run/model_new.yaml:71`；runs=`runs/BTC20261007-1838_v8chain/model.snapshot.yaml:71`、`runs/BTC20261007-2202_arm_t/model.snapshot.yaml:71`、`runs/BTC20261005-0856_v7struct_v5/model.snapshot.yaml:66`；docs=`tools/README.md:36`、`tools/README.md:70`、`docs/deepseekv4p1_argue.md:405`
- **锚损失配置键（anchor_ce/anchor_wta/temperature/hard）**（LEGACY_REPRO·REACHABLE）：prod=`config/train.yaml:134`、`config/train.yaml:135`、`config/train.yaml:136`；train=`pipeline/stages.py:194`、`pipeline/stages.py:195`、`pipeline/stages.py:3066`；slim=`.slim/deepwork/p1_v2/round37/handbook_facts.md:103`；runs=`runs/BTC20261005-0856_v7struct_v5/config.snapshot.yaml:69`、`runs/BTC20261005-0856_v7struct_v5/config.snapshot.yaml:70`、`runs/BTC20261005-0856_v7struct_v5/config.snapshot.yaml:71`；test=`tests/test_recovery_recipe.py:388`、`tests/test_recovery_recipe.py:409`、`tests/test_plan_anchor.py:396`；docs=`docs/repo_handbook.md:173`、`docs/v7_program_prereg.md:290`、`docs/v7_program_prereg.md:331`
- **plan_head.anchor* ckpt 键面**（CHECKPOINT_ONLY·REACHABLE）：train=`pipeline/stages.py:2765`、`pipeline/stages.py:2768`；eval=`tools/controls/probe_freeze_audit.py:79`、`tools/controls/probe_freeze_audit.py:82`、`tools/controls/probe_freeze_audit.py:127`；runs=`runs/round31/probes/q2_grad_conflict.md:62`、`runs/round31/probes/q2_grad_conflict_b256.json:1484`、`runs/round31/probes/q2_grad_conflict_b256.json:1487`；test=`tests/test_phase3_safe_recipe.py:121`、`tests/test_phase3_safe_recipe.py:124`
- **net/anchor.py 模块（K-anchor 工具集，371 行）**（LEGACY_REPRO·REACHABLE）：prod=`net/plan_head.py:25`、`net/model.py:48`、`net/model.py:117`；train=`tools/fit_plan_anchors.py:17`、`tools/fit_plan_anchors.py:77`、`pipeline/trainer.py:2601`；test=`tests/test_plan_anchor.py:23`；docs=`config/README.md:26`、`config/README.md:51`、`config/README.md:75`
- **anchor 纯函数（mixture/assign/lane 变换）**（LEGACY_REPRO·REACHABLE）：prod=`net/anchor.py:146`、`net/anchor.py:172`、`net/anchor.py:182`；slim=`.slim/deepwork/p1_v2/round37/handbook_facts.md:21`；test=`tests/test_plan_anchor.py:26`、`tests/test_plan_anchor.py:27`、`tests/test_plan_anchor.py:29`；docs=`docs/repo_handbook.md:56`、`docs/archive/v8_reports/net_architecture.md:289`

## C 组｜遗留 set_ego_warp（w1/w2）实验接口

| 符号 | kind | prod | train | eval | test | slim | runs | docs | 行数合计 | string | config | ckpt | 建议判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `set_ego_warp()（遗留实验）` | ident | 2 | · | 10 | · | 4 | · | 3 | 19 | 8 | · | · | LEGACY_REPRO |
| `set_ego_warp_mode()（M1 主路径）` | ident | 4 | · | 12 | · | 4 | · | 3 | 23 | 6 | · | · | KEEP |
| `w1/w2 模式串` | string | 7 | · | 14 | · | 39 | 24 | 3 | 87 | 87 | · | · | LEGACY_REPRO |
| `_ego_warp_pose_k / ego_warp 属性` | ident | 10 | · | 3 | · | 6 | 1 | 3 | 23 | · | · | · | LEGACY_REPRO |
| `assemble_pose_k()（主路径解析件）` | ident | 16 | · | 20 | · | 17 | · | 13 | 66 | 3 | · | · | KEEP |
| `od_velocity_t0()` | ident | 16 | · | 4 | · | 11 | · | 12 | 43 | 4 | · | · | KEEP |
| `WARP_MODES（模式登记）` | ident | 5 | · | · | · | 2 | · | 1 | 8 | · | · | · | KEEP |

> §5.69-① 修正：legacy w1/w2 是 analytic warp 的**逐位 oracle**（`.slim/round37/m1_gate_a.py`、`.slim/round33/b_warp_compat.py`）⇒ 先冻结 fixture 再删。

### C 组逐符号证据（每项 1 行：入口侧命中 → 其它命中）

- **set_ego_warp()（遗留实验）**（LEGACY_REPRO·REACHABLE）：prod=`net/model.py:622`、`net/model.py:681`；eval=`.slim/deepwork/p1_v2/round37/m1_gate_a.py:137`、`.slim/deepwork/p1_v2/round37/m1_gate_a.py:139`、`.slim/deepwork/p1_v2/round37/m1_gate_a.py:141`；slim=`.slim/deepwork/p1_v2/round36/d1_ego_warp_tests.py:172`、`.slim/deepwork/p1_v2/round37/handbook_facts.md:40`、`.slim/deepwork/p1_v2/round33/b_warp_record.py:9`；docs=`docs/chatgpt5p6sol_review.md:448`、`docs/chatgpt5p6sol_review.md:452`、`docs/chatgpt5p6sol_review.md:460`
- **set_ego_warp_mode()（M1 主路径）**（KEEP·REACHABLE）：prod=`net/ego_warp.py:6`、`net/model.py:90`、`net/model.py:654`；eval=`.slim/deepwork/p1_v2/round37/m4_paired.py:618`、`.slim/deepwork/p1_v2/round37/m1_gate_a.py:323`、`.slim/deepwork/p1_v2/round37/m1_gate_a.py:427`；slim=`.slim/deepwork/p1_v2/round37/m4_report.md:106`、`.slim/deepwork/p1_v2/round37/m1_report.md:11`、`.slim/deepwork/p1_v2/round37/m1_report.md:120`；docs=`net/README.md:61`、`docs/chatgpt5p6sol_review.md:453`、`docs/chatgpt5p6sol_review.md:460`
- **w1/w2 模式串**（LEGACY_REPRO·REACHABLE）：prod=`net/ego_warp.py:93`、`net/ego_warp.py:218`、`net/model.py:649`；eval=`.slim/deepwork/p1_v2/round37/m1_gate_a.py:137`、`.slim/deepwork/p1_v2/round37/m1_gate_a.py:138`、`.slim/deepwork/p1_v2/round37/m1_gate_a.py:139`；slim=`.slim/deepwork/p1_v2/round36/d1_ego_warp_evidence.md:5`、`.slim/deepwork/p1_v2/round36/d3_ld_report.md:14`、`.slim/deepwork/p1_v2/round36/d1_ego_warp_evidence.json:33`；runs=`runs/round33/b_warp.json:27`、`runs/round33/b_warp.json:28`、`runs/round33/b_warp.json:33`；docs=`docs/repo_handbook.md:60`、`docs/v7_reports/v7_p1_dagger_cycle.md:2501`、`docs/v7_reports/v7_p1_dagger_cycle.md:2507`
- **_ego_warp_pose_k / ego_warp 属性**（LEGACY_REPRO·REACHABLE）：prod=`net/ego_warp.py:4`、`net/ego_warp.py:117`、`net/ego_warp.py:123`；eval=`.slim/deepwork/p1_v2/round33/b_warp_compat.py:14`、`.slim/deepwork/p1_v2/round33/b_warp_compat.py:229`、`.slim/deepwork/p1_v2/round33/b_warp_compat.py:270`；slim=`.slim/deepwork/p1_v2/round36/d3_ld_report.md:11`、`.slim/deepwork/p1_v2/round36/d3_ld_report.md:27`、`.slim/deepwork/p1_v2/round36/d1_ego_warp_tests.py:5`；runs=`runs/round34/d3_hooks_note.md:30`；docs=`docs/deepseekv4p1_argue.md:281`、`docs/deepseekv4p1_argue.md:293`、`docs/r36_warp_layer_note.md:9`
- **assemble_pose_k()（主路径解析件）**（KEEP·REACHABLE）：prod=`net/__init__.py:11`、`net/__init__.py:61`、`net/ego_warp.py:6`；eval=`.slim/deepwork/p1_v2/round38/c2_matrix.py:393`、`.slim/deepwork/p1_v2/round38/c2_matrix.py:528`、`.slim/deepwork/p1_v2/round38/a3_matrix.py:45`；slim=`.slim/deepwork/p1_v2/round36/journal.md:60`、`.slim/deepwork/p1_v2/round36/d3_ld_report.md:14`、`.slim/deepwork/p1_v2/round36/d3_ld_report.md:28`；docs=`net/README.md:20`、`net/README.md:61`、`docs/repo_handbook.md:37`
- **od_velocity_t0()**（KEEP·REACHABLE）：prod=`net/__init__.py:13`、`net/__init__.py:67`、`net/ego_warp.py:6`；eval=`.slim/deepwork/p1_v2/round37/m1_gate_a.py:12`、`.slim/deepwork/p1_v2/round37/m1_gate_a.py:319`、`.slim/deepwork/p1_v2/round37/m1_gate_a.py:329`；slim=`.slim/deepwork/p1_v2/round36/d3_ld_report.md:13`、`.slim/deepwork/p1_v2/round36/d3_ld_report.md:92`、`.slim/deepwork/p1_v2/round36/d1_ego_warp_tests.py:42`；docs=`net/README.md:20`、`net/README.md:61`、`docs/repo_handbook.md:60`
- **WARP_MODES（模式登记）**（KEEP·REACHABLE）：prod=`net/__init__.py:8`、`net/__init__.py:58`、`net/ego_warp.py:76`；slim=`.slim/deepwork/p1_v2/round37/m1_report.md:39`、`.slim/deepwork/p1_v2/round37/handbook_facts.md:25`；docs=`docs/repo_handbook.md:60`

## D 组｜Policy/path_head 双实体（收编）

| 符号 | kind | prod | train | eval | test | slim | runs | docs | 行数合计 | string | config | ckpt | 建议判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `enable_path_head（开关镜像属性）` | ident | 4 | · | 1 | · | 2 | · | 4 | 11 | 2 | · | · | LEGACY_REPRO |
| `set_path_head()/path_head 属性` | ident | 4 | 2 | 27 | · | 17 | 73 | 8 | 131 | 81 | · | 20 | KEEP |
| `path_head_ablation 模式` | ident | 7 | · | 29 | · | 3 | · | 2 | 41 | 20 | · | 1 | ARCHITECTURE_FORBIDDEN |
| `set_path_head_ablation()` | ident | 7 | 4 | 24 | · | 9 | 66 | 4 | 114 | 75 | · | · | ARCHITECTURE_FORBIDDEN |
| `set_policy_diag()/policy_diag_enabled` | ident | 10 | · | 6 | · | 1 | · | 4 | 21 | · | · | · | LEGACY_REPRO |
| `path_head_trace / set_path_head_trace` | ident | 15 | · | 7 | · | 2 | · | 4 | 28 | · | · | · | KEEP |
| `action_generator_names()/action_source` | ident | 2 | 2 | 11 | · | 6 | 7 | 3 | 31 | 13 | · | · | KEEP |
| `self.policy（旧 PolicyHead 实体）` | ident | 8 | 2 | 5 | 11 | · | · | 1 | 27 | · | · | 2 | KEEP |
| `policy_action_mu（旧头输出键）` | ident | 11 | · | 15 | · | 2 | 5 | 4 | 37 | 15 | · | · | KEEP |
| `contract_asserts / contract_ 输出键` | ident | 11 | · | 3 | · | 1 | · | 3 | 18 | · | · | · | KEEP |

> §5.69-③ 修正：A/D 组迁移必须同步 `.slim` 22 个工具与 L-A 契约断言（否则 Gate 证据失效）。

### D 组逐符号证据（每项 1 行：入口侧命中 → 其它命中）

- **enable_path_head（开关镜像属性）**（LEGACY_REPRO·REACHABLE）：prod=`net/model.py:471`、`net/model.py:962`、`net/model.py:1041`；eval=`.slim/deepwork/p1_v2/a3/eval_a3.py:237`；slim=`.slim/deepwork/p1_v2/round37/m3_diag.py:700`、`.slim/deepwork/p1_v2/round37/m3_diag.py:704`；docs=`docs/deepseekv4p1_argue.md:413`、`docs/chatgpt5p6sol_review.md:468`、`docs/chatgpt5p6sol_review.md:483`
- **set_path_head()/path_head 属性**（KEEP·REACHABLE）：prod=`net/model.py:52`、`net/model.py:465`、`net/model.py:943`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:800`、`.slim/deepwork/p1_v2/round36/j1_train.py:842`；eval=`.slim/deepwork/p1_v2/round36/c_report.py:81`、`.slim/deepwork/p1_v2/round36/c_report.py:158`、`.slim/deepwork/p1_v2/round36/j1_eval.py:561`；slim=`.slim/deepwork/p1_v2/round36/k1_manifest.json:8`、`.slim/deepwork/p1_v2/round36/journal.md:12`、`.slim/deepwork/p1_v2/round36/journal.md:65`；runs=`runs/round36/h3_hooks.json:428`、`runs/round36/h3_hooks.json:1011`、`runs/round36/h3_hooks.json:10094`；docs=`net/README.md:16`、`net/README.md:22`、`net/README.md:79`；ckpt 键面=20（path_head.lateral、path_head.lateral.0.bias、path_head.lateral.0.weight、path_head.lateral.2.bias…）
- **path_head_ablation 模式**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/model.py:473`、`net/model.py:990`、`net/model.py:994`；eval=`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:47`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:398`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:400`；slim=`.slim/deepwork/p1_v2/round36/c_old_p1_deploy_record.diff.txt:148`、`.slim/deepwork/p1_v2/round36/c_old_eval_a3.diff.txt:81`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.diff.txt:34`；docs=`docs/chatgpt5p6sol_review.md:469`、`docs/chatgpt5p6sol_review.md:483`；ckpt 键面=1（path_head_ablation）
- **set_path_head_ablation()**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/model.py:62`、`net/model.py:78`、`net/model.py:964`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:12`、`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:1203`、`.slim/deepwork/p1_v2/round36/j1_train.py:11`；eval=`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:50`、`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:441`、`.slim/deepwork/p1_v2/round36/c_report.py:81`；slim=`.slim/deepwork/p1_v2/round36/k1_manifest.json:8`、`.slim/deepwork/p1_v2/round36/k1_manifest.json:11`、`.slim/deepwork/p1_v2/round36/journal.md:65`；runs=`runs/round36/j1_eval/gate_t1_res_ep3.md:38`、`runs/round36/j1_eval/gate_t2_res4.md:75`、`runs/round36/j1_eval/gate_t1_res_ep1.md:38`；docs=`docs/repo_handbook.md:130`、`docs/architecture_contract.md:6`、`docs/architecture_contract.md:21`
- **set_policy_diag()/policy_diag_enabled**（LEGACY_REPRO·REACHABLE）：prod=`net/model.py:74`、`net/model.py:475`、`net/model.py:479`；eval=`.slim/deepwork/p1_v2/round36/c_old_canary_eval.py:51`、`.slim/deepwork/p1_v2/round36/a_hardening_check.py:14`、`.slim/deepwork/p1_v2/round36/a_hardening_check.py:395`；slim=`.slim/deepwork/p1_v2/round37/handbook_facts.md:45`；docs=`net/README.md:53`、`net/README.md:85`、`docs/deepseekv4p1_argue.md:285`
- **path_head_trace / set_path_head_trace**（KEEP·REACHABLE）：prod=`net/model.py:75`、`net/model.py:476`、`net/model.py:997`；eval=`.slim/deepwork/p1_v2/round36/k1_smoke.py:128`、`.slim/deepwork/p1_v2/round36/k1_smoke.py:142`、`.slim/deepwork/p1_v2/round36/a_hardening_check.py:14`；slim=`.slim/deepwork/p1_v2/round36/k1_manifest.json:8`、`.slim/deepwork/p1_v2/round37/handbook_facts.md:45`；docs=`net/README.md:53`、`net/README.md:85`、`docs/deepseekv4p1_argue.md:285`
- **action_generator_names()/action_source**（KEEP·REACHABLE）：prod=`net/model.py:63`、`net/model.py:1045`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:1204`、`.slim/deepwork/p1_v2/round36/j1_train.py:1459`；eval=`.slim/deepwork/p1_v2/round36/j1_eval.py:765`、`.slim/deepwork/p1_v2/round36/c_selfproof.py:178`、`.slim/deepwork/p1_v2/round36/k1_smoke.py:18`；slim=`.slim/deepwork/p1_v2/round36/k1_manifest.json:8`、`.slim/deepwork/p1_v2/round36/k1_manifest.json:29`、`.slim/deepwork/p1_v2/round37/m1_k1_contract_gate.md:9`；runs=`runs/round36/a/k1_gate_rerun.md:9`、`runs/round36/a/k1_gate_rerun.md:19`、`runs/round36/a/k1_gate_rerun.json:119`；docs=`docs/r36_implementation_evidence.md:19`、`docs/architecture_contract.md:21`、`docs/architecture_contract.md:131`
- **self.policy（旧 PolicyHead 实体）**（KEEP·REACHABLE）：prod=`net/model.py:524`、`net/model.py:1932`、`net/model.py:2042`；train=`tools/dagger_collect.py:181`、`tools/dagger_collect.py:190`；eval=`pipeline/eval_runner.py:618`、`pipeline/eval_runner.py:628`、`pipeline/eval_runner.py:635`；test=`tests/test_probe_train_account.py:41`、`tests/test_probe_freeze_audit.py:48`、`tests/test_anchor_grad_probe.py:64`；docs=`docs/deepseekv4p1_argue.md:267`；ckpt 键面=2（policy.mu.bias、policy.mu.weight）
- **policy_action_mu（旧头输出键）**（KEEP·REACHABLE）：prod=`net/model.py:73`、`net/model.py:75`、`net/model.py:478`；eval=`.slim/deepwork/p1_v2/round36/j1_eval.py:115`、`.slim/deepwork/p1_v2/round36/h1_compat_check.py:14`、`.slim/deepwork/p1_v2/round36/h1_compat_check.py:344`；slim=`.slim/deepwork/p1_v2/round36/journal.md:12`、`.slim/deepwork/p1_v2/round37/handbook_facts.md:47`；runs=`runs/round36/a/hardening.md:9`、`runs/round36/a/hardening.md:10`、`runs/round36/a/hardening.json:62`；docs=`net/README.md:53`、`net/README.md:85`、`docs/repo_handbook.md:142`
- **contract_asserts / contract_ 输出键**（KEEP·REACHABLE）：prod=`net/model.py:82`、`net/model.py:483`、`net/model.py:1020`；eval=`.slim/deepwork/p1_v2/round36/a_hardening_check.py:23`、`.slim/deepwork/p1_v2/round36/a_hardening_check.py:606`、`.slim/deepwork/p1_v2/round36/a_hardening_check.py:609`；slim=`.slim/deepwork/p1_v2/round37/handbook_facts.md:45`；docs=`net/README.md:85`、`docs/repo_handbook.md:131`、`docs/architecture_contract.md:21`

## E 组｜path_* 旁路字段（Q4 分类输入）

| 符号 | kind | prod | train | eval | test | slim | runs | docs | 行数合计 | string | config | ckpt | 建议判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `path_left_margin` | config | 2 | 2 | 17 | · | 7 | 31 | 3 | 62 | 55 | · | · | ARCHITECTURE_FORBIDDEN |
| `path_right_margin` | config | 2 | 2 | 16 | · | 7 | 31 | 3 | 61 | 54 | · | · | ARCHITECTURE_FORBIDDEN |
| `path_static_distance` | config | 2 | · | 34 | · | 31 | 68 | 4 | 139 | 120 | · | · | ARCHITECTURE_FORBIDDEN |
| `path_dynamic_distance` | config | 2 | · | 31 | · | 31 | 68 | 4 | 136 | 117 | · | · | ARCHITECTURE_FORBIDDEN |
| `path_dynamic_ttc` | config | 2 | · | 35 | · | 33 | 68 | 15 | 153 | 121 | · | · | ARCHITECTURE_FORBIDDEN |
| `path_nav_progress` | config | 2 | · | 13 | · | 6 | 31 | 3 | 55 | 48 | · | · | ARCHITECTURE_FORBIDDEN |
| `path_distance_to_target_lane_end` | config | 2 | · | 8 | · | 4 | 31 | 3 | 48 | 41 | · | · | ARCHITECTURE_FORBIDDEN |
| `LONG_EXOG_OBS_KEYS / long_exog_from_obs（消费口）` | ident | 9 | 4 | 8 | · | 13 | 2 | 3 | 39 | · | · | · | ARCHITECTURE_FORBIDDEN |
| `_vector_from_obs（缺键静默 0 回退）` | ident | 7 | · | 1 | · | · | · | · | 8 | · | · | · | ARCHITECTURE_FORBIDDEN |

> §5.69：prod 侧爆炸半径极小（仅 net/path_head.py 1 文件 14 处）；成本在头输入 schema + 重训 + `.slim` 注入口（m6_calib/m7_fill/j1_train/j2_train_stripped_check）。训练数据由 `.slim/round36/i1_extract_dataset.py` 离线落盘（datasets/path_exec_v1），eval env 不产这三键。

### E 组逐符号证据（每项 1 行：入口侧命中 → 其它命中）

- **path_left_margin**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:54`、`net/path_head.py:238`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:757`、`.slim/deepwork/p1_v2/round36/j1_train.py:799`；eval=`.slim/deepwork/p1_v2/round36/j1_eval.py:498`、`.slim/deepwork/p1_v2/round36/j1_eval.py:499`、`.slim/deepwork/p1_v2/round36/j1_eval.py:500`；slim=`.slim/deepwork/p1_v2/round36/h1_manifest.json:17`、`.slim/deepwork/p1_v2/round38/d1_rollout_dump.py:77`、`.slim/deepwork/p1_v2/round38/d1_probe_demo.py:56`；runs=`runs/round38/c3/c3.json:105`、`runs/round38/c3/c3.json:368`、`runs/round38/c3/c3_eval_probe.json:113`；docs=`docs/repo_handbook.md:89`、`docs/chatgpt5p6sol_review.md:112`、`docs/chatgpt5p6sol_review.md:489`
- **path_right_margin**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:54`、`net/path_head.py:239`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:757`、`.slim/deepwork/p1_v2/round36/j1_train.py:799`；eval=`.slim/deepwork/p1_v2/round36/j1_eval.py:498`、`.slim/deepwork/p1_v2/round36/j1_eval.py:499`、`.slim/deepwork/p1_v2/round36/j1_eval.py:500`；slim=`.slim/deepwork/p1_v2/round36/h1_manifest.json:17`、`.slim/deepwork/p1_v2/round38/d1_rollout_dump.py:78`、`.slim/deepwork/p1_v2/round38/d1_probe_demo.py:57`；runs=`runs/round38/c3/c3.json:106`、`runs/round38/c3/c3.json:2578`、`runs/round38/c3/c3_eval_probe.json:114`；docs=`docs/repo_handbook.md:89`、`docs/chatgpt5p6sol_review.md:113`、`docs/chatgpt5p6sol_review.md:490`
- **path_static_distance**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:54`、`net/path_head.py:240`；eval=`.slim/deepwork/p1_v2/round36/j1_eval.py:396`、`.slim/deepwork/p1_v2/round36/j1_eval.py:397`、`.slim/deepwork/p1_v2/round36/h1_compat_check.py:329`；slim=`.slim/deepwork/p1_v2/round36/h1_manifest.json:18`、`.slim/deepwork/p1_v2/round38/d1_rollout_dump.py:79`、`.slim/deepwork/p1_v2/round38/d1_probe_demo.py:58`；runs=`runs/round38/a4/c1_fill_gfix/manifest.json:20`、`runs/round38/a4/c1_fill_gfix/manifest.json:80`、`runs/round38/a4/c1_fill_gfix/manifest.json:239`；docs=`docs/repo_handbook.md:89`、`docs/repo_handbook.md:311`、`docs/chatgpt5p6sol_review.md:114`
- **path_dynamic_distance**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:54`、`net/path_head.py:241`；eval=`.slim/deepwork/p1_v2/round36/h1_compat_check.py:330`、`.slim/deepwork/p1_v2/round36/k1_smoke.py:116`、`.slim/deepwork/p1_v2/round36/a_hardening_check.py:138`；slim=`.slim/deepwork/p1_v2/round36/h1_manifest.json:18`、`.slim/deepwork/p1_v2/round38/d1_rollout_dump.py:80`、`.slim/deepwork/p1_v2/round38/d1_probe_demo.py:59`；runs=`runs/round38/a4/c1_fill_gfix/manifest.json:21`、`runs/round38/a4/c1_fill_gfix/manifest.json:81`、`runs/round38/a4/c1_fill_gfix/manifest.json:240`；docs=`docs/repo_handbook.md:89`、`docs/repo_handbook.md:311`、`docs/chatgpt5p6sol_review.md:115`
- **path_dynamic_ttc**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:55`、`net/path_head.py:242`；eval=`.slim/deepwork/p1_v2/round36/j1_eval.py:402`、`.slim/deepwork/p1_v2/round36/j1_eval.py:403`、`.slim/deepwork/p1_v2/round36/h1_compat_check.py:331`；slim=`.slim/deepwork/p1_v2/round36/h1_manifest.json:18`、`.slim/deepwork/p1_v2/round38/d1_rollout_dump.py:81`、`.slim/deepwork/p1_v2/round38/d1_probe_demo.py:60`；runs=`runs/round38/a4/c1_fill_gfix/manifest.json:22`、`runs/round38/a4/c1_fill_gfix/manifest.json:82`、`runs/round38/a4/c1_fill_gfix/manifest.json:241`；docs=`docs/repo_handbook.md:89`、`docs/repo_handbook.md:254`、`docs/repo_handbook.md:311`
- **path_nav_progress**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:55`、`net/path_head.py:243`；eval=`.slim/deepwork/p1_v2/round36/h1_compat_check.py:332`、`.slim/deepwork/p1_v2/round36/k1_smoke.py:118`、`.slim/deepwork/p1_v2/round36/a_hardening_check.py:141`；slim=`.slim/deepwork/p1_v2/round36/h1_manifest.json:19`、`.slim/deepwork/p1_v2/round38/d1_rollout_dump.py:82`、`.slim/deepwork/p1_v2/round38/d1_probe_demo.py:61`；runs=`runs/round38/c3/c3.json:107`、`runs/round38/c3/c3.json:8490`、`runs/round38/c3/c3_eval_probe.json:118`；docs=`docs/repo_handbook.md:89`、`docs/chatgpt5p6sol_review.md:117`、`docs/chatgpt5p6sol_review.md:494`
- **path_distance_to_target_lane_end**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:55`、`net/path_head.py:244`；eval=`.slim/deepwork/p1_v2/round38/c3_eval_probe.py:37`、`.slim/deepwork/p1_v2/round38/c3_report.py:157`、`.slim/deepwork/p1_v2/round38/c3_report.py:201`；slim=`.slim/deepwork/p1_v2/round36/h1_manifest.json:20`、`.slim/deepwork/p1_v2/round38/c3_unit_check.py:32`、`.slim/deepwork/p1_v2/round37/handbook_facts.md:152`；runs=`runs/round38/c3/c3.json:108`、`runs/round38/c3/c3.json:9724`、`runs/round38/c3/c3_eval_probe.json:119`；docs=`docs/repo_handbook.md:89`、`docs/chatgpt5p6sol_review.md:118`、`docs/chatgpt5p6sol_review.md:495`
- **LONG_EXOG_OBS_KEYS / long_exog_from_obs（消费口）**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:115`、`net/path_head.py:137`、`net/path_head.py:236`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:694`、`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:748`、`.slim/deepwork/p1_v2/round36/j1_train.py:736`；eval=`.slim/deepwork/p1_v2/round38/c3_matrix.py:4`、`.slim/deepwork/p1_v2/round38/c3_matrix.py:10`、`.slim/deepwork/p1_v2/round38/c3_matrix.py:59`；slim=`.slim/deepwork/p1_v2/round38/c3_unit_check.py:4`、`.slim/deepwork/p1_v2/round38/c3_unit_check.py:9`、`.slim/deepwork/p1_v2/round38/c3_unit_check.py:24`；runs=`runs/round37/m3/m3_diag.json:35482`、`runs/round37/m3/m3_diag.json:35492`；docs=`docs/repo_handbook.md:58`、`docs/repo_handbook.md:89`、`docs/round37_report.md:100`
- **_vector_from_obs（缺键静默 0 回退）**（ARCHITECTURE_FORBIDDEN·REACHABLE）：prod=`net/path_head.py:894`、`net/path_head.py:950`、`net/path_head.py:954`；eval=`.slim/deepwork/p1_v2/a3/eval_a3.py:264`

## F 组｜外部 target-lane 强逻辑

| 符号 | kind | prod | train | eval | test | slim | runs | docs | 行数合计 | string | config | ckpt | 建议判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `select_target_lane()（env 侧纯函数）` | ident | 9 | 2 | 3 | 2 | 46 | 30 | 10 | 102 | 28 | · | · | KEEP |
| `TargetLaneSelection / source` | ident | 11 | · | · | · | 7 | · | 1 | 19 | · | · | · | KEEP |
| `target_lane_token（net 侧语义 token）` | ident | 14 | 7 | 6 | 9 | 33 | 128 | 7 | 204 | 153 | · | · | KEEP |
| `target_lane_valid` | ident | 28 | 2 | 9 | 6 | 4 | 4 | 6 | 59 | 32 | · | · | KEEP |
| `lane_semantic（12 维通道）` | ident | 56 | 48 | 43 | 33 | 166 | 225 | 37 | 608 | 360 | · | · | KEEP |
| `set_target_lane_dropout（训练期 dropout）` | ident | 10 | 6 | 13 | · | 12 | 32 | 2 | 75 | 21 | · | · | KEEP |
| `lane_anchor_xy（注入锚折线）` | ident | 10 | 4 | 44 | · | 40 | 136 | 6 | 240 | 205 | · | · | KEEP |
| `--lane-anchor inject（评测注入）` | string | · | · | 107 | · | 78 | 64 | 10 | 259 | 259 | · | · | LEGACY_REPRO |

> §7-F 裁决：不整体删除（'给定正确目标后的执行能力'仍用），改为来源标签 road_level / teacher_target_lane / deployment_available；生产部署对唯一 target lane 的依赖长期移除。

### F 组逐符号证据（每项 1 行：入口侧命中 → 其它命中）

- **select_target_lane()（env 侧纯函数）**（KEEP·REACHABLE）：prod=`env/obs/__init__.py:32`、`env/obs/lane_semantic.py:6`、`env/obs/lane_semantic.py:40`；train=`.slim/deepwork/p1_v2/round21/collect_semantic_v1.py:94`、`.slim/deepwork/p1_v2/round21/collect_semantic_v1.py:343`；eval=`.slim/deepwork/p1_v2/round36/c_report.py:119`、`.slim/deepwork/p1_v2/round36/c_report.py:171`、`.slim/deepwork/p1_v2/round37/m2_paired_c0c1.py:547`；slim=`.slim/deepwork/p1_v2/round36/journal.md:66`、`.slim/deepwork/p1_v2/round36/c_lane_anchor.py:14`、`.slim/deepwork/p1_v2/round36/c_lane_anchor.py:56`；runs=`runs/round36/c_smoke/g6_draft.json:39`、`runs/round36/c_smoke/report.md:20`、`runs/round36/c_smoke/report.md:107`；test=`tests/test_target_lane_continuity.py:25`、`tests/test_target_lane_continuity.py:58`；docs=`docs/r21_target_lane_selection.md:3`、`docs/deepseekv4p1_argue.md:287`、`docs/r21_lane_semantic_contract.md:58`
- **TargetLaneSelection / source**（KEEP·REACHABLE）：prod=`env/obs/lane_semantic.py:7`、`env/obs/lane_semantic.py:39`、`env/obs/lane_semantic.py:82`；slim=`.slim/deepwork/p1_v2/round36/h3_hooks_ext.py:1184`、`.slim/deepwork/p1_v2/round36/h3_hooks_ext.py:1187`、`.slim/deepwork/p1_v2/round35/g2_diag_target.py:261`；docs=`docs/r21_target_lane_selection.md:5`
- **target_lane_token（net 侧语义 token）**（KEEP·REACHABLE）：prod=`net/semantic.py:1`、`net/semantic.py:8`、`net/semantic.py:17`；train=`.slim/deepwork/p1_v2/round24/r24_train_s0s1.py:194`、`.slim/deepwork/p1_v2/round24/r24_train_s0s1.py:232`、`.slim/deepwork/p1_v2/round23/train_canary.py:195`；eval=`.slim/deepwork/p1_v2/round36/h1_skeleton_smoke.py:14`、`.slim/deepwork/p1_v2/round36/h1_skeleton_smoke.py:308`、`.slim/deepwork/p1_v2/round36/h1_skeleton_smoke.py:315`；slim=`.slim/deepwork/p1_0/journal.md:493`、`.slim/deepwork/p1_0/journal.md:494`、`.slim/deepwork/p1_v2/round36/h1_manifest.json:56`；runs=`runs/round36/h3_hooks.json:1737`、`runs/round36/h3_hooks.json:2099`、`runs/round36/h3_hooks.json:2628`；test=`tests/test_r21_3a_semantic_wiring.py:5`、`tests/test_r21_3a_semantic_wiring.py:100`、`tests/test_r21_3a_semantic_wiring.py:108`；docs=`net/README.md:19`、`docs/r21_3a_semantic_wiring.md:18`、`docs/r21_3a_semantic_wiring.md:45`
- **target_lane_valid**（KEEP·REACHABLE）：prod=`net/semantic.py:9`、`net/model.py:1427`、`net/model.py:1435`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:730`、`.slim/deepwork/p1_v2/round36/j1_train.py:772`；eval=`.slim/deepwork/p1_v2/round38/c2_matrix.py:356`、`.slim/deepwork/p1_v2/round38/c2_matrix.py:543`、`.slim/deepwork/p1_v2/round38/a3_matrix.py:625`；slim=`.slim/deepwork/p1_v2/round36/h1_manifest.json:16`、`.slim/deepwork/p1_v2/round36/h1_manifest.json:28`、`.slim/deepwork/p1_v2/round35/f3_data_census.py:573`；runs=`runs/round35/f3_data_census.md:263`、`runs/round35/f3_data_census.json:1513`、`runs/round37/m3/m3_diag.json:90`；test=`tests/test_lane_semantic_contract.py:6`、`tests/test_lane_semantic_contract.py:145`、`tests/test_lane_semantic_contract.py:158`；docs=`docs/r21_target_lane_selection.md:50`、`docs/r21_lane_semantic_contract.md:20`、`docs/r21_lane_semantic_contract.md:36`
- **lane_semantic（12 维通道）**（KEEP·REACHABLE）：prod=`net/semantic.py:1`、`net/semantic.py:6`、`net/semantic.py:8`；train=`tools/collect_expert.py:75`、`.slim/deepwork/p1_v2/round36/i1_extract_dataset.py:94`、`.slim/deepwork/p1_v2/round36/i1_extract_dataset.py:112`；eval=`.slim/deepwork/p1_v2/round36/c_report.py:171`、`.slim/deepwork/p1_v2/round36/j1_eval.py:565`、`.slim/deepwork/p1_v2/round36/c_old_eval_a3.py:394`；slim=`.slim/deepwork/p1_0/journal.md:492`、`.slim/deepwork/p1_0/journal.md:493`、`.slim/deepwork/p1_v2/round36/h1_manifest.json:16`；runs=`runs/round36/h3_hooks.json:120`、`runs/round36/h3_hooks.json:124`、`runs/round36/h3_hooks.json:128`；test=`tests/test_target_lane_continuity.py:19`、`tests/test_lane_semantic_contract.py:1`、`tests/test_lane_semantic_contract.py:23`；docs=`README.md:110`、`net/README.md:19`、`docs/repo_handbook.md:77`
- **set_target_lane_dropout（训练期 dropout）**（KEEP·REACHABLE）：prod=`net/model.py:66`、`net/model.py:489`、`net/model.py:1053`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:1199`、`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:1243`、`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:1347`；eval=`.slim/deepwork/p1_v2/round36/j1_eval.py:672`、`.slim/deepwork/p1_v2/round36/h1_skeleton_smoke.py:285`、`.slim/deepwork/p1_v2/round36/h1_skeleton_smoke.py:306`；slim=`.slim/deepwork/p1_v2/round36/journal.md:12`、`.slim/deepwork/p1_v2/round36/h1_manifest.json:53`、`.slim/deepwork/p1_v2/round36/h1_manifest.json:54`；runs=`runs/round36/h3_hooks.json:2893`、`runs/round36/h3_hooks.json:3587`、`runs/round36/h3_hooks.json:4276`；docs=`net/README.md:86`、`docs/repo_handbook.md:132`
- **lane_anchor_xy（注入锚折线）**（KEEP·REACHABLE）：prod=`net/model.py:60`、`net/model.py:956`、`net/model.py:1417`；train=`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:728`、`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:885`、`.slim/deepwork/p1_v2/round36/j1_train.py:770`；eval=`.slim/deepwork/p1_v2/round36/c_report.py:83`、`.slim/deepwork/p1_v2/round36/c_report.py:145`、`.slim/deepwork/p1_v2/round36/c_report.py:249`；slim=`.slim/deepwork/p1_v2/README.md:18`、`.slim/deepwork/p1_v2/round36/c_old_p1_deploy_record.diff.txt:7`、`.slim/deepwork/p1_v2/round36/c_old_p1_deploy_record.diff.txt:13`；runs=`runs/round36/c_selfproof.json:207`、`runs/round36/j1_eval/gate_t1_res_ep3.md:37`、`runs/round36/j1_eval/gate_t1_res_ep3.md:39`；docs=`net/README.md:81`、`docs/repo_handbook.md:90`、`docs/repo_handbook.md:312`
- **--lane-anchor inject（评测注入）**（LEGACY_REPRO·REACHABLE）：eval=`.slim/deepwork/p1_v2/round36/c_bitwise_check.py:9`、`.slim/deepwork/p1_v2/round36/c_bitwise_check.py:133`、`.slim/deepwork/p1_v2/round36/c_report.py:43`；slim=`.slim/deepwork/p1_v2/README.md:11`、`.slim/deepwork/p1_v2/README.md:18`、`.slim/deepwork/p1_v2/round36/c_old_p1_deploy_record.diff.txt:7`；runs=`runs/round36/c_smoke/bitwise_record.json:5`、`runs/round36/c_smoke/bitwise_record.json:60`、`runs/round36/c_smoke/g6_draft.json:5`；docs=`docs/repo_handbook.md:89`、`docs/repo_handbook.md:90`、`docs/repo_handbook.md:197`

## G 组｜legacy monitoring + others-28 维 legacy 重排

| 符号 | kind | prod | train | eval | test | slim | runs | docs | 行数合计 | string | config | ckpt | 建议判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `legacy_tags（monitoring 开关）` | ident | 10 | 7 | · | 5 | · | · | 6 | 28 | 2 | · | · | LEGACY_REPRO |
| `_LEGACY_* 常量/正则族` | ident | 29 | 12 | · | 5 | 1 | · | 3 | 50 | · | · | · | LEGACY_REPRO |
| `_group_of_tag / _write_legacy_grouped_tensorboard` | ident | 10 | · | · | 5 | · | · | · | 15 | · | · | · | LEGACY_REPRO |
| `_slim_tag（新 schema 重命名）` | ident | 8 | · | · | 12 | · | · | · | 20 | · | · | · | KEEP |
| `OTHERS_LEGACY_DIM / OTHERS_HEAD_LEGACY_DIM` | ident | 7 | · | · | · | 1 | · | 2 | 10 | · | · | · | LEGACY_REPRO |
| `_remap_legacy_others()（28→33 重排）` | ident | 3 | · | · | · | · | · | 3 | 6 | · | · | · | LEGACY_REPRO |
| `old-style encoders.lane/ttc ckpt 键` | ckpt | · | · | · | · | · | · | · | 0 | · | · | · | CHECKPOINT_ONLY（TESTS_DOCS_ONLY） |

> §5.69-② 修正：G 组范围不止 monitoring.py —— net/mem.py 的 others-28 维 legacy 重排/encoders 常量同批清理。

### G 组逐符号证据（每项 1 行：入口侧命中 → 其它命中）

- **legacy_tags（monitoring 开关）**（LEGACY_REPRO·REACHABLE）：prod=`pipeline/monitoring.py:14`、`pipeline/monitoring.py:68`、`pipeline/monitoring.py:155`；train=`pipeline/stages.py:214`、`pipeline/stages.py:215`、`pipeline/stages.py:221`；test=`tests/test_monitoring_tb_grouped.py:10`、`tests/test_monitoring_tb_grouped.py:292`、`tests/test_monitoring_tb_grouped.py:293`；docs=`docs/chatgpt5p6sol_review.md:545`、`docs/v7_reports/cleanup/CLEANUP_D_REPORT.md:27`、`docs/v7_reports/cleanup/CLEANUP_D_REPORT.md:146`
- **_LEGACY_* 常量/正则族**（LEGACY_REPRO·REACHABLE）：prod=`net/mem.py:90`、`net/mem.py:91`、`net/mem.py:121`；train=`.slim/deepwork/p1_v2/round26/r26_train.py:79`、`.slim/deepwork/p1_v2/round26/r26_train.py:537`、`.slim/deepwork/p1_v2/round27/r27_train.py:68`；slim=`.slim/deepwork/p1_v2/round37/handbook_facts.md:14`；test=`tests/test_dataset_contract_guard.py:9`、`tests/test_dataset_contract_guard.py:34`、`tests/test_dataset_contract_guard.py:51`；docs=`docs/repo_handbook.md:49`、`docs/deepseekv4p1_argue.md:413`、`docs/r21_lane_semantic_contract.md:83`
- **_group_of_tag / _write_legacy_grouped_tensorboard**（LEGACY_REPRO·REACHABLE）：prod=`pipeline/monitoring.py:157`、`pipeline/monitoring.py:293`、`pipeline/monitoring.py:333`；test=`tests/test_monitoring_tb_grouped.py:22`、`tests/test_monitoring_tb_grouped.py:24`、`tests/test_monitoring_tb_grouped.py:145`
- **_slim_tag（新 schema 重命名）**（KEEP·REACHABLE）：prod=`pipeline/monitoring.py:9`、`pipeline/monitoring.py:44`、`pipeline/monitoring.py:155`；test=`tests/test_monitoring_tb_grouped.py:25`、`tests/test_monitoring_tb_grouped.py:112`、`tests/test_monitoring_tb_grouped.py:114`
- **OTHERS_LEGACY_DIM / OTHERS_HEAD_LEGACY_DIM**（LEGACY_REPRO·REACHABLE）：prod=`net/mem.py:90`、`net/mem.py:91`、`net/mem.py:121`；slim=`.slim/deepwork/p1_v2/round37/handbook_facts.md:14`；docs=`docs/repo_handbook.md:49`、`docs/deepseekv4p1_argue.md:413`
- **_remap_legacy_others()（28→33 重排）**（LEGACY_REPRO·REACHABLE）：prod=`net/mem.py:110`、`net/mem.py:393`、`net/mem.py:407`；docs=`docs/v7_reports/cleanup/CLEANUP_D_REPORT.md:27`、`docs/v7_reports/cleanup/CLEANUP_D_REPORT.md:67`、`docs/v7_reports/cleanup/CLEANUP_D_REPORT.md:168`
- **old-style encoders.lane/ttc ckpt 键**（CHECKPOINT_ONLY·TESTS_DOCS_ONLY）：（全仓 0 命中）

## H 组｜cheap-path 隐式布尔（→ 显式 forward_mode 枚举）

| 符号 | kind | prod | train | eval | test | slim | runs | docs | 行数合计 | string | config | ckpt | 建议判定 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `wm_detach（no-op 形参）` | ident | 4 | 13 | · | 19 | 1 | 20 | 7 | 64 | 23 | 7 | · | LEGACY_REPRO |
| `rollout= 形参（forward 开关）` | ident | 11 | 70 | 90 | 58 | 92 | 30 | 25 | 376 | · | 1 | · | KEEP |
| `world_model= 形参（forward 开关）` | ident | 7 | 65 | 76 | 50 | 80 | 1057 | 13 | 1348 | · | 1 | · | KEEP |
| `forward_mode（提案枚举，尚不存在）` | ident | · | · | · | · | · | · | 3 | 3 | · | · | · | LEGACY_REPRO（TESTS_DOCS_ONLY） |
| `cheap path（文档口径）` | string | 7 | 13 | 21 | 15 | 33 | 7 | 38 | 134 | 134 | 2 | · | LEGACY_REPRO |

> §七-H：`rollout×world_model` 布尔组合语义混乱（'关 WM 仍 latent rollout'）；采纳显式枚举 forward_mode ∈ {policy_single_step, policy_wm_rollout, world_model_teacher_forcing, diagnostics}。

### H 组逐符号证据（每项 1 行：入口侧命中 → 其它命中）

- **wm_detach（no-op 形参）**（LEGACY_REPRO·REACHABLE）：prod=`net/model.py:152`、`net/model.py:1986`、`net/model.py:1999`；train=`tools/il_report.py:557`、`pipeline/stages.py:2366`、`pipeline/stages.py:3123`；slim=`.slim/deepwork/p1_v2/round37/handbook_facts.md:34`；runs=`runs/BTC20261009-0824_sw_pri512/stage_b/metrics.json:30`、`runs/BTC20261007-1838_v8chain/stage_b/metrics.json:30`、`runs/BTC20261007-2202_arm_t/stage_b/metrics.json:30`；test=`tests/test_stage_c_label_alignment.py:28`、`tests/test_v6_e1_router_monitor.py:111`、`tests/test_v6_e1_router_monitor.py:112`；docs=`net/README.md:60`、`docs/repo_handbook.md:112`、`docs/deepseekv4p1_argue.md:411`
- **rollout= 形参（forward 开关）**（KEEP·REACHABLE）：prod=`net/model.py:65`、`net/model.py:83`、`net/model.py:139`；train=`.slim/deepwork/p1_v2/collect_recovery_v2.py:311`、`.slim/deepwork/p1_v2/collect_recovery_v2.py:1375`、`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:19`；eval=`tools/debug_rollout_viz.py:9`、`tools/debug_rollout_viz.py:757`、`tools/controls/probe_freeze_audit.py:305`；slim=`.slim/deepwork/stage-c-v3.md:166`、`.slim/deepwork/planner-rl-feasibility.md:1025`、`.slim/deepwork/planner-rl-feasibility.md:1126`；runs=`runs/round38/a3_smoke/a3_tables.md:4`、`runs/round38/a3_smoke/a3.json:16`、`runs/round38/a1_smoke/a1_tables.md:4`；test=`tests/test_r26_delta.py:20`、`tests/test_r26_delta.py:51`、`tests/test_r26_delta.py:56`；docs=`net/README.md:49`、`net/README.md:50`、`net/README.md:52`
- **world_model= 形参（forward 开关）**（KEEP·REACHABLE）：prod=`net/model.py:139`、`net/model.py:1991`、`net/model.py:1996`；train=`.slim/deepwork/p1_v2/collect_recovery_v2.py:311`、`.slim/deepwork/p1_v2/collect_recovery_v2.py:1375`、`.slim/deepwork/p1_v2/round36/j2_train_stripped_check.py:880`；eval=`tools/debug_rollout_viz.py:9`、`tools/debug_rollout_viz.py:757`、`tools/controls/probe_freeze_audit.py:305`；slim=`.slim/deepwork/planner-rl-feasibility.md:1025`、`.slim/deepwork/p1_0/probe_controlled_forward.py:115`、`.slim/deepwork/p1_v2/action_audit.py:104`；runs=`runs/BTC20261005-071840_v7p2_s11_eval500/metrics.json:1705`、`runs/BTC20261005-071840_v7p2_s11_eval500/metrics.json:1751`、`runs/BTC20261005-071840_v7p2_s11_eval500/metrics.json:1797`；test=`tests/test_r26_delta.py:20`、`tests/test_r26_delta.py:56`、`tests/test_r26_delta.py:57`；docs=`net/README.md:50`、`net/README.md:52`、`docs/repo_handbook.md:112`
- **forward_mode（提案枚举，尚不存在）**（LEGACY_REPRO·TESTS_DOCS_ONLY）：docs=`docs/deepseekv4p1_argue.md:393`、`docs/deepseekv4p1_argue.md:411`、`docs/chatgpt5p6sol_review.md:570`
- **cheap path（文档口径）**（LEGACY_REPRO·REACHABLE）：prod=`net/model.py:64`、`net/model.py:139`、`net/model.py:1569`；train=`.slim/deepwork/p1_v2/round26/r26_train.py:10`、`.slim/deepwork/p1_v2/round26/r26_train.py:361`、`.slim/deepwork/p1_v2/round26/r26_train.py:556`；eval=`.slim/deepwork/p1_v2/round36/h1_compat_check.py:263`、`.slim/deepwork/p1_v2/round36/h1_compat_check.py:264`、`.slim/deepwork/p1_v2/round36/h1_compat_check.py:596`；slim=`.slim/deepwork/p2-contract.md:100`、`.slim/deepwork/p2-contract.md:102`、`.slim/deepwork/planner-rl-feasibility.md:1025`；runs=`runs/round36/h1/gate0_compat.md:12`、`runs/BTC20261005-1856_v7p2_s2_arm1/config.snapshot.yaml:7`、`runs/round31/probes/q2_rollout_consistency.json:57`；test=`tests/test_r26_delta.py:50`、`tests/test_r26_delta.py:51`、`tests/test_eval_reference.py:7`；docs=`net/README.md:52`、`docs/repo_handbook.md:112`、`docs/repo_handbook.md:141`


---

## 3. 判定与迁移顺序（逐组结论）

> 通用纪律（§5.68-④ + §六）：**迁移 → 删引用 → 删实现** 三笔独立提交（每笔可回退）；
> 每笔必过 **pytest 全量**（2026-10-15 实测 collect **973 项**；README 记录 966，差额 = R38-A4 新增
> `tests/test_step_geometry_modes.py` 及后续用例）+ **默认关逐位**（默认关路径键集/前向逐位不变）。
> `removal_commit` 在 `deprecation_manifest.yaml` 中：A/B/C/D 组已于 2026-10-15 回填（见本文件 §8 与 manifest `landing` 块），
> E/F/G/H 仍留空（归 P3）。

### A 组｜oneshot / plan_override / 别名 —— `ARCHITECTURE_FORBIDDEN`（8/9 项）

- **判定**：全部 8 项 `ARCHITECTURE_FORBIDDEN`（唯一动作源 + 每步单动作守则）。它们是**唯一"再犯风险最高"**的一组
  （GPT §七-A 原话），且 `PATH_HEAD_MODES` 里 `oneshot` 与生产模式同居一个枚举 ⇒ 配置误写即可复活。
- **可达面**：prod 89 行 / 4 文件（`net/model.py`、`net/path_head.py`、`net/__init__.py` + 并发新增的
  `net/probes.py`（R38-D1 探针容器引用"正式路径无 oneshot/plan_override"断言））+ eval 181 行 / 15 文件
  → **不是"死代码"而是"活着的禁则"**，必须走"迁移 + 删引用 + 删实现"三笔。
- **迁移目标**：`tools/legacy/oneshot_path.py`（`one_shot_plan` + `plan_override` 钉住逻辑 + 诊断用 `compose_path_plan` 一次性调用）。
- **同步面（硬约束，§5.69-③）**：8 个 `.slim` 工具带 `--diagnostic-oneshot`（`a_hardening_check`、`c_old_canary_eval`、
  `c_old_eval_a3`、`c_old_p1_deploy_record`、`c_selfproof`、`canary_eval`、`eval_a3`、`p1_deploy_record`）+
  `round36/j1_train.py`/`j2_train_stripped_check.py`/`j1_eval.py` 的模式声明 + **L-A 契约断言**
  （`net/model.py:1009`「structured 不使用 plan_override」、`:1774`）⇒ 断言保留（它是删枝后的护栏），只删 oneshot 分支。
- **测试面**：`tests/` 对 `oneshot`/`plan_override`/`PathExecutionHead`/`compose_path_plan` **零引用**
  （仅 `.slim/round36/a_hardening_check.py` 承担守卫验证）⇒ **CI 守卫需新建 pytest 载体**（见本文件 §5）。
- **ckpt 键面**：`meta.path_head_ablation` / `meta.path_head_mode`（声明口径，2 键）+ `path_head_*` 无参数键
  （`compose` 无参数 ⇒ oneshot 不留 state_dict 键）⇒ 迁移期由 migration layer 读出后**不透传 forward**。

### B 组｜K-anchor 旧计划头 —— `LEGACY_REPRO`（10 项）+ `CHECKPOINT_ONLY`（2 项）

- **判定**：设计上"产生独立计划候选"（GPT §七-B），长期关闭（`config/model.yaml: plan_anchor.enabled=false`）⇒ 整体迁
  `experiments/archive/k_anchor/`；`plan_head.anchor*` 键面与 `encoders.lane/ttc` 归 `CHECKPOINT_ONLY`。
- **可达面**：prod 115 行 / 5 文件（`config/model.yaml`、`config/train.yaml`、`net/plan_head.py`、`net/model.py`、
  **`net/anchor.py`（371 行工具集）**）+ train 79 行 / 4 文件（`tools/fit_plan_anchors.py`、`pipeline/stages.py` +
  `config/train.yaml` 锚损失块）。
- **迁移清单**：`net/anchor.py`（整文件）+ `net/plan_head.py:82-200`（选择/速度/残差头 + 3 buffer）+
  `net/model.py:112-123,418-451,1369,1961`（构造/装载/输出）+ `tools/fit_plan_anchors.py` +
  `config/plan_anchors_k6.json` + `config/model.yaml::plan_anchor` + `config/train.yaml::anchor_ce/anchor_wta/anchor_temperature`
  （41 行 train 命中）。
- **前置**：**先确认全仓 `num_anchors=0`**（GPT §七-B.1）——当前 `config/model.yaml` 已 `enabled=false`；
  另需扫 `runs/**/config.snapshot.yaml` 的 K-anchor run（产物面，不回改）。
- **测试面**：`tests/test_plan_anchor.py`（8 处）、`tests/test_recovery_recipe.py`、`tests/test_phase3_safe_recipe.py`
  直接构造 `num_anchors=6` ⇒ 删除时同步迁移（保留 1 个"K-anchor 归档仍可加载"的迁移单测）。
- **ckpt 键面**：13 个 ckpt 带 `plan_head.anchor_head/anchor_embed/anchor_ds/anchor_dth/anchors/residual_head/speed_head`（16 键）；
  15 个 ckpt 带 `encoders.lane/ttc`（4 键）——见 `docs/checkpoint_migration_contract.md` §2。

### C 组｜遗留 `set_ego_warp`（w1/w2）—— `LEGACY_REPRO`（3 项），`KEEP`（4 项）

- **判定**：`set_ego_warp()`/`w1`/`w2`/`_ego_warp_pose_k`（与 `ego_warp` 属性）= `LEGACY_REPRO`（迁 legacy probe）；
  `set_ego_warp_mode`/`assemble_pose_k`/`od_velocity_t0`/`WARP_MODES` = **`KEEP`**（R37 M1 主路径，不是废弃项）。
- **可达面**：prod 60 行 / 3 文件（`net/model.py` 互斥状态机 + `net/ego_warp.py` + `net/__init__.py`）+ eval 63 行。
- **硬前置（§5.69-①）**：w1/w2 是 analytic 的**逐位等价 oracle**（`m1_gate_a.py:137-141,588-590`；
  `b_warp_compat.py`）⇒ **先冻结 fixture**（把 w1/w2 期望值/逐位哈希写进 probe 模块或 npz fixture），再删实现，
  否则 M1/L-D 等价性证据链断裂。
- **同步面**：`.slim/round37/m1_gate_a.py`、`round33/b_warp_compat.py`、`.slim/round36/d1_ego_warp_tests.py`（4 文件含 `set_ego_warp(`）。

### D 组｜Policy/path_head 双实体收编 —— 2 项 `ARCHITECTURE_FORBIDDEN`、2 项 `LEGACY_REPRO`、6 项 `KEEP`

- **判定**：`path_head_ablation` + `set_path_head_ablation`（含 oneshot 分支）= `ARCHITECTURE_FORBIDDEN`；
  `enable_path_head`（开关镜像属性）、`set_policy_diag`/`policy_diag_enabled`（旧头在主 forward 里并行诊断）= `LEGACY_REPRO`。
  `set_path_head`/`path_head_trace`/`action_generator_names`/`self.policy`（默认关路径）/`policy_action_mu`（诊断键）/
  `contract_asserts` = **KEEP**（收编后仍需要：`action_generator_names` 是"唯一动作源"静态断言，`contract_asserts` 是护栏）。
- **收编目标**（GPT §七-D）：`model.policy` 单实体（`structured_action` / `action_distribution` / `auxiliary_intent`），
  旧 `PolicyHead` 迁 legacy 模型加载器，旧头诊断改离线工具。
- **顺序**（§七-D + §六-4）：**先做 checkpoint key 迁移器**（`path_head.*` 18 键 → 新命名空间）→ 再收编 → 再删开关。
- **同步面**：prod 79 行 / 3 文件、eval 128 行 / 17 文件（`--path-head {off,structured}` 三工具 + `.slim` 链）+ `tests/` 仅 11 行
  （`self.policy` 命中在 `test_stage_*` 的 stub forward 里，非契约测试）。
- **注意**：`plan_head.fusion.2.*`（K1 训练 allowlist）**与 K-anchor 无关**（§5.68-③），收编时勿混淆。

### E 组｜`path_*` 旁路字段 —— `ARCHITECTURE_FORBIDDEN`（9/9）

- **判定**：9 项全部 `ARCHITECTURE_FORBIDDEN`——违反两条守则：**"禁止依赖只有训练或评测脚本能生成的 `path_*` 旁路"**
  + **"缺失值静默填充"**（`net/path_head.py:929-943` `_vector_from_obs(default=0.0)`；`:57` 注释"缺键 → 0 + presence 掩码
  （默认 0 不静默，H3 hook 审计）"——**掩码存在但消费端仍吃 0**，M3 已证 `path_dynamic_ttc` 缺键→0 = "立即碰撞"，
  占装配错配效应 77%）。
- **可达面**：prod 仅 **`net/path_head.py` 1 文件 30 行**（`LONG_EXOG_OBS_KEYS` 声明 + `long_exog_from_obs` 消费）；
  train/eval 侧命中来自 `.slim` 注入口（`m6_calib`/`m7_fill`/`j1_train`/`j2_train_stripped_check`）与产物（runs 330 行）。
- **关键事实（本轮新增，供 Q4 裁决）**：**生产/训练/评测代码里没有任何 `path_*` 生产者**——
  - `env/obs/**`、`pipeline/**`、`tools/**` 对 7 字段**零命中**（仅 `net/path_head.py` 消费）；
  - 训练侧值来自离线数据集（`.slim/round36/i1_extract_dataset.py` → `datasets/path_exec_v1`，`_long_exog_row` 离线重算）；
  - 评测侧值来自 `.slim/round37/m6_calib.py`（中性填充）/`m7_fill.py`（真值回填）——即 §六-2 所指"只有训练或评测脚本能生成"。
  ⇒ **Q4 分类的实际动作 = 在 env/WM anchors 侧建立唯一出处（或删除字段）**，否则 prod 侧永远只能"缺键填 0"。
- **分类草案（落 `deprecation_manifest.yaml`，与 Q4 联动）**：
  `path_left/right_margin` → **并入标准 obs**（已有 `lane_semantic` 列 4/5 兜底，天然出处）；
  `path_static_distance`/`path_dynamic_distance`/`path_dynamic_ttc` → **WM anchors 确定性派生**（A1 主项 = 想象步几何接口，
  与 GPT §五 Gate C"动态距离/TTC 随 rollout 更新，不静默 hold"同源）；
  `path_nav_progress` → **并入标准 obs**（`obs['nav']` 末维 route_completion 已有兜底）；
  `path_distance_to_target_lane_end` → **并入标准 obs**（`lane_semantic` 列 11）
  或 **仅诊断**（取决于 Q4 对 F 组来源标签的裁决）。
- **前置**：与 `plan_head.fusion.2` + `StructuredPolicyHead` 输入 schema 冻结**同批**（schema 一变即须重训）；
  中间态禁止"None → 0"，每个 optional 字段必须伴随 valid mask 且消费端显式处理。

### F 组｜外部 target-lane 强逻辑 —— 7 项 `KEEP` + 1 项 `LEGACY_REPRO`

- **判定**：**不删**（GPT §七-F："当前阶段仍用于给定正确目标后的执行能力"）；只加**来源标签**
  （`road_level_navigation` / `teacher_target_lane` / `deployment_available`），长期移除生产部署对唯一 target lane 的依赖。
- **可达面**：prod 138 行 / 9 文件（`net/semantic.py`、`env/obs/lane_semantic.py`、`env/obs/builder.py`、`net/model.py`、
  `net/path_head.py` 等）⇒ 全仓最深的耦合面，任何"删 target lane"的动作都会破坏 Gate 证据，**本轮不动**。
- **唯一迁移项**：`--lane-anchor inject`（eval 107 行 / `--path-head structured` 依赖链）⇒ 与 D/E 组的 schema 冻结同批处理
  （它是"评测侧替网络选道"的注入口，属 §六-2 的"外部替网络选道"风险面，需在 schema 里显式标注来源）。

### G 组｜legacy monitoring + others-28 维 legacy —— 5 项 `LEGACY_REPRO` + 1 项 `CHECKPOINT_ONLY` + 1 项 `KEEP`

- **判定**：`pipeline/monitoring.py` 的 `legacy_tags`/`_LEGACY_*`（20 处常量+7 条 regex）/`_group_of_tag`/
  `_write_legacy_grouped_tensorboard` → `LEGACY_REPRO`（迁 `tools/legacy/monitoring_reader.py`，正式训练只写新 schema）；
  `net/mem.py::_remap_legacy_others` + `OTHERS_LEGACY_DIM`/`OTHERS_HEAD_LEGACY_DIM`（§5.69-② 扩围）→ `LEGACY_REPRO`
  （迁为显式 schema 适配器或删除，禁止"零填充 static 段"的静默重排）；
  `_slim_tag`（新 schema 重命名）= **KEEP**；`encoders.lane/ttc` 旧 ckpt 键 = `CHECKPOINT_ONLY`。
- **可达面**：prod 67 行 / 4 文件（`pipeline/monitoring.py` 31 + `pipeline/run_paths.py` + `net/mem.py` + `net/encoders.py`）
  + train 19 行（`pipeline/stages.py` 的 monitor 构造与 tag 喂入）。
- **测试面**：`tests/test_monitoring_tb_grouped.py`、`test_monitoring_v2.py`、`test_stage_b_monitor_holdout.py`
  直接覆盖 legacy 分支 ⇒ 迁移时同步（保 `legacy_tags=False` 新路径测试，legacy 用例随实现迁走）。

### H 组｜cheap-path 隐式布尔 —— 3 项 `LEGACY_REPRO` + 2 项 `KEEP`

- **判定**：`wm_detach`（**恒 no-op** 形参，`net/model.py:1870,1883`）= `LEGACY_REPRO`；`cheap path` 布尔组合语义
  （`rollout×world_model`）→ `LEGACY_REPRO`（换成显式 `forward_mode` 枚举）；`rollout=`/`world_model=` 本体 = **KEEP**
  （待枚举替换后一并迁移）；`forward_mode` = **待实现**（当前全仓 0 命中，sed 面只有文档）。
- **可达面**：prod 29 行 + train 161 行（`pipeline/stages.py` 2366/2864/5163 等）+ eval 187 行 ⇒ **训练侧最深的开关面**。
- **实施**：`forward_mode ∈ {policy_single_step, policy_wm_rollout, world_model_teacher_forcing, diagnostics}`；
  `wm_detach` 直接删（保留 1 个"旧签名透传 = 同结果"的回归单测覆盖迁移期）。
- **风险**：gate 语义（Gate B"WM 是否真跑"）依赖这组布尔 ⇒ 枚举化必须附**逐位对照**（旧 `rollout/world_model` 组合 →
  新枚举映射表 + 每组合一次前向逐位）。

---

## 4. 与 §5.69 第一轮排摸的对照（三条修正的落实）

| 组 | §5.69 第一轮（refs/文件） | 本轮（行命中；prod / train / eval） | 变化与修正落实 |
|---|---|---|---|
| A | 87/4；.slim 342/22 | **89 行 / 4 文件**；181 行 / 15 文件 | ①本轮确认 `.slim` 侧 eval 命中集中在其**评估链**（`c_old_*`/`canary_eval`/`eval_a3`/`p1_deploy_record`/`a_hardening_check`）而非"22 个工具全部"——带 `--diagnostic-oneshot` 的实际是 **8 个**；②`tests/` 对 A 组零引用（第一轮"已有测试"的说法应更正为"`.slim` 守卫脚本，无 pytest 载体"） |
| B | 80/9；fit 工具 6 | **115 行 / 5 文件**；train 79 行 / 4 文件 | 新增 **`net/anchor.py`（371 行）**单列；`config/train.yaml` 锚损失块 41 行（第一轮未计） |
| C | 28/4；.slim 50/12 | 60/15 | §5.69-①（oracle 先冻结）**维持**；本轮给出具体 oracle 位置（`m1_gate_a.py:137-141,588-590`） |
| D | 38/3（model 30） | **79 行 / 3 文件**；eval 128 行 / 17 文件 | §5.69-③ 细化：`.slim` 侧不是 22 工具，而是 **`--path-head/--diagnostic-oneshot` 装载链 8 工具 + 模式声明 3 处**；prod 3 文件（含并发新增 `net/probes.py`） |
| E | prod 14/1（仅 path_head.py） | prod 30/9（**仅 `net/path_head.py`**） | 复核一致 ⇒ "prod 爆炸半径极小"成立；**新增**：全仓无 prod/eval 生产者（见 §3-E） |
| F | 133/12；.slim 420/62 | **138 行 / 9 文件**；slim 384 行 / 39 文件 | 深耦合确认；本轮不改，只加来源标签 |
| G | 64/5（monitoring 31 / stages 21 / mem 8） | 67/10（monitoring 31 / mem+encoders 10 / run_paths） | §5.69-② 扩围已落实（`net/mem.py` 8 处 + `encoders` 常量 + `run_paths.py`） |
| H | 布尔组合语义 | prod 29 + train 161 + eval 187；`forward_mode` 0 命中 | 采纳显式枚举；`wm_detach` 确认 no-op（`net/model.py:1883`） |

---

## 5. CI 守卫清单（§八）—— 建议清单 + 现有测试载体与落点

> 结论：**五条守卫目前只有 1 条有 pytest 载体**（"禁缺失静默填 0"的部分语义在 `tests/test_step_geometry_modes.py`
> 的 presence/掩码用例里，且那是 R38-A4 的几何开关测试，不是 schema 守卫）。
> 建议新增 **`tests/test_r38_arch_guards.py`**（CPU 离线、无 metadrive；沿用 `tests/README.md` 的 stub 口径）。

| # | 守卫（GPT §八） | 建议实现（pytest） | 现有载体 / 落点 | 依赖的机器表 |
|---|---|---|---|---|
| 1 | 禁止生产目录重新引用 legacy 模块 | AST/文本扫描：`net/`、`pipeline/`、`tools/`（官方 CLI）禁出现 `tools.legacy`/`experiments.archive`/`legacy_key_mapper` 之外的 legacy 导入；白名单=迁移层 | **无** ⇒ 新建 `tests/test_r38_arch_guards.py::test_no_legacy_import_in_prod`（可仿 `tests/test_dataset_contract_guard.py` 的"失败即 SystemExit"风格） | `c5_reach_scan.json`（bins=prod 的行命中） |
| 2 | 禁止 default config 出现 deprecated 键 | 加载 `config/default.yaml`（含 includes 平铺）+ `config/model.yaml`/`train.yaml`，断言黑名单键 0 命中：`plan_anchor`、`anchor_ce/anchor_wta`、`path_*`、`num_anchors`、`oneshot`、`diagnostic_oneshot`、`wm_detach` | 现有 `tests/test_arm_configs.py`、`tests/test_phase3_recipe_switches.py` 是"配置开关行为"测试（可并列落点）；黑名单守卫**无** ⇒ 新建用例 | 扫描 `config` 维度（本文件各表 `config` 列） |
| 3 | 禁止正式 forward 含 oneshot / plan_override | 构造 structured 模型 → 断言 `model.path_head_mode() == "structured"`、`action_generator_names() == ("path_head",)`、`forward(..., )` 输出**不含** `plan_override` 语义（`plan[:,0] == action_mu` 恒等式，已有运行时契约断言）+ 源码级断言：`net/model.py` 的 `forward` 调用图不出现 `plan_override` | **运行时契约断言已存在**（`net/model.py:1009`；默认开）⇒ 守卫 = 把它下沉为 pytest；G 组 A 的运行时部分**已有**，源码级扫描**无** | A 组表（prod 12 行 `plan_override`） |
| 4 | 禁止两个动作生成器同时活跃 | `model.action_generator_names()` 单元素 + `structured` 模式下旧 `policy.head` 调用计数 = 0（`policy_diag_enabled=False`）；训练路径断言 `contract_*` 计数为 0 | **运行时断言已存在**（同 #3）；`tests/test_policy_logstd_max.py`、`tests/test_stage_c_*` 覆盖周边 ⇒ 新建用例把断言固化 | D 组表（`self.policy`/`policy_action_mu`）+ `net/model.py:1774` |
| 5 | 禁止缺失关键字段静默填 0 | `long_exog_from_obs({}, batch=1)` ⇒ 断言返回 `present` 全 0 **且**消费端不把 0 当作有效值（要么 raise，要么返回带 `*_valid` 的输出）；`mem_from_obs` 旧 others-28 ⇒ 断言不再静默零填充（改为显式适配器或 raise） | 部分语义在 `tests/test_step_geometry_modes.py`（presence/mask）+ `tests/test_p1a_static_obs.py`（旧数据回退）⇒ 新建 schema 守卫用例 | E/G 组表（`_vector_from_obs` prod 7 行；`_remap_legacy_others` 3 行） |

**CI 建议的机器化前置**：`c5_reach_scan.py` 可直接作为 CI 的"引用面快照"工具——
把 §八 的 4 份文档 + 3 份 JSON（`c5_reach_scan.json`/`c5_ckpt_keys.json`/`c5_ckpt_families.json`）作为
`removal_commit` 的对照基线；每笔删枝提交后重跑扫描，断言目标符号行命中归零（除白名单）即闭环。

---

## 6. 限制与未决

1. **未扫面**：`runs/**.json > 2 MB`（119 文件 / ≈39.6 GB）与二进制 ckpt（522 个 `.pt`）不在文本扫描内；
   ckpt 面由 `--ckpt-keys`/`--ckpt-families` 单独普查（`docs/checkpoint_migration_contract.md`）。
2. **行命中 ≠ 语义引用**：注释/文档串同计（保守高估）；判定不依赖绝对数，依赖"是否有入口侧命中 + 是否违反守则"。
3. **`.slim` 子分类按文件名模式**（`SLIM_TRAIN_PATTERNS`/`SLIM_EVAL_PATTERNS`，见脚本常量）；
   个别脚本可能被归到相邻 bin（例如 `a4_run.py` 归 eval），不影响组级结论。
4. **未决（需 CG/Q4 裁决后回填）**：
   - E 组 7 字段的最终归属（删除 / 标准 obs / WM anchors 派生 / 仅诊断）——决定 schema 冻结与重训规模；
   - F 组来源标签的落点（obs schema 字段 vs 模型侧标签）；
   - 正式入口口径：是否把结构化头链从 `.slim` 收编进 `pipeline/eval_runner.py`（否则 Clean500/Eval500 统一 protocol hash
     重跑只能继续用 `.slim` 链，且 `tools/test.py` 无法评测当前候选 ckpt 的 structured 口径）；
   - `removal_commit`：A/B/C/D 已回填（§8）；E/F/G/H 留空待 P3。

## 7. 交付自证

| 项 | 值 |
|---|---|
| 扫描脚本 | `.slim/deepwork/p1_v2/round38/c5_reach_scan.py`（只读；不 import 项目代码） |
| 机器产物 | `c5_reach_scan.json`（逐符号 bins/文件清单/ckpt 键名/证据）、`c5_reach_tables.md`（本文件内嵌表）、`c5_ckpt_keys.json`（3 代 ckpt 全键面）、`c5_ckpt_families.json`（522 ckpt 键族普查） |
| 扫描规模 | 文本文件 6308（跳过 119 个超大 json；数字随并发 lane 漂移）；符号 67 个（A9/B12/C7/D10/E9/F8/G7/H5） |
| 判定分布 | `ARCHITECTURE_FORBIDDEN` 19、`LEGACY_REPRO` 24、`CHECKPOINT_ONLY` 3、`KEEP` 21（合计 67）；可达性：`REACHABLE` 65、`TESTS_DOCS_ONLY` 2 |
| 未写代码 | 本任务仅新增 C5 文档/脚本；`net/`、`pipeline/`、`config/`、`tests/` 零改动（`git status` 自证） |
| 快照时点 | **2026-10-15 18:33（工作树快照）**。扫描期间并发 lane（R38-A4 已入库 `b9d1721`；其后 `net/**` 仍有在飞改动：`probes.py`/`mem.py`/`model.py` 等被并发改写）⇒ **数字会随该 lane 落定而漂移**；减枝前请重跑 `c5_reach_scan.py` 刷新基线（判定逻辑与 bins 定义不变，预期只影响 G/D/H 的行命中数） |
| 未写代码（自证） | 本任务只新增：4 份 C5 文档 + `.slim/deepwork/p1_v2/round38/c5_*`（1 脚本 + 4 产物/中间件）；`net/`、`pipeline/`、`config/`、`tests/` **零改动**（`git status` 中 net/** 的改动来自并发 lane，非本任务） |

---

## 8. 落地状态（2026-10-15 回填；A/B/C/D 已完成，E/F/G/H 归 P3）

> 出处：`docs/deprecation_manifest.yaml::landing`（本批回填）+ 各波执行报告
> （`.slim/deepwork/p1_v2/round38/{p1_report,p1b_report,p2a_report,p2b_report,q4a_report}.md`）。
> "已完成" = 该组清单项的迁移 → 删引用 → 删实现已入库；"残留" = 有意保留的护栏/守卫与冻结期例外（非漏项）。

### 8.1 A/B/C/D 波次（commit + 证据 + 残留）

| 组 | 波次 | 实现 commit | 机器/测试证据 | 残留（有意保留） |
|---|---|---|---|---|
| **A**（8 项） | R38-P1 | `596747c`（S1 迁 `tools/legacy/oneshot_path.py`）+ `48cb5eb`（S2 删引用 + S3 删实现） | `p1_report.md` / `p1_reach_tables_post.md`（prod 89→24 行）；`tests/test_r38_p1_oneshot_path_removed.py`、`tests/test_legacy_oneshot_path.py` | 契约⑤键 `structured_plan_override_used`（语义=结构派生"路径不存在"）；`PATH_HEAD_MODES` 两档；`path_head_mode()` 读出保留（P2b 改派生）；`net/probes.py` 守卫 flag；`.slim` 侧守卫 `allow=` 兼容 tripwire + `a3/eval_a3.py` 冻结未清（`p1b_report.md` §4.1） |
| **B**（12 项） | R38-P2a | `a226285`（S1）+ `dd0a8c2`（S2/S3） | `p2a_report.md` / `p2a_reach_tables_post.md`（12/12 符号 prod=0）；`p2a_ckpt_mapper_check.json`（8 代表 ckpt 三档分类） | `legacy_key_mapper.IGNORE_PREFIXES`（20 键，CHECKPOINT_ONLY 隔离）；13 带 K-anchor / 15 带 lane·ttc 的旧 ckpt 键面按规定忽略（`c5_ckpt_families.json`） |
| **C**（3 项） | R38-P2b | `939d1a2`（C 组旧 warp 退役 + D 组双实体收编；docs 收口 = `2cfd8e8`） | `p2b_report.md`；w1/w2 oracle fixture `tests/fixtures/r38_p2b_warp/` + `p2b_m1_gate_a.md` / `p2b_b_warp_compat.md`；`tests/test_r38_p2b_warp_legacy_removed.py` | 快照 `tools/legacy/legacy_ego_warp.py`；`WARP_MODES=("off","analytic")`（KEEP 项，net/ego_warp.py:103） |
| **D**（4 项） | R38-P2b | `939d1a2`（同上） | `p2b_report.md`（默认路径逐位：plain 25 键 / structured 73 键 `max│Δ│=0`；键集归一 sha 一致）；`p2b_keys_evidence.json` / `p2b_ckpt_mapper_check.json`；`tests/test_r38_p2b_policy_fold.py` | `set_path_head_ablation` 保留两档 legacy 诊断入口（oneshot fail-fast，net/model.py:909-931）；`set_policy_diag` fail-fast 桩（指离线工具）；`path_head` 只读 alias（:864-872）；`.slim` 调用点已接 `load_checkpoint_migrated`（`p2b_report.md` S2 表） |

- 改前基线可重放：`p1_reach_tables_pre_refresh.md` / `p2a_reach_tables_pre_refresh.md` / `p2b_reach_tables_pre_refresh.md`
  与对应 `*_scan_post.json` 配对（各波 `*_report.md` 内为该组"迁移 → 删引用 → 删实现"的逐笔可回退证据表）。
- 类目归位注记：`compose_path_plan()` 审计判定为 **KEEP**，实际随 H1 一次性六步路径**实迁** legacy 并从 `net/` 删除
  （`48cb5eb`；`landing.reclassifications`）。
- 改名映射登记：`path_head.*`（18 键）→ `policy.structured_action.*`（`landing.rename_map`；
  代码 = `tools/legacy/legacy_key_mapper.py::PREFIX_RENAME_MAP`，条件族"装配缺席 ⇒ drop + 计数"）。

### 8.2 Q4-A（E 组前置，生产者侧已落地）

- commit `9577078`：`env/obs/long_exog.py`（动态距离/TTC/静态距**公式单一出处**）+ `env/obs/path_field.py`
  （t0 段 8 列 + 同源 `<key>__present`）+ `env/obs/builder.py` 挂载（`out.update(path_long_exog_fields(env, out))`）；
  `net/path_head.py`：`LONG_EXOG_STRICT_COLUMNS=(4,5)` ⇒ dynamic_distance/dynamic_ttc **缺键 fail-fast**（禁静默 0）。
- 证据：`q4a_report.md`（builder 产出 vs M7 `truth_fill` 逐位 334 帧 `max|Δ|=0`；闭环逐位复现 C1_true）、
  `tests/test_round38_q4a_path_fields.py`。
- 边界：E 组清单本体（net 侧 schema/掩码消费等 B 类）**仍未落地**，归 P3（与重训同批）。

### 8.3 E/F/G/H → P3

| 组 | 现状（HEAD=2cfd8e8） | P3 口径 |
|---|---|---|
| E（9 项） | 未落地；生产者侧已由 Q4-A 先行 | schema 冻结 + 掩码消费（B 类）+ static 下沉；与重训同批 |
| F（1 项） | 未动（只加来源标签，不删） | 来源标签随 schema/裁决落地 |
| G（6 项） | 未落地 | monitoring `_LEGACY_*` 族 + others-28 legacy 重排迁移/删除 |
| H（3 项） | 未落地；`forward_mode` 全仓 0 命中（本审计 §2 H 组表复核） | 枚举化 + 旧布尔组合逐位对照表（含 `wm_detach` 删除） |

### 8.4 复核命令（刷新引用面 / 对账各波归属与残留）

```bash
# 全仓引用面刷新（判定逻辑与 bins 不变；与各组 post 快照对账）
tools/venv-python .slim/deepwork/p1_v2/round38/c5_reach_scan.py
# 旧 ckpt × 键面映射复核（8 代表 ckpt；重放 D 组改名与 B 组 ignore 的逐族计数）
tools/venv-python .slim/deepwork/p1_v2/round38/p2b_ckpt_mapper_check.py
# 默认关逐位/键集证据（HEAD 快照 × 工作树；重放 P2b 键面 sha 对账）
tools/venv-python .slim/deepwork/p1_v2/round38/p2b_keys_evidence.py
```
