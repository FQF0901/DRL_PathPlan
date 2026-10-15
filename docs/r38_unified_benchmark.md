# R38 统一闭环基准（草稿）：protocol hash + Clean500 定义 + 关键版本重跑

> **状态：草稿（R38-C6 产物；不 commit）**。定稿待 CG：§6 待补清单 + §7 层级/口径提案。
> **证据**：`runs/round38/c6/**`（逐臂 `run_meta.json` / `protocol.json` / episodes）、
> `.slim/deepwork/p1_v2/round38/c6_report.md` + `c6.json`（全量自动渲染）。
> **依据**：GPT R38 §五 Gate D / §九.6；R37 §1.3 同协议命令行；R37 §3 矩阵（Calib50 已完成）。

---

## 1. 目的与判定口径

- **目的**：所有关键版本（旧正式基线、当前候选、IDM+PP 规则基线）在**同一评测协议**下重跑，
  每次 run 记录可复算的 **evaluation_protocol_hash**；杜绝跨协议混比。
- **层级**（本基准提案，GPT R38 Gate D）：**Calib50（快淘汰）→ Clean500（历史统一晋级集）→
  Eval500（正式比较）**。
- **主排序（禁模糊加权）**：安全硬门（`off_road`、`collision`）→ `success` → `timeout` /
  `route completion` → 诊断量（速度/爬行/饱和等）。
- **单 seed 起步**：本 lane 各臂 1 seed（seed = spec 自带 `seed` 字段，`seed_mode=spec_native`）；
  **Eval500 3-seed + paired bootstrap/McNemar 留作后续门**（§6）。

## 2. evaluation_protocol_hash 定义（v1）

工具：`.slim/deepwork/p1_v2/round38/c6_protocol.py`（纯 CPU 只读；plan 臂与 IDM+PP 同一工具）；
每次 run 由 `.slim/deepwork/p1_v2/round38/c6_run.py` 落盘到 `<out>/protocol.json` + `<out>/run_meta.json`。

**哈希域**（canonical JSON：`sort_keys=True, separators=(",", ":")` → sha256 → 前 16 hex 为
`protocol_hash`，另存全长 `protocol_hash_full`）：

| 字段 | 取值（plan 臂） | 说明 |
|---|---|---|
| `spec_set.name` | `clean500` | 集名（§3） |
| `spec_set.sha256` | `087db3f5…`（全文） | spec 文件 sha256 |
| `spec_set.n` / `seeds_sha256` | 500 / `a2b66940…` | `(id, seed)` 排序列表 sha256（seed 口径 `spec_native`） |
| `harness.workers` / `max_steps` / `device` | 6 / 1000 / cuda | 并行与截断口径 |
| `control.tracker` / `tail` | `lqr` / `chain` | 执行口径（R37 §1.3） |
| `model.ld_mode` / `rollout_semantic` / `moe` | `semantic` / `static_role_only` / `off` | 模型口径 |
| `frozen.net_tree_sha16` | 每 run 现算 | `net/**/*.py` 逐文件 sha256 的树指纹（`m2_net_sha.py` 同源） |
| `frozen.entry.script` / `sha16` | `.slim/deepwork/p1_v2/a3/eval_a3.py` | 评测入口脚本文件 sha16 |

- **`spec_set.path` 不入 hash**（仅记录）：同一内容经相对/绝对路径引用必须同 hash；内容由
  `sha256` 全量覆盖（C6 内部 v1 修正；全部臂以修正后定义重跑，见 §5）。
- **`protocol_common_hash`** = 仅 `spec_set + harness.workers/max_steps + frozen.net_tree_sha16`
  的 sha256(hex16)：**所有臂（含 IDM+PP）必须一致**，用于证明差异只可能来自控制/模型口径字段。
- **IDM+PP 的 not_applicable 记录**：`control.tracker/tail` 与 `model.*` 取
  `not_applicable:baseline_native_control` —— 规则策略由 `PurePursuitIDMPolicy` 原生控制
  （`pipeline/eval_runner.py:604 _BaselineController`；`:1889` tracker 仅 ckpt 记录），不消费这些
  字段；其 hash 差异**仅在**这批显式登记的 n/a 字段。
- **复算命令**（对任意臂可复算；`--net-tree-sha16` 可喂冻结树）：

```bash
tools/venv-python .slim/deepwork/p1_v2/round38/c6_protocol.py \
  --spec docs/v7_reports/specs/specs_val_only500.json --set-name clean500 \
  --policy-kind ckpt --entry .slim/deepwork/p1_v2/a3/eval_a3.py        # plan 臂
tools/venv-python .slim/deepwork/p1_v2/round38/c6_protocol.py \
  --spec docs/v7_reports/specs/specs_val_only500.json --set-name clean500 \
  --policy-kind baseline --entry tools/baseline_eval.py                # IDM+PP
```

> hash 含 `net_tree_sha16` 与入口脚本 sha16 ⇒ **net/ 或 eval_a3.py 任何内容改动都会换 hash**。
> 每臂 run 前后各记指纹（`net_changed_during_run` / `entry_changed_during_run`），为 True 的臂
> 需复核/重跑（本 lane 的处置 = §5）。

## 3. Clean500 定义（核验式；不新建正式 spec）

**定义**：`Clean500 := docs/v7_reports/specs/specs_val_only500.json`
（n=500，split=val，sha256 `087db3f5c02e9cedccb8313733207f59dfa90162002b0d9bf002e5b80bcc366e`，
seeds_sha256 `a2b66940…`），即 **`env/specs/scenarios_val.json`（1000）减去
`env/specs/scenarios_eval500.json`（500，sha256 `98856105…`）**。

核验（`c6_clean500_recon.py` 全过；`runs/round38/c6/clean500_recon.json`）：

- `clean500 ∩ eval500 = ∅`（逐 id）、`clean500 ∪ eval500 = val`（逐 id，500+500=1000）✅
- 难度分布：easy 189 / medium 170 / hard 141；11 类几何各 45–46 条 ✅
- **历史一致性**：v7 预注册（`docs/v7_program_prereg.md` §0）即定义 clean500 =
  `/tmp/opencode/phase3_diag/exp/specs_val_only500.json`（该 tmp 路径已不存在）；现存两份独立副本
  （旧 worktree `/tmp/opencode/s11_v7/…`、`/tmp/opencode/p1a_wt/…`）与仓库副本**逐字节同 sha**；
  v7 与 p1_v2 两代 run 均直接引用仓库路径（例：`runs/eval/BTC20261010-*_c500`）⇒
  **仓库副本 = 历史 clean500，无需新建**。
- **clean150**（`docs/v7_reports/specs/specs_val_only150.json`，n=150）= clean500 的**前 150 条**
  （文件序）；⊂ clean500、∩ eval500 = ∅；R37 三级协议中间层即此文件 ✅
- **calib50**（`runs/p1_v2/round25/specs_calibration50.json`，n=50）为 **train split** 开发集
  （id 7xxx），与 clean500/eval500 **零交集** ⇒ 主线历史锚（legacy 0.40 / Q1 0.50 / K1 0.52 /
  T2 0.58 / 统一基线 0.56 / IDM 0.756）与 clean500 读数**不可直比**，须逐条登记（R37 §7 待令④）。

**不新建 `env/specs/scenarios_clean500.json` 的理由**：仓库已含历史 clean500 文件且被两代 run 直接
引用；新建只会制造第二真源。若 CG 要求规范化，建议后续轮次以**同 sha 复制**入库并同步
`spec_path` 登记（本 lane 仅留提案）。

## 4. 重跑结果（Clean500，n=500，单 seed；安全硬门 → success 排序）

<!-- C6_TABLE_START -->
**全部 7 臂（pin @48cb5eb；n=500；单 seed）：**

| 臂 | succ | coll | off | tmo | rc | 速度 (m/s) | 步数 | wall_s |
|---|---|---|---|---|---|---|---|---|
| IDM+PP（baseline_eval 入口） | **0.742** | 0.174 | 0.054 | 0.030 | 0.865 | 7.75 | 354.7 | 117.0 |
| IDM+PP（eval_runner 交叉核验） | 0.744 | 0.172 | 0.054 | 0.030 | 0.866 | 7.75 | 354.9 | 120.4 |
| C0 旧正式基线（K1e2） | **0.588** | 0.120 | 0.142 | 0.152 | 0.878 | 3.63 | 725.1 | 1199.5 |
| C1 当前候选（结构化 Policy） | **0.304** | 0.124 | 0.130 | 0.442 | 0.751 | 2.51 | 852.6 | 1737.5 |
| C1_fill（+TTC 中性 17.7 s） | **0.366** | 0.116 | **0.208** | 0.310 | 0.776 | 2.85 | 786.2 | 1648.4 |
| C2（+analytic warp） | **0.302** | 0.122 | 0.128 | 0.448 | 0.749 | 2.51 | 850.8 | 1744.1 |
| C1_true（+真值回填） | **0.390** | 0.132 | 0.178 | 0.300 | 0.787 | 2.94 | 775.0 | 1595.6 |

- **IDM+PP 双入口一致性**：逐 episode 仅 id=593 一处 driver 口径差（1/500；baseline_eval 记碰撞 /
  eval_runner 记到达）⇒ 锚 = 0.742–0.744；**与 v7 clean500 历史锚 0.742（coll 0.174 / off 0.054 /
  rc 0.865 / sr 0.757）逐位复现**。
- **配对翻转（episode 级，McNemar 精确）**：C0→C1 success **+18/−160**（p<1e-4）、timeout +154/−9；
  C1→C1_fill success **+41/−10**（p<1e-4）、timeout +7/−73、**off_road +44/−5（p<1e-4）**、
  collision +9/−13（p=0.52 ns）；C1→C1_true success **+50/−7**（p<1e-4）、timeout +9/−80、
  **off_road +37/−13（p=0.0009）**、collision +11/−7（ns）；IDM→C1 +27/−246、IDM→C1_fill +25/−213。
- **候选排序（clean500，主排序口径）**：`c1_true` 0.390 > `c1_fill` 0.366 > `c1` 0.304 ≈ `c2` 0.302；
  **全部候选 off（0.128–0.208）>0.10、coll（0.116–0.132）>0.08 硬门 ⇒ 均不晋级**；`c2`（warp）
  相对 `c1` 同量级（succ −0.2pp、crawl 33.4 s 同）⇒ 机制臂中性确认。
- **与 calib50（R37）跨集登记**：C0 0.580→0.588；C1 0.340→0.304；C1_fill 0.420→0.366（off 0.12→0.208）；
  C1_true 0.400→0.390（off 0.18→0.178）；IDM 0.756→0.742。**同协议但不同集，禁直比**；本表只作登记。
- **硬门对照**：安全硬门（off ≤0.10 / coll ≤0.08，R37 Calib50 门）下 **C0/C1/C1_fill/C2/C1_true 均未过**
  （off 0.128–0.208、coll 0.116–0.132）；IDM+PP 的 coll 0.17 亦不过 coll 门（历史锚同款）。

**诊断量（同批，辅助解读；非门）：**

| 臂 | crawl_mean (s) | low-speed 步占比 | speed_ratio | min_ttc 均值 | a_lat p95 | steer 饱和（mean） |
|---|---|---|---|---|---|---|
| IDM+PP | 2.73（runner 口径） | 0.032 | 0.757 | 5.76 | 1.52 | n/a |
| C0 | 0.82 | 0.012 | 0.380 | 6.69 | 2.91 | 0.70 |
| C1 | **33.44** | **0.372** | 0.262 | 7.43 | 0.68 | 0.13 |
| C1_fill | **21.44** | 0.256 | 0.300 | 6.64 | 1.10 | 0.18 |
| C2 | 33.40 | 0.371 | 0.262 | 7.39 | 0.69 | 0.13 |
| C1_true | **15.32** | 0.179 | 0.307 | 6.94 | 1.16 | 0.21 |

⇒ clean500 上接口修复（fill/true）**解除部分爬行**（crawl 33.4→21.4/15.3 s、timeout 0.442→0.310/0.300、
succ 0.304→0.366/0.390）但**横向跟不上**（off 0.130→0.208/0.178、a_lat p95 0.68→1.10/1.16）——
与 calib50 的 R37 结论同型且更重（clean500 的 off 硬门是主要否决项）。
<!-- C6_TABLE_END -->

## 5. 运行处置：并发改动与冻结树（pin）

R38 同日多 lane 并行（A4/D1/P1/C1/C2/C3/Q4A），`net/` 在评测窗口被反复改动 ⇒ 两轮处置：

1. **首轮（仓库直跑）**：`idm_pp`/`idm_pp_runner` 有效；`c0_k1e2` 期间 net/ 被 R38-D1 改动
   （flagged）；`c1` 期间被 R38-P1（oneshot 减枝）改动并触发 108 契约断言失败 + 8 NameError
   （n_error=116）⇒ **作废**；`c1_fill` 被二次改动 ⇒ 作废。旧读数留档 `*_r1_oldtree` /
   `c1_r1_oldtree_invalid`。
2. **冻结重跑（pin）**：`git archive 48cb5eb` 快照 → `/tmp/opencode/c6_pin`（`runs/`、`.venv/`、
   `datasets/` 符号链接回仓库；产物落 `runs/round38/c6/`）。pin net 树 `0311046b…`、env = HEAD
   口径（Q4A 迁移前）、eval_a3 sha16 `2ff136b84a40ad72`；**本文所有臂读数 = pin 内重跑**
   （`pin_commit=48cb5eb`，逐臂 run_meta 可查）。`c1_true` 臂另叠加 Q4A `env/obs/long_exog.py`
   （sha256 `2366a626…`；m7_fill 迁移中途 re-export 依赖；纯函数 numpy-only）——pin 内
   `m7_unit_check --phase unit` **A–D 全过**（A row-wise bitwise / B 933·933 / C 1055·1055 /
   D 5.7e-14）⇒ overlay 公式语义受核验；已记入该臂 `pin_note`。
3. **跨树复现核验**（r1 vs pin）：`idm_pp` 与 `c0_k1e2` 逐 episode 结局字段（success/collision/
   off_road/termination）**500/500 全同**（各 1–2 条 steps 微差）⇒ D1 probes（默认关）与 P1
   oneshot 删除对该协议读数无结局级影响（跨树复现证据，非逐位断言）。

## 6. 待补清单（C6）

<!-- C6_PENDING_START -->
1. **Eval500 3-seed**（paired bootstrap/McNemar + `seed_mode` 三 seed 变体）——本 lane 只给单 seed
   （`spec_native`）；正式声明留该门。
2. **Q4A（`path_*`→标准 obs/禁静默 0）与 A4（step-geometry fixed）全量落地后**需再入本基准重跑
   （`net_tree_sha16`/env 换代 ⇒ 新 protocol hash；A4 的 `c1_gfix`/`c1_fill_gfix` 已在 calib50 有读数）。
3. **R38-P1 减枝后的 ckpt 兼容迁移**：本基准 pin 已含 oneshot 删除（48cb5eb），结构化路径读数有效；
   oneshot 相关历史 ckpt 复现走 `tools/legacy/oneshot_path.py`（P1 lane 责任）。
4. **calib50 历史锚跨集登记表**（0.40–0.58 / IDM 0.756；R37 §7 待令④）。
5. **建议**：protocol hash 记录原生接入 `eval_a3`/`eval_runner`（当前为 C6 wrapper 侧）；
   clean500 是否复制入 `env/specs/`（§7）。
6. **pin 可复现性**：本次冻结根 `/tmp/opencode/c6_pin`（易失）。复现：
   `mkdir -p /tmp/opencode/c6_pin && git archive 48cb5eb | tar -x -C /tmp/opencode/c6_pin &&
   ln -s <repo>/.venv <repo>/runs <repo>/datasets /tmp/opencode/c6_pin/ &&
   cp -a <repo>/.slim/deepwork/p1_v2 /tmp/opencode/c6_pin/.slim/deepwork/`
   （`.slim` 不入库，需从当轮工作树拷贝；`c1_true` 需叠加 `env/obs/long_exog.py` sha256 `2366a626…`）。
   校验锚：net 树 `0311046b`、eval_a3 sha16 `2ff136b84a40ad72`、spec sha256 `087db3f5…`。
<!-- C6_PENDING_END -->

## 7. 层级/口径提案（待 CG）

- [ ] 确认 R38+ 层级 = Calib50 → **Clean500** → Eval500；clean150 降为"clean500 前 150 条"的
  可选快筛（不再独立晋级门），或保留独立层——二选一需 CG 拍板。
- [ ] 是否将 clean500 以同 sha 复制入 `env/specs/`（第二真源 vs 引用稳定性取舍）。
- [ ] 是否把 protocol hash 常态化接入评测入口（当前为 C6 wrapper 侧记录；建议后续
  `eval_a3`/`eval_runner` 原生落 `protocol.json`）。
- [ ] calib50 历史锚（0.40–0.58 / IDM 0.756）跨集登记表（R37 §7 待令④）。
