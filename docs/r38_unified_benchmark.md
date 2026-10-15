# R38 统一闭环基准（草稿）：protocol hash + Clean500 定义 + 关键版本重跑

> **状态：草稿（R38-C6 产物；不 commit）**。定稿待 CG：§5 待补清单 + §6 层级/口径提案。
> **证据**：`runs/round38/c6/**`（逐臂 `run_meta.json` / `protocol.json` / episodes）、
> `.slim/deepwork/p1_v2/round38/c6_report.md`（自动渲染）+ `c6.json`（全量）。
> **依据**：GPT R38 §五 Gate D / §九.6；R37 §1.3 同协议命令行；R37 台账 §3 矩阵（Calib50 已完成）。

---

## 1. 目的与判定口径

- **目的**：所有关键版本（旧正式基线、当前候选、IDM+PP 规则基线）在**同一评测协议**下重跑，
  每次 run 记录可复算的 **evaluation_protocol_hash**；杜绝"跨协议混比"（换 spec/换 tracker/换
  workers/改 net 代码后的旧读数直接与新读数对比）。
- **层级**（本基准提案，GPT R38 Gate D）：**Calib50（快淘汰）→ Clean500（历史统一晋级集）→
  Eval500（正式比较）**。
- **主排序（禁模糊加权）**：安全硬门（`off_road`、`collision`）→ `success` → `timeout` / `route
  completion` → 诊断量（速度/爬行/饱和等）。
- **单 seed 起步**：本 lane 各臂 1 seed（seed = spec 自带 `seed` 字段，`seed_mode=spec_native`）；
  **Eval500 3-seed + paired bootstrap/McNemar 留作后续门**（§5）。

## 2. evaluation_protocol_hash 定义（v1）

工具：`.slim/deepwork/p1_v2/round38/c6_protocol.py`（纯 CPU 只读；plan 臂与 IDM+PP 同一工具）；
每次 run 由 `.slim/deepwork/p1_v2/round38/c6_run.py` 落盘到 `<out>/protocol.json` + `<out>/run_meta.json`。

**哈希域**（canonical JSON：`sort_keys=True, separators=(",", ":")` → sha256 → 前 16 hex 为
`protocol_hash`，另存全长 `protocol_hash_full`）：

| 字段 | 取值（plan 臂） | 说明 |
|---|---|---|
| `spec_set.name` | `clean500` | 集名（§3） |
| `spec_set.sha256` | `087db3f5…`（全文） | spec 文件 sha256 |
| `spec_set.n` / `seeds_sha256` | 500 / `a2b66940…` | `(id, seed)` 排序列表的 sha256（seed 口径 `spec_native`） |
| `harness.workers` / `max_steps` / `device` | 6 / 1000 / cuda | 并行与截断口径 |
| `control.tracker` / `tail` | `lqr` / `chain` | 执行口径（R37 §1.3） |
| `model.ld_mode` / `rollout_semantic` / `moe` | `semantic` / `static_role_only` / `off` | 模型口径 |
| `frozen.net_tree_sha16` | 每 run 现算 | `net/**/*.py` 逐文件 sha256 的树指纹（`m2_net_sha.py` 同源） |
| `frozen.entry.script` / `sha16` | `.slim/deepwork/p1_v2/a3/eval_a3.py` | 评测入口脚本文件 sha16 |

- **`spec_set.path` 不入 hash**（仅记录）：同一内容经相对/绝对路径引用必须同 hash；内容由
  `sha256` 全量覆盖（该修正为 C6 内部 v1 定义，本 lane 已用修正后定义重跑全部臂）。
- **`protocol_common_hash`** = 仅 `spec_set + harness.workers/max_steps + frozen.net_tree_sha16`
  的 sha256(hex16)：**所有臂（含 IDM+PP）必须一致**，用于证明差异只可能来自控制/模型口径字段。
- **IDM+PP 的 not_applicable 记录**：`control.tracker/tail` 与 `model.*` 取
  `not_applicable:baseline_native_control` —— 规则策略由 `PurePursuitIDMPolicy` 原生控制
  （`pipeline/eval_runner.py:604 _BaselineController`、`:1889` tracker 仅 ckpt 记录），不消费这些
  字段；其 hash 差异**仅在**这批显式登记的 n/a 字段（报告逐条给出，见 §4 与 `c6.json`）。
- **复算命令**：

```bash
tools/venv-python .slim/deepwork/p1_v2/round38/c6_protocol.py \
  --spec docs/v7_reports/specs/specs_val_only500.json --set-name clean500 \
  --policy-kind ckpt --entry .slim/deepwork/p1_v2/a3/eval_a3.py        # plan 臂
tools/venv-python .slim/deepwork/p1_v2/round38/c6_protocol.py \
  --spec docs/v7_reports/specs/specs_val_only500.json --set-name clean500 \
  --policy-kind baseline --entry tools/baseline_eval.py                # IDM+PP
```

> 注意：hash 含 `net_tree_sha16` 与入口脚本 sha16 ⇒ **net/ 或 eval_a3.py 的任何内容改动都会换
> hash**。每臂 run 前后各记一次指纹（`net_changed_during_run` / `entry_changed_during_run`），
> 为 True 的臂须复核/重跑（本 lane 的处置见 `c6_report.md` §4）。

## 3. Clean500 定义（核验式；不新建正式 spec）

**定义**：`Clean500 := docs/v7_reports/specs/specs_val_only500.json`
（n=500，split=val，sha256 `087db3f5c02e9cedccb8313733207f59dfa90162002b0d9bf002e5b80bcc366e`，
seeds_sha256 `a2b669407dc8db2e…`），即 **`env/specs/scenarios_val.json`（1000）减去
`env/specs/scenarios_eval500.json`（500，sha256 `98856105…`）**。

核验（`c6_clean500_recon.py`，全过；`runs/round38/c6/clean500_recon.json`）：

- `clean500 ∩ eval500 = ∅`（逐 id）、`clean500 ∪ eval500 = val`（逐 id，500+500=1000）✅
- 难度分布：easy 189 / medium 170 / hard 141；11 类几何各 45–46 条 ✅
- **历史一致性**：v7 预注册（`docs/v7_program_prereg.md` §0）即定义 clean500 =
  `/tmp/opencode/phase3_diag/exp/specs_val_only500.json`（该 tmp 路径已不存在）；现存两份独立副本
  （旧 worktree `/tmp/opencode/s11_v7/…`、`/tmp/opencode/p1a_wt/…`）与仓库副本**逐字节同 sha**；
  v7 与 p1_v2 两代 run 均直接引用仓库路径（例：`runs/eval/BTC20261010-*_c500` 的 `specs_path`）
  ⇒ **仓库副本 = 历史 clean500，无需新建**。
- **clean150**（`docs/v7_reports/specs/specs_val_only150.json`，n=150）= clean500 的**前 150 条**
  （文件序）；⊂ clean500、∩ eval500 = ∅；R37 三级协议的中间层即此文件 ✅
- **calib50**（`runs/p1_v2/round25/specs_calibration50.json`，n=50）为 **train split** 开发集
  （id 7xxx），与 clean500/eval500 **零交集** ⇒ 主线历史锚点（legacy 0.40 / Q1 0.50 / K1 0.52 /
  T2 0.58 / 统一基线 0.56 / IDM 0.756）与 clean500 读数**不可直比**，须逐条登记（R37 §7 待令④）。

**不新建 `env/specs/scenarios_clean500.json` 的理由**：仓库已含历史 clean500 文件且被两代 run 直接
引用；新建只会制造第二真源。若 CG 要求规范化，建议后续轮次以**同 sha 复制**方式入库并同步
`specs_path` 登记（本 lane 仅留提案）。

## 4. 重跑结果（Clean500，n=500，单 seed；安全硬门 → success 排序）

<!-- C6_TABLE_START -->
（待各臂跑完填充；见 `.slim/deepwork/p1_v2/round38/c6_report.md` 自动表）
<!-- C6_TABLE_END -->

## 5. 待补清单

<!-- C6_PENDING_START -->
（待填）
<!-- C6_PENDING_END -->

## 6. 层级/口径提案（待 CG）

- [ ] 确认 R38+ 层级 = Calib50 → **Clean500** → Eval500；clean150 降为"clean500 前 150 条"的可选
  快筛（不再作为独立晋级门），或保留为独立层——二选一需 CG 拍板。
- [ ] 是否将 clean500 以同 sha 复制入 `env/specs/`（第二真源 vs 引用稳定性取舍）。
- [ ] 是否把 protocol hash 常态化接入评测入口（当前为 C6 wrapper 侧记录；建议后续
  `eval_a3`/`eval_runner` 原生落 `protocol.json`）。
- [ ] calib50 历史锚（0.40–0.58 / IDM 0.756）跨集登记表（R37 §7 待令④）。
