# v6 前置清理报告

- 执行时间：2026-10-01（本次会话）
- 范围：`/workspace/01_Proj/DRL_PathPlan/runs`、`/tmp`、`/tmp/opencode`
- 原则：只删「可直接删」清单；每项删前在 `/tmp/opencode` 的 `*.md/*.json/*.sh/*.py` 中 grep 引用，被外部引用则改列候选；repo 受控文件未动（`git status --short` = 0 行）
- 结果（第一轮）：删除 30 条路径，释放约 **1.37 GiB**；`runs/` 未删任何内容
- 第二轮（同日追加）：候选 B 中外部引用 = 0 项再删 18 项，释放 79,056 KB ≈ **77.2 MiB**（见第六节）；两轮累计删除 48 条路径 ≈ **1.45 GiB**

---

## 一、已删除清单

### A. `/tmp/opencode`（4 项，169,392 KB ≈ 165 MiB）

| 路径 | 大小 | 引用检查 |
|---|---|---|
| `laneT_smoke/` | 109M | 无外部引用；仅目录内部 `metrics.json`/`*.report.json` 自引用（随目录一并删除） |
| `p3_micro_test/` | 27M | 无外部引用；仅自身 `metrics.json` 自引用 |
| `smokeA/` | 18M | 无外部引用；仅自身 `metrics.json` 自引用 |
| `gpu_a_old96.log` | 13M | 无任何引用 |

### B. `/tmp`（26 条路径，1,268,940 KB ≈ 1.21 GiB）

| 路径 | 大小 | 备注 / 引用检查 |
|---|---|---|
| `/tmp/pip/` | 1.1G | 内容为单个 `torch-2.3.0a0+ebedce2-cp310-cp310-linux_x86_64.whl`（非通用缓存）；无引用。若后续需重装该版本 wheel 需重新获取 |
| `/tmp/pytest-of-root/` | 114M | pytest 临时目录（pytest-531/532…）；删除时确认无 pytest 进程运行 |
| `/tmp/pdtest/` | 18M | 内含 `numpy-1.26.4-cp310` wheel；无引用 |
| `/tmp/node-compile-cache/` | 5.8M | node 编译缓存；无引用 |
| `/tmp/python-languageserver-cancellation/` | 188K | 语言服务器 hash 子目录集；无引用 |
| `/tmp/pip-install-*` × 19 | 合计 ≈0.97M | 残留 pip 构建目录；无引用 |
| `/tmp/dbg_laneT_8xcqhv_j/` | 1012K | 调试残留；无引用 |
| `/tmp/tmpar32w_yr/` | 308K | 临时残留；无引用 |

删除前全部逐项 grep（`.md/.json/.sh/.py`，含 `phase3_diag/`、`rollback/` 内部）→ 均无引用命中。

---

## 二、保留确认（全部在位）

**`runs/`（未删任何内容，1.9G → 1.9G）**
- `runs/_refs_rlbase/` ✓ 9.6M
- 零点引用 run：`BTC20260930-1434*`、`-1437*`、`-1441*`、`-1449*`、`-1604*` 各 1 个 ✓（`-1444*` 不存在，见异常 2）
- v3 臂 5 个：`1608_v3_p0`、`1625_v3_a1`、`1645_v3_a2`、`1706_v3_a3`、`1725_v3_sa` ✓；v4/v41 臂与诊断、`BTC20261001-*` 共 70 个 ✓
- `runs/reward_viz/` ✓ 207M
- `runs/` 目录总数 218（与清理前一致，未删）

**`/tmp/opencode/` 受保护项**
- 文件：`*.md` 33、`*.json` 86、`*.sh` 49、`*.py` 90、`*.txt` 44 —— 全部保留 ✓
- `phase3_diag/` ✓ 94M、`rollback/` ✓ 54M
- `v6_supervise.sh`（当前监督脚本）未动

**repo**
- `git status --short` 输出 0 行；未修改任何受控文件、`docs/`、`.slim/` ✓

---

## 三、候选清单（只报告，未删，待用户决定）

### A. `runs/` 2026-09-27 ~ 09-29 历史目录（合计 995M）

| 日期 | 大小 | 目录数（约） |
|---|---|---|
| 09-27 | 3.5M | 3 |
| 09-28 | 260M | 14 |
| 09-29 | 732M | 47 |

- 引用情况：**被 `/tmp/opencode` 下 118 个 `*.md/*.json/*.sh/*.py` 文件引用**（如 `night_dth/*`、`phase3_diag/*`、`rl_p3_a4~a6.json`、`chain_run_log.md`、`moe_eval_then_p3.sh`、`l2_resume.sh`、`ckpt_swap.py`、`stageA_boundary.sh`、`smoke_eval/moeoff/metrics.json` 等）。**建议保留或归档，不建议直接删**。
- 另有 `runs/_refs_oldgen/`（16M，09-29，README + final.pt + primary.pt）：不在禁删清单、也不在可删清单；被 `night_report_draft.md`、`night_report_final.md`、`v4_base/docs/version_ledger.md`、`r7_red/docs/version_ledger.md` 引用 → 单列候选。

### B. `/tmp/opencode/` >1M 未分类项（约 190M）

| 项 | 大小 | 用途 | 被引用 |
|---|---|---|---|
| `oldcode/` | 2.9M | 旧代码快照（net/tools/docs/tests） | 1 文件 |
| `r7_red/` | 3.3M | r7 red 实验代码副本（tests/docs/pipeline/reward_model） | 1 文件 |
| `v4_base/` | 5.5M | v4 基座代码副本（含 docs/version_ledger.md） | 0（内部引用 _refs_oldgen） |
| `phase3_smoke_dagger/` | 4.3M | phase3 dagger smoke 产物（expert_bc.npz 等） | 0 |
| `fixb_adv_cli` / `fixb_e1_pre` / `fixb_e1_post` / `fixb_e1_post2` / `fixb_lane_cli` | 5×4.5M | fixB 阶段 c 运行产物（metrics/monitor/ckpt） | 0/1/4/2/0 文件 |
| `g1_kl_probe`、`g1_kl_probe_patched`、`g1_obs_probe2/3/4` | 5×4.5M | G1 obs/KL probe 产物（metrics + final.pt） | 5/1/1/1/1 文件 |
| `r2_probe_run`、`r2_scope_all_run`、`r4_probe_run`、`r4_plan_arm_run`、`r7_noop_probe`、`r7_noop_probe_off` | 6×4.5M | r2/r4/r7 探针运行产物 | 0/1/0/0/9/2 文件 |
| `s1_off`、`s1_off_fp`、`s1_on`、`s1_noop_probe_rot` | 4×4.5M | s1 fingerprint 对照产物 | 3/2/1/2 文件 |
| `v8r_e1`、`v9_e1`、`v10_e1`、`v11_e1`、`v12_e1` | 5×4.5M | v8r~v12 单轮 e1 运行产物 | 均 0 |
| `v4_9_base_coef0`、`v4_9_new_coef0`、`v4_9_smoke`、`v4_smoke`、`v4_smoke2` | 5×4.5M | v4.9 系数/fingerprint smoke 产物 | 0/0/0/1/0 文件 |
| `night_forensics/` | 15M | night 取证 JSON 证据（lqr 对比） | 1 文件 |
| `primary_phase1.pt` | 7.2M | 09-27 早期 phase1 checkpoint | 4 文件 |
| `final_g2.pt` | 8.6M | 09-28 早期 G2 checkpoint | 2 文件 |
| `gpu_a_new1024.log`、`gpu_a_new256.log`、`gpu_a_old256.log` | 3×3.5M | GPU A 侧 eval 日志（old96 已删） | 均 0 |

> 说明：`/tmp/opencode/*.json` 中 >1M 的 `v4_analysis_data.json`（3.4M）与 `scenarios_train_5k.json`（3.9M）受 `*.json` 保留规则保护，未列入候选。

---

## 四、`du -sh` 前后对比

| 路径 | 清理前 | 清理后 | 释放 |
|---|---|---|---|
| `/tmp/opencode` | 528M | 362M | ≈166M |
| `/tmp` | 4.5G | 3.1G | ≈1.4G |
| `runs/` | 1.9G | 1.9G | 0 |
| `df -h /workspace` | 246G used / 644G avail | 246G used / 644G avail | （显示精度内无变化） |

合计释放 ≈ **1.37 GiB**（删除集合精确计：`/tmp` 1,268,940 KB + `/tmp/opencode` 169,392 KB）。

---

## 五、异常 / 不确定项（如实报告）

1. **自引用口径**：`laneT_smoke`、`p3_micro_test`、`smokeA` 的 grep 命中均为「目录内部 JSON 对自身绝对路径的自引用」，无任何外部文件引用，按「可删」处理并已删除（共 154M，不可恢复）。若要求「任何 grep 命中即移入候选」，则此三项应按候选处理——请知悉。
2. **禁删清单字面匹配异常**：`runs/BTC20260930-1444*` 匹配 0 个目录（该目录不存在）；`runs/BTC20260930-16*_stage_c_v3_*` 字面仅匹配 1608/1625/1645 三个（`1706_*`、`1725_*` 不以 `16` 开头）。因 `runs/` 本次未删任何内容，无实际影响；仅提示禁删清单写法与实际目录不一致。
3. **`/tmp/pip` 非通用缓存**：删除的是单个 torch 2.3.0a0 wheel（1.1G）。若该 wheel 曾作为离线安装来源，后续重装需重新下载/获取。
4. **`/tmp/pytest-of-root` 活跃性**：最新子目录（pytest-532）在删除前约 10 小时内有写入；删除时已确认无 pytest 进程运行，属安全。
5. **范围外陈旧物（未动，供后续轮次参考）**：`/tmp` 下另有 378 个隐藏 `.*.so` 文件合计约 **2.9G**（多为 8~10MB 的重复编译产物），以及 11 个 `*.node-gyp` 目录（4K）；均不在本次清单，未删除。（第二轮期间该批 `.so` 已被**外部并行清理**至 27 个，非本任务操作，见第六节异常 1）
6. 未删除 `/tmp/opencode/__pycache__/`、`specs/`、`night_dth/` 等 <1M 未分类项（候选标准为 >1M）。

---

## 六、第二轮清理（候选 B：外部引用 = 0 项）

- 规则：逐项重新 grep（`*.md/*.json/*.sh/*.py/*.txt`，排除自身子树与本报告文件），有任何外部引用即保留；repo 侧同步只读复核（0 引用）；删除时无 pip/pytest 相关进程
- 删除清单（18 项，合计 79,056 KB ≈ 77.2 MiB）：

| 路径 | 大小 |
|---|---|
| `phase3_smoke_dagger/` | 4.3M |
| `fixb_adv_cli/`、`fixb_lane_cli/` | 2×4.5M |
| `r2_probe_run/`、`r4_probe_run/`、`r4_plan_arm_run/` | 3×4.5M |
| `v8r_e1/`、`v9_e1/`、`v10_e1/`、`v11_e1/`、`v12_e1/` | 5×4.5M |
| `v4_9_base_coef0/`、`v4_9_new_coef0/`、`v4_9_smoke/`、`v4_smoke2/` | 4×4.5M |
| `gpu_a_new1024.log`、`gpu_a_new256.log`、`gpu_a_old256.log` | 3×3.5M |

- **复核中因引用改为保留**：`v4_base/` —— 第一轮按 md/json/sh/py 记 0 引用；本轮扩展 `*.txt` 复核发现 `v4_fingerprint_base.txt` 引用其 `pipeline/trainer.py`（真实溯源引用）→ 保留
- 保留清单（候选 B 其余项，合计 115,556 KB ≈ 113 MiB）：
  - 指令明确保留：`night_forensics/`(15M)、`primary_phase1.pt`(7.2M)、`final_g2.pt`(8.6M)
  - 外部引用 ≥1 保留：`oldcode`(1)、`r7_red`(1)、`v4_base`(1)、`fixb_e1_pre`(1)、`fixb_e1_post`(4)、`fixb_e1_post2`(2)、`g1_kl_probe`(5)、`g1_kl_probe_patched`(1)、`g1_obs_probe2/3/4`(各 1)、`r2_scope_all_run`(1)、`r7_noop_probe`(9)、`r7_noop_probe_off`(2)、`s1_off`(3)、`s1_off_fp`(2)、`s1_on`(1)、`s1_noop_probe_rot`(2)、`v4_smoke`(1)
  - 全部 `*.md/*.json/*.sh/*.py/*.txt`、`phase3_diag/`、`rollback/` 未动；`runs/` 未动（候选 A 仍待用户决定）

### du 前后（第二轮）

| 路径 | 本轮前 | 本轮后 | 说明 |
|---|---|---|---|
| `/tmp/opencode` | 362M | 303M（309,620 KB） | 删除集 79,056 KB；期间 v6 训练在 `v6_meas/` 持续写入（18M，动态） |
| `/tmp` | 3.1G | 569M（582,600 KB） | 其中仅约 77M 来自本轮；约 2.4G 为外部并行清理（见下方异常 1） |
| `runs/` | 1.9G | 1.9G | 未动 |

### 第二轮异常 / 备注

1. **外部并行清理（非本任务操作）**：`/tmp` 顶层隐藏 `.*.so` 从 378 个（2.73 GiB）变为 27 个（253 MiB），发生在第一轮结束与本轮之间；本任务未执行任何 `.so` 删除，`/tmp` 降幅以实测为准。
2. **v6 训练进行中**：`tools/train.py --stage B ... --out /tmp/opencode/v6_meas/stage_b` 正在运行（仅写入 `v6_meas/`，未动）；`/tmp/opencode` 总量为动态值。
3. **本报告自身排除引用检查**：`v6_cleanup_report.md` 提及全部候选名，若纳入 grep 会造成自指伪引用，故检查命令中显式 `--exclude`。
4. **引用检查口径更正**：第一轮仅覆盖 md/json/sh/py，本轮扩展 txt 后检出 `v4_base` 引用；建议后续复核沿用 md/json/sh/py/txt。
5. **repo 并发活动（非本任务）**：终检时 `git status --short` 出现 3 个**未跟踪新文档** `docs/rl_reward_v5.md`、`docs/v6_net_design.md`、`docs/v6_program_prereg.md`（11:53–11:54 由并行 v6 工作创建）；本任务未修改任何受控文件（第一轮结束时 `git status` = 0）。`/tmp` 数值因并发 v6 活动有波动（终测 577M）。

**累计（两轮）**：删除 48 条路径，删除集合合计 1,517,388 KB ≈ **1.45 GiB**；`runs/` 与 repo 全程未动。

---

## 七、第三轮：历史 runs 清理 + /tmp 收尾（2026-10-01，本任务）

- 范围：`runs/` 2026-09-27~29 历史目录；`/tmp` 过期项收尾
- 原则：repo 引用感知（`docs/*.md`、`README.md`、`runs/_refs_*/README.md`、`config/*.yaml`，另加全 repo 扩展核查）→ 有引用一律 KEEP；先 dry-run 后删除；逐项精确路径 `rm -rf`；repo 受控文件未动

### A. runs/ dry-run 与删除（30 项，全部精确路径）

| 待删目录（runs/） | 大小 | 日期 |
|---|---|---|
| `BTC20260928-2031_train` |  KB |  |
| `BTC20260928-2048_eval500_p1` |  KB |  |
| `BTC20260928-2055_eval500_p2` |  KB |  |
| `BTC20260928-2139_debug_viz` |  KB |  |
| `BTC20260928-2152_debug_viz_spec23` |  KB |  |
| `BTC20260928-2255_eval500_ctrlA20` |  KB |  |
| `BTC20260928-2310_eval500_ctrlA20_p1` |  KB |  |
| `BTC20260928-232458_eval500_old_g2` |  KB |  |
| `BTC20260928-2331_ph2_oldbase` |  KB |  |
| `BTC20260928-233223_eval500_old_p1_re` |  KB |  |
| `BTC20260929-000400_eval500_ph2oldbase` |  KB |  |
| `BTC20260929-001308_eval500_ctrlA20_p1_moeoff` |  KB |  |
| `BTC20260929-0042_fixA_moeon` |  KB |  |
| `BTC20260929-030817_eval500_W2p1` |  KB |  |
| `BTC20260929-031037_eval500_W2p2` |  KB |  |
| `BTC20260929-0317_p1moe` |  KB |  |
| `BTC20260929-041111_eval500_T2p1` |  KB |  |
| `BTC20260929-041842_eval500_T2p2` |  KB |  |
| `BTC20260929-0910_vold_smoke` |  KB |  |
| `BTC20260929-1015_vold` |  KB |  |
| `BTC20260929-1045_eval500_phase2` |  KB |  |
| `BTC20260929-1046_eval500_phase2` |  KB |  |
| `BTC20260929-1054_train` |  KB |  |
| `BTC20260929-1103_eval500_phase1` |  KB |  |
| `BTC20260929-1103_eval500_phase2` |  KB |  |
| `BTC20260929-1103_eval500_phase3_r1` |  KB |  |
| `BTC20260929-1103_phase3_loop` |  KB |  |
| `BTC20260929-1103_stageB_phase3_r1` |  KB |  |
| `BTC20260929-1357_phase3_loop` |  KB |  |
| `BTC20260929-1357_train` |  KB |  |

- 删除结果：**30/30 成功，0 失败**；删除集合合计 **418,744 KB ≈ 408.9 MiB**
- `runs/`：1,925,444 KB → **1,506,700 KB**（释放 418,744 KB）
- 09-27~29 目录：58 → 28（全部为 KEEP，差集核对 28/28 在位，missing=0）

### B. 引用核查中「新增 KEEP」（原 KEEP 清单未列，但 repo 有引用 → 保留）

| 目录 | 引用位置 |
|---|---|
| `BTC20260928-1630_train` | docs/version_ledger.md:26；tools/debug_rollout_viz.py:20 |
| `BTC20260928-2201_ctrl_A20` | docs/version_ledger.md:27 |
| `BTC20260929-1105_train` | docs/version_ledger.md:29 |
| `BTC20260929-095624_eval500_L2p1` | docs/phase3_rootcause_analysis.md:249（`{100314,095624}_eval500_L2p{1,2}`） |
| `BTC20260929-1357_eval500_phase3_r1..r5`（5 个） | docs/phase3_rootcause_analysis.md:25,250（`r{k}`/`r{1..5}`） |
| `BTC20260929-1357_stageB_phase3_r1..r5`（5 个） | .slim/deepwork/phase3-rl-base-analysis.md:19（`r{1..5}`） |
| `BTC20260929-191846_stageB_phase3_exp_beta_anchor1_repro` | docs/phase3_rootcause_analysis.md:248；runs/_refs_rlbase/README.md:8 |
| `BTC20260929-195151_eval500_phase3_base` | docs/rl_stage_c_experiments.md:77 |

> 另：原 KEEP 清单中 `runs/BTC20260927-0920_stageA`（盘上不存在；docs/experiments.md:353 已注明 2026-09-28 清理删除）与 `runs/BTC20260929-1357_eval500_phase3_r`（无后缀精确名不存在，实为 `r{k}` 通配前缀，已按 r1..r5 全保留）。

### C. /tmp 收尾

- 删除 **1,171 项**（精确路径，0 失败）：>30 天旧 temp（444×`vscode-remote-containers*.js`、494×陈旧 `.sock`、145×`mcp-*`、36×`pip-*`、13×`devcontainers-*`、10×`.*node-gyp`、4×`pyright-*`、5×`yarn--*`(2024)、`gitkraken/`、`package/`、`.99*.effect`、`dsh-subprocess-*`、`node-addon-*`、12×`tmp*` 目录等）＋ 3 个近期孤儿 temp 目录（`tmpgjlwz8qz`、`tmp3t74hbb1`、`p3g_dbg_au_fr69b`，grep 0 引用）
- 释放 ≈10.6 MiB（du 块计）；`/tmp` 594M → 584M，顶层条目 1417 → 257
- **保留确认**：27 个近期 `.so` ✓（253 MiB 级）；`/tmp/opencode` 全量未动 ✓（脚本/报告/`phase3_diag/`/`rollback/`/`v6_meas` 均在）；非日志用户文件保守保留：`epona_paper_pypdf.txt`、`ulw-*.md`×2、`verify_*.py`×3、`viz_func.txt`
- 未达 30 天阈值的近期 devcontainers/mcp/pyright/socket 等（09-13 及以后）未动，留待下轮

### D. 异常 / 风险（如实报告）

1. **/tmp/opencode 历史引用不构成 KEEP**：30 个被删 runs 目录在 `/tmp/opencode/*.md/json/sh/py/txt` 中各有 4~9 个文件引用（历史脚本/日志里的路径）。按本任务「repo 文档引用」判据已删除；若后续需按旧脚本回看/续跑这些历史 run，路径将失效。
2. **repo 未动**：`git status --short` 仅 3 个并行 v6 工作的未跟踪文档（`docs/rl_reward_v5.md`、`docs/v6_net_design.md`、`docs/v6_program_prereg.md`），非本任务产生；受控文件 0 修改。
3. **隐藏文件引用**：`.slim/deepwork/phase3-rl-base-analysis.md`（未跟踪）含 `1357_stageB_phase3_r{1..5}` 引用，已据此保留；若后续清理 `.slim/` 需注意。
4. **删除前安全检查**：候选目录无符号链接、无嵌套 `.git`、无进程 cwd/打开句柄；/tmp 删除项无挂载点、无活动进程持有。

**第三轮合计**：删除 30 个 runs 目录（408.9 MiB）＋ 1,171 项 /tmp 过期项（≈10.6 MiB）。
