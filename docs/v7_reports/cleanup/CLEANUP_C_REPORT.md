# 锁版清理 C 报告（datasets 精简 + runs 瘦身 + docs 陈旧归档）

> 执行：fixer 子代理；日期：2026-10-06；范围：`/workspace/01_Proj/DRL_PathPlan`（排除 `.venv`）。
> 前置：清理 A/B（`CLEANUP_PLAN.md`、`CLEANUP_A/B_REPORT.md`）；本批为 C（**激进但逐项记录**）。
> 约束遵守：不删**现行复现/证据依赖**项；`env/`、`net/`、`pipeline/`、`tests/` 代码未动；`.gitignore` 未动；临时文件在 `/tmp/opencode`（完成后清）。
> 记录档：`CLEANUP_C_WILL_DELETE.md`（dry-run 快照）、`CLEANUP_C_DELETED.txt`（逐项删除，4858 项）。

## 0. 执行摘要（体积前后）

| 区域 | 前 | 后 | 变化 | 说明 |
|---|---|---|---|---|
| `runs/` | 3,625,662,343 B（377 项） | 1,903,488,549 B（314 项） | **−1,722,173,794 B（−1.604 GiB，−47.5%）** | 删 63 run 目录 + 142 中间 ckpt + logs/monitor |
| `datasets/` | 1,537,847,008 B（20 项） | 1,491,518,212 B（16 项） | **−46,328,796 B（−44.2 MiB，−3.0%）** | 删 4 项 v2/v3 代（详见 §5） |
| `docs/` | 12,192,434 B | 12,803,403 B | +610,969 B | 归档 6 项（移动不删）+ 清理 C 记录/归档 README 更新 |
| **删除总量** | — | — | **1,782,019,175 B（1,699.5 MiB / 1.66 GiB）** | 4858 项（脚本逐项 `os.path.getsize` 求和） |

分项：run 目录 63 项 / 772.9 MB；ckpt 142 文件 / 902.6 MB；monitor 468 文件 / 57.1 MB；datasets 4 项 / 46.3 MB；log 4181 文件 / 3.1 MB。

## 1. 引用复核方法（强制）与判定口径

方法：对每个候选逐项检索 **tracked 全仓**（`git ls-files` + `git grep`），覆盖 `docs/`（含 `docs/v7_reports`、`docs/v6_reports`、`docs/reward_audit`）、`README.md`、`config/`、`tools/`、`pipeline/`、`net/`、`tests/`；补充**花括号族**（`w{1..4}`、`r{1..5}`）、**省略号/通配**（`…`、`*_v7p0_probe*`）与短名人工判读。逐项结果：`CLEANUP_C_WILL_DELETE.md` / `CLEANUP_C_DELETED.txt`。

**保留判定**（"现行依赖"）：

- 最佳线复现链（任务指定）：w1、s11、p4extra、s14、E-β″、E-β′、IDM 基线、v6 P4 关键臂；
- 被**现行程序/证据文档**引用（v6/v7 program_report+prereg、v6_net_design、rl_reward_v5、`docs/v7_reports/*`、`docs/v6_reports/*` 实验报告、`docs/reward_audit/*`、LOCKS.md、version_ledger.md）；
- 被**代码/配置硬引用**（`config/*.yaml`、`tools/` 默认值、`pipeline/`、`net/`、`tests/`）。

**可删判定**：引用仅为 ① 历史叙述/台账（`docs/experiments.md`、`docs/version_ledger.md` 的实验记录行）；② 历史清理记录（`docs/v6_reports/v6_cleanup_report.md`、`docs/v7_reports/cleanup/*`）；③ 非受控 `.slim/*`；④ 本次归档文档（`docs/archive/*`，归档后仅剩历史引用）；⑤ 无任何 tracked 引用。此口径与清理 B 对 v5 线的处理先例一致（"历史叙述，非复现依赖"）。

## 2. `runs/` 删除明细（63 项 / 772.9 MB）

逐项见 `CLEANUP_C_DELETED.txt`；分组：

| 组 | 项数 | 体积 | 代表项 | 引用检查结论 |
|---|---|---|---|---|
| v3/v4 代 A/B 链训练 | 3 | 333.9M | `BTC20260928-1630_train`、`-2201_ctrl_A20`、`BTC20260929-1105_train` | refs=version_ledger（台账行）+ v6_cleanup_report（历史清理记录）+ tools 示例；被 v7p1b/fixA 线取代 |
| v3 phase3 官方 5 轮族 | 10 | 95.3M | `BTC20260929-1357_stageB_phase3_r1..r5`、`…_eval500_phase3_r1..r5` | refs=.slim（非受控）+ phase3_rootcause（本次归档）+ 清理记录；B 曾保留，C 删除（**数据侧 r1–r5 保留**，见 §5） |
| v3 stage_c 臂/诊断 | 6 | 143.3M | `1608/1625/1645/1706/1725_stage_c_v3_*`、`1816_diag_a2rep` | refs=v6_cleanup_report（记录）+ `docs/archive/rl_v3_evidence.sha256`；v3 线已关闭 |
| v3 P3 臂/prep | 12 | 162.4M | `0322..0501_stage_c_p3_a1..a6`、`0541/0543/0554/0617_p4prep*`、`0551_stage_c_p3_holdplan` | refs=rl_stage_c_experiments（本次归档）+ 清理记录 |
| v3 评测/零点/重复 | 22 | 22.6M | `0930-1434/1437/1441/1449`、`1743_posthoc_*`（14）、`1604/2202/2209/095624/100314/225323/195151` | refs=历史台账/归档/清理记录 |
| smoke（纯冒烟） | 6 | 31.3M | `v7p2/v7p4extra/v7s14/v7reward_smoke_u2` | 仅清理记录引用；未参与任何采纳 |
| 其它 | 4 | 4.1M | `091128_v7p1_t1_arm0u200_exact_sub135` 等 | 无现行引用 |

> 说明：**保留**了所有 v7 现行链的中间探针/评测目录（s0/s1/s2/s11/p4extra/s14/reward/p4 臂 u 扫描、v7p1b/i3 筛查、v7sb、p3fix、v7p0 probe/idm 全族）——它们是采纳表的证据源。

## 3. ckpt 瘦身（142 文件 / 902.6 MB）

规则：保留每 run 的 **adopted** ckpt + `final.pt` + `world_model.pt`/`primary.pt`/`bc.pt` + **报告显式路径引用**的 ckpt；删除未采纳/未引用的中间 probe ckpt（`ckpt_u025/u050/…`、`ckpt_epoch*`）。

- 采纳 ckpt 全部在位并 sha256 前缀复核：w1 `fdfe0808…`、s11 `a7cc091f…`、p4e s0u075 `4217abe0…`、s14 s11u025 `229bbc1e…`、v7p1b e010 `d977dec4…`、v6p3 e005 `723db1c2…`（E-β″）、`_refs_rlbase/e_beta_prime/final.pt` `c9e2d31e…`。
- 保留的报告引用 ckpt 例：p4 arm0/arm5 `ckpt_u050/u200`、0004 首跑 `ckpt_u050`（sha `9727b537…`）、struct_v5 `stage_b/ckpt_epoch010`、v7p1i3 `stage_b/ckpt_epoch005/010/015`、p3fix `stage_b/ckpt_epoch001..005`、reward/s14/p4extra 各采纳 u。
- 删除例：s11 `u025..u125,u175,u200`（7）、p4 六臂未采纳 u（约 40）、reward 未采纳 u（约 10）、v6retrain/v7p1b/v7p1i3/fixA/struct_v5 stage A/B 未引用 epoch ckpt。
- 逐文件清单：`CLEANUP_C_DELETED.txt` 的 `## CKPT` 节。

## 4. log / monitor 清理（3.1 MB + 57.1 MB）

- 删除全部 `runs/**/monitor/**`（468 文件 / 57.1 MB：tb events + monitor/metrics.csv 派生物）。
- 删除未引用 `runs/**/logs/*`（4181 文件 / 3.1 MB）。
- **保留（报告引用的证据 log，6 文件）**：`runs/BTC20261001-1631_v6retrain/logs/{stage_a,stage_b}.log`（v6_p3_report）、`runs/BTC20261001-1631_v6p3/logs/stage_b.log`（v6_p3_report）、`runs/BTC20261005-0856_v7struct_v5/logs/{stage_a,stage_b}.log`（v7_struct_retrain_v5）、`runs/BTC20261005-0856_v7struct_v5_p3/logs/stage_b.log`（v7_struct_fail_diag）。证据主体（episodes.csv / metrics.json / manifest / config.snapshot / ckpt）全部保留。

## 5. `datasets/` 删除与保留

**删除（4 项 / 46.3 MB）**：

| 项 | 体积 | 引用检查结论 |
|---|---|---|
| `BTC20260927-1734_expert500val` | 29.7M | refs=experiments.md（v2/v3 台账）+ phase3_rootcause（归档）+ v7_p0_gap_decomposition 附录对照（数值已记录于该文）→ v3 线已关闭，删除 |
| `BTC20260928-1006_dagger_r1` | 5.9M | refs=experiments.md（v2 台账）；v2 线已关闭 |
| `BTC20260928-1109_dagger_r2` | 5.5M | 同上 |
| `BTC20260928-1154_dagger_r3` | 5.2M | 同上 |

**保留（16 项 / 1.42 GiB）及理由**：

| 项 | 体积 | 保留理由（硬依赖） |
|---|---|---|
| `BTC20260926-2343_expert5k` | 297M | `pipeline/trainer.py::DEFAULT_PROBE_BATCH` + `config/train.yaml::probe_batch`（现行 run 的 metrics 记录 probe_batch=此路径）；E-β′ 训练锚（v6_prereg §1 显式 argv+sha）；`tools/make_dagger_pools.py::COVERED_BY` |
| `BTC20260929-1357_phase3_dagger_r5` | 21.7M | **E-β′ 窗口数据**（v6_prereg §1 argv + raw sha） |
| `BTC20260929-1357_phase3_dagger_r1..r4` | 88.4M | E-β′ 泛化扫描 r1–r4 窗口（B 已保留；族完整性） |
| `BTC20261001-1327_expert5k` + `_expert500val` | 330M | v6 线（E-β″）训练数据：v6_prereg §1/§3 + v6_p3_report argv |
| `BTC20261002-0941_expert5k_v4` | 300M | **w1 训练锚**（w1 manifest `--phase3-anchor-bc-dir …_v4`）+ DAgger cycle 锚 |
| `BTC20261002-0941_expert500val_v4` | 30.3M | P1-B 链 val（v7_p1b_chain） |
| `BTC20261002-0941_expert5k_v41` | 300M | 现行最佳线训练数据 + `config/plan_anchors_k6.json`/`net/anchor.py`/`tools/fit_plan_anchors.py` 默认 |
| `BTC20261002-0941_expert500val_v41` | 30.3M | 现行最佳线 val（v7_p1_iter3） |
| `BTC20261002-2329_v7p1dagger_w1..w4` | 92.6M | 现行最佳线 DAgger 窗口（P1–P4/s14 的 BC/探针数据） |

## 6. `docs/` 归档（6 项，移动不删）

`design-v1.2.md`、`dataset_stats.md`、`metrics.md`、`phase3_rootcause_analysis.md`、`rl_stage_c_experiments.md`、`rl_stage_c_v4_report.md` → `docs/archive/`（同名）。映射与 C 批说明：`docs/archive/README.md`。

路径更新（最小机械化，正文未改）：`README.md`（3 处）、`docs/v6_net_design.md`（2）、`docs/rl_reward_v5.md`（2）、`docs/v7_program_prereg.md`（1 链接）、`docs/v7_reports/README.md`（1）→ 改为 `docs/archive/...`；`docs/v6_program_prereg.md` 顶部加 1 行路径注记。

保留原位：v6/v7 program_report + prereg、`v6_net_design.md`、`rl_reward_v5.md`、`experiments.md`、`reward_audit/`、`v7_reports/`、`v6_reports/`、`LOCKS.md`、`version_ledger.md`（被 `v6_program_report.md` 引为上游依据，仍现行）。

## 7. 保留清单（结果态）

- **runs 314 项**：最佳链 w1→w1 e005→s11→p4extra/s14/reward、E-β′（182033/183716/185120/190640/191846/193529 + `_refs_rlbase`）、E-β″（v6p3/v6retrain/v6mini + v6 P4 六臂+seed11+recheck）、IDM 基线（v7p0 idm/probe 全族）、v7p1b/i3、struct_v5 链、`_refs_oldgen`、`reward_audit*`/`reward_viz`（docs/reward_audit 证据）、L2 回退（`BTC20260929-0425_fixA_nold`）及其评测。
- **datasets 16 项**：见 §5。

## 8. 验证

- **测试**：`.venv/bin/python -m pytest -q` → **694 passed**, 145 warnings（exit 0；与清理 B 同基线，代码未动）。
- **git**：本报告提交后 `git status --porcelain` 干净。
- **体积**：见 §0。
- **完整性抽查（sha256 前缀）**：见 §3；全部吻合。

## 9. 与任务口径的偏差（如实报告）

1. **datasets 降幅有限（−3.0%）**：剩余 16 项全部为硬依赖/现行引用（§5）；继续精简需用户裁决，候选见下。任务"删除 v2/v3 代"中，`0926 expert5k` 因**代码默认 + E-β′ 锚 + 现行 probe** 不能删；`1001-1327` 为 v6 线（E-β″）数据，被现行 v6 程序文档引用。
2. **v3 phase3 族（1357 runs）由 B 的"保留"改为"删除"**：B 保留依据 = `.slim`（非受控）+ phase3_rootcause（本次归档）；C 复核为历史归档引用 → 删除并记录；**对应数据 r1–r5 保留**（E-β′ 依赖）。
3. **`experiments.md` / `version_ledger.md` 的引用**按"历史台账叙述"处理（不构成复现依赖），其被引 v2/v3 项（0927/0928、1630/2201/1105）删除；如需恢复需重采/重训。
4. **进一步精简候选（未执行，待用户裁决）**：`0926 expert5k`（297M，需改 probe 默认）、`1001-1327` 5k/val（330M，v6 线已关闭）、`phase3_dagger_r1..r4`（88M，仅扫描证据）、`0941_expert5k_v4`（300M，仅 w1 锚；若接受 v41 替代需重验锚配方）。
