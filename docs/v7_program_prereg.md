# v7 程序级预注册（IL+RL 闭环性能超越 IDM）

> **状态：冻结（2026-10-02；v7-P0 + Gate A 修正集 #1）**。目标：在冻结评测协议上，**IL+RL 闭环 success 显著超过 IDM 规则基线**。
> 上游：`.slim/deepwork/v7-beat-idm.md`（P0 计划 + 外部研究 §2.5）；工具：`tools/paired_eval.py`（配对 McNemar + bootstrap CI + 多 run 汇总；单测 `tests/test_paired_eval.py`）。
> 关联：[`docs/experiments.md`](experiments.md) §4（IDM 锚）、[`docs/v6_program_prereg.md`](v6_program_prereg.md) §7.3（配对口径）、[`docs/rl_stage_c_experiments.md`](rl_stage_c_experiments.md)（历史 fixed/broken/net/z 记录）。

## 0. 锚点与口径（冻结）

| 对象 | 样本 | success | collision | off-road | 出处 |
|---|---|---|---|---|---|
| **IDM 规则基线（PurePursuitIDM）** | eval500（500） | **0.756** [0.716, 0.792] | 0.144 | 0.068 | `docs/experiments.md` §4（2026-09-27 冻结协议） |
| E-β″（v6 IL 基座） | eval500 | 0.440 | ~0.05 | ~0.50 | v6 P3/E3 |
| arm0（v6 RL 最好，单 seed） | eval500 | 0.622 | 0.154 | 0.192 | v6 P4 |

- 差距基线：best RL 0.622 vs IDM 0.756 ⇒ **−13.4pt**；IL 基座 0.440 ⇒ −31.6pt（P0 解剖对象）。
- **教师天花板（Gate A 解剖）**：tollgate 教师 IDM success = **0.778（评测口径，eval500 35/45）/ 0.644（采集口径，train 池 tollgate 专家）** ⇒ P1 IL/闭环 SFT 的 tollgate 目标不得高于教师上限；超上限须说明来源（RL 自博弈/规划改进），否则先查泄漏/口径。
- **数据纪律**：`expert500val`（`datasets/BTC20261001-1327_expert500val`；旧 v2 同名集同性质）与 eval500 为**同批 (id, seed) 场景**（开环专家数据）⇒ **禁入训练**（任何阶段：IL/SFT/DAgger/RL）；训练前 preflight 断言（`tools/dagger_collect.py::assert_no_eval_val_overlap` 同口径 + `pipeline/stages.py` 守卫）fail-closed，违反即停跑（见 §4）。
- 评测 harness：`pipeline/eval_runner.py`（同 spec / 同 seed / 同零点）；逐 episode 明细 `episodes.csv`（本协议唯一配对输入）。
- 配对口径（与 `docs/v6_program_prereg.md` §7.3 逐字一致）：`fixed` = 基线失败 → agent 成功；`broken` = 基线成功 → agent 失败；`net = fixed − broken`；`z = |net| / √(fixed+broken)`；success = `arrive_dest`；McNemar 精确检验（二项双尾，无连续性校正）；配对差 95% CI = 按场景 bootstrap（`tools/paired_eval.py`，默认 B=10000，固定 seed）。
- 集合划分（**已核验**）：`clean500`（`/tmp/opencode/phase3_diag/exp/specs_val_only500.json`）与 `eval500`（`env/specs/scenarios_eval500.json`，sha256 `98856105eca17461bbdabbf88f203102be7b585fd3de82820b17460358dd4595`）互斥且并集 = `scenarios_val.json`（1000；clean∩eval = 0，clean∪eval = val）。
- 几何标签 = `episodes.csv::primary`；难度 = `difficulty`；分层判定最小 n = 30（与 `eval_runner::per_category_n_min` 同口径，n < 30 只报不判）。

## 1. 目标

- **主目标**：多 seed（**≥5 个独立 run，建议 8–10**）下，agent 与 IDM 在**同 500 测试场景**逐 `(id, seed)` 配对：先过**方差闸**（run sd ≤ 5pt 且 min Δ ≥ −5pt），再看 **success 配对差 95% CI 下界 > 0 且 IQM > 0**（bootstrap，按 run 汇总）；方差闸未过 → 判 **"稳定性未达标，不宣称超越"**（先控方差再谈超越，见 §2.2/§3）。
- 辅助效应量：单 run 配对净胜 `Δ = net/500 ≥ +3pt` 且 `z ≥ 1.96`（v7 计划 §1 判读阈值）；多 run 均值 / sd / 中位数 / IQM / min / max。
- 必须同时报告：collision / off-road / max_step 配对差与 CI；rc / speed_ratio 相对 IDM 锚的变化。
- 反目标（失败判读）：配对差 CI 上界 ≤ 0，或安全闸未过（§5）。

## 2. 判据（冻结；揭盲前不得改）

1. **配对单元**：每个 agent run 与**同一 pinned IDM baseline run（单一 run）**在相同 spec、相同 `(id, seed)` 上配对；键集合必须逐位一致（工具默认 fail-closed；`--allow-mismatch` 仅用于显式记录的交集配对）。**多 baseline 交叉配对 = 伪重复**（同一 baseline 被反复当独立样本）⇒ 工具默认 fail-closed；`--allow-multi-baseline` 仅作基线稳定性检查并标注"非独立"（主判据被阻断，不产生超越结论）。
2. **主判据（MUST；前置 = 单 baseline pin + 方差闸 + n ≥ 5）**：
   - **方差闸**：run 级 **sd ≤ 5pt 且 min Δ ≥ −5pt** 方可进入主判据；否则判 **"稳定性未达标，不宣称超越"**（先控方差，再谈超越）；
   - **n ≥ 5**（建议 **8–10**）的**汇总配对差 95% CI 下界 > 0** 且 **IQM > 0**；汇总按 run 级读数（每个 run 一个配对差）bootstrap（按 run 重采样）；
   - **功效依据（Gate A 实测，2026-10-02）**：跨吸引子 σ≈**15.8pt** 下，n=3/μ=+13.4 功效仅 **0.53**；双峰模型 n=3→**0.35**、n=5 反降 **0.20** ⇒ **"补第 4–5 run"不是功效修复**；n=3 时 IQM≡均值（无稳健化）。**功效修复 = 方差控制**（KL 锚末值 >0/慢衰减、加大 rollout/等效 batch，见 §3），n ≥ 5（建议 8–10）用于加宽证据而非修功效。
3. **单 run 参考判据**：`Δ ≥ +3pt` 且 `z ≥ 1.96`；≥2/3 run 满足视为方向一致（不单独构成结论）。
4. **McNemar 精确 p** 与 z 一并报告（两者口径不同，互为参考）；分层报告：几何（primary）× 难度 与 primary / difficulty 边际。
5. **禁止只报最好 run**：汇总必须含每 run 明细 + 均值 / **sd** / 中位数 / IQM / min / max / CI；任何 run 剔除须预注册理由（失败/污染）并记录。
6. **安全闸**（§5）任一不通过 → 不得宣称"超越"（只能写"性能超越但安全未达标"）。
7. 工具：`tools/paired_eval.py`；报告含 2×2（fixed/broken/both_pass/both_fail）、net、z、McNemar p、配对差 CI、分层、分项与多 run 汇总（JSON + markdown；含方差闸字段 run sd / min Δ / 判定 + 单 baseline 强制）。

### 2.5 评测 pin 表（Gate A 修正集 #1 冻结；揭盲前不得改）

| pin | 值（逐字） | 说明 |
|---|---|---|
| `--policy` | `ckpt`（agent）/ `baseline`（IDM） | — |
| `--spec` | `env/specs/scenarios_eval500.json`（sha256 `98856105…`） | 主判据唯一测试集；每 run 只评一次（§4） |
| `--tracker` | **`lqr`** | 与 v6 P4 终评口径一致（闭环含控制器跟踪误差；baseline policy 不消费 tracker） |
| `--eval-reference` | **`plan`** | 历史/可比口径；**一次性 `repeat_action` 诊断**见下 |
| `--max-steps` | **`1000`**（100 s @ 0.1 s） | 与冻结基线 `runs/BTC20260927-1839_eval500_baseline`（0.756）一致 |
| `--seed`（CLI） | `0` | 固定；配对键 = `episodes.csv` 的 `(id, spec.seed)` |
| baseline | **单一 pinned run**：`runs/BTC20260927-1839_eval500_baseline`（IDM 0.756） | 主判据只允许单 baseline（多 baseline 交叉配对 = 伪重复） |

- **一次性 `repeat_action` 诊断（P1 首个完整 eval 时执行一次）**：同一 ckpt/spec 下 `--eval-reference repeat_action` vs `plan` 的 success 配对差（`tools/paired_eval.py`），量化口径敏感性并归档；若差异显著（z ≥ 1.96）须在结论标注口径依赖，改 pin 须再开 §8 锚。
- **IDM spec-seed 变体（基线稳定性，非主判据）**：同 500 模板、不同 per-scenario `seed` 的 spec 副本（逐变体记录 canonical sha256），各评一次得 success 分布（mean±sd）；**同一 spec 重复评测 = 恒等重复（确定性）**，不得计为独立 run，也不与 agent run 交叉配对。

## 3. 多 seed 协议

- **agent**：**≥5 个独立训练 seed（建议 8–10**；具体 seed 由 P1/P2 门冻结，均以阶段门冻结的 init/配方为起点）；每 run 完整训练 → 在**验证集选点**（§4）→ 测试集评估一次（§4）。
- **IDM 基线**：主判据 **pin 单一 baseline run**（§2.5：`runs/BTC20260927-1839_eval500_baseline`）。基线稳定性用 **spec-seed 变体**：同 500 模板（id/几何不变）、不同 per-scenario `seed` 的 spec 副本，各评一次得 success 分布（mean±sd）；**同一 spec 重复评测 = 恒等重复（确定性）**，不得计为独立 run；变体 run 不进入主判据、不与 agent run 交叉配对（§2.5）。
- 评测确定性（**fix-15 修正 2026-10-02**）：同 `(id, seed)` 评测**并非严格恒等**——全量复评 6 次出现 **1/500 success↔collision 翻转（id 299）+ 1–2/500 数值漂移**；单条 spec × `--workers 1` × fresh 进程 3/6 翻转 ⇒ **内禀逐 episode 熵**（已排除 PYTHONHASHSEED / numpy 全局 RNG / worker 调度 / reset seeding；锁定在 MetaDrive/Panda3D 内部，未到代码行）；量级 ≈ **±0.2pp**。配对协议（逐 `(id,seed)` 对）可吸收该噪声；**不得声称"同 spec 恒等重复"**；同 spec 重复 run 不计独立 run。
- **IDM 现口径复测锚（fix-15）**：eval500 复测与历史锚 **逐条 500/500 一致**（0.756 / coll 0.144 / off 0.068 / rc 0.882 / sr 0.740；历史 manifest `dirty=1` 无实际影响）；**clean500 IDM 锚 = 0.742** [0.702, 0.778]（coll 0.174 / off 0.054 / rc 0.865 / sr 0.757）；spec-seed 变体（+1e5/+2e5）success 0.756/0.706/0.724 ⇒ mean **0.729、sd 2.5pp**（coll sd 2.9pp）；分层 seed 敏感度：hard 14.1pp / uturn 15.6pp / merge 13.0pp / t_intersection 11.1pp（n≈45，easy 4.4pp）⇒ **几何层读数必须携带该噪声量级**；per-id 三 seed 全过仅 268/500 ⇒ 基线必须逐 `(id,seed)` 配对。
- 正向结论门槛：主判据（§2.2）+ ≥2/3 run 单 run 参考判据方向一致；单 run 仅方向性证据。
- **方差控制优先（功效修复；不靠补 run）**：KL 锚末值 > 0（0.01–0.02）/慢衰减、加大 rollout/等效 batch、晚段 EV/entropy 监控（诊断用，非闸）。Gate A 实测双峰模型 n=5 功效反降（0.20）⇒ 补 run 不加功效。
- 预算内复现顺序：**先 5 run**；方差闸未过或 CI 边界 → 先做方差控制（不改判据）再复跑；已过闸但 CI 下界 ≤ +1pt → 可加至 8–10 run 加宽证据（不改变判据，只加密）。

## 4. 选点纪律（防测试集泄漏）

- **选点在独立验证集**：keep-best / early-stop / 超参 / 课程选择只看 `clean500`（val-only，与 eval500 零交集，见 §0）或训练内 holdout；**禁止在 eval500 上 keep-best / 选点 / 调参**。
- **测试集一次评估**：每个最终 run 的 eval500 只评估一次；禁止多次评测择优、看结果调参、按 eval500 选择 ckpt/配方。违反 → 该 run 结论作废并记录污染；下一轮须使用新的留出集（或明确标注为探索性复评，不再具备一次评估的独立性）。
- 训练池与 eval500 零交集（pair 级；`tools/dagger_collect.py::assert_no_eval_val_overlap` 同口径）；spec 文件 hash 冻结。
- **`expert500val` 禁入训练（preflight 断言）**：与 eval500 同批 `(id, seed)` 场景（§0）⇒ 任何训练/SFT/DAgger/RL 池构建禁含该集；构建/开跑前 preflight 断言零重叠（`assert_no_eval_val_overlap` 同口径 + `pipeline/stages.py` 守卫）fail-closed，违反即停跑并记录污染。
- 证据：`runs/` 路径 + `episodes.csv` sha256 + `tools/paired_eval.py` 报告（json/md）；选点记录（验证集读数）与测试读数分开归档。

## 5. 安全闸（辅助，必须报告；Gate A 修正集 #1 量化）

- **collision**：绝对 ≤ **10%** **不可达**（IDM 自身 eval500 = **14.4%**）⇒ 操作性支路 = **相对**：配对 Δ ≤ IDM + **0pt**（绝对等价 ≤ 14.4% 锚）；
- **off-road**：绝对 ≤ **10%** 或配对 Δ ≤ IDM + **2pt**（IDM 锚 6.8%）；
- **speed_ratio**：**≥ 0.9 × IDM**（IDM 锚 0.740 ⇒ ≥ 0.666）；**rc** 相对 IDM 不劣化（报告均值与配对/分组差）；
- **max_step**：报告配对 Δ 与 CI（不设绝对闸）；
- 每项报 2×2 + net + McNemar p + 配对差 CI + run 级分布（mean±sd）；方向：事件项（collision / off-road / max_step）`net > 0` = 事件更多 = 更差（工具逐项标注 direction）。

## 6. 阶段门 A–E 与预算

| 门 | 阶段 | 交付边界 | 门判据（一句话） | 预算 |
|---|---|---|---|---|
| **A** | P0 锚定与差距解剖 | IDM 现口径复测（eval500 + clean500，`--policy baseline`）+ arm0/arm5 差距分解（几何×难度/终止类）+ 本预注册（含 **Gate A 修正集 #1**：方差闸/pin 表/IDM spec-seed 变体条款）+ 配对工具 | 差距可分解、协议可执行（工具单测全绿 + 修正集并入） | ~4–6h |
| **B** | P1 IL 底座 | 新 IL 基座（多轮迭代 keep-best，仅 clean500 选点）+ 评测表 | 验证集 IL success ≥ **0.55–0.60**（不触碰 eval500）+ **≥2–3 seed 或分布/方差读数**（mean±sd / 逐 seed 表） | ~25–35h |
| **C** | P2 RL 超越 | RL 臂批（奖励/算法/池/课程）+ 超 IDM 证据 | **≥2–3 seed** 的 clean500 配对 `Δ ≥ +3pt` 且 `z ≥ 1.96`（或分布/方差读数；单 seed 仅方向性）；eval500 不参与选点 | ~25–35h |
| **D** | P3 巩固与扩量 | **≥5 seed（建议 8–10）**复现 + 数据扩量 + 鲁棒性/回归闸 | **主判据**：eval500 多 seed（≥5）汇总配对差 95% CI 下界 > 0 + **方差闸（run sd ≤5pt 且 min Δ ≥−5pt）** + 安全闸（§5） | ~20–25h |
| **E** | P4 收尾 | 终验（冻结协议全量）、文档、版本 tag、失败模式记录 | 主判据成立 + 证据链完整（含 sha256/报告入库） | ~8–10h |

- 总预算 ≈ **100h**；每门 Oracle 1 初评 + ≤2 重审（沿用 `docs/v6_program_prereg.md` §0 体例）。
- 止损：Gate C 连续 2 个 run 未达单 run 参考判据 → 回退杠杆排序复审（不改判据），不追加同配方 run；**臂内止损用 u50+u100 双点闸 + keep-best**（u50 单点闸已知假阳性 3/4：u50 net −24/−20/−64 → u200 +32/+30/+3；两点均触发才停，保留现场）；Gate D 方差闸未过 → **先做方差控制**（KL 锚末值 >0/慢衰减、加大 rollout/等效 batch），不追加同配方 run；汇总 CI 上界 ≤ 0 → 停线归因。
- 监控：20min 监督；GPU 串行（臂间可比）；长跑 setsid + watcher（task/session 解耦）。

## 7. 证据、复现与变更管理

- 每条结论附：run 路径、`episodes.csv` sha256、工具报告（`runs/` 原始 + 入库副本 MANIFEST 对照）、spec hash、HEAD commit sha；**run 级分布/方差读数（mean/sd/min/max）**与 baseline 稳定性（spec-seed 变体）读数。
- 正向结论须 ≥2/3 run 方向一致 + **主判据（单 baseline pin + n≥5 + 方差闸 + 汇总 CI 下界 > 0 + IQM > 0）**；单 run 仅方向性。
- 阶段交付 focused commit（+里程碑 tag）；结论逐阶段回填本档变更记录与 `docs/experiments.md`。
- 判据/pin 变更须新开 §8 修订锚（内容 commit + 锚 commit 两段式，沿用 v6 §11 体例）。

## 8. 修订锚（沿用 `docs/v6_program_prereg.md` §11 两段式体例）

- v7-P0 建档（本文件 + `tools/paired_eval.py` + `tests/test_paired_eval.py`）内容 commit：`b711e5f`（2026-10-02）。本锚行由第二次小 commit 写入（不改动其余内容）。
- v7-P0 修正集 #1（Gate A：预注册统计修正 + 评测 pin 表 + IDM spec-seed 变体 + Gate B/C 加固 + 安全闸量化 + `expert500val` 禁训 + `paired_eval` 方差闸/单 baseline fail-closed）内容 commit：`0454308`（2026-10-02）。本锚行由第二次小 commit 写入（不改动其余内容）。
- v7-P2 首臂预注册（§9：off-road 距离型奖励 + KL 锚；base = w1 e005；2 seeds；u50+u100 双点止损）内容 commit：`2ddfa22`（2026-10-03）。本锚行由第二次小 commit 写入（不改动其余内容）。
- v7-P2 arm2/arm3 预注册（§10：KL 锚与 off_road_edge 隔离；base = w1 e005；各 2 seeds；u50+u100 双点止损）内容 commit：`a1b6950`（2026-10-05）。本锚行由第二次小 commit 写入（不改动其余内容）。
- v7 结构迭代 A（Lane A）——obs v5：LD 远场 {20,40,60,80} m + 当前车道块 `lane`（1×17）+ TTC 上下文 token `ttc`（1×12，与 nav 同组）+ schema v5（`env/obs`/`net` 令牌接线/`collect_expert` 透传/测试 654 passed/GPU 冒烟；报告 `/tmp/opencode/v7_struct_a_obs_v5.md`）内容 commit：`0291f3a`（2026-10-05）。本锚行由第二次小 commit 写入（不改动其余内容）。
- v7 结构迭代 A2（lane/ttc 训练管线接线：`BCDataset._obs_keys`/`SINGLE_SLOT_CHANNELS`/`RolloutBuffer.DEFAULT_CHANNELS` + 接线测试；选择头可学性探针：K=6 形状锚 × 现有 t0 融合 latent，留出 expert 线性/MLP balanced acc 0.47/0.53 vs chance 0.167，dagger 失败窗口 0.31/0.35；全量 659 passed；报告 `/tmp/opencode/v7_struct_a2_wiring_probe.md`）内容 commit：`fcf047e`（2026-10-05）。本锚行由第二次小 commit 写入（不改动其余内容）。
- v7 结构迭代 B 预注册（§11：K-anchor 计划头——形状锚 K=6 + 连续速度 + WTA/选择 CE + 方案 A 软混合 + 车道系对齐；锚字典 `config/plan_anchors_k6.json`（fix-3 簇大小逐位一致）+ `tools/fit_plan_anchors.py`；单变量开关默认关；Stage B 训练锚头、phase3 specific_only 自动降级、Stage C design 冻结且 PPO 零改；全量 677 passed；GPU 冒烟无 NaN；报告 `/tmp/opencode/v7_struct_b_kanchor.md`）内容 commit：`f42d648`（2026-10-05）。本锚行由第二次小 commit 写入（不改动其余内容）。

## 9. P2 首臂预注册（2026-10-03；off-road 距离型奖励 + KL 锚；w1 起点）

> 本 § 为 P2 首臂（arm1）**臂定义与判据**（经 §8 两段式立项）；P2 其余臂（奖励升档/算法/池/课程）不在本 § 范围，逐臂另行预注册。揭盲前冻结。

### 9.1 基线与臂定义（逐字冻结）

- **base = P1 DAgger w1 e005**（Gate B 覆盖：w1 为 4 窗 clean500 冠军 0.526；w4 降档案/fallback）：
  `runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt`，sha256 `fdfe0808…`（全 sha 见驱动 pin）；
  clean500 参照 run `runs/BTC20261003-045912_v7p1dagger_w1_clean500`（0.526，episodes sha `042b63…`），
  **期望锚 0.47–0.51**（0.526 为 4 窗上尾、sd 6.3pp；不按 0.526 承诺）。
  eval500 参照（**探索性，仅报告**）：`runs/BTC20261003-061634_v7p1dagger_w1_eval500_exploratory` = 0.530。
- **单变量主改 = `off_road_edge`**（BC-SAC 式离路距离型稠密项，`reward_model/terms.py`）：
  `raw = clip(1 + d_edge / edge_scale_m, 0, 2)`，weight **−0.5**、`edge_scale_m` **1.0**（半量级）。
  `d_edge` = 有符号离路距离（负 = 界内，正 = 越界）；ctx 无独立 road-edge 键时以**车道边界等效量**
  `|d_lat| − lane_half_width_m` 替代（口径 = 本 repo `out_of_road` 真实触发面：出车道/压连续实线；
  见 `terms.py::road_edge_distance_from_ctx`；校准见 §9.2）。臂文件
  `config/arms/v7_arm1_offroad.yaml`（= arm0 v5 bundle + 该追加项；终局值 rc=1 逐位不变）。
- **KL 锚（方差控制，§3 首选）**：`--kl-anchor-coef 0.05` → `--kl-anchor-final-coef 0.02`（线性慢衰减；
  末值 > 0）。**BC-SAC 固定 1:8 IL:RL 更新比未实现**（现有 PPO 无交替更新机制；`--bc-anchor` 为 loss
  coef 形式且本臂不启用）——如实记录，不作本臂判据。
- **其余 pins 逐字不动**（v6 P4 arm0 口径，保臂间可比）：`--spec env/specs/scenarios_train_dagger_r1.json`
  (500，sha `d204e803…`)、`--ckpt`(w1 e005)、`--pool local --envs 1 --trainable-scope design
  --plan-reference repeat_action --adv-norm global --updates 200 --rollout-steps 256 --ppo-epochs 2
  --minibatch-size 1024 --critic-warmup-updates 0 --max-episode-steps 200 --ckpt-every 25
  --probe-interval 25 --device cuda`；**池/课程不动**（不扩量、不换池）。
- **seeds = 0 与 11**（2 seeds 分布读数；单 seed 仅方向性，§6 Gate C 口径）。

### 9.2 校准读数（离线；base 策略在 val 子集 24 条，非测试集）

- 24 ep（clean150 前 24，w1 e005，LQR/plan 采集 ctx）：`d_edge` 均值 **−1.007 m**、**46% 步**在界内
  1 m 内（触发率）；success 回合 raw 累计均值 ≈ **26**（@scale 1.0）；oob 触发步 `on_white_continuous_line=1`。
- 触发物理：`out_of_road` = 车体（chassis）接触连续实线碰撞体（`on_white_continuous_line`，11/11
  触发步命中），发生在**车中心 clearance ≈ 车半宽（0.8–1.0 m）**——BC-SAC 的 1 m 裕度与触发面
  吻合（等效口径依据）。
- weight 取 **−0.5**（半量级）理由：车道边界代理比 BC-SAC 路缘触发面高频（46% 步触发 vs 路缘罕见），
  半量级补偿；**升档 −1.0（原量级）/降档 −0.25 留作后续臂**（不在本臂）。

### 9.3 判据（本臂；判读口径）

- **主判据（首要）**：`clean500` 配对 Δ（vs w1 e005 clean500，同协议 pinned 单一 baseline；配对工具
  `tools/paired_eval.py`）：
  - **期望带 +8–15pt 且方差可控**（2-seed mean/sd 记录）；
  - 方向：Δ ≥ +3pt 且 z ≥ 1.96 = 单 run 方向正（§2.3）；**2 seeds 方向一致**才报"方向一致"；
  - 反目标：Δ < 0 或训练崩解（u50+u100 双点触发）→ **记录并等编排决策**（不自行改判据/不加跑）。
- **次判据**：`clean500` 配对 Δ vs P1-B e010（0.314；辅助对照）；`eval500` 配对（vs P1-B e010，
  **每 seed 最终采纳 candidate 仅评一次**，禁止选点/择优）。
- **辅助项（无命中条款）**：tg45 success（IDM 0.778 参照）与 T3 S1（9 锚）——奖励/方差实验，
  tollgate 结构问题不在本臂判据内；只记录读数，不设命中门槛（避免多重比较）。
- **安全闸（§5）**：collision / off-road 配对 Δ 与绝对率、rc / speed_ratio、max_step 逐项报告。
- **止损（臂内，§6 双点口径）**：`u50` 与 `u100` **双点** sub150 闸（net < −20 **或** offΔ ≥ +0.10
  vs w1 sub150，任一点触发记一次）；**两点均触发** → early-collapse，跳过 keep-best/终评、保留现场。
  keep-best：8 候选（u25..u200）sub150 → top1–2 全量 clean500 复评 → **全量配对 net 采纳**
  （禁 sub150 直采）。
- **方差**：2 seed mean/sd；sd > 5pt 须标注"方差未达 §2.2 闸口径（n<5，不做主判据/不宣称超越）"。

### 9.4 产物与记录

- run 目录 `runs/BTC*_v7p2_*`（每臂独立）；配对报告 `tools/paired_eval.py`（json+md）；
- 驱动 `/tmp/opencode/v7_p2_driver.py`（不入 repo；fail-closed pin 断言）；报告
  `/tmp/opencode/v7_p2_arm1.md`；status `/tmp/opencode/v7_p2_status.txt`；
- 证据：逐 run `episodes.csv` sha256、spec hash、HEAD commit、ckpt sha。

## 10. P2 arm2/arm3 预注册（2026-10-05；KL 锚与 off_road_edge 隔离；w1 起点）

> 本 § 为 P2 第二/三臂（arm2/arm3）**臂定义与判据**（经 §8 两段式立项）；与 §9 同 base / 同 pins /
> 同判据口径，唯一变化 = 奖励组件取舍，用于**单变量归因**。立项依据 = arm1 seed0 已揭盲读数
> （报告 `/tmp/opencode/v7_p2_arm1.md`）；arm1 seed11 在途，不引用其读数。揭盲前冻结。

### 10.1 立项依据（arm1 seed0 的隔离需求）

- arm1（bundle + `off_road_edge` + KL）seed0 终局：clean500 **0.536** vs w1 配对 **Δ+1.0pp（ns，
  z=0.38）**；collision **+3.6pp（p=0.0153，显著劣化）**；off_road **−5.4pp（p=0.057，改善）**
  ⇒ 净持平，但两个改动（`off_road_edge`、KL 锚）**未隔离**，无法归因。
- 本 § 立项两臂做单变量隔离（同 base / 同 spec / 同 pins；每臂 2 seeds=0/11）：
  - **arm2 = bundle + KL（去 `off_road_edge`）**：与 arm3 差 = KL 锚一项；与 arm1 差 = `off_road_edge` 一项。
  - **arm3 = bundle only（去 `off_road_edge` 与 KL）**：≈ v6 P4 arm0 原配方在强基座（w1 e005）上的对照。

### 10.2 基线与臂定义（逐字冻结）

- **base = P1 DAgger w1 e005**（同 §9.1）：`runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt`，
  sha256 `fdfe080818eb355b388692269454538eed8b5cc318a9eb4e73b206f8cbd688d9`；**经 CLI `--ckpt`
  传入**（臂配置不承载 ckpt）。clean500 参照 run `runs/BTC20261003-045912_v7p1dagger_w1_clean500`
  （0.526，episodes sha `042b63…`）；**期望锚 0.47–0.51**（同 §9.1）。
- **arm2 = `config/arms/v7_arm2_bundle_kl.yaml`**：v5 bundle（rc=1 档；`arm0_bundle_rc1.yaml`
  项集/终局值逐参数一致）+ **KL 锚 0.05→0.02**；**不含 `off_road_edge`**。
- **arm3 = `config/arms/v7_arm3_bundle_only.yaml`**：v5 bundle（rc=1）only；**不含 `off_road_edge`、
  KL 关**。与 `config/arms/arm0_bundle_rc1.yaml` 的差异**仅记录**（文件 `stages.C` payload 与其逐位
  一致）：(1) base = w1 e005（arm0 = v6 P3 stage_b ckpt）；(2) KL 关 = driver 显式
  `--kl-anchor-coef 0`（arm0 = 0.05→0.0 慢衰减；Stage C 默认 0.05 ⇒ 不显式传 0 不算关）。
- **KL 锚（arm2）**：`--kl-anchor-coef 0.05` → `--kl-anchor-final-coef 0.02`（线性慢衰减；同 §9.1）。
  arm3 不启用（显式 `--kl-anchor-coef 0`）。
- **其余 pins 逐字不动**（v6 P4 arm0 口径，保臂间可比；同 §9.1）：
  `--spec env/specs/scenarios_train_dagger_r1.json`(500，sha `d204e803…`)、`--pool local --envs 1
  --trainable-scope design --plan-reference repeat_action --adv-norm global --updates 200
  --rollout-steps 256 --ppo-epochs 2 --minibatch-size 1024 --critic-warmup-updates 0
  --max-episode-steps 200 --ckpt-every 25 --probe-interval 25 --device cuda`；
  池/课程不动（不扩量、不换池）。
- **seeds = 0 与 11**（每臂 2 seeds 分布读数；单 seed 仅方向性，§6 Gate C 口径）。

### 10.3 判据（两臂同口径；揭盲前冻结）

- **主判据（首要）**：各臂 `clean500` 配对 Δ vs w1 e005（同协议 pinned 单一 baseline；配对工具
  `tools/paired_eval.py`；clean500 spec = `/tmp/opencode/phase3_diag/exp/specs_val_only500.json`，
  canon sha `d7573673…`）：
  - 方向：Δ ≥ +3pt 且 z ≥ 1.96 = 单 run 方向正（§2.3）；**2 seeds 方向一致**才报"方向一致"；
  - 预期（记录性）：两臂为**归因臂**，不做"超越 IDM"宣称；首要期望 = 不崩解（u50+u100 双点不触发）
    且 2-seed 分布可读；方向读数按 §2.3 记录，不设新承诺带；
  - 反目标：Δ < 0 或训练崩解（u50+u100 双点触发）→ 记录并等编排决策（不自行改判据/不加跑）。
- **隔离读数（本 § 核心；跨臂描述性）**：**KL 增量 = arm2 − arm3**（固定 bundle、无 `off_road_edge`）；
  **`off_road_edge` 增量 = arm1 − arm2**（固定 bundle + KL；arm1 于 §9 口径）。两两差为跨 run 描述量
  （各臂 2 seeds，无共享配对单元）——只报均值/极差与逐 seed 明细，不作独立显著性宣称（避免伪重复）。
- **次判据**：`clean500` 配对 Δ vs P1-B e010（0.314；辅助对照）；`eval500` 配对（vs P1-B e010，
  **每臂每 seed 最终采纳 candidate 仅评一次**，禁止选点/择优）。
- **辅助项（无命中条款）**：tg45 success（IDM 0.778 参照）与 T3 S1（9 锚）——只记录读数，不设命中
  门槛（避免多重比较）。
- **安全闸（§5）**：collision / off-road 配对 Δ 与绝对率、rc / speed_ratio、max_step 逐项报告。
- **止损（臂内，§6 双点口径）**：`u50` 与 `u100` **双点** sub150 闸（net < −20 **或** offΔ ≥ +0.10
  vs w1 sub150，任一点触发记一次）；**两点均触发** → early-collapse，跳过 keep-best/终评、保留现场。
  keep-best：8 候选（u25..u200）sub150 → top1–2 全量 clean500 复评 → **全量配对 net 采纳**
  （禁 sub150 直采）。
- **方差**：各臂 2 seed mean/sd；sd > 5pt 须标注"方差未达 §2.2 闸口径（n<5，不做主判据/不宣称超越）"。

### 10.4 产物与记录

- 每臂独立 run 目录 `runs/BTC*_v7p2_s{0,11}_arm{2,3}`；配对报告 `tools/paired_eval.py`（json+md）；
- 驱动 `/tmp/opencode/v7_p2_driver*.py`（不入 repo；fail-closed pin 断言：arm2 KL 0.05→0.02、
  arm3 `--kl-anchor-coef 0`、两臂均无 `off_road_edge`）；报告 `/tmp/opencode/v7_p2_arm2.md` /
  `v7_p2_arm3.md`（或合并臂报告）；status 同 §9.4；臂间顺序按 §6 GPU 串行（编排授权后接力）。
- 证据：逐 run `episodes.csv` sha256、spec hash、HEAD commit、ckpt sha。

## 11. 结构迭代 B 预注册（2026-10-05；K-anchor 计划头：形状锚 + 连续速度 + WTA 选择；方案 A）

> 本 § 为结构迭代 B（计划表示层）**臂定义与判据**（经 §8 两段式立项）。范围 = `net/plan_head.py` /
> `net/model.py`（plan 路径）/ `pipeline/trainer.py`+`pipeline/stages.py`（损失接线）/ `config/` /
> `tests/` / `docs/`；**不动 `env/`**。依据：fix-3 可行性报告 `/tmp/opencode/v7_kanchor_feasibility.md`
> 与 fix-5 选择头探针 `/tmp/opencode/v7_struct_a2_wiring_probe.md`（探针报告不入 repo，读数引用如下）。
> 揭盲前冻结。

### 11.1 立项依据（fix-3 / fix-5 读数）

- **形状锚可分、速度不宜进锚**：expert `cumdtheta` K=6 silhouette **0.72–0.77**，簇 × nav 命令
  Cramér's V=0.57；expert 路径 95.4% 方差在 PC1（速度），首步动作对后续 dθ 的 R² ≤ 0.095
  ⇒ **因子化 = 形状锚（dθ 剖面）+ 连续速度头（ds）**。
- **失败窗口存在模式级偏移**：tollgate 失败窗口 64% / merge 43% 落在 C2"左转回正"（expert tollgate
  仅 4%）；锚分布 JSD tollgate 0.32 / merge 0.29（curve 0.10 = 非模式问题，锚预期收益低）。
- **选择头可学**：留出 expert 上 latent 线性/MLP 探针 balanced acc 0.473/0.528（chance 0.167，
  多数类 macro-F1≈0.14）；失败窗口 C1/C2 召回 0.72/0.69（乐观口径）。
- **PPO 兼容（方案 A）**：soft-mixture 锚 + 连续残差保持 2 维高斯动作空间 → `sample_action` /
  `logprob_from_action` / entropy / KL 零改动；tracker `plan (6,2)` 接口不变。

### 11.2 臂定义（逐字冻结）

- **锚字典（K=6）**：`tools/fit_plan_anchors.py` 拟合（`datasets/BTC20261002-0941_expert5k_v41`；
  `train_weight>0` 且 action 链有限；moving = 6 步积分路径 ≥1 m；120k 子样 seed=0；
  `cumdtheta` KMeans `n_init=10, random_state=0`）。簇大小 **`[92519, 7823, 3012, 8461, 4515, 3670]`
  与 fix-3 逐位一致**；产物 `config/plan_anchors_k6.json`（sha256
  `79829ef705285c79a60252e9db6ac58d1b1e48830a4fe4d32327a1fa0f3873bf`）；文件缺失回退内置默认
  （同源，截断精度 ≤5e-4）。
- **模型开关**：`config/model.yaml::plan_anchor.enabled=true`（`num_anchors=6`、`path`、
  `temperature=1.0`、`hard=false`）。**关闭（默认 false）= `num_anchors=0` = 无新参数/新输出**，
  旧行为与旧 ckpt 严格加载逐位不变（可回退的单变量开关）。
- **表示与车道系对齐**：锚/残差 dθ 在 **lane 帧**表达；ego 系回投 = lane 帧 dθ + "沿车道跟随"
  剖面（`dθ_0 = Δψ + κ·ds_0`、`dθ_i = κ·ds_i`，`Δψ = lane.heading_err`（dim 1）、
  `κ = lane.curvature`（dim 3）；拟合侧对 `ld` 槽位 0 做 `−κ·5 m` 曲率修正）；
  `valid = lane_mask × near_valid`，车道无效 → 恒等（旧数据兼容）。
- **前向/输出**：`plan[:,0] = action_mu`（输出契约不变）；`plan[:,1:] = Σ_k p_k·(anchor_k + residual_k)`
  （`p = softmax(logits/τ)`；`anchor_k` 的 ds 维由连续速度头承担）；t0 一次算出的锚计划驱动
  rollout 尾段（`traj_xy`/`plan` 一致）；额外输出 `anchor_logits/probs/plan/speed/residual/ctx`；
  推理可 `hard=true`（argmax 诊断）。
- **损失（单变量开关 = `anchor_ce_weight`/`anchor_wta_weight`，默认 0 = 旧行为逐位不变）**：
  - **Stage B（BC）**：`stages.B.bc.anchor_ce_weight=1.0` + `anchor_wta_weight=1.0`（其余 BC 超参/
    数据/pins 不动）。WTA 分配 = lane 帧累积 dθ 最近锚；CE = `CE(logits/τ, index)`；回归 = 被分配锚的
    `(anchor+residual)` 链 vs 教师链（逐 step mask；NaN 尾 mask）；连续速度头与残差头同梯度。
  - **phase3**：`freeze=specific_only`（默认）下锚头随 plan head 主干冻结 →
    `anchor_ce/anchor_wta` **自动降级为 0**（`phase3_effective_config`）；本臂 phase3 **不启用**锚损失
    （保持默认 0）——单变量 = 计划表示 + Stage B 目标，避免与"all 解冻必崩"历史配方纠缠。
  - **Stage C**：`trainable_scope=design`（默认）allowlist 不含锚头 → plan head（含锚头）冻结；
    PPO 路径零改（`plan_reference=repeat_action` 训练不消费 plan；评测 `--eval-reference plan`
    消费冻结的锚计划）。
- **pins**：Stage B/C 其余 pins 与现行 v7 流水线一致（数据 v5、评测 §2.5 表）；本臂不改
  spec/池/课程/奖励。

### 11.3 判据（揭盲前冻结）

- **主判据（表示层；训练内/离线，首要）**：
  1. 选择头 CE 收敛：训练末 `bc_anchor_ce_loss ≤ 0.5·ln 6 ≈ 0.896`（初始 ≈ ln 6 = 1.792）；
  2. 分配非坍缩：`bc_anchor_assign_frac_0 < 0.90`（fix-3 straight 77% 参照）且 C1–C5 各 > 0；
  3. WTA 收敛：训练末 `bc_anchor_wta_loss` < 首 epoch 的 50%；`bc_anchor_plan_ade_m` 记录。
- **次判据（闭环，描述性方向）**：`eval500`/`clean500` 配对 vs 基座（w1 e005，同 pins；§2.3 单 run
  口径）。**明确不承诺** tollgate 0/45 修复（fix-3 §6 风险①：锚是表示不是修复，可能同时是跟踪/地图
  几何问题）；tg45 只记录，不设命中条款。
- **安全闸（§5）**：collision / off-road / rc / speed_ratio 配对报告。
- **止损（§9 口径）**：u50+u100 双点 + keep-best；反目标 = CE 不降 / 分配坍缩 / 闭环显著劣化
  （Δ < 0 且 z ≥ 1.96）→ 记录并回退（单变量开关可关回旧行为）。
- **口径说明**：主判据阈值按 Stage B 训练分布（expert 5k + 现行 BC 流程）预设；若实际分布更杂
  （如混入大量失败窗口），只记录实际值与 acc，由编排按 §8 修订锚复审，**不自行改判据**。

### 11.4 产物与记录

- 锚文件 `config/plan_anchors_k6.json` + 拟合脚本 `tools/fit_plan_anchors.py`（可复现）；
- 测试 `tests/test_plan_anchor.py`（锚加载/WTA/软混合几何/车道往返/模型集成/损失接线/PPO 兼容）；
- 报告 `/tmp/opencode/v7_struct_b_kanchor.md`（设计/改动/测试/GPU 冒烟/未决）。

### 11.5 修订锚（fix-11：phase3 安全配方重跑；2026-10-05）

> 依据根因诊断 `/tmp/opencode/v7_struct_fail_diag.md`（决定性证据）：v5+K-anchor 链 phase3
> 崩 0.0 的直接原因 = ① **phase3 训练的 specific experts/router（决定性破坏项）**：权重交换
> `p3+b10锚头 --moe-off` = 0.260（与 b10 逐 spec 一致 150/150）、MoE 开 = 0.013；②
> **`_SPECIFIC_PHASE_FREEZE` 漏 4 个锚头前缀 → 锚头被误训（叠加项）**：`b10+p3锚头` = 0.0。
> 契约修复（内容 commit `55adf90`）= 锚头加入 `specific_only` 冻结清单（实现与 §11.2
> "锚头随 plan head 主干冻结"一致）；并新增 `freeze=trunk_only` 安全配方。揭盲前冻结本修订。

- **冻结配方（重跑单组变更）**：`--phase3-freeze trunk_only` = 只训**共享主干**
  （encoders/mem_encoder/plan_head fusion/norm/ego_next/primary/policy）；冻结 WM(st_gnn)/value +
  specific experts/router/residual_scale（决定性项）+ K-anchor 锚头（叠加项）。损失 =
  action 1.0 + action_chain 0.2 + **anchor_ce 1.0 + anchor_wta 1.0（保留，梯度经冻结锚头塑形
  共享特征）** + load_balance 0.01；WM/ego_next 上游监督保守降级为 0（与失败轮口径一致）。
  锚行 `mild_weight=1.0`（与失败轮同值；不引入第二个变量）。
- **契约修复（实现层）**：`_SPECIFIC_PHASE_FREEZE` 加入 `plan_head.anchor_head.`/
  `plan_head.speed_head.`/`plan_head.residual_head.`/`plan_head.anchor_embed`；测试断言
  "phase3 后锚头权重逐位不变（两 ckpt 比对）"（`tests/test_phase3_safe_recipe.py`）。
- **epoch 级 clean150 守护**（新 CLI `--phase3-guard-spec/--phase3-guard-config/
  --phase3-guard-workers/--phase3-guard-min-success`）：每 epoch 末评测 clean150
  （`/tmp/opencode/phase3_diag/exp/specs_val_only150.json`，sha256 `81f0f958…`；LQR +
  `--eval-reference plan`）；`overall.success_rate < 0.13`（= 入口 b10 clean150 0.260 的 50%）
  或评测不可得 → 立即中止（fail-closed；`guard/guard.json` 逐 epoch 留档）。
- **重跑 pin**：起点 `runs/BTC20261005-0856_v7struct_v5/stage_b/ckpt_epoch010.pt`
  （sha256 `2f463a2186e4…`）；窗口 `datasets/BTC20261005-0856_phase3_dagger_v5`（20,080 行）+
  锚 `datasets/BTC20261005-0814_expert5k_v5`（360,572 行）；5 epoch、micro 256、`ckpt_every=1`；
  epoch 训练段墙钟 > 270s → abort（200s × 1.35 同口径）。评测 config =
  `config/arms/v7_struct_b_v5_eval.yaml`（plan_anchor.enabled=true，锚头 174/174 载入）。
- **评测（keep-best 口径）**：候选 = 入口 b10（clean150 0.260）+ 各 epoch 守护读数；按
  clean150 取最优（平手取更早/入口）→ clean500 / tg45 / T3 / eval500（**仅最终候选一次**）+
  配对（vs w1 0.526/0.530、arm1-s11 0.668/0.646、IDM 0.742/0.756、P1-B 0.314/0.312；tg45 vs
  IDM/P1-B）。**不承诺**任何数值增益；本修订只修崩塌、验证 0.26 可保持性。

## 12. P4-extra 预注册（2026-10-05；s11 + 碰撞抑制单变量；收尾后定向尝试）

> 本 § 为 v7 收尾（`v7-close-20261006`；报告 `v7_program_report.md`）后的 **P4-extra 定向尝试**：
> 在程序最佳 RL 产物 arm1-s11 u150（clean500 0.668 / eval500 0.646）上以**单变量**碰撞抑制
> 修复安全闸（§5：collision ≤ IDM）。范围 = `config/arms/v7_arm1_collision_suppress.yaml`（新）+
> `docs/`；**不碰 net/env/reward 代码语义**（唯一改动 = 臂配置中一个终局值）。揭盲前冻结。

### 12.1 立项依据

- **s11 安全闸缺口（v7 报告 §2.2/§2.4）**：collision clean500 **0.182**（IDM 锚 0.174，+0.8pp）/
  eval500 **0.202**（IDM 锚 0.144，+5.8pp）；success 0.668/0.646 为程序最佳。目标 = 保持 success
  （≥ s11 −3pp）前提下把 collision 压回闸内（判据见 §12.3）。
- **设计参照 = v6 §7.5 候选 A（arm8-A）**：碰撞终局罚 `aggregation.terminal_values.collision`
  −22 → −32（Δ=+10）。机理：碰撞总罚（`crash` −10 + terminal −22 = −32）相对出界总罚
  （`out_of_road` −8 + terminal −14 = −22）的分离由 10 → 20，压低低成功率"冒险救援"的边际收益
  （v6 口径盈亏门槛 p* 9.0% → 20.2%；v6 诊断 `/tmp/opencode/v6_collision_arm_design.md`）。
  v7 arm1 沿用同档终局值（−22），故该单变量可直接平移（v7 报告 §3.2："v6 arm8 立项逻辑在 v7
  未落地"）；v6 arm8-A 本身未跑（v6 收尾），本臂是其**在 v7 最佳产物上的落地尝试**。
- **风险（如实）**：s11 的碰撞画像未做 v6 式"救援撞 vs 成功转撞"分解——若碰撞主因不是冒险救援，
  本臂可能只回吐 success 而不降碰撞。KL 锚（0.05→0.02）+ 短程（u100）+ keep-best 限制漂移。

### 12.2 臂定义（逐字冻结）

- **起点**：`runs/BTC20261005-0601_v7p2_s11_arm1/ckpt_u150.pt`（sha256
  `a7cc091fcbda670b25c396a43e18dc39abe089053e50aa292b5fc0f19164ba2e`）。
- **配方 = `config/arms/v7_arm1_offroad.yaml` 逐位不变 + 单变量**：terms（rc 1 / speed_ratio 0.4 /
  low_speed −0.2 / comfort_* / solid_line −2 / speed_limit −5 / **crash −10** / **out_of_road −8** /
  off_road_edge −0.5·scale 1.0）、其余终局值（arrive +29 / out_of_road −14 / max_step −46 /
  error −5）、KL 锚 0.05→0.02、λ=0.95、rc=1 均不动；**唯一改动 =
  `aggregation.terminal_values.collision`: −22 → −32**。
- **配置**：`config/arms/v7_arm1_collision_suppress.yaml`；driver fail-closed 断言该单变量与其余
  生效值逐位配对（terminal collision −32，其余同 §9 pin 表）。
- **训练**：2 seeds（0/11）；updates=100（短程；ckpt_every=25 → 候选 u25/50/75/100）；pins 同 §9
  （spec `scenarios_train_dagger_r1.json` 500 / envs 1 / rollout 256 / ppo_epochs 2 /
  minibatch 1024 / max-episode-steps 200 / trainable-scope design / plan-reference repeat_action /
  adv-norm global / critic-warmup 0 / probe-interval 25 / device cuda）。
- **执行环境（pre-v5 worktree；v7 事故①纪律）**：主树已推进 v5（obs schema 与 pre-v5 ckpt 不兼容）
  ⇒ 在 `/tmp/opencode/v7_pre_v5`（@ `2f4450e`，与 s11 同代码期）以主树 venv 绝对解释器 +
  `LD_LIBRARY_PATH` 运行；worktree `runs`/`datasets`/`env/specs` 符号链接主树；**训练/评测期间
  禁改代码**；GPU 串行（≤3000 MiB 且无 train/test 进程才开跑）。

### 12.3 判据（揭盲前冻结）

- **主判据（双读数；clean500 + eval500 各判）**：
  1. **collision ≤ IDM**：clean500 ≤ **0.174**；eval500 ≤ **0.144**；
  2. **success ≥ s11 −3pp**：clean500 ≥ **0.638**（0.668−0.03）；eval500 ≥ **0.616**（0.646−0.03）。
  - **PASS = clean500 与 eval500 四条全过**；仅一侧过 → 记 **partial** 并如实报告（不宣称过闸）。
- **配对（描述性）**：vs s11 首要（success net/z/CI95 + collision/off_road delta）；vs w1
  （clean 0.526 / eval 0.530）与 vs IDM（clean 0.742 / eval 0.756）参照；clean500 全量 + eval500
  各配对一次。安全闸辅助：off-road（≤IDM+2pt）、speed_ratio（≥0.9×IDM=0.666）、rc 只报告
  （s11 已知未过，本臂不承诺修复）。
- **辅助无命中条款**：tg45（s11 0.0；IDM 0.778）/ T3（s11 0/9）只记录。
- **止损（u50+u100 双点；v6 §7.5 修订口径）**：每点 u∈{50,100} 在 clean500 子集150（sub150）上
  与 s11 sub150（succ **0.640** / coll **0.200**）配对：
  `trip = (success net < −10) or (collision delta ≥ +0.05)`；
  **两点均 trip → early-collapse 停臂**（保留现场）；单点 trip 继续（v6 u50 单点假阳性 3/4 教训）。
- **keep-best（禁止 sub150 直采）**：候选 u25/50/75/100 按 sub150 排名——① 合格 = sub150 success
  ≥ **0.61**（= s11 sub150 −3pp）；② 合格中 collision 最低优先；③ 平手取 success 高者，再平手取
  更早 update；无合格者 → 取 success 最高并标注 `no-eligible`。top1–2 全量 clean500 复评 →
  **采纳 = 全量 success ≥ 0.638 者中 collision 最低**；无合格者 → 取 success 最高（判据大概率
  fail，如实记录）。采纳 candidate 仅 eval500 一次（§9 纪律）。
- **口径说明**：单 run（每 seed）判读 + 2 seeds 描述性分布（n=2 不设方差闸结论）；任何 PASS 声明
  须以**每 seed 独立**满足主判据为准。

### 12.4 产物与记录

- 臂配置 `config/arms/v7_arm1_collision_suppress.yaml`；预注册本 §（内容+锚两段式）；
- driver `/tmp/opencode/v7_p4extra_driver.py`（不入 repo；fail-closed pin 断言：ckpt sha /
  arm 配置 sha / 单变量生效值 / spec+参照 episodes sha）；报告 `/tmp/opencode/v7_p4extra_collision.md`；
  status `/tmp/opencode/v7_p4extra_status.txt`；逐 run `episodes.csv` sha256 / metrics / ckpt sha。
- 约束：主树 repo 只加 `docs/`/`config/` 变更；worktree 内跑训练；GPU 串行；期间禁改代码。

## 变更记录

- 2026-10-02：建档（v7-P0：评测协议升级——配对 McNemar / bootstrap CI / 多 seed 汇总 / 选点纪律；工具 + 单测 + 本预注册）。
- 2026-10-02：**Gate A 修正集 #1**（揭盲前）：主判据加**方差闸**（run sd ≤5pt 且 min Δ ≥−5pt；未过 = "稳定性未达标，不宣称超越"）+ **n ≥ 5（建议 8–10）** + 功效依据（σ≈15.8pt：n=3 功效 0.53；双峰 n=5 反降 0.20；方差控制才是功效修复）；**评测 pin 表**（`--eval-reference plan` + 一次性 `repeat_action` 诊断 / `--tracker lqr` / `max_steps=1000` / 单 baseline）；**IDM 多 run 改 spec-seed 变体**（同 500 模板、不同 per-scenario seed；同 spec 重复=恒等重复）；Gate B/C 加固（≥2–3 seed 或分布/方差读数）+ **u50+u100 双点止损 + keep-best**（u50 单点假阳性 3/4）；**安全闸量化**（collision 绝对 10% 不可达——IDM 自身 14.4% ⇒ 相对支路；off-road ≤10% 或 ≤IDM+2pt；speed_ratio ≥0.9×IDM）；**教师天花板**（tollgate IDM 0.778 评测 / 0.644 采集）+ `expert500val` 禁入训练（preflight 断言）；`tools/paired_eval.py` 方差闸字段（run sd/min Δ/判定）+ 单 baseline fail-closed；单测更新。
- 2026-10-03：**P2 首臂预注册（§9，揭盲前）**：base = P1 DAgger **w1 e005**（Gate B 覆盖；clean500 0.526、期望锚 0.47–0.51）；主改 = **`off_road_edge`**（BC-SAC 式距离型稠密项，weight −0.5 / scale 1.0；缺 `d_edge` 键以车道边界等效量替代——口径 = `out_of_road` 真实触发面，含 24-ep 校准读数）+ **KL 锚 0.05→0.02**（方差控制）；其余 pins/池/课程不动；判据 = clean500 配对 vs w1 首要、vs P1-B 次，tg45/T3 辅助无命中条款，u50+u100 双点止损 + keep-best，eval500 每 seed 采纳 candidate 一次，2 seeds 分布读数；实现 = `terms.py` + `config/arms/v7_arm1_offroad.yaml` + `tests/test_v7_p2_arm_config.py` + `tests/test_reward_terms.py` 更新。
- 2026-10-05：**P2 arm2/arm3 预注册（§10，揭盲前）**：依据 arm1 seed0（flat：clean500 0.536 / Δ+1.0pp ns；collision +3.6pp 显著）立项**单变量隔离**——arm2 = bundle + KL（去 `off_road_edge`）、arm3 = bundle only（去 `off_road_edge` 与 KL；≈ v6 P4 arm0 原配方在 w1 e005 强基座上的对照，差异仅记录：base/KL 关）；同 base / 同 pins / 同判据口径（clean500 配对 vs w1 首要、vs P1-B 次，tg45/T3 辅助无命中条款，u50+u100 双点止损 + keep-best，eval500 每臂每 seed 采纳 candidate 一次，每臂 2 seeds=0/11；隔离读数 = arm2−arm3 与 arm1−arm2，跨臂描述性）；实现 = `config/arms/v7_arm2_bundle_kl.yaml` + `config/arms/v7_arm3_bundle_only.yaml` + `tests/test_v7_p2_arm_config.py` 更新。
- 2026-10-05：**结构迭代 B 预注册（§11，揭盲前）**：依据 fix-3（形状锚 K=6 sil 0.72–0.77、因子化结论、tollgate/merge 模式偏移 JSD 0.32/0.29）与 fix-5（latent 选择头可学：expert balanced acc 0.47–0.53、失败窗 C1/C2 召回 0.69–0.72）立项 **K-anchor 计划头（方案 A：soft-mixture 锚 + 连续速度 + WTA + 选择 CE；车道系对齐）**；单变量开关（`plan_anchor.enabled` + Stage B `anchor_ce/anchor_wta` 权重，默认 0 = 旧行为逐位不变）；Stage B 训练锚头（CE=WTA=1.0）、phase3 specific_only 自动降级、Stage C design 冻结且 PPO 路径零改；判据 = 表示层三项（CE ≤ 0.5·ln6、分配非坍缩、WTA 收敛）+ 闭环描述性方向 + 安全闸 + 双点止损，明确不承诺 tollgate 0/45 修复；实现 = `net/anchor.py` + `net/plan_head.py`/`net/model.py` plan 路径 + `pipeline/trainer.py`/`stages.py` 损失接线 + `config/model.yaml`/`train.yaml` + `tools/fit_plan_anchors.py` + `config/plan_anchors_k6.json` + `tests/test_plan_anchor.py`。
- 2026-10-05：**§11 修订锚（fix-11：phase3 安全配方重跑；内容 commit `55adf90`，揭盲前）**：依据根因诊断 `/tmp/opencode/v7_struct_fail_diag.md`（specific experts/router 训练 = 决定性破坏项：`p3+b10锚头 --moe-off`=0.260≡b10、MoE 开=0.013；`_SPECIFIC_PHASE_FREEZE` 漏 4 个锚头前缀 → 锚头误训：`b10+p3锚头`=0.0）；**契约修复** = 锚头加入 specific_only 冻结清单（测试断言两 ckpt 锚头权重逐位不变）+ 新 `freeze=trunk_only` 安全配方（只训共享主干；冻结 specific experts/router/residual_scale + 锚头 + WM/value；锚 CE/WTA 1.0/1.0 保留塑形共享特征；WM/ego_next 保守降级）+ epoch 级 clean150 守护（`--phase3-guard-*`，min_success=0.13 = 入口 b10 0.260 的 50%，崩塌/评测不可得即中止 fail-closed）+ keep-best（入口 b10 + 各 epoch 守护读数 → clean500/tg45/T3/eval500 一次 + 配对）。
