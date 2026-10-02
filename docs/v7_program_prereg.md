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

## 变更记录

- 2026-10-02：建档（v7-P0：评测协议升级——配对 McNemar / bootstrap CI / 多 seed 汇总 / 选点纪律；工具 + 单测 + 本预注册）。
- 2026-10-02：**Gate A 修正集 #1**（揭盲前）：主判据加**方差闸**（run sd ≤5pt 且 min Δ ≥−5pt；未过 = "稳定性未达标，不宣称超越"）+ **n ≥ 5（建议 8–10）** + 功效依据（σ≈15.8pt：n=3 功效 0.53；双峰 n=5 反降 0.20；方差控制才是功效修复）；**评测 pin 表**（`--eval-reference plan` + 一次性 `repeat_action` 诊断 / `--tracker lqr` / `max_steps=1000` / 单 baseline）；**IDM 多 run 改 spec-seed 变体**（同 500 模板、不同 per-scenario seed；同 spec 重复=恒等重复）；Gate B/C 加固（≥2–3 seed 或分布/方差读数）+ **u50+u100 双点止损 + keep-best**（u50 单点假阳性 3/4）；**安全闸量化**（collision 绝对 10% 不可达——IDM 自身 14.4% ⇒ 相对支路；off-road ≤10% 或 ≤IDM+2pt；speed_ratio ≥0.9×IDM）；**教师天花板**（tollgate IDM 0.778 评测 / 0.644 采集）+ `expert500val` 禁入训练（preflight 断言）；`tools/paired_eval.py` 方差闸字段（run sd/min Δ/判定）+ 单 baseline fail-closed；单测更新。
- 2026-10-03：**P2 首臂预注册（§9，揭盲前）**：base = P1 DAgger **w1 e005**（Gate B 覆盖；clean500 0.526、期望锚 0.47–0.51）；主改 = **`off_road_edge`**（BC-SAC 式距离型稠密项，weight −0.5 / scale 1.0；缺 `d_edge` 键以车道边界等效量替代——口径 = `out_of_road` 真实触发面，含 24-ep 校准读数）+ **KL 锚 0.05→0.02**（方差控制）；其余 pins/池/课程不动；判据 = clean500 配对 vs w1 首要、vs P1-B 次，tg45/T3 辅助无命中条款，u50+u100 双点止损 + keep-best，eval500 每 seed 采纳 candidate 一次，2 seeds 分布读数；实现 = `terms.py` + `config/arms/v7_arm1_offroad.yaml` + `tests/test_v7_p2_arm_config.py` + `tests/test_reward_terms.py` 更新。
