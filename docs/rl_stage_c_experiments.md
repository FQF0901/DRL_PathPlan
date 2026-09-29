# Stage C 版本实验（P3）· 预声明协议与臂表 · 2026-09-30

> 状态：协议**先声明后跑**（本文件在 A1 启动前提交）。结果在文末「运行结果」一节回填；
> 若时间不足按臂序截断，如实标注 `NOT RUN`。依据：G2 复核 §4 放行（P2 通过）+ pin 表
> `/tmp/opencode/rl_perf_optimized.md`。

## 0. 边界（预声明不做的事）

- 不做 plan 臂（`--plan-reference plan`）、不做 Vector 池、不做 BC 锚、不做 H 扫描。
- 不新增周期 ckpt / keep-best / 定期评测选择机制（W2 自监督、keep-best 均未接线，不混入）。
- 不新增 repo `.sh`（链脚本在 `/tmp/opencode/`）；官方库只读（不 pip install）。
- 不刷分：所有臂**同口径、同评测、同报告**；失败臂保留现场并继续其余臂。

## 1. 起跑代码状态（clean SHA）

- 起跑点 `1f58589`（repo clean，V12 性能优化，369 passed）。
- 实验中实际 HEAD = `e19d469` = `1f58589` + D3 守卫（仅 `tools/make_eval_spec.py`：
  默认 DRY-RUN、`--write` 才落盘）。`git diff 1f58589..e19d469 --stat` 仅此一文件，
  **不触及训练/评测路径**。
- 选择协议（= **last-only**）：Stage C 仅保存 `<out>/final.pt`（`pipeline/stages.py:3897`），
  无周期 ckpt、无 keep-best、无按评测选 ckpt 的机制。本实验所有臂的「模型」定义 = `final.pt`。
  **不新增**周期保存点（已确认并声明）。

## 2. Eval 协议（每臂两套都跑，均以 `final.pt`）

| 集 | 路径 | 条数 | 角色 | raw sha256 | canonical content sha256（剔除 `provenance.generated_at`） |
|---|---|---|---|---|---|
| clean500（主） | `/tmp/opencode/phase3_diag/exp/specs_val_only500.json` | 500 | val-only，从未参与任何选择 | `087db3f5c02e9cedccb8313733207f59dfa90162002b0d9bf002e5b80bcc366e` | `d7573673c970d23db3d70280d2dc04decab2a740a58083325b9f536897d1d654` |
| eval500（辅） | `env/specs/scenarios_eval500.json` | 500 | 原选择集（横向可比零点 0.436/0.328） | `98856105eca17461bbdabbf88f203102be7b585fd3de82820b17460358dd4595` | `38512b1bd39f9341a81ca9da970a55dd286d24fac588573d7bba4f248a1ae8bd` |

- 哈希口径：canonical JSON（`sort_keys=True`、紧凑分隔符、UTF-8）**剔除** `provenance.generated_at`
  后 sha256（剔除时间戳差异；同一内容重生成哈希不变）。任务文中引用的「内容哈希前缀 `087db3f5…`」
  实测为 val_only500 的**整文件字节** sha256，本报告两个值都记录。
- eval500 备注：该文件 2026-09-30 03:08 被 `make_eval_spec.py` 误触重生成（raw sha 由
  `acc326e2…` → `98856105…`）；已验证 `specs` 列表与确定性重算 `_stratified(scenarios_val.json, 500, seed=0)`
  **完全一致**（仅 `generated_at` 变），见 `runs/_refs_rlbase/README.md` 口径修正。
- 评测命令（与零点引用同款）：
  `tools/test.py --policy ckpt --ckpt <out>/final.pt --spec <SPEC> --out runs --name <NAME> --workers 6 --tracker lqr --config config/default.yaml`
  （LQR 闭环、deterministic、500 条、`max_steps=1000`）。
- 命名：`BTC<stamp>_eval500_p3<arm>_clean` / `BTC<stamp>_eval500_p3<arm>`（stamp 同该臂训练目录）。
- 配对统计：逐 `(id,seed)` vs **对应 init 的同集 eval episodes**；`fixed`（失败→成功）、
  `broken`（成功→失败）、`net = fixed − broken`、`z = |net| / √(fixed+broken)`。

## 3. 固定 pin 表（所有臂逐字相同，仅「臂变量」列不同）

| pin | 值 |
|---|---|
| 代码 | `e19d469`（训练/评测路径 = `1f58589`） |
| `trainable_scope` | `design` |
| `plan_reference` | `repeat_action` |
| pool | `local` / `envs=1` / tracker `lqr` |
| PPO | `epochs=2` / `minibatch=1024` / clip·vf·ent·γ·λ 取 config 默认 |
| H（`--rollout-steps`） | 256 |
| `--updates` | 200 |
| `--seed` | 0 |
| probe | `datasets/BTC20260926-2343_expert5k`（config `train.probe_batch` 默认） |
| `trim_memory_every` | 4（config 默认） |
| 训练 spec | `env/specs/scenarios_train_slice200.json`（raw sha256 `6df46b26…`；canonical content sha256（剔除 `generated_at`）`6873f4e7…`；200 条；G3 复核口径） |
| `critic_warmup` | 0 |
| 记录 | `--monitor --monitor-legacy-tags`（仅日志，不改变训练数值） |
| 保存 | last-only：`<out>/final.pt` |

初始化权重：

| init | 路径 | sha256 | 零点（clean500 / eval500） |
|---|---|---|---|
| E-β′ | `runs/_refs_rlbase/e_beta_prime/final.pt` | `c9e2d31e4b9d4f784335ffe693b70ddecd4b0ff0e2d4fb062c82aa947f13ea87` | 0.446 / 0.436 |
| L2 | `runs/BTC20260929-0425_fixA_nold/stage_b/final.pt` | `fef0e20e85fa449b881703ca35702edaf877220ee7c2b18a37d9f31885cd7994` | 0.318 / 0.328 |

零点引用（**不重跑**，直接引用既有 episodes）：

| 零点 | clean500 | eval500 |
|---|---|---|
| E-β′ | 0.446（`runs/BTC20260929-224519_eval500_clean_ebeta`） | 0.436（`runs/_refs_rlbase/e_beta_prime/eval_episodes.csv` ← `BTC20260929-183716_…exp_beta_anchor1`，repro 同值） |
| L2 | 0.318（`runs/BTC20260929-225323_eval500_clean_l2`） | 0.328（`runs/BTC20260929-100314_eval500_L2p2`） |

- 已记录噪声级：L2 eval500 另有 `BTC20260929-195151_eval500_phase3_base` = 0.326（±1 条/500）。
  A2 的 eval500 配对 base 预声明取 0.328 的 run（协议数字来源）。

## 4. 臂表（预声明；按优先级顺序执行）

| 臂 | init | lr | KL 锚（initial→final） | 备注 |
|---|---|---|---|---|
| A1 | E-β′ | 3e-4 | 0.05→0 | 主对照（默认配方） |
| A2 | L2 | 3e-4 | 0.05→0 | init 效应 |
| A3 | E-β′ | 1e-4 | 0.05→0 | lr 效应 |
| A4 | E-β′ | 3e-4 | 0.05→**0.01** | KL floor（防漂移） |
| A5 | E-β′ | 3e-4 | **0→0** | 无锚反事实 |
| A6（视时间） | winner 同 init | winner 同 lr | winner 同锚 | **seed=11** 复现（不换 init）；winner = 主集（clean500）配对 net 最高且 z 最高者；若并列取臂序在前者 |

命令模板（每臂实际命令 = 模板 + 臂变量；`<STAMP>` 为臂启动时刻 `%Y%m%d-%H%M`）：

```bash
tools/venv-python tools/train.py --stage C --ckpt <INIT> \
  --spec env/specs/scenarios_train_slice200.json --pool local --envs 1 \
  --trainable-scope design --plan-reference repeat_action \
  --updates 200 --rollout-steps 256 --ppo-epochs 2 --minibatch-size 1024 \
  --seed 0 --critic-warmup-updates 0 --device cuda --monitor --monitor-legacy-tags \
  --out runs/BTC<STAMP>_stage_c_p3_<arm> [--lr ...] [--kl-anchor-coef ...] [--kl-anchor-final-coef ...]
# 双评测（同 final.pt）：
tools/venv-python tools/test.py --policy ckpt --ckpt runs/BTC<STAMP>_stage_c_p3_<arm>/final.pt \
  --spec /tmp/opencode/phase3_diag/exp/specs_val_only500.json \
  --out runs --name BTC<STAMP>_eval500_p3<arm>_clean --workers 6 --tracker lqr --config config/default.yaml
tools/venv-python tools/test.py --policy ckpt --ckpt runs/BTC<STAMP>_stage_c_p3_<arm>/final.pt \
  --spec env/specs/scenarios_eval500.json \
  --out runs --name BTC<STAMP>_eval500_p3<arm> --workers 6 --tracker lqr --config config/default.yaml
```

预算与截断：每臂 ≈8 min 训练 + ≈15 min 双评测 ≈ 23 min；A1–A5 ≈ 2 h；A6 视剩余时间。
任何臂失败 → 保留目录/日志，继续其余臂。

## 5. 判定规则（预声明，不事后改口径）

1. **主判据** = clean500（val-only）配对 `net` / `z`（vs 对应 init）；eval500 为辅，用于与
   零点横向对照（0.446/0.436、0.318/0.328）。
2. 「超过 init」：`net > 0`；「显著超过」：`z ≥ 1.96`。若主集 `net ≤ 0` → 判「未超过 init」，
   即使辅集为正也如实标注不一致。
3. 与零点引用对比记录绝对 Δ（臂 succ − init 零点 succ），但**显著性判定以配对统计为准**。
4. 训练读数（reward/KL/entropy/probe 含低速吸引子 `low_speed_alert`）替代指标（surrogate），
   用于解释成功/失败机制，不单独作为判定依据。
5. 所有臂输出落盘：`/tmp/opencode/rl_p3_<arm>.json`、`/tmp/opencode/rl_p3_<arm>.log`、
   `runs/BTC<STAMP>_stage_c_p3_<arm>/`、两个 eval 目录。

## 6. 运行结果（2026-09-30 03:22 → 05:16，A1–A6 完成）

- 代码状态：训练/评测路径 = `1f58589`（其后提交仅 D3 工具守卫与 docs；A1–A5 运行时 HEAD=`38038f7`），起跑 clean。
- 每臂：`tools/train.py`（manifest 落 `<out>/manifest.txt`）→ `tools/test.py` ×2。日志
  `/tmp/opencode/rl_p3_<arm>.log`，汇总 `/tmp/opencode/rl_p3_<arm>.json`。
- A1 完整命令（其余臂只替换 `--out` 与臂变量；公共段与 §4 模板一致）：
  ```bash
  tools/venv-python tools/train.py --stage C --ckpt runs/_refs_rlbase/e_beta_prime/final.pt \
    --spec env/specs/scenarios_train_slice200.json --pool local --envs 1 \
    --trainable-scope design --plan-reference repeat_action --updates 200 --rollout-steps 256 \
    --ppo-epochs 2 --minibatch-size 1024 --seed 0 --critic-warmup-updates 0 --device cuda \
    --monitor --monitor-legacy-tags --out runs/BTC20260930-0322_stage_c_p3_a1 \
    --lr 3e-4 --kl-anchor-coef 0.05 --kl-anchor-final-coef 0
  ```

### 6.1 臂执行记录

| 臂 | init | 臂变量（相对模板） | train out | 训练 wall | eval clean / eval500 wall |
|---|---|---|---|---|---|
| A1 | E-β′ | `--lr 3e-4 --kl-anchor-coef 0.05 --kl-anchor-final-coef 0` | `runs/BTC20260930-0322_stage_c_p3_a1` | 473 s | 358 s / 357 s |
| A2 | L2 | 同 A1（仅换 init） | `runs/BTC20260930-0342_stage_c_p3_a2` | 503 s | 260 s / 258 s |
| A3 | E-β′ | `--lr 1e-4 --kl-anchor-coef 0.05 --kl-anchor-final-coef 0` | `runs/BTC20260930-0359_stage_c_p3_a3` | 468 s | 354 s / 354 s |
| A4 | E-β′ | `--lr 3e-4 --kl-anchor-coef 0.05 --kl-anchor-final-coef 0.01` | `runs/BTC20260930-0418_stage_c_p3_a4` | 471 s | 478 s / 484 s |
| A5 | E-β′ | `--lr 3e-4 --kl-anchor-coef 0 --kl-anchor-final-coef 0` | `runs/BTC20260930-0442_stage_c_p3_a5` | 481 s | 316 s / 313 s |
| A6 | E-β′ | A4 同参 + `--seed 11`（winner 复现） | `runs/BTC20260930-0501_stage_c_p3_a6` | 478 s | 199 s / 185 s |

eval 目录 = `runs/BTC<stamp>_eval500_p3<arm>_clean` 与 `runs/BTC<stamp>_eval500_p3<arm>`（stamp 同 train out）。

### 6.2 双评测结果 + 逐 (id,seed) 配对（判定主表）

配对 base = 对应 init 的同集 episodes（E-β′：clean `…224519_eval500_clean_ebeta`（0.446）/ eval500
`_refs_rlbase/e_beta_prime`（0.436）；L2：clean `…225323_eval500_clean_l2`（0.318）/ eval500
`BTC20260929-100314_eval500_L2p2`（0.328））。`fixed` = 失败→成功，`broken` = 成功→失败，`net = fixed − broken`。

| 臂 | clean500 succ | Δ vs init | fixed/broken | net | z | eval500 succ | Δ vs init | fixed/broken | net | z | 判定（§5） |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | 0.256 | −0.190 | 2 / 97 | **−95** | 9.55 | 0.244 | −0.192 | 2 / 98 | **−96** | 9.60 | 未超过 init（显著回归） |
| A2 | 0.098 | −0.220 | 6 / 116 | **−110** | 9.96 | 0.122 | −0.206 | 10 / 113 | **−103** | 9.29 | 未超过 init（显著回归） |
| A3 | 0.266 | −0.180 | 1 / 91 | **−90** | 9.38 | 0.260 | −0.176 | 0 / 88 | **−88** | 9.38 | 未超过 init（显著回归） |
| A4 | 0.322 | −0.124 | 9 / 71 | **−62** | 6.93 | 0.330 | −0.106 | 11 / 64 | **−53** | 6.12 | 未超过 init（显著回归，least-bad） |
| A5 | 0.150 | −0.296 | 8 / 156 | **−148** | 11.56 | 0.124 | −0.312 | 4 / 160 | **−156** | 12.18 | 未超过 init（显著回归） |
| A6 | 0.128 | −0.318 | 1 / 160 | **−159** | 12.53 | 0.116 | −0.320 | 1 / 161 | **−160** | 12.57 | 未超过 init（seed 复现失败） |

- 所有臂（A1–A6）在两套评测上**均显著低于对应 init**；两套评测的臂间排序完全一致
  （A4 最好、A6 最差；A1–A5 中 A5 最差），无主辅矛盾。
- 与零点引用的净 Δ（绝对 succ）：A1 −0.190/−0.192、A2 −0.220/−0.206、A3 −0.180/−0.176、
  A4 −0.124/−0.106、A5 −0.296/−0.312、A6 −0.318/−0.320（clean/eval500）。**无臂超过零点**。

### 6.3 训练读数（monitor CSV；reward=`train/returns/mean`，末 20 update 均值）

| 臂 | reward first→last | reward last20 | approx_kl last20 | entropy last20 | probe ds 0–1 m/s | probe ds 1–2 m/s | low_speed_alert | probe logstd | value EV last20 | RSS 峰值 | VRAM 峰值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | −0.82 → +11.70 | 8.711 | 0.0110 | 0.8148 | 1.814 | 2.738 | 128/200 | −1.012 | −0.008 | 2246 MB | 109 MB |
| A2 | +0.58 → +8.42 | 5.518 | 0.0140 | 0.8266 | 0.801 | 1.860 | 200/200 | −1.006 | −0.004 | 2245 MB | 109 MB |
| A3 | −0.82 → +11.80 | 12.089 | 0.0046 | 0.8228 | 0.759 | 1.619 | 200/200 | −1.008 | −0.003 | 2245 MB | 109 MB |
| A4 | −0.82 → +11.10 | 9.288 | 0.0160 | 0.8336 | 0.818 | 1.582 | 200/200 | −1.002 | −0.003 | 2246 MB | 109 MB |
| A5 | −0.82 → +7.86 | 9.475 | 0.0094 | 0.8239 | 1.135 | 2.164 | 200/200 | −1.007 | −0.002 | 2246 MB | 104 MB |
| A6 | −1.17 → +4.56 | 5.211 | 0.0291 | 0.8454 | 1.611 | 2.428 | 164/200 | −0.996 | −0.009 | 2246 MB | 109 MB |

（init 的 probe 记录：update 1 时 ds 0–1 = 0.31 m、ds 1–2 ≈ 0.73–0.79 m，即 **init 本身已在
「低速吸引子」告警区**；trim_memory_calls=51、RSS 峰值 ≈2246 MB 六臂一致，性能口径无回归。）

### 6.4 判定（按 §5 规则，不事后改口径）

1. **是否超过 init**：A1–A6 主集 net 全部为负（z 6.12–12.53）→ 全部判「未超过 init」，且为显著回归。
   辅集（eval500）同向同量级。**是否超过零点的净 Δ**：全为负，无臂可为 stage D 基座替换。
2. **剂量–反应（单变量）**：
   - KL 锚退火到 0 是主要伤害源之一——保留 floor 0.01（A4）把回归从 −95 减到 −62；
   - 完全去锚（A5）伤害最大（−148）：锚的作用在本设定下是**抑制漂移**而非加速学习；
   - lr 1e-4（A3）对比 3e-4（A1）几乎无改善（−90 vs −95）→ 不是步长问题；
   - init L2（A2）比 E-β′（A1）更差（−110 vs −95）→ 与 init 质量一致（E-β′ 更能扛）。
3. **机制读数（surrogate，不作单独判定）**：六臂训练 reward 大幅上升（末值 +4.6…+11.8、末 20 均值
   5.2–12.1）而闭环 succ 崩塌、coll 由 0.04 升到 0.046–0.114（A6 例外：coll 0.046 未升、失败模式转为
   off_road 0.83）、probe 低速档 ds 上升——一致指向**训练奖励与闭环 KPI 错配**（策略被推离保守慢速区，
   换来更多碰撞/出界），且 critic EV≈−0.002…−0.009 未学到价值。「训练没跑起来」不成立，是**训练目标跑偏**。
4. **选型判定**：本批（lr/KL/init 三维）**无臂可采纳**；如必须给一个「损伤最小反事实」= **A4**
   （clean net −62、eval500 net −53）——但该 least-bad 在 seed=11 下不复现（§6.5），故**不可作为选型依据**。
   winner 规则见 §7 第 2 条的澄清。
5. **A6**：winner=A4 的 seed=11 复现 → net −159/−160，见 §6.5。

### 6.5 A6（A4 · seed=11 复现）

- 结果：clean 0.128（Δ−0.318；fixed/broken 1/160，net **−159**，z 12.53）；eval500 0.116（Δ−0.320；
  1/161，net **−160**，z 12.57）→ 同样**显著低于 init**，且比 A4（seed 0，net −62）差得多。
- 训练读数：reward last20 5.21（−1.17→+4.56）、KL 0.0291、entropy 0.8454、probe ds 0–1 = 1.611、
  alert 164/200、logstd −0.996、value EV −0.0085。
- 失败模式与 A4（seed 0）不同：off_road 0.830（init 0.474）、coll 0.046 未升高、rc 0.372。
- 判定：**A4 的「least-bad」不复现**（−62 → −159）⇒ 损伤幅度强 seed 敏感，臂间排序只对 seed=0 成立；
  「无臂可采纳」结论进一步增强（§6.4-4）。

## 7. 异常与偏差记录

1. **执行完整性**：A1–A6 全部 train rc=0、12/12 评测 rc=0，无失败、无保留现场；无数据丢弃。
   A1–A5 链总耗时 03:22:19 → 05:01:07（99 min，与 ≈23 min/臂预算一致）；A6（05:01:49 → 05:16:11，
   ≈14.4 min）单独执行。
2. **winner 规则澄清（预声明文本的偏离）**：§4 原文「net 最高且 z 最高」在全部臂为负 net 时不可同时
   满足（z = |net|/√(fixed+broken) 度量**变化幅度**，最高 z 恰是最差臂 A5）。按协议意图（选损伤最小者）
   取 **clean net 最高 = A4**；未改判定阈值与其他口径。
3. **eval500 spec 重生成**：raw sha256 由 `acc326e2…` → `98856105…`（`generated_at` 差异），
   canonical content sha256 `38512b1b…` 不变；specs 列表已验证与确定性重算一致（§2）。
4. **零点引用噪声**：L2 的 eval500 既有 0.328（pairing base）与 0.326（`…195151_eval500_phase3_base`）
   两次记录（±1 条/500）；A2 全程按 0.328 为 base，结论对该选择不敏感（net −103，两基准差 1 条）。
5. **A5 的 clean eval 耗时偏短（316 s）**，A4 偏长（478 s）：机器负载波动，非评测口径差异（workers=6、
   同一 spec/参数；episodes 数均为 500）。
6. **A6（seed 复现）**：A4·seed 11 显著更差（net −159 vs A4 seed 0 的 −62）⇒ least-bad 不稳健；
   A6 的 KL/entropy 更高而 value EV 更差（−0.0085），提示 seed 改变了优化轨迹（机制不做深挖，如实记录）。
   A6 评测耗时明显偏短（199/185 s）——机器负载波动，同口径（workers=6、500 条）。
7. **边界提示**：A4 仍是 −62 的显著回归；「least-bad ≠ 可采纳」。若做后续（另案），应优先审查训练奖励
   adapter 与闭环 KPI 的对齐（本 lane 未做，不越界）。
