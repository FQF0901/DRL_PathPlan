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
| 训练 spec | `env/specs/scenarios_train_slice200.json`（content sha256 `2090f69e48bf…`，200 条） |
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

## 6. 运行结果（运行后回填）

_TBD_（每臂：命令/耗时/两 eval succ/net-z/训练读数/判定）。

## 7. 异常与偏差记录（运行后回填）

_TBD_。
