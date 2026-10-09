# tools

命令行入口与离线工具。统一用 `tools/venv-python` 运行（wrapper 把 venv 本地 glvnd 加进 `LD_LIBRARY_PATH` 后 exec `.venv/bin/python`，`tools/venv-python:6-12`）；裸 `python3` 可能解析到无 metadrive/GL 的解释器。`train.py` / `test.py` import 时只加载 stdlib，重依赖延迟到 `main()`，且**未识别参数原样透传**给下游（`tools/train.py:32,138`、`tools/test.py:23,113`）。`tools/` 同时是 Python 包（`__init__.py` 为空），`dagger_collect.py` 直接复用 `tools.collect_expert` 的实现。

子目录：`diagnostics/`（一次性诊断）、`measure/`（环境实测），各自有 README。

## 入口与编排

| 脚本 | 用途 | 典型调用 |
| --- | --- | --- |
| `run_config.py` | 读 `config/train.yaml` 打印 shell 用 `KEY=value`；路径/派生值唯一来源（auto 解析在 `pipeline/run_paths.py`） | `eval "$(tools/venv-python tools/run_config.py --profile train)"`；`--profile eval --policy ckpt --limit 50 --tracker lqr`；`--profile tb` |
| `train.py` | 分阶段训练入口（校验参数后转发 `pipeline.stages`）：**A**=WM 教师强制、**B**=planner BC、**C**=PPO RL | `tools/venv-python tools/train.py --stage A --bc-dir datasets/BTC<ts>_expert5k --wm-epochs 10 --out runs/train/stage_a`；B 传 `--ckpt runs/train/stage_a/final.pt`；C 传 `--spec ... --envs 2 --updates 20`；迭代恢复 `--phase3 datasets/BTC<ts>_dagger_r1` |
| `test.py` | 冻结 val 集评测入口（转发 `pipeline.eval_runner`；写 `<out>/<name>/metrics.json` + `episodes.csv`） | `tools/venv-python tools/test.py --policy ckpt --ckpt runs/<run>/stage_b/final.pt --limit 50 --workers 2`；`--policy baseline` 不需要 `--ckpt` |
| `train.sh` | 零参训练（run_config 派生路径；setsid+nohup 后台，日志 `<work_dir>/logs/stage_<x>.log`） | `STAGE=A tools/train.sh`；覆盖 `STAGE/RESUME/BC_DIR/WORK_DIR/GPUS/DEVICE/LIMIT_DATASET/EXTRA`；`PHASE3=1` / `PHASE3_ONLY=1` / `PHASE3_ROUNDS=N` 走 phase3 链 |
| `test.sh` | 零参评测（默认 `POLICY=ckpt LIMIT=50 TRACKER=lqr WORKERS=2`，spec=eval500，后台运行） | `tools/test.sh`；覆盖 `CKPT/POLICY/LIMIT/TRACKER/WORKERS/GPUS/DEVICE/SPEC/EXTRA` |
| `tb.sh` | TensorBoard（run 名固定见 `pipeline/run_paths.py::tb_spec`；端口占用自动 +1） | `tools/tb.sh`；`PORT/TB_SPEC` 可覆盖 |
| `gene_env.sh` | 生成场景 spec（转发 `python -m env.scenario.cli`，默认生成后实例化校验） | `bash tools/gene_env.sh --slice 200 --workers 8`；`--n-train/--n-val/--out-dir/--seed-range/--val-seed-range/--rng-seed/--no-validate` |
| `setup_gl_libs.sh` | 把 glvnd runtime 解包到 `.venv/gl`（不改系统；重建 venv 后需重跑） | `bash tools/setup_gl_libs.sh` |
| `venv-python` | 解释器 wrapper（加 glvnd 路径后 exec `.venv/bin/python`） | 其余工具统一以 `tools/venv-python` 启动 |

`train.py` 契约参数：`--config`（默认 `config/default.yaml`）、`--stage {A,B,C}`、`--spec`、`--ckpt`、`--out`（默认 `runs/train`）、`--phase3 <dir>`（固定阶段 B）、`--phase3-loop`、`--phase3-chain`；其余透传 `pipeline.stages`（`--wm-epochs/--bc-epochs/--ckpt-every/--batch-size/--micro-batch-size/--bc-dir/--resume/--device/--limit-dataset/--traj-aux-weight/--weight-sidecar/--critic-warmup-updates/--kl-anchor-coef/--pool/--trainable-scope` 等，完整清单见 `pipeline/stages.py` 的 parser）。

`test.py` 契约参数：`--config`、`--spec`（缺省取 `config/eval.yaml::eval.spec`）、`--ckpt`（`--policy ckpt` 必填）、`--out`（输出根，默认 `runs/eval`）、`--policy {baseline,ckpt}`；透传 `--limit/--workers/--max-steps/--tracker {exact,lqr}/--eval-reference/--device/--seed/--baseline-ref`（eval_runner 默认 workers=2、max-steps=1000、tracker=exact、seed=0，`pipeline/eval_runner.py:1582-1599`）。run 名由 `pipeline.run_paths.py::eval_run_name` 规范化（缺省 `BTC<秒级戳>_<spec kind>_<policy>`；`EXP_foo` → `BTC<戳>_EXP_foo`；规范名原样）。

`run_config.py` 输出键：train=`BC_DIR WORK_DIR STAGE STAGE_DIR LOG DETACH_PID WM_EPOCHS BC_EPOCHS CKPT_EVERY BATCH_SIZE MICRO_BATCH_SIZE TRAJ_AUX_WEIGHT CKPT RESUME`；eval=`NAME WORK_DIR OUT_ROOT LOG DETACH_PID CKPT`；tb=`TB_SPEC`。同名环境变量优先（空串也算显式覆盖，如 `RESUME=` 关闭续跑）。

## 数据构建

| 脚本 | 用途 | 典型调用与产物 |
| --- | --- | --- |
| `collect_expert.py` | 规则专家 BC 采集（schema v2；spawn 并行 + 分片合并） | `tools/venv-python tools/collect_expert.py --specs env/specs/scenarios_train.json --limit 2000 --out datasets/BTC<ts>_expert2k --workers 6`；参数 `--split {all,train,val}`、`--expert {idm,pure_pursuit}`、`--max-steps`(600)、过滤阈值（`--on-lane-frac/--roundtrip-key-mean/--roundtrip-lc-*` 等）、`--balance {weights,cap,none}`、`--workers`(0=auto，上限 10)、`--recycle-every`(150)、`--keep-shards`、`--traffic-density`、`--model-config`。产物 `expert_bc.npz` + `expert_bc.meta.json` + `report.json`（`_shards/` 默认合并后删除） |
| `dagger_collect.py` | DAgger-lite：学生策略 roll-in + 规则专家空问标签（可选失败窗口模式） | `tools/venv-python tools/dagger_collect.py --ckpt runs/<run>/stage_b/final.pt --specs env/specs/scenarios_train_5k.json --out datasets/BTC<ts>_dagger_r1 --workers 4 --window-fail-before 10`；`--labeler {pure_pursuit,idm}`(默认 pure_pursuit)、`--tracker {lqr,exact}`、`--target-fails`(1000)、`--limit`(0=全部)、`--shuffle-seed`、`--fail-terminations`(collision,out_of_road,terminal)、`--window-strict-filters`、`--device`、`--keep-shards`。产物同 collect_expert schema + `dagger.json` + `dagger_specs.json` |
| `make_dagger_pools.py` | 生成 DAgger 采集池 | `tools/venv-python tools/make_dagger_pools.py --mode pool5k`（→ `env/specs/scenarios_train_5k.json`）或 `--mode rounds`（默认，→ `env/specs/scenarios_train_dagger_r{1,2,3}.json`）；`--out` 仅 pool5k 生效。依赖 `datasets/BTC20260926-2343_expert5k/report.json`（代码常量） |
| `make_eval_spec.py` | 生成 eval500 / smoke16 spec；**默认 DRY-RUN 不写盘** | `tools/venv-python tools/make_eval_spec.py --write`（→ `env/specs/scenarios_eval500.json`、`env/specs/scenarios_smoke16.json`）；`--dry-run` 与 `--write` 互斥 |
| `mine_hard.py` | 冻结 primary 的 worst-50% 行权重挖掘 → sidecar npz | `tools/venv-python tools/mine_hard.py --ckpt runs/<run>/stage_b/primary.pt --bc-dir datasets/BTC<ts>_expert5k --out runs/<run>/stage_b/weight_sidecar.npz`；`--hard-frac`(0.5)/`--hard-weight`(1.0)/`--mild-weight`(0.1)；产物供阶段 B `--weight-sidecar` |
| `fit_plan_anchors.py` | 拟合 K-anchor 形状锚字典（离线/CPU，只读数据） | `tools/venv-python tools/fit_plan_anchors.py --dataset datasets/BTC<ts>_expert5k/expert_bc.npz --k 6 --out config/plan_anchors_k6.json`；注意输出是 tracked 配置，覆盖前先确认 |

## 评测与报告

| 脚本 | 用途 | 典型调用与产物 |
| --- | --- | --- |
| `baseline_eval.py` | 规则基线（PurePursuitIDMPolicy）批量评测，输出逐类别 + 总体 KPI JSON | `tools/venv-python tools/baseline_eval.py --specs env/specs/scenarios_val.json --workers 8 --out runs/baseline_eval/val_reference.json`；`--limit/--split/--max-steps`(1000)/`--policy-params`(JSON)/`--speed-limit-units/--render/--debug/--debug-ids`；缺省 out=`runs/baseline_eval/<specs 文件名>.json`（冻结评测协议消费 `val_reference.json`；分组判定使用同目录 `val_reference_by_primary.json`，`pipeline/eval_runner.py:30,1605-1606`） |
| `paired_eval.py` | 同场景配对评测：2×2 翻牌、McNemar 精确 p、配对差 bootstrap CI、分层/分项、多 run 汇总（含方差闸） | `tools/venv-python tools/paired_eval.py --baseline runs/<base>/episodes.csv --agent runs/<a1>/episodes.csv runs/<a2>/episodes.csv --out-dir runs/paired_eval/v7_p0`；要求输入 `(id, seed)` 集合一致（默认 fail-closed，`--allow-mismatch` 取交集）；主判据要求 pin 单一 baseline（`--allow-multi-baseline` 仅标注非独立）；`--bootstrap`(10000)/`--seed`(20261002)/`--alpha`(0.05)/`--min-stratum`(30)。产物 `paired_eval.{md,json}` |
| `reward_audit.py` | P2 奖励审计：快照 / 采集 / 离线重放+终局值反解 / 自检 | `tools/venv-python tools/reward_audit.py snapshot`；`... collect --code-mode {pre,current}`；`... analyze`；`... dry-run`。默认 ckpt=`runs/_refs_rlbase/e_beta_prime/final.pt`、池=`env/specs/scenarios_val.json`、排除=`scenarios_eval500.json`；产物 `runs/reward_audit/report/{reward_audit.md,reward_audit.json,config_draft_rc*.yaml}`（`--mirror-md` 另写） |
| `reward_audit_collect.py` | 单 episode rollout 的逐策略步 ctx 采集器（通常由 `reward_audit.py collect` 调起） | `tools/venv-python tools/reward_audit_collect.py --code-mode current --specs env/specs/scenarios_train_5k.json --spec-id 0 --spec-seed 1000 --out /tmp/opencode/ep.json`；`--code-mode pre` 需 `--pre-root`；产物单 episode JSON |
| `il_report.py` | IL 评审报告：monitor CSV + metrics.json → md/json（缺失项显式标注，不编造） | `tools/venv-python tools/il_report.py --run runs/<run>/stage_b --stage-a runs/<run>/stage_a --eval runs/<eval_run>`；`--eval` 可重复（目录或 json）；产物 `<run>/il_report/il_report.{md,json}`（`--out` 覆盖） |
| `plot_curves.py` | 离线曲线绘图（读 `monitor/metrics.csv` → PNG，不依赖 TensorBoard） | `tools/venv-python tools/plot_curves.py --stage-a runs/<run>/stage_a --stage-b runs/<run>/stage_b --out runs/<run>/plots` |

## 可视化

| 脚本 | 用途 | 典型调用与产物 |
| --- | --- | --- |
| `visualize.py` | 场景俯视图抽检（headless matplotlib，无 GL） | `tools/venv-python tools/visualize.py --specs env/specs/scenarios_train_slice200.json --random 10 --seed 0 --frames 3 --steps 120 --policy baseline --out runs/vis`；选择三选一 `--random N` / `--ids 1,5,9` / `--all-in-file`；产物 `scenario_<id>_<seed>.png`、`contact_sheet.png`、`index.md` |
| `debug_rollout_viz.py` | 失败场景回灌：闭环录制 + 离线 rollout/WM 推演 + 逐帧 GT 对比 | `tools/venv-python tools/debug_rollout_viz.py --ckpt runs/<run>/stage_b/final.pt --spec-id 10 --out runs/BTC<ts>_debug_viz`；`--spec-file`（缺省 config `eval.spec`）、`--device auto`（GPU 空闲显存不足时等待/回退 cpu）、`--max-steps`、`--torch-threads`、`--limit-frames`。产物 `frames/frame_XXX_step_XXXX.png`、`deviations.{json,csv}`、`episode.json`、`README.md` |

## 路径与命名约定

- **训练 run**：`runs/BTC<北京分钟戳>_<name>`（`run_paths.new_work_dir`）；同轮 A/B 共用一个 run 根（B 的 work_dir 取 `stage_a_ckpt` 所在 run），阶段目录 `<run>/stage_{a,b}`，另有 `logs/stage_<x>.log`、`manifest.txt`、`config.snapshot.yaml`、`model.snapshot.yaml`、`detach.pid`。
- **评测 run**：`runs/BTC<秒级戳>_<spec kind>_<policy>`（`run_paths.eval_run_name`，调用方不手拼）；`test.sh` 走 run_config 先给 `BTC<分钟戳>_eval_<tracker><limit>` / `..._eval_baseline<limit>`，已是规范名则原样沿用。
- **ckpt 命名**：阶段 A/B `<out>/final.pt` + 周期 `ckpt_epoch{NNN}.pt`；阶段 C `<out>/final.pt` + 周期 `ckpt_u{NNN}.pt`（`pipeline/stages.py:493-494,539-540`）。
- **BC 数据集 auto**：最新 `datasets/BTC*_expert*`（须含 `expert_bc.npz`，名字含 `val` 的排除，`run_paths.latest_dataset`）；dagger 输出约定 `datasets/BTC<ts>_daggerN`（`tools/dagger_collect.py:1241`）。
- **config 关系**：`config/default.yaml` 主配置（includes 自动合并）；`config/train.yaml::run` 驱动 `run_config.py`；`config/eval.yaml::eval.spec` 是评测缺省 spec；`config/model.yaml` 模型结构（`--model-config` 缺省值，写入 manifest）。
- **gitignore**：`runs/`、`datasets/`、`env/specs/*.json` 均不入库（spec 是生成产物）。

## 注意事项

- **资源纪律**：MetaDrive 每进程只能有一个 engine；任一时刻只跑一个 env-heavy 多进程任务（`pipeline/stages.py:31-32`）。`collect_expert` 每 worker 峰值 ≈1.3–1.5 GB，本机 15 GB 建议 `--workers<=8`；`--workers 0`=auto（CPU 核数取半、上限 `MAX_RECOMMENDED_WORKERS=10`）。评测与训练池互斥，`test.sh` workers 默认 2。
- **spec 隔离**：`dagger_collect.py` 与 `make_dagger_pools.py` 都硬校验训练池 ∩ `scenarios_eval500.json` / `scenarios_val.json` = ∅，命中即 `SystemExit`，不要绕过。
- **device**：训练/评测 `--device auto`（CUDA 可用则 cuda）；`diagnostics/` 的两个 rollout 诊断脚本（`forensics_offline`/`forensics_closed_loop`）默认 `--device cuda`（`forensics_report` 为纯分析、无 device 参数），无卡需显式 `--device cpu`；`debug_rollout_viz` 带显存守卫。
- **写盘守卫**：`make_eval_spec.py` 默认 DRY-RUN，必须显式 `--write` 才落盘；`fit_plan_anchors.py` 会覆盖 tracked 的 `config/plan_anchors_k6.json`。
- `test.py --out` 是输出**根**（实际写 `<out>/<name>/`），不要当 run 名用。
