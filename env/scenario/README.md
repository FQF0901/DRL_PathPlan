# env/scenario

目的：场景 spec 的生成、实例化校验与运行期脚本行为；spec 由 `env/metadrive_env.py::build_env` 消费。

## 文件职责
| 文件 | 职责 |
| --- | --- |
| `taxonomy.py` | 标签常量与 block 映射：几何 12 类（`GEOMETRY_LABELS`，`:41-54`）、可抽样 11 类（排除 `bidirection`，`:62-67`）、交通形态/控制/机动/难度集合（`:75-101`）、block 字符表（`:104-125`）、几何→block 序列（`sequence_for`，`:162`）、配额（`allocate_counts`，`:182`）、名义转向（`turns_for`，`:208`） |
| `spec.py` | `ScenarioSpec` 契约（11 字段，`:66-78`）与 `validate()`（`:128`）；JSON 读写（`save_specs/load_specs`，`:322-349`） |
| `generator.py` | `build_specs(n_train, n_val, train_seeds, val_seeds, rng_seed=0)`（`:464`）：几何分层 + 难度分层 + 事件窗口 + spawn 车道 |
| `behaviors.py` | 运行期脚本 cut-in/cut-out：`install(env, spec)`（`:1074`）、`event_state(env)`（`:765`）、`map_info(env)`（`:224`）与几何工具 |
| `labels.py` | 9 个逐步可观测标签 `LABEL_ORDER`（`:46-56`）与 `compute_step_labels(env, spec)`（`:148`） |
| `validator.py` | 逐条实例化校验 `validate_specs(...)`（`:489`）与报告写出 `write_report`（`:676`） |
| `cli.py` | spec 生成命令行入口 `python -m env.scenario.cli`（`:122`） |

## spec 契约与单位
- 字段：`id/seed/split/blocks/geometry/traffic/limits/nav/ego/difficulty/labels`（`spec.py:66-78`）；落盘包装 `{"schema_version": 1, "count": N, "specs": [...]}`（`:322-335`），`load_specs` 也接受裸列表（`:338-349`）。
- 单位：`limits`、事件 `speed_mps`、`ego.spawn_velocity` 均为 m/s；`trigger_step/duration_steps` 为 10 Hz env step（0.1 s/step）（`spec.py:14-29`）。
- `blocks` 为显式 BIG 序列（不含自动前置的 First block "I"），长度 3–5（`taxonomy.py:127-129`）；`geometry` 经 `sequence_for` 必须等于 `blocks`（`spec.py:158-164`）。
- `ego.spawn_lane_index` 固定首块 road 键 `(">", ">>")`、车道号 ∈ [0,3)（`taxonomy.py:131-139`，`spec.py:266-288`）；`spawn_longitude` ∈ [0,10) m（`:286-288`）。
- 含 `roundabout` 的几何必须记录 `traffic["density_cap"] <= ROUNDABOUT_MAX_DENSITY=0.05`（`spec.py:195-209`，`taxonomy.py:69-72`）。

## 生成器
- 难度档与单条 spec 内部随机流只由 `(rng_seed, seed)` 决定（`generator.py:212-214, 372-376`）；但几何槽位分配依赖 split 总数 n（`:452-455` 长度 n 的槽位 shuffle + `:458-460` zip seeds）→ `--slice` 产物不是全量的前缀子集；train/val seed 区间重叠直接报错（`:477-480`）。
- 可抽样几何每类保底 `min_per_label = 0 if n < 11 else max(1, min(50, n // (2·11)))`（`n<11` 退化分支；`:449-451`）；难度档 round-robin 均衡（`:459`）。
- 难度区间：easy 密度 [0.02,0.06] / medium [0.08,0.14] / hard [0.16,0.28]；事件 gap easy [18,30] / medium [12,18] / hard [6,12]（`:82-91`）；难度以 `_derive_difficulty` 派生值为准（`:322-336`）。
- 事件：easy 仅 40% 概率 cut_out；medium 必有 1 个事件；hard 必有 cut_in 且 50% 追加 1 个（`:262-278`）。
- spawn 车道：首块 3 车道取中间车道；请求侧不可达时就地改写事件 `side`，不消耗 rng（`:281-319`）。

## 运行期行为（`behaviors.py`）
- `install(env, spec)` 幂等挂载 `ScenarioBehaviorManager`（`PRIORITY=11`，`:70-71, 794-801`）；`metadrive_env.py` 在每次 `reset()` 后重放安装（`env/metadrive_env.py:551-577`）。
- 事件窗口 = `[trigger_step, trigger_step+duration_steps)`（10 Hz env step）；缺省窗口按 seed 合成（`behaviors.py:35-44, 370-380`）。
- 脚本车由 `WaypointPolicy` 直写位姿回放（0.4.3 无上游 waypoint_policy，`:461-509`）；机动结束 +0.5 s 后交回 `IDMPolicy`（`POST_WINDOW_S=0.5`，`:85, 419-421, 888-918`）。
- 状态出口：`env._scenario_events`（`:1040-1044`）与 `event_state(env)`，含 `start_step/end_step/fired/degraded` 与 actor 逐步几何 `lateral/long_rel/distance/alive`（`:657-690, 765`）。

## 逐步标签（`labels.py`）
- 9 标签：`in_intersection / near_intersection / cutin_active / cutout_active / crowded / car_following / on_curve / merging / roundabout_near`（`:46-56`）。
- 阈值：near_intersection / roundabout 30 m、merge 25 m、crowded 30 m 内 >2 辆、跟车 ≤20 m（横向 ≤0.75·车道宽）、曲率 >0.01 1/m、cut-in ≤35 m 且横向 ≤0.5·w、cut-out ≤45 m（`:59-73`）；部分阈值可由 `spec.traffic` 覆盖（`:185-187`）。
- 标签只依赖当前步可观测状态，不按 spec 类别直接置位（模块 docstring `:1-5`）。

## 校验与报告（`validator.py`）
- 每条 spec 经 `build_env` 实例化 + 轻量循迹控制器 rollout（`:129-169, 377-469`）；检查项：reset/ego 存在、导航 2 checkpoint + ref lanes、限速已写入、blocks/geometry 对账、事件被安装且触发、生成重叠（模块 docstring `:1-34`）。
- 报告结构：`meta / summary / n_failed / failure_categories / warning_categories / coverage / resample_suggestions / warning_notes / results`（`:655-673`，`warning_notes` 见 `:671`）；`summary` 含 `n_specs/passed/failed/pass_rate`（`:608-616`），`n_failed` 在顶层供 CLI 读取（`:666`）。
- 进程池按"每 worker 100 条 spec"分批回收（`:500-504`）；`workers<=1` 时在当前进程内联执行（`:515-517`）。

## 使用/运行
- 生成：`bash tools/gene_env.sh [--slice [N]] [--n-train N] [--n-val N] [--workers W] [--no-validate]`（`tools/gene_env.sh` 转发到 `env.scenario.cli`；argparse 参数见 `cli.py:34-69`）。
- 默认：train 10000 / val 1000，seed 区间 train 从 1000、val 从 5000000（`cli.py:26-31`）；生成后默认跑实例化校验并写 `validation_<name>.json`（`cli.py:97-119`）。
- 配置：`config/env.yaml::scenario`（sequence/blocks/blocks_per_map/cache/set_speed_limit_after_build/behaviors/spec_dir，`:37-58`）为口径记录；生成器与封装**不直接读该段**，实际生效常量在 `taxonomy.SEQUENCE_MIN/MAX_BLOCKS` 与 `generator.BLOCKS_PER_MAP`。

## 与其它模块的衔接
- `env/metadrive_env.py::build_env` 消费 `spec.blocks/limits/traffic/ego`；`behaviors.install` 在每次 reset 后重放。
- 训练/评测按 `config/train.yaml::data.spec`、`config/eval.yaml::eval.spec` 选择 spec；文件清单与再生成命令见 `specs/README.md`。
