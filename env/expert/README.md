# env/expert

目的：专家/参考策略层——KPI 冻结基线 与 BC 冷启动数据源。本目录当前只有一个实现文件 `pure_pursuit_idm.py`；BC 数据源另可用 MetaDrive 内置 `IDMPolicy`。

## PurePursuitIDMPolicy（`pure_pursuit_idm.py:133`）
- 接口：MetaDrive `BasePolicy` 子类，构造 `(control_object, random_seed, config=None, **params)`（`:159-191`）；`act() -> [steer, throttle] ∈ [-1,1]`（`:320`）；`params()` 导出生效参数快照（`:287`）。
- 横向：以自车当前车道为锚，沿 `current_ref_lanes → next_ref_lanes` 路由链采样前视点，纯跟踪律算期望前轮转角；含提前选道/汇入、静态障碍换道、横向偏差修正（`:356-431, 433-487, 726-769`）。
- 纵向：IDM（前车 + 静态障碍并入交互项）+ 限速约束，输出归一化油门/刹车（`:773-831`）。
- 确定性：无随机数、无积分器状态，同状态同参数必然同动作（模块 docstring `:18`）。
- 限速单位：`lane_speed_limit_mps(units="mps"|"auto"|"kmh")` 统一折算，项目默认 `"mps"`（`:97-130, 215-216`）。
- `reset()` 只清诊断；**引擎不会自动调用**，调用方须在每次 env reset 后显式调用（`:278-285`）。

## 内置 IDMPolicy（BC 数据源）
- `tools/collect_expert.py --expert idm`（默认）经 `engine.add_policy(ego.id, IDMPolicy, ego, seed)` 注册（`tools/collect_expert.py:405-411, 476-482, 1709`）；`--expert pure_pursuit` 切换为本仓库专家。
- 采集侧每 5 个 env step（0.5 s）构建一次观测/动作，并注入 `env.prev_policy_action`（`collect_expert.py:151-153, 420-427`）。

## 使用/运行（命令均已核实存在）
- 批量 KPI：`tools/venv-python tools/baseline_eval.py --specs <spec.json> [--workers 4] [--out <json>]`（参数见 `tools/baseline_eval.py:511-536`；`--out` 缺省为 `runs/baseline_eval/<specs 文件名>.json`）。
- 评测入口：`tools/venv-python tools/test.py --policy baseline`（无需 ckpt）或 `--policy ckpt --ckpt <path>`（`tools/test.py:7-8, 53`）。
- BC 采集：`tools/venv-python tools/collect_expert.py --specs ... --out ... --expert {idm,pure_pursuit}`（`tools/collect_expert.py:1705-1709`）。
- 评测判定默认读取基线参照 `runs/baseline_eval/val_reference.json`（`pipeline/eval_runner.py:130`；该文件当前不在仓库磁盘上，由 baseline_eval 产出），可用 `--baseline-ref` 覆盖（`pipeline/eval_runner.py:1587`，经 `tools/test.py` 透传）。

## 与其它模块的衔接
- 观测 static 段与专家静态障碍扫描同净距口径（走廊宽度按 obs 需要有意放宽；`env/obs/static.py:16-17, 140`）；LD 的 `speed_limit` 为原始 m/s 约定（`env/obs/ld.py:38-41`）。
- BC 采集 round-trip 校验优先使用 `env.tracking.interpolate`，缺失时退回模块内 `arc_interpolate`（`collect_expert.py:272-285`）。

## 已知不一致（未修改）
- `tools/collect_expert.py:10` 注释引用 `config/train.yaml::stages.A.bc.expert=idmpolicy`，但 `config/train.yaml` 中不存在该键；实际默认专家由 `collect_expert.py:1709` 的 argparse 决定。
