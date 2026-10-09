# env

目的：非侵入式封装 MetaDrive（0.4.3，见 `config/env.yaml:5`），提供场景 spec 消费、观测编码、专家策略与动作执行层。本包不训练网络；训练/评测在 `pipeline/`、`tools/` 中调用。

## 目录结构
| 路径 | 职责 |
| --- | --- |
| `metadrive_env.py` | `build_env(spec) -> SpecMetaDriveEnv`：按 `spec.blocks` 显式建图 + 每次 reset 重放后处理（限速/线型）与脚本行为；可选地图 LRU |
| `obs/` | 观测通道与 6 帧历史（schema v6），详见 `obs/README.md` |
| `expert/` | 规则专家 `PurePursuitIDMPolicy`（KPI 基线 / BC 备选数据源），详见 `expert/README.md` |
| `scenario/` | 场景 spec 生成、实例化校验、运行期脚本行为与逐步标签，详见 `scenario/README.md` |
| `specs/` | 生成的场景 spec JSON（gitignore）与说明，详见 `specs/README.md` |
| `tracking.py` | `(ds, dθ)` 圆弧插值与 Exact/LQR 两套跟踪器 |

## 环境封装（`metadrive_env.py`）
- `build_env(spec, *, traffic_density=None, use_render=False, lru_size=0, seed_pool=None)`（`metadrive_env.py:628`）返回 `SpecMetaDriveEnv`，**不 reset**。
- 建图：`spec.blocks` 直接作为 MetaDrive `map=` 字符串（BIG_BLOCK_SEQUENCE，`metadrive_env.py:667`）；`store_map=False`、`preload_models=False`、自车观测用 `DummyObservation`、lidar/side/line 传感器全部置 0（`metadrive_env.py:662-687`）。
- **每次 `reset()` 都重放后处理**：子类覆写 `reset()`，依次调用 `apply_postprocess()`（逐 lane 写 m/s 限速 + 可选线型覆盖）与 `install_behaviors()`（`metadrive_env.py:509-514, 517-577`）。这是 `store_map=False` 下地图每次重建的补偿路径。
- 自车 spawn：`spec.ego.spawn_lane_index` 存在时关闭随机车道（`random_spawn_lane_index=False`，`metadrive_env.py:671`）；`spawn_velocity` 标量（m/s）转成车体系向量 `[v, 0]`（`metadrive_env.py:151-181`）。
- 地图 LRU：上限 32、默认 0=关（`metadrive_env.py:52-54`）；`MetaDriveWrapper`（`:702`）按 `(blocks, seed/seed_pool, spawn 参数, density)` 复用 env，`seed_pool` 要求池内 blocks 相同（文档化契约、无运行期断言；`:729-753`）。
- 失败可见性：behaviors 安装异常记录在 `env.behavior_install_error`（`:551-577`）；后处理统计在 `env.postprocess_stats`（`:517-549`）。
- `lane_lateral_info(agent)`（`:583`）为 reward 的 `lane_lateral_offset`（回退 `dist_to_left_side`/`dist_to_right_side`）提供 info 键。

## 观测（`obs/`）
- `ObservationBuilder(config).build(env, spec)` 输出定长观测 + 6 帧历史（`obs/builder.py:141`）；`episode_step==0` 时自动 `reset()`（`:143-144`）。
- 当前帧（float32，各带 mask）：ego(1,8)、od(16,9)+`od_id`/`od_presence`、ld(16,7)、nav(1,11)、signal(1,4)、others(1,21+K)、ego_world(1,3)、route_world(64,2)（`obs/builder.py:3-19`；K=12 类 road_class → others=33，`obs/others.py:67-72, 199-201`）。
- 默认 scope：前 150 / 后 50 / 左右 25 m（`obs/builder.py:76`）；OD 槽位 = episode 内 track id（`obs/od.py:87-97`）。
- 历史：6 帧 @0.5 s（每 5 个 env step 1 帧），SE(2) 对齐到**当前** ego 系，`hist_valid` 标记补位帧（`obs/memory.py:1-22`）。
- schema 版本 `OBS_SCHEMA_VERSION=6`；`obs_fingerprint()` 返回 `v6-<内容哈希 12 hex>`（`obs/__init__.py:37-49`），BC 数据加载时比对（`pipeline/trainer.py:2956-2961`）。

## 专家与跟踪
- `expert/pure_pursuit_idm.py`：纯跟踪 + IDM 规则基线，动作 `[steer, throttle] ∈ [-1,1]`，确定性、无随机数（`pure_pursuit_idm.py:133, 320`）；用法见 `expert/README.md`。
- `tracking.py`：`interpolate(actions, dt=0.5, hz=10)` 是 `(ds, dθ) → 10 Hz 圆弧轨迹` 的唯一实现（`tracking.py:72`）；`ExactTracker`（阶段 A/B 运动学执行，`:228`）、`LqrTracker`（阶段 C 闭环，`:356`）、`roundtrip_error`（BC 数据横向保真检查，`:152`）。

## 配置入口
- `config/env.yaml`：`sim.*`（backend/version/store_map/traffic_lights 等，`:2-13`）、`timing.*`（`policy_dt=0.5`、`mpc_dt=0.1`，`:15-19`）、`obs.*`（`history_frames`/`topk_objects`/`topk_lanes`/`scope`，`:21-35`；训练/评测链路**不消费**该段——训练池 `build_pool` 无 obs_config 参数（`pipeline/trainer.py:6031`）、评测侧 `config["env"]` 为空（`pipeline/eval_runner.py:1671`）；全链路用 `ObservationBuilder` 内置默认、值与此镜像（见 `config/README.md` 未消费键表））、`scenario.*`（`:37-58`）。
- `config/train.yaml`：`data.spec=env/specs/scenarios_train.json`（`:51`）、`stages.B.phase3.spec_pool=scenarios_train_5k.json`（`:95`）；`config/eval.yaml::eval.spec=scenarios_eval500.json`（`:3`）。

## 与其它模块的衔接
- `pipeline/vector_env.py:323`、`pipeline/trainer.py:5692`、`pipeline/eval_runner.py:852` 都通过 `ObservationBuilder(obs_config)` 构建观测，经 `build_env` 建环境。
- BC 采集 `tools/collect_expert.py` 使用 `env.tracking.interpolate` 做 round-trip 校验（`collect_expert.py:272-285`），并把 `obs_fingerprint` 写入数据集 meta（`:1632-1633`）。

## 已知文档与代码不一致（以代码为准，未修改）
- LD 采样 offset 代码为 `(5,10,15,20,30)`（`obs/ld.py:76`），但 `obs/schema.py:239`（`schema_manifest()["frame"]["ld"]["semantics"]`，实机仍输出 `{0,20,40,60,80}`）、`obs/schema.py:18-23`（v5/v6 历史段）、`obs/__init__.py:23-25`、`obs/builder.py:25-27` 的文案仍写 `{0,20,40,60,80}`；`obs/ld.py` 内部注释（`:24-26, :256`）也残留旧口径。
