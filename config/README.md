# config

集中声明环境、模型、训练与评估参数（YAML，键带中文注释）。本文件说明**文件职责、加载关系与代码消费方**；
某个键是否真正生效，以代码读取点为准（速查见文末「未被代码消费的键」）。

## 加载关系

`default.yaml` 的 `includes` 顺序 = env → model → train → eval（`config/default.yaml:7-11`），
同名顶层键后者覆盖前者。两套加载器：

- **训练侧** `pipeline.stages.load_config`（`pipeline/stages.py:138-149`）：只解析**一层** `includes`、
  **顶层平铺合并**（不递归、不深合并）；主文件自身顶层键最后合并。因此臂文件必须同时列出
  `default.yaml` 与四个叶子子配置，且 `stages` 段会整体替换（详见 `config/arms/README.md`）。
- **评测侧** `pipeline.eval_runner.load_config`（`pipeline/eval_runner.py:1424-1445`）：递归解析 `includes`
  并**深度合并**（`tools/test.py --config` 走它，`tools/test.py:122-126`）。

## 各文件的作用与消费方

| 文件 | 作用 | 主要消费方（代码读取点） |
| --- | --- | --- |
| `default.yaml` | 主配置：`project` 元数据 + `includes` | 入口默认 `--config`（`tools/train.py:54`、`tools/test.py:45`） |
| `env.yaml` | 仿真/步长/观测/场景的冻结口径 | 见下 |
| `model.yaml` | 网络结构 | `pipeline.stages.build_model`（`pipeline/stages.py:152-194`）；`pipeline.vector_env.load_supervised_labels`（`pipeline/vector_env.py:104-131`）；评测/采集侧 model_config 透传（`pipeline/eval_runner.py:1691-1703`、`tools/dagger_collect.py:1284-1295`） |
| `train.yaml` | 运行开关 / 分阶段训练 / 监控 | `pipeline.stages`（`_resolve_*` 与各 stage 段）、`tools/run_config.py:56-132`、`pipeline/phase3_loop.py`（`stages.B.phase3` 段） |
| `eval.yaml` | 评测 spec / KPI / 阈值 / dataset_gate | `pipeline.eval_runner.py`（`1630-1641`、`1739-1742`、`1764-1770`；阈值判定 `398-460`） |
| `plan_anchors_k6.json` | K-anchor 计划头锚字典（K=6） | `net.anchor.load_anchor_dictionary`（`net/anchor.py:98-115`） |
| `arms/` | 可执行 RL 臂配置 | 见 `config/arms/README.md` |

### env.yaml

- 数值是冻结口径的单一记录；运行时由 `env/obs` 等模块内置默认镜像（如盒式 scope 前 150 / 后 50 /
  左 25 / 右 25：`env/obs/od.py:40-41,103`；场景 block 数与 `env/scenario/generator.py:63`、
  `env/scenario/taxonomy.py:127` 的注释一致）。
- 显式传参路径：`ObservationBuilder` 接受扁平别名 `topk_objects` / `topk_lanes` / `history_frames` 与
  `scope`（`env/obs/builder.py:29,89-101`）；但训练/评测标准链路不传——`build_pool` 无 obs_config 参数
  （`pipeline/trainer.py:6031-6043`），`LocalEnvPool` 缺省 `obs_config=None` → `ObservationBuilder` 内置默认
  （`pipeline/vector_env.py:88,632`）；评测侧只认 `config["env"]["obs"]` 形态（`pipeline/eval_runner.py:1671,1706`），
  标准平铺配置下为空。
- 部分键由环境侧固定而非读配置：`store_map=false` 写死在 `build_env`（`env/metadrive_env.py:674`）。

### model.yaml

- `build_model` 读取的键（`pipeline/stages.py:171-192`）：`hidden_dim`、`moe.primary.hidden_dim`、
  `moe.experts.count/hidden_dim`、`moe.router.hidden_dim`、`policy.trunk_hidden`、`value.net_hidden`、
  `spatial.layers`、`policy.attn_heads/attn_layers`、`world_model.rollout_steps`、
  `plan_anchor.enabled/num_anchors/path/temperature/hard`。
- `moe.router.supervised_labels` 由 `pipeline.vector_env.load_supervised_labels` 读取（8 标签固定顺序，
  路由监控/数据集契约用；`pipeline/vector_env.py:104-131`）。
- `plan_anchor.enabled=false`（默认）→ `num_anchors=0`，模型与旧版逐位一致（`pipeline/stages.py:171-174`）；
  锚字典路径缺失时回退 `net/anchor.py` 内置默认，文件非法则报错不静默回退（`net/anchor.py:98-115`）。

### train.yaml

- 消费方示例：`train.ckpt_every`（`pipeline/stages.py:468-476`）、`stages.C.max_episode_steps`（`512-528`）、
  `stages.C.trainable_scope`（`4468-4470`）、`stages.C` 其余解析器（`spec_rotation` / `policy_logstd_max` /
  `target_kl` / `lam` 等，`543-714`）、`stages.B.bc.action_dim_weights`（`3008-3009`）、
  `train.probe_batch`（`4502`）、`monitoring`（`202-208`）；
  `run.*` 与 `stages.*` 由 `tools/run_config.py:56-132` 解析为脚本变量。
- `stages.B.phase3` 段由 `pipeline/phase3_loop.py` 读取（如 `pipeline/phase3_loop.py:500-532`）。

### eval.yaml

- 消费方：`eval.spec`（`pipeline/eval_runner.py:1631`）、`mem_available_floor_mb` / `train_pool_policy` /
  `recycle_every_specs`（`1639-1641`）、`deterministic`（`1764`）、`process`（`1770`）、`thresholds`
  （`1739-1742`；判定公式见 `judge_group` / `evaluate_verdicts`，`pipeline/eval_runner.py:398-460`）。

### plan_anchors_k6.json

- 用途：K-anchor 计划头（`model.yaml::plan_anchor`）的形状锚字典，`anchors[k] = {ds[6], dtheta_lane[6]}`
  + 来源元数据（K=6、dt=0.5、拟合数据集等）。
- 生成：`tools/fit_plan_anchors.py`（expert 5k 数据 → KMeans → 写 `config/plan_anchors_k6.json`；用法见其
  docstring 的示例命令）。
- 加载：`net.anchor.load_anchor_dictionary`（`net/anchor.py:98-115`）；仅当 `plan_anchor.enabled=true`
  时构造锚头（`pipeline/stages.py:171-174`）。

## 未被代码消费的键（速查，以代码为准）

以下键当前只作文档/记录用途，仓库内无运行时读取点（负向断言可用 `grep -rn <键> --include='*.py'` 复核）：

| 文件 | 键 | 依据 |
| --- | --- | --- |
| `model.yaml` | `param_budget_estimate` | 无读取点；预算以实测参数量为准（`net/param_probe.py`） |
| `model.yaml` | `temporal.*` | `build_model` 不读（`pipeline/stages.py:152-194`）；`DrivingModel` 构造参数无对应项（`net/model.py:300-327`） |
| `model.yaml` | `moe.primary.enabled` / `router_gated` / `experts.init` | 同上；`build_model` 只读 hidden_dim/count |
| `model.yaml` | `moe.router.type` / `top_k` / `supervision` | `build_model` 不读；模型按固定 `moe_top_k=2` 构造（`net/model.py:313`） |
| `model.yaml` | `world_model.enabled` / `conditioned_on` / `loss` | `build_model` 只读 `rollout_steps`（`pipeline/stages.py:188`） |
| `model.yaml` | `policy.type` / `rollout_steps` / `action.*` | `build_model` 只读 `trunk_hidden` / `attn_heads` / `attn_layers`（`pipeline/stages.py:183,186-187`） |
| `train.yaml` | `stages.A.world_model.trainable`（`config/train.yaml:86`） | 无读取点；A 阶段可训练集硬编码为「除 `policy./value.` 外全部」（`pipeline/stages.py:1793-1801`） |
| `train.yaml` | `stages.B.bc.wm_detach`（`config/train.yaml:188`） | 注释即标 no-op；代码固定传 `wm_detach=True`（`pipeline/stages.py:2348,3254`） |
| `train.yaml` | `stages.C.wm_freeze_updates`（`config/train.yaml:203`） | 已弃用；代码只读 CLI（`pipeline/stages.py:4632`、`5112`） |
| `train.yaml` | `stages.C.kl_anchor` / `kl_anchor_coef` / `kl_anchor_final_coef`（`config/train.yaml:200-202`） | KL 锚实际只读 CLI `--kl-anchor-coef` / `--kl-anchor-final-coef`（`pipeline/stages.py:4582-4584`；默认值 `5108-5109`）；臂文件内同名键仅为记录（`config/arms/v7_arm1_offroad.yaml:9-11,47-49`） |
| `eval.yaml` | `eval.kpis` | KPI 集合硬编码于 `reward_model/kpi.py:45`（配置为镜像；`tests/test_reward_terms.py:1060` 断言两者一致） |
| `eval.yaml` | `eval.every_epochs` | 无读取点（周期评测由外部编排/driver 决定） |
| `eval.yaml` | `eval.dataset_gate.*` | 采集工具内置默认（`tools/collect_expert.py:186-188`、`tools/dagger_collect.py:1184`） |
| `env.yaml` | `sim.workers` / `timing.preview_points` / `timing.preview_horizon` / `obs.distance_to_turn` / `obs.signal_placeholder` | 无读取点；`sim.store_map=false` 由 `build_env` 固定（`env/metadrive_env.py:674`） |

> 本表为 2026-10-09 对 HEAD 代码的快照；新增/删除键时请以读取点复核，勿以本表替代代码。
