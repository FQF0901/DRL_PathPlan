# Stage A 配置消费审计（R30-5：键级 CONSUMED/IGNORED/LEGACY/DERIVED）

状态：R30-5 只读审计（2026-10-13）；证据 @HEAD `acf1b8f`；**零代码/配置改动、不 commit**。
范围：`config/train.yaml::stages.A`（含 `world_model` 块）+ `run_stage_a` 相关 CLI + 附录
`stages.C`「声明但未消费」核查。grep 范围 = `pipeline/`、`tools/`、`net/`（`--include='*.py'`，
另人工核对 `tools/train.sh` 脚本链），排除 `config/*.yaml` 自身、`docs/`、`.slim/`。

独立证据链说明：`docs/deepseekv4p1_argue.md` §5.29（C1/C4/C7）已给出结论；本文不复述其
论证，而是逐键给出**可直接复现的消费点/零命中 grep**。与 §5.29 的差异（新增/修正）：
① `world_model.loss` 不仅"语义过时"，且**全仓零读取**（§5.29 仅称"语义过时"）；
② 新增 `world_model.noise`、`stages.A.name`、`stages.C.kl_anchor*` 等未消费键；
③ `stages.A.epochs/batch/micro` 并非纯粹 IGNORED——它们经 `tools/run_config.py` → `tools/train.sh`
进入 CLI（脚本链 CONSUMED），仅"直连 `python -m pipeline.stages`"时被忽略。

## 0. 判定口径

| 判定 | 定义 |
| --- | --- |
| CONSUMED | 键被代码读取且进入训练行为（给出 `file:line` + 1 行证据） |
| IGNORED | 全仓 grep 零命中（贴命令）；或在代码中被读键名但从不使用 |
| LEGACY | 键/语义已被 v8 取代或失效（含"注释留档的不存在键"） |
| DERIVED | YAML 无声明键，生效值来自代码内部默认/推导（等价"内部推导"） |

计数口径：按表格行计数（§1.1/§1.3 配置键 18 行 + §2 CLI 36 行 + §3.1 漂移附录 16 行 = 70 行）；
§1.3 末的宏/micro 分母等纯内部派生量**不是配置键**，仅列说明、不计入。

## 1. `stages.A` 顶层与 `world_model` 键

### 1.1 逐键表

| 键 | 声明值 | 消费点（或 grep 证据） | 判定 | 备注 |
| --- | --- | --- | --- | --- |
| `stages.A.name` | `world_model_teacher_forcing` | 零命中：`grep -rn 'stage_a.get("name")\|stage_cfg.get("name")' pipeline/ tools/ --include='*.py'` → 0；`metrics.kind` 硬编码同名字符串 `pipeline/stages.py:2452` | IGNORED | 纯人读注释键；连 metrics 都不回写（硬编码与 YAML 当前一致，改动 YAML 无效果） |
| `stages.A.epochs` | `20`（`&stage_a_epochs`） | `tools/run_config.py:121` `stage_a.get("epochs", world_model.get("epochs", 20))` → `WM_EPOCHS` → `tools/train.sh:19` `--wm-epochs` → `pipeline/stages.py:2437` | CONSUMED | **仅脚本链**；直连 pipeline 时回退 `world_model.epochs`（2437）。与 `world_model.epochs` 靠 YAML anchor 同值 |
| `stages.A.batch` | `1024`（`*stage_a_batch`←`train.wm.batch_size`） | `tools/run_config.py:124` `chosen.get("batch", 1024)` → `--batch-size`（train.sh:19）→ `pipeline/stages.py:1831-1832` | CONSUMED | 仅脚本链；直连 pipeline 读 `train.wm.batch_size`（1832）。同源别名，当前值一致 |
| `stages.A.micro` | `256`（`*stage_a_micro`←`train.wm.micro_batch_size`） | `tools/run_config.py:125` `chosen.get("micro", 256)` → `--micro-batch-size` → `pipeline/stages.py:1834-1837` | CONSUMED | 同上；1837 再取 `min(macro, micro)`，1838 冒烟再收紧 |
| `stages.A.world_model.epochs` | `20`（`*stage_a_epochs`） | `pipeline/stages.py:2437` `epochs = int(args.wm_epochs or wm_cfg.get("epochs") or 10)`；`tools/run_config.py:121` 回退 | CONSUMED | CLI `--wm-epochs` 优先 |
| `stages.A.world_model.trainable` | `[encoders, mem_encoder, plan_head, st_gnn]` | 零命中：`grep -rn 'world_model.*trainable\|stage_a\["trainable"\]' pipeline/ tools/ net/ --include='*.py'` → 0（唯一读取在 `tests/test_integration_v12.py:280`，仅断言 YAML 文本） | IGNORED | 实际 = **除 `policy./value.` 外全参进 optimizer**（`stages.py:1810-1819`）；见 §1.2 死参说明 |
| `stages.A.world_model.ego_condition` | `expert_gt_actions` | 零命中：`grep -rn "ego_condition" pipeline/ tools/ net/ --include='*.py'` → 0 | IGNORED | 行为与声明一致但硬编码：`plan[:, k-1]` GT 动作链逐步喂入（`stages.py:2070-2077`） |
| `stages.A.world_model.targets` | `future_od_ld_lookup` | 零命中：`grep -rn 'get("targets")' pipeline/stages.py pipeline/trainer.py` → 0 | IGNORED | 行为硬编码在 `_future_targets`（`stages.py:1897-1927`）+ `FrameWindows.build_future`（1912） |
| `stages.A.world_model.slot_matching` | `nearest_neighbor_t0` | 零命中：`grep -rn "slot_matching" pipeline/ tools/ net/ --include='*.py'` → 0 | IGNORED | 行为由 CLI `--match-future-slots/--match-gate-m`（默认 True/8.0）驱动，且仅对**无 `od_id` 的旧数据集**生效（`stages.py:1913-1920`）；v2 数据集恒有 id ⇒ 条件失效 |
| `stages.A.world_model.loss` | `direct_multi_step` | 零读取：`grep -rn "direct_multi_step" pipeline/ tools/ net/ --include='*.py'` → 仅 `stages.py:2481`（硬编码 metrics 常量）与 `trainer.py:2273`（注释） | LEGACY | 语义已被 v8 取代：主监督 = latent consistency 1.0（`stages.py:2228-2240`，组合见 `2328`），od/ld 退为诊断 0.1/0.02；且该键**根本未被读取**（严格说是 IGNORED） |
| `stages.A.world_model.noise` | `ego_plan` | 零命中：`grep -rn 'get("noise")\|"noise"' pipeline/ tools/ net/ --include='*.py'` → 0（stage A 噪声来自 `args.plan_noise*`） | IGNORED | 真实旋钮 = CLI `--plan-noise/--plan-noise-p/ds/dtheta`（`stages.py:1856-1858, 2071-2073`）；YAML 改键无效 |
| `stages.A.world_model.ego_next_coef` | `0.1` | `stages.py:1874-1878` 解析（CLI `--wm-ego-next-coef` 优先）→ `2329-2330` `total += ego_next_coef * terms["ego_next"]` | CONSUMED | plan head 在 A 阶段唯一梯度源（MoE 关） |
| `stages.A.world_model.presence_coef` | `0.1` | `stages.py:1864-1868` → `2279`（开关）/`2331-2332` | CONSUMED | 无 presence 头时自动跳过并告警（`2281-2289`） |
| `stages.A.world_model.entry_coef` | `0.1` | `stages.py:1869-1873` → `2279`/`2332` | CONSUMED | 同上 |

### 1.2 `trainable` 的实际行为与"死参"（§5.29 C4 的独立证据）

- 实际可训练集（排除式过滤，非 YAML 列表）：

  ```python
  # pipeline/stages.py:1810-1819
  trainable = [p for name, p in model.named_parameters()
               if not name.startswith(("policy.", "value."))]
  for p in model.parameters(): p.requires_grad_(False)
  for p in trainable: p.requires_grad_(True)
  optimizer = torch.optim.Adam(trainable, lr=float(args.lr))
  ```

- A 阶段强制关 MoE：`pipeline/stages.py:1760` `model.set_moe(enabled=False)`；
  `net/model.py:633-640` → `net/plan_head.py:125-128` → `net/moe.py:171-173`
  （`if not self.moe_enabled: return primary_out` ——"专家/路由无梯度"）。
- **死参含义**：`plan_head.moe.experts.*`、`plan_head.moe.router.*`、`plan_head.moe.residual_scale`
  （`net/moe.py:113`）不匹配 `policy./value.` 前缀 ⇒ **在 optimizer 参数组中，但 A 阶段前向不执行、
  恒无梯度**（Adam 对 `grad=None` 参数不更新）；A 阶段也无 MoE 负载均衡 aux 损失
  （组合式仅 latent/od/ld/ego_next/presence/entry，`stages.py:2328-2332`）。
- YAML 列表与 `tests/test_integration_v12.py:280-283` 只校验"列表文本不含 moe/world_model/spatial"，
  与运行时排除式过滤是两套口径——测试通过 ≠ 列表被消费（假安全感）。

### 1.3 v8 权重与废弃键（YAML 无实质键）

| 键 | 声明值 | 消费点（或 grep 证据） | 判定 | 备注 |
| --- | --- | --- | --- | --- |
| `world_model.latent_coef` | YAML 无键（注释声明 1.0，`train.yaml:81-83`） | `stages.py:1883-1885` `args.wm_latent_coef if not None else 1.0`；CLI 定义 `5647-5649` | DERIVED | 生效值 = 代码内部默认 1.0（主监督），YAML 无法配置 |
| `world_model.od_coef` | YAML 无键（注释 0.1） | `stages.py:1886-1888`（默认 0.1）；CLI `5650-5651` | DERIVED | 物理诊断项 |
| `world_model.ld_coef` | YAML 无键（注释 0.02） | `stages.py:1889-1891`（默认 0.02）；CLI `5652-5653` | DERIVED | 物理诊断项；LD 目标前 4 维（`2220-2227`） |
| `world_model.ld_coef`（旧键） | 注释留档（`train.yaml:83`「旧键 ld_coef 已废弃不再读取」） | `stages.py:1881-1882` 注释明示不再读取（历史 0.0 口径） | LEGACY | git 历史键；当前 YAML 已无此键 |

派生量说明（非配置键，不计入计数）：宏/micro 损失分母由 `_term_denominators`（`stages.py:2117-2139`）
按 `w·valid·mask` 现算；梯度累积按 `MicroNum/MacroDen` 缩放（`2315-2324`）。这些是纯内部推导，
与 YAML 无关。

## 2. `run_stage_a` 相关 CLI（列全）

定义段：`pipeline/stages.py:5593-5930`；消费点均为 `run_stage_a`（1733-2950）或 `main`（5933-5944）。
脚本链 = `tools/train.sh` 以 `eval "$(tools/run_config.py)"` 取值后显式传参（`tools/train.sh:19`）。

| CLI | 定义处 | 消费点 | 判定 | 备注 |
| --- | --- | --- | --- | --- |
| `--stage` | 5595 | `main:5938` 分发 | CONSUMED | |
| `--config` | 5600 | `main:5935` `load_config`（includes 平铺合并，141-152） | CONSUMED | 无 schema 白名单（见 §3.2(f)） |
| `--model-config` | 5601 | `1741` `build_model(_load_yaml(args.model_config))` | CONSUMED | |
| `--out` | 5605 | `1737` | CONSUMED | |
| `--ckpt` | 5606 | `1754-1756` 载入初始化权重 | CONSUMED | |
| `--ckpt-every` | 5609 | `2438` `_resolve_ckpt_every`（474-497，回退 `train.ckpt_every`）→ `2703` 保存 | CONSUMED | 与 `train.ckpt_every`（`&ckpt_every`）同源 |
| `--resume` | 5614 | `1742-1757` 续跑（模型/优化器/RNG） | CONSUMED | |
| `--bc-dir` | 5617 | `1763` `BCDataset.load` | CONSUMED | |
| `--limit-dataset` | 5618 | `1763`；`1838` 冒烟上限 | CONSUMED | |
| `--allow-legacy-dataset` | 5622 | `1765` `_allow_legacy_dataset`（760） | CONSUMED | |
| `--wm-epochs` | 5628 | `2437` | CONSUMED | 帮助文本称默认值 10（YAML 实为 20）——help 漂移，见 §3.2(e) |
| `--val-frac` | 5629 | `1795` | CONSUMED | 仅在找不到独立 val-dir 的 legacy 落回时使用（1780-1797） |
| `--eval-frames` | 5630 | `1840` | CONSUMED | |
| `--plan-noise` | 5631 | `2071`（默认 True；`--no-plan-noise` 关闭） | CONSUMED | |
| `--plan-noise-p` | 5633 | `2072` | CONSUMED | metrics 只记 p/ds/dtheta、**不记 on/off**（`2469-2471`） |
| `--plan-noise-ds` | 5634 | `1857`（noise_std）→ `2073` | CONSUMED | |
| `--plan-noise-dtheta` | 5635 | `1857` → `2073` | CONSUMED | |
| `--match-future-slots` | 5636 | `1914`（metrics 回写 `2469`） | CONSUMED | 条件生效：数据集有 `od_id` 时绕过（1913）——v2 数据集下近 no-op |
| `--match-gate-m` | 5638 | `1919`（metrics `2470`） | CONSUMED | 同上 |
| `--wm-presence-coef` | 5639 | `1865` | CONSUMED | |
| `--wm-entry-coef` | 5642 | `1870` | CONSUMED | |
| `--wm-ego-next-coef` | 5644 | `1875` | CONSUMED | |
| `--wm-latent-coef` | 5647 | `1884` | CONSUMED | 默认 1.0（代码内；YAML 无键） |
| `--wm-od-coef` | 5650 | `1887` | CONSUMED | 默认 0.1 |
| `--wm-ld-coef` | 5652 | `1890` | CONSUMED | 默认 0.02 |
| `--val-dir` | 5665 | `1780` `_resolve_val_dir`（273） | CONSUMED | |
| `--history-stride` | 5662 | **A 区零引用**（1776/1951 固定 `_BC_STEP_STRIDE=5`，843）；仅 B/B3 读取（3080/4608） | IGNORED | 对阶段 A 静默 no-op |
| `--batch-size` | 5905 | `1831-1832` | CONSUMED | 优先于 `train.wm.batch_size` |
| `--materialize` | 5907 | `1851` | CONSUMED | |
| `--micro-batch-size` | 5910 | `1834-1837`（宏钳制）；`1838` 冒烟 | CONSUMED | 有效值经内部推导：`min(macro, micro)` |
| `--max-batches` | 5914 | `2605` | CONSUMED | 冒烟用 |
| `--lr` | 5915 | `1819` | CONSUMED | Adam |
| `--device` | 5918 | `1740` | CONSUMED | |
| `--seed` | 5920 | `1795`（legacy 切分）/`2498`（RNG） | CONSUMED | |
| `--monitor` | 5921 | `2493-2496` `_monitor_enabled`（205-211，回退 `monitoring.*`） | CONSUMED | |
| `--monitor-legacy-tags` | 5923 | `2495` `_monitor_legacy_tags`（227-229） | CONSUMED | |

## 3. 附录：已知漂移清单（含 `stages.C` 声明但未消费键）

### 3.1 键级漂移（`stages.C`，同文件核查）

| 键 | 声明值 | 消费点（或 grep 证据） | 判定 | 备注 |
| --- | --- | --- | --- | --- |
| `stages.C.name` | `ppo_rl` | 零命中：`grep -rn 'stage_cfg.get("name")' pipeline/ tools/` → 0 | IGNORED | 同 `stages.A.name` |
| `stages.C.tracker` | `mpc_lqr` | 零命中：`grep -rn "mpc_lqr" pipeline/ tools/ net/ --include='*.py'` → 0；`build_pool(..., tracker="lqr", ...)` 硬编码 `stages.py:5380` | IGNORED | §5.29 C7 确认；且 `mpc_lqr` 本非合法取值（`build_pool` 校验 `{exact,lqr,kinematic}`，`trainer.py:6863`；eval `--tracker` 仅 `{lqr,exact}`，`eval_runner.py:1716`） |
| `stages.C.router_supervision` | `none` | 零命中：`grep -rn "router_supervision" pipeline/ tools/ net/ --include='*.py'` → 0 | IGNORED | lane U1 去聚类后的说明性键 |
| `stages.C.kl_anchor` | `true` | 零命中（`get("kl_anchor")` 零命中）；参考模型恒构建（`5273-5275`） | IGNORED | 布尔键无开关作用 |
| `stages.C.kl_anchor_coef` | `0.05` | 零命中；生效值 = argparse 硬编码默认 0.05（`5869`）→ `5276` | IGNORED | 当前 YAML 与硬编码同值 ⇒ 静默；改 YAML 无效 |
| `stages.C.kl_anchor_final_coef` | `0.0` | 零命中；硬编码默认 0.0（`5870`）→ `5277` | IGNORED | 同上 |
| `stages.C.wm_freeze_updates` | `null` | 零命中（config 键）；CLI `--wm-freeze-updates` 仅打印弃用告警（`5326-5331`）；metrics 由 `updates` 反推（`5332`） | IGNORED | W1 后语义废弃（st_gnn 全期冻结） |
| `stages.C.wm_loss_enabled` | `false` | `5338-5343`：`true` → fail-fast 拒绝解冻 | CONSUMED | 解冻守卫（W2 未接线） |
| `stages.C.plan_reference` | `repeat_action` | `5256`（CLI 优先） | CONSUMED | |
| `stages.C.trainable_scope` | `design` | `5164` + 校验 `5165-5168` | CONSUMED | |
| `stages.C.spec_rotation` | `episode` | `_resolve_stage_c_spec_rotation`（546-557） | CONSUMED | |
| `stages.C.max_episode_steps` | `200` | `_resolve_stage_c_max_episode_steps`（515-531） | CONSUMED | |
| `stages.C.policy_logstd_max` | `null` | `_resolve_stage_c_policy_logstd_max`（561-576） | CONSUMED | |
| `stages.C.primary_lr_scale` | `0.1` | `5209` | CONSUMED | |
| `stages.C.probe_interval` | `25` | `_resolve_stage_c_probe_interval`（717-731） | CONSUMED | |
| `stages.C.ckpt_every` | `25` | `_resolve_stage_c_ckpt_every`（500-513） | CONSUMED | |

注：`stages.C.reward` 在 YAML 中为**注释块**（`train.yaml:229-233`），非声明键，未计行。

### 3.2 非键级漂移（文档/指标/测试）

| # | 漂移 | 证据 |
| --- | --- | --- |
| a | ① `world_model.loss` 语义过时（见 §1.1）；metrics 的 `ld_loss` 被**硬编码为 `"direct_multi_step"`**，与 v8 真实主监督（latent consistency）不符，且被测试固化 | `stages.py:2481`；`tests/test_stage_a_ld_loss.py:66`、`tests/test_stage_v2_smoke.py:75` |
| b | ② `world_model.trainable` 未消费（见 §1.2）；`pipeline/README.md:86` 已告警，但 `README.md:80` 仍把 `loss/ego_condition/targets/slot_matching/noise` 列为 `stages.A.*` 配置入口 | `pipeline/README.md:80,86` |
| c | ③ `stages.C.tracker: mpc_lqr` 未消费（见 §3.1）；代码强制 `"lqr"` | `stages.py:5380` |
| d | `tools/collect_expert.py` docstring 引用**不存在**的键 `config/train.yaml::stages.A.bc.expert`（`stages.A` 无 `bc` 子块） | `tools/collect_expert.py:9-10`；`grep -n "stages.A.bc" config/train.yaml` → 0 |
| e | `--wm-epochs` help 称默认 `world_model.epochs=10`，YAML 实为 20；`stages.py:2437` 兜底也是 10（配置缺失时才用） | `stages.py:5628` vs `train.yaml:74` |
| f | 无配置 schema/白名单：`load_config` 平铺合并 + 各阶段零散 `get()`，未知/旧键**静默忽略**、无"未消费键"告警 | `stages.py:141-152`；`grep -rn "未知键\|未消费" pipeline/ tools/ --include='*.py'` → 0 |
| g | `plan_noise` metrics 只回写 p/ds/dtheta，**不回写开关**；`--no-plan-noise` 的 run 会误显示同样噪声参数 | `stages.py:2469-2471` |
| h | `--match-future-slots` 在 v2 数据集（schema 强制含 `od_id`）下被 id 对齐绕过，事实上近 no-op | `stages.py:1913`；`README.md:90` |

## 4. 风险与建议（审计期间只记录不改；按键给出建议动作）

| 键/项 | 风险 | 建议动作 |
| --- | --- | --- |
| `world_model.loss` | 声明与实际主监督不符；下游按 `direct_multi_step` 解读 A 阶段损失 | 删键；把 `metrics["ld_loss"]`（2481）改为真实口径（如 `wm_main_loss="latent_consistency"`）并同步两处测试 |
| `world_model.trainable` | 运维按 YAML 判断可训范围，实际为排除式全参（含 MoE 死参） | 二选一：**白名单读取 + fail-fast**（按前缀选参、未命中报错），或删键 + 更新 `pipeline/README.md:80` 与 `tests/test_integration_v12.py:280-283` |
| `world_model.ego_condition`/`targets`/`slot_matching`/`noise` | 描述性键被误当配置旋钮（尤其 `noise`） | 加注释显式标"描述性、不参与解析"；或删除。`noise` 若要可配置，应在 `run_stage_a` 真正读键（CLI 覆盖） |
| `stages.A.epochs`/`batch`/`micro` | 两套真相：脚本链读它们、直连 pipeline 忽略 | 在 `run_stage_a` 增加 `stages.A.*` 二级 fallback（或注释/README 明示"仅脚本链"），消除歧义 |
| `stages.C.tracker` | 声明 `mpc_lqr`（非合法取值）且被硬编码 `lqr` 覆盖 | 删键或 fail-fast 校验（`build_pool` 合法集 `{exact,lqr,kinematic}`，`trainer.py:6863`）；若需可选，接 `build_pool(tracker=...)` |
| `stages.C.kl_anchor`/`kl_anchor_coef`/`kl_anchor_final_coef` | YAML 声明被静默忽略，改配置无效；当前仅因同值而未爆雷 | 接线到解析链（CLI 默认改 `None` → 回退 config）或删键 |
| `stages.C.wm_freeze_updates` | 弃用语义残留（null 无效果） | 删键（docstring 已有 W1 弃用说明） |
| `stages.A.name`/`stages.C.name` | 与 `metrics.kind`/`stage` 双源 | 删键或接入 metrics |
| `stages.C.router_supervision` | 说明性键 | 删键（或保留注释语义） |
| `--history-stride`（对 A） | 静默 no-op | `run_stage_a` 入口 fail-fast 或告警 |
| 系统级 | 无未消费键检测，漂移可长期静默 | 建议在 `load_config`/阶段入口加"已声明键白名单 + 未消费键启动告警"（只读建议；冻结期不改代码） |

## 5. 计数汇总

按表格行（§1.1=14 + §1.3=4 + §2 CLI=36 + §3.1=16 = **70 行**）：

- **CONSUMED = 51**（§1.1 7 + §2 35 + §3.1 9）
- **IGNORED = 14**（§1.1 6 + §2 1 + §3.1 7）
- **LEGACY = 2**（§1.1 `loss`、§1.3 旧 `ld_coef`）
- **DERIVED = 3**（§1.3 v8 三权重：latent/od/ld coef）

说明：`world_model.loss` 按"语义过时"归 LEGACY，但其**严格读态也是 IGNORED**（零读取），风险章按两者叠加处理。
