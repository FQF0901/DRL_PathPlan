# P4 臂配置（Stage C RL；Gate4 准备）

> **来源**：`docs/v6_program_prereg.md` §7.1（臂顺序/单变量）+ E-β″ 复算定稿
> （`docs/reward_audit/ebeta2/`；裁定记录见其 `MANIFEST.md`）。本目录是**可执行 run config**
> （`tools/train.py --stage C --config config/arms/armX.yaml ...`）；审计证据/原始产物留在
> `docs/reward_audit/ebeta2/`，此处只放组装后的臂配置（`--config` 直接消费）。

## 臂清单（按 §7.1 顺序；每臂单变量，前臂通过再开下臂）

| 臂 | 文件 | 单变量（相对 arm0） | 终局值档 | 预期/依据 |
|---|---|---|---|---|
| arm0 | `arm0_bundle_rc1.yaml` | —（bundle 底座） | rc=1：+29/−22/−14，max_step −46 | 底座可训 + 剖面基线 |
| arm1 | `arm1_rc3.yaml` | `route_completion` 1→3 + 同档终局值 | rc=3：+27/−23/−15，max_step −48 | rc 扫档（同档配对，不得跨档复用） |
| arm2 | `arm2_rc10.yaml` | `route_completion` 1→10 + 同档终局值 | rc=10：+20/−28/−18，max_step −54 | 同上 |
| arm3 | `arm3_rc30.yaml` | `route_completion` 1→30 + 同档终局值 | rc=30：+0/−41/−26，max_step −71 | 同上 |
| arm4 | `arm4_lam098.yaml` | `train.ppo.lam` 0.95→0.98 | rc=1（同 arm0） | λ 对照（§7.2）：returns/EV/崩解窗口 |
| arm5 | `arm5_ttc.yaml` | 追加 `ttc`（−0.5 / 2.0 s / 0.5 s） | rc=1（同 arm0） | 前置离线证伪已完成（collision 触发率 67.6%）→ 保留本臂、阈值 2.0 s |
| arm6 | `arm6_lane_boundary.yaml` | 追加 `lane_boundary`（−0.2 / margin 0.5 m） | rc=1（同 arm0） | 覆盖出界类（口径同 `rl_reward_v5.md` §5） |
| arm7 | `arm7_lane_center.yaml` | 追加 `lane_center`（−0.1 / deadband 0.25 m） | rc=1（同 arm0） | 车道中心偏离罚（口径同 §5） |
| arm8-A | `arm8_collision_suppress_term.yaml` | `aggregation.terminal_values.collision` −22→−32 | rc=1 其余同 arm0 | §7.5 碰撞抑制首选：碰撞/出界分离 6.4→16.4（救援盈亏门槛 9.0%→20.2%） |
| arm8-A′ | `arm8_collision_suppress_term46.yaml` | `collision` −22→−46 | rc=1 其余同 arm0 | §7.5 条件升级（A 主判据过、辅助闸未过且 A 碰撞率 ≤15.0% 才跑） |
| arm8-B | `arm8_collision_suppress_gap.yaml` | 追加 `lead_gap`（−1.0 / gap_ref 6.0 / cap 1.0） | rc=1（同 arm0） | §7.5 后备：近碰 dense（ttc 近零激活的替代；A 主判据失败才启用） |

- bundle 底座（arm0）= v5 奖励默认全量：`speed_ratio` 0.4 + `low_speed` 启用 + rc 选定档 +
  终局值定稿表；其余项参数不变。
- 终局值来源：`docs/reward_audit/ebeta2/config_draft_rc*.yaml`（数值逐档一致，dry-run 单测锁定）。
- **max_step 裁定（Orchestrator 2026-10-01；Gate4 修正）**：采用 E-β″ 点估计 −46/−48/−54/−71；
  标注 **n=8/向、不判通过、首臂后复核、不扩采**；**horizon 对齐 = 训练截断 200 策略步**
  （=100 s，与审计/评测一致；旧 600=300 s=3×）。见 `docs/v6_program_prereg.md` §7.1/§7.4
  与 ebeta2 `MANIFEST.md`（后者"显式接受"已被本修正替代）。
- **arm8（碰撞抑制，§7.5；Gate4 终审立项）**：开跑前置 = arm0 seed=11 复现闭环；顺序
  A →（A′ 条件）→ B；判据 = §7.3 主判据 + 辅助碰撞闸（collΔ ≤ +3pt 或 ≤10%，相对 E-β″ 零点）。
  诊断/推导见 `/tmp/opencode/v6_collision_arm_design.md`（产物脚本
  `v6_collision_diag.py` / `v6_collision_split.py` / `v6_collision_gap_sizing.py`，不入 repo）。

## 加载语义（重要；`includes` 为**一层平铺合并**）

`pipeline.stages.load_config`（`tools/train.py` 实际使用的加载器）**只解析一层 includes**（不递归），
且为**顶层平铺合并**（主文件顶层键整体替换 include 的同名键），不做深度合并。因此臂文件：

- `includes` 同时列出 `config/default.yaml` 与其四个叶子子配置（`env/model/train/eval`）：
  只写 `config/default.yaml` 时，平铺加载器只会带入其 `project`/`includes` 元数据而**丢失
  `train`/`data`/`run`/`monitoring` 等段**（dry-run 实测；`tests/test_p4_arm_configs.py` 已钉住）。
- arm 文件的 `stages` 段会整体替换合并结果里的 `stages`；非 reward 的 `stages.C` 键
  （`trainable_scope` / `primary_lr_scale` / `plan_reference` / `ckpt_every` / `spec_rotation` /
  `policy_logstd_max` / `probe_interval` / `wm_loss_enabled`）回落代码默认——本提交已核对与
  `config/train.yaml` 逐键一致；P4 driver 的显式 CLI pins（同 v4 口径：`--spec/--pool/--envs/
  --trainable-scope/--updates/--rollout-steps/--max-episode-steps 200/--seed/--device/...`）照旧优先
  （pin 表见 `docs/v6_program_prereg.md` §7.4）。
- **arm4 例外**：显式钉住整段 `train`（值 = `config/train.yaml`，仅 `lam: 0.98`）——只写
  `train.ppo.lam` 会把 `train` 段其余键打回代码默认（如 threads cap 8→4），整段复制保证
  「λ 单变量、其余 pins 逐位一致」。
- 递归 deep merge 加载器（`pipeline.eval_runner.load_config`；`tools/test.py --config` 走它）
  下 includes 递归解析，臂文件同样可用；若需保留 A/B 段做深度合并，可用它先解析再落盘为 run 快照。

## 验证（CPU dry-run；`tests/test_p4_arm_configs.py`）

- 逐臂：`load_config(arm)` → `stages.C.reward` → `build_reward_adapter` 通过（含 rc 档配对守卫）；
- 终局值与 `docs/reward_audit/ebeta2/config_draft_rc*.yaml` 逐档一致（**arm8-A/A′ 仅
  `collision` 单变量偏离 −32/−46**，其余逐位同 rc=1 草案）；
- 权重/参数：`speed_ratio` 0.4、`low_speed` 启用、各臂单变量项
  （ttc/lane_boundary/lane_center/lead_gap）与 arm4 `train.ppo.lam=0.98`；
- `lead_gap` 项参数（−1.0 / gap_ref 6.0 / cap 1.0）经适配器逐项断言；
- `max_step` 终局值经适配器实际结算（`terminal_key=max_step`）逐档断言。

## 使用示例

```bash
# 单臂（其余 pins 由 P4 driver 按 §7.1/v4 口径传；GPU 串行 = 臂间可比）
tools/venv-python tools/train.py --stage C --config config/arms/arm0_bundle_rc1.yaml \
  --spec env/specs/scenarios_train_dagger_r1.json --pool local --envs 1 \
  --ckpt runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt \
  --updates 200 --out runs/<run>/stage_c_arm0
```
