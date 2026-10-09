# config/arms 历史臂快照（v6 P4 奖励臂 / v7 奖励与结构臂）

> 移入：2026-10-09（v8 清理-1，18 项；同日收尾追加 3 项，共 21 项）。原路径均为
> `config/arms/<同名文件>`（现路径 `docs/archive/config_arms_legacy/<同名文件>`）。
> 原因：当前线只采用 `config/arms/v7_arm1_offroad.yaml`（保留原位、未动）；本批为 v6 P4 奖励臂与
> v7 奖励/结构臂（K-anchor）的历史快照，不再是现行可执行 run config。文件内容原样，未做任何修改。
> 逐项操作与引用检查见 `docs/cleanup/V8_CLEANUP_config_docs.md`；v6 P4 臂的历史清单另见
> `config/arms/README.md`（该 README 待按新现状重写）。

## 族与用途（每族一句话）

- **v6 P4 奖励臂**（`arm0_bundle_rc1` / `arm1_rc3` / `arm2_rc10` / `arm3_rc30` / `arm4_lam098` /
  `arm5_ttc` / `arm6_lane_boundary` / `arm7_lane_center` / `arm8_collision_suppress_{gap,term,term46}`）：
  Gate4 单变量臂链——bundle 底座、rc 扫档、λ 对照、ttc、车道边界/中心罚、碰撞抑制（预注册 §7.1/§7.5）。
- **v7 奖励臂**（`v7_arm1_ttc` / `v7_arm1_collision_suppress` / `v7_arm2_bundle_kl` /
  `v7_arm3_bundle_only` / `v7_reward_A` / `v7_reward_B` / `v7_reward_C`）：v7 §9–§14 奖励与
  KL 锚单变量对照（off_road_edge 裕度、ttc 近失、碰撞终局值、bundle/KL 隔离、speed_deficit /
  comfort_jerk_win）。
- **v7 结构臂（K-anchor）**（`v7_struct_b_v5_train` / `v7_struct_b_v5_model` / `v7_struct_b_v5_eval`）：
  v7 §11 结构迭代 B（v5 重训链）的 Stage B 训练臂 / 模型 config（plan_anchor 开）/ 评测 config 三件套。

## 文件清单（原路径 → 现路径）

| 原路径 | 现路径 |
|---|---|
| `config/arms/arm0_bundle_rc1.yaml` | `docs/archive/config_arms_legacy/arm0_bundle_rc1.yaml` |
| `config/arms/arm1_rc3.yaml` | `docs/archive/config_arms_legacy/arm1_rc3.yaml` |
| `config/arms/arm2_rc10.yaml` | `docs/archive/config_arms_legacy/arm2_rc10.yaml` |
| `config/arms/arm3_rc30.yaml` | `docs/archive/config_arms_legacy/arm3_rc30.yaml` |
| `config/arms/arm4_lam098.yaml` | `docs/archive/config_arms_legacy/arm4_lam098.yaml` |
| `config/arms/arm5_ttc.yaml` | `docs/archive/config_arms_legacy/arm5_ttc.yaml` |
| `config/arms/arm6_lane_boundary.yaml` | `docs/archive/config_arms_legacy/arm6_lane_boundary.yaml` |
| `config/arms/arm7_lane_center.yaml` | `docs/archive/config_arms_legacy/arm7_lane_center.yaml` |
| `config/arms/arm8_collision_suppress_gap.yaml` | `docs/archive/config_arms_legacy/arm8_collision_suppress_gap.yaml` |
| `config/arms/arm8_collision_suppress_term.yaml` | `docs/archive/config_arms_legacy/arm8_collision_suppress_term.yaml` |
| `config/arms/arm8_collision_suppress_term46.yaml` | `docs/archive/config_arms_legacy/arm8_collision_suppress_term46.yaml` |
| `config/arms/v7_arm1_collision_suppress.yaml` | `docs/archive/config_arms_legacy/v7_arm1_collision_suppress.yaml` |
| `config/arms/v7_arm1_ttc.yaml` | `docs/archive/config_arms_legacy/v7_arm1_ttc.yaml` |
| `config/arms/v7_arm2_bundle_kl.yaml` | `docs/archive/config_arms_legacy/v7_arm2_bundle_kl.yaml` |
| `config/arms/v7_arm3_bundle_only.yaml` | `docs/archive/config_arms_legacy/v7_arm3_bundle_only.yaml` |
| `config/arms/v7_reward_A.yaml` | `docs/archive/config_arms_legacy/v7_reward_A.yaml` |
| `config/arms/v7_reward_B.yaml` | `docs/archive/config_arms_legacy/v7_reward_B.yaml` |
| `config/arms/v7_reward_C.yaml` | `docs/archive/config_arms_legacy/v7_reward_C.yaml` |
| `config/arms/v7_struct_b_v5_eval.yaml` | `docs/archive/config_arms_legacy/v7_struct_b_v5_eval.yaml` |
| `config/arms/v7_struct_b_v5_model.yaml` | `docs/archive/config_arms_legacy/v7_struct_b_v5_model.yaml` |
| `config/arms/v7_struct_b_v5_train.yaml` | `docs/archive/config_arms_legacy/v7_struct_b_v5_train.yaml` |

## 退役测试快照（`*.py.disabled`）

- `test_p4_arm_configs.py.disabled`、`test_v7_reward_arms.py.disabled`：v6 P4 臂 / v7 §13 奖励臂
  的退役测试快照；扩展名 `.disabled` 防 pytest 收集（对应臂文件已移入本目录，测试不再维护）。
- 当前臂守卫见 `tests/test_arm_configs.py`。

## 收尾说明（2026-10-09 同日）

- `arm0_bundle_rc1.yaml`、`arm4_lam098.yaml`、`arm5_ttc.yaml` 曾因 `tests/test_stage_c_horizon.py`
  引用作合并语义夹具（`load_config` 臂文件验证 Stage C horizon 回落）而暂留 `config/arms/`；
  该测试已改为**动态遍历 `config/arms/*.yaml`**（sorted glob，先 assert 至少 1 个文件），
  三臂随即移入本目录。
- 移入后 `config/arms/` 仅剩 `README.md` + `v7_arm1_offroad.yaml`（当前线采用配方，保留原位）。
