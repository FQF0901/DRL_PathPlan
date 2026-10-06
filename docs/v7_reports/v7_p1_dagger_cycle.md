# v7-P1 DAgger cycle（唯一一轮 IL cycle）

- 状态：**done**（最终） @ 2026-10-03T05:51:48
- git：`4a46c5ec0bed4e3547eae2c7f144772cb457d7b9`  stamp：20261002-2329
- 基座：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-0941_v7p1b/stage_b/ckpt_epoch010.pt`（sha `d977dec4bf5ffcc9…`）；学生/教师：pure_pursuit；tracker=lqr（plan 口径）
- 池：`env/specs/scenarios_train_5k.json`；锚：`datasets/BTC20261002-0941_expert5k_v4`（mild=1.0）；窗口：target-fails=1000 · 失败窗口 10s · shuffle seeds 20261021..20261024
- 训练：phase3 specific_only · epochs=15 · ckpt_every=5 · batch=1024 micro=256 · action_chain_source=labels · bias_calib={w1:0.2,w2:0.2,w3:0.2,w4:0.0}

## 0. 预注册判据（冻结）

- 命中：**任一窗口 top1** 的 clean500 配对（vs P1-B e010）Δ ≥ **+5pt** 且 （T3 S1 ≥ 1 或 tg45 success ≥ 0.1）
- 全阴性 ⇒ 宣布 **IL 侧封顶**（记录，转 RL 待编排决策）；eval500 只对最终采纳 candidate 评一次。

## 1. 预检（config/ckpt/数据/spec sha）

```json
{
 "config_files": {
  "config/default.yaml": {
   "sha256": "432b2d072ae50a4398e7d9e607d3bc155b270d4f040fa7405c7d83f27b1266ca",
   "bytes": 560
  },
  "config/train.yaml": {
   "sha256": "536191589ebeb5ecc9d4f8672055e64589b7c91f65dc56cf16f2a2abc9e5df45",
   "bytes": 22235
  },
  "config/model.yaml": {
   "sha256": "2fc4a9d6bfac6b6d788ada0d85f75cca67e9d405b739eee1d959bc1d556dce4c",
   "bytes": 2812
  },
  "config/env.yaml": {
   "sha256": "ba777b868d06f416df4186637f0af7bad4ce627ad77a8cee78112b4540a0014c",
   "bytes": 3414
  },
  "config/eval.yaml": {
   "sha256": "5ff76eb7bd01654300e2e7ec278e2ab67cc19c98b95656aa72fda419d7f24c1b",
   "bytes": 3223
  }
 },
 "refs": {
  "p1b_clean500": {
   "path": "runs/BTC20261002-162639_v7p1b_screen_epoch010/episodes.csv",
   "exists": true,
   "sha256": "f11de6476d879e788a153af104c49248bc587923c846c26e8b61bc4c9e0b18f3"
  },
  "p1b_eval500": {
   "path": "runs/BTC20261002-164209_v7p1b_eval500_sel/episodes.csv",
   "exists": true,
   "sha256": "d671cd66cfd45b500ac8f0a14d13d6d33aecf201ae5d9ede0fd2a1be96f91367"
  },
  "pool": {
   "path": "/workspace/01_Proj/DRL_PathPlan/env/specs/scenarios_train_5k.json",
   "exists": true,
   "is_file": true,
   "sha256": "211575f09675975c200a6bac73cd50090b7836a7eb3515bc93393780814fb91e"
  },
  "anchor": {
   "path": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert5k_v4",
   "exists": true,
   "is_file": false,
   "sha256": null
  },
  "clean500": {
   "path": "/tmp/opencode/phase3_diag/exp/specs_val_only500.json",
   "exists": true,
   "is_file": true,
   "sha256": "087db3f5c02e9cedccb8313733207f59dfa90162002b0d9bf002e5b80bcc366e"
  },
  "clean150": {
   "path": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "exists": true,
   "is_file": true,
   "sha256": "81f0f95814e3ea8bb8c349e47b2a19121ea78d578a59264fcc8bb1a660d4afa6"
  },
  "tg45": {
   "path": "/tmp/opencode/v7_p1b_chain/specs_tollgate45.json",
   "exists": true,
   "is_file": true,
   "sha256": "27010b0ebfb59d1daf45570d5851cf47452ab3d5eddbc9d9f3ddfc6d98c523e6"
  },
  "t3_measure": {
   "path": "/tmp/opencode/v7_p1b_chain/t3_measure.py",
   "exists": true,
   "is_file": true,
   "sha256": "537df2db5a4525c7fa5e6638c3f491e8548a99844797983db1bb8dcd5787ff71"
  }
 },
 "base_ckpt": {
  "path": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-0941_v7p1b/stage_b/ckpt_epoch010.pt",
  "exists": true,
  "sha256": "d977dec4bf5ffcc9b8a6f46dfc3c305aa05a802435745f0e62bd928c7a22ca63",
  "expected_sha256": "d977dec4bf5ffcc9b8a6f46dfc3c305aa05a802435745f0e62bd928c7a22ca63",
  "match": true
 },
 "anchor": {
  "count": 360561,
  "obs_schema_version": 4,
  "others": [
   1,
   33
  ],
  "npz_sha256": "688c90fba5c8535df77de1adf2644c244c1ca153282d9f53b72d424f715038b4"
 },
 "clean150_subset": {
  "n500": 500,
  "n150": 150,
  "subset": true,
  "prefix": true
 }
}
```

## 1b. 冒烟（20 spec + 1 步训练接线）

```json
{
 "collect": {
  "dir": "/tmp/opencode/v7_p1_dagger_cycle/smoke_ds",
  "count": 279,
  "label_stats": {
   "relabeled": 1141,
   "chain_rows": 1141,
   "full_chain_rows": 1041,
   "tail_missing_steps": 300
  },
  "nan_tail_rows": 70,
  "per_step_valid": [
   279,
   265,
   251,
   237,
   223,
   209
  ],
  "checks": {
   "schema_v4": true,
   "others_33": true,
   "action_6x2": true,
   "label_scope": true,
   "window_fill": true,
   "driver_sha": true,
   "nan_tail": true,
   "label_stats": true
  },
  "pass": true
 },
 "train": {
  "dir": "/tmp/opencode/v7_p1_dagger_cycle/smoke_train",
  "action_chain_source": "labels",
  "config_hash": "7b95501a40c3f78c",
  "bias_loss": 0.19462577998638153,
  "checks": {
   "chain_src_labels": true,
   "anchor_enabled": true,
   "bias_finite": true,
   "config_hash": true
  },
  "pass": true
 },
 "pass": true
}
```

## 2.1b 窗口 1 校验

```json
{
 "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w1",
 "npz_sha256": "d51bcfbab7b29b273bba7bbce9c1c4044c86f62a4ec312089fb5c600be0fd6bb",
 "meta_sha256": "5289e90e671fbe1962e14cba5f551301633aef5e0095231f51be1334d72d49fe",
 "count": 20071,
 "obs_schema_version": 4,
 "others_shape": [
  1,
  33
 ],
 "stored_rows": 20071,
 "trainable_rows": 20047,
 "step_yield": 0.9988042449304967,
 "scanned": 1480,
 "fails": 1006,
 "driver_ckpt": "d977dec4bf5ffcc9b8a6f46dfc3c305aa05a802435745f0e62bd928c7a22ca63",
 "label_scope": "action_chain_t0_t5",
 "window_fill": "teacher_chain_nan_tail",
 "label_stats": {
  "relabeled": 95442,
  "chain_rows": 95442,
  "full_chain_rows": 88042,
  "tail_missing_steps": 22200
 },
 "fail_terminations": null,
 "action_shape": [
  20071,
  6,
  2
 ],
 "others_actual": [
  20071,
  1,
  33
 ],
 "nan_tail_rows": 5030,
 "per_step_valid": [
  20071,
  19065,
  18059,
  17053,
  16047,
  15041
 ],
 "full_chain_rows": 15041,
 "chain_coverage_stored": 0.7494,
 "others_finite": true,
 "checks": {
  "schema_v4": true,
  "others_33": true,
  "action_chain_6x2": true,
  "label_scope": true,
  "window_fill": true,
  "driver_sha": true,
  "rows_pos": true,
  "yield_ge_0.6": true,
  "nan_tail_present": true,
  "others_finite": true
 },
 "pass": true
}
```

## 3.1 窗口 1 phase3 训练

```json
{
 "rc": 0,
 "duration_s": 2944.8,
 "micro": 256,
 "epoch_walls_s": [
  186.1,
  184.5,
  184.6,
  179.1,
  177.9,
  0.0,
  179.5,
  178.4,
  178.5,
  178.3,
  177.1,
  0.0,
  177.7,
  177.8,
  176.8,
  178.1,
  177.3,
  0.0
 ],
 "aborted": null,
 "oom": false,
 "log": "/tmp/opencode/v7_p1_dagger_cycle/train_w1.log",
 "out": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w1",
 "reused": false,
 "cmd": [
  "/workspace/01_Proj/DRL_PathPlan/tools/venv-python",
  "tools/train.py",
  "--phase3",
  "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w1",
  "--phase3-round",
  "1",
  "--ckpt",
  "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-0941_v7p1b/stage_b/ckpt_epoch010.pt",
  "--out",
  "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w1",
  "--config",
  "config/default.yaml",
  "--model-config",
  "config/model.yaml",
  "--phase3-epochs",
  "15",
  "--ckpt-every",
  "5",
  "--phase3-freeze",
  "specific_only",
  "--phase3-anchor",
  "--phase3-anchor-bc-dir",
  "datasets/BTC20261002-0941_expert5k_v4",
  "--phase3-anchor-mild-weight",
  "1.0",
  "--phase3-action-chain-source",
  "labels",
  "--phase3-bias-calib-weight",
  "0.2",
  "--batch-size",
  "1024",
  "--micro-batch-size",
  "256",
  "--device",
  "cuda",
  "--seed",
  "0"
 ],
 "config_hash": "7b95501a40c3f78c",
 "epochs": 15,
 "action_chain_source": {
  "requested": "labels",
  "train": "labels",
  "val": "same_as_train",
  "default": "auto",
  "semantics": "labels = 逐行 action[0:6] 教师链（采集侧 label_scope=action_chain_t0_t5）；lookup = 同 episode 未来行 action[0] 查表（旧行为）"
 },
 "anchor": {
  "dir": "datasets/BTC20261002-0941_expert5k_v4",
  "window_rows": 20071,
  "anchor_rows": 360561,
  "merged_rows": 380632,
  "mild_weight": 1.0,
  "train_rows": 377620
 },
 "bias_calib_weight": 0.2,
 "bias_final": {
  "bc_bias_action_mu_ds": -0.001223297298635516,
  "bc_bias_action_mu_dtheta": -0.0008816964872560239,
  "bc_bias_traj_lat_h1": -0.001390279728500156,
  "bc_bias_traj_lat_h6": -0.0006527686014315923,
  "bc_bias_calib_penalty": 0.008038600617070818,
  "bc_bias_calib_loss": 0.006421981524842367
 },
 "val_loss": null,
 "ckpt_sha": {
  "ckpt_epoch005.pt": "fdfe080818eb355b388692269454538eed8b5cc318a9eb4e73b206f8cbd688d9",
  "ckpt_epoch010.pt": "6fcefa6a92599578b2c25f87ba7bf7506d49769b42a4de6afaa64ebe24b581a7",
  "ckpt_epoch015.pt": "967df4afd208f3e7f3f0b92199ad8dfb150e5963e2d99ecf8320601561447f43",
  "final.pt": "0a67fccdb64f561df3529985be0cb8dad4182605571abc4597e6db6a203a9b4d"
 }
}
```

## 4.1 窗口 1 clean150 keep-best

```json
{
 "window": 1,
 "results": [
  {
   "tag": "v7p1dagger_w1_e005",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 182.4,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w1_e005.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-005306_v7p1dagger_w1_e005",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.52,
   "collision_rate": 0.07333333333333333,
   "off_road_rate": 0.3933333333333333,
   "episodes_csv": "runs/BTC20261003-005306_v7p1dagger_w1_e005/episodes.csv",
   "episodes_sha256": "0d1b14c10f6747f1c69143572c7aaa0291e5cc2e0f4ff18cae55dbfa73fb9edf",
   "sha256": "fdfe080818eb355b388692269454538eed8b5cc318a9eb4e73b206f8cbd688d9"
  },
  {
   "tag": "v7p1dagger_w1_e010",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch010.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 163.6,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w1_e010.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-005609_v7p1dagger_w1_e010",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.3933333333333333,
   "collision_rate": 0.04666666666666667,
   "off_road_rate": 0.56,
   "episodes_csv": "runs/BTC20261003-005609_v7p1dagger_w1_e010/episodes.csv",
   "episodes_sha256": "9b4aac74d4d16502fe73310337c3cf58eaadbe15287fe14ebd939fc6d4786a87",
   "sha256": "6fcefa6a92599578b2c25f87ba7bf7506d49769b42a4de6afaa64ebe24b581a7"
  },
  {
   "tag": "v7p1dagger_w1_e015",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch015.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 178.3,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w1_e015.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-005853_v7p1dagger_w1_e015",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.26666666666666666,
   "collision_rate": 0.04,
   "off_road_rate": 0.7,
   "episodes_csv": "runs/BTC20261003-005853_v7p1dagger_w1_e015/episodes.csv",
   "episodes_sha256": "b1266bf99c267e5a08223fd71977410f8c2603d96d62b77699353fdd90e7be5c",
   "sha256": "967df4afd208f3e7f3f0b92199ad8dfb150e5963e2d99ecf8320601561447f43"
  }
 ],
 "top1_tag": "v7p1dagger_w1_e005",
 "top1_ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt",
 "top1_clean150_sr": 0.52,
 "top1_sha256": "fdfe080818eb355b388692269454538eed8b5cc318a9eb4e73b206f8cbd688d9",
 "top1_run_dir": "runs/BTC20261003-005306_v7p1dagger_w1_e005",
 "rule": "clean150 success 最大；平局取更早 epoch"
}
```

## 2.2 窗口 2 采集

```json
{
 "rc": 0,
 "duration_s": 1218.5,
 "log": "/tmp/opencode/v7_p1_dagger_cycle/collect_w2.log",
 "reused": false,
 "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w2",
 "shuffle_seed": 20261022
}
```

## 2.2b 窗口 2 校验

```json
{
 "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w2",
 "npz_sha256": "7eb6d3f544bc0b07fc4b123c7cec6792fa571715f82263b9393be4c2c32f7aa2",
 "meta_sha256": "0e324d997a6ba9976eb6dd1673c5f639a202aca5fb433d957e0bf0d4f60dec77",
 "count": 20082,
 "obs_schema_version": 4,
 "others_shape": [
  1,
  33
 ],
 "stored_rows": 20082,
 "trainable_rows": 20016,
 "step_yield": 0.9967134747535106,
 "scanned": 1440,
 "fails": 1007,
 "driver_ckpt": "d977dec4bf5ffcc9b8a6f46dfc3c305aa05a802435745f0e62bd928c7a22ca63",
 "label_scope": "action_chain_t0_t5",
 "window_fill": "teacher_chain_nan_tail",
 "label_stats": {
  "relabeled": 90436,
  "chain_rows": 90436,
  "full_chain_rows": 83236,
  "tail_missing_steps": 21600
 },
 "fail_terminations": null,
 "action_shape": [
  20082,
  6,
  2
 ],
 "others_actual": [
  20082,
  1,
  33
 ],
 "nan_tail_rows": 5035,
 "per_step_valid": [
  20082,
  19075,
  18068,
  17061,
  16054,
  15047
 ],
 "full_chain_rows": 15047,
 "chain_coverage_stored": 0.7493,
 "others_finite": true,
 "checks": {
  "schema_v4": true,
  "others_33": true,
  "action_chain_6x2": true,
  "label_scope": true,
  "window_fill": true,
  "driver_sha": true,
  "rows_pos": true,
  "yield_ge_0.6": true,
  "nan_tail_present": true,
  "others_finite": true
 },
 "pass": true
}
```

## 3.2 窗口 2 phase3 训练

```json
{
 "rc": 0,
 "duration_s": 2954.5,
 "micro": 256,
 "epoch_walls_s": [
  191.4,
  185.9,
  178.0,
  178.4,
  178.1,
  0.0,
  179.2,
  180.2,
  177.3,
  177.9,
  178.4,
  0.0,
  177.2,
  178.2,
  177.6,
  177.0,
  178.3,
  0.0
 ],
 "aborted": null,
 "oom": false,
 "log": "/tmp/opencode/v7_p1_dagger_cycle/train_w2.log",
 "out": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w2",
 "reused": false,
 "cmd": [
  "/workspace/01_Proj/DRL_PathPlan/tools/venv-python",
  "tools/train.py",
  "--phase3",
  "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w2",
  "--phase3-round",
  "2",
  "--ckpt",
  "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-0941_v7p1b/stage_b/ckpt_epoch010.pt",
  "--out",
  "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w2",
  "--config",
  "config/default.yaml",
  "--model-config",
  "config/model.yaml",
  "--phase3-epochs",
  "15",
  "--ckpt-every",
  "5",
  "--phase3-freeze",
  "specific_only",
  "--phase3-anchor",
  "--phase3-anchor-bc-dir",
  "datasets/BTC20261002-0941_expert5k_v4",
  "--phase3-anchor-mild-weight",
  "1.0",
  "--phase3-action-chain-source",
  "labels",
  "--phase3-bias-calib-weight",
  "0.2",
  "--batch-size",
  "1024",
  "--micro-batch-size",
  "256",
  "--device",
  "cuda",
  "--seed",
  "0"
 ],
 "config_hash": "7b95501a40c3f78c",
 "epochs": 15,
 "action_chain_source": {
  "requested": "labels",
  "train": "labels",
  "val": "same_as_train",
  "default": "auto",
  "semantics": "labels = 逐行 action[0:6] 教师链（采集侧 label_scope=action_chain_t0_t5）；lookup = 同 episode 未来行 action[0] 查表（旧行为）"
 },
 "anchor": {
  "dir": "datasets/BTC20261002-0941_expert5k_v4",
  "window_rows": 20082,
  "anchor_rows": 360561,
  "merged_rows": 380643,
  "mild_weight": 1.0,
  "train_rows": 377632
 },
 "bias_calib_weight": 0.2,
 "bias_final": {
  "bc_bias_action_mu_ds": -0.0011759172447752743,
  "bc_bias_action_mu_dtheta": -0.0008288917757752449,
  "bc_bias_traj_lat_h1": -0.0008647895700025928,
  "bc_bias_traj_lat_h6": -0.0012734758219067027,
  "bc_bias_calib_penalty": 0.007431634895348144,
  "bc_bias_calib_loss": 0.005958801432368354
 },
 "val_loss": null,
 "ckpt_sha": {
  "ckpt_epoch005.pt": "67776c2a86515168a10d6ab38af037cc59bae849adc7367439c37e2f1b9fd644",
  "ckpt_epoch010.pt": "e299e6aa701a435ee297488d1c23b09244f4724d89f7f0dd9cb9aea2e56a58f5",
  "ckpt_epoch015.pt": "103571103c82341985e554724579e9cb61d1e5ef1b058167d6d3a84bff2a25f0",
  "final.pt": "debdb855db6dc91dad8349b29fdcbf763754a4d5fbec4a64fc3c48cbb34ba628"
 }
}
```

## 4.2 窗口 2 clean150 keep-best

```json
{
 "window": 2,
 "results": [
  {
   "tag": "v7p1dagger_w2_e005",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w2/ckpt_epoch005.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 165.9,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w2_e005.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-021124_v7p1dagger_w2_e005",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.38,
   "collision_rate": 0.06,
   "off_road_rate": 0.56,
   "episodes_csv": "runs/BTC20261003-021124_v7p1dagger_w2_e005/episodes.csv",
   "episodes_sha256": "d4c7b634a8952a5dca254c5e68d8e4f2cc2f5516b6727066d5c411a02e2201dd",
   "sha256": "67776c2a86515168a10d6ab38af037cc59bae849adc7367439c37e2f1b9fd644"
  },
  {
   "tag": "v7p1dagger_w2_e010",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w2/ckpt_epoch010.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 170.2,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w2_e010.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-021410_v7p1dagger_w2_e010",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.43333333333333335,
   "collision_rate": 0.04666666666666667,
   "off_road_rate": 0.5133333333333333,
   "episodes_csv": "runs/BTC20261003-021410_v7p1dagger_w2_e010/episodes.csv",
   "episodes_sha256": "9cd6dee3953d1e70a0b0c5b0d9441bbe85221b864374f37b245e4ba3dbcebc13",
   "sha256": "e299e6aa701a435ee297488d1c23b09244f4724d89f7f0dd9cb9aea2e56a58f5"
  },
  {
   "tag": "v7p1dagger_w2_e015",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w2/ckpt_epoch015.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 151.4,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w2_e015.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-021700_v7p1dagger_w2_e015",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.4,
   "collision_rate": 0.04666666666666667,
   "off_road_rate": 0.5533333333333333,
   "episodes_csv": "runs/BTC20261003-021700_v7p1dagger_w2_e015/episodes.csv",
   "episodes_sha256": "2284254308b2ed5949fe46d4c0192b003eb1d28d5053d9baedd99d598044028b",
   "sha256": "103571103c82341985e554724579e9cb61d1e5ef1b058167d6d3a84bff2a25f0"
  }
 ],
 "top1_tag": "v7p1dagger_w2_e010",
 "top1_ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w2/ckpt_epoch010.pt",
 "top1_clean150_sr": 0.43333333333333335,
 "top1_sha256": "e299e6aa701a435ee297488d1c23b09244f4724d89f7f0dd9cb9aea2e56a58f5",
 "top1_run_dir": "runs/BTC20261003-021410_v7p1dagger_w2_e010",
 "rule": "clean150 success 最大；平局取更早 epoch"
}
```

## 2.3 窗口 3 采集

```json
{
 "rc": 0,
 "duration_s": 1336.6,
 "log": "/tmp/opencode/v7_p1_dagger_cycle/collect_w3.log",
 "reused": false,
 "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w3",
 "shuffle_seed": 20261023
}
```

## 2.3b 窗口 3 校验

```json
{
 "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w3",
 "npz_sha256": "e557ac6a6aa431f7640df06d762cb1aa98b896424458c384fca20423b5a63341",
 "meta_sha256": "82a39aa1c674d63c895270fd4c56d313c9db66dc2a1ffc74dfe532122ff36574",
 "count": 20055,
 "obs_schema_version": 4,
 "others_shape": [
  1,
  33
 ],
 "stored_rows": 20055,
 "trainable_rows": 20026,
 "step_yield": 0.9985539765644478,
 "scanned": 1520,
 "fails": 1005,
 "driver_ckpt": "d977dec4bf5ffcc9b8a6f46dfc3c305aa05a802435745f0e62bd928c7a22ca63",
 "label_scope": "action_chain_t0_t5",
 "window_fill": "teacher_chain_nan_tail",
 "label_stats": {
  "relabeled": 98290,
  "chain_rows": 98290,
  "full_chain_rows": 90694,
  "tail_missing_steps": 22790
 },
 "fail_terminations": null,
 "action_shape": [
  20055,
  6,
  2
 ],
 "others_actual": [
  20055,
  1,
  33
 ],
 "nan_tail_rows": 5025,
 "per_step_valid": [
  20055,
  19050,
  18045,
  17040,
  16035,
  15030
 ],
 "full_chain_rows": 15030,
 "chain_coverage_stored": 0.7494,
 "others_finite": true,
 "checks": {
  "schema_v4": true,
  "others_33": true,
  "action_chain_6x2": true,
  "label_scope": true,
  "window_fill": true,
  "driver_sha": true,
  "rows_pos": true,
  "yield_ge_0.6": true,
  "nan_tail_present": true,
  "others_finite": true
 },
 "pass": true
}
```

## 3.3 窗口 3 phase3 训练

```json
{
 "rc": 0,
 "duration_s": 2989.0,
 "micro": 256,
 "epoch_walls_s": [
  218.5,
  185.2,
  180.0,
  178.8,
  179.4,
  0.0,
  180.2,
  180.7,
  179.7,
  179.0,
  179.1,
  0.0,
  178.0,
  179.1,
  178.6,
  178.1,
  178.1,
  0.0
 ],
 "aborted": null,
 "oom": false,
 "log": "/tmp/opencode/v7_p1_dagger_cycle/train_w3.log",
 "out": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w3",
 "reused": false,
 "cmd": [
  "/workspace/01_Proj/DRL_PathPlan/tools/venv-python",
  "tools/train.py",
  "--phase3",
  "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w3",
  "--phase3-round",
  "3",
  "--ckpt",
  "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-0941_v7p1b/stage_b/ckpt_epoch010.pt",
  "--out",
  "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w3",
  "--config",
  "config/default.yaml",
  "--model-config",
  "config/model.yaml",
  "--phase3-epochs",
  "15",
  "--ckpt-every",
  "5",
  "--phase3-freeze",
  "specific_only",
  "--phase3-anchor",
  "--phase3-anchor-bc-dir",
  "datasets/BTC20261002-0941_expert5k_v4",
  "--phase3-anchor-mild-weight",
  "1.0",
  "--phase3-action-chain-source",
  "labels",
  "--phase3-bias-calib-weight",
  "0.2",
  "--batch-size",
  "1024",
  "--micro-batch-size",
  "256",
  "--device",
  "cuda",
  "--seed",
  "0"
 ],
 "config_hash": "7b95501a40c3f78c",
 "epochs": 15,
 "action_chain_source": {
  "requested": "labels",
  "train": "labels",
  "val": "same_as_train",
  "default": "auto",
  "semantics": "labels = 逐行 action[0:6] 教师链（采集侧 label_scope=action_chain_t0_t5）；lookup = 同 episode 未来行 action[0] 查表（旧行为）"
 },
 "anchor": {
  "dir": "datasets/BTC20261002-0941_expert5k_v4",
  "window_rows": 20055,
  "anchor_rows": 360561,
  "merged_rows": 380616,
  "mild_weight": 1.0,
  "train_rows": 377600
 },
 "bias_calib_weight": 0.2,
 "bias_final": {
  "bc_bias_action_mu_ds": -0.0013722261856457076,
  "bc_bias_action_mu_dtheta": -0.0010504427856146811,
  "bc_bias_traj_lat_h1": -0.0010934469575461963,
  "bc_bias_traj_lat_h6": -0.0012226883556795752,
  "bc_bias_calib_penalty": 0.007430882066663874,
  "bc_bias_calib_loss": 0.005921256709863888
 },
 "val_loss": null,
 "ckpt_sha": {
  "ckpt_epoch005.pt": "17995acd1b413244ca1aaf802066fd1ef25bf0d30a61c183bf199e123970ba10",
  "ckpt_epoch010.pt": "cbb5775886eb1846e4b9dc82b5016209697f8eb6a2b6fdb126b224e0d6ca8eb1",
  "ckpt_epoch015.pt": "4f73b1a8d5a734b34b6097a9b30bded0975328cedbbb409023637ab2e317db4c",
  "final.pt": "501e9cf2178095fb49765c8c04e4d3e0c66e8c3f8d9e6c38a4b47a0db81f75dd"
 }
}
```

## 4.3 窗口 3 clean150 keep-best

```json
{
 "window": 3,
 "results": [
  {
   "tag": "v7p1dagger_w3_e005",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w3/ckpt_epoch005.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 196.2,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w3_e005.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-033138_v7p1dagger_w3_e005",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.46,
   "collision_rate": 0.08,
   "off_road_rate": 0.4666666666666667,
   "episodes_csv": "runs/BTC20261003-033138_v7p1dagger_w3_e005/episodes.csv",
   "episodes_sha256": "6bd3926b6005787f13033ac1e1a2cf4d0dd9fb540257a29d0df159d6489a2df1",
   "sha256": "17995acd1b413244ca1aaf802066fd1ef25bf0d30a61c183bf199e123970ba10"
  },
  {
   "tag": "v7p1dagger_w3_e010",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w3/ckpt_epoch010.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 180.6,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w3_e010.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-033454_v7p1dagger_w3_e010",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.38666666666666666,
   "collision_rate": 0.03333333333333333,
   "off_road_rate": 0.5733333333333334,
   "episodes_csv": "runs/BTC20261003-033454_v7p1dagger_w3_e010/episodes.csv",
   "episodes_sha256": "1d1d75d402ec1165d03fc8e0527ca00b3811e70accfdad0746fd18eb54e4dcec",
   "sha256": "cbb5775886eb1846e4b9dc82b5016209697f8eb6a2b6fdb126b224e0d6ca8eb1"
  },
  {
   "tag": "v7p1dagger_w3_e015",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w3/ckpt_epoch015.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 161.8,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w3_e015.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-033755_v7p1dagger_w3_e015",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.20666666666666667,
   "collision_rate": 0.04666666666666667,
   "off_road_rate": 0.7333333333333333,
   "episodes_csv": "runs/BTC20261003-033755_v7p1dagger_w3_e015/episodes.csv",
   "episodes_sha256": "0da02d5071db9def39c28e41a4da5358c4b018d51dc55b1d0083c14f9d302e20",
   "sha256": "4f73b1a8d5a734b34b6097a9b30bded0975328cedbbb409023637ab2e317db4c"
  }
 ],
 "top1_tag": "v7p1dagger_w3_e005",
 "top1_ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w3/ckpt_epoch005.pt",
 "top1_clean150_sr": 0.46,
 "top1_sha256": "17995acd1b413244ca1aaf802066fd1ef25bf0d30a61c183bf199e123970ba10",
 "top1_run_dir": "runs/BTC20261003-033138_v7p1dagger_w3_e005",
 "rule": "clean150 success 最大；平局取更早 epoch"
}
```

## 2.4 窗口 4 采集

```json
{
 "rc": 0,
 "duration_s": 1234.8,
 "log": "/tmp/opencode/v7_p1_dagger_cycle/collect_w4.log",
 "reused": false,
 "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w4",
 "shuffle_seed": 20261024
}
```

## 2.4b 窗口 4 校验

```json
{
 "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w4",
 "npz_sha256": "e0a42d84ef74bcb456a51926b6a84ca550804bf1695029f2c0fc1d226d3d9c9d",
 "meta_sha256": "591465b7b5e984c73094877571a4d4824479497146f58f8e13862f72f21ee821",
 "count": 20049,
 "obs_schema_version": 4,
 "others_shape": [
  1,
  33
 ],
 "stored_rows": 20049,
 "trainable_rows": 20031,
 "step_yield": 0.9991021996109531,
 "scanned": 1460,
 "fails": 1005,
 "driver_ckpt": "d977dec4bf5ffcc9b8a6f46dfc3c305aa05a802435745f0e62bd928c7a22ca63",
 "label_scope": "action_chain_t0_t5",
 "window_fill": "teacher_chain_nan_tail",
 "label_stats": {
  "relabeled": 91038,
  "chain_rows": 91038,
  "full_chain_rows": 83738,
  "tail_missing_steps": 21900
 },
 "fail_terminations": null,
 "action_shape": [
  20049,
  6,
  2
 ],
 "others_actual": [
  20049,
  1,
  33
 ],
 "nan_tail_rows": 5025,
 "per_step_valid": [
  20049,
  19044,
  18039,
  17034,
  16029,
  15024
 ],
 "full_chain_rows": 15024,
 "chain_coverage_stored": 0.7494,
 "others_finite": true,
 "checks": {
  "schema_v4": true,
  "others_33": true,
  "action_chain_6x2": true,
  "label_scope": true,
  "window_fill": true,
  "driver_sha": true,
  "rows_pos": true,
  "yield_ge_0.6": true,
  "nan_tail_present": true,
  "others_finite": true
 },
 "pass": true
}
```

## 3.4 窗口 4 phase3 训练

```json
{
 "rc": 0,
 "duration_s": 2937.6,
 "micro": 256,
 "epoch_walls_s": [
  217.7,
  182.3,
  175.1,
  175.1,
  174.5,
  0.0,
  176.9,
  177.6,
  174.6,
  174.1,
  174.0,
  0.0,
  174.0,
  174.0,
  174.1,
  174.0,
  174.2,
  0.0
 ],
 "aborted": null,
 "oom": false,
 "log": "/tmp/opencode/v7_p1_dagger_cycle/train_w4.log",
 "out": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4",
 "reused": false,
 "cmd": [
  "/workspace/01_Proj/DRL_PathPlan/tools/venv-python",
  "tools/train.py",
  "--phase3",
  "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-2329_v7p1dagger_w4",
  "--phase3-round",
  "4",
  "--ckpt",
  "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-0941_v7p1b/stage_b/ckpt_epoch010.pt",
  "--out",
  "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4",
  "--config",
  "config/default.yaml",
  "--model-config",
  "config/model.yaml",
  "--phase3-epochs",
  "15",
  "--ckpt-every",
  "5",
  "--phase3-freeze",
  "specific_only",
  "--phase3-anchor",
  "--phase3-anchor-bc-dir",
  "datasets/BTC20261002-0941_expert5k_v4",
  "--phase3-anchor-mild-weight",
  "1.0",
  "--phase3-action-chain-source",
  "labels",
  "--phase3-bias-calib-weight",
  "0",
  "--batch-size",
  "1024",
  "--micro-batch-size",
  "256",
  "--device",
  "cuda",
  "--seed",
  "0"
 ],
 "config_hash": "7b95501a40c3f78c",
 "epochs": 15,
 "action_chain_source": {
  "requested": "labels",
  "train": "labels",
  "val": "same_as_train",
  "default": "auto",
  "semantics": "labels = 逐行 action[0:6] 教师链（采集侧 label_scope=action_chain_t0_t5）；lookup = 同 episode 未来行 action[0] 查表（旧行为）"
 },
 "anchor": {
  "dir": "datasets/BTC20261002-0941_expert5k_v4",
  "window_rows": 20049,
  "anchor_rows": 360561,
  "merged_rows": 380610,
  "mild_weight": 1.0,
  "train_rows": 377603
 },
 "bias_calib_weight": 0.0,
 "bias_final": {
  "bc_bias_action_mu_ds": 0.0004914712450778432,
  "bc_bias_action_mu_dtheta": 2.2638313716287793e-05,
  "bc_bias_traj_lat_h1": 6.762835566974844e-05,
  "bc_bias_traj_lat_h6": 0.09397595414990248,
  "bc_bias_calib_penalty": 0.0,
  "bc_bias_calib_loss": 0.0
 },
 "val_loss": null,
 "ckpt_sha": {
  "ckpt_epoch005.pt": "47e5f1c0d37dae4688e9da23b2c4723d9019d026c1caf5190d2a931638182bdf",
  "ckpt_epoch010.pt": "07a4612b764f5b674ab751f5618b49b44284cbafeb5ad34aaec318f284b3288a",
  "ckpt_epoch015.pt": "af9ad551c78fabc2dc96c748ea8906e3b1d5d016a1845fe922e37534fef03592",
  "final.pt": "77f3d5cb1cf53f6f31c0699b9326785cb971e7f4287b14789341502beb8982bd"
 }
}
```

## 4.4 窗口 4 clean150 keep-best

```json
{
 "window": 4,
 "results": [
  {
   "tag": "v7p1dagger_w4_e005",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch005.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 170.1,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w4_e005.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-045010_v7p1dagger_w4_e005",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.3933333333333333,
   "collision_rate": 0.06,
   "off_road_rate": 0.54,
   "episodes_csv": "runs/BTC20261003-045010_v7p1dagger_w4_e005/episodes.csv",
   "episodes_sha256": "68ffcca4be33cd0059c3478e86b34e07aeaebd14afd034e45e545118f2f4eb4b",
   "sha256": "47e5f1c0d37dae4688e9da23b2c4723d9019d026c1caf5190d2a931638182bdf"
  },
  {
   "tag": "v7p1dagger_w4_e010",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch010.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 188.7,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w4_e010.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-045300_v7p1dagger_w4_e010",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.35333333333333333,
   "collision_rate": 0.06666666666666667,
   "off_road_rate": 0.5533333333333333,
   "episodes_csv": "runs/BTC20261003-045300_v7p1dagger_w4_e010/episodes.csv",
   "episodes_sha256": "b0f862455e06f41197f2d79031f01768354a1bba252b7fb3f24917fb7200355d",
   "sha256": "07a4612b764f5b674ab751f5618b49b44284cbafeb5ad34aaec318f284b3288a"
  },
  {
   "tag": "v7p1dagger_w4_e015",
   "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only150.json",
   "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch015.pt",
   "policy": "ckpt",
   "rc": 0,
   "duration_s": 183.7,
   "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w4_e015.log",
   "reused": false,
   "run_dir": "runs/BTC20261003-045608_v7p1dagger_w4_e015",
   "n": 150,
   "n_error": 0,
   "success_rate": 0.3466666666666667,
   "collision_rate": 0.02666666666666667,
   "off_road_rate": 0.5933333333333334,
   "episodes_csv": "runs/BTC20261003-045608_v7p1dagger_w4_e015/episodes.csv",
   "episodes_sha256": "6de4334c35d3ad188afb0c02e838c5e18d3c6f8e8270994c8b371553a50303bd",
   "sha256": "af9ad551c78fabc2dc96c748ea8906e3b1d5d016a1845fe922e37534fef03592"
  }
 ],
 "top1_tag": "v7p1dagger_w4_e005",
 "top1_ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch005.pt",
 "top1_clean150_sr": 0.3933333333333333,
 "top1_sha256": "47e5f1c0d37dae4688e9da23b2c4723d9019d026c1caf5190d2a931638182bdf",
 "top1_run_dir": "runs/BTC20261003-045010_v7p1dagger_w4_e005",
 "rule": "clean150 success 最大；平局取更早 epoch"
}
```

## 5.1 窗口 1 top1 全量评测 + 配对

```json
{
 "window": 1,
 "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt",
 "clean500": {
  "tag": "v7p1dagger_w1_clean500",
  "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only500.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 583.2,
  "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w1_clean500.log",
  "reused": false,
  "run_dir": "runs/BTC20261003-045912_v7p1dagger_w1_clean500",
  "n": 500,
  "n_error": 0,
  "success_rate": 0.526,
  "collision_rate": 0.066,
  "off_road_rate": 0.398,
  "episodes_csv": "runs/BTC20261003-045912_v7p1dagger_w1_clean500/episodes.csv",
  "episodes_sha256": "042b63304b9c8504c7ee55f8c3c469b8f11f1fe3c14eacc891c16e6415779a72"
 },
 "tg45": {
  "tag": "v7p1dagger_w1_tg45",
  "spec": "/tmp/opencode/v7_p1b_chain/specs_tollgate45.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w1/ckpt_epoch005.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 40.0,
  "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w1_tg45.log",
  "reused": false,
  "run_dir": "runs/BTC20261003-050855_v7p1dagger_w1_tg45",
  "n": 45,
  "n_error": 0,
  "success_rate": 0.0,
  "collision_rate": 0.06666666666666667,
  "off_road_rate": 0.9333333333333333,
  "episodes_csv": "runs/BTC20261003-050855_v7p1dagger_w1_tg45/episodes.csv",
  "episodes_sha256": "56c70f5ac77e2834ceaf11e598ccae7e0493825c5b2ed022716f0b87dd1f2a2b"
 },
 "t3": {
  "n_ids": 9,
  "n_s1_resolved": 0,
  "rows": [
   {
    "id": 34,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 201.274,
    "lane_tail": [
     [
      ">>>",
      "1C0_0_",
      "1"
     ],
     [
      ">>>",
      "1C0_0_",
      "2"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 243,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 3.01,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "0"
     ],
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 164,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.75,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 12.874,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "1"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 84,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.738,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 3.441,
    "lane_tail": [
     [
      "1S0_0_",
      "2S0_0_",
      "1"
     ],
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 166,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.741,
    "moved_free_lane": true,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 2.626,
    "lane_tail": [
     [
      "1C0_1_",
      "2S0_0_",
      "0"
     ],
     [
      "2S0_0_",
      "3$0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 131,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.743,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 4.874,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "1"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 76,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 210.26,
    "lane_tail": [
     [
      ">>>",
      "1C0_0_",
      "1"
     ],
     [
      ">>>",
      "1C0_0_",
      "2"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 147,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.744,
    "moved_free_lane": true,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 2.505,
    "lane_tail": [
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ],
     [
      "2S0_0_",
      "3$0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 239,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.749,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 20.266,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "1"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   }
  ],
  "json": "/tmp/opencode/v7_p1_dagger_cycle/t3_w1.json",
  "elapsed_s": 33.8,
  "rc": 0,
  "wall_s": 34.4
 },
 "paired_clean500_vs_p1b": {
  "tag": "w1_clean500_vs_p1b",
  "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-162639_v7p1b_screen_epoch010/episodes.csv",
  "agent": "runs/BTC20261003-045912_v7p1dagger_w1_clean500/episodes.csv",
  "rc": 0,
  "allow_mismatch": false,
  "out_dir": "/tmp/opencode/v7_p1_dagger_cycle/paired_eval/w1_clean500_vs_p1b",
  "log": "/tmp/opencode/v7_p1_dagger_cycle/paired_w1_clean500_vs_p1b.log",
  "n_paired": 500,
  "rate_base": 0.314,
  "rate_agent": 0.526,
  "delta_pp": 21.2,
  "ci95_pp": [
   17.599999999999998,
   24.8
  ],
  "net": 106,
  "z": 10.106703446003278,
  "fixed": 108,
  "broken": 2,
  "both_pass": 155,
  "both_fail": 235,
  "mcnemar_exact_p": 9.40778259234277e-30,
  "verdict": {
   "delta_pp": 21.2,
   "z": 10.106703446003278,
   "mcnemar_exact_p": 9.40778259234277e-30,
   "ci95_pp": [
    17.599999999999998,
    24.8
   ],
   "ci_lower_gt_zero": true,
   "positive_3pt_z196": true
  },
  "pair_meta": {
   "n_common": 500,
   "n_baseline_only": 0,
   "n_agent_only": 0,
   "baseline_only_examples": [],
   "agent_only_examples": [],
   "allow_mismatch": false
  }
 },
 "hit": false,
 "hit_detail": {
  "delta_pp": 21.2,
  "delta_ge_5pt": true,
  "t3_s1": 0,
  "tg45_sr": 0.0
 }
}
```

## 5.2 窗口 2 top1 全量评测 + 配对

```json
{
 "window": 2,
 "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w2/ckpt_epoch010.pt",
 "clean500": {
  "tag": "v7p1dagger_w2_clean500",
  "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only500.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w2/ckpt_epoch010.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 539.8,
  "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w2_clean500.log",
  "reused": false,
  "run_dir": "runs/BTC20261003-051010_v7p1dagger_w2_clean500",
  "n": 500,
  "n_error": 0,
  "success_rate": 0.428,
  "collision_rate": 0.05,
  "off_road_rate": 0.504,
  "episodes_csv": "runs/BTC20261003-051010_v7p1dagger_w2_clean500/episodes.csv",
  "episodes_sha256": "89dc8855083cb17331f32888bcdbc35775a2e68b909c1bb41af258e85123e610"
 },
 "tg45": {
  "tag": "v7p1dagger_w2_tg45",
  "spec": "/tmp/opencode/v7_p1b_chain/specs_tollgate45.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w2/ckpt_epoch010.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 39.1,
  "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w2_tg45.log",
  "reused": false,
  "run_dir": "runs/BTC20261003-051910_v7p1dagger_w2_tg45",
  "n": 45,
  "n_error": 0,
  "success_rate": 0.0,
  "collision_rate": 0.044444444444444446,
  "off_road_rate": 0.9555555555555556,
  "episodes_csv": "runs/BTC20261003-051910_v7p1dagger_w2_tg45/episodes.csv",
  "episodes_sha256": "ea40029fad5866c01124cd7887ffb4067cd35020f44c7175f024e5fe0e377954"
 },
 "t3": {
  "n_ids": 9,
  "n_s1_resolved": 0,
  "rows": [
   {
    "id": 34,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 202.701,
    "lane_tail": [
     [
      ">>>",
      "1C0_0_",
      "1"
     ],
     [
      ">>>",
      "1C0_0_",
      "2"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 243,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 7.299,
    "lane_tail": [
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ],
     [
      "1S0_0_",
      "2S0_0_",
      "1"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 164,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.75,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 7.917,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "1"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 84,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.738,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 2.621,
    "lane_tail": [
     [
      "1S0_0_",
      "2S0_0_",
      "1"
     ],
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 166,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.741,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 3.975,
    "lane_tail": [
     [
      "1C0_1_",
      "2S0_0_",
      "1"
     ],
     [
      "1C0_1_",
      "2S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 131,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.743,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 4.575,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "1"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 76,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 213.936,
    "lane_tail": [
     [
      ">>>",
      "1C0_0_",
      "1"
     ],
     [
      ">>>",
      "1C0_0_",
      "2"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 147,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.744,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 7.581,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "0"
     ],
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 239,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.749,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 20.175,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "0"
     ],
     [
      ">>>",
      "1S0_0_",
      "1"
     ]
    ],
    "ckpt_error": null
   }
  ],
  "json": "/tmp/opencode/v7_p1_dagger_cycle/t3_w2.json",
  "elapsed_s": 34.1,
  "rc": 0,
  "wall_s": 34.7
 },
 "paired_clean500_vs_p1b": {
  "tag": "w2_clean500_vs_p1b",
  "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-162639_v7p1b_screen_epoch010/episodes.csv",
  "agent": "runs/BTC20261003-051010_v7p1dagger_w2_clean500/episodes.csv",
  "rc": 0,
  "allow_mismatch": false,
  "out_dir": "/tmp/opencode/v7_p1_dagger_cycle/paired_eval/w2_clean500_vs_p1b",
  "log": "/tmp/opencode/v7_p1_dagger_cycle/paired_w2_clean500_vs_p1b.log",
  "n_paired": 500,
  "rate_base": 0.314,
  "rate_agent": 0.428,
  "delta_pp": 11.4,
  "ci95_pp": [
   8.0,
   14.799999999999999
  ],
  "net": 57,
  "z": 6.333333333333333,
  "fixed": 69,
  "broken": 12,
  "both_pass": 145,
  "both_fail": 274,
  "mcnemar_exact_p": 7.03300083476893e-11,
  "verdict": {
   "delta_pp": 11.4,
   "z": 6.333333333333333,
   "mcnemar_exact_p": 7.03300083476893e-11,
   "ci95_pp": [
    8.0,
    14.799999999999999
   ],
   "ci_lower_gt_zero": true,
   "positive_3pt_z196": true
  },
  "pair_meta": {
   "n_common": 500,
   "n_baseline_only": 0,
   "n_agent_only": 0,
   "baseline_only_examples": [],
   "agent_only_examples": [],
   "allow_mismatch": false
  }
 },
 "hit": false,
 "hit_detail": {
  "delta_pp": 11.4,
  "delta_ge_5pt": true,
  "t3_s1": 0,
  "tg45_sr": 0.0
 }
}
```

## 5.3 窗口 3 top1 全量评测 + 配对

```json
{
 "window": 3,
 "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w3/ckpt_epoch005.pt",
 "clean500": {
  "tag": "v7p1dagger_w3_clean500",
  "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only500.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w3/ckpt_epoch005.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 618.8,
  "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w3_clean500.log",
  "reused": false,
  "run_dir": "runs/BTC20261003-052024_v7p1dagger_w3_clean500",
  "n": 500,
  "n_error": 0,
  "success_rate": 0.472,
  "collision_rate": 0.058,
  "off_road_rate": 0.458,
  "episodes_csv": "runs/BTC20261003-052024_v7p1dagger_w3_clean500/episodes.csv",
  "episodes_sha256": "90e0759e9eb1de848dcb335d376a032254dfba57934c608988622b2d609403a3"
 },
 "tg45": {
  "tag": "v7p1dagger_w3_tg45",
  "spec": "/tmp/opencode/v7_p1b_chain/specs_tollgate45.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w3/ckpt_epoch005.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 47.1,
  "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w3_tg45.log",
  "reused": false,
  "run_dir": "runs/BTC20261003-053043_v7p1dagger_w3_tg45",
  "n": 45,
  "n_error": 0,
  "success_rate": 0.022222222222222223,
  "collision_rate": 0.044444444444444446,
  "off_road_rate": 0.9555555555555556,
  "episodes_csv": "runs/BTC20261003-053043_v7p1dagger_w3_tg45/episodes.csv",
  "episodes_sha256": "b97952b23029a6fa576ba71ea99918b531c9482509c051051dadce66e7f7305f"
 },
 "t3": {
  "n_ids": 9,
  "n_s1_resolved": 0,
  "rows": [
   {
    "id": 34,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 166.938,
    "lane_tail": [
     [
      ">>>",
      "1C0_0_",
      "1"
     ],
     [
      ">>>",
      "1C0_0_",
      "2"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 243,
    "ckpt_term": "collision",
    "base_term": "arrive_dest",
    "ckpt_clearance": 1.082,
    "base_clearance": 1.74,
    "moved_free_lane": true,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 2.622,
    "lane_tail": [
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ],
     [
      "2S0_0_",
      "3$0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 164,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.75,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 13.096,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "1"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 84,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.738,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 1.971,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "0"
     ],
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 166,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.741,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 9.763,
    "lane_tail": [
     [
      "1C0_0_",
      "1C0_1_",
      "0"
     ],
     [
      "1C0_1_",
      "2S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 131,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.743,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 3.69,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "1"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 76,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 164.562,
    "lane_tail": [
     [
      ">>>",
      "1C0_0_",
      "1"
     ],
     [
      ">>>",
      "1C0_0_",
      "2"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 147,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.744,
    "moved_free_lane": true,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 2.117,
    "lane_tail": [
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ],
     [
      "2S0_0_",
      "3$0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 239,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.749,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 17.904,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "1"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   }
  ],
  "json": "/tmp/opencode/v7_p1_dagger_cycle/t3_w3.json",
  "elapsed_s": 36.6,
  "rc": 0,
  "wall_s": 37.3
 },
 "paired_clean500_vs_p1b": {
  "tag": "w3_clean500_vs_p1b",
  "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-162639_v7p1b_screen_epoch010/episodes.csv",
  "agent": "runs/BTC20261003-052024_v7p1dagger_w3_clean500/episodes.csv",
  "rc": 0,
  "allow_mismatch": false,
  "out_dir": "/tmp/opencode/v7_p1_dagger_cycle/paired_eval/w3_clean500_vs_p1b",
  "log": "/tmp/opencode/v7_p1_dagger_cycle/paired_w3_clean500_vs_p1b.log",
  "n_paired": 500,
  "rate_base": 0.314,
  "rate_agent": 0.472,
  "delta_pp": 15.8,
  "ci95_pp": [
   12.2,
   19.6
  ],
  "net": 79,
  "z": 7.784101297497916,
  "fixed": 91,
  "broken": 12,
  "both_pass": 145,
  "both_fail": 252,
  "mcnemar_exact_p": 3.4610428636474895e-16,
  "verdict": {
   "delta_pp": 15.8,
   "z": 7.784101297497916,
   "mcnemar_exact_p": 3.4610428636474895e-16,
   "ci95_pp": [
    12.2,
    19.6
   ],
   "ci_lower_gt_zero": true,
   "positive_3pt_z196": true
  },
  "pair_meta": {
   "n_common": 500,
   "n_baseline_only": 0,
   "n_agent_only": 0,
   "baseline_only_examples": [],
   "agent_only_examples": [],
   "allow_mismatch": false
  }
 },
 "hit": false,
 "hit_detail": {
  "delta_pp": 15.8,
  "delta_ge_5pt": true,
  "t3_s1": 0,
  "tg45_sr": 0.022222222222222223
 }
}
```

## 5.4 窗口 4 top1 全量评测 + 配对

```json
{
 "window": 4,
 "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch005.pt",
 "clean500": {
  "tag": "v7p1dagger_w4_clean500",
  "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only500.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch005.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 541.6,
  "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w4_clean500.log",
  "reused": false,
  "run_dir": "runs/BTC20261003-053208_v7p1dagger_w4_clean500",
  "n": 500,
  "n_error": 0,
  "success_rate": 0.378,
  "collision_rate": 0.058,
  "off_road_rate": 0.556,
  "episodes_csv": "runs/BTC20261003-053208_v7p1dagger_w4_clean500/episodes.csv",
  "episodes_sha256": "8832bf24759c1d3c35017b59914767b5c4ce4265ff81d511a36023ea4e2d14f6"
 },
 "tg45": {
  "tag": "v7p1dagger_w4_tg45",
  "spec": "/tmp/opencode/v7_p1b_chain/specs_tollgate45.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch005.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 48.5,
  "log": "/tmp/opencode/v7_p1_dagger_cycle/eval_v7p1dagger_w4_tg45.log",
  "reused": false,
  "run_dir": "runs/BTC20261003-054109_v7p1dagger_w4_tg45",
  "n": 45,
  "n_error": 0,
  "success_rate": 0.06666666666666667,
  "collision_rate": 0.15555555555555556,
  "off_road_rate": 0.8,
  "episodes_csv": "runs/BTC20261003-054109_v7p1dagger_w4_tg45/episodes.csv",
  "episodes_sha256": "ac932277aed443c40e1f72f3fcdbc1511818bf1718045cb74c06a2c48e9ffca9"
 },
 "t3": {
  "n_ids": 9,
  "n_s1_resolved": 1,
  "rows": [
   {
    "id": 34,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": true,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 2.158,
    "lane_tail": [
     [
      "1C0_1_",
      "2S0_0_",
      "0"
     ],
     [
      "2S0_0_",
      "3$0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 243,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 103.636,
    "lane_tail": [
     [
      ">>",
      ">>>",
      "0"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 164,
    "ckpt_term": "arrive_dest",
    "base_term": "arrive_dest",
    "ckpt_clearance": 1.625,
    "base_clearance": 1.75,
    "moved_free_lane": true,
    "s1_resolved": true,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 3.251,
    "lane_tail": [
     [
      "2$0_0_",
      "3S0_0_",
      "0"
     ],
     [
      "2$0_0_",
      "3S0_0_",
      "1"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 84,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.738,
    "moved_free_lane": true,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 1.829,
    "lane_tail": [
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ],
     [
      "2S0_0_",
      "3$0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 166,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.741,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 43.411,
    "lane_tail": [
     [
      "1C0_0_",
      "1C0_1_",
      "0"
     ],
     [
      "1C0_1_",
      "2S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 131,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.743,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 3.909,
    "lane_tail": [
     [
      ">>",
      ">>>",
      "0"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 76,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.74,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 203.918,
    "lane_tail": [
     [
      ">>>",
      "1C0_0_",
      "1"
     ],
     [
      ">>>",
      "1C0_0_",
      "2"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 147,
    "ckpt_term": "out_of_road",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.744,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 38.722,
    "lane_tail": [
     [
      ">>>",
      "1S0_0_",
      "0"
     ],
     [
      "1S0_0_",
      "2S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   },
   {
    "id": 239,
    "ckpt_term": "collision",
    "base_term": "arrive_dest",
    "ckpt_clearance": null,
    "base_clearance": 1.749,
    "moved_free_lane": false,
    "s1_resolved": false,
    "plan_crosses_booth": false,
    "plan_min_dist_to_booth_m": 18.358,
    "lane_tail": [
     [
      ">>",
      ">>>",
      "0"
     ],
     [
      ">>>",
      "1S0_0_",
      "0"
     ]
    ],
    "ckpt_error": null
   }
  ],
  "json": "/tmp/opencode/v7_p1_dagger_cycle/t3_w4.json",
  "elapsed_s": 39.5,
  "rc": 0,
  "wall_s": 40.1
 },
 "paired_clean500_vs_p1b": {
  "tag": "w4_clean500_vs_p1b",
  "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-162639_v7p1b_screen_epoch010/episodes.csv",
  "agent": "runs/BTC20261003-053208_v7p1dagger_w4_clean500/episodes.csv",
  "rc": 0,
  "allow_mismatch": false,
  "out_dir": "/tmp/opencode/v7_p1_dagger_cycle/paired_eval/w4_clean500_vs_p1b",
  "log": "/tmp/opencode/v7_p1_dagger_cycle/paired_w4_clean500_vs_p1b.log",
  "n_paired": 500,
  "rate_base": 0.314,
  "rate_agent": 0.378,
  "delta_pp": 6.4,
  "ci95_pp": [
   1.4000000000000001,
   11.4
  ],
  "net": 32,
  "z": 2.4688535993934706,
  "fixed": 100,
  "broken": 68,
  "both_pass": 89,
  "both_fail": 243,
  "mcnemar_exact_p": 0.01651341548603896,
  "verdict": {
   "delta_pp": 6.4,
   "z": 2.4688535993934706,
   "mcnemar_exact_p": 0.01651341548603896,
   "ci95_pp": [
    1.4000000000000001,
    11.4
   ],
   "ci_lower_gt_zero": true,
   "positive_3pt_z196": true
  },
  "pair_meta": {
   "n_common": 500,
   "n_baseline_only": 0,
   "n_agent_only": 0,
   "baseline_only_examples": [],
   "agent_only_examples": [],
   "allow_mismatch": false
  }
 },
 "hit": true,
 "hit_detail": {
  "delta_pp": 6.4,
  "delta_ge_5pt": true,
  "t3_s1": 1,
  "tg45_sr": 0.06666666666666667
 }
}
```

## 6. 采纳 / eval500

```json
{
 "adopted_window": 4,
 "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch005.pt",
 "eval500_sr": 0.344,
 "paired_eval500": {
  "delta_pp": 3.2,
  "ci95_pp": [
   -1.6,
   8.0
  ],
  "mcnemar_exact_p": 0.21431491737771186,
  "rate_base": 0.312,
  "rate_agent": 0.344
 }
}
```

## 7. 判定汇总

```json
{
 "criteria": {
  "delta_pp_ge": 5.0,
  "t3_s1_ge": 1,
  "tg45_ge": 0.1
 },
 "window_hits": {
  "w1": {
   "delta_pp": 21.2,
   "delta_ge_5pt": true,
   "t3_s1": 0,
   "tg45_sr": 0.0
  },
  "w2": {
   "delta_pp": 11.4,
   "delta_ge_5pt": true,
   "t3_s1": 0,
   "tg45_sr": 0.0
  },
  "w3": {
   "delta_pp": 15.8,
   "delta_ge_5pt": true,
   "t3_s1": 0,
   "tg45_sr": 0.022222222222222223
  },
  "w4": {
   "delta_pp": 6.4,
   "delta_ge_5pt": true,
   "t3_s1": 1,
   "tg45_sr": 0.06666666666666667
  }
 },
 "n_hits": 1,
 "adopted_window": 4,
 "adopted_ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch005.pt",
 "adopted_reason": "命中窗口中 clean500 success 最大（平局取更早窗口）",
 "eval500": {
  "success_rate": 0.344,
  "collision_rate": 0.042,
  "off_road_rate": 0.604,
  "run_dir": "runs/BTC20261003-054238_v7p1dagger_w4_eval500",
  "episodes_csv": "runs/BTC20261003-054238_v7p1dagger_w4_eval500/episodes.csv"
 },
 "paired_eval500_vs_p1b": {
  "delta_pp": 3.2,
  "ci95_pp": [
   -1.6,
   8.0
  ],
  "mcnemar_exact_p": 0.21431491737771186,
  "rate_base": 0.312,
  "rate_agent": 0.344
 },
 "verdict": "IL_HIT"
}
```



## 9. 执行摘要与审阅注记（post-hoc；不改判据）

### 9.1 判定

- **IL_HIT（预注册字面命中，仅 w4）**：判据 `Δclean500 ≥ +5pt 且（T3 S1 ≥ 1 或 tg45 ≥ 0.1）`。
  w4 满足 `Δ=+6.4pt` 且 `T3 S1=1`；两种括号读法（A∧(B∨C) 与 (A∧B)∨C）下结论一致：仅 w4 命中。
- **张力（如实）**：clean500 冠军是 **w1（0.526，Δ+21.2pt）**，三个 bias=0.2 臂（w1-3）收费站专项
  （T3）全部 0/9；唯一命中来自 **bias_calib=0 对照臂 + 单条 T3（id=164）**，且 w4 的 clean500 Δ
  CI 下界仅 +1.4pt、eval500 配对不显著。
- **eval500（仅采纳 candidate 评一次）**：w4 e005 = **0.344**，vs P1-B e010 0.312 配对 Δ **+3.2pt**
  （CI95 [-1.6, +8.0]，p=0.214，不显著；E-β″ 0.440 / IDM 0.756 参照仍高）。

### 9.2 每窗口（top1；配对 vs P1-B e010 clean500）

| 窗口 | bias | 入选 ckpt | clean150 | clean500 | 配对 Δpp (CI95) | McNemar p | tg45 (n=45) | T3 S1 | 命中 |
|---|---|---|---|---|---|---|---|---|---|
| w1 | 0.2 | e005 | 0.520 | 0.526 | +21.2 [17.6, 24.8] | 9.4e-30 | 0.000 | 0/9 | 否 |
| w2 | 0.2 | e010 | 0.433 | 0.428 | +11.4 [8.0, 14.8] | 7.0e-11 | 0.000 | 0/9 | 否 |
| w3 | 0.2 | e005 | 0.460 | 0.472 | +15.8 [12.2, 19.6] | 3.5e-16 | 0.022 | 0/9 | 否 |
| w4 | 0 | e005 | 0.393 | 0.378 | +6.4 [1.4, 11.4] | 0.017 | 0.067 | **1/9** | **是** |

- 基线：P1-B e010 clean150 = 0.320（由其在 clean500 的 episodes.csv 过滤子集 150 键逐条一致）、
  clean500 = 0.314、tg45 = 0.000、T3 S1 = 0/9。
- keep-best 规律：e005 最优（w2 例外为 e010）；e015 一致回退（0.207–0.400）→ 本配方 5–10 epoch 即
  达峰，长训（15ep）只提供 keep-best 空间。

### 9.3 T3 命中细节（w4，9 锚）

- id=164：base=arrive_dest，ckpt=arrive_dest，`moved_free_lane=True`，clearance=**1.625 m ≥ 1.5** → S1。
- 其余 8 条：7 条 out_of_road（free-lane 未达成）+ 1 条 collision（id=239），与 P1-B/此前失败签名一致
  → 收费站失败模式未被本轮 DAgger 实质修复；唯一信号为单条。

### 9.4 执行偏差（如实）

1. **首轮 w1 训练被驱动器 wall-guard 误杀**（23:51:38–23:59:28，rc=-15）：守卫自进程启动计时，含
   ~4 min 数据加载/建窗（`data=35.98s` 等），测得 469.3s > 217.5s 阈值；训练本身正常（epoch 计算
   ~211s，loss 0.0590 无 NaN）。修复：epoch 墙钟自 `[bc3] 梯度累积` 行起算 + 阈值 145s×1.75=253.75s；
   采集数据复用后重训，**数据/配方/评测口径无变化**；误杀 run 目录已删除。
2. `collect_window` 复用路径缺 `rc` 字段 → 00:03:17 一次立即 abort；00:04:01 修复重启。无数据影响。
3. 实际墙钟：采集 4×≈21.5 min（1287/1218/1337/1235 s）；训练 4×≈49 min（2945/2954/2989/2938 s，
   15 epochs × ~175–185 s）；整链 23:30:05 → 05:51:48 ≈ **6 h 22 min**（超 4–6 h 估算 ~0.5 h，原因：
   epoch 175–185s vs 基线 145s + 每窗口 ~7 min 装载）。
4. w1 采集因复用未写入 `stage collect:w1`（缺失仅为记录项）；补记：seed=20261021、scanned=1480、
   fails=1006、elapsed=1287.1s、log=`/tmp/opencode/v7_p1_dagger_cycle/collect_w1.log`。

### 9.5 数据 / 训练 / 产物（摘要）

- 数据集：`datasets/BTC20261002-2329_v7p1dagger_w{1..4}`（rows 20071/20082/20055/20049；
  trainable 20047/20016/20026/20031；链覆盖 74.93–74.94%；NaN 尾 5030/5035/5025/5025 行；
  npz sha256 见 §2.x b）。
- 训练：`runs/BTC20261002-2329_v7p1dagger_w{1..4}`；四窗口 `config_hash=7b95501a40c3f78c`；
  micro=256、batch=1024、`action_chain_source=labels`、锚 mild=1.0、bias 0.2/0.2/0.2/0（各 15 epochs，
  ckpt e005/e010/e015 + final）。
- bias 终值（train 聚合）：w1-3 `mu_dθ` −0.0008…−0.0011、`traj_lat_h6` −0.0007…−0.0013、penalty ~0.0074–0.0080；
  w4（关）`mu_dθ` +0.00002、`traj_lat_h6` **+0.094**、penalty 0（指标恒记录）。即标定项确实压低了
  聚合 signed-bias，但**未转化为收费站专项增益**。
- 采纳 ckpt：`runs/BTC20261002-2329_v7p1dagger_w4/ckpt_epoch005.pt`
  sha256 `47e5f1c0d37dae4688e9da23b2c4723d9019d026c1caf5190d2a931638182bdf`。
- eval500：`runs/BTC20261003-054238_v7p1dagger_w4_eval500`（episodes sha256 `cb0d5e03…`）。
- T3 JSON：`/tmp/opencode/v7_p1_dagger_cycle/t3_w{1..4}.json`；paired 目录 `paired_eval/`。
- git：`4a46c5e`（脏改仅 runs/datasets/tmp；`net/`、`env/` 未改）。

### 9.6 解读（不改判据）

- 强锚（v4 expert5k mild=1.0）+ 链标签 + specific_only 的 phase3 等价训练把 clean500 从 0.314 提升到
  0.378–0.526（四窗口全部 ≥ +5pt），**clean500 侧本轮 IL 增益显著**；但收费站专项（T3 9 锚 / tg45 45 条）
  几乎无改善，bias 标定（0.2）亦未带来收费站增量（唯一 T3 信号在对照组）。
- 判据字面命中，但证据强度弱：w4 Δ CI 下界 +1.4pt、T3 仅单条、eval500 配对不显著。若编排决策需要
  "clean500 最强"起点，w1 e005（0.526）更优；若以收费站信号为准，w4 e005（T3 S1=1）是唯一命中。
  转 RL 或继续 IL 的取舍留给编排决策（本轮不越权更改预注册）。
