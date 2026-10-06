# v7-P1B 链报告：obs v4 重采 → Stage A/B 重训 → tollgate 回归

- 状态：**aborted:B**（最终） @ 2026-10-02T13:25:31
- git：`ee763f47578720a8ba2ca6a123bdcb9fd89a6e8e`  stamp：20261002-0941
- 采集：`/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert5k_v4` / `/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert500val_v4`
- 训练：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-0941_v7p1b`

## 0. lane 等待

```json
{
 "duration_s": 0.0,
 "lane": {
  "test": 0,
  "train": 0,
  "spawn": 0,
  "v7p0": 0,
  "gpu_mib": 459,
  "mem_avail_kb": 10862952
 }
}
```

## 1. v4 重采（expert5k + expert500val）

```json
{
 "train": {
  "rc": 0,
  "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert5k_v4",
  "npz": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert5k_v4/expert_bc.npz",
  "npz_bytes": 298744392,
  "npz_sha256": "688c90fba5c8535df77de1adf2644c244c1ca153282d9f53b72d424f715038b4",
  "meta_sha256": "99a268b361f0d6b8b8a9992b04f792f25cd0e3b6a839eeddfef528c28504ea1d",
  "obs_schema_version": 4,
  "obs_fingerprint": "v4-e9ab75bc4c6f",
  "count": 360561,
  "others_shape": [
   1,
   33
  ],
  "yield": 259647
 },
 "val": {
  "rc": 0,
  "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert500val_v4",
  "npz": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert500val_v4/expert_bc.npz",
  "npz_bytes": 30101314,
  "npz_sha256": "bf1ca9a4a90b7e55e86aa8e80e5e03e913d552c53269ff74e1b6e312c7a07e14",
  "meta_sha256": "948be05ed0e486ef9fa5d193157c75d08969f8b8b5053c3400fa673b5f2b8fcc",
  "obs_schema_version": 4,
  "obs_fingerprint": "v4-e9ab75bc4c6f",
  "count": 36159,
  "others_shape": [
   1,
   33
  ],
  "yield": 25869
 }
}
```

## 1b. 采集抽样验证

```json
{
 "rc": 0,
 "stdout": "{\"verdict\": {\"schema_v4\": true, \"base_fields_match\": true, \"divergence_bounded\": true, \"static_non_degenerate\": true, \"all_ok\": true}, \"train_rows\": 360561, \"val_rows\": 36159, \"diverged_train\": 7, \"diverged_val\": 1, \"present_eps\": 44, \"corr_ok_eps\": 43}",
 "stderr": "",
 "json": "/tmp/opencode/v7_p1b_chain/collect_verify.json",
 "elapsed_s": {
  "expert5k": 1701.6,
  "expert500val": 130.1
 },
 "diverged_specs": {
  "train": [
   338,
   455,
   542,
   2213,
   2693,
   3074,
   3422
  ],
  "train_n": 7,
  "val": [
   644
  ],
  "val_n": 1,
  "note": "逐 spec 行数跨 run 不一致（多进程采集非确定性；v2↔v3 亦存在）→ 抽样剔除"
 },
 "train_meta": {
  "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert5k_v4",
  "obs_schema_version": 4,
  "obs_fingerprint": "v4-e9ab75bc4c6f",
  "others_shape": [
   1,
   33
  ],
  "config": {
   "limit": 5000,
   "split": "all",
   "max_steps": 600,
   "on_lane_frac": 0.5,
   "on_lane_margin": 0.3,
   "roundtrip_key_mean": 0.25,
   "roundtrip_key_max": 0.5,
   "require_dense": false,
   "roundtrip_dense_mean": 0.5,
   "balance": "weights",
   "balance_ratio": 3.0,
   "traffic_density": null
  },
  "config_matches_v3": true,
  "schema_version": 2,
  "history_storage": "per_frame_v2",
  "sha256": {
   "expert_bc.npz": "688c90fba5c8535df77de1adf2644c244c1ca153282d9f53b72d424f715038b4",
   "expert_bc.meta.json": "99a268b361f0d6b8b8a9992b04f792f25cd0e3b6a839eeddfef528c28504ea1d",
   "report.json": "4dbb5c3705829abe8227210a76c75b9edce77ce85b30da24eb52c70dbcc23334"
  },
  "npz_others_shape": [
   360561,
   1,
   33
  ],
  "npz_keys": 46,
  "ok_schema": true
 },
 "val_meta": {
  "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert500val_v4",
  "obs_schema_version": 4,
  "obs_fingerprint": "v4-e9ab75bc4c6f",
  "others_shape": [
   1,
   33
  ],
  "config": {
   "limit": null,
   "split": "all",
   "max_steps": 600,
   "on_lane_frac": 0.5,
   "on_lane_margin": 0.3,
   "roundtrip_key_mean": 0.25,
   "roundtrip_key_max": 0.5,
   "require_dense": false,
   "roundtrip_dense_mean": 0.5,
   "balance": "weights",
   "balance_ratio": 3.0,
   "traffic_density": null
  },
  "config_matches_v3": true,
  "schema_version": 2,
  "history_storage": "per_frame_v2",
  "sha256": {
   "expert_bc.npz": "bf1ca9a4a90b7e55e86aa8e80e5e03e913d552c53269ff74e1b6e312c7a07e14",
   "expert_bc.meta.json": "948be05ed0e486ef9fa5d193157c75d08969f8b8b5053c3400fa673b5f2b8fcc",
   "report.json": "cf4b072392d0a41ac1d890a99a3afdd9dc2aac2e1742cd7d9c7cfe251d8858a9"
  },
  "npz_others_shape": [
   36159,
   1,
   33
  ],
  "npz_keys": 46,
  "ok_schema": true
 },
 "train_base_compare": {
  "v4_rows": 360561,
  "v3_rows": 360407,
  "n_specs_v4": 5000,
  "sampled": 300,
  "checked": 300,
  "n_mismatch": 0,
  "mismatch_examples": [],
  "ok_base_match": true
 },
 "val_base_compare": {
  "v4_rows": 36159,
  "v3_rows": 36147,
  "n_specs_v4": 500,
  "sampled": 300,
  "checked": 300,
  "n_mismatch": 0,
  "mismatch_examples": [],
  "ok_base_match": true
 },
 "static_val": {
  "tg_episodes": 45,
  "tg_rows": 3184,
  "present_rows": 1127,
  "present_frac_of_tg_rows": 0.3539572864321608,
  "episodes_with_present": 44,
  "episodes_corr_le_m03": 43,
  "gap_norm_global_min": 0.0,
  "gap_norm_global_max": 0.9998899698257446,
  "per_episode": [
   {
    "id": 34,
    "n_present": 29,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9579098224639893,
    "corr_step_gap": -0.9807292580156645
   },
   {
    "id": 76,
    "n_present": 19,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9982244968414307,
    "corr_step_gap": -0.9965082059602037
   },
   {
    "id": 84,
    "n_present": 14,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9476443529129028,
    "corr_step_gap": -0.9982150697503022
   },
   {
    "id": 85,
    "n_present": 26,
    "gap_norm_min": 0.24787308275699615,
    "gap_norm_max": 0.9727822542190552,
    "corr_step_gap": -0.9747763535451852
   },
   {
    "id": 101,
    "n_present": 52,
    "gap_norm_min": 0.11931396275758743,
    "gap_norm_max": 0.9984970092773438,
    "corr_step_gap": -0.8520320852472586
   },
   {
    "id": 131,
    "n_present": 18,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9954881072044373,
    "corr_step_gap": -0.9967886738107679
   },
   {
    "id": 147,
    "n_present": 8,
    "gap_norm_min": 0.4079824686050415,
    "gap_norm_max": 0.9697006344795227,
    "corr_step_gap": -0.9995304755840473
   },
   {
    "id": 164,
    "n_present": 38,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9799150228500366,
    "corr_step_gap": -0.9813632764865379
   },
   {
    "id": 166,
    "n_present": 15,
    "gap_norm_min": 0.3899753987789154,
    "gap_norm_max": 0.9909659028053284,
    "corr_step_gap": -0.14446942408648883
   },
   {
    "id": 183,
    "n_present": 8,
    "gap_norm_min": 0.39777281880378723,
    "gap_norm_max": 0.955031156539917,
    "corr_step_gap": -0.9993842905373039
   },
   {
    "id": 199,
    "n_present": 22,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9686205983161926,
    "corr_step_gap": -0.9903616833544817
   },
   {
    "id": 217,
    "n_present": 21,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9686020016670227,
    "corr_step_gap": -0.717455944310927
   },
   {
    "id": 239,
    "n_present": 35,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9902212023735046,
    "corr_step_gap": -0.989582397062951
   },
   {
    "id": 243,
    "n_present": 13,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9187625050544739,
    "corr_step_gap": -0.9993061002560191
   },
   {
    "id": 273,
    "n_present": 14,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.953279435634613,
    "corr_step_gap": -0.9983162008131302
   },
   {
    "id": 293,
    "n_present": 36,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9431337118148804,
    "corr_step_gap": -0.946394404722991
   },
   {
    "id": 296,
    "n_present": 100,
    "gap_norm_min": 0.19968856871128082,
    "gap_norm_max": 0.9410763382911682,
    "corr_step_gap": -0.7429948304263563
   },
   {
    "id": 324,
    "n_present": 19,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.962699294090271,
    "corr_step_gap": -0.9790665924480233
   },
   {
    "id": 325,
    "n_present": 18,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9637721180915833,
    "corr_step_gap": -0.9894305241587862
   },
   {
    "id": 343,
    "n_present": 24,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9519301056861877,
    "corr_step_gap": -0.9786678866883206
   },
   {
    "id": 353,
    "n_present": 17,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9337354898452759,
    "corr_step_gap": -0.9909024413797417
   },
   {
    "id": 358,
    "n_present": 16,
    "gap_norm_min": 0.34236520528793335,
    "gap_norm_max": 0.9449476599693298,
    "corr_step_gap": -0.32038903273881825
   },
   {
    "id": 379,
    "n_present": 19,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9303273558616638,
    "corr_step_gap": -0.9686315002901786
   },
   {
    "id": 418,
    "n_present": 14,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9503031373023987,
    "corr_step_gap": -0.9983766444463891
   },
   {
    "id": 419,
    "n_present": 14,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9252943396568298,
    "corr_step_gap": -0.9975056521769861
   },
   {
    "id": 469,
    "n_present": 17,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.998048722743988,
    "corr_step_gap": -0.9977035382441047
   },
   {
    "id": 476,
    "n_present": 36,
    "gap_norm_min": 0.18289947509765625,
    "gap_norm_max": 0.9582919478416443,
    "corr_step_gap": -0.9481770535854367
   },
   {
    "id": 486,
    "n_present": 14,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9691162109375,
    "corr_step_gap": -0.9987739331901225
   },
   {
    "id": 520,
    "n_present": 0,
    "corr_step_gap": null,
    "gap_norm_min": null,
    "gap_norm_max": null
   },
   {
    "id": 547,
    "n_present": 23,
    "gap_norm_min": 0.3628140985965729,
    "gap_norm_max": 0.9590099453926086,
    "corr_step_gap": -0.8011653936417149
   },
   {
    "id": 607,
    "n_present": 20,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9924268126487732,
    "corr_step_gap": -0.9890733316987101
   },
   {
    "id": 644,
    "n_present": 29,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9723159670829773,
    "corr_step_gap": -0.9885017699538912
   },
   {
    "id": 658,
    "n_present": 13,
    "gap_norm_min": 0.3835510015487671,
    "gap_norm_max": 0.9852055907249451,
    "corr_step_gap": -0.9878613262235243
   },
   {
    "id": 706,
    "n_present": 17,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9820547699928284,
    "corr_step_gap": -0.658001995511602
   },
   {
    "id": 741,
    "n_present": 14,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9762976169586182,
    "corr_step_gap": -0.9989517740654141
   },
   {
    "id": 753,
    "n_present": 16,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9964278936386108,
    "corr_step_gap": -0.9986475622102818
   },
   {
    "id": 771,
    "n_present": 14,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9505398273468018,
    "corr_step_gap": -0.9983543528718931
   },
   {
    "id": 781,
    "n_present": 23,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9476039409637451,
    "corr_step_gap": -0.7814936979466369
   },
   {
    "id": 785,
    "n_present": 25,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9734904766082764,
    "corr_step_gap": -0.9896350054279094
   },
   {
    "id": 808,
    "n_present": 17,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.972496509552002,
    "corr_step_gap": -0.7198707194000789
   },
   {
    "id": 851,
    "n_present": 103,
    "gap_norm_min": 0.11323410272598267,
    "gap_norm_max": 0.9909816980361938,
    "corr_step_gap": -0.8150346506894494
   },
   {
    "id": 863,
    "n_present": 103,
    "gap_norm_min": 0.30066436529159546,
    "gap_norm_max": 0.9998899698257446,
    "corr_step_gap": -0.8632720098703239
   },
   {
    "id": 895,
    "n_present": 13,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.924574613571167,
    "corr_step_gap": -0.9985138854382007
   },
   {
    "id": 906,
    "n_present": 7,
    "gap_norm_min": 0.4374648928642273,
    "gap_norm_max": 0.9264483451843262,
    "corr_step_gap": -0.9998275473886925
   },
   {
    "id": 924,
    "n_present": 14,
    "gap_norm_min": 0.0,
    "gap_norm_max": 0.9820834398269653,
    "corr_step_gap": -0.9990708005712999
   }
  ],
  "demo_spec34_step_gapnorm": [
   [
    325,
    0.9579
   ],
   [
    330,
    0.8753
   ],
   [
    335,
    0.796
   ],
   [
    340,
    0.762
   ],
   [
    345,
    0.7437
   ],
   [
    350,
    0.725
   ],
   [
    355,
    0.7069
   ],
   [
    360,
    0.6878
   ]
  ],
  "ok_non_degenerate": true
 },
 "verdict": {
  "schema_v4": true,
  "base_fields_match": true,
  "divergence_bounded": true,
  "static_non_degenerate": true,
  "all_ok": true
 }
}
```

## 2. Stage A（20ep micro256）

```json
{
 "rc": 0,
 "duration_s": 6057.1,
 "epochs_done": 24,
 "epoch_walls_s": [
  294.9,
  290.5,
  290.6,
  290.6,
  290.3,
  0.0,
  290.4,
  289.7,
  290.7,
  289.9,
  291.0,
  0.0,
  290.1,
  291.0,
  290.3,
  290.6,
  293.0,
  0.0,
  292.8,
  291.8,
  290.9,
  290.6,
  290.7,
  0.0
 ],
 "peak_mib": 8919,
 "nav_warns": [],
 "aborted": null,
 "oom": false,
 "log": "/tmp/opencode/v7_p1b_chain/stage_a.log"
}
```

## 3. Stage B（20ep micro512）

```json
{
 "rc": -15,
 "duration_s": 1529.5,
 "epochs_done": 7,
 "epoch_walls_s": [
  194.7,
  194.7,
  195.0,
  196.3,
  195.8,
  195.7,
  196.4
 ],
 "peak_mib": 8975,
 "nav_warns": [],
 "aborted": "epoch 7 墙钟 196.4s > 基线 151s x 1.3 = 196s（abort 判据）",
 "oom": false,
 "log": "/tmp/opencode/v7_p1b_chain/stage_b.log",
 "micro": 512
}
```

## 6. 异常与偏差

- resume：跳过采集，复用既有 v4 数据目录

