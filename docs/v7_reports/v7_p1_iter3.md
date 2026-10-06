# v7-P1 iter3 报告：v4.1 数据 + 横向权重 [1.0,69.4] 的 A20+B20 重训与闭环对照

- 状态：**done**（最终） @ 2026-10-02T22:05:26
- git：`2e13dfa2a40e4d06070009240267bdcdf66aa025`  stamp：20261002-1837
- 数据：`/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert5k_v41` / `/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert500val_v41`
- 训练/评测 run：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-1837_v7p1i3`
- abort 阈值：A 254s×1.35=342.9s / B 151s×1.35=203.9s（v4 架构先例）；ckpt_every=5。
- 本轮不跑 phase3（v2 dagger/anchor 无 static 段，会把 static 清零；待 v4 版 dagger 再议）。

## TL;DR

1. **训练链正常**：A20（micro256）+ B20（micro512）全部完成，无 abort / OOM / nav WARN；
   A 墙钟 276–280s（阈值 342.9s）、B 174–188s（阈值 203.9s）、峰值 8908/8989 MiB；
   config hash `ec30e385878963bb`（P1-B `f75b1b3c42c320d6`），`action_dim_weights=[1.0,69.4]` 生效。
2. **keep-best 选点 = epoch005**（clean500 0.260）；筛选序列 epoch005/010/015/final =
   0.260 / 0.222 / 0.122 / 0.216 → 训练越久闭环越差，B20 末段坍缩形态与 v6/P1-B 一致（且更早）。
3. **闭环全面 FAIL**：T3 S1 0/9；tg45 0.022（IDM 0.778，闸 0.678）；clean500 0.260
   （E-β″ 0.440 / IDM 0.742）；eval500 0.274（E-β″ 0.440 / IDM 0.756）。
4. **配对 vs P1-B（v4 数据、无横向权重）显著更差**：clean500 Δ=−5.4pp（CI95 [−8.4,−2.4]，
   p=5.8e-4）；eval500 Δ=−3.8pp（CI95 [−6.6,−0.8]，p=0.016）；vs E-β″ −18.0/−16.6pp；
   vs IDM −48.2/−48.2pp（tg45 −75.6pp）。
5. **唯一定性变化**：T3 9 条中 2 条（id84/id166）进入自由车道（P1-B 选定 ckpt 为 0 条），
   但过闸净空 1.176/1.148 m < 1.5 m 判据，且二者均以 collision 终止。
6. **训练侧读数**（val，specific phase）：未加权动作误差 0.2058→**0.1964**（−4.6%）、
   加权 0.0375→0.0353（−5.9%）、traj MAE 0.3171→0.3247（+2.4%）；metrics 不暴露逐维 dθ，
   故无法从训练读数直接确认 iter2 TF 探针的 dθ 改善在完整训练后是否保持。
7. **结论**：v4.1 数据 + 横向权重 [1.0,69.4] **未转化为闭环收益**，且相对 P1-B 统计显著回退；
   横向表达修复（TF 探针）与闭环成功之间的断点仍在，需上游决策下一步（权重强度/解耦头/
   v4 版 dagger+phase3 救援等）。

## 0. preflight（config hash / 数据 sha / 锚参照）

```json
{
 "config_files": {
  "config/default.yaml": {
   "sha256": "432b2d072ae50a4398e7d9e607d3bc155b270d4f040fa7405c7d83f27b1266ca",
   "bytes": 560
  },
  "config/train.yaml": {
   "sha256": "a816e66d8bea81064c8f64795412d436351c597a33751bcb204515176db1738e",
   "bytes": 20816
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
 "data": {
  "train": {
   "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert5k_v41",
   "npz_sha256": "427762e4bea3a35ea722388fe614c95d2a8d61444248336ed6bb247246f3f1cc",
   "npz_bytes": 298632585,
   "meta_sha256": "8c83cc460610fd154f3413dbc6feb262771441dc1fc7199b4cd33a6a5cd1fc4a",
   "count": 360450,
   "obs_schema_version": 4,
   "obs_fingerprint": "v4-e9ab75bc4c6f",
   "others_shape": [
    1,
    33
   ],
   "trainable_rows": 265742,
   "collect_config": {
    "limit": 5000,
    "split": "all",
    "max_steps": 600,
    "on_lane_frac": 0.5,
    "on_lane_margin": 0.3,
    "roundtrip_key_mean": 0.25,
    "roundtrip_key_max": 0.5,
    "require_dense": false,
    "roundtrip_dense_mean": 0.5,
    "roundtrip_lc_lat": 0.8,
    "roundtrip_lc_first_max": 0.5,
    "roundtrip_lc_mean": 0.5,
    "balance": "weights",
    "balance_ratio": 3.0,
    "traffic_density": null
   }
  },
  "val": {
   "dir": "/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261002-0941_expert500val_v41",
   "npz_sha256": "1e79747224c6522df016d664c6779b490685584ae85bd2c28d140a3ae84597af",
   "npz_bytes": 30090891,
   "meta_sha256": "70c879405b65958ebd0341df3cb73ad72c88a7c210495bdf9ee4481dbf8ca30e",
   "count": 36148,
   "obs_schema_version": 4,
   "obs_fingerprint": "v4-e9ab75bc4c6f",
   "others_shape": [
    1,
    33
   ],
   "trainable_rows": 26474,
   "collect_config": {
    "limit": null,
    "split": "all",
    "max_steps": 600,
    "on_lane_frac": 0.5,
    "on_lane_margin": 0.3,
    "roundtrip_key_mean": 0.25,
    "roundtrip_key_max": 0.5,
    "require_dense": false,
    "roundtrip_dense_mean": 0.5,
    "roundtrip_lc_lat": 0.8,
    "roundtrip_lc_first_max": 0.5,
    "roundtrip_lc_mean": 0.5,
    "balance": "weights",
    "balance_ratio": 3.0,
    "traffic_density": null
   }
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
  "ebeta_clean500": {
   "path": "runs/BTC20261001-195447_v6p3_e3_clean500/episodes.csv",
   "exists": true,
   "sha256": "d63db75e6162af90418afc29c9639f6f002d7c850ae3866f6256c2ce1110a38a"
  },
  "ebeta_eval500": {
   "path": "runs/BTC20261001-200331_v6p3_e3_eval500/episodes.csv",
   "exists": true,
   "sha256": "3d949398c7db53956021cde344e3df7862162f2f6d8bc53779cf8d4a456c904e"
  },
  "idm_clean500": {
   "path": "runs/BTC20261002-100413_v7p0_idm_clean500/episodes.csv",
   "exists": true,
   "sha256": "9bf57b196abc49bcc881835abd3d0d966cabd9e09cd78f72f794953840594eb8"
  },
  "idm_eval500": {
   "path": "runs/BTC20260927-1839_eval500_baseline/episodes.csv",
   "exists": true,
   "sha256": "d6269ef238f11f9b0064d6d1b5d6671a53dbf5260c52ea2f1e41509a17676e7c"
  },
  "idm_tg45": {
   "path": "runs/BTC20261002-160154_v7p1b_tg45_idm/episodes.csv",
   "exists": true,
   "sha256": "367ffff92e289e40b20ef3dde97ec8b11e822d1db509b103656c333b553e679b"
  }
 },
 "action_dim_weights": [
  1.0,
  69.4
 ],
 "train_seed": null,
 "t3_ids": [
  34,
  243,
  164,
  84,
  166,
  131,
  76,
  147,
  239
 ],
 "t3_measure_sha256": "537df2db5a4525c7fa5e6638c3f491e8548a99844797983db1bb8dcd5787ff71",
 "tg45_spec_sha256": "27010b0ebfb59d1daf45570d5851cf47452ab3d5eddbc9d9f3ddfc6d98c523e6",
 "clean500_spec_sha256": "087db3f5c02e9cedccb8313733207f59dfa90162002b0d9bf002e5b80bcc366e"
}
```

## 0b. lane 等待

```json
{
 "duration_s": 0.0,
 "lane": {
  "test": 0,
  "train": 0,
  "spawn": 0,
  "v7p0": 0,
  "gpu_mib": 471,
  "mem_avail_kb": 10965548
 }
}
```

## 1. Stage A（20ep micro256）

```json
{
 "rc": 0,
 "duration_s": 5771.8,
 "epochs_done": 24,
 "epoch_walls_s": [
  277.5,
  276.7,
  277.0,
  276.9,
  276.3,
  0.0,
  276.3,
  277.1,
  276.4,
  276.6,
  276.5,
  0.0,
  276.5,
  280.2,
  276.6,
  276.7,
  276.5,
  0.0,
  277.2,
  276.8,
  276.4,
  276.2,
  276.5,
  0.0
 ],
 "peak_mib": 8908,
 "nav_warns": [],
 "aborted": null,
 "oom": false,
 "log": "/tmp/opencode/v7_p1i3/stage_a.log",
 "baseline_s": 254.0,
 "factor": 1.35,
 "threshold_s": 342.9,
 "config_hash": "ec30e385878963bb"
}
```

## 2. Stage B（20ep micro512）

```json
{
 "rc": 0,
 "duration_s": 3857.2,
 "epochs_done": 20,
 "epoch_walls_s": [
  187.2,
  187.3,
  187.7,
  188.0,
  188.2,
  188.9,
  188.5,
  188.4,
  188.5,
  188.6,
  173.8,
  174.1,
  175.0,
  174.6,
  174.6,
  174.7,
  174.9,
  175.1,
  175.3,
  176.2
 ],
 "peak_mib": 8989,
 "nav_warns": [],
 "aborted": null,
 "oom": false,
 "log": "/tmp/opencode/v7_p1i3/stage_b.log",
 "baseline_s": 151.0,
 "factor": 1.35,
 "threshold_s": 203.9,
 "micro": 512,
 "config_hash": "ec30e385878963bb"
}
```

## 3. clean500 keep-best 筛选

```json
[
 {
  "tag": "screen_epoch005",
  "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only500.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-1837_v7p1i3/stage_b/ckpt_epoch005.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 528.1,
  "log": "/tmp/opencode/v7_p1i3/eval_screen_epoch005.log",
  "timed_out": false,
  "run_dir": "runs/BTC20261002-211752_v7p1i3_screen_epoch005",
  "n": 500,
  "n_error": 0,
  "success_rate": 0.26,
  "collision_rate": 0.066,
  "off_road_rate": 0.664,
  "episodes_csv": "runs/BTC20261002-211752_v7p1i3_screen_epoch005/episodes.csv",
  "episodes_sha256": "783317731042c17adaa038b0db3bef71eb343b7182cdd108ec956288226c094d",
  "sha256": "f555a787239bd3fe096b8fb5c929f5cc7eb53423a4771a86cafb1501c7298829"
 },
 {
  "tag": "screen_epoch010",
  "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only500.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-1837_v7p1i3/stage_b/ckpt_epoch010.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 523.0,
  "log": "/tmp/opencode/v7_p1i3/eval_screen_epoch010.log",
  "timed_out": false,
  "run_dir": "runs/BTC20261002-212641_v7p1i3_screen_epoch010",
  "n": 500,
  "n_error": 0,
  "success_rate": 0.222,
  "collision_rate": 0.028,
  "off_road_rate": 0.746,
  "episodes_csv": "runs/BTC20261002-212641_v7p1i3_screen_epoch010/episodes.csv",
  "episodes_sha256": "4c47c1a0e75e5ba853ea878b63c1b2f33981389c9b400517e46f4ee8897db1e3",
  "sha256": "2ce50e3614d47e771eb9a65b42cd9d628120fd25039b60721b718391c3755e19"
 },
 {
  "tag": "screen_epoch015",
  "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only500.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-1837_v7p1i3/stage_b/ckpt_epoch015.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 579.0,
  "log": "/tmp/opencode/v7_p1i3/eval_screen_epoch015.log",
  "timed_out": false,
  "run_dir": "runs/BTC20261002-213524_v7p1i3_screen_epoch015",
  "n": 500,
  "n_error": 0,
  "success_rate": 0.122,
  "collision_rate": 0.056,
  "off_road_rate": 0.79,
  "episodes_csv": "runs/BTC20261002-213524_v7p1i3_screen_epoch015/episodes.csv",
  "episodes_sha256": "6eb5e8371cfc1ab7bdcd7a16068caf81181e8d931f66a2640c27339cbc43ed9a",
  "sha256": "d6e70ab47b737ca2665f08e8eadfe430b3ce542e1892944d120e619cbc302c8f"
 },
 {
  "tag": "screen_final",
  "spec": "/tmp/opencode/phase3_diag/exp/specs_val_only500.json",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-1837_v7p1i3/stage_b/final.pt",
  "policy": "ckpt",
  "rc": 0,
  "duration_s": 601.4,
  "log": "/tmp/opencode/v7_p1i3/eval_screen_final.log",
  "timed_out": false,
  "run_dir": "runs/BTC20261002-214503_v7p1i3_screen_final",
  "n": 500,
  "n_error": 0,
  "success_rate": 0.216,
  "collision_rate": 0.048,
  "off_road_rate": 0.718,
  "episodes_csv": "runs/BTC20261002-214503_v7p1i3_screen_final/episodes.csv",
  "episodes_sha256": "0ceca7963f6371d5da2ada81ea8ae8318cdfe7d22496c310209efee51221d210",
  "sha256": "ea7a96c4b487cb252c03b48c8916458bea8a78f3f5487825502030ad1b2009de"
 }
]
```

## 3b. 选定 ckpt

```json
{
 "tag": "screen_epoch005",
 "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-1837_v7p1i3/stage_b/ckpt_epoch005.pt",
 "clean500_success_rate": 0.26,
 "clean500_run_dir": "runs/BTC20261002-211752_v7p1i3_screen_epoch005",
 "sha256": "f555a787239bd3fe096b8fb5c929f5cc7eb53423a4771a86cafb1501c7298829",
 "rule": "clean500 success 最大；平局取更早 epoch"
}
```

## 4.1 T3 九条锚（选定 ckpt）

```json
{
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
   "plan_min_dist_to_booth_m": 154.086,
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
   "plan_min_dist_to_booth_m": 5.778,
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
   "id": 164,
   "ckpt_term": "out_of_road",
   "base_term": "arrive_dest",
   "ckpt_clearance": null,
   "base_clearance": 1.75,
   "moved_free_lane": false,
   "s1_resolved": false,
   "plan_crosses_booth": false,
   "plan_min_dist_to_booth_m": 9.944,
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
   "ckpt_term": "collision",
   "base_term": "arrive_dest",
   "ckpt_clearance": 1.176,
   "base_clearance": 1.738,
   "moved_free_lane": true,
   "s1_resolved": false,
   "plan_crosses_booth": false,
   "plan_min_dist_to_booth_m": 2.298,
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
   "ckpt_term": "collision",
   "base_term": "arrive_dest",
   "ckpt_clearance": 1.148,
   "base_clearance": 1.741,
   "moved_free_lane": true,
   "s1_resolved": false,
   "plan_crosses_booth": false,
   "plan_min_dist_to_booth_m": 2.797,
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
   "plan_min_dist_to_booth_m": 4.019,
   "lane_tail": [
    [
     ">>",
     ">>>",
     "1"
    ],
    [
     ">>>",
     "1S0_0_",
     "1"
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
   "plan_min_dist_to_booth_m": 214.395,
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
   "plan_min_dist_to_booth_m": 6.017,
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
   "id": 239,
   "ckpt_term": "collision",
   "base_term": "arrive_dest",
   "ckpt_clearance": null,
   "base_clearance": 1.749,
   "moved_free_lane": false,
   "s1_resolved": false,
   "plan_crosses_booth": false,
   "plan_min_dist_to_booth_m": 7.392,
   "lane_tail": [
    [
     ">>>",
     "1S0_0_",
     "1"
    ],
    [
     "1S0_0_",
     "2$0_0_",
     "1"
    ]
   ],
   "ckpt_error": null
  }
 ],
 "json": "/tmp/opencode/v7_p1i3/t3_measure_selected.json",
 "elapsed_s": 44.2,
 "rc": 0,
 "wall_s": 44.8
}
```

## 4.2 tollgate45（选定 ckpt；IDM 同批 0.778）

```json
{
 "tag": "tg45_sel",
 "spec": "/tmp/opencode/v7_p1b_chain/specs_tollgate45.json",
 "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-1837_v7p1i3/stage_b/ckpt_epoch005.pt",
 "policy": "ckpt",
 "rc": 0,
 "duration_s": 45.4,
 "log": "/tmp/opencode/v7_p1i3/eval_tg45_sel.log",
 "timed_out": false,
 "run_dir": "runs/BTC20261002-215549_v7p1i3_tg45_sel",
 "n": 45,
 "n_error": 0,
 "success_rate": 0.022222222222222223,
 "collision_rate": 0.2,
 "off_road_rate": 0.8666666666666667,
 "episodes_csv": "runs/BTC20261002-215549_v7p1i3_tg45_sel/episodes.csv",
 "episodes_sha256": "a3e8a2312594df062aad4ea65bf31aca956ba1bea123d4e50b9aaa8d0f5f9b0f"
}
```

## 4.3 eval500（选定 ckpt；E-β″ 0.440 / IDM 0.756）

```json
{
 "tag": "eval500_sel",
 "spec": "env/specs/scenarios_eval500.json",
 "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-1837_v7p1i3/stage_b/ckpt_epoch005.pt",
 "policy": "ckpt",
 "rc": 0,
 "duration_s": 529.7,
 "log": "/tmp/opencode/v7_p1i3/eval_eval500_sel.log",
 "timed_out": false,
 "run_dir": "runs/BTC20261002-215634_v7p1i3_eval500_sel",
 "n": 500,
 "n_error": 0,
 "success_rate": 0.274,
 "collision_rate": 0.062,
 "off_road_rate": 0.652,
 "episodes_csv": "runs/BTC20261002-215634_v7p1i3_eval500_sel/episodes.csv",
 "episodes_sha256": "803df03f06afbf4db5fb90b4ac756a832ad84bd771a0d856854c63f1b3235623"
}
```

## 4.4 paired:sel_clean500_vs_ebeta

```json
{
 "tag": "sel_clean500_vs_ebeta",
 "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-195447_v6p3_e3_clean500/episodes.csv",
 "agent": "runs/BTC20261002-211752_v7p1i3_screen_epoch005/episodes.csv",
 "rc": 0,
 "allow_mismatch": false,
 "out_dir": "/tmp/opencode/v7_p1i3/paired_eval/sel_clean500_vs_ebeta",
 "log": "/tmp/opencode/v7_p1i3/paired_sel_clean500_vs_ebeta.log",
 "n_paired": 500,
 "rate_base": 0.44,
 "rate_agent": 0.26,
 "delta_pp": -18.0,
 "ci95_pp": [
  -21.8,
  -14.2
 ],
 "net": -90,
 "z": 8.504200642707612,
 "fixed": 11,
 "broken": 101,
 "both_pass": 119,
 "both_fail": 269,
 "mcnemar_exact_p": 2.262519224355612e-19,
 "verdict": {
  "delta_pp": -18.0,
  "z": 8.504200642707612,
  "mcnemar_exact_p": 2.262519224355612e-19,
  "ci95_pp": [
   -21.8,
   -14.2
  ],
  "ci_lower_gt_zero": false,
  "positive_3pt_z196": false
 },
 "pair_meta": {
  "n_common": 500,
  "n_baseline_only": 0,
  "n_agent_only": 0,
  "baseline_only_examples": [],
  "agent_only_examples": [],
  "allow_mismatch": false
 }
}
```

## 4.4 paired:sel_clean500_vs_idm

```json
{
 "tag": "sel_clean500_vs_idm",
 "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-100413_v7p0_idm_clean500/episodes.csv",
 "agent": "runs/BTC20261002-211752_v7p1i3_screen_epoch005/episodes.csv",
 "rc": 0,
 "allow_mismatch": false,
 "out_dir": "/tmp/opencode/v7_p1i3/paired_eval/sel_clean500_vs_idm",
 "log": "/tmp/opencode/v7_p1i3/paired_sel_clean500_vs_idm.log",
 "n_paired": 500,
 "rate_base": 0.742,
 "rate_agent": 0.26,
 "delta_pp": -48.199999999999996,
 "ci95_pp": [
  -53.2,
  -43.2
 ],
 "net": -241,
 "z": 14.376854669152635,
 "fixed": 20,
 "broken": 261,
 "both_pass": 110,
 "both_fail": 109,
 "mcnemar_exact_p": 1.0795480871270188e-54,
 "verdict": {
  "delta_pp": -48.199999999999996,
  "z": 14.376854669152635,
  "mcnemar_exact_p": 1.0795480871270188e-54,
  "ci95_pp": [
   -53.2,
   -43.2
  ],
  "ci_lower_gt_zero": false,
  "positive_3pt_z196": false
 },
 "pair_meta": {
  "n_common": 500,
  "n_baseline_only": 0,
  "n_agent_only": 0,
  "baseline_only_examples": [],
  "agent_only_examples": [],
  "allow_mismatch": false
 }
}
```

## 4.4 paired:sel_clean500_vs_p1b

```json
{
 "tag": "sel_clean500_vs_p1b",
 "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-162639_v7p1b_screen_epoch010/episodes.csv",
 "agent": "runs/BTC20261002-211752_v7p1i3_screen_epoch005/episodes.csv",
 "rc": 0,
 "allow_mismatch": false,
 "out_dir": "/tmp/opencode/v7_p1i3/paired_eval/sel_clean500_vs_p1b",
 "log": "/tmp/opencode/v7_p1i3/paired_sel_clean500_vs_p1b.log",
 "n_paired": 500,
 "rate_base": 0.314,
 "rate_agent": 0.26,
 "delta_pp": -5.4,
 "ci95_pp": [
  -8.4,
  -2.4
 ],
 "net": -27,
 "z": 3.5151005964822444,
 "fixed": 16,
 "broken": 43,
 "both_pass": 114,
 "both_fail": 327,
 "mcnemar_exact_p": 0.0005843646280443461,
 "verdict": {
  "delta_pp": -5.4,
  "z": 3.5151005964822444,
  "mcnemar_exact_p": 0.0005843646280443461,
  "ci95_pp": [
   -8.4,
   -2.4
  ],
  "ci_lower_gt_zero": false,
  "positive_3pt_z196": false
 },
 "pair_meta": {
  "n_common": 500,
  "n_baseline_only": 0,
  "n_agent_only": 0,
  "baseline_only_examples": [],
  "agent_only_examples": [],
  "allow_mismatch": false
 }
}
```

## 4.4 paired:sel_eval500_vs_ebeta

```json
{
 "tag": "sel_eval500_vs_ebeta",
 "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-200331_v6p3_e3_eval500/episodes.csv",
 "agent": "runs/BTC20261002-215634_v7p1i3_eval500_sel/episodes.csv",
 "rc": 0,
 "allow_mismatch": false,
 "out_dir": "/tmp/opencode/v7_p1i3/paired_eval/sel_eval500_vs_ebeta",
 "log": "/tmp/opencode/v7_p1i3/paired_sel_eval500_vs_ebeta.log",
 "n_paired": 500,
 "rate_base": 0.44,
 "rate_agent": 0.274,
 "delta_pp": -16.6,
 "ci95_pp": [
  -20.4,
  -12.8
 ],
 "net": -83,
 "z": 8.023912859079006,
 "fixed": 12,
 "broken": 95,
 "both_pass": 125,
 "both_fail": 268,
 "mcnemar_exact_p": 3.485417306224237e-17,
 "verdict": {
  "delta_pp": -16.6,
  "z": 8.023912859079006,
  "mcnemar_exact_p": 3.485417306224237e-17,
  "ci95_pp": [
   -20.4,
   -12.8
  ],
  "ci_lower_gt_zero": false,
  "positive_3pt_z196": false
 },
 "pair_meta": {
  "n_common": 500,
  "n_baseline_only": 0,
  "n_agent_only": 0,
  "baseline_only_examples": [],
  "agent_only_examples": [],
  "allow_mismatch": false
 }
}
```

## 4.4 paired:sel_eval500_vs_idm

```json
{
 "tag": "sel_eval500_vs_idm",
 "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20260927-1839_eval500_baseline/episodes.csv",
 "agent": "runs/BTC20261002-215634_v7p1i3_eval500_sel/episodes.csv",
 "rc": 0,
 "allow_mismatch": false,
 "out_dir": "/tmp/opencode/v7_p1i3/paired_eval/sel_eval500_vs_idm",
 "log": "/tmp/opencode/v7_p1i3/paired_sel_eval500_vs_idm.log",
 "n_paired": 500,
 "rate_base": 0.756,
 "rate_agent": 0.274,
 "delta_pp": -48.199999999999996,
 "ci95_pp": [
  -53.0,
  -43.4
 ],
 "net": -241,
 "z": 14.694029336477586,
 "fixed": 14,
 "broken": 255,
 "both_pass": 123,
 "both_fail": 108,
 "mcnemar_exact_p": 1.883486921248398e-58,
 "verdict": {
  "delta_pp": -48.199999999999996,
  "z": 14.694029336477586,
  "mcnemar_exact_p": 1.883486921248398e-58,
  "ci95_pp": [
   -53.0,
   -43.4
  ],
  "ci_lower_gt_zero": false,
  "positive_3pt_z196": false
 },
 "pair_meta": {
  "n_common": 500,
  "n_baseline_only": 0,
  "n_agent_only": 0,
  "baseline_only_examples": [],
  "agent_only_examples": [],
  "allow_mismatch": false
 }
}
```

## 4.4 paired:sel_eval500_vs_p1b

```json
{
 "tag": "sel_eval500_vs_p1b",
 "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-164209_v7p1b_eval500_sel/episodes.csv",
 "agent": "runs/BTC20261002-215634_v7p1i3_eval500_sel/episodes.csv",
 "rc": 0,
 "allow_mismatch": false,
 "out_dir": "/tmp/opencode/v7_p1i3/paired_eval/sel_eval500_vs_p1b",
 "log": "/tmp/opencode/v7_p1i3/paired_sel_eval500_vs_p1b.log",
 "n_paired": 500,
 "rate_base": 0.312,
 "rate_agent": 0.274,
 "delta_pp": -3.8,
 "ci95_pp": [
  -6.6000000000000005,
  -0.8
 ],
 "net": -19,
 "z": 2.516611478423583,
 "fixed": 19,
 "broken": 38,
 "both_pass": 118,
 "both_fail": 325,
 "mcnemar_exact_p": 0.01634810389737615,
 "verdict": {
  "delta_pp": -3.8,
  "z": 2.516611478423583,
  "mcnemar_exact_p": 0.01634810389737615,
  "ci95_pp": [
   -6.6000000000000005,
   -0.8
  ],
  "ci_lower_gt_zero": false,
  "positive_3pt_z196": false
 },
 "pair_meta": {
  "n_common": 500,
  "n_baseline_only": 0,
  "n_agent_only": 0,
  "baseline_only_examples": [],
  "agent_only_examples": [],
  "allow_mismatch": false
 }
}
```

## 4.4 paired:sel_tg45_vs_idm

```json
{
 "tag": "sel_tg45_vs_idm",
 "baseline": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-160154_v7p1b_tg45_idm/episodes.csv",
 "agent": "runs/BTC20261002-215549_v7p1i3_tg45_sel/episodes.csv",
 "rc": 0,
 "allow_mismatch": false,
 "out_dir": "/tmp/opencode/v7_p1i3/paired_eval/sel_tg45_vs_idm",
 "log": "/tmp/opencode/v7_p1i3/paired_sel_tg45_vs_idm.log",
 "n_paired": 45,
 "rate_base": 0.7777777777777778,
 "rate_agent": 0.022222222222222223,
 "delta_pp": -75.55555555555556,
 "ci95_pp": [
  -86.66666666666667,
  -62.22222222222222
 ],
 "net": -34,
 "z": 5.8309518948453,
 "fixed": 0,
 "broken": 34,
 "both_pass": 1,
 "both_fail": 10,
 "mcnemar_exact_p": 1.1641532182693481e-10,
 "verdict": {
  "delta_pp": -75.55555555555556,
  "z": 5.8309518948453,
  "mcnemar_exact_p": 1.1641532182693481e-10,
  "ci95_pp": [
   -86.66666666666667,
   -62.22222222222222
  ],
  "ci_lower_gt_zero": false,
  "positive_3pt_z196": false
 },
 "pair_meta": {
  "n_common": 45,
  "n_baseline_only": 0,
  "n_agent_only": 0,
  "baseline_only_examples": [],
  "agent_only_examples": [],
  "allow_mismatch": false
 }
}
```

## 5. 判定汇总

```json
{
 "selected_ckpt": {
  "tag": "screen_epoch005",
  "ckpt": "/workspace/01_Proj/DRL_PathPlan/runs/BTC20261002-1837_v7p1i3/stage_b/ckpt_epoch005.pt",
  "clean500_success_rate": 0.26,
  "clean500_run_dir": "runs/BTC20261002-211752_v7p1i3_screen_epoch005",
  "sha256": "f555a787239bd3fe096b8fb5c929f5cc7eb53423a4771a86cafb1501c7298829",
  "rule": "clean500 success 最大；平局取更早 epoch"
 },
 "config_hash_A": "ec30e385878963bb",
 "config_hash_B": "ec30e385878963bb",
 "stage_A_threshold_s": 342.9,
 "stage_B_threshold_s": 203.9,
 "t3_n_s1_resolved": 0,
 "t3_criterion_met": false,
 "tg45_success_rate": 0.022222222222222223,
 "tg45_idm_ref_frozen": 0.7778,
 "tg45_gate": 0.6778,
 "tg45_criterion_met": false,
 "clean500_success_rate": 0.26,
 "clean500_ebeta_ref": 0.44,
 "eval500_success_rate": 0.274,
 "eval500_ebeta_ref": 0.44,
 "gateB_band": [
  0.55,
  0.6
 ],
 "gateB_clean_in_band": false,
 "gateB_eval_in_band": false,
 "paired_vs_p1b_clean500": {
  "delta_pp": -5.4,
  "ci95_pp": [
   -8.4,
   -2.4
  ],
  "mcnemar_exact_p": 0.0005843646280443461,
  "verdict": {
   "delta_pp": -5.4,
   "z": 3.5151005964822444,
   "mcnemar_exact_p": 0.0005843646280443461,
   "ci95_pp": [
    -8.4,
    -2.4
   ],
   "ci_lower_gt_zero": false,
   "positive_3pt_z196": false
  },
  "rate_base": 0.314,
  "rate_agent": 0.26
 },
 "paired_vs_p1b_eval500": {
  "delta_pp": -3.8,
  "ci95_pp": [
   -6.6000000000000005,
   -0.8
  ],
  "mcnemar_exact_p": 0.01634810389737615,
  "verdict": {
   "delta_pp": -3.8,
   "z": 2.516611478423583,
   "mcnemar_exact_p": 0.01634810389737615,
   "ci95_pp": [
    -6.6000000000000005,
    -0.8
   ],
   "ci_lower_gt_zero": false,
   "positive_3pt_z196": false
  },
  "rate_base": 0.312,
  "rate_agent": 0.274
 },
 "paired_vs_ebeta_clean500": {
  "delta_pp": -18.0,
  "ci95_pp": [
   -21.8,
   -14.2
  ],
  "mcnemar_exact_p": 2.262519224355612e-19,
  "verdict": {
   "delta_pp": -18.0,
   "z": 8.504200642707612,
   "mcnemar_exact_p": 2.262519224355612e-19,
   "ci95_pp": [
    -21.8,
    -14.2
   ],
   "ci_lower_gt_zero": false,
   "positive_3pt_z196": false
  },
  "rate_base": 0.44,
  "rate_agent": 0.26
 },
 "paired_vs_ebeta_eval500": {
  "delta_pp": -16.6,
  "ci95_pp": [
   -20.4,
   -12.8
  ],
  "mcnemar_exact_p": 3.485417306224237e-17,
  "verdict": {
   "delta_pp": -16.6,
   "z": 8.023912859079006,
   "mcnemar_exact_p": 3.485417306224237e-17,
   "ci95_pp": [
    -20.4,
    -12.8
   ],
   "ci_lower_gt_zero": false,
   "positive_3pt_z196": false
  },
  "rate_base": 0.44,
  "rate_agent": 0.274
 },
 "paired_vs_idm_clean500": {
  "delta_pp": -48.199999999999996,
  "ci95_pp": [
   -53.2,
   -43.2
  ],
  "mcnemar_exact_p": 1.0795480871270188e-54,
  "verdict": {
   "delta_pp": -48.199999999999996,
   "z": 14.376854669152635,
   "mcnemar_exact_p": 1.0795480871270188e-54,
   "ci95_pp": [
    -53.2,
    -43.2
   ],
   "ci_lower_gt_zero": false,
   "positive_3pt_z196": false
  },
  "rate_base": 0.742,
  "rate_agent": 0.26
 },
 "paired_vs_idm_eval500": {
  "delta_pp": -48.199999999999996,
  "ci95_pp": [
   -53.0,
   -43.4
  ],
  "mcnemar_exact_p": 1.883486921248398e-58,
  "verdict": {
   "delta_pp": -48.199999999999996,
   "z": 14.694029336477586,
   "mcnemar_exact_p": 1.883486921248398e-58,
   "ci95_pp": [
    -53.0,
    -43.4
   ],
   "ci_lower_gt_zero": false,
   "positive_3pt_z196": false
  },
  "rate_base": 0.756,
  "rate_agent": 0.274
 },
 "paired_vs_idm_tg45": {
  "delta_pp": -75.55555555555556,
  "ci95_pp": [
   -86.66666666666667,
   -62.22222222222222
  ],
  "mcnemar_exact_p": 1.1641532182693481e-10,
  "verdict": {
   "delta_pp": -75.55555555555556,
   "z": 5.8309518948453,
   "mcnemar_exact_p": 1.1641532182693481e-10,
   "ci95_pp": [
    -86.66666666666667,
    -62.22222222222222
   ],
   "ci_lower_gt_zero": false,
   "positive_3pt_z196": false
  },
  "rate_base": 0.7777777777777778,
  "rate_agent": 0.022222222222222223
 },
 "phase3_run": false,
 "phase3_skip_reason": "v2 dagger/anchor 无 static 段，会把 static 清零；待 v4 版 dagger 再议"
}
```

## 7. 结论、对照表与偏差记录

### 7.1 对照表（选定 ckpt = epoch005）

| 项目 | iter3（v4.1+[1,69.4]） | P1-B（v4+[1,1]） | E-β″ | IDM | 判据 | 判定 |
|---|---|---|---|---|---|---|
| T3 S1 解除 | **0/9** | 0/9 | — | 9/9 baseline 过闸 | =9 | FAIL |
| tollgate45 | **0.022** | 0.000 | — | 0.778 | ≥0.678 | FAIL |
| clean500 | **0.260** | 0.314 | 0.440 | 0.742 | Gate B 0.55–0.60 | FAIL |
| eval500 | **0.274** | 0.312 | 0.440 | 0.756 | Gate B 0.55–0.60 | FAIL |
| paired vs P1-B (clean500) | Δ **−5.4pp** CI[−8.4,−2.4] p=5.8e-4 | — | — | — | — | 显著更差 |
| paired vs P1-B (eval500) | Δ **−3.8pp** CI[−6.6,−0.8] p=0.016 | — | — | — | — | 更差 |

### 7.2 训练侧读数（stage_b val，specific phase；聚合口径）

| run | config hash | act_err（未加权） | act_err（加权） | traj MAE | mu_ds |
|---|---|---|---|---|---|
| P1-B | f75b1b3c42c320d6 | 0.2058 | 0.0375 | 0.3171 | 3.535 |
| iter3 | ec30e385878963bb | **0.1964** | 0.0353 | 0.3247 | 3.525 |

- 聚合动作误差仅小幅改善（−4.6%/−5.9%），traj 略回退（+2.4%）；`metrics.json` 无逐维
  dθ 误差字段 → 无法从训练读数验证 iter2 TF 探针（dθ MAE −60%、val T3 corr 0.84）在完整
  训练后是否保持。闭环回退与 TF 侧小幅改善并存，断点归因需后续 TF 探针/诊断（本轮未跑，未扩大验证范围）。

### 7.3 偏差与异常记录

- **无操作异常**：全链 rc=0；Stage A/B 无 abort、无 OOM、无 nav WARN；未改 repo（git clean）；
  训练/评测均 GPU 串行（每步 wait_lane）。
- A 阶段 `epochs_done=24` 为日志正则命中含 "epoch" 的 eval 行导致的已知计数伪影
  （20 真实 epoch + 4 个 0.0 间隔），与 P1-B 报告一致，非训练异常。
- clean500（验证集）用于 4 个候选选点，属 keep-best 协议；**eval500 仅对选定 ckpt 评估一次**
  （无二次触碰偏差，优于 P1-B2 的记录）。
- T3 判据按预注册：S1 解除 = 变道至自由车道 且 过闸净空 ≥1.5 m；本轮 2/9 满足前者、0/9 满足后者。
- phase3 本轮未跑（v2 dagger/anchor 无 static 段，会把 static 清零；待 v4 版 dagger 再议）。


