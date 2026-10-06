# v7 结构迭代 B：v5 + K-anchor 重训链报告（A → B → 筛查 → dagger → phase3 → 终评）

- 生成：2026-10-05T14:11:09+0800 · 状态：**DONE**
- 仓库 HEAD（启动时）：`0065264a87f79f84d0355743bb2e325b1888ff5f` · stamp：`20261005-0856`
- AB run：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5` · P3 run：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3` · dagger：`/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261005-0856_phase3_dagger_v5`
- 预注册：`docs/v7_program_prereg.md` §11（锚字典 sha `79829ef7…`）

## 0. 预检

- ok=True：v5 schema v5（train datasets/BTC20261005-0814_expert5k_v5 / val datasets/BTC20261005-0814_expert500val_v5）；fix-8 报告在；pins/spec/参照/臂配置/干净树/GPU 全部通过
- pins：
  - `datasets/BTC20261005-0814_expert5k_v5/expert_bc.npz` = `8076ecf4731c046f8309f12c271600287f6c15471307ce9d20fd4c5d3d71dcdd`
  - `datasets/BTC20261005-0814_expert500val_v5/expert_bc.npz` = `fb5e58a5622813c4cc8ea06085653099fe5d3982a43268f7e4cb8f0a14fadbde`
  - `/tmp/opencode/phase3_diag/exp/specs_val_only500.json` = `087db3f5c02e9cedccb8313733207f59dfa90162002b0d9bf002e5b80bcc366e`
  - `/tmp/opencode/phase3_diag/exp/specs_val_only150.json` = `81f0f95814e3ea8bb8c349e47b2a19121ea78d578a59264fcc8bb1a660d4afa6`
  - `env/specs/scenarios_eval500.json` = `98856105eca17461bbdabbf88f203102be7b585fd3de82820b17460358dd4595`
  - `/tmp/opencode/v7_p1b_chain/specs_tollgate45.json` = `27010b0ebfb59d1daf45570d5851cf47452ab3d5eddbc9d9f3ddfc6d98c523e6`
  - `env/specs/scenarios_train_5k.json` = `211575f09675975c200a6bac73cd50090b7836a7eb3515bc93393780814fb91e`
  - `config/plan_anchors_k6.json` = `79829ef705285c79a60252e9db6ac58d1b1e48830a4fe4d32327a1fa0f3873bf`
  - `sub150==w1_first150` = `True`
  - `w1_clean500_episodes_sha256` = `042b63304b9c8504c7ee55f8c3c469b8f11f1fe3c14eacc891c16e6415779a72`
  - `w1_clean500_succ` = `0.526`
  - `w1_eval500_episodes_sha256` = `6597d2eae18b88ee7155b917eef1f608b9da4a3b2916e17f967db79b010e88a7`
  - `w1_eval500_succ` = `0.53`
  - `s11_clean500_episodes_sha256` = `6419fc8f2ff88282c4be3257b528270f0c9e0f3119d94adf13d77d9ffccb3f94`
  - `s11_clean500_succ` = `0.668`
  - `s11_eval500_episodes_sha256` = `41790134c94637d56cfcf8a2a63491dbe8c78e292996f2becd528f5ac84173f5`
  - `s11_eval500_succ` = `0.646`
  - `idm_clean500_episodes_sha256` = `9bf57b196abc49bcc881835abd3d0d966cabd9e09cd78f72f794953840594eb8`
  - `idm_clean500_succ` = `0.742`
  - `idm_eval500_episodes_sha256` = `d6269ef238f11f9b0064d6d1b5d6671a53dbf5260c52ea2f1e41509a17676e7c`
  - `idm_eval500_succ` = `0.756`
  - `p1b_clean500_episodes_sha256` = `f11de6476d879e788a153af104c49248bc587923c846c26e8b61bc4c9e0b18f3`
  - `p1b_clean500_succ` = `0.314`
  - `p1b_eval500_episodes_sha256` = `d671cd66cfd45b500ac8f0a14d13d6d33aecf201ae5d9ede0fd2a1be96f91367`
  - `p1b_eval500_succ` = `0.312`
  - `idm_tg45_episodes_sha256` = `367ffff92e289e40b20ef3dde97ec8b11e822d1db509b103656c333b553e679b`
  - `idm_tg45_succ` = `0.7778`
  - `p1b_tg45_episodes_sha256` = `None`
  - `p1b_tg45_succ` = `0.0`
  - `w1_tg45_episodes_sha256` = `None`
  - `w1_tg45_succ` = `0.0`
  - `s11_tg45_episodes_sha256` = `None`
  - `s11_tg45_succ` = `0.0`
  - `config_arm` = `config/arms/v7_struct_b_v5_train.yaml`
  - `config_model` = `config/arms/v7_struct_b_v5_model.yaml`
  - `config_eval` = `config/arms/v7_struct_b_v5_eval.yaml`

## Stage A（v5 expert5k · 20ep · micro256）

- rc=0 · duration=6214.2s · epochs=24 · peak=9001MiB · aborted=None · oom=False · nav WARN=0
- epoch 墙钟：[331.2, 298.0, 293.0, 290.5, 290.9, 0.0, 294.6, 292.9, 291.6, 291.7, 292.4, 0.0, 293.6, 291.4, 292.0, 291.8, 291.1, 0.0, 293.2, 291.6, 291.7, 295.0, 294.2, 0.0]（基线 254s × 1.35 = 343s）
- log：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5/logs/stage_a.log`

## Stage B（v5 · 20ep · micro512 · plan_anchor enabled）

- rc=0 · duration=4086.3s · epochs=20 · peak=8407MiB · aborted=None · oom=False · nav WARN=0
- epoch 墙钟：[187.2, 188.3, 187.0, 189.1, 188.0, 190.9, 193.8, 197.8, 198.4, 190.6, 185.3, 185.5, 186.2, 185.4, 185.4, 186.2, 185.7, 185.6, 186.4, 186.2]（基线 151s × 1.35 = 204s）
- log：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5/logs/stage_b.log`

### Stage B K-anchor 表示层读数（§11.3 主判据）

```json
{
 "primary_ce": {
  "first": 0.588115,
  "last": 0.309417,
  "min": 0.309417,
  "max": 0.588115,
  "n": 10
 },
 "primary_wta": {
  "first": 0.162072,
  "last": 0.105384,
  "min": 0.105384,
  "max": 0.162072,
  "n": 10
 },
 "primary_assign_frac_last": {
  "c0": 0.7513639570633345,
  "c1": 0.032788869025180425,
  "c2": 0.024466000666192842,
  "c3": 0.08539759291721133,
  "c4": 0.07371614558923008,
  "c5": 0.032267434802170175
 },
 "primary_plan_ade_m": {
  "first": 0.327044,
  "last": 0.243078,
  "min": 0.243078,
  "max": 0.327044,
  "n": 10
 },
 "specific_ce": {
  "first": 0.356644,
  "last": 0.333361,
  "min": 0.333361,
  "max": 0.356644,
  "n": 10
 },
 "specific_wta": {
  "first": 0.130342,
  "last": 0.123851,
  "min": 0.123851,
  "max": 0.130342,
  "n": 10
 },
 "specific_assign_frac_last": {
  "c0": 0.7513639570633345,
  "c1": 0.032788869025180425,
  "c2": 0.024466000666192842,
  "c3": 0.08539759291721133,
  "c4": 0.07371614558923008,
  "c5": 0.032267434802170175
 },
 "specific_plan_ade_m": {
  "first": 0.292266,
  "last": 0.283053,
  "min": 0.283053,
  "max": 0.292266,
  "n": 10
 }
}
```

## phase3（v5 dagger + v5 锚 · specific_only · mild=1.0 · 5ep）

- rc=0 · duration=1276.8s · epochs=5 · peak=3685MiB · aborted=None · oom=False · nav WARN=0
- epoch 墙钟：[223.7, 186.7, 169.6, 173.3, 172.5]（基线 200s × 1.35 = 270s）
- log：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3/logs/stage_b.log`

## 筛查（Stage B ckpt → dagger 学生）

| ckpt | sub150 succ | sub150 net vs w1 | z | clean500 succ | clean500 net vs w1 | z |
|---|---|---|---|---|---|---|
| ckpt_epoch005.pt | 0.03333333333333333 | -73 | 8.54 | — | — | — |
| ckpt_epoch010.pt | 0.26 | -39 | 6.09 | 0.254 | -136 | 11.33 |
| ckpt_epoch015.pt | 0.2 | -48 | 6.79 | 0.216 | -155 | 11.92 |
| ckpt_epoch020.pt | 0.16 | -54 | 7.22 | — | — | — |
| final.pt | 0.16 | -54 | 7.22 | — | — | — |

- **采纳**：`ckpt_epoch010.pt`（keep-best：全量 clean500 配对 vs w1 net 采纳（禁止 sub150 直采））
- ckpt sha256：`2f463a2186e41f523328902d26fcdcc1512d7885f6ea44fe18db0dc56ed9d55c`

## 筛查（phase3 ckpt → 终评候选）

| ckpt | sub150 succ | sub150 net vs w1 | z | clean500 succ | clean500 net vs w1 | z |
|---|---|---|---|---|---|---|
| ckpt_epoch005.pt | 0.0 | -78 | 8.83 | 0.0 | -263 | 16.22 |
| final.pt | 0.0 | -78 | 8.83 | 0.0 | -263 | 16.22 |

- **采纳**：`ckpt_epoch005.pt`（keep-best：全量 clean500 配对 vs w1 net 采纳（禁止 sub150 直采））
- ckpt sha256：`842b8e34bf27450bb1a44b3ee7c83d60dfbec310099e56dbf293dc83956e589b`

## dagger 采集（v5）

- rc=0 · dir=`/workspace/01_Proj/DRL_PathPlan/datasets/BTC20261005-0856_phase3_dagger_v5` · rows=20080 · trainable=20008 · schema v5 · elapsed=1978.823s
- 学生 ckpt：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5/stage_b/ckpt_epoch010.pt`（sha `2f463a2186e4…`）· tracker=lqr(plan)（plan 口径）· 教师=pure_pursuit
- 链覆盖：（label_scope=action_chain_t0_t5）：relabeled=147843 行 · 全链（6/6 步）=139293 行 · 缺失链步=25650 步（NaN 尾；训练侧按有限性 mask）
- 失败/scanned：1004 fails / 1710 specs（target=1000）

## 终评（phase3 采纳 ckpt）

- ckpt：`/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-0856_v7struct_v5_p3/stage_b/ckpt_epoch005.pt`（sha `842b8e34bf27…`）
- clean500 succ=0.0（n=500，err=0）· eval500 succ=0.0（n=500，err=0）· tg45 succ=0.0

### 配对（tools/paired_eval.py）

**clean500**（agent `/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-134825_v7sb_p3_ckpt_epoch005_clean500`）

| baseline | n | base succ | Δpt | z | p | 95% CI |
|---|---|---|---|---|---|---|
| w1 | 500 | 0.526 | -52.6 | 16.217274740226856 | 1.349401336733507e-79 | [-56.99999999999999, -48.199999999999996] |
| s11 | 500 | 0.668 | -66.8 | 18.275666882497063 | 5.714936956411375e-101 | [-70.8, -62.8] |
| idm | 500 | 0.742 | -74.2 | 19.26136028425822 | 4.15816390625796e-112 | [-78.0, -70.19999999999999] |
| p1b | 500 | 0.314 | -31.4 | 12.529964086141668 | 1.0947644252537633e-47 | [-35.4, -27.400000000000002] |

**eval500**（agent `/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-140006_v7sb_final_eval500`）

| baseline | n | base succ | Δpt | z | p | 95% CI |
|---|---|---|---|---|---|---|
| w1 | 500 | 0.53 | -53.0 | 16.278820596099706 | 3.3735033418337674e-80 | [-57.4, -48.6] |
| s11 | 500 | 0.646 | -64.60000000000001 | 17.97220075561143 | 1.1704190886730496e-97 | [-68.8, -60.4] |
| idm | 500 | 0.756 | -75.6 | 19.44222209522358 | 3.248565551764031e-114 | [-79.4, -72.0] |
| p1b | 500 | 0.312 | -31.2 | 12.489995996796797 | 2.1895288505075267e-47 | [-35.4, -27.200000000000003] |

**tg45**（agent `/workspace/01_Proj/DRL_PathPlan/runs/BTC20261005-140438_v7sb_final_tg45`）

| baseline | n | base succ | Δpt | z | p | 95% CI |
|---|---|---|---|---|---|---|
| idm | 45 | 0.7777777777777778 | -77.77777777777779 | 5.916079783099616 | 5.820766091346741e-11 | [-88.88888888888889, -64.44444444444444] |
| p1b | 45 | 0.0 | 0.0 | None | 1.0 | [0.0, 0.0] |
| w1 | 45 | 0.0 | 0.0 | None | 1.0 | [0.0, 0.0] |
| s11 | 45 | 0.0 | 0.0 | None | 1.0 | [0.0, 0.0] |

- T3：n_s1_resolved=0/9（有效）

### 判读

- 表示层主判据 未全部满足；闭环描述性 clean500 Δ vs w1=-52.6pt
- §11 表示层主判据：{"ce_le_0.5ln6": true, "ce_last": 0.333361, "wta_converged": false, "wta_first_last": [0.162072, 0.123851], "assign_noncollapse": true, "assign_frac_last": {"c0": 0.7513639570633345, "c1": 0.032788869025180425, "c2": 0.024466000666192842, "c3": 0.08539759291721133, "c4": 0.07371614558923008, "c5": 0.032267434802170175}}
- 闭环（描述性方向）：clean500 0.0（Δ vs w1 Nonept，z=None）；eval500 0.0；tg45 0.0
- 明确不承诺 tollgate 0/45 修复（§11.3）；tg45 只记录。
- 参照 runs（w1/arm1-s11/IDM/P1-B）为 pre-v5 代码期产物；本候选评测在 v5 代码 + plan_anchor 开启下执行（§11 次判据口径，差异如实记录）。

## 异常与备注

- phase3 epoch1 234.1s 触发 abort（原基线 145s×1.35=196s，来自 v6 P3 链）。复核：本链 phase3 组成 = v5 锚 360,572 + 窗口 20,080 = 380,652 行；同组成实测（P1 DAgger cycle，v4 锚+窗口）epoch1=207s（data=36s 含一次性装载）、稳态 175–181s；本链 epoch1 data=69.7s（一次性）→ 基线改锚 200s（同组成上包络）×1.35=270s 后 resume。属操作参数修正（任务未对 phase3 指定 abort 阈值），不改任何判据；原 abort 现场保留在 P3/logs/stage_b.log 与 driver log。
- 评测配置 `config/arms/v7_struct_b_v5_eval.yaml`（plan_anchor.enabled=true）；eval/dagger 模型构造透传 plan_anchor 的接线修复 commit `3ac92d7`（缺省行为不变）。

- 驱动日志：`/tmp/opencode/v7_struct_retrain_chain.log` · monitor：`/tmp/opencode/v7_struct_retrain_monitor.log` · 状态：`/tmp/opencode/v7_struct_retrain_status.txt`

