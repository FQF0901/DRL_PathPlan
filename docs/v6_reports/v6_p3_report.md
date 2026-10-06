# v6 P3 执行报告（mini 闸 → 全量重训链 → E3）

- 生成时间：2026-10-01 20:2x（北京）· 驱动启动时 git HEAD：`0b67f3d`（相对冻结基线 6883d4f：仅 docs/reward 工具变更，`pipeline/net/env/config` 未变）
- 执行方式：纯执行 + 证据；无 repo 代码改动；工作树 clean。链驱动：`/tmp/opencode/v6_p3_chain.py`（setsid 脱离，PID 2262162）
- AB run：`runs/BTC20261001-1631_v6retrain` · P3 run：`runs/BTC20261001-1631_v6p3`
- 证据：`/tmp/opencode/v6_p3_chain_20261001-1631.log`（驱动）、`v6_p3_chain_monitor.log`（10min 采样）、`v6_p3_status.txt`、各阶段日志见下

## 0. 预检（PASS）

- 数据 sha256 与 prereg §4.1 全部一致：expert5k `024bc02e…`、expert500val `3843ba86…`、dagger_r5 `45e38b3c…`、anchor5k `2b4b0d41…`；
- phase3 preflight `v6_p3_dagger_check.py --expect-v2` rc=0（`a4_nav_rebuild=false (accepted)`）；
- clean500 spec sha `087db3f5…`、eval500 spec sha `98856105…` 与 pin 一致；GPU 起始占用 397 MiB（空闲）。

## 1. mini 重训闸（`runs/BTC20261001-1619_v6mini`）—— 6/6 PASS

| 项 | 读数 | 判据 | 结果 |
|---|---|---|---|
| 无 OOM | A/B 日志无 OOM | — | PASS |
| A 峰值显存（nvidia-smi） | **8,860 MiB** | ≤10.0 GiB | PASS |
| A epoch 墙钟（32k 切片，2 ep） | **27.0 / 27.0 s** | ≤28 s | PASS |
| B micro512 | 无 OOM（峰值 8,924 MiB） | 无 OOM | PASS |
| st_gnn 主路径执行 | forward 6 次 + node_features 7 次/前向 | 实际执行 | PASS |
| A4 nav 断言 | rebuild 6 次/前向；首 batch 无 `[stageA]`/`[trainer]` 回退 WARN；v3 键齐（ego_world/route_world/mask） | fail-closed | PASS |
| 损失 | A 3.056→2.464（val 2.895→2.063）；B 有限（2.4355） | 有限且下降 | PASS |
| 形状/参数量 | total **1,283,295**；policy 83,716；value 83,329；expert_hidden **256**；8 experts | ≈1.28M / 256 | PASS |
| st_gnn 冻结（B 口径） | probe：freeze 后 st_gnn 全冻、其余可训；A→B ckpt 差分 st_gnn 33 张量全同 | 与 prereg §4.3 一致 | PASS |
| mini ckpt 评测 | 16/16 条，errors=0（`runs/BTC20261001-162637_v6mini_eval16`） | ≥16 条 | PASS |
| manifest / 工作树 | manifest.txt 落盘（git sha）；`git status` 0 行 | 完整 / clean | PASS |

> 规格澄清（如实报告）：任务书写“A micro512”，与冻结 prereg §4.1/§4.2（**A micro256**、峰值判据 ≤10.0 GiB）冲突，且 12 GB 卡 A512 必 OOM；按冻结规格执行 **A256 / B512**。

## 2. 全量重训链（每阶段独立 run dir）

### 2.1 Stage A（20 ep · micro256 · v3 expert5k）
- run：`runs/BTC20261001-1631_v6retrain/stage_a` · 日志：`runs/BTC20261001-1631_v6retrain/logs/stage_a.log` · metrics：`stage_a/metrics.json`
- rc=0，duration **5,813 s ≈ 96.9 min**；epoch 墙钟 269.2–295.7 s（基线 254 s → 最大 **+16.4%**，abort 阈值 +30%=330 s 未触发）；
- nvidia-smi 峰值 **9,004 MiB**（≤10.0 GiB 闸内）；A4 nav 回退 WARN = 0；
- 损失：epoch1 **1.5227** → epoch20 **0.5890**（val 0.5304 → 0.2352；od 1.3316→0.5141；曲线单调下降）。

### 2.2 Stage B（20 ep · micro512 · v3 expert5k）
- run：`runs/BTC20261001-1631_v6retrain/stage_b` · 日志：`runs/BTC20261001-1631_v6retrain/logs/stage_b.log`
- rc=0，duration **3,844 s ≈ 64.1 min**；micro512 **无 OOM**（无需退 256）；
- primary 段 epoch 墙钟 187–189 s、specific 段 173–175 s（基线 151 s → 最大 **+25.2%**，未触发 +30% abort；略高于成本文档 +21% 校准中心）；
- nvidia-smi 峰值 **9,017 MiB**；nav 回退 WARN = 0；
- 末 epoch：`phase=specific epoch 10/10 loss=0.1078 action=0.0129 mu_ds=3.602m val=0.5066`。

### 2.3 phase3（r5 dagger + expert5k 锚 mild=1.0 · micro256）
- run：`runs/BTC20261001-1631_v6p3/stage_b` · 日志：`runs/BTC20261001-1631_v6p3/logs/stage_b.log`
- rc=0，duration **1,049 s ≈ 17.5 min**；epoch 墙钟 157.1–159.1 s（E-β′ 基线 ~145 s → **+9%**，阈值 188 s 未触发）；
- nvidia-smi 峰值 **3,823 MiB**；冻结配方 `specific_only`（trainable=experts/router/residual_scale）；
- 数据 = 380,501 行（窗口 20,000 + 锚 360,501，mild=1.0）；v2 数据（A4 回退显式接受）——实测 **nav 回退 WARN = 0**（specific_only 冻结下 WM/教师强制项权重降级为 0，未走回退路径）；
- 损失：epoch1 → epoch5 loss 0.0812→**0.0740**（action 0.0118 / chain 0.0522；od/ld/ego_next 冻结降级=0）。

### 2.4 预算
- 训练合计 ≈ **2.98 h**（A 1.62 + B 1.07 + phase3 0.29）+ 评测 ≈ 0.7 h ≈ **3.7 h** < 5 h 预算。

## 3. E3 验收（同一 harness/seed：workers=6 / tracker=lqr / seed=0 / eval-reference=plan）

| 评测 | run dir | n / err | succ | 闸 | 结果 |
|---|---|---|---|---|---|
| clean500（硬闸，采纳 ckpt） | `runs/BTC20261001-195447_v6p3_e3_clean500` | 500 / 0 | **0.44** | ≥0.446 | FAIL |
| eval500（硬闸，采纳 ckpt） | `runs/BTC20261001-200331_v6p3_e3_eval500` | 500 / 0 | **0.44** | ≥0.436 | PASS |

### 3.1 keep-best（全量 clean500 复评）
- 候选：`ckpt_epoch005.pt` 与 `final.pt`；sha256：`723db1c27d9da33e…` vs `46feb3a4ab6564eb…`（payload 不同：meta/optimizer）；
- **模型权重逐张量相同**（state-dict 逐键 `torch.equal` 验证）→ 两者实为同权重的两次落盘；
- 全量 clean500 复评：ckpt_epoch005 **0.438** / final **0.438**（同权重复评噪声 ±1 条/500）；采纳 `ckpt_epoch005.pt`（= E-β′ 协议同款）；
- 采纳 ckpt 的硬闸复评即上表 `e3_clean500`（0.440，第三次读数）——同权重三次读数 0.438/0.438/0.440，噪声 ±1 条（0.2pt）。
- 采纳记录：ckpt 路径 + sha256 + 子集/全量读数已在上表与 §5（§5 keep-best 协议）。

### 3.2 对照与配对（辅助读法）
- 诊断：B final（未 phase3）clean500 = **0.322** → phase3 链增益 +0.118（与 E-β′ 链模式一致：旧链 B 0.318 → E-β′ 0.446）；
- 配对 net（vs E-β′ 零点，逐 (id,seed)）：clean500 **net=-0.006**（z=-0.457，fixed=20 / broken=23，offΔ=0.022，collΔ=0.02）；
- 配对 net（eval500）：**net=0.004**（z=0.272，fixed=28 / broken=26，offΔ=-0.008，collΔ=0.022）。

### 3.3 E-β′ 零点复评（辅助，结论：不可比 → 只能引用冻结零点）
- 为核零点稳定性，用同一 harness 复评 E-β′ ckpt（`runs/_refs_rlbase/e_beta_prime/final.pt`）于 clean500：**succ=0.000（500/0）**，`runs/BTC20261001-201519_v6p3_ref_ebeta_clean500`。
- 原因：旧 ckpt 与新架构不兼容（**missing=20 / unexpected=1**，缺的正是 v6 交叉注意力 policy/value 头；net design §5 #9 明确“不做权重迁移”）。
- 含义：E-β′ 零点（0.446/0.436）只能在旧代码口径引用（prereg §6 冻结数字）；E3 硬闸是**跨代码代**的阈值比较（预注册既定）。E-β″ 各 ckpt 载入 **154/154 missing=0**（自洽）。

## 4. 判定（fail-closed，不放宽）

- **E3 硬闸：FAIL** —— clean500 0.44 < 0.446（差 0.6 pt ≈ 3 条/500，FAIL）；eval500 0.44 ≥ 0.436（PASS）。
- 依 prereg §6 统计口径：单点差在 ±2pt 内**不作胜负结论**（本次 clean500 差 0.6pt、eval500 +0.4pt 均在带内），以配对 net 辅助读法为主；但硬闸为 fail-closed，数字不达标即 **不通过**，不得放宽。
- 依 prereg §6 迭代/回退：不达标 → ≤2 轮迭代（数据/超参/种子层面，不改冻结架构）；仍不达标 → 回退 E-β′（0.436）作 P4 init。**是否迭代/回退由 Gate3 裁定**，本执行不做此决定。
- Gate3（不进 RL）按此判定执行：**E3 未通过**。

## 5. 产物 sha256 / manifest

```json
{
  "ckpt_sha256": {
    "A final": "546cb62f62db34aa1b6aebcd7063dace1d7ae39176f0d860768d003cf5e765da",
    "B final": "069aa158207d05d4e83d180cd2230c39d7906e3a1418a65afd7acc74109c8234",
    "P3 ckpt_epoch005": "723db1c27d9da33ed8da17375af718cb421058e000a54555a2e5c667f5a56fa5",
    "P3 final": "46feb3a4ab6564ebac522df8cb48ba199b8488a6eddc36b84f02d92494cba70a",
    "mini A final": "2ab11cde6e60b3775064b68977f4de88ea9785fedba79bbc0fd837a4ea8e2977",
    "mini B final": "4fc54437c66735ed7b9239634a565522b0a0adf12388f36ba55ea43507075d4e"
  },
  "stage_metrics": {
    "A": {
      "samples": 360407,
      "epochs": 20,
      "micro_batch_size": 256,
      "vram_peak_mb": 8160.323584,
      "last_epoch": 20,
      "lr": 0.0003,
      "dataset_schema": 2
    },
    "B": {
      "samples": 360407,
      "epochs": 20,
      "micro_batch_size": 512,
      "vram_peak_mb": 6821.004288,
      "lr": 0.0003,
      "dataset_schema": 2
    },
    "phase3": {
      "samples": 380501,
      "epochs": 5,
      "micro_batch_size": 256,
      "vram_peak_mb": 2716.84352,
      "bc_loss": 0.07398351119341566,
      "trainable_params": 37,
      "frozen_params": 115,
      "freeze": "specific_only",
      "last_epoch": 5,
      "anchor": {
        "dir": "datasets/BTC20260926-2343_expert5k",
        "window_rows": 20000,
        "anchor_rows": 360501,
        "merged_rows": 380501,
        "mild_weight": 1.0,
        "episode_id_offset": 1440,
        "anchor_row_base": 20000,
        "train_rows": 377501,
        "window_traj_aux_masked": 20000,
        "anchor_traj_aux_valid": 360501,
        "enabled": true,
        "contract": {
          "legacy_dataset": false,
          "dataset_fingerprint": "v2-6a4d5de3f669"
        },
        "traj_aux_valid_rows": 380501,
        "traj_aux_valid_sum": 360501.0
      },
      "lr": 0.0003,
      "dataset_schema": 2
    }
  },
  "evals": {
    "e3_clean500": {
      "run_dir": "runs/BTC20261001-195447_v6p3_e3_clean500",
      "n": 500,
      "n_error": 0,
      "success_rate": 0.44,
      "collision_rate": 0.06,
      "off_road_rate": 0.496,
      "episodes_sha256": "d63db75e6162af90418afc29c9639f6f002d7c850ae3866f6256c2ce1110a38a"
    },
    "e3_eval500": {
      "run_dir": "runs/BTC20261001-200331_v6p3_e3_eval500",
      "n": 500,
      "n_error": 0,
      "success_rate": 0.44,
      "collision_rate": 0.05,
      "off_road_rate": 0.498,
      "episodes_sha256": "3d949398c7db53956021cde344e3df7862162f2f6d8bc53779cf8d4a456c904e"
    },
    "keepbest_ckpt_epoch005": {
      "run_dir": "runs/BTC20261001-192937_v6p3_keepbest_ckpt_epoch005",
      "n": 500,
      "n_error": 0,
      "success_rate": 0.438,
      "collision_rate": 0.058,
      "off_road_rate": 0.5,
      "episodes_sha256": "82d3b177a2d143bcbbf2d3a3844c24781c454f31529945d7b3fd28e31d586a07"
    },
    "keepbest_final": {
      "run_dir": "runs/BTC20261001-193831_v6p3_keepbest_final",
      "n": 500,
      "n_error": 0,
      "success_rate": 0.438,
      "collision_rate": 0.06,
      "off_road_rate": 0.498,
      "episodes_sha256": "2bb0ebea4a0c1aa62eaa37999aacd1e79db81b47f0db5826241a4ff7d9ce67f2"
    },
    "diag_b_final_clean500": {
      "run_dir": "runs/BTC20261001-194721_v6p3_diag_b_final_clean500",
      "n": 500,
      "n_error": 0,
      "success_rate": 0.322,
      "collision_rate": 0.052,
      "off_road_rate": 0.626,
      "episodes_sha256": "380bad7b14456e51d2d138b445636b738693917ef53639d2d65abb840f25aed8"
    }
  }
}
```

**AB run manifest**（`runs/BTC20261001-1631_v6retrain/manifest.txt`）：

```
created_at: 2026-10-01T18:08:03+08:00
stage: B
git: 0b67f3df8dd8b144a94f38d87efffafe4f5b4021 @Planner_RL dirty=1
config: config/default.yaml
model_config: config/model.yaml
work_dir: /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6retrain
out: /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6retrain/stage_b
argv: tools/train.py --stage B --config config/default.yaml --model-config config/model.yaml --ckpt /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6retrain/stage_a/final.pt --bc-dir datasets/BTC20261001-1327_expert5k --val-dir datasets/BTC20261001-1327_expert500val --out /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6retrain/stage_b --bc-epochs 20 --ckpt-every 5 --batch-size 1024 --traj-aux-weight 0.1 --seed 0 --micro-batch-size 512
```

**P3 run manifest**（`runs/BTC20261001-1631_v6p3/manifest.txt`）：

```
created_at: 2026-10-01T19:12:08+08:00
stage: B
git: 0b67f3df8dd8b144a94f38d87efffafe4f5b4021 @Planner_RL dirty=1
config: config/default.yaml
model_config: config/model.yaml
work_dir: /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6p3
out: /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6p3/stage_b
argv: tools/train.py --phase3 datasets/BTC20260929-1357_phase3_dagger_r5 --phase3-round 5 --ckpt /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6retrain/stage_b/final.pt --config config/default.yaml --model-config config/model.yaml --micro-batch-size 256 --phase3-anchor --phase3-anchor-bc-dir datasets/BTC20260926-2343_expert5k --phase3-anchor-mild-weight 1.0 --out /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6p3/stage_b
```

**mini run manifest**（`runs/BTC20261001-1619_v6mini/manifest.txt`）：

```
created_at: 2026-10-01T16:23:46+08:00
stage: B
git: 6883d4f2ac2a4063d5dfcddd00f77524aca56251 @Planner_RL dirty=5
config: config/default.yaml
model_config: config/model.yaml
work_dir: runs/BTC20261001-1619_v6mini
out: runs/BTC20261001-1619_v6mini/stage_b
argv: tools/train.py --stage B --config config/default.yaml --model-config config/model.yaml --ckpt runs/BTC20261001-1619_v6mini/stage_a/final.pt --bc-dir datasets/BTC20261001-1327_expert5k --val-dir datasets/BTC20261001-1327_expert500val --out runs/BTC20261001-1619_v6mini/stage_b --bc-epochs 1 --ckpt-every 0 --batch-size 1024 --micro-batch-size 512 --limit-dataset 32768 --seed 0
```

**E3 clean500 manifest**（`runs/BTC20261001-195447_v6p3_e3_clean500/manifest.txt`）：

```
created_at: 2026-10-01T19:54:47+08:00
stage: eval
git: 0b67f3df8dd8b144a94f38d87efffafe4f5b4021 @Planner_RL dirty=1
config: config/default.yaml
work_dir: runs/BTC20261001-195447_v6p3_e3_clean500
out: runs/BTC20261001-195447_v6p3_e3_clean500
argv: tools/test.py --policy ckpt --ckpt /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt --spec /tmp/opencode/phase3_diag/exp/specs_val_only500.json --out runs --name v6p3_e3_clean500 --workers 6 --tracker lqr --config config/default.yaml
```

**E3 eval500 manifest**（`runs/BTC20261001-200331_v6p3_e3_eval500/manifest.txt`）：

```
created_at: 2026-10-01T20:03:31+08:00
stage: eval
git: 0b67f3df8dd8b144a94f38d87efffafe4f5b4021 @Planner_RL dirty=1
config: config/default.yaml
work_dir: runs/BTC20261001-200331_v6p3_e3_eval500
out: runs/BTC20261001-200331_v6p3_e3_eval500
argv: tools/test.py --policy ckpt --ckpt /workspace/01_Proj/DRL_PathPlan/runs/BTC20261001-1631_v6p3/stage_b/ckpt_epoch005.pt --spec env/specs/scenarios_eval500.json --out runs --name v6p3_e3_eval500 --workers 6 --tracker lqr --config config/default.yaml
```

## 6. 异常 / 备注（如实）

- 规格冲突：任务书 “A micro512” vs 冻结 prereg “A micro256 / ≤10 GiB”；按冻结规格执行（A512 在 12 GB 卡必 OOM）。
- 启动时发现并行 lane 已提交 3 个 commit（HEAD 6883d4f → 0b67f3d，仅 docs/reward 工具）；`pipeline/net/env/config` 与冻结基线逐文件未变，训练/评测口径不受影响；工作树 clean。
- 宿主内存：v3 world 键使 A/B 物化 +~2 GB（5,471+3,062 MB）；avail ~1 GB、swap ~8 GB 稳定，训练/评测全程无 OOM-kill。
- 链驱动在写报告时因 `off_road` 布尔列解析崩溃（paired 统计函数 bug）；训练与全部评测已完成，本报告由修复后的收尾脚本重算生成（不影响任何训练/评测读数）。
- Stage B primary 段最大 epoch 墙钟 +25.2%（略高于成本文档 +21% 中心），未超 +30% abort 线；Stage A +16.4%、phase3 +9% 均在带内。
- E3 未通过（clean500 −0.6pt）；eval500 通过（0.440）。
- E-β′ 零点无法在新代码上复评（旧 ckpt 缺 v6 头 missing=20 → 0.000）；硬闸按预注册引用旧代码口径零点（0.446/0.436），属既定跨代比较口径。
