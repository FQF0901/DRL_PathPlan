# 锁版记录（LOCKS）

- **2026-09-30（用户拍板）**：Stage A 与 Stage B（phase1/2/3）锁版；Stage C（RL）自此开工。
  - `stageA-lock-v1`：A = 20 epochs + `ld_coef=0`（详情 `config/train.yaml` stages.A；评测口径 eval500）。
  - `stageB-lock-v1`：B = phase1 primary（60 可训）/ phase2 specific（37 可训）/ phase3 DAgger 机制（R1 续训 / R4 基座护栏 / keep-best；anchor 默认关；LD 不计损失）；评测口径 eval500 + val-only500。
  - `rl-baseline-start`：Stage C（RL）开工基线——**其后所有 RL 改动以本 tag 为 diff 基线**（逐版本提交，便于 review）。
  - 权重：A/B 链产物见 `runs/BTC20260929-0425_fixA_nold/`；RL init 候选 = `runs/_refs_rlbase/e_beta_prime/final.pt`（0.436；sha256 见该目录 README）。
