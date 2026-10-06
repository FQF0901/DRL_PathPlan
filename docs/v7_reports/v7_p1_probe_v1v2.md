# P1 复盘探针 V1+V2：plan 尾链 vs μ（定位闭环执行量断点）

- 日期：2026-10-02 22:28–22:48（北京）；GPU：RTX 4070 12GB，全程串行；只读 ckpt/env，写 `/tmp` + `runs/`。
- 探针目录：`/tmp/opencode/v7_p1_probe/`（脚本 + JSON + 日志）；结果 run 根：`runs/BTC20261002-22*_v7p1probe_*`。
- 两 ckpt：
  - **P1-B** `runs/BTC20261002-0941_v7p1b/stage_b/ckpt_epoch010.pt`（v4 数据、无 `action_dim_weights`；stage_b val `bc_traj_mae_m=0.3208`）
  - **i3** `runs/BTC20261002-1837_v7p1i3/stage_b/ckpt_epoch005.pt`（v4.1 数据、`action_dim_weights=[1.0, 69.4]`、`traj_aux_weight=0.1`；stage_b val `bc_traj_mae_m=0.3855`）

---

## 0. 结论（先给判定）

**M1（plan 尾链是断点）——模型侧成立、闭环侧部分成立，但闭环口径有一处关键退化需注明：**

1. **V2（模型侧，最强证据）**：i3 的改进**全部集中在 head/μ**（对同一专家帧：μ dθ MAE 0.02220→0.01527，−31%；head 横向 MAE 0.0203→0.0146，−28%），而 **plan 尾链没有改善**：`train_weight>0` 口径尾链横向 MAE 0.1703→**0.1787（+4.9% 更差）**、LQR 预瞄窗（第 2–3 步）heading 误差 0.0194→**0.0201（+3.6% 更差）**；all 口径尾链 MAE 持平/略好（0.2720→0.2638）但 ratio 同样恶化。**两口径一致**的稳健信号是：**尾链/头误差比 8.4→12.2（tw>0）/ 8.5→12.5（all）**，以及 **i3 尾链横向 bias 放大 5–10 倍**（第 2/3 步 −0.007→−0.045，第 6 步 −0.023→−0.096，系统性右偏；dθ 偏差呈振荡：−0.0042,+0.0061,−0.0049,+0.0009,+0.0078 而 P1B 近零）。→ **"μ 改善没有传导到闭环实际跟踪的尾链"成立**；i3 尾链幅值更接近专家（pred 0.0141 vs 专家 0.0157，P1B 仅 0.0115），但偏差结构更差。
2. **V1（闭环侧）**：预注册判据在 **clean150 形式上满足**（i3−P1B 的差：plan −8.67pp → repeat −1.33pp；交互 +7.33pp，CI95 [+0.88,+13.78] 不含 0）；但 **repeat_action 口径把双方都打到 ~0 成功率**（P1B 0.320→0.013，i3 0.233→0.000，off-road 89–92%），是**地板效应**，不能单独作为"尾链=断点"的闭环证明。tg45 上判据**不满足**（交互 −2.22pp，ns）；T3（9 条）双方全 0，无功效。
3. **消融揭示的两层机理**（比"尾链 vs μ"二选一更准确）：
   - **collision 组件由 plan 尾链介导**：i3 的 collision 超额（tg45 +13.3pp、T3 +22.2pp）在 repeat 下消失（+2.2pp、0pp）；
   - **rc 组件在 repeat 下反而放大**：clean150 rc 差 −0.060（plan）→ **−0.147**（repeat）；说明 i3 的 μ/A-hold 执行本身也受损（i3 μ 幅值大，A-hold 6 步外推发散）。
   - 尾链对 i3 是"rc 的救命绳、collision 的祸根"：plan 模式 rc 0.546（vs P1B 0.606），repeat 只剩 0.165（vs 0.312）。
4. **DAgger cycle 的直接输入**：尾链需要**直接监督 + bias 校正**（现仅 traj_aux=0.1 监督积分位置）；采集口径**不能用 repeat_action**（退化，已知 E-β′ repeat 零点 succ≈2%）；μ 的 dθ 仍有 −0.0043 rad/step 恒定欠转偏置，需一并处理。

---

## 1. V1 两口径闭环对照（12 run，GPU 串行，总墙钟 10m02s）

口径：`tools/venv-python tools/test.py --policy ckpt --ckpt <ckpt> --spec <spec> --out runs --name v7p1probe_* --workers 6 --tracker lqr --config config/default.yaml --eval-reference {plan|repeat_action}`，seed=0，确定性。

Spec（sha256）：
- tg45：`/tmp/opencode/v7_p1b_chain/specs_tollgate45.json`（45 条）`27010b0e…c523e6`
- T3：`/tmp/opencode/v7_p1_probe/specs_t3_9.json`（9 条 = eval500 ids 34,243,164,84,166,131,76,147,239）`ce235c9c…76f991`
- clean500 sub150：`/tmp/opencode/phase3_diag/exp/specs_val_only150.json`（150 条）`81f0f958…d4afa6`

### 1.1 结果表（overall；|dθ| = `action_dtheta_abs_mean_rad`，实际下发首动作 μ）

| spec | ckpt | ref | n | succ | collision | off_road | rc_mean | \|dθ\| | steer | mean_speed | run dir |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tg45 | P1B | plan | 45 | 0.0000 | 0.0667 | 0.9333 | 0.529 | 0.0081 | 0.339 | — | `runs/BTC20261002-222821_v7p1probe_p1b_tg45_plan` |
| tg45 | P1B | repeat | 45 | 0.0000 | 0.0000 | 0.9333 | 0.373 | 0.0054 | 0.208 | — | `runs/BTC20261002-222857_v7p1probe_p1b_tg45_repeat_action` |
| tg45 | i3 | plan | 45 | 0.0222 | 0.2000 | 0.8667 | 0.544 | 0.0534 | 0.517 | — | `runs/BTC20261002-222917_v7p1probe_i3_tg45_plan` |
| tg45 | i3 | repeat | 45 | 0.0000 | 0.0222 | 0.8444 | 0.237 | 0.0271 | 0.444 | — | `runs/BTC20261002-223004_v7p1probe_i3_tg45_repeat_action` |
| T3 | P1B | plan | 9 | 0.0000 | 0.1111 | 0.8889 | 0.606 | 0.0079 | 0.319 | — | `runs/BTC20261002-223027_v7p1probe_p1b_t3_plan` |
| T3 | P1B | repeat | 9 | 0.0000 | 0.0000 | 0.8889 | 0.354 | 0.0062 | 0.236 | — | `runs/BTC20261002-223041_v7p1probe_p1b_t3_repeat_action` |
| T3 | i3 | plan | 9 | 0.0000 | 0.3333 | 0.8889 | 0.635 | 0.0631 | 0.564 | — | `runs/BTC20261002-223051_v7p1probe_i3_t3_plan` |
| T3 | i3 | repeat | 9 | 0.0000 | 0.0000 | 0.8889 | 0.245 | 0.0237 | 0.402 | — | `runs/BTC20261002-223109_v7p1probe_i3_t3_repeat_action` |
| clean150 | P1B | plan | 150 | **0.3200** | 0.0867 | 0.5933 | 0.606 | 0.0083 | 0.379 | 4.93 | `runs/BTC20261002-223120_v7p1probe_p1b_clean150_plan` |
| clean150 | P1B | repeat | 150 | **0.0133** | 0.0067 | 0.8867 | 0.312 | 0.0053 | 0.200 | 2.99 | `runs/BTC20261002-223344_v7p1probe_p1b_clean150_repeat_action` |
| clean150 | i3 | plan | 150 | **0.2333** | 0.0867 | 0.6800 | 0.546 | 0.0600 | 0.566 | 3.86 | `runs/BTC20261002-223438_v7p1probe_i3_clean150_plan` |
| clean150 | i3 | repeat | 150 | **0.0000** | 0.0133 | 0.9200 | 0.165 | 0.0330 | 0.524 | 1.92 | `runs/BTC20261002-223731_v7p1probe_i3_clean150_repeat_action` |

复现性：新 plan 基线与历史 `runs/BTC20261002-215549_v7p1i3_tg45_sel`（0.0222/0.200/0.544/0.05342）和 `…164134_v7p1b_tg45_sel`（0.0/0.0667/0.529/0.00807）在 **episode 级逐条一致**（success/collision/off_road/steps/rc 相同）；仅 `duration_s`（墙钟）与少量 float 求和噪声不同。

### 1.2 配对对比（同一 (id,seed) 配对；Δ=成功率差，CI95 正态近似）

| spec | 对比 | Δ | CI95 | a_only/b_only |
|---|---|---|---|---|
| clean150 | i3−P1B @ plan | **−0.0867** | [−0.1480,−0.0254] | 5/18 |
| clean150 | i3−P1B @ repeat | **−0.0133** | [−0.0318,+0.0051] | 0/2 |
| clean150 | repeat−plan @ i3 | −0.2333 | [−0.3012,−0.1654] | 0/35 |
| clean150 | repeat−plan @ P1B | −0.3067 | [−0.3807,−0.2326] | 0/46 |
| clean150 | **交互 (i3−P1B) repeat−plan** | **+0.0733** | **[+0.0088,+0.1378]** | — |
| tg45 | i3−P1B @ plan | +0.0222 | [−0.0213,+0.0658] | 1/0 |
| tg45 | i3−P1B @ repeat | 0.0000 | [0,0] | 0/0 |
| tg45 | 交互 | −0.0222 | [−0.0658,+0.0213] | — |
| T3 | 全部 | 0.0000 | — | 双方全 0 |

**判据检验**（"若 i3 在 repeat_action 下相对变好（差缩小/翻转）⇒ plan 尾链是断点"）：
- clean150：**形式上满足**（差 −8.67pp→−1.33pp，交互 +7.33pp 显著）；但 repeat 臂 succ≤1.3%，属**地板效应**（见 §1.3），只能算弱证据。
- tg45：**不满足**（差 +2.22pp→0；交互 −2.22pp ns）。
- T3：无功效（全 0）。

### 1.3 口径退化警告（必须与结论一起读）

- `repeat_action`（参考 = `repeat(μ,6)`，A-hold 语义）在这三个 spec 上把**两个 ckpt 都打到 ~0 成功**（0.000–0.013），off-road 84–92%、clean150 速度掉到 1.9–3.0 m/s、rc 只有 plan 口径的 ~40–55%。这与历史已知口径一致（`rl_v3_posthoc.md`：E-β′ repeat 零点 succ=2.0% / off=97.3%）。
- 因此 **repeat 臂的绝对成功率没有解释力**；它只能作为"移除 plan 尾链"的**消融方向**：i3 的 collision 超额在该臂消失，但 rc 超额放大。
- 控制实验（`mu_consistency.py`）：`rollout=False`（repeat 路径）与 `rollout=True`（plan 路径）的 `action_mu` 对两 ckpt **逐位一致**（max diff 0.0），排除"cheap/full 前向差异"混淆；两口径的差异只来自跟踪参考。
- 注意 `action_dtheta_abs_mean_rad` 是 `references[0]=μ` 的 dθ，但两口径下闭环访问状态不同（plan 模式 0.0083/0.0600 vs repeat 0.0053/0.0330），**不是同一状态分布上的 μ 幅值对比**；plan 模式 i3 的 7.2× 放大（0.0083→0.0600）含状态分布效应。

### 1.4 clean150 失败模式构成（terminations）

| run | arrive_dest | collision | out_of_road | max_step | rc | speed |
|---|---|---|---|---|---|---|
| P1B plan | 0.320 | 0.087 | 0.593 | 0 | 0.606 | 4.93 |
| P1B repeat | 0.013 | 0.007 | 0.887 | 0.093 | 0.312 | 2.99 |
| i3 plan | 0.233 | 0.087 | 0.660 | 0.020 | 0.546 | 3.86 |
| i3 repeat | 0.000 | 0.013 | 0.920 | 0.067 | 0.165 | 1.92 |

i3 在 plan 模式比 P1B 多 6.7pp off-road、少 8.7pp 到达；repeat 模式双方都变成"爬行 + 出路"。

### 1.5 clean150 分组（by_primary，n,succ）

| geometry | P1B plan | i3 plan | P1B repeat | i3 repeat |
|---|---|---|---|---|
| uturn (18) | 0.222 | 0.056 | 0.000 | 0.000 |
| t_intersection (14) | 0.214 | 0.143 | 0.000 | 0.000 |
| intersection (12) | 0.417 | 0.083 | 0.000 | 0.000 |
| curve (14) | 0.214 | 0.000 | 0.000 | 0.000 |
| straight (11) | 0.273 | 0.273 | 0.000 | 0.000 |
| ramp_in (14) | 0.571 | 0.357 | 0.000 | 0.000 |
| ramp_out (16) | 0.500 | 0.562 | 0.000 | 0.000 |
| merge (12) | 0.250 | 0.417 | 0.083 | 0.000 |
| roundabout (15) | 0.267 | 0.133 | 0.000 | 0.000 |
| split (8) | 0.875 | 0.875 | 0.125 | 0.000 |
| tollgate (16) | 0.000 | 0.000 | 0.000 | 0.000 |

plan 模式复现了 i3 转弯回退（uturn −16.7pp、intersection −33.3pp、curve −21.4pp）；repeat 模式所有组≈0，无区分力（floor）。

---

## 2. V2 plan 尾链教师强制误差（同一专家帧，两 ckpt）

方法：`plan_tail_probe.py` 对 val 数据集全帧前向 `rollout=True, world_model=False`，取 `plan (6,2)`、`traj_xy (6,2)`、`action_mu`；积分自检 `max|integrate(plan)−traj_xy|≈1.1e-3 m`（float 精度）。误差口径：点误差沿**专家路径法向** = 横向（lat，左正/右负），切向 = 纵向；heading err = 累计 dθ 差。子集 `train_weight>0`（n=25869；headline），另有 all 口径（结论同向，见 JSON）。

### 2.1 v4（primary，n=25869；专家 |dθ| step0=0.03271）

**逐 step（lat_mae / lat_bias / heading_err_mae / dθ_mae / dθ_abs_pred vs exp）：**

| step | P1B lat / bias / head_err / dθmae | i3 lat / bias / head_err / dθmae | P1B dθ\|pred\|/\|exp\| | i3 dθ\|pred\|/\|exp\| |
|---|---|---|---|---|---|
| 0 (μ) | 0.0203 / −0.0029 / 0.0131 / 0.01310 | **0.0146** / −0.0065 / **0.0097** / **0.00966** | 0.0106 / 0.0153 | 0.0149 / 0.0153 |
| 1 | **0.0632** / −0.0061 / 0.0178 / 0.01330 | 0.0546 / **−0.0270** / 0.0182 / 0.01280 | 0.0122 / 0.0133 | 0.0116 / 0.0133 |
| 2 (LQR 窗) | 0.1115 / −0.0067 / 0.0211 / 0.01437 | 0.1075 / **−0.0449** / 0.0220 / 0.01661 | 0.0103 / 0.0141 | 0.0133 / 0.0141 |
| 3 | 0.1638 / −0.0057 / 0.0256 / 0.01559 | 0.1691 / **−0.0621** / 0.0265 / 0.01746 | 0.0098 / 0.0148 | 0.0143 / 0.0148 |
| 4 | 0.2218 / −0.0098 / 0.0301 / 0.01769 | 0.2406 / **−0.0865** / 0.0318 / 0.01836 | 0.0120 / 0.0164 | 0.0142 / 0.0164 |
| 5 | 0.2912 / −0.0228 / 0.0371 / 0.02105 | 0.3218 / **−0.0955** / 0.0389 / 0.02365 | 0.0130 / 0.0199 | 0.0171 / 0.0199 |

**聚合：**

| 指标 | P1B | i3 | Δ(i3−P1B) |
|---|---|---|---|
| μ dθ MAE | 0.02220 | **0.01527** | **−31%** |
| μ dθ bias | −0.00047 | −0.00431 | 恒定欠转 0.0043 rad/step |
| head lat MAE（step0） | 0.0203 | 0.0146 | −28% |
| head heading err（step0） | 0.0131 | 0.0097 | −26% |
| tail lat MAE（step1–5） | 0.1703 | 0.1787 | **+4.9%** |
| **tail/head 比** | **8.4** | **12.2** | 恶化 |
| LQR 窗（step1–2）lat | 0.0874 | 0.0810 | −7%（仍略好） |
| LQR 窗 heading err | 0.0194 | 0.0201 | **+3.6%（转差）** |
| tail heading err | 0.0263 | 0.0275 | +4.6% |
| tail dθ MAE | 0.01640 | 0.01778 | +8.4% |
| tail dθ 幅值 pred / exp | 0.01145 / 0.01571（−27%） | 0.01409 / 0.01571（−10%） | i3 幅值更准 |

**all 口径对照（n=36159，不做 train_weight 过滤）**：head 0.0321→0.0211（−34%）、tail 0.2720→0.2638（−3.0%，i3 略好）、**tail/head 比 8.5→12.5（同样恶化）**、LQR 窗 lat 0.1279→0.1199 / heading 0.0411→0.0381（i3 略好）、但 **lat bias 仍为 P1B 的 5–10 倍**（step2 −0.0087→−0.0437、step5 +0.0020→−0.0695）、μ bias −0.0005→−0.0043。
→ 稳健结论是"**改进未传导到尾链（ratio 恶化），且 i3 尾链带大系统偏置**"；"尾链绝对误差变差"仅在 `train_weight>0` 成立（+4.9%），all 口径下 i3 尾链 MAE 持平/略好。

**分层（tail lat MAE；tw>0）：**

| 组 | n | P1B | i3 | Δ |
|---|---|---|---|---|
| turning | 8560 | 0.1803 | 0.1923 | +6.7% |
| non_turning | 10783 | 0.1595 | 0.1653 | +3.6% |
| other | 6526 | 0.1751 | 0.1830 | +4.5% |

逐 geometry Δ(i3−P1B) tail：t_intersection +0.0180、curve +0.0127、split +0.0121、uturn +0.0115、straight +0.0104、ramp_out +0.0085、tollgate +0.0061、intersection +0.0059、roundabout +0.0050、ramp_in +0.0032、merge +0.0013 —— **全面转差，转弯/路口最重**。

### 2.2 v41（镜像验证，n=26474）

| 指标 | P1B | i3 |
|---|---|---|
| μ dθ MAE | 0.02221 | 0.01526 |
| head lat MAE | 0.0222 | 0.0153 |
| tail lat MAE | 0.1813 | 0.1846 |
| tail/head 比 | 8.2 | 12.1 |
| LQR 窗 lat | 0.0944 | 0.0845 |
| LQR 窗 heading err | 0.0215 | 0.0212 |
| tail heading err | 0.0282 | 0.0284 |
| tail dθ MAE | 0.01706 | 0.01835 |
| turning tail | 0.1927 | 0.1984 |
| μ dθ bias | −0.00046 | −0.00430 |

结论与 v4 逐项同向（v4/v41 行数 36159/36148，不混用）。all 口径 v41 也与 v4 all 完全同构（ratio 8.5→12.5、bias 放大 5–10×）。

### 2.3 V2 读数

1. **头尾监督不对称被量化**：同一策略、同一帧上，尾链横向误差是头部的 **8–12 倍**；head 由 action loss 直接监督（i3 的 dθ 权重 69.4× 只作用于它），尾链只经 `traj_aux=0.1` 的积分位置损失 → i3 的收益全部落在 head。
2. **i3 尾链的问题不只是幅值，而是偏置**：dθ 幅值 i3 更接近专家（−10% vs −27%），但**横向 bias 从第 2 步起比 P1B 大 4–6 倍且单调增长**（−0.027→−0.096），dθ 逐 step 偏差正负交替（振荡）。闭环 LQR 跟踪一条持续偏一侧的参考 → 稳态转向偏置，与转弯组 collision/off-road 回退方向一致。
3. **LQR 预瞄窗是分界点**（tw>0）：step1–2 横向 MAE 仍略好（−7%），但 heading 误差已转差（+3.6%），step3 起横向全面转差 → 与"预瞄 Ld≈1–1.5 s ≈ plan 第 2–3 步"的 M1 设定吻合：μ 的改善在第 1 步后开始衰减，第 3 步后反被尾链误差主导。all 口径下预瞄窗仍略好，但 bias 结构（5–10×）不变。

---

## 3. 判定综合（回答预注册问题）

| 命题 | 证据 | 判定 |
|---|---|---|
| plan 尾链是 i3 闭环回退的断点 | V2：head −28%/−31% 改善、tail +4.9% 变差（tw>0；all 持平）、ratio 8.4→12.2 / 8.5→12.5、LQR 窗 heading 转差（tw>0）、bias 放大 5–10× | **模型侧成立** |
| 闭环判据"i3 在 repeat 下相对变好" | clean150 交互 +7.33pp [CI 不含 0]；tg45 ns；T3 无功效；repeat 臂地板（~0%） | **弱成立（floor 混淆）** |
| i3 的 collision 超额由尾链介导 | plan 模式超额 +13.3/+22.2pp → repeat 0/+2.2pp | **成立** |
| i3 的 rc 缺损也来自 μ/A-hold 执行 | repeat 下 rc 差反而放大（−0.060→−0.147）；|dθ| 6.2× P1B；速度 1.92 vs 2.99 | **成立（第二断点）** |
| repeat_action 可作为"纯 μ"闭环参照 | 双方 succ 0–1.3%、off 84–92%、rc 崩塌；μ cheap/full 已验证一致 | **不可，仅可作消融方向** |

---

## 4. 对 DAgger cycle 的设计输入

1. **尾链直接监督（首要）**：现尾链只吃 `traj_aux=0.1` 的积分位置 MSE。建议加/提高 **action-chain 损失**（`plan[:,1:6]` vs 专家 `action[:,1:6]`，dθ 维沿用 i3 的权重），或在转弯帧提高 traj 权重；否则 dθ 加权只会继续只改善 head（V2 已示）。
2. **bias 校正**：i3 尾链横向 bias 随步数线性放大（右偏），dθ 偏差振荡；在 DAgger 数据中按几何（uturn/t_intersection/curve/intersection）过采样并用**专家 action chain** 作为监督目标，优先修 bias 而非再压 MAE。
3. **采集口径**：用 `plan` 口径收集（repeat 退化，会教出爬行/出路策略）；但 DAgger 的状态分布应落在 **i3 plan 尾链诱发的失败态**（转弯/岗亭碰撞）上——这恰好是闭环 `plan` 模式访问到的状态，故"on-policy plan 收集 + 尾链直接监督"是自洽的 cycle 设计。
4. **μ 侧残差**：i3 μ dθ bias −0.0043 rad/step（欠转）与闭环 7× 幅值放大并存；cycle 里保留一个 μ bias 监控项（V2 脚本已产出）。
5. **验证口径**：后续用 V1 的 clean150 sub150 + tg45 + T3 三件套（总墙钟 ~10 min）作为 cycle 前后闭环对照；report 时不要单独引用 repeat 臂成功率。

---

## 5. 异常 / 口径注释（如实）

1. **repeat_action 退化**：clean150/tg45/T3 上 succ 0–1.3%、off-road 84–92%，与历史已知零点（~2%）一致；这是口径性质，不是本探针 bug。
2. **T3 无功效**：9 条双方在两种口径下 succ 全 0（collision 有差：i3 plan 0.333 vs P1B 0.111 → repeat 0 vs 0）；如需 T3 结论需用 S1 签名（`t3_measure.py`）而非 succ。
3. **`action_dtheta_abs_mean_rad` 的解释**：它是 `references[0]=μ`，但两口径闭环状态不同，不能当"同一状态上的 μ 幅值"用；μ 幅值的干净对照见 V2（同帧）。
4. **v4/v41 行数不同**（36159 vs 36148）且 v41 放宽了 roundtrip LC 约束；两数据集分开报告，结论一致。
5. **`squeeze_single_slot`/数据历史**：v4 无 `od_hist` 等 mem 历史数组，`build_obs_batch` 走 `_entries` 重拼路径（与训练侧一致），非异常。
6. V1 的 plan 基线（tg45）与历史 `_sel` run **episode 级逐条一致**（success/collision/off_road/steps/rc），证明本轮环境/确定性无漂移；`duration_s`（墙钟）不同属预期。

---

## 6. 产物清单

脚本与 JSON（全部在 `/tmp/opencode/v7_p1_probe/`）：
- `run_v1.sh`（12 run 串行驱动）、`collect_v1.py`、`v1_results.json`（含配对 CI）、`v1_logs/*.log`（每 run 原始日志）
- `run_v2.sh`、`plan_tail_probe.py`、`plan_tail_probe_v4.json`、`plan_tail_probe_v41.json`、`v2_v4.log`、`v2_v41.log`、`v2_v4b.log`、`v2_v41b.log`
- `mu_consistency.py`、`mu_consistency.json/.log`（cheap/full μ 逐位一致）
- `specs_t3_9.json`（T3 9 条子集）

V1 run dirs（`runs/`，manifest/argv/metrics/episodes 齐全）：见 §1.1 表；共 12 个 `BTC20261002-22*_v7p1probe_*`。

总耗时：V1 10m02s（≤25min 预算），V2 ≈5min（≤30min 预算），全程 GPU 串行，无 repo 改动。
