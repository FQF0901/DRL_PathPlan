# Round-33C 总报告（Action Conditioning V2 = B0/B1；四任务）

> 依据：`docs/deepseekv4p1_argue.md` §5.43（GPT `49e04d7` 批准：B0/B1 + 四任务 + 三分支决策树 + 路线图登记）。
> 纪律：生产改动**默认关 + 逐位兼容**（Gate 0 6/6 PASS）；零其他改动；评测链 v2 适配自证（npz BIT-EXACT）。
> 产物：`runs/round33/canary_v2/*`（ckpt/eval_b1/gate3_b1/gate4_b1）、`runs/round33/{c_gain_chain_b1,c_stratified_b1}.*`；脚本 `.slim/deepwork/p1_v2/round33/{canary_v2_train,v2_compat_check,c_gain_chain,c_stratified}.py` + `run_r33c_b1*.sh`。

## 0. 执行摘要

- **生产 v2**（`net/` +52/−42）：动作特征 3 维 `[ds/10, dθ/1.2, κ/1.2]`（删共线 speed-intent）+ **删可学习 α，残差增益固定 1** + 惰性构造保持；Gate 0：默认关=HEAD 逐位、启用=4 键、零初始化恒等、梯度隔离、966 tests passed。
- **训练器 v2**：`canary_v2_train.py`（arm=b1 = action_ego + ego_phys 8 键；ego 物理目标 v2 `[delta_progress, delta_speed, delta_heading]`；C-A expert-only v8；M0 e2 init）。
- **B1 训练**：22,532 步/4 ep（~30 min）；final latent 0.0406 / ego_phys 0.0080 / action_absmax 0.620。
- **评测链 v2 适配**（canary_eval / p1_deploy_record / eval_a3）：`--arm b1`；AC1 血统 ckpt fail-fast；自证 BIT-EXACT。

## 1. 硬门表（GPT 列表）

| 门 | B1 | 判定 |
|---|---|---|
| Action Swap top-1 ≥ 0.35 | **0.192**（base 0.101） | ✗ |
| Action Swap top-3 ≥ 0.85 | **0.914**（base 0.635） | ✓ |
| ds-progress 单调率 ≥ 0.90 | **0.60**（base 0.743） | ✗ |
| 接口恶化 ≤ 3%（E_if/E_action/overshoot） | **−10.1% / −11.9% / −12.3%**（全部改善） | ✓✓ |
| lane-relative 不退化 | lane_lat +0.006 / lane_head +0.054 | ✓ |
| OD ego 泄漏不增 | od+ld_pool prog **+7.3%** / speed **+19.1%**；od_pool prog **+46.9%** | ✗（部分） |
| success ≥ 0.58 | **0.560**（3 翻正/3 翻负） | ✗ |
| off-road ≤ 0.14 | **0.200**（+3 新：merge×2〔7579/7608〕+ tollgate×1〔7790〕；修复 0） | ✗ |
| collision ≤ 0.08 | **0.060**（4→3；收费站 2→1） | ✓ |
| timeout ≤ 11 | **9** | ✓ |
| route completion ≥ 0.862 | **0.855** | ✗ |
| 停止条件（接口恶化>5% 或碰撞再恶化） | 接口改善、碰撞改善 | 未触发 |

## 2. Gate 1/2 明细（vs base）

| 指标 | base | B1 | Δ |
|---|---|---|---|
| top-1 / top-3 | 0.101 / 0.635 | 0.192 / 0.914 | +0.091 / +0.279 |
| FD dθ(0.01)/ds(0.10) ‖∂z_ego‖ mean(k) | 38.74 | **2.98**（k1 49.8→1.9；k6 26.7→4.6） | **−35.8**（门 ≤10× ✓✓） |
| ds 单调（WM probe, ds±0.25） | 0.743 | 0.60 | −0.143 |
| ΔR²_WM state prog / speed | +0.0119 / +0.0065 | **+0.0272 / +0.0295** | 全臂最佳 |
| lane 保留 lat / head | 0.456 / 0.415 | 0.461 / 0.469 | 改善 |
| 泄漏 od+ld_pool prog / speed | — | +0.0259 / +0.0422 | +7.3% / **+19.1%** |
| 泄漏 od_pool prog | — | +0.0552 | **+46.9%** |

## 3. Gate 3/4 + 分层

**Gate 3（50-ep 同协议，vs base 1.66m）**：E_if 1.734→**1.558（−10.1%）**；E_action 1.619→**1.427（−11.9%）**；overshoot 1.660→**1.456（−12.3%）**；E_real 0.343→0.328（−4.3%）⇒ **四项全改善**（对比 AC1：+20.9%/+17.2%/+16.5% 恶化）。

**分层（primary；base→B1）**：tollgate succ 0→0（off 3→4、coll 2→1、rc 0.697 持平）；**merge succ 1→0、off 2→4**；t_intersection succ 3→5（+0.4）、tmo 2→0；ramp_out succ 4→3；straight coll 0→1；curve rc 0.793→0.815。

**收费站专项（5 ep）**：coll 2→1（7784 仍在）；min_ttc min 1.43→2.28、mean 4.84→5.49；7784 末决策步：B1 v_ego 1.92、d_stop_plan 4.02m vs 障碍 12.9m（**纵向余量 +8.84m 转正**）却仍碰撞 ⇒ **非纯纵向问题（横向路径/几何嫌疑）**。

**碰撞清单（B1，3 例）**：7511（uturn 侧碰）、7583（curve 侧碰）、7784（收费站岗亭，no_od_candidate）。

## 4. 四增益链（lane B；h=6 主读数）

| 增益 | B0 | **B1** | AC1 |
|---|---|---|---|
| G_WM(h=6) | 24.78 | **57.87** | 77.38 |
| G_backbone(h=5) | 1.189 | **0.4969** | 0.4804 |
| G_head(h=5) | 0.1849 | **0.2132** | 0.1816 |
| G_total(h=6) | 0.0216 | **0.1571** | 0.3483 |

- **B1 把 G_total(h6) 从 AC1 的 0.348 降到 0.157（≈−55%）**，仍高于 base ×7.3；G_backbone/head 与 AC1 基本持平 ⇒ **放大仍在 WM 侧**（延续 lane B 的 B0/AC1 判读；GPT"backbone/head 每单位解释过强"假设未被支持）。

## 5. 分析

1. **v2 = 稳定器而非推进器**：接口/overshoot 被修复（AC1 恶化 +17~21% → B1 改善 −10~−12%）、coll/tmo 改善；但 succ 未动（0.56）、off-road 回退（merge/tollgate）、top-1/单调/泄漏未过门。
2. **收费站层（0 succ，两跑皆然）**：B1 纵向停车余量已转正却仍碰撞 ⇒ 下一轮须做**路径级**归因（plan κ/dθ 剖面 vs 车道/岗亭几何），不能只补纵向。
3. **泄漏（od_pool +46.9%）**：Gate 2 防线失败——"action 只进 ego transition"不足以防消息传递串扰（与 R32/R33 同型）。
4. **单调 0.60**（AC1 0.202 / AC1-P 0.946）：v2 的 phys 监督部分改善方向性但未达门；与 FD 比值 2.98 形成"敏感度已好、方向性未好"的组合。
5. **off-road 回退定位**：3 新例（merge 7579/7608、tollgate 7790）——7790 由碰撞转 off（避碰但出路）；merge 两例为汇入区。

## 6. 决策建议（按 GPT 三分支 + 补充；待裁）

- 分支 1（全门过→60/40）：✗ 不适用。
- 分支 2（动作过+接口恶化→Policy 接口稳健训练）：✗ 不适用（接口改善；动作未全过）。
- **分支 3（仍收费站碰撞→静态障碍显式表示 / 走廊占用 / path-conditioned 纵向剖面；不启 RL）：适用**（收费站仍 1 碰撞）。
- **我方补充（3 条）**：①先做**收费站/merge 路径级只读归因**（plan κ/dθ 剖面 vs 几何；merge off 2→4）；②泄漏防线重估（od_pool +46.9% 未过门）；③**B1 不晋级**（不作新 baseline），保留为**诊断参考**（接口修复价值已证）。
- 候选下一步：R34 = 静态障碍显式表示 + 走廊占用 + path-conditioned 纵向剖面（+ 上述只读归因先行）。
