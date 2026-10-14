# Round-34A 总报告（路径级 + 输入契约 + hooks 只读审计）

> 依据：`docs/deepseekv4p1_argue.md` §5.45（GPT `82e815b` 批准 34A：只读、6 类分类、≥90% 归因、bit-exact）。
> 验收：**归因覆盖率 18/19 = 0.947 ✓**；默认关/开关 bit-exact ✓；干预可复现 ✓；坐标系/单位/mask 文档化 ✓（hooks 缺口清单见 §3）。
> 产物：`runs/round34/{d1_path_attribution,d2_static_contract,d3_leakage_causality}.{json,md}`、`runs/round34/d3_hooks_note.md`；脚本 `.slim/deepwork/p1_v2/round34/d*.py`。

## 1. D1 路径级归因（19 ep：b1 11 + base 8；覆盖率 18/19=0.947）

**分类表（规则写死 + 证据列）**

| 类 | n | episodes | 判定要点 |
|---|---|---|---|
| **目标通道选择错误** | **6** | b1 7784/7789/7795/7797、base 7784/7790 | 末窗路径最近接近位姿落在岗亭车道；自车/路径末段同在该车道（未完成必需变道）；同 road 有空闲车道 |
| **路径本身越界** | **6** | b1 7559/7579/7580/7608、base 7559/7580 | 路径前向扫掠首次无效 ≤ 执行首次无效且持续（≥2/5 末窗步）或实质（≥3 m） |
| **路径本身撞障碍** | **6** | b1 7511/7583/7790、base 7789/7795/7797 | 路径与静态物接触真值或矩形距离 ≤0.3m（静态）/≤1.5m（OD 车） |
| 路径可行但 LQR 跟踪出界 | **0** | — | — |
| 静态障碍未进模型 | **0** | — | 全部 present=1 |
| 障碍进模型但 Policy 未响应 | **0** | — | 全部有横向响应 |
| 未归因 | 1 | base/7579 | max_step 超时（纵向进度类，超 6 类口径） |

**必答问题**：**Q1 收费站失败路径级占比 = 10/10 = 1.00**（跟踪级 0/10）；**Q2 merge off-road 路径级 6/7 = 0.86**（跟踪级 0/7）——路径前向首次无效比执行早 26–837 env step；末窗路径横向余量 ≤0.01m 或已离道，而执行仍留 1.0–1.5m；跟踪残差仅 0.09–0.90m ⇒ **横向路径为主、LQR 跟踪为次**。

**代表证据**：b1/7784（收费站碰撞）——路径最近接近位姿落在岗亭车道（min_dist 0.0m），探针 `crash_building=True`，且碰撞步实采 obs 静态段 `present=1, gap=2.25m, rel=0` ⇒ **岗亭在模型输入内、Policy 有横向响应却仍驶入**（路径级车道选择错误）；b1/7559（merge 越界）——路径首无效比执行早 49–74 步、路径横向余量 −1.75m vs 执行 +1.32m；b1/7511——OD 车距路径 1.09m、闭合速度 −5.41 m/s。

**自证**：A 链 `lane_lat` 重算 vs 录制最大误差 **0.0m**（命中率 1.0）；B 链同协议复跑 3 ep **轨迹 exact（0.0m）**、`env__lqr_preview_index` 重算 **1326/1326 逐位一致**、obs 静态段 **1326/1326 一致**。**发现既有工具 bug**：`forensics_closed_loop._make_on_lane_fn` 固定 heading=(1,0) 且未关 `ray_localization` 默认 filter ⇒ 反向车道漏判（已绕开并记录）。

## 2. D2 静态障碍输入契约（8 runs）

- **表示**：岗亭**确实进 obs**（`others[16:21]` static 5 维：present/gap_norm/rel_bucket；8/8 恒为 TollGateBuilding，gap 误差 ≤1.4e-6m）。
- **消费**：网络确实消费（置零干预 → enc_ctx relΔ 0.71–0.81、latent 0.13–0.19）。
- **衰减 + 方向**：PolicyHead 衰减到 **|Δds| 0.031–0.060m**（≈0.15× 自然漂移）；**方向 = presence→加速（与安全相反）**；corridor_in 反向加速；zero 去障碍→减速（29/35 步）。
- **判定**：`present_static_5d` / `consumed_head_attenuated`（7/8）/ `presence_increases_ds`（8/8）⇒ **"表示缺失"归因不成立**；但 static 仅为 5 维摘要（无形状/精确横向）——34B 若要形状级避让仍有依据。
- **自证**：默认关 bit-exact（40 键×2ep）；复跑 812 步×6 数组 max|Δ|=0；干预可复现。

## 3. D3 泄漏因果性 + hooks

- **泄漏伤害闭环：无证据**（预注册：均值/点二列 ≈0、CI 含 0、两臂不同向）；附秩口径 12/12 弱不利相关（中位 ρ=−0.335，长度/tollgate 控制后仍在，主要来自 WM 侧）——**非因果**，建议 34B/35 加干预对照结案。
- **hooks 38 项审计**：纯缺口=effective rank / slot permutation / ckpt hash / schema version；部分=normalized obs、LD slot identity、frame/units/valid_range 元数据、ego pose delta、anchor warp 前后、**部署侧 PolicyBackbone latent / plan 累计 pose / LQR reference 遥测**等；34B 前最小实现 M1–M7 已列（范式=默认关 + bit-exact）。

## 4. 综合结论（34A 验收答案）

1. **失败是路径级的**：tollgate 10/10、merge 6/7；**LQR 跟踪 0 例、表示缺失 0 例、未响应 0 例**（GPT 假设的"表示缺失"支路被 D2 否证）。
2. **最大失败类 = 目标通道/车道选择错误（6/19）**——路径驶入岗亭车道且未完成必需变道；次之路径越界（6，merge 为主）与路径撞障碍（6）。
3. **静态障碍信号存在但被衰减 + 方向反转**（D2）——path-conditioned 纵向监督（34B）可解。
4. **泄漏非因果**（D3）——R35 优先级不因此上调。
5. **34B 设计输入**：横向头必须含**路线/车道目标条件化**（否则通道选择失败不解）；路径须**走廊/可行性约束**；纵向须**路径条件化 + 正确符号**；**LQR 保持不动**（0 例跟踪问题）。

## 5. 34B 建议范围（待确认）

- **最小原型**：横向 κ 3 控制点（near/mid/far）→ 平滑插值 κ(s) → 积分路径；纵向 v_ref[6]（输入=候选路径 + 路径上静态障碍距离 + 走廊占用 + 路线/导航意图）；外部兼容 `dθ_t = κ_t·ds_t`（LQR/环境/评测不变）。
- **34A 驱动的新增条件**：①route/lane-target token（通道选择）；②走廊占用/边界（越界约束）；③静态障碍在路径上的距离（符号正确监督）。
- **预注册**：同数据/同 epoch **A/B vs 现 policy**；可回滚（默认关）；失败即回退。
