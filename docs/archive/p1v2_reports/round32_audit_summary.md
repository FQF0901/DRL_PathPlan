# Round-32 Level-1 汇总（WM Architecture Audit；全只读、零训练）

> 依据：`docs/deepseekv4p1_argue.md` §5.37（GPT `d86e91a` 三级决策树；Level 1 先行）。
> 统一口径：K1e2 + **MoE-off** + K-anchor-off + static_role_only + kappa-compose；baseline 0.56（仅开发参考）。
> 产物：`runs/round32/{p1_dchain, p2_action_fd, p3_ego_sufficiency, p4_lipschitz}.{json,md}`；decision 草案 `runs/round32/decision.md`。

## 0. 结论速览（Level 1）

1. **P1（D0–D3 部署链四段归因）**：**主瓶颈段 = D2→D3（latent transition）≈97%**（overall share）；D0→D1（pose/执行）2.9%、D1→D2（图几何）0.0% ⇒ **协议支路与动态图支路被正式排除为主因**。
2. **P2（Action Fidelity）**：**动作条件化失效**——ds 通道 FD 敏感度 26–50× 弱于 dθ/κ 通道；Action Swap matched top-1=10.2%、margin≈0（近随机）⇒ WM 的 next-state 预测几乎不区分动作。
3. **P3（Ego State Sufficiency / Gate C）**：**real latent 充分**（progress/speed/heading R² 0.89–0.98；lane 目标 ld_slots 视图 0.90/0.93）⇒ 排除 GO_STATE_REPRESENTATION_FIX；**WM 保留率：ego-dynamics ✓（0.91–1.10）、lane-relative ✗（0.61/0.77 <80%）**；附带发现：WM 的 od/ld latent 泄漏 ego 运动量（ld_slots 上 WM 解码 progress/speed R²≈0.9–0.99）。
4. **P4（Gate D 接口）**：A_k 温和增长（p50 0.098→0.206，×2.1；分子 ×3.1、分母 +44%）；Lipschitz 中位比 WM/real 0.73–0.78（**不过敏**）、尾部 P95 ≈5.7–6.1（尾部放大）⇒ 接口问题为**次要/尾部**。
⇒ **Level-1 判定指向：`GO_ACTION_CONDITIONING`（主）**；lane 信息保留（Level-2：ego 物理锚/车道锚候选）与接口尾部为次。

## 1. P1：D0–D3 部署链归因（`runs/round32/p1_dchain.*`；50 ep / MoE-off 正式）

- 录制：`runs/p1_v2/round32/r32_K1e2_moeoff`（R4 扩展：+raw OD/LD/ego 字段；record 08:38–08:40）；replay 21.5s；D0 重放自检 = 0；旧 R4 50-ep 对账 D0 逐位复现（Δ=0.00000）。

**Gate E（ego 1−cos；overall share）**：

| 段 | overall | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|---|
| D0→D1（pose/执行） | **0.029** | 0.003 | −0.004 | 0.019 | 0.040 | 0.074 |
| D1→D2（图几何） | **0.000** | 0.000 | −0.000 | 0.000 | 0.001 | 0.001 |
| D2→D3（latent transition） | **0.970** | 0.997 | 1.004 | 0.981 | 0.959 | 0.925 |

**各层 E_state（ego 1−cos，跨 ep mean）**：D0 0.191→0.298、D1 0.190→0.276、D2 0.190→0.276、D3=0；static 基线 0.358→0.390（k5：WM 仅 1.3× 优于 static）。
**链级（raw）**：E_if ds 0.657→1.844；E_action ds 0.555→1.731；E_real ds 0.345→0.343（平）；plan_ds 2.81→3.98 vs expert 2.34→2.32（overshoot 0.469→1.660）。
**位姿敏感度自检**：OD 位姿置零 → d_od ≤0.0131、LD 置零 → d_ego ≤0.0115（k5）⇒ D1→D2≈0 是"模型位姿通道弱"而非 replay 失效。
**caveats**：D1/D2 动作级（π(ẑ_k)）需 t0 上下文键（本轮录制未含）→ unavailable（链级 E_if/E_action 仅 D0/D3 端点）。

## 2. P2：Action Fidelity（`runs/round32/p2_action_fd.*`；N=200；semantic_v1_independent）

- **口径说明**：v8 无语义键 ⇒ K1e2 semantic 离线不可跑（fail-fast 复现）；主口径 = `semantic_v1_independent`（K1 独立集）+ K1e2+MoE-off；v8 走 M0 baseline 备用。

**FD Jacobian（mean ‖·‖，k=1/3/6）**：

| combo | z_ego | z_od_pool | a_next |
|---|---|---|---|
| ds ±0.10 | 0.0175 / 0.0733 / 0.3095 | 0.0027 / 0.0129 / 0.0290 | 0.0034 / 0.0089 / 0.0405 |
| dθ ±0.01 | 0.8705 / 3.685 / 8.262 | 0.0929 / 0.4051 / 0.816 | 0.1192 / 0.2135 / 0.7912 |
| κ ±0.01 | 0.8722 / 3.668 / 8.164 | 0.0902 / 0.4041 / 0.8086 | 0.1195 / 0.2090 / 0.7881 |

⇒ **ds 通道弱 26–50×**；解析 autograd 全 None（detach 证据）；pose 端点非零（通路活着）；clip 率 0。

**Action Swap（E_all；matched/same/random/opposite/zero）**：matched **top-1=0.102、top-3=0.638**；rank 分布 {1:102, 2:282, 3:254, 4:205, 5:157}；margin mean **−0.002**；变体 spread ~4e-3 ≪ 基准误差 ~0.08（近并列支配）；逐通道 E_ld 最差（top1=0.007）。
⇒ **WM 预测对动作近不可区分**（"动作条件化失效"的最强证据）。

## 3. P3：Ego State Sufficiency（`runs/round32/p3_ego_sufficiency.*`；Gate C）

- 主池 33,056 anchors/500 ep（ridge；state 视图）+ ld_slots 视图（lane 目标主口径）；独立池 6,559 复现同模式。

| 目标（视图） | real R²（k1/3/5） | WM 保留（mean·med） | frozen 对照 |
|---|---|---|---|
| progress（state） | .970/.938/.894 | **1.03·1.03** | 1.00·1.00 |
| speed（state） | .975/.975/.975 | **0.91·0.91** | 0.87·0.87 |
| heading（state） | .976/.894/.777 | **1.10·1.09** | 1.08·1.06 |
| lane_lat（ld_slots） | **.901/.906/.907** | **0.61·0.62** ✗ | 0.59·0.58 |
| lane_head（ld_slots） | **.936/.931/.928** | **0.77·0.74** ✗ | 0.68·0.65 |

- Gate C 对照：real 充分 ✓（ego 0.89–0.98；lane ld_slots 0.90/0.93）；**WM 保留：ego ✓ / lane ✗（<80%）**。
- 附带发现：mean 池化丢 z_ld 槽位结构（lane_lat 0.40→0.93 逐槽展平）；**WM 的 od/ld latent 泄漏 ego 运动量**（ld_slots 上 WM 解 progress/speed R²≈0.9–0.99，real 仅 0.08–0.14）。

## 4. P4：Gate D 接口（`runs/round32/p4_lipschitz.*`）

- **A_k**（R4 50-ep；A_l2_abs）：p50 **0.098→0.206**（k1→5；分子 ×3.1、分母 +44%）；P95 0.187→0.314；road_class：ramp_in 最高 0.187、tollgate/roundabout 0.11–0.13。
- **Lipschitz**（FD，N=64 代用状态）：real 锚 p50≈0.029–0.032 / WM 锚 0.019–0.027 ⇒ 中位比 **0.73–0.78**；P95 比 **≈5.7–6.1**（尾部放大）。
- caveats：A_k 用 MoE-on 录制（统一口径重录可并入后续）；FD 状态为代用（R4 缺 obs/上下文）。

## 5. Gate 对照与决策草案

| Gate | 结果 | 判定 |
|---|---|---|
| A 工具可信 | D0 逐位复现、自检=0、复刻路径=0、clip=0 | ✅ |
| B Action Fidelity | ds 通道弱 26–50×；Swap 近随机 | ❌（未过） |
| C State Sufficiency | real ✓；WM 保留 ego ✓ / lane ✗ | ◐ |
| D Policy Amplification | A_k 温和（×2.1）；Lipschitz 尾部 6× | ◐（次要） |
| E 部署归因 | D2→D3 97%（主） | ✅（已归因） |
| F 新结构许可 | 待裁决 | — |

**Decision 草案（`runs/round32/decision.md`）：`GO_ACTION_CONDITIONING`（主）**——依据：Gate B 未过（动作条件化失效）+ Gate E 主段=latent transition（其机制=转移头无法用动作解释 ego 演化）。次要项：lane 保留率不足（Level-2 的 ego 物理锚/车道锚候选）、接口尾部放大（观测）。

**未选支路及理由**：GO_ROLLOUT_PROTOCOL_FIX（D0→D1 2.9%，非主因）；GO_DYNAMIC_GRAPH_FIX（D1→D2 0.0%）；GO_STATE_REPRESENTATION_FIX（real latent 充分）；GO_POLICY_INTERFACE_ROBUSTNESS（A_k 温和，仅尾部）；GO_ALTERNATING_WM_POLICY_TRAINING（Level-3，未到）。

**下一步（待裁）**：① action-conditioning canary（零初始化动作残差，`alpha=0` 逐位兼容；GPT 已预批"若 fidelity 未过再增加"）；② Level-2 条件项（lane 保留/ego 锚/prior-residual/rank）按需启动；③ 统一口径 A_k 重录（MoE-off）可并入②。
