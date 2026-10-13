# Round-31 审计汇总（Q2 + WM fidelity + Stage C contract + M2；全只读、零训练）

> 依据：`docs/deepseekv4p1_argue.md` §5.34（GPT `f4faec8` 裁决：批准 Q2 审计 / WM action fidelity / Stage C contract，暂停一切训练）。
> 纪律：零训练、零生产代码改动。产物：`runs/round31/probes/*`、`runs/round31/moe_off/*`、`docs/round31_stagec_contract_audit.md`；脚本 `.slim/deepwork/p1_v2/round31/*`。

## 0. 结论速览（8 条）

1. **Q2-a（协议内退化）**：存在但中等——Stage B 协议（val 真态、policy 自回归、冻结 WM、无执行器）E_state 0.022→0.065（k1→5，canonical）、E_if L2 0.145→0.357；rollout 尾 ADE 0.544 vs oracle 0.202（**2.7×**）。
2. **Q2-b（监督冲突）**：**不支持**——L_action vs L_traj 共享参数梯度余弦全正（0 conflict 标记；批均 +0.23~+0.29）。
3. **Q2-c（oracle 反事实）**：预测态链代价亚米级（ΔADE=0.341 / ΔFDE=0.796，canonical）——"若 WM 完美"的收益有界。
4. **M2（部署链分解）**：部署 E_z **不由 pose/执行错配驱动**——实测位姿链重放 vs 预测链 Δ ≤ 8.7% rel（k5），且位姿敏感度本身弱；**重规划替换真实**（执行动作≈当前 μ、远离 plan 尾段 |Δds|→2.0 m、松一致率 0.24→0.06）。
5. **WM 条件化（whole-scene mirror）**：scene/action 对 latent 的响应比 **24× / 5612× / 1392×**（ego/od/ld）——动作/位姿条件化弱为一级异常；mirror 一致率 restricted 0.842（<0.95，首轮诊断呈现）。
6. **本轮最大新发现**：**K1 的 base 网络与 M0 逐位相同**（K1-only=20 键：`semantic_wiring.*` + β；base 参数 0 改动）⇒ "K1 训练破坏 latent 空间"排除；而 expert 协议 E_state=0.026→0.08 vs 部署 regime E_z=0.203→0.316（**~8–10×**）⇒ **WM 转移误差强烈依赖状态/未来分布（policy-driven）**，不能用模型差异或位姿执行解释。
7. **契约（Lane D）**：S3（repeat+LQR）≠ S2（plan+LQR）结构级；两条 eval 栈默认口径相反（eval_runner=exact/仅 plan[0] vs eval_a3=lqr）；exact 仅消费 plan[0]；kappa-compose 仅存在于诊断侧；**MoE 口径差已量化（±2–4pp，矩阵结论稳健）**。
8. **Q2 综合判决（我方）**：Stage B 监督 = "中等间接性代价 + 无冲突证据 + 非灾难" ⇒ 不构成"一级非法监督"；GPT 的"一级风险"降级为"**已量化的中等风险**"；部署 regime 的 8–10× 放大才是主问题（候选=状态分布/目标分布/交通耦合/重规划语义，需一个归因探针分离）。

## 1. Lane A：Q2-a/c（`runs/round31/probes/q2_rollout_consistency.{json,md}`）

**协议**：val 行 t0 真态 → policy 自回归 rollout（冻结 WM）→ 目标=真实未来行编码；N=300（canonical 240）；pri512 stage_b e020 主口径、M0 副口径；自检全过（`plan[0]==mu`、积分 vs traj6 max 2.8e-6）。

**主表（canonical，L2 为 (ds,dθ) 向量范数）**：

| k | E_state ego 1−cos | E_if L2 | E_deploy L2 | E_real L2 | E_deploy/E_real | E_if/E_real |
|---|---|---|---|---|---|---|
| 1 | 0.0222 | 0.145 | 0.225 | 0.135 | 1.67 | 1.08 |
| 3 | 0.0407 | 0.236 | 0.493 | 0.325 | 1.52 | 0.73 |
| 5 | 0.0649 | 0.357 | 0.789 | 0.498 | 1.58 | 0.72 |

**轨迹（vs `traj6`）**：ADE rollout/oracle = 0.544/0.202、FDE = 1.184/0.388（canonical）；配对 ΔADE=0.341、ΔFDE=0.796。
**M0 副口径**：E_state all 0.026→~0.08（与 pri512 同量级）；ADE/FDE 同量级 ⇒ **模型差异被排除**。

## 2. Lane B：Q2-b 梯度冲突（`runs/round31/probes/q2_grad_conflict.{json,md}`）

- L_action（首步，weight 1.0，复刻 `trainer.py:4290-4296`）vs L_traj（`bc_trajectory_loss`，weight 0.1）：**全局 cos 全正**（repeat0 +0.052；B=256 +0.445；8 批均值 +0.234）。
- 逐组（repeat0）：policy −0.056（批均 +0.318）、encoders +0.363、mem +0.705、embed_nav +0.751、fusion +0.204、router +0.684；**0 个 cos<−0.3**。
- frozen 变体（真实训练冻结：仅 experts/router/residual_scale）：experts +0.270、router +0.684（均正）。
- st_gnn/value/ego_next 对两损失均不可达（"无监督"）；72/166 参数无监督。
- **判定**：监督方向不冲突；"教错误状态上的修正"最强形式不成立（方向性证据；不裁决目标语义合法性）。

## 3. Lane C：whole-scene mirror（`runs/round31/probes/wm_mirror.{json,md}`）

- 构造 sanity：镜像对合/先验镜像/位姿镜像 max|err|=0；R30-3 action-only 分支**逐位复现**。
- **scene vs action 响应比**：1−cos（pooled）scene+action 2.46e-2/1.478e-1/4.24e-2 vs action-only 1.018e-3/2.6e-5/3.0e-5 ⇒ **24×/5612×/1392×**（ego/od/ld）。
- mirror 一致率：a_next dθ 反号 restricted **0.842**（all 0.486，近零噪声主导）；near_mirror 0.33%；od.dy/ld.dy 反号 0.814/0.613（ld.curvature 0.981）。
- **判定**：WM 由场景主导、动作/位姿条件化弱（正确构造下坐实）；mirror ≥95% 门未过（非等变模型 + t0 冻结简化口径 ⇒ 首轮诊断，不设门）。

## 4. Lane D：Stage C 执行契约（`docs/round31_stagec_contract_audit.md`）

- 四链路映射矩阵：collect↔update=**是**；collect↔eval=部分；**collect↔deploy=否（S3≠S2）**；update↔deploy/eval=部分；eval↔deploy=部分（默认口径）。
- 不一致清单 I1–I7（S3≠S2 / exact 仅 plan[0] + 两栈默认相反 / kappa-compose 仅 eval / MoE on-off / `stages.C.tracker=mpc_lqr` 死键 / update 零消费 plan / eval 丢弃 reward）。
- **MoE 敏感性复跑**（`runs/round31/moe_off/`）：训练口径（MoE-off）K1e2 lqr_plan **0.560**（vs 0.520；逐 episode 48/50 一致）、exact 0.560、repeat 0.060 ⇒ 结论稳健，±2–4pp。

## 5. Lane E：M2 三分分解（`runs/round31/probes/m2_decompose.{json,md}`）

- **action-faithful replay**（实测位姿链替预测链）：Δ(实测−预测) = −0.0007/−0.0077/−0.0336（k1/3/5），相对降幅 **−0.3%/−2.3%/−8.7%** ⇒ pose/执行错配**不是** E_z 主因。
- 链偏离：预测链 vs 实测 |Δxy| 0.725→8.17 m（k1→5）——"计划想象轨迹"与实况差距大，但**换入实测位姿仍不改善**（结合 §3：位姿敏感度弱）。
- **重规划替换量化**：â_{i+k} vs plan(i)[k]：|Δds| 0.350→2.025、松一致率 0.237→0.059；vs μ(i+k−1)：|Δds|≈0.35 恒定 ⇒ 执行动作≈"当前策略动作"，**plan 尾段在部署中被逐步丢弃**。
- 复现核验：rec_pred_cos 逐位复现 R4（0.2033/0.2746/0.3219/0.3164/0.3158）。

## 6. 综合与下一步（供裁）

**Q2 综合判决（我方）**：Stage B 监督的"合法性"问题**量级为中等、无冲突证据**；部署 regime 的 **8–10× 放大**（同一 base 网络：expert 协议 0.026→0.08 vs 部署 0.203→0.316）与 **WM 动作/位姿条件化弱（24×–5612×）** 是两个真正的一级异常。

**候选队列（零训练优先）**：
1. **部署状态归因探针**（新）：小规模录制（补录 raw obs + 未来帧）→ 在**部署状态**上跑 Lane A 协议 → 分离"起始状态分布 vs 目标分布/协议"两假设（~1–2h）。
2. **action/pose conditioning**（A5 候选）：证据已足（§3/§5）；建议归因（1）先行。
3. **契约修复批次**：MoE 口径统一（训练/评测）、eval 栈默认口径、S3→S2 口径决策。
4. **A4（policy-conditioned WM 适配 canary）**：前提由 §0-6 的量级支持；建议 1 完成后启动。
5. **2-step BPTT canary**：条件已备（R30 账本 + 本轮梯度证据），优先级后置。
