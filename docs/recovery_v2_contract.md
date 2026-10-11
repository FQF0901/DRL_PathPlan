# recovery_v2 采集契约（同状态）

> P1-v2 Step 2 规范。原则：**每一行的 observation / baseline plan / expert label 必须来自同一时刻、同一状态**。
> 禁止 Pass-1 observation 配 Pass-2 label（v1 教训：9,377/11,694 行为反事实错位；见 `docs/p1_2_diagnosis.md`）。

## 1. 行字段（同状态）

- `obs_t`：学生可见的全部正式观测通道（含 mask）+ `history_t`（6 帧历史，含 valid mask）；
- `student_plan_t` / `student_action_t`：**基线**（学生模型在该状态的计划/首步动作；用于 residual target 与 gate 审计）；
- `expert_action_t`：当前状态单步专家动作（同一状态重规划）；
- `expert_recovery_plan_t`：当前状态 6 点恢复链（ego 系 ds/dθ）；
- `actual_state_t`：ego 位姿/速度等（世界系）；
- `margin / T_plan / T_cross`：该状态锚点量（口径同 v1；T_cross_source 显式标注）；
- `intervention label`：**同一状态下 baseline 将失败 ∧ expert 能在规定 horizon 内恢复**（构造规则随行留档）；
- `control 行（non-intervention）`：同源学生轨迹上 **plan-valid-through-horizon** 的步（当前 + 后 6 决策步 footprint/exec 有效、无邻近锚点事件），须覆盖**晚段/高 rc**（step ≥ onset+30、rc 分层）；与 intervention 行同契约同源，用于 **gate 激活率约束与探针净增量复核**；
- `recovery outcome`：专家接管实测（`expert_recovered` / `recovery_success_horizon` / `min_margin_after_takeover` / `route_progress_after_takeover`）；
- `mode 粗标签`：仅用于分层/误差分析/未来 4B 研究输入；**先验证稳定性/类内方差/邻步切换率**，不得作训练目标或 4B 启动依据。

## 2. 采集机制（v2）

- **逐点 counterfactual**：对学生真实轨迹的每个决策步 t（或锚定步集合），从 t 的真实状态查询 expert（当前状态重规划）；
- **k≥1 帧必须保存 Pass-2 对应 obs**（v2 修复点；v1 的错位即源于缺此）；
- `baseline_plan` 由学生模型在该状态前向取得（采集时或离线复算，须与 obs 同状态）；
- 每行独立可复算：obs→plan→label 链完整留档。

## 3. 验收（进入训练前）

- 同状态匹配率 **100%**（重放抽检 + 构造核验）；
- 随机重放 **≥1,000 行**（确定性重放一致）；
- episode 级 train/val split、**无泄漏**；
- expert recovery 成功率 **≥80%**；
- action-chain 与 recovery trajectory 一致性通过（构造守卫）；
- **control 行覆盖**：晚段（step ≥ onset+30）高 rc 匹配态，数量与 intervention 行同量级（供 gate 激活率约束与探针复核）；
- pilot 规模 **2k–3k 行**（通过后再扩容；不先追 10k）。

## 4. 反模式（禁止）

- Pass-1 obs × Pass-2 label；k≥1 无同源 obs；
- batch 级替换采样造成 >10× 曝光且不留档 —— 必须记录 `raw sample ratio / sampler exposure / loss weight / effective importance ratio`；
- 未验证的 mode/label 作为训练目标；
- control 行缺晚段/高 rc 覆盖（Step 1 的混淆教训：早期负样本会让"净增量"无法确证）。

## 5. 与下游的接口（gated residual）

- 结构：`final_plan = base_plan + gate(obs, base_plan) × residual(obs, base_plan)`；
- 训练：正常样本监督 `gate=0`（保持 base plan）；recovery 样本监督 `residual = expert_plan − base_plan`（行内 baseline_plan）；
  residual 仅在 intervention 有效时产生梯度；gate 与 residual 梯度分开记录；
- 占比：recovery 自然占比 **2%–5%**；base plan 作 anchor，防正常驾驶能力漂移。
