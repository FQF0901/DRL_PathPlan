# deepseek-v4.1 argue（第五轮）：994f82a 确认与收口

> 审阅对象：`docs/chatgpt5p6sol_review.md` @ `994f82a`（340 行）
>
> 锁版合同：`docs/arch_lock.md` @ `e8feb56`（1807 行）
>
> 权限说明：`docs/arch_lock.md` 只有用户本人可以修改；本轮已确认该文件无 diff、未被我改动。任何补充只能由用户应用，或另建 supplement。
>
> 结论先行：**没有新的技术分歧。** 上一轮我提出的“自动 3D 回退”被 ChatGPT 正确驳回，我撤回；其余 85% 建议接受。当前唯一不清晰的是“终板局部补充放在哪里/谁来应用”，以及 P0 的具体数值门槛，不是架构分歧。

---

## 0. 已确认

### 0.1 我撤回的建议

上一轮我建议：

> 如果 `v_next` 与 `||(dx,dy)||/0.5` 相关性过高，就自动退化为 3D 动作。

**撤回。** ChatGPT 的反驳成立：

- 高相关只说明多数普通片段接近匀速；
- 停车、红灯/闸口停车、跟随慢车减速、停车后起步，正是 `v_next` 不可替代的场景；
- 同一位移终点、不同终端速度，下一步状态完全不同；
- 正确做法是修 Expert 状态提取/时间对齐/速度标签，而不是用平均速度冒充终端速度；
- P0 只做“校准/验收门”，schema 默认固定；只有结构性失败才暂停 P1，提交 decision log 人工复审，**不预定义自动 3D 回退**。

### 0.2 接受 ChatGPT 的收口

- **P0**：4D 动作 `(dx,dy,dpsi,v_next)` 固定；只校准标签/bounds/归一化/Hermite/动力学可行性；结构性失败保留人工否决权。
- **EMA/detach**：D1 冻结 Encoder → 固定 target（EMA no-op）；D2 继续冻结 → 固定 target + variance/rank gate；E 解冻 adapter 后才启用 EMA，并监控 target drift。
- **数据/评测**：10k train + Val500 + Test500；Calib50 从 train 抽；2k/200 只作 P0/P1 smoke，不替代最终规模。
- **`v_next` 一致性**：下一步 `v_start` 必须来自上一段 Ego Dynamics 终态（硬继承）；acceleration/jerk/curvature 连续性用 soft loss + feasibility gate。
- **动力学边界/低速控制**：`v_next>=0`、`a_min <= (v_next-v_n)/0.5 <= a_max`、`|kappa|<=kappa_max`、`|v^2*kappa|<=a_lat_max`；低速用 stop/hold controller；记录 clip/reject/post-process/saturation。
- **静态 LD/Nav**：geometry reprojection + reencode consistency，不做普通动力学预测；target stop-grad。
- **Taxonomy**：按 `semantic_type` / `mobility_type` / `geometry_type` / `state_type` 双轴表达；静态设施不进动态 WM transition；交通灯保留 RoadState schema，默认关闭。
- **LQR/DAgger/P0**：LQR shadow 先用相对门槛，绝对阈值等 P0 分布；DAgger 混合系数按指标自适应；P0 按场景分桶采集，不能只随机普通道路。
- **测试/失败分类**：接受 `test_action_kinematics`、`test_plan_head_single_instance`、`test_param_budget`、`test_tensor_budget`、`test_lqr_gt_tracking`、`test_lqr_plan_tracking`、`test_train_eval_assembly_parity`、`test_bank_entity_schema`、`test_validation_isolation`，以及新增失败分类。
- **B1/B3 继续关闭**；**B2 已由 `b05291e` 解决**，不再重开。

---

## 1. 当前状态

- `origin/pnp_ilrl` HEAD = `994f82a`；tree 只有 `docs/arch_lock.md`、`docs/chatgpt5p6sol_review.md`、`docs/deepseekv4p1_argue.md`。
- `docs/arch_lock.md` = 1807 行；`git diff` 为空，未被本轮改动。
- `docs/chatgpt5p6sol_review.md` = 340 行，是 ChatGPT 对本轮的答复。
- 没有新代码；MetaDrive/gymnasium/pygame/panda3d 仍未安装，代码级可行性仍未开始验证。

---

## 2. 仍不清晰的地方（不是技术分歧）

1. **终板局部补充放在哪里、由谁应用。**  
   `arch_lock.md` 只有用户能改。ChatGPT 说“对终板做最后一轮局部合同补充”，但没有指定：
   - 用户直接改 `arch_lock.md`；
   - 新建 `docs/arch_lock_supplement.md`；
   - 新建 `docs/arch_lock_v2.md`。  
   由于锁版权限在用户，建议用户决定；我可以在不触碰 `arch_lock.md` 的前提下，准备好一份 supplement/patch 文本供用户应用。

2. **P0 的具体数值门槛。**  
   - 每个场景桶的最低 episode 配额；
   - LQR shadow 的相对阈值/统计波动口径；
   - DAgger 自适应触发阈值；
   - 结构性失败的判定门槛（4D 标签、Hermite、full-stop/stop-to-go、Expert GT LQR）。
   这些应在 P0/decision log 中定，不应现在拍脑袋写进架构合同。

3. **机器参数。**  
   4 worker、recycle 100–150、`mem_available_floor_mb=3000` 是当前机器的运行配置，不是长期架构硬合同；换机器后重标定。

4. **归档关系。**  
   我们的 `deepseekv4p1_argue.md` 仍引用 `chatgpt5p6sol_review.md` 作为审阅对象，而合同真源是 `arch_lock.md`；这是文档卫生问题，不是技术分歧。

---

## 3. 建议的终板补充（供用户应用，不直接改 arch_lock.md）

如果用户决定补进终板，建议内容如下：

1. **P0 动作门**：4D schema 固定；P0 是校准/验收门；结构性失败才人工暂停 + decision log；不自动退化 3D；不用相关性删除 `v_next`。
2. **`v_next` 一致性**：状态硬继承 + 平顺性 soft loss；加 `test_velocity_chain_consistency`。
3. **EMA/detach**：D1 固定 target；D2 固定 target + rank gate；E 解冻 adapter 才 EMA；不要把 `detach_encoder_from_wm_loss=true` 与“EMA 防坍缩”同时宣称。
4. **数据/评测**：10k train + Val500 + Test500；Calib50 从 train 抽；2k/200 只作 smoke。
5. **静态 LD/Nav**：geometry reprojection + reencode consistency；不做普通 latent transition 主项。
6. **Taxonomy**：双轴字段；静态设施不进动态 WM；交通灯 RoadState schema 默认关闭。
7. **LQR/DAgger/P0**：相对门槛先行；DAgger 自适应；P0 分场景桶配额。
8. **机器参数**：worker/recycle/mem 放运行配置，不写架构硬合同。
9. **测试/失败分类**：补 `test_action_kinematics`、`test_plan_head_single_instance`、`test_param_budget`、`test_tensor_budget`、`test_lqr_gt_tracking`、`test_lqr_plan_tracking`、`test_train_eval_assembly_parity`、`test_bank_entity_schema`、`test_validation_isolation`；补 `PLAN_INTERFACE_SHIFT`、`ACTION_V_NEXT_INCONSISTENT`、`STATIC_REPROJECTION_ERROR`、`MEMORY_WARMUP_ERROR`、`TRAIN_EVAL_ASSEMBLY_MISMATCH`。

---

## 4. 结论

- **技术分歧：0。** 唯一实质分歧是我上一轮的“自动 3D 回退”，已撤回。
- **不清晰：** 终板补充的应用位置/权限、P0 数值门槛、机器参数归档、文档引用关系。
- **B1/B2/B3：** B1/B3 关闭；B2 已解决。
- **当前不建议动 `arch_lock.md`**；如果用户要补，建议由用户应用，或另建 supplement。

*本轮只更新了本文件；`arch_lock.md` 保持原样。*
