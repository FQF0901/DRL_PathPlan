1. 自检问题：
1.1. 生产路径仍是 legacy
- StructuredPlanner / 旧 EgoConditionedLatentWM / stack_history 仍被 pipeline/* 使用；
- 新 DriveModel 还没有进入 train/eval/PPO/checkpoint。
1.2. 目标编码器和 loss 未接
- TargetEncoder 只被测试使用；
- pipeline/losses.py 不存在；
- 没有 OD content / OD geometry / LD residual loss；
- 没有新 checkpoint schema。
1.3. imagine 仍有语义缺口
- imagine 的 nav/ego 默认保持 t0；
- 若需要真正的 policy_n，还需要每步 Nav/Ego 输入或 adapter；
- world_model.rollout 仍是不带 ego 投影的第二套 rollout，和 DriveModel.imagine 不统一。
1.4. schema / env 边界
- BankWindow 的 OD/LD split 与合同单一 schema 是工程化偏差，需要 owner sign-off；
- env.obs ROI 仍是 20/80/20/20，不是 80/20/25/25。


2. 部分修复
- build_net 维度校验不完整 — net/build_net.py:60-68
- 只校验四个子模块 hidden 彼此相等，没有校验顶层 model.hidden 与它们相等。
- 复现：model.hidden=256 + 四个子模块 hidden=128 可以 build，model.hidden 仍是 256；这与 docs/implementation_status.md 声称的 hidden 一致校验不符。
- 另外 §13.4 的 action bounds/limits 仍没有从 config/model.yaml 读取；当前配置只有 action_dim，没有 action bounds。
- LD type 继承仍是 fallback — net/encoder.py:95-98
- 已从 lane_role 改成 road_type_norm * 15，比之前正确；但 env.obs 仍没有 ld_taxonomy，所以不是真正的 “LD type 从 t0 继承”，只是 road type 代理。docs/implementation_status.md 也标为“部分”。
- valid 传播仍不完整
- WM 现在接受/返回 od_valid/ld_valid 并做 mask（net/world_model.py:431-443），这是进步。
- 但 imagine 始终使用当前 window 的 valid（net/model.py:539-540），没有预测/更新 imagined slot 的 validity；advance 也只是把当前 valid 往后 shift。
- 对固定 top-16、不建模遮挡的版本可能可接受，但 §6.5 的 valid=... 还不是“预测数值有效”。
- ego/nav 逐步传播仍不完整
- imagine 支持外部 ego_pose 序列，geometry 会投影（net/model.py:557-568）。
- 但 nav / ego feature 在 6 步中保持不变；imagine_from_real_bank 默认不传 ego_pose，所以 geometry 投影是 identity。
- docs/implementation_status.md 也承认“ego/nav 的逐步传播仍依赖外部 ego_pose，没有内置 ego dynamics”。
- multi-step 训练接口有了，但实际不可用
- FactorizedEgoConditionedLatentWM.rollout 已去掉 @torch.no_grad()，支持 detach — net/world_model.py:463-487。
- 但 OD geometry 的 in-place 赋值会让 backward 失败（见第 4 节），所以 L_od_geo 和 imagine(detach=False) 仍不能训练。
- imagine 与 world_model.rollout 仍是两套 loop
- DriveModel.imagine 自己实现了完整循环（net/model.py:532-608），没有调用 FactorizedEgoConditionedLatentWM.rollout（net/world_model.py:463-487）。
- 功能上 now works，但 §13.5 的“模型级 rollout 在 world_model，model 只调用”仍未满足。
- project_ld 仍耦合私有实现
- project_ld 仍调用 BankWindowBuilder._transform_geometry（net/world_model.py:346），而 BankWindowBuilder.project_geometry 已经是公共函数；两套递归逻辑重复。
- 测试覆盖仍低于 §9
- 新增了 8 个测试，但仍缺 §9.1 FIFO/reset/reject imagined/padding/slot 复用、§9.3 B>1/k<6/permutation、§9.4 LD residual 越界 valid=0/future map 置空/rollout NaN、§9.5 无 stack_history/batch/no risk-value 等。
- test_imagine_advances_window_and_geometry 实际没有断言 geometry/depth；test_imagine_from_real_bank_appends 没有检查 ImagineBank 内容，只检查 RealBank 没被写回。
3. 未修复（合同层面）
- live path 仍是 legacy，这是最大的未修复项
- pipeline/train_bc.py:79,1079、eval.py:52、rollout.py:22、train_ppo.py:75、checkpointing.py:18,120-123 仍用 StructuredPlanner；
- pipeline/train_wm.py:23、probe_wm.py:19 仍用 legacy EgoConditionedLatentWM；
- env.obs.stack_history 仍在用；
- 新 DriveModel/build_net 仍只被 test/test_new_net.py 使用。
- 因此 §7/§9.5 的“无 value/risk head、无 stack_history 主路径”事实验收仍不成立。docs/implementation_status.md 也明确写了“新路径仍不是 live path”。
- Legacy 代码仍留在 net/
- net/model.py:103-411 仍有 StructuredPlanner / TemporalPool / count_parameters；
- net/world_model.py:26-172 仍有 EgoConditionedLatentWM / masked_mse / rollout_latents；
- net/param_probe.py 仍不在 §13.1 目标结构。
- 虽然 TemporalPool 的 high bug 已修，但“legacy 与 new 混在同一文件”的结构问题仍在。
- BankWindow / FrameRecord schema 仍非字面
- BankWindow 仍没有合同 §4 列出的 slot: Tensor [N]（net/mem_bank.py:106-127）；
- FrameRecord 仍是 od/ld 两个 tuple，而不是统一 entities（net/mem_bank.py:74-88）；
- EntityRecord.type 仍叫 type_id；
- ego_velocity 仍是 optional，不构成硬违反，但 schema 仍不是字面。
- Target encoder 训练接线未做
- TargetEncoder 仍只出现在 test/test_new_net.py；
- 没有 z* = target_encoder(o_{t+1}) 的训练 target 构造，没有 EMA 调度，没有 target 不进 Bank/policy 的接线验证。
- pipeline/losses.py 仍不存在
- 新 WM 的 L_od_content / L_od_geo / L_ld_res 没有落地文件。
- DriveModel checkpoint 未接
- pipeline/checkpointing.py:120-123 仍只构造 StructuredPlanner；train_bc.py:890 仍写 "architecture": "StructuredPlanner"。
- pipeline/rollout.py 仍不是 model.imagine 的薄 wrapper
- pipeline/rollout.py:78-95 仍是 legacy StructuredPlanner 的重复预测。
- action bounds 配置缺失
- §13.4 要求 config/model.yaml 负责 action bounds/limits；当前 config/model.yaml 只有 action_dim，没有 bounds/limits；build_net 也不读 action 段。
- env 侧相邻偏差仍在（不在 net/，但影响 Bank 内容）
- OD ROI 仍是 x[-20,80], y[-20,20]（env/obs.py:774），合同是 80/20/25/25；
- LD gate 仍是 s0[-15,80], |lat|≤20（env/obs.py:166-168），合同是 80/20/25/25；
- env.obs 没有 ld_taxonomy；
- env.obs 的 pose 是 float64，虽然 BankWindow 现在会转 dtype，但 env 仍不是新 Bank 接口。
4. 修复引入的新问题
High：WM OD geometry 的 in-place 赋值导致 backward 失败
net/world_model.py:447-452：
od_next_geometry = od_geometry.clone()
od_next_geometry[..., :3] = od_geometry[..., :3] + self.od_geometry(od_next) * od_mask
od_next_geometry[..., 2] = torch.atan2(
    torch.sin(od_next_geometry[..., 2]),
    torch.cos(od_next_geometry[..., 2]),
)
od_next_geometry[..., 2] = ... 的 RHS 使用了同一个 view，赋值后 autograd 版本号冲突。复现：
out = wm(..., od_geometry=base_od, ld_geometry=base_ld)
out["od_geometry"].sum().backward()
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation ... CosBackward0
- out["od"].sum().backward() 可以过；
- out["ld_geometry"].sum().backward() 可以过；
- out["od_geometry"].sum().backward() 失败。
- 因此 §6.4 的 L_od_geo 目前不能训练；imagine(detach=False) 也会在同一处失败。
建议：不要对 od_next_geometry 做 in-place；先算 yaw = atan2(...)，再用 torch.cat([xy, yaw.unsqueeze(-1), vel], dim=-1) 构造新 tensor。
Medium：build_net 顶层 hidden 不一致仍可 build
net/build_net.py:60-68 只检查四个子模块 hidden 相等，没有检查顶层 model.hidden。复现：
build_net({"model": {"hidden": 256, "encoder": {"hidden": 128},
                     "temporal_spatial": {"hidden": 128},
                     "planner": {"hidden": 128},
                     "world_model": {"hidden": 128}}})
# 成功，model.hidden=256, encoder.hidden=128
建议加 if hidden != enc_hidden: raise。
Medium：负 stable_id 未被拒绝
net/mem_bank.py:489-524 只检查重复 stable_id/slot，没有检查 stable_id < 0。复现 valid record with stable_id=-1 会被接受，od_valid 为 True；这会让 temporal attention 把所有 -1 当成同一身份，违反 §2.4 的 stable-ID 语义。docs/implementation_status.md 声称“负 stable_id fail-fast”，代码实际没有。
Low：DriveModel.imagine.__doc__ 是 None
net/model.py:519-531 把 docstring 放在 if horizon is None 之后，Python 不会把它当作 imagine 的 docstring。DriveModel.imagine.__doc__ 实测为 None。建议移到函数第一行。
Low/API：WM 现在强制要求 ld_geometry
net/world_model.py:439-440：
if ld_geometry is None:
    raise ValueError("LD residual requires base ld_geometry")
如果只做 OD content one-step 训练、不准备 LD base，现在会直接报错。合同 §6.1 的 OD content WM 和 §6.3 的 LD branch 可以解耦；如果训练入口需要 OD-only，建议把 LD branch 做成可选，或明确要求所有调用都传 ld_geometry。