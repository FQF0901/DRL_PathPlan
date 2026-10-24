1. 接受
高优先级
- WM OD geometry 的 in-place 赋值导致 backward 失败：接受，已修；现在用 torch.cat 构造新 tensor，backward 通过。
- build_net 顶层 model.hidden 未校验：接受，已修；现在把顶层 hidden 也加入一致性检查。
- LD ROI 的 rear-20 是死条件：接受，而且这条很重要。
- 旧代码先 s0 = clip(s0, 0, lane.length)，再判断 s0 < -20；
- 因此 rear 20 永远不会生效；
- 我已改为先对 raw s0 判断 ROI，再 clip 给 lane.position()。
- 我上一轮说“env ROI 已修”对 LD 是不准确的，这里更正。
- action bounds 仍未从 config 接执行层：接受；已在 config/model.yaml 增加 action 段，但还没驱动 env/action 执行。
- imagine 的 nav/ego 仍是 t0 常量：接受；我已增加可选 nav_seq / ego_seq，但默认仍用当前输入，没有内置 ego dynamics。
- world_model.rollout 与 imagine 是两套 rollout：接受；前者仍没有 ego 投影，后者是正确路径。
- live path 仍是 legacy：接受，仍是最大 blocker。
- TargetEncoder / pipeline/losses.py / 新 checkpoint 未接：接受。
- pipeline/rollout.py 不是 model.imagine 的薄 wrapper：接受。
- 测试覆盖不足：接受。
- BankWindow schema 与合同字面不一致：接受为工程偏差，需要 owner sign-off。
已确认修复
- negative stable_id 的反驳成立：env.obs._stable_hash 是 signed int64，负值是合法实体 ID；padding 由 valid=False 表达。reviewer 已撤回该条，我同意。
- TemporalPool 的 D1 reshape bug 已修。
- imagine 的 per-step TSB、geometry 投影、ImagineBank、policy feedback、detach 可用。
- build_net hidden 校验、param_probe budget/path、action_space 迁移可用。
2. 需要反驳/解释
反驳
1. “env ROI 仍是 20/80/20/20”这句话已部分过时
- OD 已是前 80 / 后 20 / 左右 25；
- 只有 LD rear gate 之前是死条件，现在已修；
- 所以不能再说整个 env ROI 仍是旧值。
2. “negative stable_id 应拒绝”不是正确要求
- reviewer 后来已撤回；
- 负 ID 是合法 signed hash，不能按符号拒绝；
- 唯一要求是 padding 的 -1 只能出现在 valid=False。
解释
- BankWindow 没有独立 slot: Tensor[N]
- 固定 16 slot 下，slot 是张量维度；
- OD/LD 拆成两套张量是工程实现，不是功能 bug；
- 但按 arch_lock 字面标准，仍需 owner sign-off。
- horizon=6 的语义
- 当前返回 6 个 policy，只有 5 个 imagined transition；
- 如果合同把“6 步 rollout”定义为 6 个 imagined frame，就需要再明确；
- 这是规范歧义，不是单纯代码 bug。
- world_model.rollout
- 已去掉 @torch.no_grad，但还没有 ego-motion projection；
- 不应作为 D2/training 的正式 rollout；
- 正式路径应只保留 DriveModel.imagine。
- WM 强制要求 ld_geometry
- 已改为可选：没有 LD base 时 LD branch 返回 None；
- 但训练入口应明确是 OD-only 还是 OD+LD。
- project_ld 私有耦合
- 已改为调用公共 BankWindowBuilder.project_geometry；
- 这条已修。
3. 仍未完成
这些不是“解释”能解决的，仍是 blocker：
1. Full cutover 未完成
- StructuredPlanner、旧 EgoConditionedLatentWM、stack_history 仍被 pipeline 使用；
- DriveModel 还没有成为 train_bc/eval/train_ppo/checkpointing/train_wm 的 live path。
2. TargetEncoder / loss / checkpoint
- 还没接 z* = target_encoder(o_{t+1})；
- 没有 pipeline/losses.py；
- 没有新 checkpoint schema；
- 旧 train_wm 仍用旧 WM。
3. world_model.rollout 仍不是正式路径
- 没有 source→target ego 投影；
- 与 DriveModel.imagine 两套语义。
4. 测试覆盖
- 当前定向 50 passed，full suite 104 passed / 1 skipped / 1 failed / 7 errors；
- 失败仍是 MetaDrive/Panda3D headless；
- §9 的 FIFO/reset/reject/permutation/rollout NaN/config 测试仍不足。
5. schema
- BankWindow 的 OD/LD split、FrameRecord 的 od/ld tuple、type_id 命名需要 owner 决策。
4. 当前判定
- 最新 net 修复：核心问题已接受/修复。
- 整体 net contract：仍 NOT READY，主要因为 live path、target/loss/checkpoint 和正式 rollout 未完成。
- docs/implementation_status.md 已按此状态更新。
这轮我没有把 pipeline 强行切到新路径；Full cutover 需要同时改 env.obs、collectors、train/eval/PPO、loss 和 checkpoint，不能只改 import，否则会把未验证的 Bank 直接接进训练。