总体判断：RL 与 WM 先解耦、最后再联合是对的，但当前方案有 4 个高风险点。最严重的不是 WM，而是 Group-relative PPO 的信用分配和共享 TSB 的联合训练漂移。 arch_lock.md 目前仍属于设计合同，Group-relative PPO、3/6-step WM 和新 Joint Trainer 尚未实现。

可以保留
RL 只在真实环境训练，WM 不进入 reward、advantage 或 RL forward，能避免早期 model bias 污染策略。
Real/Imagine Bank 只存 OD/LD，Nav 只进 Policy，WM 用 action 条件化，职责清楚。
Policy 与 WM 共享 Encoder/TSB 表征，Policy hidden 不进入 WM，能够减少两套世界表征，但不让导航目标污染环境转移。
WM 先 one-step，再 3/6-step，最后 Joint，顺序合理。
Joint 将 L_state 和 L_policy 分开 backward，并显式控制 action/target 的 stop-gradient，方向正确。
必须修改
1. 当前“GRPO”信用分配过粗

当前设计把整条 episode 的 group advantage 广播到该 episode 所有 step：

A_i = episode return_i - group mean
同一个 A_i 乘该 episode 每个 action 的 ratio


这会把碰撞前的错误动作和早期正确动作同奖同罚；最长 60 秒、约 120 个 policy decisions 时方差会很大。

建议：

第一版可以保留 episode-level GRPO，但必须同时实现 per-step active-set group baseline；
按时间 t 对仍存活的同 seed trajectories 做 group normalization；
terminal success/collision 回传可保留，progress/comfort 用 step-level advantage；
若同组有效样本少于 3 条或组内 return std 太小，该组跳过更新；
不要把 episode-level 广播写成长期最终方案。
2. 唯一 Gate 只看 success 不够

当前唯一硬门槛是 success ≥ 72.8%，collision/off-road 只人工观察；当前基线本身 collision 已有 17.8%。这允许 RL 通过更激进的策略提升成功率，同时显著增加事故。

至少增加非劣化约束：

success ≥ 72.8%
collision ≤ pre-RL baseline + 统计容差
off-road ≤ pre-RL baseline + 统计容差


不必现在拍死绝对值，但不能只靠人工发现安全退化。

3. Shared TSB 在 Joint 中会双向漂移

Joint 中：

L_state 更新 WM、decoder、TSB；
L_policy 更新 Policy、TSB；
Encoder 冻结。

虽然分开 backward，但共享 TSB 仍同时被两个目标拉扯，可能导致：

WM 更准，但真实环境 Policy 下降
或
Policy 适应 WM 误差，WM state fidelity 下降


当前仅记录分支 grad norm 不够。

建议改成分阶段：

固定 Policy/TSB，只训练 WM；
固定 Policy Head，用 decision-consistency 微调 WM 或小型 WM adapter；
固定 WM，只用小学习率训练 Policy 的 imagined-state adapter；
最后才允许 TSB 小学习率联合更新；
若两个分支在 TSB 上梯度 cosine 长期为负，使用 PCGrad 或停止共享 TSB 更新。

同时 Joint 每轮必须混入真实 Bank 状态上的 RL/DAgger anchor loss，不能只在 imagined state 上做 consistency。

4. 当前 L_policy 定义仍可能让 Policy 学会迁就错误 WM

文档允许 WM 误差下 corrective action 不同，这是合理的；但如果只比较 predicted pose 与 actual pose，Policy 可能学会“补偿 WM 的系统性错误”，然后回到真实 ENV 反而退化。

建议 Joint 的 Policy 目标拆成：

L_robust:
  在扰动后的 imagined latent 上仍输出安全、可行的恢复动作

L_anchor:
  在真实 Bank 上保持 real-env Policy/GRPO 行为

L_KL:
  约束相对 pre-joint Policy 的漂移


不要要求 imagined action 与 real action 完全一致，但要保证在真实状态上的表现不下降。

WM 还有两个结构风险
5. LD 纯 latent residual + absolute decoder 风险较高

当前明确“不做 geometry reprojection”，要求网络直接预测未来 LD latent，再绝对解码 10D polyline。等于让小 WM 同时学习 ego-motion compensation、车道延伸和地图补全。

这与之前收敛的混合方案相比风险更高。建议至少保留基线对照：

future LD =
ego-motion reprojection
+ learned residual


若坚持纯 decoder，必须用 3/6-step lane topology、continuity、coverage 做否决测试，不能只看 LD L1。

6. 不建模 birth/death 会让 3 秒 rollout 偏乐观

当前 WM 只保留 t0 可见实体，不预测新进入目标、遮挡后重现或消失。文档已经承认密集交通会偏乐观。

第一版可以不做完整 birth head，但至少需要：

survival/valid probability；
对 t0 ROI 边缘实体做风险标识；
Joint/Policy 对缺失新实体进行 token dropout 或 adversarial occlusion augmentation；
dense traffic 单独报告，不得被总体均值掩盖。
实现前的阻断项

在开跑 RL/Joint 前必须解决：

DRL_PathPlan_Code.txt 中现有 PPO 路径仍包含 value、cost value 和 GAE，而新合同说无 critic、无 GAE。必须保证新 Trainer 不误复用旧语义。
验证 TruncatedNormal 边界采样与 log_prob 一致性，否则 PPO ratio 无效。
验证同 scene seed 的 G 条 rollout 除 policy sampling 外环境随机性一致，否则 group comparison 无意义。
移除 WM 中 target-only free-slot 新实体监督，并停用 legacy one-step + expert NLL Joint Trainer。
彻底切断 val500 的 checkpoint selection，改用 train2000 内固定 holdout。
定案

RL 主方向可执行，但不建议原样开长训。

优先修改顺序：

GRPO 从“episode advantage 全步广播”升级到 active-set step baseline；
success gate 增加 collision/off-road 非劣化条件；
Joint 改成 WM 对齐、Policy 鲁棒适配、最后有限联合，避免共享 TSB 同时漂移；
在纯 LD decoder 和 reprojection + residual 之间做硬对照；
清理 Trainer、distribution 和 val 泄漏等实现阻断项。

一句话说：RL 与 WM 解耦是优点，Joint 的目标也对；但当前 GRPO 的信用分配太粗，Joint 对共享 TSB 的保护不足，纯 LD 未来预测和不可见新实体是后续 WM rollout 的主要盲区。