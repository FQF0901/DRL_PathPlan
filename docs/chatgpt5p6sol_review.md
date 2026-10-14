裁决：已达成共识，可以开工

Coding Agent 关于初始化的反驳成立：

理论上“双零初始化”会导致零梯度；
本轮训练实际将 alpha_ego 冷启动为 1.0，梯度、参数更新、optimizer 白名单和共享参数冻结均已验证；
Round-32 canary 的变化可以归因于动作条件化分支；
不再围绕初始化继续审计。

正式版本统一简化为：

隐藏层：Kaiming/Xavier
输出层：zero-init
残差增益：固定为1，删除可学习alpha或不再依赖外部冷启动


既保持初始前向不变，也避免隐式训练契约。

批准立即开工
1. Round-33A，限时只读归因

只回答三个问题：

AC1 新增碰撞主要属于追尾、交叉口冲突、merge/cut-in、侧碰还是静态障碍物；
ds overshoot 来自初始 Policy action、WM 响应放大，还是后续 Policy 正反馈；
Gate 3 指标的重复运行噪声是否低于当前恶化幅度。

不要扩展新的通用 probes。

2. Round-33B，解析 ego-motion warp 对照

执行：

W0：当前实现
W1：LD anchor做ego SE(2)坐标补偿
W2：LD warp + OD恒速运动先验后做ego SE(2)补偿


验证：

lane-relative信息保留
OD中的ego-motion泄漏
E_interface
h5 ds overshoot


若至少三项明确改善，进入 GO_ANALYTIC_EGO_WARP。

3. Round-33C，修正版 canary

三臂：

B0：当前基线
B1：action-conditioning v2
B2：B1 + analytic ego-motion warp


B1 统一使用：

动作特征：[ds, dtheta, kappa]
隐藏层正常初始化
输出层zero-init
固定残差增益
ego物理监督：delta_progress、delta_speed、delta_heading

硬门

晋级必须同时满足：

Action Swap top-1 >= 0.35
top-3 >= 0.85
ds-progress单调率 >= 0.90
E_interface/E_action/overshoot不得恶化超过3%
lane-relative信息不得退化
OD ego-motion泄漏不得增加
success >= 0.58
off-road <= 0.14
collision <= 0.08
timeout <= 11
route completion >= 0.862


任何接口指标恶化超过 5%，或 collision 再次恶化，立即停止。

继续冻结
60/40 Policy数据适配
DAgger
MoE
PPO/GRPO
BPTT
Policy解冻
LQR修改
WM主体扩容


一句话给 Coding Agent：

反驳接受，初始化问题结案。按 Round-33A → ego-motion warp → B0/B1/B2 canary 开工，重点解决动作条件化增强后产生的 rollout overshoot 与 collision，不再扩大审计范围。