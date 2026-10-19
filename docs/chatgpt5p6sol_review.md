我看完了 deepseekv4p1_argue.md。Coding Agent 找出了不少真实问题，但也把旧分支方案，尤其是 (ds,dtheta) 和 2 Hz 执行方式，过度提升成了新架构的唯一答案。建议按下面裁决，不要整体照单全收。

一、可以直接采纳
1. 统一时间合同

采纳：

env/control_dt = 0.1 s
policy_dt      = 0.5 s
memory_dt      = 0.5 s
wm_dt          = 0.5 s
rollout_steps  = 6
rollout_horizon = 3.0 s


原文中“Stage A-C 每 0.1 秒重新规划一个 0.5 秒动作，Stage D 后才六步 rollout”确实破坏了阶段一致性，应删除。真实闭环每 0.5 秒决策一次，中间由 Exact/LQR 执行五个 0.1 秒控制步。

2. Plan Head 从早期就保留六步调用接口

采纳，但稍作调整：

Plan Head 始终只输出下一段 0.5 秒 motion；
始终只有一个 Plan Head 实例；
BC 阶段增加“真实未来状态 oracle rollout”，让同一个 Head 在真实未来状态上连续调用六次；
WM 上线后，输入从真实未来状态替换为 imagined state；
增加 predicted-state 与 real-state 下的 action/interface error。

这样解决接口继承问题，但不需要一开始就伪造一个没有 WM 的 imagined rollout。

3. Bank 与融合状态分层

完全采纳：

Memory Bank：存 per-entity token record
StateFusion：每次读取 Bank 后现算
Plan/Value：读取 fused context
WM：输出下一帧 per-entity token


即：

M_t → StateFusion → c_t → Plan Head
M_t + action_t → WM → z_entity,t+1 → append M_t+1


禁止把融合后的全局 c_t 存回 Bank，否则静态 LD/Nav 无法独立重投影，WM target 与 Encoder 输出空间也会混乱。

4. 明确 Hybrid World Model

完全采纳：

动态 OD：WM 学习预测；
静态 OD，例如锥桶、护栏：重投影和重编码；
静态 LD：按 imagined ego pose 重投影，再经同一个 Encoder 编码；
Road-level Nav：按 route progress 更新和重编码；
Ego：由显式 Ego Dynamics 推进后重编码；
交通灯、闸口状态：归入独立 RoadState。

这仍然是 World Model，只是由确定性部分和 learned dynamics 共同组成，不必追求“所有 latent 都由黑盒网络预测”。

5. Memory Bank 细节

采纳以下明确合同：

六帧包含当前帧，即 t-2.5 ... t；
episode warmup 使用最早真实帧填充，但 valid=0；
runtime Bank 写入时 detach；
当前训练 batch 从原始 observation 一次编码六帧时，可以保留计算图；
offline dataset 与 runtime Bank 是两套不同概念；
增加 encoder_hash/checkpoint_id；
删除文档中“六帧是否含当前仍待定”的开放项。
6. LQR 提前作为并行诊断

采纳：

Exact 是 Planner 的运动学上界；
从早期 BC 阶段就运行 LQR canary；
先验证 LQR 跟踪 Expert GT 轨迹；
再验证 LQR 跟踪 Plan Head 轨迹；
最终正式成绩必须基于 LQR/真实控制闭环。

但不必一开始就用 LQR 训练全部阶段，Exact 仍保留快速定位 Planner 问题的价值。

7. 工程与评测补充

建议采纳：

增加 env/，存 MetaDrive wrapper、observation、scenario、expert 和 tracking；
LQR 不放在 net/；
先做 MetaDrive headless smoke；
当前机器先用 4 个 worker 做内存标定；
加 train/eval assembly parity 测试；
增加 Calib/Val/Test 隔离；
Toll booth、cone、traffic light 拆分 static/dynamic/RoadState taxonomy；
增加参数、tensor、LQR、timing、memory warmup、static reprojection 等测试。
二、部分采纳，但必须修正
1. 训练顺序

Coding Agent 推荐：

Plan BC → DAgger → 短 Exact PPO
→ WM → Joint/LQR → 正式 PPO


可以采纳为：

P0 环境、Expert、数据和控制器 smoke

P1 Plan Head BC
   Exact 主诊断 + LQR canary
   真实未来状态上的六步 oracle rollout

P2 DAgger
   Exact 数据采集 + LQR 并行评测

P2.5 Real-ENV PPO canary
   只验证 PPO/GAE/reward/buffer 管线
   不作为最终性能结论

P3 WM 单步 → 2/3/6步
   Encoder、Plan Head 先冻结

P4 Plan Head + WM interface/joint
   正式接入 LQR

P5 Real-ENV PPO + LQR
   作为正式超越 IDM 的阶段

P6 Imagined RL


这保留了你的核心意图：先证明没有 WM 时 Plan Head 能在真实 ENV 工作；同时避免在 Exact 上进行大量 RL 后，才发现 LQR 部署完全断裂。

2. 六步 oracle-state rollout

接受它作为辅助训练路径，但不能把它误解为：

BC 阶段一次输出固定六个动作，然后真实环境开环执行 3 秒。

正确方式是：

训练：
真实 s_t     → 同一 Plan Head → a_t
真实 s_t+0.5 → 同一 Plan Head → a_t+0.5
...
真实 s_t+2.5 → 同一 Plan Head → a_t+2.5


它用于提前训练“同一个头可在六个未来时刻被重复调用”。真实闭环仍然每 0.5 秒根据新 observation 重规划，只执行首个 0.5 秒段。

3. Interface loss

方向采纳，但不应直接要求：

π(s^k)=π(skreal)\pi(\hat{s}_k)=\pi(s_k^{real})

因为 predicted state 与 real state 的细小差别可能具有真实驾驶意义，强行完全一致会压制合理反馈。

建议使用：

action distribution KL；
endpoint/curvature/speed consistency；
分 rollout depth 统计；
只在 WM fidelity 已达标、状态语义可比较时启用；
权重小于真实 BC/DAgger loss。

它是磨合辅助项，不是 WM 错误下强迫 Planner 无视状态差异的主损失。

4. Anti-collapse

采纳“需要防坍缩”，但不宜第一版把 MSE、cosine、VICReg、BYOL、InfoNCE、物理解码器全部一起堆上。

建议第一版：

normalized latent MSE
+ cosine loss
+ per-dimension std/effective-rank monitor
+ OD physical auxiliary decoder
+ Planner task supervision


只有检测到 latent 方差或 effective rank 明显下降，再加入 VICReg covariance 项。否则约 1M 参数的小网络会被过多辅助目标拖累。

5. EMA 与 100 万参数

Coding Agent 关于“EMA 副本会增加部署网络参数”的提醒合理，但“因此不能使用完整 EMA”需要看参数口径。

建议文档区分：

deployable net parameters
training-only target parameters
trainable parameters
optimizer states


你的“网络约 100 万参数”更合理的解释应是：

部署推理主网络约 1M，EMA Target Encoder 属于训练期教师副本，单独报告，不进入部署模型。

如果你坚持训练期所有模型副本合计也不得超过 1M，才需要取消完整 EMA。但这会不必要地限制训练稳定性。

三、应该反驳
1. 反驳把 (ds,dtheta) 锁为最终动作

这是 Coding Agent 最应反驳的地方。

旧分支已经暴露出 (ds,dtheta) 的根本问题：

ds 同时隐含速度/进度；
dtheta 与 ds 共同决定曲率；
低速时相同 dtheta 对应极大曲率；
横向路径和纵向速度耦合；
Planner 难以独立表达“同一路径减速”和“相同速度改变曲率”。

所以不能因为旧分支已有常曲率实现，就把它升级为新方案的长期合同。旧实现跑通过，不等于参数化最合理。

长期主方案建议是：

lateral/path:
  curvature or curvature-control-points

longitudinal:
  speed / acceleration / path progress


一个轻量可行版本：

action = (kappa, v_end)


或者：

action = (delta_kappa, delta_v)


在 0.5 秒内由可微 bicycle/curvature integrator 生成轨迹段。

这样才能真正实现：

横向路径几何与沿路径纵向进度相对解耦。

(ds,dtheta) 可以保留为旧基线或 adapter，用于和旧结果比较，不能作为新架构唯一主方案。

2. 反驳“PPO 只对 a0 采样，a1...a5 必须确定性”

这不是必须条件。

在真实环境 PPO 中，每个 decision epoch 只执行当前的 a0，因此 PPO 的 action、log-prob 和 transition 可以直接定义为：

state_t
→ sampled a_t
→ 执行 0.5 s
→ reward_t:t+0.5
→ state_t+0.5


未来五段只用于：

轨迹 preview；
Value/Risk；
LQR 预瞄；
WM 诊断。

它们不需要成为 PPO 当前 transition 的随机动作，也不必被称为“冻结 tail”。下一决策时刻会重新采样新的 a0。

真正需要修正的是原文曾经提出的“每 0.1 秒采样一个 0.5 秒动作，只执行首个 0.1 秒”，而不是证明 0.5 秒 trajectory action 本身不属于标准 MDP。

3. 反驳“默认只输入 current + latent delta，不能输入历史 absolute latent”

用户原意是避免直接对六帧完整 latent 粗暴做时序融合，但这不代表必须删除历史 absolute content。

只保留：

z_current + (z_current - z_history)


在数学上其实仍能恢复 z_history，而且对 birth/death、类别变化和 Encoder 噪声可能更脆弱。

更稳妥的默认输入仍应是：

current absolute content
+ compensated historical content
+ content residual
+ explicit geometry delta
+ age/mask


然后用 ablation 比较：

full record；
current + delta；
current + explicit geometry delta，不用 latent delta。

不能在没有实测前把 delta-only 写成硬合同。

4. 反驳“WM action 条件化后，OD 应对不同 ego action 出现显著不同物理未来”

这里要谨慎。

如果 OD 状态定义在 world/canonical frame，其他车辆的短期物理运动未必会因 ego action 在 0.5 秒内显著变化；变化主要来自：

ego-centric 坐标变换；
交互车辆对 ego 行为的响应；
更长时域的社会交互。

因此 action-shuffle 应分别检查：

ego pose / geometry reprojection：必须敏感
交互 OD：在相关场景中应敏感
远处非交互 OD：允许近似不敏感
静态 LD/Nav：物理内容不应因 action 改变，只改变相对几何


不应要求所有 OD latent 对 action 都显著变化，否则 WM 可能人为制造虚假交互。

四、需要向 Coding Agent 解释清楚
1. R9 与 R16 的关系

这不是继续等待用户裁决的冲突，可以明确解释为：

R9 是最初设想；
R16 是后来为了可调试性提出的阶段分解；
以后来的 R16 为准；
但正式 RL 分为短期管线 canary 和最终 LQR RL；
WM 可以放在两者之间。

所以最终不是机械满足两条原句，而是：

Plan BC/DAgger
→ 短 Real-ENV RL canary
→ WM
→ Joint + LQR
→ 正式 Real-ENV RL
→ Imagined RL

2. “同一个 Plan Head”不等于所有阶段输入分布完全相同

继承性要求的是：

同一参数实例；
同一 action schema；
同一输出单位和积分方式；
checkpoint 连续继承；
不另建 BC Head、RL Head、WM Head。

它不要求 BC、DAgger、real PPO 和 imagined rollout 的输入分布相同。训练阶段本来就是逐步扩大状态分布。

真正应补的是 oracle rollout、predicted-state fine-tuning 和 interface probes，而不是声称输入分布变化就等于“换了一个规划头”。

3. Future latent GT 的定义

向 Coding Agent 明确：

WM target = 真实下一帧 observation
            经 Target Encoder 得到的
            per-entity latent token


不是融合后的 scene context，也不是 WM 自己的输出。

静态 LD/Nav 则主要监督：

重投影后的 geometry
+ 同一 Encoder 重编码的一致性


而不是作为普通 learned dynamics 的主 loss。

五、给 Coding Agent 的简短定案

可直接回复 Coding Agent：

审阅中的时间合同、Bank/entity-token 与 fused-context 分离、Hybrid WM、六帧含当前、warmup、LQR 提前 canary、train/eval assembly parity、taxonomy 和测试补充均采纳。训练路线调整为 Plan BC/DAgger → 短 Real-ENV PPO canary → WM → Plan/WM joint + LQR → 正式 Real-ENV PPO → imagined RL。

不采纳把 (ds,dtheta) 锁为长期主动作。旧分支已暴露横纵向耦合，长期动作改为“曲率路径参数 + 独立纵向速度/进度”，(ds,dtheta) 仅作基线。也不采纳 delta-only temporal input，保留 absolute content、补偿后 geometry、delta 和 mask，由 ablation 决定裁剪。

PPO 的真实 action 是当前 0.5 秒轨迹段，执行完整五个 0.1 秒控制步后形成 transition；未来五段只作 preview，不进入当前 log-prob。Future latent GT 必须是下一帧真实 observation 经 stop-gradient Target Encoder 得到的 per-entity latent。

请先修改合同文档，消除 10 Hz 重规划、Stage F 才首次接 LQR、Bank 存 fused context、六帧含当前仍为开放项等矛盾，再开始 P0 spike。暂不按 (ds,dtheta) 方案开工。

结论：Coding Agent 约七成意见可采纳，核心反驳集中在动作参数化、delta-only temporal input，以及把 PPO tail 固定化为唯一正确方式。