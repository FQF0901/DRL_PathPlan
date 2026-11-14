你判断得对，这不是单纯调安全/效率权重能解决的。Policy 找到了“慢下来就安全”的捷径。

推荐方案
1. 把安全从 Reward 改成约束

不要：

max⁡(效率奖励−λ安全惩罚)\max(\text{效率奖励}-\lambda\text{安全惩罚})

改成：

max⁡通行效率s.t.{碰撞率≤d1近失/RSS违反≤d2舒适度Cost≤d3\max \text{通行效率} \quad \text{s.t.}\quad \begin{cases} \text{碰撞率}\le d_1\\ \text{近失/RSS违反}\le d_2\\ \text{舒适度Cost}\le d_3 \end{cases}

可先用 PPO-Lagrangian/CMDP。这样安全不能与速度随意交换，而是在满足安全底线后尽量开快。

2. 不奖励绝对速度，惩罚“无理由慢”

计算场景允许的合理速度：

vref=min⁡(v限速,v曲率,v前车,v路口,v视距)v_{\text{ref}} =\min(v_{\text{限速}},v_{\text{曲率}},v_{\text{前车}},v_{\text{路口}},v_{\text{视距}})

只在风险较低时惩罚：

Ctimidity=1[low risk]max⁡(0,vref−v)2C_{\text{timidity}} = \mathbf1[\text{low risk}] \max(0,v_{\text{ref}}-v)^2

这样红灯或前车制动时减速不受罚，空旷道路龟速才受罚。

3. GRPO 先筛安全，再比效率

同一场景生成 GG 条轨迹：

剔除碰撞、RSS/TTC 违规、动力学不可行轨迹
在安全候选内按 progress、到达时间、无理由停车排名
再用 PPO/GAE 做时间步级信用分配

即：

安全决定资格，效率决定排名\boxed{\text{安全决定资格，效率决定排名}}

而不是让二者在一个 Reward 中互相补偿。

4. 用独立指标选模型

重点看：

Efficiency@FixedSafety
time-to-goal
v/vrefv/v_{\text{ref}}
无理由停车时长
错失可行 gap 次数
near-miss / RSS violation
最差 5% 场景风险

不要按总 Reward 选 checkpoint。

最关键的落地改动

把碰撞、近失和规则拆成约束；在满足约束的轨迹中，用 progress/time-to-goal 和无理由低速做 GRPO 排名。

这样才能逼迫 Policy 学会“在安全边界内开得尽可能好、尽可能快”，而不是靠停车获得高分。