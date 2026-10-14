给 Coding Agent 的下一阶段开工方案
一、阶段目标

下一阶段先不做自主选道、DAgger、MoE 或强化学习，优先建立两项基础能力：

1. 给定合法目标车道，稳定生成车道内可执行路径；
2. 给定目标车道或相邻车道，稳定完成保持、变道和 merge 续接。


理由：

必须先证明网络能执行正确的车道决策，再训练网络自主做车道决策。否则收费站、merge 失败无法区分“选错车道”与“选对但不会执行”。

二、架构方案
1. 横向规划改为曲率路径

新增横向路径头，第一版输出三个曲率控制点：

近端曲率 κ_near
中段曲率 κ_mid
远端曲率 κ_far


通过平滑插值和积分得到连续路径：

θ(s)=θ0+∫0sκ(u) du\theta(s)=\theta_0+\int_0^s\kappa(u)\,du x(s)=x0+∫0scos⁡θ(u) du,y(s)=y0+∫0ssin⁡θ(u) dux(s)=x_0+\int_0^s\cos\theta(u)\,du,\quad y(s)=y_0+\int_0^s\sin\theta(u)\,du

路径应以合法车道中心线为锚，网络预测曲率残差，不从零生成任意轨迹。

第一阶段仅支持：

保持当前车道
给定左侧车道
给定右侧车道
给定路线续接车道


暂不让网络自主选择目标车道。

2. 纵向规划必须条件化于横向路径

纵向头读取：

横向路径曲率
路径道路边界余量
路径上的静态障碍距离
路径上的动态对象距离与TTC
当前速度
导航进度


输出未来六步：

沿路径累计进度 s[1:6]
或速度 v[1:6]


外部仍转换为现有接口：

ds_k = s_k - s_(k-1)
dtheta_k = kappa_k × ds_k


这样固定 LQR、环境接口和历史评测工具都不需要修改。

3. Target-lane 的使用边界

本阶段允许使用外部合法 target lane，只用于测试“正确决策能否被执行”。

必须同时增加随机 dropout：

训练初期：target-lane token dropout 20%
后期：逐步提升到50%


避免新网络继续完全依赖外部 target-lane。

外部障碍感知选道规则只能作为诊断上界，不得进入生产模型。

三、训练范围
训练
横向曲率路径头
路径条件化纵向头
必要的PolicyBackbone末层
候选路径几何特征编码层

冻结
World Model
Encoder主体
Memory Encoder主体
旧PolicyHead
MoE experts与router
ValueHead
LQR

数据

优先使用：

普通直道和弯道
给定目标车道的左右变道
merge续接
收费站中已知可行闸口路径
路口转向前的给定合法车道


收费站的可行闸口只用于“给定目标后的路径执行”测试，不用于生产选道规则。

四、训练任务顺序
Task 1：车道保持

训练：

给定当前合法车道
→ 生成连续曲率路径
→ 生成合理速度/进度

Task 2：给定目标车道的变道

训练：

给定左/右相邻合法车道
→ 提前启动变道
→ 连续跨越车道
→ 在目标车道回正

Task 3：Merge 续接

训练：

保持当前车道或给定续接车道
→ 不压线
→ 不产生无意义横向漂移

Task 4：收费站可行闸口执行

给定正确闸口车道，验证：

能否提前换道
路径footprint是否避开岗亭
速度是否与狭窄通道匹配


如果给定正确闸口仍经常失败，禁止进入自主闸口选择训练。

五、必须增加的 Probes / Hooks

所有 hook 默认关闭，关闭时必须：

前向max_abs_diff = 0
不改变RNG
不改变checkpoint
不改变参数更新顺序

1. 输入契约 Hook

记录：

道路级导航
target-lane来源
目标车道几何
车道slots及mask
静态对象slots
动态对象slots
坐标系、单位、时间戳

2. 横向路径 Hook

每个控制点和每个采样位置保存：

κ_near/mid/far
κ(s)
x/y/theta
相对目标车道横向误差
相对目标车道航向误差
道路边界余量
曲率变化率

3. 纵向规划 Hook

保存：

每步s/v/ds
路径曲率
最近静态障碍距离
最近动态对象距离
TTC
计划停车距离
可通行路径长度

4. Footprint 几何 Hook

对连续插值轨迹记录：

首次越界位置
首次碰撞位置
最小道路余量
最小静态障碍余量
道路内比例


不能只检查六个离散点。

5. LQR 对照 Hook

记录：

规划路径
插值后的LQR reference
实际车辆轨迹
raw steer/throttle
饱和比例
规划误差与跟踪误差


明确区分：

路径本身错误
路径正确但LQR跟踪错误

6. 网络性状 Hook

逐模块记录：

输入输出shape
均值、方差、范围
effective rank
激活饱和
梯度范数
参数更新范数
target-lane dropout状态

7. 干预 Probe

至少执行：

保持道路级route不变，交换目标车道
保持目标车道不变，改变route command
静态障碍移近/移远
动态对象TTC增大/减小
左右镜像
target-lane token置零


检查路径方向和速度响应是否符合物理直觉。

六、验证准则
Gate 0：代码与数据契约
旧模型默认路径逐位不变
checkpoint严格加载无缺失
训练参数集合与optimizer一致
坐标系、单位、mask全部可验证
所有新hook默认关闭bit-exact

Gate 1：普通车道保持

在普通道路子集：

success >= 0.90
计划footprint越界率 <= 2%
collision接近0
timeout <= 5%
曲率和曲率变化率不超过专家P95×1.25
LQR steering saturation显著低于当前基线

Gate 2：给定车道的变道执行
合法target下变道完成率 >= 90%
路径越界率 <= 5%
变道方向正确率 >= 95%
变道后回正率 >= 90%
不必要反复换道率 <= 5%

Gate 3：Merge

相对当前基线：

merge计划越界下降 >= 50%
merge实际off-road下降 >= 50%
本车道续接候选成功率 >= 90%
路径正确但LQR跟踪失败比例不得上升

Gate 4：收费站正确目标执行

在给定可行闸口的条件下：

可行路径覆盖 >= 95%
岗亭footprint collision下降 >= 80%
提前换道完成率 >= 85%
收费站success显著非零，目标 >= 60%
collision不得高于基线


若给定正确闸口仍无法通过，不得进入自主闸口选择。

Gate 5：纵向单调性
障碍更近 → 计划速度不得增加，正确率 >=95%
TTC更低 → 计划进度不得增加，正确率 >=95%
路径曲率更大 → 参考速度不得提高，正确率 >=90%
障碍移除 → 应恢复进度，正确率 >=90%

Gate 6：综合闭环

第一阶段新模型至少达到：

calib50 success >= 0.62
off-road <= 0.10
collision <= 0.08
timeout <= 0.10
route completion不下降


通过后进入 clean150；否则停止并按场景归因。

七、下一阶段禁止事项
暂不训练自主车道/闸口选择
暂不删除target-lane token
暂不修改World Model
暂不做ego-motion warp重训
暂不进入DAgger
暂不启用MoE
暂不启动PPO/GRPO
暂不修改LQR
暂不扩大网络容量

八、通过后的后续顺序
1. 给定车道下的保持、变道和merge执行过门
2. 网络内部生成并评分车道/闸口候选
3. 移除外部唯一target-lane，保留道路级导航
4. World Model适配新的“曲率路径+速度”动作表示
5. 加入OD/LD显式anchor和ego-motion坐标补偿
6. DAgger同时训练Policy与World Model
7. MoE专项能力
8. RL优化长期闭环收益

给 Coding Agent 的一句话开工令

下一阶段只攻克“给定合法目标车道后的路径执行能力”：用车道锚定的横向曲率路径和路径条件化纵向速度替代内部 ds/dtheta 耦合表示，完成普通保持、变道、merge和正确收费闸口执行；全程保留输入、网络、路径、footprint与LQR动态 hooks。未过路径执行与安全门，不进入自主选道、World Model重训、DAgger、MoE或RL。