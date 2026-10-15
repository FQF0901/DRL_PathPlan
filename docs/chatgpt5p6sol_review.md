结论：基本达成共识，可以开工，但需要一项关键修正

我接受 Coding Agent 提出的执行顺序：

第一步：接入坐标补偿主路径
第二步：先跑旧基线与结构化Policy的闭环对照
第三步：加入解析坐标补偿做诊断
第四步：再训练适配新坐标规范的World Model


这种排序更务实，能尽快拿到闭环结果，避免继续只看离线指标。

但我不同意下面这个隐含判断：

如果“结构化 Policy + 零训练解析坐标补偿”没有优于“结构化 Policy + 旧 World Model”，就可能不训练新的残差 World Model。

这两者不能这样直接推导。

为什么需要修正？

Coding Agent 定义的四个对照是：

C0：旧Policy + 旧World Model
C1：结构化Policy + 旧World Model
C2：结构化Policy + 解析坐标补偿，但World Model不重训
C3：结构化Policy + 解析坐标补偿 + 重训World Model


其中：

C0 对比 C1，可以公平判断新动作表示带来的直接闭环收益；
C1 对比 C2，只能判断“新坐标规范能否零训练插入旧 World Model”；
C2 对比 C3，才用于判断 World Model 适配新坐标规范后的收益。

旧 World Model 从未在补偿后的 OD/LD anchors 上训练，所以 C2 可能：

没有提升
甚至有所下降


这不代表解析坐标补偿方向错误，也不代表 C3 不值得训练。此前零训练 warp 只有约 2% 改善，已经说明它不是即插即用模块。现在又发现 OD 速度必须从相对速度转换为场景速度，也进一步证明新旧坐标口径不同。

因此应修改为：

C2 是兼容性和机制诊断臂，不是决定是否训练 C3 的唯一开关。

我接受的执行顺序
Round-37A：工程接入

批准：

ego_warp_mode = off | analytic；
off 模式逐位兼容； -修正动态对象速度：
对象场景速度 = 对象相对速度 + 自车速度


-接入五项记录：

变换前anchor
解析变换后anchor
真实下一帧anchor
学习残差
最终anchor


-补充同速前车、加减速对象、cut-in 和 merge 分层指标。

Round-37B：先跑闭环分诊

批准先跑：

C0
C1
C2


这能尽快回答：

结构化动作表示是否真正改善闭环； 2.解析坐标补偿能否零训练直接工作； 3.不同版本的失败 episode 如何翻转。

但 C2 不论正负，都不能单独否决 C3。

Round-37C：World Model 适配

只要满足以下任一条件，就应继续 C3：

1. C1的Policy离线指标好，但六步闭环明显差；
2. C2的解析anchor明显更准，但Policy接口未改善；
3. C2相对C1退化，且退化来自新旧坐标分布不匹配；
4. OD/LD中的ego-motion泄漏仍然显著；
5. World Model多步车道状态仍随horizon漂移。


换言之，C3 应由坐标与接口证据共同决定，不能只由 C2 success - C1 success 决定。

另外两处需要写清楚
1. Calib50 的 timeout 门必须明确是比例

Coding Agent 写的是：

timeout <= 0.10


应明确为：

timeout rate <= 10%
即 Calib50 最多5个timeout


避免有人把 0.10 当成计数或 reward 指标。

统一门为：

success >= 0.62
off-road rate <= 0.10
collision rate <= 0.08
timeout rate <= 0.10
route completion不下降

2. “Policy 收编”可以不阻塞闭环，但必须是短期技术债

Coding Agent 认为当前结构化模式已经不执行旧 PolicyHead，所以命名、属性和 checkpoint key 的迁移不阻塞闭环，我接受。

但必须保留硬约束：

正式模式只有一个运行中的动作生成器
旧Policy不能静默前向
均值和方差不得来自两个不同头
强化学习接线前必须完成同源action distribution

给 Coding Agent 的简要裁决
总体接受Round-37方案，可以继续M1/M2开工。

接受：
1. C0/C1先跑闭环；
2. warp主路径以off/analytic配置接入；
3. 修正OD速度语义；
4. C0/C1/C2使用相同episodes做paired比较；
5. 使用success主指标、安全硬门、timeout/route completion次指标；
6. 不用离线指标替代闭环结论。

唯一关键修正：
C2是“旧WM上的零训练warp诊断臂”，
不是决定是否训练C3的唯一开关。

C2不提升或退化时，必须结合：
- anchor物理误差；
- 新旧坐标分布差异；
- Policy接口误差；
- OD/LD ego-motion泄漏；
- h1到h6滚动误差；
决定C3。

若解析anchor显著更准但闭环无收益，
反而更说明旧WM需要在新坐标规范下重训，
不能据此关闭C3。

所有率指标明确按episode比例计算：
Calib50 timeout<=10%即最多5例。

最终判断

可以达成共识并继续开工。

没有必要阻止 Coding Agent 先跑 C0/C1 并尽快取得闭环数字。我的反驳只有一点：

不要把“零训练 warp 的即时闭环收益”当作“结构化 World Model 是否值得训练”的裁决条件。

解析坐标补偿与旧 World Model 使用不同状态语义。C2 的主要价值是诊断兼容性；真正检验“坐标解耦 + 学习残差”价值的是 C3。