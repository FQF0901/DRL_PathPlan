我基本同意，而且这会把主线重新收敛到你最初想要的架构，而不是继续围绕 env.obs.stack_history → TemporalPool 打补丁。

但需要把边界说清楚：

Real/Imagine Memory Bank 应成为唯一历史状态接口，ST-GNN 直接读取 Bank；stack_history 和 TemporalPool 都退出网络主路径。
 不过 Bank 不能只是把当前 [T,N,D] 张量换个类名保存。它必须保存“实体记录 + 坐标元数据 + 生命周期元数据”，由一个统一的 Bank View Builder 完成 SE(2) 补偿、stable-ID 对齐、slot 构造和 mask 生成，然后再喂给 ST-GNN。

当前网络确实有空间建模，但只是把 Ego/Nav/OD/LD 当前 token 拼在一起，经过两层 joint Transformer。它能做全局 token 交互，却不是显式的图结构，也没有清楚限定 OD↔OD、LD↔LD、OD↔LD、Nav/Ego 条件化等关系；当前顺序是先 same-ID TemporalPool，再做当前帧空间融合。

因此，我建议不是“在现有 TemporalPool 后再加 ST-GNN”，而是：

删除 TemporalPool 主路径，让 ST-GNN 同时承担实体级时序聚合和当前时刻空间交互。

一、Real/Imagine Memory Bank 应该怎样设计
1. 不再保留 env.obs.stack_history

这里我同意你的判断。

env.obs 的职责应缩减为：

生成当前时刻结构化 observation；
为当前 OD/LD/Nav 提供稳定 ID 和几何 anchor；
提供 ego pose、timestamp、scene/episode ID；
不生成历史 stack；
不负责把历史实体硬排入当前 slot。

历史由统一的 MemoryBank 管理：

MetaDrive Env
    ↓ current structured observation
Current-frame Encoder
    ↓ per-entity token records
RealMemoryBank.append(...)
    ↓
BankViewBuilder
    ├─ SE(2) transform
    ├─ stable-ID trajectory construction
    ├─ lifecycle masks
    └─ sparse spatiotemporal graph
          ↓
        ST-GNN
          ↓
    fused current state
          ├─ Policy / Value / Risk Head
          └─ Ego-conditioned latent WM
                     ↓
             ImagineMemoryBank
                     ↓
                  ST-GNN
                     ↓
                  next rollout


当前代码已经有 6 帧历史、OD/LD stable ID、ego pose 和历史 SE(2) 补偿的部分基础，但历史仍以 list/tensor 形式组织，不是正式的 per-entity token record Bank，Real/Imagine Bank 也都未实现。

2. Bank 里不应该只存 latent

每个时间帧建议存一组不可变 FrameRecord：

FrameRecord
├─ episode_id
├─ timestamp
├─ ego_pose_world
├─ ego_state
├─ source: REAL | IMAGINED
├─ rollout_depth
├─ parent_frame_id
├─ action_from_parent
│
├─ OD entity records
│   ├─ stable_id
│   ├─ content_latent
│   ├─ geometry_anchor
│   ├─ taxonomy
│   ├─ present
│   └─ observed_timestamp
│
├─ LD entity records
│   ├─ stable_id
│   ├─ content_latent
│   ├─ geometry_anchor/polyline
│   ├─ road/lane/chunk metadata
│   ├─ present
│   └─ observed_timestamp
│
├─ Nav record
└─ encoder_version


必须区分：

Content latent

表示“它是什么”：

车辆类型、运动状态编码；
车道结构属性；
拓扑属性；
限速、道路类型；
动态实体行为语义。
Geometry anchor

表示“它在哪里”：

OD：中心、heading、尺寸、速度向量等；
LD：polyline 点、lane heading、chunk anchor 等；
Nav：route polyline、road ID、route progress；
Ego：world pose、速度、yaw rate。
Metadata

表示：

stable ID；
timestamp；
source；
rollout depth；
encoder version；
valid/presence；
episode。

这样做的关键原因是：

SE(2) 补偿只作用于 geometry，不直接旋转普通 latent。

若把几何位置完全压进 latent，Bank 取历史时就无法通过解析坐标变换保证历史和当前处于同一坐标系，WM 也会重新学一遍自车坐标变化。

3. Real Bank 和 Imagine Bank 的关系
RealMemoryBank

只接收真实环境产生的 observation：

容量 6 帧；
每帧间隔约 0.5 秒；
episode/scene 切换立即清空；
应存 detached encoder outputs；
不接受 WM 生成结果；
不接受 rollout 后的反写。
ImagineMemoryBank

每次 rollout 开始：

从 Real Bank 建立 snapshot；
复制 record 或采用 copy-on-write；
保留原帧 source=REAL；
WM 每 rollout 一步 append：
source=IMAGINED
rollout_depth=1...6
parent_frame_id
action_from_parent
FIFO 容量仍为 6；
rollout 完毕销毁；
绝不回写 Real Bank。

注意，这里的“复制”不一定要物理复制完整张量。可以：

Real Bank 只读；
Imagine Bank 保存父 Bank 引用；
仅保存 rollout 新生成的 frame；
Bank View 时合并读取。

这样避免每个 rollout 复制大量 token。

二、当前网络是否有空间建模
有，但不够结构化

当前网络流程大致是：

OD / LD / Nav / Ego Encoder
        ↓
Same-ID TemporalPool
        ↓
Current token set
        ↓
Joint spatial Transformer × 2
        ↓
Query pooling / Plan / Value-Risk


Spatial Fusion 会将 Ego、Nav、OD、lane token 加上 modality embedding 后一起送进 Transformer，Ego 始终有效；同时有 learned lane/OD query attention。也就是说它确实允许不同 token 发生空间交互。

但它有三个不足：

1. 拓扑没有显式表达

模型不知道一条边代表：

同一个实体的历史；
相邻车道；
predecessor/successor；
左右邻车道；
front/rear vehicle；
vehicle-to-lane；
route-to-lane；
ego-to-object。

它只能从 token 内容和 embedding 中自己推断。

2. 时间和空间割裂

TemporalPool 先把六帧压成一个 token，空间模块只看压缩后的当前 token。时间聚合过程看不到周边空间关系，空间建模过程也看不到历史结构。

例如，“前车减速导致 ego 应减速”实际上是一个时空联合问题：

同一车辆历史速度下降；
它位于 ego 当前车道前方；
gap 正在缩短；
相邻车道是否可行；
route 是否允许换道。

TemporalPool 加 joint Transformer 能近似学习，但信息先被时间压缩，且当前 TemporalPool 还存在实体/时间轴混排错误。

3. 计算关系过于稠密

所有有效 token 进入 joint Transformer，理论上任意 token 都可互相注意。对只有十余个实体时可运行，但关系语义弱，也不利于未来 WM 保持实体身份和局部因果。

所以我的判断是：

当前网络“有空间 attention”，但还不能称为真正的 ST-GNN。最合理的做法是保留 Encoder 和部分 Spatial Transformer 参数预算，替换 TemporalPool，并把空间交互改造成 typed sparse graph message passing。

三、ST-GNN 应该读取什么

ST-GNN 不应直接读取 Bank 的内部存储结构，而应读取一次性构建的 BankView。

BankView 的逻辑形态
Nodes
├─ Ego nodes:    每帧 1 个
├─ OD nodes:     每帧若干动态/静态物体
├─ LD nodes:     每帧若干 lane/chunk
└─ Nav nodes:    当前 route/context

Edges
├─ temporal same-entity
├─ ego-motion
├─ OD ↔ OD spatial relation
├─ OD ↔ LD association
├─ LD ↔ LD topology
├─ Ego ↔ OD
├─ Ego ↔ LD
├─ Nav ↔ LD
└─ optional cross-time interaction


每个节点特征由以下部分组成：

node_feature =
    content_latent
  + transformed_geometry_embedding
  + relative_time_embedding
  + modality_embedding
  + source_embedding
  + rollout_depth_embedding
  + lifecycle_embedding


但是不建议简单 concat 后直接过巨型 MLP。更适合：

content latent 作为主状态；
geometry/time/source/lifecycle 生成 edge bias 或 FiLM/gating；
typed edge 使用不同的小型 message MLP；
共享 hidden size 约 96 或 128。
四、历时帧 SE(2) 补偿怎么处理
原则：Bank 保存原始参考系，View Builder 按查询时刻补偿

不要在 append 时把历史永久改写到某个当前 ego frame，因为“当前”会不断变化。

建议：

Real Bank

每帧保存：

ego_pose_world[t]
实体在 ego_t 或 world 下的 geometry；
明确 frame_id。
构建当前时刻 BankView

以查询时刻 qq 的 ego frame 为统一坐标系，对历史时刻 tt：

Tq←t=Tworld←egoq−1Tworld←egotT_{q \leftarrow t} = T^{-1}_{world \leftarrow ego_q} T_{world \leftarrow ego_t}

然后：

pi(q)=Tq←tpi(t)p_i^{(q)} = T_{q \leftarrow t}p_i^{(t)}

heading：

ψi(q)=wrap⁡(ψi(t)+ψtego−ψqego)\psi_i^{(q)} = \operatorname{wrap} \left( \psi_i^{(t)}+\psi_t^{ego}-\psi_q^{ego} \right)

速度向量只旋转，不平移：

vi(q)=Rq←tvi(t)v_i^{(q)} = R_{q \leftarrow t}v_i^{(t)}
对 LD polyline

每个历史 lane point 都做同一刚体变换：

xy 旋转和平移；
heading 旋转；
longitudinal scalar、宽度、限速不变；
lat_offset 不能沿用旧值，应在变换后根据当前几何重算，或不将其作为历史绝对特征。

当前代码已经对 OD/LD 历史几何做了一部分 SE(2) 补偿，但审阅发现 LD 的 lat_offset 没有随坐标变换重算，所以现有补偿仍不完整。

Imagine Bank

Imagine 帧必须维护 predicted ego pose：

T^k+1=T^k⊕ΔT(ak)\hat T_{k+1} = \hat T_k \oplus \Delta T(a_k)

再将所有历史 real/imagine record 统一变换到 rollout 当前的 predicted ego frame。

这恰好让 WM 显式 ego-conditioned：

ego dynamics 负责推进 predicted ego pose；
View Builder 负责解析坐标变换；
WM 只预测实体自身的 residual dynamics；
不再迫使 latent 学“自车向前，所有道路向后移动”。
五、stable ID 应如何对齐 slot

这里最关键的修正是：

ST-GNN 内部不应该依赖跨帧固定 slot。stable ID 应用于建 temporal edge，而不是先把所有历史强行重排成当前 slot。

当前实现是把历史 OD/LD 按 stable ID 对齐到当前帧 slot 顺序。这会导致当前帧已经死亡或离开视野的实体直接消失，无法显式表达 death；当前代码确实只保留当前 slot universe，历史中存在但当前不存在的实体会被丢弃。

推荐方案：ragged entity records → packed nodes
Bank 内部

每帧实体列表 independently packed：

frame t-2: [id=17, id=42]
frame t-1: [id=17, id=42, id=73]
frame t:   [id=17, id=73]

Graph Builder

根据 stable ID 建 temporal edges：

17(t-2) → 17(t-1) → 17(t)
42(t-2) → 42(t-1)          death before t
                    73(t-1) → 73(t)   birth at t-1


无需让 42 必须占用当前帧的某个 slot。

GPU batch

构图后再临时 packed：

node_x      [total_nodes, H]
node_type   [total_nodes]
frame_idx   [total_nodes]
entity_id   [total_nodes]
batch_idx   [total_nodes]
edge_index  [2, total_edges]
edge_type   [total_edges]
edge_attr   [total_edges, E]
node_valid  [total_nodes]


如果不希望引入 PyTorch Geometric，也可以使用定长上限：

[B, T, Nmax, H]


但 slot 只在每帧内部有效，跨帧对齐由 entity_id 与 temporal-edge index 完成，而不是要求同一实体永远占同一 slot。

六、hist_valid、per-entity mask、birth/death 怎么处理
1. 不再保留单一含混的 hist_valid

建议拆成几个不同层次的 mask。

Frame valid

表示该帧是否真实存在：

frame_valid[B,T]


用途：

episode 启动不足 6 帧；
reset 后历史不足；
时间戳异常；
缺帧。

不能再用“复制最早帧但 hist_valid=0”来伪造历史内容。Bank 中不存在就是不存在，View Builder 仅在 batch padding 时补零。

Entity present

表示实体在该帧是否被 observation 观测：

entity_present[B,T,N]


它不同于：

数据 padding；
生命周期 birth/death；
occluded；
out-of-range；
WM 未预测。
Node valid

表示某个 packed/padded node 是否可以参与计算：

node_valid = frame_valid & entity_present & schema_valid

Source mask
source ∈ {REAL, IMAGINED}


最好作为 categorical embedding，而不是仅 Boolean mask。

Rollout depth
depth = 0, 1, ..., 6


Real frame 可统一 depth=0，Imagine frame 使用 1 至 6。

2. Birth mask

对实体 ii 在时刻 tt：

birthi,t=presenti,t∧¬presenti,t−1birth_{i,t} = present_{i,t} \land \neg present_{i,t-1}

但要区分两种情况：

真正新生

实体新进入仿真或新生成。

Observation birth

实体原本存在，只是刚进入 sensor/crop 范围。

对网络而言都可使用 birth=1，但若环境提供 actor lifetime，建议追加：

birth_reason =
    SPAWNED
  | ENTERED_RANGE
  | REAPPEARED
  | UNKNOWN


第一版不必过度复杂，至少提供：

is_birth
age_since_observed
has_temporal_parent
Birth 节点处理

没有 temporal parent 的节点：

不接收 temporal incoming message；
使用 learned birth embedding；
仍参与当前帧 spatial message passing；
WM target 中不应强迫 current-slot predictor 预测一个完全未知的新实体。
3. Death mask

Death 不能作为当前帧节点的普通 mask，因为实体在当前帧已经不存在。

有两种处理：

Planner ST-GNN

一般不需要创建 ghost death node：

历史实体节点仍保留在 Bank；
它没有通往当前帧的 temporal continuation；
当前 planner 只 readout 当前 active nodes；
历史 death 信息可通过最后一次历史节点间接影响，但不要让死实体作为当前障碍物。
WM 训练

需要显式 existence head：

palivet+1=fexist(zit,context,at)p_{alive}^{t+1} = f_{exist}(z_i^t,\text{context},a_t)

对当前存在实体预测：

survive；
disappear/death/out-of-range。

Birth 则不能完全从 current-slot autoregression 生成，至少第一阶段应：

只监督 matched survivors；
existence loss 监督 disappearance；
unmatched target births 单独统计；
暂不要求 WM 生成未知新实体。

否则一开始就做变长集合生成，会把任务复杂度推得过高。

4. Readout mask

ST-GNN 最终保留所有历史节点做 message passing，但 planner/value head 只 readout：

当前查询时刻；
active；
valid；
对 Imagine rollout 则是当前 imagined depth 的 active nodes。
readout_mask =
    node.frame == query_frame
    and node.present
    and node.valid


因此，历史节点提供时序信息，却不会被误当成当前障碍物。

七、ST-GNN 第一版别做得过重

基于约 100 万参数目标，我建议第一版不是完整 dense space-time Transformer，而是“两阶段、共享权重”的 typed GNN。

Stage A：Temporal message passing

只连接 same-ID temporal edges：

OD_i(t-k) → OD_i(t)
LD_i(t-k) → LD_i(t)
Ego(t-k)  → Ego(t)


edge attributes：

Δt\Delta t
compensated relative pose；
geometry delta；
birth/age；
source/depth。

不要先把历史 latent 求平均或 pool 成单 token。让每条 temporal edge 单独产生 message，再 attention 聚合到当前节点。

Stage B：Current-frame spatial message passing

只在 query frame 建边：

OD 与 OD
距离阈值；
front/rear；
TTC 或相对速度；
top-K nearest。
LD 与 LD
predecessor/successor；
left/right adjacency；
same road；
merge/split/crossing。
OD 与 LD
vehicle-to-associated-lane；
nearby lane candidates；
lateral/longitudinal relation。
Ego/Nav
Ego ↔ nearby OD；
Ego ↔ candidate lanes；
Nav ↔ route-related LD；
不输入外部推荐车道。
Stage C：readout
Ego token 作为全局 decision token；
learned OD/Lane query 可保留；
Plan/Value/Risk Head 暂时尽量不改。

这样能够形成：

Bank → Temporal typed messages → Spatial typed messages → Heads


同时也自然成为未来 WM 的 backbone。

八、它如何成为 ego-conditioned latent WM 的雏形

ST-GNN 与 WM 不应是两个完全独立的大网络。

建议共享：

Bank schema；
Bank View Builder；
geometry encoder；
edge construction；
一部分 ST-GNN message functions。

然后分两个输出：

Planner state encoder

以当前节点为 readout：

z_t = STGNN(RealBankView_t)


输出给：

Plan Head；
Value/Risk Head。
World Model transition

以相同图为基础，显式注入动作：

Δz_dynamic, alive_logit =
    WMHead(z_t, spatial_context_t, action_t, ego_delta_t)


其中：

Ego 由解析动力学推进；
LD/Nav 由 SE(2) 重投影和 route progress 更新；
dynamic OD 才由 WM 预测 residual；
Static OD/LD 尽量不走 learned transition；
生成下一 imagined FrameRecord；
append 到 Imagine Bank；
下一步再重新构图并通过 ST-GNN。

这比当前把 OD/LD/Nav/Ego latent 一起输入同一个 residual transition 更符合职责边界。当前 WM 是按照当前 slot 顺序预测各模态 latent，还没有 Imagine Bank、ego dynamics 和真正的 Plan-conditioned rollout。

九、需要防止的两个新坑
1. 不要把 Bank 写成 latent cache

如果 Encoder 正在训练，而 Bank 里保存跨 batch 的旧 latent，就会出现：

前几帧来自旧版 encoder；
当前帧来自新版 encoder；
ST-GNN 同时读取多个表征基底。

因此：

IL/DAgger 训练

更稳妥的方案是：

dataset/episode buffer 保存结构化 frame records；
在一个训练 sample 内统一经过当前 encoder；
Bank 是逻辑 Bank，不一定缓存训练时 latent。
Online rollout 或 Encoder 冻结后

可以缓存 detached latent。

也就是说：

“不要 obs.stack_history”不等于“必须永久缓存旧 latent”。

接口是 Bank，训练实现可 lazily encode，在线推理可 cached encode。

2. 不要让 ST-GNN 一次取代所有模块后直接大训

替换 TemporalPool 是较大结构变更，必须先做三层验证：

Bank 层
reset；
FIFO；
timestamp；
source/depth；
no writeback；
episode 隔离。
View 层
纯平移；
纯旋转；
平移+旋转；
same-ID temporal edge；
birth/death；
slot permutation invariant；
padding invariant。
Model 层
entity isolation；
permutation equivariance；
masked nodes 无贡献；
action perturbation sensitivity；
imagined depth 可区分；
ST-GNN 单帧退化与原空间模型行为可比较。
十、我建议现在调整最近三轮 Task
下一步：Bank Contract + Bank View，不训练
删除网络对 stack_history 的依赖；
实现 Real/Imagine Bank 的统一 record schema；
实现 append/reset/FIFO/snapshot/copy-on-write；
实现 SE(2) View Builder；
ragged entity → packed graph；
stable-ID temporal edges；
frame/entity/source/depth/birth masks；
完整单元测试和可视化 probe。

暂时保留旧网络作为对照，但不再扩展 TemporalPool。

下两步：ST-GNN 替换 TemporalPool，保留原 Head
删除 TemporalPool 主路径；
实现 temporal same-ID typed message passing；
实现 query-frame spatial graph；
保留原 Encoder、Plan Head、Value/Risk Head；
单帧模式作为 T=1 自然退化；
在同一 Bank API 下比较：
T=1；
T=6；
temporal-only；
temporal + spatial。

这里需要比较的是分别训练的模型，不能只在同一 checkpoint 上 mask 历史。

下三步：以 ST-GNN 为共享 backbone 接入最小 WM
Ego dynamics 显式推进；
LD/Nav 解析重投影；
dynamic OD residual transition；
existence/death head；
一步 predicted frame append 到 Imagine Bank；
再由同一 ST-GNN 读取；
先做 one-step D1；
D1 过门后才做 6-step；
不联合 Plan Head，不启用 PPO。
最终建议

我赞成将架构主线正式修订为：

Structured current observation → Real Memory Bank → Bank View Builder → ST-GNN → Planner/Value
 Real Bank snapshot → Imagine Memory Bank → ego-conditioned WM append → ST-GNN recurrent rollout

并明确废弃：

env.obs.stack_history 作为正式历史接口；
TemporalPool；
将历史强制对齐到当前 slot 后再做时序；
只靠单个 hist_valid 描述所有有效性；
learned WM 同时预测 Ego、LD、Nav 和 dynamic OD；
Imagine rollout 重复查询同一上下文。

你提到的 concern 是成立的，但不是 ST-GNN 难以落地的理由。正确解法是：

Bank 保存事实；View Builder 负责坐标与对齐；Graph Builder 负责关系；ST-GNN 负责学习；WM 只负责不可解析的动态转移。

这样各模块职责清晰，也正好避免之前 latent 中 ego 运动、地图几何、实体语义彼此串扰的问题。