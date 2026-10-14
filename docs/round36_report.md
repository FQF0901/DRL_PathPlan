# Round-36 总报告（执行优先准备：H1 骨架 + H2 数据规格 + H3 hooks）

> 依据：`docs/deepseekv4p1_argue.md` §5.52（CG 终止回退 `acd22c1`；直接执行 `7c0186f` 执行优先）。
> 纪律：生产改动默认关 + 逐位兼容（Gate 0 8/8）；只读数据；零训练；966 tests passed。
> 产物：`net/path_head.py`（已入库 `6944436`）、`runs/round36/{h1,h2_*,h3_*}.*`；脚本 `.slim/deepwork/p1_v2/round36/h{1,2,3}_*.py`。

## 1. H1 执行头骨架（Gate 0 8/8；smoke 3/3）

- **横向**（`PathExecutionHead`）：车道中心线锚（3 控制站位 κ_ref）+ **near/mid/far 曲率残差**（tanh×0.1、末层零初始化 ⇒ 初始 κ=锚曲率）；κ(s) 分段线性 → 精确积分。
- **纵向**：κ_ctrl + 外源 8 维 → `ds=10·sigmoid`（初始 3.5 m/步）→ 6 步 progress_s。
- **组合层**：`plan[:,0]==action_mu`（逐位）、`dθ_k=κ̄_k·ds_k`、`traj` 重放、截断+掩码；总参数 33,433。
- **Gate 0**：默认关三口径（full/cheap/semantic）HEAD **逐位 max\|Δ\|=0**；旧 ckpt strict **0/0/0**；LQR 契约（interp ≤3.2e-6、κ 积分 6e-8）；**966 tests passed**；smoke：梯度仅 allowlist（18 可训 / 164 冻结）、dropout 契约。
- **Caveats**：曲率空间锚定（路径从自车原点出发，偏移由残差头学）；dθ 截断（掩码+原始值）；启用后 rollout 尾段由组合层一次给出（不再逐步策略自回归）；`train.py` flags 未接线（等价路径=manifest `python_reference_core`）。

## 2. H2 given-target 监督数据规格 + 数据门

- **标签规格**（逐决策步）：T1 keep（907/907 ep，53,606 步）/ T2 change（2,579 ep，11,818 步：L 7,508+R 4,310）/ T3 merge（441 ep，24,746 步）/ T4 gate（455 ep，闸口可得 **317 ep=69.7%**，15,543 步；gate gap 3.5m、匹配 p50 0.084m）/ T5 junction（906 ep，42,022 步）。
- **专家路径在目标车道锚系下表达**：`lat_err`（左正 m）/`head_err`（rad）/`s_anchor`/`valid`（≤5m）。
- **数据门**：全库 **265,801 usable 行、99.98% 有标签、0 error**；自证 15/15 与 F3 一致（455/441/906/907）；字段坐标系/单位/mask 文档化。
- **数据计划更新**：新增 **DP-6（离线 target 标签 + 锚系残差，纯建图重放）**；**DP-2 不再是 T1–T4 前置**；DP-1/DP-4 仍用于覆盖评估。
- **关键 caveat**：**T4 专家多为晚换道**（换道步距闸口 p50 25.8m）⇒ 直接模仿会编码"晚换道"；G4"提前换道 ≥85%"需训练侧单独处理（课程/加权/首步约束）。

## 3. H3 hooks 扩展（555 键；默认关 bit-exact）

- §五 7 类全覆盖：①输入契约（复用 F2）②横向（κ_nmf/κ(s)/位姿/目标车道横·航向误差/边界余量/曲率变化率）③纵向（s/v/ds/accel/静态·动态距/TTC/计划停车距离/可通行长度）④**footprint 连续 ≤1m 插值（非 6 点）**+9 点+env 中心三口径（首越界/首碰撞/最小余量/道路内比例；含执行侧）⑤LQR 对照（+区间饱和比/跟踪误差/**规划误差→`plan_ok/track_ok` 四象限**；样例 60 双好/0 路径错/**2 路径好但跟踪错**）⑥网络性状 + dropout 状态 ⑦干预 8 臂。
- 自证：默认关三方逐位（base≡f2off≡h3off）、开关只增不改、前向 max\|Δ\|=0、RNG/ckpt sha 不变。
- Caveat：H1 新头本 run 未启用（K1 无新键）⇒ head 条件键 n/a（合成模块验证管线可用）；mirror 网络等变率 0（与 R31 弱响应一致）。

## 4. 下一步

1. **DP-6 全量提取（I1 开工）**：T1–T4 数据集（含锚系残差 + 纵向外源几何）。
2. 训练接线（T1→T2→T3→T4）与 G1–G6 评测。
3. 待 CG 裁项：①G1 普通道路子集口径（专用 normal 集）；②T4"晚换道"处理策略（课程/加权/首步约束）。
