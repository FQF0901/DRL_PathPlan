# f523339 实现审计报告（对照 arch_lock @ 94d271e）

> **审计对象**：`f5233393ff3f5369d72c9d106319f57fdbae5247`（HEAD）
> **合同基线**：`docs/arch_lock.md`，提交 `94d271e2ca8b4c495b9166941de198426dec8b53`；两处文件内容一致（`git diff 94d271e..HEAD -- docs/arch_lock.md` 为空）。
> **审计方式**：6 个只读审计 lane 分别覆盖 env/schema、dataset/expert/split、net/action/WM、train/eval pipeline、证据 artifact、repo 结构；本报告只采用能落到代码行、测试或 artifact 字段的结论。
> **证据标记**：【复现】本次实际运行验证；【代码】代码路径直接可见；【artifact】本地 run 文件可查；【推断】由代码路径推导，未做闭环复现。
> **限制**：未重跑 500 场景闭环、未重新训练；全量单测在 `tools/venv-python` 下为 `100 passed, 1 skipped`。`runs/`、`datasets/`、`env/specs/` 均被 gitignore，因此 artifact 只能在本机核对，clone 后不可复现。
> **本文件替代旧版内容**：旧版是 2026-10-20 对 `deepseekv4p1_argue.md` 的答复，阶段已过时，已全部清理。

---

## 0. 结论摘要

1. **P0：`TemporalPool` 的 `[B,T,N,H] -> [B*N,T,H]` reshape 缺少 `permute`。** OD/LD 的“时序池化”实际把不同实体、不同帧的 token 混在一起；per-entity stable-ID history 在进入 Plan Head 之前已经被打乱。用“每个实体恒定值 1..16”的最小用例复现，输出首 8 维为 `[-1.43, -1.28, -0.05, ...]`，最大误差 17.24，而不是各实体常量。该 bug 从 `c8c4ce2` 起一直存在，当前所有 checkpoint（包括 r3b）都带着它训练。**因此当前 Val500 67.6%、history 1/2/4/6、WM D1/D2 都不能作为“合同所述 per-entity temporal fusion”的证据；修好后必须重训、重测。**
2. **P0（证据链）：r3 的 exact/LQR artifact 早于 realized-action 修复。** `runs/val500_clean_r3_p0fixed` 的 500/500 条 `realized_action=null` 且 config 无 `split_manifest`；`runs/lqr50_clean_r3_p0fixed` 的 50/50 同样为 null。它们不是 HEAD `--strict-checkpoint`/manifest 口径的产物，却被 `implementation_status.md` 用作 “p0fixed” 表并与 r3b 比较。r3b artifact（500/500 有 realized、有 manifest）可用，但 r3↔r3b 是跨代码版本比较，不能当作同口径 A/B。
3. **P1：WM 不是合同定义的 latent WM。** `train_wm.py` 的 target 是同一个冻结 planner encoder 的输出，不是独立 EMA Target Encoder；target 是 6 帧 temporal-pooled latent，和 current 共享 5/6 历史；`net/world_model.py` 对 static LD、Nav、Ego 都做 learned transition，违反 §7.3/§9.2。D1/D2 只能算 smoke，不是 gate。
4. **P1：动作执行主链正确，但可行性门仍有洞。** `action=[0.1,0.05,0,0], v0=0` 时 `curvature_checked=False`，`max|κ|=24` 仍被判 `feasible=True`；`action=[0.6,0,0,1], v0=1` 接受后 `max speed=1.288 > max(v0,v1)=1`。这会让不可行/超速段进入训练和评测。
5. **P1：DAgger/BC 数据语义有实质错误。** DAgger 的 `label_valid` 要求 `n_substeps==5`，把“学生段中碰撞/到点后未跑满 5 步”的完整 expert lift 标签也置 0；BC 的 `terminal_failed` 0.2 权重又被 `label_valid=0` 乘成 0。最需要恢复的终止状态被丢弃，失败 expert 降权机制基本失效。
6. **P1：评测指标与口径不足。** eval 只用 `spec.geometry[0]` 做分桶，而主标签在 `spec.labels["geometry"]`；Val500 实测 468/500 不一致，路口/匝道/收费站多被记成 straight。exact 在 `env.step` 之后才 `tracker.apply`，collision/off-road/arrival 的 `info` 来自 teleport 前的物理步；exact 的 `tracking_error_xy≈2.5e-6 m` 是 round-trip，不是控制跟踪。`timeout` 在 trunc 路径仍是 `False`，`stuck` 与 timeout 大量重叠，`comfort` 是 plan-side，`feasible` 是 episode-level bool。
7. **P1：工程合同没有落地。** 全仓库没有 `import yaml`；`config/` 只有 5/12 个文件且均未被读取；checkpoint 没有 optimizer/RNG/stage/global step/split hash/encoder version/git dirty；测试依赖 gitignored `env/specs`，clean clone 不可跑；`env/obs.py` 1,444 行、`pipeline/train_ppo.py` 1,663 行、`net/action_space.py` 1,371 行，69 处 broad except，3 套 `safe_brake_action`/`resolve_action` mirror 已经漂移。
8. **总体判断：这是一个“可运行的研究原型 + 局部好设计”，不是合同意义上的规范/解耦/高内聚实现。** 好的局部设计包括 PSTL 4D 动作层、`TruncatedNormal`、checkpoint strict load、scene-seed split/manifest、`pipeline/execution.py` 的方向；但 P0 时序 bug、证据链混版本、WM 语义偏离、DAgger 数据错误和工程合同缺失，决定了当前不能把任何 gate/最终结论当作已通过。

---

## 1. 审计基线与方法

| 项 | 事实 |
|---|---|
| HEAD | `f5233393ff3f5369d72c9d106319f57fdbae5247` |
| 合同 | `docs/arch_lock.md` @ `94d271e`，正文 1,924 行；HEAD 未改合同正文 |
| tracked 文件 | 71 个；其中 Python 19,667 行 |
| 测试 | 16 个 test 文件、101 个 collected；`tools/venv-python -m pytest -q` = **100 passed, 1 skipped** |
| 数据/运行 artifact | `datasets/` 2.1 GiB、`runs/` 143 MiB；两者均 gitignore |
| specs | `env/specs/` gitignore；`test_obs_schema.py`、`test_splits.py`、`test_lqr_smoke.py` 直接依赖 |
| 合同批注 | `arch_lock.md` 的【】批注只更新到 `639e958`，与 HEAD 大面积不一致（文件数、测试数、cost 进 policy、Test500 split 等均过期） |

**审计 lane**：

1. `env/obs.py`、`env/metadrive_env.py`、`env/tracking.py`：schema/几何/ID/history、Exact/LQR。
2. `dataset/*`、`env/expert/*`、`env/scenario/*`、`config/splits/*`：采集、审计、DAgger、split。
3. `net/model.py`、`net/action_space.py`、`net/world_model.py`：Plan Head、PSTL、WM。
4. `pipeline/train_bc.py`、`pipeline/train_ppo.py`、`pipeline/execution.py`、`checkpointing.py`：训练/PPO/执行/checkpoint。
5. `pipeline/eval.py`、`eval_metrics.py`、`select_checkpoint.py`、`probe_wm.py`、`train_wm.py` + `runs/`：评测/证据链。
6. 全 repo：目录/文件/函数归属、重复、依赖方向、配置、测试、文档、死代码。

---

## 2. P0：`TemporalPool` 维度错位（最先修）

### 2.1 代码事实

`net/model.py:104-118`：

```python
def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
    # tokens [B,T,N,H], valid [B,T,N]
    B, T, N, Hd = tokens.shape
    x = tokens.reshape(B * N, T, Hd)          # <-- 缺 permute(0,2,1,3)
    q = x[:, -1:, :]
    ...
    mask = ~(valid.reshape(B * N, T) > 0.5)   # <-- 同样错位
```

`tokens` 在 `encode_entities()` 中由 `[B*T,N,H]` reshape 成 `[B,T,N,H]`（`net/model.py:274-277`），是 `[B,T,N,H]` 连续布局。直接把 `B` 和 `N` 合并，等价于把“时间轴”和“实体轴”交叉重排：

```text
x[b*N+n, t] = tokens.reshape 后的第 (b*N+n)*T+t 个元素
             = tokens[b, t_orig, n_orig]  (t_orig*N + n_orig = n*T+t)
```

只有 `N=1` 的 Nav/Ego 不受影响；`OD_SLOTS=16`、`LD_SLOTS=16` 的 OD/LD 全部错位。

### 2.2 最小复现【复现】

```python
import torch
from net.model import TemporalPool
B,T,N,H=1,6,16,8
tokens=torch.zeros(B,T,N,H)
for n in range(N):
    tokens[:,:,n,:]=float(n+1)   # 每个实体在全部 6 帧都是常量 n+1
valid=torch.ones(B,T,N)
pool=TemporalPool(hidden=H,heads=4,frames=T).eval()
out=pool(tokens,valid)
```

预期：`out[0,n]` 应为常量 `n+1`。实际首 8 维：

```text
[-1.43, -1.28, -0.05, -1.35, -1.26, -0.38, -1.31, -1.24]
max abs err vs 1..16 = 17.24
```

这说明每个输出 slot 看到的是混合后的别的实体/帧，而不是自己的 6 帧历史。

### 2.3 影响

- `od_t`/`ld_t` 进入 `plan` 的 6 个 latent 中的两个（`net/model.py:377-379`），也进入 spatial token（`351-356`）和 query attention（`367-372`）。**stable-ID 对齐在最后一步被破坏。**
- `encode_entities()` 的 `od`/`ld` 也是 WM 的 current/target 输入（`train_wm.py:141-174`），因此 WM 的“per-entity target”同样没有按实体对齐。
- `hist_valid`/`od_mask`/`ld_mask` 的 mask 走同一个错误 reshape，history 消融不是“按帧屏蔽”，而是“按错误切片屏蔽”。`implementation_status.md` 中 “history 1/2/4/6 = 16/38/52/68%” 只能说明“mask 掉了一些 token 会掉点”，不能证明 per-entity history 的因果收益。
- 该代码从 `c8c4ce2`（2026-10-20）存在至今；`runs/bc_2k_dagger_r3b/best.pt`（2026-10-21 23:52）也是在这个 bug 下训练的。**当前所有 checkpoint 都不是合同架构的合格 checkpoint。**

### 2.4 根因与修复

- 根因：`reshape` 被当成 `permute+reshape` 使用；测试只查 shape/finite（`test/test_model_shapes.py:60-110`），没有“每实体常量历史 -> 每实体常量输出”的语义测试。
- 修复：

```python
x = tokens.permute(0, 2, 1, 3).reshape(B * N, T, Hd)
mask = ~(valid.permute(0, 2, 1).reshape(B * N, T) > 0.5)
```

- 必须补的单测：
  1. 每个实体历史为常量 `c_n` 时，输出保持 `c_n`；
  2. 交换实体顺序，输出按同一顺序交换（equivariance）；
  3. `hist_valid` 只屏蔽某一帧/某一实体时，其他实体输出不变；
  4. `B>1`、`N>1` 同时成立时逐实体对齐。
- 修复后**必须重训**；不能把旧 checkpoint 直接加载到修正后的模型上继续评估（shape 相同但语义不同）。

---

## 3. 超出 arch_lock scope 的改动

> 判定口径：合同明确禁止/未授权的机制为“明确超出”；合同未要求但属于工程实现细节的为“设计债/灰区”；阶段顺序越级单独标注。

| # | 项目 | 证据 | 与合同关系 | 判定 |
|---|---|---|---|---|
| S1 | WM 对 static LD、Nav、Ego 做 learned transition | `net/world_model.py:57-60,120-125`；`pipeline/train_wm.py:203-208` | §7.3：只有 dynamic OD/RoadState 进入 learned transition；Static LD 用 geometry reprojection + re-encode；Nav 按 route progress 更新；Ego 由 ego dynamics 推进 | **明确违反** |
| S2 | WM D1/D2 在 Stage C 之前启动 | `net/world_model.py`、`pipeline/train_wm.py`、`pipeline/probe_wm.py` 存在；`docs/decision_log.md` 记录 owner progression policy | §12.1/§13/§20：真实 RL 先于 imagined RL；P4 在 P3 后 | **流程越级**；已记录，但不能算 gate pass |
| S3 | 用规则 TTC/urgent/gap 监督 `q_od`，用 role=0 监督 `q_lane` | `pipeline/train_bc.py:653-708`；`net/model.py:326-335,367-388` | §3.1/§3.5/§22.9：不允许以规则形式直接告诉 Planner 哪辆车需要避让；环境语义经 Encoder，不得规则旁路 | **灰区偏违反**：规则没有直接进 plan 输入，但塑造了 plan 的 query context |
| S4 | DAgger 采集使用 `--shield-ttc` 规则强制 expert replay | `dataset/collect_dagger.py:580-590,790-793,1323-1334` | §11.3 只要求 student 访问状态上的 expert 查询；未授权 TTC shield | **设计债/灰区**：只用于采集，未用于 eval/PPO；但改变 student 状态分布，且未写 decision log |
| S5 | `lane_lateral_info()` 在 `agent.lane` 缺失时读 `navigation.current_ref_lanes[0]` | `env/metadrive_env.py:583-625` | §3.3 严禁 lane-level oracle；该函数当前无调用者 | **潜在泄漏**：当前不影响 Student；若未来接到 reward `lane_center` 项会引入 lane-level 参考 |
| S6 | PPO 额外 `info_cost`、`w_shield=3`、`w_student_failure=2`、`source_weight_lift=0.5`、`w_infeasible=0.2` 等 | `pipeline/train_ppo.py:178-185`；`pipeline/train_bc.py:206-221,425-497` | 合同未列这些量；§23 要求开放项写 decision log | **设计债**：可解释性不足，未记录候选/回滚 |
| S7 | `--history-frames` 只做推理 mask，不是训练 ablation | `pipeline/eval.py:191,266,548`；`pipeline/execution.py:109-129` | §13 A2/§19 要求同 seed 重训 1/2/4/6 帧 | **降级实现**：不能替代训练 ablation |
| S8 | `pipeline/rollout.py:78-96` 的 6 步 preview 对同一 batch 重复查询 | `pipeline/rollout.py:78-96`；无生产调用 | §2.2/§6.2 要求自回归 + t0 累计 | **误导性 API**：已标 PLACEHOLDER，但不应保留在可调用主路径 |

**明确未越界**：无视觉 Backbone、无 K-sampling/多候选、无生成式/Diffusion WM、无额外动作头；Nav 当前帧不读 `get_checkpoints/current_ref_lanes/next_ref_lanes/final_lane`（`env/obs.py:10-17`，`test_obs_schema.py:153-170` 有 monkeypatch 测试）。

---

## 4. 合同计划但 HEAD 仍未完成

| 条款 | 合同要求 | HEAD 状态 | 证据 |
|---|---|---|---|
| §5.1 Real Memory Bank | 6 帧 per-entity token record、detach、FIFO、reset、per-entity transform metadata | **缺失**；只有 `stack_history` 张量 | `env/obs.py:1253-1444`；全仓库无 MemoryBank 类 |
| §5.2 Imagine Memory Bank | 从 Real Bank 复制、FIFO append、source/depth、禁止回写 | **缺失** | 无类/无 rollout |
| §4.1-4.2 content latent / transform metadata / delta / Δt | content latent、geometry anchor、transform metadata 分离；current+historical+residual+geometry delta+Δt+mask | **缺失**；只有 temporal attention | `net/model.py:249-305` 无 content/geometry 拆分 |
| §4.3 birth/death mask | 区分 current/historical、birth、death、padding | **缺失**；只有 presence mask | `env/obs.py:1227-1249`；历史 ID 被当前 ID 覆盖（`1327/1348/1373`） |
| §3.2 boundary polyline / lane type / traffic light / RoadState | boundary polyline+ID、lane type、dynamic road-state token | **缺失/部分** | `env/obs.py:263-268` 只有中心线点与 `left_type/right_type`；无 RoadState |
| §6.2 六步 t0 累计 | 6 个 0.5 s 点转换到 rollout-t0 frame | **缺失** | `pipeline/rollout.py:78-96` 同一 batch 重复；`eval.py:527` 明写 NOT_IMPLEMENTED |
| §6.3/§7.1 完整 3 s LQR | 6 点插值 0.1 s、LQR 预瞄完整 3 s | **缺失**；只执行 0.5 s/5 点 | `pipeline/eval.py:350-364`；`env/tracking.py` 可接收更长参考但未接入 |
| §7.2 WM condition | `(dx,dy,v_n,v_{n+1},dpsi,kappa,a_lon)` | **弱**：只把 4D action MLP 加到 token | `net/world_model.py:96-100`；`train_wm.py:175-183` |
| §7.4 uncertainty head | OOD/rollout depth 误差/截断 | **缺失** | `net/world_model.py` 只有 od/ld/nav/ego 四个 head |
| §9.2/§10.3 EMA Target Encoder | 独立 target、stop-grad、EMA、防坍缩 | **缺失**；同一个冻结 planner encoder | `train_wm.py:253-257`；`target_drift=0.0` 硬编码 |
| §9.4 防坍缩 | EMA target、variance/rank monitor、decoder/aux task | **缺失** | 无 variance/rank/decoder |
| §10.2 YAML detach 开关 | 全套可配置 detach/EMA | **缺失**；无 YAML loader | 全仓库无 `import yaml` |
| §11.3 failure-prioritized buffer | 失败优先采样、覆盖率触发、逐轮混合 | **缺失**；只有训练后置权重 | `train_bc.py:475-480`；无 priority sampler |
| §11.2 multi_solution / expert audit | 多解降权、逐样本特权审计、`expert_audit.md` | **缺失** | `multi_solution` 字段不存在；`privileged_expert_sample` 恒 1 |
| §13 A2 1/2/4/6 训练 ablation | 同 seed 重训四档 | **缺失**；只有推理 mask | `pipeline/execution.py:109-129` |
| §13 D1 数据覆盖/OOD | Expert+DAgger+PPO+恢复扰动；`(dx,dy,dyaw,a,kappa)` 覆盖报告 | **缺失** | `runs/wm_d1_final` 只用 train_bc_2k + dagger_r1/r2 |
| §13 D2 课程/三种实验 | 1→2→3→6 课程；GT/GT、pred/GT、pred/Plan 三种 | **缺失**；只有 recorded-action 自回归 probe | `pipeline/probe_wm.py` |
| §14 Test500/多 seed/分场景桶/mean time-to-failure | 只读 Test500、多 seed、11 类桶、失败时间/距离 | **部分/未完成** | Test500 代码路径存在但未运行；桶只用 `geometry[0]`；无多 seed |
| §14.4 闭环 checkpoint 选择 | 安全门槛→success→RC→comfort/efficiency，Val500 | **降级**：exact-only、50 scene、阈值硬编码 | `pipeline/select_checkpoint.py:41-42,49,59` |
| §17/§18/§23 工程合同 | 目录树、YAML、resolved config、provenance、tools/docs/tests、decision log | **大部分缺失** | 见 §8 |

---

## 5. 已实现但与合同有偏差

### 5.1 数据/schema/坐标

| 偏差 | 证据 | 影响 |
|---|---|---|
| OD 类型过滤掉行人、TrafficBarrier、TrafficWarning | `env/obs.py:771` 只收 `BaseVehicle/TrafficCone/BaseBuilding` | 行人和部分障碍物对 Planner 不可见，安全相关 |
| `StaticDefaultVehicle` 被标为 dynamic | `env/obs.py:94-105` 对 `BaseVehicle` 返回 `static_flag=0`；`_od_taxonomy` 对 `BaseVehicle` 固定 `mobility=1`（`480-494`） | 静态车被当可动实体，影响避让和 WM transition |
| `od_age` 只是 6 帧窗口内累计 | `_build_od` 写 0（`env/obs.py:795`）；`stack_history` 从最老帧重新累计（`1406-1416`），当前帧最多 `5*0.1=0.5 s`，归一化后 0.167 | 常驻实体的“年龄”永远不超过 0.5 s，不是真实 observation age |
| 历史帧的 `od_ids/ld_ids` 被当前 slot ID 覆盖 | `env/obs.py:1327,1348,1373` | 无法恢复“这一帧是哪个实体死了/新生”，birth/death/WM ID 对齐失效 |
| LD stable ID 不含 chunk，但 docstring 说含 | 代码 `env/obs.py:882` 只 hash lane pair + lane_id；模块 docstring `26-28` 说 `ld_chunk_s` 是 ID 一部分 | 文档与实现矛盾；跨 20 m chunk 时同一 ID 的采样点会移动 |
| 历史 LD 的 `lat_offset` 未补偿 | `_compensate_ld_point` 只改 dx/dy/heading（`env/obs.py:1183-1189`）；`lat_offset` 在 `872` 是历史 ego 的横向偏移 | 补偿后的 lane 点与 `lat_offset` 不自洽 |
| `_lane_candidates` 先 clip 再判断 `s0 < -15` | `env/obs.py:163-169`：`s0 = clip(s0,0,len)` 后 `s0 < -15` 永假 | 后方远处 lane 可能进入 LD slots，挤掉前方 lane |
| `nav_t` 未乘当前 `nav_mask` | `net/model.py:283,377-379`：spatial token 用 mask，但 plan 输入直接用 `nav_t` | 当前 Nav 无效、历史 Nav 有效时，stale Nav 进入 Plan/Value |
| `od_mask/ld_mask` 缺省为全 1 | `net/model.py:214-218` | 调用方省略 mask 时 padding 被当 valid（当前 batch 都传，风险在 API） |

### 5.2 动作/轨迹/PSTL

| 偏差 | 证据 | 影响 |
|---|---|---|
| 短路径 κ 门漏洞 | `net/action_space.py:849-853`：`length<0.25` 且 `v0=v1=0` 且 `dpsi=0` 时 `curvature_checked=False`；`1025-1037` 跳过 κ/a_lat 检查。【复现】`[0.1,0.05,0,0], v0=0` -> `feasible=True, max|κ|=24` | 不可行短横移进入训练/评测 |
| 速度 overshoot | `net/action_space.py:803-818,858-875` 只检查 `s_dot>=0`，没有 `speeds<=max(v0,v1)` 或单调上界。【复现】`[0.6,0,0,1], v0=1` -> `max speed=1.288` | 速度先升后降，LQR/舒适性变差 |
| `gate()` 对负 `v_next` 抛 ValueError | `net/action_space.py:719-720` | 调用方若期望“返回 rejected”，会崩溃；PPO/DAgger 的 fallback 路径没有统一捕获 |
| LQR `(N,3)` 参考把 0.5 s 间距当 0.1 s | `env/tracking.py:639-643` 用 `chords / (1/reference_hz)` | 未来 3 s/0.5 s 参考若用 3 列，速度会放大约 5 倍 |
| `tracking.py` 仍保留 2D `(ds,dtheta)` 策略入口 | `env/tracking.py:1-8,73-124,631` | 当前生产用 4D，但未来 3 s wiring 可能误用；与 §22.5 单一动作语义冲突 |
| exact 的 steering 是 PSTL κ 合成，不是 expert 实测 | `env/tracking.py:430-433,459-463`；BC 数据是 expert steering | exact 训练/评测的 steer/a_lat 分布与 BC 不一致；不能把 exact 误差直接解释为 Planner 质量 |

### 5.3 World Model

| 偏差 | 证据 | 影响 |
|---|---|---|
| target 不是独立 Target Encoder | `train_wm.py:253-257` 加载同一个 planner；`_encode_rows` 直接 `encode_entities`（`53-80`）；无 EMA | D1/D2 固定 target 可解释，但 Stage E/§22.11 不满足 |
| target 是 6 帧 pooled latent | `train_wm.py:141-174`；`encode_entities` 返回 temporal pool | current 与 target 共享 5/6 历史，action 条件信号被平滑；D1 不是真正 one-step |
| static LD/Nav/Ego 做 learned transition | `net/world_model.py:57-60,120-125`；`train_wm.py:203-208` | 违反 §7.3；地图可能漂移/幻觉 |
| WM loss 等权相加 | `train_wm.py:203-209` `total=sum(losses.values())` | 没有 §9.2 的 λ_od/λ_roadstate/λ_ego/λ_aux |
| D1 val 空时回退 train | `train_wm.py:260-264` | 小数据/`val_frac=0` 时在训练 scene 上选模；`len(val_idx)==0` 检查不可达 |
| val_loss 与 persistence 估计量不一致 | `train_wm.py:290-301`：val_loss 是 batch-mean 平均，persistence 是 sample-weighted | 0.616 vs 1.051 不是同一估计量；比较有偏 |
| probe 不校验 encoder/data | `pipeline/probe_wm.py:58-99` | 用错 encoder/data 仍静默出数 |
| “held-out”只对 WM 成立 | `runs/wm_d1_final` 的 200 个 val seeds 中 90 个在 planner `train_seeds`、110 个在 planner `val_seeds`，0 个 unseen | 冻结 encoder 见过全部 WM val 场景；WM 泛化证据被高估 |
| D2 不是训练课程 | `probe_wm.py` 只做 one-step WM 自回归 6 步 | 无 1/2/3/6 课程、无 GT/Plan action 对照、无 latent rank gate |

### 5.4 训练/RL

| 偏差 | 证据 | 影响 |
|---|---|---|
| DAgger `label_valid` 把完整 lift 标签判 0 | `collect_dagger.py:402-407` 要求 `n_substeps==5`；`892-898` 用该值 | 学生段中碰撞/到点后 `n_substeps<5`，完整 expert lift 标签被丢弃；Stage B 最需要的终止恢复样本缺失 |
| `terminal_failed` 0.2 权重被 `label_valid` 乘成 0 | `collect.py:390,526`；`train_bc.py:410-431`：先 `w*=0.2`，再 `w*=label_valid` | 失败 expert 末段被硬删，而不是 0.2 降权；失败前段仍 1.0 |
| `expert_plan_valid` 语义漂移 | collect=success；dagger=label_valid（`collect_dagger.py:951`） | 两套数据不能按同一列做 expert 失败分析 |
| `privileged_expert_sample` 恒 1 | `collect.py:329-335`；`collect_dagger.py:567-573` | `w_privileged` 结构性无效；§11.2 审计/降权未落地 |
| `multi_solution` 不存在 | `collect.py`/`collect_dagger.py` dtype 无该字段 | 多解场景仍单轨迹 L2，违反 §11.2 |
| BC val 是随机 scene split，不是 Val500 | `train_bc.py:1068,1158-1159`；无 `--split-manifest` | `best.pt` 在内部 val 上选，不是冻结 Val500；A1/A2 对比不可审计 |
| `action_in_bounds` 硬 0 | `train_bc.py:431` | OOB 标签被删除，不是降权；与 §18.3 “OOB 0.07-0.3% 未用于权重”的旧批注不同 |
| PPO reward/cost 量级失衡 | `train_ppo.py:105-108` success 20、collision cost 1；`399-411`；`cost_coef=0.1` | collision 对 policy 的有效权重比 success 弱约 200 倍；安全信号被 success 淹没 |
| PPO fallback 被 mask 出 policy 与 cost | `train_ppo.py:668` `policy_valid = valid & ~masked`；`1080,1148-1153` | 策略对“采样到不可行动作”没有直接梯度；fallback 率不会自动下降 |
| PPO cost 项用未裁剪 ratio | `train_ppo.py:1143-1153`：reward surrogate clip，cost 项 `cost_ratio*cost_adv` 未 clip | cost 梯度可随 ratio 放大；无 cost limit/对偶 |
| PPO 不强制 split manifest | `train_ppo.py:1452-1456` 只 `load_specs(args.specs)` | 可在 train specs 上跑；Stage C 的冻结 val 结论不可审计 |
| PPO 无 BC/DAgger aux loss | `train_ppo.py:1151-1168` 只有 PPO + value + cost | §13 C 要求保留 BC/DAgger auxiliary/replay |
| PPO 无 reward/cost hacking audit | 全仓库无代码/报告 | §12.3 |

### 5.5 评测

| 偏差 | 证据 | 影响 |
|---|---|---|
| 分桶用 `spec.geometry[0]` | `pipeline/eval.py:484`；主标签在 `spec.labels["geometry"]` | Val500 468/500 不一致；路口/匝道/收费站多被记成 straight/curve，§14.3 结论不可用 |
| exact 在 `env.step` 后 `tracker.apply` | `eval.py:372-378`；`tracking.py:435-468` | `info` 的 crash/out_of_road/arrive_dest 来自 teleport 前物理步；可能滞后/漏检，exact 闭环成绩的解释力下降 |
| exact `tracking_error_xy` 是 round-trip | `eval.py:377-384` 用 realized vs 同一 reference；r3b 约 `2.45e-6 m` | 不是控制跟踪能力；与 LQR 的 preview error 不可比 |
| `timeout` 在 trunc 路径为 False | `eval.py:440-444,476-479`：`termination="timeout"`，但 `timeout` 只在 `if not done` 置 True | timeout rate 低估 |
| `stuck` 与 timeout 重叠 | `eval.py:64-66,424-430`；r3b 23 个 stuck 中 22 个同时 timeout | stuck rate 不是独立失败模式 |
| `comfort` 是 plan-side | `eval.py:339` `tr_exec.comfortable` | 不是 realized comfort；r3b 99.6% 只是规划侧指标 |
| `feasible` 是 episode-level bool | `eval.py:499` `infeasible_count==0` | 与 `infeasible_count` 的 per-episode 平均数并存；r3b 93.6% feasible 但平均 2.64 次 infeasible |
| `wrong_route` 是 RC 下降代理 | `eval.py:67-69,446-448`；Val500 全 0 | 未验证 wrong-way/wrong-road；不能当作真实违规率 |
| `--strict-checkpoint` 是 opt-in | `eval.py:547,567`；默认 `strict=False` | gate 运行忘记 flag 时 legacy/不兼容 checkpoint 只 warning |
| `--allow-unfrozen-split`/`--allow-final-test` 可绕过 | `eval.py:543-544,562-565` | 治理 loophole；gate 结论可被手动放宽 |
| selector 50 scene/exact-only/硬编码阈值 | `select_checkpoint.py:41-42,49,59` | 0.20/0.20 未读 `config/eval_thresholds.yaml`；50 scene 的 r3b−r3 +10 pp（CI −2..+22），Val500 仅 +1.4 pp（CI −1.8..+4.8） |
| selector 不清理旧 `selected.pt` | `select_checkpoint.py:70-84` | 无 eligible 时 `selection.json` 为 null，但旧 `selected.pt` 仍可被下游加载 |
| `selected.pt` 元数据仍是 open-loop 说明 | `runs/selected_checkpoint50/selected.pt` config | 与“闭环选择”矛盾；无阈值/原因/CI/hash |

---

## 6. 代码逻辑问题（按严重度）

> 表中“影响”含推断；触发条件与代码路径为事实。P0=当前结论/后续训练前必须修；P1=Stage E/F 或对外结论前必须修；P2=可后置。

| # | 级别 | file:line | 触发 | 影响 | 根因/建议 |
|---|---|---|---|---|---|
| 1 | **P0** | `net/model.py:104-118` | 任意 `B>1` 或 `N>1` 的 OD/LD | temporal pool 混实体/混帧；plan、WM target、history ablation 全部失真 | 缺 `permute(0,2,1,3)`/`permute(0,2,1)`；修复后重训 |
| 2 | **P0** | `pipeline/eval.py:372-378` | exact 每个 0.1 s 子步 | `info` 来自 teleport 前物理步；collision/off-road/arrival/RC 可能滞后/漏检 | `env.step` 后 `tracker.apply`，info 未重算；apply 后重算或读取下一 step |
| 3 | **P0** | `runs/val500_clean_r3_p0fixed`、`runs/lqr50_clean_r3_p0fixed` | 作为 “p0fixed” 证据 | 500/500、50/50 `realized_action=null`，无 manifest；与 HEAD 口径不一致 | artifact 生成于 a639959 之前；不要与 r3b 同表比较 |
| 4 | **P1** | `dataset/collect_dagger.py:402-407,892-898`；`pipeline/train_bc.py:426` | 学生段中终止（`n_substeps<5`） | 完整 expert lift 标签被判 `label_valid=0`；恢复样本丢失 | `label_valid` 混淆“标签完整性”和“学生执行步数”；拆分 `segment_complete` |
| 5 | **P1** | `dataset/collect.py:390,526`；`pipeline/train_bc.py:410-431` | 失败发生在段中 | `terminal_failed` 的 0.2 被 `label_valid=0` 乘成 0；失败段被删而非降权 | 先乘 0.2 再硬乘 label_valid；顺序/语义错误 |
| 6 | **P1** | `net/model.py:283,377-379` | 当前 nav 无效、历史 nav 有效 | stale Nav 进入 plan | `nav_t` 未乘 `nav_mask`；显式置零并加单测 |
| 7 | **P1** | `net/action_space.py:849-853,1025-1037` | `v0=v1=0`、短横移、`dpsi=0` | `max|κ|=24` 仍 feasible | `curvature_checked` 只看 length/v/dpsi；对所有 `chord>eps` 检查 κ |
| 8 | **P1** | `net/action_space.py:803-818,858-875` | `v_next` 大于 `v0` 但路径短 | 速度 overshoot（1.288 > 1） | 只检查 `s_dot>=0`；加 `speeds<=max(v0,v1)`/限幅 |
| 9 | **P1** | `net/world_model.py:57-60,120-125`；`train_wm.py:203-208` | 任意 WM 训练 | static LD/Nav/Ego 被 learned transition；地图漂移 | 未实现 §7.3 的 static/dynamic 路由 |
| 10 | **P1** | `train_wm.py:253-257`；`53-80` | 任意 WM | target 用同一冻结 planner encoder，不是独立 EMA target | 无 Target Encoder；D1 之后 Stage E 不可用 |
| 11 | **P1** | `train_wm.py:260-264` | `val_frac=0`/数据不足 | `val_idx=train_idx`，在训练 scene 上选模 | 空 val 静默回退；应 raise |
| 12 | **P1** | `train_wm.py:290-301` | val 最后 batch 小于 batch_size | val_loss 与 persistence 估计量不一致 | batch-mean 平均 vs sample-weighted |
| 13 | **P1** | `pipeline/train_ppo.py:105-108,399-411,1151-1153` | collision/off-road | success +20 vs collision cost 1×0.1；安全信号弱约 200 倍 | 量级未对齐；无 cost limit/对偶 |
| 14 | **P1** | `pipeline/train_ppo.py:668,1080,1148-1153` | 采样到不可行动作 | fallback 被 mask 出 policy/cost；策略无直接梯度降低 fallback 率 | 只用 mask；应加 feasibility-aware 分布或未 mask 的辅助惩罚 |
| 15 | **P1** | `pipeline/train_ppo.py:1143-1153` | cost_adv>0 且 ratio 大 | cost 项未裁剪，梯度可放大 | reward surrogate clip，cost surrogate 没 clip |
| 16 | **P1** | `pipeline/eval.py:484` | 任意 Val500/Test500 | 468/500 场景分桶错误；关键场景退化不可检测 | 用 `spec.labels["geometry"]` |
| 17 | **P1** | `env/obs.py:163-169` | ego 在 lane 起点后方 | `s0 < -15` 永假；后方 lane 占满 slots | 先保留 raw s 做 gate，再 clip |
| 18 | **P1** | `env/obs.py:94-105,480-494` | `StaticDefaultVehicle`/静止车辆 | static 标记 0、mobility 1；静态车被当动态 | 按类/速度/可动性分类 |
| 19 | **P1** | `env/obs.py:1327,1348,1373` | 任意 history | 历史 ID 被当前 slot ID 覆盖；birth/death 无法恢复 | 设计牺牲 per-frame ID；额外保存 `*_ids_src` |
| 20 | **P1** | `env/obs.py:795,1406-1416` | 常驻实体 | od_age 最大 0.5 s，不是真实 age | 窗口内累计；按 episode first_seen 维护 |
| 21 | **P1** | `env/obs.py:1183-1189,872` | 历史帧有横移/旋转 | `lat_offset` 未补偿 | 只补偿 dx/dy/heading；重算或置零 |
| 22 | **P1** | `train_bc.py:1068,1158-1159` | BC 训练 | 内部随机 val，不是冻结 Val500；`best.pt` 不可审计 | 无 `--split-manifest`；接入 manifest |
| 23 | **P1** | `pipeline/train_ppo.py:1452-1456` | PPO | 可在 train specs 上跑，无 val 强制 | 无 manifest 校验 |
| 24 | **P1** | `pipeline/select_checkpoint.py:41-42,49,59` | 选模 | 阈值硬编码、exact-only、50 scene | 未读 `eval_thresholds.yaml` |
| 25 | **P2** | `pipeline/eval.py:440-444,476-479` | `trunc=True` | `termination="timeout"` 但 `timeout=False` | 两套来源；统一 |
| 26 | **P2** | `pipeline/eval.py:462,524` | `--max-steps 0` | `realized_action` 未绑定，UnboundLocalError | 循环外初始化 |
| 27 | **P2** | `pipeline/train_ppo.py:220-233` | 连续低速后恢复 | `stuck_run` 永不复位；停车避障被持续惩罚 | sticky flag；按段/恢复复位 |
| 28 | **P2** | `pipeline/eval_metrics.py:87-94,139-146` | legacy 缺 `stuck` | 缺失当 False；伪“显著退化” | 缺失用 NaN/sentinel |
| 29 | **P2** | `pipeline/checkpointing.py:75-81` | `action_low/high` 长度≠4/None | 裸 ValueError/TypeError | 先校验 shape/finite |
| 30 | **P2** | `env/tracking.py:667` | 每次 LQR act | `float(array)` deprecation；NumPy 2 会报错 | `.item()` 或 `gain[0] @ error` |
| 31 | **P2** | `train_wm.py:93-98` | 不同 `--seed` | `max_pairs` 子采样永远 RNG 0 | 用 `args.seed`/分层采样 |
| 32 | **P2** | `probe_wm.py:58-99` | encoder/data 与 WM config 不一致 | 静默用错 encoder/data | 校验 config/encoder hash |
| 33 | **P2** | `net/action_space.py:719-720` | `v_next<0` | `gate` 抛 ValueError，不是 rejected | 先 bounds 检查/捕获 |
| 34 | **P2** | `dataset/collect.py:792-793`；`collect_dagger.py:1432-1433` | 写 npz 被 kill/磁盘满 | 截断文件 | temp+rename |

---

## 7. 证据链审计

### 7.1 artifact 与代码版本

| claim | artifact | 判定 |
|---|---|---|
| r3 exact 66.2/12.2/8.6 | `runs/val500_clean_r3_p0fixed/metrics.json` | **artifact 过期**：500/500 `realized_action=null`，无 `split_manifest`；mtime 早于 `a639959` 的 realized/split 修复；不能称 “p0fixed” |
| r3b exact 67.6/15.6/7.4/stuck 4.6 | `runs/val500_clean_r3b_p0fixed/metrics.json` | **有证据**：500 unique seeds、realized 全有、manifest；但模型带 TemporalPool bug；单 seed |
| r3 LQR 34/2/12/52 | `runs/lqr50_clean_r3_p0fixed/metrics.json` | **artifact 过期**：50/50 `realized_action=null`，无 manifest |
| r3b LQR 48/6/10/36 | `runs/lqr50_clean_r3b_canary/metrics.json` | **有证据（canary）**：50 scene；stuck 16%；feasible 48%、fallback 30%——含大量 postprocess，不是纯 LQR 跟踪 |
| r3b−legacy success +2.6 pp (CI −1.2..+5.8) | 两侧 500 seeds | **数值可复算**；但 legacy 是旧 schema/旧 checkpoint，无 stuck/timeout/comfort，不是同 harness |
| Expert 73.0/14.4/6.6/6.0 | `/tmp/opencode/expert_val500/expert_bc.npz` + `runs/expert_val500/metrics.json` | **四数有证据**；`stuck=0` 是 `collect.py:576-577` 硬编码，不是测量 |
| PPO 20 episodes/399 transitions | `runs/ppo_v8_canary/contract.json` | **有证据（wiring）**：logp diff 0、mask 31.6%、postprocess 30.1%；无性能 claim |
| WM D1 0.616 vs persistence 1.051 | `runs/wm_d1_final/metrics.json` | **窄 smoke**：内部 10% scene split、四项未加权和、val_loss 估计量不一致；90/200 WM val scenes 在 planner train |
| WM D2 depth6 od 0.522 vs 0.673 | `runs/wm_d2_probe_final2/metrics.json` | **有证据（diagnostic）**：真 AR、recorded action；不是 1/2/3/6 课程，不是 gate |
| history 1/2/4/6 = 16/38/52/68% | `runs/hist50_k1/k2/k4` + selector | **数值有证据**：同一 checkpoint mask `hist_valid`；但 TemporalPool bug 使“per-entity history”解释无效；不是训练 ablation |
| selector r3b | `runs/selected_checkpoint50/selection.json` | **有证据（50-scene）**：r3b eligible 68/16/10；r3 58/12/10；但 50 scene/exact-only/阈值未记录；`selected.pt` metadata 仍写 open-loop |
| Test500 未跑 | `config/splits/test500.json`、`env/specs/scenarios_test500.json` | **正确**：500 seeds 5,000,500..5,000,999，全 `split=test`，未运行；旧批注 “split=val” 已过时 |
| 100 passed/1 skipped | pytest | **有证据**（wrapper 下） |

### 7.2 文档漂移

- `docs/implementation_status.md` 的 `runs 69 MiB` 实际 `143M`；`PPO v8 batch 1.1–2.6e-5` 实际 update max 1.7e-5..5.05e-5；`r4/r6` 只有 pid/log，没有结构化 metrics 目录。
- `docs/implementation_status.md:191` 说 “Train/evaluate only through the frozen manifest”，但 `train_bc` 没有 `--split-manifest`，`train_ppo` 完全没校验；只有 `eval`/`select_checkpoint` 有。
- `docs/arch_lock.md` 的【】批注：`58 tracked files` vs 实际 71；`82 passed` vs 100；`无 docs/decision_log.md` vs 文件存在；`cost 不进 policy`、`Test500 split=val`、`prev_action 仍是 requested`、`short-path κ 漏洞仍在` 等多处已被 HEAD 推翻。
- `docs/deepseekv4p1_argue.md` 仍含 “Test500 不是真正 test split” 等旧结论；`docs/chatgpt5p6sol_review.md` 旧版是 P0 前答复。两份都应标 superseded/归档，避免被当成当前事实。

**根因**：合同正文与“审计批注”混在同一文件，批注手动更新、无 CI/日期门禁；运行 evidence 不入库，文档数字靠人工抄写。建议把 `arch_lock.md` 拆成“规范正文”和单独的 `implementation_status.md`/`decision_log.md`，每次交付用脚本生成状态矩阵。

---

## 8. 整体 repo：规范/解耦/内聚评估

### 8.1 结论

**不合格（partial），不是“规范、解耦、高内聚”的合同实现。** 目录级分层（`env/`、`dataset/`、`net/`、`pipeline/`、`test/`）是合理的；但文件级职责、依赖方向、重复实现、配置真源和测试 hermeticity 都不达标。它是“能跑的研究原型 + 若干好局部设计”。

### 8.2 文件/函数规模

| 文件 | 行数 | 主要职责 | 问题 |
|---|---:|---|---|
| `pipeline/train_ppo.py` | 1,663 | rollout/buffer/reward/GAE/PPO/CLI/checkpoint | 6 类职责；`collect_episode` 279 行、`main` 224 行 |
| `dataset/collect_dagger.py` | 1,593 | student 推理/expert replay/shield/label/audit/CLI/worker | 与 `collect.py` 大段复制；`run_episode` 288 行、`main` 283 行 |
| `env/obs.py` | 1,444 | schema/几何/stable ID/Nav/LD/history | 合同要求的 `observation_adapter/schema/id_alignment/coordinate_transform/sequence_builder` 全塞一处 |
| `net/action_space.py` | 1,371 | bounds/归一化/PSTL/feasibility/gate/project | `action_to_trajectory` 302 行；`resolve_action` 又在 eval/PPO/DAgger 三份 |
| `pipeline/train_bc.py` | 1,288 | 数据加载/target/权重/loss/val/checkpoint/CLI | `main` 272 行；`_plan_weights` 是隐式数据政策引擎 |
| `env/scenario/behaviors.py` | 1,088 | 场景事件/行为 | 可后置拆分 |
| `dataset/collect.py` | 911 | expert 采集 | 与 dagger 9 个函数完全相同 |
| `env/expert/pure_pursuit_idm.py` | 884 | IDM+PP/规则选道 | 与 MetaDrive 内部 API 强耦合 |
| `env/tracking.py` | 838 | 2D legacy + Exact + LQR + DARE | 模块头仍定义 2D 策略动作 |
| `pipeline/eval.py` | 657 | rollout/fallback/metrics/CLI | `run_episode` 347 行、61 分支、嵌套 7 |

全仓库有 22 个 >100 行函数；`env/scenario/spec.py:128 validate` 分支约 88；`env/obs.py:928 _build_nav` 分支约 48；`pipeline/train_ppo.py:772 collect_episode` 分支约 37。

### 8.3 重复实现与漂移

- `dataset/collect.py` 与 `collect_dagger.py`：9 个函数源码完全相同（`_wrap/_expert_class/_safe_float/_lift_action/_pack_rows/_rate/_audit_summary/_git_commit/_limits_dict`），`_audit_fields`/`_label_diagnostics` 约 95–99% 相似；`_backfill_outcomes`、`done`、`terminal_failed`、`expert_plan_valid` 已出现语义分叉。
- `safe_brake_action` 3 份（`pipeline/eval.py:127`、`pipeline/train_ppo.py:264`、`dataset/collect_dagger.py:289`）；`_trajectory` 3 份；`_batch_from_history` 3 份；`_safe_float` 5 份；`_limits_dict` 4 份。
- `resolve_action` 3 份（`train_ppo.py:275`、`eval.py` 内联 `286-335`、`collect_dagger.py:316`），已经漂移：eval 多负速度后检、DAgger 多一次 bounds clip、PPO 多 `INVALID_ACTION` code。
- `prev_action` 4 份：eval/PPO 用 measured realized；collect 用 measured action；DAgger 用 `executed_action`（command）。`pipeline/execution.py:65` 的 `DecisionState` 只在 `test/test_execution_parity.py` 使用，生产零引用——所谓“单一真源”没有落地。
- `pipeline/rollout.py` 仅测试引用；`env/tracking.py` 2D legacy 无生产调用；`env/metadrive_env.py:583 lane_lateral_info` 无调用；`dataset/splits.py:64 unique_seed_count`、`env/tracking.py:153 roundtrip_error` 无调用。

### 8.4 依赖方向

- `net/model.py:59` `from env.obs import ...`：模型层依赖环境层，`net` 不能独立测试/部署。
- `dataset/collect_dagger.py:72-73` import `net.model` + `pipeline.checkpointing`；`pipeline/train_bc.py:64` import `dataset.collect`：形成 `dataset ↔ pipeline` 包级循环。
- 建议：把 schema/常量/geometry 放到 neutral `common/`（或 `env/schema.py`），checkpoint loader 移到 `common/checkpointing.py`；`dataset` 不反向依赖 `pipeline`。

### 8.5 配置与可复现性

- 全仓库 `import yaml`/`yaml.safe_load` = **0**；`config/` 只有 5/12 个文件，`runtime_local.yaml`、`eval_thresholds.yaml`、`model.yaml`、`env.yaml` 均无读取点。
- `config/train_bc.yaml:2` 指向不存在的 `datasets/bc_2k`；实际 canonical 是 `datasets/train_bc_2k`。
- checkpoint 只保存 `model/config/epoch/metrics`（BC）或 `model/config/training`（PPO），没有 optimizer/scheduler/scaler、EMA、stage、global step、RNG、split hash、encoder version、git commit/dirty、依赖版本。
- `runs/`、`datasets/`、`env/specs/` 被 gitignore；`runs/expert_val500/metrics.json` 的 `source` 指向 `/tmp/opencode/...`；clean clone 后证据链断裂。

### 8.6 测试

- 100 passed/1 skipped，但测试以 shape/finite/单函数为主；`test_model_shapes.py` 不可能发现 TemporalPool 语义错位；`test_world_model.py` 只查 shape/grad/rollout 长度/persistence MSE，没有 action sensitivity/static LD/Nav/uncertainty。
- `test_obs_schema.py:36`、`test_splits.py:27`、`test_lqr_smoke.py:48` 依赖 gitignored `env/specs`；`test_eval.py:180` 的闭环 smoke 默认 skip；无 reset/causality/`test_train_eval_assembly_parity`/`test_detach_contract`。
- 无 `pyproject.toml`/lint/type/CI；`tools/venv-python` 假设 `.venv` 存在。

### 8.7 文档/目录

- 合同 §17 的 24 个 `net/` 文件实际只有 4 个；`model.py` 单体替代 encoders/fusion/planner/value。对 1M 参数、405 行的规模，这可算“合理的简化”，但文档不应声称已按合同分层。
- `docs/` 只有 5 个 tracked 文件；缺 `expert_audit.md`、`failure_taxonomy.md`、`training_stages.md`、`evaluation_protocol.md` 等。`decision_log.md` 只有 4 条，§23 的 13 个开放项没有完整“候选/实验/同口径结果/回退条件”。
- `env/expert/README.md`、`env/scenario/README.md` 引用了不存在的 `tools/*.py`/`pipeline/eval_runner.py`/`env/obs/static.py`。

### 8.8 值得保留的好设计

- `net/action_space.py` 的 PSTL 4D 端点/终端速度精确，`test_action_space.py` 对 full-stop/stop-to-go/rotation 有覆盖。
- `pipeline/distributions.py` 的截断 Normal log_prob 数学正确，`test_distributions.py` 有积分测试。
- `pipeline/checkpointing.py` 的 schema/bounds/decision 校验 + `strict=True`，比旧版明显好。
- `dataset/splits.py` 的 scene-seed split/manifest/SHA 校验方向正确；`freeze_splits.py` 已把 test500 设为 `split="test"`。
- `pipeline/execution.py` 的 `relative_action/segment_dt/limit_history` 方向正确，只是没有被四个 producer 真正共用。
- `env/scenario` 的 taxonomy/spec/generator/validator 分层清晰，场景标签可区分。

---

## 9. 对 arch_lock 的状态矩阵（HEAD 校正）

> 只列关键条款；“部分”=有实现但有实质偏差/证据不足。

| 条款 | HEAD 状态 | 说明 |
|---|---|---|
| §3.1 OD schema | 部分 | stable ID/taxonomy 有；行人/barrier 缺失；static vehicle 误分类；od_age 窗口内 |
| §3.2 LD schema | 部分 | centerline/road type/limit 有；boundary polyline/lane type/traffic light 缺；chunk doc/code 矛盾 |
| §3.3 Road-level Nav | 基本满足 | current frame 无 lane oracle；历史单位已修；`road_geometry` 全 lane 均值是边界情况 |
| §3.4 Ego | 部分 | v_long/v_lat/yaw/yaw_rate 有；wheelbase 无；prev_action 多为 realized，DAgger 仍 command |
| §4.1-4.2 补偿/ID/content | **不合格** | TemporalPool P0；content latent/delta/Δt 无；LD lat_offset 未补偿；Nav 历史单位正确 |
| §4.3 birth/death | 缺失 | 只有 presence mask；历史 ID 被当前 ID 覆盖 |
| §5.1-5.2 Memory Bank | 缺失 | 无 Real/Imagine Bank |
| §6.1 4D Plan Head | 部分 | 主链 4D/PSTL；短路径 κ、速度 overshoot、3 s 无 |
| §6.2 六步 t0 | 缺失 | `rollout.py` 重复同一 batch |
| §6.3 插值/LQR | 部分 | 0.5 s/5 点；3 s 未接；3 列参考速度有 5× 风险 |
| §6.4 动作语义 | 部分 | 生产 4D；tracking 2D legacy 仍活；DAgger prev_action 不一致 |
| §6.5 PPO 概率动作 | 部分 | logp 绑定正确；cost/fallback/limit 不满足 |
| §7.1 Ego Dynamics/LQR | 部分 | PSTL 推进；无独立 Ego Dynamics；LQR 只 0.5 s |
| §7.2 WM condition | 部分 | 4D action 有；缺 v_n/κ/a_lon/ego dynamics |
| §7.3 WM 输出拆分 | **违反** | static LD/Nav/Ego 做 learned transition |
| §7.4 uncertainty | 缺失 | 无 |
| §8 参数预算 | 部分 | 总 1,053,142；模块分配偏差大；WM 167,952（若计入 deployable，合计 1,221,094 > 1.2M） |
| §9.1 Plan loss | 部分 | 多分项；κ/jerk 代理；无 3 s |
| §9.2 WM loss | 部分/违反 | 四项等权；无 Target Encoder；无分模态/深度/距离/静态动态统计 |
| §9.3 Value/Risk | 部分 | 分头；value 仅 episode outcome；risk episode 广播 |
| §9.4 防坍缩 | 缺失 | 无 EMA/variance/rank/decoder |
| §10 detach | 部分 | 无 YAML 开关；PPO/BC 硬编码 detach |
| §11.1 Expert | 部分 | collector 有；wrong_route/stuck 硬编码 0 |
| §11.2 特权审计 | 部分/缺失 | 字段常量；multi_solution 无；单轨迹 L2 |
| §11.3 DAgger | 部分 | 3 轮；无 failure buffer/逐轮混合；label_valid bug |
| §12 RL | 部分 | PPO canary；cost 量级/fallback/limit 不满足；无 hacking audit |
| §13 Stage A1/A2/B/C | 部分/ provisional | 6 帧提前；A2 训练 ablation 无；B/C 数据/门禁未闭 |
| §13 D1/D2 | 部分/ smoke | WM 有；Target/static/depth 课程/uncertainty 不满足；且越级 |
| §13 E/F | 未开始 | 无 joint training/3 s LQR |
| §14.1 数据划分 | 部分 | Val500/Test500 manifest 有；train 不强制；Test500 未跑 |
| §14.2 指标 | 部分 | success/collision/off-road/RC 有；stuck/comfort/feasible/wrong_route 语义不足；mean time 无 |
| §14.3 场景桶 | 不合格 | 只用 `geometry[0]`，468/500 错 |
| §14.4 checkpoint 选择 | 部分 | selector 存在；50 scene/exact-only/硬编码 |
| §15 probes | 大部分缺失 | 无 latent/causality/gradient/WM 可视化 |
| §16 failure taxonomy | 缺失 | 无代码/统计 |
| §17 目录树 | 部分 | env/dataset/net/pipeline 有替代；细粒度文件缺失 |
| §18 工程约定 | 部分/不合格 | YAML/provenance/日志/测试门槛/fallback 计数不满足 |
| §19 ablation | 大部分缺失 | 只有 exact vs LQR 0.5 s、history 推理 mask |
| §20 优先级 | 部分 | P0/P1 多项未闭；P4 提前 |
| §22 硬约束 | 1 满足/9 部分/6 未实现 | 见上 |
| §24 证据链 | 未闭合 | 见 §7 |

---

## 10. 修复优先级

### P0（在任何继续训练/对外结论前）

1. **修 `TemporalPool`**：`permute` + 语义测试；**重训所有 checkpoint**，重跑 Val500/LQR。不要复用旧 checkpoint。
2. **重建证据链**：删除/标记 `runs/val500_clean_r3_p0fixed`、`runs/lqr50_clean_r3_p0fixed` 的 “p0fixed” 用法；所有 gate artifact 记录 checkpoint SHA、git commit/dirty、split SHA、代码版本；`runs` 至少保留可重建 manifest（不必入 git，但要有 checksum）。
3. **修 exact 评测**：`tracker.apply` 后重算 collision/off-road/arrival/RC，或读取下一 step 的 post-apply info；明确 exact `tracking_error_xy` 是 round-trip，不作为控制跟踪。
4. **修 DAgger/BC 数据语义**：拆分 `label_valid` 与 `segment_complete`；完整 lift 在 `n_substeps<5` 时仍有效；`terminal_failed` 0.2 应在 `label_valid` 之后仍能生效（或按 fatal termination 单独处理）。
5. **修动作可行性门**：短路径零速横移检查 κ；速度上界/单调；`gate` 不抛裸异常。

### P1（Stage E/F、正式 LQR、最终验收前）

1. WM：独立 Target Encoder/EMA/stop-grad；static LD/Nav/Ego 改 reprojection/re-encode/dynamics；D1 分模态/深度/距离/静态动态统计；D1 数据加入 PPO/恢复扰动；空 val raise。
2. PPO：reward/cost 量级对齐；cost limit/对偶或 clip；fallback 的 policy 梯度/可行性惩罚；split manifest 强制；保留 BC/DAgger aux；hacking audit。
3. BC：冻结 Val500 manifest、失败 expert segment-level 权重、`multi_solution`/privileged 审计、checkpoint 闭环选择。
4. Eval：`spec.labels["geometry"]` 分桶；timeout/trunc 统一；stuck/comfort/feasible/wrong_route 的定义和分母；多 seed；Test500 只读 gate。
5. Engineering：YAML loader + resolved config；checkpoint provenance；`env/specs` track 小 fixture；CI/lint；broad except 计数；把 `resolve_action`/`safe_brake`/`_batch_from_history` 收敛到 `pipeline/execution.py`，四个 producer 共用。

### P2（可后置）

- 拆 `env/obs.py`、`env/tracking.py`、`net/model.py`、`pipeline/train_ppo.py`、`pipeline/train_bc.py`、`dataset/collect_dagger.py`。
- 删除/隔离 2D legacy、`rollout.plan_actions`、`lane_lateral_info`、`DecisionState` 死代码。
- 补 §17 tools/docs、§23 decision log、failure taxonomy。
- 归档 `deepseekv4p1_argue.md`/旧 review，给 `arch_lock.md` 批注加日期/失效标记。

---

## 11. 附录：本次实际核查

### 11.1 命令/复现

```bash
git rev-parse HEAD
git diff --stat 94d271e..HEAD
tools/venv-python -m pytest -q          # 100 passed, 1 skipped
tools/venv-python -m net.param_probe     # deployable_total 1,053,142
```

TemporalPool 复现：

```python
# 每实体常量 1..16，T=6,N=16,H=8
out = TemporalPool(hidden=8, heads=4, frames=6).eval()(tokens, valid)
# 预期 out[0,n] == n+1；实际首 8 维:
# [-1.43, -1.28, -0.05, -1.35, -1.26, -0.38, -1.31, -1.24]
```

动作 gate 复现：

```python
action_to_trajectory(np.array([0.1, 0.05, 0.0, 0.0]), 0.0)
# feasible=True, max|κ|=24.0, curvature_checked=False
action_to_trajectory(np.array([0.6, 0.0, 0.0, 1.0]), 1.0)
# feasible=True, max speed=1.288 > max(v0,v1)=1
```

### 11.2 artifact 字段核查

- `runs/val500_clean_r3_p0fixed/metrics.json`：500 rows，`realized_action is None` 500；config 无 `split_manifest`。
- `runs/val500_clean_r3b_p0fixed/metrics.json`：500 rows，`realized_action is None` 0；config 有 `split_manifest`。
- `runs/lqr50_clean_r3_p0fixed/metrics.json`：50 rows，null 50；无 manifest。
- `runs/lqr50_clean_r3b_canary/metrics.json`：50 rows，null 0；有 manifest；success 0.48、collision 0.06、off-road 0.10、timeout 0.36、stuck 0.16、comfort 0.98。
- `runs/ppo_v8_canary/contract.json`：20 episodes、399 transitions、mask 0.316、postprocess 0.301。
- `runs/wm_d1_final/metrics.json`：200 val seeds；与 `runs/bc_2k_dagger_r3b/best.pt` 的 `train_seeds/val_seeds` 比较，90 在 train、110 在 val、0 unseen。
- `config/splits/val500.json` / `test500.json`：各 500 seeds，SHA 与 specs 匹配；Test500 全 `split=test`。

### 11.3 未验证/保留意见

- 未重跑 500 场景 exact/LQR；未重新训练；因此不对“修复 TemporalPool 后的性能”做预测。
- exact `info` 滞后对 collision/off-road 的具体影响未做受控复现；代码顺序是事实，影响为推断。
- WM held-out overlap 的结论基于 `runs/wm_d1_final` 与 `best.pt` 的 seed 列表；不同 artifact 可能有不同 split 语义。
- `TemporalPool` 修复后，旧的性能和 history 结论全部需要重跑；本报告不保留旧数字作为架构证据。

---

**最终结论**：`f523339` 不是合同验收状态。最优先的动作不是继续调 PPO/WM，而是修 `TemporalPool`、重训、重建同口径证据链；在此之前，Val500 67.6%、history 1/2/4/6、WM D1/D2 只能作为“原型可运行”的诊断，不能作为合同 gate 或最终结论。
