# net

目的：P2 驾驶策略网络（v2：mem-bank + 注意力聚合 + plan-head MoE + ST-GNN 递归 rollout），
主入口 `net.model.DrivingModel`。契约见 `docs/design-v1.2.md` §2/§3。

## 模块
- `encoders.py`：各模态线性投影到 H + 类型/轨道 id 嵌入 + LayerNorm；mem 帧与 rollout
  合成帧共享权重；OD 9 维 / LD 7 维 / ego 8 维 / others `F_o`（env schema v2 默认 28）。
- `mem.py`：**mem-bank 输入契约**（env schema v2 键名 `*_hist` + `od_id_hist/od_presence_hist`、
  `hist_valid`；others 维 = 28；伴随数组缺失时的回退）、`MemBank.clone()/shift_*` 拷贝隔离、
  `MemEncoder` 编码聚合。
- `temporal.py`：`TemporalAttention`——对 mem 的 T 帧做掩码注意力池化（帧龄嵌入 + 
  `hist_valid`/槽位掩码；全无效列输出严格 0）。
- `spatial.py`：2 层消息传递（OD↔ego、LD↔ego、LD↔LD 相邻、OD↔OD 近邻，边=相对位姿）。
- `plan_head.py`：4 个 mem 聚合 + nav/signal → 融合 + **MoE** → 下一时刻 ego 特征；
  MoE 只在这里（规格第 3 条）。
- `moe.py`：primary 恒激活 + 8 个零初始化残差 expert，**top-2 软混合**（router 取 top-2
  后 softmax；`router_logits(8)` 是训练侧软目标/KL 的监督面）。
- `st_gnn.py`：Spatio-temporal GNN——更新后 ego mem + od/ld mem → t0 帧的下一时刻
  OD/LD 预测（残差解码、零初始化 ⇒ 未训练 = 匀速/静止先验）+ presence/entry logits。
- `model.py`：装配、`arc_step` 运动学单一实现、**递归 rollout（拷贝隔离 + 合成帧 detach +
  action/pose 链可微）**、`SUPERVISED_LABELS`；`encode()` / `plan_step()` 为单步公开入口
  （Stage A 教师强制用 `plan_step` 取 `ego_next` 监督，见 design-v1.2 §2.3 方案①）。
- `param_probe.py`：参数探针 + 前向/rollout 形状冒烟（断言 ≤1.5M）。

## I/O（B 维在前）
- 输入（env schema v2）：`ego_hist(6,1,8)`、`od_hist(6,16,9)` + `od_id_hist/od_presence_hist`、
  `ld_hist(6,16,7)`、`others_hist(6,1,F_o)`、各自 `_mask` 与 `hist_valid(6)`；
  当前帧 `ego/od/ld/others/nav/signal` 可选（作为回退/上下文 token）。
- 输出：`action_mu/action_logstd (2)`、`value (1)`、`traj_xy (6,2)`（t=0.5..3.0 s，t0 自车系）、
  `plan (6,2)`（rollout 实际执行的 6 个动作，`plan[:,0] == action_mu`）、
  `od_pred (6,16,5)`、`ld_pred (6,16,4)`（LD 只作输入/诊断，**无未来 LD 监督**）、
  `od_presence_pred (6,16)`、`od_entry_pred (6,16)`（logits）、`router_logits (8)`、
  `expert_weights (8)`（top-2 分布）、`latent (H)`。
- `forward(..., rollout=False, world_model=False)` 为 PPO cheap path：省略 traj/多步预测键，
  其余输出逐位一致；`world_model=False`（rollout=True）时仍跑 rollout 但不返回预测键。

## rollout 语义（固定，无消融开关）
1. 从真 mem **拷贝 4 份**（`MemBank.clone`）；真 mem 绝不写回（断言测试）。
2. 每步：plan head(mem 副本) → 下一 ego 特征 → 挤入 ego 副本（弹最老帧）→ ST-GNN →
   t0 帧下一 OD/LD 预测 → SE(2) 逆变换到当前累积位姿系 → 挤入各自副本（弹最老帧）；×6。
3. **detach**：真实帧（step0）不 detach（检测任务）；rollout 合成的后续帧在挤入前
   `detach()`（预测任务，状态链不回传）；**action/pose 链保持可微**（6 点轨迹目标仍能
   训练 plan head 的动作链）。`wm_detach` 形参保留但恒为 no-op。
4. **Stage A 教师强制**（`pipeline.stages`）：每步挤入 GT ego 帧（detach）前先调
   `plan_step` 取 `ego_next` 并与目标帧真实 ego 前 6 维做监督 —— plan head/MoE 在 A 阶段
   的唯一梯度来源（design-v1.2 §2.3）。

## 规模与用法
- 参数（`tools/venv-python net/param_probe.py`）：默认 H=96 → **658,879**（v1 净 −7,384）；
  训练配置 H=128/experts 256 → **1,149,663**（v1 净 −25,212），均 ≤1.5M。
- 注意：policy/ST-GNN 输出层零初始化，首个 backward 不会向主干回传（残差零初始化的
  标准行为）；训练几步后恢复。
- 按 config 构建用 `pipeline.stages.build_model`（includes 平铺合并后的根配置）。
