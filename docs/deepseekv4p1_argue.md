> **Historical note (2026-11-07):** 本文件记录 exact execution / stable-ID contract 之前的
> 规划与审阅状态。当前合同见 `docs/arch_lock.md`，当前 baseline 见
> `docs/baseline_exact.md`；文中旧 checkpoint、all-zero fixture fallback、test 数字均已过期。
>
# arch_lock pre-RL review response

> 2026-11-05 update: 当前合同不再使用 Phase 2b/2c 等阶段编号；下文相关段落是当时的审阅记录，实施状态见文末 §7。
>
> 审阅对象：`docs/arch_lock.md`
> 日期：2026-11-03
> 结论：**接受“当前不能直接锁为 pre-RL contract”的主判断**。核心架构与代码基本一致，但文档确实混入了 run-specific 结果、未实现的 gate 和少量 schema/责任表错误；这些必须先修正，再锁合同。

---

## 0. 总判断

审阅对核心架构的肯定部分与我的核对结果一致：

- raw Bank、6 帧 / 16 OD + 16 LD slot；
- TSB 不做 nav、不做 entity geometry projection；
- WM latent residual + direct 5D/10D geometry decoder；
- target-only 新实体 content target 已经接通；
- encoder/TSB/target 在 WM 训练时冻结，当前不启用 EMA；
- policy 无 value/risk head；
- 6 policy / 6 action / 6 transition，不写回 RealBank；
- raw `history_mode=1` loader 拒绝 projected/missing；
- r3 的 61.0% / 18.0% 指标和 checkpoint 可在 run artifacts 中复现；
- DriveModel 参数量 1,270,748，默认 BC 配方与 `config/train_bc.yaml` 一致。

但审阅中有 **1 条事实错误** 和 **1 条部分事实错误**：

1. `pipeline/select_checkpoint.py:52` 当前是 `p.error(...)`，不是 `parser.error(...)`，不存在 `NameError`。该 finding 是旧版本残留。
2. “`train_bc.INPUT_KEYS` 不含 `od_age/od_anchor/od_taxonomy`”不准确：`od_anchor` 和 `od_taxonomy` 都在 `INPUT_KEYS` 中并被 encoder 使用；真正的问题是 `od_age` 是零值死特征、`ld_taxonomy` 缺失。

除这两条外，P0/P1/P2 的其余意见基本接受或部分接受。

### 结论速览

| 编号 | 主题 | 结论 |
|---|---|---|
| P0-1 | `ego_velocity` | 接受 |
| P1-1 | exact tracking / eval fallback | 接受 |
| P1-2 | `model.py` 责任表 | 部分接受：改文档，不搬代码 |
| P1-3 | Phase 2b/2c 命名 | 接受，统一 2c |
| P1-4 | Phase 2c 回滚条件 | 接受，改成 `policy_tol` 实际逻辑 |
| P1-5 | WM gate / target drift / latent norm | 接受，标 TODO/proposed |
| P1-6 | `probe_wm` persistence | 接受，区分 initial-frame / previous-step |
| P1-7 | valid mask / 新实体 slot | 接受 |
| P1-8 | slot fallback | 部分接受，文档标例外并加 production diagnostic |
| P1-9 | §0.1 run-specific 细节 | 接受，移到 `docs/baseline_r3.md` |
| P2-1 | selector 双定义 | 接受，明确 canonical gate |
| P2-2 | `ego_pose` 7 个 | 部分接受/解释：概念 7 个，API 6 个 |
| P2-3 | PolicyHead 结构 | 接受 |
| P2-4 | world frame 表述 | 接受 |
| P2-5 | `rollout_depth` 用途 | 接受 |
| P2-6 | Bank 字段列表 | 接受 |
| P2-7 | `train_bc` env step | 接受 |
| P2-8 | §13 文件表 / §14.1 | 接受 |
| P2-9 | Phase 3/4 状态 | 接受 |
| P2-10 | `--allow-random-val` | 接受，标 legacy debug |
| P2-11 | 默认 ckpt 旧 run | 接受 |
| P2-12 | `select_checkpoint` NameError | **反驳：当前是 `p.error`，不存在该 bug** |
| P2-13 | probe 隔离 transition error | 部分接受，改为“隔离 action 输入” |
| P2-14 | `od_age` / anchor / taxonomy | 部分接受、部分反驳 |

---

## 1. P0

### P0-1 `FrameRecord.ego_velocity` 是文档虚构字段

**结论：接受。**

`net/mem_bank.py` 的 `FrameRecord` 字段是：

```text
episode_id, frame_id, timestamp, ego_pose,
od, ld, source, rollout_depth,
parent_frame_id, action_from_parent, encoder_version
```

没有 `ego_velocity`。全仓库只有 `docs/arch_lock.md` 出现该字段。

处理：

- 从 §2.1 schema 删除 `ego_velocity`；
- 不新增代码字段。当前 `ego_velocity` 不是网络输入，也不是 Bank 合同的一部分。

---

## 2. P1

### P1-1 exact tracking 与 eval/PPO fallback 自相矛盾

**结论：接受。**

文档多处写：

```text
requested_action == executed_action
LQR 后才允许区分
```

但 `pipeline/eval.py` 和 `pipeline/train_ppo.py` 当前就会在 PSTL infeasible 时做：

```text
v_next projection
safe-brake
记录 requested_action / executed_action / fallback code
mask 或重算 PPO log-prob
```

§9.1 自己已经写了这个例外，因此是合同内部矛盾。

处理：统一改为：

```text
BC/DAgger policy 训练路径：exact tracking，requested == executed。
eval/PPO 安全执行层：允许显式 v_next projection / safe-brake；
必须记录 requested/executed，并在 RL loss 中 mask 或重算 log-prob。
```

删除 §0 #14、§9.1 开头、§14 里“当前一律 exact tracking”的绝对表述。

### P1-2 `net/model.py` 责任表与代码不符

**结论：部分接受。接受文档修正；不接受为了这张表把 helper 搬进 `mem_bank.py`。**

当前 `net/model.py` 确实还包含：

- `build_history_window` / `build_history_window_fast`；
- `_sanitize_ids` / `_align_history`；
- `prepare_current_encoding` / `fifo_append_safe` / `align_window_to_current`。

这些是 **raw-history → BankWindow 的 adapter / FIFO-safe alignment**，不是 Bank 存储本身。`mem_bank.py` 负责 schema、FIFO、stable-ID/valid 和 `BankWindow`；如果把这些 helper 搬进 `mem_bank.py`，会让存储层反向依赖 encoder 语义，依赖方向更差。

处理：

- 在 §13 责任表明确：`model.py` 负责顶层组合、forward、rollout，以及 raw-history adapter / FIFO-safe alignment；
- 修正依赖图：

```text
mem_bank.py
  ↑
encoder.py
  ↑
world_model.py
  ↑
model.py
  ↑
build_net.py

planner.py
  ↑
model.py
```

`planner.py` 不依赖 `world_model.py`；`world_model.py` 不依赖 `planner.py`；`build_net.py` 依赖 `model.py`，不是反向。

如果未来确实想让 `model.py` 变薄，可以单独抽出 `pipeline/history_window.py`，但这是重构，不是本次锁合同的必要条件。

### P1-3 Phase 2b / 2c 命名不一致

**结论：接受。统一为 Phase 2c。**

当前不一致：

```text
docs/arch_lock.md          Phase 2c
pipeline/train_joint.py    "Phase 2b"
pipeline/train_joint.py    "joint_phase": "2b"
config/train_joint.yaml    "Phase 2b"
test/test_train_joint.py   "Phase 2b"
```

处理：

- 文档保持 Phase 2a（one-step WM）→ 2b（horizon probe）→ 2c（joint fine-tune）；
- 把 `train_joint.py`、`config/train_joint.yaml`、`test/test_train_joint.py` 的 2b 改为 2c；
- checkpoint 的 `joint_phase` 新值写 `"2c"`；
- 旧 checkpoint 里的 `"2b"` 只作为历史 metadata，不做路径分派。

### P1-4 Phase 2c 的“policy success 回滚”没有实现

**结论：接受。**

`pipeline/train_joint.py` 当前判断是：

```python
policy_budget = policy_tol * abs(initial_policy)
policy_ok = (val_policy - initial_policy) <= policy_budget
if val_wm < best_val and policy_ok:
    ...
```

只比较 validation policy loss，没有 closed-loop success。文档写成“policy success 退化则回滚”是不准确的。

处理：

- 文档改成实际实现：

```text
val policy loss 相对初始值的退化不超过 policy_tol，
且 val WM loss 改善，才接受 checkpoint。
```

- 如果 owner 要 closed-loop success 作为回滚条件，需要单独实现 `train_joint` 的 closed-loop validation；在此之前不能写成已实现合同。

### P1-5 WM gate / target drift / latent norm / collapse 被写成当前行为

**结论：接受。**

实际状态：

- `pipeline/train_wm.py` 只按 `val_loss` 选 best；
- `pipeline/probe_wm.py` 只输出 content/geometry/persistence 均值；
- 没有 target drift、latent norm、latent collapse 指标，也没有 pass/fail gate。

处理：

- §7.6 的“记录 target drift / latent norm / decoder 误差”标为 TODO，或删除；
- §12.6 的 WM gate 保留为 **proposed gate**，明确写“当前未实现，只定义目标”；
- 在实现之前，不能把“必须优于 persistence/zero、不能 collapse”写成当前已执行行为。

### P1-6 `probe_wm` 的 persistence 定义与文档不一致

**结论：接受，部分修正。**

`pipeline/probe_wm.py` 在 depth loop 前捕获：

```python
base_od = window.od_content[:, -1].clone()
base_ld = window.ld_content[:, -1].clone()
```

然后每个 depth 都用 `base_od/base_ld` 作为 persistence baseline。这是 **initial-frame persistence**，不是文档写的 `z_hat_t = z_hat_{t-1}`（previous-step persistence）。

处理：

- 当前指标改名为 `initial_frame_persistence`；
- 另加 `previous_step_persistence`，用上一步 `z_od/z_ld` 作为 baseline；
- 文档分别定义两者，不再混用。

### P1-7 valid mask / 新实体 slot 的表述不准确

**结论：接受。**

当前 `pipeline/train_wm.py` 使用的是：

- `od_tmask/ld_tmask`：content target mask；
- `od_gmask/ld_gmask`：geometry target mask；
- `od_valid = maximum(current_valid, tmask)`；
- `od_current_valid` 区分 target-only slot；
- 新实体 content target 会填入当前帧 free slot；
- 当前 16 slot 全满时，新实体不再监督。

文档写“mask 用 valid”“valid mask 来自当前帧”过于笼统。

处理：

- 改成 target mask 语义：

```text
content 用 content target mask；
geometry 用 geometry target mask；
current valid 只用于区分当前实体和 target-only 新实体；
新实体只在当前帧 free slot 中监督，满 16 slot 时不监督。
```

- 审阅提到的“上一轮 content target 缺失”已经过时，不再重复。

### P1-8 slot fallback 是合同未记录的例外

**结论：部分接受。**

当前确实存在：

- `net/model.py` duplicate stable ID → slot ID；
- `_sanitize_ids` 同样的 fallback；
- `pipeline/train_wm.py` 当前 ID 全 0 时 slot alignment。

这与“stable_id 才是身份、不能用 slot 当 stable identity”存在实质例外。

处理：

- §2.4/§10.5 明确：slot fallback 只用于 legacy/synthetic fixture 或 duplicate-ID 退化；
- 生产数据出现 duplicate stable ID 时，优先 fail-fast 或记录 diagnostic counter，不再静默使用 slot identity；
- 不建议直接删除 fixture fallback，否则旧测试/旧数据无法加载；但要在合同里标为“非生产身份路径”。

### P1-9 §0.1 混入 run-specific 试错/结果细节

**结论：接受。**

§0.1 当前包含：

- r3 的 init 路径、epochs、lr、near_weight、w_shield；
- data mix 行数；
- focus 1268 scenes、expert_ratio、shield_ttc；
- checkpoint 路径、val500 指标、r1 对比、easy/medium/hard 失败分布。

这些是实验记录，不是架构不变量，下一次 run 就会过期。

处理：

- 把详细数字/recipe 移到 `docs/baseline_r3.md`；
- `arch_lock.md` 只保留：

```text
当前 baseline 是 DAgger r3；
raw Bank + TSB + planner；
val500 gate 规则；
WM 不在 policy 前向路径。
```

- §0.1 不再承担“当前 run 结果”的职责。

---

## 3. P2

### P2-1 canonical gate：train_bc vs select_checkpoint

**结论：接受文档说明。**

当前有两个 selector：

- `pipeline/train_bc.py::_closed_loop_score`：纯 lexicographic success-first，无 collision/off-road 硬阈值；
- `pipeline/select_checkpoint.py`：先 `collision <= 0.20`、`off_road <= 0.20` 硬过滤，再按同一 score 排序。

r3 由 `train_bc` 选出，没有经过 `select_checkpoint` 硬阈值。

处理：

- 文档明确：当前 canonical gate 是 `train_bc` 的 lexicographic success-first；
- `select_checkpoint.py` 是可选的安全硬阈值 selector，不是当前 baseline 的选择路径；
- 如果 owner 要统一，后续把 eligibility 规则抽成共享函数，但不要在本轮偷偷改变 r3 的选择语义。

### P2-2 `ego_pose t0..t6 共 7 个` 与 API 暴露不一致

**结论：部分接受 / 解释。**

概念上 rollout 有 7 个 pose：

```text
t0 初始 pose + t1..t6 六次 transition
```

但：

- `ImagineMemoryBank`/BankWindow 只保留最后 6 帧；
- `imagine_rollout` 返回 6 个 policy dict，不返回 7 个 pose 的轨迹。

处理：

- 文档写成“概念上 t0..t6；Bank 保留 t1..t6；API 返回 6 个 policy，不暴露 7-pose 轨迹”。

### P2-3 PolicyHead 结构过简

**结论：接受。**

实际 `PlanHead` 是：

```text
query = Linear(2*hidden, hidden)([ego, nav])
cross-attention over [OD, LD] with valid mask
attn = norm(attn + query)
attn = attn + FFN(attn)
context = MLP([attn, ego, nav, masked_mean(OD), masked_mean(LD)])
action_mu = sigmoid(Linear(context))
action_logstd = clamp(Linear(context), -5, 0)
```

处理：文档按上述结构写，不再只写“query = MLP”。

### P2-4 world frame 表述过强

**结论：接受。**

`world frame` 不直接作为网络输入，但 `ego_pose` 用于计算 `delta_pose(t -> query)`，再进入 `FrameEmbed`。

处理：改成“world pose 不直接输入网络，只用于计算 frame-level delta_pose”。

### P2-5 rollout_depth 用途被夸大

**结论：接受。**

代码里 `rollout_depth` 只用于：

- 存储；
- FIFO / `advance` 的 `<=6` 检查；
- imagined/real 标记。

没有 loss 加权，也没有误差诊断。

处理：删除“loss 加权/误差诊断”，或标为未来用途。

### P2-6 Bank 只保存字段列表不完整

**结论：接受。**

`BankWindow` 还包含：

```text
od_type, ld_type, source, rollout_depth, frame_valid
```

处理：§5/§10 的字段列表补全。

### P2-7 `train_bc` 不负责 env step 不成立

**结论：接受。**

`train_bc.run_closed_loop_validation` 会：

- `build_env`；
- 调用 `pipeline.eval.run_episode`；
- 做闭环 validation。

处理：改成“`train_bc` 不负责训练数据采集/env rollout，但负责 closed-loop validation 调度”。

### P2-8 §13 文件表缺失 / §14.1 引用不存在

**结论：接受。**

缺失：

```text
env/obs.py
dataset/collect.py
dataset/splits.py
env/tracking.py
```

另外 `dataset/splits.py` 引用 `docs/arch_lock.md §14.1`，但 arch_lock 没有 §14.1。

处理：

- 补文件表；
- 把 §14.1 引用改为 §14，或在 arch_lock 增加 §14.1。

### P2-9 Phase 3/4 状态不清

**结论：接受。**

当前：

- `pipeline/train_ppo.py` 是 PPO contract canary，不是 GRPO；
- LQR 有 eval 支持（`pipeline/eval.py` controller=lqr），但不在 baseline；
- GRPO 尚未实现。

处理：阶段表加状态标签：

```text
Phase 3 GRPO: proposal / not implemented
Phase 4 LQR: eval-only / not baseline
```

### P2-10 `train_wm --allow-random-val`

**结论：接受。**

`--allow-random-val` 默认 false，是 legacy debug 开关。文档说“WM 固定 val500”时补充：

```text
默认固定 val500；--allow-random-val 仅用于 legacy debug，不进入 gate。
```

### P2-11 config 默认 ckpt 仍指向旧 run

**结论：接受。**

当前：

- `config/train_wm.yaml` 的 `ckpt: runs/bc_2k_raw/best_closed_loop.pt`；
- `config/train_ppo.yaml` 的 `ckpt` 同样指向 `runs/bc_2k_raw`；
- `runs/bc_2k_raw` 指向 `bc_2k_raw_261101094603`，这是 IL-only 3% baseline，不是 r3。

处理：

- 把默认 ckpt 更新为 `runs/bc_dagger_raw_r3_261101235342/best_closed_loop.pt`；
- 或者在 YAML 注释里明确标为“template，必须用 --ckpt 覆盖”，不能让它看起来是当前 baseline。

### P2-12 `select_checkpoint.py:52` NameError

**结论：反驳。当前代码没有这个 bug。**

当前代码：

```python
p = argparse.ArgumentParser(...)
...
if args.split == "test":
    p.error("Test500 is read-only; checkpoint selection must use val")
```

第 52 行是 `p.error`，不是 `parser.error`。`git blame` 显示该行在 `628f917` 已经是 `p.error`。

审阅引用的 `parser is not defined` 来自旧版本/旧 review。当前代码不需要修；可以补一个 `--split test` CLI 回归测试，防止未来回退。

### P2-13 `probe_wm` “隔离 WM transition error”不严格

**结论：部分接受。**

`probe_wm.py` 使用真实 action 序列，确实隔离了 **action 输入误差**；但：

- ego/pose 仍按 `imagine` 的递推方式更新；
- TSB ego token 来自模型 ego encoder；
- 因此没有完全隔离 ego/pose 误差。

处理：文档改成“隔离 action 输入，不隔离 ego/pose/TSB 递推误差”；要完整评估，需要 live-env parallel probe。

### P2-14 od_age / anchor / taxonomy

**结论：部分接受，部分反驳。**

审阅说：

> `train_bc.INPUT_KEYS` 不含 `od_age/od_anchor/od_taxonomy`

这不准确：

- `INPUT_KEYS` 明确包含 `od_anchor`、`od_taxonomy`；
- `net/encoder.py` 用 `od_anchor` 构造 OD geometry（`_geometry_od`）；
- `od_taxonomy` 用于 OD type（`_type`）。

真正的问题是：

- `od_age` 在 `env/obs.py` 中永远是 0，且不在 `INPUT_KEYS` / `KEEP_KEYS`，是死特征；
- `ld_taxonomy` 没有提供，`_type` 当前 fallback 到 road type；
- 这些需要在合同里明确为“未使用/死字段”或后续清理。

处理：

- 文档写清楚 `od_anchor` / `od_taxonomy` 是特征，不是 audit；
- `od_age` 标为 dead feature，后续删除或接入真实语义；
- `ld_taxonomy` 标为 current fallback，不是完整 taxonomy。

---

## 4. 哪些只改文档，哪些要改代码

### 4.1 只改文档

- 删除 `ego_velocity`；
- 统一 exact tracking / eval fallback 表述；
- 修正 `model.py` 责任表和依赖图；
- Phase 2c rollback 条件；
- WM gate / target drift / latent norm 标 TODO；
- valid mask / 新实体 slot 语义；
- slot fallback 例外；
- §0.1 详细数字移到 baseline 文档；
- selector 双定义说明；
- ego_pose 7 / API 6；
- PolicyHead 结构；
- world frame 表述；
- rollout_depth 用途；
- Bank 字段补全；
- train_bc env 表述；
- §13 文件表 / §14.1 引用；
- Phase 3/4 状态；
- allow-random-val 说明；
- od_age / ld_taxonomy 说明。

### 4.2 要改代码 / 配置

1. Phase 2c 命名：
   - `pipeline/train_joint.py` docstring、`joint_phase`；
   - `config/train_joint.yaml` 注释；
   - `test/test_train_joint.py` docstring。

2. `probe_wm` persistence：
   - 当前 `initial_frame_persistence` 改名；
   - 增加 `previous_step_persistence`。

3. `config/train_wm.yaml` / `config/train_ppo.yaml` 默认 ckpt：
   - 指向 r3，或标为 template。

4. duplicate stable ID：
   - 生产路径优先 fail-fast 或 diagnostic counter；
   - fixture fallback 保留并注释。

5. `select_checkpoint`：
   - 当前没有 NameError；
   - 可补 `--split test` CLI 回归测试，不需要修 bug。

### 4.3 不接受的审阅意见

1. **`select_checkpoint.py:52` NameError**：当前是 `p.error`，不是 `parser.error`；该 finding 已过期。
2. **`train_bc.INPUT_KEYS` 不含 `od_anchor/od_taxonomy`**：事实错误；两者都在 `INPUT_KEYS` 并被 encoder 使用。真正的问题是 `od_age` 死特征和 `ld_taxonomy` 缺失。

---

## 5. 建议修订顺序

1. 删除 `ego_velocity`，统一 exact tracking。
2. 修 `model.py` 责任表和依赖图。
3. 统一 Phase 2c 命名；把 rollback 条件改成 `policy_tol` 的实际逻辑。
4. 把 target drift / latent norm / collapse 标 TODO；WM gate 标 proposed。
5. 修 `probe_wm` persistence 命名/基线。
6. 修 valid mask / 新实体 slot 表述。
7. 把 §0.1 run-specific 数字移到 `docs/baseline_r3.md`。
8. 明确 canonical gate（train_bc lexicographic）与 `select_checkpoint` 的关系。
9. 补 §13 文件表、§14.1 引用、Phase 3/4 状态、allow-random-val、默认 ckpt。
10. 处理 `od_age` / `ld_taxonomy`，并补 duplicate-ID 的 production diagnostic。

修订完成后，`docs/arch_lock.md` 才适合锁为 pre-RL architecture contract。

---

## 6. 已核对为正确的部分

审阅列出的“已核实正确”部分，我基本同意，且重新核对如下：

- `BankWindow` schema 除 `ego_velocity` 外正确；
- 6 帧、16 OD/LD slot、ROI 80/20/25、top-16 stable-ID 排序正确；
- TSB 无 nav、无 geometry projection、frame-level `delta_pose` 正确；
- WM residual base、zero-init、direct 5D/10D decoder 正确；
- target-only 新实体 content target 已在 `pipeline/train_wm.py` 接通；
- 当前不启用 EMA；`train_wm` 冻结 encoder/TSB/target；
- probe depth 1..6 使用 `BankWindow.advance`；
- policy 无 value/risk，aux 为 collision/offroad/timeout/urgency/progress；
- 6 policy / 6 action / 6 transition，不写回 RealBank；
- raw loader 拒绝 projected/missing；
- r3 指标/path/override 可在 `runs/bc_dagger_raw_r3_261101235342` 验证；
- DriveModel 参数 1,270,748；
- 默认 BC 配方与 `config/train_bc.yaml` 一致。

审阅中的 P0/P1 文档问题大部分成立；核心架构不需要推翻，主要是把“已实现”“proposed gate”“run-specific 结果”三类内容分开。

---

## 7. 实施状态（2026-11-05）

- 不再使用 Phase 2b/2c 等阶段编号；训练目标按 IL/DAgger、WM one-step、WM horizon probe、joint fine-tune、GRPO、LQR 描述。
- duplicate non-zero stable ID 已在 `net/model.py` 和 `pipeline/train_wm.py` fail-fast；全 0 fixture 仍允许 slot fallback。
- `pipeline/probe_wm.py` 已区分 `initial_frame` 与 `previous_step` persistence；WM recurrence 为 `z_hat_{t+1}=z_hat_t+r_z,t`，persistence 是 `r_z=0`。
- `config/train_wm.yaml`、`config/train_ppo.yaml` 默认 ckpt 已指向 r3；`config/train_joint.yaml` 指向 r3 并注明 WM 未训练。
- `docs/arch_lock.md` 已同步上述现状；代码测试 `56 passed`。
