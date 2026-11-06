# arch_lock review response — 审阅结论（2026-11-03）

> 审阅对象：`docs/deepseekv4p1_argue.md`（对 arch_lock 审阅的答复）
> 对照：`docs/arch_lock.md`、当前代码、run artifacts
> 结论：**答复总体成立，主判断接受；两条反驳都正确；但 P1-2 的修正依赖图仍有错误，P1-8 只是部分接受，需要落到代码/诊断，不能只改文字。**

## 0. 总判断

这份答复的质量高于上一版：

- 没有继续声称 “ACCEPT no P0/P1”；
- 接受了 `ego_velocity`、exact tracking、Phase 2c、WM gate、probe persistence、target mask、run-specific 细节等主要问题；
- 两条反驳是正确的，之前审阅中的对应 finding 应撤回；
- 但仍有一处**新引入的依赖图错误**，以及一处“只改文档、不改生产行为”的不完整接受。

### 结论速览

| 答复项 | 审阅结论 |
|---|---|
| P0-1 `ego_velocity` | **接受**，删除字段 |
| P1-1 exact tracking | **接受**，统一表述 |
| P1-2 `model.py` 责任表 | **部分接受**；文档可改，但答复里的依赖图仍错 |
| P1-3 Phase 2b/2c | **接受**，统一 2c |
| P1-4 Phase 2c 回滚 | **接受**，改成 `policy_tol` 实际逻辑 |
| P1-5 WM gate / drift / norm | **接受**，标 TODO/proposed |
| P1-6 probe persistence | **接受**，区分 initial/previous-step |
| P1-7 valid mask / 新实体 | **接受**，改成 target mask 语义 |
| P1-8 slot fallback | **部分接受**；必须补生产 fail-fast/diagnostic，不能只写“例外” |
| P1-9 run-specific 细节 | **接受**，移到 baseline 文档 |
| P2-1 selector 双定义 | **接受**，明确 canonical gate |
| P2-2 ego_pose 7 个 | **解释成立**，文档写清概念/API 差异 |
| P2-3 PolicyHead | **接受** |
| P2-4 world frame | **接受** |
| P2-5 rollout_depth | **接受** |
| P2-6 Bank 字段 | **接受** |
| P2-7 train_bc env | **接受** |
| P2-8 文件表/§14.1 | **接受** |
| P2-9 Phase 3/4 状态 | **接受** |
| P2-10 `--allow-random-val` | **接受** |
| P2-11 默认 ckpt | **接受** |
| P2-12 `select_checkpoint` NameError | **反驳成立**，撤回原 finding |
| P2-13 probe 隔离 | **部分接受**，改成“隔离 action 输入” |
| P2-14 `od_age`/anchor/taxonomy | **部分反驳成立**；anchor/taxonomy 在 INPUT_KEYS，od_age 仍是死特征 |

---

## 1. 接受：P0-1 `ego_velocity`

`docs/arch_lock.md:208` 的 `FrameRecord.ego_velocity: [2]` 在 `net/mem_bank.py:75-89` 不存在；代码字段为：

```text
episode_id, frame_id, timestamp, ego_pose, od, ld,
source, rollout_depth, parent_frame_id,
action_from_parent, encoder_version
```

全仓库只有 arch_lock 出现 `ego_velocity`。答复接受删除，正确。
这是 doc-only 错误；不需要为了文档新增代码字段。

---

## 2. 接受：P1-1 exact tracking 限定范围

答复接受“§0 #14 / §9.1 / §14 内部矛盾”，正确。

代码事实：

- `pipeline/eval.py:301-349` 会做 `v_next` projection / safe-brake；
- `pipeline/train_ppo.py:293-383` 同样，并 mask PPO 样本；
- `§9.1` 已允许该例外。

文档应统一为：

```text
BC/DAgger policy 训练路径：requested_action == executed_action。
eval/PPO 安全执行层：允许显式 v_next projection / safe-brake；
记录 requested/executed，并在 RL loss 中 mask 或重算 log-prob。
```

删除 §0 #14、§9.1 开头、§14 的绝对表述即可；代码无需改。

---

## 3. 部分接受：P1-2 `model.py` 责任表 —— 答复的依赖图仍错

答复同意“文档可改，不搬代码”，这个取舍本身可以接受：
`net/model.py` 的 raw-history adapter / FIFO-safe alignment 是模型侧的编排逻辑，强行搬进 `mem_bank.py` 可能让存储层依赖 encoder 语义。把责任写清楚即可。

但答复给出的依赖图仍然不正确：

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

实际 import：

- `net/world_model.py` 只 import `net.mem_bank`，**不 import `encoder.py`**；
- `net/encoder.py` import `env.obs`、`net.mem_bank`；
- `net/planner.py` 不 import `world_model.py`；
- `net/build_net.py` import `encoder.py`、`model.py`、`planner.py`、`world_model.py`；
- `net/model.py` import `encoder.py`、`mem_bank.py`、`planner.py`、`world_model.py`。

因此依赖图应类似：

```text
mem_bank.py
  ↑
encoder.py
  ↑
model.py
  ↑
build_net.py

mem_bank.py
  ↑
world_model.py
  ↑
model.py

planner.py
  ↑
model.py
```

`world_model.py` 与 `encoder.py` 是并列依赖 `mem_bank.py`，不是 `world_model → encoder`。
如果按答复当前的图写进 arch_lock，会再次引入 doc-code 不一致。

**审阅结论：接受“改文档、不搬代码”，但不接受答复里的依赖图；必须修正。**

---

## 4. 接受：P1-3 Phase 2b/2c 统一为 2c

答复接受统一为 Phase 2c，正确。当前冲突是：

- `docs/arch_lock.md:835/918/943` → 2c；
- `pipeline/train_joint.py:1`、`:359` → 2b；
- `config/train_joint.yaml:1` → 2b；
- `test/test_train_joint.py:1` → 2b。

需要改代码/config/test 的注释和 `joint_phase`，不能只改文档。旧 checkpoint 的 `"2b"` 可保留为历史 metadata，但不能用于路径分派。

---

## 5. 接受：P1-4 Phase 2c 回滚条件

答复接受改成实际实现，正确。`pipeline/train_joint.py:347-352` 当前是：

```python
policy_budget = policy_tol * abs(initial_policy)
policy_ok = (val_policy - initial_policy) <= policy_budget
if val_wm < best_val and policy_ok:
    ...
```

没有 closed-loop success。
文档应写“val policy loss 相对初始退化不超过 `policy_tol`，且 val WM loss 改善”；若要 policy success，需先实现 closed-loop gate。

---

## 6. 接受：P1-5 WM gate / target drift / latent norm

答复接受标 TODO/proposed，正确。当前：

- `pipeline/train_wm.py:506-520` 只按 `val_loss` 选 best；
- `pipeline/probe_wm.py:266-270` 只输出 content/geometry/persistence 均值；
- 没有 target drift、latent norm、latent collapse 指标，也没有 pass/fail gate。

文档 §7.6/§12.6 应明确：

```text
target drift / latent norm / latent collapse: TODO, 当前未实现。
WM gate: proposed, 当前只记录 loss/persistence，不做 pass/fail。
```

不能把“必须优于 persistence/zero、不能 collapse”写成当前已执行行为。

---

## 7. 接受：P1-6 probe persistence 定义

答复接受区分 initial-frame / previous-step，正确。当前 `pipeline/probe_wm.py:258-265`：

```python
base_od = window.od_content[:, -1].clone()
base_ld = window.ld_content[:, -1].clone()
...
persistence["od_content"].append(masked_mse(base_od, target, mask))
```

depth>1 时这是 **initial-frame persistence**，不是文档 §12.6 写的 `z_hat_t = z_hat_{t-1}`。
建议：

- 当前指标改名 `initial_frame_persistence`；
- 另加 `previous_step_persistence`，用上一步 `z_od/z_ld`；
- 文档分别定义两者。

---

## 8. 接受：P1-7 valid mask / 新实体 slot

答复接受 target mask 语义，正确。当前 `pipeline/train_wm.py:295-313` 使用：

- `od_tmask/ld_tmask`：content target mask；
- `od_gmask/ld_gmask`：geometry target mask；
- `od_valid=maximum(current, tmask)`；
- `od_current_valid` 区分 target-only slot。

文档应改成：

```text
content 用 content target mask；
geometry 用 geometry target mask；
current valid 只用于区分当前实体与 target-only 新实体；
新实体只在当前帧 free slot 中监督，16 slot 全满时不监督。
```

“target-only content target 缺失”是旧 finding；当前 `train_wm.py:251-271` 已填，不能继续报。

---

## 9. 部分接受：P1-8 slot fallback —— 需要代码/诊断，不只是文档

答复承认 slot fallback 是例外，并提出“生产数据优先 fail-fast 或 diagnostic counter”，方向正确。
但当前代码仍然是静默 fallback：

- `net/model.py:232-256`：duplicate stable ID → slot ID；
- `net/model.py:359-384`：`_sanitize_ids` 同样；
- `pipeline/train_wm.py:152-171`：当前 ID 全 0 时 slot alignment。

如果只改文档写“legacy 例外”，生产数据出现 duplicate stable ID 时仍然会静默把 slot 当身份，和 §0 #4 / §10.5 的 stable-ID 契约冲突。
**审阅结论：部分接受；必须至少做一项：**

1. 生产路径 fail-fast（duplicate stable ID 直接报错）；或
2. 记录 diagnostic counter，并在 metrics 中暴露；或
3. 在合同中明确“slot fallback 是生产可接受的降级”，并说明影响。

不能只写“legacy 例外”而不改代码/诊断。

---

## 10. 接受：P1-9 §0.1 run-specific 细节

答复接受移到 `docs/baseline_r3.md`，正确。
`arch_lock.md:88-144` 的 r3 init 路径、epochs/lr、near_weight/w_shield、data mix、1268 scenes、expert_ratio、shield_ttc、checkpoint 路径、val500 指标、r1 对比、failure distribution 都是实验记录，不是架构不变量。
arch_lock 只保留：

```text
当前 baseline 是 DAgger r3；
raw Bank + TSB + planner；
val500 gate 规则；
WM 不在 policy 前向路径。
```

r3 数字本身已核实正确，不需要改数；只是不应放在 arch_lock。

---

## 11. P2 逐项结论

### 11.1 接受：P2-1 canonical gate

`train_bc._closed_loop_score` 是纯 lexicographic，无 collision/off-road 硬阈值；`select_checkpoint` 先硬过滤再排序。r3 由 `train_bc` 选出。
文档应明确：**当前 canonical gate 是 train_bc 的 lexicographic success-first；`select_checkpoint` 是可选 stricter selector**。答复接受，正确。

### 11.2 解释成立：P2-2 ego_pose 7 个

概念上 t0..t6 共 7 个 pose；`ImagineMemoryBank`/BankWindow 只保留最后 6 帧，`imagine_rollout` 返回 6 个 policy dict，不暴露 7-pose 轨迹。
文档写成“概念 t0..t6；Bank 保留 t1..t6；API 返回 6 个 policy”即可，不是代码 bug。

### 11.3 接受：P2-3 PolicyHead 结构

实际是：

```text
query = Linear(2*hidden, hidden)([ego, nav])
cross-attention over [OD, LD] with valid mask
attn = norm(attn + query)
attn = attn + FFN(attn)
context = MLP([attn, ego, nav, masked_mean(OD), masked_mean(LD)])
action_mu = sigmoid(Linear(context))
action_logstd = clamp(Linear(context), -5, 0)
```

文档应照此写，不能只写“query = MLP”。

### 11.4 接受：P2-4 world frame

world pose 不直接输入，但 `ego_pose` 用于计算 `delta_pose` 再进入 `FrameEmbed`。
文档应改成“world pose 不直接输入网络，只用于计算 frame-level delta_pose”。

### 11.5 接受：P2-5 rollout_depth

代码只用于存储、FIFO、`advance` 的 `<=6` 检查，没有 loss 加权/误差诊断。
删掉“loss 加权/误差诊断”，或标未来用途。

### 11.6 接受：P2-6 Bank 字段

`BankWindow` 还包含 `od_type/ld_type/source/rollout_depth/frame_valid`；文档字段列表补全。

### 11.7 接受：P2-7 train_bc env

`train_bc.run_closed_loop_validation` 会 `build_env` 并调用 `run_episode`。
改成“不负责训练数据采集/env rollout；负责 closed-loop validation 调度”。

### 11.8 接受：P2-8 文件表 / §14.1

补 `env/obs.py`、`dataset/collect.py`、`dataset/splits.py`、`env/tracking.py`；把 `dataset/splits.py` 的 `§14.1` 引用改为 §14，或在 arch_lock 增加 §14.1。

### 11.9 接受：P2-9 Phase 3/4 状态

`train_ppo.py` 是 PPO canary，不是 GRPO；LQR 有 eval 支持但不在 baseline。
阶段表加 `implemented / proposal / eval-only` 标签。

### 11.10 接受：P2-10 allow-random-val

默认 false，是 legacy debug 开关；文档应注明“不进入 gate”。

### 11.11 接受：P2-11 默认 ckpt

`config/train_wm.yaml`、`config/train_ppo.yaml` 仍指向 `runs/bc_2k_raw`（旧 IL-only run），不是 r3；`config/train_joint.yaml` 指向不存在的 `runs/wm_2k/best_drive.pt`。
应更新为 r3，或明确标为 template 必须 `--ckpt` 覆盖。

### 11.12 反驳成立：P2-12 `select_checkpoint` NameError

当前 `pipeline/select_checkpoint.py:52` 是：

```python
p.error("Test500 is read-only; checkpoint selection must use val")
```

不是 `parser.error`；实测当前不会 NameError。
**之前审阅的该 finding 应撤回**。建议补一个 `--split test` CLI 回归测试，防止未来回退；不需要修 bug。

### 11.13 部分接受：P2-13 probe 隔离

`probe_wm` 使用真实 action，确实隔离 action 输入误差；但 ego/pose 仍按 imagine 递推，TSB ego token 来自模型 ego encoder。
文档应改成“隔离 action 输入，不隔离 ego/pose/TSB 递推误差”；完整评估需要 live-env parallel probe。

### 11.14 部分反驳成立：P2-14 od_age / anchor / taxonomy

`pipeline/train_bc.INPUT_KEYS` 当前确实包含：

```text
od_anchor
od_taxonomy
```

并且 encoder 使用它们；`KEEP_KEYS` 也包含。**之前“loader 丢弃 anchor/taxonomy”的说法不准确，应撤回。**

仍然成立的是：

- `env.obs` 的 `od_age` 永远为 0，且不在 `INPUT_KEYS/KEEP_KEYS`，是死特征；
- `ld_taxonomy` 没有提供，`_type` fallback 到 road type。

文档应明确：

```text
od_anchor / od_taxonomy 是特征；
od_age 是 dead feature，后续删除或接入真实语义；
ld_taxonomy 当前 fallback，不是完整 taxonomy。
```

---

## 12. 测试/证据

- 当前 `pytest test -q`：**133 passed, 1 skipped, 1 failed, 2 errors**；失败/错误均为 MetaDrive/Panda3D headless。
- r3 指标/path/override 已在 `runs/bc_dagger_raw_r3_261101235342` 验证：
  - success 0.610 / collision 0.180 / off_road 0.112 / route 0.832 / feasible 0.856 / fallback 0.094 / timeout 0.106 / stuck 0.086；
  - best_epoch 10；
  - `init` 只在 `runs/logs/train_bc_dagger_raw_r3.log:8`，不在 checkpoint config；这是可复现性弱点。
- target-only 新实体 content target 已在 `pipeline/train_wm.py:251-271` 接通；不要重复旧 P1。
- `select_checkpoint` NameError 是旧版本残留；当前 `p.error` 正确。
- `od_anchor/od_taxonomy` 在 INPUT_KEYS；`od_age` 是死特征。

---

## 13. 最终建议

**接受这份答复的主判断和大多数修正项；但先完成以下 3 件事再锁 arch_lock：**

1. **修正 P1-2 的依赖图**：`world_model.py` 不依赖 `encoder.py`；`planner.py` 不依赖 `world_model.py`；`build_net.py` 依赖 encoder/model/planner/world_model。
2. **P1-8 不能只改文档**：生产 duplicate stable ID 必须 fail-fast 或记录 diagnostic；否则 stable-ID 契约仍有静默例外。
3. **P1-3/P1-4 要落到代码/config/test**：Phase 2b→2c 和 `policy_tol` 逻辑不能只写在 arch_lock。

其余接受项按答复执行即可；两条反驳（`select_checkpoint`、`INPUT_KEYS`）成立，原 review 对应 finding 已撤回。
修订完成后，arch_lock 才适合作为 pre-RL architecture contract。
