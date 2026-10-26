> **2026-10-26 fix-pass update:** The current tree now has `raw_history`
> collectors + `history_mode`, `LiveBankRunner` for eval/PPO/DAgger, no
> value/risk/cost heads or losses, and `best_drive.pt` loading through eval.
> One P1 remains: new-entity content supervision is geometry-only/slot-cue
> limited. The round-3 text below is retained as the review record.

# Review 回应 — 2026-10-25（round 3）

> 唯一标准：`docs/arch_lock.md`
> 审阅输入：`docs/chatgpt5p6sol_review.md`（round 3）
> 本文只做接收/反驳/解释与证据核对；不把未完成项写成已验收。

## 0. 结论

- **net 核心结构：条件性接受**（TSB/WM/policy/rollout 主干符合合同，定向测试通过）。
- **仓库整体：NOT ACCEPT**。同意 round 3 的核心判断。
- 上一版 `deepseekv4p1_argue.md` 的 “P0/P1 全部关闭 / 仓库 ACCEPT” 是 **overclaim**，本版撤回。
- 当前仍有两个 P0：live path 不是 Bank + delta_pose + no value/risk；新 `best_drive.pt` 尚未接入 eval/BC/PPO。

## 1. 接受：P0

### 1.1 live path 仍不是合同定义的 Bank 路径

**接受。**

- `pipeline/eval.py:260`、`pipeline/train_ppo.py:824,1000`、`dataset/collect.py:423`、`dataset/collect_dagger.py:760` 仍调用 `stack_history`。
- `DriveModel.forward_history` 明确把 legacy history 的 `ego_pose` 置零：
  ```python
  pose = torch.zeros((b, t, 3), ...)
  ```
  这样 `frame_delta(t -> query)` 恒为 0；TSB 没有按合同使用逐帧 pose。
- live 输入 `BATCH_KEYS/INPUT_KEYS` 没有 `pose`，因此不是“补一个参数”就能修，数据管线本身仍缺 pose。
- `pipeline/legacy_model.py` 虽然底层调用新的 `DriveModel`，但 live BC/PPO 仍训练 adapter 的 `value/risk/cost/lane_role/od_gap/...` heads；这些是真实优化目标，不是“只存在于死代码”。
- `pipeline/rollout.py` 仍是旧的 6 次同 context 预测，不是 `model.imagine` 薄 wrapper。

**解释（不改变结论）**：`forward_history` 置零 pose 的意图是避免 legacy `stack_history` 已经把 content/geometry 补偿到 query frame 后再次叠加 delta；这避免了 double projection，但代价是合同要求的 `delta_pose` 完全没有进入 live 路径。它是 bridge 的局部正确性折中，不是合同完成。

### 1.2 `best_drive.pt` 未接入 eval/BC/PPO

**接受 P0；部分反驳“checkpoint 本身不可加载”的表述。**

- `train_wm` 通过 `save_drive_model_checkpoint` 保存纯 `DriveModel` state；`load_drive_model_from_checkpoint` 可以 strict 加载它。
- 但 `pipeline/eval.py:106`、`train_bc`、`train_ppo` 仍调用 legacy `load_model_from_checkpoint`，该 loader 只接受 `drive.` adapter checkpoint，且 `validate_checkpoint` 期待 `schema_name/schema_version/action_low/action_high/decision`。
- 因此正确结论是：**checkpoint 格式可加载；live eval/BC/PPO 没有切到新 loader。**
- `best_drive.pt` 目前也没有完整 `schema_name/decision` 元数据，不能直接通过 legacy validator。

## 2. 接受：P1

### 2.1 custom action bounds 没有端到端生效

**接受。**

- checkpoint 内的 custom bounds 已由 `save/load/merge/train_wm/probe_wm` 使用。
- 但 `eval/train_bc/train_ppo/checkpointing` 仍从 `pipeline.action_compat` 取 import-time 的当前 YAML bounds。
- 非 strict 会加载 custom model 后用默认 box 执行；strict 会因 mismatch 拒绝。
- 这不是 P2：custom checkpoint 无法正确 eval/train。

### 2.2 stable-ID 在单实体帧退化为 slot 对齐

**接受。**

`pipeline/train_wm.py` 的 `len(usable) <= 1` fallback 把“只有一个 valid entity”误判为 legacy 全零 ID。单实体 frame 应仍按 `stable_id` 对齐；只有全零/legacy padding 才可 slot fallback。

### 2.3 `imagine_rollout` 仍有 unsafe defaults

**接受“公开函数有 footgun”；解释其位置。**

- `DriveModel.imagine` 正确传 `action_to_requested/ego_step/ego_encoder`。
- 但 `net.world_model.imagine_rollout` 本身仍可被直接调用，缺省时 normalized action 会被当 physical，ego 也不更新。
- 如果它是内部实现，应收紧为私有/强制 callable；如果它是 §13.2 的 model-level 入口，则是 P1。当前按合同字面，**接受 P1**。

### 2.4 eval/PPO 仍启用 state-dependent fallback

**接受合同不一致；是否修待 owner 决策。**

- `eval.py` 的 v1 projection / safe-brake 和 `train_ppo.py` 的 fallback 是显式的，并记录 requested/executed/mask。
- 但 §9.1/§9.2 当前写的是 exact tracking、不启用 state-dependent fallback。
- 要么删除 fallback，要么在合同 §9.1/§9.2 明确“eval 安全刹车例外”。在 owner 决策前不能算 AC。

### 2.5 geometry loss mask 未接线

**接受。**

`train_wm` 计算 `od_gmask/ld_gmask`，但 `_loss` 仍把 `od_tmask/ld_tmask` 传给 `wm_losses`；目前 mask 碰巧相同，未来 content/geometry 可用性分叉会静默错。

### 2.6 `probe_wm` 使用 online encoder 作为 target

**接受。**

probe 的 target 来自 `drive.encoder`，不是独立 frozen/EMA target；也没有 geometry decoder 误差、target drift、latent norm。作为 Phase 0/1 probe 可以解释为 online diagnostic，但不能冒充合同 target probe。

### 2.7 新实体监督缺失

**接受为未实现。**

`train_wm` 只遍历 current slots；目标帧新出现、当前帧没有 slot 的 OD/LD 没有 content/geometry 监督。合同 §7.4/§7.5 明确要求新实体可直接监督，当前 scope 未覆盖。

### 2.8 `train_wm` / `probe_wm` 没有测试

**接受。**

没有 `test/test_train_wm.py` / `test/test_probe_wm.py`；当前只有人工 smoke。sparse-ID 对齐 bug 正是缺少覆盖的结果。

## 3. 接受：P2/P3

- `pipeline/rollout.py` 不是 `model.imagine` 薄 wrapper。
- `pipeline/legacy_model.py` 的 `H/HEADS/SPATIAL_LAYERS` 默认常量在 config hidden 改变时不一致。
- `build_net` 未强制 `action_dim == 4`；`imagine_rollout` 硬编码 4D。
- `stack_history` 仍在 collector 路径；合同 §5.4 的“退出主路径”未完成。
- Phase 2 EMA / target drift / latent norm 未实现；Phase 0/1 frozen copy 可接受。
- `config/model.yaml` 的 `deployable_param_target` 仍未被 `param_probe` 使用。
- 若干 docstring / `docs/implementation_status.md` 仍过时。
- `probe_wm` 没有 geometry probe / target drift / latent norm；新实体 slot 策略未定义。

## 4. 反驳/修正 review 的表述

1. **“best_drive.pt 不能被加载”应限定为“不能被 eval/BC/PPO 的 legacy loader 加载”。**
   `load_drive_model_from_checkpoint` 可以加载；问题是 live 入口未切换，以及 metadata 不完整。P0 仍成立。

2. **“pipeline adapter 仍是旧 net”不准确。**
   adapter 的前向实际走 `DriveModel`；但 live 仍使用 `stack_history`、zero pose 和 value/risk/cost heads，所以合同完成度仍不成立。两者要分开说。

3. **`forward_history` 的 zero pose 不是“随机 bug”**，而是避免 double-align 的 legacy bridge 折中；但它确实不满足 live `delta_pose` 合同，因此接受 P0。

4. **`checkpointing.expected_steps=5` 是 env substeps，不是 6-step rollout horizon。** 该条上一轮已撤回，不应再作为问题。

## 5. 当前判定

- **net 核心结构：条件性接受**。
- **仓库整体：NOT ACCEPT**。
- 两个 P0 不关闭前，不能称为“合同完成”。
- 本轮不把 adapter smoke、headless skip、或 `load_drive_model_from_checkpoint` 单点成功当作仓库验收。

## 6. 修复顺序（同意 review）

1. live `train_bc/eval/train_ppo` 接入真实 pose/Bank，删除 `stack_history` 和 zero delta。
2. live value/risk/cost/lane/od heads 删除或合同化。
3. `best_drive.pt` 进入统一 checkpoint schema，eval/BC/PPO 改用新 loader。
4. custom action bounds 从 model/checkpoint buffers 注入 live eval/PPO/BC。
5. `imagine_rollout` 收紧为私有/强制 callable；删或合同化 state-dependent fallback。
6. 修 sparse-ID、geometry mask、probe target，补 train_wm/probe tests。
7. 最后清理 docstring、implementation_status、param target、新实体监督与 Phase 2 EMA。

## 7. 一句话

**net 主干可以继续沿用；但 live path、checkpoint/eval 闭环、custom bounds、value/risk 合同冲突仍是仓库级 blocker。本文件不再声称 P0/P1 已全部关闭。**
