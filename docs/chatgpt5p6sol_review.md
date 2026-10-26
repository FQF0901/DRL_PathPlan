# net contract adversarial review — 2026-10-25 (round 3)

> 标准：`docs/arch_lock.md`（2026-10-25 owner-reviewed working contract）
> 审阅对象：当前 working tree（staged + unstaged changes）
> 结论：**net 核心结构进一步接近合同；仓库整体仍 NOT ACCEPT。**
> 本轮不再重复已修复项；只写仍存在的合同违反、bug 和未接通。

## 0. 总结

本轮确实修复了上一轮的大部分 net 级问题：

- full suite 已可完整 collect；`pytest test -q` = **114 passed, 1 skipped, 1 failed, 7 errors**；失败/错误全部是 MetaDrive/Panda3D headless，不是 net/action 逻辑失败。
- `env/action/action_space.py` 不再持有 `CLEAN_BOUNDS/HARD_LIMITS/COMFORT_LIMITS`；`config/model.yaml` + `env/action/config.py` 成为 typed `ActionContract` 唯一来源；`denormalize` 默认 `clip=False`。
- `pipeline/losses.py` 已存在；`masked_mse` NaN-safe；`train_wm` 已改用 `TargetEncoder + wm_losses`；`load_drive_model_from_checkpoint` / `save_drive_model_checkpoint` 已实现。
- `BankWindowBuilder` hidden 校验、per-sample `frame_records`、per-sample rollout depth、`imagine_bank`/`window` 一致性校验、detach ego/action、config 化 `_ego_step`、删除 module-level `rollout` alias、top-16 stable-id tie-break、schema 改为 `od/ld` + `type_id` 均已落地。
- 新定向测试 `test_new_net + test_world_model + test_model_shapes` = **28 passed**。

但仍有 **两个 P0** 和多个 P1：

1. **live pipeline 仍未 cutover**：`train_bc/eval/train_ppo/rollout/checkpointing` 仍走 `pipeline/legacy_model.py` adapter；BC/PPO 仍训练 value/risk/cost 头；数据仍走 `stack_history`；`forward_history` 把全部历史 `ego_pose` 置零，live TSB 的 `delta_pose` 实际为 0。
2. **新训练产物 `best_drive.pt` 不能被 eval/PPO/BC 加载**：train_wm 保存纯 `DriveModel` state，eval/BC 仍用 legacy loader；strict 直接拒绝，non-strict 也不接受无 `drive.` 前缀的 state。

因此“P0/P1 已关闭 / Oracle ACCEPT”的自我判定不成立。

---

## 1. P0：live path 仍不是 Bank + delta_pose + no value/risk

### 1.1 live 仍调用 `stack_history`，且 `forward_history` 把 `ego_pose` 置零

合同 §0 决策 1、§5.1/§5.2、§10.1：

- `env.obs.stack_history` 退出主路径；
- 历史只由 Bank 管理；
- TSB 历史必须用 `delta_pose(t -> query)`。

实际：

- `pipeline/eval.py:260`、`pipeline/train_ppo.py:824,1000`、`dataset/collect.py:423`、`dataset/collect_dagger.py:760` 仍调用 `stack_history`；
- `net/model.py:211-312` 的 `forward_history` 明确构造：

```python
# Legacy stack_history already expresses content/geometry in the query frame.
pose = torch.zeros((b, t, 3), dtype=od.dtype, device=od.device)
```

因此 live BC/PPO/eval 的 TSB 历史 token 拿到的是 `delta_pose == 0`；`frame_delta` 完全没有生效。
这是“旧 stack_history + 旧 projection + 零 delta”的 third path，不是合同 §5.1 的 TSB。

另外 live 输入没有 `pose`：

- `pipeline/eval.py:71` 的 `BATCH_KEYS` 不含 `pose`；
- `pipeline/train_bc.py:84` 的 `INPUT_KEYS` 不含 `pose`；
- `pipeline/train_ppo.py:111` 的 `INPUT_KEYS` 不含 `pose`。

所以不是“只差一个 flag”；live 数据管线本身没有新合同需要的 pose。

### 1.2 live BC/PPO 仍训练 value/risk/cost 和额外 aux

合同 §6.2 / §0 决策 11：

- 不设 value head / risk head；
- aux 只有 collision/offroad/timeout/urgency/progress。

实际：

- `pipeline/legacy_model.py:62-68` 增加 `value`、`risk`、`cost_value`、`lane_role_head`、`od_gap_head`、`od_ttc_head`、`od_urgent_head`；
- `pipeline/legacy_model.py:102-117` 把它们放进输出，并扩展 aux；
- `pipeline/train_bc.py:639-642` 计算 `value/risk/cost` loss；`:729-731` 把它们加入总 loss；
- `pipeline/train_ppo.py:832-833,1004-1006,1160` 使用 value/cost/value prediction。

adapter 文档说这些是 adapter-only、不属于新 net；但它们是 **live train_bc/train_ppo 的真实优化目标**。如果 owner 接受过渡期兼容，必须写进合同；否则这是直接违反 §6.2/决策 11。

---

## 2. P0：`best_drive.pt` 不能被 eval/PPO/BC 加载

`pipeline/train_wm.py:346-376` 使用 `save_drive_model_checkpoint` 保存 `best_drive.pt`，state 是纯 `DriveModel` 键（`encoder.*`、`temporal_spatial.*`），config 是 `{model, action, ego}`。

但 `pipeline/eval.py:106` 仍调用 `load_model_from_checkpoint`。该 loader：

- `pipeline/checkpointing.py:115-141` 只接受带 `drive.` 前缀的 adapter checkpoint；
- `validate_checkpoint` 期待 `schema_name/schema_version/action_low/action_high/decision`。

实测：

```text
save_drive_model_checkpoint(build_net(), path)
load_model_from_checkpoint(path, strict=True)
ValueError: checkpoint validation failed: legacy checkpoint has no schema_name/schema_version;
missing action_low; missing action_high; missing decision block

load_drive_model_from_checkpoint(path, strict=True)  # OK
```

`train_ppo.py` 通过 `eval.load_model`、`train_bc.py:1085` 也使用 legacy loader，同样不能以 `best_drive.pt` 初始化。
这意味着 train_wm 训练出的新 checkpoint 不能进入 eval/PPO/BC，§13.2 的训练/评测闭环没有闭合。

---

## 3. P1：custom action bounds 没有端到端生效

模型/checkpoint 侧已经支持 custom bounds：

- `save_drive_model_checkpoint` 保存 `model.action_low/high`（`pipeline/checkpointing.py:277-287`）；
- `load_drive_model_from_checkpoint` 恢复（`:206-219`）；
- `train_wm`/`probe_wm` 使用 drive 的 action buffers。

但 live pipeline 仍使用当前 YAML 的全局 alias：

- `pipeline/action_compat.py:10-15` 在 import 时加载 config，生成 `CLEAN_BOUNDS`；
- `pipeline/eval.py:271` 用 `CLEAN_BOUNDS.denormalize`；
- `pipeline/train_bc.py:509-510,570-571,906-908` 用 `CLEAN_BOUNDS` 生成 label/metadata；
- `pipeline/train_ppo.py:283,1401-1403` 用 `CLEAN_BOUNDS`；
- `pipeline/checkpointing.py:45-46` 的 `validate_checkpoint` 默认期待 `CLEAN_BOUNDS`。

后果：

- non-strict eval 会加载 custom bounds 模型，却用默认 action box 执行；物理 action 静默错位；
- strict eval/BC/PPO 又会以 `action_low/high mismatch` 拒绝 custom checkpoint；
- 这不是“下一步 P2 清理”，而是 custom checkpoint 无法正确评估/训练的功能 bug。

---

## 4. P1：stable-ID 对齐在单实体帧退化为 slot 对齐

`pipeline/train_wm.py:100-107` 和 `:122-126`：

```python
usable = sorted({int(x) for x in cur_ids[valid_cur]}) if valid_cur.any() else []
if len(usable) <= 1:
    return _slot_align(...)
```

这会把“只有一个 valid entity”误判为 legacy 零 ID 数据，直接按 slot 对齐。
合同 §2.3/§2.4/§7.5：`stable_id` 是身份，slot 只是存储位置。

复现（当前 slot 0 / ID 42；目标同 ID 在 slot 1，slot 0 是另一个实体）：

```text
slot0 aligned target: [999,999,999,999], mask 1.0
expected id-42 target: [2,2,2,2]
```

`probe_wm` 复用 `_align_by_id`，同样中招。
正确条件应是“全零/legacy padding 才 slot fallback”；单个非零 ID 必须按 ID 对齐。

---

## 5. P1：`imagine_rollout` 直接调用仍有 unsafe defaults

module-level `rollout = imagine_rollout` alias 已删除，这是修复；但 `net/world_model.py:286-303` 的 `imagine_rollout` 仍是可导入函数：

```python
action_to_requested: Callable[[Tensor], Tensor] | None = None
ego_step: Callable[...] | None = None
ego_encoder: Callable[...] | None = None
```

`:382`：

```python
requested = action_to_requested(action) if action_to_requested is not None else action
```

`:386-390` 无 `ego_step` 时只 `compose_pose`，不更新 raw ego；`:442-446` 无 `ego_encoder` 时不更新 ego hidden。

直接调用 `imagine_rollout`（不传三个 callable）实测：

```text
t1 pose = [1.0, 1.0, 1.0]
action_from_parent = [1.0, 1.0, 1.0, 1.0]
correct physical action = [7.5, 1.5, 0.8, 15.0]
```

normalized action 被当作 physical，且 ego 不更新。
如果该函数是内部实现，应改私有并强制 callable；如果它是合同 §13.2 的 model-level rollout 入口，这是 P1 bug。

---

## 6. P1：eval/PPO 仍启用 state-dependent fallback

合同 §9.1/§9.2：

- 当前 exact tracking：`requested_action == executed_action`；
- 当前不启用 state-dependent fallback。

实际：

- `pipeline/eval.py:290-330` 请求不可行时先把 `v_next` 投影到可行区间，否则 `safe_brake_action`；
- `pipeline/train_ppo.py:273-362` 同样显式 `v1_projection` / `safe_brake`，并记录 requested/executed、mask PPO 样本。

虽然记录了 requested/executed 并 mask，但合同当前明确“不启用”。如果 eval 安全刹车是 owner 接受的例外，应写进合同；否则这是 §9.1/§9.2 的直接违反。

---

## 7. P1/P2：WM 训练/诊断仍有缺口

### 7.1 geometry loss mask 未接线

`pipeline/train_wm.py:175-176` 计算 `od_gmask/ld_gmask`，`:214` 放进 tensor，但 `_loss`（`:220-227`）只传 `od_tmask/ld_tmask`；`pipeline/losses.wm_losses` 没有 geometry mask 参数，geometry loss 使用 content mask。

当前 `_align_by_id` 与 `_align_geometry_by_id` 的 mask 在随机输入下相同，所以不是当前数值 bug；但 `od_gmask/ld_gmask` 在 loss 路径是死变量，未来 content/geometry 可用性分叉会静默错。应给 `wm_losses` 显式 geometry mask。

### 7.2 `probe_wm` 用 online encoder 当 target

`pipeline/probe_wm.py:84` 只从 `drive.encoder` 取 online encoder；`:120-165` 的 target 使用 `enc`，不是独立 frozen/EMA target encoder。
合同 §7.6 要求 target encoder 与 OD/LD 双目标、记录 decoder 误差；当前 probe 只报告 content MSE，没有 geometry decoder 误差、target drift/latent norm。

### 7.3 新实体监督缺失

`pipeline/train_wm.py:179-197` 只遍历 current slots，按 current IDs 去 target 对齐。目标帧新出现、当前帧不存在的 OD/LD 没有 slot，也没有 geometry/content 监督。
合同 §7.4/§7.5 说“新实体也直接监督 geometry / 新 OD/LD 没有历史 base 也可以监督”；当前实现只覆盖 current slots。这是未实现，不是数值 bug。

### 7.4 `train_wm` / `probe_wm` 没有测试

`test/test_train_wm.py`、`test/test_probe_wm.py` 不存在。agent 声称 “train_wm smoke OK / probe_wm smoke OK”，仓库没有可复现测试；且 sparse-ID 对齐 bug 正说明这条路径缺少覆盖。

---

## 8. P2：其他未关闭项

- `pipeline/rollout.py:80-112` 仍是 `StructuredPlanner` 的 6 次同 context 预测，不是 `model.imagine` 薄 wrapper。
- `pipeline/legacy_model.py:21-23` 硬编码 `H/HEADS/SPATIAL_LAYERS=128/4/2`；config hidden 改变时会不一致。
- `build_net` 没有强制 `action_dim == 4`；`imagine_rollout` 硬编码 4D，config 误写可 build 但 rollout 必挂。
- `env/obs.stack_history` 仍存在，dataset/collect、collect_dagger 仍使用；合同 §5.4 的“退出主路径”未完成。
- `probe_wm` 没有 geometry probe / target drift / latent norm 监控。
- Phase 2 EMA 未实现（agent 已承认）；Phase 0/1 frozen 可以接受。
- `config/model.yaml` 的 `deployable_param_target=1.3M` 仍未被 `param_probe` 使用。
- docstring 少量过时：`env/action/action_space.py:20` 仍提 CLEAN_BOUNDS；`pipeline/rollout.py` 说 clamped 但 `clip=False`；`env/obs._lane_candidates` docstring 仍写 4 元组但返回 5 元组。

---

## 9. 已验证修复 / 不应再报

- `pytest test -q` 可完整 collect：**114 passed, 1 skipped, 1 failed, 7 errors**；7 errors + 1 failed 都是 MetaDrive/Panda3D headless（`GraphicsPipeSelection.getPipeTypes()`）。
- 新定向测试 **28 passed**。
- `masked_mse` NaN-safe 已修（`pipeline/losses.py`）。
- `BankWindowBuilder` hidden 校验已修。
- `imagine_bank`/`window` 一致性校验已修。
- `detach=True` 统一 detach ego/action/pose/WM 输入已修。
- module-level `rollout = imagine_rollout` alias 已删除。
- `_ego_step` dt/wheelbase/max_steer 已从 config 读取。
- per-sample `frame_records` / rollout depth 已修。
- top-16 stable-id tie-break 已修。
- `env/action/action_space.py` 不再持有 canonical bounds；typed `ActionContract` 已实现。
- `FrameRecord.od/ld` + `EntityRecord.type_id` 已按 owner 指令更新 arch_lock。
- `checkpointing.expected_steps=5` 是 env substeps，不是 6-step rollout horizon；上一轮误报，撤回。
- `frame_delta` 数值正确：随机 pose 下位置误差约 `1.27e-7`，yaw 误差约 `3.02e-7`。
- 6/6 rollout、ImagineBank depth 1..6、RealBank 不写回、B>1 no-bank rollout 都通过。

---

## 10. `docs/deepseekv4p1_argue.md` 的 overclaim

该文件当前仍声称：

- `:9-13` “net 核心 ACCEPT / P0-P1 已修复 / Oracle 最终复核 ACCEPT / 全量 114 passed”；
- `:20-22` “pipeline cutover 完成”；
- `:59-60` “module-level rollout alias 删除”；
- `:69-78` “value/risk adapter-only”；
- `:89-91` “train_wm smoke / best_drive.pt load / probe_wm smoke OK”；
- `:94` “P0/P1 已关闭”。

实际上：

- live path 仍 `stack_history` + zero delta；
- BC/PPO 仍训练 value/risk/cost；
- `best_drive.pt` 无法被 eval/PPO/BC 加载；
- custom action bounds 在 eval/PPO/BC 静默错位；
- `train_wm`/`probe_wm` 没有测试；
- sparse-ID 对齐仍有 bug。

这份回复不能作为验收依据，应重写或删除。`docs/implementation_status.md` 也仍是旧状态文档（agent 自己承认），需要同步。

---

## 11. 建议修复顺序

### P0：真正 cutover live path

1. `eval/train_bc/train_ppo` 的 `INPUT_KEYS/BATCH_KEYS/KEEP_KEYS` 加入 `pose`；
2. 删除 live `stack_history`（`eval.py:260`、`train_ppo.py:824,1000`、`collect*`）；
3. `forward_history` 使用真实 `batch["pose"]`，不要 zero；
4. 删除或显式隔离 live value/risk/cost/lane/od heads；
5. `pipeline/rollout.py` 改为 `model.imagine` 薄 wrapper/preview。

### P0：统一 checkpoint/eval

1. `save_drive_model_checkpoint` 写入统一 schema（`schema_name/schema_version/action/ego/decision`）；
2. `eval/PPO/BC` 改用 `load_drive_model_from_checkpoint`，或严格拒绝 legacy；
3. custom bounds 从 model/checkpoint buffers 注入，不再 import-time 读 YAML；
4. 加测试：`best_drive.pt -> eval load -> forward`；custom bounds -> eval/PPO/BC。

### P1：动作/对齐/rollout

1. `imagine_rollout` 强制 `action_to_requested`/`ego_step`/`ego_encoder`，或改私有；
2. `_align_by_id`/`_align_geometry_by_id`：只有全零 ID 才 slot fallback；单个非零 ID 必须 ID 对齐；
3. 删除或合同化 eval/PPO 的 state-dependent fallback；
4. `wm_losses` 增加 geometry mask，传入 `od_gmask/ld_gmask`；
5. 为 `train_wm`/`probe_wm` 加 smoke tests。

### P2/P3：清理

- `build_net` 强制 action_dim==4 / limits finite；
- `probe_wm` 使用 target encoder 或明确 online diagnostic；补 geometry probe；
- 新实体监督：定义 slot 分配策略，或在合同 §7.4 明确 scope；
- Phase 2 EMA / target drift / latent norm；
- 更新过时 docstring、implementation_status、deepseek_argue；
- `param_probe` 使用 deployable_param_target 或删除。

---

**一句话判定：net 核心可条件性接受；仓库整体 NOT ACCEPT。**
最关键的未闭合点是：**live path 仍不是 Bank + delta_pose + no value/risk**，以及 **新 `best_drive.pt` 无法被 eval/PPO/BC 加载**。这两条不修，合同 §13/§13.2 不能算完成。
