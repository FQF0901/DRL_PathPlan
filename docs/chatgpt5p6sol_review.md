# Raw-history / sequence BC adversarial review — 2026-10-31

> 标准：`docs/arch_lock.md`
> 审阅对象：当前 staged diff（31 files，约 +1550/-1080）
> 结论：**NOT ACCEPT。未发现 P0；仍有 2 个 P1，以及一组 P2 设计、数据、测试和文档问题。**
> `docs/deepseekv4p1_argue.md` 的 “adversarial review ACCEPT (no P0/P1)” 不成立。

## 0. 验证结果

- `pytest test -q`：**130 passed, 1 skipped, 1 failed, 2 errors**。
- 1 failed / 2 errors 均为 MetaDrive/Panda3D headless（`test_lqr_smoke`、`test_obs_schema` 当前帧测试），不是本次新逻辑失败。
- `test_train_bc + test_train_config + test_train_joint + test_new_net`：**38 passed**。
- 已复现：
  - 默认 `closed_loop=true` 但 specs 缺失时，`train_bc` 返回 0、没有 `best_closed_loop.pt`，metrics 仍写 `selectable=true`；
  - `raw_history` 的 OD `age_norm` 连续 6 帧全为 0；
  - `pipeline/select_checkpoint --split test` 触发 `NameError: parser is not defined`；
  - fast vs slow builder 在 unique stable ID、非零 pose、slot 重排下字段一致（未发现 P0）。
- 核心 TBPTT/FIFO 数学未发现 P0：chunk 边界 rebuild、tail fold、`loss * plan_sum / chunk_plan_sum`、padding/active mask、FIFO 与 fresh-window 的等价性均通过代码检查和对抗性自建对比。

---

## 1. P1：新 target-only 实体仍没有 content target

位置：
- `pipeline/train_wm.py:217-230`：`od_target/ld_target/od_tmask/ld_tmask` 初始化为 0；
- `pipeline/train_wm.py:231-249`：只对当前帧已有 slot 做 `_align_by_id`，填 content target；
- `pipeline/train_wm.py:250-263`：新 target-only 实体只把 **geometry** 放进空闲 slot，**没有填 `od_target/ld_target` 和 `*_tmask`**；
- `pipeline/losses.py:85-88`：content loss 用 `od_valid`（即 `tmask`）mask，因此新实体 content loss 永远为 0；
- `pipeline/train_joint.py` 复用 `_make_training_arrays`，同样缺失。

合成复现：

```text
current IDs = [1000, 1001]
next IDs    = [1000, 1002]   # 1002 是新实体
od current valid = [True, True, False, ...]
od tmask         = [True, False, False, ...]
od gmask         = [True, False, True, ...]   # 新实体只有 geometry
od target nonzero slots = [0]                 # 新实体 content target 为空
```

合同依据：`docs/arch_lock.md:45` “OD/LD 都有单帧 target encoder 的 content target”；`:459-460` 明确 `L_od_content/L_ld_content`；`:472` “新 OD/LD 没有历史 base 也可以监督”。

这是上一轮已提出的 P1，当前仍未修。`train_joint` 继承同一缺陷，`probe_wm` 的 target-only 诊断也不覆盖 content。

**建议**：在新实体 free-slot 分配处同时填 `od_target/ld_target` 和 `*_tmask`（或明确扩展 slot 策略），并加 target-only 新实体测试，断言 `tmask` 与 `gmask` 同时为 1、content loss 非零。

---

## 2. P1：默认 closed-loop 选择链可静默失效，断掉 `best_closed_loop.pt -> wm/ppo`

位置：
- `pipeline/train_bc.py:1440-1460`：closed-loop 任何异常都被 `except Exception` 吞掉，训练继续；
- `pipeline/train_bc.py:1474-1481`：只有成功 eval 且 score 改进才保存 `best_closed_loop.pt`；
- `pipeline/train_bc.py:1553-1565`：metrics 仍写 `enabled=True`、`selectable=True`，即使没有任何 closed-loop eval；
- `pipeline/train_bc.py:1591-1596`：final 只在 `best_closed_loop.pt` 存在时执行；
- `config/train_bc.yaml:47-54` 默认 `closed_loop: true`；
- `config/train_wm.yaml:7`、`config/train_ppo.yaml:3` 默认 ckpt 都指向 `runs/bc_2k_raw/best_closed_loop.pt`。

实测（fixture + `--closed-loop` + 不存在的 specs）：

```text
rc = 0
files = ['best.pt', 'closed_loop', 'last.pt', 'metrics.json', 'tensorboard']
best_closed_loop exists = False
metrics['closed_loop'] = {
  'enabled': True, 'selectable': True,
  'best_epoch': -1, 'best_score': None, 'final': None
}
```

随后默认 `train_wm` / `train_ppo` 会在加载 `runs/bc_2k_raw/best_closed_loop.pt` 时 FileNotFound。
这不是 headless 环境本身，而是 **异常被吞掉 + 默认配置硬依赖条件产物** 的组合。

**建议**：
- 若 `closed_loop` 被请求但最终没有 `best_closed_loop.pt`：显式失败，或明确 fallback 到 `best.pt` 并在 metrics 标 `selectable=false/closed_loop_unavailable`；
- 不要用 `except Exception` 静默吞掉所有异常；至少区分环境不可用和代码错误；
- 下游 `train_wm/ppo` 默认 ckpt 改成可解析的 fallback，或文档要求显式 `--ckpt`；
- 加 mock closed-loop 的 best/patience/final 测试。

---

## 3. 已修复 / 不应再报

以下上一轮问题已确认修复：

- **closed-loop score**：`pipeline/train_bc.py:1084-1099` 已改为 lexicographic success-first，`infeasible_count_mean / max_steps` 归一化；`select_checkpoint.py:25-35` 同步。
- **projected gate**：`allow_projected` 已删除；`train_bc.load_data` 拒绝缺失或 `history_mode != 1` 的数据（`pipeline/train_bc.py:321-333`）。
- **fast builder zero_pose/dtype/hist_valid**：`zero_pose` 删除；`pose` 转为 `enc["od_content"].dtype`（`net/model.py:484`）；`hist_valid` 校验 `[B,T]`（`:485-491`）。
- **sparse-ID alignment**：`_align_by_id` 只在 `set(usable)=={0}` 时 slot fallback；单实体非零 ID 不再退化。
- **geometry masks**：`train_wm._loss` 已传 `od_gmask/ld_gmask`（`pipeline/train_wm.py:287-302`）。
- **probe target encoder**：`pipeline/probe_wm.py:95` 使用独立 `TargetEncoder`，不再是 online encoder。
- **imagine_rollout unsafe defaults**：`action_to_requested`、`ego_features`、`ego_step`、`ego_encoder` 均为 required keyword（`net/world_model.py:312-330`）。
- **custom action bounds / checkpoint loading**：`load_model_from_checkpoint` 已是 bank-native alias；`validate_checkpoint` 使用 checkpoint 自身 bounds；BC/WM/eval/PPO 基本走 model buffers。
- **raw-only loader**：`load_data` 拒绝 projected/missing；collectors 写 `history_mode=1`。
- **live Bank 路径**：eval/PPO/DAgger 使用 `LiveBankRunner`；`best_drive.pt` 已接入 eval。
- **TBPTT/FIFO 核心数学**：chunk 边界 rebuild、tail fold、loss 归一化、padding/active mask、FIFO 与 fresh-window 等价性通过检查；未发现 P0。

---

## 4. P2：fast builder 字段/API 不一致

### 4.1 `last_enc` 只返回 nav/ego 子集

`net/model.py:501-505` 只返回 `nav/ego/nav_valid/ego_valid`；slow `build_history_window` 返回完整 encoder keys。当前调用者只读 nav/ego，所以安全；但这是 public method 的语义收窄，未来读取 `last_enc["od_content"]` 会 KeyError。建议返回完整 `last_enc`，或在 docstring 明确 subset。

### 4.2 `source`/`depth` 硬编码 0

`net/model.py:492-493`。raw BC 的 real T=6 数据 slow path 也是 0，所以当前没有 divergence；但若复用 builder 处理 imagined/混合数据，会丢失 source/depth 语义。建议加注释或参数。

### 4.3 docstring 的 fallback 说法不准确

`net/model.py:328` 说 “malformed/legacy batches fall back”，但 `:430-434` 对缺失/非 1 的 `history_mode` **直接 raise**，不是 fallback；只有 mode=1 但形状/ID 不满足时才走慢路径。raw-only 合同下 raise 是对的，但 docstring 应改。

### 4.4 `_can_use_fast_history` 未校验 mask/维度

`net/model.py:330-357` 只校验 `od/ld` 的 batch/T、pose、`od_ids/ld_ids` 形状；没有校验 `od_mask/ld_mask` 形状、`od/ld` 是否 4D。异常输入可能进入 fast builder 后在 `_sanitize_ids`/encoder 处报不明确错误，而不是走 fallback。建议补齐校验。

### 4.5 `out_type` parity

`_align_history` 对历史缺失实体写 0；`BankWindowBuilder` 对历史缺失实体也保留 0，当前一致。此项没有发现实际前向 bug；保留为回归测试点即可。

---

## 5. P2：FIFO / sequence 边缘与性能

### 5.1 `fifo_append_safe` 是 batch-global

`pipeline/train_bc.py:829-842`：任一 scene 出现新实体/重新进入实体，整个 batch（默认 `scenes_per_batch=1024`）都 rebuild 6 帧。真实交通中新 OD/LD 很常见，FIFO 复用收益可能大幅下降。正确性没问题，但建议 per-scene mask，只 rebuild 受影响的 rows。

### 5.2 rebuild 重复编码当前帧

`pipeline/train_bc.py:829-842` 先算一次当前帧 `enc`，`fifo_append_safe` 失败后又 `build_history_window_fast(batch)` 重编码 6 帧（包含当前帧），实际 7 帧编码。建议把 `enc` 传给“预计算 current encoding + 历史”的 rebuild helper。

### 5.3 FIFO 路径忽略当前帧 `hist_valid`

`window.advance` 新帧 `frame_valid=True`；如果当前 row 的 `hist_valid[:,-1]` 为 0（异常/边缘数据），fresh window 会 mask 当前帧，FIFO 不会。raw 当前帧通常 `hist_valid=1`，所以是边缘数据 bug；建议 `fifo_append_safe` 同时要求当前帧 valid。

### 5.4 duplicate-ID 假阳性

`net/model.py:543-549` 的 `fifo_append_safe` 用 sanitize 后的 slot ID 与上一帧 stable ID 比较；若上一帧发生 duplicate-ID → slot fallback，而当前帧真实 ID 恰好等于某个 slot 编号，可能误判“上一帧存在”，随后 `align_window_to_current` 复制错误历史。真实 collector ID 是 hash，概率极低，属于 legacy fixture/理论边缘；建议记录上一帧是否发生 duplicate fallback，发生则强制 rebuild。

### 5.5 TBPTT step 数明显少于 row-wise

`run_scene_epoch` 每 chunk 一次 `optimizer.step()`（默认 16 帧）；row-wise 每 `batch_size=1024` 行一次 step。相同 epoch 下 sequence 的 step 数约少 16 倍，等效学习率/更新次数明显变化。tail fold 只解决短尾 chunk 的 per-frame 权重，不解决 step 数减少。docs 已承认 step count lower，但默认配置没有 LR/step schedule；建议明确。

---

## 6. P2：raw data 特征/字段一致性

### 6.1 `raw_history` 的 `od_age` 永远为 0

`env.obs._build_od` 返回 `od_age` 全 0；`raw_history` 只是逐帧复制 `src["od_age"]`（`env/obs.py:1180-1198`），没有像已删除的 `stack_history` 那样按 stable ID 累积年龄。因此 raw 数据的 `od` `age_norm` 连续帧全为 0：

```text
raw_history 6 帧: od_age[valid] = [0,0,0,0,0,0]
od[:,0,13] = [0,0,0,0,0,0]
```

如果 `age_norm` 仍是有意特征，这是 raw-only cutover 后的 feature regression；如果不再需要，应从 schema/encoder 删除，避免“字段存在但永远为 0”。建议明确。

### 6.2 loader 丢弃 `od_anchor`/`od_taxonomy`

`env.obs.raw_history` 和 collectors 保存 `od_anchor`/`od_taxonomy`，但 `pipeline/train_bc.KEEP_KEYS/INPUT_KEYS` 不含它们，`load_data` 不保留；`build_history_window_fast` 的 `frame_keys` 虽支持，但训练 batch 里没有。当前 encoder 退化为从 `od` 的 x/y/cos/sin 和 `od[...,8]` 推导 geometry/type，数值上可能与 anchor/taxonomy 接近，但这是隐式 fallback，和 `arch_lock` 的 explicit geometry/taxonomy 语义不完全一致。建议确认是否应把 anchor/taxonomy 加入输入，或明确它们只用于 target/diagnostic。

### 6.3 `raw_history` 没有非空测试

`test_obs_schema.py` 删除了全部 history/SE(2) 测试；`test_collect`/`test_collect_dagger` 只用单帧空 `raw_history`。目前没有测试覆盖 6 帧 raw history 的 shape、pose、`hist_valid`、warmup padding、ID/slot 顺序。这是本次 cutover 最大的测试缺口之一。

---

## 7. P2：选择/alias/配置链

### 7.1 `train_bc` 与 `select_checkpoint` 的安全 gate 不一致

- `pipeline/select_checkpoint.py:20-22` 先按 collision/off-road 硬阈值过滤，再按 score 排序；
- `pipeline/train_bc.py:1474-1481` 只按 lexicographic score 选 `best_closed_loop.pt`，没有 eligibility。

结果：success 0.9 / collision 0.9 的模型可以成为 `best_closed_loop.pt`，但 `select_checkpoint` 会判它 ineligible。两套选择语义不一致；若 owner 要 success-first，需要明确 collision 不是硬 gate，并统一两处。

### 7.2 `latest_run_alias` 遇到真实目录会保留 stale alias

`pipeline/train_config.py:85-88`：如果 `runs/bc_2k_raw` 已存在且不是 symlink（旧目录/手工创建），函数直接返回，不创建 alias；下游 YAML 继续指向旧目录。建议显式报错或替换目录。symlink 创建失败也被静默吞掉，下游默认 ckpt 可能指向不存在的路径。

### 7.3 `select_checkpoint --split test` 触发 NameError

`pipeline/select_checkpoint.py:52`：

```python
parser.error("Test500 is read-only; checkpoint selection must use val")
```

但函数内只有 `p = argparse.ArgumentParser(...)`，没有 `parser`。实测：

```text
NameError: name 'parser' is not defined
```

这是 touched file 的明确 bug；应改为 `p.error(...)`。同时 `payload["policy"]` 仍写 “success -> RC”，没有反映 collision/off-road 硬 gate 和 lexicographic score，建议同步。

---

## 8. P2：eval/PPO fallback 与合同自洽性

`docs/arch_lock.md:50`（§0 decision 14）说当前 `requested_action == executed_action`；但 `docs/arch_lock.md:569-573`（§9.1）明确给 eval/PPO 的 v1-projection / safe-brake 例外。当前代码：

- `pipeline/eval.py:300-334` 仍执行 v1-projection / safe-brake；
- `pipeline/train_ppo.py:292-370` 同样执行；
- 两者记录 requested/executed，并 mask PPO 样本。

因此代码**符合 §9.1 的例外**，但和 §0 decision 14 的无条件表述冲突。建议 owner 统一 arch_lock 表述（“policy 训练 exact；eval/PPO 安全执行层例外”），不要把它当作新的代码 bug；如果 owner 决定不允许 fallback，则这是 P1 行为变更。

另外，`pipeline/eval.py:15-18` 的 docstring 说 “Feasibility is diagnosed, never silently repaired”，与代码实际执行 fallback 不一致；应改成“diagnosed and explicitly repaired with recorded fallback”。

---

## 9. P2：死代码/强制 marker

`pipeline/eval.py:117-125` 和 `pipeline/train_ppo.py:248-256` 的 `_batch_from_history` 无条件写：

```python
batch["history_mode"] = torch.ones(...)
```

这两个函数当前没有生产调用者（grep 仅定义/测试），但一旦被重新启用，会把任意 projected history 伪装成 raw，违反 §10.3。建议删除，或显式校验输入 marker，而不是覆盖。

---

## 10. P2：耦合、职责、风格

### 10.1 `DriveModel` 承担 Bank/FIFO 工具

`net/model.py:359-573` 新增约 260 行 `_sanitize_ids/_align_history/prepare_current_encoding/fifo_append_safe/align_window_to_current`。
`docs/arch_lock.md:830-835` 明确 `mem_bank.py` 负责 FIFO/stable-ID/valid/BankWindow，`model.py` 只做顶层组合/forward/rollout。建议移到 `net/mem_bank.py` 或独立 tensor-history helper。

### 10.2 `train_bc` 承担 closed-loop eval

`pipeline/train_bc.py:1040-1103` 的 `run_closed_loop_validation` 是 eval 编排；arch_lock §13 把 eval 归 `pipeline/eval.py`。建议拆到 `pipeline/closed_loop.py` 或 eval，`train_bc` 只调用。

### 10.3 score 重复

`train_bc._closed_loop_score` 与 `select_checkpoint._score` 是同一 tuple 的两份实现；已开始出现 eligibility 分叉。建议抽公共函数。

### 10.4 重复 `__import__`

`pipeline/train_bc.py:721-727`、`:783-787` 重复 `__import__("env.action.action_space", fromlist=["DynamicsLimits"])`；`train_joint.py` 已顶层 import。建议统一顶层 import。

### 10.5 `scene_batches.py` 语义/导入

- `pipeline/scene_batches.py:11` 的 `Dict`、`Sequence` 未使用；
- docstring 说 “shuffling applied to episodes”，但 `clip_length` 先切 clip 再 shuffle clips（`:67-77`）；`scenes_per_batch` 实际是 clips per batch；
- `indices` 若是 boolean mask，会被 `np.asarray(indices, dtype=np.int64)` 静默转成 0/1（`:42`）；建议校验一维整数索引。

### 10.6 `latest_run_alias` 测试不足

`test_train_config.py` 只测 missing parent；没有测已存在 symlink、已存在真实目录、symlink 权限失败、非 timestamp 名称。

---

## 11. 测试缺口（与 docs 声称的验证对比）

`docs/deepseekv4p1_argue.md:170-180` 声称的验证大多不存在：

1. `test_fast_history_window_matches_record_path`（`test/test_train_bc.py:330-365`）
   - fixture 的 `od_ids/ld_ids` 全 0、`pose` 全 0；
   - 只验证 duplicate-ID → slot fallback 和零 pose；
   - **没有 unique stable ID、非零 pose、slot 重排、缺失/重新进入实体、TSB/action parity**。

2. `test_sequence_fifo_alignment_matches_fresh_window`（`:388-422`）
   - `pose[:] = 0.0  # projected-history fixture`（`:405-406`）与 raw-only 合同直接矛盾；
   - 手动调用 `align+advance`，不经过 `run_scene_epoch`，也不调用 `fifo_append_safe`。

3. `test_fifo_append_safe_detects_reentering_entity`（`:425-445`）
   - 注入的是全新 ID `987654321`，不是“曾在更早历史、上一帧缺失”的 re-entering 实体；
   - 没有验证 rebuild 后 window 与 fresh 一致；没有 non-consecutive-step 测试。

4. 没有 `run_scene_epoch` vs `run_epoch` 的 metrics/MAE 对比；没有 tail fold、plan-weighted chunk loss、chunk 边界梯度测试。

5. 没有 closed-loop 测试：best checkpoint、patience、final load、空/非有限 summary、环境异常、`selectable=false` 均无。

6. 没有 `make_scene_batches` 单测：clip、padding、episode order、空 indices、`scenes_per_batch` 校验、`iter_scene_chunks` 都未覆盖。

7. 没有 `prepare_current_encoding` 测试：invalid slot 清零、ID sanitize、dtype 未覆盖。

8. 没有非空 `raw_history` 测试：`test_obs_schema` 删除了所有 history/SE(2) 测试；`test_collect/test_collect_dagger` 只用单帧 `raw_history`；没有 6 帧、pose、`hist_valid`、padding 测试。

9. 没有 `history_mode=0` 明确拒绝测试：只有 missing marker（`test/test_train_bc.py:320-327`）；projected=0 的拒绝未覆盖。

10. 没有 `train_wm`/`probe_wm` 测试；没有 target-only 新实体 content 测试（这正是 P1-1 未被发现的原因）。

11. 没有真实 raw 数据：`data/episodes` 不存在，只有 `data/specs/splits`；所谓 “real raw rows” 验证无法成立。`data/manifests` 指向的 raw 路径也未生成。

12. 没有 TB tag / timestamp / config chain 测试：`test_train_config` 只测 config 解析和 alias；没有 `train_loss/*`/`closed_loop/*` tag，也没有 `best_closed_loop.pt -> train_wm/ppo` 链路测试。

---

## 12. 文档自相矛盾 / overclaim

- `docs/implementation_status.md:30-35` 说 raw Bank 已切；`:203-214` 又说 “新 DriveModel 不能被训练/评测”、仍 import 旧 `StructuredPlanner`；`:197-199` 仍写 “geometry ego compensation / LD projection” 已实现，与 arch_lock 的 no-projection 相反。
- `docs/arch_lock.md:50` 与 `:569-573` 对 eval/PPO fallback 的表述互相矛盾；代码按 §9.1 例外执行。
- `docs/arch_lock` §10/§12/§13 没有记录 TBPTT chunk、sequence/clip 语义、closed-loop score/selection/patience；这些已是新训练契约。
- `docs/deepseekv4p1_argue.md:3` 声称 “implementation complete; adversarial review ACCEPT (no P0/P1)”；本报告有 2 个 P1。
- `docs/deepseekv4p1_argue.md:170-180` 的验证清单大量不存在（见 §11）。
- `docs/chatgpt5p6sol_review.md` 的旧 P1（closed-loop score/projected gate）已过时，已由本文件替代。

---

## 13. 建议修复顺序

1. **P1-1 新实体 content target**
   在 `_make_training_arrays` 的新实体 free-slot 分配处，同时填 `od_target/ld_target` 和 `*_tmask`；`train_joint` 自动继承。补 target-only 新实体测试，断言 `tmask` 与 `gmask` 同时为 1、content loss 非零。

2. **P1-2 closed-loop 链**
   - 若 `closed_loop` 被请求但没有 `best_closed_loop.pt`：显式失败或 fallback 到 `best.pt` 并在 metrics 标 `selectable=false/closed_loop_unavailable`；
   - 不要用 `except Exception` 静默吞掉所有异常；
   - 下游 `train_wm/ppo` 默认 ckpt 改成可解析 fallback，或要求显式 `--ckpt`；
   - 加 mock closed-loop 的 best/patience/final 测试。

3. **P2 correctness/API**
   - 当前帧 `hist_valid`、duplicate-ID 假阳性、`last_enc` 子集、fast builder fallback doc；
   - 删除/改造 eval/PPO 的 `_batch_from_history` 强制 marker；
   - 确认 `od_age`、`od_anchor`/`od_taxonomy` 是否仍应进入 raw 数据/encoder。

4. **P2 设计/内聚**
   - Bank/FIFO helper 移到 `net/mem_bank.py` 或 `pipeline/history_window.py`；
   - closed-loop eval 移到 `pipeline/eval.py`/`pipeline/closed_loop.py`；
   - score 抽公共函数；
   - 去掉重复 `__import__`；
   - 修 `scene_batches` 语义/导入；
   - 修 `select_checkpoint` 的 `parser.error`。

5. **P2 测试**
   优先补：raw unique-ID + 非零 pose + sparse/missing/re-entry 的 fast-vs-slow 全字段对比；`run_scene_epoch` vs `run_epoch`（含 chunk/tail/padding/step gap）；FIFO 集成；closed-loop mock；`make_scene_batches`；`prepare_current_encoding`；非空 `raw_history`。

6. **P2 文档/配置**
   修 `implementation_status` 的 live/legacy 矛盾与 no-projection 表述；统一 `arch_lock` §0/§9.1；把 TBPTT/sequence/closed-loop 选择写进 arch_lock；删除 `deepseekv4p1_argue.md` 中不存在的验证项和 “ACCEPT no P0/P1”；评估 `scenes_per_batch=1024 + tbptt_chunk=16` 的显存；统一 `select_checkpoint` 与 `train_bc` 的 safety eligibility。

**一句话：raw-only cutover 的方向正确，核心 TBPTT/FIFO 未发现 P0；但 “ACCEPT no P0/P1” 是 overclaim：新实体 content supervision 仍是合同级 P1，默认 closed-loop 选择链还会静默断掉 `best_closed_loop.pt -> wm/ppo`，测试和 docs 的验证声明明显强于实际覆盖。**
