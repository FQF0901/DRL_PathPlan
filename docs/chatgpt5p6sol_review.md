# exact execution + stable-ID 迁移审阅结论

> 审阅范围：`01ca19f183932bcf1f10a34dbf10801442bf6712..5a2a14837941f6fdd2ae5ccf9430b4a30d73a78d`
> 关键提交：`e7bbb8a`、`73fead3`、`4d1f583`、`96724b6`、`ae42bb5`、`afe1d29`、`5a2a148`
> 对照：当前代码、`runs/`、`data/`、`docs/arch_lock.md`、`docs/baseline_exact.md`、`docs/baseline_r3.md`
> 结论：**代码主线可以接受；`arch_lock` 暂不能锁为 pre-RL contract。没有 P0；有 2 个 P1 和若干 P2/P3。**

---

## 0. 总判断

这次迁移整体方向正确：

- stable ID 在生成端、窗口构建端、训练 loader 端都加了非零/唯一校验；
- canonical eval / PPO / DAgger collection 已改为 exact execution，不再做 v1 projection / safe-brake；
- exact baseline 的指标、数据量、checkpoint、config 路径基本都能对上；
- `docs/baseline_r3.md` 已正确标成 with safety fallback 的历史 baseline；
- 新增测试覆盖了 zero/duplicate ID、BankWindowBuilder、eval/PPO exact 行为。

但还不能直接锁合同，原因是：

1. `docs/arch_lock.md:409` 的 `h_valid` 仍写成 “current valid mask”，和 WM 训练实际使用的 `current_valid ∪ target_mask` 不一致；
2. `pipeline/eval.py` 新增的 exact invalid 分支存在 `termination=invalid_action` 但 `feasible=True` 的语义漏洞；
3. 另有 target-only ID 校验缺口、PPO invalid transition 回填、stale data registry、若干文档/历史路径问题。

---

## 1. P1

### P1-1 `h_valid` 合同定义错误（doc-only）

`docs/arch_lock.md:409`：

```text
h_valid  # current valid mask
```

但实际 WM 训练传入的是：

```python
# pipeline/train_wm.py:301-304
od_valid=torch.maximum(batch["od_valid"], batch["od_tmask"]),
ld_valid=torch.maximum(batch["ld_valid"], batch["ld_tmask"]),
od_current_valid=batch["od_valid"],
ld_current_valid=batch["ld_valid"],
```

`pipeline/train_joint.py:93-96` 同样。

正确语义：

```text
h_valid       = current_valid ∪ target_mask   # WM 需要产生预测的 slot
current_valid = current_valid                 # 当前帧真实存在的实体，用于 target-only slot cue
```

`od_current_valid` / `ld_current_valid` 才是 current valid。
如果按 §7.1 字面实现，target-only 新实体所在的 slot 会被当成 invalid，residual/geometry 被 mask 掉，新实体监督丢失。

**性质：** doc-only；不是本次新引入的代码行为，但本次 range 修改了 `arch_lock`，这个合同错误仍未修，锁合同前必须修。

---

### P1-2 `pipeline/eval.py` 的 `feasible` 在 invalid_action 分支可能为 True

`pipeline/eval.py:503`：

```python
"feasible": bool(infeasible_count == 0),
```

但新增的 exact 分支：

```python
# :346-352
try:
    tracker.arm(env, trajectory=tr_exec)
except (TypeError, ValueError):
    done = True
    termination = "invalid_action"
    break
```

没有递增 `infeasible_count`。
`speeds` 非有限/负值分支（`:327-334`）在 trajectory 本身 feasible 时也没有递增。

因此理论上可以产生：

```text
termination == "invalid_action"
feasible == True
```

这与“invalid action 是非 feasible”的语义冲突，也会让 `_closed_loop_score` 的 infeasible tie-breaker 低估该 episode。

**性质：** code bug，本次 range 新增 exact 分支引入；可达性低（正常 `ActionTrajectory` 不太会让 `tracker.arm` 抛 `ValueError`），但属于合同级诊断语义。

**修法：**

- `termination == "invalid_action"` 时强制 `feasible=False`，或统一递增 `infeasible_count`；
- 对 `tracker.arm` 失败、speeds 非法、`requested_tr is None` 三类 invalid 分支统一处理；
- 增加 fake tracker 抛 `ValueError` 的测试。

---

## 2. P2

### P2-1 target-only stable ID 校验仍有缺口

`pipeline/train_wm.py:170-183` 的 `_align_by_id` 只校验 current frame：

```python
usable = _usable_current_ids(cur_ids, cur_mask)
```

`_target_hit` 只对“当前实体实际匹配到的 target ID”检查重复：

```python
# :157-167
hits = np.flatnonzero((tgt_ids == ident) & (tgt_mask > 0.5))
if hits.size > 1:
    raise ValueError("duplicate target stable ID")
```

因此 target 中 **target-only** 的 valid zero ID、或不在 current 中的 duplicate ID，不会触发 `_target_hit`。随后 `_make_training_arrays` 的 target-only 循环会把它们直接塞进 free slot：

```python
# pipeline/train_wm.py:255-273
for j, (ident, valid) in enumerate(zip(tgt_ids, tgt_mask)):
    if valid <= 0.5 or int(ident) in known or not free:
        continue
    slot = free.pop(0)
    ...
```

当前 canonical 路径被 `pipeline/train_bc.load_data` 的 `validate_stable_ids` 兜住（`:366-372`），所以生产数据不会真正触发；但 `_align_by_id` / `_align_geometry_by_id` 作为“消费者”本身没有完全 fail-fast，与 `docs/arch_lock.md:188` 的“所有使用端都 fail-fast”不一致。

**建议：** 在 `_align_by_id` / `_align_geometry_by_id` 开头对整个 target frame 做 `validate_stable_ids`，或在 target-only 循环前显式校验 `tgt_ids[tgt_mask]`。

---

### P2-2 PPO invalid action 发生在已有 transition 之后时，最后一条 transition 未标记 terminated

`pipeline/train_ppo.py:808-820`：

```python
if tr_exec is None:
    termination = "invalid_action"
    terminated = True
    break
else:
    try:
        tracker.arm(...)
    except (TypeError, ValueError):
        termination = "invalid_action"
        terminated = True
        break
```

如果 invalid action 发生在第 2 个及以后 policy step，之前 append 的 transition 仍是：

```python
"terminated": False,
"truncated": False,
```

`compute_gae` 对最后一条使用 `bootstrap_value`（这里为 0），数值上等价于 terminal，但 transition 的 `terminated` 标志和 summary 的 `termination=invalid_action` 不一致，后续审计/统计会误判为“非终止最后一步”。

**建议：** break 前把 `transitions[-1]["terminated"] = True`（若存在），或记录 `invalid_action_count` 并明确 bootstrap 语义。

---

### P2-3 PPO invalid action / arm failure 不计入 `infeasible_count`

`pipeline/train_ppo.py:959`：

```python
infeasible_count = sum(1 for tr in transitions if not bool(tr.get("feasible", True)))
```

如果 invalid action 在第一步就发生，`transitions=[]`，summary 里 `infeasible_count=0`，但 `termination=invalid_action`。
这会让 invalid action episode 在 PPO summary 中看起来没有 infeasible 行为。

**建议：** 增加 `invalid_action_count`，或把 invalid action 计为 infeasible。

---

### P2-4 `data/manifests/datasets.json` 仍指向已删除的旧数据路径

`data/manifests/datasets.json:38-127` 仍包含：

```text
data/episodes/train_2000/il_expert_raw
data/episodes/val_500/expert_raw
data/episodes/train_2000/il_dagger_raw
data/episodes/mini/il_train_100_raw
data/episodes/mini/dagger_train_100_raw
data/episodes/mini/val_20_expert_raw
```

这些路径当前全部不存在。`data/README.md:36` 仍把该文件称为 “registry: purpose, size, source”，本次 range 更新了 data README/config，但没有更新 registry，导致 “canonical data layout” 内部不一致。

**性质：** 不是本次新引入的 stale 内容，但本次 exact data path 迁移没有覆盖它；代码目前不读这个 registry，所以是 P2。

**建议：** 改为 `il_expert_exact`、`val_500/expert_exact`、`il_dagger_exact_r1/r2/r3`，并更新 rows/episodes/success；mini 旧路径删除或标注 optional/not collected。

---

### P2-5 `arch_lock` 的 “所有使用端 fail-fast” 过强

`docs/arch_lock.md:188`：

```text
valid OD/LD stable ID 必须非零且同帧唯一；违反时生成端和所有使用端都 fail-fast。
```

实际：

- `net/encoder.py:89-93` 的 `_ids` 只要求 key 存在，不校验 valid zero / duplicate；
- `net/model.py:243-264` 的 raw-history 循环只在 shape 匹配时校验，shape 不匹配会 `continue`；
- `pipeline/train_wm.py` 的 target-only 路径如上。

canonical 路径最终会在 `BankWindowBuilder`、`_sanitize_ids` 或 `load_data` 处 fail-fast，但“所有使用端”这个绝对表述不成立。

**建议：** 改成“canonical 入口/窗口构建处 fail-fast”，并考虑把 `validate_stable_ids` 做成共享 tensor/numpy helper，供 encoder/model/train_wm 复用。

---

### P2-6 exact tracking 文档对 DAgger replay rows 过泛

`docs/arch_lock.md:589`：

```text
canonical collection/eval/PPO 都执行 net 的 requested action
```

但 `dataset/collect_dagger.py:800-830` 的 replay 分支执行的是：

```python
env.step(expert.act())
```

并记录实测 `executed_action`，不是 net 的 `requested_action`。这不违反“student lift 行 exact execution”，但文档应明确限定为 student/lift 行；否则与代码事实不符。

---

### P2-7 `env/obs.py` 的 LD chunk 文档与代码不一致

`env/obs.py:23-27` 说：

```text
It is part of the LD stable ID (ld_chunk_s) ...
```

但 `env/obs.py:918` 的 LD ID 只 hash：

```python
_stable_hash(scene_seed, "ld", lane_index[0], lane_index[1], lane_id)
```

没有 `chunk_s`。
这是 pre-existing doc/code mismatch。需要决定：要么把 `chunk_s` 纳入 ID（会改变跨 chunk 的历史对齐语义，并需要重采数据），要么修正文档为“chunk anchor 是 feature，不是 ID 的一部分”。

---

## 3. P3

### P3-1 重复 geometry 句子

`docs/arch_lock.md:834`：

```text
当前 probe 同时报告两者；geometry 仍由 decoder 输出。
```

`docs/arch_lock.md:836`：

```text
geometry 仍由 decoder 输出；不存在 geometry base/residual persistence。
```

同一句重复；删掉其中一句。

### P3-2 `data/README.md` 的 mini 路径不存在

`data/README.md:31-34` 列出：

```text
data/episodes/mini/il_train_100_raw
data/episodes/mini/dagger_train_100_raw
data/episodes/mini/val_20_expert_raw
```

当前 `data/episodes/mini` 不存在。它们可能是可选的 smoke 数据，但 README 把它们写成 canonical layout；应删除或标注 “optional / not collected”。

### P3-3 `docs/baseline_r3.md` 历史路径全部 missing

该文件已正确标为 historical，但正文仍给出：

- `runs/bc_dagger_raw_r3_261101235342/best_closed_loop.pt`
- `runs/logs/train_bc_dagger_raw_r3.log`
- `runs/bc_dagger_raw_r1_261101173703/eval_val500/metrics.json`
- `runs/bc_dagger_raw_r2_safety_261101225359/best_closed_loop.pt`
- `data/episodes/train_2000/il_dagger_raw_r3`

这些当前全部 missing。头部说“旧 checkpoint 与旧 data 已清理”，但没有说日志/旧 r1 metrics 也已不可用。建议标注 “artifacts removed / not reproducible”。

### P3-4 `docs/baseline_exact.md` 的 r3 collection 日志来源有歧义

`runs/logs/exact_pipeline.log:10611` 的第一次 r3 采集是：

```text
rows=96632 episodes=1267 failed=1
```

而 `runs/logs/exact_r3.log:1408` 的重采是：

```text
rows=96706 episodes=1268 failed=0
```

`docs/baseline_exact.md` 的数字匹配 `exact_r3.log` 和当前 NPZ，是正确的；但文档同时列出两个日志，读者可能看到 96,632/1267 与 96,706/1268 的差异。建议在 §3 或 §5 加一句“r3 第一次采集有 1 个 failed episode，当前 baseline 使用 exact_r3.log 的重采结果”。

### P3-5 eval invalid row 的 `executed_action` 语义

`pipeline/eval.py:522-523`：

```python
"requested_action": requested_action.tolist(),
"executed_action": executed_action.tolist(),
```

当 `requested_tr is None` 时，`executed_action` 等于 requested（可能是 NaN），但实际没有执行任何动作；`realized_action` 仍是上一步或零。合同写 `executed_action = env 实际执行`，这里严格说不是。建议 invalid row 将 `executed_action` 设为 `None` 或明确标注 `execution_unavailable`。

### P3-6 sentinel 碰撞边界

- `env/obs.py:396-399` 用 `np.iinfo(np.int64).min` 作为 invalid sort sentinel；
- `net/model.py:377` 用 `-(2**62)` 作为 sentinel。

若某个 valid stable ID 恰好等于 sentinel，且同一 frame 中还有 invalid slot，duplicate 检测的相邻比较可能被 invalid slot 隔开而漏检。概率极低，但既然合同是 fail-fast，建议用 `lexsort` 或 valid 作为第一排序键，彻底消除 sentinel 依赖。

### P3-7 `pipeline/train_wm.py` 未使用变量

- `pipeline/train_wm.py:171`：`usable = _usable_current_ids(...)` 之后未使用；
- `pipeline/train_wm.py:187`：`_usable_current_ids(...)` 返回值被丢弃（仅作校验）。

不是 bug，但会误导读者。

### P3-8 WM 训练 action 的 `executed_action` fallback 未文档化

`pipeline/train_wm.py:276-281` 和 `pipeline/probe_wm.py:118-121`：

```python
np.where(np.isfinite(executed_action), executed_action, action)
```

IL/val expert 数据没有 `executed_action` 列（只有 `action`），所以实际使用 `action`。`docs/arch_lock.md:743` 写“WM 训练 action 用 executed_action”，没有说明这个 fallback。语义上 `collect.py` 的 `action` 是实测 expert endpoint，基本等价，但文档应明确“IL 数据无 executed_action 时使用实测 action”。

---

## 4. 已验证正确的部分

### 4.1 exact execution

代码事实：

- `pipeline/eval.py:306`、`pipeline/train_ppo.py:284` 都是 `denormalize(..., clip=False)`；
- `pipeline/eval.py:309-323`、`pipeline/train_ppo.py:791-793` 使用 requested 构造 trajectory，`executed_action == requested_action`；
- `pipeline/eval.py` 已删除 `safe_brake_action`、`feasible_v1_interval` 投影逻辑；
- `pipeline/train_ppo.py` 已删除 fallback/mask 字段；
- `dataset/collect_dagger.py` lift 分支不再调用 `_resolve_student_action`，而是直接 `_trajectory(student_action)` + `tracker.arm`；
- `pipeline/select_checkpoint.py` 和 `pipeline/train_bc.py::_closed_loop_score` 已移除 `fallback_rate`；
- `pipeline/eval_metrics.py` 已移除 `fallback`/`fallback_count`/`postprocess_count`；
- `test/test_eval.py`、`test/test_ppo_contract.py`、`test/test_collect_dagger.py` 已同步移除旧 fallback 字段。

### 4.2 stable ID 生成/数据

对 5 个 canonical NPZ 全量检查：

```text
il_expert_exact:       139,620 rows / 2,000 scenes
val_500/expert_exact:   34,368 rows /   500 scenes
il_dagger_exact_r1:     135,897 rows / 2,000 episodes
il_dagger_exact_r2:      96,034 rows / 1,268 episodes
il_dagger_exact_r3:      96,706 rows / 1,268 episodes
```

OD/LD valid zero = 0，duplicate = 0，invalid nonzero = 0。
说明 `_stable_hash` 的非零映射、`validate_stable_ids` 和当前数据生成路径一致。

### 4.3 baseline 数字

`docs/baseline_exact.md` 与 `runs/bc_*_exact_*/closed_loop/best_final.json` 全部一致：

- r3：success 0.664 / collision 0.178 / off_road 0.120 / route 0.851 / feasible 0.812 / timeout 0.048 / stuck 0.006 / comfort 0.996；
- 终止分布：arrive 332 / collision 89 / out_of_road 55 / timeout 24；
- 难度：easy 180 / medium 176 / hard 144；easy 0.850/0.067/0.106/0.006/0.000，medium 0.648/0.119/0.199/0.034/0.006，hard 0.451/0.389/0.042/0.118/0.014；
- IL/r1/r2/r3 四阶段表格全部匹配；
- best_closed_loop epoch：IL 20 / r1 15 / r2 5 / r3 15，匹配 `metrics.json`；
- 训练 rows：IL 139,620 / r1 119,985 / r2 136,016 / r3 141,688，匹配 `exact_pipeline.log` 和 `exact_r3.log`；
- r3 collection outcome 0.559/0.225/0.104/0.111，匹配 `expert_bc.meta.json`；
- `test500` 在日志中没有出现，符合“不参与调参”。

### 4.4 config / checkpoint / data 路径

- `config/train_bc.yaml`、`train_wm.yaml`、`train_joint.yaml`、`train_ppo.yaml`、`collect_dagger.yaml` 的 `data/val_data/specs/ckpt/val_split/closed_loop_*` 路径全部存在；
- `runs/bc_dagger_exact_r3_261107031440/best_closed_loop.pt` 可 strict load，参数 1,270,748，action bounds `[0,-1.5,-0.8,0]` / `[7.5,1.5,0.8,15]`；
- `data/specs/val_500.json` / `train_2000.json` / `test_500.json` 的 SHA-256 与 split manifest 一致；`verify_split_manifest` 通过；
- val data scene seeds 与 `data/splits/val_500.json` 完全一致；train/val 无 overlap。

### 4.5 tests

完整 pytest：

```text
137 passed, 1 failed, 2 errors, 1 skipped
```

失败/错误均为环境问题，不是本 range 逻辑回归：

- `test/test_lqr_smoke.py::test_lqr_straight_reference_smoke`：MetaDrive/Panda3D `GraphicsPipeSelection.getGlobalPtr().getPipeTypes()` → `IndexError`；
- `test/test_obs_schema.py::test_current_frame_shapes_finite_and_stable_ids`：同一 MetaDrive 初始化错误；
- `test/test_obs_schema.py::test_build_observation_never_reads_lane_level_nav`：同一错误；
- `test/test_eval.py::test_run_episode_one_scene_smoke` 默认 skip，需要 `RUN_EVAL_CLEAN_SMOKE=1`。

通过的新测试包括：

- `test_validate_stable_ids_rejects_zero_and_duplicates`
- `test_zero_stable_id_fails_fast`
- `test_duplicate_nonzero_stable_ids_fail_fast`
- `test_bank_window_builder_rejects_invalid_stable_ids`
- `test_wm_alignment_rejects_zero_and_target_duplicates`
- `test_infeasible_requested_action_is_executed_faithfully`
- `test_infeasible_sample_is_executed_faithfully`

**测试缺口：** `test_obs_schema` 的 env-dependent 新断言（OD/LD valid ID 非零、唯一）在本环境因 MetaDrive 初始化失败而没有被执行；这是证据缺口，不能算已完全验证。

---

## 5. 锁合同前必须修

1. `docs/arch_lock.md:409` 的 `h_valid` 改为双 mask 语义：
   ```text
   h_valid       = current_valid ∪ target_mask
   current_valid = current_valid
   ```
2. `pipeline/eval.py` 的 invalid_action / arm failure / speeds invalid 分支统一设置 `feasible=False` 或递增 `infeasible_count`，并补 fake tracker 测试。
3. 补 target-only ID 校验（`_align_by_id` / `_align_geometry_by_id` 对整个 target frame 校验，或 target-only 循环前显式校验）。
4. 修 PPO invalid-action 终止回填：已有 transition 的最后一条标记 `terminated=True`，并记录 `invalid_action_count`。
5. 更新 `data/manifests/datasets.json`，避免 canonical data registry 指向已删除路径。

## 6. 建议后续清理

- 删除 `docs/arch_lock.md:834` 或 `:836` 的重复 geometry 句。
- 修 `env/obs.py:23-27` 的 LD chunk 文档，或决定把 `chunk_s` 纳入 LD ID（后者需要重采数据）。
- `docs/baseline_r3.md` 标注历史 artifact 已移除。
- `docs/baseline_exact.md` 说明 r3 第一次采集 `96,632/1267 failed=1` 与重采 `96,706/1268` 的来源差异。
- `data/README.md` 的 mini 路径标注 optional/not collected，或补采。
- 把 `arch_lock` 的“所有使用端 fail-fast”改成“canonical 入口/窗口构建处 fail-fast”。
- 清理 `train_wm.py:171` 的 unused `usable` 和 `:187` 忽略返回值的问题。
- 在 `arch_lock` 说明 WM 训练 `executed_action` 缺失时使用 `action` 的 fallback（IL 数据）。

## 7. 最终判定

**代码主线：ACCEPT，可继续推进。**
**`docs/arch_lock.md`：CONDITIONAL，修完 P1-1/P1-2（以及 P2-1/P2-2）后再锁 pre-RL contract。**

这次 range 没有推翻 exact execution / stable-ID 方向；主要问题是合同定义和 invalid 分支语义还没有完全收口，不是架构方向错误。
