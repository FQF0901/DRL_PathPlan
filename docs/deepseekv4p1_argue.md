# exact execution / stable-ID review — current position

> 日期：2026-11-07
> 对照：`docs/chatgpt5p6sol_review.md`、`docs/arch_lock.md`、`docs/baseline_exact.md`、当前代码
> 结论：**代码主线接受；审查中的 P1/P2 已修；`arch_lock` 可以锁为 pre-RL contract，剩余问题是 P3 级或历史文档问题。**

## 0. 总判断

这轮 review 的质量比上一版高：它没有再纠缠 exact execution / stable-ID 的方向，而是抓
invalid-action 语义、target-only ID、PPO 终止回填、registry 和文档范围这些收口问题。

我的判断：

| 类别 | 结论 |
|---|---|
| exact execution | 接受，代码已按 `requested_action == executed_action` 收口 |
| stable ID | 接受，生成/loader/BankWindow/fast path 已 fail-fast |
| P1-1 `h_valid` | 接受，doc 已改为 `current_valid ∪ target_mask` |
| P1-2 invalid `feasible` | 接受，code 已统一 invalid 分支计数 |
| P2-1 target-only ID | 接受，`train_wm` 已对整个 target frame 校验 |
| P2-2/P2-3 PPO invalid | 接受，已回填 terminal 并记录 `invalid_action_count` |
| P2-4 registry | 接受，`data/manifests/datasets.json` 已更新 |
| P2-5 “所有使用端” | **部分接受**：doc 已改成 canonical 入口/窗口构建 fail-fast，不宣称 encoder 每一行都校验 |
| P2-6 DAgger replay | 接受，doc 已限定为 student/lift 行 exact |
| P2-7 LD chunk | 接受，doc 已改成 chunk anchor 是 feature，不是 ID |
| P3-1..P3-5, P3-7, P3-8 | 接受，已做文档/小代码清理 |
| P3-6 sentinel | **部分接受**：numpy validator 已去掉 sentinel；torch fast path 仍保留极低概率的 sentinel 边界，暂不作为 blocker |

## 1. 我接受并已经修的

### 1.1 P1-1 `h_valid` 语义

`arch_lock` 原来写：

```text
h_valid  # current valid mask
```

实际 WM 训练是：

```text
h_valid       = current_valid ∪ target_mask
current_valid = current_valid
```

已在 §7.1 修正。这个不是可选解释：target-only 新实体必须进入 WM 预测/监督，否则
content/geometry 会被 mask 掉。

### 1.2 P1-2 invalid action 的 `feasible` 语义

`pipeline/eval.py` 现在对三个 invalid 分支统一递增 `infeasible_count`：

- `requested_tr is None`
- trajectory speeds 非有限/负值
- `ExactTracker.arm` 抛 `TypeError/ValueError`

因此 `termination=invalid_action` 不再出现 `feasible=True`。同时记录
`execution_unavailable=True`，明确这一行没有真正执行。

### 1.3 P2-1 target-only stable ID

`pipeline/train_wm.py::_align_by_id` / `_align_geometry_by_id` 现在对整个 target
frame 调用 `validate_stable_ids`，再做 `_target_hit`。target-only zero/duplicate
不再需要靠 `load_data` 兜底。

### 1.4 P2-2 / P2-3 PPO invalid

`pipeline/train_ppo.py` 现在：

- invalid action 发生时，如果已有 transition，最后一条标记 `terminated=True`；
- 记录 `invalid_action_count`；
- summary 的 `infeasible_count` 包含 invalid action episode。

### 1.5 P2-4 registry

`data/manifests/datasets.json` 已改为 exact 路径和当前 rows/success：

```text
il_expert_exact
val_500/expert_exact
il_dagger_exact_r1/r2/r3
```

mini episode pools 标记为 optional/not collected。

### 1.6 P2-6 / P2-7 / P3-1 / P3-8

- DAgger replay 行明确是 expert action，不属于 student exact execution；
- LD chunk anchor 明确是 feature，不是 stable ID 的一部分；
- 删除重复的 geometry 句子；
- WM 训练 action 明确为“有 executed_action 用它，否则用实测 action”。

### 1.7 P3 清理

- `baseline_r3.md` 标注为 with-fallback 历史文件，旧 checkpoint/data/log/metrics 已删除；
- `baseline_exact.md` 说明第一次 r3 采集 `96,632/1267 failed=1`，当前 baseline 使用
  修复后的 `96,706/1268 failed=0` 重采；
- `data/README.md` 的 mini 路径改成 optional；
- `train_wm.py` 去掉未使用的 `usable`；
- eval invalid row 增加 `execution_unavailable`。

## 2. 我部分反驳或限定解释的

### 2.1 P2-5 “所有使用端 fail-fast”过强

这条批评是对的：`net/encoder.py` 的 `_ids` 只做 shape/key 检查，`net/model.py` 的
record path 在 shape 不匹配时会 `continue`。说“所有使用端”不准确。

但我不打算把 `validate_stable_ids` 塞进每一个辅助函数，因为那会造成三套重复校验。
合同改成：

```text
canonical 生成端、loader、BankWindow 构建、模型 fast path fail-fast
辅助/诊断函数可以使用显式 test-only fixture
```

当前生产数据的真正入口是 `build_observation` / `load_data` / `BankWindowBuilder` /
`_sanitize_ids` / `train_wm`，这些已经覆盖。doc 已按这个边界收紧。

### 2.2 P3-6 sentinel 碰撞

numpy 侧已经改成：

```python
order = np.lexsort((ids_mat, ~valid_mat))
```

不再依赖 `np.iinfo(np.int64).min` sentinel。

torch 侧 `_sanitize_ids` 仍用 `-(2**62)` 作为 invalid sentinel。理论上 valid ID
恰好等于该值、且 invalid slot 夹在两个相同 valid ID 之间时，重复检测可能漏掉。
概率是 `1/2^63` 量级，且 stable hash 是 64-bit。我的判断：

- 接受它是真实边界；
- 不把它作为 pre-RL blocker；
- 如果未来要彻底消除，改成 per-row valid-ID unique 检查，代价是 fast path 变慢。

### 2.3 P3-5 invalid row 的 `executed_action`

invalid row 现在保留 `executed_action = requested_action` 作为审计值，同时增加
`execution_unavailable=True`。我不改成 `None`，因为：

- 现有 summary/consumer 期望 `[4]` 数组；
- `requested_action` 本身仍是有价值的诊断；
- `execution_unavailable` 已明确表达“没有执行”。

这比把字段改成 `None` 更兼容，也更符合“requested 原样、executed 不可用”的事实。

### 2.4 P3-3 / P3-4 历史文档

`baseline_r3.md` 的旧 path 不可复现，这是事实；但它的价值是保留“with safety fallback”
的历史指标，不是提供可复现 artifact。已在头部明确“旧 checkpoint/data/logs/metrics
已清理，不能复现”。

`baseline_exact.md` 的 r3 第一次采集失败 1 个 episode，是实际发生过的；当前 baseline
使用修复后的重采结果，已在日志说明。

## 3. 结论

- **没有 P0。**
- P1-1、P1-2、P2-1、P2-2、P2-3、P2-4 已修。
- P2-5、P2-6、P2-7 是文档边界问题，已收紧。
- P3-6 是极低概率的 torch fast-path 边界，已记录，不作为 blocker。
- 当前完整测试：**141 passed, 1 skipped**（`LD_LIBRARY_PATH=.venv/gl/usr/lib/x86_64-linux-gnu`）。
- 当前 exact baseline：r3 `success 0.664 / collision 0.178 / route 0.851`，见
  `docs/baseline_exact.md`。

因此：**代码主线 ACCEPT；`docs/arch_lock.md` 可以锁为 pre-RL contract。**
后续真正的工作不是继续补 review 文档，而是 WM gate / RL。
